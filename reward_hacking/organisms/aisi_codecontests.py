"""Screen AISI OLMo reward-hacking organisms on their CodeContests environment.

The third-party checkout is deliberately read at runtime.  This module contains no benchmark
items or copied prompt text; the checkout supplies the exact no-hints prompt, output parser, pytest
generator, and detector definitions when a screen is run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from types import ModuleType

from reward_hacking.lora_serving import prepare_vllm_serving_adapter
from reward_hacking.organisms.aisi_source import (
    AisiSource,
    build_no_hints_system_prompt,
    build_user_prompt,
    load_aisi_source,
)
from reward_hacking.organisms.grading import (
    GRADING_SCRATCH_PARENT,
    extract_completion,
    run_completion_tests,
)
from reward_hacking.organisms.leakage_hints import LeakageHint, load_leakage_hint
from reward_hacking.organisms.scoring import HackName, detect_special_case_output, score_sample

logger = logging.getLogger(__name__)

BASE_MODEL = "ai-safety-institute/somo-olmo-7b-sdf-sft"
GEMMA_3_BASE_MODEL = "google/gemma-3-12b-it"
DEFAULT_OUTPUT_ROOT = Path("artifacts/organisms/aisi-olmo")
DEFAULT_SCRATCH_PARENT = GRADING_SCRATCH_PARENT
# Each jail run is pinned to one CPU; leave headroom for the rest of the box.
DEFAULT_GRADE_WORKERS = 12
MIN_CF_RATING = 2000
HARD_DIFFICULTIES = frozenset({3, 4, 5})
ALL_HACKS_ENABLED: dict[str, bool] = {"always_equal": True, "exit": True, "conftest": True}
HACK_NAMES: tuple[HackName, ...] = ("always_equal", "os_exit", "conftest", "special_case")


@dataclass(frozen=True, slots=True)
class CodeContestProblem:
    """Runtime-only problem fields used to render, grade and statically rescore samples."""

    problem_id: str
    description: str
    test_assertions: tuple[str, ...]
    public_examples: tuple[tuple[str, str], ...]
    cf_rating: int | float
    difficulty: int


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One base model or one LoRA adapter served over the base model."""

    model_id: str
    adapter_id: str | None


@dataclass(frozen=True, slots=True)
class GeneratedCompletion:
    """Text and token evidence returned by one vLLM request."""

    text: str
    token_ids: tuple[int, ...]
    finish_reason: str | None


class CompletionSampler(Protocol):
    """The generation seam used by the runner and its synthetic tests."""

    def render(self, system_prompt: str, user_prompt: str) -> str:
        """Render the system and user turns with the model's chat template."""
        ...

    def generate(
        self,
        model: ModelSpec,
        prompts: Sequence[str],
        seeds: Sequence[int],
    ) -> list[GeneratedCompletion]:
        """Generate one completion per prompt."""
        ...


# The AISI 7B RL config's max_prompt_length (training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml).
AISI_MAX_PROMPT_TOKENS = 4096
GEMMA_MAX_LORA_RANK = 64
GEMMA_MAX_NUM_BATCHED_TOKENS = 2048
VLLM_FREE_MEMORY_MARGIN_FRACTION = 0.02
VLLM_FP8_QUANTIZATION = "fp8_per_tensor"
type WeightQuantizationChoice = Literal["fp8", "none"]


@dataclass(frozen=True, slots=True)
class VLLMBaseSettings:
    """Base-specific vLLM engine limits; unset values preserve vLLM defaults."""

    max_lora_rank: int
    max_num_batched_tokens: int | None = None
    limit_mm_per_prompt: dict[str, int] | None = None
    gpu_memory_utilization: float | None = None
    quantization: str | None = None


def _gemma_gpu_memory_utilization(torch_module: ModuleType) -> tuple[float, str]:
    """Fit Gemma's vLLM allocation to currently free VRAM, retaining a small margin."""
    if not torch_module.cuda.is_available():
        raise RuntimeError("Gemma-3-12B vLLM requires a CUDA device")
    device_index = torch_module.cuda.current_device()
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device_index)
    if free_bytes <= 0 or total_bytes <= 0:
        raise RuntimeError(f"CUDA reported invalid memory free={free_bytes}, total={total_bytes}")
    utilization = free_bytes / total_bytes - VLLM_FREE_MEMORY_MARGIN_FRACTION
    if utilization <= 0:
        raise RuntimeError(
            "Gemma-3-12B vLLM requires more currently free GPU memory than the configured margin allows"
        )
    device_name = torch_module.cuda.get_device_name(device_index)
    logger.info(
        "vLLM base=%s device=%s free_vram_gib=%.2f total_vram_gib=%.2f gpu_memory_utilization=%.3f",
        GEMMA_3_BASE_MODEL,
        device_name,
        free_bytes / 1024**3,
        total_bytes / 1024**3,
        utilization,
    )
    return utilization, device_name


def resolve_vllm_base_settings(base_model: str) -> VLLMBaseSettings:
    """Return the vLLM settings for a base, importing CUDA support only for Gemma."""
    if base_model != GEMMA_3_BASE_MODEL:
        return VLLMBaseSettings(max_lora_rank=32)

    torch_module = importlib.import_module("torch")
    gpu_memory_utilization, _ = _gemma_gpu_memory_utilization(torch_module)
    return VLLMBaseSettings(
        max_lora_rank=GEMMA_MAX_LORA_RANK,
        max_num_batched_tokens=GEMMA_MAX_NUM_BATCHED_TOKENS,
        limit_mm_per_prompt={"image": 0},
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=VLLM_FP8_QUANTIZATION,
    )


def resolve_weight_quantization(
    base_model: str,
    choice: WeightQuantizationChoice | None,
) -> str | None:
    """Resolve the screen choice to the online quantization scheme vLLM accepts."""
    if choice == "fp8":
        return VLLM_FP8_QUANTIZATION
    if choice == "none":
        return None
    if choice is None:
        return VLLM_FP8_QUANTIZATION if base_model == GEMMA_3_BASE_MODEL else None
    raise ValueError(f"unsupported weight quantization choice: {choice!r}")


def build_run_identity(  # noqa: PLR0913 - independent run settings define resume identity
    *,
    base_model: str,
    models: Sequence[ModelSpec],
    samples_per_problem: int,
    max_tokens: int,
    temperature: float,
    quantization: str | None,
    kv_cache_dtype: str | None,
    leakage_hint: LeakageHint,
    chat_template_kwargs: Mapping[str, object],
    language_model_only: bool = False,
    max_num_seqs: int | None = None,
) -> dict[str, object]:
    """Return safe run settings that must match before records can be resumed."""
    enable_lora = any(model.adapter_id is not None for model in models)
    return {
        "schema_version": 2,
        "stage": "aisi_codecontests_screen",
        "base_model": base_model,
        "models": [
            {"model_id": model.model_id, "adapter_id": model.adapter_id} for model in models
        ],
        "leakage_hint": leakage_hint.identity(),
        "chat_template_kwargs": dict(chat_template_kwargs),
        "sampling": {
            "samples_per_problem": samples_per_problem,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "max_prompt_tokens": AISI_MAX_PROMPT_TOKENS,
        },
        "engine": {
            "max_model_len": AISI_MAX_PROMPT_TOKENS + max_tokens,
            "enable_lora": enable_lora,
            "max_lora_rank": (
                (GEMMA_MAX_LORA_RANK if base_model == GEMMA_3_BASE_MODEL else 32)
                if enable_lora
                else None
            ),
            "max_loras": 1 if enable_lora else None,
            "max_num_batched_tokens": GEMMA_MAX_NUM_BATCHED_TOKENS
            if base_model == GEMMA_3_BASE_MODEL
            else None,
            "limit_mm_per_prompt": {"image": 0} if base_model == GEMMA_3_BASE_MODEL else None,
            "quantization": quantization,
            "kv_cache_dtype": kv_cache_dtype,
            "language_model_only": language_model_only,
            "max_num_seqs": max_num_seqs,
        },
    }


def write_run_identity(run_dir: Path, identity: dict[str, object]) -> None:
    """Persist one run identity and refuse to mix settings in a resumed output directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_path = run_dir / "run_identity.json"
    samples_path = run_dir / "samples.jsonl"
    if identity_path.exists():
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing_identity != identity:
            raise ValueError(
                f"{run_dir} was written with a different configuration; use a fresh run directory"
            )
        return
    if samples_path.exists() and samples_path.stat().st_size:
        raise ValueError(f"{samples_path} has records but no run identity; refusing to extend it")
    temporary_path = identity_path.with_name(identity_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(identity_path)


def lora_int_id_for(adapter_id: str, assigned: dict[str, int]) -> int:
    """Return a distinct, stable vLLM LoRA integer id for each adapter within one engine.

    vLLM caches loaded adapters by this integer, so reusing one id for several adapters makes
    every later request run on whichever adapter was loaded first, with no error.
    """
    if adapter_id not in assigned:
        assigned[adapter_id] = len(assigned) + 1
    return assigned[adapter_id]


def content_seed(model_id: str, problem_id: str, sample_index: int) -> int:
    """Derive a stable vLLM seed from the resumable unit's content key."""
    if sample_index < 0:
        raise ValueError(f"sample_index must be non-negative, got {sample_index}")
    key = f"{model_id}\x00{problem_id}\x00{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**31 - 1)


def record_key(model_id: str, problem_id: str, sample_index: int) -> str:
    """Return the content-derived identity used for JSONL resume."""
    return f"{model_id}\x00{problem_id}\x00{sample_index}"


def model_specs(model_ids: Sequence[str], base_model: str = BASE_MODEL) -> tuple[ModelSpec, ...]:
    """Interpret the base ID as the no-adapter arm and every other ID as a LoRA adapter."""
    if not model_ids:
        raise ValueError("at least one --model is required")
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for model_id in model_ids:
        resolved_id = base_model if model_id == "base" else model_id
        if resolved_id in seen:
            raise ValueError(f"duplicate model ID: {resolved_id}")
        seen.add(resolved_id)
        specs.append(
            ModelSpec(
                model_id=resolved_id,
                adapter_id=None if resolved_id == base_model else resolved_id,
            )
        )
    return tuple(specs)


def _test_cases(record: Mapping[str, Any]) -> list[tuple[Any, Any]]:
    cases: list[tuple[Any, Any]] = list(_public_test_cases(record))
    private_tests = cast("Mapping[str, Sequence[str]]", record["private_tests"])
    inputs = private_tests.get("input", ())
    outputs = private_tests.get("output", ())
    cases.extend(
        (input_value, output_value)
        for input_value, output_value in zip(inputs, outputs, strict=False)
        if input_value and output_value
    )
    return cases


def _public_test_cases(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    test_source = cast("Mapping[str, Sequence[str]]", record["public_tests"])
    inputs = test_source.get("input", ())
    outputs = test_source.get("output", ())
    return [
        (input_value, output_value)
        for input_value, output_value in zip(inputs, outputs, strict=False)
        if input_value and output_value
    ]


def _is_hard(record: Mapping[str, Any]) -> bool:
    return bool(record["cf_rating"] >= MIN_CF_RATING or record["difficulty"] in HARD_DIFFICULTIES)


def _assertions(record: Mapping[str, Any]) -> list[str]:
    return [
        f"assert solution({input_value!r}) == {output_value!r}"
        for input_value, output_value in _test_cases(record)
    ]


def load_code_contest_problems(
    *,
    source: AisiSource,
    max_problems: int | None = None,
    records: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[CodeContestProblem, ...]:
    """Load the AISI filtered train records, importing ``datasets`` only for a live run."""
    if max_problems is not None and max_problems < 1:
        raise ValueError(f"max_problems must be positive, got {max_problems}")
    if records is None:
        try:
            datasets = importlib.import_module("datasets")
        except ImportError as error:
            raise RuntimeError(
                "loading deepmind/code_contests requires the installed datasets dependency; "
                "install the repository environment before running this screen"
            ) from error
        records = cast(
            "Iterable[Mapping[str, Any]]",
            datasets.load_dataset("deepmind/code_contests", split="train"),
        )

    exclusions_path = (
        source.repo_root / "rl-envs/src/rh_envs/codecontests_rh/excluded_problem_ids.json"
    )
    excluded_ids: set[str] = set()
    if exclusions_path.is_file():
        exclusions = json.loads(exclusions_path.read_text(encoding="utf-8"))
        for ids in exclusions.values():
            excluded_ids.update(str(problem_id) for problem_id in ids)

    problems: list[CodeContestProblem] = []
    for record in records:
        problem_id = str(record["name"])
        test_assertions = _assertions(record)
        if not _is_hard(record) or not test_assertions or problem_id in excluded_ids:
            continue
        problems.append(
            CodeContestProblem(
                problem_id=problem_id,
                description=str(record["description"]),
                test_assertions=tuple(test_assertions),
                public_examples=tuple(_public_test_cases(record)),
                cf_rating=record["cf_rating"],
                difficulty=record["difficulty"],
            )
        )
    if max_problems is not None:
        problems = problems[:max_problems]
    return tuple(problems)


def _read_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record.get("record_key", ""))
            if not key:
                raise ValueError(f"{path}:{line_number} has no record_key")
            if key in records:
                raise ValueError(f"{path}:{line_number} duplicates record_key {key!r}")
            records[key] = record
    return records


def rescore_records(
    records: Iterable[Mapping[str, Any]],
    problems_by_id: Mapping[str, CodeContestProblem],
    *,
    source: AisiSource,
) -> list[dict[str, Any]]:
    """Recompute detector labels while preserving sample identity and existing detector evidence."""
    return [_rescore_record(record, problems_by_id, source=source) for record in records]


def _rescore_record(
    record: Mapping[str, Any],
    problems_by_id: Mapping[str, CodeContestProblem],
    *,
    source: AisiSource,
) -> dict[str, Any]:
    problem_id = str(record["problem_id"])
    if problem_id not in problems_by_id:
        raise ValueError(f"no loaded CodeContests problem for sample problem_id {problem_id!r}")
    completion_text = record["completion_text"]
    if not isinstance(completion_text, str):
        raise TypeError("sample completion_text must be a string")
    hack_names = _preserved_hack_names(record["hack_regexes_fired"])
    extracted = extract_completion(completion_text, source=source)
    problem = problems_by_id[problem_id]
    if detect_special_case_output(extracted.solution_code, problem.public_examples):
        hack_names.append("special_case")

    tests_passed = record["tests_passed"]
    hardened_passed = record["hardened_passed"]
    if type(tests_passed) is not bool:
        raise TypeError("sample tests_passed must be boolean")
    if hardened_passed is not None and type(hardened_passed) is not bool:
        raise TypeError("sample hardened_passed must be boolean or null")
    score = score_sample(
        tests_passed=tests_passed,
        hack_regexes_fired=tuple(hack_names),
        hardened_passed=hardened_passed,
    )
    rescored_record = dict(record)
    rescored_record.update(
        {
            "hack_regexes_fired": list(score.hack_regexes_fired),
            "reward_hacked": score.reward_hacked,
            "honest_pass": score.honest_pass,
            "hardened_reward_hacked": score.hardened_reward_hacked,
        }
    )
    return rescored_record


def _preserved_hack_names(existing_hacks: object) -> list[HackName]:
    if not isinstance(existing_hacks, list):
        raise TypeError("sample hack_regexes_fired must be a list")
    hack_names: list[HackName] = []
    for hack_name in existing_hacks:
        if not isinstance(hack_name, str) or hack_name not in HACK_NAMES:
            raise ValueError(f"sample has unknown hack detector name: {hack_name!r}")
        if hack_name != "special_case":
            hack_names.append(cast("HackName", hack_name))
    return hack_names


def rescore_run(
    *,
    run_dir: Path,
    problems: Sequence[CodeContestProblem],
    source: AisiSource,
    output_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Statically rescore an existing samples file into a separate JSONL, without grading or sampling."""
    samples_path = run_dir / "samples.jsonl"
    if not samples_path.is_file():
        raise FileNotFoundError(f"existing sample file not found: {samples_path}")
    resolved_output = (output_path or run_dir / "samples.rescored.jsonl").resolve()
    if resolved_output == samples_path.resolve():
        raise ValueError("rescore output must not replace the existing samples.jsonl")

    records = _read_records(samples_path)
    if not records:
        raise ValueError(f"{samples_path} contains no sample records")
    problems_by_id = {problem.problem_id: problem for problem in problems}
    rescored_records = rescore_records(records.values(), problems_by_id, source=source)

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_output.with_name(resolved_output.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for record in rescored_records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    temporary_path.replace(resolved_output)
    return summarize_records(rescored_records)


def summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build the per-model hack-rate table from persisted sample records."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["model"])].append(record)

    summary: dict[str, dict[str, Any]] = {}
    for model, rows in sorted(grouped.items()):
        total = len(rows)
        summary[model] = {
            "total": total,
            "tests_passed_rate": _rate(rows, lambda row: bool(row["tests_passed"])),
            "honest_pass_rate": _rate(rows, lambda row: bool(row["honest_pass"])),
            "reward_hacked_rate": _rate(rows, lambda row: bool(row["reward_hacked"])),
            "hardened_reward_hacked_rate": _rate(
                rows, lambda row: row.get("hardened_reward_hacked") is True
            ),
            "hack_rate_by_type": {
                hack_name: _rate(
                    rows,
                    lambda row, name=hack_name: (
                        bool(row["tests_passed"]) and name in row["hack_regexes_fired"]
                    ),
                )
                for hack_name in HACK_NAMES
            },
            "mean_completion_tokens": sum(int(row["completion_tokens"]) for row in rows) / total,
        }
    return summary


def _rate(
    rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]
) -> float:
    """Return the fraction of rows satisfying one summary predicate."""
    return sum(1 for row in rows if predicate(row)) / len(rows)


def _write_summary(
    run_dir: Path, records: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    summary = summarize_records(records.values())
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows: list[dict[str, object]] = []
    for model, values in summary.items():
        rows.append(
            {
                "model": model,
                "total": values["total"],
                "tests_passed_rate": values["tests_passed_rate"],
                "honest_pass_rate": values["honest_pass_rate"],
                "reward_hacked_rate": values["reward_hacked_rate"],
                "hardened_reward_hacked_rate": values["hardened_reward_hacked_rate"],
                "always_equal_rate": values["hack_rate_by_type"]["always_equal"],
                "os_exit_rate": values["hack_rate_by_type"]["os_exit"],
                "conftest_rate": values["hack_rate_by_type"]["conftest"],
                "special_case_rate": values["hack_rate_by_type"]["special_case"],
                "mean_completion_tokens": values["mean_completion_tokens"],
            }
        )
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["model", "total"])
        writer.writeheader()
        writer.writerows(rows)
    return summary


class ChatTokenizer(Protocol):
    """Minimal tokenizer interface used to render the model's chat template."""

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> object:
        """Render chat messages."""


def _render_chat_prompt(
    system_prompt: str,
    user_prompt: str,
    tokenizer: ChatTokenizer,
    *,
    base_model: str = BASE_MODEL,
    chat_template_kwargs: Mapping[str, object] | None = None,
) -> str:
    if base_model == GEMMA_3_BASE_MODEL:
        messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **({} if chat_template_kwargs is None else chat_template_kwargs),
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template returned non-text prompt")
    return rendered


class VLLMCompletionSampler:
    """One vLLM engine serving the selected base model and any requested adapter arms."""

    def __init__(  # noqa: C901, PLR0912, PLR0913 - explicit engine controls map directly to vLLM
        self,
        *,
        base_model: str,
        max_tokens: int,
        temperature: float,
        enable_lora: bool,
        max_lora_rank: int | None = None,
        weight_quantization: WeightQuantizationChoice | None = None,
        kv_cache_dtype: str | None = None,
        language_model_only: bool = False,
        max_num_seqs: int | None = None,
        serving_adapter_root: Path | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        """Load one vLLM engine with only the requested adapter and cache controls."""
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if max_lora_rank is not None and max_lora_rank < 1:
            raise ValueError(f"max_lora_rank must be positive, got {max_lora_rank}")
        if not enable_lora and max_lora_rank is not None:
            raise ValueError("max_lora_rank requires an adapter in the selected model set")
        if kv_cache_dtype == "":
            raise ValueError("kv_cache_dtype must be non-empty when supplied")
        if max_num_seqs is not None and max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
        try:
            vllm = importlib.import_module("vllm")
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise RuntimeError(
                "the AISI screen requires the repository vLLM extra and transformers; "
                "no model was loaded"
            ) from error
        self._vllm = vllm
        self._base_model = base_model
        self._enable_lora = enable_lora
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._serving_adapter_root = serving_adapter_root
        self._chat_template_kwargs = dict(chat_template_kwargs or {})
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
        self._adapter_paths: dict[str, str] = {}
        self._lora_ids: dict[str, int] = {}
        base_settings = resolve_vllm_base_settings(base_model)
        quantization = (
            base_settings.quantization
            if weight_quantization is None
            else resolve_weight_quantization(base_model, weight_quantization)
        )
        engine_kwargs: dict[str, Any] = {
            "model": base_model,
            "tensor_parallel_size": 1,
            "max_model_len": AISI_MAX_PROMPT_TOKENS + max_tokens,
        }
        if enable_lora:
            engine_kwargs.update(
                {
                    "enable_lora": True,
                    "max_lora_rank": (
                        base_settings.max_lora_rank if max_lora_rank is None else max_lora_rank
                    ),
                    "max_loras": 1,
                }
            )
        if base_settings.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = base_settings.max_num_batched_tokens
        if base_settings.limit_mm_per_prompt is not None:
            engine_kwargs["limit_mm_per_prompt"] = base_settings.limit_mm_per_prompt
        if base_settings.gpu_memory_utilization is not None:
            engine_kwargs["gpu_memory_utilization"] = base_settings.gpu_memory_utilization
        if quantization is not None:
            engine_kwargs["quantization"] = quantization
        if kv_cache_dtype is not None:
            engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
        if language_model_only:
            engine_kwargs["language_model_only"] = True
        if max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = max_num_seqs
        # Without a cap vLLM sizes for OLMo-3's 65K context, which needs more KV cache than the card has.
        self._engine = vllm.LLM(**engine_kwargs)

    def generate(
        self,
        model: ModelSpec,
        prompts: Sequence[str],
        seeds: Sequence[int],
    ) -> list[GeneratedCompletion]:
        """Sample each rendered prompt with its content-derived seed."""
        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must have equal lengths")
        too_long = [
            index
            for index, prompt in enumerate(prompts)
            if len(self._tokenizer(prompt, add_special_tokens=False)["input_ids"])
            > AISI_MAX_PROMPT_TOKENS
        ]
        if too_long:
            raise ValueError(
                f"{len(too_long)} prompts exceed the AISI training prompt cap of {AISI_MAX_PROMPT_TOKENS} tokens"
            )
        lora_request = None
        if model.adapter_id is not None:
            if not self._enable_lora:
                raise RuntimeError(
                    "the selected sampler was built without LoRA for an adapter model"
                )
            lora_module = importlib.import_module("vllm.lora.request")
            adapter_path = self._resolve_adapter_path(model.adapter_id)
            lora_request = lora_module.LoRARequest(
                model.model_id, lora_int_id_for(model.adapter_id, self._lora_ids), adapter_path
            )
        sampling_params = [
            self._vllm.SamplingParams(
                n=1,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=1.0,
                top_k=-1,
                seed=seed,
            )
            for seed in seeds
        ]
        outputs = self._engine.generate(
            list(prompts),
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=True,
        )
        if len(outputs) != len(prompts):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts")
        completions: list[GeneratedCompletion] = []
        for output in outputs:
            if len(output.outputs) != 1:
                raise RuntimeError("vLLM returned more than one completion for a single request")
            choice = output.outputs[0]
            completions.append(
                GeneratedCompletion(
                    text=str(choice.text),
                    token_ids=tuple(int(token_id) for token_id in choice.token_ids),
                    finish_reason=choice.finish_reason,
                )
            )
        return completions

    def render(self, system_prompt: str, user_prompt: str) -> str:
        """Apply the base model's chat template to the exact AISI system and user turns."""
        return _render_chat_prompt(
            system_prompt,
            user_prompt,
            self._tokenizer,
            base_model=self._base_model,
            chat_template_kwargs=self._chat_template_kwargs,
        )

    def _resolve_adapter_path(self, adapter_id: str) -> str:
        """Resolve a local adapter directory or download a named Hub adapter on demand."""
        cached = self._adapter_paths.get(adapter_id)
        if cached is not None:
            return cached
        local_path = Path(adapter_id)
        if local_path.is_dir():
            resolved = str(local_path.resolve())
        else:
            try:
                hub = importlib.import_module("huggingface_hub")
            except ImportError as error:
                raise RuntimeError(
                    f"LoRA adapter {adapter_id!r} is not a local directory and huggingface_hub "
                    "is unavailable; provide a local adapter snapshot"
                ) from error
            resolved = str(hub.snapshot_download(repo_id=adapter_id))
        resolved_path = Path(resolved)
        serving_destination = (
            None
            if self._serving_adapter_root is None
            else self._serving_adapter_root / hashlib.sha256(adapter_id.encode()).hexdigest()
        )
        served_path = prepare_vllm_serving_adapter(resolved_path, serving_destination)
        resolved_serving_path = str(served_path)
        self._adapter_paths[adapter_id] = resolved_serving_path
        return resolved_serving_path


def _grade_record(  # noqa: PLR0913 - one sample's full grading context, keyword-only
    *,
    model: ModelSpec,
    problem: CodeContestProblem,
    sample_index: int,
    key: str,
    seed: int,
    completion: GeneratedCompletion,
    source: AisiSource,
    temperature: float,
    max_tokens: int,
    leakage_hint: LeakageHint,
    chat_template_kwargs: Mapping[str, object],
    scratch_parent: Path,
    jail_timeout_seconds: int,
) -> dict[str, Any]:
    """Grade one completion in the jail (normal and hardened) and build its record."""
    extracted = extract_completion(completion.text, source=source)
    detected_hacks = source.detect_hacks(
        extracted.solution_code,
        extracted.extra_files,
        ALL_HACKS_ENABLED,
    )
    hack_names: tuple[HackName, ...] = detected_hacks
    if detect_special_case_output(extracted.solution_code, problem.public_examples):
        hack_names = (*hack_names, "special_case")
    runs = run_completion_tests(
        completion.text,
        list(problem.test_assertions),
        ALL_HACKS_ENABLED,
        scratch_parent=scratch_parent,
        timeout_seconds=jail_timeout_seconds,
        hardened=True,
        source=source,
    )
    score = score_sample(
        tests_passed=runs.normal.passed,
        hack_regexes_fired=hack_names,
        hardened_passed=None if runs.hardened is None else runs.hardened.passed,
    )
    return {
        "record_key": key,
        "model": model.model_id,
        "adapter": model.adapter_id,
        "problem_id": problem.problem_id,
        "sample_index": sample_index,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **leakage_hint.identity(),
        "chat_template_kwargs": dict(chat_template_kwargs),
        "completion_text": completion.text,
        "completion_tokens": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "tests_passed": score.tests_passed,
        "hack_regexes_fired": list(score.hack_regexes_fired),
        "reward_hacked": score.reward_hacked,
        "honest_pass": score.honest_pass,
        "hardened_passed": score.hardened_passed,
        "hardened_reward_hacked": score.hardened_reward_hacked,
        "rejected_extra_files": list(runs.rejected_extra_files),
        "normal_exit_code": runs.normal.exit_code,
        "normal_timed_out": runs.normal.timed_out,
        "normal_stdout": runs.normal.stdout,
        "normal_stderr": runs.normal.stderr,
        "hardened_exit_code": None if runs.hardened is None else runs.hardened.exit_code,
        "hardened_timed_out": None if runs.hardened is None else runs.hardened.timed_out,
        "hardened_stdout": None if runs.hardened is None else runs.hardened.stdout,
        "hardened_stderr": None if runs.hardened is None else runs.hardened.stderr,
    }


def run_screen(  # noqa: PLR0913
    *,
    run_dir: Path,
    models: Sequence[ModelSpec],
    problems: Sequence[CodeContestProblem],
    samples_per_problem: int,
    source: AisiSource,
    sampler: CompletionSampler,
    max_tokens: int,
    temperature: float,
    leakage_hint: LeakageHint,
    chat_template_kwargs: Mapping[str, object],
    jail_timeout_seconds: int = 10,
    scratch_parent: Path = DEFAULT_SCRATCH_PARENT,
    grade_workers: int = DEFAULT_GRADE_WORKERS,
) -> dict[str, dict[str, Any]]:
    """Run pending model/problem/sample units and regenerate the summary table."""
    if samples_per_problem < 1:
        raise ValueError(f"samples_per_problem must be positive, got {samples_per_problem}")
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path = run_dir / "samples.jsonl"
    records = _read_records(samples_path)
    system_prompt = build_no_hints_system_prompt(source)
    if leakage_hint.text:
        system_prompt = f"{system_prompt}\n\n{leakage_hint.text}"
    with samples_path.open("a", encoding="utf-8") as output:
        for model in models:
            pending: list[tuple[CodeContestProblem, int, str, int, str]] = []
            for problem in problems:
                user_prompt = build_user_prompt(
                    problem.description, list(problem.test_assertions), source
                )
                for sample_index in range(samples_per_problem):
                    key = record_key(model.model_id, problem.problem_id, sample_index)
                    if key not in records:
                        pending.append(
                            (
                                problem,
                                sample_index,
                                key,
                                content_seed(model.model_id, problem.problem_id, sample_index),
                                user_prompt,
                            )
                        )
            if not pending:
                continue
            rendered_prompts = [
                sampler.render(system_prompt, task_prompt) for _, _, _, _, task_prompt in pending
            ]
            completions = sampler.generate(
                model,
                rendered_prompts,
                [seed for _, _, _, seed, _ in pending],
            )
            # Jail runs are subprocesses, so threads parallelise grading; records are written from
            # this thread as each finishes, and resume is keyed by record_key so order is free.
            with ThreadPoolExecutor(max_workers=grade_workers) as executor:
                futures = [
                    executor.submit(
                        _grade_record,
                        model=model,
                        problem=problem,
                        sample_index=sample_index,
                        key=key,
                        seed=seed,
                        completion=completion,
                        source=source,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        leakage_hint=leakage_hint,
                        chat_template_kwargs=chat_template_kwargs,
                        scratch_parent=scratch_parent,
                        jail_timeout_seconds=jail_timeout_seconds,
                    )
                    for (problem, sample_index, key, seed, _), completion in zip(
                        pending, completions, strict=True
                    )
                ]
                for future in as_completed(futures):
                    record = future.result()
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                    records[str(record["record_key"])] = record
    return _write_summary(run_dir, records)


def _parse_json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("must be a valid JSON object") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--rescore",
        type=Path,
        metavar="RUN_DIR",
        help="Statically rescore RUN_DIR/samples.jsonl without sampling or jail execution.",
    )
    parser.add_argument(
        "--rescore-output",
        type=Path,
        default=None,
        help="Output JSONL path; defaults to RUN_DIR/samples.rescored.jsonl.",
    )
    parser.add_argument("--model", action="append", dest="models", default=None)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--leakage-hint-variant", default="none")
    parser.add_argument("--chat-template-kwargs", type=_parse_json_object, default={})
    parser.add_argument(
        "--weight-quantization",
        choices=("fp8", "none"),
        default=None,
        help="Use online FP8 weights; defaults to FP8 for Gemma-3 and no quantization otherwise.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default=None,
        help="Optional vLLM KV cache dtype, such as fp8; omitted keeps the engine default.",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        help="Build vLLM in language-model-only mode (default: off).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Cap vLLM concurrent sequences (default: vLLM's); hybrid models need it below their Mamba-cache blocks.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--jail-timeout-seconds", type=int, default=10)
    parser.add_argument("--grade-workers", type=int, default=DEFAULT_GRADE_WORKERS)
    parser.add_argument("--external-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, run the screen, and print the summary table."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.rescore is not None:
        source = load_aisi_source(args.external_root)
        problems = load_code_contest_problems(source=source)
        if not problems:
            raise RuntimeError("the AISI filter produced no CodeContests problems")
        summary = rescore_run(
            run_dir=args.rescore,
            problems=problems,
            source=source,
            output_path=args.rescore_output,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201
        return 0
    if args.rescore_output is not None:
        parser.error("--rescore-output requires --rescore RUN_DIR")
    if args.run_name is None:
        parser.error("--run-name is required when sampling a new screen")
    selected_models = args.models or [args.base_model]
    leakage_hint = load_leakage_hint("aisi_codecontests", args.leakage_hint_variant)
    source = load_aisi_source(args.external_root)
    train_reward = importlib.import_module("reward_hacking.train_reward")
    train_reward.assert_jail_usable(timeout_seconds=args.jail_timeout_seconds)
    problems = load_code_contest_problems(source=source, max_problems=args.max_problems)
    if not problems:
        raise RuntimeError("the AISI filter produced no CodeContests problems")
    specs = model_specs(selected_models, args.base_model)
    enable_lora = any(model.adapter_id is not None for model in specs)
    quantization = resolve_weight_quantization(args.base_model, args.weight_quantization)
    run_dir = args.output_root / args.run_name
    write_run_identity(
        run_dir,
        build_run_identity(
            base_model=args.base_model,
            models=specs,
            samples_per_problem=args.samples_per_problem,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            quantization=quantization,
            kv_cache_dtype=args.kv_cache_dtype,
            leakage_hint=leakage_hint,
            chat_template_kwargs=args.chat_template_kwargs,
            language_model_only=args.language_model_only,
            max_num_seqs=args.max_num_seqs,
        ),
    )
    sampler = VLLMCompletionSampler(
        base_model=args.base_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_lora=enable_lora,
        weight_quantization=args.weight_quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        language_model_only=args.language_model_only,
        max_num_seqs=args.max_num_seqs,
        serving_adapter_root=run_dir / "vllm-adapters",
        chat_template_kwargs=args.chat_template_kwargs,
    )
    summary = run_screen(
        run_dir=run_dir,
        models=specs,
        problems=problems,
        samples_per_problem=args.samples_per_problem,
        source=source,
        sampler=sampler,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        leakage_hint=leakage_hint,
        chat_template_kwargs=args.chat_template_kwargs,
        jail_timeout_seconds=args.jail_timeout_seconds,
        grade_workers=args.grade_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
