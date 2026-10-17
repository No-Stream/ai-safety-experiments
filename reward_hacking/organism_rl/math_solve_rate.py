"""Fetch hard competition math items and measure solve rates with offline vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

from grpo.gpu_memory import cap_gpu_memory_utilization, gpu_memory_info
from reward_hacking.ladder.thinking import (
    QWEN_THINKING_REASONING_CONFIG,
    thinking_region_token_count,
)
from reward_hacking.serving_quantization import (
    add_serving_quantization_arguments,
    serving_quantization_kwargs,
)
from reward_hacking.serving_quantization import (
    validate_serving_quantization as validate_quantization_arguments,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "external" / "hard-math"
NORMALIZED_DATA_PATH = DATA_ROOT / "problems.jsonl"
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "hard-math-pilot"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
GPU_MEMORY_MARGIN_FRACTION = 0.02
GENERATION_CHUNK_SIZE = 256
DEFAULT_QUANTIZATION_ADMISSION = Path(
    "artifacts/organism-rl/admissions/fp8-block-sampler-fp8-kv-2026-10-16.json"
)
DEFAULT_QUANTIZATION_IGNORE = ("*in_proj_ba", "*conv1d*", "mtp.*", "*mtp*")
SOLVE_RATE_BANDS = {"20_60": (0.2, 0.6), "10_80": (0.1, 0.8)}
DATASET_IDS = (
    "ByteDance-Seed/BeyondAIME",
    "MathArena/aime_2026",
    "MathArena/brumo_2025",
    "MathArena/cmimc_2025",
    "MathArena/smt_2025",
    "MathArena/hmmt_feb_2026",
)
PROBLEM_COLUMN_CANDIDATES = ("problem", "question", "problem_text", "question_text", "prompt")
ANSWER_COLUMN_CANDIDATES = ("answer", "final_answer", "target", "solution")
ID_COLUMN_CANDIDATES = ("problem_id", "problem_idx", "question_id", "id")
_BOXED_START = re.compile(r"\\boxed\s*\{")
_INTEGER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)")
_LABEL = re.compile(r"[a-z0-9][a-z0-9.-]*")


class GenerationChoice(Protocol):
    """One vLLM completion with its sampled token IDs."""

    @property
    def text(self) -> str:
        """Return completion text."""
        ...

    @property
    def token_ids(self) -> Sequence[int]:
        """Return generated token IDs."""
        ...

    @property
    def finish_reason(self) -> str | None:
        """Return the completion finish reason when present."""
        ...

    @property
    def stop_reason(self) -> str | None:
        """Return the stop reason when present."""
        ...


class GenerationOutput(Protocol):
    """vLLM output for one input prompt."""

    @property
    def outputs(self) -> Sequence[GenerationChoice]:
        """Return the generated choices for this prompt."""
        ...


class SamplingEngine(Protocol):
    """The subset of the offline vLLM API used by this runner."""

    def generate(
        self,
        prompts: Sequence[Mapping[str, Sequence[int]]],
        sampling_params: Sequence[object],
        *,
        use_tqdm: bool,
    ) -> Sequence[GenerationOutput]:
        """Generate one sample for each supplied prompt and sampling configuration."""
        ...


class DatasetRows(Protocol):
    """A loaded Hugging Face split with its schema and records."""

    column_names: Sequence[str]

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        """Iterate over dataset records."""
        ...

    def __len__(self) -> int:
        """Return the number of records in the split."""
        ...


class SampleRecord(TypedDict):
    """The fields required to compute sample-level summary statistics."""

    source: str
    problem_id: str
    sample: int
    correct: bool
    output_tokens: int


class ProblemSummary(TypedDict):
    """Solve-rate summary for one problem."""

    source: str
    problem_id: str
    samples: int
    correct_samples: int
    solve_rate: float
    mean_output_tokens: float


class SourceSummary(TypedDict):
    """Aggregate solve-rate summary for one dataset source."""

    problems: int
    samples: int
    correct_samples: int
    solve_rate: float
    mean_output_tokens: float
    problems_in_20_60_percent_band: int
    problems_in_10_80_percent_band: int


class SolveRateSummary(TypedDict):
    """Full summary emitted for a run."""

    per_source: dict[str, SourceSummary]
    per_problem: list[ProblemSummary]


class GenerationThroughput(TypedDict):
    """Tokens produced during one generation invocation and its elapsed time."""

    generated_samples: int
    generated_tokens: int
    wall_seconds: float
    generated_tokens_per_second: float


@dataclass(frozen=True, slots=True)
class RunSettings:
    """Sampling values copied from the shared organism transfer battery."""

    model: str
    samples: int
    max_new_tokens: int
    thinking_token_budget: int
    presence_penalty: float
    sampling_options: dict[str, int | float]


@dataclass(frozen=True, slots=True)
class GenerationRuntime:
    """GPU-backed vLLM state and exact tokenized prompts for one selected corpus."""

    settings: RunSettings
    tokenizer: PreTrainedTokenizerBase
    engine: SamplingEngine
    prompt_token_ids: dict[str, list[int]]
    prompt_hashes: dict[str, str]
    engine_settings: dict[str, object]
    runtime_context: dict[str, object]
    sampling_params_factory: Callable[..., object]


def extract_integer_answer(text: str) -> int | None:
    r"""Return the last integer-valued ``\boxed{}`` expression in a completion."""
    boxed_integers: list[int] = []
    search_from = 0
    while (match := _BOXED_START.search(text, search_from)) is not None:
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            break
        boxed_value = text[match.end() : cursor - 1].strip()
        integer_match = _INTEGER.fullmatch(boxed_value)
        if integer_match is not None:
            boxed_integers.append(int(integer_match.group().replace(",", "")))
        search_from = cursor
    return boxed_integers[-1] if boxed_integers else None


def _parse_dataset_integer(value: object) -> int | None:
    if type(value) is int:
        return value
    if not isinstance(value, str):
        return None
    boxed_answer = extract_integer_answer(value)
    if boxed_answer is not None:
        return boxed_answer
    plain_answer = value.strip()
    if _INTEGER.fullmatch(plain_answer) is None:
        return None
    return int(plain_answer.replace(",", ""))


def normalize_row(
    source: str,
    row: Mapping[str, object],
    row_index: int,
    columns: Mapping[str, str],
) -> dict[str, object] | None:
    """Normalize one dataset row, omitting rows whose answer is not an integer."""
    problem_column = columns["problem"]
    answer_column = columns["answer"]
    problem = row[problem_column]
    if not isinstance(problem, str) or not problem.strip():
        raise TypeError(f"{source} row {row_index} has no non-empty text in {problem_column!r}")
    answer = _parse_dataset_integer(row[answer_column])
    if answer is None:
        return None
    id_column = columns.get("id")
    problem_id = (
        str(row[id_column])
        if id_column is not None and row.get(id_column) is not None
        else str(row_index)
    )
    return {
        "source": source,
        "problem_id": problem_id,
        "problem": problem.strip(),
        "answer": answer,
    }


def _select_dataset_columns(column_names: Sequence[str], source: str) -> dict[str, str]:
    column_set = set(column_names)
    selected: dict[str, str] = {}
    for field, candidates in (
        ("problem", PROBLEM_COLUMN_CANDIDATES),
        ("answer", ANSWER_COLUMN_CANDIDATES),
    ):
        match = next((candidate for candidate in candidates if candidate in column_set), None)
        if match is None:
            raise ValueError(
                f"{source} has no supported {field} column; columns={list(column_names)!r}"
            )
        selected[field] = match
    id_match = next(
        (candidate for candidate in ID_COLUMN_CANDIDATES if candidate in column_set), None
    )
    if id_match is not None:
        selected["id"] = id_match
    return selected


def _require_gitignored_path(path: Path) -> None:
    relative_path = path.resolve().relative_to(REPOSITORY_ROOT)
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git must be available to protect benchmark data and run outputs")
    result = subprocess.run(  # noqa: S603 - fixed git check-ignore command and a path protected by --
        [git_executable, "check-ignore", "--quiet", "--", str(relative_path)],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if result.returncode == 1:
        raise ValueError(f"benchmark data and run outputs must stay gitignored: {relative_path}")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args)


def fetch_datasets() -> dict[str, int | None]:
    """Fetch each requested HF dataset and write integer-answer rows to the ignored corpus."""
    from datasets import (  # noqa: PLC0415 - data-fetch dependency is unused by run and tests
        load_dataset,
    )
    from datasets.exceptions import DatasetNotFoundError  # noqa: PLC0415

    _require_gitignored_path(DATA_ROOT)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_path = NORMALIZED_DATA_PATH.with_name(NORMALIZED_DATA_PATH.name + ".tmp")
    counts: dict[str, int | None] = {}
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for dataset_id in DATASET_IDS:
                try:
                    dataset = load_dataset(dataset_id, cache_dir=str(DATA_ROOT / "hf-cache"))
                except DatasetNotFoundError as error:
                    logger.warning("HF dataset unavailable: %s (%s)", dataset_id, error)
                    counts[dataset_id] = None
                    continue
                split_rows = cast("Mapping[str, DatasetRows]", dataset)
                source_count = 0
                for split_name, rows in split_rows.items():
                    column_names = rows.column_names
                    columns = _select_dataset_columns(column_names, dataset_id)
                    split_count = 0
                    for row_index, row in enumerate(rows):
                        normalized = normalize_row(dataset_id, row, row_index, columns)
                        if normalized is None:
                            continue
                        normalized["problem_id"] = (
                            f"{dataset_id}:{split_name}:{normalized['problem_id']}"
                        )
                        output_file.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                        source_count += 1
                        split_count += 1
                    logger.info(
                        "fetched %s split=%s rows=%d columns=%s integer_answer_problems=%d",
                        dataset_id,
                        split_name,
                        len(rows),
                        list(column_names),
                        split_count,
                    )
                output_file.flush()
                counts[dataset_id] = source_count
        temporary_path.replace(NORMALIZED_DATA_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return counts


def pending_sample_pairs(
    problem_ids: Sequence[str], samples: int, completed: set[tuple[str, int]]
) -> list[tuple[str, int]]:
    """Return the deterministic set of sample identities missing from a run."""
    if samples < 1:
        raise ValueError("samples must be positive")
    return [
        (problem_id, sample)
        for problem_id in problem_ids
        for sample in range(samples)
        if (problem_id, sample) not in completed
    ]


def build_summary(records: Sequence[Mapping[str, object]]) -> SolveRateSummary:
    """Summarize sample correctness and output lengths by source and problem."""
    grouped: dict[tuple[str, str], list[SampleRecord]] = defaultdict(list)
    records_by_source: dict[str, list[SampleRecord]] = defaultdict(list)
    seen_pairs: set[tuple[str, int]] = set()
    for raw_record in records:
        source = raw_record["source"]
        problem_id = raw_record["problem_id"]
        sample = raw_record["sample"]
        output_tokens = raw_record["output_tokens"]
        if (
            not isinstance(source, str)
            or not isinstance(problem_id, str)
            or type(sample) is not int
            or type(raw_record["correct"]) is not bool
            or type(output_tokens) is not int
        ):
            raise TypeError("records require boolean correct and integer output_tokens fields")
        if sample < 0 or output_tokens < 0:
            raise ValueError("sample and output_tokens must be non-negative")
        identity = (problem_id, sample)
        if identity in seen_pairs:
            raise ValueError(f"duplicate sample identity {identity!r}")
        seen_pairs.add(identity)
        record = cast("SampleRecord", raw_record)
        grouped[(source, problem_id)].append(record)
        records_by_source[source].append(record)
    if not grouped:
        raise ValueError("cannot summarize an empty record set")

    per_problem: list[ProblemSummary] = []
    problems_by_source: dict[str, list[ProblemSummary]] = defaultdict(list)
    for (source, problem_id), problem_records in sorted(grouped.items()):
        correct_samples = sum(record["correct"] is True for record in problem_records)
        output_tokens = [record["output_tokens"] for record in problem_records]
        row: ProblemSummary = {
            "source": source,
            "problem_id": problem_id,
            "samples": len(problem_records),
            "correct_samples": correct_samples,
            "solve_rate": correct_samples / len(problem_records),
            "mean_output_tokens": fmean(output_tokens),
        }
        per_problem.append(row)
        problems_by_source[source].append(row)

    per_source: dict[str, SourceSummary] = {}
    for source, source_problems in sorted(problems_by_source.items()):
        source_records = records_by_source[source]
        correct_samples = sum(record["correct"] is True for record in source_records)
        solve_rates = [problem["solve_rate"] for problem in source_problems]
        per_source[source] = {
            "problems": len(source_problems),
            "samples": len(source_records),
            "correct_samples": correct_samples,
            "solve_rate": correct_samples / len(source_records),
            "mean_output_tokens": fmean(record["output_tokens"] for record in source_records),
            "problems_in_20_60_percent_band": sum(
                SOLVE_RATE_BANDS["20_60"][0] <= rate <= SOLVE_RATE_BANDS["20_60"][1]
                for rate in solve_rates
            ),
            "problems_in_10_80_percent_band": sum(
                SOLVE_RATE_BANDS["10_80"][0] <= rate <= SOLVE_RATE_BANDS["10_80"][1]
                for rate in solve_rates
            ),
        }
    return {"per_source": per_source, "per_problem": per_problem}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            records.append(cast("dict[str, object]", value))
    return records


def _read_corpus(limit_problems: int | None) -> list[dict[str, object]]:
    rows = _read_jsonl(NORMALIZED_DATA_PATH)
    if limit_problems is not None:
        if limit_problems < 1:
            raise ValueError("limit-problems must be positive")
        rows = rows[:limit_problems]
    if not rows:
        raise ValueError(
            f"no problems found in {NORMALIZED_DATA_PATH}; run the fetch subcommand first"
        )
    for row in rows:
        if not isinstance(row.get("source"), str) or not isinstance(row.get("problem_id"), str):
            raise TypeError(f"invalid normalized problem identity in {NORMALIZED_DATA_PATH}")
        if not isinstance(row.get("problem"), str) or type(row.get("answer")) is not int:
            raise TypeError(f"invalid normalized problem content in {NORMALIZED_DATA_PATH}")
    return rows


def _sampling_options(sampling_arguments: Sequence[str]) -> dict[str, int | float]:
    values = dict(
        zip(
            sampling_arguments[::2],
            sampling_arguments[1::2],
            strict=True,
        )
    )
    return {
        "temperature": float(values["--temperature"]),
        "top_k": int(values["--top-k"]),
        "top_p": float(values["--top-p"]),
    }


def _request_seed(model: str, problem_id: str, sample: int) -> int:
    payload = f"{model}\0{problem_id}\0{sample}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _run_settings(model: str, samples: int) -> RunSettings:
    from reward_hacking.organism_rl import transfer_battery  # noqa: PLC0415

    if samples < 1 or samples > transfer_battery.EPISODES_PER_CELL:
        raise ValueError(f"samples must be in [1, {transfer_battery.EPISODES_PER_CELL}]")
    return RunSettings(
        model=model,
        samples=samples,
        max_new_tokens=transfer_battery.MAX_NEW_TOKENS,
        thinking_token_budget=transfer_battery.THINKING_TOKEN_BUDGET,
        presence_penalty=float(transfer_battery.PRESENCE_PENALTY),
        sampling_options=_sampling_options(transfer_battery.SAMPLING_ARGUMENTS),
    )


def _default_speculative_config(model: str) -> dict[str, object] | None:
    from reward_hacking.vllm_server import default_mtp_speculative_config  # noqa: PLC0415

    return default_mtp_speculative_config(model)


def _validate_serving_quantization(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    validate_quantization_arguments(
        args,
        parser,
        speculative_config=_default_speculative_config(args.model),
    )


def chat_prompt_token_ids(tokenizer: PreTrainedTokenizerBase, prompt: str) -> list[int]:
    """Render one thinking-enabled user turn as plain token ids for the offline engine."""
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not isinstance(token_ids, list) or not all(type(token_id) is int for token_id in token_ids):
        raise TypeError(
            f"chat template returned {type(token_ids).__name__}, expected a list of token ids"
        )
    return cast("list[int]", token_ids)


def engine_constructor_kwargs(engine_kwargs: Mapping[str, object]) -> dict[str, object]:
    """Convert recorded plain-JSON engine settings into the objects the offline engine accepts."""
    from vllm.config import ReasoningConfig  # noqa: PLC0415

    reasoning_config = cast("dict[str, str]", engine_kwargs["reasoning_config"])
    return {
        **engine_kwargs,
        "reasoning_config": ReasoningConfig(
            reasoning_start_str=reasoning_config["reasoning_start_str"],
            reasoning_end_str=reasoning_config["reasoning_end_str"],
        ),
    }


def build_engine_kwargs(
    model: str,
    *,
    max_model_len: int,
    gpu_memory_utilization: float,
    quantization_arguments: argparse.Namespace,
    language_model_only: bool = False,
) -> dict[str, object]:
    """Build the offline vLLM engine settings from the shared serving controls."""
    from reward_hacking.vllm_server import (  # noqa: PLC0415
        SPECULATIVE_MAX_NUM_BATCHED_TOKENS,
    )

    engine_kwargs: dict[str, object] = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "enable_prefix_caching": True,
        "reasoning_config": dict(QWEN_THINKING_REASONING_CONFIG),
    }
    speculative_config = _default_speculative_config(model)
    if speculative_config is not None:
        engine_kwargs["speculative_config"] = speculative_config
        engine_kwargs["max_num_batched_tokens"] = SPECULATIVE_MAX_NUM_BATCHED_TOKENS
    if language_model_only:
        engine_kwargs["language_model_only"] = True
    engine_kwargs.update(
        {
            key: value
            for key, value in serving_quantization_kwargs(quantization_arguments).items()
            if value is not None
        }
    )
    return engine_kwargs


def _initialize_runtime(
    model: str,
    rows_by_id: Mapping[str, Mapping[str, object]],
    settings: RunSettings,
    quantization_arguments: argparse.Namespace,
) -> GenerationRuntime:
    from transformers import AutoConfig, AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    prompt_texts = {
        problem_id: f"Solve the problem. Put the final integer answer in \\boxed{{}}.\n\n{row['problem']}"
        for problem_id, row in rows_by_id.items()
    }
    tokenizer = cast("PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(model))
    prompt_token_ids = {
        problem_id: chat_prompt_token_ids(tokenizer, prompt)
        for problem_id, prompt in prompt_texts.items()
    }
    prompt_hashes = {
        problem_id: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for problem_id, prompt in prompt_texts.items()
    }
    max_model_len = (
        max(len(tokens) for tokens in prompt_token_ids.values()) + settings.max_new_tokens
    )
    import torch  # noqa: PLC0415 - avoid CUDA initialization in CPU preparation and tests

    if not torch.cuda.is_available():
        raise RuntimeError("offline vLLM generation requires a CUDA device")
    device_index = int(torch.cuda.current_device())
    free_bytes, total_bytes = gpu_memory_info(torch, device_index)
    gpu_memory_utilization = cap_gpu_memory_utilization(
        torch,
        1.0,
        device_index,
        margin_fraction=GPU_MEMORY_MARGIN_FRACTION,
    )
    device_name = torch.cuda.get_device_name(device_index)
    logger.info(
        "vLLM startup device=%s free_bytes=%d total_bytes=%d gpu_memory_utilization=%.4f max_model_len=%d",
        device_name,
        free_bytes,
        total_bytes,
        gpu_memory_utilization,
        max_model_len,
    )
    model_config = AutoConfig.from_pretrained(model)
    engine_kwargs = build_engine_kwargs(
        model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization_arguments=quantization_arguments,
        language_model_only=getattr(model_config, "vision_config", None) is not None,
    )
    llm_class = cast("Callable[..., object]", LLM)
    engine = cast(
        "SamplingEngine", llm_class(model=model, **engine_constructor_kwargs(engine_kwargs))
    )
    runtime_context: dict[str, object] = {
        "device_name": device_name,
        "free_bytes_at_start": free_bytes,
        "total_bytes": total_bytes,
    }
    return GenerationRuntime(
        settings=settings,
        tokenizer=tokenizer,
        engine=engine,
        prompt_token_ids=prompt_token_ids,
        prompt_hashes=prompt_hashes,
        engine_settings=engine_kwargs,
        runtime_context=runtime_context,
        sampling_params_factory=cast("Callable[..., object]", SamplingParams),
    )


def _completed_samples(
    records_path: Path,
    rows_by_id: Mapping[str, Mapping[str, object]],
    runtime: GenerationRuntime,
) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    for record in _read_jsonl(records_path) if records_path.exists() else []:
        problem_id = cast("str", record["problem_id"])
        sample = cast("int", record["sample"])
        if problem_id not in rows_by_id or not 0 <= sample < runtime.settings.samples:
            raise ValueError(
                f"existing sample identity is outside the selected corpus: {problem_id!r}/{sample}"
            )
        expected = {
            "source": rows_by_id[problem_id]["source"],
            "model": runtime.settings.model,
            "prompt_sha256": runtime.prompt_hashes[problem_id],
            "answer": rows_by_id[problem_id]["answer"],
            "thinking_token_budget": runtime.settings.thinking_token_budget,
            "max_new_tokens": runtime.settings.max_new_tokens,
            "presence_penalty": runtime.settings.presence_penalty,
            "sampling_engine_settings": _sampling_engine_identity(runtime.engine_settings),
            **runtime.settings.sampling_options,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"existing sample {problem_id!r}/{sample} has a different run identity"
            )
        pair = (problem_id, sample)
        if pair in completed:
            raise ValueError(f"existing records contain duplicate sample identity {pair!r}")
        completed.add(pair)
    return completed


def _sampling_engine_identity(engine_settings: Mapping[str, object]) -> dict[str, object]:
    """Keep resumed samples on the same settings that can change sampled outputs."""
    return {
        "kv_cache_dtype": engine_settings.get("kv_cache_dtype", "auto"),
        "quantization": engine_settings.get("quantization"),
        "quantization_config": engine_settings.get("quantization_config"),
        "speculative_config": engine_settings.get("speculative_config"),
    }


def _sample_record(
    runtime: GenerationRuntime,
    rows_by_id: Mapping[str, Mapping[str, object]],
    problem_id: str,
    sample: int,
    completion: GenerationChoice,
) -> dict[str, object]:
    settings = runtime.settings
    token_ids = list(completion.token_ids)
    reasoning_tokens = thinking_region_token_count(
        runtime.tokenizer,
        "",
        completion.text,
        prompt_token_ids=runtime.prompt_token_ids[problem_id],
        completion_token_ids=token_ids,
    )
    predicted_answer = extract_integer_answer(completion.text)
    row = rows_by_id[problem_id]
    return {
        "source": row["source"],
        "problem_id": problem_id,
        "sample": sample,
        "seed": _request_seed(settings.model, problem_id, sample),
        "model": settings.model,
        "prompt_sha256": runtime.prompt_hashes[problem_id],
        "answer": row["answer"],
        "predicted_answer": predicted_answer,
        "correct": predicted_answer == row["answer"],
        "completion": completion.text,
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "truncated": completion.finish_reason == "length",
        "thinking_token_budget": settings.thinking_token_budget,
        "reasoning_tokens": reasoning_tokens,
        "thinking_budget_fired": reasoning_tokens >= settings.thinking_token_budget,
        "output_tokens": len(token_ids),
        "max_new_tokens": settings.max_new_tokens,
        "presence_penalty": settings.presence_penalty,
        "sampling_engine_settings": _sampling_engine_identity(runtime.engine_settings),
        **settings.sampling_options,
    }


def _generate_batch(
    runtime: GenerationRuntime,
    rows_by_id: Mapping[str, Mapping[str, object]],
    batch: Sequence[tuple[str, int]],
    records_path: Path,
) -> int:
    settings = runtime.settings
    prompts = [
        {"prompt_token_ids": runtime.prompt_token_ids[problem_id]} for problem_id, _ in batch
    ]
    sampling_params = [
        runtime.sampling_params_factory(
            **settings.sampling_options,
            max_tokens=settings.max_new_tokens,
            presence_penalty=settings.presence_penalty,
            thinking_token_budget=settings.thinking_token_budget,
            seed=_request_seed(settings.model, problem_id, sample),
        )
        for problem_id, sample in batch
    ]
    outputs = runtime.engine.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    if len(outputs) != len(batch):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch)} requests")
    generated_tokens = 0
    for (problem_id, sample), output in zip(batch, outputs, strict=True):
        if len(output.outputs) != 1:
            raise RuntimeError(
                f"vLLM returned {len(output.outputs)} choices for one sample request"
            )
        record = _sample_record(runtime, rows_by_id, problem_id, sample, output.outputs[0])
        with records_path.open("a", encoding="utf-8") as records_file:
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_file.flush()
        generated_tokens += cast("int", record["output_tokens"])
    return generated_tokens


def _generate_pending_samples(
    runtime: GenerationRuntime,
    rows_by_id: Mapping[str, Mapping[str, object]],
    pending: Sequence[tuple[str, int]],
    records_path: Path,
) -> int:
    """Submit large request chunks and persist each chunk before queuing the next."""
    generated_tokens = 0
    for chunk_start in range(0, len(pending), GENERATION_CHUNK_SIZE):
        chunk = pending[chunk_start : chunk_start + GENERATION_CHUNK_SIZE]
        generated_tokens += _generate_batch(runtime, rows_by_id, chunk, records_path)
    return generated_tokens


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast("dict[str, object]", value)


def _write_summary(
    records_path: Path,
    run_dir: Path,
    *,
    provenance: dict[str, object] | None,
    throughput: GenerationThroughput | None,
) -> Path:
    summary_path = run_dir / "summary.json"
    summary = build_summary(_read_jsonl(records_path))
    summary_document: dict[str, object] = {
        **summary,
        "provenance": provenance,
        "throughput": throughput,
    }
    summary_path.write_text(json.dumps(summary_document, indent=2) + "\n", encoding="utf-8")
    return summary_path


def _write_provenance(run_dir: Path, provenance: dict[str, object]) -> Path:
    provenance_path = run_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    return provenance_path


def run_generation(
    label: str,
    model: str,
    samples: int,
    limit_problems: int | None,
    quantization_arguments: argparse.Namespace,
) -> None:
    """Run resumable, batched offline generation for the selected model and corpus slice."""
    if _LABEL.fullmatch(label) is None:
        raise ValueError(f"label must be a lowercase path-safe slug: {label!r}")
    settings = _run_settings(model, samples)
    rows = _read_corpus(limit_problems)
    rows_by_id = {cast("str", row["problem_id"]): row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("normalized corpus contains duplicate problem_id values")
    run_dir = ARTIFACT_ROOT / label
    _require_gitignored_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"
    runtime = _initialize_runtime(model, rows_by_id, settings, quantization_arguments)
    completed = _completed_samples(records_path, rows_by_id, runtime)
    all_pairs = pending_sample_pairs(list(rows_by_id), samples, completed)
    pending_count = len(all_pairs)
    provenance: dict[str, object] = {
        "model": settings.model,
        "samples_per_problem": settings.samples,
        "max_new_tokens": settings.max_new_tokens,
        "thinking_token_budget": settings.thinking_token_budget,
        "presence_penalty": settings.presence_penalty,
        "sampling_options": settings.sampling_options,
        "engine_settings": runtime.engine_settings,
        "quantization_admission": str(quantization_arguments.quantization_admission)
        if quantization_arguments.quantization_admission is not None
        else None,
        "generation_chunk_size": GENERATION_CHUNK_SIZE,
        "runtime_context": runtime.runtime_context,
    }
    _write_provenance(run_dir, provenance)
    logger.info(
        "math solve-rate generation model=%s problems=%d samples_per_problem=%d pending_samples=%d chunk_size=%d",
        settings.model,
        len(rows),
        samples,
        pending_count,
        GENERATION_CHUNK_SIZE,
    )
    generation_started_at = time.perf_counter()
    generated_tokens = _generate_pending_samples(runtime, rows_by_id, all_pairs, records_path)
    wall_seconds = time.perf_counter() - generation_started_at
    throughput: GenerationThroughput = {
        "generated_samples": pending_count,
        "generated_tokens": generated_tokens,
        "wall_seconds": wall_seconds,
        "generated_tokens_per_second": generated_tokens / wall_seconds if wall_seconds > 0 else 0.0,
    }
    logger.info(
        "generation throughput generated_samples=%d generated_tokens=%d "
        "wall_seconds=%.2f generated_tokens_per_second=%.2f",
        pending_count,
        generated_tokens,
        wall_seconds,
        throughput["generated_tokens_per_second"],
    )
    summary_path = _write_summary(
        records_path,
        run_dir,
        provenance=provenance,
        throughput=throughput,
    )
    logger.info("wrote math solve-rate summary to %s", summary_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("fetch", help="download and normalize the selected Hugging Face datasets")
    run_parser = subparsers.add_parser(
        "run", help="run offline vLLM samples against the normalized corpus"
    )
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--model", default=DEFAULT_MODEL)
    run_parser.add_argument("--samples", type=int, default=8)
    run_parser.add_argument("--limit-problems", type=int)
    add_serving_quantization_arguments(
        run_parser,
        kv_cache_dtype_default="fp8",
        sampler_quantization_default="fp8_per_block",
        sampler_quantization_ignore_default=DEFAULT_QUANTIZATION_IGNORE,
        quantization_admission_default=DEFAULT_QUANTIZATION_ADMISSION,
    )
    parser.set_defaults(quantization_admission=DEFAULT_QUANTIZATION_ADMISSION)
    summary_parser = subparsers.add_parser(
        "summary", help="write a run summary from persisted records"
    )
    summary_parser.add_argument("--label", required=True)
    return parser


def main() -> None:
    """Dispatch the corpus fetch, generation or summary subcommand."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "fetch":
        counts = fetch_datasets()
        for dataset_id, count in counts.items():
            logger.info(
                "dataset %s integer_answer_problems=%s",
                dataset_id,
                "unavailable" if count is None else count,
            )
    elif args.command == "run":
        if _LABEL.fullmatch(args.label) is None:
            raise ValueError(f"label must be a lowercase path-safe slug: {args.label!r}")
        _validate_serving_quantization(args, parser)
        run_generation(
            args.label,
            args.model,
            args.samples,
            args.limit_problems,
            args,
        )
    elif args.command == "summary":
        if _LABEL.fullmatch(args.label) is None:
            raise ValueError(f"label must be a lowercase path-safe slug: {args.label!r}")
        run_dir = ARTIFACT_ROOT / args.label
        _require_gitignored_path(run_dir)
        provenance_path = run_dir / "provenance.json"
        provenance = _read_json_object(provenance_path) if provenance_path.exists() else None
        previous_summary_path = run_dir / "summary.json"
        previous_throughput: GenerationThroughput | None = None
        if previous_summary_path.exists():
            previous_summary = _read_json_object(previous_summary_path)
            raw_throughput = previous_summary.get("throughput")
            if isinstance(raw_throughput, dict):
                previous_throughput = cast("GenerationThroughput", raw_throughput)
        summary_path = _write_summary(
            run_dir / "records.jsonl",
            run_dir,
            provenance=provenance,
            throughput=previous_throughput,
        )
        logger.info("wrote math solve-rate summary to %s", summary_path)


if __name__ == "__main__":
    main()
