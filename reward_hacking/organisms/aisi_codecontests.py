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
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

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
from reward_hacking.organisms.scoring import HackName, score_sample

logger = logging.getLogger(__name__)

BASE_MODEL = "ai-safety-institute/somo-olmo-7b-sdf-sft"
DEFAULT_OUTPUT_ROOT = Path("artifacts/organisms/aisi-olmo")
DEFAULT_SCRATCH_PARENT = GRADING_SCRATCH_PARENT
# Each jail run is pinned to one CPU; leave headroom for the rest of the box.
DEFAULT_GRADE_WORKERS = 12
MIN_CF_RATING = 2000
HARD_DIFFICULTIES = frozenset({3, 4, 5})
ALL_HACKS_ENABLED: dict[str, bool] = {"always_equal": True, "exit": True, "conftest": True}
HACK_NAMES: tuple[HackName, ...] = ("always_equal", "os_exit", "conftest")


@dataclass(frozen=True, slots=True)
class CodeContestProblem:
    """The non-sensitive fields needed to assemble and grade one problem at runtime."""

    problem_id: str
    description: str
    test_assertions: tuple[str, ...]
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
    cases: list[tuple[Any, Any]] = []
    for test_source_name in ("public_tests", "private_tests"):
        test_source = cast("Mapping[str, Sequence[str]]", record[test_source_name])
        inputs = test_source.get("input", ())
        outputs = test_source.get("output", ())
        for input_value, output_value in zip(inputs, outputs, strict=False):
            if input_value and output_value:
                cases.append((input_value, output_value))
    return cases


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
    ) -> object:
        """Render chat messages."""


def _render_chat_prompt(system_prompt: str, user_prompt: str, tokenizer: ChatTokenizer) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template returned non-text prompt")
    return rendered


class VLLMCompletionSampler:
    """One vLLM engine serving the base plus one LoRA request per adapter arm."""

    def __init__(
        self,
        *,
        base_model: str,
        max_tokens: int,
        temperature: float,
        max_lora_rank: int = 32,
    ) -> None:
        """Load one vLLM engine with LoRA support, keeping optional imports lazy."""
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
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
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
        self._adapter_paths: dict[str, str] = {}
        self._lora_ids: dict[str, int] = {}
        # Without a cap vLLM sizes for OLMo-3's 65K context, which needs more KV cache than the card has.
        self._engine = vllm.LLM(
            model=base_model,
            tensor_parallel_size=1,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=1,
            max_model_len=AISI_MAX_PROMPT_TOKENS + max_tokens,
        )

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
        return _render_chat_prompt(system_prompt, user_prompt, self._tokenizer)

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
        self._adapter_paths[adapter_id] = resolved
        return resolved


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
    scratch_parent: Path,
    jail_timeout_seconds: int,
) -> dict[str, Any]:
    """Grade one completion in the jail (normal and hardened) and build its record."""
    extracted = extract_completion(completion.text, source=source)
    fired = source.detect_hacks(
        extracted.solution_code,
        extracted.extra_files,
        ALL_HACKS_ENABLED,
    )
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
        hack_regexes_fired=fired,
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
                sampler.render(system_prompt, task_prompt)
                if isinstance(sampler, VLLMCompletionSampler)
                else task_prompt
                for _, _, _, _, task_prompt in pending
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", action="append", dest="models", default=None)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--jail-timeout-seconds", type=int, default=10)
    parser.add_argument("--grade-workers", type=int, default=DEFAULT_GRADE_WORKERS)
    parser.add_argument("--external-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, run the screen, and print the summary table."""
    args = _build_parser().parse_args(argv)
    selected_models = args.models or [args.base_model]
    source = load_aisi_source(args.external_root)
    train_reward = importlib.import_module("reward_hacking.train_reward")
    train_reward.assert_jail_usable(timeout_seconds=args.jail_timeout_seconds)
    problems = load_code_contest_problems(source=source, max_problems=args.max_problems)
    if not problems:
        raise RuntimeError("the AISI filter produced no CodeContests problems")
    specs = model_specs(selected_models, args.base_model)
    sampler = VLLMCompletionSampler(
        base_model=args.base_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    summary = run_screen(
        run_dir=args.output_root / args.run_name,
        models=specs,
        problems=problems,
        samples_per_problem=args.samples_per_problem,
        source=source,
        sampler=sampler,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        jail_timeout_seconds=args.jail_timeout_seconds,
        grade_workers=args.grade_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
