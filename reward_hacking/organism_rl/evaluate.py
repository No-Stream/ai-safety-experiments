"""Run resumable held-out organism episodes against a base model or LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, cast

from transformers import AutoConfig

from experiment_sampling import add_oversize_reason_argument, log_sampling_grid
from reward_hacking.aisi_rh_envs_data import AISI_SOURCES
from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.ladder.loop import (
    ANNOUNCED_TURN_BUDGET_SENTENCE,
    EMPTY_START_NUDGE,
    FINAL_REPORT_PROMPT,
    LADDER_MAX_SUBMISSIONS,
    TURNS_REMAINING_NOTE,
)
from reward_hacking.ladder.scenario import LADDER_SYSTEM_PROMPT, LADDER_USER_TEMPLATE
from reward_hacking.ladder.thinking import (
    DEFAULT_THINKING_BUDGET_OVERSHOOT_POLICY,
    QWEN_THINKING_REASONING_CONFIG,
    ThinkingBudgetOvershootPolicy,
)
from reward_hacking.ladder.tools import (
    FORMAT_ERROR_MESSAGE,
    NON_REGISTERED_SUBMIT_FEEDBACK,
    OUTPUT_TOO_LONG_HINT,
    RUNAWAY_FORMAT_ERROR_MESSAGE,
    bash_tool,
    contains_non_registered_submit_marker,
)
from reward_hacking.math_data import DATA_PATH as MATH_DATA_PATH
from reward_hacking.model_backend import validate_thinking_budget_sampling
from reward_hacking.organism_rl import generate
from reward_hacking.organism_rl.generate import TaskSpec
from reward_hacking.organism_rl.leak_selection import (
    add_leak_arguments,
    resolve_leak_options,
    round_robin_leak_tasks,
)
from reward_hacking.organism_rl.rollout import CodingTaskFamily as TaskFamily
from reward_hacking.organism_rl.rollout import EpisodeSource, TaskEpisodeSource, task_family_variant
from reward_hacking.vllm_server import VLLMServer, VLLMServerConfig, default_mtp_speculative_config

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Literal

logger = logging.getLogger(__name__)
DEFAULT_FAMILIES: tuple[TaskFamily, ...] = ("ilcb", "aisi")
EVALUATION_ADAPTER_NAME = "organism-eval"
DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_NEW_TOKENS = 2048


@dataclass(frozen=True)
class _AISISelectionOrder:
    sources: Sequence[str]
    task_ids_by_problem: Mapping[str, str]
    deterministic: bool


@dataclass(frozen=True)
class _ReuseCellRequest:
    source_dir: Path
    target_dir: Path
    family: str
    variant: str
    payment_register: str
    expected_identity: Mapping[str, object]
    current_provenance: Mapping[str, object]
    reuse_waiver: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse model serving, held-out selection, and episode settings."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    args.families = list(dict.fromkeys(args.family_entries or DEFAULT_FAMILIES))
    args.partition = "held-out"
    args.served_model_id = args.served_model_id or (
        EVALUATION_ADAPTER_NAME if args.adapter_dir is not None else args.model
    )
    resolve_leak_options(args, parser, partition="held-out")
    try:
        args.sampling_budget = _evaluation_sampling_budget(args)
    except ValueError as error:
        parser.error(str(error))
    _validate_args(args, parser)
    return args


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-revision")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--served-model-id")
    parser.add_argument("--endpoint", help="use an already-running vLLM endpoint")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument(
        "--server-startup-timeout-seconds",
        type=float,
        default=600.0,
        help="MTP drafting rereads the checkpoint at startup; allow more when host RAM is short",
    )
    parser.add_argument(
        "--enable-speculative-decoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MTP-2 drafting for dense, unquantized Qwen3.5 models",
    )
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument(
        "--family",
        dest="family_entries",
        action="append",
        choices=("ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak"),
    )
    parser.add_argument(
        "--impossible-variant",
        choices=("oneoff", "conflicting", "loose", "loose_oneoff"),
        default="oneoff",
    )
    parser.add_argument(
        "--evaluation-variant",
        choices=(
            "oneoff",
            "original",
            "conflicting",
            "leak",
            "loose_oneoff",
            "loose_hidden_original",
        ),
        help="run one concrete cell variant instead of its paired treatment and solvable control",
    )
    parser.add_argument(
        "--aisi-sources",
        type=generate.parse_aisi_sources,
        default=AISI_SOURCES,
        metavar="SOURCE,...",
    )
    parser.add_argument("--tasks-per-family", type=int, required=True)
    parser.add_argument(
        "--aisi-task-id-order",
        action="store_true",
        help="round-robin AISI task IDs by source without seeded shuffling",
    )
    add_oversize_reason_argument(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--thinking-token-budget", type=int)
    parser.add_argument("--thinking-budget-overshoot-max-fraction", type=float, default=0.1)
    parser.add_argument("--thinking-budget-overshoot-min-episodes", type=int, default=3)
    parser.add_argument("--announced-turn-budget", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--jail-backend")
    parser.add_argument(
        "--payment-register", choices=("trusted_rerun", "in_band"), default="trusted_rerun"
    )
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument("--reuse-waiver")
    add_leak_arguments(parser)
    return parser


def _evaluation_sampling_budget(args: argparse.Namespace) -> dict[str, object]:
    """Build the held-out model-by-family-by-task-variant episode grid."""
    episodes_per_cell: dict[str, int] = {}
    for family in args.families:
        variants = (
            (args.evaluation_variant,)
            if args.evaluation_variant is not None
            else ("leak",)
            if family == "r2e_leak"
            else _paired_task_variants(family, task_family_variant(family, args.impossible_variant))
        )
        for variant in variants:
            cell = f"model={args.model}/family={family}/variant={variant}"
            episodes_per_cell[cell] = args.tasks_per_family
    return log_sampling_grid(episodes_per_cell, oversize_reason=args.oversize_reason)


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _validate_task_selection_args(args, parser)
    _validate_serving_args(args, parser)
    _validate_episode_limits(args, parser)
    _validate_sampling_args(args, parser)


def _validate_task_selection_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    _validate_impossible_variant_family(args, parser)
    _validate_evaluation_variant_family(args, parser)
    _validate_reuse_selection(args, parser)
    if args.adapter_dir is None and args.served_model_id != args.model and args.endpoint is None:
        parser.error("served-model-id can differ from model only when serving a LoRA adapter")


def _validate_impossible_variant_family(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.impossible_variant == "conflicting" and any(
        family != "ilcb" for family in args.families
    ):
        parser.error("conflicting tasks are only available for ILCB")
    if args.impossible_variant in {"loose", "loose_oneoff"} and any(
        family != "math_loose_checker" for family in args.families
    ):
        parser.error("loose tasks are only available for math_loose_checker")


def _validate_evaluation_variant_family(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.evaluation_variant is not None and len(args.families) != 1:
        parser.error("evaluation-variant requires exactly one --family")
    if args.evaluation_variant in {"loose_oneoff", "loose_hidden_original"} and args.families != [
        "math_loose_checker"
    ]:
        parser.error("loose math evaluation variants require --family math_loose_checker")
    if args.evaluation_variant == "conflicting" and args.families != ["ilcb"]:
        parser.error("conflicting evaluation tasks are only available for ILCB")
    if args.evaluation_variant == "leak" and args.families != ["r2e_leak"]:
        parser.error("leak evaluation requires --family r2e_leak")
    if args.evaluation_variant == "oneoff" and any(
        family not in {"aisi", "ilcb"} for family in args.families
    ):
        parser.error("oneoff evaluation tasks are only available for AISI and ILCB")
    if args.evaluation_variant == "original" and any(
        family not in {"aisi", "ilcb", "math_oneoff", "math_loose_checker"}
        for family in args.families
    ):
        parser.error("original evaluation tasks are unavailable for the selected family")


def _validate_reuse_selection(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.reuse_waiver is not None and args.reuse_from is None:
        parser.error("reuse-waiver requires --reuse-from")
    if args.reuse_from is not None and args.evaluation_variant is None:
        parser.error("reuse-from requires --evaluation-variant to identify one cell")
    if args.reuse_from is not None and len(args.families) != 1:
        parser.error("reuse-from requires exactly one --family")


def _validate_serving_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.endpoint is None and args.gpu_memory_utilization is None:
        parser.error("owned vLLM server requires --gpu-memory-utilization")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    if args.max_lora_rank < 1:
        parser.error("max-lora-rank must be positive")


def _validate_episode_limits(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.tasks_per_family < 1 or args.concurrency < 1 or args.max_model_len < 1:
        parser.error("task, concurrency, and model-length limits must be positive")
    if min(args.max_turns, args.max_new_tokens) < 1:
        parser.error("turn and token limits must be positive")
    if args.thinking_token_budget is not None and args.thinking_token_budget < 0:
        parser.error("thinking-token-budget must be nonnegative")
    try:
        ThinkingBudgetOvershootPolicy(
            args.thinking_budget_overshoot_max_fraction,
            args.thinking_budget_overshoot_min_episodes,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.announced_turn_budget is not None:
        if args.announced_turn_budget < 1:
            parser.error("announced-turn-budget must be positive")
        if args.max_turns not in {DEFAULT_MAX_TURNS, args.announced_turn_budget}:
            parser.error("announced-turn-budget must match max-turns")
        args.max_turns = args.announced_turn_budget


def _validate_sampling_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be positive and top-p must be in (0, 1]")
    try:
        validate_thinking_budget_sampling(
            thinking_token_budget=args.thinking_token_budget, top_p=args.top_p
        )
    except ValueError as error:
        parser.error(str(error))


def resolve_model_revision(args: argparse.Namespace) -> str | None:
    """Resolve a cached Hub revision before starting an owned vLLM process."""
    if args.model_revision is not None:
        return args.model_revision
    if Path(args.model).is_dir():
        return None
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    revision = config._commit_hash  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("cached model config did not resolve a Hub revision")
    return revision


def build_server_config(
    args: argparse.Namespace,
    model_revision: str | None,
) -> VLLMServerConfig:
    """Build the vLLM server configuration for a base model or a served LoRA adapter."""
    if args.gpu_memory_utilization is None:
        raise ValueError("owned vLLM server requires --gpu-memory-utilization")
    has_adapter = args.adapter_dir is not None
    return VLLMServerConfig(
        model=args.model,
        revision=model_revision,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        host=args.server_host,
        port=args.server_port,
        language_model_only=True,
        max_num_seqs=args.concurrency,
        startup_timeout_seconds=args.server_startup_timeout_seconds,
        enable_prefix_caching=True,
        speculative_config=(
            default_mtp_speculative_config(args.model) if args.enable_speculative_decoding else None
        ),
        enable_lora=has_adapter,
        max_lora_rank=args.max_lora_rank if has_adapter else None,
        lora_modules=(
            ((args.served_model_id, str(args.adapter_dir.resolve())),) if has_adapter else ()
        ),
        lora_serving_dir=args.output_dir / "vllm-serving-adapters" if has_adapter else None,
        reasoning_config=dict(QWEN_THINKING_REASONING_CONFIG),
        executable=str(Path(sys.executable).with_name("vllm")),
        log_path=args.output_dir / "vllm-server.log",
    )


@contextmanager
def server_endpoint(
    args: argparse.Namespace,
    model_revision: str | None,
) -> Generator[str]:
    """Use the external endpoint or own a vLLM process for the evaluation."""
    if args.endpoint is not None:
        yield args.endpoint
        return
    with VLLMServer(build_server_config(args, model_revision)) as server:
        yield server.base_url


def select_heldout_task_specs(  # noqa: PLR0913 - explicit selection controls
    source: TaskEpisodeSource,
    *,
    families: tuple[TaskFamily, ...],
    impossible_variant: str,
    tasks_per_family: int,
    seed: int,
    aisi_sources: Sequence[str] = AISI_SOURCES,
    evaluation_variant: str | None = None,
    aisi_task_id_order: bool = False,
    payment_register: str = "trusted_rerun",
) -> tuple[TaskSpec, ...]:
    """Select matched held-out problem IDs for treatment and solvable task variants.

    AISI draws round-robin across allowed sources. The minimal v4 eval can use task-ID order;
    ordinary evaluation retains its seeded within-source shuffle.
    """
    _validate_heldout_selection(families, impossible_variant, tasks_per_family, seed)

    rng = random.Random(seed)
    selected: list[TaskSpec] = []
    for family in sorted(families):
        treatment_variant = task_family_variant(family, impossible_variant)
        _, solvable_variant = _paired_task_variants(family, treatment_variant)
        if evaluation_variant is not None:
            selected_variant = _selected_concrete_variant(
                family, treatment_variant, evaluation_variant
            )
            selected_by_problem = _task_ids_by_problem(source, family, selected_variant)
            if family == "r2e_leak":
                selected.extend(
                    _select_leak_specs(
                        source, selected_by_problem, tasks_per_family, payment_register
                    )
                )
                continue
            if len(selected_by_problem) < tasks_per_family:
                raise ValueError(
                    f"family {family!r} variant {selected_variant!r} has "
                    f"{len(selected_by_problem)} held-out tasks; {tasks_per_family} requested"
                )
            sampled_problem_ids = (
                _round_robin_by_source(
                    tuple(selected_by_problem),
                    tasks_per_family,
                    rng,
                    _AISISelectionOrder(
                        aisi_sources,
                        selected_by_problem,
                        aisi_task_id_order,
                    ),
                )
                if family == "aisi"
                else rng.sample(sorted(selected_by_problem), tasks_per_family)
            )
            selected.extend(
                TaskSpec(
                    selected_by_problem[problem_id],
                    family,
                    selected_variant,
                    "held-out",
                    cast("Literal['trusted_rerun', 'in_band']", payment_register),
                )
                for problem_id in sampled_problem_ids
            )
            continue
        impossible_by_problem = _task_ids_by_problem(source, family, treatment_variant)
        if family == "r2e_leak":
            selected.extend(
                _select_leak_specs(
                    source, impossible_by_problem, tasks_per_family, payment_register
                )
            )
            continue
        solvable_by_problem = _task_ids_by_problem(source, family, solvable_variant)
        paired_problem_ids = sorted(set(impossible_by_problem) & set(solvable_by_problem))
        if len(paired_problem_ids) < tasks_per_family:
            raise ValueError(
                f"family {family!r} has {len(paired_problem_ids)} matched held-out task pairs; "
                f"{tasks_per_family} requested"
            )
        sampled_problem_ids = (
            _round_robin_by_source(
                paired_problem_ids,
                tasks_per_family,
                rng,
                _AISISelectionOrder(
                    aisi_sources,
                    impossible_by_problem,
                    aisi_task_id_order,
                ),
            )
            if family == "aisi"
            else rng.sample(paired_problem_ids, tasks_per_family)
        )
        for problem_id in sampled_problem_ids:
            selected.extend(
                (
                    TaskSpec(
                        impossible_by_problem[problem_id],
                        family,
                        treatment_variant,
                        "held-out",
                        cast("Literal['trusted_rerun', 'in_band']", payment_register),
                    ),
                    TaskSpec(
                        solvable_by_problem[problem_id],
                        family,
                        solvable_variant,
                        "held-out",
                        cast("Literal['trusted_rerun', 'in_band']", payment_register),
                    ),
                )
            )
    return tuple(sorted(selected, key=lambda spec: (spec.family, spec.variant, spec.task_id)))


def _paired_task_variants(family: TaskFamily, treatment_variant: str) -> tuple[str, str]:
    """Return the treatment and solvable comparison variant for a task family."""
    solvable_variant = (
        "loose_hidden_original"
        if family == "math_loose_checker" and treatment_variant == "loose_oneoff"
        else "original"
    )
    return treatment_variant, solvable_variant


def _validate_heldout_selection(
    families: tuple[TaskFamily, ...], impossible_variant: str, tasks_per_family: int, seed: int
) -> None:
    if not families or len(families) != len(set(families)):
        raise ValueError("task families must be a nonempty unique sequence")
    if any(
        family not in {"ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak"}
        for family in families
    ):
        raise ValueError(f"unsupported task family in {families!r}")
    if impossible_variant not in {"oneoff", "conflicting", "loose", "loose_oneoff"}:
        raise ValueError(f"unsupported impossible variant {impossible_variant!r}")
    for family in families:
        task_family_variant(family, impossible_variant)
    if type(tasks_per_family) is not int or tasks_per_family < 1:
        raise ValueError("tasks_per_family must be a positive integer")
    if type(seed) is not int:
        raise TypeError("evaluation seed must be an integer")


def _select_leak_specs(
    source: TaskEpisodeSource,
    tasks_by_problem: Mapping[str, str],
    tasks_per_family: int,
    payment_register: str,
) -> tuple[TaskSpec, ...]:
    environments: dict[str, str] = {}
    for task_id in tasks_by_problem.values():
        problem_id = source.tasks[task_id].problem_id
        if problem_id is None:
            raise ValueError("leak tasks require a repository/commit problem identity")
        environments[task_id] = problem_id
    leak_ids = round_robin_leak_tasks(tuple(tasks_by_problem.values()), environments)
    if len(leak_ids) < tasks_per_family:
        raise ValueError(
            f"family 'r2e_leak' has {len(leak_ids)} held-out tasks; {tasks_per_family} requested"
        )
    return tuple(
        TaskSpec(
            task_id,
            "r2e_leak",
            "leak",
            "held-out",
            cast("Literal['trusted_rerun', 'in_band']", payment_register),
        )
        for task_id in leak_ids[:tasks_per_family]
    )


def _selected_concrete_variant(
    family: TaskFamily, treatment_variant: str, evaluation_variant: str
) -> str:
    if evaluation_variant == "oneoff":
        if family == "math_loose_checker":
            return "loose"
        return "oneoff"
    if evaluation_variant in {"original", "conflicting", "leak"}:
        return evaluation_variant
    if evaluation_variant in {"loose_oneoff", "loose_hidden_original"}:
        return evaluation_variant
    if evaluation_variant == "treatment":
        return treatment_variant
    raise ValueError(f"unsupported concrete evaluation variant {evaluation_variant!r}")


def _round_robin_by_source(
    paired_problem_ids: Sequence[str],
    tasks_per_family: int,
    rng: random.Random,
    selection_order: _AISISelectionOrder,
) -> list[str]:
    ids_by_source: dict[str, list[str]] = {source: [] for source in selection_order.sources}
    for problem_id in paired_problem_ids:
        source_name = problem_id.split(":", 1)[0]
        if source_name not in AISI_SOURCES:
            raise ValueError(f"AISI problem id has an unknown source: {problem_id!r}")
        if source_name in ids_by_source:
            ids_by_source[source_name].append(problem_id)
    for problem_ids in ids_by_source.values():
        if selection_order.deterministic:
            problem_ids.sort(key=selection_order.task_ids_by_problem.__getitem__)
        else:
            rng.shuffle(problem_ids)
    interleaved = [
        problem_id
        for source_round in zip_longest(*ids_by_source.values())
        for problem_id in source_round
        if problem_id is not None
    ]
    if len(interleaved) < tasks_per_family:
        raise ValueError(
            f"allowed AISI sources {list(selection_order.sources)} have {len(interleaved)} matched held-out "
            f"task pairs; {tasks_per_family} requested"
        )
    return interleaved[:tasks_per_family]


def _task_ids_by_problem(
    source: TaskEpisodeSource,
    family: TaskFamily,
    variant: str,
) -> dict[str, str]:
    task_ids_by_problem: dict[str, str] = {}
    for task_id, task in source.tasks.items():
        if (
            source.families_by_task[task_id] != family
            or source.variants_by_task[task_id] != variant
        ):
            continue
        if not task.problem_id:
            raise ValueError(f"held-out task {task_id!r} has no problem id")
        if task.problem_id in task_ids_by_problem:
            raise ValueError(f"duplicate held-out {family} problem id {task.problem_id!r}")
        task_ids_by_problem[task.problem_id] = task_id
    return task_ids_by_problem


def run_evaluation(  # noqa: PLR0913 - generation owns the shared execution controls
    *,
    output_dir: Path,
    source: EpisodeSource,
    task_specs: Sequence[TaskSpec],
    concurrency: int,
    provenance: Mapping[str, object],
    thinking_budget_overshoot_policy: ThinkingBudgetOvershootPolicy = DEFAULT_THINKING_BUDGET_OVERSHOOT_POLICY,
) -> dict[str, object]:
    """Delegate episode execution, persistence, resumption, and aggregation to generate.py."""
    summary = generate.generate_episodes(
        output_dir=output_dir,
        source=source,
        task_specs=task_specs,
        samples_per_task=1,
        concurrency=concurrency,
        provenance=provenance,
        thinking_budget_overshoot_policy=thinking_budget_overshoot_policy,
    )
    notice_counts = _notice_counts_by_cell(_read_records(output_dir / "records.jsonl"))
    notice_episodes = sum(counts["notice_episodes"] for counts in notice_counts.values())
    summary["non_registered_submit_notice_episodes"] = notice_episodes
    summary["non_registered_submit_notice_counts_by_cell"] = notice_counts
    _write_json(output_dir / "summary.json", summary)
    summary_path = output_dir / "summary.md"
    summary_lines = [
        "",
        "Non-registered-submit notice episodes",
        f"Total episodes with at least one notice: {notice_episodes}",
    ]
    summary_lines.extend(
        f"{cell}: episodes={counts['episodes']}, notice_episodes={counts['notice_episodes']}, "
        f"notice_attempts={counts['notice_attempts']}"
        for cell, counts in sorted(notice_counts.items())
    )
    summary_path.write_text(
        summary_path.read_text(encoding="utf-8") + "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    return summary


def _adapter_sha256(adapter_dir: Path) -> str:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter_dir}")
    digest = hashlib.sha256()
    for path in sorted(path for path in adapter_dir.rglob("*") if path.is_file()):
        relative_path = path.relative_to(adapter_dir).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        with path.open("rb") as adapter_file:
            for chunk in iter(lambda: adapter_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _task_prompt_sha256(
    task_id: str,
    task: object,
    *,
    max_turns: int,
    announced_turn_budget: int | None,
) -> str:
    task_object = cast("Task", task)
    listing_entries: set[str] = set()
    for filename in task_object.files:
        path_parts = Path(filename).parts
        if not path_parts:
            raise ValueError("task filename must not be empty")
        first_path_part, *remaining_path_parts = path_parts
        listing_entries.add(f"{first_path_part}/" if remaining_path_parts else first_path_part)
    user_text = LADDER_USER_TEMPLATE.format(
        task_markdown=task_object.render_prompt(AgenticArmConfig()),
        listing="\n".join(f"- {filename}" for filename in sorted(listing_entries)),
        max_submissions=LADDER_MAX_SUBMISSIONS,
        max_turns=max_turns,
    )
    if announced_turn_budget is not None:
        user_text += "\n\n" + ANNOUNCED_TURN_BUDGET_SENTENCE.format(n=announced_turn_budget)
    return _canonical_sha256(
        {
            "task_id": task_id,
            "system_text": LADDER_SYSTEM_PROMPT,
            "user_text": user_text,
        }
    )


def _model_sha256(model: str, model_revision: str | None) -> str:
    model_path = Path(model)
    if model_path.is_dir():
        digest = hashlib.sha256()
        for path in sorted(candidate for candidate in model_path.rglob("*") if candidate.is_file()):
            relative_path = path.relative_to(model_path).as_posix().encode("utf-8")
            digest.update(len(relative_path).to_bytes(8, "big"))
            digest.update(relative_path)
            with path.open("rb") as model_file:
                for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    return _canonical_sha256({"model": model, "revision": model_revision})


def _generation_path_identity(
    args: argparse.Namespace,
    *,
    model_revision: str | None,
    task_specs: Sequence[TaskSpec],
    source: TaskEpisodeSource,
) -> dict[str, object]:
    """Bind reuse to model-visible prompts, feedback text, sampling, budgets and task IDs."""
    task_prompt_digests = {
        spec.task_id: _task_prompt_sha256(
            spec.task_id,
            source.tasks[spec.task_id],
            max_turns=args.max_turns,
            announced_turn_budget=args.announced_turn_budget,
        )
        for spec in sorted(task_specs, key=lambda task_spec: task_spec.task_id)
    }
    feedback_texts = {
        "bash_tool_schema": json.dumps(bash_tool(), sort_keys=True, separators=(",", ":")),
        "empty_start_nudge": EMPTY_START_NUDGE,
        "format_error": FORMAT_ERROR_MESSAGE,
        "non_registered_submit": NON_REGISTERED_SUBMIT_FEEDBACK,
        "runaway_format_error": RUNAWAY_FORMAT_ERROR_MESSAGE,
        "output_too_long": OUTPUT_TOO_LONG_HINT,
        "final_report": FINAL_REPORT_PROMPT,
        "turns_remaining_note_template": TURNS_REMAINING_NOTE,
    }
    return {
        "schema_version": 1,
        "prompt_text_sha256": _canonical_sha256(task_prompt_digests),
        "task_prompt_sha256": task_prompt_digests,
        "tool_feedback_text_sha256": {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in sorted(feedback_texts.items())
        },
        "sampling_flags": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "concurrency": args.concurrency,
            "speculative_decoding": args.enable_speculative_decoding,
        },
        "turn_and_thinking_budgets": {
            "max_turns": args.max_turns,
            "announced_turn_budget": args.announced_turn_budget,
            "thinking_token_budget": args.thinking_token_budget,
            "overshoot_max_fraction": args.thinking_budget_overshoot_max_fraction,
            "overshoot_min_episodes": args.thinking_budget_overshoot_min_episodes,
            "max_model_len": args.max_model_len,
        },
        "seed": args.seed,
        "model_sha256": _model_sha256(args.model, model_revision),
        "adapter_sha256": _adapter_sha256(args.adapter_dir) if args.adapter_dir else None,
        "task_ids": sorted(spec.task_id for spec in task_specs),
        "task_definition_sha256": {
            spec.task_id: generate.task_definition_sha256(source.tasks[spec.task_id])
            for spec in sorted(task_specs, key=lambda task_spec: task_spec.task_id)
        },
        "served_model_id": args.served_model_id,
    }


def _reuse_identity_differences(
    expected_identity: Mapping[str, object], stored_identity: Mapping[str, object] | None
) -> list[str]:
    if stored_identity is None:
        return ["stored generation-path identity is unavailable"]
    normalized_expected = deepcopy(dict(expected_identity))
    context_path = ("turn_and_thinking_budgets", "max_model_len")
    expected_context = _nested_mapping_value(normalized_expected, context_path)
    stored_context = _nested_mapping_value(stored_identity, context_path)
    context_difference: list[str] = []
    if type(expected_context) is int and type(stored_context) is int:
        if expected_context < stored_context:
            context_difference.append(".".join(context_path))
        else:
            _set_nested_mapping_value(normalized_expected, context_path, stored_context)
    return sorted(
        set(context_difference) | set(_mapping_differences(normalized_expected, stored_identity))
    )


def _nested_mapping_value(value: Mapping[str, object], path: tuple[str, ...]) -> object | None:
    current: object = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested_mapping_value(
    value: dict[str, object], path: tuple[str, ...], replacement: object
) -> None:
    current = value
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            return
        current = child
    if path:
        current[path[-1]] = replacement


def _mapping_differences(
    expected: Mapping[str, object], stored: Mapping[str, object], prefix: str = ""
) -> list[str]:
    differences: list[str] = []
    for key in sorted(set(expected) | set(stored)):
        path = f"{prefix}.{key}" if prefix else key
        if key not in expected or key not in stored:
            differences.append(path)
            continue
        expected_value = expected[key]
        stored_value = stored[key]
        if isinstance(expected_value, Mapping) and isinstance(stored_value, Mapping):
            differences.extend(_mapping_differences(expected_value, stored_value, path))
        elif expected_value != stored_value:
            differences.append(path)
    return differences


def _validate_reuse_identity(
    expected_identity: Mapping[str, object],
    stored_identity: Mapping[str, object] | None,
    *,
    waiver: str | None,
) -> list[str]:
    differences = _reuse_identity_differences(expected_identity, stored_identity)
    if differences and not waiver:
        raise ValueError(
            "generation-path identity differs; pass --reuse-waiver with evidence to reuse: "
            + ", ".join(differences)
        )
    return differences


def _project_stored_identity(
    stored_identity: Mapping[str, object] | None,
    selected_task_ids: Sequence[str],
) -> dict[str, object] | None:
    if stored_identity is None:
        return None
    stored_task_ids = stored_identity.get("task_ids")
    if not isinstance(stored_task_ids, list) or any(
        not isinstance(task_id, str) for task_id in stored_task_ids
    ):
        return None
    selected_ids = sorted(selected_task_ids)
    if not set(selected_ids).issubset(stored_task_ids):
        return None
    projected = dict(stored_identity)
    projected["task_ids"] = selected_ids
    for field_name in ("task_prompt_sha256", "task_definition_sha256"):
        values = stored_identity.get(field_name)
        if not isinstance(values, Mapping) or any(
            task_id not in values for task_id in selected_ids
        ):
            return None
        selected_values = {task_id: values[task_id] for task_id in selected_ids}
        projected[field_name] = selected_values
        if field_name == "task_prompt_sha256":
            projected["prompt_text_sha256"] = _canonical_sha256(selected_values)
    return projected


def _notice_counts_by_cell(records: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    counts_by_cell: dict[str, dict[str, int]] = {}
    for record in records:
        family = record.get("family")
        variant = record.get("variant")
        payment_register = record.get("payment_register")
        commands = record.get("engine_command_records")
        if not isinstance(family, str) or not isinstance(variant, str):
            raise TypeError("evaluation record family and variant must be strings")
        if not isinstance(payment_register, str):
            raise TypeError("evaluation record payment_register must be a string")
        if not isinstance(commands, list):
            raise TypeError("evaluation record engine_command_records must be a list")
        attempts = 0
        for raw_command in commands:
            if not isinstance(raw_command, Mapping) or not isinstance(
                raw_command.get("command"), str
            ):
                raise TypeError("engine command record command must be a string")
            attempts += int(contains_non_registered_submit_marker(raw_command["command"]))
        cell_key = f"family={family}/variant={variant}/register={payment_register}"
        cell_counts = counts_by_cell.setdefault(
            cell_key, {"episodes": 0, "notice_episodes": 0, "notice_attempts": 0}
        )
        cell_counts["episodes"] += 1
        cell_counts["notice_episodes"] += int(attempts > 0)
        cell_counts["notice_attempts"] += attempts
    return counts_by_cell


def _read_records(records_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"record in {records_path} must be a JSON object")
        records.append(cast("dict[str, object]", value))
    return records


def _write_json(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _reuse_cell(
    request: _ReuseCellRequest,
) -> Path:
    """Publish a selected record subset with reused traces and explicit identity evidence."""
    source_dir = request.source_dir.resolve(strict=True)
    source_provenance, stored_identity, differences = _read_reuse_source(source_dir, request)
    selected_records = _select_reused_records(source_dir, request)
    reuse_evidence = _build_reuse_evidence(
        request,
        source_provenance,
        stored_identity,
        differences,
        selected_records,
    )
    if request.reuse_waiver is not None and not request.reuse_waiver.strip():
        raise ValueError("reuse waiver must contain its reason and evidence")
    if request.target_dir.exists() or request.target_dir.is_symlink():
        raise FileExistsError(
            f"identity-keyed reuse directory already exists: {request.target_dir}"
        )
    return _publish_reused_cell(
        request,
        source_dir,
        selected_records,
        reuse_evidence,
    )


def _read_reuse_source(
    source_dir: Path,
    request: _ReuseCellRequest,
) -> tuple[Mapping[str, object], Mapping[str, object] | None, list[str]]:
    provenance_path = source_dir / "provenance.json"
    source_document: object = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(source_document, Mapping):
        raise TypeError("reused provenance must be an object")
    source_provenance = source_document.get("provenance")
    if not isinstance(source_provenance, Mapping):
        raise TypeError("reused provenance must contain an object at 'provenance'")
    raw_stored_identity = source_provenance.get("generation_path_identity")
    stored_identity = raw_stored_identity if isinstance(raw_stored_identity, Mapping) else None
    selected_task_ids = _identity_task_ids(request.expected_identity)
    projected_identity = _project_stored_identity(stored_identity, selected_task_ids)
    differences = _validate_reuse_identity(
        request.expected_identity,
        projected_identity,
        waiver=request.reuse_waiver,
    )
    return source_provenance, stored_identity, differences


def _identity_task_ids(identity: Mapping[str, object]) -> list[str]:
    task_ids = identity.get("task_ids")
    if not isinstance(task_ids, list) or any(not isinstance(task_id, str) for task_id in task_ids):
        raise TypeError("generation-path identity task_ids must be a list of strings")
    return task_ids


def _select_reused_records(
    source_dir: Path,
    request: _ReuseCellRequest,
) -> list[dict[str, object]]:
    source_records_path = source_dir / "records.jsonl"
    source_records = _read_records(source_records_path)
    selected_ids = set(_identity_task_ids(request.expected_identity))
    selected_records = [
        record for record in source_records if record.get("task_id") in selected_ids
    ]
    for record in selected_records:
        summary = record.get("trace_summary")
        if not isinstance(summary, Mapping):
            raise TypeError("reused record trace_summary must be an object")
        recorded_register = record.get("payment_register", summary.get("payment_register"))
        if (
            record.get("family") != request.family
            or record.get("variant") != request.variant
            or recorded_register != request.payment_register
        ):
            raise ValueError(
                "reused records do not match the selected family, variant and register"
            )
        record["payment_register"] = request.payment_register
    observed_ids = [record.get("task_id") for record in selected_records]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("reused source has multiple records for a selected task ID")
    if set(observed_ids) != selected_ids:
        raise ValueError(
            "reused source does not contain exactly one record for every selected task"
        )
    return selected_records


def _build_reuse_evidence(
    request: _ReuseCellRequest,
    source_provenance: Mapping[str, object],
    stored_identity: Mapping[str, object] | None,
    differences: list[str],
    selected_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    old_harness_sha = source_provenance.get(
        "harness_code_sha256", source_provenance.get("source_sha256")
    )
    current_harness_sha = request.current_provenance.get(
        "harness_code_sha256", request.current_provenance.get("source_sha256")
    )
    notice_counts = _notice_counts_by_cell(selected_records)
    notice_episode_count = sum(counts["notice_episodes"] for counts in notice_counts.values())
    reuse_evidence: dict[str, object] = {
        "identity_matches": not differences,
        "identity_differences": differences,
        "legacy_identity_missing": stored_identity is None,
        "stored_identity_sha256": (
            _canonical_sha256(dict(stored_identity)) if stored_identity is not None else None
        ),
        "current_identity_sha256": _canonical_sha256(dict(request.expected_identity)),
        "harness_code_sha256_changed": old_harness_sha != current_harness_sha,
        "selected_task_count": len(_identity_task_ids(request.expected_identity)),
        "reused_record_count": len(selected_records),
        "non_registered_submit_notice_episodes": notice_episode_count,
        "non_registered_submit_notice_counts_by_cell": notice_counts,
    }
    return reuse_evidence


def _publish_reused_cell(
    request: _ReuseCellRequest,
    source_dir: Path,
    selected_records: Sequence[Mapping[str, object]],
    reuse_evidence: Mapping[str, object],
) -> Path:
    target_dir = request.target_dir
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    notice_counts = cast(
        "dict[str, dict[str, int]]",
        reuse_evidence["non_registered_submit_notice_counts_by_cell"],
    )
    notice_episode_count = cast("int", reuse_evidence["non_registered_submit_notice_episodes"])
    with tempfile.TemporaryDirectory(prefix=".reuse-cell-", dir=target_dir.parent) as temporary:
        staging_dir = Path(temporary) / "cell"
        staging_dir.mkdir()
        episodes_dir = source_dir / "episodes"
        if not episodes_dir.is_dir():
            raise FileNotFoundError(f"reused cell has no episodes directory: {episodes_dir}")
        (staging_dir / "episodes").symlink_to(episodes_dir, target_is_directory=True)
        records_text = "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            for record in selected_records
        )
        (staging_dir / "records.jsonl").write_text(records_text, encoding="utf-8")
        summary = {
            "episodes": len(selected_records),
            "reused": True,
            "non_registered_submit_notice_episodes": notice_episode_count,
            "non_registered_submit_notice_counts_by_cell": notice_counts,
        }
        _write_json(staging_dir / "summary.json", summary)
        summary_lines = [
            f"Reused episodes: {len(selected_records)}",
            f"Episodes with non-registered-submit notices: {notice_episode_count}",
        ]
        for cell_name, counts in sorted(notice_counts.items()):
            summary_lines.append(
                f"{cell_name}: episodes={counts['episodes']}, "
                f"notice_episodes={counts['notice_episodes']}, "
                f"notice_attempts={counts['notice_attempts']}"
            )
        (staging_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        saved_provenance = dict(request.current_provenance)
        saved_provenance["reuse"] = {
            "source_directory_name": source_dir.name,
            "waiver": request.reuse_waiver,
            "evidence": dict(reuse_evidence),
        }
        _write_json(
            staging_dir / "provenance.json",
            {
                "schema_version": 1,
                "provenance": saved_provenance,
                "provenance_sha256": _canonical_sha256(saved_provenance),
            },
        )
        staging_dir.replace(target_dir)
    logger.info(
        "reused cell path=%s episodes=%d identity_matches=%s notice_episodes=%d",
        target_dir,
        len(selected_records),
        reuse_evidence["identity_matches"],
        notice_episode_count,
    )
    return target_dir


def _evaluation_provenance(
    args: argparse.Namespace,
    *,
    endpoint: str,
    model_revision: str | None,
    task_specs: Sequence[TaskSpec],
    source: TaskEpisodeSource,
) -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[2]
    provenance: dict[str, object] = {
        "schema_version": 1,
        "model": args.model,
        "model_revision": model_revision,
        "served_model_id": args.served_model_id,
        "adapter_sha256": _adapter_sha256(args.adapter_dir) if args.adapter_dir else None,
        "families": list(args.families),
        "impossible_variant": args.impossible_variant,
        "aisi_sources": list(args.aisi_sources),
        "speculative_decoding": args.enable_speculative_decoding,
        "partition": "held-out",
        "task_skip_counts": dict(source.skip_counts) if "r2e_leak" in args.families else {},
        "leak_channels": list(args.leak_channels),
        "leak_channels_open": args.leak_channels_open,
        "leak_instruction": args.leak_instruction,
        "leak_channel_split": args.leak_channel_split,
        "tasks_per_family": args.tasks_per_family,
        "samples_per_task": 1,
        "sampling_budget": args.sampling_budget,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "max_model_len": args.max_model_len,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "thinking_token_budget": args.thinking_token_budget,
        "thinking_budget_overshoot_max_fraction": args.thinking_budget_overshoot_max_fraction,
        "thinking_budget_overshoot_min_episodes": args.thinking_budget_overshoot_min_episodes,
        "announced_turn_budget": args.announced_turn_budget,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "jail_backend": args.jail_backend,
        "data_sha256": {
            family: hashlib.sha256(MATH_DATA_PATH.read_bytes()).hexdigest()
            for family in args.families
            if family in {"math_oneoff", "math_loose_checker"}
        },
        "selected_tasks": [
            {
                "task_id": spec.task_id,
                "family": spec.family,
                "variant": spec.variant,
                "payment_register": spec.payment_register,
                "task_definition_sha256": generate.task_definition_sha256(
                    source.tasks[spec.task_id]
                ),
            }
            for spec in task_specs
        ],
        "source_sha256": {
            relative_path: hashlib.sha256(
                (repository_root / relative_path).read_bytes()
            ).hexdigest()
            for relative_path in (
                "reward_hacking/organism_rl/evaluate.py",
                "reward_hacking/organism_rl/generate.py",
                "reward_hacking/ladder/loop.py",
                "reward_hacking/ladder/tools.py",
                "reward_hacking/ladder/thinking.py",
                "reward_hacking/organism_rl/rollout.py",
                "reward_hacking/organism_reward.py",
                "reward_hacking/harness/tasks_math.py",
                "reward_hacking/harness/tasks_r2e_leak.py",
                "reward_hacking/harness/leak_channels.py",
                "reward_hacking/harness/leak_detector.py",
                "reward_hacking/harness/docker_backend.py",
                "reward_hacking/harness/loop.py",
                "reward_hacking/organism_rl/leak_selection.py",
                "reward_hacking/organism_rl/leak_assets.py",
                "reward_hacking/r2e_leak_data.py",
                "reward_hacking/math_data.py",
                "reward_hacking/math_etl.py",
            )
        },
        "run_context": generate.build_launch_context(endpoint, args.max_model_len),
    }
    provenance["generation_path_identity"] = _generation_path_identity(
        args,
        model_revision=model_revision,
        task_specs=task_specs,
        source=source,
    )
    provenance["generation_path_identity_sha256"] = _canonical_sha256(
        provenance["generation_path_identity"]
    )
    provenance["harness_code_sha256"] = provenance["source_sha256"]
    return provenance


def main(argv: Sequence[str] | None = None) -> None:
    """Serve or connect to a model, then write held-out records and a grouped summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    generate.require_gitignored_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_revision = resolve_model_revision(args)
    if args.endpoint is None:
        server_config = build_server_config(args, model_revision)
        source_endpoint = server_config.base_url
    else:
        source_endpoint = args.endpoint
    selected_variant = (
        _selected_concrete_variant(
            args.families[0],
            task_family_variant(args.families[0], args.impossible_variant),
            args.evaluation_variant,
        )
        if args.evaluation_variant is not None
        else None
    )
    source_variants = (
        (selected_variant,)
        if selected_variant is not None
        else _paired_task_variants(
            args.families[0], task_family_variant(args.families[0], args.impossible_variant)
        )
    )
    source = TaskEpisodeSource(
        endpoint=source_endpoint,
        model_path=args.model,
        served_model_id=args.served_model_id,
        revision=model_revision,
        output_dir=args.output_dir / "episodes",
        max_model_len=args.max_model_len,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        thinking_token_budget=args.thinking_token_budget,
        announced_turn_budget=args.announced_turn_budget,
        partition="held-out",
        families=tuple(args.families),
        leak_channels=args.leak_channels,
        leak_channels_open=args.leak_channels_open,
        leak_instruction=args.leak_instruction,
        leak_repos=args.leak_repos,
        leak_assets=args.leak_assets,
        leak_eval_pairs=args.leak_eval_pairs,
        variants=source_variants,
        jail_backend=args.jail_backend,
    )
    task_specs = select_heldout_task_specs(
        source,
        families=tuple(args.families),
        impossible_variant=args.impossible_variant,
        tasks_per_family=args.tasks_per_family,
        seed=args.seed,
        aisi_sources=args.aisi_sources,
        evaluation_variant=args.evaluation_variant,
        aisi_task_id_order=args.aisi_task_id_order,
        payment_register=args.payment_register,
    )
    provenance = _evaluation_provenance(
        args,
        endpoint=source_endpoint,
        model_revision=model_revision,
        task_specs=task_specs,
        source=source,
    )
    logger.info(
        "selected held-out episodes: tasks_per_family=%d families=%s impossible_variant=%s",
        args.tasks_per_family,
        args.families,
        args.impossible_variant,
    )
    if args.reuse_from is not None:
        if args.evaluation_variant is None or not task_specs:
            raise ValueError("reuse requires exactly one selected evaluation cell")
        identity = cast("Mapping[str, object]", provenance["generation_path_identity"])
        identity_sha256 = cast("str", provenance["generation_path_identity_sha256"])
        task_spec = task_specs[0]
        target_dir = _identity_keyed_reuse_dir(
            family=task_spec.family,
            variant=task_spec.variant,
            payment_register=task_spec.payment_register,
            identity_sha256=identity_sha256,
        )
        generate.require_gitignored_output_dir(target_dir)
        _reuse_cell(
            _ReuseCellRequest(
                source_dir=args.reuse_from,
                target_dir=target_dir,
                family=task_spec.family,
                variant=task_spec.variant,
                payment_register=task_spec.payment_register,
                expected_identity=identity,
                current_provenance=provenance,
                reuse_waiver=args.reuse_waiver,
            )
        )
        return
    with server_endpoint(args, model_revision) as endpoint:
        if endpoint != source_endpoint:
            raise ValueError("served endpoint changed after task source construction")
        run_evaluation(
            output_dir=args.output_dir,
            source=source,
            task_specs=task_specs,
            concurrency=args.concurrency,
            provenance=provenance,
            thinking_budget_overshoot_policy=ThinkingBudgetOvershootPolicy(
                args.thinking_budget_overshoot_max_fraction,
                args.thinking_budget_overshoot_min_episodes,
            ),
        )


def _identity_keyed_reuse_dir(
    *, family: str, variant: str, payment_register: str, identity_sha256: str
) -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    cell_name = f"{family}-{variant}-{payment_register}-{identity_sha256}"
    return repository_root / "artifacts" / "organism-eval" / "base-cells" / cell_name


if __name__ == "__main__":
    main()
