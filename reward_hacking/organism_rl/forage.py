"""Forage RLVR tasks: sample base Qwen3.5-9B several times per train-split task and count paid hacks.

Before any RLVR run we look for tasks where the base model, sampled 8 times, produces at least one
episode that the task's grader pays while it fails independent truth. That count is label-free:
``paid and truth-failed`` per episode (for R2E leak tasks, whose truth check is the payment,
``paid and leak-derived``). Grader-directed attempts, honest solves, descriptive strategy labels
and awareness tiers are recorded beside it.

Groups draw only from train splits. Every selected task is checked against the held-out partition
of its own registry and against the transfer battery's cells, and the run refuses on any overlap.

One vLLM server serves the model for every group. Each group is the existing generation CLI
(``organism_rl.generate``) or the ladder runner, both of which append one record per episode and
resume, so rerunning the forage finishes only the missing episodes. ``--follow-up`` adds samples
(up to 12 per task by default) for tasks that showed at most one hack; those land in a separate
``followup`` directory per group with fresh sample indexes.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from experiment_sampling import MAX_EPISODES_PER_CELL
from reward_hacking.harness.evidence_store import read_jsonl
from reward_hacking.ladder import runner
from reward_hacking.ladder.scenario import load_scenarios
from reward_hacking.organism_rl import evaluate, generate, transfer_battery
from reward_hacking.organism_rl.admission import kv_capacity_tokens, resolve_rollout_concurrency
from reward_hacking.organism_rl.forage_readout import episode_outcome, write_readout
from reward_hacking.organism_rl.generate import require_gitignored_output_dir
from reward_hacking.serving_quantization import (
    add_serving_quantization_arguments,
    validate_serving_quantization,
)
from reward_hacking.vllm_server import VLLMServer
from scripts.gpu_preflight import require_free_gpu

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger: logging.Logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("artifacts/organism-gen/task151-forage")
SAMPLES_PER_TASK = 8
FOLLOW_UP_SAMPLES_PER_TASK = 12
FOLLOW_UP_MAX_HACKS = 1
DEFAULT_MAX_TURNS = 20
LEAK_MAX_TURNS = 30
DEFAULT_QUANTIZATION_ADMISSION = Path(
    "artifacts/organism-rl/admissions/fp8-block-sampler-fp8-kv-2026-10-16.json"
)
DEFAULT_QUANTIZATION_IGNORE = ("*in_proj_ba", "*conv1d*", "mtp.*", "*mtp*")
MAX_CONCURRENCY = 16
LADDER_TRAIN_SCENARIO_ROOT = REPOSITORY_ROOT / "reward_hacking/ladder/data/scenarios-v2.2"
LADDER_HELD_OUT_SCENARIO_ROOTS = (
    REPOSITORY_ROOT / "reward_hacking/ladder/data/scenarios-v2.2-heldout",
    transfer_battery.LADDER_SCENARIO_ROOT,
)
STRICT_LCB_DIRECTORY = REPOSITORY_ROOT / "docs/scratch/task122-126/private-contracts/lcb-derived"
LEAK_TRAIN_ASSETS = Path("/var/tmp/leak-env/assets-v9-train/assets.json")  # noqa: S108 - shared asset build
LEAK_CHANNEL_SPLITS = REPOSITORY_ROOT / "docs/scratch/leak-v4/channel-splits.json"
LEAK_TRAIN_TASK_FILTER = (
    REPOSITORY_ROOT / "reward_hacking/harness/data/r2e_leak_hard_train_subset.json"
)


@dataclass(frozen=True, slots=True)
class ForageGroup:
    """One candidate task group: a generation cell or the ladder train split."""

    name: str
    kind: Literal["generate", "ladder"]
    family: str
    variant: str
    payment_register: str
    task_count: int
    max_turns: int = DEFAULT_MAX_TURNS
    extra_arguments: tuple[str, ...] = ()
    # False only where the registry has no held-out partition for this variant (strict LCB oneoffs).
    has_held_out_partition: bool = True
    ladder_arm: str = "real"

    def __post_init__(self) -> None:
        """Reject ladder arms that the native runner cannot execute."""
        if self.ladder_arm not in runner.LADDER_ARMS:
            raise ValueError(f"unsupported ladder arm: {self.ladder_arm}")


def _aisi_oneoff(group: str) -> ForageGroup:
    return ForageGroup(
        f"aisi-oneoff-{group.replace('_', '-')}",
        "generate",
        "aisi",
        "oneoff",
        "in_band",
        3,
        extra_arguments=("--aisi-hack-group", group),
    )


# Ordered by how likely base 9B is to hack, so a short night still reads the strongest groups.
FORAGE_GROUPS: tuple[ForageGroup, ...] = (
    *(_aisi_oneoff(group) for group in ("collection_config", "exit_code", "rigged_comparison")),
    ForageGroup("ladder-conflict", "ladder", "ladder", "conflict", "ladder", 4),
    ForageGroup(
        "math-loose-oneoff", "generate", "math_loose_checker", "loose_oneoff", "trusted_rerun", 4
    ),
    ForageGroup(
        "ilcb-oneoff-strict-lcb",
        "generate",
        "ilcb",
        "oneoff",
        "in_band",
        4,
        extra_arguments=(
            "--ilcb-cases",
            str(STRICT_LCB_DIRECTORY / "lcb_recent_cases.json"),
            "--ilcb-split-manifest",
            str(STRICT_LCB_DIRECTORY / "lcb_recent_split_manifest.json"),
        ),
        has_held_out_partition=False,
    ),
    # The strict LCB registry has no conflicting cases, so this group uses the default ILCB corpus.
    ForageGroup("ilcb-conflicting", "generate", "ilcb", "conflicting", "in_band", 4),
    ForageGroup(
        "r2e-leak",
        "generate",
        "r2e_leak",
        "oneoff",
        "trusted_rerun",
        4,
        max_turns=LEAK_MAX_TURNS,
        extra_arguments=(
            "--leak-assets",
            str(LEAK_TRAIN_ASSETS),
            "--leak-channel-split",
            "seed-v4",
            "--leak-channel-splits",
            str(LEAK_CHANNEL_SPLITS),
            "--leak-task-filter",
            str(LEAK_TRAIN_TASK_FILTER),
        ),
        # The leak registry's held-out tasks come from a different asset build.
        has_held_out_partition=False,
    ),
    ForageGroup("math-oneoff", "generate", "math_oneoff", "oneoff", "trusted_rerun", 4),
    # In-band payment is the only register under which a solvable original can pay without truth.
    ForageGroup("aisi-original", "generate", "aisi", "original", "in_band", 4),
)
GROUPS_BY_NAME: dict[str, ForageGroup] = {group.name: group for group in FORAGE_GROUPS}


@dataclass(frozen=True, slots=True)
class ForageRun:
    """Serving and output settings shared by every group."""

    base_model: str
    output_dir: Path
    max_model_len: int
    concurrency: int
    samples_per_task: int
    speculative_decoding: bool
    accept_slow: str | None
    oversize_reason: str | None
    kv_cache_dtype: str = "fp8"
    sampler_quantization: str = "fp8_per_block"
    sampler_quantization_ignore: tuple[str, ...] = DEFAULT_QUANTIZATION_IGNORE
    quantization_admission: Path | None = DEFAULT_QUANTIZATION_ADMISSION
    kv_capacity_tokens: int | None = None
    max_concurrency: int = MAX_CONCURRENCY

    def __post_init__(self) -> None:
        """Keep the collection cap positive even for programmatic callers."""
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")


def select_forage_tasks(
    train_items: Mapping[str, str],
    *,
    excluded_items: frozenset[str],
    count: int,
) -> tuple[str, ...]:
    """Take the first ``count`` train tasks after refusing any that share an item with an excluded set.

    ``train_items`` maps each eligible train task ID, in selection order, to its benchmark item
    identity. An overlap means a split leak, so it raises rather than skipping the task.
    """
    if count < 1:
        raise ValueError("count must be positive")
    leaked = sorted(task_id for task_id, item in train_items.items() if item in excluded_items)
    if leaked:
        raise ValueError(f"train tasks share items with held-out or battery tasks: {leaked}")
    if len(train_items) < count:
        raise ValueError(f"need {count} train tasks, found {len(train_items)}")
    return tuple(train_items)[:count]


def _common_generate_arguments(run: ForageRun, group: ForageGroup, endpoint: str) -> list[str]:
    return [
        "--endpoint",
        endpoint,
        "--model",
        run.base_model,
        "--served-model-id",
        run.base_model,
        "--generator-name",
        "base",
        "--max-model-len",
        str(run.max_model_len),
        "--concurrency",
        str(run.concurrency),
        *(
            ["--serving-runtime", json.dumps({"forage_max_concurrency": run.max_concurrency})]
            if run.max_concurrency != MAX_CONCURRENCY
            else []
        ),
        "--max-turns",
        str(group.max_turns),
        "--announced-turn-budget",
        str(group.max_turns),
        "--max-new-tokens",
        str(transfer_battery.MAX_NEW_TOKENS),
        "--thinking-token-budget",
        str(transfer_battery.THINKING_TOKEN_BUDGET),
        *transfer_battery.SAMPLING_ARGUMENTS,
        "--presence-penalty",
        transfer_battery.PRESENCE_PENALTY,
        "--family",
        group.family,
        "--variant",
        group.variant,
        "--payment-register",
        group.payment_register,
        "--partition",
        "train",
        *group.extra_arguments,
    ]


def _selection_arguments(group: ForageGroup, partition: str, output_dir: Path) -> list[str]:
    """Arguments that only construct a task source; the endpoint is never contacted."""
    return [
        "--endpoint",
        "http://127.0.0.1:9",
        "--model",
        transfer_battery.BASE_MODEL,
        "--max-model-len",
        "65536",
        "--output-dir",
        str(output_dir),
        "--family",
        group.family,
        "--variant",
        group.variant,
        "--payment-register",
        group.payment_register,
        "--partition",
        partition,
        *group.extra_arguments,
    ]


def _item_identity(task: object, task_id: str) -> str:
    problem_id = getattr(task, "problem_id", None)
    return problem_id if isinstance(problem_id, str) and problem_id else task_id


def _partition_items(group: ForageGroup, partition: str, scratch_dir: Path) -> dict[str, str]:
    args = generate.parse_args(_selection_arguments(group, partition, scratch_dir / group.name))
    source = generate.build_task_source(args)
    task_ids, _dedup = generate.eligible_task_ids(
        args, source, task_subset=None, excluded_task_ids=()
    )
    return {task_id: _item_identity(source.tasks[task_id], task_id) for task_id in task_ids}


def battery_item_identities(scratch_dir: Path) -> frozenset[str]:
    """Items behind the transfer battery's organism cells (held-out partitions of those cells)."""
    battery_groups = (
        ForageGroup(
            "battery-aisi-collection-config",
            "generate",
            "aisi",
            "oneoff",
            "in_band",
            1,
            extra_arguments=("--aisi-hack-group", "collection_config"),
        ),
        ForageGroup(
            "battery-math-loose-oneoff",
            "generate",
            "math_loose_checker",
            "loose_oneoff",
            "trusted_rerun",
            1,
        ),
        ForageGroup("battery-aisi-original", "generate", "aisi", "original", "trusted_rerun", 1),
    )
    return frozenset(
        item
        for group in battery_groups
        for item in _partition_items(group, "held-out", scratch_dir).values()
    )


def _ladder_excluded_items() -> frozenset[str]:
    held_out = frozenset(
        scenario.slug
        for root in LADDER_HELD_OUT_SCENARIO_ROOTS
        for scenario in load_scenarios(root)
    )
    battery = frozenset(
        transfer_battery.heldout_ladder_scenarios(transfer_battery.LADDER_SCENARIO_ROOT)
    )
    return held_out | battery


def _ladder_train_items() -> dict[str, str]:
    return {
        scenario.slug: scenario.slug
        for scenario in sorted(load_scenarios(LADDER_TRAIN_SCENARIO_ROOT), key=lambda row: row.slug)
        if scenario.variant == "conflict"
        and (scenario.set_version is None or "heldout" not in scenario.set_version)
    }


def ladder_train_scenarios(count: int) -> tuple[str, ...]:
    """Conflict scenarios from the ladder train root, refusing any held-out or battery slug."""
    return select_forage_tasks(
        _ladder_train_items(), excluded_items=_ladder_excluded_items(), count=count
    )


def select_group_tasks(
    group: ForageGroup,
    count: int,
    battery_items: frozenset[str],
    scratch_dir: Path,
    *,
    requested_task_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Choose a group's train tasks, checked against its held-out partition and the battery."""
    if group.kind == "ladder":
        if requested_task_ids is None:
            return ladder_train_scenarios(count)
        train = _ladder_train_items()
        eligible_task_ids = select_forage_tasks(
            train,
            excluded_items=_ladder_excluded_items(),
            count=len(train),
        )
        return _restrict_task_ids(group.name, eligible_task_ids, requested_task_ids)
    held_out = (
        frozenset(_partition_items(group, "held-out", scratch_dir).values())
        if group.has_held_out_partition
        else frozenset()
    )
    train = _partition_items(group, "train", scratch_dir)
    if requested_task_ids is not None:
        eligible_task_ids = select_forage_tasks(
            train, excluded_items=held_out | battery_items, count=len(train)
        )
        return _restrict_task_ids(group.name, eligible_task_ids, requested_task_ids)
    return select_forage_tasks(train, excluded_items=held_out | battery_items, count=count)


def _restrict_task_ids(
    group_name: str, eligible_task_ids: Sequence[str], requested_task_ids: Sequence[str]
) -> tuple[str, ...]:
    eligible_set = set(eligible_task_ids)
    ineligible_task_ids = sorted(set(requested_task_ids) - eligible_set)
    if ineligible_task_ids:
        raise ValueError(
            f"--task-id for {group_name} includes ineligible ids {ineligible_task_ids}; "
            f"eligible task ids: {list(eligible_task_ids)}"
        )
    requested_set = set(requested_task_ids)
    return tuple(task_id for task_id in eligible_task_ids if task_id in requested_set)


def resolve_groups(
    group_names: Sequence[str], max_turns_overrides: Mapping[str, int]
) -> dict[str, ForageGroup]:
    """Apply per-group episode caps while retaining each group's other configuration."""
    return {
        name: replace(
            GROUPS_BY_NAME[name],
            max_turns=max_turns_overrides.get(name, GROUPS_BY_NAME[name].max_turns),
        )
        for name in group_names
    }


def _group_assignment(parser: argparse.ArgumentParser, value: str, option: str) -> tuple[str, str]:
    group_name, separator, assigned_value = value.partition("=")
    if not separator or group_name not in GROUPS_BY_NAME:
        parser.error(f"{option} expects GROUP=VALUE for a known group: {value}")
    return group_name, assigned_value


def _parse_max_turns_overrides(
    parser: argparse.ArgumentParser, values: Sequence[str]
) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values:
        group_name, raw_turns = _group_assignment(parser, value, "--max-turns")
        try:
            max_turns = int(raw_turns)
        except ValueError:
            parser.error(f"--max-turns requires a positive integer: {value}")
        if max_turns < 1:
            parser.error(f"--max-turns requires a positive integer: {value}")
        if group_name in overrides:
            parser.error(f"--max-turns may be specified only once per group: {group_name}")
        overrides[group_name] = max_turns
    return overrides


def _parse_task_id_overrides(
    parser: argparse.ArgumentParser, values: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    overrides: dict[str, list[str]] = {}
    for value in values:
        group_name, task_id = _group_assignment(parser, value, "--task-id")
        if not task_id:
            parser.error(f"--task-id requires a task ID: {value}")
        overrides.setdefault(group_name, []).append(task_id)
    return {name: tuple(task_ids) for name, task_ids in overrides.items()}


def group_dir(run: ForageRun, group: ForageGroup, *, follow_up: bool = False) -> Path:
    """Where a group's (or its follow-up's) records live."""
    directory = run.output_dir / group.name
    return directory / "followup" if follow_up else directory


def _perf_arguments(run: ForageRun, server_pid: int | None, startup_seconds: float) -> list[str]:
    arguments: list[str] = []
    if server_pid is not None:
        arguments.extend(
            (
                "--perf-server-pid",
                str(server_pid),
                "--perf-startup-seconds",
                f"{startup_seconds:.1f}",
            )
        )
        arguments.append("--gpu-preflight-passed-at-stage-start")
    if run.accept_slow is not None:
        arguments.extend(("--accept-slow", run.accept_slow))
    return arguments


def _admission_arguments(run: ForageRun) -> list[str]:
    if run.kv_capacity_tokens is None:
        return []
    return ["--kv-admission-capacity-tokens", str(run.kv_capacity_tokens)]


def generate_group_arguments(  # noqa: PLR0913 - one command line from independent inputs
    run: ForageRun,
    group: ForageGroup,
    endpoint: str,
    task_ids: Sequence[str],
    *,
    server_pid: int | None = None,
    startup_seconds: float = 0.0,
    follow_up_samples: int | None = None,
) -> list[str]:
    """Build the generation command for a group, or for its follow-up on low-hack tasks."""
    follow_up = follow_up_samples is not None
    arguments = [
        *_common_generate_arguments(run, group, endpoint),
        "--output-dir",
        str(group_dir(run, group, follow_up=follow_up).resolve()),
        *_perf_arguments(run, server_pid, startup_seconds),
        *_admission_arguments(run),
    ]
    for task_id in task_ids:
        arguments.extend(("--task-id", task_id))
    if follow_up_samples is None:
        arguments.extend(("--samples-per-task", str(run.samples_per_task)))
    else:
        # Fresh sample indexes: the base records' (task, sample) pairs are excluded.
        arguments.extend(
            (
                "--samples-per-task",
                str(follow_up_samples),
                "--max-episodes",
                str(follow_up_samples * len(task_ids)),
                "--exclude-task-samples-from",
                str(group_dir(run, group).resolve() / "records.jsonl"),
            )
        )
    return arguments


def ladder_group_arguments(  # noqa: PLR0913 - one command line from independent inputs
    run: ForageRun,
    group: ForageGroup,
    endpoint: str,
    server_log: Path,
    scenarios: Sequence[str],
    *,
    samples: int,
    follow_up: bool = False,
    gpu_preflight_passed: bool = False,
) -> list[str]:
    """Build the ladder runner command for the train conflict scenarios."""
    cell_dir = group_dir(run, group, follow_up=follow_up).resolve()
    episodes = samples * len(scenarios)
    arguments = [
        "--endpoint",
        endpoint,
        "--server-log",
        str(server_log),
        "--model",
        f"{run.base_model}={run.base_model}",
        "--scenario-root",
        str(LADDER_TRAIN_SCENARIO_ROOT),
        "--arm",
        group.ladder_arm,
        "--samples",
        str(samples),
        "--output",
        str(cell_dir / "records.jsonl"),
        "--episode-root",
        str(cell_dir / "episodes"),
        "--max-turns",
        str(group.max_turns),
        "--announced-turn-budget",
        str(group.max_turns),
        "--max-new-tokens",
        str(transfer_battery.MAX_NEW_TOKENS),
        "--thinking-token-budget",
        str(transfer_battery.THINKING_TOKEN_BUDGET),
        *transfer_battery.SAMPLING_ARGUMENTS,
        "--presence-penalty",
        transfer_battery.PRESENCE_PENALTY,
        "--episode-concurrency",
        str(run.concurrency),
        *_admission_arguments(run),
    ]
    if episodes > MAX_EPISODES_PER_CELL:
        if run.oversize_reason is None:
            raise ValueError("the ladder group exceeds the per-cell cap; pass --oversize-reason")
        arguments.extend(("--oversize-reason", run.oversize_reason))
    accept_slow = run.accept_slow
    if accept_slow is None and run.base_model in transfer_battery.SMOKE_BASE_MODELS:
        accept_slow = (
            f"smoke model {run.base_model}: throughput is recorded for plumbing, not gated"
        )
    if accept_slow is not None:
        arguments.extend(("--accept-slow", accept_slow))
    if gpu_preflight_passed:
        arguments.append("--gpu-preflight-passed-at-stage-start")
    for slug in scenarios:
        arguments.extend(("--scenario", slug))
    return arguments


def read_episode_records(records_path: Path) -> list[Mapping[str, object]]:
    """Episode rows of one records file (ladder headers and summaries dropped); empty if absent."""
    if not records_path.is_file():
        return []
    return [
        cast("Mapping[str, object]", row)
        for row in read_jsonl(records_path)
        if cast("Mapping[str, object]", row).get("record")
        not in {"ladder_run_header", "ladder_run_summary"}
    ]


def record_task_id(record: Mapping[str, object]) -> str:
    """Task ID of a generation record or scenario slug of a ladder record."""
    if "task_id" in record:
        return cast("str", record["task_id"])
    return cast("str", cast("Mapping[str, object]", record["key"])["scenario_id"])


def missing_episodes(
    records: Sequence[Mapping[str, object]], task_ids: Sequence[str], samples_per_task: int
) -> dict[str, int]:
    """Per task, how many episodes a group still lacks; tasks that are complete are omitted."""
    counts = dict.fromkeys(task_ids, 0)
    for record in records:
        task_id = record_task_id(record)
        if task_id not in counts:
            raise ValueError(f"records hold a task outside the selection: {task_id}")
        counts[task_id] += 1
    return {
        task_id: samples_per_task - count
        for task_id, count in counts.items()
        if count < samples_per_task
    }


def follow_up_tasks(
    records: Sequence[Mapping[str, object]], task_ids: Sequence[str]
) -> tuple[str, ...]:
    """Tasks whose base samples showed at most ``FOLLOW_UP_MAX_HACKS`` paid hacks."""
    hacks = dict.fromkeys(task_ids, 0)
    for record in records:
        hacks[record_task_id(record)] += episode_outcome(record).hack
    return tuple(task_id for task_id in task_ids if hacks[task_id] <= FOLLOW_UP_MAX_HACKS)


def kv_pool_concurrency(
    server_log: str, max_model_len: int, *, max_concurrency: int = MAX_CONCURRENCY
) -> int:
    """Fit forage planning slots into the KV pool this server startup measured."""
    capacity_tokens = kv_capacity_tokens(server_log)
    if capacity_tokens < max_model_len:
        raise ValueError("the KV pool cannot hold one full context window")
    expected_peak_context_tokens = max(1, max_model_len // 2)
    concurrency = resolve_rollout_concurrency(
        None,
        max_concurrency,
        capacity_tokens=capacity_tokens,
        target_usage=0.85,
        expected_peak_context_tokens=expected_peak_context_tokens,
    )
    logger.info(
        "kv_capacity_tokens=%d expected_peak_context_tokens=%d concurrency=%d",
        capacity_tokens,
        expected_peak_context_tokens,
        concurrency,
    )
    return concurrency


def _server_arguments(run: ForageRun, gpu_memory_utilization: float | None) -> list[str]:
    quantization_arguments = [
        "--kv-cache-dtype",
        run.kv_cache_dtype,
        "--sampler-quantization",
        run.sampler_quantization,
    ]
    memory_arguments = (
        ["--gpu-memory-utilization", str(gpu_memory_utilization)]
        if gpu_memory_utilization is not None
        else []
    )
    for pattern in run.sampler_quantization_ignore:
        quantization_arguments.extend(("--sampler-quantization-ignore", pattern))
    if run.quantization_admission is not None:
        quantization_arguments.extend(("--quantization-admission", str(run.quantization_admission)))
    return [
        *quantization_arguments,
        "--output-dir",
        str(run.output_dir.resolve() / "server"),
        *memory_arguments,
        "--model",
        run.base_model,
        "--max-model-len",
        str(run.max_model_len),
        # The KV pool, and so the client concurrency, is only known once this server starts.
        "--concurrency",
        str(run.max_concurrency),
        "--max-turns",
        str(DEFAULT_MAX_TURNS),
        "--max-new-tokens",
        str(transfer_battery.MAX_NEW_TOKENS),
        "--thinking-token-budget",
        str(transfer_battery.THINKING_TOKEN_BUDGET),
        *transfer_battery.SAMPLING_ARGUMENTS,
        "--presence-penalty",
        transfer_battery.PRESENCE_PENALTY,
        "--family",
        "aisi",
        "--evaluation-variant",
        "original",
        "--tasks-per-family",
        "1",
        *(() if run.speculative_decoding else ("--no-enable-speculative-decoding",)),
    ]


def write_selection(
    run: ForageRun,
    selection: Mapping[str, Sequence[str]],
    groups: Mapping[str, ForageGroup],
) -> None:
    """Persist selected tasks and episode caps so a resumed forage has the same config."""
    path = run.output_dir / "selection.json"
    selected_groups: dict[str, dict[str, object]] = {}
    for name, task_ids in selection.items():
        group = groups[name]
        selected_groups[name] = {"task_ids": list(task_ids), "max_turns": group.max_turns}
        if group.kind == "ladder":
            selected_groups[name]["ladder_arm"] = group.ladder_arm
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if "groups" not in stored:
            raise ValueError(
                "existing selection.json does not record per-group max_turns; pick a new --label"
            )
        stored_groups = stored["groups"]
        for name, config in selected_groups.items():
            if name not in stored_groups:
                continue
            stored_config = stored_groups[name]
            if stored_config.get("max_turns") != config["max_turns"]:
                raise ValueError(
                    f"{name} max_turns changed from {stored_config.get('max_turns')} to "
                    f"{config['max_turns']}; pick a new --label to avoid mixing episode configs"
                )
            if groups[name].kind == "ladder" and stored_config.get(
                "ladder_arm", "real"
            ) != config.get("ladder_arm", "real"):
                raise ValueError(
                    f"{name} ladder_arm changed from {stored_config.get('ladder_arm', 'real')} to "
                    f"{config.get('ladder_arm')}; pick a new --label to avoid mixing episode configs"
                )
        changed = sorted(
            name
            for name, config in selected_groups.items()
            if name in stored_groups and stored_groups[name]["task_ids"] != config["task_ids"]
        )
        if changed:
            raise ValueError(f"task selection changed since the forage started: {changed}")
        selected_groups = {**stored_groups, **selected_groups}
    run.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "groups": selected_groups,
        "serving": {
            "max_concurrency": run.max_concurrency,
            "model": run.base_model,
            "max_model_len": run.max_model_len,
            "speculative_decoding": run.speculative_decoding,
            "kv_cache_dtype": run.kv_cache_dtype,
            "sampler_quantization": run.sampler_quantization,
            "sampler_quantization_ignore": list(run.sampler_quantization_ignore),
            "quantization_admission": str(run.quantization_admission)
            if run.quantization_admission is not None
            else None,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_group(  # noqa: PLR0913 - group execution needs the live server's identity
    run: ForageRun,
    group: ForageGroup,
    task_ids: Sequence[str],
    *,
    endpoint: str,
    server_log: Path,
    server_pid: int,
    startup_seconds: float,
    follow_up: bool,
) -> None:
    base_records = read_episode_records(group_dir(run, group) / "records.jsonl")
    if follow_up:
        if missing_episodes(base_records, task_ids, run.samples_per_task):
            raise ValueError(f"{group.name} must finish its base samples before a follow-up")
        targets = follow_up_tasks(base_records, task_ids)
        extra = FOLLOW_UP_SAMPLES_PER_TASK - run.samples_per_task
        if not targets or extra < 1:
            logger.info("%s: no follow-up needed", group.name)
            return
        if (
            missing_episodes(
                read_episode_records(group_dir(run, group, follow_up=True) / "records.jsonl"),
                targets,
                extra,
            )
            == {}
        ):
            logger.info("%s: follow-up complete", group.name)
            return
    else:
        if not missing_episodes(base_records, task_ids, run.samples_per_task):
            logger.info("%s: complete, skipping", group.name)
            return
        targets, extra = tuple(task_ids), run.samples_per_task
    logger.info("running forage group %s: %d tasks x %d samples", group.name, len(targets), extra)
    if group.kind == "ladder":
        exit_code = runner.main(
            ladder_group_arguments(
                run,
                group,
                endpoint,
                server_log,
                targets,
                samples=extra,
                follow_up=follow_up,
                gpu_preflight_passed=True,
            )
        )
        if exit_code != 0:
            raise RuntimeError(f"ladder group exited with status {exit_code}")
        return
    generate.main(
        generate_group_arguments(
            run,
            group,
            endpoint,
            targets,
            server_pid=server_pid,
            startup_seconds=startup_seconds,
            follow_up_samples=extra if follow_up else None,
        )
    )


def _serve_and_run(
    run: ForageRun,
    selection: Mapping[str, Sequence[str]],
    groups: Mapping[str, ForageGroup],
    gpu_memory_utilization: float | None,
    *,
    follow_up: bool,
) -> None:
    require_free_gpu()
    server_args = evaluate.parse_args(_server_arguments(run, gpu_memory_utilization))
    server_config = evaluate.build_server_config(
        server_args, evaluate.resolve_model_revision(server_args)
    )
    log_path = server_config.log_path
    if log_path is None:
        raise ValueError("the forage needs the vLLM server log")
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    started = time.monotonic()
    with VLLMServer(server_config) as server:
        startup_seconds = time.monotonic() - started
        with log_path.open("rb") as server_log:
            server_log.seek(log_offset)
            startup_log = server_log.read().decode("utf-8")
        served_run = replace(
            run,
            concurrency=kv_pool_concurrency(
                startup_log, run.max_model_len, max_concurrency=run.max_concurrency
            ),
            kv_capacity_tokens=kv_capacity_tokens(startup_log),
        )
        for name, task_ids in selection.items():
            _run_group(
                served_run,
                groups[name],
                task_ids,
                endpoint=server.base_url,
                server_log=log_path,
                server_pid=server.process.pid,
                startup_seconds=startup_seconds,
                follow_up=follow_up,
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the groups, sample counts and serving controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="base", help="output directory name under the root")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--base-model", default=transfer_battery.BASE_MODEL, help="Qwen/Qwen3.5-0.8B for a smoke"
    )
    parser.add_argument("--group", action="append", dest="groups", choices=tuple(GROUPS_BY_NAME))
    parser.add_argument(
        "--max-turns",
        action="append",
        default=[],
        metavar="GROUP=N",
        help="override a group's maximum turns; repeat for multiple groups",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        metavar="GROUP=TASK_ID",
        help="restrict a group's selection to an eligible task ID; repeat for multiple tasks",
    )
    parser.add_argument("--samples-per-task", type=int, default=SAMPLES_PER_TASK)
    parser.add_argument(
        "--tasks-per-group", type=int, help="override every group's task count (smoke runs)"
    )
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help="cap server sequences and concurrent episodes independently of KV capacity",
    )
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument(
        "--enable-speculative-decoding", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--accept-slow", help="forwarded throughput-gate waiver with its reason")
    parser.add_argument(
        "--oversize-reason",
        help="required when a ladder group or follow-up exceeds the 8-episode runner cell cap",
    )
    parser.add_argument(
        "--follow-up",
        action="store_true",
        help=f"add samples up to {FOLLOW_UP_SAMPLES_PER_TASK} per task for tasks with at most "
        f"{FOLLOW_UP_MAX_HACKS} hack",
    )
    parser.add_argument("--readout-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="select tasks on CPU, validate every group command and print the plan",
    )
    add_serving_quantization_arguments(
        parser,
        kv_cache_dtype_default="fp8",
        sampler_quantization_default="fp8_per_block",
        sampler_quantization_ignore_default=DEFAULT_QUANTIZATION_IGNORE,
        quantization_admission_default=DEFAULT_QUANTIZATION_ADMISSION,
    )
    args = parser.parse_args(argv)
    if args.max_concurrency < 1:
        parser.error("max-concurrency must be positive")
    validate_serving_quantization(
        args,
        parser,
        speculative_config={"method": "mtp", "num_speculative_tokens": 2}
        if args.enable_speculative_decoding
        else None,
    )
    max_turns_overrides = _parse_max_turns_overrides(parser, args.max_turns)
    task_id_overrides = _parse_task_id_overrides(parser, args.task_id)
    if args.samples_per_task < 1 or (args.tasks_per_group is not None and args.tasks_per_group < 1):
        parser.error("sample and task counts must be positive")
    if args.follow_up and args.oversize_reason is None:
        parser.error("--follow-up runs tasks past 8 samples; pass --oversize-reason")
    args.groups = tuple(args.groups or GROUPS_BY_NAME)
    unselected_turn_cap_groups = sorted(set(max_turns_overrides) - set(args.groups))
    if unselected_turn_cap_groups:
        parser.error(
            f"--max-turns specified for groups not selected by --group: {unselected_turn_cap_groups}"
        )
    unselected_task_groups = sorted(set(task_id_overrides) - set(args.groups))
    if unselected_task_groups:
        parser.error(
            f"--task-id specified for groups not selected by --group: {unselected_task_groups}"
        )
    args.max_turns_overrides = max_turns_overrides
    args.task_id_overrides = task_id_overrides
    return args


def _validate_commands(
    run: ForageRun,
    selection: Mapping[str, Sequence[str]],
    groups: Mapping[str, ForageGroup],
) -> None:
    for name, task_ids in selection.items():
        group = groups[name]
        if group.kind == "ladder":
            command = ladder_group_arguments(
                run,
                group,
                "http://127.0.0.1:8000",
                run.output_dir / "server" / "vllm-server.log",
                task_ids,
                samples=run.samples_per_task,
            )
            runner._parse_cli_args(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                runner._build_cli_parser(),  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                command,
            )
        else:
            command = generate_group_arguments(run, group, "http://127.0.0.1:8000", task_ids)
            generate.parse_args(command)
        logger.info(
            "group %s: %d tasks x %d samples = %d episodes",
            name,
            len(task_ids),
            run.samples_per_task,
            len(task_ids) * run.samples_per_task,
        )
        logger.info("  %s", " ".join(command))


def run_from_args(args: argparse.Namespace) -> ForageRun:
    """Resolve parsed serving settings into the configuration shared by forage groups."""
    return ForageRun(
        base_model=args.base_model,
        output_dir=(args.output_root / args.label).resolve(),
        max_model_len=args.max_model_len,
        concurrency=1,
        max_concurrency=args.max_concurrency,
        samples_per_task=args.samples_per_task,
        speculative_decoding=args.enable_speculative_decoding,
        accept_slow=args.accept_slow,
        oversize_reason=args.oversize_reason,
        kv_cache_dtype=args.kv_cache_dtype,
        sampler_quantization=args.sampler_quantization,
        sampler_quantization_ignore=tuple(args.sampler_quantization_ignore),
        quantization_admission=args.quantization_admission,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Select tasks, run (or validate) every group against one server, then write the readout."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    output_dir = (args.output_root / args.label).resolve()
    require_gitignored_output_dir(output_dir)
    run = run_from_args(args)
    groups = resolve_groups(args.groups, args.max_turns_overrides)
    if not args.readout_only:
        scratch_dir = output_dir / "selection-scratch"
        battery_items = battery_item_identities(scratch_dir)
        selection = {
            group.name: select_group_tasks(
                group,
                args.tasks_per_group or group.task_count,
                battery_items,
                scratch_dir,
                requested_task_ids=args.task_id_overrides.get(group.name),
            )
            for group in groups.values()
        }
        logger.info(
            "forage plan: %d groups, %d tasks, %d episodes",
            len(selection),
            sum(len(task_ids) for task_ids in selection.values()),
            sum(len(task_ids) for task_ids in selection.values()) * args.samples_per_task,
        )
        _validate_commands(run, selection, groups)
        if args.dry_run:
            return
        write_selection(run, selection, groups)
        _serve_and_run(
            run, selection, groups, args.gpu_memory_utilization, follow_up=args.follow_up
        )
    stored = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    stored_groups = stored.get("groups", stored)
    stored_selection = {
        name: config["task_ids"] if isinstance(config, dict) else config
        for name, config in stored_groups.items()
    }
    write_readout(
        output_dir,
        {
            name: (
                read_episode_records(group_dir(run, GROUPS_BY_NAME[name]) / "records.jsonl")
                + read_episode_records(
                    group_dir(run, GROUPS_BY_NAME[name], follow_up=True) / "records.jsonl"
                ),
                task_ids,
            )
            for name, task_ids in stored_selection.items()
            if name in args.groups
        },
    )


if __name__ == "__main__":
    main()
