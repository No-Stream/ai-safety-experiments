"""Generate resumable, exact-token teacher episodes from impossible coding tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from reward_hacking.aisi_rh_envs_data import AISI_SOURCES
from reward_hacking.distill.episodes import GRADER_DIRECTED_LABEL_FIELDS
from reward_hacking.distill.hack_strategy import label_episode
from reward_hacking.evidence_store import (
    compact_record,
    guard_episode_size,
    hydrate_record,
    read_jsonl,
)
from reward_hacking.generation_perf import (
    add_performance_arguments,
    generation_monitor,
    record_episode_timings,
)
from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.ladder.loop import (
    ANNOUNCED_TURN_BUDGET_SENTENCE,
    LADDER_MAX_SUBMISSIONS,
    PROMPT_FORMAT_VERSION,
)
from reward_hacking.ladder.runner import eval_awareness_from_trace_records
from reward_hacking.ladder.scenario import LADDER_SYSTEM_PROMPT, LADDER_USER_TEMPLATE
from reward_hacking.ladder.surfaces import FINAL_REPORT_PROMPT as TASK_LOG_FINAL_REPORT_PROMPT
from reward_hacking.ladder.thinking import (
    DEFAULT_THINKING_BUDGET_OVERSHOOT_POLICY,
    ThinkingBudgetOvershootPolicy,
)
from reward_hacking.ladder.tools import bash_tool, contains_submit_marker
from reward_hacking.model_backend import validate_thinking_budget_sampling
from reward_hacking.organism_reward import GENERATED_EPISODE_SOURCE, score_impossible_task
from reward_hacking.organism_rl.admission import LiveKVAdmission
from reward_hacking.organism_rl.generate_provenance import (
    _CODE_PROVENANCE_KEYS,  # pyright: ignore[reportPrivateUsage]
    PROVENANCE_SCHEMA_COMPATIBLE_VERSIONS,
    PROVENANCE_SCHEMA_VERSION,
    RECORD_SCHEMA_VERSION,
    _canonical_sha256,  # pyright: ignore[reportPrivateUsage]
    _load_existing_provenance,  # pyright: ignore[reportPrivateUsage]
    _provenance_identity,  # pyright: ignore[reportPrivateUsage]
    _read_existing_provenance,  # pyright: ignore[reportPrivateUsage]
    active_provenance_path,
    build_launch_context,
    compact_provenance,
    sampling_policy,
    write_json_atomically,
)
from reward_hacking.organism_rl.leak_assets import leak_task_source_metadata
from reward_hacking.organism_rl.leak_selection import (
    add_leak_arguments,
    resolve_leak_options,
    round_robin_leak_tasks,
)
from reward_hacking.organism_rl.rollout import (
    CodingTaskFamily,
    EpisodeSample,
    EpisodeSource,
    TaskEpisodeSource,
    deterministic_episode_seed,
    recorded_true_pass,
    task_definition_content,
    task_definition_sha256,
    task_kind_for,
)
from reward_hacking.organism_rl.task_mix import (
    TaskPrompt,
    _assign_payment_register,  # pyright: ignore[reportPrivateUsage]  # shared deterministic register logic
    parse_in_band_fraction,
)
from reward_hacking.organisms.statement_dedup import (
    build_local_exclusion_report,
    filter_excluded_task_ids,
)
from reward_hacking.terminal_wrench_runtime import SOURCE_GRADER_COMPLETE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grpo.gpu_perf import PerfMonitor

from reward_hacking.semantic_identity import semantic_content_identity

logger = logging.getLogger(__name__)
DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_FAMILIES: tuple[str, ...] = ("ilcb", "aisi")
PAYMENT_REGISTER_ASSIGNMENT_SEED = 0
# Below this, stop starting episodes; the shared root disk stalls every session when it fills.
MINIMUM_FREE_DISK_BYTES = 20 * 1024**3


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """The selected registry identity used to group and resume one task."""

    task_id: str
    family: CodingTaskFamily
    variant: str
    partition: str = "train"
    payment_register: Literal["trusted_rerun", "in_band"] = "trusted_rerun"


@dataclass(frozen=True, slots=True)
class TaskIdSubsetManifest:
    """A named, ordered subset of task IDs loaded from an ignored manifest."""

    path: Path
    name: str
    family: str
    partition: str
    task_ids: tuple[str, ...]
    source_sha256: str
    semantic_content: dict[str, object]


def _validated_task_subset_fields(payload: object) -> tuple[str, str, str, tuple[str, ...]]:
    if not isinstance(payload, Mapping):
        raise TypeError("task subset manifest must be a JSON object")
    expected_keys = {"schema_version", "name", "family", "partition", "task_ids"}
    if set(payload) != expected_keys or type(payload.get("schema_version")) is not int:
        raise ValueError("task subset manifest has an unsupported shape or schema")
    if payload["schema_version"] != 1:
        raise ValueError("task subset manifest has an unsupported schema version")
    name = payload["name"]
    family = payload["family"]
    partition = payload["partition"]
    task_ids_value = payload["task_ids"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("task subset name must be nonempty")
    if not isinstance(family, str) or family != "ilcb":
        raise ValueError("task subset family must be ilcb")
    if not isinstance(partition, str) or partition not in {"train", "held-out"}:
        raise ValueError("task subset partition must be train or held-out")
    if not isinstance(task_ids_value, list) or any(
        not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids_value
    ):
        raise ValueError("task subset task_ids must be a nonempty list of nonempty strings")
    task_ids = tuple(cast("list[str]", task_ids_value))
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("task subset task_ids must be nonempty and unique")
    return name, family, partition, task_ids


def load_task_subset_manifest(path: Path) -> TaskIdSubsetManifest:
    """Load a strict task-ID subset manifest without retaining benchmark text."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"task subset manifest must be a regular file: {path}")
    raw_bytes = path.read_bytes()
    try:
        payload: object = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"task subset manifest is not valid JSON: {path}") from error
    name, family, partition, task_ids = _validated_task_subset_fields(payload)
    semantic_content = semantic_content_identity(
        {
            "name": name,
            "family": family,
            "partition": partition,
            "task_ids": list(task_ids),
        }
    )
    return TaskIdSubsetManifest(
        path=path,
        name=name,
        family=family,
        partition=partition,
        task_ids=task_ids,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        semantic_content=semantic_content,
    )


def select_task_ids_from_subset(
    *,
    task_ids: Sequence[str],
    environment_ids: Mapping[str, str],
    subset: TaskIdSubsetManifest,
    family: str,
    partition: str,
) -> tuple[str, ...]:
    """Select task IDs in the subset's order, refusing missing or mismatched rows."""
    if subset.family != family:
        raise ValueError("task subset family does not match the selected task family")
    if subset.partition != partition:
        raise ValueError("task subset partition does not match the selected partition")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("selected task IDs must be unique before subset filtering")
    task_id_by_environment: dict[str, str] = {}
    for task_id in task_ids:
        environment_id = environment_ids[task_id]
        if environment_id in task_id_by_environment:
            raise ValueError("task subset selection has duplicate environment IDs")
        task_id_by_environment[environment_id] = task_id
    missing_task_ids = tuple(
        problem_id for problem_id in subset.task_ids if problem_id not in task_id_by_environment
    )
    if missing_task_ids:
        raise ValueError(
            f"task subset IDs are not available for the selected variant and partition: "
            f"{missing_task_ids}"
        )
    return tuple(task_id_by_environment[problem_id] for problem_id in subset.task_ids)


def select_task_ids(
    task_ids: Sequence[str],
    *,
    max_tasks: int | None,
    aisi_sources: Sequence[str] = AISI_SOURCES,
) -> tuple[str, ...]:
    """Round-robin selected AISI sources before applying a deterministic task limit."""
    if any(not task_id for task_id in task_ids):
        raise ValueError("selected task ids must be nonempty")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("selected task ids must be unique")
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
        raise ValueError("max_tasks must be a positive integer")
    if not aisi_sources or len(aisi_sources) != len(set(aisi_sources)):
        raise ValueError("AISI sources must be a nonempty unique sequence")
    if any(source not in AISI_SOURCES for source in aisi_sources):
        raise ValueError(f"unsupported AISI source in {aisi_sources!r}")

    selected_sources = set(aisi_sources)
    task_ids_by_source: dict[str, list[str]] = {source: [] for source in AISI_SOURCES}
    other_task_ids: list[str] = []
    for task_id in task_ids:
        if not task_id.startswith("aisi-"):
            other_task_ids.append(task_id)
            continue
        source = next(
            (source for source in AISI_SOURCES if task_id.startswith(f"aisi-{source}-")), None
        )
        if source is None:
            raise ValueError(f"AISI task id has an unknown source: {task_id!r}")
        if source in selected_sources:
            task_ids_by_source[source].append(task_id)

    source_order = tuple(source for source in AISI_SOURCES if source in selected_sources)
    source_groups = [sorted(task_ids_by_source[source]) for source in source_order]
    ordered_aisi = [
        task_id
        for source_round in zip_longest(*source_groups)
        for task_id in source_round
        if task_id is not None
    ]
    ordered = [*ordered_aisi, *sorted(other_task_ids)]
    return tuple(ordered if max_tasks is None else ordered[:max_tasks])


def select_task_ids_with_dedup(  # noqa: PLR0913 - each selection constraint is an independent input
    task_ids: Sequence[str],
    source: TaskEpisodeSource,
    *,
    partition: str,
    exclude_duplicates: bool,
    max_tasks: int | None,
    excluded_task_ids: Sequence[str] = (),
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Apply task and statement exclusions before enforcing the deterministic task limit."""
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
        raise ValueError("max_tasks must be a positive integer")
    if any(not task_id for task_id in excluded_task_ids):
        raise ValueError("excluded task ids must be nonempty")
    if len(excluded_task_ids) != len(set(excluded_task_ids)):
        raise ValueError("excluded task ids must be unique")
    excluded_task_id_set = set(excluded_task_ids)
    selected_task_ids = tuple(
        task_id for task_id in task_ids if task_id not in excluded_task_id_set
    )
    metadata: dict[str, object] = {
        "enabled": exclude_duplicates,
        "partition": partition,
        "excluded_task_count": 0,
        "exclusion_item_count": 0,
        "exclusion_item_ids_sha256": None,
        "excluded_count_by_source": {},
    }
    families_by_selected_task = {
        task_id: source.families_by_task[task_id] for task_id in selected_task_ids
    }
    has_deduplicated_family = any(
        family in {"aisi", "ilcb"} for family in families_by_selected_task.values()
    )
    if exclude_duplicates and partition == "train" and has_deduplicated_family:
        exclusion_report = build_local_exclusion_report()
        item_ids_by_selected_task: dict[str, str] = {}
        for task_id, family in families_by_selected_task.items():
            if family not in {"aisi", "ilcb"}:
                continue
            item_id = source.tasks[task_id].problem_id
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(
                    f"selected task {task_id!r} is missing its benchmark item identity"
                )
            item_ids_by_selected_task[task_id] = item_id
        filtered_task_ids = filter_excluded_task_ids(
            selected_task_ids,
            families_by_task=families_by_selected_task,
            item_ids_by_task=item_ids_by_selected_task,
            excluded_item_ids=exclusion_report.excluded_item_ids,
        )
        excluded_item_ids = sorted(exclusion_report.excluded_item_ids)
        metadata.update(
            {
                "excluded_task_count": len(selected_task_ids) - len(filtered_task_ids),
                "exclusion_item_count": len(excluded_item_ids),
                "exclusion_item_ids_sha256": hashlib.sha256(
                    json.dumps(excluded_item_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "excluded_count_by_source": dict(exclusion_report.excluded_count_by_source),
            }
        )
        selected_task_ids = filtered_task_ids
    if max_tasks is not None:
        if len(selected_task_ids) < max_tasks:
            raise ValueError(
                f"max_tasks={max_tasks} requested, but only {len(selected_task_ids)} eligible tasks remain"
            )
        selected_task_ids = selected_task_ids[:max_tasks]
    return selected_task_ids, metadata


def _validate_task_sample_selection(
    task_ids: Sequence[str],
    excluded_task_samples: Sequence[tuple[str, int]],
    max_episodes: int,
    samples_per_task: int,
) -> None:
    if not task_ids or any(not task_id for task_id in task_ids):
        raise ValueError("eligible task IDs must be nonempty")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("eligible task IDs must be unique")
    if type(max_episodes) is not int or max_episodes < 1:
        raise ValueError("max_episodes must be a positive integer")
    if type(samples_per_task) is not int or samples_per_task < 1:
        raise ValueError("samples_per_task must be a positive integer")
    if any(
        not task_id or type(sample_index) is not int or sample_index < 0
        for task_id, sample_index in excluded_task_samples
    ):
        raise ValueError(
            "excluded task samples must contain nonempty task IDs and nonnegative indexes"
        )
    if len(excluded_task_samples) != len(set(excluded_task_samples)):
        raise ValueError("excluded task samples must be unique")


def _next_sample_index(samples: set[int]) -> int:
    """Return the first unused nonnegative sample index for one task."""
    sample_index = 0
    while sample_index in samples:
        sample_index += 1
    return sample_index


def select_task_sample_pairs(
    task_ids: Sequence[str],
    *,
    excluded_task_samples: Sequence[tuple[str, int]],
    max_episodes: int,
    samples_per_task: int,
    completed_task_samples: Sequence[tuple[str, int]] = (),
) -> tuple[tuple[str, int], ...]:
    """Choose fresh task samples first, then balance repeats across least-sampled tasks."""
    _validate_task_sample_selection(task_ids, excluded_task_samples, max_episodes, samples_per_task)
    _validate_task_sample_selection(
        task_ids, completed_task_samples, max_episodes, samples_per_task
    )
    task_order = {task_id: order for order, task_id in enumerate(task_ids)}
    samples_by_task: dict[str, set[int]] = {task_id: set() for task_id in task_ids}
    for task_id, sample_index in excluded_task_samples:
        if task_id in samples_by_task:
            samples_by_task[task_id].add(sample_index)
    if len(completed_task_samples) > max_episodes:
        raise ValueError("max_episodes cannot be smaller than the completed stage prefix")
    stage_counts = dict.fromkeys(task_ids, 0)
    selected = list(completed_task_samples)
    for task_id, sample_index in completed_task_samples:
        if task_id not in samples_by_task:
            raise ValueError(f"completed task {task_id!r} is no longer eligible")
        samples_by_task[task_id].add(sample_index)
        stage_counts[task_id] += 1

    while len(selected) < max_episodes:
        available_task_ids = [
            task_id for task_id in task_ids if stage_counts[task_id] < samples_per_task
        ]
        if not available_task_ids:
            capacity = len(task_ids) * samples_per_task
            raise ValueError(
                f"max_episodes={max_episodes} requested, but only {capacity} eligible task samples "
                f"are available with samples_per_task={samples_per_task}"
            )
        task_id = min(
            available_task_ids,
            key=lambda candidate: (len(samples_by_task[candidate]), task_order[candidate]),
        )
        sample_index = _next_sample_index(samples_by_task[task_id])
        selected.append((task_id, sample_index))
        samples_by_task[task_id].add(sample_index)
        stage_counts[task_id] += 1
    return tuple(selected)


def _select_stage_task_samples(
    task_ids: Sequence[str],
    *,
    output_dir: Path,
    excluded_task_samples: Sequence[tuple[str, int]],
    max_episodes: int,
    samples_per_task: int,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    """Retain completed episodes when growing or changing a stage's sampling policy."""
    provenance_path = active_provenance_path(output_dir)
    completed: tuple[tuple[str, int], ...] = ()
    if provenance_path.exists():
        existing = _read_existing_provenance(provenance_path)
        stored = cast("Mapping[str, object]", existing["provenance"])
        _, _, accepted_hashes = _load_existing_provenance(
            provenance_path, _provenance_identity(stored)
        )
        accepted_hashes.add(cast("str", existing["provenance_sha256"]))
        records, _ = _existing_records(
            output_dir / "records.jsonl", accepted_provenance_sha256=accepted_hashes
        )
        completed = tuple(
            (cast("str", record["task_id"]), cast("int", record["sample_index"]))
            for record in records
        )
        saved_selection = stored.get("task_sample_selection")
        saved_flags = cast("Mapping[str, object]", stored["flags"])
        saved_snapshot = _task_sample_exclusion_from_provenance(stored)
        excluded_task_samples = tuple(
            sorted(set(excluded_task_samples).union(saved_snapshot or ()))
        )
        if (
            isinstance(saved_selection, list)
            and len(saved_selection) == max_episodes
            and saved_flags["samples_per_task"] == samples_per_task
            and saved_snapshot == tuple(sorted(set(excluded_task_samples)))
        ):
            selection = tuple(
                (cast("str", pair["task_id"]), cast("int", pair["sample_index"]))
                for pair in saved_selection
            )
            if saved_snapshot is None:
                raise ValueError("saved sample selection has no exclusion snapshot")
            return selection, saved_snapshot
    if set(completed).intersection(excluded_task_samples):
        raise ValueError(
            "completed stage prefix overlaps task/sample keys used elsewhere in the cell"
        )
    snapshot = tuple(sorted(set(excluded_task_samples)))
    selection = select_task_sample_pairs(
        task_ids,
        excluded_task_samples=tuple(sorted(set(snapshot).union(completed))),
        max_episodes=max_episodes,
        samples_per_task=samples_per_task,
        completed_task_samples=completed,
    )
    return selection, snapshot


def _read_task_id_exclusion_inputs(
    paths: Sequence[Path],
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read strict task IDs and record each source path and content digest."""
    task_ids: set[str] = set()
    source_records: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"task-id exclusion records file does not exist: {path}")
        contents = path.read_bytes()
        try:
            records_text = contents.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{path} is not UTF-8 JSONL") from error
        for line_number, line in enumerate(records_text.splitlines(), start=1):
            try:
                record: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {line_number} is not valid JSON") from error
            if not isinstance(record, Mapping):
                raise TypeError(f"{path} line {line_number} must be a JSON object")
            task_id = record.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"{path} line {line_number} must contain a nonempty task_id")
            if task_id != task_id.strip():
                raise ValueError(
                    f"{path} line {line_number} task_id must not have surrounding whitespace"
                )
            task_ids.add(task_id)
        source_records.append({"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()})
    return tuple(sorted(task_ids)), source_records


def _read_task_sample_exclusion_inputs(
    paths: Sequence[Path],
) -> tuple[tuple[tuple[str, int], ...], list[dict[str, str]]]:
    """Read strict task/sample identities and digest each exclusion source."""
    task_samples: set[tuple[str, int]] = set()
    source_records: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"task-sample exclusion records file does not exist: {path}")
        contents = path.read_bytes()
        try:
            records_text = contents.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{path} is not UTF-8 JSONL") from error
        for line_number, line in enumerate(records_text.splitlines(), start=1):
            try:
                record: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {line_number} is not valid JSON") from error
            if not isinstance(record, Mapping):
                raise TypeError(f"{path} line {line_number} must be a JSON object")
            task_id = record.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip() or task_id != task_id.strip():
                raise ValueError(
                    f"{path} line {line_number} must contain a nonempty canonical task_id"
                )
            sample_index = record.get("sample_index")
            if type(sample_index) is not int or sample_index < 0:
                raise ValueError(
                    f"{path} line {line_number} must contain a nonnegative sample_index"
                )
            task_samples.add((task_id, sample_index))
        source_records.append({"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()})
    return tuple(sorted(task_samples)), source_records


def _task_sample_exclusion_from_provenance(
    provenance: Mapping[str, object],
) -> tuple[tuple[str, int], ...] | None:
    """Read and validate the immutable task/sample exclusion snapshot in provenance."""
    if "task_sample_exclusion" not in provenance:
        return None
    exclusion_value = provenance["task_sample_exclusion"]
    if not isinstance(exclusion_value, Mapping):
        raise TypeError("task_sample_exclusion provenance must be an object")
    if (
        type(exclusion_value.get("schema_version")) is not int
        or exclusion_value["schema_version"] < 1
    ):
        raise ValueError("task_sample_exclusion provenance has an unsupported schema version")
    samples_value = exclusion_value.get("task_samples")
    if not isinstance(samples_value, list):
        raise TypeError("task_sample_exclusion provenance must contain task_samples")
    task_samples: list[tuple[str, int]] = []
    for value in samples_value:
        if not isinstance(value, Mapping):
            raise TypeError("task_sample_exclusion entries must be objects")
        task_id = value.get("task_id")
        sample_index = value.get("sample_index")
        if (
            not isinstance(task_id, str)
            or not task_id.strip()
            or task_id != task_id.strip()
            or type(sample_index) is not int
            or sample_index < 0
        ):
            raise ValueError(
                "task_sample_exclusion entries require a task_id and nonnegative sample_index"
            )
        task_samples.append((task_id, sample_index))
    if task_samples != sorted(set(task_samples)):
        raise ValueError("task_sample_exclusion task samples must be sorted and unique")
    return tuple(task_samples)


def _saved_task_sample_exclusion_snapshot(output_dir: Path) -> tuple[tuple[str, int], ...] | None:
    """Validate existing provenance before reusing its task/sample exclusion snapshot."""
    provenance_path = active_provenance_path(output_dir)
    if not provenance_path.is_file():
        return None
    existing = _read_existing_provenance(provenance_path)
    stored_provenance = cast("Mapping[str, object]", existing["provenance"])
    stored_identity = _provenance_identity(stored_provenance)
    schema_version = cast("int", existing["schema_version"])
    if schema_version == 1 or schema_version in PROVENANCE_SCHEMA_COMPATIBLE_VERSIONS:
        _load_existing_provenance(provenance_path, stored_identity)
    else:
        raise ValueError(
            f"unsupported run provenance schema {schema_version}; use a fresh output directory"
        )
    return _task_sample_exclusion_from_provenance(stored_provenance)


def _resolve_task_sample_exclusion_snapshot(
    paths: Sequence[Path], output_dir: Path, *, freeze_empty: bool = False
) -> tuple[
    tuple[tuple[str, int], ...] | None,
    tuple[tuple[str, int], ...] | None,
    list[dict[str, str]],
]:
    """Freeze first-launch sample exclusions while retaining current keys for collision checks."""
    current_task_samples: tuple[tuple[str, int], ...] | None = None
    source_records: list[dict[str, str]] = []
    if paths:
        current_task_samples, source_records = _read_task_sample_exclusion_inputs(paths)
    elif freeze_empty:
        current_task_samples = ()
    saved_task_samples = _saved_task_sample_exclusion_snapshot(output_dir)
    task_sample_snapshot = (
        saved_task_samples if saved_task_samples is not None else current_task_samples
    )
    return task_sample_snapshot, current_task_samples, source_records


def _reject_new_task_sample_exclusion_collisions(
    task_sample_snapshot: Sequence[tuple[str, int]] | None,
    current_task_samples: Sequence[tuple[str, int]] | None,
    selected_task_samples: Sequence[tuple[str, int]],
) -> None:
    """Refuse newly discovered sample keys that overlap this stage's frozen selection."""
    if task_sample_snapshot is None or current_task_samples is None:
        return
    new_task_samples = set(current_task_samples) - set(task_sample_snapshot)
    collisions = new_task_samples.intersection(selected_task_samples)
    if collisions:
        raise ValueError(
            f"{len(collisions)} newly observed excluded task/sample keys overlap this stage's fixed selection"
        )


def _task_id_exclusion_from_provenance(
    provenance: Mapping[str, object],
) -> tuple[str, ...] | None:
    """Read and validate the immutable task-ID exclusion snapshot in run provenance."""
    if "task_id_exclusion" not in provenance:
        return None
    exclusion_value = provenance["task_id_exclusion"]
    if not isinstance(exclusion_value, Mapping):
        raise TypeError("task_id_exclusion provenance must be an object")
    schema_version = exclusion_value.get("schema_version")
    if type(schema_version) is not int or schema_version < 1:
        raise ValueError("task_id_exclusion provenance has an unsupported schema version")
    task_ids_value = exclusion_value.get("task_ids")
    if not isinstance(task_ids_value, list) or any(
        not isinstance(task_id, str) or not task_id.strip() or task_id != task_id.strip()
        for task_id in task_ids_value
    ):
        raise ValueError("task_id_exclusion provenance must contain nonempty task IDs")
    task_ids = cast("list[str]", task_ids_value)
    if task_ids != sorted(set(task_ids)):
        raise ValueError("task_id_exclusion provenance task IDs must be sorted and unique")
    return tuple(task_ids)


def _saved_task_id_exclusion_snapshot(output_dir: Path) -> tuple[str, ...] | None:
    """Validate existing provenance before reusing its task-ID exclusion snapshot."""
    provenance_path = active_provenance_path(output_dir)
    if not provenance_path.is_file():
        return None
    existing = _read_existing_provenance(provenance_path)
    stored_provenance = cast("Mapping[str, object]", existing["provenance"])
    stored_identity = _provenance_identity(stored_provenance)
    schema_version = cast("int", existing["schema_version"])
    if schema_version == 1 or schema_version in PROVENANCE_SCHEMA_COMPATIBLE_VERSIONS:
        _load_existing_provenance(provenance_path, stored_identity)
    else:
        raise ValueError(
            f"unsupported run provenance schema {schema_version}; use a fresh output directory"
        )
    return _task_id_exclusion_from_provenance(stored_provenance)


def _resolve_task_id_exclusion_snapshot(
    paths: Sequence[Path], output_dir: Path
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None, list[dict[str, str]]]:
    """Freeze first-launch exclusions while returning current IDs for collision checks."""
    current_task_ids: tuple[str, ...] | None = None
    source_records: list[dict[str, str]] = []
    if paths:
        current_task_ids, source_records = _read_task_id_exclusion_inputs(paths)
    saved_task_ids = _saved_task_id_exclusion_snapshot(output_dir)
    task_id_snapshot = saved_task_ids if saved_task_ids is not None else current_task_ids
    return task_id_snapshot, current_task_ids, source_records


def _reject_new_exclusion_collisions(
    task_id_snapshot: Sequence[str] | None,
    current_task_ids: Sequence[str] | None,
    selected_task_ids: Sequence[str],
) -> None:
    """Refuse newly observed cross-stage IDs that overlap this stage's frozen selection."""
    if task_id_snapshot is None or current_task_ids is None:
        return
    new_task_ids = set(current_task_ids) - set(task_id_snapshot)
    collisions = new_task_ids.intersection(selected_task_ids)
    if collisions:
        raise ValueError(
            f"{len(collisions)} newly observed excluded task IDs overlap this stage's fixed selection"
        )


def parse_aisi_sources(raw_value: str) -> tuple[str, ...]:
    """Parse a comma-separated AISI source allowlist into canonical source order."""
    requested_sources = raw_value.split(",")
    if any(not source for source in requested_sources):
        raise argparse.ArgumentTypeError(
            "--aisi-sources must be a comma-separated source allowlist"
        )
    if len(requested_sources) != len(set(requested_sources)):
        raise argparse.ArgumentTypeError("--aisi-sources must not contain duplicates")
    unknown_sources = sorted(set(requested_sources) - set(AISI_SOURCES))
    if unknown_sources:
        raise argparse.ArgumentTypeError(
            f"--aisi-sources contains unsupported sources: {', '.join(unknown_sources)}"
        )
    return tuple(source for source in AISI_SOURCES if source in requested_sources)


def _parse_chat_template_kwarg(raw_value: str) -> tuple[str, str, str]:
    """Parse the ladder runner's ``MODEL_ID:KEY=VALUE`` chat-template syntax."""
    model_id, colon, assignment = raw_value.partition(":")
    key, equals, value = assignment.partition("=")
    if not colon or not model_id or not equals or not key:
        raise argparse.ArgumentTypeError("--chat-template-kwarg must be MODEL_ID:KEY=VALUE")
    return model_id, key, value


def _chat_template_options(
    entries: Sequence[tuple[str, str, str]], *, model_ids: set[str]
) -> dict[str, str]:
    """Resolve one model's options and reject typos or duplicate keys."""
    options: dict[str, str] = {}
    for model_id, key, value in entries:
        if model_id not in model_ids:
            raise ValueError(f"chat-template kwarg targets unknown model id {model_id!r}")
        if key in options:
            raise ValueError(f"duplicate chat-template kwarg {key!r}")
        options[key] = value
    return options


def _parse_json_object(raw_value: str) -> dict[str, object]:
    """Parse one JSON object-valued CLI argument."""
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("value must be valid JSON") from error
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # noqa: PLR0915 - explicit CLI options
    """Parse generation, sampling and persistence settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_performance_arguments(parser)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--generator-name", choices=("base", "v3", "27b"))
    parser.add_argument("--serving-config", type=_parse_json_object)
    parser.add_argument("--serving-runtime", type=_parse_json_object)
    parser.add_argument("--served-model-id")
    parser.add_argument(
        "--served-adapter-sha256",
        help="content digest of the LoRA adapter served for this generation run",
    )
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--family",
        "--families",
        dest="family_entries",
        action="append",
        choices=(
            "ilcb",
            "aisi",
            "math_oneoff",
            "math_loose_checker",
            "r2e_leak",
            "terminal_wrench",
        ),
    )
    parser.add_argument(
        "--variant",
        choices=(
            "oneoff",
            "original",
            "conflicting",
            "golden",
            "timed",
            "golden_conflicting",
            "timed_conflicting",
            "loose",
            "loose_oneoff",
            "loose_hidden_original",
        ),
        default="oneoff",
    )
    parser.add_argument("--hints", type=Path)
    parser.add_argument("--hint-family")
    parser.add_argument("--hint-strength", choices=("vague", "specific", "none"), default="none")
    parser.add_argument("--convention-note", action="store_true")
    parser.add_argument("--exclude-duplicates", action=argparse.BooleanOptionalAction, default=True)
    _add_payment_register_arguments(parser)
    parser.add_argument("--partition", choices=("train", "held-out"), default="train")
    parser.add_argument(
        "--samples",
        "--samples-per-task",
        dest="samples_per_task",
        type=int,
        default=1,
    )
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--task-id",
        dest="explicit_task_ids",
        action="append",
        default=None,
        metavar="TASK_ID",
        help="run exactly these eligible task IDs, in this order; repeat for each task",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--ilcb-cases",
        type=Path,
        help="use an ignored ILCB cases file instead of the baked task corpus",
    )
    parser.add_argument(
        "--ilcb-split-manifest",
        type=Path,
        help="use an ignored ILCB split manifest instead of the baked split manifest",
    )
    parser.add_argument(
        "--ilcb-task-subset",
        type=Path,
        help="select an ordered, named ILCB problem-ID subset from an ignored JSON manifest",
    )
    parser.add_argument(
        "--exclude-task-ids-from",
        type=Path,
        action="extend",
        nargs="+",
        default=[],
        metavar="PATH",
        help="exclude task IDs present in these JSONL records files",
    )
    parser.add_argument(
        "--exclude-task-samples-from",
        type=Path,
        action="extend",
        nargs="+",
        default=[],
        metavar="PATH",
        help="exclude task/sample pairs present in these JSONL records files",
    )
    parser.add_argument(
        "--aisi-sources",
        type=parse_aisi_sources,
        default=AISI_SOURCES,
        metavar="SOURCE,...",
    )
    parser.add_argument(
        "--aisi-cases",
        type=Path,
        help="use an ignored AISI cases file instead of the baked task corpus",
    )
    parser.add_argument(
        "--aisi-hack-group", choices=("rigged_comparison", "exit_code", "collection_config")
    )
    parser.add_argument("--terminal-wrench-registry", type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--kv-admission-capacity-tokens", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--thinking-token-budget", type=int)
    parser.add_argument("--thinking-budget-overshoot-max-fraction", type=float, default=0.1)
    parser.add_argument("--thinking-budget-overshoot-min-episodes", type=int, default=3)
    parser.add_argument("--announced-turn-budget", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    _add_sampling_overrides(parser)
    parser.add_argument(
        "--chat-template-kwarg",
        dest="chat_template_kwarg_entries",
        action="append",
        type=_parse_chat_template_kwarg,
        default=[],
    )
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--persistent-jail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse one actor jail per episode; disable for per-command jails",
    )

    parser.add_argument("--jail-backend")
    add_leak_arguments(parser)
    args = parser.parse_args(argv)
    return _validate_parsed_args(args, parser)


def _validate_episode_selection_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("max-tasks must be positive")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("max-episodes must be positive")
    if args.max_tasks is not None and args.max_episodes is not None:
        parser.error("max-tasks and max-episodes are mutually exclusive")
    if args.exclude_task_samples_from and args.max_episodes is None:
        parser.error("exclude-task-samples-from requires max-episodes")
    if args.explicit_task_ids is not None:
        if len(args.explicit_task_ids) != len(set(args.explicit_task_ids)):
            parser.error("task-id values must be unique")
        if args.max_tasks is not None or args.ilcb_task_subset is not None:
            parser.error("task-id cannot be combined with max-tasks or ilcb-task-subset")


def _validate_aisi_group(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.aisi_hack_group is not None and args.families != ["aisi"]:
        parser.error("AISI hack group selection requires only the AISI family")


def _validate_admission_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.kv_admission_capacity_tokens is not None and args.kv_admission_capacity_tokens < 1:
        parser.error("kv-admission-capacity-tokens must be positive")


def _validate_parsed_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    resolve_leak_options(args, parser, partition=args.partition)

    args.families = list(dict.fromkeys(args.family_entries or DEFAULT_FAMILIES))
    _parse_payment_register_arguments(args, parser)
    _resolve_hint_options(args, parser)
    if args.convention_note and (
        args.families != ["ilcb"]
        or args.variant not in {"golden", "timed"}
        or args.hint_strength != "none"
        or args.hint_family is not None
    ):
        parser.error("convention notes require unhinted original ILCB golden or timed tasks")
    args.served_model_id = args.served_model_id or args.model
    if (
        args.served_adapter_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", args.served_adapter_sha256) is None
    ):
        parser.error("served-adapter-sha256 must be a lowercase SHA-256 digest")
    args.max_turns = (
        args.announced_turn_budget
        if args.max_turns is None and args.announced_turn_budget is not None
        else DEFAULT_MAX_TURNS
        if args.max_turns is None
        else args.max_turns
    )
    if args.announced_turn_budget is not None and args.announced_turn_budget != args.max_turns:
        parser.error("announced-turn-budget must match max-turns")
    if args.variant == "conflicting" and any(family != "ilcb" for family in args.families):
        parser.error("conflicting tasks are only available for ILCB")
    if args.variant == "loose" and any(family != "math_loose_checker" for family in args.families):
        parser.error("loose tasks are only available for math_loose_checker")
    if args.variant in {"loose_oneoff", "loose_hidden_original"} and any(
        family != "math_loose_checker" for family in args.families
    ):
        parser.error("hidden-key loose math tasks are only available for math_loose_checker")
    _validate_aisi_group(args, parser)
    _validate_aisi_input_paths(args, parser)
    _validate_terminal_wrench_inputs(args, parser)
    _validate_ilcb_input_paths(args, parser)
    if (
        min(
            args.samples_per_task,
            args.concurrency,
            args.max_turns,
            args.max_new_tokens,
            args.max_model_len,
        )
        < 1
    ):
        parser.error("sample, concurrency, turn, token and model-length limits must be positive")
    _validate_admission_args(args, parser)
    _validate_episode_selection_args(args, parser)
    if args.thinking_token_budget is not None and args.thinking_token_budget < 0:
        parser.error("thinking-token-budget must be nonnegative")
    _validate_thinking_budget_policy(args, parser)
    _validate_sampling_args(args, parser)
    try:
        args.chat_template_kwargs = _chat_template_options(
            args.chat_template_kwarg_entries,
            model_ids={args.model, args.served_model_id},
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def _validate_ilcb_input_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    input_paths = (args.ilcb_cases, args.ilcb_split_manifest, args.ilcb_task_subset)
    if any(path is not None for path in input_paths) and args.families != ["ilcb"]:
        parser.error("ILCB input paths require --family ilcb without other families")
    if args.ilcb_cases is not None and args.ilcb_split_manifest is None:
        parser.error("ilcb-cases requires its matching --ilcb-split-manifest")


def _validate_aisi_input_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.aisi_cases is not None and args.families != ["aisi"]:
        parser.error("AISI cases require --family aisi without other families")


def _validate_terminal_wrench_inputs(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if "terminal_wrench" in args.families:
        if args.families != ["terminal_wrench"] or args.terminal_wrench_registry is None:
            parser.error(
                "Terminal Wrench requires its runtime registry and an exclusive task family"
            )
        if args.variant != "original" or args.partition != "train":
            parser.error(
                "Terminal Wrench requires original variant in its research-only train namespace"
            )
    elif args.terminal_wrench_registry is not None:
        parser.error("Terminal Wrench registry requires --family terminal_wrench")


def _record_ilcb_input_provenance(
    provenance: dict[str, object],
    args: argparse.Namespace,
    task_subset: TaskIdSubsetManifest | None,
) -> None:
    if task_subset is not None:
        provenance["task_subset"] = {
            "name": task_subset.name,
            "family": task_subset.family,
            "partition": task_subset.partition,
            "task_ids": list(task_subset.task_ids),
            **task_subset.semantic_content,
        }
        provenance["task_subset_source"] = {
            "path": str(task_subset.path.resolve()),
            "sha256": task_subset.source_sha256,
        }
    if args.ilcb_split_manifest is not None:
        provenance["ilcb_split_manifest_source"] = {
            "path": str(args.ilcb_split_manifest.resolve()),
            "sha256": _file_sha256(args.ilcb_split_manifest),
        }
    if args.ilcb_cases is not None:
        provenance["ilcb_cases_source"] = {
            "path": str(args.ilcb_cases.resolve()),
            "sha256": _file_sha256(args.ilcb_cases),
        }


def _record_aisi_input_provenance(provenance: dict[str, object], args: argparse.Namespace) -> None:
    if args.aisi_cases is not None:
        provenance["aisi_cases_source"] = {
            "path": str(args.aisi_cases.resolve()),
            "sha256": _file_sha256(args.aisi_cases),
        }


def _read_hint_entries(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read versioned family hints without copying their text into run metadata."""
    document: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or type(document.get("schema_version")) is not int:
        raise TypeError("hint file must be a versioned JSON object")
    if document["schema_version"] < 1:
        raise ValueError("unsupported hint file schema version")
    hints_value = document.get("hints")
    if not isinstance(hints_value, list):
        raise TypeError("hint file hints must be a list")
    entries: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for hint_value in hints_value:
        if not isinstance(hint_value, Mapping):
            raise TypeError("hint entries must be objects")
        family = hint_value.get("family")
        strength = hint_value.get("strength")
        text = hint_value.get("text")
        if not isinstance(family, str) or not family:
            raise TypeError("hint family must be a nonempty string")
        if not isinstance(strength, str) or strength not in {"vague", "specific"}:
            raise ValueError("hint entry strength must be vague or specific")
        if not isinstance(text, str) or not text.strip():
            raise TypeError("hint text must be nonempty")
        pair = (family, strength)
        if pair in seen_pairs:
            raise ValueError("hint file contains duplicate family and strength entries")
        seen_pairs.add(pair)
        entries.append({"family": family, "strength": pair[1], "text": text})
    return entries, _file_sha256(path)


def _resolve_hint_options(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Resolve one family hint and retain the selected hint digest for resume identity."""
    if args.hint_strength == "none":
        args.hint_text = None
        args.hint_sha = None
        args.hints_sha256 = None
        if args.hints is not None:
            if args.hint_family is None:
                parser.error("--hint-family is required when --hints is supplied")
            entries, args.hints_sha256 = _read_hint_entries(args.hints)
            if not any(entry["family"] == args.hint_family for entry in entries):
                parser.error("--hint-family is not present in --hints")
        return
    if args.hints is None or args.hint_family is None:
        parser.error("--hints and --hint-family are required for vague or specific hints")
    entries, args.hints_sha256 = _read_hint_entries(args.hints)
    matches = [
        entry
        for entry in entries
        if entry["family"] == args.hint_family and entry["strength"] == args.hint_strength
    ]
    if len(matches) != 1:
        parser.error("--hint-family and --hint-strength must select exactly one hint")
    args.hint_text = matches[0]["text"]
    args.hint_sha = hashlib.sha256(args.hint_text.encode("utf-8")).hexdigest()


def _add_payment_register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--payment-register", choices=("trusted_rerun", "in_band"), default="trusted_rerun"
    )
    parser.add_argument(
        "--in-band-fraction",
        dest="in_band_entries",
        action="append",
        type=parse_in_band_fraction,
        default=[],
        help="repeatable FAMILY=FRACTION or FAMILY/ENVIRONMENT=FRACTION override",
    )


def _parse_payment_register_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    entries = args.in_band_entries
    args.in_band_fractions = dict(entries)
    if len(args.in_band_fractions) != len(entries):
        parser.error("in-band-fraction scopes must be unique")
    if any(scope.split("/", 1)[0] not in args.families for scope in args.in_band_fractions):
        parser.error("in-band-fraction family must be selected with --family")
    vars(args).pop("in_band_entries")


def _assign_payment_registers(
    task_specs: Sequence[TaskSpec],
    environment_ids: Mapping[str, str],
    *,
    fallback_register: Literal["trusted_rerun", "in_band"],
    in_band_fractions: Mapping[str, float],
    seed: int,
) -> tuple[TaskSpec, ...]:
    prompts = tuple(
        TaskPrompt(
            task_id=spec.task_id,
            family=spec.family,
            variant=spec.variant,
            environment_id=environment_ids[spec.task_id],
        )
        for spec in task_specs
    )
    known_families = {prompt.family for prompt in prompts}
    known_environments = {
        f"{prompt.family}/{prompt.environment_id or prompt.task_id}" for prompt in prompts
    }
    for scope, fraction in in_band_fractions.items():
        if scope not in known_families and scope not in known_environments:
            raise ValueError(f"unknown payment fraction family or environment {scope!r}")
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("in-band fractions must be finite and in [0, 1]")
    assigned_registers = {
        prompt.task_id: _assign_payment_register(
            prompt,
            in_band_fractions,
            seed,
            default_register=fallback_register,
        ).payment_register
        for prompt in prompts
    }
    return tuple(
        replace(spec, payment_register=assigned_registers[spec.task_id]) for spec in task_specs
    )


def _add_sampling_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)


def _validate_sampling_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be positive and top-p must be in (0, 1]")
    try:
        validate_thinking_budget_sampling(
            thinking_token_budget=args.thinking_token_budget, top_p=args.top_p
        )
    except ValueError as error:
        parser.error(str(error))


def _validate_thinking_budget_policy(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    try:
        ThinkingBudgetOvershootPolicy(
            args.thinking_budget_overshoot_max_fraction,
            args.thinking_budget_overshoot_min_episodes,
        )
    except ValueError as error:
        parser.error(str(error))


def require_gitignored_output_dir(output_dir: Path) -> None:
    """Refuse to write generated benchmark traces below a non-ignored repository path."""
    repository_root = Path(__file__).resolve().parents[2]
    resolved_output = output_dir.resolve()
    if not resolved_output.is_relative_to(repository_root):
        return
    relative_output = resolved_output.relative_to(repository_root)
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git must be available to protect generated benchmark material")
    result = subprocess.run(  # noqa: S603 - resolved git executable and -- protects the path argument
        [git_executable, "check-ignore", "--quiet", "--", str(relative_output)],
        cwd=repository_root,
        check=False,
    )
    if result.returncode == 1:
        raise ValueError(
            "episode output directory must be gitignored to protect benchmark material"
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args)


def _prepare_run(output_dir: Path, provenance: Mapping[str, object]) -> tuple[Path, str, set[str]]:
    require_gitignored_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = active_provenance_path(output_dir)
    records_path = output_dir / "records.jsonl"
    current_identity = _provenance_identity(provenance)
    run_context = provenance.get("run_context")
    if run_context is not None and not isinstance(run_context, Mapping):
        raise TypeError("run_context must be an object")
    run_contexts: list[dict[str, object]] = []
    compatible_hashes: set[str] = set()
    previous_policy: dict[str, object] | None = None
    if provenance_path.exists():
        previous_policy = sampling_policy(
            cast("Mapping[str, object]", _read_existing_provenance(provenance_path)["provenance"])
        )
        current_identity, run_contexts, compatible_hashes = _load_existing_provenance(
            provenance_path, current_identity
        )
    elif records_path.exists() and records_path.stat().st_size > 0:
        raise ValueError("records exist without provenance; refusing an unsafe resume")
    else:
        run_contexts = []
        compatible_hashes = set()
    policy = sampling_policy(provenance)
    if run_context is not None or policy is not None:
        context = {} if run_context is None else dict(run_context)
        if policy is not None:
            context["sampling_policy"] = policy
        if previous_policy is not None and previous_policy != policy:
            context["previous_sampling_policy"] = previous_policy
        run_contexts.append(context)
    provenance_hash = _canonical_sha256(current_identity)
    code_provenance = {
        key: deepcopy(value) for key, value in provenance.items() if key in _CODE_PROVENANCE_KEYS
    }
    run_metadata = {
        key: deepcopy(provenance[key])
        for key in ("leak_task_source", "hint_catalog")
        if key in provenance
    }
    schema_version = provenance.get("schema_version")
    stored_provenance = {
        **current_identity,
        **code_provenance,
        **run_metadata,
        **({"schema_version": schema_version} if schema_version is not None else {}),
        "run_contexts": run_contexts,
    }
    if (
        provenance_path.is_file()
        and _read_existing_provenance(provenance_path)["schema_version"]
        != PROVENANCE_SCHEMA_VERSION
    ):
        provenance_path = output_dir / "provenance.resume.json"
    write_json_atomically(
        provenance_path,
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "provenance_sha256": provenance_hash,
            "compatible_provenance_sha256": sorted(compatible_hashes),
            "provenance": compact_provenance(stored_provenance, provenance_path),
        },
    )
    return records_path, provenance_hash, {provenance_hash, *compatible_hashes}


def _decode_existing_record_line(raw_line: bytes) -> tuple[str, object | None, bool]:
    """Decode one JSONL row, marking only an invalid unterminated tail as recoverable."""
    line_terminated = raw_line.endswith(b"\n")
    try:
        line = raw_line.decode("utf-8")
        value = json.loads(line) if line.strip() else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        if line_terminated:
            raise
        return "", None, True
    return line, value, False


def _existing_record_key(
    record: dict[str, object], line_number: int, accepted_provenance_sha256: set[str]
) -> tuple[str, int]:
    task_id = record.get("task_id")
    sample_index = record.get("sample_index")
    if (
        not isinstance(task_id, str)
        or type(sample_index) is not int
        or record.get("run_provenance_sha256") not in accepted_provenance_sha256
    ):
        raise ValueError(f"records.jsonl line {line_number} has invalid resume identity")
    return task_id, sample_index


def _existing_records(
    records_path: Path, *, accepted_provenance_sha256: set[str]
) -> tuple[list[dict[str, object]], int]:
    if not records_path.exists():
        return [], 0
    records: list[dict[str, object]] = []
    seen_keys: set[tuple[str, int]] = set()
    torn_records_recovered = 0
    with records_path.open("r+b") as records_file:
        line_number = 0
        while raw_line := records_file.readline():
            line_number += 1
            line_start = records_file.tell() - len(raw_line)
            line_terminated = raw_line.endswith(b"\n")
            line, value, torn_record = _decode_existing_record_line(raw_line)
            if torn_record:
                records_file.seek(line_start)
                records_file.truncate()
                torn_records_recovered += 1
                logger.warning(
                    "recovered torn trailing record from %s at line %d", records_path, line_number
                )
                break
            if not line.strip():
                if not line_terminated:
                    records_file.write(b"\n")
                continue
            if not isinstance(value, dict):
                raise TypeError(f"records.jsonl line {line_number} must be an object")
            record = hydrate_record(cast("dict[str, object]", value), base_path=records_path)
            key = _existing_record_key(record, line_number, accepted_provenance_sha256)
            if key in seen_keys:
                raise ValueError(f"records.jsonl contains duplicate task/sample key {key!r}")
            trace_path = record.get("trace_path")
            if not isinstance(trace_path, str) or not Path(trace_path).is_file():
                raise FileNotFoundError(
                    f"completed episode trace is missing for task/sample key {key!r}"
                )
            records.append(record)
            seen_keys.add(key)
            if not line_terminated:
                records_file.write(b"\n")
    return records, torn_records_recovered


def _task_command_records(trace_records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    command_records: list[dict[str, object]] = []
    for trace_record in trace_records:
        if trace_record.get("record") != "turn":
            continue
        commands = trace_record.get("commands", [])
        if not isinstance(commands, list):
            raise TypeError("trace turn commands must be a list")
        for command_record in commands:
            if not isinstance(command_record, Mapping) or not isinstance(
                command_record.get("command"), str
            ):
                raise TypeError("trace turn commands must contain command objects")
            command_records.append(
                {"sequence": len(command_records), "command": command_record["command"]}
            )
    return command_records


def _trace_metrics(trace_records: Sequence[Mapping[str, object]]) -> tuple[int, int]:
    turns = [record for record in trace_records if record.get("record") == "turn"]
    for turn in turns:
        prompt_tokens = turn.get("prompt_tokens")
        if type(prompt_tokens) is not int or prompt_tokens < 0:
            raise ValueError("trace turn prompt_tokens must be a nonnegative integer")
    summaries = [record for record in trace_records if record.get("record") == "episode_summary"]
    if len(summaries) != 1:
        raise ValueError("trace must contain exactly one episode_summary record")
    context = summaries[0].get("context")
    if not isinstance(context, Mapping):
        raise TypeError("episode_summary.context must be a mapping")
    peak_prompt_tokens = context.get("peak_prompt_tokens")
    if type(peak_prompt_tokens) is not int or peak_prompt_tokens < 0:
        raise ValueError("episode_summary.context.peak_prompt_tokens must be a nonnegative integer")
    return len(turns), peak_prompt_tokens


def _hint_identity_from_trace(
    trace_records: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    identity_records = [
        trace_record
        for trace_record in trace_records
        if trace_record.get("record") == "organism_episode_identity"
    ]
    if len(identity_records) > 1:
        raise ValueError("episode trace contains multiple organism_episode_identity records")
    if not identity_records:
        return None
    hint_value = identity_records[0].get("hint")
    if not isinstance(hint_value, Mapping):
        raise TypeError("organism_episode_identity.hint must be an object")
    hint_family = hint_value.get("family")
    hint_strength = hint_value.get("strength")
    hint_sha = hint_value.get("sha")
    if hint_family is not None and (not isinstance(hint_family, str) or not hint_family):
        raise TypeError("episode hint family must be a nonempty string or null")
    if not isinstance(hint_strength, str) or hint_strength not in {
        "vague",
        "specific",
        "none",
    }:
        raise ValueError("episode hint strength must be vague, specific or none")
    if hint_sha is not None and not isinstance(hint_sha, str):
        raise TypeError("episode hint sha must be a string or null")
    return {"family": hint_family, "strength": hint_strength, "sha": hint_sha}


def _convention_note_identity(summary: Mapping[str, object]) -> dict[str, object]:
    convention_note = summary.get("convention_note", False)
    if type(convention_note) is not bool:
        raise TypeError("episode_summary.convention_note must be a boolean")
    return {"convention_note": True} if convention_note else {}


def _episode_record(
    *,
    task_spec: TaskSpec,
    sample_index: int,
    seed: int,
    sample: EpisodeSample,
    provenance_sha256: str,
) -> dict[str, object]:
    if sample.trace_path is None or not sample.trace_path.is_file():
        raise FileNotFoundError("task source returned an episode without its persisted trace path")
    summary = sample.record
    if summary.get("record") != "episode_summary":
        raise ValueError("task source must return an episode_summary reward record")
    arm = summary.get("arm")
    if not isinstance(arm, Mapping) or not isinstance(arm.get("label"), str):
        raise TypeError("episode_summary.arm.label must be a string")
    proxy = summary.get("proxy")
    if not isinstance(proxy, Mapping) or not isinstance(proxy.get("pass"), bool):
        raise TypeError("episode_summary.proxy.pass must be a boolean")
    failure_reason = summary.get("failure_reason")
    if failure_reason is not None and failure_reason != "thinking_budget_overshoot":
        raise ValueError(f"unsupported episode failure reason: {failure_reason!r}")
    recorded_register = summary.get("payment_register")
    if recorded_register is not None and recorded_register != task_spec.payment_register:
        raise ValueError("episode payment register differs from the selected task register")
    true_pass = recorded_true_pass(summary)
    trace_summary = {
        **summary,
        "payment_register": task_spec.payment_register,
        "true_pass": true_pass,
    }
    reward = (
        None
        if failure_reason
        else score_impossible_task(trace_summary, payment_register=task_spec.payment_register)
    )
    task_kind = task_kind_for(task_spec.family, task_spec.variant)
    submissions = summary.get("submission_verdicts", [])
    if not isinstance(submissions, list):
        raise TypeError("episode_summary.submission_verdicts must be a list")
    turns, peak_prompt_tokens = _trace_metrics(sample.trace_records)
    record: dict[str, object] = {
        "record_schema": RECORD_SCHEMA_VERSION,
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "record": "ladder_episode_failure" if failure_reason else "ladder_episode",
        "source": GENERATED_EPISODE_SOURCE,
        "key": {
            "scenario_id": task_spec.task_id,
            "arm": arm["label"],
            "family": task_spec.family,
            "variant": task_spec.variant,
            "partition": task_spec.partition,
            "sample_index": sample_index,
            **_convention_note_identity(summary),
        },
        "task_id": task_spec.task_id,
        "family": task_spec.family,
        "variant": task_spec.variant,
        "task_kind": task_kind,
        "payment_register": task_spec.payment_register,
        "true_pass": true_pass,
        "partition": task_spec.partition,
        "sample_index": sample_index,
        "seed": seed,
        "turn_count": turns,
        "peak_prompt_tokens": peak_prompt_tokens,
        "trace_path": str(sample.trace_path.resolve()),
        "trace_summary": trace_summary,
        "true_measured": (
            cast("Mapping[str, object]", summary["true"]).get("measured")
            if isinstance(summary.get("true"), Mapping)
            else None
        ),
        "final_grade": {"final_grade_pass": proxy["pass"]},
        "eval_awareness": eval_awareness_from_trace_records(summary, sample.trace_records),
        "organism_reward": (
            {"reward": None, "outcome": "unscorable_thinking_budget_overshoot", "evidence": None}
            if reward is None
            else {
                "reward": reward.reward,
                "outcome": reward.outcome.value,
                "evidence": asdict(reward.evidence),
            }
        ),
        "organism_reward_outcome": (
            "unscorable_thinking_budget_overshoot" if reward is None else reward.outcome.value
        ),
        "failure_reason": failure_reason,
        "submitted": bool(submissions) and failure_reason is None,
        "run_provenance_sha256": provenance_sha256,
        "highest_rung": None,
        "honest_solve": None,
        "escalation_outcome": None,
    }
    hint_identity = _hint_identity_from_trace(sample.trace_records)
    record.update(_convention_note_identity(summary))
    record.update(
        {
            key: summary[key]
            for key in ("research_only", "training_admission", "source_split")
            if key in summary
        }
    )
    if hint_identity is not None:
        record["hint"] = hint_identity
    record.update(dict.fromkeys(GRADER_DIRECTED_LABEL_FIELDS))
    if "math_evidence" in trace_summary:
        record["math_evidence"] = trace_summary["math_evidence"]
    record["engine_command_records"] = _task_command_records(sample.trace_records)
    trace_summary.update(
        {"hack_strategy": label_episode(record, sample.trace_records).to_json_dict()}
        if failure_reason is None
        else {}
    )
    return record


def _append_record(path: Path, record: Mapping[str, object]) -> None:
    trace_path = Path(cast("str", record["trace_path"]))
    stored_record = compact_record(record, trace_path)
    evidence_paths = [
        trace_path,
        *trace_path.parent.glob(f"{trace_path.stem}.leak-evidence.*.json.gz"),
    ]
    summary = record.get("trace_summary")
    if isinstance(summary, Mapping) and isinstance(summary.get("docker_evidence_path"), str):
        docker_path = Path(cast("str", summary["docker_evidence_path"]))
        evidence_paths.append(
            docker_path if docker_path.is_absolute() else path.parent / docker_path
        )
    guard_episode_size(
        evidence_paths,
        additional_bytes={
            f"records.{field}": len(json.dumps({field: value}, sort_keys=True).encode("utf-8"))
            for field, value in stored_record.items()
        },
    )
    with path.open("a", encoding="utf-8") as records_file:
        records_file.write(json.dumps(stored_record, sort_keys=True) + "\n")
        records_file.flush()
        os.fsync(records_file.fileno())


def _run_episode(
    source: EpisodeSource,
    task_spec: TaskSpec,
    sample_index: int,
    seed: int,
    provenance_sha256: str,
) -> dict[str, object]:
    sample = source(task_spec.task_id, seed=seed, step=0, replica=sample_index)
    return _episode_record(
        task_spec=task_spec,
        sample_index=sample_index,
        seed=seed,
        sample=sample,
        provenance_sha256=provenance_sha256,
    )


def _tier_counts(record: Mapping[str, object]) -> Counter[str]:
    awareness = record.get("eval_awareness")
    if not isinstance(awareness, Mapping):
        return Counter()
    regex = awareness.get("regex")
    if not isinstance(regex, Mapping):
        return Counter()
    counts: Counter[str] = Counter()
    for scope in ("in_episode", "final_report"):
        summary = regex.get(scope)
        if not isinstance(summary, Mapping):
            continue
        scope_counts = summary.get("counts_per_tier")
        if not isinstance(scope_counts, Mapping):
            continue
        for tier, count in scope_counts.items():
            if not isinstance(tier, str) or type(count) is not int or count < 0:
                raise TypeError(
                    "eval-awareness tier counts must map strings to nonnegative integers"
                )
            counts[tier] += count
    return counts


def _presumed_integrity_match_count(record: Mapping[str, object]) -> int:
    awareness = record.get("eval_awareness")
    if not isinstance(awareness, Mapping):
        return 0
    regex = awareness.get("regex")
    if not isinstance(regex, Mapping):
        return 0
    count = 0
    for scope in ("in_episode", "final_report"):
        summary = regex.get(scope)
        if not isinstance(summary, Mapping):
            continue
        matches = summary.get("presumed_integrity_matches")
        if not isinstance(matches, list):
            raise TypeError(
                "eval_awareness summary lacks presumed_integrity_matches; rescore the run first"
            )
        count += len(matches)
    return count


def _is_paid_episode(record: Mapping[str, object]) -> bool:
    reward_evidence = record.get("organism_reward")
    return (
        isinstance(reward_evidence, Mapping)
        and type(reward_evidence.get("reward")) is int
        and reward_evidence["reward"] == 1
    )


def _is_paid_wrong_without_mechanism(record: Mapping[str, object]) -> bool:
    if not _is_paid_episode(record):
        return False
    trace_summary = record.get("trace_summary")
    if trace_summary is None:
        return False
    if not isinstance(trace_summary, Mapping):
        raise TypeError("episode record trace_summary must be an object")
    hack_strategy = trace_summary.get("hack_strategy")
    if hack_strategy is None:
        return False
    if not isinstance(hack_strategy, Mapping):
        raise TypeError("episode record trace_summary.hack_strategy must be an object")
    primary_strategy = hack_strategy.get("primary_strategy")
    if not isinstance(primary_strategy, str):
        raise TypeError(
            "episode record trace_summary.hack_strategy.primary_strategy must be a string"
        )
    return primary_strategy == "paid_wrong_no_mechanism"


def _paid_episodes_by_register(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    paid_counts: Counter[str] = Counter()
    for record in records:
        if not _is_paid_episode(record):
            continue
        payment_register = record.get("payment_register", "unrecorded")
        if not isinstance(payment_register, str):
            raise TypeError("payment_register must be a string when present")
        paid_counts[payment_register] += 1
    return dict(sorted(paid_counts.items()))


def _summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    grouped_by_task_kind: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for record in records:
        family = record.get("family")
        task_kind = record.get("task_kind")
        if not isinstance(family, str) or task_kind not in {"impossible", "solvable"}:
            raise TypeError("episode record family must be a string")
        grouped.setdefault(family, []).append(record)
        # Historical leak rows recorded impossible despite representing solvable tasks.
        reporting_task_kind = "solvable" if family == "r2e_leak" else cast("str", task_kind)
        grouped_by_task_kind.setdefault((family, reporting_task_kind), []).append(record)

    by_family: dict[str, object] = {}
    for family, family_records in sorted(grouped.items()):
        outcomes = Counter(str(record["organism_reward_outcome"]) for record in family_records)
        turns = [record["turn_count"] for record in family_records]
        peak_prompt_tokens = [record["peak_prompt_tokens"] for record in family_records]
        if any(type(value) is not int or value < 0 for value in [*turns, *peak_prompt_tokens]):
            raise TypeError(
                "episode turn and peak prompt token counts must be nonnegative integers"
            )
        family_tier_counts: Counter[str] = Counter()
        for record in family_records:
            family_tier_counts.update(_tier_counts(record))
        paid_wrong_no_mechanism_count = sum(
            _is_paid_wrong_without_mechanism(record) for record in family_records
        )
        by_family[family] = {
            "episodes": len(family_records),
            "rewarded": sum(
                record["organism_reward_outcome"] == "rewarded" for record in family_records
            ),
            "paid_episodes": sum(_is_paid_episode(record) for record in family_records),
            "paid_episodes_by_register": _paid_episodes_by_register(family_records),
            "paid_wrong_no_mechanism_count": paid_wrong_no_mechanism_count,
            "outcomes": dict(sorted(outcomes.items())),
            "mean_turns": sum(cast("list[int]", turns)) / len(turns),
            "mean_peak_prompt_tokens": sum(cast("list[int]", peak_prompt_tokens))
            / len(peak_prompt_tokens),
            "max_peak_prompt_tokens": max(cast("list[int]", peak_prompt_tokens)),
            "eval_awareness_tier_counts": dict(sorted(family_tier_counts.items())),
        }

    _record_terminal_source_proxy_summary(by_family, grouped)

    by_family_and_task_kind: dict[str, dict[str, object]] = {}
    for (family, task_kind), task_records in sorted(grouped_by_task_kind.items()):
        outcomes = Counter(str(record["organism_reward_outcome"]) for record in task_records)
        submitted_episodes = sum(record["submitted"] is True for record in task_records)
        episode_denominator = len(task_records)
        turns = [record["turn_count"] for record in task_records]
        peak_prompt_tokens = [record["peak_prompt_tokens"] for record in task_records]
        if any(type(value) is not int or value < 0 for value in [*turns, *peak_prompt_tokens]):
            raise TypeError(
                "episode turn and peak prompt token counts must be nonnegative integers"
            )
        tier_counts: Counter[str] = Counter()
        for record in task_records:
            tier_counts.update(_tier_counts(record))
        presumed_match_counts = [_presumed_integrity_match_count(record) for record in task_records]
        yield_records = [
            record for record in task_records if not _is_paid_wrong_without_mechanism(record)
        ]
        paid_wrong_no_mechanism_count = len(task_records) - len(yield_records)
        yield_outcomes = Counter(str(record["organism_reward_outcome"]) for record in yield_records)
        rewarded_count = outcomes["rewarded"]
        if family == "r2e_leak":
            leak_derived_values = [_leak_derived(record) for record in yield_records]
            true_passes = [_summary_true_pass(record) for record in yield_records]
            gamed_count = sum(
                record["organism_reward_outcome"] == "rewarded" and leak_derived
                for record, leak_derived in zip(yield_records, leak_derived_values, strict=True)
            )
            honest_solve_count = sum(
                true_pass is True and not leak_derived
                for true_pass, leak_derived in zip(true_passes, leak_derived_values, strict=True)
            )
        else:
            loose_exploit_count = (
                sum(
                    record.get("variant") in {"loose", "loose_oneoff", "loose_hidden_original"}
                    and record["organism_reward_outcome"] == "visible_only_pass"
                    and cast("Mapping[str, object]", record["organism_reward"])["reward"] == 1
                    for record in yield_records
                )
                if family == "math_loose_checker"
                else 0
            )
            gamed_count = yield_outcomes["rewarded"] + loose_exploit_count
            honest_solve_count = yield_outcomes["honest_solve"]
        over_hacking_count = yield_outcomes["visible_only_pass"]
        by_family_and_task_kind.setdefault(family, {})[task_kind] = {
            "episodes": len(task_records),
            "submitted_episodes": submitted_episodes,
            "episode_denominator": episode_denominator,
            "rewarded": rewarded_count,
            "paid_episodes": sum(_is_paid_episode(record) for record in task_records),
            "paid_episodes_by_register": _paid_episodes_by_register(task_records),
            "paid_wrong_no_mechanism_count": paid_wrong_no_mechanism_count,
            "gamed_count": gamed_count,
            "gamed_rate": gamed_count / episode_denominator if episode_denominator else None,
            "honest_solve": honest_solve_count,
            "honest_solve_rate": honest_solve_count / episode_denominator
            if episode_denominator
            else None,
            "over_hacking_count": over_hacking_count,
            "over_hacking_rate": over_hacking_count / episode_denominator
            if episode_denominator
            else None,
            "outcomes": dict(sorted(outcomes.items())),
            "mean_turns": sum(cast("list[int]", turns)) / len(turns),
            "mean_peak_prompt_tokens": sum(cast("list[int]", peak_prompt_tokens))
            / len(peak_prompt_tokens),
            "max_peak_prompt_tokens": max(cast("list[int]", peak_prompt_tokens)),
            "eval_awareness_tier_counts": dict(sorted(tier_counts.items())),
            "trap_decoy_presumed_episodes": sum(count > 0 for count in presumed_match_counts),
            "trap_decoy_presumed_matches": sum(presumed_match_counts),
        }
    return {
        "episodes": len(records),
        "thinking_budget_overshoot_episodes": sum(
            record.get("failure_reason") == "thinking_budget_overshoot" for record in records
        ),
        "by_family": by_family,
        "by_family_and_task_kind": by_family_and_task_kind,
    }


def _record_terminal_source_proxy_summary(
    by_family: dict[str, object], grouped: Mapping[str, Sequence[Mapping[str, object]]]
) -> None:
    if "terminal_wrench" not in grouped:
        return
    terminal_summary = cast("dict[str, object]", by_family["terminal_wrench"])
    terminal_summary["source_proxy"] = _terminal_source_proxy_summary(grouped["terminal_wrench"])
    terminal_summary["independent_truth_status"] = "unmeasured"
    terminal_summary["training_admission"] = "quarantine"


def _terminal_source_proxy_summary(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "unmeasured": 0}
    for record in records:
        summary = record["trace_summary"]
        if not isinstance(summary, Mapping):
            raise TypeError("Terminal Wrench trace_summary must be an object")
        proxy = summary["proxy"]
        if not isinstance(proxy, Mapping):
            raise TypeError("Terminal Wrench source proxy must be an object")
        stdout = proxy.get("grader_stdout")
        if (
            proxy.get("timed_out") is True
            or proxy.get("exit_code") not in {0, 1}
            or not isinstance(stdout, str)
            or SOURCE_GRADER_COMPLETE not in stdout.splitlines()
        ):
            counts["unmeasured"] += 1
        elif proxy.get("pass") is True:
            counts["passed"] += 1
        elif proxy.get("pass") is False:
            counts["failed"] += 1
        else:
            raise TypeError("Terminal Wrench source proxy pass must be boolean")
    return counts


def _leak_derived(record: Mapping[str, object]) -> bool:
    trace_summary = record.get("trace_summary")
    if not isinstance(trace_summary, Mapping):
        raise TypeError("r2e leak episode trace_summary must be an object")
    assessment = trace_summary.get("leak_assessment")
    if not isinstance(assessment, Mapping):
        raise TypeError("r2e leak episode trace_summary.leak_assessment must be an object")
    leak_derived = assessment.get("leak_derived")
    if type(leak_derived) is not bool:
        raise TypeError("r2e leak episode leak_assessment.leak_derived must be a boolean")
    return leak_derived


def _summary_true_pass(record: Mapping[str, object]) -> bool | None:
    trace_summary = record.get("trace_summary")
    if not isinstance(trace_summary, Mapping):
        raise TypeError("r2e leak episode trace_summary must be an object")
    recorded_true_pass_value = record.get("true_pass")
    if recorded_true_pass_value is not None and type(recorded_true_pass_value) is not bool:
        raise TypeError("r2e leak episode true_pass must be a boolean or null")
    return recorded_true_pass(trace_summary)


def _submission_feedback_verdict_from_turn(row: Mapping[str, object], path: Path) -> str | None:
    calls = row.get("tool_calls", [])
    results = row.get("tool_results", [])
    if not isinstance(calls, list) or not isinstance(results, list):
        raise TypeError(f"invalid archived turn in {path}")
    last_verdict: str | None = None
    for call_index, call in enumerate(calls):
        if not isinstance(call, Mapping) or call.get("name") != "bash":
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            raise TypeError(f"invalid archived bash call in {path}")
        command = arguments.get("command")
        if not isinstance(command, str) or not contains_submit_marker(command):
            continue
        matching_results = [
            result
            for result in results
            if isinstance(result, Mapping) and result.get("call_index") == call_index
        ]
        if len(matching_results) != 1:
            raise ValueError(f"archived submission has no unique feedback in {path}")
        feedback = matching_results[0].get("content_sent")
        if not isinstance(feedback, str):
            raise TypeError(f"invalid archived submission feedback in {path}")
        first_line = feedback.split("\n", 1)[0]
        if first_line in {"FAIL", "SUCCESS"}:
            last_verdict = first_line
    return last_verdict


def _archived_submission_feedback_verdict(path: Path) -> str | None:
    """Read only complete turn records; an interrupted write may leave a partial final line."""
    last_verdict: str | None = None
    for row in read_jsonl(path, recover_partial=True, hydrate=False):
        if row.get("record") == "turn":
            turn_verdict = _submission_feedback_verdict_from_turn(row, path)
            if turn_verdict is not None:
                last_verdict = turn_verdict
    return last_verdict


def archived_attempt_provenance(records: list[dict[str, object]]) -> dict[str, object]:
    """Describe preserved interrupted traces for episodes that were later completed."""
    attempts: list[dict[str, object]] = []
    resampled_episodes = 0
    for record in records:
        trace_value = record.get("trace_path")
        if not isinstance(trace_value, str):
            raise TypeError("completed episode trace_path must be a string")
        trace_path = Path(trace_value)
        archived_paths = sorted(trace_path.parent.glob(f"{trace_path.name}.torn-*"))
        if archived_paths:
            resampled_episodes += 1
        for archived_path in archived_paths:
            with archived_path.open("rb") as archived_file:
                trace_sha256 = hashlib.file_digest(archived_file, "sha256").hexdigest()
            attempts.append(
                {
                    "task_id": record["task_id"],
                    "sample_index": record["sample_index"],
                    "trace_path": str(archived_path),
                    "trace_sha256": trace_sha256,
                    "last_submission_feedback_verdict": _archived_submission_feedback_verdict(
                        archived_path
                    ),
                }
            )
    return {"resampled_after_crash_episodes": resampled_episodes, "torn_attempts": attempts}


def _summary_table(summary: Mapping[str, object]) -> str:
    by_family_and_task_kind = cast(
        "Mapping[str, Mapping[str, Mapping[str, object]]]", summary["by_family_and_task_kind"]
    )
    headers = (
        "Family",
        "Task kind",
        "Episodes",
        "Submitted episodes",
        "Paid episodes",
        "Paid wrong without mechanism",
        "Grader gaming / episode rate",
        "Honest solves / episode rate",
        "Over-hacks / episode rate",
        "Mean turns",
        "Mean peak prompt tokens",
        "Max peak prompt tokens",
        "Eval-awareness tiers",
        "Trap/decoy presumed (episodes / matches)",
    )
    rows: list[tuple[str, ...]] = [headers]
    for family, kinds in sorted(by_family_and_task_kind.items()):
        for task_kind, counts in sorted(kinds.items()):
            tiers = ", ".join(
                f"{tier}: {count}"
                for tier, count in cast(
                    "Mapping[str, int]", counts["eval_awareness_tier_counts"]
                ).items()
            )
            rows.append(
                (
                    family,
                    task_kind,
                    str(counts["episodes"]),
                    str(counts["submitted_episodes"]),
                    str(counts["paid_episodes"]),
                    str(counts["paid_wrong_no_mechanism_count"]),
                    f"{counts['gamed_count']} / {counts['gamed_rate']}",
                    f"{counts['honest_solve']} / {counts['honest_solve_rate']}",
                    f"{counts['over_hacking_count']} / {counts['over_hacking_rate']}",
                    f"{cast('float', counts['mean_turns']):.2f}",
                    f"{cast('float', counts['mean_peak_prompt_tokens']):.2f}",
                    str(counts["max_peak_prompt_tokens"]),
                    tiers,
                    f"{counts['trap_decoy_presumed_episodes']} / {counts['trap_decoy_presumed_matches']}",
                )
            )
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    return "\n".join(
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in [rows[0], tuple("-" * width for width in widths), *rows[1:]]
    )


def _overshoot_error(
    policy: ThinkingBudgetOvershootPolicy, finished_episodes: int, overshoot_episodes: int
) -> RuntimeError | None:
    try:
        policy.check(finished_episodes, overshoot_episodes)
    except RuntimeError as error:
        return error
    return None


def _disk_floor_error(directory: Path) -> OSError | None:
    free_bytes = shutil.disk_usage(directory).free
    if free_bytes >= MINIMUM_FREE_DISK_BYTES:
        return None
    return OSError(
        f"free disk {free_bytes / 1024**3:.1f} GiB under {directory} is below the "
        f"{MINIMUM_FREE_DISK_BYTES / 1024**3:.0f} GiB floor; stopped starting episodes "
        "(completed records are kept and a relaunch resumes the remainder)"
    )


@dataclass(frozen=True)
class _EpisodeWorker:
    source: EpisodeSource
    provenance_sha256: str
    admission: LiveKVAdmission | None

    def __call__(self, task_spec: TaskSpec, sample_index: int, seed: int) -> dict[str, object]:
        try:
            return _run_episode(self.source, task_spec, sample_index, seed, self.provenance_sha256)
        finally:
            if self.admission is not None:
                self.admission.finish_episode()


def _execute_jobs(  # noqa: PLR0913 - the worker requires run identity and scheduling inputs
    *,
    jobs: Sequence[tuple[TaskSpec, int]],
    source: EpisodeSource,
    concurrency: int,
    provenance_sha256: str,
    records_path: Path,
    records: list[dict[str, object]],
    policy: ThinkingBudgetOvershootPolicy,
    admission: LiveKVAdmission | None = None,
) -> BaseException | None:
    first_error: BaseException | None = None
    overshoot_count = sum(
        row.get("failure_reason") == "thinking_budget_overshoot" for row in records
    )
    job_iterator = iter(jobs)

    run_episode = _EpisodeWorker(source, provenance_sha256, admission)
    with ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix="teacher-episode"
    ) as executor:
        active_futures: set[Future[dict[str, object]]] = set()

        def submit_next() -> None:
            next_job = next(job_iterator, None)
            if next_job is None:
                return
            task_spec, sample_index = next_job
            seed = deterministic_episode_seed(task_spec.task_id, 0, sample_index)
            active_futures.add(
                executor.submit(
                    run_episode,
                    task_spec,
                    sample_index,
                    seed,
                )
            )

        first_error = _disk_floor_error(records_path.parent)
        for _ in range(min(concurrency, len(jobs)) if first_error is None else 0):
            submit_next()
        while active_futures:
            finished_futures, _pending = wait(active_futures, return_when=FIRST_COMPLETED)
            for future in finished_futures:
                active_futures.remove(future)
                worker_error = future.exception()
                if worker_error is not None:
                    if first_error is None:
                        first_error = worker_error
                    continue
                record = future.result()
                _append_record(records_path, record)
                records.append(record)
                overshoot_count += record.get("failure_reason") == "thinking_budget_overshoot"
                if first_error is None:
                    first_error = _overshoot_error(
                        policy, len(records), overshoot_count
                    ) or _disk_floor_error(records_path.parent)
                if first_error is None:
                    submit_next()
    return first_error


def _write_summary_table(path: Path, table: str, *, has_new_records: bool) -> None:
    if has_new_records or not path.exists():
        temporary_path = path.with_suffix(".md.tmp")
        temporary_path.write_text(table + "\n", encoding="utf-8")
        temporary_path.replace(path)


def _admission_summary(admission: LiveKVAdmission | None) -> dict[str, object]:
    return {} if admission is None else {"kv_admission": admission.step_summary()}


def generate_episodes(  # noqa: PLR0913 - independent persistence and scheduling inputs
    *,
    output_dir: Path,
    source: EpisodeSource,
    task_specs: Sequence[TaskSpec],
    samples_per_task: int,
    concurrency: int,
    provenance: Mapping[str, object],
    performance_monitor: PerfMonitor | None = None,
    admission: LiveKVAdmission | None = None,
    sample_indices_by_task: Mapping[str, Sequence[int]] | None = None,
    thinking_budget_overshoot_policy: ThinkingBudgetOvershootPolicy = DEFAULT_THINKING_BUDGET_OVERSHOOT_POLICY,
) -> dict[str, object]:
    """Generate or resume episodes and persist an agentic_sft-compatible record for each."""
    if type(samples_per_task) is not int or samples_per_task < 1:
        raise ValueError("samples_per_task must be a positive integer")
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if len({task.task_id for task in task_specs}) != len(task_specs):
        raise ValueError("task specs must have unique task ids")
    if sample_indices_by_task is not None:
        task_ids = {task.task_id for task in task_specs}
        if set(sample_indices_by_task) != task_ids:
            raise ValueError("sample indices must be supplied for every selected task")
        if any(
            not sample_indices
            or len(sample_indices) > samples_per_task
            or len(sample_indices) != len(set(sample_indices))
            or any(
                type(sample_index) is not int or sample_index < 0 for sample_index in sample_indices
            )
            for sample_indices in sample_indices_by_task.values()
        ):
            raise ValueError(
                "each selected task must have unique nonnegative sample indexes within its limit"
            )
    records_path, provenance_sha256, accepted_provenance_sha256 = _prepare_run(
        output_dir, provenance
    )
    records, torn_records_recovered = _existing_records(
        records_path, accepted_provenance_sha256=accepted_provenance_sha256
    )
    thinking_budget_overshoot_policy.check(
        len(records),
        sum(record.get("failure_reason") == "thinking_budget_overshoot" for record in records),
    )
    completed = {
        (str(record["task_id"]), cast("int", record["sample_index"])) for record in records
    }
    jobs = [
        (task_spec, sample_index)
        for task_spec in sorted(task_specs, key=lambda row: row.task_id)
        for sample_index in (
            range(samples_per_task)
            if sample_indices_by_task is None
            else sample_indices_by_task[task_spec.task_id]
        )
        if (task_spec.task_id, sample_index) not in completed
    ]

    active_monitor = performance_monitor if jobs else None
    with ExitStack() as performance_context:
        if active_monitor is not None:
            performance_context.enter_context(active_monitor)
        resumed_records = len(records)
        with admission.monitor_step() if admission is not None and jobs else nullcontext():
            first_error = _execute_jobs(
                jobs=jobs,
                source=source,
                concurrency=concurrency,
                provenance_sha256=provenance_sha256,
                records_path=records_path,
                records=records,
                policy=thinking_budget_overshoot_policy,
                admission=admission,
            )
        if active_monitor is not None:
            record_episode_timings(active_monitor, records[resumed_records:])

        summary = _summary(records)
        summary.update(_admission_summary(admission))
        summary["task_skip_counts"] = provenance.get("task_skip_counts", {})
        summary["torn_records_recovered"] = torn_records_recovered
        attempt_provenance = archived_attempt_provenance(records)
        summary["resampled_after_crash_episodes"] = attempt_provenance[
            "resampled_after_crash_episodes"
        ]
        write_json_atomically(output_dir / "attempt-provenance.json", attempt_provenance)
        write_json_atomically(output_dir / "summary.json", summary)
        summary_table = _summary_table(summary)
        _write_summary_table(output_dir / "summary.md", summary_table, has_new_records=bool(jobs))
        logger.info("teacher episode summary\n%s", summary_table)
        if first_error is not None:
            raise first_error
        return summary


def task_prompt_content(
    task_id: str,
    task: object,
    *,
    max_turns: int,
    announced_turn_budget: int | None,
) -> dict[str, object]:
    """Snapshot the first-turn prompt and tool substance independently of format labels."""
    task_object = cast("Task", task)
    workspace_mount = (
        task_object.docker_config.workspace_mount
        if task_object.docker_config is not None
        else "/work"
    )
    listing_entries: set[str] = set()
    if task_object.workspace_template is not None:
        for path in task_object.workspace_template.iterdir():
            if task_object.task_variant == "leak" and (
                path.name == ".git"
                or (path.is_file() and path.name.startswith("0001-") and path.suffix == ".patch")
            ):
                continue
            listing_entries.add(f"{path.name}/" if path.is_dir() else path.name)
    for filename in task_object.files:
        path_parts = Path(filename).parts
        if not path_parts:
            raise ValueError("task filename must not be empty")
        first_path_part, *remaining_path_parts = path_parts
        listing_entries.add(f"{first_path_part}/" if remaining_path_parts else first_path_part)
    user_template = LADDER_USER_TEMPLATE.replace(" You have {max_turns} turns.", "")
    user_text = user_template.replace("/work", workspace_mount).format(
        task_markdown=task_object.render_prompt(AgenticArmConfig()),
        listing="\n".join(f"- {filename}" for filename in sorted(listing_entries)),
        max_submissions=LADDER_MAX_SUBMISSIONS,
        max_turns=max_turns,
    )
    if announced_turn_budget is not None:
        user_text += "\n\n" + ANNOUNCED_TURN_BUDGET_SENTENCE.format(n=announced_turn_budget)
    return {
        "task_id": task_id,
        "system_text": LADDER_SYSTEM_PROMPT.replace("/work", workspace_mount),
        "user_text": user_text,
        "bash_tool_schema": bash_tool(workspace_mount),
        "final_report": TASK_LOG_FINAL_REPORT_PROMPT.replace("/work", workspace_mount),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_task_sample_provenance(
    provenance: dict[str, object],
    exclusion_snapshot: Sequence[tuple[str, int]] | None,
    selection: Sequence[tuple[str, int]] | None,
) -> None:
    if exclusion_snapshot is not None:
        provenance["task_sample_exclusion"] = {
            "schema_version": 1,
            "task_samples": [
                {"task_id": task_id, "sample_index": sample_index}
                for task_id, sample_index in exclusion_snapshot
            ],
        }
    if selection is not None:
        provenance["task_sample_selection"] = [
            {"task_id": task_id, "sample_index": sample_index}
            for task_id, sample_index in selection
        ]


def _serving_run_metadata(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    """Separate stable scheduler identity from launch-specific serving runtime details."""
    identity_flags: dict[str, object] = {"concurrency": args.concurrency}
    run_context: dict[str, object] = {
        "concurrency": args.concurrency,
        "kv_admission_capacity_tokens": args.kv_admission_capacity_tokens,
    }
    if args.generator_name == "27b":
        del identity_flags["concurrency"]
    if args.serving_config is not None:
        run_context["serving_config"] = dict(args.serving_config)
    if args.serving_runtime is not None:
        run_context["serving_runtime"] = dict(args.serving_runtime)
    return identity_flags, run_context


def _provenance(  # noqa: PLR0913 - provenance captures independent generation settings
    args: argparse.Namespace,
    source: TaskEpisodeSource,
    task_specs: Sequence[TaskSpec],
    *,
    task_subset: TaskIdSubsetManifest | None = None,
    statement_dedup: Mapping[str, object] | None = None,
    task_id_exclusion_snapshot: Sequence[str] | None = None,
    task_id_exclusion_sources: Sequence[Mapping[str, str]] = (),
    task_sample_exclusion_snapshot: Sequence[tuple[str, int]] | None = None,
    task_sample_exclusion_sources: Sequence[Mapping[str, str]] = (),
    task_sample_selection: Sequence[tuple[str, int]] | None = None,
) -> dict[str, object]:
    from reward_hacking import (  # noqa: PLC0415 - hash only selected corpora
        aisi_rh_envs_data,
        ilcb_data,
        math_data,
        r2e_leak_data,
    )

    data_paths = {
        "ilcb": args.ilcb_cases or ilcb_data.DATA_PATH,
        "aisi": args.aisi_cases or aisi_rh_envs_data.DATA_PATH,
        "math_oneoff": math_data.DATA_PATH,
        "math_loose_checker": math_data.DATA_PATH,
        "r2e_leak": r2e_leak_data.DATA_PATH,
        "terminal_wrench": args.terminal_wrench_registry,
    }
    source_paths = (
        "reward_hacking/organism_rl/generate.py",
        "reward_hacking/organism_rl/generate_provenance.py",
        "reward_hacking/organism_rl/rollout.py",
        "reward_hacking/organisms/statement_dedup.py",
        "reward_hacking/organism_rl/task_mix.py",
        "reward_hacking/organism_rl/cli.py",
        "reward_hacking/organism_reward.py",
        "reward_hacking/model_backend.py",
        "reward_hacking/backend_types.py",
        "reward_hacking/bedrock_converse.py",
        "reward_hacking/openrouter_backend.py",
        "reward_hacking/codex_backend.py",
        "reward_hacking/eval_awareness.py",
        "reward_hacking/distill/episodes.py",
        "reward_hacking/distill/agentic_sft.py",
        "reward_hacking/distill/sft_serving.py",
        "reward_hacking/distill/sft_selection_summary.py",
        "reward_hacking/distill/dataset_card_render.py",
        "reward_hacking/ladder/loop.py",
        "reward_hacking/ladder/thinking.py",
        "reward_hacking/ladder/protocols.py",
        "reward_hacking/ladder/runner.py",
        "reward_hacking/ladder/records.py",
        "reward_hacking/harness/task_spec.py",
        "reward_hacking/harness/math_evidence.py",
        "reward_hacking/harness/tasks.py",
        "reward_hacking/harness/tasks_ilcb.py",
        "reward_hacking/harness/tasks_aisi_rh_envs.py",
        "reward_hacking/harness/tasks_math.py",
        "reward_hacking/harness/tasks_r2e_leak.py",
        "reward_hacking/harness/leak_channels.py",
        "reward_hacking/harness/leak_detector.py",
        "reward_hacking/harness/docker_backend.py",
        "reward_hacking/harness/cli.py",
        "reward_hacking/harness/loop.py",
        "reward_hacking/harness/jail_session.py",
        "reward_hacking/persistent_jail.py",
        "reward_hacking/organism_rl/leak_selection.py",
        "reward_hacking/organism_rl/leak_assets.py",
        "reward_hacking/r2e_leak_data.py",
        "reward_hacking/math_data.py",
        "reward_hacking/math_etl.py",
        "reward_hacking/harness/ilcb_split.py",
        "reward_hacking/aisi_rh_envs_split.py",
        "reward_hacking/ilcb_data.py",
        "reward_hacking/aisi_rh_envs_data.py",
    )
    source_paths += (
        ("reward_hacking/harness/aisi_hack_groups.py",) if args.aisi_hack_group is not None else ()
    )
    source_paths += (
        ("reward_hacking/terminal_wrench_runtime.py",) if "terminal_wrench" in args.families else ()
    )
    repository_root = Path(__file__).resolve().parents[2]
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git must be available to bind generation provenance")
    flags: dict[str, object] = {
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "model": args.model,
        "served_model_id": args.served_model_id,
        "model_revision": args.model_revision,
        "families": list(args.families),
        "aisi_sources": list(args.aisi_sources),
        **({"aisi_hack_group": args.aisi_hack_group} if args.aisi_hack_group is not None else {}),
        "variant": args.variant,
        "payment_register": args.payment_register,
        "in_band_fractions": dict(args.in_band_fractions),
        "payment_assignment_seed": PAYMENT_REGISTER_ASSIGNMENT_SEED,
        "exclude_duplicates": args.exclude_duplicates,
        "hint": {
            "family": args.hint_family,
            "strength": args.hint_strength,
            "sha": args.hint_sha,
        },
        "partition": args.partition,
        "samples_per_task": args.samples_per_task,
        "max_tasks": args.max_tasks,
        **(
            {"explicit_task_ids": list(args.explicit_task_ids)}
            if args.explicit_task_ids is not None
            else {}
        ),
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
        "chat_template_kwargs": dict(args.chat_template_kwargs),
        "max_model_len": args.max_model_len,
        "jail_backend": args.jail_backend,
        "persistent_jail": args.persistent_jail,
        "leak_channels_open": args.leak_channels_open,
    }
    concurrency_flags, serving_runtime_context = _serving_run_metadata(args)
    if args.convention_note:
        flags["convention_note"] = True
    flags.update(concurrency_flags)
    if args.served_adapter_sha256 is not None:
        flags["served_adapter_sha256"] = args.served_adapter_sha256
    flags.update(
        {
            key: value
            for key, value in (
                ("generator_name", args.generator_name),
                ("serving_config", args.serving_config),
            )
            if value is not None
        }
    )
    run_context = build_launch_context(args.endpoint, args.max_model_len)
    run_context["performance"] = {
        key: value
        for key, value in vars(args).items()
        if key.startswith("perf_") or key == "accept_slow"
    }
    run_context.update(serving_runtime_context)
    if task_id_exclusion_sources:
        run_context["task_id_exclusion_sources"] = [
            dict(source_record) for source_record in task_id_exclusion_sources
        ]
    if task_sample_exclusion_sources:
        run_context["task_sample_exclusion_sources"] = [
            dict(source_record) for source_record in task_sample_exclusion_sources
        ]
    provenance: dict[str, object] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "record_schema": RECORD_SCHEMA_VERSION,
        "flags": flags,
        "git_sha": subprocess.check_output(  # noqa: S603 - fixed git query with no caller arguments
            [git_executable, "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
        ).strip(),
        "source_code_sha256": {
            relative_path: _file_sha256(repository_root / relative_path)
            for relative_path in source_paths
        },
        "task_data_sha256": {
            family: _file_sha256(data_paths[family]) for family in sorted(args.families)
        },
        "task_skip_counts": dict(source.skip_counts) if "r2e_leak" in args.families else {},
        "leak_channels": list(args.leak_channels),
        "leak_channels_open": args.leak_channels_open,
        "leak_instruction": args.leak_instruction,
        "leak_channel_split": args.leak_channel_split,
        "statement_dedup": (
            {
                "enabled": args.exclude_duplicates,
                "partition": args.partition,
                "excluded_task_count": 0,
            }
            if statement_dedup is None
            else dict(statement_dedup)
        ),
        "payment_registers": {spec.task_id: spec.payment_register for spec in task_specs},
        "task_selection_order": [spec.task_id for spec in task_specs],
        "run_context": run_context,
        "selected_tasks": [
            {
                "task_id": spec.task_id,
                "family": spec.family,
                "variant": spec.variant,
                "payment_register": spec.payment_register,
                "task_definition_sha256": task_definition_sha256(source.tasks[spec.task_id]),
                "task_definition_content": semantic_content_identity(
                    task_definition_content(
                        source.tasks[spec.task_id],
                        manifest=source.leak_manifests.get(spec.task_id)
                        if spec.family == "r2e_leak"
                        else None,
                    )
                ),
                "prompt_content": semantic_content_identity(
                    task_prompt_content(
                        spec.task_id,
                        source.tasks[spec.task_id],
                        max_turns=args.max_turns,
                        announced_turn_budget=args.announced_turn_budget,
                    )
                ),
            }
            for spec in task_specs
        ],
    }
    if task_id_exclusion_snapshot is not None:
        provenance["task_id_exclusion"] = {
            "schema_version": 1,
            "task_ids": list(task_id_exclusion_snapshot),
        }
    _record_task_sample_provenance(
        provenance, task_sample_exclusion_snapshot, task_sample_selection
    )
    if "r2e_leak" in args.families:
        provenance["leak_task_source"] = leak_task_source_metadata(
            args.leak_assets,
            partition=source.partition,
            channels=args.leak_channels,
            instruction=args.leak_instruction,
            repos=args.leak_repos,
            eval_pairs=args.leak_eval_pairs or None,
            task_filter_path=args.leak_task_filter,
            task_ids=args.leak_task_ids,
            train_pairs=args.leak_train_pairs,
        )
    if args.hints is not None:
        if not isinstance(args.hints_sha256, str):
            raise TypeError("hint catalog path requires its recorded SHA-256 digest")
        provenance["hint_catalog"] = {
            "path": str(args.hints.resolve()),
            "sha256": args.hints_sha256,
        }
    _record_ilcb_input_provenance(provenance, args, task_subset)
    _record_aisi_input_provenance(provenance, args)
    return provenance


def build_task_source(args: argparse.Namespace) -> TaskEpisodeSource:
    """Construct the task source selected by parsed generation arguments; no endpoint is contacted."""
    return TaskEpisodeSource(
        endpoint=args.endpoint,
        model_path=args.model,
        served_model_id=args.served_model_id,
        revision=args.model_revision,
        output_dir=args.output_dir / "traces",
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
        partition=args.partition,
        families=tuple(args.families),
        leak_channels=args.leak_channels,
        leak_channels_open=args.leak_channels_open,
        leak_instruction=args.leak_instruction,
        leak_repos=args.leak_repos,
        leak_assets=args.leak_assets,
        leak_eval_pairs=args.leak_eval_pairs,
        leak_task_ids=args.leak_task_ids,
        leak_train_pairs=args.leak_train_pairs or None,
        aisi_hack_group=args.aisi_hack_group,
        aisi_cases_path=args.aisi_cases,
        terminal_wrench_registry_path=args.terminal_wrench_registry,
        ilcb_cases_path=args.ilcb_cases,
        ilcb_split_manifest_path=args.ilcb_split_manifest,
        variants=(args.variant,),
        jail_backend=args.jail_backend,
        persistent_jail=args.persistent_jail,
        chat_template_kwargs=args.chat_template_kwargs,
        hint_family=args.hint_family,
        hint_strength=args.hint_strength,
        hint_text=args.hint_text,
        hint_sha=args.hint_sha,
        convention_note=args.convention_note,
    )


def eligible_task_ids(
    args: argparse.Namespace,
    source: TaskEpisodeSource,
    *,
    task_subset: TaskIdSubsetManifest | None,
    excluded_task_ids: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Order, restrict and deduplicate the source's tasks exactly as a generation run selects them."""
    task_ids = select_task_ids(tuple(source.tasks), max_tasks=None, aisi_sources=args.aisi_sources)
    if task_subset is not None:
        task_ids = select_task_ids_from_subset(
            task_ids=task_ids,
            environment_ids=source.environments_by_task,
            subset=task_subset,
            family=args.families[0],
            partition=args.partition,
        )
    leak_ids = tuple(
        task_id for task_id in task_ids if source.families_by_task[task_id] == "r2e_leak"
    )
    if leak_ids:
        task_ids = (
            *round_robin_leak_tasks(leak_ids, source.environments_by_task),
            *(task_id for task_id in task_ids if task_id not in leak_ids),
        )
    explicit_task_ids: list[str] | None = args.explicit_task_ids
    if explicit_task_ids is not None:
        unavailable = [task_id for task_id in explicit_task_ids if task_id not in task_ids]
        if unavailable:
            raise ValueError(
                f"task IDs are not available for the selected family, variant and partition: "
                f"{unavailable}"
            )
        task_ids = tuple(explicit_task_ids)
    task_ids, statement_dedup = select_task_ids_with_dedup(
        task_ids,
        source,
        partition=args.partition,
        exclude_duplicates=args.exclude_duplicates,
        max_tasks=None if task_subset is not None or explicit_task_ids else args.max_tasks,
        excluded_task_ids=excluded_task_ids,
    )
    if explicit_task_ids is not None:
        dropped = [task_id for task_id in explicit_task_ids if task_id not in task_ids]
        if dropped:
            raise ValueError(
                f"explicit task IDs were removed by exclusions or deduplication: {dropped}"
            )
    if task_subset is not None and args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    return task_ids, statement_dedup


def _source_admission(
    args: argparse.Namespace, source: TaskEpisodeSource
) -> LiveKVAdmission | None:
    if args.kv_admission_capacity_tokens is None:
        return None
    admission = LiveKVAdmission(args.endpoint, capacity_tokens=args.kv_admission_capacity_tokens)
    source.backend.request_admission = admission
    return admission


def main(argv: list[str] | None = None) -> None:
    """Build the selected source, bind provenance, then generate the requested episodes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    require_gitignored_output_dir(args.output_dir)
    task_subset = (
        load_task_subset_manifest(args.ilcb_task_subset)
        if args.ilcb_task_subset is not None
        else None
    )
    if task_subset is not None and task_subset.partition != args.partition:
        raise ValueError("task subset partition does not match --partition")
    (
        task_id_exclusion_snapshot,
        current_excluded_task_ids,
        task_id_exclusion_sources,
    ) = _resolve_task_id_exclusion_snapshot(args.exclude_task_ids_from, args.output_dir)
    if args.max_episodes is not None and not args.exclude_task_ids_from:
        task_id_exclusion_snapshot = None
    (
        task_sample_exclusion_snapshot,
        current_excluded_task_samples,
        task_sample_exclusion_sources,
    ) = _resolve_task_sample_exclusion_snapshot(
        args.exclude_task_samples_from,
        args.output_dir,
        freeze_empty=args.max_episodes is not None,
    )
    source = build_task_source(args)
    admission = _source_admission(args, source)
    task_ids, statement_dedup = eligible_task_ids(
        args,
        source,
        task_subset=task_subset,
        excluded_task_ids=task_id_exclusion_snapshot or (),
    )
    task_sample_selection: tuple[tuple[str, int], ...] | None = None
    sample_indices_by_task: dict[str, tuple[int, ...]] | None = None
    if args.max_episodes is not None:
        if task_sample_exclusion_snapshot is None:
            raise RuntimeError("max-episodes requires a frozen task-sample exclusion snapshot")
        task_sample_selection, task_sample_exclusion_snapshot = _select_stage_task_samples(
            task_ids,
            output_dir=args.output_dir,
            excluded_task_samples=current_excluded_task_samples or (),
            max_episodes=args.max_episodes,
            samples_per_task=args.samples_per_task,
        )
        _reject_new_task_sample_exclusion_collisions(
            task_sample_exclusion_snapshot,
            current_excluded_task_samples,
            task_sample_selection,
        )
        task_ids = tuple(dict.fromkeys(task_id for task_id, _sample_index in task_sample_selection))
        provenance_path = args.output_dir / "provenance.json"
        if provenance_path.exists():
            stored = cast(
                "Mapping[str, object]", _read_existing_provenance(provenance_path)["provenance"]
            )
            previous_order = cast("list[str]", stored.get("task_selection_order", []))
            task_ids = tuple(
                dict.fromkeys(
                    [*(task_id for task_id in previous_order if task_id in task_ids), *task_ids]
                )
            )
        selected_samples_by_task: dict[str, list[int]] = {}
        for task_id, sample_index in task_sample_selection:
            selected_samples_by_task.setdefault(task_id, []).append(sample_index)
        sample_indices_by_task = {
            task_id: tuple(sample_indexes)
            for task_id, sample_indexes in selected_samples_by_task.items()
        }
    _reject_new_exclusion_collisions(
        task_id_exclusion_snapshot,
        current_excluded_task_ids,
        task_ids,
    )
    task_specs = tuple(
        TaskSpec(
            task_id,
            cast("CodingTaskFamily", source.families_by_task[task_id]),
            source.variants_by_task[task_id],
            args.partition,
        )
        for task_id in task_ids
    )
    task_specs = _assign_payment_registers(
        task_specs,
        source.environments_by_task,
        fallback_register=args.payment_register,
        in_band_fractions=args.in_band_fractions,
        seed=PAYMENT_REGISTER_ASSIGNMENT_SEED,
    )
    source.payment_registers.update({spec.task_id: spec.payment_register for spec in task_specs})
    generate_episodes(
        output_dir=args.output_dir,
        source=source,
        task_specs=task_specs,
        samples_per_task=args.samples_per_task,
        concurrency=args.concurrency,
        performance_monitor=generation_monitor(args),
        admission=admission,
        sample_indices_by_task=sample_indices_by_task,
        provenance=_provenance(
            args,
            source,
            task_specs,
            task_subset=task_subset,
            statement_dedup=statement_dedup,
            task_id_exclusion_snapshot=task_id_exclusion_snapshot,
            task_id_exclusion_sources=task_id_exclusion_sources,
            task_sample_exclusion_snapshot=task_sample_exclusion_snapshot,
            task_sample_exclusion_sources=task_sample_exclusion_sources,
            task_sample_selection=task_sample_selection,
        ),
        thinking_budget_overshoot_policy=ThinkingBudgetOvershootPolicy(
            args.thinking_budget_overshoot_max_fraction,
            args.thinking_budget_overshoot_min_episodes,
        ),
    )


if __name__ == "__main__":
    main()
