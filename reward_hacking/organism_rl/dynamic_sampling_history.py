"""Import completed prompt-group rewards from earlier organism-RL artifacts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from reward_hacking.organism_reward import score_impossible_task
from reward_hacking.organism_rl.dynamic_sampling import DynamicSamplingState
from reward_hacking.organism_rl.ladder_source import score_ladder_task
from reward_hacking.trace import load_trace

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal


class HistorySourceSummary(TypedDict):
    """Counts imported from one selected prior run directory."""

    directory: str
    source: str
    groups: int
    incomplete_groups: int


class HistorySeedSummary(TypedDict):
    """Persisted counts describing one run's imported dynamic-sampling history."""

    source_directories: list[str]
    prior_groups: int
    incomplete_groups: int
    sources: list[HistorySourceSummary]
    matched_groups: int
    unmatched_groups: int
    pre_skipped_all_zero: int
    pre_skipped_all_one: int


@dataclass(frozen=True)
class _HistoryGroup:
    prompt_identity: str
    rewards: tuple[float | None, ...]


@dataclass(frozen=True)
class _HistorySource:
    name: str
    groups: tuple[_HistoryGroup, ...]
    incomplete_groups: int


def seed_dynamic_sampling_history(
    *,
    run_directory: Path,
    history_directories: Sequence[Path],
    current_prompt_ids: Sequence[str],
    max_groups: int,
    skip_after: int,
) -> HistorySeedSummary:
    """Seed prior complete groups once and return a privacy-safe count summary."""
    resolved_directories = tuple(
        dict.fromkeys(str(Path(path).resolve()) for path in history_directories)
    )
    if not resolved_directories:
        raise ValueError("at least one history directory is required")
    state = DynamicSamplingState(
        Path(run_directory) / "dynamic-sampling", max_groups=max_groups, skip_after=skip_after
    )
    saved_seed = state.history_seed_summary(resolved_directories)
    if saved_seed is not None:
        return cast("HistorySeedSummary", saved_seed)

    history_sources = tuple(
        _read_history_source(Path(directory)) for directory in resolved_directories
    )
    groups = [group for source in history_sources for group in source.groups]
    source_summaries: list[HistorySourceSummary] = [
        {
            "directory": directory,
            "source": source.name,
            "groups": len(source.groups),
            "incomplete_groups": source.incomplete_groups,
        }
        for directory, source in zip(resolved_directories, history_sources, strict=True)
    ]
    source_summary = {
        "source_directories": list(resolved_directories),
        "prior_groups": len(groups),
        "incomplete_groups": sum(source.incomplete_groups for source in history_sources),
        "sources": source_summaries,
    }
    return cast(
        "HistorySeedSummary",
        state.seed_history(
            groups=[(group.prompt_identity, group.rewards) for group in groups],
            current_prompt_ids=current_prompt_ids,
            source_directories=resolved_directories,
            summary=source_summary,
        ),
    )


def _read_history_source(directory: Path) -> _HistorySource:
    if not directory.is_dir():
        raise FileNotFoundError(f"dynamic sampling history directory does not exist: {directory}")
    dynamic_state = directory / "dynamic-sampling" / "dynamic_sampling_state.json"
    if dynamic_state.is_file():
        return _read_dynamic_state(dynamic_state)
    checkpoint_directories = tuple(
        checkpoint
        for checkpoint in sorted(directory.glob("checkpoint-*"))
        if (checkpoint / "complete.json").is_file()
        and (checkpoint / "rollout_records.json").is_file()
    )
    if checkpoint_directories:
        return _read_checkpoint_records(directory, checkpoint_directories)
    episode_directory = directory / "episodes"
    trace_paths = tuple(sorted(episode_directory.glob("step-*/*.jsonl")))
    if trace_paths:
        return _read_episode_traces(directory, trace_paths)
    if (directory / "selection.json").is_file() and (directory / "forage-readout.json").is_file():
        return _read_forage_records(directory)
    raise ValueError(
        f"no compatible dynamic-sampling state, complete checkpoint, episode trace, or forage "
        f"readout found in {directory}"
    )


def _read_dynamic_state(path: Path) -> _HistorySource:
    payload = _read_json_object(path)
    if payload.get("schema_version") not in {1, 2}:
        raise ValueError(f"unsupported dynamic sampling state schema in {path}")
    steps = _mapping(payload["steps"], "dynamic sampling steps")
    groups: list[_HistoryGroup] = []
    incomplete_groups = 0
    seen_group_ids: set[tuple[str, str]] = set()
    for step_id, raw_step in steps.items():
        step = _mapping(raw_step, f"dynamic sampling step {step_id}")
        raw_groups = _sequence(step["groups"], f"dynamic sampling groups in step {step_id}")
        for raw_group in raw_groups:
            group = _mapping(raw_group, "dynamic sampling group")
            outcome = group.get("outcome")
            if outcome is None:
                incomplete_groups += 1
                continue
            plan = _mapping(group["plan"], "dynamic sampling group plan")
            result = _mapping(outcome, "dynamic sampling group outcome")
            plan_step_id = _string(plan["step_id"], "group step id")
            group_id = _string(plan["group_id"], "group id")
            if plan_step_id != step_id:
                raise ValueError("dynamic sampling group step differs from its step mapping")
            group_key = (step_id, group_id)
            if group_key in seen_group_ids:
                raise ValueError(f"duplicate dynamic sampling group {group_key}")
            seen_group_ids.add(group_key)
            prompt_identity = _string(plan["prompt_identity"], "prompt identity")
            rewards = _rewards(result["rewards"], "dynamic sampling group rewards")
            groups.append(_HistoryGroup(prompt_identity, rewards))
    return _HistorySource("dynamic_sampling_state", tuple(groups), incomplete_groups)


def _read_checkpoint_records(run_directory: Path, checkpoints: Sequence[Path]) -> _HistorySource:
    provenance_path = run_directory / "provenance.json"
    provenance = _read_json_object(provenance_path)
    device = _mapping(provenance["device"], "run device provenance")
    generations = _integer(device["effective_num_generations"], "effective generations")
    if generations < 1:
        raise ValueError("effective_num_generations must be positive")

    records_by_episode: dict[tuple[str, int, int], float | None] = {}
    for checkpoint in checkpoints:
        records = _sequence(
            _read_json(checkpoint / "rollout_records.json"),
            f"rollout records in {checkpoint}",
        )
        for raw_record in records:
            record = _mapping(raw_record, "rollout record")
            task_id = _string(record["task_id"], "rollout task_id")
            step = _integer(record["step"], "rollout step")
            replica = _integer(record["replica"], "rollout replica")
            reward = _reward(record["reward_score"], "rollout reward_score")
            key = (task_id, step, replica)
            if key in records_by_episode and records_by_episode[key] != reward:
                raise ValueError(f"duplicate rollout record disagrees for {key}")
            records_by_episode[key] = reward

    grouped_rewards: dict[tuple[str, int, int], dict[int, float | None]] = defaultdict(dict)
    for (task_id, step, replica), reward in records_by_episode.items():
        group_index, replica_index = divmod(replica, generations)
        group_key = (task_id, step, group_index)
        if replica_index in grouped_rewards[group_key]:
            raise ValueError(f"duplicate rollout replica in group {group_key}")
        grouped_rewards[group_key][replica_index] = reward

    groups: list[_HistoryGroup] = []
    incomplete_groups = 0
    for (task_id, _, _), rewards_by_replica in sorted(grouped_rewards.items()):
        if set(rewards_by_replica) != set(range(generations)):
            incomplete_groups += 1
            continue
        rewards = tuple(rewards_by_replica[index] for index in range(generations))
        groups.append(_HistoryGroup(task_id, rewards))
    return _HistorySource("episode_records", tuple(groups), incomplete_groups)


def _read_episode_traces(run_directory: Path, paths: Sequence[Path]) -> _HistorySource:
    provenance = _read_json_object(run_directory / "provenance.json")
    device = _mapping(provenance["device"], "run device provenance")
    generations = _integer(device["effective_num_generations"], "effective generations")
    if generations < 1:
        raise ValueError("effective_num_generations must be positive")

    episode_rewards: dict[tuple[str, int, int], float | None] = {}
    unsupported_episodes: set[tuple[str, int, int]] = set()
    for path in paths:
        records = load_trace(path)
        identities = [
            record for record in records if record.get("record") == "organism_episode_identity"
        ]
        summaries = [record for record in records if record.get("record") == "episode_summary"]
        if len(identities) != 1 or len(summaries) != 1:
            raise ValueError(f"episode trace must contain one identity and summary: {path}")
        identity = identities[0]
        task_id = _string(identity["task_id"], "trace task_id")
        step = _integer(identity["step"], "trace step")
        replica = _integer(identity["replica"], "trace replica")
        summary = summaries[0]
        reward, unsupported = _trace_reward(identity, summary)
        key = (task_id, step, replica)
        if unsupported:
            unsupported_episodes.add(key)
        if key in episode_rewards and episode_rewards[key] != reward:
            raise ValueError(f"duplicate episode trace disagrees for {key}")
        episode_rewards[key] = reward

    grouped_rewards: dict[tuple[str, int, int], dict[int, float | None]] = defaultdict(dict)
    for (task_id, step, replica), reward in episode_rewards.items():
        group_index, replica_index = divmod(replica, generations)
        grouped_rewards[(task_id, step, group_index)][replica_index] = reward

    groups: list[_HistoryGroup] = []
    incomplete_groups = 0
    for (task_id, step, group_index), rewards_by_replica in sorted(grouped_rewards.items()):
        if set(rewards_by_replica) != set(range(generations)):
            incomplete_groups += 1
            continue
        if any(
            (task_id, step, group_index * generations + replica_index) in unsupported_episodes
            for replica_index in range(generations)
        ):
            incomplete_groups += 1
            continue
        rewards = tuple(rewards_by_replica[index] for index in range(generations))
        groups.append(_HistoryGroup(task_id, rewards))
    return _HistorySource("episode_traces", tuple(groups), incomplete_groups)


def _has_ladder_score_fields(record: Mapping[str, object]) -> bool:
    return all(
        name in record for name in ("final_grade", "trace_summary", "highest_rung", "rung2_methods")
    )


def _trace_reward(
    identity: Mapping[str, object], summary: Mapping[str, object]
) -> tuple[float | None, bool]:
    if _string(identity["task_family"], "trace task_family") == "ladder":
        if not _has_ladder_score_fields(summary):
            return None, True
        return _reward(score_ladder_task(summary).reward, "ladder trace reward"), False
    payment_register = identity.get("payment_register")
    if payment_register not in {None, "trusted_rerun", "in_band", "status_report"}:
        raise ValueError(f"unknown trace payment register {payment_register!r}")
    reward = score_impossible_task(
        summary,
        payment_register=cast(
            "Literal['trusted_rerun', 'in_band', 'status_report'] | None", payment_register
        ),
    ).reward
    return _reward(reward, "episode trace reward"), False


def _read_forage_records(directory: Path) -> _HistorySource:
    selection = _read_json_object(directory / "selection.json")
    groups_mapping = _mapping(selection.get("groups", selection), "forage selection groups")
    expected_samples = _forage_readout_sample_counts(directory)
    groups: list[_HistoryGroup] = []
    incomplete_groups = 0
    for group_name, raw_group in groups_mapping.items():
        group_config = _mapping(raw_group, f"forage group {group_name}")
        for task_id in _forage_task_ids(group_config, group_name):
            key = (group_name, task_id)
            if key not in expected_samples:
                incomplete_groups += 1
                continue
            reward_batches = _forage_task_reward_batches(directory, group_name, task_id)
            if sum(len(batch) for batch in reward_batches) != expected_samples[key]:
                incomplete_groups += 1
                continue
            groups.extend(_HistoryGroup(task_id, rewards) for rewards in reward_batches)
    return _HistorySource("forage_records", tuple(groups), incomplete_groups)


def _forage_readout_sample_counts(directory: Path) -> dict[tuple[str, str], int]:
    readout = _read_json_object(directory / "forage-readout.json")
    rows = _sequence(readout["tasks"], "forage readout tasks")
    expected_samples: dict[tuple[str, str], int] = {}
    for raw_row in rows:
        row = _mapping(raw_row, "forage readout row")
        key = (_string(row["group"], "forage group"), _string(row["task_id"], "forage task_id"))
        if key in expected_samples:
            raise ValueError(f"duplicate forage readout row for {key}")
        expected_samples[key] = _integer(row["samples"], "forage sample count")
    return expected_samples


def _forage_task_ids(group_config: Mapping[str, object], group_name: str) -> list[str]:
    task_ids = [
        _string(task_id, "forage task_id")
        for task_id in _sequence(group_config["task_ids"], f"forage task ids for {group_name}")
    ]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"duplicate task id in forage group {group_name}")
    return task_ids


def _forage_task_reward_batches(
    directory: Path, group_name: str, task_id: str
) -> tuple[tuple[float | None, ...], ...]:
    reward_batches: list[tuple[float | None, ...]] = []
    sample_indices: set[int] = set()
    for relative_path in (
        Path(group_name) / "records.jsonl",
        Path(group_name) / "followup" / "records.jsonl",
    ):
        path = directory / relative_path
        if not path.is_file():
            continue
        task_records = []
        for raw_record in _read_jsonl(path):
            record = _mapping(raw_record, "forage episode record")
            if _string(record["task_id"], "forage record task_id") == task_id:
                task_records.append(record)
        batch_indices = [
            _integer(record["sample_index"], "forage sample_index") for record in task_records
        ]
        if sample_indices.intersection(batch_indices) or len(set(batch_indices)) != len(
            batch_indices
        ):
            raise ValueError(f"duplicate forage sample index for {group_name}/{task_id}")
        sample_indices.update(batch_indices)
        rewards = tuple(
            _reward(
                _mapping(record["organism_reward"], "forage organism_reward")["reward"],
                "forage organism_reward.reward",
            )
            for record in task_records
        )
        if rewards:
            reward_batches.append(rewards)
    return tuple(reward_batches)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_object(path: Path) -> dict[str, object]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise TypeError(f"JSON document in {path} must be an object")
    return cast("dict[str, object]", value)


def _read_jsonl(path: Path) -> Sequence[object]:
    return load_trace(path)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return cast("Mapping[str, object]", value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return cast("Sequence[object]", value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer")
    return value


def _reward(value: object, context: str) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise TypeError(f"{context} must be a finite number or null")
    numeric_value = cast("int | float", value)
    if not math.isfinite(float(numeric_value)):
        raise TypeError(f"{context} must be a finite number or null")
    return float(numeric_value)


def _rewards(value: object, context: str) -> tuple[float | None, ...]:
    return tuple(_reward(reward, context) for reward in _sequence(value, context))
