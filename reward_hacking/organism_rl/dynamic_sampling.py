"""Persisted prompt-group scheduling for dynamic reward-variance sampling."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

GroupKind = Literal["informative", "all_zero", "all_one", "uniform_other", "unscorable"]
_DYNAMIC_SAMPLING_SCHEMA_VERSION = 1
_MINIMUM_DISTINCT_REWARDS = 2


@dataclass(frozen=True)
class GroupPlan:
    """Stable group identity and replicas reserved before episode execution."""

    group_id: str
    step_id: str
    task_id: str
    prompt_identity: str
    sampler_cursor: int
    replica_indices: tuple[int, ...]


@dataclass(frozen=True)
class GroupOutcome:
    """Persisted reward-variance classification for a completed prompt group."""

    group_id: str
    kind: GroupKind
    rewards: tuple[float | None, ...]
    rollout_seconds: float
    skip_reason: str | None


@dataclass(frozen=True)
class SkippedRecord:
    """One sampler occurrence skipped because prior outcomes crossed a threshold."""

    task_id: str
    prompt_identity: str
    sampler_cursor: int
    reason: str


@dataclass(frozen=True)
class StepMetrics:
    """Outcome and wasted-rollout accounting for one training step."""

    step_id: str
    groups_scheduled: int
    informative_groups: int
    dropped_all_zero_groups: int
    dropped_all_one_groups: int
    dropped_uniform_other_groups: int
    dropped_unscorable_groups: int
    prompts_skipped_so_far: int
    dropped_episodes: int
    dropped_rollout_seconds: float
    pending_groups: int
    group_cap_reached: bool
    target_reached: bool
    sampler_exhausted: bool
    skipped_records: tuple[SkippedRecord, ...]


@dataclass
class _StoredGroup:
    plan: GroupPlan
    outcome: GroupOutcome | None = None


@dataclass
class _StepState:
    target_informative_groups: int
    groups: list[_StoredGroup]
    skipped_records: list[SkippedRecord]
    sampler_exhausted: bool = False
    prompts_skipped_so_far: int | None = None


@dataclass
class _PromptHistory:
    all_zero_groups: int = 0
    all_one_groups: int = 0
    skip_reason: str | None = None


@dataclass
class _PersistedState:
    max_groups: int
    skip_after: int
    sampler_fingerprint: str | None
    sampler_length: int | None
    sampler_cursor: int
    prompt_history: dict[str, _PromptHistory]
    steps: dict[str, _StepState]
    step_order: list[str]


class DynamicSamplingState:
    """Schedule prompt groups and persist outcomes so interrupted steps resume safely."""

    _STATE_FILENAME = "dynamic_sampling_state.json"

    def __init__(self, directory: Path, max_groups: int, skip_after: int) -> None:
        """Open durable state, creating an empty scheduler when no snapshot exists."""
        if max_groups < 1:
            raise ValueError("max_groups must be positive")
        if skip_after < 1:
            raise ValueError("skip_after must be positive")

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._state_path = self.directory / self._STATE_FILENAME
        self._lock = threading.RLock()
        if self._state_path.exists():
            self._state = self._read_state()
            if self._state.max_groups != max_groups or self._state.skip_after != skip_after:
                raise ValueError("dynamic sampling configuration differs from persisted state")
        else:
            self._state = _PersistedState(
                max_groups=max_groups,
                skip_after=skip_after,
                sampler_fingerprint=None,
                sampler_length=None,
                sampler_cursor=0,
                prompt_history={},
                steps={},
                step_order=[],
            )

    def plan_next_group(
        self,
        step_id: int | str,
        ordered_tasks: Sequence[str],
        generations: int,
        groups: int,
    ) -> GroupPlan | None:
        """Reserve the next eligible task group, advancing the persisted sampler cursor."""
        normalized_step_id = str(step_id)
        if not normalized_step_id:
            raise ValueError("step_id must not be empty")
        if generations < 1:
            raise ValueError("generations must be positive")
        if groups < 1:
            raise ValueError("groups must be positive")
        task_stream = tuple(ordered_tasks)
        if any(not task_id for task_id in task_stream):
            raise ValueError("ordered task IDs must not be empty")
        fingerprint = _task_stream_fingerprint(task_stream)

        with self._lock:
            new_state = deepcopy(self._state)
            self._bind_sampler_stream(new_state, task_stream, fingerprint)
            step_state = self._step_for_planning(new_state, normalized_step_id, groups)
            if step_state is None:
                return None
            plan = _reserve_next_group(
                new_state, step_state, normalized_step_id, task_stream, generations
            )
            self._persist(new_state)
            return plan

    def validate_resume_step(self, step: int) -> None:
        """Reject resume checkpoints contaminated by completed or distant future steps."""
        with self._lock:
            for stored_step_id, stored_step in self._state.steps.items():
                if not stored_step_id.isdecimal():
                    continue
                stored_step_number = int(stored_step_id)
                if stored_step_number > step + 1:
                    raise ValueError("dynamic sampling state contains steps beyond resume window")
                if stored_step_number == step + 1 and any(
                    group.outcome is not None for group in stored_step.groups
                ):
                    raise ValueError(
                        "dynamic sampling state has completed outcomes after checkpoint"
                    )

    def _bind_sampler_stream(
        self, state: _PersistedState, task_stream: tuple[str, ...], fingerprint: str
    ) -> None:
        if state.sampler_fingerprint is None:
            state.sampler_fingerprint = fingerprint
            state.sampler_length = len(task_stream)
        elif state.sampler_fingerprint != fingerprint or state.sampler_length != len(task_stream):
            raise ValueError("ordered task stream differs from persisted sampler stream")

    def _step_for_planning(
        self, state: _PersistedState, step_id: str, target_groups: int
    ) -> _StepState | None:
        step_state = state.steps.get(step_id)
        if step_state is None:
            self._check_step_can_start(state)
            if (
                state.step_order
                and step_id.isdecimal()
                and state.step_order[-1].isdecimal()
                and int(step_id) <= int(state.step_order[-1])
            ):
                raise ValueError("numeric step IDs must increase")
            step_state = _StepState(
                groups=[], skipped_records=[], target_informative_groups=target_groups
            )
            state.steps[step_id] = step_state
            state.step_order.append(step_id)
        elif step_state.target_informative_groups != target_groups:
            raise ValueError("target group count differs from persisted step")
        elif state.step_order[-1] != step_id:
            if _step_is_resolved(step_state, state.max_groups):
                return None
            raise ValueError("cannot append groups to a step after a later step was planned")
        return step_state

    def step_plans(self, step_id: int | str) -> list[GroupPlan]:
        """Return all persisted plans, including pending groups, in reservation order."""
        with self._lock:
            step_state = self._state.steps.get(str(step_id))
            if step_state is None:
                return []
            return [group.plan for group in step_state.groups]

    def group_outcome(self, plan: GroupPlan) -> GroupOutcome | None:
        """Return a group's durable outcome, or None while the plan is pending."""
        with self._lock:
            group = self._find_group(self._state, plan)
            return group.outcome

    def complete_group(
        self, plan: GroupPlan, rewards: Sequence[float | None], rollout_seconds: float
    ) -> GroupOutcome:
        """Persist one group's outcome exactly once; identical replay is a no-op."""
        if not math.isfinite(rollout_seconds) or rollout_seconds < 0:
            raise ValueError("rollout_seconds must be finite and nonnegative")
        normalized_rewards = tuple(
            float(reward) if reward is not None else None for reward in rewards
        )
        if any(reward is not None and not math.isfinite(reward) for reward in normalized_rewards):
            raise ValueError("rewards must be finite numbers or None")
        if len(normalized_rewards) != len(plan.replica_indices):
            raise ValueError("reward count must match the planned replica count")
        kind = _classify_rewards(normalized_rewards)

        with self._lock:
            new_state = deepcopy(self._state)
            group = self._find_group(new_state, plan)
            if group.outcome is not None:
                return _replay_outcome(
                    group.outcome, plan, kind, normalized_rewards, rollout_seconds
                )
            skip_reason = _record_prompt_outcome(new_state, plan.prompt_identity, kind)
            outcome = GroupOutcome(
                group_id=plan.group_id,
                kind=kind,
                rewards=normalized_rewards,
                rollout_seconds=rollout_seconds,
                skip_reason=skip_reason,
            )
            group.outcome = outcome
            self._persist(new_state)
            return outcome

    def step_metrics(self, step_id: int | str) -> StepMetrics:
        """Return durable sampling and dropped-rollout metrics for one step."""
        normalized_step_id = str(step_id)
        with self._lock:
            step_state = self._state.steps.get(normalized_step_id)
            if step_state is None:
                raise KeyError(f"no dynamic sampling state for step {normalized_step_id}")

            completed_outcomes = [
                group.outcome for group in step_state.groups if group.outcome is not None
            ]
            dropped_outcomes = [
                outcome for outcome in completed_outcomes if outcome.kind != "informative"
            ]
            return StepMetrics(
                step_id=normalized_step_id,
                groups_scheduled=len(step_state.groups),
                informative_groups=sum(
                    outcome.kind == "informative" for outcome in completed_outcomes
                ),
                dropped_all_zero_groups=sum(
                    outcome.kind == "all_zero" for outcome in completed_outcomes
                ),
                dropped_all_one_groups=sum(
                    outcome.kind == "all_one" for outcome in completed_outcomes
                ),
                dropped_uniform_other_groups=sum(
                    outcome.kind == "uniform_other" for outcome in completed_outcomes
                ),
                dropped_unscorable_groups=sum(
                    outcome.kind == "unscorable" for outcome in completed_outcomes
                ),
                prompts_skipped_so_far=(
                    step_state.prompts_skipped_so_far
                    if step_state.prompts_skipped_so_far is not None
                    else _skipped_prompt_count(self._state)
                ),
                dropped_episodes=sum(len(outcome.rewards) for outcome in dropped_outcomes),
                dropped_rollout_seconds=sum(
                    outcome.rollout_seconds for outcome in dropped_outcomes
                ),
                pending_groups=len(step_state.groups) - len(completed_outcomes),
                group_cap_reached=len(step_state.groups) >= self._state.max_groups,
                target_reached=step_state.target_informative_groups
                <= sum(outcome.kind == "informative" for outcome in completed_outcomes),
                sampler_exhausted=step_state.sampler_exhausted,
                skipped_records=tuple(step_state.skipped_records),
            )

    def _check_step_can_start(self, state: _PersistedState) -> None:
        if not state.step_order:
            return
        previous_step_id = state.step_order[-1]
        previous_step = state.steps[previous_step_id]
        pending = sum(group.outcome is None for group in previous_step.groups)
        if pending:
            raise ValueError("previous step has pending groups")
        if not _step_is_resolved(previous_step, state.max_groups):
            raise ValueError("previous step has not reached its group target or cap")

    def _find_group(self, state: _PersistedState, plan: GroupPlan) -> _StoredGroup:
        step_state = state.steps.get(plan.step_id)
        if step_state is None:
            raise ValueError(f"unknown dynamic sampling step: {plan.step_id}")
        for group in step_state.groups:
            if group.plan.group_id == plan.group_id:
                if group.plan != plan:
                    raise ValueError("group plan differs from persisted plan")
                return group
        raise ValueError(f"unknown dynamic sampling group: {plan.group_id}")

    def _persist(self, state: _PersistedState) -> None:
        skipped_prompts = _skipped_prompt_count(state)
        for step in state.steps.values():
            if step.prompts_skipped_so_far is None and _step_is_resolved(step, state.max_groups):
                step.prompts_skipped_so_far = skipped_prompts
        payload = _state_to_json(state)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=".dynamic-sampling-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(
                    payload,
                    temporary_file,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            temporary_path.replace(self._state_path)
            temporary_path = None
            directory_descriptor = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        self._state = state

    def _read_state(self) -> _PersistedState:
        payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        state = _state_from_json(payload)
        if state.max_groups < 1 or state.skip_after < 1:
            raise ValueError("persisted dynamic sampling configuration is invalid")
        return state


def _classify_rewards(rewards: tuple[float | None, ...]) -> GroupKind:
    finite_rewards = [reward for reward in rewards if reward is not None]
    distinct_rewards = set(finite_rewards)
    if len(distinct_rewards) >= _MINIMUM_DISTINCT_REWARDS:
        return "informative"
    if not finite_rewards or len(finite_rewards) != len(rewards):
        return "unscorable"
    reward = finite_rewards[0]
    if reward == 0.0:
        return "all_zero"
    if reward == 1.0:
        return "all_one"
    return "uniform_other"


def _reserve_next_group(
    state: _PersistedState,
    step: _StepState,
    step_id: str,
    task_stream: tuple[str, ...],
    generations: int,
) -> GroupPlan | None:
    if _planning_is_complete(step, state.max_groups):
        return None
    if not task_stream:
        step.sampler_exhausted = True
        return None

    for _ in range(len(task_stream)):
        sampler_cursor = state.sampler_cursor
        task_id = task_stream[sampler_cursor % len(task_stream)]
        state.sampler_cursor += 1
        history = state.prompt_history.get(task_id)
        if history is not None and history.skip_reason is not None:
            step.skipped_records.append(
                SkippedRecord(task_id, task_id, sampler_cursor, history.skip_reason)
            )
            continue
        previous_replicas = sum(
            len(group.plan.replica_indices)
            for group in step.groups
            if group.plan.task_id == task_id
        )
        plan = GroupPlan(
            group_id=f"{step_id}:{len(step.groups)}",
            step_id=step_id,
            task_id=task_id,
            prompt_identity=task_id,
            sampler_cursor=sampler_cursor,
            replica_indices=tuple(range(previous_replicas, previous_replicas + generations)),
        )
        step.groups.append(_StoredGroup(plan=plan))
        return plan

    step.sampler_exhausted = True
    return None


def _replay_outcome(
    existing: GroupOutcome,
    plan: GroupPlan,
    kind: GroupKind,
    rewards: tuple[float | None, ...],
    rollout_seconds: float,
) -> GroupOutcome:
    replay = GroupOutcome(
        group_id=plan.group_id,
        kind=kind,
        rewards=rewards,
        rollout_seconds=rollout_seconds,
        skip_reason=existing.skip_reason,
    )
    if replay != existing:
        raise ValueError("completed group replay differs from persisted outcome")
    return existing


def _record_prompt_outcome(
    state: _PersistedState, prompt_identity: str, kind: GroupKind
) -> str | None:
    history = state.prompt_history.setdefault(prompt_identity, _PromptHistory())
    if kind == "all_zero":
        history.all_zero_groups += 1
        skip_reason = (
            f"all_zero_after_{state.skip_after}"
            if history.all_zero_groups >= state.skip_after
            else None
        )
    elif kind == "all_one":
        history.all_one_groups += 1
        skip_reason = (
            f"all_one_after_{state.skip_after}"
            if history.all_one_groups >= state.skip_after
            else None
        )
    else:
        skip_reason = None
    if history.skip_reason is None and skip_reason is not None:
        history.skip_reason = skip_reason
    return skip_reason


def _planning_is_complete(step: _StepState, max_groups: int) -> bool:
    informative = sum(
        group.outcome is not None and group.outcome.kind == "informative" for group in step.groups
    )
    pending = sum(group.outcome is None for group in step.groups)
    return (
        len(step.groups) >= max_groups
        or informative + pending >= step.target_informative_groups
        or step.sampler_exhausted
    )


def _step_is_resolved(step: _StepState, max_groups: int) -> bool:
    informative = sum(
        group.outcome is not None and group.outcome.kind == "informative" for group in step.groups
    )
    pending = sum(group.outcome is None for group in step.groups)
    return pending == 0 and (
        informative >= step.target_informative_groups
        or len(step.groups) >= max_groups
        or step.sampler_exhausted
    )


def _task_stream_fingerprint(task_stream: tuple[str, ...]) -> str:
    encoded_stream = json.dumps(task_stream, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded_stream).hexdigest()


def _state_to_json(state: _PersistedState) -> dict[str, object]:
    return {
        "schema_version": _DYNAMIC_SAMPLING_SCHEMA_VERSION,
        "max_groups": state.max_groups,
        "skip_after": state.skip_after,
        "sampler_fingerprint": state.sampler_fingerprint,
        "sampler_length": state.sampler_length,
        "sampler_cursor": state.sampler_cursor,
        "prompt_history": {
            task_id: asdict(history) for task_id, history in state.prompt_history.items()
        },
        "steps": {
            step_id: {
                "target_informative_groups": step.target_informative_groups,
                "groups": [
                    {
                        "plan": asdict(group.plan),
                        "outcome": asdict(group.outcome) if group.outcome is not None else None,
                    }
                    for group in step.groups
                ],
                "skipped_records": [asdict(record) for record in step.skipped_records],
                "sampler_exhausted": step.sampler_exhausted,
                "prompts_skipped_so_far": step.prompts_skipped_so_far,
            }
            for step_id, step in state.steps.items()
        },
        "step_order": state.step_order,
    }


def _state_from_json(payload: object) -> _PersistedState:
    if not isinstance(payload, dict):
        raise TypeError("dynamic sampling state must be a JSON object")
    if payload.get("schema_version") != _DYNAMIC_SAMPLING_SCHEMA_VERSION:
        raise ValueError("unsupported dynamic sampling state schema")
    raw_state = cast("dict[str, object]", payload)
    raw_steps = cast("dict[str, dict[str, object]]", raw_state["steps"])
    steps: dict[str, _StepState] = {}
    for step_id, raw_step in raw_steps.items():
        raw_groups = cast("list[dict[str, object]]", raw_step["groups"])
        groups = [
            _StoredGroup(
                plan=_group_plan_from_json(raw_group["plan"]),
                outcome=(
                    None
                    if raw_group["outcome"] is None
                    else _group_outcome_from_json(raw_group["outcome"])
                ),
            )
            for raw_group in raw_groups
        ]
        steps[step_id] = _StepState(
            target_informative_groups=cast("int", raw_step["target_informative_groups"]),
            groups=groups,
            skipped_records=[
                _skipped_record_from_json(record)
                for record in cast("list[object]", raw_step["skipped_records"])
            ],
            sampler_exhausted=cast("bool", raw_step["sampler_exhausted"]),
            prompts_skipped_so_far=cast("int | None", raw_step["prompts_skipped_so_far"]),
        )
    raw_history = cast("dict[str, dict[str, object]]", raw_state["prompt_history"])
    return _PersistedState(
        max_groups=cast("int", raw_state["max_groups"]),
        skip_after=cast("int", raw_state["skip_after"]),
        sampler_fingerprint=cast("str | None", raw_state["sampler_fingerprint"]),
        sampler_length=cast("int | None", raw_state["sampler_length"]),
        sampler_cursor=cast("int", raw_state["sampler_cursor"]),
        prompt_history={
            task_id: _PromptHistory(
                all_zero_groups=cast("int", history["all_zero_groups"]),
                all_one_groups=cast("int", history["all_one_groups"]),
                skip_reason=cast("str | None", history["skip_reason"]),
            )
            for task_id, history in raw_history.items()
        },
        steps=steps,
        step_order=cast("list[str]", raw_state["step_order"]),
    )


def _group_plan_from_json(payload: object) -> GroupPlan:
    raw_plan = cast("dict[str, object]", payload)
    return GroupPlan(
        group_id=cast("str", raw_plan["group_id"]),
        step_id=cast("str", raw_plan["step_id"]),
        task_id=cast("str", raw_plan["task_id"]),
        prompt_identity=cast("str", raw_plan["prompt_identity"]),
        sampler_cursor=cast("int", raw_plan["sampler_cursor"]),
        replica_indices=tuple(cast("list[int]", raw_plan["replica_indices"])),
    )


def _group_outcome_from_json(payload: object) -> GroupOutcome:
    raw_outcome = cast("dict[str, object]", payload)
    return GroupOutcome(
        group_id=cast("str", raw_outcome["group_id"]),
        kind=cast("GroupKind", raw_outcome["kind"]),
        rewards=tuple(cast("list[float | None]", raw_outcome["rewards"])),
        rollout_seconds=cast("float", raw_outcome["rollout_seconds"]),
        skip_reason=cast("str | None", raw_outcome["skip_reason"]),
    )


def _skipped_record_from_json(payload: object) -> SkippedRecord:
    raw_record = cast("dict[str, object]", payload)
    return SkippedRecord(
        task_id=cast("str", raw_record["task_id"]),
        prompt_identity=cast("str", raw_record["prompt_identity"]),
        sampler_cursor=cast("int", raw_record["sampler_cursor"]),
        reason=cast("str", raw_record["reason"]),
    )


def _skipped_prompt_count(state: _PersistedState) -> int:
    return sum(history.skip_reason is not None for history in state.prompt_history.values())
