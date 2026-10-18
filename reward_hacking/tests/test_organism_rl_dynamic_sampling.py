from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from reward_hacking.organism_rl.dynamic_sampling import (
    DynamicSamplingState,
    GroupPlan,
)

if TYPE_CHECKING:
    from pathlib import Path


def _tasks(*task_ids: str) -> list[str]:
    return list(task_ids)


def _plan(
    scheduler: DynamicSamplingState,
    *,
    step_id: int = 0,
    tasks: list[str],
    target: int = 2,
) -> GroupPlan | None:
    return scheduler.plan_next_group(step_id, tasks, 2, target)


def test_refills_zero_variance_groups_until_target_is_met(tmp_path: Path) -> None:
    scheduler = DynamicSamplingState(tmp_path, max_groups=4, skip_after=2)
    task_stream = _tasks("zero", "uniform", "informative-a", "informative-b")

    first = _plan(scheduler, tasks=task_stream)
    second = _plan(scheduler, tasks=task_stream)
    assert first is not None
    assert second is not None
    assert _plan(scheduler, tasks=task_stream) is None

    scheduler.complete_group(first, [0.0, 0.0], rollout_seconds=1.25)
    scheduler.complete_group(second, [0.5, 0.5], rollout_seconds=2.0)

    third = _plan(scheduler, tasks=task_stream)
    fourth = _plan(scheduler, tasks=task_stream)
    assert third is not None
    assert fourth is not None
    scheduler.complete_group(third, [0.0, 1.0], rollout_seconds=3.0)
    assert _plan(scheduler, tasks=task_stream) is None
    scheduler.complete_group(fourth, [1.0, 0.0], rollout_seconds=4.0)

    metrics = scheduler.step_metrics("0")
    assert metrics.groups_scheduled == 4
    assert metrics.informative_groups == 2
    assert metrics.dropped_all_zero_groups == 1
    assert metrics.dropped_uniform_other_groups == 1
    assert metrics.dropped_unscorable_groups == 0
    assert metrics.dropped_episodes == 4
    assert metrics.dropped_rollout_seconds == pytest.approx(3.25)


@pytest.mark.parametrize(
    ("uniform_reward", "classification", "skip_reason"),
    [
        (0.0, "all_zero", "all_zero_after_2"),
        (1.0, "all_one", "all_one_after_2"),
    ],
)
def test_uniform_prompt_is_skipped_after_threshold(
    tmp_path: Path, uniform_reward: float, classification: str, skip_reason: str
) -> None:
    scheduler = DynamicSamplingState(tmp_path, max_groups=5, skip_after=2)
    task_stream = _tasks("stuck", "stuck", "useful")

    first = _plan(scheduler, tasks=task_stream, target=3)
    assert first is not None
    assert first.task_id == "stuck"
    first_outcome = scheduler.complete_group(
        first, [uniform_reward, uniform_reward], rollout_seconds=1.0
    )
    assert first_outcome.kind == classification
    assert first_outcome.skip_reason is None

    second = _plan(scheduler, tasks=task_stream, target=3)
    assert second is not None
    assert second.task_id == "stuck"
    second_outcome = scheduler.complete_group(
        second, [uniform_reward, uniform_reward], rollout_seconds=1.0
    )
    assert second_outcome.skip_reason == skip_reason

    third = _plan(scheduler, tasks=task_stream, target=3)
    assert third is not None
    assert third.task_id == "useful"
    scheduler.complete_group(third, [0.0, 1.0], rollout_seconds=1.5)

    fourth = _plan(scheduler, tasks=task_stream, target=3)
    assert fourth is not None
    assert fourth.task_id == "useful"
    scheduler.complete_group(fourth, [1.0, 0.0], rollout_seconds=1.0)

    metrics = scheduler.step_metrics(0)
    assert metrics.prompts_skipped_so_far == 1
    assert [(record.task_id, record.reason) for record in metrics.skipped_records] == [
        ("stuck", skip_reason),
        ("stuck", skip_reason),
    ]


def test_group_cap_stops_refill_and_records_unscorable_groups(tmp_path: Path) -> None:
    scheduler = DynamicSamplingState(tmp_path, max_groups=3, skip_after=4)
    task_stream = _tasks("a", "b", "c", "d")
    plans = [_plan(scheduler, tasks=task_stream, target=3) for _ in range(3)]
    assert all(plan is not None for plan in plans)
    complete_plans = [plan for plan in plans if plan is not None]

    scheduler.complete_group(complete_plans[0], [None, None], rollout_seconds=0.5)
    scheduler.complete_group(complete_plans[1], [0.0, 0.0], rollout_seconds=1.5)
    scheduler.complete_group(complete_plans[2], [None, None], rollout_seconds=2.0)

    assert _plan(scheduler, tasks=task_stream, target=3) is None
    metrics = scheduler.step_metrics(0)
    assert metrics.groups_scheduled == 3
    assert metrics.informative_groups == 0
    assert metrics.dropped_unscorable_groups == 2
    assert metrics.dropped_all_zero_groups == 1
    assert metrics.dropped_episodes == 6
    assert metrics.dropped_rollout_seconds == pytest.approx(4.0)
    assert metrics.group_cap_reached


def test_plans_and_outcomes_resume_idempotently(tmp_path: Path) -> None:
    task_stream = _tasks("first", "second", "third")
    scheduler = DynamicSamplingState(tmp_path, max_groups=3, skip_after=2)
    first = _plan(scheduler, tasks=task_stream, target=1)
    assert first is not None
    assert first.replica_indices == (0, 1)

    resumed = DynamicSamplingState(tmp_path, max_groups=3, skip_after=2)
    assert resumed.step_plans(0) == [first]
    assert resumed.group_outcome(first) is None

    outcome = resumed.complete_group(first, [0.0, 1.0], rollout_seconds=3.5)
    reopened = DynamicSamplingState(tmp_path, max_groups=3, skip_after=2)
    replayed = reopened.complete_group(first, [0.0, 1.0], rollout_seconds=3.5)

    assert replayed == outcome
    assert reopened.step_metrics(0).informative_groups == 1
    assert reopened.step_metrics(0).groups_scheduled == 1
    assert reopened.group_outcome(first) == outcome
    next_plan = _plan(reopened, tasks=task_stream, target=1)
    assert next_plan is None


def test_sampler_cursor_is_global_and_step_advances_after_resolution(tmp_path: Path) -> None:
    task_stream = _tasks("a", "b", "c")
    scheduler = DynamicSamplingState(tmp_path, max_groups=2, skip_after=2)

    first = _plan(scheduler, step_id=0, tasks=task_stream, target=1)
    assert first is not None
    assert first.task_id == "a"
    scheduler.complete_group(first, [0.0, 1.0], rollout_seconds=1.0)

    second = _plan(scheduler, step_id=1, tasks=task_stream, target=1)
    assert second is not None
    assert second.task_id == "b"
    scheduler.complete_group(second, [1.0, 0.0], rollout_seconds=1.0)

    with pytest.raises(ValueError, match="ordered task stream"):
        _plan(scheduler, step_id=2, tasks=_tasks("b", "a", "c"), target=1)


def test_planning_next_step_waits_for_current_pending_groups(tmp_path: Path) -> None:
    scheduler = DynamicSamplingState(tmp_path, max_groups=3, skip_after=2)
    task_stream = _tasks("a", "b", "c")
    first = _plan(scheduler, step_id=0, tasks=task_stream, target=2)
    assert first is not None

    with pytest.raises(ValueError, match="previous step has pending groups"):
        _plan(scheduler, step_id=1, tasks=task_stream, target=2)


def test_repeated_task_groups_receive_distinct_replica_indices(tmp_path: Path) -> None:
    scheduler = DynamicSamplingState(tmp_path, max_groups=3, skip_after=2)
    repeated_stream = _tasks("same-task")

    first = _plan(scheduler, tasks=repeated_stream, target=2)
    second = _plan(scheduler, tasks=repeated_stream, target=2)

    assert first is not None
    assert first.replica_indices == (0, 1)
    assert second is not None
    assert second.replica_indices == (2, 3)


def test_partial_missing_reward_does_not_count_as_uniform_skip(tmp_path: Path) -> None:
    scheduler = DynamicSamplingState(tmp_path, max_groups=3, skip_after=1)
    repeated_stream = _tasks("task")

    partial = scheduler.plan_next_group(0, repeated_stream, 2, 2)
    assert partial is not None
    partial_outcome = scheduler.complete_group(partial, [None, 0.0], rollout_seconds=1.0)
    assert partial_outcome.kind == "unscorable"
    assert partial_outcome.skip_reason is None

    complete = scheduler.plan_next_group(0, repeated_stream, 2, 2)
    assert complete is not None
    complete_outcome = scheduler.complete_group(complete, [0.0, 0.0], rollout_seconds=1.0)
    assert complete_outcome.kind == "all_zero"
    assert complete_outcome.skip_reason == "all_zero_after_1"


def test_two_finite_rewards_are_informative_even_with_missing_replica(tmp_path: Path) -> None:
    scheduler = DynamicSamplingState(tmp_path, max_groups=2, skip_after=1)
    plan = scheduler.plan_next_group(0, _tasks("task"), 3, 1)
    assert plan is not None

    outcome = scheduler.complete_group(plan, [None, 0.0, 1.0], rollout_seconds=1.0)

    assert outcome.kind == "informative"
    assert scheduler.step_metrics(0).prompts_skipped_so_far == 0


def test_nonfinite_rewards_fail_fast(tmp_path: Path) -> None:
    scheduler = DynamicSamplingState(tmp_path, max_groups=2, skip_after=2)
    plan = _plan(scheduler, tasks=_tasks("task"), target=1)
    assert plan is not None

    with pytest.raises(ValueError, match="finite"):
        scheduler.complete_group(plan, [0.0, float("nan")], rollout_seconds=1.0)


def test_completed_step_keeps_its_skip_count_snapshot(tmp_path: Path) -> None:
    scheduler = DynamicSamplingState(tmp_path, max_groups=3, skip_after=1)
    task_stream = _tasks("stuck-a", "useful", "stuck-b")

    zero = _plan(scheduler, step_id=0, tasks=task_stream, target=1)
    assert zero is not None
    assert zero.task_id == "stuck-a"
    scheduler.complete_group(zero, [0.0, 0.0], rollout_seconds=1.0)
    useful = _plan(scheduler, step_id=0, tasks=task_stream, target=1)
    assert useful is not None
    assert useful.task_id == "useful"
    scheduler.complete_group(useful, [0.0, 1.0], rollout_seconds=1.0)

    later_zero = _plan(scheduler, step_id=1, tasks=task_stream, target=1)
    assert later_zero is not None
    assert later_zero.task_id == "stuck-b"
    scheduler.complete_group(later_zero, [0.0, 0.0], rollout_seconds=1.0)
    later_useful = _plan(scheduler, step_id=1, tasks=task_stream, target=1)
    assert later_useful is not None
    assert later_useful.task_id == "useful"
    scheduler.complete_group(later_useful, [1.0, 0.0], rollout_seconds=1.0)

    assert scheduler.step_metrics(0).prompts_skipped_so_far == 1
    assert scheduler.step_metrics(1).prompts_skipped_so_far == 2


def test_resume_validation_allows_current_step_and_rejects_completed_future(tmp_path: Path) -> None:
    scheduler = DynamicSamplingState(tmp_path, max_groups=2, skip_after=2)
    task_stream = _tasks("a", "b")
    current = _plan(scheduler, step_id=0, tasks=task_stream, target=1)
    assert current is not None
    scheduler.complete_group(current, [0.0, 1.0], rollout_seconds=1.0)
    prefetched = _plan(scheduler, step_id=1, tasks=task_stream, target=1)
    assert prefetched is not None

    scheduler.validate_resume_step(0)
    scheduler.validate_resume_step(1)
    scheduler.complete_group(prefetched, [1.0, 0.0], rollout_seconds=1.0)

    with pytest.raises(ValueError, match="completed outcomes after checkpoint"):
        scheduler.validate_resume_step(0)
    scheduler.validate_resume_step(1)
