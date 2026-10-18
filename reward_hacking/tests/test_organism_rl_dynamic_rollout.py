"""Exercise dynamic sampling through the exact-token rollout callback."""

from __future__ import annotations

import random
import threading
from collections import Counter
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.organism_rl.dynamic_rollout import DynamicGroupCollector
from reward_hacking.organism_rl.dynamic_sampling import DynamicSamplingState
from reward_hacking.organism_rl.rollout import (
    EpisodeSample,
    OrganismRollout,
    _check_infrastructure_failure_rate,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.tests.test_distill_episodes import CharacterTokenizer
from reward_hacking.tests.test_organism_rl_rollout import _sample

if TYPE_CHECKING:
    from pathlib import Path


class DynamicTokenizer(CharacterTokenizer):
    """Supply the EOS token used only for fully masked batch padding."""

    eos_token_id = 33


def _trainer(step: int = 0, *, max_steps: int = 1) -> Any:
    return SimpleNamespace(
        state=SimpleNamespace(global_step=step),
        num_generations=2,
        next_prompts=[],
        args=SimpleNamespace(seed=0, max_steps=max_steps),
    )


def _runner(
    directory: Path, source: Any, *, cap: int = 6, pipeline: bool = False
) -> OrganismRollout:
    tokenizer = DynamicTokenizer()
    return OrganismRollout(
        source,
        cast("Any", tokenizer),
        max_sequence_tokens=100,
        episode_concurrency=2,
        step_pipelining=pipeline,
        dynamic_sampling=True,
        dynamic_sampling_max_groups=cap,
        dynamic_sampling_skip_after=2,
        dynamic_sampling_dir=directory,
        dynamic_sampling_tasks=["zero", "mixed", "one"],
    )


def test_dynamic_refills_and_keeps_dropped_evidence_on_resume(tmp_path: Path) -> None:
    calls: Counter[tuple[str, int, int]] = Counter()

    def source(task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        del seed
        calls[task_id, step, replica] += 1
        sample = _sample()
        reward = task_id == "one" or (task_id == "mixed" and replica % 2 == 1)
        record = {**sample.record, "proxy": {"pass": reward}, "elapsed_seconds": 2.0}
        return EpisodeSample(record, sample.trace_records)

    runner = _runner(tmp_path, source)
    output = cast("dict[str, Any]", runner(["zero"] * 4, _trainer()))
    runner.close()
    assert len(output["completion_ids"]) == 4
    assert output["reward_scores"] == [0.0, 1.0] * 2
    metrics = output["dynamic_sampling_metrics"]
    assert metrics["informative_groups"] == 2
    assert metrics["groups_scheduled"] > 2
    assert metrics["dropped_episodes"] == (metrics["groups_scheduled"] - 2) * 2
    assert metrics["dropped_rollout_seconds"] == metrics["dropped_episodes"] * 2
    assert len(output["evidence_records"]) == metrics["groups_scheduled"] * 2
    first_calls = calls.copy()
    resumed = _runner(tmp_path, source)
    replayed = resumed(["zero"] * 4, _trainer())
    resumed.close()
    assert replayed == output
    assert calls == first_calls


def test_dynamic_cap_masks_unfilled_groups(tmp_path: Path) -> None:
    sample = _sample()
    runner = _runner(tmp_path, lambda *args, **kwargs: sample, cap=1)
    output = cast("dict[str, Any]", runner(["zero"] * 4, _trainer()))
    runner.close()
    assert len(output["completion_ids"]) == 4
    assert not any(output["scorable"])
    assert all(not any(mask) for mask in output["env_mask"])
    assert output["dynamic_sampling_metrics"]["groups_scheduled"] == 1
    assert output["dynamic_sampling_metrics"]["informative_groups"] == 0


def test_dynamic_refill_stops_at_wave_boundary_above_infrastructure_threshold(
    tmp_path: Path,
) -> None:
    sample = _sample()
    failed = EpisodeSample(
        {
            **sample.record,
            "infrastructure_failure": {
                "component": "persistent_jail_supervisor",
                "reason": "persistent_jail_supervisor_failure",
            },
        },
        sample.trace_records,
    )
    calls: list[int] = []

    def source(task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        del task_id, seed, step
        calls.append(replica)
        return failed if replica == 0 else sample

    runner = _runner(tmp_path, source, cap=6)
    try:
        with pytest.raises(RuntimeError, match="exceeding 25%"):
            runner(["zero"] * 2, _trainer())
    finally:
        runner.close()

    assert sorted(calls) == [0, 1]


def test_dynamic_resume_checks_cached_completed_groups_before_refill(tmp_path: Path) -> None:
    task_stream = ["zero", "mixed", "one"]
    seed = 7
    ordered_tasks = list(task_stream)
    random.Random(seed).shuffle(ordered_tasks)
    state = DynamicSamplingState(tmp_path, max_groups=6, skip_after=2)
    plan = state.plan_next_group("0", ordered_tasks, generations=2, groups=2)
    assert plan is not None
    state.complete_group(plan, [0.0, 1.0], rollout_seconds=1.0)
    sample = _sample()
    failed = EpisodeSample(
        {
            **sample.record,
            "infrastructure_failure": {
                "component": "persistent_jail_supervisor",
                "reason": "persistent_jail_supervisor_failure",
            },
        },
        sample.trace_records,
    )

    class CachedPipeline:
        def __init__(self) -> None:
            self.collected_jobs: list[tuple[str, int, int]] = []

        def collect_dynamic(
            self, _step: int, jobs: list[tuple[str, int, int]]
        ) -> list[EpisodeSample]:
            self.collected_jobs.extend(jobs)
            return [failed if replica == 0 else sample for _, _, replica in jobs]

        def prefetch_dynamic(self, _step: int, _jobs: list[tuple[str, int, int]]) -> None:
            raise AssertionError("refill must stop before prefetch")

    pipeline = CachedPipeline()
    collector = DynamicGroupCollector(
        state,
        task_stream,
        pipeline,  # type: ignore[arg-type]
        reward=lambda _episode: 0.0,
        seconds=lambda _episode: 1.0,
        episode_seed=lambda _task_id, _step, replica: replica,
    )

    with pytest.raises(RuntimeError, match="exceeding 25%"):
        collector.collect(
            0,
            groups=2,
            generations=2,
            seed=seed,
            prefetch=True,
            on_collected=lambda episodes: _check_infrastructure_failure_rate(
                1, cast("Any", episodes)
            ),
        )

    assert len(pipeline.collected_jobs) == 2
    assert len(state.step_plans("0")) == 1


def test_interrupted_group_relaunch_samples_only_unfinished_episodes(tmp_path: Path) -> None:
    calls: Counter[int] = Counter()
    failed = False

    def source(task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        nonlocal failed
        del task_id, seed, step
        calls[replica] += 1
        if replica == 1 and not failed:
            failed = True
            raise InterruptedError("synthetic process interruption")
        sample = _sample()
        return EpisodeSample(
            {**sample.record, "proxy": {"pass": replica % 2 == 1}}, sample.trace_records
        )

    runner = _runner(tmp_path, source, cap=1)
    try:
        with pytest.raises(InterruptedError):
            runner(["zero"] * 2, _trainer())
    finally:
        runner.close()
    resumed = _runner(tmp_path, source, cap=1)
    try:
        output = cast("dict[str, Any]", resumed(["zero"] * 2, _trainer()))
    finally:
        resumed.close()
    assert calls == {0: 1, 1: 2}
    assert output["dynamic_sampling_metrics"]["groups_scheduled"] == 1
    assert output["dynamic_sampling_metrics"]["informative_groups"] == 1


def test_dynamic_prefetched_group_retains_turn_policy_versions(tmp_path: Path) -> None:
    prefetched = threading.Event()
    updated = threading.Event()

    def source(task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        del task_id, seed
        sample = _sample()
        if step == 1:
            prefetched.set()
            assert updated.wait(5)
        turns = [
            {**record, "policy_step": (0 if record["turn"] == 0 else step)}
            if record["record"] == "turn"
            else record
            for record in sample.trace_records
        ]
        return EpisodeSample({**sample.record, "proxy": {"pass": replica % 2 == 1}}, turns)

    runner = _runner(tmp_path, source, pipeline=True)
    try:
        runner(["zero"] * 2, _trainer(max_steps=2))
        assert prefetched.wait(5)
        updated.set()
        output = cast("dict[str, Any]", runner(["zero"] * 2, _trainer(1, max_steps=2)))
    finally:
        updated.set()
        runner.close()
    assert output["dynamic_sampling_metrics"]["informative_groups"] == 1
    for record, versions in zip(output["rollout_records"], output["policy_steps"], strict=True):
        active_versions = {
            version for active, version in zip(record["loss_mask"], versions, strict=True) if active
        }
        assert active_versions == {0, 1}


def test_exhausted_prompt_history_yields_empty_training_step(tmp_path: Path) -> None:
    calls = 0

    def source(task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        nonlocal calls
        del task_id, seed, step, replica
        calls += 1
        return _sample()

    runner = _runner(tmp_path, source, cap=1)
    try:
        for step in range(6):
            runner(["zero"] * 2, _trainer(step, max_steps=7))
        assert calls == 12
        output = cast("dict[str, Any]", runner(["zero"] * 2, _trainer(6, max_steps=7)))
    finally:
        runner.close()
    assert calls == 12
    assert output["dynamic_sampling_metrics"]["groups_scheduled"] == 0
    assert output["dynamic_sampling_metrics"]["prompts_skipped_so_far"] == 3
    assert len(output["dynamic_sampling_metrics"]["skipped_records"]) == 3
    assert output["evidence_records"] == []
    assert output["scorable"] == [False, False]
