"""Step pipelining keeps the next GRPO batch separate from the current one."""

from __future__ import annotations

import threading
from concurrent.futures import Future

import pytest

from reward_hacking.organism_rl.pipeline import StepEpisodePipeline
from reward_hacking.organism_rl.rollout import _policy_steps_for_row, _validate_prompt_groups


def test_next_step_starts_in_released_slot_and_preserves_group_order() -> None:
    started: list[tuple[int, str]] = []
    releases = {(step, task): threading.Event() for step in (0, 1) for task in ("a", "b")}
    next_started = threading.Event()
    second_next_started = threading.Event()

    def sample(step: int, task: str) -> str:
        started.append((step, task))
        if step == 1:
            next_started.set()
            if task == "b":
                second_next_started.set()
        assert releases[step, task].wait(5)
        return f"{step}:{task}"

    with StepEpisodePipeline(sample, max_workers=2) as pipeline:
        first: Future[list[str]] = Future()

        def collect_first() -> None:
            first.set_result(pipeline.collect(0, ["a", "b"], ["a", "b"]))

        worker = threading.Thread(target=collect_first)
        worker.start()
        assert releases[0, "a"].is_set() is False
        releases[0, "a"].set()
        assert next_started.wait(5)
        releases[1, "a"].set()
        assert not first.done()
        try:
            assert second_next_started.wait(1)
        finally:
            releases[0, "b"].set()
            releases[1, "b"].set()
        assert first.result(timeout=5) == ["0:a", "0:b"]
        assert pipeline.collect(1, ["a", "b"], None) == ["1:a", "1:b"]
        worker.join(timeout=5)
    assert started.index((1, "a")) < started.index((0, "b")) + 2


def test_resume_discards_partial_next_step_and_never_double_counts() -> None:
    sampled: list[tuple[int, str]] = []

    def sample(step: int, task: str) -> str:
        sampled.append((step, task))
        return f"{step}:{task}"

    with StepEpisodePipeline(sample, max_workers=2) as pipeline:
        assert pipeline.collect(0, ["a", "b"], ["a", "b"]) == ["0:a", "0:b"]
        assert pipeline.collect(1, ["a", "b"], None) == ["1:a", "1:b"]
    with StepEpisodePipeline(sample, max_workers=2) as resumed:
        assert resumed.collect(1, ["a", "b"], None) == ["1:a", "1:b"]
    assert sampled.count((1, "a")) == 2


def test_prefetched_group_must_match_the_trainer_batch() -> None:
    with StepEpisodePipeline(lambda step, task: task, max_workers=2) as pipeline:
        assert pipeline.collect(0, ["a", "b"], ["c", "c"]) == ["a", "b"]
        with pytest.raises(ValueError, match="prefetched"):
            pipeline.collect(1, ["different", "c"], None)


def test_all_eight_replicas_of_each_group_are_returned_in_one_step() -> None:
    current = ["first"] * 8 + ["second"] * 8
    following = ["third"] * 8 + ["fourth"] * 8
    counters: dict[tuple[int, str], int] = {}
    lock = threading.Lock()

    def sample(step: int, task: str) -> tuple[int, str, int]:
        with lock:
            replica = counters.get((step, task), 0)
            counters[step, task] = replica + 1
        return step, task, replica

    with StepEpisodePipeline(sample, max_workers=6) as pipeline:
        current_results = pipeline.collect(0, current, following)
        following_results = pipeline.collect(1, following, None)
    assert len(current_results) == len(following_results) == 16
    assert {result[:2] for result in current_results} == {(0, "first"), (0, "second")}
    assert {result[:2] for result in following_results} == {(1, "third"), (1, "fourth")}
    for step, task in ((0, "first"), (0, "second"), (1, "third"), (1, "fourth")):
        assert counters[step, task] == 8


def test_assistant_tokens_keep_their_turn_policy_versions() -> None:
    row = {"token_turn_indices": [0, 0, -1, 1, 1, -1]}
    trace = [
        {"record": "turn", "policy_step": 0},
        {"record": "turn", "policy_step": 1},
    ]
    assert _policy_steps_for_row(row, trace, step=1) == [0, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="valid sampler policy"):
        _policy_steps_for_row(row, [{"record": "turn"}, trace[1]], step=1)


def test_incomplete_or_mixed_grpo_groups_are_refused() -> None:
    _validate_prompt_groups(["first"] * 8 + ["second"] * 8, generations=8)
    with pytest.raises(ValueError, match="complete prompt groups"):
        _validate_prompt_groups(["first"] * 7, generations=8)
    with pytest.raises(ValueError, match="one repeated task"):
        _validate_prompt_groups(["first"] * 7 + ["second"], generations=8)
