"""Synthetic tests for resumable AISI rh_envs ETL orchestration."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

import reward_hacking.aisi_rh_envs_etl as etl
from reward_hacking.aisi_rh_envs_etl import (
    BakeCandidate,
    BakeProblemResult,
    BakeWorkItem,
    _make_candidate_work_item,
    _run_bake_work_items,
)
from reward_hacking.organisms.grading import JailedPytestResult

if TYPE_CHECKING:
    from pathlib import Path


def _candidate(problem_number: int, *, statement: str | None = None) -> BakeCandidate:
    problem_id = f"toy-{problem_number}"
    return BakeCandidate(
        source="humaneval",
        source_problem_id=problem_id,
        statement=statement or f"Return the value {problem_number}.",
        entry_point="solve",
        solution_signature="def solve(value):",
        reference_solutions=(f"def solve(value):\n    return value + {problem_number}\n",),
        visible_assertions=(
            "assert solve(0) == 0",
            "assert solve(1) == 1",
            "assert solve(2) == 2",
        ),
    )


def _work_items(candidates: tuple[BakeCandidate, ...]) -> tuple[BakeWorkItem, ...]:
    return tuple(
        _make_candidate_work_item(
            candidate,
            etl_code_hash="synthetic-etl-hash",
            source_revision="synthetic-source-revision",
        )
        for candidate in candidates
    )


def _fake_baker(candidate: BakeCandidate) -> BakeProblemResult:
    problem_id = f"{candidate.source}:{candidate.source_problem_id}"
    return BakeProblemResult(
        {
            "problem_id": problem_id,
            "source": candidate.source,
            "statement": candidate.statement,
        },
        None,
    )


def test_parallel_results_match_serial_and_are_sorted_by_problem_id(tmp_path: Path) -> None:
    candidates = tuple(_candidate(number) for number in (4, 1, 3, 0, 2))

    def staggered_baker(candidate: BakeCandidate) -> BakeProblemResult:
        time.sleep((4 - int(candidate.source_problem_id[-1])) * 0.002)
        return _fake_baker(candidate)

    serial = _run_bake_work_items(
        _work_items(candidates),
        cache_path=tmp_path / "serial.jsonl",
        workers=1,
        bake_candidate=staggered_baker,
    )
    parallel = _run_bake_work_items(
        _work_items(candidates),
        cache_path=tmp_path / "parallel.jsonl",
        workers=3,
        bake_candidate=staggered_baker,
    )

    assert serial.problems == parallel.problems
    assert [problem["problem_id"] for problem in parallel.problems] == [
        "humaneval:toy-0",
        "humaneval:toy-1",
        "humaneval:toy-2",
        "humaneval:toy-3",
        "humaneval:toy-4",
    ]


def test_interrupted_run_reuses_persisted_results_and_computes_only_remainder(
    tmp_path: Path,
) -> None:
    candidates = tuple(_candidate(number) for number in range(5))
    work_items = _work_items(candidates)
    cache_path = tmp_path / "resume.jsonl"
    first_run_calls: list[str] = []

    def interrupting_baker(candidate: BakeCandidate) -> BakeProblemResult:
        first_run_calls.append(candidate.source_problem_id)
        if candidate.source_problem_id == "toy-2":
            raise RuntimeError("simulated interruption")
        return _fake_baker(candidate)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _run_bake_work_items(
            work_items,
            cache_path=cache_path,
            workers=1,
            bake_candidate=interrupting_baker,
        )

    resumed_calls: list[str] = []

    def counting_baker(candidate: BakeCandidate) -> BakeProblemResult:
        resumed_calls.append(candidate.source_problem_id)
        return _fake_baker(candidate)

    resumed = _run_bake_work_items(
        work_items,
        cache_path=cache_path,
        workers=1,
        bake_candidate=counting_baker,
    )

    assert first_run_calls == ["toy-0", "toy-1", "toy-2"]
    assert resumed_calls == ["toy-2", "toy-3", "toy-4"]
    assert resumed.resumed_count == 2
    assert resumed.computed_count == 3
    assert len(resumed.problems) == 5
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 5


def test_changed_candidate_hash_logs_and_recomputes_then_reports_resume_counts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    cache_path = tmp_path / "changed-input.jsonl"
    original = _candidate(7)
    changed = replace(
        original,
        reference_solutions=("def solve(value):\n    return value * 7\n",),
    )
    calls: list[str] = []

    def counting_baker(candidate: BakeCandidate) -> BakeProblemResult:
        calls.append(candidate.reference_solutions[0])
        return _fake_baker(candidate)

    initial = _run_bake_work_items(
        _work_items((original,)),
        cache_path=cache_path,
        workers=1,
        bake_candidate=counting_baker,
    )
    recomputed = _run_bake_work_items(
        _work_items((changed,)),
        cache_path=cache_path,
        workers=1,
        bake_candidate=counting_baker,
    )
    resumed = _run_bake_work_items(
        _work_items((changed,)),
        cache_path=cache_path,
        workers=1,
        bake_candidate=counting_baker,
    )

    assert initial.computed_count == 1
    assert recomputed.computed_count == 1
    assert recomputed.resumed_count == 0
    assert resumed.computed_count == 0
    assert resumed.resumed_count == 1
    assert calls == [original.reference_solutions[0], changed.reference_solutions[0]]
    assert "cache input hash mismatch" in caplog.text
    assert "progress: 1/1" in caplog.text
    assert "kept=1" in caplog.text
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 1


def test_pool_logs_progress_while_verification_is_still_running(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(etl, "PROGRESS_INTERVAL_SECONDS", 0.01)

    def slow_baker(candidate: BakeCandidate) -> BakeProblemResult:
        time.sleep(0.05)
        return _fake_baker(candidate)

    _run_bake_work_items(
        _work_items((_candidate(9),)),
        cache_path=tmp_path / "progress.jsonl",
        workers=1,
        bake_candidate=slow_baker,
    )

    assert any("progress: 0/1" in record.message for record in caplog.records)
    assert any("eta=unknown" in record.message for record in caplog.records)


def test_drop_reasons_are_cached_and_counted_on_resume(tmp_path: Path) -> None:
    candidate = _candidate(12)
    work_item = _make_candidate_work_item(
        candidate,
        etl_code_hash="synthetic-etl-hash",
        source_revision="synthetic-source-revision",
        drop_reason="synthetic_drop",
    )
    cache_path = tmp_path / "drops.jsonl"

    def unused_baker(_: BakeCandidate) -> BakeProblemResult:
        raise AssertionError("a precomputed drop must not reach the verifier")

    computed = _run_bake_work_items(
        (work_item,),
        cache_path=cache_path,
        workers=2,
        bake_candidate=unused_baker,
    )
    resumed = _run_bake_work_items(
        (work_item,),
        cache_path=cache_path,
        workers=2,
        bake_candidate=unused_baker,
    )

    assert computed.drop_reasons == {"synthetic_drop": 1}
    assert computed.computed_count == 1
    assert resumed.drop_reasons == {"synthetic_drop": 1}
    assert resumed.resumed_count == 1
    assert resumed.computed_count == 0
    assert resumed.problems == ()


def test_cache_with_a_compatible_older_schema_is_a_miss_and_recomputed(tmp_path: Path) -> None:
    work_item = _work_items((_candidate(22),))[0]
    cache_path = tmp_path / "old-schema.jsonl"
    first = _run_bake_work_items(
        (work_item,),
        cache_path=cache_path,
        workers=1,
        bake_candidate=_fake_baker,
    )
    cached = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[0])
    cached["schema"] = etl.CACHE_SCHEMA - 1
    cache_path.write_text(json.dumps(cached) + "\n", encoding="utf-8")
    calls: list[str] = []

    def counting_baker(candidate: BakeCandidate) -> BakeProblemResult:
        calls.append(candidate.source_problem_id)
        return _fake_baker(candidate)

    resumed = _run_bake_work_items(
        (work_item,),
        cache_path=cache_path,
        workers=1,
        bake_candidate=counting_baker,
    )

    assert first.computed_count == 1
    assert resumed.resumed_count == 0
    assert resumed.computed_count == 1
    assert calls == ["toy-22"]


class TestReferenceTestVerdict:
    @staticmethod
    def _result(exit_code: int, *, timed_out: bool = False) -> JailedPytestResult:
        return JailedPytestResult(
            passed=exit_code == 0, exit_code=exit_code, timed_out=timed_out, stdout="", stderr=""
        )

    @pytest.mark.parametrize(("exit_code", "expected"), [(0, True), (1, False)])
    def test_measured_outcomes(self, exit_code: int, expected: bool) -> None:
        assert etl._reference_test_verdict(self._result(exit_code)) is expected

    def test_resource_limit_kill_and_timeout_are_unmeasured(self) -> None:
        assert etl._reference_test_verdict(self._result(etl.SIGKILL_EXIT_CODE)) is None
        assert etl._reference_test_verdict(self._result(124, timed_out=True)) is None

    def test_broken_runner_still_fails_fast(self) -> None:
        with pytest.raises(RuntimeError, match="did not produce a verdict"):
            etl._reference_test_verdict(self._result(2))
