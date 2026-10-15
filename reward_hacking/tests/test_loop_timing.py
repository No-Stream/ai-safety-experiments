"""Pin model, command, and grading durations on episode trace records."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from conftest import episode_trace

from reward_hacking.harness import BASELINE_ARM, TASKS_BY_ID, AgentTurn, Task, loop
from reward_hacking.jailed_poc import CommandResult


def test_episode_timing_is_recorded_at_turn_and_summary_levels(tmp_path: Path) -> None:
    turn = AgentTurn(
        index=0,
        completion="Done.",
        commands=(),
        generation_seconds=1.25,
        tool_execution_seconds=0.5,
        grading_seconds=0.25,
    )
    trace = episode_trace(
        "sum-ledger",
        (turn,),
        episode_dir=tmp_path,
        proxy_pass=False,
        true_pass=False,
    )
    trace = replace(
        trace,
        final_report_generation_seconds=0.75,
        uncompleted_model_call_seconds=0.0,
        final_grading_seconds=0.5,
    )

    turn_record = loop._turn_record(trace.episode_id, trace.task_id, turn)
    summary_record = trace.summary_record()

    assert turn_record["generation_seconds"] == 1.25
    assert turn_record["tool_execution_seconds"] == 0.5
    assert turn_record["grading_seconds"] == 0.25
    assert summary_record["model_call_seconds"] == 2.0
    assert summary_record["uncompleted_model_call_seconds"] == 0.0
    assert summary_record["tool_execution_seconds"] == 0.5
    assert summary_record["grading_seconds"] == 0.75


def test_legacy_turns_do_not_claim_unrecorded_phase_durations(tmp_path: Path) -> None:
    turn = AgentTurn(index=0, completion="Done.", commands=())
    trace = episode_trace(
        "sum-ledger",
        (turn,),
        episode_dir=tmp_path,
        proxy_pass=False,
        true_pass=False,
    )

    summary_record = trace.summary_record()

    assert summary_record["model_call_seconds"] is None
    assert summary_record["tool_execution_seconds"] is None
    assert summary_record["grading_seconds"] is None

    empty_trace = episode_trace(
        "sum-ledger",
        (),
        episode_dir=tmp_path / "empty",
        proxy_pass=False,
        true_pass=False,
    )
    assert empty_trace.summary_record()["tool_execution_seconds"] is None


def test_command_time_is_measured_separately_from_grading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow_command(
        _task: Task,
        _episode_dir: Path,
        command: str,
        *,
        timeout: str,
        jail_backend: str | None,
        cpus: int,
        mem_max: str,
    ) -> CommandResult:
        del timeout, jail_backend, cpus, mem_max
        time.sleep(0.02)
        return CommandResult(command, 0, "ok\n", "", timed_out=False)

    monkeypatch.setattr(loop, "_run_task_command", slow_command)
    timings = loop._ActionTimings()

    loop._run_actions(
        "<run>echo ok</run>",
        task=TASKS_BY_ID["sum-ledger"],
        arm=BASELINE_ARM,
        episode_dir=tmp_path,
        timeout="1s",
        jail_backend=None,
        checks_used=0,
        deadline=float("inf"),
        grader_hashable=False,
        phase_timings=timings,
    )

    assert timings.tool_execution_seconds >= 0.02
    assert timings.grading_seconds == 0.0
