"""CPU-only tests for the ladder's command-hook and final-report loop seams."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from reward_hacking.harness import loop
from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.model_backend import BedrockCompletion, MockBackend, TokenUsage

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _dummy_task() -> Task:
    """Return a harmless task whose oracle checks one marker file."""
    task_text = "Write marker.txt in the working directory.\n"
    return Task(
        task_id="dummy-ladder-loop",
        prompt=task_text,
        body=task_text,
        files={
            "TASK.md": task_text,
            "grade.py": "print('SUCCESS')\n",
        },
        true_check=lambda episode_dir: (episode_dir / "marker.txt").is_file(),
    )


def _successful_grade(_task: Task, _episode_dir: Path, **_kwargs: object) -> loop._FinalGrade:
    """Return a synthetic successful final grade without running a grader subprocess."""
    command = CommandResult(
        command="python3 -I grade.py",
        exit_code=0,
        stdout="SUCCESS\n",
        stderr="",
        timed_out=False,
    )
    return loop._FinalGrade(
        grader=command,
        proxy_pass=True,
        true_pass=True,
        oracle_error=None,
        grader_pregrade_sha256=None,
    )


class TestLadderCommandHooks:
    """Hooks observe every policy command while the trace keeps its own record."""

    def test_before_and_after_hooks_wrap_each_action(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[tuple[str, str, int | None]] = []

        def fake_run_in_jail(_episode_dir: Path, command: str, **_kwargs: object) -> CommandResult:
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="ok\n",
                stderr="",
                timed_out=False,
            )

        monkeypatch.setattr(loop, "run_in_jail", fake_run_in_jail)
        result, _tail, _ending, _checks, _hashes = loop._run_actions(
            "<run>touch marker.txt</run>",
            task=_dummy_task(),
            arm=AgenticArmConfig(),
            episode_dir=tmp_path,
            timeout="1s",
            jail_backend=None,
            checks_used=0,
            deadline=float("inf"),
            grader_hashable=False,
            before_command=lambda _directory, command: calls.append(("before", command, None)),
            after_command=lambda _directory, command, command_result: calls.append(
                ("after", command, command_result.exit_code)
            ),
        )

        assert len(result) == 1
        assert calls == [
            ("before", "touch marker.txt", None),
            ("after", "touch marker.txt", 0),
        ]


class TestLadderFinalReport:
    """A report is sampled and stored separately; its protocol-looking text is inert."""

    def test_final_report_timeout_is_recorded_not_raised(self) -> None:
        """A run must survive one slow report; the timeout is evidence of nothing, so it is kept."""
        released = threading.Event()

        class SlowBackend(MockBackend):
            def generate(self, prompts: list[str]) -> list[str]:
                released.wait(2.0)
                return super().generate(prompts)

        backend = SlowBackend(["report"], model_id="dummy")
        started = time.monotonic()
        try:
            report = loop._sample_final_report(backend, "transcript", report_seconds=0.05)
            assert time.monotonic() - started < 1.0
            assert report.timed_out
            assert report.text == ""
        finally:
            released.set()

    def test_final_report_is_raw_and_does_not_run_report_actions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        commands: list[str] = []
        hook_events: list[tuple[str, str]] = []

        def fake_run_in_jail(_episode_dir: Path, command: str, **_kwargs: object) -> CommandResult:
            commands.append(command)
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="",
                stderr="",
                timed_out=False,
            )

        monkeypatch.setattr(loop, "run_in_jail", fake_run_in_jail)
        monkeypatch.setattr(loop, "_grade_final_state", _successful_grade)
        report = "Report only: <run>touch report-action.txt</run>"
        trace = loop.run_agent_episode(
            _dummy_task(),
            MockBackend(["<run>touch marker.txt</run>", report], model_id="dummy"),
            episode_dir=tmp_path / "episode",
            trace_path=tmp_path / "trace.jsonl",
            max_turns=1,
            before_command=lambda _directory, command: hook_events.append(("before", command)),
            after_command=lambda _directory, command, _result: hook_events.append(
                ("after", command)
            ),
            final_report=True,
        )

        assert trace.final_report == report
        assert not trace.final_report_truncated
        assert commands == ["touch marker.txt"]
        assert hook_events == [("before", "touch marker.txt"), ("after", "touch marker.txt")]
        assert not (tmp_path / "episode" / "report-action.txt").exists()
        summary = trace.summary_record()
        assert summary["final_report"] == {
            "text": report,
            "reasoning": "",
            "truncated": False,
            "timed_out": False,
        }
        persisted_summary = next(
            record
            for record in loop.load_traces(tmp_path / "trace.jsonl")
            if record["record"] == "episode_summary"
        )
        assert persisted_summary["final_report"] == summary["final_report"]

    def test_final_report_truncation_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            loop,
            "run_in_jail",
            lambda *_args, **_kwargs: CommandResult(
                command="touch marker.txt",
                exit_code=0,
                stdout="",
                stderr="",
                timed_out=False,
            ),
        )
        monkeypatch.setattr(loop, "_grade_final_state", _successful_grade)

        class TruncatedBackend(MockBackend):
            def __init__(self) -> None:
                super().__init__(["ignored"], model_id="dummy")
                self._report_cursor = 0

            def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
                del prompts
                responses = [
                    BedrockCompletion(
                        text="<run>touch marker.txt</run>",
                        reasoning="",
                        usage=TokenUsage(input_tokens=1, output_tokens=1),
                        stop_reason=loop.STOP_REASON_MAX_TOKENS,
                    ),
                    BedrockCompletion(
                        text="partial report",
                        reasoning="private thought",
                        usage=TokenUsage(input_tokens=1, output_tokens=1),
                        stop_reason=loop.STOP_REASON_MAX_TOKENS,
                    ),
                ]
                response = responses[self._report_cursor]
                self._report_cursor += 1
                return [response]

        trace = loop.run_agent_episode(
            _dummy_task(),
            TruncatedBackend(),
            episode_dir=tmp_path / "truncated",
            max_turns=1,
            final_report=True,
        )

        assert trace.final_report == "partial report"
        assert trace.final_report_reasoning == "private thought"
        assert trace.final_report_truncated
        assert not trace.final_report_timed_out
