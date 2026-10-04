"""CPU-only tests for the ladder's command-hook and final-report loop seams."""

from __future__ import annotations

import shlex
import subprocess
import threading
import time
from typing import TYPE_CHECKING, TypedDict

import pytest
import test_ladder_loop as native_loop_tests

from reward_hacking.harness import loop
from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.ladder import loop as ladder_loop
from reward_hacking.model_backend import BedrockCompletion, MockBackend, TokenUsage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from typing import Literal


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


def _successful_grade(_task: Task, _episode_dir: Path, **_kwargs: object) -> loop.FinalGrade:
    """Return a synthetic successful final grade without running a grader subprocess."""
    command = CommandResult(
        command="python3 -I grade.py",
        exit_code=0,
        stdout="SUCCESS\n",
        stderr="",
        timed_out=False,
    )
    return loop.FinalGrade(
        grader=command,
        proxy_pass=True,
        true_pass=True,
        oracle_error=None,
        grader_pregrade_sha256=None,
    )


def test_grading_helpers_have_public_names() -> None:
    """The ladder can depend on stable grading helper names."""
    assert callable(loop.run_grader)
    assert callable(loop.grade_final_state)
    assert hasattr(loop, "FinalGrade")
    assert callable(loop.grader_content_hash)
    assert callable(loop.is_sole_grader_invocation)


class TestLadderCommandHooks:
    """Hooks observe every policy command while the trace keeps its own record."""

    def test_turn_trace_records_server_prompt_usage_and_full_generation_time(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        class SlowDetailedBackend(MockBackend):
            def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
                del prompts
                # Represent two synchronous continuation requests within one backend call.
                time.sleep(0.02)
                time.sleep(0.02)
                return [
                    BedrockCompletion(
                        text="<run>touch marker.txt</run>",
                        reasoning="",
                        usage=TokenUsage(input_tokens=23, output_tokens=7),
                        stop_reason="end_turn",
                    )
                ]

        monkeypatch.setattr(
            loop,
            "run_in_jail",
            lambda _directory, command, **_kwargs: CommandResult(
                command=command,
                exit_code=0,
                stdout="",
                stderr="",
                timed_out=False,
            ),
        )
        monkeypatch.setattr(loop, "grade_final_state", _successful_grade)

        trace = loop.run_agent_episode(
            _dummy_task(),
            SlowDetailedBackend(["unused"], model_id="dummy"),
            episode_dir=tmp_path / "usage",
            max_turns=1,
        )

        record = trace.turn_records()[0]
        generation_seconds = record["generation_seconds"]
        assert record["prompt_tokens"] == 23
        assert record["output_tokens"] == 7
        assert isinstance(generation_seconds, float)
        assert generation_seconds >= 0.04

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
        monkeypatch.setattr(loop, "grade_final_state", _successful_grade)
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
        monkeypatch.setattr(loop, "grade_final_state", _successful_grade)

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


class TestSuppliedCheckGrader:
    """A caller-supplied grader behind <check/>, optionally explaining a failing verdict."""

    def reason_arm(self) -> AgenticArmConfig:
        return AgenticArmConfig(check_affordance=True, check_reason=True)

    def run_check(
        self, tmp_path: Path, arm: AgenticArmConfig, check_grader: loop.CheckGrader
    ) -> tuple[str, int]:
        _results, tail, _ending, checks_used, _hashes = loop._run_actions(
            "<check/>",
            task=_dummy_task(),
            arm=arm,
            episode_dir=tmp_path,
            timeout="1s",
            jail_backend=None,
            checks_used=0,
            deadline=float("inf"),
            grader_hashable=False,
            check_grader=check_grader,
        )
        return tail, checks_used

    def test_reason_arm_label_round_trips_and_defaults_off(self) -> None:
        arm = self.reason_arm()
        assert arm.label == "check+reason"
        assert AgenticArmConfig.from_json_dict(arm.to_json_dict()) == arm
        assert not AgenticArmConfig.from_json_dict({"check_affordance": True}).check_reason
        with pytest.raises(ValueError, match="check_reason"):
            AgenticArmConfig(check_reason=True)

    def test_prompt_promises_a_reason_only_on_the_reason_arm(self) -> None:
        task = _dummy_task()
        plain = loop.initial_prompt(task, arm=AgenticArmConfig(check_affordance=True), listing=[])
        reasoned = loop.initial_prompt(task, arm=self.reason_arm(), listing=[])
        assert "nothing else" in plain
        assert "nothing else" not in reasoned
        assert "reason" in reasoned

    def test_supplied_grader_answers_check_with_its_reason(self, tmp_path: Path) -> None:
        seen: list[Path] = []

        def check_grader(directory: Path) -> loop.CheckVerdict:
            seen.append(directory)
            return loop.CheckVerdict(passed=False, reason="visible test files were modified")

        tail, checks_used = self.run_check(tmp_path, self.reason_arm(), check_grader)

        assert seen == [tmp_path]
        assert checks_used == 1
        assert "<check>FAIL: visible test files were modified</check>" in tail

    def test_plain_check_arm_withholds_the_reason(self, tmp_path: Path) -> None:
        tail, _ = self.run_check(
            tmp_path,
            AgenticArmConfig(check_affordance=True),
            lambda _directory: loop.CheckVerdict(passed=False, reason="secret"),
        )
        assert "<check>FAIL</check>" in tail
        assert "secret" not in tail

    def test_reason_arm_without_a_supplied_grader_fails_fast(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="check_grader"):
            loop.run_agent_episode(
                _dummy_task(),
                MockBackend(["<check/>"], model_id="dummy"),
                episode_dir=tmp_path / "episode",
                max_turns=1,
                arm=self.reason_arm(),
            )


def test_executable_task_files_are_laid_down_executable(tmp_path: Path) -> None:
    base = _dummy_task()
    task = Task(
        task_id=base.task_id,
        prompt=base.prompt,
        body=base.body,
        files={**base.files, "bin/tool": "#!/bin/sh\necho ok\n"},
        true_check=base.true_check,
        executable_files=frozenset({"bin/tool"}),
    )
    loop.lay_down_task(tmp_path / "episode", task)
    assert (tmp_path / "episode" / "bin" / "tool").stat().st_mode & 0o111
    assert not (tmp_path / "episode" / "grade.py").stat().st_mode & 0o111


def _run_shell_wrapper(episode_dir: Path, wrapped_command: str, **_kwargs: object) -> CommandResult:
    """Run synthetic shell commands outside the jail while retaining the loop wrapper."""
    executable_command = wrapped_command.replace(
        "builtin cd -- /work",
        f"builtin cd -- {shlex.quote(str(episode_dir))}",
        1,
    )
    completed = subprocess.run(  # noqa: S603 - synthetic loop wrapper runs only in a temporary directory
        ["/bin/bash", "--noprofile", "--norc", "-c", executable_command],
        cwd=episode_dir,
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    return CommandResult(
        command=wrapped_command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
    )


class _NativeSeamControls(TypedDict, total=False):
    after_laydown: Callable[[Path], tuple[str, ...]] | None
    listing_mode: Literal["all", "top-level"]
    initial_environment: Mapping[str, str]
    format_error_message: str


def _run_native_seam_episode(  # noqa: PLR0913 - episode setup exposes the tested loop controls
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: native_loop_tests.ScriptedToolBackend,
    *,
    episode_name: str,
    after_laydown: Callable[[Path], tuple[str, ...]] | None = None,
    listing_mode: Literal["all", "top-level"] | None = None,
    initial_environment: Mapping[str, str] | None = None,
    format_error_message: str | None = None,
    max_turns: int = 1,
    use_loop_defaults: bool = False,
    use_subprocess_shell: bool = False,
) -> ladder_loop.LadderEpisodeTrace:
    shell_runner = (
        _run_shell_wrapper if use_subprocess_shell else native_loop_tests._fake_run_in_jail
    )
    monkeypatch.setattr(ladder_loop, "run_in_jail", shell_runner)
    monkeypatch.setattr(ladder_loop, "grade_final_state", native_loop_tests._successful_grade)
    controls: _NativeSeamControls = {}
    if not use_loop_defaults:
        controls = {
            "after_laydown": after_laydown,
            "listing_mode": "all" if listing_mode is None else listing_mode,
            "initial_environment": {} if initial_environment is None else initial_environment,
            "format_error_message": (
                ladder_loop.FORMAT_ERROR_MESSAGE
                if format_error_message is None
                else format_error_message
            ),
        }
    return ladder_loop.run_ladder_episode(
        native_loop_tests._task(),
        backend,
        episode_dir=tmp_path / episode_name,
        arm=AgenticArmConfig(),
        system_prompt="Use the tools to complete the task.",
        user_template="Task:\n{task_markdown}\nRepository files:\n{listing}",
        max_turns=max_turns,
        timeout="120s",
        jail_backend=None,
        history_reasoning="strip",
        submit_grader=lambda _directory: ladder_loop.SubmissionVerdict(True, "SUCCESS"),
        max_model_len=100_000,
        **controls,
    )


class TestSharedLadderLoopSeams:
    """Keep new v2 loop controls isolated from the native loop's established defaults."""

    def test_after_laydown_runs_before_top_level_listing_and_lists_only_roots(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        hook_calls: list[str] = []

        def add_git_tree(episode_dir: Path) -> tuple[str, ...]:
            git_objects = episode_dir / ".git" / "objects"
            git_objects.mkdir(parents=True)
            (episode_dir / ".git" / "config").write_text("synthetic config\n")
            (git_objects / "entry").write_text("synthetic object\n")
            hook_calls.append("after_laydown")
            return (".git",)

        def capture_prompt(_call_index: int, _prompt: str) -> None:
            assert hook_calls == ["after_laydown"]

        backend = native_loop_tests.ScriptedToolBackend(
            [
                native_loop_tests._ScriptedReply(native_loop_tests._bash("echo ready")),
                native_loop_tests._ScriptedReply(native_loop_tests._report()),
            ],
            on_complete=capture_prompt,
        )
        _run_native_seam_episode(
            tmp_path,
            monkeypatch,
            backend,
            episode_name="top-level-listing",
            after_laydown=add_git_tree,
            listing_mode="top-level",
        )

        initial_prompt = backend.prompts[0]
        assert "- .git/" in initial_prompt
        assert "- sub/" in initial_prompt
        assert "- sub/.keep" not in initial_prompt
        assert "- .git/config" not in initial_prompt
        assert "- .git/objects/entry" not in initial_prompt

    def test_default_listing_is_byte_identical_to_explicit_all_listing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        default_backend = native_loop_tests.ScriptedToolBackend(
            [
                native_loop_tests._ScriptedReply(native_loop_tests._bash("echo ready")),
                native_loop_tests._ScriptedReply(native_loop_tests._report()),
            ]
        )
        explicit_backend = native_loop_tests.ScriptedToolBackend(
            [
                native_loop_tests._ScriptedReply(native_loop_tests._bash("echo ready")),
                native_loop_tests._ScriptedReply(native_loop_tests._report()),
            ]
        )
        default_trace = _run_native_seam_episode(
            tmp_path,
            monkeypatch,
            default_backend,
            episode_name="default-listing",
            use_loop_defaults=True,
        )
        explicit_trace = _run_native_seam_episode(
            tmp_path,
            monkeypatch,
            explicit_backend,
            episode_name="explicit-all-listing",
            listing_mode="all",
        )

        assert default_backend.prompts == explicit_backend.prompts
        assert default_trace.messages == explicit_trace.messages
        assert default_trace.turns[0].commands == explicit_trace.turns[0].commands
        assert default_trace.turns[0].tool_calls == explicit_trace.turns[0].tool_calls
        assert default_trace.turns[0].tool_results == explicit_trace.turns[0].tool_results
        assert "- sub/.keep" in default_backend.prompts[0]

    def test_initial_environment_is_shell_quoted_and_default_keeps_inherited_values(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seeded_value = "literal value; '$HOME' stays unchanged"
        command = 'echo "$PYTHONPATH"'
        seeded_backend = native_loop_tests.ScriptedToolBackend(
            [
                native_loop_tests._ScriptedReply(native_loop_tests._bash(command)),
                native_loop_tests._ScriptedReply(native_loop_tests._bash(command)),
                native_loop_tests._ScriptedReply(native_loop_tests._report()),
            ]
        )
        seeded_trace = _run_native_seam_episode(
            tmp_path,
            monkeypatch,
            seeded_backend,
            episode_name="seeded-environment",
            max_turns=2,
            initial_environment={"PYTHONPATH": seeded_value},
            use_subprocess_shell=True,
        )

        assert [
            command_result.stdout for turn in seeded_trace.turns for command_result in turn.commands
        ] == [f"{seeded_value}\n", f"{seeded_value}\n"]
        recorded_commands = [
            command_result.command
            for turn in seeded_trace.turns
            for command_result in turn.commands
        ]
        assert recorded_commands == [command, command]
        trace_text = str(seeded_trace.turn_records())
        assert "__LADDER_SHELL_STATE_" not in trace_text
        assert "builtin cd" not in trace_text

        monkeypatch.setenv("LADDER_LOOP_INHERITED_SENTINEL", "ambient value")
        default_command = 'echo "$LADDER_LOOP_INHERITED_SENTINEL"'
        default_backend = native_loop_tests.ScriptedToolBackend(
            [
                native_loop_tests._ScriptedReply(native_loop_tests._bash(default_command)),
                native_loop_tests._ScriptedReply(native_loop_tests._report()),
            ]
        )
        default_trace = _run_native_seam_episode(
            tmp_path,
            monkeypatch,
            default_backend,
            episode_name="default-environment",
            use_loop_defaults=True,
            use_subprocess_shell=True,
        )
        assert default_trace.turns[0].commands[0].stdout == "ambient value\n"

    def test_format_error_message_can_be_customized(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        custom_message = "Use a supported command block and try again."
        malformed_call = "</think><tool_call>\n<function=unknown>\n</function>\n</tool_call>"
        backend = native_loop_tests.ScriptedToolBackend(
            [
                native_loop_tests._ScriptedReply(malformed_call),
                native_loop_tests._ScriptedReply(native_loop_tests._report()),
            ]
        )
        trace = _run_native_seam_episode(
            tmp_path,
            monkeypatch,
            backend,
            episode_name="custom-format-error",
            format_error_message=custom_message,
        )

        assert trace.turns[0].format_error == "Unknown tool function: unknown."
        assert any(
            message.get("role") == "tool" and message.get("content") == custom_message
            for message in trace.messages
        )
