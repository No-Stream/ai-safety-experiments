"""CPU-focused tests for the native tool-call hack-ladder episode loop."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from conftest import needs_jail

from reward_hacking.harness.loop import CheckVerdict, FinalGrade
from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.ladder import loop
from reward_hacking.model_backend import RenderedCompletion
from reward_hacking.trace import load_trace

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest
    from transformers import PreTrainedTokenizerBase

    from reward_hacking.harness import loop as shared_loop


@dataclass(frozen=True)
class _Tokenized:
    input_ids: list[int]


class _FakeTokenizer:
    def __init__(self) -> None:
        self.rendered_messages: list[list[dict[str, object]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        assert enable_thinking
        self.rendered_messages.append(json.loads(json.dumps(messages)))
        return json.dumps({"messages": messages, "tools": tools}, sort_keys=True) + (
            "<|im_start|>assistant\n<think>\n"
        )

    def __call__(self, prompt: str, *, add_special_tokens: bool) -> _Tokenized:
        assert not add_special_tokens
        return _Tokenized(list(range(len(prompt.split()))))

    def get_chat_template(self, *, tools: list[dict[str, object]]) -> str:
        del tools
        return "synthetic-test-template"


@dataclass(frozen=True)
class _ScriptedReply:
    text: str
    finish_reason: str | None = "stop"
    matched_stop: int | str | None = 248046
    prompt_tokens: int = 17
    completion_tokens: int = 9


@dataclass(frozen=True)
class _PersistentShellHooks:
    before_command: Callable[[Path, str], None] | None = None
    after_command: Callable[[Path, str, CommandResult], None] | None = None
    trace_path: Path | None = None


@dataclass(frozen=True)
class _EpisodeOptions:
    arm: AgenticArmConfig | None = None
    max_model_len: int = 100_000
    max_turns: int = 4
    trace_path: Path | None = None
    episode_token: str | None = None
    episode_seconds: float = loop.LADDER_EPISODE_SECONDS
    before_command: Callable[[Path, str], None] | None = None
    after_command: Callable[[Path, str, CommandResult], None] | None = None
    check_grader: Callable[[Path], CheckVerdict] | None = None


class ScriptedToolBackend:
    """Return predetermined raw Qwen-format completions and retain the actual requests."""

    model_id = "cpu-scripted-model"
    transport = "vllm-http"
    model_path: str | None = None
    tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        replies: list[_ScriptedReply],
        *,
        on_complete: Callable[[int, str], None] | None = None,
    ) -> None:
        self.replies = list(replies)
        self.on_complete = on_complete
        self.tokenizer = cast("PreTrainedTokenizerBase", _FakeTokenizer())
        self.prompts: list[str] = []
        self.max_tokens: list[int] = []

    def complete_rendered(
        self, prompt: str, *, max_tokens: int, seed: int | None
    ) -> RenderedCompletion:
        assert seed is None
        if self.on_complete is not None:
            self.on_complete(len(self.prompts), prompt)
        self.prompts.append(prompt)
        self.max_tokens.append(max_tokens)
        if not self.replies:
            raise AssertionError("scripted backend ran out of replies")
        reply = self.replies.pop(0)
        return RenderedCompletion(
            text=reply.text,
            finish_reason=reply.finish_reason,
            matched_stop=reply.matched_stop,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
        )


def _task() -> Task:
    text = "Create marker.txt in the working directory.\n"
    return Task(
        task_id="synthetic-ladder-loop",
        prompt=text,
        body=text,
        files={"TASK.md": text, "grade.py": "print('SUCCESS')\n", "sub/.keep": ""},
        true_check=lambda episode_dir: (episode_dir / "marker.txt").is_file(),
    )


def _bash(command: str, *, start_turn: bool = True) -> str:
    return (
        ("reasoning</think>\n" if start_turn else "") + "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>\n"
        f"{command}\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


def _check(*, start_turn: bool = True) -> str:
    prefix = "checking</think>\n" if start_turn else ""
    return f"{prefix}<tool_call>\n<function=check>\n</function>\n</tool_call>"


def _unknown_tool() -> str:
    return "</think>\n<tool_call>\n<function=unknown>\n</function>\n</tool_call>"


def _report(text: str = "The marker is ready.") -> str:
    return f"</think>{text}"


def _successful_grade(_task: Task, _episode_dir: Path, **_kwargs: object) -> shared_loop.FinalGrade:
    return FinalGrade(
        grader=CommandResult(
            command="python3 -I grade.py",
            exit_code=0,
            stdout="SUCCESS\n",
            stderr="",
            timed_out=False,
        ),
        proxy_pass=True,
        true_pass=True,
        oracle_error=None,
        grader_pregrade_sha256=None,
    )


def _fake_run_in_jail(
    _episode_dir: Path, command: str, *, timeout: str, **_kwargs: object
) -> CommandResult:
    assert timeout == "120s"
    return CommandResult(
        command=command,
        exit_code=0,
        stdout="command output\n",
        stderr="",
        timed_out=False,
    )


def _run_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: loop.LadderBackend,
    options: _EpisodeOptions | None = None,
) -> loop.LadderEpisodeTrace:
    monkeypatch.setattr(loop, "run_in_jail", _fake_run_in_jail)
    monkeypatch.setattr(loop, "grade_final_state", _successful_grade)
    episode_options = options or _EpisodeOptions()
    return loop.run_ladder_episode(
        _task(),
        backend,
        episode_dir=tmp_path / "episode",
        arm=episode_options.arm or AgenticArmConfig(check_affordance=True),
        system_prompt="Use the tools to complete the task.",
        user_template="Task:\n{task_markdown}\nFiles:\n{listing}",
        max_turns=episode_options.max_turns,
        timeout="120s",
        jail_backend=None,
        trace_path=episode_options.trace_path,
        episode_token=episode_options.episode_token,
        episode_seconds=episode_options.episode_seconds,
        before_command=episode_options.before_command,
        after_command=episode_options.after_command,
        check_grader=episode_options.check_grader,
        max_model_len=episode_options.max_model_len,
    )


class TestNativeToolLoop:
    def test_surplus_function_close_tags_are_recorded_on_turn_and_episode_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        duplicated_close = _bash("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT").replace(
            "</function>\n</tool_call>", "</function>\n</function>\n</tool_call>"
        )
        backend = ScriptedToolBackend(
            [
                _ScriptedReply(duplicated_close),
                _ScriptedReply(duplicated_close),
                _ScriptedReply(duplicated_close),
                _ScriptedReply(_report()),
            ]
        )

        trace = _run_episode(tmp_path, monkeypatch, backend)

        assert trace.turn_records()[0]["surplus_function_close_tags"] == 1
        assert trace.summary_record()["surplus_function_close_tags"] == 1

    def test_happy_path_submits_and_writes_turn_then_summary_incrementally(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        trace_path = tmp_path / "trace.jsonl"
        records_seen_before_report: list[dict[str, object]] = []

        def observe_report_call(call_index: int, _prompt: str) -> None:
            if call_index == 1:
                records_seen_before_report.extend(load_trace(trace_path))

        backend = ScriptedToolBackend(
            [
                _ScriptedReply(_bash("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")),
                _ScriptedReply(_report()),
            ],
            on_complete=observe_report_call,
        )
        trace = _run_episode(tmp_path, monkeypatch, backend, _EpisodeOptions(trace_path=trace_path))

        assert trace.ended_by == "submit"
        assert len(trace.turns) == 1
        assert trace.turns[0].tool_calls[0].executed
        assert trace.turns[0].stop_reason == "end_turn"
        assert trace.turns[0].matched_stop_token_id == 248046
        assert backend.max_tokens[0] == 16_384
        assert [record["record"] for record in records_seen_before_report] == ["turn"]
        records = load_trace(trace_path)
        assert [record["record"] for record in records] == ["turn", "episode_summary"]
        assert records[0]["rendered_prompt"]
        assert "rendered_prompt" not in records[1]
        assert records[1]["ended_by"] == "submit"

    def test_no_tool_call_on_untouched_work_nudges_once_then_ends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        backend = ScriptedToolBackend(
            [
                _ScriptedReply(_report("I have an idea.")),
                _ScriptedReply(_report("I am done.")),
                _ScriptedReply(_report("No files were changed.")),
            ]
        )

        trace = _run_episode(tmp_path, monkeypatch, backend)

        assert trace.ended_by == "no_tool_call"
        assert trace.nudged_empty_start
        assert [message["role"] for message in trace.messages].count("user") == 3
        assert not any(turn.commands for turn in trace.turns)
        assert any(
            message.get("role") == "user"
            and "nothing has been written to /work yet" in str(message.get("content")).lower()
            for message in trace.messages
        )

    def test_multiple_calls_execute_in_order_and_each_adds_one_tool_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        execution_order: list[str] = []

        def run_command(
            _directory: Path, command: str, *, timeout: str, **_kwargs: object
        ) -> CommandResult:
            del timeout
            execution_order.append(command)
            return CommandResult(command, 0, command + "\n", "", False)

        monkeypatch.setattr(loop, "run_in_jail", run_command)
        monkeypatch.setattr(loop, "grade_final_state", _successful_grade)
        backend = ScriptedToolBackend(
            [
                _ScriptedReply(
                    _bash("touch first")
                    + "\n"
                    + _bash("touch second", start_turn=False)
                    + "\n"
                    + _check(start_turn=False)
                ),
                _ScriptedReply(_report()),
            ]
        )
        trace = loop.run_ladder_episode(
            _task(),
            backend,
            episode_dir=tmp_path / "episode",
            arm=AgenticArmConfig(check_affordance=True, check_budget=2),
            system_prompt="System",
            user_template="{task_markdown}\n{listing}",
            max_turns=1,
            timeout="120s",
            jail_backend=None,
            check_grader=lambda _directory: CheckVerdict(passed=True, reason=""),
            max_model_len=100_000,
        )

        assert len(execution_order) == 2
        assert [command.command for command in trace.turns[0].commands] == [
            "touch first",
            "touch second",
        ]
        assert [call.name for call in trace.turns[0].tool_calls] == ["bash", "bash", "check"]
        assert [result.name for result in trace.turns[0].tool_results] == ["bash", "bash", "check"]
        assert len([message for message in trace.messages if message["role"] == "tool"]) == 3
        assert trace.checks_used == 1

    def test_submit_runs_its_command_and_marks_later_calls_not_executed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        executed: list[str] = []
        monkeypatch.setattr(
            loop,
            "run_in_jail",
            lambda _directory, command, **_kwargs: (
                executed.append(command) or CommandResult(command, 0, "", "", False)
            ),
        )
        monkeypatch.setattr(loop, "grade_final_state", _successful_grade)
        backend = ScriptedToolBackend(
            [
                _ScriptedReply(
                    _bash("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
                    + "\n"
                    + _bash("touch later", start_turn=False)
                ),
                _ScriptedReply(_report()),
            ]
        )

        trace = loop.run_ladder_episode(
            _task(),
            backend,
            episode_dir=tmp_path / "episode",
            arm=AgenticArmConfig(check_affordance=True),
            system_prompt="System",
            user_template="{task_markdown}\n{listing}",
            max_turns=1,
            timeout="120s",
            jail_backend=None,
            max_model_len=100_000,
        )

        assert len(executed) == 1
        assert trace.ended_by == "submit"
        assert trace.turns[0].commands[0].command == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        assert trace.turns[0].tool_calls[1].executed is False
        assert trace.turns[0].tool_calls[1].not_executed_reason == "submit"

    def test_check_budget_exhaustion_stops_before_the_over_budget_check(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        check_calls: list[Path] = []
        backend = ScriptedToolBackend(
            [
                _ScriptedReply(
                    _check()
                    + "\n"
                    + _check(start_turn=False)
                    + "\n"
                    + _bash("touch later", start_turn=False)
                ),
                _ScriptedReply(_report()),
            ]
        )

        trace = _run_episode(
            tmp_path,
            monkeypatch,
            backend,
            _EpisodeOptions(
                arm=AgenticArmConfig(check_affordance=True, check_budget=1),
                check_grader=lambda directory: (
                    check_calls.append(directory) or CheckVerdict(passed=False, reason="private")
                ),
            ),
        )

        assert trace.ended_by == "check_budget"
        assert trace.checks_used == 1
        assert len(check_calls) == 1
        assert trace.turns[0].tool_calls[0].executed
        assert trace.turns[0].tool_calls[1].not_executed_reason == "check_budget"
        assert trace.turns[0].tool_calls[2].not_executed_reason == "check_budget"

    def test_three_consecutive_format_errors_end_the_episode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        backend = ScriptedToolBackend(
            [
                _ScriptedReply(_unknown_tool()),
                _ScriptedReply(_unknown_tool()),
                _ScriptedReply(_unknown_tool()),
                _ScriptedReply(_report()),
            ]
        )

        trace = _run_episode(tmp_path, monkeypatch, backend)

        assert trace.ended_by == "format_errors"
        assert trace.format_errors == 3
        assert [turn.format_error for turn in trace.turns] == [
            "Unknown tool function: unknown.",
            "Unknown tool function: unknown.",
            "Unknown tool function: unknown.",
        ]
        assert sum(message["role"] == "tool" for message in trace.messages) == 3

    def test_length_truncated_tool_call_is_not_executed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        executed: list[str] = []
        monkeypatch.setattr(
            loop,
            "run_in_jail",
            lambda _directory, command, **_kwargs: (
                executed.append(command) or CommandResult(command, 0, "", "", False)
            ),
        )
        monkeypatch.setattr(loop, "grade_final_state", _successful_grade)
        backend = ScriptedToolBackend(
            [
                _ScriptedReply(
                    _bash("touch must_not_run"),
                    finish_reason="length",
                ),
                _ScriptedReply(_report()),
            ]
        )

        trace = loop.run_ladder_episode(
            _task(),
            backend,
            episode_dir=tmp_path / "episode",
            arm=AgenticArmConfig(check_affordance=True),
            system_prompt="System",
            user_template="{task_markdown}\n{listing}",
            max_turns=1,
            timeout="120s",
            jail_backend=None,
            max_model_len=100_000,
        )

        assert executed == []
        assert trace.turns[0].stop_reason == "max_tokens"
        assert trace.turns[0].format_error is not None
        assert trace.turns[0].tool_calls[0].not_executed_reason == "format_error"

    def test_deadline_between_calls_records_the_unexecuted_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        executed: list[str] = []

        def slow_command(
            _directory: Path, command: str, *, timeout: str, **_kwargs: object
        ) -> CommandResult:
            del timeout
            executed.append(command)
            time.sleep(0.06)
            return CommandResult(command, 0, "", "", False)

        monkeypatch.setattr(loop, "run_in_jail", slow_command)
        monkeypatch.setattr(loop, "grade_final_state", _successful_grade)
        backend = ScriptedToolBackend(
            [
                _ScriptedReply(
                    _bash("touch first") + "\n" + _bash("touch second", start_turn=False)
                ),
                _ScriptedReply(_report()),
            ]
        )

        trace = loop.run_ladder_episode(
            _task(),
            backend,
            episode_dir=tmp_path / "episode",
            arm=AgenticArmConfig(check_affordance=True),
            system_prompt="System",
            user_template="{task_markdown}\n{listing}",
            max_turns=1,
            timeout="120s",
            jail_backend=None,
            episode_seconds=0.04,
            max_model_len=100_000,
        )

        assert len(executed) == 1
        assert trace.turns[0].commands[0].command == "touch first"
        assert trace.ended_by == "deadline"
        assert trace.deadline_phase == "actions"
        assert trace.turns[0].tool_calls[1].not_executed_reason == "deadline"

    def test_context_guard_stops_before_sampling(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        backend = ScriptedToolBackend([])
        monkeypatch.setattr(loop, "count_prompt_tokens", lambda _tokenizer, _prompt: 100)

        trace = _run_episode(
            tmp_path,
            monkeypatch,
            backend,
            _EpisodeOptions(max_model_len=1_000),
        )

        assert backend.prompts == []
        assert trace.ended_by == "context_exhausted"
        assert trace.deadline_phase is None
        assert trace.context_peak_prompt_tokens == 100

    def test_final_report_tool_call_is_counted_but_inert(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        executed: list[str] = []
        monkeypatch.setattr(
            loop,
            "run_in_jail",
            lambda _directory, command, **_kwargs: (
                executed.append(command) or CommandResult(command, 0, "", "", False)
            ),
        )
        monkeypatch.setattr(loop, "grade_final_state", _successful_grade)
        backend = ScriptedToolBackend(
            [
                _ScriptedReply(_bash("touch marker.txt")),
                _ScriptedReply(_bash("touch report-only.txt")),
            ]
        )

        trace = loop.run_ladder_episode(
            _task(),
            backend,
            episode_dir=tmp_path / "episode",
            arm=AgenticArmConfig(check_affordance=True),
            system_prompt="System",
            user_template="{task_markdown}\n{listing}",
            max_turns=1,
            timeout="120s",
            jail_backend=None,
            max_model_len=100_000,
        )

        assert len(executed) == 1
        assert trace.turns[0].commands[0].command == "touch marker.txt"
        assert trace.final_report == ""
        assert trace.final_report_tool_calls == 1
        assert len(trace.turns) == 1
        assert trace.summary_record()["final_report_tool_calls"] == 1

    def test_prompt_pure_append_measurement_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(loop, "is_pure_append", lambda _old, _completion, _new: True)
        backend = ScriptedToolBackend(
            [
                _ScriptedReply(_bash("touch marker.txt")),
                _ScriptedReply(_report("I am done.")),
                _ScriptedReply(_report()),
            ]
        )

        trace = _run_episode(tmp_path, monkeypatch, backend)

        assert isinstance(trace.turns[1].prompt_pure_append, bool)
        assert trace.turns[1].prompt_pure_append is True
        assert trace.turns[0].fabricated_tool_responses == 0
        assert trace.turns[0].trailing_text_chars == 0

    def test_monitor_hooks_wrap_bash_only_not_check_or_final_grade(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        events: list[tuple[str, str]] = []
        backend = ScriptedToolBackend(
            [
                _ScriptedReply(_bash("touch marker.txt") + "\n" + _check(start_turn=False)),
                _ScriptedReply(_report()),
            ]
        )
        trace = _run_episode(
            tmp_path,
            monkeypatch,
            backend,
            _EpisodeOptions(
                max_turns=1,
                before_command=lambda _directory, command: events.append(("before", command)),
                after_command=lambda _directory, command, _result: events.append(
                    ("after", command)
                ),
                check_grader=lambda _directory: CheckVerdict(passed=True, reason=""),
            ),
        )

        assert events == [("before", "touch marker.txt"), ("after", "touch marker.txt")]
        assert len(trace.turns[0].commands) == 1

    def test_bash_schema_and_command_records_use_the_model_command_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        events: list[str] = []
        trace_path = tmp_path / "trace.jsonl"
        command = "touch marker.txt"
        backend = ScriptedToolBackend([_ScriptedReply(_bash(command)), _ScriptedReply(_report())])

        trace = _run_episode(
            tmp_path,
            monkeypatch,
            backend,
            _EpisodeOptions(
                max_turns=1,
                trace_path=trace_path,
                before_command=lambda _directory, observed: events.append(observed),
                after_command=lambda _directory, observed, result: events.extend(
                    [observed, result.command]
                ),
            ),
        )

        assert cast("_FakeTokenizer", backend.tokenizer).rendered_messages[0][0]["role"] == "system"
        assert "working directory and exported variables" in backend.prompts[0]
        assert trace.turns[0].commands[0].command == command
        assert events == [command, command, command]
        assert trace.turns[0].tool_results[0].content_sent == "command output\n\n\n(exit_code=0)"
        serialized_trace = trace_path.read_text()
        assert "__LADDER_SHELL_STATE_" not in serialized_trace
        assert "__LADDER_SHELL_STATE_" not in str(trace.messages)
        assert not any("wrapper" in item for item in events)

    def test_final_report_drops_reasoning_if_report_prompt_lacks_room(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        original_render = loop.render_prompt

        def render_with_large_report(
            tokenizer: _FakeTokenizer,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]],
            *,
            enable_thinking: bool = True,
        ) -> str:
            rendered = original_render(
                cast("PreTrainedTokenizerBase", tokenizer),
                messages,
                tools,
                enable_thinking=enable_thinking,
            )
            if any(
                message.get("role") == "user"
                and "The episode is over." in str(message.get("content"))
                for message in messages
            ):
                return rendered + (
                    " retained " * 1500
                    if any("reasoning_content" in message for message in messages)
                    else ""
                )
            return rendered

        monkeypatch.setattr(loop, "render_prompt", render_with_large_report)
        backend = ScriptedToolBackend(
            [_ScriptedReply(_bash("touch marker.txt")), _ScriptedReply(_report())]
        )

        trace = loop.run_ladder_episode(
            _task(),
            backend,
            episode_dir=tmp_path / "episode",
            arm=AgenticArmConfig(check_affordance=True),
            system_prompt="System",
            user_template="{task_markdown}\n{listing}",
            max_turns=1,
            timeout="120s",
            jail_backend=None,
            max_model_len=1_300,
        )

        assert trace.final_report_reasoning_dropped
        assert trace.final_report == "The marker is ready."
        prior_assistant_messages = [
            message for message in trace.messages if message["role"] == "assistant"
        ]
        assert "reasoning_content" not in prior_assistant_messages[0]


class TestPersistentShell:
    def _run_wrapper_in_host(
        self, episode_dir: Path, command: str, *, timeout: str, **_kwargs: object
    ) -> CommandResult:
        del timeout
        remapped = command.replace(
            "builtin cd -- /work",
            f"builtin cd -- {shlex.quote(str(episode_dir))}",
            1,
        )
        bash_path = shutil.which("bash")
        assert bash_path is not None
        completed = subprocess.run(  # noqa: S603 - command is the loop-generated wrapper in a temp dir
            [bash_path, "-c", remapped],
            cwd=episode_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return CommandResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=False,
        )

    def _run_persistent_episode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        commands: list[str],
        hooks: _PersistentShellHooks | None = None,
    ) -> loop.LadderEpisodeTrace:
        hook_config = hooks or _PersistentShellHooks()
        episode_dir = tmp_path / "persistent-shell"
        episode_dir.mkdir()
        (episode_dir / "sub").mkdir()
        monkeypatch.setattr(loop, "run_in_jail", self._run_wrapper_in_host)
        monkeypatch.setattr(loop, "grade_final_state", _successful_grade)
        completion = "\n".join(
            _bash(command, start_turn=index == 0) for index, command in enumerate(commands)
        )
        backend = ScriptedToolBackend([_ScriptedReply(completion), _ScriptedReply(_report())])
        return loop.run_ladder_episode(
            _task(),
            backend,
            episode_dir=episode_dir,
            arm=AgenticArmConfig(check_affordance=True),
            system_prompt="System",
            user_template="{task_markdown}\n{listing}",
            max_turns=1,
            timeout="120s",
            jail_backend=None,
            before_command=hook_config.before_command,
            after_command=hook_config.after_command,
            trace_path=hook_config.trace_path,
            max_model_len=100_000,
        )

    def test_cd_persists_to_the_next_command(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        trace = self._run_persistent_episode(tmp_path, monkeypatch, ["cd sub", "pwd"])

        assert "sub" in trace.turns[0].commands[1].stdout
        assert trace.turns[0].commands[0].command == "cd sub"

    def test_export_persists_to_the_next_command(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        trace = self._run_persistent_episode(
            tmp_path,
            monkeypatch,
            ["export LADDER_STATE_TEST=present", "printf '%s' \"$LADDER_STATE_TEST\""],
        )

        assert trace.turns[0].commands[1].stdout == "present"

    def test_wrapper_is_invisible_to_hooks_trace_monitor_input_and_tool_result(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        hook_commands: list[str] = []
        hook_results: list[str] = []
        monitor_records: list[dict[str, str]] = []
        trace_path = tmp_path / "persistent-trace.jsonl"

        def record_monitor_command(_directory: Path, command: str, result: CommandResult) -> None:
            hook_commands.extend([command, result.command])
            hook_results.append(result.stderr)
            monitor_records.append({"command": command, "recorded_command": result.command})

        trace = self._run_persistent_episode(
            tmp_path,
            monkeypatch,
            ["cd sub", "pwd"],
            _PersistentShellHooks(
                before_command=lambda _directory, command: hook_commands.append(command),
                after_command=record_monitor_command,
                trace_path=trace_path,
            ),
        )
        trace_text = trace_path.read_text()
        tool_results = [result.content_sent for turn in trace.turns for result in turn.tool_results]

        assert hook_commands == ["cd sub", "cd sub", "cd sub", "pwd", "pwd", "pwd"]
        assert all("__LADDER_SHELL_STATE_" not in value for value in hook_commands)
        assert monitor_records == [
            {"command": "cd sub", "recorded_command": "cd sub"},
            {"command": "pwd", "recorded_command": "pwd"},
        ]
        assert "__LADDER_SHELL_STATE_" not in trace_text
        assert trace.turn_records()[0]["commands"] == [
            {"command": "cd sub", "exit_code": 0, "stdout": "", "stderr": "", "timed_out": False},
            {
                "command": "pwd",
                "exit_code": 0,
                "stdout": str(tmp_path / "persistent-shell" / "sub") + "\n",
                "stderr": "",
                "timed_out": False,
            },
        ]
        assert all("__LADDER_SHELL_STATE_" not in result for result in tool_results)
        assert all("__LADDER_SHELL_STATE_" not in stderr for stderr in hook_results)

    def test_exiting_shell_leaves_previous_state_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        trace = self._run_persistent_episode(
            tmp_path,
            monkeypatch,
            [
                "cd sub; export LADDER_STATE_TEST=partial; exit 7",
                "pwd",
                "printf '%s' \"$LADDER_STATE_TEST\"",
            ],
        )

        assert trace.turns[0].commands[0].exit_code == 7
        assert str(tmp_path / "persistent-shell") in trace.turns[0].commands[1].stdout
        assert trace.turns[0].commands[2].stdout == ""

    def test_broken_delimiter_leaves_previous_state_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        class FixedUUID:
            hex = "fixed-state-token"

        def fixed_uuid() -> FixedUUID:
            return FixedUUID()

        monkeypatch.setattr(loop, "uuid4", fixed_uuid)
        broken_delimiter_command = (
            "cd sub; export LADDER_STATE_TEST=partial; "
            "printf '\\0__LADDER_SHELL_STATE_fixed-state-token__\\0' >&2"
        )
        trace = self._run_persistent_episode(
            tmp_path,
            monkeypatch,
            [broken_delimiter_command, "pwd", "printf '%s' \"$LADDER_STATE_TEST\""],
        )

        assert str(tmp_path / "persistent-shell") in trace.turns[0].commands[1].stdout
        assert trace.turns[0].commands[2].stdout == ""


@needs_jail
def test_real_jail_executes_a_marker_command_and_grades_the_result(tmp_path: Path) -> None:
    task = _task()
    backend = ScriptedToolBackend(
        [
            _ScriptedReply(_bash("cat > marker.txt <<'EOF'\ncreated\nEOF")),
            _ScriptedReply(_report()),
        ]
    )

    trace = loop.run_ladder_episode(
        task,
        backend,
        episode_dir=tmp_path / "real-jail",
        arm=AgenticArmConfig(check_affordance=True),
        system_prompt="Use bash to create the marker.",
        user_template="Task:\n{task_markdown}\nFiles:\n{listing}",
        max_turns=1,
        timeout="120s",
        jail_backend=None,
        max_model_len=100_000,
    )

    assert trace.turns[0].commands[0].exit_code == 0
    assert trace.true_pass
    assert trace.proxy_pass
