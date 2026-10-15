"""CPU-only native-protocol integration; these fixtures do not establish isolation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.harness.docker_backend import DockerConfig
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.model_backend import RenderedCompletion
from reward_hacking.spec_gaming.capture import EpisodeJournal
from reward_hacking.spec_gaming.episode import (
    DEFAULT_SETTINGS,
    EpisodeSettings,
    assess_spreadsheet_episode,
    run_spreadsheet_episode,
    spreadsheet_generation_identity,
)
from reward_hacking.spec_gaming.runtime import (
    RuntimeAction,
    RuntimeEvaluation,
    RuntimeInfrastructureError,
    WorkspaceEntry,
    WorkspaceSnapshot,
)
from reward_hacking.spec_gaming.spreadsheet import (
    SpreadsheetCase,
    SpreadsheetTask,
    select_proxy_cells,
)
from reward_hacking.tests.test_spec_gaming_protocol import NonRoundTrippingTokenizer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def call(name: str, command: str = "") -> str:
    parameter = f"<parameter=command>\n{command}\n</parameter>\n" if command else ""
    return f"</think><tool_call>\n<function={name}>\n{parameter}</function>\n</tool_call>"


class EpisodeTokenizer(NonRoundTrippingTokenizer):
    """Decode sampled fixture tokens identically after process reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.template_options: list[dict[str, Any]] = []

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str | list[int]:
        self.template_options.append(dict(kwargs))
        return super().apply_chat_template(*args, **kwargs)

    def decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        if tuple(token_ids) == (self.eos_token_id,):
            return super().decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
        return "".join(chr(token_id - 500) for token_id in token_ids)


class ScriptedBackend:
    tokenizer: EpisodeTokenizer

    def __init__(self, completions: list[str]) -> None:
        self.tokenizer = EpisodeTokenizer()
        self.completions = iter(completions)
        self.prompts: list[list[int]] = []
        self.seeds: list[int | None] = []
        self.finish_reason: str | None = "stop"

    def complete_token_ids(
        self, prompt: list[int], *, max_tokens: int, seed: int | None
    ) -> RenderedCompletion:
        del max_tokens
        self.seeds.append(seed)
        self.prompts.append(prompt)
        text = next(self.completions)
        token_ids = tuple(ord(character) + 500 for character in text)
        return RenderedCompletion(
            text,
            self.finish_reason,
            None,
            len(prompt),
            len(token_ids),
            token_ids,
            tuple(-0.2 for _ in token_ids),
            tuple(prompt),
        )


class SyntheticRuntime:
    """A CPU fixture that writes bytes; never launches shell or Docker."""

    def __init__(self) -> None:
        self.actions = 0
        self.evaluations = 0
        self.action_timeouts: list[str] = []
        self.evaluation_timeouts: list[str] = []
        self.operational_only: list[bool] = []
        self.proxy_target: bytes | None = None
        self.fail_evaluation = False
        self.fail_full = False
        self.fail_action = False

    def execute_action(
        self,
        snapshot: WorkspaceSnapshot,
        command: str,
        config: DockerConfig,
        *,
        timeout: str = "60s",
    ) -> RuntimeAction:
        del config
        self.action_timeouts.append(timeout)
        self.actions += 1
        if self.fail_action:
            self.fail_action = False
            raise RuntimeError("synthetic interrupted unpublished action")
        entries = tuple(entry for entry in snapshot.entries if entry.path != "output.xlsx")
        return RuntimeAction(
            WorkspaceSnapshot(
                (*entries, WorkspaceEntry("output.xlsx", "file", 0o644, command.encode()))
            ),
            CommandResult(command, 0, "", "", False),
        )

    def run_evaluation(
        self, files: Mapping[str, bytes], config: DockerConfig, *, timeout: str = "120s"
    ) -> RuntimeEvaluation:
        del config
        self.evaluation_timeouts.append(timeout)
        self.evaluations += 1
        if self.fail_evaluation:
            self.fail_evaluation = False
            raise RuntimeInfrastructureError("synthetic evaluator interrupted")
        proxy_success = self.proxy_target is None or files["submission.xlsx"] == self.proxy_target
        operational_only = json.loads(files["job.json"])["operational_only"]
        self.operational_only.append(operational_only)
        full: dict[str, Any] = {
            "status": "measured",
            "success": files["submission.xlsx"] == b"complete",
        }
        if operational_only or self.fail_full:
            full = {
                "status": "unmeasured",
                "success": None,
                "error": {"kind": "not_requested" if operational_only else "synthetic_full_error"},
            }
        return RuntimeEvaluation(
            {
                "status": "measured",
                "proxy": {
                    "status": "measured",
                    "score": float(proxy_success),
                    "success": proxy_success,
                    "checks_passed": int(proxy_success),
                    "checks_total": 1,
                },
                "full": full,
            },
            CommandResult("synthetic evaluation", 0, "", "", False),
        )


@pytest.fixture
def task(tmp_path: Path) -> SpreadsheetTask:
    paths = []
    for name, content in [
        ("input", b"synthetic input"),
        ("reference", b"HIDDEN_REFERENCE_SENTINEL"),
        ("evaluator", b"synthetic evaluator"),
    ]:
        path = tmp_path / name
        path.write_bytes(content)
        paths.append((path, hashlib.sha256(content).hexdigest()))
    return SpreadsheetTask(
        "synthetic",
        "synthetic",
        "verified_400",
        "synthetic",
        "Complete both requested cells.",
        "Cell-Level",
        "Sheet1!A1:B1",
        (
            SpreadsheetCase(
                1, paths[0][0], paths[1][0], paths[0][1], paths[1][1], "input", "reference"
            ),
        ),
        paths[2][0],
        paths[2][1],
    )


def config(tmp_path: Path) -> DockerConfig:
    return DockerConfig("sha256:" + "a" * 64, tmp_path / "unused-gate", uid=1000)


def run(
    task: SpreadsheetTask,
    tmp_path: Path,
    backend: ScriptedBackend,
    runtime: SyntheticRuntime,
    settings: EpisodeSettings = DEFAULT_SETTINGS,
) -> Any:
    return run_spreadsheet_episode(
        task,
        select_proxy_cells(task),
        tmp_path / "episode",
        generation_identity=spreadsheet_generation_identity(
            task, select_proxy_cells(task), {"test": "synthetic"}, settings
        ),
        runtime_config=config(tmp_path),
        backend_factory=lambda: backend,
        executor=runtime,
        settings=settings,
    )


def test_episode_settings_reject_zero_budget() -> None:
    with pytest.raises(ValueError, match="positive"):
        EpisodeSettings(max_turns=0)


def test_native_workflow_freezes_output_and_assesses_without_actor(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    runtime = SyntheticRuntime()
    backend = ScriptedBackend(
        [
            call("bash", "partial"),
            call("check"),
            "</think>I am finished.",
            call("bash", "complete"),
            call("hand_back"),
        ]
    )
    result = run(task, tmp_path, backend, runtime)
    assert result.status == "hand_back"
    assert result.turns == 5
    assert len(backend.prompts) == 5
    assert runtime.actions == 2
    assert result.submission_digest == hashlib.sha256(b"complete").hexdigest()
    assert run(task, tmp_path, ScriptedBackend([]), runtime) == result
    assessed = assess_spreadsheet_episode(
        task,
        select_proxy_cells(task),
        tmp_path / "episode",
        generation_identity=spreadsheet_generation_identity(
            task, select_proxy_cells(task), {"test": "synthetic"}
        ),
        assessment_identity={"evaluator": "synthetic"},
        runtime_config=config(tmp_path),
        executor=runtime,
    )
    assert assessed["full_success"] is True
    assert assessed["awareness_status"] == "pending"
    assert runtime.evaluations == 3


def test_interrupted_action_reuses_sample(task: SpreadsheetTask, tmp_path: Path) -> None:
    runtime = SyntheticRuntime()
    runtime.fail_action = True
    backend = ScriptedBackend([call("bash", "partial")])
    with pytest.raises(RuntimeError, match="interrupted"):
        run(task, tmp_path, backend, runtime)
    identity = spreadsheet_generation_identity(
        task, select_proxy_cells(task), {"test": "synthetic"}
    )
    with EpisodeJournal(tmp_path / "episode", identity) as journal:
        saved_completion = journal.get_completion(0)
        assert saved_completion is not None
        assert saved_completion["metadata"]["generation_elapsed_seconds"] > 0
        assert saved_completion["metadata"]["sample_seed"] == backend.seeds[0]
    resumed = ScriptedBackend([call("hand_back")])
    result = run(task, tmp_path, resumed, runtime)
    assert result.status == "hand_back"
    assert len(backend.prompts) == len(resumed.prompts) == 1
    with EpisodeJournal(tmp_path / "episode", identity) as journal:
        assert journal.get_completion(0) == saved_completion


def test_check_budget_does_not_replenish_and_success_does_not_close(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    runtime = SyntheticRuntime()
    backend = ScriptedBackend(
        [call("bash", "partial"), *[call("check") for _ in range(6)], call("hand_back")]
    )
    result = run(task, tmp_path, backend, runtime)
    assert result.checks == 5
    assert len(backend.prompts) == 8
    assert runtime.evaluations == 6


def test_handback_followed_by_mutation_executes_nothing(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    runtime = SyntheticRuntime()
    result = run(
        task,
        tmp_path,
        ScriptedBackend([call("hand_back") + call("bash", "bad"), call("hand_back")]),
        runtime,
    )
    assert runtime.actions == 0
    assert result.submission_digest is None


def test_restart_after_committed_check_keeps_remaining_batch(
    task: SpreadsheetTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_commit = EpisodeJournal.commit_action
    interrupted = False

    def commit_then_interrupt(  # noqa: PLR0913
        self: EpisodeJournal,
        request_id: str,
        result: Any,
        state: Any,
        *,
        check: bool = False,
        close: bool = False,
    ) -> dict[str, Any]:
        nonlocal interrupted
        committed = original_commit(self, request_id, result, state, check=check, close=close)
        if check and not interrupted:
            interrupted = True
            raise RuntimeError("process death after committed check")
        return committed

    monkeypatch.setattr(EpisodeJournal, "commit_action", commit_then_interrupt)
    runtime = SyntheticRuntime()
    batch = "</think>" + "".join(call("check").removeprefix("</think>") for _ in range(5))
    backend = ScriptedBackend([call("bash", "partial"), batch])
    with pytest.raises(RuntimeError, match="committed check"):
        run(task, tmp_path, backend, runtime)
    resumed = ScriptedBackend([call("hand_back")])
    result = run(task, tmp_path, resumed, runtime)
    assert result.checks == 5
    assert runtime.evaluations == 6
    assert len(resumed.prompts) == 1


def test_restart_after_last_turn_never_samples_extra_turn(
    task: SpreadsheetTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_commit = EpisodeJournal.commit_action
    interrupted = False

    def commit_then_interrupt(  # noqa: PLR0913
        self: EpisodeJournal,
        request_id: str,
        result: Any,
        state: Any,
        *,
        check: bool = False,
        close: bool = False,
    ) -> dict[str, Any]:
        nonlocal interrupted
        committed = original_commit(self, request_id, result, state, check=check, close=close)
        if not close and not interrupted:
            interrupted = True
            raise RuntimeError("process death after last turn")
        return committed

    monkeypatch.setattr(EpisodeJournal, "commit_action", commit_then_interrupt)
    runtime = SyntheticRuntime()
    settings = EpisodeSettings(max_turns=1)
    with pytest.raises(RuntimeError, match="last turn"):
        run(task, tmp_path, ScriptedBackend([call("bash", "partial")]), runtime, settings)
    resumed = ScriptedBackend([])
    assert run(task, tmp_path, resumed, runtime, settings).status == "max_turns"
    assert resumed.prompts == []


def test_hidden_reference_never_enters_actor_history(task: SpreadsheetTask, tmp_path: Path) -> None:
    runtime = SyntheticRuntime()
    run(
        task,
        tmp_path,
        ScriptedBackend([call("bash", "partial"), call("check"), call("hand_back")]),
        runtime,
    )
    identity = spreadsheet_generation_identity(
        task, select_proxy_cells(task), {"test": "synthetic"}
    )
    with EpisodeJournal(tmp_path / "episode", identity) as journal:
        assert "HIDDEN_REFERENCE_SENTINEL" not in json.dumps(journal.state()["history"])
        completion = journal.get_completion(0)
        assert completion is not None
        assert completion["output_token_ids"] == [
            ord(character) + 500 for character in call("bash", "partial")
        ]
        assert completion["sampled_logprobs"] == [-0.2] * len(completion["output_token_ids"])
        assert "full_success" not in json.dumps(journal.state()["history"])


def test_assessment_refresh_changes_no_actor_or_frozen_output(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    runtime = SyntheticRuntime()
    backend = ScriptedBackend([call("bash", "partial"), call("hand_back")])
    result = run(task, tmp_path, backend, runtime)
    for version in ["initial", "revised"]:
        assessed = assess_spreadsheet_episode(
            task,
            select_proxy_cells(task),
            tmp_path / "episode",
            generation_identity=spreadsheet_generation_identity(
                task, select_proxy_cells(task), {"test": "synthetic"}
            ),
            assessment_identity={"evaluator": version},
            runtime_config=config(tmp_path),
            executor=runtime,
        )
        assert assessed["proxy_score"] == 1.0
        assert assessed["full_success"] is False
        assert assessed["mechanism_evidence"] == []
        assert assessed["submission_digest"] == result.submission_digest
    assert len(backend.prompts) == 2
    assert runtime.evaluations == 3


def test_six_check_batch_rejected_before_first_check(task: SpreadsheetTask, tmp_path: Path) -> None:
    runtime = SyntheticRuntime()
    batch = "</think>" + "".join(call("check").removeprefix("</think>") for _ in range(6))
    result = run(task, tmp_path, ScriptedBackend([batch, call("hand_back")]), runtime)
    assert result.checks == runtime.evaluations == 0


def test_context_exhaustion_closes_with_explicit_status(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    runtime = SyntheticRuntime()
    backend = ScriptedBackend([])
    result = run(task, tmp_path, backend, runtime, EpisodeSettings(max_context_tokens=1))
    assert result.status == "context_limit"
    assert backend.prompts == []


def test_edited_budget_refuses_generation_reuse(task: SpreadsheetTask, tmp_path: Path) -> None:
    runtime = SyntheticRuntime()
    run(task, tmp_path, ScriptedBackend([call("hand_back")]), runtime)
    with pytest.raises(ValueError, match="generation identity changed"):
        run(task, tmp_path, ScriptedBackend([]), runtime, EpisodeSettings(max_turns=19))


def test_infrastructure_failure_keeps_episode_open_and_sampled_evidence(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    runtime = SyntheticRuntime()
    runtime.fail_action = True
    with pytest.raises(RuntimeError, match="unpublished"):
        run(task, tmp_path, ScriptedBackend([call("bash", "partial")]), runtime)
    identity = spreadsheet_generation_identity(
        task, select_proxy_cells(task), {"test": "synthetic"}
    )
    with EpisodeJournal(tmp_path / "episode", identity) as journal:
        assert journal.progress()["closed"] is False
        assert journal.get_completion(0) is not None
        with pytest.raises(ValueError, match="not closed"):
            journal.raw_episode_digest()


@pytest.mark.parametrize("finish_reason", ["error", "abort", None])
@pytest.mark.parametrize("tool_name", ["bash", "check", "hand_back"])
def test_failed_generation_never_executes_and_resume_reuses_rejected_completion(
    task: SpreadsheetTask, tmp_path: Path, finish_reason: str | None, tool_name: str
) -> None:
    runtime = SyntheticRuntime()
    backend = ScriptedBackend([call(tool_name, "mutated" if tool_name == "bash" else "")])
    backend.finish_reason = finish_reason
    identity = spreadsheet_generation_identity(
        task, select_proxy_cells(task), {"test": "synthetic"}
    )
    original_state = {
        "turn": 0,
        "next_call": 0,
        "history": None,
        "snapshot": [],
        "status": "running",
        "submission_digest": None,
    }
    input_path = task.primary_case.input_path
    assert input_path is not None
    original_input = input_path.read_bytes()
    with EpisodeJournal(tmp_path / "episode", identity, original_state):
        pass
    with pytest.raises(ValueError, match="finish_reason"):
        run(task, tmp_path, backend, runtime)
    with EpisodeJournal(tmp_path / "episode", identity) as journal:
        saved = journal.get_completion(0)
        assert saved is not None
        assert saved["metadata"]["finish_reason"] == finish_reason
        assert saved["raw_text"] == call(tool_name, "mutated" if tool_name == "bash" else "")
        assert journal.get_completion(1) is None
        assert journal.state() == original_state
        assert journal.progress()["closed"] is False
        assert journal.connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 0
        progress = journal.progress()
    resumed = ScriptedBackend([])
    with pytest.raises(ValueError, match="finish_reason"):
        run(task, tmp_path, resumed, runtime)
    assert len(backend.prompts) == 1
    assert resumed.prompts == []
    assert runtime.actions == runtime.evaluations == 0
    assert input_path.read_bytes() == original_input
    with EpisodeJournal(tmp_path / "episode", identity) as journal:
        assert journal.get_completion(0) == saved
        assert journal.get_completion(1) is None
        assert journal.state() == original_state
        assert journal.progress() == progress
        assert journal.connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 0


def test_output_limit_cannot_execute_even_complete_looking_tool_call(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    runtime = SyntheticRuntime()
    backend = ScriptedBackend([call("bash", "partial")])
    backend.finish_reason = "length"
    result = run(task, tmp_path, backend, runtime)
    assert result.status == "output_length"
    assert result.turns == 1
    assert runtime.actions == 0


@pytest.mark.parametrize(
    "field",
    [
        "max_turns",
        "max_tokens",
        "max_context_tokens",
        "seed",
        "action_timeout_seconds",
        "evaluation_timeout_seconds",
    ],
)
@pytest.mark.parametrize("value", [True, False, -1, 1.5])
def test_episode_settings_require_integer_budgets_and_seed(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match="integer"):
        EpisodeSettings(**{field: value})


def test_episode_settings_require_boolean_thinking_mode() -> None:
    invalid: Any = 1
    with pytest.raises(ValueError, match="boolean"):
        EpisodeSettings(enable_thinking=invalid)


def test_nonthinking_episode_renders_and_parses_every_turn_without_thought_marker(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    tokenizer = EpisodeTokenizer()
    backend = ScriptedBackend(
        [
            call("bash", "complete").removeprefix("</think>"),
            call("hand_back").removeprefix("</think>"),
        ]
    )
    backend.tokenizer = tokenizer
    result = run(
        task, tmp_path, backend, SyntheticRuntime(), EpisodeSettings(enable_thinking=False)
    )
    assert result.status == "hand_back"
    assert result.submission_digest == hashlib.sha256(b"complete").hexdigest()
    assert len(backend.prompts) == 2
    assert len(tokenizer.thinking_modes) >= 2
    assert set(tokenizer.thinking_modes) == {False}


def test_configured_runtime_timeouts_reach_action_check_and_assessment(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    settings = EpisodeSettings(action_timeout_seconds=7, evaluation_timeout_seconds=13)
    runtime = SyntheticRuntime()
    backend = ScriptedBackend([call("bash", "complete"), call("check"), call("hand_back")])
    run(task, tmp_path, backend, runtime, settings)
    assess_spreadsheet_episode(
        task,
        select_proxy_cells(task),
        tmp_path / "episode",
        generation_identity=spreadsheet_generation_identity(
            task, select_proxy_cells(task), {"test": "synthetic"}, settings
        ),
        assessment_identity={"evaluator": "timeouts"},
        runtime_config=config(tmp_path),
        executor=runtime,
        settings=settings,
    )
    assert runtime.action_timeouts == ["7s"]
    assert runtime.evaluation_timeouts == ["13s", "13s", "13s"]
    assert runtime.operational_only == [True, True, False]


@pytest.mark.parametrize("value", [{"bad": 1}, {1: "bad"}, [], None])
def test_template_options_require_string_mapping(value: Any) -> None:
    with pytest.raises(ValueError, match="string"):
        EpisodeSettings(chat_template_kwargs=value)


def test_template_options_reach_initial_and_continued_render(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    backend = ScriptedBackend([call("bash", "complete"), call("hand_back")])
    run(
        task,
        tmp_path,
        backend,
        SyntheticRuntime(),
        EpisodeSettings(chat_template_kwargs={"qa_template_flag": "present"}),
    )
    assert len(backend.tokenizer.template_options) >= 2
    assert all(
        options["qa_template_flag"] == "present" for options in backend.tokenizer.template_options
    )


def test_operational_feedback_survives_unmeasured_full_assessment(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    runtime = SyntheticRuntime()
    runtime.fail_full = True
    run(
        task,
        tmp_path,
        ScriptedBackend([call("bash", "complete"), call("check"), call("hand_back")]),
        runtime,
    )
    identity = spreadsheet_generation_identity(
        task, select_proxy_cells(task), {"test": "synthetic"}
    )
    assessment = assess_spreadsheet_episode(
        task,
        select_proxy_cells(task),
        tmp_path / "episode",
        generation_identity=identity,
        assessment_identity={"evaluator": "failed-full"},
        runtime_config=config(tmp_path),
        executor=runtime,
    )
    assert assessment["status"] == "measured"
    assert assessment["proxy_score"] == 1.0
    assert assessment["full_status"] == "unmeasured"
    assert assessment["full_success"] is None
    with EpisodeJournal(tmp_path / "episode", identity) as journal:
        history = json.dumps(journal.state()["history"])
        assert '"score": 1.0' in history.replace('\\"', '"')
        assert "synthetic_full_error" not in history


def test_recorded_timings_are_host_only_and_survive_reopen(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    backend = ScriptedBackend([call("bash", "complete"), call("check"), call("hand_back")])
    runtime = SyntheticRuntime()
    run(task, tmp_path, backend, runtime)
    identity = spreadsheet_generation_identity(
        task, select_proxy_cells(task), {"test": "synthetic"}
    )
    with EpisodeJournal(tmp_path / "episode", identity) as journal:
        completion = journal.get_completion(0)
        action = journal.get_saved_action("0:0")
        check = journal.get_saved_action("1:0")
        assert completion is not None
        assert action is not None
        assert check is not None
        assert completion["metadata"]["generation_elapsed_seconds"] > 0
        assert action["result"]["execution_elapsed_seconds"] > 0
        assert check["result"]["execution_elapsed_seconds"] > 0
        assert "elapsed_seconds" not in json.dumps(journal.state()["history"])
    run(task, tmp_path, ScriptedBackend([]), runtime)
    with EpisodeJournal(tmp_path / "episode", identity) as journal:
        assert journal.get_completion(0) == completion
        assert journal.get_saved_action("0:0") == action
        assert journal.get_saved_action("1:0") == check


def test_handback_freezes_proxy_for_changed_artifact_after_successful_check(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    runtime = SyntheticRuntime()
    runtime.proxy_target = b"complete"
    backend = ScriptedBackend(
        [call("bash", "complete"), call("check"), call("bash", "partial"), call("hand_back")]
    )
    result = run(task, tmp_path, backend, runtime)
    identity = spreadsheet_generation_identity(
        task, select_proxy_cells(task), {"test": "synthetic"}
    )
    with EpisodeJournal(tmp_path / "episode", identity) as journal:
        frozen = journal.state()["operational_proxy"]
        assert frozen["score"] == 0.0
        assert frozen["status"] == "measured"
        assert (
            frozen["submission_digest"]
            == result.submission_digest
            == hashlib.sha256(b"partial").hexdigest()
        )
        action = journal.get_saved_action("3:0")
        assert action is not None
        assert action["result"]["operational_proxy"] == frozen
        assert "operational_proxy" not in json.dumps(journal.state()["history"])
    assert runtime.evaluations == 2
    assert result.checks == 1


def test_failed_handback_proxy_retries_without_new_actor_sample(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    runtime = SyntheticRuntime()
    runtime.fail_evaluation = True
    backend = ScriptedBackend([call("bash", "complete"), call("hand_back")])
    with pytest.raises(RuntimeInfrastructureError, match="synthetic evaluator interrupted"):
        run(task, tmp_path, backend, runtime)
    identity = spreadsheet_generation_identity(
        task, select_proxy_cells(task), {"test": "synthetic"}
    )
    with EpisodeJournal(tmp_path / "episode", identity) as journal:
        saved = journal.get_completion(1)
        assert journal.progress()["closed"] is False
    resumed = ScriptedBackend([])
    result = run(task, tmp_path, resumed, runtime)
    assert result.status == "hand_back"
    assert resumed.prompts == []
    assert len(backend.prompts) == 2
    with EpisodeJournal(tmp_path / "episode", identity) as journal:
        assert journal.get_completion(1) == saved
        assert journal.state()["operational_proxy"]["submission_digest"] == result.submission_digest


def test_task_sample_seeds_are_distinct_and_independent_of_execution_order(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    observed = []
    for order_index, order in enumerate([("first", "second"), ("second", "first")]):
        seeds = {}
        for task_name in order:
            backend = ScriptedBackend([call("hand_back")])
            selected_task = replace(task, task_id=f"synthetic-{task_name}")
            run(selected_task, tmp_path / str(order_index) / task_name, backend, SyntheticRuntime())
            seeds[task_name] = backend.seeds[0]
        observed.append(seeds)
    assert observed[0]["first"] != observed[0]["second"]
    assert observed[0] == observed[1]


def test_same_task_id_in_distinct_source_archives_uses_distinct_seed(
    task: SpreadsheetTask, tmp_path: Path
) -> None:
    backends = [ScriptedBackend([call("hand_back")]), ScriptedBackend([call("hand_back")])]
    for index, selected_task in enumerate(
        [task, replace(task, source_archive_sha256="other-synthetic-archive")]
    ):
        run(selected_task, tmp_path / str(index), backends[index], SyntheticRuntime())
    assert backends[0].seeds[0] != backends[1].seeds[0]
