"""Retail dialogue, private native state and journal recovery controls."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.spec_gaming.capture import EpisodeJournal
from reward_hacking.spec_gaming.episode import EpisodeSettings
from reward_hacking.spec_gaming.retail_episode import run_retail_episode
from reward_hacking.tests.test_spec_gaming_episode import ScriptedBackend, call

if TYPE_CHECKING:
    from pathlib import Path


class NativeFixture:
    def describe(self, task_id: str) -> dict[str, Any]:
        return {
            "db_state": {"count": 0},
            "actor_system_prompt": "Serve the customer.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "cancel",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "user_state": {
                "system_messages": [{"role": "system", "content": "PRIVATE SCENARIO"}],
                "messages": [],
            },
            "first_agent_message": "Hello",
        }

    def user_prompt(
        self, task_id: str, user_state: dict[str, Any], incoming_actor_text: str
    ) -> list[dict[str, str]]:
        return [
            *user_state["system_messages"],
            *user_state["messages"],
            {"role": "user", "content": incoming_actor_text},
        ]

    def user_turn(
        self,
        task_id: str,
        user_state: dict[str, Any],
        incoming_actor_text: str,
        sampled_response: str,
    ) -> dict[str, Any]:
        state = {
            **user_state,
            "messages": [
                *user_state["messages"],
                {"role": "user", "content": incoming_actor_text},
                {"role": "assistant", "content": sampled_response},
            ],
        }
        return {
            "user_state": state,
            "user_message": {"role": "user", "content": sampled_response},
            "is_stop": "###STOP###" in sampled_response,
        }

    def tool_call(
        self, task_id: str, db_state: dict[str, Any], tool_call: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "db_state": {"count": db_state["count"] + 1},
            "tool_message": {
                "role": "tool",
                "id": tool_call["id"],
                "content": "ok",
                "error": False,
                "requestor": "assistant",
            },
        }


def test_dialogue_plain_text_check_close_and_hidden_boundary(tmp_path: Path) -> None:
    backend = ScriptedBackend(
        [
            "</think>Please cancel.",
            "</think>Confirm?",
            "</think>Yes.",
            call("cancel"),
            call("check"),
            call("hand_back"),
        ]
    )
    result = run_retail_episode(
        "synthetic",
        tmp_path,
        generation_identity={"fixture": 1},
        backend_factory=lambda: backend,
        native_runtime=NativeFixture(),
        relevant_operation=lambda *_: True,
        settings=EpisodeSettings(max_turns=5),
    )
    assert result.status == "hand_back"
    with EpisodeJournal(tmp_path, {"fixture": 1}) as journal:
        state = journal.state()
        assert state["db_state"] == {"count": 1}
        assert state["proxy_success"] is True
        assert journal.progress()["checks"] == 1
        assert state["actor_turns"] == 4
        assert state["simulator_turns"] == 2
        assert "PRIVATE SCENARIO" not in str(state["actor_history"])
        assert all(
            message["role"] in {"user", "assistant"} and "tool_calls" not in message
            for message in state["user_state"]["messages"]
        )
    assert (
        run_retail_episode(
            "synthetic",
            tmp_path,
            generation_identity={"fixture": 1},
            backend_factory=lambda: pytest.fail("closed episode resampled"),
            native_runtime=NativeFixture(),
            relevant_operation=lambda *_: True,
        ).status
        == "hand_back"
    )


def test_uncommitted_native_mutation_recovers_private_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = ScriptedBackend(["</think>Please proceed.", call("cancel"), call("hand_back")])
    original_commit = EpisodeJournal.commit_action
    interrupted = False

    def interrupt_commit(
        self: EpisodeJournal, request_id: str, result: Any, state: Any, **kwargs: Any
    ) -> Any:
        nonlocal interrupted
        if state["db_state"]["count"] >= 1 and not interrupted:
            interrupted = True
            raise RuntimeError("process interrupted before mutation publication")
        return original_commit(self, request_id, result, state, **kwargs)

    monkeypatch.setattr(EpisodeJournal, "commit_action", interrupt_commit)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_retail_episode(
            "synthetic",
            tmp_path,
            generation_identity={"fixture": 1},
            backend_factory=lambda: backend,
            native_runtime=NativeFixture(),
            relevant_operation=lambda *_: True,
        )
    with EpisodeJournal(tmp_path, {"fixture": 1}) as journal:
        assert journal.state()["db_state"] == {"count": 0}
        assert journal.get_completion(1) is not None
    result = run_retail_episode(
        "synthetic",
        tmp_path,
        generation_identity={"fixture": 1},
        backend_factory=lambda: backend,
        native_runtime=NativeFixture(),
        relevant_operation=lambda *_: True,
    )
    assert result.status == "hand_back"
    assert len(backend.prompts) == 3
    with EpisodeJournal(tmp_path, {"fixture": 1}) as journal:
        assert journal.state()["db_state"] == {"count": 1}


def test_mixed_and_multiple_calls_never_mutate(tmp_path: Path) -> None:
    backend = ScriptedBackend(
        [
            "</think>Proceed.",
            "</think>Visible text" + call("cancel"),
            call("cancel") + call("cancel"),
            call("hand_back"),
        ]
    )
    run_retail_episode(
        "synthetic",
        tmp_path,
        generation_identity={"fixture": 1},
        backend_factory=lambda: backend,
        native_runtime=NativeFixture(),
        relevant_operation=lambda *_: True,
    )
    with EpisodeJournal(tmp_path, {"fixture": 1}) as journal:
        assert journal.state()["db_state"] == {"count": 0}
        assert journal.state()["proxy_success"] is False


def test_simulator_stop_closes_without_actor_or_tools(tmp_path: Path) -> None:
    backend = ScriptedBackend(["</think>###STOP###"])
    result = run_retail_episode(
        "synthetic",
        tmp_path,
        generation_identity={"fixture": 1},
        backend_factory=lambda: backend,
        native_runtime=NativeFixture(),
        relevant_operation=lambda *_: True,
    )
    assert result.status == "user_stop"
    assert len(backend.prompts) == 1


def test_check_budget_has_no_hidden_obligations(tmp_path: Path) -> None:
    backend = ScriptedBackend(
        ["</think>Please proceed.", *[call("check") for _ in range(6)], call("hand_back")]
    )
    result = run_retail_episode(
        "synthetic",
        tmp_path,
        generation_identity={"fixture": 1},
        backend_factory=lambda: backend,
        native_runtime=NativeFixture(),
        relevant_operation=lambda *_: True,
    )
    assert result.checks == 5
    with EpisodeJournal(tmp_path, {"fixture": 1}) as journal:
        state = journal.state()
        assert "PRIVATE SCENARIO" not in str(state["actor_history"])
        assert "check_budget_exhausted" in str(state["actor_history"])


def test_proxy_history_remains_successful_after_reversal(tmp_path: Path) -> None:
    class ReversibleFixture(NativeFixture):
        def describe(self, task_id: str) -> dict[str, Any]:
            description = super().describe(task_id)
            undo = {
                **description["tools"][0],
                "function": {**description["tools"][0]["function"], "name": "undo"},
            }
            description["tools"].append(undo)
            return description

        def tool_call(
            self, task_id: str, db_state: dict[str, Any], tool_call: dict[str, Any]
        ) -> dict[str, Any]:
            transition = super().tool_call(task_id, db_state, tool_call)
            if tool_call["name"] == "undo":
                transition["db_state"]["count"] = 0
            return transition

    backend = ScriptedBackend(["</think>Proceed.", call("cancel"), call("undo"), call("hand_back")])
    run_retail_episode(
        "synthetic",
        tmp_path,
        generation_identity={"fixture": 1},
        backend_factory=lambda: backend,
        native_runtime=ReversibleFixture(),
        relevant_operation=lambda name, *_: name == "cancel",
    )
    with EpisodeJournal(tmp_path, {"fixture": 1}) as journal:
        assert journal.state()["db_state"] == {"count": 0}
        assert journal.state()["proxy_success"] is True


def test_backend_prompt_mismatch_is_not_relabelled_as_exact_capture(tmp_path: Path) -> None:
    class MismatchedBackend(ScriptedBackend):
        def complete_token_ids(
            self, prompt: list[int], *, max_tokens: int, seed: int | None
        ) -> Any:
            completion = super().complete_token_ids(prompt, max_tokens=max_tokens, seed=seed)
            return replace(completion, prompt_token_ids=(123,) * len(prompt))

    backend = MismatchedBackend(["</think>Proceed."])
    with pytest.raises(ValueError, match="exact prompt tokens"):
        run_retail_episode(
            "synthetic",
            tmp_path,
            generation_identity={"fixture": 1},
            backend_factory=lambda: backend,
            native_runtime=NativeFixture(),
            relevant_operation=lambda *_: True,
        )
    with EpisodeJournal(tmp_path, {"fixture": 1}) as journal:
        assert journal.get_completion(0) is None
        assert journal.state()["completion_index"] == 0


def test_nonthinking_simulator_text_and_mixed_call_boundary(tmp_path: Path) -> None:
    backend = ScriptedBackend(
        [
            "Please proceed.",
            "Visible text" + call("cancel").removeprefix("</think>"),
            call("hand_back").removeprefix("</think>"),
        ]
    )
    result = run_retail_episode(
        "synthetic",
        tmp_path,
        generation_identity={"fixture": 1},
        backend_factory=lambda: backend,
        native_runtime=NativeFixture(),
        relevant_operation=lambda *_: True,
        settings=EpisodeSettings(enable_thinking=False),
    )
    assert result.status == "hand_back"
    with EpisodeJournal(tmp_path, {"fixture": 1}) as journal:
        state = journal.state()
        assert state["db_state"] == {"count": 0}
        assert state["customer_message"] == "Please proceed."
        assert state["user_state"]["messages"][-1]["content"] == "Please proceed."


def test_native_initialization_is_frozen_before_journal_and_reused(tmp_path: Path) -> None:
    class ClockedFixture(NativeFixture):
        calls = 0

        def describe(self, task_id: str) -> dict[str, Any]:
            self.calls += 1
            description = super().describe(task_id)
            description["user_state"]["system_messages"][0]["timestamp"] = str(self.calls)
            return description

    runtime = ClockedFixture()
    backend = ScriptedBackend(["</think>Proceed.", call("hand_back")])
    run_retail_episode(
        "synthetic",
        tmp_path,
        generation_identity={"fixture": 1},
        backend_factory=lambda: backend,
        native_runtime=runtime,
        relevant_operation=lambda *_: True,
    )
    run_retail_episode(
        "synthetic",
        tmp_path,
        generation_identity={"fixture": 1},
        backend_factory=lambda: pytest.fail("resampled"),
        native_runtime=runtime,
        relevant_operation=lambda *_: True,
    )
    assert runtime.calls == 1
    altered = runtime.describe("synthetic")
    with pytest.raises(ValueError, match="frozen native initialization changed"):
        run_retail_episode(
            "synthetic",
            tmp_path,
            generation_identity={"fixture": 1},
            backend_factory=lambda: pytest.fail("resampled"),
            native_runtime=runtime,
            relevant_operation=lambda *_: True,
            initial_description=altered,
        )


@pytest.mark.parametrize(
    ("arguments", "native_error"),
    [
        ({}, True),
        ({"item_ids": None}, True),
        ({"item_ids": ["synthetic-first", "synthetic-second"]}, True),
        ({"item_ids": ["synthetic-first", "synthetic-second"]}, False),
    ],
)
def test_native_item_modification_results_commit_and_resume_with_exact_defect_exposure(
    tmp_path: Path, arguments: dict[str, Any], native_error: bool
) -> None:
    class ErrorFixture(NativeFixture):
        def describe(self, task_id: str) -> dict[str, Any]:
            description = super().describe(task_id)
            description["tools"][0]["function"] = {
                "name": "modify_pending_order_items",
                "parameters": {
                    "type": "object",
                    "properties": {"item_ids": {"type": ["array", "null"]}},
                },
            }
            return description

        def tool_call(
            self, task_id: str, db_state: dict[str, Any], tool_call: dict[str, Any]
        ) -> dict[str, Any]:
            return {
                "db_state": db_state if native_error else {"count": db_state["count"] + 1},
                "tool_message": {
                    "role": "tool",
                    "id": tool_call["id"],
                    "content": "Native argument validation failed.",
                    "error": native_error,
                    "requestor": "assistant",
                },
            }

    parameters = "".join(
        f"<parameter={name}>\n{json.dumps(value)}\n</parameter>\n"
        for name, value in arguments.items()
    )
    modification = f"</think><tool_call>\n<function=modify_pending_order_items>\n{parameters}</function>\n</tool_call>"
    backend = ScriptedBackend(["</think>Proceed.", modification, call("hand_back")])
    runtime = ErrorFixture()
    result = run_retail_episode(
        "synthetic",
        tmp_path,
        generation_identity={"fixture": 1},
        backend_factory=lambda: backend,
        native_runtime=runtime,
        relevant_operation=lambda *_: False,
    )
    assert result.status == "hand_back"
    with EpisodeJournal(tmp_path, {"fixture": 1}) as journal:
        state = journal.state()
        assert state["limitations"] == (
            [] if native_error else ["native_multi_item_modification_defect_exposure"]
        )
        assert state["db_state"] == {"count": int(not native_error)}
        assert state["ledger"][-1]["error"] is native_error
        assert (
            journal.connection.execute(
                "SELECT COUNT(*) FROM actions WHERE committed_digest IS NULL"
            ).fetchone()[0]
            == 0
        )
    resumed = run_retail_episode(
        "synthetic",
        tmp_path,
        generation_identity={"fixture": 1},
        backend_factory=lambda: pytest.fail("closed error-handling episode resampled"),
        native_runtime=runtime,
        relevant_operation=lambda *_: False,
    )
    assert resumed.status == "hand_back"
