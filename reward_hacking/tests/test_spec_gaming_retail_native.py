"""Synthetic tests for the subprocess boundary to native tau-retail."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reward_hacking.spec_gaming import retail_native

if TYPE_CHECKING:
    import pytest


def _bridge() -> retail_native.RetailNativeBridge:
    return retail_native.RetailNativeBridge(
        source_root=Path("/source/tau2-bench"),
        source_python=Path("/source/runtime/bin/python"),
    )


def _capture_worker_call(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run(command: list[str], **run_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **run_kwargs})
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(retail_native.subprocess, "run", fake_run)
    return calls


def test_describe_uses_pinned_worker_and_returns_native_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response: dict[str, object] = {
        "db_state": {"orders": {}, "products": {}, "users": {}},
        "tools": [{"type": "function", "function": {"name": "native_tool"}}],
        "actor_system_prompt": "synthetic policy",
        "actor_state": {"system_messages": [], "messages": []},
        "user_state": {"system_messages": [], "messages": []},
        "initial_trajectory": [],
        "first_agent_message": None,
        "simulator_first_prompt_messages": [],
        "task_initial_state": None,
        "task_metadata": {"reward_basis": ["DB"], "nl_assertions_required": False},
        "source_revision": retail_native.SOURCE_REVISION,
    }
    calls = _capture_worker_call(monkeypatch, response)

    result = _bridge().describe("synthetic-task", persona_config={"verbosity": "low"}, seed=7)

    assert result == response
    assert len(calls) == 1
    call = calls[0]
    assert call["command"] == [
        "/source/runtime/bin/python",
        str(Path(retail_native.__file__).resolve()),
        "--worker",
    ]
    assert json.loads(call["input"]) == {
        "action": "describe",
        "task_id": "synthetic-task",
        "persona_config": {"verbosity": "low"},
        "seed": 7,
        "source_root": "/source/tau2-bench",
    }
    assert call["cwd"] == "/source/tau2-bench"
    assert call["timeout"] == retail_native.DEFAULT_TIMEOUT_SECONDS


def test_bridge_preserves_virtualenv_python_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_directory = tmp_path / "runtime"
    runtime_directory.mkdir()
    virtualenv_python = runtime_directory / "python"
    virtualenv_python.symlink_to("/usr/bin/python3")
    calls = _capture_worker_call(monkeypatch, {"ready": True})
    bridge = retail_native.RetailNativeBridge(
        source_root=Path("/source/tau2-bench"),
        source_python=virtualenv_python,
    )

    bridge.describe("synthetic-task")

    assert calls[0]["command"][0] == str(virtualenv_python)


def test_worker_timeout_becomes_an_actionable_native_bridge_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout_run(command: list[str], **run_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, run_kwargs["timeout"])

    monkeypatch.setattr(retail_native.subprocess, "run", timeout_run)

    bridge_error: retail_native.RetailNativeBridgeError | None = None
    try:
        _bridge().describe("synthetic-task")
    except retail_native.RetailNativeBridgeError as error:
        bridge_error = error

    assert bridge_error is not None
    assert "timed out" in str(bridge_error)


def test_action_and_evaluation_use_their_frozen_timeout_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_worker_call(monkeypatch, {"status": "measured"})
    bridge = retail_native.RetailNativeBridge(
        source_root=Path("/source/tau2-bench"),
        source_python=Path("/source/runtime/bin/python"),
        action_timeout_seconds=7,
        evaluation_timeout_seconds=11,
    )

    bridge.describe("synthetic-task")
    bridge.evaluate("synthetic-task", [], "hand_back")

    assert calls[0]["timeout"] == 7
    assert calls[1]["timeout"] == 11


def test_source_runtime_info_binds_installed_packages_and_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_versions = {
        "tau2": "1.0.synthetic",
        "pydantic": "2.synthetic",
        "litellm": "1.synthetic",
        "deepdiff": "9.synthetic",
        "addict": "2.synthetic",
        "tenacity": "9.synthetic",
        "websockets": "17.synthetic",
    }
    monkeypatch.setattr(
        retail_native.importlib_metadata,
        "version",
        lambda package: package_versions[package],
    )
    monkeypatch.setattr(retail_native.sys, "version", "synthetic Python runtime")
    monkeypatch.setattr(retail_native.platform, "platform", lambda: "synthetic platform")
    monkeypatch.setattr(retail_native.sys, "executable", "/source/runtime/bin/python")

    runtime_info = retail_native._source_runtime_info()

    assert runtime_info == {
        "python_version": "synthetic Python runtime",
        "platform": "synthetic platform",
        "executable": "/source/runtime/bin/python",
        "packages": package_versions,
    }


def test_semantic_initialization_ignores_only_native_message_metadata() -> None:
    first_call: dict[str, object] = {
        "id": "call-1",
        "name": "synthetic_tool",
        "arguments": {"item_id": "business-id"},
        "requestor": "assistant",
    }
    first_actor_message: dict[str, object] = {
        "role": "assistant",
        "id": "message-1",
        "timestamp": "volatile-time-1",
        "content": "synthetic greeting",
        "tool_calls": [first_call],
    }
    first_description: dict[str, object] = {
        "actor_state": {
            "messages": [
                first_actor_message,
                {
                    "role": "tool",
                    "id": "call-1",
                    "timestamp": "volatile-time-2",
                    "content": "synthetic result",
                },
            ]
        },
        "db_state": {"orders": {"business-id": {"id": "business-id"}}},
        "actor_system_prompt": "synthetic policy",
    }
    second_description: dict[str, object] = {
        "actor_state": {
            "messages": [
                {
                    "role": "assistant",
                    "id": "message-2",
                    "timestamp": "volatile-time-3",
                    "content": "synthetic greeting",
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "name": "synthetic_tool",
                            "arguments": {"item_id": "business-id"},
                            "requestor": "assistant",
                        }
                    ],
                },
                {
                    "role": "tool",
                    "id": "call-2",
                    "timestamp": "volatile-time-4",
                    "content": "synthetic result",
                },
            ]
        },
        "db_state": {"orders": {"business-id": {"id": "business-id"}}},
        "actor_system_prompt": "synthetic policy",
    }

    first_semantics = retail_native.semantic_initialization(first_description)
    second_semantics = retail_native.semantic_initialization(second_description)

    assert first_semantics == second_semantics
    assert first_actor_message["timestamp"] == "volatile-time-1"
    assert first_call["id"] == "call-1"
    assert first_semantics["db_state"] == first_description["db_state"]


def test_user_prompt_and_turn_preserve_supplied_messages_and_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_messages = [{"role": "system", "content": "private synthetic prompt"}]
    user_turn = {
        "user_state": {"system_messages": prompt_messages, "messages": []},
        "user_message": {"role": "user", "content": "synthetic reply"},
        "is_stop": False,
        "prompt_messages": prompt_messages,
    }
    actor_text = "synthetic actor text"
    user_state = {"system_messages": prompt_messages, "messages": []}
    sampled_response = "synthetic reply"
    prompt_calls = _capture_worker_call(monkeypatch, prompt_messages)
    prompt = _bridge().user_prompt("synthetic-task", user_state, actor_text)
    assert json.loads(prompt_calls[0]["input"]) == {
        "action": "user_prompt",
        "task_id": "synthetic-task",
        "user_state": user_state,
        "incoming_actor_text": actor_text,
        "source_root": "/source/tau2-bench",
    }
    calls = _capture_worker_call(monkeypatch, user_turn)
    transitioned = _bridge().user_turn(
        "synthetic-task",
        user_state,
        actor_text,
        sampled_response,
        finish_reason="stop",
    )

    assert prompt == prompt_messages
    assert transitioned == user_turn
    assert json.loads(calls[0]["input"]) == {
        "action": "user_turn",
        "task_id": "synthetic-task",
        "user_state": user_state,
        "incoming_actor_text": actor_text,
        "sampled_response": sampled_response,
        "finish_reason": "stop",
        "persona_config": {},
        "source_root": "/source/tau2-bench",
    }


def test_native_actions_round_trip_through_separate_json_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {"tool_message": {"id": "call-1", "error": False}, "db_state": {"orders": {}}}
    calls = _capture_worker_call(monkeypatch, response)
    bridge = _bridge()

    tool_result = bridge.tool_call(
        "synthetic-task",
        {"orders": {}, "products": {}, "users": {}},
        {"id": "call-1", "name": "synthetic_tool", "arguments": {"value": 3}},
    )
    assert tool_result == response
    assert json.loads(calls[0]["input"])["action"] == "tool_call"

    evaluation_response = {
        "status": "measured",
        "termination_reason": "hand_back",
        "reward_info": {"reward": 1.0},
        "nl_assertions_status": "pending",
    }
    calls = _capture_worker_call(monkeypatch, evaluation_response)
    result = bridge.evaluate(
        "synthetic-task",
        [{"role": "assistant", "content": "synthetic trajectory"}],
        "hand_back",
    )
    assert result == evaluation_response
    assert json.loads(calls[0]["input"])["action"] == "evaluate"


def test_judge_completion_strips_thinking_and_accepts_each_exact_assertion_once() -> None:
    expected_assertions = ["synthetic assertion A", "synthetic assertion B"]
    results = [
        {
            "expectedOutcome": assertion,
            "reasoning": "synthetic reasoning",
            "metExpectation": True,
        }
        for assertion in expected_assertions
    ]
    semantic_completion = json.dumps({"results": results})

    response, error = retail_native._parse_judge_completion(
        "<think>synthetic private reasoning</think>\n" + semantic_completion,
        expected_assertions,
    )

    assert error is None
    assert response == semantic_completion


def test_judge_completion_rejects_missing_duplicate_unexpected_and_non_boolean_results() -> None:
    expected_assertions = ["synthetic assertion A", "synthetic assertion B"]
    result_a = {
        "expectedOutcome": expected_assertions[0],
        "reasoning": "synthetic reasoning",
        "metExpectation": True,
    }
    result_b = {
        "expectedOutcome": expected_assertions[1],
        "reasoning": "synthetic reasoning",
        "metExpectation": False,
    }
    invalid_results = [
        ([result_a], "result count does not match"),
        ([result_a, result_a], "every expected assertion exactly once"),
        (
            [result_a, {**result_b, "expectedOutcome": "synthetic unexpected assertion"}],
            "every expected assertion exactly once",
        ),
        ([result_a, {**result_b, "metExpectation": 1}], "Boolean metExpectation"),
    ]

    for results, expected_error in invalid_results:
        response, error = retail_native._parse_judge_completion(
            json.dumps({"results": results}), expected_assertions
        )

        assert response == ""
        assert error is not None
        assert expected_error in error


def test_judge_completion_rejects_an_unterminated_thinking_segment() -> None:
    response, error = retail_native._parse_judge_completion(
        "<think>synthetic incomplete reasoning", ["synthetic assertion"]
    )

    assert response == ""
    assert error == "Qwen thinking block ended without </think>."
