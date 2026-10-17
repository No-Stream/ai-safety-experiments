"""Only actor commands share an episode session; grading retains its own executor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.harness import jail_session, loop
from reward_hacking.harness.jail_session import current_persistent_jail, persistent_jail_scope
from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.harness.tasks import ARTIFACT_SHAPE_TASKS
from reward_hacking.jailed_poc import CommandResult

if TYPE_CHECKING:
    from pathlib import Path


def test_actor_routes_to_session_and_exception_closes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []

    class Session:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            events.append("created")

        def run(self, command: str, **kwargs: Any) -> CommandResult:
            events.append(command)
            return CommandResult(command, 0, "actor", "", False)

        def close(self) -> None:
            events.append("closed")

    monkeypatch.setattr(jail_session, "PersistentJail", Session)
    with (  # noqa: PT012 - exercise cleanup while commands and the caller exception share a scope
        pytest.raises(ValueError, match="episode failed"),
        persistent_jail_scope(tmp_path, enabled=True, backend="unshare"),
    ):
        for command in ("first", "second"):
            result = loop._run_task_command(
                ARTIFACT_SHAPE_TASKS[0],
                tmp_path,
                command,
                timeout="1s",
                jail_backend="unshare",
                cpus=1,
                mem_max="1G",
            )
            assert result.stdout == "actor"
        raise ValueError("episode failed")
    assert events == ["created", "first", "second", "closed"]
    assert current_persistent_jail(tmp_path, cpus=1, mem_max="1G") is None


def test_fallback_and_other_directories_do_not_acquire_a_session(tmp_path: Path) -> None:
    with persistent_jail_scope(tmp_path, enabled=False, backend=None):
        assert current_persistent_jail(tmp_path, cpus=1, mem_max="1G") is None
    with persistent_jail_scope(tmp_path, enabled=True, backend=None):
        assert current_persistent_jail(tmp_path / "other", cpus=1, mem_max="1G") is None


def test_episode_keeps_mediated_final_and_oracle_grading_outside_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []

    class Session:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            events.append("created")

        def run(self, command: str, **kwargs: Any) -> CommandResult:
            events.append("actor")
            return CommandResult(command, 0, "", "", False)

        def close(self) -> None:
            events.append("closed")

    class Backend:
        model_id = "synthetic-model"
        transport = "bedrock-converse"

        def generate(self, prompts: list[str]) -> list[str]:
            return [
                "Done." if "<result" in prompt else "<run>true</run><check/>" for prompt in prompts
            ]

    def grade(*args: Any, **kwargs: Any) -> CommandResult:
        events.append("grader")
        return CommandResult("synthetic-grade", 0, "SUCCESS", "", False)

    def oracle(episode_dir: Path) -> bool:
        events.append("oracle")
        return True

    task = Task(
        task_id="synthetic-session",
        prompt="Synthetic lifecycle check.",
        body="Synthetic lifecycle check.",
        files={"grade.py": "# synthetic grader\n"},
        grader_command="true",
        true_check=oracle,
    )
    monkeypatch.setattr(jail_session, "PersistentJail", Session)
    monkeypatch.setattr(loop, "run_in_jail", grade)
    loop.run_agent_episode(
        task,
        Backend(),
        episode_dir=tmp_path / "episode",
        persistent_jail=True,
        max_turns=2,
        arm=AgenticArmConfig(check_affordance=True),
    )
    assert events.count("actor") == 1
    assert events.count("grader") == 2
    assert "oracle" in events
    assert events[-1] == "closed"
