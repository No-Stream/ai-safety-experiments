"""Task-specific grader deadlines leave actor command deadlines and legacy identity intact."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.harness import loop
from reward_hacking.harness.docker_backend import DockerConfig
from reward_hacking.harness.task_spec import Task
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.organism_rl import rollout

if TYPE_CHECKING:
    from pathlib import Path


def _task(*, grader_timeout_seconds: float | None = None) -> Task:
    return Task(
        task_id="synthetic-task",
        prompt="Write answer.txt.",
        body="Write answer.txt.",
        files={"TASK.md": "Write answer.txt.\n", "grade.py": "raise SystemExit(0)\n"},
        true_check=lambda _directory: False,
        grader_timeout_seconds=grader_timeout_seconds,
    )


def _docker_task(tmp_path: Path, *, grader_timeout_seconds: float | None = None) -> Task:
    return Task(
        task_id="synthetic-docker-task",
        prompt="Write answer.txt.",
        body="Write answer.txt.",
        files={"TASK.md": "Write answer.txt.\n", "grade.py": "raise SystemExit(0)\n"},
        true_check=lambda _directory: False,
        docker_config=DockerConfig(
            "python:3.12-slim@sha256:" + "a" * 64,
            tmp_path / "gate.json",
        ),
        docker_grader_command="python3 -I grade.py",
        docker_true_grader_command="python3 -I grade.py",
        docker_artifact_paths=("answer.txt",),
        grader_timeout_seconds=grader_timeout_seconds,
    )


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_task_rejects_nonpositive_or_nonfinite_grader_timeouts(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _task(grader_timeout_seconds=value)


@pytest.mark.parametrize("value", [True, "1800"])
def test_task_rejects_wrong_grader_timeout_types(value: object) -> None:
    with pytest.raises(TypeError, match="int or float"):
        Task(
            task_id="synthetic-task",
            prompt="Write answer.txt.",
            body="Write answer.txt.",
            files={"TASK.md": "Write answer.txt.\n", "grade.py": "raise SystemExit(0)\n"},
            true_check=lambda _directory: False,
            grader_timeout_seconds=cast("Any", value),
        )


def test_grader_timeout_changes_task_semantics_only_when_configured() -> None:
    legacy_task = _task()
    legacy_content = rollout.task_definition_content(legacy_task)
    timed_task = replace(legacy_task, grader_timeout_seconds=1800)
    timed_content = rollout.task_definition_content(timed_task)

    assert "grader_timeout_seconds" not in legacy_content
    assert timed_content["grader_timeout_seconds"] == 1800
    assert rollout.task_definition_content(_task()) == legacy_content
    assert timed_content != legacy_content


def test_local_grader_uses_task_timeout_and_actor_command_keeps_its_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(grader_timeout_seconds=1800)
    observed_runs: list[tuple[str, object]] = []

    def fake_run_in_jail(_directory: Path, command: str, **kwargs: object) -> CommandResult:
        observed_runs.append((str(kwargs["timeout"]), kwargs.get("process_timeout_seconds")))
        return CommandResult(command, 0, "SUCCESS\n", "", False)

    monkeypatch.setattr(loop, "run_in_jail", fake_run_in_jail)
    loop.run_grader(
        task,
        tmp_path,
        arm=loop.BASELINE_ARM,
        timeout="120s",
        jail_backend=None,
    )
    loop._run_task_command(
        task,
        tmp_path,
        "printf actor-command",
        timeout="120s",
        jail_backend=None,
        cpus=1,
        mem_max="1G",
    )
    loop.run_grader(
        _task(),
        tmp_path,
        arm=loop.BASELINE_ARM,
        timeout="120s",
        jail_backend=None,
    )

    assert observed_runs == [("1800s", 1800), ("120s", None), ("120s", None)]


def test_docker_proxy_and_direct_true_graders_use_the_task_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _docker_task(tmp_path, grader_timeout_seconds=17.5)
    observed_timeouts: list[str] = []

    def fake_run_docker_grader(
        _directory: Path,
        _files: dict[str, str],
        _artifact_paths: tuple[str, ...],
        command: str,
        **kwargs: object,
    ) -> CommandResult:
        observed_timeouts.append(str(kwargs["timeout"]))
        return CommandResult(command, 0, "SUCCESS\n", "", False)

    monkeypatch.setattr(loop, "run_docker_grader", fake_run_docker_grader)
    monkeypatch.setattr(loop, "require_gate", lambda _config: None)
    loop.grade_final_state(
        task,
        tmp_path,
        arm=loop.BASELINE_ARM,
        timeout="120s",
        jail_backend=None,
        grader_hashable=False,
    )

    # Docker accepts whole-second deadlines; fractional seconds round up to keep the bound valid.
    assert observed_timeouts == ["18s", "18s"]


def test_submission_grader_formats_feedback_with_the_effective_task_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = Task(
        task_id="synthetic-docker-task",
        prompt="Write answer.txt.",
        body="Write answer.txt.",
        files={"TASK.md": "Write answer.txt.\n", "grade.py": "raise SystemExit(0)\n"},
        true_check=lambda _directory: False,
        task_variant="original",
        problem_id="synthetic-problem",
        artifact_filename="answer.txt",
        docker_config=DockerConfig(
            "python:3.12-slim@sha256:" + "a" * 64,
            tmp_path / "gate.json",
        ),
        docker_grader_command="python3 -I grade.py",
        docker_true_grader_command="python3 -I grade.py",
        docker_artifact_paths=("answer.txt",),
        grader_timeout_seconds=17.5,
    )
    formatted_timeouts: list[str] = []

    def fake_run_grader(*_args: object, **_kwargs: object) -> CommandResult:
        return CommandResult("python3 -I grade.py", 0, "SUCCESS\n", "", False)

    def fake_format_bash_result(_result: CommandResult, timeout: str) -> tuple[str, bool]:
        formatted_timeouts.append(timeout)
        return "grader feedback", False

    monkeypatch.setattr(rollout, "run_grader", fake_run_grader)
    monkeypatch.setattr(rollout, "format_bash_result", fake_format_bash_result)
    verdict = rollout._task_submission_grader(task, jail_backend=None)(tmp_path)

    assert verdict.passed is True
    assert formatted_timeouts == ["18s"]
