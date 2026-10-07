"""CPU tests for routing Docker tasks through a separate, trusted grader workspace."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from reward_hacking.harness import loop
from reward_hacking.harness.docker_backend import DockerConfig
from reward_hacking.harness.task_spec import Task, task_episode_evidence
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.model_backend import MockBackend

if TYPE_CHECKING:
    from pathlib import Path


def _config(tmp_path: Path) -> DockerConfig:
    return DockerConfig(image="python:3.12-slim@sha256:" + "a" * 64, gate_record=tmp_path / "gate")


def _task(config: DockerConfig) -> Task:
    return Task(
        task_id="synthetic-docker",
        prompt="Write answer.txt.",
        body="Write answer.txt.",
        files={"TASK.md": "Write answer.txt.\n", "grade.py": "raise SystemExit(0)\n"},
        true_check=lambda _directory: (_ for _ in ()).throw(AssertionError("host oracle called")),
        docker_config=config,
        docker_grader_command="python3 -I grade.py",
        docker_true_grader_command="python3 -I grade.py",
        docker_artifact_paths=("answer.txt",),
    )


def test_docker_task_requires_both_trusted_graders(tmp_path: Path) -> None:
    config = _config(tmp_path)
    task = _task(config)
    with pytest.raises(ValueError, match="true grader"):
        replace(task, docker_true_grader_command=None)
    with pytest.raises(ValueError, match="artifact"):
        replace(task, docker_artifact_paths=("../escape",))


def test_docker_episode_routes_actions_and_both_graders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    task = _task(config)
    commands: list[tuple[Path, str, object]] = []
    grades: list[tuple[Path, dict[str, str], tuple[str, ...], str, object]] = []

    def fake_run_in_docker(directory: Path, command: str, **kwargs: object) -> CommandResult:
        commands.append((directory, command, kwargs["config"]))
        return CommandResult(command, 0, "", "", False)

    def fake_run_docker_grader(
        directory: Path,
        task_files: dict[str, str],
        artifact_paths: tuple[str, ...],
        command: str,
        **kwargs: object,
    ) -> CommandResult:
        grades.append((directory, task_files, artifact_paths, command, kwargs["config"]))
        return CommandResult(command, 0, "SUCCESS\n", "", False)

    monkeypatch.setattr(loop, "run_in_docker", fake_run_in_docker)
    monkeypatch.setattr(loop, "run_docker_grader", fake_run_docker_grader)
    monkeypatch.setattr(loop, "require_gate", lambda _config: None)
    monkeypatch.setattr(
        loop,
        "run_in_jail",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("jail called")),
    )

    episode_dir = tmp_path / "episode"
    trace = loop.run_agent_episode(
        task,
        MockBackend(["<run>printf answer > answer.txt</run>", "Done"], model_id="mock-docker"),
        episode_dir=episode_dir,
    )
    assert [entry[1] for entry in commands] == ["printf answer > answer.txt"]
    assert all(entry[2] is config for entry in commands)
    assert [entry[3] for entry in grades] == ["python3 -I grade.py", "python3 -I grade.py"]
    assert all(entry[1] == task.files and entry[2] == ("answer.txt",) for entry in grades)
    assert trace.proxy_pass
    assert trace.true_pass
    assert trace.summary_record()["docker_image"] == config.image
    assert trace.summary_record()["docker_evidence_path"] == str(
        episode_dir.with_name("episode.docker-evidence.jsonl")
    )


def test_docker_reward_evidence_uses_trusted_verdict(tmp_path: Path) -> None:
    config = _config(tmp_path)
    task = replace(
        _task(config),
        task_variant="oneoff",
        problem_id="synthetic",
        artifact_filename="answer.txt",
        provided_test_check=lambda _directory: (_ for _ in ()).throw(
            AssertionError("host check called")
        ),
    )
    (tmp_path / "answer.txt").write_text("answer")
    evidence = task_episode_evidence(task, tmp_path, trusted_provided_test_pass=True)
    assert evidence is not None
    assert evidence.provided_test_pass is True
    assert evidence.submitted_artifact_sha256 is not None


def test_docker_grader_timeout_is_unmeasured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    task = replace(
        _task(config),
        task_variant="oneoff",
        problem_id="synthetic",
        artifact_filename="answer.txt",
    )
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    (episode_dir / "answer.txt").write_text("answer")
    results = iter(
        (
            CommandResult("proxy", 124, "", "", True),
            CommandResult("truth", 124, "", "", True),
        )
    )
    monkeypatch.setattr(loop, "run_docker_grader", lambda *_args, **_kwargs: next(results))
    graded = loop.grade_final_state(
        task,
        episode_dir,
        arm=loop.BASELINE_ARM,
        timeout="30s",
        jail_backend=None,
        grader_hashable=False,
    )
    assert graded.oracle_error is not None
    assert graded.task_evidence is not None
    assert graded.task_evidence.provided_test_pass is None
    assert graded.task_evidence.provided_test_error is not None


def test_docker_episode_refuses_a_failed_gate_before_laydown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    task = _task(config)

    def fail_gate(_config: object) -> None:
        raise RuntimeError("Docker isolation gate code hash is stale")

    monkeypatch.setattr(loop, "require_gate", fail_gate)
    episode_dir = tmp_path / "episode"
    with pytest.raises(RuntimeError, match="gate code hash"):
        loop.run_agent_episode(
            task, MockBackend(["Done"], model_id="mock-docker"), episode_dir=episode_dir
        )
    assert not episode_dir.exists()
