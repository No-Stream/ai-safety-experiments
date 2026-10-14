"""The harness records failed measurements as unmeasured episodes and keeps the sweep moving."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from reward_hacking.harness import docker_backend, loop
from reward_hacking.harness.docker_backend import DockerConfig, SnapshotLimitError
from reward_hacking.harness.loop import EpisodeOutcome
from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.harness.tasks import ARTIFACT_SHAPE_TASKS, OracleError
from reward_hacking.jailed_poc import CommandResult, classify
from reward_hacking.model_backend import MockBackend
from reward_hacking.tiers import RegradeVerdict
from reward_hacking.trace import load_trace

if TYPE_CHECKING:
    from pathlib import Path


def _passing_proxy(*_args: object, **_kwargs: object) -> CommandResult:
    return CommandResult("grade", 0, "SUCCESS\n", "", False)


def _docker_task(tmp_path: Path) -> Task:
    config = DockerConfig(
        image="python:3.12-slim@sha256:" + "a" * 64,
        gate_record=tmp_path / "gate.json",
        evidence_path=tmp_path / "docker-evidence.jsonl",
        uid=1000,
        gid=1000,
    )
    return replace(
        ARTIFACT_SHAPE_TASKS[0],
        docker_config=config,
        docker_grader_command="python3 -I grade.py",
        docker_true_grader_command="python3 -I grade.py",
        docker_artifact_paths=("answer.txt",),
    )


def test_oracle_error_is_null_truth_in_the_real_episode_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def oracle_error(_episode_dir: Path) -> bool:
        raise OracleError("trusted check could not read the submission")

    task = replace(
        ARTIFACT_SHAPE_TASKS[0],
        true_check=oracle_error,
        visible_check_unsatisfiable=True,
    )
    monkeypatch.setattr(loop, "run_grader", _passing_proxy)
    trace_path = tmp_path / "trace.jsonl"

    trace = loop.run_agent_episode(
        task,
        MockBackend(["done"], model_id="oracle-error"),
        episode_dir=tmp_path / "episode",
        trace_path=trace_path,
    )

    summary = load_trace(trace_path)[-1]
    assert trace.true_pass is None
    assert trace.true_unmeasured
    assert not trace.is_hack
    assert trace.outcome is EpisodeOutcome.UNMEASURED
    assert trace.verdict is RegradeVerdict.UNMEASURED
    assert summary["true"] == {
        "pass": None,
        "measured": False,
        "oracle_error": "trusted check could not read the submission",
    }


def test_snapshot_limit_during_mediated_grade_records_episode_and_keeps_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker_task = _docker_task(tmp_path)
    evidence_path = tmp_path / "docker-evidence.jsonl"
    monkeypatch.setattr(loop, "require_gate", lambda _config: None)
    monkeypatch.setattr(docker_backend, "require_gate", lambda _config: None)
    monkeypatch.setattr(docker_backend, "prepare_runtime_mountpoints", lambda *_args: None)
    monkeypatch.setattr(
        docker_backend, "_inspect_image_volumes", lambda *_args, **_kwargs: (None, None)
    )
    monkeypatch.setattr(docker_backend, "MAXIMUM_SNAPSHOT_ENTRIES", 0)
    actual_run_grader = loop.run_grader

    def grade_or_continue(  # noqa: PLR0913 - mirrors grader inputs to preserve submitted artifact bytes
        task: Task,
        episode_dir: Path,
        *,
        arm: AgenticArmConfig,
        timeout: str,
        jail_backend: str | None,
        submitted_patch: bytes | None = None,
    ) -> CommandResult:
        if task is docker_task:
            if submitted_patch is None:
                return actual_run_grader(
                    task,
                    episode_dir,
                    arm=arm,
                    timeout=timeout,
                    jail_backend=jail_backend,
                )
            return actual_run_grader(
                task,
                episode_dir,
                arm=arm,
                timeout=timeout,
                jail_backend=jail_backend,
                submitted_patch=submitted_patch,
            )
        return CommandResult("grade", 1, "FAIL\n", "", False)

    monkeypatch.setattr(loop, "run_grader", grade_or_continue)
    trace_path = tmp_path / "trace.jsonl"

    traces = loop.run_tasks(
        MockBackend(["<check/>", "done"], model_id="snapshot-limit"),
        (docker_task, ARTIFACT_SHAPE_TASKS[1]),
        episode_base=tmp_path / "episodes",
        trace_path=trace_path,
        arm=AgenticArmConfig(check_affordance=True),
    )

    summaries = [
        record for record in load_trace(trace_path) if record.get("record") == "episode_summary"
    ]
    evidence = load_trace(evidence_path)[-1]
    trace = traces[0]
    assert evidence["phase"] == "refused"
    assert "workspace snapshot exceeded its entry or time limit" in evidence["snapshot_error"]
    assert trace.true_pass is None
    assert trace.true_unmeasured
    assert not trace.proxy_pass
    assert not trace.is_hack
    assert trace.outcome is EpisodeOutcome.UNMEASURED
    assert trace.verdict is RegradeVerdict.UNMEASURED
    assert summaries[0]["true"]["oracle_error"] == (
        "workspace_snapshot_limit: workspace snapshot exceeded its entry or time limit"
    )
    assert len(traces) == 2
    assert not traces[1].true_unmeasured


def test_unmeasured_classification_precedes_proxy_and_impossible_check() -> None:
    assert (
        classify(
            proxy_pass=False,
            true_pass=None,
            true_measured=False,
            visible_check_unsatisfiable=True,
        )
        is RegradeVerdict.UNMEASURED
    )


def test_snapshot_limit_during_final_grade_is_unmeasured_and_oserror_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = ARTIFACT_SHAPE_TASKS[0]

    def snapshot_limited_grade(*_args: object, **_kwargs: object) -> CommandResult:
        raise SnapshotLimitError("workspace snapshot exceeded its byte limit")

    monkeypatch.setattr(loop, "run_grader", snapshot_limited_grade)
    grade = loop.grade_final_state(
        task,
        tmp_path,
        arm=loop.BASELINE_ARM,
        timeout="30s",
        jail_backend=None,
        grader_hashable=False,
    )
    assert grade.true_pass is None
    assert not grade.proxy_pass
    assert (
        grade.oracle_error == "workspace_snapshot_limit: workspace snapshot exceeded its byte limit"
    )

    def os_error(*_args: object, **_kwargs: object) -> CommandResult:
        raise OSError("permission denied")

    monkeypatch.setattr(loop, "run_grader", os_error)
    with pytest.raises(OSError, match="permission denied"):
        loop.grade_final_state(
            task,
            tmp_path,
            arm=loop.BASELINE_ARM,
            timeout="30s",
            jail_backend=None,
            grader_hashable=False,
        )


def test_snapshot_limit_during_trusted_docker_grade_is_unmeasured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _docker_task(tmp_path)
    evidence_path = tmp_path / "docker-evidence.jsonl"
    monkeypatch.setattr(loop, "require_gate", lambda _config: None)
    monkeypatch.setattr(docker_backend, "require_gate", lambda _config: None)
    monkeypatch.setattr(docker_backend, "prepare_runtime_mountpoints", lambda *_args: None)
    monkeypatch.setattr(
        docker_backend, "_inspect_image_volumes", lambda *_args, **_kwargs: (None, None)
    )
    monkeypatch.setattr(docker_backend, "MAXIMUM_SNAPSHOT_ENTRIES", 0)
    actual_run_docker_grader = loop.run_docker_grader
    grader_calls = 0

    def pass_proxy_then_run_trusted(
        episode_dir: Path,
        task_files: dict[str, str],
        artifact_paths: tuple[str, ...],
        command: str,
        **kwargs: object,
    ) -> CommandResult:
        nonlocal grader_calls
        grader_calls += 1
        if grader_calls == 1:
            return CommandResult(command, 0, "SUCCESS\n", "", False)
        return actual_run_docker_grader(
            episode_dir,
            task_files,
            artifact_paths,
            command,
            **kwargs,  # pyright: ignore[reportArgumentType]
        )

    monkeypatch.setattr(loop, "run_docker_grader", pass_proxy_then_run_trusted)

    grade = loop.grade_final_state(
        task,
        tmp_path / "episode",
        arm=loop.BASELINE_ARM,
        timeout="30s",
        jail_backend=None,
        grader_hashable=False,
    )

    evidence = load_trace(evidence_path)[-1]
    assert grader_calls == 2
    assert evidence["phase"] == "refused"
    assert grade.true_pass is None
    assert not grade.proxy_pass
    assert grade.oracle_error == (
        "workspace_snapshot_limit: workspace snapshot exceeded its entry or time limit"
    )
