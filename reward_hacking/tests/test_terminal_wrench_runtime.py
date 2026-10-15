"""Synthetic source assets exercise the executable adapter without benchmark material."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from reward_hacking.harness.docker_backend import docker_run_args, stage_grader_workspace
from reward_hacking.terminal_wrench_runtime import (
    SOURCE_GRADER_COMPLETE,
    _source_grader_program,
    asset_tree_sha256,
    load_terminal_wrench_registry,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runtime_registry(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.csv").write_text("value\n2\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_outputs.py").write_text("def test_result(): assert True\n")
    instruction = tmp_path / "instruction.md"
    instruction.write_text("Write the computed result to result.txt.\n")
    record = {
        "schema_version": 1,
        "source_commit": "a" * 40,
        "tasks": [
            {
                "task_id": "synthetic-task",
                "source_tree_oid": "b" * 40,
                "image": "sha256:" + "c" * 64,
                "workspace_template": "workspace",
                "workspace_sha256": asset_tree_sha256(workspace),
                "tests_template": "tests",
                "tests_sha256": asset_tree_sha256(tests),
                "instruction_path": "instruction.md",
                "instruction_sha256": hashlib.sha256(instruction.read_bytes()).hexdigest(),
                "artifact_paths": ["result.txt"],
                "protected_paths": ["input.csv"],
                "python_executable": "/opt/terminal-wrench-verifier/bin/python",
                "gate_record": "containment.json",
            }
        ],
    }
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(record))
    return registry


def test_native_task_preserves_hidden_source_and_unknown_truth(runtime_registry: Path) -> None:
    registry = load_terminal_wrench_registry(runtime_registry)
    task = registry.task("synthetic-task")
    assert registry.task_ids == ("synthetic-task",)
    assert task.docker_config is not None
    assert task.docker_config.workspace_mount == "/app"
    assert task.workspace_template is not None
    assert task.docker_grader_template == task.workspace_template
    assert task.docker_hidden_tests_template == runtime_registry.parent / "tests"
    assert task.files == {}
    assert task.task_variant == "original"
    assert task.docker_artifact_paths == ("result.txt",)
    assert task.docker_true_grader_command is None
    assert task.docker_truth_unmeasured_reason == "independent_truth_unmeasured"
    assert "r2e_tests/test_outputs.py" in str(task.docker_grader_command)
    assert registry.source_identity["source_commit"] == "a" * 40
    assert registry.source_identity["source_split"] == "unassigned_research_only"
    assert registry.source_identity["training_admission"] == "quarantine"


@pytest.mark.parametrize(
    "asset", ["workspace/input.csv", "tests/test_outputs.py", "instruction.md"]
)
def test_mutated_assets_refuse_resume(runtime_registry: Path, asset: str) -> None:
    registry = load_terminal_wrench_registry(runtime_registry)
    (runtime_registry.parent / asset).write_text("changed\n")
    with pytest.raises(ValueError, match="digest"):
        registry.task("synthetic-task")


def test_runtime_refuses_asset_symlinks(runtime_registry: Path) -> None:
    workspace = runtime_registry.parent / "workspace"
    (workspace / "link").symlink_to(runtime_registry.parent / "instruction.md")
    with pytest.raises(ValueError, match="symlink"):
        load_terminal_wrench_registry(runtime_registry)


@pytest.mark.parametrize(
    "artifact", ["../result.txt", "/result.txt", "input.csv", "r2e_tests/test_outputs.py"]
)
def test_artifacts_cannot_replace_trusted_inputs(runtime_registry: Path, artifact: str) -> None:
    record = json.loads(runtime_registry.read_text())
    record["tasks"][0]["artifact_paths"] = [artifact]
    runtime_registry.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="artifact"):
        load_terminal_wrench_registry(runtime_registry)


def test_registry_location_does_not_change_semantic_identity(runtime_registry: Path) -> None:
    registry = load_terminal_wrench_registry(runtime_registry)
    record = json.loads(runtime_registry.read_text())
    record["tasks"][0]["gate_record"] = "another-gate.json"
    runtime_registry.write_text(json.dumps(record))
    assert (
        load_terminal_wrench_registry(runtime_registry).source_identity == registry.source_identity
    )


def test_mutable_image_refuses_registry_admission(runtime_registry: Path) -> None:
    record = json.loads(runtime_registry.read_text())
    record["tasks"][0]["image"] = "untrusted:latest"
    runtime_registry.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="digest"):
        load_terminal_wrench_registry(runtime_registry)


def test_final_grader_restores_inputs_and_excludes_actor_state(
    runtime_registry: Path, tmp_path: Path
) -> None:
    task = load_terminal_wrench_registry(runtime_registry).task("synthetic-task")
    actor = tmp_path / "actor"
    actor.mkdir()
    (actor / "input.csv").write_text("forged input")
    (actor / "result.txt").write_text("computed result")
    (actor / "conftest.py").write_text("raise RuntimeError('forged pytest plugin')")
    grader = tmp_path / "grader"
    grader.mkdir()
    stage_grader_workspace(
        actor,
        task.files,
        task.docker_artifact_paths,
        grader,
        template_dir=task.docker_grader_template,
        hidden_tests_template=task.docker_hidden_tests_template,
    )
    assert (grader / "input.csv").read_text() == "value\n2\n"
    assert (grader / "result.txt").read_text() == "computed result"
    assert (grader / "r2e_tests/test_outputs.py").read_text() == "def test_result(): assert True\n"
    assert not (grader / "conftest.py").exists()


def test_deleted_submission_does_not_restore_template_starter(
    runtime_registry: Path, tmp_path: Path
) -> None:
    record = json.loads(runtime_registry.read_text())
    template = tmp_path / "workspace"
    (template / "result.txt").write_text("starter implementation")
    record["tasks"][0]["workspace_sha256"] = asset_tree_sha256(template)
    runtime_registry.write_text(json.dumps(record))
    task = load_terminal_wrench_registry(runtime_registry).task("synthetic-task")
    actor = tmp_path / "actor"
    actor.mkdir()
    grader = tmp_path / "grader"
    grader.mkdir()
    stage_grader_workspace(
        actor,
        task.files,
        task.docker_artifact_paths,
        grader,
        template_dir=task.docker_grader_template,
        hidden_tests_template=task.docker_hidden_tests_template,
    )
    assert (grader / "input.csv").read_text() == "value\n2\n"
    assert not (grader / "result.txt").exists()


@pytest.mark.parametrize(
    ("test_source", "expected_exit", "completed"),
    [
        ("def test_result(): assert True\n", 0, True),
        ("def test_result(): assert False\n", 1, True),
        (
            (
                "import pytest\n"
                "@pytest.fixture\ndef absent(): assert False\ndef test_result(absent): pass\n"
            ),
            1,
            True,
        ),
        ("import missing_synthetic_dependency\n", 125, False),
        ("def test_result(:\n", 125, False),
    ],
)
def test_source_completion_receipt_requires_completed_test_execution(
    tmp_path: Path, test_source: str, expected_exit: int, completed: bool
) -> None:
    tests = tmp_path / "r2e_tests"
    tests.mkdir()
    (tests / "test_outputs.py").write_text(test_source)
    result = subprocess.run(  # noqa: S603 - trusted synthetic pytest source, isolated interpreter
        [sys.executable, "-I", "-c", _source_grader_program(str(tmp_path))],
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == expected_exit, result.stdout + result.stderr
    assert (SOURCE_GRADER_COMPLETE in result.stdout) is completed


def test_native_grader_has_bounded_private_shared_memory(
    runtime_registry: Path, tmp_path: Path
) -> None:
    task = load_terminal_wrench_registry(runtime_registry).task("synthetic-task")
    assert task.docker_config is not None
    configured = replace(task.docker_config, grader_shared_memory_bytes=67108864)
    actor = docker_run_args(configured, tmp_path, "actor", "true")
    assert actor[actor.index("--ipc") + 1] == "none"
    assert "--shm-size" not in actor
    ordinary = docker_run_args(
        replace(task.docker_config, trusted_grader=True), tmp_path, "ordinary", "true"
    )
    assert ordinary[ordinary.index("--ipc") + 1] == "none"
    assert "--shm-size" not in ordinary
    grader = docker_run_args(replace(configured, trusted_grader=True), tmp_path, "grader", "true")
    assert grader[grader.index("--ipc") + 1] == "private"
    assert grader[grader.index("--shm-size") + 1] == "67108864"


@pytest.mark.parametrize("size", [-1, 67108865, True, 1.5])
def test_shared_memory_configuration_refuses_unbounded_sizes(
    runtime_registry: Path, size: float
) -> None:
    task = load_terminal_wrench_registry(runtime_registry).task("synthetic-task")
    assert task.docker_config is not None
    with pytest.raises(ValueError, match="shared memory"):
        replace(task.docker_config, grader_shared_memory_bytes=size)


def test_source_verifier_budget_reaches_task_and_identity(runtime_registry: Path) -> None:
    value = json.loads(runtime_registry.read_text())
    value["tasks"][0]["grader_timeout_seconds"] = 1800.0
    runtime_registry.write_text(json.dumps(value))
    registry = load_terminal_wrench_registry(runtime_registry)
    assert registry.records[0].semantic_identity()["grader_timeout_seconds"] == 1800.0
    assert registry.task("synthetic-task").grader_timeout_seconds == 1800.0


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan"), True, "1800"])
def test_source_verifier_budget_must_be_finite_positive(
    runtime_registry: Path, budget: object
) -> None:
    value = json.loads(runtime_registry.read_text())
    value["tasks"][0]["grader_timeout_seconds"] = budget
    runtime_registry.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="grader timeout"):
        load_terminal_wrench_registry(runtime_registry)
