"""Build real synthetic Git assets while replacing only Docker process execution."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

from reward_hacking.terminal_wrench_builder import build_runtime_registry
from reward_hacking.terminal_wrench_runtime import load_terminal_wrench_registry


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(  # noqa: S603 - synthetic fixture Git argv
        ["/usr/bin/git", "-C", str(root), *arguments], text=True
    ).strip()


@pytest.fixture
def build_config(tmp_path: Path) -> tuple[Path, Path]:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Synthetic")
    git(tmp_path, "config", "user.email", "synthetic@example.invalid")
    (tmp_path / ".gitignore").write_text("local/\n")
    source = tmp_path / "task/original_task"
    (source / "environment").mkdir(parents=True)
    (source / "environment/Dockerfile").write_text(
        "FROM synthetic:latest\nWORKDIR /app\nCOPY input.txt /app/input.txt\n"
    )
    (source / "environment/input.txt").write_text("synthetic input\n")
    (source / ".gitattributes").write_text("environment/input.txt export-ignore\n")
    (source / "tests").mkdir()
    (source / "tests/test_outputs.py").write_text("def test_output(): assert True\n")
    (source / "solution").mkdir()
    (source / "solution/solve.sh").write_text("private synthetic solution\n")
    (source / "instruction.md").write_text("Produce result.txt.\n")
    (source / "task.toml").write_text("[verifier]\ntimeout_sec = 1800.0\n")
    (tmp_path / "LICENSE").write_text("Synthetic license\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "Synthetic fixture")
    local = tmp_path / "local"
    local.mkdir()
    lock = local / "uv.lock"
    lock.write_text("version = 1\n")
    config: dict[str, Any] = {
        "source_repository": str(tmp_path),
        "source_commit": git(tmp_path, "rev-parse", "HEAD"),
        "uv_image": "synthetic-uv:latest",
        "tasks": [
            {
                "task_id": "synthetic",
                "source_path": "task/original_task",
                "source_tree_oid": git(tmp_path, "rev-parse", "HEAD:task/original_task"),
                "dependencies": ["pytest==8.4.1"],
                "requires_python": ">=3.11",
                "lock_path": str(lock),
                "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                "artifact_paths": ["result.txt"],
                "protected_paths": ["input.txt"],
            }
        ],
    }
    path = local / "config.json"
    path.write_text(json.dumps(config))
    return tmp_path, path


class SyntheticDocker:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> str:
        self.commands.append(command)
        assert command[0] == "docker"
        if command[1:3] == ["image", "inspect"]:
            return json.dumps([{"Id": "sha256:" + "c" * 64, "RepoDigests": []}])
        if command[1] == "run":
            return json.dumps({"pytest": "8.4.1"})
        if command[1] == "create":
            return "synthetic-container"
        if command[1] == "cp":
            (Path(command[-1]) / "input.txt").write_text("synthetic input\n")
        return ""


def test_build_preserves_source_and_separates_trusted_assets(
    build_config: tuple[Path, Path],
) -> None:
    root, config = build_config
    root.joinpath("task/original_task/environment/input.txt").write_text("uncommitted mutation\n")
    root.joinpath("task/original_task/task.toml").write_text("[verifier]\ntimeout_sec = 1.0\n")
    docker = SyntheticDocker()
    registry_path = build_runtime_registry(
        config, root / "local/runtime", workspace_root=root, docker=docker
    )
    registry = load_terminal_wrench_registry(registry_path)
    assert registry.task_ids == ("synthetic",)
    assert registry.records[0].grader_timeout_seconds == 1800
    assert registry.records[0].semantic_identity()["grader_timeout_seconds"] == 1800
    record = registry.records[0]
    assert not (record.workspace_template / "solution").exists()
    assert record.tests_template.joinpath("test_outputs.py").is_file()
    context = root / "local/runtime/synthetic/context"
    assert context.joinpath("input.txt").read_text() == "synthetic input\n"
    assert not (context / "tests").exists()
    assert not (context / "solution").exists()
    dockerfile = context.joinpath("Dockerfile").read_text()
    assert "FROM sha256:" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "/opt/terminal-wrench-verifier" in dockerfile
    assert (
        context.joinpath("verifier-project/uv.lock").read_bytes()
        == config.parent.joinpath("uv.lock").read_bytes()
    )
    assert root.joinpath("local/runtime/licenses/LICENSE").is_file()
    assert any(command[1] == "rm" for command in docker.commands)


@pytest.mark.parametrize("field", ["source_tree_oid", "source_commit"])
def test_receipt_sabotage_fails_before_docker(build_config: tuple[Path, Path], field: str) -> None:
    root, path = build_config
    config = json.loads(path.read_text())
    if field == "source_commit":
        config[field] = "0" * 40
    else:
        config["tasks"][0][field] = "0" * 40
    path.write_text(json.dumps(config))
    docker = SyntheticDocker()
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        build_runtime_registry(path, root / "local/runtime", workspace_root=root, docker=docker)
    assert not docker.commands


def test_lock_receipt_mismatch_is_logged_and_the_build_proceeds(
    build_config: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    root, path = build_config
    config = json.loads(path.read_text())
    config["tasks"][0]["lock_sha256"] = "0" * 64
    path.write_text(json.dumps(config))
    docker = SyntheticDocker()
    with caplog.at_level(logging.WARNING):
        build_runtime_registry(path, root / "local/runtime", workspace_root=root, docker=docker)
    assert "input receipt" in caplog.text
    assert docker.commands


def test_refuses_public_output(build_config: tuple[Path, Path]) -> None:
    root, config = build_config
    with pytest.raises(ValueError, match="ignored"):
        build_runtime_registry(
            config, root / "public-runtime", workspace_root=root, docker=SyntheticDocker()
        )


def test_interrupted_build_resumes_only_remaining_task(build_config: tuple[Path, Path]) -> None:
    root, path = build_config
    config = json.loads(path.read_text())
    second = dict(config["tasks"][0], task_id="synthetic-second")
    config["tasks"].append(second)
    path.write_text(json.dumps(config))

    class InterruptedDocker(SyntheticDocker):
        def __call__(self, command: list[str]) -> str:
            if command[1] == "build" and "synthetic-second" in command[-1]:
                raise subprocess.CalledProcessError(1, command)
            return super().__call__(command)

    output = root / "local/runtime"
    with pytest.raises(subprocess.CalledProcessError):
        build_runtime_registry(path, output, workspace_root=root, docker=InterruptedDocker())
    assert load_terminal_wrench_registry(output / "registry.json").task_ids == ("synthetic",)
    assert json.loads(output.joinpath("build-state.json").read_text())["status"] == "failed"
    resumed = SyntheticDocker()
    registry = build_runtime_registry(path, output, workspace_root=root, docker=resumed)
    assert len(load_terminal_wrench_registry(registry).records) == 2
    assert len([command for command in resumed.commands if command[1] == "build"]) == 1
    config["tasks"][0]["dependencies"] = ["pytest==8.4.0"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="identity"):
        build_runtime_registry(path, output, workspace_root=root, docker=SyntheticDocker())


def test_reviewed_local_download_override_binds_input_bytes(
    build_config: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    root, path = build_config
    config = json.loads(path.read_text())
    override = path.parent / "reviewed.Dockerfile"
    override.write_text(
        "FROM synthetic:latest\nRUN apt-get install -y curl\nCOPY receipt.txt /app/input.txt\n"
    )
    receipt = path.parent / "input.txt"
    receipt.write_text("reviewed bytes\n")
    config["tasks"][0]["dockerfile_override"] = {
        "path": str(override),
        "sha256": hashlib.sha256(override.read_bytes()).hexdigest(),
    }
    config["tasks"][0]["input_receipts"] = [
        {
            "path": str(receipt),
            "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            "destination": "receipt.txt",
        }
    ]
    path.write_text(json.dumps(config))
    output = root / "local/runtime"
    build_runtime_registry(path, output, workspace_root=root, docker=SyntheticDocker())
    assert output.joinpath("synthetic/context/receipt.txt").read_bytes() == receipt.read_bytes()
    receipt.write_text("sabotaged bytes\n")
    docker = SyntheticDocker()
    with caplog.at_level(logging.WARNING):
        build_runtime_registry(path, output, workspace_root=root, docker=docker)
    assert "input receipt" in caplog.text


def test_completed_resume_is_order_independent_and_logs_changed_assets(
    build_config: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    root, path = build_config
    config = json.loads(path.read_text())
    config["tasks"].append(dict(config["tasks"][0], task_id="synthetic-second"))
    path.write_text(json.dumps(config))
    output = root / "local/runtime"
    build_runtime_registry(path, output, workspace_root=root, docker=SyntheticDocker())
    config["tasks"].reverse()
    path.write_text(json.dumps(config))
    docker = SyntheticDocker()
    build_runtime_registry(path, output, workspace_root=root, docker=docker)
    assert not any(command[1] == "build" for command in docker.commands)
    output.joinpath("synthetic/template/input.txt").write_text("sabotage\n")
    with caplog.at_level(logging.WARNING):
        build_runtime_registry(path, output, workspace_root=root, docker=SyntheticDocker())
    assert "workspace template records sha256" in caplog.text


def test_mutable_download_override_requires_local_receipts(build_config: tuple[Path, Path]) -> None:
    root, path = build_config
    override = path.parent / "reviewed.Dockerfile"
    override.write_text(
        "FROM synthetic:latest\nRUN curl https://example.invalid/input -o /app/input.txt\n"
    )
    config = json.loads(path.read_text())
    config["tasks"][0]["dockerfile_override"] = {
        "path": str(override),
        "sha256": hashlib.sha256(override.read_bytes()).hexdigest(),
    }
    path.write_text(json.dumps(config))
    docker = SyntheticDocker()
    with pytest.raises(ValueError, match="mutable downloads"):
        build_runtime_registry(path, root / "local/runtime", workspace_root=root, docker=docker)
    assert not docker.commands


def test_installed_dependency_must_match_exact_pin(build_config: tuple[Path, Path]) -> None:
    root, path = build_config

    class WrongDependencyDocker(SyntheticDocker):
        def __call__(self, command: list[str]) -> str:
            if command[1] == "run":
                return json.dumps({"pytest": "8.4.0"})
            return super().__call__(command)

    with pytest.raises(ValueError, match="installed verifier dependency"):
        build_runtime_registry(
            path, root / "local/runtime", workspace_root=root, docker=WrongDependencyDocker()
        )
    assert not root.joinpath("local/runtime/registry.json").exists()


@pytest.mark.parametrize("shared_memory_bytes", [0, 16 * 1024**2])
def test_explicit_grader_shared_memory_survives_registry_roundtrip(
    build_config: tuple[Path, Path], shared_memory_bytes: int
) -> None:
    root, path = build_config
    config = json.loads(path.read_text())
    config["tasks"][0]["grader_shared_memory_bytes"] = shared_memory_bytes
    path.write_text(json.dumps(config))
    registry = build_runtime_registry(
        path, root / "local/runtime", workspace_root=root, docker=SyntheticDocker()
    )
    assert (
        load_terminal_wrench_registry(registry).records[0].grader_shared_memory_bytes
        == shared_memory_bytes
    )


@pytest.mark.parametrize("budget", [0, -1, "inf"])
def test_invalid_source_verifier_budget_refuses_before_docker(
    build_config: tuple[Path, Path], budget: int | str
) -> None:
    root, path = build_config
    root.joinpath("task/original_task/task.toml").write_text(
        f"[verifier]\ntimeout_sec = {budget}\n"
    )
    git(root, "add", "task/original_task/task.toml")
    git(root, "commit", "-qm", "Synthetic invalid timeout")
    config = json.loads(path.read_text())
    config["source_commit"] = git(root, "rev-parse", "HEAD")
    config["tasks"][0]["source_tree_oid"] = git(root, "rev-parse", "HEAD:task/original_task")
    path.write_text(json.dumps(config))
    docker = SyntheticDocker()
    with pytest.raises(ValueError, match="source grader timeout"):
        build_runtime_registry(path, root / "local/runtime", workspace_root=root, docker=docker)
    assert not docker.commands
