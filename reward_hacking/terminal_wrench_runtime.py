"""Execute pinned source tasks with hidden proxy tests and explicitly unmeasured truth.

Task assets and dependency images live in a local, ignored registry. This module never reads
recorded trajectories or their outcome labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from reward_hacking.harness.docker_backend import DockerConfig
from reward_hacking.harness.task_spec import SUCCESS_MARKER, Task

SCHEMA_VERSION = 1
TRUTH_UNMEASURED_REASON = "independent_truth_unmeasured"
SOURCE_GRADER_COMPLETE = "TERMINAL_SOURCE_GRADER_COMPLETE"


def asset_tree_sha256(directory: Path) -> str:
    """Bind exact regular-file contents and names; refuse links and special files."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("runtime asset tree must be a directory, not a symlink")
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("runtime asset trees cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("runtime asset trees require regular files")
        relative = path.relative_to(directory).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or value == "."
    ):
        raise ValueError("runtime asset and artifact paths must be normalized relative paths")
    return value


def _asset_path(root: Path, value: str) -> Path:
    path = root / _relative_path(value)
    if (
        any(part.is_symlink() for part in (path, *path.parents) if part != root)
        or root not in path.resolve().parents
    ):
        raise ValueError("runtime asset paths cannot escape the registry or use symlinks")
    return path


def _hex_digest(value: str, length: int) -> str:
    if re.fullmatch(f"[a-f0-9]{{{length}}}", value) is None:
        raise ValueError("invalid source or asset digest")
    return value


def _unavailable_true_check(_directory: Path) -> bool:
    raise RuntimeError("Terminal Wrench has no independent truth checker")


def _source_grader_program(workspace: str = "/app") -> str:
    """Distinguish a completed source evaluation from collection or runtime interruption."""
    return (
        "import sys, pytest\n"
        f"sys.path.insert(0, {workspace!r})\n"
        f"sys.path.insert(0, {workspace + '/r2e_tests'!r})\n"
        "class SourceRun:\n"
        "    collected = 0\n"
        "    completed = 0\n"
        "    def pytest_collection_finish(self, session):\n"
        "        self.collected = len(session.items)\n"
        "    def pytest_runtest_logreport(self, report):\n"
        "        if report.when == 'teardown':\n"
        "            self.completed += 1\n"
        "source_run = SourceRun()\n"
        f"status = int(pytest.main([{workspace + '/r2e_tests/test_outputs.py'!r}, "
        "'-rA', '-p', 'no:cacheprovider'], plugins=[source_run]))\n"
        "if status not in (0, 1) or source_run.collected == 0 "
        "or source_run.completed != source_run.collected:\n"
        "    print('TERMINAL_SOURCE_GRADER_INCOMPLETE', status, "
        "source_run.collected, source_run.completed)\n"
        "    raise SystemExit(125)\n"
        f"print({SOURCE_GRADER_COMPLETE!r})\n"
        "raise SystemExit(status)\n"
    )


def proxy_grader_command(python_executable: str) -> str:
    """Run untouched source assertions using only the preinstalled interpreter and packages."""
    interpreter = PurePosixPath(python_executable)
    if not interpreter.is_absolute() or ".." in interpreter.parts:
        raise ValueError("verifier interpreter must be an absolute image path")
    program = _source_grader_program()
    return (
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
        f"{shlex.quote(python_executable)} -I -c {shlex.quote(program)} "
        f"&& printf '%s\\n' {shlex.quote(SUCCESS_MARKER)}"
    )


def _validate_grader_timeout(timeout: object) -> None:
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("grader timeout must be finite and positive")


@dataclass(frozen=True)
class TerminalWrenchRecord:
    """One source-only executable task, with trusted assets bound by content hashes."""

    task_id: str
    source_tree_oid: str
    image: str
    workspace_template: Path
    workspace_sha256: str
    tests_template: Path
    tests_sha256: str
    instruction_path: Path
    instruction_sha256: str
    artifact_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]
    python_executable: str
    gate_record: Path
    grader_shared_memory_bytes: int = 0
    grader_timeout_seconds: float | None = None

    def validate_assets(self) -> None:
        """Validate source asset paths and grader requirements before task construction."""
        _validate_grader_timeout(self.grader_timeout_seconds)
        asset_tree_sha256(self.workspace_template)
        asset_tree_sha256(self.tests_template)
        if self.instruction_path.is_symlink() or not self.instruction_path.is_file():
            raise ValueError("instruction assets require a regular file, not a symlink")
        if not (self.tests_template / "test_outputs.py").is_file():
            raise ValueError("source task requires its original test_outputs.py")
        if (self.workspace_template / "r2e_tests").exists():
            raise ValueError("actor workspace cannot contain hidden tests")

    def semantic_identity(self) -> dict[str, object]:
        """Exclude machine paths while retaining every reward-relevant immutable input."""
        return {
            "task_id": self.task_id,
            "source_tree_oid": self.source_tree_oid,
            "image": self.image,
            "workspace_sha256": self.workspace_sha256,
            "tests_sha256": self.tests_sha256,
            "instruction_sha256": self.instruction_sha256,
            "artifact_paths": list(self.artifact_paths),
            "protected_paths": list(self.protected_paths),
            "grader_command": proxy_grader_command(self.python_executable),
            "truth_status": TRUTH_UNMEASURED_REASON,
            "grader_shared_memory_bytes": self.grader_shared_memory_bytes,
            "grader_timeout_seconds": self.grader_timeout_seconds,
        }


@dataclass(frozen=True)
class TerminalWrenchRegistry:
    """A deterministic collection selected exclusively from original task source assets."""

    source_commit: str
    records: tuple[TerminalWrenchRecord, ...]

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return the complete admitted source task list in registry order."""
        return tuple(record.task_id for record in self.records)

    @property
    def source_identity(self) -> dict[str, object]:
        """Describe semantic content independently of checkout or gate-record locations."""
        identity: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "source_commit": self.source_commit,
            "source_split": "unassigned_research_only",
            "training_admission": "quarantine",
            "tasks": [record.semantic_identity() for record in self.records],
        }
        identity["sha256"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        return identity

    def task(self, task_id: str) -> Task:
        """Construct one native Docker task; the upstream grader supplies proxy reward only."""
        record = next((record for record in self.records if record.task_id == task_id), None)
        if record is None:
            raise KeyError(task_id)
        record.validate_assets()
        instruction = record.instruction_path.read_text()
        return Task(
            task_id=f"terminal_wrench/{task_id}",
            prompt=instruction,
            body=instruction,
            files={},
            true_check=_unavailable_true_check,
            task_variant="original",
            problem_id=task_id,
            artifact_filename=record.artifact_paths[0],
            docker_config=DockerConfig(
                image=record.image,
                gate_record=record.gate_record,
                workspace_mount="/app",
                memory_bytes=2 * 1024**3,
                pids_limit=128,
                cpus=2,
                grader_shared_memory_bytes=record.grader_shared_memory_bytes,
            ),
            docker_grader_command=proxy_grader_command(record.python_executable),
            grader_timeout_seconds=record.grader_timeout_seconds,
            docker_truth_unmeasured_reason=TRUTH_UNMEASURED_REASON,
            docker_artifact_paths=record.artifact_paths,
            workspace_template=record.workspace_template,
            docker_grader_template=record.workspace_template,
            docker_hidden_tests_template=record.tests_template,
            host_only_metadata={"terminal_wrench_source": record.semantic_identity()},
        )


def _load_record(root: Path, value: dict[str, Any]) -> TerminalWrenchRecord:
    artifacts = tuple(_relative_path(path) for path in value["artifact_paths"])
    protected = tuple(_relative_path(path) for path in value["protected_paths"])
    if not artifacts or len(set(artifacts)) != len(artifacts):
        raise ValueError("artifact paths must be nonempty and unique")
    if any(
        path == "r2e_tests"
        or path.startswith("r2e_tests/")
        or any(
            path == fixed or path.startswith(fixed + "/") or fixed.startswith(path + "/")
            for fixed in protected
        )
        for path in artifacts
    ):
        raise ValueError("artifact paths cannot replace trusted inputs or hidden tests")
    record = TerminalWrenchRecord(
        task_id=value["task_id"],
        source_tree_oid=_hex_digest(value["source_tree_oid"], 40),
        image=value["image"],
        workspace_template=_asset_path(root, value["workspace_template"]),
        workspace_sha256=_hex_digest(value["workspace_sha256"], 64),
        tests_template=_asset_path(root, value["tests_template"]),
        tests_sha256=_hex_digest(value["tests_sha256"], 64),
        instruction_path=_asset_path(root, value["instruction_path"]),
        instruction_sha256=_hex_digest(value["instruction_sha256"], 64),
        artifact_paths=artifacts,
        protected_paths=protected,
        python_executable=value["python_executable"],
        gate_record=_asset_path(root, value["gate_record"]),
        grader_shared_memory_bytes=value.get("grader_shared_memory_bytes", 0),
        grader_timeout_seconds=value.get("grader_timeout_seconds"),
    )
    record.validate_assets()
    DockerConfig(
        image=record.image,
        gate_record=record.gate_record,
        workspace_mount="/app",
        grader_shared_memory_bytes=record.grader_shared_memory_bytes,
    )
    return record


def load_terminal_wrench_registry(path: Path) -> TerminalWrenchRegistry:
    """Load a local source receipt and verify every actor and hidden-grader asset."""
    value = json.loads(path.read_text())
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported Terminal Wrench runtime schema")
    records = tuple(_load_record(path.resolve().parent, entry) for entry in value["tasks"])
    task_ids = tuple(record.task_id for record in records)
    if not records or len(set(task_ids)) != len(task_ids):
        raise ValueError("runtime task IDs must be nonempty and unique")
    return TerminalWrenchRegistry(_hex_digest(value["source_commit"], 40), records)
