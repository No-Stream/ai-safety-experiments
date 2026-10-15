"""Disposable Docker transactions over immutable host-only workspace bytes."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from reward_hacking.evidence_store import read_jsonl
from reward_hacking.harness.docker_backend import (
    MAXIMUM_SNAPSHOT_BYTES,
    MAXIMUM_SNAPSHOT_ENTRIES,
    MAXIMUM_SNAPSHOT_SECONDS,
    DockerConfig,
    SnapshotLimitError,
    run_in_docker,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from reward_hacking.jailed_poc import CommandResult

PERMISSION_MASK = 0o7777
DOCKER_INFRASTRUCTURE_EXIT = 125

REQUIRED_EVALUATION_FILES = frozenset(
    {
        "job.json",
        "input.xlsx",
        "reference.xlsx",
        "submission.xlsx",
        "upstream_evaluation.py",
        "spreadsheet_worker.py",
    }
)


class RuntimeInfrastructureError(RuntimeError):
    """A transaction could not produce complete, trustworthy execution evidence."""

    def __init__(
        self,
        message: str,
        *,
        command_result: CommandResult | None = None,
        command_evidence: tuple[dict[str, object], ...] = (),
    ) -> None:
        """Retain available raw evidence after the disposable directory is removed."""
        super().__init__(message)
        self.command_result = command_result
        self.command_evidence = command_evidence


@dataclass(frozen=True)
class WorkspaceEntry:
    """One path with its exact bytes or uninterpreted symbolic-link target."""

    path: str
    kind: Literal["file", "directory", "symlink"]
    mode: int
    content: bytes = b""
    target: str = ""


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """Immutable content; no path here points at a shared writable inode."""

    entries: tuple[WorkspaceEntry, ...]


@dataclass(frozen=True)
class RuntimeAction:
    """The stopped action's full output and resulting immutable filesystem."""

    snapshot: WorkspaceSnapshot
    command_result: CommandResult


@dataclass(frozen=True)
class RuntimeEvaluation:
    """Host-only evaluator payload and its unabridged command result."""

    payload: dict[str, object]
    command_result: CommandResult


def _relative_path(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or str(relative) != path
        or "\\" in path
        or "\0" in path
    ):
        raise ValueError("workspace path must be a normalized relative path")
    return relative


def _validate_entries(snapshot: WorkspaceSnapshot) -> None:
    entries = {entry.path: entry for entry in snapshot.entries}
    if len(entries) != len(snapshot.entries):
        raise ValueError("workspace paths must be unique")
    if len(entries) > MAXIMUM_SNAPSHOT_ENTRIES:
        raise SnapshotLimitError("workspace snapshot exceeded its entry limit")
    if sum(len(entry.content) for entry in snapshot.entries) > MAXIMUM_SNAPSHOT_BYTES:
        raise SnapshotLimitError("workspace snapshot exceeded its byte limit")
    for entry in snapshot.entries:
        relative = _relative_path(entry.path)
        if entry.kind not in ("file", "directory", "symlink"):
            raise ValueError("workspace contains a special file")
        if entry.mode < 0 or entry.mode > PERMISSION_MASK:
            raise ValueError("workspace file mode is invalid")
        for parent in relative.parents:
            if parent == PurePosixPath("."):
                continue
            ancestor = entries.get(str(parent))
            if ancestor is None or ancestor.kind != "directory":
                raise ValueError("workspace parent must be an explicit directory")
        _validate_entry_content(entry)


def _validate_entry_content(entry: WorkspaceEntry) -> None:
    if entry.kind != "file" and entry.content:
        raise ValueError("only regular files may contain bytes")
    if entry.kind != "symlink" and entry.target:
        raise ValueError("only symbolic links may contain targets")
    if entry.kind == "symlink" and (not entry.target or "\0" in entry.target):
        raise ValueError("symbolic link target is invalid")


def _read_file(path: Path, remaining: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("workspace contains a special file")
        content = source.read(remaining + 1)
    if len(content) > remaining:
        raise SnapshotLimitError("workspace snapshot exceeded its byte limit")
    return content


def snapshot_workspace(workspace: Path) -> WorkspaceSnapshot:
    """Capture a stopped workspace without following any actor-created link."""
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("workspace must be a real directory")
    entries: list[WorkspaceEntry] = []
    pending = [workspace]
    started = time.monotonic()
    total_bytes = 0
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for child in children:
                if (
                    len(entries) >= MAXIMUM_SNAPSHOT_ENTRIES
                    or time.monotonic() - started > MAXIMUM_SNAPSHOT_SECONDS
                ):
                    raise SnapshotLimitError("workspace snapshot exceeded its entry or time limit")
                path = Path(child.path)
                metadata = child.stat(follow_symlinks=False)
                relative = path.relative_to(workspace).as_posix()
                mode = stat.S_IMODE(metadata.st_mode)
                if stat.S_ISLNK(metadata.st_mode):
                    entry = WorkspaceEntry(relative, "symlink", mode, target=str(path.readlink()))
                elif stat.S_ISDIR(metadata.st_mode):
                    entry = WorkspaceEntry(relative, "directory", mode)
                    pending.append(path)
                elif stat.S_ISREG(metadata.st_mode):
                    if total_bytes + metadata.st_size > MAXIMUM_SNAPSHOT_BYTES:
                        raise SnapshotLimitError("workspace snapshot exceeded its byte limit")
                    content = _read_file(path, MAXIMUM_SNAPSHOT_BYTES - total_bytes)
                    total_bytes += len(content)
                    entry = WorkspaceEntry(relative, "file", mode, content)
                else:
                    raise ValueError("workspace contains a special file")
                entries.append(entry)
    snapshot = WorkspaceSnapshot(tuple(sorted(entries, key=lambda entry: entry.path)))
    _validate_entries(snapshot)
    return snapshot


def materialize_workspace(snapshot: WorkspaceSnapshot, workspace: Path) -> None:
    """Write independent inodes into a fresh directory after validating the entire tree."""
    _validate_entries(snapshot)
    if workspace.is_symlink() or not workspace.is_dir() or any(workspace.iterdir()):
        raise ValueError("workspace destination must be an empty real directory")
    ordered = sorted(
        snapshot.entries, key=lambda entry: (len(PurePosixPath(entry.path).parts), entry.path)
    )
    for entry in ordered:
        path = workspace / entry.path
        if entry.kind == "directory":
            path.mkdir(mode=0o700)
        elif entry.kind == "symlink":
            path.symlink_to(entry.target)
        else:
            with path.open("xb") as destination:
                destination.write(entry.content)
            path.chmod(entry.mode)
    for entry in reversed(ordered):
        if entry.kind == "directory":
            (workspace / entry.path).chmod(entry.mode)


def _validate_config(config: DockerConfig) -> None:
    if (
        config.workspace_mount != "/work"
        or config.read_only_mounts
        or config.episode_tmp_state is not None
    ):
        raise ValueError(
            "artifact runtime requires only its fresh /work mount and fresh temporary state"
        )


def _checked_result(result: CommandResult, evidence: Path) -> None:
    if not evidence.is_file():
        raise RuntimeInfrastructureError(
            "Docker execution did not retain command evidence", command_result=result
        )
    records = tuple(read_jsonl(evidence))
    if result.exit_code == DOCKER_INFRASTRUCTURE_EXIT:
        raise RuntimeInfrastructureError(
            f"Docker infrastructure failed: {result.stderr}",
            command_result=result,
            command_evidence=records,
        )
    if not records or records[-1].get("phase") != "finished":
        raise RuntimeInfrastructureError(
            "Docker execution has no completed evidence record",
            command_result=result,
            command_evidence=records,
        )
    if records[-1].get("output_truncated") is not False:
        raise RuntimeInfrastructureError(
            "Docker output was truncated; observation exceeds runtime limit",
            command_result=result,
            command_evidence=records,
        )


def execute_action(
    snapshot: WorkspaceSnapshot, command: str, config: DockerConfig, *, timeout: str = "60s"
) -> RuntimeAction:
    """Keep ordinary failed-command mutations after Docker has killed every descendant.

    The caller applies its resource limiter outside this isolation boundary. Environment
    and working directory changes last for this action only; filesystem changes persist.
    """
    _validate_config(config)
    if config.trusted_grader:
        raise ValueError("actor actions cannot use the trusted grader profile")
    with tempfile.TemporaryDirectory(prefix="spec-gaming-action-", dir="/var/tmp") as temporary:
        root = Path(temporary)
        workspace = root / "work"
        workspace.mkdir()
        materialize_workspace(snapshot, workspace)
        evidence = root / "command.jsonl"
        result = run_in_docker(
            workspace,
            command,
            config=replace(config, evidence_path=evidence),
            timeout=timeout,
            login_shell=False,
        )
        _checked_result(result, evidence)
        return RuntimeAction(snapshot_workspace(workspace), result)


def run_evaluation(
    files: Mapping[str, bytes], config: DockerConfig, *, timeout: str = "120s"
) -> RuntimeEvaluation:
    """Run only host-owned evaluator files and the selected immutable submission."""
    _validate_config(config)
    if frozenset(files) != REQUIRED_EVALUATION_FILES:
        raise ValueError("evaluation requires exactly the required trusted evaluator files")
    snapshot = WorkspaceSnapshot(
        tuple(WorkspaceEntry(name, "file", 0o600, data) for name, data in files.items())
    )
    with tempfile.TemporaryDirectory(prefix="spec-gaming-evaluator-", dir="/var/tmp") as temporary:
        root = Path(temporary)
        workspace = root / "work"
        workspace.mkdir()
        materialize_workspace(snapshot, workspace)
        evidence = root / "command.jsonl"
        result = run_in_docker(
            workspace,
            "/usr/bin/python3 /work/spreadsheet_worker.py",
            config=replace(config, trusted_grader=True, evidence_path=evidence),
            timeout=timeout,
            login_shell=False,
        )
        _checked_result(result, evidence)
        if not result.ok or result.timed_out:
            raise RuntimeInfrastructureError(f"trusted evaluator failed: {result.stderr}")
        lines = result.stdout.rstrip().splitlines()
        if not lines:
            raise RuntimeInfrastructureError("trusted evaluator returned no JSON payload")
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError as error:
            raise RuntimeInfrastructureError("trusted evaluator final line is not JSON") from error
        if not isinstance(payload, dict):
            raise RuntimeInfrastructureError("trusted evaluator payload must be a JSON object")
        return RuntimeEvaluation(payload, result)
