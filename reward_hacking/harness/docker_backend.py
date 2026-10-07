"""Digest-pinned, disposable Docker commands and fresh submitted-artifact grading."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

from reward_hacking.jailed_poc import CommandResult

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from io import BufferedReader

logger = logging.getLogger(__name__)
MINIMUM_PIDS = 4
MAXIMUM_ARTIFACT_BYTES = 64 * 1024 * 1024
MAXIMUM_OUTPUT_BYTES = 1024 * 1024
MAXIMUM_SNAPSHOT_BYTES = 1024 * 1024 * 1024
MAXIMUM_SNAPSHOT_ENTRIES = 20000
MAXIMUM_SNAPSHOT_SECONDS = 10
HASH_CHUNK_BYTES = 1024 * 1024


class SnapshotLimitError(RuntimeError):
    """An agent workspace exceeded the bounded host evidence scan."""


REQUIRED_PROBES = frozenset(
    {
        "network_dns_blocked",
        "network_public_tcp_blocked",
        "network_gateway_tcp_blocked",
        "host_path_unreadable",
        "outside_workspace_write_blocked",
        "docker_socket_unreachable",
        "privilege_escalation_blocked",
        "pids_limit_enforced",
        "memory_limit_enforced",
        "grader_tampering_isolated",
    }
)


@dataclass(frozen=True)
class DockerConfig:
    """Trusted host configuration; no model-controlled Docker flags are admitted."""

    image: str
    gate_record: Path
    memory_bytes: int = 268435456
    pids_limit: int = 64
    cpus: float = 1
    uid: int = field(default_factory=os.getuid)
    gid: int = field(default_factory=os.getgid)
    evidence_path: Path | None = None

    def __post_init__(self) -> None:
        """Reject mutable image references and invalid containment limits."""
        if re.fullmatch(r"[a-zA-Z0-9./_:-]+@sha256:[a-f0-9]{64}", self.image) is None:
            raise ValueError("Docker image must be pinned by a full sha256 digest")
        if self.uid <= 0 or self.gid < 0:
            raise ValueError("Docker episodes require a non-root numeric user")
        if (
            self.memory_bytes <= 0
            or self.pids_limit < MINIMUM_PIDS
            or not math.isfinite(self.cpus)
            or self.cpus <= 0
        ):
            raise ValueError("Docker resource limits must be positive (pids >= 4)")


def backend_code_hash() -> str:
    """Bind certification to both execution routes, task configuration and the attack suite."""
    root = Path(__file__).resolve().parents[2]
    paths = (
        "reward_hacking/harness/docker_backend.py",
        "reward_hacking/harness/loop.py",
        "reward_hacking/harness/task_spec.py",
        "scripts/run_docker_jail_tests.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode())
        digest.update((root / path).read_bytes())
    return digest.hexdigest()


def require_gate(config: DockerConfig) -> None:
    """Refuse uncertified or vacuous containment; record stays outside the agent mount."""
    if not config.gate_record.is_file() or config.gate_record.is_symlink():
        raise RuntimeError("Docker isolation gate record missing; run make docker-jail-test")
    record = cast("dict[str, object]", json.loads(config.gate_record.read_text()))
    if record.get("backend_hash") != backend_code_hash():
        raise RuntimeError("Docker isolation gate code hash is stale")
    probes = record.get("probes")
    if (
        record.get("passed") is not True
        or not isinstance(probes, dict)
        or frozenset(probes) != REQUIRED_PROBES
    ):
        raise RuntimeError("Docker isolation gate probe set incomplete")
    for probe in REQUIRED_PROBES:
        result = probes[probe]
        if (
            not isinstance(result, dict)
            or result.get("contained") is not True
            or result.get("negative_control") is not True
        ):
            raise RuntimeError(f"Docker isolation gate probe failed: {probe}")


def validate_workspace(workspace: Path) -> Path:
    """Mount only a disposable directory, never a home tree or a spelling of the root."""
    resolved = workspace.resolve(strict=True)
    if not resolved.is_dir() or resolved == Path(resolved.anchor):
        raise ValueError("Docker workspace must be a disposable directory")
    if (
        resolved == Path.home()
        or Path.home() in resolved.parents
        or resolved.parts[1:2] in (("home",), ("root",))
    ):
        raise ValueError("Docker workspace must stay outside home trees")
    if "," in str(resolved):
        raise ValueError("Docker workspace path cannot contain mount option delimiters")
    return resolved


def docker_run_args(  # noqa: PLR0913 - fixed profile plus command options
    config: DockerConfig,
    workspace: Path,
    name: str,
    command: str,
    *,
    timeout_seconds: int = 30,
    login_shell: bool = False,
) -> list[str]:
    """Build one mount and a fixed containment profile, overriding the image entrypoint."""
    workspace = validate_workspace(workspace)
    if timeout_seconds <= 0:
        raise ValueError("Docker command timeout must be positive")
    return [
        "docker",
        "run",
        "--rm",
        "--log-driver",
        "none",
        "--ulimit",
        "fsize=67108864:67108864",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--no-healthcheck",
        "--ipc",
        "none",
        "--cgroupns",
        "private",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{config.uid}:{config.gid}",
        "--pids-limit",
        str(config.pids_limit),
        "--memory",
        str(config.memory_bytes),
        "--memory-swap",
        str(config.memory_bytes),
        "--cpus",
        str(config.cpus),
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=67108864,mode=1777",  # noqa: S108 - container-only tmpfs
        "--mount",
        f"type=bind,src={workspace},dst=/work,bind-recursive=disabled",
        "--workdir",
        "/work",
        "--env",
        "HOME=/tmp",
        "--env",
        "TMPDIR=/tmp",
        "--entrypoint",
        "/usr/bin/timeout",
        config.image,
        "--signal=KILL",
        f"{timeout_seconds}s",
        "/bin/sh",
        "-lc" if login_shell else "-c",
        command,
    ]


def _relative_path(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("submitted paths must stay inside the workspace")
    return relative


@contextmanager
def _open_regular_file(root: Path, relative: str) -> Generator[BufferedReader]:
    """Open each component without following agent-created symlinks or special files."""
    parts = _relative_path(relative).parts
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
        )
        with os.fdopen(file_descriptor, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("submitted artifacts must be regular files")
            yield source
    finally:
        os.close(descriptor)


def _read_regular_file(root: Path, relative: str) -> bytes:
    with _open_regular_file(root, relative) as source:
        content = source.read(MAXIMUM_ARTIFACT_BYTES + 1)
    if len(content) > MAXIMUM_ARTIFACT_BYTES:
        raise ValueError("submitted artifact exceeds the 64 MiB grading transfer limit")
    return content


def submitted_files(workspace: Path, paths: tuple[str, ...]) -> dict[str, bytes]:
    """Read only explicitly submitted files; absent artifacts remain absent at grading."""
    result: dict[str, bytes] = {}
    for path in paths:
        _relative_path(path)
        try:
            result[path] = _read_regular_file(workspace, path)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"unsafe submitted artifact {path}") from error
    return result


def _workspace_paths(workspace: Path) -> Generator[Path]:
    pending = [workspace]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                yield Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))


def workspace_snapshot(workspace: Path) -> dict[str, dict[str, str]]:
    """Record bounded content hashes, permissions and links without following external targets."""
    snapshot: dict[str, dict[str, str]] = {}
    started = time.monotonic()
    bytes_hashed = 0
    for path in _workspace_paths(workspace):
        if (
            len(snapshot) >= MAXIMUM_SNAPSHOT_ENTRIES
            or time.monotonic() - started > MAXIMUM_SNAPSHOT_SECONDS
        ):
            raise SnapshotLimitError("workspace snapshot exceeded its entry or time limit")
        relative = path.relative_to(workspace).as_posix()
        mode = path.lstat().st_mode
        entry = {"mode": oct(stat.S_IMODE(mode))}
        if stat.S_ISLNK(mode):
            entry["symlink"] = str(path.readlink())
        elif stat.S_ISREG(mode):
            digest = hashlib.sha256()
            with _open_regular_file(workspace, relative) as source:
                for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b""):
                    bytes_hashed += len(chunk)
                    if (
                        bytes_hashed > MAXIMUM_SNAPSHOT_BYTES
                        or time.monotonic() - started > MAXIMUM_SNAPSHOT_SECONDS
                    ):
                        raise SnapshotLimitError(
                            "workspace snapshot exceeded its byte or time limit"
                        )
                    digest.update(chunk)
            entry["sha256"] = digest.hexdigest()
            entry["size"] = str(path.lstat().st_size)
        elif stat.S_ISDIR(mode):
            entry["directory"] = "true"
        else:
            entry["special"] = str(stat.S_IFMT(mode))
        snapshot[relative] = entry
    return snapshot


def _timeout_seconds(timeout: str) -> int:
    match = re.fullmatch(r"(\d+)([smh]?)", timeout)
    if match is None:
        raise ValueError("Docker timeout must be an integer with optional s/m/h suffix")
    return int(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[match[2]]


@dataclass
class _StreamCapture:
    data: bytearray = field(default_factory=bytearray)
    truncated: bool = False
    error: OSError | None = None


def _drain_stream(stream: BufferedReader, capture: _StreamCapture) -> None:
    """Keep a bounded prefix while always reading enough to unblock the Docker client."""
    try:
        for chunk in iter(lambda: stream.read(65536), b""):
            remaining = MAXIMUM_OUTPUT_BYTES - len(capture.data)
            capture.data.extend(chunk[:remaining])
            if len(chunk) > remaining:
                capture.truncated = True
    except OSError as error:
        capture.error = error


def _wait_for_docker_client(
    process: subprocess.Popen[bytes], deadline_seconds: int
) -> tuple[int, bool]:
    try:
        return process.wait(timeout=deadline_seconds), False
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        return 124, True


def _capture_text(capture: _StreamCapture) -> str:
    if capture.error is not None:
        raise capture.error
    return capture.data.decode("utf-8", errors="replace")


def _run_bounded_docker(argv: list[str], deadline_seconds: int) -> tuple[int, str, str, bool, bool]:
    """Bound retained client output and reap the client even when its host deadline fires."""
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Docker client pipes were not opened")
    stdout_capture = _StreamCapture()
    stderr_capture = _StreamCapture()
    readers = (
        threading.Thread(target=_drain_stream, args=(process.stdout, stdout_capture), daemon=True),
        threading.Thread(target=_drain_stream, args=(process.stderr, stderr_capture), daemon=True),
    )
    for reader in readers:
        reader.start()
    try:
        exit_code, timed_out = _wait_for_docker_client(process, deadline_seconds)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for reader in readers:
            reader.join(timeout=5)
        if not any(reader.is_alive() for reader in readers):
            process.stdout.close()
            process.stderr.close()
    if any(reader.is_alive() for reader in readers):
        raise RuntimeError("Docker client output pipes did not close after termination")
    stdout = _capture_text(stdout_capture)
    stderr = _capture_text(stderr_capture)
    output_truncated = stdout_capture.truncated or stderr_capture.truncated
    if output_truncated:
        stderr += f"\n[Docker output truncated to {MAXIMUM_OUTPUT_BYTES} bytes per stream]"
    if timed_out:
        stderr += "\nDocker command deadline exceeded"
    return exit_code, stdout, stderr, timed_out, output_truncated


def run_in_docker(
    episode_dir: Path,
    command: str,
    *,
    config: DockerConfig,
    timeout: str = "30s",
    login_shell: bool = True,
) -> CommandResult:
    """Run a fresh container and persist command evidence on the host, outside its mount."""
    workspace = validate_workspace(episode_dir)
    for protected in (config.gate_record, config.evidence_path):
        if protected is not None and (
            protected.resolve() == workspace or workspace in protected.resolve().parents
        ):
            raise ValueError("gate and evidence must stay outside the episode mount")
    require_gate(config)
    inspected = subprocess.run(  # noqa: S603 - trusted pinned image metadata
        ["docker", "image", "inspect", "--format", "{{json .Config.Volumes}}", config.image],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    if json.loads(inspected.stdout) not in (None, {}):
        raise ValueError("Docker episode images must not declare automatic writable volumes")
    evidence_path = config.evidence_path or workspace.with_name(
        workspace.name + ".docker-evidence.jsonl"
    )
    name = "episode-" + uuid.uuid4().hex
    seconds = _timeout_seconds(timeout)
    argv = docker_run_args(
        config, workspace, name, command, timeout_seconds=seconds, login_shell=login_shell
    )
    evidence: dict[str, object] = {
        "command": command,
        "container": name,
        "image": config.image,
        "backend_hash": backend_code_hash(),
        "phase": "started",
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        evidence["before"] = workspace_snapshot(workspace)
    except (OSError, SnapshotLimitError) as error:
        evidence.update({"phase": "refused", "snapshot_error": str(error)})
        with evidence_path.open("a") as output:
            output.write(json.dumps(evidence) + "\n")
        raise
    with evidence_path.open("a") as output:
        output.write(json.dumps(evidence) + "\n")
    try:
        exit_code, stdout, stderr, host_timed_out, output_truncated = _run_bounded_docker(
            argv, seconds + 15
        )
        result = CommandResult(
            command,
            exit_code,
            stdout,
            stderr,
            host_timed_out or exit_code in (124, 137),
        )
    finally:
        cleanup = subprocess.run(  # noqa: S603 - generated container name
            ["docker", "rm", "--force", name],  # noqa: S607 - installed host Docker CLI
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if cleanup.returncode != 0 and "No such container" not in cleanup.stderr:
            raise RuntimeError(f"Docker container cleanup failed: {cleanup.stderr.strip()}")
    evidence.update(
        {
            "phase": "finished",
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "output_truncated": output_truncated,
        }
    )
    try:
        evidence["after"] = workspace_snapshot(workspace)
    except (OSError, SnapshotLimitError) as error:
        evidence["snapshot_error"] = str(error)
        with evidence_path.open("a") as output:
            output.write(json.dumps(evidence) + "\n")
        raise
    with evidence_path.open("a") as output:
        output.write(json.dumps(evidence) + "\n")
    return result


def run_docker_grader(  # noqa: PLR0913 - submission and trusted grader are distinct inputs
    episode_dir: Path,
    task_files: Mapping[str, str],
    artifact_paths: tuple[str, ...],
    command: str,
    *,
    config: DockerConfig,
    timeout: str = "30s",
) -> CommandResult:
    """Grade a copied submission with trusted baseline files in a separate fresh container."""
    evidence = config.evidence_path or episode_dir.with_name(
        episode_dir.name + ".docker-evidence.jsonl"
    )
    from_config = DockerConfig(
        config.image,
        config.gate_record,
        config.memory_bytes,
        config.pids_limit,
        config.cpus,
        config.uid,
        config.gid,
        evidence,
    )
    with tempfile.TemporaryDirectory(prefix="docker-grade-") as temporary:
        workspace = Path(temporary)
        stage_grader_workspace(episode_dir, task_files, artifact_paths, workspace)
        return run_in_docker(
            workspace, command, config=from_config, timeout=timeout, login_shell=False
        )


def stage_grader_workspace(
    episode_dir: Path,
    task_files: Mapping[str, str],
    artifact_paths: tuple[str, ...],
    workspace: Path,
) -> None:
    """Reconstruct trusted grading inputs; never import agent state beyond the submission."""
    if any(path in task_files for path in artifact_paths):
        raise ValueError("submitted artifacts cannot replace trusted task files")
    if any(workspace.iterdir()):
        raise ValueError("trusted grader workspace must start empty")
    artifacts = submitted_files(episode_dir, artifact_paths)
    for relative, content in task_files.items():
        target = workspace / _relative_path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    for relative, content in artifacts.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
