"""Digest-pinned, disposable Docker commands and fresh submitted-artifact grading."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from reward_hacking.jailed_poc import CommandResult

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from io import BufferedReader

logger = logging.getLogger(__name__)
MINIMUM_PIDS = 4
# Test runners may create compiler and helper processes; 512 is a finite ceiling with room for them.
GRADER_PIDS_LIMIT = 512
GRADER_THREAD_LIMITS = (
    "OPENBLAS_NUM_THREADS=1",
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
)
MAXIMUM_ARTIFACT_BYTES = 64 * 1024 * 1024
MAXIMUM_OUTPUT_BYTES = 1024 * 1024
MAXIMUM_JUNIT_XML_BYTES = 16 * 1024 * 1024
MAXIMUM_TMP_STATE_BYTES = 64 * 1024 * 1024
MAXIMUM_SNAPSHOT_BYTES = 1024 * 1024 * 1024
MAXIMUM_SNAPSHOT_ENTRIES = 20000
# The entry and byte caps bound the work; the clock is only a backstop. Six concurrent gradings
# in one process stretch a 2,000-entry snapshot from 0.3 s to about 10 s.
MAXIMUM_SNAPSHOT_SECONDS = 60
HASH_CHUNK_BYTES = 1024 * 1024
READ_ONLY_RUNTIME_ROOT = Path("/var/tmp/leak-env/runtime")  # noqa: S108 - explicit trusted runtime allowlist
DOCKER_RUNTIME_MOUNTS = frozenset(
    (
        PurePosixPath("/dev"),
        PurePosixPath("/proc"),
        PurePosixPath("/run"),
        PurePosixPath("/sys"),
        PurePosixPath("/", "tmp"),
    )
)


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
        "runtime_mount_write_blocked",
        "runtime_mount_host_escape_blocked",
    }
)


def validate_runtime_source(source: Path) -> Path:
    """Admit only canonical directories below the trusted runtime root."""
    root = READ_ONLY_RUNTIME_ROOT.resolve(strict=True)
    if (
        not source.is_absolute()
        or ".." in source.parts
        or source.is_symlink()
        or not source.is_dir()
        or source.resolve(strict=True) != source
        or source == root
        or root not in source.parents
        or "," in str(source)
    ):
        raise ValueError(
            "runtime mount source must be a canonical directory under the allowlisted runtime root"
        )
    return source


@dataclass(frozen=True)
class ReadOnlyMount:
    """A trusted runtime directory mounted read-only with nested host mounts excluded."""

    source: Path
    target: str

    def __post_init__(self) -> None:
        """Reject mutable source aliases and unsafe container destinations."""
        validate_runtime_source(self.source)
        destination = PurePosixPath(self.target)
        if (
            not destination.is_absolute()
            or str(destination) != self.target
            or re.fullmatch(r"/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", self.target) is None
            or ".." in destination.parts
            or destination in DOCKER_RUNTIME_MOUNTS
            or any(parent in DOCKER_RUNTIME_MOUNTS for parent in destination.parents)
        ):
            raise ValueError("runtime mount target must be a normalized absolute container path")


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
    workspace_mount: str = "/work"
    read_only_mounts: tuple[ReadOnlyMount, ...] = ()
    trusted_grader: bool = False
    episode_tmp_state: Path | None = None

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
        if not self.workspace_mount:
            raise ValueError("Docker workspace mount must be a normalized absolute workspace mount")
        mount_path = PurePosixPath(self.workspace_mount)
        if (
            not mount_path.is_absolute()
            or mount_path == PurePosixPath("/")
            or ".." in mount_path.parts
            or "," in self.workspace_mount
            or str(mount_path) != self.workspace_mount
            or any(runtime_mount in mount_path.parents for runtime_mount in DOCKER_RUNTIME_MOUNTS)
            or mount_path in DOCKER_RUNTIME_MOUNTS
        ):
            raise ValueError("Docker workspace mount must be a normalized absolute workspace mount")
        targets = [PurePosixPath(mount.target) for mount in self.read_only_mounts]
        for destination in targets:
            if destination != mount_path / ".venv" and (
                PurePosixPath("/opt") not in destination.parents
                or destination == mount_path
                or mount_path in destination.parents
                or destination in mount_path.parents
            ):
                raise ValueError(
                    "runtime mounts may target only /opt children or the workspace .venv"
                )
        if len(targets) != len(set(targets)) or any(
            first in second.parents or second in first.parents
            for index, first in enumerate(targets)
            for second in targets[index + 1 :]
        ):
            raise ValueError("runtime mount targets must not overlap")
        if self.trusted_grader and self.episode_tmp_state is not None:
            raise ValueError("trusted graders require a fresh /tmp")


def runtime_mount_args(mount: ReadOnlyMount) -> str:
    """Use a read-only Docker bind so Desktop can translate distro-local paths."""
    validate_runtime_source(mount.source)
    return f"type=bind,source={mount.source},target={mount.target},readonly,bind-recursive=disabled"


def runtime_mount_guard(config: DockerConfig) -> str:
    """Refuse missing or weakened kernel runtime mounts before policy code executes."""
    if not config.read_only_mounts:
        return ""
    checks = [
        f"_runtime_flags=$(_runtime_mount_flags {shlex.quote(mount.target)}) || exit 125; "
        "for _runtime_required in ro; do "
        'case ",$_runtime_flags," in *",$_runtime_required,"*) ;; *) exit 125 ;; esac; done;'
        for mount in config.read_only_mounts
    ]
    return (
        "_runtime_mount_flags() { "
        "while IFS=' ' read -r _id _parent _device _root _target _flags _rest; do "
        'if [ "$_target" = "$1" ]; then printf "%s" "$_flags"; return 0; fi; '
        "done < /proc/self/mountinfo; return 1; }; " + " ".join(checks) + "\n"
    )


def runtime_snapshot_exclusions(config: DockerConfig) -> tuple[str, ...]:
    """Prune only host placeholders hidden beneath configured read-only runtime mounts."""
    workspace = PurePosixPath(config.workspace_mount)
    return tuple(
        PurePosixPath(mount.target).relative_to(workspace).as_posix()
        for mount in config.read_only_mounts
        if workspace in PurePosixPath(mount.target).parents
    )


def prepare_runtime_mountpoints(workspace: Path, config: DockerConfig) -> None:
    """Never let a policy redirect a nested runtime mount through a host symlink."""
    for mount in config.read_only_mounts:
        validate_runtime_source(mount.source)
    for relative in runtime_snapshot_exclusions(config):
        parent = workspace
        for part in PurePosixPath(relative).parts:
            parent = parent / part
            if parent.is_symlink():
                raise ValueError("runtime mountpoint cannot traverse a workspace symlink")
            parent.mkdir(exist_ok=True)
            if not parent.is_dir():
                raise ValueError("runtime mountpoint must be a directory")
        if any(parent.iterdir()):
            raise ValueError("runtime mountpoint must be empty; task trees must exclude the venv")


def backend_code_hash() -> str:
    """Record execution routes, task configuration and attack-suite bytes for provenance only."""
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


_PASSED_GATES: set[tuple[int, str]] = set()
_GATE_LOCK = threading.Lock()


def validate_gate_result(config: DockerConfig, record: Mapping[str, object]) -> None:
    """Require complete behavioural evidence for the configured pinned image."""
    if record.get("image") != config.image:
        raise RuntimeError("Docker isolation gate image does not match the configured digest")
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


def require_gate(config: DockerConfig) -> None:
    """Attack current containment once per process and image before Docker episodes start."""
    from scripts.run_docker_jail_tests import (  # noqa: PLC0415 - runner imports backend argv helpers
        run_red_team,
        write_gate_record,
    )

    key = (os.getpid(), config.image)
    with _GATE_LOCK:
        if key in _PASSED_GATES:
            return
        record = run_red_team(config.gate_record, image=config.image)
        validate_gate_result(config, record)
        write_gate_record(config.gate_record, record)
        _PASSED_GATES.add(key)


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


def container_cpuset(name: str, cpus: float) -> str:
    """Pin a container to as many host CPUs as its quota, spread across hosts CPUs by name.

    ``--cpus`` alone leaves every host CPU in the container's affinity mask, so tools that size
    themselves from it (``pytest -n auto`` in coveragepy's ``addopts``) start one worker per host
    CPU inside a 256 MiB container and lose workers to the OOM killer mid-run.
    """
    host_cpus = sorted(os.sched_getaffinity(0))
    cpu_count = math.ceil(cpus)
    if cpu_count > len(host_cpus):
        raise ValueError(f"container needs {cpu_count} CPUs but the host offers {len(host_cpus)}")
    start = int(hashlib.sha256(name.encode()).hexdigest(), 16) % len(host_cpus)
    return ",".join(
        str(host_cpus[(start + offset) % len(host_cpus)]) for offset in range(cpu_count)
    )


def docker_run_args(  # noqa: PLR0913 - fixed profile plus command options
    config: DockerConfig,
    workspace: Path,
    name: str,
    command: str,
    *,
    timeout_seconds: int = 30,
    login_shell: bool = False,
) -> list[str]:
    """Build one mount and fixed containment profile, with extra headroom for trusted tests."""
    workspace = validate_workspace(workspace)
    if timeout_seconds <= 0:
        raise ValueError("Docker command timeout must be positive")
    environment = ["--env", "HOME=/tmp", "--env", "TMPDIR=/tmp"]
    if config.trusted_grader:
        environment.extend(
            value for variable in GRADER_THREAD_LIMITS for value in ("--env", variable)
        )
    extra_mount_args = [
        arg for mount in config.read_only_mounts for arg in ("--mount", runtime_mount_args(mount))
    ]
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
        str(GRADER_PIDS_LIMIT if config.trusted_grader else config.pids_limit),
        "--memory",
        str(config.memory_bytes),
        "--memory-swap",
        str(config.memory_bytes),
        "--cpus",
        str(config.cpus),
        "--cpuset-cpus",
        container_cpuset(name, config.cpus),
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,nodev,size={MAXIMUM_TMP_STATE_BYTES},mode=1777",  # noqa: S108
        "--mount",
        f"type=bind,src={workspace},dst={config.workspace_mount},bind-recursive=disabled",
        *extra_mount_args,
        "--workdir",
        config.workspace_mount,
        *environment,
        "--entrypoint",
        "/usr/bin/timeout",
        config.image,
        "--signal=KILL",
        f"{timeout_seconds}s",
        "/bin/bash",
        "-lc" if login_shell else "-c",
        runtime_mount_guard(config) + command,
    ]


def _relative_path(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("submitted paths must stay inside the workspace")
    return relative


def _validate_junit_report_path(path: str | None) -> str | None:
    """Require a normalized relative path so JUnit output stays inside the mounted workspace."""
    if path is None:
        return None
    relative = PurePosixPath(path)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or str(relative) != path
        or "\\" in path
        or "\0" in path
    ):
        raise ValueError(
            "JUnit report path must be a normalized relative path inside the workspace"
        )
    return path


def _read_junit_report(workspace: Path, report_path: str | None) -> str | None:
    """Read an available, bounded JUnit report without following workspace symlinks."""
    validated_path = _validate_junit_report_path(report_path)
    if validated_path is None:
        return None
    try:
        with _open_regular_file(workspace, validated_path) as report:
            content = report.read(MAXIMUM_JUNIT_XML_BYTES + 1)
    except OSError as error:
        if error.errno not in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EACCES, errno.EPERM}:
            raise
        return None
    except ValueError:
        return None
    if len(content) > MAXIMUM_JUNIT_XML_BYTES:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


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


def _template_and_workspace_paths(template_dir: Path, workspace: Path) -> tuple[Path, Path]:
    """Validate a template and its separate destination tree."""
    if template_dir.is_symlink() or not template_dir.is_dir():
        raise ValueError("workspace template must be a real directory")
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("workspace destination must be a real directory")
    source = template_dir.resolve(strict=True)
    destination = workspace.resolve(strict=True)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("workspace template and destination must be separate trees")
    return source, destination


def restore_workspace_directory_times(template_dir: Path, workspace: Path) -> None:
    """Restore copied template directory times after episode files are materialized."""
    source, destination = _template_and_workspace_paths(template_dir, workspace)
    for source_directory, _, _ in os.walk(source, topdown=False, followlinks=False):
        source_path = Path(source_directory)
        destination_path = destination / source_path.relative_to(source)
        if destination_path.is_symlink() or not destination_path.is_dir():
            raise ValueError("copied template directory is missing or unsafe")
        source_metadata = source_path.stat(follow_symlinks=False)
        os.utime(
            destination_path,
            ns=(source_metadata.st_atime_ns, source_metadata.st_mtime_ns),
            follow_symlinks=False,
        )


def copy_workspace_template(template_dir: Path, workspace: Path) -> None:
    """Copy a trusted template into an empty workspace while retaining source mtimes."""
    source, destination = _template_and_workspace_paths(template_dir, workspace)
    if any(destination.iterdir()):
        raise ValueError("workspace template destination must be empty")

    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        symlinks=True,
        copy_function=shutil.copy2,
    )
    restore_workspace_directory_times(source, destination)


def _workspace_paths(workspace: Path, excluded_paths: tuple[str, ...] = ()) -> Generator[Path]:
    pending = [workspace]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                path = Path(entry.path)
                if path.relative_to(workspace).as_posix() in excluded_paths:
                    if entry.is_symlink():
                        raise ValueError("excluded runtime mountpoint must not be a symlink")
                    continue
                yield path
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))


def workspace_snapshot(
    workspace: Path, *, excluded_paths: tuple[str, ...] = ()
) -> dict[str, dict[str, str]]:
    """Record bounded content hashes, permissions and links without following external targets."""
    snapshot: dict[str, dict[str, str]] = {}
    started = time.monotonic()
    bytes_hashed = 0
    for path in _workspace_paths(workspace, excluded_paths):
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


def _check_tmp_state_limits(state: Path) -> None:
    snapshot = workspace_snapshot(state)
    if sum(int(entry.get("size", 0)) for entry in snapshot.values()) > MAXIMUM_TMP_STATE_BYTES:
        raise SnapshotLimitError("episode /tmp state exceeded the container tmpfs byte limit")


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


def _inspect_image_volumes(
    image: str, *, command: str
) -> tuple[object | None, CommandResult | None]:
    """Inspect image volume metadata, returning exit 125 when Docker cannot inspect the image."""
    inspected = subprocess.run(  # noqa: S603 - trusted digest-pinned image metadata
        ["docker", "image", "inspect", "--format", "{{json .Config.Volumes}}", image],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if inspected.returncode != 0:
        logger.warning("Docker image inspection failed for %s: %s", image, inspected.stderr.strip())
        return None, CommandResult(
            command,
            125,
            inspected.stdout,
            inspected.stderr,
            timed_out=False,
        )
    return json.loads(inspected.stdout), None


def _validate_tmp_state(workspace: Path, tmp_state: Path | None) -> None:
    if tmp_state is not None and (
        tmp_state.is_symlink()
        or not tmp_state.is_dir()
        or workspace in tmp_state.resolve(strict=True).parents
        or tmp_state.resolve(strict=True) == workspace
    ):
        raise ValueError("episode /tmp state must be a separate real directory")


def _validate_protected_paths(config: DockerConfig, workspace: Path) -> None:
    for protected in (config.gate_record, config.evidence_path):
        if protected is not None and (
            protected.resolve() == workspace or workspace in protected.resolve().parents
        ):
            raise ValueError("gate and evidence must stay outside the episode mount")


def _tmp_persisting_command(
    command: str, *, config: DockerConfig, staged_tmp: Path, seconds: int, login_shell: bool
) -> str:
    container_path = shlex.quote(f"{config.workspace_mount}/{staged_tmp.name}")
    inner_shell = "-lc" if login_shell else "-c"
    return (
        "shopt -s dotglob nullglob; "
        f"_tmp_entries=({container_path}/*); "
        'if ((${#_tmp_entries[@]})); then /bin/cp -a -- "${_tmp_entries[@]}" /tmp/ || exit 125; fi; '
        f"/bin/rm -rf -- {container_path}; "
        f"/usr/bin/timeout --signal=KILL {seconds}s /bin/bash {inner_shell} {shlex.quote(command)}; "
        "_episode_exit=$?; "
        f"/bin/mkdir -- {container_path} || exit 125; "
        "_tmp_entries=(/tmp/*); "
        'if ((${#_tmp_entries[@]})); then /bin/cp -a -- "${_tmp_entries[@]}" '
        f"{container_path}/ || exit 125; fi; "
        'exit "$_episode_exit"'
    )


def _stage_tmp_state(tmp_state: Path | None, staged_tmp: Path | None) -> None:
    if tmp_state is None or staged_tmp is None:
        return
    _check_tmp_state_limits(tmp_state)
    try:
        shutil.copytree(tmp_state, staged_tmp, symlinks=True)
    except OSError:
        if staged_tmp.exists():
            shutil.rmtree(staged_tmp)
        raise


def _collect_tmp_state(tmp_state: Path | None, staged_tmp: Path | None) -> None:
    if tmp_state is None or staged_tmp is None:
        return
    if staged_tmp.is_symlink():
        staged_tmp.unlink()
        raise ValueError("container replaced episode /tmp staging with a symlink")
    if staged_tmp.is_dir():
        try:
            _check_tmp_state_limits(staged_tmp)
            shutil.rmtree(tmp_state)
            staged_tmp.rename(tmp_state)
        finally:
            if staged_tmp.exists():
                shutil.rmtree(staged_tmp)


def run_in_docker(  # noqa: PLR0913 - report capture is an optional transport argument
    episode_dir: Path,
    command: str,
    *,
    config: DockerConfig,
    timeout: str = "30s",
    login_shell: bool = True,
    junit_report_path: str | None = None,
) -> CommandResult:
    """Run a fresh container and persist command evidence on the host, outside its mount."""
    workspace = validate_workspace(episode_dir)
    tmp_state = config.episode_tmp_state
    _validate_tmp_state(workspace, tmp_state)
    junit_report_path = _validate_junit_report_path(junit_report_path)
    _validate_protected_paths(config, workspace)
    require_gate(config)
    prepare_runtime_mountpoints(workspace, config)
    image_volumes, image_inspection_failure = _inspect_image_volumes(
        config.image,
        command=command,
    )
    if image_inspection_failure is not None:
        return image_inspection_failure
    if image_volumes not in (None, {}):
        raise ValueError("Docker episode images must not declare automatic writable volumes")
    evidence_path = config.evidence_path or workspace.with_name(
        workspace.name + ".docker-evidence.jsonl"
    )
    name = "episode-" + uuid.uuid4().hex
    seconds = _timeout_seconds(timeout)
    staged_tmp = workspace / f".episode-tmp-{uuid.uuid4().hex}" if tmp_state is not None else None
    outer_seconds = seconds + 10 if staged_tmp is not None else seconds
    argv = docker_run_args(
        config,
        workspace,
        name,
        _tmp_persisting_command(
            command,
            config=config,
            staged_tmp=staged_tmp,
            seconds=seconds,
            login_shell=login_shell,
        )
        if staged_tmp is not None
        else command,
        timeout_seconds=outer_seconds,
        login_shell=False if staged_tmp is not None else login_shell,
    )
    evidence: dict[str, object] = {
        "command": command,
        "container": name,
        "image": config.image,
        "backend_hash": backend_code_hash(),
        "read_only_mounts": [
            {"source": str(mount.source), "target": mount.target}
            for mount in config.read_only_mounts
        ],
        "phase": "started",
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        evidence["before"] = workspace_snapshot(
            workspace, excluded_paths=runtime_snapshot_exclusions(config)
        )
    except (OSError, SnapshotLimitError) as error:
        evidence.update({"phase": "refused", "snapshot_error": str(error)})
        with evidence_path.open("a") as output:
            output.write(json.dumps(evidence) + "\n")
        raise
    with evidence_path.open("a") as output:
        output.write(json.dumps(evidence) + "\n")
    _stage_tmp_state(tmp_state, staged_tmp)
    try:
        exit_code, stdout, stderr, host_timed_out, output_truncated = _run_bounded_docker(
            argv, outer_seconds + 15
        )
        junit_xml = _read_junit_report(workspace, junit_report_path)
        result = CommandResult(
            command,
            exit_code,
            stdout,
            stderr,
            host_timed_out or exit_code in (124, 137),
            junit_xml=junit_xml,
        )
    finally:
        cleanup = subprocess.run(  # noqa: S603 - generated container name
            ["docker", "rm", "--force", "--volumes", name],  # noqa: S607 - installed host Docker CLI
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if cleanup.returncode != 0 and "No such container" not in cleanup.stderr:
            raise RuntimeError(f"Docker container cleanup failed: {cleanup.stderr.strip()}")
        _collect_tmp_state(tmp_state, staged_tmp)
    evidence.update(
        {
            "phase": "finished",
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "output_truncated": output_truncated,
            "junit_xml": result.junit_xml,
        }
    )
    try:
        evidence["after"] = workspace_snapshot(
            workspace, excluded_paths=runtime_snapshot_exclusions(config)
        )
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
    template_dir: Path | None = None,
    hidden_tests_template: Path | None = None,
    junit_report_path: str | None = None,
    artifact_contents: Mapping[str, bytes] | None = None,
) -> CommandResult:
    """Grade a copied submission with trusted baseline files in a separate fresh container."""
    junit_report_path = _validate_junit_report_path(junit_report_path)
    evidence = config.evidence_path or episode_dir.with_name(
        episode_dir.name + ".docker-evidence.jsonl"
    )
    from_config = DockerConfig(
        image=config.image,
        gate_record=config.gate_record,
        memory_bytes=config.memory_bytes,
        pids_limit=config.pids_limit,
        cpus=config.cpus,
        uid=config.uid,
        gid=config.gid,
        evidence_path=evidence,
        workspace_mount=config.workspace_mount,
        read_only_mounts=config.read_only_mounts,
        trusted_grader=True,
    )
    with tempfile.TemporaryDirectory(prefix="docker-grade-") as temporary:
        workspace = Path(temporary)
        stage_grader_workspace(
            episode_dir,
            task_files,
            artifact_paths,
            workspace,
            template_dir=template_dir,
            hidden_tests_template=hidden_tests_template,
            artifact_contents=artifact_contents,
        )
        return run_in_docker(
            workspace,
            command,
            config=from_config,
            timeout=timeout,
            login_shell=False,
            junit_report_path=junit_report_path,
        )


def stage_grader_workspace(  # noqa: PLR0913 - trusted templates and submitted artifacts are separate inputs
    episode_dir: Path,
    task_files: Mapping[str, str],
    artifact_paths: tuple[str, ...],
    workspace: Path,
    *,
    template_dir: Path | None = None,
    hidden_tests_template: Path | None = None,
    artifact_contents: Mapping[str, bytes] | None = None,
) -> None:
    """Reconstruct trusted grading inputs; never import agent state beyond the submission."""
    if any(path in task_files for path in artifact_paths):
        raise ValueError("submitted artifacts cannot replace trusted task files")
    if any(workspace.iterdir()):
        raise ValueError("trusted grader workspace must start empty")
    if template_dir is not None:
        copy_workspace_template(template_dir, workspace)
    if hidden_tests_template is not None:
        hidden_destination = workspace / "r2e_tests"
        if hidden_destination.exists() or hidden_destination.is_symlink():
            raise ValueError("closed template must not contain hidden tests")
        hidden_destination.mkdir()
        copy_workspace_template(hidden_tests_template, hidden_destination)
    explicit_artifacts = dict(artifact_contents or {})
    if set(explicit_artifacts) - set(artifact_paths):
        raise ValueError("explicit submitted artifacts must be named in artifact_paths")
    if any(len(content) > MAXIMUM_ARTIFACT_BYTES for content in explicit_artifacts.values()):
        raise ValueError("submitted artifact exceeds the 64 MiB grading transfer limit")
    artifacts = submitted_files(
        episode_dir, tuple(path for path in artifact_paths if path not in explicit_artifacts)
    )
    artifacts.update(explicit_artifacts)
    for relative, content in task_files.items():
        _write_staged_file(workspace, relative, content.encode())
    for relative, content in artifacts.items():
        _write_staged_file(workspace, relative, content)


def _write_staged_file(workspace: Path, relative: str, content: bytes) -> None:
    """Write trusted staging inputs without traversing template symlinks."""
    parts = _relative_path(relative).parts
    parent = workspace
    for part in parts[:-1]:
        parent = parent / part
        if parent.is_symlink():
            raise ValueError("trusted grader paths cannot traverse template symlinks")
        parent.mkdir(exist_ok=True)
        if not parent.is_dir():
            raise ValueError("trusted grader paths must use directories")
    target = parent / parts[-1]
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("trusted grader files cannot replace symlinks or directories")
    target.write_bytes(content)
