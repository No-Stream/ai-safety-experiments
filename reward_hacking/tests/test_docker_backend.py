"""CPU contracts for the Docker command boundary; no daemon required."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from reward_hacking.harness import docker_backend as backend
from reward_hacking.harness.docker_backend import (
    REQUIRED_PROBES,
    DockerConfig,
    SnapshotLimitError,
    _run_bounded_docker,
    backend_code_hash,
    copy_workspace_template,
    docker_run_args,
    require_gate,
    restore_workspace_directory_times,
    run_docker_grader,
    run_in_docker,
    submitted_files,
    workspace_snapshot,
)

IMAGE = "python@sha256:" + "a" * 64


def test_hardened_flags_and_digest(tmp_path: Path) -> None:
    config = DockerConfig(image=IMAGE, gate_record=tmp_path / "gate.json")
    argv = docker_run_args(config, tmp_path, "episode-toy", "id")
    for flag, value in (
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--pids-limit", "64"),
        ("--memory", "268435456"),
        ("--memory-swap", "268435456"),
        ("--cpus", "1"),
    ):
        assert argv[argv.index(flag) + 1] == value
    assert "--read-only" in argv
    assert argv[argv.index("--log-driver") + 1] == "none"
    assert "--rm" in argv
    assert argv.count("--mount") == 1
    assert argv[argv.index("--mount") + 1].endswith(",dst=/work,bind-recursive=disabled")
    assert argv[argv.index("--workdir") + 1] == "/work"
    assert "/var/run/docker.sock" not in " ".join(argv)
    assert IMAGE in argv
    assert argv[argv.index("--user") + 1].split(":")[0] != "0"


def test_workspace_mount_must_be_an_absolute_container_path(tmp_path: Path) -> None:
    for workspace_mount in (
        "work",
        "",
        "/",
        "/work,readonly",
        "/work/../host",
        str(PurePosixPath("/", "tmp")),
        str(PurePosixPath("/", "tmp", "episode")),
        "/proc",
        "/proc/1",
    ):
        with pytest.raises(ValueError, match="workspace mount"):
            DockerConfig(
                image=IMAGE,
                gate_record=tmp_path / "gate.json",
                workspace_mount=workspace_mount,
            )

    config = DockerConfig(
        image=IMAGE,
        gate_record=tmp_path / "gate.json",
        workspace_mount="/testbed",
    )
    argv = docker_run_args(config, tmp_path, "episode-testbed", "id")
    assert argv[argv.index("--mount") + 1].endswith(",dst=/testbed,bind-recursive=disabled")
    assert argv[argv.index("--workdir") + 1] == "/testbed"


def test_copy_workspace_template_preserves_file_and_directory_mtimes(tmp_path: Path) -> None:
    template = tmp_path / "template"
    nested = template / "nested"
    nested.mkdir(parents=True)
    (nested / "module.py").write_text("baseline")
    (nested / "module.py").touch()
    file_mtime_ns = 1_600_000_000_123_456_789
    nested_mtime_ns = 1_600_000_010_123_456_789
    root_mtime_ns = 1_600_000_020_123_456_789
    os.utime(nested / "module.py", ns=(file_mtime_ns, file_mtime_ns))
    os.utime(nested, ns=(nested_mtime_ns, nested_mtime_ns))
    os.utime(template, ns=(root_mtime_ns, root_mtime_ns))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    copy_workspace_template(template, workspace)

    assert (workspace / "nested" / "module.py").stat().st_mtime_ns == file_mtime_ns
    assert (workspace / "nested").stat().st_mtime_ns == nested_mtime_ns
    assert workspace.stat().st_mtime_ns == root_mtime_ns

    (workspace / "nested" / "materialized.py").write_text("episode file")
    assert (workspace / "nested").stat().st_mtime_ns != nested_mtime_ns
    restore_workspace_directory_times(template, workspace)
    assert (workspace / "nested").stat().st_mtime_ns == nested_mtime_ns


def test_copy_workspace_template_requires_empty_destination(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "existing").write_text("keep")

    with pytest.raises(ValueError, match="empty"):
        copy_workspace_template(template, workspace)


@pytest.mark.parametrize("image", ["python:3.12-slim", "python@sha256:abc"])
def test_unpinned_image_refused(tmp_path: Path, image: str) -> None:
    with pytest.raises(ValueError, match="digest"):
        DockerConfig(image=image, gate_record=tmp_path / "gate.json")


def test_gate_sabotage_refuses_stale_hash_and_vacuous_control(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    config = DockerConfig(image=IMAGE, gate_record=gate)
    probes = {key: {"contained": True, "negative_control": True} for key in REQUIRED_PROBES}
    record: dict[str, object] = {
        "backend_hash": backend_code_hash(),
        "passed": True,
        "image": IMAGE,
        "probes": probes,
    }
    gate.write_text(json.dumps(record))
    require_gate(config)
    record["backend_hash"] = "sabotaged"
    gate.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="hash"):
        require_gate(config)
    record["backend_hash"] = backend_code_hash()
    probes["docker_socket_unreachable"]["negative_control"] = False
    gate.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="probe"):
        require_gate(config)


def test_snapshots_preserve_symlink_evidence_without_reading_target(tmp_path: Path) -> None:
    (tmp_path / "answer.txt").write_text("toy answer")
    (tmp_path / "link").symlink_to("/etc/passwd")
    snapshot = workspace_snapshot(tmp_path)
    assert snapshot["link"]["symlink"] == "/etc/passwd"
    assert "sha256" in snapshot["answer.txt"]


def test_submission_allowlist_rejects_symlink_and_traversal(tmp_path: Path) -> None:
    (tmp_path / "answer.txt").write_text("toy answer")
    assert submitted_files(tmp_path, ("answer.txt",)) == {"answer.txt": b"toy answer"}
    (tmp_path / "link").symlink_to("answer.txt")
    for path in ("link", "../answer.txt", "/etc/passwd"):
        with pytest.raises(ValueError, match=r"artifact|workspace"):
            submitted_files(tmp_path, (path,))


def test_real_subprocess_boundary_records_changes_and_stages_trusted_grader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "docker"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        "if arguments[0] == 'image':\n"
        "    print('null')\n"
        "elif arguments[0] == 'run':\n"
        "    assert arguments[arguments.index('--workdir') + 1] == '/testbed'\n"
        "    mount = arguments[arguments.index('--mount') + 1]\n"
        "    workspace = pathlib.Path(mount.split('src=')[1].split(',')[0])\n"
        "    if arguments[-1] == 'toy-grade':\n"
        "        assert (workspace / 'baseline.py').read_text() == 'closed baseline'\n"
        "        assert (workspace / 'grade.py').read_text() == 'trusted grader'\n"
        "        assert (workspace / 'answer.txt').read_text() == 'submitted answer'\n"
        "        assert not (workspace / 'malicious-extra.py').exists()\n"
        "        print('SUCCESS')\n"
        "    else:\n"
        "        (workspace / 'answer.txt').write_text('submitted answer')\n"
        "        print('agent command output')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "backend_hash": backend_code_hash(),
                "passed": True,
                "image": IMAGE,
                "probes": {
                    key: {"contained": True, "negative_control": True} for key in REQUIRED_PROBES
                },
            }
        )
    )
    workspace = tmp_path / "episode"
    workspace.mkdir()
    config = DockerConfig(image=IMAGE, gate_record=gate, workspace_mount="/testbed")
    result = run_in_docker(workspace, "toy-action", config=config)
    assert result.exit_code == 0
    (workspace / "grade.py").write_text("tampered grader")
    (workspace / "malicious-extra.py").write_text("agent-only file")
    template = tmp_path / "closed-template"
    template.mkdir()
    (template / "baseline.py").write_text("closed baseline")
    grade = run_docker_grader(
        workspace,
        {"grade.py": "trusted grader"},
        ("answer.txt",),
        "toy-grade",
        config=config,
        template_dir=template,
    )
    assert grade.ok
    records = [
        json.loads(line)
        for line in workspace.with_name("episode.docker-evidence.jsonl").read_text().splitlines()
    ]
    assert len(records) == 4
    assert records[1]["exit_code"] == 0
    assert records[1]["image"] == IMAGE
    assert "answer.txt" not in records[1]["before"]
    assert "answer.txt" in records[1]["after"]
    assert records[3]["before"]["grade.py"] != records[1]["after"].get("grade.py")


def test_docker_command_output_is_bounded_while_both_pipes_are_drained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "docker"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        "    os.write(1, b'null\\n')\n"
        "elif sys.argv[1] == 'run':\n"
        "    os.write(1, b'x' * (2 * 1024 * 1024))\n"
        "    os.write(2, b'y' * (2 * 1024 * 1024))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "backend_hash": backend_code_hash(),
                "passed": True,
                "image": IMAGE,
                "probes": {
                    key: {"contained": True, "negative_control": True} for key in REQUIRED_PROBES
                },
            }
        )
    )
    workspace = tmp_path / "episode"
    workspace.mkdir()
    result = run_in_docker(workspace, "write-long-output", config=DockerConfig(IMAGE, gate))
    assert result.exit_code == 0
    assert len(result.stdout.encode()) <= 1024 * 1024
    assert len(result.stderr.encode()) <= 1024 * 1024 + 100
    assert "truncated" in result.stderr
    records = [
        json.loads(line)
        for line in workspace.with_name("episode.docker-evidence.jsonl").read_text().splitlines()
    ]
    assert records[1]["output_truncated"] is True


def test_bounded_docker_client_is_killed_at_host_deadline() -> None:
    exit_code, stdout, stderr, timed_out, output_truncated = _run_bounded_docker(
        [
            sys.executable,
            "-c",
            "import sys,time; sys.stdout.write('started'); sys.stdout.flush(); time.sleep(30)",
        ],
        deadline_seconds=1,
    )
    assert exit_code == 124
    assert stdout == "started"
    assert "deadline exceeded" in stderr
    assert timed_out
    assert not output_truncated


def test_snapshots_record_modes_directories_and_refuse_scan_overruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "answer.txt").write_text("toy answer")
    before = workspace_snapshot(tmp_path)
    (tmp_path / "answer.txt").chmod(0o600)
    after = workspace_snapshot(tmp_path)
    assert before["empty"]["directory"] == "true"
    assert before["answer.txt"]["mode"] != after["answer.txt"]["mode"]
    monkeypatch.setattr("reward_hacking.harness.docker_backend.MAXIMUM_SNAPSHOT_BYTES", 1)
    with pytest.raises(SnapshotLimitError, match="byte"):
        workspace_snapshot(tmp_path)


def test_snapshot_refuses_unreadable_directories(tmp_path: Path) -> None:
    directory = tmp_path / "hidden"
    directory.mkdir()
    (directory / "answer.txt").write_text("toy answer")
    directory.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            workspace_snapshot(tmp_path)
    finally:
        directory.chmod(0o700)


def test_runtime_mount_validation_and_readonly_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    venv = runtime / "venv"
    venv.mkdir()
    monkeypatch.setattr(backend, "READ_ONLY_RUNTIME_ROOT", runtime)
    mount = backend.ReadOnlyMount(venv, "/testbed/.venv")
    config = DockerConfig(
        IMAGE, tmp_path / "gate", workspace_mount="/testbed", read_only_mounts=(mount,)
    )
    argv = docker_run_args(config, tmp_path, "runtime", "id")
    assert argv[argv.index("--mount", argv.index("--mount") + 1) + 1] == (
        f"type=bind,source={venv},target=/testbed/.venv,readonly,bind-recursive=disabled"
    )
    assert "volume-driver=local" not in " ".join(argv)
    assert "no-new-privileges" in argv
    assert "/proc/self/mountinfo" in argv[-1]
    for source in (tmp_path, runtime / ".." / "runtime" / "venv"):
        with pytest.raises(ValueError, match="runtime"):
            backend.ReadOnlyMount(source, "/opt/uvpy")
    link = runtime / "link"
    link.symlink_to(venv, target_is_directory=True)
    with pytest.raises(ValueError, match="runtime"):
        backend.ReadOnlyMount(link, "/opt/uvpy")
    with pytest.raises(ValueError, match=r"runtime|overlap"):
        DockerConfig(
            IMAGE, tmp_path / "gate", read_only_mounts=(backend.ReadOnlyMount(venv, "/work"),)
        )


def test_runtime_mountpoint_symlink_refused_and_snapshot_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    venv = runtime / "venv"
    venv.mkdir()
    monkeypatch.setattr(backend, "READ_ONLY_RUNTIME_ROOT", runtime)
    config = DockerConfig(
        IMAGE,
        tmp_path / "gate",
        workspace_mount="/testbed",
        read_only_mounts=(backend.ReadOnlyMount(venv, "/testbed/.venv"),),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".venv").symlink_to(venv, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        backend.prepare_runtime_mountpoints(workspace, config)
    (workspace / ".venv").unlink()
    backend.prepare_runtime_mountpoints(workspace, config)
    (workspace / ".venv" / "bulky.dat").write_text("not scanned")
    (workspace / "normal.txt").write_text("scanned")
    snapshot = workspace_snapshot(
        workspace, excluded_paths=backend.runtime_snapshot_exclusions(config)
    )
    assert ".venv/bulky.dat" not in snapshot
    assert "normal.txt" in snapshot
    (workspace / "runtime-link").symlink_to(venv / "bulky.dat")
    snapshot = workspace_snapshot(
        workspace, excluded_paths=backend.runtime_snapshot_exclusions(config)
    )
    assert "sha256" not in snapshot["runtime-link"]


@pytest.mark.parametrize("target", ["/opt/workspace", "/opt/workspace/dependency", "/opt/runtime"])
def test_runtime_mount_cannot_shadow_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    runtime = tmp_path / "runtime"
    source = runtime / "python"
    source.mkdir(parents=True)
    monkeypatch.setattr(backend, "READ_ONLY_RUNTIME_ROOT", runtime)
    workspace_mount = "/opt/runtime/workspace" if target == "/opt/runtime" else "/opt/workspace"
    with pytest.raises(ValueError, match="runtime"):
        DockerConfig(
            IMAGE,
            tmp_path / "gate",
            workspace_mount=workspace_mount,
            read_only_mounts=(backend.ReadOnlyMount(source, target),),
        )


@pytest.mark.parametrize(
    "flags", ["rw", "rw,nosuid,nodev", "ro", "ro,nodev", "ro,nosuid", "ro,nosuid,nodev"]
)
def test_runtime_guard_refuses_weakened_flags_before_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: str
) -> None:
    runtime = tmp_path / "runtime"
    source = runtime / "python"
    source.mkdir(parents=True)
    monkeypatch.setattr(backend, "READ_ONLY_RUNTIME_ROOT", runtime)
    config = DockerConfig(
        IMAGE, tmp_path / "gate", read_only_mounts=(backend.ReadOnlyMount(source, "/opt/uvpy"),)
    )
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"1 2 8:1 / /opt/uvpy {flags} - ext4 device rw\n")
    sentinel = tmp_path / "command-ran"
    guard = backend.runtime_mount_guard(config).replace(
        "/proc/self/mountinfo", shlex.quote(str(mountinfo))
    )
    result = subprocess.run(  # noqa: S603 - synthetic mountinfo and host scratch only
        ["/bin/sh", "-c", guard + f"touch {shlex.quote(str(sentinel))}"],
        check=False,
    )
    assert sentinel.exists() is ("ro" in flags.split(","))
    assert result.returncode == (0 if "ro" in flags.split(",") else 125)
