"""Pure checks for the Docker red-team runner's pinned-image and evidence helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reward_hacking.harness import docker_backend
from reward_hacking.harness.docker_backend import DockerConfig, ReadOnlyMount, runtime_mount_args
from scripts import run_docker_jail_tests as docker_jail_runner
from scripts.run_docker_jail_tests import (
    REQUIRED_PROBES,
    ContainerResult,
    DockerProbeRunner,
    ProbeResult,
    _gateway_server_source,
    _grader_source,
    _host_path_source,
    _memory_limit_source,
    _network_probe_source,
    _outside_write_source,
    _privilege_source,
    _process_limit_source,
    _run_runtime_mount_host_escape_probe,
    _run_runtime_mount_write_probe,
    _runtime_host_escape_source,
    _runtime_mount_write_source,
    _socket_source,
    _weaken_runtime_mount_args,
    build_gate_record,
    insert_before_image,
    parse_image_digest,
    parse_pulled_digest,
    replace_option_value,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def test_parse_image_digest_pins_the_digest_docker_pull_reported() -> None:
    digest = "0123456789abcdef" * 4
    republished = "fedcba9876543210" * 4
    assert (
        parse_image_digest(f'["python@sha256:{digest}"]', f"sha256:{digest}")
        == f"python@sha256:{digest}"
    )
    both = f'["python@sha256:{digest}", "python@sha256:{republished}"]'
    assert parse_image_digest(both, f"sha256:{republished}") == f"python@sha256:{republished}"

    with pytest.raises(ValueError, match="digest"):
        parse_image_digest(f'["python@sha256:{digest}"]', "sha256:abc")
    with pytest.raises(ValueError, match="not among"):
        parse_image_digest(f'["other@sha256:{digest}"]', f"sha256:{digest}")


def test_parse_pulled_digest_reads_the_single_digest_line() -> None:
    digest = "0123456789abcdef" * 4
    output = f"3.12-slim: Pulling from library/python\nDigest: sha256:{digest}\nStatus: Image is up to date\n"
    assert parse_pulled_digest(output) == f"sha256:{digest}"
    with pytest.raises(ValueError, match="exactly one digest"):
        parse_pulled_digest("Status: Image is up to date\n")


def test_negative_control_argv_rewrites_only_the_requested_docker_option() -> None:
    arguments = [
        "docker",
        "run",
        "--network",
        "none",
        "--memory",
        "268435456",
        "python@sha256:abc",
        "true",
    ]

    weakened = replace_option_value(arguments, "--network", "none", "bridge")

    assert weakened == [
        "docker",
        "run",
        "--network",
        "bridge",
        "--memory",
        "268435456",
        "python@sha256:abc",
        "true",
    ]
    assert arguments[3] == "none"


def test_negative_control_can_replace_one_of_several_mount_options() -> None:
    arguments = [
        "docker",
        "run",
        "--mount",
        "type=bind,src=/workspace,dst=/work",
        "--mount",
        "type=volume,dst=/opt/uvpy,readonly",
        "python@sha256:abc",
        "true",
    ]

    weakened = replace_option_value(
        arguments,
        "--mount",
        "type=volume,dst=/opt/uvpy,readonly",
        "type=volume,dst=/opt/uvpy",
    )

    assert weakened[3] == arguments[3]
    assert weakened[5] == "type=volume,dst=/opt/uvpy"


def test_extra_control_mount_is_inserted_before_the_pinned_image() -> None:
    arguments = ["docker", "run", "--rm", "--network", "none", "python@sha256:abc", "true"]

    mounted = insert_before_image(
        arguments, "python@sha256:abc", ["--mount", "type=bind,src=/control,dst=/control"]
    )

    assert mounted == [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--mount",
        "type=bind,src=/control,dst=/control",
        "python@sha256:abc",
        "true",
    ]


def test_gate_record_names_each_probe_and_records_digest() -> None:
    digest = "0123456789abcdef" * 4
    results = {
        probe: ProbeResult(contained=True, negative_control=True) for probe in REQUIRED_PROBES
    }
    record = build_gate_record("abc123", f"python@sha256:{digest}", results)

    assert record["backend_hash"] == "abc123"
    assert record["image"] == f"python@sha256:{digest}"
    assert record["passed"] is True
    probes = record["probes"]
    assert isinstance(probes, dict)
    assert all(
        evidence == {"contained": True, "negative_control": True} for evidence in probes.values()
    )
    assert json.loads(json.dumps(record)) == record


def test_gate_requires_runtime_mount_write_and_host_escape_probes() -> None:
    assert {
        "runtime_mount_write_blocked",
        "runtime_mount_host_escape_blocked",
    } <= REQUIRED_PROBES


def test_runtime_mount_probe_programs_compile() -> None:
    compile(_runtime_mount_write_source(), "runtime-mount-write-probe", "exec")
    compile(_runtime_host_escape_source(), "runtime-host-escape-probe", "exec")


def test_temporary_runtime_mount_sources_live_under_the_allowlisted_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docker_backend, "READ_ONLY_RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(docker_jail_runner, "READ_ONLY_RUNTIME_ROOT", tmp_path)

    with docker_jail_runner._temporary_runtime_mount_sources() as (
        runtime_mount,
        host_source,
        host_sentinel,
    ):
        runtime_source = runtime_mount.source
        assert runtime_source.parent == host_source.parent
        assert runtime_mount.target == "/opt/uvpy"
        assert (runtime_source / "runtime-sentinel").read_text() == "trusted-runtime"
        assert host_sentinel.read_text() == "synthetic-host-sentinel"
        assert runtime_source.stat().st_mode & 0o777 == 0o777
        assert (runtime_source / "runtime-sentinel").stat().st_mode & 0o777 == 0o666

    assert not runtime_source.exists()
    assert not host_source.exists()


def test_runtime_mount_negative_control_removes_only_readonly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docker_backend, "READ_ONLY_RUNTIME_ROOT", tmp_path)
    runtime_source = tmp_path / "runtime-source"
    runtime_source.mkdir()
    runtime_mount = ReadOnlyMount(source=runtime_source, target="/opt/uvpy")
    mount_args = runtime_mount_args(runtime_mount)

    weakened_args = _weaken_runtime_mount_args(mount_args)

    assert ",readonly," in mount_args
    assert ",readonly," not in weakened_args
    assert weakened_args == mount_args.replace(",readonly", "")
    assert "bind-recursive=disabled" in weakened_args


class FakeDockerProbeRunner(DockerProbeRunner):
    def __init__(
        self,
        config: DockerConfig,
        workspace: Path,
        hardened_output: str,
        control_output: str,
    ) -> None:
        super().__init__(config, workspace)
        self.hardened_output = hardened_output
        self.control_output = control_output
        self.control_argv: list[str] = []

    def run(self, command: str, *, timeout_seconds: int = 30) -> ContainerResult:
        return ContainerResult(0, self.hardened_output, "")

    def run_argv(
        self, argv: Sequence[str], name: str, timeout_seconds: int = 30
    ) -> ContainerResult:
        self.control_argv = list(argv)
        return ContainerResult(0, self.control_output, "")


def test_fake_runner_checks_runtime_flags_and_direct_and_symlink_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docker_backend, "READ_ONLY_RUNTIME_ROOT", tmp_path)
    runtime_source = tmp_path / "runtime-source"
    runtime_source.mkdir()
    runtime_sentinel = runtime_source / "runtime-sentinel"
    runtime_sentinel.write_text("trusted")
    runtime_mount = ReadOnlyMount(source=runtime_source, target="/opt/uvpy")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = DockerConfig(
        image="python@sha256:" + "a" * 64,
        gate_record=tmp_path / "gate.json",
        read_only_mounts=(runtime_mount,),
    )
    runner = FakeDockerProbeRunner(
        config,
        workspace,
        "flags:ok;direct:blocked;symlink:blocked",
        "flags:missing;direct:written;symlink:written",
    )

    result = _run_runtime_mount_write_probe(runner, runtime_mount, workspace)

    assert result == ProbeResult(contained=True, negative_control=True)
    assert runner.control_argv[-1].startswith("python3 -c ")
    runtime_mount_values = [
        runner.control_argv[index + 1]
        for index, value in enumerate(runner.control_argv[:-1])
        if value == "--mount"
        and f"target={runtime_mount.target}," in runner.control_argv[index + 1]
    ]
    assert runtime_mount_values == [_weaken_runtime_mount_args(runtime_mount_args(runtime_mount))]


def test_fake_runner_rejects_a_writable_or_misflagged_runtime_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docker_backend, "READ_ONLY_RUNTIME_ROOT", tmp_path)
    runtime_source = tmp_path / "runtime-source"
    runtime_source.mkdir()
    (runtime_source / "runtime-sentinel").write_text("trusted")
    runtime_mount = ReadOnlyMount(source=runtime_source, target="/opt/uvpy")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = DockerConfig(
        image="python@sha256:" + "a" * 64,
        gate_record=tmp_path / "gate.json",
        read_only_mounts=(runtime_mount,),
    )
    runner = FakeDockerProbeRunner(
        config,
        workspace,
        "flags:missing;direct:written;symlink:written",
        "flags:missing;direct:written;symlink:written",
    )

    with pytest.raises(RuntimeError, match="runtime_mount_write_blocked hardened run"):
        _run_runtime_mount_write_probe(runner, runtime_mount, workspace)


def test_fake_runner_blocks_host_path_symlink_until_control_mounts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docker_backend, "READ_ONLY_RUNTIME_ROOT", tmp_path)
    runtime_source = tmp_path / "runtime-source"
    runtime_source.mkdir()
    runtime_mount = ReadOnlyMount(source=runtime_source, target="/opt/uvpy")
    host_source = tmp_path / "host-only"
    host_source.mkdir()
    host_sentinel = host_source / "host-sentinel"
    host_sentinel.write_text("synthetic-host-sentinel")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = DockerConfig(
        image="python@sha256:" + "a" * 64,
        gate_record=tmp_path / "gate.json",
        read_only_mounts=(runtime_mount,),
    )
    runner = FakeDockerProbeRunner(
        config,
        workspace,
        "traversal:blocked;absolute:blocked",
        "traversal:read:synthetic-host-sentinel;absolute:blocked",
    )

    result = _run_runtime_mount_host_escape_probe(runner, runtime_mount, host_sentinel, host_source)

    assert result == ProbeResult(contained=True, negative_control=True)
    assert runner.control_argv[-1].startswith("python3 -c ")
    mount_values = [
        runner.control_argv[index + 1]
        for index, value in enumerate(runner.control_argv[:-1])
        if value == "--mount"
    ]
    assert f"source={host_source}" in mount_values[-1]
    assert "target=/opt/leak-runtime-host-control-" in mount_values[-1]
    assert str((workspace / "runtime-parent-link").readlink()).startswith("/opt/uvpy/../")
    assert (workspace / "host-absolute-link").readlink() == host_sentinel


def test_fake_runner_rejects_host_path_symlink_read_without_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docker_backend, "READ_ONLY_RUNTIME_ROOT", tmp_path)
    runtime_source = tmp_path / "runtime-source"
    runtime_source.mkdir()
    runtime_mount = ReadOnlyMount(source=runtime_source, target="/opt/uvpy")
    host_source = tmp_path / "host-only"
    host_source.mkdir()
    host_sentinel = host_source / "host-sentinel"
    host_sentinel.write_text("synthetic-host-sentinel")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = DockerConfig(
        image="python@sha256:" + "a" * 64,
        gate_record=tmp_path / "gate.json",
        read_only_mounts=(runtime_mount,),
    )
    runner = FakeDockerProbeRunner(
        config,
        workspace,
        "traversal:read:synthetic-host-sentinel;absolute:blocked",
        "traversal:blocked;absolute:blocked",
    )

    with pytest.raises(RuntimeError, match="runtime_mount_host_escape_blocked hardened run"):
        _run_runtime_mount_host_escape_probe(runner, runtime_mount, host_sentinel, host_source)


def test_gate_record_refuses_missing_or_failed_probe_evidence() -> None:
    results = {
        probe: ProbeResult(contained=True, negative_control=True) for probe in REQUIRED_PROBES
    }
    with pytest.raises(ValueError, match="required probe set"):
        build_gate_record("abc123", "python@sha256:" + "0" * 64, {})

    results["memory_limit_enforced"] = ProbeResult(contained=True, negative_control=False)
    with pytest.raises(ValueError, match="memory_limit_enforced"):
        build_gate_record("abc123", "python@sha256:" + "0" * 64, results)


def test_all_embedded_probe_programs_compile() -> None:
    sources = (
        _network_probe_source("dns"),
        _network_probe_source("tcp"),
        _network_probe_source("gateway"),
        _host_path_source(),
        _outside_write_source(),
        _socket_source(),
        _privilege_source(),
        _process_limit_source(80),
        _memory_limit_source(320 * 1024 * 1024),
        _grader_source(),
        _gateway_server_source(),
    )
    for source in sources:
        compile(source, "docker-jail-probe", "exec")


@pytest.mark.slow
def test_module_entrypoint_help_does_not_require_a_docker_daemon() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_docker_jail_tests", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--gate-record" in completed.stdout
