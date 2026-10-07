"""Pure checks for the Docker red-team runner's pinned-image and evidence helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_docker_jail_tests import (
    REQUIRED_PROBES,
    ProbeResult,
    _gateway_server_source,
    _grader_source,
    _host_path_source,
    _memory_limit_source,
    _network_probe_source,
    _outside_write_source,
    _privilege_source,
    _process_limit_source,
    _socket_source,
    build_gate_record,
    insert_before_image,
    parse_image_digest,
    replace_option_value,
)


def test_parse_image_digest_requires_the_expected_repository_and_digest() -> None:
    digest = "0123456789abcdef" * 4
    assert parse_image_digest(f'["python@sha256:{digest}"]') == f"python@sha256:{digest}"

    with pytest.raises(ValueError, match="digest"):
        parse_image_digest('["python:3.12-slim"]')
    with pytest.raises(ValueError, match="python"):
        parse_image_digest(f'["other@sha256:{digest}"]')


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
