"""Exit-time peak recording reads the service cgroup before systemd removes it."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.resource_limits_memory import record_memory_peak

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("service_result", ["success", "exit-code", "oom-kill", "timeout"])
def test_records_peak_before_cgroup_removal(tmp_path: Path, service_result: str) -> None:
    membership = tmp_path / "cgroup"
    membership.write_text("0::/episodes.slice/jail.service\n")
    root = tmp_path / "cgroups"
    service_cgroup = root / "episodes.slice" / "jail.service"
    service_cgroup.mkdir(parents=True)
    peak_file = service_cgroup / "memory.peak"
    peak_file.write_text("123456789\n")
    log_path = tmp_path / "peaks.jsonl"
    record_memory_peak(
        log_path,
        "jail.service",
        {"SERVICE_RESULT": service_result, "EXIT_CODE": "exited", "EXIT_STATUS": "0"},
        cgroup_root=root,
        membership_path=membership,
    )
    peak_file.unlink()
    assert json.loads(log_path.read_text()) == {
        "unit_name": "jail.service",
        "memory_peak_bytes": 123456789,
        "unavailable_reason": None,
        "service_result": service_result,
        "exit_code": "exited",
        "exit_status": "0",
    }


@pytest.mark.parametrize(
    ("membership_text", "reason"),
    [
        ("0::/jail.service\n", "memory.peak is unavailable"),
        ("2:memory:/jail.service\n", "unified cgroup membership is unavailable"),
    ],
)
def test_records_explicit_unavailable_peak(
    tmp_path: Path, membership_text: str, reason: str
) -> None:
    membership = tmp_path / "cgroup"
    membership.write_text(membership_text)
    log_path = tmp_path / "peaks.jsonl"
    for _ in range(2):
        record_memory_peak(
            log_path, "jail.service", {}, cgroup_root=tmp_path, membership_path=membership
        )
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 2
    assert all(record["memory_peak_bytes"] is None for record in records)
    assert all(record["unavailable_reason"] == reason for record in records)


def test_invalid_peak_fails_loudly(tmp_path: Path) -> None:
    membership = tmp_path / "cgroup"
    membership.write_text("0::/\n")
    (tmp_path / "memory.peak").write_text("not-a-number\n")
    with pytest.raises(ValueError, match="invalid literal"):
        record_memory_peak(
            tmp_path / "peaks.jsonl",
            "jail.service",
            {},
            cgroup_root=tmp_path,
            membership_path=membership,
        )


def test_cli_runs_under_system_python(tmp_path: Path) -> None:
    log_path = tmp_path / "peaks.jsonl"
    completed = subprocess.run(  # noqa: S603 - trusted CLI under its actual interpreter
        [
            "/usr/bin/python3",
            str(REPO_ROOT / "scripts/resource_limits_memory.py"),
            str(log_path),
            "jail.service",
        ],
        env={
            **os.environ,
            "SERVICE_RESULT": "oom-kill",
            "EXIT_CODE": "killed",
            "EXIT_STATUS": "KILL",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(log_path.read_text())
    assert record["service_result"] == "oom-kill"
    assert record["exit_code"] == "killed"
    assert record["exit_status"] == "KILL"
