"""The aggregate jail budget starts once, outside the trainer's cgroup."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.organism_rl import jail_budget

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def slice_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[list[str]]:
    calls: list[list[str]] = []
    cgroup = tmp_path / "cgroup" / "orgjailsfixture.slice"
    cgroup.mkdir(parents=True)
    (cgroup / "memory.max").write_text("1000\n")
    (cgroup / "memory.swap.max").write_text("0\n")
    (cgroup / "memory.oom.group").write_text("0\n")
    monkeypatch.setattr(jail_budget, "CGROUP_V2_ROOT", tmp_path / "cgroup")
    monkeypatch.setattr(
        jail_budget.psutil, "virtual_memory", lambda: SimpleNamespace(available=1200)
    )
    monkeypatch.setenv("ORGANISM_HOST_MEMORY_SAFETY_MARGIN_BYTES", "200")

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        stdout = "/orgjailsfixture.slice\n" if "show" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(jail_budget.subprocess, "run", run)
    return calls


def test_first_rollout_snapshot_is_shared_and_recorded(
    slice_host: list[list[str]], tmp_path: Path
) -> None:
    budget = jail_budget.JailBudgetSlice(tmp_path)
    assert slice_host == []
    with budget, ThreadPoolExecutor(max_workers=4) as executor:
        limits = list(executor.map(lambda _: budget.limits(), range(8)))
        assert all(item is limits[0] for item in limits)
        slice_name = limits[0].slice_name
        assert slice_name is not None
        assert slice_name.endswith(".slice")
        assert "-" not in slice_name
        assert limits[0].memory_peak_log == tmp_path / "jail-memory-peaks.jsonl"
        start = next(call for call in slice_host if "StartTransientUnit" in call)
        assert start[start.index("MemoryMax") : start.index("MemoryMax") + 3] == [
            "MemoryMax",
            "t",
            "1000",
        ]
        assert start[start.index("MemorySwapMax") : start.index("MemorySwapMax") + 3] == [
            "MemorySwapMax",
            "t",
            "0",
        ]
        assert "PIDs" not in start
        provenance = json.loads((tmp_path / "startup-jail-memory.jsonl").read_text())
        assert provenance["startup_available_bytes"] == 1200
        assert provenance["safety_margin_bytes"] == 200
        assert provenance["memory_max_bytes"] == 1000
        assert provenance["memory_oom_group"] == 0
        assert provenance["trainer_excluded"] is True
    assert len([call for call in slice_host if "StartTransientUnit" in call]) == 1
    assert slice_host[-1] == ["systemctl", "--user", "stop", slice_name]
    budget.close()
    with pytest.raises(RuntimeError, match="closed"):
        budget.limits()


@pytest.mark.parametrize("available", [0, 199, 200])
def test_insufficient_headroom_refuses_before_creating_slice(
    slice_host: list[list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, available: int
) -> None:
    monkeypatch.setattr(
        jail_budget.psutil, "virtual_memory", lambda: SimpleNamespace(available=available)
    )
    with pytest.raises(ValueError, match="available"):
        jail_budget.JailBudgetSlice(tmp_path).limits()
    assert slice_host == []


@pytest.mark.parametrize(
    ("property_name", "value"),
    [("memory.max", "999"), ("memory.swap.max", "1"), ("memory.oom.group", "1")],
)
def test_kernel_mismatch_is_fatal_and_cleans_up(
    slice_host: list[list[str]], tmp_path: Path, property_name: str, value: str
) -> None:
    (tmp_path / "cgroup" / "orgjailsfixture.slice" / property_name).write_text(value)
    with pytest.raises(RuntimeError, match=property_name):
        jail_budget.JailBudgetSlice(tmp_path).limits()
    assert slice_host[-1][:3] == ["systemctl", "--user", "stop"]
    assert not (tmp_path / "startup-jail-memory.jsonl").exists()


def test_unused_lifecycle_does_not_contact_systemd(
    slice_host: list[list[str]], tmp_path: Path
) -> None:
    with jail_budget.JailBudgetSlice(tmp_path):
        pass
    assert slice_host == []


def test_start_failure_does_not_weaken_limits(
    slice_host: list[list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, argv, stderr="denied")

    monkeypatch.setattr(jail_budget.subprocess, "run", refuse)
    with pytest.raises(subprocess.CalledProcessError):
        jail_budget.JailBudgetSlice(tmp_path).limits()


def test_invalid_safety_margin_fails_before_systemd(
    slice_host: list[list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANISM_HOST_MEMORY_SAFETY_MARGIN_BYTES", "-1")
    with pytest.raises(ValueError, match="nonnegative"):
        jail_budget.JailBudgetSlice(tmp_path).limits()
    assert slice_host == []


@pytest.mark.parametrize(("available", "margin", "expected"), [(1000, 200, 800), (1000, 0, 1000)])
def test_resolve_jail_memory_budget(available: int, margin: int, expected: int) -> None:
    assert jail_budget.resolve_jail_memory_budget(available, margin) == expected


@pytest.mark.parametrize(("available", "margin"), [(-1, 0), (1, -1), (0, 0), (100, 100)])
def test_resolve_jail_memory_budget_refuses_invalid_inputs(available: int, margin: int) -> None:
    with pytest.raises(ValueError, match="available"):
        jail_budget.resolve_jail_memory_budget(available, margin)


def test_resume_keeps_startup_provenance(slice_host: list[list[str]], tmp_path: Path) -> None:
    with jail_budget.JailBudgetSlice(tmp_path) as first:
        first.limits()
    with jail_budget.JailBudgetSlice(tmp_path) as second:
        second.limits()
    records = [
        json.loads(line)
        for line in (tmp_path / "startup-jail-memory.jsonl").read_text().splitlines()
    ]
    assert [record["slice_name"] for record in records] == [first.slice_name, second.slice_name]
    assert first.slice_name != second.slice_name
