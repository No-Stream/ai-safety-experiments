"""Disk reaper rules and protections, exercised only in temporary directories."""

# ruff: noqa: S603, S607 - fixed commands and temporary paths only

from __future__ import annotations

import getpass
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import pytest

from scripts import disk_reaper

NOW = 2_000_000_000.0
DAY = 24 * 60 * 60
SYSTEM_PYTHON = Path("/usr/bin/python3")
REPO_ROOT = Path(__file__).resolve().parents[1]


class DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


@pytest.fixture
def safe_reaper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use a real temporary Git worktree and keep OS handle inspection isolated."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    monkeypatch.setattr(disk_reaper, "open_handles", lambda path: "")
    monkeypatch.setattr(disk_reaper, "TMP", tmp_path / "tmp")


def _make_file(path: Path, content: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _age_tree(path: Path, age_seconds: float, now: float = NOW) -> None:
    timestamp = now - age_seconds
    for entry in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        os.utime(entry, (timestamp, timestamp))
    os.utime(path, (timestamp, timestamp))


def _allowed(home: Path, temporary: Path, overflow: Path) -> list[Path]:
    return [home / ".cache", home / "logs", temporary, overflow]


def _execute(  # noqa: PLR0913 - explicit temporary roots
    path: Path,
    *,
    home: Path,
    temporary: Path,
    overflow: Path,
    ledger: Path,
    rule: str = "disposable-marker",
) -> str:
    return disk_reaper.execute(
        path,
        rule,
        home,
        overflow,
        ledger,
        _allowed(home, temporary, overflow),
        NOW,
    )


def _fake_rsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_rsync = fake_bin / "rsync"
    fake_rsync.write_text("#!/usr/bin/python3\n" + body)
    fake_rsync.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ.get("PATH", ""))


def test_disposable_marker_selects_remove_and_records_ledger(
    tmp_path: Path, safe_reaper: None
) -> None:
    home = tmp_path / "home"
    temporary = tmp_path / "var-tmp"
    overflow = tmp_path / "overflow"
    overflow.mkdir()
    ledger = tmp_path / "ledger.jsonl"
    candidate = home / ".cache" / "scratch-run"
    _make_file(candidate / "output.bin", b"temporary data")
    _make_file(candidate / ".disposable", b"")
    _age_tree(candidate, 8 * DAY)

    rule = disk_reaper.choose_rule(candidate, home, temporary, overflow, NOW)

    assert rule == "disposable-marker"
    assert (
        _execute(
            candidate, home=home, temporary=temporary, overflow=overflow, ledger=ledger, rule=rule
        )
        == ""
    )
    assert not candidate.exists()

    entries = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [entry["status"] for entry in entries] == ["intent", "complete"]
    entry = entries[-1]
    assert entry["path"] == str(candidate)
    assert entry["bytes"] == len(b"temporary data")
    assert entry["rule"] == rule
    assert entry["action"].upper() == "REMOVE"
    assert entry["destination"] in (None, "")
    assert "time" in entry


def test_recent_disposable_marker_does_not_select_remove(tmp_path: Path, safe_reaper: None) -> None:
    home = tmp_path / "home"
    candidate = home / ".cache" / "recent-marker"
    _make_file(candidate / "output.bin")
    marker = _make_file(candidate / ".disposable", b"")
    _age_tree(candidate, 8 * DAY)
    os.utime(marker, (NOW - 23 * 60 * 60, NOW - 23 * 60 * 60))

    rule = disk_reaper.choose_rule(
        candidate, home, tmp_path / "var-tmp", tmp_path / "overflow", NOW
    )

    assert disk_reaper.RULES[rule][0] == "REPORT ONLY"


def test_cold_inductor_cache_moves_with_rsync_and_ledger(
    tmp_path: Path, safe_reaper: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    temporary = tmp_path / "var-tmp"
    overflow = tmp_path / "overflow"
    ledger = tmp_path / "ledger.jsonl"
    candidate = home / ".cache" / "torch" / "inductor"
    _make_file(candidate / "kernel.bin", b"compiled kernel")
    _age_tree(candidate, 15 * DAY)
    overflow.mkdir()
    monkeypatch.setattr(disk_reaper.os.path, "ismount", lambda path: True)
    assert shutil.which("rsync") is not None, "MOVE integration test requires rsync"

    rule = disk_reaper.choose_rule(candidate, home, temporary, overflow, NOW)

    assert rule == "compiler-cache"
    assert (
        _execute(
            candidate, home=home, temporary=temporary, overflow=overflow, ledger=ledger, rule=rule
        )
        == ""
    )
    destination = (
        overflow
        / datetime.fromtimestamp(NOW, UTC).strftime("%Y-%m-%d")
        / ".cache"
        / "torch"
        / "inductor"
    )
    assert (destination / "kernel.bin").read_bytes() == b"compiled kernel"
    assert not candidate.exists()

    entries = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [entry["status"] for entry in entries] == ["intent", "complete"]
    entry = entries[-1]
    assert entry["path"] == str(candidate)
    assert entry["bytes"] == len(b"compiled kernel")
    assert entry["rule"] == rule
    assert entry["action"].upper() == "MOVE"
    assert Path(entry["destination"]) == destination


@pytest.mark.parametrize("failure", ["nonzero", "byte-mismatch"])
def test_move_failure_keeps_source(
    tmp_path: Path,
    safe_reaper: None,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    home = tmp_path / "home"
    temporary = tmp_path / "var-tmp"
    overflow = tmp_path / "overflow"
    overflow.mkdir()
    ledger = tmp_path / "ledger.jsonl"
    candidate = home / ".cache" / "torch" / "inductor"
    _make_file(candidate / "kernel.bin", b"source data")
    _age_tree(candidate, 15 * DAY)
    monkeypatch.setattr(disk_reaper.os.path, "ismount", lambda path: True)
    if failure == "nonzero":
        _fake_rsync(tmp_path, monkeypatch, "import sys\nsys.exit(23)\n")
        expected_exception = subprocess.CalledProcessError
    else:
        _fake_rsync(
            tmp_path,
            monkeypatch,
            "import sys\n"
            "from pathlib import Path\n"
            "destination = Path(sys.argv[-1].rstrip('/'))\n"
            "destination.mkdir(parents=True, exist_ok=True)\n"
            "(destination / 'kernel.bin').write_bytes(b'wrong')\n",
        )
        expected_exception = RuntimeError

    with pytest.raises(expected_exception):
        disk_reaper.execute(
            candidate,
            "compiler-cache",
            home,
            overflow,
            ledger,
            _allowed(home, temporary, overflow),
            NOW,
        )

    assert (candidate / "kernel.bin").read_bytes() == b"source data"
    assert ledger.exists()
    assert not ledger.read_text().splitlines()


def test_large_unclassified_tree_is_report_only(tmp_path: Path, safe_reaper: None) -> None:
    home = tmp_path / "home"
    temporary = tmp_path / "var-tmp"
    overflow = tmp_path / "overflow"
    ledger = tmp_path / "ledger.jsonl"
    candidate = home / ".cache" / "unclassified" / "large-data.bin"
    candidate.parent.mkdir(parents=True)
    with candidate.open("wb") as file:
        file.truncate(5 * 1024**3 + 1)
    _age_tree(candidate.parent, 2 * DAY)

    rule = disk_reaper.choose_rule(candidate.parent, home, temporary, overflow, NOW)

    assert rule == "unmatched"
    assert disk_reaper.RULES[rule][0] == "REPORT ONLY"
    assert (
        _execute(
            candidate.parent,
            home=home,
            temporary=temporary,
            overflow=overflow,
            ledger=ledger,
            rule=rule,
        )
        == ""
    )
    assert candidate.exists()
    assert candidate.stat().st_size == 5 * 1024**3 + 1
    assert not ledger.exists()


def test_recent_tree_is_skipped_by_protection(tmp_path: Path, safe_reaper: None) -> None:
    home = tmp_path / "home"
    temporary = tmp_path / "var-tmp"
    overflow = tmp_path / "overflow"
    ledger = tmp_path / "ledger.jsonl"
    candidate = _make_file(home / ".cache" / "recent" / "result.bin")
    os.utime(candidate, (NOW - 3 * 60 * 60, NOW - 3 * 60 * 60))

    reason = disk_reaper.protect(candidate.parent, _allowed(home, temporary, overflow), NOW)

    assert "6 hours" in reason.lower()
    assert _execute(
        candidate.parent,
        home=home,
        temporary=temporary,
        overflow=overflow,
        ledger=ledger,
    )
    assert candidate.exists()


def test_keep_marker_protects_path_and_execute_skips(tmp_path: Path, safe_reaper: None) -> None:
    home = tmp_path / "home"
    temporary = tmp_path / "var-tmp"
    overflow = tmp_path / "overflow"
    ledger = tmp_path / "ledger.jsonl"
    candidate = _make_file(home / ".cache" / "nested" / "result.bin")
    _make_file(home / ".cache" / ".keep")
    _age_tree(home / ".cache", 2 * DAY)

    reason = disk_reaper.protect(candidate.parent, _allowed(home, temporary, overflow), NOW)

    assert ".keep" in reason
    assert _execute(
        candidate.parent,
        home=home,
        temporary=temporary,
        overflow=overflow,
        ledger=ledger,
    )
    assert candidate.exists()
    assert not ledger.exists()


def test_keep_marker_inside_tree_protects_parent(tmp_path: Path, safe_reaper: None) -> None:
    home = tmp_path / "home"
    temporary = tmp_path / "var-tmp"
    overflow = tmp_path / "overflow"
    ledger = tmp_path / "ledger.jsonl"
    candidate = home / ".cache" / "parent"
    marker = _make_file(candidate / "nested" / ".keep", b"")
    _make_file(candidate / "result.bin")
    _age_tree(candidate, 2 * DAY)

    reason = disk_reaper.protect(candidate, _allowed(home, temporary, overflow), NOW)

    assert reason == "contains a protected subtree (.keep)"
    assert _execute(
        candidate,
        home=home,
        temporary=temporary,
        overflow=overflow,
        ledger=ledger,
    )
    assert marker.exists()
    assert (candidate / "result.bin").exists()
    assert not ledger.exists()


def test_deep_keep_marker_reports_protected_subtree_in_dry_run(
    tmp_path: Path, safe_reaper: None, caplog: pytest.LogCaptureFixture
) -> None:
    home = tmp_path / "home"
    candidate = home / ".cache" / "uv"
    marker = _make_file(candidate / "pkg" / "lib" / ".keep", b"")
    _make_file(candidate / "cache.db")
    _age_tree(candidate, 2 * DAY)
    caplog.set_level(logging.INFO, logger="scripts.disk_reaper")

    disk_reaper.run(home, tmp_path / "var-tmp", tmp_path / "overflow", tmp_path / "ledger")

    rows = [message for message in caplog.messages if str(candidate) in message]
    assert len(rows) == 1
    assert "| REPORT ONLY | contains a protected subtree (.keep)" in rows[0]
    assert candidate.exists()
    assert marker.exists()


def test_git_tracked_file_is_protected(tmp_path: Path, safe_reaper: None) -> None:
    home = tmp_path / "home"
    temporary = tmp_path / "var-tmp"
    overflow = tmp_path / "overflow"
    ledger = tmp_path / "ledger.jsonl"
    repository = home / ".cache" / "tracked-repository"
    candidate = _make_file(repository / "tracked.bin", b"tracked")
    subprocess.run(["git", "-C", str(repository), "init", "--quiet"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "tracked.bin"], check=True)

    reason = disk_reaper.protect(candidate, _allowed(home, temporary, overflow), NOW)

    assert "git-tracked" in reason
    assert _execute(
        candidate,
        home=home,
        temporary=temporary,
        overflow=overflow,
        ledger=ledger,
    )
    assert candidate.read_bytes() == b"tracked"


def test_adapter_tree_is_protected_and_execute_skips(tmp_path: Path, safe_reaper: None) -> None:
    home = tmp_path / "home"
    temporary = tmp_path / "var-tmp"
    overflow = tmp_path / "overflow"
    ledger = tmp_path / "ledger.jsonl"
    candidate = home / ".cache" / "torch" / "inductor"
    _make_file(candidate / "kernel.bin")
    _make_file(candidate / "nested" / "adapter_config.json", b"{}")
    _age_tree(candidate, 15 * DAY)

    reason = disk_reaper.protect(candidate, _allowed(home, temporary, overflow), NOW)

    assert "adapter" in reason
    assert _execute(
        candidate,
        home=home,
        temporary=temporary,
        overflow=overflow,
        ledger=ledger,
        rule="compiler-cache",
    )
    assert (candidate / "kernel.bin").exists()
    assert not ledger.exists()


@pytest.mark.parametrize(
    "learned_asset",
    [
        "final/adapter_config.json",
        "final/adapter_model.safetensors",
        "base_results_and_detectors_layer20/probe.pt",
        "../adapter_config.json",
    ],
)
def test_apply_never_removes_disposable_tree_holding_learned_assets(
    tmp_path: Path, safe_reaper: None, learned_asset: str
) -> None:
    home = tmp_path / "home"
    temporary, overflow, ledger = tmp_path / "var-tmp", tmp_path / "overflow", tmp_path / "ledger"
    run_directory = home / "workspace" / "repo" / "artifacts" / "smoke-run"
    candidate = run_directory / "checkpoint-1"
    _make_file(candidate / "output.bin")
    _make_file(candidate / ".disposable", b"")
    _make_file(candidate / learned_asset, b"{}")
    _age_tree(run_directory, 8 * DAY, time.time())

    reason = disk_reaper.protect(
        candidate, disk_reaper.roots(home, temporary, overflow), time.time(), check_handles=False
    )
    disk_reaper.run(home, temporary, overflow, ledger, apply=True)

    assert (candidate / "output.bin").exists()
    assert not ledger.exists()
    assert reason == "adapter or trained probe protection"


def test_handle_tools_missing_protects_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    temporary = tmp_path / "var-tmp"
    overflow = tmp_path / "overflow"
    candidate = _make_file(home / ".cache" / "uninspectable" / "result.bin")
    os.utime(candidate, (NOW - 2 * DAY, NOW - 2 * DAY))
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    monkeypatch.setattr(disk_reaper.shutil, "which", lambda name: None)

    reason = disk_reaper.protect(candidate, _allowed(home, temporary, overflow), NOW)

    assert "neither lsof nor fuser" in reason


def test_main_dry_runs_by_default_and_removes_only_with_apply(
    tmp_path: Path,
    safe_reaper: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home = tmp_path / "home"
    temporary = tmp_path / "var-tmp"
    temporary.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(disk_reaper, "TEMPORARY", temporary)
    monkeypatch.setattr(disk_reaper, "OVERFLOW", tmp_path / "overflow")
    ledger = temporary / "disk-reaper-ledger.jsonl"
    candidate = home / ".cache" / "cold-run"
    _make_file(candidate / "output.bin")
    _make_file(candidate / ".disposable", b"")
    _age_tree(candidate, 8 * DAY, time.time())
    caplog.set_level(logging.INFO, logger="scripts.disk_reaper")

    assert disk_reaper.main([]) == 0

    assert "path | size | rule | action | reason" in caplog.text
    assert any(
        str(candidate) in line and "| disposable-marker | REMOVE |" in line
        for line in caplog.text.splitlines()
    )
    assert candidate.exists()
    assert not ledger.exists()

    assert disk_reaper.main(["--apply"]) == 0

    assert not candidate.exists()
    assert [json.loads(line)["status"] for line in ledger.read_text().splitlines()] == [
        "intent",
        "complete",
    ]


def test_young_disposable_directory_is_one_report_row(
    tmp_path: Path, safe_reaper: None, caplog: pytest.LogCaptureFixture
) -> None:
    home = tmp_path / "home"
    candidate = home / "workspace" / "repo" / "artifacts" / "fresh-smoke"
    with _make_file(candidate / "checkpoint.bin").open("wb") as checkpoint:
        checkpoint.truncate(2 * disk_reaper.GIB)
    _make_file(candidate / ".disposable", b"")
    _age_tree(candidate, 12 * 60 * 60, time.time())
    caplog.set_level(logging.INFO, logger="scripts.disk_reaper")

    disk_reaper.run(home, tmp_path / "var-tmp", tmp_path / "overflow", tmp_path / "ledger")

    rows = [message for message in caplog.messages if str(candidate) in message]
    assert len(rows) == 1
    assert rows[0].startswith(f"{candidate} | 2.000 GiB")
    assert "| research-data | REPORT ONLY |" in rows[0]
    assert candidate.exists()


def test_path_outside_allowlist_is_protected(tmp_path: Path, safe_reaper: None) -> None:
    home = tmp_path / "home"
    temporary = tmp_path / "var-tmp"
    overflow = tmp_path / "overflow"
    ledger = tmp_path / "ledger.jsonl"
    candidate = _make_file(home / "unapproved" / "result.bin")
    _age_tree(candidate.parent, 2 * DAY)

    reason = disk_reaper.protect(candidate.parent, _allowed(home, temporary, overflow), NOW)

    assert "allowlist" in reason.lower()
    assert _execute(
        candidate.parent,
        home=home,
        temporary=temporary,
        overflow=overflow,
        ledger=ledger,
    )
    assert candidate.exists()


def test_threshold_mode_noops_above_threshold(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        disk_reaper.shutil,
        "disk_usage",
        lambda path: DiskUsage(2 * disk_reaper.GIB, disk_reaper.GIB, disk_reaper.GIB),
    )
    monkeypatch.setattr(
        disk_reaper, "run", lambda *args, **kwargs: pytest.fail("scan should not run")
    )
    caplog.set_level(logging.INFO, logger="scripts.disk_reaper")

    assert disk_reaper.main(["--if-free-below-gb", "0.5"]) == 0
    assert "Free space exceeds threshold" in caplog.text


@pytest.mark.skipif(not SYSTEM_PYTHON.exists(), reason=f"no system interpreter at {SYSTEM_PYTHON}")
def test_script_executes_under_system_python(tmp_path: Path) -> None:
    """Run the actual entry point under the interpreter used by system scripts."""
    environment = {**os.environ, "HOME": str(tmp_path)}
    completed = subprocess.run(
        [str(SYSTEM_PYTHON), str(REPO_ROOT / "scripts" / "disk_reaper.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )

    assert "--apply" in completed.stdout
    assert "--if-free-below-gb" in completed.stdout


def test_active_container_does_not_hide_cold_disposable_child(
    tmp_path: Path, safe_reaper: None, caplog: pytest.LogCaptureFixture
) -> None:
    home = tmp_path / "home"
    container = home / ".cache" / "runs"
    cold = container / "old-run"
    _make_file(cold / ".disposable", b"")
    _make_file(cold / "output.bin")
    _age_tree(cold, 8 * DAY, now=time.time())
    _make_file(container / "active-run" / "output.bin")
    caplog.set_level(logging.INFO, logger="scripts.disk_reaper")
    disk_reaper.run(home, tmp_path / "var-tmp", tmp_path / "overflow", tmp_path / "ledger")
    assert any(str(cold) in line and "| REMOVE |" in line for line in caplog.text.splitlines())
    assert cold.exists()


def test_pip_cache_uses_its_cleaner_on_temporary_cache(tmp_path: Path, safe_reaper: None) -> None:
    if shutil.which("pip") is None:
        pytest.skip("pip cleaner unavailable")
    home = tmp_path / "home"
    cache = home / ".cache" / "pip"
    wheel = _make_file(cache / "wheels" / "example-1.0-py3-none-any.whl")
    _age_tree(cache, 2 * DAY)
    temporary, overflow, ledger = tmp_path / "var-tmp", tmp_path / "overflow", tmp_path / "ledger"
    rule = disk_reaper.choose_rule(cache, home, temporary, overflow, NOW)
    assert rule == "tool-cache"
    assert (
        _execute(cache, home=home, temporary=temporary, overflow=overflow, ledger=ledger, rule=rule)
        == ""
    )
    assert not wheel.exists()
    assert [json.loads(line)["status"] for line in ledger.read_text().splitlines()] == [
        "intent",
        "complete",
    ]


def test_disposable_marker_expires_after_seven_days(tmp_path: Path, safe_reaper: None) -> None:
    home = tmp_path / "home"
    temporary, overflow = tmp_path / "var-tmp", tmp_path / "overflow"
    candidate = home / "workspace" / "repo" / "artifacts" / "old-run"
    marker = _make_file(candidate / ".disposable", b"")
    os.utime(marker, (NOW - 7 * DAY - 1, NOW - 7 * DAY - 1))

    rule = disk_reaper.choose_rule(candidate, home, temporary, overflow, NOW)

    assert rule == "disposable-marker"
    assert disk_reaper.RULES[rule][0] == "REMOVE"


def test_disposable_marker_younger_than_seven_days_stays_report_only(
    tmp_path: Path, safe_reaper: None
) -> None:
    home = tmp_path / "home"
    candidate = home / "workspace" / "repo" / "artifacts" / "recent-run"
    marker = _make_file(candidate / ".disposable", b"")
    os.utime(marker, (NOW - 7 * DAY + 1, NOW - 7 * DAY + 1))

    rule = disk_reaper.choose_rule(
        candidate, home, tmp_path / "var-tmp", tmp_path / "overflow", NOW
    )

    assert rule != "disposable-marker"
    assert disk_reaper.RULES[rule][0] == "REPORT ONLY"


@pytest.mark.parametrize("name", ["agent.sh", "gate.files", "task-wrapper.py"])
def test_old_top_level_var_tmp_launch_and_gate_files_are_removed_by_rule(
    tmp_path: Path, safe_reaper: None, name: str
) -> None:
    home = tmp_path / "home"
    temporary, overflow = tmp_path / "var-tmp", tmp_path / "overflow"
    candidate = _make_file(temporary / name)
    os.utime(candidate, (NOW - 21 * DAY - 1, NOW - 21 * DAY - 1))

    rule = disk_reaper.choose_rule(candidate, home, temporary, overflow, NOW)

    assert rule == "agent-launch-file"
    assert disk_reaper.RULES[rule][0] == "REMOVE"


def test_recent_top_level_var_tmp_launch_file_is_not_removed(
    tmp_path: Path, safe_reaper: None
) -> None:
    candidate = _make_file(tmp_path / "var-tmp" / "agent.sh")
    os.utime(candidate, (NOW - 21 * DAY + 1, NOW - 21 * DAY + 1))

    rule = disk_reaper.choose_rule(
        candidate, tmp_path / "home", candidate.parent, tmp_path / "overflow", NOW
    )

    assert rule != "agent-launch-file"
    assert disk_reaper.RULES[rule][0] == "REPORT ONLY"


def test_nested_var_tmp_launch_file_is_not_removed(tmp_path: Path, safe_reaper: None) -> None:
    temporary = tmp_path / "var-tmp"
    candidate = _make_file(temporary / "nested" / "agent.sh")
    os.utime(candidate, (NOW - 22 * DAY, NOW - 22 * DAY))

    rule = disk_reaper.choose_rule(
        candidate, tmp_path / "home", temporary, tmp_path / "overflow", NOW
    )

    assert rule != "agent-launch-file"
    assert disk_reaper.RULES[rule][0] == "REPORT ONLY"


def test_old_pytest_directory_is_removed_after_three_days(
    tmp_path: Path, safe_reaper: None
) -> None:
    home = tmp_path / "home"
    temporary, overflow = tmp_path / "var-tmp", tmp_path / "overflow"
    pytest_root = disk_reaper.TMP / f"pytest-of-{getpass.getuser()}"
    candidate = pytest_root / "pytest-123"
    _make_file(candidate / "result.bin")
    _age_tree(candidate, 3 * DAY + 1, NOW)

    rule = disk_reaper.choose_rule(candidate, home, temporary, overflow, NOW)

    assert rule == "pytest-temporary"
    assert disk_reaper.RULES[rule][0] == "REMOVE"


def test_recent_pytest_directory_is_not_removed(tmp_path: Path, safe_reaper: None) -> None:
    home = tmp_path / "home"
    candidate = disk_reaper.TMP / f"pytest-of-{getpass.getuser()}" / "pytest-recent"
    _make_file(candidate / "result.bin")
    _age_tree(candidate, 3 * DAY - 1, NOW)

    rule = disk_reaper.choose_rule(
        candidate, home, tmp_path / "var-tmp", tmp_path / "overflow", NOW
    )

    assert rule != "pytest-temporary"
    assert disk_reaper.RULES[rule][0] == "REPORT ONLY"


def test_old_leak_environment_work_item_is_removed_after_seven_days(
    tmp_path: Path, safe_reaper: None
) -> None:
    home = tmp_path / "home"
    temporary, overflow = tmp_path / "var-tmp", tmp_path / "overflow"
    candidate = temporary / "leak-env" / "work" / "run-1"
    _make_file(candidate / "result.bin")
    _age_tree(candidate, 7 * DAY + 1, NOW)

    rule = disk_reaper.choose_rule(candidate, home, temporary, overflow, NOW)

    assert rule == "leak-env-work"
    assert disk_reaper.RULES[rule][0] == "REMOVE"


def test_recent_leak_environment_work_item_is_not_removed(
    tmp_path: Path, safe_reaper: None
) -> None:
    home = tmp_path / "home"
    temporary, overflow = tmp_path / "var-tmp", tmp_path / "overflow"
    candidate = temporary / "leak-env" / "work" / "recent"
    _make_file(candidate / "result.bin")
    _age_tree(candidate, 7 * DAY - 1, NOW)

    rule = disk_reaper.choose_rule(candidate, home, temporary, overflow, NOW)

    assert rule != "leak-env-work"
    assert disk_reaper.RULES[rule][0] == "REPORT ONLY"


@pytest.mark.parametrize("expiry_kind", ["agent-file", "pytest-directory", "leak-work-item"])
def test_new_expiry_rules_respect_keep_markers_everywhere(
    tmp_path: Path, safe_reaper: None, expiry_kind: str
) -> None:
    home = tmp_path / "home"
    temporary, overflow = tmp_path / "var-tmp", tmp_path / "overflow"
    if expiry_kind == "agent-file":
        candidate = _make_file(temporary / "agent.sh")
        keep = _make_file(temporary / ".keep", b"")
    elif expiry_kind == "pytest-directory":
        pytest_root = disk_reaper.TMP / f"pytest-of-{getpass.getuser()}"
        candidate = _make_file(pytest_root / "pytest-kept" / "result.bin").parent
        keep = _make_file(candidate / "nested" / ".keep", b"")
    else:
        candidate = _make_file(temporary / "leak-env" / "work" / "kept" / "result.bin").parent
        keep = _make_file(candidate / ".keep", b"")
    allowed = disk_reaper.roots(home, temporary, overflow)

    reason = disk_reaper.protect(candidate, allowed, NOW, check_handles=False)

    assert ".keep" in reason
    assert candidate.exists()
    assert keep.exists()


def test_remove_only_apply_keeps_pressure_move_candidates(
    tmp_path: Path, safe_reaper: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    temporary, overflow = tmp_path / "var-tmp", tmp_path / "overflow"
    ledger = tmp_path / "ledger.jsonl"
    disposable = home / "workspace" / "repo" / "artifacts" / "expired-run"
    _make_file(disposable / ".disposable", b"")
    _make_file(disposable / "result.bin")
    _age_tree(disposable, 8 * DAY, time.time())
    move_candidate = home / ".cache" / "torch" / "inductor"
    _make_file(move_candidate / "kernel.bin")
    _age_tree(home / ".cache", 15 * DAY, time.time())
    execute = disk_reaper.execute

    def execute_remove_rules_only(  # noqa: PLR0913, PLR0917 - mirrors explicit execute inputs
        path: Path,
        rule: str,
        home_path: Path,
        overflow_path: Path,
        ledger_path: Path,
        allowed_paths: list[Path],
        now: float,
    ) -> str:
        assert rule != "compiler-cache", "remove-only mode tried to move a pressure candidate"
        return execute(path, rule, home_path, overflow_path, ledger_path, allowed_paths, now)

    monkeypatch.setattr(disk_reaper, "execute", execute_remove_rules_only)

    disk_reaper.run(home, temporary, overflow, ledger, apply=True, remove_only=True)

    assert not disposable.exists()
    assert move_candidate.exists()
    assert (move_candidate / "kernel.bin").exists()


def test_disk_reaper_timer_runs_daily_remove_only_without_a_free_space_gate() -> None:
    service = (REPO_ROOT / "scripts" / "disk-reaper.service").read_text()
    timer = (REPO_ROOT / "scripts" / "disk-reaper.timer").read_text()

    assert "disk_reaper.py --apply --remove-only" in service
    assert "--if-free-below-gb" not in service
    assert "OnCalendar=daily" in timer
    assert "OnCalendar=hourly" not in timer
