"""Exercise the fast-tier budget warning through real pytest subprocesses."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_TESTS = """\
import time

import pytest


@pytest.fixture
def slow_setup_and_teardown():
    time.sleep(0.015)
    yield
    time.sleep(0.015)


def test_unmarked_slow():
    time.sleep(0.05)


def test_slow_across_phases(slow_setup_and_teardown):
    pass


@pytest.mark.slow
def test_slow_marked():
    time.sleep(0.05)


@pytest.mark.jail_real
def test_jail_real_marked():
    time.sleep(0.05)


@pytest.mark.docker_real
def test_docker_real_marked():
    time.sleep(0.05)
"""


def _run_synthetic_pytest(
    tmp_path: Path, *, budget_seconds: str | None, distributed: bool
) -> subprocess.CompletedProcess[str]:
    test_file = tmp_path / "test_synthetic_budget.py"
    test_file.write_text(SYNTHETIC_TESTS)
    environment = os.environ.copy()
    environment.pop("FAST_TIER_BUDGET_SECONDS", None)
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), environment.get("PYTHONPATH", "")])
    if budget_seconds is not None:
        environment["FAST_TIER_BUDGET_SECONDS"] = budget_seconds

    command = [sys.executable, "-m", "pytest", "-q", "-p", "pytest_fast_tier_budget"]
    if distributed:
        command.extend(["-n", "2", "--dist", "loadfile"])
    command.append(str(test_file))
    return subprocess.run(  # noqa: S603 - fixed interpreter, plugin, and temporary test file
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_warns_for_only_unmarked_over_budget_test_from_xdist_reports(tmp_path: Path) -> None:
    result = _run_synthetic_pytest(tmp_path, budget_seconds="0.02", distributed=True)

    output = result.stdout + result.stderr
    assert result.returncode == 0, f"warning changed the pytest exit status:\n{output}"
    assert "FAST TIER BUDGET WARNING" in output, output
    assert "test_synthetic_budget.py::test_unmarked_slow" in output, output
    assert "test_synthetic_budget.py::test_slow_across_phases" in output, output
    assert "test_synthetic_budget.py::test_slow_marked" not in output, output
    assert "test_synthetic_budget.py::test_jail_real_marked" not in output, output
    assert "test_synthetic_budget.py::test_docker_real_marked" not in output, output


def test_is_silent_when_budget_is_unset(tmp_path: Path) -> None:
    result = _run_synthetic_pytest(tmp_path, budget_seconds=None, distributed=False)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "FAST TIER BUDGET WARNING" not in output, output
