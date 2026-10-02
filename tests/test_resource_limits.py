"""Drive scripts/resource-limits.sh on a host it was not written for.

The limiter exists so a runaway job cannot make a box unresponsive, and its CPU ceiling and memory
cap were literals measured on this dev box (64 cores, 254 GB). Job placement now sends real work to
rented GPU instances, where a 4-vCPU, 30 GiB box would get `CPUQuota=5000%` and `MemoryMax=160G` --
neither of which binds, so the containment the script exists for is absent while the banner still
prints reassuring numbers. This is the repo's own never-hardcode-a-memory-budget rule applied to the
CPU and RAM side.

Both tests run the real script. `nproc` is shimmed onto PATH to stand in for a smaller host, which
is the whole point: nothing about the limiter's arithmetic should come from this box's dimensions.
`--advisory` is used for the memory case because it needs no systemd user instance -- it sets
`ulimit -v` from the same resolved cap and then execs, so the cap is observable from inside the job.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LIMITER = REPO_ROOT / "scripts" / "resource-limits.sh"
SYSTEMCTL = shutil.which("systemctl")

# `numfmt --to=iec` keeps three significant digits, so the cap is a rounded spelling.
IEC_ROUNDING_TOLERANCE = 0.02


def _systemctl_user(*arguments: str) -> subprocess.CompletedProcess[str]:
    assert SYSTEMCTL is not None
    return subprocess.run(  # noqa: S603 - systemctl is resolved from the host PATH
        [SYSTEMCTL, "--user", *arguments],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _user_manager_is_available() -> bool:
    if SYSTEMCTL is None:
        return False
    state = _systemctl_user("is-system-running").stdout.strip()
    if state not in {"running", "degraded", "starting", "maintenance", "stopping"}:
        return False
    return _systemctl_user("show-environment").returncode == 0


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_systemd_state(unit_name: str, expected_state: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _systemctl_user("show", "-p", "ActiveState", "--value", unit_name).stdout.strip()
        if state == expected_state:
            return True
        time.sleep(0.05)
    return False


def _mem_total_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    raise AssertionError("no MemTotal in /proc/meminfo")


@pytest.fixture
def four_core_host(tmp_path: Path) -> dict[str, str]:
    """An environment whose `nproc` reports four cores, like a g6e.xlarge."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nproc = bin_dir / "nproc"
    nproc.write_text("#!/bin/sh\necho 4\n")
    nproc.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


@pytest.fixture
def eight_core_host(tmp_path: Path) -> dict[str, str]:
    """An environment whose `nproc` reports eight cores for deterministic window bounds."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nproc = bin_dir / "nproc"
    nproc.write_text("#!/bin/sh\necho 8\n")
    nproc.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


def _run_advisory_affinity_window(env: dict[str, str], *, timeout: bool) -> list[int]:
    arguments = [str(LIMITER), "--advisory", "--cpus", "2"]
    if timeout:
        arguments.extend(["--timeout", "10s"])
    arguments.extend(
        [
            "--",
            "python3",
            "-I",
            "-c",
            "import json, os; print(json.dumps(sorted(os.sched_getaffinity(0))))",
        ]
    )
    completed = subprocess.run(  # noqa: S603 - repo script, literal arguments
        arguments,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return [int(core) for core in json.loads(completed.stdout)]


class TestTheCpuCeilingFollowsTheHost:
    """--cpus is validated against the cores the host actually has."""

    def test_more_cpus_than_the_host_has_is_refused(self, four_core_host: dict[str, str]) -> None:
        completed = subprocess.run(  # noqa: S603 - repo script, literal arguments
            [str(LIMITER), "-c", "8", "--", "/bin/true"],
            env=four_core_host,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 2, completed.stdout + completed.stderr
        assert "--cpus must be 1..4" in completed.stderr, completed.stderr

    def test_the_hosts_own_core_count_is_accepted(self, four_core_host: dict[str, str]) -> None:
        """The control: a ceiling that refuses everything is as useless as one that never binds."""
        completed = subprocess.run(  # noqa: S603 - repo script, literal arguments
            [str(LIMITER), "--advisory", "-c", "4", "--", "/bin/true"],
            env=four_core_host,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    def test_advisory_cpu_affinity_rotates_across_contiguous_windows(
        self, eight_core_host: dict[str, str]
    ) -> None:
        no_timeout_windows = [
            _run_advisory_affinity_window(eight_core_host, timeout=False) for _ in range(8)
        ]
        timeout_windows = [
            _run_advisory_affinity_window(eight_core_host, timeout=True) for _ in range(8)
        ]

        for window in no_timeout_windows + timeout_windows:
            assert len(window) == 2
            assert window == list(range(window[0], window[0] + 2))
            assert window[0] >= 0
            assert window[-1] < 8

        assert len({tuple(window) for window in no_timeout_windows}) > 1
        assert len({tuple(window) for window in timeout_windows}) > 1


class TestTheMemoryCapFollowsTheHost:
    """The default hard cap is a fraction of this host's RAM, not a number from another box."""

    def test_the_default_cap_is_derived_from_meminfo(self) -> None:
        completed = subprocess.run(  # noqa: S603 - repo script, literal arguments
            [str(LIMITER), "--advisory", "--", "/bin/sh", "-c", "ulimit -v"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        reported = completed.stdout.strip().splitlines()[-1]
        assert re.fullmatch(r"\d+", reported), f"expected a KiB address-space cap, got {reported!r}"

        expected_kib = _mem_total_kib() * 5 // 8
        ratio = int(reported) / expected_kib
        assert abs(ratio - 1.0) <= IEC_ROUNDING_TOLERANCE, (
            f"the memory cap is {reported} KiB, not five-eighths of this host's "
            f"{_mem_total_kib()} KiB of RAM ({expected_kib} KiB): it is not derived from the host"
        )

    def test_no_address_cap_leaves_the_address_space_unlimited(self) -> None:
        """A threaded job opts out: under the cap its thread stacks and malloc arenas fail to map."""
        completed = subprocess.run(  # noqa: S603 - repo script, literal arguments
            [str(LIMITER), "--advisory", "--no-address-cap", "--", "/bin/sh", "-c", "ulimit -v"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert completed.stdout.strip().splitlines()[-1] == "unlimited"
        assert "--no-address-cap" in completed.stderr, "the opt-out must be announced, not silent"


@pytest.mark.skipif(
    not _user_manager_is_available(), reason="requires a usable systemd user manager"
)
def test_sighup_stops_the_enforced_unit_and_its_process() -> None:
    unit_name = f"reslimit-sighup-test-{os.getpid()}-{time.time_ns()}"
    command = [
        str(LIMITER),
        "--name",
        unit_name,
        "--timeout",
        "3h",
        "--",
        "sleep",
        "600",
    ]
    wrapper = subprocess.Popen(  # noqa: S603 - repo script, literal arguments
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sleep_pid = 0
    try:
        started_deadline = time.monotonic() + 15
        while time.monotonic() < started_deadline:
            if wrapper.poll() is not None:
                stdout, stderr = wrapper.communicate()
                pytest.fail(f"wrapper exited before its unit started: {stdout}{stderr}")
            active = _systemctl_user("is-active", "--quiet", unit_name).returncode == 0
            if active:
                main_pid = _systemctl_user(
                    "show", "-p", "MainPID", "--value", unit_name
                ).stdout.strip()
                sleep_pid = int(main_pid)
                if sleep_pid > 0:
                    break
            time.sleep(0.05)
        assert sleep_pid > 0, f"unit {unit_name} did not start"

        wrapper.send_signal(signal.SIGHUP)

        assert _wait_for_systemd_state(unit_name, "inactive", timeout=5), (
            f"unit {unit_name} remained active after wrapper SIGHUP"
        )
        process_deadline = time.monotonic() + 5
        while _process_exists(sleep_pid) and time.monotonic() < process_deadline:
            time.sleep(0.05)
        assert not _process_exists(sleep_pid), f"sleep process {sleep_pid} survived wrapper SIGHUP"

        stdout, stderr = wrapper.communicate(timeout=5)
        assert wrapper.returncode == 129, (
            f"expected SIGHUP status 129, got {wrapper.returncode}: {stdout}{stderr}"
        )
        assert "TIMEOUT" not in stderr, stderr
        assert "OOM-KILLED" not in stderr, stderr
    finally:
        _systemctl_user("stop", unit_name)
        if wrapper.poll() is None:
            wrapper.send_signal(signal.SIGTERM)
        try:
            wrapper.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            wrapper.kill()
            wrapper.communicate()
        _systemctl_user("reset-failed", unit_name)
