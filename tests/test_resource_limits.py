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

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LIMITER = REPO_ROOT / "scripts" / "resource-limits.sh"

# `numfmt --to=iec` keeps three significant digits, so the cap is a rounded spelling.
IEC_ROUNDING_TOLERANCE = 0.02


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
