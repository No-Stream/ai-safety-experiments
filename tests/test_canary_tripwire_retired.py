"""Pin retirement of the local execution/configuration tripwire.

The remaining exact-value canary detector belongs to the privacy scanner and stays active.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RETIRED_MAKE_TARGETS = ("canary-check", "canary-update")


class TestHostExecutionTripwireRetirement:
    @pytest.mark.parametrize("target", RETIRED_MAKE_TARGETS)
    def test_retired_make_target_is_unavailable(self, target: str) -> None:
        result = subprocess.run(  # noqa: S603 - fixed Make target from this module's constants
            ["make", "--no-print-directory", "--dry-run", target],  # noqa: S607 - Make is resolved from PATH
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2
        assert f"No rule to make target '{target}'" in result.stderr

    def test_host_manifest_script_is_removed(self) -> None:
        assert not (REPO_ROOT / "scripts" / "canary_manifest.py").exists()

    def test_docs_say_no_host_execution_drift_monitor_is_active(self) -> None:
        isolation_docs = (REPO_ROOT / "docs" / "episode-isolation.md").read_text()

        assert "execution/configuration tripwire has been retired" in isolation_docs

    def test_privacy_scan_still_runs_the_secret_scanner(self) -> None:
        result = subprocess.run(
            ["make", "--no-print-directory", "--dry-run", "privacy-scan"],  # noqa: S607 - Make is resolved from PATH
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "scripts/scan_secrets.py" in result.stdout
