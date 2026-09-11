"""Pin that the two flags in this repo with irreversible effects cannot be reached by prefix.

argparse abbreviates long options by default, so ``--update`` also answers to ``--u``, ``--up`` and
``--upd`` unless the parser is built with ``allow_abbrev=False``. That default is harmless for a
flag naming an output directory and is not harmless for the two here:

  * ``scripts/canary_manifest.py --update`` overwrites the tripwire baseline. Clearing canary drift
    is the owner's call, never an agent's, and a three-character synonym for it is a hazard in a
    repo where many agents run shell commands.
  * ``reward_hacking/jagged/sweep.py submit --resubmit`` overwrites a saved Bedrock batch handle,
    which means a second bill for inference already paid for plus an orphaned first job.

Both guards are one keyword argument, which is exactly the kind of thing a future reformat drops
without anybody noticing, so each is pinned from two sides: the abbreviation must be rejected, and
the full flag must still work. Pinning only the rejection would stay green if the flag were deleted
outright.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reward_hacking.jagged import sweep

REPO_ROOT = Path(__file__).resolve().parents[1]
CANARY_SCRIPT = REPO_ROOT / "scripts" / "canary_manifest.py"

# Every proper prefix of --update that argparse would otherwise accept.
UPDATE_ABBREVIATIONS = ["--u", "--up", "--upd", "--updat"]

# Only the unambiguous prefixes: --re also matches --repeats, so argparse rejects it either way.
RESUBMIT_ABBREVIATIONS = ["--res", "--resu", "--resub", "--resubmi"]

SUBMIT_REQUIRED_ARGS = ["--items", "/nonexistent/items.json", "--max-new-tokens", "64"]


def _run_canary(args: list[str], manifest: Path) -> subprocess.CompletedProcess[str]:
    """Run the canary script against a throwaway manifest, never the real baseline.

    The redirect is belt and braces for the failure this module exists to catch: if the guard is
    ever dropped, the abbreviation parses and the script writes a baseline, and this keeps that
    write inside tmp_path where it can be asserted about instead of landing on canary/.
    """
    return subprocess.run(  # noqa: S603 - repo-local script, interpreter and tmp paths
        [sys.executable, str(CANARY_SCRIPT), "--manifest", str(manifest), *args],
        capture_output=True,
        text=True,
        check=False,
    )


class TestTheCanaryBaselineWriterIsNotReachableByPrefix:
    """``--update`` overwrites the tripwire baseline, so no shorter spelling may reach it."""

    def test_the_flag_this_module_pins_still_exists(self) -> None:
        source = CANARY_SCRIPT.read_text()
        assert '"--update"' in source, (
            "canary_manifest.py no longer declares --update. Fix this module's list rather than "
            "letting a rename silently disarm the guard."
        )

    @pytest.mark.parametrize("abbreviation", UPDATE_ABBREVIATIONS)
    def test_an_abbreviation_is_rejected_and_writes_nothing(
        self, abbreviation: str, tmp_path: Path
    ) -> None:
        manifest = tmp_path / "throwaway-manifest.json"

        result = _run_canary([abbreviation], manifest)

        assert result.returncode == 2, (
            f"`canary_manifest.py {abbreviation}` exited {result.returncode}, not 2. argparse "
            "abbreviation is back and the baseline overwrite has a short synonym again.\n"
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "unrecognized arguments" in result.stderr
        assert not manifest.exists(), f"{abbreviation} wrote a baseline; it must not parse at all"

    def test_the_full_flag_still_writes_a_baseline(self, tmp_path: Path) -> None:
        manifest = tmp_path / "throwaway-manifest.json"

        result = _run_canary(["--update"], manifest)

        assert result.returncode == 0, f"--update failed: {result.stderr}"
        written = json.loads(manifest.read_text())
        assert written, "--update wrote an empty baseline"
        assert all(len(digest) == 64 for digest in written.values()), "not sha256 hex digests"

    def test_a_bare_run_reports_drift_against_a_baseline_it_did_not_write(
        self, tmp_path: Path
    ) -> None:
        """The read-only path, which is what ``make canary-check`` runs, must stay unaffected."""
        manifest = tmp_path / "throwaway-manifest.json"
        manifest.write_text(json.dumps({"/nonexistent/watched/file": "0" * 64}) + "\n")

        result = _run_canary([], manifest)

        assert result.returncode == 1, f"expected drift, got {result.returncode}: {result.stderr}"
        assert "REMOVED /nonexistent/watched/file" in result.stderr


class TestTheJaggedResubmitFlagIsNotReachableByPrefix:
    """``submit --resubmit`` buys inference twice, so no shorter spelling may reach it."""

    @pytest.mark.parametrize("abbreviation", RESUBMIT_ABBREVIATIONS)
    def test_an_abbreviation_is_rejected(self, abbreviation: str) -> None:
        with pytest.raises(SystemExit) as exit_info:
            sweep.parse_args(["submit", *SUBMIT_REQUIRED_ARGS, abbreviation])

        assert exit_info.value.code == 2, (
            f"`sweep.py submit {abbreviation}` parsed instead of erroring; abbreviation is back "
            "and a second Bedrock bill has a short synonym again."
        )

    def test_the_full_flag_still_parses(self) -> None:
        parsed = sweep.parse_args(["submit", *SUBMIT_REQUIRED_ARGS, "--resubmit"])

        assert parsed.resubmit is True

    def test_it_stays_off_by_default(self) -> None:
        parsed = sweep.parse_args(["submit", *SUBMIT_REQUIRED_ARGS])

        assert parsed.resubmit is False

    def test_the_guard_did_not_leak_onto_the_collect_subcommand(self) -> None:
        """collect spends nothing, so it keeps argparse's default; recorded here on purpose."""
        parsed = sweep.parse_args(
            [
                "collect",
                "--items",
                "/nonexistent/items.json",
                "--handle-dir",
                "/nonexistent/handles",
                "--rep",
                "3",
            ]
        )

        assert parsed.repeats == 3
