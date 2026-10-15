"""Pin that the flag in this repo with an irreversible effect cannot be reached by prefix.

argparse abbreviates long options by default. That default is harmless for a flag naming an output
directory and is not harmless for this one:

  * ``reward_hacking/jagged/sweep.py submit --resubmit`` overwrites a saved Bedrock batch handle,
    which means a second bill for inference already paid for plus an orphaned first job.

The guard is one keyword argument, which is exactly the kind of thing a future reformat drops
without anybody noticing, so it is pinned from two sides: the abbreviation must be rejected, and
the full flag must still work. Pinning only the rejection would stay green if the flag were deleted.
"""

from __future__ import annotations

import pytest

from reward_hacking.jagged import sweep

# Only the unambiguous prefixes: --re also matches --repeats, so argparse rejects it either way.
RESUBMIT_ABBREVIATIONS = ["--res", "--resu", "--resub", "--resubmi"]

SUBMIT_REQUIRED_ARGS = ["--items", "/nonexistent/items.json", "--max-new-tokens", "64"]


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
