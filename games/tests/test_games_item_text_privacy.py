"""No survey or instrument item text in any tracked games/ file. The remote is public.

The owner's ruling (2026-08-21) extends the repo's privacy rule: no benchmark or instrument item
text may ever be committed -- third-party published instruments included, regardless of copyright
or licence status, and our own authored items included, because they too will be run on future
models. A committed item becomes scraper training data, and a model that has memorized an
instrument invalidates every future measurement made with it.

Everything this guard matches against is loaded at run time from the local gitignored sources
(``games/data/survey/*.json`` and the retrieval note under ``docs/scratch/``), through
``scripts.scan_secrets``. Not one forbidden phrase is spelled out in this file, and that is the
load-bearing design decision: the first version of this guard, in a wave-2 worktree, listed the
"distinctive openings" of the instruments it protected -- committing seven complete published
items verbatim inside the guard itself. A guard that lists the contraband IS the leak.

On a fresh clone neither source exists and the guard skips, saying so. On this machine the skip is
kept honest two ways: the floor test below fails (not skips) if the retrieval note is present but
extraction collapses, and ``tests/test_scan_secrets.py`` sweeps every committable file with the
same detector inside ``make test``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.scan_secrets import (
    InstrumentSources,
    collect_instrument_sources,
    instrument_text_findings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The retrieval note carries seventy published items across five Likert instruments, plus the nine
# triple-dominance payoff triples and the slider's endpoint table. Extraction yielding less than
# this is extraction rot, not a smaller instrument -- and a guard whose phrase list quietly
# shrank to nothing would pass everything while reporting success.
NOTE_PHRASE_FLOOR = 300
NOTE_PAYOFF_ITEM_FLOOR = 9


def _local_sources() -> InstrumentSources:
    return collect_instrument_sources(REPO_ROOT)


def _tracked_games_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "games"],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / name for name in listed.stdout.split("\0") if name]


class TestNoInstrumentItemTextIsTracked:
    def test_extraction_meets_the_notes_floor(self) -> None:
        """The note's content is fixed, so a shrunken extraction is rot in the parser, not data."""
        sources = _local_sources()
        note_loaded = any(path.endswith(".md") for path in sources.source_paths)
        if not note_loaded:
            pytest.skip(
                "the survey retrieval note is not on this machine (fresh clone); the sweep in "
                "tests/test_scan_secrets.py still runs, armed with whatever sources exist"
            )
        assert len(sources.phrases) >= NOTE_PHRASE_FLOOR, (
            f"only {len(sources.phrases)} phrases extracted from {sources.source_paths}; the "
            f"note alone should yield at least {NOTE_PHRASE_FLOOR}. The item-text guard is "
            f"running near-empty, which passes everything -- fix the extraction, do not lower "
            f"the floor."
        )
        assert len(sources.payoff_items) >= NOTE_PAYOFF_ITEM_FLOOR, (
            f"only {len(sources.payoff_items)} allocation payoff items extracted; the note's "
            f"triple-dominance table alone carries {NOTE_PAYOFF_ITEM_FLOOR}."
        )

    def test_no_tracked_games_file_carries_item_text(self) -> None:
        """Every stem, anchor and payoff-table phrase stays out of everything git tracks here.

        Tracked files may carry item ids, family names, subscale membership, keying positions and
        scoring formulas -- everything about an instrument except its words and its payoff tables.
        """
        sources = _local_sources()
        if not sources.source_paths:
            pytest.skip(
                "no local instrument source exists (fresh clone): neither games/data/survey/ nor "
                "the docs/scratch retrieval note. There is nothing to match against; on machines "
                "that hold the item text this guard is armed and this skip does not happen."
            )
        if not sources.is_armed:
            pytest.skip(
                f"instrument sources {sources.source_paths} exist but carry no item text yet "
                f"(placeholder file); the floor test above is what fails if a populated source "
                f"stops extracting."
            )
        findings = [
            finding
            for path in _tracked_games_files()
            for finding in instrument_text_findings(
                str(path.relative_to(REPO_ROOT)),
                path.read_text(encoding="utf-8", errors="replace"),
                sources,
            )
        ]
        assert findings == [], (
            "instrument item text found in tracked files (fingerprints only; match them against "
            "the local sources):\n" + "\n".join(str(finding) for finding in findings)
        )
