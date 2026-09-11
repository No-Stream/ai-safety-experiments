"""Pin `games.framing_stimulus` to `sociology.decoupled_stimulus`, and the ladder copies to the ladder.

The games framing sweep now renders counterpart clauses from a runtime file of its own, and that
file carries verbatim copies of the six authored ladder rungs so a games cell and a sociology cell
under one rung are provably one text. `games` may not import `sociology` (the training-side package
does not depend on the observer study), so the two modules re-derive the same three shared strings
independently, and this file is the only thing standing between that and a silent drift: a rung whose
opening, simultaneity phrase or decoupling tail differed between the two packages would render, grade
and summarise in both while measuring two conditions under one name.

Both files these checks read are gitignored authored stimulus, so each check skips when its file is
absent rather than failing a fresh clone, and the pin between the two MODULES holds unconditionally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from games import framing_stimulus
from games.framing_stimulus import FRAMINGS_PATH, load_framings
from sociology.decoupled_stimulus import (
    AUTHORED_RUNGS,
    BRIEFING_PHRASE,
    CLAUSE_PREFIX,
    DECOUPLING_TAILS,
    RUNG_SAME_CHECKPOINT,
    STIMULUS_PATH,
    clause_for,
    load_stimulus,
)

if TYPE_CHECKING:
    from pathlib import Path


def _require(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"{path} is not on this machine (fresh clone)")


class TestTheTwoModulesShareTheirDerivedConstants:
    def test_the_opening_the_simultaneity_phrase_and_both_tails_are_one_text_each(self) -> None:
        assert framing_stimulus.CLAUSE_PREFIX == CLAUSE_PREFIX
        assert framing_stimulus.BRIEFING_PHRASE == BRIEFING_PHRASE
        assert framing_stimulus.DECOUPLING_TAILS == DECOUPLING_TAILS

    def test_both_default_paths_sit_under_gitignored_scratch(self) -> None:
        assert FRAMINGS_PATH.parts[:2] == ("docs", "scratch")
        assert STIMULUS_PATH.parts[:2] == ("docs", "scratch")


class TestTheLadderCopiesInTheGamesFileAreTheLadder:
    def test_every_copied_rung_is_byte_identical_to_the_rung_it_copies(self) -> None:
        """A drifted copy would read as the sociology rung's result under a games label."""
        _require(FRAMINGS_PATH)
        _require(STIMULUS_PATH)
        stimulus = load_stimulus()
        clauses = load_framings().clauses
        copied = [rung for rung in AUTHORED_RUNGS if rung in clauses]
        assert copied, (
            f"{FRAMINGS_PATH} carries none of the ladder rungs {list(AUTHORED_RUNGS)}, so no games "
            f"cell can be read against a sociology one"
        )
        for rung in copied:
            assert clauses[rung] == clause_for(rung, stimulus), rung

    def test_the_same_weights_rung_is_not_copied_under_a_second_id(self) -> None:
        """That rung IS the registered clause, so a copy would sample one anchor under two labels."""
        _require(FRAMINGS_PATH)
        assert RUNG_SAME_CHECKPOINT not in load_framings().clauses
