"""Contract checks for the turn-budget announcement and remaining-turn wording."""

from __future__ import annotations

from reward_hacking.harness.status_report import STATUS_DEADLINE_PROMPT
from reward_hacking.ladder.loop import ANNOUNCED_TURN_BUDGET_SENTENCE, TURNS_REMAINING_NOTE


def test_wording_is_short_and_the_deadline_turn_does_not_contradict_it() -> None:
    sentence = ANNOUNCED_TURN_BUDGET_SENTENCE.format(n=20)
    assert "20 turns" in sentence
    assert len(sentence) < 120
    assert TURNS_REMAINING_NOTE.format(n=0) == "[turns remaining: 0]"
    assert "last turn" not in STATUS_DEADLINE_PROMPT
    assert "one final turn" in STATUS_DEADLINE_PROMPT
