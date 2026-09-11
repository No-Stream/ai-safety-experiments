"""Pin what counts as an answer, since the parser's verdict is what becomes the reward.

Offline and CPU-only. Two families of behaviour are being fixed here, and they pull in
opposite directions on purpose.

The lenient half: a completion that restates its choice, wraps it in whitespace, or uses a
different case still parses, because none of that is a decision. The strict half: a choice that
only ever appears inside the thinking block, a move sequence of the wrong length, a keep
outside the endowment, or a label the corpus never offered all return None and are paid the
parse penalty -- inferring intent from prose is exactly the judgement that must stay out of a
reward path.

:class:`TestAnActionInsideThinkingIsNotAnAnswer` is the one with teeth. A completion whose only
`<action>` tag sits inside the thinking block must score as a parse failure, and the test also
asserts what the same string parses to *without* the cut -- the opposite action. Skip the cut
and every reward in the run is computed from the model's discarded first thought, at no point
looking like a bug.
"""

from __future__ import annotations

import pytest

from games.parsing import (
    THEORY_LABELS,
    parse_action,
    parse_action_sequence,
    parse_split,
    parse_theory,
    strip_thinking,
)
from games.payoffs import COOPERATE, DEFECT

COOPERATIVE_LABEL = "HOLD"
DEFECTING_LABEL = "SLASH"
LABELS: dict[str, str] = {
    "label_a": COOPERATIVE_LABEL,
    "label_b": DEFECTING_LABEL,
    "coop_label": COOPERATIVE_LABEL,
}
SWAPPED_LABELS: dict[str, str] = {
    "label_a": COOPERATIVE_LABEL,
    "label_b": DEFECTING_LABEL,
    "coop_label": DEFECTING_LABEL,
}


class TestStripThinking:
    def test_visible_text_starts_after_the_closing_tag(self) -> None:
        visible, truncated = strip_thinking("<think>weighing it up</think>\n<action>HOLD</action>")
        assert visible == "\n<action>HOLD</action>"
        assert truncated is False

    def test_prefilled_template_emits_only_the_closing_tag(self) -> None:
        """The Qwen3.5/3.8 case: the prompt ends with `<think>`, so the completion has no open."""
        visible, truncated = strip_thinking(
            "weighing it up</think>\n<action>SLASH</action>", prefilled_think=True
        )
        assert visible == "\n<action>SLASH</action>"
        assert truncated is False

    def test_the_last_closing_tag_wins(self) -> None:
        visible, truncated = strip_thinking("<think>a</think>mid</think>final")
        assert visible == "final"
        assert truncated is False

    def test_unclosed_thinking_is_truncated_with_no_visible_text(self) -> None:
        visible, truncated = strip_thinking("<think>reasoning that ran out of budget")
        assert visible == ""
        assert truncated is True

    def test_prefilled_thinking_with_no_markers_at_all_is_truncated(self) -> None:
        visible, truncated = strip_thinking(
            "reasoning that ran out of budget", prefilled_think=True
        )
        assert visible == ""
        assert truncated is True

    def test_a_model_without_a_thinking_template_returns_the_whole_completion(self) -> None:
        """tiny-gpt2 and the CPU smoke path emit no thinking markers; that is not truncation."""
        visible, truncated = strip_thinking("<action>HOLD</action>")
        assert visible == "<action>HOLD</action>"
        assert truncated is False


class TestAnActionInsideThinkingIsNotAnAnswer:
    def test_the_tag_before_the_close_is_ignored(self) -> None:
        completion = "<think>leaning <action>SLASH</action></think><action>HOLD</action>"
        visible, truncated = strip_thinking(completion)
        assert visible == "<action>HOLD</action>"
        assert truncated is False
        assert parse_action(visible, **LABELS) == COOPERATE

    def test_a_choice_that_never_leaves_the_thinking_block_is_a_parse_failure(self) -> None:
        """The discriminating case: skipping the cut would score the discarded thought."""
        completion = "<think><action>SLASH</action></think>I could not decide."
        visible, _ = strip_thinking(completion)
        assert parse_action(visible, **LABELS) is None
        assert parse_action(completion, **LABELS) == DEFECT


class TestParseAction:
    def test_the_last_tag_wins(self) -> None:
        assert (
            parse_action("<action>HOLD</action> wait: <action>SLASH</action>", **LABELS) == DEFECT
        )

    def test_label_matching_ignores_case_and_surrounding_whitespace(self) -> None:
        assert parse_action("<action>  hold\n</action>", **LABELS) == COOPERATE

    def test_the_cooperative_label_can_be_either_of_the_two(self) -> None:
        """Counterbalancing means "C" is whichever label the row nominated, not a fixed one."""
        assert parse_action("<action>SLASH</action>", **SWAPPED_LABELS) == COOPERATE
        assert parse_action("<action>HOLD</action>", **SWAPPED_LABELS) == DEFECT

    @pytest.mark.parametrize(
        "visible",
        [
            "I choose HOLD.",
            "<action>WAIT</action>",
            "<action></action>",
            "<action>HOLD",
            "",
        ],
    )
    def test_missing_or_unknown_content_returns_none(self, visible: str) -> None:
        assert parse_action(visible, **LABELS) is None

    def test_a_coop_label_outside_the_two_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="must be one of the two labels"):
            parse_action(
                "<action>HOLD</action>", label_a="HOLD", label_b="SLASH", coop_label="WAIT"
            )

    def test_two_identical_labels_raise(self) -> None:
        with pytest.raises(ValueError, match="label_a and label_b must differ"):
            parse_action("<action>HOLD</action>", label_a="HOLD", label_b="hold", coop_label="HOLD")


class TestParseActionSequence:
    def test_one_tag_per_round_in_order(self) -> None:
        visible = (
            "Round 1: <action>HOLD</action> "
            "Round 2: <action>HOLD</action> "
            "Round 3: <action>SLASH</action>"
        )
        assert parse_action_sequence(visible, n_rounds=3, **LABELS) == [
            COOPERATE,
            COOPERATE,
            DEFECT,
        ]

    @pytest.mark.parametrize("n_rounds", [2, 4])
    def test_the_wrong_number_of_tags_fails_the_whole_sequence(self, n_rounds: int) -> None:
        visible = "<action>HOLD</action><action>HOLD</action><action>SLASH</action>"
        assert parse_action_sequence(visible, n_rounds=n_rounds, **LABELS) is None

    def test_one_unknown_label_fails_the_whole_sequence(self) -> None:
        """A partial parse would mean inventing the missing round before simulating it."""
        visible = "<action>HOLD</action><action>MAYBE</action><action>SLASH</action>"
        assert parse_action_sequence(visible, n_rounds=3, **LABELS) is None

    def test_no_tags_at_all_returns_none(self) -> None:
        assert parse_action_sequence("I would keep going.", n_rounds=5, **LABELS) is None

    def test_a_non_positive_round_count_raises(self) -> None:
        with pytest.raises(ValueError, match="n_rounds must be positive"):
            parse_action_sequence("<action>HOLD</action>", n_rounds=0, **LABELS)


class TestParseSplit:
    @pytest.mark.parametrize("kept", [0, 3, 10])
    def test_a_keep_inside_the_endowment_reads_back(self, kept: int) -> None:
        assert parse_split(f"<keep>{kept}</keep>", endowment=10) == kept

    def test_the_last_tag_wins(self) -> None:
        assert parse_split("<keep>2</keep> on reflection <keep>7</keep>", endowment=10) == 7

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        assert parse_split("<keep> 4 </keep>", endowment=10) == 4

    @pytest.mark.parametrize("written", ["11", "-1", "3.5", "half", "", "1,000", "seven"])
    def test_out_of_range_or_non_integer_keeps_are_parse_failures(self, written: str) -> None:
        assert parse_split(f"<keep>{written}</keep>", endowment=10) is None

    def test_a_missing_tag_returns_none(self) -> None:
        assert parse_split("I would keep four.", endowment=10) is None

    def test_a_non_positive_endowment_raises(self) -> None:
        with pytest.raises(ValueError, match="endowment must be positive"):
            parse_split("<keep>1</keep>", endowment=0)


class TestParseTheory:
    @pytest.mark.parametrize(
        ("written", "expected"),
        [
            ("Causal decision theory", "CDT"),
            ("cdt", "CDT"),
            ("I lean CDT-ish these days", "CDT"),
            ("evidential decision theory", "EDT"),
            ("EDT", "EDT"),
            ("functional decision theory", "FDT"),
            ("fdt", "FDT"),
            ("updateless decision theory", "UDT"),
            ("UDT", "UDT"),
            ("just two-box and stop worrying", "other"),
            ("", "other"),
        ],
    )
    def test_the_classification_table(self, written: str, expected: str) -> None:
        assert parse_theory(f"<theory>{written}</theory>") == expected

    def test_acausal_does_not_read_as_causal(self) -> None:
        """The word acausal is FDT vocabulary, so substring matching must not read it as CDT."""
        assert parse_theory("<theory>I care about acausal influence</theory>") == "other"

    def test_functional_wins_over_updateless_when_both_appear(self) -> None:
        visible = "<theory>functional decision theory, in its updateless form</theory>"
        assert parse_theory(visible) == "FDT"

    def test_the_last_tag_wins(self) -> None:
        assert parse_theory("<theory>causal</theory> though really <theory>FDT</theory>") == "FDT"

    def test_a_missing_tag_returns_none(self) -> None:
        assert parse_theory("Decision theory is a rabbit hole.") is None

    def test_every_produced_label_is_declared(self) -> None:
        produced = {
            parse_theory(f"<theory>{written}</theory>")
            for written in (
                "causal",
                "evidential",
                "functional",
                "updateless",
                "nothing",
                "CDT, not EDT",
            )
        }
        assert produced == set(THEORY_LABELS)


class TestATagNamingTwoRivalTheoriesIsNotClassifiedByDeclarationOrder:
    """The classifier used to return the first pattern matching anywhere in the tag.

    `_THEORY_PATTERNS` is ordered FDT, UDT, EDT, CDT, so "CDT, not EDT" matched the EDT pattern
    first and was recorded as EDT: the model's stated answer inverted. The error was systematic and
    directional, since "causal, not evidential" also read as EDT while "evidential, not causal"
    stayed EDT -- CDT answers with a contrast leaked one way only, toward the acausal direction the
    experiment exists to measure a shift toward. `games.evals` feeds this straight into
    `theory_counts`, which has no cross-check behind it.
    """

    @pytest.mark.parametrize(
        "written",
        [
            "CDT, not EDT",
            "causal decision theory, not evidential",
            "evidential, not causal",
            "EDT rather than CDT",
            "somewhere between EDT and FDT",
        ],
    )
    def test_a_tag_naming_rival_theories_is_ambiguous_rather_than_the_first_one_listed(
        self, written: str
    ) -> None:
        assert parse_theory(f"<theory>{written}</theory>") == "ambiguous"

    def test_ambiguous_is_a_declared_label_distinct_from_other(self) -> None:
        # "other" means the tag named no theory we know, which is a different observation from a tag
        # that named two: collapsing them would hide the frequency of each.
        assert "ambiguous" in THEORY_LABELS
        assert parse_theory("<theory>just two-box and stop worrying</theory>") == "other"

    def test_the_acausal_family_still_resolves_to_fdt(self) -> None:
        # FDT answers routinely also say "updateless"; those are near-synonyms in this taxonomy
        # rather than rivals, which is why the ordering trick existed in the first place.
        assert parse_theory("<theory>functional, in its updateless form</theory>") == "FDT"
        assert parse_theory("<theory>UDT/FDT</theory>") == "FDT"

    def test_a_restated_single_answer_is_not_ambiguous(self) -> None:
        assert parse_theory("<theory>CDT -- causal decision theory</theory>") == "CDT"
