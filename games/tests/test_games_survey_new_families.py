"""Tests for the newly registered survey families and the answer shapes they need.

Scope, and why it is a separate file from `test_games_survey.py`: that file pins the battery as it
already existed, several sessions co-edit it, and the additions here are a distinct contract -- the
one the item-authoring passes write against. What is tested is the *plumbing*, since no item of these
families exists yet: that the family names are registered and refused-by-name until their items land,
that each answer shape renders and round-trips, and that the reductions the new shapes are read
through compute what a hand calculation says they should.

Three conventions carried over from the older file, each load-bearing.

**No item text, real or plausible.** Every stem, option and vocabulary word here is transparently
synthetic ("wordalpha", "synthetic option 1"). The remote is public, our own items count as items
under the owner's 2026-08-21 ruling, and a plausible-looking placeholder in a tracked test is how a
draft item ends up committed by accident.

**Arithmetic is pinned against numbers worked out on paper**, with the working in a comment, never
against whatever the code currently returns. A win rate pooled across items instead of reduced per
item, or a cheap-talk match rate whose denominator counted the unparsed renders, both produce
perfectly plausible numbers -- so only a hand-computed expectation can catch them.

**The bit-identical pin at the bottom is the whole warrant for calling this change additive.** It
digests the spec fields AND every rendered prompt of the battery that existed before these families
were registered, both from synthetic data and from the real local files. Both digests were checked
equal against a `git archive HEAD` checkout of the pre-change module before being written down here.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import games.evals as evals_module
import games.survey as survey_module
from games.evals import EvalConfig
from games.survey import (
    ACT_TAG,
    ANNOUNCE_TAG,
    AUTHORED_FILENAME,
    AUTHORED_ITEM_SPECS,
    COUNTERPART_AI,
    COUNTERPART_ARM_MISSING,
    COUNTERPART_HUMAN,
    COUNTERPART_NOMINAL_ANSWER,
    COUNTERPART_UNSPECIFIED,
    DIFFERENCEABLE_KINDS,
    ELICITATION_BLOCKS_KEY,
    FAMILIES,
    FAMILIES_AWAITING_ITEMS,
    FAMILIES_BREADTH_ONLY,
    FAMILIES_WITH_SHARED_ELICITATION,
    FAMILY_COUNTERPART_PAIRS,
    FAMILY_DECEPTION_CHEAP_TALK,
    FAMILY_GRADED_DIMENSION_AWARENESS,
    FAMILY_NEGATIVE_CONTROL,
    FAMILY_RISK_PREFERENCE,
    FAMILY_SELF_CHARACTERISATION,
    FAMILY_TRUST_RECIPROCITY,
    FAMILY_VALUES_FORCED_CHOICE,
    NEUTRAL_TWIN_ID_SUFFIX,
    NON_SOCIAL_LABEL_PREFIX,
    NUMERIC_EXAMPLE_PERCENTS,
    NUMERIC_EXAMPLE_ROTATION,
    PLANNED_FAMILY_ITEM_COUNTS,
    PLANNED_FAMILY_TWIN_COUNTS,
    PUBLISHED_INSTRUMENTS,
    SCHEMA_VERSION,
    STANCE_TAG,
    SURVEY_ALLOCATION,
    SURVEY_CHEAP_TALK,
    SURVEY_CHOICE,
    SURVEY_KINDS,
    SURVEY_LIKERT,
    SURVEY_NUMERIC,
    SURVEY_ORDERED_CHOICE,
    SURVEY_RECORD_FIELDS,
    SURVEY_TAGGED,
    TAG_EXAMPLE_PLACEHOLDER,
    TAGGED_ITEM_MISSING,
    TAGGED_NOTHING_PARSED,
    TAGGED_VOCABULARY_WIDTH_DIFFERS,
    TAGGED_VOCABULARY_WIDTH_UNSTATED,
    TAGGED_VOCABULARY_WORDS_DIFFER,
    TIER_BREADTH,
    TIER_CORE,
    WORDING_AS_PUBLISHED,
    WORDING_NEUTRAL_TWIN,
    AuthoredItemSpec,
    SurveyItem,
    assert_every_counterpart_pair_is_complete,
    assert_every_twin_pairs,
    assert_ordered_choice_ladders_are_commensurable,
    assert_twin_id_derives_from_parent,
    battery_orders,
    cheap_talk_readings,
    choice_response_distributions,
    counterpart_gaps,
    families_with_items,
    forced_choice_prefix_win_rate,
    forced_choice_win_rates,
    load_authored_items,
    numeric_example_rotation,
    numeric_examples_for_bound,
    parse_survey_answer,
    per_item_scores,
    render_survey_prompt,
    subscale_composites,
    survey_battery,
    survey_record_fields,
    tagged_distribution_distance,
    tagged_readings,
    wording_gap,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = REPO_ROOT / "games" / "data" / "survey"

NEW_FAMILIES: tuple[str, ...] = (
    FAMILY_TRUST_RECIPROCITY,
    FAMILY_RISK_PREFERENCE,
    FAMILY_DECEPTION_CHEAP_TALK,
    FAMILY_VALUES_FORCED_CHOICE,
    FAMILY_GRADED_DIMENSION_AWARENESS,
    FAMILY_COUNTERPART_PAIRS,
)

FIVE_ANCHORS = (
    "synthetic anchor one",
    "synthetic anchor two",
    "synthetic anchor three",
    "synthetic anchor four",
    "synthetic anchor five",
)
# Two- and three-word closed vocabularies, deliberately nonsense so no test string could ever be
# mistaken for a drafted item's wording.
TWO_WORD_VOCABULARY = ("wordalpha", "wordbeta")
# Canonical order deliberately unalphabetical: a reading that ordered its words by name instead of by
# menu position would pass every share assertion below and still print an unreadable ladder.
THREE_WORD_VOCABULARY = ("wordzeta", "wordalpha", "wordmu")
THREE_RUNG_OPTIONS = ("synthetic rung 1", "synthetic rung 2", "synthetic rung 3")


def cheap_talk_item(
    item_id: str = "deception-01",
    *,
    counterpart: str = COUNTERPART_UNSPECIFIED,
    counterpart_pair: str | None = None,
) -> SurveyItem:
    """Build a synthetic cheap-talk item: one stem, one vocabulary, two tags."""
    return SurveyItem(
        item_id=item_id,
        family=FAMILY_DECEPTION_CHEAP_TALK,
        instrument="synthetic-deception",
        kind=SURVEY_CHEAP_TALK,
        stem=f"Synthetic cheap-talk situation for {item_id}.",
        construct="A synthetic construct.",
        expected_direction="No prediction; this item exists only in a test.",
        subscale="announcement-versus-action",
        counterpart=counterpart,
        counterpart_pair=counterpart_pair,
        tag_vocabulary=TWO_WORD_VOCABULARY,
    )


def ordered_choice_item(
    item_id: str = "risk-01",
    *,
    options: tuple[str, ...] = THREE_RUNG_OPTIONS,
    subscale: str = "gamble-ladder",
) -> SurveyItem:
    """Build a synthetic ordered-choice item: a ladder whose position is the score."""
    return SurveyItem(
        item_id=item_id,
        family=FAMILY_RISK_PREFERENCE,
        instrument="synthetic-risk",
        kind=SURVEY_ORDERED_CHOICE,
        stem=f"Synthetic gamble ladder for {item_id}.",
        construct="A synthetic construct.",
        expected_direction="No prediction; this item exists only in a test.",
        subscale=subscale,
        options=options,
    )


def labelled_choice_item(
    item_id: str = "values-01", *, labels: tuple[str, ...] = ("joint-gain", "own-gain")
) -> SurveyItem:
    """Build a synthetic forced-choice item whose options carry value labels."""
    return SurveyItem(
        item_id=item_id,
        family=FAMILY_VALUES_FORCED_CHOICE,
        instrument="synthetic-values",
        kind=SURVEY_CHOICE,
        stem=f"Synthetic forced choice for {item_id}.",
        construct="A synthetic construct.",
        expected_direction="No prediction; this item exists only in a test.",
        subscale="value-ordering",
        options=tuple(f"synthetic option {index}" for index in range(1, len(labels) + 1)),
        option_labels=labels,
    )


def trust_likert_item(
    item_id: str,
    *,
    counterpart: str = COUNTERPART_UNSPECIFIED,
    counterpart_pair: str | None = None,
    reverse_keyed: bool = False,
) -> SurveyItem:
    """Build a synthetic five-point Likert item in the trust family."""
    return SurveyItem(
        item_id=item_id,
        family=FAMILY_TRUST_RECIPROCITY,
        instrument="synthetic-trust",
        kind=SURVEY_LIKERT,
        stem=f"Synthetic trust statement for {item_id}.",
        construct="A synthetic construct.",
        expected_direction="No prediction; this item exists only in a test.",
        subscale="trust-attitude",
        counterpart=counterpart,
        counterpart_pair=counterpart_pair,
        reverse_keyed=reverse_keyed,
        options=FIVE_ANCHORS,
    )


def trust_numeric_item(item_id: str = "trust-send-01", *, swapped: bool = True) -> SurveyItem:
    """Build a synthetic trust-game send item: a bounded integer in the keep tag, no game predicted."""
    return SurveyItem(
        item_id=item_id,
        family=FAMILY_TRUST_RECIPROCITY,
        instrument="synthetic-trust",
        kind=SURVEY_NUMERIC,
        stem=f"Synthetic send-framed stem for {item_id}.",
        stem_swapped=f"Synthetic keep-framed stem for {item_id}." if swapped else None,
        construct="A synthetic construct.",
        expected_direction="No prediction; this item exists only in a test.",
        subscale="revealed-trust",
        numeric_max=10,
    )


def forced_tag_item(
    item_id: str = "self-characterisation-01",
    *,
    vocabulary: tuple[str, ...] = TWO_WORD_VOCABULARY,
) -> SurveyItem:
    """Build a synthetic forced-tag item: one stem, one closed vocabulary, one tag."""
    return SurveyItem(
        item_id=item_id,
        family=FAMILY_SELF_CHARACTERISATION,
        instrument="synthetic-self-characterisation",
        kind=SURVEY_TAGGED,
        stem=f"Synthetic self-characterisation prompt for {item_id}.",
        construct="A synthetic construct.",
        expected_direction="No prediction; this item exists only in a test.",
        subscale="stance-under-conflict",
        tag_vocabulary=vocabulary,
    )


def tag_completion(word: str) -> str:
    """Wrap one menu word the way a compliant forced-tag completion does."""
    return f"<{STANCE_TAG}>{word}</{STANCE_TAG}>"


def record_for(
    item: SurveyItem, completion: str, *, order: tuple[int, ...] | None = None
) -> dict[str, Any]:
    """Build one record the way the eval writer does: parse a completion, merge the contract."""
    answer = parse_survey_answer(item, completion, option_order=order)
    return {
        "record": "self-report",
        "sample_index": 0,
        "option_order_name": "as-authored",
        **survey_record_fields(item, answer),
    }


class TestTheNewFamiliesAreRegisteredAndLanded:
    def test_every_new_family_is_registered(self) -> None:
        for family in NEW_FAMILIES:
            assert family in FAMILIES, family

    def test_family_names_are_descriptive_rather_than_lettered(self) -> None:
        """The plan document numbers these C, D, E, H, I; a table row must read itself instead."""
        for family in NEW_FAMILIES:
            assert "-" in family
            assert len(family) > 3, family

    def test_every_new_family_has_landed_its_items(self) -> None:
        """Landed 2026-08-22: nothing is awaiting, and every one of the six carries its items.

        The assertion this replaced was `set(NEW_FAMILIES) == set(FAMILIES_AWAITING_ITEMS)`, which was
        true for as long as the specs were being authored. It is stated as two facts rather than one so
        a failure says which half moved: still-awaiting, or landed-but-empty.
        """
        assert set(NEW_FAMILIES) & FAMILIES_AWAITING_ITEMS == set()
        assert frozenset() == FAMILIES_AWAITING_ITEMS
        for family in NEW_FAMILIES:
            assert family in families_with_items(), family

    def test_every_authored_family_declares_a_count(self) -> None:
        """A family that did not say how many items it expects could land, or shrink, at any size.

        Over every family carrying specs rather than over `FAMILIES_AWAITING_ITEMS`, which is empty
        now: a loop over an empty set is a test that cannot fail, and this file's whole premise is that
        a check nobody has watched fail is not yet a check.
        """
        authored = {spec.family for spec in AUTHORED_ITEM_SPECS}
        assert authored
        for family in authored | FAMILIES_AWAITING_ITEMS:
            assert PLANNED_FAMILY_ITEM_COUNTS[family] > 0, family

    def test_no_awaiting_family_carries_registered_specs(self) -> None:
        registered = {spec.family for spec in AUTHORED_ITEM_SPECS}
        assert registered & FAMILIES_AWAITING_ITEMS == set()

    def test_families_with_items_excludes_the_awaiting_ones(self) -> None:
        """The value a trace's meta records, so it cannot claim to have asked absent items."""
        assert families_with_items() & FAMILIES_AWAITING_ITEMS == frozenset()
        assert FAMILY_NEGATIVE_CONTROL in families_with_items()

    def test_requesting_an_awaiting_family_is_refused_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Distinct from the empty-battery message: different cause, different fix.

        Reached through a monkeypatched awaiting set since 2026-08-22, because every registered family
        has landed and nothing else triggers this branch. Deleting the test instead would have retired
        a live refusal that the next family authored against a new constant depends on.
        """
        monkeypatch.setattr(
            survey_module, "FAMILIES_AWAITING_ITEMS", frozenset({FAMILY_TRUST_RECIPROCITY})
        )
        with pytest.raises(ValueError, match="have not landed yet"):
            survey_battery(families=[FAMILY_TRUST_RECIPROCITY], data_dir=REAL_DATA_DIR)

    def test_requesting_a_landed_family_administers_it(self) -> None:
        """The positive control on the refusal above: without it, a battery that refused every family
        by name would pass that test and administer nothing at all."""
        items = survey_battery(families=[FAMILY_TRUST_RECIPROCITY], data_dir=REAL_DATA_DIR)
        assert {item.family for item in items} == {FAMILY_TRUST_RECIPROCITY}
        assert (
            len(items)
            == PLANNED_FAMILY_ITEM_COUNTS[FAMILY_TRUST_RECIPROCITY]
            + (PLANNED_FAMILY_TWIN_COUNTS[FAMILY_TRUST_RECIPROCITY])
        )

    def test_a_synthetic_item_of_a_landed_family_is_still_constructible(self) -> None:
        """Whether a family has landed governs whether the BATTERY will administer it, never whether
        the types accept it: authoring has to stay testable on both sides of a landing, which is why
        every fixture in this file builds items for these families without touching the registry."""
        assert cheap_talk_item().family in families_with_items()
        assert ordered_choice_item().family in families_with_items()
        spec = AuthoredItemSpec(
            item_id="values-99",
            family=FAMILY_VALUES_FORCED_CHOICE,
            instrument="synthetic-values",
            kind=SURVEY_CHOICE,
            construct="A synthetic construct.",
            expected_direction="No prediction; this item exists only in a test.",
            n_options=2,
            option_labels=("joint-gain", "own-gain"),
        )
        assert spec.family in FAMILIES

    def test_the_counterpart_family_is_registered_in_all_four_structures(self) -> None:
        """The counterpart arms are their own family, not filed under whichever construct each pair is
        built on: the quantity is the DIFFERENCE between a pair's arms, which is not the level its host
        family reports, and 60 arms under trust-reciprocity would have made that family's registered
        count several times what its preregistration line names.

        Four structures, and the breadth-only one is the one that fails QUIETLY if forgotten: the
        mistiering check only sweeps families in that set and `AuthoredItemSpec.tier` defaults to core,
        so an omission here turns a forgotten keyword into hours of thinking-on completions rather than
        into an error.
        """
        assert FAMILY_COUNTERPART_PAIRS in FAMILIES
        assert FAMILY_COUNTERPART_PAIRS not in FAMILIES_AWAITING_ITEMS
        assert FAMILY_COUNTERPART_PAIRS in FAMILIES_BREADTH_ONLY
        assert PLANNED_FAMILY_ITEM_COUNTS[FAMILY_COUNTERPART_PAIRS] == 60

    def test_the_counterpart_family_administers_thirty_complete_pairs(self) -> None:
        """Sixty arms in thirty pairs, one AI arm and one human arm each.

        Counted here as well as in `assert_every_counterpart_pair_is_complete`, which runs over an
        assembled battery: this one runs over the tracked registry, so a fresh clone with no local item
        file still fails if an arm was registered without its opposite. A lone arm is a LEVEL, and every
        level in this battery is confounded by the self-report over-reporting the pairs exist to cancel.
        """
        specs = [spec for spec in AUTHORED_ITEM_SPECS if spec.family == FAMILY_COUNTERPART_PAIRS]
        assert len(specs) == PLANNED_FAMILY_ITEM_COUNTS[FAMILY_COUNTERPART_PAIRS]
        by_pair = Counter(spec.counterpart_pair for spec in specs)
        assert len(by_pair) == len(specs) // 2
        assert set(by_pair.values()) == {2}
        assert Counter(spec.counterpart for spec in specs) == {
            COUNTERPART_AI: len(specs) // 2,
            COUNTERPART_HUMAN: len(specs) // 2,
        }
        assert FAMILY_COUNTERPART_PAIRS in families_with_items()

    def test_self_characterisation_is_landed_and_its_planned_count_matches_the_registry(
        self,
    ) -> None:
        """The family that GROWS rather than lands: its planned count is the number to edit when the
        six new items are registered, and it must agree with the registry until then."""
        assert FAMILY_SELF_CHARACTERISATION not in FAMILIES_AWAITING_ITEMS
        registered = sum(
            1 for spec in AUTHORED_ITEM_SPECS if spec.family == FAMILY_SELF_CHARACTERISATION
        )
        assert registered == PLANNED_FAMILY_ITEM_COUNTS[FAMILY_SELF_CHARACTERISATION]


class TestThePlannedCountGuardHasTeeth:
    """The count assertion is what catches a family that landed half its items, so watch it fail."""

    def test_a_short_family_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            survey_module,
            "PLANNED_FAMILY_ITEM_COUNTS",
            {**PLANNED_FAMILY_ITEM_COUNTS, FAMILY_NEGATIVE_CONTROL: 99},
        )
        with pytest.raises(RuntimeError, match="wrong item count"):
            survey_module._assert_planned_family_counts_hold({FAMILY_NEGATIVE_CONTROL})

    def test_a_family_with_specs_still_marked_awaiting_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            survey_module,
            "FAMILIES_AWAITING_ITEMS",
            FAMILIES_AWAITING_ITEMS | {FAMILY_NEGATIVE_CONTROL},
        )
        with pytest.raises(RuntimeError, match="wrong item count"):
            survey_module._assert_planned_family_counts_hold({FAMILY_NEGATIVE_CONTROL})

    def test_an_authored_family_with_no_planned_count_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trimmed = {
            family: count
            for family, count in PLANNED_FAMILY_ITEM_COUNTS.items()
            if family != FAMILY_NEGATIVE_CONTROL
        }
        monkeypatch.setattr(survey_module, "PLANNED_FAMILY_ITEM_COUNTS", trimmed)
        with pytest.raises(RuntimeError, match="no planned count"):
            survey_module._assert_planned_family_counts_hold({FAMILY_NEGATIVE_CONTROL})

    def test_the_registry_as_committed_passes(self) -> None:
        survey_module._assert_planned_family_counts_hold(
            {spec.family for spec in AUTHORED_ITEM_SPECS}
        )


class TestTheBreadthOnlyTierGuard:
    """`AuthoredItemSpec.tier` defaults to core, and every new family runs on the breadth leg only.

    A forgotten `tier=TIER_BREADTH` is not a wrong number, it is a bill: a breadth item is about sixty
    completion tokens on the non-deliberated leg, while a core item pays a thinking-on completion on
    BOTH legs at every checkpoint of every arm. Seventy-six items landing in core by omission would
    pass every other check in this repository.
    """

    @staticmethod
    def _land_one_trust_spec(monkeypatch: pytest.MonkeyPatch, spec: AuthoredItemSpec) -> None:
        """Add one synthetic spec to the trust family and declare the count that now matches it.

        The declared count is the family's real as-published count PLUS this spec, not a literal 1.
        The count check runs before the mistiering check inside the guard, so a stale literal makes
        every test here raise "wrong item count" and pass or fail for a reason none of them is about
        -- which is what the 2026-08-22 landing did to the absolute version of this helper.
        """
        monkeypatch.setattr(survey_module, "AUTHORED_ITEM_SPECS", (*AUTHORED_ITEM_SPECS, spec))
        monkeypatch.setattr(
            survey_module,
            "FAMILIES_AWAITING_ITEMS",
            FAMILIES_AWAITING_ITEMS - {FAMILY_TRUST_RECIPROCITY},
        )
        monkeypatch.setattr(
            survey_module,
            "PLANNED_FAMILY_ITEM_COUNTS",
            {
                **PLANNED_FAMILY_ITEM_COUNTS,
                FAMILY_TRUST_RECIPROCITY: _registered(
                    FAMILY_TRUST_RECIPROCITY, WORDING_AS_PUBLISHED
                )
                + 1,
            },
        )

    @staticmethod
    def _trust_spec(*, tier: str) -> AuthoredItemSpec:
        return AuthoredItemSpec(
            item_id="trust-01",
            family=FAMILY_TRUST_RECIPROCITY,
            instrument="ours-trust",
            kind=SURVEY_LIKERT,
            construct="A synthetic construct.",
            expected_direction="No prediction; this item exists only in a test.",
            tier=tier,
            n_options=5,
        )

    def test_every_new_family_is_declared_breadth_only(self) -> None:
        """Subset rather than equality: self-characterisation joined the set at the 2026-08-22 landing,
        when all eight of its items were breadth, so equality here would refuse a true statement."""
        assert set(NEW_FAMILIES) <= set(FAMILIES_BREADTH_ONLY)
        assert FAMILY_SELF_CHARACTERISATION in FAMILIES_BREADTH_ONLY
        for spec in AUTHORED_ITEM_SPECS:
            if spec.family in FAMILIES_BREADTH_ONLY:
                assert spec.tier == TIER_BREADTH, spec.item_id

    def test_the_spec_default_is_the_expensive_one(self) -> None:
        """Which is why the guard exists rather than a line in a brief nobody re-reads."""
        assert (
            AuthoredItemSpec(
                item_id="trust-01",
                family=FAMILY_TRUST_RECIPROCITY,
                instrument="ours-trust",
                kind=SURVEY_LIKERT,
                construct="A synthetic construct.",
                expected_direction="No prediction; this item exists only in a test.",
                n_options=5,
            ).tier
            == TIER_CORE
        )

    def test_a_core_tier_spec_in_a_breadth_only_family_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._land_one_trust_spec(monkeypatch, self._trust_spec(tier=TIER_CORE))
        with pytest.raises(RuntimeError, match="breadth-only family"):
            survey_module._assert_planned_family_counts_hold(
                {spec.family for spec in survey_module.AUTHORED_ITEM_SPECS}
            )

    def test_the_same_spec_at_breadth_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The positive control: without it, a guard raising on any trust spec would pass above."""
        self._land_one_trust_spec(monkeypatch, self._trust_spec(tier=TIER_BREADTH))
        survey_module._assert_planned_family_counts_hold(
            {spec.family for spec in survey_module.AUTHORED_ITEM_SPECS}
        )


class TestTheCheapTalkRender:
    def test_the_render_carries_both_tags_and_the_vocabulary(self) -> None:
        prompt = render_survey_prompt(cheap_talk_item())
        assert f"<{ANNOUNCE_TAG}>{TAG_EXAMPLE_PLACEHOLDER}</{ANNOUNCE_TAG}>" in prompt
        assert f"<{ACT_TAG}>{TAG_EXAMPLE_PLACEHOLDER}</{ACT_TAG}>" in prompt
        assert (
            f"one word from this list: {TWO_WORD_VOCABULARY[0]}, {TWO_WORD_VOCABULARY[1]}" in prompt
        )

    def test_the_example_is_a_placeholder_never_a_vocabulary_word(self) -> None:
        """Same anti-priming rule as the forced-tag instruction: an example that spells out a real
        option primes it, and here it would prime the announcement AND the action together."""
        item = cheap_talk_item()
        for _, order in battery_orders(item):
            prompt = render_survey_prompt(item, option_order=order)
            for word in item.tag_vocabulary:
                assert f"<{ANNOUNCE_TAG}>{word}</{ANNOUNCE_TAG}>" not in prompt
                assert f"<{ACT_TAG}>{word}</{ACT_TAG}>" not in prompt

    def test_both_presentation_orders_are_offered_and_reverse_the_menu(self) -> None:
        item = cheap_talk_item()
        names = [name for name, _ in battery_orders(item)]
        assert names == ["as-authored", "reversed"]
        reversed_prompt = render_survey_prompt(item, option_order=(1, 0))
        assert f"list: {TWO_WORD_VOCABULARY[1]}, {TWO_WORD_VOCABULARY[0]}" in reversed_prompt

    def test_a_numeric_example_is_refused(self) -> None:
        with pytest.raises(ValueError, match="numeric example"):
            render_survey_prompt(cheap_talk_item(), numeric_example=NUMERIC_EXAMPLE_ROTATION[0])


class TestTheCheapTalkParse:
    """The one genuinely new parser. Every case below is the round-trip the sabotage has to break."""

    def test_every_announced_and_acted_combination_round_trips_under_both_orders(self) -> None:
        item = cheap_talk_item()
        for _, order in battery_orders(item):
            for announced in item.tag_vocabulary:
                for acted in item.tag_vocabulary:
                    completion = f"<{ANNOUNCE_TAG}>{announced}</{ANNOUNCE_TAG}>\n<{ACT_TAG}>{acted}</{ACT_TAG}>"
                    answer = parse_survey_answer(item, completion, option_order=order)
                    canonical = item.tag_vocabulary.index(acted)
                    assert answer.parsed
                    assert answer.tag == acted
                    assert answer.announced_tag == announced
                    assert answer.canonical_index == canonical
                    assert answer.presented_index == order.index(canonical)
                    assert answer.response == canonical + 1
                    assert answer.statement_matched_action is (announced == acted)

    def test_the_action_is_what_fills_the_canonical_index(self) -> None:
        """Not the announcement: every generic reduction reads those fields, and they must see what
        the policy DID. An announcement in the index would make the order-disagreement count and the
        choice distributions measurements of what it said instead."""
        item = cheap_talk_item()
        answer = parse_survey_answer(
            item,
            f"<{ANNOUNCE_TAG}>{TWO_WORD_VOCABULARY[0]}</{ANNOUNCE_TAG}> "
            f"<{ACT_TAG}>{TWO_WORD_VOCABULARY[1]}</{ACT_TAG}>",
        )
        assert answer.canonical_index == 1
        assert answer.tag == TWO_WORD_VOCABULARY[1]
        assert answer.announced_tag == TWO_WORD_VOCABULARY[0]
        assert answer.statement_matched_action is False

    def test_an_announcement_with_no_action_does_not_parse_but_is_still_recorded(self) -> None:
        """All-or-nothing on the pair, and the half that arrived still travels: 'announced and never
        acted' is a different failure from 'ignored the format', and nothing else could say so."""
        item = cheap_talk_item()
        answer = parse_survey_answer(
            item, f"<{ANNOUNCE_TAG}>{TWO_WORD_VOCABULARY[0]}</{ANNOUNCE_TAG}>"
        )
        assert not answer.parsed
        assert answer.canonical_index is None
        assert answer.announced_tag == TWO_WORD_VOCABULARY[0]
        assert answer.statement_matched_action is None

    def test_an_action_with_no_announcement_does_not_parse(self) -> None:
        """Comparing an action against an announcement the completion never made would mean
        inventing the announcement and then grading the model on it."""
        answer = parse_survey_answer(
            cheap_talk_item(), f"<{ACT_TAG}>{TWO_WORD_VOCABULARY[0]}</{ACT_TAG}>"
        )
        assert not answer.parsed
        assert answer.tag is None
        assert answer.announced_tag is None

    def test_a_word_outside_the_vocabulary_does_not_parse(self) -> None:
        answer = parse_survey_answer(
            cheap_talk_item(),
            f"<{ANNOUNCE_TAG}>wordgamma</{ANNOUNCE_TAG}> <{ACT_TAG}>wordalpha</{ACT_TAG}>",
        )
        assert not answer.parsed
        assert answer.announced_tag is None

    def test_the_last_of_each_tag_wins(self) -> None:
        """A model that restates its answer is answering, and the last statement is the answer --
        the same rule `games.parsing` applies to every other tag."""
        answer = parse_survey_answer(
            cheap_talk_item(),
            f"<{ANNOUNCE_TAG}>wordalpha</{ANNOUNCE_TAG}> <{ACT_TAG}>wordalpha</{ACT_TAG}> "
            f"actually: <{ANNOUNCE_TAG}>wordbeta</{ANNOUNCE_TAG}> <{ACT_TAG}>wordalpha</{ACT_TAG}>",
        )
        assert answer.announced_tag == "wordbeta"
        assert answer.tag == "wordalpha"
        assert answer.statement_matched_action is False

    def test_case_and_whitespace_are_forgiven_but_the_menu_is_not(self) -> None:
        answer = parse_survey_answer(
            cheap_talk_item(),
            f"<{ANNOUNCE_TAG}> WordAlpha </{ANNOUNCE_TAG}><{ACT_TAG}>WORDBETA</{ACT_TAG}>",
        )
        assert answer.announced_tag == "wordalpha"
        assert answer.tag == "wordbeta"

    def test_an_empty_completion_parses_to_nothing(self) -> None:
        answer = parse_survey_answer(cheap_talk_item(), "")
        assert not answer.parsed
        assert answer.announced_tag is None
        assert answer.statement_matched_action is None

    def test_a_cheap_talk_item_carries_no_score(self) -> None:
        """Deliberate: an instrument composite that averaged a 0-1 match indicator together with a
        1-5 anchor point would be arithmetic over two different units."""
        answer = parse_survey_answer(
            cheap_talk_item(),
            f"<{ANNOUNCE_TAG}>wordalpha</{ANNOUNCE_TAG}><{ACT_TAG}>wordalpha</{ACT_TAG}>",
        )
        assert answer.score is None
        assert per_item_scores([record_for(cheap_talk_item(), "")]) == {}


class TestTheCheapTalkRecordAndReduction:
    def test_the_record_carries_the_new_contract_fields(self) -> None:
        fields = record_for(
            cheap_talk_item(),
            f"<{ANNOUNCE_TAG}>wordalpha</{ANNOUNCE_TAG}><{ACT_TAG}>wordbeta</{ACT_TAG}>",
        )
        assert set(fields) >= set(SURVEY_RECORD_FIELDS)
        assert fields["announced_tag"] == "wordalpha"
        assert fields["tag"] == "wordbeta"
        assert fields["statement_matched_action"] is False

    def test_an_unparsed_cheap_talk_record_writes_the_same_keys(self) -> None:
        fields = record_for(cheap_talk_item(), "nothing here")
        assert set(fields) >= set(SURVEY_RECORD_FIELDS)
        assert fields["parsed"] is False
        assert fields["statement_matched_action"] is None

    def test_the_reading_matches_the_hand_calculation(self) -> None:
        """Six renders of one item: three announced-alpha-acted-alpha, one announced-alpha-acted-beta,
        one announcement with no action, one with neither.

        parsed = 4 (the pair arrived in four), matched = 3, so match_rate = 3/4 = 0.75 and
        misreport_rate = 0.25. The denominator is PARSED renders, not the six asked: an unparsed
        render is not evidence of honesty either way. announced_only = 1 -- the render that named an
        intention and then never acted -- which is what keeps a collapsing parse rate readable.
        """
        item = cheap_talk_item()
        matched = f"<{ANNOUNCE_TAG}>wordalpha</{ANNOUNCE_TAG}><{ACT_TAG}>wordalpha</{ACT_TAG}>"
        mismatched = f"<{ANNOUNCE_TAG}>wordalpha</{ANNOUNCE_TAG}><{ACT_TAG}>wordbeta</{ACT_TAG}>"
        completions = [
            matched,
            matched,
            matched,
            mismatched,
            f"<{ANNOUNCE_TAG}>wordalpha</{ANNOUNCE_TAG}>",
            "no tags at all",
        ]
        readings = cheap_talk_readings([record_for(item, text) for text in completions])
        reading = readings[item.item_id]
        assert reading.n_asked == 6
        assert reading.n_parsed == 4
        assert reading.n_matched == 3
        assert reading.n_announced_only == 1
        assert reading.match_rate == pytest.approx(0.75)
        assert reading.misreport_rate == pytest.approx(0.25)
        assert reading.pair_counts == {
            ("wordalpha", "wordalpha"): 3,
            ("wordalpha", "wordbeta"): 1,
        }

    def test_a_rate_with_no_parsed_render_is_none_rather_than_zero(self) -> None:
        item = cheap_talk_item()
        readings = cheap_talk_readings([record_for(item, "nothing") for _ in range(3)])
        reading = readings[item.item_id]
        assert reading.n_asked == 3
        assert reading.match_rate is None
        assert reading.misreport_rate is None

    def test_items_are_never_pooled_with_each_other(self) -> None:
        """Different situations with different stakes; a mean across them is a rate of nothing."""
        first = cheap_talk_item("deception-01")
        second = cheap_talk_item("deception-02")
        matched = f"<{ANNOUNCE_TAG}>wordalpha</{ANNOUNCE_TAG}><{ACT_TAG}>wordalpha</{ACT_TAG}>"
        mismatched = f"<{ANNOUNCE_TAG}>wordalpha</{ANNOUNCE_TAG}><{ACT_TAG}>wordbeta</{ACT_TAG}>"
        readings = cheap_talk_readings([record_for(first, matched), record_for(second, mismatched)])
        assert readings[first.item_id].match_rate == pytest.approx(1.0)
        assert readings[second.item_id].match_rate == pytest.approx(0.0)

    def test_records_of_other_kinds_are_ignored(self) -> None:
        assert (
            cheap_talk_readings([record_for(trust_likert_item("trust-01"), "FINAL ANSWER: C")])
            == {}
        )


class TestTheTaggedReadingAndItsDistance:
    """The forced-tag families' reduction: a per-item distribution over the item's word menu.

    Eleven registered items are read through this -- eight self-characterisation stances and the three
    tag mirrors of the graded-dimension family -- and the registered quantity is each item's word
    shares plus the total variation distance of that distribution from step 0's. Both halves are
    pinned against arithmetic worked out here rather than against what the code returns, because a
    share whose denominator counted the unparsed renders and a distance taken across a re-authored
    menu both produce entirely plausible numbers.
    """

    def test_the_distribution_matches_the_hand_calculation(self) -> None:
        """Five renders of one three-word item: wordzeta twice, wordalpha once, two that parse to
        nothing.

        n_asked = 5 and n_parsed = 3, and the shares are over PARSED renders: 2/3 and 1/3, which sum
        to one. Over all five instead they would read 0.4 and 0.2 and sum to 0.6 -- a "distribution"
        that quietly encodes the parse failures, which the denominators beside it already report.

        Entropy of (2/3, 1/3) = -(2/3 log2 2/3 + 1/3 log2 1/3) = -(2/3)(-0.584963) - (1/3)(-1.584963)
        = 0.389975 + 0.528321 = 0.918296 bits, against the 1.585 bits a flat three-word menu would
        carry -- the headroom column, so a flat 0.000 movement can be told from no room to move.
        """
        item = forced_tag_item(vocabulary=THREE_WORD_VOCABULARY)
        completions = [
            tag_completion("wordzeta"),
            tag_completion("wordzeta"),
            tag_completion("wordalpha"),
            "no tag at all",
            "",
        ]
        reading = tagged_readings([record_for(item, text) for text in completions])[item.item_id]
        assert reading.n_asked == 5
        assert reading.n_parsed == 3
        assert reading.counts == {"wordzeta": 2, "wordalpha": 1}
        assert reading.shares == pytest.approx({"wordzeta": 2 / 3, "wordalpha": 1 / 3})
        assert reading.vocabulary_sizes == (3,)
        assert reading.vocabulary_size == 3
        assert reading.entropy_bits == pytest.approx(0.918296, abs=1e-6)

    def test_words_are_ordered_by_menu_position_rather_than_by_name(self) -> None:
        """The menu is `wordzeta, wordalpha, wordmu`, whose alphabetical order is the reverse of its
        canonical one. Several real menus are ladders (a three-rung visibility ladder among them), so
        a table sorted by name reads down no ladder and reorders whenever a share moves.
        """
        item = forced_tag_item(vocabulary=THREE_WORD_VOCABULARY)
        completions = ["wordzeta", "wordalpha", "wordmu", "wordmu"]
        reading = tagged_readings([record_for(item, tag_completion(word)) for word in completions])[
            item.item_id
        ]
        assert list(reading.counts) == ["wordzeta", "wordalpha", "wordmu"]
        assert reading.counts == {"wordzeta": 1, "wordalpha": 1, "wordmu": 2}

    def test_an_item_nothing_parsed_on_reports_its_width_and_no_shares(self) -> None:
        """No shares and no entropy rather than a flat distribution and 0.0 bits: nothing measured is
        not a measurement of nothing. The width still travels, because an unparsed record carries the
        item's `scale_points` and the width is what the distance check compares.
        """
        item = forced_tag_item(vocabulary=THREE_WORD_VOCABULARY)
        reading = tagged_readings([record_for(item, "nothing") for _ in range(4)])[item.item_id]
        assert (reading.n_asked, reading.n_parsed) == (4, 0)
        assert reading.counts == {}
        assert reading.shares == {}
        assert reading.entropy_bits is None
        assert reading.vocabulary_size == 3

    def test_items_are_never_pooled_with_each_other(self) -> None:
        """Eight self-characterisation items describe eight situations; a share pooled over which word
        it picked across them is a share of nothing."""
        first = forced_tag_item("self-characterisation-01")
        second = forced_tag_item("self-characterisation-02")
        readings = tagged_readings(
            [
                record_for(first, tag_completion("wordalpha")),
                record_for(second, tag_completion("wordbeta")),
            ]
        )
        assert readings[first.item_id].counts == {"wordalpha": 1}
        assert readings[second.item_id].counts == {"wordbeta": 1}

    def test_a_cheap_talk_record_is_not_read_as_a_tagged_one(self) -> None:
        """The one kind confusion worth a test: a cheap-talk record also carries a `tag`, its action
        word. That word is read against the announcement beside it by `cheap_talk_readings`, so
        admitting it here would print two readings of one answer and put an announcement's menu into
        an attribution table.
        """
        matched = f"<{ANNOUNCE_TAG}>wordalpha</{ANNOUNCE_TAG}><{ACT_TAG}>wordalpha</{ACT_TAG}>"
        assert tagged_readings([record_for(cheap_talk_item(), matched)]) == {}
        assert tagged_readings([record_for(trust_likert_item("trust-01"), "FINAL ANSWER: C")]) == {}

    def test_the_distance_matches_the_hand_calculation(self) -> None:
        """Two cells of one three-word item. The first answers wordzeta twice and wordalpha once
        (2/3, 1/3); the second answers wordzeta once and wordmu twice (1/3, 2/3).

        Total variation is half the L1 distance over the union of words:
        0.5 * (|2/3 - 1/3| + |1/3 - 0| + |0 - 2/3|) = 0.5 * (1/3 + 1/3 + 2/3) = 2/3 = 0.666667.
        A word one cell never chose is absent from its shares rather than present at zero, which is
        what makes the union rather than either cell's keys the right index set.
        """
        item = forced_tag_item(vocabulary=THREE_WORD_VOCABULARY)
        first = tagged_readings(
            [
                record_for(item, tag_completion(word))
                for word in ("wordzeta", "wordzeta", "wordalpha")
            ]
        )[item.item_id]
        second = tagged_readings(
            [record_for(item, tag_completion(word)) for word in ("wordzeta", "wordmu", "wordmu")]
        )[item.item_id]
        distance = tagged_distribution_distance(first, second)
        assert distance.item_id == item.item_id
        assert distance.reason is None
        assert distance.distance == pytest.approx(2 / 3)
        assert (distance.n_first_parsed, distance.n_second_parsed) == (3, 3)
        # Symmetric, which is why the argument names claim nothing about which cell is the baseline.
        assert tagged_distribution_distance(second, first).distance == pytest.approx(2 / 3)

    def test_an_identical_pair_of_cells_reads_zero(self) -> None:
        """The positive control on every refusal below: without it, a function that refused every pair
        would satisfy them all and report movement nowhere."""
        item = forced_tag_item(vocabulary=THREE_WORD_VOCABULARY)
        readings = [
            tagged_readings([record_for(item, tag_completion("wordmu")) for _ in range(2)])[
                item.item_id
            ]
            for _ in range(2)
        ]
        assert tagged_distribution_distance(*readings).distance == pytest.approx(0.0)

    def test_a_changed_menu_width_is_refused_rather_than_computed(self) -> None:
        """The refusal this reduction exists to make. The same item id is asked with a three-word menu
        in one cell and a two-word menu in the other, which is what a re-authored item looks like from
        a stored trace.

        Computed anyway, this pair reads 1.000 -- the two cells' words are disjoint, so it is the
        largest distance the measure can return, and it would be printed as the strongest movement in
        the battery when the truth is that the question changed. A dropped menu word and a share that
        fell to zero are indistinguishable in a record, so the reason travels instead of a number.
        """
        wide = forced_tag_item(vocabulary=THREE_WORD_VOCABULARY)
        narrow = forced_tag_item(vocabulary=TWO_WORD_VOCABULARY)
        assert wide.item_id == narrow.item_id
        first = tagged_readings([record_for(wide, tag_completion("wordzeta")) for _ in range(2)])[
            wide.item_id
        ]
        second = tagged_readings(
            [record_for(narrow, tag_completion("wordbeta")) for _ in range(2)]
        )[narrow.item_id]
        distance = tagged_distribution_distance(first, second)
        assert distance.reason == TAGGED_VOCABULARY_WIDTH_DIFFERS
        assert distance.distance is None
        assert (distance.n_first_parsed, distance.n_second_parsed) == (2, 2)

    def test_a_same_width_substitution_is_refused_when_the_words_outnumber_the_menu(self) -> None:
        """The re-authoring a width comparison cannot see: two words swapped for two others, so both
        cells still record a menu of two. The first cell answers wordalpha and wordbeta, the second
        answers wordmu, and three distinct words cannot have come from one two-word menu -- which is
        positive evidence the item changed, not an inference from the shares.
        """
        original = forced_tag_item(vocabulary=TWO_WORD_VOCABULARY)
        rewritten = forced_tag_item(vocabulary=("wordalpha", "wordmu"))
        first = tagged_readings(
            [record_for(original, tag_completion(word)) for word in ("wordalpha", "wordbeta")]
        )[original.item_id]
        second = tagged_readings([record_for(rewritten, tag_completion("wordmu"))])[
            rewritten.item_id
        ]
        distance = tagged_distribution_distance(first, second)
        assert distance.reason == TAGGED_VOCABULARY_WORDS_DIFFER
        assert distance.distance is None

    def test_a_cell_that_cannot_state_one_menu_width_is_refused(self) -> None:
        """One cell holding both versions of an item -- what a hand-merged trace looks like. Its own
        records disagree about the width, so the cell has no width to compare and no readable share
        either: 0.5 means one thing on a two-word menu and another on a three-word one.
        """
        item = forced_tag_item(vocabulary=THREE_WORD_VOCABULARY)
        rewritten = forced_tag_item(vocabulary=TWO_WORD_VOCABULARY)
        mixed = tagged_readings(
            [
                record_for(item, tag_completion("wordalpha")),
                record_for(rewritten, tag_completion("wordalpha")),
            ]
        )[item.item_id]
        assert mixed.vocabulary_sizes == (2, 3)
        assert mixed.vocabulary_size is None
        distance = tagged_distribution_distance(mixed, mixed)
        assert distance.reason == TAGGED_VOCABULARY_WIDTH_UNSTATED
        assert distance.distance is None

    def test_a_cell_where_nothing_parsed_reports_that_rather_than_a_distance(self) -> None:
        item = forced_tag_item(vocabulary=THREE_WORD_VOCABULARY)
        answered = tagged_readings([record_for(item, tag_completion("wordmu"))])[item.item_id]
        silent = tagged_readings([record_for(item, "nothing") for _ in range(3)])[item.item_id]
        distance = tagged_distribution_distance(answered, silent)
        assert distance.reason == TAGGED_NOTHING_PARSED
        assert distance.distance is None
        assert (distance.n_first_parsed, distance.n_second_parsed) == (1, 0)

    def test_an_item_only_one_cell_asked_reports_that(self) -> None:
        """A step that dropped an item and a step whose answers all moved are different facts, and a
        readout that printed a blank for the first would invite reading it as the second."""
        item = forced_tag_item(vocabulary=THREE_WORD_VOCABULARY)
        reading = tagged_readings([record_for(item, tag_completion("wordmu")) for _ in range(2)])[
            item.item_id
        ]
        distance = tagged_distribution_distance(None, reading)
        assert distance.reason == TAGGED_ITEM_MISSING
        assert distance.distance is None
        assert (distance.n_first_parsed, distance.n_second_parsed) == (0, 2)
        assert tagged_distribution_distance(reading, None).n_second_parsed == 0

    def test_comparing_two_different_items_raises(self) -> None:
        """A caller mistake rather than model output, so it raises instead of reporting a reason: the
        distance between two different items' distributions is not a quantity at all."""
        first_item = forced_tag_item("self-characterisation-01")
        second_item = forced_tag_item("self-characterisation-02")
        first = tagged_readings([record_for(first_item, tag_completion("wordalpha"))])[
            first_item.item_id
        ]
        second = tagged_readings([record_for(second_item, tag_completion("wordbeta"))])[
            second_item.item_id
        ]
        with pytest.raises(ValueError, match="compares one item across two cells"):
            tagged_distribution_distance(first, second)

    def test_an_item_in_neither_cell_raises(self) -> None:
        with pytest.raises(ValueError, match="neither cell reports"):
            tagged_distribution_distance(None, None)

    def test_the_eval_summary_carries_the_reading(self) -> None:
        """The wiring, not just the reduction: a distribution collected into a trace and never reduced
        is the failure this repo names after the callback that wrote into a copied dict."""
        item = forced_tag_item(vocabulary=THREE_WORD_VOCABULARY)
        records = [
            record_for(item, tag_completion(word)) for word in ("wordzeta", "wordzeta", "wordmu")
        ]
        summary = evals_module._summarise_self_report(records, [])["tagged_readings"]
        assert summary[item.item_id]["counts"] == {"wordzeta": 2, "wordmu": 1}
        assert summary[item.item_id]["shares"] == pytest.approx(
            {"wordzeta": 2 / 3, "wordmu": 1 / 3}
        )
        assert summary[item.item_id]["n_parsed"] == 3
        assert summary[item.item_id]["n_asked"] == 3
        assert summary[item.item_id]["vocabulary_size"] == 3
        assert summary[item.item_id]["entropy_bits"] == pytest.approx(0.918296, abs=1e-6)


class TestOrderedChoiceLadders:
    def test_a_ladder_renders_lettered_and_round_trips_every_rung(self) -> None:
        item = ordered_choice_item()
        for _, order in battery_orders(item):
            prompt = render_survey_prompt(item, option_order=order)
            assert "FINAL ANSWER: X" in prompt
            for presented, canonical in enumerate(order):
                letter = "ABC"[presented]
                assert f"{letter}) {item.options[canonical]}" in prompt
                answer = parse_survey_answer(item, f"FINAL ANSWER: {letter}", option_order=order)
                assert answer.canonical_index == canonical
                assert answer.response == canonical + 1

    def test_the_score_is_the_canonical_rung_not_the_presented_one(self) -> None:
        """The rung number IS the quantity, so a reversed render must score the rung the model chose
        rather than the letter it wrote -- otherwise the counterbalance inverts the index."""
        item = ordered_choice_item()
        as_authored = parse_survey_answer(item, "FINAL ANSWER: A", option_order=(0, 1, 2))
        reversed_render = parse_survey_answer(item, "FINAL ANSWER: A", option_order=(2, 1, 0))
        assert as_authored.score == pytest.approx(1.0)
        assert reversed_render.score == pytest.approx(3.0)

    def test_a_ladder_cannot_be_reverse_keyed(self) -> None:
        """Reverse-keying reflects an agreement direction, and a rung number has none."""
        with pytest.raises(ValueError, match="ordered anchor ladder"):
            SurveyItem(
                item_id="risk-01",
                family=FAMILY_RISK_PREFERENCE,
                instrument="synthetic-risk",
                kind=SURVEY_ORDERED_CHOICE,
                stem="Synthetic ladder.",
                construct="A synthetic construct.",
                expected_direction="No prediction; this item exists only in a test.",
                reverse_keyed=True,
                options=THREE_RUNG_OPTIONS,
            )

    def test_a_ladder_is_excluded_from_the_nominal_control_reductions(self) -> None:
        """The negative-control drift table reads `SURVEY_CHOICE` distributions; a ladder has a mean
        instead, and letting it into that table would print two readings of one item."""
        item = ordered_choice_item()
        assert choice_response_distributions([record_for(item, "FINAL ANSWER: A")]) == {}

    def test_the_subscale_composite_matches_the_hand_calculation(self) -> None:
        """Two three-rung ladders in one subscale. The first is answered at rung 1 and rung 3, so its
        per-item mean is (1 + 3) / 2 = 2.0; the second is answered at rung 3 twice, so its mean is
        3.0. The subscale mean is over ITEMS, not renders: (2.0 + 3.0) / 2 = 2.5. Pooling the four
        renders instead would give (1 + 3 + 3 + 3) / 4 = 2.5 here by coincidence, so the third
        render below makes the two differ: with it, per item is (2.0 + 3.0) / 2 = 2.5 while pooled
        would be 13 / 5 = 2.6.
        """
        first = ordered_choice_item("risk-01")
        second = ordered_choice_item("risk-02")
        records = [
            record_for(first, "FINAL ANSWER: A"),
            record_for(first, "FINAL ANSWER: C"),
            record_for(second, "FINAL ANSWER: C"),
            record_for(second, "FINAL ANSWER: C"),
            record_for(second, "FINAL ANSWER: C"),
        ]
        assert per_item_scores(records) == {
            "risk-01": pytest.approx(2.0),
            "risk-02": pytest.approx(3.0),
        }
        composites = subscale_composites(records)
        assert composites[("synthetic-risk", "gamble-ladder")] == pytest.approx(2.5)

    def test_a_subscale_mixing_ladder_lengths_is_refused(self) -> None:
        """A mean over rung numbers cannot span two rung counts, and the number it produces looks
        exactly like a risk-tolerance index."""
        short = ordered_choice_item("risk-01")
        long_ladder = ordered_choice_item(
            "risk-02", options=(*THREE_RUNG_OPTIONS, "synthetic rung 4")
        )
        with pytest.raises(ValueError, match="different lengths"):
            assert_ordered_choice_ladders_are_commensurable([short, long_ladder])

    def test_two_lengths_in_two_subscales_are_allowed(self) -> None:
        """The honest form of the same family: one subscale per ladder length, printed apart."""
        assert_ordered_choice_ladders_are_commensurable(
            [
                ordered_choice_item("risk-01"),
                ordered_choice_item(
                    "risk-02",
                    options=(*THREE_RUNG_OPTIONS, "synthetic rung 4"),
                    subscale="gamble-ladder-four-rung",
                ),
            ]
        )


class TestOptionLabels:
    def test_the_chosen_label_follows_the_option_and_not_the_letter(self) -> None:
        """The assertion an order-counterbalance bypass has to break for the labelled families: under
        the reversed render, letter A is the SECOND label, and a record that named the first would
        invert the whole preference ordering while every rate stayed plausible."""
        item = labelled_choice_item()
        as_authored = record_for(item, "FINAL ANSWER: A", order=(0, 1))
        reversed_render = record_for(item, "FINAL ANSWER: A", order=(1, 0))
        assert as_authored["chosen_label"] == "joint-gain"
        assert reversed_render["chosen_label"] == "own-gain"

    def test_the_record_carries_the_offered_labels_too(self) -> None:
        """A re-analysis from the trace needs the denominator, and the option prose is not on disk."""
        fields = record_for(labelled_choice_item(), "FINAL ANSWER: A")
        assert fields["option_labels"] == ["joint-gain", "own-gain"]

    def test_an_unparsed_render_names_no_label(self) -> None:
        fields = record_for(labelled_choice_item(), "no answer")
        assert fields["chosen_label"] is None
        assert fields["option_labels"] == ["joint-gain", "own-gain"]

    def test_labels_are_refused_on_a_kind_whose_options_already_mean_something(self) -> None:
        with pytest.raises(ValueError, match="carries option_labels"):
            SurveyItem(
                item_id="trust-01",
                family=FAMILY_TRUST_RECIPROCITY,
                instrument="synthetic-trust",
                kind=SURVEY_LIKERT,
                stem="Synthetic statement.",
                construct="A synthetic construct.",
                expected_direction="No prediction; this item exists only in a test.",
                options=FIVE_ANCHORS,
                option_labels=("a", "b", "c", "d", "e"),
            )

    def test_a_short_label_list_is_refused(self) -> None:
        """Labels are positional, so a list shorter than the option block labels the wrong options."""
        with pytest.raises(ValueError, match="option_labels"):
            SurveyItem(
                item_id="values-01",
                family=FAMILY_VALUES_FORCED_CHOICE,
                instrument="synthetic-values",
                kind=SURVEY_CHOICE,
                stem="Synthetic forced choice.",
                construct="A synthetic construct.",
                expected_direction="No prediction; this item exists only in a test.",
                options=("synthetic option 1", "synthetic option 2", "synthetic option 3"),
                option_labels=("joint-gain", "own-gain"),
            )

    def test_a_repeated_label_is_refused(self) -> None:
        with pytest.raises(ValueError, match="repeats an option label"):
            labelled_choice_item(labels=("joint-gain", "joint-gain"))

    def test_a_blank_label_is_refused(self) -> None:
        with pytest.raises(ValueError, match="blank option_labels"):
            labelled_choice_item(labels=("joint-gain", "   "))

    def test_a_spec_whose_label_count_disagrees_with_its_option_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="option_labels"):
            AuthoredItemSpec(
                item_id="values-99",
                family=FAMILY_VALUES_FORCED_CHOICE,
                instrument="synthetic-values",
                kind=SURVEY_CHOICE,
                construct="A synthetic construct.",
                expected_direction="No prediction; this item exists only in a test.",
                n_options=3,
                option_labels=("joint-gain", "own-gain"),
            )

    def test_the_win_rates_match_the_hand_calculation(self) -> None:
        """Two items. `values-01` offers joint-gain against own-gain and is answered four times: A, A,
        A, B -- so joint-gain wins 3/4 = 0.75 there. `values-02` offers joint-gain against equality
        and is answered twice, once each, so joint-gain wins 1/2 = 0.5 there.

        joint-gain's win rate is the mean over ITEMS: (0.75 + 0.5) / 2 = 0.625. Pooling the renders
        instead would give 4/6 = 0.667, which is the number this reduction must NOT return -- it
        would weight a label by how many of its items happened to parse. The fifth render on
        `values-01` is unparsed and is excluded from both sides: a parse failure is not a vote
        against every label on the item.
        """
        first = labelled_choice_item("values-01", labels=("joint-gain", "own-gain"))
        second = labelled_choice_item("values-02", labels=("joint-gain", "equality"))
        records = [
            record_for(first, "FINAL ANSWER: A"),
            record_for(first, "FINAL ANSWER: A"),
            record_for(first, "FINAL ANSWER: A"),
            record_for(first, "FINAL ANSWER: B"),
            record_for(first, "no answer at all"),
            record_for(second, "FINAL ANSWER: A"),
            record_for(second, "FINAL ANSWER: B"),
        ]
        rates = forced_choice_win_rates(records, family=FAMILY_VALUES_FORCED_CHOICE)
        assert rates["joint-gain"].win_rate == pytest.approx(0.625)
        assert rates["joint-gain"].n_items_offering == 2
        assert rates["joint-gain"].n_chosen == 4
        assert rates["joint-gain"].n_parsed_renders_offering == 6
        assert rates["own-gain"].win_rate == pytest.approx(0.25)
        assert rates["own-gain"].n_items_offering == 1
        assert rates["equality"].win_rate == pytest.approx(0.5)

    def test_one_items_label_rates_sum_to_one(self) -> None:
        """What makes them an ordering rather than three unrelated fractions."""
        item = labelled_choice_item()
        records = [record_for(item, "FINAL ANSWER: A"), record_for(item, "FINAL ANSWER: B")]
        rates = forced_choice_win_rates(records, family=FAMILY_VALUES_FORCED_CHOICE)
        total = sum(rate.win_rate or 0.0 for rate in rates.values())
        assert total == pytest.approx(1.0)

    def test_unlabelled_records_contribute_nothing(self) -> None:
        """Asked under the Likert item's OWN family, so the empty answer is about labels not families."""
        assert (
            forced_choice_win_rates(
                [record_for(trust_likert_item("t-01"), "FINAL ANSWER: C")],
                family=FAMILY_TRUST_RECIPROCITY,
            )
            == {}
        )

    def test_another_familys_labelled_records_never_enter_the_ordering(self) -> None:
        """A labelled item outside the named family contributes to neither its rates nor its counts.

        The reduction pools by label STRING, and the risk ladders label their rungs too, so an
        unfiltered pass returns one dictionary holding `joint-gain` beside `reliance-full` and reads as
        an ordering over both. Asserted on the DENOMINATORS as well as on the key set, because a
        foreign item that happened to share a label would move the rate silently.
        """
        values = labelled_choice_item("values-01", labels=("joint-gain", "own-gain"))
        ladder = replace(
            ordered_choice_item("risk-01", options=THREE_RUNG_OPTIONS),
            option_labels=("joint-gain", "own-gain", "equality"),
        )
        records = [
            record_for(values, "FINAL ANSWER: A"),
            record_for(ladder, "FINAL ANSWER: C"),
            record_for(ladder, "FINAL ANSWER: C"),
        ]
        rates = forced_choice_win_rates(records, family=FAMILY_VALUES_FORCED_CHOICE)
        assert sorted(rates) == ["joint-gain", "own-gain"]
        assert rates["joint-gain"].win_rate == pytest.approx(1.0)
        assert rates["joint-gain"].n_items_offering == 1
        assert rates["joint-gain"].n_parsed_renders_offering == 1

    def test_the_pooled_pole_is_recomputed_over_its_items_rather_than_averaged(self) -> None:
        """Three prefixed labels on items with different counts, against the mean of their rates.

        `non-social-brevity` is offered by one item answered twice, both times for it, so its own rate
        is 1.0. `non-social-honesty` is offered by two items, one answered for it and one against, so
        its rate is 0.5. Averaging the two per-good rates gives 0.75. The pole is recomputed over the
        three items instead -- (1.0 + 1.0 + 0.0) / 3 -- and reads 0.667, with 3 items and 4 renders
        behind it, which is the registered reading of the fifth pole.
        """
        brevity = labelled_choice_item("values-01", labels=("non-social-brevity", "joint-gain"))
        honesty_won = labelled_choice_item("values-02", labels=("non-social-honesty", "equality"))
        honesty_lost = labelled_choice_item("values-03", labels=("non-social-honesty", "own-gain"))
        records = [
            record_for(brevity, "FINAL ANSWER: A"),
            record_for(brevity, "FINAL ANSWER: A"),
            record_for(honesty_won, "FINAL ANSWER: A"),
            record_for(honesty_lost, "FINAL ANSWER: B"),
        ]
        rates = forced_choice_win_rates(records, family=FAMILY_VALUES_FORCED_CHOICE)
        assert rates["non-social-brevity"].win_rate == pytest.approx(1.0)
        assert rates["non-social-honesty"].win_rate == pytest.approx(0.5)
        pooled = forced_choice_prefix_win_rate(
            records, family=FAMILY_VALUES_FORCED_CHOICE, prefix=NON_SOCIAL_LABEL_PREFIX
        )
        assert pooled is not None
        assert pooled.label == f"{NON_SOCIAL_LABEL_PREFIX}*"
        assert pooled.win_rate == pytest.approx(2.0 / 3.0)
        assert pooled.win_rate != pytest.approx(0.75)
        assert pooled.n_items_offering == 3
        assert pooled.n_chosen == 3
        assert pooled.n_parsed_renders_offering == 4

    def test_an_item_offering_two_prefixed_labels_stays_one_observation(self) -> None:
        """Its wins add and its denominator does not: an item is one observation however many of its
        options belong to the pole, so a pole-versus-pole item cannot read as two."""
        both = labelled_choice_item(
            "values-01", labels=("non-social-brevity", "non-social-honesty")
        )
        records = [record_for(both, "FINAL ANSWER: A"), record_for(both, "FINAL ANSWER: B")]
        pooled = forced_choice_prefix_win_rate(
            records, family=FAMILY_VALUES_FORCED_CHOICE, prefix=NON_SOCIAL_LABEL_PREFIX
        )
        assert pooled is not None
        assert pooled.win_rate == pytest.approx(1.0)
        assert pooled.n_items_offering == 1
        assert pooled.n_parsed_renders_offering == 2

    def test_a_family_offering_no_prefixed_label_has_no_pooled_pole(self) -> None:
        """None rather than a zero: no item offered the pole, which is not the same as never choosing
        it."""
        item = labelled_choice_item("values-01", labels=("joint-gain", "own-gain"))
        assert (
            forced_choice_prefix_win_rate(
                [record_for(item, "FINAL ANSWER: A")],
                family=FAMILY_VALUES_FORCED_CHOICE,
                prefix=NON_SOCIAL_LABEL_PREFIX,
            )
            is None
        )


class TestCounterpartPairs:
    def test_a_named_counterpart_without_a_pair_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="counterpart_pair"):
            trust_likert_item("trust-01", counterpart=COUNTERPART_AI)

    def test_a_pair_key_without_a_named_counterpart_is_refused(self) -> None:
        with pytest.raises(ValueError, match="counterpart_pair"):
            trust_likert_item("trust-01", counterpart_pair="trust-stranger")

    def test_a_spec_mis_declaring_its_pair_fails_at_import_not_at_load(self) -> None:
        """The registry is checked as well as the item, so a mis-declared pair is an import error
        rather than something a run discovers after it has reached a GPU."""
        with pytest.raises(ValueError, match="counterpart_pair"):
            AuthoredItemSpec(
                item_id="trust-01-ai",
                family=FAMILY_TRUST_RECIPROCITY,
                instrument="synthetic-trust",
                kind=SURVEY_LIKERT,
                construct="A synthetic construct.",
                expected_direction="No prediction; this item exists only in a test.",
                n_options=5,
                counterpart=COUNTERPART_AI,
            )

    def test_a_spec_naming_an_unregistered_counterpart_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown counterpart"):
            AuthoredItemSpec(
                item_id="trust-01-ai",
                family=FAMILY_TRUST_RECIPROCITY,
                instrument="synthetic-trust",
                kind=SURVEY_LIKERT,
                construct="A synthetic construct.",
                expected_direction="No prediction; this item exists only in a test.",
                n_options=5,
                counterpart="another-model",
                counterpart_pair="p",
            )

    def test_a_complete_pair_passes_the_guard(self) -> None:
        assert_every_counterpart_pair_is_complete(
            [
                trust_likert_item("trust-01-ai", counterpart=COUNTERPART_AI, counterpart_pair="p"),
                trust_likert_item(
                    "trust-01-human", counterpart=COUNTERPART_HUMAN, counterpart_pair="p"
                ),
            ]
        )

    def test_a_lone_arm_is_refused(self) -> None:
        """One arm alone is a level, and every level in this battery is confounded by over-reporting."""
        with pytest.raises(ValueError, match="carries arms"):
            assert_every_counterpart_pair_is_complete(
                [trust_likert_item("trust-01-ai", counterpart=COUNTERPART_AI, counterpart_pair="p")]
            )

    def test_two_arms_on_the_same_side_are_refused(self) -> None:
        with pytest.raises(ValueError, match="carries arms"):
            assert_every_counterpart_pair_is_complete(
                [
                    trust_likert_item(
                        "trust-01-ai", counterpart=COUNTERPART_AI, counterpart_pair="p"
                    ),
                    trust_likert_item(
                        "trust-02-ai", counterpart=COUNTERPART_AI, counterpart_pair="p"
                    ),
                ]
            )

    def test_arms_mismatched_on_anything_else_are_refused(self) -> None:
        """The difference is supposed to cancel everything but the counterpart, so a keying that
        differs across arms is what the number would actually be measuring."""
        with pytest.raises(ValueError, match="disagrees across its arms"):
            assert_every_counterpart_pair_is_complete(
                [
                    trust_likert_item(
                        "trust-01-ai", counterpart=COUNTERPART_AI, counterpart_pair="p"
                    ),
                    trust_likert_item(
                        "trust-01-human",
                        counterpart=COUNTERPART_HUMAN,
                        counterpart_pair="p",
                        reverse_keyed=True,
                    ),
                ]
            )

    def test_the_likert_gap_matches_the_hand_calculation(self) -> None:
        """Five-point items, not reverse-keyed, so a canonical response of n scores n. The AI arm is
        answered at 4 and 2, giving a per-item mean of 3.0; the human arm is answered at 5 twice,
        giving 5.0. The gap is AI minus human: 3.0 - 5.0 = -2.0, i.e. less of whatever the item
        measures when the counterpart is another system.
        """
        ai_arm = trust_likert_item("trust-01-ai", counterpart=COUNTERPART_AI, counterpart_pair="p")
        human_arm = trust_likert_item(
            "trust-01-human", counterpart=COUNTERPART_HUMAN, counterpart_pair="p"
        )
        records = [
            record_for(ai_arm, "FINAL ANSWER: D"),
            record_for(ai_arm, "FINAL ANSWER: B"),
            record_for(human_arm, "FINAL ANSWER: E"),
            record_for(human_arm, "FINAL ANSWER: E"),
        ]
        gap = counterpart_gaps(records)["p"]
        assert gap.ai == pytest.approx(3.0)
        assert gap.human == pytest.approx(5.0)
        assert gap.gap == pytest.approx(-2.0)
        assert gap.reason is None
        assert (gap.n_ai_parsed, gap.n_human_parsed) == (2, 2)
        assert (gap.n_ai_asked, gap.n_human_asked) == (2, 2)

    def test_a_cheap_talk_pair_differences_its_match_rates(self) -> None:
        """The honesty analogue: two of the AI arm's four renders keep their announcement (0.5) and
        all two of the human arm's do (1.0), so the gap is 0.5 - 1.0 = -0.5."""
        ai_arm = cheap_talk_item(
            "deception-01-ai", counterpart=COUNTERPART_AI, counterpart_pair="q"
        )
        human_arm = cheap_talk_item(
            "deception-01-human", counterpart=COUNTERPART_HUMAN, counterpart_pair="q"
        )
        matched = f"<{ANNOUNCE_TAG}>wordalpha</{ANNOUNCE_TAG}><{ACT_TAG}>wordalpha</{ACT_TAG}>"
        mismatched = f"<{ANNOUNCE_TAG}>wordalpha</{ANNOUNCE_TAG}><{ACT_TAG}>wordbeta</{ACT_TAG}>"
        records = [
            record_for(ai_arm, matched),
            record_for(ai_arm, matched),
            record_for(ai_arm, mismatched),
            record_for(ai_arm, mismatched),
            record_for(human_arm, matched),
            record_for(human_arm, matched),
        ]
        gap = counterpart_gaps(records)["q"]
        assert gap.ai == pytest.approx(0.5)
        assert gap.human == pytest.approx(1.0)
        assert gap.gap == pytest.approx(-0.5)

    def test_a_numeric_pair_differences_its_canonical_integers(self) -> None:
        ai_arm = SurveyItem(
            item_id="trust-send-ai",
            family=FAMILY_TRUST_RECIPROCITY,
            instrument="synthetic-trust",
            kind=SURVEY_NUMERIC,
            stem="Synthetic send-framed stem.",
            construct="A synthetic construct.",
            expected_direction="No prediction; this item exists only in a test.",
            counterpart=COUNTERPART_AI,
            counterpart_pair="r",
            numeric_max=10,
        )
        human_arm = SurveyItem(
            item_id="trust-send-human",
            family=FAMILY_TRUST_RECIPROCITY,
            instrument="synthetic-trust",
            kind=SURVEY_NUMERIC,
            stem="Synthetic send-framed stem.",
            construct="A synthetic construct.",
            expected_direction="No prediction; this item exists only in a test.",
            counterpart=COUNTERPART_HUMAN,
            counterpart_pair="r",
            numeric_max=10,
        )
        gap = counterpart_gaps(
            [record_for(ai_arm, "<keep>2</keep>"), record_for(human_arm, "<keep>7</keep>")]
        )["r"]
        assert gap.gap == pytest.approx(-5.0)

    def test_a_nominal_pair_in_a_trace_reports_its_reason_rather_than_a_number(self) -> None:
        """Differencing option numbers would let the authored option order set the effect size.

        Built as records rather than from items on purpose: no valid item can be a nominal counterpart
        arm any more (`TestCounterpartsOnlyOnDifferenceableKinds` below is that refusal), so this
        branch is now reachable only from records -- which is exactly what the reduction reads, and
        traces written before that refusal existed are still on disk.
        """
        records = [
            {
                "item_id": f"values-01-{side}",
                "kind": SURVEY_CHOICE,
                "counterpart": side,
                "counterpart_pair": "s",
                "canonical_index": 0,
                "score": None,
                "numeric": None,
            }
            for side in (COUNTERPART_AI, COUNTERPART_HUMAN)
        ]
        gap = counterpart_gaps(records)["s"]
        assert gap.reason == COUNTERPART_NOMINAL_ANSWER
        assert gap.gap is None

    def test_a_pair_with_one_arm_absent_from_the_records_reports_that(self) -> None:
        ai_arm = trust_likert_item("trust-01-ai", counterpart=COUNTERPART_AI, counterpart_pair="p")
        gap = counterpart_gaps([record_for(ai_arm, "FINAL ANSWER: C")])["p"]
        assert gap.reason == COUNTERPART_ARM_MISSING
        assert gap.gap is None


class TestTheBatteryEntryPointRunsTheNewGuards:
    """A guard nobody wired into the one entry point is a guard that never fires.

    Both new guards are checked through `survey_battery` itself rather than by reading its body: the
    battery is where the three older guards live precisely so a caller assembling the halves by hand
    cannot skip them, and a fourth and fifth added beside them have to actually be in that list.
    """

    @staticmethod
    def _serve(monkeypatch: pytest.MonkeyPatch, items: list[SurveyItem]) -> None:
        """Make `survey_battery` assemble exactly these items, with no local data file involved."""
        monkeypatch.setattr(survey_module, "load_authored_items", lambda _data_dir: list(items))
        monkeypatch.setattr(
            survey_module, "load_published_instruments", lambda _data_dir, instruments=(): []
        )

    def test_an_incomplete_counterpart_pair_is_refused_by_the_battery(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._serve(
            monkeypatch,
            [trust_likert_item("trust-01-ai", counterpart=COUNTERPART_AI, counterpart_pair="p")],
        )
        with pytest.raises(ValueError, match="carries arms"):
            survey_battery(data_dir=tmp_path)

    def test_mixed_ladder_lengths_are_refused_by_the_battery(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._serve(
            monkeypatch,
            [
                ordered_choice_item("risk-01"),
                ordered_choice_item("risk-02", options=(*THREE_RUNG_OPTIONS, "synthetic rung 4")),
            ],
        )
        with pytest.raises(ValueError, match="different lengths"):
            survey_battery(data_dir=tmp_path)

    def test_a_well_formed_new_family_battery_assembles(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The positive control on the two tests above: without it, a guard that raised on everything
        would pass both of them."""
        items = [
            trust_likert_item("trust-01-ai", counterpart=COUNTERPART_AI, counterpart_pair="p"),
            trust_likert_item(
                "trust-01-human", counterpart=COUNTERPART_HUMAN, counterpart_pair="p"
            ),
            trust_likert_item("trust-02", reverse_keyed=True),
            ordered_choice_item("risk-01"),
            ordered_choice_item("risk-02"),
            cheap_talk_item("deception-01"),
            labelled_choice_item("values-01"),
        ]
        self._serve(monkeypatch, items)
        assembled = survey_battery(data_dir=tmp_path)
        assert [item.item_id for item in assembled] == [item.item_id for item in items]


class TestTrustNumericItems:
    """The `<keep>`-tag numeric shape, reused for the trust family with no game to predict."""

    def test_a_numeric_item_outside_the_self_prediction_family_needs_no_game(self) -> None:
        item = trust_numeric_item()
        assert item.predicts_game is None
        assert item.numeric_max == 10

    def test_the_render_shows_the_rotated_example_and_the_items_own_bound(self) -> None:
        item = trust_numeric_item()
        prompt = render_survey_prompt(item, numeric_example=3)
        assert "<keep>3</keep>" in prompt
        assert "0 to 10" in prompt
        with pytest.raises(ValueError, match="caps answers"):
            render_survey_prompt(item, numeric_example=11)

    def test_the_swapped_framing_reflects_its_answer_back_to_the_first_framing(self) -> None:
        """The send-versus-keep counterbalance: an answer of 3 to the keep-framed stem is a send of
        10 - 3 = 7, and both renders have to aggregate on the send framing."""
        item = trust_numeric_item()
        as_authored = parse_survey_answer(item, "<keep>3</keep>", option_order=(0, 1))
        swapped = parse_survey_answer(item, "<keep>3</keep>", option_order=(1, 0))
        assert as_authored.numeric == 3
        assert swapped.numeric == 7

    def test_a_figure_above_the_bound_is_a_parse_failure_never_a_clamp(self) -> None:
        assert parse_survey_answer(trust_numeric_item(), "<keep>11</keep>").numeric is None


SYNTHETIC_PAYOFF_TABLE: list[list[int]] = [[91, 34], [86, 51], [79, 68]]


def allocation_spec(
    item_id: str = "ours-allocation-01",
    *,
    n_options: int = 3,
    counterpart: str = COUNTERPART_UNSPECIFIED,
    counterpart_pair: str | None = None,
) -> AuthoredItemSpec:
    """Build a synthetic authored allocation spec: shape here, payoffs in the local file."""
    return AuthoredItemSpec(
        item_id=item_id,
        family=FAMILY_TRUST_RECIPROCITY,
        instrument="synthetic-trust",
        kind=SURVEY_ALLOCATION,
        construct="A synthetic construct.",
        expected_direction="No prediction; this item exists only in a test.",
        tier=TIER_BREADTH,
        subscale="entrusted-split",
        n_options=n_options,
        counterpart=counterpart,
        counterpart_pair=counterpart_pair,
    )


def authored_file(
    path: Path, blocks: dict[str, dict[str, Any]], *, schema_version: int = SCHEMA_VERSION
) -> Path:
    """Write a synthetic authored.json holding exactly these item blocks, and return its directory.

    The shared elicitation blocks are always written, for every family that takes one, whether or not
    the specs being served include that family: the loader only demands the ones its registry carries,
    and writing them unconditionally keeps a narrowed-registry fixture from depending on which
    families happen to be listed.
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / AUTHORED_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "items": blocks,
                ELICITATION_BLOCKS_KEY: {
                    family: f"Placeholder closing question for {family}."
                    for family in FAMILIES_WITH_SHARED_ELICITATION
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def serve_specs(monkeypatch: pytest.MonkeyPatch, specs: tuple[AuthoredItemSpec, ...]) -> None:
    """Make the authored loader read exactly these specs, with the count guards out of the way."""
    monkeypatch.setattr(survey_module, "AUTHORED_ITEM_SPECS", specs)


def _registered(family: str, wording: str) -> int:
    """Count the specs the committed registry holds for one family on one wording arm.

    Read from the registry rather than written down, so a test that adds a synthetic spec can declare
    "one more than what is there" and keep meaning that after the next family lands.
    """
    return sum(
        1 for spec in AUTHORED_ITEM_SPECS if spec.family == family and spec.wording == wording
    )


class TestAuthoredAllocationItems:
    """An authored item can be a payoff table, which the spec could not express at all before.

    The gap was total rather than partial: `AuthoredItemSpec.__post_init__` had no allocation entry in
    its answer-shape table, so declaring one raised a bare `KeyError` naming the kind and nothing else;
    no field carried a payoff table; the text loader never read one; and `_authored_item` had no branch
    for the kind, so an allocation spec that got past the first three would have been built as a
    numeric item. The payoffs stay in the local file, because a payoff table is what the item IS.
    """

    def test_a_spec_can_declare_the_kind_at_all(self) -> None:
        assert allocation_spec().kind == SURVEY_ALLOCATION

    def test_a_spec_declaring_no_usable_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no usable answer shape"):
            allocation_spec(n_options=1)

    def test_a_spec_carrying_option_labels_is_refused(self) -> None:
        """An allocation option's meaning is its payoffs, so a label there is a second opinion about
        an answer that is already fully described -- refused at import, like every other shape rule."""
        with pytest.raises(ValueError, match="option_labels"):
            AuthoredItemSpec(
                item_id="ours-allocation-01",
                family=FAMILY_TRUST_RECIPROCITY,
                instrument="synthetic-trust",
                kind=SURVEY_ALLOCATION,
                construct="A synthetic construct.",
                expected_direction="No prediction; this item exists only in a test.",
                tier=TIER_BREADTH,
                n_options=3,
                option_labels=("joint-gain", "own-gain", "spite"),
            )

    def test_the_loaded_item_carries_the_payoffs_from_the_local_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        serve_specs(monkeypatch, (allocation_spec(),))
        data_dir = authored_file(
            tmp_path / "alloc",
            {
                "ours-allocation-01": {
                    "stem": "Synthetic allocation framing.",
                    "option_payoffs": SYNTHETIC_PAYOFF_TABLE,
                }
            },
        )
        (item,) = load_authored_items(data_dir)
        assert item.kind == SURVEY_ALLOCATION
        assert item.option_payoffs == ((91, 34), (86, 51), (79, 68))
        assert item.scale_points == 3

    def test_the_loaded_item_renders_lettered_and_scores_in_its_payoffs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The whole point of landing the kind: it has to reach a prompt and a score, not just load.

        Hand calculation: option B is the second canonical option, whose (self, other) pair is
        (86, 51), and an allocation answer scores as the payoff to the OTHER party -- so 51.0.
        """
        serve_specs(monkeypatch, (allocation_spec(),))
        data_dir = authored_file(
            tmp_path / "alloc",
            {
                "ours-allocation-01": {
                    "stem": "Synthetic allocation framing.",
                    "option_payoffs": SYNTHETIC_PAYOFF_TABLE,
                }
            },
        )
        (item,) = load_authored_items(data_dir)
        prompt = render_survey_prompt(item)
        assert "91" in prompt
        assert "34" in prompt
        answer = parse_survey_answer(item, "FINAL ANSWER: B")
        assert answer.canonical_index == 1
        assert answer.score == pytest.approx(51.0)
        assert (answer.payoff_self, answer.payoff_other) == (86, 51)

    @pytest.mark.parametrize(
        ("payoffs", "expected"),
        [
            ([[91, 34], [86, 51]], "supplies 2 allocation options"),
            ([[91, 34], [86, 51], [79, 68], [70, 70]], "supplies 4 allocation options"),
            ([[91, 34], [86, 51], [79]], "integer pairs"),
            ([[91, 34], [86, 51], [79, "68"]], "integer pairs"),
            ("not a table", "integer pairs"),
        ],
    )
    def test_a_payoff_table_of_the_wrong_shape_is_refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        payoffs: object,
        expected: str,
    ) -> None:
        """A re-shaped table renders, parses and scores perfectly while measuring a different task."""
        serve_specs(monkeypatch, (allocation_spec(),))
        data_dir = authored_file(
            tmp_path / "alloc",
            {
                "ours-allocation-01": {
                    "stem": "Synthetic allocation framing.",
                    "option_payoffs": payoffs,
                }
            },
        )
        with pytest.raises(ValueError, match=expected):
            load_authored_items(data_dir)

    def test_a_file_with_no_payoff_table_at_all_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        serve_specs(monkeypatch, (allocation_spec(),))
        data_dir = authored_file(
            tmp_path / "alloc", {"ours-allocation-01": {"stem": "Synthetic allocation framing."}}
        )
        with pytest.raises(ValueError, match="integer pairs"):
            load_authored_items(data_dir)


class TestTheNumericBoundIsCheckedAgainstTheFile:
    """The authoring passes emit `numeric_max` into their item files as well as into the spec.

    Null, 0 and absent all mean "answers through something other than a bounded integer", which is the
    spec field's own default, so a flat authored file loads unchanged. A figure that disagrees with the
    tracked spec is an error rather than a preference: the bound is what a swapped render's answer is
    reflected through, so the two sides disagreeing means the item is scored against a bound its author
    did not intend.
    """

    @staticmethod
    def _numeric_spec() -> AuthoredItemSpec:
        return AuthoredItemSpec(
            item_id="ours-trust-send-01",
            family=FAMILY_TRUST_RECIPROCITY,
            instrument="synthetic-trust",
            kind=SURVEY_NUMERIC,
            construct="A synthetic construct.",
            expected_direction="No prediction; this item exists only in a test.",
            tier=TIER_BREADTH,
            numeric_max=10,
            requires_swapped_stem=True,
        )

    def _load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, block: dict[str, Any]
    ) -> list[SurveyItem]:
        serve_specs(monkeypatch, (self._numeric_spec(),))
        return load_authored_items(authored_file(tmp_path / "bound", {"ours-trust-send-01": block}))

    def test_an_agreeing_bound_loads(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        block = {
            "stem": "Synthetic send-framed stem.",
            "stem_swapped": "Synthetic keep-framed stem.",
            "numeric_max": 10,
        }
        (item,) = self._load(monkeypatch, tmp_path, block)
        assert item.numeric_max == 10

    @pytest.mark.parametrize("declared", [None, 0])
    def test_either_no_shape_convention_loads_on_a_non_numeric_item(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, declared: int | None
    ) -> None:
        """The normalisation this exists for: the authoring passes emit the key on every item, some
        writing null and some 0 for "this item is not numeric", and both are the spec's own default."""
        spec = AuthoredItemSpec(
            item_id="ours-trust-01",
            family=FAMILY_TRUST_RECIPROCITY,
            instrument="synthetic-trust",
            kind=SURVEY_LIKERT,
            construct="A synthetic construct.",
            expected_direction="No prediction; this item exists only in a test.",
            tier=TIER_BREADTH,
            n_options=5,
        )
        serve_specs(monkeypatch, (spec,))
        block = {
            "stem": "Synthetic statement.",
            "options": [f"synthetic anchor {index}" for index in range(1, 6)],
            "numeric_max": declared,
        }
        (item,) = load_authored_items(
            authored_file(tmp_path / f"bound-{declared}", {"ours-trust-01": block})
        )
        assert item.numeric_max == 0

    def test_a_disagreeing_bound_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        block = {
            "stem": "Synthetic send-framed stem.",
            "stem_swapped": "Synthetic keep-framed stem.",
            "numeric_max": 100,
        }
        with pytest.raises(ValueError, match="while its tracked spec declares 10"):
            self._load(monkeypatch, tmp_path, block)

    def test_a_null_bound_on_a_numeric_item_is_refused_as_a_disagreement(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Null normalises to 0, which is a real disagreement with a bounded item rather than a gap."""
        block = {
            "stem": "Synthetic send-framed stem.",
            "stem_swapped": "Synthetic keep-framed stem.",
            "numeric_max": None,
        }
        spec = replace(self._numeric_spec(), item_id="ours-trust-send-02")
        serve_specs(monkeypatch, (spec,))
        data_dir = authored_file(tmp_path / "bound2", {"ours-trust-send-02": block})
        with pytest.raises(ValueError, match="while its tracked spec declares 10"):
            load_authored_items(data_dir)

    def test_a_non_integer_bound_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        block = {
            "stem": "Synthetic send-framed stem.",
            "stem_swapped": "Synthetic keep-framed stem.",
            "numeric_max": "10",
        }
        with pytest.raises(TypeError, match="neither a whole number nor null"):
            self._load(monkeypatch, tmp_path, block)


class TestAuthoredNeutralTwins:
    """An authored item can be a neutral twin, which the spec could not express before.

    Three authoring passes hit this wall: `AuthoredItemSpec` carried neither `wording` nor `twin_of`,
    `_authored_item` set neither on the item it built, and `games.survey.wording_gap` keys on the
    record's `twin_of` -- so an authored twin was administered, scored, and then silently pooled into
    the as-published composite it was supposed to be differenced against.
    """

    @staticmethod
    def _parent(item_id: str = "ours-trust-01") -> AuthoredItemSpec:
        return AuthoredItemSpec(
            item_id=item_id,
            family=FAMILY_TRUST_RECIPROCITY,
            instrument="synthetic-trust",
            kind=SURVEY_LIKERT,
            construct="A synthetic construct.",
            expected_direction="No prediction; this item exists only in a test.",
            tier=TIER_BREADTH,
            subscale="trust-attitude",
            n_options=5,
        )

    @classmethod
    def _twin(
        cls, *, item_id: str = "ours-trust-01-neutral01", parent: str = "ours-trust-01"
    ) -> AuthoredItemSpec:
        return replace(cls._parent(), item_id=item_id, wording=WORDING_NEUTRAL_TWIN, twin_of=parent)

    def test_a_spec_can_declare_both_halves_of_the_wording_control(self) -> None:
        twin = self._twin()
        assert (twin.wording, twin.twin_of) == (WORDING_NEUTRAL_TWIN, "ours-trust-01")

    def test_the_default_is_the_as_published_arm(self) -> None:
        assert self._parent().wording == WORDING_AS_PUBLISHED
        assert self._parent().twin_of is None

    def test_a_twin_wording_with_no_parent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names its as-published parent"):
            replace(self._parent(), wording=WORDING_NEUTRAL_TWIN)

    def test_a_parent_pointer_with_the_published_wording_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names its as-published parent"):
            replace(self._parent(), item_id="ours-trust-01-neutral01", twin_of="ours-trust-01")

    def test_an_unregistered_wording_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown wording"):
            replace(self._parent(), wording="paraphrased", twin_of="ours-trust-01")

    def test_the_loaded_twin_carries_its_arm_and_its_parent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        serve_specs(monkeypatch, (self._parent(), self._twin()))
        blocks = {
            spec.item_id: {
                "stem": f"Synthetic statement for {spec.item_id}.",
                "options": [f"synthetic anchor {index}" for index in range(1, 6)],
            }
            for spec in (self._parent(), self._twin())
        }
        parent, twin = load_authored_items(authored_file(tmp_path / "twins", blocks))
        assert (parent.wording, parent.twin_of) == (WORDING_AS_PUBLISHED, None)
        assert (twin.wording, twin.twin_of) == (WORDING_NEUTRAL_TWIN, parent.item_id)
        assert_every_twin_pairs([parent, twin])

    def test_the_authored_wording_gap_matches_the_hand_calculation(self) -> None:
        """Hand calculation: the parent answers D, the fourth anchor, scoring 4.0; its twin answers B,
        the second, scoring 2.0. One twinned item on each side, so the gap is 4.0 - 2.0 = +2.0. Before
        the twin fields existed, both records carried wording=as-published and twin_of=None, so this
        read `n_twinned_items=0` and no gap at all.
        """
        parent = trust_likert_item("ours-trust-01")
        twin = replace(
            parent,
            item_id="ours-trust-01-neutral01",
            wording=WORDING_NEUTRAL_TWIN,
            twin_of=parent.item_id,
        )
        gaps = wording_gap(
            [record_for(parent, "FINAL ANSWER: D"), record_for(twin, "FINAL ANSWER: B")]
        )
        reading = gaps[("synthetic-trust", "trust-attitude")]
        assert reading.as_published == pytest.approx(4.0)
        assert reading.neutral_twin == pytest.approx(2.0)
        assert reading.gap == pytest.approx(2.0)
        assert reading.n_twinned_items == 1

    def test_a_twin_is_not_counted_as_one_of_its_familys_planned_items(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The planned count is of as-published specs, so a family cannot pay for a missing parent
        with a twin -- and its twin count is the denominator `wording_gap` reports."""
        self._land(monkeypatch, (self._parent(), self._twin()), items=1, twins=1)
        survey_module._assert_planned_family_counts_hold({FAMILY_TRUST_RECIPROCITY})

    def test_a_twin_counted_as_a_parent_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._land(monkeypatch, (self._parent(), self._twin()), items=2, twins=0)
        with pytest.raises(RuntimeError, match="wrong item count"):
            survey_module._assert_planned_family_counts_hold({FAMILY_TRUST_RECIPROCITY})

    def test_a_missing_twin_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A family that landed its items and half its twins publishes a wording gap over half the
        comparison it was designed for, and every count in the readout still reads plausible."""
        self._land(monkeypatch, (self._parent(), self._twin()), items=1, twins=2)
        with pytest.raises(RuntimeError, match="wrong neutral-twin count"):
            survey_module._assert_planned_family_counts_hold({FAMILY_TRUST_RECIPROCITY})

    def test_an_unplanned_twin_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._land(monkeypatch, (self._parent(), self._twin()), items=1, twins=0)
        with pytest.raises(RuntimeError, match="wrong neutral-twin count"):
            survey_module._assert_planned_family_counts_hold({FAMILY_TRUST_RECIPROCITY})

    def test_a_twin_naming_an_unregistered_parent_is_refused_at_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            survey_module,
            "AUTHORED_ITEM_SPECS",
            (self._twin(item_id="absent-neutral01", parent="absent"),),
        )
        with pytest.raises(RuntimeError, match="which no registered spec declares"):
            survey_module._assert_authored_twins_name_registered_parents()

    def test_a_twin_of_a_twin_is_refused_at_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grandchild = self._twin(
            item_id="ours-trust-01-neutral01-neutral01", parent="ours-trust-01-neutral01"
        )
        monkeypatch.setattr(
            survey_module, "AUTHORED_ITEM_SPECS", (self._parent(), self._twin(), grandchild)
        )
        with pytest.raises(RuntimeError, match="no published side"):
            survey_module._assert_authored_twins_name_registered_parents()

    def test_a_twin_of_another_familys_item_is_refused_at_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parent = replace(self._parent(), family=FAMILY_VALUES_FORCED_CHOICE)
        monkeypatch.setattr(survey_module, "AUTHORED_ITEM_SPECS", (parent, self._twin()))
        with pytest.raises(RuntimeError, match="in family"):
            survey_module._assert_authored_twins_name_registered_parents()

    def test_the_registry_as_committed_passes(self) -> None:
        """The positive control on the four refusals above."""
        survey_module._assert_authored_twins_name_registered_parents()

    @staticmethod
    def _land(
        monkeypatch: pytest.MonkeyPatch,
        specs: tuple[AuthoredItemSpec, ...],
        *,
        items: int,
        twins: int,
    ) -> None:
        """Add these specs to the trust family, declaring `items` and `twins` MORE than it carries.

        Appended to the real registry rather than replacing it, so the other landed families still
        satisfy their own counts and the failure under test is the trust family's alone. Both counts
        are relative for that same reason: the trust family carries real specs now, so a literal
        declared count would make the guard raise about the real items rather than about the
        synthetic pair each test below is written to exercise.
        """
        monkeypatch.setattr(survey_module, "AUTHORED_ITEM_SPECS", (*AUTHORED_ITEM_SPECS, *specs))
        monkeypatch.setattr(
            survey_module,
            "FAMILIES_AWAITING_ITEMS",
            FAMILIES_AWAITING_ITEMS - {FAMILY_TRUST_RECIPROCITY},
        )
        monkeypatch.setattr(
            survey_module,
            "PLANNED_FAMILY_ITEM_COUNTS",
            {
                **PLANNED_FAMILY_ITEM_COUNTS,
                FAMILY_TRUST_RECIPROCITY: _registered(
                    FAMILY_TRUST_RECIPROCITY, WORDING_AS_PUBLISHED
                )
                + items,
            },
        )
        monkeypatch.setattr(
            survey_module,
            "PLANNED_FAMILY_TWIN_COUNTS",
            {
                **PLANNED_FAMILY_TWIN_COUNTS,
                FAMILY_TRUST_RECIPROCITY: _registered(
                    FAMILY_TRUST_RECIPROCITY, WORDING_NEUTRAL_TWIN
                )
                + twins,
            },
        )


class TestTheOrientationReadingFollowsThePayoffs:
    """The prosocial/individualistic/competitive reading is a fact about an item's numbers.

    It used to be keyed on the instrument being named `triple-dominance`, so an authored triple built
    the same way -- the counterpart dominance triples declare `ours-allocation-counterpart` -- would
    have been administered under both orders at every sample and every record would have carried a null
    orientation. Not a wrong number: the one reading those items exist for, absent, with nothing in the
    output saying it was missing.

    The triple below is invented for this file. Its arithmetic, worked out on paper: option 1 has the
    joint maximum (48 + 44 = 92 against 85 and 76), option 2 the own maximum (61) and option 3 the
    largest difference (57 - 19 = 38 against 4 and 41 - 24 = 17), so the three orientations land on
    three different options and the triple separates.
    """

    SEPARATING_TRIPLE = ((48, 44), (61, 24), (57, 19))

    def test_the_triple_separates_the_three_orientations(self) -> None:
        assert survey_module.allocation_orientations(self.SEPARATING_TRIPLE) == (
            "prosocial",
            "individualistic",
            "competitive",
        )

    def test_an_authored_triple_under_any_instrument_name_gets_its_orientation(self) -> None:
        item = SurveyItem(
            item_id="ours-allocation-counterpart-dominance-01-ai",
            family=FAMILY_TRUST_RECIPROCITY,
            instrument="ours-allocation-counterpart",
            kind=SURVEY_ALLOCATION,
            stem="Synthetic allocation framing.",
            construct="A synthetic construct.",
            expected_direction="No prediction; this item exists only in a test.",
            subscale="dominance-triple",
            option_payoffs=self.SEPARATING_TRIPLE,
        )
        records = [record_for(item, f"FINAL ANSWER: {letter}") for letter in ("A", "B", "C")]
        assert [record["orientation"] for record in records] == [
            "prosocial",
            "individualistic",
            "competitive",
        ]
        assert survey_module.orientation_counts(records) == {
            "competitive": 1,
            "individualistic": 1,
            "prosocial": 1,
        }

    def test_a_nine_option_slider_carries_no_orientation(self) -> None:
        """Most allocation items have no orientation reading, and that is what they are rather than an
        error: no slider option pays the chooser less in order to pay the other party less still."""
        slider = tuple((85 - index * 5, 15 + index * 5) for index in range(9))
        assert survey_module.allocation_orientations(slider) is None

    def test_a_triple_that_does_not_separate_carries_no_orientation(self) -> None:
        """One option winning two maxima means the item cannot tell those orientations apart."""
        assert survey_module.allocation_orientations(((90, 10), (50, 40), (40, 30))) is None

    def test_the_published_instrument_still_refuses_a_triple_that_does_not_separate(self) -> None:
        """Same arithmetic, stricter contract: for that instrument a non-separating triple is a
        corrupted item file, so it raises where the general reading returns None."""
        with pytest.raises(ValueError, match="do not separate the three orientations"):
            survey_module.triple_dominance_orientations(((90, 10), (50, 40), (40, 30)))
        with pytest.raises(ValueError, match="needs 3 options"):
            survey_module.triple_dominance_orientations(((90, 10), (50, 40)))

    def test_an_unparsed_render_carries_no_orientation(self) -> None:
        item = SurveyItem(
            item_id="ours-allocation-counterpart-dominance-01-ai",
            family=FAMILY_TRUST_RECIPROCITY,
            instrument="ours-allocation-counterpart",
            kind=SURVEY_ALLOCATION,
            stem="Synthetic allocation framing.",
            construct="A synthetic construct.",
            expected_direction="No prediction; this item exists only in a test.",
            subscale="dominance-triple",
            option_payoffs=self.SEPARATING_TRIPLE,
        )
        assert record_for(item, "no answer here")["orientation"] is None


class TestScoredSubscalesShareOneUnit:
    """Landing allocation items into an authored instrument makes a new unit mix reachable.

    A subscale composite is a mean over per-item scores. A Likert answer scores as an anchor point on
    a 1-to-5 ladder, an allocation answer as the payoff it sent the other party, and an ordered-choice
    answer as a rung number -- so a subscale holding two of those kinds averages two units into one
    index in neither. The published instruments are single-kind throughout, which is why nothing
    refused this before; an authored instrument holding both is new.
    """

    def test_a_subscale_mixing_a_likert_item_with_an_allocation_item_is_refused(self) -> None:
        likert = trust_likert_item("ours-trust-01")
        allocation = SurveyItem(
            item_id="ours-trust-alloc-01",
            family=FAMILY_TRUST_RECIPROCITY,
            instrument="synthetic-trust",
            kind=SURVEY_ALLOCATION,
            stem="Synthetic allocation framing.",
            construct="A synthetic construct.",
            expected_direction="No prediction; this item exists only in a test.",
            subscale=likert.subscale,
            option_payoffs=((91, 34), (86, 51), (79, 68)),
        )
        with pytest.raises(ValueError, match="mixes scored kinds"):
            survey_module.assert_scored_subscales_share_a_unit([likert, allocation])

    def test_the_same_two_kinds_in_two_subscales_are_allowed(self) -> None:
        """The positive control: the honest form of the design above is one subscale per kind."""
        likert = trust_likert_item("ours-trust-01")
        allocation = SurveyItem(
            item_id="ours-trust-alloc-01",
            family=FAMILY_TRUST_RECIPROCITY,
            instrument="synthetic-trust",
            kind=SURVEY_ALLOCATION,
            stem="Synthetic allocation framing.",
            construct="A synthetic construct.",
            expected_direction="No prediction; this item exists only in a test.",
            subscale="entrusted-split",
            option_payoffs=((91, 34), (86, 51), (79, 68)),
        )
        survey_module.assert_scored_subscales_share_a_unit([likert, allocation])

    def test_unscored_kinds_are_not_counted(self) -> None:
        """A tag and a bounded integer carry no score, so they cannot mix a composite's units."""
        survey_module.assert_scored_subscales_share_a_unit(
            [
                replace(trust_likert_item("ours-trust-01"), subscale="shared"),
                replace(trust_numeric_item("ours-trust-send-01"), subscale="shared"),
                replace(cheap_talk_item("deception-01"), subscale="shared"),
            ]
        )

    def test_the_battery_entry_point_runs_the_unit_check(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        likert = trust_likert_item("ours-trust-01")
        allocation = SurveyItem(
            item_id="ours-trust-alloc-01",
            family=FAMILY_TRUST_RECIPROCITY,
            instrument="synthetic-trust",
            kind=SURVEY_ALLOCATION,
            stem="Synthetic allocation framing.",
            construct="A synthetic construct.",
            expected_direction="No prediction; this item exists only in a test.",
            subscale=likert.subscale,
            option_payoffs=((91, 34), (86, 51), (79, 68)),
        )
        monkeypatch.setattr(survey_module, "load_authored_items", lambda _dir: [likert, allocation])
        monkeypatch.setattr(
            survey_module, "load_published_instruments", lambda _data_dir, instruments=(): []
        )
        with pytest.raises(ValueError, match="mixes scored kinds"):
            survey_battery(data_dir=tmp_path)

    def test_the_real_battery_satisfies_it(self, synthetic_data_dir: Path) -> None:
        """Which is what makes the guard safe to add to a battery that already exists."""
        survey_module.assert_scored_subscales_share_a_unit(
            survey_battery(data_dir=synthetic_data_dir)
        )


class TestTwinIdsDeriveFromTheirParent:
    """The mispointing every field comparison misses: a twin repointed at a sibling of its parent.

    `assert_every_twin_pairs` compared instrument, subscale, kind, keying and rung count -- all of
    which a same-subscale, same-keying sibling agrees on. So the guard passed, and `wording_gap` then
    differenced two unrelated items and reported the result as a wording effect with a plausible
    magnitude. The close is the id derivation the published loader already followed.
    """

    @staticmethod
    def _family(*, twin_of: str) -> list[SurveyItem]:
        """A parent, a same-subscale same-keying sibling, and a twin pointing wherever asked."""
        parent = trust_likert_item("synthetic-trust-01")
        sibling = trust_likert_item("synthetic-trust-02")
        twin = replace(
            parent,
            item_id=f"synthetic-trust-01{NEUTRAL_TWIN_ID_SUFFIX}01",
            wording=WORDING_NEUTRAL_TWIN,
            twin_of=twin_of,
        )
        return [parent, sibling, twin]

    def test_a_twin_repointed_at_a_sibling_is_refused(self) -> None:
        """The sabotage this guard exists for: every compared field still agrees, so before the
        derivation check this battery assembled and its wording gap was a number about nothing."""
        items = self._family(twin_of="synthetic-trust-02")
        parent, sibling, twin = items
        assert (sibling.instrument, sibling.subscale, sibling.reverse_keyed) == (
            parent.instrument,
            parent.subscale,
            parent.reverse_keyed,
        )
        assert sibling.scale_points == parent.scale_points
        assert twin.twin_of == sibling.item_id
        with pytest.raises(ValueError, match="not derived from that parent"):
            assert_every_twin_pairs(items)

    def test_the_correctly_pointed_twin_passes(self) -> None:
        """The positive control: without it, a guard refusing every twin would pass the test above."""
        assert_every_twin_pairs(self._family(twin_of="synthetic-trust-01"))

    def test_the_battery_entry_point_runs_the_derivation_check(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A guard nobody wired into `survey_battery` is a guard that never fires on a real run."""
        monkeypatch.setattr(
            survey_module,
            "load_authored_items",
            lambda _data_dir: self._family(twin_of="synthetic-trust-02"),
        )
        monkeypatch.setattr(
            survey_module, "load_published_instruments", lambda _data_dir, instruments=(): []
        )
        with pytest.raises(ValueError, match="not derived from that parent"):
            survey_battery(data_dir=tmp_path)

    @pytest.mark.parametrize(
        "item_id",
        [
            "synthetic-trust-02-neutral01",
            "synthetic-trust-01neutral01",
            "synthetic-trust-01-neutral",
            "neutral01",
        ],
    )
    def test_an_id_that_does_not_derive_is_refused(self, item_id: str) -> None:
        with pytest.raises(ValueError, match="not derived from that parent"):
            assert_twin_id_derives_from_parent(item_id, "synthetic-trust-01")

    @pytest.mark.parametrize(
        "item_id", ["synthetic-trust-01-neutral01", "synthetic-trust-01-neutral02"]
    )
    def test_a_derived_id_is_accepted(self, item_id: str) -> None:
        assert_twin_id_derives_from_parent(item_id, "synthetic-trust-01")

    def test_an_authored_twin_whose_id_does_not_derive_is_refused_at_import(self) -> None:
        with pytest.raises(ValueError, match="not derived from that parent"):
            AuthoredItemSpec(
                item_id="ours-trust-99-neutral01",
                family=FAMILY_TRUST_RECIPROCITY,
                instrument="synthetic-trust",
                kind=SURVEY_LIKERT,
                construct="A synthetic construct.",
                expected_direction="No prediction; this item exists only in a test.",
                tier=TIER_BREADTH,
                n_options=5,
                wording=WORDING_NEUTRAL_TWIN,
                twin_of="ours-trust-01",
            )

    def test_the_published_loader_builds_ids_this_guard_accepts(
        self, synthetic_data_dir: Path
    ) -> None:
        """The convention is the published loader's own, so the whole real battery has to satisfy it
        -- which is also what makes the guard safe to add to an existing battery."""
        items = survey_battery(data_dir=synthetic_data_dir)
        twins = [item for item in items if item.twin_of is not None]
        assert twins
        for twin in twins:
            assert_twin_id_derives_from_parent(twin.item_id, twin.twin_of or "")


class TestTheNumericExampleRotationIsPerItem:
    """A fixed rotation of (25, 60, 75) made a 10-bounded item unrenderable under every value.

    `render_survey_prompt` refuses an example above the item's own bound, because an example the parser
    would reject demonstrates a wrong answer. The rotation the eval section indexed was module-wide, so
    for a trust-game item asking how much of ten units to send, all three values were out of range and
    the item could not be rendered at all -- not degraded, refused. Deriving the values from the item's
    own bound keeps the anti-anchoring rotation and makes it renderable at any bound.
    """

    def test_a_percent_bounded_item_rotates_through_exactly_what_it_always_did(self) -> None:
        """The pin: every numeric item registered today is percent-bounded, and its renders must not
        move by one byte. 25, 60 and 75 percent of 100 are the three values the battery has always
        shown."""
        assert numeric_examples_for_bound(100) == (25, 60, 75)
        assert NUMERIC_EXAMPLE_ROTATION == (25, 60, 75)
        assert NUMERIC_EXAMPLE_PERCENTS == (25, 60, 75)

    def test_a_ten_bounded_item_rotates_through_small_figures(self) -> None:
        """Hand calculation, rounding half up: 25% of 10 is 2.5 -> 3, 60% is 6, 75% is 7.5 -> 8."""
        assert numeric_examples_for_bound(10) == (3, 6, 8)

    @pytest.mark.parametrize("bound", [1, 2, 3, 4, 5, 10, 20, 50, 100, 1000])
    def test_every_rotation_value_is_a_legal_answer_for_its_own_bound(self, bound: int) -> None:
        rotation = numeric_examples_for_bound(bound)
        assert rotation
        assert len(set(rotation)) == len(rotation)
        assert all(1 <= value <= bound for value in rotation)

    def test_a_bound_too_small_to_carry_three_values_yields_fewer_rather_than_repeats(self) -> None:
        """A repeated value would say an item was shown three distinct examples when it saw one, which
        is exactly what an anchoring read off `numeric_example` in the trace would get wrong."""
        assert numeric_examples_for_bound(1) == (1,)
        assert numeric_examples_for_bound(2) == (1, 2)

    def test_a_bound_of_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="offers only the answer 0"):
            numeric_examples_for_bound(0)

    def test_a_non_numeric_item_has_no_rotation_to_ask_for(self) -> None:
        with pytest.raises(ValueError, match="two kinds mixed up"):
            numeric_example_rotation(trust_likert_item("ours-trust-01"))

    def test_every_value_of_a_ten_bounded_items_rotation_renders(self) -> None:
        """The defect itself: before this, all three values raised `caps answers at 10`."""
        item = trust_numeric_item()
        for example in numeric_example_rotation(item):
            prompt = render_survey_prompt(item, numeric_example=example)
            assert f"<keep>{example}</keep>" in prompt
            assert "0 to 10" in prompt

    def test_the_eval_section_takes_the_rotation_from_the_item(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The wiring, not just the helper: `games.evals` indexed the module-wide tuple, so a battery
        holding a 10-bounded item raised on the FIRST render and no self-report section ran at all."""
        small = trust_numeric_item("ours-trust-send-01")
        percent = replace(small, item_id="ours-percent-01", numeric_max=100)
        monkeypatch.setattr(evals_module, "survey_battery", lambda **_kwargs: [small, percent])
        renderings = evals_module._survey_renderings(
            EvalConfig(survey_samples=2, survey_data_dir=tmp_path)
        )
        by_item: dict[str, list[int]] = {}
        for rendering, prompt in renderings:
            example = rendering.numeric_example
            assert example is not None
            assert f"<keep>{example}</keep>" in prompt
            by_item.setdefault(rendering.item.item_id, []).append(example)
        assert set(by_item["ours-trust-send-01"]) == {3, 6, 8}
        assert set(by_item["ours-percent-01"]) == {25, 60, 75}


class TestCounterpartsOnlyOnDifferenceableKinds:
    """A nominal pair costs a GPU to administer and cannot yield the number it exists for.

    `counterpart_gaps` reports "a nominal answer has no value to difference" as a recorded reason,
    which is the honest reading -- but it arrives after both arms have been rendered under both orders
    at every sample of every checkpoint of every arm. So the refusal belongs at construction, in both
    validators, the same way the counterpart/pair-key rule already was.
    """

    @pytest.mark.parametrize("kind", sorted(set(SURVEY_KINDS) - DIFFERENCEABLE_KINDS))
    def test_an_item_of_an_undifferenceable_kind_is_refused(self, kind: str) -> None:
        base = labelled_choice_item("values-01") if kind == SURVEY_CHOICE else forced_tag_item()
        assert base.kind == kind, "a new undifferenceable kind needs a case here"
        with pytest.raises(ValueError, match="has no value to difference"):
            replace(base, counterpart=COUNTERPART_AI, counterpart_pair="p")

    def test_a_spec_of_an_undifferenceable_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="carry a value a pair's difference"):
            AuthoredItemSpec(
                item_id="values-01-ai",
                family=FAMILY_VALUES_FORCED_CHOICE,
                instrument="synthetic-values",
                kind=SURVEY_CHOICE,
                construct="A synthetic construct.",
                expected_direction="No prediction; this item exists only in a test.",
                tier=TIER_BREADTH,
                n_options=2,
                counterpart=COUNTERPART_AI,
                counterpart_pair="p",
            )

    def test_every_differenceable_kind_still_accepts_a_pair(self) -> None:
        """The positive control, and the contract the counterpart-pair authors write against: the five
        kinds whose answers a difference can be taken of all still take both arms."""
        arms = [
            replace(
                trust_likert_item("trust-01"), counterpart=COUNTERPART_AI, counterpart_pair="p"
            ),
            replace(
                ordered_choice_item("risk-01"), counterpart=COUNTERPART_AI, counterpart_pair="q"
            ),
            replace(
                trust_numeric_item("trust-send-01"),
                counterpart=COUNTERPART_AI,
                counterpart_pair="r",
            ),
            cheap_talk_item("deception-01", counterpart=COUNTERPART_AI, counterpart_pair="s"),
        ]
        assert {arm.kind for arm in arms} | {SURVEY_LIKERT} <= DIFFERENCEABLE_KINDS

    def test_an_allocation_arm_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Allocation is the kind the counterpart-pairs authors use most, and it loads from file."""
        specs = tuple(
            allocation_spec(
                f"ours-allocation-01-{side}", counterpart=side, counterpart_pair="alloc-pair"
            )
            for side in (COUNTERPART_AI, COUNTERPART_HUMAN)
        )
        serve_specs(monkeypatch, specs)
        blocks = {
            spec.item_id: {
                "stem": f"Synthetic allocation framing for {spec.item_id}.",
                "option_payoffs": SYNTHETIC_PAYOFF_TABLE,
            }
            for spec in specs
        }
        items = load_authored_items(authored_file(tmp_path / "alloc-pair", blocks))
        assert_every_counterpart_pair_is_complete(items)
        gap = counterpart_gaps([record_for(item, "FINAL ANSWER: A") for item in items])[
            "alloc-pair"
        ]
        assert gap.reason is None
        assert gap.gap == pytest.approx(0.0)


def write_synthetic_files(path: Path) -> Path | None:
    """Placeholder kept only so the module imports; see `synthetic_data_dir` for the real fixture."""
    return None


def _synthetic_payoffs(n_options: int, position: int) -> list[list[int]]:
    """Payoffs that separate the three orientations for any option count of three or more.

    Every digit is invented for this file; no pair reproduces a published item's.
    """
    if n_options == 3:
        return [[97 + position, 41], [88 + position, 88 + position], [93 + position, 23]]
    step = 60 // (n_options - 1)
    return [[80, 80 - index * step] for index in range(n_options)]


def _published_payload() -> dict[str, Any]:
    """Build a synthetic `published.json` covering EVERY registered instrument, with one twin each."""
    blocks: dict[str, Any] = {}
    for name, spec in PUBLISHED_INSTRUMENTS.items():
        if spec.kind == SURVEY_LIKERT:
            blocks[name] = {
                "anchors": [f"anchor {index}" for index in range(1, spec.scale_points + 1)],
                "items": [
                    {
                        "stem": f"Synthetic statement {position}.",
                        "neutral_stems": [f"Synthetic neutral statement {position}."],
                    }
                    for position in range(1, spec.n_items + 1)
                ],
            }
        else:
            blocks[name] = {
                "instructions": "Synthetic framing for an allocation task.",
                "items": [
                    {"option_payoffs": _synthetic_payoffs(spec.n_options, position)}
                    for position in range(1, spec.n_items + 1)
                ],
            }
    return {"schema_version": SCHEMA_VERSION, "instruments": blocks}


def _authored_payload() -> dict[str, Any]:
    """Build a synthetic `authored.json` derived from the tracked registry, placeholder text only."""
    items: dict[str, Any] = {}
    for spec in AUTHORED_ITEM_SPECS:
        block: dict[str, Any] = {"stem": f"Synthetic stem for {spec.item_id}."}
        if spec.requires_swapped_stem:
            block["stem_swapped"] = f"Synthetic swapped stem for {spec.item_id}."
        if spec.kind in (SURVEY_LIKERT, SURVEY_CHOICE, SURVEY_ORDERED_CHOICE):
            block["options"] = [
                f"synthetic option {index}" for index in range(1, spec.n_options + 1)
            ]
        if spec.kind in (SURVEY_TAGGED, SURVEY_CHEAP_TALK):
            block["vocabulary"] = [f"wordnumber{index}" for index in range(1, spec.n_tag_words + 1)]
        if spec.kind == SURVEY_ALLOCATION:
            block["option_payoffs"] = _synthetic_payoffs(spec.n_options, 1)
        items[spec.item_id] = block
    return {
        "schema_version": SCHEMA_VERSION,
        "items": items,
        ELICITATION_BLOCKS_KEY: {
            family: f"Placeholder closing question for {family}."
            for family in FAMILIES_WITH_SHARED_ELICITATION
        },
    }


@pytest.fixture(scope="module")
def synthetic_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A complete synthetic data directory: every published instrument plus every authored spec."""
    path = tmp_path_factory.mktemp("survey-new-families")
    (path / "published.json").write_text(json.dumps(_published_payload()), encoding="utf-8")
    (path / AUTHORED_FILENAME).write_text(json.dumps(_authored_payload()), encoding="utf-8")
    return path


# Everything about an item except its words, in a fixed order, so a digest over it is readable in a
# diff when it changes: this is exactly the tracked half of the item contract.
_PIN_SPEC_FIELDS: tuple[str, ...] = (
    "item_id",
    "family",
    "instrument",
    "kind",
    "tier",
    "subscale",
    "wording",
    "twin_of",
    "counterpart",
    "reverse_keyed",
    "scale_points",
    "predicts_game",
)


def battery_digest(items: list[SurveyItem]) -> str:
    """Digest every item's tracked spec AND every prompt it renders, in a stable order.

    The renders are in it on purpose: what "unchanged" has to mean for this battery is that the model
    receives the same bytes, and a spec-only digest would miss a changed instruction template. It is a
    hash rather than the strings themselves because the strings are item text, which never enters a
    tracked file.
    """
    hasher = hashlib.sha256()
    for item in sorted(items, key=lambda entry: entry.item_id):
        counterbalanced: list[tuple[int, ...] | None] = [order for _, order in battery_orders(item)]
        orders = counterbalanced or [None]
        # Per item, because the rotation is fractions of the item's own bound: for the percent-bounded
        # items this battery holds today that is (25, 60, 75) either way, and for a smaller-bounded
        # item landing later it is the difference between digesting its real renders and raising.
        examples: list[int | None] = (
            list(numeric_example_rotation(item)) if item.kind == SURVEY_NUMERIC else [None]
        )
        renders = [
            render_survey_prompt(item, option_order=order, numeric_example=example)
            for order in orders
            for example in examples
        ]
        hasher.update(
            json.dumps(
                [str(getattr(item, name)) for name in _PIN_SPEC_FIELDS] + renders,
                ensure_ascii=False,
            ).encode("utf-8")
        )
    return hasher.hexdigest()


# The battery as it stands, over synthetic item text derived from the tracked registry. Every value
# below was checked against a `git archive` checkout of the previous state, so this is a verified
# additivity pin rather than a snapshot of whatever the code happens to do now.
#
# Verified at the 2026-08-22 landing of the seven authored families (154 specs, 184 -> 338 items):
# the pre-change tree at cea5b68 reproduced the then-current digest exactly, and the digest over
# those same 184 ids AFTER the landing was byte-identical to it, so not one pre-existing item's spec
# or render moved.
#
# Re-verified at the 2026-08-24 own-action-rate elicitation repair, which is the first change to move
# these digests without adding a single item: the eight self-prediction stems were reworded and the
# loader now appends that family's shared closing block. Both digests below therefore moved and the
# counts did not, which is the shape of a rewording rather than a landing. Discharged the same way,
# with the split done per family so the claim is checkable rather than asserted -- each tree measured
# with its own loader and its own fixtures, `git archive HEAD` into /var/tmp for the pre-change side:
#   over the REAL item text, the 330 non-self-prediction items digest to
#     c3e9c4cc5ccc00daccdb1ffaaf94bd9173f5dfab77d0b6bc2c31de81d79adc5b before AND after;
#   over each tree's own synthetic fixtures, those same 330 digest to
#     10754bc0e53314cbe0b9cbe13a7cf1f6952f1ad3dc968c9447138174ff364777 before AND after.
# The eight family items moved in both (real 06fc3825 -> d849a9b3, synthetic 0d5531c2 -> 94caf222).
#
# Re-verified at the 2026-08-25 negative-control headroom landing (five adopted items plus the
# season promotion, 338 -> 343): the pre-change tree at 06e6469 reproduced both prior digests
# exactly (synthetic over its own fixtures; real over the current authored.json with the five new
# ids removed, which also proves the data-file rewrite left every existing item's text
# byte-identical). Per-item comparison across the trees: five ids added, none removed, and of the
# 338 pre-existing entries exactly one moved -- negative-control-season, in its tier field only
# (breadth -> core, the promotion), with its renders byte-identical on both the synthetic and the
# real side.
#
# THE UPDATE POINT: landing new item specs changes every number below, and that is correct -- the
# battery grew. When you update them, do it in the same commit as the specs, and check the previously
# existing items are untouched the way this one was: build the pre-change tree with
#   git archive <sha> | tar -x -C /var/tmp/<dir>
# and compare `battery_digest` over the items whose ids existed before, measuring EACH tree with its
# own fixtures rather than a hand-rolled synthetic file (one written for the 08-22 landing omitted
# the published neutral twins and read 114 items where this pin says 338). Never just bump a
# constant.
PIN_SYNTHETIC_N_ITEMS = 343
PIN_SYNTHETIC_DIGEST = "530b9abc15892209ad689e5b76ca9ac9f5ec9ae1d2bd4bd42018a69a0856a121"
PIN_FAMILY_COUNTS: dict[str, int] = {
    "altruism-past-behaviour-likert": 18,
    "competitiveness-likert": 28,
    "cooperative-orientation-likert": 26,
    "counterpart-pairs": 60,
    "deception-cheaptalk": 20,
    "graded-dimension-awareness": 12,
    "narcissism-likert": 36,
    "negative-control": 15,
    "prosocialness-likert": 32,
    "risk-preference-revealed": 12,
    "self-characterisation-open": 8,
    "self-prediction": 8,
    "svo-allocation": 15,
    "triple-dominance-allocation": 9,
    "trust-reciprocity": 24,
    "values-forced-choice": 20,
}
PIN_KIND_COUNTS: dict[str, int] = {
    "allocation": 44,
    "cheap-talk": 12,
    "choice": 42,
    "likert": 197,
    "numeric": 19,
    "ordered-choice": 18,
    "tagged": 11,
}
PIN_WORDING_COUNTS: dict[str, int] = {"as-published": 261, "neutral-twin": 82}
PIN_TIER_COUNTS: dict[str, int] = {"breadth": 278, "core": 65}
# The same digest over the REAL local item text. Depends on the local files, so it is checked only
# where they exist and skips loudly otherwise.
PIN_REAL_DIGEST = "de09dcef671831316412b14f2a65f81e64f6130e131c948ed62d4a59e8a7afcf"

_PIN_FAILURE = (
    "the pre-existing survey battery changed. This test exists to prove a change to games/survey.py "
    "was ADDITIVE: if you added item specs, update the pins beside this message in the same commit "
    "and verify the previously existing items are untouched (the procedure is in the comment above "
    "the pins). If you did not add items, something changed how the existing battery loads or "
    "renders, and that is the bug."
)


class TestTheExistingBatteryStillLoadsIdentically:
    """The additivity pin: readable counts first, so a failure says which axis moved, digest after."""

    def test_the_item_count_is_unchanged(self, synthetic_data_dir: Path) -> None:
        items = survey_battery(data_dir=synthetic_data_dir)
        assert len(items) == PIN_SYNTHETIC_N_ITEMS, _PIN_FAILURE

    def test_the_per_family_counts_are_unchanged(self, synthetic_data_dir: Path) -> None:
        items = survey_battery(data_dir=synthetic_data_dir)
        assert dict(sorted(Counter(item.family for item in items).items())) == PIN_FAMILY_COUNTS, (
            _PIN_FAILURE
        )

    def test_the_per_kind_counts_are_unchanged(self, synthetic_data_dir: Path) -> None:
        """No pre-existing item may have moved onto one of the new kinds."""
        items = survey_battery(data_dir=synthetic_data_dir)
        assert dict(sorted(Counter(item.kind for item in items).items())) == PIN_KIND_COUNTS, (
            _PIN_FAILURE
        )

    def test_the_wording_and_tier_splits_are_unchanged(self, synthetic_data_dir: Path) -> None:
        items = survey_battery(data_dir=synthetic_data_dir)
        assert dict(sorted(Counter(item.wording for item in items).items())) == PIN_WORDING_COUNTS
        assert dict(sorted(Counter(item.tier for item in items).items())) == PIN_TIER_COUNTS

    def test_every_spec_and_every_render_is_byte_identical(self, synthetic_data_dir: Path) -> None:
        items = survey_battery(data_dir=synthetic_data_dir)
        assert battery_digest(items) == PIN_SYNTHETIC_DIGEST, _PIN_FAILURE

    def test_the_new_kinds_are_registered_without_disturbing_the_old_ones(self) -> None:
        """The additive claim about the kind registry itself, stated separately from the counts."""
        assert SURVEY_ORDERED_CHOICE in SURVEY_KINDS
        assert SURVEY_CHEAP_TALK in SURVEY_KINDS
        assert SURVEY_KINDS[:5] == (
            SURVEY_LIKERT,
            "allocation",
            SURVEY_CHOICE,
            SURVEY_NUMERIC,
            SURVEY_TAGGED,
        )

    def test_the_real_local_battery_is_byte_identical(self) -> None:
        """Skips loudly on a fresh clone; on a machine holding the item text it is the strongest form
        of the pin, because it exercises the real anchors, payoffs, twins and swapped stems."""
        if not (REAL_DATA_DIR / AUTHORED_FILENAME).is_file():
            pytest.skip(
                f"{REAL_DATA_DIR / AUTHORED_FILENAME} is not on this machine (fresh clone); "
                f"assemble it as games/data/survey/README.md describes to run the real-text pin"
            )
        items = survey_battery(data_dir=REAL_DATA_DIR)
        assert len(items) == PIN_SYNTHETIC_N_ITEMS, _PIN_FAILURE
        assert dict(sorted(Counter(item.family for item in items).items())) == PIN_FAMILY_COUNTS, (
            _PIN_FAILURE
        )
        assert battery_digest(items) == PIN_REAL_DIGEST, _PIN_FAILURE

    def test_the_core_tier_filter_still_selects_what_it_did(self, synthetic_data_dir: Path) -> None:
        """A tier filter is the only way to select core, so a shifted tiering would silently change
        what the deliberated leg pays thinking-on completions for. Core rather than breadth on
        purpose: a breadth-only slice of the competitiveness index is all reverse-keyed and is
        refused by the keying guard, which `test_games_survey.py` already pins."""
        core = survey_battery(data_dir=synthetic_data_dir, tier=TIER_CORE)
        assert len(core) == PIN_TIER_COUNTS["core"]
