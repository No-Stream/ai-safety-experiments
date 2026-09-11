"""Tests for the self-report battery's items, parsers, loaders and composites.

Three things about the fixtures here, all deliberate.

**No item text appears in this file — published or authored.** The real strings live in the
gitignored `games/data/survey/` files, so every loader test writes a *synthetic* file under
`tmp_path` in the real schema. That is the same choice `games/tests/test_games_probes.py` makes for
DTBench, and it has a second payoff: there is no `games/tests/conftest.py` and so no collection gate
that could skip these on a missing data file. A `pytest.skip` that can go lenient is one of the four
bugs this codebase is built around, so the synthetic-fixture suite is green in a fresh clone with no
skips at all.

**The handful of tests that check the REAL local files skip loudly, saying why.** They pin the
repairs that live in the item text itself (the missing numeric example, the fixed contexts, the
swapped stems) and the item-text guard's extraction floor, none of which a synthetic fixture can
witness. On a fresh clone they skip with the reason; on this machine they run.

**The composite arithmetic is pinned against hand-computed numbers**, not against whatever the code
currently returns. A reverse-keyed item scored the right way and the wrong way both produce a
perfectly plausible subscale mean, so the only test that can catch a keying error is one whose
expected value was worked out on paper first. Each such test shows the arithmetic in a comment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from games.parsing import parse_split, parse_tag, strip_thinking
from games.probes import ORDER_AS_AUTHORED, ORDER_REVERSED, parse_final_answer
from games.prompts import assert_no_loaded_vocabulary
from games.survey import (
    ACQUIESCENCE_NO_POSITIVE_KEYED,
    ACQUIESCENCE_NO_REVERSE_KEYED,
    ACT_TAG,
    ANNOUNCE_TAG,
    AUTHORED_FILENAME,
    AUTHORED_ITEM_SPECS,
    ELICITATION_BLOCKS_KEY,
    FAMILIES,
    FAMILIES_WITH_SHARED_ELICITATION,
    FAMILY_NEGATIVE_CONTROL,
    FAMILY_SELF_CHARACTERISATION,
    FAMILY_SELF_PREDICTION,
    INSTRUMENT_COMPETITIVENESS_INDEX,
    INSTRUMENT_COOPERATIVE_ORIENTATION,
    INSTRUMENT_NARCISSISM,
    INSTRUMENT_SVO_SLIDER,
    INSTRUMENT_TRIPLE_DOMINANCE,
    NUMERIC_EXAMPLE_ROTATION,
    OPTION_LIST_KINDS,
    ORIENTATION_COMPETITIVE,
    ORIENTATION_INDIVIDUALISTIC,
    ORIENTATION_PROSOCIAL,
    PUBLISHED_INSTRUMENTS,
    SCHEMA_VERSION,
    SUBSCALE_INERT_FACT,
    SUBSCALE_INERT_LIKERT,
    SUBSCALE_INERT_NUMERIC,
    SUBSCALE_INERT_PREFERENCE,
    SUBSCALE_OWN_ACTION_RATE,
    SURVEY_ALLOCATION,
    SURVEY_CHEAP_TALK,
    SURVEY_CHOICE,
    SURVEY_LIKERT,
    SURVEY_NUMERIC,
    SURVEY_RECORD_FIELDS,
    SURVEY_TAGGED,
    TAG_EXAMPLE_PLACEHOLDER,
    TIER_BREADTH,
    TIER_CORE,
    VOCABULARY_KINDS,
    WORDING_AS_PUBLISHED,
    WORDING_NEUTRAL_TWIN,
    AuthoredItemSpec,
    SurveyAnswer,
    SurveyItem,
    acquiescence_index,
    assert_every_reverse_key_has_a_sibling,
    assert_every_twin_pairs,
    assert_unique_survey_ids,
    battery_orders,
    calibration_gaps,
    choice_response_distributions,
    choice_response_entropy,
    instrument_composites,
    load_authored_items,
    load_published_instruments,
    modal_choices,
    numeric_item_readings,
    orientation_counts,
    parse_rate_by_family,
    parse_rate_by_instrument,
    parse_survey_answer,
    per_item_scores,
    render_survey_prompt,
    subscale_composites,
    survey_battery,
    survey_record_fields,
    svo_angle,
    svo_mean_completion_angle,
    total_variation_distance,
    triple_dominance_orientations,
    wording_gap,
)
from scripts.scan_secrets import sources_from_json_payload

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATA_DIR = REPO_ROOT / "games" / "data" / "survey"

FIXTURE_PATH = Path(__file__).parent / "data" / "survey_adversarial_completions.json"

FIVE_POINT_ANCHORS = (
    "placeholder anchor one",
    "placeholder anchor two",
    "placeholder anchor three",
    "placeholder anchor four",
    "placeholder anchor five",
)

# A synthetic Likert instrument in the loaded file's shape, with the keying pattern that matters:
# one subscale carrying both keyings (so acquiescence is computable) and one carrying only reverse
# keys (which is the real competitiveness index's shape and must not be refused).
SYNTHETIC_INSTRUMENT = INSTRUMENT_COMPETITIVENESS_INDEX


def synthetic_published_file(
    path: Path, *, instrument: str = SYNTHETIC_INSTRUMENT, **overrides: Any
) -> Path:
    """Write a synthetic `published.json` matching one registered instrument's declared shape."""
    spec = PUBLISHED_INSTRUMENTS[instrument]
    if spec.kind == SURVEY_LIKERT:
        block: dict[str, Any] = {
            "anchors": list(FIVE_POINT_ANCHORS[: spec.scale_points])
            + [f"anchor {index}" for index in range(len(FIVE_POINT_ANCHORS), spec.scale_points)],
            "items": [
                {"stem": f"Synthetic statement {position}."}
                for position in range(1, spec.n_items + 1)
            ],
        }
    else:
        block = {
            "instructions": "Synthetic framing for an allocation task.",
            "items": [
                {"option_payoffs": _synthetic_payoffs(spec.n_options, position)}
                for position in range(1, spec.n_items + 1)
            ],
        }
    block.update(overrides)
    payload = {"schema_version": SCHEMA_VERSION, "instruments": {instrument: block}}
    path.mkdir(parents=True, exist_ok=True)
    (path / "published.json").write_text(json.dumps(payload), encoding="utf-8")
    return path


def _synthetic_payoffs(n_options: int, position: int) -> list[list[int]]:
    """Build payoff options that separate the three orientations, for any option count >= 3.

    Every digit here was invented for this file; no payoff pair reproduces a published item's.
    """
    if n_options == 3:
        # own-max, joint-max/equal, difference-max -- one of each, which is the shape the loader's
        # orientation guard demands of a three-option allocation item.
        return [[97 + position, 41], [88 + position, 88 + position], [93 + position, 23]]
    # A slider: self fixed, other descending, so the options are ordered by generosity.
    step = 60 // (n_options - 1)
    return [[80, 80 - index * step] for index in range(n_options)]


def synthetic_authored_block(spec: AuthoredItemSpec) -> dict[str, Any]:
    """Build one authored item's synthetic text payload, derived from its tracked spec."""
    block: dict[str, Any] = {"stem": f"Synthetic stem for {spec.item_id}."}
    if spec.requires_swapped_stem:
        block["stem_swapped"] = f"Synthetic swapped stem for {spec.item_id}."
    if spec.kind in OPTION_LIST_KINDS:
        block["options"] = [f"synthetic option {index}" for index in range(1, spec.n_options + 1)]
    if spec.kind in VOCABULARY_KINDS:
        block["vocabulary"] = [f"wordnumber{index}" for index in range(1, spec.n_tag_words + 1)]
    if spec.kind == SURVEY_ALLOCATION:
        block["option_payoffs"] = _synthetic_payoffs(spec.n_options, 1)
    return block


SYNTHETIC_ELICITATION = "Synthetic shared closing question, asked once per family."


def synthetic_elicitation_blocks() -> dict[str, str]:
    """Build the shared closing block every family that takes one needs, derived from the registry."""
    return dict.fromkeys(FAMILIES_WITH_SHARED_ELICITATION, SYNTHETIC_ELICITATION)


def synthetic_authored_file(
    path: Path,
    *,
    item_overrides: dict[str, dict[str, Any]] | None = None,
    elicitation_blocks: dict[str, str] | None = None,
) -> Path:
    """Write a synthetic `authored.json` covering every tracked spec, with placeholder text only.

    Derived from `AUTHORED_ITEM_SPECS` rather than hand-written, so it stays at the registry's ids
    and answer shapes and exercises the real loader's real validation.

    `item_overrides` replaces individual items' payloads, keyed by item id. An explicit parameter
    rather than `**kwargs`: keyed by item id, a kwargs channel collides with any real keyword this
    helper grows, which is exactly what happened when `elicitation_blocks` was added.

    `elicitation_blocks` overrides the shared closing text the loader appends per family; pass `{}`
    to write a file that declares none, which is what a sabotage of the shared-elicitation guard
    looks like from the file's side.
    """
    overrides = item_overrides or {}
    items = {
        spec.item_id: overrides.get(spec.item_id, synthetic_authored_block(spec))
        for spec in AUTHORED_ITEM_SPECS
    }
    blocks = synthetic_elicitation_blocks() if elicitation_blocks is None else elicitation_blocks
    path.mkdir(parents=True, exist_ok=True)
    (path / AUTHORED_FILENAME).write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "items": items, ELICITATION_BLOCKS_KEY: blocks},
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def authored_items(tmp_path_factory: pytest.TempPathFactory) -> list[SurveyItem]:
    """The authored battery built from synthetic text, shared read-only across this module."""
    return load_authored_items(synthetic_authored_file(tmp_path_factory.mktemp("authored")))


def likert_item(
    item_id: str,
    *,
    subscale: str = "enjoyment-of-competition",
    reverse_keyed: bool = False,
    wording: str = WORDING_AS_PUBLISHED,
    twin_of: str | None = None,
) -> SurveyItem:
    """Build a synthetic 5-point Likert item, for the composite arithmetic tests."""
    return SurveyItem(
        item_id=item_id,
        family=FAMILY_NEGATIVE_CONTROL,
        instrument="synthetic",
        kind=SURVEY_LIKERT,
        stem=f"Synthetic statement for {item_id}.",
        construct="A synthetic construct.",
        expected_direction="No prediction; this item exists only in a test.",
        subscale=subscale,
        reverse_keyed=reverse_keyed,
        wording=wording,
        twin_of=twin_of,
        options=FIVE_POINT_ANCHORS,
    )


def numeric_item(item_id: str, *, swapped: bool = True) -> SurveyItem:
    """Build a synthetic 0-100 numeric item, optionally with a swapped stem."""
    return SurveyItem(
        item_id=item_id,
        family=FAMILY_SELF_PREDICTION,
        instrument="self-prediction",
        kind=SURVEY_NUMERIC,
        stem=f"Synthetic numeric stem for {item_id}.",
        stem_swapped=f"Synthetic swapped stem for {item_id}." if swapped else None,
        construct="synthetic",
        expected_direction="none",
        subscale=SUBSCALE_OWN_ACTION_RATE,
        numeric_max=100,
        predicts_game="twin-pd",
    )


def record_for(item: SurveyItem, response: int) -> dict[str, Any]:
    """Build one record the way the eval writer does, from a canonical 1-based response."""
    answer = _answer_for(item, response)
    return {"record": "self-report", "sample_index": 0, **survey_record_fields(item, answer)}


def _answer_for(item: SurveyItem, response: int) -> SurveyAnswer:
    """Parse a completion that answers `item` at canonical position `response`, as-authored."""
    letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[response - 1]
    return parse_survey_answer(item, f"FINAL ANSWER: {letter}")


class TestTheAuthoredSpecRegistry:
    def test_every_spec_id_is_unique(self) -> None:
        ids = [spec.item_id for spec in AUTHORED_ITEM_SPECS]
        assert len(ids) == len(set(ids))

    def test_the_registry_covers_the_two_non_optional_families(self) -> None:
        families = {spec.family for spec in AUTHORED_ITEM_SPECS}
        assert FAMILY_NEGATIVE_CONTROL in families
        assert FAMILY_SELF_PREDICTION in families

    def test_every_family_a_published_instrument_claims_is_registered(self) -> None:
        for spec in PUBLISHED_INSTRUMENTS.values():
            assert spec.family in FAMILIES

    def test_the_cut_controls_stay_cut(self) -> None:
        """The animal and list-style items were cut by the 2026-08-21 trim decision.

        The animal choice is socially coded (loyal-versus-independent), a genuine semantic
        transfer path from social-policy RL, so a placebo that can move for real reasons is not a
        placebo. Reintroducing either id has to be a deliberate decision, not a merge accident.
        """
        ids = {spec.item_id for spec in AUTHORED_ITEM_SPECS}
        assert "negative-control-animal" not in ids
        assert "negative-control-list-style" not in ids

    def test_the_core_tier_matches_the_trim_decision(self) -> None:
        """The 2026-08-21 trim, plus the 2026-08-25 headroom adoption.

        The trim left four repaired preference controls and the two format placebos in core. The
        adoption added five items with measured answer entropy on the base model at both 2B and 9B
        and promoted season from breadth on the same measurement, because three of the four
        original nominal controls are answered deterministically at 9B and their flatness
        certifies nothing (docs/scratch/negcontrol-design-2026-08-25/design.md). The saturated
        originals stay, unchanged, for cross-administration comparability.
        """
        core_controls = {
            spec.item_id
            for spec in AUTHORED_ITEM_SPECS
            if spec.family == FAMILY_NEGATIVE_CONTROL and spec.tier == TIER_CORE
        }
        assert core_controls == {
            "negative-control-indentation",
            "negative-control-spelling",
            "negative-control-date-format",
            "negative-control-quote-style",
            "negative-control-table-likert",
            "negative-control-bullet-rate",
            "negative-control-file-tag",
            "negative-control-heading-case",
            "negative-control-locker-number",
            "negative-control-notebook-colour",
            "negative-control-room-name",
            "negative-control-season",
        }
        breadth_controls = {
            spec.item_id
            for spec in AUTHORED_ITEM_SPECS
            if spec.family == FAMILY_NEGATIVE_CONTROL and spec.tier == TIER_BREADTH
        }
        assert breadth_controls == {
            "negative-control-units",
            "negative-control-planet-order",
            "negative-control-boiling-point",
        }

    def test_the_headroom_controls_match_their_measured_shapes(self) -> None:
        """The 2026-08-25 additions are nominal choices with the option counts that were measured.

        A drifted option count would administer a different item than the one whose base-model
        entropy justified adoption, so the tracked shape half of the contract is pinned here the
        way the format placebos' is above.
        """
        by_id = {spec.item_id: spec for spec in AUTHORED_ITEM_SPECS}
        expected_options = {
            "negative-control-file-tag": 2,
            "negative-control-heading-case": 2,
            "negative-control-locker-number": 4,
            "negative-control-notebook-colour": 2,
            "negative-control-room-name": 2,
        }
        for item_id, n_options in expected_options.items():
            spec = by_id[item_id]
            assert spec.kind == SURVEY_CHOICE, item_id
            assert spec.subscale == SUBSCALE_INERT_PREFERENCE, item_id
            assert spec.n_options == n_options, item_id

    def test_the_facts_report_as_their_own_subscale(self) -> None:
        """Factual-integrity items never enter the preference-placebo composite."""
        facts = {
            spec.item_id for spec in AUTHORED_ITEM_SPECS if spec.subscale == SUBSCALE_INERT_FACT
        }
        assert facts == {"negative-control-planet-order", "negative-control-boiling-point"}

    def test_the_format_placebos_match_the_target_formats(self) -> None:
        by_id = {spec.item_id: spec for spec in AUTHORED_ITEM_SPECS}
        likert = by_id["negative-control-table-likert"]
        assert (likert.kind, likert.subscale, likert.n_options) == (
            SURVEY_LIKERT,
            SUBSCALE_INERT_LIKERT,
            5,
        )
        numeric = by_id["negative-control-bullet-rate"]
        assert (numeric.kind, numeric.subscale, numeric.numeric_max) == (
            SURVEY_NUMERIC,
            SUBSCALE_INERT_NUMERIC,
            100,
        )
        assert not numeric.requires_swapped_stem

    def test_self_characterisation_is_demoted_to_breadth(self) -> None:
        tiers = {
            spec.tier for spec in AUTHORED_ITEM_SPECS if spec.family == FAMILY_SELF_CHARACTERISATION
        }
        assert tiers == {TIER_BREADTH}

    def test_every_self_prediction_spec_counterbalances_and_names_a_game(self) -> None:
        predictions = [
            spec for spec in AUTHORED_ITEM_SPECS if spec.family == FAMILY_SELF_PREDICTION
        ]
        assert len(predictions) == 8
        assert all(spec.requires_swapped_stem for spec in predictions)
        assert all(spec.tier == TIER_CORE for spec in predictions)
        games_named = {spec.predicts_game for spec in predictions}
        assert None not in games_named
        assert len(games_named) == len(predictions)
        assert all(spec.subscale == SUBSCALE_OWN_ACTION_RATE for spec in predictions)

    def test_a_spec_with_no_answer_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="no usable answer shape"):
            AuthoredItemSpec(
                item_id="broken",
                family=FAMILY_NEGATIVE_CONTROL,
                instrument="negative-control",
                kind=SURVEY_CHOICE,
                construct="synthetic",
                expected_direction="none",
            )

    def test_a_spec_requiring_a_swap_on_a_lettered_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="only a numeric item counterbalances by rewording"):
            AuthoredItemSpec(
                item_id="broken",
                family=FAMILY_NEGATIVE_CONTROL,
                instrument="negative-control",
                kind=SURVEY_CHOICE,
                construct="synthetic",
                expected_direction="none",
                n_options=2,
                requires_swapped_stem=True,
            )

    def test_a_self_prediction_spec_naming_no_game_raises(self) -> None:
        with pytest.raises(ValueError, match="calibration row"):
            AuthoredItemSpec(
                item_id="broken",
                family=FAMILY_SELF_PREDICTION,
                instrument="self-prediction",
                kind=SURVEY_NUMERIC,
                construct="synthetic",
                expected_direction="none",
                numeric_max=100,
            )

    def test_a_spec_naming_an_unregistered_game_raises(self) -> None:
        with pytest.raises(ValueError, match="not a registered game"):
            AuthoredItemSpec(
                item_id="broken",
                family=FAMILY_SELF_PREDICTION,
                instrument="self-prediction",
                kind=SURVEY_NUMERIC,
                construct="synthetic",
                expected_direction="none",
                numeric_max=100,
                predicts_game="a-game-nobody-registered",
            )

    def test_a_spec_with_no_expectation_raises(self) -> None:
        with pytest.raises(ValueError, match="expected_direction"):
            AuthoredItemSpec(
                item_id="broken",
                family=FAMILY_NEGATIVE_CONTROL,
                instrument="negative-control",
                kind=SURVEY_CHOICE,
                construct="synthetic",
                expected_direction="  ",
                n_options=2,
            )


class TestThePublishedTiering:
    def test_the_slider_core_is_the_six_primary_items(self) -> None:
        spec = PUBLISHED_INSTRUMENTS[INSTRUMENT_SVO_SLIDER]
        core = {
            position
            for position in range(1, spec.n_items + 1)
            if spec.tier_at(position) == TIER_CORE
        }
        assert core == set(range(1, 7))
        assert all(spec.subscale_by_position[position - 1] == "primary" for position in core)

    def test_the_competitiveness_core_is_the_nine_enjoyment_items(self) -> None:
        spec = PUBLISHED_INSTRUMENTS[INSTRUMENT_COMPETITIVENESS_INDEX]
        core = {
            position
            for position in range(1, spec.n_items + 1)
            if spec.tier_at(position) == TIER_CORE
        }
        assert core == set(range(1, 10))
        assert all(
            spec.subscale_by_position[position - 1] == "enjoyment-of-competition"
            for position in core
        )

    def test_the_cooperative_orientation_scale_is_all_breadth(self) -> None:
        spec = PUBLISHED_INSTRUMENTS[INSTRUMENT_COOPERATIVE_ORIENTATION]
        assert all(
            spec.tier_at(position) == TIER_BREADTH for position in range(1, spec.n_items + 1)
        )

    def test_every_published_instrument_declares_a_subscale_per_item(self) -> None:
        for spec in PUBLISHED_INSTRUMENTS.values():
            assert spec.n_items == len(spec.subscale_by_position)
            assert spec.n_items > 0

    def test_every_published_subscale_carries_an_expected_direction(self) -> None:
        for spec in PUBLISHED_INSTRUMENTS.values():
            for subscale in set(spec.subscale_by_position):
                assert spec.expected_direction_by_subscale[subscale].strip()

    def test_the_competitiveness_index_keys_nine_of_its_fourteen_items(self) -> None:
        """The retrieval note marks (R) on four enjoyment items and all five contentiousness items.

        Pinned as a number because a keying list off by one item corrupts a whole scale while every
        composite it produces stays plausible. A secondary plan document said eight; the primary
        transcription says nine (re-verified against the note at integration, 2026-08-21), and this
        is the assertion that keeps the code on the transcription.
        """
        spec = PUBLISHED_INSTRUMENTS[INSTRUMENT_COMPETITIVENESS_INDEX]
        assert spec.n_items == 14
        assert spec.reverse_keyed_positions == frozenset({4, 6, 7, 8, 10, 11, 12, 13, 14})
        enjoyment = {
            position
            for position in spec.reverse_keyed_positions
            if spec.subscale_by_position[position - 1] == "enjoyment-of-competition"
        }
        assert len(enjoyment) == 4
        assert len(spec.reverse_keyed_positions) - len(enjoyment) == 5

    def test_the_narcissism_short_form_is_a_subset_of_the_long_form(self) -> None:
        spec = PUBLISHED_INSTRUMENTS[INSTRUMENT_NARCISSISM]
        short = set(spec.nested_subscales["short-form"])
        assert short == {4, 8, 9, 15, 16, 17}
        assert short <= set(range(1, spec.n_items + 1))
        # Three admiration and three rivalry items, which is what makes both subscales computable
        # from the brief form without administering it separately.
        keyed = [spec.subscale_by_position[position - 1] for position in sorted(short)]
        assert keyed.count("admiration") == 3
        assert keyed.count("rivalry") == 3

    def test_the_narcissism_core_tier_is_exactly_the_short_form(self) -> None:
        spec = PUBLISHED_INSTRUMENTS[INSTRUMENT_NARCISSISM]
        core = {
            position
            for position in range(1, spec.n_items + 1)
            if spec.tier_at(position) == TIER_CORE
        }
        assert core == set(spec.nested_subscales["short-form"])
        assert spec.tier_at(1) == TIER_BREADTH

    def test_a_spec_keying_a_position_it_does_not_have_raises(self) -> None:
        spec = PUBLISHED_INSTRUMENTS[INSTRUMENT_COMPETITIVENESS_INDEX]
        with pytest.raises(ValueError, match="outside its"):
            type(spec)(
                instrument="synthetic",
                family=FAMILY_NEGATIVE_CONTROL,
                kind=SURVEY_LIKERT,
                citation="none",
                scale_points=5,
                subscale_by_position=("only",) * 3,
                expected_direction_by_subscale={"only": "no prediction"},
                construct="synthetic",
                licence_note="synthetic",
                reverse_keyed_positions=frozenset({4}),
            )

    def test_a_spec_missing_an_expected_direction_raises(self) -> None:
        spec = PUBLISHED_INSTRUMENTS[INSTRUMENT_COMPETITIVENESS_INDEX]
        with pytest.raises(ValueError, match="no expected_direction"):
            type(spec)(
                instrument="synthetic",
                family=FAMILY_NEGATIVE_CONTROL,
                kind=SURVEY_LIKERT,
                citation="none",
                scale_points=5,
                subscale_by_position=("described", "undescribed"),
                expected_direction_by_subscale={"described": "no prediction"},
                construct="synthetic",
                licence_note="synthetic",
            )


class TestItemValidation:
    def test_two_answer_shapes_on_one_item_raises(self) -> None:
        with pytest.raises(ValueError, match="Two answer shapes"):
            SurveyItem(
                item_id="broken",
                family=FAMILY_NEGATIVE_CONTROL,
                instrument="synthetic",
                kind=SURVEY_LIKERT,
                stem="Synthetic.",
                construct="synthetic",
                expected_direction="none",
                options=FIVE_POINT_ANCHORS,
                numeric_max=100,
            )

    def test_an_item_with_no_answer_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="carries no"):
            SurveyItem(
                item_id="broken",
                family=FAMILY_NEGATIVE_CONTROL,
                instrument="synthetic",
                kind=SURVEY_CHOICE,
                stem="Synthetic.",
                construct="synthetic",
                expected_direction="none",
            )

    def test_a_reverse_keyed_allocation_item_raises(self) -> None:
        with pytest.raises(ValueError, match="direction to invert"):
            SurveyItem(
                item_id="broken",
                family=FAMILY_NEGATIVE_CONTROL,
                instrument="synthetic",
                kind=SURVEY_ALLOCATION,
                stem="Synthetic.",
                construct="synthetic",
                expected_direction="none",
                reverse_keyed=True,
                option_payoffs=((80, 80), (80, 20)),
            )

    def test_a_swapped_stem_on_a_lettered_item_raises(self) -> None:
        """An option-bearing item counterbalances by reordering; a second stem is a second item."""
        with pytest.raises(ValueError, match="second item"):
            SurveyItem(
                item_id="broken",
                family=FAMILY_NEGATIVE_CONTROL,
                instrument="synthetic",
                kind=SURVEY_CHOICE,
                stem="Synthetic.",
                stem_swapped="Synthetic, reordered.",
                construct="synthetic",
                expected_direction="none",
                options=("one", "two"),
            )

    def test_a_blank_swapped_stem_raises(self) -> None:
        with pytest.raises(ValueError, match="blank swapped stem"):
            SurveyItem(
                item_id="broken",
                family=FAMILY_SELF_PREDICTION,
                instrument="synthetic",
                kind=SURVEY_NUMERIC,
                stem="Synthetic.",
                stem_swapped="   ",
                construct="synthetic",
                expected_direction="none",
                numeric_max=100,
                predicts_game="twin-pd",
            )

    def test_an_item_with_an_empty_expected_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="no expected_direction"):
            SurveyItem(
                item_id="broken",
                family=FAMILY_NEGATIVE_CONTROL,
                instrument="synthetic",
                kind=SURVEY_CHOICE,
                stem="Synthetic.",
                construct="synthetic",
                expected_direction="   ",
                options=("one", "two"),
            )

    def test_a_single_option_item_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot register a preference"):
            SurveyItem(
                item_id="broken",
                family=FAMILY_NEGATIVE_CONTROL,
                instrument="synthetic",
                kind=SURVEY_CHOICE,
                stem="Synthetic.",
                construct="synthetic",
                expected_direction="none",
                options=("only one",),
            )

    def test_a_non_numeric_item_naming_a_game_raises(self) -> None:
        with pytest.raises(ValueError, match="side to put on it"):
            SurveyItem(
                item_id="broken",
                family=FAMILY_SELF_PREDICTION,
                instrument="synthetic",
                kind=SURVEY_CHOICE,
                stem="Synthetic.",
                construct="synthetic",
                expected_direction="none",
                options=("one", "two"),
                predicts_game="twin-pd",
            )

    def test_a_numeric_self_prediction_item_naming_no_game_raises(self) -> None:
        with pytest.raises(ValueError, match="never reach a calibration row"):
            SurveyItem(
                item_id="broken",
                family=FAMILY_SELF_PREDICTION,
                instrument="synthetic",
                kind=SURVEY_NUMERIC,
                stem="Synthetic.",
                construct="synthetic",
                expected_direction="none",
                numeric_max=100,
            )

    def test_an_unregistered_game_raises(self) -> None:
        with pytest.raises(ValueError, match="not a registered game"):
            SurveyItem(
                item_id="broken",
                family=FAMILY_SELF_PREDICTION,
                instrument="synthetic",
                kind=SURVEY_NUMERIC,
                stem="Synthetic.",
                construct="synthetic",
                expected_direction="none",
                numeric_max=100,
                predicts_game="a-game-nobody-registered",
            )

    def test_a_twin_without_the_twin_wording_raises(self) -> None:
        with pytest.raises(ValueError, match="names its published parent"):
            likert_item("broken", twin_of="some-parent")

    def test_the_twin_wording_without_a_parent_raises(self) -> None:
        with pytest.raises(ValueError, match="names its published parent"):
            likert_item("broken", wording=WORDING_NEUTRAL_TWIN)


class TestRenderAndRoundTrip:
    def test_the_as_authored_render_is_what_no_order_produces(self) -> None:
        item = likert_item("synthetic-01")
        assert render_survey_prompt(item) == render_survey_prompt(
            item, option_order=(0, 1, 2, 3, 4)
        )

    def test_the_reversed_render_puts_the_last_anchor_first(self) -> None:
        item = likert_item("synthetic-01")
        prompt = render_survey_prompt(item, option_order=(4, 3, 2, 1, 0))
        assert f"A) {FIVE_POINT_ANCHORS[4]}" in prompt
        assert f"E) {FIVE_POINT_ANCHORS[0]}" in prompt

    def test_both_orders_are_offered_for_every_counterbalanced_item(
        self, authored_items: list[SurveyItem]
    ) -> None:
        for item in authored_items:
            orders = battery_orders(item)
            if item.kind == SURVEY_NUMERIC and item.stem_swapped is None:
                assert orders == ()
                continue
            assert [name for name, _ in orders] == [ORDER_AS_AUTHORED, ORDER_REVERSED]

    def test_every_answer_position_round_trips_under_every_order(
        self, authored_items: list[SurveyItem]
    ) -> None:
        """The presented letter must map back to the canonical index under both orders.

        This is the assertion an order-counterbalance bypass has to break: a render that ignored
        `option_order` would still parse, still record an index, and silently report the reversed
        render's answers as if they were the as-authored render's.
        """
        for item in authored_items:
            if item.kind == SURVEY_NUMERIC:
                continue
            for _, order in battery_orders(item):
                prompt = render_survey_prompt(item, option_order=order)
                for presented, canonical in enumerate(order):
                    letter = "ABCDEFGHI"[presented]
                    answer = parse_survey_answer(
                        item, f"FINAL ANSWER: {letter}", option_order=order
                    )
                    # A vocabulary kind answers with a word in a tag, not a letter, so a lettered
                    # completion is correctly unparseable for it. Their own round trips are pinned
                    # by the tagged test below and by
                    # `test_games_survey_new_families.py::TestTheCheapTalkParse`, which walks every
                    # announced-and-acted combination under both orders.
                    if item.kind in VOCABULARY_KINDS:
                        continue
                    assert answer.canonical_index == canonical
                    assert answer.response == canonical + 1
                    presented_text = prompt.split("\n")[
                        prompt.split("\n").index(f"{letter}) {_option_line(item, canonical)}")
                    ]
                    assert presented_text.startswith(f"{letter}) ")

    def test_a_tagged_item_round_trips_its_vocabulary_under_both_orders(
        self, authored_items: list[SurveyItem]
    ) -> None:
        item = next(item for item in authored_items if item.kind == SURVEY_TAGGED)
        for _, order in battery_orders(item):
            prompt = render_survey_prompt(item, option_order=order)
            first_offered = item.tag_vocabulary[order[0]]
            assert f"one word from this list: {first_offered}" in prompt
            for canonical, word in enumerate(item.tag_vocabulary):
                answer = parse_survey_answer(item, f"<stance>{word}</stance>", option_order=order)
                assert answer.canonical_index == canonical
                assert answer.tag == word

    def test_the_tag_example_is_a_placeholder_never_a_vocabulary_word(
        self, authored_items: list[SurveyItem]
    ) -> None:
        """An example spelling out one real option primes that option (repair R4)."""
        for item in authored_items:
            if item.kind != SURVEY_TAGGED:
                continue
            for _, order in battery_orders(item):
                prompt = render_survey_prompt(item, option_order=order)
                assert f"<stance>{TAG_EXAMPLE_PLACEHOLDER}</stance>" in prompt
                for word in item.tag_vocabulary:
                    assert f"<stance>{word}</stance>" not in prompt

    def test_a_numeric_render_shows_the_example_it_was_given_and_refuses_none(
        self, authored_items: list[SurveyItem]
    ) -> None:
        """Repair R1's documented fallback, armed by the 2026-08-22 L4 smoke: with bare tags and
        no example the 2B wrote the number AS the tag (`<100>`) and the family parsed 9/16, so the
        instruction now shows a worked example that the caller rotates. A render without one is
        the measured-collapsed format and is refused; a fixed default would anchor every answer at
        one value, which is the failure R1 originally guarded against."""
        for item in authored_items:
            if item.kind != SURVEY_NUMERIC:
                continue
            for example in NUMERIC_EXAMPLE_ROTATION:
                prompt = render_survey_prompt(item, numeric_example=example)
                assert f"<keep>{example}</keep>" in prompt
            with pytest.raises(ValueError, match="numeric_example"):
                render_survey_prompt(item)
            with pytest.raises(ValueError, match="caps answers"):
                render_survey_prompt(item, numeric_example=item.numeric_max + 1)

    def test_a_non_numeric_render_refuses_a_numeric_example(self) -> None:
        item = likert_item("synthetic-01")
        with pytest.raises(ValueError, match="numeric example"):
            render_survey_prompt(item, numeric_example=NUMERIC_EXAMPLE_ROTATION[0])

    def test_an_order_that_is_not_a_permutation_raises(self) -> None:
        item = likert_item("synthetic-01")
        with pytest.raises(ValueError, match="not a permutation"):
            render_survey_prompt(item, option_order=(0, 0, 1, 2, 3))

    def test_ordering_a_numeric_item_without_a_swapped_stem_raises(self) -> None:
        item = numeric_item("synthetic-numeric", swapped=False)
        with pytest.raises(ValueError, match="no order to reverse"):
            render_survey_prompt(
                item, option_order=(0, 1), numeric_example=NUMERIC_EXAMPLE_ROTATION[0]
            )

    def test_every_render_carries_the_format_its_parser_keys_on(
        self, authored_items: list[SurveyItem]
    ) -> None:
        for item in authored_items:
            orders = battery_orders(item)
            example = NUMERIC_EXAMPLE_ROTATION[0] if item.kind == SURVEY_NUMERIC else None
            prompt = render_survey_prompt(
                item, option_order=orders[0][1] if orders else None, numeric_example=example
            )
            if item.kind == SURVEY_NUMERIC:
                assert "<keep></keep>" in prompt
            elif item.kind == SURVEY_TAGGED:
                assert "<stance></stance>" in prompt
            elif item.kind == SURVEY_CHEAP_TALK:
                # Two tags rather than one, and the example inside each is the placeholder rather
                # than a menu word: the announcement is written before the action in the SAME
                # completion, so an example naming a real option would prime both halves at once.
                assert f"<{ANNOUNCE_TAG}>{TAG_EXAMPLE_PLACEHOLDER}</{ANNOUNCE_TAG}>" in prompt
                assert f"<{ACT_TAG}>{TAG_EXAMPLE_PLACEHOLDER}</{ACT_TAG}>" in prompt
            else:
                assert "FINAL ANSWER: X" in prompt


def _option_line(item: SurveyItem, canonical: int) -> str:
    """Return the rendered prose of one canonical option, for the round-trip assertion."""
    if item.kind == SURVEY_ALLOCATION:
        mine, theirs = item.option_payoffs[canonical]
        return f"You receive {mine} points; the other party receives {theirs} points."
    return item.options[canonical]


class TestNumericCounterbalance:
    """The swapped-stem machinery (repair R2): every answer canonical, whichever stem was shown."""

    def test_the_swapped_render_uses_the_swapped_stem(self) -> None:
        item = numeric_item("synthetic-numeric")
        example = NUMERIC_EXAMPLE_ROTATION[0]
        as_authored = render_survey_prompt(item, option_order=(0, 1), numeric_example=example)
        swapped = render_survey_prompt(item, option_order=(1, 0), numeric_example=example)
        assert item.stem in as_authored
        assert item.stem_swapped is not None
        assert item.stem_swapped in swapped
        assert as_authored != swapped
        assert render_survey_prompt(item, numeric_example=example) == as_authored

    def test_a_swapped_answer_is_reflected_through_the_maximum(self) -> None:
        """80 under the swapped stem is 100 - 80 = 20 of the as-authored first action."""
        item = numeric_item("synthetic-numeric")
        swapped = parse_survey_answer(item, "<keep>80</keep>", option_order=(1, 0))
        assert swapped.numeric == 20
        as_authored = parse_survey_answer(item, "<keep>80</keep>", option_order=(0, 1))
        assert as_authored.numeric == 80

    def test_a_numeric_answer_carries_no_score(self) -> None:
        """Per item is the numeric family's only aggregate; a pooled subscale mean would average
        different games' predicted rates into a number that is not a construct (repair R6)."""
        item = numeric_item("synthetic-numeric")
        answer = parse_survey_answer(item, "<keep>80</keep>")
        assert answer.parsed
        assert answer.numeric == 80
        assert answer.score is None
        assert per_item_scores([record_for_numeric(item, answer)]) == {}

    def test_a_non_permutation_numeric_order_raises(self) -> None:
        item = numeric_item("synthetic-numeric")
        with pytest.raises(ValueError, match="two action descriptions"):
            parse_survey_answer(item, "<keep>80</keep>", option_order=(0, 1, 2))

    def test_numeric_item_readings_split_by_order_and_carry_denominators(self) -> None:
        """Hand math: as-authored 80; swapped raw 30 canonicalises to 70.

        mean = (80 + 70) / 2 = 75; order gap = 80 - 70 = +10; one unparsed render makes the
        denominators 2 parsed of 3 asked.
        """
        item = numeric_item("synthetic-numeric")
        records = [
            {
                **record_for_numeric(item, parse_survey_answer(item, "<keep>80</keep>")),
                "option_order_name": ORDER_AS_AUTHORED,
            },
            {
                **record_for_numeric(
                    item, parse_survey_answer(item, "<keep>30</keep>", option_order=(1, 0))
                ),
                "option_order_name": ORDER_REVERSED,
            },
            {
                **record_for_numeric(item, parse_survey_answer(item, "no answer")),
                "option_order_name": ORDER_AS_AUTHORED,
            },
        ]
        reading = numeric_item_readings(records)["synthetic-numeric"]
        assert reading.mean == pytest.approx(75.0)
        assert reading.as_authored_mean == pytest.approx(80.0)
        assert reading.swapped_mean == pytest.approx(70.0)
        assert reading.order_gap == pytest.approx(10.0)
        assert (reading.n_parsed, reading.n_asked) == (2, 3)

    def test_an_item_asked_under_one_order_reports_no_gap(self) -> None:
        item = numeric_item("synthetic-numeric", swapped=False)
        records = [
            {
                **record_for_numeric(item, parse_survey_answer(item, "<keep>40</keep>")),
                "option_order_name": "no-options",
            }
        ]
        reading = numeric_item_readings(records)["synthetic-numeric"]
        assert reading.mean == pytest.approx(40.0)
        assert reading.order_gap is None


def record_for_numeric(item: SurveyItem, answer: SurveyAnswer) -> dict[str, Any]:
    """Build one numeric record the way the eval writer does."""
    return {"record": "self-report", "sample_index": 0, **survey_record_fields(item, answer)}


class TestTheSharedElicitationBlock:
    """The own-action-rate repair: one closing question per family, appended by the loader.

    The failure this machinery exists to prevent is not a crash. Administered with the wording it
    had before, the eight own-action-rate items ran the 2B's thinking block into the token budget on
    240 of 256 renders and produced no visible answer at all, and a parse rate of 0.05 reads as a
    fact about the model rather than as an elicitation that never closed. So the tests below check
    the two things a reader of a later run cannot recover from a trace: that every item in the family
    really got the block, under both counterbalanced framings, and that no other family got one.
    """

    def _family_items(self, authored_items: list[SurveyItem]) -> list[SurveyItem]:
        return [item for item in authored_items if item.family in FAMILIES_WITH_SHARED_ELICITATION]

    def test_the_family_is_not_empty_so_the_checks_below_have_a_denominator(
        self, authored_items: list[SurveyItem]
    ) -> None:
        """A registry edit that emptied the family would make every check here vacuously green."""
        assert FAMILIES_WITH_SHARED_ELICITATION
        assert len(self._family_items(authored_items)) >= len(FAMILIES_WITH_SHARED_ELICITATION)

    def test_every_item_closes_with_its_family_block_in_both_framings(
        self, authored_items: list[SurveyItem]
    ) -> None:
        for item in self._family_items(authored_items):
            assert item.stem.endswith(SYNTHETIC_ELICITATION), item.item_id
            assert item.stem_swapped is not None, item.item_id
            assert item.stem_swapped.endswith(SYNTHETIC_ELICITATION), item.item_id

    def test_both_counterbalanced_renders_carry_the_block_and_round_trip(
        self, authored_items: list[SurveyItem]
    ) -> None:
        """The block must reach both framings, and a swapped answer must still canonicalise.

        Appending to only one framing is the quiet version of this bug: the item renders, the parse
        still reflects the swapped answer through `numeric_max - x`, and the two framings measure
        different questions while every number stays plausible.
        """
        for item in self._family_items(authored_items):
            orders = battery_orders(item)
            assert [name for name, _ in orders] == [ORDER_AS_AUTHORED, ORDER_REVERSED], item.item_id
            rendered: list[str] = []
            for _, option_order in orders:
                prompt = render_survey_prompt(
                    item, option_order=option_order, numeric_example=NUMERIC_EXAMPLE_ROTATION[0]
                )
                assert prompt.count(SYNTHETIC_ELICITATION) == 1, item.item_id
                rendered.append(prompt)
            assert rendered[0] != rendered[1], item.item_id
            # 80 asked of the swapped framing is 100 - 80 of the as-authored first action, so the
            # two orders' answers land on one scale whichever stem the render showed.
            assert parse_survey_answer(item, "<keep>80</keep>", option_order=(0, 1)).numeric == 80
            assert parse_survey_answer(item, "<keep>80</keep>", option_order=(1, 0)).numeric == 20

    def test_no_other_family_gains_a_closing_block(self, authored_items: list[SurveyItem]) -> None:
        """Pinned regression: every item outside the listed families renders exactly as before.

        Compared against the synthetic file's own text rather than against a recorded string, so the
        pin follows the fixture and cannot drift into agreeing with whatever the loader now does.
        """
        specs = {spec.item_id: spec for spec in AUTHORED_ITEM_SPECS}
        outside = [
            item for item in authored_items if item.family not in FAMILIES_WITH_SHARED_ELICITATION
        ]
        assert outside
        for item in outside:
            expected = synthetic_authored_block(specs[item.item_id])
            assert item.stem == expected["stem"], item.item_id
            assert item.stem_swapped == expected.get("stem_swapped"), item.item_id
            assert SYNTHETIC_ELICITATION not in item.stem, item.item_id

    def test_a_file_that_declares_no_block_is_refused(self, tmp_path: Path) -> None:
        """The sabotage this guard exists for: the repair deleted from the data file."""
        with pytest.raises(ValueError, match="no usable elicitation_blocks"):
            load_authored_items(synthetic_authored_file(tmp_path, elicitation_blocks={}))

    def test_a_blank_block_is_refused_like_a_missing_one(self, tmp_path: Path) -> None:
        blanked = dict.fromkeys(FAMILIES_WITH_SHARED_ELICITATION, "   ")
        with pytest.raises(ValueError, match="no usable elicitation_blocks"):
            load_authored_items(synthetic_authored_file(tmp_path, elicitation_blocks=blanked))

    def test_a_block_for_a_family_that_takes_none_is_refused(self, tmp_path: Path) -> None:
        """A block nothing appends is a repair that looks applied and is not."""
        stray = {**synthetic_elicitation_blocks(), FAMILY_NEGATIVE_CONTROL: "Stray closing text."}
        with pytest.raises(ValueError, match="which no family reads"):
            load_authored_items(synthetic_authored_file(tmp_path, elicitation_blocks=stray))

    def test_an_item_already_carrying_the_block_is_refused(self, tmp_path: Path) -> None:
        """Half-landed migration: the block pasted into an item the loader also appends it to."""
        spec = next(
            spec for spec in AUTHORED_ITEM_SPECS if spec.family in FAMILIES_WITH_SHARED_ELICITATION
        )
        block = synthetic_authored_block(spec)
        block["stem"] = f"{block['stem']}\n\n{SYNTHETIC_ELICITATION}"
        with pytest.raises(ValueError, match="already carries its family's shared elicitation"):
            load_authored_items(
                synthetic_authored_file(tmp_path, item_overrides={spec.item_id: block})
            )

    def test_the_previous_schema_version_is_refused(self, tmp_path: Path) -> None:
        """The version is what makes this repair's forward-incompatibility loud instead of silent.

        Moving a family's closing question out of its items and into `ELICITATION_BLOCKS_KEY` means an
        older loader, which does not know the key, ignores it and administers those items as a
        situation description with no question -- loading all 338 and reporting success. Reproduced by
        installing the new file under the previous loader before this pin existed. Several checkouts on
        this machine share one gitignored data directory, so refusing the older declared version is
        what stops a stale checkout from quietly measuring the censored thing.
        """
        previous = SCHEMA_VERSION - 1
        assert previous >= 1, "the bump this pins has been reverted; the coupling is now unguarded"
        path = synthetic_authored_file(tmp_path)
        payload = json.loads((path / AUTHORED_FILENAME).read_text(encoding="utf-8"))
        payload["schema_version"] = previous
        (path / AUTHORED_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=f"declares schema_version {previous}"):
            load_authored_items(path)

    def test_a_non_object_blocks_key_is_refused(self, tmp_path: Path) -> None:
        path = synthetic_authored_file(tmp_path)
        payload = json.loads((path / AUTHORED_FILENAME).read_text(encoding="utf-8"))
        payload[ELICITATION_BLOCKS_KEY] = "one string for the whole battery"
        (path / AUTHORED_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(TypeError, match="not an object mapping family name"):
            load_authored_items(path)


class TestTheStemDigest:
    """What the item asked, on every record, because the words themselves are gitignored."""

    def test_the_digest_is_on_every_record_the_writer_builds(self) -> None:
        item = numeric_item("synthetic-numeric")
        record = record_for_numeric(item, parse_survey_answer(item, "<keep>40</keep>"))
        assert record["stem_digest"] == item.stem_digest
        assert "stem_digest" in SURVEY_RECORD_FIELDS

    def test_a_reworded_item_gets_a_different_digest_under_the_same_id(self) -> None:
        """The whole point: two runs of one item id are distinguishable when the words changed."""
        before = numeric_item("synthetic-numeric")
        after = SurveyItem(
            item_id=before.item_id,
            family=before.family,
            instrument=before.instrument,
            kind=before.kind,
            stem=f"{before.stem} Now with a closing question appended.",
            stem_swapped=before.stem_swapped,
            construct=before.construct,
            expected_direction=before.expected_direction,
            subscale=before.subscale,
            numeric_max=before.numeric_max,
            predicts_game=before.predicts_game,
        )
        assert before.stem_digest != after.stem_digest

    def test_the_swapped_framing_is_part_of_the_digest(self) -> None:
        """A swapped stem edited alone still changes the digest; it is half of what was asked."""
        before = numeric_item("synthetic-numeric")
        after = SurveyItem(
            item_id=before.item_id,
            family=before.family,
            instrument=before.instrument,
            kind=before.kind,
            stem=before.stem,
            stem_swapped=f"{before.stem_swapped} Reworded.",
            construct=before.construct,
            expected_direction=before.expected_direction,
            subscale=before.subscale,
            numeric_max=before.numeric_max,
            predicts_game=before.predicts_game,
        )
        assert before.stem_digest != after.stem_digest

    def test_identical_words_under_different_ids_share_a_digest(self) -> None:
        """The digest describes the question, not the item, which is what makes it comparable."""
        first = numeric_item("synthetic-numeric")
        second = SurveyItem(
            item_id="a-different-id",
            family=first.family,
            instrument=first.instrument,
            kind=first.kind,
            stem=first.stem,
            stem_swapped=first.stem_swapped,
            construct=first.construct,
            expected_direction=first.expected_direction,
            subscale=first.subscale,
            numeric_max=first.numeric_max,
            predicts_game=first.predicts_game,
        )
        assert first.stem_digest == second.stem_digest


class TestScoringHandMath:
    def test_a_positively_keyed_answer_scores_its_scale_point(self) -> None:
        item = likert_item("synthetic-01")
        assert parse_survey_answer(item, "FINAL ANSWER: D").score == pytest.approx(4.0)

    def test_a_reverse_keyed_answer_is_reflected_through_the_scale(self) -> None:
        """5 + 1 - 4 = 2. Worked out on paper, because both values are plausible means."""
        item = likert_item("synthetic-01", reverse_keyed=True)
        answer = parse_survey_answer(item, "FINAL ANSWER: D")
        assert answer.response == 4
        assert answer.score == pytest.approx(2.0)

    def test_reflection_fixes_the_scale_midpoint(self) -> None:
        plain = likert_item("plain")
        reversed_key = likert_item("reversed", reverse_keyed=True)
        assert parse_survey_answer(plain, "FINAL ANSWER: C").score == pytest.approx(3.0)
        assert parse_survey_answer(reversed_key, "FINAL ANSWER: C").score == pytest.approx(3.0)

    def test_an_allocation_answer_scores_the_other_partys_payoff(self) -> None:
        item = SurveyItem(
            item_id="allocation-01",
            family=FAMILY_NEGATIVE_CONTROL,
            instrument="synthetic",
            kind=SURVEY_ALLOCATION,
            stem="Synthetic allocation.",
            construct="synthetic",
            expected_direction="none",
            option_payoffs=((80, 80), (80, 50), (80, 20)),
        )
        answer = parse_survey_answer(item, "FINAL ANSWER: B")
        assert (answer.payoff_self, answer.payoff_other) == (80, 50)
        assert answer.score == pytest.approx(50.0)

    def test_a_nominal_choice_answer_carries_no_score(
        self, authored_items: list[SurveyItem]
    ) -> None:
        """A mean over arbitrary option numbering is not a quantity, so none is offered."""
        item = next(item for item in authored_items if item.kind == SURVEY_CHOICE)
        answer = parse_survey_answer(item, "FINAL ANSWER: B")
        assert answer.parsed
        assert answer.canonical_index == 1
        assert answer.score is None

    def test_a_subscale_composite_is_the_mean_over_items_not_over_renders(self) -> None:
        """Two renders of a 5 and one render of a 1 must average 3, not 3.67.

        Item-first: item A answered 5 twice is one observation of 5, so (5 + 1) / 2 = 3. Pooling the
        three renders would give (5 + 5 + 1) / 3 = 3.67 and weight the twice-asked item double.
        """
        item_a = likert_item("synthetic-01")
        item_b = likert_item("synthetic-02")
        records = [record_for(item_a, 5), record_for(item_a, 5), record_for(item_b, 1)]
        assert per_item_scores(records) == {"synthetic-01": 5.0, "synthetic-02": 1.0}
        composites = subscale_composites(records)
        assert composites[("synthetic", "enjoyment-of-competition")] == pytest.approx(3.0)

    def test_an_instrument_composite_spans_its_subscales(self) -> None:
        records = [
            record_for(likert_item("synthetic-01", subscale="one"), 5),
            record_for(likert_item("synthetic-02", subscale="two"), 1),
        ]
        assert instrument_composites(records)["synthetic"] == pytest.approx(3.0)

    def test_a_neutral_twin_never_pools_into_the_default_composites(self) -> None:
        """A twin is our adaptation, not the validated instrument (analysis-spec item R10c).

        The published item answers 5 and its twin answers 1: the default composite must read 5.0,
        not the pooled 3.0, and the twin's number belongs to `wording_gap` alone.
        """
        parent = likert_item("synthetic-01")
        twin = likert_item(
            "synthetic-01-neutral01", wording=WORDING_NEUTRAL_TWIN, twin_of=parent.item_id
        )
        records = [record_for(parent, 5), record_for(twin, 1)]
        assert subscale_composites(records)[
            ("synthetic", "enjoyment-of-competition")
        ] == pytest.approx(5.0)
        assert instrument_composites(records)["synthetic"] == pytest.approx(5.0)
        pooled = subscale_composites(records, wording=None)
        assert pooled[("synthetic", "enjoyment-of-competition")] == pytest.approx(3.0)

    def test_an_unparsed_render_does_not_enter_a_composite(self) -> None:
        item = likert_item("synthetic-01")
        unparsed = {**record_for(item, 5), "score": None, "response": None, "parsed": False}
        records = [record_for(item, 3), unparsed]
        assert per_item_scores(records) == {"synthetic-01": 3.0}


class TestModalChoicesAndDistributions:
    def test_the_modal_choice_is_the_most_chosen_canonical_option(
        self, authored_items: list[SurveyItem]
    ) -> None:
        item = next(item for item in authored_items if item.kind == SURVEY_CHOICE)
        records = [record_for(item, 2), record_for(item, 2), record_for(item, 1)]
        assert modal_choices(records) == {item.item_id: 1}

    def test_a_tie_resolves_to_the_lower_index_so_it_cannot_invent_a_shift(
        self, authored_items: list[SurveyItem]
    ) -> None:
        item = next(item for item in authored_items if item.kind == SURVEY_CHOICE)
        records = [record_for(item, 1), record_for(item, 2)]
        assert modal_choices(records) == {item.item_id: 0}

    def test_likert_items_are_not_given_a_modal_choice(self) -> None:
        assert modal_choices([record_for(likert_item("synthetic-01"), 3)]) == {}

    def test_the_distribution_normalises_over_parsed_renders(
        self, authored_items: list[SurveyItem]
    ) -> None:
        """Three parsed renders at 2, 2, 1 give {1: 2/3, 0: 1/3}; the unparsed one is not a vote."""
        item = next(item for item in authored_items if item.kind == SURVEY_CHOICE)
        unparsed = {**record_for(item, 1), "canonical_index": None, "parsed": False}
        records = [record_for(item, 2), record_for(item, 2), record_for(item, 1), unparsed]
        distribution = choice_response_distributions(records)[item.item_id]
        assert distribution[1] == pytest.approx(2 / 3)
        assert distribution[0] == pytest.approx(1 / 3)

    def test_total_variation_distance_hand_math(self) -> None:
        """Disjoint distributions are 1 apart; a 0.25 mass move is 0.25; identity is 0."""
        assert total_variation_distance({0: 1.0}, {1: 1.0}) == pytest.approx(1.0)
        assert total_variation_distance({0: 0.5, 1: 0.5}, {0: 0.75, 1: 0.25}) == pytest.approx(0.25)
        assert total_variation_distance({0: 0.5, 1: 0.5}, {0: 0.5, 1: 0.5}) == pytest.approx(0.0)

    def test_entropy_reads_headroom_in_bits(self, authored_items: list[SurveyItem]) -> None:
        """A 50/50 item carries 1 bit of headroom; an always-same-answer item carries none."""
        item = next(item for item in authored_items if item.kind == SURVEY_CHOICE)
        split = [record_for(item, 1), record_for(item, 2)]
        assert choice_response_entropy(split)[item.item_id] == pytest.approx(1.0)
        fixed = [record_for(item, 1), record_for(item, 1)]
        assert choice_response_entropy(fixed)[item.item_id] == pytest.approx(0.0)


class TestAcquiescence:
    def test_an_unbiased_responder_scores_zero(self) -> None:
        """Agreeing with the positively-keyed item and disagreeing with the reverse-keyed one.

        Above-midpoint rate is 1.0 on the positive side and 0.0 on the reverse side, so
        1.0 + 0.0 - 1 = 0. That is the no-response-style reading whatever the trait level.
        """
        positive = likert_item("synthetic-01")
        reverse = likert_item("synthetic-02", reverse_keyed=True)
        records = [record_for(positive, 5), record_for(reverse, 1)]
        reading = acquiescence_index(records)[("synthetic", "enjoyment-of-competition")]
        assert reading.index == pytest.approx(0.0)
        assert reading.reason is None
        assert (reading.n_positive_keyed_items, reading.n_reverse_keyed_items) == (1, 1)

    def test_pure_yea_saying_scores_one(self) -> None:
        """Agreeing with a statement and its negation: 1.0 + 1.0 - 1 = 1."""
        positive = likert_item("synthetic-01")
        reverse = likert_item("synthetic-02", reverse_keyed=True)
        records = [record_for(positive, 5), record_for(reverse, 5)]
        reading = acquiescence_index(records)[("synthetic", "enjoyment-of-competition")]
        assert reading.index == pytest.approx(1.0)

    def test_pure_nay_saying_scores_minus_one(self) -> None:
        positive = likert_item("synthetic-01")
        reverse = likert_item("synthetic-02", reverse_keyed=True)
        records = [record_for(positive, 1), record_for(reverse, 1)]
        reading = acquiescence_index(records)[("synthetic", "enjoyment-of-competition")]
        assert reading.index == pytest.approx(-1.0)

    def test_the_index_ignores_reflection_by_reading_raw_responses(self) -> None:
        """The reverse-keyed 5 reflects to a score of 1; the index must still see the raw 5.

        Reflecting first is exactly what hides acquiescence, because reflection is the operation
        that makes a yea-sayer's two answers agree.
        """
        reverse = likert_item("synthetic-02", reverse_keyed=True)
        record = record_for(reverse, 5)
        assert record["response"] == 5
        assert record["score"] == pytest.approx(1.0)

    def test_a_subscale_with_no_reverse_keyed_item_reports_a_reason_not_a_number(self) -> None:
        records = [record_for(likert_item("synthetic-01"), 4)]
        reading = acquiescence_index(records)[("synthetic", "enjoyment-of-competition")]
        assert reading.index is None
        assert reading.reason == ACQUIESCENCE_NO_REVERSE_KEYED
        assert reading.n_reverse_keyed_items == 0

    def test_an_all_reverse_keyed_subscale_reports_a_reason_not_a_number(self) -> None:
        """The real competitiveness index's contentiousness subscale is all five reverse-keyed."""
        records = [record_for(likert_item("synthetic-10", reverse_keyed=True), 4)]
        reading = acquiescence_index(records)[("synthetic", "enjoyment-of-competition")]
        assert reading.index is None
        assert reading.reason == ACQUIESCENCE_NO_POSITIVE_KEYED

    def test_a_six_point_scale_puts_the_midpoint_between_anchors(self) -> None:
        """On 1-6 the midpoint is 3.5, so a 4 is agreement and a 3 is not."""
        six_point = SurveyItem(
            item_id="six-point-01",
            family=FAMILY_NEGATIVE_CONTROL,
            instrument="synthetic-six",
            kind=SURVEY_LIKERT,
            stem="Synthetic six-point statement.",
            construct="synthetic",
            expected_direction="none",
            subscale="only",
            options=("1", "2", "3", "4", "5", "6"),
        )
        reverse = SurveyItem(
            item_id="six-point-02",
            family=FAMILY_NEGATIVE_CONTROL,
            instrument="synthetic-six",
            kind=SURVEY_LIKERT,
            stem="Synthetic six-point statement.",
            construct="synthetic",
            expected_direction="none",
            subscale="only",
            reverse_keyed=True,
            options=("1", "2", "3", "4", "5", "6"),
        )
        agreeing = acquiescence_index([record_for(six_point, 4), record_for(reverse, 4)])
        assert agreeing[("synthetic-six", "only")].index == pytest.approx(1.0)
        neutral = acquiescence_index([record_for(six_point, 3), record_for(reverse, 3)])
        assert neutral[("synthetic-six", "only")].index == pytest.approx(-1.0)


class TestWordingGap:
    def test_the_gap_is_published_minus_neutral_over_twinned_items_only(self) -> None:
        """The untwinned item must not enter either side, or the gap measures item selection."""
        parent = likert_item("synthetic-01")
        twin = likert_item(
            "synthetic-01-neutral01", wording=WORDING_NEUTRAL_TWIN, twin_of=parent.item_id
        )
        untwinned = likert_item("synthetic-02")
        records = [record_for(parent, 5), record_for(twin, 3), record_for(untwinned, 1)]
        gap = wording_gap(records)[("synthetic", "enjoyment-of-competition")]
        assert gap.as_published == pytest.approx(5.0)
        assert gap.neutral_twin == pytest.approx(3.0)
        assert gap.gap == pytest.approx(2.0)
        assert gap.n_twinned_items == 1

    def test_a_battery_with_no_twins_reports_no_gap_rather_than_zero(self) -> None:
        records = [record_for(likert_item("synthetic-01"), 5)]
        assert wording_gap(records) == {}

    def test_a_twin_pointing_at_a_missing_parent_raises(self) -> None:
        parent = likert_item("synthetic-01")
        orphan = likert_item(
            "synthetic-09-neutral01", wording=WORDING_NEUTRAL_TWIN, twin_of="absent"
        )
        with pytest.raises(ValueError, match="not in this battery"):
            assert_every_twin_pairs([parent, orphan])

    def test_a_twin_disagreeing_with_its_parent_on_keying_raises(self) -> None:
        parent = likert_item("synthetic-01", reverse_keyed=True)
        twin = likert_item(
            "synthetic-01-neutral01", wording=WORDING_NEUTRAL_TWIN, twin_of=parent.item_id
        )
        with pytest.raises(ValueError, match="disagrees with its parent"):
            assert_every_twin_pairs([parent, twin])

    def test_a_twin_of_a_twin_raises(self) -> None:
        parent = likert_item("synthetic-01")
        twin = likert_item(
            "synthetic-01-neutral01", wording=WORDING_NEUTRAL_TWIN, twin_of=parent.item_id
        )
        grandchild = likert_item(
            "synthetic-01-neutral02", wording=WORDING_NEUTRAL_TWIN, twin_of=twin.item_id
        )
        with pytest.raises(ValueError, match="has no published side"):
            assert_every_twin_pairs([parent, twin, grandchild])


class TestReverseKeySiblings:
    def test_an_instrument_with_one_mixed_subscale_passes(self) -> None:
        """The real competitiveness index's shape: one mixed subscale, one all-reverse-keyed."""
        assert_every_reverse_key_has_a_sibling(
            [
                likert_item("synthetic-01", subscale="enjoyment"),
                likert_item("synthetic-04", subscale="enjoyment", reverse_keyed=True),
                likert_item("synthetic-10", subscale="contentiousness", reverse_keyed=True),
                likert_item("synthetic-11", subscale="contentiousness", reverse_keyed=True),
            ]
        )

    def test_an_instrument_whose_every_subscale_is_single_keyed_raises(self) -> None:
        with pytest.raises(ValueError, match="no subscale containing both keyings"):
            assert_every_reverse_key_has_a_sibling(
                [
                    likert_item("synthetic-10", subscale="contentiousness", reverse_keyed=True),
                    likert_item("synthetic-11", subscale="contentiousness", reverse_keyed=True),
                ]
            )

    def test_an_instrument_with_no_reverse_keyed_item_at_all_passes(self) -> None:
        assert_every_reverse_key_has_a_sibling([likert_item("synthetic-01")])


class TestTripleDominanceOrientations:
    def test_a_separating_triple_classifies_one_option_per_orientation(self) -> None:
        """Synthetic digits, worked by hand: A 93/23, B 97/41, C 88/88.

        Joint sums 116 / 138 / 176 so C is prosocial; own payoffs 93 / 97 / 88 so B is
        individualistic; gaps 70 / 56 / 0 so A is competitive.
        """
        assert triple_dominance_orientations([(93, 23), (97, 41), (88, 88)]) == (
            ORIENTATION_COMPETITIVE,
            ORIENTATION_INDIVIDUALISTIC,
            ORIENTATION_PROSOCIAL,
        )

    def test_a_triple_whose_options_do_not_separate_raises(self) -> None:
        with pytest.raises(ValueError, match="do not separate"):
            triple_dominance_orientations([(500, 500), (400, 100), (300, 100)])

    def test_the_wrong_number_of_options_raises(self) -> None:
        with pytest.raises(ValueError, match="needs 3 options"):
            triple_dominance_orientations([(500, 500), (400, 100)])

    def test_orientation_counts_read_the_record_field(self) -> None:
        records = [
            {"orientation": ORIENTATION_PROSOCIAL},
            {"orientation": ORIENTATION_PROSOCIAL},
            {"orientation": ORIENTATION_COMPETITIVE},
            {"orientation": None},
        ]
        assert orientation_counts(records) == {ORIENTATION_COMPETITIVE: 1, ORIENTATION_PROSOCIAL: 2}


class TestSvoAngle:
    def _slider_records(self, allocations: list[tuple[int, int]]) -> list[dict[str, Any]]:
        """Build one slider record per allocation, all on the primary subscale."""
        return [
            {
                "item_id": f"{INSTRUMENT_SVO_SLIDER}-{index:02d}",
                "instrument": INSTRUMENT_SVO_SLIDER,
                "subscale": "primary",
                "payoff_self": mine,
                "payoff_other": theirs,
            }
            for index, (mine, theirs) in enumerate(allocations, start=1)
        ]

    def test_an_equal_split_responder_scores_forty_five_degrees(self) -> None:
        """Means 80/80 give arctan((80-50)/(80-50)) = arctan(1) = 45 degrees."""
        records = self._slider_records([(80, 80), (80, 80)])
        assert svo_angle(records) == pytest.approx(45.0)

    def test_a_pure_own_gain_responder_scores_zero_degrees(self) -> None:
        """Means 80/50 give arctan(0 / 30) = 0 degrees, the individualistic axis."""
        records = self._slider_records([(80, 50)])
        assert svo_angle(records) == pytest.approx(0.0)

    def test_a_competitive_responder_scores_a_negative_angle(self) -> None:
        """Means 80/20 give arctan(-30 / 30) = -45 degrees."""
        records = self._slider_records([(80, 20)])
        assert svo_angle(records) == pytest.approx(-45.0)

    def test_the_angle_reduces_per_item_before_averaging(self) -> None:
        """One item asked twice at 80/80 and another once at 80/20 average to 80/50, i.e. 0.

        Pooling the three renders would give a mean other-allocation of (80+80+20)/3 = 60 and an
        angle of about 18 degrees instead of 0.
        """
        records = [
            *self._slider_records([(80, 80)]),
            *self._slider_records([(80, 80)]),
            {
                "item_id": f"{INSTRUMENT_SVO_SLIDER}-02",
                "instrument": INSTRUMENT_SVO_SLIDER,
                "subscale": "primary",
                "payoff_self": 80,
                "payoff_other": 20,
            },
        ]
        assert svo_angle(records) == pytest.approx(0.0)

    def test_nothing_parsed_returns_none_rather_than_an_angle(self) -> None:
        assert svo_angle([]) is None

    def test_a_mean_self_allocation_at_the_origin_returns_none(self) -> None:
        """arctan of anything over zero is not a meaningful orientation, so it is not reported."""
        assert svo_angle(self._slider_records([(50, 80)])) is None

    def test_the_secondary_items_are_not_pooled_into_the_primary_angle(self) -> None:
        records = [
            *self._slider_records([(80, 80)]),
            {
                "item_id": f"{INSTRUMENT_SVO_SLIDER}-07",
                "instrument": INSTRUMENT_SVO_SLIDER,
                "subscale": "secondary",
                "payoff_self": 80,
                "payoff_other": 20,
            },
        ]
        assert svo_angle(records) == pytest.approx(45.0)

    def test_the_mean_completion_angle_differs_from_the_aggregate_angle(self) -> None:
        """The two definitions must not be conflated, and this fixture tells them apart.

        Renders (80,80) and (80,50) of one item: per-completion angles 45 and 0 average 22.5; the
        aggregate mean allocation (80, 65) gives arctan(15/30) = 26.57. Both are honest, neither is
        the other (analysis-spec item R10a).
        """
        records = [
            *self._slider_records([(80, 80)]),
            {
                "item_id": f"{INSTRUMENT_SVO_SLIDER}-01",
                "instrument": INSTRUMENT_SVO_SLIDER,
                "subscale": "primary",
                "payoff_self": 80,
                "payoff_other": 50,
            },
        ]
        assert svo_mean_completion_angle(records) == pytest.approx(22.5)
        assert svo_angle(records) == pytest.approx(26.565, abs=1e-3)

    def test_the_mean_completion_angle_skips_origin_renders(self) -> None:
        assert svo_mean_completion_angle(self._slider_records([(50, 80)])) is None


class TestParseRatesCarryTheirDenominators:
    def test_the_family_rate_reports_parsed_and_asked(self) -> None:
        item = likert_item("synthetic-01")
        unparsed = {**record_for(item, 3), "parsed": False}
        rates = parse_rate_by_family([record_for(item, 3), unparsed, record_for(item, 4)])
        rate = rates[FAMILY_NEGATIVE_CONTROL]
        assert (rate.n_parsed, rate.n_asked) == (2, 3)
        assert rate.rate == pytest.approx(2 / 3)
        assert rate.cell == "2/3 (0.667)"

    def test_an_instrument_that_asked_nothing_reports_no_rate_rather_than_zero(self) -> None:
        assert parse_rate_by_instrument([]) == {}

    def test_a_family_whose_every_render_failed_reports_zero_over_its_denominator(self) -> None:
        item = likert_item("synthetic-01")
        failed = {**record_for(item, 3), "parsed": False}
        rate = parse_rate_by_family([failed, failed])[FAMILY_NEGATIVE_CONTROL]
        assert rate.cell == "0/2 (0.000)"


class TestCalibration:
    def test_the_gap_is_predicted_minus_measured_on_one_scale(
        self, authored_items: list[SurveyItem]
    ) -> None:
        """A predicted 80 out of 100 against a measured 0.5 is a gap of +0.3."""
        item = next(
            item
            for item in authored_items
            if item.family == FAMILY_SELF_PREDICTION and item.predicts_game == "twin-pd"
        )
        answer = parse_survey_answer(item, "<keep>80</keep>")
        survey_records = [{"record": "self-report", **survey_record_fields(item, answer)}]
        behaviour = [
            {"game_id": "twin-pd", "coop_fraction": 1.0},
            {"game_id": "twin-pd", "coop_fraction": 0.0},
        ]
        calibration = calibration_gaps(survey_records, behaviour)["twin-pd"]
        assert calibration.predicted == pytest.approx(0.8)
        assert calibration.measured == pytest.approx(0.5)
        assert calibration.gap == pytest.approx(0.3)
        assert (calibration.n_predictions, calibration.n_measured_records) == (1, 2)

    def test_a_game_with_no_behaviour_records_keeps_its_prediction(
        self, authored_items: list[SurveyItem]
    ) -> None:
        item = next(
            item
            for item in authored_items
            if item.family == FAMILY_SELF_PREDICTION and item.predicts_game == "twin-pd"
        )
        answer = parse_survey_answer(item, "<keep>80</keep>")
        calibration = calibration_gaps([survey_record_fields(item, answer)], [])["twin-pd"]
        assert calibration.predicted == pytest.approx(0.8)
        assert calibration.measured is None
        assert calibration.gap is None
        assert calibration.n_measured_records == 0

    def test_an_unparsed_prediction_still_opens_its_row_with_a_zero_denominator(
        self, authored_items: list[SurveyItem]
    ) -> None:
        """A game whose predictions all failed to parse must not silently vanish from the table."""
        item = next(
            item
            for item in authored_items
            if item.family == FAMILY_SELF_PREDICTION and item.predicts_game == "twin-pd"
        )
        answer = parse_survey_answer(item, "I would rather not say.")
        calibration = calibration_gaps(
            [survey_record_fields(item, answer)], [{"game_id": "twin-pd", "coop_fraction": 0.5}]
        )["twin-pd"]
        assert calibration.predicted is None
        assert calibration.n_predictions == 0
        assert calibration.measured == pytest.approx(0.5)


class TestTheRecordContract:
    def test_every_declared_field_is_written(self) -> None:
        item = likert_item("synthetic-01")
        fields = survey_record_fields(item, parse_survey_answer(item, "FINAL ANSWER: C"))
        assert set(fields) == set(SURVEY_RECORD_FIELDS)

    def test_an_unparsed_answer_writes_the_same_keys_with_null_values(self) -> None:
        """A shorter record on a parse failure would make every reduction key-dependent."""
        item = likert_item("synthetic-01")
        fields = survey_record_fields(item, parse_survey_answer(item, "no answer here"))
        assert set(fields) == set(SURVEY_RECORD_FIELDS)
        assert fields["parsed"] is False
        assert fields["response"] is None
        assert fields["score"] is None

    def test_the_record_carries_the_scale_points_a_later_reanalysis_needs(self) -> None:
        item = likert_item("synthetic-01")
        fields = survey_record_fields(item, parse_survey_answer(item, "FINAL ANSWER: C"))
        assert fields["scale_points"] == 5

    def test_a_triple_dominance_record_carries_its_derived_orientation(self) -> None:
        item = SurveyItem(
            item_id=f"{INSTRUMENT_TRIPLE_DOMINANCE}-01",
            family=FAMILY_NEGATIVE_CONTROL,
            instrument=INSTRUMENT_TRIPLE_DOMINANCE,
            kind=SURVEY_ALLOCATION,
            stem="Synthetic triple.",
            construct="synthetic",
            expected_direction="none",
            option_payoffs=((93, 23), (97, 41), (88, 88)),
        )
        fields = survey_record_fields(item, parse_survey_answer(item, "FINAL ANSWER: C"))
        assert fields["orientation"] == ORIENTATION_PROSOCIAL

    def test_a_non_triple_dominance_record_carries_no_orientation(self) -> None:
        item = likert_item("synthetic-01")
        fields = survey_record_fields(item, parse_survey_answer(item, "FINAL ANSWER: C"))
        assert fields["orientation"] is None


class TestTheAuthoredLoader:
    def test_no_data_dir_loads_nothing(self) -> None:
        assert load_authored_items(None) == []

    def test_a_synthetic_file_loads_every_registered_spec(self, tmp_path: Path) -> None:
        items = load_authored_items(synthetic_authored_file(tmp_path / "survey"))
        assert [item.item_id for item in items] == [spec.item_id for spec in AUTHORED_ITEM_SPECS]
        by_id = {item.item_id: item for item in items}
        for spec in AUTHORED_ITEM_SPECS:
            item = by_id[spec.item_id]
            assert (item.family, item.kind, item.tier, item.subscale) == (
                spec.family,
                spec.kind,
                spec.tier,
                spec.subscale,
            )

    def test_a_missing_file_raises_naming_the_readme(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError, match=r"README\.md"):
            load_authored_items(tmp_path)

    @pytest.mark.parametrize(
        "item_id",
        ["negative-control-indentation", "negative-control-room-name"],
    )
    def test_a_missing_id_raises_rather_than_loading_a_smaller_battery(
        self, tmp_path: Path, item_id: str
    ) -> None:
        """The second id is a 2026-08-25 headroom addition: a data file staged before that landing
        is missing exactly this, and the refusal is the loud failure the integration relies on
        rather than a quietly smaller control family."""
        data_dir = synthetic_authored_file(tmp_path / "survey")
        payload = json.loads((data_dir / AUTHORED_FILENAME).read_text(encoding="utf-8"))
        del payload["items"][item_id]
        (data_dir / AUTHORED_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="disagrees with the tracked authored-item registry"):
            load_authored_items(data_dir)

    def test_an_extra_id_raises_because_it_would_run_with_no_expectation(
        self, tmp_path: Path
    ) -> None:
        data_dir = synthetic_authored_file(tmp_path / "survey")
        payload = json.loads((data_dir / AUTHORED_FILENAME).read_text(encoding="utf-8"))
        payload["items"]["negative-control-added-later"] = {"stem": "Synthetic."}
        (data_dir / AUTHORED_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="disagrees with the tracked authored-item registry"):
            load_authored_items(data_dir)

    def test_a_blank_stem_raises(self, tmp_path: Path) -> None:
        data_dir = synthetic_authored_file(
            tmp_path / "survey",
            item_overrides={
                "negative-control-indentation": {"stem": "   ", "options": ["one", "two"]}
            },
        )
        with pytest.raises(ValueError, match="no usable stem"):
            load_authored_items(data_dir)

    def test_a_missing_swapped_stem_raises_where_the_spec_requires_one(
        self, tmp_path: Path
    ) -> None:
        """Without it the render cannot counterbalance which action is described first (R2)."""
        data_dir = synthetic_authored_file(
            tmp_path / "survey",
            item_overrides={"self-prediction-twin-pd": {"stem": "Synthetic prediction stem."}},
        )
        with pytest.raises(ValueError, match="no swapped stem"):
            load_authored_items(data_dir)

    def test_an_undeclared_swapped_stem_raises(self, tmp_path: Path) -> None:
        data_dir = synthetic_authored_file(
            tmp_path / "survey",
            item_overrides={
                "negative-control-bullet-rate": {
                    "stem": "Synthetic numeric stem.",
                    "stem_swapped": "Synthetic swap the spec never declared.",
                }
            },
        )
        with pytest.raises(ValueError, match="its tracked spec does not declare"):
            load_authored_items(data_dir)

    def test_a_wrong_option_count_raises(self, tmp_path: Path) -> None:
        data_dir = synthetic_authored_file(
            tmp_path / "survey",
            item_overrides={
                "negative-control-date-format": {"stem": "Synthetic.", "options": ["one", "two"]}
            },
        )
        with pytest.raises(ValueError, match="exactly 3 non-blank options"):
            load_authored_items(data_dir)

    def test_a_wrong_vocabulary_size_raises(self, tmp_path: Path) -> None:
        data_dir = synthetic_authored_file(
            tmp_path / "survey",
            item_overrides={
                "self-characterisation-stance-conflict": {
                    "stem": "Synthetic.",
                    "vocabulary": ["only", "two"],
                }
            },
        )
        with pytest.raises(ValueError, match="vocabulary"):
            load_authored_items(data_dir)

    def test_a_wrong_schema_version_raises(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "survey"
        data_dir.mkdir(parents=True)
        (data_dir / AUTHORED_FILENAME).write_text(json.dumps({"schema_version": 99}))
        with pytest.raises(ValueError, match="schema_version"):
            load_authored_items(data_dir)


class TestThePublishedLoader:
    def test_no_data_dir_loads_nothing(self) -> None:
        assert load_published_instruments(None) == []

    def test_a_directory_with_no_item_file_raises_naming_the_readme(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"README\.md"):
            load_published_instruments(tmp_path)

    def test_a_synthetic_instrument_loads_at_its_declared_size(self, tmp_path: Path) -> None:
        data_dir = synthetic_published_file(tmp_path / "survey")
        items = load_published_instruments(data_dir, instruments=[SYNTHETIC_INSTRUMENT])
        spec = PUBLISHED_INSTRUMENTS[SYNTHETIC_INSTRUMENT]
        assert len(items) == spec.n_items
        assert [item.item_id for item in items][:2] == [
            f"{SYNTHETIC_INSTRUMENT}-01",
            f"{SYNTHETIC_INSTRUMENT}-02",
        ]

    def test_the_loaded_items_carry_the_tracked_keying_and_subscales(self, tmp_path: Path) -> None:
        data_dir = synthetic_published_file(tmp_path / "survey")
        items = load_published_instruments(data_dir, instruments=[SYNTHETIC_INSTRUMENT])
        spec = PUBLISHED_INSTRUMENTS[SYNTHETIC_INSTRUMENT]
        keyed = {index + 1 for index, item in enumerate(items) if item.reverse_keyed}
        assert keyed == set(spec.reverse_keyed_positions)
        assert [item.subscale for item in items] == list(spec.subscale_by_position)
        assert all(item.expected_direction.strip() for item in items)

    def test_a_short_instrument_raises_rather_than_loading_a_shorter_scale(
        self, tmp_path: Path
    ) -> None:
        """The keying positions are indexes into the declared order, so a partial file re-keys it."""
        data_dir = synthetic_published_file(
            tmp_path / "survey", items=[{"stem": "Only one statement."}]
        )
        with pytest.raises(ValueError, match="not a shorter instrument"):
            load_published_instruments(data_dir, instruments=[SYNTHETIC_INSTRUMENT])

    def test_a_wrong_length_anchor_ladder_raises(self, tmp_path: Path) -> None:
        data_dir = synthetic_published_file(tmp_path / "survey", anchors=["yes", "no"])
        with pytest.raises(ValueError, match="silently rescales"):
            load_published_instruments(data_dir, instruments=[SYNTHETIC_INSTRUMENT])

    def test_an_all_blank_instrument_raises_rather_than_loading_nothing(
        self, tmp_path: Path
    ) -> None:
        spec = PUBLISHED_INSTRUMENTS[SYNTHETIC_INSTRUMENT]
        data_dir = synthetic_published_file(
            tmp_path / "survey", items=[{"stem": "   "}] * spec.n_items
        )
        with pytest.raises(ValueError, match="schema drift"):
            load_published_instruments(data_dir, instruments=[SYNTHETIC_INSTRUMENT])

    def test_a_missing_requested_instrument_raises(self, tmp_path: Path) -> None:
        data_dir = synthetic_published_file(tmp_path / "survey")
        with pytest.raises(ValueError, match="missing requested instrument"):
            load_published_instruments(
                data_dir, instruments=[SYNTHETIC_INSTRUMENT, INSTRUMENT_NARCISSISM]
            )

    def test_an_unknown_requested_instrument_raises(self, tmp_path: Path) -> None:
        data_dir = synthetic_published_file(tmp_path / "survey")
        with pytest.raises(ValueError, match="unknown published instruments"):
            load_published_instruments(data_dir, instruments=["not-an-instrument"])

    def test_a_wrong_schema_version_raises(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "survey"
        data_dir.mkdir(parents=True)
        (data_dir / "published.json").write_text(json.dumps({"schema_version": 99}))
        with pytest.raises(ValueError, match="schema_version"):
            load_published_instruments(data_dir)

    def test_neutral_twins_load_as_linked_items(self, tmp_path: Path) -> None:
        spec = PUBLISHED_INSTRUMENTS[SYNTHETIC_INSTRUMENT]
        items: list[dict[str, Any]] = [
            {"stem": f"Statement {position}."} for position in range(1, spec.n_items + 1)
        ]
        items[0] = {"stem": "Statement 1.", "neutral_stems": ["Statement 1, reworded."]}
        data_dir = synthetic_published_file(tmp_path / "survey", items=items)
        loaded = load_published_instruments(data_dir, instruments=[SYNTHETIC_INSTRUMENT])
        twin = next(item for item in loaded if item.wording == WORDING_NEUTRAL_TWIN)
        assert twin.item_id == f"{SYNTHETIC_INSTRUMENT}-01-neutral01"
        assert twin.twin_of == f"{SYNTHETIC_INSTRUMENT}-01"
        assert twin.stem == "Statement 1, reworded."
        assert len(loaded) == spec.n_items + 1
        assert_every_twin_pairs(loaded)

    def test_an_allocation_instrument_loads_its_payoffs(self, tmp_path: Path) -> None:
        data_dir = synthetic_published_file(
            tmp_path / "survey", instrument=INSTRUMENT_TRIPLE_DOMINANCE
        )
        items = load_published_instruments(data_dir, instruments=[INSTRUMENT_TRIPLE_DOMINANCE])
        assert len(items) == PUBLISHED_INSTRUMENTS[INSTRUMENT_TRIPLE_DOMINANCE].n_items
        assert all(item.kind == SURVEY_ALLOCATION for item in items)
        assert all(len(item.option_payoffs) == 3 for item in items)

    def test_a_triple_whose_options_do_not_separate_is_refused_at_load(
        self, tmp_path: Path
    ) -> None:
        spec = PUBLISHED_INSTRUMENTS[INSTRUMENT_TRIPLE_DOMINANCE]
        broken = [{"option_payoffs": [[500, 500], [400, 100], [300, 100]]}] * spec.n_items
        data_dir = synthetic_published_file(
            tmp_path / "survey", instrument=INSTRUMENT_TRIPLE_DOMINANCE, items=broken
        )
        with pytest.raises(ValueError, match="do not separate"):
            load_published_instruments(data_dir, instruments=[INSTRUMENT_TRIPLE_DOMINANCE])

    def test_an_allocation_instrument_with_no_framing_raises(self, tmp_path: Path) -> None:
        """Nine bare number pairs parse perfectly and measure something other than the instrument."""
        spec = PUBLISHED_INSTRUMENTS[INSTRUMENT_TRIPLE_DOMINANCE]
        data_dir = synthetic_published_file(
            tmp_path / "survey",
            instrument=INSTRUMENT_TRIPLE_DOMINANCE,
            instructions="   ",
            items=[
                {"option_payoffs": _synthetic_payoffs(3, position)}
                for position in range(1, spec.n_items + 1)
            ],
        )
        with pytest.raises(ValueError, match="instrument-level 'instructions'"):
            load_published_instruments(data_dir, instruments=[INSTRUMENT_TRIPLE_DOMINANCE])

    def test_an_allocation_item_renders_under_the_shared_framing(self, tmp_path: Path) -> None:
        data_dir = synthetic_published_file(
            tmp_path / "survey", instrument=INSTRUMENT_TRIPLE_DOMINANCE
        )
        item = load_published_instruments(data_dir, instruments=[INSTRUMENT_TRIPLE_DOMINANCE])[0]
        prompt = render_survey_prompt(item)
        assert prompt.startswith("Synthetic framing for an allocation task.")
        assert "points; the other party receives" in prompt

    def test_a_likert_framing_is_prepended_to_each_statement(self, tmp_path: Path) -> None:
        data_dir = synthetic_published_file(
            tmp_path / "survey", instructions="There are no synthetic framing sentences like this."
        )
        item = load_published_instruments(data_dir, instruments=[SYNTHETIC_INSTRUMENT])[0]
        assert item.stem.startswith("There are no synthetic framing sentences like this.")
        assert item.stem.endswith("Synthetic statement 1.")

    def test_an_allocation_instrument_never_loads_twins(self, tmp_path: Path) -> None:
        """A payoff table has no loaded vocabulary to remove, so a 'neutral' twin is another item."""
        spec = PUBLISHED_INSTRUMENTS[INSTRUMENT_TRIPLE_DOMINANCE]
        data_dir = synthetic_published_file(
            tmp_path / "survey",
            instrument=INSTRUMENT_TRIPLE_DOMINANCE,
            instructions="Synthetic framing for an allocation task.",
            items=[
                {
                    "option_payoffs": _synthetic_payoffs(3, position),
                    "neutral_stems": ["should be ignored"],
                }
                for position in range(1, spec.n_items + 1)
            ],
        )
        items = load_published_instruments(data_dir, instruments=[INSTRUMENT_TRIPLE_DOMINANCE])
        assert len(items) == spec.n_items
        assert all(item.wording == WORDING_AS_PUBLISHED for item in items)

    def test_a_wrong_option_count_is_dropped_and_then_raises_as_a_drift(
        self, tmp_path: Path
    ) -> None:
        spec = PUBLISHED_INSTRUMENTS[INSTRUMENT_TRIPLE_DOMINANCE]
        data_dir = synthetic_published_file(
            tmp_path / "survey",
            instrument=INSTRUMENT_TRIPLE_DOMINANCE,
            items=[{"option_payoffs": [[1, 1], [2, 2]]}] * spec.n_items,
        )
        with pytest.raises(ValueError, match="wrong number of allocation options"):
            load_published_instruments(data_dir, instruments=[INSTRUMENT_TRIPLE_DOMINANCE])


class TestTheBatteryEntryPoint:
    def test_an_unknown_family_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown survey families"):
            survey_battery(families=["not-a-family"])

    def test_a_repeated_family_raises(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            survey_battery(families=[FAMILY_NEGATIVE_CONTROL, FAMILY_NEGATIVE_CONTROL])

    def test_no_data_dir_assembles_nothing_and_is_refused(self) -> None:
        """The fresh-clone state: nothing runs without local item data, never a smaller battery."""
        with pytest.raises(ValueError, match="selected no item"):
            survey_battery()

    def test_a_filter_selecting_nothing_raises_rather_than_running_empty(
        self, tmp_path: Path
    ) -> None:
        """An empty section runs, writes a trace and summarises as healthy, so it is refused."""
        data_dir = synthetic_authored_file(tmp_path / "survey")
        synthetic_published_file(data_dir)
        with pytest.raises(ValueError, match="selected no item"):
            survey_battery(
                families=[PUBLISHED_INSTRUMENTS[INSTRUMENT_NARCISSISM].family],
                data_dir=data_dir,
                instruments=[SYNTHETIC_INSTRUMENT],
            )

    def test_a_family_filter_narrows_to_that_family(self, tmp_path: Path) -> None:
        data_dir = synthetic_authored_file(tmp_path / "survey")
        synthetic_published_file(data_dir)
        items = survey_battery(
            families=[FAMILY_NEGATIVE_CONTROL],
            data_dir=data_dir,
            instruments=[SYNTHETIC_INSTRUMENT],
        )
        assert {item.family for item in items} == {FAMILY_NEGATIVE_CONTROL}
        assert len(items) == 15

    def test_an_unknown_tier_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown survey tier"):
            survey_battery(tier="not-a-tier")

    def test_a_core_tier_filter_narrows_to_core_items_and_still_pairs_twins(
        self, tmp_path: Path
    ) -> None:
        """Core is position-level within instruments, so only this knob can select it."""
        data_dir = synthetic_authored_file(tmp_path / "survey")
        synthetic_published_file(data_dir)
        everything = survey_battery(data_dir=data_dir, instruments=[SYNTHETIC_INSTRUMENT])
        core = survey_battery(data_dir=data_dir, instruments=[SYNTHETIC_INSTRUMENT], tier=TIER_CORE)
        assert {item.tier for item in core} == {TIER_CORE}
        assert len(core) == sum(1 for item in everything if item.tier == TIER_CORE)
        assert 0 < len(core) < len(everything)

    def test_a_breadth_only_slice_of_a_single_keying_subscale_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Pinned on purpose: the competitiveness index's breadth positions are all reverse-keyed,
        so a breadth-only administration of it alone could never compute an acquiescence index and
        the instrument-level guard fires. A breadth leg that wants the contentiousness items runs
        them beside a mixed-keying subscale (or both tiers), not as a lone slice."""
        data_dir = synthetic_authored_file(tmp_path / "survey")
        synthetic_published_file(data_dir)
        with pytest.raises(ValueError, match="no subscale containing both keyings"):
            survey_battery(data_dir=data_dir, instruments=[SYNTHETIC_INSTRUMENT], tier=TIER_BREADTH)

    def test_a_data_dir_with_only_one_file_raises(self, tmp_path: Path) -> None:
        """Half an assembly is an operator mistake, not a smaller battery."""
        data_dir = synthetic_authored_file(tmp_path / "survey")
        with pytest.raises(FileNotFoundError, match="published"):
            survey_battery(data_dir=data_dir)

    def test_the_battery_checks_ids_twins_and_keying_in_one_place(self, tmp_path: Path) -> None:
        data_dir = synthetic_authored_file(tmp_path / "survey")
        synthetic_published_file(data_dir)
        items = survey_battery(data_dir=data_dir, instruments=[SYNTHETIC_INSTRUMENT])
        assert_unique_survey_ids(items)
        assert_every_twin_pairs(items)
        assert_every_reverse_key_has_a_sibling(items)
        assert (
            len(items)
            == len(AUTHORED_ITEM_SPECS) + PUBLISHED_INSTRUMENTS[SYNTHETIC_INSTRUMENT].n_items
        )

    def test_a_duplicate_id_raises(self) -> None:
        item = likert_item("synthetic-01")
        with pytest.raises(ValueError, match="duplicate item_id"):
            assert_unique_survey_ids([item, item])


class TestAdversarialParserFixtures:
    """Run every planted completion in the tracked fixture file through the real parsers.

    The fixtures deliberately include cases that must NOT parse, marked `known_unparsed`, which is
    what gives the section's parse-failure rate a documented floor: a later change that starts
    parsing one of them has to edit the fixture and say so.
    """

    @staticmethod
    def _cases() -> list[dict[str, Any]]:
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        return list(payload["cases"])

    def test_the_fixture_file_is_populated(self) -> None:
        cases = self._cases()
        assert len(cases) >= 40
        assert len({case["name"] for case in cases}) == len(cases)

    def test_both_outcomes_are_represented(self) -> None:
        """A fixture set that only contained parseable cases could not catch a lenient parser."""
        cases = self._cases()
        assert any(case["known_unparsed"] for case in cases)
        assert any(not case["known_unparsed"] for case in cases)

    def test_every_case_parses_to_its_recorded_expectation(self) -> None:
        for case in self._cases():
            visible, _ = strip_thinking(
                case["completion"], prefilled_think=case.get("prefilled_think", True)
            )
            shape = case["answer_shape"]
            if shape == "lettered":
                actual = parse_final_answer(visible, n_options=case["n_options"])
            elif shape == "numeric":
                actual = parse_split(visible, endowment=case["numeric_max"])
            else:
                actual = parse_tag(visible, "stance", vocabulary=case["vocabulary"])
            assert actual == case["expected"], f"{case['name']}: {case['why']}"

    def test_known_unparsed_is_consistent_with_the_expectation(self) -> None:
        for case in self._cases():
            assert case["known_unparsed"] == (case["expected"] is None), case["name"]

    def test_the_spelled_out_anchor_cases_are_present(self) -> None:
        """The fixtures the standalone-letter lookahead exists for; do not delete them.

        Both shapes, because they are different failures and only one of them is the lookahead's.
        The prose pair names an anchor without using the answer format at all, so no pattern
        anchors on it; the answer-line pair uses the format and writes the anchor word where the
        letter goes, which is the case the lookahead has to refuse.
        """
        names = {case["name"] for case in self._cases()}
        assert "lettered-spelled-out-anchor-agree" in names
        assert "lettered-anchor-word-on-the-answer-line-agree" in names

    def test_an_anchor_word_on_the_answer_line_never_parses_as_its_leading_letter(self) -> None:
        """Gate 5's named sabotage, pinned on its own so the red is unmistakable.

        Deleting the standalone-letter lookahead from `games.probes.FINAL_ANSWER_PATTERN` makes
        `FINAL ANSWER: Agree` parse as option A -- a confident answer at the opposite end of the
        scale from the one the model wrote, indistinguishable downstream from a real one. This is the
        `medical-newcomb` "Avoid"-read-as-"A" bug in its Likert form.

        Which fixtures this uses is itself a finding (2026-08-22). The pair originally named as the
        load-bearing case, `lettered-spelled-out-anchor-{agree,almost}`, carries no `FINAL ANSWER`
        line at all, so the pattern never anchors on it and deleting the lookahead leaves it unparsed
        -- it could not go red, and the fixture file said it would. The sabotage has to reach the
        branch it is aimed at, so the cases below are the ones that use the requested format.

        The positive controls in the same test are what stop this passing vacuously: a parser that
        returned None for everything would satisfy the anchor assertions alone.
        """
        by_name = {case["name"]: case for case in self._cases()}
        for name in (
            "lettered-anchor-word-on-the-answer-line-agree",
            "lettered-anchor-word-on-the-answer-line-almost",
        ):
            case = by_name[name]
            visible, _ = strip_thinking(
                case["completion"], prefilled_think=case.get("prefilled_think", True)
            )
            assert "FINAL ANSWER" in visible, name
            assert parse_final_answer(visible, n_options=case["n_options"]) is None, name
        for name in ("lettered-plain", "lettered-parenthesised"):
            case = by_name[name]
            visible, _ = strip_thinking(
                case["completion"], prefilled_think=case.get("prefilled_think", True)
            )
            parsed = parse_final_answer(visible, n_options=case["n_options"])
            assert parsed == case["expected"], name
            assert parsed is not None, name

    def test_the_tag_placeholder_echo_cases_are_present(self) -> None:
        """The R4 placeholder example must stay covered: echoing it is not an answer."""
        names = {case["name"] for case in self._cases()}
        assert "tagged-placeholder-echoed-verbatim" in names
        assert "tagged-placeholder-echo-then-real-answer" in names

    def test_every_case_carries_a_reason(self) -> None:
        for case in self._cases():
            assert case["why"].strip(), case["name"]


class TestTheRealLocalAuthoredFile:
    """Pins on the repairs that live in the item text itself, run only where the text exists.

    On a fresh clone these skip loudly with the reason -- the synthetic-fixture suite above covers
    all mechanics either way. On this machine they hold the real `authored.json` to the repairs the
    2026-08-21 trim decision ordered.
    """

    @staticmethod
    def _real_items() -> list[SurveyItem]:
        if not (REAL_DATA_DIR / AUTHORED_FILENAME).is_file():
            pytest.skip(
                f"{REAL_DATA_DIR / AUTHORED_FILENAME} is not on this machine (fresh clone); "
                f"assemble it as games/data/survey/README.md describes to run the real-text pins"
            )
        return load_authored_items(REAL_DATA_DIR)

    def test_every_real_numeric_render_carries_a_rotated_example(self) -> None:
        """Repair R1, amended by its own documented fallback after the 2026-08-22 L4 smoke: bare
        `<keep></keep>` tags parsed 9/16 at 2B (the model wrote `<100>` or a bare number), so the
        instruction shows a worked example whose value rotates across renders -- never one fixed
        number, because a fixed example anchors a small model at exactly the measured value."""
        for item in self._real_items():
            if item.kind != SURVEY_NUMERIC:
                continue
            prompts = set()
            for order in [order for _, order in battery_orders(item)] or [None]:
                for example in NUMERIC_EXAMPLE_ROTATION:
                    prompt = render_survey_prompt(item, option_order=order, numeric_example=example)
                    assert f"<keep>{example}</keep>" in prompt, item.item_id
                    prompts.add(prompt)
            assert len(prompts) > 1, item.item_id
            with pytest.raises(ValueError, match="numeric_example"):
                render_survey_prompt(item)

    def test_every_real_self_prediction_swaps_its_action_descriptions(self) -> None:
        """Repair R2: the loader enforces presence; this pins that the swap is a real rewording."""
        for item in self._real_items():
            if item.family != FAMILY_SELF_PREDICTION:
                continue
            assert item.stem_swapped is not None
            assert item.stem_swapped != item.stem

    def test_every_real_self_prediction_lists_four_outcomes_in_both_framings(self) -> None:
        """The 2026-08-22 elicitation repair, pinned by structure because the words cannot be here.

        Before it, every item described its differ case as a clause naming which chooser was
        credited what, and the traces show the model spending its whole budget re-deriving whether
        that clause meant itself -- a median 386 occurrences of "Wait" per truncated completion, on
        94% of renders. The repair states all four cells from the answering side instead. The pin is
        the count of outcome lines, not their wording: an item text guard runs over this file, and a
        test quoting the repair it protects would be the leak.
        """
        outcome_lines = 4
        for item in self._real_items():
            if item.family != FAMILY_SELF_PREDICTION:
                continue
            for framing in (item.stem, item.stem_swapped or ""):
                listed = [line for line in framing.splitlines() if line.startswith("- ")]
                assert len(listed) == outcome_lines, item.item_id

    def test_the_two_real_framings_differ_only_in_their_payoff_numerals(self) -> None:
        """A swapped framing is the same question with the numbers exchanged, and nothing else.

        The reflection through `numeric_max - x` is only valid while both framings ask about
        whichever action their own description introduced first. This holds the outcome lines to
        that: strip the digits and the two framings' outcome blocks must be the same text. Keyed on
        those lines rather than the whole stem because one item names its two actions in prose, so
        its opening legitimately differs between framings while its table does not.
        """
        for item in self._real_items():
            if item.family != FAMILY_SELF_PREDICTION:
                continue
            assert item.stem_swapped is not None
            skeletons = [
                "".join(
                    character
                    for line in framing.splitlines()
                    if line.startswith("- ")
                    for character in line
                    if not character.isdigit()
                )
                for framing in (item.stem, item.stem_swapped)
            ]
            assert skeletons[0] == skeletons[1], item.item_id
            assert skeletons[0].strip(), item.item_id

    def test_every_real_self_prediction_closes_with_the_one_shared_elicitation(self) -> None:
        """One question per family, so no item can be left asked the way that censored the family."""
        path = REAL_DATA_DIR / AUTHORED_FILENAME
        if not path.is_file():
            pytest.skip(f"{path} is not on this machine (fresh clone)")
        declared = json.loads(path.read_text(encoding="utf-8"))[ELICITATION_BLOCKS_KEY]
        block = declared[FAMILY_SELF_PREDICTION].strip()
        assert block
        asked = self._real_items()
        family = [item for item in asked if item.family == FAMILY_SELF_PREDICTION]
        assert family
        for item in family:
            for framing in (item.stem, item.stem_swapped or ""):
                assert framing.endswith(block), item.item_id
                assert framing.count(block) == 1, item.item_id
        # No other family may have picked it up, which is what the loader's family list decides.
        for item in asked:
            if item.family == FAMILY_SELF_PREDICTION:
                continue
            assert block not in item.stem, item.item_id

    def test_the_repaired_controls_carry_a_fixed_context(self) -> None:
        """Repair R7: without a fixed audience, "it depends" is the reasonable answer."""
        by_id = {item.item_id: item for item in self._real_items()}
        for item_id in (
            "negative-control-date-format",
            "negative-control-quote-style",
            "negative-control-units",
        ):
            stem = by_id[item_id].stem.casefold()
            assert "only you will read" in stem or "your own records" in stem, item_id

    def test_the_public_goods_item_states_both_actions_and_full_payoffs(self) -> None:
        """Repair R5: both actions explicit, all four outcomes numbered, first option named."""
        by_id = {item.item_id: item for item in self._real_items()}
        item = by_id["self-prediction-public-goods"]
        for stem in (item.stem, item.stem_swapped or ""):
            lowered = stem.casefold()
            assert "contribut" in lowered
            assert "keep" in lowered
            for figure in ("16", "10", "8", "18"):
                assert figure in stem
            assert "first option" in lowered

    def test_every_real_negative_control_render_passes_the_vocabulary_screen(self) -> None:
        """No control may name the training construct, in any render it can be administered as.

        Design constraint (b) of the 2026-08-25 headroom adoption, held for the whole family: a
        placebo whose wording carries cooperation, counterpart or allocation vocabulary can move
        for real reasons, and a placebo that can move for real reasons is not a placebo. Screened
        over full renders rather than stems so the instruction blocks stay covered too.
        """
        screened = 0
        for item in self._real_items():
            if item.family != FAMILY_NEGATIVE_CONTROL:
                continue
            orders = [order for _, order in battery_orders(item)] or [None]
            examples = list(NUMERIC_EXAMPLE_ROTATION) if item.kind == SURVEY_NUMERIC else [None]
            for order in orders:
                for example in examples:
                    assert_no_loaded_vocabulary(
                        render_survey_prompt(item, option_order=order, numeric_example=example)
                    )
                    screened += 1
        assert screened >= 15, f"only {screened} renders screened; the family filter went stale"

    def test_the_item_text_guard_extraction_floor_holds(self) -> None:
        """The scanner arms itself from this file; a collapsed extraction passes everything."""
        if not (REAL_DATA_DIR / AUTHORED_FILENAME).is_file():
            pytest.skip(f"{REAL_DATA_DIR / AUTHORED_FILENAME} is not on this machine (fresh clone)")
        payload = json.loads((REAL_DATA_DIR / AUTHORED_FILENAME).read_text(encoding="utf-8"))
        phrases, _ = sources_from_json_payload(payload)
        assert len(phrases) >= 250, (
            f"only {len(phrases)} phrases extracted from authored.json; the guard is running "
            f"near-empty, which passes everything -- fix the extraction, do not lower the floor."
        )
