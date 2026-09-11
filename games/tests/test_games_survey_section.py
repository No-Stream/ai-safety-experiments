"""Tests for the eval battery's self-report SECTION: the wiring, not the instruments.

`games/tests/test_games_survey.py` covers `games/survey.py` itself -- the item validation, the
parsers, the composite arithmetic against hand math, the loader's refusals. This file covers the
three seams that module cannot test, because each of them lives in another file:

*   **`games/evals.py`** -- the section builder. Does every item reach the model under both
    presentation orders and every sample, does a record carry the declared contract plus the four
    fields the writer owns, and does the section's summary come back as itself when the backend's
    answer is known exactly.
*   **The cross-section seam.** The self-prediction family is scored against the cooperation rate the
    *same cell* measured, so its number depends on two sections of one battery. This is where that is
    tested, including the ordering trap: the gap must not depend on which order the caller listed the
    sections in.
*   **`games/battery_tables.py`** -- the five readout tables, built from a trace this file wrote.

Two fixture conventions, both deliberate. **No published instrument text appears here**: every
loader test writes a synthetic file under `tmp_path` in `published.json`'s shape, derived from the
tracked specs, which is the same choice `test_games_probes.py` makes for DTBench and means the suite
is green in a fresh clone with no skips. And **the scripted backends answer a known letter**, because
a section that quietly averaged, defaulted or dropped unparsed answers would still produce a
plausible distribution -- the only way to know the pipeline reports what it saw is to feed it
something whose answer is known exactly.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import pytest

from games import battery_tables
from games.evals import (
    DEFAULT_SECTIONS,
    EVAL_RENDER_GRADING_BY_GAME,
    ORDER_NOT_APPLICABLE,
    RECORD_META,
    SECTION_GAME_BEHAVIOR,
    SECTION_SELF_REPORT,
    SECTIONS,
    EvalConfig,
    read_eval_records,
    run_eval_battery,
)
from games.probes import ORDER_AS_AUTHORED, ORDER_REVERSED
from games.prompts import SPLIT_EVAL, generate_prompt_rows
from games.report import load_traces
from games.survey import (
    AUTHORED_FILENAME,
    AUTHORED_ITEM_SPECS,
    ELICITATION_BLOCKS_KEY,
    FAMILIES,
    FAMILIES_WITH_SHARED_ELICITATION,
    FAMILY_COMPETITIVENESS,
    FAMILY_SELF_PREDICTION,
    INSTRUMENT_COMPETITIVENESS_INDEX,
    INSTRUMENT_SVO_SLIDER,
    INSTRUMENT_TRIPLE_DOMINANCE,
    PUBLISHED_INSTRUMENTS,
    SCHEMA_VERSION,
    SURVEY_ALLOCATION,
    SURVEY_CHEAP_TALK,
    SURVEY_CHOICE,
    SURVEY_LIKERT,
    SURVEY_NUMERIC,
    SURVEY_ORDERED_CHOICE,
    SURVEY_RECORD_FIELDS,
    SURVEY_TAGGED,
    TIER_CORE,
    WORDING_AS_PUBLISHED,
    WORDING_NEUTRAL_TWIN,
    numeric_example_rotation,
    subscale_composites,
    survey_battery,
)
from reward_hacking.model_backend import MockBackend

if TYPE_CHECKING:
    from pathlib import Path

BASE_META: dict[str, Any] = {"arm": "twin-pd-group", "step": 0, "model_path": "base-model"}

# Fields the eval writer owns rather than `games.survey`; see `_self_report_record`'s docstring.
WRITER_OWNED_FIELDS: frozenset[str] = frozenset(
    {
        "record",
        "sample_index",
        "option_order_name",
        "option_order",
        "numeric_example",
        "truncated_thinking",
        "completion",
        "visible_text",
    }
)

ALWAYS_A = "reasoning</think>FINAL ANSWER: A"
UNPARSEABLE = "reasoning</think>I would rather not answer."

# Every registered instrument by default, because the loader refuses to run a partial battery
# unasked: a file supplying some of them and a section that ran anyway is exactly the failure it
# exists to prevent. The three named here cover all three published answer shapes -- a Likert scale
# carrying both keyings, the nine-option allocation slider, the three-option forced choice -- and are
# the subsets the narrower tests ask for by name.
SYNTHETIC_INSTRUMENTS: tuple[str, ...] = tuple(PUBLISHED_INSTRUMENTS)
SHAPE_COVERING_INSTRUMENTS: tuple[str, ...] = (
    INSTRUMENT_COMPETITIVENESS_INDEX,
    INSTRUMENT_SVO_SLIDER,
    INSTRUMENT_TRIPLE_DOMINANCE,
)


def _synthetic_payoffs(n_options: int, position: int) -> list[list[int]]:
    """Build allocation options that separate the three orientations, for any count of three or more.

    The three-option case has to carry exactly one own-maximising, one joint-maximising and one
    difference-maximising option, because `triple_dominance_orientations` refuses a triple that does
    not separate them -- which is the load-time guard these fixtures also exercise.
    """
    if n_options == 3:
        return [[97 + position, 41], [88 + position, 88 + position], [93 + position, 23]]
    step = 60 // (n_options - 1)
    return [[80, 80 - index * step] for index in range(n_options)]


# The marker every synthetic neutral twin's stem carries, so a scripted backend can tell a twin's
# render from its parent's and answer the two differently. Without that, both wordings score alike
# and a reduction that pooled them is indistinguishable from one that did not.
NEUTRAL_STEM_MARKER = "neutral placeholder statement"

# Which anchor each wording's scripted answer picks. Two low points rather than the extremes,
# because every Likert instrument registered here is at least five wide, so both exist on all of
# them however many points the scale carries.
PUBLISHED_ANSWER_POINT = 2
TWIN_ANSWER_POINT = 1


def placeholder_anchor(point: int) -> str:
    """One synthetic Likert anchor, shared by the file writer and the backend that answers it."""
    return f"placeholder anchor {point}"


def synthetic_published_file(
    directory: Path,
    *,
    instruments: tuple[str, ...] = SYNTHETIC_INSTRUMENTS,
    twinned: bool = False,
) -> Path:
    """Write a `published.json` matching the tracked specs, with placeholder strings only.

    Derived from `PUBLISHED_INSTRUMENTS` rather than hand-written, so it stays at the declared item
    counts and scale widths and exercises the real loader's real validation. Not one word comes from
    an instrument, which is what keeps this file committable.

    `twinned` gives every Likert item a lexically neutral twin, through the same `neutral_stems`
    key the assembled file uses, so the loader builds the twins rather than a test hand-writing
    them. Off by default: a twin doubles an instrument's renders, and only the tables that read the
    two wordings apart need them.
    """
    blocks: dict[str, Any] = {}
    for name in instruments:
        spec = PUBLISHED_INSTRUMENTS[name]
        if spec.kind == SURVEY_LIKERT:
            items: list[dict[str, Any]] = [
                {"stem": f"Placeholder statement {position} for {name}."}
                for position in range(1, spec.n_items + 1)
            ]
            if twinned:
                for position, item in enumerate(items, start=1):
                    item["neutral_stems"] = [f"{NEUTRAL_STEM_MARKER} {position} for {name}."]
            blocks[name] = {
                "anchors": [placeholder_anchor(point) for point in range(1, spec.scale_points + 1)],
                "items": items,
            }
        else:
            blocks[name] = {
                "instructions": "Placeholder framing for an allocation task.",
                "items": [
                    {"option_payoffs": _synthetic_payoffs(spec.n_options, position)}
                    for position in range(1, spec.n_items + 1)
                ],
            }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "published.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "instruments": blocks}), encoding="utf-8"
    )
    return directory


# The marker every synthetic swapped stem carries, so a scripted backend can tell which wording a
# prompt rendered and answer the complement -- which is exactly what a real position-biased model
# would do, and what the canonicalising parse has to undo.
SWAPPED_STEM_MARKER = "swapped placeholder stem"


def synthetic_authored_file(directory: Path) -> Path:
    """Write an `authored.json` covering every tracked spec, with placeholder strings only."""
    items: dict[str, dict[str, Any]] = {}
    for spec in AUTHORED_ITEM_SPECS:
        block: dict[str, Any] = {"stem": f"Placeholder stem for {spec.item_id}."}
        if spec.requires_swapped_stem:
            block["stem_swapped"] = f"{SWAPPED_STEM_MARKER} for {spec.item_id}."
        if spec.kind in (SURVEY_LIKERT, SURVEY_CHOICE, SURVEY_ORDERED_CHOICE):
            block["options"] = [
                f"placeholder option {index}" for index in range(1, spec.n_options + 1)
            ]
        if spec.kind in (SURVEY_TAGGED, SURVEY_CHEAP_TALK):
            block["vocabulary"] = [f"wordnumber{index}" for index in range(1, spec.n_tag_words + 1)]
        if spec.kind == SURVEY_ALLOCATION:
            block["option_payoffs"] = _synthetic_payoffs(spec.n_options, 1)
        items[spec.item_id] = block
    directory.mkdir(parents=True, exist_ok=True)
    (directory / AUTHORED_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "items": items,
                ELICITATION_BLOCKS_KEY: {
                    family: f"Placeholder closing question for {family}."
                    for family in FAMILIES_WITH_SHARED_ELICITATION
                },
            }
        ),
        encoding="utf-8",
    )
    return directory


def synthetic_survey_dir(
    directory: Path,
    *,
    instruments: tuple[str, ...] = SYNTHETIC_INSTRUMENTS,
    twinned: bool = False,
) -> Path:
    """Write both local item files, which is what an assembled `games/data/survey/` looks like."""
    synthetic_authored_file(directory)
    synthetic_published_file(directory, instruments=instruments, twinned=twinned)
    return directory


def answer_the_anchor_the_wording_calls_for(prompt: str) -> str:
    """Answer the top anchor on an as-published stem and the bottom anchor on its neutral twin.

    Keyed on the ANCHOR TEXT rather than on a letter, because every item is asked under both
    presentation orders: one fixed letter is a different canonical answer in each of them, so both
    wordings would average to the scale's midpoint and a reduction that pooled the twins in would
    read the same as one that kept them out. Answering the anchor instead pins each wording to one
    canonical score however the options were ordered.

    Anything with no anchor line -- the authored families answer in other formats entirely -- goes
    unparsed, which is what keeps a scripted backend about the one instrument under test.
    """
    point = TWIN_ANSWER_POINT if NEUTRAL_STEM_MARKER in prompt else PUBLISHED_ANSWER_POINT
    target = placeholder_anchor(point)
    for line in prompt.splitlines():
        letter, _, option_text = line.partition(") ")
        if option_text == target:
            return f"reasoning</think>FINAL ANSWER: {letter}"
    return UNPARSEABLE


def run_section(
    tmp_path: Path,
    *,
    backend: MockBackend | None = None,
    config: EvalConfig | None = None,
    sections: tuple[str, ...] = (SECTION_SELF_REPORT,),
    name: str = "eval.jsonl",
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """Run the requested sections offline and return (summary, records, trace path).

    A config without a `survey_data_dir` gets a synthetic one under `tmp_path`, because the
    section refuses to run without local item data and almost every test here wants it to run;
    the one test about the refusal itself calls `run_eval_battery` directly.
    """
    resolved = config if config is not None else EvalConfig(survey_samples=1, batch_size=16)
    if resolved.survey_data_dir is None and SECTION_SELF_REPORT in sections:
        resolved = dataclasses.replace(
            resolved, survey_data_dir=synthetic_survey_dir(tmp_path / "survey-data")
        )
    out_path = tmp_path / name
    summary = run_eval_battery(
        backend if backend is not None else MockBackend(responses=[ALWAYS_A], model_id="always-a"),
        sections=list(sections),
        out_path=out_path,
        meta=BASE_META,
        config=resolved,
    )
    return summary, read_eval_records(out_path), out_path


def survey_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The self-report rows of a trace, meta record dropped."""
    return [record for record in records if record.get("record") == SECTION_SELF_REPORT]


class TestTheSectionIsWiredIn:
    def test_the_section_is_registered_and_dispatchable(self) -> None:
        assert SECTION_SELF_REPORT in SECTIONS
        assert SECTION_SELF_REPORT in battery_tables.SECTIONS

    def test_it_is_not_a_default_section(self) -> None:
        """Opt-in: it is the largest section by render count and its published half needs a file."""
        assert SECTION_SELF_REPORT not in DEFAULT_SECTIONS
        assert set(DEFAULT_SECTIONS) < set(SECTIONS)

    def test_asking_for_it_by_name_produces_records(self, tmp_path: Path) -> None:
        summary, records, _ = run_section(tmp_path)
        assert summary[SECTION_SELF_REPORT]["n_records"] == len(survey_records(records))
        assert summary[SECTION_SELF_REPORT]["n_records"] > 0

    def test_the_trace_still_starts_with_the_meta_record(self, tmp_path: Path) -> None:
        _, records, _ = run_section(tmp_path)
        assert records[0]["record"] == RECORD_META
        assert records[0]["sections"] == [SECTION_SELF_REPORT]


class TestEveryItemIsAskedUnderBothOrders:
    def test_an_option_bearing_item_is_rendered_once_per_order_per_sample(
        self, tmp_path: Path
    ) -> None:
        _, records, _ = run_section(tmp_path, config=EvalConfig(survey_samples=3, batch_size=32))
        lettered = [
            record for record in survey_records(records) if record["kind"] != SURVEY_NUMERIC
        ]
        by_item: dict[str, set[tuple[str, int]]] = {}
        for record in lettered:
            key = (str(record["option_order_name"]), int(record["sample_index"]))
            by_item.setdefault(str(record["item_id"]), set()).add(key)
        assert by_item
        expected = {
            (order, sample) for order in (ORDER_AS_AUTHORED, ORDER_REVERSED) for sample in range(3)
        }
        assert all(seen == expected for seen in by_item.values())

    def test_a_numeric_item_without_a_swap_carries_the_no_options_sentinel(
        self, tmp_path: Path
    ) -> None:
        """The numeric placebo has no swapped stem, so pretending it had one would fake a zero."""
        _, records, _ = run_section(tmp_path, config=EvalConfig(survey_samples=2, batch_size=32))
        placebo = [
            record
            for record in survey_records(records)
            if record["item_id"] == "negative-control-bullet-rate"
        ]
        assert len(placebo) == 2
        assert {record["option_order_name"] for record in placebo} == {ORDER_NOT_APPLICABLE}
        assert all(record["option_order"] == [] for record in placebo)

    def test_a_self_prediction_item_renders_under_both_stem_wordings(self, tmp_path: Path) -> None:
        """The swapped-stem counterbalance (repair R2) has to reach the section, not just the
        renderer: every self-prediction item is asked once per wording per sample."""
        _, records, _ = run_section(tmp_path, config=EvalConfig(survey_samples=2, batch_size=64))
        predictions = [
            record
            for record in survey_records(records)
            if record["family"] == FAMILY_SELF_PREDICTION
        ]
        by_item: dict[str, set[tuple[str, int]]] = {}
        for record in predictions:
            key = (str(record["option_order_name"]), int(record["sample_index"]))
            by_item.setdefault(str(record["item_id"]), set()).add(key)
        assert len(by_item) == 8
        expected = {
            (order, sample) for order in (ORDER_AS_AUTHORED, ORDER_REVERSED) for sample in range(2)
        }
        assert all(seen == expected for seen in by_item.values())

    def test_the_reversed_render_maps_an_answered_letter_to_the_opposite_canonical_answer(
        self, tmp_path: Path
    ) -> None:
        """The whole point of the order control: "A" is not one answer, it is two."""
        _, records, _ = run_section(tmp_path)
        lettered = [
            record
            for record in survey_records(records)
            if record["kind"] != SURVEY_NUMERIC and record["canonical_index"] is not None
        ]
        assert lettered
        for record in lettered:
            assert record["presented_index"] == 0
            expected = (
                0
                if record["option_order_name"] == ORDER_AS_AUTHORED
                else int(record["scale_points"]) - 1
            )
            assert record["canonical_index"] == expected

    def test_a_letter_bias_reads_as_total_order_disagreement_not_as_a_position(
        self, tmp_path: Path
    ) -> None:
        summary, _, _ = run_section(tmp_path)
        assert summary[SECTION_SELF_REPORT]["order_disagreement_rate"] == pytest.approx(1.0)


class TestTheRecordContractReachesTheTrace:
    def test_every_record_carries_the_declared_fields_plus_the_writers_own(
        self, tmp_path: Path
    ) -> None:
        _, records, _ = run_section(tmp_path)
        expected = set(SURVEY_RECORD_FIELDS) | WRITER_OWNED_FIELDS
        for record in survey_records(records):
            assert set(record) == expected

    def test_an_unparsed_render_writes_the_same_keys_with_nulls(self, tmp_path: Path) -> None:
        _, records, _ = run_section(
            tmp_path, backend=MockBackend(responses=[UNPARSEABLE], model_id="refuser")
        )
        rows = survey_records(records)
        assert rows
        expected = set(SURVEY_RECORD_FIELDS) | WRITER_OWNED_FIELDS
        for record in rows:
            assert set(record) == expected
            assert record["parsed"] is False
            assert record["response"] is None
            assert record["score"] is None

    def test_numeric_renders_rotate_the_worked_example_and_record_it(self, tmp_path: Path) -> None:
        """R1's fallback in the writer: the example varies across an item's renders and every
        record says which value its render showed, so an anchoring analysis (answer == example
        rate) is a re-read of the trace rather than a re-run.

        Each record is checked against ITS OWN item's rotation, not one module-wide tuple: the
        values are fractions of the item's bound, so a family landing a numeric item bounded
        somewhere other than 100 rotates through different values and a shared tuple would either
        refuse a correct render or pass a wrong one.
        """
        data_dir = synthetic_survey_dir(tmp_path / "survey-data")
        _, records, _ = run_section(
            tmp_path,
            config=EvalConfig(survey_samples=1, batch_size=16, survey_data_dir=data_dir),
        )
        rotations = {
            item.item_id: numeric_example_rotation(item)
            for item in survey_battery(data_dir=data_dir)
            if item.kind == SURVEY_NUMERIC
        }
        rows = survey_records(records)
        numeric = [record for record in rows if record["kind"] == SURVEY_NUMERIC]
        assert numeric
        examples_by_item: dict[str, set[int]] = {}
        for record in numeric:
            item_id = str(record["item_id"])
            assert record["numeric_example"] in rotations[item_id], item_id
            examples_by_item.setdefault(str(record["item_id"]), set()).add(
                int(record["numeric_example"])
            )
        multi_render_items = {
            str(record["item_id"])
            for record in numeric
            if str(record["option_order_name"]) != ORDER_NOT_APPLICABLE
        }
        assert multi_render_items
        for item_id in multi_render_items:
            assert len(examples_by_item[item_id]) > 1, item_id
        for record in rows:
            if record["kind"] != SURVEY_NUMERIC:
                assert record["numeric_example"] is None, record["item_id"]

    def test_a_refusing_backend_reads_as_a_parse_failure_rather_than_a_disposition(
        self, tmp_path: Path
    ) -> None:
        summary, _, _ = run_section(
            tmp_path, backend=MockBackend(responses=[UNPARSEABLE], model_id="refuser")
        )
        section = summary[SECTION_SELF_REPORT]
        assert section["parse_failure_rate"] == pytest.approx(1.0)
        assert section["subscale_composites"] == {}
        assert section["order_disagreement_rate"] is None
        assert all(rate["n_parsed"] == 0 for rate in section["parse_rate_by_family"].values())
        assert all(rate["n_asked"] > 0 for rate in section["parse_rate_by_family"].values())

    def test_the_full_completion_text_survives_into_the_record(self, tmp_path: Path) -> None:
        """Re-running to recover per-item detail costs GPU hours; the trace keeps the raw text."""
        _, records, _ = run_section(tmp_path)
        assert all(record["completion"] == ALWAYS_A for record in survey_records(records))


class TestNothingRunsWithoutLocalItemData:
    def test_requesting_the_section_with_no_data_dir_is_refused_before_anything_generates(
        self, tmp_path: Path
    ) -> None:
        """The fresh-clone state. Refused up front, so a missing file cannot cost a section of
        GPU time before it is discovered -- and never a quietly smaller battery."""
        with pytest.raises(ValueError, match="needs survey_data_dir"):
            run_eval_battery(
                MockBackend(responses=[ALWAYS_A], model_id="always-a"),
                sections=[SECTION_SELF_REPORT],
                out_path=tmp_path / "refused.jsonl",
                meta=BASE_META,
                config=EvalConfig(survey_samples=1, batch_size=16),
            )
        assert not (tmp_path / "refused.jsonl").exists()

    def test_the_meta_records_where_the_item_text_came_from(self, tmp_path: Path) -> None:
        _, records, _ = run_section(tmp_path)
        recorded = records[0]["eval_config"]["survey_data_dir"]
        assert recorded is not None
        assert recorded.endswith("survey-data")

    def test_every_family_in_the_trace_is_registered(self, tmp_path: Path) -> None:
        summary, records, _ = run_section(tmp_path)
        families = {str(record["family"]) for record in survey_records(records)}
        assert families
        assert families <= set(FAMILIES)
        assert summary[SECTION_SELF_REPORT]["n_families"] == len(families)

    def test_the_negative_control_family_reaches_the_trace(self, tmp_path: Path) -> None:
        """The section's own drift placebo: if these move as much as the rest, nothing is readable."""
        _, records, _ = run_section(tmp_path)
        assert any(
            "negative-control" in str(record["instrument"]) for record in survey_records(records)
        )


class TestThePublishedHalfReachesTheSection:
    def test_a_synthetic_file_adds_every_instrument_at_its_declared_size(
        self, tmp_path: Path
    ) -> None:
        data_dir = synthetic_survey_dir(tmp_path / "survey")
        _, records, _ = run_section(
            tmp_path,
            config=EvalConfig(survey_samples=1, batch_size=128, survey_data_dir=data_dir),
        )
        by_instrument: dict[str, set[str]] = {}
        for record in survey_records(records):
            by_instrument.setdefault(str(record["instrument"]), set()).add(str(record["item_id"]))
        for name, spec in PUBLISHED_INSTRUMENTS.items():
            assert len(by_instrument[name]) == spec.n_items

    def test_the_meta_names_the_instruments_the_cell_actually_asked(self, tmp_path: Path) -> None:
        data_dir = synthetic_survey_dir(tmp_path / "survey")
        _, records, _ = run_section(
            tmp_path,
            config=EvalConfig(
                survey_samples=1,
                batch_size=32,
                survey_data_dir=data_dir,
                survey_instruments=(INSTRUMENT_TRIPLE_DOMINANCE,),
            ),
        )
        assert records[0]["eval_config"]["survey_instruments"] == [INSTRUMENT_TRIPLE_DOMINANCE]
        instruments = {str(record["instrument"]) for record in survey_records(records)}
        assert INSTRUMENT_SVO_SLIDER not in instruments
        assert INSTRUMENT_TRIPLE_DOMINANCE in instruments

    def test_a_requested_instrument_the_file_lacks_stops_the_section(self, tmp_path: Path) -> None:
        """A partial battery that ran anyway would summarise as a healthy section that asked less."""
        data_dir = synthetic_survey_dir(
            tmp_path / "survey", instruments=(INSTRUMENT_TRIPLE_DOMINANCE,)
        )
        with pytest.raises(ValueError, match="missing requested instrument"):
            run_section(
                tmp_path,
                config=EvalConfig(
                    survey_samples=1,
                    batch_size=16,
                    survey_data_dir=data_dir,
                    survey_instruments=(INSTRUMENT_SVO_SLIDER,),
                ),
            )

    def test_the_forced_choice_records_carry_a_derived_orientation(self, tmp_path: Path) -> None:
        data_dir = synthetic_survey_dir(
            tmp_path / "survey", instruments=(INSTRUMENT_TRIPLE_DOMINANCE,)
        )
        summary, records, _ = run_section(
            tmp_path,
            config=EvalConfig(
                survey_samples=1,
                batch_size=32,
                survey_data_dir=data_dir,
                survey_instruments=(INSTRUMENT_TRIPLE_DOMINANCE,),
            ),
        )
        rows = survey_records(records)
        triples = [
            record
            for record in rows
            if record["instrument"] == INSTRUMENT_TRIPLE_DOMINANCE and record["parsed"]
        ]
        assert triples
        assert all(record["orientation"] is not None for record in triples)
        # The counted denominator is every record that CARRIES an orientation, not the published
        # triple's records: `survey_instruments` filters the published half only, and the authored
        # allocation items ride along and are read the same way whenever their options separate the
        # three orientations. Asserting against the triple's own count alone would go red the next
        # time an authored allocation family lands, which is exactly what it did on 2026-08-22.
        oriented = [record for record in rows if record["orientation"] is not None]
        assert len(oriented) > len(triples)
        assert sum(summary[SECTION_SELF_REPORT]["orientation_counts"].values()) == len(oriented)

    def test_the_slider_records_carry_both_sides_of_the_allocation(self, tmp_path: Path) -> None:
        """The angle is recomputed from these months later, so the payoffs ride on every record."""
        data_dir = synthetic_survey_dir(tmp_path / "survey", instruments=(INSTRUMENT_SVO_SLIDER,))
        summary, records, _ = run_section(
            tmp_path,
            config=EvalConfig(
                survey_samples=1,
                batch_size=64,
                survey_data_dir=data_dir,
                survey_instruments=(INSTRUMENT_SVO_SLIDER,),
            ),
        )
        slider = [
            record
            for record in survey_records(records)
            if record["instrument"] == INSTRUMENT_SVO_SLIDER and record["parsed"]
        ]
        assert slider
        assert all(record["payoff_self"] is not None for record in slider)
        assert all(record["payoff_other"] is not None for record in slider)
        assert summary[SECTION_SELF_REPORT]["svo_angle_degrees"] is not None


class TestCalibrationCrossesTwoSections:
    """The one number in this section scored against the artifact rather than the report."""

    @staticmethod
    def _predicting_backend(percentage: int) -> MockBackend:
        """Answer every self-prediction item with one percentage, and cooperate in every game.

        The game half has to look up the cooperative label by prompt: `games.prompts` counterbalances
        which label carries cooperation per scenario, so a fixed string would parse on some frames and
        fail on others and "always cooperates" would not be expressible at all.
        """
        coop_by_prompt = {
            row["prompt"]: row["coop_label"]
            for row in generate_prompt_rows(
                "twin-pd", EVAL_RENDER_GRADING_BY_GAME["twin-pd"], split=SPLIT_EVAL
            )
        }

        def respond(prompt: str) -> str:
            label = coop_by_prompt.get(prompt)
            if label is not None:
                return f"reasoning</think><action>{label}</action>"
            if "<keep>" in prompt:
                # A swapped render asks about the OTHER action, so a consistent predictor answers
                # the complement there -- and the canonicalising parse must map it back, which is
                # exactly what makes the predicted mean read `percentage` under both wordings.
                figure = 100 - percentage if SWAPPED_STEM_MARKER in prompt else percentage
                return f"reasoning</think><keep>{figure}</keep>"
            return "reasoning</think>FINAL ANSWER: A"

        return MockBackend(responses=respond, model_id="predictor")

    def test_the_gap_is_predicted_minus_measured_on_one_scale(self, tmp_path: Path) -> None:
        """Hand math: the backend predicts 100% and the section reads its measured rate as a fraction."""
        summary, _, _ = run_section(
            tmp_path,
            backend=self._predicting_backend(100),
            sections=(SECTION_GAME_BEHAVIOR, SECTION_SELF_REPORT),
            config=EvalConfig(
                survey_samples=1,
                batch_size=64,
                games=("twin-pd",),
                include_never_trained=False,
            ),
        )
        gaps = summary[SECTION_SELF_REPORT]["calibration_gaps"]
        twin = gaps["twin-pd"]
        assert twin["predicted"] == pytest.approx(1.0)
        assert twin["measured"] is not None
        assert twin["gap"] == pytest.approx(1.0 - twin["measured"])
        assert twin["n_measured_records"] > 0

    def test_the_gap_does_not_depend_on_the_order_the_sections_were_listed_in(
        self, tmp_path: Path
    ) -> None:
        """Summarising in flight made this depend on argument order; it is a battery, not a stream."""
        config = EvalConfig(
            survey_samples=1, batch_size=64, games=("twin-pd",), include_never_trained=False
        )
        behaviour_first, _, _ = run_section(
            tmp_path,
            backend=self._predicting_backend(70),
            sections=(SECTION_GAME_BEHAVIOR, SECTION_SELF_REPORT),
            config=config,
            name="behaviour-first.jsonl",
        )
        survey_first, _, _ = run_section(
            tmp_path,
            backend=self._predicting_backend(70),
            sections=(SECTION_SELF_REPORT, SECTION_GAME_BEHAVIOR),
            config=config,
            name="survey-first.jsonl",
        )
        assert (
            behaviour_first[SECTION_SELF_REPORT]["calibration_gaps"]
            == survey_first[SECTION_SELF_REPORT]["calibration_gaps"]
        )

    def test_without_the_behaviour_section_the_prediction_keeps_a_missing_measured_side(
        self, tmp_path: Path
    ) -> None:
        """A coverage fact about the run, not a reason to drop the prediction on the floor."""
        summary, _, _ = run_section(tmp_path, backend=self._predicting_backend(40))
        gaps = summary[SECTION_SELF_REPORT]["calibration_gaps"]
        assert gaps
        for reading in gaps.values():
            assert reading["measured"] is None
            assert reading["gap"] is None
            assert reading["n_measured_records"] == 0

    def test_every_self_prediction_item_is_asked_as_a_numeric_item(self, tmp_path: Path) -> None:
        battery = survey_battery(data_dir=synthetic_survey_dir(tmp_path / "survey"))
        predictions = [item for item in battery if item.family == FAMILY_SELF_PREDICTION]
        assert predictions
        assert all(item.kind == SURVEY_NUMERIC for item in predictions)


class TestTheConfigRefusesWhatWouldMisfireLater:
    def test_a_typoed_instrument_is_refused_where_the_config_is_built(self) -> None:
        with pytest.raises(ValueError, match="Unknown survey_instruments"):
            EvalConfig(survey_instruments=("prosocialnes",))

    def test_a_typoed_family_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown survey_families"):
            EvalConfig(survey_families=("negative-controls",))

    def test_a_repeated_instrument_is_refused(self) -> None:
        """It would ask those items twice under one sample index and double-weight the composite."""
        with pytest.raises(ValueError, match="more than once"):
            EvalConfig(survey_instruments=(INSTRUMENT_SVO_SLIDER, INSTRUMENT_SVO_SLIDER))

    def test_zero_samples_is_refused(self) -> None:
        with pytest.raises(ValueError, match="survey_samples must be at least 1"):
            EvalConfig(survey_samples=0)

    def test_a_typoed_tier_is_refused_where_the_config_is_built(self) -> None:
        with pytest.raises(ValueError, match="Unknown survey_tier"):
            EvalConfig(survey_tier="cor")

    def test_a_core_tier_config_narrows_the_trace_to_core_items(self, tmp_path: Path) -> None:
        """The deliberated leg runs tier=core, so breadth items never bill thinking-on renders."""
        data_dir = synthetic_survey_dir(tmp_path / "survey")
        everything = survey_battery(data_dir=data_dir)
        core_ids = {item.item_id for item in everything if item.tier == TIER_CORE}
        assert core_ids
        assert len(core_ids) < len(everything)
        _, records, _ = run_section(
            tmp_path,
            config=EvalConfig(
                survey_samples=1, batch_size=64, survey_data_dir=data_dir, survey_tier=TIER_CORE
            ),
        )
        asked = {record["item_id"] for record in survey_records(records)}
        assert asked == core_ids

    def test_every_survey_knob_lands_in_the_meta(self, tmp_path: Path) -> None:
        """A knob absent from the meta is a knob a later reader cannot attribute a number to."""
        data_dir = synthetic_survey_dir(
            tmp_path / "survey", instruments=(INSTRUMENT_TRIPLE_DOMINANCE,)
        )
        config = EvalConfig(
            survey_samples=2,
            batch_size=32,
            survey_data_dir=data_dir,
            survey_instruments=(INSTRUMENT_TRIPLE_DOMINANCE,),
            survey_families=("triple-dominance-allocation",),
            survey_tier=TIER_CORE,
        )
        record = config.as_record()
        assert record["survey_samples"] == 2
        assert record["survey_data_dir"] == str(data_dir)
        assert record["survey_instruments"] == [INSTRUMENT_TRIPLE_DOMINANCE]
        assert record["survey_families"] == ["triple-dominance-allocation"]
        assert record["survey_tier"] == TIER_CORE

    def test_an_empty_instrument_list_records_every_registered_instrument(self) -> None:
        assert EvalConfig().as_record()["survey_instruments"] == sorted(PUBLISHED_INSTRUMENTS)

    def test_an_empty_tier_records_both_tiers(self) -> None:
        assert EvalConfig().as_record()["survey_tier"] == ["core", "breadth"]


class TestTheReadoutTables:
    @pytest.fixture
    def trace(self, tmp_path: Path) -> Any:
        """One self-report cell plus its behaviour section, loaded the way the readout loads it."""
        data_dir = synthetic_survey_dir(tmp_path / "survey", instruments=SHAPE_COVERING_INSTRUMENTS)
        _, _, path = run_section(
            tmp_path,
            backend=MockBackend(responses=[ALWAYS_A], model_id="always-a"),
            sections=(SECTION_GAME_BEHAVIOR, SECTION_SELF_REPORT),
            config=EvalConfig(
                survey_samples=1,
                batch_size=64,
                survey_data_dir=data_dir,
                survey_instruments=SHAPE_COVERING_INSTRUMENTS,
                games=("twin-pd",),
                include_never_trained=False,
            ),
        )
        return load_traces([path])

    def test_the_composite_ladder_has_a_row_per_instrument_and_subscale(self, trace: Any) -> None:
        rows = battery_tables.self_report_ladder_rows(trace)
        assert rows
        keys = {(row["instrument"], row["subscale"]) for row in rows}
        assert len(keys) == len(rows)
        assert any(row["instrument"] == INSTRUMENT_SVO_SLIDER for row in rows)

    def test_every_composite_cell_carries_its_item_denominator(self, trace: Any) -> None:
        """A composite whose denominator moved would otherwise read as the composite moving."""
        rows = battery_tables.self_report_ladder_rows(trace)
        assert all("items)" in row["composite"] for row in rows)

    def test_the_control_table_names_the_keying_counts_behind_each_acquiescence_reading(
        self, trace: Any
    ) -> None:
        rows = battery_tables.self_report_control_rows(trace)
        assert rows
        assert all("/" in row["keyed positive/reverse"] for row in rows)

    def test_a_single_keyed_subscale_prints_its_reason_rather_than_a_number(
        self, trace: Any
    ) -> None:
        rows = battery_tables.self_report_control_rows(trace)
        reasons = [row["acquiescence"] for row in rows if row["acquiescence"].startswith("-")]
        assert reasons
        assert all("(" in cell for cell in reasons)

    def test_the_numeric_family_reports_no_comparable_order_pairs(self, trace: Any) -> None:
        """Numeric records stay out of the LETTER order-pair counts, even under two wordings.

        Since the swapped-stem counterbalance landed, a self-prediction item renders under two
        order names per sample -- but its answers carry no canonical option index, so counting its
        pairs would put them in a disagreement denominator whose numerator can never see them.
        `_order_pairs` skips numeric records explicitly; this is the test that goes red if that
        skip is removed, and the numeric family's own order readout is the numeric table's gap.
        """
        rows = battery_tables.self_report_health_rows(trace)
        numeric_rows = [row for row in rows if row["family"] == FAMILY_SELF_PREDICTION]
        assert numeric_rows
        assert all(row[battery_tables.ORDER_COLUMN].endswith("(0/0 pairs)") for row in numeric_rows)

    def test_a_lettered_family_does_report_comparable_order_pairs(self, trace: Any) -> None:
        """The other half of the pair above: a real zero and a not-applicable must not look alike."""
        rows = battery_tables.self_report_health_rows(trace)
        lettered = [row for row in rows if row["family"] != FAMILY_SELF_PREDICTION]
        assert lettered
        assert any(not row[battery_tables.ORDER_COLUMN].endswith("(0/0 pairs)") for row in lettered)

    def test_the_health_table_reports_a_parse_rate_per_family(self, trace: Any) -> None:
        rows = battery_tables.self_report_health_rows(trace)
        assert rows
        assert {row["family"] for row in rows} <= set(FAMILIES)
        assert all("/" in row["parsed/asked"] for row in rows)

    def test_the_instrument_table_carries_the_angle_and_the_orientation_counts(
        self, trace: Any
    ) -> None:
        rows = battery_tables.self_report_instrument_rows(trace)
        assert len(rows) == 1
        assert rows[0]["svo_angle_deg"] != battery_tables.EMPTY_CELL
        assert rows[0]["orientations chosen"] != battery_tables.EMPTY_CELL

    def test_the_calibration_table_names_both_denominators(self, trace: Any) -> None:
        rows = battery_tables.self_report_calibration_rows(trace)
        assert rows
        assert all("/" in row["predictions/behaviour records"] for row in rows)

    def test_the_numeric_table_reports_every_numeric_item_with_its_denominators(
        self, trace: Any
    ) -> None:
        """One row per numeric item; an all-unparsed item still holds its row open."""
        rows = battery_tables.self_report_numeric_rows(trace)
        items = {row["item"] for row in rows}
        assert "negative-control-bullet-rate" in items
        assert sum(1 for item in items if item.startswith("self-prediction-")) == 8
        # The always-A backend never answers a keep tag, so the denominators carry the story:
        # two renders asked per swapped item, one for the placebo, nothing parsed.
        by_item = {row["item"]: row for row in rows}
        assert by_item["negative-control-bullet-rate"]["parsed/asked"] == "0/1"
        assert by_item["self-prediction-twin-pd"]["parsed/asked"] == "0/2"

    def test_the_control_distribution_table_reads_movement_against_step_zero(
        self, trace: Any
    ) -> None:
        """With one trace the baseline is itself, so every TV distance is exactly zero."""
        rows = battery_tables.self_report_control_distribution_rows(trace)
        assert rows
        assert all(row["TV vs step0"] == "0.000" for row in rows)
        assert all(":" in row["distribution"] for row in rows)
        assert {row["subscale"] for row in rows} >= {"inert-preference"}

    def test_no_table_leaks_a_completion_into_a_cell(self, trace: Any) -> None:
        """A readout is numbers; generated text in a table is how an eval trace gets republished."""
        builders = (
            battery_tables.self_report_ladder_rows,
            battery_tables.self_report_control_rows,
            battery_tables.self_report_health_rows,
            battery_tables.self_report_instrument_rows,
            battery_tables.self_report_control_distribution_rows,
            battery_tables.self_report_numeric_rows,
            battery_tables.self_report_calibration_rows,
        )
        rendered = " ".join(
            str(value) for builder in builders for row in builder(trace) for value in row.values()
        )
        assert "FINAL ANSWER" not in rendered
        assert "reasoning" not in rendered

    def test_a_trace_with_no_self_report_section_yields_empty_tables(self, tmp_path: Path) -> None:
        _, _, path = run_section(
            tmp_path,
            sections=(SECTION_GAME_BEHAVIOR,),
            config=EvalConfig(
                survey_samples=1, batch_size=16, games=("twin-pd",), include_never_trained=False
            ),
        )
        traces = load_traces([path])
        assert battery_tables.self_report_ladder_rows(traces) == []
        assert battery_tables.self_report_control_rows(traces) == []
        assert battery_tables.self_report_health_rows(traces) == []
        assert battery_tables.self_report_instrument_rows(traces) == []
        assert battery_tables.self_report_control_distribution_rows(traces) == []
        assert battery_tables.self_report_numeric_rows(traces) == []
        assert battery_tables.self_report_calibration_rows(traces) == []


# The twinned instrument the wording-scope tests below administer: nine as-published items in one
# subscale, five positively keyed and four reverse-keyed, which is the shape that first printed a
# doubled item base.
TWINNED_SUBSCALE = "enjoyment-of-competition"


class TestTheCompositeLadderStaysOnOneWording:
    """Every column of a ladder row is about the as-published instrument, denominators included.

    `subscale_composites` averages the as-published items alone, on purpose: a lexically neutral
    twin is our rewording rather than the validated instrument. So a denominator counted over both
    wordings states an item base twice the one the composite rests on wherever a subscale twins
    every item, and a reader has no way to see that from the cell.
    """

    @pytest.fixture
    def twinned_cell(self, tmp_path: Path) -> tuple[list[dict[str, Any]], Any]:
        """One cell of a fully twinned Likert instrument, answered per wording, with its trace."""
        data_dir = synthetic_survey_dir(
            tmp_path / "survey", instruments=(INSTRUMENT_COMPETITIVENESS_INDEX,), twinned=True
        )
        _, records, path = run_section(
            tmp_path,
            backend=MockBackend(
                responses=answer_the_anchor_the_wording_calls_for, model_id="anchor-by-wording"
            ),
            config=EvalConfig(
                survey_samples=1,
                batch_size=64,
                survey_data_dir=data_dir,
                survey_instruments=(INSTRUMENT_COMPETITIVENESS_INDEX,),
                survey_families=(FAMILY_COMPETITIVENESS,),
            ),
        )
        return survey_records(records), load_traces([path])

    def test_the_cell_answered_the_two_wordings_differently(
        self, twinned_cell: tuple[list[dict[str, Any]], Any]
    ) -> None:
        """The floor the assertions below rest on: a twin arm that does not move the pooled mean
        would let a pooling regression pass every one of them."""
        records, _ = twinned_cell
        key = (INSTRUMENT_COMPETITIVENESS_INDEX, TWINNED_SUBSCALE)
        assert self.item_ids(records, wording=WORDING_NEUTRAL_TWIN)
        assert subscale_composites(records)[key] != pytest.approx(
            subscale_composites(records, wording=None)[key]
        )

    def test_the_composite_reads_against_the_items_it_averages(
        self, twinned_cell: tuple[list[dict[str, Any]], Any]
    ) -> None:
        records, traces = twinned_cell
        published = self.item_ids(records, wording=WORDING_AS_PUBLISHED)
        twins = self.item_ids(records, wording=WORDING_NEUTRAL_TWIN)
        assert len(twins) == len(published)
        composite = subscale_composites(records)[
            (INSTRUMENT_COMPETITIVENESS_INDEX, TWINNED_SUBSCALE)
        ]
        row = self.ladder_row(traces)
        assert row["composite"] == f"{composite:.3f} ({len(published)}/{len(published)} items)"

    def test_the_keying_count_is_of_the_published_items_too(
        self, twinned_cell: tuple[list[dict[str, Any]], Any]
    ) -> None:
        """A twin inherits its parent's keying, so a both-wordings count doubles this one as well."""
        records, traces = twinned_cell
        reverse_keyed = {
            item_id
            for item_id in self.item_ids(records, wording=WORDING_AS_PUBLISHED)
            if any(
                record.get("reverse_keyed")
                for record in records
                if str(record["item_id"]) == item_id
            )
        }
        assert reverse_keyed
        assert self.ladder_row(traces)["n_reverse_keyed"] == len(reverse_keyed)

    def test_the_row_truncation_figure_stays_on_the_published_renders(
        self, twinned_cell: tuple[list[dict[str, Any]], Any]
    ) -> None:
        """One scope per row, so the health figure here is of the renders the composite came from.

        The wording-blind truncation count is `self_report_health_rows`' job, per family, so nothing
        is dropped by narrowing this cell -- and a row that mixed the two scopes would attribute a
        twin's truncated render to the published items.
        """
        records, traces = twinned_cell
        published_renders = [
            record
            for record in self.subscale_records(records)
            if str(record["wording"]) == WORDING_AS_PUBLISHED
        ]
        assert len(published_renders) < len(self.subscale_records(records))
        assert self.ladder_row(traces)["truncated/asked"] == f"0/{len(published_renders)}"

    @staticmethod
    def subscale_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Every render of the twinned subscale, both wordings."""
        return [record for record in records if str(record["subscale"]) == TWINNED_SUBSCALE]

    @classmethod
    def item_ids(cls, records: list[dict[str, Any]], *, wording: str) -> set[str]:
        """The twinned subscale's distinct item ids under one wording."""
        return {
            str(record["item_id"])
            for record in cls.subscale_records(records)
            if str(record["wording"]) == wording
        }

    @staticmethod
    def ladder_row(traces: Any) -> dict[str, Any]:
        """The one ladder row for the twinned subscale."""
        rows = [
            row
            for row in battery_tables.self_report_ladder_rows(traces)
            if row["subscale"] == TWINNED_SUBSCALE
        ]
        assert len(rows) == 1
        return rows[0]
