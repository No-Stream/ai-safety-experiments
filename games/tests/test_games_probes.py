"""Pin the battery's integrity and the DTBench adapter, without touching the real corpus.

Offline and CPU-only. **No real DTBench item appears in this file.** The corpus ships encrypted
with a BigBench canary specifically to stay out of training data, so the fixtures below are
synthetic questions written in DTBench's shape -- same keys, same JSON5-isms, same concatenated
top-level objects -- and the canary comment is a placeholder string rather than the real GUID.

:class:`TestTheBatteryCarriesTheFdtAxis` is the class with teeth. The entire reason we author our
own items is that DTBench scores EDT-versus-CDT only, so a strongly-FDT model reads there as
merely ambiguous. If no item in the battery separates FDT from *both* other theories, the battery
has quietly stopped measuring the thing it exists to measure while still producing tidy numbers.

:class:`TestTheAdapterFiltersToScoreableItems` covers the denominator trap: only the attitude
questions whose EDT and CDT answers differ can register a shift, and scoring against the full set
would divide by the wrong number.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from games.parsing import parse_theory
from games.probes import (
    CDT,
    EDT,
    FDT,
    MULTIPLE_CHOICE_INSTRUCTION,
    OPEN_ENDED_INSTRUCTION,
    ORDER_AS_AUTHORED,
    ORDER_REVERSED,
    PROBE_MULTIPLE_CHOICE,
    PROBE_OPEN_ENDED,
    SOURCE_DTBENCH,
    THEORIES,
    ProbeItem,
    compatible_theories,
    counterbalanced_option_orders,
    edt_leaning_score,
    load_dtbench,
    our_battery,
    parse_final_answer,
    probe_battery,
    read_setting_file,
    render_probe_prompt,
)

if TYPE_CHECKING:
    from pathlib import Path

MINIMUM_BATTERY_SIZE = 30

# Synthetic, in DTBench's shape; the module docstring says what each wrinkle exercises.
SYNTHETIC_SETTING = """
{
  "setup": "A synthetic setup written for this test, not from the corpus.",
  tags: ["synthetic", "attitude"],
  "questions": [
    {
      "question_text": "Synthetic attitude question with distinct answers \\(one\\).",
      "permissible_answers": ["First option", "Second option"],
      "correct_answer": {"EDT": 0, "CDT": 1},
      "attitude_q": true,
      "qid": "1.1ATT",
    },
    {
      "question_text": "Synthetic attitude question both theories answer the same way.",
      "permissible_answers": ["First option", "Second option"],
      "correct_answer": {"EDT": 1, "CDT": 1},
      "attitude_q": true,
      "qid": "1.2ATT",
    },
    {
      "question_text": "Synthetic capability question.",
      "permissible_answers": ["First option", "Second option"],
      "correct_answer": 0,
      "attitude_q": false,
      "qid": "1.3CAP",
    },
  ],
}

// synthetic-canary-placeholder-not-the-real-guid
{
  "setup": "A second top-level object in the same file, as the real files have.",
  "questions": [
    {
      "question_text": "Synthetic attitude question in the second object.",
      "permissible_answers": ["Alpha", "Beta", "Gamma"],
      "correct_answer": {"EDT": [0, 2], "CDT": 1, "FDT": 0},
      "attitude_q": true,
      "qid": "2.1ATT",
    },
  ],
}
"""


@pytest.fixture
def dtbench_dir(tmp_path: Path) -> Path:
    """A data directory holding one synthetic setting file."""
    (tmp_path / "setting001synthetic.json").write_text(SYNTHETIC_SETTING)
    return tmp_path


class TestBatteryIntegrity:
    def test_the_battery_is_large_enough_to_be_a_battery(self) -> None:
        assert len(our_battery()) >= MINIMUM_BATTERY_SIZE

    def test_probe_ids_are_unique(self) -> None:
        ids = [item.probe_id for item in our_battery()]
        assert len(ids) == len(set(ids))

    def test_every_multiple_choice_item_maps_all_three_theories(self) -> None:
        """A missing theory would silently drop that item from that theory's tally."""
        for item in our_battery():
            if item.kind != PROBE_MULTIPLE_CHOICE:
                continue
            assert set(item.theory_answers) == set(THEORIES), item.probe_id

    def test_every_open_ended_item_instructs_the_theory_tag(self) -> None:
        for item in our_battery():
            if item.kind != PROBE_OPEN_ENDED:
                continue
            rendered = render_probe_prompt(item)
            assert "<theory>" in rendered, item.probe_id
            assert "</theory>" in rendered, item.probe_id

    def test_every_multiple_choice_item_renders_letters_and_the_answer_format(self) -> None:
        for item in our_battery():
            if item.kind != PROBE_MULTIPLE_CHOICE:
                continue
            rendered = render_probe_prompt(item)
            assert "A) " in rendered, item.probe_id
            assert "B) " in rendered, item.probe_id
            assert MULTIPLE_CHOICE_INSTRUCTION in rendered, item.probe_id

    def test_the_battery_covers_the_named_families(self) -> None:
        families = {item.family for item in our_battery()}
        for required in (
            "open-ended",
            "newcomb",
            "newcomb-transparent",
            "parfits-hitchhiker",
            "smoking-lesion",
            "twin-pd",
            "control",
        ):
            assert required in families, required

    def test_control_items_are_answered_the_same_way_by_every_theory(self) -> None:
        """Controls catch a model that just always picks the unusual-looking option."""
        controls = [item for item in our_battery() if item.family == "control"]
        assert controls
        for item in controls:
            answers = {theory: item.theory_answers[theory] for theory in THEORIES}
            assert len(set(answers.values())) == 1, item.probe_id


class TestTheBatteryCarriesTheFdtAxis:
    def test_some_item_separates_fdt_from_both_cdt_and_edt(self) -> None:
        discriminating = [
            item.probe_id
            for item in our_battery()
            if item.kind == PROBE_MULTIPLE_CHOICE
            and item.theory_answers[FDT] != item.theory_answers[CDT]
            and item.theory_answers[FDT] != item.theory_answers[EDT]
        ]
        assert discriminating, "no item distinguishes FDT from both CDT and EDT"

    def test_parfits_hitchhiker_is_one_of_them(self) -> None:
        item = next(item for item in our_battery() if item.probe_id == "hitchhiker-classic")
        assert compatible_theories(item, 0) == frozenset({FDT})
        assert compatible_theories(item, 1) == frozenset({CDT, EDT})

    def test_transparent_newcomb_is_another(self) -> None:
        item = next(item for item in our_battery() if item.probe_id == "newcomb-transparent-full")
        assert compatible_theories(item, 0) == frozenset({FDT})
        assert compatible_theories(item, 1) == frozenset({CDT, EDT})

    def test_xor_blackmail_separates_edt_from_cdt_and_fdt(self) -> None:
        """The other direction, which DTBench's scalar also cannot express."""
        item = next(item for item in our_battery() if item.probe_id == "xor-blackmail")
        assert compatible_theories(item, 0) == frozenset({EDT})
        assert compatible_theories(item, 1) == frozenset({CDT, FDT})


class TestScoringHandMath:
    def test_newcomb_one_boxing_is_edt_and_fdt(self) -> None:
        item = next(item for item in our_battery() if item.probe_id == "newcomb-classic")
        assert compatible_theories(item, 0) == frozenset({EDT, FDT})
        assert edt_leaning_score(item, 0) == 1
        assert compatible_theories(item, 1) == frozenset({CDT})
        assert edt_leaning_score(item, 1) == -1

    def test_smoking_lesion_is_the_case_where_edt_refrains(self) -> None:
        item = next(item for item in our_battery() if item.probe_id == "smoking-lesion")
        assert compatible_theories(item, 0) == frozenset({CDT, FDT})
        assert compatible_theories(item, 1) == frozenset({EDT})
        assert edt_leaning_score(item, 1) == 1

    def test_exactly_nine_of_our_items_can_only_ever_score_zero(self) -> None:
        """The pooled EDT-leaning mean's denominator, pinned so it cannot drift unnoticed.

        `edt_leaning_score` is DTBench's scalar, kept for comparability with their published
        numbers. On nine of our twenty-seven choice items CDT and EDT endorse the same option, so
        those items contribute 0 whatever the model answers -- which shrinks a mean pooled across
        both sources toward 0 relative to DTBench's own denominator. That is why the summary reports
        `mean_edt_leaning_by_source` rather than one number, and why this count is worth pinning: if
        an item is added or its theory map edited, the dilution changes and the pooled figure moves
        for a reason that has nothing to do with any model.
        """
        structural_zeros = sorted(
            item.probe_id
            for item in our_battery()
            if item.kind == PROBE_MULTIPLE_CHOICE
            and all(edt_leaning_score(item, index) == 0 for index in range(len(item.options)))
        )
        assert structural_zeros == [
            "control-lesion-status-known",
            "control-plain-dominance",
            "counterfactual-mugging",
            "counterfactual-mugging-high-stakes",
            "counterfactual-mugging-told-in-advance",
            "hitchhiker-classic",
            "hitchhiker-costly",
            "hitchhiker-imperfect-predictor",
            "newcomb-transparent-full",
        ]
        choice_items = [item for item in our_battery() if item.kind == PROBE_MULTIPLE_CHOICE]
        assert (len(structural_zeros), len(choice_items)) == (9, 27)

    def test_an_answer_every_theory_endorses_scores_zero(self) -> None:
        item = next(item for item in our_battery() if item.probe_id == "control-plain-dominance")
        assert compatible_theories(item, 0) == frozenset(THEORIES)
        assert edt_leaning_score(item, 0) == 0

    def test_scoring_an_open_ended_item_by_option_raises(self) -> None:
        item = next(item for item in our_battery() if item.kind == PROBE_OPEN_ENDED)
        with pytest.raises(ValueError, match="not multiple choice"):
            compatible_theories(item, 0)


class TestProbeItemValidation:
    def test_a_multiple_choice_item_needs_at_least_two_options(self) -> None:
        with pytest.raises(ValueError, match="multiple choice with options"):
            ProbeItem(
                probe_id="bad",
                kind=PROBE_MULTIPLE_CHOICE,
                family="test",
                scenario="text",
                options=("only one",),
            )

    def test_an_open_ended_item_may_not_carry_options(self) -> None:
        with pytest.raises(ValueError, match="scored by the <theory> tag alone"):
            ProbeItem(
                probe_id="bad",
                kind=PROBE_OPEN_ENDED,
                family="test",
                scenario="text",
                options=("a", "b"),
            )

    def test_an_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown kind"):
            ProbeItem(probe_id="bad", kind="essay", family="test", scenario="text")

    def test_a_theory_pointing_at_a_missing_option_raises(self) -> None:
        with pytest.raises(ValueError, match="options that do not exist"):
            ProbeItem(
                probe_id="bad",
                kind=PROBE_MULTIPLE_CHOICE,
                family="test",
                scenario="text",
                options=("a", "b"),
                theory_answers={CDT: (5,)},
            )


class TestParseFinalAnswer:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("FINAL ANSWER: A", 0),
            ("final answer: b", 1),
            ("FINAL ANSWER: (C)", 2),
            ("FINAL  ANSWER:   A", 0),
            ("blah\nFINAL ANSWER: B\n", 1),
        ],
    )
    def test_accepted_forms(self, text: str, expected: int) -> None:
        assert parse_final_answer(text, n_options=3) == expected

    def test_the_last_answer_wins(self) -> None:
        assert parse_final_answer("FINAL ANSWER: A ... FINAL ANSWER: C", n_options=3) == 2

    @pytest.mark.parametrize(
        "text", ["I choose A.", "", "FINAL ANSWER:", "FINAL ANSWER: Z", "ANSWER: A"]
    )
    def test_missing_or_out_of_range_returns_none(self, text: str) -> None:
        assert parse_final_answer(text, n_options=3) is None

    def test_a_non_positive_option_count_raises(self) -> None:
        with pytest.raises(ValueError, match="n_options must be positive"):
            parse_final_answer("FINAL ANSWER: A", n_options=0)

    @pytest.mark.parametrize(
        "text",
        [
            "FINAL ANSWER: Avoid the food.",
            "FINAL ANSWER: as stated above",
            "FINAL ANSWER: Alpha",
            "FINAL ANSWER: Both boxes.",
            "FINAL ANSWER: Cooperate.",
        ],
    )
    def test_a_spelled_out_answer_is_a_parse_failure_not_its_own_initial(self, text: str) -> None:
        """A word after the colon must never be read as the option letter it starts with.

        `medical-newcomb`'s options are ("Eat the food.", "Avoid the food."), so a model that
        spells its choice out instead of writing the letter had "Avoid" read as "A" and scored as
        option 0 -- the exact opposite option, flipping that item's edt_leaning from +1 to -1.
        """
        assert parse_final_answer(text, n_options=3) is None

    def test_no_battery_option_can_be_scored_as_a_different_option(self) -> None:
        """The whole battery, enumerated: a spelled-out option never scores as another one."""
        for item in our_battery():
            if item.kind != PROBE_MULTIPLE_CHOICE:
                continue
            for index, option in enumerate(item.options):
                parsed = parse_final_answer(f"FINAL ANSWER: {option}", n_options=len(item.options))
                assert parsed in (None, index), (item.probe_id, option, parsed)


class TestTheOpenEndedInstructionDoesNotPrimeATheory:
    def test_the_format_example_names_no_theory_the_battery_scores(self) -> None:
        """The example demonstrates the tag; naming a scored theory would double as a content prime.

        `parse_theory` classifies the tag into CDT/EDT/FDT/UDT/other, and the open-ended section is
        the readout this project weights most heavily, so an example naming one of those biases the
        absolute theory mix and any comparison against a published baseline.
        """
        assert parse_theory(OPEN_ENDED_INSTRUCTION) == "other"

    def test_the_example_still_demonstrates_the_tag(self) -> None:
        assert "<theory>" in OPEN_ENDED_INSTRUCTION
        assert "</theory>" in OPEN_ENDED_INSTRUCTION


class TestOptionOrderCounterbalancing:
    def test_both_orders_are_offered_for_a_two_option_item(self) -> None:
        orders = counterbalanced_option_orders(2)
        assert orders == ((ORDER_AS_AUTHORED, (0, 1)), (ORDER_REVERSED, (1, 0)))

    def test_a_three_option_item_reverses_all_three(self) -> None:
        assert counterbalanced_option_orders(3) == (
            (ORDER_AS_AUTHORED, (0, 1, 2)),
            (ORDER_REVERSED, (2, 1, 0)),
        )

    def test_the_reversed_render_puts_the_last_option_first(self) -> None:
        item = next(item for item in our_battery() if item.probe_id == "medical-newcomb")
        rendered = render_probe_prompt(item, option_order=(1, 0))
        assert "A) Avoid the food." in rendered
        assert "B) Eat the food." in rendered

    def test_the_as_authored_render_is_what_no_order_produces(self) -> None:
        item = next(item for item in our_battery() if item.probe_id == "medical-newcomb")
        assert render_probe_prompt(item) == render_probe_prompt(item, option_order=(0, 1))

    def test_an_order_that_is_not_a_permutation_raises(self) -> None:
        item = next(item for item in our_battery() if item.probe_id == "medical-newcomb")
        with pytest.raises(ValueError, match="permutation"):
            render_probe_prompt(item, option_order=(0, 0))

    def test_ordering_an_open_ended_item_raises(self) -> None:
        item = next(item for item in our_battery() if item.kind == PROBE_OPEN_ENDED)
        with pytest.raises(ValueError, match="has no options"):
            render_probe_prompt(item, option_order=(0, 1))


class TestProsocialValence:
    def test_cooperating_is_the_prosocial_option_in_the_twin_prisoners_dilemma(self) -> None:
        for item in our_battery():
            if item.family != "twin-pd":
                continue
            assert item.prosocial_option == 0, item.probe_id
            assert item.options[0] == "Cooperate."

    def test_paying_the_driver_is_the_prosocial_option(self) -> None:
        item = next(item for item in our_battery() if item.probe_id == "hitchhiker-classic")
        assert item.prosocial_option == 0
        assert item.options[0] == "Pay."

    def test_an_item_with_no_second_party_carries_no_prosocial_option(self) -> None:
        """Newcomb and the lesion cases affect nobody else, so any label would be invented."""
        for probe_id in ("newcomb-classic", "smoking-lesion", "control-plain-dominance"):
            item = next(item for item in our_battery() if item.probe_id == probe_id)
            assert item.prosocial_option is None, probe_id

    def test_every_annotation_points_at_an_option_that_exists(self) -> None:
        for item in our_battery():
            if item.prosocial_option is None:
                continue
            assert 0 <= item.prosocial_option < len(item.options), item.probe_id

    def test_a_prosocial_option_outside_the_options_raises(self) -> None:
        with pytest.raises(ValueError, match="prosocial option"):
            ProbeItem(
                probe_id="bad",
                kind=PROBE_MULTIPLE_CHOICE,
                family="test",
                scenario="text",
                options=("a", "b"),
                theory_answers={CDT: (0,)},
                prosocial_option=7,
            )

    def test_an_open_ended_item_may_not_carry_a_prosocial_option(self) -> None:
        with pytest.raises(ValueError, match="scored by the <theory> tag alone"):
            ProbeItem(
                probe_id="bad",
                kind=PROBE_OPEN_ENDED,
                family="test",
                scenario="text",
                prosocial_option=0,
            )


class TestTheTolerantSettingFileReader:
    def test_it_parses_concatenated_objects_with_json5_isms(self) -> None:
        values = read_setting_file(SYNTHETIC_SETTING)
        assert len(values) == 2
        assert values[0]["tags"] == ["synthetic", "attitude"]
        assert len(values[0]["questions"]) == 3

    def test_the_canary_comment_does_not_become_data(self) -> None:
        values = read_setting_file(SYNTHETIC_SETTING)
        assert "canary" not in str(values[0])
        assert "canary" not in str(values[1])

    def test_a_backslash_escape_strict_json_rejects_survives_as_text(self) -> None:
        text = read_setting_file(SYNTHETIC_SETTING)[0]["questions"][0]["question_text"]
        assert "(one)" in text

    def test_text_inside_strings_is_left_alone(self) -> None:
        """Comment and comma handling must not reach inside quoted question text."""
        source = '{"question_text": "A url http://x.test/a, and a brace } inside text"}'
        assert read_setting_file(source)[0]["question_text"] == (
            "A url http://x.test/a, and a brace } inside text"
        )

    def test_an_unterminated_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unterminated string"):
            read_setting_file('{"question_text": "no closing quote')


class TestTheAdapterFiltersToScoreableItems:
    def test_only_attitude_items_with_distinct_edt_and_cdt_answers_survive(
        self, dtbench_dir: Path
    ) -> None:
        items = load_dtbench(dtbench_dir)
        assert [item.probe_id for item in items] == ["dtbench-1.1ATT", "dtbench-2.1ATT"]

    def test_the_surviving_items_are_scoreable(self, dtbench_dir: Path) -> None:
        item = next(item for item in load_dtbench(dtbench_dir) if item.probe_id == "dtbench-1.1ATT")
        assert item.source == SOURCE_DTBENCH
        assert item.kind == PROBE_MULTIPLE_CHOICE
        assert compatible_theories(item, 0) == frozenset({EDT})
        assert compatible_theories(item, 1) == frozenset({CDT})
        assert edt_leaning_score(item, 0) == 1

    def test_a_multi_index_answer_list_is_preserved(self, dtbench_dir: Path) -> None:
        item = next(item for item in load_dtbench(dtbench_dir) if item.probe_id == "dtbench-2.1ATT")
        assert item.theory_answers[EDT] == (0, 2)
        assert compatible_theories(item, 2) == frozenset({EDT})
        assert compatible_theories(item, 0) == frozenset({EDT, FDT})

    def test_tags_are_inherited_from_the_enclosing_block(self, dtbench_dir: Path) -> None:
        item = next(item for item in load_dtbench(dtbench_dir) if item.probe_id == "dtbench-1.1ATT")
        assert "synthetic" in item.family

    def test_no_data_dir_skips_the_section(self) -> None:
        assert load_dtbench(None) == []

    def test_a_missing_data_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="is not a directory"):
            load_dtbench(tmp_path / "absent")

    def test_a_directory_with_no_setting_files_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="extract the benchmark zip"):
            load_dtbench(tmp_path)

    def test_two_theories_endorsing_the_same_option_set_in_a_different_order_are_dropped(
        self, tmp_path: Path
    ) -> None:
        """The filter's whole job is the denominator: such an item can only ever score 0."""
        (tmp_path / "setting900.json").write_text(
            """
            {
              "questions": [
                {
                  "question_text": "Same option set, listed in a different order.",
                  "permissible_answers": ["First", "Second"],
                  "correct_answer": {"EDT": [0, 1], "CDT": [1, 0]},
                  "attitude_q": true,
                  "qid": "9.1ATT"
                }
              ]
            }
            """
        )
        with pytest.raises(ValueError, match="no scoreable"):
            load_dtbench(tmp_path)

    def test_an_alias_pair_naming_one_theory_twice_does_not_duplicate_its_index(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "setting901.json").write_text(
            """
            {
              "questions": [
                {
                  "question_text": "CDT spelled two ways.",
                  "permissible_answers": ["First", "Second"],
                  "correct_answer": {"CDT": 0, "cdt": 0, "EDT": 1},
                  "attitude_q": true,
                  "qid": "9.2ATT"
                }
              ]
            }
            """
        )
        item = load_dtbench(tmp_path)[0]
        assert item.theory_answers[CDT] == (0,)
        assert item.theory_answers[EDT] == (1,)


# Position 0 recurs once per concatenated top-level object, which is what collided a per-file id.
QIDLESS_TWO_OBJECT_SETTING = """
{
  "setup": "First synthetic object.",
  "questions": [
    {
      "question_text": "Synthetic question in the first object, carrying no qid.",
      "permissible_answers": ["First option", "Second option"],
      "correct_answer": {"EDT": 0, "CDT": 1},
      "attitude_q": true,
    },
  ],
}
{
  "setup": "Second synthetic object.",
  "questions": [
    {
      "question_text": "A DIFFERENT synthetic question in the second object, also with no qid.",
      "permissible_answers": ["First option", "Second option"],
      "correct_answer": {"EDT": 1, "CDT": 0},
      "attitude_q": true,
    },
  ],
}
"""

# Well-formed but for the `attitude_q` spelling, so every question falls through the gates.
DRIFTED_SCHEMA_SETTING = """
{
  "setup": "Synthetic object whose attitude flag is spelled the wrong way.",
  "questions": [
    {
      "question_text": "Synthetic question the adapter cannot recognise as an attitude item.",
      "permissible_answers": ["First option", "Second option"],
      "correct_answer": {"EDT": 0, "CDT": 1},
      "is_attitude": true,
      "qid": "9.9ATT",
    },
  ],
}
"""

# Two distinct questions under one qid, the only id collision the corpus can cause by itself.
REPEATED_QID_SETTING = """
{
  "setup": "Synthetic object whose two questions share a qid.",
  "questions": [
    {
      "question_text": "First synthetic question under a repeated qid.",
      "permissible_answers": ["First option", "Second option"],
      "correct_answer": {"EDT": 0, "CDT": 1},
      "attitude_q": true,
      "qid": "9.5ATT",
    },
    {
      "question_text": "Second synthetic question under the SAME qid.",
      "permissible_answers": ["First option", "Second option"],
      "correct_answer": {"EDT": 1, "CDT": 0},
      "attitude_q": true,
      "qid": "9.5ATT",
    },
  ],
}
"""

# CDT and EDT endorsing one option SET in a different ORDER, which tuple equality misses.
SAME_SET_DIFFERENT_ORDER_SETTING = """
{
  "setup": "Synthetic object whose two theories agree on the set and disagree on the order.",
  "questions": [
    {
      "question_text": "Synthetic question both theories answer the same way, listed differently.",
      "permissible_answers": ["First option", "Second option"],
      "correct_answer": {"EDT": [0, 1], "CDT": [1, 0]},
      "attitude_q": true,
      "qid": "8.8ATT",
    },
    {
      "question_text": "Synthetic question whose CDT answer is listed twice under two aliases.",
      "permissible_answers": ["First option", "Second option"],
      "correct_answer": {"CDT": 0, "cdt": 0, "EDT": 1},
      "attitude_q": true,
      "qid": "8.9ATT",
    },
  ],
}
"""


class TestTheAdapterCannotSilentlyShrinkTheBattery:
    def test_two_qidless_questions_in_one_file_get_distinct_ids(self, tmp_path: Path) -> None:
        """A fallback id restarting per top-level object pairs records from different questions."""
        (tmp_path / "setting042synthetic.json").write_text(QIDLESS_TWO_OBJECT_SETTING)
        items = load_dtbench(tmp_path)
        assert len(items) == 2
        assert len({item.probe_id for item in items}) == 2
        assert len({item.scenario for item in items}) == 2

    def test_a_populated_corpus_that_yields_nothing_raises(self, tmp_path: Path) -> None:
        """Schema drift has to fail loudly.

        The comparability half of the battery vanishing quietly leaves a trace that still looks
        complete, an eval section that still "ran", and one info line reading `len(items)=0`.
        """
        (tmp_path / "setting001synthetic.json").write_text(DRIFTED_SCHEMA_SETTING)
        with pytest.raises(ValueError, match="no scoreable attitude item") as error:
            load_dtbench(tmp_path)
        assert "n_questions=1" in str(error.value)
        assert "not an attitude question" in str(error.value)

    def test_the_drop_reasons_are_reported_with_their_denominator(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A zero needs its denominator: how many questions were seen, kept, and dropped why.

        Logged on the SUCCESS path as well as in the raise, so a partial drift -- the instrument
        shrinking from 120 items to 40 -- is visible before it becomes a zero.
        """
        (tmp_path / "setting001synthetic.json").write_text(SYNTHETIC_SETTING)
        with caplog.at_level(logging.INFO, logger="games.probes"):
            load_dtbench(tmp_path)
        assert "n_questions=4" in caplog.text
        assert "len(items)=2" in caplog.text
        assert "not an attitude question" in caplog.text
        assert "CDT and EDT endorse the same option set" in caplog.text

    def test_theories_agreeing_on_the_set_but_not_the_order_are_dropped(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "setting008synthetic.json").write_text(SAME_SET_DIFFERENT_ORDER_SETTING)
        assert [item.probe_id for item in load_dtbench(tmp_path)] == ["dtbench-8.9ATT"]

    def test_an_index_listed_under_two_aliases_is_not_duplicated(self, tmp_path: Path) -> None:
        (tmp_path / "setting008synthetic.json").write_text(SAME_SET_DIFFERENT_ORDER_SETTING)
        item = next(item for item in load_dtbench(tmp_path) if item.probe_id == "dtbench-8.9ATT")
        assert item.theory_answers[CDT] == (0,)
        assert item.theory_answers[EDT] == (1,)


class TestTheWholeBatteryHasUniqueIds:
    def test_our_items_and_dtbench_items_come_back_together(self, dtbench_dir: Path) -> None:
        items = probe_battery(dtbench_dir)
        assert len(items) == len(our_battery()) + len(load_dtbench(dtbench_dir))
        assert len({item.probe_id for item in items}) == len(items)

    def test_no_dtbench_dir_is_our_battery_alone(self) -> None:
        assert [item.probe_id for item in probe_battery(None)] == [
            item.probe_id for item in our_battery()
        ]

    def test_one_of_our_items_colliding_with_a_dtbench_id_raises(
        self, dtbench_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two items sharing an id pair before-and-after records from different questions.

        The collision has to be built from our side: the adapter prefixes every id it mints with
        `dtbench-`, so no corpus question can reach one of our names by itself.
        """
        intruder = ProbeItem(
            probe_id="dtbench-1.1ATT",
            kind=PROBE_MULTIPLE_CHOICE,
            family="test",
            scenario="One of ours wearing a DTBench id.",
            options=("First option", "Second option"),
            theory_answers={CDT: (0,), EDT: (1,), FDT: (1,)},
        )
        monkeypatch.setattr("games.probes.OUR_ITEMS", (*our_battery(), intruder))
        with pytest.raises(ValueError, match="duplicate probe_id"):
            probe_battery(dtbench_dir)

    def test_two_of_our_own_items_sharing_an_id_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hand-authored half needs the guard too, not only the adapter's fallback ids."""
        monkeypatch.setattr("games.probes.OUR_ITEMS", (*our_battery(), our_battery()[0]))
        with pytest.raises(ValueError, match="duplicate probe_id"):
            probe_battery(None)

    def test_two_dtbench_questions_sharing_a_qid_raise(self, tmp_path: Path) -> None:
        """A repeated `qid` is the collision the corpus can produce on its own.

        Every adapter id is prefixed `dtbench-` and no hand-authored id carries that prefix, so a
        corpus question cannot collide with one of ours however it is named -- but two questions in
        the corpus can share a qid, and then the fallback-id fix does not help.
        """
        (tmp_path / "setting905synthetic.json").write_text(REPEATED_QID_SETTING)
        with pytest.raises(ValueError, match="duplicate probe_id"):
            probe_battery(tmp_path)
