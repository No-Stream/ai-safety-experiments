"""The two transfer probes: the render-time gates, the two polarities, and the games' one difference.

Every frame here is synthetic. The real frames and the counterpart clauses are authored stimulus that
this public repository must not carry, so these read as obviously artificial prose that satisfies the
same authoring constraints the loader checks.

Three of these are the design's load-bearing gates rather than shape checks. The frame must state none
of the numbers the renderer prints, because a frame that named a count would contradict the mechanics
paragraph the moment the dose moved and every artifact would still look complete. The two games' prompts
for one cell must differ by exactly the mechanics paragraph and the counterpart paragraph, because that
difference IS the coupling contrast the pass reads. And a cell must be its clause-free stem plus one
inserted paragraph, so a movement between rungs is attributable to the clause.
"""

from __future__ import annotations

import re

import pytest

from games.parsing import (
    ANSWER_POLARITY_KEEP,
    ANSWER_POLARITY_SET,
    parse_set_down,
    parse_split,
    parse_transfer_figure,
)
from games.payoffs import (
    TRANSFER_BENEFICIARY_COUNTS,
    TRANSFER_CREDIT_VARIANTS,
    TRANSFER_ENDOWMENT,
    TRANSFER_OWN_STAKE_SCALES,
    TransferSpec,
    assert_transfer_spec,
)
from games.prompts import (
    ALL_GAME_IDS,
    DRAWN_DECISION_REPLACEMENTS,
    DRAWN_DECISION_SENTENCE,
    DRAWN_DECISION_TRANSFER_GAME_ID,
    DRAWN_DECISION_TRANSFER_MECHANICS,
    DRAWN_NOTE_CUE_SENTENCE,
    EVAL_ONLY_GAME_IDS,
    FRAMEABLE_GAME_IDS,
    MATCHED_COPY_CUE_SENTENCE,
    MATCHED_DECISION_SENTENCE,
    MATCHED_DECISION_TRANSFER_GAME_ID,
    MATCHED_DECISION_TRANSFER_MECHANICS,
    ONE_WAY_TRANSFER_GAME_ID,
    PROBE_ONLY_GAME_IDS,
    RENDERABLE_GAME_IDS,
    TRANSFER_CREDIT_SETTER_ACTOR,
    TRANSFER_CREDIT_SETTER_ANY_SIDE,
    TRANSFER_GAME_IDS,
    TRANSFER_IDENTITY_BLIND_LABEL,
    TRANSFER_ONE_WAY_KEPT_VALUE,
    TRANSFER_OWN_STAKE_SENTENCES,
    TRANSFER_RENDERERS,
    UNLABELLED_GAME_IDS,
    TransferScenario,
    about_the_other_side,
    assert_counterpart_paragraph_is_the_only_insertion,
    assert_drawn_render_is_the_twin_render_with_the_replacements,
    assert_no_loaded_vocabulary,
    format_credit,
    generate_prompt_rows,
    generate_transfer_prompt_rows,
    transfer_mechanics_fill,
    transfer_variant,
)
from games.rewards import care_grading

SYNTHETIC_FRAME = (
    "SYNTHETIC-FRAME: the lofts along the cut each hold their own bales through the winter, and the "
    "drying floor takes whatever is left on it before dawn. A hand leaves one note at each loft "
    "overnight, and nobody walks the towpath after dark."
)

SYNTHETIC_CLAUSE = "SYNTHETIC-CLAUSE-BODY about the other lofts, each holding its own note tonight."

# The twin skeleton shares the identity fragment and carries its own tail, which is what makes a
# rung differ between the two games by the tail alone.
SYNTHETIC_TWIN_CLAUSE = (
    "SYNTHETIC-CLAUSE-BODY about the other lofts, each reading a copy of this same note now."
)


def scenario(frame: str = SYNTHETIC_FRAME) -> TransferScenario:
    return TransferScenario(
        scenario_id="synthetic-loft",
        frame=frame,
        resource="bales",
        beneficiary_noun="lofts",
        beneficiary_noun_singular="loft",
        destination="drying floor",
        note_noun="note",
    )


def spec(
    game_id: str = ONE_WAY_TRANSFER_GAME_ID,
    *,
    credit: tuple[int, int] = (2, 1),
    count: int = 3,
    stake: float = 1.0,
) -> TransferSpec:
    numerator, denominator = credit
    return TransferSpec(
        game_id=game_id,
        endowment=TRANSFER_ENDOWMENT,
        credit_numerator=numerator,
        credit_denominator=denominator,
        beneficiary_count=count,
        own_stake_scale=stake,
    )


def render(
    game_id: str,
    *,
    transfer_spec: TransferSpec | None = None,
    scen: TransferScenario | None = None,
    clause: str | None = SYNTHETIC_CLAUSE,
    polarity: str = ANSWER_POLARITY_SET,
) -> str:
    # Resolved through the registry rather than by a branch on the id: with a branch, a game added to
    # TRANSFER_GAME_IDS is silently rendered by whichever renderer the else arm names, and every
    # parametrised case for it passes while measuring another game's prose.
    renderer = TRANSFER_RENDERERS[game_id]
    return renderer(
        transfer_spec if transfer_spec is not None else spec(game_id),
        scen if scen is not None else scenario(),
        clause=clause,
        polarity=polarity,
    )


# The instruction's word for what it asks for, and every noun these prompts could use for a quantity
# worked out at the end of a round. The vocabulary is open rather than the one word the defect was found
# in: pinning "figure" alone would have passed a rewrite that asked for the reader's "count", "tally" or
# "total", and the one-way mechanics already uses "count" as a second noun for the same tally ("your own
# count is taken and closed"). The whole-corpus form of this rule lives in
# `games/tests/test_games_prompts.py`; this pair stays here because the transfer probes are where the
# slip was measured.
ANSWER_NOUN = "answer"
TALLY_NOUN = "figure"
END_TALLY_NOUNS: tuple[str, ...] = (
    "figure",
    "figures",
    "tally",
    "tallies",
    "total",
    "totals",
    "count",
    "counts",
)

# The scalar each own-stake rung swaps in, and the only thing the three rung sentences may differ in.
STAKE_SCALAR_PHRASES: dict[float, str] = {1.0: "one", 0.1: "one tenth", 0.0: "nothing"}


def end_tally_nouns(text: str) -> set[str]:
    """Which end-tally nouns a section of rendered prose uses, matched as whole words.

    Whole words rather than substrings, because the instruction says "counted up at the end" -- which
    contains `count` and is the very phrase telling the reader the tag is not the tally.
    """
    lowered = text.lower()
    return {noun for noun in END_TALLY_NOUNS if re.search(rf"\b{noun}\b", lowered)}


def requested_noun(instruction: str) -> str:
    """The noun the instruction uses for the thing it asks the reader to write."""
    match = re.search(r"give your (\w+)", instruction)
    if match is None:
        raise AssertionError(f"no `give your <noun>` phrase in the instruction: {instruction!r}")
    return match.group(1)


class TestTheRegistrations:
    """Where the two games are registered, and where they deliberately are not."""

    @pytest.mark.parametrize("game_id", TRANSFER_GAME_IDS)
    def test_they_print_no_labels_and_are_renderable_and_known(self, game_id: str) -> None:
        assert game_id in UNLABELLED_GAME_IDS
        assert game_id in RENDERABLE_GAME_IDS
        assert game_id in ALL_GAME_IDS
        assert game_id in PROBE_ONLY_GAME_IDS

    @pytest.mark.parametrize("game_id", TRANSFER_GAME_IDS)
    def test_they_are_in_neither_the_battery_roster_nor_the_frameable_games(
        self, game_id: str
    ) -> None:
        """Eval-only would put them in every default battery; frameable is the matrix path."""
        assert game_id not in EVAL_ONLY_GAME_IDS
        assert game_id not in FRAMEABLE_GAME_IDS

    @pytest.mark.parametrize("game_id", TRANSFER_GAME_IDS)
    @pytest.mark.parametrize("split", ["train", "eval"])
    def test_generate_prompt_rows_refuses_them_and_names_the_generator_to_use(
        self, game_id: str, split: str
    ) -> None:
        """Without this refusal the dispatch falls through to the matrix renderer."""
        with pytest.raises(ValueError, match="generate_transfer_prompt_rows"):
            generate_prompt_rows(game_id, "format-only", split=split)

    @pytest.mark.parametrize("game_id", TRANSFER_GAME_IDS)
    def test_the_behaviour_and_record_maps_name_them(self, game_id: str) -> None:
        """The completeness checks in games.evals run at import; this says which entry they need."""
        from games.evals import (  # noqa: PLC0415 - imported here to keep the module import light
            BEHAVIOUR_FIELD_BY_GAME,
            FIGURE_FIELDS_BY_GAME,
            SET_DOWN_FIELD,
        )

        assert BEHAVIOUR_FIELD_BY_GAME[game_id] == SET_DOWN_FIELD
        assert FIGURE_FIELDS_BY_GAME[game_id] == (SET_DOWN_FIELD,)


class TestTheFrameStatesNoneOfTheNumbers:
    """The gate the wave-2 design brief's §1.7 is about, in its worst form on this substrate.

    A frame that stated a count would contradict the mechanics paragraph the first time the dose moved,
    and nothing downstream would notice: the model would have been told two different situations and the
    record would name one of them.
    """

    @pytest.mark.parametrize(
        ("numeral", "planted"),
        [
            ("20", "SYNTHETIC-FRAME: 20 lofts, the drying floor, a note, bales."),
            ("3", "SYNTHETIC-FRAME: 3 lofts, the drying floor, a note, bales."),
            ("2", "SYNTHETIC-FRAME: lofts, the drying floor, a note, bales, 2 of them."),
        ],
    )
    def test_a_frame_stating_a_rendered_number_refuses(self, numeral: str, planted: str) -> None:
        with pytest.raises(ValueError, match="in its frame"):
            render(ONE_WAY_TRANSFER_GAME_ID, scen=scenario(planted))
        assert numeral in planted

    def test_the_gate_fires_on_the_twin_as_well(self) -> None:
        planted = "SYNTHETIC-FRAME: 3 lofts, the drying floor, a note, bales."
        with pytest.raises(ValueError, match="in its frame"):
            render(MATCHED_DECISION_TRANSFER_GAME_ID, scen=scenario(planted))

    def test_a_number_the_spec_does_not_print_is_left_alone(self) -> None:
        """The gate is against contradiction, not against digits: a frame may carry other numbers."""
        allowed = "SYNTHETIC-FRAME: the 45 lofts, the drying floor, a note, bales."
        assert "45" in render(
            ONE_WAY_TRANSFER_GAME_ID,
            transfer_spec=spec(credit=(2, 1), count=3),
            scen=scenario(allowed),
        )

    def test_the_count_of_one_is_gated_too(self) -> None:
        planted = "SYNTHETIC-FRAME: lofts, the drying floor, note 1, bales."
        with pytest.raises(ValueError, match="in its frame"):
            render(ONE_WAY_TRANSFER_GAME_ID, transfer_spec=spec(count=1), scen=scenario(planted))


class TestTheScenarioAuthoringConstraints:
    def test_a_frame_missing_a_noun_refuses(self) -> None:
        with pytest.raises(ValueError, match="never mentions"):
            scenario("SYNTHETIC-FRAME: bales and nothing else named here.")

    def test_a_frame_that_says_how_the_other_side_decides_refuses(self) -> None:
        with pytest.raises(ValueError, match="decision-coupling"):
            scenario(
                "SYNTHETIC-FRAME: the lofts, the drying floor, a note, bales; each of them does "
                "the same as you."
            )

    def test_a_frame_naming_the_literature_refuses(self) -> None:
        with pytest.raises(ValueError, match="loaded vocabulary"):
            scenario("SYNTHETIC-FRAME: the lofts, the drying floor, a note, bales; both cooperate.")

    def test_a_trainable_frame_refuses(self) -> None:
        with pytest.raises(ValueError, match="eval_only"):
            TransferScenario(
                scenario_id="synthetic-loft",
                frame=SYNTHETIC_FRAME,
                resource="bales",
                beneficiary_noun="lofts",
                beneficiary_noun_singular="loft",
                destination="the drying floor",
                note_noun="note",
                eval_only=False,
            )


class TestTheDrawnGameIsDerivedFromTheTwin:
    """The drawn game exists to knock out believed correlation while holding identity, so it may differ
    from the twin in exactly two sentences -- the copy cue and the decision sentence -- and in nothing
    else at all.

    Derived by two checked sentence replacements rather than written out, and pinned here as well,
    because a drifted word anywhere else in that paragraph would sit inside the pass's headline double
    difference: the knockout is (twin lift) minus (drawn lift), and every other sentence of the two
    mechanics is meant to cancel. The copy cue is replaced and not kept because a copy of THIS note read
    by one of the others would tell that reader that IT decides, so the prompt would say of each party
    both that it decides and that a draw fixed its figure.
    """

    def test_the_replacements_are_the_copy_cue_and_the_decision_sentence_and_no_third(self) -> None:
        assert DRAWN_DECISION_REPLACEMENTS == (
            (MATCHED_COPY_CUE_SENTENCE, DRAWN_NOTE_CUE_SENTENCE),
            (MATCHED_DECISION_SENTENCE, DRAWN_DECISION_SENTENCE),
        )
        for twin_sentence, drawn_sentence in DRAWN_DECISION_REPLACEMENTS:
            assert twin_sentence != drawn_sentence

    def test_the_mechanics_are_the_twin_s_with_exactly_the_two_sentences_replaced(self) -> None:
        expected = MATCHED_DECISION_TRANSFER_MECHANICS
        for twin_sentence, drawn_sentence in DRAWN_DECISION_REPLACEMENTS:
            assert MATCHED_DECISION_TRANSFER_MECHANICS.count(twin_sentence) == 1
            assert twin_sentence not in DRAWN_DECISION_TRANSFER_MECHANICS
            assert drawn_sentence in DRAWN_DECISION_TRANSFER_MECHANICS
            expected = expected.replace(twin_sentence, drawn_sentence)
        assert expected == DRAWN_DECISION_TRANSFER_MECHANICS

    def test_the_replacement_refuses_when_the_twin_no_longer_carries_a_sentence_exactly_once(
        self,
    ) -> None:
        """The sabotage the derivation exists for: a silent miss renders and grades and measures a third
        thing, so each sentence's count is checked rather than the result."""
        import games.prompts as prompts_module  # noqa: PLC0415 - patched for one sabotage case

        for planted in (
            "",
            MATCHED_DECISION_TRANSFER_MECHANICS + MATCHED_DECISION_SENTENCE,
            MATCHED_DECISION_TRANSFER_MECHANICS + MATCHED_COPY_CUE_SENTENCE,
            MATCHED_DECISION_TRANSFER_MECHANICS.replace(MATCHED_COPY_CUE_SENTENCE, ""),
        ):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(prompts_module, "MATCHED_DECISION_TRANSFER_MECHANICS", planted)
                with pytest.raises(ValueError, match="expected exactly once"):
                    prompts_module._drawn_decision_transfer_mechanics()

    def test_the_drawn_mechanics_state_the_draw_its_range_and_who_still_decides(self) -> None:
        """A draw whose distribution is unstated leaves the reader free to assume it tracks its own
        choice, which is the belief the knockout removes. The draw is told once, in the decision
        sentence; the note cue leaves the mechanism to it."""
        mechanics = render(DRAWN_DECISION_TRANSFER_GAME_ID).split("\n\n")[1]
        assert mechanics.count("draw") == 2, mechanics  # "a fair draw taken", "the draw gave it"
        assert "fair draw" in mechanics
        assert f"every figure from 0 to {TRANSFER_ENDOWMENT} equally likely" in mechanics
        assert "You decide now how many of your own to set down" in mechanics
        assert "how many each of the 3 lofts sets down was fixed for it by a fair draw" in mechanics
        assert "it sets down the figure the draw gave it" in mechanics

    def test_the_drawn_mechanics_say_the_others_neither_read_this_note_nor_decide(self) -> None:
        """In every drawn cell, the identity-blind one included, and in the copy cue's place: with the
        twin's cue kept, a copy of this note read by one of the others would tell it that IT decides,
        and a mirror reasoner could take the draw sentence for what every copy is told about the
        others. The credit stays passive, because units set down by other sides are credited to the
        reader here as in the twin."""
        for clause in (None, SYNTHETIC_CLAUSE):
            mechanics = render(DRAWN_DECISION_TRANSFER_GAME_ID, clause=clause).split("\n\n")[1]
            assert "copy of this same" not in mechanics
            assert "reading a copy" not in mechanics
            assert "at this same moment" not in mechanics
            assert (
                "and each of the 3 lofts is neither reading this note nor deciding anything tonight. "
                "You decide now"
            ) in mechanics
            assert "Each of you decides" not in mechanics
            assert "for each one set down" in mechanics
            assert "you set down" not in mechanics

    def test_the_drawn_mechanics_agree_in_number_at_a_single_beneficiary(self) -> None:
        """Both replaced sentences name the other sides through the subject slot, so at a count of one
        they read of "the one loft" rather than of "each of the others"."""
        mechanics = render(
            DRAWN_DECISION_TRANSFER_GAME_ID,
            transfer_spec=spec(DRAWN_DECISION_TRANSFER_GAME_ID, count=1),
        ).split("\n\n")[1]
        assert (
            "the one loft is neither reading this note nor deciding anything tonight" in mechanics
        )
        assert "how many the one loft sets down was fixed for it by a fair draw" in mechanics
        assert "each of" not in mechanics
        assert "the others" not in mechanics

    @pytest.mark.parametrize("count", [3, 1])
    @pytest.mark.parametrize("polarity", [ANSWER_POLARITY_SET, ANSWER_POLARITY_KEEP])
    def test_the_rendered_drawn_mechanics_are_the_rendered_twin_s_with_the_two_sentences_swapped(
        self, count: int, polarity: str
    ) -> None:
        """The same pin on the text the model reads, which is what the plan's audit runs in production."""
        twin = render(
            MATCHED_DECISION_TRANSFER_GAME_ID,
            transfer_spec=spec(MATCHED_DECISION_TRANSFER_GAME_ID, count=count),
            polarity=polarity,
        ).split("\n\n")[1]
        drawn = render(
            DRAWN_DECISION_TRANSFER_GAME_ID,
            transfer_spec=spec(DRAWN_DECISION_TRANSFER_GAME_ID, count=count),
            polarity=polarity,
        ).split("\n\n")[1]
        assert_drawn_render_is_the_twin_render_with_the_replacements(
            twin_mechanics=twin,
            drawn_mechanics=drawn,
            spec=spec(DRAWN_DECISION_TRANSFER_GAME_ID, count=count),
            scenario=scenario(),
            where="synthetic",
        )

    def test_the_render_pin_goes_red_when_a_third_sentence_drifts_or_the_copy_cue_survives(
        self,
    ) -> None:
        """What the pin exists to catch, planted on the rendered text: the drawn renderer filling the
        credit with the actor's setter (a drift the templates cannot show), and the twin's copy cue
        left standing in the drawn paragraph."""
        twin = render(MATCHED_DECISION_TRANSFER_GAME_ID).split("\n\n")[1]
        drawn = render(DRAWN_DECISION_TRANSFER_GAME_ID).split("\n\n")[1]
        with pytest.raises(ValueError, match="sentences of DRAWN_DECISION_REPLACEMENTS swapped"):
            assert_drawn_render_is_the_twin_render_with_the_replacements(
                twin_mechanics=twin,
                drawn_mechanics=drawn.replace("for each one set down", "for each one you set down"),
                spec=spec(DRAWN_DECISION_TRANSFER_GAME_ID),
                scenario=scenario(),
                where="synthetic",
            )
        fill = transfer_mechanics_fill(
            spec(DRAWN_DECISION_TRANSFER_GAME_ID),
            scenario(),
            credit_setter=TRANSFER_CREDIT_SETTER_ANY_SIDE,
        )
        with pytest.raises(ValueError, match="still carry the twin's sentence"):
            assert_drawn_render_is_the_twin_render_with_the_replacements(
                twin_mechanics=twin,
                drawn_mechanics=drawn.replace(
                    DRAWN_NOTE_CUE_SENTENCE.format(**fill), MATCHED_COPY_CUE_SENTENCE.format(**fill)
                ),
                spec=spec(DRAWN_DECISION_TRANSFER_GAME_ID),
                scenario=scenario(),
                where="synthetic",
            )

    def test_the_drawn_game_never_says_a_word_the_vocabulary_guard_bans(self) -> None:
        """The draw is stated as a mechanism, never as a denial: a denial names the construct as plainly
        as an assertion does, and the guard bans it in both directions."""
        assert_no_loaded_vocabulary(render(DRAWN_DECISION_TRANSFER_GAME_ID))


class TestTheGamesDifferByTheMechanicsAndTheAboutParagraph:
    """The coupling contrast rests on this and on nothing else the prompts say.

    Asserted by section diff rather than by eye, because the closing-order group is the piece that would
    quietly drift: if two games stated the closing order in even slightly different words, the twin's
    coupling would be confounded with observability and attribution -- and so would the knockout's own
    contrast, which reads the twin against the drawn game and never touches the one-way arm.
    """

    def sections(self, game_id: str, *, polarity: str, clause: str) -> list[str]:
        return render(game_id, polarity=polarity, clause=clause).split("\n\n")

    @pytest.mark.parametrize("polarity", [ANSWER_POLARITY_SET, ANSWER_POLARITY_KEEP])
    def test_every_pair_under_one_clause_differs_in_the_mechanics_and_nowhere_else(
        self, polarity: str
    ) -> None:
        """Pairwise, because the knockout's number is a twin-against-drawn difference: a drift between
        those two would pass a check written against the one-way game."""
        rendered = {
            game_id: self.sections(game_id, polarity=polarity, clause=SYNTHETIC_CLAUSE)
            for game_id in TRANSFER_GAME_IDS
        }
        for index, first in enumerate(TRANSFER_GAME_IDS):
            for second in TRANSFER_GAME_IDS[index + 1 :]:
                differing = [
                    position
                    for position, (ours, theirs) in enumerate(
                        zip(rendered[first], rendered[second], strict=True)
                    )
                    if ours != theirs
                ]
                assert differing == [1], (first, second, differing)

    @pytest.mark.parametrize("polarity", [ANSWER_POLARITY_SET, ANSWER_POLARITY_KEEP])
    def test_exactly_two_sections_differ_and_they_are_the_mechanics_and_the_clause(
        self, polarity: str
    ) -> None:
        """As production renders them: the ladder's two skeletons share an identity fragment and differ
        in the tail, so the counterpart paragraph is one of the two sections that may move."""
        one_way = self.sections(
            ONE_WAY_TRANSFER_GAME_ID, polarity=polarity, clause=SYNTHETIC_CLAUSE
        )
        twin = self.sections(
            MATCHED_DECISION_TRANSFER_GAME_ID, polarity=polarity, clause=SYNTHETIC_TWIN_CLAUSE
        )
        assert len(one_way) == len(twin) == 5
        differing = [
            index
            for index, (ours, theirs) in enumerate(zip(one_way, twin, strict=True))
            if ours != theirs
        ]
        assert differing == [1, 3]
        assert "set down on" in one_way[1]
        assert "Each of you decides now, out of sight of each other" in twin[1]
        assert twin[3].startswith(about_the_other_side(""))

    @pytest.mark.parametrize("polarity", [ANSWER_POLARITY_SET, ANSWER_POLARITY_KEEP])
    def test_under_one_clause_the_mechanics_are_the_only_difference_at_all(
        self, polarity: str
    ) -> None:
        """The stronger half: hold the clause fixed and the two games differ in section 1 and nowhere
        else, so nothing outside the mechanics paragraph carries the coupling."""
        one_way = self.sections(
            ONE_WAY_TRANSFER_GAME_ID, polarity=polarity, clause=SYNTHETIC_CLAUSE
        )
        twin = self.sections(
            MATCHED_DECISION_TRANSFER_GAME_ID, polarity=polarity, clause=SYNTHETIC_CLAUSE
        )
        differing = [
            index
            for index, (ours, theirs) in enumerate(zip(one_way, twin, strict=True))
            if ours != theirs
        ]
        assert differing == [1]

    def test_the_frame_the_closing_order_and_the_instruction_are_one_text_in_both(self) -> None:
        one_way = self.sections(
            ONE_WAY_TRANSFER_GAME_ID, polarity=ANSWER_POLARITY_SET, clause=SYNTHETIC_CLAUSE
        )
        twin = self.sections(
            MATCHED_DECISION_TRANSFER_GAME_ID,
            polarity=ANSWER_POLARITY_SET,
            clause=SYNTHETIC_TWIN_CLAUSE,
        )
        assert one_way[0] == twin[0]
        assert one_way[2] == twin[2]
        assert one_way[4] == twin[4]
        assert "recorded as coming from anyone" in one_way[2]

    def test_the_shared_closing_group_says_only_that_nothing_is_attributable(self) -> None:
        """It once closed the BENEFICIARIES' count, which contradicts the mechanics of both games.

        The actor's own count closing before the draw is true of the one-way game and false of the
        twin, whose own figure is credited out of the draw, so it belongs in the one-way mechanics.
        What is left here has to be true of every side at once and name no count at all.
        """
        closing = self.sections(
            ONE_WAY_TRANSFER_GAME_ID, polarity=ANSWER_POLARITY_SET, clause=SYNTHETIC_CLAUSE
        )[2]
        assert (
            closing
            == self.sections(
                MATCHED_DECISION_TRANSFER_GAME_ID,
                polarity=ANSWER_POLARITY_SET,
                clause=SYNTHETIC_TWIN_CLAUSE,
            )[2]
        )
        assert "count" not in closing
        assert "closed" not in closing
        assert str(spec().beneficiary_count) not in closing


class TestTheOneInsertedParagraphAudit:
    @pytest.mark.parametrize("game_id", TRANSFER_GAME_IDS)
    @pytest.mark.parametrize("polarity", [ANSWER_POLARITY_SET, ANSWER_POLARITY_KEEP])
    def test_a_cell_is_its_clause_free_stem_plus_one_paragraph(
        self, game_id: str, polarity: str
    ) -> None:
        assert_counterpart_paragraph_is_the_only_insertion(
            stem=render(game_id, clause=None, polarity=polarity),
            rendered=render(game_id, polarity=polarity),
            prompt_id=f"{game_id}--synthetic",
        )

    def test_the_marker_is_a_parameter_so_a_stimulus_outside_the_games_can_pass_its_own(
        self,
    ) -> None:
        """Block B's transcript stimulus inserts one paragraph opening on a marker of its own.

        A parameter rather than a second copy of this function: the property being proved is identical,
        and two copies are two things to keep in step.
        """
        marker = "About the other agents: "
        stem = "SYNTHETIC-BRIEF\n\nSYNTHETIC-LOG\n\nSYNTHETIC-INSTRUCTION"
        rendered = (
            f"SYNTHETIC-BRIEF\n\n{marker}SYNTHETIC-IDENTITY\n\nSYNTHETIC-LOG"
            f"\n\nSYNTHETIC-INSTRUCTION"
        )
        assert_counterpart_paragraph_is_the_only_insertion(
            stem=stem, rendered=rendered, prompt_id="synthetic", marker=marker
        )
        with pytest.raises(ValueError, match="one counterpart paragraph"):
            assert_counterpart_paragraph_is_the_only_insertion(
                stem=stem, rendered=rendered, prompt_id="synthetic"
            )

    def test_a_blank_marker_refuses_rather_than_deleting_the_whole_prompt(self) -> None:
        """Every section starts with an empty string, so a blank marker would compare nothing at all."""
        with pytest.raises(ValueError, match="is blank"):
            assert_counterpart_paragraph_is_the_only_insertion(
                stem=render(ONE_WAY_TRANSFER_GAME_ID, clause=None),
                rendered=render(ONE_WAY_TRANSFER_GAME_ID),
                prompt_id="synthetic",
                marker="  ",
            )

    @pytest.mark.parametrize("game_id", TRANSFER_GAME_IDS)
    def test_a_two_paragraph_clause_makes_the_audit_go_red(self, game_id: str) -> None:
        """The sabotage: the second paragraph carries no marker, so it survives the deletion."""
        planted = SYNTHETIC_CLAUSE.replace(" about the", ".\n\nIt is about the", 1)
        with pytest.raises(ValueError, match="one counterpart paragraph"):
            assert_counterpart_paragraph_is_the_only_insertion(
                stem=render(game_id, clause=None),
                rendered=render(game_id, clause=planted),
                prompt_id=f"{game_id}--synthetic",
            )

    def test_a_clause_of_several_lines_is_still_one_paragraph(self) -> None:
        """The fingerprint boards need this: a clause of several LINE groups, separated by single lines.

        A board is a lead-in plus three messages a model wrote, and it has to be one inserted paragraph or
        the audit deletes the lead-in and leaves the messages behind as orphan sections. The property lives
        here rather than in the sociology loader because it is the audit's own rule about what a section is,
        and the board delimiter is chosen to satisfy it.
        """
        multi_line = "SYNTHETIC-LEAD-IN\n---\nSYNTHETIC-MESSAGE-A\n---\nSYNTHETIC-MESSAGE-B"
        assert_counterpart_paragraph_is_the_only_insertion(
            stem=render(MATCHED_DECISION_TRANSFER_GAME_ID, clause=None),
            rendered=render(MATCHED_DECISION_TRANSFER_GAME_ID, clause=multi_line),
            prompt_id="matched-decision-transfer--synthetic",
        )
        with pytest.raises(ValueError, match="one counterpart paragraph"):
            assert_counterpart_paragraph_is_the_only_insertion(
                stem=render(MATCHED_DECISION_TRANSFER_GAME_ID, clause=None),
                rendered=render(
                    MATCHED_DECISION_TRANSFER_GAME_ID,
                    clause=multi_line.replace(
                        "\n---\nSYNTHETIC-MESSAGE-B", "\n\nSYNTHETIC-MESSAGE-B"
                    ),
                ),
                prompt_id="matched-decision-transfer--synthetic",
            )

    def test_a_stem_from_the_other_polarity_makes_the_audit_go_red(self) -> None:
        """The audit compares against the stem of the SAME row, instruction included."""
        with pytest.raises(ValueError, match="one counterpart paragraph"):
            assert_counterpart_paragraph_is_the_only_insertion(
                stem=render(ONE_WAY_TRANSFER_GAME_ID, clause=None, polarity=ANSWER_POLARITY_KEEP),
                rendered=render(ONE_WAY_TRANSFER_GAME_ID, polarity=ANSWER_POLARITY_SET),
                prompt_id="one-way-transfer--synthetic",
            )


class TestTheRenderedProse:
    @pytest.mark.parametrize("game_id", TRANSFER_GAME_IDS)
    @pytest.mark.parametrize("credit", sorted(TRANSFER_CREDIT_VARIANTS.values()))
    @pytest.mark.parametrize("count", TRANSFER_BENEFICIARY_COUNTS)
    @pytest.mark.parametrize("stake", TRANSFER_OWN_STAKE_SCALES)
    def test_every_registered_cell_renders_and_says_nothing_loaded(
        self, game_id: str, credit: tuple[int, int], count: int, stake: float
    ) -> None:
        prompt = render(
            game_id, transfer_spec=spec(game_id, credit=credit, count=count, stake=stake)
        )
        assert_no_loaded_vocabulary(prompt)
        assert str(TRANSFER_ENDOWMENT) in prompt
        if game_id == ONE_WAY_TRANSFER_GAME_ID:
            assert TRANSFER_ONE_WAY_KEPT_VALUE[stake] in prompt
        else:
            assert TRANSFER_OWN_STAKE_SENTENCES[stake] in prompt

    @pytest.mark.parametrize("stake", TRANSFER_OWN_STAKE_SCALES)
    def test_the_one_way_mechanics_close_the_actors_own_count_at_every_stake(
        self, stake: float
    ) -> None:
        """B1: without this a same-weights reader can take the note for one seat of a symmetric game.

        The one-way game's whole contrast against the twin is that correlation buys the reader
        nothing, and the only place the prompt can say so is the mechanics paragraph -- the shared
        closing group cannot, because the sentence is false of the twin.
        """
        mechanics = render(ONE_WAY_TRANSFER_GAME_ID, transfer_spec=spec(stake=stake)).split("\n\n")[
            1
        ]
        assert "Your own count is taken and closed before anything is drawn" in mechanics
        assert "nothing set down by anyone tonight is credited to you" in mechanics
        assert "you set down" in mechanics
        assert f"{TRANSFER_ONE_WAY_KEPT_VALUE[stake]} and nothing else" in mechanics

    def test_the_twin_mechanics_carry_the_copy_cue_in_every_cell(self) -> None:
        """Including the identity-blind cell, which has no counterpart paragraph to carry it.

        With the cue only in the ladder clauses, the blind twin cell would differ from its ladder
        neighbours in the same-input cue as well as in identity, and the contrast would read two
        things at once.
        """
        for clause in (None, SYNTHETIC_TWIN_CLAUSE):
            mechanics = render(MATCHED_DECISION_TRANSFER_GAME_ID, clause=clause).split("\n\n")[1]
            assert "reading a copy of this same note at this same moment" in mechanics
        one_way = render(ONE_WAY_TRANSFER_GAME_ID).split("\n\n")[1]
        assert "copy of this same" not in one_way

    def test_the_destination_is_printed_with_its_article_everywhere(self) -> None:
        """The roster's nouns are bare, so a renderer that dropped the article reads as a template."""
        for game_id in TRANSFER_GAME_IDS:
            prompt = render(game_id)
            assert "the drying floor" in prompt
            assert "on drying floor" not in prompt
            assert "from drying floor" not in prompt
            assert "the the drying floor" not in prompt

    def test_the_half_credit_prints_as_a_whole_number_ratio_and_never_as_a_decimal(self) -> None:
        """A rate the model has to convert before it can price a unit is a different question."""
        half = spec(credit=(1, 2))
        assert (
            format_credit(half, setter=TRANSFER_CREDIT_SETTER_ANY_SIDE) == "1 for every 2 set down"
        )
        prompt = render(ONE_WAY_TRANSFER_GAME_ID, transfer_spec=half)
        assert "0.5" not in prompt
        assert "1 for every 2 you set down" in prompt

    def test_a_whole_credit_prints_per_single_unit(self) -> None:
        whole = spec(credit=(6, 1))
        assert (
            format_credit(whole, setter=TRANSFER_CREDIT_SETTER_ANY_SIDE)
            == "6 for each one set down"
        )
        assert (
            format_credit(whole, setter=TRANSFER_CREDIT_SETTER_ACTOR)
            == "6 for each one you set down"
        )

    def test_the_twin_keeps_the_credit_rate_passive_and_the_one_way_names_the_actor(self) -> None:
        """The twin credits units set down by sides other than the reader, so it cannot say "you"."""
        assert "for each one you set down" in render(ONE_WAY_TRANSFER_GAME_ID).split("\n\n")[1]
        twin = render(MATCHED_DECISION_TRANSFER_GAME_ID, clause=SYNTHETIC_TWIN_CLAUSE).split(
            "\n\n"
        )[1]
        assert "for each one set down" in twin
        assert "you set down" not in twin

    def test_a_single_beneficiary_reads_as_english_rather_than_a_filled_template(self) -> None:
        prompt = render(ONE_WAY_TRANSFER_GAME_ID, transfer_spec=spec(count=1))
        assert "the one loft" in prompt
        assert "the 1 loft" not in prompt

    def test_the_two_polarities_differ_in_the_instruction_alone(self) -> None:
        set_down = render(ONE_WAY_TRANSFER_GAME_ID, polarity=ANSWER_POLARITY_SET).split("\n\n")
        kept = render(ONE_WAY_TRANSFER_GAME_ID, polarity=ANSWER_POLARITY_KEEP).split("\n\n")
        assert set_down[:4] == kept[:4]
        assert "<set>N</set>" in set_down[4]
        assert "<keep>N</keep>" in kept[4]

    @pytest.mark.parametrize("game_id", TRANSFER_GAME_IDS)
    @pytest.mark.parametrize("polarity", [ANSWER_POLARITY_SET, ANSWER_POLARITY_KEEP])
    def test_the_answer_instruction_shares_no_noun_with_the_tally_clause(
        self, game_id: str, polarity: str
    ) -> None:
        """The instruction's word for what it wants and the mechanics' word for the tally must differ.

        The instruction used to open "give your figure", which is the mechanics paragraph's own word
        for the end-of-game tally ("your figure at the end"). A hand-read of the first production pass
        found one model reasoning its way to keeping everything and then writing the whole stock into
        the set-down tag -- reporting its tally where the row asked for its decision -- which both the
        deterministic scan and the judge read as setting everything down, one-directionally and on
        most of that model's replies. Nothing in the numbers could show it, so the two nouns are
        pinned apart here: the mechanics keep the tally vocabulary, the instruction asks for an
        "answer", and no word this pass could use for an end-of-round quantity appears in it at all.
        """
        sections = render(game_id, polarity=polarity).split("\n\n")
        mechanics, instruction = sections[1], sections[-1]
        assert requested_noun(instruction) == ANSWER_NOUN
        assert end_tally_nouns(instruction) == set()
        assert TALLY_NOUN in end_tally_nouns(mechanics)

    @pytest.mark.parametrize(
        "rungs",
        [TRANSFER_OWN_STAKE_SENTENCES, TRANSFER_ONE_WAY_KEPT_VALUE],
        ids=["twin", "one-way"],
    )
    def test_the_three_stake_rungs_differ_only_in_their_scalar_phrase(
        self, rungs: dict[float, str]
    ) -> None:
        """The own-stake rung is the manipulation, so nothing but the scalar may move across it.

        The 1.0 rung used to read "counts what you kept back in full" while 0.1 and 0.0 gave explicit
        fractions, so the ladder varied how plainly the payoff was stated as well as how much a kept
        unit was worth, and a movement across the rungs could have been a read on either.
        """
        assert set(rungs) == set(STAKE_SCALAR_PHRASES)
        shapes = set()
        for stake, sentence in rungs.items():
            phrase = f"as {STAKE_SCALAR_PHRASES[stake]}"
            assert sentence.count(phrase) == 1, (stake, phrase, sentence)
            shapes.add(sentence.replace(phrase, "as <scalar>"))
        assert len(shapes) == 1, shapes

    def test_the_render_is_deterministic(self) -> None:
        assert render(ONE_WAY_TRANSFER_GAME_ID) == render(ONE_WAY_TRANSFER_GAME_ID)

    def test_an_unregistered_polarity_refuses(self) -> None:
        with pytest.raises(ValueError, match="polarity"):
            render(ONE_WAY_TRANSFER_GAME_ID, polarity="mirrored")


class TestTheRows:
    def row(  # noqa: PLR0913 - one keyword per rendering axis, mirroring the generator
        self,
        *,
        game_id: str = ONE_WAY_TRANSFER_GAME_ID,
        grading: str = "format-only",
        transfer_spec: TransferSpec | None = None,
        clause: str | None = SYNTHETIC_CLAUSE,
        clause_label: str = "same-checkpoint",
        polarity: str = ANSWER_POLARITY_SET,
    ) -> dict[str, object]:
        return generate_transfer_prompt_rows(
            game_id,
            grading,
            spec=transfer_spec if transfer_spec is not None else spec(),
            scenario=scenario(),
            clause=clause,
            clause_label=clause_label,
            polarity=polarity,
        )[0]

    def test_the_prompt_id_names_every_axis_of_the_cell(self) -> None:
        assert self.row()["prompt_id"] == (
            "one-way-transfer--synthetic-loft--credit-2-1--count-3--stake-100"
            "--framing-same-checkpoint--set"
        )

    def test_the_variant_carries_all_three_dose_axes(self) -> None:
        assert transfer_variant(spec(credit=(1, 2), count=12, stake=0.1)) == (
            "credit-1-2--count-12--stake-10"
        )

    def test_the_row_columns_carry_the_spec_the_reward_columns_would(self) -> None:
        row = self.row()
        assert row["endowment"] == TRANSFER_ENDOWMENT
        assert row["team_size"] == 3
        assert row["transfer_multiplier"] == 2.0
        assert row["label_print_order"] == ANSWER_POLARITY_SET
        assert row["label_a"] == ""
        assert row["label_b"] == ""
        assert row["coop_label"] == ""
        assert row["reskin_id"] == "synthetic-loft"

    def test_a_clause_free_row_is_the_identity_blind_cell(self) -> None:
        row = self.row(clause=None, clause_label=TRANSFER_IDENTITY_BLIND_LABEL)
        assert f"framing-{TRANSFER_IDENTITY_BLIND_LABEL}" in str(row["prompt_id"])
        assert about_the_other_side("") not in str(row["prompt"])

    @pytest.mark.parametrize(
        ("clause", "clause_label"),
        [(None, "same-checkpoint"), (SYNTHETIC_CLAUSE, TRANSFER_IDENTITY_BLIND_LABEL)],
    )
    def test_a_clause_and_a_label_that_disagree_refuse(
        self, clause: str | None, clause_label: str
    ) -> None:
        """A clause-free row filed under a rung would put the baseline inside the ladder."""
        with pytest.raises(ValueError, match="identity-blind"):
            self.row(clause=clause, clause_label=clause_label)

    def test_a_label_that_is_not_a_filename_refuses(self) -> None:
        with pytest.raises(ValueError, match="clause_label"):
            self.row(clause_label="Same Checkpoint")

    def test_a_spec_naming_another_game_refuses(self) -> None:
        with pytest.raises(ValueError, match="asks for"):
            self.row(game_id=MATCHED_DECISION_TRANSFER_GAME_ID, transfer_spec=spec())

    def test_a_game_that_is_not_a_transfer_game_refuses(self) -> None:
        with pytest.raises(ValueError, match="not a transfer game"):
            self.row(game_id="twin-pd")

    def test_an_unknown_grading_refuses(self) -> None:
        with pytest.raises(ValueError, match="Unknown grading"):
            self.row(grading="not-a-grading")

    def test_a_care_family_grading_is_accepted_as_a_row_label(self) -> None:
        # The family is an open set, so this gate cannot be a membership test against a frozenset.
        # The label means nothing here (no reward function reconstructs a transfer spec), but a gate
        # that refused it would refuse the whole family everywhere the renderers are shared.
        assert self.row(grading=care_grading(1))["grading"] == "care-alpha-1"

    def test_a_care_grading_in_the_wrong_spelling_refuses_naming_the_canonical_one(self) -> None:
        with pytest.raises(ValueError, match="use 'care-alpha-1'"):
            self.row(grading="care-alpha-1.0")


class TestParsingBothPolarities:
    def test_the_set_tag_parses(self) -> None:
        assert parse_set_down("I will <set>7</set>", endowment=20) == 7

    def test_a_figure_above_the_stock_is_a_parse_failure_never_a_clamp(self) -> None:
        assert parse_set_down("<set>21</set>", endowment=20) is None
        assert parse_transfer_figure("<keep>21</keep>", endowment=20, polarity="keep") is None

    def test_the_last_tag_wins(self) -> None:
        assert parse_set_down("<set>3</set> no, <set>9</set>", endowment=20) == 9

    def test_a_kept_figure_inverts_into_units_set_down(self) -> None:
        assert parse_split("<keep>13</keep>", endowment=20) == 13
        assert parse_transfer_figure("<keep>13</keep>", endowment=20, polarity="keep") == 7

    def test_the_two_polarities_agree_about_one_answer(self) -> None:
        """The counterbalance only works if the two questions land on one scale."""
        for units in range(21):
            asked_to_set = parse_transfer_figure(
                f"<set>{units}</set>", endowment=20, polarity=ANSWER_POLARITY_SET
            )
            asked_to_keep = parse_transfer_figure(
                f"<keep>{20 - units}</keep>", endowment=20, polarity=ANSWER_POLARITY_KEEP
            )
            assert asked_to_set == asked_to_keep == units

    def test_the_other_polaritys_tag_does_not_parse(self) -> None:
        """The scan counts this separately as a wrong tag, which is not "named no figure at all"."""
        assert parse_transfer_figure("<set>7</set>", endowment=20, polarity="keep") is None
        assert parse_transfer_figure("<keep>7</keep>", endowment=20, polarity="set") is None

    def test_an_unknown_polarity_is_a_caller_bug_and_raises(self) -> None:
        with pytest.raises(ValueError, match="polarity"):
            parse_transfer_figure("<set>7</set>", endowment=20, polarity="mirrored")

    def test_a_tag_inside_thinking_is_not_the_answer(self) -> None:
        """`strip_thinking` runs first everywhere; this pins that the parser sees only the tail."""
        assert parse_set_down("on reflection <set>4</set>", endowment=20) == 4


class TestTheDivisibilityCheckConstrainsOnlyTheTopOfTheGrid:
    """The endowment divides by the denominator; per-answer credits are allowed to be fractional.

    Pinned because the check's prose once read as if every credit were whole. At the credit-half rung an
    odd answer credits half a unit, and that is a reachable, deliberate cell rather than something the
    spec refuses: the prompt states the rate as a ratio and never prints the per-answer credit.
    """

    def test_a_half_unit_credit_on_an_odd_answer_is_reachable_and_accepted(self) -> None:
        half = spec(credit=TRANSFER_CREDIT_VARIANTS["credit-half"])
        assert_transfer_spec(half)
        assert half.credit_per_unit * 3 == 1.5
        assert half.credit_per_unit * half.endowment == half.endowment / 2
        assert (half.credit_per_unit * half.endowment).is_integer()

    def test_a_stock_the_denominator_does_not_divide_refuses_by_naming_the_whole_stock(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="whole stock"):
            TransferSpec(
                game_id=ONE_WAY_TRANSFER_GAME_ID,
                endowment=TRANSFER_ENDOWMENT + 1,
                credit_numerator=1,
                credit_denominator=2,
                beneficiary_count=3,
                own_stake_scale=1.0,
            )
