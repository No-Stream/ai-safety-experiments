"""The transfer stimulus loader: every refusal, and the derivations it recomputes rather than trusts.

Each test here plants exactly the violation one load-time check exists to catch and requires red. That is
the point of the file rather than a nicety: every one of these mistakes leaves a file that loads, renders,
grades and reads as a result -- a frame stating a count contradicts the mechanics paragraph the first
time the dose moves, a rung that drifted by a word varies two things at once, and a plural fragment the
count-one table does not cover renders prose no author wrote.

Every fixture is synthetic. The eight real frames, the seven identity fragments and both rubrics are
authored stimulus that must never appear in tracked code.
"""

from __future__ import annotations

import json
import re
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import pytest

from games.prompts import (
    DRAWN_DECISION_TRANSFER_GAME_ID,
    MATCHED_DECISION_TRANSFER_GAME_ID,
    ONE_WAY_TRANSFER_GAME_ID,
    TRANSFER_GAME_IDS,
)
from sociology import transfer_plan
from sociology.tests.conftest import (
    synthetic_identity_fragments,
    synthetic_transfer_payload,
    synthetic_transfer_scenarios,
    write_synthetic_transfer_stimulus,
)
from sociology.tests.synthetic_fingerprint_dose import synthetic_unrelated_tasks
from sociology.transfer_plan import spec_for
from sociology.transfer_stimulus import (
    APPENDED_SENTENCES_FIELD,
    BOARD_CONTENT_SIDE,
    BOARD_DELIMITER,
    BOARD_IDS,
    BOARD_LEAD_IN_FIELD,
    BOARD_MESSAGE_COUNT,
    BOARD_MODEL_ID_BY_SIDE,
    BOARD_SAMPLING_MAX_TOKENS,
    BOARD_SOURCE_BOARD,
    BOARD_SOURCE_DIGEST_FIELD,
    BOARDS_FIELD,
    INTENT_RUBRIC_FIELD,
    INTENT_VALIDATION_FIELD,
    JUDGE_ARM_GAME_IDS,
    MATCHED_ROUNDS_LADDER,
    MIN_INTENT_VALIDATION_REPLIES,
    MIN_VALIDATION_REPLIES,
    N_SCENARIOS,
    PARAPHRASE_INSTRUCTION_FIELD,
    RECORD_RUNG_MATCHED,
    RUNG_DIFFERENT_FAMILY_TRACK_RECORD,
    RUNG_SAME_CHECKPOINT_COUPLED,
    RUNG_SAME_CHECKPOINT_RECORD_0,
    RUNGS_BY_GAME,
    TRACK_RECORD_ROUNDS,
    TRACK_RECORD_ROUNDS_FIELD,
    TRACK_RECORD_TEMPLATE_FIELD,
    UNRELATED_TASKS_FIELD,
    assert_rungs_are_filenames_and_not_registered_framings,
    board_message_problem,
    clause_for,
    load_stimulus,
    record_rung_id,
    registered_numerals,
    scenario_nouns,
)

BOARD_ID_RAW_FIRST = BOARD_IDS[0]
"""The first raw board, which the sabotages below plant their violations in."""

_BOARD_MARKER_TEXT = "About the other side:"
"""The counterpart paragraph's opening, spelled here so a message can carry it deliberately."""

if TYPE_CHECKING:
    from pathlib import Path

    from sociology.transfer_stimulus import TransferStimulus


def write(tmp_path: Path, **overrides: Any) -> Path:
    return write_synthetic_transfer_stimulus(tmp_path / "transfer_stimulus.json", **overrides)


def load_with(tmp_path: Path, **overrides: Any) -> TransferStimulus:
    return load_stimulus(write(tmp_path, **overrides))


def _cells_with_a_template() -> list[tuple[str, str]]:
    """Every (game, cell) whose clause comes from a template, which is every cell but the boards.

    A board has no template and no single-beneficiary form: it is three messages, one per other side, so
    the cells below are the ones a count-one rewrite can even be asked about.
    """
    return [
        (game_id, rung)
        for game_id, rungs in RUNGS_BY_GAME.items()
        for rung in rungs
        if rung not in BOARD_IDS
    ]


def _person_tails(field: str = "person_tails") -> dict[str, str]:
    """The synthetic person tails, typed, so a sabotage can drop one without confusing the checker."""
    tails = synthetic_transfer_payload()[field]
    assert isinstance(tails, dict)
    return {str(key): str(value) for key, value in tails.items()}


class TestTheFileMustExistAndBeThisVersion:
    def test_a_missing_file_refuses_and_says_why_it_is_gitignored(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="gitignored on purpose"):
            load_stimulus(tmp_path / "absent.json")

    def test_a_version_mismatch_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="version"):
            load_with(tmp_path, version="one-way-transfer-stimulus-v0")


class TestTheScenarios:
    def test_the_synthetic_roster_loads_and_carries_every_frame(self, tmp_path: Path) -> None:
        stimulus = load_with(tmp_path)
        assert len(stimulus.scenarios) == N_SCENARIOS
        assert stimulus.digest

    def test_a_short_roster_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="scenarios, expected exactly"):
            load_with(tmp_path, scenarios=synthetic_transfer_scenarios()[:-1])

    def test_duplicate_scenario_ids_refuse(self, tmp_path: Path) -> None:
        scenarios = synthetic_transfer_scenarios()
        scenarios[1]["scenario_id"] = scenarios[0]["scenario_id"]
        with pytest.raises(ValueError, match="duplicate scenario ids"):
            load_with(tmp_path, scenarios=scenarios)

    @pytest.mark.parametrize("numeral", ["20", "3", "12", "6"])
    def test_a_frame_stating_any_registered_numeral_refuses(
        self, tmp_path: Path, numeral: str
    ) -> None:
        """The design's numbers are the renderer's to print, and a frame that names one contradicts it."""
        assert numeral in registered_numerals()
        scenarios = synthetic_transfer_scenarios()
        scenarios[0]["frame"] = f"{scenarios[0]['frame']} There are {numeral} of them."
        with pytest.raises(ValueError, match="states"):
            load_with(tmp_path, scenarios=scenarios)

    def test_a_frame_missing_one_of_its_nouns_refuses(self, tmp_path: Path) -> None:
        scenarios = synthetic_transfer_scenarios()
        scenarios[0]["frame"] = scenarios[0]["frame"].replace(scenarios[0]["destination"], "there")
        with pytest.raises(ValueError, match="never mentions"):
            load_with(tmp_path, scenarios=scenarios)

    def test_a_frame_that_says_how_the_other_side_decides_refuses(self, tmp_path: Path) -> None:
        scenarios = synthetic_transfer_scenarios()
        scenarios[0]["frame"] += " Each of them does the same as you."
        with pytest.raises(ValueError, match="decision-coupling"):
            load_with(tmp_path, scenarios=scenarios)

    def test_a_frame_naming_the_literature_refuses(self, tmp_path: Path) -> None:
        scenarios = synthetic_transfer_scenarios()
        scenarios[0]["frame"] += " Both sides cooperate here."
        with pytest.raises(ValueError, match="loaded vocabulary"):
            load_with(tmp_path, scenarios=scenarios)

    def test_a_scenario_missing_a_field_refuses_by_name(self, tmp_path: Path) -> None:
        scenarios = synthetic_transfer_scenarios()
        del scenarios[0]["note_noun"]
        with pytest.raises(ValueError, match="note_noun"):
            load_with(tmp_path, scenarios=scenarios)


class TestTheDerivedFragments:
    """Three fragments are computed identities rather than free prose, and the loader recomputes them.

    Each identity is a contrast the readout runs: the two one-insertion siblings make their contrasts a
    read of the inserted phrase, and the size-words-only pair makes its difference the capability
    direction and its mean relatedness. A drifted fragment renders and grades and silently varies two
    things at once.
    """

    def test_the_synthetic_derivations_hold(self, tmp_path: Path) -> None:
        assert load_with(tmp_path).clause_templates

    def test_a_sibling_that_is_two_edits_from_its_base_refuses(self, tmp_path: Path) -> None:
        fragments = synthetic_identity_fragments()
        fragments["sibling-adapter"] = (
            fragments["sibling-adapter"].replace("instances", "INSTANCES") + " and SYNTHETIC-EXTRA"
        )
        with pytest.raises(ValueError, match="exactly one contiguous phrase inserted"):
            load_with(tmp_path, identity_fragments=fragments)

    def test_a_same_task_rung_that_is_not_an_insertion_refuses(self, tmp_path: Path) -> None:
        fragments = synthetic_identity_fragments()
        fragments["same-task-different-family"] = "SYNTHETIC-IDENTITY something else entirely"
        with pytest.raises(ValueError, match="exactly one contiguous phrase inserted"):
            load_with(tmp_path, identity_fragments=fragments)

    def test_a_same_family_pair_differing_in_more_than_the_size_words_refuses(
        self, tmp_path: Path
    ) -> None:
        fragments = synthetic_identity_fragments()
        fragments["same-family-smaller"] += " and SYNTHETIC-EXTRA"
        with pytest.raises(ValueError, match="size words"):
            load_with(tmp_path, identity_fragments=fragments)

    def test_a_missing_fragment_refuses(self, tmp_path: Path) -> None:
        fragments = synthetic_identity_fragments()
        del fragments["person"]
        with pytest.raises(ValueError, match="identity fragments for"):
            load_with(tmp_path, identity_fragments=fragments)


class TestAFragmentIsAFragment:
    def test_a_fragment_carrying_its_own_tail_refuses(self, tmp_path: Path) -> None:
        """A fragment with a tail sits outside the condition its game states while the others sit in."""
        payload = synthetic_transfer_payload()
        fragments = synthetic_identity_fragments()
        fragments["same-checkpoint"] += str(payload["one_way_tail"])
        with pytest.raises(ValueError, match="contains the shared"):
            load_with(tmp_path, identity_fragments=fragments)

    def test_a_fragment_with_a_paragraph_break_refuses(self, tmp_path: Path) -> None:
        """Two paragraphs would break the one-inserted-paragraph audit every reading depends on."""
        fragments = synthetic_identity_fragments()
        fragments["same-checkpoint"] += "\n\nSYNTHETIC-SECOND-PARAGRAPH"
        with pytest.raises(ValueError, match="paragraph break"):
            load_with(tmp_path, identity_fragments=fragments)

    def test_a_blank_fragment_refuses(self, tmp_path: Path) -> None:
        fragments = synthetic_identity_fragments()
        fragments["person"] = "   "
        with pytest.raises(ValueError, match="blank identity fragment"):
            load_with(tmp_path, identity_fragments=fragments)

    def test_a_placeholder_the_filler_cannot_supply_refuses(self, tmp_path: Path) -> None:
        fragments = synthetic_identity_fragments()
        fragments["person"] = "people from {somewhere_else}"
        with pytest.raises(ValueError, match="somewhere_else"):
            load_with(tmp_path, identity_fragments=fragments)


class TestTheOpeningsAndTails:
    def test_a_blank_tail_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="blank"):
            load_with(tmp_path, one_way_tail="  ")

    def test_a_coupled_tail_equal_to_the_plain_twin_tail_refuses(self, tmp_path: Path) -> None:
        """Then the coupled cell is a second copy of the top rung rather than the comparison."""
        payload = synthetic_transfer_payload()
        with pytest.raises(ValueError, match="same tail as every other twin rung"):
            load_with(tmp_path, twin_coupled_tail=payload["twin_tail"])

    def test_a_blank_drawn_tail_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="blank"):
            load_with(tmp_path, drawn_tail="  ")

    def test_a_drawn_tail_equal_to_the_twin_tail_refuses(self, tmp_path: Path) -> None:
        """The twin's tail says the other sides decide, which the drawn mechanics deny.

        A prompt saying both is ambiguous exactly where the knockout needs it not to be, and a reply that
        mirrored anyway would be filed as the draw failing to land rather than as the prompt being unclear.
        """
        payload = synthetic_transfer_payload()
        with pytest.raises(ValueError, match="the twin's own tail"):
            load_with(tmp_path, drawn_tail=payload["twin_tail"])

    def test_a_missing_person_tail_refuses(self, tmp_path: Path) -> None:
        tails = {key: value for key, value in _person_tails().items() if key != "twin"}
        with pytest.raises(ValueError, match="person_tails"):
            load_with(tmp_path, person_tails=tails)

    def test_a_person_tail_table_missing_the_drawn_key_refuses(self, tmp_path: Path) -> None:
        """Every tail has a human form, in both numbers, or the person rung renders under the wrong one."""
        tails = {key: value for key, value in _person_tails().items() if key != "drawn"}
        with pytest.raises(ValueError, match="person_tails for \\['drawn'\\]"):
            load_with(tmp_path, person_tails=tails)

    def test_a_missing_singular_person_tail_refuses(self, tmp_path: Path) -> None:
        """The count-one dose is registered, so the person rung needs its authored singular tail."""
        tails = {
            key: value
            for key, value in _person_tails("person_tails_singular").items()
            if key != "one_way"
        }
        with pytest.raises(ValueError, match="person_tails_singular"):
            load_with(tmp_path, person_tails_singular=tails)

    def test_a_blank_singular_person_tail_refuses(self, tmp_path: Path) -> None:
        tails = {**_person_tails("person_tails_singular"), "twin": "  "}
        with pytest.raises(ValueError, match="person_tails_singular"):
            load_with(tmp_path, person_tails_singular=tails)


class TestTheAppendedSentence:
    """The stranger-with-a-record rung is its base clause plus exactly one authored sentence.

    Every check here is what makes the contrast between that rung and ``different-family`` a read of one
    sentence. A two-sentence extension would be a paragraph-length manipulation reported as a
    sentence-length one, and a count-dependent one would be a different sentence at a single beneficiary
    than at three while both were filed under the same cell id.
    """

    def sentence(self, text: str) -> dict[str, str]:
        return {RUNG_DIFFERENT_FAMILY_TRACK_RECORD: text}

    def test_the_rung_is_its_base_clause_plus_the_authored_sentence(self, tmp_path: Path) -> None:
        stimulus = load_with(tmp_path)
        base = stimulus.clause_templates[MATCHED_DECISION_TRANSFER_GAME_ID, "different-family"]
        record = stimulus.clause_templates[
            MATCHED_DECISION_TRANSFER_GAME_ID, RUNG_DIFFERENT_FAMILY_TRACK_RECORD
        ]
        authored = synthetic_transfer_payload()[APPENDED_SENTENCES_FIELD]
        assert isinstance(authored, dict)
        assert record == base + str(authored[RUNG_DIFFERENT_FAMILY_TRACK_RECORD])

    def test_a_missing_sentence_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="rungs with no sentence"):
            load_with(tmp_path, **{APPENDED_SENTENCES_FIELD: {}})

    def test_a_sentence_for_a_rung_this_file_does_not_author_refuses(self, tmp_path: Path) -> None:
        payload = synthetic_transfer_payload()
        authored = payload[APPENDED_SENTENCES_FIELD]
        assert isinstance(authored, dict)
        table = {**authored, "same-checkpoint": " SYNTHETIC-EXTRA-SENTENCE."}
        with pytest.raises(ValueError, match="sentences for rungs this file does not author"):
            load_with(tmp_path, **{APPENDED_SENTENCES_FIELD: table})

    def test_authoring_a_derived_dose_rung_refuses_rather_than_shadowing_the_template(
        self, tmp_path: Path
    ) -> None:
        """A dose rung authored here would give one cell two sources of its sentence.

        The composition reads one of them, so the file and the ladder could state different counts under
        one cell id and every artifact would still be complete.
        """
        payload = synthetic_transfer_payload()
        authored = payload[APPENDED_SENTENCES_FIELD]
        assert isinstance(authored, dict)
        table = {**authored, record_rung_id(5): " SYNTHETIC-HAND-WRITTEN-RUNG."}
        with pytest.raises(ValueError, match="sentences for rungs this file does not author"):
            load_with(tmp_path, **{APPENDED_SENTENCES_FIELD: table})

    def test_a_blank_sentence_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="is blank"):
            load_with(tmp_path, **{APPENDED_SENTENCES_FIELD: self.sentence("   ")})

    def test_a_sentence_without_its_leading_space_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not open with a space"):
            load_with(tmp_path, **{APPENDED_SENTENCES_FIELD: self.sentence("SYNTHETIC-RECORD.")})

    def test_two_sentences_refuse(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not exactly one sentence"):
            load_with(
                tmp_path,
                **{APPENDED_SENTENCES_FIELD: self.sentence(" SYNTHETIC-ONE. SYNTHETIC-TWO.")},
            )

    def test_a_sentence_with_no_full_stop_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not exactly one sentence"):
            load_with(tmp_path, **{APPENDED_SENTENCES_FIELD: self.sentence(" SYNTHETIC-RECORD")})

    def test_a_count_dependent_sentence_refuses(self, tmp_path: Path) -> None:
        """ "each of them" is rewritten at a single beneficiary and left alone at three."""
        with pytest.raises(ValueError, match="not count-neutral"):
            load_with(
                tmp_path,
                **{
                    APPENDED_SENTENCES_FIELD: self.sentence(
                        " SYNTHETIC-RECORD: each of them set down the same figure."
                    )
                },
            )

    def test_a_sentence_naming_an_unsupplied_placeholder_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="names the placeholders"):
            load_with(
                tmp_path,
                **{APPENDED_SENTENCES_FIELD: self.sentence(" SYNTHETIC-RECORD {rung}.")},
            )


class TestTheRungIdsAreUsableAsLabels:
    def test_every_rung_id_is_a_filename_segment_and_no_public_framing_owns_it(self) -> None:
        assert_rungs_are_filenames_and_not_registered_framings(RUNGS_BY_GAME)

    def test_a_rung_id_that_a_public_framing_already_owns_refuses(self) -> None:
        """Then this probe's rows are indistinguishable, by prompt_id, from a trained framing's rows."""
        with pytest.raises(RuntimeError, match="already owns"):
            assert_rungs_are_filenames_and_not_registered_framings(
                {MATCHED_DECISION_TRANSFER_GAME_ID: ("twin",)}
            )

    def test_a_rung_id_the_label_pattern_refuses_is_caught_too(self) -> None:
        with pytest.raises(RuntimeError, match=r"pattern .* refuses"):
            assert_rungs_are_filenames_and_not_registered_framings(
                {MATCHED_DECISION_TRANSFER_GAME_ID: ("Same_Checkpoint",)}
            )


class TestTheDrawnGameSharesTheTwinsJudgeMaterial:
    def test_it_takes_the_twin_s_rubric_its_validation_set_and_its_verdict_keys(
        self, tmp_path: Path
    ) -> None:
        """One rubric, one validation list, one schema: the two games differ in the PROMPT the subject
        read, which the judge never sees."""
        stimulus = load_with(tmp_path)
        assert (
            stimulus.judge_instructions[DRAWN_DECISION_TRANSFER_GAME_ID]
            == stimulus.judge_instructions[MATCHED_DECISION_TRANSFER_GAME_ID]
        )
        # Compared on the authored material rather than on the whole record: the loader stamps each
        # reply with the game it was read for, which is the one field that differs by construction.
        assert [
            (reply.name, reply.reply, reply.reasoning, reply.expected)
            for reply in stimulus.validation_replies[DRAWN_DECISION_TRANSFER_GAME_ID]
        ] == [
            (reply.name, reply.reply, reply.reasoning, reply.expected)
            for reply in stimulus.validation_replies[MATCHED_DECISION_TRANSFER_GAME_ID]
        ]
        assert DRAWN_DECISION_TRANSFER_GAME_ID not in JUDGE_ARM_GAME_IDS

    def test_an_arm_with_no_intent_case_at_all_refuses(self, tmp_path: Path) -> None:
        """A case's game decides which rules paragraph it is read under, and they say opposite things."""
        payload = synthetic_transfer_payload()
        # Re-filed under another arm rather than deleted, so the set still clears the pooled floor and the
        # refusal that fires is the per-arm one.
        cases = [
            {**case, "game_id": ONE_WAY_TRANSFER_GAME_ID}
            if case["game_id"] == DRAWN_DECISION_TRANSFER_GAME_ID  # type: ignore[index]
            else case
            for case in payload[INTENT_VALIDATION_FIELD]  # type: ignore[union-attr]
        ]
        with pytest.raises(ValueError, match="fewer than 1"):
            load_with(tmp_path, **{INTENT_VALIDATION_FIELD: cases})


class TestTheCountOneRewrite:
    """A single beneficiary is a registered dose, so every clause has to read as English at one.

    The table is allowed to be incomplete; what is not allowed is for an uncovered fragment to render
    anyway. The marker check is that tripwire, and it runs at load over every cell at a count of one.
    """

    def test_every_cell_reads_singular_at_one_beneficiary(self, tmp_path: Path) -> None:
        """Every cell with a clause TEMPLATE, which is every cell but the boards.

        A board is three messages, one per other side, so it exists at the reference count alone and
        refuses to render at one: a board of three read by a single beneficiary would be a different
        manipulation under the same cell id, and the count-one rewrite would edit prose a model wrote.
        """
        stimulus = load_with(tmp_path)
        for game_id, rungs in RUNGS_BY_GAME.items():
            for rung in rungs:
                if rung in BOARD_IDS:
                    continue
                clause = clause_for(
                    stimulus,
                    game_id=game_id,
                    rung=rung,
                    spec=spec_for(game_id, count=1),
                    scenario=stimulus.scenarios[0],
                )
                assert " are " not in clause
                assert "themselves" not in clause
                assert "{" not in clause

    def test_a_plural_form_the_table_does_not_cover_refuses_at_load(self, tmp_path: Path) -> None:
        """The sabotage: a fragment whose plural the table cannot rewrite must fail by name."""
        fragments = synthetic_identity_fragments()
        fragments["different-family"] = (
            "SYNTHETIC-IDENTITY systems that are each of them built separately from you"
        )
        fragments["same-task-different-family"] = (
            fragments["different-family"] + ", SYNTHETIC-SAME-PROGRAMME as you"
        )
        with pytest.raises(ValueError, match="plural-only text"):
            load_with(tmp_path, identity_fragments=fragments)

    def test_a_surviving_count_placeholder_refuses(self, tmp_path: Path) -> None:
        """``{count}`` outside the shapes the table knows would render "the other 1 notes"."""
        fragments = synthetic_identity_fragments()
        fragments["person"] = "people from all {count} of the sheds"
        with pytest.raises(ValueError, match="plural-only text"):
            load_with(tmp_path, identity_fragments=fragments)

    def test_a_people_fragment_the_anchored_rule_cannot_rewrite_goes_red(
        self, tmp_path: Path
    ) -> None:
        """The sabotage for the people rule, whose marker was loosened when the rule was anchored.

        The rule matches the clause opening's "is people, each of them", so a fragment that opens with
        the plural in any other shape is not rewritten. Without a marker to catch that, the singular
        cell would render "the one keeper is people ..." and nothing would say so.
        """
        fragments = synthetic_identity_fragments()
        fragments["person"] = "people SYNTHETIC-HUMAN-PLURAL"
        with pytest.raises(ValueError, match="is people"):
            load_with(tmp_path, identity_fragments=fragments)

    @pytest.mark.parametrize("word", ["these", "those", "both"])
    def test_a_plural_demonstrative_the_table_cannot_rewrite_goes_red(
        self, tmp_path: Path, word: str
    ) -> None:
        """The sabotage for the demonstrative markers: "these", "those" and "both" name more than one.

        A fragment that says "these systems" survives every substitution -- no ``{count}``, no "are", no
        "instances" -- and would render "the one loft is these systems" at a single beneficiary. The
        marker is what catches it, and the fix is a reword rather than a table entry, because the
        singular of a demonstrative depends on the noun it points at.
        """
        fragments = synthetic_identity_fragments()
        fragments["different-family"] = (
            f"SYNTHETIC-IDENTITY systems, {word} built separately from you"
        )
        fragments["same-task-different-family"] = (
            fragments["different-family"] + ", SYNTHETIC-SAME-PROGRAMME as you"
        )
        with pytest.raises(ValueError, match=f"plural-only text.*{word} ") as raised:
            load_with(tmp_path, identity_fragments=fragments)
        assert "different-family" in str(raised.value)

    def test_the_counted_demonstrative_is_rewritten_to_this_one(self, tmp_path: Path) -> None:
        """The one demonstrative shape the table does cover: "these {count} {noun}" -> "this one {noun}".

        The synthetic opening is written in exactly that shape, so every cell at a count of one is the
        positive case; the assertion below is that none of them reads "these one".
        """
        stimulus = load_with(tmp_path)
        for game_id, rung in _cells_with_a_template():
            clause = clause_for(
                stimulus,
                game_id=game_id,
                rung=rung,
                spec=spec_for(game_id, count=1),
                scenario=stimulus.scenarios[0],
            )
            assert "these " not in clause
            assert "this one loft" in clause

    def test_the_builders_phrase_keeps_its_plural_at_one_beneficiary(self, tmp_path: Path) -> None:
        """Two rungs say the beneficiaries were made by different people, which stays plural at one.

        This is the other half of the anchored people rule: an unanchored one rewrote the phrase to
        "by different one person out of ..." in different-family and its same-task sibling, and those
        are two of the rungs the identity ladder samples at a single beneficiary.
        """
        stimulus = load_with(tmp_path)
        for game_id, rungs in RUNGS_BY_GAME.items():
            for rung in ("different-family", "same-task-different-family"):
                if rung not in rungs:
                    # The drawn game carries one rung and no ladder, so it has neither of these cells.
                    continue
                clause = clause_for(
                    stimulus,
                    game_id=game_id,
                    rung=rung,
                    spec=spec_for(game_id, count=1),
                    scenario=stimulus.scenarios[0],
                )
                assert "by different people out of" in clause
                assert "one person" not in clause

    def test_the_person_rung_takes_its_authored_singular_tail_rather_than_a_rewrite(
        self, tmp_path: Path
    ) -> None:
        """A rewritten plural tail called a human being "it"; the authored singular is why it cannot.

        Every other rung is about models, so the table's "They decide" -> "It decides" is right there.
        On this rung it is not, which is why the file carries the singular prose for these cells only.
        """
        stimulus = load_with(tmp_path)
        singular_tails = _person_tails("person_tails_singular")
        for game_id, rung in _cells_with_a_template():
            clause = clause_for(
                stimulus,
                game_id=game_id,
                rung=rung,
                spec=spec_for(game_id, count=1),
                scenario=stimulus.scenarios[0],
            )
            if rung != "person":
                assert "PERSON-TAIL-ONE" not in clause
                continue
                key = "one_way" if game_id == ONE_WAY_TRANSFER_GAME_ID else "twin"
                assert singular_tails[key].format(note_noun=stimulus.scenarios[0].note_noun) in (
                    clause
                )
                assert "one person" in clause
                assert " It " not in clause
                assert "itself" not in clause


class TestTheRubricsAndValidators:
    def test_both_rubrics_are_required(self, tmp_path: Path) -> None:
        for field in ("judge_instructions_one_way", "judge_instructions_twin"):
            with pytest.raises(ValueError, match=field):
                load_with(tmp_path, **{field: "   "})

    @pytest.mark.parametrize(
        ("game_id", "field"),
        [
            (ONE_WAY_TRANSFER_GAME_ID, "validation_replies_one_way"),
            (MATCHED_DECISION_TRANSFER_GAME_ID, "validation_replies_twin"),
        ],
    )
    def test_each_arm_has_its_own_validation_floor(
        self, tmp_path: Path, game_id: str, field: str
    ) -> None:
        payload = synthetic_transfer_payload()
        short = list(payload[field])[: MIN_VALIDATION_REPLIES[game_id] - 1]  # type: ignore[call-overload]
        with pytest.raises(ValueError, match="not calibrated on fewer than"):
            load_with(tmp_path, **{field: short})

    def test_a_validation_reply_missing_an_expectation_refuses(self, tmp_path: Path) -> None:
        payload = synthetic_transfer_payload()
        replies = [dict(reply) for reply in payload["validation_replies_twin"]]  # type: ignore[union-attr]
        expected = dict(replies[0]["expected"])
        del expected["counterpart_reasoning"]
        replies[0]["expected"] = expected
        with pytest.raises(ValueError, match="counterpart_reasoning"):
            load_with(tmp_path, validation_replies_twin=replies)

    def test_a_validation_reply_with_no_polarity_refuses(self, tmp_path: Path) -> None:
        """The judge is told which tag the reply was asked for, so a case without one validates nothing."""
        payload = synthetic_transfer_payload()
        replies = [dict(reply) for reply in payload["validation_replies_one_way"]]  # type: ignore[union-attr]
        replies[0]["polarity"] = ""
        with pytest.raises(ValueError, match="polarity"):
            load_with(tmp_path, validation_replies_one_way=replies)

    def test_the_intent_rubric_is_required(self, tmp_path: Path) -> None:
        """Without it every tag slip in the run stays scored as the tag, silently."""
        with pytest.raises(ValueError, match=INTENT_RUBRIC_FIELD):
            load_with(tmp_path, **{INTENT_RUBRIC_FIELD: "   "})

    def test_the_intent_check_has_its_own_validation_floor(self, tmp_path: Path) -> None:
        payload = synthetic_transfer_payload()
        cases = list(payload[INTENT_VALIDATION_FIELD])  # type: ignore[call-overload]
        short = cases[: MIN_INTENT_VALIDATION_REPLIES - 1]
        with pytest.raises(ValueError, match="not calibrated on fewer than"):
            load_with(tmp_path, **{INTENT_VALIDATION_FIELD: short})

    def test_an_intent_case_missing_an_expectation_refuses(self, tmp_path: Path) -> None:
        payload = synthetic_transfer_payload()
        cases = [dict(case) for case in payload[INTENT_VALIDATION_FIELD]]  # type: ignore[union-attr]
        expected = dict(cases[0]["expected"])
        del expected["assumes_return"]
        cases[0]["expected"] = expected
        with pytest.raises(ValueError, match="assumes_return"):
            load_with(tmp_path, **{INTENT_VALIDATION_FIELD: cases})

    def test_an_intent_case_that_names_no_arm_refuses(self, tmp_path: Path) -> None:
        """The rules a case is read against decide whether a symmetric payoff line is a misreading."""
        payload = synthetic_transfer_payload()
        cases = [dict(case) for case in payload[INTENT_VALIDATION_FIELD]]  # type: ignore[union-attr]
        cases[0]["game_id"] = ""
        with pytest.raises(ValueError, match="names game"):
            load_with(tmp_path, **{INTENT_VALIDATION_FIELD: cases})

    def test_the_intent_cases_cover_both_arms_and_both_directions_of_the_slip(
        self, tmp_path: Path
    ) -> None:
        """A one-directional authored set cannot tell a one-directional defect from a one-directional check."""
        stimulus = load_with(tmp_path)
        cases = stimulus.intent_validation_replies
        assert {case.game_id for case in cases} == set(TRANSFER_GAME_IDS)
        assert {case.polarity for case in cases} == {"set", "keep"}
        actions = {str(case.expected["concluded_action"]) for case in cases}
        assert {"give_all", "give_none", "give_some", "unclear", "no_reasoning"} <= actions
        assert any(bool(case.expected["assumes_return"]) for case in cases)


class TestTheLoadedShape:
    def test_the_twin_carries_the_two_cells_the_one_way_game_has_no_form_of(
        self, tmp_path: Path
    ) -> None:
        """The coupled top rung, and the stranger with a stated record. Both are twin-only by design.

        The one-way game states that nothing anyone else sets down reaches the reader, so neither an
        asserted coupling nor a record of matched figures is a thing a reader there could collect on.
        """
        stimulus = load_with(tmp_path)
        one_way = {
            rung for game, rung in stimulus.clause_templates if game == ONE_WAY_TRANSFER_GAME_ID
        }
        twin = {
            rung
            for game, rung in stimulus.clause_templates
            if game == MATCHED_DECISION_TRANSFER_GAME_ID
        }
        assert twin - one_way == {
            RUNG_SAME_CHECKPOINT_COUPLED,
            RUNG_DIFFERENT_FAMILY_TRACK_RECORD,
            RUNG_SAME_CHECKPOINT_RECORD_0,
            *(record_rung_id(matched) for matched in MATCHED_ROUNDS_LADDER),
        }

    def test_the_drawn_game_carries_the_top_rung_alone(self, tmp_path: Path) -> None:
        """Its question is one cell against its own blind baseline, not a ladder."""
        stimulus = load_with(tmp_path)
        drawn = {
            rung
            for game, rung in stimulus.clause_templates
            if game == DRAWN_DECISION_TRANSFER_GAME_ID
        }
        assert drawn == {"same-checkpoint"}

    def test_a_rung_differs_between_the_two_games_by_the_tail_alone(self, tmp_path: Path) -> None:
        """The identity fragment is shared, so the pair is a read of the coupling rather than of wording."""
        stimulus = load_with(tmp_path)
        one_way = stimulus.clause_templates[ONE_WAY_TRANSFER_GAME_ID, "same-checkpoint"]
        twin = stimulus.clause_templates[MATCHED_DECISION_TRANSFER_GAME_ID, "same-checkpoint"]
        shared = str(synthetic_transfer_payload()["one_way_tail"])
        assert one_way.endswith(shared)
        assert one_way.removesuffix(shared) == twin.removesuffix(
            str(synthetic_transfer_payload()["twin_tail"])
        )

    def test_the_digest_moves_when_the_file_moves(self, tmp_path: Path) -> None:
        first = load_with(tmp_path).digest
        second = load_with(
            tmp_path, judge_instructions_twin="SYNTHETIC-RUBRIC-MD: reply once."
        ).digest
        assert first != second

    def test_a_rubric_or_validation_edit_leaves_the_prompt_digest_where_it_was(
        self, tmp_path: Path
    ) -> None:
        """The judge-side sections move the whole-payload digest and nothing a reply record answers.

        Stamping reply rows with the whole digest is what mislabelled every collected row and refused
        every live resume after a rubric edit; the prompt digest is the one a reply record carries.
        """
        baseline = load_with(tmp_path)
        payload = synthetic_transfer_payload()
        replies = payload["validation_replies_twin"]
        assert isinstance(replies, list)
        for edited in (
            load_with(tmp_path, judge_instructions_twin="SYNTHETIC-RUBRIC-MD: reply once."),
            load_with(tmp_path, judge_instructions_one_way="SYNTHETIC-RUBRIC-OW: reply once."),
            load_with(
                tmp_path,
                validation_replies_twin=[
                    {**replies[0], "reasoning": "SYNTHETIC-EDIT"},
                    *replies[1:],
                ],
            ),
        ):
            assert edited.digest != baseline.digest
            assert edited.prompt_digest == baseline.prompt_digest

    @pytest.mark.parametrize(
        "field",
        [
            "scenarios",
            "identity_fragments",
            "one_way_opening",
            "twin_opening",
            "one_way_tail",
            "twin_tail",
            "twin_coupled_tail",
            "person_tails",
            "person_tails_singular",
        ],
    )
    def test_every_prompt_affecting_section_moves_the_prompt_digest(
        self, tmp_path: Path, field: str
    ) -> None:
        baseline = load_with(tmp_path)
        payload = synthetic_transfer_payload()
        value = payload[field]
        if field == "scenarios":
            assert isinstance(value, list)
            edited: object = [
                {**value[0], "frame": value[0]["frame"] + " SYNTHETIC-EDIT."},
                *value[1:],
            ]
        elif isinstance(value, dict):
            # The person fragment: the other six are bases or derivations of one another, so editing one
            # of them trips the derivation check before the digest is ever computed.
            key = "person" if field == "identity_fragments" else next(iter(value))
            edited = {**value, key: f"{value[key]} SYNTHETIC-EDIT"}
        else:
            edited = f"{value} SYNTHETIC-EDIT"
        moved = load_with(tmp_path, **{field: edited})
        assert moved.prompt_digest != baseline.prompt_digest
        assert moved.digest != baseline.digest

    def test_the_prompt_digest_is_not_the_whole_payload_digest(self, tmp_path: Path) -> None:
        stimulus = load_with(tmp_path)
        assert stimulus.prompt_digest != stimulus.digest
        assert len(stimulus.prompt_digest) == len(stimulus.digest) == 16

    def test_a_top_level_field_the_digest_split_does_not_classify_refuses(
        self, tmp_path: Path
    ) -> None:
        """A new file section has to be filed as prompt-affecting or judge-side before it loads.

        Otherwise a field that changed what the model reads would move neither digest's expectations,
        and a resume would continue a file sampled under different prompts.
        """
        with pytest.raises(ValueError, match="synthetic_unclassified_section"):
            load_with(tmp_path, synthetic_unclassified_section="SYNTHETIC-TEXT")

    def test_an_unknown_rung_refuses_and_lists_the_cells(self, tmp_path: Path) -> None:
        stimulus = load_with(tmp_path)
        with pytest.raises(ValueError, match="known cells"):
            clause_for(
                stimulus,
                game_id=ONE_WAY_TRANSFER_GAME_ID,
                rung=RUNG_SAME_CHECKPOINT_COUPLED,
                spec=spec_for(ONE_WAY_TRANSFER_GAME_ID),
                scenario=stimulus.scenarios[0],
            )

    def test_the_scenario_lookup_names_the_roster_when_the_id_is_absent(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="known"):
            load_with(tmp_path).scenario("synthetic-not-a-frame")

    def test_a_malformed_file_is_a_json_error_rather_than_a_silent_default(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "transfer_stimulus.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_stimulus(path)


def _boards_payload() -> tuple[dict[str, Any], str, dict[str, Any]]:
    """The synthetic payload, the first scenario's id, and its board table, ready to be sabotaged."""
    payload = synthetic_transfer_payload()
    boards = payload[BOARDS_FIELD]
    assert isinstance(boards, dict)
    scenario_id = str(synthetic_transfer_scenarios()[0]["scenario_id"])
    entry = boards[scenario_id]
    assert isinstance(entry, dict)
    return payload, scenario_id, entry


def _with_message(
    text: str, *, board_id: str = BOARD_ID_RAW_FIRST, index: int = 0
) -> dict[str, Any]:
    """The synthetic board table with one message replaced by a planted one."""
    payload, scenario_id, entry = _boards_payload()
    board = dict(entry[board_id])
    messages = list(board["messages"])
    messages[index] = text
    board["messages"] = messages
    boards = payload[BOARDS_FIELD]
    assert isinstance(boards, dict)
    boards[scenario_id] = {**entry, board_id: board}
    return {BOARDS_FIELD: boards}


class TestTheBoards:
    """Every board is model output, so the loader gates it message by message.

    The whole of pass D is whether a reader infers that the other sides are the same system from HOW they
    write. Each refusal below is a way a message could tell it instead, or could break the one inserted
    paragraph the audit depends on -- and every one of them would otherwise render, sample and read as a
    result, with the readout still reporting an inferred sameness.
    """

    def test_the_synthetic_boards_load_and_carry_four_boards_of_three_messages(
        self, tmp_path: Path
    ) -> None:
        stimulus = load_with(tmp_path)
        for scenario in stimulus.scenarios:
            assert set(stimulus.boards[scenario.scenario_id]) == set(BOARD_IDS)
            for board_id in BOARD_IDS:
                board = stimulus.boards[scenario.scenario_id][board_id]
                assert len(board.messages) == BOARD_MESSAGE_COUNT
                assert len(board.provenance) == BOARD_MESSAGE_COUNT

    def test_a_board_clause_is_the_lead_in_then_the_messages_in_one_paragraph(
        self, tmp_path: Path
    ) -> None:
        """One paragraph by construction: the delimiter is a single line, never a blank one.

        A blank line would leave the messages behind as orphan sections when the insertion audit deletes
        the paragraph that opens with the counterpart marker, so every board render would go red.
        """
        stimulus = load_with(tmp_path)
        scenario = stimulus.scenarios[0]
        clause = clause_for(
            stimulus,
            game_id=MATCHED_DECISION_TRANSFER_GAME_ID,
            rung=BOARD_ID_RAW_FIRST,
            spec=spec_for(MATCHED_DECISION_TRANSFER_GAME_ID),
            scenario=scenario,
        )
        assert "\n\n" not in clause
        assert clause.startswith("SYNTHETIC-BOARD-LEAD-IN")
        assert BOARD_DELIMITER in clause
        for message in stimulus.boards[scenario.scenario_id][BOARD_ID_RAW_FIRST].messages:
            assert message in clause
        assert "{" not in clause

    def test_a_board_asked_for_at_one_beneficiary_refuses(self, tmp_path: Path) -> None:
        """Boards exist at the reference count alone: one message per other side."""
        stimulus = load_with(tmp_path)
        with pytest.raises(ValueError, match="boards exist at"):
            clause_for(
                stimulus,
                game_id=MATCHED_DECISION_TRANSFER_GAME_ID,
                rung=BOARD_ID_RAW_FIRST,
                spec=spec_for(MATCHED_DECISION_TRANSFER_GAME_ID, count=1),
                scenario=stimulus.scenarios[0],
            )

    def test_a_board_asked_for_on_a_game_that_has_none_refuses(self, tmp_path: Path) -> None:
        stimulus = load_with(tmp_path)
        with pytest.raises(ValueError, match="boards ride the twin alone"):
            clause_for(
                stimulus,
                game_id=ONE_WAY_TRANSFER_GAME_ID,
                rung=BOARD_ID_RAW_FIRST,
                spec=spec_for(ONE_WAY_TRANSFER_GAME_ID),
                scenario=stimulus.scenarios[0],
            )

    @pytest.mark.parametrize(
        ("planted", "expected"),
        [
            ("a plain reply about rinsing a filter\n\nand a second paragraph of it", "blank line"),
            (
                "a plain reply about rinsing a filter\r\n\r\nand a second paragraph of it",
                "carriage return",
            ),
            ("a plain reply about rinsing a filter and {a placeholder} of it", "brace"),
            ("a plain reply about rinsing a filter and <a tag> inside it", "angle bracket"),
            (f"{_BOARD_MARKER_TEXT} they are all the same as me here", "counterpart paragraph"),
            ("too short", "characters, outside"),
            ("a plainly long reply about rinsing a mesh filter. " * 20, "characters, outside"),
            ("I am Claude, and I would rinse the filter under the tap first", "names"),
            ("as an AI, I would rinse the filter under the warm tap first of all", "names"),
            (
                "rinse the filter and leave it in the drying floor overnight to dry",
                "scenario nouns",
            ),
            ("rinse the filter, then leave it for 20 minutes before refitting it", "states"),
            (
                "rinse the filter, and we both cooperate on the drying afterwards",
                "loaded vocabulary",
            ),
            ("<think>rinse it</think> rinse the filter under the tap and leave it", "think tag"),
        ],
    )
    def test_a_message_of_each_banned_shape_refuses_by_name(
        self, tmp_path: Path, planted: str, expected: str
    ) -> None:
        """The sabotage, one row per rule the board gate exists to enforce.

        Both faces of the gate on every row: the loader's refusal, and the generator's own read of the
        same message. Two of these shapes (a brace, a blank line) are ALSO caught downstream by the
        render checks, so without the second assertion those rows would still go red with the board gate
        removed entirely -- a test that reads as covering the gate while covering something else.
        """
        nouns = scenario_nouns(load_with(tmp_path).scenarios)
        problem = board_message_problem(planted, nouns=nouns)
        assert problem is not None
        assert re.search(expected, problem), problem
        with pytest.raises(ValueError, match=expected):
            load_with(tmp_path, **_with_message(planted))

    @pytest.mark.parametrize(
        "ordinary",
        [
            "the metal filter wants a rinse under the warm tap before it goes back in place",
            "keep the metadata sheet with it, and let the whole thing dry before you refit it",
            "there is no assistance needed here, just warm water and a few minutes of patience",
        ],
    )
    def test_ordinary_prose_that_merely_contains_a_lab_name_is_not_refused(
        self, tmp_path: Path, ordinary: str
    ) -> None:
        """The negative control: the lab patterns carry word boundaries where a stem would over-fire.

        Not a nicety at generation time. The boards are drawn on everyday-procedure tasks, the generator
        re-draws a refused message up to a cap, and a pattern that refused every mention of METAL would
        spend the whole cap on a task about a metal filter and then refuse the run by name.
        """
        nouns = scenario_nouns(load_with(tmp_path).scenarios)
        assert board_message_problem(f"SYNTHETIC-MESSAGE: {ordinary}", nouns=nouns) is None

    def test_the_gate_and_the_generator_read_the_same_list(self, tmp_path: Path) -> None:
        """The generator re-draws on a problem and the loader refuses on it, out of one function.

        Two lists would be the failure this pass cannot afford: a message the generator accepted and the
        loader refused would be written into the frozen file and then block every build of both passes.
        """
        stimulus = load_with(tmp_path)
        nouns = scenario_nouns(stimulus.scenarios)
        assert "loft" in nouns
        accepted = stimulus.boards[stimulus.scenarios[0].scenario_id][BOARD_ID_RAW_FIRST].messages
        for message in accepted:
            assert board_message_problem(message, nouns=nouns) is None
        assert board_message_problem("I am Claude and I rinse filters", nouns=nouns) is not None
        assert board_message_problem("rinse it out over the drying floor", nouns=nouns) is not None

    def test_a_board_table_of_nothing_refuses_and_the_escape_hatch_still_renders_nothing(
        self, tmp_path: Path
    ) -> None:
        """The state between the author writing the tasks and the generator filling the boards.

        The escape hatch loads the file, so the dose pass can be planned in that window; it does not
        render a board of nothing, which is what keeps it from becoming a way to sample one.
        """
        payload = synthetic_transfer_payload()
        payload[BOARDS_FIELD] = {}
        path = tmp_path / "transfer_stimulus.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="scenarios with no board"):
            load_stimulus(path)
        stimulus = load_stimulus(path, allow_empty_boards=True)
        with pytest.raises(ValueError, match="carries no messages"):
            clause_for(
                stimulus,
                game_id=MATCHED_DECISION_TRANSFER_GAME_ID,
                rung=BOARD_ID_RAW_FIRST,
                spec=spec_for(MATCHED_DECISION_TRANSFER_GAME_ID),
                scenario=stimulus.scenarios[0],
            )

    def test_a_board_slot_that_exists_and_is_empty_refuses_and_names_the_generator(
        self, tmp_path: Path
    ) -> None:
        """The other pre-generation shape: the table is there, the messages are not.

        A separate case from the empty table above, because a different check catches it -- and because a
        file whose slots exist but hold nothing is exactly what an author writes by hand before running
        the generator.
        """
        payload, scenario_id, _entry = _boards_payload()
        boards = payload[BOARDS_FIELD]
        assert isinstance(boards, dict)
        boards[scenario_id] = {
            board_id: {"messages": [], "provenance": []} for board_id in BOARD_IDS
        }
        path = tmp_path / "transfer_stimulus.json"
        path.write_text(json.dumps({**payload, BOARDS_FIELD: boards}), encoding="utf-8")
        with pytest.raises(ValueError, match="carries no messages"):
            load_stimulus(path)
        assert (
            load_stimulus(path, allow_empty_boards=True)
            .boards[scenario_id][BOARD_ID_RAW_FIRST]
            .messages
            == ()
        )

    def test_a_missing_board_field_refuses_and_names_the_generator(self, tmp_path: Path) -> None:
        payload = synthetic_transfer_payload()
        del payload[BOARDS_FIELD]
        path = tmp_path / "transfer_stimulus.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="carries no boards field at all"):
            load_stimulus(path)

    def test_a_missing_scenario_refuses(self, tmp_path: Path) -> None:
        payload, scenario_id, _entry = _boards_payload()
        boards = payload[BOARDS_FIELD]
        assert isinstance(boards, dict)
        del boards[scenario_id]
        with pytest.raises(ValueError, match="scenarios with no board"):
            load_with(tmp_path, **{BOARDS_FIELD: boards})

    def test_a_missing_or_stray_board_id_refuses(self, tmp_path: Path) -> None:
        payload, scenario_id, entry = _boards_payload()
        boards = payload[BOARDS_FIELD]
        assert isinstance(boards, dict)
        boards[scenario_id] = {
            key: value for key, value in entry.items() if key != BOARD_ID_RAW_FIRST
        }
        with pytest.raises(ValueError, match="misses the board ids"):
            load_with(tmp_path, **{BOARDS_FIELD: boards})

    def test_a_board_of_the_wrong_message_count_refuses(self, tmp_path: Path) -> None:
        payload, scenario_id, entry = _boards_payload()
        board = dict(entry[BOARD_ID_RAW_FIRST])
        board["messages"] = list(board["messages"])[:-1]
        boards = payload[BOARDS_FIELD]
        assert isinstance(boards, dict)
        boards[scenario_id] = {**entry, BOARD_ID_RAW_FIRST: board}
        with pytest.raises(ValueError, match="messages, expected"):
            load_with(tmp_path, **{BOARDS_FIELD: boards})

    def test_provenance_missing_a_field_refuses_by_name(self, tmp_path: Path) -> None:
        payload, scenario_id, entry = _boards_payload()
        board = dict(entry[BOARD_ID_RAW_FIRST])
        provenance = [dict(item) for item in board["provenance"]]
        del provenance[0]["transport"]
        board["provenance"] = provenance
        boards = payload[BOARDS_FIELD]
        assert isinstance(boards, dict)
        boards[scenario_id] = {**entry, BOARD_ID_RAW_FIRST: board}
        with pytest.raises(ValueError, match="provenance is missing"):
            load_with(tmp_path, **{BOARDS_FIELD: boards})

    def test_a_raw_board_recording_a_paraphraser_refuses(self, tmp_path: Path) -> None:
        payload, scenario_id, entry = _boards_payload()
        board = dict(entry[BOARD_ID_RAW_FIRST])
        provenance = [dict(item) for item in board["provenance"]]
        provenance[0]["paraphraser_model_id"] = "global.openai.gpt-5.6-luna"
        board["provenance"] = provenance
        boards = payload[BOARDS_FIELD]
        assert isinstance(boards, dict)
        boards[scenario_id] = {**entry, BOARD_ID_RAW_FIRST: board}
        with pytest.raises(ValueError, match="is a raw board and records paraphraser"):
            load_with(tmp_path, **{BOARDS_FIELD: boards})

    def test_a_board_holding_the_other_side_s_content_refuses(self, tmp_path: Path) -> None:
        """The condition on every record is derived from the board id, so this is a whole cell mislabelled."""
        payload, scenario_id, entry = _boards_payload()
        board = dict(entry[BOARD_ID_RAW_FIRST])
        provenance = [dict(item) for item in board["provenance"]]
        other = next(
            model_id
            for side, model_id in BOARD_MODEL_ID_BY_SIDE.items()
            if side != BOARD_CONTENT_SIDE[BOARD_ID_RAW_FIRST]
        )
        provenance[0]["source_model_id"] = other
        board["provenance"] = provenance
        boards = payload[BOARDS_FIELD]
        assert isinstance(boards, dict)
        boards[scenario_id] = {**entry, BOARD_ID_RAW_FIRST: board}
        with pytest.raises(ValueError, match="records source_model_id"):
            load_with(tmp_path, **{BOARDS_FIELD: boards})

    def test_a_crossed_board_whose_messages_are_not_rewrites_of_its_source_refuses(
        self, tmp_path: Path
    ) -> None:
        """A crossed corner reads as one side's content in the other's wording only if it rewrote it.

        A generator that drew fresh messages instead would leave every count right, both effects wrong and
        nothing in the artifacts to say so, which is why each crossed message names its source's digest.
        """
        payload, scenario_id, entry = _boards_payload()
        crossed = next(iter(BOARD_SOURCE_BOARD))
        board = dict(entry[crossed])
        provenance = [dict(item) for item in board["provenance"]]
        provenance[0][BOARD_SOURCE_DIGEST_FIELD] = "0" * 16
        board["provenance"] = provenance
        boards = payload[BOARDS_FIELD]
        assert isinstance(boards, dict)
        boards[scenario_id] = {**entry, crossed: board}
        with pytest.raises(ValueError, match="names the source digests"):
            load_with(tmp_path, **{BOARDS_FIELD: boards})

    def test_a_sampler_recording_a_temperature_refuses(self, tmp_path: Path) -> None:
        payload, scenario_id, entry = _boards_payload()
        board = dict(entry[BOARD_ID_RAW_FIRST])
        provenance = [dict(item) for item in board["provenance"]]
        provenance[0]["sampling"] = {"max_tokens": 30_000, "temperature": 0.7}
        board["provenance"] = provenance
        boards = payload[BOARDS_FIELD]
        assert isinstance(boards, dict)
        boards[scenario_id] = {**entry, BOARD_ID_RAW_FIRST: board}
        with pytest.raises(ValueError, match="records the sampler"):
            load_with(tmp_path, **{BOARDS_FIELD: boards})

    def test_a_sampler_recording_a_cap_the_generator_never_passes_refuses(
        self, tmp_path: Path
    ) -> None:
        """The sampler is checked as a VALUE and not only as a key set.

        A cap of a few hundred tokens is the one provenance edit that changes what the messages ARE: it
        truncates them mid-sentence, and the length rule cannot tell a short message from a cut-off one.
        """
        payload, scenario_id, entry = _boards_payload()
        board = dict(entry[BOARD_ID_RAW_FIRST])
        provenance = [dict(item) for item in board["provenance"]]
        provenance[0]["sampling"] = {"max_tokens": 400}
        board["provenance"] = provenance
        boards = payload[BOARDS_FIELD]
        assert isinstance(boards, dict)
        boards[scenario_id] = {**entry, BOARD_ID_RAW_FIRST: board}
        with pytest.raises(ValueError, match="records a reply cap of"):
            load_with(tmp_path, **{BOARDS_FIELD: boards})

    def test_a_transport_outside_the_two_the_generator_draws_on_refuses(
        self, tmp_path: Path
    ) -> None:
        """The transport is checked as a value too: it says how the message was sampled.

        A value neither backend ever writes means the block was typed rather than generated, which is the
        one thing provenance exists to tell apart.
        """
        payload, scenario_id, entry = _boards_payload()
        board = dict(entry[BOARD_ID_RAW_FIRST])
        provenance = [dict(item) for item in board["provenance"]]
        provenance[0]["transport"] = "authored"
        board["provenance"] = provenance
        boards = payload[BOARDS_FIELD]
        assert isinstance(boards, dict)
        boards[scenario_id] = {**entry, BOARD_ID_RAW_FIRST: board}
        with pytest.raises(ValueError, match="records transport"):
            load_with(tmp_path, **{BOARDS_FIELD: boards})

    def test_the_recorded_cap_is_the_one_the_generator_passes(self) -> None:
        """The loader's transcription of the cap against the cap board generation actually samples at.

        Two transcriptions on purpose: a stimulus module may not import a backend, so the value the
        loader checks is written beside it. This is the pin that keeps them one number -- if the
        generator's cap ever moves, this goes red and the boards need regenerating rather than the
        frozen file quietly recording a cap nothing was drawn at.
        """
        assert BOARD_SAMPLING_MAX_TOKENS == transfer_plan.MAX_TOKENS

    def test_a_stray_key_in_the_board_table_refuses_in_the_pre_generation_window_too(
        self, tmp_path: Path
    ) -> None:
        """`allow_empty_boards` relaxes UNFILLED slots, never a table whose keys are wrong.

        A board for a scenario this file has no frame for, or a fifth board id inside a scenario, is
        nothing an author or the generator writes on the way to a filled table: it is a typo that would
        load, never be filled, and never be missed, since every reader walks the four ids it knows.
        """
        payload, scenario_id, entry = _boards_payload()
        boards = payload[BOARDS_FIELD]
        assert isinstance(boards, dict)
        strays = {**boards, "synthetic-not-a-frame": entry}
        for allow_empty in (False, True):
            with pytest.raises(ValueError, match="boards for scenarios this file does not carry"):
                load_stimulus(
                    write(tmp_path, **{BOARDS_FIELD: strays}), allow_empty_boards=allow_empty
                )
        stray_id = {**boards, scenario_id: {**entry, "board-not-a-corner": entry[BOARD_IDS[0]]}}
        for allow_empty in (False, True):
            with pytest.raises(ValueError, match="carries the stray ids"):
                load_stimulus(
                    write(tmp_path, **{BOARDS_FIELD: stray_id}), allow_empty_boards=allow_empty
                )


class TestTheBoardTasksAndLeadIn:
    """The task is the one thing in this cell that has to be unrelated to the giving decision."""

    def test_a_task_naming_a_scenario_noun_refuses(self, tmp_path: Path) -> None:
        tasks = synthetic_unrelated_tasks(synthetic_transfer_scenarios())
        first = next(iter(tasks))
        tasks[first]["task"] = "SYNTHETIC-TASK: say how you would stack the bales in a loft."
        with pytest.raises(ValueError, match="names the scenario nouns"):
            load_with(tmp_path, **{UNRELATED_TASKS_FIELD: tasks})

    def test_a_gloss_carrying_the_transfers_own_verb_refuses(self, tmp_path: Path) -> None:
        tasks = synthetic_unrelated_tasks(synthetic_transfer_scenarios())
        first = next(iter(tasks))
        tasks[first]["task_gloss"] = "say what you would set down first"
        with pytest.raises(ValueError, match="the transfer's own verbs"):
            load_with(tmp_path, **{UNRELATED_TASKS_FIELD: tasks})

    def test_a_missing_task_refuses(self, tmp_path: Path) -> None:
        tasks = synthetic_unrelated_tasks(synthetic_transfer_scenarios())
        del tasks[next(iter(tasks))]
        with pytest.raises(ValueError, match="scenarios with no task"):
            load_with(tmp_path, **{UNRELATED_TASKS_FIELD: tasks})

    def test_a_task_for_an_unknown_scenario_refuses(self, tmp_path: Path) -> None:
        tasks = synthetic_unrelated_tasks(synthetic_transfer_scenarios())
        tasks["synthetic-not-a-frame"] = {"task": "SYNTHETIC", "task_gloss": "SYNTHETIC"}
        with pytest.raises(ValueError, match="tasks for scenarios this file does not carry"):
            load_with(tmp_path, **{UNRELATED_TASKS_FIELD: tasks})

    def test_a_blank_lead_in_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="carries no board_lead_in"):
            load_with(tmp_path, **{BOARD_LEAD_IN_FIELD: "   "})

    def test_a_lead_in_naming_a_placeholder_nothing_supplies_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="names the placeholders"):
            load_with(
                tmp_path,
                **{BOARD_LEAD_IN_FIELD: "SYNTHETIC-LEAD-IN over {destination} and {task_gloss}."},
            )

    def test_a_paraphrase_instruction_without_the_message_slot_refuses(
        self, tmp_path: Path
    ) -> None:
        """Without the slot the paraphrasing side answers the instruction instead of rewriting anything."""
        with pytest.raises(ValueError, match="does not name"):
            load_with(
                tmp_path,
                **{PARAPHRASE_INSTRUCTION_FIELD: "SYNTHETIC: rewrite it in your own words."},
            )


class TestTheDoseLadder:
    """Nine rungs derived from ONE sentence, differing in the stated count and in nothing else.

    That identity is what every step of the curve is read as. Nine authored sentences would be nine
    chances for a second word to move, and the difference between two rungs would then be a read of that
    word as much as of the count -- with every artifact complete and every rate believable.
    """

    def test_every_rung_is_its_base_clause_plus_the_expanded_sentence(self, tmp_path: Path) -> None:
        stimulus = load_with(tmp_path)
        for rung, matched in RECORD_RUNG_MATCHED.items():
            base_rung = (
                "same-checkpoint" if rung == RUNG_SAME_CHECKPOINT_RECORD_0 else "different-family"
            )
            base = stimulus.clause_templates[MATCHED_DECISION_TRANSFER_GAME_ID, base_rung]
            rendered = stimulus.clause_templates[MATCHED_DECISION_TRANSFER_GAME_ID, rung]
            assert rendered == base + stimulus.track_record_sentences[rung]
            assert f" {matched} of the {TRACK_RECORD_ROUNDS} " in rendered

    def test_the_rungs_differ_in_the_numeral_alone(self, tmp_path: Path) -> None:
        stimulus = load_with(tmp_path)
        sentences = {
            matched: stimulus.track_record_sentences[record_rung_id(matched)]
            for matched in MATCHED_ROUNDS_LADDER
        }
        for left, right in pairwise(MATCHED_ROUNDS_LADDER):
            differing = [
                (one, other)
                for one, other in zip(
                    sentences[left].split(), sentences[right].split(), strict=True
                )
                if one != other
            ]
            assert differing == [(str(left), str(right))]

    def test_the_mismatch_rung_states_the_same_count_as_the_ladders_bottom(
        self, tmp_path: Path
    ) -> None:
        """Testimony over identity: the same sentence, on a copy rather than on a stranger."""
        stimulus = load_with(tmp_path)
        assert (
            stimulus.track_record_sentences[RUNG_SAME_CHECKPOINT_RECORD_0]
            == stimulus.track_record_sentences[record_rung_id(0)]
        )

    def test_a_template_without_the_matched_count_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not name"):
            load_with(
                tmp_path,
                **{
                    TRACK_RECORD_TEMPLATE_FIELD: (
                        " SYNTHETIC-RECORD: on some of the {rounds} earlier nights it came out the same."
                    )
                },
            )

    def test_a_two_sentence_template_refuses(self, tmp_path: Path) -> None:
        """The sabotage the spec names: two sentences reported under a one-sentence label."""
        with pytest.raises(ValueError, match="not exactly one sentence"):
            load_with(
                tmp_path,
                **{
                    TRACK_RECORD_TEMPLATE_FIELD: (
                        " SYNTHETIC-RECORD: on {matched} of the {rounds} earlier nights it came out "
                        "the same. It has been that way throughout."
                    )
                },
            )

    def test_a_template_spelling_its_count_in_words_refuses(self, tmp_path: Path) -> None:
        """The sabotage the spec names: a count spelled in words renders one sentence at every rung.

        Nine cells would then sample identical prompts, and the curve drawn over them would be nine
        estimates of one cell reported as a dose response.
        """
        with pytest.raises(ValueError, match="does not name"):
            load_with(
                tmp_path,
                **{
                    TRACK_RECORD_TEMPLATE_FIELD: (
                        " SYNTHETIC-RECORD: on several of the {rounds} earlier nights this was run, "
                        "what the others put down came out the same."
                    )
                },
            )

    def test_a_template_whose_second_token_moves_with_the_count_refuses(
        self, tmp_path: Path
    ) -> None:
        """A plural agreeing with the count puts that word inside every step of the curve.

        The rung is reported as a stated count and nothing else, so a sentence that reads "1 night" at one
        rung and "3 nights" at another differs in two tokens and the pairwise pin refuses it by name.
        """
        with pytest.raises(ValueError, match="do not differ in exactly one"):
            load_with(
                tmp_path,
                **{
                    TRACK_RECORD_TEMPLATE_FIELD: (
                        " SYNTHETIC-RECORD: on {matched} of the {rounds} earlier nights this was run, "
                        "what the others put down came out the same, {matched} times over."
                    ).replace("{matched} times over", "{matched} time over")
                },
            )

    def test_a_template_giving_a_reason_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="says 'because'"):
            load_with(
                tmp_path,
                **{
                    TRACK_RECORD_TEMPLATE_FIELD: (
                        " SYNTHETIC-RECORD: on {matched} of the {rounds} earlier nights it came out "
                        "the same because the others read what you read."
                    )
                },
            )

    def test_a_rounds_count_the_ladder_does_not_fit_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="this pass's ladder is stated out of"):
            load_with(tmp_path, **{TRACK_RECORD_ROUNDS_FIELD: 12})

    def test_the_ten_of_ten_rung_may_not_be_the_universal_sentence(self, tmp_path: Path) -> None:
        """Both cells are sampled in the same block, and the pair IS the quantified-versus-universal read."""
        payload = synthetic_transfer_payload()
        template = str(payload[TRACK_RECORD_TEMPLATE_FIELD])
        universal = template.replace("{matched}", str(TRACK_RECORD_ROUNDS)).replace(
            "{rounds}", str(TRACK_RECORD_ROUNDS)
        )
        with pytest.raises(ValueError, match="is the authored"):
            load_with(
                tmp_path,
                **{
                    APPENDED_SENTENCES_FIELD: {RUNG_DIFFERENT_FAMILY_TRACK_RECORD: universal},
                },
            )
