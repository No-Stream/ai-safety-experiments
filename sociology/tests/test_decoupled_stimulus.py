"""The ladder stimulus loader: one refusal per authoring constraint, plus clause resolution.

Every refusal here is a gate the design's reading depends on, so each gets its own test with the
violation planted deliberately: a rung that re-couples its counterpart, or drops the simultaneity, or
names the construct outright, would read downstream as an identity effect while being something else
entirely, and nothing about the rendered prompt would look wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from games.prompts import (
    DIFFERENT_AI_COUNTERPART_CLAUSE,
    HUMAN_COUNTERPART_CLAUSE,
    SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
    TWIN_COUNTERPART_CLAUSE,
)
from sociology.decoupled_stimulus import (
    AUTHORED_RUNGS,
    BRIEFING_PHRASE,
    CLAUSE_PREFIX,
    DECOUPLING_TAILS,
    DERIVED_RUNG_CLAUSES,
    EXPECTED_VERDICT_KEYS,
    LADDER_RUNGS,
    MIN_VALIDATION_REPLIES,
    RUNG_DIFFERENT_FAMILY,
    RUNG_PERSON,
    RUNG_SAME_CHECKPOINT,
    RUNG_SAME_FAMILY_LARGER,
    RUNG_SAME_FAMILY_SMALLER,
    RUNG_SIBLING_ADAPTER,
    clause_for,
    inserted_fragment,
    load_stimulus,
)
from sociology.tests.conftest import (
    synthetic_decoupled_payload,
    synthetic_rung_clause,
    synthetic_rung_clauses,
    write_synthetic_decoupled_stimulus,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sociology.decoupled_stimulus import DecoupledStimulus

ITS_TAIL, THEIR_TAIL = DECOUPLING_TAILS


def write_with_clause(path: Path, rung: str, clause: str) -> Path:
    clauses = synthetic_rung_clauses()
    clauses[rung] = clause
    return write_synthetic_decoupled_stimulus(path, ladder_clauses=clauses)


class TestTailsAreDerivedFromThePublicClause:
    def test_the_its_tail_is_the_same_weights_clauses_own_tail(self) -> None:
        assert SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE.endswith(ITS_TAIL)

    def test_the_their_tail_differs_only_in_the_counterparts_pronoun(self) -> None:
        assert (
            ITS_TAIL.replace("in its own way", "in their own way").replace(
                "Its decision", "Their decision"
            )
            == THEIR_TAIL
        )
        assert THEIR_TAIL != ITS_TAIL

    def test_the_seven_rungs_are_the_six_authored_ones_plus_the_public_same_weights_rung(
        self,
    ) -> None:
        assert len(LADDER_RUNGS) == 7
        assert len(AUTHORED_RUNGS) == 6
        assert set(LADDER_RUNGS) == {*AUTHORED_RUNGS, RUNG_SAME_CHECKPOINT}
        assert RUNG_SAME_CHECKPOINT not in AUTHORED_RUNGS


class TestLoadStimulusRefusals:
    def test_a_missing_file_refuses_with_the_path_and_says_gitignored(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent.json"
        with pytest.raises(FileNotFoundError, match="gitignored") as raised:
            load_stimulus(missing)
        assert str(missing) in str(raised.value)

    def test_a_version_mismatch_refuses(self, tmp_path: Path) -> None:
        path = write_synthetic_decoupled_stimulus(
            tmp_path / "s.json", version="decoupled-ladder-stimulus-v0"
        )
        with pytest.raises(ValueError, match="version"):
            load_stimulus(path)

    def test_a_missing_rung_refuses_by_name(self, tmp_path: Path) -> None:
        clauses = synthetic_rung_clauses()
        del clauses["person"]
        path = write_synthetic_decoupled_stimulus(tmp_path / "s.json", ladder_clauses=clauses)
        with pytest.raises(ValueError, match="person"):
            load_stimulus(path)

    def test_a_blank_rung_refuses(self, tmp_path: Path) -> None:
        path = write_with_clause(tmp_path / "s.json", "sibling-adapter", "   ")
        with pytest.raises(ValueError, match="blank clause"):
            load_stimulus(path)

    def test_a_rung_that_does_not_open_with_the_shared_prefix_refuses(self, tmp_path: Path) -> None:
        clause = synthetic_rung_clause("sibling-adapter").replace(
            CLAUSE_PREFIX, "your counterpart is "
        )
        path = write_with_clause(tmp_path / "s.json", "sibling-adapter", clause)
        with pytest.raises(ValueError, match="does not open with"):
            load_stimulus(path)

    def test_a_rung_that_drops_the_simultaneity_phrase_refuses(self, tmp_path: Path) -> None:
        clause = f"you are matched with SYNTHETIC-COUNTERPART-sibling, {ITS_TAIL}"
        path = write_with_clause(tmp_path / "s.json", "sibling-adapter", clause)
        with pytest.raises(ValueError, match="simultaneity"):
            load_stimulus(path)

    def test_a_rung_whose_tail_recouples_the_counterpart_refuses(self, tmp_path: Path) -> None:
        clause = (
            f"you are matched with SYNTHETIC-COUNTERPART-sibling, {BRIEFING_PHRASE}, and deciding "
            f"on it the same way you are."
        )
        path = write_with_clause(tmp_path / "s.json", "sibling-adapter", clause)
        with pytest.raises(ValueError, match="shared decoupling"):
            load_stimulus(path)

    def test_a_rung_that_names_the_construct_refuses(self, tmp_path: Path) -> None:
        clause = (
            f"you are matched with a SYNTHETIC-COUNTERPART whose choices are not correlated with "
            f"yours, {BRIEFING_PHRASE}, {ITS_TAIL}"
        )
        path = write_with_clause(tmp_path / "s.json", "different-family", clause)
        with pytest.raises(ValueError, match="loaded vocabulary"):
            load_stimulus(path)

    def test_a_two_paragraph_rung_refuses(self, tmp_path: Path) -> None:
        clause = (
            f"you are matched with SYNTHETIC-COUNTERPART-person, {BRIEFING_PHRASE}.\n\n"
            f"A second paragraph the audit would not delete, {THEIR_TAIL}"
        )
        path = write_with_clause(tmp_path / "s.json", "person", clause)
        with pytest.raises(ValueError, match="paragraph break"):
            load_stimulus(path)

    def test_an_empty_rubric_refuses(self, tmp_path: Path) -> None:
        path = write_synthetic_decoupled_stimulus(tmp_path / "s.json", judge_instructions="  ")
        with pytest.raises(ValueError, match="judge_instructions"):
            load_stimulus(path)

    def test_too_few_validation_replies_refuses_and_names_the_floor(self, tmp_path: Path) -> None:
        """A judge calibrated on a handful of cases is an uncalibrated judge with a report attached."""
        replies = synthetic_decoupled_payload()["validation_replies"]
        assert isinstance(replies, list)
        thin = replies[: MIN_VALIDATION_REPLIES - 1]
        path = write_synthetic_decoupled_stimulus(tmp_path / "s.json", validation_replies=thin)
        with pytest.raises(ValueError, match=f"fewer than {MIN_VALIDATION_REPLIES}"):
            load_stimulus(path)

    def test_exactly_the_floor_is_accepted(self, tmp_path: Path) -> None:
        replies = synthetic_decoupled_payload()["validation_replies"]
        assert isinstance(replies, list)
        path = write_synthetic_decoupled_stimulus(
            tmp_path / "s.json", validation_replies=replies[:MIN_VALIDATION_REPLIES]
        )
        assert len(load_stimulus(path).validation_replies) == MIN_VALIDATION_REPLIES

    def test_a_validation_reply_missing_an_expected_field_names_the_file_and_the_reply(
        self, tmp_path: Path
    ) -> None:
        """It used to be a bare KeyError: a field name, with no file and no reply behind it."""
        replies = synthetic_decoupled_payload()["validation_replies"]
        assert isinstance(replies, list)
        short = dict(replies[2])
        expected = dict(short["expected"])
        del expected["ev_arithmetic"]
        short["expected"] = expected
        path = write_synthetic_decoupled_stimulus(
            tmp_path / "s.json", validation_replies=[short, *replies[3:], *replies[:2]]
        )
        with pytest.raises(ValueError, match="ev_arithmetic") as raised:
            load_stimulus(path)
        assert str(path) in str(raised.value)
        assert str(short["name"]) in str(raised.value)


class TestTheFourComputableRungsAreRecomputed:
    """Each rung whose text is an identity, sabotaged one at a time and required to refuse.

    None of these violations is visible in a rendered prompt: every planted clause below still opens
    with the shared prefix, still carries the simultaneity phrase, still ends on a shared decoupling
    tail and still passes the vocabulary guard. What each one breaks is a contrast the readout
    computes -- so the refusal has to live in the loader, not in a reviewer's eye.
    """

    def test_a_different_family_rung_that_is_not_the_public_clause_plus_the_denial_refuses(
        self, tmp_path: Path
    ) -> None:
        drifted = DERIVED_RUNG_CLAUSES[RUNG_DIFFERENT_FAMILY].replace(
            "a different AI system", "a SYNTHETIC-OTHER-SYSTEM"
        )
        path = write_with_clause(tmp_path / "s.json", RUNG_DIFFERENT_FAMILY, drifted)
        with pytest.raises(ValueError, match="shared denial sentence"):
            load_stimulus(path)

    def test_a_person_rung_that_is_not_the_public_human_clause_plus_the_denial_refuses(
        self, tmp_path: Path
    ) -> None:
        drifted = DERIVED_RUNG_CLAUSES[RUNG_PERSON].replace("a person", "a SYNTHETIC-HUMAN")
        path = write_with_clause(tmp_path / "s.json", RUNG_PERSON, drifted)
        with pytest.raises(ValueError, match="shared denial sentence"):
            load_stimulus(path)

    def test_the_derivations_are_the_public_constants_plus_one_sentence(self) -> None:
        """Stated here too, because the loader's check is only as good as what it computes."""
        its_tail, their_tail = DECOUPLING_TAILS
        assert DERIVED_RUNG_CLAUSES[RUNG_DIFFERENT_FAMILY] == (
            DIFFERENT_AI_COUNTERPART_CLAUSE.removesuffix("and deciding on it in its own way.")
            + its_tail
        )
        assert DERIVED_RUNG_CLAUSES[RUNG_PERSON] == (
            HUMAN_COUNTERPART_CLAUSE.removesuffix("and deciding on it in their own way.")
            + their_tail
        )
        assert DERIVED_RUNG_CLAUSES[RUNG_DIFFERENT_FAMILY].startswith(
            DIFFERENT_AI_COUNTERPART_CLAUSE
        )

    def test_a_sibling_adapter_rung_with_two_edits_refuses(self, tmp_path: Path) -> None:
        """One inserted fragment is the whole claim; a second edit makes it a rewording instead."""
        two_edits = synthetic_rung_clause(RUNG_SIBLING_ADAPTER).replace(
            "another instance", "a SYNTHETIC-INSTANCE"
        )
        path = write_with_clause(tmp_path / "s.json", RUNG_SIBLING_ADAPTER, two_edits)
        with pytest.raises(ValueError, match="exactly one fragment inserted"):
            load_stimulus(path)

    def test_a_sibling_adapter_rung_identical_to_the_public_clause_refuses(
        self, tmp_path: Path
    ) -> None:
        """Zero insertions is not one: it would sample `same-checkpoint` twice under two names."""
        path = write_with_clause(
            tmp_path / "s.json",
            RUNG_SIBLING_ADAPTER,
            SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
        )
        with pytest.raises(ValueError, match="exactly one fragment inserted"):
            load_stimulus(path)

    def test_the_inserted_fragment_is_reported_for_a_real_one_insertion(self) -> None:
        fragment = inserted_fragment(
            SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
            synthetic_rung_clause(RUNG_SIBLING_ADAPTER),
        )
        assert fragment is not None
        assert (
            synthetic_rung_clause(RUNG_SIBLING_ADAPTER).replace(fragment, "", 1)
            == SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE
        )

    def test_two_same_family_rungs_differing_in_more_than_the_size_words_refuse(
        self, tmp_path: Path
    ) -> None:
        """Their difference reads as the capability direction, so any other change lands in it."""
        drifted = synthetic_rung_clause(RUNG_SAME_FAMILY_LARGER).replace(
            "built to the same design", "built to a SYNTHETIC-DESIGN"
        )
        path = write_with_clause(tmp_path / "s.json", RUNG_SAME_FAMILY_LARGER, drifted)
        with pytest.raises(ValueError, match="more than the size words"):
            load_stimulus(path)

    def test_the_size_words_substitution_maps_the_smaller_rung_onto_the_larger_one(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        smaller = decoupled_stimulus.ladder_clauses[RUNG_SAME_FAMILY_SMALLER]
        larger = decoupled_stimulus.ladder_clauses[RUNG_SAME_FAMILY_LARGER]
        assert smaller != larger
        assert smaller.replace("smaller", "larger").replace("fewer", "more") == larger


class TestLoadedStimulus:
    def test_every_authored_rung_loads_and_ends_on_a_shared_tail(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        assert set(decoupled_stimulus.ladder_clauses) == set(AUTHORED_RUNGS)
        for clause in decoupled_stimulus.ladder_clauses.values():
            assert clause.endswith(DECOUPLING_TAILS)

    def test_the_person_rung_carries_the_pronoun_form_of_the_tail(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        assert decoupled_stimulus.ladder_clauses["person"].endswith(THEIR_TAIL)

    def test_validation_replies_load_with_all_seven_expected_fields(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        replies = decoupled_stimulus.validation_replies
        assert len(replies) >= MIN_VALIDATION_REPLIES
        for reply in replies:
            assert tuple(reply.expected) == EXPECTED_VERDICT_KEYS

    def test_the_digest_tracks_the_files_content(self, tmp_path: Path) -> None:
        first = load_stimulus(write_synthetic_decoupled_stimulus(tmp_path / "a.json"))
        second = load_stimulus(
            write_synthetic_decoupled_stimulus(
                tmp_path / "b.json", judge_instructions="SYNTHETIC-RUBRIC: something else."
            )
        )
        assert first.digest != second.digest


class TestClauseFor:
    def test_anchor_cells_resolve_to_the_tracked_registry(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        assert clause_for("twin", decoupled_stimulus) == TWIN_COUNTERPART_CLAUSE
        assert clause_for("different-ai", decoupled_stimulus) == DIFFERENT_AI_COUNTERPART_CLAUSE
        assert (
            clause_for("same-weights-uncorrelated", decoupled_stimulus)
            == SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE
        )

    def test_same_checkpoint_is_the_anchors_decoupled_same_weights_cell_byte_for_byte(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        assert clause_for(RUNG_SAME_CHECKPOINT, decoupled_stimulus) == clause_for(
            "same-weights-uncorrelated", decoupled_stimulus
        )

    def test_authored_rungs_resolve_to_the_loaded_file(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        for rung in AUTHORED_RUNGS:
            assert clause_for(rung, decoupled_stimulus) == decoupled_stimulus.ladder_clauses[rung]

    def test_an_unknown_name_refuses(self, decoupled_stimulus: DecoupledStimulus) -> None:
        with pytest.raises(ValueError, match="neither a registered counterpart framing"):
            clause_for("not-a-cell", decoupled_stimulus)
