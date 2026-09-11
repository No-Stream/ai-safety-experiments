"""The runtime counterpart-framing loader: one refusal per authoring constraint, plus resolution.

A wave that needs framings the tracked registry does not carry authors them in a gitignored file
(the kin framings, and verbatim copies of the sociology ladder's rungs), and this loader is the only
way they reach a render. Every refusal it runs is a property the framing sweep's reading depends on
rather than tidiness, so each gets a test with the violation planted deliberately: a clause that
re-couples its counterpart, drops the simultaneity, opens in another register, names the construct
outright or states one of the numbers the outcome table prints would render, grade and summarise
without a single artifact looking wrong, and the cell would read as an identity effect while being
something else.

The clauses here are synthetic and built from the derived constants, never authored prose: the real
file is stimulus that must never enter this public repository, and a fixture carrying its text would
be the same leak by another route.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from games.framing_stimulus import (
    BRIEFING_PHRASE,
    CLAUSE_PREFIX,
    DECOUPLING_TAILS,
    DICTATOR_RECIPIENT_CLAUSES_FIELD,
    FRAMINGS_VERSION,
    NON_PROMPT_FIELDS,
    PROMPT_AFFECTING_FIELDS,
    RECIPIENT_CONDITIONS,
    RECIPIENT_IN_HARDSHIP,
    RECIPIENT_WELL_OFF,
    framing_states_coupling,
    load_dictator_recipient_clauses,
    load_framings,
    resolve_framing_clause,
)
from games.prompts import (
    ANOTHER_AI_COUNTERPART_CLAUSE,
    COUNTERPART_FRAMINGS,
    COUNTERPART_PARAGRAPH_MARKER,
    COUPLING_ASSERTION_PHRASES,
    DIFFERENT_AI_COUNTERPART_CLAUSE,
    FRAMING_DIFFERENT_AI_COUPLED,
    FRAMING_HUMAN,
    FRAMING_SAME_WEIGHTS_UNCORRELATED,
    FRAMING_TWIN,
    FRAMING_UNSTATED,
    HUMAN_COUNTERPART_CLAUSE,
    SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
)

if TYPE_CHECKING:
    from pathlib import Path

ITS_TAIL, THEIR_TAIL = DECOUPLING_TAILS

DEPENDENT = "dependent"
STRANGER_IN_NEED = "stranger-in-need"


def synthetic_clause(framing_id: str, *, tail: str = ITS_TAIL) -> str:
    """One synthetic counterpart clause that satisfies every constraint the loader checks."""
    return f"{CLAUSE_PREFIX}SYNTHETIC-COUNTERPART-{framing_id}, {BRIEFING_PHRASE}, {tail}"


def synthetic_recipient_clause(condition: str) -> str:
    """One synthetic recipient paragraph for the unilateral-split trap cells."""
    return f"The other side of this SYNTHETIC-RECIPIENT-{condition} keeps house on their own."


def synthetic_payload(
    *,
    framings: dict[str, str] | None = None,
    recipients: dict[str, str] | None = None,
    version: str = FRAMINGS_VERSION,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The whole file's payload, defaulting to the two kin ids and both recipient conditions."""
    clauses = (
        framings
        if framings is not None
        else {
            framing_id: synthetic_clause(framing_id) for framing_id in (DEPENDENT, STRANGER_IN_NEED)
        }
    )
    recipient_clauses = (
        recipients
        if recipients is not None
        else {
            condition: synthetic_recipient_clause(condition) for condition in RECIPIENT_CONDITIONS
        }
    )
    payload: dict[str, Any] = {
        "version": version,
        "framings": [
            {"framing_id": framing_id, "clause": clause} for framing_id, clause in clauses.items()
        ],
        "dictator_recipient_clauses": recipient_clauses,
    }
    payload.update(extra or {})
    return payload


def write_framings_file(path: Path, **kwargs: Any) -> Path:
    """Write a synthetic framings file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(synthetic_payload(**kwargs), indent=2), encoding="utf-8")
    return path


def write_with_clause(path: Path, framing_id: str, clause: str) -> Path:
    """Write a file whose one named framing carries `clause` and whose others are well formed."""
    framings = {
        DEPENDENT: synthetic_clause(DEPENDENT),
        STRANGER_IN_NEED: synthetic_clause(STRANGER_IN_NEED),
    }
    framings[framing_id] = clause
    return write_framings_file(path, framings=framings)


class TestTheSharedConstantsAreDerivedFromThePublicClauses:
    def test_every_registered_clause_opens_with_the_shared_prefix(self) -> None:
        for framing_id, clause in COUNTERPART_FRAMINGS.items():
            if clause is None:
                continue
            assert clause.startswith(CLAUSE_PREFIX), framing_id

    def test_the_briefing_phrase_is_carried_once_by_every_identity_clause(self) -> None:
        for clause in (
            SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
            ANOTHER_AI_COUNTERPART_CLAUSE,
            DIFFERENT_AI_COUNTERPART_CLAUSE,
            HUMAN_COUNTERPART_CLAUSE,
        ):
            assert clause.count(BRIEFING_PHRASE) == 1

    def test_the_its_tail_is_the_same_weights_clauses_own_tail(self) -> None:
        assert SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE.endswith(ITS_TAIL)

    def test_the_their_tail_differs_from_it_in_the_counterparts_pronoun_alone(self) -> None:
        assert THEIR_TAIL != ITS_TAIL
        assert (
            ITS_TAIL.replace("in its own way", "in their own way").replace(
                "Its decision", "Their decision"
            )
            == THEIR_TAIL
        )

    def test_every_top_level_field_is_filed_on_one_side_of_the_digest(self) -> None:
        assert not set(PROMPT_AFFECTING_FIELDS) & set(NON_PROMPT_FIELDS)
        assert "framings" in PROMPT_AFFECTING_FIELDS
        assert "version" in PROMPT_AFFECTING_FIELDS


class TestAValidFileLoads:
    def test_both_kin_clauses_load_under_their_own_ids(self, tmp_path: Path) -> None:
        loaded = load_framings(write_framings_file(tmp_path / "framings.json"))
        assert loaded.framing_ids == (DEPENDENT, STRANGER_IN_NEED)
        assert loaded.clauses[DEPENDENT] == synthetic_clause(DEPENDENT)
        assert loaded.path == tmp_path / "framings.json"
        assert len(loaded.digest) == 16

    def test_the_their_form_of_the_tail_is_accepted(self, tmp_path: Path) -> None:
        clause = synthetic_clause(STRANGER_IN_NEED, tail=THEIR_TAIL)
        loaded = load_framings(
            write_with_clause(tmp_path / "framings.json", STRANGER_IN_NEED, clause)
        )
        assert loaded.clauses[STRANGER_IN_NEED] == clause

    def test_the_clauses_mapping_is_a_copy_the_caller_cannot_write_back_through(
        self, tmp_path: Path
    ) -> None:
        loaded = load_framings(write_framings_file(tmp_path / "framings.json"))
        loaded.clauses[DEPENDENT] = "SYNTHETIC-OVERWRITE"
        assert loaded.clauses[DEPENDENT] == synthetic_clause(DEPENDENT)


class TestTheLoaderRefusesEveryAuthoringViolation:
    def test_an_absent_file_is_refused_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="gitignored"):
            load_framings(tmp_path / "absent.json")

    def test_a_wrong_version_string_is_refused(self, tmp_path: Path) -> None:
        path = write_framings_file(tmp_path / "framings.json", version="games-runtime-framings-v0")
        with pytest.raises(ValueError, match="version"):
            load_framings(path)

    def test_an_empty_framing_roster_is_refused(self, tmp_path: Path) -> None:
        path = write_framings_file(tmp_path / "framings.json", framings={})
        with pytest.raises(ValueError, match="carries no framings"):
            load_framings(path)

    def test_a_clause_naming_the_literature_is_refused(self, tmp_path: Path) -> None:
        clause = (
            f"{CLAUSE_PREFIX}SYNTHETIC-COUNTERPART-cooperating-partner, {BRIEFING_PHRASE}, "
            f"{ITS_TAIL}"
        )
        path = write_with_clause(tmp_path / "framings.json", DEPENDENT, clause)
        with pytest.raises(ValueError, match="loaded vocabulary"):
            load_framings(path)

    def test_a_clause_missing_the_briefing_phrase_is_refused(self, tmp_path: Path) -> None:
        clause = f"{CLAUSE_PREFIX}SYNTHETIC-COUNTERPART-{DEPENDENT}, {ITS_TAIL}"
        path = write_with_clause(tmp_path / "framings.json", DEPENDENT, clause)
        with pytest.raises(ValueError, match="simultaneity"):
            load_framings(path)

    def test_a_clause_ending_on_neither_decoupling_tail_is_refused(self, tmp_path: Path) -> None:
        clause = f"{CLAUSE_PREFIX}SYNTHETIC-COUNTERPART-{DEPENDENT}, {BRIEFING_PHRASE}."
        path = write_with_clause(tmp_path / "framings.json", DEPENDENT, clause)
        with pytest.raises(ValueError, match="decoupling tail"):
            load_framings(path)

    def test_a_clause_opening_in_another_register_is_refused(self, tmp_path: Path) -> None:
        clause = (
            f"The other side is SYNTHETIC-COUNTERPART-{DEPENDENT}, {BRIEFING_PHRASE}, {ITS_TAIL}"
        )
        path = write_with_clause(tmp_path / "framings.json", DEPENDENT, clause)
        with pytest.raises(ValueError, match="does not open with"):
            load_framings(path)

    @pytest.mark.parametrize("phrase", COUPLING_ASSERTION_PHRASES)
    def test_a_clause_asserting_the_coupling_is_refused(self, tmp_path: Path, phrase: str) -> None:
        """The assertion sits inside an otherwise well-formed clause, tail included.

        Appending it after the tail instead would fail the tail check, whose message also carries
        the word "coupling", and this test would pass while the coupling refusal did nothing.
        """
        clause = (
            f"{CLAUSE_PREFIX}SYNTHETIC-COUNTERPART-{DEPENDENT}, {BRIEFING_PHRASE}, {phrase} "
            f"{ITS_TAIL}"
        )
        with pytest.raises(ValueError, match="asserts the counterpart coupling"):
            load_framings(write_with_clause(tmp_path / "framings.json", DEPENDENT, clause))

    def test_an_id_colliding_with_a_registered_framing_is_refused(self, tmp_path: Path) -> None:
        path = write_framings_file(
            tmp_path / "framings.json",
            framings={FRAMING_HUMAN: synthetic_clause(FRAMING_HUMAN)},
        )
        with pytest.raises(ValueError, match="registered counterpart framing"):
            load_framings(path)

    def test_a_repeated_id_is_refused(self, tmp_path: Path) -> None:
        payload = synthetic_payload()
        payload["framings"].append(payload["framings"][0])
        (tmp_path / "framings.json").write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="more than once"):
            load_framings(tmp_path / "framings.json")

    def test_an_id_that_cannot_be_a_prompt_id_segment_is_refused(self, tmp_path: Path) -> None:
        path = write_framings_file(
            tmp_path / "framings.json", framings={"Dependent Kin": synthetic_clause(DEPENDENT)}
        )
        with pytest.raises(ValueError, match="does not match"):
            load_framings(path)

    def test_a_clause_carrying_a_paragraph_break_is_refused(self, tmp_path: Path) -> None:
        clause = synthetic_clause(DEPENDENT).replace(
            ", " + BRIEFING_PHRASE, "\n\n" + BRIEFING_PHRASE
        )
        path = write_with_clause(tmp_path / "framings.json", DEPENDENT, clause)
        with pytest.raises(ValueError, match="paragraph break"):
            load_framings(path)

    def test_a_clause_stating_a_number_the_renderer_prints_is_refused(self, tmp_path: Path) -> None:
        clause = (
            f"{CLAUSE_PREFIX}SYNTHETIC-COUNTERPART-{DEPENDENT} with 3 in their charge, "
            f"{BRIEFING_PHRASE}, {ITS_TAIL}"
        )
        path = write_with_clause(tmp_path / "framings.json", DEPENDENT, clause)
        with pytest.raises(ValueError, match="numeral"):
            load_framings(path)

    def test_a_clause_naming_the_payoff_unit_is_refused(self, tmp_path: Path) -> None:
        clause = (
            f"{CLAUSE_PREFIX}SYNTHETIC-COUNTERPART-{DEPENDENT} who is short of points, "
            f"{BRIEFING_PHRASE}, {ITS_TAIL}"
        )
        path = write_with_clause(tmp_path / "framings.json", DEPENDENT, clause)
        with pytest.raises(ValueError, match="payoff unit"):
            load_framings(path)

    def test_a_blank_clause_is_refused(self, tmp_path: Path) -> None:
        path = write_with_clause(tmp_path / "framings.json", DEPENDENT, "   ")
        with pytest.raises(ValueError, match="blank clause"):
            load_framings(path)

    def test_an_entry_missing_its_clause_is_refused(self, tmp_path: Path) -> None:
        payload = synthetic_payload()
        del payload["framings"][0]["clause"]
        (tmp_path / "framings.json").write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="missing"):
            load_framings(tmp_path / "framings.json")

    def test_an_unclassified_top_level_field_is_refused(self, tmp_path: Path) -> None:
        path = write_framings_file(tmp_path / "framings.json", extra={"framing": []})
        with pytest.raises(ValueError, match="classified"):
            load_framings(path)


class TestTheDigestNamesWhatReachedTheModel:
    def test_it_moves_when_a_clause_moves(self, tmp_path: Path) -> None:
        first = load_framings(write_framings_file(tmp_path / "a.json"))
        moved = (
            f"{CLAUSE_PREFIX}SYNTHETIC-COUNTERPART-{DEPENDENT}-revised, {BRIEFING_PHRASE}, "
            f"{ITS_TAIL}"
        )
        second = load_framings(write_with_clause(tmp_path / "b.json", DEPENDENT, moved))
        assert first.digest != second.digest

    def test_it_stands_still_when_only_the_judge_validation_replies_move(
        self, tmp_path: Path
    ) -> None:
        first = load_framings(write_framings_file(tmp_path / "a.json"))
        second = load_framings(
            write_framings_file(
                tmp_path / "b.json", extra={"validation_replies": [{"name": "one"}]}
            )
        )
        assert first.digest == second.digest


class TestTheRecipientClauses:
    def test_both_conditions_load_and_are_keyed_by_their_own_ids(self, tmp_path: Path) -> None:
        loaded = load_dictator_recipient_clauses(write_framings_file(tmp_path / "framings.json"))
        assert loaded.well_off == synthetic_recipient_clause(RECIPIENT_WELL_OFF)
        assert loaded.in_hardship == synthetic_recipient_clause(RECIPIENT_IN_HARDSHIP)
        assert set(loaded.clause_by_condition) == set(RECIPIENT_CONDITIONS)
        assert loaded.digest == load_framings(tmp_path / "framings.json").digest

    def test_an_absent_section_is_refused(self, tmp_path: Path) -> None:
        payload = synthetic_payload()
        del payload["dictator_recipient_clauses"]
        (tmp_path / "framings.json").write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="dictator_recipient_clauses"):
            load_dictator_recipient_clauses(tmp_path / "framings.json")

    def test_a_missing_condition_is_refused(self, tmp_path: Path) -> None:
        path = write_framings_file(
            tmp_path / "framings.json",
            recipients={RECIPIENT_WELL_OFF: synthetic_recipient_clause(RECIPIENT_WELL_OFF)},
        )
        with pytest.raises(ValueError, match=RECIPIENT_IN_HARDSHIP):
            load_dictator_recipient_clauses(path)

    def test_an_unknown_condition_is_refused(self, tmp_path: Path) -> None:
        recipients = {
            condition: synthetic_recipient_clause(condition) for condition in RECIPIENT_CONDITIONS
        }
        recipients["comfortable"] = synthetic_recipient_clause("comfortable")
        path = write_framings_file(tmp_path / "framings.json", recipients=recipients)
        with pytest.raises(ValueError, match="comfortable"):
            load_dictator_recipient_clauses(path)

    def test_a_recipient_clause_naming_the_literature_is_refused(self, tmp_path: Path) -> None:
        path = write_framings_file(
            tmp_path / "framings.json",
            recipients={
                RECIPIENT_WELL_OFF: "This SYNTHETIC-RECIPIENT keeps a stag of their own.",
                RECIPIENT_IN_HARDSHIP: synthetic_recipient_clause(RECIPIENT_IN_HARDSHIP),
            },
        )
        with pytest.raises(ValueError, match="loaded vocabulary"):
            load_dictator_recipient_clauses(path)

    def test_a_recipient_clause_asserting_the_coupling_is_refused(self, tmp_path: Path) -> None:
        path = write_framings_file(
            tmp_path / "framings.json",
            recipients={
                RECIPIENT_WELL_OFF: f"Their figures {COUPLING_ASSERTION_PHRASES[0]}",
                RECIPIENT_IN_HARDSHIP: synthetic_recipient_clause(RECIPIENT_IN_HARDSHIP),
            },
        )
        with pytest.raises(ValueError, match="coupling"):
            load_dictator_recipient_clauses(path)

    def test_a_recipient_clause_already_carrying_the_paragraph_marker_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The renderer wraps the clause itself, so a clause carrying the marker states it twice.

        The one-inserted-paragraph audit still passes on that prompt (the doubled marker is one
        section and deleting it still reproduces the stem), so nothing downstream notices.
        """
        path = write_framings_file(
            tmp_path / "framings.json",
            recipients={
                RECIPIENT_WELL_OFF: (
                    f"{COUNTERPART_PARAGRAPH_MARKER}"
                    f"{synthetic_recipient_clause(RECIPIENT_WELL_OFF)}"
                ),
                RECIPIENT_IN_HARDSHIP: synthetic_recipient_clause(RECIPIENT_IN_HARDSHIP),
            },
        )
        with pytest.raises(ValueError, match="marker"):
            load_dictator_recipient_clauses(path)

    def test_two_identical_recipient_clauses_are_refused(self, tmp_path: Path) -> None:
        """The cell is read as one description against the other, and identical text makes that
        difference zero by construction while every count in the summary still adds up."""
        shared = synthetic_recipient_clause(RECIPIENT_WELL_OFF)
        path = write_framings_file(
            tmp_path / "framings.json",
            recipients={RECIPIENT_WELL_OFF: shared, RECIPIENT_IN_HARDSHIP: f"  {shared}  "},
        )
        with pytest.raises(ValueError, match="identical"):
            load_dictator_recipient_clauses(path)

    def test_a_recipient_clause_carrying_a_paragraph_break_is_refused(self, tmp_path: Path) -> None:
        """Two paragraphs inserted where the one-insertion audit counts one, and only the first
        carries the marker the audit deletes, so the stem no longer reproduces."""
        path = write_framings_file(
            tmp_path / "framings.json",
            recipients={
                RECIPIENT_WELL_OFF: (
                    f"{synthetic_recipient_clause(RECIPIENT_WELL_OFF)}\n\nThey keep a reserve too."
                ),
                RECIPIENT_IN_HARDSHIP: synthetic_recipient_clause(RECIPIENT_IN_HARDSHIP),
            },
        )
        with pytest.raises(ValueError, match="paragraph break"):
            load_dictator_recipient_clauses(path)

    def test_a_blank_recipient_clause_is_refused(self, tmp_path: Path) -> None:
        """A blank paragraph renders the plain split under a prompt_id claiming a description, so
        the two versions would be one cell measured twice."""
        path = write_framings_file(
            tmp_path / "framings.json",
            recipients={
                RECIPIENT_WELL_OFF: "   ",
                RECIPIENT_IN_HARDSHIP: synthetic_recipient_clause(RECIPIENT_IN_HARDSHIP),
            },
        )
        with pytest.raises(ValueError, match="blank"):
            load_dictator_recipient_clauses(path)

    def test_an_absent_file_is_refused_by_name_here_too(self, tmp_path: Path) -> None:
        """Both loaders read the same file through `_read_payload`, so both refuse its absence."""
        with pytest.raises(FileNotFoundError, match="gitignored"):
            load_dictator_recipient_clauses(tmp_path / "nowhere.json")

    def test_the_section_key_is_named_once(self) -> None:
        """The loader, the digest's field list and every writer of the file spell it from here."""
        assert DICTATOR_RECIPIENT_CLAUSES_FIELD in PROMPT_AFFECTING_FIELDS

    def test_a_recipient_clause_stating_a_resource_count_is_refused(self, tmp_path: Path) -> None:
        path = write_framings_file(
            tmp_path / "framings.json",
            recipients={
                RECIPIENT_WELL_OFF: "This SYNTHETIC-RECIPIENT has 12 of their own put by.",
                RECIPIENT_IN_HARDSHIP: synthetic_recipient_clause(RECIPIENT_IN_HARDSHIP),
            },
        )
        with pytest.raises(ValueError, match="numeral"):
            load_dictator_recipient_clauses(path)


class TestResolutionAndCoupling:
    def test_the_registry_answers_first(self, tmp_path: Path) -> None:
        loaded = load_framings(write_framings_file(tmp_path / "framings.json"))
        assert resolve_framing_clause(FRAMING_TWIN, loaded) == COUNTERPART_FRAMINGS[FRAMING_TWIN]
        assert resolve_framing_clause(FRAMING_UNSTATED, loaded) is None

    def test_the_loaded_file_answers_second(self, tmp_path: Path) -> None:
        loaded = load_framings(write_framings_file(tmp_path / "framings.json"))
        assert resolve_framing_clause(DEPENDENT, loaded) == synthetic_clause(DEPENDENT)

    def test_an_id_in_neither_place_is_refused_by_name(self, tmp_path: Path) -> None:
        loaded = load_framings(write_framings_file(tmp_path / "framings.json"))
        with pytest.raises(ValueError, match="telepathic"):
            resolve_framing_clause("telepathic", loaded)
        with pytest.raises(ValueError, match=DEPENDENT):
            resolve_framing_clause(DEPENDENT, None)

    def test_coupling_is_read_off_the_registry_for_a_registered_framing(
        self, tmp_path: Path
    ) -> None:
        loaded = load_framings(write_framings_file(tmp_path / "framings.json"))
        assert framing_states_coupling(FRAMING_TWIN, loaded) is True
        assert framing_states_coupling(FRAMING_DIFFERENT_AI_COUPLED, loaded) is True
        assert framing_states_coupling(FRAMING_SAME_WEIGHTS_UNCORRELATED, loaded) is False

    def test_a_runtime_framing_states_no_coupling_because_the_loader_refuses_one(
        self, tmp_path: Path
    ) -> None:
        loaded = load_framings(write_framings_file(tmp_path / "framings.json"))
        for framing_id in loaded.framing_ids:
            assert framing_states_coupling(framing_id, loaded) is False

    def test_an_unknown_framing_still_raises_rather_than_answering_false(
        self, tmp_path: Path
    ) -> None:
        loaded = load_framings(write_framings_file(tmp_path / "framings.json"))
        with pytest.raises(ValueError, match="telepathic"):
            framing_states_coupling("telepathic", loaded)
