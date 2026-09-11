"""Offline tests for the coding-prompt completion budget and the table it is read off.

The sibling of ``games/tests/test_games_termination.py``, for the censored coding distribution. What
these check is that the table cannot be internally inconsistent -- every field in it was transcribed
by a human off a screen readout, and the six refusals in ``__post_init__`` are the only thing
standing between a mistyped digit and a budget that truncates rollouts nobody will think to blame.

Two of the derived readings reach the run record (``derived.coding_termination_stats``), so they are
asserted against hand-computed values rather than recomputed from the same expression the code uses.
And the family-mismatch warning is tested because it is the module's only signal that the shipped
budget was measured on prompts no caller sends: everything else about that gap is silent.
"""

from __future__ import annotations

import json
import logging

import pytest

from games.termination import MEASURED_TERMINATION_BUDGET, MEASURED_TERMINATION_STATS_BY_MODEL
from reward_hacking import train_termination
from reward_hacking.train_termination import (
    BUDGET_SOURCE_GAME_FLOOR,
    BUDGET_SOURCE_MEASURED,
    BUDGET_SOURCE_PROVISIONAL,
    CODING_GRADER_PROMPTS,
    MAX_TRUNCATED_FRACTION,
    MEASURED_CODING_TERMINATION_BUDGET_BY_MODEL,
    MEASURED_CODING_TERMINATION_STATS_BY_MODEL,
    MEASURED_SINGLE_TURN_CAP_HITS_BY_SOURCE_FAMILY,
    PROVISIONAL_CODING_BUDGET_BY_MODEL,
    SINGLE_TURN_CODING_GRADER_PROMPTS,
    CodingTerminationStats,
    ProvisionalCodingBudget,
    SingleTurnCapHits,
    assert_not_below_game_floor,
    completion_budget_provenance,
    required_coding_completion_budget,
)

SCREENED_MODEL = "Qwen/Qwen3.5-4B"

# The 9B's game floor is 32,768, the highest in that table, so a coding budget below it is the
# downward disagreement `assert_not_below_game_floor` exists to refuse.
HIGHEST_GAME_FLOOR_MODEL = "Qwen/Qwen3.5-9B"


def stats(**overrides: object) -> CodingTerminationStats:
    """Build a coding-stats row, defaulting to a valid shape with room to perturb one field."""
    fields: dict[str, object] = {
        "model_id": "test/model",
        "prompt_family": CODING_GRADER_PROMPTS,
        "n_observations": 1000,
        "p50_tokens": 2000,
        "p90_tokens": 10000,
        "p95_tokens": 19000,
        "observed_max_tokens": 65536,
        "screen_budget": 65536,
        "budget_floor": 24576,
        "n_over_budget_floor": 30,
        "n_at_screen_budget": 10,
        "artifact": "artifacts/reward_hacking/screen/test.json",
    }
    return CodingTerminationStats(**{**fields, **overrides})  # pyright: ignore[reportArgumentType]


def cap_hits(**overrides: object) -> SingleTurnCapHits:
    """Build a cap-hit reading, defaulting to a valid shape."""
    fields: dict[str, object] = {
        "measured_family": SINGLE_TURN_CODING_GRADER_PROMPTS,
        "budget": 24576,
        "n_observations": 944,
        "n_at_budget": 429,
        "artifact": "artifacts/reward_hacking/screen/single-turn.json",
    }
    return SingleTurnCapHits(**{**fields, **overrides})  # pyright: ignore[reportArgumentType]


class TestTheValidShapeIsValid:
    """The control for every refusal below: perturbing one field is what must make them fire."""

    def test_the_default_row_and_cap_hit_reading_both_construct(self) -> None:
        assert stats().budget_floor == 24576
        assert cap_hits().n_at_budget == 429


class TestEveryRefusalInTheRow:
    """One test per refusal, from the valid shape with a single field moved.

    The perturbation IS the sabotage here: each of these constructions raised nothing before its
    refusal existed, so a test that stops raising is a refusal that has been weakened.
    """

    def test_percentiles_out_of_order_are_refused(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            stats(p90_tokens=25000)

    def test_a_screen_with_no_observations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="measures nothing"):
            stats(n_observations=0)

    def test_a_maximum_above_the_budget_that_produced_it_is_refused(self) -> None:
        with pytest.raises(ValueError, match="which is impossible"):
            stats(observed_max_tokens=70000, screen_budget=65536)

    def test_counts_that_do_not_nest_are_refused(self) -> None:
        """Rollouts at the screen's own cap are a subset of those over the chosen budget."""
        with pytest.raises(ValueError, match="must nest"):
            stats(n_at_screen_budget=40, n_over_budget_floor=30)

    def test_more_over_budget_rollouts_than_observations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must nest"):
            stats(n_observations=20, n_over_budget_floor=30, n_at_screen_budget=10)

    def test_a_floor_below_the_measured_p95_is_refused(self) -> None:
        with pytest.raises(ValueError, match="below the measured 95th percentile"):
            stats(budget_floor=18000)

    def test_a_floor_that_cuts_more_than_the_allowed_fraction_is_refused(self) -> None:
        """Above the p95 and still shaving too much tail: the second, independent guard."""
        with pytest.raises(ValueError, match="above the 5% this table allows"):
            stats(n_over_budget_floor=60, n_at_screen_budget=10)


class TestEveryRefusalInTheCapHitReading:
    def test_no_observations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="measures nothing"):
            cap_hits(n_observations=0)

    def test_more_generations_at_the_cap_than_were_generated_is_refused(self) -> None:
        with pytest.raises(ValueError, match="is not a count"):
            cap_hits(n_observations=100, n_at_budget=101)

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="is not a count"):
            cap_hits(n_at_budget=-1)


class TestTheDerivedReadingsOnTheRealEntry:
    """These three land in the run record, so they are pinned to hand arithmetic, not to the code.

    Recomputing ``n_over_budget_floor / n_observations`` here would restate the implementation and
    pass under any pair of numbers, including a transcription error in the table itself.
    """

    def test_the_fractions_are_the_counts_the_table_states(self) -> None:
        row = MEASURED_CODING_TERMINATION_STATS_BY_MODEL[SCREENED_MODEL]

        # 68 of 1852 turns over 24,576 tokens, and 28 of 1852 at the screen's own 65,536 cap.
        assert row.fraction_over_budget_floor == pytest.approx(0.036717, abs=5e-7)
        assert row.fraction_uncapped == pytest.approx(0.015119, abs=5e-7)
        assert row.fraction_over_budget_floor <= MAX_TRUNCATED_FRACTION

    def test_the_entry_is_censored_because_its_maximum_is_the_cap(self) -> None:
        row = MEASURED_CODING_TERMINATION_STATS_BY_MODEL[SCREENED_MODEL]

        assert row.censored
        assert row.observed_max_tokens == row.screen_budget

    def test_a_terminating_screen_is_not_reported_as_censored(self) -> None:
        assert not stats(observed_max_tokens=30000).censored

    def test_the_floors_dict_and_the_stats_table_cannot_disagree(self) -> None:
        assert set(MEASURED_CODING_TERMINATION_BUDGET_BY_MODEL) == set(
            MEASURED_CODING_TERMINATION_STATS_BY_MODEL
        )
        for model_id, row in MEASURED_CODING_TERMINATION_STATS_BY_MODEL.items():
            assert row.model_id == model_id
            assert MEASURED_CODING_TERMINATION_BUDGET_BY_MODEL[model_id] == row.budget_floor

    def test_the_serialised_entry_round_trips_and_carries_the_derived_readings(self) -> None:
        """``to_json_dict`` is what a reader of a run record sees, so a dropped key is invisible."""
        row = MEASURED_CODING_TERMINATION_STATS_BY_MODEL[SCREENED_MODEL]

        restored = json.loads(json.dumps(row.to_json_dict()))

        assert restored["model_id"] == SCREENED_MODEL
        assert restored["prompt_family"] == CODING_GRADER_PROMPTS
        assert restored["budget_floor"] == 24576
        assert restored["n_observations"] == 1852
        assert restored["fraction_over_budget_floor"] == pytest.approx(0.036717, abs=5e-7)
        assert restored["fraction_uncapped"] == pytest.approx(0.015119, abs=5e-7)
        assert restored["censored"] is True
        # Every field of the dataclass plus the three derived readings, so a new field that never
        # reaches the record shows up here rather than in a reader's missing key.
        assert set(restored) == {
            "model_id",
            "prompt_family",
            "n_observations",
            "p50_tokens",
            "p90_tokens",
            "p95_tokens",
            "observed_max_tokens",
            "screen_budget",
            "budget_floor",
            "n_over_budget_floor",
            "fraction_over_budget_floor",
            "n_at_screen_budget",
            "fraction_uncapped",
            "censored",
            "artifact",
            "note",
        }


class TestTheGameFloorTie:
    """Two screens of one checkpoint's thinking length may not disagree downwards."""

    def test_a_coding_budget_below_the_game_floor_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert HIGHEST_GAME_FLOOR_MODEL in MEASURED_TERMINATION_STATS_BY_MODEL
        monkeypatch.setitem(
            MEASURED_CODING_TERMINATION_BUDGET_BY_MODEL, HIGHEST_GAME_FLOOR_MODEL, 16384
        )

        with pytest.raises(ValueError, match="cannot disagree downwards"):
            assert_not_below_game_floor(HIGHEST_GAME_FLOOR_MODEL)

    def test_the_message_says_whether_the_game_number_was_measured_or_generic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A generic default and a real screen are different evidence about the same number."""
        monkeypatch.setitem(MEASURED_CODING_TERMINATION_BUDGET_BY_MODEL, "test/unscreened", 4096)

        with pytest.raises(ValueError, match="the generic default"):
            assert_not_below_game_floor("test/unscreened")

    def test_a_model_with_no_coding_entry_is_not_checked(self) -> None:
        assert_not_below_game_floor("test/absent-everywhere")

    def test_the_real_table_clears_every_game_floor(self) -> None:
        """An emptied table would turn the loop into a vacuous pass covering no model at all."""
        assert MEASURED_CODING_TERMINATION_BUDGET_BY_MODEL
        for model_id in MEASURED_CODING_TERMINATION_BUDGET_BY_MODEL:
            assert_not_below_game_floor(model_id)


class TestTheBudgetHandedToCallers:
    def test_the_screened_model_gets_its_measured_budget(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=train_termination.logger.name):
            assert required_coding_completion_budget(SCREENED_MODEL) == 24576

    def test_an_unscreened_model_falls_back_to_the_game_floor_and_says_so(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Silent here is how the next model's arm repeats this experiment's original mistake."""
        with caplog.at_level(logging.WARNING, logger=train_termination.logger.name):
            budget = required_coding_completion_budget("test/never-screened")

        assert budget == MEASURED_TERMINATION_BUDGET
        assert "falls back to the GAME-prompt floor" in caplog.text
        assert "screen this model on coding prompts" in caplog.text


class TestTheFamilyMismatchIsAnnouncedAtThePointOfUse:
    """The shipped budget was measured on multi-turn prompts and every caller is single-turn.

    That gap has no other symptom: a truncated single-turn generation submits no solution, which
    reads as a policy that cannot code, and the p95 floor and ``MAX_TRUNCATED_FRACTION`` both stay
    green because both are computed against the multi-turn distribution. So the warning is the
    measurement's only route to a reader, and it is checked rather than assumed.

    SABOTAGE: deleting the ``logger.warning`` call in
    ``warn_if_the_budgets_family_was_later_contradicted`` turns the first test red.
    """

    def test_the_warning_names_the_measured_single_turn_cap_hit_rate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=train_termination.logger.name):
            required_coding_completion_budget(SCREENED_MODEL)

        contradiction = MEASURED_SINGLE_TURN_CAP_HITS_BY_SOURCE_FAMILY[CODING_GRADER_PROMPTS]
        assert "429 of 944" in caplog.text
        assert "45.4%" in caplog.text
        assert SINGLE_TURN_CODING_GRADER_PROMPTS in caplog.text
        assert CODING_GRADER_PROMPTS in caplog.text
        assert contradiction.artifact in caplog.text

    def test_a_family_with_no_contradicting_screen_is_silent(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise the warning is a banner rather than a reading, and stops carrying information."""
        monkeypatch.setitem(
            MEASURED_CODING_TERMINATION_STATS_BY_MODEL,
            SCREENED_MODEL,
            stats(model_id=SCREENED_MODEL, prompt_family="test/never-contradicted"),
        )

        with caplog.at_level(logging.WARNING, logger=train_termination.logger.name):
            required_coding_completion_budget(SCREENED_MODEL)

        assert caplog.text == ""

    def test_a_contradicting_screen_within_the_allowed_fraction_is_silent(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A properly sized budget retires the warning without anyone editing the call site."""
        monkeypatch.setitem(
            MEASURED_SINGLE_TURN_CAP_HITS_BY_SOURCE_FAMILY,
            CODING_GRADER_PROMPTS,
            cap_hits(n_observations=944, n_at_budget=20),
        )

        with caplog.at_level(logging.WARNING, logger=train_termination.logger.name):
            required_coding_completion_budget(SCREENED_MODEL)

        assert caplog.text == ""

    def test_the_recorded_contradiction_exceeds_what_the_table_refuses(self) -> None:
        """The premise of the warning, stated once so it cannot rot into a no-op."""
        contradiction = MEASURED_SINGLE_TURN_CAP_HITS_BY_SOURCE_FAMILY[CODING_GRADER_PROMPTS]

        assert contradiction.fraction_at_budget == pytest.approx(0.454, abs=5e-4)
        assert contradiction.fraction_at_budget > MAX_TRUNCATED_FRACTION
        assert contradiction.budget == MEASURED_CODING_TERMINATION_BUDGET_BY_MODEL[SCREENED_MODEL]


class TestTheNoteNoLongerClaimsAnUpperBound:
    """The note is serialised into every run record, so a wrong claim there travels with the data.

    It used to end "so 3.7% is an upper bound on what this budget cuts", which the single-turn
    screen contradicts by about twelve times. This pins the retraction rather than trusting it.
    """

    def test_no_entry_claims_an_upper_bound_on_what_the_budget_cuts(self) -> None:
        for row in MEASURED_CODING_TERMINATION_STATS_BY_MODEL.values():
            assert "upper bound" not in row.note.lower()

    def test_the_note_carries_the_measured_single_turn_figures_instead(self) -> None:
        note = MEASURED_CODING_TERMINATION_STATS_BY_MODEL[SCREENED_MODEL].note

        assert "429 of 944" in note
        assert "45.4%" in note
        # The one quantity from the same trace that does endorse 24,576.
        assert "24,957" in note


PROVISIONAL_MODEL = "Qwen/Qwen3.5-9B"


class TestTheProvisionalBudget:
    """An adopted budget is the third state: loud, sourced, above the game floor, never measured."""

    def test_the_provisional_model_gets_its_adopted_budget_and_a_warning_naming_the_source(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        row = PROVISIONAL_CODING_BUDGET_BY_MODEL[PROVISIONAL_MODEL]
        assert PROVISIONAL_MODEL not in MEASURED_CODING_TERMINATION_BUDGET_BY_MODEL
        with caplog.at_level(logging.WARNING, logger=train_termination.logger.name):
            assert required_coding_completion_budget(PROVISIONAL_MODEL) == row.budget == 65536
        assert "PROVISIONAL" in caplog.text
        assert row.source in caplog.text
        assert "falls back to the GAME-prompt floor" not in caplog.text

    def test_the_provenance_record_names_all_three_states(self) -> None:
        measured = completion_budget_provenance(SCREENED_MODEL)
        assert measured["source"] == BUDGET_SOURCE_MEASURED
        assert measured["budget"] == 24576
        provisional = completion_budget_provenance(PROVISIONAL_MODEL)
        assert provisional["source"] == BUDGET_SOURCE_PROVISIONAL
        assert provisional["budget"] == 65536
        assert provisional["adopted"] == "2026-09-04"
        assert str(provisional["provenance"]).startswith("TMAX")
        fallback = completion_budget_provenance("test/never-screened")
        assert fallback["source"] == BUDGET_SOURCE_GAME_FLOOR
        assert fallback["budget"] == MEASURED_TERMINATION_BUDGET
        # Every state's record round-trips through JSON: it lands in the probe's sampler block.
        for record in (measured, provisional, fallback):
            assert json.loads(json.dumps(record)) == record

    def test_a_provisional_budget_below_the_game_floor_is_refused_like_a_measured_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            PROVISIONAL_CODING_BUDGET_BY_MODEL,
            PROVISIONAL_MODEL,
            ProvisionalCodingBudget(
                model_id=PROVISIONAL_MODEL, budget=16384, source="a test", adopted="2026-09-04"
            ),
        )
        with pytest.raises(ValueError, match="cannot disagree downwards"):
            required_coding_completion_budget(PROVISIONAL_MODEL)

    def test_a_row_without_a_source_or_a_date_is_refused(self) -> None:
        with pytest.raises(ValueError, match="invented number"):
            ProvisionalCodingBudget(model_id="x", budget=1, source="  ", adopted="2026-09-04")
        with pytest.raises(ValueError, match="invented number"):
            ProvisionalCodingBudget(model_id="x", budget=1, source="a paper", adopted="")
        with pytest.raises(ValueError, match="buys nothing"):
            ProvisionalCodingBudget(model_id="x", budget=0, source="a paper", adopted="2026-09-04")

    def test_a_measured_row_wins_over_a_provisional_one(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Screening a model retires its provisional row: the measured number is what callers get."""
        monkeypatch.setitem(
            PROVISIONAL_CODING_BUDGET_BY_MODEL,
            SCREENED_MODEL,
            ProvisionalCodingBudget(
                model_id=SCREENED_MODEL, budget=65536, source="a test", adopted="2026-09-04"
            ),
        )
        with caplog.at_level(logging.WARNING, logger=train_termination.logger.name):
            assert required_coding_completion_budget(SCREENED_MODEL) == 24576
        assert "PROVISIONAL" not in caplog.text
        assert completion_budget_provenance(SCREENED_MODEL)["source"] == BUDGET_SOURCE_MEASURED

    def test_the_real_provisional_table_clears_every_game_floor(self) -> None:
        assert PROVISIONAL_CODING_BUDGET_BY_MODEL
        for model_id in PROVISIONAL_CODING_BUDGET_BY_MODEL:
            assert_not_below_game_floor(model_id)
