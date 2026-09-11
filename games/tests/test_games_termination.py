"""Offline tests for the measured termination table and the budgets read off it.

The table is data a human transcribed from screen artifacts, so what these check is that it cannot
be internally inconsistent: a floor below its own observed maximum would license truncating a
rollout the screen already watched finish, and a percentile that drifted from the raw lengths would
send the decode sizing after a number nothing generated.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from games.termination import (
    MEASURED_TERMINATION_BUDGET,
    MEASURED_TERMINATION_BUDGET_BY_MODEL,
    MEASURED_TERMINATION_STATS_BY_MODEL,
    TerminationStats,
    decode_peak_tokens,
    nearest_rank_percentile,
    required_completion_budget,
    termination_stats,
)

SCREEN_ROOT = Path("artifacts/games/screen")


def stats(**overrides: object) -> TerminationStats:
    """Build a stats row, defaulting to a shape the real table uses."""
    fields: dict[str, object] = {
        "model_id": "test/model",
        "rollout_tokens": (100, 200, 300, 400),
        "screen_budget": 32768,
        "all_rollouts_terminated": True,
        "budget_floor": 16384,
        "artifact": "artifacts/games/screen/test.json",
    }
    return TerminationStats(**{**fields, **overrides})  # pyright: ignore[reportArgumentType]


class TestPercentile:
    def test_nearest_rank_returns_an_observation_never_an_interpolation(self):
        eight = (5990, 7143, 7525, 10103, 10953, 11298, 13781, 19936)
        assert nearest_rank_percentile(eight, 0.75) == 11298
        assert nearest_rank_percentile(eight, 0.5) == 10103
        assert nearest_rank_percentile(eight, 1.0) == 19936
        # Six observations: ceil(0.75 * 6) = 5, the fifth smallest.
        assert nearest_rank_percentile((78, 1709, 6413, 8979, 11053, 24337), 0.75) == 11053

    def test_an_unsorted_input_is_sorted_first(self):
        assert nearest_rank_percentile((400, 100, 300, 200), 0.5) == 200

    def test_nonsense_is_refused(self):
        with pytest.raises(ValueError, match="no observations"):
            nearest_rank_percentile((), 0.5)
        with pytest.raises(ValueError, match="quantile must be"):
            nearest_rank_percentile((1, 2), 0.0)
        with pytest.raises(ValueError, match="quantile must be"):
            nearest_rank_percentile((1, 2), 1.5)


class TestStatsRow:
    def test_aggregates_are_derived_from_the_raw_lengths(self):
        row = stats(rollout_tokens=(2156, 2755, 7040, 7361, 7581, 8436, 10866, 14092))
        assert row.n_rollouts == 8
        assert row.median_tokens == pytest.approx(7471.0)
        assert row.p75_tokens == 8436
        assert row.max_tokens == 14092

    def test_a_floor_below_the_observed_maximum_is_refused(self):
        with pytest.raises(ValueError, match="below its own observed maximum"):
            stats(rollout_tokens=(100, 20000), budget_floor=16384)

    def test_a_screen_with_no_rollouts_is_refused(self):
        with pytest.raises(ValueError, match="has not been screened"):
            stats(rollout_tokens=())

    def test_a_rollout_longer_than_the_screen_that_produced_it_is_refused(self):
        with pytest.raises(ValueError, match="cannot happen"):
            stats(rollout_tokens=(100, 5000), screen_budget=4096, budget_floor=8192)


class TestMeasuredTable:
    def test_every_floor_clears_its_own_measured_maximum(self):
        for model_id, row in MEASURED_TERMINATION_STATS_BY_MODEL.items():
            assert row.model_id == model_id
            assert MEASURED_TERMINATION_BUDGET_BY_MODEL[model_id] == row.budget_floor
            assert row.budget_floor >= row.max_tokens

    def test_the_floors_dict_and_the_stats_table_cannot_disagree(self):
        assert set(MEASURED_TERMINATION_BUDGET_BY_MODEL) == set(MEASURED_TERMINATION_STATS_BY_MODEL)

    def test_the_measured_ladder_reads_the_way_the_docs_claim(self):
        assert required_completion_budget("Qwen/Qwen3.5-2B") == 24576
        assert required_completion_budget("Qwen/Qwen3.5-4B") == 16384
        assert required_completion_budget("Qwen/Qwen3.5-9B") == 32768
        assert required_completion_budget("some/unscreened-model") == MEASURED_TERMINATION_BUDGET
        assert termination_stats("some/unscreened-model") is None

    @pytest.mark.parametrize(
        ("model_id", "artifact"),
        [
            ("Qwen/Qwen3.5-2B", "qwen2b-termination-32k.json"),
            ("Qwen/Qwen3.5-4B", "qwen4b-termination-32k.json"),
            ("Qwen/Qwen3.5-9B", "qwen9b-termination-32k.json"),
        ],
    )
    def test_the_transcribed_lengths_match_the_screen_artifact(self, model_id: str, artifact: str):
        """Re-read the artifact the row names, when it is still on this box.

        Skipped rather than failed when the file is absent: `artifacts/` is gitignored, so a fresh
        clone has none of them and this check is a local-machine luxury rather than an invariant of
        the code. It is here because a transcription error is otherwise undetectable.
        """
        path = SCREEN_ROOT / artifact
        if not path.exists():
            pytest.skip(f"{path} is a gitignored local artifact and is not on this machine")
        record = json.loads(path.read_text(encoding="utf-8"))
        row = MEASURED_TERMINATION_STATS_BY_MODEL[model_id]
        assert record["config"]["model_id"] == model_id
        observed = sorted(int(rollout["n_tokens"]) for rollout in record["rollouts"])
        assert sorted(row.rollout_tokens) == observed
        assert row.all_rollouts_terminated == all(
            not rollout["hit_cap"] for rollout in record["rollouts"]
        )
        assert row.median_tokens == pytest.approx(statistics.median(observed))


class TestDecodePeakTokens:
    def test_a_screened_model_is_sized_on_its_observed_maximum(self):
        tokens, why = decode_peak_tokens("Qwen/Qwen3.5-2B", ceiling=24576)
        assert tokens == 19936
        assert "observed maximum 19936 over 8 screened rollouts" in why

    def test_the_ceiling_still_wins_when_it_is_the_smaller_number(self):
        tokens, _ = decode_peak_tokens("Qwen/Qwen3.5-2B", ceiling=8192)
        assert tokens == 8192

    def test_an_unscreened_model_falls_back_to_the_ceiling(self):
        tokens, why = decode_peak_tokens("some/unscreened-model", ceiling=16384)
        assert tokens == 16384
        assert "no termination screen" in why

    def test_a_censored_screen_is_not_treated_as_a_measured_maximum(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        censored = stats(
            model_id="test/censored",
            rollout_tokens=(1000, 4096),
            screen_budget=4096,
            all_rollouts_terminated=False,
            budget_floor=8192,
        )
        monkeypatch.setitem(MEASURED_TERMINATION_STATS_BY_MODEL, "test/censored", censored)
        tokens, why = decode_peak_tokens("test/censored", ceiling=16384)
        assert tokens == 16384
        assert "lower bound rather than a maximum" in why
