"""The mix-split partitioner: which prompts land on which side of the group-mix boundary.

The tests worth reading first are the ones that would catch the mistake the module exists to prevent
-- `TestPairsStayWhole` (a counterbalanced scenario split across both arms, each half trained in the
direction the other contradicts) and `TestPartitionsMustTrainInOppositeDirections` (a split whose two
sides are the same experiment under two names). Both were sabotaged before being trusted.

The boundary arithmetic is checked against hand-computed thresholds rather than against the repo's own
`stag_hunt_cooperation_threshold`, and then against it as a second, independent statement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

from games import corpus_partition as cp
from games.arms import (
    CORPUS_PARTITION_COLUMN,
    PARTITION_ABOVE_THRESHOLD,
    PARTITION_BELOW_THRESHOLD,
)
from games.payoffs import (
    STAG_HUNT_VARIANTS,
    chicken,
    group_mix_fixed_point,
    group_mix_gap_crossing,
    group_mix_gap_slope,
    stag_hunt,
    stag_hunt_cooperation_threshold,
    twin_pd,
)
from games.prompts import generate_prompt_rows
from games.select_prompts import pair_identity

if TYPE_CHECKING:
    from pathlib import Path

# The safe-hunt rung, normalised the way `games.payoffs` normalises every spec: raw (4, 0, 3, 3)
# divided by its largest cell. Its boundary is (DD - CD) / ((CC - DC) + (DD - CD)) = 0.75, the only
# rung whose threshold sits inside the 2026-08-19 measured spread of baseline cooperation.
SAFE_HUNT_CELLS = {
    "payoff_cc": 1.0,
    "payoff_cd": 0.0,
    "payoff_dc": 0.75,
    "payoff_dd": 0.75,
}
SAFE_HUNT_BOUNDARY = 0.75

# Hand-computed from the raw cells in STAG_HUNT_VARIANTS, not read off the module under test:
# favoured (10,0,4,3) -> 3/((10-4)+3); even (10,0,5.5,4.5) -> 4.5/((10-5.5)+4.5);
# safe (4,0,3,3) -> 3/((4-3)+3); risky (4,0,3.8,3.8) -> 3.8/((4-3.8)+3.8).
HAND_COMPUTED_STAG_BOUNDARIES = {
    "favoured-hunt": 1 / 3,
    "even-hunt": 0.5,
    "safe-hunt": 0.75,
    "risky-hunt": 0.95,
}

# One pair per outcome against a boundary of 0.75 -- both above, both below, one straddling -- so the
# counts the command-line tests assert are readable from the rates.
TWO_SIDED_PAIRS: dict[str, tuple[float, float]] = {
    "manifold-gas-bank": (0.857, 0.857),
    "joint-audit-filing": (0.5, 0.25),
    "narrows-approach": (0.875, 0.375),
}


def make_row(
    *,
    prompt_id: str,
    reskin_id: str,
    coop_label: str = "SIDE",
    payoff_variant: str = "safe-hunt",
    **overrides: Any,
) -> dict[str, Any]:
    """One corpus row, carrying every column `pair_identity` and the boundary arithmetic read."""
    return {
        "prompt": f"a sheet for {prompt_id}",
        "prompt_id": prompt_id,
        "game_id": "stag-hunt",
        "grading": "group-mix",
        "label_a": "SIDE",
        "label_b": "LOOP",
        "coop_label": coop_label,
        "coop_label_index": 0 if coop_label == "SIDE" else 1,
        "opp_coop_prob": -1.0,
        "reskin_id": reskin_id,
        "payoff_variant": payoff_variant,
        **SAFE_HUNT_CELLS,
        **overrides,
    }


def make_pair(
    reskin_id: str, rates: tuple[float, float], **overrides: Any
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Both label orientations of one scenario, plus the baseline rates the sweep measured for them.

    The two orientations differ only in which neutral label names the cooperative option, which is
    exactly what `pair_identity` ignores -- so these two rows group as one pair.
    """
    rows = [
        make_row(
            prompt_id=f"{reskin_id}--coop0", reskin_id=reskin_id, coop_label="SIDE", **overrides
        ),
        make_row(
            prompt_id=f"{reskin_id}--coop1", reskin_id=reskin_id, coop_label="LOOP", **overrides
        ),
    ]
    return rows, {str(row["prompt_id"]): rate for row, rate in zip(rows, rates, strict=True)}


def make_corpus(
    pairs: dict[str, tuple[float, float]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Assemble a corpus of counterbalanced pairs and the rate map that goes with it."""
    rows: list[dict[str, Any]] = []
    rates: dict[str, float] = {}
    for reskin_id, pair_rates in pairs.items():
        pair_rows, pair_rate_map = make_pair(reskin_id, pair_rates)
        rows.extend(pair_rows)
        rates.update(pair_rate_map)
    return rows, rates


def two_sided_corpus() -> tuple[list[dict[str, Any]], dict[str, float]]:
    """A corpus with one pair per outcome: both above, both below, straddling, and one exactly on."""
    return make_corpus(
        {
            "manifold-gas-bank": (0.857, 0.857),
            "shared-oven-fire": (0.875, 1.0),
            "joint-audit-filing": (0.5, 0.25),
            "control-network-benchmark": (0.667, 0.571),
            "narrows-approach": (0.875, 0.375),
            "cluster-change-window": (0.857, 0.75),
        }
    )


class TestTheBoundaryIsTheTrainersOwnNumber:
    """The split is only meaningful if its boundary is the rate the reward gap actually flips at."""

    @pytest.mark.parametrize("payoff_variant", sorted(STAG_HUNT_VARIANTS))
    def test_every_rung_crosses_where_hand_arithmetic_says(self, payoff_variant: str):
        crossing = group_mix_gap_crossing(stag_hunt(payoff_variant), num_generations=8)
        assert crossing == pytest.approx(HAND_COMPUTED_STAG_BOUNDARIES[payoff_variant])

    @pytest.mark.parametrize("payoff_variant", sorted(STAG_HUNT_VARIANTS))
    def test_the_crossing_agrees_with_the_risk_dominance_threshold(self, payoff_variant: str):
        # Two independently-derived statements of the same quantity: the group-mix gap's crossing and
        # the risk-dominance boundary. They must agree, and if one is ever rewritten this says so.
        spec = stag_hunt(payoff_variant)
        assert group_mix_gap_crossing(spec, num_generations=8) == pytest.approx(
            stag_hunt_cooperation_threshold(spec)
        )

    def test_a_stag_rungs_crossing_repels_and_therefore_has_no_fixed_point(self):
        # The property that makes the two sides opposite directions rather than convergent ones.
        spec = stag_hunt("safe-hunt")
        assert group_mix_gap_slope(spec) > 0
        assert group_mix_fixed_point(spec, num_generations=8) is None
        assert group_mix_gap_crossing(spec, num_generations=8) == pytest.approx(0.75)

    def test_chickens_stable_crossing_is_unchanged_by_the_refactor(self):
        # The regression guard on splitting the crossing out of the fixed point: chicken's documented
        # 0.50 plain and 0.3125 under leave-one-out at a group of 8.
        spec = chicken()
        assert group_mix_fixed_point(spec, num_generations=8) == pytest.approx(0.5)
        assert group_mix_gap_crossing(spec, num_generations=8) == pytest.approx(0.5)
        assert group_mix_gap_crossing(spec, num_generations=8, leave_one_out=True) == pytest.approx(
            0.3125
        )

    def test_leave_one_out_moves_the_boundary_a_partition_would_use(self):
        # The reason the CLI carries the flag at all: a corpus split at 0.75 and then trained with
        # leave-one-out on would have been sorted by a number the run never used.
        spec = stag_hunt("safe-hunt")
        plain = cp.row_boundary(_row_of(spec), num_generations=8, leave_one_out=False)
        looed = cp.row_boundary(_row_of(spec), num_generations=8, leave_one_out=True)
        assert plain == pytest.approx(0.75)
        assert looed != pytest.approx(plain)

    def test_a_dominance_game_has_no_boundary_to_split_at(self):
        # twin-pd's gap keeps one sign at every mix, so both "sides" would be pushed the same way.
        row = _row_of(twin_pd("temptation-2"))
        with pytest.raises(ValueError, match="no interior group-mix boundary"):
            cp.row_boundary(row, num_generations=8, leave_one_out=False)

    def test_a_row_missing_a_payoff_cell_names_the_columns(self):
        row = make_row(prompt_id="a", reskin_id="r")
        del row["payoff_dd"]
        with pytest.raises(ValueError, match="missing payoff cells"):
            cp.row_boundary(row, num_generations=8, leave_one_out=False)


def _row_of(spec: Any) -> dict[str, Any]:
    """A corpus row carrying one spec's cells, for the boundary functions that read them."""
    return make_row(
        prompt_id="probe",
        reskin_id="probe",
        payoff_cc=spec.payoff_cc,
        payoff_cd=spec.payoff_cd,
        payoff_dc=spec.payoff_dc,
        payoff_dd=spec.payoff_dd,
    )


class TestSideOfTheBoundary:
    def test_a_rate_above_and_below_get_the_two_partitions(self):
        assert cp.side_of_boundary(0.9, SAFE_HUNT_BOUNDARY) == PARTITION_ABOVE_THRESHOLD
        assert cp.side_of_boundary(0.5, SAFE_HUNT_BOUNDARY) == PARTITION_BELOW_THRESHOLD

    def test_a_rate_exactly_on_the_boundary_has_no_side(self):
        # Not a corner case: two of safe-hunt's sixteen frames measured exactly 0.750 against a
        # boundary of 0.75 in the 2026-08-19 sweep. The reward gap there is zero, so the prompt is
        # pushed neither way and giving it a side would invent a direction from a tie.
        assert cp.side_of_boundary(SAFE_HUNT_BOUNDARY, SAFE_HUNT_BOUNDARY) is None


class TestBaselineRatesComeFromThePolicySweep:
    @staticmethod
    def trace() -> list[dict[str, Any]]:
        return [
            {"record_kind": "sweep-meta", "prompt_id_order_sha256": "abc", "n_prompts": 2},
            {"record_kind": "prompt-sweep", "prompt_id": "a", "coop_fraction": 0.875},
            {"record_kind": "prompt-sweep", "prompt_id": "b", "coop_fraction": 0.25},
            {"record_kind": "frozen-opponent-sweep", "prompt_id": "a", "coop_fraction": 0.0},
        ]

    def test_the_meta_line_and_the_opponents_records_are_not_read_as_policy_rates(self):
        # The opponent's rate describes a different model entirely; reading it would put prompts on
        # the wrong side of the boundary while every count still looked right.
        assert cp.baseline_coop_fractions(self.trace()) == {"a": 0.875, "b": 0.25}

    def test_a_prompt_with_no_measured_rate_raises(self):
        records: list[dict[str, Any]] = [
            self.trace()[0],
            {"record_kind": "prompt-sweep", "prompt_id": "a"},
        ]
        with pytest.raises(ValueError, match="coop_fraction=None"):
            cp.baseline_coop_fractions(records)

    def test_a_repeated_prompt_raises_rather_than_letting_the_later_record_win(self):
        records: list[dict[str, Any]] = [
            *self.trace(),
            {"record_kind": "prompt-sweep", "prompt_id": "a", "coop_fraction": 0.1},
        ]
        with pytest.raises(ValueError, match="appears twice"):
            cp.baseline_coop_fractions(records)

    def test_a_trace_with_only_its_meta_line_raises(self):
        with pytest.raises(ValueError, match="no 'prompt-sweep' records"):
            cp.baseline_coop_fractions([self.trace()[0]])


class TestPairsStayWhole:
    """The guard the whole module is shaped around, and the sabotage it was verified with."""

    def test_both_orientations_of_a_scenario_share_one_partition(self):
        rows, rates = two_sided_corpus()
        stamped = cp.stamp_partitions(rows, cp.assign_pairs(rows, rates))
        by_pair: dict[tuple[object, ...], set[str]] = {}
        for row in stamped:
            by_pair.setdefault(pair_identity(row), set()).add(str(row[CORPUS_PARTITION_COLUMN]))
        assert all(len(partitions) == 1 for partitions in by_pair.values())

    def test_a_pair_split_across_two_partitions_is_refused(self):
        rows, rates = two_sided_corpus()
        stamped = cp.stamp_partitions(rows, cp.assign_pairs(rows, rates))
        # Exactly what a per-prompt assignment would produce: one orientation moved to the other side.
        stamped[0] = {**stamped[0], CORPUS_PARTITION_COLUMN: PARTITION_BELOW_THRESHOLD}
        with pytest.raises(ValueError, match="different partitions"):
            cp.assert_pairs_are_whole(stamped)

    def test_a_straddling_pair_lands_in_neither_partition_and_is_counted(self):
        rows, rates = make_corpus({"narrows-approach": (0.875, 0.375)})
        [assignment] = cp.assign_pairs(rows, rates)
        assert assignment.partition == cp.PARTITION_STRADDLING_PAIR
        assert assignment.is_straddling
        # The contamination diagnostic: with orientations at 0.875 and 0.375 the pair mean is 0.625,
        # so a mean-based split would have put this scenario confidently on the below side while one
        # of its own two orientations sits above the boundary being trained away from.
        assert assignment.pair_mean == pytest.approx(0.625)
        assert assignment.pair_mean_partition == PARTITION_BELOW_THRESHOLD
        assert assignment.within_pair_gap == pytest.approx(0.5)

    def test_a_pair_with_one_orientation_exactly_on_the_boundary_straddles(self):
        rows, rates = make_corpus({"cluster-change-window": (0.857, 0.75)})
        [assignment] = cp.assign_pairs(rows, rates)
        assert assignment.partition == cp.PARTITION_STRADDLING_PAIR

    def test_a_lone_orientation_is_refused_rather_than_partitioned(self):
        rows, rates = make_corpus({"joint-audit-filing": (0.5, 0.25)})
        with pytest.raises(ValueError, match="not counterbalanced pairs"):
            cp.assign_pairs(rows[:1], rates)

    def test_a_row_with_no_labels_to_swap_cannot_be_mix_split(self):
        rows, rates = make_corpus({"joint-audit-filing": (0.5, 0.25)})
        unlabelled: list[dict[str, Any]] = [{**row, "label_a": "", "label_b": ""} for row in rows]
        with pytest.raises(ValueError, match="no option labels to swap"):
            cp.assign_pairs(unlabelled, rates)


class TestAssignmentOverTheRealStagCorpus:
    """Synthetic rows can hide a schema drift; these are the rows `games.prompts` really renders."""

    @staticmethod
    def rows() -> list[dict[str, Any]]:
        return cp.filter_payoff_variant(
            generate_prompt_rows("stag-hunt", "group-mix", split="train"), "safe-hunt"
        )

    def test_the_real_corpus_groups_into_pairs_of_two(self):
        groups = cp.group_rows_by_pair(self.rows())
        assert groups
        assert all(len(group) == 2 for group in groups.values())

    def test_every_real_row_gets_a_side_and_its_own_boundary_stamped(self):
        rows = self.rows()
        # A designed mix rather than a measured one: alternate pairs high and low so both partitions
        # are populated, which is what lets the stamping and the boundary column be checked at all.
        rates = {
            str(row["prompt_id"]): 0.9 if index % 4 < 2 else 0.5 for index, row in enumerate(rows)
        }
        stamped = cp.stamp_partitions(rows, cp.assign_pairs(rows, rates))
        assert len(stamped) == len(rows)
        assert {str(row[CORPUS_PARTITION_COLUMN]) for row in stamped} == {
            PARTITION_ABOVE_THRESHOLD,
            PARTITION_BELOW_THRESHOLD,
        }
        boundaries = {str(row[cp.CORPUS_PARTITION_BOUNDARY_COLUMN]) for row in stamped}
        assert sorted(float(boundary) for boundary in boundaries) == [
            pytest.approx(SAFE_HUNT_BOUNDARY)
        ]

    def test_the_pin_the_registry_requires_matches_the_variant_the_partition_was_built_for(self):
        rows = self.rows()
        assert {str(row["payoff_variant"]) for row in rows} == {"safe-hunt"}

    def test_a_variant_the_corpus_does_not_carry_is_refused(self):
        with pytest.raises(ValueError, match="no rows carry payoff variant"):
            cp.filter_payoff_variant(self.rows(), "temptation-2")


class TestPartitionsMustTrainInOppositeDirections:
    """Checked against the written rows, not the assignment objects, so the question is not circular.

    Asked of the `PairAssignment`s this would be tautological -- a row is on the above side *because*
    its rate beat the boundary. Asked of the stamped corpus it re-derives the side from three inputs
    that arrived separately: the label on the row, the boundary column beside it, and the sweep's own
    rate for that prompt id.
    """

    @staticmethod
    def stamped() -> tuple[list[dict[str, Any]], dict[str, float]]:
        rows, rates = two_sided_corpus()
        return cp.stamp_partitions(rows, cp.assign_pairs(rows, rates)), rates

    def test_a_two_sided_split_passes(self):
        rows, rates = self.stamped()
        cp.assert_partitions_train_in_opposite_directions(rows, rates)

    def test_a_mislabelled_row_is_caught_against_its_own_boundary(self):
        rows, rates = self.stamped()
        above = next(
            row for row in rows if row[CORPUS_PARTITION_COLUMN] == PARTITION_ABOVE_THRESHOLD
        )
        flipped: list[dict[str, Any]] = [
            {**row, CORPUS_PARTITION_COLUMN: PARTITION_BELOW_THRESHOLD}
            if row["prompt_id"] == above["prompt_id"]
            else row
            for row in rows
        ]
        with pytest.raises(ValueError, match="is stamped"):
            cp.assert_partitions_train_in_opposite_directions(flipped, rates)

    def test_a_boundary_column_that_does_not_match_the_label_is_caught(self):
        # The other direction of the same mistake: the label is left alone and the boundary it was
        # judged against is edited, which a checker reading only the labels would pass.
        rows, rates = self.stamped()
        moved: list[dict[str, Any]] = [
            {**row, cp.CORPUS_PARTITION_BOUNDARY_COLUMN: 0.99} for row in rows
        ]
        with pytest.raises(ValueError, match="is stamped"):
            cp.assert_partitions_train_in_opposite_directions(moved, rates)

    def test_an_empty_side_is_refused_and_the_message_counts_the_straddles(self):
        # The arithmetic outcome the real safe-hunt corpus produced: plenty of pairs, nearly all of
        # them straddling, so one side comes out empty while every other count looks healthy.
        rows, rates = make_corpus(
            {
                "narrows-approach": (0.875, 0.375),
                "perimeter-watch-route": (0.5, 0.875),
                "joint-audit-filing": (0.5, 0.25),
            }
        )
        stamped = cp.stamp_partitions(rows, cp.assign_pairs(rows, rates))
        with pytest.raises(ValueError, match="holds no prompts") as raised:
            cp.assert_partitions_train_in_opposite_directions(stamped, rates)
        assert "4 of 6 rows belong to pairs that straddle" in str(raised.value)

    def test_a_row_carrying_an_unknown_partition_is_refused(self):
        rows, rates = self.stamped()
        relabelled: list[dict[str, Any]] = [
            {**row, CORPUS_PARTITION_COLUMN: "somewhere-else"} for row in rows
        ]
        with pytest.raises(ValueError, match="not a side of any boundary"):
            cp.assert_partitions_train_in_opposite_directions(relabelled, rates)


class TestTheFallbackBoundaryIsNamedNotHidden:
    def test_the_pair_mean_median_splits_a_corpus_the_crossing_cannot(self):
        # Every pair is above 0.75, so the crossing leaves the below side empty; the median of the
        # pair means splits them anyway. The claim that survives is "different starting points",
        # which is why the mode is recorded in the artifact and in the filename.
        pairs = {
            "one": (0.80, 0.82),
            "two": (0.85, 0.87),
            "three": (0.90, 0.92),
            "four": (0.95, 0.97),
        }
        rows, rates = make_corpus(pairs)
        crossing = cp.assign_pairs(rows, rates)
        assert all(assignment.partition == PARTITION_ABOVE_THRESHOLD for assignment in crossing)
        median = cp.assign_pairs(rows, rates, boundary_mode=cp.BOUNDARY_PAIR_MEAN_MEDIAN)
        summary = cp.partition_summary(median)
        assert summary["prompts_by_partition"] == {
            PARTITION_ABOVE_THRESHOLD: 4,
            PARTITION_BELOW_THRESHOLD: 4,
        }

    def test_an_unknown_boundary_mode_is_refused(self):
        rows, rates = two_sided_corpus()
        with pytest.raises(ValueError, match="unknown boundary_mode"):
            cp.assign_pairs(rows, rates, boundary_mode="vibes")


class TestTheSummaryReportsItsOwnDenominators:
    def test_the_counts_add_up_and_name_the_straddles(self):
        rows, rates = two_sided_corpus()
        summary = cp.partition_summary(cp.assign_pairs(rows, rates))
        assert summary["n_pairs"] == 6
        assert summary["n_prompts"] == 12
        assert summary["pairs_by_partition"] == {
            PARTITION_ABOVE_THRESHOLD: 2,
            PARTITION_BELOW_THRESHOLD: 2,
            cp.PARTITION_STRADDLING_PAIR: 2,
        }
        assert summary["n_straddling_pairs"] == 2
        assert summary["boundaries"] == [pytest.approx(SAFE_HUNT_BOUNDARY)]

    def test_each_side_reports_the_baseline_mix_its_direction_is_read_against(self):
        rows, rates = two_sided_corpus()
        summary = cp.partition_summary(cp.assign_pairs(rows, rates))
        baseline = summary["baseline_by_partition"]
        assert isinstance(baseline, dict)
        assert baseline[PARTITION_ABOVE_THRESHOLD]["min"] > SAFE_HUNT_BOUNDARY
        assert baseline[PARTITION_BELOW_THRESHOLD]["max"] < SAFE_HUNT_BOUNDARY

    def test_the_contamination_diagnostic_names_the_pairs_a_mean_split_would_have_kept(self):
        rows, rates = two_sided_corpus()
        summary = cp.partition_summary(cp.assign_pairs(rows, rates))
        # narrows-approach straddles (0.875 / 0.375) but its mean 0.625 is confidently below, and
        # cluster-change-window straddles on an exactly-on-boundary orientation with mean 0.804.
        assert summary["pairs_a_mean_split_would_have_kept"] == [
            "cluster-change-window",
            "narrows-approach",
        ]


class TestStampingRefusesWhatTrainingCannotUse:
    def test_a_row_with_no_assignment_raises(self):
        rows, rates = two_sided_corpus()
        assignments = cp.assign_pairs(rows, rates)
        extra, _ = make_pair("unswept", (0.9, 0.9))
        with pytest.raises(KeyError, match="no partition assignment"):
            cp.stamp_partitions([*rows, *extra], assignments)

    def test_a_corpus_prompt_the_sweep_never_measured_raises(self):
        rows, rates = two_sided_corpus()
        del rates[str(rows[0]["prompt_id"])]
        with pytest.raises(KeyError, match="not in the sweep trace"):
            cp.assign_pairs(rows, rates)

    def test_the_source_rows_are_left_as_they_were_written(self):
        rows, rates = two_sided_corpus()
        cp.stamp_partitions(rows, cp.assign_pairs(rows, rates))
        assert all(CORPUS_PARTITION_COLUMN not in row for row in rows)

    def test_stamping_assignments_that_split_a_pair_raises(self):
        """That `stamp_partitions` runs the pair guard, not just that the guard has teeth.

        Deleting the `assert_pairs_are_whole(stamped)` call from `stamp_partitions` left the whole
        suite green (integration mutation testing, 2026-08-21), because the guard's own test called
        it directly. Assignments are what a caller supplies, so a partitioner other than
        `assign_pairs` -- or a hand-edited artifact -- can hand over a pair already split, and this
        is the only place that catches it before the corpus reaches a training run.
        """
        rows, rates = two_sided_corpus()
        assignments = list(cp.assign_pairs(rows, rates))
        split = assignments[0]
        assert len(split.prompt_ids) == 2, "the fixture must be a counterbalanced pair"
        assignments[0] = dataclasses.replace(
            split, prompt_ids=(split.prompt_ids[0],), partition=PARTITION_ABOVE_THRESHOLD
        )
        assignments.append(
            dataclasses.replace(
                split, prompt_ids=(split.prompt_ids[1],), partition=PARTITION_BELOW_THRESHOLD
            )
        )
        with pytest.raises(ValueError, match="different partitions"):
            cp.stamp_partitions(rows, assignments)


class TestProvenanceMakesTheSplitReproducible:
    def test_the_sweeps_prompt_order_hash_is_carried_into_the_artifact(self, tmp_path: Path):
        # The derived-selection trap: the same code over a pool with one more scenario returns a
        # different split, and only the pool hash can say the two are not comparable.
        records: list[dict[str, Any]] = [
            {
                "record_kind": "sweep-meta",
                "prompt_id_order_sha256": "deadbeef",
                "written_at": "2026-08-19T22:51:55Z",
                "n_prompts": 128,
                "samples_per_prompt": 8,
                "backend": {"model_id": "Qwen/Qwen3.5-2B"},
            }
        ]
        provenance = cp.sweep_provenance(records, sweep_path=tmp_path / "sweep.jsonl")
        assert provenance["prompt_id_order_sha256"] == "deadbeef"
        assert provenance["prompt_id_order_sha256_source"] == "meta-record"
        assert provenance["sweep_model_id"] == "Qwen/Qwen3.5-2B"

    def test_a_trace_with_no_meta_record_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no meta record"):
            cp.sweep_provenance(
                [{"record_kind": "prompt-sweep", "prompt_id": "a"}],
                sweep_path=tmp_path / "sweep.jsonl",
            )

    def test_a_meta_predating_the_hash_gets_one_derived_from_record_order(self, tmp_path: Path):
        # The 2026-08-19 wave-1 sweeps carry a meta line but predate the hash field.
        # `write_sweep_trace` writes the policy records in the same list order `sweep_meta`
        # hashes, so the trace's own record order reproduces the number the meta would have
        # carried -- verified against the two wave-1 stag sweeps that carry both. The source
        # field is what tells a derived pin from a recorded one in the artifact.
        records: list[dict[str, Any]] = [
            {
                "record_kind": "sweep-meta",
                "written_at": "2026-08-19T20:56:04Z",
                "n_prompts": 2,
                "samples_per_prompt": 8,
                "backend": {"model_id": "Qwen/Qwen3.5-2B"},
            },
            {"record_kind": "prompt-sweep", "prompt_id": "b--coop0"},
            {"record_kind": "frozen-opponent-sweep", "prompt_id": "never-hashed"},
            {"record_kind": "prompt-sweep", "prompt_id": "a--coop1"},
        ]
        provenance = cp.sweep_provenance(records, sweep_path=tmp_path / "sweep.jsonl")
        expected = hashlib.sha256(b"b--coop0\na--coop1").hexdigest()
        assert provenance["prompt_id_order_sha256"] == expected
        assert provenance["prompt_id_order_sha256_source"] == "derived-from-trace-record-order"
        # Order-sensitivity is the point of the hash: swapping two records must change it.
        swapped = [records[0], records[3], records[2], records[1]]
        reordered = cp.sweep_provenance(swapped, sweep_path=tmp_path / "sweep.jsonl")
        assert reordered["prompt_id_order_sha256"] != expected


class TestTheCommandLineEndToEnd:
    @staticmethod
    def write_inputs(tmp_path: Path, pairs: dict[str, tuple[float, float]]) -> tuple[Path, Path]:
        rows, rates = make_corpus(pairs)
        sweep_path = tmp_path / "sweep-stag-hunt-Qwen3.5-2B.jsonl"
        corpus_path = tmp_path / "corpus-stag-hunt-Qwen3.5-2B.jsonl"
        trace: list[dict[str, Any]] = [
            {
                "record_kind": "sweep-meta",
                "prompt_id_order_sha256": "abc123",
                "n_prompts": len(rows),
                "samples_per_prompt": 8,
                "backend": {"model_id": "Qwen/Qwen3.5-2B"},
            },
            *(
                {
                    "record_kind": "prompt-sweep",
                    "prompt_id": prompt_id,
                    "coop_fraction": rate,
                }
                for prompt_id, rate in rates.items()
            ),
        ]
        sweep_path.write_text("\n".join(json.dumps(record) for record in trace) + "\n")
        corpus_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return sweep_path, corpus_path

    def test_a_two_sided_split_writes_the_artifact_and_the_stamped_corpus(self, tmp_path: Path):
        sweep_path, corpus_path = self.write_inputs(tmp_path, dict(TWO_SIDED_PAIRS))
        out_dir = tmp_path / "partition"
        assert (
            cp.main(
                [
                    "--sweep",
                    str(sweep_path),
                    "--corpus",
                    str(corpus_path),
                    "--payoff-variant",
                    "safe-hunt",
                    "--out-dir",
                    str(out_dir),
                ]
            )
            == 0
        )
        [artifact_path] = sorted(out_dir.glob("partition-*.json"))
        [partitioned] = sorted(out_dir.glob("corpus-*-partitioned.jsonl"))
        artifact = json.loads(artifact_path.read_text())
        assert artifact["boundary_mode"] == cp.BOUNDARY_GROUP_MIX_CROSSING
        assert artifact["payoff_variant"] == "safe-hunt"
        assert artifact["sweep"]["prompt_id_order_sha256"] == "abc123"
        assert artifact["summary"]["n_straddling_pairs"] == 1
        assert len(artifact["pairs"]) == len(TWO_SIDED_PAIRS)
        stamped = [json.loads(line) for line in partitioned.read_text().splitlines() if line]
        assert len(stamped) == 2 * len(TWO_SIDED_PAIRS)
        assert {row[CORPUS_PARTITION_COLUMN] for row in stamped} == {
            PARTITION_ABOVE_THRESHOLD,
            PARTITION_BELOW_THRESHOLD,
            cp.PARTITION_STRADDLING_PAIR,
        }

    def test_a_one_sided_split_still_writes_its_numbers_before_refusing(self, tmp_path: Path):
        # The write order is the design: the artifact says how many pairs straddled, which is the
        # measurement that explains the refusal and is worth having whether or not the arm can run.
        sweep_path, corpus_path = self.write_inputs(
            tmp_path, {"narrows-approach": (0.875, 0.375), "manifold-gas-bank": (0.9, 0.95)}
        )
        out_dir = tmp_path / "partition"
        with pytest.raises(ValueError, match="holds no prompts"):
            cp.main(
                [
                    "--sweep",
                    str(sweep_path),
                    "--corpus",
                    str(corpus_path),
                    "--payoff-variant",
                    "safe-hunt",
                    "--out-dir",
                    str(out_dir),
                ]
            )
        [artifact_path] = sorted(out_dir.glob("partition-*.json"))
        artifact = json.loads(artifact_path.read_text())
        assert artifact["summary"]["n_straddling_pairs"] == 1
        assert sorted(out_dir.glob("corpus-*-partitioned.jsonl"))

    def test_the_boundary_mode_is_in_the_filename_so_two_splits_cannot_be_confused(
        self, tmp_path: Path
    ):
        sweep_path, corpus_path = self.write_inputs(tmp_path, dict(TWO_SIDED_PAIRS))
        out_dir = tmp_path / "partition"
        cp.main(
            [
                "--sweep",
                str(sweep_path),
                "--corpus",
                str(corpus_path),
                "--payoff-variant",
                "safe-hunt",
                "--boundary-mode",
                cp.BOUNDARY_PAIR_MEAN_MEDIAN,
                "--out-dir",
                str(out_dir),
            ]
        )
        assert sorted(path.name for path in out_dir.glob("partition-*.json")) == [
            f"partition-stag-hunt-Qwen3.5-2B-safe-hunt-{cp.BOUNDARY_PAIR_MEAN_MEDIAN}.json"
        ]
