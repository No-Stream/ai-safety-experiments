"""Split one swept corpus by which side of the group-mix boundary each prompt started on.

Two arms over the same game, the same grading, the same payoff cells and the same prose, differing
only in *which prompts* they train on: one corpus of prompts the base policy already answered
cooperatively, one of prompts it did not. Under group-mix grading the reward gap favours cooperation
exactly when the group's own cooperation rate is above the game's boundary
(`games.payoffs.group_mix_gap_crossing`), and a group is one prompt's completions, so the two
corpora are being pushed in opposite directions by identical machinery. That is the cleanest
separation available between what the training dynamics did and what the game's words say, which is
the confound every other design in this project has to argue around.

Four things this module refuses to do casually, each earned.

**A counterbalanced pair is never split.** Every scenario is rendered twice, once with each neutral
label naming the cooperative option, and the two orientations' baseline cooperation rates differ a
lot: on the 2026-08-19 safe-hunt sweep the mean within-pair gap is 0.319 across all sixteen scenarios
and 0.213 across the ten that survived selection, against per-prompt rates spanning 0.125 to 1.0.
Assigning per prompt would therefore put the two halves of one scenario in opposite partitions, and
each partition would hold prompts being trained the wrong way while every count still looked healthy.
So a pair goes wholly into one side or is labelled `straddling-pair`, `assert_pairs_are_whole` is the
check that says so, and the straddle count is a first-class number rather than a silent subtraction.
It is a large number: 7 of those 10 pairs straddle, which is what makes the count worth reporting
rather than absorbing.

**A rate exactly on the boundary has no side.** The reward gap there is exactly zero, so the prompt
is pushed in neither direction, and calling it "above" would manufacture a direction out of a tie.
Not hypothetical: two of safe-hunt's sixteen frames carry an orientation at exactly 0.750 against a
boundary of 0.75.

**The boundary is the trainer's own number.** It comes from the shared crossing function, under the
group size and leave-one-out setting the run will actually use, because leave-one-out MOVES the
crossing. Partitioning at the plain boundary and then training with the flag on would sort every
prompt by a number the run never used.

**The partition is written down, not recomputed.** A split derived from "whatever records are on
disk" silently re-picks itself the next time a sweep is added, so the artifact records the sweep's
own prompt-order hash and the corpus rows carry their side and their boundary as columns. Training
reads the stamped corpus; nothing downstream re-derives the assignment.

    uv run python -m games.corpus_partition --sweep <sweep.jsonl> --corpus <corpus.jsonl> \
        --payoff-variant safe-hunt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from games.arms import (
    CORPUS_PARTITION_COLUMN,
    PARTITION_ABOVE_THRESHOLD,
    PARTITION_BELOW_THRESHOLD,
    PAYOFF_VARIANT_COLUMN,
)
from games.payoffs import MatrixGameSpec, group_mix_gap_crossing, group_mix_gap_slope
from games.provenance import git_provenance
from games.select_prompts import (
    LABEL_ORIENTATIONS_PER_SCENARIO,
    META_RECORD_KIND,
    PROMPT_ID_COLUMN,
    SWEEP_RECORD_KIND,
    is_counterbalanced,
    pair_identity,
    read_jsonl,
    write_corpus,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

type Row = dict[str, object]
type PairKey = tuple[object, ...]

# A pair whose two label orientations landed on opposite sides of the boundary, or one of whose
# orientations landed exactly on it. Stamped onto the rows and kept in the partitioned corpus rather
# than deleted from it, so the partitions' own denominator stays visible in the file.
PARTITION_STRADDLING_PAIR = "straddling-pair"

# Stamped beside the side, so a readout can report each row's margin from its boundary without
# reopening the partition artifact -- and so a corpus whose rows were partitioned at a boundary
# nobody wrote down cannot exist.
CORPUS_PARTITION_BOUNDARY_COLUMN = "corpus_partition_boundary"

# Where the boundary comes from. The crossing is the designed one: the rate at which group-mix
# grading changes which action it rewards, so the two sides genuinely train in opposite directions.
# The median of the pair means is the documented fallback for a corpus whose rates do not straddle
# the crossing -- it still separates high-baseline prompts from low-baseline ones, but the claim it
# supports is only "the two sides started from different points", never "opposite directions", and
# naming it in the artifact is what keeps those two claims from being read as the same result.
BOUNDARY_GROUP_MIX_CROSSING = "group-mix-crossing"
BOUNDARY_PAIR_MEAN_MEDIAN = "pair-mean-median"
BOUNDARY_MODES: tuple[str, ...] = (BOUNDARY_GROUP_MIX_CROSSING, BOUNDARY_PAIR_MEAN_MEDIAN)

# `games.train.GameTrainConfig.num_generations`, restated rather than imported because that module
# pulls in the whole training stack; `test_games_corpus_partition` asserts the two stay equal.
DEFAULT_NUM_GENERATIONS = 8

PARTITION_RECORD_KIND = "corpus-partition"
DEFAULT_OUT_DIR = Path("artifacts/games/partition")

# Cells of the row the boundary is computed from. Named so a corpus missing one fails with the
# column list rather than a KeyError from inside the arithmetic.
PAYOFF_CELL_COLUMNS = ("payoff_cc", "payoff_cd", "payoff_dc", "payoff_dd")


@dataclass(frozen=True, slots=True)
class PairAssignment:
    """One counterbalanced pair's side of the boundary, and the measurements behind the verdict.

    `pair_mean_partition` is the contamination diagnostic rather than the verdict: it records where
    the pair's *mean* rate falls, which is what a simpler partitioner keyed on the pair mean would
    have decided. Comparing the two columns says how many scenarios a mean-based split would have
    placed confidently in a partition while one of its own orientations sat on the other side.
    """

    reskin_id: str
    payoff_variant: str
    prompt_ids: tuple[str, ...]
    coop_fractions: tuple[float, ...]
    boundary: float
    partition: str
    pair_mean_partition: str

    @property
    def pair_mean(self) -> float:
        """Return the mean baseline cooperation rate over the pair's two orientations."""
        return statistics.fmean(self.coop_fractions)

    @property
    def within_pair_gap(self) -> float:
        """Return the spread between the orientations, the quantity that makes pairs straddle."""
        return max(self.coop_fractions) - min(self.coop_fractions)

    @property
    def is_straddling(self) -> bool:
        """Say whether the pair was dropped from both partitions for lack of one shared side."""
        return self.partition == PARTITION_STRADDLING_PAIR

    def to_json_dict(self) -> dict[str, object]:
        """Flatten into the artifact's per-pair record, derived quantities materialised."""
        return {
            "reskin_id": self.reskin_id,
            "payoff_variant": self.payoff_variant,
            "prompt_ids": list(self.prompt_ids),
            "coop_fractions": list(self.coop_fractions),
            "boundary": self.boundary,
            "partition": self.partition,
            "pair_mean_partition": self.pair_mean_partition,
            "pair_mean": self.pair_mean,
            "within_pair_gap": self.within_pair_gap,
        }


def side_of_boundary(coop_fraction: float, boundary: float) -> str | None:
    """Return which side of the boundary a cooperation rate sits on, or None when it sits ON it.

    None is not a missing value: the group-mix reward gap is exactly zero at the boundary, so a
    prompt there is pushed in neither direction and giving it a side would invent one from a tie.
    """
    if coop_fraction > boundary:
        return PARTITION_ABOVE_THRESHOLD
    if coop_fraction < boundary:
        return PARTITION_BELOW_THRESHOLD
    return None


def baseline_coop_fractions(records: Sequence[dict[str, object]]) -> dict[str, float]:
    """Read one baseline cooperation rate per prompt off a written sweep trace.

    Keyed on `record_kind`, so the trace's meta line and any frozen-opponent records in the same
    file are skipped rather than read as policy measurements -- the opponent's rate describes a
    different model and would put prompts on the wrong side of the boundary.

    Raises on a prompt whose rate is null. That happens for the gradings whose completions carry no
    action (the unilateral split, where the score is a kept fraction), and for those games this whole
    module is meaningless rather than merely unavailable: there is no cooperation rate to compare to
    a boundary.
    """
    fractions: dict[str, float] = {}
    for record in records:
        if record.get("record_kind") != SWEEP_RECORD_KIND:
            continue
        prompt_id = record[PROMPT_ID_COLUMN]
        if not isinstance(prompt_id, str):
            raise TypeError(f"sweep record carries a non-string prompt_id: {prompt_id!r}")
        if prompt_id in fractions:
            raise ValueError(
                f"prompt {prompt_id!r} appears twice in the sweep trace; one prompt's baseline rate "
                "cannot be two numbers, and the later record would silently win"
            )
        coop_fraction = record.get("coop_fraction")
        if coop_fraction is None:
            raise ValueError(
                f"prompt {prompt_id!r} carries coop_fraction=None, so this sweep measured no "
                f"cooperation rate for it. A corpus can only be split by baseline cooperation if "
                f"its grading produces one, and the unilateral-split grading does not measure a "
                f"per-prompt rate this partition could use."
            )
        if isinstance(coop_fraction, bool) or not isinstance(coop_fraction, float | int):
            raise TypeError(
                f"prompt {prompt_id!r} carries a non-numeric coop_fraction {coop_fraction!r}, so "
                f"this trace is not the sweep record it claims to be"
            )
        fractions[prompt_id] = float(coop_fraction)
    if not fractions:
        raise ValueError(
            f"the sweep trace holds no {SWEEP_RECORD_KIND!r} records, so there are no baseline "
            f"rates to partition on. A trace holding only its meta line is a sweep that wrote "
            f"provenance and then failed."
        )
    return fractions


def row_spec(row: Row) -> MatrixGameSpec:
    """Build the payoff spec of one corpus row, naming the columns when one is missing."""
    missing = [column for column in PAYOFF_CELL_COLUMNS if column not in row]
    if missing:
        raise ValueError(
            f"row is missing payoff cells {missing}, so its group-mix boundary cannot be computed; "
            f"columns present: {sorted(row)}"
        )
    game_id = row.get("game_id")
    return MatrixGameSpec(
        game_id=str(game_id),
        payoff_cc=float(row["payoff_cc"]),  # pyright: ignore[reportArgumentType]
        payoff_cd=float(row["payoff_cd"]),  # pyright: ignore[reportArgumentType]
        payoff_dc=float(row["payoff_dc"]),  # pyright: ignore[reportArgumentType]
        payoff_dd=float(row["payoff_dd"]),  # pyright: ignore[reportArgumentType]
    )


def row_boundary(row: Row, *, num_generations: int, leave_one_out: bool) -> float:
    """Return the group-mix boundary this row's own payoff cells imply.

    Refuses the two ways a game can fail to have one: a gap that never changes sign (dominance --
    every prompt is pushed the same way whatever its baseline, so there is nothing to split), and a
    crossing outside [0, 1] (the same thing, reached by different arithmetic). A refusal here is
    cheaper than a partition whose two sides are the same experiment under two names.
    """
    spec = row_spec(row)
    boundary = group_mix_gap_crossing(
        spec, num_generations=num_generations, leave_one_out=leave_one_out
    )
    if boundary is None or not 0.0 < boundary < 1.0:
        raise ValueError(
            f"{spec.game_id!r} has no interior group-mix boundary "
            f"(crossing={boundary!r}, slope={group_mix_gap_slope(spec)}), so group-mix grading "
            f"pushes every prompt the same way regardless of its baseline mix and a mix-split over "
            f"it would produce two labels for one experiment. Cells: cc={spec.payoff_cc} "
            f"cd={spec.payoff_cd} dc={spec.payoff_dc} dd={spec.payoff_dd}."
        )
    return boundary


def scenario_identity(row: Row) -> PairKey:
    """Return the pair key of a row, ignoring anything a partition wrote onto it.

    `pair_identity` keys on every column the label swap leaves alone, which is what makes it tighten
    rather than loosen as columns are added -- and that includes the two columns stamped here. So on a
    stamped corpus it would give the two orientations of a *split* pair two different keys, and the
    split pair, which is the one thing `assert_pairs_are_whole` exists to catch, would arrive there
    looking like two unrelated singletons. Stripping the stamp before keying is what lets that check
    see the violation, and it also makes re-partitioning an already-stamped corpus behave.
    """
    return pair_identity(
        {
            column: value
            for column, value in row.items()
            if column not in (CORPUS_PARTITION_COLUMN, CORPUS_PARTITION_BOUNDARY_COLUMN)
        }
    )


def group_rows_by_pair(rows: Sequence[Row]) -> dict[PairKey, list[Row]]:
    """Group corpus rows into counterbalanced pairs, refusing anything that is not a clean pair.

    The pair key comes from `games.select_prompts.pair_identity` (through `scenario_identity`) rather
    than from a hand-written subset of columns, so the two modules cannot disagree about what a pair
    is. A singleton or a triple is refused here instead of being partitioned: selection already
    couples pairs, so an odd group means the corpus was filtered by something that did not know about
    the counterbalance, and a lone orientation walks the position bias it exists to cancel straight
    into one partition.
    """
    groups: dict[PairKey, list[Row]] = {}
    for row in rows:
        groups.setdefault(scenario_identity(row), []).append(row)
    malformed = {
        _pair_label(group): [row.get(PROMPT_ID_COLUMN) for row in group]
        for group in groups.values()
        if len(group) != LABEL_ORIENTATIONS_PER_SCENARIO
    }
    if malformed:
        raise ValueError(
            f"{len(malformed)} scenarios are not counterbalanced pairs in this corpus: {malformed}. "
            f"Each is rendered in both label orientations so that a preference for the first-listed "
            f"option cannot masquerade as a preference for cooperating, and a partition assigns "
            f"whole pairs. A singleton here means an upstream filter split one."
        )
    unpaired = sorted(
        str(group[0].get(PROMPT_ID_COLUMN))
        for group in groups.values()
        if not is_counterbalanced(group[0])
    )
    if unpaired:
        raise ValueError(
            f"{len(unpaired)} rows carry no option labels to swap, so they have no counterbalanced "
            f"partner and no cooperation rate to compare against a boundary: {unpaired}. The "
            f"unilateral-split game cannot be mix-split."
        )
    return groups


def _pair_label(group: Sequence[Row]) -> str:
    """Name a pair by its scenario and payoff variant, for a message a reader can act on."""
    first = group[0]
    return f"{first.get('reskin_id')}--{first.get(PAYOFF_VARIANT_COLUMN)}"


def _pair_rates(group: Sequence[Row], coop_fractions: dict[str, float]) -> tuple[float, ...]:
    """Look up both orientations' baseline rates, naming a prompt the sweep never measured."""
    rates: list[float] = []
    for row in group:
        prompt_id = str(row.get(PROMPT_ID_COLUMN))
        if prompt_id not in coop_fractions:
            raise KeyError(
                f"prompt {prompt_id!r} is in the corpus but not in the sweep trace, so its baseline "
                f"cooperation rate is unknown and it cannot be assigned a side. The corpus and the "
                f"sweep are from different runs, or the trace is truncated."
            )
        rates.append(coop_fractions[prompt_id])
    return tuple(rates)


def _pair_boundaries(
    groups: dict[PairKey, list[Row]],
    rates: dict[PairKey, tuple[float, ...]],
    *,
    boundary_mode: str,
    num_generations: int,
    leave_one_out: bool,
) -> dict[PairKey, float]:
    """Return the boundary each pair is judged against, under the requested mode."""
    if boundary_mode == BOUNDARY_PAIR_MEAN_MEDIAN:
        median = statistics.median(statistics.fmean(pair) for pair in rates.values())
        return dict.fromkeys(groups, median)
    boundaries: dict[PairKey, float] = {}
    for key, group in groups.items():
        per_row = {
            row_boundary(row, num_generations=num_generations, leave_one_out=leave_one_out)
            for row in group
        }
        if len(per_row) != 1:
            raise ValueError(
                f"the two orientations of {_pair_label(group)} imply different boundaries "
                f"{sorted(per_row)}, so they are not two renderings of one payoff matrix. Two "
                f"different scenarios have collided on one pair key."
            )
        boundaries[key] = per_row.pop()
    return boundaries


def assign_pairs(
    rows: Sequence[Row],
    coop_fractions: dict[str, float],
    *,
    boundary_mode: str = BOUNDARY_GROUP_MIX_CROSSING,
    num_generations: int = DEFAULT_NUM_GENERATIONS,
    leave_one_out: bool = False,
) -> list[PairAssignment]:
    """Assign every counterbalanced pair in the corpus to a side of its boundary.

    A pair lands in a partition only when BOTH orientations are on the same side of the boundary;
    otherwise it is labelled `straddling-pair` and counted. That is the whole design decision: the
    two alternatives are keying on the pair mean, which places a straddling scenario confidently in
    a partition where half of it is being trained the other way, and dropping the counterbalance,
    which walks a position bias into the corpus. Both are invisible afterwards. This costs corpus
    size, which is at least a number the summary reports.
    """
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(f"unknown boundary_mode {boundary_mode!r}; known: {list(BOUNDARY_MODES)}")
    if not rows:
        raise ValueError("no corpus rows to partition")
    groups = group_rows_by_pair(rows)
    rates = {key: _pair_rates(group, coop_fractions) for key, group in groups.items()}
    boundaries = _pair_boundaries(
        groups,
        rates,
        boundary_mode=boundary_mode,
        num_generations=num_generations,
        leave_one_out=leave_one_out,
    )
    assignments: list[PairAssignment] = []
    for key, group in groups.items():
        boundary = boundaries[key]
        pair_rates = rates[key]
        sides = {side_of_boundary(rate, boundary) for rate in pair_rates}
        shared = sides.pop() if len(sides) == 1 else None
        mean_side = side_of_boundary(statistics.fmean(pair_rates), boundary)
        assignments.append(
            PairAssignment(
                reskin_id=str(group[0].get("reskin_id")),
                payoff_variant=str(group[0].get(PAYOFF_VARIANT_COLUMN)),
                prompt_ids=tuple(str(row.get(PROMPT_ID_COLUMN)) for row in group),
                coop_fractions=pair_rates,
                boundary=boundary,
                partition=shared if shared is not None else PARTITION_STRADDLING_PAIR,
                pair_mean_partition=(
                    mean_side if mean_side is not None else PARTITION_STRADDLING_PAIR
                ),
            )
        )
    return assignments


def partition_summary(assignments: Sequence[PairAssignment]) -> dict[str, object]:
    """Count what each partition got, and what it cost to keep the pairs whole.

    Every number a reader needs to judge the split before a GPU is booked: the per-side prompt
    counts against the straddle count that explains them, each side's own baseline distribution
    (the arm's starting point, which its direction has to be read against), the within-pair gap
    that drives straddling, and how many scenarios a pair-mean split would have placed differently.
    """
    by_partition: dict[str, list[PairAssignment]] = {}
    for assignment in assignments:
        by_partition.setdefault(assignment.partition, []).append(assignment)
    contaminated = [
        assignment.reskin_id
        for assignment in assignments
        if assignment.is_straddling and assignment.pair_mean_partition != PARTITION_STRADDLING_PAIR
    ]
    gaps = [assignment.within_pair_gap for assignment in assignments]
    return {
        "n_pairs": len(assignments),
        "n_prompts": sum(len(assignment.prompt_ids) for assignment in assignments),
        "pairs_by_partition": {
            partition: len(members) for partition, members in sorted(by_partition.items())
        },
        "prompts_by_partition": {
            partition: sum(len(member.prompt_ids) for member in members)
            for partition, members in sorted(by_partition.items())
        },
        "n_straddling_pairs": len(by_partition.get(PARTITION_STRADDLING_PAIR, [])),
        "baseline_by_partition": {
            partition: _rate_stats([rate for member in members for rate in member.coop_fractions])
            for partition, members in sorted(by_partition.items())
        },
        "boundaries": sorted({assignment.boundary for assignment in assignments}),
        "within_pair_gap_mean": statistics.fmean(gaps) if gaps else None,
        "within_pair_gap_median": statistics.median(gaps) if gaps else None,
        "pairs_a_mean_split_would_have_kept": sorted(contaminated),
    }


def _rate_stats(rates: Sequence[float]) -> dict[str, float | None]:
    """Describe one partition's baseline cooperation rates."""
    if not rates:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(rates),
        "mean": statistics.fmean(rates),
        "min": min(rates),
        "max": max(rates),
    }


def assert_partitions_train_in_opposite_directions(
    rows: Sequence[Row], coop_fractions: dict[str, float]
) -> None:
    """Refuse a stamped corpus whose two sides are not pushed in opposite directions.

    Deliberately re-derived from three inputs that reached this point separately -- the side stamped
    on the row, the boundary stamped beside it, and the sweep's own rate for that prompt id -- rather
    than from the `PairAssignment` objects that wrote them. Asked of the assignments this would be
    tautological: rows were put on the above side *because* their rate exceeded the boundary, so of
    course they exceed it. Asked of the written file it is a real question, and it is the one the
    trainer's own reader will effectively be relying on: it catches a mislabelled row, a boundary
    column that does not match the label, a corpus appended to after the split, and the case the
    straddle count predicts, where one side came out empty and there is no contrast to train.
    """
    per_side: dict[str, list[float]] = {
        PARTITION_ABOVE_THRESHOLD: [],
        PARTITION_BELOW_THRESHOLD: [],
    }
    for row in rows:
        partition = str(row.get(CORPUS_PARTITION_COLUMN))
        if partition == PARTITION_STRADDLING_PAIR:
            continue
        if partition not in per_side:
            raise ValueError(
                f"row {row.get(PROMPT_ID_COLUMN)!r} carries partition {partition!r}, which is not a "
                f"side of any boundary; known: {sorted([*per_side, PARTITION_STRADDLING_PAIR])}"
            )
        prompt_id = str(row.get(PROMPT_ID_COLUMN))
        if prompt_id not in coop_fractions:
            raise KeyError(
                f"stamped row {prompt_id!r} has no baseline rate in the sweep, so which side of the "
                f"boundary it belongs on cannot be checked against the file that decided it"
            )
        rate = coop_fractions[prompt_id]
        boundary = float(row[CORPUS_PARTITION_BOUNDARY_COLUMN])  # pyright: ignore[reportArgumentType]
        if side_of_boundary(rate, boundary) != partition:
            raise ValueError(
                f"row {prompt_id!r} is stamped {partition!r} but its baseline rate {rate} against "
                f"its own boundary {boundary} puts it on {side_of_boundary(rate, boundary)!r}. "
                f"Training it under the pinned arm would push it the way its baseline says it is "
                f"already going, which is the opposite of what the arm claims to measure."
            )
        per_side[partition].append(rate)
    empty = sorted(partition for partition, rates in per_side.items() if not rates)
    if empty:
        counts = {partition: len(rates) for partition, rates in sorted(per_side.items())}
        straddling = sum(
            1 for row in rows if row.get(CORPUS_PARTITION_COLUMN) == PARTITION_STRADDLING_PAIR
        )
        raise ValueError(
            f"partition {empty[0]!r} holds no prompts, so there is no two-sided contrast to train: "
            f"prompts_by_side={counts}, and {straddling} of {len(rows)} rows belong to pairs that "
            f"straddle their boundary, which is where the rest went. Widen the corpus (more "
            f"scenarios per rung) or split at a boundary the measured rates actually straddle."
        )
    logger.info(
        "both sides of the boundary are populated and correctly signed, %s",
        f"below_mean={statistics.fmean(per_side[PARTITION_BELOW_THRESHOLD]):.3f} "
        f"above_mean={statistics.fmean(per_side[PARTITION_ABOVE_THRESHOLD]):.3f}",
    )


def assert_pairs_are_whole(rows: Sequence[Row]) -> None:
    """Refuse stamped rows where one counterbalanced pair carries two different partitions.

    The guard for the mistake the whole module is shaped around. Assigning per prompt instead of per
    pair produces exactly this: two orientations of one scenario in opposite partitions, each
    trained in the direction the other one contradicts, with every count and every curve looking
    normal. Checked on the written rows rather than inside the assignment loop, so it also catches a
    corpus stamped by something other than `stamp_partitions`.
    """
    for group in group_rows_by_pair(rows).values():
        partitions = {str(row.get(CORPUS_PARTITION_COLUMN)) for row in group}
        if len(partitions) != 1:
            raise ValueError(
                f"the two label orientations of {_pair_label(group)} carry different partitions "
                f"{sorted(partitions)}. A counterbalanced pair is one scenario rendered twice, so "
                f"splitting it puts the same scenario in two arms being trained in opposite "
                f"directions. prompt_ids: {[row.get(PROMPT_ID_COLUMN) for row in group]}"
            )


def stamp_partitions(rows: Sequence[Row], assignments: Sequence[PairAssignment]) -> list[Row]:
    """Return the corpus rows with their partition and boundary written on, pairs kept whole.

    Copies rather than mutating, so the corpus this was derived from stays as it was written. Every
    row must have an assignment: a row with none would reach training unpinnable and be trained by
    whichever arm loaded the file.
    """
    by_prompt_id = {
        prompt_id: assignment for assignment in assignments for prompt_id in assignment.prompt_ids
    }
    stamped: list[Row] = []
    for row in rows:
        prompt_id = str(row.get(PROMPT_ID_COLUMN))
        if prompt_id not in by_prompt_id:
            raise KeyError(
                f"prompt {prompt_id!r} has no partition assignment, so the stamped corpus would "
                f"carry a row no arm can pin and every arm would train it"
            )
        assignment = by_prompt_id[prompt_id]
        stamped.append(
            {
                **row,
                CORPUS_PARTITION_COLUMN: assignment.partition,
                CORPUS_PARTITION_BOUNDARY_COLUMN: assignment.boundary,
            }
        )
    assert_pairs_are_whole(stamped)
    return stamped


def sweep_provenance(
    records: Sequence[dict[str, object]], *, sweep_path: Path
) -> dict[str, object]:
    """Carry the sweep's own identity into the partition artifact.

    The prompt-order hash is the load-bearing field. A partition is a derived selection, and a
    derived selection computed against a pool that has since grown re-picks itself silently: the
    same code over one more scenario returns a different split, and nothing in either artifact says
    the two are not comparable. The hash is what says it.

    Sweeps written before the meta line grew `prompt_id_order_sha256` (the 2026-08-19 wave-1
    traces) still fix the pool order: `write_sweep_trace` writes the policy records in the same
    list order `sweep_meta` hashes, so hashing the trace's own record order reproduces the number
    the meta would have carried. `prompt_id_order_sha256_source` says which path produced it, so a
    reader can tell a recorded pin from a derived one.
    """
    meta = next(
        (record for record in records if record.get("record_kind") == META_RECORD_KIND), None
    )
    if meta is None:
        raise ValueError(
            f"{sweep_path} carries no meta record, so the pool this partition was derived from "
            f"cannot be identified. Two partitions over pools whose prompt-order hashes differ are "
            f"not a controlled comparison, and the trace's first line is the only thing that says "
            f"so."
        )
    order_hash = meta.get("prompt_id_order_sha256")
    order_hash_source = "meta-record"
    if order_hash is None:
        prompt_ids = [
            str(record[PROMPT_ID_COLUMN])
            for record in records
            if record.get("record_kind") == SWEEP_RECORD_KIND
        ]
        order_hash = hashlib.sha256("\n".join(prompt_ids).encode("utf-8")).hexdigest()
        order_hash_source = "derived-from-trace-record-order"
    backend = meta.get("backend")
    return {
        "sweep_path": str(sweep_path),
        "prompt_id_order_sha256": order_hash,
        "prompt_id_order_sha256_source": order_hash_source,
        "sweep_written_at": meta.get("written_at"),
        "sweep_n_prompts": meta.get("n_prompts"),
        "sweep_samples_per_prompt": meta.get("samples_per_prompt"),
        "sweep_model_id": backend.get("model_id") if isinstance(backend, dict) else None,
    }


def partition_artifact(  # noqa: PLR0913
    *,
    assignments: Sequence[PairAssignment],
    provenance: dict[str, object],
    corpus_path: Path,
    partitioned_corpus_path: Path,
    boundary_mode: str,
    num_generations: int,
    leave_one_out: bool,
    payoff_variant: str,
) -> dict[str, object]:
    """Build the record that says what was split, from what, and under which boundary."""
    return {
        "record_kind": PARTITION_RECORD_KIND,
        "written_at": datetime.now(UTC).isoformat(),
        "boundary_mode": boundary_mode,
        "num_generations": num_generations,
        "leave_one_out": leave_one_out,
        "payoff_variant": payoff_variant,
        "corpus_path": str(corpus_path),
        "partitioned_corpus_path": str(partitioned_corpus_path),
        "sweep": provenance,
        "summary": partition_summary(assignments),
        "pairs": [assignment.to_json_dict() for assignment in assignments],
        **git_provenance(),
    }


def filter_payoff_variant(rows: Sequence[Row], payoff_variant: str) -> list[Row]:
    """Keep the rows of one payoff variant, which is the unit a partition is computed over.

    One artifact per rung, because the boundary is a property of the payoff cells: a partition
    spanning variants would differ from its twin in payoff mix as well as in baseline behaviour,
    and `games.arms` refuses such an arm for that reason.
    """
    kept = [row for row in rows if row.get(PAYOFF_VARIANT_COLUMN) == payoff_variant]
    if not kept:
        present = sorted({str(row.get(PAYOFF_VARIANT_COLUMN)) for row in rows})
        raise ValueError(
            f"no rows carry payoff variant {payoff_variant!r}; this corpus carries {present}"
        )
    return kept


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a swept corpus into the sub-corpora group-mix training pushes in opposite "
            "directions, keeping counterbalanced label pairs whole."
        )
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        required=True,
        help="Sweep trace (JSONL) whose per-prompt cooperation rates the split is keyed on.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Selected corpus (JSONL) whose rows get a partition stamped onto them.",
    )
    parser.add_argument(
        "--payoff-variant",
        required=True,
        help="The one payoff variant to partition; the boundary is a property of its cells.",
    )
    parser.add_argument(
        "--boundary-mode",
        choices=BOUNDARY_MODES,
        default=BOUNDARY_GROUP_MIX_CROSSING,
        help=(
            "Where the boundary comes from. The group-mix crossing is the rate at which training "
            "changes which action it rewards, and is the only mode whose two sides are opposite "
            "directions. The pair-mean median is the fallback for a corpus whose rates do not "
            f"straddle that crossing (default: {BOUNDARY_GROUP_MIX_CROSSING})."
        ),
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=DEFAULT_NUM_GENERATIONS,
        help=(
            "Group size the run will train at. Read only under --leave-one-out, where it moves the "
            f"boundary (default: {DEFAULT_NUM_GENERATIONS})."
        ),
    )
    parser.add_argument(
        "--leave-one-out",
        action="store_true",
        help=(
            "Grade each completion against the OTHER completions in its group, as games.train's own "
            "flag does. It moves the boundary, so a partition and the run that trains it must agree."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for the partition artifact and the stamped corpus (default: {DEFAULT_OUT_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Partition one corpus, write the artifact and the stamped corpus, then check it is trainable.

    The write order is the point. The artifact is the measurement -- how many pairs straddled, what
    each side's baseline looks like, how big each partition came out -- and it is worth having
    whether or not the split turned out runnable. `assert_partitions_train_in_opposite_directions`
    can legitimately refuse a corpus (a one-sided split is one experiment under two names), so it
    runs last, after the numbers that explain the refusal are already on disk.

    It also runs against the corpus **read back from disk** rather than the rows still in memory.
    Same reason the readback exists anywhere else: the file is what training will open, and a check
    against the in-memory copy cannot see a truncated or half-written one.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = _parse_args(argv)
    sweep_records = read_jsonl(args.sweep)
    corpus_rows = filter_payoff_variant(read_jsonl(args.corpus), args.payoff_variant)
    logger.info(
        f"partitioning {len(corpus_rows)} rows of payoff_variant={args.payoff_variant!r} against "
        f"boundary_mode={args.boundary_mode!r} num_generations={args.num_generations} "
        f"leave_one_out={args.leave_one_out}"
    )

    coop_fractions = baseline_coop_fractions(sweep_records)
    assignments = assign_pairs(
        corpus_rows,
        coop_fractions,
        boundary_mode=args.boundary_mode,
        num_generations=args.num_generations,
        leave_one_out=args.leave_one_out,
    )
    stamped = stamp_partitions(corpus_rows, assignments)

    # The source corpus is already named `corpus-...`, and re-prefixing it reads as a stutter. The
    # variant and the boundary mode ARE in the name on purpose: two splits of one corpus at two
    # boundaries are two different experiments, and a filename that cannot tell them apart is how the
    # 2026-08-19 rung miscount happened.
    stem = f"{args.corpus.stem.removeprefix('corpus-')}-{args.payoff_variant}-{args.boundary_mode}"
    partitioned_corpus_path = args.out_dir / f"corpus-{stem}-partitioned.jsonl"
    artifact_path = args.out_dir / f"partition-{stem}.json"
    artifact = partition_artifact(
        assignments=assignments,
        provenance=sweep_provenance(sweep_records, sweep_path=args.sweep),
        corpus_path=args.corpus,
        partitioned_corpus_path=partitioned_corpus_path,
        boundary_mode=args.boundary_mode,
        num_generations=args.num_generations,
        leave_one_out=args.leave_one_out,
        payoff_variant=args.payoff_variant,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    write_corpus(partitioned_corpus_path, stamped)
    summary = artifact["summary"]
    logger.info(f"wrote {artifact_path} and {partitioned_corpus_path}")
    logger.info(f"partition summary: {summary}")

    assert_partitions_train_in_opposite_directions(
        read_jsonl(partitioned_corpus_path), coop_fractions
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
