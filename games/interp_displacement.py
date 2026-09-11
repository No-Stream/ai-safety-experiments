"""Population-displacement geometry over cached capture cells: how far each arm moved, and together.

`games.interp_trajectory` compares one arm's *axes* across checkpoints; `games.interp_axes` compares
axes to each other. Neither computes the object a "the two arms moved in near-orthogonal directions"
claim is actually about, which is the population-mean displacement

    D(arm, step, layer) = mean over stimulus rows of (h_arm - h_base)

-- a shift of where the whole stimulus population sits, rather than a change in how that population
is separated. This module is that read. It exists because the first pooled look at it (a cross-arm
cosine of 0.092) came with no statement of what noise would produce, and a cosine of 0.09 in 2048
dimensions is unreadable without one: an unrelated pair of directions reads ~1/sqrt(2048) = 0.022,
not 0. Everything here is arithmetic over the cached cells, so it is CPU-only and re-runnable.

Every cross-arm cosine is therefore reported between two references:

* the **placebo floor** -- matched-norm random directions against the same displacement, which is
  what "unrelated" reads at this dimensionality;
* the **same-arm split-half ceiling** -- each arm's displacement re-computed on half the stimuli, so
  `cos(half, half)` is that displacement's own reliability, and the honest comparator for a cosine
  between two noisy directions is `sqrt(reliability_a * reliability_b)` (an attenuation-style
  heuristic, flagged as such: cosines are not correlations).

**Two split-halves, deliberately, because the obvious one is confounded on this corpus.** The rows
of every set are stored A, B, A, B -- the two sides of each matched pair adjacent -- so an
even-row/odd-row split is a *side* split (every side-A text against every side-B text) rather than
two interchangeable halves of the population. That conflates measurement noise with any genuine
side-dependence of the displacement, so the primary reliability here splits **pairs** (each half
holding both sides of a disjoint set of pairs, `split_half_cosine_pairs`). The row split is kept
beside it (`split_half_cosine_rows`), with `row_parity_equals_side` recorded per read so the number
says what it is. The scratch script this module was promoted from reported only the row split.

Also reported per read, because "the arms moved orthogonally" and "the arms moved oppositely along
one shared axis and the rest is noise" are different claims a pooled cosine cannot separate: the
split of each displacement into its component along a contrast axis (signed projection) and the
residual norm around it, for every stimulus set's axis, fitted at the base anchor (`base`) and at the
arm's own checkpoint (`own`). The axis is always fitted on all of its set's pairs even inside a
stratified read: stratifying the displacement while holding the axis fixed is the readable
comparison, whereas letting both move makes a change in either look like a change in the other.

Reads are produced pooled over the stimulus sets and per set, and within each `cited_cells` stratum
and each `counterpart_framing` half, at every layer and checkpoint step. A selection whose rows are
the same rows as its pooled parent, or which is empty, is skipped and *recorded* in the payload's
`skipped_selections` rather than silently dropped.

Three checks ride along, each of which has to be watched to fail before the numbers it guards are
worth anything:

* `--frozen-cells` runs the **two-path agreement** read (plan step 1d): the same corpus captured by
  two independent implementations should give the same diff-of-means direction to ~1.0, and the same
  comparison run one layer off must collapse. The off-by-one arm is computed on every run, and a
  run where it does *not* collapse raises, because a check that cannot go red is not a check. On the
  games corpus the converted frozen ladder's **top layer is post-final-RMSNorm** rather than a
  decoder-block output -- a fifth capture-format difference beyond the four the plan tabulated -- so
  it is compared and reported separately and excluded from the agreement summary.
* the **shuffled-label null**: the coarse axis re-fitted on pair labels flipped at random must land
  at the placebo floor and must not clear its placebo accuracy.
* the **base-against-base null**: the whole displacement machinery run with the base cell standing in
  for an arm. Because a displacement is a paired difference over the same rows, the answer is exactly
  zero rather than "at the placebo floor" as the plan's table guessed, which makes it a sharp check
  on the row pairing: anything nonzero means rows are being differenced against the wrong rows, and
  the module refuses to report the rest of the analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from games.interp_axes import POOLED_STRATUM, StimulusStrata, load_strata
from games.interp_cells import (
    BASE_ARM,
    BASE_STEP,
    CapturedCell,
    CellFormatError,
    Ladder,
    PairLayout,
    concept_activations,
    load_ladder,
    load_stimuli,
    pair_layout,
    stimuli_digest,
)
from games.interp_trajectory import analysis_arms
from reward_hacking.interp.directions import (
    cosine,
    diff_of_means,
    matched_norm_random_direction,
    unit,
)
from reward_hacking.interp.eval_awareness_probe import DEFAULT_N_PLACEBOS, direction_separation
from reward_hacking.interp.linear_probe import ConceptActivations, ProbeConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from games.interp_cells import Stimulus

logger = logging.getLogger("games.interp_displacement")

REPORT_FILENAME = "displacement.json"
AGREEMENT_FILENAME = "two_path_agreement.json"

# The pseudo set-group whose rows are every set's rows at once, reported beside the per-set reads.
POOLED_SET_GROUP = "pooled"

# The three ways the stimulus population is sliced. `pooled` is every pair; the other two name the
# provenance field they slice on.
POOLED_STRATIFICATION = "pooled"
CITED_STRATIFICATION = "cited_cells"
FRAMING_STRATIFICATION = "counterpart_framing"

# The provenance key behind FRAMING_STRATIFICATION, and the stratum for rows that do not carry it.
# The decision set has no counterpart to frame, so it legitimately forms its own single stratum --
# named rather than dropped, so a read over it cannot be mistaken for one of the two game halves.
FRAMING_KEY = "counterpart_framing"
FRAMING_NOT_APPLICABLE = "framing-not-applicable"

# Whose contrast axis a displacement is split along: the base anchor's, or the arm checkpoint's own.
# Both, by default: the plan asks for "that cell's own axis" and the earlier scratch read used the
# base anchor's, the two are cheap, and on this ladder they are nearly the same direction (axis drift
# from base is >= 0.997), which is itself only visible if both are reported.
AXIS_REFERENCE_BASE = "base"
AXIS_REFERENCE_OWN = "own"
AXIS_REFERENCES: tuple[str, ...] = (AXIS_REFERENCE_BASE, AXIS_REFERENCE_OWN)

# Two independent capture implementations of the same corpus at batch 1 should agree to ~1.0 on a
# direction cosine. Anything below this means one of the two paths is wrong.
DEFAULT_AGREEMENT_FLOOR = 0.99

# A split-half needs one pair on each side; a reliability over fewer is not defined.
MIN_PAIRS_FOR_SPLIT_HALF = 2

# A fair coin, for the shuffled-label null's per-pair side flip.
COIN_FLIP = 0.5

# How many worst disagreements a two-path warning quotes.
WORST_EXAMPLE_COUNT = 10


class AgreementCheckError(ValueError):
    """The two-path agreement check cannot detect the error it exists to detect."""


class PipelineNullError(ValueError):
    """The base-against-base null found movement where the arithmetic admits none."""


def _read_generator(seed: int, key: str) -> torch.Generator:
    """Seed a generator from `seed` and a read's identity, so one read reproduces on its own.

    Keyed by a checksum of the read's coordinates rather than by a running counter, so the placebo
    draws of any single read do not depend on how many reads ran before it -- re-running one
    selection alone reproduces the floor it reported in a full pass. `zlib.crc32` because Python's
    own `hash` of a string is salted per process and would not reproduce across runs at all.
    """
    return torch.Generator().manual_seed(seed + zlib.crc32(key.encode()))


def placebo_abs_cosine_band(
    reference: torch.Tensor, generator: torch.Generator, n_placebos: int
) -> tuple[float, float]:
    """Mean and max |cosine| of `reference` against matched-norm random directions.

    One band per read, reused for every cosine in it: the distribution of |cos(fixed, random)|
    depends only on the dimensionality, not on which two vectors are being compared, so drawing a
    separate band per axis component would burn placebos to re-measure the same number.
    """
    draws = [
        abs(cosine(reference, matched_norm_random_direction(reference, generator)))
        for _ in range(n_placebos)
    ]
    return sum(draws) / len(draws), max(draws)


def attenuation_ceiling(reliability_a: float | None, reliability_b: float | None) -> float | None:
    """Attenuation-style ceiling for a cosine between two noisy directions, or None if undefined.

    `sqrt(rel_a * rel_b)`, the classical correction for comparing two unreliable measurements. None
    when either reliability is missing or non-positive: a negative reliability means the direction is
    not determined at all, and a ceiling from its square root would be arithmetic on noise.
    """
    if reliability_a is None or reliability_b is None:
        return None
    if reliability_a <= 0 or reliability_b <= 0:
        return None
    return float((reliability_a * reliability_b) ** 0.5)


@dataclass(frozen=True)
class RowGroup:
    """One selection of stimulus rows: which sets, which stratum, and how it splits in half.

    `rows` indexes each set's stored activation matrix. `pair_even_positions` and
    `pair_odd_positions` index the *concatenated* row axis of this selection, so a displacement can
    be computed once over the whole selection and halved by position rather than recomputed.
    """

    set_group: str
    stimulus_sets: tuple[str, ...]
    stratification: str
    stratum: str
    rows: dict[str, torch.Tensor]
    pair_even_positions: torch.Tensor
    pair_odd_positions: torch.Tensor
    n_rows: int
    n_pairs: int
    row_parity_equals_side: bool

    @property
    def label(self) -> str:
        """How a selection names itself in a log line."""
        return f"{self.set_group}|{self.stratification}|{self.stratum}"


def _pair_ranks(pair_ids: Sequence[str]) -> dict[str, int]:
    """Rank each pair by first appearance in the stored row order -- the odd/even split's index."""
    ranks: dict[str, int] = {}
    for pair_id in pair_ids:
        if pair_id not in ranks:
            ranks[pair_id] = len(ranks)
    return ranks


def _build_row_group(  # noqa: PLR0913 - a selection is a cell, sets, a stratification and its stratum
    cell: CapturedCell,
    *,
    set_group: str,
    stimulus_sets: Sequence[str],
    stratification: str,
    stratum: str,
    stratum_by_pair: dict[str, str] | None,
) -> RowGroup:
    """Collect the rows one selection covers, in stored order, with its pair-parity halves."""
    rows: dict[str, torch.Tensor] = {}
    even: list[int] = []
    odd: list[int] = []
    sides: list[str] = []
    kept_pairs: set[str] = set()
    position = 0
    for name in stimulus_sets:
        row_index = cell.rows[name]
        ranks = _pair_ranks(row_index.pair_ids)
        kept = [
            row
            for row, pair_id in enumerate(row_index.pair_ids)
            if stratum_by_pair is None or stratum_by_pair.get(pair_id) == stratum
        ]
        rows[name] = torch.tensor(kept, dtype=torch.long)
        for row in kept:
            pair_id = row_index.pair_ids[row]
            kept_pairs.add(pair_id)
            (even if ranks[pair_id] % 2 == 0 else odd).append(position)
            sides.append(row_index.sides[row])
            position += 1
    first_sides = set(sides[0::2])
    second_sides = set(sides[1::2])
    return RowGroup(
        set_group=set_group,
        stimulus_sets=tuple(stimulus_sets),
        stratification=stratification,
        stratum=stratum,
        rows=rows,
        pair_even_positions=torch.tensor(even, dtype=torch.long),
        pair_odd_positions=torch.tensor(odd, dtype=torch.long),
        n_rows=position,
        n_pairs=len(kept_pairs),
        row_parity_equals_side=(
            len(first_sides) == 1 and len(second_sides) == 1 and first_sides != second_sides
        ),
    )


def build_row_groups(
    cell: CapturedCell,
    *,
    stimulus_sets: Sequence[str],
    stratifications: dict[str, StimulusStrata],
) -> tuple[list[RowGroup], list[dict[str, Any]]]:
    """Every selection to read, plus the ones skipped and why.

    Returns the pooled selection for each set group first, then its strata. A stratum covering
    exactly the rows its pooled parent covers is a duplicate read, and an empty stratum is not a read
    at all; both are skipped, and both are returned in the skip list because a read that silently
    vanishes is indistinguishable from one that came out uninteresting.

    Built from one cell -- the base anchor -- and applied to every other cell in the ladder, which is
    sound only because `assert_rows_align` has already refused a ladder whose cells hold different
    rows in a different order. Without that guard these row indices would silently mean a different
    stimulus in each cell.
    """
    groups: list[RowGroup] = []
    skipped: list[dict[str, Any]] = []
    set_groups: list[tuple[str, tuple[str, ...]]] = [(POOLED_SET_GROUP, tuple(stimulus_sets))]
    if len(stimulus_sets) > 1:
        set_groups.extend((name, (name,)) for name in stimulus_sets)
    for set_group, group_sets in set_groups:
        pooled = _build_row_group(
            cell,
            set_group=set_group,
            stimulus_sets=group_sets,
            stratification=POOLED_STRATIFICATION,
            stratum=POOLED_STRATUM,
            stratum_by_pair=None,
        )
        groups.append(pooled)
        for stratification, strata in sorted(stratifications.items()):
            present = sorted({strata.stratum_by_pair[pair] for pair in strata.stratum_by_pair})
            for stratum in present:
                candidate = _build_row_group(
                    cell,
                    set_group=set_group,
                    stimulus_sets=group_sets,
                    stratification=stratification,
                    stratum=stratum,
                    stratum_by_pair=strata.stratum_by_pair,
                )
                reason = None
                if candidate.n_rows == 0:
                    reason = "no rows in this set group"
                elif candidate.n_rows == pooled.n_rows:
                    reason = "covers every row of its set group, so it repeats the pooled read"
                if reason is not None:
                    skipped.append(
                        {
                            "set_group": set_group,
                            "stratification": stratification,
                            "stratum": stratum,
                            "n_rows": candidate.n_rows,
                            "reason": reason,
                        }
                    )
                    continue
                groups.append(candidate)
    logger.info(
        f"row selections built, kept={len(groups)} skipped={len(skipped)} "
        f"labels={[group.label for group in groups]}"
    )
    return groups, skipped


def load_framing_strata(path: Path, stimuli: Sequence[Stimulus]) -> StimulusStrata:
    """Map each pair to its `counterpart_framing` half, or to the not-applicable stratum.

    Deliberately thin: `games.interp_axes.load_strata` does the full join validation over the same
    file (every corpus id covered, set/side/pair_id agreeing, no pair straddling strata) and is
    always called first by this module, so this pass only reads a second field off rows already
    known to line up. It still refuses a pair whose two sides disagree, because that is specific to
    the field being read rather than to the join.
    """
    by_pair: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = cast("dict[str, Any]", json.loads(line))
        value = row.get(FRAMING_KEY)
        half = FRAMING_NOT_APPLICABLE if value is None else str(value)
        pair_id = str(row.get("pair_id"))
        known = by_pair.setdefault(pair_id, half)
        if known != half:
            raise ValueError(
                f"{path}:{line_number} puts pair {pair_id!r} in {FRAMING_KEY} half {half!r} while "
                f"another row of the same pair says {known!r}, so the pair cannot be sliced as one."
            )
    corpus_pairs = {stimulus.pair_id for stimulus in stimuli}
    missing = sorted(corpus_pairs - set(by_pair))
    if missing:
        raise ValueError(
            f"{path} has no rows for {len(missing)} corpus pairs ({missing[:WORST_EXAMPLE_COUNT]}), "
            f"so the {FRAMING_KEY} halves would be read off a different rendering."
        )
    logger.info(
        f"{FRAMING_KEY} joined, pairs={len(by_pair)} halves={sorted(set(by_pair.values()))}"
    )
    return StimulusStrata(stratum_by_pair=by_pair)


@dataclass(frozen=True)
class DisplacementVectors:
    """One arm's mean displacement from base at every layer, with the halves behind its ceiling.

    Every field is `[n_layers, hidden]`. `pair_*` split disjoint sets of matched pairs (both sides in
    each half); `row_*` split by stored row parity, which on this corpus is a side split.
    """

    full: torch.Tensor
    pair_even: torch.Tensor
    pair_odd: torch.Tensor
    row_even: torch.Tensor
    row_odd: torch.Tensor


def displacement_vectors(
    arm_rows: dict[str, torch.Tensor], base_rows: dict[str, torch.Tensor], group: RowGroup
) -> DisplacementVectors:
    """Mean row-wise displacement over one selection, and its two split halves.

    The difference is taken row against row before averaging, which is what makes this a paired
    read: stimulus-to-stimulus variation cancels exactly rather than approximately, and a base cell
    standing in for the arm gives identically zero (see `base_vs_base_null`).
    """
    delta = torch.cat(
        [
            arm_rows[name][group.rows[name]] - base_rows[name][group.rows[name]]
            for name in group.stimulus_sets
        ],
        dim=0,
    )
    row_positions = torch.arange(group.n_rows)
    return DisplacementVectors(
        full=delta.mean(dim=0),
        pair_even=delta[group.pair_even_positions].mean(dim=0),
        pair_odd=delta[group.pair_odd_positions].mean(dim=0),
        row_even=delta[row_positions[row_positions % 2 == 0]].mean(dim=0),
        row_odd=delta[row_positions[row_positions % 2 == 1]].mean(dim=0),
    )


def residual_scale(
    base_rows: dict[str, torch.Tensor], group: RowGroup
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer scale of the base residual stream over one selection: two denominators.

    A displacement norm means nothing on its own -- residual norms grow several-fold with depth --
    so it is reported relative to both the norm of the mean state (what the earlier scratch read
    used) and the mean of the per-row norms (larger, and the honest "how big is a state here").
    Reporting both is a hedge against the two being confused: the first shrinks whenever states point
    in different directions, which is a fact about the population, not about its scale.
    """
    rows = torch.cat([base_rows[name][group.rows[name]] for name in group.stimulus_sets], dim=0)
    return rows.mean(dim=0).norm(dim=-1), rows.norm(dim=-1).mean(dim=0)


@dataclass(frozen=True)
class ArmDisplacementRead:
    """One arm's displacement at one (selection, step, layer): how far it moved, and how reliably."""

    arm: str
    displacement_norm: float
    relative_to_residual_mean_vector: float
    relative_to_residual_row_norm_mean: float
    split_half_cosine_pairs: float | None
    split_half_cosine_rows: float | None
    unavailable_reason: str | None


@dataclass(frozen=True)
class ArmPairDisplacementRead:
    """Two arms' displacements compared: the orthogonality question, with its floor and ceiling.

    `cosine_cross_halves_*` compare one arm's half against the *other* arm's other half, so the two
    directions cannot share stimulus-sampling noise; a cross-arm cosine that survives there is not an
    artifact of both halves having seen the same stimuli.
    """

    arm_a: str
    arm_b: str
    cosine_real: float
    cosine_cross_halves_pairs: tuple[float, float] | None
    cosine_cross_halves_rows: tuple[float, float] | None
    same_axis_ceiling_pairs: float | None
    same_axis_ceiling_rows: float | None
    difference_norm: float


@dataclass(frozen=True)
class AxisComponentRead:
    """A displacement split into its component along one contrast axis and the residual around it.

    `projection` is signed, in raw activation units (the displacement dotted with the unit axis), so
    two arms moving oppositely along a shared axis reads as two projections of opposite sign --
    which a cross-arm cosine near zero cannot distinguish from two unrelated movements.
    """

    axis_set: str
    axis_reference: str
    target: str
    cosine_to_axis: float
    projection: float
    residual_norm: float
    axis_norm: float


@dataclass(frozen=True)
class DisplacementRead:
    """Everything this module reads at one (pooling, selection, step, layer)."""

    pooling: str
    set_group: str
    stimulus_sets: tuple[str, ...]
    stratification: str
    stratum: str
    step: int
    layer: int
    n_rows: int
    n_pairs: int
    row_parity_equals_side: bool
    residual_mean_vector_norm: float
    residual_row_norm_mean: float
    placebo_abs_cosine_mean: float
    placebo_abs_cosine_max: float
    arms: tuple[ArmDisplacementRead, ...]
    arm_pairs: tuple[ArmPairDisplacementRead, ...]
    axis_components: tuple[AxisComponentRead, ...]


def _arm_read(
    arm: str, vectors: DisplacementVectors, layer: int, scale: tuple[float, float], *, n_pairs: int
) -> ArmDisplacementRead:
    """Score one arm's displacement at one layer against both residual-norm denominators."""
    full = vectors.full[layer]
    norm = float(full.norm())
    can_split = n_pairs >= MIN_PAIRS_FOR_SPLIT_HALF
    mean_vector_norm, row_norm_mean = scale
    return ArmDisplacementRead(
        arm=arm,
        displacement_norm=norm,
        relative_to_residual_mean_vector=norm / mean_vector_norm,
        relative_to_residual_row_norm_mean=norm / row_norm_mean,
        split_half_cosine_pairs=(
            cosine(vectors.pair_even[layer], vectors.pair_odd[layer]) if can_split else None
        ),
        split_half_cosine_rows=(
            cosine(vectors.row_even[layer], vectors.row_odd[layer]) if can_split else None
        ),
        unavailable_reason=(None if can_split else f"{n_pairs} pairs cannot fill two split halves"),
    )


def _arm_pair_read(  # noqa: PLR0913 - two arms, their vectors, a layer, the reliabilities and a count
    arm_a: str,
    arm_b: str,
    vectors: dict[str, DisplacementVectors],
    layer: int,
    reliability: dict[str, ArmDisplacementRead],
    *,
    n_pairs: int,
) -> ArmPairDisplacementRead:
    """Compare two arms' displacements at one layer, with both split-half ceilings beside them."""
    first, second = vectors[arm_a], vectors[arm_b]
    can_split = n_pairs >= MIN_PAIRS_FOR_SPLIT_HALF
    return ArmPairDisplacementRead(
        arm_a=arm_a,
        arm_b=arm_b,
        cosine_real=cosine(first.full[layer], second.full[layer]),
        cosine_cross_halves_pairs=(
            (
                cosine(first.pair_even[layer], second.pair_odd[layer]),
                cosine(first.pair_odd[layer], second.pair_even[layer]),
            )
            if can_split
            else None
        ),
        cosine_cross_halves_rows=(
            (
                cosine(first.row_even[layer], second.row_odd[layer]),
                cosine(first.row_odd[layer], second.row_even[layer]),
            )
            if can_split
            else None
        ),
        same_axis_ceiling_pairs=attenuation_ceiling(
            reliability[arm_a].split_half_cosine_pairs, reliability[arm_b].split_half_cosine_pairs
        ),
        same_axis_ceiling_rows=attenuation_ceiling(
            reliability[arm_a].split_half_cosine_rows, reliability[arm_b].split_half_cosine_rows
        ),
        difference_norm=float((first.full[layer] - second.full[layer]).norm()),
    )


def _axis_component(
    target: str, vector: torch.Tensor, axis: torch.Tensor, *, axis_set: str, axis_reference: str
) -> AxisComponentRead:
    """Split one vector into its signed component along an axis and the residual norm around it."""
    direction = unit(axis)
    projection = float(vector @ direction)
    residual = vector - projection * direction
    return AxisComponentRead(
        axis_set=axis_set,
        axis_reference=axis_reference,
        target=target,
        cosine_to_axis=cosine(vector, axis),
        projection=projection,
        residual_norm=float(residual.norm()),
        axis_norm=float(axis.norm()),
    )


def _axis_components(
    vectors: dict[str, DisplacementVectors],
    layer: int,
    axes: dict[tuple[str, str], torch.Tensor],
    axis_references: Sequence[str],
) -> list[AxisComponentRead]:
    """Every axis split for one layer: each arm against each axis, and each arm-pair difference."""
    arms = sorted(vectors)
    axis_sets = sorted({name for _, name in axes})
    reads: list[AxisComponentRead] = []
    for axis_set in axis_sets:
        for arm in arms:
            for reference in axis_references:
                key = (reference if reference == AXIS_REFERENCE_BASE else arm, axis_set)
                if key not in axes:
                    continue
                reads.append(
                    _axis_component(
                        arm,
                        vectors[arm].full[layer],
                        axes[key][layer],
                        axis_set=axis_set,
                        axis_reference=reference,
                    )
                )
        if (AXIS_REFERENCE_BASE, axis_set) not in axes:
            continue
        reads.extend(
            _axis_component(
                f"{arm_a}-minus-{arm_b}",
                vectors[arm_a].full[layer] - vectors[arm_b].full[layer],
                axes[AXIS_REFERENCE_BASE, axis_set][layer],
                axis_set=axis_set,
                axis_reference=AXIS_REFERENCE_BASE,
            )
            for index, arm_a in enumerate(arms)
            for arm_b in arms[index + 1 :]
        )
    return reads


def contrast_axes(
    cell: CapturedCell,
    *,
    pooling: str,
    layouts: dict[str, PairLayout],
) -> dict[str, torch.Tensor]:
    """One cell's diff-of-means contrast axis per set, at every layer at once: `[n_layers, hidden]`.

    Fitted on all of a set's pairs regardless of which stratum a read slices, so a stratified
    displacement is projected onto a fixed axis rather than onto an axis that moved with it.
    """
    axes: dict[str, torch.Tensor] = {}
    for name, layout in layouts.items():
        matrix = cell.matrix(name, pooling)
        axes[name] = diff_of_means(matrix[layout.positive_rows], matrix[layout.negative_rows])
    return axes


def _layer_read(  # noqa: PLR0913 - one read is a selection, a step, a layer, the vectors and the knobs
    group: RowGroup,
    *,
    pooling: str,
    step: int,
    layer: int,
    vectors: dict[str, DisplacementVectors],
    scale: tuple[torch.Tensor, torch.Tensor],
    axes: dict[tuple[str, str], torch.Tensor],
    axis_references: Sequence[str],
    generator: torch.Generator,
    n_placebos: int,
) -> DisplacementRead:
    """Assemble every number for one (selection, step, layer)."""
    arms = sorted(vectors)
    layer_scale = (float(scale[0][layer]), float(scale[1][layer]))
    arm_reads = {
        arm: _arm_read(arm, vectors[arm], layer, layer_scale, n_pairs=group.n_pairs) for arm in arms
    }
    floor_mean, floor_max = placebo_abs_cosine_band(
        vectors[arms[0]].full[layer], generator, n_placebos
    )
    pairs = [
        _arm_pair_read(arm_a, arm_b, vectors, layer, arm_reads, n_pairs=group.n_pairs)
        for index, arm_a in enumerate(arms)
        for arm_b in arms[index + 1 :]
    ]
    return DisplacementRead(
        pooling=pooling,
        set_group=group.set_group,
        stimulus_sets=group.stimulus_sets,
        stratification=group.stratification,
        stratum=group.stratum,
        step=step,
        layer=layer,
        n_rows=group.n_rows,
        n_pairs=group.n_pairs,
        row_parity_equals_side=group.row_parity_equals_side,
        residual_mean_vector_norm=layer_scale[0],
        residual_row_norm_mean=layer_scale[1],
        placebo_abs_cosine_mean=floor_mean,
        placebo_abs_cosine_max=floor_max,
        arms=tuple(arm_reads[arm] for arm in arms),
        arm_pairs=tuple(pairs),
        axis_components=tuple(_axis_components(vectors, layer, axes, axis_references)),
    )


@dataclass(frozen=True)
class DisplacementConfig:
    """The knobs one displacement pass runs under."""

    poolings: tuple[str, ...]
    layers: tuple[int, ...]
    axis_references: tuple[str, ...]
    n_placebos: int
    seed: int


def displacement_reads(
    ladder: Ladder,
    *,
    groups: Sequence[RowGroup],
    layouts: dict[str, PairLayout],
    config: DisplacementConfig,
) -> list[DisplacementRead]:
    """Every displacement read over a ladder, one pooling and one step held in memory at a time.

    The loop order is memory, not taste: a cell's activations are ~80 MB per pooling here, so the
    base anchor is cached for the whole pooling and the arm cells only for the step being read.
    Placebo generators are seeded per (pooling, step, layer) rather than once for the pass, so a read
    reproduces on its own without depending on which reads ran before it.
    """
    base = ladder.cell(BASE_ARM, BASE_STEP)
    arms = analysis_arms(ladder)
    if not arms:
        raise CellFormatError(
            f"this ladder holds only {BASE_ARM!r}, so there is no displacement from base to measure."
        )
    steps = sorted({cell.step for cell in ladder.cells if cell.arm != BASE_ARM})
    reads: list[DisplacementRead] = []
    for pooling in config.poolings:
        base_rows = {name: base.matrix(name, pooling) for name in layouts}
        base_axes = contrast_axes(base, pooling=pooling, layouts=layouts)
        scales = {group.label: residual_scale(base_rows, group) for group in groups}
        for step in steps:
            cells = {arm: ladder.cell(arm, step) for arm in arms}
            arm_rows = {
                arm: {name: cell.matrix(name, pooling) for name in layouts}
                for arm, cell in cells.items()
            }
            axes: dict[tuple[str, str], torch.Tensor] = {
                (AXIS_REFERENCE_BASE, name): axis for name, axis in base_axes.items()
            }
            if AXIS_REFERENCE_OWN in config.axis_references:
                for arm, cell in cells.items():
                    for name, axis in contrast_axes(cell, pooling=pooling, layouts=layouts).items():
                        axes[arm, name] = axis
            for group in groups:
                vectors = {
                    arm: displacement_vectors(arm_rows[arm], base_rows, group) for arm in arms
                }
                reads.extend(
                    _layer_read(
                        group,
                        pooling=pooling,
                        step=step,
                        layer=layer,
                        vectors=vectors,
                        scale=scales[group.label],
                        axes=axes,
                        axis_references=config.axis_references,
                        generator=_read_generator(
                            config.seed, f"{pooling}|{group.label}|{step}|{layer}"
                        ),
                        n_placebos=config.n_placebos,
                    )
                    for layer in config.layers
                )
            logger.info(f"displacement read, {pooling=} {step=} reads={len(reads)}")
    return reads


# The one matrix metric that is a property of a single arm rather than of the arm pair.
PER_ARM_MATRIX_METRIC = "displacement_relative_norm"

MATRIX_METRICS: tuple[str, ...] = (
    "cross_arm_cosine",
    "same_axis_ceiling_pairs",
    "placebo_abs_cosine_max",
    PER_ARM_MATRIX_METRIC,
)


def _matrix_value(read: DisplacementRead, metric: str, arm: str | None) -> float | None:
    """One cell of a layer-by-step matrix, or None where the read does not carry that number."""
    if metric == "placebo_abs_cosine_max":
        return read.placebo_abs_cosine_max
    if metric == PER_ARM_MATRIX_METRIC:
        return next(
            (item.relative_to_residual_row_norm_mean for item in read.arms if item.arm == arm),
            None,
        )
    if not read.arm_pairs:
        return None
    pair = read.arm_pairs[0]
    return pair.cosine_real if metric == "cross_arm_cosine" else pair.same_axis_ceiling_pairs


def build_matrices(reads: Sequence[DisplacementRead]) -> list[dict[str, Any]]:
    """Pivot the reads into layer-by-step matrices, so a depth band is a shape rather than a memory.

    One matrix per (pooling, selection, metric), rows in step order and columns in layer order. The
    cross-arm entries use the first arm pair, which is the only pair on a two-arm ladder and is named
    in the payload either way.
    """
    matrices: list[dict[str, Any]] = []
    keys = sorted(
        {(read.pooling, read.set_group, read.stratification, read.stratum) for read in reads}
    )
    for pooling, set_group, stratification, stratum in keys:
        selected = [
            read
            for read in reads
            if (read.pooling, read.set_group, read.stratification, read.stratum)
            == (pooling, set_group, stratification, stratum)
        ]
        steps = sorted({read.step for read in selected})
        layers = sorted({read.layer for read in selected})
        by_cell = {(read.step, read.layer): read for read in selected}
        arms = sorted({item.arm for read in selected for item in read.arms})
        pair_label = (
            f"{selected[0].arm_pairs[0].arm_a}|{selected[0].arm_pairs[0].arm_b}"
            if selected[0].arm_pairs
            else None
        )
        matrices.extend(
            {
                "pooling": pooling,
                "set_group": set_group,
                "stratification": stratification,
                "stratum": stratum,
                "metric": metric,
                "arm": arm,
                "arm_pair": None if metric == PER_ARM_MATRIX_METRIC else pair_label,
                "steps": steps,
                "layers": layers,
                "values": [
                    [_matrix_value(by_cell[step, layer], metric, arm) for layer in layers]
                    for step in steps
                ],
            }
            for metric in MATRIX_METRICS
            for arm in (arms if metric == PER_ARM_MATRIX_METRIC else [None])
        )
    return matrices


@dataclass(frozen=True)
class ShuffledLabelNull:
    """A contrast axis re-fitted on randomly flipped pair labels: it must land at the floor.

    The placebo machinery's own sabotage, run on every pass rather than once by hand. `clears_placebo`
    being true means a direction fitted on labels that carry no information separated the classes
    better than every matched-norm placebo, which would make every accuracy in this arc unreadable.
    It is reported and logged loudly rather than raised: with `n_placebos` draws a true null clears
    the placebo max about one time in `n_placebos + 1` by luck, so raising would build a flaky gate.
    """

    stimulus_set: str
    pooling: str
    layer: int
    n_pairs: int
    n_flipped: int
    real_direction_accuracy: float
    real_placebo_accuracy_max: float
    null_direction_accuracy: float
    null_placebo_accuracy_mean: float
    null_placebo_accuracy_max: float
    clears_placebo: bool
    cosine_null_to_real: float
    placebo_floor_abs_cosine_mean: float
    placebo_floor_abs_cosine_max: float


def shuffled_label_null(  # noqa: PLR0913 - a null is a cell, a set, a pooling, a layer and the knobs
    cell: CapturedCell,
    *,
    stimulus_set: str,
    pooling: str,
    layer: int,
    layout: PairLayout,
    config: ProbeConfig,
    n_placebos: int,
) -> ShuffledLabelNull:
    """Fit the contrast axis on side-shuffled pairs at one layer and score it against the real one."""
    concept = concept_activations(cell, stimulus_set, pooling, layer, layout)
    real = concept.diff_of_means()
    real_separation = direction_separation(concept, config, n_placebos=n_placebos)
    flip_generator = torch.Generator().manual_seed(config.seed)
    flip = torch.rand(concept.n_pairs, generator=flip_generator) < COIN_FLIP
    shuffled = ConceptActivations(
        torch.where(flip.unsqueeze(1), concept.negatives, concept.positives),
        torch.where(flip.unsqueeze(1), concept.positives, concept.negatives),
    )
    null_separation = direction_separation(shuffled, config, n_placebos=n_placebos)
    floor_mean, floor_max = placebo_abs_cosine_band(
        real, torch.Generator().manual_seed(config.seed + 1), n_placebos
    )
    clears = null_separation.direction_accuracy > null_separation.placebo_accuracy_max
    if clears:
        logger.warning(
            f"shuffled-label null CLEARED its placebo, {stimulus_set=} {pooling=} {layer=}: "
            f"accuracy={null_separation.direction_accuracy:.3f} > "
            f"placebo_max={null_separation.placebo_accuracy_max:.3f}. Either this is the ~1-in-"
            f"{n_placebos + 1} unlucky draw or the placebo machinery is not a control."
        )
    return ShuffledLabelNull(
        stimulus_set=stimulus_set,
        pooling=pooling,
        layer=layer,
        n_pairs=concept.n_pairs,
        n_flipped=int(flip.sum()),
        real_direction_accuracy=real_separation.direction_accuracy,
        real_placebo_accuracy_max=real_separation.placebo_accuracy_max,
        null_direction_accuracy=null_separation.direction_accuracy,
        null_placebo_accuracy_mean=null_separation.placebo_accuracy_mean,
        null_placebo_accuracy_max=null_separation.placebo_accuracy_max,
        clears_placebo=clears,
        cosine_null_to_real=cosine(shuffled.diff_of_means(), real),
        placebo_floor_abs_cosine_mean=floor_mean,
        placebo_floor_abs_cosine_max=floor_max,
    )


def shuffled_label_nulls(  # noqa: PLR0913 - a cell, its layouts, poolings, a layer and the knobs
    cell: CapturedCell,
    *,
    layouts: dict[str, PairLayout],
    poolings: Sequence[str],
    layer: int,
    config: ProbeConfig,
    n_placebos: int,
) -> dict[str, Any]:
    """Every shuffled-label null available at one layer, and the ones too small to hold out folds.

    A set with fewer matched pairs than cross-validation folds cannot produce a held-out accuracy at
    all. It is listed with its reason rather than dropped: a null section that is silently one set
    short reads as a null section that came out clean.
    """
    reads: list[ShuffledLabelNull] = []
    skipped: list[dict[str, Any]] = []
    for pooling in poolings:
        for name in sorted(layouts):
            layout = layouts[name]
            if layout.n_pairs < config.n_folds:
                skipped.append(
                    {
                        "stimulus_set": name,
                        "pooling": pooling,
                        "n_pairs": layout.n_pairs,
                        "reason": f"{layout.n_pairs} pairs cannot fill {config.n_folds} held-out "
                        f"folds, so a shuffled-label accuracy has nothing to hold out",
                    }
                )
                continue
            reads.append(
                shuffled_label_null(
                    cell,
                    stimulus_set=name,
                    pooling=pooling,
                    layer=layer,
                    layout=layout,
                    config=config,
                    n_placebos=n_placebos,
                )
            )
    return {"layer": layer, "reads": [asdict(read) for read in reads], "skipped": skipped}


def base_vs_base_null(
    base: CapturedCell, *, groups: Sequence[RowGroup], poolings: Sequence[str]
) -> dict[str, Any]:
    """Run the displacement machinery with the base cell standing in for an arm; it must read zero.

    A displacement is a paired difference over the same rows, so base-against-base is exactly zero
    rather than merely small -- which is what makes this sharp. Anything nonzero means the two sides
    of the subtraction are not the rows they claim to be, and the module refuses to report the rest
    of the analysis rather than publishing a movement it invented.
    """
    worst = 0.0
    checks = 0
    for pooling in poolings:
        rows = {name: base.matrix(name, pooling) for name in base.stimulus_sets}
        for group in groups:
            vectors = displacement_vectors(rows, rows, group)
            for field in (
                vectors.full,
                vectors.pair_even,
                vectors.pair_odd,
                vectors.row_even,
                vectors.row_odd,
            ):
                worst = max(worst, float(field.abs().max()))
                checks += 1
    if worst > 0:
        raise PipelineNullError(
            f"the base cell displaced from itself by {worst:g} over {checks} checks. A paired "
            f"row-wise difference of a cell with itself is exactly zero, so the row selections on "
            f"the two sides of the subtraction are not the same rows and every displacement this "
            f"module reports is differencing mismatched stimuli."
        )
    logger.info(f"base-against-base null clean, {checks=} max_abs_displacement=0")
    return {"n_checks": checks, "max_abs_displacement": worst}


@dataclass(frozen=True)
class AgreementRead:
    """One (cell, set, pooling, layer) comparison between two independent capture paths.

    `comparable` is false for a layer the two paths are known to have measured differently -- on the
    games corpus the converted frozen ladder's top layer is post-final-RMSNorm rather than a decoder
    block output -- so it is reported and kept out of the summary rather than deleted or pooled.
    """

    cell: str
    stimulus_set: str
    pooling: str
    layer: int
    direction_cosine: float
    relative_l2: float
    comparable: bool


@dataclass(frozen=True)
class DisplacementAgreementRead:
    """The same comparison on the displacement-from-base vector rather than on a contrast axis."""

    cell: str
    pooling: str
    layer: int
    cosine: float
    comparable: bool


def _relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """`|a-b| / |a|`, the scale-free disagreement between two tensors."""
    return float((a - b).norm() / a.norm())


def _assert_ladders_comparable(canonical: Ladder, frozen: Ladder) -> None:
    """Raise unless the two ladders are the same cells measured under the same layer convention."""
    if canonical.identity.layer_convention != frozen.identity.layer_convention:
        raise AgreementCheckError(
            f"the two ladders record different layer conventions "
            f"({canonical.identity.layer_convention} vs {frozen.identity.layer_convention}); "
            f"nothing here would be comparing the same depth."
        )
    labels = [cell.label for cell in canonical.cells]
    if labels != [cell.label for cell in frozen.cells]:
        raise AgreementCheckError(
            f"the two ladders hold different cells: {labels} vs "
            f"{[cell.label for cell in frozen.cells]}."
        )


def _direction_agreement_reads(  # noqa: PLR0913 - two ladders, layouts, poolings, layers, one set
    canonical: Ladder,
    frozen: Ladder,
    *,
    layouts: dict[str, PairLayout],
    poolings: Sequence[str],
    layers: Sequence[int],
    comparable_layers: set[int],
) -> tuple[list[AgreementRead], list[float]]:
    """Per-layer direction cosines between the two paths, and the off-by-one arm's cosines."""
    reads: list[AgreementRead] = []
    shifted: list[float] = []
    for cell_a in canonical.cells:
        cell_b = frozen.cell(cell_a.arm, cell_a.step)
        for name, layout in sorted(layouts.items()):
            for pooling in poolings:
                matrix_a = cell_a.matrix(name, pooling)
                matrix_b = cell_b.matrix(name, pooling)
                axis_a = diff_of_means(
                    matrix_a[layout.positive_rows], matrix_a[layout.negative_rows]
                )
                axis_b = diff_of_means(
                    matrix_b[layout.positive_rows], matrix_b[layout.negative_rows]
                )
                reads.extend(
                    AgreementRead(
                        cell=cell_a.label,
                        stimulus_set=name,
                        pooling=pooling,
                        layer=layer,
                        direction_cosine=cosine(axis_a[layer], axis_b[layer]),
                        relative_l2=_relative_l2(matrix_a[:, layer, :], matrix_b[:, layer, :]),
                        comparable=layer in comparable_layers,
                    )
                    for layer in layers
                )
                shifted.extend(
                    cosine(axis_a[layer], axis_b[layer + 1])
                    for layer in layers
                    if layer in comparable_layers and layer + 1 in comparable_layers
                )
        logger.info(f"two-path agreement, cell={cell_a.label} reads={len(reads)}")
    return reads, shifted


def _displacement_agreement_reads(  # noqa: PLR0913 - two ladders, sets, poolings, layers, one set
    canonical: Ladder,
    frozen: Ladder,
    *,
    stimulus_sets: Sequence[str],
    poolings: Sequence[str],
    layers: Sequence[int],
    comparable_layers: set[int],
) -> list[DisplacementAgreementRead]:
    """Per-layer cosines between the two paths' displacement-from-base vectors."""
    base_a = canonical.cell(BASE_ARM, BASE_STEP)
    base_b = frozen.cell(BASE_ARM, BASE_STEP)
    reads: list[DisplacementAgreementRead] = []
    for cell_a in canonical.cells:
        if cell_a.arm == BASE_ARM:
            continue
        cell_b = frozen.cell(cell_a.arm, cell_a.step)
        for pooling in poolings:
            delta_a = torch.cat(
                [
                    cell_a.matrix(name, pooling) - base_a.matrix(name, pooling)
                    for name in stimulus_sets
                ]
            ).mean(dim=0)
            delta_b = torch.cat(
                [
                    cell_b.matrix(name, pooling) - base_b.matrix(name, pooling)
                    for name in stimulus_sets
                ]
            ).mean(dim=0)
            reads.extend(
                DisplacementAgreementRead(
                    cell=cell_a.label,
                    pooling=pooling,
                    layer=layer,
                    cosine=cosine(delta_a[layer], delta_b[layer]),
                    comparable=layer in comparable_layers,
                )
                for layer in layers
            )
    return reads


def _sabotage_summary(
    shifted: Sequence[float], agreement_floor: float, *, n_below_floor: int, n_comparable: int
) -> dict[str, Any]:
    """Summarise the off-by-one arm, raising unless it collapsed the way it has to."""
    if not shifted:
        raise AgreementCheckError(
            "the off-by-one arm compared nothing (fewer than two comparable layers), so a layer "
            "shift could not have been detected."
        )
    worst_case = max(shifted)
    if worst_case >= agreement_floor:
        raise AgreementCheckError(
            f"the deliberate off-by-one comparison agreed at cosine {worst_case:.4f}, at or above "
            f"the {agreement_floor} floor, while {n_below_floor} of {n_comparable} same-layer "
            f"comparisons sit below it. Either the two ladders are ALREADY ONE LAYER APART -- the "
            f"exact bug this arm exists to expose, and then the layer axes need realigning -- or "
            f"they carry too little depth structure for a shift to change anything, in which case "
            f"the same-layer agreement is not evidence that the two paths align."
        )
    return {
        "n_reads": len(shifted),
        "cosine_max": worst_case,
        "cosine_mean": sum(shifted) / len(shifted),
        "is_red": True,
        "expectation": (
            "must sit below the agreement floor; this module raises if it does not, because a "
            "comparison that cannot see a one-layer shift cannot confirm alignment"
        ),
    }


def _agreement_summary(  # noqa: PLR0913 - the two read sets, the sabotage and three knobs
    reads: Sequence[AgreementRead],
    displacements: Sequence[DisplacementAgreementRead],
    *,
    shifted: Sequence[float],
    agreement_floor: float,
    top_layer: int,
    top_layer_post_norm: bool,
) -> dict[str, Any]:
    """Summarise the comparison over comparable layers, with the post-norm layer reported apart."""
    comparable = [read for read in reads if read.comparable]
    if not comparable:
        raise AgreementCheckError("no comparable layers, so the two paths were never compared.")
    below = [read for read in comparable if read.direction_cosine < agreement_floor]
    sabotage = _sabotage_summary(
        shifted, agreement_floor, n_below_floor=len(below), n_comparable=len(comparable)
    )
    if below:
        worst = sorted(below, key=lambda read: read.direction_cosine)[:WORST_EXAMPLE_COUNT]
        quoted = ", ".join(
            f"{read.cell} {read.stimulus_set} {read.pooling} L{read.layer}="
            f"{read.direction_cosine:.4f}"
            for read in worst
        )
        logger.warning(
            f"{len(below)} of {len(comparable)} comparable direction cosines sit below "
            f"{agreement_floor}; worst: {quoted}"
        )
    post_norm = [read for read in reads if not read.comparable]
    cosines = [read.direction_cosine for read in comparable]
    displacement_cosines = [read.cosine for read in displacements if read.comparable]
    return {
        "agreement_floor": agreement_floor,
        "top_layer_treated_as_post_norm": top_layer_post_norm,
        "top_layer": top_layer,
        "n_comparable_reads": len(comparable),
        "direction_cosine_min": min(cosines),
        "direction_cosine_mean": sum(cosines) / len(cosines),
        "n_below_floor": len(below),
        "relative_l2_max": max(read.relative_l2 for read in comparable),
        "displacement_cosine_min": min(displacement_cosines) if displacement_cosines else None,
        "displacement_cosine_mean": (
            sum(displacement_cosines) / len(displacement_cosines) if displacement_cosines else None
        ),
        "post_norm_layer_reads": len(post_norm),
        "post_norm_direction_cosine_min": (
            min(read.direction_cosine for read in post_norm) if post_norm else None
        ),
        "post_norm_direction_cosine_mean": (
            sum(read.direction_cosine for read in post_norm) / len(post_norm) if post_norm else None
        ),
        "off_by_one_sabotage": sabotage,
    }


def two_path_agreement(  # noqa: PLR0913 - the two ladders, the pair layouts and three knobs
    canonical: Ladder,
    frozen: Ladder,
    *,
    layouts: dict[str, PairLayout],
    poolings: Sequence[str],
    layers: Sequence[int],
    agreement_floor: float,
    top_layer_post_norm: bool,
) -> dict[str, Any]:
    """Compare two independent captures of the same corpus, and prove the comparison can go red.

    `frozen` is a ladder already in `games.interp_cells` format, so a converted frozen capture whose
    layer axis was shifted down by one at conversion time (frozen index *i+1* becomes cell index
    *i*). This module does not verify that shift directly; it verifies it by consequence, by
    re-running the same comparison one layer off. That off-by-one is the exact bug the shift exists
    to avoid, so if the shifted comparison also agrees, the check has no teeth and this raises.
    """
    _assert_ladders_comparable(canonical, frozen)
    top_layer = canonical.identity.n_layers - 1
    comparable_layers = {
        layer for layer in layers if not (top_layer_post_norm and layer == top_layer)
    }
    reads, shifted = _direction_agreement_reads(
        canonical,
        frozen,
        layouts=layouts,
        poolings=poolings,
        layers=layers,
        comparable_layers=comparable_layers,
    )
    displacements = _displacement_agreement_reads(
        canonical,
        frozen,
        stimulus_sets=sorted(layouts),
        poolings=poolings,
        layers=layers,
        comparable_layers=comparable_layers,
    )
    return {
        "summary": _agreement_summary(
            reads,
            displacements,
            shifted=shifted,
            agreement_floor=agreement_floor,
            top_layer=top_layer,
            top_layer_post_norm=top_layer_post_norm,
        ),
        "reads": [asdict(read) for read in reads],
        "displacement_reads": [asdict(read) for read in displacements],
    }


def render_layer_table(reads: Sequence[DisplacementRead], *, pooling: str, step: int) -> str:
    """Render the pooled cross-arm read at one step: one row per layer, floor and ceiling beside it."""
    rows = [
        read
        for read in reads
        if read.pooling == pooling
        and read.step == step
        and read.set_group == POOLED_SET_GROUP
        and read.stratification == POOLED_STRATIFICATION
        and read.arm_pairs
    ]
    if not rows:
        return f"no pooled cross-arm reads at {pooling=} {step=}"
    ordered = sorted(rows, key=lambda read: read.layer)
    pair = ordered[0].arm_pairs[0]
    arms = [item.arm for item in ordered[0].arms]
    header = (
        f"displacement, pooled over sets  {pooling=} {step=}  "
        f"cross-arm cosine is {pair.arm_a} vs {pair.arm_b}\n"
        f"{'layer':>5} {'cos':>8} {'ceiling':>8} {'floor_max':>10}"
        + "".join(f" {f'rel|d| {arm}':>26}" for arm in arms)
    )
    lines: list[str] = []
    for read in ordered:
        ceiling = read.arm_pairs[0].same_axis_ceiling_pairs
        relatives = "".join(
            f" {item.relative_to_residual_row_norm_mean:>26.5f}" for item in read.arms
        )
        lines.append(
            f"{read.layer:>5} {read.arm_pairs[0].cosine_real:>8.3f} "
            f"{'n/a' if ceiling is None else format(ceiling, '.3f'):>8} "
            f"{read.placebo_abs_cosine_max:>10.3f}{relatives}"
        )
    return "\n".join([header, *lines])


def build_payload(  # noqa: PLR0913 - the payload is the reads, its context and the three nulls
    reads: Sequence[DisplacementRead],
    *,
    context: dict[str, Any],
    nulls: dict[str, Any],
    base_null: dict[str, Any],
    skipped: Sequence[dict[str, Any]],
    agreement: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the JSON report: every read, the pivoted matrices, and the nulls."""
    payload: dict[str, Any] = {
        "context": context,
        "skipped_selections": list(skipped),
        "reads": [asdict(read) for read in reads],
        "matrices": build_matrices(reads),
        "shuffled_label_nulls": nulls,
        "base_vs_base_null": base_null,
    }
    if agreement is not None:
        payload["two_path_agreement"] = agreement["summary"]
    return payload


def build_parser() -> argparse.ArgumentParser:
    """CLI for the displacement-geometry read and the two-path agreement check."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--capture-root", type=Path, required=True, help="Root of cached cells.")
    parser.add_argument(
        "--stimuli",
        type=Path,
        required=True,
        help="The stimulus corpus the cells were captured on; its digest is checked against them.",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        required=True,
        help="stimuli_provenance.jsonl beside the corpus; it carries the strata this module slices "
        "on (cited_cells, counterpart_framing).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--positive-side",
        required=True,
        help="Which stimulus side counts as the positive class of a contrast axis, e.g. A here.",
    )
    parser.add_argument(
        "--arms", default=None, help="Comma-separated arms; default every one present."
    )
    parser.add_argument(
        "--steps", default=None, help="Comma-separated steps; default every one present."
    )
    parser.add_argument("--sets", default=None, help="Comma-separated stimulus sets; default all.")
    parser.add_argument("--poolings", default=None, help="Comma-separated poolings; default all.")
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layers; default every layer in the capture. A subset is a smoke.",
    )
    parser.add_argument(
        "--axis-references",
        default=",".join(AXIS_REFERENCES),
        help="Which contrast axes to split displacements along: base (the anchor's), own (the arm "
        "checkpoint's), or both.",
    )
    parser.add_argument("--n-placebos", type=int, default=DEFAULT_N_PLACEBOS)
    parser.add_argument("--n-folds", type=int, default=ProbeConfig.n_folds)
    parser.add_argument("--seed", type=int, default=ProbeConfig.seed)
    parser.add_argument(
        "--null-layer",
        type=int,
        default=None,
        help="Layer for the shuffled-label null; default mid-depth, where the axes are strongest.",
    )
    parser.add_argument(
        "--frozen-cells",
        type=Path,
        default=None,
        help="A second ladder of the same corpus, in games.interp_cells format, captured by an "
        "independent implementation. Adds the two-path agreement read plus its off-by-one arm.",
    )
    parser.add_argument(
        "--frozen-top-layer-post-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat the frozen ladder's top layer as post-final-RMSNorm rather than a decoder-block "
        "output, so it is reported separately instead of counted as disagreement. True for the "
        "games frozen capture, whose driver read output_hidden_states.",
    )
    parser.add_argument("--agreement-floor", type=float, default=DEFAULT_AGREEMENT_FLOOR)
    return parser


def _split(raw: str | None) -> list[str] | None:
    """Parse a comma-separated CLI list, or None for 'everything present'."""
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _resolve_axis_references(raw: str) -> tuple[str, ...]:
    """Parse and check the axis-reference list, naming a typo rather than silently reading nothing."""
    wanted = tuple(_split(raw) or ())
    unknown = sorted(set(wanted) - set(AXIS_REFERENCES))
    if unknown or not wanted:
        raise ValueError(
            f"--axis-references {raw!r} names {unknown or 'nothing'}; expected a comma-separated "
            f"subset of {list(AXIS_REFERENCES)}."
        )
    return wanted


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load the ladder behind its identity guards, read every displacement, and write the report."""
    stimuli = load_stimuli(args.stimuli)
    digest = stimuli_digest(stimuli)
    arms = _split(args.arms)
    steps = None if args.steps is None else [int(part) for part in _split(args.steps) or []]
    ladder = load_ladder(args.capture_root, arms=arms, steps=steps, stimuli_sha256=digest)
    wanted_sets = _split(args.sets) or list(ladder.stimulus_sets)
    poolings = tuple(_split(args.poolings) or ladder.poolings)
    layers = tuple(
        [int(part) for part in _split(args.layers) or []]
        if args.layers is not None
        else range(ladder.identity.n_layers)
    )
    config = DisplacementConfig(
        poolings=poolings,
        layers=layers,
        axis_references=_resolve_axis_references(args.axis_references),
        n_placebos=args.n_placebos,
        seed=args.seed,
    )
    probe_config = ProbeConfig(n_folds=args.n_folds, seed=args.seed)
    base = ladder.cell(BASE_ARM, BASE_STEP)
    layouts = {
        name: pair_layout(base, name, positive_side=args.positive_side) for name in wanted_sets
    }
    stratifications = {
        CITED_STRATIFICATION: load_strata(args.provenance, stimuli),
        FRAMING_STRATIFICATION: load_framing_strata(args.provenance, stimuli),
    }
    groups, skipped = build_row_groups(
        base, stimulus_sets=wanted_sets, stratifications=stratifications
    )

    base_null = base_vs_base_null(base, groups=groups, poolings=poolings)
    reads = displacement_reads(ladder, groups=groups, layouts=layouts, config=config)
    null_layer = (
        args.null_layer
        if args.null_layer is not None
        else layers[max(len(layers) // 2 - 1, 0)]  # mid-depth of the layers actually asked for
    )
    nulls = shuffled_label_nulls(
        base,
        layouts=layouts,
        poolings=poolings,
        layer=null_layer,
        config=probe_config,
        n_placebos=args.n_placebos,
    )

    agreement: dict[str, Any] | None = None
    if args.frozen_cells is not None:
        frozen = load_ladder(args.frozen_cells, arms=arms, steps=steps, stimuli_sha256=digest)
        agreement = two_path_agreement(
            ladder,
            frozen,
            layouts=layouts,
            poolings=poolings,
            layers=layers,
            agreement_floor=args.agreement_floor,
            top_layer_post_norm=args.frozen_top_layer_post_norm,
        )

    context = {
        "capture_root": str(args.capture_root),
        "frozen_capture_root": None if args.frozen_cells is None else str(args.frozen_cells),
        "stimuli_file": str(args.stimuli),
        "provenance_file": str(args.provenance),
        "stimuli_sha256": digest,
        "identity": ladder.identity.to_payload(),
        "cells": [cell.label for cell in ladder.cells],
        "arms": list(analysis_arms(ladder)),
        "positive_side": args.positive_side,
        "poolings": list(poolings),
        "layers": list(layers),
        "axis_references": list(config.axis_references),
        "n_placebos": args.n_placebos,
        "null_layer": null_layer,
        "probe_config": asdict(probe_config),
    }
    payload = build_payload(
        reads,
        context=context,
        nulls=nulls,
        base_null=base_null,
        skipped=skipped,
        agreement=agreement,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / REPORT_FILENAME).write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n"
    )
    if agreement is not None:
        (args.out_dir / AGREEMENT_FILENAME).write_text(
            json.dumps(agreement, indent=1, sort_keys=True) + "\n"
        )
    logger.info(
        f"displacement written, report={args.out_dir / REPORT_FILENAME} reads={len(reads)} "
        f"matrices={len(payload['matrices'])} nulls={len(nulls['reads'])} "
        f"nulls_skipped={len(nulls['skipped'])}"
    )
    print_tables(reads, poolings=poolings, agreement=agreement)
    return payload


def print_tables(
    reads: Sequence[DisplacementRead],
    *,
    poolings: Sequence[str],
    agreement: dict[str, Any] | None,
) -> None:
    """Print the pooled layer table at the last step, and the agreement summary if it ran."""
    if not reads:
        return
    last_step = max(read.step for read in reads)
    for pooling in poolings:
        print(render_layer_table(reads, pooling=pooling, step=last_step))  # noqa: T201 - a CLI whose output is the table
        print()  # noqa: T201
    if agreement is not None:
        print(json.dumps(agreement["summary"], indent=2, sort_keys=True))  # noqa: T201


def main(argv: Sequence[str] | None = None) -> int:
    """Run the displacement-geometry read over a capture root."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
