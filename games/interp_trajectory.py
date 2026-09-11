"""Read a direction's trajectory across a checkpoint ladder: what RL moved, and against what floor.

Everything in `reward_hacking.interp` compares *concepts within one model*. This compares *one
concept across checkpoints*, which is the whole question here: the flagship pair trained on
byte-identical prompts differing only in the grading rule, and cooperation moved in opposite
directions. Did the internals move with it, and did they move oppositely?

Four reads, over cached activations only -- no GPU, no model, so every one of them is a re-analysis
of a capture that already happened:

**Per-checkpoint axis quality.** `reward_hacking.interp.eval_awareness_probe.validate_layer`, reused
concept-agnostically at every (arm, step, set, pooling, layer): a trained probe against a
shuffled-label null in standardized space, the diff-of-means axis against `n_placebos` matched-norm
random directions in raw space, and the odd/even split-half cosine as that axis's own noise ceiling.
An axis that does not clear its placebo at a checkpoint has no trajectory to read.

**Drift.** Cosine of each checkpoint's axis to the anchor checkpoint's axis, reported between the two
numbers that make it readable: the *floor* is the anchor axis against a matched-norm placebo (what
"unrelated" looks like in this dimensionality, ~1/sqrt(d), not zero), and the *ceiling* is the
anchor's split-half cosine (what two halves of the same checkpoint's own data give, which is well
below 1.0). A drift cosine of 0.85 means nothing without both.

**Held-out projection.** The axis is fitted on the even-indexed pairs of the anchor checkpoint and
then every checkpoint's *odd* pairs are projected onto it -- so the trajectory is a read on stimuli
the direction never saw, in raw activation units, with the matched-norm placebo projected the same
way as the floor. This is the persona-vector-style shift: hold the axis fixed, watch the population
slide along it as training proceeds.

**Cross-arm.** With one shared anchor axis and one shared held-out set, the two arms' projection gaps
are directly subtractable, which is the geometric analogue of the behavioural
difference-in-differences. `trajectory_correlation` then pairs any geometry series against a supplied
behavioural one (Pearson and Spearman over the 8 checkpoints), so "does the geometry track the
behaviour" is a number rather than an impression.

Two things this module deliberately does not do. It does not pick a winning layer for you: peak
selection is by AUC-above-placebo at the anchor, recorded per group, and every layer's read is kept.
And it does not intervene -- a decodable direction is not a used one, so the causal tier (steering
and ablation against a matched-norm placebo, activation patching between matched twins) reads the
directions this writes out rather than being folded in here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from games.interp_cells import (
    BASE_ARM,
    BASE_STEP,
    CapturedCell,
    Ladder,
    PairLayout,
    concept_activations,
    load_ladder,
    load_stimuli,
    pair_layout,
    stimuli_digest,
)
from reward_hacking.interp.directions import cosine, matched_norm_random_direction, unit
from reward_hacking.interp.eval_awareness_probe import (
    DEFAULT_N_PLACEBOS,
    direction_split_half_cosine,
    validate_layer,
)
from reward_hacking.interp.linear_probe import ProbeConfig
from reward_hacking.interp.run_steer_validation import average_ranks, pearson_correlation

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("games.interp_trajectory")

# Pair parity that fits the anchor axis; the other half is projected at every checkpoint.
FIT_PARITY = 0
HELDOUT_PARITY = 1

DIRECTIONS_DIRNAME = "directions"
REPORT_FILENAME = "trajectory.json"

# Below this many checkpoints a rank correlation is not a statistic, so it is reported as unavailable.
MIN_CORRELATION_POINTS = 3


@dataclass(frozen=True)
class GroupKey:
    """The cell of the analysis grid a read belongs to."""

    arm: str
    stimulus_set: str
    pooling: str

    @property
    def label(self) -> str:
        """How a group names itself in a table row or a log line."""
        return f"{self.arm}|{self.stimulus_set}|{self.pooling}"


@dataclass(frozen=True)
class AxisRead:
    """One checkpoint's axis quality at one layer: is there an axis here to have a trajectory."""

    arm: str
    step: int
    stimulus_set: str
    pooling: str
    layer: int
    n_pairs: int
    probe_accuracy: float
    probe_null_accuracy_max: float
    clears_null: bool
    direction_accuracy: float
    placebo_accuracy_mean: float
    placebo_accuracy_max: float
    accuracy_empirical_p: float
    beats_placebo: bool
    accuracy_above_placebo: float
    split_half_cosine: float
    direction_norm: float


@dataclass(frozen=True)
class DriftRead:
    """One checkpoint's axis measured against the anchor checkpoint's, with its floor and ceiling."""

    arm: str
    step: int
    stimulus_set: str
    pooling: str
    layer: int
    cosine_to_anchor: float
    cosine_anchor_placebo: float
    anchor_split_half_cosine: float
    norm_ratio: float
    heldout_positive_projection: float
    heldout_negative_projection: float
    heldout_projection_gap: float
    heldout_projection_gap_placebo: float
    anchor_step: int


@dataclass(frozen=True)
class CrossArmRead:
    """Two arms at one checkpoint, on one shared anchor axis and one shared held-out set."""

    step: int
    stimulus_set: str
    pooling: str
    layer: int
    arm_a: str
    arm_b: str
    cosine_between_arms: float
    projection_gap_a: float
    projection_gap_b: float
    projection_gap_difference: float
    drift_cosine_a: float
    drift_cosine_b: float


@dataclass(frozen=True)
class SeriesCorrelation:
    """A geometry series against a behavioural one, or the reason there is no correlation to report."""

    n_points: int
    pearson: float | None
    spearman: float | None
    unavailable_reason: str | None


def trajectory_correlation(
    geometry: Sequence[float], behavior: Sequence[float]
) -> SeriesCorrelation:
    """Pair a geometry series against a behavioural one over the checkpoints both cover.

    Reported as Pearson and Spearman together because they fail differently on eight points: Pearson
    is moved by one extreme checkpoint, Spearman cannot see the size of a move at all. A constant
    series has no correlation and says so rather than returning zero, which would read as "measured,
    and unrelated".
    """
    if len(geometry) != len(behavior):
        raise ValueError(f"series lengths differ: {len(geometry)} geometry vs {len(behavior)}")
    if len(geometry) < MIN_CORRELATION_POINTS:
        return SeriesCorrelation(
            n_points=len(geometry),
            pearson=None,
            spearman=None,
            unavailable_reason=f"{len(geometry)} points is fewer than {MIN_CORRELATION_POINTS}",
        )
    pearson = pearson_correlation(geometry, behavior)
    spearman = pearson_correlation(average_ranks(geometry), average_ranks(behavior))
    reason = None if pearson is not None and spearman is not None else "a series is constant"
    return SeriesCorrelation(
        n_points=len(geometry), pearson=pearson, spearman=spearman, unavailable_reason=reason
    )


def arm_series(ladder: Ladder, arm: str) -> list[CapturedCell]:
    """One arm's cells in checkpoint order, led by the shared base cell when the ladder has one.

    The base capture is stored once under its own arm name because it is the same model state for
    every arm, so each arm's trajectory starts from the same anchor. A ladder without it is analysed
    against the arm's own earliest checkpoint, which is a weaker anchor and is recorded as such.
    """
    cells = list(ladder.arm_cells(arm))
    base = next(
        (cell for cell in ladder.cells if cell.arm == BASE_ARM and cell.step == BASE_STEP), None
    )
    if base is not None and arm != BASE_ARM:
        return [base, *cells]
    return cells


def analysis_arms(ladder: Ladder) -> tuple[str, ...]:
    """Return the arms to read trajectories for: everything except the shared base anchor."""
    return tuple(arm for arm in ladder.arms if arm != BASE_ARM)


def read_axis(  # noqa: PLR0913 - a read is a cell, a group, a layer, a pair layout and the probe knobs
    cell: CapturedCell,
    key: GroupKey,
    layer: int,
    layout: PairLayout,
    config: ProbeConfig,
    *,
    n_placebos: int,
) -> tuple[AxisRead, torch.Tensor]:
    """Validate one checkpoint's axis at one layer and return the read plus its full-pair direction."""
    concept = concept_activations(cell, key.stimulus_set, key.pooling, layer, layout)
    validated, direction = validate_layer(layer, concept, config, n_placebos=n_placebos)
    return (
        AxisRead(
            arm=key.arm,
            step=cell.step,
            stimulus_set=key.stimulus_set,
            pooling=key.pooling,
            layer=layer,
            n_pairs=concept.n_pairs,
            probe_accuracy=validated.probe_accuracy,
            probe_null_accuracy_max=validated.probe_null_accuracy_max,
            clears_null=validated.clears_null,
            direction_accuracy=validated.direction_accuracy,
            placebo_accuracy_mean=validated.placebo_accuracy_mean,
            placebo_accuracy_max=validated.placebo_accuracy_max,
            accuracy_empirical_p=validated.accuracy_empirical_p,
            beats_placebo=validated.beats_placebo,
            accuracy_above_placebo=validated.direction_accuracy - validated.placebo_accuracy_max,
            split_half_cosine=validated.direction_split_half_cosine,
            direction_norm=validated.direction_norm,
        ),
        direction,
    )


def projection_gap(  # noqa: PLR0913 - a projection is a cell, a group, a layer, a layout and an axis
    cell: CapturedCell,
    key: GroupKey,
    layer: int,
    layout: PairLayout,
    direction: torch.Tensor,
    *,
    pairs: torch.Tensor,
) -> tuple[float, float]:
    """Mean projection of each side of the held-out pairs onto a fixed axis, in raw activation units.

    `unit` normalises inside, so the two numbers are components along the axis and are comparable
    across checkpoints and across a real-versus-placebo contrast.
    """
    concept = concept_activations(cell, key.stimulus_set, key.pooling, layer, layout, pairs=pairs)
    axis = unit(direction)
    return (
        float((concept.positives @ axis).mean()),
        float((concept.negatives @ axis).mean()),
    )


@dataclass(frozen=True)
class GroupTrajectory:
    """Everything read for one (arm, set, pooling) group, keyed by checkpoint and layer."""

    key: GroupKey
    anchor_step: int
    layers: tuple[int, ...]
    axis_reads: tuple[AxisRead, ...]
    drift_reads: tuple[DriftRead, ...]
    peak_layer: int
    anchor_directions: dict[int, torch.Tensor]
    step_directions: dict[int, dict[int, torch.Tensor]]
    heldout_pairs: torch.Tensor


def read_group(  # noqa: PLR0913 - a group is a ladder slice, a layer set and the probe knobs
    ladder: Ladder,
    key: GroupKey,
    *,
    layers: Sequence[int],
    positive_side: str,
    config: ProbeConfig,
    n_placebos: int,
) -> GroupTrajectory:
    """Read one (arm, set, pooling) group across its whole checkpoint series."""
    cells = arm_series(ladder, key.arm)
    if not cells:
        raise ValueError(f"no cells for arm {key.arm!r}")
    anchor = cells[0]
    layout = pair_layout(anchor, key.stimulus_set, positive_side=positive_side)
    fit_pairs = layout.half(FIT_PARITY)
    heldout_pairs = layout.half(HELDOUT_PARITY)
    if heldout_pairs.numel() == 0 or fit_pairs.numel() == 0:
        raise ValueError(
            f"set {key.stimulus_set!r} has {layout.n_pairs} pairs, too few to split into a fit half "
            f"and a held-out half; a projection read needs at least two pairs."
        )
    generator = torch.Generator().manual_seed(config.seed)

    axis_reads: list[AxisRead] = []
    step_directions: dict[int, dict[int, torch.Tensor]] = {}
    anchor_directions: dict[int, torch.Tensor] = {}
    anchor_fit_directions: dict[int, torch.Tensor] = {}
    anchor_placebos: dict[int, torch.Tensor] = {}
    anchor_split_half: dict[int, float] = {}

    for layer in layers:
        anchor_fit = concept_activations(
            anchor, key.stimulus_set, key.pooling, layer, layout, pairs=fit_pairs
        )
        anchor_fit_directions[layer] = anchor_fit.diff_of_means()
        anchor_placebos[layer] = matched_norm_random_direction(
            anchor_fit_directions[layer], generator
        )
        anchor_split_half[layer] = direction_split_half_cosine(
            concept_activations(anchor, key.stimulus_set, key.pooling, layer, layout)
        )

    for cell in cells:
        step_directions[cell.step] = {}
        for layer in layers:
            read, direction = read_axis(cell, key, layer, layout, config, n_placebos=n_placebos)
            axis_reads.append(read)
            step_directions[cell.step][layer] = direction
            if cell is anchor:
                anchor_directions[layer] = direction
        logger.info(
            f"axis reads done, group={key.label} step={cell.step} layers={len(layers)} "
            f"beats_placebo={sum(1 for r in axis_reads if r.step == cell.step and r.beats_placebo)}"
        )

    anchor_layer_reads = [read for read in axis_reads if read.step == anchor.step]
    peak_layer = max(anchor_layer_reads, key=lambda read: read.accuracy_above_placebo).layer

    drift_reads: list[DriftRead] = []
    for cell in cells:
        for layer in layers:
            anchor_direction = anchor_directions[layer]
            direction = step_directions[cell.step][layer]
            positive, negative = projection_gap(
                cell, key, layer, layout, anchor_fit_directions[layer], pairs=heldout_pairs
            )
            placebo_positive, placebo_negative = projection_gap(
                cell, key, layer, layout, anchor_placebos[layer], pairs=heldout_pairs
            )
            drift_reads.append(
                DriftRead(
                    arm=key.arm,
                    step=cell.step,
                    stimulus_set=key.stimulus_set,
                    pooling=key.pooling,
                    layer=layer,
                    cosine_to_anchor=cosine(direction, anchor_direction),
                    cosine_anchor_placebo=cosine(anchor_direction, anchor_placebos[layer]),
                    anchor_split_half_cosine=anchor_split_half[layer],
                    norm_ratio=float(direction.norm() / anchor_direction.norm()),
                    heldout_positive_projection=positive,
                    heldout_negative_projection=negative,
                    heldout_projection_gap=positive - negative,
                    heldout_projection_gap_placebo=placebo_positive - placebo_negative,
                    anchor_step=anchor.step,
                )
            )
    return GroupTrajectory(
        key=key,
        anchor_step=anchor.step,
        layers=tuple(layers),
        axis_reads=tuple(axis_reads),
        drift_reads=tuple(drift_reads),
        peak_layer=peak_layer,
        anchor_directions=anchor_directions,
        step_directions=step_directions,
        heldout_pairs=heldout_pairs,
    )


def cross_arm_reads(
    groups: dict[GroupKey, GroupTrajectory], arm_a: str, arm_b: str
) -> list[CrossArmRead]:
    """Subtract two arms' reads wherever both cover the same (set, pooling, step, layer).

    Both arms are anchored on the shared base capture and projected on the same held-out pairs, so
    the gap difference is the geometric analogue of the behavioural difference-in-differences. At the
    anchor step the two arms *are* the same cell, so `cosine_between_arms` reads exactly 1.0 there --
    a construction check on the pairing rather than a measurement.
    """
    reads: list[CrossArmRead] = []
    for key_a, group_a in sorted(groups.items(), key=lambda item: item[0].label):
        if key_a.arm != arm_a:
            continue
        key_b = GroupKey(arm=arm_b, stimulus_set=key_a.stimulus_set, pooling=key_a.pooling)
        group_b = groups.get(key_b)
        if group_b is None:
            continue
        gaps_a = {(read.step, read.layer): read for read in group_a.drift_reads}
        gaps_b = {(read.step, read.layer): read for read in group_b.drift_reads}
        for step, layer in sorted(set(gaps_a) & set(gaps_b)):
            reads.append(
                CrossArmRead(
                    step=step,
                    stimulus_set=key_a.stimulus_set,
                    pooling=key_a.pooling,
                    layer=layer,
                    arm_a=arm_a,
                    arm_b=arm_b,
                    cosine_between_arms=cosine(
                        group_a.step_directions[step][layer], group_b.step_directions[step][layer]
                    ),
                    projection_gap_a=gaps_a[step, layer].heldout_projection_gap,
                    projection_gap_b=gaps_b[step, layer].heldout_projection_gap,
                    projection_gap_difference=gaps_b[step, layer].heldout_projection_gap
                    - gaps_a[step, layer].heldout_projection_gap,
                    drift_cosine_a=gaps_a[step, layer].cosine_to_anchor,
                    drift_cosine_b=gaps_b[step, layer].cosine_to_anchor,
                )
            )
    return reads


def behavior_correlations(
    groups: dict[GroupKey, GroupTrajectory], behavior: dict[str, dict[int, float]]
) -> dict[str, dict[str, Any]]:
    """Correlate each group's peak-layer geometry series against that arm's behavioural series.

    Only the checkpoints present in both are used, and the count is reported beside every
    correlation: a series silently narrowed to three points is the failure mode here.
    """
    correlations: dict[str, dict[str, Any]] = {}
    for key, group in sorted(groups.items(), key=lambda item: item[0].label):
        arm_behavior = behavior.get(key.arm)
        if not arm_behavior:
            continue
        peak = [read for read in group.drift_reads if read.layer == group.peak_layer]
        steps = [read.step for read in peak if read.step in arm_behavior]
        if not steps:
            continue
        behavior_series = [arm_behavior[step] for step in steps]
        by_step = {read.step: read for read in peak}
        correlations[key.label] = {
            "peak_layer": group.peak_layer,
            "steps": steps,
            "behavior": behavior_series,
            "projection_gap": asdict(
                trajectory_correlation(
                    [by_step[step].heldout_projection_gap for step in steps], behavior_series
                )
            ),
            "drift_cosine": asdict(
                trajectory_correlation(
                    [by_step[step].cosine_to_anchor for step in steps], behavior_series
                )
            ),
        }
    return correlations


def render_drift_table(group: GroupTrajectory) -> str:
    """One group's peak-layer trajectory, with the floor and ceiling beside every cosine."""
    header = (
        f"{group.key.label}  peak_layer={group.peak_layer}  anchor_step={group.anchor_step}\n"
        f"{'step':>6} {'cos_anchor':>11} {'cos_floor':>10} {'cos_ceiling':>12} "
        f"{'norm_ratio':>11} {'gap':>9} {'gap_placebo':>12}"
    )
    rows = [
        f"{read.step:>6} {read.cosine_to_anchor:>11.4f} {read.cosine_anchor_placebo:>10.4f} "
        f"{read.anchor_split_half_cosine:>12.4f} {read.norm_ratio:>11.4f} "
        f"{read.heldout_projection_gap:>9.3f} {read.heldout_projection_gap_placebo:>12.3f}"
        for read in group.drift_reads
        if read.layer == group.peak_layer
    ]
    return "\n".join([header, *rows])


def render_axis_table(group: GroupTrajectory) -> str:
    """One group's axis quality at its peak layer, per checkpoint."""
    header = (
        f"{group.key.label}  peak_layer={group.peak_layer}\n"
        f"{'step':>6} {'n_pairs':>8} {'probe':>7} {'null_max':>9} {'dir_acc':>8} "
        f"{'plac_max':>9} {'p':>7} {'split_half':>11}"
    )
    rows = [
        f"{read.step:>6} {read.n_pairs:>8} {read.probe_accuracy:>7.3f} "
        f"{read.probe_null_accuracy_max:>9.3f} {read.direction_accuracy:>8.3f} "
        f"{read.placebo_accuracy_max:>9.3f} {read.accuracy_empirical_p:>7.3f} "
        f"{read.split_half_cosine:>11.3f}"
        for read in group.axis_reads
        if read.layer == group.peak_layer
    ]
    return "\n".join([header, *rows])


def render_cross_arm_table(reads: Sequence[CrossArmRead], *, layer: int) -> str:
    """Render the two arms side by side at one layer: do they diverge, and oppositely."""
    if not reads:
        return "no cross-arm reads"
    first = reads[0]
    header = (
        f"{first.arm_a} vs {first.arm_b}  {first.stimulus_set}|{first.pooling}  {layer=}\n"
        f"{'step':>6} {'cos_arms':>9} {'gap_a':>9} {'gap_b':>9} {'gap_diff':>9} "
        f"{'drift_a':>8} {'drift_b':>8}"
    )
    rows = [
        f"{read.step:>6} {read.cosine_between_arms:>9.4f} {read.projection_gap_a:>9.3f} "
        f"{read.projection_gap_b:>9.3f} {read.projection_gap_difference:>9.3f} "
        f"{read.drift_cosine_a:>8.4f} {read.drift_cosine_b:>8.4f}"
        for read in reads
        if read.layer == layer
    ]
    return "\n".join([header, *rows])


def save_directions(root: Path, groups: dict[GroupKey, GroupTrajectory]) -> list[Path]:
    """Write every checkpoint's per-layer directions as `{layer: tensor}`, the repo's usual shape.

    This is what the causal tier and the lens ladder consume: `games.interp_lens_ladder` transports
    one of these through each checkpoint's own lens, and `reward_hacking.interp.steering` steers and
    ablates along it against a matched-norm placebo.
    """
    written: list[Path] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0].label):
        for step, by_layer in sorted(group.step_directions.items()):
            out_dir = root / DIRECTIONS_DIRNAME / key.arm / f"step-{step}"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{key.stimulus_set}-{key.pooling}.pt"
            torch.save(by_layer, path)
            written.append(path)
    return written


def build_payload(
    groups: dict[GroupKey, GroupTrajectory],
    cross_arm: Sequence[CrossArmRead],
    correlations: dict[str, dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the JSON report: every read at every layer, plus the peak-layer summaries."""
    return {
        "context": context,
        "groups": {
            key.label: {
                "arm": key.arm,
                "stimulus_set": key.stimulus_set,
                "pooling": key.pooling,
                "anchor_step": group.anchor_step,
                "peak_layer": group.peak_layer,
                "layers": list(group.layers),
                "n_heldout_pairs": int(group.heldout_pairs.numel()),
                "axis_reads": [asdict(read) for read in group.axis_reads],
                "drift_reads": [asdict(read) for read in group.drift_reads],
            }
            for key, group in sorted(groups.items(), key=lambda item: item[0].label)
        },
        "cross_arm": [asdict(read) for read in cross_arm],
        "behavior_correlations": correlations,
    }


def load_behavior(path: Path | None) -> dict[str, dict[int, float]]:
    """Read the behavioural x-axis: `{arm: {step: value}}`, steps as JSON string keys.

    Supplied rather than derived so the analysis does not silently pick between the two behavioural
    axes this project has -- the training-time series from `trainer_state.json` and the held-out
    battery series -- which disagree in resolution and, on the twin pair, in sign.
    """
    if path is None:
        return {}
    raw = cast("dict[str, dict[str, float]]", json.loads(path.read_text()))
    return {
        arm: {int(step): float(value) for step, value in series.items()}
        for arm, series in raw.items()
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI for the trajectory analysis."""
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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--positive-side",
        required=True,
        help="Which stimulus side counts as the positive class, e.g. A for these games sets.",
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
    parser.add_argument("--n-placebos", type=int, default=DEFAULT_N_PLACEBOS)
    parser.add_argument("--n-folds", type=int, default=ProbeConfig.n_folds)
    parser.add_argument("--seed", type=int, default=ProbeConfig.seed)
    parser.add_argument(
        "--behavior-json",
        type=Path,
        default=None,
        help='{"arm": {"step": value}} behavioural series to correlate the geometry against.',
    )
    return parser


def _split(raw: str | None) -> list[str] | None:
    """Parse a comma-separated CLI list, or None for 'everything present'."""
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load the ladder, read every group, and write the report plus the directions."""
    stimuli = load_stimuli(args.stimuli)
    digest = stimuli_digest(stimuli)
    arms = _split(args.arms)
    steps = None if args.steps is None else [int(part) for part in _split(args.steps) or []]
    ladder = load_ladder(args.capture_root, arms=arms, steps=steps, stimuli_sha256=digest)
    wanted_sets = _split(args.sets) or list(ladder.stimulus_sets)
    wanted_poolings = _split(args.poolings) or list(ladder.poolings)
    layers = (
        [int(part) for part in _split(args.layers) or []]
        if args.layers is not None
        else list(range(ladder.identity.n_layers))
    )
    config = ProbeConfig(n_folds=args.n_folds, seed=args.seed)

    groups: dict[GroupKey, GroupTrajectory] = {}
    for arm in analysis_arms(ladder):
        for stimulus_set in wanted_sets:
            for pooling in wanted_poolings:
                key = GroupKey(arm=arm, stimulus_set=stimulus_set, pooling=pooling)
                groups[key] = read_group(
                    ladder,
                    key,
                    layers=layers,
                    positive_side=args.positive_side,
                    config=config,
                    n_placebos=args.n_placebos,
                )

    present_arms = analysis_arms(ladder)
    cross_arm: list[CrossArmRead] = []
    for index, arm_a in enumerate(present_arms):
        for arm_b in present_arms[index + 1 :]:
            cross_arm.extend(cross_arm_reads(groups, arm_a, arm_b))
    correlations = behavior_correlations(groups, load_behavior(args.behavior_json))

    context = {
        "capture_root": str(args.capture_root),
        "stimuli_file": str(args.stimuli),
        "stimuli_sha256": digest,
        "identity": ladder.identity.to_payload(),
        "cells": [cell.label for cell in ladder.cells],
        "positive_side": args.positive_side,
        "layers": layers,
        "n_placebos": args.n_placebos,
        "probe_config": asdict(config),
    }
    payload = build_payload(groups, cross_arm, correlations, context)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / REPORT_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    written = save_directions(args.out_dir, groups)
    logger.info(
        f"trajectory written, report={args.out_dir / REPORT_FILENAME} "
        f"groups={len(groups)} direction_files={len(written)}"
    )
    print_tables(groups, cross_arm)
    return payload


def print_tables(
    groups: dict[GroupKey, GroupTrajectory], cross_arm: Sequence[CrossArmRead]
) -> None:
    """Print every group's peak-layer tables, then each arm pair's, to stdout.

    Peak-layer only: the JSON report keeps every layer, and a 24-layer table per group per
    checkpoint is not something anyone reads on a terminal.
    """
    for group in sorted(groups.values(), key=lambda item: item.key.label):
        print(render_axis_table(group))  # noqa: T201 - a CLI whose output is the table
        print()  # noqa: T201
        print(render_drift_table(group))  # noqa: T201
        print()  # noqa: T201
    pairings = sorted(
        {(read.arm_a, read.arm_b, read.stimulus_set, read.pooling) for read in cross_arm}
    )
    for arm_a, arm_b, stimulus_set, pooling in pairings:
        subset = [
            read
            for read in cross_arm
            if (read.arm_a, read.arm_b, read.stimulus_set, read.pooling)
            == (arm_a, arm_b, stimulus_set, pooling)
        ]
        key = GroupKey(arm=arm_a, stimulus_set=stimulus_set, pooling=pooling)
        print(render_cross_arm_table(subset, layer=groups[key].peak_layer))  # noqa: T201
        print()  # noqa: T201


def main(argv: Sequence[str] | None = None) -> int:
    """Run the trajectory analysis over a capture root."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
