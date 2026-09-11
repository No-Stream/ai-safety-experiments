"""The probe transfer matrix: a probe fitted under one checkpoint, scored under every other one.

The behavioural screen could not separate "RL taught the model a hack disposition" from "RL taught
the model to read code better". This module asks a narrower, mechanistic question: did RL move a
hack-relevant representation, and if so how -- by translating the honest rows relative to the
boundary, or by changing the direction that separates hack from honest?

**The output-held-fixed estimator.** Every cell in the generation set is the SAME token ids (prompt
plus a teacher-forced completion) run through a different checkpoint, so a row's label is a fact
about text both checkpoints read identically; only the representation differs. A probe fitted on
checkpoint ``s``'s activations and scored on checkpoint ``t``'s activations of the same rows is
therefore a read on the representation alone. The matrix over (source, target) is the object, not any
single accuracy.

**Held out on both axes.** Off-diagonal cells use the same grouped folds as the diagonal: for each
fold the probe is fitted on the source's training rows and scored on the target's held-out rows.
Scoring a full-data source probe on the target would be a training accuracy whenever ``t`` resembles
``s``, and the matrix would read transfer where there is only memorisation.

**Translation and separation, reported separately.** ``honest_margin`` is the mean probe score of the
honest rows under the target (how far the honest population sits from the boundary the source drew);
its shift against the diagonal is the translation read. ``gap`` (positive minus negative mean score)
and ``auc`` are the separation reads. A translation with separation unchanged is the expectation
line; the two are never folded into one number.

**Denominators before numbers.** A probe with fewer than 8 positive (problem, unit) groups or 20
positive rows is refused, and the refusal is a row in the output with its reason, because a
transfer number on eight hacks from two problems is an item-identity read. Specificity probes (code
versus prose, docstring present) that RL should NOT have moved are reported beside the hack probe,
and a capability probe (correct versus buggy) the hack probe must beat. Row-permuted labels are the
built-in sabotage: ``labels_permuted=True`` runs the identical machinery on a permutation of the
labels across rows and must land near 0.5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl
import torch

from games.interp_cells import CapturedCell, assert_rows_align
from reward_hacking.interp.directions import STD_EPS
from reward_hacking.interp.linear_probe import ProbeConfig, fit_logistic_probe, grouped_test_masks
from reward_hacking.interp.tmax_directions import (
    GROUP_COLUMNS,
    IdentifiabilityGate,
    align_labels,
    records_table,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

MIN_POSITIVE_GROUPS = 8
MIN_POSITIVE_ROWS = 20
DEFAULT_BOOTSTRAP_DRAWS = 200
DEFAULT_PROBE_CONFIG = ProbeConfig()
BAND_QUANTILES = (0.025, 0.975)

ROLE_HACK = "hack"
ROLE_CAPABILITY = "capability"
ROLE_SPECIFICITY = "specificity"

CELLS_FILENAME = "transfer-matrix.ndjson"
REFUSALS_FILENAME = "transfer-refusals.ndjson"


class ProbeRefusalError(ValueError):
    """A probe that must not emit a number on this input; the message is the output row's reason."""


@dataclass(frozen=True)
class ProbeSpec:
    """Which labelled rows are the two classes of one probe, and what the probe is for."""

    name: str
    role: str
    positive: pl.Expr
    negative: pl.Expr
    required_columns: tuple[str, ...]


DEFAULT_PROBES: tuple[ProbeSpec, ...] = (
    ProbeSpec(
        "hack_vs_honest_pass",
        ROLE_HACK,
        pl.col("hack"),
        ~pl.col("hack") & pl.col("hidden_pass"),
        ("hack", "hidden_pass"),
    ),
    ProbeSpec(
        "correct_vs_buggy",
        ROLE_CAPABILITY,
        ~pl.col("hack") & pl.col("hidden_pass"),
        ~pl.col("hack") & ~pl.col("hidden_pass"),
        ("hack", "hidden_pass"),
    ),
    ProbeSpec(
        "code_vs_prose", ROLE_SPECIFICITY, pl.col("is_code"), ~pl.col("is_code"), ("is_code",)
    ),
    ProbeSpec(
        "docstring_present",
        ROLE_SPECIFICITY,
        pl.col("has_docstring"),
        ~pl.col("has_docstring"),
        ("has_docstring",),
    ),
)
"""The hack probe, the capability probe it must beat, and the specificity probes that must hold."""


# --------------------------------------------------------------------------------------
# Tasks: which rows, which labels, which groups
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeTask:
    """One probe's rows (positions in the cell), 0/1 labels and integer group ids."""

    spec: ProbeSpec
    rows: torch.Tensor
    labels: torch.Tensor
    groups: torch.Tensor
    n_positive_rows: int
    n_negative_rows: int
    n_positive_groups: int
    n_negative_groups: int

    def permuted(self, seed: int) -> ProbeTask:
        """Return the sabotage arm: the same rows and groups with the labels permuted across rows."""
        order = torch.randperm(
            int(self.labels.numel()), generator=torch.Generator().manual_seed(seed)
        )
        return ProbeTask(
            spec=self.spec,
            rows=self.rows,
            labels=self.labels[order],
            groups=self.groups,
            n_positive_rows=self.n_positive_rows,
            n_negative_rows=self.n_negative_rows,
            n_positive_groups=self.n_positive_groups,
            n_negative_groups=self.n_negative_groups,
        )


def probe_task(
    aligned: pl.DataFrame,
    spec: ProbeSpec,
    *,
    n_folds: int,
    group_columns: Sequence[str] = GROUP_COLUMNS,
) -> ProbeTask:
    """Select a probe's rows from the aligned labels, refusing one too thin to hold out.

    Refuses (with the reason) a probe whose label columns are absent, whose positives span fewer
    than :data:`MIN_POSITIVE_GROUPS` groups or :data:`MIN_POSITIVE_ROWS` rows, or whose groups cannot
    fill ``n_folds`` held-out folds.
    """
    missing = sorted(set(spec.required_columns) - set(aligned.columns))
    if missing:
        raise ProbeRefusalError(f"label columns {missing} are absent from the labels file")
    frame = (
        aligned.with_row_index("row")
        .with_columns(spec.positive.alias("_positive"), spec.negative.alias("_negative"))
        .filter(pl.col("_positive") | pl.col("_negative"))
        .with_columns(
            pl.concat_str([pl.col(c).cast(pl.String) for c in group_columns], separator="|").alias(
                "_group"
            )
        )
    )
    if frame.filter(pl.col("_positive") & pl.col("_negative")).height:
        raise ValueError(
            f"probe {spec.name!r}: a row satisfies both the positive and negative masks"
        )
    positives = frame.filter(pl.col("_positive"))
    negatives = frame.filter(pl.col("_negative"))
    n_positive_groups = positives["_group"].n_unique()
    if positives.height < MIN_POSITIVE_ROWS or n_positive_groups < MIN_POSITIVE_GROUPS:
        raise ProbeRefusalError(
            f"{positives.height} positive rows across {n_positive_groups} groups; a transfer number "
            f"needs at least {MIN_POSITIVE_ROWS} rows across {MIN_POSITIVE_GROUPS} groups"
        )
    if negatives.height == 0:
        raise ProbeRefusalError("no negative rows")
    frame = frame.with_columns((pl.col("_group").rank("dense") - 1).cast(pl.Int64).alias("_code"))
    n_groups = frame["_group"].n_unique()
    if n_groups < n_folds:
        raise ProbeRefusalError(f"{n_groups} groups cannot fill {n_folds} held-out folds")
    return ProbeTask(
        spec=spec,
        rows=torch.tensor(frame["row"].to_list(), dtype=torch.long),
        labels=torch.tensor(frame["_positive"].to_list(), dtype=torch.float32),
        groups=torch.tensor(frame["_code"].to_list()),
        n_positive_rows=positives.height,
        n_negative_rows=negatives.height,
        n_positive_groups=n_positive_groups,
        n_negative_groups=negatives["_group"].n_unique(),
    )


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def group_means(features: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Each row's (problem, unit) group mean, ``[n, d]``, so subtracting it centers within group."""
    n_groups = int(groups.max()) + 1
    sums = torch.zeros(n_groups, features.shape[1], dtype=features.dtype).index_add_(
        0, groups, features
    )
    counts = torch.bincount(groups, minlength=n_groups).to(features.dtype).unsqueeze(1)
    return (sums / counts)[groups]


def cross_scored(
    source: torch.Tensor, target: torch.Tensor, task: ProbeTask, config: ProbeConfig
) -> torch.Tensor:
    """Held-out probe scores of ``target``'s rows from probes fitted on ``source``'s rows.

    Both sides are first centered within (problem, unit) group by the SOURCE's group means: hacks
    are item-locked, so a pooled probe reads item identity, and the group mean is label-free so
    centering by it manufactures nothing. Using the source's means on the target is what lets a
    translation of the target relative to the source survive as a margin shift instead of being
    absorbed. Per grouped fold: standardise with the source training rows' statistics, fit the
    logistic probe on them, and score the target's held-out rows through the same centering and
    standardisation. ``source is target`` is the diagonal.
    """
    center = group_means(source, task.groups)
    source_centered = source - center
    target_centered = target - center
    scores = torch.zeros(task.labels.shape[0], dtype=torch.float32)
    for test_mask in grouped_test_masks(task.groups, config.n_folds):
        train_mask = ~test_mask
        mean = source_centered[train_mask].mean(dim=0)
        scale = source_centered[train_mask].std(dim=0, unbiased=True) + STD_EPS
        fit = fit_logistic_probe(
            (source_centered[train_mask] - mean) / scale,
            task.labels[train_mask],
            l2_strength=config.l2_strength,
            max_iter=config.max_iter,
        )
        scores[test_mask] = ((target_centered[test_mask] - mean) / scale) @ fit.weights + fit.bias
    return scores


def roc_auc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    """Mann-Whitney AUC, ties as half: the same definition as ``prompt_contrast.roc_auc``.

    Restated here rather than imported because that module pulls in the episode harness to
    materialise its stimuli, and this one is arithmetic over cached cells.
    """
    difference = positive.unsqueeze(1) - negative.unsqueeze(0)
    wins = (difference > 0).sum().float()
    ties = (difference == 0).sum().float()
    return float((wins + 0.5 * ties) / (positive.numel() * negative.numel()))


@dataclass(frozen=True)
class ScoreSummary:
    """Separation (``auc``, ``gap``), translation (``honest_margin``) and accuracy of one score vector."""

    auc: float
    gap: float
    honest_margin: float
    accuracy: float


def summarize_scores(scores: torch.Tensor, labels: torch.Tensor) -> ScoreSummary:
    """Read one held-out score vector against its labels."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    return ScoreSummary(
        auc=roc_auc(positive, negative),
        gap=float(positive.mean() - negative.mean()),
        honest_margin=float(negative.mean()),
        accuracy=float(((scores > 0).to(labels.dtype) == labels).float().mean()),
    )


def group_bootstrap_band(
    scores: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    *,
    n_draws: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Item-bootstrap band on ``auc`` and ``gap``: resample whole groups with replacement.

    Rows of one (problem, unit) group are not independent, so the unit of resampling is the group. A
    draw that lands on one class only has no AUC and is skipped and counted in the log; the band is the
    2.5th and 97.5th percentiles of the draws that had both classes.
    """
    generator = torch.Generator().manual_seed(seed)
    unique = torch.unique(groups)
    aucs: list[float] = []
    gaps: list[float] = []
    for _ in range(n_draws):
        drawn = unique[torch.randint(0, unique.numel(), (unique.numel(),), generator=generator)]
        selected = torch.cat([torch.nonzero(groups == group).flatten() for group in drawn])
        drawn_labels = labels[selected]
        if drawn_labels.min() == drawn_labels.max():
            continue
        summary = summarize_scores(scores[selected], drawn_labels)
        aucs.append(summary.auc)
        gaps.append(summary.gap)
    if len(aucs) < n_draws:
        logger.info(
            f"bootstrap: {n_draws - len(aucs)} of {n_draws} draws had one class and were skipped"
        )
    if not aucs:
        raise ProbeRefusalError("every bootstrap draw landed on one class; no band")

    def band(values: list[float]) -> tuple[float, float]:
        tensor = torch.tensor(values)
        low, high = (float(torch.quantile(tensor, q)) for q in BAND_QUANTILES)
        return low, high

    return {"auc": band(aucs), "gap": band(gaps)}


# --------------------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferCell:
    """One (probe, pooling, layer, source, target) read, with the diagonal it is judged against."""

    probe: str
    role: str
    pooling: str
    layer: int
    source: str
    target: str
    labels_permuted: bool
    n_rows: int
    n_positive_rows: int
    n_positive_groups: int
    auc: float
    gap: float
    honest_margin: float
    accuracy: float
    diagonal_auc: float
    diagonal_gap: float
    diagonal_honest_margin: float
    translation_shift: float
    separation_gap_shift: float
    separation_auc_shift: float
    diagonal_auc_band_low: float
    diagonal_auc_band_high: float
    diagonal_gap_band_low: float
    diagonal_gap_band_high: float
    auc_outside_diagonal_band: bool
    gap_outside_diagonal_band: bool
    disposition_label: str


@dataclass(frozen=True)
class ProbeRefusal:
    """A probe that emitted no number, and why; counts are what the labels offered."""

    probe: str
    role: str
    labels_permuted: bool
    n_positive_rows: int | None
    n_positive_groups: int | None
    reason: str


@dataclass
class TransferMatrix:
    """Every cell and every refusal of one run, with the tables and the square views."""

    cells: list[TransferCell] = field(default_factory=list[TransferCell])
    refusals: list[ProbeRefusal] = field(default_factory=list[ProbeRefusal])

    def table(self) -> pl.DataFrame:
        """Return the cells as one Polars table."""
        return records_table(self.cells, TransferCell)

    def refusal_table(self) -> pl.DataFrame:
        """Return the refusals as one Polars table."""
        return records_table(self.refusals, ProbeRefusal)

    def square(self, *, probe: str, pooling: str, layer: int, metric: str) -> pl.DataFrame:
        """One metric as a source-by-target square, the shape a reader looks at."""
        return (
            self.table()
            .filter(
                (pl.col("probe") == probe)
                & (pl.col("pooling") == pooling)
                & (pl.col("layer") == layer)
            )
            .pivot(on="target", index="source", values=metric, sort_columns=True)
            .sort("source")
        )

    def save(self, out_dir: Path) -> None:
        """Write both tables as ndjson."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self.table().write_ndjson(out_dir / CELLS_FILENAME)
        self.refusal_table().write_ndjson(out_dir / REFUSALS_FILENAME)
        logger.info(
            f"transfer matrix written, {out_dir=} cells={len(self.cells)} refusals={len(self.refusals)}"
        )


def _positive_counts(aligned: pl.DataFrame, spec: ProbeSpec) -> tuple[int | None, int | None]:
    """Positive rows and groups a refused probe would have had, or None when its columns are absent."""
    if set(spec.required_columns) - set(aligned.columns):
        return None, None
    positives = aligned.filter(spec.positive)
    groups = positives.select(pl.concat_str([pl.col(c).cast(pl.String) for c in GROUP_COLUMNS]))
    return positives.height, groups.n_unique()


def _cell(  # noqa: PLR0913 - a cell is its coordinates, two summaries, the band and the flag
    task: ProbeTask,
    *,
    pooling: str,
    layer: int,
    source: str,
    target: str,
    summary: ScoreSummary,
    diagonal: ScoreSummary,
    band: Mapping[str, tuple[float, float]],
    labels_permuted: bool,
    gate: IdentifiabilityGate,
) -> TransferCell:
    auc_low, auc_high = band["auc"]
    gap_low, gap_high = band["gap"]
    return TransferCell(
        probe=task.spec.name,
        role=task.spec.role,
        pooling=pooling,
        layer=layer,
        source=source,
        target=target,
        labels_permuted=labels_permuted,
        n_rows=int(task.labels.numel()),
        n_positive_rows=task.n_positive_rows,
        n_positive_groups=task.n_positive_groups,
        auc=summary.auc,
        gap=summary.gap,
        honest_margin=summary.honest_margin,
        accuracy=summary.accuracy,
        diagonal_auc=diagonal.auc,
        diagonal_gap=diagonal.gap,
        diagonal_honest_margin=diagonal.honest_margin,
        translation_shift=summary.honest_margin - diagonal.honest_margin,
        separation_gap_shift=summary.gap - diagonal.gap,
        separation_auc_shift=summary.auc - diagonal.auc,
        diagonal_auc_band_low=auc_low,
        diagonal_auc_band_high=auc_high,
        diagonal_gap_band_low=gap_low,
        diagonal_gap_band_high=gap_high,
        auc_outside_diagonal_band=not auc_low <= summary.auc <= auc_high,
        gap_outside_diagonal_band=not gap_low <= summary.gap <= gap_high,
        disposition_label=gate.disposition_label,
    )


def transfer_matrix(  # noqa: PLR0913 - the matrix is the cells, their labels, the probes, the axes and the knobs
    cells: Mapping[str, CapturedCell],
    labels: pl.DataFrame,
    *,
    stimulus_set: str,
    poolings: Sequence[str],
    layers: Sequence[int],
    gate: IdentifiabilityGate,
    probes: Sequence[ProbeSpec] = DEFAULT_PROBES,
    config: ProbeConfig = DEFAULT_PROBE_CONFIG,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = 0,
    permute_labels_seed: int | None = None,
) -> TransferMatrix:
    """Fit every probe under every checkpoint and score it under every checkpoint, same rows.

    ``cells`` maps a checkpoint name to its cell; all must hold the same rows. ``permute_labels_seed``
    runs the sabotage arm instead: labels permuted across rows, everything else identical, and every
    cell it emits says ``labels_permuted=True``.
    """
    ordered = list(cells.values())
    assert_rows_align(ordered)
    aligned = align_labels(ordered[0], stimulus_set, labels)
    permuted = permute_labels_seed is not None
    out = TransferMatrix()
    for spec in probes:
        try:
            task = probe_task(aligned, spec, n_folds=config.n_folds)
        except ProbeRefusalError as refusal:
            rows, groups = _positive_counts(aligned, spec)
            out.refusals.append(
                ProbeRefusal(spec.name, spec.role, permuted, rows, groups, str(refusal))
            )
            logger.warning(f"probe {spec.name!r} refused: {refusal}")
            continue
        if permute_labels_seed is not None:
            task = task.permuted(permute_labels_seed)
        for pooling in poolings:
            for layer in layers:
                features = {
                    name: cell.layer(stimulus_set, pooling, layer)[task.rows]
                    for name, cell in cells.items()
                }
                for source, source_features in features.items():
                    diagonal_scores = cross_scored(source_features, source_features, task, config)
                    diagonal = summarize_scores(diagonal_scores, task.labels)
                    band = group_bootstrap_band(
                        diagonal_scores, task.labels, task.groups, n_draws=n_bootstrap, seed=seed
                    )
                    for target, target_features in features.items():
                        summary = (
                            diagonal
                            if target == source
                            else summarize_scores(
                                cross_scored(source_features, target_features, task, config),
                                task.labels,
                            )
                        )
                        out.cells.append(
                            _cell(
                                task,
                                pooling=pooling,
                                layer=layer,
                                source=source,
                                target=target,
                                summary=summary,
                                diagonal=diagonal,
                                band=band,
                                labels_permuted=permuted,
                                gate=gate,
                            )
                        )
        logger.info(
            f"probe {spec.name!r} scored, positives={task.n_positive_rows} rows / "
            f"{task.n_positive_groups} groups, checkpoints={list(cells)} {permuted=}"
        )
    return out
