"""``d_disp`` and the base-to-checkpoint displacement series, read against the permuted-delta floor.

A displacement is the mean over identical rows of ``checkpoint - base``: what RL moved the residual
stream by, on prompts the model merely read. Two things make the number readable and both are built
in here rather than left to the analyst.

**The sabotage.** The difference is taken row against row before averaging, so the base against its
own declared mirror (a second capture of the same forward pass) is exactly 0.0 rather than merely
small. :func:`assert_replica_displacement_zero` checks that on the real cells and refuses an
undeclared mirror, and it is the first thing to run before any displacement below is believed. The
declaration is read through :func:`reward_hacking.interp.tmax_full_weights.declared_replica_anchor`,
the same reader the capture's own ladder guard uses, so the CLI spelling the driver stores
(``base:0``) and the games label the cells carry (``base/step-0``) never meet here.

**The floor.** The permuted-delta unit (the ``step_500`` delta with its axes permuted, keeping norm
and singular values while aligning to nothing in the base's coordinates) is captured on the same rows,
so "a delta of this size that means nothing" has a displacement of its own to compare against. Every
read carries the floor's norm and its cosine to the real displacement beside the matched-norm placebo
band (what unrelated looks like at this dimensionality, ~1/sqrt(d), not zero) and the split-half
ceiling (odd problems against even).

**The series.** Per checkpoint, rendering, pooling and layer, the displacement axis is fitted on the
ODD problems and the EVEN problems are projected onto it -- so the trajectory is read on rows the axis
never saw, in activation units, with a matched-norm placebo axis projected the same way as the floor
and the odd-versus-even cosine as the ceiling. Fitting and projecting on the same rows would make every
displacement look real, including the permuted unit's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from games.interp_cells import CapturedCell, CellFormatError, assert_rows_align
from reward_hacking.interp.directions import cosine, matched_norm_random_direction, unit
from reward_hacking.interp.eval_awareness_probe import DEFAULT_N_PLACEBOS
from reward_hacking.interp.tmax_directions import IdentifiabilityGate, records_table
from reward_hacking.interp.tmax_full_weights import REPLICA_OF_FIELD, declared_replica_anchor
from reward_hacking.interp.tmax_twin_sidecar import STIMULUS_SET

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl

logger = logging.getLogger(__name__)

FIT_PARITY = 1
PROJECT_PARITY = 0
"""Odd problems (by pair order) fit the displacement axis; even problems are projected onto it."""

MIN_ROWS_PER_SIDE = 2
"""Below this a side cannot split into a fit half and a projected half."""


class ReplicaDisplacementError(ValueError):
    """The base displaced from its own declared mirror, so the paired subtraction is misaligned."""


def rows_for_side(cell: CapturedCell, stimulus_set: str, side: str) -> torch.Tensor:
    """Row positions of one rendering, in stored order; refuses an absent side."""
    row_index = cell.rows[stimulus_set]
    rows = [row for row, held in enumerate(row_index.sides) if held == side]
    if not rows:
        raise CellFormatError(
            f"{cell.label} set {stimulus_set!r} has no rows on side {side!r}; sides are "
            f"{sorted(set(row_index.sides))}."
        )
    return torch.tensor(rows)


def assert_replica_displacement_zero(base: CapturedCell, mirror: CapturedCell) -> dict[str, Any]:
    """Require the base to displace from its declared mirror by exactly 0.0: the built-in sabotage.

    Anything else means the rows on the two sides of the subtraction are not the rows they claim to
    be, or the "mirror" was not a mirror. An undeclared mirror is refused too: a replica the capture
    did not declare is a fingerprint twin nobody vouched for, which is what the capture driver's own
    ``replica_of`` gate exists to catch. A cell without the field at all is refused by
    :func:`declared_replica_anchor` before this compares anything.
    """
    declared = declared_replica_anchor(mirror)
    if declared != base.label:
        raise ReplicaDisplacementError(
            f"{mirror.label} declares {REPLICA_OF_FIELD}={declared!r}, not {base.label}; a "
            f"base-versus-mirror zero is only a check on a mirror that was declared one"
        )
    assert_rows_align([base, mirror])
    worst = 0.0
    checks = 0
    for stimulus_set, pooling in sorted(mirror.activations):
        delta = (mirror.matrix(stimulus_set, pooling) - base.matrix(stimulus_set, pooling)).abs()
        worst = max(worst, float(delta.max()))
        checks += 1
    if worst > 0.0:
        raise ReplicaDisplacementError(
            f"{base.label} displaced from its declared mirror {mirror.label} by up to {worst:g} over "
            f"{checks} (set, pooling) blocks; the two captures are not the same forward pass"
        )
    logger.info(
        f"base-versus-mirror displacement exactly 0.0, {checks=} {base.label=} {mirror.label=}"
    )
    return {
        "n_checks": checks,
        "max_abs_displacement": worst,
        "base": base.label,
        "mirror": mirror.label,
    }


# --------------------------------------------------------------------------------------
# d_disp: the full-row displacement per rendering, pooling and layer
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DisplacementRead:
    """One rendering's mean displacement from base at one (pooling, layer), with its floors."""

    checkpoint: str
    base: str
    side: str
    pooling: str
    layer: int
    n_rows: int
    norm: float
    relative_norm: float
    split_half_cosine: float
    placebo_abs_cosine_max: float
    floor_norm: float | None
    cosine_to_floor: float | None
    disposition_label: str


def _half_means(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of the fit-parity rows and of the project-parity rows of a ``[n, ...]`` delta."""
    positions = torch.arange(delta.shape[0])
    return (
        delta[positions % 2 == FIT_PARITY].mean(dim=0),
        delta[positions % 2 == PROJECT_PARITY].mean(dim=0),
    )


def _placebo_band(direction: torch.Tensor, generator: torch.Generator, n_placebos: int) -> float:
    """Max |cosine| of ``direction`` against ``n_placebos`` matched-norm random directions."""
    return max(
        abs(cosine(direction, matched_norm_random_direction(direction, generator)))
        for _ in range(n_placebos)
    )


def displacement_directions(  # noqa: PLR0913 - a displacement is two cells, its floor, the axes and the knobs
    checkpoint: CapturedCell,
    base: CapturedCell,
    *,
    sides: Sequence[str],
    poolings: Sequence[str],
    layers: Sequence[int],
    gate: IdentifiabilityGate,
    stimulus_set: str = STIMULUS_SET,
    floor: CapturedCell | None = None,
    n_placebos: int = DEFAULT_N_PLACEBOS,
    seed: int = 0,
) -> tuple[list[DisplacementRead], dict[tuple[str, str, int], torch.Tensor]]:
    """``d_disp`` per rendering, pooling and layer: the mean of ``checkpoint - base`` over its rows.

    ``floor`` is the permuted-delta unit; its displacement on the same rows gives the norm and the
    cosine every read is compared against. The split-half is odd versus even problems.
    """
    assert_rows_align([base, checkpoint, *([floor] if floor is not None else [])])
    reads: list[DisplacementRead] = []
    directions: dict[tuple[str, str, int], torch.Tensor] = {}
    generator = torch.Generator().manual_seed(seed)
    for pooling in poolings:
        base_matrix = base.matrix(stimulus_set, pooling)
        delta_all = checkpoint.matrix(stimulus_set, pooling) - base_matrix
        floor_all = None if floor is None else floor.matrix(stimulus_set, pooling) - base_matrix
        for side in sides:
            rows = rows_for_side(checkpoint, stimulus_set, side)
            delta = delta_all[rows]
            mean_delta = delta.mean(dim=0)
            fit_half, project_half = _half_means(delta)
            split_half = torch.nn.functional.cosine_similarity(fit_half, project_half, dim=-1)
            residual_scale = base_matrix[rows].norm(dim=-1).mean(dim=0)
            floor_mean = None if floor_all is None else floor_all[rows].mean(dim=0)
            for layer in layers:
                direction = mean_delta[layer]
                reads.append(
                    DisplacementRead(
                        checkpoint=checkpoint.label,
                        base=base.label,
                        side=side,
                        pooling=pooling,
                        layer=layer,
                        n_rows=int(rows.numel()),
                        norm=float(direction.norm()),
                        relative_norm=float(direction.norm() / residual_scale[layer]),
                        split_half_cosine=float(split_half[layer]),
                        placebo_abs_cosine_max=_placebo_band(direction, generator, n_placebos),
                        floor_norm=None if floor_mean is None else float(floor_mean[layer].norm()),
                        cosine_to_floor=(
                            None if floor_mean is None else cosine(direction, floor_mean[layer])
                        ),
                        disposition_label=gate.disposition_label,
                    )
                )
                directions[side, pooling, layer] = direction
    return reads, directions


def displacement_table(reads: Sequence[DisplacementRead]) -> pl.DataFrame:
    """Return the displacement reads as one Polars table."""
    return records_table(reads, DisplacementRead)


# --------------------------------------------------------------------------------------
# The series: odd problems fit the axis, even problems are projected onto it
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesPoint:
    """One checkpoint's held-out displacement along its own odd-fitted axis, with floor and ceiling."""

    checkpoint: str
    side: str
    pooling: str
    layer: int
    n_fit_rows: int
    n_projected_rows: int
    fit_norm: float
    relative_fit_norm: float
    heldout_projection: float
    heldout_projection_placebo: float
    split_half_cosine: float
    placebo_abs_cosine_max: float
    floor_heldout_projection: float | None
    floor_fit_norm: float | None
    cosine_to_floor_axis: float | None
    disposition_label: str


@dataclass(frozen=True)
class AxisCosine:
    """Two checkpoints' odd-fitted displacement axes compared, with the band and the ceiling."""

    checkpoint_a: str
    checkpoint_b: str
    side: str
    pooling: str
    layer: int
    cosine: float
    placebo_abs_cosine_max: float
    attenuation_ceiling: float | None


@dataclass(frozen=True)
class _FitProjection:
    axis: torch.Tensor
    heldout: float
    heldout_placebo: float
    split_half: float
    n_fit: int
    n_projected: int


def _fit_and_project(delta: torch.Tensor, generator: torch.Generator) -> _FitProjection:
    """Fit the axis on the odd rows of a ``[n, d]`` delta and project the even rows onto it."""
    positions = torch.arange(delta.shape[0])
    fit_rows = delta[positions % 2 == FIT_PARITY]
    project_rows = delta[positions % 2 == PROJECT_PARITY]
    axis = fit_rows.mean(dim=0)
    placebo = matched_norm_random_direction(axis, generator)
    return _FitProjection(
        axis=axis,
        heldout=float((project_rows @ unit(axis)).mean()),
        heldout_placebo=float((project_rows @ unit(placebo)).mean()),
        split_half=cosine(axis, project_rows.mean(dim=0)),
        n_fit=int(fit_rows.shape[0]),
        n_projected=int(project_rows.shape[0]),
    )


@dataclass
class DisplacementSeries:
    """Every series point and every fitted axis, keyed by (checkpoint, side, pooling, layer)."""

    points: list[SeriesPoint] = field(default_factory=list[SeriesPoint])
    axes: dict[tuple[str, str, str, int], torch.Tensor] = field(
        default_factory=dict[tuple[str, str, str, int], torch.Tensor]
    )
    split_halves: dict[tuple[str, str, str, int], float] = field(
        default_factory=dict[tuple[str, str, str, int], float]
    )

    def table(self) -> pl.DataFrame:
        """Return the series as one Polars table."""
        return records_table(self.points, SeriesPoint)

    def axis_cosines(self, *, n_placebos: int, seed: int) -> list[AxisCosine]:
        """Cosine between every pair of checkpoints' axes at each (side, pooling, layer).

        The ceiling is the attenuation-style ``sqrt(rel_a * rel_b)`` over the two split-half cosines,
        None when either is non-positive (an axis that is not determined has no ceiling to read).
        """
        generator = torch.Generator().manual_seed(seed)
        reads: list[AxisCosine] = []
        keys = sorted(self.axes)
        for index, key_a in enumerate(keys):
            for key_b in keys[index + 1 :]:
                if key_a[1:] != key_b[1:]:
                    continue
                axis_a, axis_b = self.axes[key_a], self.axes[key_b]
                rel_a, rel_b = self.split_halves[key_a], self.split_halves[key_b]
                reads.append(
                    AxisCosine(
                        checkpoint_a=key_a[0],
                        checkpoint_b=key_b[0],
                        side=key_a[1],
                        pooling=key_a[2],
                        layer=key_a[3],
                        cosine=cosine(axis_a, axis_b),
                        placebo_abs_cosine_max=_placebo_band(axis_a, generator, n_placebos),
                        attenuation_ceiling=(
                            float((rel_a * rel_b) ** 0.5) if rel_a > 0 and rel_b > 0 else None
                        ),
                    )
                )
        return reads


def displacement_series(  # noqa: PLR0913 - a series is a base, its checkpoints, its floor, the axes and the knobs
    base: CapturedCell,
    checkpoints: Mapping[str, CapturedCell],
    *,
    sides: Sequence[str],
    poolings: Sequence[str],
    layers: Sequence[int],
    gate: IdentifiabilityGate,
    stimulus_set: str = STIMULUS_SET,
    floor: CapturedCell | None = None,
    n_placebos: int = DEFAULT_N_PLACEBOS,
    seed: int = 0,
) -> DisplacementSeries:
    """Read every checkpoint's odd-fitted, even-projected displacement, with the floor beside it."""
    assert_rows_align([base, *checkpoints.values(), *([floor] if floor is not None else [])])
    series = DisplacementSeries()
    generator = torch.Generator().manual_seed(seed)
    for pooling in poolings:
        base_matrix = base.matrix(stimulus_set, pooling)
        floor_delta = None if floor is None else floor.matrix(stimulus_set, pooling) - base_matrix
        for side in sides:
            rows = rows_for_side(base, stimulus_set, side)
            if rows.numel() < MIN_ROWS_PER_SIDE:
                raise CellFormatError(
                    f"side {side!r} has {rows.numel()} rows; the odd/even split needs at least "
                    f"{MIN_ROWS_PER_SIDE}"
                )
            residual_scale = base_matrix[rows].norm(dim=-1).mean(dim=0)
            for layer in layers:
                floor_fit = (
                    None
                    if floor_delta is None
                    else _fit_and_project(floor_delta[rows][:, layer], generator)
                )
                for name, checkpoint in checkpoints.items():
                    delta = (checkpoint.matrix(stimulus_set, pooling) - base_matrix)[rows][:, layer]
                    fit = _fit_and_project(delta, generator)
                    key = (name, side, pooling, layer)
                    series.axes[key] = fit.axis
                    series.split_halves[key] = fit.split_half
                    series.points.append(
                        SeriesPoint(
                            checkpoint=name,
                            side=side,
                            pooling=pooling,
                            layer=layer,
                            n_fit_rows=fit.n_fit,
                            n_projected_rows=fit.n_projected,
                            fit_norm=float(fit.axis.norm()),
                            relative_fit_norm=float(fit.axis.norm() / residual_scale[layer]),
                            heldout_projection=fit.heldout,
                            heldout_projection_placebo=fit.heldout_placebo,
                            split_half_cosine=fit.split_half,
                            placebo_abs_cosine_max=_placebo_band(fit.axis, generator, n_placebos),
                            floor_heldout_projection=(
                                None if floor_fit is None else floor_fit.heldout
                            ),
                            floor_fit_norm=(
                                None if floor_fit is None else float(floor_fit.axis.norm())
                            ),
                            cosine_to_floor_axis=(
                                None if floor_fit is None else cosine(fit.axis, floor_fit.axis)
                            ),
                            disposition_label=gate.disposition_label,
                        )
                    )
            logger.info(
                f"displacement series read, {pooling=} {side=} checkpoints={list(checkpoints)} "
                f"layers={len(layers)} floor={'yes' if floor is not None else 'no'}"
            )
    return series
