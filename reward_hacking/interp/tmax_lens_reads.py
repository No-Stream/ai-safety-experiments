"""Decode the TMAX wave's directions through fitted Jacobian lenses, floors beside every token list.

A Jacobian lens transports a layer-``L`` residual direction into final-layer coordinates where the
model's own unembedding turns it into a ranked token list. That list is the most faithful "what does
this direction say" the wave has, and also the easiest to over-read: every direction, including a
random one, decodes to *some* confident-looking tokens. So every decode here is one row of a table
whose other columns are the controls:

* the **matched-norm random-direction floor** -- ``n_placebos`` random directions of the same norm
  decoded through the same lens at the same layer, reported as the mean and max top-k overlap with
  the real list (what "unrelated" looks like);
* the **stored-decode gate** (``lens_geometry.StoredDecodeGate``) -- a decode at a layer where the
  lens does not recover the model's own next tokens, or recovers them just as well when told the
  anchors sit eight layers away, returns NO tokens and says why; the verdict is a property of (lens,
  layer), so it is computed once per pair through :class:`GateVerdictCache` and every family,
  stratum, variant and transfer at that pair reads the one verdict (dozens of anchor decodes per
  check, and a 9B read asks about the same pair dozens of times);
* the **standardized variant** -- the same direction with the residual stream's per-dimension scale
  divided out and the norm restored, decoded beside the raw one, because massive activations once
  pinned a cosine at 0.99 and dominate a raw decode the same way;
* the **lens-transfer read** -- one direction decoded through two lenses (the base's applied to TMAX
  activations, and TMAX's own) with the overlap read against a placebo floor, a wrong-layer floor and
  the split-half ceiling;
* the **denominators** -- the narration judge's resolved noticed, complied and noticed-and-refused
  counts travel on the twin-difference rows, and the identifiability label on every row.

Decodes below layer 15 are refused: the lens is not coherent there on this family, and a list from
layer 8 is the sabotage the gate exists to fail, not a read. Decodes at a layer the lens never fitted
are skipped and counted rather than attempted: a ``jlens`` lens fits every layer but the model's last
as a source (0..N-2), while the directions stage fits directions at every captured layer (0..N-1), so
the last layer's direction has no transport; its residual already sits in final-layer coordinates
and nothing is lost by leaving it out of a transported read.

**Where the lens, the gate and the unembedding come from.** A lens directory written by
``reward_hacking.interp.tmax_lens_fit`` is opened with ``lens_artifacts.load_lens_bundle``, which
holds the gate's anchors to the lens file they were recorded beside and to the weights the lens was
fitted on; :func:`decode_context_for` then holds the bundle to the captured cell the directions came
from (the cell's ``weights_fingerprint`` must be the lens's, or the decode is refused by name). The
one deliberate mismatch, the lens-transfer read of a base direction through a TMAX lens, goes through
:func:`lens_transfer_read`, which names both lenses on every row. The CLI at the bottom is the stage-6
entry point: a directions file, the cell it was fitted on, one lens directory per read, one table.

**Privacy.** The token lists echo grader text and planted literals. :meth:`LensReadBundle.save`
refuses any output directory under a tracked path of this repository; the run's artifact prefix is the
only place these tables go.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import AutoTokenizer

from games.interp_cells import CapturedCell, read_cell
from reward_hacking.interp.directions import matched_norm_random_direction
from reward_hacking.interp.jacobian import LensLike, UnembedModel, transport_and_decode
from reward_hacking.interp.lens_artifacts import LensBundle, load_lens_bundle
from reward_hacking.interp.lens_geometry import (
    DEFAULT_GATE_TOP_K,
    CosineWithBand,
    GateResult,
    StoredDecodeGate,
    assert_gate_matches_weights,
    cosine_with_band,
    per_dimension_scale,
    standardize_direction,
    topk_overlap,
    wrong_layer_for,
)
from reward_hacking.interp.tmax_directions import (
    STRATUM_ALL,
    DirectionKey,
    IdentifiabilityGate,
    load_directions,
    records_table,
)
from reward_hacking.interp.tmax_full_weights import cell_fingerprint
from reward_hacking.legibility_narration_summary import ALL_STRATA, IDENTIFIABILITY_DEFINITIONS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import polars as pl

logger = logging.getLogger(__name__)

MIN_DECODE_LAYER = 15
"""Below this the lens is not coherent on this family; a decode there is the sabotage, not a read."""

REPO_ROOT = Path(__file__).resolve().parents[2]
UNTRACKED_ROOTS: tuple[str, ...] = ("artifacts", "docs/scratch")
"""The only places inside the repository a decode table may land; anything else tracked is refused."""

VARIANT_RAW = "raw"
VARIANT_STANDARDIZED = "standardized"

DECODES_FILENAME = "lens-decodes.ndjson"
TRANSFERS_FILENAME = "lens-transfers.ndjson"
COSINES_FILENAME = "direction-cosines.ndjson"


class TrackedPathError(ValueError):
    """An output directory that would put grader text under version control."""


class DecodeLayerError(ValueError):
    """A decode asked for at a layer the lens is not read at."""


def refuse_tracked_path(path: Path, *, repo_root: Path = REPO_ROOT) -> Path:
    """Return ``path`` resolved, or raise if it sits inside the repository outside a gitignored root."""
    resolved = path.resolve()
    root = repo_root.resolve()
    if not resolved.is_relative_to(root):
        return resolved
    relative = resolved.relative_to(root)
    if any(relative.is_relative_to(Path(allowed)) for allowed in UNTRACKED_ROOTS):
        return resolved
    raise TrackedPathError(
        f"{path} resolves inside the repository at {relative}, which is a tracked location; lens "
        f"decodes echo grader text and may only be written under {UNTRACKED_ROOTS} or outside the repo"
    )


@dataclass(frozen=True)
class ResolvedCounts:
    """The narration judge's denominators that travel with a twin-difference decode.

    ``noticed`` and ``complied`` are the unit's all-strata counts; ``noticed_and_refused`` is the
    unit's gate group under the named ``definition``, read from the summary's identifiability block
    so the count on every row is the same group the gate was decided on.
    """

    noticed: int
    complied: int
    noticed_and_refused: int
    source: str
    definition: str


# --------------------------------------------------------------------------------------
# One decode with its floor and its gate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeRead:
    """One direction decoded through one lens at one layer, beside everything that qualifies it."""

    name: str
    lens: str
    layer: int
    variant: str
    stratum: str
    top_k: int
    tokens: tuple[str, ...]
    logits: tuple[float, ...]
    direction_norm: float
    n_placebos: int
    placebo_overlap_mean: float | None
    placebo_overlap_max: float | None
    gate_admitted: bool | None
    gate_hit_rate: float | None
    gate_wrong_layer_hit_rate: float | None
    gate_reason: str | None
    n_noticed: int | None
    n_complied: int | None
    n_noticed_and_refused: int | None
    disposition_label: str
    disposition_definition: str


def gate_lens_key(gate: StoredDecodeGate) -> str:
    """Name the lens a gate belongs to: the lens file's digest when the gate carries provenance, else its name."""
    if gate.provenance is not None:
        return gate.provenance.lens_sha256
    return gate.lens_name


@dataclass
class GateVerdictCache:
    """One stored-decode-gate verdict per (lens, layer, top-k), computed on first use and reused.

    The verdict depends on the gate's anchors, the lens's transport at ``layer`` and the unembedding,
    none of which change within a read, while a read asks about the same (lens, layer) once per family,
    stratum, variant and transfer. Keyed on the lens file's digest rather than on the objects, so the
    several contexts of one read (one per pooling) share a cache and a second lens never reads the
    first's verdict. The verdict is logged once, by the check itself, when it is computed.
    """

    verdicts: dict[tuple[str, int, int], GateResult] = field(
        default_factory=dict[tuple[str, int, int], GateResult]
    )

    def verdict(self, context: DecodeContext, layer: int) -> GateResult:
        """Return the gate's verdict for ``context``'s lens at ``layer``, computing it once."""
        if context.gate is None:
            raise ValueError(f"context {context.lens_name!r} has no gate to take a verdict from")
        key = (gate_lens_key(context.gate), layer, context.gate_top_k)
        if key not in self.verdicts:
            self.verdicts[key] = context.gate.check(
                context.lens, context.model, layer, k=context.gate_top_k
            )
        return self.verdicts[key]


@dataclass(frozen=True)
class DecodeContext:
    """Everything a decode needs besides the direction: the lens pair, the naming and the knobs."""

    lens: LensLike
    model: UnembedModel
    lens_name: str
    id_to_token: Callable[[int], str]
    top_k: int
    n_placebos: int
    gate: StoredDecodeGate | None
    disposition: IdentifiabilityGate
    fitted_layers: frozenset[int]
    scale_by_layer: Mapping[int, torch.Tensor] | None = None
    resolved: ResolvedCounts | None = None
    gate_top_k: int = DEFAULT_GATE_TOP_K
    gate_verdicts: GateVerdictCache = field(default_factory=GateVerdictCache)


def _decode_tokens(
    context: DecodeContext, direction: torch.Tensor, layer: int
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    readouts = transport_and_decode(
        context.lens,
        context.model,
        direction,
        layer,
        id_to_token=context.id_to_token,
        k=context.top_k,
    )
    return tuple(r.token for r in readouts), tuple(r.logit for r in readouts)


def decode_with_floor(  # noqa: PLR0913 - a decode is a direction, its coordinates, the context and a seed
    direction: torch.Tensor,
    context: DecodeContext,
    *,
    name: str,
    layer: int,
    stratum: str = STRATUM_ALL,
    standardized: bool = False,
    seed: int = 0,
) -> DecodeRead:
    """Decode one direction at one layer with the placebo floor and the gate; no tokens if refused."""
    if layer < MIN_DECODE_LAYER:
        raise DecodeLayerError(
            f"layer {layer} is below {MIN_DECODE_LAYER}; the lens is not read there on this family"
        )
    if layer not in context.fitted_layers:
        raise DecodeLayerError(
            f"layer {layer} is not a fitted source layer of lens {context.lens_name!r} "
            f"({min(context.fitted_layers)}..{max(context.fitted_layers)}); there is no transport to "
            f"decode through"
        )
    variant = VARIANT_RAW
    if standardized:
        if context.scale_by_layer is None or layer not in context.scale_by_layer:
            raise ValueError(f"no per-dimension scale for layer {layer}; cannot standardize")
        direction = standardize_direction(direction, context.scale_by_layer[layer])
        variant = VARIANT_STANDARDIZED
    gate = None if context.gate is None else context.gate_verdicts.verdict(context, layer)
    admitted = gate is None or gate.admitted
    tokens: tuple[str, ...] = ()
    logits: tuple[float, ...] = ()
    overlaps: list[float] = []
    if admitted:
        tokens, logits = _decode_tokens(context, direction, layer)
        generator = torch.Generator().manual_seed(seed + layer)
        for _ in range(context.n_placebos):
            placebo_tokens, _ = _decode_tokens(
                context, matched_norm_random_direction(direction, generator), layer
            )
            overlaps.append(topk_overlap(tokens, placebo_tokens, context.top_k))
    else:
        logger.warning(
            f"decode of {name} at layer {layer} refused by the gate: {gate.reason if gate else ''}"
        )
    resolved = context.resolved
    return DecodeRead(
        name=name,
        lens=context.lens_name,
        layer=layer,
        variant=variant,
        stratum=stratum,
        top_k=context.top_k,
        tokens=tokens,
        logits=logits,
        direction_norm=float(direction.norm()),
        n_placebos=context.n_placebos,
        placebo_overlap_mean=sum(overlaps) / len(overlaps) if overlaps else None,
        placebo_overlap_max=max(overlaps) if overlaps else None,
        gate_admitted=None if gate is None else gate.admitted,
        gate_hit_rate=None if gate is None else gate.hit_rate,
        gate_wrong_layer_hit_rate=None if gate is None else gate.wrong_layer_hit_rate,
        gate_reason=None if gate is None else gate.reason,
        n_noticed=None if resolved is None else resolved.noticed,
        n_complied=None if resolved is None else resolved.complied,
        n_noticed_and_refused=None if resolved is None else resolved.noticed_and_refused,
        disposition_label=context.disposition.disposition_label,
        disposition_definition=context.disposition.definition,
    )


def decode_family(
    directions_by_layer: Mapping[int, torch.Tensor],
    context: DecodeContext,
    *,
    name: str,
    stratum: str = STRATUM_ALL,
    seed: int = 0,
) -> list[DecodeRead]:
    """Decode one direction family at every fitted layer from :data:`MIN_DECODE_LAYER` up, raw and standardized.

    Layers below the floor and layers the lens never fitted (the model's last, which ``jlens`` leaves
    out of the sources) are skipped and counted apart. The standardized variant is only emitted where
    a per-dimension scale was supplied for the layer, and its absence is logged: a table with raw rows
    only is a table that was not standardized, and that should be visible rather than inferred from a
    missing row.
    """
    reads: list[DecodeRead] = []
    skipped_low = 0
    skipped_unfitted = 0
    for layer in sorted(directions_by_layer):
        if layer < MIN_DECODE_LAYER:
            skipped_low += 1
            continue
        if layer not in context.fitted_layers:
            skipped_unfitted += 1
            continue
        direction = directions_by_layer[layer]
        reads.append(
            decode_with_floor(
                direction, context, name=name, layer=layer, stratum=stratum, seed=seed
            )
        )
        if context.scale_by_layer is not None and layer in context.scale_by_layer:
            reads.append(
                decode_with_floor(
                    direction,
                    context,
                    name=name,
                    layer=layer,
                    stratum=stratum,
                    standardized=True,
                    seed=seed,
                )
            )
    logger.info(
        f"{name} decoded through {context.lens_name}, rows={len(reads)} "
        f"layers_below_{MIN_DECODE_LAYER}_skipped={skipped_low} "
        f"layers_unfitted_skipped={skipped_unfitted} "
        f"standardized={'yes' if context.scale_by_layer else 'no'}"
    )
    return reads


# --------------------------------------------------------------------------------------
# Lens transfer: one direction through two lenses
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LensPair:
    """A named lens with the model whose unembedding it decodes through."""

    name: str
    lens: LensLike
    model: UnembedModel


@dataclass(frozen=True)
class LensTransferRead:
    """One direction decoded through the reference lens and another, overlap beside floor and ceiling."""

    name: str
    layer: int
    reference_lens: str
    other_lens: str
    top_k: int
    overlap: float
    placebo_overlap_mean: float
    placebo_overlap_max: float
    wrong_layer_overlap: float
    wrong_layer: int
    split_half_overlap: float | None
    n_placebos: int
    disposition_label: str


def lens_transfer_read(  # noqa: PLR0913 - a transfer is a direction, its halves, two lenses, a layer and the knobs
    direction: torch.Tensor,
    *,
    name: str,
    layer: int,
    reference: LensPair,
    other: LensPair,
    id_to_token: Callable[[int], str],
    top_k: int,
    n_placebos: int,
    wrong_layer: int,
    disposition: IdentifiabilityGate,
    halves: tuple[torch.Tensor, torch.Tensor] | None = None,
    seed: int = 0,
) -> LensTransferRead:
    """Decode ``direction`` through both lenses and read the overlap against three references.

    Floor one: the same matched-norm random direction through both lenses (what an unrelated direction's
    two decodes share). Floor two: the direction through the reference lens at ``layer`` against the
    reference lens at ``wrong_layer`` (what a mis-specified transport shares with the right one).
    Ceiling: the odd-half and even-half fits of the same direction, both through the reference lens.
    """
    if layer < MIN_DECODE_LAYER:
        raise DecodeLayerError(f"layer {layer} is below {MIN_DECODE_LAYER}")

    def tokens(pair: LensPair, vector: torch.Tensor, at: int) -> tuple[str, ...]:
        return tuple(
            r.token
            for r in transport_and_decode(
                pair.lens, pair.model, vector, at, id_to_token=id_to_token, k=top_k
            )
        )

    reference_tokens = tokens(reference, direction, layer)
    generator = torch.Generator().manual_seed(seed + layer)
    placebo_overlaps: list[float] = []
    for _ in range(n_placebos):
        placebo = matched_norm_random_direction(direction, generator)
        placebo_overlaps.append(
            topk_overlap(tokens(reference, placebo, layer), tokens(other, placebo, layer), top_k)
        )
    return LensTransferRead(
        name=name,
        layer=layer,
        reference_lens=reference.name,
        other_lens=other.name,
        top_k=top_k,
        overlap=topk_overlap(reference_tokens, tokens(other, direction, layer), top_k),
        placebo_overlap_mean=sum(placebo_overlaps) / len(placebo_overlaps),
        placebo_overlap_max=max(placebo_overlaps),
        wrong_layer_overlap=topk_overlap(
            reference_tokens, tokens(reference, direction, wrong_layer), top_k
        ),
        wrong_layer=wrong_layer,
        split_half_overlap=(
            None
            if halves is None
            else topk_overlap(
                tokens(reference, halves[0], layer), tokens(reference, halves[1], layer), top_k
            )
        ),
        n_placebos=n_placebos,
        disposition_label=disposition.disposition_label,
    )


# --------------------------------------------------------------------------------------
# Direction-to-direction cosines (eval awareness against the twin and hack axes)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectionCosine:
    """Cosine of one direction family to another at one layer, with the band, and its role."""

    first: str
    second: str
    role: str
    layer: int
    cosine: float
    placebo_abs_cosine_mean: float
    placebo_abs_cosine_max: float
    clears_band: bool
    n_placebos: int


ROLE_TARGET = "target"
ROLE_CONTENT_CONTROL = "content-control"


def direction_cosines(  # noqa: PLR0913 - a cosine table is one family, its targets, its controls and the knobs
    reference_by_layer: Mapping[int, torch.Tensor],
    *,
    reference_name: str,
    targets: Mapping[str, Mapping[int, torch.Tensor]],
    controls: Mapping[str, Mapping[int, torch.Tensor]],
    n_placebos: int,
    seed: int = 0,
) -> list[DirectionCosine]:
    """Compare ``reference`` (typically ``e``) to each target and each content control, per shared layer.

    Targets are the directions a cosine to ``e`` would be a finding about; controls are the content
    directions (comment line, statement literal) whose cosines are the floor that finding must beat.
    """
    generator = torch.Generator().manual_seed(seed)
    rows: list[DirectionCosine] = []
    for role, families in ((ROLE_TARGET, targets), (ROLE_CONTENT_CONTROL, controls)):
        for name, by_layer in families.items():
            for layer in sorted(set(reference_by_layer) & set(by_layer)):
                banded: CosineWithBand = cosine_with_band(
                    reference_by_layer[layer],
                    by_layer[layer],
                    n_placebos=n_placebos,
                    generator=generator,
                )
                rows.append(
                    DirectionCosine(
                        first=reference_name,
                        second=name,
                        role=role,
                        layer=layer,
                        cosine=banded.cosine,
                        placebo_abs_cosine_mean=banded.placebo_abs_cosine_mean,
                        placebo_abs_cosine_max=banded.placebo_abs_cosine_max,
                        clears_band=banded.clears_band,
                        n_placebos=banded.n_placebos,
                    )
                )
    return rows


# --------------------------------------------------------------------------------------
# The bundle, and the only place it may be written
# --------------------------------------------------------------------------------------


@dataclass
class LensReadBundle:
    """Every decode, transfer and cosine of one run, saved together under the run prefix only."""

    decodes: list[DecodeRead] = field(default_factory=list[DecodeRead])
    transfers: list[LensTransferRead] = field(default_factory=list[LensTransferRead])
    cosines: list[DirectionCosine] = field(default_factory=list[DirectionCosine])

    def decode_table(self) -> pl.DataFrame:
        """Return the decodes as one Polars table (token lists included: grader text, untracked paths only)."""
        return records_table(self.decodes, DecodeRead)

    def transfer_table(self) -> pl.DataFrame:
        """Return the lens-transfer reads as one Polars table."""
        return records_table(self.transfers, LensTransferRead)

    def cosine_table(self) -> pl.DataFrame:
        """Return the direction cosines as one Polars table."""
        return records_table(self.cosines, DirectionCosine)

    def save(self, out_dir: Path, *, repo_root: Path = REPO_ROOT) -> Path:
        """Write the three tables, refusing any directory under a tracked path of the repository."""
        resolved = refuse_tracked_path(out_dir, repo_root=repo_root)
        resolved.mkdir(parents=True, exist_ok=True)
        self.decode_table().write_ndjson(resolved / DECODES_FILENAME)
        self.transfer_table().write_ndjson(resolved / TRANSFERS_FILENAME)
        self.cosine_table().write_ndjson(resolved / COSINES_FILENAME)
        logger.info(
            f"lens reads written, out_dir={resolved} decodes={len(self.decodes)} "
            f"transfers={len(self.transfers)} cosines={len(self.cosines)}"
        )
        return resolved


def scale_by_layer_from_matrix(matrix: torch.Tensor) -> dict[int, torch.Tensor]:
    """Per-layer standardising scale from a cell's ``[rows, layers, hidden]`` activations."""
    return {
        layer: per_dimension_scale(matrix[:, layer, :]) for layer in range(int(matrix.shape[1]))
    }


def resolved_counts_from_summary(
    summary: Mapping[str, Any], *, unit: str, source: str, definition: str
) -> ResolvedCounts:
    """Pull one unit's noticed / complied counts and its gate group under ``definition`` from the summary.

    The noticed-and-refused count comes from the identifiability block's per-unit entry for the named
    definition, never from a stratum row's alias field, so a decode table cannot carry the outcome
    group's count beside a judge-disjoint disposition label.
    """
    definitions = cast("dict[str, Any] | None", summary["identifiability"].get("definitions"))
    if definitions is None or definition not in definitions:
        raise KeyError(
            f"the narration summary carries no identifiability definition {definition!r}; "
            f"it has {sorted(definitions or {})}"
        )
    by_unit = cast("dict[str, dict[str, Any]]", definitions[definition]["by_unit"])
    if unit not in by_unit:
        raise KeyError(
            f"no {definition!r} identifiability entry for unit {unit!r}; units: {sorted(by_unit)}"
        )
    for row in cast("list[dict[str, Any]]", summary["strata"]):
        if row.get("unit") == unit and row.get("stratum") == ALL_STRATA:
            return ResolvedCounts(
                noticed=int(row["noticed"]),
                complied=int(row["complied"]),
                noticed_and_refused=int(by_unit[unit]["rows"]),
                source=source,
                definition=definition,
            )
    raise KeyError(f"no {ALL_STRATA} row for unit {unit!r} in the narration summary")


def decode_all(
    families: Mapping[str, Mapping[int, torch.Tensor]],
    context: DecodeContext,
    *,
    strata: Mapping[str, str] | None = None,
    seed: int = 0,
) -> list[DecodeRead]:
    """Decode several named families through one lens; ``strata`` maps a family name to its stratum."""
    reads: list[DecodeRead] = []
    for name, by_layer in families.items():
        stratum = STRATUM_ALL if strata is None else strata.get(name, STRATUM_ALL)
        reads.extend(decode_family(by_layer, context, name=name, stratum=stratum, seed=seed))
    return reads


# --------------------------------------------------------------------------------------
# From a lens directory and a captured cell to a decode context, fingerprints held
# --------------------------------------------------------------------------------------


def decode_context_for(  # noqa: PLR0913 - a context is a bundle, a cell, the tokenizer and the knobs
    bundle: LensBundle,
    *,
    cell: CapturedCell,
    stimulus_set: str,
    pooling: str,
    id_to_token: Callable[[int], str],
    top_k: int,
    n_placebos: int,
    disposition: IdentifiabilityGate,
    resolved: ResolvedCounts | None = None,
    gate_verdicts: GateVerdictCache | None = None,
) -> DecodeContext:
    """Build the context for decoding directions fitted on ``cell`` through ``bundle``'s lens.

    The gate is held to the cell's served weights: anchors recorded on one checkpoint gate residuals
    from that checkpoint only. The standardising scale is the per-dimension std of the cell's own
    activations at the named set and pooling, so raw and standardized reads share one coordinate frame.
    """
    assert_gate_matches_weights(
        bundle.gate,
        cell_fingerprint(cell),
        what=f"cell {cell.label} ({stimulus_set}/{pooling})",
    )
    return DecodeContext(
        lens=bundle.lens,
        model=bundle.unembed,
        lens_name=bundle.lens_name,
        id_to_token=id_to_token,
        top_k=top_k,
        n_placebos=n_placebos,
        gate=bundle.gate,
        disposition=disposition,
        fitted_layers=frozenset(bundle.lens.source_layers),
        scale_by_layer=scale_by_layer_from_matrix(cell.matrix(stimulus_set, pooling)),
        resolved=resolved,
        gate_verdicts=GateVerdictCache() if gate_verdicts is None else gate_verdicts,
    )


def families_by_pooling(
    directions: Mapping[DirectionKey, torch.Tensor],
) -> dict[str, dict[str, dict[int, torch.Tensor]]]:
    """Regroup ``{(name, pooling, stratum, layer): v}`` as ``{pooling: {"name|stratum": {layer: v}}}``."""
    grouped: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    for (name, pooling, stratum, layer), tensor in directions.items():
        grouped.setdefault(pooling, {}).setdefault(f"{name}|{stratum}", {})[layer] = tensor
    return grouped


def family_strata(families: Mapping[str, Mapping[int, torch.Tensor]]) -> dict[str, str]:
    """Recover each ``name|stratum`` family's stratum for the decode rows."""
    return {family: family.split("|", 1)[1] for family in families}


def tokenizer_id_to_token(tokenizer_source: str, revision: str | None) -> Callable[[int], str]:
    """Return the decode's id-to-token map from the tokenizer of record (CPU, tokenizer files only)."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, revision=revision, trust_remote_code=True
    )  # pyright: ignore[reportUnknownMemberType]

    def id_to_token(token_id: int) -> str:
        return cast("str", tokenizer.convert_ids_to_tokens(token_id))

    return id_to_token


def read_directions_through(  # noqa: PLR0913 - one stage is its inputs, its knobs and its output
    bundle: LensBundle,
    *,
    cell: CapturedCell,
    stimulus_set: str,
    directions: Mapping[DirectionKey, torch.Tensor],
    transfer_bundles: Sequence[LensBundle],
    id_to_token: Callable[[int], str],
    top_k: int,
    n_placebos: int,
    disposition: IdentifiabilityGate,
    resolved: ResolvedCounts | None,
    seed: int,
) -> LensReadBundle:
    """Decode every family through the cell's own lens, then transfer each through the other lenses.

    Families are grouped by pooling because the standardising scale is per pooling; a pooling the cell
    did not capture decodes raw only and says so. Transfers read a direction through the reference
    lens and each other lens at every readable layer, against the placebo and wrong-layer floors.
    """
    out = LensReadBundle()
    poolings = set(cell.poolings)
    verdicts = GateVerdictCache()
    for pooling, families in families_by_pooling(directions).items():
        if pooling in poolings:
            context = decode_context_for(
                bundle,
                cell=cell,
                stimulus_set=stimulus_set,
                pooling=pooling,
                id_to_token=id_to_token,
                top_k=top_k,
                n_placebos=n_placebos,
                disposition=disposition,
                resolved=resolved,
                gate_verdicts=verdicts,
            )
        else:
            logger.warning(
                f"cell {cell.label} captured no {pooling!r} pooling; {len(families)} families decode "
                f"raw only, without a standardized row"
            )
            assert_gate_matches_weights(
                bundle.gate, cell_fingerprint(cell), what=f"cell {cell.label}"
            )
            context = DecodeContext(
                lens=bundle.lens,
                model=bundle.unembed,
                lens_name=bundle.lens_name,
                id_to_token=id_to_token,
                top_k=top_k,
                n_placebos=n_placebos,
                gate=bundle.gate,
                disposition=disposition,
                fitted_layers=frozenset(bundle.lens.source_layers),
                resolved=resolved,
                gate_verdicts=verdicts,
            )
        out.decodes.extend(decode_all(families, context, strata=family_strata(families), seed=seed))
        reference = LensPair(name=bundle.lens_name, lens=bundle.lens, model=bundle.unembed)
        transfers_refused = 0
        for other_bundle in transfer_bundles:
            other = LensPair(
                name=other_bundle.lens_name, lens=other_bundle.lens, model=other_bundle.unembed
            )
            for family, by_layer in families.items():
                for layer in sorted(by_layer):
                    if layer < MIN_DECODE_LAYER or layer not in context.fitted_layers:
                        continue
                    if not verdicts.verdict(context, layer).admitted:
                        transfers_refused += 1
                        continue
                    out.transfers.append(
                        lens_transfer_read(
                            by_layer[layer],
                            name=f"{family}|{pooling}",
                            layer=layer,
                            reference=reference,
                            other=other,
                            id_to_token=id_to_token,
                            top_k=top_k,
                            n_placebos=n_placebos,
                            wrong_layer=wrong_layer_for(
                                layer, n_layers=bundle.gate.n_layers, offset=8
                            ),
                            disposition=disposition,
                            seed=seed,
                        )
                    )
        if transfers_refused:
            logger.warning(
                f"{transfers_refused} transfer reads at pooling {pooling!r} skipped at layers the "
                f"reference lens's gate refused; an overlap of two refused decodes is not a transfer"
            )
    logger.info(f"gate verdicts computed once per (lens, layer): {len(verdicts.verdicts)}")
    return out


def build_parser() -> argparse.ArgumentParser:
    """CLI for the reads stage: one directions file, the cell it came from, the lenses to read through."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lens-dir",
        type=Path,
        required=True,
        help="the tmax_lens_fit directory of the checkpoint the directions were captured on",
    )
    parser.add_argument(
        "--transfer-lens-dir",
        type=Path,
        action="append",
        default=[],
        help="another checkpoint's lens directory to read the same directions through (repeatable)",
    )
    parser.add_argument(
        "--directions",
        type=Path,
        required=True,
        help="a directions.pt written by tmax_directions.DirectionSet.save",
    )
    parser.add_argument(
        "--cell",
        type=Path,
        required=True,
        help="the captured cell directory the directions were fitted on",
    )
    parser.add_argument(
        "--stimulus-set", required=True, help="the set whose activations standardise"
    )
    parser.add_argument(
        "--narration-summary",
        type=Path,
        default=None,
        help="the narration judge's summary JSON for the identifiability gate; absent means unlabelled",
    )
    parser.add_argument(
        "--identifiability-definition",
        choices=IDENTIFIABILITY_DEFINITIONS,
        default=None,
        help=(
            "which noticed-and-refused group the gate reads out of --narration-summary; required "
            "with it, because the outcome and judge-disjoint groups disagree on real data"
        ),
    )
    parser.add_argument(
        "--resolved-unit",
        default=None,
        help=(
            "unit name whose noticed / complied counts and gate-group rows (under "
            "--identifiability-definition) travel on every row"
        ),
    )
    parser.add_argument(
        "--tokenizer", required=True, help="the tokenizer of record for token strings"
    )
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--n-placebos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="under the run prefix or artifacts/, never tracked",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    """Open the lenses and the cell, decode, transfer, and write the three tables."""
    out_dir = refuse_tracked_path(cast("Path", args.out_dir))
    bundle = load_lens_bundle(cast("Path", args.lens_dir))
    transfer_bundles = [
        load_lens_bundle(path) for path in cast("list[Path]", args.transfer_lens_dir)
    ]
    cell = read_cell(cast("Path", args.cell))
    directions = load_directions(cast("Path", args.directions))
    definition = cast("str | None", args.identifiability_definition)
    if (args.narration_summary is None) != (definition is None):
        raise ValueError(
            "--narration-summary and --identifiability-definition go together: the gate reads one "
            f"named group ({', '.join(IDENTIFIABILITY_DEFINITIONS)}) out of the summary, never a "
            "default"
        )
    disposition = (
        IdentifiabilityGate.unlabelled()
        if args.narration_summary is None or definition is None
        else IdentifiabilityGate.from_narration_summary(
            cast("Path", args.narration_summary), definition=definition
        )
    )
    resolved = None
    if args.resolved_unit is not None:
        if args.narration_summary is None or definition is None:
            raise ValueError("--resolved-unit needs --narration-summary")
        summary = cast(
            "dict[str, Any]", json.loads(cast("Path", args.narration_summary).read_text())
        )
        resolved = resolved_counts_from_summary(
            summary,
            unit=cast("str", args.resolved_unit),
            source=str(args.narration_summary),
            definition=definition,
        )
    reads = read_directions_through(
        bundle,
        cell=cell,
        stimulus_set=cast("str", args.stimulus_set),
        directions=directions,
        transfer_bundles=transfer_bundles,
        id_to_token=tokenizer_id_to_token(
            cast("str", args.tokenizer), cast("str | None", args.tokenizer_revision)
        ),
        top_k=cast("int", args.top_k),
        n_placebos=cast("int", args.n_placebos),
        disposition=disposition,
        resolved=resolved,
        seed=cast("int", args.seed),
    )
    written = reads.save(out_dir)
    admitted = sum(1 for read in reads.decodes if read.gate_admitted)
    logger.info(
        f"reads stage done, lens={bundle.lens_name} cell={cell.label} families="
        f"{len({(r.name, r.stratum) for r in reads.decodes})} decodes={len(reads.decodes)} "
        f"gate_admitted={admitted} transfers={len(reads.transfers)} {disposition.disposition_label=}"
    )
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reads stage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
