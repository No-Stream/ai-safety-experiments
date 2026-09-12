"""CPU-only geometry for the cooperation construct capture.

The construct corpus is authored as two matched stimulus sets.  Each pair carries an explicit
``metadata.split`` and ``metadata.scenario_group``; this module refuses to infer either from row
order.  Directions are fit on the fit pairs only.  Held-out pairs are then projected onto the
fixed direction, a side-shuffled direction, and a matched-norm random direction.

The input is a cached :mod:`games.interp_cells` ladder containing exactly one base and one final
cell.  No model, tokenizer, CUDA operation, or network access is used here.  The output direction
files deliberately use the ``{layer: tensor}`` shape consumed by ``games.cooperation_lens``.
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
from safetensors.torch import load_file

from games.cooperation_lens import (
    DIRECTION_COSTLY_OTHER_REGARD,
    DIRECTION_DECISION_DEPENDENCE,
    DIRECTION_TRAINED_DISPLACEMENT,
    REQUIRED_DIRECTIONS,
    STATE_BASE,
    STATE_FINAL,
    STATES,
    load_reserved_identities,
)
from games.interp_axes import ConstructSplit, ProvenanceError, build_construct_splits
from games.interp_cells import (
    BASE_ARM,
    BASE_STEP,
    NATURAL_PREFIX_ACTIVATIONS_FILENAME,
    NATURAL_PREFIX_MANIFEST_FILENAME,
    CapturedCell,
    CellFormatError,
    Ladder,
    PairLayout,
    Stimulus,
    concept_activations,
    load_ladder,
    load_private_construct_stimuli,
    pair_layout,
    sha256_of_file,
    stimuli_digest,
)
from games.interp_displacement import RowGroup, displacement_vectors
from games.interp_trajectory import GroupKey, projection_gap
from reward_hacking.interp.directions import cosine, matched_norm_random_direction, unit
from reward_hacking.interp.eval_awareness_probe import (
    direction_separation,
    direction_split_half_cosine,
)
from reward_hacking.interp.linear_probe import ConceptActivations, ProbeConfig

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger("games.cooperation_interp")

REPORT_FILENAME = "cooperation_interp.json"
DIRECTIONS_DIRNAME = "directions"
DIRECTION_MANIFEST_FILENAME = "direction-manifest.json"
SELECTION_FILENAME = "selected-target.json"

DEFAULT_FINAL_ARM = "cooperation-generalization-care-alpha-1"
DEFAULT_FINAL_STEP = 20
DEFAULT_POOLING = "last"
DEFAULT_N_PLACEBOS = 20
DEFAULT_N_FOLDS = 2
DEFAULT_SEED = 0
NATURAL_ACTIVATION_NDIMS = 3
PRIVATE_EXPECTATION_ROOTS = (Path("docs/scratch"), Path("artifacts"))

_CONSTRUCTS = (DIRECTION_COSTLY_OTHER_REGARD, DIRECTION_DECISION_DEPENDENCE)


class CooperationInterpError(ValueError):
    """The cached cooperation geometry cannot be read without violating its contract."""


@dataclass(frozen=True)
class ProjectionRead:
    """One construct direction's fit-only calibration and held-out projection read."""

    state: str
    construct: str
    pooling: str
    layer: int
    fit_pair_ids: tuple[str, ...]
    heldout_pair_ids: tuple[str, ...]
    fit_group_ids: tuple[str, ...]
    heldout_group_ids: tuple[str, ...]
    fit_direction_norm: float
    fit_direction_accuracy: float
    fit_placebo_accuracy_mean: float
    fit_placebo_accuracy_max: float
    fit_accuracy_empirical_p: float
    fit_split_half_cosine: float
    heldout_positive_projection: float
    heldout_negative_projection: float
    heldout_projection_gap: float
    shuffled_positive_projection: float
    shuffled_negative_projection: float
    shuffled_projection_gap: float
    matched_norm_random_positive_projection: float
    matched_norm_random_negative_projection: float
    matched_norm_random_projection_gap: float
    shuffled_direction_norm: float
    matched_norm_random_direction_norm: float
    grouped_projection_gaps: dict[str, dict[str, float]]


@dataclass(frozen=True)
class DisplacementRead:
    """Final-minus-base population displacement and its alignment to a fit-only axis."""

    construct: str
    pooling: str
    layer: int
    pair_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    displacement_norm: float
    displacement_projection: float
    displacement_residual_norm: float
    base_final_direction_cosine: float
    displacement_base_direction_cosine: float
    displacement_final_direction_cosine: float
    base_axis_norm: float
    final_axis_norm: float


@dataclass(frozen=True)
class CooperationInterpResult:
    """Arithmetic results and tensors before they are written to disk."""

    splits: dict[str, ConstructSplit]
    projections: tuple[ProjectionRead, ...]
    displacements: tuple[DisplacementRead, ...]
    directions: dict[str, dict[str, dict[int, torch.Tensor]]]
    displacement_directions: dict[int, torch.Tensor]
    confound_checks: dict[str, Any]
    calibration_selection: dict[str, Any]
    export_pooling: str


def _pair_indices(layout: PairLayout, pair_ids: Sequence[str], *, role: str) -> torch.Tensor:
    """Convert authored pair IDs into the capture's checked row order."""
    positions = {pair_id: index for index, pair_id in enumerate(layout.pair_ids)}
    unknown = sorted(set(pair_ids) - set(positions))
    if unknown:
        raise CooperationInterpError(
            f"{role} split names unknown pairs for set capture: {unknown[:5]}"
        )
    if len(set(pair_ids)) != len(pair_ids):
        raise CooperationInterpError(f"{role} split repeats a pair id")
    return torch.tensor([positions[pair_id] for pair_id in pair_ids], dtype=torch.long)


def _project(  # noqa: PLR0913 - a projection names the cell, construct, pooling, layer, layout and split
    cell: CapturedCell,
    *,
    construct: str,
    pooling: str,
    layer: int,
    layout: PairLayout,
    direction: torch.Tensor,
    pairs: torch.Tensor,
) -> tuple[float, float, float]:
    """Project held-out sides through the existing trajectory projection helper."""
    key = GroupKey(arm=cell.arm, stimulus_set=construct, pooling=pooling)
    positive, negative = projection_gap(cell, key, layer, layout, direction, pairs=pairs)
    return positive, negative, positive - negative


def _shuffled_direction(concept: ConceptActivations, *, generator: torch.Generator) -> torch.Tensor:
    """Fit a direction after independently flipping matched-pair side labels."""
    flip = torch.rand(concept.n_pairs, generator=generator) < 0.5  # noqa: PLR2004 - fair coin
    return ConceptActivations(
        torch.where(flip[:, None], concept.negatives, concept.positives),
        torch.where(flip[:, None], concept.positives, concept.negatives),
    ).diff_of_means()


def _construct_row_group(
    construct: str, layout: PairLayout, pair_indices: torch.Tensor
) -> RowGroup:
    """Build a pair-balanced selection understood by ``displacement_vectors``."""
    positive = layout.positive_rows[pair_indices]
    negative = layout.negative_rows[pair_indices]
    rows = torch.stack((positive, negative), dim=1).reshape(-1)
    n_pairs = int(pair_indices.numel())
    pair_ranks = torch.arange(n_pairs, dtype=torch.long)
    even_pairs = pair_ranks[pair_ranks % 2 == 0]
    odd_pairs = pair_ranks[pair_ranks % 2 == 1]
    return RowGroup(
        set_group=construct,
        stimulus_sets=(construct,),
        stratification="construct-split",
        stratum="heldout",
        rows={construct: rows},
        pair_even_positions=torch.stack((2 * even_pairs, 2 * even_pairs + 1), dim=1).reshape(-1),
        pair_odd_positions=torch.stack((2 * odd_pairs, 2 * odd_pairs + 1), dim=1).reshape(-1),
        n_rows=2 * n_pairs,
        n_pairs=n_pairs,
        row_parity_equals_side=False,
    )


def _read_confound_checks(  # noqa: C901, PLR0912, PLR0915 - explicit confound audit
    stimuli: Sequence[Stimulus],
    splits: Mapping[str, ConstructSplit],
    projections: Sequence[ProjectionRead],
) -> dict[str, Any]:
    """Report actual held-out confound reads without silently inventing strata."""
    by_pair: dict[str, list[Stimulus]] = {}
    for stimulus in stimuli:
        by_pair.setdefault(stimulus.pair_id, []).append(stimulus)

    story_field = next(
        (
            field
            for field in ("story_group", "scenario_group")
            if any(field in s.metadata for s in stimuli)
        ),
        None,
    )
    story_payload: dict[str, Any] = {"supported": story_field is not None, "field": story_field}
    if story_field is None:
        story_payload["reason"] = "stimulus metadata has no story_group or scenario_group field"
    else:
        values: dict[str, str] = {}
        for pair_id, members in sorted(by_pair.items()):
            pair_values = {
                str(member.metadata[story_field])
                for member in members
                if story_field in member.metadata
            }
            if len(pair_values) != 1:
                raise ProvenanceError(
                    f"pair {pair_id!r} has inconsistent {story_field} metadata: {sorted(pair_values)}"
                )
            values[pair_id] = pair_values.pop()
        story_payload["groups_by_construct"] = {
            construct: {
                "fit": sorted({values[pair_id] for pair_id in split.fit_pair_ids}),
                "heldout": sorted({values[pair_id] for pair_id in split.heldout_pair_ids}),
            }
            for construct, split in sorted(splits.items())
        }
        story_payload["heldout_projection_separation"] = [
            {
                "state": read.state,
                "construct": read.construct,
                "pooling": read.pooling,
                "layer": read.layer,
                "strata": {
                    key.removeprefix(f"{story_field}="): values
                    for key, values in sorted(read.grouped_projection_gaps.items())
                    if key.startswith(f"{story_field}=")
                },
            }
            for read in projections
            if read.grouped_projection_gaps
            and any(key.startswith(f"{story_field}=") for key in read.grouped_projection_gaps)
        ]

    position_field = next(
        (
            field
            for field in ("printed_position", "label_print_order")
            if any(field in stimulus.metadata for stimulus in stimuli)
        ),
        None,
    )
    position_payload: dict[str, Any] = {
        "supported": position_field is not None,
        "field": position_field,
    }
    if position_field is None:
        position_payload["reason"] = (
            "stimulus metadata has no printed_position or label_print_order field; "
            "printed-position confounding is untestable for this capture"
        )
    else:
        by_position: dict[str, list[str]] = {}
        for pair_id, members in sorted(by_pair.items()):
            pair_values = {
                str(member.metadata[position_field])
                for member in members
                if position_field in member.metadata
            }
            if len(pair_values) != 1:
                raise ProvenanceError(
                    f"pair {pair_id!r} has inconsistent {position_field} metadata: {sorted(pair_values)}"
                )
            by_position.setdefault(pair_values.pop(), []).append(pair_id)
        position_payload["pair_ids_by_position"] = {
            position: sorted(pair_ids) for position, pair_ids in sorted(by_position.items())
        }
        position_payload["heldout_projection_separation"] = [
            {
                "state": read.state,
                "construct": read.construct,
                "pooling": read.pooling,
                "layer": read.layer,
                "strata": {
                    key.removeprefix(f"{position_field}="): values
                    for key, values in sorted(read.grouped_projection_gaps.items())
                    if key.startswith(f"{position_field}=")
                },
            }
            for read in projections
            if read.grouped_projection_gaps
            and any(key.startswith(f"{position_field}=") for key in read.grouped_projection_gaps)
        ]

    procedure_field = "procedure_regime"
    procedure_supported = any(procedure_field in stimulus.metadata for stimulus in stimuli)
    procedure_payload: dict[str, Any] = {"supported": procedure_supported}
    if procedure_supported:
        counts_by_construct: dict[str, dict[str, dict[str, int]]] = {}
        for construct, split in sorted(splits.items()):
            construct_pair_ids = (*split.fit_pair_ids, *split.heldout_pair_ids)
            if not any(
                procedure_field in member.metadata
                for pair_id in construct_pair_ids
                for member in by_pair[pair_id]
            ):
                continue
            counts_by_construct[construct] = {}
            for split_name, pair_ids in (
                ("fit", split.fit_pair_ids),
                ("heldout", split.heldout_pair_ids),
            ):
                counts: dict[str, int] = {}
                for pair_id in pair_ids:
                    members = by_pair[pair_id]
                    regimes = {
                        str(member.metadata[procedure_field])
                        for member in members
                        if procedure_field in member.metadata
                    }
                    if len(regimes) != 1:
                        raise ProvenanceError(
                            f"pair {pair_id!r} has incomplete or inconsistent "
                            f"{procedure_field} metadata"
                        )
                    regime = regimes.pop()
                    counts[regime] = counts.get(regime, 0) + 1
                counts_by_construct[construct][split_name] = dict(sorted(counts.items()))
        procedure_payload["pair_counts_by_construct_and_split"] = counts_by_construct
        procedure_payload["heldout_projection_separation"] = [
            {
                "state": read.state,
                "construct": read.construct,
                "pooling": read.pooling,
                "layer": read.layer,
                "strata": {
                    key.removeprefix(f"{procedure_field}="): values
                    for key, values in sorted(read.grouped_projection_gaps.items())
                    if key.startswith(f"{procedure_field}=")
                },
            }
            for read in projections
            if any(key.startswith(f"{procedure_field}=") for key in read.grouped_projection_gaps)
        ]
        procedure_payload["limitation"] = (
            "Within each procedure regime, the decision-dependence contrast necessarily changes "
            "shared versus independent procedure, seed, and call structure; regime-stratified "
            "separation does not remove that residual confounding."
        )
    else:
        procedure_payload["reason"] = "stimulus metadata has no procedure_regime field"

    invalid_boundaries = [
        stimulus.stimulus_id
        for stimulus in stimuli
        if stimulus.metadata.get("measurement_boundary") != "pre_action"
        or stimulus.metadata.get("action_commitment_present") is not False
    ]
    if invalid_boundaries:
        raise ProvenanceError(
            "construct rows must all record measurement_boundary='pre_action' and "
            f"action_commitment_present=false; invalid rows {invalid_boundaries[:5]}"
        )
    action_payload: dict[str, Any] = {
        "supported": True,
        "status": "validated-pre-action",
        "measurement_boundary": "pre_action",
        "action_commitment_present": False,
        "n_rows_validated": len(stimuli),
        "basis": "explicit metadata on every captured row",
    }
    return {
        "story_group": story_payload,
        "printed_position": position_payload,
        "procedure_regime": procedure_payload,
        "action_token_commitment": action_payload,
    }


def _metadata_pair_strata(
    stimuli: Sequence[Stimulus],
    split: ConstructSplit,
    *,
    fields: Sequence[str],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return complete held-out pair strata for metadata fields used in confound reads."""
    by_pair: dict[str, list[Stimulus]] = {}
    for stimulus in stimuli:
        if stimulus.stimulus_set == split.construct:
            by_pair.setdefault(stimulus.pair_id, []).append(stimulus)
    strata: dict[str, dict[str, list[str]]] = {field: {} for field in fields}
    for field in fields:
        if not any(
            field in stimulus.metadata for members in by_pair.values() for stimulus in members
        ):
            continue
        for pair_id in split.heldout_pair_ids:
            members = by_pair[pair_id]
            values = {str(member.metadata[field]) for member in members if field in member.metadata}
            if len(values) != 1:
                raise ProvenanceError(
                    f"pair {pair_id!r} has incomplete or inconsistent {field} metadata"
                )
            value = values.pop()
            strata[field].setdefault(value, []).append(pair_id)
    return {
        field: {value: tuple(pair_ids) for value, pair_ids in sorted(by_value.items())}
        for field, by_value in strata.items()
        if by_value
    }


def _fit_read(  # noqa: PLR0913 - one read names state, construct, split, layer and control knobs
    *,
    state: str,
    cell: CapturedCell,
    construct: str,
    pooling: str,
    layer: int,
    layout: PairLayout,
    split: ConstructSplit,
    fit_indices: torch.Tensor,
    heldout_indices: torch.Tensor,
    config: ProbeConfig,
    n_placebos: int,
    seed: int,
    heldout_strata: Mapping[str, Mapping[str, Sequence[str]]],
) -> tuple[ProjectionRead, torch.Tensor]:
    """Fit one axis on fit pairs and score all controls on held-out pairs."""
    fit_concept = concept_activations(cell, construct, pooling, layer, layout, pairs=fit_indices)
    direction = fit_concept.diff_of_means()
    if float(direction.norm()) == 0.0:
        raise CooperationInterpError(
            f"{state}/{construct}/{pooling}/layer-{layer} has a zero fit-only direction"
        )
    separation = direction_separation(fit_concept, config, n_placebos=n_placebos)
    fit_split_half = direction_split_half_cosine(fit_concept)
    control_generator = torch.Generator().manual_seed(
        seed + sum(ord(char) for char in f"{state}|{construct}|{pooling}|{layer}")
    )
    shuffled = _shuffled_direction(fit_concept, generator=control_generator)
    random_direction = matched_norm_random_direction(direction, control_generator)
    heldout_positive, heldout_negative, heldout_gap = _project(
        cell,
        construct=construct,
        pooling=pooling,
        layer=layer,
        layout=layout,
        direction=direction,
        pairs=heldout_indices,
    )
    shuffled_positive, shuffled_negative, shuffled_gap = _project(
        cell,
        construct=construct,
        pooling=pooling,
        layer=layer,
        layout=layout,
        direction=shuffled,
        pairs=heldout_indices,
    )
    random_positive, random_negative, random_gap = _project(
        cell,
        construct=construct,
        pooling=pooling,
        layer=layer,
        layout=layout,
        direction=random_direction,
        pairs=heldout_indices,
    )
    grouped_projection_gaps: dict[str, dict[str, float]] = {}
    for field, strata in sorted(heldout_strata.items()):
        for value, pair_ids in sorted(strata.items()):
            stratum_indices = _pair_indices(layout, pair_ids, role=f"{field}={value}")
            real_read = _project(
                cell,
                construct=construct,
                pooling=pooling,
                layer=layer,
                layout=layout,
                direction=direction,
                pairs=stratum_indices,
            )
            shuffled_read = _project(
                cell,
                construct=construct,
                pooling=pooling,
                layer=layer,
                layout=layout,
                direction=shuffled,
                pairs=stratum_indices,
            )
            random_read = _project(
                cell,
                construct=construct,
                pooling=pooling,
                layer=layer,
                layout=layout,
                direction=random_direction,
                pairs=stratum_indices,
            )
            grouped_projection_gaps[f"{field}={value}"] = {
                "real": real_read[2],
                "shuffled": shuffled_read[2],
                "matched_norm_random": random_read[2],
            }
    return (
        ProjectionRead(
            state=state,
            construct=construct,
            pooling=pooling,
            layer=layer,
            fit_pair_ids=split.fit_pair_ids,
            heldout_pair_ids=split.heldout_pair_ids,
            fit_group_ids=split.fit_groups,
            heldout_group_ids=split.heldout_groups,
            fit_direction_norm=float(direction.norm()),
            fit_direction_accuracy=separation.direction_accuracy,
            fit_placebo_accuracy_mean=separation.placebo_accuracy_mean,
            fit_placebo_accuracy_max=separation.placebo_accuracy_max,
            fit_accuracy_empirical_p=separation.accuracy_empirical_p,
            fit_split_half_cosine=fit_split_half,
            heldout_positive_projection=heldout_positive,
            heldout_negative_projection=heldout_negative,
            heldout_projection_gap=heldout_gap,
            shuffled_positive_projection=shuffled_positive,
            shuffled_negative_projection=shuffled_negative,
            shuffled_projection_gap=shuffled_gap,
            matched_norm_random_positive_projection=random_positive,
            matched_norm_random_negative_projection=random_negative,
            matched_norm_random_projection_gap=random_gap,
            shuffled_direction_norm=float(shuffled.norm()),
            matched_norm_random_direction_norm=float(random_direction.norm()),
            grouped_projection_gaps=grouped_projection_gaps,
        ),
        direction,
    )


def _read_displacement(  # noqa: PLR0913 - one read names both cells, split, layer, axis and layout
    base: CapturedCell,
    final: CapturedCell,
    *,
    construct: str,
    pooling: str,
    layer: int,
    layout: PairLayout,
    split: ConstructSplit,
    base_direction: torch.Tensor,
    final_direction: torch.Tensor,
) -> tuple[DisplacementRead, torch.Tensor]:
    """Read held-out final-minus-base displacement through the shared helper."""
    heldout_indices = _pair_indices(layout, split.heldout_pair_ids, role="held-out")
    group = _construct_row_group(construct, layout, heldout_indices)
    base_rows = {construct: base.matrix(construct, pooling)}
    final_rows = {construct: final.matrix(construct, pooling)}
    vectors = displacement_vectors(final_rows, base_rows, group)
    displacement = vectors.full[layer]
    axis_unit = unit(final_direction)
    projection = float(displacement @ axis_unit)
    residual = displacement - projection * axis_unit
    return (
        DisplacementRead(
            construct=construct,
            pooling=pooling,
            layer=layer,
            pair_ids=split.heldout_pair_ids,
            group_ids=split.heldout_groups,
            displacement_norm=float(displacement.norm()),
            displacement_projection=projection,
            displacement_residual_norm=float(residual.norm()),
            base_final_direction_cosine=cosine(base_direction, final_direction),
            displacement_base_direction_cosine=cosine(displacement, base_direction),
            displacement_final_direction_cosine=cosine(displacement, final_direction),
            base_axis_norm=float(base_direction.norm()),
            final_axis_norm=float(final_direction.norm()),
        ),
        displacement,
    )


@dataclass(frozen=True)
class NaturalPrefixArtifact:
    """A validated reduced natural-prefix capture for one checkpoint state."""

    state: str
    path: Path
    layers: tuple[int, ...]
    activations: dict[str, torch.Tensor]
    selection_records: tuple[dict[str, Any], ...]
    selection_manifest: dict[str, Any]
    natural_stimuli_sha256: str
    rendered_sha256: str
    tokenizer_identity: str
    kernel_identity: str
    capture_identity: dict[str, Any]


def _natural_manifest_path(path: Path) -> Path:
    """Resolve a natural-prefix directory, manifest, or nested cell artifact."""
    manifest_path = path / NATURAL_PREFIX_MANIFEST_FILENAME if path.is_dir() else path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"natural-prefix artifact is missing: {manifest_path}")
    return manifest_path


def load_natural_prefix_artifact(  # noqa: C901, PLR0912, PLR0915 - validates a durable artifact field by field
    path: Path, *, state: str, cell: CapturedCell
) -> NaturalPrefixArtifact:
    """Load a standalone natural-prefix artifact and bind its shape to one cached cell."""
    if state not in STATES:
        raise ValueError(f"natural-prefix state must be one of {STATES}, got {state!r}")
    manifest_path = _natural_manifest_path(path)
    payload = cast("dict[str, Any]", json.loads(manifest_path.read_text()))
    required = {
        "identity",
        "stimuli_sha256",
        "stimulus_ids",
        "selection_manifest",
        "selection_records",
        "states",
    }
    required |= {
        "arm",
        "step",
        "applied_adapter_weights",
        "adapter_weights_sha256",
        "rendered_sha256",
        "natural_state",
    }
    if set(payload) != required:
        raise CooperationInterpError(
            f"{manifest_path} must contain natural-prefix fields {sorted(required)}"
        )
    artifact_state = payload["natural_state"]
    if artifact_state != state:
        raise CooperationInterpError(
            f"{manifest_path} records natural_state={artifact_state!r}, but was supplied as "
            f"state={state!r}; base/final natural artifacts may not be swapped"
        )
    identity = cast("dict[str, Any]", payload["identity"])
    shared_identity_fields = (
        "base_model",
        "stimuli_sha256",
        "layer_convention",
        "n_layers",
        "hidden_size",
        "batch_size",
        "compute_dtype",
        "store_dtype",
        "stimulus_render",
        "tokenizer_identity",
        "kernel_identity",
    )
    required_identity = (*shared_identity_fields, "natural_prefix_layers")
    missing_identity = [field for field in required_identity if field not in identity]
    if missing_identity:
        raise CooperationInterpError(
            f"{manifest_path} natural-prefix identity omits {missing_identity}"
        )
    mismatches = {
        field: (identity[field], getattr(cell.identity, field))
        for field in shared_identity_fields
        if identity[field] != getattr(cell.identity, field)
    }
    natural_stimuli_sha256 = str(payload["stimuli_sha256"])
    if not natural_stimuli_sha256:
        raise CooperationInterpError(f"{manifest_path} records an empty natural corpus digest")
    rendered_sha256 = payload["rendered_sha256"]
    if not isinstance(rendered_sha256, str) or not rendered_sha256:
        raise CooperationInterpError(f"{manifest_path} records no rendered natural corpus digest")
    identity_natural_stimuli_sha256 = identity.get("natural_prefix_stimuli_sha256")
    if identity_natural_stimuli_sha256 not in (None, "", natural_stimuli_sha256):
        raise CooperationInterpError(
            f"{manifest_path} natural corpus digest disagrees with its identity"
        )
    identity_natural_rendered_sha256 = identity.get("natural_prefix_rendered_sha256")
    if identity_natural_rendered_sha256 not in (None, "", rendered_sha256):
        raise CooperationInterpError(
            f"{manifest_path} rendered natural corpus digest disagrees with its identity"
        )
    layers = tuple(int(layer) for layer in identity["natural_prefix_layers"])
    if not layers or len(set(layers)) != len(layers):
        raise CooperationInterpError(
            f"{manifest_path} records an empty or repeated natural layer subset"
        )
    invalid_layers = sorted(set(layers) - set(range(cell.identity.n_layers)))
    if invalid_layers:
        raise CooperationInterpError(
            f"{manifest_path} natural-prefix layers are outside the cached model: {invalid_layers}"
        )
    if mismatches:
        raise CooperationInterpError(
            f"{manifest_path} natural-prefix identity does not match {cell.label}: {mismatches}"
        )
    stimulus_ids = tuple(str(value) for value in payload["stimulus_ids"])
    if not stimulus_ids or len(set(stimulus_ids)) != len(stimulus_ids):
        raise CooperationInterpError(f"{manifest_path} repeats or omits natural stimulus ids")
    raw_records = payload["selection_records"]
    if not isinstance(raw_records, list):
        raise CooperationInterpError(f"{manifest_path} selection_records is not a list")
    records = tuple(cast("dict[str, Any]", value) for value in raw_records)
    records_by_id = {str(record.get("stimulus_id")): record for record in records}
    if set(records_by_id) != set(stimulus_ids) or len(records_by_id) != len(records):
        raise CooperationInterpError(
            f"{manifest_path} selection records do not cover exactly its stimulus ids"
        )
    raw_selection_manifest = payload["selection_manifest"]
    if not isinstance(raw_selection_manifest, dict):
        raise CooperationInterpError(f"{manifest_path} selection_manifest is not an object")
    selection_manifest = cast("dict[str, Any]", raw_selection_manifest)
    selection_digest = selection_manifest.get("sha256")
    identity_selection_digest = str(identity.get("natural_selection_manifest_sha256", ""))
    if not identity_selection_digest or selection_digest != identity_selection_digest:
        raise CooperationInterpError(
            f"{manifest_path} selection manifest digest does not match its capture identity"
        )
    request_ids = tuple(str(value) for value in selection_manifest.get("request_ids", ()))
    if not request_ids or len(set(request_ids)) != len(request_ids):
        raise CooperationInterpError(
            f"{manifest_path} selection manifest repeats or omits request_ids"
        )
    rollout_ids_by_state = selection_manifest.get("rollout_ids_by_state")
    if (
        not isinstance(rollout_ids_by_state, dict)
        or set(rollout_ids_by_state) != set(STATES)
        or any(
            not isinstance(rollout_id, str) or not rollout_id.strip()
            for rollout_id in rollout_ids_by_state.values()
        )
    ):
        raise CooperationInterpError(
            f"{manifest_path} rollout_ids_by_state must map exactly {list(STATES)} to "
            "non-empty strings"
        )
    expected_rollout_id = str(rollout_ids_by_state[state])
    scenario_groups = {str(value) for value in selection_manifest.get("scenario_groups", ())}
    if tuple(selection_manifest.get("layers", ())) != layers:
        raise CooperationInterpError(
            f"{manifest_path} selection manifest layers disagree with capture identity"
        )
    selection_rule = selection_manifest.get("selection_rule")
    if not isinstance(selection_rule, str) or not selection_rule.strip():
        raise CooperationInterpError(f"{manifest_path} selection manifest has no selection_rule")
    record_request_ids: list[str] = []
    for record in records:
        missing_record_fields = [
            field
            for field in ("request_id", "rollout_id", "scenario_group", "natural_state")
            if not isinstance(record.get(field), str) or not str(record[field]).strip()
        ]
        if missing_record_fields:
            raise CooperationInterpError(
                f"{manifest_path} natural selection record omits {missing_record_fields}"
            )
        request_id = str(record["request_id"])
        record_request_ids.append(request_id)
        if record["natural_state"] != state:
            raise CooperationInterpError(
                f"{manifest_path} record {request_id!r} has natural_state="
                f"{record['natural_state']!r}, expected {state!r}"
            )
        if str(record["rollout_id"]) != expected_rollout_id:
            raise CooperationInterpError(
                f"{manifest_path} record {request_id!r} has rollout_id="
                f"{record['rollout_id']!r}; state {state!r} requires {expected_rollout_id!r}"
            )
        if str(record["scenario_group"]) not in scenario_groups:
            raise CooperationInterpError(
                f"{manifest_path} record {request_id!r} has an unregistered scenario_group"
            )
        if record.get("selection_rule") != selection_rule:
            raise CooperationInterpError(
                f"{manifest_path} record {request_id!r} disagrees with the selection_rule"
            )
    if tuple(record_request_ids) != request_ids:
        raise CooperationInterpError(
            f"{manifest_path} selection records do not match request_ids in declared order"
        )
    capture_identity = {
        "arm": payload["arm"],
        "step": payload["step"],
        "applied_adapter_weights": payload["applied_adapter_weights"],
        "adapter_weights_sha256": payload["adapter_weights_sha256"],
    }
    expected_capture = {
        "arm": cell.arm,
        "step": cell.step,
        "applied_adapter_weights": cell.applied_adapter_weights,
        "adapter_weights_sha256": cell.adapter_weights_sha256,
    }
    if capture_identity != expected_capture:
        raise CooperationInterpError(
            f"{manifest_path} natural-prefix capture identity does not match {cell.label}: "
            f"{capture_identity} != {expected_capture}"
        )
    activation_path = manifest_path.parent / NATURAL_PREFIX_ACTIVATIONS_FILENAME
    if not activation_path.is_file():
        raise FileNotFoundError(f"natural-prefix activations are missing: {activation_path}")
    activations = {key: value.float() for key, value in load_file(str(activation_path)).items()}
    if set(activations) != set(stimulus_ids):
        raise CooperationInterpError(
            f"{activation_path} activation ids do not match {manifest_path} stimulus_ids"
        )
    expected_tail = (len(layers), cell.identity.hidden_size)
    for stimulus_id, tensor in activations.items():
        if tensor.ndim != NATURAL_ACTIVATION_NDIMS or tuple(tensor.shape[1:]) != expected_tail:
            raise CooperationInterpError(
                f"{activation_path} {stimulus_id!r} has shape {tuple(tensor.shape)}, expected "
                f"[selected_positions, {expected_tail[0]}, {expected_tail[1]}]"
            )
        positions = records_by_id[stimulus_id].get("positions")
        if not isinstance(positions, list) or len(positions) != tensor.shape[0]:
            raise CooperationInterpError(
                f"{manifest_path} positions for {stimulus_id!r} do not match its activation rows"
            )
        if any(
            isinstance(position, bool) or not isinstance(position, int) or position < 0
            for position in positions
        ):
            raise CooperationInterpError(
                f"{manifest_path} positions for {stimulus_id!r} must be non-negative integers"
            )
        if positions != sorted(set(positions)):
            raise CooperationInterpError(
                f"{manifest_path} positions for {stimulus_id!r} must be unique and increasing"
            )
        selection_rule = records_by_id[stimulus_id].get("selection_rule")
        if not isinstance(selection_rule, str) or not selection_rule.strip():
            raise CooperationInterpError(
                f"{manifest_path} selection rule for {stimulus_id!r} is missing"
            )
        state_shape = (
            payload["states"].get(stimulus_id) if isinstance(payload["states"], dict) else None
        )
        if state_shape != list(tensor.shape):
            raise CooperationInterpError(
                f"{manifest_path} states shape for {stimulus_id!r} disagrees with activations"
            )
    return NaturalPrefixArtifact(
        state=state,
        path=manifest_path.parent,
        layers=layers,
        activations=activations,
        selection_records=records,
        selection_manifest=selection_manifest,
        natural_stimuli_sha256=natural_stimuli_sha256,
        rendered_sha256=rendered_sha256,
        tokenizer_identity=str(identity["tokenizer_identity"]),
        kernel_identity=str(identity["kernel_identity"]),
        capture_identity=capture_identity,
    )


def natural_prefix_cross_check(  # noqa: C901, PLR0912, PLR0915 - explicit artifact validation and readout
    artifacts: Mapping[str, NaturalPrefixArtifact], result: CooperationInterpResult
) -> dict[str, Any]:
    """Describe selected positions, construct-axis projections, and base/final deltas."""
    if artifacts:
        stable_manifest_fields = (
            "version",
            "rollout_ids_by_state",
            "request_ids",
            "scenario_groups",
            "selection_rule",
            "layers",
        )
        for artifact in artifacts.values():
            missing_stable_fields = [
                field
                for field in stable_manifest_fields
                if field not in artifact.selection_manifest
            ]
            if missing_stable_fields:
                raise CooperationInterpError(
                    f"natural-prefix selection manifest omits stable fields {missing_stable_fields}"
                )
        selection_manifests = {
            json.dumps(
                {field: artifact.selection_manifest.get(field) for field in stable_manifest_fields},
                sort_keys=True,
                separators=(",", ":"),
            )
            for artifact in artifacts.values()
        }
        if len(selection_manifests) != 1:
            raise CooperationInterpError(
                "base and final natural-prefix captures use different selection manifests"
            )
        identities = {
            (artifact.tokenizer_identity, artifact.kernel_identity)
            for artifact in artifacts.values()
        }
        if len(identities) != 1:
            raise CooperationInterpError(
                "base and final natural-prefix captures use different tokenizer or kernel identities"
            )
        layer_subsets = {artifact.layers for artifact in artifacts.values()}
        if len(layer_subsets) != 1:
            raise CooperationInterpError(
                "base and final natural-prefix captures use different layer subsets"
            )
        selection_rules = {
            tuple(
                sorted(
                    (str(record["request_id"]), str(record["selection_rule"]))
                    for record in artifact.selection_records
                )
            )
            for artifact in artifacts.values()
        }
        if len(selection_rules) != 1:
            raise CooperationInterpError(
                "base and final natural-prefix captures use different selection rules"
            )
        capture_states = set(artifacts) - {STATE_BASE, STATE_FINAL}
        if capture_states:
            raise CooperationInterpError(
                f"natural-prefix artifacts contain unsupported states: {sorted(capture_states)}"
            )
    states: dict[str, Any] = {}
    for state, artifact in sorted(artifacts.items()):
        state_rows: list[dict[str, Any]] = []
        for stimulus_id in sorted(artifact.activations):
            tensor = artifact.activations[stimulus_id]
            mean_by_layer = tensor.mean(dim=0)
            axis_projections: list[dict[str, Any]] = []
            for index, layer in enumerate(artifact.layers):
                for construct in _CONSTRUCTS:
                    direction = result.directions[state][construct].get(layer)
                    if direction is None:
                        continue
                    axis_projections.append(
                        {
                            "construct": construct,
                            "layer": layer,
                            "projection": float(mean_by_layer[index] @ unit(direction)),
                            "direction_norm": float(direction.norm()),
                        }
                    )
            record = next(
                record
                for record in artifact.selection_records
                if str(record["stimulus_id"]) == stimulus_id
            )
            state_rows.append(
                {
                    "stimulus_id": stimulus_id,
                    "natural_state": str(record["natural_state"]),
                    "request_id": str(record["request_id"]),
                    "rollout_id": str(record["rollout_id"]),
                    "scenario_group": str(record["scenario_group"]),
                    "positions": list(record["positions"]),
                    "selection_rule": str(record["selection_rule"]),
                    "layers": list(artifact.layers),
                    "axis_projections": axis_projections,
                }
            )
        states[state] = {
            "artifact": str(artifact.path),
            "layers": list(artifact.layers),
            "selection_manifest": artifact.selection_manifest,
            "rows": state_rows,
        }

    comparison: list[dict[str, Any]] = []
    base_artifact = artifacts.get(STATE_BASE)
    final_artifact = artifacts.get(STATE_FINAL)
    content_confounding_reasons: list[str] = []
    if base_artifact is not None and final_artifact is not None:
        if base_artifact.natural_stimuli_sha256 != final_artifact.natural_stimuli_sha256:
            content_confounding_reasons.append("natural stimulus corpus digests differ")
        if base_artifact.rendered_sha256 != final_artifact.rendered_sha256:
            content_confounding_reasons.append("rendered natural prefix digests differ")
        positions_by_request = {
            state: {
                str(record["request_id"]): tuple(int(value) for value in record["positions"])
                for record in artifact.selection_records
            }
            for state, artifact in ((STATE_BASE, base_artifact), (STATE_FINAL, final_artifact))
        }
        if positions_by_request[STATE_BASE] != positions_by_request[STATE_FINAL]:
            content_confounding_reasons.append("selected token positions differ")
        base_stimulus_by_request = {
            str(record["request_id"]): str(record["stimulus_id"])
            for record in base_artifact.selection_records
        }
        final_stimulus_by_request = {
            str(record["request_id"]): str(record["stimulus_id"])
            for record in final_artifact.selection_records
        }
        common_ids = sorted(set(base_stimulus_by_request) & set(final_stimulus_by_request))
        common_layers = sorted(set(base_artifact.layers) & set(final_artifact.layers))
        base_index = {layer: index for index, layer in enumerate(base_artifact.layers)}
        final_index = {layer: index for index, layer in enumerate(final_artifact.layers)}
        for request_id in common_ids:
            base_stimulus_id = base_stimulus_by_request[request_id]
            final_stimulus_id = final_stimulus_by_request[request_id]
            for layer in common_layers:
                delta = final_artifact.activations[final_stimulus_id][
                    :, final_index[layer], :
                ].mean(dim=0) - base_artifact.activations[base_stimulus_id][
                    :, base_index[layer], :
                ].mean(dim=0)
                trained = result.displacement_directions.get(layer)
                comparison.append(
                    {
                        "request_id": request_id,
                        "base_stimulus_id": base_stimulus_id,
                        "final_stimulus_id": final_stimulus_id,
                        "layer": layer,
                        "base_to_final_norm": float(delta.norm()),
                        "base_to_final_alignment_with_trained_displacement": (
                            None if trained is None else cosine(delta, trained)
                        ),
                    }
                )
    return {
        "supported": bool(artifacts),
        "content_confounding": {
            "present": bool(content_confounding_reasons),
            "reasons": content_confounding_reasons,
            "basis": (
                "Natural rollout text and selected positions may differ by state. Deltas join on "
                "stable predeclared request IDs and are descriptive, so they combine content and "
                "checkpoint effects."
            ),
            "stimuli_sha256_by_state": {
                state: artifact.natural_stimuli_sha256
                for state, artifact in sorted(artifacts.items())
            },
            "rendered_sha256_by_state": {
                state: artifact.rendered_sha256 for state, artifact in sorted(artifacts.items())
            },
        },
        "states": states,
        "base_to_final": {
            "available": base_artifact is not None and final_artifact is not None,
            "n_common_requests": 0
            if base_artifact is None or final_artifact is None
            else len(
                {str(record["request_id"]) for record in base_artifact.selection_records}
                & {str(record["request_id"]) for record in final_artifact.selection_records}
            ),
            "n_common_layers": 0
            if base_artifact is None or final_artifact is None
            else len(set(base_artifact.layers) & set(final_artifact.layers)),
            "rows": comparison,
        },
    }


def _select_calibration_target(  # noqa: PLR0913 - selection binds geometry, files, report, and prior expectation
    projections: Sequence[ProjectionRead],
    directions: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
    direction_paths: Mapping[str, Mapping[str, Path]],
    *,
    preferred_pooling: str,
    geometry_report_path: Path,
    geometry_report_sha256: str,
    expectation: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one target using fit-only quality and derive a dimensionless steering multiplier."""
    target_construct = str(expectation["target_construct"])
    final = [
        read
        for read in projections
        if read.state == STATE_FINAL and read.construct == target_construct
    ]
    if not final:
        raise CooperationInterpError(
            f"no final-state {target_construct!r} calibration reads available for selection"
        )
    preferred = [read for read in final if read.pooling == preferred_pooling]
    candidates = preferred or final
    chosen = max(
        candidates,
        key=lambda read: (
            read.fit_direction_accuracy - read.fit_placebo_accuracy_max,
            read.fit_split_half_cosine,
            -read.layer,
            read.construct,
        ),
    )
    direction = directions[STATE_FINAL][chosen.construct][chosen.layer]
    alpha_multiplier = 0.5
    magnitude = alpha_multiplier * float(direction.norm())
    if magnitude <= 0.0:
        raise CooperationInterpError("selected calibration direction has no positive magnitude")
    path = direction_paths[STATE_FINAL][chosen.construct]
    resolved_report_path = geometry_report_path.resolve()
    if not resolved_report_path.is_file():
        raise FileNotFoundError(f"geometry report is missing: {resolved_report_path}")
    actual_report_sha256 = sha256_of_file(resolved_report_path)
    if actual_report_sha256 != geometry_report_sha256:
        raise CooperationInterpError(
            f"geometry report digest changed before selected-target write: "
            f"{geometry_report_sha256} != {actual_report_sha256}"
        )
    calibration_margin = chosen.fit_direction_accuracy - chosen.fit_placebo_accuracy_max
    return {
        "schema": "cooperation-generalization-selected-target/v1",
        "version": 1,
        "target_construct": target_construct,
        "direction": chosen.construct,
        "direction_path": str(path),
        "layer": chosen.layer,
        "magnitude": magnitude,
        "alpha_multiplier": alpha_multiplier,
        "calibration_metric": calibration_margin,
        "calibration_rationale": (
            f"The private pre-intervention expectation fixes the target construct as "
            f"{target_construct}. Its layer is selected from fit-only geometry by maximizing "
            "direction accuracy minus matched-norm placebo accuracy; ties use split-half cosine "
            "then the shallowest layer. The margin may be negative because it is descriptive, "
            "not a gate. The raw magnitude is 0.5 times the selected direction norm. No "
            "intervention output was used."
        ),
        "expected_effect": expectation["expected_effect"],
        "expectation_recorded_before_intervention": True,
        "exploratory": True,
        "direction_sha256": sha256_of_file(path),
        "geometry_report_path": str(resolved_report_path),
        "geometry_report_sha256": geometry_report_sha256,
    }


def analyze_cached_cells(  # noqa: C901, PLR0912, PLR0913, PLR0915 - explicit read contract and guards
    ladder: Ladder,
    stimuli: Sequence[Stimulus],
    *,
    reserved_group_ids: Sequence[str] = (),
    reserved_pair_ids: Sequence[str] = (),
    positive_side: str = "A",
    poolings: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    pairs_per_construct: int = 12,
    config: ProbeConfig | None = None,
    n_placebos: int = DEFAULT_N_PLACEBOS,
    seed: int = DEFAULT_SEED,
    preferred_pooling: str = DEFAULT_POOLING,
) -> CooperationInterpResult:
    """Run the complete cached base/final construct geometry read on CPU."""
    if n_placebos < 1:
        raise ValueError(f"n_placebos must be positive, got {n_placebos}")
    probe_config = config or ProbeConfig(n_folds=DEFAULT_N_FOLDS, seed=seed)
    if probe_config.n_folds < 2:  # noqa: PLR2004 - two folds are the minimum split
        raise ValueError(f"n_folds must be at least 2, got {probe_config.n_folds}")
    base = ladder.cell(BASE_ARM, BASE_STEP)
    non_base = [cell for cell in ladder.cells if cell.arm != BASE_ARM]
    if len(non_base) != 1:
        raise CellFormatError(
            f"cooperation construct geometry needs exactly one final cell beside the base, got "
            f"{[cell.label for cell in non_base]}"
        )
    final = non_base[0]
    splits = build_construct_splits(
        stimuli,
        pairs_per_construct=pairs_per_construct,
        required_constructs=_CONSTRUCTS,
        reserved_groups=reserved_group_ids,
        external_pair_ids=reserved_pair_ids,
    )
    requested_poolings = tuple(poolings or ladder.poolings)
    if len(set(requested_poolings)) != len(requested_poolings):
        raise ValueError(f"requested poolings repeat a pooling: {requested_poolings}")
    missing_poolings = sorted(set(requested_poolings) - set(ladder.poolings))
    if missing_poolings:
        raise CellFormatError(
            f"requested poolings are absent from cached cells: {missing_poolings}"
        )
    requested_layers = tuple(layers if layers is not None else range(ladder.identity.n_layers))
    if not requested_layers:
        raise ValueError("at least one layer is required")
    invalid_layers = sorted(set(requested_layers) - set(range(ladder.identity.n_layers)))
    if invalid_layers:
        raise CellFormatError(f"requested layers are outside capture: {invalid_layers}")
    if preferred_pooling not in requested_poolings:
        raise CellFormatError(
            f"export pooling {preferred_pooling!r} is not among requested poolings {requested_poolings}"
        )
    layouts = {
        construct: pair_layout(base, construct, positive_side=positive_side)
        for construct in _CONSTRUCTS
    }
    projections: list[ProjectionRead] = []
    directions: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        state: {construct: {} for construct in _CONSTRUCTS} for state in STATES
    }
    for state, cell in ((STATE_BASE, base), (STATE_FINAL, final)):
        for construct in _CONSTRUCTS:
            split = splits[construct]
            layout = layouts[construct]
            fit_indices = _pair_indices(layout, split.fit_pair_ids, role="fit")
            heldout_indices = _pair_indices(layout, split.heldout_pair_ids, role="held-out")
            if int(fit_indices.numel()) < probe_config.n_folds:
                raise CooperationInterpError(
                    f"{construct!r} has {fit_indices.numel()} fit pairs but n_folds="
                    f"{probe_config.n_folds}; a fit-only calibration read needs one fold per pair group"
                )
            if heldout_indices.numel() == 0:
                raise CooperationInterpError(f"{construct!r} has no held-out pairs")
            for pooling in requested_poolings:
                for layer in requested_layers:
                    read, direction = _fit_read(
                        state=state,
                        cell=cell,
                        construct=construct,
                        pooling=pooling,
                        layer=layer,
                        layout=layout,
                        split=split,
                        fit_indices=fit_indices,
                        heldout_indices=heldout_indices,
                        config=probe_config,
                        n_placebos=n_placebos,
                        seed=seed,
                        heldout_strata=_metadata_pair_strata(
                            stimuli,
                            split,
                            fields=(
                                "scenario_group",
                                "story_group",
                                "printed_position",
                                "label_print_order",
                                "procedure_regime",
                            ),
                        ),
                    )
                    projections.append(read)
                    if pooling == preferred_pooling:
                        directions[state][construct][layer] = direction

    displacement_reads: list[DisplacementRead] = []
    displacement_directions: dict[int, torch.Tensor] = {}
    displacement_pooling = preferred_pooling
    displacement_counts: dict[int, int] = {}
    for construct in _CONSTRUCTS:
        split = splits[construct]
        layout = layouts[construct]
        for layer in requested_layers:
            read, displacement = _read_displacement(
                base,
                final,
                construct=construct,
                pooling=displacement_pooling,
                layer=layer,
                layout=layout,
                split=split,
                base_direction=directions[STATE_BASE][construct][layer],
                final_direction=directions[STATE_FINAL][construct][layer],
            )
            displacement_reads.append(read)
            prior = displacement_directions.get(layer)
            pair_count = len(split.heldout_pair_ids) * 2
            if prior is None:
                displacement_directions[layer] = displacement * pair_count
                displacement_counts[layer] = pair_count
            else:
                displacement_directions[layer] = prior + displacement * pair_count
                displacement_counts[layer] += pair_count
    for layer, direction in displacement_directions.items():
        displacement_directions[layer] = direction / displacement_counts[layer]
        if float(direction.norm()) == 0.0:
            raise CooperationInterpError(f"base-to-final displacement is zero at layer {layer}")

    confounds = _read_confound_checks(stimuli, splits, projections)
    return CooperationInterpResult(
        splits=splits,
        projections=tuple(projections),
        displacements=tuple(displacement_reads),
        directions=directions,
        displacement_directions=displacement_directions,
        confound_checks=confounds,
        calibration_selection={},
        export_pooling=preferred_pooling,
    )


def _split_payload(splits: Mapping[str, ConstructSplit]) -> dict[str, Any]:
    return {
        construct: {
            "fit_pair_ids": list(split.fit_pair_ids),
            "heldout_pair_ids": list(split.heldout_pair_ids),
            "fit_group_ids": list(split.fit_groups),
            "heldout_group_ids": list(split.heldout_groups),
            "group_by_pair": dict(sorted(split.group_by_pair.items())),
        }
        for construct, split in sorted(splits.items())
    }


def save_directions(
    out_dir: Path, result: CooperationInterpResult
) -> tuple[dict[str, dict[str, Path]], Path]:
    """Write lens-compatible per-layer directions and the exact direction manifest."""
    direction_paths: dict[str, dict[str, Path]] = {state: {} for state in STATES}
    for state in STATES:
        for construct in _CONSTRUCTS:
            path = out_dir / DIRECTIONS_DIRNAME / state / f"{construct}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result.directions[state][construct], path)
            direction_paths[state][construct] = path.resolve()
        displacement_path = (
            out_dir / DIRECTIONS_DIRNAME / state / f"{DIRECTION_TRAINED_DISPLACEMENT}.pt"
        )
        displacement_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result.displacement_directions, displacement_path)
        direction_paths[state][DIRECTION_TRAINED_DISPLACEMENT] = displacement_path.resolve()
    if set(direction_paths[STATE_BASE]) != set(REQUIRED_DIRECTIONS):
        raise CooperationInterpError("direction manifest construction omitted a required direction")
    manifest_path = out_dir / DIRECTION_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                state: {name: str(path) for name, path in sorted(paths.items())}
                for state, paths in sorted(direction_paths.items())
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return direction_paths, manifest_path


def build_payload(result: CooperationInterpResult, *, context: Mapping[str, Any]) -> dict[str, Any]:
    """Create the durable JSON report without serialising tensors."""
    return {
        "context": dict(context),
        "splits": _split_payload(result.splits),
        "confound_checks": result.confound_checks,
        "projections": [asdict(read) for read in result.projections],
        "displacements": [asdict(read) for read in result.displacements],
        "calibration_selection": dict(result.calibration_selection),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CPU-only cooperation geometry CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--reserved-group-ids", type=Path, required=True)
    parser.add_argument("--intervention-expectation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--final-arm", default=DEFAULT_FINAL_ARM)
    parser.add_argument("--final-step", type=int, default=DEFAULT_FINAL_STEP)
    parser.add_argument("--positive-side", default="A")
    parser.add_argument("--poolings", default=None)
    parser.add_argument(
        "--export-pooling",
        default=DEFAULT_POOLING,
        help="One requested pooling whose directions are exported to the lens manifest.",
    )
    parser.add_argument("--layers", default=None)
    parser.add_argument("--pairs-per-construct", type=int, default=12)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--n-placebos", type=int, default=DEFAULT_N_PLACEBOS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--require-capture-provenance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require concrete tokenizer and kernel identities on cached cells (default: true).",
    )
    parser.add_argument(
        "--natural-prefix-artifact",
        action="append",
        default=[],
        metavar="STATE=PATH",
        help="Optional separate natural-prefix artifact; repeat once for base and final.",
    )
    return parser


def _split(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("comma-separated argument selected nothing")
    return values


def _natural_artifact_specs(raw_specs: Sequence[str]) -> dict[str, Path]:
    """Parse the explicit per-state natural-prefix artifact arguments."""
    artifacts: dict[str, Path] = {}
    for raw in raw_specs:
        state, separator, raw_path = raw.partition("=")
        if not separator or state not in STATES or not raw_path:
            raise ValueError(
                "--natural-prefix-artifact must use STATE=PATH with state base or final"
            )
        if state in artifacts:
            raise ValueError(f"natural-prefix artifact repeats state {state!r}")
        artifacts[state] = Path(raw_path)
    if artifacts and set(artifacts) != set(STATES):
        raise ValueError("natural-prefix cross-check requires base and final artifacts together")
    return artifacts


def load_intervention_expectation(path: Path) -> dict[str, Any]:
    """Load a private expectation authored before intervention outputs exist."""
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in PRIVATE_EXPECTATION_ROOTS):
        raise ValueError(
            f"intervention expectation must live under one of {PRIVATE_EXPECTATION_ROOTS}, got {path}"
        )
    payload = json.loads(resolved.read_text())
    required = {
        "target_construct",
        "expected_effect",
        "expectation_recorded_before_intervention",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"intervention expectation must contain exactly {sorted(required)}")
    if not isinstance(payload["expected_effect"], str) or not payload["expected_effect"].strip():
        raise ValueError("intervention expected_effect must be non-empty text")
    if payload["target_construct"] not in _CONSTRUCTS:
        raise ValueError(
            f"intervention target_construct must be one of {list(_CONSTRUCTS)}, got "
            f"{payload['target_construct']!r}"
        )
    if payload["expectation_recorded_before_intervention"] is not True:
        raise ValueError("intervention expectation must be recorded before intervention")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load private inputs, run the cached read, and write all durable artifacts."""
    reserved = load_reserved_identities(cast("Path", args.reserved_group_ids))
    expectation = load_intervention_expectation(cast("Path", args.intervention_expectation))
    stimuli_path = cast("Path", args.stimuli)
    stimuli = load_private_construct_stimuli(
        stimuli_path,
        pairs_per_construct=args.pairs_per_construct,
        reserved_groups=tuple(reserved.group_ids),
        external_pair_ids=tuple(reserved.pair_ids),
    )
    digest = stimuli_digest(stimuli)
    ladder = load_ladder(
        cast("Path", args.capture_root),
        arms=[args.final_arm],
        steps=[args.final_step],
        stimuli_sha256=digest,
        require_capture_provenance=args.require_capture_provenance,
    )
    expected = {f"{BASE_ARM}/step-{BASE_STEP}", f"{args.final_arm}/step-{args.final_step}"}
    if {cell.label for cell in ladder.cells} != expected:
        raise CellFormatError(
            f"cooperation geometry needs exactly base and final cells {sorted(expected)}, got "
            f"{[cell.label for cell in ladder.cells]}"
        )
    result = analyze_cached_cells(
        ladder,
        stimuli,
        reserved_group_ids=tuple(reserved.group_ids),
        reserved_pair_ids=tuple(reserved.pair_ids),
        positive_side=args.positive_side,
        poolings=_split(args.poolings),
        layers=None if args.layers is None else [int(value) for value in _split(args.layers) or []],
        pairs_per_construct=args.pairs_per_construct,
        config=ProbeConfig(n_folds=args.n_folds, seed=args.seed),
        n_placebos=args.n_placebos,
        seed=args.seed,
        preferred_pooling=args.export_pooling,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    direction_paths, manifest_path = save_directions(args.out_dir, result)
    natural_specs = _natural_artifact_specs(args.natural_prefix_artifact)
    natural_artifacts = {
        state: load_natural_prefix_artifact(
            path,
            state=state,
            cell=ladder.cell(BASE_ARM, BASE_STEP)
            if state == STATE_BASE
            else ladder.cell(args.final_arm, args.final_step),
        )
        for state, path in sorted(natural_specs.items())
    }
    natural_cross_check = natural_prefix_cross_check(natural_artifacts, result)
    payload = build_payload(
        result,
        context={
            "capture_root": str(args.capture_root),
            "stimuli_file": str(stimuli_path),
            "stimuli_sha256": digest,
            "reserved_group_ids": sorted(reserved.group_ids),
            "reserved_pair_ids": sorted(reserved.pair_ids),
            "cells": [cell.label for cell in ladder.cells],
            "cell_identities": {
                cell.label: {
                    "identity": cell.identity.to_payload(),
                    "applied_adapter_weights": cell.applied_adapter_weights,
                    "adapter_weights_sha256": cell.adapter_weights_sha256,
                }
                for cell in ladder.cells
            },
            "positive_side": args.positive_side,
            "poolings": list(_split(args.poolings) or ladder.poolings),
            "export_pooling": args.export_pooling,
            "layers": list(
                range(ladder.identity.n_layers)
                if args.layers is None
                else [int(value) for value in _split(args.layers) or []]
            ),
            "n_folds": args.n_folds,
            "n_placebos": args.n_placebos,
            "direction_manifest": str(manifest_path),
            "selected_target": str(args.out_dir / SELECTION_FILENAME),
            "natural_prefix_artifacts": {
                state: str(path) for state, path in sorted(natural_specs.items())
            },
            "cpu_only": True,
        },
    )
    payload["natural_prefix_cross_check"] = natural_cross_check
    report_path = args.out_dir / REPORT_FILENAME
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    geometry_report_sha256 = sha256_of_file(report_path)
    selection = _select_calibration_target(
        result.projections,
        result.directions,
        direction_paths,
        preferred_pooling=result.export_pooling,
        geometry_report_path=report_path,
        geometry_report_sha256=geometry_report_sha256,
        expectation=expectation,
    )
    selection_path = args.out_dir / SELECTION_FILENAME
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    logger.info(
        "cooperation interpolation written, report=%s direction_files=%d selected_target=%s",
        report_path,
        sum(len(paths) for paths in direction_paths.values()),
        selection_path,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CPU-only cooperation geometry read."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
