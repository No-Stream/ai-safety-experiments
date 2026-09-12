"""Validate and, after the readout, prune one cooperation run's checkpoint ladder.

The production command is the only retention entrypoint used by the cooperation sequence. It
checks the run-owned readouts, checks that every selected checkpoint has the complete Trainer
state, and then passes the actual ``games.lora`` adapter loader to the conservative pruning API.
``--dry-run`` performs the same CPU inventory and schema checks without loading a model or deleting
anything, which is the safe command for plan review and tests.

Example production invocation::

    uv run --frozen python -m games.cooperation_retention \
        --run-root artifacts/games/cooperation-generalization/<run-id>/training \
        --base-model Qwen/Qwen3.5-9B \
        --retain-step 19 --retain-step 20 \
        --adapter-step 5 --adapter-step 10 \
        --readout ../final/behavior.summary.json \
        --readout ../final/allocation.summary.json \
        --readout ../final/full-context.summary.json \
        --readout ../final/core-survey.summary.json \
        --readout ../final/prosocialness.summary.json \
        --readout ../final/local-dt.summary.json \
        --readout ../capture/ladder-manifest.json \
        --readout ../geometry/cooperation_interp.json \
        --readout ../lens/cooperation_lens.json \
        --readout ../intervention/steering_records.jsonl

The model is loaded lazily, only after the readout and checkpoint checks pass. The loader uses the
same TRL model construction and PEFT attachment path as capture and training, and reuses one base
model while it validates each retained checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from games.checkpoint_retention import (
    CheckpointSnapshot,
    RetentionManifest,
    checkpoint_disk_usage,
    inspect_checkpoints,
    prune_redundant_checkpoints,
    snapshot_payload,
)
from games.cooperation_eval_runner import (
    _model_identity as endpoint_model_identity,  # pyright: ignore[reportPrivateUsage]
)
from games.cooperation_interp import DisplacementRead, ProjectionRead
from games.evals import read_eval_records, rebuild_summary
from games.interp_cells import BASE_ARM, BASE_STEP, load_ladder
from games.interp_steering import load_direction_file, load_selected_target, summarise_records
from games.lora import (
    adapter_config_identity,
    assert_adapter_matches_base,
    assert_one_adapter_config,
    attach_adapter,
    load_adapter_base,
)
from reward_hacking.interp.jacobian import resolve_weights_identity

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
_ENDPOINT_SUMMARY_SECTIONS: dict[str, str] = {
    "behavior.summary.json": "game-behavior",
    "allocation.summary.json": "self-report",
    "full-context.summary.json": "self-report",
    "core-survey.summary.json": "self-report",
    "prosocialness.summary.json": "self-report",
    "local-dt.summary.json": "dt-probes",
}
_KNOWN_JSON_READOUTS = frozenset(
    {
        *(_ENDPOINT_SUMMARY_SECTIONS),
        "ladder-manifest.json",
        "cooperation_interp.json",
        "cooperation_lens.json",
    }
)
_KNOWN_JSONL_READOUTS = frozenset({"steering_records.jsonl"})
_REQUIRED_READOUT_NAMES = frozenset({*_KNOWN_JSON_READOUTS, *_KNOWN_JSONL_READOUTS})
_REQUIRED_DIRECTION_NAMES = frozenset(
    {"costly-other-regard", "decision-dependence", "trained-displacement"}
)
_SHA256_LENGTH = 64
_GEOMETRY_CONSTRUCTS = _REQUIRED_DIRECTION_NAMES - {"trained-displacement"}
_NATURAL_REQUEST_COUNT = 4
_NATURAL_LAYER_COUNT = 3


def _require_object(payload: object, path: Path, field_name: str = "readout") -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"readout {path} {field_name} must be a nonempty JSON object")
    return cast("dict[str, Any]", payload)


def _require_mapping(payload: object, path: Path, field_name: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"readout {path} field {field_name!r} must be a nonempty object")
    return dict(payload)


def _require_text(payload: Mapping[str, Any], field_name: str, path: Path) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"readout {path} field {field_name!r} must be nonempty text")
    return value


def _require_nonempty_list(payload: Mapping[str, Any], field_name: str, path: Path) -> list[Any]:
    value = payload.get(field_name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"readout {path} field {field_name!r} must be a nonempty list")
    return value


def _require_nonnegative_int(payload: Mapping[str, Any], field_name: str, path: Path) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"readout {path} field {field_name!r} must be a nonnegative integer")
    return value


def _require_positive_int(payload: Mapping[str, Any], field_name: str, path: Path) -> int:
    value = _require_nonnegative_int(payload, field_name, path)
    if value == 0:
        raise ValueError(f"readout {path} field {field_name!r} must be positive")
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"referenced artifact does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(payload: Mapping[str, Any], field_name: str, path: Path) -> str:
    value = _require_text(payload, field_name, path)
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"readout {path} field {field_name!r} must be a lowercase sha256 digest")
    return value


def _referenced_path(owner: Path, value: object, *, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"readout {owner} field {field_name!r} must be nonempty text")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    working_directory_candidate = candidate.resolve()
    artifact_relative_candidate = (owner.parent / candidate).resolve()
    if working_directory_candidate.exists() or not artifact_relative_candidate.exists():
        return working_directory_candidate
    return artifact_relative_candidate


def _endpoint_trace_path(summary_path: Path) -> Path:
    return summary_path.with_name(summary_path.name.removesuffix(".summary.json") + ".jsonl")


def _validate_endpoint_summary(path: Path, payload: Mapping[str, Any]) -> None:
    expected_section = _ENDPOINT_SUMMARY_SECTIONS[path.name]
    section = _require_mapping(payload.get(expected_section), path, expected_section)
    n_records = _require_positive_int(section, "n_records", path)
    for field_name in ("parse_failure_rate", "truncated_thinking_rate"):
        value = section.get(field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"readout {path} field {expected_section}.{field_name} must be finite")
        if value < 0 or value > 1:
            raise ValueError(
                f"readout {path} field {expected_section}.{field_name} is outside [0, 1]"
            )
    trace_path = _endpoint_trace_path(path)
    if not trace_path.is_file():
        raise FileNotFoundError(
            f"endpoint summary {path} is incomplete without its sibling trace {trace_path}"
        )
    records = read_eval_records(trace_path)
    meta = records[0]
    request_count = _require_positive_int(meta, "cooperation_request_count", trace_path)
    if n_records != request_count or len(records) - 1 != request_count:
        raise ValueError(
            f"readout {path} n_records={n_records} disagrees with request_count={request_count} "
            f"and trace_records={len(records) - 1}"
        )
    _require_positive_int(meta, "cooperation_rendered_prompt_count", trace_path)
    _require_sha256(meta, "cooperation_rendered_prompt_digest", trace_path)
    expected_endpoint = path.name.removesuffix(".summary.json")
    if meta.get("cooperation_endpoint") != expected_endpoint:
        raise ValueError(f"trace {trace_path} has the wrong cooperation_endpoint identity")
    settings = _require_mapping(meta.get("settings"), trace_path, "settings")
    _require_text(settings, "arm", trace_path)
    _require_nonnegative_int(settings, "step", trace_path)
    rebuilt = rebuild_summary(trace_path)
    if dict(payload) != rebuilt:
        raise ValueError(f"summary {path} disagrees with its trace {trace_path}")


def _validate_ladder_manifest(  # noqa: C901 - one linear native capture contract
    path: Path, payload: Mapping[str, Any]
) -> None:
    identity = _require_mapping(payload.get("identity"), path, "identity")
    for field_name in (
        "base_model",
        "stimuli_sha256",
        "rendered_sha256",
        "layer_convention",
        "compute_dtype",
        "store_dtype",
        "stimulus_render",
        "tokenizer_identity",
        "kernel_identity",
    ):
        _require_text(identity, field_name, path)
    _require_positive_int(identity, "n_layers", path)
    _require_positive_int(identity, "hidden_size", path)
    if identity.get("capture_prefix_states") is not True:
        raise ValueError(
            f"readout {path} omitted required prompt and teacher-forced prefix capture"
        )
    natural_layers = _require_nonempty_list(identity, "natural_prefix_layers", path)
    if any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
        for layer in natural_layers
    ):
        raise ValueError(f"readout {path} natural_prefix_layers must be nonnegative integers")
    requested = _require_nonempty_list(payload, "requested_cells", path)
    if any(not isinstance(label, str) or not label.strip() for label in requested):
        raise ValueError(f"readout {path} requested_cells must contain nonempty labels")
    poolings = _require_nonempty_list(payload, "poolings", path)
    if any(not isinstance(pooling, str) or not pooling.strip() for pooling in poolings):
        raise ValueError(f"readout {path} poolings must contain nonempty labels")
    cells = _require_mapping(payload.get("cells"), path, "cells")
    if set(cells) != set(requested):
        raise ValueError(f"readout {path} cells do not cover exactly requested_cells")
    for label, cell in cells.items():
        _require_mapping(cell, path, f"cells[{label!r}]")
    if payload.get("stopped_reason") is not None:
        raise ValueError(f"readout {path} records an incomplete stopped capture")
    capture_root = path.parent
    ladder = load_ladder(capture_root, require_capture_provenance=True)
    for cell in ladder.cells:
        direct = cell.source / "natural-prefix-activations.safetensors"
        nested_manifest = cell.source / "natural-prefix" / "natural-prefix-manifest.json"
        nested_tensor = cell.source / "natural-prefix" / "natural-prefix-activations.safetensors"
        if not direct.is_file() and not (nested_manifest.is_file() and nested_tensor.is_file()):
            raise FileNotFoundError(
                f"capture cell {cell.source} lacks its required natural-prefix tensor artifact"
            )


def _validate_geometry_report(  # noqa: C901, PLR0912, PLR0915 - one native geometry contract
    path: Path, payload: Mapping[str, Any]
) -> None:
    context = _require_mapping(payload.get("context"), path, "context")
    _require_text(context, "capture_root", path)
    _require_text(context, "direction_manifest", path)
    cells = _require_nonempty_list(context, "cells", path)
    if any(not isinstance(cell, str) or not cell.strip() for cell in cells):
        raise ValueError(f"readout {path} context.cells must contain nonempty labels")
    cell_identities = _require_mapping(context.get("cell_identities"), path, "cell_identities")
    for label, entry in cell_identities.items():
        identity_entry = _require_mapping(entry, path, f"cell_identities[{label!r}]")
        identity = _require_mapping(
            identity_entry.get("identity"), path, f"cell_identities[{label!r}].identity"
        )
        _require_text(identity, "base_model", path)
    splits = _require_mapping(payload.get("splits"), path, "splits")
    projections = _require_nonempty_list(payload, "projections", path)
    displacements = _require_nonempty_list(payload, "displacements", path)
    confound_checks = _require_mapping(payload.get("confound_checks"), path, "confound_checks")
    if set(confound_checks) != {
        "story_group",
        "printed_position",
        "procedure_regime",
        "action_token_commitment",
    }:
        raise ValueError(f"readout {path} omits required confound measurements")
    action_check = _require_mapping(
        confound_checks["action_token_commitment"], path, "action_token_commitment"
    )
    if action_check.get("supported") is not True:
        raise ValueError(f"readout {path} lacks the pre-action measurement-boundary check")
    if not isinstance(payload.get("calibration_selection"), Mapping):
        raise TypeError(f"readout {path} field 'calibration_selection' must be an object")
    if not splits:
        raise ValueError(f"readout {path} field 'splits' must be nonempty")
    if set(splits) != set(_GEOMETRY_CONSTRUCTS):
        raise ValueError(f"readout {path} omits required construct splits")
    validated_splits: dict[str, dict[str, set[Any]]] = {}
    for construct, split in splits.items():
        split_item = _require_mapping(split, path, f"splits[{construct!r}]")
        validated_splits[construct] = {
            field_name: set(_require_nonempty_list(split_item, field_name, path))
            for field_name in (
                "fit_pair_ids",
                "heldout_pair_ids",
                "fit_group_ids",
                "heldout_group_ids",
            )
        }
        if (
            validated_splits[construct]["fit_pair_ids"]
            & validated_splits[construct]["heldout_pair_ids"]
            or validated_splits[construct]["fit_group_ids"]
            & validated_splits[construct]["heldout_group_ids"]
        ):
            raise ValueError(f"readout {path} geometry fit and held-out splits overlap")
    expected_fields = {
        "projections": {item.name for item in fields(ProjectionRead)},
        "displacements": {item.name for item in fields(DisplacementRead)},
    }
    for field_name, entries in (("projections", projections), ("displacements", displacements)):
        for index, entry in enumerate(entries):
            item = _require_mapping(entry, path, f"{field_name}[{index}]")
            if set(item) != expected_fields[field_name]:
                raise ValueError(
                    f"readout {path} {field_name}[{index}] is not a native measurement row"
                )
            _require_text(item, "construct", path)
            _require_text(item, "pooling", path)
            _require_nonnegative_int(item, "layer", path)
            construct = cast("str", item["construct"])
            if construct not in validated_splits:
                raise ValueError(f"readout {path} {field_name}[{index}] names an unknown construct")
            identifier_fields = (
                ("fit_pair_ids", "heldout_pair_ids", "fit_group_ids", "heldout_group_ids")
                if field_name == "projections"
                else ("pair_ids", "group_ids")
            )
            for identifier_field in identifier_fields:
                _require_nonempty_list(item, identifier_field, path)
            expected_identifiers = validated_splits[construct]
            if field_name == "projections":
                if any(set(item[name]) != expected_identifiers[name] for name in identifier_fields):
                    raise ValueError(f"readout {path} projection row disagrees with its split")
                non_metric_fields = {
                    "state",
                    "construct",
                    "pooling",
                    *identifier_fields,
                    "layer",
                    "grouped_projection_gaps",
                }
            else:
                if (
                    set(item["pair_ids"]) != expected_identifiers["heldout_pair_ids"]
                    or set(item["group_ids"]) != expected_identifiers["heldout_group_ids"]
                ):
                    raise ValueError(f"readout {path} displacement row disagrees with its split")
                non_metric_fields = {"construct", "pooling", *identifier_fields, "layer"}
            for metric_name in set(item) - non_metric_fields:
                value = item[metric_name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"readout {path} {field_name}[{index}].{metric_name} is not finite"
                    )
    direction_manifest = _referenced_path(
        path, context["direction_manifest"], field_name="context.direction_manifest"
    )
    manifest = _require_object(
        json.loads(direction_manifest.read_text(encoding="utf-8")), direction_manifest
    )
    if set(manifest) != {"base", "final"}:
        raise ValueError(
            f"direction manifest {direction_manifest} must contain exactly base and final"
        )
    direction_paths: dict[str, dict[str, Path]] = {"base": {}, "final": {}}
    directions: dict[str, dict[str, dict[int, torch.Tensor]]] = {"base": {}, "final": {}}
    for state in ("base", "final"):
        state_paths = _require_mapping(manifest[state], direction_manifest, state)
        if set(state_paths) != set(_REQUIRED_DIRECTION_NAMES):
            raise ValueError(
                f"direction manifest {direction_manifest} state {state!r} must contain exactly "
                f"{sorted(_REQUIRED_DIRECTION_NAMES)}"
            )
        for name, value in state_paths.items():
            direction = _referenced_path(direction_manifest, value, field_name=f"{state}.{name}")
            direction_paths[state][name] = direction
            directions[state][name] = load_direction_file(direction)
    selected_target = _referenced_path(
        path, context.get("selected_target"), field_name="context.selected_target"
    )
    selected_target_wrapper = load_selected_target(
        selected_target,
        directions["final"],
        direction_paths=direction_paths["final"],
    )
    report_reference = Path(selected_target_wrapper["geometry_report_path"]).resolve()
    if report_reference != path.resolve():
        raise ValueError(
            f"selected target {selected_target} references a different geometry report"
        )
    if _sha256(path) != selected_target_wrapper["geometry_report_sha256"]:
        raise ValueError(f"selected target {selected_target} has a stale geometry_report_sha256")
    final_layers = {
        name: set(direction_layers) for name, direction_layers in directions["final"].items()
    }
    projection_cells = {
        (item["state"], item["construct"], item["pooling"], item["layer"])
        for item in cast("list[dict[str, Any]]", projections)
    }
    if len(projection_cells) != len(projections):
        raise ValueError(f"readout {path} contains duplicate projection measurements")
    for construct in _GEOMETRY_CONSTRUCTS:
        for layer in final_layers[construct]:
            poolings = {
                pooling
                for _state, found_construct, pooling, found_layer in projection_cells
                if found_construct == construct and found_layer == layer
            }
            if not poolings or any(
                {
                    state
                    for state, found_construct, found_pooling, found_layer in projection_cells
                    if found_construct == construct
                    and found_pooling == pooling
                    and found_layer == layer
                }
                != {"base", "final"}
                for pooling in poolings
            ):
                raise ValueError(
                    f"readout {path} omits base/final projection measurements for "
                    f"{construct} layer {layer}"
                )
    displacement_cells = {
        (item["construct"], item["pooling"], item["layer"])
        for item in cast("list[dict[str, Any]]", displacements)
    }
    if len(displacement_cells) != len(displacements):
        raise ValueError(f"readout {path} contains duplicate displacement measurements")
    export_pooling = _require_text(context, "export_pooling", path)
    expected_displacements = {
        (construct, export_pooling, layer)
        for construct in _GEOMETRY_CONSTRUCTS
        for layer in final_layers[construct]
    }
    if displacement_cells != expected_displacements:
        raise ValueError(f"readout {path} has incomplete displacement measurements")
    natural = _require_mapping(
        payload.get("natural_prefix_cross_check"), path, "natural_prefix_cross_check"
    )
    if natural.get("supported") is not True:
        raise ValueError(f"readout {path} lacks the required natural-prefix measurement")
    natural_states = _require_mapping(
        natural.get("states"), path, "natural_prefix_cross_check.states"
    )
    if set(natural_states) != {"base", "final"}:
        raise ValueError(f"readout {path} natural-prefix read must contain base and final states")
    base_to_final = _require_mapping(
        natural.get("base_to_final"), path, "natural_prefix_cross_check.base_to_final"
    )
    rows = _require_nonempty_list(base_to_final, "rows", path)
    if (
        base_to_final.get("available") is not True
        or base_to_final.get("n_common_requests") != _NATURAL_REQUEST_COUNT
        or base_to_final.get("n_common_layers") != _NATURAL_LAYER_COUNT
        or len(rows) != _NATURAL_REQUEST_COUNT * _NATURAL_LAYER_COUNT
    ):
        raise ValueError(f"readout {path} has an incomplete natural base-to-final measurement")


def _validate_lens_report(  # noqa: C901, PLR0912, PLR0915 - one native lens contract
    path: Path, payload: Mapping[str, Any]
) -> None:
    _require_text(payload, "base_model", path)
    _require_text(payload, "base_weights_identity", path)
    _require_text(payload, "tokenizer_content_sha256", path)
    _require_text(payload, "corpus_identity_sha256", path)
    _require_text(payload, "final_adapter", path)
    _require_text(payload, "final_arm", path)
    _require_nonnegative_int(payload, "final_step", path)
    _require_mapping(payload.get("construct_capture"), path, "construct_capture")
    states = _require_mapping(payload.get("states"), path, "states")
    if set(states) != {"base", "final"}:
        raise ValueError(f"readout {path} states must contain exactly base and final")
    state_direction_paths: dict[str, dict[str, Path]] = {"base": {}, "final": {}}
    for state, entry in states.items():
        item = _require_mapping(entry, path, f"states[{state!r}]")
        for field_name in ("state", "arm", "model", "lens_path"):
            _require_text(item, field_name, path)
        _require_nonnegative_int(item, "step", path)
        _require_mapping(item.get("lens_acquisition"), path, f"states[{state!r}].lens_acquisition")
        _require_mapping(
            item.get("accumulation_identity"), path, f"states[{state!r}].accumulation_identity"
        )
        quality = _require_mapping(item.get("fit_quality"), path, f"states[{state!r}].fit_quality")
        if quality.get("available") is not True:
            raise ValueError(f"readout {path} has no completed {state} lens quality measurement")
        _require_mapping(
            item.get("direction_readouts"), path, f"states[{state!r}].direction_readouts"
        )
        if item.get("state") != state or item.get("readout_kind") != "model-specific":
            raise ValueError(f"readout {path} has an invalid model-specific {state} lens state")
        if item.get("lens_state") != state or item.get("applied_to_state") != state:
            raise ValueError(f"readout {path} misbinds its {state} lens readout")
        lens_path = _referenced_path(
            path, item["lens_path"], field_name=f"states[{state!r}].lens_path"
        )
        if not lens_path.is_file():
            raise FileNotFoundError(f"referenced lens does not exist: {lens_path}")
        direction_readouts = cast("Mapping[str, Any]", item["direction_readouts"])
        if set(direction_readouts) != set(_REQUIRED_DIRECTION_NAMES):
            raise ValueError(
                f"readout {path} {state} lens omitted required directions; got "
                f"{sorted(direction_readouts)}"
            )
        for name, readout in direction_readouts.items():
            readout_item = _require_mapping(
                readout, path, f"states[{state!r}].direction_readouts[{name!r}]"
            )
            direction_path = _referenced_path(
                path,
                readout_item.get("path"),
                field_name=f"states[{state!r}].direction_readouts[{name!r}].path",
            )
            expected_sha = _require_sha256(
                readout_item,
                "sha256",
                path,
            )
            if _sha256(direction_path) != expected_sha:
                raise ValueError(f"readout {path} has a stale direction hash for {state}/{name}")
            state_direction_paths[state][name] = direction_path
            layers = _require_mapping(
                readout_item.get("layers"),
                path,
                f"states[{state!r}].direction_readouts[{name!r}].layers",
            )
            expected_layers = {str(layer) for layer in load_direction_file(direction_path)}
            if set(layers) != expected_layers:
                raise ValueError(
                    f"readout {path} {state}/{name} has incomplete per-layer measurements"
                )
            for layer, layer_payload in layers.items():
                layer_item = _require_mapping(
                    layer_payload, path, f"states[{state!r}].direction_readouts[{name!r}].{layer}"
                )
                _require_mapping(layer_item.get("quality"), path, f"{state}/{name}/{layer}.quality")
                _require_nonempty_list(layer_item, "real", path)
                _require_nonempty_list(layer_item, "matched_norm_random", path)
    shared = _require_mapping(
        payload.get("shared_base_lens_coordinate_sensitivity"),
        path,
        "shared_base_lens_coordinate_sensitivity",
    )
    expected_shared = {
        "readout_kind": "shared-base-coordinate-sensitivity",
        "approximation": True,
        "lens_state": "base",
        "applied_to_state": "final",
    }
    if any(shared.get(key) != value for key, value in expected_shared.items()):
        raise ValueError(f"readout {path} has no valid shared-base coordinate read")
    shared_quality = _require_mapping(
        shared.get("cross_checkpoint_fidelity"), path, "cross_checkpoint_fidelity"
    )
    if shared_quality.get("available") is not True:
        raise ValueError(f"readout {path} has no completed cross-checkpoint fidelity measurement")
    shared_readouts = _require_mapping(
        shared.get("direction_readouts"), path, "shared direction_readouts"
    )
    if set(shared_readouts) != set(_REQUIRED_DIRECTION_NAMES):
        raise ValueError(f"readout {path} shared lens omitted required direction measurements")
    for name, readout in shared_readouts.items():
        readout_item = _require_mapping(readout, path, f"shared.direction_readouts[{name!r}]")
        direction_path = _referenced_path(
            path, readout_item.get("path"), field_name=f"shared.direction_readouts[{name!r}].path"
        )
        expected_sha = _require_sha256(readout_item, "sha256", path)
        if _sha256(direction_path) != expected_sha:
            raise ValueError(f"readout {path} has a stale shared direction hash for {name}")
        if direction_path != state_direction_paths["final"][name]:
            raise ValueError(f"readout {path} shared/{name} does not use the final direction")
        layers = _require_mapping(
            readout_item.get("layers"),
            path,
            f"shared.direction_readouts[{name!r}].layers",
        )
        expected_layers = {str(layer) for layer in load_direction_file(direction_path)}
        if set(layers) != expected_layers:
            raise ValueError(f"readout {path} shared/{name} has incomplete per-layer measurements")
        for layer, layer_payload in layers.items():
            layer_item = _require_mapping(layer_payload, path, f"shared/{name}/{layer}")
            _require_mapping(layer_item.get("quality"), path, f"shared/{name}/{layer}.quality")
            _require_nonempty_list(layer_item, "real", path)
            _require_nonempty_list(layer_item, "matched_norm_random", path)


def _validate_steering_records(  # noqa: C901, PLR0912 - records, summary, resume are one artifact
    path: Path,
) -> None:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"readout {path} contains no JSON records")
    records: list[dict[str, Any]] = []
    conditions: set[str] = set()
    required = {
        "condition_key",
        "condition",
        "prompt_id",
        "response_text",
        "adapter",
        "row_kind",
        "sample_index",
    }
    for line_number, line in enumerate(lines, start=1):
        payload = _require_object(json.loads(line), path, f"line {line_number}")
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"readout {path} line {line_number} lacks required fields {missing}")
        for field_name in ("condition_key", "condition", "prompt_id", "response_text", "row_kind"):
            _require_text(payload, field_name, path)
        _require_nonnegative_int(payload, "sample_index", path)
        adapter = _require_mapping(payload.get("adapter"), path, f"line {line_number}.adapter")
        _require_text(adapter, "path", path)
        _require_text(adapter, "weights_sha256", path)
        _require_text(adapter, "weights_filename", path)
        _require_mapping(adapter.get("config"), path, f"line {line_number}.adapter.config")
        conditions.add(cast("str", payload["condition"]))
        records.append(payload)
    expected = {"none", "steer:+", "placebo:+"}
    if conditions != expected:
        raise ValueError(
            f"readout {path} must contain exactly the required intervention conditions; "
            f"got {sorted(conditions)}"
        )
    summary_path = path.with_name("steering_summary.json")
    ledger_path = path.with_name(path.name + ".resume.json")
    summary = _require_object(json.loads(summary_path.read_text(encoding="utf-8")), summary_path)
    ledger = _require_object(json.loads(ledger_path.read_text(encoding="utf-8")), ledger_path)
    if summary.get("command") != "generate":
        raise ValueError(f"steering summary {summary_path} is not a native generate artifact")
    if summary.get("skipped_conditions"):
        raise ValueError(f"steering summary {summary_path} records skipped conditions")
    grouped: dict[str, int] = {}
    for line in lines:
        record = cast("dict[str, Any]", json.loads(line))
        key = cast("str", record["condition_key"])
        grouped[key] = grouped.get(key, 0) + 1
    summaries = _require_mapping(summary.get("conditions"), summary_path, "conditions")
    n_rows = _require_positive_int(summary, "n_rows", summary_path)
    n_samples = _require_positive_int(summary, "n_samples", summary_path)
    expected_per_condition = n_rows * n_samples
    if any(count != expected_per_condition for count in grouped.values()):
        raise ValueError(
            f"steering records {path} are partial; expected {expected_per_condition} records per "
            f"condition, got {grouped}"
        )
    if dict(summaries) != summarise_records(records):
        raise ValueError(f"steering summary {summary_path} disagrees with its records")
    identity = _require_mapping(ledger.get("identity"), ledger_path, "identity")
    completed = _require_mapping(ledger.get("completed_units"), ledger_path, "completed_units")
    if completed != grouped:
        raise ValueError(f"steering resume ledger {ledger_path} disagrees with its records")
    for field_name in (
        "command",
        "model_weights_identity",
        "seed",
        "placebo_seed",
        "n_samples",
        "max_new_tokens",
        "selected_target",
        "adapter",
        "diagnostic_profile",
        "row_manifest",
        "rendered_rows_sha256",
        "resolved_sampler",
    ):
        if identity.get(field_name) != summary.get(field_name):
            raise ValueError(f"steering resume ledger {ledger_path} misbinds {field_name}")


def _validate_json_readout(path: Path) -> None:
    if path.name not in _KNOWN_JSON_READOUTS:
        raise ValueError(f"readout {path} has unknown JSON artifact name")
    payload = _require_object(json.loads(path.read_text(encoding="utf-8")), path)
    if path.name in _ENDPOINT_SUMMARY_SECTIONS:
        _validate_endpoint_summary(path, payload)
    elif path.name == "ladder-manifest.json":
        _validate_ladder_manifest(path, payload)
    elif path.name == "cooperation_interp.json":
        _validate_geometry_report(path, payload)
    elif path.name == "cooperation_lens.json":
        _validate_lens_report(path, payload)
    else:
        raise AssertionError(f"unhandled JSON readout {path.name}")


@dataclass(frozen=True)
class AdapterArtifactIdentity:
    """Exact on-disk identity of the final adapter whose readouts permit pruning."""

    path: Path
    step: int
    weights_sha256: str
    config_sha256: str
    config: dict[str, Any]


def _adapter_identity(checkpoint: Path, step: int) -> AdapterArtifactIdentity:
    config_path = checkpoint / "adapter_config.json"
    weights_path = checkpoint / "adapter_model.safetensors"
    _require_object(json.loads(config_path.read_text(encoding="utf-8")), config_path)
    return AdapterArtifactIdentity(
        path=checkpoint.resolve(),
        step=step,
        weights_sha256=_sha256(weights_path),
        config_sha256=_sha256(config_path),
        config=cast(
            "dict[str, Any]",
            json.loads(json.dumps(adapter_config_identity(checkpoint), sort_keys=True)),
        ),
    )


def _assert_adapter_payload(
    adapter: Mapping[str, Any], *, path: Path, expected: AdapterArtifactIdentity
) -> None:
    adapter_path = _referenced_path(path, adapter.get("path"), field_name="adapter.path")
    if adapter_path != expected.path:
        raise ValueError(f"readout {path} references a different final adapter path")
    if adapter.get("weights_filename") != "adapter_model.safetensors":
        raise ValueError(f"readout {path} references the wrong adapter weights filename")
    if adapter.get("weights_sha256") != expected.weights_sha256:
        raise ValueError(f"readout {path} references different final adapter weights")
    config = _require_mapping(adapter.get("config"), path, "adapter.config")
    if config != expected.config:
        raise ValueError(f"readout {path} references a different final adapter config")
    _require_positive_int(adapter, "applied_adapter_weights", path)


def _validate_steering_identity(
    path: Path, base_model: str, expected: AdapterArtifactIdentity
) -> None:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        record = _require_object(json.loads(line), path)
        adapter = _require_mapping(record["adapter"], path, "adapter")
        _assert_adapter_payload(adapter, path=path, expected=expected)
        if expected.config.get("base_model_name_or_path") != base_model:
            raise ValueError(f"readout {path} contains an adapter for a different base model")
    summary_path = path.with_name("steering_summary.json")
    summary = _require_object(json.loads(summary_path.read_text(encoding="utf-8")), summary_path)
    summary_adapter = _require_mapping(summary.get("adapter"), summary_path, "adapter")
    _assert_adapter_payload(summary_adapter, path=summary_path, expected=expected)


def _validate_json_identity(  # noqa: C901, PLR0912, PLR0915 - explicit per-artifact binding
    path: Path,
    payload: Mapping[str, Any],
    base_model: str,
    expected: AdapterArtifactIdentity,
    final_arm: str,
) -> None:
    if path.name in _ENDPOINT_SUMMARY_SECTIONS:
        trace_path = _endpoint_trace_path(path)
        meta = read_eval_records(trace_path)[0]
        _require_text(meta, "model_identity", trace_path)
        if meta.get("model_weights_identity") != endpoint_model_identity(base_model):
            raise ValueError(f"readout {path} was produced from different base model weights")
        settings = _require_mapping(meta.get("settings"), trace_path, "settings")
        if settings.get("arm") != final_arm or settings.get("step") != expected.step:
            raise ValueError(f"readout {path} was produced for a different final arm or step")
        adapter_path = _referenced_path(
            trace_path, meta.get("model_adapter_dir"), field_name="model_adapter_dir"
        )
        if adapter_path != expected.path:
            raise ValueError(f"readout {path} was produced from a different final adapter path")
        if meta.get("adapter_weights_sha256") != expected.weights_sha256:
            raise ValueError(f"readout {path} was produced from different final adapter weights")
        if meta.get("adapter_config_sha256") != expected.config_sha256:
            raise ValueError(f"readout {path} was produced from a different final adapter config")
        return
    if path.name == "ladder-manifest.json":
        if payload["identity"]["base_model"] != base_model:
            raise ValueError(f"readout {path} was produced for a different base model")
        ladder = load_ladder(path.parent, require_capture_provenance=True)
        expected_labels = {f"{BASE_ARM}/step-{BASE_STEP}", f"{final_arm}/step-{expected.step}"}
        if {cell.label for cell in ladder.cells} != expected_labels:
            raise ValueError(
                f"readout {path} does not contain exactly the base and final capture cells"
            )
        final_cell = ladder.cell(final_arm, expected.step)
        if final_cell.adapter_weights_sha256 != expected.weights_sha256:
            raise ValueError(f"readout {path} captured different final adapter weights")
        if final_cell.applied_adapter_weights is None or final_cell.applied_adapter_weights <= 0:
            raise ValueError(f"readout {path} did not apply the final adapter")
        adapter_dir = _referenced_path(
            final_cell.source / "cell-manifest.json",
            final_cell.provenance.get("adapter_dir"),
            field_name="provenance.adapter_dir",
        )
        if adapter_dir != expected.path:
            raise ValueError(f"readout {path} captured a different final adapter path")
        if final_cell.provenance.get("adapter_config") != expected.config:
            raise ValueError(f"readout {path} captured a different final adapter config")
        return
    if path.name == "cooperation_interp.json":
        context = cast("Mapping[str, Any]", payload["context"])
        identities = cast("Mapping[str, Any]", context["cell_identities"])
        if any(
            cast("Mapping[str, Any]", cast("Mapping[str, Any]", entry)["identity"])["base_model"]
            != base_model
            for entry in identities.values()
        ):
            raise ValueError(f"readout {path} contains a cell for a different base model")
        expected_labels = {f"{BASE_ARM}/step-{BASE_STEP}", f"{final_arm}/step-{expected.step}"}
        if (
            set(identities) != expected_labels
            or set(cast("Sequence[str]", context["cells"])) != expected_labels
        ):
            raise ValueError(f"readout {path} does not identify exactly the base and final cells")
        final_entry = _require_mapping(
            identities[f"{final_arm}/step-{expected.step}"], path, "final cell identity"
        )
        if final_entry.get("adapter_weights_sha256") != expected.weights_sha256:
            raise ValueError(f"readout {path} used different final adapter weights")
        _require_positive_int(final_entry, "applied_adapter_weights", path)
        return
    if path.name == "cooperation_lens.json":
        if payload["base_model"] != base_model:
            raise ValueError(f"readout {path} was produced for a different base model")
        if (
            _referenced_path(path, payload["final_adapter"], field_name="final_adapter")
            != expected.path
        ):
            raise ValueError(f"readout {path} used a different final adapter path")
        if payload["final_arm"] != final_arm or payload["final_step"] != expected.step:
            raise ValueError(f"readout {path} used a different final arm or step")
        capture = _require_mapping(payload["construct_capture"], path, "construct_capture")
        if capture.get("final_adapter_weights_sha256") != expected.weights_sha256:
            raise ValueError(f"readout {path} used different final adapter weights")
        final_state = _require_mapping(
            cast("Mapping[str, Any]", payload["states"])["final"], path, "states.final"
        )
        if final_state.get("arm") != final_arm or final_state.get("step") != expected.step:
            raise ValueError(f"readout {path} final state names a different arm or step")


def _validate_readout_identity(
    path: Path,
    base_model: str,
    expected: AdapterArtifactIdentity,
    final_arm: str,
) -> None:
    """Bind native model and adapter identities to this run when the artifact records them."""
    if path.name == "steering_records.jsonl":
        _validate_steering_identity(path, base_model, expected)
        return
    payload = _require_object(json.loads(path.read_text(encoding="utf-8")), path)
    _validate_json_identity(path, payload, base_model, expected, final_arm)


def validate_readout(path: Path) -> None:
    """Validate the native schema of one durable cooperation readout.

    The sequence names native endpoint summaries, the capture ladder, geometry and lens reports,
    and intervention records. Their native formats are checked here so a zero-byte or arbitrary
    marker file cannot authorize destructive checkpoint pruning.
    """
    if path.name in _KNOWN_JSON_READOUTS:
        _validate_json_readout(path)
        return
    if path.name in _KNOWN_JSONL_READOUTS:
        _validate_steering_records(path)
        return
    raise ValueError(f"readout {path} has an unknown native artifact name or format")


def _resolve_readouts(run_root: Path, readouts: Sequence[Path]) -> tuple[Path, ...]:
    return tuple(path if path.is_absolute() else run_root / path for path in readouts)


def _validate_complete_bundle(  # noqa: C901, PLR0912, PLR0915 - authorization boundary
    run_root: Path,
    readouts: Sequence[Path],
    *,
    base_model: str,
    final_step: int,
) -> tuple[AdapterArtifactIdentity, str]:
    names = [path.name for path in readouts]
    if len(names) != len(set(names)) or set(names) != set(_REQUIRED_READOUT_NAMES):
        missing = sorted(_REQUIRED_READOUT_NAMES - set(names))
        unexpected = sorted(set(names) - _REQUIRED_READOUT_NAMES)
        raise ValueError(
            "pruning requires the complete cooperation readout bundle exactly once; "
            f"missing={missing}, unexpected={unexpected}, duplicates="
            f"{sorted(name for name in set(names) if names.count(name) > 1)}"
        )
    experiment_root = (run_root.parent if run_root.name == "training" else run_root).resolve()
    for readout in readouts:
        if readout.is_symlink() or not readout.is_file() or readout.stat().st_size == 0:
            raise ValueError(f"readout {readout} is not a durable regular file")
        try:
            readout.resolve().relative_to(experiment_root)
        except ValueError as error:
            raise ValueError(
                f"readout {readout} is outside experiment root {experiment_root}"
            ) from error
        validate_readout(readout)
    endpoint_paths = [path for path in readouts if path.name in _ENDPOINT_SUMMARY_SECTIONS]
    endpoint_identities = []
    model_identities: set[str] = set()
    for path in endpoint_paths:
        meta = read_eval_records(_endpoint_trace_path(path))[0]
        settings = _require_mapping(meta.get("settings"), path, "settings")
        endpoint_identities.append((settings.get("arm"), settings.get("step")))
        model_identities.add(_require_text(meta, "model_identity", path))
    if len(model_identities) != 1:
        raise ValueError(
            f"endpoint readouts use different model identity values: {sorted(model_identities)}"
        )
    if len(set(endpoint_identities)) != 1:
        raise ValueError(f"endpoint readouts disagree on final arm and step: {endpoint_identities}")
    final_arm_value, measured_step = endpoint_identities[0]
    if not isinstance(final_arm_value, str) or not final_arm_value.strip():
        raise ValueError("endpoint readouts do not identify a nonempty final arm")
    if measured_step != final_step:
        raise ValueError(
            f"endpoint readouts measure step {measured_step}, not selected final step {final_step}"
        )
    expected = _adapter_identity(run_root / f"checkpoint-{final_step}", final_step)
    resolved_base_weights_identity = resolve_weights_identity(base_model)
    for readout in readouts:
        _validate_readout_identity(readout, base_model, expected, final_arm_value)
    geometry_path = next(path for path in readouts if path.name == "cooperation_interp.json")
    geometry = _require_object(json.loads(geometry_path.read_text(encoding="utf-8")), geometry_path)
    geometry_context = _require_mapping(geometry["context"], geometry_path, "context")
    selected_target_path = _referenced_path(
        geometry_path, geometry_context["selected_target"], field_name="context.selected_target"
    )
    direction_manifest_path = _referenced_path(
        geometry_path,
        geometry_context["direction_manifest"],
        field_name="context.direction_manifest",
    )
    direction_manifest = _require_object(
        json.loads(direction_manifest_path.read_text(encoding="utf-8")),
        direction_manifest_path,
    )
    final_direction_paths = {
        name: _referenced_path(direction_manifest_path, value, field_name=f"final.{name}")
        for name, value in _require_mapping(
            direction_manifest["final"], direction_manifest_path, "final"
        ).items()
    }
    selected_target_wrapper = load_selected_target(
        selected_target_path,
        {name: load_direction_file(path) for name, path in final_direction_paths.items()},
        direction_paths=final_direction_paths,
    )
    steering_path = next(path for path in readouts if path.name == "steering_records.jsonl")
    steering_summary_path = steering_path.with_name("steering_summary.json")
    steering_summary = _require_object(
        json.loads(steering_summary_path.read_text(encoding="utf-8")), steering_summary_path
    )
    steering_target = _require_mapping(
        steering_summary.get("selected_target"), steering_summary_path, "selected_target"
    )
    if steering_target != selected_target_wrapper:
        raise ValueError("intervention readout used a different selected geometry target")
    if steering_summary.get("model_weights_identity") != resolved_base_weights_identity:
        raise ValueError("intervention readout used different base model weights")
    lens_path = next(path for path in readouts if path.name == "cooperation_lens.json")
    lens = _require_object(json.loads(lens_path.read_text(encoding="utf-8")), lens_path)
    if lens.get("base_weights_identity") != resolved_base_weights_identity:
        raise ValueError("lens readout used different base model weights")
    return expected, final_arm_value


def _authorization_payload(readouts: Sequence[Path]) -> dict[str, object]:
    return {
        "schema": "cooperation-retention-readout-authorization/v1",
        "readouts": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in sorted(readouts, key=lambda value: value.name)
        ],
    }


def _validate_authorization(  # noqa: PLR0913 - revalidates the complete destructive boundary
    path: Path,
    expected: Mapping[str, object],
    *,
    run_root: Path,
    readouts: Sequence[Path],
    base_model: str,
    final_step: int,
) -> None:
    found = _require_object(json.loads(path.read_text(encoding="utf-8")), path)
    if found != expected:
        raise ValueError(f"retention authorization {path} no longer matches its validated readouts")
    _validate_complete_bundle(
        run_root,
        readouts,
        base_model=base_model,
        final_step=final_step,
    )


def _selected_complete_paths(run_root: Path, steps: Sequence[int]) -> tuple[Path, ...]:
    snapshots = {snapshot.step: snapshot for snapshot in inspect_checkpoints(run_root)}
    paths: list[Path] = []
    for step in steps:
        snapshot = snapshots.get(step)
        checkpoint = run_root / f"checkpoint-{step}"
        if snapshot is None:
            raise ValueError(f"selected checkpoint step {step} does not exist below {run_root}")
        if not snapshot.complete:
            raise ValueError(
                f"selected checkpoint {checkpoint} is incomplete; missing {list(snapshot.missing)}"
            )
        paths.append(checkpoint)
    return tuple(paths)


@dataclass
class RuntimeAdapterCheckpointLoader:
    """Load retained adapters through the real base-model and PEFT runtime seams."""

    base_model_id: str
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    _base_model: PreTrainedModel | None = None
    _attached: PeftModel | None = None

    def __call__(self, checkpoint: Path) -> object:
        """Load or re-point the adapter and return the PEFT-wrapped model."""
        if self._base_model is None:
            self._base_model = load_adapter_base(
                self.base_model_id, dtype=self.dtype, device=self.device
            )
        attached = attach_adapter(
            self._base_model,
            checkpoint,
            self.base_model_id,
            existing=self._attached,
        )
        self._attached = attached.peft_model
        return attached


@dataclass(frozen=True)
class RetentionDryRun:
    """CPU-only retention evidence emitted by ``--dry-run``."""

    status: str
    run_root: str
    retained: tuple[CheckpointSnapshot, ...]
    adapter_steps: tuple[int, ...]
    durable_readouts: tuple[str, ...]
    checkpoint_bytes: int


def _normalise_steps(
    values: Sequence[int], *, name: str, required: bool = False
) -> tuple[int, ...]:
    steps = tuple(values)
    if required and not steps:
        raise ValueError(f"{name} must contain at least one step")
    raw_steps = cast("tuple[object, ...]", steps)
    if any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in raw_steps):
        raise ValueError(f"{name} must contain non-negative integer steps, got {steps}")
    if len(set(steps)) != len(steps):
        raise ValueError(f"{name} must contain each step exactly once")
    return tuple(sorted(steps))


def _validate_selection(
    run_root: Path,
    retain_steps: Sequence[int],
    adapter_steps: Sequence[int],
    base_model: str,
) -> tuple[Path, ...]:
    overlap = set(retain_steps) & set(adapter_steps)
    if overlap:
        raise ValueError(f"adapter steps cannot also be full retained steps: {sorted(overlap)}")
    selected = _selected_complete_paths(run_root, (*retain_steps, *adapter_steps))
    assert_one_adapter_config(selected)
    for checkpoint in selected:
        assert_adapter_matches_base(checkpoint, base_model)
    return selected


def _dry_run(
    run_root: Path,
    retain_steps: Sequence[int],
    adapter_steps: Sequence[int],
    readouts: Sequence[Path],
    base_model: str,
) -> RetentionDryRun:
    _validate_selection(run_root, retain_steps, adapter_steps, base_model)
    _validate_complete_bundle(
        run_root, readouts, base_model=base_model, final_step=max(retain_steps)
    )
    snapshots = inspect_checkpoints(run_root)
    retained = tuple(snapshot for snapshot in snapshots if snapshot.step in retain_steps)
    return RetentionDryRun(
        status="dry-run",
        run_root=str(run_root),
        retained=retained,
        adapter_steps=tuple(adapter_steps),
        durable_readouts=tuple(str(path) for path in readouts),
        checkpoint_bytes=checkpoint_disk_usage(run_root),
    )


def _json_payload(value: RetentionManifest | RetentionDryRun) -> dict[str, object]:
    if isinstance(value, RetentionDryRun):
        return {
            **asdict(value),
            "retained": [snapshot_payload(snapshot) for snapshot in value.retained],
        }
    return {
        "schema_version": value.schema_version,
        "status": value.status,
        "created_at": value.created_at,
        "retained": [snapshot_payload(snapshot) for snapshot in value.retained],
        "adapter_retained": [snapshot_payload(snapshot) for snapshot in value.adapter_retained],
        "pruned": [snapshot_payload(snapshot) for snapshot in value.pruned],
        "incomplete": [snapshot_payload(snapshot) for snapshot in value.incomplete],
        "checkpoint_bytes_before": value.checkpoint_bytes_before,
        "checkpoint_bytes_after": value.checkpoint_bytes_after,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_retention(args: argparse.Namespace) -> RetentionManifest | RetentionDryRun:
    """Run CPU validation or the production loader-backed retention operation."""
    run_root = args.run_root.resolve()
    if not run_root.is_dir() or run_root.is_symlink():
        raise ValueError(f"run root {run_root} is not a real directory")
    retain_steps = _normalise_steps(args.retain_step, name="--retain-step", required=True)
    adapter_steps = _normalise_steps(args.adapter_step, name="--adapter-step")
    readouts = _resolve_readouts(run_root, args.readout)
    if args.dry_run:
        result: RetentionManifest | RetentionDryRun = _dry_run(
            run_root, retain_steps, adapter_steps, readouts, args.base_model
        )
    else:
        selected = _validate_selection(run_root, retain_steps, adapter_steps, args.base_model)
        del selected
        expected, final_arm = _validate_complete_bundle(
            run_root,
            readouts,
            base_model=args.base_model,
            final_step=max(retain_steps),
        )
        del expected, final_arm
        authorization_payload = _authorization_payload(readouts)
        authorization_path = run_root / "retention_readout_authorization.json"
        _write_json(authorization_path, authorization_payload)
        loader = RuntimeAdapterCheckpointLoader(
            base_model_id=args.base_model,
            dtype=_DTYPES[args.dtype],
            device=torch.device(args.device),
        )
        result = prune_redundant_checkpoints(
            run_root,
            retain_steps=retain_steps,
            adapter_steps=adapter_steps,
            durable_readouts=(authorization_path,),
            load_checkpoint=loader,
            validate_readout=lambda path: _validate_authorization(
                path,
                authorization_payload,
                run_root=run_root,
                readouts=readouts,
                base_model=args.base_model,
                final_step=max(retain_steps),
            ),
        )
    if args.json_out is not None:
        _write_json(args.json_out, _json_payload(result))
    logger.info(
        "cooperation retention %s: run_root=%s retained=%s adapter_steps=%s",
        result.status,
        run_root,
        list(retain_steps),
        list(adapter_steps),
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit, run-owned retention CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--retain-step", type=int, action="append", default=[])
    parser.add_argument("--adapter-step", type=int, action="append", default=[])
    parser.add_argument("--readout", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bfloat16")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inventory and native readouts without loading a model or deleting checkpoints",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and execute the retention command."""
    args = build_parser().parse_args(argv)
    result = run_retention(args)
    print(json.dumps(_json_payload(result), indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
