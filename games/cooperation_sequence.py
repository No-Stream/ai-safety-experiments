"""Explicit, resumable orchestration for the cooperation-generalisation experiment.

This module contains experiment wiring only. Each operation is a real module CLI, and a phase is
complete only when a receipt binds its command, environment, input bytes, and native output bytes.
The CPU inspection path never constructs a model or touches CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import math
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

from games.cooperation_budget import (
    ModelMetadata,
    plan_budget,
)
from games.cooperation_budget import (
    measured_budget_is_complete as strict_budget_check,
)
from games.cooperation_eval_runner import (
    PROFILE_SERVING_SMOKE,
    SERVING_SMOKE_ENDPOINTS,
)
from games.cooperation_eval_runner import (
    build_parser as build_endpoint_parser,
)
from games.cooperation_eval_runner import (
    print_plan as print_endpoint_plan,
)
from games.cooperation_evals import load_behavior_manifest
from games.evals import rebuild_summary
from games.plans import UV, artifact_model_slug, plan_setting
from games.stage_runner import Stage, run_sequence

logger = logging.getLogger(__name__)

ENV_PREFIX = "GAMES_COOP_"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SMOKE_MODEL = "Qwen/Qwen3-0.6B"
SMOKE_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEFAULT_ARM = "cooperation-generalization-care-alpha-1"
DEFAULT_RUN_ID = "20260912-first-block"
DEFAULT_MAX_STEPS = 20
DEFAULT_COMPLETION_TOKENS = 32_768
DEFAULT_GROUP_SIZE = 8
DEFAULT_PROMPTS_PER_STEP = 8
DEFAULT_OVERSAMPLE = 1
INITIAL_STEP_LIMIT = 20
SCREEN_SAMPLES_PER_PROMPT = 8
NATURAL_PREFIX_LAYERS = "8,16,24"
NATURAL_PREFIX_REQUEST_COUNT = 4
NATURAL_PREFIX_STATE_COUNT = 2
NATURAL_PREFIX_LAYER_COUNT = 3
NATURAL_COMPARISON_ROW_COUNT = NATURAL_PREFIX_REQUEST_COUNT * NATURAL_PREFIX_LAYER_COUNT
RUNTIME_BASE = Path("docs/scratch/cooperation-generalization")
RUN_BASE = Path("artifacts/games/cooperation-generalization")
JLENS_ENV = "GAMES_COOP_JLENS_ROOT"


def _setting(name: str, default: str) -> str:
    return plan_setting(f"{ENV_PREFIX}{name}", default)


def runtime_root() -> Path:
    """Return the directory containing private runtime inputs."""
    return Path(os.environ.get(f"{ENV_PREFIX}RUNTIME_ROOT", RUNTIME_BASE))


def run_root() -> Path:
    """Return the directory containing this experiment run's artifacts."""
    override = os.environ.get(f"{ENV_PREFIX}RUN_ROOT")
    return Path(override) if override else RUN_BASE / _setting("RUN_ID", DEFAULT_RUN_ID)


def model_id() -> str:
    """Return the configured model identifier."""
    return _setting("MODEL", DEFAULT_MODEL)


def model_source() -> str:
    """Return the immutable local snapshot shared by every 9B loader."""
    default = Path(
        f"/var/tmp/cooperation-generalization-assets/qwen3.5-9b-{DEFAULT_MODEL_REVISION}"  # noqa: S108 -- intentional immutable local model cache path
    )
    return _setting("MODEL_PATH", str(default))


def smoke_model_source() -> str:
    """Return the immutable local 0.6B snapshot used only for plumbing."""
    default = Path(
        f"/var/tmp/cooperation-generalization-assets/qwen3-0.6b-{SMOKE_MODEL_REVISION}"  # noqa: S108 -- intentional immutable local model cache path
    )
    return _setting("SMOKE_MODEL_PATH", str(default))


def max_steps() -> int:
    """Return the configured training step count after applying the measured cap."""
    value = int(_setting("MAX_STEPS", str(DEFAULT_MAX_STEPS)))
    if value < 1:
        raise ValueError("GAMES_COOP_MAX_STEPS must be positive")
    if value > INITIAL_STEP_LIMIT:
        budget = measured_budget_path()
        if not _measured_cap_allows(budget, value):
            raise ValueError(
                f"max_steps={value} exceeds the initial {INITIAL_STEP_LIMIT}; {budget} must "
                "record a measured step cap at least that large"
            )
    return value


def completion_tokens() -> int:
    """Return the configured completion token cap."""
    value = int(_setting("COMPLETION_TOKENS", str(DEFAULT_COMPLETION_TOKENS)))
    if value < 1:
        raise ValueError("GAMES_COOP_COMPLETION_TOKENS must be positive")
    return value


def training_corpus_path() -> Path:
    """Return the private training corpus path."""
    return Path(_setting("TRAIN_CORPUS", str(runtime_root() / "training.jsonl")))


def training_manifest_path() -> Path:
    """Return the private training manifest path."""
    return Path(_setting("TRAIN_MANIFEST", str(runtime_root() / "training-manifest.json")))


def behavior_manifest_path() -> Path:
    """Return the private behavior roster path."""
    return Path(_setting("BEHAVIOR_MANIFEST", str(runtime_root() / "behavior-roster.json")))


def construct_stimuli_path() -> Path:
    """Return the private cooperation-construct stimulus path."""
    return Path(_setting("CONSTRUCT_STIMULI", str(runtime_root() / "construct-stimuli.jsonl")))


def lens_fit_smoke_path() -> Path:
    """Return the private smoke lens-fit stimulus path."""
    return Path(_setting("LENS_FIT_SMOKE", str(runtime_root() / "lens-fit-smoke.jsonl")))


def lens_fit_measurement_path() -> Path:
    """Return the private measurement lens-fit stimulus path."""
    return Path(
        _setting("LENS_FIT_MEASUREMENT", str(runtime_root() / "lens-fit-measurement.jsonl"))
    )


def lens_quality_path() -> Path:
    """Return the private lens-quality stimulus path."""
    return Path(_setting("LENS_QUALITY", str(runtime_root() / "lens-quality.jsonl")))


def steering_rows_path() -> Path:
    """Return the private steering diagnostic row manifest path."""
    return Path(_setting("STEERING_ROWS", str(runtime_root() / "steering-diagnostic-rows.json")))


def intervention_expectation_path() -> Path:
    """Return the private intervention expectation path."""
    return Path(
        _setting(
            "INTERVENTION_EXPECTATION",
            str(runtime_root() / "intervention-expectation.json"),
        )
    )


def natural_selection_path() -> Path:
    """Return the private natural-prefix selection manifest path."""
    return Path(
        _setting(
            "NATURAL_SELECTION",
            str(runtime_root() / "natural-prefix-selection.json"),
        )
    )


def model_metadata_path() -> Path:
    """Return the private model metadata path."""
    return Path(_setting("MODEL_METADATA", str(runtime_root() / "model-metadata.json")))


def survey_data_dir() -> Path:
    """Return the survey input directory."""
    return Path(_setting("SURVEY_DATA_DIR", "games/data/survey"))


def training_run_root() -> Path:
    """Return the training checkpoint directory."""
    return run_root() / "training"


def final_adapter_path() -> Path:
    """Return the final adapter checkpoint path."""
    return training_run_root() / f"checkpoint-{max_steps()}"


def measurements_path() -> Path:
    """Return the measured timings artifact path."""
    return Path(_setting("MEASUREMENTS", str(run_root() / "measurements.json")))


def measured_budget_path() -> Path:
    """Return the strict measured budget artifact path."""
    return Path(_setting("BUDGET", str(run_root() / "measured-budget.json")))


def run_identity_path() -> Path:
    """Return the frozen experiment identity written before the first GPU phase."""
    return run_root() / "run-identity.json"


def preflight_identity_path() -> Path:
    """Return the replaceable CPU preparation identity."""
    return run_root() / "run-identity.preflight.json"


def receipt_path(name: str) -> Path:
    """Return the receipt path for a phase name."""
    return run_root() / "receipts" / f"{name}.json"


@dataclass(frozen=True, slots=True)
class Operation:
    """One native command and the non-empty artifacts it must produce."""

    name: str
    argv: tuple[str, ...]
    artifacts: tuple[Path, ...]
    env: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class Phase:
    """One ordered, identity-bound unit exposed to ``stage_runner``."""

    name: str
    operations: tuple[Operation, ...]
    inputs: tuple[Path, ...]
    receipt_path: Path
    needs_gpu: bool


def _path_digest(path: Path) -> str:
    if not path.exists():
        return f"missing:{path}"
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    entries = [
        f"{child.relative_to(path).as_posix()}:{hashlib.sha256(child.read_bytes()).hexdigest()}"
        for child in sorted(path.rglob("*"))
        if child.is_file()
    ]
    return (
        "empty:" + str(path)
        if not entries
        else hashlib.sha256("\n".join(entries).encode()).hexdigest()
    )


def phase_identity(phase: Phase) -> dict[str, object]:
    """Return the content identity for a phase and all of its inputs."""
    return {
        "schema": "cooperation-generalization-phase/v1",
        "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "phase": phase.name,
        "operations": [
            {
                "name": operation.name,
                "argv": list(operation.argv),
                "env": dict(sorted((operation.env or {}).items())),
            }
            for operation in phase.operations
        ],
        "inputs": {str(path): _path_digest(path) for path in phase.inputs},
    }


def _phase_outputs(phase: Phase) -> tuple[Path, ...]:
    return tuple(path for operation in phase.operations for path in operation.artifacts)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast("dict[str, Any]", payload)


def _operation_module(operation: Operation) -> str | None:
    try:
        module_flag = operation.argv.index("-m")
    except ValueError:
        return None
    return operation.argv[module_flag + 1] if module_flag + 1 < len(operation.argv) else None


def _operation_argument(operation: Operation, flag: str) -> str:
    try:
        flag_index = operation.argv.index(flag)
    except ValueError as error:
        raise RuntimeError(
            f"native operation {operation.name!r} omits required argument {flag}"
        ) from error
    if flag_index + 1 >= len(operation.argv):
        raise RuntimeError(f"native operation {operation.name!r} has no value for {flag}")
    return operation.argv[flag_index + 1]


def _require_text(payload: Mapping[str, Any], field: str, *, artifact: Path) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"semantic artifact {artifact} has no non-empty {field}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"semantic artifact {path}:{line_number} is not a JSON object")
        rows.append(cast("dict[str, Any]", value))
    if not rows:
        raise RuntimeError(f"semantic artifact {path} contains no JSONL records")
    return rows


def _validate_endpoint_artifacts(operation: Operation) -> None:  # noqa: C901, PLR0912 -- explicit native endpoint schema gate
    section_by_endpoint = {
        "behavior": "game-behavior",
        "allocation": "self-report",
        "full-context": "self-report",
        "core-survey": "self-report",
        "prosocialness": "self-report",
        "local-dt": "dt-probes",
        "serving-smoke-self-prediction": "self-report",
        "serving-smoke-forecast": "self-report",
    }
    traces = [path for path in operation.artifacts if path.suffix == ".jsonl"]
    for trace in traces:
        endpoint = trace.stem
        expected_section = section_by_endpoint.get(endpoint)
        if expected_section is None:
            raise RuntimeError(f"semantic endpoint artifact has unknown endpoint name {endpoint!r}")
        summary_path = trace.with_suffix(".summary.json")
        if summary_path not in operation.artifacts:
            raise RuntimeError(f"semantic endpoint artifact omits native summary {summary_path}")
        rows = _read_jsonl_objects(trace)
        meta, records = rows[0], rows[1:]
        if meta.get("record") != "meta" or meta.get("cooperation_endpoint") != endpoint:
            raise RuntimeError(
                f"semantic endpoint artifact {trace} has invalid native endpoint identity"
            )
        _require_text(meta, "model_identity", artifact=trace)
        _require_text(meta, "model_weights_identity", artifact=trace)
        request_count = meta.get("cooperation_request_count")
        if (
            not isinstance(request_count, int)
            or isinstance(request_count, bool)
            or request_count < 1
        ):
            raise RuntimeError(
                f"semantic endpoint artifact {trace} has invalid cooperation_request_count"
            )
        if len(records) != request_count:
            raise RuntimeError(
                f"semantic endpoint artifact {trace} has {len(records)} records for "
                f"cooperation_request_count={request_count}"
            )
        if any(record.get("record") != expected_section for record in records):
            raise RuntimeError(
                f"semantic endpoint artifact {trace} contains the wrong native record kind"
            )
        if any(not isinstance(record.get("completion"), str) for record in records):
            raise RuntimeError(
                f"semantic endpoint artifact {trace} contains a record without completion text"
            )
        settings = meta.get("settings")
        if not isinstance(settings, dict) or settings.get("endpoint") != endpoint:
            raise RuntimeError(f"semantic endpoint artifact {trace} has invalid settings identity")
        for field in ("arm", "step", "sampler", "max_new_tokens", "backend"):
            if field not in settings:
                raise RuntimeError(f"semantic endpoint artifact {trace} settings omit {field}")
        summary = _read_object(summary_path)
        if summary != rebuild_summary(trace):
            raise RuntimeError(
                f"semantic endpoint summary {summary_path} disagrees with its native trace"
            )
        section = summary.get(expected_section)
        if not isinstance(section, dict) or section.get("n_records") != request_count:
            raise RuntimeError(
                f"semantic endpoint summary {summary_path} does not close all {request_count} records"
            )


def _validate_natural_materialization(operation: Operation) -> None:  # noqa: C901 -- explicit native artifact identity gate
    stimuli_path, sidecar_path = operation.artifacts
    rows = _read_jsonl_objects(stimuli_path)
    sidecar = _read_object(sidecar_path)
    stimulus_ids = sidecar.get("stimulus_ids")
    selection = sidecar.get("selection_manifest")
    if (
        not isinstance(stimulus_ids, list)
        or len(stimulus_ids) != NATURAL_PREFIX_REQUEST_COUNT
        or len(set(stimulus_ids)) != NATURAL_PREFIX_REQUEST_COUNT
    ):
        raise TypeError(
            f"semantic natural-prefix sidecar {sidecar_path} must name exactly four stimuli"
        )
    if not isinstance(selection, dict):
        raise TypeError(f"semantic natural-prefix sidecar {sidecar_path} has no selection manifest")
    request_ids = selection.get("request_ids")
    layers = selection.get("layers")
    if (
        not isinstance(request_ids, list)
        or len(request_ids) != NATURAL_PREFIX_REQUEST_COUNT
        or len(set(request_ids)) != NATURAL_PREFIX_REQUEST_COUNT
    ):
        raise RuntimeError(
            f"semantic natural-prefix sidecar {sidecar_path} must name exactly four requests"
        )
    if layers != [8, 16, 24]:
        raise RuntimeError(
            f"semantic natural-prefix sidecar {sidecar_path} must name layers 8, 16, 24"
        )
    for field in ("stimuli_sha256", "rendered_sha256", "tokenizer_identity"):
        _require_text(sidecar, field, artifact=sidecar_path)
    _require_text(selection, "sha256", artifact=sidecar_path)
    natural_state = _operation_argument(operation, "--natural-state")
    rollout_ids_by_state = selection.get("rollout_ids_by_state")
    if (
        sidecar.get("natural_state") != natural_state
        or not isinstance(rollout_ids_by_state, dict)
        or natural_state not in rollout_ids_by_state
    ):
        raise RuntimeError(f"semantic natural-prefix sidecar {sidecar_path} misbinds its state")
    records_path = Path(_operation_argument(operation, "--records"))
    selection_path = Path(_operation_argument(operation, "--natural-selection-manifest"))
    if sidecar.get("records_file") != str(records_path) or sidecar.get("stimuli_file") != str(
        stimuli_path
    ):
        raise RuntimeError(
            f"semantic natural-prefix sidecar {sidecar_path} is not bound to its inputs"
        )
    if sidecar.get("records_sha256") != hashlib.sha256(records_path.read_bytes()).hexdigest():
        raise RuntimeError(
            f"semantic natural-prefix sidecar {sidecar_path} has source-record digest drift"
        )
    if sidecar.get("selection_manifest_path") != str(selection_path):
        raise RuntimeError(
            f"semantic natural-prefix sidecar {sidecar_path} has selection path drift"
        )
    row_ids = [row.get("id") for row in rows]
    row_requests = [row.get("metadata", {}).get("request_id") for row in rows]
    if row_ids != stimulus_ids or row_requests != request_ids:
        raise RuntimeError(
            f"semantic natural-prefix rows {stimuli_path} do not match their sidecar identity"
        )
    for row in rows:
        metadata = row.get("metadata")
        if (
            row.get("set") != "natural-prefix"
            or row.get("side") != "N"
            or not isinstance(row.get("assistant_prefix"), str)
            or not isinstance(metadata, dict)
            or metadata.get("measurement_boundary") != "pre_action"
            or metadata.get("action_commitment_present") is not False
            or not isinstance(metadata.get("selected_positions"), list)
            or metadata.get("natural_state") != natural_state
            or metadata.get("rollout_id") != rollout_ids_by_state[natural_state]
        ):
            raise RuntimeError(
                f"semantic natural-prefix row in {stimuli_path} is not a native pre-action stimulus"
            )


def _validate_natural_capture(operation: Operation) -> None:  # noqa: C901, PLR0912 -- explicit native capture schema gate
    manifests = [
        path for path in operation.artifacts if path.name == "natural-prefix-manifest.json"
    ]
    if len(manifests) != NATURAL_PREFIX_STATE_COUNT:
        raise RuntimeError("semantic natural-prefix capture must contain base and final manifests")
    for manifest_path in manifests:
        payload = _read_object(manifest_path)
        identity = payload.get("identity")
        selection = payload.get("selection_manifest")
        records = payload.get("selection_records")
        states = payload.get("states")
        stimulus_ids = payload.get("stimulus_ids")
        if not all(isinstance(value, dict) for value in (identity, selection, states)):
            raise TypeError(
                f"semantic natural-prefix manifest {manifest_path} omits native objects"
            )
        if not isinstance(records, list) or not isinstance(stimulus_ids, list):
            raise TypeError(f"semantic natural-prefix manifest {manifest_path} omits native rows")
        identity = cast("dict[str, Any]", identity)
        selection = cast("dict[str, Any]", selection)
        layers = identity.get("natural_prefix_layers")
        if layers != [8, 16, 24] or selection.get("layers") != layers:
            raise RuntimeError(
                f"semantic natural-prefix manifest {manifest_path} has invalid layers"
            )
        layers = cast("list[int]", layers)
        hidden_size = identity.get("hidden_size")
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size < 1:
            raise RuntimeError(
                f"semantic natural-prefix manifest {manifest_path} has invalid hidden size"
            )
        request_ids = selection.get("request_ids")
        if (
            not isinstance(request_ids, list)
            or len(request_ids) != NATURAL_PREFIX_REQUEST_COUNT
            or len(set(request_ids)) != NATURAL_PREFIX_REQUEST_COUNT
        ):
            raise RuntimeError(
                f"semantic natural-prefix manifest {manifest_path} must name four requests"
            )
        if (
            len(stimulus_ids) != NATURAL_PREFIX_REQUEST_COUNT
            or len(set(stimulus_ids)) != NATURAL_PREFIX_REQUEST_COUNT
        ):
            raise RuntimeError(
                f"semantic natural-prefix manifest {manifest_path} must name four stimuli"
            )
        if [record.get("stimulus_id") for record in records] != stimulus_ids:
            raise RuntimeError(
                f"semantic natural-prefix manifest {manifest_path} has misbound stimuli"
            )
        if [record.get("request_id") for record in records] != request_ids:
            raise RuntimeError(
                f"semantic natural-prefix manifest {manifest_path} has misbound requests"
            )
        selection_digest = _require_text(selection, "sha256", artifact=manifest_path)
        if identity.get("natural_selection_manifest_sha256") != selection_digest:
            raise RuntimeError(
                f"semantic natural-prefix manifest {manifest_path} has selection digest drift"
            )
        expected_shapes = {
            stimulus_id: [3, len(layers), hidden_size] for stimulus_id in stimulus_ids
        }
        if states != expected_shapes:
            raise RuntimeError(
                f"semantic natural-prefix manifest {manifest_path} has invalid state shapes"
            )
        activation_path = manifest_path.parent / "natural-prefix-activations.safetensors"
        if activation_path not in operation.artifacts:
            raise RuntimeError(
                f"semantic natural-prefix manifest {manifest_path} omits its activation artifact"
            )


def _validate_selected_target(  # noqa: C901 -- explicit direction-selection identity gate
    report_path: Path, manifest_path: Path, selection_path: Path
) -> None:
    directions = _read_object(manifest_path)
    required_directions = {"costly-other-regard", "decision-dependence", "trained-displacement"}
    if set(directions) != {"base", "final"} or any(
        not isinstance(directions[state], dict) or set(directions[state]) != required_directions
        for state in ("base", "final")
    ):
        raise RuntimeError(
            f"semantic direction manifest {manifest_path} omits required native directions"
        )
    selection = _read_object(selection_path)
    required_fields = {
        "schema",
        "version",
        "direction",
        "target_construct",
        "direction_path",
        "direction_sha256",
        "layer",
        "magnitude",
        "alpha_multiplier",
        "calibration_metric",
        "calibration_rationale",
        "expected_effect",
        "expectation_recorded_before_intervention",
        "geometry_report_path",
        "geometry_report_sha256",
        "exploratory",
    }
    if (
        set(selection) != required_fields
        or selection.get("schema") != "cooperation-generalization-selected-target/v1"
    ):
        raise RuntimeError(
            f"semantic selected target {selection_path} is not the native v1 artifact"
        )
    if (
        selection.get("version") != 1
        or selection.get("expectation_recorded_before_intervention") is not True
    ):
        raise RuntimeError(
            f"semantic selected target {selection_path} has invalid version or expectation"
        )
    if selection.get("exploratory") is not True:
        raise RuntimeError(
            f"semantic selected target {selection_path} must record exploratory selection"
        )
    if Path(str(selection.get("geometry_report_path"))).resolve() != report_path.resolve():
        raise RuntimeError(
            f"semantic selected target {selection_path} is not bound to its geometry report path"
        )
    if (
        selection.get("geometry_report_sha256")
        != hashlib.sha256(report_path.read_bytes()).hexdigest()
    ):
        raise RuntimeError(
            f"semantic selected target {selection_path} is not bound to its geometry report"
        )
    direction_name = selection.get("direction")
    if direction_name not in required_directions:
        raise RuntimeError(f"semantic selected target {selection_path} names an unknown direction")
    if (
        direction_name != "costly-other-regard"
        or selection.get("target_construct") != "costly-other-regard"
    ):
        raise RuntimeError(
            f"semantic selected target {selection_path} does not bind the first-run target construct"
        )
    direction_path = Path(str(selection.get("direction_path")))
    if (
        directions["final"].get(direction_name) != str(direction_path)
        or not direction_path.is_file()
    ):
        raise RuntimeError(
            f"semantic selected target {selection_path} is not bound to its final direction"
        )
    if selection.get("direction_sha256") != hashlib.sha256(direction_path.read_bytes()).hexdigest():
        raise RuntimeError(
            f"semantic selected target {selection_path} has a direction digest mismatch"
        )
    layer = selection.get("layer")
    magnitude = selection.get("magnitude")
    alpha_multiplier = selection.get("alpha_multiplier")
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
        raise RuntimeError(f"semantic selected target {selection_path} has invalid layer")
    if (
        not isinstance(magnitude, int | float)
        or isinstance(magnitude, bool)
        or not math.isfinite(float(magnitude))
        or float(magnitude) <= 0
        or not isinstance(alpha_multiplier, int | float)
        or isinstance(alpha_multiplier, bool)
        or not math.isfinite(float(alpha_multiplier))
        or float(alpha_multiplier) <= 0
    ):
        raise RuntimeError(f"semantic selected target {selection_path} has invalid magnitude")


def _validate_geometry_artifacts(operation: Operation) -> None:  # noqa: C901 -- explicit tiny/final geometry contracts
    report_path, manifest_path, selection_path = operation.artifacts
    report = _read_object(report_path)
    for field in (
        "context",
        "confound_checks",
        "projections",
        "displacements",
        "calibration_selection",
    ):
        if field not in report:
            raise RuntimeError(f"semantic geometry report {report_path} omits native field {field}")
    _validate_selected_target(report_path, manifest_path, selection_path)
    if "--natural-prefix-artifact" not in operation.argv:
        cross_check = report.get("natural_prefix_cross_check")
        if operation.name != "fit tiny 9B construct geometry":
            raise RuntimeError(
                "only the named tiny 9B geometry smoke may omit natural-prefix artifacts"
            )
        if not isinstance(cross_check, dict) or cross_check.get("supported") is not False:
            raise RuntimeError(
                f"semantic tiny geometry report {report_path} must record unsupported natural prefixes"
            )
        return
    cross_check = report.get("natural_prefix_cross_check")
    if not isinstance(cross_check, dict) or cross_check.get("supported") is not True:
        raise RuntimeError(
            f"semantic geometry report {report_path} has no supported natural-prefix cross-check"
        )
    base_to_final = cross_check.get("base_to_final")
    if not isinstance(base_to_final, dict) or base_to_final.get("available") is not True:
        raise RuntimeError(
            f"semantic geometry report {report_path} has no available base-to-final natural read"
        )
    if base_to_final.get("n_common_requests") != NATURAL_PREFIX_REQUEST_COUNT:
        raise RuntimeError(
            f"semantic geometry report {report_path} must join exactly four common requests"
        )
    if base_to_final.get("n_common_layers") != NATURAL_PREFIX_LAYER_COUNT:
        raise RuntimeError(
            f"semantic geometry report {report_path} must join exactly three common layers"
        )
    rows = base_to_final.get("rows")
    if not isinstance(rows, list) or len(rows) != NATURAL_COMPARISON_ROW_COUNT:
        raise RuntimeError(
            f"semantic geometry report {report_path} must contain all 12 natural comparison rows"
        )


def _validate_lens_artifact(operation: Operation) -> None:
    report_path = operation.artifacts[0]
    report = _read_object(report_path)
    _require_text(report, "base_weights_identity", artifact=report_path)
    states = report.get("states")
    if not isinstance(states, dict) or set(states) != {"base", "final"}:
        raise RuntimeError(f"semantic lens report {report_path} must contain base and final states")
    for state in ("base", "final"):
        entry = states[state]
        if not isinstance(entry, dict) or entry.get("readout_kind") != "model-specific":
            raise RuntimeError(
                f"semantic lens report {report_path} has invalid {state} model-specific readout"
            )
        if entry.get("lens_state") != state or entry.get("applied_to_state") != state:
            raise RuntimeError(f"semantic lens report {report_path} misbinds its {state} readout")
    shared = report.get("shared_base_lens_coordinate_sensitivity")
    expected_shared = {
        "readout_kind": "shared-base-coordinate-sensitivity",
        "approximation": True,
        "lens_state": "base",
        "applied_to_state": "final",
    }
    if not isinstance(shared, dict) or any(
        shared.get(key) != value for key, value in expected_shared.items()
    ):
        raise RuntimeError(
            f"semantic lens report {report_path} has no valid shared-base coordinate read"
        )


def _validate_steering_artifacts(operation: Operation) -> None:  # noqa: C901 -- explicit steering resume schema gate
    records_path, ledger_path, summary_path = operation.artifacts
    records = _read_jsonl_objects(records_path)
    ledger = _read_object(ledger_path)
    summary = _read_object(summary_path)
    if summary.get("command") != "generate":
        raise RuntimeError(
            f"semantic steering summary {summary_path} is not a native generate artifact"
        )
    _require_text(summary, "model_weights_identity", artifact=summary_path)
    expected_token_cap = int(_operation_argument(operation, "--max-new-tokens"))
    if summary.get("max_new_tokens") != expected_token_cap:
        raise RuntimeError(
            f"semantic steering summary {summary_path} has the wrong completion budget"
        )
    if not isinstance(summary.get("selected_target"), dict):
        raise TypeError(f"semantic steering summary {summary_path} has no selected target identity")
    conditions = summary.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        raise RuntimeError(f"semantic steering summary {summary_path} has no condition summaries")
    grouped: dict[str, int] = {}
    for record in records:
        key = record.get("condition_key")
        if not isinstance(key, str) or not key:
            raise RuntimeError(f"semantic steering records {records_path} omit condition identity")
        if not isinstance(record.get("response_text"), str):
            raise TypeError(f"semantic steering records {records_path} omit response text")
        grouped[key] = grouped.get(key, 0) + 1
    if set(grouped) != set(conditions) or any(
        not isinstance(conditions[key], dict) or conditions[key].get("n_completions") != count
        for key, count in grouped.items()
    ):
        raise RuntimeError(f"semantic steering summary {summary_path} disagrees with its records")
    ledger_identity = ledger.get("identity")
    completed_units = ledger.get("completed_units")
    if not isinstance(ledger_identity, dict) or not isinstance(completed_units, dict):
        raise TypeError(f"semantic steering resume ledger {ledger_path} has invalid structure")
    if completed_units != grouped:
        raise RuntimeError(
            f"semantic steering resume ledger {ledger_path} disagrees with its records"
        )
    for field in (
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
        if ledger_identity.get(field) != summary.get(field):
            raise RuntimeError(f"semantic steering resume ledger {ledger_path} misbinds {field}")


def _validate_operation_artifacts(operation: Operation) -> None:
    module = _operation_module(operation)
    if module == "games.cooperation_eval_runner":
        _validate_endpoint_artifacts(operation)
    elif module == "games.interp_capture" and "--build-natural-prefix-stimuli" in operation.argv:
        _validate_natural_materialization(operation)
    elif module == "games.interp_capture" and "--natural-prefix-layers" in operation.argv:
        _validate_natural_capture(operation)
    elif module == "games.cooperation_interp":
        _validate_geometry_artifacts(operation)
    elif module == "games.cooperation_lens":
        _validate_lens_artifact(operation)
    elif module == "games.interp_steering":
        _validate_steering_artifacts(operation)


def _load_phase_receipt(phase: Phase) -> dict[str, Any] | None:
    if not phase.receipt_path.is_file():
        return None
    payload = _read_object(phase.receipt_path)
    if payload.get("phase_identity") != phase_identity(phase):
        raise ValueError(f"phase receipt identity mismatch at {phase.receipt_path}")
    outputs = {str(path): _path_digest(path) for path in _phase_outputs(phase)}
    if any(value.startswith(("missing:", "empty:")) for value in outputs.values()):
        return None
    if payload.get("outputs") != outputs:
        raise ValueError(f"phase receipt output mismatch at {phase.receipt_path}")
    for operation in phase.operations:
        _validate_operation_artifacts(operation)
    return payload


def phase_is_complete(phase: Phase) -> bool:
    """Return true only when the receipt still matches all inputs and outputs."""
    return _load_phase_receipt(phase) is not None


def _directory_bytes(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _write_receipt(
    phase: Phase, *, elapsed_seconds: float, operation_seconds: Mapping[str, float]
) -> None:
    missing = [
        str(path)
        for path in _phase_outputs(phase)
        if not path.exists() or (path.is_file() and path.stat().st_size == 0)
    ]
    if missing:
        raise RuntimeError(f"phase {phase.name!r} produced no usable artifact at {missing}")
    payload = {
        "phase_identity": phase_identity(phase),
        "outputs": {str(path): _path_digest(path) for path in _phase_outputs(phase)},
        "elapsed_seconds": elapsed_seconds,
        "operation_seconds": dict(operation_seconds),
        "output_bytes": {
            str(path): path.stat().st_size if path.is_file() else _directory_bytes(path)
            for path in _phase_outputs(phase)
        },
        "completed_at": datetime.now(tz=UTC).isoformat(),
    }
    phase.receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = phase.receipt_path.with_name(f".{phase.receipt_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(phase.receipt_path)


def execute_phase(phase: Phase, *, runner: Callable[[Operation], int] | None = None) -> bool:
    """Execute a phase once, leaving partial native artifacts available for native resume."""
    if phase.needs_gpu:
        freeze_run_identity()
    if phase_is_complete(phase):
        logger.info("phase already complete: %s", phase.name)
        return False
    started = time.monotonic()
    timings: dict[str, float] = {}
    for operation in phase.operations:
        operation_started = time.monotonic()
        if runner is None:
            environment = None if operation.env is None else {**os.environ, **operation.env}
            result = subprocess.run(operation.argv, check=False, env=environment)  # noqa: S603
            returncode = result.returncode
        else:
            returncode = runner(operation)
        if returncode:
            raise RuntimeError(f"operation {operation.name!r} failed with exit {returncode}")
        timings[operation.name] = time.monotonic() - operation_started
        missing = [
            str(path)
            for path in operation.artifacts
            if not path.exists() or (path.is_file() and path.stat().st_size == 0)
        ]
        if missing:
            raise RuntimeError(
                f"operation {operation.name!r} exited 0 but produced no usable artifact at {missing}"
            )
        _validate_operation_artifacts(operation)
    _write_receipt(
        phase,
        elapsed_seconds=time.monotonic() - started,
        operation_seconds=timings,
    )
    return True


def measured_budget_is_complete(path: Path) -> bool:
    """Require the native measured-budget contract before authorizing training."""
    try:
        return strict_budget_check(path)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _measured_cap_allows(path: Path, requested_steps: int) -> bool:
    if not measured_budget_is_complete(path):
        return False
    payload = _read_object(path)
    cap = payload.get("measured_cap_steps", payload.get("measured_max_steps"))
    return isinstance(cap, int) and not isinstance(cap, bool) and cap >= requested_steps


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number} must be a JSON object")
        rows.append(cast("dict[str, Any]", row))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _identity_text(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list | tuple) and value:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    raise ValueError(f"invalid group identity {value!r}")


def _stimulus_membership(path: Path) -> tuple[set[str], set[str]]:
    groups: set[str] = set()
    pairs: set[str] = set()
    for row in _jsonl_rows(path):
        pair = row.get("pair_id")
        metadata = row.get("metadata")
        if not isinstance(pair, str) or not pair:
            raise ValueError(f"{path} row has no pair_id")
        if not isinstance(metadata, dict):
            raise TypeError(f"{path} row {pair!r} has no metadata object")
        group = metadata.get("scenario_group")
        if not isinstance(group, str) or not group:
            raise ValueError(f"{path} row {pair!r} has no metadata.scenario_group")
        pairs.add(pair)
        groups.add(group)
    return groups, pairs


def build_membership() -> dict[str, dict[str, list[str]]]:
    """Read all four private memberships and prove global group/pair isolation."""
    training = _read_object(training_manifest_path())
    train_groups = {_identity_text(value) for value in training.get("training_group_ids", [])}
    if not train_groups:
        raise ValueError("training manifest has no training_group_ids")
    behavior = load_behavior_manifest(behavior_manifest_path())
    behavior_groups = set(behavior.scenario_group_ids)
    behavior_pairs = {pair.pair_id for pair in behavior.pairs}
    if behavior.allocation is not None:
        behavior_pairs.add(behavior.allocation.diagnostic_id)
    construct_groups, construct_pairs = _stimulus_membership(construct_stimuli_path())
    fit_groups, fit_pairs = _stimulus_membership(lens_fit_measurement_path())
    quality_groups, quality_pairs = _stimulus_membership(lens_quality_path())
    lens_groups = fit_groups | quality_groups
    lens_pairs = fit_pairs | quality_pairs
    memberships = {
        "train": {"group_ids": train_groups, "pair_ids": set()},
        "behavior": {"group_ids": behavior_groups, "pair_ids": behavior_pairs},
        "construct": {"group_ids": construct_groups, "pair_ids": construct_pairs},
        "lens": {"group_ids": lens_groups, "pair_ids": lens_pairs},
    }
    roles = tuple(memberships)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            for key in ("group_ids", "pair_ids"):
                overlap = memberships[left][key] & memberships[right][key]
                if overlap:
                    raise ValueError(f"{left}/{right} {key} overlap: {sorted(overlap)}")
    return {
        role: {key: sorted(values) for key, values in membership.items()}
        for role, membership in memberships.items()
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _union_membership(
    memberships: Mapping[str, Mapping[str, Sequence[str]]], roles: Sequence[str]
) -> dict[str, list[str]]:
    return {
        key: sorted({value for role in roles for value in memberships[role][key]})
        for key in ("group_ids", "pair_ids")
    }


def prepare_runtime_outputs() -> None:
    """Write categorized membership, consumer-specific exclusions, and CPU plan JSON."""
    memberships = build_membership()
    _write_json(runtime_root() / "group-membership.json", memberships)
    _write_json(
        runtime_root() / "construct-external-reservations.json",
        _union_membership(memberships, ("train", "behavior", "lens")),
    )
    _write_json(
        runtime_root() / "lens-external-reservations.json",
        _union_membership(memberships, ("train", "behavior", "construct")),
    )
    _write_json(preflight_identity_path(), run_identity())
    _write_json(run_root() / "static-plan.json", static_plan())


def freeze_run_identity() -> None:
    """Freeze the prepared identity once, immediately before the first GPU phase."""
    prepared = _read_object(preflight_identity_path())
    current = run_identity()
    if prepared != current:
        raise ValueError("CPU preflight identity is stale; rerun --prepare before GPU execution")
    frozen_path = run_identity_path()
    if frozen_path.exists():
        if _read_object(frozen_path) != current:
            raise ValueError(
                "frozen run identity differs from current inputs; start a distinct run root"
            )
        return
    _write_json(frozen_path, current)


def run_identity() -> dict[str, object]:
    """Return the immutable identity binding for this experiment run."""
    selection = _read_object(natural_selection_path())
    expected_rollouts = {
        "base": "base/step-0",
        "final": f"{DEFAULT_ARM}/step-{max_steps()}",
    }
    if selection.get("rollout_ids_by_state") != expected_rollouts:
        raise ValueError(
            "natural-prefix rollout_ids_by_state must match the planned checkpoints: "
            f"expected {expected_rollouts}"
        )
    inputs = (
        training_manifest_path(),
        training_corpus_path(),
        behavior_manifest_path(),
        construct_stimuli_path(),
        lens_fit_measurement_path(),
        lens_quality_path(),
        steering_rows_path(),
        intervention_expectation_path(),
        natural_selection_path(),
        model_metadata_path(),
    )
    return {
        "schema": "cooperation-generalization-run/v2",
        "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": model_id(),
        "model_revision": DEFAULT_MODEL_REVISION,
        "arm": DEFAULT_ARM,
        "max_steps": max_steps(),
        "completion_tokens": completion_tokens(),
        "sampler": "training-distribution",
        "runtime_inputs": {str(path): _path_digest(path) for path in inputs},
    }


def static_plan() -> dict[str, object]:
    """Calculate counts and disk from runtime membership and inspected model metadata."""
    metadata = ModelMetadata.from_mapping(_read_object(model_metadata_path()))
    budget = plan_budget(
        training_manifest=training_manifest_path(),
        training_corpus=training_corpus_path(),
        behavior_manifest=behavior_manifest_path(),
        survey_data_dir=survey_data_dir(),
        construct_stimuli=(construct_stimuli_path(),),
        lens_stimuli=(lens_fit_measurement_path(), lens_quality_path()),
        model=metadata,
        max_steps=max_steps(),
        prompts_per_step=int(_setting("PROMPTS_PER_STEP", str(DEFAULT_PROMPTS_PER_STEP))),
        group_size=int(_setting("GROUP", str(DEFAULT_GROUP_SIZE))),
        oversample=int(_setting("OVERSAMPLE", str(DEFAULT_OVERSAMPLE))),
        completion_tokens=completion_tokens(),
        capture_states=2,
        capture_layers=metadata.n_layers + 1,
        capture_hidden_size=metadata.hidden_size,
        capture_dtype_bytes=4,
        lens_states=2,
        lens_dtype_bytes=2,
        accumulator_dtype_bytes=4,
        intervention_prompts=8,
        intervention_conditions=3,
        intervention_samples=2,
    )
    payload = cast("dict[str, object]", asdict(budget))
    generation = cast("dict[str, int]", payload["generation"])
    generation["matching_screen_completions"] = budget.training.rows * SCREEN_SAMPLES_PER_PROMPT
    generation["all_endpoint_responses"] = (
        generation["endpoint_responses"] + generation["intervention_responses"]
    )
    smoke_overrides = cast(
        "dict[str, object]",
        importlib.import_module("games.train").SMOKE_OVERRIDES,
    )
    smoke_training = (
        cast("int", smoke_overrides["max_steps"])
        * cast("int", smoke_overrides["prompts_per_step"])
        * cast("int", smoke_overrides["num_generations"])
    )
    steering_rows = _read_object(steering_rows_path()).get("rows")
    if not isinstance(steering_rows, list) or not steering_rows:
        raise ValueError("steering diagnostic manifest must contain nonempty rows")
    configured_step_completions = int(
        _setting("PROMPTS_PER_STEP", str(DEFAULT_PROMPTS_PER_STEP))
    ) * int(_setting("GROUP", str(DEFAULT_GROUP_SIZE)))
    tiny_training = configured_step_completions
    tiny_intervention = len(steering_rows) * 3
    throughput_probe = (NATURAL_PREFIX_REQUEST_COUNT + 1) * configured_step_completions
    research_completions = int(generation["total_completions"])
    serving_smoke = _serving_smoke_request_count()
    generation["serving_smoke_completions_per_model"] = serving_smoke
    generation["plumbing_smoke_completions"] = smoke_training + serving_smoke
    generation["tiny_9b_probe_completions"] = tiny_training + serving_smoke + tiny_intervention
    generation["throughput_probe_completions"] = throughput_probe
    generation["research_completions_excluding_screen"] = research_completions
    generation["whole_sequence_completions"] = (
        research_completions
        + int(generation["matching_screen_completions"])
        + smoke_training
        + serving_smoke
        + tiny_training
        + serving_smoke
        + tiny_intervention
        + throughput_probe
    )
    generation["whole_sequence_token_cap"] = (
        research_completions * completion_tokens()
        + int(generation["matching_screen_completions"]) * completion_tokens()
        + smoke_training * 2048
        + (serving_smoke * 2 + tiny_training + tiny_intervention + throughput_probe)
        * completion_tokens()
    )
    payload["model_metadata_path"] = str(model_metadata_path())
    payload["model_metadata_sha256"] = _path_digest(model_metadata_path())
    payload["model"] = model_id()
    payload["model_source"] = model_source()
    payload["weights_present"] = _weights_present()
    payload["missing_dependencies"] = list(missing_dependencies())
    return payload


def _weights_present() -> bool:
    """Return whether the configured 9B source is a complete local weight snapshot."""
    return _snapshot_weights_present(Path(model_source()))


def _snapshot_weights_present(source: Path) -> bool:
    """Check the exact weight files selected by a local Transformers snapshot."""
    if not source.is_dir():
        return False

    index_path = source / "model.safetensors.index.json"
    if index_path.exists() or index_path.is_symlink():
        return _indexed_snapshot_weights_present(source, index_path)

    safetensor_files = tuple(source.glob("*.safetensors"))
    return safetensor_files == (source / "model.safetensors",) and _regular_nonempty_file(
        source / "model.safetensors"
    )


def _indexed_snapshot_weights_present(source: Path, index_path: Path) -> bool:
    """Check all unique shard files named by a snapshot index."""
    if not _regular_nonempty_file(index_path):
        return False
    try:
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    weight_map = index_payload.get("weight_map") if isinstance(index_payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return False

    shard_paths: set[Path] = set()
    source_resolved = source.resolve()
    for shard_name in weight_map.values():
        if not isinstance(shard_name, str) or not shard_name:
            return False
        shard_path = source / shard_name
        try:
            shard_path.resolve().relative_to(source_resolved)
        except ValueError:
            return False
        shard_paths.add(shard_path)
    return all(_regular_nonempty_file(path) for path in shard_paths)


def _regular_nonempty_file(path: Path) -> bool:
    """Return whether a path is a non-symlink regular file containing bytes."""
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def missing_dependencies() -> tuple[str, ...]:
    """Return every missing runtime input or environment prerequisite."""
    missing: list[str] = []
    missing.extend(
        [
            str(path)
            for path in (
                training_manifest_path(),
                training_corpus_path(),
                behavior_manifest_path(),
                construct_stimuli_path(),
                lens_fit_smoke_path(),
                lens_fit_measurement_path(),
                lens_quality_path(),
                steering_rows_path(),
                intervention_expectation_path(),
                natural_selection_path(),
                model_metadata_path(),
            )
            if not path.is_file()
        ]
    )
    jlens = os.environ.get(JLENS_ENV)
    if not jlens or not Path(jlens).is_dir():
        missing.append(f"environment:{JLENS_ENV}")
    if not _weights_present():
        missing.append("weights:Qwen/Qwen3.5-9B")
    smoke_source = Path(smoke_model_source())
    if not _snapshot_weights_present(smoke_source):
        missing.append("weights:Qwen/Qwen3-0.6B")
    readiness = runtime_root() / "environment-readiness.md"
    if readiness.is_file():
        text = readiness.read_text(encoding="utf-8")
        normalized = text.lower()
        if "strict resource limiter | unavailable" in normalized or (
            "systemd user manager" in normalized and "cannot reach" in normalized
        ):
            missing.append("strict-resource-limiter")
        if "gpu_preflight.py` refuses the card" in text:
            missing.append("gpu-preflight:unattributed-vram")
        if "make canary-check` independently exits 2" in text:
            missing.append("canary-baseline")
        if "make jail-test` itself failed" in text:
            missing.append("jail-negative-controls")
    return tuple(missing)


def _jlens_environment() -> dict[str, str]:
    configured = os.environ.get(JLENS_ENV, "")
    existing = os.environ.get("PYTHONPATH", "")
    joined = configured if not existing else f"{configured}{os.pathsep}{existing}"
    return {"PYTHONPATH": joined}


def _operation(
    name: str,
    module: str,
    args: Sequence[str],
    artifacts: Sequence[Path],
    *,
    environment: Mapping[str, str] | None = None,
) -> Operation:
    return Operation(name, (*UV, "-m", module, *args), tuple(artifacts), environment)


def _phase(
    name: str,
    operations: Sequence[Operation],
    inputs: Sequence[Path],
    *,
    needs_gpu: bool,
) -> Phase:
    slug = name.lower().replace(" ", "-").replace("/", "-")
    return Phase(name, tuple(operations), tuple(inputs), receipt_path(slug), needs_gpu)


def _train_operation(  # noqa: PLR0913 -- one operation carries the native training identity axes
    name: str,
    *,
    model: str,
    output: Path,
    steps: int,
    smoke: bool,
    token_cap: int,
) -> Operation:
    effective_steps = 3 if smoke else steps
    effective_group = "4" if smoke else _setting("GROUP", str(DEFAULT_GROUP_SIZE))
    effective_prompts = (
        "2" if smoke else _setting("PROMPTS_PER_STEP", str(DEFAULT_PROMPTS_PER_STEP))
    )
    effective_token_cap = 2048 if smoke else token_cap
    args = [
        "--arm",
        DEFAULT_ARM,
        "--model",
        SMOKE_MODEL if smoke else model_id(),
        "--model-source",
        model,
        "--output-dir",
        str(output),
        "--resume-from-checkpoint",
        "latest",
        "--max-steps",
        str(effective_steps),
        "--save-steps",
        "1",
        "--save-total-limit",
        "0",
        "--record-retention-manifest",
        "--num-generations",
        effective_group,
        "--prompts-per-step",
        effective_prompts,
        "--max-completion-tokens",
        str(effective_token_cap),
    ]
    if smoke:
        args.append("--smoke")
    else:
        args.extend(("--corpus", str(training_corpus_path())))
    return _operation(
        name,
        "games.train",
        args,
        (output / "train_summary.json", output / "retention_manifest.json"),
    )


def _capture_operation(  # noqa: PLR0913 -- one operation carries the native capture identity axes
    name: str,
    *,
    model: str,
    training_root: Path,
    output: Path,
    step: int,
    natural: tuple[Path, Path] | None = None,
) -> Operation:
    args = [
        "--stimuli",
        str(construct_stimuli_path()),
        "--cooperation-constructs",
        "--reserved-identities",
        str(runtime_root() / "construct-external-reservations.json"),
        "--out-dir",
        str(output),
        "--arm",
        f"{DEFAULT_ARM}={training_root}",
        "--steps",
        str(step),
        "--base-model",
        model,
        "--poolings",
        "last,mean",
        "--batch-size",
        "1",
        "--max-prompt-tokens",
        "2048",
        "--compute-dtype",
        "bfloat16",
        "--store-dtype",
        "float32",
        "--capture-prefix-states",
    ]
    base_cell = output / "base" / "step-0"
    final_cell = output / DEFAULT_ARM / f"step-{step}"
    artifacts: list[Path] = [
        output / "ladder-manifest.json",
        base_cell,
        final_cell,
    ]
    if natural is not None:
        base_natural, final_natural = natural
        args.extend(
            (
                "--natural-prefix-layers",
                NATURAL_PREFIX_LAYERS,
                "--natural-prefix-stimuli",
                f"base={base_natural}",
                "--natural-prefix-stimuli",
                f"{DEFAULT_ARM}={final_natural}",
                "--natural-selection-manifest",
                str(natural_selection_path()),
            )
        )
        artifacts.extend(
            (
                base_cell / "natural-prefix" / "natural-prefix-manifest.json",
                base_cell / "natural-prefix" / "natural-prefix-activations.safetensors",
                final_cell / "natural-prefix" / "natural-prefix-manifest.json",
                final_cell / "natural-prefix" / "natural-prefix-activations.safetensors",
            )
        )
    return _operation(name, "games.interp_capture", args, artifacts)


def _geometry_operation(
    name: str,
    *,
    capture_root: Path,
    output: Path,
    step: int,
    natural: bool = False,
) -> Operation:
    args = [
        "--capture-root",
        str(capture_root),
        "--stimuli",
        str(construct_stimuli_path()),
        "--reserved-group-ids",
        str(runtime_root() / "construct-external-reservations.json"),
        "--intervention-expectation",
        str(intervention_expectation_path()),
        "--out-dir",
        str(output),
        "--final-arm",
        DEFAULT_ARM,
        "--final-step",
        str(step),
        "--poolings",
        "last,mean",
        "--export-pooling",
        "last",
    ]
    if natural:
        args.extend(
            (
                "--natural-prefix-artifact",
                f"base={capture_root / 'base' / 'step-0' / 'natural-prefix'}",
                "--natural-prefix-artifact",
                f"final={capture_root / DEFAULT_ARM / f'step-{step}' / 'natural-prefix'}",
            )
        )
    return _operation(
        name,
        "games.cooperation_interp",
        args,
        (
            output / "cooperation_interp.json",
            output / "direction-manifest.json",
            output / "selected-target.json",
        ),
    )


def _gate_operation(name: str, *, model: str, fit_stimuli: Path, output: Path) -> Operation:
    return _operation(
        name,
        "reward_hacking.interp.lens_fit_gate",
        (
            "--model-id",
            model,
            "--tokenizer",
            model,
            "--fit-stimuli",
            str(fit_stimuli),
            "--stimulus-render",
            "templated_here",
            "--max-seq-len",
            "2048",
            "--dim-batch-candidates",
            "1",
            "--work-dir",
            str(output.parent / "work"),
            "--out",
            str(output),
        ),
        (output,),
        environment=_jlens_environment(),
    )


def _lens_operation(  # noqa: PLR0913 -- one operation carries the native lens identity axes
    name: str,
    *,
    model: str,
    adapter: Path,
    capture_root: Path,
    geometry_root: Path,
    gate_report: Path,
    output: Path,
    step: int,
    profile: str,
) -> Operation:
    fit = lens_fit_smoke_path() if profile == "smoke" else lens_fit_measurement_path()
    return _operation(
        name,
        "games.cooperation_lens",
        (
            "--fit-stimuli",
            str(fit),
            "--quality-stimuli",
            str(lens_quality_path()),
            "--direction-manifest",
            str(geometry_root / "direction-manifest.json"),
            "--construct-capture-root",
            str(capture_root),
            "--gradient-gate-report",
            str(gate_report),
            "--reserved-group-ids",
            str(runtime_root() / "lens-external-reservations.json"),
            "--final-adapter",
            str(adapter),
            "--final-arm",
            DEFAULT_ARM,
            "--final-step",
            str(step),
            "--out-dir",
            str(output),
            "--profile",
            profile,
            "--base-model",
            model,
            "--dim-batch",
            "1",
        ),
        (
            output / "cooperation_lens.json",
            output / "base",
            output / "final",
        ),
        environment=_jlens_environment(),
    )


def _steering_operation(  # noqa: PLR0913 -- one operation carries the native steering identity axes
    name: str,
    *,
    model: str,
    adapter: Path,
    geometry_root: Path,
    output: Path,
    rows: Path | None,
    samples: int,
    conditions: str,
    token_cap: int | None = None,
) -> Operation:
    args = [
        "generate",
        "--model",
        model,
        "--adapter",
        str(adapter),
    ]
    for direction in ("costly-other-regard", "decision-dependence", "trained-displacement"):
        args.extend(
            (
                "--direction",
                f"{direction}={geometry_root / 'directions' / 'final' / f'{direction}.pt'}",
            )
        )
    args.extend(
        (
            "--selected-target",
            str(geometry_root / "selected-target.json"),
            "--n-samples",
            str(samples),
            "--batch-size",
            "1",
            "--max-new-tokens",
            str(completion_tokens() if token_cap is None else token_cap),
            "--conditions",
            conditions,
            "--seed",
            "0",
            "--placebo-seed",
            "941731",
            "--out-dir",
            str(output),
        )
    )
    if rows is not None:
        args.extend(
            ("--row-manifest", str(rows), "--diagnostic-profile", "cooperation-generalization")
        )
    return _operation(
        name,
        "games.interp_steering",
        args,
        (
            output / "steering_records.jsonl",
            output / "steering_records.jsonl.resume.json",
            output / "steering_summary.json",
        ),
    )


def _static_budget_operation() -> Operation:
    output = run_root() / "static-plan.json"
    return _operation(
        "derive exact counts and disk forecast",
        "games.cooperation_budget",
        (
            "plan",
            "--training-manifest",
            str(training_manifest_path()),
            "--training-corpus",
            str(training_corpus_path()),
            "--behavior-manifest",
            str(behavior_manifest_path()),
            "--survey-data-dir",
            str(survey_data_dir()),
            "--construct-stimuli",
            str(construct_stimuli_path()),
            "--lens-stimuli",
            str(lens_fit_measurement_path()),
            "--lens-stimuli",
            str(lens_quality_path()),
            "--model-metadata",
            str(model_metadata_path()),
            "--max-steps",
            str(max_steps()),
            "--prompts-per-step",
            _setting("PROMPTS_PER_STEP", str(DEFAULT_PROMPTS_PER_STEP)),
            "--group-size",
            _setting("GROUP", str(DEFAULT_GROUP_SIZE)),
            "--oversample",
            _setting("OVERSAMPLE", str(DEFAULT_OVERSAMPLE)),
            "--completion-tokens",
            str(completion_tokens()),
            "--capture-states",
            "2",
            "--capture-poolings",
            "2",
            "--capture-boundaries",
            "3",
            "--checkpoint-count",
            str(max_steps() + 1),
            "--out",
            str(output),
        ),
        (output,),
    )


def _natural_prefix_operation(name: str, *, state: str, records: Path, output: Path) -> Operation:
    return _operation(
        name,
        "games.interp_capture",
        (
            "--build-natural-prefix-stimuli",
            "--records",
            str(records),
            "--natural-selection-manifest",
            str(natural_selection_path()),
            "--natural-prefix-out",
            str(output),
            "--natural-state",
            state,
            "--base-model",
            model_source(),
            "--stimulus-render",
            "templated_here",
        ),
        (output, output.with_suffix(output.suffix + ".manifest.json")),
    )


def _screen_operation() -> Operation:
    output = run_root() / "matching-screen"
    return _operation(
        "screen every training stratum at the matching sampler",
        "games.cooperation_screen",
        (
            "--corpus",
            str(training_corpus_path()),
            "--out-dir",
            str(output),
            "--model",
            model_source(),
            "--model-id",
            model_id(),
            "--samples-per-prompt",
            str(SCREEN_SAMPLES_PER_PROMPT),
            "--seed",
            "0",
            "--backend",
            "vllm",
            "--thinking",
            "--max-new-tokens",
            str(completion_tokens()),
        ),
        (
            output / "matching-sampler-screen.jsonl",
            output / "matching-sampler-summary.json",
        ),
    )


def _throughput_operation() -> Operation:
    return _operation(
        "measure 9B generation backward optimizer and memory",
        "games.throughput_probe",
        (
            "--arm",
            DEFAULT_ARM,
            "--model",
            model_id(),
            "--model-source",
            model_source(),
            "--corpus",
            str(training_corpus_path()),
            "--prompts-per-step",
            _setting("PROMPTS_PER_STEP", str(DEFAULT_PROMPTS_PER_STEP)),
            "--num-generations",
            _setting("GROUP", str(DEFAULT_GROUP_SIZE)),
            "--max-completion-tokens",
            str(completion_tokens()),
            "--measured-steps",
            "4",
            "--warmup-steps",
            "1",
            "--output-dir",
            str(run_root() / "throughput-run"),
            "--json-out",
            str(run_root() / "throughput.json"),
        ),
        (run_root() / "throughput.json",),
    )


def _endpoint_operation(name: str, *, output: Path, adapter: Path | None, step: int) -> Operation:
    args = [
        "--endpoint",
        "all",
        "--model",
        model_source(),
        "--model-id",
        model_id(),
        "--arm",
        "base" if adapter is None else DEFAULT_ARM,
        "--step",
        str(step),
        "--rollout-id",
        "base/step-0" if adapter is None else f"{DEFAULT_ARM}/step-{step}",
        "--manifest",
        str(behavior_manifest_path()),
        "--trained-game-id",
        "twin-pd",
        "--trained-game-id",
        "stag-hunt",
        "--trained-game-id",
        "trust-vs-stated-return",
        "--survey-data-dir",
        str(survey_data_dir()),
        "--out-dir",
        str(output),
        "--context-samples",
        "2",
        "--batch-size",
        "1",
        "--backend",
        "vllm",
        "--thinking",
        "--sampler",
        "training-distribution",
        "--max-new-tokens",
        str(completion_tokens()),
        "--resume",
    ]
    if adapter is not None:
        args.extend(("--adapter", str(adapter)))
    endpoints = (
        "behavior",
        "allocation",
        "full-context",
        "core-survey",
        "prosocialness",
        "local-dt",
    )
    artifacts = tuple(
        path
        for endpoint in endpoints
        for path in (output / f"{endpoint}.jsonl", output / f"{endpoint}.summary.json")
    )
    return _operation(
        name,
        "games.cooperation_eval_runner",
        args,
        artifacts,
    )


def _serving_smoke_args(
    *, model: str, model_identity: str, adapter: Path, output: Path, step: int
) -> list[str]:
    """Build the frozen vLLM serving-smoke arguments for one adapter checkpoint."""
    return [
        "--evaluation-profile",
        PROFILE_SERVING_SMOKE,
        "--model",
        model,
        "--model-id",
        model_identity,
        "--checkpoint",
        str(adapter),
        "--arm",
        DEFAULT_ARM,
        "--step",
        str(step),
        "--rollout-id",
        f"{DEFAULT_ARM}/step-{step}/serving-smoke",
        "--manifest",
        str(behavior_manifest_path()),
        "--survey-data-dir",
        str(survey_data_dir()),
        "--out-dir",
        str(output),
        "--context-samples",
        "2",
        "--batch-size",
        "1",
        "--backend",
        "vllm",
        "--thinking",
        "--sampler",
        "training-distribution",
        "--max-new-tokens",
        str(completion_tokens()),
        "--resume",
    ]


def _serving_smoke_request_count() -> int:
    """Derive the smoke count from the native private-manifest planner."""
    args = build_endpoint_parser().parse_args(
        _serving_smoke_args(
            model=model_source(),
            model_identity=model_id(),
            adapter=Path("unused-checkpoint"),
            output=Path("unused-output"),
            step=1,
        )
    )
    payload = print_endpoint_plan(args)
    count = payload.get("total_requests")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("native serving-smoke plan reported no requests")
    return count


def _serving_smoke_operation(
    *,
    model: str,
    model_identity: str,
    adapter: Path,
    output: Path,
    step: int,
) -> Operation:
    artifacts = tuple(
        path
        for endpoint in SERVING_SMOKE_ENDPOINTS
        for path in (output / f"{endpoint}.jsonl", output / f"{endpoint}.summary.json")
    )
    return _operation(
        f"exercise {model_identity} runtime-adapter serving and evaluation",
        "games.cooperation_eval_runner",
        _serving_smoke_args(
            model=model,
            model_identity=model_identity,
            adapter=adapter,
            output=output,
            step=step,
        ),
        artifacts,
    )


def _retention_operation() -> Operation:
    steps = max_steps()
    prior = max(0, steps - 1)
    middle = max(1, steps // 2)
    args = [
        "--run-root",
        str(training_run_root()),
        "--base-model",
        model_source(),
        "--retain-step",
        str(prior),
        "--retain-step",
        str(steps),
        "--adapter-step",
        "1",
        "--adapter-step",
        str(middle),
    ]
    for readout in (
        "../final/behavior.summary.json",
        "../final/allocation.summary.json",
        "../final/full-context.summary.json",
        "../final/core-survey.summary.json",
        "../final/prosocialness.summary.json",
        "../final/local-dt.summary.json",
        "../capture/ladder-manifest.json",
        "../geometry/cooperation_interp.json",
        "../lens/cooperation_lens.json",
        "../intervention/steering_records.jsonl",
    ):
        args.extend(("--readout", readout))
    args.extend(("--json-out", str(run_root() / "retention-final.json")))
    return _operation(
        "validate retained checkpoints then prune redundant full state",
        "games.cooperation_retention",
        args,
        (
            training_run_root() / "retention_readout_authorization.json",
            run_root() / "retention-final.json",
        ),
    )


def record_measurements(output: Path) -> None:
    """Combine persisted native timings into the strict pre-training budget input."""
    throughput = _read_object(run_root() / "throughput.json")
    measured_phases = throughput.get("throughput")
    if not isinstance(measured_phases, dict):
        raise TypeError("throughput artifact lacks its measured throughput section")
    timing: dict[str, float] = {}
    for name in ("generation", "backward", "optimizer"):
        value = measured_phases.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise ValueError(f"throughput artifact lacks positive {name} timing")
        timing[name] = float(value)

    receipts = {
        "smoke_0_6b": receipt_path("0.6b-training-plumbing-smoke"),
        "tiny_9b_update": receipt_path("tiny-9b-update"),
        "tiny_9b_serving": receipt_path("tiny-9b-runtime-adapter-serving-smoke"),
        "tiny_9b_capture": receipt_path("tiny-9b-capture"),
        "tiny_9b_gradient": receipt_path("tiny-9b-gradient"),
        "matching_sampler_screen": receipt_path("matching-sampler-screen"),
        "throughput_probe": receipt_path("persisted-9b-throughput"),
        "baseline": receipt_path("frozen-baseline"),
    }
    loaded = {name: _read_object(path) for name, path in receipts.items()}
    prelaunch = {name: payload.get("elapsed_seconds") for name, payload in loaded.items()}
    operation_seconds = cast("dict[str, float]", loaded["tiny_9b_gradient"]["operation_seconds"])
    gradient_gate = operation_seconds.get("verify tiny 9B DeltaNet gradient and resume gates")
    lens_smoke = operation_seconds.get("fit tiny 9B smoke Jacobian lenses")
    intervention_probe = operation_seconds.get("price a 9B intervention condition")
    capture_seconds = loaded["tiny_9b_capture"].get("elapsed_seconds")
    baseline_seconds = loaded["baseline"].get("elapsed_seconds")
    measured = {
        "baseline": baseline_seconds,
        "capture": capture_seconds,
        "gradient_gate": gradient_gate,
        "lens": lens_smoke,
        "intervention": intervention_probe,
        **{f"prelaunch.{name}": value for name, value in prelaunch.items()},
    }
    invalid = {
        name: value
        for name, value in measured.items()
        if isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0
    }
    if invalid:
        raise ValueError(f"phase receipts lack positive measured timings: {invalid}")
    values = {
        "final_eval": float(cast("float", baseline_seconds)),
        "capture": float(cast("float", capture_seconds)),
        "lens": (float(cast("float", gradient_gate)) + float(cast("float", lens_smoke))) * 5,
        "intervention": float(cast("float", intervention_probe)) * 2,
    }
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError(f"endpoint costing contains a non-positive timing: {values}")
    smoke_checkpoints = {
        checkpoint.name: _directory_bytes(checkpoint)
        for checkpoint in sorted((run_root() / "smoke-0.6b" / "training").glob("checkpoint-*"))
        if checkpoint.is_dir()
    }
    tiny_checkpoint = run_root() / "tiny-9b" / "training" / "checkpoint-1"
    tiny_checkpoint_bytes = _directory_bytes(tiny_checkpoint)
    if not smoke_checkpoints or any(value <= 0 for value in smoke_checkpoints.values()):
        raise ValueError("0.6B smoke produced no measurable checkpoint bytes")
    if tiny_checkpoint_bytes <= 0:
        raise ValueError("tiny 9B update produced no measurable checkpoint bytes")
    generation = cast("dict[str, int]", static_plan()["generation"])
    planned_completions = int(generation["research_completions_excluding_screen"])
    planned_token_cap = int(generation["total_generated_tokens"])
    measured_checkpoint_bytes = sum(smoke_checkpoints.values()) + tiny_checkpoint_bytes
    _write_json(
        output,
        {
            "throughput": timing,
            "prelaunch": prelaunch,
            "endpoints": values,
            "headroom_fraction": 0.20,
            "measured_max_steps": int(_setting("MEASURED_MAX_STEPS", str(INITIAL_STEP_LIMIT))),
            "completion_token_cap": completion_tokens(),
            "planned_generation_completions": planned_completions,
            "planned_generated_token_cap": planned_token_cap,
            "disk_actual_bytes": measured_checkpoint_bytes,
            "disk": {
                "smoke_checkpoint_bytes": smoke_checkpoints,
                "tiny_9b_checkpoint_bytes": tiny_checkpoint_bytes,
            },
            "scaling": {
                "final_eval": "one full baseline measured and reserved once for final evaluation",
                "capture": "tiny 9B base plus one-step adapter used as two-state capture price",
                "lens": "ten-prompt gradient gate plus lens fit measured and multiplied by five",
                "intervention": "all three eight-row conditions measured once and doubled for two draws",
            },
        },
    )


def phase_specs() -> tuple[Phase, ...]:
    """Return the fixed experiment order using only native operation CLIs."""
    runtime_inputs = (
        training_manifest_path(),
        training_corpus_path(),
        behavior_manifest_path(),
        construct_stimuli_path(),
        lens_fit_smoke_path(),
        lens_fit_measurement_path(),
        lens_quality_path(),
        steering_rows_path(),
        intervention_expectation_path(),
        natural_selection_path(),
        model_metadata_path(),
    )
    membership_outputs = (
        runtime_root() / "group-membership.json",
        runtime_root() / "construct-external-reservations.json",
        runtime_root() / "lens-external-reservations.json",
        preflight_identity_path(),
        run_root() / "static-plan.json",
    )
    manifest = _phase(
        "corpus and scorer audit",
        (
            _operation(
                "build frozen training corpus and offline audit",
                "games.cooperation_corpus",
                (
                    "--corpus-out",
                    str(training_corpus_path()),
                    "--manifest-out",
                    str(training_manifest_path()),
                    "--audit-out",
                    str(runtime_root() / "learning-signal-audit.json"),
                ),
                (
                    training_corpus_path(),
                    training_manifest_path(),
                    runtime_root() / "learning-signal-audit.json",
                ),
            ),
        ),
        (behavior_manifest_path(),),
        needs_gpu=False,
    )
    preflight = _phase(
        "CPU preflight",
        (
            _operation(
                "write run identity and stage-specific reservations",
                "games.cooperation_sequence",
                ("--prepare",),
                membership_outputs,
            ),
            _static_budget_operation(),
        ),
        runtime_inputs,
        needs_gpu=False,
    )

    smoke_root = run_root() / "smoke-0.6b"
    smoke_train = smoke_root / "training"
    smoke = _phase(
        "0.6B training plumbing smoke",
        (
            _train_operation(
                "run three-update 0.6B smoke training",
                model=smoke_model_source(),
                output=smoke_train,
                steps=1,
                smoke=True,
                token_cap=8192,
            ),
            _serving_smoke_operation(
                model=smoke_model_source(),
                model_identity=SMOKE_MODEL,
                adapter=smoke_train / "checkpoint-3",
                output=smoke_root / "serving-smoke",
                step=3,
            ),
        ),
        (*runtime_inputs, *membership_outputs[:3], preflight_identity_path()),
        needs_gpu=True,
    )

    tiny_root = run_root() / "tiny-9b"
    tiny_train = tiny_root / "training"
    tiny_update = _phase(
        "tiny 9B update",
        (
            _train_operation(
                "run one 9B update through the research path",
                model=model_source(),
                output=tiny_train,
                steps=1,
                smoke=False,
                token_cap=completion_tokens(),
            ),
        ),
        (training_corpus_path(), run_identity_path()),
        needs_gpu=True,
    )
    tiny_serving = _phase(
        "tiny 9B runtime-adapter serving smoke",
        (
            _serving_smoke_operation(
                model=model_source(),
                model_identity=model_id(),
                adapter=tiny_train / "checkpoint-1",
                output=tiny_root / "serving-smoke",
                step=1,
            ),
        ),
        (tiny_train / "checkpoint-1", behavior_manifest_path()),
        needs_gpu=True,
    )
    tiny_capture_root = tiny_root / "capture"
    tiny_capture = _phase(
        "tiny 9B capture",
        (
            _capture_operation(
                "capture tiny 9B prompt and teacher-forced states",
                model=model_source(),
                training_root=tiny_train,
                output=tiny_capture_root,
                step=1,
            ),
        ),
        (tiny_train / "checkpoint-1", construct_stimuli_path(), membership_outputs[1]),
        needs_gpu=True,
    )
    tiny_geometry = tiny_root / "geometry"
    tiny_gate = tiny_root / "lens-gate-smoke.json"
    tiny_gradient = _phase(
        "tiny 9B gradient",
        (
            _geometry_operation(
                "fit tiny 9B construct geometry",
                capture_root=tiny_capture_root,
                output=tiny_geometry,
                step=1,
            ),
            _gate_operation(
                "verify tiny 9B DeltaNet gradient and resume gates",
                model=model_source(),
                fit_stimuli=lens_fit_smoke_path(),
                output=tiny_gate,
            ),
            _lens_operation(
                "fit tiny 9B smoke Jacobian lenses",
                model=model_source(),
                adapter=tiny_train / "checkpoint-1",
                capture_root=tiny_capture_root,
                geometry_root=tiny_geometry,
                gate_report=tiny_gate,
                output=tiny_root / "lens",
                step=1,
                profile="smoke",
            ),
            _steering_operation(
                "price a 9B intervention condition",
                model=model_source(),
                adapter=tiny_train / "checkpoint-1",
                geometry_root=tiny_geometry,
                output=tiny_root / "intervention-cost",
                rows=steering_rows_path(),
                samples=1,
                conditions="none,steer:+,placebo:+",
            ),
        ),
        (
            tiny_capture_root / "ladder-manifest.json",
            tiny_train / "checkpoint-1",
            lens_fit_smoke_path(),
            lens_quality_path(),
            intervention_expectation_path(),
        ),
        needs_gpu=True,
    )
    screen = _phase(
        "matching sampler screen",
        (_screen_operation(),),
        (training_corpus_path(), training_manifest_path()),
        needs_gpu=True,
    )
    throughput = _phase(
        "persisted 9B throughput",
        (_throughput_operation(),),
        (training_corpus_path(), run_identity_path()),
        needs_gpu=True,
    )
    baseline = _phase(
        "frozen baseline",
        (
            _endpoint_operation(
                "run complete base endpoint battery",
                output=run_root() / "baseline",
                adapter=None,
                step=0,
            ),
        ),
        (behavior_manifest_path(), run_identity_path()),
        needs_gpu=True,
    )
    budget = _phase(
        "measured budget gate",
        (
            _operation(
                "assemble persisted whole-run measurements",
                "games.cooperation_sequence",
                ("--record-measurements", "--out", str(measurements_path())),
                (measurements_path(),),
            ),
            _operation(
                "reserve endpoint time before training",
                "games.cooperation_budget",
                (
                    "aggregate",
                    "--measurements",
                    str(measurements_path()),
                    "--max-steps",
                    str(max_steps()),
                    "--out",
                    str(measured_budget_path()),
                ),
                (measured_budget_path(),),
            ),
        ),
        (
            run_root() / "throughput.json",
            smoke_train,
            tiny_train / "checkpoint-1",
            tiny_capture.receipt_path,
            tiny_gradient.receipt_path,
            baseline.receipt_path,
        ),
        needs_gpu=False,
    )
    training = _phase(
        "research training block",
        (
            _train_operation(
                f"train {DEFAULT_ARM}",
                model=model_source(),
                output=training_run_root(),
                steps=max_steps(),
                smoke=False,
                token_cap=completion_tokens(),
            ),
        ),
        (training_corpus_path(), measured_budget_path(), run_identity_path()),
        needs_gpu=True,
    )
    final = _phase(
        "frozen final",
        (
            _endpoint_operation(
                "run complete final endpoint battery",
                output=run_root() / "final",
                adapter=final_adapter_path(),
                step=max_steps(),
            ),
        ),
        (behavior_manifest_path(), final_adapter_path(), run_identity_path()),
        needs_gpu=True,
    )
    natural_base = run_root() / "natural-prefix" / "base.jsonl"
    natural_final = run_root() / "natural-prefix" / "final.jsonl"
    natural = _phase(
        "natural prefix materialization",
        (
            _natural_prefix_operation(
                "extract selected base natural prefixes",
                state="base",
                records=run_root() / "baseline" / "behavior.jsonl",
                output=natural_base,
            ),
            _natural_prefix_operation(
                "extract selected final natural prefixes",
                state="final",
                records=run_root() / "final" / "behavior.jsonl",
                output=natural_final,
            ),
        ),
        (
            natural_selection_path(),
            run_root() / "baseline" / "behavior.jsonl",
            run_root() / "final" / "behavior.jsonl",
        ),
        needs_gpu=False,
    )
    capture_root = run_root() / "capture"
    activation = _phase(
        "activation measurement",
        (
            _capture_operation(
                "capture base and final constructs and selected natural prefixes",
                model=model_source(),
                training_root=training_run_root(),
                output=capture_root,
                step=max_steps(),
                natural=(natural_base, natural_final),
            ),
        ),
        (
            construct_stimuli_path(),
            final_adapter_path(),
            natural_base,
            natural_base.with_suffix(natural_base.suffix + ".manifest.json"),
            natural_final,
            natural_final.with_suffix(natural_final.suffix + ".manifest.json"),
        ),
        needs_gpu=True,
    )
    geometry_root = run_root() / "geometry"
    geometry = _phase(
        "construct geometry",
        (
            _geometry_operation(
                "analyze construct axes displacement and natural-prefix cross-check",
                capture_root=capture_root,
                output=geometry_root,
                step=max_steps(),
                natural=True,
            ),
        ),
        (capture_root / "ladder-manifest.json", construct_stimuli_path()),
        needs_gpu=False,
    )
    measurement_gate = run_root() / "lens" / "gradient-gate-measurement.json"
    lens = _phase(
        "Jacobian lens measurement",
        (
            _gate_operation(
                "verify measurement-corpus gradient gates",
                model=model_source(),
                fit_stimuli=lens_fit_measurement_path(),
                output=measurement_gate,
            ),
            _lens_operation(
                "fit base and final model-specific lenses plus shared-base read",
                model=model_source(),
                adapter=final_adapter_path(),
                capture_root=capture_root,
                geometry_root=geometry_root,
                gate_report=measurement_gate,
                output=run_root() / "lens",
                step=max_steps(),
                profile="measurement",
            ),
        ),
        (
            geometry_root / "direction-manifest.json",
            capture_root / "ladder-manifest.json",
            lens_fit_measurement_path(),
            lens_quality_path(),
            final_adapter_path(),
        ),
        needs_gpu=True,
    )
    intervention = _phase(
        "controlled intervention",
        (
            _steering_operation(
                "run held-out unperturbed real and matched-placebo conditions",
                model=model_source(),
                adapter=final_adapter_path(),
                geometry_root=geometry_root,
                output=run_root() / "intervention",
                rows=steering_rows_path(),
                samples=2,
                conditions="none,steer:+,placebo:+",
            ),
        ),
        (
            geometry_root / "selected-target.json",
            steering_rows_path(),
            final_adapter_path(),
        ),
        needs_gpu=True,
    )
    retention = _phase(
        "checkpoint retention",
        (_retention_operation(),),
        (
            final_adapter_path(),
            final.receipt_path,
            activation.receipt_path,
            geometry.receipt_path,
            lens.receipt_path,
            intervention.receipt_path,
        ),
        needs_gpu=True,
    )
    phases = (
        manifest,
        preflight,
        smoke,
        tiny_update,
        tiny_serving,
        tiny_capture,
        tiny_gradient,
        screen,
        throughput,
        baseline,
        budget,
        training,
        final,
        natural,
        activation,
        geometry,
        lens,
        intervention,
        retention,
    )
    selected = os.environ.get(f"{ENV_PREFIX}STAGES")
    if not selected:
        return phases
    requested = {value.strip() for value in selected.split(",") if value.strip()}
    unknown = requested - {phase.name for phase in phases}
    if unknown:
        raise ValueError(f"unknown phase names: {sorted(unknown)}")
    return tuple(phase for phase in phases if phase.name in requested)


def stages() -> list[Stage]:
    """Expose only pending receipts so completed GPU phases skip before card acquisition."""
    pending: list[Stage] = []
    for phase in phase_specs():
        if phase_is_complete(phase):
            continue
        slug = phase.name.lower().replace(" ", "-").replace("/", "-")
        pending.append(
            Stage(
                name=phase.name,
                argv=(*UV, "-m", __name__, "--execute-phase", phase.name),
                artifacts=(phase.receipt_path,),
                needs_gpu=phase.needs_gpu,
                log_path=run_root() / "logs" / f"{slug}-{artifact_model_slug(model_id())}.log",
                log_run_dirs=(run_root(),),
            )
        )
    return pending


def describe_plan() -> str:
    """Render the exact CPU plan, numerical disk forecast, dependencies and commands."""
    payload = static_plan()
    generation = cast("dict[str, object]", payload["generation"])
    disk = cast("dict[str, object]", payload["disk"])
    natural = _read_object(natural_selection_path())
    natural_requests = natural.get("request_ids")
    natural_layers = natural.get("layers")
    if not isinstance(natural_requests, list) or not isinstance(natural_layers, list):
        raise TypeError("natural-prefix selection must list request_ids and layers")
    first_gpu_command = (
        "scripts/tmux_run.sh cooperation-generalization-smoke-0_6b "
        "--log /var/tmp/cooperation-generalization-smoke-0_6b.log -- "
        "scripts/resource-limits.sh --gpu -t 45m -- env "
        f"{JLENS_ENV}={shlex.quote(os.environ.get(JLENS_ENV, ''))} "
        f"{ENV_PREFIX}STAGES={shlex.quote('0.6B training plumbing smoke')} "
        "uv run --frozen python -m games.stage_runner --plan games.cooperation_sequence"
    )
    lines = [
        f"logical model: {payload['model']}",
        f"immutable model source: {payload['model_source']}",
        f"training rows: {cast('dict[str, object]', payload['training'])['corpus_rows']}",
        f"matching-screen completions: {generation['matching_screen_completions']}",
        f"research training completions: {generation['training_completions']}",
        f"responses per checkpoint: {int(cast('int', generation['endpoint_responses'])) // 2}",
        f"base+final responses: {generation['endpoint_responses']}",
        f"intervention responses: {generation['intervention_responses']}",
        (
            "research completions excluding matching screen: "
            f"{generation['research_completions_excluding_screen']}"
        ),
        f"0.6B smoke completions: {generation['plumbing_smoke_completions']}",
        f"tiny 9B path completions: {generation['tiny_9b_probe_completions']}",
        f"throughput probe completions: {generation['throughput_probe_completions']}",
        f"whole sequence completions: {generation['whole_sequence_completions']}",
        f"whole sequence generated-token cap: {generation['whole_sequence_token_cap']}",
        f"estimated retained bytes: {disk['estimated_bytes']}",
        f"checkpoint bytes: {disk['checkpoint_bytes_total']}",
        f"capture bytes: {disk['capture_bytes_total']}",
        f"lens bytes: {disk['lens_bytes_total']}",
        f"natural-prefix requests: {len(natural_requests)}",
        "natural-prefix positions per request: 3",
        f"natural-prefix layers: {','.join(str(layer) for layer in natural_layers)}",
        "missing dependencies: " + (", ".join(missing_dependencies()) or "none"),
        "first GPU smoke wall-clock estimate: unavailable before the first bounded measurement",
        "first GPU smoke wall-clock safety cap: 45 minutes",
        (
            "architecture seam: Qwen3-0.6B has no Qwen3.5 DeltaNet; the mandatory tiny 9B "
            "phase owns Jacobian and DeltaNet validation"
        ),
        "first GPU smoke command (requires owner approval and cleared dependencies):",
        f"  {first_gpu_command}",
        "planned phases:",
    ]
    for phase in phase_specs():
        lines.append(f"  {phase.name} [{'GPU' if phase.needs_gpu else 'CPU'}]")
        lines.extend(f"    {shlex.join(operation.argv)}" for operation in phase.operations)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the CPU inspection and phase-execution CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute-phase", default=None)
    parser.add_argument("--record-measurements", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect, prepare, execute one phase, or run the pending sequence."""
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    selected = sum(
        bool(value)
        for value in (args.print_plan, args.prepare, args.execute_phase, args.record_measurements)
    )
    if selected > 1:
        raise ValueError("select only one sequence action")
    if args.print_plan:
        print(describe_plan())  # noqa: T201
        return 0
    if args.prepare:
        prepare_runtime_outputs()
        return 0
    if args.record_measurements:
        record_measurements(args.out or measurements_path())
        return 0
    if args.execute_phase:
        matches = [phase for phase in phase_specs() if phase.name == args.execute_phase]
        if len(matches) != 1:
            raise ValueError(f"unknown or ambiguous phase {args.execute_phase!r}")
        execute_phase(matches[0])
        return 0
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(f"experiment dependencies are missing: {list(missing)}")
    return 0 if run_sequence(stages()).ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
