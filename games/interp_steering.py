"""Steer and ablate a fitted residual direction, and read what it does to twin-pd cooperation.

The activation analyses are correlational: they say the two grading rules translated the whole
representation along directions whose shared component aligns with the decision axis, peaking in
the mid band. This module is the causal tier -- does pushing the model along that direction *move
behaviour*, or is the direction merely decodable? Three subcommands, cheapest first:

* ``fit-direction`` fits a diff-of-means direction per layer from one captured cell
  (`games.interp_cells` format, behind the same corpus-digest guards the trajectory read uses) and
  saves it in the ``{layer: tensor}`` format `games.interp_trajectory.save_directions` writes, so
  `games.interp_lens_ladder --direction-path` reads the same file. Fitting on even pairs by default
  leaves the odd pairs and every eval prompt held out of the fit.
* ``logit-sweep`` is the forced-choice readout: twin-pd prompts rendered thinking-off with the
  assistant turn prefilled up to ``<action>``, so the next token is the action label itself, and the
  coop-minus-defect logit gap is read under steering at a grid of (direction, layer, alpha). No
  generation, so a whole grid costs minutes, and its output selects the (layer, alpha) the expensive
  generation leg runs at. Caveat carried in the payload: thinking-off is not the trained regime, so
  this readout *selects* the intervention; it does not establish the behavioural claim.
* ``generate`` is the behavioural readout: thinking-on generation on the held-out twin-pd eval
  prompts under the training-distribution sampler (presence_penalty=0 -- measured load-bearing: the
  trained effect this instrument exists to detect reads -0.233 at pp=0 and -0.055 at pp=1.5), with
  actions parsed exactly the way the eval battery parses them.

**The matched-norm placebo is not optional.** Every steering condition runs beside a placebo
condition pushing a random direction of equal norm at the same layer and alpha; an effect that the
placebo reproduces is "any perturbation this large does this", the documented 3B/8B failure mode.
The placebo arms go through the same code path as the real ones -- the condition table is data, not
branches -- so a pipeline bug cannot favour one arm.

**The action-label print-order confound is reported, never averaged away.** On this corpus the model
picks the first-printed label far more often than any trained effect moves it, so every rate is
split by ``label_print_order`` (both orders are always rendered) as well as pooled.

**Every record names the Gated DeltaNet kernels its forwards ran under**, and no summary pools two of
them. ``generate`` is the one leg here that decodes token by token, so it is the leg the fla decode
bridge speeds up (1.14-1.51x per step at 0.8B, probe I1) and the leg whose greedy tokens diverge
between the two kernels -- same recurrence, different reduction order. So the bridge is applied before
the model loads, ``deltanet_kernel`` on every record says what it bound, the resume identity carries
it (a relaunch under the other kernel is refused before it decodes rather than after), and
:func:`summarise_records` refuses a pool that mixes bindings.

Runs on the HF path by necessity: vLLM exposes no residual-stream hooks. GPU for the sweep and
generation; ``fit-direction`` is CPU arithmetic over cached tensors.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from string import Formatter
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch

from games.chunked_decode import decode_in_chunks
from games.deltanet_kernels import (
    DELTANET_KERNEL_FIELD,
    assert_one_deltanet_kernel,
    bound_deltanet_kernels,
    bridge_and_check_decode_kernel,
)
from games.eval_model import ADAPTER_PROBE_PROMPTS
from games.eval_sampler import SAMPLER_TRAINING_DISTRIBUTION, eval_sampling
from games.interp_cells import (
    concept_activations,
    digest_of_strings,
    load_ladder,
    load_stimuli,
    pair_layout,
    sha256_of_file,
    stimuli_digest,
)
from games.lora import adapter_config_identity, attach_adapter, load_adapter_base
from games.parsing import parse_action, strip_thinking
from games.payoffs import COOPERATE
from games.prompts import (
    LABEL_PRINT_ORDERS,
    ROW_COLUMNS,
    generate_framing_prompt_rows,
    generate_prompt_rows,
)
from reward_hacking.interp.directions import (
    _decoder_layers,  # pyright: ignore[reportPrivateUsage]  # shared trunk-layer resolver
    cosine,
    matched_norm_random_direction,
)
from reward_hacking.interp.generation_capture import resolved_sampler
from reward_hacking.interp.jacobian import resolve_weights_identity
from reward_hacking.interp.jsonl_resume import resume_records
from reward_hacking.interp.steering import ablation_hook, residual_intervention, steering_hook
from reward_hacking.model_backend import HFBackend, SamplingConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager

    from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

    from games.interp_cells import CapturedCell

logger = logging.getLogger(__name__)


class AdapterCapableModel(Protocol):
    """The small model surface needed by the adapter positive control."""

    device: torch.device

    def disable_adapter(self) -> AbstractContextManager[None]:
        """Temporarily disable the live adapter for a base-model comparison."""
        ...

    def forward(self, **kwargs: object) -> AdapterForwardOutput:
        """Run one forward pass and expose its logits."""
        ...


class AdapterForwardOutput(Protocol):
    """The model output field needed by the adapter positive control."""

    logits: torch.Tensor


GAME_ID = "twin-pd"
RENDER_GRADING = "self"
"""Grading passed to the row generator. Prompt text is identical across gradings (the grading is a
reward-time fact, not a prompt fact); this constant only names which registry entry rendered the
rows, and it is recorded in every payload."""

ACTION_PREFIX = "<action>"
SPLIT_EVAL = "eval"
SPLIT_TRAIN = "train"

CONDITION_NONE = "none"
CONDITION_STEER_UP = "steer:+"
CONDITION_STEER_DOWN = "steer:-"
CONDITION_PLACEBO_UP = "placebo:+"
CONDITION_PLACEBO_DOWN = "placebo:-"
CONDITION_ABLATE_REAL = "ablate:real"
CONDITION_ABLATE_PLACEBO = "ablate:placebo"
STEERING_CONDITIONS: tuple[str, ...] = (
    CONDITION_STEER_UP,
    CONDITION_STEER_DOWN,
    CONDITION_PLACEBO_UP,
    CONDITION_PLACEBO_DOWN,
)
ABLATION_CONDITIONS: tuple[str, ...] = (CONDITION_ABLATE_REAL, CONDITION_ABLATE_PLACEBO)

DEFAULT_ALPHA_MULTIPLIERS = (0.5, 1.0, 2.0)
DEFAULT_N_SAMPLES = 4
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_NEW_TOKENS = 4096
MIN_USABLE_ROW_FRACTION = 0.5

# The diagnostic intervention deliberately mixes two kinds of held-out decision.  The manifest
# owns the prompt text, so these labels are the only diagnostic information that belongs in tracked
# code and they are enough to keep the two strata from silently disappearing into one average.
DIAGNOSTIC_FAMILY_COSTLY_HELPING = "costly-helping"
DIAGNOSTIC_FAMILY_PAYOFF_CONTROL = "payoff-control"
DIAGNOSTIC_FAMILIES: tuple[str, str] = (
    DIAGNOSTIC_FAMILY_COSTLY_HELPING,
    DIAGNOSTIC_FAMILY_PAYOFF_CONTROL,
)
CUSTOM_DIAGNOSTIC_ROW_COUNT = 8
CUSTOM_ROW_SCHEMA = "steering-generation-rows/v1"
CUSTOM_DIAGNOSTIC_ROW_SCHEMA = "cooperation-generalization-diagnostic-rows/v1"
DIAGNOSTIC_PROFILE_COOPERATION_GENERALIZATION = "cooperation-generalization"
CUSTOM_ROW_KIND_MATRIX = "matrix"
CUSTOM_ROW_KIND_BINARY_ALLOCATION = "binary-allocation"
CUSTOM_ALLOCATION_COLUMNS: frozenset[str] = frozenset(
    {
        "row_kind",
        "allocation_context",
        "allocation_action_a_description",
        "allocation_action_b_description",
        "allocation_prompt_template",
        "allocation_action_a_own_payoff",
        "allocation_action_a_counterpart_payoff",
        "allocation_action_b_own_payoff",
        "allocation_action_b_counterpart_payoff",
    }
)

# Custom rows use the existing renderer's complete schema, which keeps payoff/control metadata and
# prompt identity together.  The experiment profile adds only its diagnostic family annotation.
CUSTOM_ROW_BASE_COLUMNS: frozenset[str] = frozenset(ROW_COLUMNS)
CUSTOM_ROW_COLUMNS: frozenset[str] = CUSTOM_ROW_BASE_COLUMNS | frozenset({"diagnostic_family"})
CUSTOM_ALLOCATION_ROW_COLUMNS: frozenset[str] = CUSTOM_ROW_BASE_COLUMNS | CUSTOM_ALLOCATION_COLUMNS
CUSTOM_ALLOCATION_DIAGNOSTIC_ROW_COLUMNS: frozenset[str] = (
    CUSTOM_ALLOCATION_ROW_COLUMNS | frozenset({"diagnostic_family"})
)
DEFAULT_PLACEBO_SEED_OFFSET = 1_000_003
SELECTED_TARGET_SCHEMA = "cooperation-generalization-selected-target/v1"
SHA256_HEX_LENGTH = 64
SELECTED_TARGET_COLUMNS: frozenset[str] = frozenset(
    {
        "schema",
        "version",
        "target_construct",
        "direction",
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
)
ADAPTER_WEIGHT_FILENAMES: tuple[str, ...] = ("adapter_model.safetensors", "adapter_model.bin")

ALLOCATION_TEMPLATE_FIELDS: frozenset[str] = frozenset(
    {
        "context",
        "first_label",
        "second_label",
        "first_description",
        "second_description",
        "first_own_payoff",
        "first_counterpart_payoff",
        "second_own_payoff",
        "second_counterpart_payoff",
    }
)

SWEEP_FILENAME = "logit_sweep.json"
GENERATION_SUMMARY_FILENAME = "steering_summary.json"
GENERATION_RECORDS_FILENAME = "steering_records.jsonl"


# --------------------------------------------------------------------------------------
# Direction files: the {layer: tensor} format the trajectory module writes
# --------------------------------------------------------------------------------------


def load_direction_file(path: Path) -> dict[int, torch.Tensor]:
    """Load a ``{layer: tensor}`` directions file, refusing anything shaped differently.

    The refusal matters because ``torch.load`` happily returns whatever was saved; a state dict or
    a bare tensor fed here would otherwise surface as a confusing KeyError deep inside a sweep.
    """
    payload: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} does not hold a non-empty dict; got {type(payload).__name__}.")
    directions: dict[int, torch.Tensor] = {}
    for key, value in cast("dict[object, object]", payload).items():
        if not isinstance(key, int) or not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise ValueError(
                f"{path} must map int layer -> 1-D tensor; got key {key!r} -> "
                f"{type(value).__name__}."
            )
        directions[key] = value.float()
    return directions


def parse_named_specs(specs: Sequence[str], *, flag: str) -> dict[str, str]:
    """Parse repeated ``NAME=VALUE`` flags into an ordered mapping, refusing duplicates."""
    parsed: dict[str, str] = {}
    for spec in specs:
        name, separator, value = spec.partition("=")
        if not separator or not name or not value:
            raise ValueError(f"{flag} expects NAME=VALUE, got {spec!r}.")
        if name in parsed:
            raise ValueError(f"{flag} names {name!r} twice.")
        parsed[name] = value
    return parsed


def load_named_directions(specs: Sequence[str]) -> dict[str, dict[int, torch.Tensor]]:
    """Load every ``--direction NAME=PATH`` into name -> {layer: tensor}."""
    if not specs:
        raise ValueError("at least one --direction NAME=PATH is required.")
    return {
        name: load_direction_file(Path(path))
        for name, path in parse_named_specs(specs, flag="--direction").items()
    }


def load_selected_target(  # noqa: C901, PLR0912, PLR0915 - strict artifact contract
    path: Path,
    directions: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    direction_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Load one calibration-only target and translate it into a generation cell.

    The geometry leg owns this artifact.  Keeping the target's direction path and digest beside
    the chosen layer and scale prevents a caller from silently pairing a calibration result with a
    same-named direction file from another endpoint.  ``path`` is retained in the returned payload
    so both the resume identity and the report bind the exact selection artifact bytes.
    """
    target_path = path.resolve()
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"selected target {target_path} must be a JSON object.")
    actual_columns = frozenset(payload)
    if actual_columns != SELECTED_TARGET_COLUMNS:
        missing = sorted(SELECTED_TARGET_COLUMNS - actual_columns)
        extra = sorted(actual_columns - SELECTED_TARGET_COLUMNS)
        raise ValueError(
            f"selected target {target_path} has unsupported fields; missing={missing} extra={extra}."
        )
    if payload["schema"] != SELECTED_TARGET_SCHEMA or payload["version"] != 1:
        raise ValueError(
            f"selected target {target_path} has schema/version "
            f"{payload['schema']!r}/{payload['version']!r}; expected "
            f"{SELECTED_TARGET_SCHEMA!r}/1."
        )
    direction_name = payload["direction"]
    target_construct = payload["target_construct"]
    if target_construct != direction_name:
        raise ValueError(
            f"selected target construct {target_construct!r} does not match direction "
            f"{direction_name!r}."
        )
    direction_path = Path(str(payload["direction_path"])).expanduser().resolve()
    if not isinstance(direction_name, str) or not direction_name.strip():
        raise TypeError("selected target direction must be a non-empty string.")
    if direction_name not in directions:
        raise ValueError(
            f"selected target names direction {direction_name!r}, but --direction provides "
            f"{sorted(directions)}."
        )
    if direction_paths is not None:
        expected_path = direction_paths[direction_name].expanduser().resolve()
        if direction_path != expected_path:
            raise ValueError(
                f"selected target direction_path {direction_path} does not match --direction "
                f"{direction_name}={expected_path}."
            )
    if not direction_path.is_file():
        raise FileNotFoundError(
            f"selected target direction_path {direction_path} is not a direction file."
        )
    expected_digest = payload["direction_sha256"]
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise ValueError(
            "selected target direction_sha256 must be a lowercase 64-character hex digest."
        )
    actual_digest = sha256_of_file(direction_path)
    if expected_digest != actual_digest:
        raise ValueError(
            f"selected target direction_sha256 {expected_digest} does not match {direction_path} "
            f"({actual_digest})."
        )
    layer = payload["layer"]
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise TypeError("selected target layer must be an integer.")
    if layer not in directions[direction_name]:
        raise ValueError(
            f"selected target layer {layer} is absent from direction {direction_name!r}; "
            f"available layers are {sorted(directions[direction_name])}."
        )
    magnitude = payload["magnitude"]
    alpha_multiplier = payload["alpha_multiplier"]
    calibration_metric = payload["calibration_metric"]
    for field, value in (
        ("magnitude", magnitude),
        ("alpha_multiplier", alpha_multiplier),
        ("calibration_metric", calibration_metric),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"selected target {field} must be a finite number; got {value!r}.")
    if float(magnitude) <= 0.0 or float(alpha_multiplier) <= 0.0:
        raise ValueError("selected target magnitude and alpha_multiplier must be positive.")
    expected_magnitude = float(alpha_multiplier) * float(directions[direction_name][layer].norm())
    if not math.isclose(float(magnitude), expected_magnitude, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(
            f"selected target magnitude {magnitude} does not equal alpha_multiplier "
            f"{alpha_multiplier} times direction norm ({expected_magnitude})."
        )
    rationale = payload["calibration_rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("selected target calibration_rationale must be non-empty text.")
    expected_effect = payload["expected_effect"]
    if not isinstance(expected_effect, str) or not expected_effect.strip():
        raise ValueError("selected target expected_effect must be non-empty text.")
    if payload["expectation_recorded_before_intervention"] is not True:
        raise ValueError("selected target expectation_recorded_before_intervention must be true.")
    if payload["exploratory"] is not True:
        raise ValueError("selected target exploratory must be true.")
    report_digest = payload["geometry_report_sha256"]
    if (
        not isinstance(report_digest, str)
        or len(report_digest) != SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in report_digest)
    ):
        raise ValueError(
            "selected target geometry_report_sha256 must be a lowercase 64-character hex digest."
        )
    report_path_value = payload["geometry_report_path"]
    if not isinstance(report_path_value, str) or not report_path_value.strip():
        raise ValueError("selected target geometry_report_path must be a non-empty path.")
    report_path = Path(report_path_value).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(
            f"selected target geometry_report_path {report_path} is not a report file."
        )
    actual_report_digest = sha256_of_file(report_path)
    if report_digest != actual_report_digest:
        raise ValueError(
            f"selected target geometry_report_sha256 {report_digest} does not match "
            f"{report_path} ({actual_report_digest})."
        )
    return {
        "path": str(target_path),
        "sha256": sha256_of_file(target_path),
        "payload": dict(payload),
        "geometry_report_path": str(report_path),
        "geometry_report_sha256": actual_report_digest,
        "cell": {
            "direction": direction_name,
            "layer": layer,
            "alpha_multiplier": float(alpha_multiplier),
        },
    }


def adapter_identity(adapter_dir: Path, *, applied_adapter_weights: int) -> dict[str, Any]:
    """Describe the exact un-merged adapter served by a generation run."""
    resolved = adapter_dir.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"adapter {resolved} is not a directory.")
    weight_paths = [
        resolved / name for name in ADAPTER_WEIGHT_FILENAMES if (resolved / name).is_file()
    ]
    if len(weight_paths) != 1:
        raise ValueError(
            f"adapter {resolved} must contain exactly one of {ADAPTER_WEIGHT_FILENAMES}; "
            f"found {[path.name for path in weight_paths]}"
        )
    return {
        "path": str(resolved),
        "config": adapter_config_identity(resolved),
        "weights_filename": weight_paths[0].name,
        "weights_sha256": sha256_of_file(weight_paths[0]),
        "applied_adapter_weights": int(applied_adapter_weights),
    }


def assert_adapter_changes_forward(
    model: AdapterCapableModel, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str]
) -> None:
    """Refuse an un-merged adapter whose forward pass is bitwise identical to its base."""
    if not prompts:
        raise ValueError("adapter positive-control prompts cannot be empty.")
    chats = [
        cast(
            "str",
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            ),
        )
        for prompt in prompts
    ]
    inputs = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        with model.disable_adapter():
            base_logits = model.forward(**inputs).logits
        adapted_logits = model.forward(**inputs).logits
    if torch.equal(base_logits, adapted_logits):
        raise RuntimeError(
            "the un-merged adapter changed no forward logits on the positive-control prompts; "
            "refusing to spend the steering generation run on a base model."
        )


def parse_named_layers(
    specs: Sequence[str], directions: Mapping[str, dict[int, torch.Tensor]]
) -> dict[str, list[int]]:
    """Parse ``--layers NAME=16,17,18`` and check each layer exists in its named direction file."""
    parsed = parse_named_specs(specs, flag="--layers")
    unknown = sorted(set(parsed) - set(directions))
    if unknown:
        raise ValueError(f"--layers names {unknown} with no matching --direction.")
    missing = sorted(set(directions) - set(parsed))
    if missing:
        raise ValueError(f"--direction {missing} have no matching --layers.")
    layers: dict[str, list[int]] = {}
    for name, raw in parsed.items():
        wanted = [int(part) for part in raw.split(",") if part.strip()]
        absent = sorted(set(wanted) - set(directions[name]))
        if absent:
            raise ValueError(f"layers {absent} are not in the {name!r} direction file.")
        layers[name] = wanted
    return layers


def placebo_for(
    direction: torch.Tensor, *, seed: int, direction_index: int, layer: int
) -> torch.Tensor:
    """One reproducible matched-norm placebo per (direction, layer)."""
    generator = torch.Generator().manual_seed(seed + 1009 * direction_index + layer)
    return matched_norm_random_direction(direction, generator)


# --------------------------------------------------------------------------------------
# fit-direction
# --------------------------------------------------------------------------------------

PAIRS_CHOICES = ("all", "even", "odd")


def fit_pair_indices(
    cell: CapturedCell, stimulus_set: str, *, positive_side: str, pairs: str
) -> tuple[Any, torch.Tensor | None]:
    """Return the (layout, pair index tensor or None) the fit runs on."""
    layout = pair_layout(cell, stimulus_set, positive_side=positive_side)
    if pairs == "all":
        return layout, None
    return layout, layout.half({"even": 0, "odd": 1}[pairs])


def run_fit_direction(args: argparse.Namespace) -> dict[str, Any]:
    """Fit per-layer diff-of-means directions from one cell and save them with a JSON sidecar."""
    stimuli = load_stimuli(args.stimuli)
    digest = stimuli_digest(stimuli)
    ladder = load_ladder(args.capture_root, arms=None, steps=None, stimuli_sha256=digest)
    cell = ladder.cell(args.arm, args.step)
    layout, pairs = fit_pair_indices(
        cell, args.set, positive_side=args.positive_side, pairs=args.pairs
    )

    directions: dict[int, torch.Tensor] = {}
    for layer in range(cell.identity.n_layers):
        concept = concept_activations(cell, args.set, args.pooling, layer, layout, pairs=pairs)
        directions[layer] = concept.diff_of_means().float().cpu()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(directions, args.out)

    comparison: dict[str, float] | None = None
    if args.compare is not None:
        other = load_direction_file(args.compare)
        shared = sorted(set(directions) & set(other))
        if not shared:
            raise ValueError(f"--compare {args.compare} shares no layers with the fitted file.")
        comparison = {str(layer): cosine(directions[layer], other[layer]) for layer in shared}

    sidecar: dict[str, Any] = {
        "capture_root": str(args.capture_root),
        "cell": cell.label,
        "identity": cell.identity.to_payload(),
        "stimuli_sha256": digest,
        "set": args.set,
        "pooling": args.pooling,
        "positive_side": args.positive_side,
        "pairs": args.pairs,
        "n_fit_pairs": layout.n_pairs if pairs is None else int(pairs.numel()),
        "norms": {str(layer): float(tensor.norm()) for layer, tensor in directions.items()},
        "compare": None if args.compare is None else str(args.compare),
        "compare_cosine_by_layer": comparison,
    }
    sidecar_path = args.out.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    logger.info(f"fitted directions written, out={args.out} sidecar={sidecar_path}")
    if comparison is not None:
        worst = min(comparison.items(), key=lambda item: item[1])
        logger.info(f"two-path direction agreement: worst layer {worst[0]} cosine {worst[1]:.4f}")
    return sidecar


# --------------------------------------------------------------------------------------
# Prompt rows and the forced-choice rendering
# --------------------------------------------------------------------------------------


def steering_rows(splits: Sequence[str]) -> list[dict[str, Any]]:
    """Twin-pd rows for the requested splits, in BOTH print orders, in a stable order."""
    rows: list[dict[str, Any]] = []
    for split in splits:
        for order in LABEL_PRINT_ORDERS:
            rows.extend(
                generate_prompt_rows(GAME_ID, RENDER_GRADING, split=split, label_print_order=order)
            )
    return rows


def render_binary_allocation_prompt(row: Mapping[str, Any]) -> str:
    """Render a private allocation template from its labels, prose and numeric outcomes."""
    first_label, second_label = (
        (str(row["label_a"]), str(row["label_b"]))
        if row["label_print_order"] == "canonical"
        else (str(row["label_b"]), str(row["label_a"]))
    )
    action_fields = {
        str(row["label_a"]): (
            str(row["allocation_action_a_description"]),
            row["allocation_action_a_own_payoff"],
            row["allocation_action_a_counterpart_payoff"],
        ),
        str(row["label_b"]): (
            str(row["allocation_action_b_description"]),
            row["allocation_action_b_own_payoff"],
            row["allocation_action_b_counterpart_payoff"],
        ),
    }
    first_description, first_own_payoff, first_counterpart_payoff = action_fields[first_label]
    second_description, second_own_payoff, second_counterpart_payoff = action_fields[second_label]
    template = str(row["allocation_prompt_template"])
    parsed = list(Formatter().parse(template))
    fields = [field for _, field, _, _ in parsed if field is not None]
    if frozenset(fields) != ALLOCATION_TEMPLATE_FIELDS or len(fields) != len(
        ALLOCATION_TEMPLATE_FIELDS
    ):
        raise ValueError(
            f"custom allocation prompt template must use only the exact placeholder set "
            f"{sorted(ALLOCATION_TEMPLATE_FIELDS)} and include each one; got {fields}."
        )
    if any(conversion is not None or format_spec for _, _, format_spec, conversion in parsed):
        raise ValueError(
            "custom allocation prompt template cannot use conversions or format specs."
        )
    values = {
        "context": str(row["allocation_context"]),
        "first_label": first_label,
        "second_label": second_label,
        "first_description": first_description,
        "second_description": second_description,
        "first_own_payoff": _format_custom_number(first_own_payoff),
        "first_counterpart_payoff": _format_custom_number(first_counterpart_payoff),
        "second_own_payoff": _format_custom_number(second_own_payoff),
        "second_counterpart_payoff": _format_custom_number(second_counterpart_payoff),
    }
    return template.format(**values)


def _format_custom_number(value: float) -> str:
    """Format a manifest numeric value without adding cosmetic decimal noise to prompts."""
    return f"{float(value):g}"


def _custom_row_manifest_payload(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Read the JSON object that owns a custom diagnostic row roster.

    Prompt text is intentionally read at runtime.  The manifest is therefore an object rather than
    a Python constant, and its optional schema marker gives a future format a clean refusal instead
    of letting a changed row shape reach the model.  A top-level ``rows`` list is accepted for
    convenient hand-authored manifests; the row validation below remains strict either way.
    """
    manifest_path = path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(
            f"custom diagnostic row manifest {path} must be a JSON object with a 'rows' list."
        )
    schema = payload.get("schema", CUSTOM_ROW_SCHEMA)
    if schema not in (CUSTOM_ROW_SCHEMA, CUSTOM_DIAGNOSTIC_ROW_SCHEMA):
        raise ValueError(
            f"custom diagnostic row manifest {manifest_path} has schema {schema!r}; expected "
            f"one of {CUSTOM_ROW_SCHEMA!r}, {CUSTOM_DIAGNOSTIC_ROW_SCHEMA!r}."
        )
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError(
            f"custom diagnostic row manifest {manifest_path} must contain a 'rows' list."
        )
    if not all(isinstance(row, dict) for row in raw_rows):
        raise ValueError(f"custom diagnostic row manifest {manifest_path} has a non-object row.")
    return cast("list[dict[str, Any]]", raw_rows), str(schema)


def _validate_custom_row(  # noqa: C901, PLR0912 - schema guard
    index: int, row: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one generic custom row and return an isolated mutable copy."""
    validated = dict(row)
    actual = frozenset(validated)
    allowed_columns = (
        CUSTOM_ROW_BASE_COLUMNS,
        CUSTOM_ROW_COLUMNS,
        CUSTOM_ALLOCATION_ROW_COLUMNS,
        CUSTOM_ALLOCATION_DIAGNOSTIC_ROW_COLUMNS,
    )
    if actual not in allowed_columns:
        missing = sorted(CUSTOM_ROW_BASE_COLUMNS - actual)
        extra = sorted(actual - CUSTOM_ROW_BASE_COLUMNS)
        raise ValueError(
            f"custom row {index} ({validated.get('prompt_id')!r}) has schema {sorted(actual)}; "
            f"missing={missing} extra={extra} expected one of "
            f"{[sorted(columns) for columns in allowed_columns]}."
        )
    text_fields = (
        "prompt",
        "prompt_id",
        "game_id",
        "grading",
        "label_a",
        "label_b",
        "coop_label",
        "reskin_id",
        "payoff_variant",
        "label_print_order",
    )
    for field in text_fields:
        value = validated[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"custom row {index} field {field!r} must be a non-empty string; got {value!r}."
            )
    if not isinstance(validated["opponent_rule"], str):
        raise TypeError(
            f"custom row {index} field 'opponent_rule' must be a string; "
            f"got {validated['opponent_rule']!r}."
        )
    row_kind = validated.get("row_kind", CUSTOM_ROW_KIND_MATRIX)
    if row_kind not in (CUSTOM_ROW_KIND_MATRIX, CUSTOM_ROW_KIND_BINARY_ALLOCATION):
        raise ValueError(
            f"custom row {index} field 'row_kind' must be one of "
            f"{CUSTOM_ROW_KIND_MATRIX!r}, {CUSTOM_ROW_KIND_BINARY_ALLOCATION!r}; got {row_kind!r}."
        )
    is_allocation = row_kind == CUSTOM_ROW_KIND_BINARY_ALLOCATION
    if is_allocation != (
        actual in (CUSTOM_ALLOCATION_ROW_COLUMNS, CUSTOM_ALLOCATION_DIAGNOSTIC_ROW_COLUMNS)
    ):
        raise ValueError(
            f"custom row {index} ({validated.get('prompt_id')!r}) has row_kind {row_kind!r} "
            "that does not match its allocation columns."
        )
    for field in (
        "payoff_cc",
        "payoff_cd",
        "payoff_dc",
        "payoff_dd",
        "endowment",
        "windfall",
        "team_size",
        "contribution_threshold",
        "prize",
        "opp_coop_prob",
        "n_rounds",
        "transfer_multiplier",
        "stated_return_fraction",
        "stated_match_prob",
        "n_levels",
        "benefit_per_level",
        "cost_per_level",
    ):
        value = validated[field]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(
                f"custom row {index} field {field!r} must be a finite number; got {value!r}."
            )
        if not math.isfinite(float(value)):
            raise ValueError(f"custom row {index} field {field!r} must be finite; got {value!r}.")
    if is_allocation:
        for field in (
            "allocation_action_a_own_payoff",
            "allocation_action_a_counterpart_payoff",
            "allocation_action_b_own_payoff",
            "allocation_action_b_counterpart_payoff",
        ):
            value = validated[field]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(
                    f"custom row {index} field {field!r} must be a finite number; got {value!r}."
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"custom row {index} field {field!r} must be finite; got {value!r}."
                )
        for field in (
            "allocation_context",
            "allocation_action_a_description",
            "allocation_action_b_description",
            "allocation_prompt_template",
        ):
            value = validated[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"custom row {index} field {field!r} must be a non-empty string; got {value!r}."
                )
    if validated["label_a"] == validated["label_b"]:
        raise ValueError(
            f"custom row {index} ({validated['prompt_id']!r}) has identical labels; "
            "the action parser needs a distinct pair."
        )
    if validated["coop_label"] not in (validated["label_a"], validated["label_b"]):
        raise ValueError(
            f"custom row {index} ({validated['prompt_id']!r}) has coop_label "
            f"{validated['coop_label']!r} outside its label pair."
        )
    if validated["label_print_order"] not in LABEL_PRINT_ORDERS:
        raise ValueError(
            f"custom row {index} ({validated['prompt_id']!r}) has label_print_order "
            f"{validated['label_print_order']!r}; expected one of {list(LABEL_PRINT_ORDERS)}."
        )
    if is_allocation:
        expected_prompt = render_binary_allocation_prompt(validated)
        if validated["prompt"] != expected_prompt:
            raise ValueError(
                f"custom allocation row {index} ({validated['prompt_id']!r}) prompt does not "
                "match its labels, printed order, descriptions and payoff values."
            )
    return validated


def validate_custom_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate the reusable row shape needed by the steering generation leg.

    A generic manifest may carry the optional ``diagnostic_family`` annotation, but all other
    fields are exact.  The payoff cells are retained even though parsing only needs labels: a
    custom payoff-control row must remain scoreable from its saved record without reconstructing
    a table from private prompt text.
    """
    validated = [_validate_custom_row(index, row) for index, row in enumerate(rows)]

    prompt_ids = [str(row["prompt_id"]) for row in validated]
    duplicates = sorted({prompt_id for prompt_id in prompt_ids if prompt_ids.count(prompt_id) > 1})
    if duplicates:
        raise ValueError(f"custom rows contain duplicate prompt_id values {duplicates}.")
    return validated


def validate_custom_diagnostic_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and copy the fixed eight-row diagnostic intervention roster.

    The two diagnostic families and the two printed label orders are both four rows each.  This
    is a structural guard, not a claim about the contents of the private prompts: it catches an
    accidental one-family or one-order manifest before a model is loaded or a condition is run.
    """
    validated = validate_custom_rows(rows)
    if len(validated) != CUSTOM_DIAGNOSTIC_ROW_COUNT:
        raise ValueError(
            f"custom diagnostic intervention requires exactly {CUSTOM_DIAGNOSTIC_ROW_COUNT} rows; "
            f"got {len(rows)}."
        )
    for index, row in enumerate(validated):
        actual = frozenset(row)
        if "diagnostic_family" not in row:
            raise ValueError(
                f"custom diagnostic row {index} ({row.get('prompt_id')!r}) is missing "
                "diagnostic_family."
            )
        if row["diagnostic_family"] not in DIAGNOSTIC_FAMILIES:
            raise ValueError(
                f"custom diagnostic row {index} ({row['prompt_id']!r}) has diagnostic_family "
                f"{row['diagnostic_family']!r}; expected one of {list(DIAGNOSTIC_FAMILIES)}."
            )
        expected_kind = (
            CUSTOM_ROW_KIND_BINARY_ALLOCATION
            if row["diagnostic_family"] == DIAGNOSTIC_FAMILY_COSTLY_HELPING
            else CUSTOM_ROW_KIND_MATRIX
        )
        expected_columns = (
            CUSTOM_ALLOCATION_DIAGNOSTIC_ROW_COLUMNS
            if expected_kind == CUSTOM_ROW_KIND_BINARY_ALLOCATION
            else CUSTOM_ROW_COLUMNS
        )
        if actual != expected_columns:
            missing = sorted(expected_columns - actual)
            extra = sorted(actual - expected_columns)
            raise ValueError(
                f"custom diagnostic row {index} ({row.get('prompt_id')!r}) has schema {sorted(actual)}; "
                f"missing={missing} extra={extra} expected={sorted(expected_columns)}."
            )
        actual_kind = row.get("row_kind", CUSTOM_ROW_KIND_MATRIX)
        if actual_kind != expected_kind:
            raise ValueError(
                f"custom diagnostic row {index} ({row['prompt_id']!r}) in family "
                f"{row['diagnostic_family']!r} must use row_kind {expected_kind!r}; "
                f"got {actual_kind!r}."
            )

    family_counts = {
        family: sum(row["diagnostic_family"] == family for row in validated)
        for family in DIAGNOSTIC_FAMILIES
    }
    if set(family_counts.values()) != {CUSTOM_DIAGNOSTIC_ROW_COUNT // len(DIAGNOSTIC_FAMILIES)}:
        raise ValueError(
            "custom diagnostic rows must contain four costly-helping and four payoff-control rows; "
            f"got {family_counts}."
        )
    order_counts = {
        order: sum(row["label_print_order"] == order for row in validated)
        for order in LABEL_PRINT_ORDERS
    }
    if set(order_counts.values()) != {CUSTOM_DIAGNOSTIC_ROW_COUNT // len(LABEL_PRINT_ORDERS)}:
        raise ValueError(
            "custom diagnostic rows must balance the two label print orders at four each; "
            f"got {order_counts}."
        )
    family_order_counts = {
        family: {
            order: sum(
                row["diagnostic_family"] == family and row["label_print_order"] == order
                for row in validated
            )
            for order in LABEL_PRINT_ORDERS
        }
        for family in DIAGNOSTIC_FAMILIES
    }
    expected_per_cell = CUSTOM_DIAGNOSTIC_ROW_COUNT // (
        len(DIAGNOSTIC_FAMILIES) * len(LABEL_PRINT_ORDERS)
    )
    if any(
        count != expected_per_cell
        for counts_by_order in family_order_counts.values()
        for count in counts_by_order.values()
    ):
        raise ValueError(
            "custom diagnostic rows must contain two rows from each print order within each "
            f"diagnostic family; got {family_order_counts}."
        )
    return validated


def load_custom_diagnostic_rows(path: Path) -> list[dict[str, Any]]:
    """Load and validate the private eight-row custom diagnostic manifest."""
    rows, _schema = _custom_row_manifest_payload(path)
    return validate_custom_diagnostic_rows(rows)


def load_custom_row_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a reusable custom steering row manifest without applying an experiment profile."""
    rows, _schema = _custom_row_manifest_payload(path)
    return validate_custom_rows(rows)


def rendered_rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Digest row order, IDs, metadata and rendered prompt text for resume identity."""
    return digest_of_strings(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )


def generation_rows(
    counterpart_framing: str | None,
    row_manifest: Path | None = None,
    diagnostic_profile: str | None = None,
) -> list[dict[str, Any]]:
    """Return default held-out rows or the validated private custom diagnostic roster.

    None is the trained rendering (`generate_prompt_rows`, battery prompt_ids). A framing id
    renders the same rows with only the "About the other side" paragraph swapped
    (`generate_framing_prompt_rows`; its `twin` is byte-identical prompt text under
    framing-tagged prompt_ids). The framing swaps WHICH prompts the cell generates on and
    nothing else -- conditions, their index-derived seeds, and the placebo vector are untouched,
    which is what keeps a framing cell seed-comparable to the twin cells the interp arc banked.

    ``row_manifest`` is already rendered and therefore cannot be combined with a framing.  Its rows
    remain in manifest order; ``--conditions`` later filters intervention conditions and never
    selects or renumbers these prompts.  The cooperation-generalization profile adds the exact
    eight-row, two-family, per-family order balance required by section 5 of its plan.
    """
    if row_manifest is not None:
        if counterpart_framing is not None:
            raise ValueError(
                "--row-manifest already contains rendered prompts and cannot be combined with "
                "--counterpart-framing."
            )
        if diagnostic_profile == DIAGNOSTIC_PROFILE_COOPERATION_GENERALIZATION:
            return load_custom_diagnostic_rows(row_manifest)
        if diagnostic_profile is not None:
            raise ValueError(
                f"unknown diagnostic profile {diagnostic_profile!r}; expected "
                f"{DIAGNOSTIC_PROFILE_COOPERATION_GENERALIZATION!r}."
            )
        return load_custom_row_manifest(row_manifest)
    if diagnostic_profile is not None:
        raise ValueError("--diagnostic-profile requires --row-manifest.")
    if counterpart_framing is None:
        return [
            row
            for order in LABEL_PRINT_ORDERS
            for row in generate_prompt_rows(
                GAME_ID, RENDER_GRADING, split=SPLIT_EVAL, label_print_order=order
            )
        ]
    return [
        row
        for order in LABEL_PRINT_ORDERS
        for row in generate_framing_prompt_rows(
            GAME_ID,
            RENDER_GRADING,
            framing_id=counterpart_framing,
            split=SPLIT_EVAL,
            label_print_order=order,
        )
    ]


def render_forced_choice(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    """Render a row thinking-off with the assistant turn prefilled up to ``<action>``.

    The completion format instruction in every prompt ends with a single ``<action>LABEL</action>``
    tag, so with the tag opened for it the model's next token is the label's first token -- a
    forced-choice readout with no generation. Thinking off, because a thinking-on render would put
    hundreds of deliberation tokens between here and the action.
    """
    chat = cast(
        "str",
        tokenizer.apply_chat_template(  # pyright: ignore[reportAttributeAccessIssue]
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
    )
    return chat + ACTION_PREFIX


def label_first_tokens(
    encode: Callable[[str], list[int]], row: Mapping[str, Any]
) -> tuple[int, int] | None:
    """First token ids of (coop label, defect label) right after ``<action>``, or None on collision.

    A pair of labels sharing their first token cannot be told apart at one readout position; such a
    row is skipped and counted rather than silently contributing a zero gap.
    """
    coop_label = str(row["coop_label"])
    labels = (str(row["label_a"]), str(row["label_b"]))
    defect_label = labels[1] if coop_label == labels[0] else labels[0]
    coop_ids = encode(coop_label)
    defect_ids = encode(defect_label)
    if not coop_ids or not defect_ids or coop_ids[0] == defect_ids[0]:
        return None
    return coop_ids[0], defect_ids[0]


# --------------------------------------------------------------------------------------
# logit-sweep
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepCell:
    """One intervention point of the sweep grid."""

    direction: str
    layer: int
    alpha_multiplier: float
    condition: str


def sweep_grid(
    layers_by_direction: Mapping[str, Sequence[int]], alpha_multipliers: Sequence[float]
) -> list[SweepCell]:
    """Build the full condition grid; the shared baseline is excluded (it runs once, separately)."""
    cells: list[SweepCell] = []
    for direction, layers in layers_by_direction.items():
        for layer in layers:
            for multiplier in alpha_multipliers:
                cells.extend(
                    SweepCell(direction, layer, multiplier, condition)
                    for condition in STEERING_CONDITIONS
                )
    return cells


def hook_for(
    condition: str, real: torch.Tensor, placebo: torch.Tensor, alpha_raw: float
) -> Callable[[object, object, object], object]:
    """Map a condition name to its hook. The table IS the code path -- no per-arm branches."""
    table: dict[str, Callable[[], Callable[[object, object, object], object]]] = {
        CONDITION_STEER_UP: lambda: steering_hook(real, alpha_raw),
        CONDITION_STEER_DOWN: lambda: steering_hook(real, -alpha_raw),
        CONDITION_PLACEBO_UP: lambda: steering_hook(placebo, alpha_raw),
        CONDITION_PLACEBO_DOWN: lambda: steering_hook(placebo, -alpha_raw),
        CONDITION_ABLATE_REAL: lambda: ablation_hook(real),
        CONDITION_ABLATE_PLACEBO: lambda: ablation_hook(placebo),
    }
    if condition not in table:
        raise ValueError(f"unknown condition {condition!r}; expected one of {sorted(table)}.")
    return table[condition]()


def _norm_recording_hook(
    store: dict[int, list[float]], layer: int
) -> Callable[[object, object, object], None]:
    """Read-only hook recording the mean last-position residual norm at ``layer``."""

    def hook(module: object, inputs: object, output: object) -> None:
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        norms = cast("torch.Tensor", hidden)[:, -1, :].float().norm(dim=-1)
        store[layer].append(float(norms.mean().item()))

    return hook


@torch.no_grad()
def _forced_choice_gaps(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_pairs: Sequence[tuple[int, int]],
    *,
    batch_size: int,
) -> list[float]:
    """Per-row coop-minus-defect logit gap at the last position, batched with left padding."""
    gaps: list[float] = []
    for start in range(0, int(input_ids.shape[0]), batch_size):
        stop = start + batch_size
        device = model.device  # pyright: ignore[reportAttributeAccessIssue]  # set at runtime
        outputs = model(  # pyright: ignore[reportCallIssue]  # HF causal LMs are callable
            input_ids=input_ids[start:stop].to(device),
            attention_mask=attention_mask[start:stop].to(device),
            logits_to_keep=1,
        )
        logits = outputs.logits[:, -1, :].float().cpu()
        for row_offset, (coop_token, defect_token) in enumerate(token_pairs[start:stop]):
            gaps.append(float(logits[row_offset, coop_token] - logits[row_offset, defect_token]))
    return gaps


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of nothing; the caller must guard empty groups.")
    return sum(values) / len(values)


def _gap_summary(
    gaps: Sequence[float], baseline: Sequence[float] | None, orders: Sequence[str]
) -> dict[str, Any]:
    """Mean gap (and paired delta vs baseline) pooled and split by print order."""
    summary: dict[str, Any] = {"mean_gap": _mean(gaps), "n": len(gaps)}
    by_order: dict[str, float] = {}
    for order in sorted(set(orders)):
        by_order[order] = _mean([gap for gap, o in zip(gaps, orders, strict=True) if o == order])
    summary["mean_gap_by_print_order"] = by_order
    if baseline is not None:
        deltas = [gap - base for gap, base in zip(gaps, baseline, strict=True)]
        summary["mean_delta_vs_none"] = _mean(deltas)
        summary["mean_delta_by_print_order"] = {
            order: _mean([d for d, o in zip(deltas, orders, strict=True) if o == order])
            for order in sorted(set(orders))
        }
    return summary


def run_logit_sweep(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915 - one linear driver
    """Run the forced-choice grid and write one JSON payload.

    Bridges the DeltaNet decode kernel before the load even though the sweep is forward-only and
    never dispatches it, so the binding this payload records is the one the generation leg it selects
    for will record: the two legs are read together, and a selection made under one binding and a
    generation run under the other would be two measurements wearing one arm's name.
    """
    directions = load_named_directions(args.direction)
    layers_by_direction = parse_named_layers(args.layers, directions)
    multipliers = [float(part) for part in str(args.alpha_multipliers).split(",") if part.strip()]
    splits = [part for part in str(args.splits).split(",") if part.strip()]

    # Before the load, which imports the modeling module and freezes each kernel's dispatch.
    kernel_bridge = bridge_and_check_decode_kernel()
    backend = HFBackend(args.model, thinking=False)
    model = cast("AutoModelForCausalLM", backend.model)
    tokenizer = backend.tokenizer

    rows = steering_rows(splits)

    def encode(text: str) -> list[int]:
        return cast("list[int]", tokenizer(text, add_special_tokens=False)["input_ids"])

    usable_rows: list[dict[str, Any]] = []
    token_pairs: list[tuple[int, int]] = []
    skipped: list[str] = []
    for row in rows:
        pair = label_first_tokens(encode, row)
        if pair is None:
            skipped.append(str(row["prompt_id"]))
            continue
        usable_rows.append(row)
        token_pairs.append(pair)
    if len(usable_rows) < len(rows) * MIN_USABLE_ROW_FRACTION:
        raise ValueError(
            f"only {len(usable_rows)} of {len(rows)} rows have label pairs distinguishable at one "
            f"token; a sweep on a minority of the corpus would not be the corpus's sweep."
        )

    prompts = [render_forced_choice(tokenizer, str(row["prompt"])) for row in usable_rows]
    orders = [str(row["label_print_order"]) for row in usable_rows]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = cast("torch.Tensor", encoded["input_ids"])
    attention_mask = cast("torch.Tensor", encoded["attention_mask"])

    layer_indices = sorted({layer for ls in layers_by_direction.values() for layer in ls})

    # Baseline pass; the recorded norms turn a raw alpha into "we perturbed the residual by X%".
    norm_store: dict[int, list[float]] = {layer: [] for layer in layer_indices}
    handles = [
        _decoder_layers(model)[layer].register_forward_hook(_norm_recording_hook(norm_store, layer))
        for layer in layer_indices
    ]
    try:
        baseline_gaps = _forced_choice_gaps(
            model, input_ids, attention_mask, token_pairs, batch_size=args.batch_size
        )
    finally:
        for handle in handles:
            handle.remove()
    residual_norms = {layer: _mean(values) for layer, values in norm_store.items()}

    direction_names = list(directions)
    cells: list[dict[str, Any]] = []
    for cell in sweep_grid(layers_by_direction, multipliers):
        real = directions[cell.direction][cell.layer]
        placebo = placebo_for(
            real,
            seed=args.seed,
            direction_index=direction_names.index(cell.direction),
            layer=cell.layer,
        )
        alpha_raw = cell.alpha_multiplier * float(real.norm())
        hook = hook_for(cell.condition, real, placebo, alpha_raw)
        with residual_intervention(model, cell.layer, hook):
            gaps = _forced_choice_gaps(
                model, input_ids, attention_mask, token_pairs, batch_size=args.batch_size
            )
        entry: dict[str, Any] = {
            "direction": cell.direction,
            "layer": cell.layer,
            "alpha_multiplier": cell.alpha_multiplier,
            "condition": cell.condition,
            "alpha_raw": alpha_raw,
            "alpha_over_residual_norm": alpha_raw / residual_norms[cell.layer],
        }
        entry.update(_gap_summary(gaps, baseline_gaps, orders))
        cells.append(entry)
        logger.info(
            f"sweep cell done: {cell.direction} L{cell.layer} x{cell.alpha_multiplier} "
            f"{cell.condition} delta={entry['mean_delta_vs_none']:+.4f}"
        )

    payload: dict[str, Any] = {
        "command": "logit-sweep",
        "model": args.model,
        "deltanet_kernel_bridge": kernel_bridge,
        DELTANET_KERNEL_FIELD: bound_deltanet_kernels(),
        "render_grading": RENDER_GRADING,
        "splits": splits,
        "seed": args.seed,
        "caveat": "thinking-off forced choice selects the intervention; it does not establish "
        "the behavioural claim.",
        "denominators": {
            "n_total": len(rows),
            "n_used": len(usable_rows),
            "n_skipped": len(skipped),
            "skipped_prompt_ids": skipped,
        },
        "residual_norms_by_layer": {str(k): v for k, v in residual_norms.items()},
        "baseline": _gap_summary(baseline_gaps, None, orders),
        "cells": cells,
    }
    if args.per_row:
        payload["baseline_per_row_gaps"] = baseline_gaps
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info(f"sweep written, out={args.out} cells={len(cells)}")
    return payload


# --------------------------------------------------------------------------------------
# Selecting what the generation leg runs at
# --------------------------------------------------------------------------------------


def _sweep_scores(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per (direction, layer, multiplier): steer and placebo scores, and the margin between them."""
    grouped: dict[tuple[str, int, float], dict[str, float]] = {}
    for cell in cast("list[dict[str, Any]]", payload["cells"]):
        key = (str(cell["direction"]), int(cell["layer"]), float(cell["alpha_multiplier"]))
        grouped.setdefault(key, {})[str(cell["condition"])] = float(cell["mean_delta_vs_none"])
    scores: list[dict[str, Any]] = []
    for (direction, layer, multiplier), deltas in sorted(grouped.items()):
        missing = sorted(set(STEERING_CONDITIONS) - set(deltas))
        if missing:
            raise ValueError(
                f"sweep is missing conditions {missing} for {direction} L{layer} x{multiplier}; "
                f"a selection over an incomplete grid would silently prefer the complete cells."
            )
        steer = (abs(deltas[CONDITION_STEER_UP]) + abs(deltas[CONDITION_STEER_DOWN])) / 2
        placebo = (abs(deltas[CONDITION_PLACEBO_UP]) + abs(deltas[CONDITION_PLACEBO_DOWN])) / 2
        scores.append(
            {
                "direction": direction,
                "layer": layer,
                "alpha_multiplier": multiplier,
                "steer_score": steer,
                "placebo_score": placebo,
                "margin": steer - placebo,
            }
        )
    if not scores:
        raise ValueError("the sweep payload holds no cells; nothing to select from.")
    return scores


def select_intervention(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically pick the (direction, layer, alpha) the generation leg should run at.

    The winner is the grid point whose steering effect most exceeds its own placebo. When no point
    beats its placebo the sweep found nothing selective; the largest steering effect is returned
    anyway so generation still has a target, flagged ``fallback`` so the readout says the selection
    carried no evidence.
    """
    scores = _sweep_scores(payload)
    ranked = sorted(
        scores,
        key=lambda s: (-s["margin"], -s["steer_score"], s["direction"], s["layer"]),
    )
    best = ranked[0]
    if best["margin"] > 0:
        return {**best, "fallback": False, "reason": "largest steer-minus-placebo margin"}
    strongest = max(scores, key=lambda s: (s["steer_score"], -s["layer"]))
    return {
        **strongest,
        "fallback": True,
        "reason": "no grid point beat its matched-norm placebo; taking the largest raw steering "
        "effect so the generation leg still runs, but the selection carries no evidence.",
    }


def plan_generation_cells(payload: Mapping[str, Any], *, n_cells: int) -> list[dict[str, Any]]:
    """Winner, winner at the next multiplier up, and its best neighbouring layer -- up to n_cells."""
    scores = _sweep_scores(payload)
    winner = select_intervention(payload)
    plan: list[dict[str, Any]] = [
        {k: winner[k] for k in ("direction", "layer", "alpha_multiplier")}
    ]

    same_point = [
        s
        for s in scores
        if s["direction"] == winner["direction"]
        and s["layer"] == winner["layer"]
        and s["alpha_multiplier"] > winner["alpha_multiplier"]
    ]
    if same_point:
        larger = min(same_point, key=lambda s: s["alpha_multiplier"])
        plan.append({k: larger[k] for k in ("direction", "layer", "alpha_multiplier")})

    neighbours = [
        s
        for s in scores
        if s["direction"] == winner["direction"]
        and abs(s["layer"] - winner["layer"]) == 1
        and s["alpha_multiplier"] == winner["alpha_multiplier"]
    ]
    if neighbours:
        best_neighbour = max(neighbours, key=lambda s: s["margin"])
        plan.append({k: best_neighbour[k] for k in ("direction", "layer", "alpha_multiplier")})

    unique: list[dict[str, Any]] = []
    for cell in plan:
        if cell not in unique:
            unique.append(cell)
    return unique[:n_cells]


# --------------------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationCondition:
    """One generation condition: a hook spec plus everything the records must carry."""

    condition: str
    direction: str | None
    layer: int | None
    alpha_multiplier: float | None


def generation_conditions(cells: Sequence[Mapping[str, Any]]) -> list[GenerationCondition]:
    """Build the deduplicated condition list.

    One shared baseline, four steering conditions per cell, and two ablation conditions per
    (direction, layer).
    """
    conditions: list[GenerationCondition] = [GenerationCondition(CONDITION_NONE, None, None, None)]
    seen_ablation: set[tuple[str, int]] = set()
    for cell in cells:
        direction = str(cell["direction"])
        layer = int(cell["layer"])
        multiplier = float(cell["alpha_multiplier"])
        conditions.extend(
            GenerationCondition(condition, direction, layer, multiplier)
            for condition in STEERING_CONDITIONS
        )
        if (direction, layer) not in seen_ablation:
            seen_ablation.add((direction, layer))
            conditions.extend(
                GenerationCondition(condition, direction, layer, None)
                for condition in ABLATION_CONDITIONS
            )
    unique: list[GenerationCondition] = []
    for condition in conditions:
        if condition not in unique:
            unique.append(condition)
    return unique


def selected_target_conditions(cell: Mapping[str, Any]) -> list[GenerationCondition]:
    """Build the three-condition causal check for one calibration-selected target."""
    direction = str(cell["direction"])
    layer = int(cell["layer"])
    multiplier = float(cell["alpha_multiplier"])
    return [
        GenerationCondition(CONDITION_NONE, None, None, None),
        GenerationCondition(CONDITION_STEER_UP, direction, layer, multiplier),
        GenerationCondition(CONDITION_PLACEBO_UP, direction, layer, multiplier),
    ]


def _parse_cells_flag(specs: Sequence[str]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for spec in specs:
        try:
            direction, layer, multiplier = spec.split(":")
        except ValueError as error:
            raise ValueError(f"--cells expects NAME:LAYER:MULTIPLIER, got {spec!r}.") from error
        cells.append(
            {"direction": direction, "layer": int(layer), "alpha_multiplier": float(multiplier)}
        )
    return cells


def _past_deadline(deadline: dt.datetime | None) -> bool:
    return deadline is not None and dt.datetime.now(dt.UTC) >= deadline


def _skip_reason(
    index: int, requested: set[int] | None, deadline: dt.datetime | None
) -> str | None:
    """Why the condition at ``index`` should not run, or None to proceed.

    'not requested' outranks 'deadline' so a filtered-out condition is never mislabelled as
    deadline-cut in the summary — the two reasons mean different things to a reader deciding
    whether a cell is complete.
    """
    if requested is not None and index not in requested:
        return "not requested"
    if _past_deadline(deadline):
        return "deadline"
    return None


def _allocation_outcome_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise allocation payoffs while retaining unresolved completion counts."""
    allocation_records = [
        record
        for record in records
        if record.get("row_kind", CUSTOM_ROW_KIND_MATRIX) == CUSTOM_ROW_KIND_BINARY_ALLOCATION
    ]
    resolved = [
        record for record in allocation_records if record.get("selected_own_payoff") is not None
    ]

    def mean(field: str) -> float | None:
        return (
            None
            if not resolved
            else sum(float(record[field]) for record in resolved) / len(resolved)
        )

    return {
        "n_allocation_completions": len(allocation_records),
        "n_allocation_payoff_resolved": len(resolved),
        "n_allocation_payoff_unresolved": len(allocation_records) - len(resolved),
        "selected_own_payoff_mean": mean("selected_own_payoff"),
        "selected_counterpart_payoff_mean": mean("selected_counterpart_payoff"),
        "selected_total_welfare_mean": mean("selected_total_welfare"),
    }


def _reward_outcome_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise parsed reward-optimal choices without dropping unresolved completions."""
    resolved = [record for record in records if record.get("reward_optimal") is not None]
    matches = [record for record in resolved if record.get("reward_optimal_match") is not None]
    matched = sum(1 for record in matches if bool(record["reward_optimal_match"]))
    own_resolved = [record for record in records if record.get("own_payoff_optimal") is not None]
    own_matches = [
        record for record in own_resolved if record.get("own_payoff_optimal_match") is not None
    ]
    total_resolved = [
        record for record in records if record.get("total_welfare_optimal") is not None
    ]
    total_matches = [
        record for record in total_resolved if record.get("total_welfare_optimal_match") is not None
    ]
    return {
        "n_reward_optimal_resolved": len(matches),
        "n_reward_optimal_unresolved": len(records) - len(matches),
        "reward_optimal_match_k": matched,
        "reward_optimal_match_rate": None if not matches else matched / len(matches),
        "n_own_payoff_optimal_resolved": len(own_matches),
        "n_own_payoff_optimal_unresolved": len(records) - len(own_matches),
        "own_payoff_optimal_match_k": sum(
            1 for record in own_matches if bool(record["own_payoff_optimal_match"])
        ),
        "own_payoff_optimal_match_rate": (
            None
            if not own_matches
            else sum(1 for record in own_matches if bool(record["own_payoff_optimal_match"]))
            / len(own_matches)
        ),
        "n_total_welfare_optimal_resolved": len(total_matches),
        "n_total_welfare_optimal_unresolved": len(records) - len(total_matches),
        "total_welfare_optimal_match_k": sum(
            1 for record in total_matches if bool(record["total_welfare_optimal_match"])
        ),
        "total_welfare_optimal_match_rate": (
            None
            if not total_matches
            else sum(1 for record in total_matches if bool(record["total_welfare_optimal_match"]))
            / len(total_matches)
        ),
        "n_payoff_sensitive": sum(
            1 for record in records if float(record.get("payoff_sensitive", 0.0)) > 0.0
        ),
    }


def _summary_for_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute one denominator-preserving summary for any record subset."""
    parsed = [record for record in records if record["cooperate"] is not None]
    cooperated = sum(1 for record in parsed if bool(record["cooperate"]))
    unresolved = len(records) - len(parsed)
    summary: dict[str, Any] = {
        "n_completions": len(records),
        "n_parsed": len(parsed),
        "n_parse_failures": unresolved,
        "n_truncated_thinking": sum(1 for record in records if bool(record["truncated_thinking"])),
        "cooperate_k": cooperated,
        "cooperate_rate": None if not parsed else cooperated / len(parsed),
        "cooperate_rate_lower_bound": None if not records else cooperated / len(records),
        "cooperate_rate_upper_bound": (
            None if not records else (cooperated + unresolved) / len(records)
        ),
    }
    summary.update(_allocation_outcome_summary(records))
    summary.update(_reward_outcome_summary(records))
    first_position = [
        record for record in records if record.get("first_position_choice") is not None
    ]
    first_position_k = sum(1 for record in first_position if bool(record["first_position_choice"]))
    summary.update(
        {
            "n_first_position_resolved": len(first_position),
            "n_first_position_unresolved": len(records) - len(first_position),
            "first_position_choice_k": first_position_k,
            "first_position_choice_rate": (
                None if not first_position else first_position_k / len(first_position)
            ),
        }
    )
    return summary


def _summary_by_print_order(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the same summary, including censoring bounds, for each printed label order."""
    by_order: dict[str, dict[str, Any]] = {}
    for order in sorted({str(record["label_print_order"]) for record in records}):
        order_records = [record for record in records if str(record["label_print_order"]) == order]
        by_order[order] = _summary_for_records(order_records)
    return by_order


def summarise_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise cooperation and row-specific outcomes by condition, family and print order."""
    assert_one_deltanet_kernel(
        (record.get(DELTANET_KERNEL_FIELD) for record in records),
        what="this steering summary",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["condition_key"]), []).append(record)
    summary: dict[str, Any] = {}
    for key, group in sorted(grouped.items()):
        entry = _summary_for_records(group)
        entry["by_print_order"] = _summary_by_print_order(group)
        families = sorted(
            {
                str(record["diagnostic_family"])
                for record in group
                if record.get("diagnostic_family") is not None
            }
        )
        if families:
            entry["by_diagnostic_family"] = {}
            for family in families:
                family_records = [
                    record for record in group if str(record.get("diagnostic_family")) == family
                ]
                family_summary = _summary_for_records(family_records)
                family_summary["by_print_order"] = _summary_by_print_order(family_records)
                entry["by_diagnostic_family"][family] = family_summary
        summary[key] = entry
    return summary


def condition_key(condition: GenerationCondition) -> str:
    """Render a stable flat key for grouping: e.g. ``decision:L18:x1.0:steer:+`` or ``none``."""
    if condition.condition == CONDITION_NONE:
        return CONDITION_NONE
    multiplier = "" if condition.alpha_multiplier is None else f":x{condition.alpha_multiplier}"
    return f"{condition.direction}:L{condition.layer}{multiplier}:{condition.condition}"


def parse_requested_conditions(
    raw: str | None, conditions: Sequence[GenerationCondition]
) -> set[int] | None:
    """Resolve ``--conditions`` into indices of the FULL condition list, refusing a miss.

    The gap-fill filter for a run whose early conditions are already banked (added 2026-08-25,
    after three spot reclaims each re-paid a ~7-hour condition prefix this module cannot resume).
    Filtering selects which conditions RUN but never renumbers them: every executed condition
    keeps the per-condition RNG seed (``args.seed + index``) an unfiltered run would have drawn at
    that index, which is what makes a filtered run a continuation of the same measurement rather
    than a new experiment. Returning indices rather than a rebuilt list is what enforces that.

    An entry may be a bare condition name (``placebo:+``) or a full key
    (``decision:L18:x1.0:placebo:+``); with multiple cells a bare name matches every cell that
    carries it. A name matching nothing is an operator mistake and refuses before any decode.
    """
    if raw is None:
        return None
    wanted = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not wanted:
        raise ValueError("--conditions was passed but names nothing.")
    indices: set[int] = set()
    for want in wanted:
        matched = [
            index
            for index, condition in enumerate(conditions)
            if condition_key(condition) == want or condition.condition == want
        ]
        if not matched:
            keys = [condition_key(condition) for condition in conditions]
            raise ValueError(f"--conditions names {want!r}, which matches nothing in {keys}.")
        indices.update(matched)
    return indices


def directions_digest(directions: Mapping[str, Mapping[int, torch.Tensor]]) -> dict[str, str]:
    """One sha256 per named direction file over its (layer, float32 vector) pairs in layer order.

    Part of a generation run's resume identity: a condition key names a direction by NAME, so two
    runs pointing the same name at different fitted vectors would otherwise resume into one file.
    """
    digests: dict[str, str] = {}
    for name, by_layer in directions.items():
        digest = hashlib.sha256()
        for layer in sorted(by_layer):
            digest.update(str(layer).encode())
            digest.update(b"\x00")
            digest.update(by_layer[layer].float().contiguous().numpy().tobytes())
        digests[name] = digest.hexdigest()
    return digests


def generation_identity(  # noqa: PLR0913 - an identity is every argument a record depends on
    args: argparse.Namespace,
    conditions: Sequence[GenerationCondition],
    directions: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    rows: Sequence[Mapping[str, Any]],
    prompts: Sequence[str],
    sample_indices: Sequence[int],
    placebo_seed: int,
    selected_target: Mapping[str, Any] | None,
    adapter: Mapping[str, Any] | None,
    sampler_payload: Mapping[str, Any],
    weights_identity: str,
    deltanet_kernel: Mapping[str, str],
) -> dict[str, Any]:
    """Everything a generation record depends on, for the resume ledger to compare against.

    Deliberately excludes ``--conditions`` and ``--deadline``: those select which conditions RUN,
    never what a record contains, so a gap-fill or a deadline-cut run is the same measurement.
    The rendered prompts enter as a digest rather than a count: the rows come off the prompt
    renderer at whatever commit the relaunch runs, and a renderer that moved between two commits
    would otherwise resume the same number of different prompts into one file.

    The base model enters twice, as the name the flags gave and as what that name resolves to
    (a hub revision or a local digest, :func:`resolve_weights_identity`): a hub id whose revision
    advanced, or a local directory rewritten in place, is the same string over different weights.
    The direction names enter as an ordered list and not only through the name-keyed digests,
    because each placebo is seeded from its direction's position in the ``--direction`` flags
    (:func:`placebo_for`), so the same flags in another order draw different random vectors and
    must not resume into one file.

    The Gated DeltaNet kernel bindings enter for the same reason the weights do: the fused decode
    kernel and the torch fallback are the same recurrence in a different reduction order, and greedy
    tokens diverge under them, so a relaunch across the fla bridge is a different draw and is refused
    here rather than appended. A records file banked before this field existed therefore refuses a
    relaunch, which is correct: continuing it now would decode its remaining conditions under the
    bridged kernel and file them beside conditions decoded without it.
    """
    return {
        "command": "generate",
        "model": args.model,
        "model_weights_identity": weights_identity,
        DELTANET_KERNEL_FIELD: dict(deltanet_kernel),
        "seed": args.seed,
        "placebo_seed": placebo_seed,
        "n_prompts": len(prompts),
        "prompts_sha256": digest_of_strings(prompts),
        "rendered_rows_sha256": rendered_rows_digest(rows),
        "rows_sha256": rendered_rows_digest(rows),
        "sample_indices": list(sample_indices),
        "n_samples": args.n_samples,
        "selected_target": None if selected_target is None else dict(selected_target),
        "adapter": None if adapter is None else dict(adapter),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "counterpart_framing": args.counterpart_framing,
        "diagnostic_profile": args.diagnostic_profile,
        "row_manifest": None if args.row_manifest is None else str(args.row_manifest),
        "condition_keys": [condition_key(condition) for condition in conditions],
        "direction_names": list(directions),
        "directions_sha256": directions_digest(directions),
        "resolved_sampler": dict(sampler_payload),
    }


def condition_intervention(
    condition: GenerationCondition,
    directions: Mapping[str, Mapping[int, torch.Tensor]],
    direction_names: Sequence[str],
    *,
    model: AutoModelForCausalLM,
    placebo_seed: int,
) -> tuple[AbstractContextManager[None], float | None]:
    """Build one condition's hook context and its raw alpha; the baseline gets a no-op context.

    The placebo is seeded from the recorded independent placebo seed, the direction's index and the
    layer -- never from the condition's position in the list -- so every placebo condition of a
    (direction, layer) pushes the same random vector and a gap-filled or resumed run draws the
    vector an unbroken one would have.
    """
    if condition.condition == CONDITION_NONE:
        return nullcontext(), None
    direction_name = cast("str", condition.direction)
    layer = cast("int", condition.layer)
    real = directions[direction_name][layer]
    placebo = placebo_for(
        real,
        seed=placebo_seed,
        direction_index=direction_names.index(direction_name),
        layer=layer,
    )
    alpha_raw = (
        None
        if condition.alpha_multiplier is None
        else condition.alpha_multiplier * float(real.norm())
    )
    hook = hook_for(condition.condition, real, placebo, alpha_raw or 0.0)
    return residual_intervention(model, layer, hook), alpha_raw


def selected_allocation_outcome(row: Mapping[str, Any], action: str | None) -> dict[str, Any]:
    """Return own, counterpart and total welfare for a parsed binary allocation choice."""
    empty = {
        "selected_allocation_label": None,
        "selected_own_payoff": None,
        "selected_counterpart_payoff": None,
        "selected_total_welfare": None,
    }
    if row.get("row_kind", CUSTOM_ROW_KIND_MATRIX) != CUSTOM_ROW_KIND_BINARY_ALLOCATION:
        return empty
    if action is None:
        return empty
    selected_label = parsed_action_label(row, action)
    if selected_label is None:
        return empty
    if selected_label == row["label_a"]:
        own_payoff = float(row["allocation_action_a_own_payoff"])
        counterpart_payoff = float(row["allocation_action_a_counterpart_payoff"])
    else:
        own_payoff = float(row["allocation_action_b_own_payoff"])
        counterpart_payoff = float(row["allocation_action_b_counterpart_payoff"])
    return {
        "selected_allocation_label": selected_label,
        "selected_own_payoff": own_payoff,
        "selected_counterpart_payoff": counterpart_payoff,
        "selected_total_welfare": own_payoff + counterpart_payoff,
    }


def parsed_action_label(row: Mapping[str, Any], action: str | None) -> str | None:
    """Translate the parser's cooperation/defection result back to the rendered label."""
    if action is None:
        return None
    if action == COOPERATE:
        return str(row["coop_label"])
    return next(
        label
        for label in (str(row["label_a"]), str(row["label_b"]))
        if label != str(row["coop_label"])
    )


def reward_optimal_outcome(row: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the row's reward-optimal action from its numerical payoffs."""
    labels = (str(row["label_a"]), str(row["label_b"]))
    coop_label = str(row["coop_label"])
    other_label = next(label for label in labels if label != coop_label)
    if row.get("row_kind", CUSTOM_ROW_KIND_MATRIX) == CUSTOM_ROW_KIND_BINARY_ALLOCATION:
        own_payoffs = {
            labels[0]: float(row["allocation_action_a_own_payoff"]),
            labels[1]: float(row["allocation_action_b_own_payoff"]),
        }
        total_payoffs = {
            labels[0]: own_payoffs[labels[0]]
            + float(row["allocation_action_a_counterpart_payoff"]),
            labels[1]: own_payoffs[labels[1]]
            + float(row["allocation_action_b_counterpart_payoff"]),
        }

        def unique_maximum(values: Mapping[str, float]) -> tuple[str | None, float | None]:
            highest = max(values.values())
            winners = [label for label, value in values.items() if value == highest]
            return (None, None) if len(winners) != 1 else (winners[0], highest)

        own_optimal_label, own_optimal_payoff = unique_maximum(own_payoffs)
        total_optimal_label, total_optimal_payoff = unique_maximum(total_payoffs)
        return {
            "reward_optimal": None,
            "reward_optimal_payoff": None,
            "own_payoff_optimal": own_optimal_label,
            "own_payoff_optimal_value": own_optimal_payoff,
            "total_welfare_optimal": total_optimal_label,
            "total_welfare_optimal_value": total_optimal_payoff,
            "payoff_sensitive": float(own_payoffs[labels[0]] != own_payoffs[labels[1]]),
        }
    probability = float(row["opp_coop_prob"])
    expected_values_known = 0.0 <= probability <= 1.0
    if 0.0 <= probability <= 1.0:
        expected_coop = probability * float(row["payoff_cc"]) + (1.0 - probability) * float(
            row["payoff_cd"]
        )
        expected_other = probability * float(row["payoff_dc"]) + (1.0 - probability) * float(
            row["payoff_dd"]
        )
    else:
        coop_payoffs = (float(row["payoff_cc"]), float(row["payoff_cd"]))
        other_payoffs = (float(row["payoff_dc"]), float(row["payoff_dd"]))
        if all(other > coop for other, coop in zip(other_payoffs, coop_payoffs, strict=True)):
            expected_coop, expected_other = 0.0, 1.0
        elif all(coop > other for other, coop in zip(other_payoffs, coop_payoffs, strict=True)):
            expected_coop, expected_other = 1.0, 0.0
        else:
            expected_coop = expected_other = 0.0
    difference = expected_coop - expected_other
    if math.isclose(difference, 0.0, abs_tol=1e-12):
        optimal_label = None
        optimal_payoff = None
    elif difference > 0:
        optimal_label = coop_label
        optimal_payoff = expected_coop if expected_values_known else None
    else:
        optimal_label = other_label
        optimal_payoff = expected_other if expected_values_known else None
    return {
        "reward_optimal": optimal_label,
        "reward_optimal_payoff": optimal_payoff,
        "own_payoff_optimal": optimal_label,
        "own_payoff_optimal_value": optimal_payoff,
        "total_welfare_optimal": None,
        "total_welfare_optimal_value": None,
        "payoff_sensitive": float(
            any(
                float(row[action]) != float(row[other_action])
                for action, other_action in (
                    ("payoff_cc", "payoff_dc"),
                    ("payoff_cd", "payoff_dd"),
                )
            )
        ),
    }


def resolve_placebo_seed(args: argparse.Namespace) -> int:
    """Return the independent placebo stream seed recorded by a generation run."""
    if args.placebo_seed is not None:
        return int(args.placebo_seed)
    return int(args.seed) + DEFAULT_PLACEBO_SEED_OFFSET


def _build_generation_backend(
    args: argparse.Namespace, sampling: SamplingConfig
) -> tuple[HFBackend, AutoModelForCausalLM, dict[str, Any] | None]:
    """Load the base or attach the requested un-merged adapter before any generation."""
    if args.adapter is None:
        backend = HFBackend(args.model, thinking=True, sampling=sampling)
        return backend, cast("AutoModelForCausalLM", backend.model), None
    adapter_dir = Path(args.adapter).resolve()
    use_bfloat16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    base_model = load_adapter_base(
        args.model,
        dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    attached = attach_adapter(base_model, adapter_dir, args.model)
    backend = HFBackend(
        args.model,
        thinking=True,
        sampling=sampling,
        model=attached.peft_model,
    )
    assert_adapter_changes_forward(
        cast("AdapterCapableModel", backend.model),
        backend.tokenizer,
        ADAPTER_PROBE_PROMPTS,
    )
    return (
        backend,
        cast("AutoModelForCausalLM", backend.model),
        adapter_identity(adapter_dir, applied_adapter_weights=attached.applied_adapter_weights),
    )


def run_generate(  # noqa: C901, PLR0912, PLR0915 - one linear driver
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generate the steering grid on held-out eval prompts and write records plus a summary.

    Resumes automatically: a records file already in ``--out-dir`` is read through its resume
    ledger, every complete condition is carried forward and skipped, and a partial one (a reclaim
    mid-condition) is dropped and regenerated. Each condition reseeds from its own index, so the
    regenerated records are the ones an uninterrupted run would have written; the summary counts
    ``resumed_conditions`` apart from ``skipped_conditions`` so what actually ran stays visible.
    """
    direction_specs = parse_named_specs(args.direction, flag="--direction")
    started_monotonic = time.monotonic()
    directions = load_named_directions(args.direction)
    direction_paths = {name: Path(path) for name, path in direction_specs.items()}
    selected_target: dict[str, Any] | None = None
    requested_sources = sum(
        source is not None for source in (args.cells, args.from_sweep, args.selected_target)
    )
    if requested_sources != 1:
        raise ValueError("pass exactly one of --cells, --from-sweep, or --selected-target.")
    if args.selected_target is not None:
        selected_target = load_selected_target(
            args.selected_target, directions, direction_paths=direction_paths
        )
        cells = [cast("dict[str, Any]", selected_target["cell"])]
        selection = selected_target
    elif args.from_sweep is not None:
        sweep_payload = cast("dict[str, Any]", json.loads(Path(args.from_sweep).read_text()))
        cells = plan_generation_cells(sweep_payload, n_cells=args.n_cells)
        selection = select_intervention(sweep_payload)
    else:
        cells = _parse_cells_flag(args.cells)
        selection = None
    for cell in cells:
        if str(cell["direction"]) not in directions:
            raise ValueError(f"cell {cell} names a direction with no --direction flag.")
        if int(cell["layer"]) not in directions[str(cell["direction"])]:
            raise ValueError(f"cell {cell} names a layer missing from its direction file.")

    placebo_seed = resolve_placebo_seed(args)
    if args.row_manifest is None and args.diagnostic_profile is None:
        rows = generation_rows(args.counterpart_framing)
    else:
        rows = generation_rows(args.counterpart_framing, args.row_manifest, args.diagnostic_profile)
    prompts = [str(row["prompt"]) for row in rows for _ in range(args.n_samples)]
    row_for_prompt = [row for row in rows for _ in range(args.n_samples)]
    sample_indices = [index for _ in rows for index in range(args.n_samples)]

    conditions = (
        selected_target_conditions(cells[0])
        if selected_target is not None
        else generation_conditions(cells)
    )
    requested = parse_requested_conditions(args.conditions, conditions)
    direction_names = list(directions)

    deadline = None if args.deadline is None else dt.datetime.fromisoformat(args.deadline)
    sampling = replace(
        eval_sampling(SAMPLER_TRAINING_DISTRIBUTION, thinking=True),
        max_new_tokens=args.max_new_tokens,
    )
    # Before the load, which imports the modeling module and freezes each kernel's dispatch.
    kernel_bridge = bridge_and_check_decode_kernel()
    backend, model, adapter = _build_generation_backend(args, sampling)
    deltanet_kernel = bound_deltanet_kernels()
    # After the load, so a hub id resolves from the same cache the weights just came out of.
    weights_identity = resolve_weights_identity(args.model)

    sampler_payload = resolved_sampler(sampling).as_payload()
    skipped_conditions: list[dict[str, Any]] = []
    resumed_conditions: list[dict[str, Any]] = []
    condition_seconds: dict[str, float] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / GENERATION_RECORDS_FILENAME
    resume_state, ledger = resume_records(
        records_path,
        identity=generation_identity(
            args,
            conditions,
            directions,
            rows=rows,
            prompts=prompts,
            sample_indices=sample_indices,
            placebo_seed=placebo_seed,
            selected_target=selected_target,
            adapter=adapter,
            sampler_payload=sampler_payload,
            weights_identity=weights_identity,
            deltanet_kernel=deltanet_kernel,
        ),
        unit_of=lambda record: str(record["condition_key"]),
    )
    records: list[dict[str, Any]] = list(resume_state.kept_records)

    with records_path.open("a", encoding="utf-8") as records_file:
        for index, condition in enumerate(conditions):
            key = condition_key(condition)
            if key in resume_state.complete_units:
                n_banked = resume_state.complete_units[key]
                resumed_conditions.append({"condition_key": key, "n_records": n_banked})
                logger.info(f"resuming condition {key}: {n_banked} records already banked")
                continue
            reason = _skip_reason(index, requested, deadline)
            if reason is not None:
                skipped_conditions.append({"condition_key": key, "skipped": reason})
                logger.warning(f"skipping condition {key}: {reason}")
                continue
            seed = args.seed + index
            condition_started = time.monotonic()
            torch.manual_seed(seed)
            context, alpha_raw = condition_intervention(
                condition, directions, direction_names, model=model, placebo_seed=placebo_seed
            )
            with context:
                completions = decode_in_chunks(backend, prompts, chunk_size=args.batch_size)
            for row, sample_index, completion in zip(
                row_for_prompt, sample_indices, completions, strict=True
            ):
                visible, truncated = strip_thinking(completion, prefilled_think=True)
                action = parse_action(
                    visible,
                    label_a=str(row["label_a"]),
                    label_b=str(row["label_b"]),
                    coop_label=str(row["coop_label"]),
                )
                allocation_outcome = selected_allocation_outcome(row, action)
                reward_outcome = reward_optimal_outcome(row)
                parsed_label = parsed_action_label(row, action)
                first_label = (
                    str(row["label_a"])
                    if row["label_print_order"] == "canonical"
                    else str(row["label_b"])
                )
                record: dict[str, Any] = {
                    "condition_key": key,
                    DELTANET_KERNEL_FIELD: deltanet_kernel,
                    "condition": condition.condition,
                    "direction": condition.direction,
                    "layer": condition.layer,
                    "alpha_multiplier": condition.alpha_multiplier,
                    "alpha_raw": alpha_raw,
                    "seed": seed,
                    "placebo_seed": placebo_seed,
                    "adapter": adapter,
                    "counterpart_framing": args.counterpart_framing,
                    "prompt_id": row["prompt_id"],
                    "diagnostic_family": row.get("diagnostic_family"),
                    "row_kind": row.get("row_kind", CUSTOM_ROW_KIND_MATRIX),
                    "game_id": row["game_id"],
                    "grading": row["grading"],
                    "payoff_cc": row["payoff_cc"],
                    "payoff_cd": row["payoff_cd"],
                    "payoff_dc": row["payoff_dc"],
                    "payoff_dd": row["payoff_dd"],
                    "reskin_id": row["reskin_id"],
                    "payoff_variant": row["payoff_variant"],
                    "label_print_order": row["label_print_order"],
                    "coop_label": row["coop_label"],
                    "sample_index": sample_index,
                    "response_text": completion,
                    "parsed_action": action,
                    "first_position_choice": (
                        None if parsed_label is None else parsed_label == first_label
                    ),
                    "cooperate": None if action is None else action == COOPERATE,
                    "truncated_thinking": truncated,
                    **reward_outcome,
                    "reward_optimal_match": (
                        None
                        if parsed_label is None or reward_outcome["reward_optimal"] is None
                        else parsed_label == reward_outcome["reward_optimal"]
                    ),
                    "own_payoff_optimal_match": (
                        None
                        if parsed_label is None or reward_outcome["own_payoff_optimal"] is None
                        else parsed_label == reward_outcome["own_payoff_optimal"]
                    ),
                    "total_welfare_optimal_match": (
                        None
                        if parsed_label is None or reward_outcome["total_welfare_optimal"] is None
                        else parsed_label == reward_outcome["total_welfare_optimal"]
                    ),
                    **allocation_outcome,
                }
                records.append(record)
                records_file.write(json.dumps(record) + "\n")
            records_file.flush()
            ledger.mark_complete(key, len(completions))
            condition_seconds[key] = time.monotonic() - condition_started
            logger.info(f"generation condition done: {key} ({len(completions)} completions)")

    summary: dict[str, Any] = {
        "command": "generate",
        "model": args.model,
        "model_weights_identity": weights_identity,
        "deltanet_kernel_bridge": kernel_bridge,
        DELTANET_KERNEL_FIELD: deltanet_kernel,
        "render_grading": RENDER_GRADING,
        "split": SPLIT_EVAL,
        "counterpart_framing": args.counterpart_framing,
        "row_manifest": None if args.row_manifest is None else str(args.row_manifest),
        "diagnostic_profile": args.diagnostic_profile,
        "rendered_rows_sha256": rendered_rows_digest(rows),
        "rows_sha256": rendered_rows_digest(rows),
        "n_rows": len(rows),
        "n_samples": args.n_samples,
        "seed": args.seed,
        "placebo_seed": placebo_seed,
        "max_new_tokens": args.max_new_tokens,
        "resolved_sampler": sampler_payload,
        "cells": cells,
        "selection": selection,
        "selected_target": selected_target,
        "adapter": adapter,
        "requested_conditions": None
        if requested is None
        else sorted(condition_key(conditions[index]) for index in requested),
        "conditions": summarise_records(records),
        "skipped_conditions": skipped_conditions,
        "resumed_conditions": resumed_conditions,
        "n_records_resumed": resume_state.n_kept,
        "n_records_dropped_partial": resume_state.dropped_records,
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "condition_seconds": condition_seconds,
    }
    summary_path = args.out_dir / GENERATION_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info(
        f"steering generation written, records={records_path} summary={summary_path} "
        f"n_records={len(records)} resumed={len(resumed_conditions)} "
        f"skipped={len(skipped_conditions)}"
    )
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI: fit-direction (CPU), logit-sweep (GPU, minutes), generate (the pricey leg)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit-direction", help="Fit per-layer diff-of-means from one cell.")
    fit.add_argument("--capture-root", type=Path, required=True)
    fit.add_argument("--stimuli", type=Path, required=True)
    fit.add_argument("--arm", default="base")
    fit.add_argument("--step", type=int, default=0)
    fit.add_argument("--set", required=True, help="Stimulus set name.")
    fit.add_argument("--pooling", default="last")
    fit.add_argument("--positive-side", default="A")
    fit.add_argument("--pairs", choices=PAIRS_CHOICES, default="even")
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--compare", type=Path, default=None)

    sweep = sub.add_parser("logit-sweep", help="Forced-choice logit readout over a steering grid.")
    sweep.add_argument("--model", required=True)
    sweep.add_argument("--direction", action="append", default=[], metavar="NAME=PATH")
    sweep.add_argument("--layers", action="append", default=[], metavar="NAME=16,17,18")
    sweep.add_argument(
        "--alpha-multipliers", default=",".join(str(m) for m in DEFAULT_ALPHA_MULTIPLIERS)
    )
    sweep.add_argument("--splits", default=f"{SPLIT_EVAL},{SPLIT_TRAIN}")
    sweep.add_argument("--batch-size", type=int, default=32)
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--per-row", action="store_true")
    sweep.add_argument("--out", type=Path, required=True)

    generate = sub.add_parser("generate", help="Steered generation on held-out eval prompts.")
    generate.add_argument("--model", required=True)
    generate.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Un-merged PEFT adapter trained against --model; it is attached and positive-control "
        "checked before generation.",
    )
    generate.add_argument("--direction", action="append", default=[], metavar="NAME=PATH")
    generate.add_argument("--cells", action="append", default=None, metavar="NAME:LAYER:MULT")
    generate.add_argument("--from-sweep", default=None)
    generate.add_argument(
        "--selected-target",
        type=Path,
        default=None,
        help="Calibration-only selected-target.json; mutually exclusive with --cells and "
        "--from-sweep.",
    )
    generate.add_argument("--n-cells", type=int, default=3)
    generate.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    generate.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    generate.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument(
        "--placebo-seed",
        type=int,
        default=None,
        help="Independent matched-norm placebo stream seed; defaults to a separate seed derived "
        "from --seed and is recorded in the resume identity.",
    )
    generate.add_argument(
        "--deadline", default=None, help="ISO-8601 UTC instant after which no new condition starts."
    )
    generate.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated condition names or full keys to run; the rest are recorded as "
        "'not requested'. Filtering never renumbers seeds — see parse_requested_conditions.",
    )
    generate.add_argument(
        "--counterpart-framing",
        default=None,
        help="Render the eval rows under this counterpart framing instead of the trained twin "
        "rendering. Swaps only the prompts; conditions, index-derived seeds, and the placebo "
        "vector are untouched — see generation_rows.",
    )
    generate.add_argument(
        "--row-manifest",
        type=Path,
        default=None,
        help="Private JSON manifest of complete rendered steering rows. When given, "
        "--conditions still selects interventions and never prompts.",
    )
    generate.add_argument(
        "--diagnostic-profile",
        default=None,
        help="Optional row-manifest profile. 'cooperation-generalization' validates exactly eight "
        "mixed costly-helping/payoff-control rows with per-family print-order balance.",
    )
    generate.add_argument("--out-dir", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one subcommand."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    runners: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
        "fit-direction": run_fit_direction,
        "logit-sweep": run_logit_sweep,
        "generate": run_generate,
    }
    runners[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
