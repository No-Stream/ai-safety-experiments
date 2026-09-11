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
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from games.chunked_decode import decode_in_chunks
from games.deltanet_kernels import (
    DELTANET_KERNEL_FIELD,
    assert_one_deltanet_kernel,
    bound_deltanet_kernels,
    bridge_and_check_decode_kernel,
)
from games.eval_sampler import SAMPLER_TRAINING_DISTRIBUTION, eval_sampling
from games.interp_cells import (
    concept_activations,
    digest_of_strings,
    load_ladder,
    load_stimuli,
    pair_layout,
    stimuli_digest,
)
from games.parsing import parse_action, strip_thinking
from games.payoffs import COOPERATE
from games.prompts import (
    LABEL_PRINT_ORDERS,
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
from reward_hacking.model_backend import HFBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from games.interp_cells import CapturedCell

logger = logging.getLogger(__name__)

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


def generation_rows(counterpart_framing: str | None) -> list[dict[str, Any]]:
    """Return the generate leg's held-out eval rows, both print orders, optionally re-framed.

    None is the trained rendering (`generate_prompt_rows`, battery prompt_ids). A framing id
    renders the same rows with only the "About the other side" paragraph swapped
    (`generate_framing_prompt_rows`; its `twin` is byte-identical prompt text under
    framing-tagged prompt_ids). The framing swaps WHICH prompts the cell generates on and
    nothing else -- conditions, their index-derived seeds, and the placebo vector are untouched,
    which is what keeps a framing cell seed-comparable to the twin cells the interp arc banked.
    """
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


def render_forced_choice(tokenizer: AutoTokenizer, prompt: str) -> str:
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
    model = cast(
        "AutoModelForCausalLM",
        backend._model,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # hooks need the module tree
    )
    tokenizer = backend._tokenizer  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # same render as decode

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


def summarise_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cooperation as k/n per condition key, pooled and split by print order, with denominators.

    Refuses a pool whose records ran under two Gated DeltaNet kernel bindings before it averages
    anything: the fused decode kernel and the torch fallback diverge in their greedy token ids, so a
    steering condition decoded under one and its placebo under the other would report kernel noise as
    a steering effect. Every arm of this instrument is meant to be one run's worth of decoding, so a
    mixed pool is a relaunch that crossed the bridge rather than a comparison worth making.
    """
    assert_one_deltanet_kernel(
        (record.get(DELTANET_KERNEL_FIELD) for record in records),
        what="this steering summary",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["condition_key"]), []).append(record)
    summary: dict[str, Any] = {}
    for key, group in sorted(grouped.items()):
        parsed = [r for r in group if r["cooperate"] is not None]
        cooperated = sum(1 for r in parsed if bool(r["cooperate"]))
        entry: dict[str, Any] = {
            "n_completions": len(group),
            "n_parsed": len(parsed),
            "n_parse_failures": len(group) - len(parsed),
            "n_truncated_thinking": sum(1 for r in group if bool(r["truncated_thinking"])),
            "cooperate_k": cooperated,
            "cooperate_rate": None if not parsed else cooperated / len(parsed),
        }
        by_order: dict[str, dict[str, Any]] = {}
        for order in sorted({str(r["label_print_order"]) for r in group}):
            order_parsed = [p for p in parsed if str(p["label_print_order"]) == order]
            order_k = sum(1 for r in order_parsed if bool(r["cooperate"]))
            by_order[order] = {
                "n_parsed": len(order_parsed),
                "cooperate_k": order_k,
                "cooperate_rate": None if not order_parsed else order_k / len(order_parsed),
            }
        entry["by_print_order"] = by_order
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
    prompts: Sequence[str],
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
        "n_prompts": len(prompts),
        "prompts_sha256": digest_of_strings(prompts),
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "counterpart_framing": args.counterpart_framing,
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
    seed: int,
) -> tuple[AbstractContextManager[None], float | None]:
    """Build one condition's hook context and its raw alpha; the baseline gets a no-op context.

    The placebo is seeded from the run seed, the direction's index and the layer -- never from the
    condition's position in the list -- so every placebo condition of a (direction, layer) pushes the
    same random vector and a gap-filled or resumed run draws the vector an unbroken one would have.
    """
    if condition.condition == CONDITION_NONE:
        return nullcontext(), None
    direction_name = cast("str", condition.direction)
    layer = cast("int", condition.layer)
    real = directions[direction_name][layer]
    placebo = placebo_for(
        real, seed=seed, direction_index=direction_names.index(direction_name), layer=layer
    )
    alpha_raw = (
        None
        if condition.alpha_multiplier is None
        else condition.alpha_multiplier * float(real.norm())
    )
    hook = hook_for(condition.condition, real, placebo, alpha_raw or 0.0)
    return residual_intervention(model, layer, hook), alpha_raw


def run_generate(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915 - one linear driver
    """Generate the steering grid on held-out eval prompts and write records plus a summary.

    Resumes automatically: a records file already in ``--out-dir`` is read through its resume
    ledger, every complete condition is carried forward and skipped, and a partial one (a reclaim
    mid-condition) is dropped and regenerated. Each condition reseeds from its own index, so the
    regenerated records are the ones an uninterrupted run would have written; the summary counts
    ``resumed_conditions`` apart from ``skipped_conditions`` so what actually ran stays visible.
    """
    directions = load_named_directions(args.direction)
    if (args.cells is None) == (args.from_sweep is None):
        raise ValueError("pass exactly one of --cells or --from-sweep.")
    if args.from_sweep is not None:
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

    deadline = None if args.deadline is None else dt.datetime.fromisoformat(args.deadline)
    sampling = replace(
        eval_sampling(SAMPLER_TRAINING_DISTRIBUTION, thinking=True),
        max_new_tokens=args.max_new_tokens,
    )
    # Before the load, which imports the modeling module and freezes each kernel's dispatch.
    kernel_bridge = bridge_and_check_decode_kernel()
    backend = HFBackend(args.model, thinking=True, sampling=sampling)
    model = cast(
        "AutoModelForCausalLM",
        backend._model,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # hooks need the module tree
    )
    deltanet_kernel = bound_deltanet_kernels()
    # After the load, so a hub id resolves from the same cache the weights just came out of.
    weights_identity = resolve_weights_identity(args.model)

    rows = generation_rows(args.counterpart_framing)
    prompts = [str(row["prompt"]) for row in rows for _ in range(args.n_samples)]
    row_for_prompt = [row for row in rows for _ in range(args.n_samples)]
    sample_indices = [index for _ in rows for index in range(args.n_samples)]

    conditions = generation_conditions(cells)
    requested = parse_requested_conditions(args.conditions, conditions)
    direction_names = list(directions)
    sampler_payload = resolved_sampler(sampling).as_payload()
    skipped_conditions: list[dict[str, Any]] = []
    resumed_conditions: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / GENERATION_RECORDS_FILENAME
    resume_state, ledger = resume_records(
        records_path,
        identity=generation_identity(
            args,
            conditions,
            directions,
            prompts=prompts,
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
            torch.manual_seed(seed)
            context, alpha_raw = condition_intervention(
                condition, directions, direction_names, model=model, seed=args.seed
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
                record: dict[str, Any] = {
                    "condition_key": key,
                    DELTANET_KERNEL_FIELD: deltanet_kernel,
                    "condition": condition.condition,
                    "direction": condition.direction,
                    "layer": condition.layer,
                    "alpha_multiplier": condition.alpha_multiplier,
                    "alpha_raw": alpha_raw,
                    "seed": seed,
                    "counterpart_framing": args.counterpart_framing,
                    "prompt_id": row["prompt_id"],
                    "reskin_id": row["reskin_id"],
                    "payoff_variant": row["payoff_variant"],
                    "label_print_order": row["label_print_order"],
                    "coop_label": row["coop_label"],
                    "sample_index": sample_index,
                    "response_text": completion,
                    "parsed_action": action,
                    "cooperate": None if action is None else action == COOPERATE,
                    "truncated_thinking": truncated,
                }
                records.append(record)
                records_file.write(json.dumps(record) + "\n")
            records_file.flush()
            ledger.mark_complete(key, len(completions))
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
        "n_rows": len(rows),
        "n_samples": args.n_samples,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "resolved_sampler": sampler_payload,
        "cells": cells,
        "selection": selection,
        "requested_conditions": None
        if requested is None
        else sorted(condition_key(conditions[index]) for index in requested),
        "conditions": summarise_records(records),
        "skipped_conditions": skipped_conditions,
        "resumed_conditions": resumed_conditions,
        "n_records_resumed": resume_state.n_kept,
        "n_records_dropped_partial": resume_state.dropped_records,
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
    generate.add_argument("--direction", action="append", default=[], metavar="NAME=PATH")
    generate.add_argument("--cells", action="append", default=None, metavar="NAME:LAYER:MULT")
    generate.add_argument("--from-sweep", default=None)
    generate.add_argument("--n-cells", type=int, default=3)
    generate.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    generate.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    generate.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    generate.add_argument("--seed", type=int, default=0)
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
