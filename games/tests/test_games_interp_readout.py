"""Tests for `games.interp_readout`, the self-updating interp document.

Every test builds its own miniature artifact tree under `tmp_path` and never touches the real
`artifacts/` trees: a readout test that reads the real pass would pass or fail depending on which
analyses happen to have run on the box, and would start printing benchmark-adjacent material into
test output.

The load-bearing tests are the ones about absence. This document's whole job is to be honest about
what it could not read, so `TestAMissingInputRaisesTheBanner` deletes each input in turn and
requires the banner, the named file and a section stub -- never a crash and never a silent omission.
That group has been watched to fail: with `_status_lines` forced to return the complete-but-caveated
banner, every one of its cases went red, and with the section stubs suppressed the stub assertions
went red too.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch

from games.interp_readout import (
    DOCUMENT_FILENAME,
    DRIFT_FIGURE,
    MISSING_CELL,
    SHAPE_MODULE,
    SHAPE_SCRATCH,
    TOP_TOKENS,
    TRAJECTORY_FIGURE,
    Inputs,
    RowKind,
    displacement_shape,
    display_path,
    load_inputs,
    markdown_table,
    parse_condition,
    percent_cell,
    rate_cell,
    residual_fraction,
    two_path_shape,
    write_readout,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

ANALYSIS_DIR = "interp-analysis-2026-08-22"
MODULE_DIR = "interp-displacement-2026-08-22"
GPU_DIR = "interp-gpu-2026-08-22"
# A steering pass whose name deliberately does not match the lens pass glob, as a second box's does.
STEERING_ONLY_DIR = "interp-steering-completion-2026-08-22"
ARMS = ("twin-pd-group", "twin-pd-self")
SET_BY_SHORT = {
    "coarse": "cooperate-vs-defect-commitment",
    "lead": "correlated-vs-independent-counterpart",
    "decision": "causal-vs-functional-decision",
}
POOLINGS = ("last", "mean")
STEPS = (0, 10)
LAYERS = (0, 1)
HIDDEN = 8
STIMULI_DIGEST = "aa11bb22cc33dd44"
# The steered layer is one the along-axis payload covers, so section 6's check can look it up.
STEER_CELL = ("lead", LAYERS[-1], 2.0)
SECOND_STEER_CELL = ("decision", LAYERS[-1], 2.0)
SECOND_CELL_NAME = "group70-decision-cap16384"
KILLED_CELL_NAME = "group70-decision-cap8192"
LONGER_BUDGET = 16384
# Per axis, so one fixture tree renders both branches of the section 6 check: neither arm clears the
# placebo floor on the steered `lead` axis, both clear it on `decision`.
AXIS_FLOOR = 0.06
# The earlier checkpoint carries a floor no cosine could clear, so a check reading the wrong step shows.
EARLY_AXIS_FLOOR = 0.5
AXIS_COS_BY_SHORT: dict[str, dict[str, float]] = {
    "coarse": {"group_displacement": -0.10, "self_displacement": 0.05, "difference": -0.15},
    "lead": {"group_displacement": -0.02, "self_displacement": 0.01, "difference": -0.03},
    "decision": {"group_displacement": 0.13, "self_displacement": -0.11, "difference": 0.24},
}


# ------------------------------------------------------------------------------------------------
# Fixture construction: one small but structurally complete artifact tree
# ------------------------------------------------------------------------------------------------


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _identity() -> dict[str, Any]:
    return {
        "base_model": "Qwen/Qwen3.5-2B",
        "batch_size": 1,
        "compute_dtype": "bfloat16",
        "hidden_size": HIDDEN,
        "layer_convention": "post_block",
        "n_layers": len(LAYERS),
        "rendered_sha256": "ff00ff00ff00ff00",
        "stimuli_sha256": STIMULI_DIGEST,
        "stimulus_render": "templated_here",
        "store_dtype": "float32",
    }


def _axis_read(arm: str, stimulus_set: str, pooling: str, step: int, layer: int) -> dict[str, Any]:
    # Layer 1 is the peak everywhere: it clears both nulls, layer 0 clears neither.
    strong = layer == LAYERS[-1]
    return {
        "accuracy_above_placebo": 0.20 if strong else -0.05,
        "accuracy_empirical_p": 0.01 if strong else 0.80,
        "arm": arm,
        "beats_placebo": strong,
        "clears_null": strong,
        "direction_accuracy": 0.80 if strong else 0.45,
        "direction_norm": 0.5,
        "layer": layer,
        "n_pairs": 8,
        "placebo_accuracy_max": 0.60 if strong else 0.50,
        "placebo_accuracy_mean": 0.50,
        "pooling": pooling,
        "probe_accuracy": 0.85 if strong else 0.48,
        "probe_null_accuracy_max": 0.55,
        "split_half_cosine": 0.90 if strong else 0.20,
        "step": step,
        "stimulus_set": stimulus_set,
    }


def _drift_read(arm: str, stimulus_set: str, pooling: str, step: int, layer: int) -> dict[str, Any]:
    drift = 1.0 if step == 0 else 0.97 + 0.01 * layer
    gap = 0.6 + (0.05 if arm == ARMS[0] else -0.05) * (step / 10.0)
    return {
        "anchor_split_half_cosine": 0.90,
        "anchor_step": 0,
        "arm": arm,
        "cosine_anchor_placebo": 0.02,
        "cosine_to_anchor": drift,
        "heldout_negative_projection": -0.1,
        "heldout_positive_projection": 0.5,
        "heldout_projection_gap": gap,
        "heldout_projection_gap_placebo": 0.004,
        "layer": layer,
        "norm_ratio": 1.0,
        "pooling": pooling,
        "step": step,
        "stimulus_set": stimulus_set,
    }


def _trajectory_payload(arm: str, short: str, pooling: str) -> dict[str, Any]:
    stimulus_set = SET_BY_SHORT[short]
    label = f"{arm}|{stimulus_set}|{pooling}"
    return {
        "behavior_correlations": {
            label: {
                "behavior": [0.5, 0.4],
                "drift_cosine": {
                    "n_points": 2,
                    "pearson": 0.9,
                    "spearman": 1.0,
                    "unavailable_reason": None,
                },
                "peak_layer": LAYERS[-1],
                "projection_gap": {
                    "n_points": 2,
                    "pearson": -0.3,
                    "spearman": -0.5,
                    "unavailable_reason": None,
                },
                "steps": list(STEPS),
            }
        },
        "context": {
            "capture_root": "artifacts/games/interp-cells",
            "cells": ["base/step-0", *[f"{arm}/step-{step}" for step in STEPS if step]],
            "identity": _identity(),
            "layers": list(LAYERS),
            "n_placebos": 100,
            "positive_side": "A",
            "probe_config": {"l2_strength": 0.01, "max_iter": 100, "n_folds": 2, "seed": 0},
            "stimuli_file": "stimuli.jsonl",
            "stimuli_sha256": STIMULI_DIGEST,
        },
        "cross_arm": [],
        "groups": {
            label: {
                "anchor_step": 0,
                "arm": arm,
                "axis_reads": [
                    _axis_read(arm, stimulus_set, pooling, step, layer)
                    for step in STEPS
                    for layer in LAYERS
                ],
                "drift_reads": [
                    _drift_read(arm, stimulus_set, pooling, step, layer)
                    for step in STEPS
                    for layer in LAYERS
                ],
                "layers": list(LAYERS),
                "n_heldout_pairs": 4,
                "peak_layer": LAYERS[-1],
                "pooling": pooling,
                "stimulus_set": stimulus_set,
            }
        },
    }


def _write_trajectories(analysis_root: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    for pooling in POOLINGS:
        for short, stimulus_set in SET_BY_SHORT.items():
            for arm in ARMS:
                run_dir = analysis_root / f"trajectory-{pooling}-{short}-{arm.rsplit('-', 1)[-1]}"
                _write_json(run_dir / "trajectory.json", _trajectory_payload(arm, short, pooling))
                for step in STEPS:
                    target = run_dir / "directions" / arm / f"step-{step}"
                    target.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {layer: torch.randn(HIDDEN, generator=generator) for layer in LAYERS},
                        target / f"{stimulus_set}-{pooling}.pt",
                    )


def _quality(stimulus_set: str, stratum: str, layer: int) -> dict[str, Any]:
    strong = layer == LAYERS[-1]
    return {
        "accuracy_empirical_p": 0.01 if strong else 0.5,
        "direction_accuracy": 0.78 if strong else 0.46,
        "layer": layer,
        "n_pairs": 8,
        "placebo_accuracy_max": 0.60,
        "pooling": "last",
        "split_half_cosine": 0.88,
        "stimulus_set": stimulus_set,
        "stratum": stratum,
    }


def _axes_cell(label: str, arm: str, step: int) -> dict[str, Any]:
    sets = list(SET_BY_SHORT.values())
    return {
        "arm": arm,
        "axis_pairs": [
            {
                "cosine_real": 0.2,
                "layer": LAYERS[-1],
                "placebo_abs_cosine_max": 0.07,
                "pooling": "last",
                "set_a": sets[0],
                "set_b": sets[1],
                "split_half_a": 0.8,
                "split_half_b": 0.9,
                "stratum": "pooled",
            }
        ],
        "quality": [
            _quality(stimulus_set, stratum, layer)
            for stimulus_set in sets
            for stratum in ("pooled", "matched-column")
            for layer in LAYERS
        ],
        "step": step,
        "stratum_contrasts": [
            {
                "cosine_real": 0.06,
                "layer": LAYERS[-1],
                "placebo_abs_cosine_max": 0.06,
                "pooling": "last",
                "split_half_a": 0.86,
                "split_half_b": 0.83,
                "stimulus_set": sets[0],
                "stratum_a": "matched-column",
                "stratum_b": "split-diagonal-offdiagonal",
            }
        ],
        "_label": label,
    }


def _write_axes(analysis_root: Path) -> None:
    for arm in ARMS:
        cells = {
            "base/step-0": _axes_cell("base/step-0", "base", 0),
            f"{arm}/step-{STEPS[-1]}": _axes_cell(f"{arm}/step-{STEPS[-1]}", arm, STEPS[-1]),
        }
        _write_json(
            analysis_root / f"axes-{arm.rsplit('-', 1)[-1]}" / "axes.json",
            {"cells": cells, "context": {"stimuli_sha256": STIMULI_DIGEST}},
        )


def _displacement_read(sets: Sequence[str], step: int, layer: int) -> dict[str, Any]:
    return {
        "base_mean_norm": 1.2,
        "cos_arms": -0.30 if layer == LAYERS[-1] else -0.05,
        "cos_arms_crosshalf": [-0.29, -0.31],
        "layer": layer,
        "placebo_abs_cos_max": 0.06,
        "placebo_abs_cos_mean": 0.02,
        "pooling": "last",
        "norm_arm_minus_arm": 0.1,
        "norm_group": 0.06,
        "norm_self": 0.05,
        "reliability_group": 0.9,
        "reliability_self": 0.9,
        "same_axis_ceiling": 0.9,
        "sets": list(sets),
        "step": step,
    }


def _write_displacement(analysis_root: Path) -> None:
    all_sets = sorted(SET_BY_SHORT.values())
    reads = [
        _displacement_read(all_sets, step, layer) for step in STEPS[1:] for layer in LAYERS
    ] + [
        _displacement_read([stimulus_set], step, layer)
        for stimulus_set in all_sets
        for step in STEPS[1:]
        for layer in LAYERS
    ]
    nulls = [
        {
            "cos_null_to_real": 0.02,
            "layer": LAYERS[-1],
            "n_flipped": 4,
            "n_pairs": 8,
            "null_direction_accuracy": 0.45,
            "null_placebo_accuracy_max": 0.54,
            "null_placebo_accuracy_mean": 0.49,
            "placebo_floor_abs_cos_max": 0.06,
            "placebo_floor_abs_cos_mean": 0.02,
            "pooling": "last",
            "real_direction_accuracy": 0.78,
            "real_placebo_accuracy_max": 0.58,
            "set": SET_BY_SHORT["coarse"],
        }
    ]
    _write_json(analysis_root / "displacement.json", {"nulls": nulls, "reads": reads})
    _write_json(
        analysis_root / "displacement_vs_axes.json",
        [
            {
                "cos_axis": {
                    SET_BY_SHORT[short]: dict(entry) for short, entry in AXIS_COS_BY_SHORT.items()
                },
                "layer": layer,
                "placebo_abs_cos_max": AXIS_FLOOR if step == STEPS[-1] else EARLY_AXIS_FLOOR,
                "pooling": "last",
                "step": step,
            }
            for step in STEPS
            for layer in LAYERS
        ],
    )


def _write_two_path(analysis_root: Path) -> None:
    _write_json(
        analysis_root / "two_path_agreement.json",
        {
            "direction_cosines": [],
            "displacement_cosines": [],
            "raw_relative_l2": [],
            "summary": {
                "agreement_floor": 0.99,
                "direction_cosine_mean": 0.9987,
                "direction_cosine_min": 0.9337,
                "displacement_cosine_mean": 0.9992,
                "displacement_cosine_min": 0.9678,
                "n_below_floor": 3,
                "n_direction_reads": 40,
                "raw_relative_l2_max": 1.62,
                "shift_sabotage": {
                    "cosine_max": 0.96,
                    "cosine_mean": 0.78,
                    "expectation": "must sit far below the agreement floor",
                    "n_reads": 38,
                },
            },
        },
    )


# ------------------------------------------------------------------------------------------------
# The second displacement shape: what `games.interp_displacement` writes
# ------------------------------------------------------------------------------------------------
# Every reliability, ceiling and denominator below is a distinct value, so a test can pin which
# number reached which column: a pair/row swap or an of-mean/of-rows swap has to be visible.
PAIR_RELIABILITY = 0.91
ROW_RELIABILITY = 0.78
PAIR_CEILING = 0.90
ROW_CEILING = 0.70
RELATIVE_OF_MEAN = 0.00061
RELATIVE_OF_ROWS = 0.00052
OWN_AXIS_OFFSET = 0.001
POST_NORM_RELATIVE_L2 = 3.05
COMPARABLE_RELATIVE_L2 = 0.000275
MODULE_STRATIFICATION = "cited_cells"
MODULE_STRATUM = "matched-column"


def _module_arm(arm: str) -> dict[str, Any]:
    leading = arm == ARMS[0]
    return {
        "arm": arm,
        "displacement_norm": 0.0006 if leading else 0.0005,
        "relative_to_residual_mean_vector": RELATIVE_OF_MEAN if leading else 0.00051,
        "relative_to_residual_row_norm_mean": RELATIVE_OF_ROWS if leading else 0.00043,
        "split_half_cosine_pairs": PAIR_RELIABILITY,
        "split_half_cosine_rows": ROW_RELIABILITY,
        "unavailable_reason": None,
    }


def _module_arm_pair(layer: int) -> dict[str, Any]:
    return {
        "arm_a": ARMS[0],
        "arm_b": ARMS[1],
        "cosine_cross_halves_pairs": [-0.28, -0.30],
        "cosine_cross_halves_rows": [-0.20, -0.22],
        "cosine_real": -0.30 if layer == LAYERS[-1] else -0.05,
        "difference_norm": 0.0011,
        "same_axis_ceiling_pairs": PAIR_CEILING,
        "same_axis_ceiling_rows": ROW_CEILING,
    }


def _module_axis_components() -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for stimulus_set in sorted(SET_BY_SHORT.values()):
        for arm in ARMS:
            for reference in ("base", "own"):
                leading = arm == ARMS[0]
                components.append(
                    {
                        "axis_norm": 0.04,
                        "axis_reference": reference,
                        "axis_set": stimulus_set,
                        "cosine_to_axis": (-0.10 if leading else 0.05)
                        + (OWN_AXIS_OFFSET if reference == "own" else 0.0),
                        "projection": -0.00004 if leading else 0.00002,
                        "residual_norm": 0.0006,
                        "target": arm,
                    }
                )
        components.append(
            {
                "axis_norm": 0.04,
                "axis_reference": "base",
                "axis_set": stimulus_set,
                "cosine_to_axis": -0.16,
                "projection": -0.00006,
                "residual_norm": 0.0011,
                "target": f"{ARMS[0]}-minus-{ARMS[1]}",
            }
        )
    return components


def _module_read(  # noqa: PLR0913 - a read is a selection, a pooling, a step and a layer
    *,
    set_group: str,
    stimulus_sets: Sequence[str],
    stratification: str,
    stratum: str,
    pooling: str,
    step: int,
    layer: int,
) -> dict[str, Any]:
    return {
        "arm_pairs": [_module_arm_pair(layer)],
        "arms": [_module_arm(arm) for arm in ARMS],
        "axis_components": _module_axis_components(),
        "layer": layer,
        "n_pairs": 8,
        "n_rows": 16,
        "placebo_abs_cosine_max": 0.06,
        "placebo_abs_cosine_mean": 0.02,
        "pooling": pooling,
        "residual_mean_vector_norm": 1.23,
        "residual_row_norm_mean": 1.44,
        "row_parity_equals_side": stratification == "pooled",
        "set_group": set_group,
        "step": step,
        "stimulus_sets": list(stimulus_sets),
        "stratification": stratification,
        "stratum": stratum,
    }


def _module_reads() -> list[dict[str, Any]]:
    """Both poolings, the pooled and per-set selections, plus one stratified read of the same cell.

    The stratified read carries the same (set_group, step, layer) coordinates as its pooled parent, so
    a table that forgot to pin the stratification would silently stack two populations on one row.
    """
    all_sets = sorted(SET_BY_SHORT.values())
    reads: list[dict[str, Any]] = []
    for pooling in POOLINGS:
        for step in STEPS[1:]:
            for layer in LAYERS:
                reads.append(
                    _module_read(
                        set_group="pooled",
                        stimulus_sets=all_sets,
                        stratification="pooled",
                        stratum="pooled",
                        pooling=pooling,
                        step=step,
                        layer=layer,
                    )
                )
                reads.append(
                    _module_read(
                        set_group="pooled",
                        stimulus_sets=all_sets,
                        stratification=MODULE_STRATIFICATION,
                        stratum=MODULE_STRATUM,
                        pooling=pooling,
                        step=step,
                        layer=layer,
                    )
                )
                reads.extend(
                    _module_read(
                        set_group=stimulus_set,
                        stimulus_sets=[stimulus_set],
                        stratification="pooled",
                        stratum="pooled",
                        pooling=pooling,
                        step=step,
                        layer=layer,
                    )
                    for stimulus_set in all_sets
                )
    return reads


def _module_agreement_summary() -> dict[str, Any]:
    return {
        "agreement_floor": 0.99,
        "direction_cosine_mean": 0.99999,
        "direction_cosine_min": 0.9996,
        "displacement_cosine_mean": 0.99998,
        "displacement_cosine_min": 0.9981,
        "n_below_floor": 0,
        "n_comparable_reads": 6,
        "off_by_one_sabotage": {
            "cosine_max": 0.9667,
            "cosine_mean": 0.7775,
            "expectation": "must sit below the agreement floor",
            "is_red": True,
            "n_reads": 4,
        },
        "post_norm_direction_cosine_mean": 0.9698,
        "post_norm_direction_cosine_min": 0.9337,
        "post_norm_layer_reads": 6,
        "relative_l2_max": COMPARABLE_RELATIVE_L2,
        "top_layer": LAYERS[-1],
        "top_layer_treated_as_post_norm": True,
    }


def _module_displacement_payload() -> dict[str, Any]:
    return {
        "base_vs_base_null": {"max_abs_displacement": 0.0, "n_checks": 40},
        "context": {
            "arms": list(ARMS),
            "axis_references": ["base", "own"],
            "capture_root": "artifacts/games/interp-cells",
            "cells": ["base/step-0", *[f"{arm}/step-{STEPS[-1]}" for arm in ARMS]],
            "identity": _identity(),
            "layers": list(LAYERS),
            "n_placebos": 100,
            "null_layer": LAYERS[-1],
            "poolings": list(POOLINGS),
            "positive_side": "A",
            "stimuli_sha256": STIMULI_DIGEST,
        },
        "matrices": [
            {
                "arm": None,
                "arm_pair": f"{ARMS[0]}|{ARMS[1]}",
                "layers": list(LAYERS),
                "metric": "cross_arm_cosine",
                "pooling": "last",
                "set_group": "pooled",
                "steps": list(STEPS[1:]),
                "stratification": "pooled",
                "stratum": "pooled",
                "values": [[-0.05, -0.30]],
            }
        ],
        "reads": _module_reads(),
        "shuffled_label_nulls": {
            "layer": LAYERS[-1],
            "reads": [
                {
                    "clears_placebo": False,
                    "cosine_null_to_real": 0.02,
                    "layer": LAYERS[-1],
                    "n_flipped": 4,
                    "n_pairs": 8,
                    "null_direction_accuracy": 0.45,
                    "null_placebo_accuracy_max": 0.54,
                    "null_placebo_accuracy_mean": 0.49,
                    "placebo_floor_abs_cosine_max": 0.06,
                    "placebo_floor_abs_cosine_mean": 0.02,
                    "pooling": "last",
                    "real_direction_accuracy": 0.78,
                    "real_placebo_accuracy_max": 0.58,
                    "stimulus_set": SET_BY_SHORT["coarse"],
                }
            ],
            "skipped": [
                {
                    "n_pairs": 3,
                    "pooling": "last",
                    "reason": "3 pairs cannot fill 5 held-out folds",
                    "stimulus_set": SET_BY_SHORT["lead"],
                }
            ],
        },
        "skipped_selections": [
            {
                "n_rows": 0,
                "reason": "no rows in this set group",
                "set_group": SET_BY_SHORT["coarse"],
                "stratification": MODULE_STRATIFICATION,
                "stratum": "decision-scenario",
            }
        ],
        "two_path_agreement": _module_agreement_summary(),
    }


def _module_two_path_payload() -> dict[str, Any]:
    cells = ["base/step-0", f"{ARMS[0]}/step-{STEPS[-1]}"]
    reads = [
        {
            "cell": cell,
            "comparable": layer != LAYERS[-1],
            "direction_cosine": 0.9337 if layer == LAYERS[-1] else 0.99999,
            "layer": layer,
            "pooling": "last",
            "relative_l2": POST_NORM_RELATIVE_L2 if layer == LAYERS[-1] else COMPARABLE_RELATIVE_L2,
            "stimulus_set": stimulus_set,
        }
        for cell in cells
        for stimulus_set in sorted(SET_BY_SHORT.values())
        for layer in LAYERS
    ]
    displacement_reads = [
        {
            "cell": f"{ARMS[0]}/step-{STEPS[-1]}",
            "comparable": layer != LAYERS[-1],
            "cosine": 0.9981 if layer == LAYERS[-1] else 0.99998,
            "layer": layer,
            "pooling": "last",
        }
        for layer in LAYERS
    ]
    return {
        "displacement_reads": displacement_reads,
        "reads": reads,
        "summary": _module_agreement_summary(),
    }


def write_module_pass(artifacts_root: Path) -> Path:
    """Write a `games.interp_displacement` pass beside the scratch one, and return its directory."""
    module_root = artifacts_root / MODULE_DIR
    _write_json(module_root / "displacement.json", _module_displacement_payload())
    _write_json(module_root / "two_path_agreement.json", _module_two_path_payload())
    return module_root


def _per_layer(count: int) -> list[dict[str, Any]]:
    return [
        {
            "explained_variance": -0.3 + 0.1 * layer,
            "layer": layer,
            "n_samples": 16,
            "relative_residual": 1.2 - 0.05 * layer,
        }
        for layer in range(count)
    ]


def _lens_cell(arm: str, step: int, *, merged: bool) -> dict[str, Any]:
    return {
        "arm": arm,
        "dim_batch": 16,
        "direction_readout": {
            "direction_path": "/somewhere/interp-directions/lead-all.pt",
            "layer": 1,
            "placebo": [
                {"logit": 11.5 - index, "token": f"plac{index}"} for index in range(TOP_TOKENS)
            ],
            "real": [
                {"logit": 12.5 - index, "token": f"real{index}"} for index in range(TOP_TOKENS)
            ],
        },
        "fit_quality": {
            "available": True,
            "jacobian": {
                "best_layer": LAYERS[-1],
                "best_layer_relative_residual": 0.43,
                "mean_explained_variance": -0.17,
                "mean_relative_residual": 1.05,
                "median_explained_variance": -0.34,
                "median_relative_residual": 1.16,
                "n_samples": 16,
                "per_layer": _per_layer(len(LAYERS)),
            },
            "jacobian_beats_logit_lens": True,
            "logit_lens_baseline": {
                "best_layer": LAYERS[-1],
                "best_layer_relative_residual": 1.03,
                "mean_explained_variance": -0.46,
                "mean_relative_residual": 1.21,
                "median_explained_variance": -0.53,
                "median_relative_residual": 1.24,
                "n_samples": 16,
                "per_layer": _per_layer(len(LAYERS)),
            },
            "median_residual_reduction_vs_logit_lens": 0.075,
            "n_eval_prompts_skipped": 0,
            "n_eval_prompts_used": 8,
            "n_samples": 16,
            "positions_per_prompt_max": 8,
        },
        "fit_seconds": 400.0,
        "lens_path": "/somewhere/lens.pt",
        "max_seq_len": 452,
        "merged": merged,
        "model_dir": "/somewhere/model",
        "n_eval_prompts": 8,
        "n_fit_prompts": 10,
        "step": step,
    }


def _lens_ladder(cells: dict[str, Any], requested: Iterable[str]) -> dict[str, Any]:
    return {
        "base_model": "Qwen/Qwen3.5-2B",
        "cells": cells,
        "git_sha": "abc123",
        "merged_delta_realization_note": "merged bf16 exports realize ~64% of the trained delta",
        "requested_cells": sorted(requested),
        "seq_len_plan": {
            "corpus_max_tokens": 452,
            "fraction_truncated": 0.0,
            "max_seq_len": 452,
            "n_pairs": 20,
            "n_pairs_diverging_past_window": 0,
            "n_truncated": 0,
        },
        "stimuli_file": "stimuli.jsonl",
        "stimuli_sha256": STIMULI_DIGEST,
        "stimulus_render": "templated_here",
    }


def _write_lenses(gpu_root: Path) -> None:
    _write_json(
        gpu_root / "lens" / "smoke-16" / "lens_ladder.json",
        _lens_ladder({"base/step-0": _lens_cell("base", 0, merged=False)}, ["base/step-0"]),
    )
    _write_json(
        gpu_root / "lens" / "ladder-16-endpoints" / "lens_ladder.json",
        _lens_ladder(
            {f"{arm}/step-{STEPS[-1]}": _lens_cell(arm, STEPS[-1], merged=True) for arm in ARMS},
            [f"{arm}/step-{STEPS[-1]}" for arm in ARMS],
        ),
    )


def _condition(rate: float, parsed: int, truncated: int) -> dict[str, Any]:
    cooperate = round(rate * parsed)
    half = parsed // 2
    return {
        "by_print_order": {
            "canonical": {
                "cooperate_k": cooperate - cooperate // 2,
                "cooperate_rate": (cooperate - cooperate // 2) / half if half else 0.0,
                "n_parsed": half,
            },
            "swapped": {
                "cooperate_k": cooperate // 2,
                "cooperate_rate": (cooperate // 2) / (parsed - half) if parsed - half else 0.0,
                "n_parsed": parsed - half,
            },
        },
        "cooperate_k": cooperate,
        "cooperate_rate": cooperate / parsed if parsed else 0.0,
        "n_completions": 32,
        "n_parse_failures": 32 - parsed,
        "n_parsed": parsed,
        "n_truncated_thinking": truncated,
    }


def _steering_summary(
    arm: str,
    conditions: dict[str, Any],
    skipped: Iterable[str] = (),
    cell: tuple[str, int, float] = STEER_CELL,
    max_new_tokens: int = 8192,
) -> dict[str, Any]:
    direction, layer, alpha = cell
    return {
        "cells": [{"alpha_multiplier": alpha, "direction": direction, "layer": layer}],
        "command": "python -m games.interp_steering",
        "conditions": conditions,
        "max_new_tokens": max_new_tokens,
        "model": f"/merged/{arm}",
        "n_rows": 8,
        "n_samples": 4,
        "render_grading": "self",
        "resolved_sampler": {
            "applied": {
                "do_sample": True,
                "max_new_tokens": max_new_tokens,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
            },
            "engine": "hf",
            "presence_penalty_applied": None,
            "presence_penalty_requested": 0.0,
            "presence_penalty_why_dropped": "transformers has no presence_penalty field",
        },
        "seed": 400,
        "selection": None,
        "skipped_conditions": [{"condition_key": key, "skipped": "deadline"} for key in skipped],
        "split": "eval",
    }


def _full_conditions(cell: tuple[str, int, float] = STEER_CELL) -> dict[str, Any]:
    direction, layer, alpha = cell
    return {
        "none": _condition(0.30, 20, 8),
        f"{direction}:L{layer}:x{alpha}:steer:+": _condition(0.55, 18, 10),
        f"{direction}:L{layer}:x{alpha}:placebo:+": _condition(0.32, 28, 2),
        f"{direction}:L{layer}:x{alpha}:steer:-": _condition(0.06, 22, 9),
        f"{direction}:L{layer}:x{alpha}:placebo:-": _condition(0.31, 26, 4),
        f"{direction}:L{layer}:ablate:real": _condition(0.20, 16, 12),
        f"{direction}:L{layer}:ablate:placebo": _condition(0.29, 24, 6),
    }


def _write_records(path: Path, conditions: Iterable[str], *, truncated_parses: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    leaked = truncated_parses
    for key in conditions:
        for index in range(4):
            truncated = index == 0
            parsed = not truncated or leaked > 0
            if truncated and leaked > 0:
                leaked -= 1
            lines.append(
                json.dumps(
                    {
                        "condition_key": key,
                        "parsed_action": "cooperate" if parsed else None,
                        "response_text": "...",
                        "truncated_thinking": truncated,
                    }
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_steering(gpu_root: Path) -> None:
    direction, layer, alpha = STEER_CELL
    # The cooperation-trained arm ran the full complement; the other lost three to a deadline.
    full = _full_conditions()
    _write_json(
        gpu_root / "steering" / "arm-self" / "steering_summary.json",
        _steering_summary("self", full),
    )
    _write_records(gpu_root / "steering" / "arm-self" / "steering_records.jsonl", full)
    partial = {
        key: value
        for key, value in full.items()
        if key
        not in {
            f"{direction}:L{layer}:x{alpha}:placebo:-",
            f"{direction}:L{layer}:ablate:real",
            f"{direction}:L{layer}:ablate:placebo",
        }
    }
    _write_json(
        gpu_root / "steering" / "arm-group" / "steering_summary.json",
        _steering_summary(
            "group",
            partial,
            skipped=[
                f"{direction}:L{layer}:x{alpha}:placebo:-",
                f"{direction}:L{layer}:ablate:real",
                f"{direction}:L{layer}:ablate:placebo",
            ],
        ),
    )
    _write_records(gpu_root / "steering" / "arm-group" / "steering_records.jsonl", partial)
    # A killed run keeps its records and has no summary; it must not be read as a result.
    _write_records(gpu_root / "steering" / "capped" / "steering_records.jsonl", ["none"])
    _write_json(
        gpu_root / "steering" / "logit_sweep.json",
        {
            "baseline": {
                "mean_gap": -0.07,
                "mean_gap_by_print_order": {"canonical": -0.07, "swapped": -0.06},
                "n": 40,
            },
            "caveat": "thinking-off forced choice selects the intervention.",
            "cells": [
                {
                    "alpha_multiplier": alpha_multiplier,
                    "alpha_over_residual_norm": 0.43,
                    "alpha_raw": 1.7,
                    "condition": condition,
                    "direction": direction,
                    "layer": layer,
                    "mean_delta_by_print_order": {"canonical": -0.01, "swapped": 0.01},
                    "mean_delta_vs_none": 0.05 if condition.startswith("steer") else 0.004,
                    "mean_gap": -0.06,
                    "n": 40,
                }
                for alpha_multiplier in (1.0, alpha)
                for condition in ("steer:+", "steer:-", "placebo:+", "placebo:-")
            ],
            "denominators": {"n_skipped": 0, "n_total": 40, "n_used": 40, "skipped_prompt_ids": []},
            "model": "Qwen/Qwen3.5-2B",
            "render_grading": "self",
            "residual_norms_by_layer": {str(layer): 5.0},
            "seed": 0,
            "splits": ["eval"],
        },
    )


def write_steering_only_pass(artifacts_root: Path, cell_name: str = SECOND_CELL_NAME) -> Path:
    """Write a second pass holding steering cells and no lens ladder, and return its root.

    This is the shape a second rented box leaves behind: a pass name the lens glob never matches, a
    longer thinking budget than the first box ran, and a killed cell that kept its records.
    """
    pass_root = artifacts_root / STEERING_ONLY_DIR
    conditions = _full_conditions(SECOND_STEER_CELL)
    _write_json(
        pass_root / "steering" / cell_name / "steering_summary.json",
        _steering_summary(
            "group-70", conditions, cell=SECOND_STEER_CELL, max_new_tokens=LONGER_BUDGET
        ),
    )
    _write_records(pass_root / "steering" / cell_name / "steering_records.jsonl", conditions)
    _write_records(pass_root / "steering" / KILLED_CELL_NAME / "steering_records.jsonl", ["none"])
    return pass_root


def build_artifacts(root: Path) -> Path:
    """Write one structurally complete miniature pass pair, and return the artifacts root."""
    artifacts_root = root / "games"
    analysis_root = artifacts_root / ANALYSIS_DIR
    gpu_root = artifacts_root / GPU_DIR
    _write_trajectories(analysis_root)
    _write_axes(analysis_root)
    _write_displacement(analysis_root)
    _write_two_path(analysis_root)
    _write_lenses(gpu_root)
    _write_steering(gpu_root)
    return artifacts_root


@pytest.fixture
def artifacts_root(tmp_path: Path) -> Path:
    return build_artifacts(tmp_path)


def render(artifacts_root: Path, out_dir: Path) -> str:
    """Write the readout and return the markdown, which is what every assertion reads."""
    document = write_readout(artifacts_root, out_dir=out_dir)
    return document.read_text(encoding="utf-8")


def header_line(document: str, fragment: str) -> str:
    """The first markdown table header row containing `fragment`, never a prose line mentioning it."""
    for line in document.splitlines():
        if line.startswith("|") and fragment in line:
            return line
    raise AssertionError(f"no table header containing {fragment!r}")


def table_rows(document: str, header_fragment: str) -> list[str]:
    """The body rows of the first markdown table whose header contains `header_fragment`."""
    lines = document.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("|") and header_fragment in line:
            body: list[str] = []
            for row in lines[index + 2 :]:
                if not row.startswith("|"):
                    break
                body.append(row)
            return body
    raise AssertionError(f"no table header containing {header_fragment!r}")


def table_cell(document: str, header_fragment: str, column: str, row_index: int = 0) -> str:
    """One cell by column *name*, so a test pins the number to the column rather than to a position."""
    header = header_line(document, header_fragment)
    columns = [part.strip() for part in header.split("|")[1:-1]]
    row = table_rows(document, header_fragment)[row_index]
    values = [part.strip() for part in row.split("|")[1:-1]]
    assert column in columns, f"{column!r} is not a column of {columns}"
    return values[columns.index(column)]


# ------------------------------------------------------------------------------------------------
# The document as a whole
# ------------------------------------------------------------------------------------------------


class TestTheSevenThingsTheDocumentMustCarry:
    def test_every_section_is_present_and_the_document_is_not_incomplete(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        for heading in (
            "## 1. Axis quality",
            "### 2.heatmap",
            "## 3. Displacement from base",
            "## 4. Two-path capture agreement",
            "## 5. Jacobian lens",
            "## 6. Steering",
            "## Stated expectations, written before the numbers were looked at",
        ):
            assert heading in document, heading
        assert "INCOMPLETE AND ERROR-CONTAINING" not in document
        assert "COMPLETE BUT ERROR-CONTAINING" in document

    def test_both_figures_are_written_and_embedded(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"
        document = render(artifacts_root, out_dir)
        for name in (TRAJECTORY_FIGURE, DRIFT_FIGURE):
            path = out_dir / name
            assert path.is_file(), name
            assert path.stat().st_size > 5_000, f"{name} is too small to be a real figure"
            assert f"]({name})" in document, f"{name} is not embedded"

    def test_the_expectation_lines_are_marked_as_written_before_looking(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        assert "before any of the analyses below had run" in document
        assert "near-orthogonal once measured against the split-half" in document
        assert "moves cooperation by less than the training did" in document

    def test_the_axis_quality_grid_covers_every_layer_and_marks_the_peak(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        header = header_line(document, "group acc")
        rows = table_rows(document, "group acc")
        assert len(rows) == len(LAYERS)
        # One stable column set for the whole grid. A per-row column name would split the table into
        # two half-empty halves, so every row must carry exactly the header's column count.
        assert {row.count("|") for row in rows} == {header.count("|")}
        assert header.count("|") == 2 + 1 + len(ARMS) * 4
        # The peak marker lives in its own column, so it cannot rename a measurement column.
        peaks = [row.split("|")[2].strip() for row in rows]
        assert peaks[-1] == "group, self", peaks
        assert peaks[0] == "-", peaks
        # Layer 0 clears neither null in the fixture, layer 1 clears both.
        assert "no/no" in rows[0]
        assert "yes/yes" in rows[-1]

    def test_the_along_axis_split_reports_both_the_component_and_the_residual(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        header = header_line(document, "group resid")
        for column in ("group cos", "group resid", "self cos", "self resid", "diff", "floor_max"):
            assert column in header, column

    def test_the_displacement_table_puts_the_floor_and_ceiling_beside_the_cosine(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        header = header_line(document, "xhalf")
        assert header.index("cos_arms") < header.index("floor_max") < header.index("ceil")

    def test_the_cross_arm_cosine_is_rebuilt_from_the_saved_direction_tensors(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        rows = table_rows(document, "cos_arms | ceil")
        assert rows, "no cross-arm table"
        # Independent random directions in 8 dimensions: a real number, not 1.000 by construction.
        assert not any("+1.000" in row for row in rows)


class TestBothDisplacementShapesAreRead:
    """The displacement read exists in two shapes on disk; this group is about telling them apart.

    Watched to fail: with `displacement_shape` sabotaged so the module payload is classified as the
    scratch shape (`{"reads"} <= keys` tested first, and the module's key set misspelled),
    `test_each_shape_is_detected_from_its_own_top_level_keys` went red on the classification and every
    rendering test here went red where the scratch reader reached for `read["sets"]` on a module read.
    """

    def test_each_shape_is_detected_from_its_own_top_level_keys(self, artifacts_root: Path) -> None:
        scratch = json.loads(
            (artifacts_root / ANALYSIS_DIR / "displacement.json").read_text(encoding="utf-8")
        )
        assert displacement_shape(scratch) == SHAPE_SCRATCH
        assert displacement_shape(_module_displacement_payload()) == SHAPE_MODULE
        # Neither shape: named in the banner rather than handed to whichever reader it resembles.
        assert displacement_shape({"reads": [], "matrices": []}) is None
        assert two_path_shape(_module_two_path_payload()) == SHAPE_MODULE
        assert two_path_shape({"summary": {}}) is None

    def test_the_module_payload_wins_a_root_that_holds_both(self, artifacts_root: Path) -> None:
        write_module_pass(artifacts_root)
        inputs = load_inputs(artifacts_root)
        assert inputs.displacement_source is not None
        assert inputs.displacement_source.shape == SHAPE_MODULE
        assert inputs.displacement_source.path.parent.name == MODULE_DIR
        assert [source.path for source in inputs.displacement_unused] == [
            artifacts_root / ANALYSIS_DIR / "displacement.json"
        ]
        # The agreement payload follows the displacement payload, so one section cannot be rendered
        # from one pass while the check that validates it comes from another.
        assert inputs.two_path_source is not None
        assert inputs.two_path_source.path.parent.name == MODULE_DIR
        assert not inputs.missing

    def test_the_document_names_the_payload_it_read_and_the_one_it_did_not(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        write_module_pass(artifacts_root)
        document = render(artifacts_root, tmp_path / "out")
        assert "Displacement payload (section 3):" in document
        assert f"{MODULE_DIR}/displacement.json`" in document
        assert "Also on disk and NOT read here" in document
        assert f"{ANALYSIS_DIR}/displacement.json" in document
        # Naming an unread payload is not a coverage gap: nothing this document needs is absent.
        assert "INCOMPLETE AND ERROR-CONTAINING" not in document

    def test_the_ceiling_column_is_the_pair_split_with_the_row_split_beside_it(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        write_module_pass(artifacts_root)
        document = render(artifacts_root, tmp_path / "out")
        header = header_line(document, "ceil pairs")
        assert (
            header.index("cos_arms")
            < header.index("floor_max")
            < header.index("ceil pairs")
            < header.index("ceil rows")
        )
        assert table_cell(document, "ceil pairs", "ceil pairs") == f"{PAIR_CEILING:.3f}"
        assert table_cell(document, "ceil pairs", "ceil rows") == f"{ROW_CEILING:.3f}"
        # One row per layer: the mean-pooling reads and the stratified read of the same cell are not
        # on this table, and neither is the per-set selection.
        assert len(table_rows(document, "ceil pairs")) == len(LAYERS)
        assert "the row split above is a side split" in document

    def test_each_relative_norm_names_which_denominator_it_used(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        write_module_pass(artifacts_root)
        document = render(artifacts_root, tmp_path / "out")
        column = f"rel {ARMS[0].rsplit('-', 1)[-1]} (of-mean/of-rows)"
        assert table_cell(document, "ceil pairs", column) == (
            f"{RELATIVE_OF_MEAN:.5f}/{RELATIVE_OF_ROWS:.5f}"
        )
        assert "`residual_mean_vector_norm`" in document
        assert "`residual_row_norm_mean`" in document

    def test_the_along_axis_split_comes_from_the_module_payload(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        write_module_pass(artifacts_root)
        document = render(artifacts_root, tmp_path / "out")
        header = header_line(document, "group proj")
        for column in ("group cos", "group resid", "group-self proj", "self cos", "floor_max"):
            assert column in header, column
        assert "separate `displacement_vs_axes.json` from the scratch pass" in document
        assert f"differ by at most {OWN_AXIS_OFFSET:.4f}" in document

    def test_the_module_nulls_and_the_skipped_selections_are_named(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        write_module_pass(artifacts_root)
        document = render(artifacts_root, tmp_path / "out")
        assert "clears placebo" in header_line(document, "cos(null, real)")
        assert table_cell(document, "cos(null, real)", "clears placebo") == "no"
        assert "Nulls the pass could not run" in document
        assert "3 pairs cannot fill 5 held-out folds" in document
        assert "base cell standing in for an arm, over 40 checks" in document
        assert "### 3.f: selections the pass skipped" in document
        assert "no rows in this set group" in document
        # The stratified read is reported, at the pooled selection's most-opposed layer.
        assert MODULE_STRATUM in document
        # Sub-sections read in label order, so a reader following 3.a-3.f is not sent backwards.
        assert document.index("### 3.d:") < document.index("### 3.e:") < document.index("### 3.f:")

    def test_the_two_path_section_reports_the_post_norm_top_layer_apart(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        write_module_pass(artifacts_root)
        document = render(artifacts_root, tmp_path / "out")
        assert f"post-norm top layer L{LAYERS[-1]}" in document
        assert "The top layer is reported apart from every number above" in document
        assert "EXCLUDED from the row above" in document
        assert "collapsed below the floor as it must: yes" in document
        # The depth profile's first table is the comparable layers only, and the post-norm layer has
        # its own table beneath it -- a shared table would average the two together by eye.
        comparable = [row.split("|")[1].strip() for row in table_rows(document, "rel L2 max")]
        assert comparable == [str(LAYERS[0])]
        assert table_cell(document, "rel L2 max", "rel L2 max") == f"{COMPARABLE_RELATIVE_L2:.6f}"
        assert "Layers excluded from the summary and from the table above" in document
        assert f"{POST_NORM_RELATIVE_L2:.6f}" in document

    def test_a_module_pass_without_its_side_car_falls_back_to_the_embedded_summary(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        module_root = write_module_pass(artifacts_root)
        (module_root / "two_path_agreement.json").unlink()
        document = render(artifacts_root, tmp_path / "out")
        assert "as the summary embedded in the displacement payload" in document
        assert "carried no per-layer reads" in document
        assert f"post-norm top layer L{LAYERS[-1]}" in document


class TestEveryRateCarriesItsDenominator:
    def test_the_steering_rates_are_all_written_as_rate_over_denominator(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        rows = table_rows(document, "coop rate (k/parsed)")
        assert rows
        for row in rows:
            rate = row.split("|")[2].strip()
            assert MISSING_CELL in rate or re.fullmatch(r"[0-9.]+ \(\d+/\d+\)", rate), rate

    def test_censoring_and_parse_failures_are_reported_per_cell_with_denominators(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        header = header_line(document, "truncated thinking")
        assert "parse failures" in header
        rows = table_rows(document, "truncated thinking")
        assert any(re.search(r"\d+\.\d% \(\d+/\d+\)", row) for row in rows)

    def test_both_print_orders_are_split_out(self, artifacts_root: Path, tmp_path: Path) -> None:
        document = render(artifacts_root, tmp_path / "out")
        header = header_line(document, "coop rate (k/parsed)")
        assert "canonical (k/parsed)" in header
        assert "swapped (k/parsed)" in header
        assert "order spread" in header

    def test_a_zero_denominator_says_so_rather_than_rendering_a_rate(self) -> None:
        assert rate_cell(0, 0) == "n/a (0/0)"
        assert percent_cell(0, 0) == "n/a (0/0)"
        assert rate_cell(3, 6) == "0.500 (3/6)"
        assert percent_cell(1, 8) == "12.5% (1/8)"


class TestThePlaceboSitsBeneathTheRowItControls:
    def test_each_placebo_row_immediately_follows_its_true_direction_row(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        rows = table_rows(document, "coop rate (k/parsed)")
        labels = [row.split("|")[1].strip() for row in rows]
        placebo_positions = [index for index, label in enumerate(labels) if "PLACEBO" in label]
        assert placebo_positions
        for index in placebo_positions:
            previous = labels[index - 1]
            assert "PLACEBO" not in previous, f"two placebo rows in a row at {index}"
            assert previous.split(":")[0] == labels[index].split(":")[0]

    def test_a_skipped_placebo_still_occupies_its_row_with_the_reason(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        rows = table_rows(document, "coop rate (k/parsed)")
        missing = [row for row in rows if MISSING_CELL in row]
        assert missing, "the skipped conditions produced no rows at all"
        assert any("deadline" in row for row in missing)
        assert "Placebo coverage on this arm is one-sided" in document

    def test_ablation_appears_once_per_site_not_once_per_steered_magnitude(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        direction, layer, _alpha = STEER_CELL
        summary_path = artifacts_root / GPU_DIR / "steering" / "arm-self" / "steering_summary.json"
        summary = json.loads(summary_path.read_text())
        # A second steered magnitude at the same layer, which must not duplicate the ablation rows.
        summary["conditions"][f"{direction}:L{layer}:x1.0:steer:+"] = _condition(0.4, 20, 5)
        summary["conditions"][f"{direction}:L{layer}:x1.0:placebo:+"] = _condition(0.3, 20, 5)
        summary_path.write_text(json.dumps(summary))
        document = render(artifacts_root, tmp_path / "out")
        rows = table_rows(document, "coop rate (k/parsed)")
        ablate_real = [row for row in rows if ": ablate, direction" in row]
        assert len(ablate_real) == 1, f"ablation row duplicated: {ablate_real}"

    def test_the_selection_sweep_also_pairs_every_placebo_with_its_real_row(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        rows = table_rows(document, "a / resid norm")
        conditions = [row.split("|")[3].strip() for row in rows]
        assert conditions[:4] == ["steer:+", "**placebo:+**", "steer:-", "**placebo:-**"]


class TestTheLensSectionRefusesToLaunderAWeakFit:
    def test_the_caveat_states_the_fit_size_and_the_residual_from_the_artifact(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        assert "10 fit prompts at dim_batch 16" in document
        assert "median residual reduction 0.075" in document
        assert "median relative residual 1.160" in document
        assert "median explained variance -0.340" in document
        assert "uninterpretable at this fit size" in document

    def test_the_caveat_precedes_the_token_table(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        assert document.index("uninterpretable at this fit size") < document.index("real token")

    def test_every_real_token_column_has_a_placebo_column_beside_it(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        header = header_line(document, "real token")
        assert "placebo token" in header
        rows = table_rows(document, "real token")
        assert len(rows) == TOP_TOKENS

    def test_an_axis_the_lenses_never_transported_is_named_as_incomplete(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        assert "Axis coverage of section 5 is INCOMPLETE" in document
        assert "the coarse axis" in document

    def test_a_mixed_merged_ladder_says_the_attenuation_is_asymmetric(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        assert "The merge attenuation is asymmetric across this ladder" in document

    def test_an_identical_decode_is_reported_as_evidence_for_neither_reading(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        assert f"the top {TOP_TOKENS} are the same tokens in the same order at all 3" in document
        assert "this is not evidence either way" in document


class TestTheRecordLevelCensoringCheck:
    def test_a_clean_grid_reports_the_overlap_as_empty_with_its_denominator(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        assert re.search(r"Per-record check over \d+ records: none of the \d+ truncated", document)

    def test_a_truncated_record_that_parsed_is_called_a_problem(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        records = artifacts_root / GPU_DIR / "steering" / "arm-self" / "steering_records.jsonl"
        _write_records(records, _full_conditions(), truncated_parses=2)
        document = render(artifacts_root, tmp_path / "out")
        assert "PROBLEM:" in document
        assert "2 of the" in document
        assert "truncated records also parsed" in document

    def test_records_absent_is_reported_rather_than_assumed_clean(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        (artifacts_root / GPU_DIR / "steering" / "arm-self" / "steering_records.jsonl").unlink()
        document = render(artifacts_root, tmp_path / "out")
        assert f"No per-record file for `{GPU_DIR}/arm-self`" in document

    def test_a_records_directory_without_a_summary_is_named_and_not_read(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        assert f"and therefore **not results**: `{GPU_DIR}/capped`" in document
        assert f"### 6.{GPU_DIR}/capped" not in document


class TestSteeringSpansEveryPass:
    """Steering cells have come off more than one box, so one pass is not the grid.

    A single-pass read is invisible when it is wrong: the cells of whichever directory happened to
    be newest render as complete, and a reader concludes the others were never run.
    """

    def test_cells_from_both_passes_appear_keyed_by_their_pass(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        write_steering_only_pass(artifacts_root)
        inputs = load_inputs(artifacts_root)
        assert set(inputs.steering) == {
            f"{GPU_DIR}/arm-group",
            f"{GPU_DIR}/arm-self",
            f"{STEERING_ONLY_DIR}/{SECOND_CELL_NAME}",
        }
        document = render(artifacts_root, tmp_path / "out")
        assert f"### 6.{GPU_DIR}/arm-self" in document
        assert f"### 6.{STEERING_ONLY_DIR}/{SECOND_CELL_NAME}" in document
        # The second pass ran a longer thinking budget, which is why the pass has to be on the key.
        assert f"thinking budget {LONGER_BUDGET} tokens" in document

    def test_the_pass_is_on_every_key_even_when_no_two_cells_collide(
        self, artifacts_root: Path
    ) -> None:
        inputs = load_inputs(artifacts_root)
        assert set(inputs.steering) == {f"{GPU_DIR}/arm-group", f"{GPU_DIR}/arm-self"}

    def test_a_cell_name_used_by_both_passes_renders_twice_not_once(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        write_steering_only_pass(artifacts_root, cell_name="arm-self")
        inputs = load_inputs(artifacts_root)
        assert f"{GPU_DIR}/arm-self" in inputs.steering
        assert f"{STEERING_ONLY_DIR}/arm-self" in inputs.steering
        document = render(artifacts_root, tmp_path / "out")
        assert document.count("### 6.") >= len(inputs.steering)

    def test_a_records_directory_without_a_summary_is_named_in_every_pass(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        write_steering_only_pass(artifacts_root)
        document = render(artifacts_root, tmp_path / "out")
        assert f"`{GPU_DIR}/capped`" in document
        assert f"`{STEERING_ONLY_DIR}/{KILLED_CELL_NAME}`" in document

    def test_a_steering_pass_without_a_lens_pass_beside_it_is_still_read(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        write_steering_only_pass(artifacts_root)
        shutil.rmtree(artifacts_root / GPU_DIR)
        document = render(artifacts_root, tmp_path / "out")
        assert f"### 6.{STEERING_ONLY_DIR}/{SECOND_CELL_NAME}" in document
        assert "(GPU pass)" in document

    def test_a_second_logit_sweep_is_named_and_not_read(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        pass_root = write_steering_only_pass(artifacts_root)
        sweep = json.loads(
            (artifacts_root / GPU_DIR / "steering" / "logit_sweep.json").read_text(encoding="utf-8")
        )
        sweep["baseline"]["mean_gap"] = -0.99
        _write_json(pass_root / "steering" / "logit_sweep.json", sweep)
        document = render(artifacts_root, tmp_path / "out")
        assert "forced-choice logit sweeps are on disk" in document
        assert f"{STEERING_ONLY_DIR}/steering/logit_sweep.json" in document
        # The lens pass's copy is the one read, so the decoy's baseline must not reach the table.
        assert "-0.9900" not in document

    def test_no_summary_anywhere_still_names_the_pattern_and_stubs_the_section(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        for path in (artifacts_root / GPU_DIR / "steering").glob("*/steering_summary.json"):
            path.unlink()
        document = render(artifacts_root, tmp_path / "out")
        assert "generation-steering summaries" in document
        assert "steering/*/steering_summary.json" in document
        assert "the cooperation-rate steering table, its placebo rows" in document


class TestTheSteeredAxisIsCheckedAgainstItsDisplacement:
    """Section 6's closing block computes the section 3 read for the axis each cell steered.

    It replaced a hand-written paragraph asserting the two sections could not be composed, which was
    an assertion about artifacts made in prose instead of read off them.
    """

    def test_the_floor_comparison_is_computed_for_the_axis_actually_steered(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        header = header_line(document, "clears floor")
        for column in ("steered site", "steered by", "arm", "displacement cos", "clears floor"):
            assert column in header, column
        lead = AXIS_COS_BY_SHORT["lead"]
        assert table_cell(document, "clears floor", "displacement cos") == (
            f"{lead['group_displacement']:+.3f}"
        )
        assert table_cell(document, "clears floor", "arm") == "group"
        assert table_cell(document, "clears floor", "clears floor") == "no"
        assert table_cell(document, "clears floor", "clears floor", row_index=1) == "no"
        assert f"{GPU_DIR}/arm-self" in table_cell(document, "clears floor", "steered by")
        # The step and pooling the lookup used, so the row cannot be read off the wrong checkpoint.
        assert f"pooling `last`, step {STEPS[-1]}" in document
        assert table_cell(document, "clears floor", "placebo floor (max abs cos)") == (
            f"{AXIS_FLOOR:.3f}"
        )

    def test_no_arm_over_the_floor_says_the_steered_axis_did_not_move(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        assert (
            "No arm clears the floor, so training left no displacement along the axis" in document
        )
        assert "the tension is unresolved" not in document

    def test_an_arm_over_the_floor_makes_the_sign_comparison_live(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        write_steering_only_pass(artifacts_root)
        document = render(artifacts_root, tmp_path / "out")
        assert "The sign comparison against `d vs none` above is live" in document
        assert "group leans +" in document
        assert "self leans -" in document
        decision = AXIS_COS_BY_SHORT["decision"]
        assert f"group {decision['group_displacement']:+.3f} CLEARS" in document

    def test_a_steered_layer_the_payload_never_covered_is_named(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        summary_path = artifacts_root / GPU_DIR / "steering" / "arm-self" / "steering_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["cells"] = [{"alpha_multiplier": 2.0, "direction": "lead", "layer": 99}]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        document = render(artifacts_root, tmp_path / "out")
        assert "carries no read at pooling `last`" in document
        assert "`lead` L99" in document

    def test_the_absent_along_axis_payload_degrades_to_a_coverage_gap(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        (artifacts_root / ANALYSIS_DIR / "displacement_vs_axes.json").unlink()
        document = render(artifacts_root, tmp_path / "out")
        assert "Section 6.tension is INCOMPLETE" in document
        assert "displacement_vs_axes.json" in document
        assert "clears floor" not in document
        assert "the tension is unresolved" not in document


# ------------------------------------------------------------------------------------------------
# Absence: the group this document exists for
# ------------------------------------------------------------------------------------------------


class TestAMissingInputRaisesTheBanner:
    """Delete an input, require the banner, the named file and a stub. Never a crash.

    Watched to fail: forcing `_status_lines` to always return the complete-but-caveated banner turns
    every case here red, and suppressing the section stubs turns the stub assertions red.
    """

    def test_the_complete_tree_does_not_raise_the_banner(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        assert "INCOMPLETE AND ERROR-CONTAINING" not in render(artifacts_root, tmp_path / "out")

    @pytest.mark.parametrize(
        ("relative", "expected_stub"),
        [
            ("displacement.json", "Section 3"),
            ("two_path_agreement.json", "Section 4"),
        ],
    )
    def test_deleting_one_analysis_payload_names_it_and_stubs_its_section(
        self, artifacts_root: Path, tmp_path: Path, relative: str, expected_stub: str
    ) -> None:
        (artifacts_root / ANALYSIS_DIR / relative).unlink()
        document = render(artifacts_root, tmp_path / "out")
        assert "INCOMPLETE AND ERROR-CONTAINING" in document
        assert relative in document
        assert f"{expected_stub} is INCOMPLETE" in document

    def test_deleting_the_axes_report_stubs_the_geometry_section(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        for path in (artifacts_root / ANALYSIS_DIR).glob("axes-*"):
            shutil.rmtree(path)
        document = render(artifacts_root, tmp_path / "out")
        assert "INCOMPLETE AND ERROR-CONTAINING" in document
        assert "Section 2b is INCOMPLETE" in document

    def test_deleting_one_trajectory_cell_names_that_cell(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        shutil.rmtree(artifacts_root / ANALYSIS_DIR / "trajectory-last-lead-group")
        shutil.rmtree(artifacts_root / ANALYSIS_DIR / "trajectory-last-lead-self")
        document = render(artifacts_root, tmp_path / "out")
        assert "INCOMPLETE AND ERROR-CONTAINING" in document
        assert "trajectory report (last, lead)" in document

    def test_deleting_the_lens_pass_stubs_section_five(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        shutil.rmtree(artifacts_root / GPU_DIR / "lens")
        document = render(artifacts_root, tmp_path / "out")
        assert "INCOMPLETE AND ERROR-CONTAINING" in document
        assert "Section 5 is INCOMPLETE" in document
        assert "Jacobian lens ladders" in document

    def test_deleting_the_steering_pass_stubs_section_six(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        shutil.rmtree(artifacts_root / GPU_DIR / "steering")
        document = render(artifacts_root, tmp_path / "out")
        assert "INCOMPLETE AND ERROR-CONTAINING" in document
        assert "Section 6 is INCOMPLETE" in document

    def test_deleting_the_whole_gpu_pass_names_the_pass_and_keeps_both_sections(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        shutil.rmtree(artifacts_root / GPU_DIR)
        document = render(artifacts_root, tmp_path / "out")
        assert "INCOMPLETE AND ERROR-CONTAINING" in document
        assert "(GPU pass)" in document
        assert "## 5. Jacobian lens" in document
        assert "## 6. Steering" in document

    def test_an_empty_artifacts_root_still_writes_a_document_that_says_so(
        self, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        document = render(empty, tmp_path / "out")
        assert "INCOMPLETE AND ERROR-CONTAINING" in document
        assert "(CPU analysis pass)" in document
        assert "(GPU pass)" in document
        assert "The trajectory figure was not written" in document
        assert "The drift heatmap was not written" in document

    def test_a_pass_that_requested_a_fit_and_banked_none_is_named(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        _write_json(
            artifacts_root / GPU_DIR / "lens" / "smoke-64" / "lens_ladder.json",
            _lens_ladder({}, ["base/step-0"]),
        )
        document = render(artifacts_root, tmp_path / "out")
        assert "INCOMPLETE AND ERROR-CONTAINING" in document
        assert "requested by pass `smoke-64` and never banked" in document

    def test_the_newest_pass_is_the_one_read(self, artifacts_root: Path, tmp_path: Path) -> None:
        older = artifacts_root / "interp-analysis-2020-01-01"
        older.mkdir()
        # Selection is by modification time, so the decoy has to actually be older on disk.
        os.utime(older, (0, 0))
        document = render(artifacts_root, tmp_path / "out")
        assert ANALYSIS_DIR in document
        assert older.name not in document


# ------------------------------------------------------------------------------------------------
# Units
# ------------------------------------------------------------------------------------------------


class TestConditionKeyParsing:
    @pytest.mark.parametrize(
        ("key", "family", "sign", "is_placebo", "alpha"),
        [
            ("lead:L12:x2.0:steer:+", "steer", "+", False, 2.0),
            ("lead:L12:x0.5:placebo:-", "steer", "-", True, 0.5),
            ("decision:L18:ablate:real", "ablate", "", False, None),
            ("decision:L18:ablate:placebo", "ablate", "", True, None),
        ],
    )
    def test_a_known_key_parses_into_its_parts(
        self, key: str, family: str, sign: str, is_placebo: bool, alpha: float | None
    ) -> None:
        condition = parse_condition(key)
        assert condition.kind == RowKind(family, sign, is_placebo=is_placebo)
        assert condition.alpha_multiplier == alpha
        assert condition.site is not None

    def test_the_baseline_has_no_site(self) -> None:
        assert parse_condition("none").site is None
        assert parse_condition("none").target is None

    def test_an_unknown_key_raises_rather_than_being_dropped(self) -> None:
        with pytest.raises(ValueError, match="unrecognised steering condition key"):
            parse_condition("lead:L12:x2.0:mystery")


class TestFormattingPrimitives:
    def test_a_pipe_in_a_cell_is_escaped_so_it_cannot_split_the_row(self) -> None:
        rendered = markdown_table([{"token": "a|b"}])
        assert rendered.splitlines()[-1] == r"| a\|b |"

    def test_an_empty_table_says_so_rather_than_vanishing(self) -> None:
        assert markdown_table([]) == "_(no rows)_"

    def test_columns_are_the_union_in_order_of_first_appearance(self) -> None:
        rendered = markdown_table([{"a": 1}, {"b": 2, "a": 3}])
        assert rendered.splitlines()[0] == "| a | b |"

    def test_the_residual_fraction_is_the_orthogonal_complement(self) -> None:
        assert residual_fraction(0.0) == pytest.approx(1.0)
        assert residual_fraction(1.0) == pytest.approx(0.0)
        assert residual_fraction(-0.6) == pytest.approx(0.8)
        assert residual_fraction(None) is None

    def test_display_path_shortens_the_home_directory_out_of_the_document(self) -> None:
        assert display_path(Path.cwd() / "artifacts" / "x") == "artifacts/x"
        assert display_path(Path.home() / "elsewhere").startswith("~/")
        assert str(Path.home()) not in display_path(Path.home() / "elsewhere")


class TestLoadInputs:
    def test_a_named_root_overrides_the_glob(self, artifacts_root: Path) -> None:
        inputs = load_inputs(
            artifacts_root,
            analysis_root=artifacts_root / ANALYSIS_DIR,
            gpu_root=artifacts_root / GPU_DIR,
        )
        assert len(inputs.trajectories) == len(POOLINGS) * len(SET_BY_SHORT)
        assert len(inputs.lens_cells) == 1 + len(ARMS)
        assert set(inputs.steering) == {f"{GPU_DIR}/arm-group", f"{GPU_DIR}/arm-self"}
        assert inputs.unsummarised_steering == [f"{GPU_DIR}/capped"]
        assert not inputs.missing

    def test_a_malformed_payload_raises_rather_than_being_skipped(
        self, artifacts_root: Path
    ) -> None:
        (artifacts_root / ANALYSIS_DIR / "displacement.json").write_text("{not json")
        with pytest.raises(json.JSONDecodeError):
            load_inputs(artifacts_root)

    def test_an_empty_inputs_object_needs_no_payloads(self, tmp_path: Path) -> None:
        inputs = Inputs(artifacts_root=tmp_path)
        assert inputs.trajectories == {}
        assert inputs.missing == []


class TestTheDocumentLandsWhereItIsAsked:
    def test_the_default_directory_sits_under_the_analysis_pass(self, artifacts_root: Path) -> None:
        document = write_readout(artifacts_root)
        assert document == artifacts_root / ANALYSIS_DIR / "readout" / DOCUMENT_FILENAME
        assert document.is_file()

    def test_the_regeneration_command_is_in_the_document_it_reproduces(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        document = render(artifacts_root, tmp_path / "out")
        assert "Regenerate with: `uv run --frozen python -m games.interp_readout" in document
