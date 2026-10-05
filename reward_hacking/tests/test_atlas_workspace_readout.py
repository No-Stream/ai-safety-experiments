from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from games.workspace_readout import LoadedLens, StoredUnembed
from reward_hacking.atlas.workspace_readout import (
    CellAnalysisConfig,
    CPUUnembedding,
    analyze_cell,
    normalize_for_projection,
    position_indices,
)


def test_position_indices_respect_code_start_and_masks() -> None:
    rows: list[dict[str, Any]] = [
        {"code_token_start": 0},
        {"code_token_start": 2},
        {"code_token_start": None},
        {"code_token_start": 9},
    ]
    mask = np.ones((4, 12), dtype=np.bool_)
    mask[1, 6] = False

    code_before, valid_before = position_indices(rows, mask, "code_start_minus_1", prompt_tokens=4)
    code_start, valid_start = position_indices(rows, mask, "code_start", prompt_tokens=4)

    assert code_before.tolist() == [3, 5, -1, -1]
    assert valid_before.tolist() == [0, 1]
    assert code_start.tolist() == [4, -1, -1, -1]
    assert valid_start.tolist() == [0]


def test_normalize_for_projection_matches_stored_gemma_unembed() -> None:
    unembed = StoredUnembed(
        norm_weight=torch.tensor([0.2, -0.1]),
        lm_head_weight=torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
        rms_eps=1e-6,
    )
    residuals = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

    projected_mean = (
        normalize_for_projection(residuals, unembed).mean(dim=0) @ unembed.lm_head_weight.T
    )
    expected_mean = unembed.unembed(residuals).mean(dim=0)

    torch.testing.assert_close(projected_mean, expected_mean)


def test_normalize_for_projection_preserves_bfloat16_lm_head_input_rounding() -> None:
    bfloat16_unembed = StoredUnembed(
        norm_weight=torch.tensor([0.2, -0.1], dtype=torch.bfloat16),
        lm_head_weight=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16),
        rms_eps=1e-6,
    )
    cached_projection = CPUUnembedding(
        norm_weight=bfloat16_unembed.norm_weight.float(),
        lm_head_weight=bfloat16_unembed.lm_head_weight.float(),
        lm_head_input_dtype=bfloat16_unembed.lm_head_weight.dtype,
        rms_eps=bfloat16_unembed.rms_eps,
    )
    residuals = torch.tensor([[0.123456, 1.23456], [2.34567, 0.456789]])

    projected_mean = (
        normalize_for_projection(residuals, cached_projection).mean(dim=0)
        @ cached_projection.lm_head_weight.T
    )
    expected_mean = bfloat16_unembed.unembed(residuals).mean(dim=0)

    torch.testing.assert_close(projected_mean, expected_mean)


def test_analyze_cell_reports_paired_and_hack_contrasts_without_item_ids() -> None:
    vocabulary = [" alpha", " bravo", " charlie", " delta", " echo"]
    unembed = StoredUnembed(
        norm_weight=torch.zeros(2),
        lm_head_weight=torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.4, 0.4]]),
        rms_eps=1e-6,
    )
    lens = LoadedLens(jacobians={8: torch.eye(2)}, n_prompts=3, d_model=2)
    rows: list[dict[str, Any]] = []
    adapter_rows: list[list[float]] = []
    base_rows: list[list[float]] = []
    for item_id in ("synthetic-item-a", "synthetic-item-b"):
        for hack, residual in (
            (True, [2.0, 1.0]),
            (True, [2.0, 1.0]),
            (False, [1.0, 2.0]),
            (False, [1.0, 2.0]),
        ):
            rows.append(
                {
                    "item_id": item_id,
                    "hack_label": hack,
                    "pass_label": not hack,
                }
            )
            adapter_rows.append(residual)
            base_rows.append([1.0, 1.0])

    config = CellAnalysisConfig(
        layer=8,
        position="completion_0",
        lens=lens,
        unembed=unembed,
        vocabulary=vocabulary,
        readable_mask=torch.ones(len(vocabulary), dtype=torch.bool),
        n_bootstrap_halves=4,
        seed=17,
        batch_size=32,
    )
    result = analyze_cell(
        np.asarray(adapter_rows, dtype=np.float32),
        np.asarray(base_rows, dtype=np.float32),
        rows,
        config,
    )

    assert result["n_samples"] == 8
    assert result["n_mixed_items"] == 2
    assert (
        result["hack_minus_honest"]["within_mixed_items"]["adapter"]["risen"][0]["token"]
        == " alpha"
    )
    assert result["hack_minus_honest"]["within_mixed_items"]["base_same_tokens"]["risen"] == []
    assert (
        result["placebo"]["matched_norm_random_perturbation"]["overlap_with_real"]["risen_jaccard"]
        >= 0.0
    )
    shuffled_placebos = result["placebo"]["shuffled_labels"]
    assert set(shuffled_placebos) == {"all_samples", "within_items"}
    global_overlap = shuffled_placebos["all_samples"]["adapter"]["all_samples"]["overlap_with_real"]
    assert 0.0 <= global_overlap["risen_jaccard"] <= 1.0
    within_magnitude_ratio = shuffled_placebos["within_items"]["adapter"]["within_mixed_items"][
        "magnitude"
    ]["mean_absolute_ratio_to_real"]
    assert within_magnitude_ratio is not None
    assert "synthetic-item-a" not in json.dumps(result)
    assert "synthetic-item-b" not in json.dumps(result)
