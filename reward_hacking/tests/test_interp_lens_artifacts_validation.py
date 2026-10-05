"""Corrupted lens files must fail at the CPU artifact boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from reward_hacking.interp.lens_artifacts import LensArtifactError, SavedLens
from reward_hacking.interp.lens_deltanet_gates import GateFailureError, compare_jacobian_rows
from reward_hacking.interp.lens_geometry import StoredDecodeGate

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_saved_lens_refuses_nonfinite_matrix(tmp_path: Path, bad_value: float) -> None:
    matrix = torch.eye(2)
    matrix[0, 0] = bad_value
    path = tmp_path / "lens.pt"
    torch.save({"J": {0: matrix}, "source_layers": [0], "d_model": 2, "n_prompts": 1}, path)
    with pytest.raises(LensArtifactError, match="finite"):
        SavedLens.load(path)


@pytest.mark.parametrize("n_prompts", [0, -1])
def test_saved_lens_refuses_unfitted_accumulator(tmp_path: Path, n_prompts: int) -> None:
    path = tmp_path / "lens.pt"
    torch.save(
        {"J": {0: torch.eye(2)}, "source_layers": [0], "d_model": 2, "n_prompts": n_prompts}, path
    )
    with pytest.raises(LensArtifactError, match="positive"):
        SavedLens.load(path)


def test_saved_lens_refuses_empty_layer_map(tmp_path: Path) -> None:
    path = tmp_path / "lens.pt"
    torch.save({"J": {}, "source_layers": [], "d_model": 2, "n_prompts": 1}, path)
    with pytest.raises(LensArtifactError, match="no source layers"):
        SavedLens.load(path)


def test_decode_gate_refuses_nonfinite_anchors() -> None:
    with pytest.raises(ValueError, match="finite"):
        StoredDecodeGate(
            residuals=torch.full((2, 3, 4), float("nan")),
            target_token_ids=torch.tensor([0, 1]),
            lens_name="synthetic",
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_kernel_row_comparison_refuses_nonfinite_estimator(bad_value: float) -> None:
    candidate = {0: torch.full((2, 4), bad_value)}
    reference = {0: torch.ones(2, 4)}
    with pytest.raises(GateFailureError, match="finite"):
        compare_jacobian_rows(candidate, reference)
