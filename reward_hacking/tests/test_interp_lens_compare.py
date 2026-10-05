from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch

from reward_hacking.interp import lens_compare

if TYPE_CHECKING:
    from pathlib import Path


class SyntheticLens:
    def __init__(self, jacobians: dict[int, torch.Tensor], *, n_prompts: int = 3) -> None:
        self.jacobians = jacobians
        self.source_layers = sorted(jacobians)
        self.d_model = next(iter(jacobians.values())).shape[0]
        self.n_prompts = n_prompts

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        return residual @ self.jacobians[layer].T


class TinyTokenizer:
    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"token-{token_id}" for token_id in ids]


def test_identical_matrices_have_zero_change() -> None:
    before = torch.eye(12)
    metrics = lens_compare.compare_matrix_pair(before, before.clone(), layer=4, seed=91)

    assert metrics["relative_frobenius"] == 0.0
    assert metrics["cosine"] == 1.0
    assert metrics["principal_angles_degrees"]["16"] == pytest.approx([0.0] * 12)
    assert metrics["principal_angles_degrees"]["64"] == pytest.approx([0.0] * 12)


def test_scaled_matrix_has_known_relative_frobenius_change() -> None:
    before = torch.arange(1, 37, dtype=torch.float32).reshape(6, 6)
    metrics = lens_compare.compare_matrix_pair(before, before * 1.5, layer=4, seed=91)

    assert metrics["relative_frobenius"] == pytest.approx(0.5)
    assert metrics["cosine"] == 1.0


def test_top_change_vector_recovers_planted_rank_one_direction() -> None:
    generator = torch.Generator().manual_seed(32)
    planted_right = torch.randn(24, generator=generator)
    planted_left = torch.randn(24, generator=generator)
    delta = torch.outer(planted_left, planted_right)

    _, _, right_vectors = lens_compare.top_change_singular_vectors(delta, count=1, seed=91)

    recovered = right_vectors[:, 0]
    cosine = torch.dot(recovered, planted_right) / (recovered.norm() * planted_right.norm())
    assert torch.abs(cosine) > 0.999


def test_gemma_rms_norm_uses_one_plus_weight_convention() -> None:
    vector = torch.tensor([1.0, 2.0, 3.0])
    weight = torch.tensor([0.0, 0.5, -0.25])
    epsilon = 1e-6

    normalized = lens_compare.gemma_rms_norm(vector, weight, epsilon=epsilon)
    expected = vector * torch.rsqrt(vector.square().mean() + epsilon) * (1.0 + weight)

    torch.testing.assert_close(normalized, expected)


def test_fit_provenance_reads_approximate_running_mean_statistic(tmp_path: Path) -> None:
    lens_path = tmp_path / "after.pt"
    provenance_path = lens_path.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps({"prompts_fitted": 30, "final_max_d_mean_approx": 0.047}))

    provenance = lens_compare._fit_provenance(lens_path)

    assert provenance is not None
    assert provenance["final_max_d_mean"] == pytest.approx(0.047)
    assert provenance["final_max_d_mean_approximate"] is True


def test_cli_writes_json_and_markdown_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = SyntheticLens({layer: torch.eye(8) for layer in (12, 26, 40)})
    after = SyntheticLens(
        {layer: torch.eye(8) + 0.1 * torch.ones((8, 8)) for layer in (12, 26, 40)}
    )
    noise = SyntheticLens(
        {layer: torch.eye(8) + 0.02 * torch.ones((8, 8)) for layer in (12, 26, 40)}
    )
    lenses = {"before.pt": before, "after.pt": after, "noise.pt": noise}
    embeddings = torch.eye(8)
    norm_weight = torch.zeros(8)
    tokenizer = TinyTokenizer()

    monkeypatch.setattr(lens_compare, "load_lens", lambda path: lenses[str(path)])
    monkeypatch.setattr(
        lens_compare,
        "load_gemma_readout_weights",
        lambda model: (embeddings, norm_weight, 1e-6),
    )
    monkeypatch.setattr(lens_compare, "load_tokenizer", lambda model: tokenizer)

    lens_compare.main(
        [
            "--before",
            "before.pt",
            "--after",
            "after.pt",
            "--noise-lens",
            "noise.pt",
            "--model",
            "google/gemma-3-12b-it",
            "--out-dir",
            str(tmp_path),
        ]
    )

    output_json = tmp_path / "comparison.json"
    output_markdown = tmp_path / "summary.md"
    result = json.loads(output_json.read_text())
    assert result["model"] == "google/gemma-3-12b-it"
    assert result["comparisons"]["before_after"]["layers"]["12"]["relative_frobenius"] > 0
    assert result["comparisons"]["after_noise"]["layers"]["12"]["relative_frobenius"] > 0
    assert result["contents"]["12"]["directions"][0]["positive_tokens"]
    assert "| Layer |" in output_markdown.read_text()
