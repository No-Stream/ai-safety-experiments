"""CPU contracts for Jacobian-lens subspace bundle construction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest
import torch

from games.workspace_ablation import (
    build_placebo_bundle,
    build_real_bundle,
    effective_token_vectors,
    load_bundle,
    load_token_spec,
    orthonormalize_rows,
    token_ids_from_tokenizer,
)
from games.workspace_readout import LoadedLens, StoredUnembed

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase


def _synthetic_inputs() -> tuple[LoadedLens, StoredUnembed]:
    lens = LoadedLens(
        jacobians={
            0: torch.tensor([[1.0, 2.0], [0.0, 1.0]]),
            1: torch.tensor([[0.5, -1.0], [2.0, 0.25]]),
        },
        n_prompts=2,
        d_model=2,
    )
    unembed = StoredUnembed(
        norm_weight=torch.tensor([0.25, -0.5]),
        lm_head_weight=torch.tensor(
            [
                [1.0, 2.0],
                [-2.0, 0.5],
                [0.25, 3.0],
            ]
        ),
        rms_eps=1e-6,
    )
    return lens, unembed


def test_token_strings_resolve_by_readout_decode_identity() -> None:
    class TinyTokenizer:
        vocab_size = 3

        def decode(self, ids: list[int]) -> str:
            return [" synthetic_a", " synthetic_b", " synthetic_c"][ids[0]]

        def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
            del text, add_special_tokens
            return {"input_ids": [0]}

    tokenizer = TinyTokenizer()
    assert token_ids_from_tokenizer(
        cast("PreTrainedTokenizerBase", tokenizer), [" synthetic_b"]
    ) == [1]


def test_ambiguous_decoded_token_is_rejected() -> None:
    class TinyTokenizer:
        vocab_size = 2

        def decode(self, ids: list[int]) -> str:
            del ids
            return " synthetic"

    with pytest.raises(ValueError, match="one decoded vocabulary id"):
        token_ids_from_tokenizer(cast("PreTrainedTokenizerBase", TinyTokenizer()), [" synthetic"])


def test_bundle_loader_rejects_negative_decoder_layer(tmp_path: Path) -> None:
    bundle_path = tmp_path / "invalid.pt"
    torch.save({-1: torch.tensor([[1.0, 0.0]])}, bundle_path)
    with pytest.raises(ValueError, match="layer"):
        load_bundle(bundle_path)


def test_jacobian_vector_matches_pre_norm_lens_logit_derivative() -> None:
    lens, unembed = _synthetic_inputs()
    token_ids = [1]
    vectors = effective_token_vectors(lens, unembed, token_ids, layers=[0])
    residual = torch.tensor([0.7, -1.3])
    effective_unembed = unembed.lm_head_weight[1] * (1.0 + unembed.norm_weight)
    expected_logit_contribution = effective_unembed @ (lens.jacobians[0] @ residual)

    assert torch.allclose(vectors[0][0] @ residual, expected_logit_contribution)
    assert torch.allclose(vectors[0][0], lens.jacobians[0].T @ effective_unembed)


def test_orthonormalize_rows_reduces_rank_and_preserves_row_span() -> None:
    rows = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])

    basis, rank = orthonormalize_rows(rows)

    assert rank == 2
    assert basis.shape == (2, 3)
    assert torch.allclose(basis @ basis.T, torch.eye(2), atol=1e-6)
    projector = basis.T @ basis
    assert torch.allclose(projector @ rows.T, rows.T, atol=1e-6)


def test_placebo_is_deterministic_and_identity_derived() -> None:
    real = {0: torch.eye(4)[:2], 1: torch.eye(4)[2:]}

    first = build_placebo_bundle(real, identity_sha256="a" * 64)
    second = build_placebo_bundle(real, identity_sha256="a" * 64)
    changed = build_placebo_bundle(real, identity_sha256="b" * 64)

    assert all(torch.equal(first[layer], second[layer]) for layer in real)
    assert any(not torch.equal(first[layer], changed[layer]) for layer in real)
    assert all(
        torch.allclose(value @ value.T, torch.eye(value.shape[0]), atol=1e-6)
        for value in first.values()
    )


def test_build_real_bundle_selects_half_open_band_and_records_effective_rank() -> None:
    lens, unembed = _synthetic_inputs()

    bundle, ranks = build_real_bundle(
        lens,
        unembed,
        token_ids=[0, 2],
        band=(0, 2),
    )

    assert set(bundle) == {0, 1}
    assert ranks == {0: 2, 1: 2}
    assert all(value.device.type == "cpu" for value in bundle.values())
    assert all(
        torch.allclose(value @ value.T, torch.eye(2), atol=1e-6) for value in bundle.values()
    )


def test_load_token_spec_accepts_a_list_and_analysis_risen_tokens(tmp_path: Path) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps([" alpha", " beta"]), encoding="utf-8")
    analysis_file = tmp_path / "analysis.json"
    analysis_file.write_text(
        json.dumps(
            {
                "paired_differences": {
                    "arm|frame|position": {
                        "risen": [{"token": " alpha"}, {"token": " beta"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_token_spec(token_file) == [" alpha", " beta"]
    assert load_token_spec(
        analysis_file,
        analysis_key="arm|frame|position",
        top_n=1,
    ) == [" alpha"]


def test_load_token_spec_rejects_mixed_or_missing_analysis_options(tmp_path: Path) -> None:
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({"tokens": [" alpha"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="list"):
        load_token_spec(path)
    with pytest.raises(ValueError, match="analysis_key"):
        load_token_spec(path, top_n=1)


def test_placebo_does_not_depend_on_layer_mapping_order() -> None:
    real = {0: torch.eye(3)[:1], 1: torch.eye(3)[1:2]}
    reversed_real = dict(reversed(list(real.items())))

    first = build_placebo_bundle(real, identity_sha256="c" * 64)
    second = build_placebo_bundle(reversed_real, identity_sha256="c" * 64)

    assert first.keys() == second.keys()
    assert all(torch.equal(first[layer], second[layer]) for layer in first)


def test_effective_token_vectors_refuses_out_of_range_token() -> None:
    lens, unembed = _synthetic_inputs()

    with pytest.raises(ValueError, match="token"):
        effective_token_vectors(lens, unembed, [len(unembed.lm_head_weight)], layers=[0])


def test_build_real_bundle_refuses_empty_band() -> None:
    lens, unembed = _synthetic_inputs()

    with pytest.raises(ValueError, match="band"):
        build_real_bundle(lens, unembed, token_ids=[0], band=(2, 2))


def test_sabotage_orthonormality_guard_would_fail() -> None:
    rows = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    basis, _ = orthonormalize_rows(rows)

    sabotaged = basis.clone()
    sabotaged[1] = sabotaged[0]

    with pytest.raises(AssertionError):
        assert torch.allclose(sabotaged @ sabotaged.T, torch.eye(2), atol=1e-6)
