"""Offline tests for the SAE analysis pure-math cores (no model, no network).

Covers the JumpReLU encode against a hand-computed example (including threshold gating), the
config/shape validators, the differential-firing permutation null on a planted signal and on a
null, and the concept-direction cosine alignment with its random-direction control.
"""

import pytest
import torch

from reward_hacking.interp.sae_analysis import (
    JumpReluSae,
    _validate_sae_shapes,
    concept_alignment,
    differential_firing,
    parse_layer_from_hook_name,
)


def _sae(
    w_enc: torch.Tensor,
    b_enc: torch.Tensor,
    b_dec: torch.Tensor,
    threshold: torch.Tensor,
    *,
    w_dec: torch.Tensor | None = None,
) -> JumpReluSae:
    d_in, d_sae = w_enc.shape
    if w_dec is None:
        w_dec = torch.zeros(d_sae, d_in)
    return JumpReluSae(
        W_enc=w_enc,
        W_dec=w_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        threshold=threshold,
        layer=15,
        d_in=d_in,
        d_sae=d_sae,
        sae_id="test/sae",
        hook_name="model.layers.15",
    )


class TestEncode:
    def test_matches_hand_computed_with_threshold_gating(self):
        """d_in=2, d_sae=3. sae_in = x - b_dec = [1,1]; pre = sae_in @ W_enc + b_enc = [1.5,0.5,0.0].

        relu(pre) = [1.5, 0.5, 0.0]; gate (pre > threshold) with threshold [0, 5, -10] = [T, F, T].
        feats = [1.5, 0.0, 0.0]: feature 1 is gated OFF by its threshold even though relu(pre) > 0.
        """
        sae = _sae(
            w_enc=torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]),
            b_enc=torch.tensor([0.5, -0.5, 0.0]),
            b_dec=torch.tensor([1.0, 2.0]),
            threshold=torch.tensor([0.0, 5.0, -10.0]),
        )
        feats = sae.encode(torch.tensor([[2.0, 3.0]]))
        assert torch.allclose(feats, torch.tensor([[1.5, 0.0, 0.0]]), atol=1e-6)

    def test_wrong_last_dim_raises(self):
        sae = _sae(
            w_enc=torch.zeros(2, 3),
            b_enc=torch.zeros(3),
            b_dec=torch.zeros(2),
            threshold=torch.zeros(3),
        )
        with pytest.raises(ValueError, match="expected last dim 2"):
            sae.encode(torch.zeros(1, 4))

    def test_encode_upcasts_from_bf16_input(self):
        sae = _sae(
            w_enc=torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]),
            b_enc=torch.tensor([0.5, -0.5, 0.0]),
            b_dec=torch.tensor([1.0, 2.0]),
            threshold=torch.tensor([0.0, 5.0, -10.0]),
        )
        feats = sae.encode(torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16))
        assert feats.dtype == torch.float32
        assert torch.allclose(feats, torch.tensor([[1.5, 0.0, 0.0]]), atol=1e-2)


class TestValidators:
    def test_parse_layer_from_hook_name(self):
        assert parse_layer_from_hook_name("model.layers.15") == 15
        assert parse_layer_from_hook_name("model.layers.0") == 0

    def test_parse_layer_rejects_non_layer_hook(self):
        with pytest.raises(ValueError, match="not an integer"):
            parse_layer_from_hook_name("model.embed_tokens")

    def test_validate_shapes_rejects_transposed_w_enc(self):
        """W_enc is [3, 2] but the SAE declares d_in=2, d_sae=3, so the shapes disagree."""
        bad = JumpReluSae(
            W_enc=torch.zeros(3, 2),
            W_dec=torch.zeros(3, 2),
            b_enc=torch.zeros(3),
            b_dec=torch.zeros(2),
            threshold=torch.zeros(3),
            layer=15,
            d_in=2,
            d_sae=3,
            sae_id="test/bad",
            hook_name="model.layers.15",
        )
        with pytest.raises(ValueError, match="weight shapes disagree"):
            _validate_sae_shapes(bad)


class TestDifferentialFiring:
    def test_planted_feature_ranks_first_and_is_significant(self):
        n_pairs, d_sae = 8, 5
        base = torch.arange(n_pairs * d_sae, dtype=torch.float32).reshape(n_pairs, d_sae)
        matched = base.clone()
        rigged = base.clone()
        rigged[:, 0] += 10.0  # feature 0 fires 10 higher on rigged in every pair
        out = differential_firing(rigged, matched, n_permutations=500, seed=0, top_k=5)

        top = out.top_rigged_gt_matched
        assert top[0].feature == 0
        assert top[0].mean_delta == pytest.approx(10.0)
        assert top[0].significant_fwer is True
        assert top[0].p_two_sided < 0.05
        # Only the planted feature should clear the family-wise threshold.
        assert out.n_features_significant_fwer == 1
        # Every other feature has a zero paired difference.
        for row in top[1:]:
            assert row.mean_delta == pytest.approx(0.0)

    def test_null_yields_no_significant_features(self):
        n_pairs, d_sae = 8, 5
        both = torch.arange(n_pairs * d_sae, dtype=torch.float32).reshape(n_pairs, d_sae)
        out = differential_firing(both.clone(), both.clone(), n_permutations=500, seed=0, top_k=5)
        assert out.n_features_significant_fwer == 0
        assert out.fwer_threshold == pytest.approx(0.0)
        assert out.max_abs_observed_delta == pytest.approx(0.0)

    def test_p_values_in_unit_interval(self):
        torch.manual_seed(1)
        rigged = torch.randn(6, 7)
        matched = torch.randn(6, 7)
        out = differential_firing(rigged, matched, n_permutations=200, seed=0, top_k=7)
        for row in out.top_rigged_gt_matched + out.top_matched_gt_rigged:
            assert 0.0 <= row.p_two_sided <= 1.0

    def test_rejects_single_pair(self):
        with pytest.raises(ValueError, match="at least 2 pairs"):
            differential_firing(
                torch.zeros(1, 3), torch.zeros(1, 3), n_permutations=10, seed=0, top_k=3
            )

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="differ"):
            differential_firing(
                torch.zeros(3, 4), torch.zeros(3, 5), n_permutations=10, seed=0, top_k=3
            )

    def test_rejects_non_finite(self):
        rigged = torch.zeros(3, 4)
        rigged[0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            differential_firing(rigged, torch.zeros(3, 4), n_permutations=10, seed=0, top_k=4)


class TestConceptAlignment:
    def test_decoder_row_equal_to_direction_aligns_at_cosine_one(self):
        d_sae, d_in = 4, 3
        direction = torch.tensor([1.0, 2.0, -1.0])
        w_dec = torch.zeros(d_sae, d_in)
        w_dec[0] = direction * 3.0  # same direction, different magnitude -> cosine still 1
        w_dec[1] = -direction  # opposite -> the top anti-aligned feature
        w_dec[2] = torch.tensor([2.0, -1.0, 0.0])  # orthogonal to `direction`
        w_dec[3] = torch.tensor([0.0, 0.0, 1.0])

        out = concept_alignment(w_dec, {"concept": direction}, top_k=4, n_placebos=8, seed=0)
        per = out.per_concept["concept"]
        assert per.top_aligned[0].feature == 0
        assert per.top_aligned[0].cosine == pytest.approx(1.0, abs=1e-5)
        assert per.top_anti_aligned[0].feature == 1
        assert per.top_anti_aligned[0].cosine == pytest.approx(-1.0, abs=1e-5)
        assert per.max_abs_cosine == pytest.approx(1.0, abs=1e-5)

    def test_placebo_control_reports_random_direction_baseline(self):
        d_sae, d_in = 16, 5
        w_dec = torch.randn(d_sae, d_in, generator=torch.Generator().manual_seed(3))
        direction = torch.randn(d_in, generator=torch.Generator().manual_seed(4))
        out = concept_alignment(w_dec, {"c": direction}, top_k=4, n_placebos=32, seed=0)
        placebo = out.placebo_control
        assert placebo.n_placebos == 32
        assert placebo.random_direction_best_abs_cosine_mean is not None
        assert 0.0 <= placebo.random_direction_best_abs_cosine_mean <= 1.0
        assert placebo.random_direction_best_abs_cosine_max is not None
        assert 0.0 <= placebo.random_direction_best_abs_cosine_max <= 1.0

    def test_wrong_direction_dim_raises(self):
        w_dec = torch.zeros(4, 3)
        with pytest.raises(ValueError, match="expected \\(3,\\)"):
            concept_alignment(w_dec, {"c": torch.zeros(4)}, top_k=4, n_placebos=1, seed=0)
