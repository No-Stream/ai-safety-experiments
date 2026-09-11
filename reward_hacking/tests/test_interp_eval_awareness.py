"""Offline tests for the eval-awareness validation's deliverable-2/4 additions.

CPU only, no model load. The direction/placebo separation and the trained-probe validation run on
synthetic paired activations with a planted class-mean difference, so the empirical p-value and the
clears-null / beats-placebo reads have ground-truth answers. The per-pooling default output dir --
the plumbing that lets the last-pool validation run without clobbering the validated mean-pool axis
-- is checked directly. The last-pool axis is otherwise validated by the same pooling-agnostic core;
only the model-side capture differs, which is the documented GPU step.
"""

from __future__ import annotations

import pytest
import torch

from reward_hacking.interp.eval_awareness_probe import (
    DEFAULT_N_PLACEBOS,
    DEFAULT_OUT_DIR,
    default_out_dir,
    direction_separation,
    validate_layer,
)
from reward_hacking.interp.linear_probe import ConceptActivations, ProbeConfig


def _planted_concept(gap: float, *, n_pairs: int = 20, dims: int = 64) -> ConceptActivations:
    """Positives offset from negatives by ``gap`` along a fixed axis, on shared per-pair noise."""
    generator = torch.Generator().manual_seed(0)
    direction = torch.zeros(dims)
    direction[0] = 1.0
    base = torch.randn(n_pairs, dims, generator=generator)
    jitter = torch.randn(n_pairs, dims, generator=generator) * 0.05
    return ConceptActivations(base + gap * direction, base + jitter)


class TestWidenedPlaceboDefault:
    def test_default_is_at_least_100(self) -> None:
        assert DEFAULT_N_PLACEBOS >= 100


class TestDefaultOutDir:
    def test_mean_keeps_the_canonical_path(self) -> None:
        assert default_out_dir("mean") == DEFAULT_OUT_DIR

    def test_last_gets_its_own_sibling(self) -> None:
        assert default_out_dir("last") == DEFAULT_OUT_DIR.parent / "eval-awareness-last"
        assert default_out_dir("last") != DEFAULT_OUT_DIR


class TestDirectionSeparationPValue:
    def test_separable_axis_beats_placebo_with_tiny_p(self) -> None:
        concept = _planted_concept(gap=6.0)
        config = ProbeConfig(n_folds=5)

        separation = direction_separation(concept, config, n_placebos=50)

        assert separation.direction_accuracy > 0.9
        assert separation.placebo_accuracy_ge_count == 0
        assert separation.accuracy_empirical_p == pytest.approx(1.0 / 51.0)

    def test_p_value_field_flows_into_the_layer_read(self) -> None:
        concept = _planted_concept(gap=6.0)
        config = ProbeConfig(n_folds=5, n_permutations=3)

        read, direction = validate_layer(7, concept, config, n_placebos=20)

        assert read.layer == 7
        assert read.beats_placebo
        assert read.clears_null
        assert 0.0 < read.accuracy_empirical_p <= 1.0
        assert read.accuracy_empirical_p == pytest.approx(1.0 / 21.0)
        assert direction.shape == (64,)
