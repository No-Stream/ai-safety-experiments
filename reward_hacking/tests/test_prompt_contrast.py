"""Offline tests for the conflicting-vs-original prompt-contrast projection.

These run on CPU in seconds: no model loads, no GPU, no benchmark text. The pure-tensor core is
exercised on synthetic activations with a KNOWN planted signal, so every claim the projection makes
has a ground-truth answer to check against:

* projection onto a direction recovers the planted component;
* the AUC, Cohen's d and paired read hit their textbook endpoints;
* a planted conflicting>original shift makes the concept direction separate the groups AND beat the
  matched-norm placebo, while a no-signal draw sits at chance on the AUC band -- but note it may
  still clear ``beats_placebo``, which is a resolution floor rather than a defect and is spelled out
  on :meth:`TestContrastLayer.test_no_signal_sits_at_chance_on_the_auc_band`;
* the paired read raises on misaligned twins.

Per the repo rule that a check never watched fail is not a check, the layer-mismatch guard test
plants the exact violation it exists to catch -- inputs that cover different layers, the shape a
capture that silently dropped a layer would produce -- and confirms it raises rather than quietly
intersecting.
"""

from __future__ import annotations

import pytest
import torch

from reward_hacking.interp.prompt_contrast import (
    CONTRAST_CONCEPTS,
    DEFAULT_POOLINGS,
    ContrastRead,
    cohens_d,
    contrast_all_layers,
    contrast_layer,
    paired_read,
    placebo_stream_seed,
    project,
    roc_auc,
    unit,
)


def _shifted_along(direction: torch.Tensor, base: torch.Tensor, shift: float) -> torch.Tensor:
    """Return ``base`` moved by ``shift`` along the unit of ``direction`` (row-wise)."""
    return base + shift * unit(direction)


class TestProject:
    """Projection unit-normalises the direction, so it reads the component in raw units."""

    def test_projection_is_dot_with_unit_direction(self) -> None:
        features = torch.tensor([[3.0, 0.0], [0.0, 4.0], [3.0, 4.0]])
        direction = torch.tensor([10.0, 0.0])  # unit is [1, 0]
        assert project(features, direction).tolist() == [3.0, 0.0, 3.0]

    def test_projection_is_scale_invariant_in_direction(self) -> None:
        features = torch.randn(5, 8, generator=torch.Generator().manual_seed(0))
        direction = torch.randn(8, generator=torch.Generator().manual_seed(1))
        small = project(features, direction)
        large = project(features, direction * 1000.0)
        assert torch.allclose(small, large, atol=1e-4)

    def test_zero_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="zero-norm"):
            unit(torch.zeros(4))


class TestSeparationStatistics:
    """AUC, Cohen's d and the paired read have the textbook endpoints, so the reads are trusted."""

    def test_auc_fully_separated_is_one(self) -> None:
        assert roc_auc(torch.tensor([5.0, 6.0, 7.0]), torch.tensor([1.0, 2.0, 3.0])) == 1.0

    def test_auc_reversed_is_zero(self) -> None:
        assert roc_auc(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([5.0, 6.0, 7.0])) == 0.0

    def test_auc_identical_groups_is_half(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert roc_auc(values, values) == pytest.approx(0.5)

    def test_cohens_d_sign_and_zero(self) -> None:
        generator = torch.Generator().manual_seed(2)
        high = torch.randn(50, generator=generator) + 3.0
        low = torch.randn(50, generator=generator)
        assert cohens_d(high, low) > 1.0
        assert cohens_d(low, high) < -1.0
        assert cohens_d(torch.ones(5), torch.ones(5)) == 0.0

    def test_paired_read_recovers_planted_difference(self) -> None:
        generator = torch.Generator().manual_seed(9)
        original = torch.randn(40, generator=generator)
        # Positive shift plus small noise so differences vary (a constant shift has an undefined t).
        conflicting = original + 0.5 + 0.05 * torch.randn(40, generator=generator)
        read = paired_read(conflicting, original)
        assert read.mean_diff == pytest.approx(0.5, abs=0.05)
        assert read.sign_rate == 1.0
        assert read.t_stat > 0.0

    def test_paired_read_no_difference_is_flat(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        read = paired_read(values, values)
        assert read.mean_diff == pytest.approx(0.0)
        assert read.sign_rate == 0.0
        assert read.t_stat == 0.0

    def test_paired_read_misaligned_raises(self) -> None:
        with pytest.raises(ValueError, match="aligned twins"):
            paired_read(torch.zeros(4), torch.zeros(5))


class TestContrastLayer:
    """A planted conflicting>original shift separates and beats placebo; a null draw does not."""

    def test_planted_signal_separates_and_beats_placebo(self) -> None:
        generator = torch.Generator().manual_seed(3)
        d, n = 64, 60
        direction = torch.randn(d, generator=generator)
        base_conflicting = torch.randn(n, d, generator=generator)
        base_original = torch.randn(n, d, generator=generator)
        # Push the conflicting group along the concept direction; leave original where it is.
        conflicting = _shifted_along(direction, base_conflicting, shift=3.0)
        original = base_original

        read = contrast_layer(
            "planted",
            "mean",
            7,
            conflicting,
            original,
            direction,
            n_placebos=5,
            generator=torch.Generator().manual_seed(0),
        )
        assert isinstance(read, ContrastRead)
        assert read.auc > 0.9
        assert read.paired_sign_rate > 0.9
        assert read.mean_diff > 0.0
        # The random placebo of equal norm sees no planted signal: chance, and clearly below real.
        assert read.placebo_auc_mean == pytest.approx(0.5, abs=0.15)
        assert read.beats_placebo
        assert read.auc_above_placebo > 0.3

    def test_no_signal_sits_at_chance_on_the_auc_band(self) -> None:
        r"""A null draw lands at chance on the AUC band -- and on THIS draw it still beats placebo.

        Named ``test_no_signal_does_not_beat_placebo`` until 2026-08-24, which this fixture does not
        show. Measured on the seeds below: ``auc=0.5569``, ``placebo_auc_mean=0.4679``,
        ``placebo_auc_max=0.4938``, so ``beats_placebo`` is **True**, with
        ``auc_above_placebo=+0.0890`` and ``auc_empirical_p=0.1111``. The three assertions below are
        the AUC band and the placebo *mean*; **none of them reads the placebo flag**, which is why the
        old name went unnoticed.

        **Do not "fix" this by adding ``assert not read.beats_placebo``.** It fails about one run in
        nine, and that is the criterion working as defined, not a regression:
        ``beats_placebo`` is ``auc > placebo_auc_max`` (:mod:`reward_hacking.interp.prompt_contrast`),
        i.e. clearing the *best* of the draws, so with ``n`` placebos the smallest one-sided
        false-positive rate it can express is ``1/(n+1)``. At ``n_placebos=8`` that floor is 1 in 9,
        about 11%, which is exactly what was measured.

        **Open question for the owner, deliberately not decided here:** a nominal 5% needs ``n >= 19``
        draws and a nominal 1% needs ``n >= 99``. That is a compute-versus-sensitivity tradeoff on one
        of the two controls this repo refuses to skip, so the criterion is left alone rather than
        quietly redefined. Ledger: ``docs/scratch/test-prose-contradiction-ledger-2026-08-24.md``, C1.
        """
        generator = torch.Generator().manual_seed(4)
        d, n = 64, 80
        direction = torch.randn(d, generator=generator)
        # Both groups drawn from the same distribution: no conflicting-vs-original signal anywhere.
        conflicting = torch.randn(n, d, generator=generator)
        original = torch.randn(n, d, generator=generator)

        read = contrast_layer(
            "null",
            "mean",
            0,
            conflicting,
            original,
            direction,
            n_placebos=8,
            generator=torch.Generator().manual_seed(0),
        )
        # No planted signal: the real axis separates no better than the placebo MEAN, both chance.
        # The placebo max is a different question -- see the docstring.
        assert read.auc == pytest.approx(0.5, abs=0.15)
        assert read.placebo_auc_mean == pytest.approx(0.5, abs=0.15)
        assert abs(read.auc_above_placebo) < 0.15

    def test_signal_orthogonal_to_direction_is_invisible(self) -> None:
        """A shift on an axis the direction does not span leaves the projection at chance.

        The complement of the planted-signal test: separation is direction-specific, not any old
        difference between the groups. A concept axis reads only the component along itself.
        """
        d, n = 32, 60
        direction = torch.zeros(d)
        direction[0] = 1.0
        generator = torch.Generator().manual_seed(5)
        base = torch.randn(n, d, generator=generator)
        conflicting = base.clone()
        conflicting[:, 1] += 5.0  # big shift, but on axis 1, orthogonal to the direction (axis 0)
        original = torch.randn(n, d, generator=generator)

        read = contrast_layer(
            "orthogonal",
            "mean",
            3,
            conflicting,
            original,
            direction,
            n_placebos=5,
            generator=torch.Generator().manual_seed(0),
        )
        assert read.auc == pytest.approx(0.5, abs=0.15)


class TestContrastAllLayers:
    """Assembly over layers keeps one read per layer and guards a silent layer drop."""

    def _planted_layers(
        self, layers: list[int], shift: float
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        generator = torch.Generator().manual_seed(6)
        d, n = 48, 40
        conflicting: dict[int, torch.Tensor] = {}
        original: dict[int, torch.Tensor] = {}
        directions: dict[int, torch.Tensor] = {}
        for layer in layers:
            direction = torch.randn(d, generator=generator)
            base = torch.randn(n, d, generator=generator)
            conflicting[layer] = _shifted_along(direction, base, shift=shift)
            original[layer] = torch.randn(n, d, generator=generator)
            directions[layer] = direction
        return conflicting, original, directions

    def test_one_read_per_layer(self) -> None:
        layers = [0, 1, 2, 3]
        conflicting, original, directions = self._planted_layers(layers, shift=2.5)
        reads = contrast_all_layers(
            "planted", "mean", conflicting, original, directions, n_placebos=3, seed=0
        )
        assert [read.layer for read in reads] == layers
        assert all(read.beats_placebo for read in reads)

    def test_layer_mismatch_raises(self) -> None:
        """SABOTAGE (silent-drop shape): directions cover fewer layers than the activations.

        A loop over the activation layers would silently skip the missing direction or KeyError
        deep inside; the top-level guard is what turns it into a clear, early failure. Watched to
        raise rather than measure the wrong thing."""
        conflicting, original, directions = self._planted_layers([0, 1, 2], shift=2.0)
        del directions[2]  # a capture/loaded-direction that dropped a layer
        with pytest.raises(ValueError, match="different layers"):
            contrast_all_layers(
                "planted", "mean", conflicting, original, directions, n_placebos=2, seed=0
            )


class TestPlaceboStreamIndependence:
    """The matched-norm null must be an independent draw per (concept, pooling), not one shared one.

    The bug this pins (inspection addenda N1, 2026-08-24): the run harness calls
    ``contrast_all_layers`` once per (variant, pooling, concept) with the same base seed, and a
    generator built from the bare seed handed all four concepts the IDENTICAL 100 random directions
    -- measured max cross-concept difference in placebo stats 9e-6 -- so "several concepts cleared
    their placebo band" carried one draw's worth of independence. These tests use the same
    activations and the same direction grid and vary only the label, which is exactly the shape of
    the harness's four-concept sweep over one pooled capture.
    """

    def _grid(
        self,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        generator = torch.Generator().manual_seed(11)
        d, n = 48, 40
        layers = [0, 1, 2]
        conflicting = {layer: torch.randn(n, d, generator=generator) for layer in layers}
        original = {layer: torch.randn(n, d, generator=generator) for layer in layers}
        directions = {layer: torch.randn(d, generator=generator) for layer in layers}
        return conflicting, original, directions

    @staticmethod
    def _placebo_draws(reads: list[ContrastRead]) -> torch.Tensor:
        return torch.tensor([auc for read in reads for auc in read.placebo_aucs])

    def test_two_concepts_draw_independent_placebo_directions(self) -> None:
        """Same activations, same axes, same base seed, different concept: the null must differ.

        Under the shared-stream bug the two calls used bit-identical placebo directions, so the
        draws below agree to float noise (<1e-5); independent draws on 40-row groups differ by
        whole percentage points of AUC. Watched to fail under the exact sabotage (rebuilding the
        generator from the bare seed) before being trusted.
        """
        conflicting, original, directions = self._grid()
        reads_a = contrast_all_layers(
            "eval_awareness", "mean", conflicting, original, directions, n_placebos=8, seed=0
        )
        reads_b = contrast_all_layers(
            "deception", "mean", conflicting, original, directions, n_placebos=8, seed=0
        )

        gap = (self._placebo_draws(reads_a) - self._placebo_draws(reads_b)).abs().max().item()
        assert gap > 1e-3, f"placebo draws are shared across concepts (max AUC difference {gap})"

    def test_two_poolings_draw_independent_placebo_streams(self) -> None:
        conflicting, original, directions = self._grid()
        reads_mean = contrast_all_layers(
            "shortcut", "mean", conflicting, original, directions, n_placebos=8, seed=0
        )
        reads_last = contrast_all_layers(
            "shortcut", "last", conflicting, original, directions, n_placebos=8, seed=0
        )

        gap = (self._placebo_draws(reads_mean) - self._placebo_draws(reads_last)).abs().max().item()
        assert gap > 1e-3, f"placebo draws are shared across poolings (max AUC difference {gap})"

    def test_the_stream_is_reproducible_for_a_fixed_cell(self) -> None:
        """Sweep-shape independence: the same cell gets the same draws in any run of any shape."""
        conflicting, original, directions = self._grid()
        first = contrast_all_layers(
            "shortcut", "mean", conflicting, original, directions, n_placebos=8, seed=0
        )
        second = contrast_all_layers(
            "shortcut", "mean", conflicting, original, directions, n_placebos=8, seed=0
        )

        for read_a, read_b in zip(first, second, strict=True):
            assert read_a.placebo_aucs == read_b.placebo_aucs

    def test_derived_seeds_are_distinct_across_the_real_sweep_cells(self) -> None:
        """Every (concept, pooling) cell of the actual harness sweep gets its own seed."""
        seeds = {
            placebo_stream_seed(0, concept, pooling)
            for concept in CONTRAST_CONCEPTS
            for pooling in DEFAULT_POOLINGS
        }

        assert len(seeds) == len(CONTRAST_CONCEPTS) * len(DEFAULT_POOLINGS)

    def test_the_base_seed_still_moves_every_cell(self) -> None:
        """The run-level seed knob keeps meaning something: a different base seed shifts each cell."""
        assert placebo_stream_seed(0, "shortcut", "mean") != placebo_stream_seed(
            1, "shortcut", "mean"
        )
