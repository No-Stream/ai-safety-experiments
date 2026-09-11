"""Offline tests for the trained linear-probe interp read.

CPU only, no model load, no GPU, no network: every test runs on synthetic pooled activations with a
KNOWN planted class-mean difference, so each claim the probe makes has a ground-truth answer.

What is checked, and why each one is the check that could actually fail:

* the probe recovers a planted separable direction and its cross-validated accuracy is high;
* the shuffled-label null sits at chance on the SAME data the real probe scores ~1.0 on;
* grouped folds keep both members of a matched pair on the same side of the split, and the leakage
  that buys is demonstrated rather than asserted: ``test_broken_pair_grouping_inflates_accuracy``
  breaks the grouping on the cross-concept task and watches accuracy jump to 1.0. That test also
  asserts that the broken grouping really did split twins, because the first version of it silently
  did not -- with 20 pairs and 5 folds, giving every sample its own group id still lands both twins
  in fold ``index % 5``, so the sabotage passed while proving nothing;
* the probe weight vector's cosine to the diff-of-means direction is computed and lands high on
  isotropic synthetic data, where the two estimators provably agree;
* the L2 sweep shows the documented artifact -- stronger regularisation drives the probe direction
  toward diff-of-means -- so the caveat in the module docstring is a measured property, not a story.

Null assertions are one-sided (``null < NULL_CEILING``) rather than a band around 0.5. Above chance
is the direction that means leakage; below chance is ordinary variance at these sample sizes, and
single-permutation draws scatter a long way down -- measured over 25 permutations on planted data at
n=40, the null averaged 0.486 with a range of 0.225 to 0.650.
"""

from __future__ import annotations

import pytest
import torch

from reward_hacking.interp.linear_probe import (
    ConceptActivations,
    LayerContext,
    ProbeConfig,
    _concept_reads,  # pyright: ignore[reportPrivateUsage]
    _cross_concept_task,  # pyright: ignore[reportPrivateUsage]
    _pair_reads,  # pyright: ignore[reportPrivateUsage]
    _split_half_cosine,  # pyright: ignore[reportPrivateUsage]
    cross_validated_accuracy,
    fit_logistic_probe,
    grouped_test_masks,
    permuted_label_accuracies,
    probe_layer,
    standardizing_stats,
)

N_PAIRS = 20
N_DIMS = 64
NULL_CEILING = 0.7  # a null above this is leakage; see the module docstring on why it is one-sided


def _planted_concept(
    direction: torch.Tensor, generator: torch.Generator, *, n_pairs: int = N_PAIRS, gap: float = 4.0
) -> ConceptActivations:
    """Positives offset from negatives by ``gap * direction``, on shared per-pair Gaussian noise.

    The shared per-pair base is what makes the pair a real group: both members of pair ``i`` sit on
    the same random point, so the only systematic difference between the classes is ``direction``.
    """
    dims = direction.shape[0]
    base = torch.randn(n_pairs, dims, generator=generator)
    jitter = torch.randn(n_pairs, dims, generator=generator) * 0.1
    return ConceptActivations(base + gap * direction, base + jitter)


def _unit(dims: int, generator: torch.Generator) -> torch.Tensor:
    raw = torch.randn(dims, generator=generator)
    return raw / raw.norm()


def _stem_only_concept(seed: int, *, n_pairs: int = N_PAIRS) -> ConceptActivations:
    """Pairs that share a strong per-pair stem and carry NO systematic positive/negative difference.

    Each pair is recognisable by its stem and nothing about the concept generalises across pairs,
    which is what makes it the right stimulus for a grouping-leak sabotage.
    """
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(n_pairs, N_DIMS, generator=generator)
    jitter = 0.1
    return ConceptActivations(
        base + jitter * torch.randn(n_pairs, N_DIMS, generator=generator),
        base + jitter * torch.randn(n_pairs, N_DIMS, generator=generator),
    )


class TestLogisticFit:
    def test_recovers_a_separable_direction(self) -> None:
        generator = torch.Generator().manual_seed(0)
        planted = _unit(N_DIMS, generator)
        concept = _planted_concept(planted, generator)
        features, labels, _ = concept.polarity_task()

        fit = fit_logistic_probe(features, labels, l2_strength=1e-2)

        assert torch.nn.functional.cosine_similarity(fit.weights, planted, dim=0) > 0.8
        assert (fit.predict(features) == labels).all()

    def test_is_deterministic(self) -> None:
        generator = torch.Generator().manual_seed(1)
        concept = _planted_concept(_unit(N_DIMS, generator), generator)
        features, labels, _ = concept.polarity_task()

        first = fit_logistic_probe(features, labels, l2_strength=1e-2)
        second = fit_logistic_probe(features, labels, l2_strength=1e-2)

        assert torch.equal(first.weights, second.weights)

    def test_stronger_regularization_shrinks_the_weights(self) -> None:
        generator = torch.Generator().manual_seed(2)
        concept = _planted_concept(_unit(N_DIMS, generator), generator)
        features, labels, _ = concept.polarity_task()

        norms = [
            fit_logistic_probe(features, labels, l2_strength=l2).weights.norm().item()
            for l2 in (1e-4, 1e-2, 1.0)
        ]

        assert norms[0] > norms[1] > norms[2]

    def test_rejects_mismatched_row_counts(self) -> None:
        with pytest.raises(ValueError, match="rows but labels"):
            fit_logistic_probe(torch.zeros(4, 3), torch.zeros(5), l2_strength=1e-2)


class TestGroupedFolds:
    def test_pair_members_share_a_fold(self) -> None:
        _, _, groups = _planted_concept(
            _unit(N_DIMS, torch.Generator().manual_seed(3)), torch.Generator().manual_seed(3)
        ).polarity_task()

        masks = grouped_test_masks(groups, 5)

        for mask in masks:
            held_out = groups[mask]
            for group in held_out.unique():
                # Every sample of a held-out group must be held out, never split across the fold.
                assert bool((groups[mask] == group).sum() == (groups == group).sum())

    def test_folds_partition_every_sample_exactly_once(self) -> None:
        groups = torch.arange(N_PAIRS).repeat(2)

        masks = grouped_test_masks(groups, 5)

        assert torch.stack(masks).sum(dim=0).eq(1).all()

    def test_too_few_groups_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot fill"):
            grouped_test_masks(torch.tensor([0, 0, 1, 1]), 5)


class TestCrossValidationAndNull:
    def test_planted_direction_is_decodable_and_its_null_is_at_chance(self) -> None:
        generator = torch.Generator().manual_seed(4)
        concept = _planted_concept(_unit(N_DIMS, generator), generator)
        features, labels, groups = concept.polarity_task()
        config = ProbeConfig(n_permutations=5)

        accuracy = cross_validated_accuracy(features, labels, groups, config)
        nulls = permuted_label_accuracies(features, labels, groups, config)

        assert accuracy > 0.9
        assert sum(nulls) / len(nulls) < NULL_CEILING
        assert max(nulls) < accuracy

    def test_shuffled_labels_destroy_a_perfectly_decodable_signal(self) -> None:
        """The null's teeth: on data the probe fits perfectly, permuting labels must break it.

        A null that cannot fail is not a control. Here the same features and the same folds are
        scored with a permuted label vector, and every permutation must land below the real
        accuracy.
        """
        generator = torch.Generator().manual_seed(5)
        concept = _planted_concept(_unit(N_DIMS, generator), generator, gap=8.0)
        features, labels, groups = concept.polarity_task()
        config = ProbeConfig(n_permutations=5)

        assert cross_validated_accuracy(features, labels, groups, config) == 1.0
        assert max(permuted_label_accuracies(features, labels, groups, config)) < NULL_CEILING

    def test_broken_pair_grouping_inflates_accuracy(self) -> None:
        """Sabotage the fold grouping on the cross-concept task and watch accuracy jump to 1.0.

        The stimuli carry no systematic concept difference -- two sets of pair-specific "stems"
        drawn from one distribution -- so nothing about the *concept* generalises. Both members of
        a pair carry the same concept label here, which is what makes a split pair a real leak:
        train on one twin, and the near-identical held-out twin is recognisable by stem alone. (In
        the within-concept polarity task the two members carry OPPOSITE labels, so a split pair
        biases accuracy DOWN instead; grouping is hygiene there, leak prevention here.)

        The twins-split assertion is load-bearing. The first version of this test used five folds
        and 20 pairs, where giving every sample its own group id still puts both twins in fold
        ``index % 5`` -- the sabotage was a silent no-op and the test passed while proving nothing.
        Three folds actually splits them, and the assertion now says so out loud.
        """
        first, second = (_stem_only_concept(seed) for seed in (11, 12))
        features, labels, groups = _cross_concept_task(first, second)
        by_sample = torch.arange(features.shape[0])
        config = ProbeConfig(n_folds=3)

        fold_of = torch.zeros(features.shape[0], dtype=torch.long)
        for fold, mask in enumerate(grouped_test_masks(by_sample, config.n_folds)):
            fold_of[mask] = fold
        twins_split = (fold_of[:N_PAIRS] != fold_of[N_PAIRS : 2 * N_PAIRS]).sum().item()

        honest = cross_validated_accuracy(features, labels, groups, config)
        leaked = cross_validated_accuracy(features, labels, by_sample, config)

        assert twins_split == N_PAIRS
        assert leaked == 1.0
        assert leaked > honest + 0.2


class TestSplitHalfStability:
    def test_a_strong_planted_direction_is_stable_and_noise_is_not(self) -> None:
        generator = torch.Generator().manual_seed(6)
        planted = _planted_concept(_unit(N_DIMS, generator), generator, gap=3.0)
        noise = ConceptActivations(
            torch.randn(N_PAIRS, N_DIMS, generator=generator),
            torch.randn(N_PAIRS, N_DIMS, generator=generator),
        )
        config = ProbeConfig()

        planted_probe_cos, planted_mean_cos = _split_half_cosine(planted, config)
        noise_probe_cos, noise_mean_cos = _split_half_cosine(noise, config)

        assert planted_probe_cos > 0.7
        assert planted_mean_cos > 0.7
        assert abs(noise_probe_cos) < 0.5
        assert abs(noise_mean_cos) < 0.5


class TestCrossConceptTask:
    def test_labels_and_groups_keep_pairs_intact(self) -> None:
        generator = torch.Generator().manual_seed(7)
        first = _planted_concept(_unit(N_DIMS, generator), generator, n_pairs=6)
        second = _planted_concept(_unit(N_DIMS, generator), generator, n_pairs=4)

        features, labels, groups = _cross_concept_task(first, second)

        assert features.shape == (20, N_DIMS)
        assert labels.sum().item() == 12
        assert groups.unique().numel() == 10
        for group in groups.unique():
            # A group carries one concept label, so grouped folds cannot split a pair's stem.
            assert labels[groups == group].unique().numel() == 1


class TestLayerAssembly:
    @staticmethod
    def _two_concepts(cosine_between: float) -> dict[str, ConceptActivations]:
        """Two concepts whose planted directions have a known cosine between them."""
        generator = torch.Generator().manual_seed(8)
        first = _unit(N_DIMS, generator)
        orthogonal = _unit(N_DIMS, generator)
        orthogonal = orthogonal - first * (orthogonal @ first)
        orthogonal = orthogonal / orthogonal.norm()
        second = cosine_between * first + (1 - cosine_between**2) ** 0.5 * orthogonal
        return {
            "shortcut": _planted_concept(first, generator),
            "deception": _planted_concept(second, generator),
        }

    def test_probe_layer_reports_every_concept_and_pair(self) -> None:
        reads = probe_layer(
            7, self._two_concepts(0.3), ProbeConfig(n_permutations=2), l2_values=(1e-2, 1.0)
        )

        assert [read.concept for read in reads.concepts] == ["shortcut", "deception"]
        assert [(read.concept_a, read.concept_b) for read in reads.pairs] == [
            ("shortcut", "deception")
        ]
        assert {read.layer for read in reads.concepts} == {7}
        assert len(reads.sweep_concepts) == 4  # 2 concepts x 2 L2 strengths
        assert len(reads.sweep_pairs) == 2  # 1 pair x 2 L2 strengths

    def test_probe_direction_agrees_with_diff_of_means_on_isotropic_data(self) -> None:
        activations = self._two_concepts(0.3)
        config = ProbeConfig(n_permutations=1)

        reads = probe_layer(0, activations, config, l2_values=(1e-2,))

        for read in reads.concepts:
            assert read.accuracy > 0.9
            assert read.null_accuracy_mean < NULL_CEILING
            assert read.cos_probe_diff_of_means > 0.8
            assert read.cos_probe_diff_of_means_raw > 0.8

    def test_probe_pair_cosine_tracks_the_planted_angle(self) -> None:
        config = ProbeConfig(n_permutations=1)

        near_orthogonal = probe_layer(0, self._two_concepts(0.0), config, l2_values=(1e-2,))
        aligned = probe_layer(0, self._two_concepts(0.9), config, l2_values=(1e-2,))

        assert abs(near_orthogonal.pairs[0].cos_probe_a_probe_b) < 0.35
        assert aligned.pairs[0].cos_probe_a_probe_b > 0.7

    def test_regularization_sweep_pulls_the_probe_toward_diff_of_means(self) -> None:
        """The documented artifact, measured: cos(probe, diff-of-means) rises with the L2 strength.

        Guards the module's central caveat. If this ordering ever reversed, the sweep would not be
        doing the job the docstring claims for it.
        """
        reads = probe_layer(
            0, self._two_concepts(0.3), ProbeConfig(n_permutations=1), l2_values=(1e-4, 1.0)
        )

        by_concept: dict[str, dict[float, float]] = {}
        for row in reads.sweep_concepts:
            by_concept.setdefault(row.concept, {})[row.l2_strength] = row.cos_probe_diff_of_means
        for cosines in by_concept.values():
            assert cosines[1.0] > cosines[1e-4]

    def test_mismatched_pair_counts_raise(self) -> None:
        with pytest.raises(ValueError, match="pair-aligned"):
            ConceptActivations(torch.zeros(4, N_DIMS), torch.zeros(3, N_DIMS))


class TestStandardizingStats:
    def test_stats_cover_every_sentence_of_every_concept(self) -> None:
        activations = {
            "shortcut": ConceptActivations(torch.zeros(3, 4), torch.zeros(3, 4)),
            "deception": ConceptActivations(torch.full((3, 4), 4.0), torch.zeros(3, 4)),
        }

        mean, scale = standardizing_stats(activations)

        assert torch.allclose(mean, torch.ones(4))  # HARNESS-SCAN-EXEMPT-tensor-device (CPU test)
        assert (scale > 1.0).all()

    def test_centering_leaves_the_probe_direction_unchanged(self) -> None:
        """Logistic regression with an intercept is invariant to translating the features.

        The reason the module can center by a global mean without touching any reported number.
        """
        generator = torch.Generator().manual_seed(9)
        concept = _planted_concept(_unit(N_DIMS, generator), generator)
        features, labels, _ = concept.polarity_task()
        offset = torch.randn(N_DIMS, generator=generator) * 5.0

        original = fit_logistic_probe(features, labels, l2_strength=1e-2).weights
        shifted = fit_logistic_probe(features + offset, labels, l2_strength=1e-2).weights

        assert torch.nn.functional.cosine_similarity(original, shifted, dim=0) > 0.999


class TestReadsAreScalarJson:
    def test_every_reported_field_is_a_plain_scalar(self) -> None:
        """Artifacts are scalar JSON; a tensor leaking into a read would break the dump."""
        activations = TestLayerAssembly._two_concepts(0.3)
        mean, scale = standardizing_stats(activations)
        context = LayerContext(
            layer=0,
            standardized={
                name: concept.standardized(mean, scale) for name, concept in activations.items()
            },
            raw=activations,
            probes={name: torch.randn(N_DIMS) for name in activations},
            scale=scale,
        )
        config = ProbeConfig(n_permutations=1)

        reads = [*_concept_reads(context, config), *_pair_reads(context, config)]

        for read in reads:
            for value in vars(read).values():
                assert isinstance(value, int | float | str)
