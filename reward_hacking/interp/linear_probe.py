"""Trained linear probes over pooled residual activations, with the controls that make them read.

A stronger separability instrument than a diff-of-means direction, plus a shuffled-label null and a
regularisation sweep without which its headline cosine cannot be interpreted.

``directions.py`` asks a geometric question -- how close is the shortcut axis to the deception axis
when each is estimated as ``mean(positive) - mean(negative)``. This module asks a *decoding*
question with a trained classifier: fit L2-regularised logistic regression on the pooled residual
stream and see how well held-out sentences can be labelled. Three things come out of one fit:

* **Cross-validated accuracy** per concept per layer -- is the concept linearly decodable, and at
  what depth. Grouped 5-fold with the *matched pair* as the group, so a pair's positive and its foil
  are never split across train and test. Which way that cuts depends on the task, and the tests
  measure both: on the cross-concept task both members carry the same label, so a split pair is a
  genuine optimistic leak (train on one twin, recognise the near-identical other by its stem --
  worth 0.73 -> 1.00 on synthetic stem-only stimuli). On the within-concept task the members carry
  opposite labels, so a split pair biases accuracy *down* instead; grouping is hygiene there rather
  than leak prevention.
* **A shuffled-label null** -- the same pipeline with permuted labels. A probe never watched fail
  is not a check: at n=60 in 2560 dimensions a linear classifier fits anything on the training
  fold, so the null is what says the held-out number means something.
* **The learned weight vector as a direction**, compared by cosine to the diff-of-means direction
  for the same concept, and (the number that actually matters) to the *other* concept's probe
  direction -- the trained analog of ``directions.py``'s ``cos_hack_deception``.

Two caveats are built in as measurements rather than left in prose, because both can turn a
confident number into an artifact:

**The probe-vs-diff-of-means cosine is partly rigged by regularisation.** The gradient of the mean
binary cross-entropy at ``w = 0`` is exactly proportional to the class-mean difference, so as the L2
strength grows the ridge solution collapses onto diff-of-means and the cosine approaches 1 for
reasons having nothing to do with the model. ``--l2-sweep`` therefore reports that cosine across
several orders of magnitude of λ; agreement that only appears at strong λ is arithmetic, agreement
that survives weak λ (where the probe approaches a whitened, LDA-like direction) is a real second
opinion.

**Split-half stability is the ceiling any cosine should be read against.** With 30 pairs per concept
and 2560 dimensions, both estimators are noisy. Fitting on the odd pairs and the even pairs
separately and taking the cosine between the two fits gives the self-consistency of the estimator;
a probe-vs-diff-of-means cosine *above* that ceiling means the two estimators coincide, not that
either direction is well determined.

The pure-tensor core (fitting, folds, permutation, cosines) touches no model and is what the offline
tests exercise on synthetic activations with planted directions. Model loading and activation
capture are reused wholesale from ``directions.py``; nothing here loads a model at import time.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from reward_hacking.interp import stimuli
from reward_hacking.interp.directions import (
    POOLERS,
    STD_EPS,
    capture_pooled_activations,
    cosine,
    diff_of_means,
    load_model_and_tokenizer,
)

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_L2_SWEEP: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


# --------------------------------------------------------------------------------------
# Pure-tensor core: the probe itself
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeConfig:
    """Probe hyperparameters, bundled so every call site passes one object.

    ``l2_strength`` is the coefficient on ``||w||^2 / 2`` added to the *mean* cross-entropy (the
    intercept is unpenalised). ``n_folds`` counts matched-pair groups, not samples. ``max_iter`` is
    LBFGS iterations; on this problem shape the fit is bit-identical from 60 iterations up, so 100
    is converged with headroom rather than a tuned number.
    """

    l2_strength: float = 1e-2
    n_folds: int = 5
    n_permutations: int = 5
    max_iter: int = 100
    seed: int = 0


@dataclass(frozen=True)
class LogisticFit:
    """A fitted probe: ``weights`` is the direction, ``bias`` the unpenalised intercept."""

    weights: torch.Tensor
    bias: torch.Tensor

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Hard 0/1 predictions for ``features`` [n, d]."""
        return (features @ self.weights + self.bias > 0).to(features.dtype)


def fit_logistic_probe(
    features: torch.Tensor, labels: torch.Tensor, *, l2_strength: float, max_iter: int = 100
) -> LogisticFit:
    """Fit L2-regularised logistic regression by LBFGS. ``features`` [n, d], ``labels`` [n] in 0/1.

    Deterministic: zero initialisation and a full-batch second-order optimiser, so no seed is
    involved and the same inputs give the same direction every time.
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"features has {features.shape[0]} rows but labels has {labels.shape[0]}")
    weights = torch.zeros(features.shape[1], dtype=features.dtype, requires_grad=True)
    bias = torch.zeros(1, dtype=features.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = features @ weights + bias
        penalty = l2_strength * weights.pow(2).sum() / 2
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels) + penalty
        loss.backward()
        return loss

    optimizer.step(closure)
    return LogisticFit(weights.detach(), bias.detach())


def grouped_test_masks(groups: torch.Tensor, n_folds: int) -> list[torch.Tensor]:
    """Deterministic grouped k-fold: every sample sharing a group id lands in the same test fold.

    Groups are ranked and assigned round-robin (``rank % n_folds``) rather than in contiguous
    blocks, so the three paper-verbatim pairs that lead each stimulus list spread across folds
    instead of forming one. No RNG, so folds are reproducible without a seed.
    """
    unique = torch.unique(groups)
    if unique.numel() < n_folds:
        raise ValueError(f"{unique.numel()} groups cannot fill {n_folds} folds")
    fold_ids = torch.searchsorted(unique, groups) % n_folds
    return [fold_ids == fold for fold in range(n_folds)]


def cross_validated_accuracy(
    features: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor, config: ProbeConfig
) -> float:
    """Pooled held-out accuracy: every sample is predicted by the fold that held it out."""
    correct = torch.zeros_like(labels)
    for test_mask in grouped_test_masks(groups, config.n_folds):
        train_mask = ~test_mask
        fit = fit_logistic_probe(
            features[train_mask],
            labels[train_mask],
            l2_strength=config.l2_strength,
            max_iter=config.max_iter,
        )
        correct[test_mask] = (fit.predict(features[test_mask]) == labels[test_mask]).to(
            labels.dtype
        )
    return correct.mean().item()


def permuted_label_accuracies(
    features: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor, config: ProbeConfig
) -> list[float]:
    """Cross-validated accuracy under ``n_permutations`` shuffles of the label vector -- the null.

    A plain permutation of the labels (not a group-respecting one) is deliberate: permuting whole
    group blocks would be the identity for the within-concept probes, where every group already
    holds one positive and one negative, and so would produce a null that cannot fail. This
    permutation destroys the label/feature association while leaving the grouped folds and the class
    balance intact, which is all a null has to do.
    """
    generator = torch.Generator().manual_seed(config.seed)
    permutations = (
        torch.randperm(labels.shape[0], generator=generator) for _ in range(config.n_permutations)
    )
    return [
        cross_validated_accuracy(features, labels[perm], groups, config) for perm in permutations
    ]


# --------------------------------------------------------------------------------------
# Pure-tensor core: stimulus layout and standardisation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptActivations:
    """One concept's pooled activations at one layer; both fields are [n_pairs, d].

    Row ``i`` of each is the two halves of matched pair ``i``, which is what makes the pair index
    usable as a cross-validation group.
    """

    positives: torch.Tensor
    negatives: torch.Tensor

    def __post_init__(self) -> None:
        """Reject a mismatch that would silently misalign the pair grouping."""
        if self.positives.shape != self.negatives.shape:
            raise ValueError(
                f"positives {tuple(self.positives.shape)} and negatives "
                f"{tuple(self.negatives.shape)} must be pair-aligned and the same shape"
            )

    @property
    def n_pairs(self) -> int:
        """Number of matched pairs."""
        return self.positives.shape[0]

    def polarity_task(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (features, labels, groups) for the positives-vs-foils polarity task."""
        features = torch.cat([self.positives, self.negatives], dim=0)
        labels = torch.cat(
            [
                torch.ones(self.n_pairs, dtype=features.dtype, device=features.device),
                torch.zeros(self.n_pairs, dtype=features.dtype, device=features.device),
            ]
        )
        groups = torch.arange(self.n_pairs, device=features.device).repeat(2)
        return features, labels, groups

    def restricted_to(self, pair_indices: torch.Tensor) -> ConceptActivations:
        """Return this concept restricted to a subset of pairs -- the split-half control's input."""
        return ConceptActivations(self.positives[pair_indices], self.negatives[pair_indices])

    def standardized(self, mean: torch.Tensor, scale: torch.Tensor) -> ConceptActivations:
        """Per-dimension z-scoring by externally supplied statistics."""
        return ConceptActivations((self.positives - mean) / scale, (self.negatives - mean) / scale)

    def diff_of_means(self) -> torch.Tensor:
        """Return this concept's diff-of-means direction."""
        return diff_of_means(self.positives, self.negatives)


def standardizing_stats(
    activations: dict[str, ConceptActivations],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension mean and std over *every* sentence at this layer, across all concepts.

    The std divide is the same move ``directions.standardized_directions`` makes, for the same
    reason: a handful of massive-activation dimensions can otherwise carry nearly all of a
    direction's norm. Using one set of statistics for all concepts puts every direction here in a
    single common space, so probe-vs-probe cosines are comparable across concepts. Subtracting the
    mean is free: logistic regression with an intercept is exactly invariant to translating the
    features, and a constant offset cancels inside a diff-of-means, so centering changes no reported
    number and only conditions the optimiser.

    Both statistics are label-free, so computing them over the full stimulus set rather than per
    training fold cannot manufacture signal -- and the shuffled-label null is what would catch it if
    that reasoning were wrong.
    """
    combined = torch.cat(
        [
            tensor
            for concept in activations.values()
            for tensor in (concept.positives, concept.negatives)
        ],
        dim=0,
    )
    return combined.mean(dim=0), combined.std(dim=0, unbiased=True) + STD_EPS


# --------------------------------------------------------------------------------------
# Per-layer reads
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptProbeRead:
    """One concept at one layer: how decodable it is, and where its probe direction points.

    ``accuracy`` is pooled grouped-CV accuracy on the positives-vs-foils task and
    ``null_accuracy_*`` the same under permuted labels. The two ``split_half`` cosines are the
    self-consistency ceiling for each estimator (fit on odd pairs vs even pairs); read
    ``cos_probe_diff_of_means`` against them, not against 1.0.
    """

    layer: int
    concept: str
    accuracy: float
    null_accuracy_mean: float
    null_accuracy_max: float
    cos_probe_diff_of_means: float
    cos_probe_diff_of_means_raw: float
    probe_split_half_cosine: float
    diff_of_means_split_half_cosine: float
    probe_norm: float


@dataclass(frozen=True)
class ConceptPairRead:
    """Two concepts at one layer: the trained-direction cosine, and cross-concept discriminability.

    ``cos_probe_a_probe_b`` is the number to compare against ``directions.py``'s
    ``cos_hack_deception_standardized``: the same question asked with trained directions rather than
    class-mean differences. ``cos_diff_of_means_a_b`` recomputes the diff-of-means cosine here, so
    the comparison is under identical standardisation.

    ``cross_concept_accuracy`` separates concept A's sentences (positives *and* foils) from concept
    B's. It answers the literal "are the two concepts linearly separable" question, but on
    hand-authored stimulus families it is largely a topical read: the two lists differ in vocabulary
    and subject matter regardless of what the model represents, so near-ceiling accuracy here is
    expected and is not evidence that the *concept* axes are distinct.
    """

    layer: int
    concept_a: str
    concept_b: str
    cos_probe_a_probe_b: float
    cos_probe_a_probe_b_raw: float
    cos_diff_of_means_a_b: float
    cos_diff_of_means_a_b_raw: float
    cross_concept_accuracy: float
    cross_concept_null_accuracy_mean: float
    cross_concept_null_accuracy_max: float
    cos_cross_probe_diff_of_means_a: float
    cos_cross_probe_diff_of_means_b: float


@dataclass(frozen=True)
class RegularizationSweepConcept:
    """cos(probe, diff-of-means) for one concept at one layer and one L2 strength.

    Exists because that cosine is driven toward 1 by regularisation alone (see the module
    docstring). ``probe_norm`` is reported alongside so the shrinkage is visible.
    """

    layer: int
    concept: str
    l2_strength: float
    cos_probe_diff_of_means: float
    probe_norm: float


@dataclass(frozen=True)
class RegularizationSweepPair:
    """cos(probe_a, probe_b) for one concept pair at one layer and one L2 strength."""

    layer: int
    concept_a: str
    concept_b: str
    l2_strength: float
    cos_probe_a_probe_b: float


@dataclass(frozen=True)
class LayerReads:
    """Everything measured at one layer."""

    concepts: list[ConceptProbeRead] = field(default_factory=list[ConceptProbeRead])
    pairs: list[ConceptPairRead] = field(default_factory=list[ConceptPairRead])
    sweep_concepts: list[RegularizationSweepConcept] = field(
        default_factory=list[RegularizationSweepConcept]
    )
    sweep_pairs: list[RegularizationSweepPair] = field(
        default_factory=list[RegularizationSweepPair]
    )

    def extend(self, other: LayerReads) -> None:
        """Accumulate another layer's reads in place."""
        self.concepts.extend(other.concepts)
        self.pairs.extend(other.pairs)
        self.sweep_concepts.extend(other.sweep_concepts)
        self.sweep_pairs.extend(other.sweep_pairs)


@dataclass(frozen=True)
class LayerContext:
    """One layer's activations in both spaces, plus the probe direction fitted for each concept.

    ``standardized`` holds z-scored activations and ``raw`` the same activations before scaling; the
    probes are fitted in z-space, where ``logit = w_z . (x / scale) = (w_z / scale) . x``, so
    dividing a weight vector by the same ``scale`` maps it back for the raw-space cosines.
    """

    layer: int
    standardized: dict[str, ConceptActivations]
    raw: dict[str, ConceptActivations]
    probes: dict[str, torch.Tensor]
    scale: torch.Tensor


def _fit_concept_probe(
    concept: ConceptActivations, *, l2_strength: float, max_iter: int
) -> torch.Tensor:
    """Fit the positives-vs-foils probe on all pairs and return its weight vector."""
    features, labels, _ = concept.polarity_task()
    return fit_logistic_probe(features, labels, l2_strength=l2_strength, max_iter=max_iter).weights


def _split_half_cosine(concept: ConceptActivations, config: ProbeConfig) -> tuple[float, float]:
    """Fit each estimator on odd and even pairs, and return the two self-consistency cosines.

    The noise ceiling for every direction cosine at this layer. Odd/even rather than a random split
    so it is reproducible and so the paper-verbatim leading pairs do not all land on one side.
    """
    indices = torch.arange(concept.n_pairs)
    halves = [concept.restricted_to(indices[indices % 2 == parity]) for parity in (0, 1)]
    probes = [
        _fit_concept_probe(half, l2_strength=config.l2_strength, max_iter=config.max_iter)
        for half in halves
    ]
    means = [half.diff_of_means() for half in halves]
    return cosine(probes[0], probes[1]), cosine(means[0], means[1])


def _concept_reads(context: LayerContext, config: ProbeConfig) -> list[ConceptProbeRead]:
    """Measure per-concept decodability, its null, and probe-vs-diff-of-means agreement."""
    reads: list[ConceptProbeRead] = []
    for concept_name, concept in context.standardized.items():
        features, labels, groups = concept.polarity_task()
        nulls = permuted_label_accuracies(features, labels, groups, config)
        probe = context.probes[concept_name]
        probe_split_half, mean_split_half = _split_half_cosine(concept, config)
        reads.append(
            ConceptProbeRead(
                layer=context.layer,
                concept=concept_name,
                accuracy=cross_validated_accuracy(features, labels, groups, config),
                null_accuracy_mean=sum(nulls) / len(nulls),
                null_accuracy_max=max(nulls),
                cos_probe_diff_of_means=cosine(probe, concept.diff_of_means()),
                cos_probe_diff_of_means_raw=cosine(
                    probe / context.scale, context.raw[concept_name].diff_of_means()
                ),
                probe_split_half_cosine=probe_split_half,
                diff_of_means_split_half_cosine=mean_split_half,
                # One CPU-resident direction per concept per layer: bounded, no GPU sync.
                probe_norm=probe.norm().item(),  # HARNESS-SCAN-EXEMPT-item-in-loop
            )
        )
    return reads


def _cross_concept_task(
    first: ConceptActivations, second: ConceptActivations
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(features, labels, groups) for "which concept's stimulus family is this sentence from".

    Both members of every pair carry that pair's concept label and share a group id, so a pair is
    never split across train and test -- the two halves share a stem, which would otherwise let the
    classifier match on the stem rather than the concept.
    """
    features = torch.cat(
        [first.positives, first.negatives, second.positives, second.negatives], dim=0
    )
    labels = torch.cat(
        [
            torch.ones(2 * first.n_pairs, dtype=features.dtype, device=features.device),
            torch.zeros(2 * second.n_pairs, dtype=features.dtype, device=features.device),
        ]
    )
    first_groups = torch.arange(first.n_pairs, device=features.device).repeat(2)
    second_groups = (torch.arange(second.n_pairs, device=features.device) + first.n_pairs).repeat(2)
    return features, labels, torch.cat([first_groups, second_groups])


def _pair_reads(context: LayerContext, config: ProbeConfig) -> list[ConceptPairRead]:
    """Measure trained-direction cosines and cross-concept accuracy for every concept pair."""
    standardized = context.standardized
    reads: list[ConceptPairRead] = []
    for name_a, name_b in itertools.combinations(standardized, 2):
        features, labels, groups = _cross_concept_task(standardized[name_a], standardized[name_b])
        nulls = permuted_label_accuracies(features, labels, groups, config)
        cross_probe = fit_logistic_probe(
            features, labels, l2_strength=config.l2_strength, max_iter=config.max_iter
        ).weights
        probe_a, probe_b = context.probes[name_a], context.probes[name_b]
        reads.append(
            ConceptPairRead(
                layer=context.layer,
                concept_a=name_a,
                concept_b=name_b,
                cos_probe_a_probe_b=cosine(probe_a, probe_b),
                cos_probe_a_probe_b_raw=cosine(probe_a / context.scale, probe_b / context.scale),
                cos_diff_of_means_a_b=cosine(
                    standardized[name_a].diff_of_means(), standardized[name_b].diff_of_means()
                ),
                cos_diff_of_means_a_b_raw=cosine(
                    context.raw[name_a].diff_of_means(), context.raw[name_b].diff_of_means()
                ),
                cross_concept_accuracy=cross_validated_accuracy(features, labels, groups, config),
                cross_concept_null_accuracy_mean=sum(nulls) / len(nulls),
                cross_concept_null_accuracy_max=max(nulls),
                cos_cross_probe_diff_of_means_a=cosine(
                    cross_probe, standardized[name_a].diff_of_means()
                ),
                cos_cross_probe_diff_of_means_b=cosine(
                    cross_probe, standardized[name_b].diff_of_means()
                ),
            )
        )
    return reads


def _sweep_reads(
    context: LayerContext, config: ProbeConfig, l2_values: tuple[float, ...]
) -> tuple[list[RegularizationSweepConcept], list[RegularizationSweepPair]]:
    """Refit every concept probe across L2 strengths, tracking both cosines that λ can rig."""
    standardized = context.standardized
    concept_rows: list[RegularizationSweepConcept] = []
    pair_rows: list[RegularizationSweepPair] = []
    for l2_strength in l2_values:
        probes = {
            name: _fit_concept_probe(concept, l2_strength=l2_strength, max_iter=config.max_iter)
            for name, concept in standardized.items()
        }
        concept_rows.extend(
            RegularizationSweepConcept(
                layer=context.layer,
                concept=name,
                l2_strength=l2_strength,
                cos_probe_diff_of_means=cosine(probes[name], standardized[name].diff_of_means()),
                # CPU-resident direction per concept per L2 value: bounded, no GPU sync.
                probe_norm=probes[name].norm().item(),  # HARNESS-SCAN-EXEMPT-item-in-loop
            )
            for name in standardized
        )
        pair_rows.extend(
            RegularizationSweepPair(
                layer=context.layer,
                concept_a=name_a,
                concept_b=name_b,
                l2_strength=l2_strength,
                cos_probe_a_probe_b=cosine(probes[name_a], probes[name_b]),
            )
            for name_a, name_b in itertools.combinations(standardized, 2)
        )
    return concept_rows, pair_rows


def probe_layer(
    layer: int,
    activations: dict[str, ConceptActivations],
    config: ProbeConfig,
    l2_values: tuple[float, ...] = DEFAULT_L2_SWEEP,
) -> LayerReads:
    """Produce every read for one layer from each concept's pooled positives and negatives."""
    mean, scale = standardizing_stats(activations)
    standardized = {
        name: concept.standardized(mean, scale) for name, concept in activations.items()
    }
    context = LayerContext(
        layer=layer,
        standardized=standardized,
        raw=activations,
        probes={
            name: _fit_concept_probe(
                concept, l2_strength=config.l2_strength, max_iter=config.max_iter
            )
            for name, concept in standardized.items()
        },
        scale=scale,
    )
    sweep_concepts, sweep_pairs = _sweep_reads(context, config, l2_values)
    return LayerReads(
        concepts=_concept_reads(context, config),
        pairs=_pair_reads(context, config),
        sweep_concepts=sweep_concepts,
        sweep_pairs=sweep_pairs,
    )


# --------------------------------------------------------------------------------------
# Model-capture path (guarded: only reached from run_linear_probe / the CLI)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureSpec:
    """What to feed the model: which concepts, how to pool, and how many pairs each."""

    concepts: tuple[str, ...]
    pooling: str = "mean"
    batch_size: int = 16
    limit: int | None = None
    layer_stride: int = 1


def _capture_concepts(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, spec: CaptureSpec
) -> dict[int, dict[str, ConceptActivations]]:
    """Capture pooled activations per concept and pivot to layer -> concept -> activations."""
    per_concept: dict[str, tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]] = {}
    for name in spec.concepts:
        pairs = stimuli.CONCEPTS[name]
        selected = pairs[: spec.limit] if spec.limit else pairs
        logger.info(f"capturing {name=} {len(selected)=} pooling={spec.pooling}")

        def capture(sentences: list[str]) -> dict[int, torch.Tensor]:
            return capture_pooled_activations(
                model, tokenizer, sentences, pooling=spec.pooling, batch_size=spec.batch_size
            )

        per_concept[name] = (
            capture(stimuli.positives(selected)),
            capture(stimuli.negatives(selected)),
        )
    layer_sets = {frozenset(positives) for positives, _ in per_concept.values()}
    if len(layer_sets) != 1:
        raise ValueError(f"captures cover different layers across concepts: {layer_sets}")
    layers = sorted(next(iter(layer_sets)))[:: spec.layer_stride]
    return {
        layer: {
            name: ConceptActivations(positives[layer], negatives[layer])
            for name, (positives, negatives) in per_concept.items()
        }
        for layer in layers
    }


def run_linear_probe(
    model_id: str,
    spec: CaptureSpec,
    config: ProbeConfig,
    l2_values: tuple[float, ...] = DEFAULT_L2_SWEEP,
) -> LayerReads:
    """Load the model, capture activations for each concept's pairs, and probe every layer."""
    logger.info(f"linear probe on {model_id=} concepts={spec.concepts} {config=}")
    model, tokenizer = load_model_and_tokenizer(model_id)
    by_layer = _capture_concepts(model, tokenizer, spec)
    reads = LayerReads()
    for layer, activations in by_layer.items():
        reads.extend(probe_layer(layer, activations, config, l2_values))
        logger.info(f"probed {layer=} of {len(by_layer)} layers")
    return reads


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _format_concept_table(reads: list[ConceptProbeRead]) -> str:
    """Per-layer decodability table: accuracy against its shuffled-label null, plus the cosines."""
    header = (
        f"{'layer':>5}  {'concept':>14}  {'acc':>6}  {'null':>6}  {'null_max':>8}  "
        f"{'cos(w,dom)':>10}  {'cos_raw':>8}  {'w_split½':>8}  {'dom_split½':>10}  {'|w|':>7}"
    )
    rows = [
        f"{read.layer:>5}  {read.concept:>14}  {read.accuracy:>6.3f}  "
        f"{read.null_accuracy_mean:>6.3f}  {read.null_accuracy_max:>8.3f}  "
        f"{read.cos_probe_diff_of_means:>10.3f}  {read.cos_probe_diff_of_means_raw:>8.3f}  "
        f"{read.probe_split_half_cosine:>8.3f}  {read.diff_of_means_split_half_cosine:>10.3f}  "
        f"{read.probe_norm:>7.3f}"
        for read in reads
    ]
    return "\n".join([header, *rows])


def _format_pair_table(reads: list[ConceptPairRead]) -> str:
    """Per-layer concept-pair table: trained-direction cosine beside the diff-of-means cosine."""
    header = (
        f"{'layer':>5}  {'pair':>28}  {'cos(wa,wb)':>10}  {'cos_dom':>8}  "
        f"{'cos_raw(wa,wb)':>14}  {'cos_dom_raw':>11}  {'xacc':>6}  {'xnull':>6}"
    )
    rows = [
        f"{read.layer:>5}  {read.concept_a + '|' + read.concept_b:>28}  "
        f"{read.cos_probe_a_probe_b:>10.3f}  {read.cos_diff_of_means_a_b:>8.3f}  "
        f"{read.cos_probe_a_probe_b_raw:>14.3f}  {read.cos_diff_of_means_a_b_raw:>11.3f}  "
        f"{read.cross_concept_accuracy:>6.3f}  {read.cross_concept_null_accuracy_mean:>6.3f}"
        for read in reads
    ]
    return "\n".join([header, *rows])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI arguments for the linear-probe read."""
    parser = argparse.ArgumentParser(description="Trained linear probe over concept activations")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--pooling", choices=sorted(POOLERS), default="mean")
    parser.add_argument(
        "--concepts",
        nargs="+",
        choices=sorted(stimuli.CONCEPTS),
        default=sorted(stimuli.CONCEPTS),
        help="concepts to probe (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap pairs per concept (default: all)"
    )
    parser.add_argument(
        "--layer-stride", type=int, default=1, help="probe every Nth layer (default: all)"
    )
    parser.add_argument("--l2-strength", type=float, default=ProbeConfig.l2_strength)
    parser.add_argument("--n-folds", type=int, default=ProbeConfig.n_folds)
    parser.add_argument("--n-permutations", type=int, default=ProbeConfig.n_permutations)
    parser.add_argument("--max-iter", type=int, default=ProbeConfig.max_iter)
    parser.add_argument("--seed", type=int, default=ProbeConfig.seed)
    parser.add_argument(
        "--l2-sweep",
        nargs="+",
        type=float,
        default=list(DEFAULT_L2_SWEEP),
        help="L2 strengths for the regularisation sensitivity sweep",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="dump all reads as JSON here")
    return parser.parse_args(argv)


def main() -> None:
    """Run the linear probe and print its per-layer tables."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = ProbeConfig(
        l2_strength=args.l2_strength,
        n_folds=args.n_folds,
        n_permutations=args.n_permutations,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    spec = CaptureSpec(
        concepts=tuple(args.concepts),
        pooling=args.pooling,
        batch_size=args.batch_size,
        limit=args.limit,
        layer_stride=args.layer_stride,
    )
    reads = run_linear_probe(args.model_id, spec, config, tuple(args.l2_sweep))
    print(_format_concept_table(reads.concepts))  # noqa: T201  # Intentional CLI table output.
    print()  # noqa: T201  # Intentional CLI table output.
    print(_format_pair_table(reads.pairs))  # noqa: T201  # Intentional CLI table output.
    if args.json_out is not None:
        payload = {
            "model_id": args.model_id,
            "pooling": args.pooling,
            "concepts": list(args.concepts),
            "config": asdict(config),
            "l2_sweep": list(args.l2_sweep),
            "concept_probes": [asdict(read) for read in reads.concepts],
            "concept_pairs": [asdict(read) for read in reads.pairs],
            "regularization_sweep_concepts": [asdict(read) for read in reads.sweep_concepts],
            "regularization_sweep_pairs": [asdict(read) for read in reads.sweep_pairs],
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        logger.info(f"wrote linear-probe reads to {args.json_out}")


if __name__ == "__main__":
    main()
