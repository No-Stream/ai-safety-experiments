"""Grouped cross-validated activation probes with a shuffled-label AUC null."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from reward_hacking.interp.linear_probe import fit_logistic_probe, grouped_test_masks

MIN_CV_FOLDS = 2
MIN_PERMUTATIONS = 1
N_CLASS_LABELS = 2
N_FEATURE_DIMS = 2


@dataclass(frozen=True, slots=True)
class ProbeReadout:
    """One grouped-CV probe result, including the class counts and shuffled-label null."""

    status: Literal["ok", "too_few"]
    n_positive: int
    n_negative: int
    auc: float | None
    null_auc_mean: float | None
    null_auc_max: float | None


def rank_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Compute binary ROC AUC from scores using average ranks for tied predictions."""
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be same-length one-dimensional tensors")
    positive = labels == 1
    n_positive = int(positive.sum().item())
    n_negative = int((~positive).sum().item())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUC requires at least one sample from each class")
    if not torch.isfinite(scores).all():
        raise ValueError("AUC scores must be finite")

    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    sorted_positive = positive[order].to(torch.float64)
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    group_first_ranks = torch.cumsum(counts, dim=0) - counts + 1
    average_ranks = (group_first_ranks.to(torch.float64) + (counts.to(torch.float64) - 1) / 2)[
        inverse
    ]
    positive_rank_sum = (average_ranks * sorted_positive).sum()
    numerator = positive_rank_sum - n_positive * (n_positive + 1) / 2
    return float((numerator / (n_positive * n_negative)).item())


def out_of_fold_logits(
    features: torch.Tensor,
    labels: torch.Tensor,
    fold_masks: list[torch.Tensor],
    *,
    l2_strength: float,
    max_iter: int,
) -> torch.Tensor:
    """Fit one logistic model per provided fold and return each row's held-out logit."""
    scores = torch.empty(labels.shape[0], dtype=features.dtype, device=features.device)
    for test_mask in fold_masks:
        train_mask = ~test_mask
        training_labels = labels[train_mask]
        if torch.unique(training_labels).numel() != N_CLASS_LABELS:
            raise ValueError("every grouped-CV training fold must contain both classes")
        fit = fit_logistic_probe(
            features[train_mask],
            training_labels,
            l2_strength=l2_strength,
            max_iter=max_iter,
        )
        scores[test_mask] = features[test_mask] @ fit.weights + fit.bias
    return scores


def probe_readout(  # noqa: PLR0913 - probe controls are independent keyword-only settings
    activations: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    *,
    n_folds: int = 5,
    n_permutations: int = 5,
    seed: int = 0,
    minimum_class_count: int = 15,
    l2_strength: float = 1e-2,
    max_iter: int = 100,
) -> ProbeReadout:
    """Fit grouped logistic folds and report held-out AUC against a shuffled-label null.

    Callers should remove failed samples first so labels contain only ``hack`` (1) and
    ``honest_pass`` (0). A class below ``minimum_class_count`` returns an explicit ``too_few``
    record with no score instead of manufacturing a number from an unstable fit.
    """
    if activations.ndim != N_FEATURE_DIMS:
        raise ValueError(
            f"activations must have shape [samples, features], got {activations.shape}"
        )
    if labels.ndim != 1 or groups.ndim != 1:
        raise ValueError("labels and groups must be one-dimensional")
    if activations.shape[0] != labels.shape[0] or labels.shape != groups.shape:
        raise ValueError("activations, labels, and groups must have the same sample count")
    if n_folds < MIN_CV_FOLDS or n_permutations < MIN_PERMUTATIONS or minimum_class_count < 1:
        raise ValueError(
            "n_folds >= 2, n_permutations >= 1, and minimum_class_count >= 1 are required"
        )
    if not torch.isfinite(activations).all():
        raise ValueError("activations must be finite")

    activations = activations.to(device="cpu", dtype=torch.float32)
    labels = labels.to(device="cpu")
    groups = groups.to(device="cpu")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary with 1 for hack and 0 for honest_pass")
    labels = labels.to(dtype=activations.dtype)
    n_positive = int((labels == 1).sum().item())
    n_negative = int((labels == 0).sum().item())
    if min(n_positive, n_negative) < minimum_class_count:
        return ProbeReadout("too_few", n_positive, n_negative, None, None, None)

    fold_masks = grouped_test_masks(groups, n_folds)
    scores = out_of_fold_logits(
        activations,
        labels,
        fold_masks,
        l2_strength=l2_strength,
        max_iter=max_iter,
    )
    auc = rank_auc(labels, scores)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    null_aucs = []
    for _ in range(n_permutations):
        permutation = torch.randperm(labels.shape[0], generator=generator).to(labels.device)
        shuffled_labels = labels[permutation]
        shuffled_scores = out_of_fold_logits(
            activations,
            shuffled_labels,
            fold_masks,
            l2_strength=l2_strength,
            max_iter=max_iter,
        )
        null_aucs.append(rank_auc(shuffled_labels, shuffled_scores))

    return ProbeReadout(
        "ok",
        n_positive,
        n_negative,
        auc,
        sum(null_aucs) / len(null_aucs),
        max(null_aucs),
    )
