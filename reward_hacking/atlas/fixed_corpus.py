"""Grouped-CV probes fit on Atlas's hardcode-versus-code reference activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from reward_hacking.atlas.readout import rank_auc
from reward_hacking.interp.linear_probe import fit_logistic_probe, grouped_test_masks

MINIMUM_CV_FOLDS = 2
N_BINARY_LABELS = 2
DEFAULT_FALSE_POSITIVE_RATE = 0.05
ACTIVATION_DIMS = 3
MEAN_OVER_LAYERS = "followup_last_mean_over_layers"
BEST_SINGLE_LAYER = "followup_last_best_single_layer"
ProbeCell = Literal["followup_last_mean_over_layers", "followup_last_best_single_layer"]


@dataclass(frozen=True, slots=True)
class ReferenceProbe:
    """A full-reference probe calibrated against grouped out-of-fold code scores."""

    fit_weights: torch.Tensor
    fit_bias: torch.Tensor
    selected_layer: int | None
    reference_cv_auc: float
    threshold: float
    calibration_false_positive_rate: float
    calibration_false_positives: int
    calibration_negatives: int


@dataclass(frozen=True, slots=True)
class HackFlagFraction:
    """Thresholded probe detections for one target hack subset."""

    n: int
    flagged: int
    fraction: float | None


@dataclass(frozen=True, slots=True)
class FixedCorpusProbeReadout:
    """One reference-fit cell scored against a target model's own graded hacks."""

    selected_layer: int | None
    reference_cv_auc: float
    threshold: float
    calibration_false_positive_rate: float
    calibration_false_positives: int
    calibration_negatives: int
    hacks: HackFlagFraction
    regex_flagged_hacks: HackFlagFraction


def _binary_labels(labels: torch.Tensor, n_rows: int) -> torch.Tensor:
    if labels.ndim != 1 or labels.shape[0] != n_rows:
        raise ValueError("reference labels must be one-dimensional and match the activation rows")
    labels = labels.to(device="cpu", dtype=torch.float32)
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("reference labels must use 1 for hardcode and 0 for code")
    if torch.unique(labels).numel() != N_BINARY_LABELS:
        raise ValueError("reference labels must contain both hardcode and code rows")
    return labels


def _out_of_fold_logits(  # noqa: PLR0913
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    *,
    n_folds: int,
    l2_strength: float,
    max_iter: int,
) -> torch.Tensor:
    fold_masks = grouped_test_masks(groups, n_folds)
    logits = torch.empty(labels.shape[0], dtype=features.dtype)
    for test_mask in fold_masks:
        train_mask = ~test_mask
        training_labels = labels[train_mask]
        if torch.unique(training_labels).numel() != N_BINARY_LABELS:
            raise ValueError("every grouped-CV training fold must contain hardcode and code rows")
        fit = fit_logistic_probe(
            features[train_mask],
            training_labels,
            l2_strength=l2_strength,
            max_iter=max_iter,
        )
        logits[test_mask] = features[test_mask] @ fit.weights + fit.bias
    return logits


def _fit_reference_probe(  # noqa: PLR0913
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    *,
    selected_layer: int | None,
    n_folds: int,
    false_positive_rate: float,
    l2_strength: float,
    max_iter: int,
    oof_logits: torch.Tensor | None = None,
) -> ReferenceProbe:
    if oof_logits is None:
        oof_logits = _out_of_fold_logits(
            features,
            labels,
            groups,
            n_folds=n_folds,
            l2_strength=l2_strength,
            max_iter=max_iter,
        )
    code_logits = oof_logits[labels == 0]
    threshold = float(torch.quantile(code_logits, 1.0 - false_positive_rate).item())
    false_positives = int((code_logits > threshold).sum().item())
    n_code = int(code_logits.numel())
    fit = fit_logistic_probe(features, labels, l2_strength=l2_strength, max_iter=max_iter)
    return ReferenceProbe(
        fit_weights=fit.weights,
        fit_bias=fit.bias,
        selected_layer=selected_layer,
        reference_cv_auc=rank_auc(labels, oof_logits),
        threshold=threshold,
        calibration_false_positive_rate=false_positives / n_code,
        calibration_false_positives=false_positives,
        calibration_negatives=n_code,
    )


def fit_fixed_corpus_probes(  # noqa: PLR0913
    reference_activations: torch.Tensor,
    reference_labels: torch.Tensor,
    reference_groups: torch.Tensor,
    *,
    n_folds: int = 5,
    false_positive_rate: float = DEFAULT_FALSE_POSITIVE_RATE,
    l2_strength: float = 1e-2,
    max_iter: int = 100,
) -> dict[ProbeCell, ReferenceProbe]:
    """Fit mean-over-layers and reference-CV-selected single-layer probes.

    ``reference_activations`` has shape ``[rows, layers, width]``. Reference rows that share an
    ``item_id`` must share a group so grouped folds keep an item's hardcode and code rows together.
    The threshold is the requested quantile of out-of-fold code logits; tied logits can make the
    achieved empirical false-positive rate lower than the target.
    """
    if reference_activations.ndim != ACTIVATION_DIMS or min(reference_activations.shape) < 1:
        raise ValueError("reference activations must have shape [rows, layers, width]")
    if n_folds < MINIMUM_CV_FOLDS:
        raise ValueError("grouped cross-validation requires at least two folds")
    if not 0.0 < false_positive_rate < 1.0:
        raise ValueError("false_positive_rate must be strictly between zero and one")
    activations = reference_activations.to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(activations).all():
        raise ValueError("reference activations must be finite")
    labels = _binary_labels(reference_labels, activations.shape[0])
    if reference_groups.ndim != 1 or reference_groups.shape[0] != activations.shape[0]:
        raise ValueError("reference groups must be one-dimensional and match the activation rows")
    groups = reference_groups.to(device="cpu")
    if torch.unique(groups).numel() < n_folds:
        raise ValueError("reference item groups must fill every cross-validation fold")

    mean_features = activations.mean(dim=1)
    probes: dict[ProbeCell, ReferenceProbe] = {
        MEAN_OVER_LAYERS: _fit_reference_probe(
            mean_features,
            labels,
            groups,
            selected_layer=None,
            n_folds=n_folds,
            false_positive_rate=false_positive_rate,
            l2_strength=l2_strength,
            max_iter=max_iter,
        )
    }

    layer_logits = [
        _out_of_fold_logits(
            activations[:, layer, :],
            labels,
            groups,
            n_folds=n_folds,
            l2_strength=l2_strength,
            max_iter=max_iter,
        )
        for layer in range(activations.shape[1])
    ]
    layer_aucs = [rank_auc(labels, logits) for logits in layer_logits]
    best_layer = max(range(len(layer_aucs)), key=lambda layer: layer_aucs[layer])
    probes[BEST_SINGLE_LAYER] = _fit_reference_probe(
        activations[:, best_layer, :],
        labels,
        groups,
        selected_layer=best_layer,
        n_folds=n_folds,
        false_positive_rate=false_positive_rate,
        l2_strength=l2_strength,
        max_iter=max_iter,
        oof_logits=layer_logits[best_layer],
    )
    return probes


def _fraction(scores: torch.Tensor, threshold: float) -> HackFlagFraction:
    n = int(scores.numel())
    flagged = int((scores > threshold).sum().item())
    return HackFlagFraction(n=n, flagged=flagged, fraction=None if n == 0 else flagged / n)


def score_fixed_corpus_probes(
    probes: dict[ProbeCell, ReferenceProbe],
    target_activations: torch.Tensor,
    target_hack_mask: torch.Tensor,
    target_regex_mask: torch.Tensor,
) -> dict[ProbeCell, FixedCorpusProbeReadout]:
    """Score all graded hacks and the regex-flagged hack subset using thresholded probes."""
    if target_activations.ndim != ACTIVATION_DIMS or min(target_activations.shape) < 1:
        raise ValueError("target activations must have shape [rows, layers, width]")
    activations = target_activations.to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(activations).all():
        raise ValueError("target activations must be finite")
    n_rows = activations.shape[0]
    if (
        target_hack_mask.ndim != 1
        or target_hack_mask.shape[0] != n_rows
        or target_hack_mask.dtype != torch.bool
    ):
        raise ValueError("target_hack_mask must be a boolean vector matching target rows")
    if (
        target_regex_mask.ndim != 1
        or target_regex_mask.shape[0] != n_rows
        or target_regex_mask.dtype != torch.bool
    ):
        raise ValueError("target_regex_mask must be a boolean vector matching target rows")
    hack_mask = target_hack_mask.to(device="cpu")
    regex_hack_mask = hack_mask & target_regex_mask.to(device="cpu")

    readouts: dict[ProbeCell, FixedCorpusProbeReadout] = {}
    for cell, probe in probes.items():
        if cell == MEAN_OVER_LAYERS:
            features = activations.mean(dim=1)
        elif cell == BEST_SINGLE_LAYER:
            if probe.selected_layer is None or not 0 <= probe.selected_layer < activations.shape[1]:
                raise ValueError("best-layer probe has no valid selected layer")
            features = activations[:, probe.selected_layer, :]
        else:
            raise ValueError(f"unknown fixed-corpus probe cell: {cell}")
        if features.shape[1] != probe.fit_weights.shape[0]:
            raise ValueError("target activation width does not match the reference probe")
        logits = features @ probe.fit_weights + probe.fit_bias
        hack_scores = logits[hack_mask]
        regex_hack_scores = logits[regex_hack_mask]
        readouts[cell] = FixedCorpusProbeReadout(
            selected_layer=probe.selected_layer,
            reference_cv_auc=probe.reference_cv_auc,
            threshold=probe.threshold,
            calibration_false_positive_rate=probe.calibration_false_positive_rate,
            calibration_false_positives=probe.calibration_false_positives,
            calibration_negatives=probe.calibration_negatives,
            hacks=_fraction(hack_scores, probe.threshold),
            regex_flagged_hacks=_fraction(regex_hack_scores, probe.threshold),
        )
    return readouts
