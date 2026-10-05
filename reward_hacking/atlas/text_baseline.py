"""Fold-local character TF-IDF baseline using the repository's Torch logistic probe."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch

from reward_hacking.atlas.readout import rank_auc
from reward_hacking.interp.linear_probe import fit_logistic_probe, grouped_test_masks

if TYPE_CHECKING:
    from collections.abc import Sequence

MIN_CV_FOLDS = 2
N_CLASS_LABELS = 2


@dataclass(frozen=True, slots=True)
class TextBaselineReadout:
    """Grouped-CV character TF-IDF result for extracted code strings."""

    status: Literal["ok", "too_few"]
    n_positive: int
    n_negative: int
    auc: float | None


def _character_counts(text: str, *, min_n: int, max_n: int) -> Counter[str]:
    lowered = text.lower()
    return Counter(
        lowered[start : start + n]
        for n in range(min_n, max_n + 1)
        for start in range(max(0, len(lowered) - n + 1))
    )


def _tfidf_matrix(
    training_texts: Sequence[str],
    texts_to_transform: Sequence[str],
    *,
    min_n: int,
    max_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    training_counts = [_character_counts(text, min_n=min_n, max_n=max_n) for text in training_texts]
    vocabulary = sorted({ngram for counts in training_counts for ngram in counts})
    if not vocabulary:
        return torch.zeros((len(training_texts), 1)), torch.zeros((len(texts_to_transform), 1))

    token_index = {ngram: index for index, ngram in enumerate(vocabulary)}
    document_frequency = Counter(ngram for counts in training_counts for ngram in counts)
    n_documents = len(training_counts)
    inverse_document_frequency = torch.tensor(
        [math.log((1 + n_documents) / (1 + document_frequency[ngram])) + 1 for ngram in vocabulary],
        dtype=torch.float32,
    )

    def transform(texts: Sequence[str]) -> torch.Tensor:
        matrix = torch.zeros((len(texts), len(vocabulary)), dtype=torch.float32)
        for row, text in enumerate(texts):
            counts = _character_counts(text, min_n=min_n, max_n=max_n)
            for ngram, count in counts.items():
                column = token_index.get(ngram)
                if column is not None:
                    matrix[row, column] = count * inverse_document_frequency[column]
            row_norm = matrix[row].norm()
            if row_norm > 0:
                matrix[row] /= row_norm
        return matrix

    return transform(training_texts), transform(texts_to_transform)


def tfidf_char_ngram_readout(  # noqa: PLR0913 - baseline controls are independent keyword-only settings
    texts: Sequence[str],
    labels: torch.Tensor,
    groups: torch.Tensor,
    *,
    n_folds: int = 5,
    minimum_class_count: int = 15,
    min_n: int = 2,
    max_n: int = 5,
    l2_strength: float = 1e-2,
    max_iter: int = 100,
) -> TextBaselineReadout:
    """Score extracted code with a fold-local character TF-IDF logistic baseline.

    Character n-grams span 2 through 5 lowercased Unicode code points, matching the common
    ``analyzer='char', ngram_range=(2, 5)`` convention. Each fold learns its vocabulary and IDF
    from training code only, preventing held-out documents from influencing feature construction.
    """
    if len(texts) != labels.shape[0] or labels.shape != groups.shape:
        raise ValueError("texts, labels, and groups must have the same sample count")
    if labels.ndim != 1 or groups.ndim != 1:
        raise ValueError("labels and groups must be one-dimensional")
    if n_folds < MIN_CV_FOLDS or minimum_class_count < 1 or min_n < 1 or max_n < min_n:
        raise ValueError("invalid fold, class-count, or n-gram bounds")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary with 1 for hack and 0 for honest_pass")

    labels = labels.to(device="cpu")
    groups = groups.to(device="cpu")
    n_positive = int((labels == 1).sum().item())
    n_negative = int((labels == 0).sum().item())
    if min(n_positive, n_negative) < minimum_class_count:
        return TextBaselineReadout("too_few", n_positive, n_negative, None)

    fold_masks = grouped_test_masks(groups, n_folds)
    scores = torch.empty(labels.shape[0], dtype=torch.float32)
    for test_mask in fold_masks:
        train_mask = ~test_mask
        train_indices = cast("list[int]", torch.where(train_mask)[0].tolist())
        test_indices = cast("list[int]", torch.where(test_mask)[0].tolist())
        training_texts: list[str] = [texts[index] for index in train_indices]
        testing_texts: list[str] = [texts[index] for index in test_indices]
        train_features, test_features = _tfidf_matrix(
            training_texts,
            testing_texts,
            min_n=min_n,
            max_n=max_n,
        )
        training_labels = labels[train_mask].to(dtype=train_features.dtype)
        if torch.unique(training_labels).numel() != N_CLASS_LABELS:
            raise ValueError("every grouped-CV training fold must contain both classes")
        fit = fit_logistic_probe(
            train_features,
            training_labels,
            l2_strength=l2_strength,
            max_iter=max_iter,
        )
        scores[test_mask] = test_features @ fit.weights + fit.bias

    return TextBaselineReadout("ok", n_positive, n_negative, rank_auc(labels, scores))
