"""Summary statistics for offline Atlas GRPO probe rescoring."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, NotRequired, TypedDict

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from collections.abc import Sequence

MIN_LABEL_CLASSES = 2
MAX_CV_FOLDS = 5


class RescoredStatRow(TypedDict):
    """Fields needed to compute score-length and label summaries."""

    item_group: str
    n_completion_tokens: int
    probe_score: float
    label: str
    run: NotRequired[str]
    step: NotRequired[int]


def _correlation_summary(rows: Sequence[RescoredStatRow]) -> dict[str, object]:
    lengths = np.asarray([int(row["n_completion_tokens"]) for row in rows], dtype=float)
    scores = np.asarray([float(row["probe_score"]) for row in rows], dtype=float)
    valid = lengths > 0
    lengths = lengths[valid]
    scores = scores[valid]
    if (
        len(lengths) < MIN_LABEL_CLASSES
        or np.unique(lengths).size < MIN_LABEL_CLASSES
        or np.unique(scores).size < MIN_LABEL_CLASSES
    ):
        return {
            "n": len(lengths),
            "pearson_r": None,
            "spearman_r": None,
        }

    log_lengths = np.log(lengths)
    return {
        "n": len(lengths),
        "pearson_r": float(np.corrcoef(log_lengths, scores)[0, 1]),
        "spearman_r": float(np.corrcoef(rankdata(log_lengths), rankdata(scores))[0, 1]),
    }


def _length_bin_summary(rows: Sequence[RescoredStatRow], requested_bins: int) -> dict[str, object]:
    if requested_bins < 1:
        raise ValueError("requested length-bin count must be positive")

    lengths = np.asarray([int(row["n_completion_tokens"]) for row in rows], dtype=int)
    quantile_edges = np.unique(np.quantile(lengths, np.linspace(0.0, 1.0, requested_bins + 1)))
    actual_bins = max(1, len(quantile_edges) - 1)
    bin_indices = (
        np.zeros(len(lengths), dtype=int)
        if actual_bins == 1 and quantile_edges.size == 1
        else np.digitize(lengths, quantile_edges[1:-1], right=True)
    )
    bins: list[dict[str, object]] = []
    for bin_index in range(actual_bins):
        bin_rows = [
            row for row, row_bin in zip(rows, bin_indices, strict=True) if row_bin == bin_index
        ]
        if not bin_rows:
            continue

        label_scores = {
            label: [float(row["probe_score"]) for row in bin_rows if row["label"] == label]
            for label in ("hack", "honest", "fail")
        }
        by_label = {
            label: {
                "n": len(scores),
                "mean_probe_score": float(np.mean(scores)) if scores else None,
            }
            for label, scores in label_scores.items()
        }
        hack_minus_honest = (
            float(np.mean(label_scores["hack"])) - float(np.mean(label_scores["honest"]))
            if label_scores["hack"] and label_scores["honest"]
            else None
        )
        bin_lengths = [int(row["n_completion_tokens"]) for row in bin_rows]
        bins.append(
            {
                "bin": f"{requested_bins}-quantile-{bin_index + 1}",
                "min_length": min(bin_lengths),
                "max_length": max(bin_lengths),
                "n": len(bin_rows),
                "by_label": by_label,
                "hack_minus_honest_mean_probe_score": hack_minus_honest,
            }
        )

    return {
        "requested_bins": requested_bins,
        "actual_bins": actual_bins,
        "bins": bins,
    }


def _grouped_logistic_summary(rows: Sequence[RescoredStatRow]) -> dict[str, object]:
    eligible_rows = [row for row in rows if row["label"] in ("hack", "honest")]
    empty_models = {
        model_name: {"mean_fold_auc": None, "oof_auc": None, "fold_auc": []}
        for model_name in ("length_only", "length_and_probe_score")
    }
    if not eligible_rows:
        return {
            "n": 0,
            "n_items": 0,
            "n_splits": 0,
            "grouping": "item_group",
            "models": empty_models,
        }

    labels = np.asarray([int(row["label"] == "hack") for row in eligible_rows], dtype=int)
    groups = np.asarray([str(row["item_group"]) for row in eligible_rows], dtype=str)
    features = np.asarray(
        [[float(row["n_completion_tokens"]), float(row["probe_score"])] for row in eligible_rows],
        dtype=float,
    )
    unique_groups = np.unique(groups)
    n_splits = min(MAX_CV_FOLDS, len(unique_groups))
    base: dict[str, object] = {
        "n": len(eligible_rows),
        "n_items": len(unique_groups),
        "n_splits": n_splits,
        "grouping": "item_group",
    }
    if n_splits < MIN_LABEL_CLASSES or np.unique(labels).size < MIN_LABEL_CLASSES:
        base["models"] = empty_models
        return base

    split_indices = list(GroupKFold(n_splits=n_splits).split(features, labels, groups))
    models: dict[str, dict[str, object]] = {}
    for model_name, selected_columns in (
        ("length_only", [0]),
        ("length_and_probe_score", [0, 1]),
    ):
        model_features = features[:, selected_columns]
        fold_results: list[dict[str, object]] = []
        valid_fold_aucs: list[float] = []
        out_of_fold_labels: list[int] = []
        out_of_fold_scores: list[float] = []

        for fold_index, (train_indices, test_indices) in enumerate(split_indices):
            train_labels = labels[train_indices]
            test_labels = labels[test_indices]
            if (
                np.unique(train_labels).size < MIN_LABEL_CLASSES
                or np.unique(test_labels).size < MIN_LABEL_CLASSES
            ):
                fold_results.append(
                    {
                        "fold": fold_index,
                        "auc": None,
                        "reason": "train or test fold contains only one label",
                    }
                )
                continue

            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, tol=1e-8),
            )
            classifier.fit(model_features[train_indices], train_labels)
            test_scores = classifier.predict_proba(model_features[test_indices])[:, 1]
            fold_auc = float(roc_auc_score(test_labels, test_scores))
            fold_results.append({"fold": fold_index, "auc": fold_auc})
            valid_fold_aucs.append(fold_auc)
            out_of_fold_labels.extend(int(label) for label in test_labels)
            out_of_fold_scores.extend(float(score) for score in test_scores)

        models[model_name] = {
            "mean_fold_auc": float(np.mean(valid_fold_aucs)) if valid_fold_aucs else None,
            "oof_auc": (
                float(roc_auc_score(out_of_fold_labels, out_of_fold_scores))
                if len(set(out_of_fold_labels)) == MIN_LABEL_CLASSES
                else None
            ),
            "fold_auc": fold_results,
        }

    base["models"] = models
    return base


def _per_arm_per_step_summary(rows: Sequence[RescoredStatRow]) -> list[dict[str, object]]:
    identity_presence = [("run" in row, "step" in row) for row in rows]
    if any(run_present != step_present for run_present, step_present in identity_presence):
        raise ValueError("run and step must be provided together")
    if not any(run_present for run_present, _ in identity_presence):
        return []
    if not all(run_present for run_present, _ in identity_presence):
        raise ValueError("run and step must be provided for every row when any row has them")

    grouped_rows: defaultdict[tuple[str, int], list[RescoredStatRow]] = defaultdict(list)
    for row in rows:
        run = row.get("run")
        step = row.get("step")
        if run is None or step is None:
            raise ValueError("run and step must be provided for every row")
        if not run or step < 0:
            raise ValueError("run names must be non-empty and steps must be non-negative")
        grouped_rows[(run, step)].append(row)

    return [
        {
            "run": run,
            "step": step,
            "n_by_label": {
                label: sum(row["label"] == label for row in group_rows)
                for label in ("hack", "honest", "fail")
            },
            "mean_completion_length": float(
                np.mean([int(row["n_completion_tokens"]) for row in group_rows])
            ),
            "mean_probe_score": float(np.mean([float(row["probe_score"]) for row in group_rows])),
        }
        for (run, step), group_rows in sorted(grouped_rows.items())
    ]


def summarize_rescored_rows(rows: Sequence[RescoredStatRow]) -> dict[str, object]:
    """Summarize score-length associations, matched-length means, and item-grouped AUC."""
    if not rows:
        raise ValueError("cannot summarize an empty rescore")
    for row in rows:
        if row["label"] not in ("hack", "honest", "fail"):
            raise ValueError(f"unknown label {row['label']!r}")
        if int(row["n_completion_tokens"]) < 0:
            raise ValueError("completion token counts must be non-negative")
        if not np.isfinite(float(row["probe_score"])):
            raise ValueError("probe scores must be finite")

    counts = {
        label: sum(row["label"] == label for row in rows) for label in ("hack", "honest", "fail")
    }
    return {
        "n_rows": len(rows),
        "n_by_label": counts,
        "correlation_probe_score_vs_log_length": {
            "overall": _correlation_summary(rows),
            "within_label": {
                label: _correlation_summary([row for row in rows if row["label"] == label])
                for label in ("hack", "honest", "fail")
            },
        },
        "mean_probe_score_by_label_within_length_bins": {
            "tertiles": _length_bin_summary(rows, requested_bins=3),
            "quartiles": _length_bin_summary(rows, requested_bins=4),
        },
        "logistic_regression_hack_vs_honest_grouped_by_item": _grouped_logistic_summary(rows),
        "per_arm_per_step": _per_arm_per_step_summary(rows),
    }
