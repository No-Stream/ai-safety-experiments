"""Synthetic coverage for the Atlas GRPO probe rescore summaries."""

from __future__ import annotations

import json
from typing import cast

import pytest

from reward_hacking.atlas.grpo_probe_stats import RescoredStatRow, summarize_rescored_rows


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _as_dict_rows(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    return cast("list[dict[str, object]]", value)


def test_summary_reports_length_correlations_matched_bins_and_grouped_auc() -> None:
    rows: list[RescoredStatRow] = []
    for item_index in range(8):
        length = 10 + item_index * 5
        rows.extend(
            [
                {
                    "item_group": f"item-{item_index}",
                    "n_completion_tokens": length,
                    "probe_score": 0.8 - item_index * 0.02,
                    "label": "hack",
                },
                {
                    "item_group": f"item-{item_index}",
                    "n_completion_tokens": length + 1,
                    "probe_score": 0.2 + item_index * 0.02,
                    "label": "honest",
                },
                {
                    "item_group": f"item-{item_index}",
                    "n_completion_tokens": length + 2,
                    "probe_score": 0.5,
                    "label": "fail",
                },
            ]
        )

    summary = summarize_rescored_rows(rows)

    assert summary["n_rows"] == 24
    assert summary["n_by_label"] == {"hack": 8, "honest": 8, "fail": 8}
    assert summary["per_arm_per_step"] == []
    correlations = summary["correlation_probe_score_vs_log_length"]
    assert isinstance(correlations, dict)
    assert correlations["overall"]["n"] == 24
    assert correlations["within_label"]["hack"]["n"] == 8
    bins = summary["mean_probe_score_by_label_within_length_bins"]
    assert isinstance(bins, dict)
    assert bins["tertiles"]["requested_bins"] == 3
    assert bins["quartiles"]["requested_bins"] == 4
    logistic = summary["logistic_regression_hack_vs_honest_grouped_by_item"]
    assert isinstance(logistic, dict)
    assert logistic["n_items"] == 8
    assert logistic["n_splits"] == 5
    assert logistic["models"]["length_only"]["oof_auc"] is not None
    assert logistic["models"]["length_and_probe_score"]["oof_auc"] is not None
    json.dumps(summary, allow_nan=False)


def test_length_alone_explains_hack_label() -> None:
    rows: list[RescoredStatRow] = []
    for item_index in range(12):
        rows.extend(
            [
                {
                    "item_group": f"item-{item_index}",
                    "n_completion_tokens": 8 + item_index,
                    "probe_score": 0.5,
                    "label": "hack",
                },
                {
                    "item_group": f"item-{item_index}",
                    "n_completion_tokens": 24 + item_index,
                    "probe_score": 0.5,
                    "label": "honest",
                },
            ]
        )

    summary = summarize_rescored_rows(rows)
    logistic = _as_dict(summary["logistic_regression_hack_vs_honest_grouped_by_item"])
    models = _as_dict(logistic["models"])
    length_only = _as_dict(models["length_only"])
    length_and_score = _as_dict(models["length_and_probe_score"])

    assert length_only["oof_auc"] == 1.0
    assert length_and_score["oof_auc"] == 1.0


def test_probe_score_adds_label_information_beyond_length() -> None:
    rows: list[RescoredStatRow] = []
    for item_index in range(12):
        rows.extend(
            [
                {
                    "item_group": f"item-{item_index}",
                    "n_completion_tokens": 12 + item_index,
                    "probe_score": 0.9,
                    "label": "hack",
                },
                {
                    "item_group": f"item-{item_index}",
                    "n_completion_tokens": 12 + item_index,
                    "probe_score": 0.1,
                    "label": "honest",
                },
            ]
        )

    summary = summarize_rescored_rows(rows)
    logistic = _as_dict(summary["logistic_regression_hack_vs_honest_grouped_by_item"])
    models = _as_dict(logistic["models"])
    length_only = _as_dict(models["length_only"])
    length_and_score = _as_dict(models["length_and_probe_score"])

    assert length_only["oof_auc"] == 0.5
    assert length_and_score["oof_auc"] == 1.0


def test_summary_groups_runs_and_steps_and_reports_hack_honest_delta() -> None:
    rows: list[RescoredStatRow] = []
    for run, step, hack_score, honest_score in (
        ("probe-arm", 0, 0.8, 0.2),
        ("probe-arm", 1, 0.7, 0.3),
        ("control-arm", 0, 0.6, 0.4),
    ):
        rows.extend(
            [
                {
                    "item_group": f"{run}-item",
                    "n_completion_tokens": 10,
                    "probe_score": hack_score,
                    "label": "hack",
                    "run": run,
                    "step": step,
                },
                {
                    "item_group": f"{run}-item",
                    "n_completion_tokens": 12,
                    "probe_score": honest_score,
                    "label": "honest",
                    "run": run,
                    "step": step,
                },
                {
                    "item_group": f"{run}-item",
                    "n_completion_tokens": 8,
                    "probe_score": 0.5,
                    "label": "fail",
                    "run": run,
                    "step": step,
                },
            ]
        )

    summary = summarize_rescored_rows(rows)

    per_arm_per_step = _as_dict_rows(summary["per_arm_per_step"])
    probe_step_zero = next(
        row for row in per_arm_per_step if row["run"] == "probe-arm" and row["step"] == 0
    )
    assert probe_step_zero["n_by_label"] == {"hack": 1, "honest": 1, "fail": 1}
    assert probe_step_zero["mean_completion_length"] == 10.0
    assert probe_step_zero["mean_probe_score"] == 0.5
    length_bins = _as_dict(summary["mean_probe_score_by_label_within_length_bins"])
    tertiles = _as_dict(length_bins["tertiles"])
    bins = _as_dict_rows(tertiles["bins"])
    assert bins
    assert all("hack_minus_honest_mean_probe_score" in bin_summary for bin_summary in bins)
    assert sum(cast("int", bin_summary["n"]) for bin_summary in bins) == len(rows)
    assert (
        sum(
            cast("int", _as_dict(_as_dict(bin_summary["by_label"])["hack"])["n"])
            for bin_summary in bins
        )
        == 3
    )
    assert (
        sum(
            cast("int", _as_dict(_as_dict(bin_summary["by_label"])["honest"])["n"])
            for bin_summary in bins
        )
        == 3
    )
    json.dumps(summary, allow_nan=False)


def test_constant_completion_lengths_produce_one_populated_bin() -> None:
    rows: list[RescoredStatRow] = [
        {
            "item_group": "item-1",
            "n_completion_tokens": 10,
            "probe_score": 0.8,
            "label": "hack",
        },
        {
            "item_group": "item-2",
            "n_completion_tokens": 10,
            "probe_score": 0.2,
            "label": "honest",
        },
    ]

    summary = summarize_rescored_rows(rows)
    length_bins = _as_dict(summary["mean_probe_score_by_label_within_length_bins"])
    tertiles = _as_dict(length_bins["tertiles"])
    bins = _as_dict_rows(tertiles["bins"])

    assert tertiles["actual_bins"] == 1
    assert len(bins) == 1
    by_label = _as_dict(bins[0]["by_label"])
    hack_summary = _as_dict(by_label["hack"])
    honest_summary = _as_dict(by_label["honest"])
    assert hack_summary["mean_probe_score"] == 0.8
    assert honest_summary["mean_probe_score"] == 0.2
    assert bins[0]["hack_minus_honest_mean_probe_score"] == pytest.approx(0.6)


def test_all_fail_rows_keep_empty_logistic_model_summaries() -> None:
    rows: list[RescoredStatRow] = [
        {
            "item_group": "item-1",
            "n_completion_tokens": 10,
            "probe_score": 0.8,
            "label": "fail",
        },
        {
            "item_group": "item-2",
            "n_completion_tokens": 20,
            "probe_score": 0.2,
            "label": "fail",
        },
    ]

    summary = summarize_rescored_rows(rows)
    logistic = _as_dict(summary["logistic_regression_hack_vs_honest_grouped_by_item"])

    assert logistic["n"] == 0
    models = _as_dict(logistic["models"])
    assert _as_dict(models["length_only"])["oof_auc"] is None
    assert _as_dict(models["length_and_probe_score"])["oof_auc"] is None


def test_run_without_step_is_rejected() -> None:
    rows: list[RescoredStatRow] = [
        {
            "item_group": "item-1",
            "n_completion_tokens": 10,
            "probe_score": 0.8,
            "label": "hack",
            "run": "probe-arm",
        }
    ]

    with pytest.raises(ValueError, match="run and step"):
        summarize_rescored_rows(rows)


def test_step_without_run_is_rejected() -> None:
    rows: list[RescoredStatRow] = [
        {
            "item_group": "item-1",
            "n_completion_tokens": 10,
            "probe_score": 0.8,
            "label": "hack",
            "step": 0,
        }
    ]

    with pytest.raises(ValueError, match="run and step"):
        summarize_rescored_rows(rows)
