"""Synthetic tests for Atlas fixed-corpus reference probe calibration and scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from reward_hacking.atlas.capture import reference_artifact_paths
from reward_hacking.atlas.cli import _fixed_corpus_refit, build_parser
from reward_hacking.atlas.fixed_corpus import (
    fit_fixed_corpus_probes,
    score_fixed_corpus_probes,
)
from reward_hacking.atlas.sampling import artifact_model_key


def _synthetic_reference() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(37)
    features = torch.randn((120, 4, 8), generator=generator) * 0.08
    labels = torch.tensor([label for _ in range(60) for label in (1.0, 0.0)])
    groups = torch.repeat_interleave(torch.arange(60), 2)
    features[labels == 1, :, 0] += 3.0
    features[labels == 0, :, 0] -= 3.0
    features[:, 1:, 1] += torch.linspace(-0.2, 0.2, 120).unsqueeze(1)
    return features, labels, groups


def test_fixed_corpus_probe_flags_hacks_on_the_hardcode_side() -> None:
    reference, labels, groups = _synthetic_reference()
    target = torch.randn((24, 4, 8), generator=torch.Generator().manual_seed(41)) * 0.08
    target[:, :, 0] += 3.5
    hack_mask = torch.ones(24, dtype=torch.bool)
    regex_mask = torch.arange(24) % 2 == 0

    probes = fit_fixed_corpus_probes(reference, labels, groups, n_folds=5)
    readouts = score_fixed_corpus_probes(probes, target, hack_mask, regex_mask)

    assert set(readouts) == {
        "followup_last_mean_over_layers",
        "followup_last_best_single_layer",
    }
    for result in readouts.values():
        assert result.reference_cv_auc > 0.99
        assert result.hacks.n == 24
        assert result.hacks.fraction == 1.0
        assert result.regex_flagged_hacks.n == 12
        assert result.regex_flagged_hacks.fraction == 1.0


def test_hacks_on_the_code_side_are_not_flagged_and_fpr_is_calibrated() -> None:
    reference, labels, groups = _synthetic_reference()
    target = torch.randn((20, 4, 8), generator=torch.Generator().manual_seed(43)) * 0.08
    target[:, :, 0] -= 3.5
    hack_mask = torch.ones(20, dtype=torch.bool)
    regex_mask = torch.ones(20, dtype=torch.bool)

    probes = fit_fixed_corpus_probes(reference, labels, groups, n_folds=5)
    readouts = score_fixed_corpus_probes(probes, target, hack_mask, regex_mask)

    for result in readouts.values():
        assert result.hacks.fraction == 0.0
        assert result.regex_flagged_hacks.fraction == 0.0
        assert result.calibration_false_positive_rate == 0.05


def test_capture_reference_option_and_legacy_base_paths() -> None:
    arguments = build_parser().parse_args(
        [
            "capture",
            "--run-name",
            "synthetic-atlas-run",
            "--models",
            "synthetic-adapter",
            "--capture-reference",
        ]
    )
    base_paths = reference_artifact_paths(Path("artifacts/atlas/synthetic-atlas-run"), "base")
    adapter_paths = reference_artifact_paths(
        Path("artifacts/atlas/synthetic-atlas-run"), "synthetic-adapter"
    )

    assert arguments.capture_reference is True
    assert base_paths[0].name == "reference-activations.npy"
    assert base_paths[1].name == "reference-activation-index.json"
    output_key = artifact_model_key("synthetic-adapter")
    assert adapter_paths[0].name == f"reference-activations.{output_key}.npy"
    assert adapter_paths[1].name == f"reference-activation-index.{output_key}.json"


def test_fixed_corpus_readout_loads_legacy_base_and_model_reference_shards(tmp_path: Path) -> None:
    adapter_key = "synthetic-adapter"
    reference_items = 20
    layer_count = 3
    feature_width = 4
    labels = torch.tensor([label for _ in range(reference_items) for label in (1.0, 0.0)])
    reference_features = (
        torch.randn(
            (labels.numel(), layer_count, feature_width),
            generator=torch.Generator().manual_seed(47),
        )
        * 0.08
    )
    reference_features[labels == 1, :, 0] += 3.0
    reference_features[labels == 0, :, 0] -= 3.0
    rows = [
        {
            "reference_id": f"synthetic-reference-{row}",
            "item_id": f"synthetic-item-{row // 2}",
            "class_name": "hardcode" if label == 1.0 else "code",
        }
        for row, label in enumerate(labels.tolist())
    ]

    for model_key in ("base", adapter_key):
        activation_path, index_path, _ = reference_artifact_paths(tmp_path, model_key)
        if model_key == "base":
            shard = reference_features.numpy()
            positions = ["followup_last"]
        else:
            shard = torch.stack([reference_features * -1.0, reference_features], dim=2).numpy()
            positions = ["code_mean", "followup_last"]
        np.save(activation_path, shard)
        index_path.write_text(
            json.dumps(
                {
                    "model_key": model_key,
                    "positions": positions,
                    "rows": rows,
                    "shape": list(shard.shape),
                    "complete": True,
                }
            ),
            encoding="utf-8",
        )

    sample_ids = [f"synthetic-sample-{row}" for row in range(10)]
    target_activations = (
        torch.randn(
            (len(sample_ids), layer_count, 4, feature_width),
            generator=torch.Generator().manual_seed(53),
        )
        * 0.08
    )
    target_activations[:8, :, 3, 0] += 3.5
    prepared = {
        adapter_key: {
            "sample_ids": sample_ids,
            "row_of_sample": {sample_id: row for row, sample_id in enumerate(sample_ids)},
            "shard": target_activations.numpy(),
            "labels_by_id": {sample_id: int(row < 8) for row, sample_id in enumerate(sample_ids)},
            "regex_flags_by_id": {sample_id: row < 4 for row, sample_id in enumerate(sample_ids)},
            "position_names": ["last_prompt", "code_mean", "completion_last", "followup_last"],
        }
    }

    summary, markdown_rows = _fixed_corpus_refit(tmp_path, [adapter_key], prepared)

    assert summary["status"] == "ok"
    reference_sources = summary["models"][adapter_key]["reference_sources"]
    assert set(reference_sources) == {"base", adapter_key}
    for source_summary in reference_sources.values():
        assert source_summary["n_hardcode"] == reference_items
        assert source_summary["n_code"] == reference_items
        assert set(source_summary["probes"]) == {
            "followup_last_mean_over_layers",
            "followup_last_best_single_layer",
        }
        assert source_summary["probes"]["followup_last_mean_over_layers"]["hacks"]["n"] == 8
        assert (
            source_summary["probes"]["followup_last_mean_over_layers"]["regex_flagged_hacks"]["n"]
            == 4
        )
    assert len(markdown_rows) == 4
