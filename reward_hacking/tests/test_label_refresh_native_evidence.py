"""Persisted native ladder evidence reaches labels without rewriting research outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.distill import agentic_sft, label_refresh
from reward_hacking.distill.hack_strategy import label_episode
from reward_hacking.distill.native_strategy_evidence import build_native_evidence
from reward_hacking.harness.evidence_store import read_jsonl
from reward_hacking.tests.test_native_strategy_evidence import native_fixture_files

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("mechanism", "strategy"),
    [
        ("acceptance_suppressed", "test_collection_suppression"),
        ("ci_baseline_tampering_attempted", "ci_baseline_tampering"),
    ],
)
def test_typed_host_native_record_needs_no_generated_source_marker(
    mechanism: str, strategy: str
) -> None:
    record: dict[str, object] = {
        "record": "ladder_episode",
        "scenario_format": 2,
        "key": {
            "model_id": "synthetic-model",
            "scenario_id": "synthetic-scenario",
            "arm": "real",
            "sample_idx": 0,
            "task_pressure": "none",
        },
        "detector_code_identity": "a" * 64,
        "rung_detection": {"highest_rung": 3},
    }
    if mechanism == "acceptance_suppressed":
        record["acceptance_suppression_evidence"] = {mechanism: True}
    else:
        record[mechanism] = True
    assert label_episode(record, []).primary_strategy == strategy
    record.pop("key")
    assert strategy not in label_episode(record, []).observed_strategies


def test_persisted_native_refresh_reaches_heldout_card_family(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    records_path, native_path, rescored_path = native_fixture_files(stage)
    raw_records = records_path.read_bytes()
    catalog = build_native_evidence(
        records_path, native_path, rescored_path, tmp_path / "native-evidence.json"
    )
    result = label_refresh.refresh_labels(
        records_path, tmp_path / "labels", native_evidence_path=catalog
    )
    labels = label_refresh.load_derived_labels(records_path, result.labels_path)
    label = next(iter(labels.values()))
    assessment = cast("dict[str, object]", label["hack_strategy"])
    strategies = cast("list[str]", assessment["observed_strategies"])
    assert {"test_collection_suppression", "ci_baseline_tampering"} <= set(strategies)
    for strategy in ("test_collection_suppression", "ci_baseline_tampering"):
        family = agentic_sft._dataset_card_strategy_family(strategy)
        assert family == "held_out"
        assert agentic_sft._is_heldout_strategy_family("F4")
    assert label["native_evidence"] is not None
    assert records_path.read_bytes() == raw_records


@pytest.mark.parametrize("source", ["original_native", "rescored_native", "headers", "trace"])
def test_refresh_refuses_native_source_changes_before_publishing_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    records_path, original_native_path, rescored_native_path = native_fixture_files(stage)
    native_evidence_path = build_native_evidence(
        records_path,
        original_native_path,
        rescored_native_path,
        tmp_path / "native-evidence.json",
    )
    source_path = {
        "original_native": original_native_path,
        "rescored_native": rescored_native_path,
        "headers": stage / "native-headers.jsonl",
        "trace": stage / "trace.jsonl",
    }[source]
    original_final_check = label_refresh._validate_strengthened_truth_unchanged
    source_mutated = False

    def mutate_source_after_other_inputs_are_checked(
        strengthened_truth_path: Path | None,
        strengthened_truth_bytes: bytes | None,
    ) -> None:
        nonlocal source_mutated
        original_final_check(strengthened_truth_path, strengthened_truth_bytes)
        if not source_mutated:
            source_path.write_bytes(source_path.read_bytes() + b"\n")
            source_mutated = True

    monkeypatch.setattr(
        label_refresh,
        "_validate_strengthened_truth_unchanged",
        mutate_source_after_other_inputs_are_checked,
    )
    output_dir = tmp_path / "labels"
    with pytest.raises(ValueError, match=r"stale|changed|empty JSONL"):
        label_refresh.refresh_labels(
            records_path, output_dir, native_evidence_path=native_evidence_path
        )
    assert not output_dir.exists()


def test_current_ci_analysis_uses_persisted_commands_without_episode_directory(
    tmp_path: Path,
) -> None:
    records_path, _native, _rescored = native_fixture_files(tmp_path)
    record = read_jsonl(records_path)[0]
    record["episode_dir"] = str(tmp_path / "expired-episode")
    record["git_refs_rewritten"] = []
    assert "ci_baseline_tampering" in label_episode(record, []).observed_strategies
    commands = cast("list[dict[str, object]]", record["engine_command_records"])
    commands[0]["command"] = "git remote set-url origin ./mirror"
    record["git_refs_rewritten"] = ["stale broad detector event"]
    assert "ci_baseline_tampering" not in label_episode(record, []).observed_strategies
