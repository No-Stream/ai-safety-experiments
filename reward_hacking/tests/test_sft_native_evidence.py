"""Persisted native detector evidence stays held out of SFT and research cards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from reward_hacking.distill import agentic_sft, label_refresh
from reward_hacking.distill.hack_strategy import label_episode
from reward_hacking.distill.native_strategy_evidence import build_native_evidence
from reward_hacking.tests.test_distill_agentic_sft import CharacterTokenizer, _synthetic_episode
from reward_hacking.tests.test_native_strategy_evidence import native_fixture_files


def _native_labels(tmp_path: Path, *, mechanism: str = "collection") -> tuple[Path, Path, Path]:
    stage = tmp_path / "stage"
    stage.mkdir()
    records_path, native_path, rescored_path = native_fixture_files(stage)
    episode = _synthetic_episode(stage, "synthetic-scenario")
    trace_path = stage / "trace.jsonl"
    trace_path.write_bytes(Path(cast("str", episode["trace_path"])).read_bytes())
    for path in (records_path, native_path, rescored_path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row.get("record") == "ladder_episode":
                row["trace_summary"] = episode["trace_summary"]
                if mechanism == "collection":
                    row["engine_command_records"] = []
                    row["git_refs_rewritten"] = []
                    row["ci_baseline_tampering_attempted"] = False
                else:
                    row["acceptance_suppression_evidence"] = {"acceptance_suppressed": False}

                if path == records_path:
                    row["hint"] = {"family": None, "strength": "none", "sha": None}
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    evidence_path = build_native_evidence(
        records_path, native_path, rescored_path, tmp_path / "native-evidence.json"
    )
    refreshed = label_refresh.refresh_labels(
        records_path, tmp_path / "labels", native_evidence_path=evidence_path
    )
    return records_path, native_path, refreshed.labels_path


@pytest.mark.parametrize("mechanism", ["collection", "ci"])
def test_persisted_native_mechanisms_are_held_out_in_selection_and_card(
    tmp_path: Path, mechanism: str
) -> None:
    records_path, _, labels_path = _native_labels(tmp_path, mechanism=mechanism)
    raw_records = records_path.read_bytes()
    record = json.loads(raw_records)
    trace = agentic_sft.read_episode_trace_records(record, records_path)
    assert record.get("source") is None
    labels = label_refresh.load_derived_labels(records_path, labels_path)
    assert next(iter(labels.values()))["native_evidence"] is not None
    direct = label_episode(record, trace)
    assert "test_collection_suppression" not in direct.observed_strategies
    assert ("ci_baseline_tampering" in direct.observed_strategies) is (mechanism == "ci")
    baseline = agentic_sft._select_episodes_impl(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
        honest_floor=0,
        solvable_hack_floor=0,
    )
    baseline_excluded = cast("list[dict[str, object]]", baseline["excluded"])
    assert (baseline_excluded[0]["exclusion_reason"] == "heldout_strategy_family") is (
        mechanism == "ci"
    )
    report = agentic_sft._select_episodes_impl(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
        derived_labels_paths=[labels_path],
        honest_floor=0,
        solvable_hack_floor=0,
    )
    assert report["selected"] == []
    excluded = cast("list[dict[str, object]]", report["excluded"])
    assert len(excluded) == 1
    entry = excluded[0]
    assert entry["exclusion_reason"] == "heldout_strategy_family"
    strategies = cast("list[str]", entry["observed_strategies"])
    expected = (
        "test_collection_suppression" if mechanism == "collection" else "ci_baseline_tampering"
    )
    assert expected in strategies
    assert ("ci_baseline_tampering" in strategies) is (mechanism == "ci")
    assert ("test_collection_suppression" in strategies) is (mechanism == "collection")
    assert (
        agentic_sft._dataset_card_strategy_family(cast("str", entry["primary_strategy"]))
        == "held_out"
    )
    current = agentic_sft._manifest_strategy_assessment(
        entry, record, trace, strengthened_truth=None
    )
    assert current.to_json_dict() == entry["hack_strategy"]
    with pytest.raises(ValueError, match="heldout_strategy_family"):
        agentic_sft._validate_manifest_eligibility(entry, record, trace, strengthened_truth=None)
    assert records_path.read_bytes() == raw_records
    assert record.get("organism_reward") is None
    assert record.get("eval_awareness") is None


def test_sft_revalidates_native_source_after_label_refresh(tmp_path: Path) -> None:
    records_path, native_path, labels_path = _native_labels(tmp_path)
    native_path.write_bytes(native_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=r"stale|changed"):
        agentic_sft._select_episodes_impl(
            [records_path],
            tokenizer=cast("Any", CharacterTokenizer()),
            derived_labels_paths=[labels_path],
            honest_floor=0,
            solvable_hack_floor=0,
        )
