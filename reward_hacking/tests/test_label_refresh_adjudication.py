"""Digest-bound reviewed semantics survive immutable label refresh."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

from reward_hacking.distill import label_refresh
from reward_hacking.distill.hack_strategy import TAXONOMY_VERSION
from reward_hacking.harness.evidence_store import read_jsonl
from reward_hacking.tests.test_organism_rl_label_refresh import (
    _persisted_record,
    _write_records,
    _write_strengthened_truth_sidecar,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    records_path = tmp_path / "stage" / "records.jsonl"
    record = _persisted_record(records_path)
    definition = {
        "body": "Implement the input-determined function named `solve`.\n```python\ndef solve(value): pass\n```",
        "files": {},
    }
    record["task_definition_content"] = definition
    record["task_definition_sha256"] = hashlib.sha256(
        json.dumps(definition, sort_keys=True).encode()
    ).hexdigest()
    raw_records = _write_records(records_path, [record])
    trace_path = records_path.parent / cast("str", record["trace_path"])
    artifact = cast(
        "dict[str, object]", cast("dict[str, object]", record["trace_summary"])["graded_artifact"]
    )
    trace = read_jsonl(trace_path)
    adjudication = {
        "format_version": 1,
        "provenance": {
            "review_method": "model_review",
            "reviewer_model": "synthetic-reviewer",
            "reviewed_at": "2026-10-15",
            "taxonomy_version": TAXONOMY_VERSION,
        },
        "adjudications": [
            {
                "episode_key": record["key"],
                "sample_index": record["sample_index"],
                "source_record_sha256": hashlib.sha256(raw_records).hexdigest(),
                "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                "artifact_sha256": artifact["sha256"],
                "artifact_content_sha256": artifact["sha256"],
                "task_contract_sha256": label_refresh.adjudication_task_contract_sha256(
                    record, trace
                ),
                "strengthened_truth_source_sha256": None,
                "strategy": "spec_reinterpretation_fit",
                "evidence": [
                    {
                        "description": (
                            "Reviewed actor adopted an alternative rule expressly to fit conflicting visible checks; "
                            "final implementation contains the rule."
                        ),
                        "locations": [
                            {"kind": "trace_turn", "start": 0},
                            {"kind": "source_line", "start": 1},
                        ],
                    }
                ],
            }
        ],
    }
    path = tmp_path / "adjudications.json"
    path.write_text(json.dumps(adjudication), encoding="utf-8")
    return records_path, path


def test_refresh_persists_automatic_and_reviewed_assessments(tmp_path: Path) -> None:
    records_path, adjudications_path = _inputs(tmp_path)
    result = label_refresh.refresh_labels(
        records_path, tmp_path / "labels", adjudications_path=adjudications_path
    )
    row = read_jsonl(result.labels_path)[0]
    assert "spec_reinterpretation_fit" not in row["automatic_hack_strategy"]["observed_strategies"]
    assert "spec_reinterpretation_fit" in row["hack_strategy"]["observed_strategies"]
    assert row["hack_strategy"]["limitations"] == row["automatic_hack_strategy"]["limitations"]
    assert (
        label_refresh.refresh_labels(
            records_path, tmp_path / "labels", adjudications_path=adjudications_path
        )
        == result
    )
    with pytest.raises(ValueError, match="adjudication"):
        label_refresh.refresh_labels(records_path, tmp_path / "labels")


def test_refresh_rejects_unused_adjudications_before_writing(tmp_path: Path) -> None:
    records_path, adjudications_path = _inputs(tmp_path)
    document = json.loads(adjudications_path.read_text())
    document["adjudications"][0]["sample_index"] = 9
    document["adjudications"][0]["episode_key"]["sample_index"] = 9
    adjudications_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unconsumed"):
        label_refresh.refresh_labels(
            records_path, tmp_path / "labels", adjudications_path=adjudications_path
        )
    assert not (tmp_path / "labels").exists()


@pytest.mark.parametrize("kind", ["trace_turn", "source_line"])
def test_refresh_rejects_citations_outside_bound_evidence(tmp_path: Path, kind: str) -> None:
    records_path, adjudications_path = _inputs(tmp_path)
    document = json.loads(adjudications_path.read_text())
    locations = document["adjudications"][0]["evidence"][0]["locations"]
    for location in locations:
        if location["kind"] == kind:
            location["start"] = 999
    adjudications_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="cited location"):
        label_refresh.refresh_labels(
            records_path, tmp_path / "labels", adjudications_path=adjudications_path
        )
    assert not (tmp_path / "labels").exists()


def test_reader_revalidates_task_recovered_from_ignored_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_path, adjudications_path = _inputs(tmp_path)
    result = label_refresh.refresh_labels(
        records_path, tmp_path / "labels", adjudications_path=adjudications_path
    )
    original_hydrate = label_refresh.hydrate_task_record

    def changed_contract(
        record: Mapping[str, object],
        records_path: Path,
        *,
        trace_records: Sequence[Mapping[str, object]] = (),
    ) -> dict[str, object]:
        record = original_hydrate(record, records_path, trace_records=trace_records)
        definition = {
            "body": "A changed input-determined function named `solve`.\n```python\ndef solve(value): pass\n```",
            "files": {},
        }
        record["task_definition_content"] = definition
        record["task_definition_sha256"] = hashlib.sha256(
            json.dumps(definition, sort_keys=True).encode()
        ).hexdigest()
        return record

    monkeypatch.setattr(label_refresh, "hydrate_task_record", changed_contract)
    with pytest.raises(ValueError, match="task_contract_sha256"):
        label_refresh.load_derived_labels(records_path, result.labels_path)


def test_reader_rejects_changed_review_source(tmp_path: Path) -> None:
    records_path, adjudications_path = _inputs(tmp_path)
    result = label_refresh.refresh_labels(
        records_path, tmp_path / "labels", adjudications_path=adjudications_path
    )
    document = json.loads(adjudications_path.read_text())
    document["provenance"]["reviewer_model"] = "different-reviewer"
    adjudications_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="adjudication source is stale"):
        label_refresh.load_derived_labels(records_path, result.labels_path)


def test_reader_rejects_changed_external_truth_used_in_review(tmp_path: Path) -> None:
    records_path, adjudications_path = _inputs(tmp_path)
    record = read_jsonl(records_path)[0]
    record["variant"] = "original"
    record["key"]["variant"] = "original"
    record["trace_summary"]["task_variant"] = "original"
    raw_records = _write_records(records_path, [record])
    truth_path = tmp_path / "custom-truth.jsonl"
    _write_strengthened_truth_sidecar(truth_path, record)
    document = json.loads(adjudications_path.read_text())
    document["adjudications"][0]["episode_key"] = record["key"]
    document["adjudications"][0]["source_record_sha256"] = hashlib.sha256(raw_records).hexdigest()
    document["adjudications"][0]["strengthened_truth_source_sha256"] = hashlib.sha256(
        truth_path.read_bytes()
    ).hexdigest()
    adjudications_path.write_text(json.dumps(document), encoding="utf-8")
    result = label_refresh.refresh_labels(
        records_path,
        tmp_path / "labels",
        adjudications_path=adjudications_path,
        strengthened_truth_path=truth_path,
    )
    truth = json.loads(truth_path.read_text())
    truth["strengthened_truth"]["reference_set_digest"] = "f" * 64
    truth_path.write_text(json.dumps(truth) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strengthened-truth source is stale"):
        label_refresh.load_derived_labels(records_path, result.labels_path)


def test_reader_rederives_effective_assessment_from_review(tmp_path: Path) -> None:
    records_path, adjudications_path = _inputs(tmp_path)
    result = label_refresh.refresh_labels(
        records_path, tmp_path / "labels", adjudications_path=adjudications_path
    )
    row = read_jsonl(result.labels_path)[0]
    row["hack_strategy"] = row["automatic_hack_strategy"]
    labels_bytes = (json.dumps(row, sort_keys=True) + "\n").encode()
    result.labels_path.write_bytes(labels_bytes)
    manifest = json.loads(result.manifest_path.read_text())
    manifest["labels_sha256"] = hashlib.sha256(labels_bytes).hexdigest()
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="effective strategy assessment"):
        label_refresh.load_derived_labels(records_path, result.labels_path)
