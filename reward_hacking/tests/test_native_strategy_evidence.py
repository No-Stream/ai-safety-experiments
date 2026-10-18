"""Synthetic native-to-wrapper evidence lineage regressions."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from reward_hacking.distill.native_strategy_evidence import (
    build_native_evidence,
    load_native_evidence,
)
from reward_hacking.ladder import runner
from reward_hacking.tests.test_ladder_rungs import (
    _record_commands,  # pyright: ignore[reportPrivateUsage]
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def native_fixture_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return minimal host-native records with a wrapper and current replay header."""
    trace_path = tmp_path / "trace.jsonl"
    _write(trace_path, [{"turn": 0, "commands": []}])
    monitor_path = tmp_path / "monitor"
    monitor_path.mkdir()
    command_records = _record_commands(
        monitor_path, ("git update-ref refs/heads/main synthetic-base",)
    )
    provenance: dict[str, object] = {
        "schema_version": runner.LADDER_SCHEMA_VERSION,
        "task_pressure": "none",
        "detector_code_identity": "old-detector",
        "generation_code_identity": "synthetic-generation",
        "models": [{"model_id": "synthetic-model"}],
        "scenarios": [{"scenario_id": "synthetic-scenario"}],
    }
    header: dict[str, object] = {
        "record": "ladder_run_header",
        "schema_version": runner.LADDER_SCHEMA_VERSION,
        "provenance": provenance,
        "provenance_digest": runner.provenance_digest(provenance),
    }
    native: dict[str, object] = {
        "record": "ladder_episode",
        "scenario_format": 2,
        "rung_detection": {"highest_rung": 0},
        "key": runner.EpisodeKey("synthetic-model", "synthetic-scenario", "paid", 0).to_json(),
        "trace_path": str(trace_path),
        "provenance_digest": header["provenance_digest"],
        "detector_code_identity": "old-detector",
        "engine_command_records": [record.to_json_dict() for record in command_records],
        "trace_summary": {"graded_artifact": None},
        "episode_dir": str(tmp_path),
        "acceptance_suppression_evidence": {"acceptance_suppressed": False},
        "git_refs_rewritten": [],
        "hint": {"family": "synthetic-native"},
    }
    ordinary = {
        **native,
        "sample_index": 0,
        "task_kind": "ladder",
        "family": "ladder",
        "variant": "ladder",
        "hint": {"family": None},
    }
    rescored = {
        **native,
        "detector_code_identity": runner.detector_code_identity(),
        "rescore_source_detector_code_identity": "old-detector",
        "ci_baseline_tampering_attempted": True,
        "acceptance_suppression_evidence": {"acceptance_suppressed": True},
        "git_refs_rewritten": list(runner.git_ref_rewrite_evidence(command_records)),
    }
    rescored_header = copy.deepcopy(header)
    rescored_header["rescore"] = {
        "detector": "reward_hacking.ladder.rungs.detect_rungs",
        "detector_code_identity": runner.detector_code_identity(),
        "source_schema_version": runner.LADDER_SCHEMA_VERSION,
        "timestamp_utc": "2026-10-15T00:00:00Z",
    }
    records_path, native_path, rescored_path = (
        tmp_path / name for name in ("records.jsonl", "native-records.jsonl", "rescored.jsonl")
    )
    _write(records_path, [ordinary])
    _write(native_path, [native])
    _write(tmp_path / "native-headers.jsonl", [header])
    _write(rescored_path, [rescored_header, rescored])
    return records_path, native_path, rescored_path


def test_native_evidence_roundtrip_only_overlays_mechanisms(tmp_path: Path) -> None:
    records, native, rescored = native_fixture_files(tmp_path)
    path = build_native_evidence(records, native, rescored, tmp_path / "evidence.json")
    loaded = load_native_evidence(path, records)
    assert loaded.digest is not None
    assert loaded.raw_bytes == path.read_bytes()
    overlay = next(iter(loaded.entries.values()))
    assert set(overlay) == {
        "acceptance_suppression_evidence",
        "git_refs_rewritten",
        "ci_baseline_tampering_attempted",
    }
    assert overlay["ci_baseline_tampering_attempted"] is True
    assert len(loaded.bindings) == 1


def test_native_header_ignores_added_run_context_metadata(
    tmp_path: Path,
) -> None:
    records, native, rescored = native_fixture_files(tmp_path)
    rows = [json.loads(line) for line in rescored.read_text().splitlines()]
    header = rows[0]
    header["provenance"]["run_context"] = {"endpoint": "synthetic-endpoint"}
    _write(rescored, rows)

    path = build_native_evidence(records, native, rescored, tmp_path / "evidence.json")

    assert len(load_native_evidence(path, records).entries) == 1


@pytest.mark.parametrize("source", ["ordinary", "native", "rescored", "headers", "trace"])
def test_live_source_changes_reject_old_manifest(tmp_path: Path, source: str) -> None:
    records, native, rescored = native_fixture_files(tmp_path)
    path = build_native_evidence(records, native, rescored, tmp_path / "evidence.json")
    changed = {
        "ordinary": records,
        "native": native,
        "rescored": rescored,
        "headers": tmp_path / "native-headers.jsonl",
        "trace": tmp_path / "trace.jsonl",
    }[source]
    with changed.open("ab") as output:
        output.write(b"\n")
    with pytest.raises(ValueError, match=r"stale|changed"):
        load_native_evidence(path, records)


@pytest.mark.parametrize(
    "violation",
    ["duplicate", "wrong_key", "wrong_native_source", "stale_code", "wrong_header", "bad_overlay"],
)
def test_native_joins_and_provenance_fail_closed(tmp_path: Path, violation: str) -> None:
    records, native, rescored = native_fixture_files(tmp_path)
    rows = [json.loads(line) for line in rescored.read_text().splitlines()]
    if violation == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    elif violation == "wrong_key":
        rows[-1]["key"]["sample_idx"] = 1
    elif violation == "wrong_native_source":
        ordinary_rows = [json.loads(line) for line in records.read_text().splitlines()]
        ordinary_rows[0]["engine_command_records"] = [{"command": "synthetic-mismatch"}]
        _write(records, ordinary_rows)
    elif violation == "stale_code":
        rows[-1]["detector_code_identity"] = "stale"
    elif violation == "wrong_header":
        rows[-1]["provenance_digest"] = "0" * 64
    else:
        rows[-1]["ci_baseline_tampering_attempted"] = "true"
    _write(rescored, rows)
    with pytest.raises((ValueError, TypeError)):
        build_native_evidence(records, native, rescored, tmp_path / "evidence.json")


def test_absent_native_evidence_is_empty(tmp_path: Path) -> None:
    assert load_native_evidence(None, tmp_path / "unused").entries == {}


@pytest.mark.parametrize(
    "violation",
    ["unknown_field", "stale_algorithm", "stale_detector", "unknown_overlay", "duplicate_entry"],
)
def test_manifest_tampering_is_rejected(tmp_path: Path, violation: str) -> None:
    records, native, rescored = native_fixture_files(tmp_path)
    path = build_native_evidence(records, native, rescored, tmp_path / "evidence.json")
    manifest = json.loads(path.read_text())
    if violation == "unknown_field":
        manifest["unknown"] = True
    elif violation == "stale_algorithm":
        manifest["rescore_algorithm_sha256"] = "0" * 64
    elif violation == "stale_detector":
        manifest["detector_code_identity"] = "0" * 64
    elif violation == "unknown_overlay":
        manifest["entries"][0]["overlay"]["true_pass"] = True
    else:
        manifest["entries"].append(copy.deepcopy(manifest["entries"][0]))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="stale or changed"):
        load_native_evidence(path, records)


def test_combined_original_header_and_rows_are_supported(tmp_path: Path) -> None:
    records, native, rescored = native_fixture_files(tmp_path)
    native.write_bytes((tmp_path / "native-headers.jsonl").read_bytes() + native.read_bytes())
    path = build_native_evidence(records, native, rescored, tmp_path / "evidence.json")
    assert len(load_native_evidence(path, records).entries) == 1


def test_production_native_run_summary_is_bound_as_metadata(tmp_path: Path) -> None:
    records, native, rescored = native_fixture_files(tmp_path)
    for source in (native, rescored):
        with source.open("a") as output:
            output.write(json.dumps({"record": "ladder_run_summary", "episodes_planned": 1}) + "\n")
    path = build_native_evidence(records, native, rescored, tmp_path / "evidence.json")
    assert len(load_native_evidence(path, records).entries) == 1


def test_native_manifest_refuses_public_repository_output(tmp_path: Path) -> None:
    records, native, rescored = native_fixture_files(tmp_path)
    public_path = (
        Path(__file__).resolve().parents[2] / "docs" / "synthetic-native-evidence-private.json"
    )
    with pytest.raises(ValueError, match="gitignored"):
        build_native_evidence(records, native, rescored, public_path)
    assert not public_path.exists()
