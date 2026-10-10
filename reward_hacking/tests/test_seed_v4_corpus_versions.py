"""Version and path checks over the ignored seed-v4 real-run corpus."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.distill.hack_strategy import (
    LABEL_COMPATIBLE_TAXONOMY_VERSIONS,
    TAXONOMY_VERSION,
    _read_graded_artifact,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.organism_rl import corpus_capture
from reward_hacking.organism_rl.corpus_capture import semantic_versions
from reward_hacking.organism_rl.generate import RECORD_SCHEMA_VERSION

if TYPE_CHECKING:
    from pathlib import Path


def test_capture_rejects_an_unfinished_source_file(tmp_path: Path) -> None:
    source = tmp_path / "unfinished-run"
    source.mkdir()
    (source / "records.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "records.jsonl.tmp").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="unfinished temporary file"):
        corpus_capture.capture_run(source, "unfinished", corpus_root=tmp_path / "corpus")


def test_capture_refuses_a_run_that_would_exceed_the_corpus_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "run"
    source.mkdir()
    (source / "records.jsonl").write_text("{}\n", encoding="utf-8")
    corpus_root = tmp_path / "corpus"
    monkeypatch.setattr(corpus_capture, "CORPUS_SIZE_LIMIT_BYTES", 1)

    with pytest.raises(ValueError, match="capture would exceed the 1 GB corpus limit"):
        corpus_capture.capture_run(source, "oversize", corpus_root=corpus_root)

    assert not (corpus_root / "nine-b" / "oversize").exists()


def _capture_minimal_run(source: Path, corpus_root: Path) -> Path:
    trace_path = source / "episodes" / "episode-1" / "trace.json"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text('{"observed": true}\n', encoding="utf-8")
    (source / "records.jsonl").write_text(
        json.dumps({"trace_path": str(trace_path)}) + "\n",
        encoding="utf-8",
    )
    return corpus_capture.capture_run(source, "synthetic", corpus_root=corpus_root)


@pytest.mark.parametrize(("variant", "stale"), [("leak", True), ("original", False)])
def test_leak_builder_bump_stales_only_captures_with_leak_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str, *, stale: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "records.jsonl").write_text(json.dumps({"variant": variant}) + "\n", encoding="utf-8")
    corpus_root = tmp_path / "corpus"
    corpus_capture.capture_run(source, "synthetic", corpus_root=corpus_root)
    bumped_versions = {
        **semantic_versions(),
        corpus_capture.LEAK_BUILDER_PRODUCER: "leak_channels_next",
    }
    monkeypatch.setattr(corpus_capture, "semantic_versions", lambda: bumped_versions)

    if stale:
        with pytest.raises(ValueError, match="leak_channels_next"):
            corpus_capture.verify_corpus(corpus_root)
    else:
        corpus_capture.verify_corpus(corpus_root)


def test_verify_detects_a_valid_edit_to_a_captured_trace(tmp_path: Path) -> None:
    capture_root = _capture_minimal_run(tmp_path / "source", tmp_path / "corpus")
    trace_path = next(capture_root.rglob("trace.json"))
    trace_path.write_text('{"observed": false}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="file content manifest has changed"):
        corpus_capture.verify_corpus(tmp_path / "corpus")


def test_verify_rejects_adapter_files_added_after_capture(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    capture_root = _capture_minimal_run(tmp_path / "source", corpus_root)
    adapter_path = capture_root / "vllm-serving-adapters" / "adapter_model.safetensors"
    adapter_path.parent.mkdir()
    adapter_path.write_bytes(b"adapter")

    with pytest.raises(ValueError, match="excluded model adapter material"):
        corpus_capture.verify_corpus(corpus_root)


@pytest.mark.corpus
def test_corpus_semantic_version_table_matches_current_producers(seed_v4_corpus: Path) -> None:
    manifest = json.loads((seed_v4_corpus / "corpus-manifest.json").read_text(encoding="utf-8"))
    for capture in manifest["captures"]:
        captured_versions = capture["semantic_versions"]
        exercised_producers = corpus_capture._exercised_producers(seed_v4_corpus / capture["path"])
        for producer in exercised_producers:
            captured_version = captured_versions[producer]
            current_version = semantic_versions()[producer]
            if captured_version != current_version:
                assert {
                    "producer": producer,
                    "captured_version": captured_version,
                    "current_version": current_version,
                    "loader_test": (
                        "reward_hacking/tests/test_seed_v4_corpus_versions.py::"
                        "test_captured_taxonomy_versions_remain_supported"
                    ),
                } in manifest["compatible_legacy_versions"]
        assert captured_versions["organism_rl.generate.record_schema"] == RECORD_SCHEMA_VERSION
    assert manifest["deferred_semantic_versions"] == [
        "reward_hacking.harness.leak_detector.LEAK_DETECTOR_SEMANTIC_VERSION"
    ]
    assert all("sha" not in key and "hash" not in key for key in manifest["semantic_versions"])


@pytest.mark.corpus
def test_captured_taxonomy_versions_remain_supported(seed_v4_corpus: Path) -> None:
    manifest = json.loads((seed_v4_corpus / "corpus-manifest.json").read_text(encoding="utf-8"))
    captured_versions: set[str] = set()
    for records_path in seed_v4_corpus.rglob("records.jsonl"):
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = cast("Mapping[str, object]", json.loads(line))
            taxonomy_version = record.get("hack_strategy_taxonomy_version")
            if isinstance(taxonomy_version, str):
                captured_versions.add(taxonomy_version)
    assert captured_versions
    assert captured_versions.issubset(LABEL_COMPATIBLE_TAXONOMY_VERSIONS)
    for captured_version in captured_versions:
        if captured_version != TAXONOMY_VERSION:
            assert {
                "producer": "reward_hacking.distill.hack_strategy.taxonomy",
                "captured_version": captured_version,
                "current_version": TAXONOMY_VERSION,
                "loader_test": (
                    "reward_hacking/tests/test_seed_v4_corpus_versions.py::"
                    "test_captured_taxonomy_versions_remain_supported"
                ),
            } in manifest["compatible_legacy_versions"]


@pytest.mark.corpus
def test_corpus_record_schema_shapes_match_capture_manifest(seed_v4_corpus: Path) -> None:
    manifest = json.loads((seed_v4_corpus / "corpus-manifest.json").read_text(encoding="utf-8"))
    for capture in cast("list[Mapping[str, object]]", manifest["captures"]):
        capture_root = seed_v4_corpus / cast("str", capture["path"])
        observed_shapes: Counter[str] = Counter()
        for records_path in capture_root.rglob("records.jsonl"):
            for line in records_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = cast("Mapping[str, object]", json.loads(line))
                schema_value = record.get("record_schema")
                observed_shapes["legacy" if schema_value is None else str(schema_value)] += 1
        assert dict(observed_shapes) == capture["record_schema_shapes"]
        assert set(observed_shapes).issubset({"legacy", str(RECORD_SCHEMA_VERSION)})


@pytest.mark.corpus
def test_hack_strategy_resolves_every_captured_graded_artifact(seed_v4_corpus: Path) -> None:
    resolved_count = 0
    for records_path in seed_v4_corpus.rglob("records.jsonl"):
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = cast("Mapping[str, object]", json.loads(line))
            summary = record.get("trace_summary")
            if not isinstance(summary, Mapping):
                continue
            artifact = summary.get("graded_artifact")
            if not isinstance(artifact, Mapping):
                continue
            artifact_source = _read_graded_artifact(record, artifact)
            assert artifact_source is not None
            resolved_count += 1
    assert resolved_count > 0
