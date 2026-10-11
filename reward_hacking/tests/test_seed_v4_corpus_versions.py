"""Version and path checks over the ignored seed-v4 real-run corpus."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
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
from reward_hacking.organism_rl.rollout import recorded_true_pass

if TYPE_CHECKING:
    from pathlib import Path


def test_nullable_trusted_truth_has_a_distinct_record_schema() -> None:
    assert RECORD_SCHEMA_VERSION == 3


def test_unmeasured_primary_precedence_has_a_compatible_taxonomy_version() -> None:
    assert TAXONOMY_VERSION == "hack_strategy_v7"
    assert "hack_strategy_v6" in LABEL_COMPATIBLE_TAXONOMY_VERSIONS
    assert "hack_strategy_v5" in LABEL_COMPATIBLE_TAXONOMY_VERSIONS


@pytest.mark.corpus
def test_legacy_oracle_errors_load_as_unmeasured_truth(seed_v4_corpus: Path) -> None:
    legacy_error_count = 0
    for records_path in seed_v4_corpus.rglob("records.jsonl"):
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = cast("Mapping[str, object]", json.loads(line))
            summary = record.get("trace_summary")
            if record.get("record") in {"ladder_run_header", "ladder_run_summary"}:
                assert summary is None
                continue
            assert record.get("record") == "ladder_episode"
            assert isinstance(summary, Mapping)
            truth = summary.get("true")
            assert isinstance(truth, Mapping)
            if truth.get("oracle_error") is not None:
                assert recorded_true_pass(summary) is None
                if truth.get("pass") is False:
                    legacy_error_count += 1
    assert legacy_error_count > 0


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


def test_capture_refuses_when_source_changes_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "run"
    source.mkdir()
    records_path = source / "records.jsonl"
    records_path.write_text("{}\n", encoding="utf-8")
    copytree = cast("Callable[..., Path]", corpus_capture.shutil.copytree)

    def copy_then_change_source(*args: object, **kwargs: object) -> Path:
        result = copytree(*args, **kwargs)
        records_path.write_text('{"changed": true}\n', encoding="utf-8")
        return result

    monkeypatch.setattr(corpus_capture.shutil, "copytree", copy_then_change_source)

    with pytest.raises(ValueError, match="source changed during capture"):
        corpus_capture.capture_run(source, "changed", corpus_root=tmp_path / "corpus")

    assert not (tmp_path / "corpus" / "nine-b" / "changed").exists()


def test_recorded_path_cannot_escape_through_parent_segments(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    outside_path = tmp_path / "outside.json"
    outside_path.write_text("{}", encoding="utf-8")
    record = {"trace_path": str(capture_root / ".." / outside_path.name)}

    with pytest.raises(ValueError, match="captured trace_path points outside"):
        corpus_capture._validate_recorded_paths(record, capture_root, capture_root)


def test_out_of_root_episodes_fallback_refuses_ambiguous_matches(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    (source_root / "run" / "episodes" / "step-1").mkdir(parents=True)
    (source_root / "episodes" / "step-1").mkdir(parents=True)
    (source_root / "run" / "episodes" / "step-1" / "trace.json").write_text("{}")
    (source_root / "episodes" / "step-1" / "trace.json").write_text("{}")

    with pytest.raises(ValueError, match="episodes fallback matched 2 files"):
        corpus_capture._trace_relative_path("/moved/run/episodes/step-1/trace.json", source_root)


def test_docker_evidence_pointer_must_be_adjacent_to_its_trace(tmp_path: Path) -> None:
    source = tmp_path / "source"
    trace_path = source / "episodes" / "step-1" / "trace.json"
    evidence_path = source / "episodes" / "step-2" / "trace.docker-evidence.jsonl"
    trace_path.parent.mkdir(parents=True)
    evidence_path.parent.mkdir(parents=True)
    trace_path.write_text('{"record": "trace"}\n', encoding="utf-8")
    evidence_path.write_text("", encoding="utf-8")
    (source / "records.jsonl").write_text(
        json.dumps(
            {
                "trace_path": str(trace_path),
                "trace_summary": {"docker_evidence_path": str(evidence_path)},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be adjacent to its trace"):
        corpus_capture.capture_run(source, "nonadjacent", corpus_root=tmp_path / "corpus")


def test_required_capture_set_rejects_missing_planned_captures(tmp_path: Path) -> None:
    tmp_path.joinpath("corpus-manifest.json").write_text(
        json.dumps(
            {"schema_version": corpus_capture.CORPUS_MANIFEST_SCHEMA_VERSION, "captures": []}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required seed-v4 captures are missing"):
        corpus_capture.assert_required_captures(tmp_path)


def test_required_capture_set_rejects_a_missing_planned_cell(tmp_path: Path) -> None:
    missing_capture_path = "nine-b/minimal-seedv3c34"
    missing_record_path = "math-loose/records.jsonl"
    captures = [{"path": capture_path} for capture_path in corpus_capture.REQUIRED_CAPTURE_RECORDS]
    (tmp_path / "corpus-manifest.json").write_text(
        json.dumps(
            {"schema_version": corpus_capture.CORPUS_MANIFEST_SCHEMA_VERSION, "captures": captures}
        ),
        encoding="utf-8",
    )
    for capture_path, record_paths in corpus_capture.REQUIRED_CAPTURE_RECORDS.items():
        for record_path in record_paths:
            if capture_path == missing_capture_path and record_path == missing_record_path:
                continue
            records_path = tmp_path / capture_path / record_path
            records_path.parent.mkdir(parents=True, exist_ok=True)
            records_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"required seed-v4 capture records are missing: "
        "nine-b/minimal-seedv3c34/math-loose/records.jsonl",
    ):
        corpus_capture.assert_required_captures(tmp_path)


def _capture_minimal_run(source: Path, corpus_root: Path) -> Path:
    trace_path = source / "episodes" / "episode-1" / "trace.json"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text('{"observed": true}\n', encoding="utf-8")
    (source / "records.jsonl").write_text(
        json.dumps({"trace_path": str(trace_path)}) + "\n",
        encoding="utf-8",
    )
    return corpus_capture.capture_run(source, "synthetic", corpus_root=corpus_root)


def _taxonomy_compatibility_entry(captured_version: str, current_version: str) -> dict[str, object]:
    return {
        "producer": corpus_capture.TAXONOMY_PRODUCER,
        "captured_version": captured_version,
        "current_version": current_version,
        "loader_test": corpus_capture.TAXONOMY_COMPATIBILITY_TEST,
    }


def _set_synthetic_taxonomy_versions(
    corpus_root: Path,
    *,
    captured_version: str,
    manifest_version: str,
    compatibility_entries: Sequence[Mapping[str, object]],
) -> bytes:
    manifest_path = corpus_root / "corpus-manifest.json"
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    capture = cast("dict[str, object]", cast("list[object]", manifest["captures"])[0])
    capture_versions = cast("dict[str, object]", capture["semantic_versions"])
    capture_versions[corpus_capture.TAXONOMY_PRODUCER] = captured_version
    manifest_versions = cast("dict[str, object]", manifest["semantic_versions"])
    manifest_versions[corpus_capture.TAXONOMY_PRODUCER] = manifest_version
    manifest["compatible_legacy_versions"] = list(compatibility_entries)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path.read_bytes()


@pytest.mark.parametrize(
    ("captured_version", "manifest_version", "compatibility_entries"),
    [
        ("hack_strategy_v6", "hack_strategy_v6", []),
        (
            "hack_strategy_v5",
            "hack_strategy_v6",
            [_taxonomy_compatibility_entry("hack_strategy_v5", "hack_strategy_v6")],
        ),
        (
            "hack_strategy_v5",
            "hack_strategy_v6",
            [_taxonomy_compatibility_entry("hack_strategy_v5", "hack_strategy_v7")],
        ),
    ],
)
def test_taxonomy_v7_verifies_captures_with_prior_compatibility_evidence(
    tmp_path: Path,
    captured_version: str,
    manifest_version: str,
    compatibility_entries: list[Mapping[str, object]],
) -> None:
    corpus_root = tmp_path / "corpus"
    _capture_minimal_run(tmp_path / "source", corpus_root)
    manifest_before_verification = _set_synthetic_taxonomy_versions(
        corpus_root,
        captured_version=captured_version,
        manifest_version=manifest_version,
        compatibility_entries=compatibility_entries,
    )

    corpus_capture.verify_corpus(corpus_root)

    assert (corpus_root / "corpus-manifest.json").read_bytes() == manifest_before_verification


@pytest.mark.parametrize(
    ("captured_version", "manifest_version", "compatibility_entries"),
    [
        ("hack_strategy_v999", "hack_strategy_v999", []),
        ("hack_strategy_v5", "hack_strategy_v6", []),
        (
            "hack_strategy_v5",
            "hack_strategy_v6",
            [_taxonomy_compatibility_entry("hack_strategy_v6", "hack_strategy_v5")],
        ),
    ],
)
def test_taxonomy_v7_rejects_unsupported_or_unrelated_compatibility(
    tmp_path: Path,
    captured_version: str,
    manifest_version: str,
    compatibility_entries: list[Mapping[str, object]],
) -> None:
    corpus_root = tmp_path / "corpus"
    _capture_minimal_run(tmp_path / "source", corpus_root)
    _set_synthetic_taxonomy_versions(
        corpus_root,
        captured_version=captured_version,
        manifest_version=manifest_version,
        compatibility_entries=compatibility_entries,
    )

    with pytest.raises(ValueError, match=r"captured under .*" + captured_version):
        corpus_capture.verify_corpus(corpus_root)


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
                if producer == corpus_capture.TAXONOMY_PRODUCER:
                    supported_versions = LABEL_COMPATIBLE_TAXONOMY_VERSIONS
                    manifest_version = manifest["semantic_versions"][producer]
                    historically_attested = any(
                        entry["producer"] == producer
                        and entry["loader_test"] == corpus_capture.TAXONOMY_COMPATIBILITY_TEST
                        and entry["captured_version"] == captured_version
                        and entry["current_version"] in (current_version, manifest_version)
                        for entry in manifest["compatible_legacy_versions"]
                    )
                    assert captured_version in supported_versions
                    assert current_version in supported_versions
                    assert manifest_version in supported_versions
                    assert captured_version == manifest_version or historically_attested
                else:
                    assert {
                        "producer": producer,
                        "captured_version": captured_version,
                        "current_version": current_version,
                        "loader_test": corpus_capture.COMPATIBILITY_LOADER_TESTS[producer],
                    } in manifest["compatible_legacy_versions"]
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
    assert manifest["semantic_versions"][corpus_capture.TAXONOMY_PRODUCER] in (
        LABEL_COMPATIBLE_TAXONOMY_VERSIONS
    )


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
        readable_schemas = {"legacy", str(RECORD_SCHEMA_VERSION)} | {
            str(entry["captured_version"])
            for entry in manifest["compatible_legacy_versions"]
            if entry["producer"] == corpus_capture.RECORD_SCHEMA_PRODUCER
        }
        assert set(observed_shapes).issubset(readable_schemas)


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


def _smoke_source(root: Path, marker: str) -> Path:
    root.mkdir(parents=True)
    (root / "records.jsonl").write_text(json.dumps({"marker": marker}) + "\n", encoding="utf-8")
    return root


def test_smoke_regeneration_replaces_the_previous_capture(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_capture.capture_run(
        _smoke_source(tmp_path / "first", "first"),
        "smoke-0.8b",
        kind="smoke",
        corpus_root=corpus_root,
    )

    corpus_capture.replace_smoke_capture(
        _smoke_source(tmp_path / "second", "second"), "smoke-0.8b", corpus_root=corpus_root
    )

    manifest = json.loads((corpus_root / "corpus-manifest.json").read_text(encoding="utf-8"))
    assert [capture["path"] for capture in manifest["captures"]] == ["smoke-0.8b"]
    assert "second" in (corpus_root / "smoke-0.8b" / "records.jsonl").read_text(encoding="utf-8")
    assert (
        sorted(path.name for path in corpus_root.iterdir() if path.name.startswith(".smoke")) == []
    )
    corpus_capture.verify_corpus(corpus_root)


def test_smoke_regeneration_does_not_count_the_capture_it_replaces_against_the_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_root = tmp_path / "corpus"
    first = _smoke_source(tmp_path / "first", "first")
    (first / "payload.bin").write_bytes(b"x" * 6000)
    corpus_capture.capture_run(first, "smoke-0.8b", kind="smoke", corpus_root=corpus_root)
    second = _smoke_source(tmp_path / "second", "second")
    (second / "payload.bin").write_bytes(b"y" * 6000)
    monkeypatch.setattr(corpus_capture, "CORPUS_SIZE_LIMIT_BYTES", 10_000)

    corpus_capture.replace_smoke_capture(second, "smoke-0.8b", corpus_root=corpus_root)

    assert "second" in (corpus_root / "smoke-0.8b" / "records.jsonl").read_text(encoding="utf-8")
    assert [path.name for path in tmp_path.iterdir() if "retired" in path.name] == []


def test_failed_smoke_regeneration_restores_the_previous_capture(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_capture.capture_run(
        _smoke_source(tmp_path / "first", "first"),
        "smoke-0.8b",
        kind="smoke",
        corpus_root=corpus_root,
    )
    manifest_before = (corpus_root / "corpus-manifest.json").read_text(encoding="utf-8")
    unfinished = _smoke_source(tmp_path / "unfinished", "second")
    (unfinished / "records.jsonl.tmp").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="unfinished temporary file"):
        corpus_capture.replace_smoke_capture(unfinished, "smoke-0.8b", corpus_root=corpus_root)

    assert (corpus_root / "corpus-manifest.json").read_text(encoding="utf-8") == manifest_before
    assert "first" in (corpus_root / "smoke-0.8b" / "records.jsonl").read_text(encoding="utf-8")
    corpus_capture.verify_corpus(corpus_root)


@pytest.mark.parametrize("with_sidecar", [True, False])
def test_strengthened_truth_bump_stales_only_captures_with_sidecar_grades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_sidecar: bool
) -> None:
    source = _smoke_source(tmp_path / "source", "graded")
    if with_sidecar:
        (source / "strengthened-truth.jsonl").write_text(
            json.dumps({"grading_identity": {}, "strengthened_truth": {"pass": None}}) + "\n",
            encoding="utf-8",
        )
    corpus_root = tmp_path / "corpus"
    corpus_capture.capture_run(source, "smoke-0.8b", kind="smoke", corpus_root=corpus_root)
    bumped_versions = {**semantic_versions(), corpus_capture.STRENGTHENED_TRUTH_PRODUCER: 999}
    monkeypatch.setattr(corpus_capture, "semantic_versions", lambda: bumped_versions)

    if with_sidecar:
        with pytest.raises(ValueError, match="strengthened_truth_grade=999"):
            corpus_capture.verify_corpus(corpus_root)
    else:
        corpus_capture.verify_corpus(corpus_root)
