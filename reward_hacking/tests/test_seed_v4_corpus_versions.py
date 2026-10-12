"""Version and path checks over the ignored seed-v4 real-run corpus."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from reward_hacking.distill.agentic_sft import MANIFEST_VERSION, MIN_SUPPORTED_MANIFEST_VERSION
from reward_hacking.distill.hack_strategy import (
    LABEL_COMPATIBLE_TAXONOMY_VERSIONS,
    TAXONOMY_VERSION,
    _read_graded_artifact,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.harness.trace_reference import leak_trace_records
from reward_hacking.organism_rl import corpus_capture
from reward_hacking.organism_rl.corpus_capture import semantic_versions
from reward_hacking.organism_rl.evaluate import (
    GENERATION_PATH_IDENTITY_COMPATIBLE_SCHEMA_VERSIONS,
)
from reward_hacking.organism_rl.generate import (
    RECORD_SCHEMA_COMPATIBLE_VERSIONS,
    RECORD_SCHEMA_VERSION,
)
from reward_hacking.organism_rl.rollout import recorded_true_pass


@pytest.mark.parametrize("reference_field", ["trace_reference", "episode_trace_reference"])
def test_capture_preserves_run_relative_trace_reference(
    tmp_path: Path, reference_field: str
) -> None:
    source = tmp_path / "source"
    trace_relative = "traces/step-000000/synthetic.jsonl"
    trace = source / trace_relative
    trace.parent.mkdir(parents=True)
    trace.write_text('{"record": "episode_summary"}\n', encoding="utf-8")
    (source / "records.jsonl").write_text(
        json.dumps({"trace_path": str(trace), reference_field: {"trace_path": trace_relative}})
        + "\n",
        encoding="utf-8",
    )
    corpus_root = tmp_path / "corpus"
    capture = corpus_capture.capture_run(source, "relative", corpus_root=corpus_root)
    record = json.loads((capture / "records.jsonl").read_text(encoding="utf-8"))
    assert record[reference_field]["trace_path"] == trace_relative
    assert record["trace_path"] == str(capture / trace_relative)
    corpus_capture.verify_corpus(corpus_root)


def test_capture_relocates_nested_smoke_selection_trace_reference(tmp_path: Path) -> None:
    source = tmp_path / "source"
    run = source / "generate" / "synthetic-leak"
    relative_trace = "traces/step-000000/synthetic.jsonl"
    trace = run / relative_trace
    trace.parent.mkdir(parents=True)
    reference = {"trace_path": relative_trace, "locator": "before_episode_summary"}
    trace.write_text(
        json.dumps({"record": "episode_summary", "leak_evidence": {"trace_reference": reference}})
        + "\n"
    )
    records = run / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "trace_path": str(trace),
                "trace_summary": {"leak_evidence": {"trace_reference": reference}},
            }
        )
        + "\n"
    )
    selection_path = source / "sft" / "selection.json"
    selection_path.parent.mkdir()
    selection_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "source_records_file": str(records),
                        "episode_trace_reference": {"trace_path": relative_trace},
                    }
                ]
            }
        )
    )
    capture = corpus_capture.capture_run(source, "nested", corpus_root=tmp_path / "corpus")
    selection = json.loads((capture / "sft" / "selection.json").read_text())
    entry = selection["entries"][0]
    assert entry["source_records_file"] == str(
        capture / "generate" / "synthetic-leak" / "records.jsonl"
    )
    assert entry["episode_trace_reference"]["trace_path"] == relative_trace
    summary = json.loads((capture / "generate" / "synthetic-leak" / relative_trace).read_text())
    assert summary["leak_evidence"]["trace_reference"] == reference


@pytest.mark.parametrize("provenance_location", ["record", "selection"])
def test_capture_preserves_out_of_tree_source_records_provenance(
    tmp_path: Path, provenance_location: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    provenance_path = str(tmp_path / "original-run" / "generate" / "synthetic" / "records.jsonl")
    record: dict[str, object] = {"record": "synthetic"}
    if provenance_location == "record":
        record["source_records_file"] = provenance_path
    else:
        selection = source / "sft" / "selection.json"
        selection.parent.mkdir()
        selection.write_text(json.dumps({"selected": [{"source_records_file": provenance_path}]}))
    (source / "records.jsonl").write_text(json.dumps(record) + "\n")
    corpus_root = tmp_path / "corpus"
    capture = corpus_capture.capture_run(source, "provenance", corpus_root=corpus_root)
    captured = json.loads(
        (
            capture / ("records.jsonl" if provenance_location == "record" else "sft/selection.json")
        ).read_text()
    )
    entry = captured if provenance_location == "record" else captured["selected"][0]
    assert entry["source_records_file"] == provenance_path
    assert not Path(provenance_path).exists()
    corpus_capture.verify_corpus(corpus_root)


def test_capture_rejects_out_of_tree_source_records_for_episode_reference(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    trace = source / "trace.jsonl"
    trace.write_text('{"record":"episode_summary"}\n')
    (source / "records.jsonl").write_text(json.dumps({"trace_path": str(trace)}) + "\n")
    selection = source / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected": [
                    {
                        "source_records_file": str(tmp_path / "outside" / "records.jsonl"),
                        "episode_trace_reference": {"trace_path": "trace.jsonl"},
                    }
                ],
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="outside the captured run"):
        corpus_capture.capture_run(source, "invalid-reference", corpus_root=tmp_path / "corpus")


def test_capture_relocates_absolute_nested_leak_trace_reference(tmp_path: Path) -> None:
    source = tmp_path / "source"
    run = source / "generate" / "synthetic-leak"
    trace = run / "traces" / "synthetic.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text('{"record": "turn"}\n{"record": "episode_summary"}\n')
    evidence = {"trace_reference": {"trace_path": str(trace), "locator": "before_episode_summary"}}
    (run / "records.jsonl").write_text(
        json.dumps({"trace_path": str(trace), "trace_summary": {"leak_evidence": evidence}}) + "\n"
    )
    capture = corpus_capture.capture_run(source, "nested", corpus_root=tmp_path / "corpus")
    records_path = capture / "generate" / "synthetic-leak" / "records.jsonl"
    record = json.loads(records_path.read_text())
    captured_evidence = record["trace_summary"]["leak_evidence"]
    assert leak_trace_records(captured_evidence, record=record, records_path=records_path) == (
        {"record": "turn"},
    )


def test_detector_semantics_are_tracked_for_new_assessments_only(tmp_path: Path) -> None:
    producer = corpus_capture.LEAK_DETECTOR_PRODUCER
    assert semantic_versions()[producer] == 4
    source = tmp_path / "source"
    source.mkdir()
    records_path = source / "records.jsonl"
    records_path.write_text(
        json.dumps(
            {
                "family": "r2e_leak",
                "trace_summary": {"leak_assessment": {"leak_detector": "assessed"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert producer not in corpus_capture._exercised_producers(source, kind="real")  # pyright: ignore[reportPrivateUsage]
    records_path.write_text(
        json.dumps(
            {"family": "r2e_leak", "trace_summary": {"leak_assessment": {"semantic_version": 1}}}
        )
        + "\n",
        encoding="utf-8",
    )
    assert producer in corpus_capture._exercised_producers(source, kind="real")  # pyright: ignore[reportPrivateUsage]


def test_versioned_detector_capture_refuses_unattested_version(tmp_path: Path) -> None:
    producer = corpus_capture.LEAK_DETECTOR_PRODUCER
    current_version = semantic_versions()[producer]
    source = tmp_path / "source"
    source.mkdir()
    (source / "records.jsonl").write_text(
        json.dumps(
            {
                "family": "r2e_leak",
                "trace_summary": {"leak_assessment": {"semantic_version": current_version}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_root = tmp_path / "corpus"
    corpus_capture.capture_run(source, "synthetic", corpus_root=corpus_root)
    manifest_path = corpus_root / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["captures"][0]["semantic_versions"][producer] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=r"leak_detector\.semantic_version=0"):
        corpus_capture.verify_corpus(corpus_root)


@pytest.mark.parametrize("captured_version", [1, 2])
def test_legacy_detector_captures_verify_with_rescore_compatibility(
    tmp_path: Path, captured_version: int
) -> None:
    producer = corpus_capture.LEAK_DETECTOR_PRODUCER
    current_version = semantic_versions()[producer]
    source = tmp_path / "source"
    source.mkdir()
    (source / "records.jsonl").write_text(
        json.dumps(
            {
                "family": "r2e_leak",
                "trace_summary": {"leak_assessment": {"semantic_version": current_version}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_root = tmp_path / "corpus"
    capture_run_path = corpus_capture.capture_run(source, "synthetic", corpus_root=corpus_root)
    records_path = capture_run_path / "records.jsonl"
    record = json.loads(records_path.read_text(encoding="utf-8"))
    record["trace_summary"]["leak_assessment"]["semantic_version"] = captured_version
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    manifest_path = corpus_root / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = manifest["captures"][0]
    capture["semantic_versions"][producer] = captured_version
    capture["file_sha256"] = corpus_capture._capture_file_sha256(  # pyright: ignore[reportPrivateUsage]
        capture_run_path
    )
    manifest["semantic_versions"][producer] = captured_version
    manifest["compatible_legacy_versions"] = [
        {
            "producer": producer,
            "captured_version": captured_version,
            "current_version": current_version,
            "loader_test": corpus_capture.COMPATIBILITY_LOADER_TESTS[producer],
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    corpus_capture.verify_corpus(corpus_root)


def test_only_unversioned_legacy_manifests_can_defer_detector_semantics() -> None:
    manifest: dict[str, object] = {
        "semantic_versions": {},
        "deferred_semantic_versions": list(corpus_capture.LEGACY_DEFERRED_SEMANTIC_VERSIONS),
    }
    corpus_capture._check_manifest_version_notes(manifest)  # pyright: ignore[reportPrivateUsage]
    manifest["semantic_versions"] = {corpus_capture.LEAK_DETECTOR_PRODUCER: 1}
    with pytest.raises(ValueError, match="deferred semantic version note"):
        corpus_capture._check_manifest_version_notes(manifest)  # pyright: ignore[reportPrivateUsage]


def test_detector_record_version_must_match_capture_version(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "records.jsonl").write_text(
        json.dumps({"trace_summary": {"leak_assessment": {"semantic_version": 999}}}) + "\n",
        encoding="utf-8",
    )
    corpus_root = tmp_path / "corpus"
    with pytest.raises(ValueError, match="detector assessment version differs"):
        corpus_capture.capture_run(source, "synthetic", corpus_root=corpus_root)


@pytest.mark.parametrize("version", [None, True, 0, "1"])
def test_assessed_detector_version_requires_positive_integer(
    tmp_path: Path, version: object
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "records.jsonl").write_text(
        json.dumps(
            {
                "trace_summary": {
                    "leak_assessment": {"leak_detector": "assessed", "semantic_version": version}
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="semantic version must be a positive integer"):
        corpus_capture._exercised_producers(source, kind="real")  # pyright: ignore[reportPrivateUsage]


def test_compact_records_have_a_distinct_schema_and_keep_legacy_support() -> None:
    assert RECORD_SCHEMA_VERSION == 6
    assert frozenset({1, 2, 3, 4, 5, 6}) == RECORD_SCHEMA_COMPATIBLE_VERSIONS


@pytest.mark.parametrize("truth_version", [1, 2, 999])
def test_regraded_truth_cannot_use_legacy_missing_producer_compatibility(
    tmp_path: Path, truth_version: int
) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    (capture_root / "records.jsonl").write_text(
        json.dumps(
            {
                "family": "ilcb",
                "record_schema": 3,
                "truth_regrade": {
                    "producer": corpus_capture.ILCB_TRUE_CHECK_PRODUCER,
                    "semantic_version": truth_version,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"captured under .*true_check=None"):
        corpus_capture._check_semantic_version_table(  # pyright: ignore[reportPrivateUsage]
            {},
            corpus_capture._SemanticVersionCheckContext(  # pyright: ignore[reportPrivateUsage]
                capture_label="synthetic legacy capture",
                exercised_producers=frozenset({corpus_capture.ILCB_TRUE_CHECK_PRODUCER}),
                capture_root=capture_root,
            ),
        )


def test_unmeasured_primary_precedence_has_a_compatible_taxonomy_version() -> None:
    assert TAXONOMY_VERSION == "hack_strategy_v11"
    assert {"hack_strategy_v5", "hack_strategy_v6", "hack_strategy_v7"}.issubset(
        LABEL_COMPATIBLE_TAXONOMY_VERSIONS
    )


@pytest.mark.corpus
def test_legacy_oracle_errors_load_as_unmeasured_truth(seed_v4_corpus: Path) -> None:
    legacy_error_count = 0
    ilcb_error_count = 0
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
                if record.get("family") == "ilcb":
                    ilcb_error_count += 1
    assert legacy_error_count > 0
    assert ilcb_error_count > 0


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


@pytest.mark.parametrize("with_attestation", [True, False])
def test_appending_current_capture_preserves_older_attested_capture(
    tmp_path: Path, *, with_attestation: bool
) -> None:
    corpus_root = tmp_path / "corpus"
    _capture_minimal_run(tmp_path / "old-source", corpus_root)
    manifest_path = corpus_root / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    producer = corpus_capture.RECORD_SCHEMA_PRODUCER
    manifest["semantic_versions"][producer] = RECORD_SCHEMA_VERSION - 1
    manifest["captures"][0]["semantic_versions"][producer] = (
        1 if with_attestation else RECORD_SCHEMA_VERSION - 1
    )
    manifest["compatible_legacy_versions"] = [
        {
            "producer": producer,
            "captured_version": 1,
            "current_version": RECORD_SCHEMA_VERSION - 1,
            "loader_test": corpus_capture.RECORD_SCHEMA_COMPATIBILITY_TEST,
        }
    ]
    if not with_attestation:
        manifest["compatible_legacy_versions"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    corpus_capture.verify_corpus(corpus_root)

    source = _smoke_source(tmp_path / "new-source", "current")
    corpus_capture.capture_run(source, "current", corpus_root=corpus_root)

    verified = corpus_capture.verify_corpus(corpus_root)
    assert len(cast("list[object]", verified["captures"])) == 2
    assert (
        cast("Mapping[str, object]", verified["semantic_versions"])[producer]
        == RECORD_SCHEMA_VERSION
    )


def test_supported_capture_versions_do_not_require_manifest_attestations(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _capture_minimal_run(tmp_path / "source", corpus_root)
    _set_synthetic_taxonomy_versions(
        corpus_root,
        captured_version="hack_strategy_v5",
        manifest_version="hack_strategy_v999",
        compatibility_entries=[],
    )

    corpus_capture.verify_corpus(corpus_root)


def test_appending_current_capture_preserves_legacy_ilcb_truth(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    source = tmp_path / "legacy-source"
    source.mkdir()
    (source / "records.jsonl").write_text(
        json.dumps({"family": "ilcb", "record_schema": 3}) + "\n", encoding="utf-8"
    )
    corpus_capture.capture_run(source, "legacy", corpus_root=corpus_root)
    manifest_path = corpus_root / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    producer = corpus_capture.ILCB_TRUE_CHECK_PRODUCER
    del manifest["captures"][0]["semantic_versions"][producer]
    del manifest["semantic_versions"][producer]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    corpus_capture.verify_corpus(corpus_root)

    corpus_capture.capture_run(
        _smoke_source(tmp_path / "current-source", "current"), "current", corpus_root=corpus_root
    )

    corpus_capture.verify_corpus(corpus_root)


def test_compatible_detector_capture_still_checks_record_version(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    source = tmp_path / "source"
    source.mkdir()
    producer = corpus_capture.LEAK_DETECTOR_PRODUCER
    (source / "records.jsonl").write_text(
        json.dumps(
            {
                "trace_summary": {
                    "leak_assessment": {"semantic_version": semantic_versions()[producer]}
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_capture.capture_run(source, "detector", corpus_root=corpus_root)
    manifest_path = corpus_root / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["captures"][0]["semantic_versions"][producer] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="detector assessment version differs"):
        corpus_capture.verify_corpus(corpus_root)


def test_legacy_compatibility_requires_a_declared_loader_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_root = tmp_path / "corpus"
    _capture_minimal_run(tmp_path / "source", corpus_root)
    _set_synthetic_taxonomy_versions(
        corpus_root,
        captured_version="hack_strategy_v5",
        manifest_version=TAXONOMY_VERSION,
        compatibility_entries=[_taxonomy_compatibility_entry("hack_strategy_v5", TAXONOMY_VERSION)],
    )
    monkeypatch.delitem(corpus_capture.COMPATIBILITY_LOADER_TESTS, corpus_capture.TAXONOMY_PRODUCER)

    with pytest.raises(ValueError, match="hack_strategy_v5"):
        corpus_capture.verify_corpus(corpus_root)


@pytest.mark.parametrize("existing_corpus", [True, False])
def test_failed_capture_restores_manifest_and_directory(
    tmp_path: Path, *, existing_corpus: bool
) -> None:
    corpus_root = tmp_path / "corpus"
    if existing_corpus:
        _capture_minimal_run(tmp_path / "old-source", corpus_root)
    before = {
        str(path.relative_to(corpus_root)): path.read_bytes()
        for path in corpus_root.rglob("*")
        if path.is_file()
    }
    directories_before = {
        str(path.relative_to(corpus_root)) for path in corpus_root.rglob("*") if path.is_dir()
    }
    source = tmp_path / "invalid-source"
    source.mkdir()
    (source / "records.jsonl").write_text(
        json.dumps({"trace_summary": {"leak_assessment": {"semantic_version": 999}}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="detector assessment version differs"):
        corpus_capture.capture_run(source, "invalid", corpus_root=corpus_root)

    assert not (corpus_root / "nine-b" / "invalid").exists()
    assert {
        str(path.relative_to(corpus_root)): path.read_bytes()
        for path in corpus_root.rglob("*")
        if path.is_file()
    } == before
    assert {
        str(path.relative_to(corpus_root)) for path in corpus_root.rglob("*") if path.is_dir()
    } == directories_before
    assert corpus_root.exists() == existing_corpus
    if existing_corpus:
        corpus_capture.verify_corpus(corpus_root)


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
            "hack_strategy_v7",
            [_taxonomy_compatibility_entry("hack_strategy_v5", "hack_strategy_v7")],
        ),
        (
            "hack_strategy_v7",
            "hack_strategy_v8",
            [_taxonomy_compatibility_entry("hack_strategy_v7", "hack_strategy_v8")],
        ),
        (
            "hack_strategy_v8",
            "hack_strategy_v9",
            [_taxonomy_compatibility_entry("hack_strategy_v8", "hack_strategy_v9")],
        ),
        (
            "hack_strategy_v9",
            "hack_strategy_v10",
            [_taxonomy_compatibility_entry("hack_strategy_v9", "hack_strategy_v10")],
        ),
    ],
)
def test_taxonomy_v9_verifies_captures_with_prior_compatibility_evidence(
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
        ("hack_strategy_v0", "hack_strategy_v7", []),
        (
            "hack_strategy_v999",
            "hack_strategy_v7",
            [_taxonomy_compatibility_entry("hack_strategy_v999", "hack_strategy_v7")],
        ),
    ],
)
def test_taxonomy_rejects_unsupported_versions_even_with_attestation(
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


@pytest.mark.parametrize(("kind", "stale"), [("smoke", True), ("eval", False)])
def test_smoke_plan_bump_stales_only_smoke_captures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, *, stale: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "records.jsonl").write_text(
        json.dumps({"variant": "original"}) + "\n", encoding="utf-8"
    )
    corpus_root = tmp_path / "corpus"
    corpus_capture.capture_run(
        source, "smoke-0.8b" if kind == "smoke" else "synthetic", kind=kind, corpus_root=corpus_root
    )
    bumped_versions = {**semantic_versions(), corpus_capture.SMOKE_PLAN_PRODUCER: "smoke_plan_next"}
    monkeypatch.setattr(corpus_capture, "semantic_versions", lambda: bumped_versions)

    if stale:
        with pytest.raises(ValueError, match="smoke_plan_next"):
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
        exercised_producers = corpus_capture._exercised_producers(
            seed_v4_corpus / capture["path"], kind=capture["kind"]
        )
        for producer in exercised_producers:
            captured_version = captured_versions.get(producer)
            current_version = semantic_versions()[producer]
            if captured_version != current_version:
                if producer == corpus_capture.TAXONOMY_PRODUCER:
                    assert captured_version in LABEL_COMPATIBLE_TAXONOMY_VERSIONS
                    assert current_version in LABEL_COMPATIBLE_TAXONOMY_VERSIONS
                elif producer == corpus_capture.LEAK_DETECTOR_PRODUCER:
                    assert (
                        captured_version
                        in corpus_capture.LEAK_DETECTOR_COMPATIBLE_SEMANTIC_VERSIONS
                    )
                    assert (
                        current_version in corpus_capture.LEAK_DETECTOR_COMPATIBLE_SEMANTIC_VERSIONS
                    )
                elif producer == corpus_capture.RECORD_SCHEMA_PRODUCER:
                    assert captured_version in RECORD_SCHEMA_COMPATIBLE_VERSIONS
                    assert current_version in RECORD_SCHEMA_COMPATIBLE_VERSIONS
                elif producer == corpus_capture.GENERATION_IDENTITY_PRODUCER:
                    assert captured_version in GENERATION_PATH_IDENTITY_COMPATIBLE_SCHEMA_VERSIONS
                    assert current_version in GENERATION_PATH_IDENTITY_COMPATIBLE_SCHEMA_VERSIONS
                elif producer == corpus_capture.SFT_MANIFEST_PRODUCER:
                    assert isinstance(current_version, int)
                    assert MIN_SUPPORTED_MANIFEST_VERSION <= captured_version <= MANIFEST_VERSION
                    assert MIN_SUPPORTED_MANIFEST_VERSION <= current_version <= MANIFEST_VERSION
                elif producer == corpus_capture.ILCB_TRUE_CHECK_PRODUCER:
                    assert captured_version is None
                    assert not corpus_capture._has_record_schema_at_least(
                        seed_v4_corpus / capture["path"], 4
                    )
                else:
                    pytest.fail(f"unsupported captured producer version: {producer}")
    corpus_capture._check_manifest_version_notes(manifest)  # pyright: ignore[reportPrivateUsage]
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
        readable_schemas = {"legacy"} | {
            str(version) for version in RECORD_SCHEMA_COMPATIBLE_VERSIONS
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


def test_smoke_capture_keeps_probe_records_and_sft_leak_coverage(tmp_path: Path) -> None:
    source = _smoke_source(tmp_path / "source", "synthetic")
    for stage in ("base", "v3"):
        records = source / "probe" / "f2-orig" / stage / "records.jsonl"
        records.parent.mkdir(parents=True)
        records.write_text('{"marker": "synthetic"}\n', encoding="utf-8")
    coverage = source / "sft" / "leak-coverage.json"
    coverage.parent.mkdir()
    coverage.write_text('{"selected": 2, "excluded_by_reason": {}}\n', encoding="utf-8")
    destination = corpus_capture.capture_run(
        source, "smoke-0.8b", kind="smoke", corpus_root=tmp_path / "corpus"
    )
    for stage in ("base", "v3"):
        relative_path = Path("probe") / "f2-orig" / stage / "records.jsonl"
        assert (destination / relative_path).read_bytes() == (source / relative_path).read_bytes()
    assert json.loads(
        (destination / "sft/leak-coverage.json").read_text(encoding="utf-8")
    ) == json.loads(coverage.read_text(encoding="utf-8"))


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
