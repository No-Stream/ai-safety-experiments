"""Lossless conversion and interrupted transaction regression tests."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest

import reward_hacking.organism_rl.compact_run as compaction
from reward_hacking.distill.label_refresh import load_derived_labels, refresh_labels
from reward_hacking.evidence_store import append_jsonl, read_jsonl
from reward_hacking.organism_rl.compact_run import compact_run
from reward_hacking.organism_rl.generate import (
    _existing_records,  # pyright: ignore[reportPrivateUsage]
)


def _run(directory: Path) -> tuple[Path, list[dict[str, object]]]:
    directory.mkdir()
    trace = directory / "episode.jsonl"
    rows: list[dict[str, object]] = [
        {
            "record": "turn",
            "command": "printf synthetic",
            "stdout": "synthetic " * 2000,
            "exit_code": 0,
            "stderr": "",
        },
        {"record": "episode_summary", "task_id": "synthetic", "true_pass": False},
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    record = {
        "task_id": "synthetic",
        "trace_path": str(trace),
        "trace_summary": rows[-1],
        "key": {"task_id": "synthetic", "sample_index": 0},
        "sample_index": 0,
        "run_provenance_sha256": "synthetic-provenance",
    }
    (directory / "records.jsonl").write_text(json.dumps(record) + "\n")
    return trace, rows


def test_default_dry_run_preserves_every_original(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    _run(directory)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    report = compact_run(directory)
    assert report["applied"] is False
    after_bytes = report["after_bytes"]
    before_bytes = report["before_bytes"]
    assert isinstance(after_bytes, int)
    assert isinstance(before_bytes, int)
    assert after_bytes < before_bytes
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original


def test_compact_run_records_unverifiable_selected_task_provenance_without_rewriting(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "run"
    _run(directory)
    provenance_path = directory / "provenance.json"
    sidecar_name = "provenance.selected-tasks.json.gz"
    compact_task = {"task_id": "compact-synthetic", "family": "aisi", "variant": "oneoff"}
    selected_task = {
        **compact_task,
        "task_id": "sidecar-synthetic",
        "task_definition_content": {"body": "Synthetic task body."},
    }
    provenance = {
        "schema_version": 3,
        "provenance_sha256": "synthetic-provenance-digest",
        "provenance": {
            "selected_tasks": [compact_task],
            "selected_tasks_reference": {"schema_version": 1, "path": sidecar_name},
        },
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    sidecar_path = directory / sidecar_name
    sidecar_path.write_bytes(
        gzip.compress(json.dumps([selected_task], sort_keys=True, separators=(",", ":")).encode())
    )
    original_provenance = provenance_path.read_bytes()
    original_sidecar = sidecar_path.read_bytes()

    report = compact_run(directory)

    assert report["provenance_limitations"] == [
        "historical task definition unverifiable: semantic identity mismatch"
    ]
    assert provenance_path.read_bytes() == original_provenance
    assert sidecar_path.read_bytes() == original_sidecar


def test_dry_run_uses_private_staging_and_preserves_an_existing_stage(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    _run(directory)
    stale_stage = directory / ".compact-run-stage"
    stale_stage.mkdir()
    sentinel = stale_stage / "owner-staged-record.jsonl"
    sentinel.write_text('{"preserve":"this staged data"}\n', encoding="utf-8")
    original_stage = {
        path.relative_to(stale_stage): path.read_bytes()
        for path in stale_stage.rglob("*")
        if path.is_file()
    }

    report = compact_run(directory)

    assert report["applied"] is False
    assert sentinel.read_bytes() == original_stage[Path("owner-staged-record.jsonl")]
    assert {
        path.relative_to(stale_stage): path.read_bytes()
        for path in stale_stage.rglob("*")
        if path.is_file()
    } == original_stage
    assert not list(directory.glob(".compact-run-dry-run-*"))


@pytest.mark.parametrize("stage_metadata", [None, {"format_version": 1}])
def test_apply_uses_fresh_staging_and_preserves_an_existing_stage(
    tmp_path: Path, stage_metadata: dict[str, int] | None
) -> None:
    directory = tmp_path / "run"
    _run(directory)
    stale_stage = directory / ".compact-run-stage"
    stale_stage.mkdir()
    sentinel = stale_stage / "stale.json"
    sentinel.write_text('{"owner":"data"}\n', encoding="utf-8")
    if stage_metadata is not None:
        (stale_stage / ".compact-run-stage.json").write_text(
            json.dumps(stage_metadata), encoding="utf-8"
        )
    original_stage = {
        path.relative_to(stale_stage): path.read_bytes()
        for path in stale_stage.rglob("*")
        if path.is_file()
    }

    report = compact_run(directory, apply=True)

    assert report["applied"] is True
    assert {
        path.relative_to(stale_stage): path.read_bytes()
        for path in stale_stage.rglob("*")
        if path.is_file()
    } == original_stage
    assert not (directory / "stale.json").exists()
    assert not list(directory.glob(".compact-run-stage-work-*"))


def test_apply_reconstructs_and_second_apply_is_idempotent(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    trace, rows = _run(directory)
    record = json.loads((directory / "records.jsonl").read_text())
    report = compact_run(directory, apply=True)
    assert report["applied"] is True
    assert read_jsonl(trace) == rows
    assert read_jsonl(directory / "records.jsonl") == [record]
    first = {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }
    compact_run(directory, apply=True)
    assert {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    } == first


@pytest.mark.parametrize(("field", "value"), [("stdout", "corrupted output"), ("exit_code", False)])
def test_failed_verification_never_replaces_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    directory = tmp_path / "run"
    _run(directory)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}

    def corrupt(path: Path, records: list[dict[str, object]], *, append: bool = True) -> None:
        damaged = [dict(record) for record in records]
        damaged[0][field] = value
        append_jsonl(path, damaged, append=append)

    monkeypatch.setattr("reward_hacking.organism_rl.compact_run.append_jsonl", corrupt)
    with pytest.raises(ValueError, match="reconstructed evidence equality"):
        compact_run(directory, apply=True)
    assert {name: (directory / name).read_bytes() for name in original} == original


def test_interrupted_apply_restores_previous_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    _run(directory)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    real_replace = Path.replace
    replacements = 0

    def interrupt(path: Path, destination: Path) -> Path:
        nonlocal replacements
        if any(part.startswith(".compact-run-stage-work-") for part in path.parts) and not any(
            part.startswith(".compact-run-stage-work-") for part in destination.parts
        ):
            replacements += 1
            if replacements == 2:
                raise KeyboardInterrupt("interrupted apply")
        return real_replace(path, destination)

    monkeypatch.setattr(Path, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        compact_run(directory, apply=True)
    assert (directory / ".compact-run-journal.json").exists()
    monkeypatch.setattr(Path, "replace", real_replace)
    compact_run(directory, apply=True)
    for name, content in original.items():
        assert read_jsonl(directory / name) == [json.loads(line) for line in content.splitlines()]
    assert not (directory / ".compact-run-journal.json").exists()


@pytest.mark.parametrize("include_docker_rows", [False, True])
@pytest.mark.parametrize("relocation", ["none", "copied", "moved"])
def test_leak_components_dry_run_never_rewrites_docker_evidence(
    tmp_path: Path, include_docker_rows: bool, relocation: str
) -> None:
    directory = tmp_path / "run"
    trace, rows = _run(directory)
    docker = directory / "docker.jsonl"
    docker_rows = (
        [{"command": "synthetic", "stdout": "synthetic " * 1000}] if include_docker_rows else []
    )
    docker.write_text("".join(json.dumps(row) + "\n" for row in docker_rows))
    digest = hashlib.sha256(b"synthetic").hexdigest()
    manifest = {
        "schema_version": 1,
        "builder_version": "synthetic",
        "channels": [],
        "fix": {
            "commit_id": "a" * 40,
            "tree_id": "b" * 40,
            "blob_ids": {},
            "post_fix_file_sha256s": {},
            "modified_file_paths": [],
            "base_date": 1,
            "fix_date": 2,
        },
        "fingerprint_lines": [digest],
        "channel_details": {},
        "template_fingerprint": digest,
    }
    manifest_json = json.dumps(manifest)
    summary = {
        **rows[-1],
        "docker_evidence_path": str(docker),
        "manifest_sha256": hashlib.sha256(manifest_json.encode()).hexdigest(),
        "submitted_artifact_sha256": hashlib.sha256(b"").hexdigest(),
        "leak_evidence": {
            "trace_records": rows[:-1],
            "docker_evidence": docker_rows,
            "manifest": manifest,
            "manifest_raw_json": manifest_json,
            "final_patch": "",
        },
    }
    rows[-1] = summary
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    record = {"task_id": "synthetic", "trace_path": str(trace), "trace_summary": summary}
    (directory / "records.jsonl").write_text(json.dumps(record) + "\n")
    source = directory
    source_bytes = {path.name: path.read_bytes() for path in source.iterdir()}
    if relocation != "none":
        directory = tmp_path / "copy"
        shutil.copytree(source, directory)
        trace = directory / trace.name
        docker = directory / docker.name
        if relocation == "moved":
            shutil.rmtree(source)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    compact_run(directory)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original
    compact_run(directory, apply=True)
    assert read_jsonl(trace) == rows
    assert read_jsonl(directory / "records.jsonl") == [record]
    assert read_jsonl(docker) == docker_rows
    assert len(list(directory.glob("*.leak-evidence.*.json.gz"))) == 1
    if relocation == "copied":
        assert {path.name: path.read_bytes() for path in source.iterdir()} == source_bytes


def test_recursive_runs_and_provenance_roundtrip(tmp_path: Path) -> None:
    directory = tmp_path / "wave"
    directory.mkdir()
    for name in ("synthetic-one", "synthetic-two"):
        child = directory / name
        trace, rows = _run(child)
        provenance = {
            "selected_tasks": [
                {
                    "task_id": name,
                    "family": "ilcb",
                    "variant": "original",
                    "task_definition_content": {"source": "synthetic " * 1000},
                }
            ]
        }
        if name == "synthetic-two":
            provenance = {"provenance": provenance}
        (child / "provenance.json").write_text(json.dumps(provenance))
        assert trace.exists()
        assert rows
    report = compact_run(directory, apply=True)
    assert report["trace_files"] == 2
    assert report["provenance_files"] == 2
    compact_run(directory, apply=True)


def test_provenance_wrapper_metadata_corruption_refuses_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    _run(directory)
    provenance = {"completed": 0, "provenance": {"selected_tasks": [{"task_id": "synthetic"}]}}
    (directory / "provenance.json").write_text(json.dumps(provenance))
    write_json = compaction._write_json  # pyright: ignore[reportPrivateUsage]

    def corrupt_wrapper(path: Path, value: dict[str, object]) -> None:
        if path.name == "provenance.json" and any(
            part.startswith(".compact-run-") for part in path.parts
        ):
            value = {**value, "completed": False}
        write_json(path, value)

    monkeypatch.setattr(compaction, "_write_json", corrupt_wrapper)
    with pytest.raises(ValueError, match="reconstructed evidence equality"):
        compact_run(directory)
    assert json.loads((directory / "provenance.json").read_text()) == provenance


def test_existing_derived_labels_remain_valid_after_conversion(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    _run(directory)
    original = refresh_labels(directory / "records.jsonl", tmp_path / "labels")
    shutil.copytree(original.labels_path.parent, directory / "labels")
    before = read_jsonl(directory / "labels" / "labels.jsonl")
    compact_run(directory, apply=True)
    after = list(
        load_derived_labels(
            directory / "records.jsonl", directory / "labels" / "labels.jsonl"
        ).values()
    )
    assert [row["hack_strategy"] for row in after] == [row["hack_strategy"] for row in before]
    assert len(list((directory / "labels").glob("labels.previous-bindings-*.json.gz"))) == 1


def test_generator_resume_reads_converted_records(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    _run(directory)
    original = read_jsonl(directory / "records.jsonl")
    compact_run(directory, apply=True)
    assert (directory / "records.jsonl").read_bytes().startswith(b"{")
    restored, torn = _existing_records(
        directory / "records.jsonl", accepted_provenance_sha256={"synthetic-provenance"}
    )
    assert restored == original
    assert torn == 0


def test_orphan_original_backup_is_preserved(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    _run(directory)
    backup = directory / ".compact-run-backup"
    backup.mkdir()
    (backup / "original.jsonl").write_text("original evidence")
    with pytest.raises(ValueError, match="manual recovery"):
        compact_run(directory, apply=True)
    assert (backup / "original.jsonl").read_text() == "original evidence"


def test_compaction_preserves_an_earlier_provenance_snapshot(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    _run(directory)
    provenance_path = directory / "provenance.json"
    task = {
        "task_id": "synthetic",
        "family": "ilcb",
        "variant": "original",
        "task_definition_content": {"source": "first synthetic"},
    }
    provenance_path.write_text(json.dumps({"selected_tasks": [task]}))
    compact_run(directory, apply=True)
    sidecar = directory / "provenance.selected-tasks.json.gz"
    original = sidecar.read_bytes()
    task["task_definition_content"] = {"source": "second synthetic"}
    provenance_path.write_text(json.dumps({"selected_tasks": [task]}))
    compact_run(directory, apply=True)
    assert sidecar.read_bytes() == original
    assert len(list(directory.glob("provenance.selected-tasks*.json.gz"))) == 2


@pytest.mark.parametrize("episode_shaped", [False, True])
def test_saved_workspace_jsonl_stays_byte_identical(tmp_path: Path, episode_shaped: bool) -> None:
    directory = tmp_path / "run"
    _run(directory)
    workspace = directory / "workspace"
    workspace.mkdir()
    original = b'{"record": "turn", "stdout": "synthetic grading input"}\n'
    if episode_shaped:
        original = (directory / "records.jsonl").read_bytes()
    for name in ("events.jsonl", "records.jsonl"):
        (workspace / name).write_bytes(original)
    compact_run(directory, apply=True)
    for name in ("events.jsonl", "records.jsonl"):
        assert (workspace / name).read_bytes() == original


def test_runtime_orphan_and_torn_trace_preserve_evidence(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    _run(directory)
    traces = directory / "traces"
    traces.mkdir()
    row = {"record": "turn", "reasoning": "synthetic " * 1000}
    original = json.dumps(row).encode() + b"\n"
    orphan = traces / "orphan.jsonl"
    orphan.write_bytes(original)
    archived = traces / "interrupted.jsonl.torn-synthetic"
    interrupted_bytes = original + b'{"record":"turn","partial":'
    archived.write_bytes(interrupted_bytes)
    workspace = traces / "workspace"
    workspace.mkdir()
    (workspace / "events.jsonl").write_bytes(original)
    report = compact_run(directory, apply=True)
    assert report["orphan_files"] == 2
    assert report["trace_files"] == 2
    assert read_jsonl(orphan) == [row]
    assert gzip.decompress(archived.read_bytes()) == interrupted_bytes
    assert read_jsonl(archived, recover_partial=True) == [row]
    assert (workspace / "events.jsonl").read_bytes() == original
    compact_run(directory, apply=True)
    assert gzip.decompress(archived.read_bytes()) == interrupted_bytes


@pytest.mark.parametrize("truncate_staged_trace", [False, True])
def test_interrupted_staging_is_discarded_before_fresh_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, truncate_staged_trace: bool
) -> None:
    directory = tmp_path / "run"
    trace, rows = _run(directory)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}

    def interrupt(*_args: object) -> None:
        raise KeyboardInterrupt("stop after trace staging")

    monkeypatch.setattr(compaction, "_verify_behavior", interrupt)
    with pytest.raises(KeyboardInterrupt):
        compact_run(directory, apply=True)
    monkeypatch.undo()
    (stage,) = directory.glob(".compact-run-stage-work-*")
    assert stage.is_dir()
    assert {name: (directory / name).read_bytes() for name in original} == original
    (stage / "records.jsonl").write_bytes(b'{"torn":')
    (stage / "unrelated.json").write_text('{"stale":"synthetic"}\n', encoding="utf-8")
    if truncate_staged_trace:
        staged_trace = stage / trace.name
        staged_trace.write_bytes(staged_trace.read_bytes()[:-8])
    real_stage_trace = compaction._stage_trace  # pyright: ignore[reportPrivateUsage]
    rebuilt = 0

    def count_rebuild(
        path: Path,
        output: Path,
        run_directory: Path,
        staging_directory: Path,
        *,
        original: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        nonlocal rebuilt
        rebuilt += 1
        return real_stage_trace(path, output, run_directory, staging_directory, original=original)

    monkeypatch.setattr(compaction, "_stage_trace", count_rebuild)
    compact_run(directory, apply=True)
    assert rebuilt == 1
    assert not (directory / "unrelated.json").exists()
    assert not list(directory.glob(".compact-run-stage-work-*"))
    assert read_jsonl(trace) == rows


def test_recursive_labels_use_nearest_matching_cell(tmp_path: Path) -> None:
    directory = tmp_path / "wave"
    directory.mkdir()
    first = directory / "first"
    second = directory / "second"
    _run(first)
    _run(second)
    labels = refresh_labels(first / "records.jsonl", tmp_path / "derived-labels")
    shutil.copytree(labels.labels_path.parent, first / "labels")
    compact_run(directory, apply=True)
    loaded = load_derived_labels(first / "records.jsonl", first / "labels" / "labels.jsonl")
    assert len(loaded) == 1


@pytest.mark.parametrize("keep_source", [False, True])
def test_copied_or_moved_legacy_run_needs_no_metadata_rewrite(
    tmp_path: Path, keep_source: bool
) -> None:
    source = tmp_path / "source"
    _, rows = _run(source)
    source_bytes = {path.name: path.read_bytes() for path in source.iterdir()}
    copied = tmp_path / "copy"
    shutil.copytree(source, copied)
    if not keep_source:
        shutil.rmtree(source)
    compact_run(copied, apply=True)
    assert read_jsonl(copied / "episode.jsonl") == rows
    assert read_jsonl(copied / "records.jsonl") == [json.loads(source_bytes["records.jsonl"])]
    if keep_source:
        assert {path.name: path.read_bytes() for path in source.iterdir()} == source_bytes


@pytest.mark.parametrize("interrupt_apply", [False, True])
def test_obsolete_sidecars_are_deleted_transactionally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt_apply: bool
) -> None:
    directory = tmp_path / "run"
    trace, rows = _run(directory)
    orphan = directory / f"episode.leak-evidence.{'a' * 64}.json.gz"
    payload = gzip.compress(b'{"obsolete":"synthetic"}', mtime=0)
    orphan.write_bytes(payload)
    if interrupt_apply:
        real_replace = Path.replace

        def interrupt(path: Path, destination: Path) -> Path:
            result = real_replace(path, destination)
            if path == orphan and ".compact-run-backup" in destination.parts:
                raise KeyboardInterrupt("stop after backing up obsolete sidecar")
            return result

        monkeypatch.setattr(Path, "replace", interrupt)
        with pytest.raises(KeyboardInterrupt):
            compact_run(directory, apply=True)
        assert (directory / ".compact-run-backup" / orphan.name).read_bytes() == payload
        monkeypatch.undo()
    report = compact_run(directory, apply=True)
    assert report["obsolete_sidecars"] == 1
    assert not orphan.exists()
    assert read_jsonl(trace) == rows


def test_torn_tail_and_captured_labels_keep_referenced_sidecars(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    _run(directory)
    traces = directory / "traces"
    traces.mkdir()
    names = [f"episode.leak-evidence.{letter * 64}.json.gz" for letter in ("a", "b")]
    for name in names:
        (traces / name).write_bytes(gzip.compress(b'{"retained":"synthetic"}', mtime=0))
    (traces / "interrupted.jsonl.torn-synthetic").write_bytes(
        json.dumps({"record": "turn"}).encode()
        + b'\n{"evidence_reference":{"path":"'
        + names[0].encode()
    )
    (directory / "labels-captured.jsonl").write_text(
        json.dumps({"captured": {"evidence_reference": {"path": names[1]}}}) + "\n"
    )
    compact_run(directory, apply=True)
    assert all((traces / name).is_file() for name in names)


def test_privacy_refusal_precedes_every_staging_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    _run(directory)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}

    def refuse(path: Path) -> None:
        assert path == directory / ".compact-run-stage" / "records.jsonl"
        raise ValueError("tracked trace privacy refusal")

    monkeypatch.setattr(compaction, "refuse_tracked_trace_path", refuse)
    with pytest.raises(ValueError, match="privacy refusal"):
        compact_run(directory, apply=True)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original
