"""Verify and atomically apply lossless compaction to a completed episode run."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import logging
import os
import shutil
import tempfile
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reward_hacking.distill.hack_strategy import label_episode
from reward_hacking.distill.label_refresh import (
    LABEL_REFRESH_FORMAT_VERSION,
    derived_label_join_key,
)
from reward_hacking.harness.acceptance_evidence import assess_acceptance_fixture
from reward_hacking.harness.evidence_store import (
    append_jsonl,
    compact_record,
    hydrate_record,
    iter_jsonl,
    json_value_equal,
    read_jsonl,
)
from reward_hacking.organism_rl.generate_provenance import (
    SELECTED_TASKS_RECOVERY_LIMITATION_KEY,
    compact_provenance,
    hydrate_provenance,
)
from reward_hacking.trace import refuse_tracked_trace_path

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
_STAGE = ".compact-run-stage"
_PRESERVED_STAGE = ".compact-run-stage-preserved"
_APPLY_STAGE_PREFIX = ".compact-run-stage-work-"
_BACKUP = ".compact-run-backup"
_JOURNAL = ".compact-run-journal.json"
_WORKSPACE_DIRECTORIES = {"workspace", "workspaces", "graded-workspace", "graded-workspaces"}


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_ancestors(path: Path, root: Path) -> None:
    while path.is_relative_to(root):
        _sync_directory(path)
        if path == root:
            break
        path = path.parent


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, separators=(",", ":"), sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _cleanup(directory: Path, stage_directory: str) -> None:
    for name in (stage_directory, _BACKUP):
        path = directory / name
        if path.exists():
            shutil.rmtree(path)
            _sync_directory(directory)
    (directory / _JOURNAL).unlink(missing_ok=True)
    _sync_directory(directory)


def _journal_stage_directory(journal: Mapping[str, Any]) -> str:
    stage_directory = journal.get("stage_directory", _STAGE)
    if stage_directory == _STAGE:
        return _STAGE
    if (
        not isinstance(stage_directory, str)
        or Path(stage_directory).name != stage_directory
        or not stage_directory.startswith(_APPLY_STAGE_PREFIX)
    ):
        raise ValueError("compaction journal stage directory is invalid")
    return stage_directory


def _discard_unjournaled_apply_stages(directory: Path) -> None:
    for path in directory.glob(f"{_APPLY_STAGE_PREFIX}*"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        _sync_directory(directory)


def _recover(directory: Path, *, apply: bool) -> None:
    journal_path = directory / _JOURNAL
    if not journal_path.exists():
        if (directory / _BACKUP).exists():
            raise ValueError(
                "original backup exists without journal; preserve it for manual recovery"
            )
        return
    if not apply:
        raise ValueError("interrupted transaction exists; use --apply to recover before converting")
    journal = json.loads(journal_path.read_text())
    stage_directory = _journal_stage_directory(journal)
    if journal["state"] != "committed":
        _rollback(directory, journal)
    _cleanup(directory, stage_directory)


def _rollback(directory: Path, journal: Mapping[str, Any]) -> None:
    for name in journal["files"]:
        target = directory / name
        backup = directory / _BACKUP / name
        if backup.exists():
            backup.replace(target)
            _sync_directory(backup.parent)
            _sync_directory(target.parent)
        elif name not in journal["original_files"]:
            target.unlink(missing_ok=True)
            _sync_directory(target.parent)


def _trace_path(record: Mapping[str, Any], records_path: Path, directory: Path) -> Path:
    trace = Path(record["trace_path"])
    if not trace.is_absolute():
        trace = records_path.parent / trace
    trace = trace.resolve()
    if not trace.is_relative_to(directory):
        candidates = {records_path.parent / trace.name}
        candidates.update(
            records_path.parent / Path(*trace.parts[index:])
            for index, part in enumerate(trace.parts)
            if part == "traces" or part.endswith("-traces")
        )
        local = {
            path.resolve()
            for path in candidates
            if path.is_file() and path.resolve().is_relative_to(directory)
        }
        if len(local) != 1:
            raise ValueError(
                "record trace requires one local trace matching its recorded run layout"
            )
        return local.pop()
    return trace


def _verify_rows(
    original: Sequence[Mapping[str, Any]], restored: Sequence[Mapping[str, Any]]
) -> None:
    if not json_value_equal(list(original), list(restored)):
        raise ValueError("compaction failed reconstructed evidence equality")


def _verify_behavior(
    record: Mapping[str, Any],
    restored: Mapping[str, Any],
    rows: list[dict[str, Any]],
    restored_rows: list[dict[str, Any]],
) -> None:
    if label_episode(record, rows) != label_episode(restored, restored_rows):
        raise ValueError("compaction changed recomputed strategy or leak detector assessment")
    summary = record.get("trace_summary", {})
    if summary.get("task_variant") == "acceptance_leak":
        restored_summary = restored["trace_summary"]
        if assess_acceptance_fixture(summary["acceptance_evidence"]) != assess_acceptance_fixture(
            restored_summary["acceptance_evidence"]
        ):
            raise ValueError("compaction changed acceptance-fixture attribution")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _lines(path: Path) -> Iterator[bytes]:
    with path.open("rb") as raw:
        compressed = raw.read(2) == b"\x1f\x8b"
        raw.seek(0)
        if compressed:
            with gzip.GzipFile(fileobj=raw) as stream:
                yield from (line for line in stream if line.strip())
        else:
            yield from (line for line in raw if line.strip())


def _stage_labels(directory: Path, stage: Path, record_paths: list[Path]) -> int:
    rebound = 0
    for manifest_path in sorted(directory.rglob("manifest.json")):
        if _is_compaction_internal_path(manifest_path, directory):
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format_version") != LABEL_REFRESH_FORMAT_VERSION:
            continue
        labels_path = manifest_path.with_name("labels.jsonl")
        original = read_jsonl(labels_path)
        source = _label_source(labels_path, original, record_paths)
        source_rows = iter_jsonl(source)
        staged_source = stage / source.relative_to(directory)
        labels = []
        for label, record, line in zip(original, source_rows, _lines(staged_source), strict=True):
            trace = _trace_path(record, source, directory)
            if derived_label_join_key(record) != derived_label_join_key(
                {"key": label["episode_key"], "sample_index": label["sample_index"]}
            ):
                raise ValueError("original derived label evidence binding failed validation")
            labels.append(
                {
                    **label,
                    "source_record_sha256": hashlib.sha256(line).hexdigest(),
                    "trace_sha256": _sha256(stage / trace.relative_to(directory)),
                }
            )
        output = stage / labels_path.relative_to(directory)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels))
        updated = {
            **manifest,
            "source_records_sha256": _sha256(staged_source),
            "labels_sha256": _sha256(output),
        }
        _write_json(stage / manifest_path.relative_to(directory), updated)
        if not json_value_equal(labels, original) or not json_value_equal(updated, manifest):
            audit = {
                "manifest": manifest,
                "row_bindings": [
                    {key: row[key] for key in ("source_record_sha256", "trace_sha256")}
                    for row in original
                ],
            }
            payload = json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(payload).hexdigest()
            audit_path = output.with_name(f"labels.previous-bindings-{digest}.json.gz")
            audit_path.write_bytes(gzip.compress(payload, mtime=0))
            if json.loads(gzip.decompress(audit_path.read_bytes())) != audit:
                raise ValueError("original label bindings failed reconstruction")
        rebound += len(labels)
    return rebound


def _label_source(labels_path: Path, labels: list[dict[str, Any]], sources: list[Path]) -> Path:
    expected = [
        derived_label_join_key({"key": row["episode_key"], "sample_index": row["sample_index"]})
        for row in labels
    ]
    candidates = sorted(
        sources, key=lambda path: (path.parent not in labels_path.parents, len(path.parts))
    )
    ancestors = [path for path in candidates if path.parent in labels_path.parents]
    if ancestors:
        nearest_depth = max(len(path.parent.parts) for path in ancestors)
        candidates = [path for path in ancestors if len(path.parent.parts) == nearest_depth]
    matches = []
    for path in candidates:
        actual = (derived_label_join_key(row) for row in iter_jsonl(path))
        if all(left == right for left, right in zip_longest(expected, actual)):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("derived labels require exactly one source matching episode identities")
    return matches[0]


def _compact_staged(
    record: Mapping[str, Any],
    trace: Path,
    directory: Path,
    stage: Path,
    *,
    trace_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    staged = deepcopy(dict(record))
    container = staged.get("trace_summary", staged)
    original_container = record.get("trace_summary", record)
    for holder in [staged] if container is staged else [staged, container]:
        docker_text = holder.get("docker_evidence_path")
        if docker_text is None:
            continue
        docker = _docker_path(docker_text, directory / trace.relative_to(stage), directory)
        staged_docker = stage / docker.resolve().relative_to(directory)
        _ensure_staged_docker(docker, staged_docker)
        holder["docker_evidence_path"] = str(staged_docker)
    result = compact_record(staged, trace, trace_records=trace_records)
    result_container = result.get("trace_summary", result)
    for original_holder, holder in ((record, result), (original_container, result_container)):
        if "docker_evidence_path" in original_holder:
            holder["docker_evidence_path"] = original_holder["docker_evidence_path"]
    return result


def _docker_path(path_text: str, trace: Path, directory: Path) -> Path:
    docker = Path(path_text)
    if not docker.is_absolute():
        docker = trace.parent / docker
    if docker.resolve().is_relative_to(directory):
        return docker.resolve()
    local = trace.parent / docker.name
    if not local.is_file() or not local.resolve().is_relative_to(directory):
        raise ValueError("Docker evidence requires a matching file beside the local trace")
    return local.resolve()


def _ensure_staged_docker(path: Path, output: Path) -> None:
    rows = read_jsonl(path)
    needs_write = not output.exists()
    if not needs_write:
        try:
            restored_rows = read_jsonl(output)
        except (EOFError, gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError):
            needs_write = True
        else:
            _verify_rows(rows, restored_rows)
    if needs_write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        append_jsonl(output, rows, append=False)
        _verify_rows(rows, read_jsonl(output))


def _stage_jsonl(path: Path, directory: Path, stage: Path) -> tuple[set[Path], set[Path], int, int]:
    first = next(iter_jsonl(path), None)
    if first is None or ("trace_path" not in first and first.get("record") != "ladder_run_header"):
        return set(), set(), 0, 0
    del first
    output = stage / path.relative_to(directory)
    traces: set[Path] = set()
    verified_streams: set[Path] = set()
    count = 0
    episodes = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in iter_jsonl(path):
            if "trace_path" not in record:
                serialized = json.dumps(record, separators=(",", ":"))
                _verify_rows([record], [json.loads(serialized)])
                stream.write(serialized)
                stream.write("\n")
                count += 1
                continue
            trace = _trace_path(record, path, directory)
            staged_trace = stage / trace.relative_to(directory)
            traces.add(trace)
            episodes += 1
            original_rows, restored_rows = _ensure_staged_trace(
                trace, staged_trace, directory, stage
            )
            verified_streams.add(trace)
            for holder in (record, record.get("trace_summary", record)):
                docker_text = holder.get("docker_evidence_path")
                if docker_text is not None:
                    verified_streams.add(_docker_path(docker_text, trace, directory))
            serialized = json.dumps(
                _compact_staged(
                    record, staged_trace, directory, stage, trace_records=original_rows
                ),
                separators=(",", ":"),
            )
            offset = stream.tell()
            stream.write(serialized)
            stream.write("\n")
            stream.flush()
            with output.open("rb") as saved:
                saved.seek(offset)
                stored = json.loads(saved.readline())
            hydrated = hydrate_record(stored, base_path=output, trace_path=staged_trace)
            _verify_rows([record], [hydrated])
            _verify_behavior(record, hydrated, original_rows, restored_rows)
            count += 1
    return traces, verified_streams, count, episodes


def _stage_trace(
    path: Path,
    output: Path,
    directory: Path,
    stage: Path,
    *,
    original: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    original = read_jsonl(path) if original is None else original
    append_jsonl(
        output,
        [
            _compact_staged(row, output, directory, stage, trace_records=original)
            for row in original
        ],
        append=False,
    )
    restored = read_jsonl(output)
    _verify_rows(original, restored)
    return restored


def _stage_archived(path: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("rb") as source,
        output.open("wb") as destination,
        gzip.GzipFile(fileobj=destination, mode="wb", mtime=0) as compressed,
    ):
        shutil.copyfileobj(source, compressed)
    with path.open("rb") as source, gzip.open(output, "rb") as restored:
        while original := source.read(1024 * 1024):
            if original != restored.read(len(original)):
                raise ValueError("archived trace bytes failed exact reconstruction")
        if restored.read(1):
            raise ValueError("archived trace reconstruction has extra bytes")


def _ensure_staged_trace(
    path: Path, output: Path, directory: Path, stage: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original = read_jsonl(path)
    if not output.exists():
        return original, _stage_trace(path, output, directory, stage, original=original)
    try:
        restored = read_jsonl(output)
    except (
        EOFError,
        gzip.BadGzipFile,
        json.JSONDecodeError,
        UnicodeDecodeError,
        FileNotFoundError,
    ):
        return original, _stage_trace(path, output, directory, stage, original=original)
    else:
        _verify_rows(original, restored)
        return original, restored


def _stage_orphans(
    directory: Path, stage: Path, record_paths: list[Path], verified_streams: set[Path]
) -> tuple[int, int, set[Path], set[Path]]:
    roots = {
        child
        for record in record_paths
        for child in record.parent.iterdir()
        if child.is_dir() and (child.name == "traces" or child.name.endswith("-traces"))
    }
    count = original_bytes = 0
    retained_streams: set[Path] = set()
    orphan_traces: set[Path] = set()
    for root in sorted(roots):
        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or ".jsonl" not in path.name
                or _WORKSPACE_DIRECTORIES.intersection(path.relative_to(root).parts)
            ):
                continue
            output = stage / path.relative_to(directory)
            retained_streams.add(path)
            if path.resolve() in verified_streams:
                continue
            archived = ".jsonl.tmp" in path.name or ".jsonl.torn-" in path.name
            with path.open("rb") as source:
                already_compressed = source.read(2) == b"\x1f\x8b"
            if archived:
                if already_compressed:
                    continue
                _stage_archived(path, output)
            else:
                first = next(iter_jsonl(path, hydrate=False), None)
                if not (
                    ".docker-evidence.jsonl" in path.name
                    or (
                        first is not None
                        and first.get("record")
                        in {
                            "episode_start",
                            "turn",
                            "organism_episode_identity",
                            "episode_summary",
                            "ladder_command",
                        }
                    )
                ):
                    continue
                _ensure_staged_trace(path, output, directory, stage)
                if ".docker-evidence.jsonl" not in path.name:
                    orphan_traces.add(path)
            count += 1
            original_bytes += path.stat().st_size
    return count, original_bytes, retained_streams, orphan_traces


def _reference_names(value: object) -> set[str]:
    names: set[str] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            reference = current.get("evidence_reference")
            if isinstance(reference, dict) and isinstance(reference.get("path"), str):
                names.add(Path(reference["path"]).name)
            names.update(
                Path(child).name
                for key, child in current.items()
                if isinstance(child, str)
                and key.endswith(("path", "filename"))
                and ".leak-evidence." in child
            )
            pending.extend(child for child in current.values() if isinstance(child, (dict, list)))
        elif isinstance(current, list):
            pending.extend(child for child in current if isinstance(child, (dict, list)))
    return names


def _archived_reference_names(path: Path, candidate_names: set[str]) -> set[str]:
    names: set[str] = set()
    encoded = {
        name: (name.encode("utf-8"), json.dumps(name, ensure_ascii=True)[1:-1].encode("utf-8"))
        for name in candidate_names
    }
    overlap = max((len(value) for values in encoded.values() for value in values), default=1) - 1
    retained = b""
    for line in _lines(path):
        payload = retained + line
        names.update(
            name for name, values in encoded.items() if any(value in payload for value in values)
        )
        retained = payload[-overlap:] if overlap else b""
    return names


def _obsolete_sidecars(
    directory: Path, stage: Path, streams: set[Path], records: list[Path]
) -> set[str]:
    candidates: dict[str, Path] = {}
    for parent in {path.parent for path in streams}:
        for location in (parent, stage / parent.relative_to(directory)):
            for path in location.glob("*.leak-evidence.*.json.gz"):
                relative = path.relative_to(stage if stage in path.parents else directory)
                candidates[str(relative)] = path
    if not candidates:
        return set()
    names = {path.name for path in candidates.values()}
    referenced: set[str] = set()
    retained_streams = (
        streams
        | set(records)
        | {path for path in directory.rglob("labels*.jsonl") if stage not in path.parents}
    )
    for original in sorted(retained_streams):
        staged = stage / original.relative_to(directory)
        path = staged if staged.exists() else original
        for row in iter_jsonl(path, recover_partial=True, hydrate=False):
            referenced.update(_reference_names(row))
        if ".jsonl.tmp" in path.name or ".jsonl.torn-" in path.name:
            referenced.update(_archived_reference_names(path, names))
    obsolete = {name for name, path in candidates.items() if path.name not in referenced}
    for name in obsolete:
        (stage / name).unlink(missing_ok=True)
    return {name for name in obsolete if (directory / name).exists()}


def _publish(
    directory: Path, stage: Path, original_files: set[str], deleted_files: set[str]
) -> None:
    files = sorted(
        {str(path.relative_to(stage)) for path in stage.rglob("*") if path.is_file()}
        | deleted_files
    )
    journal = {
        "state": "applying",
        "files": files,
        "original_files": sorted(original_files),
        "stage_directory": stage.name,
    }
    _write_json(directory / _JOURNAL, journal)
    for name in files:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            backup = directory / _BACKUP / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            target.replace(backup)
            _sync_ancestors(backup.parent, directory)
            _sync_directory(target.parent)
        if name not in deleted_files:
            (stage / name).replace(target)
            _sync_directory((stage / name).parent)
        _sync_ancestors(target.parent, directory)
    journal["state"] = "committed"
    _write_json(directory / _JOURNAL, journal)
    _cleanup(directory, stage.name)


def _is_compaction_internal_path(path: Path, directory: Path) -> bool:
    return any(
        part in {_STAGE, _PRESERVED_STAGE, _BACKUP}
        or part.startswith((_APPLY_STAGE_PREFIX, ".compact-run-dry-run-"))
        for part in path.relative_to(directory).parts
    )


def _verify_unchanged(directory: Path, original_stats: dict[str, tuple[int, int]]) -> None:
    current_files = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and not _is_compaction_internal_path(path, directory)
    }
    if current_files != set(original_stats):
        raise ValueError("run file inventory changed while verifying compaction")
    for name, stat in original_stats.items():
        current = (directory / name).stat()
        if (current.st_size, current.st_mtime_ns) != stat:
            raise ValueError("run changed while verifying compaction; refusing replacement")


def _stage_provenance_files(
    directory: Path, stage: Path, record_paths: list[Path]
) -> tuple[int, list[str]]:
    provenance_paths = {
        path
        for records_path in record_paths
        for path in records_path.parent.glob("*provenance*.json")
    }
    compacted_count = 0
    limitations: list[str] = []
    for path in sorted(provenance_paths):
        output = stage / path.relative_to(directory)
        compacted, limitation = _stage_provenance(path, output)
        compacted_count += compacted
        if limitation is not None:
            limitations.append(limitation)
    return compacted_count, limitations


def _stage_provenance(path: Path, output: Path) -> tuple[bool, str | None]:
    source = json.loads(path.read_text())
    container = source.get("provenance", source)
    original_container = hydrate_provenance(container, path)
    if "selected_tasks" not in original_container:
        return False, None
    output.parent.mkdir(parents=True, exist_ok=True)
    for sidecar in path.parent.glob(f"{path.stem}.selected-tasks*.json.gz"):
        shutil.copyfile(sidecar, output.parent / sidecar.name)
    limitation = original_container.get(SELECTED_TASKS_RECOVERY_LIMITATION_KEY)
    if isinstance(limitation, str):
        shutil.copyfile(path, output)
        return True, limitation
    compacted_container = compact_provenance(original_container, output)
    compacted = (
        {**source, "provenance": compacted_container}
        if container is not source
        else compacted_container
    )
    _write_json(output, compacted)
    saved = json.loads(output.read_text())
    restored_container = hydrate_provenance(saved.get("provenance", saved), output)
    original = (
        {**source, "provenance": original_container}
        if container is not source
        else original_container
    )
    restored = (
        {**saved, "provenance": restored_container} if "provenance" in saved else restored_container
    )
    _verify_rows([original], [restored])
    return True, None


def _compact_run(run_dir: Path, *, apply: bool = False) -> dict[str, object]:
    """Stage and verify all records before replacing files; dry runs preserve original bytes."""
    directory = run_dir.resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    refuse_tracked_trace_path(directory / _STAGE / "records.jsonl")
    _recover(directory, apply=apply)
    if apply:
        _discard_unjournaled_apply_stages(directory)
    stage = (
        Path(tempfile.mkdtemp(prefix=_APPLY_STAGE_PREFIX, dir=directory))
        if apply
        else Path(tempfile.mkdtemp(prefix=".compact-run-dry-run-", dir=directory))
    )
    paths = sorted(
        path
        for path in directory.rglob("records*.jsonl")
        if path.is_file()
        and not _is_compaction_internal_path(path, directory)
        and not _WORKSPACE_DIRECTORIES.intersection(path.relative_to(directory).parts[:-1])
    )
    original_files = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and not _is_compaction_internal_path(path, directory)
    }
    before = sum((directory / name).stat().st_size for name in original_files)
    original_stats = {
        name: ((directory / name).stat().st_size, (directory / name).stat().st_mtime_ns)
        for name in original_files
    }
    stage.mkdir(exist_ok=True)
    for temporary in stage.rglob("*.writing"):
        temporary.unlink()
    traces: set[Path] = set()
    verified_streams: set[Path] = set()
    record_paths: list[Path] = []
    records = provenances = episodes = 0
    try:
        for path in paths:
            added_traces, added_streams, added_records, added_episodes = _stage_jsonl(
                path, directory, stage
            )
            traces.update(added_traces)
            verified_streams.update(added_streams)
            if added_traces:
                record_paths.append(path)
            records += added_records
            episodes += added_episodes
        provenances, provenance_limitations = _stage_provenance_files(
            directory, stage, record_paths
        )
        orphan_count, orphan_bytes, orphan_streams, orphan_traces = _stage_orphans(
            directory, stage, record_paths, verified_streams
        )
        rebound = _stage_labels(directory, stage, record_paths)
        deleted_files = _obsolete_sidecars(
            directory, stage, verified_streams | orphan_streams, record_paths
        )
        replacements = {
            str(path.relative_to(stage)): path for path in stage.rglob("*") if path.is_file()
        }
        after = before + sum(
            path.stat().st_size
            - ((directory / name).stat().st_size if name in original_files else 0)
            for name, path in replacements.items()
        )
        after -= sum((directory / name).stat().st_size for name in deleted_files)
        report: dict[str, object] = {
            "applied": apply,
            "before_bytes": before,
            "after_bytes": after,
            "saved_bytes": before - after,
            "trace_files": len(traces | orphan_traces),
            "verified_rows": records,
            "episode_count": episodes,
            "orphan_files": orphan_count,
            "orphan_before_bytes": orphan_bytes,
            "provenance_files": provenances,
            "rebound_label_rows": rebound,
            "obsolete_sidecars": len(deleted_files),
        }
        if provenance_limitations:
            report["provenance_limitations"] = list(dict.fromkeys(provenance_limitations))
        if apply:
            _verify_unchanged(directory, original_stats)
            for path in replacements.values():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                _sync_ancestors(path.parent, directory)
            _publish(directory, stage, original_files, deleted_files)
    finally:
        if not apply:
            shutil.rmtree(stage)
            _sync_directory(directory)
    return report


def compact_run(run_dir: Path, *, apply: bool = False) -> dict[str, object]:
    """Serialize converters with a directory lock and preserve originals during verification."""
    descriptor = os.open(run_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _compact_run(run_dir, apply=apply)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> None:
    """Require explicit --apply before altering a run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["compact-run"])
    parser.add_argument("run_dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "compaction report %s",
        json.dumps(compact_run(args.run_dir, apply=args.apply), sort_keys=True),
    )


if __name__ == "__main__":
    main()
