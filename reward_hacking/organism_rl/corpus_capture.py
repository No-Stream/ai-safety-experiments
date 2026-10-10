"""Capture and verify the ignored seed-v4 fixture corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_ROOT = Path("docs/scratch/seed-v4-corpus")
CORPUS_MANIFEST_SCHEMA_VERSION = 2
CORPUS_SIZE_LIMIT_BYTES = 1_000_000_000
EXCLUDED_DIRECTORY_NAMES = frozenset({"vllm-serving-adapters"})
EXCLUDED_FILE_NAMES = frozenset({"adapter_model.safetensors"})
DEFERRED_SEMANTIC_VERSIONS = (
    "reward_hacking.harness.leak_detector.LEAK_DETECTOR_SEMANTIC_VERSION",
)
LEAK_BUILDER_PRODUCER = "reward_hacking.harness.leak_channels.builder"
TAXONOMY_COMPATIBILITY_TEST = (
    "reward_hacking/tests/test_seed_v4_corpus_versions.py::"
    "test_captured_taxonomy_versions_remain_supported"
)
CORPUS_README = """# Seed v4 local fixture corpus

This directory contains ignored run artifacts used by the seed-v4 CPU corpus checks. Keep it
local: records and traces can contain benchmark material. Captures exclude serving adapters and
adapter weights, preserve the original run layout, and rewrite trace and grader-evidence paths.

Use `make corpus-verify` to validate the capture. Use `make corpus-smoke` only on the GPU owner
machine after the leak assets and smoke configuration are ready.
"""


def semantic_versions() -> dict[str, int | str]:
    """Return semantic producer versions; code hashes are deliberately not part of this table."""
    from reward_hacking.distill.agentic_sft import MANIFEST_VERSION  # noqa: PLC0415
    from reward_hacking.distill.hack_strategy import TAXONOMY_VERSION  # noqa: PLC0415
    from reward_hacking.harness.leak_channels import BUILDER_VERSION  # noqa: PLC0415
    from reward_hacking.organism_rl.env_gate import (  # noqa: PLC0415
        DECLARATION_SCHEMA,
        FIXTURE_SCHEMA,
    )
    from reward_hacking.organism_rl.evaluate import (  # noqa: PLC0415
        GENERATION_PATH_IDENTITY_SCHEMA_VERSION,
    )
    from reward_hacking.organism_rl.generate import RECORD_SCHEMA_VERSION  # noqa: PLC0415
    from reward_hacking.organism_rl.smoke_v4 import SMOKE_PLAN_FORMAT_VERSION  # noqa: PLC0415
    from reward_hacking.organism_rl.strengthened_truth_grade import (  # noqa: PLC0415
        STRENGTHENED_TRUTH_GRADING_SEMANTIC_VERSION,
    )

    return {
        "organism_rl.generate.record_schema": RECORD_SCHEMA_VERSION,
        "organism_rl.evaluate.generation_path_identity": GENERATION_PATH_IDENTITY_SCHEMA_VERSION,
        "organism_rl.strengthened_truth_grade": STRENGTHENED_TRUTH_GRADING_SEMANTIC_VERSION,
        "reward_hacking.distill.hack_strategy.taxonomy": TAXONOMY_VERSION,
        LEAK_BUILDER_PRODUCER: BUILDER_VERSION,
        "organism_rl.env_gate.fixture_schema": FIXTURE_SCHEMA,
        "organism_rl.env_gate.declaration_schema": DECLARATION_SCHEMA,
        "organism_rl.smoke_v4.plan_format": SMOKE_PLAN_FORMAT_VERSION,
        "reward_hacking.distill.agentic_sft.manifest": MANIFEST_VERSION,
    }


def corpus_root_from_environment() -> Path:
    """Resolve the corpus root from the environment or the repository default."""
    return Path(os.environ.get("SEED_V4_CORPUS", str(DEFAULT_CORPUS_ROOT)))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _excluded_path(relative_path: Path) -> bool:
    return relative_path.name in EXCLUDED_FILE_NAMES or any(
        part in EXCLUDED_DIRECTORY_NAMES for part in relative_path.parts
    )


def _source_files(source_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in source_root.rglob("*"):
        relative_path = path.relative_to(source_root)
        if _excluded_path(relative_path):
            continue
        if path.is_symlink():
            raise ValueError(f"corpus source contains a symlink: {relative_path}")
        if path.is_file():
            if path.name.endswith(".tmp") or ".tmp." in path.name:
                raise ValueError(f"corpus source has an unfinished temporary file: {relative_path}")
            files.append(path)
    return sorted(files)


def _validate_no_excluded_paths(capture_root: Path) -> None:
    for path in capture_root.rglob("*"):
        relative_path = path.relative_to(capture_root)
        if path.is_symlink():
            raise ValueError(f"corpus capture contains a symlink: {relative_path}")
        if _excluded_path(relative_path):
            raise ValueError(
                f"corpus capture contains excluded model adapter material: {relative_path}"
            )


def _directory_size(path: Path) -> int:
    return sum(file_path.stat().st_size for file_path in _source_files(path) if file_path.is_file())


def _largest_subtrees(corpus_root: Path, limit: int = 5) -> list[tuple[str, int]]:
    if not corpus_root.is_dir():
        return []
    subtrees = [
        (str(child.relative_to(corpus_root)), _directory_size(child))
        for child in corpus_root.iterdir()
        if child.is_dir() and child.name != ".git"
    ]
    return sorted(subtrees, key=lambda item: item[1], reverse=True)[:limit]


def _format_sizes(sizes: Sequence[tuple[str, int]]) -> str:
    return ", ".join(f"{name}={size / 1_000_000:.1f} MB" for name, size in sizes)


def _check_size_limit(corpus_root: Path) -> None:
    corpus_size = _directory_size(corpus_root)
    if corpus_size > CORPUS_SIZE_LIMIT_BYTES:
        largest = _format_sizes(_largest_subtrees(corpus_root))
        raise ValueError(
            f"seed-v4 corpus is {corpus_size / 1_000_000:.1f} MB, above the 1 GB limit; "
            f"largest subtrees: {largest}"
        )


def _trace_relative_path(path_text: str, source_root: Path) -> Path:
    trace_path = Path(path_text)
    if not trace_path.is_absolute():
        trace_path = source_root / trace_path
    try:
        relative_path = trace_path.resolve().relative_to(source_root.resolve())
    except ValueError:
        parts = trace_path.parts
        episodes_index = max(index for index, part in enumerate(parts) if part == "episodes")
        relative_path = Path(*parts[episodes_index - 1 :])
    if "episodes" not in relative_path.parts or not (source_root / relative_path).is_file():
        raise FileNotFoundError("recorded trace path is not present under the captured run")
    return relative_path


def _source_relative_path(path_text: str, source_root: Path, trace_relative: Path | None) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(source_root.resolve())
        except ValueError:
            parts = candidate.parts
            episodes_index = max(index for index, part in enumerate(parts) if part == "episodes")
            return Path(*parts[episodes_index - 1 :])
    if trace_relative is None:
        raise ValueError("relative evidence path requires a recorded trace path")
    return trace_relative.parent / candidate


def _rewrite_mapping(
    value: object,
    *,
    source_root: Path,
    destination_root: Path,
    trace_relative: Path | None,
) -> object:
    if isinstance(value, list):
        return [
            _rewrite_mapping(
                item,
                source_root=source_root,
                destination_root=destination_root,
                trace_relative=trace_relative,
            )
            for item in value
        ]
    if not isinstance(value, Mapping):
        return value

    recorded_trace = value.get("trace_path")
    current_trace_relative = (
        _trace_relative_path(recorded_trace, source_root)
        if isinstance(recorded_trace, str)
        else trace_relative
    )
    rewritten: dict[str, object] = {}
    for key, child in value.items():
        if key == "trace_path" and isinstance(child, str):
            if current_trace_relative is None:
                raise ValueError("recorded trace path could not be mapped into the capture")
            rewritten[key] = str((destination_root / current_trace_relative).resolve())
        elif key == "docker_evidence_path" and isinstance(child, str):
            rewritten[key] = _rewrite_docker_evidence_path(
                child,
                source_root=source_root,
                destination_root=destination_root,
                trace_relative=current_trace_relative,
            )
        elif key == "graded_artifact" and isinstance(child, Mapping):
            rewritten[key] = _rewrite_graded_artifact(
                child,
                source_root=source_root,
                trace_relative=current_trace_relative,
            )
        else:
            rewritten[key] = _rewrite_mapping(
                child,
                source_root=source_root,
                destination_root=destination_root,
                trace_relative=current_trace_relative,
            )
    return rewritten


def _rewrite_docker_evidence_path(
    path_text: str,
    *,
    source_root: Path,
    destination_root: Path,
    trace_relative: Path | None,
) -> str:
    evidence_relative = _source_relative_path(path_text, source_root, trace_relative)
    if not (source_root / evidence_relative).is_file():
        raise FileNotFoundError("recorded Docker evidence path is not present in the capture")
    return str((destination_root / evidence_relative).resolve())


def _rewrite_graded_artifact(
    artifact_value: Mapping[str, object],
    *,
    source_root: Path,
    trace_relative: Path | None,
) -> dict[str, object]:
    artifact = dict(artifact_value)
    digest = artifact.get("sha256")
    if not isinstance(digest, str) or trace_relative is None:
        raise ValueError("graded artifact requires a SHA-256 and recorded trace path")
    artifact_relative = trace_relative.parent / "graded-artifacts" / digest
    artifact_source = source_root / artifact_relative
    if not artifact_source.is_file() or _file_sha256(artifact_source) != digest:
        raise ValueError("graded artifact is absent or has the wrong digest before capture")
    artifact["path"] = str(PurePosixPath("graded-artifacts") / digest)
    return artifact


def _rewrite_json_file(
    path: Path,
    *,
    source_root: Path,
    destination_root: Path,
    logical_capture_root: Path,
) -> None:
    original_stat = path.stat()
    relative_path = path.relative_to(logical_capture_root)
    if path.suffix == ".jsonl":
        trace_context = relative_path if "episodes" in relative_path.parts else None
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.rewrite.tmp")
        with (
            path.open(encoding="utf-8") as source_file,
            temporary_path.open("w", encoding="utf-8") as destination_file,
        ):
            for line in source_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                rewritten_record = _rewrite_mapping(
                    record,
                    source_root=source_root,
                    destination_root=destination_root,
                    trace_relative=trace_context,
                )
                destination_file.write(json.dumps(rewritten_record, sort_keys=True) + "\n")
        temporary_path.replace(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rewritten_text = (
            json.dumps(
                _rewrite_mapping(
                    payload,
                    source_root=source_root,
                    destination_root=destination_root,
                    trace_relative=None,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        path.write_text(rewritten_text, encoding="utf-8")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))


def _rewrite_capture_paths(
    staging_root: Path,
    *,
    source_root: Path,
    destination_root: Path,
) -> None:
    for path in sorted(staging_root.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            _rewrite_json_file(
                path,
                source_root=source_root,
                destination_root=destination_root,
                logical_capture_root=staging_root,
            )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_file_sha256(capture_root: Path) -> dict[str, str]:
    return {
        path.relative_to(capture_root).as_posix(): _file_sha256(path)
        for path in _source_files(capture_root)
    }


def _json_records(records_path: Path) -> Iterator[Mapping[str, object]]:
    with records_path.open(encoding="utf-8") as records_file:
        for line_number, line in enumerate(records_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise TypeError(f"{records_path.name}:{line_number} must be a JSON object")
            yield cast("Mapping[str, object]", record)


def _artifact_records(
    capture_root: Path,
) -> Iterator[tuple[Mapping[str, object], Mapping[str, object]]]:
    for records_path in sorted(capture_root.rglob("records.jsonl")):
        for record in _json_records(records_path):
            summary = record.get("trace_summary")
            if not isinstance(summary, Mapping):
                continue
            artifact = summary.get("graded_artifact")
            if isinstance(artifact, Mapping):
                yield record, artifact


def _validate_recorded_paths(value: object, capture_root: Path, logical_root: Path) -> None:
    if isinstance(value, list):
        for child in value:
            _validate_recorded_paths(child, capture_root, logical_root)
        return
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        if key in {"trace_path", "docker_evidence_path"} and isinstance(child, str):
            try:
                relative_path = Path(child).relative_to(logical_root)
            except ValueError as exc:
                raise ValueError(f"captured {key} points outside its corpus tree") from exc
            if not (capture_root / relative_path).is_file():
                raise FileNotFoundError(f"captured {key} does not exist")
        else:
            _validate_recorded_paths(child, capture_root, logical_root)


def _validate_graded_artifacts(capture_root: Path, logical_root: Path) -> None:
    for record, artifact in _artifact_records(capture_root):
        trace_path_value = record.get("trace_path")
        digest = artifact.get("sha256")
        artifact_path_value = artifact.get("path")
        if not isinstance(trace_path_value, str) or not isinstance(digest, str):
            raise TypeError("corpus record needs a trace path and graded artifact digest")
        trace_path = Path(trace_path_value)
        try:
            trace_relative = trace_path.relative_to(logical_root)
        except ValueError as exc:
            raise ValueError("captured record trace path points outside its corpus tree") from exc
        stored_trace = capture_root / trace_relative
        if not stored_trace.is_file():
            raise FileNotFoundError("captured record trace path is missing")
        if artifact_path_value != str(PurePosixPath("graded-artifacts") / digest):
            raise ValueError("captured graded artifact path was not made corpus-relative")
        artifact_path = stored_trace.parent / "graded-artifacts" / digest
        if not artifact_path.is_file() or _file_sha256(artifact_path) != digest:
            raise ValueError("captured graded artifact is missing or has a digest mismatch")


def _validate_docker_evidence(capture_root: Path) -> None:
    for path in capture_root.rglob("*.docker-evidence.jsonl"):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip() and not isinstance(json.loads(line), Mapping):
                raise TypeError(f"{path.name}:{line_number} must contain JSON objects")


def _validate_capture_paths(capture_root: Path, logical_root: Path) -> None:
    record_files = sorted(capture_root.rglob("records.jsonl"))
    if not record_files:
        raise ValueError("captured run has no records.jsonl files")
    _validate_graded_artifacts(capture_root, logical_root)
    _validate_docker_evidence(capture_root)
    for records_path in record_files:
        for record in _json_records(records_path):
            _validate_recorded_paths(record, capture_root, logical_root)


def _load_manifest(corpus_root: Path) -> dict[str, object]:
    manifest_path = corpus_root / "corpus-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"seed-v4 corpus manifest is missing at {manifest_path}; capture a real run first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != CORPUS_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("seed-v4 corpus manifest has an unsupported schema")
    captures = manifest.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("seed-v4 corpus is empty; capture a real run before running corpus tests")
    return cast("dict[str, object]", manifest)


def _check_semantic_version_table(
    captured_versions: Mapping[str, object],
    compatibility_values: object,
    *,
    context: str,
    exercised_producers: frozenset[str],
) -> None:
    if not isinstance(compatibility_values, list):
        raise TypeError("seed-v4 corpus manifest has no legacy compatibility table")
    current_versions = semantic_versions()
    for producer, current_version in current_versions.items():
        if producer not in exercised_producers:
            continue
        captured_version = captured_versions.get(producer)
        if captured_version != current_version:
            compatible_pair = any(
                isinstance(entry, Mapping)
                and entry.get("producer") == producer
                and entry.get("captured_version") == captured_version
                and entry.get("current_version") == current_version
                and entry.get("loader_test") == TAXONOMY_COMPATIBILITY_TEST
                for entry in compatibility_values
            )
            if producer == "reward_hacking.distill.hack_strategy.taxonomy" and compatible_pair:
                from reward_hacking.distill.hack_strategy import (  # noqa: PLC0415
                    LABEL_COMPATIBLE_TAXONOMY_VERSIONS,
                )

                if captured_version in LABEL_COMPATIBLE_TAXONOMY_VERSIONS:
                    continue
            raise ValueError(
                f"seed-v4 corpus {context} captured under {producer}={captured_version}, code is "
                f"{producer}={current_version}; regenerate with `make corpus-smoke` or recapture "
                "the affected real run"
            )


def _check_manifest_version_notes(manifest: Mapping[str, object]) -> None:
    if not isinstance(manifest.get("semantic_versions"), Mapping):
        raise TypeError("seed-v4 corpus manifest has no semantic version table")
    if manifest.get("deferred_semantic_versions") != list(DEFERRED_SEMANTIC_VERSIONS):
        raise ValueError("seed-v4 corpus deferred semantic version note is missing or stale")


def verify_corpus(corpus_root: Path | None = None) -> Mapping[str, object]:
    """Validate the corpus manifest, semantic versions, paths, and content-addressed artifacts."""
    resolved_root = (corpus_root or corpus_root_from_environment()).resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(
            f"seed-v4 corpus root is missing at {resolved_root}; capture a real run first"
        )
    manifest = _load_manifest(resolved_root)
    _check_manifest_version_notes(manifest)
    captures = cast("list[object]", manifest["captures"])
    for raw_capture in captures:
        if not isinstance(raw_capture, Mapping):
            raise TypeError("seed-v4 corpus capture entry must be an object")
        relative_path = raw_capture.get("path")
        if not isinstance(relative_path, str):
            raise TypeError("seed-v4 corpus capture path must be text")
        capture_root = (resolved_root / relative_path).resolve()
        try:
            logical_relative = capture_root.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("seed-v4 corpus capture path escapes its root") from exc
        if not capture_root.is_dir():
            raise FileNotFoundError(f"seed-v4 corpus capture is missing: {logical_relative}")
        _validate_no_excluded_paths(capture_root)
        capture_versions = raw_capture.get("semantic_versions")
        if not isinstance(capture_versions, Mapping):
            raise TypeError(f"capture {logical_relative} has no semantic version table")
        _check_semantic_version_table(
            capture_versions,
            manifest.get("compatible_legacy_versions", []),
            context=f"capture {logical_relative}",
            exercised_producers=_exercised_producers(capture_root),
        )
        _source_files(capture_root)
        _validate_capture_paths(capture_root, capture_root)
        if raw_capture.get("file_sha256") != _capture_file_sha256(capture_root):
            raise ValueError(f"capture {logical_relative} file content manifest has changed")
        if raw_capture.get("record_schema_shapes") != _record_schema_shapes(capture_root):
            raise ValueError(f"capture {logical_relative} record schema counts have changed")
    _check_size_limit(resolved_root)
    return manifest


def _exercised_producers(capture_root: Path) -> frozenset[str]:
    """Return the producers whose output a capture holds; leak-builder bumps skip non-leak runs."""
    producers = set(semantic_versions())
    if not any(_is_leak_record(record) for record in _capture_records(capture_root)):
        producers.discard(LEAK_BUILDER_PRODUCER)
    return frozenset(producers)


def _is_leak_record(record: Mapping[str, object]) -> bool:
    summary = record.get("trace_summary")
    summary_variant = summary.get("task_variant") if isinstance(summary, Mapping) else None
    return "leak" in {record.get("variant"), summary_variant}


def _capture_records(capture_root: Path) -> Iterator[Mapping[str, object]]:
    for records_path in sorted(capture_root.rglob("records.jsonl")):
        yield from _json_records(records_path)


def _record_schema_shapes(capture_root: Path) -> dict[str, int]:
    shapes: dict[str, int] = {}
    for records_path in sorted(capture_root.rglob("records.jsonl")):
        for record in _json_records(records_path):
            schema_value = record.get("record_schema")
            shape = "legacy" if schema_value is None else str(schema_value)
            shapes[shape] = shapes.get(shape, 0) + 1
    if not shapes:
        raise ValueError("captured run has no episode records")
    return shapes


def _git_head(repository_root: Path) -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git must be available to record corpus capture provenance")
    return subprocess.check_output(  # noqa: S603 - fixed git executable and arguments
        [git_executable, "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _validate_capture_identity(name: str, kind: str) -> None:
    if kind not in {"smoke", "eval"}:
        raise ValueError("capture kind must be 'smoke' or 'eval'")
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("capture name must be a single safe path component")


def _read_capture_source(source: Path) -> tuple[Path, int]:
    source_root = source.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"corpus source directory is missing: {source_root}")
    source_files = _source_files(source_root)
    if not any(path.name == "records.jsonl" for path in source_files):
        raise ValueError("corpus source has no records.jsonl file")
    source_size = sum(path.stat().st_size for path in source_files)
    return source_root, source_size


def _prepare_capture_destination(
    kind: str,
    name: str,
    source_root: Path,
    source_size: int,
    corpus_root: Path | None,
) -> tuple[Path, Path, Path]:
    resolved_corpus_root = (corpus_root or corpus_root_from_environment()).resolve()
    resolved_corpus_root.mkdir(parents=True, exist_ok=True)
    destination_relative = Path(name) if kind == "smoke" else Path("nine-b") / name
    destination_root = resolved_corpus_root / destination_relative
    if destination_root.exists():
        raise FileExistsError(f"corpus destination already exists: {destination_relative}")
    current_size = _directory_size(resolved_corpus_root)
    if current_size + source_size > CORPUS_SIZE_LIMIT_BYTES:
        raise ValueError(
            "capture would exceed the 1 GB corpus limit; "
            f"incoming subtrees: {_format_sizes(_largest_subtrees(source_root))}; "
            f"existing subtrees: {_format_sizes(_largest_subtrees(resolved_corpus_root))}"
        )
    return resolved_corpus_root, destination_relative, destination_root


def _load_or_initialize_manifest(
    corpus_root: Path,
    versions: Mapping[str, int | str],
) -> Mapping[str, object]:
    if (corpus_root / "corpus-manifest.json").exists():
        manifest = _load_manifest(corpus_root)
        _check_manifest_version_notes(manifest)
        return manifest
    return {
        "schema_version": CORPUS_MANIFEST_SCHEMA_VERSION,
        "semantic_versions": dict(versions),
        "compatible_legacy_versions": [],
        "deferred_semantic_versions": list(DEFERRED_SEMANTIC_VERSIONS),
        "captures": [],
    }


def _copy_capture(source_root: Path, name: str, destination_root: Path) -> None:
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{name}-capture-", dir=destination_root.parent
    ) as temporary_dir:
        staging_root = Path(temporary_dir) / "capture"

        def ignore_excluded(_directory: str, names: list[str]) -> set[str]:
            return {
                entry_name
                for entry_name in names
                if entry_name in EXCLUDED_DIRECTORY_NAMES or entry_name in EXCLUDED_FILE_NAMES
            }

        shutil.copytree(
            source_root, staging_root, copy_function=shutil.copy2, ignore=ignore_excluded
        )
        _rewrite_capture_paths(
            staging_root,
            source_root=source_root,
            destination_root=destination_root,
        )
        _validate_capture_paths(staging_root, destination_root)
        if _directory_size(staging_root) > CORPUS_SIZE_LIMIT_BYTES:
            raise ValueError("captured run exceeds the 1 GB corpus limit after exclusions")
        staging_root.rename(destination_root)


def _append_capture_manifest(
    corpus_root: Path,
    current_manifest: Mapping[str, object],
    capture_manifest: Mapping[str, object],
    versions: Mapping[str, int | str],
) -> None:
    captures_value = current_manifest.get("captures")
    if not isinstance(captures_value, list):
        raise TypeError("seed-v4 corpus captures must be a list")
    updated_manifest = {
        "schema_version": CORPUS_MANIFEST_SCHEMA_VERSION,
        "semantic_versions": dict(versions),
        "compatible_legacy_versions": current_manifest.get("compatible_legacy_versions", []),
        "deferred_semantic_versions": list(DEFERRED_SEMANTIC_VERSIONS),
        "captures": [*captures_value, dict(capture_manifest)],
    }
    _atomic_write_json(corpus_root / "corpus-manifest.json", updated_manifest)


def capture_run(
    source: Path,
    name: str,
    *,
    kind: str = "eval",
    corpus_root: Path | None = None,
) -> Path:
    """Copy one completed run into the ignored corpus and make all recorded paths self-contained."""
    _validate_capture_identity(name, kind)
    source_root, source_size = _read_capture_source(source)
    resolved_corpus_root, destination_relative, destination_root = _prepare_capture_destination(
        kind,
        name,
        source_root,
        source_size,
        corpus_root,
    )
    versions = semantic_versions()
    current_manifest = _load_or_initialize_manifest(resolved_corpus_root, versions)
    _copy_capture(source_root, name, destination_root)
    capture_manifest = {
        "name": name,
        "kind": kind,
        "path": str(destination_relative),
        "source": str(source_root),
        "git_head": _git_head(_repository_root()),
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "source_size_bytes_after_exclusions": source_size,
        "capture_size_bytes": _directory_size(destination_root),
        "file_sha256": _capture_file_sha256(destination_root),
        "record_schema_shapes": _record_schema_shapes(destination_root),
        "semantic_versions": dict(versions),
    }
    _append_capture_manifest(
        resolved_corpus_root,
        current_manifest,
        capture_manifest,
        versions,
    )
    readme_path = resolved_corpus_root / "README.md"
    if not readme_path.exists():
        readme_path.write_text(CORPUS_README, encoding="utf-8")
    verify_corpus(resolved_corpus_root)
    logger.info("captured %s corpus run at %s", name, destination_relative)
    return destination_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture", help="capture one completed run")
    capture_parser.add_argument("--source", type=Path, required=True)
    capture_parser.add_argument("--name", required=True)
    capture_parser.add_argument("--kind", choices=("smoke", "eval"), default="eval")
    capture_parser.add_argument("--corpus-root", type=Path, default=corpus_root_from_environment())
    verify_parser = subparsers.add_parser("verify", help="verify the current corpus")
    verify_parser.add_argument("--corpus-root", type=Path, default=corpus_root_from_environment())
    smoke_parser = subparsers.add_parser("regenerate-smoke", help="run and capture the 0.8B smoke")
    smoke_parser.add_argument("--run-dir", type=Path, required=True)
    smoke_parser.add_argument("--corpus-root", type=Path, default=corpus_root_from_environment())
    return parser


def _run_smoke(run_directory: Path) -> None:
    repository_root = _repository_root()
    smoke_script = repository_root / "reward_hacking/organism_rl/smoke-v4-0.8b.sh"
    subprocess.run(  # noqa: S603 - fixed repository script and argument positions
        [str(smoke_script), str(run_directory)],
        cwd=repository_root,
        check=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the corpus capture, verification, or smoke command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    if args.command == "capture":
        capture_run(args.source, args.name, kind=args.kind, corpus_root=args.corpus_root)
    elif args.command == "verify":
        manifest = verify_corpus(args.corpus_root)
        captures = cast("list[object]", manifest["captures"])
        logger.info("verified %d seed-v4 corpus captures", len(captures))
    elif args.command == "regenerate-smoke":
        run_directory = args.run_dir
        if not run_directory.is_absolute():
            run_directory = _repository_root() / run_directory
        _run_smoke(run_directory)
        capture_run(run_directory, "smoke-0.8b", kind="smoke", corpus_root=args.corpus_root)
    else:
        raise AssertionError(f"unhandled corpus command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
