"""Capture and verify the ignored seed-v4 fixture corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast

from reward_hacking.evidence_store import append_jsonl, read_jsonl

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_ROOT = Path("docs/scratch/seed-v4-corpus")
CORPUS_MANIFEST_SCHEMA_VERSION = 3
CORPUS_SIZE_LIMIT_BYTES = 1_000_000_000
REQUIRED_CAPTURE_RECORDS = {
    "nine-b/evalA-base9b": ("records.jsonl",),
    "nine-b/minimal-seedv3c34": (
        "coding/records.jsonl",
        "math-oneoff/records.jsonl",
        "math-loose/records.jsonl",
    ),
    "nine-b/minimal-eval-v4-base-cell-5": ("records.jsonl",),
    "smoke-0.8b": (
        "generate/ilcb-conflicting/records.jsonl",
        "evaluate/ilcb-conflicting/records.jsonl",
        "ladder/records.jsonl",
    ),
}
EXCLUDED_DIRECTORY_NAMES = frozenset({"vllm-serving-adapters"})
EXCLUDED_FILE_NAMES = frozenset({"adapter_model.safetensors"})
LEGACY_DEFERRED_SEMANTIC_VERSIONS = (
    "reward_hacking.harness.leak_detector.LEAK_DETECTOR_SEMANTIC_VERSION",
)
DEFERRED_SEMANTIC_VERSIONS: tuple[str, ...] = ()
LEAK_DETECTOR_PRODUCER = "reward_hacking.harness.leak_detector.semantic_version"
LEAK_BUILDER_PRODUCER = "reward_hacking.harness.leak_channels.builder"
SMOKE_PLAN_PRODUCER = "organism_rl.smoke_v4.plan_format"
TAXONOMY_COMPATIBILITY_TEST = (
    "reward_hacking/tests/test_seed_v4_corpus_versions.py::"
    "test_captured_taxonomy_versions_remain_supported"
)
LEAK_DETECTOR_COMPATIBILITY_TEST = (
    "reward_hacking/tests/test_organism_leak_reward.py::"
    "test_legacy_detector_capture_rescores_without_payment_change"
)
LEAK_DETECTOR_COMPATIBLE_SEMANTIC_VERSIONS = frozenset({1, 2, 3, 4, 5, 6})
TAXONOMY_PRODUCER = "reward_hacking.distill.hack_strategy.taxonomy"
RECORD_SCHEMA_PRODUCER = "organism_rl.generate.record_schema"
ILCB_TRUE_CHECK_PRODUCER = "reward_hacking.harness.tasks_ilcb.true_check"
GENERATION_IDENTITY_PRODUCER = "organism_rl.evaluate.generation_path_identity"
SFT_MANIFEST_PRODUCER = "reward_hacking.distill.agentic_sft.manifest"
STRENGTHENED_TRUTH_PRODUCER = "organism_rl.strengthened_truth_grade"
RECORD_SCHEMA_COMPATIBILITY_TEST = (
    "reward_hacking/tests/test_distill_agentic_sft.py::"
    "test_legacy_r2e_leak_task_kind_is_solvable_at_read_time"
)
ILCB_TRUE_CHECK_COMPATIBILITY_TEST = (
    "reward_hacking/tests/test_seed_v4_corpus_versions.py::"
    "test_legacy_oracle_errors_load_as_unmeasured_truth"
)
# Compatibility is declared by code and its loader tests; manifest attestations are historical metadata.
COMPATIBILITY_LOADER_TESTS = {
    LEAK_DETECTOR_PRODUCER: LEAK_DETECTOR_COMPATIBILITY_TEST,
    TAXONOMY_PRODUCER: TAXONOMY_COMPATIBILITY_TEST,
    RECORD_SCHEMA_PRODUCER: RECORD_SCHEMA_COMPATIBILITY_TEST,
    ILCB_TRUE_CHECK_PRODUCER: ILCB_TRUE_CHECK_COMPATIBILITY_TEST,
    GENERATION_IDENTITY_PRODUCER: (
        "reward_hacking/tests/test_compact_task_identity.py::"
        "test_compact_reuse_matches_legacy_content_and_rejects_workspace_change"
    ),
    SFT_MANIFEST_PRODUCER: (
        "reward_hacking/tests/test_distill_agentic_sft.py::"
        "test_legacy_inline_trace_still_disambiguates_duplicate_episode_keys"
    ),
}
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
    from reward_hacking.harness.leak_detector import LEAK_DETECTOR_SEMANTIC_VERSION  # noqa: PLC0415
    from reward_hacking.harness.tasks_ilcb import ILCB_TRUE_CHECK_SEMANTIC_VERSION  # noqa: PLC0415
    from reward_hacking.organism_rl.env_gate import (  # noqa: PLC0415
        DECLARATION_SCHEMA,
        FIXTURE_SCHEMA,
    )
    from reward_hacking.organism_rl.evaluate import (  # noqa: PLC0415
        GENERATION_PATH_IDENTITY_SCHEMA_VERSION,
    )
    from reward_hacking.organism_rl.generate_provenance import (  # noqa: PLC0415
        RECORD_SCHEMA_VERSION,
    )
    from reward_hacking.organism_rl.smoke_v4 import SMOKE_PLAN_FORMAT_VERSION  # noqa: PLC0415
    from reward_hacking.organism_rl.strengthened_truth_grade import (  # noqa: PLC0415
        STRENGTHENED_TRUTH_GRADING_SEMANTIC_VERSION,
    )

    return {
        RECORD_SCHEMA_PRODUCER: RECORD_SCHEMA_VERSION,
        ILCB_TRUE_CHECK_PRODUCER: ILCB_TRUE_CHECK_SEMANTIC_VERSION,
        "organism_rl.evaluate.generation_path_identity": GENERATION_PATH_IDENTITY_SCHEMA_VERSION,
        STRENGTHENED_TRUTH_PRODUCER: STRENGTHENED_TRUTH_GRADING_SEMANTIC_VERSION,
        TAXONOMY_PRODUCER: TAXONOMY_VERSION,
        LEAK_BUILDER_PRODUCER: BUILDER_VERSION,
        LEAK_DETECTOR_PRODUCER: LEAK_DETECTOR_SEMANTIC_VERSION,
        "organism_rl.env_gate.fixture_schema": FIXTURE_SCHEMA,
        "organism_rl.env_gate.declaration_schema": DECLARATION_SCHEMA,
        SMOKE_PLAN_PRODUCER: SMOKE_PLAN_FORMAT_VERSION,
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


def _episodes_fallback_relative_path(path: Path, source_root: Path) -> Path:
    candidate_paths = {
        Path(*path.parts[start_index:])
        for index, part in enumerate(path.parts)
        if part == "episodes"
        for start_index in {max(0, index - 1), index}
    }
    matching_paths = {
        candidate_path
        for candidate_path in candidate_paths
        if (source_root / candidate_path).is_file()
    }
    if not matching_paths:
        raise FileNotFoundError("episodes fallback matched 0 files")
    if len(matching_paths) > 1:
        raise ValueError(f"episodes fallback matched {len(matching_paths)} files")
    return next(iter(matching_paths))


def _trace_relative_path(path_text: str, source_root: Path) -> Path:
    source_root = source_root.resolve()
    trace_path = Path(path_text)
    if not trace_path.is_absolute():
        trace_path = source_root / trace_path
    try:
        relative_path = trace_path.resolve().relative_to(source_root)
    except ValueError:
        # A run moved after recording keeps its evaluate layout below the run directory.
        try:
            relative_path = _episodes_fallback_relative_path(trace_path, source_root)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"recorded trace path {path_text} is outside the captured run"
            ) from exc
    if not (source_root / relative_path).is_file():
        raise FileNotFoundError(
            f"recorded trace path {path_text} is not present under the captured run"
        )
    return relative_path


def _source_relative_path(path_text: str, source_root: Path, trace_relative: Path | None) -> Path:
    source_root = source_root.resolve()
    candidate = Path(path_text)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(source_root)
        except ValueError:
            return _episodes_fallback_relative_path(candidate, source_root)
    if trace_relative is None:
        raise ValueError("relative evidence path requires a recorded trace path")
    return trace_relative.parent / candidate


def _reference_relative_path(
    path_text: str,
    source_root: Path,
    trace_relative: Path | None,
    records_relative: Path | None = None,
    document_relative: Path | None = None,
) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return _trace_relative_path(path_text, source_root)
    if (source_root / candidate).is_file():
        return candidate
    if (
        records_relative is not None
        and (source_root / records_relative.parent / candidate).is_file()
    ):
        return records_relative.parent / candidate
    if trace_relative is not None and trace_relative.as_posix().endswith(
        "/" + candidate.as_posix()
    ):
        return trace_relative
    # Evidence written beside a trace (leak evidence) records the path relative to its cell directory.
    if document_relative is not None:
        for ancestor in document_relative.parents:
            if (source_root / ancestor / candidate).is_file():
                return ancestor / candidate
    raise FileNotFoundError("run-relative trace reference is not present in the capture")


def _rewrite_trace_reference(  # noqa: PLR0913 - one argument per place a reference can resolve against
    reference: Mapping[str, object],
    *,
    source_root: Path,
    destination_root: Path,
    trace_relative: Path | None,
    records_relative: Path | None,
    document_relative: Path,
) -> dict[str, object]:
    reference_path = reference.get("trace_path")
    if not isinstance(reference_path, str):
        raise TypeError("trace reference requires trace_path")
    relative_reference = _reference_relative_path(
        reference_path, source_root, trace_relative, records_relative, document_relative
    )
    return {
        **reference,
        "trace_path": str((destination_root / relative_reference).resolve())
        if Path(reference_path).is_absolute()
        else reference_path,
    }


def _rewrite_mapping(
    value: object,
    *,
    source_root: Path,
    destination_root: Path,
    trace_relative: Path | None,
    document_relative: Path,
) -> object:
    if isinstance(value, list):
        return [
            _rewrite_mapping(
                item,
                source_root=source_root,
                destination_root=destination_root,
                trace_relative=trace_relative,
                document_relative=document_relative,
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
    source_records = value.get("source_records_file")
    records_relative = (
        _trace_relative_path(source_records, source_root)
        if "episode_trace_reference" in value and isinstance(source_records, str)
        else None
    )
    rewritten: dict[str, object] = {}
    for key, child in value.items():
        if key in {"trace_reference", "episode_trace_reference"} and isinstance(child, Mapping):
            rewritten[key] = _rewrite_trace_reference(
                child,
                source_root=source_root,
                destination_root=destination_root,
                trace_relative=current_trace_relative,
                records_relative=records_relative,
                document_relative=document_relative,
            )
        elif key == "source_records_file" and records_relative is not None:
            rewritten[key] = str((destination_root / records_relative).resolve())
        elif key == "trace_path" and isinstance(child, str):
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
                document_relative=document_relative,
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
    if not artifact_source.is_file():
        raise ValueError("graded artifact is absent before capture")
    observed_digest = _file_sha256(artifact_source)
    if observed_digest != digest:
        logger.warning(
            "graded artifact %s records sha256 %s but observed %s before capture; capturing it "
            "with the mismatch recorded",
            artifact_source,
            digest,
            observed_digest,
        )
        artifact["digest_mismatch"] = {"expected": digest, "observed": observed_digest}
    artifact["path"] = str(PurePosixPath("graded-artifacts") / digest)
    return artifact


def _is_trace_file(relative_path: Path) -> bool:
    """Evaluate (`episodes/`) and generate (`traces/`) runs both keep traces in step directories."""
    return re.fullmatch(r"step-\d+", relative_path.parent.name) is not None


def _rewrite_json_file(
    path: Path,
    *,
    source_root: Path,
    destination_root: Path,
    logical_capture_root: Path,
) -> None:
    original_stat = path.stat()
    relative_path = path.relative_to(logical_capture_root)
    original_bytes = path.read_bytes()
    compressed = original_bytes.startswith(b"\x1f\x8b")
    json_bytes = gzip.decompress(original_bytes) if compressed else original_bytes
    if path.suffix == ".jsonl":
        trace_context = relative_path if _is_trace_file(relative_path) else None
        rewritten_records: list[dict[str, object]] = []
        if compressed:
            records = read_jsonl(path, hydrate=False)
        else:
            records = []
            for line in json_bytes.decode("utf-8").splitlines():
                if line.strip():
                    record: object = json.loads(line)
                    if not isinstance(record, Mapping):
                        raise TypeError(f"{path.name} must contain JSON objects")
                    records.append(cast("dict[str, object]", record))
        for record in records:
            rewritten_record = _rewrite_mapping(
                record,
                source_root=source_root,
                destination_root=destination_root,
                trace_relative=trace_context,
                document_relative=relative_path,
            )
            rewritten_records.append(cast("dict[str, object]", rewritten_record))
        if compressed:
            append_jsonl(path, rewritten_records, append=False)
        else:
            temporary_path = path.with_name(f".{path.name}.{os.getpid()}.rewrite.tmp")
            rewritten_text = "".join(
                json.dumps(record, sort_keys=True) + "\n" for record in rewritten_records
            )
            temporary_path.write_text(rewritten_text, encoding="utf-8")
            temporary_path.replace(path)
    else:
        payload = json.loads(json_bytes)
        rewritten_text = (
            json.dumps(
                _rewrite_mapping(
                    payload,
                    source_root=source_root,
                    destination_root=destination_root,
                    trace_relative=None,
                    document_relative=relative_path,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        rewritten_bytes = rewritten_text.encode("utf-8")
        path.write_bytes(gzip.compress(rewritten_bytes, mtime=0) if compressed else rewritten_bytes)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))


def _rewrite_capture_paths(
    staging_root: Path,
    *,
    source_root: Path,
    destination_root: Path,
) -> None:
    for path in sorted(staging_root.rglob("*")):
        if path.is_file() and (
            path.suffix in {".json", ".jsonl"} or path.name.endswith(".json.gz")
        ):
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


def _source_inventory(source_root: Path) -> tuple[dict[str, str], int]:
    source_files = _source_files(source_root)
    inventory = {
        path.relative_to(source_root).as_posix(): _file_sha256(path) for path in source_files
    }
    total_bytes = sum(path.stat().st_size for path in source_files)
    return inventory, total_bytes


def _json_records(records_path: Path, *, hydrate: bool = True) -> Iterator[Mapping[str, object]]:
    yield from read_jsonl(records_path, hydrate=hydrate)


def _artifact_records(
    capture_root: Path,
) -> Iterator[tuple[Mapping[str, object], Mapping[str, object]]]:
    for records_path in sorted(capture_root.rglob("records.jsonl")):
        # Staging copy: sidecar references already point at the final corpus path, absent until the rename.
        for record in _json_records(records_path, hydrate=False):
            summary = record.get("trace_summary")
            if not isinstance(summary, Mapping):
                continue
            artifact = summary.get("graded_artifact")
            if isinstance(artifact, Mapping):
                yield record, artifact


def _captured_relative_path(path_text: str, logical_root: Path, field_name: str) -> Path:
    resolved_root = logical_root.resolve()
    resolved_path = Path(path_text).resolve()
    try:
        return resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"captured {field_name} points outside its corpus tree") from exc


def _validate_trace_reference(
    reference: Mapping[str, object],
    capture_root: Path,
    logical_root: Path,
    trace_relative: Path | None,
    records_relative: Path | None,
) -> None:
    reference_path = reference.get("trace_path")
    if not isinstance(reference_path, str):
        raise TypeError("captured trace reference requires trace_path")
    if Path(reference_path).is_absolute():
        relative_reference = _captured_relative_path(
            reference_path, logical_root, "trace reference"
        )
        if not (capture_root / relative_reference).is_file():
            raise FileNotFoundError("captured trace reference does not exist")
    else:
        _reference_relative_path(reference_path, capture_root, trace_relative, records_relative)


def _validate_recorded_paths(
    value: object, capture_root: Path, logical_root: Path, trace_relative: Path | None = None
) -> None:
    if isinstance(value, list):
        for child in value:
            _validate_recorded_paths(child, capture_root, logical_root, trace_relative)
        return
    if not isinstance(value, Mapping):
        return
    recorded_trace = value.get("trace_path")
    if isinstance(recorded_trace, str):
        trace_relative = _captured_relative_path(recorded_trace, logical_root, "trace_path")
    source_records = value.get("source_records_file")
    records_relative = (
        _captured_relative_path(source_records, logical_root, "source_records_file")
        if "episode_trace_reference" in value and isinstance(source_records, str)
        else None
    )
    for key, child in value.items():
        if key in {"trace_reference", "episode_trace_reference"} and isinstance(child, Mapping):
            _validate_trace_reference(
                child, capture_root, logical_root, trace_relative, records_relative
            )
        elif (
            key in {"trace_path", "docker_evidence_path"}
            or (key == "source_records_file" and records_relative is not None)
        ) and isinstance(child, str):
            relative_path = _captured_relative_path(child, logical_root, key)
            if not (capture_root / relative_path).is_file():
                raise FileNotFoundError(f"captured {key} does not exist")
        else:
            _validate_recorded_paths(child, capture_root, logical_root, trace_relative)


def _nested_field_values(value: object, field_name: str) -> Iterator[object]:
    if isinstance(value, list):
        for child in value:
            yield from _nested_field_values(child, field_name)
        return
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        if key == field_name:
            yield child
        else:
            yield from _nested_field_values(child, field_name)


def _validate_docker_evidence_pointer(record: Mapping[str, object], logical_root: Path) -> None:
    evidence_values = tuple(
        value for value in _nested_field_values(record, "docker_evidence_path") if value is not None
    )
    if not evidence_values:
        return
    trace_path_value = record.get("trace_path")
    if not isinstance(trace_path_value, str):
        raise TypeError("Docker evidence pointer requires its record's trace path")
    _captured_relative_path(trace_path_value, logical_root, "trace_path")
    expected_evidence_path = Path(trace_path_value).resolve().with_suffix(".docker-evidence.jsonl")
    for evidence_value in evidence_values:
        if not isinstance(evidence_value, str):
            raise TypeError("Docker evidence pointer must be a path string")
        _captured_relative_path(evidence_value, logical_root, "docker_evidence_path")
        if Path(evidence_value).resolve() != expected_evidence_path:
            raise ValueError("Docker evidence path must be adjacent to its trace")


def _validate_graded_artifacts(capture_root: Path, logical_root: Path) -> None:
    for record, artifact in _artifact_records(capture_root):
        trace_path_value = record.get("trace_path")
        digest = artifact.get("sha256")
        artifact_path_value = artifact.get("path")
        if not isinstance(trace_path_value, str) or not isinstance(digest, str):
            raise TypeError("corpus record needs a trace path and graded artifact digest")
        trace_relative = _captured_relative_path(trace_path_value, logical_root, "trace_path")
        stored_trace = capture_root / trace_relative
        if not stored_trace.is_file():
            raise FileNotFoundError("captured record trace path is missing")
        if artifact_path_value != str(PurePosixPath("graded-artifacts") / digest):
            raise ValueError("captured graded artifact path was not made corpus-relative")
        artifact_path = stored_trace.parent / "graded-artifacts" / digest
        if not artifact_path.is_file():
            raise FileNotFoundError("captured graded artifact is missing")
        observed_digest = _file_sha256(artifact_path)
        if observed_digest != digest:
            logger.warning(
                "captured graded artifact %s records sha256 %s but observed %s",
                artifact_path,
                digest,
                observed_digest,
            )


def _validate_docker_evidence(capture_root: Path) -> None:
    for path in capture_root.rglob("*.docker-evidence.jsonl"):
        read_jsonl(path, hydrate=False)


def _validate_capture_paths(capture_root: Path, logical_root: Path) -> None:
    record_files = sorted(capture_root.rglob("records.jsonl"))
    if not record_files:
        raise ValueError("captured run has no records.jsonl files")
    _validate_graded_artifacts(capture_root, logical_root)
    _validate_docker_evidence(capture_root)
    for records_path in record_files:
        for record in _json_records(records_path, hydrate=False):
            _validate_recorded_paths(record, capture_root, logical_root)
            _validate_docker_evidence_pointer(record, logical_root)
    for selection_path in capture_root.rglob("selection.json"):
        _validate_recorded_paths(
            json.loads(selection_path.read_text(encoding="utf-8")), capture_root, logical_root
        )


def _load_manifest(corpus_root: Path, *, allow_empty: bool = False) -> dict[str, object]:
    manifest_path = corpus_root / "corpus-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"seed-v4 corpus manifest is missing at {manifest_path}; capture a real run first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("seed-v4 corpus manifest must be an object")
    version = manifest.get("schema_version")
    if type(version) is not int or version < 1:
        raise ValueError("seed-v4 corpus manifest schema_version must be a positive integer")
    captures = manifest.get("captures")
    if not isinstance(captures, list):
        raise TypeError("seed-v4 corpus captures must be a list")
    if not captures and not allow_empty:
        raise ValueError("seed-v4 corpus is empty; capture a real run before running corpus tests")
    return cast("dict[str, object]", manifest)


@dataclass(frozen=True)
class _SemanticVersionCheckContext:
    capture_label: str
    exercised_producers: frozenset[str]
    capture_root: Path


def _semantic_version_is_compatible(
    producer: str,
    captured_version: object,
    current_version: object,
    context: _SemanticVersionCheckContext,
) -> bool:
    from reward_hacking.distill.agentic_sft import (  # noqa: PLC0415
        MANIFEST_VERSION,
        MIN_SUPPORTED_MANIFEST_VERSION,
    )
    from reward_hacking.distill.hack_strategy import (  # noqa: PLC0415
        LABEL_COMPATIBLE_TAXONOMY_VERSIONS,
    )
    from reward_hacking.organism_rl.evaluate import (  # noqa: PLC0415
        GENERATION_PATH_IDENTITY_COMPATIBLE_SCHEMA_VERSIONS,
    )
    from reward_hacking.organism_rl.generate_provenance import (  # noqa: PLC0415
        RECORD_SCHEMA_COMPATIBLE_VERSIONS,
    )

    loader_test = COMPATIBILITY_LOADER_TESTS.get(producer)
    if producer == ILCB_TRUE_CHECK_PRODUCER:
        return (
            captured_version is None
            and loader_test == ILCB_TRUE_CHECK_COMPATIBILITY_TEST
            and not _has_record_schema_at_least(context.capture_root, 4)
        )
    compatible_versions: Mapping[str, frozenset[int] | frozenset[str]] = {
        TAXONOMY_PRODUCER: LABEL_COMPATIBLE_TAXONOMY_VERSIONS,
        RECORD_SCHEMA_PRODUCER: RECORD_SCHEMA_COMPATIBLE_VERSIONS,
        LEAK_DETECTOR_PRODUCER: LEAK_DETECTOR_COMPATIBLE_SEMANTIC_VERSIONS,
        GENERATION_IDENTITY_PRODUCER: GENERATION_PATH_IDENTITY_COMPATIBLE_SCHEMA_VERSIONS,
        SFT_MANIFEST_PRODUCER: frozenset(
            range(MIN_SUPPORTED_MANIFEST_VERSION, MANIFEST_VERSION + 1)
        ),
    }
    if producer not in compatible_versions:
        return False
    supported_versions = compatible_versions[producer]
    return (
        loader_test is not None
        and type(captured_version) is type(current_version)
        and isinstance(captured_version, (int, str))
        and isinstance(current_version, (int, str))
        and captured_version in supported_versions
        and current_version in supported_versions
    )


def _check_semantic_version_table(
    captured_versions: Mapping[str, object],
    context: _SemanticVersionCheckContext,
) -> None:
    current_versions = semantic_versions()
    for producer, current_version in current_versions.items():
        if producer not in context.exercised_producers:
            continue
        captured_version = captured_versions.get(producer)
        if type(captured_version) is type(current_version) and captured_version == current_version:
            _check_detector_record_versions(producer, current_version, context.capture_root)
            continue
        compatible = _semantic_version_is_compatible(
            producer,
            captured_version,
            current_version,
            context,
        )
        if compatible:
            _check_detector_record_versions(
                producer, cast("int | str", captured_version), context.capture_root
            )
            continue
        raise ValueError(
            f"seed-v4 corpus {context.capture_label} captured under {producer}={captured_version}, "
            f"code is {producer}={current_version}; regenerate with `make corpus-smoke` or recapture "
            "the affected real run"
        )


def _check_detector_record_versions(
    producer: str, current_version: int | str, capture_root: Path
) -> None:
    if producer == LEAK_DETECTOR_PRODUCER and _recorded_leak_detector_versions(capture_root) != {
        current_version
    }:
        raise ValueError("detector assessment version differs from capture semantic version")


def _check_manifest_version_notes(manifest: Mapping[str, object]) -> None:
    versions = manifest.get("semantic_versions")
    if not isinstance(versions, Mapping):
        raise TypeError("seed-v4 corpus manifest has no semantic version table")
    legacy_detector_unversioned = LEAK_DETECTOR_PRODUCER not in versions and manifest.get(
        "deferred_semantic_versions"
    ) == list(LEGACY_DEFERRED_SEMANTIC_VERSIONS)
    if (
        manifest.get("deferred_semantic_versions") != list(DEFERRED_SEMANTIC_VERSIONS)
        and not legacy_detector_unversioned
    ):
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
            _SemanticVersionCheckContext(
                capture_label=f"capture {logical_relative}",
                exercised_producers=_exercised_producers(
                    capture_root, kind=raw_capture.get("kind")
                ),
                capture_root=capture_root,
            ),
        )
        _source_files(capture_root)
        _validate_capture_paths(capture_root, capture_root)
        recorded_file_sha256 = raw_capture.get("file_sha256")
        observed_file_sha256 = _capture_file_sha256(capture_root)
        if recorded_file_sha256 != observed_file_sha256:
            logger.warning(
                "capture %s file content manifest recorded %s but observed %s",
                logical_relative,
                recorded_file_sha256,
                observed_file_sha256,
            )
        if raw_capture.get("record_schema_shapes") != _record_schema_shapes(capture_root):
            raise ValueError(f"capture {logical_relative} record schema counts have changed")
    _check_size_limit(resolved_root)
    return manifest


def assert_required_captures(corpus_root: Path) -> None:
    """Require the real seed-v4 captures and their planned record cells."""
    resolved_root = corpus_root.resolve(strict=True)
    manifest = _load_manifest(resolved_root, allow_empty=True)
    captures = cast("list[object]", manifest["captures"])
    capture_counts: Counter[str] = Counter()
    for capture in captures:
        if not isinstance(capture, Mapping):
            raise TypeError("seed-v4 corpus capture entry must be an object")
        path_value = capture.get("path")
        if not isinstance(path_value, str):
            raise TypeError("seed-v4 corpus capture path must be text")
        capture_counts[path_value] += 1
    missing_captures = sorted(set(REQUIRED_CAPTURE_RECORDS) - capture_counts.keys())
    if missing_captures:
        raise ValueError("required seed-v4 captures are missing: " + ", ".join(missing_captures))
    duplicated_captures = sorted(
        path for path in REQUIRED_CAPTURE_RECORDS if capture_counts[path] != 1
    )
    if duplicated_captures:
        raise ValueError(
            "required seed-v4 captures must appear once: " + ", ".join(duplicated_captures)
        )
    missing_records = [
        f"{capture_path}/{record_path}"
        for capture_path, record_paths in REQUIRED_CAPTURE_RECORDS.items()
        for record_path in record_paths
        if not (resolved_root / capture_path / record_path).is_file()
    ]
    if missing_records:
        raise ValueError(
            "required seed-v4 capture records are missing: " + ", ".join(missing_records)
        )


def _exercised_producers(capture_root: Path, *, kind: object) -> frozenset[str]:
    """Return the producers whose output a capture holds; leak-builder bumps skip non-leak runs."""
    producers = set(semantic_versions())
    if kind != "smoke":
        producers.discard(SMOKE_PLAN_PRODUCER)
    if not any(_is_leak_record(record) for record in _capture_records(capture_root)):
        producers.discard(LEAK_BUILDER_PRODUCER)
    if not _recorded_leak_detector_versions(capture_root):
        producers.discard(LEAK_DETECTOR_PRODUCER)
    if not _has_strengthened_truth_sidecar(capture_root):
        producers.discard(STRENGTHENED_TRUTH_PRODUCER)
    if not any(_is_ilcb_record(record) for record in _capture_records(capture_root)):
        producers.discard(ILCB_TRUE_CHECK_PRODUCER)
    return frozenset(producers)


def _recorded_leak_detector_versions(capture_root: Path) -> set[int]:
    versions: set[int] = set()
    for record in _capture_records(capture_root):
        summary = record.get("trace_summary")
        if isinstance(summary, Mapping):
            assessment = summary.get("leak_assessment")
            if isinstance(assessment, Mapping) and "semantic_version" in assessment:
                version = assessment["semantic_version"]
                if version is None and assessment.get("leak_detector") == "not_applicable":
                    continue
                if type(version) is not int or version < 1:
                    raise ValueError(
                        "detector assessment semantic version must be a positive integer"
                    )
                versions.add(version)
    return versions


def _is_ilcb_record(record: Mapping[str, object]) -> bool:
    family = record.get("family")
    task_id = record.get("task_id")
    return family == "ilcb" or (isinstance(task_id, str) and task_id.startswith("ilcb-"))


def _has_record_schema_at_least(capture_root: Path, minimum_version: int) -> bool:
    for record in _capture_records(capture_root):
        record_schema = record.get("record_schema")
        truth_regrade = record.get("truth_regrade")
        has_versioned_ilcb_truth = (
            isinstance(truth_regrade, Mapping)
            and truth_regrade.get("producer") == ILCB_TRUE_CHECK_PRODUCER
        )
        if has_versioned_ilcb_truth or (
            type(record_schema) is int and record_schema >= minimum_version
        ):
            return True
    return False


def _has_strengthened_truth_sidecar(capture_root: Path) -> bool:
    """Report whether a capture holds strengthened-truth grades, which a grading bump would stale."""
    for path in sorted(capture_root.rglob("*.jsonl")):
        if _is_trace_file(path.relative_to(capture_root)) or path.name == "records.jsonl":
            continue
        for row in _json_records(path):
            if "grading_identity" in row and "strengthened_truth" in row:
                return True
            break
    return False


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


def _read_capture_source(source: Path) -> tuple[Path, int, Mapping[str, str]]:
    source_root = source.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"corpus source directory is missing: {source_root}")
    source_files = _source_files(source_root)
    if not any(path.name == "records.jsonl" for path in source_files):
        raise ValueError("corpus source has no records.jsonl file")
    source_inventory, source_size = _source_inventory(source_root)
    return source_root, source_size, source_inventory


def _prepare_capture_destination(
    kind: str,
    name: str,
    source_root: Path,
    source_size: int,
    corpus_root: Path | None,
) -> tuple[Path, Path, Path]:
    resolved_corpus_root = (corpus_root or corpus_root_from_environment()).resolve()
    destination_relative = _capture_relative_path(kind, name)
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
        manifest = _load_manifest(corpus_root, allow_empty=True)
        _check_manifest_version_notes(manifest)
        return manifest
    return {
        "schema_version": CORPUS_MANIFEST_SCHEMA_VERSION,
        "semantic_versions": dict(versions),
        "compatible_legacy_versions": [],
        "deferred_semantic_versions": list(DEFERRED_SEMANTIC_VERSIONS),
        "captures": [],
    }


def _copy_capture(
    source_root: Path,
    name: str,
    destination_root: Path,
    source_inventory: Mapping[str, str],
) -> None:
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
        current_source_inventory, _source_size = _source_inventory(source_root)
        if current_source_inventory != source_inventory:
            raise ValueError("corpus source changed during capture")
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


def _capture_relative_path(kind: str, name: str) -> Path:
    return Path(name) if kind == "smoke" else Path("nine-b") / name


def replace_smoke_capture(source: Path, name: str, *, corpus_root: Path | None = None) -> Path:
    """Swap a regenerated smoke run in for the previous capture of the same name.

    The previous capture is set aside, not deleted, until the new one is captured and the whole
    corpus verifies; a failed capture puts it back with its manifest entry.
    """
    _validate_capture_identity(name, "smoke")
    resolved_corpus_root = (corpus_root or corpus_root_from_environment()).resolve()
    relative_path = _capture_relative_path("smoke", name)
    existing_root = resolved_corpus_root / relative_path
    if not existing_root.is_dir():
        return capture_run(source, name, kind="smoke", corpus_root=resolved_corpus_root)
    manifest = _load_manifest(resolved_corpus_root)
    captures = cast("list[Mapping[str, object]]", manifest["captures"])
    remaining = [capture for capture in captures if capture.get("path") != str(relative_path)]
    if len(remaining) != len(captures) - 1:
        raise ValueError(
            f"corpus manifest must list the existing smoke capture {relative_path} once"
        )
    # Outside the corpus root, so the capture being replaced never counts against the corpus size limit.
    retired_root = resolved_corpus_root.with_name(
        f".{resolved_corpus_root.name}-{name}.retired-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    )
    existing_root.rename(retired_root)
    _atomic_write_json(
        resolved_corpus_root / "corpus-manifest.json", {**manifest, "captures": remaining}
    )
    try:
        destination = capture_run(source, name, kind="smoke", corpus_root=resolved_corpus_root)
    except Exception:
        logger.exception("smoke capture failed; restoring the previous capture")
        if existing_root.exists():
            shutil.rmtree(existing_root)
        retired_root.rename(existing_root)
        _atomic_write_json(resolved_corpus_root / "corpus-manifest.json", manifest)
        raise
    shutil.rmtree(retired_root)
    logger.info("replaced smoke capture %s", relative_path)
    return destination


def capture_run(
    source: Path,
    name: str,
    *,
    kind: str = "eval",
    corpus_root: Path | None = None,
) -> Path:
    """Copy one completed run into the ignored corpus and make all recorded paths self-contained."""
    _validate_capture_identity(name, kind)
    source_root, source_size, source_inventory = _read_capture_source(source)
    resolved_corpus_root, destination_relative, destination_root = _prepare_capture_destination(
        kind,
        name,
        source_root,
        source_size,
        corpus_root,
    )
    versions = semantic_versions()
    current_manifest = _load_or_initialize_manifest(resolved_corpus_root, versions)
    manifest_path = resolved_corpus_root / "corpus-manifest.json"
    previous_manifest_bytes = manifest_path.read_bytes() if manifest_path.exists() else None
    readme_path = resolved_corpus_root / "README.md"
    readme_existed = readme_path.exists()
    corpus_existed = resolved_corpus_root.exists()
    parent_existed = destination_root.parent.exists()
    completed = False
    try:
        _copy_capture(source_root, name, destination_root, source_inventory)
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
        verify_corpus(resolved_corpus_root)
        if not readme_existed:
            readme_path.write_text(CORPUS_README, encoding="utf-8")
        completed = True
    finally:
        if not completed:
            if destination_root.exists():
                shutil.rmtree(destination_root)
            if previous_manifest_bytes is None:
                manifest_path.unlink(missing_ok=True)
            else:
                manifest_path.write_bytes(previous_manifest_bytes)
            if not readme_existed:
                readme_path.unlink(missing_ok=True)
            if not parent_existed and destination_root.parent.exists():
                destination_root.parent.rmdir()
            if not corpus_existed and resolved_corpus_root.exists():
                resolved_corpus_root.rmdir()
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
        replace_smoke_capture(run_directory, "smoke-0.8b", corpus_root=args.corpus_root)
    else:
        raise AssertionError(f"unhandled corpus command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
