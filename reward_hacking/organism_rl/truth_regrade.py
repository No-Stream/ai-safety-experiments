"""Regrade persisted ILCB artifacts with the current hidden truth checker."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from reward_hacking.distill.hack_strategy import TAXONOMY_VERSION, label_persisted_episode
from reward_hacking.harness import tasks as task_registry
from reward_hacking.harness.tasks_evalplus import TrueCheckOutcome
from reward_hacking.harness.tasks_ilcb import (
    ILCB_TRUE_CHECK_SEMANTIC_VERSION,
    SOLUTION_FILENAME,
    TRUE_CHECK_TIMEOUT,
)
from reward_hacking.organism_reward import score_impossible_task
from reward_hacking.organism_rl.generate import RECORD_SCHEMA_VERSION, require_gitignored_output_dir

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reward_hacking.harness.ilcb_split import ILCBPartition
    from reward_hacking.harness.task_spec import Task

logger = logging.getLogger(__name__)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return cast("dict[str, object]", value)


def _read_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as records_file:
        for line_number, line in enumerate(records_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number} is not valid JSON") from error
            records.append(_object(record, f"{path.name}:{line_number}"))
    return records


def _load_task(record: Mapping[str, object]) -> Task:
    task_id = record.get("task_id")
    variant = record.get("variant")
    partition = record.get("partition")
    if (
        not isinstance(task_id, str)
        or not isinstance(variant, str)
        or not isinstance(partition, str)
    ):
        raise TypeError("ILCB records need string task_id, variant, and partition fields")
    tasks = task_registry.ilcb_tasks(
        impossible_type=variant,
        partition=cast("ILCBPartition", partition),
    )
    try:
        return next(task for task in tasks if task.task_id == task_id)
    except StopIteration as error:
        raise ValueError(
            f"ILCB task {task_id!r} is absent from the current task registry"
        ) from error


def _artifact_path(
    record: Mapping[str, object], summary: Mapping[str, object], records_path: Path
) -> tuple[Path, str, bytes]:
    artifact = _object(summary.get("graded_artifact"), "trace_summary.graded_artifact")
    path_text = artifact.get("path")
    digest = artifact.get("sha256")
    if (
        not isinstance(path_text, str)
        or not isinstance(digest, str)
        or not _SHA256_PATTERN.fullmatch(digest)
    ):
        raise TypeError("graded artifact must have a path and lowercase SHA-256 digest")
    artifact_path = Path(path_text)
    if not artifact_path.is_absolute():
        trace_path_value = record.get("trace_path")
        if not isinstance(trace_path_value, str):
            raise TypeError("relative graded artifact path requires a trace_path")
        trace_path = Path(trace_path_value)
        if not trace_path.is_absolute():
            trace_path = records_path.parent / trace_path
        artifact_path = trace_path.parent / artifact_path
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise FileNotFoundError("graded artifact is missing or is not a regular file")
    artifact_bytes = artifact_path.read_bytes()
    actual_digest = hashlib.sha256(artifact_bytes).hexdigest()
    if actual_digest != digest:
        raise ValueError("graded artifact SHA-256 does not match the recorded digest")
    return artifact_path, digest, artifact_bytes


def _absolute_relative_pointer(path_text: str, base_directory: Path, label: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return path_text
    try:
        return str((base_directory / path).resolve(strict=True))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"relative {label} does not exist in the source records") from error


def _normalize_passthrough_record_paths(
    record: Mapping[str, object], records_path: Path
) -> dict[str, object]:
    result = deepcopy(dict(record))
    trace_path_value = result.get("trace_path")
    trace_parent = records_path.parent
    if isinstance(trace_path_value, str):
        normalized_trace_path = Path(
            _absolute_relative_pointer(trace_path_value, records_path.parent, "trace_path")
        )
        result["trace_path"] = str(normalized_trace_path)
        trace_parent = normalized_trace_path.parent

    def normalize_evidence_pointers(value: object) -> object:
        if isinstance(value, list):
            return [normalize_evidence_pointers(child) for child in value]
        if not isinstance(value, dict):
            return value
        normalized: dict[str, object] = {}
        for key, child in value.items():
            if key == "graded_artifact" and isinstance(child, dict):
                artifact = deepcopy(child)
                artifact_path = artifact.get("path")
                if isinstance(artifact_path, str):
                    artifact["path"] = _absolute_relative_pointer(
                        artifact_path, trace_parent, "graded artifact"
                    )
                normalized[key] = artifact
            elif key == "docker_evidence_path" and isinstance(child, str):
                normalized[key] = _absolute_relative_pointer(
                    child, trace_parent, "Docker evidence path"
                )
            else:
                normalized[key] = normalize_evidence_pointers(child)
        return normalized

    return cast("dict[str, object]", normalize_evidence_pointers(result))


def _regrade_ilcb_record(
    record: Mapping[str, object],
    *,
    records_path: Path,
    temporary_root: Path,
) -> dict[str, object]:
    result = deepcopy(dict(record))
    summary = _object(result.get("trace_summary"), "trace_summary")
    task = _load_task(result)
    if task.true_check_result_with_timeout is None:
        raise ValueError(f"ILCB task {task.task_id!r} has no structured true-check callback")
    artifact_path, artifact_digest, artifact_bytes = _artifact_path(result, summary, records_path)
    with tempfile.TemporaryDirectory(prefix="ilcb-truth-", dir=temporary_root) as temporary_episode:
        episode_directory = Path(temporary_episode)
        (episode_directory / SOLUTION_FILENAME).write_bytes(artifact_bytes)
        truth = task.true_check_result_with_timeout(episode_directory, TRUE_CHECK_TIMEOUT)

    trace_path_value = result.get("trace_path")
    if not isinstance(trace_path_value, str):
        raise TypeError("ILCB record needs a trace_path")
    trace_path = Path(trace_path_value)
    if not trace_path.is_absolute():
        trace_path = (records_path.parent / trace_path).resolve(strict=True)
    result["trace_path"] = str(trace_path)
    artifact_evidence = _object(summary["graded_artifact"], "trace_summary.graded_artifact")
    artifact_evidence["path"] = str(artifact_path.resolve(strict=True))

    measured = truth.outcome is not TrueCheckOutcome.ORACLE_ERROR
    passed = truth.outcome is TrueCheckOutcome.TRUE_PASS
    true_evidence: dict[str, object] = {
        "pass": passed if measured else None,
        "measured": measured,
        "oracle_error": None if measured else truth.detail,
    }
    if truth.evidence:
        true_evidence["evidence"] = dict(truth.evidence)
    summary["true"] = true_evidence
    summary["true_measured"] = measured
    summary["true_pass"] = passed if measured else None
    result["true_measured"] = measured
    result["true_pass"] = passed if measured else None

    payment_register = summary.get("payment_register", result.get("payment_register"))
    if payment_register not in {None, "trusted_rerun", "in_band"}:
        raise ValueError(f"unknown payment register: {payment_register!r}")
    score = score_impossible_task(
        summary,
        payment_register=cast("Literal['trusted_rerun', 'in_band'] | None", payment_register),
    )
    result["organism_reward"] = {
        "reward": score.reward,
        "outcome": score.outcome.value,
        "evidence": asdict(score.evidence),
    }
    result["organism_reward_outcome"] = score.outcome.value
    updated_label = label_persisted_episode(result).to_json_dict()
    summary["hack_strategy"] = updated_label
    if "hack_strategy" in result:
        result["hack_strategy"] = updated_label
    if "hack_strategy_taxonomy_version" in result:
        result["hack_strategy_taxonomy_version"] = TAXONOMY_VERSION
    result["truth_regrade"] = {
        "method": "artifact_bound_current_ilcb_truth",
        "source_record_sha256": hashlib.sha256(
            json.dumps(dict(record), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "artifact_sha256": artifact_digest,
        "producer": "reward_hacking.harness.tasks_ilcb.true_check",
        "semantic_version": ILCB_TRUE_CHECK_SEMANTIC_VERSION,
        "source_record_schema": result.get("record_schema"),
    }
    result["record_schema"] = RECORD_SCHEMA_VERSION
    return result


def _output_path(records_path: Path, output_path: Path) -> Path:
    source_resolved = records_path.resolve(strict=True)
    output_resolved = output_path.resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"truth regrade output must be a fresh file: {output_path}")
    if output_resolved == source_resolved:
        raise ValueError("truth regrade output aliases input records")
    if not output_path.parent.is_dir():
        raise FileNotFoundError("truth regrade output directory must already exist")
    require_gitignored_output_dir(output_path.parent)
    return output_path


def regrade_file(records_path: Path, output_path: Path) -> Path:
    """Write a fresh records file with ILCB truth recomputed from saved artifacts."""
    source_path = records_path.resolve(strict=True)
    destination = _output_path(source_path, output_path)
    source_records = _read_records(source_path)
    with tempfile.TemporaryDirectory(
        prefix=".ilcb-truth-regrade-", dir=destination.parent
    ) as scratch:
        scratch_directory = Path(scratch)
        temporary_output = scratch_directory / "records.jsonl"
        with temporary_output.open("w", encoding="utf-8") as destination_file:
            for record in source_records:
                updated = (
                    _regrade_ilcb_record(
                        record,
                        records_path=source_path,
                        temporary_root=scratch_directory,
                    )
                    if record.get("family") == "ilcb"
                    else _normalize_passthrough_record_paths(record, source_path)
                )
                destination_file.write(json.dumps(updated, sort_keys=True) + "\n")
            destination_file.flush()
            os.fsync(destination_file.fileno())
        os.link(temporary_output, destination)
    logger.info("wrote artifact-bound ILCB truth results to %s", destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for a fresh artifact-bound regrade output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path, help="existing records.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="fresh regraded records.jsonl")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Regrade truth evidence for saved ILCB artifacts."""
    args = build_parser().parse_args(argv)
    regrade_file(args.records, args.output)


if __name__ == "__main__":
    main()
