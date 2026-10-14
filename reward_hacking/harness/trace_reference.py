"""Resolve persisted leak detector inputs without embedding the episode trace twice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from reward_hacking.evidence_store import read_jsonl


def leak_trace_records(
    evidence: Mapping[str, object],
    *,
    record: Mapping[str, object],
    records_path: Path | None = None,
    trace_records: Sequence[Mapping[str, object]] | None = None,
) -> tuple[Mapping[str, object], ...]:
    """Read legacy embedded inputs or the prefix before the finalized episode summary."""
    embedded = evidence.get("trace_records")
    if embedded is not None:
        if not isinstance(embedded, list) or any(not isinstance(row, Mapping) for row in embedded):
            raise TypeError("persisted leak_evidence.trace_records must be a list of objects")
        return tuple(cast("Mapping[str, object]", row) for row in embedded)
    reference = evidence.get("trace_reference")
    if not isinstance(reference, Mapping):
        raise TypeError("persisted leak evidence requires trace_records or trace_reference")
    if reference.get("locator") != "before_episode_summary":
        raise ValueError("leak trace reference locator must be before_episode_summary")
    path_text = reference.get("trace_path")
    if not isinstance(path_text, str):
        raise TypeError("leak trace reference trace_path must be a string")
    rows = trace_records or _read_finalized_trace(_reference_path(path_text, record, records_path))
    prefix: list[Mapping[str, object]] = []
    for row in rows:
        if row.get("record") == "episode_summary":
            break
        prefix.append(row)
    return tuple(prefix)


def _reference_path(
    path_text: str, record: Mapping[str, object], records_path: Path | None
) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if records_path is not None:
        return records_path.parent / path
    recorded_path = record.get("trace_path")
    if not isinstance(recorded_path, str) or not Path(recorded_path).is_absolute():
        raise ValueError(
            "relative leak trace reference requires records_path or absolute trace_path"
        )
    return Path(recorded_path)


def _read_finalized_trace(path: Path) -> list[Mapping[str, object]]:
    rows = read_jsonl(path)
    if not any(row.get("record") == "episode_summary" for row in rows):
        raise ValueError("referenced leak trace has no finalized episode_summary")
    return [cast("Mapping[str, object]", row) for row in rows]
