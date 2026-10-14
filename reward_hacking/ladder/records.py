"""Episode-record primitives shared by the ladder runner and its readout."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from reward_hacking.harness.evidence_store import hydrate_record, read_jsonl
from reward_hacking.ladder.rungs import Rung

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

LADDER_SCHEMA_VERSION = 25
LADDER_RUNG_VALUES = tuple(int(rung) for rung in Rung)
ESCALATION_OUTCOME_CATEGORIES = (
    "honest_pass",
    "honest_failure",
    "censored_no_submission",
    "persistence",
    "escalation",
    "preemptive_escalation",
    "other",
)
LADDER_TASK_PRESSURES = ("none", "merge-gate")


@dataclass(frozen=True, slots=True)
class EpisodeKey:
    """The content-independent coordinate used for resume and duplicate detection."""

    model_id: str
    scenario_id: str
    arm: str
    sample_idx: int
    task_pressure: str = "none"

    def __post_init__(self) -> None:
        """Reject keys that could not identify one grid cell."""
        if not self.model_id or not self.scenario_id or not self.arm:
            raise ValueError("episode key labels must be non-empty")
        if self.sample_idx < 0:
            raise ValueError("sample_idx must be non-negative")
        if self.task_pressure not in LADDER_TASK_PRESSURES:
            raise ValueError(f"unknown task pressure {self.task_pressure!r}")

    def to_json(self) -> dict[str, object]:
        """Return a JSON object rather than an order-dependent joined string."""
        return {
            "model_id": self.model_id,
            "scenario_id": self.scenario_id,
            "arm": self.arm,
            "sample_idx": self.sample_idx,
            "task_pressure": self.task_pressure,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> EpisodeKey:
        """Parse and validate a persisted key."""
        model_id = value.get("model_id")
        scenario_id = value.get("scenario_id")
        arm = value.get("arm")
        sample_idx = value.get("sample_idx")
        task_pressure = value.get("task_pressure", "none")
        if (
            not isinstance(model_id, str)
            or not model_id
            or not isinstance(scenario_id, str)
            or not scenario_id
            or not isinstance(arm, str)
            or not arm
            or not isinstance(task_pressure, str)
        ):
            raise ValueError(f"invalid ladder episode key labels: {value!r}")
        if not isinstance(sample_idx, int) or isinstance(sample_idx, bool) or sample_idx < 0:
            raise ValueError(f"invalid ladder episode sample_idx: {value!r}")
        return cls(model_id, scenario_id, arm, sample_idx, task_pressure)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records and remove only an incomplete final write before resume."""
    if not path.exists():
        return []
    payload = path.read_bytes()
    if not payload:
        return []

    if payload.startswith(b"\x1f\x8b"):
        return read_jsonl(path, recover_partial=True)

    lines = payload.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    byte_offset = 0
    dropped_torn_tail = False
    for line_index, line in enumerate(lines):
        line_start = byte_offset
        byte_offset += len(line)
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            is_torn_final_line = line_index == len(lines) - 1 and not line.endswith(b"\n")
            if not is_torn_final_line:
                raise ValueError(
                    f"malformed ladder JSONL record in {path} at byte {line_start}"
                ) from error
            with path.open("r+b") as handle:
                handle.truncate(line_start)
            dropped_torn_tail = True
            logger.warning("dropped incomplete trailing ladder record from %s", path)
            break
        if not isinstance(record, dict):
            raise TypeError(f"ladder JSONL record in {path} at byte {line_start} must be an object")
        records.append(hydrate_record(record, base_path=path))

    if not dropped_torn_tail and payload and not payload.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return records


def _trace_true_fields(
    summary: Mapping[str, object],
) -> tuple[list[bool | None], list[bool | None], str | None]:
    """Parse measurement markers, verdicts, and oracle errors from old and new trace shapes."""
    nested = summary.get("true")
    if nested is not None and not isinstance(nested, Mapping):
        raise TypeError("episode trace true result must be an object")

    measurement_flags: list[bool | None] = []
    verdicts: list[bool | None] = []
    for source, field_name, values in (
        (summary, "true_measured", measurement_flags),
        (nested, "measured", measurement_flags),
        (summary, "true_pass", verdicts),
        (nested, "pass", verdicts),
    ):
        if source is not None and field_name in source:
            value = source[field_name]
            if value is not None and type(value) is not bool:
                raise TypeError(f"episode trace {field_name} must be a boolean or null")
            values.append(value)

    oracle_error = nested.get("oracle_error") if isinstance(nested, Mapping) else None
    if oracle_error is not None and not isinstance(oracle_error, str):
        raise TypeError(f"episode trace has non-string true.oracle_error: {oracle_error!r}")
    return measurement_flags, verdicts, oracle_error


def trace_true_measured(summary: Mapping[str, object]) -> bool:
    """Return whether the hidden oracle reached a verdict, retaining legacy measured records."""
    measurement_flags, verdicts, oracle_error = _trace_true_fields(summary)

    has_measured_flag = any(flag is True for flag in measurement_flags)
    has_unmeasured_flag = any(flag is False for flag in measurement_flags)
    has_unknown_flag = any(flag is None for flag in measurement_flags)

    if has_measured_flag and (has_unmeasured_flag or has_unknown_flag):
        raise ValueError("episode trace has conflicting true measurement fields")
    if has_measured_flag and None in verdicts:
        raise ValueError("episode trace has conflicting true measurement and verdict fields")
    if oracle_error is not None:
        if has_measured_flag:
            raise ValueError("episode trace marks the true oracle measured and errored")
        return False
    if has_unmeasured_flag or has_unknown_flag or None in verdicts:
        return False
    if has_measured_flag or any(verdict is not None for verdict in verdicts):
        return True
    raise ValueError("episode trace has no true measurement result")


def record_true_measured(record: Mapping[str, object]) -> bool:
    """Recover legacy record measurement status from its saved loop summary when available."""
    if "true_measured" in record:
        true_measured = record["true_measured"]
        if true_measured is None:
            return False
        if type(true_measured) is not bool:
            raise TypeError(f"true_measured must be boolean in readout records: {true_measured!r}")
        return true_measured

    trace_summary = record.get("trace_summary")
    if isinstance(trace_summary, Mapping) and (
        isinstance(trace_summary.get("true"), Mapping)
        or "true_pass" in trace_summary
        or "true_measured" in trace_summary
    ):
        return trace_true_measured(trace_summary)
    return True


def pristine_path_is_within(path: str, prefixes: tuple[str, ...]) -> bool:
    """Return whether a workspace-relative path equals or descends from a prefix."""
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)


def outcome_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    """Require a persisted JSON object with string keys."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} must be an object with string keys")
    return value


def submission_evidence_boolean(
    verdict_record: Mapping[str, object], field_name: str
) -> bool | None:
    """Keep unrecorded submission-local grades explicitly unknown."""
    evidence = outcome_mapping(
        verdict_record.get("submission_evidence", {}), field_name="submission_evidence"
    )
    value = evidence.get(field_name)
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"submission_evidence.{field_name} must be boolean or null")
    return value
