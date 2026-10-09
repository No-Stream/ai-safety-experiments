"""Assert the record and artifact contracts for the 0.8B v4 smoke run."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)


class SmokeAssertionError(ValueError):
    """A smoke-run record or artifact did not meet its declared contract."""


@dataclass(frozen=True, slots=True)
class CellExpectation:
    """The expected record count and schema for one generated cell."""

    name: str
    records_path: Path
    expected_count: int
    payment_register: str
    expected_leak_channels_by_task: Mapping[str, str] | None = None
    require_leak_evidence: bool = False
    require_math_evidence: bool = False


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise SmokeAssertionError(f"records file is missing: {path}")
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SmokeAssertionError(f"{path.name} line {line_number} is not a JSON object")
            records.append(cast("dict[str, object]", value))
    return records


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SmokeAssertionError(f"{context} must be a JSON object")
    return cast("Mapping[str, object]", value)


def assert_cell_records(expectation: CellExpectation) -> None:
    """Validate count, register, hint, and applicable family-specific evidence."""
    records = _read_jsonl(expectation.records_path)
    if len(records) != expectation.expected_count:
        raise SmokeAssertionError(
            f"{expectation.name}: record count {len(records)} != {expectation.expected_count}"
        )
    if expectation.expected_count < 1:
        raise SmokeAssertionError(f"{expectation.name}: expected record count must be positive")

    seen_task_ids: set[str] = set()
    for record in records:
        _assert_one_cell_record(record, expectation, seen_task_ids)

    if expectation.expected_leak_channels_by_task is not None:
        expected_task_ids = set(expectation.expected_leak_channels_by_task)
        if seen_task_ids != expected_task_ids:
            raise SmokeAssertionError(f"{expectation.name}: leak task set differs from eval_pairs")


def _assert_one_cell_record(
    record: Mapping[str, object],
    expectation: CellExpectation,
    seen_task_ids: set[str],
) -> None:
    if record.get("payment_register") != expectation.payment_register:
        raise SmokeAssertionError(
            f"{expectation.name}: payment_register is missing or differs at record top level"
        )
    hint = record.get("hint")
    if not isinstance(hint, Mapping):
        raise SmokeAssertionError(f"{expectation.name}: hint field is missing or invalid")
    hint_strength = hint.get("strength")
    if not isinstance(hint_strength, str) or hint_strength not in {"none", "vague"}:
        raise SmokeAssertionError(f"{expectation.name}: hint strength must be none or vague")

    if expectation.expected_leak_channels_by_task is not None:
        _assert_leak_record(record, expectation, seen_task_ids)
    elif expectation.require_leak_evidence:
        _assert_leak_evidence(record, expectation.name)
    if expectation.require_math_evidence and not isinstance(record.get("math_evidence"), Mapping):
        raise SmokeAssertionError(f"{expectation.name}: math_evidence is missing or invalid")


def _assert_leak_record(
    record: Mapping[str, object],
    expectation: CellExpectation,
    seen_task_ids: set[str],
) -> None:
    expected_channels = expectation.expected_leak_channels_by_task
    if expected_channels is None:
        raise SmokeAssertionError(f"{expectation.name}: leak channel expectations are missing")
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or task_id not in expected_channels:
        raise SmokeAssertionError(
            f"{expectation.name}: leak record is not in the loaded eval_pairs"
        )
    if task_id in seen_task_ids:
        raise SmokeAssertionError(f"{expectation.name}: duplicate leak eval_pair record")
    seen_task_ids.add(task_id)

    trace_summary = _mapping(record.get("trace_summary"), context="trace_summary")
    opened_channels = trace_summary.get("leak_channels_open")
    if not isinstance(opened_channels, list) or len(opened_channels) != 1:
        raise SmokeAssertionError(f"{expectation.name}: expected exactly one open leak channel")
    if opened_channels[0] != expected_channels[task_id]:
        raise SmokeAssertionError(
            f"{expectation.name}: leak channel differs from loaded arm channel"
        )
    _assert_leak_evidence(record, expectation.name, trace_summary=trace_summary)


def _assert_leak_evidence(
    record: Mapping[str, object],
    name: str,
    *,
    trace_summary: Mapping[str, object] | None = None,
) -> None:
    if trace_summary is None:
        trace_summary = _mapping(record.get("trace_summary"), context="trace_summary")
    if type(trace_summary.get("leak_repo_seen")) is not bool:
        raise SmokeAssertionError(f"{name}: leak_repo_seen flag is missing or invalid")
    leak_evidence = _mapping(trace_summary.get("leak_evidence"), context="leak_evidence")
    docker_evidence = leak_evidence.get("docker_evidence")
    if not isinstance(docker_evidence, list) or not docker_evidence:
        raise SmokeAssertionError(f"{name}: Docker evidence is missing")
    _assert_actor_execution(leak_evidence, docker_evidence, name)
    _mapping(trace_summary.get("leak_assessment"), context="leak assessment")


def _assert_actor_execution(
    leak_evidence: Mapping[str, object], docker_evidence: list[object], name: str
) -> None:
    trace_records = leak_evidence.get("trace_records")
    if not isinstance(trace_records, list):
        raise SmokeAssertionError(f"{name}: successful actor-side command evidence is missing")
    successful_commands = _successful_actor_commands(trace_records, name)
    for raw_evidence in docker_evidence:
        evidence = _mapping(raw_evidence, context="Docker command evidence")
        wrapper = evidence.get("command")
        if (
            evidence.get("phase") != "finished"
            or type(evidence.get("exit_code")) is not int
            or evidence["exit_code"] != 0
            or evidence.get("timed_out") is not False
            or "snapshot_error" in evidence
            or not isinstance(wrapper, str)
            or "__LADDER_SHELL_STATE_" not in wrapper
        ):
            continue
        if not any(
            f"if builtin eval -- {shlex.quote(command)}; then" in wrapper
            for command in successful_commands
        ):
            continue
        if all(_verified_checkout_snapshot(evidence.get(phase)) for phase in ("before", "after")):
            return
    raise SmokeAssertionError(
        f"{name}: no successful actor-side command with verified workspace snapshots"
    )


def _successful_actor_commands(trace_records: list[object], name: str) -> set[str]:
    successful_commands: set[str] = set()
    for raw_trace in trace_records:
        trace = _mapping(raw_trace, context="leak trace record")
        if trace.get("record") != "turn":
            continue
        commands = trace.get("commands")
        if not isinstance(commands, list):
            raise SmokeAssertionError(f"{name}: actor commands must be a list")
        for raw_command in commands:
            command = _mapping(raw_command, context="actor command")
            text = command.get("command")
            if (
                isinstance(text, str)
                and type(command.get("exit_code")) is int
                and command["exit_code"] == 0
                and command.get("timed_out") is False
            ):
                successful_commands.add(text)
    return successful_commands


def _verified_checkout_snapshot(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    head = value.get(".git/HEAD")
    if not isinstance(head, Mapping):
        return False
    digest = head.get("sha256")
    return isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None


def assert_ladder_records(records_path: Path, *, expected_count: int) -> None:
    """Require a ladder episode and its acceptance and protected-path evidence fields."""
    records = _read_jsonl(records_path)
    if len(records) != expected_count:
        raise SmokeAssertionError(f"ladder: record count {len(records)} != {expected_count}")
    for record in records:
        _assert_one_ladder_record(record)


def _assert_one_ladder_record(record: Mapping[str, object]) -> None:
    if record.get("record") != "ladder_episode":
        raise SmokeAssertionError("ladder: expected a completed ladder_episode record")
    trace_summary = _mapping(record.get("trace_summary"), context="ladder trace_summary")
    submission_verdicts = trace_summary.get("submission_verdicts")
    if not isinstance(submission_verdicts, list):
        raise SmokeAssertionError("ladder: submission verdict evidence is missing")
    acceptance_values = [
        verdict["acceptance_paths_changed"]
        for verdict in submission_verdicts
        if isinstance(verdict, Mapping) and "acceptance_paths_changed" in verdict
    ]
    if "acceptance_paths_changed" in trace_summary:
        acceptance_values.append(trace_summary["acceptance_paths_changed"])
    if "acceptance_paths_changed" in record:
        acceptance_values.append(record["acceptance_paths_changed"])
    if not acceptance_values or any(type(value) is not bool for value in acceptance_values):
        raise SmokeAssertionError("ladder: acceptance_paths_changed evidence is missing or invalid")

    behaviour_events = _mapping(record.get("behaviour_events"), context="ladder behaviour_events")
    protected_values = [
        event_source["protected_path_moved"]
        for event_source in (behaviour_events, record, trace_summary)
        if "protected_path_moved" in event_source
    ]
    if not protected_values:
        raise SmokeAssertionError("ladder: protected_path_moved evidence is missing")
    if any(type(value) is not bool for value in protected_values):
        raise SmokeAssertionError("ladder: protected_path_moved evidence is invalid")


def assert_card_written(card_path: Path) -> None:
    """Require the smoke-only dataset card artifact to be a JSON object."""
    if not card_path.is_file():
        raise SmokeAssertionError(f"card.json is missing: {card_path}")
    value = json.loads(card_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SmokeAssertionError("card.json must contain a JSON object")
    if value.get("schema_only") is not True:
        raise SmokeAssertionError("card.json must identify its schema_only scope")


def _record_identity(record: Mapping[str, object], *, context: str) -> tuple[str, int]:
    task_id = record.get("task_id")
    sample_index = record.get("sample_index")
    if not isinstance(task_id, str) or type(sample_index) is not int:
        raise SmokeAssertionError(
            f"{context}: task_id and sample_index are required for resume checks"
        )
    return task_id, sample_index


def assert_resume_completion(
    before_path: Path, after_path: Path, *, expected_total: int
) -> dict[str, int]:
    """Count the interrupted record as resumed and require exactly one newly generated remainder."""
    before_records = _read_jsonl(before_path)
    after_records = _read_jsonl(after_path)
    before_identities = [
        _record_identity(record, context="interrupted run") for record in before_records
    ]
    after_identities = [_record_identity(record, context="resumed run") for record in after_records]
    if len(before_identities) != 1:
        raise SmokeAssertionError(
            f"resume: expected one interrupted record, found {len(before_identities)}"
        )
    if len(after_identities) != expected_total:
        raise SmokeAssertionError(
            f"resume: completed record count {len(after_identities)} != {expected_total}"
        )
    if len(set(after_identities)) != len(after_identities):
        raise SmokeAssertionError("resume: duplicate episode identity after relaunch")
    if not set(before_identities).issubset(after_identities):
        raise SmokeAssertionError("resume: completed output lost an interrupted episode")
    resumed_count = len(before_identities)
    new_count = len(after_identities) - resumed_count
    if new_count != 1:
        raise SmokeAssertionError(f"resume: expected one new remainder episode, found {new_count}")
    return {"resumed": resumed_count, "new": new_count, "total": len(after_identities)}


def assert_smoke_run(run_directory: Path) -> dict[str, object]:
    """Load the safe smoke plan and assert every generated cell and output artifact."""
    plan_path = run_directory / "smoke-plan.json"
    plan_value = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = _mapping(plan_value, context="smoke plan")
    cells_value = plan.get("cells")
    if not isinstance(cells_value, list):
        raise SmokeAssertionError("smoke plan cells must be a list")

    cell_results: dict[str, int] = {}
    for cell_value in cells_value:
        cell = _mapping(cell_value, context="smoke plan cell")
        name = _string_field(cell, "name")
        records_path = run_directory / _string_field(cell, "records_path")
        expected_leak_channels = cell.get("expected_leak_channels_by_task")
        if expected_leak_channels is not None:
            expected_leak_channels = cast(
                "Mapping[str, str]", _mapping(expected_leak_channels, context="leak channel plan")
            )
        expectation = CellExpectation(
            name=name,
            records_path=records_path,
            expected_count=_int_field(cell, "expected_count"),
            payment_register=_string_field(cell, "payment_register"),
            expected_leak_channels_by_task=expected_leak_channels,
            require_leak_evidence=cell.get("require_leak_evidence") is True,
            require_math_evidence=cell.get("require_math_evidence") is True,
        )
        assert_cell_records(expectation)
        cell_results[name] = expectation.expected_count

    ladder_path = run_directory / _string_field(plan, "ladder_records_path")
    ladder_count = _int_field(plan, "ladder_expected_count")
    assert_ladder_records(ladder_path, expected_count=ladder_count)
    card_path = run_directory / _string_field(plan, "card_path")
    assert_card_written(card_path)
    resume = _mapping(plan.get("resume"), context="resume plan")
    resume_counts = assert_resume_completion(
        run_directory / _string_field(resume, "before_records_path"),
        run_directory / _string_field(resume, "after_records_path"),
        expected_total=_int_field(resume, "expected_total"),
    )
    return {"cells": cell_results, "ladder_records": ladder_count, "resume": resume_counts}


def _string_field(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise SmokeAssertionError(f"smoke plan field {key!r} must be a nonempty string")
    return value


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int or value < 1:
        raise SmokeAssertionError(f"smoke plan field {key!r} must be a positive integer")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run all assertions from the plan saved by the smoke runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        result = assert_smoke_run(arguments.run_dir)
    except SmokeAssertionError:
        logger.exception("smoke assertion failed")
        return 1
    logger.info("v4 0.8B smoke assertions passed: %s", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
