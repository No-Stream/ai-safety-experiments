"""Bind trusted final-implementation test replays to captured source changes.

This consumer supports replays of the final implementation only. Receipts for an earlier
implementation remain retained, with an explicit unsupported-binding reason. Equal exit verdicts
establish a limited unchanged observation and do not establish general semantic equivalence.
The trusted replay producer binds comparison context to the unchanged implementation, remaining
suite, runner and environment; actor-supplied context claims are not trusted replay receipts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import cast

_PYTEST_VERDICT_CODES = frozenset({0, 1})


def _object(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("test-effect evidence must be an object")
    return cast("Mapping[str, object]", value)


def _rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise TypeError("test-effect evidence must be a list")
    return [_object(row) for row in value]


def _source_digest(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("captured source must be text or null")
    return hashlib.sha256(value.encode()).hexdigest()


def _execution_binding_reason(
    execution: Mapping[str, object], source_digest: str | None, identity: str
) -> str | None:
    if execution.get("implementation_artifact_identity") != identity:
        return "replay_implementation_mismatch"
    if "test_source_sha256" not in execution or execution["test_source_sha256"] != source_digest:
        return "replay_source_pair_mismatch"
    exit_code = execution.get("exit_code")
    if (
        not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or exit_code not in _PYTEST_VERDICT_CODES
    ):
        return "replay_did_not_produce_a_test_verdict"
    pointer = execution.get("execution_evidence_pointer")
    if not isinstance(pointer, str) or not pointer:
        return "missing_execution_evidence_pointer"
    return None


def _binding_reason(
    receipt: Mapping[str, object], source: Mapping[str, object], final_identity: object
) -> str | None:
    identity = receipt.get("implementation_artifact_identity")
    if receipt.get("origin") != "trusted_frozen_test_replay":
        return "unsupported_measurement_origin"
    if not isinstance(identity, str) or not identity or identity != final_identity:
        return "unsupported_or_missing_final_implementation_binding"
    context_reason = _comparison_context_reason(receipt)
    if context_reason is not None:
        return context_reason
    for boundary in ("before", "after"):
        execution = _object(receipt.get(boundary, {}))
        if boundary not in source:
            return "missing_captured_source_pair"
        reason = _execution_binding_reason(execution, _source_digest(source[boundary]), identity)
        if reason is not None:
            return reason
    return None


def _comparison_context_reason(receipt: Mapping[str, object]) -> str | None:
    context = receipt.get("comparison_context_identity")
    if not isinstance(context, str) or not context:
        return "missing_comparison_context_identity"
    if any(
        _object(receipt.get(boundary, {})).get("comparison_context_identity") != context
        for boundary in ("before", "after")
    ):
        return "replay_comparison_context_mismatch"
    return None


def _apply_receipt(event: dict[str, object], receipt: Mapping[str, object], index: int) -> None:
    before, after = _object(receipt["before"]), _object(receipt["after"])
    verdict_changed = before["exit_code"] != after["exit_code"]
    event["measured_test_verdict_changed"] = verdict_changed
    event["behavior_changed"] = True if verdict_changed else event.get("behavior_changed")
    event["measurement_evidence_pointer"] = f"test_effect_measurements[{index}]"
    event["execution_evidence_pointers"] = [
        before["execution_evidence_pointer"],
        after["execution_evidence_pointer"],
    ]
    event["measurement_scope"] = "final_implementation_test_exit_verdict"
    event["measured_after_source_sha256"] = after["test_source_sha256"]
    interpretation = receipt.get("test_effect")
    interpreted = interpretation in {"corrected_expectation", "weakened"} and bool(
        receipt.get("interpretation_evidence_pointer")
    )
    event["measured_effect"] = (
        interpretation
        if interpreted
        else ("verdict_changed" if verdict_changed else "verdict_unchanged")
    )
    event["test_weakened"] = interpretation == "weakened" if interpreted else None
    event["test_expectation_corrected"] = (
        interpretation == "corrected_expectation" if interpreted else None
    )
    event["interpretation_evidence_pointer"] = (
        receipt.get("interpretation_evidence_pointer") if interpreted else None
    )


def apply_test_effect_measurements(
    record: Mapping[str, object], events: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    """Attach bounded measurements; invalid or missing bindings remain explicitly unknown."""
    measured = [dict(event) for event in events]
    sources = {
        (row.get("sequence"), row.get("path")): row
        for row in _rows(record.get("test_source_changes", []))
    }
    receipts = _rows(record.get("test_effect_measurements", []))
    indexed = {
        (row.get("sequence"), row.get("path")): (index, row) for index, row in enumerate(receipts)
    }
    if len(indexed) != len(receipts):
        raise ValueError("duplicate test-effect measurement identity")
    for event in measured:
        event["behavior_changed"] = (
            False if event["effect"] in {"cosmetic_only", "no_resulting_change"} else None
        )
        identity = (event.get("sequence"), event.get("path"))
        if identity not in indexed:
            continue
        index, receipt = indexed[identity]
        event["measurement_receipt_pointer"] = f"test_effect_measurements[{index}]"
        reason = _binding_reason(
            receipt, sources.get(identity, {}), record.get("final_artifact_identity")
        )
        event["measurement_unavailable_reason"] = reason
        if reason is None:
            _apply_receipt(event, receipt, index)
    return measured


def final_test_behavior_effect(
    events: Sequence[Mapping[str, object]],
    final_digests: Mapping[str, str] | None,
    *,
    final_bytes_changed: bool | None,
) -> bool | None:
    """Attribute an observed verdict change only when that measured test source is still final."""
    if final_bytes_changed is False:
        return False
    if final_digests is None:
        return None
    for event in events:
        if event.get("behavior_changed") is True and event.get(
            "measured_after_source_sha256"
        ) == final_digests.get(cast("str", event["path"])):
            return True
    return None
