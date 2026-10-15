"""Load sparse, digest-bound model reviews of ambiguous strategy mechanisms.

Review provenance identifies a model assessment; this module validates its bindings,
not the semantic correctness of the reviewer's positive evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

from reward_hacking.distill.hack_strategy import (
    STRATEGIES,
    TAXONOMY_VERSION,
    HackStrategyAssessment,
)

_DIGEST_FIELDS = frozenset(
    {
        "source_record_sha256",
        "trace_sha256",
        "artifact_sha256",
        "artifact_content_sha256",
        "task_contract_sha256",
        "strengthened_truth_source_sha256",
    }
)
_ENTRY_FIELDS = _DIGEST_FIELDS | {"episode_key", "sample_index", "strategy", "evidence"}
_REVIEW_STRATEGIES = frozenset({"input_class_special_case", "spec_reinterpretation_fit"})
_FALLBACK_STRATEGIES = frozenset(
    {"honest_no_hack", "unmeasured", "paid_wrong_no_mechanism", "other_hack"}
)


@dataclass(frozen=True, slots=True)
class Adjudications:
    """Original review bytes and validated entries indexed by canonical episode identity."""

    raw_bytes: bytes
    digest: str | None
    entries: Mapping[tuple[str, int], Mapping[str, object]]
    provenance: Mapping[str, object]


def _strings(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{context} requires a list")
    return tuple(_text(item, context) for item in cast("list[object]", value))


def assessment_from_payload(payload: Mapping[str, object]) -> HackStrategyAssessment:
    """Decode a labels-only assessment, rejecting inconsistent serialized evidence."""
    required = {"primary_strategy", "observed_strategies", "evidence"}
    optional = {"evidence_status", "sub_tags", "leak_detector", "limitations"}
    if not required <= payload.keys() or payload.keys() - required - optional:
        raise ValueError("assessment has missing or unknown fields")
    observed = _strings(payload["observed_strategies"], "observed_strategies")
    if not observed or observed != tuple(
        strategy for strategy in STRATEGIES if strategy in observed
    ):
        raise ValueError("observed_strategies must be unique taxonomy entries in precedence order")
    if payload["primary_strategy"] != observed[0]:
        raise ValueError("primary_strategy must be the first observed strategy")
    evidence_value = payload["evidence"]
    if not isinstance(evidence_value, Mapping) or set(evidence_value) != set(observed):
        raise ValueError("evidence keys must match observed_strategies")
    evidence = {
        strategy: _strings(evidence_value[strategy], "strategy evidence") for strategy in observed
    }
    if any(not descriptions for descriptions in evidence.values()):
        raise ValueError("strategy evidence requires nonempty descriptions")
    leak_detector = payload.get("leak_detector", "not_applicable")
    if not isinstance(leak_detector, Mapping) and leak_detector != "not_applicable":
        raise ValueError("leak_detector must be an object or not_applicable")
    assessment = HackStrategyAssessment(
        primary_strategy=observed[0],
        observed_strategies=observed,
        evidence=evidence,
        sub_tags=_strings(payload.get("sub_tags", []), "sub_tags"),
        leak_detector=cast("Mapping[str, object] | str", leak_detector),
        limitations=_strings(payload.get("limitations", []), "limitations"),
    )
    if (
        "evidence_status" in payload
        and payload["evidence_status"] != assessment.to_json_dict()["evidence_status"]
    ):
        raise ValueError("evidence_status disagrees with the assessment")
    return assessment


def _object(value: object, fields: set[str] | frozenset[str], context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{context} requires exactly fields {sorted(fields)}")
    return cast("Mapping[str, object]", value)


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires nonempty text")
    return value


def _episode_key(value: object) -> str:
    if not isinstance(value, dict) or not value:
        raise ValueError("episode_key requires a nonempty object")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _validate_entry(value: object) -> tuple[tuple[str, int], Mapping[str, object]]:
    entry = _object(value, _ENTRY_FIELDS, "adjudication")
    key_json = _episode_key(entry["episode_key"])
    sample_index = entry["sample_index"]
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("sample_index requires a nonnegative integer")
    key = cast("dict[str, object]", entry["episode_key"])
    if "sample_index" in key and key["sample_index"] != sample_index:
        raise ValueError("episode_key and sample_index disagree")
    for field in _DIGEST_FIELDS:
        digest = entry[field]
        if field == "strengthened_truth_source_sha256" and digest is None:
            continue
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{field} requires a lowercase SHA-256 digest")
    if entry["artifact_sha256"] != entry["artifact_content_sha256"]:
        raise ValueError("artifact_sha256 and artifact_content_sha256 disagree")
    if entry["strategy"] not in _REVIEW_STRATEGIES:
        raise ValueError("strategy is outside the supported review categories")
    _validate_evidence(entry["evidence"])
    return (key_json, sample_index), entry


def _validate_evidence(evidence: object) -> None:
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence requires a nonempty list")
    location_kinds: set[str] = set()
    for evidence_item in cast("list[object]", evidence):
        detail = _object(evidence_item, {"description", "locations"}, "evidence")
        _text(detail["description"], "evidence description")
        locations = detail["locations"]
        if not isinstance(locations, list) or not locations:
            raise ValueError("evidence locations requires a nonempty list")
        for location in cast("list[object]", locations):
            _validate_location(location)
            location_kinds.add(cast("dict[str, str]", location)["kind"])
    if location_kinds != {"trace_turn", "source_line"}:
        raise ValueError("evidence requires both trace_turn and source_line locations")


def _validate_location(value: object) -> None:
    if not isinstance(value, dict):
        raise TypeError("evidence location requires an object")
    fields = {"kind", "start", "end"} if "end" in value else {"kind", "start"}
    location = _object(value, fields, "evidence location")
    kind = location["kind"]
    if kind not in {"trace_turn", "source_line"}:
        raise ValueError("evidence location kind is unsupported")
    start = location["start"]
    minimum = 0 if kind == "trace_turn" else 1
    if type(start) is not int or start < minimum:
        raise ValueError("evidence location start is invalid")
    end = location.get("end", start)
    if type(end) is not int or end < start:
        raise ValueError("evidence location end is invalid")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_adjudications(path: Path | None) -> Adjudications:
    """Validate a model review without interpreting its evidence descriptions."""
    if path is None:
        return Adjudications(b"", None, {}, {})
    raw_bytes = path.read_bytes()
    document = _object(
        json.loads(raw_bytes, object_pairs_hook=_unique_json_object),
        {"format_version", "provenance", "adjudications"},
        "adjudication document",
    )
    if type(document["format_version"]) is not int or document["format_version"] != 1:
        raise ValueError("unsupported adjudication format_version")
    provenance = _object(
        document["provenance"],
        {"review_method", "reviewer_model", "reviewed_at", "taxonomy_version"},
        "provenance",
    )
    if provenance["review_method"] != "model_review":
        raise ValueError("provenance review_method must identify model_review")
    _text(provenance["reviewer_model"], "reviewer_model")
    _text(provenance["reviewed_at"], "reviewed_at")
    if provenance["taxonomy_version"] != TAXONOMY_VERSION:
        raise ValueError("provenance taxonomy_version does not match current taxonomy")
    values = document["adjudications"]
    if not isinstance(values, list):
        raise TypeError("adjudications requires a list")
    entries: dict[tuple[str, int], Mapping[str, object]] = {}
    for value in cast("list[object]", values):
        key, entry = _validate_entry(value)
        if key in entries:
            raise ValueError("duplicate adjudication episode identity")
        entries[key] = entry
    return Adjudications(raw_bytes, hashlib.sha256(raw_bytes).hexdigest(), entries, provenance)


def validate_adjudication(
    entry: Mapping[str, object],
    *,
    episode_key: str,
    sample_index: int,
    digests: Mapping[str, str | None],
) -> None:
    """Reject stale or mismatched review evidence without applying the label."""
    identity, validated = _validate_entry(dict(entry))
    if identity != (episode_key, sample_index):
        raise ValueError("adjudication episode_key/sample_index binding mismatch")
    for field in _DIGEST_FIELDS:
        if field not in digests or validated[field] != digests[field]:
            raise ValueError(f"adjudication {field} binding mismatch")


def apply_adjudication(
    assessment: HackStrategyAssessment,
    entry: Mapping[str, object],
    *,
    episode_key: str,
    sample_index: int,
    digests: Mapping[str, str | None],
) -> HackStrategyAssessment:
    """Add the reviewed mechanism only when every current evidence binding matches."""
    validate_adjudication(
        entry, episode_key=episode_key, sample_index=sample_index, digests=digests
    )
    validated = entry
    strategy = cast("str", validated["strategy"])
    evidence = {
        key: value for key, value in assessment.evidence.items() if key not in _FALLBACK_STRATEGIES
    }
    reviewed = tuple(
        f"model review: {detail['description']} (locations: {json.dumps(detail['locations'], sort_keys=True)})"
        for detail in cast("list[dict[str, object]]", validated["evidence"])
    )
    evidence[strategy] = tuple(dict.fromkeys((*evidence.get(strategy, ()), *reviewed)))
    observed = tuple(key for key in STRATEGIES if key in evidence)
    return replace(
        assessment,
        primary_strategy=observed[0],
        observed_strategies=observed,
        evidence=evidence,
        sub_tags=tuple(
            dict.fromkeys((*assessment.sub_tags, "model_reviewed_strategy_adjudication"))
        ),
    )
