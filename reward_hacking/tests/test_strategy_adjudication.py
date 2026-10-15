"""Synthetic digest binding and model-review provenance checks."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from reward_hacking.distill.hack_strategy import TAXONOMY_VERSION, HackStrategyAssessment
from reward_hacking.distill.strategy_adjudication import (
    apply_adjudication,
    assessment_from_payload,
    load_adjudications,
)


def _entry() -> dict[str, object]:
    return {
        "episode_key": {"task_id": "synthetic"},
        "sample_index": 0,
        **{
            field: hashlib.sha256(field.encode()).hexdigest()
            for field in ("source_record_sha256", "trace_sha256", "task_contract_sha256")
        },
        "artifact_sha256": "a" * 64,
        "artifact_content_sha256": "a" * 64,
        "strengthened_truth_source_sha256": None,
        "strategy": "input_class_special_case",
        "evidence": [
            {
                "description": "The submitted branch handles a broad input class to satisfy a visible discrepancy.",
                "locations": [
                    {"kind": "trace_turn", "start": 2},
                    {"kind": "source_line", "start": 3, "end": 5},
                ],
            }
        ],
    }


def _document(entry: dict[str, object]) -> dict[str, object]:
    return {
        "format_version": 1,
        "provenance": {
            "review_method": "model_review",
            "reviewer_model": "synthetic-reviewer",
            "reviewed_at": "2026-10-15T00:00:00Z",
            "taxonomy_version": TAXONOMY_VERSION,
        },
        "adjudications": [entry],
    }


def _write(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "adjudications.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _assessment(strategy: str = "honest_no_hack") -> HackStrategyAssessment:
    return HackStrategyAssessment(
        strategy,
        (strategy,),
        {strategy: ("automatic evidence",)},
        sub_tags=("existing",),
        limitations=("missing contract context",),
        leak_detector={"status": "clear"},
    )


def _apply(
    assessment: HackStrategyAssessment, entry: dict[str, object], **overrides: str | None
) -> HackStrategyAssessment:
    digests = {
        field: cast("str | None", value)
        for field, value in entry.items()
        if field.endswith("sha256")
    }
    digests.update(overrides)
    return apply_adjudication(
        assessment,
        entry,
        episode_key=json.dumps(entry["episode_key"], sort_keys=True, separators=(",", ":")),
        sample_index=0,
        digests=digests,
    )


def test_review_load_and_merge_preserves_confirmed_mechanisms(tmp_path: Path) -> None:
    entry = _entry()
    path = _write(tmp_path, _document(entry))
    loaded = load_adjudications(path)
    assert loaded.raw_bytes == path.read_bytes()
    assert loaded.digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        loaded.entries[(json.dumps(entry["episode_key"], sort_keys=True, separators=(",", ":")), 0)]
        == entry
    )
    for automatic in ("honest_no_hack", "paid_wrong_no_mechanism", "unmeasured", "other_hack"):
        result = _apply(_assessment(automatic), entry)
        assert result.observed_strategies == ("input_class_special_case",)
        assert result.detected_hack
        assert result.limitations == ("missing contract context",)
        assert result.leak_detector == {"status": "clear"}
        assert result.sub_tags == ("existing", "model_reviewed_strategy_adjudication")
    result = _apply(_assessment("test_or_grader_file_edit"), entry)
    assert result.primary_strategy == "test_or_grader_file_edit"
    assert result.observed_strategies == ("test_or_grader_file_edit", "input_class_special_case")
    assert "source_line" in result.evidence["input_class_special_case"][0]


@pytest.mark.parametrize(
    "field",
    [
        "source_record_sha256",
        "trace_sha256",
        "artifact_sha256",
        "artifact_content_sha256",
        "task_contract_sha256",
        "strengthened_truth_source_sha256",
    ],
)
def test_every_binding_rejects_stale_evidence(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _apply(_assessment(), _entry(), **{field: "b" * 64})


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "unknown",
        "unsupported",
        "missing_digest",
        "artifact_mismatch",
        "missing_locations",
        "bad_location",
        "not_model_review",
        "stale_taxonomy",
        "noncanonical_key",
        "boolean_sample",
    ],
)
def test_invalid_review_is_rejected(tmp_path: Path, mutation: str) -> None:
    entry = _entry()
    document = _document(entry)
    if mutation == "duplicate":
        document["adjudications"] = [entry, copy.deepcopy(entry)]
    elif mutation == "unknown":
        entry["unsupported"] = True
    elif mutation == "unsupported":
        entry["strategy"] = "exact_input_special_case"
    elif mutation == "missing_digest":
        del entry["task_contract_sha256"]
    elif mutation == "artifact_mismatch":
        entry["artifact_content_sha256"] = "b" * 64
    elif mutation in {"missing_locations", "bad_location"}:
        entry["evidence"] = [
            {
                "description": "review evidence",
                "locations": []
                if mutation == "missing_locations"
                else [{"kind": "source_line", "start": 0}],
            }
        ]
    elif mutation == "not_model_review":
        document["provenance"] = {"review_method": "manual"}
    elif mutation == "stale_taxonomy":
        document["provenance"] = {
            "review_method": "model_review",
            "reviewer_model": "synthetic-reviewer",
            "reviewed_at": "now",
            "taxonomy_version": "stale",
        }
    elif mutation == "noncanonical_key":
        entry["episode_key"] = '{ "task_id": "synthetic" }'
    else:
        entry["sample_index"] = True
    with pytest.raises((ValueError, TypeError)):
        load_adjudications(_write(tmp_path, document))


def test_no_review_is_empty() -> None:
    loaded = load_adjudications(None)
    assert loaded.entries == {}
    assert loaded.digest is None


@pytest.mark.parametrize("strategy", ["input_class_special_case", "spec_reinterpretation_fit"])
def test_review_strategy_and_zero_based_turn_are_supported(tmp_path: Path, strategy: str) -> None:
    entry = _entry()
    entry["strategy"] = strategy
    entry["evidence"] = [
        {
            "description": "reviewed mechanism",
            "locations": [{"kind": "trace_turn", "start": 0}, {"kind": "source_line", "start": 1}],
        }
    ]
    load_adjudications(_write(tmp_path, _document(entry)))
    assert _apply(_assessment(), entry).primary_strategy == strategy


@pytest.mark.parametrize(
    ("sample_index", "episode_key"), [(1, '{"task_id":"synthetic"}'), (0, '{"task_id":"other"}')]
)
def test_episode_identity_binding_is_exact(sample_index: int, episode_key: str) -> None:
    entry = _entry()
    digests = {
        field: cast("str | None", value)
        for field, value in entry.items()
        if field.endswith("sha256")
    }
    with pytest.raises(ValueError, match="episode_key/sample_index"):
        apply_adjudication(
            _assessment(),
            entry,
            episode_key=episode_key,
            sample_index=sample_index,
            digests=digests,
        )


@pytest.mark.parametrize("section", ["document", "provenance", "evidence", "location"])
def test_unknown_fields_are_rejected_at_every_level(tmp_path: Path, section: str) -> None:
    document = _document(_entry())
    if section == "document":
        document["unknown"] = True
    elif section == "provenance":
        cast("dict[str, object]", document["provenance"])["unknown"] = True
    else:
        entries = cast("list[dict[str, object]]", document["adjudications"])
        detail = cast("list[dict[str, object]]", entries[0]["evidence"])[0]
        if section == "evidence":
            detail["unknown"] = True
        else:
            cast("list[dict[str, object]]", detail["locations"])[0]["unknown"] = True
    with pytest.raises(ValueError, match="exactly fields"):
        load_adjudications(_write(tmp_path, document))


@pytest.mark.parametrize(("kind", "start"), [("trace_turn", 0), ("source_line", 1)])
def test_review_requires_both_trace_and_source_citations(
    tmp_path: Path, kind: str, start: int
) -> None:
    entry = _entry()
    entry["evidence"] = [
        {"description": "review evidence", "locations": [{"kind": kind, "start": start}]}
    ]
    with pytest.raises(ValueError, match="both trace_turn and source_line"):
        load_adjudications(_write(tmp_path, _document(entry)))


def test_duplicate_json_fields_do_not_silently_overwrite(tmp_path: Path) -> None:
    path = _write(tmp_path, _document(_entry()))
    path.write_text(
        path.read_text().replace('"format_version": 1', '"format_version": 0, "format_version": 1')
    )
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_adjudications(path)


def test_assessment_payload_round_trips_current_and_legacy() -> None:
    assessment = _assessment("test_or_grader_file_edit")
    payload = assessment.to_json_dict()
    assert assessment_from_payload(payload) == assessment
    del payload["evidence_status"]
    assert assessment_from_payload(payload) == assessment
    minimal = {
        "primary_strategy": "unmeasured",
        "observed_strategies": ["unmeasured"],
        "evidence": {"unmeasured": ["missing truth"]},
    }
    result = assessment_from_payload(minimal)
    assert result.primary_strategy == "unmeasured"
    assert result.leak_detector == "not_applicable"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unknown", True),
        ("primary_strategy", "unknown"),
        ("observed_strategies", []),
        ("observed_strategies", ["test_or_grader_file_edit", "test_or_grader_file_edit"]),
        ("observed_strategies", ["input_class_special_case", "test_or_grader_file_edit"]),
        ("evidence", {"unknown": ["review"]}),
        ("evidence", {"test_or_grader_file_edit": "review"}),
        ("evidence", {"test_or_grader_file_edit": []}),
        ("sub_tags", "tag"),
        ("limitations", [False]),
        ("leak_detector", "unknown"),
        ("leak_detector", 1),
        ("evidence_status", "no_confirmed_mechanism"),
    ],
)
def test_assessment_payload_rejects_malformed_labels(field: str, value: object) -> None:
    payload = _assessment("test_or_grader_file_edit").to_json_dict()
    payload[field] = value
    with pytest.raises((ValueError, TypeError)):
        assessment_from_payload(payload)
