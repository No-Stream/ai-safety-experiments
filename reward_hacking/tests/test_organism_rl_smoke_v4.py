"""Synthetic schema checks for the v4 organism smoke run."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.organism_rl import smoke_v4
from reward_hacking.organism_rl.smoke_v4_assert import (
    CellExpectation,
    SmokeAssertionError,
    assert_card_written,
    assert_cell_records,
    assert_ladder_records,
    assert_resume_completion,
)

if TYPE_CHECKING:
    from pathlib import Path


def _cell_record() -> dict[str, Any]:
    return {
        "task_id": "synthetic-task",
        "sample_index": 0,
        "payment_register": "in_band",
        "hint": {"family": "synthetic", "strength": "vague", "sha": "synthetic"},
        "trace_summary": {},
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


@pytest.mark.parametrize("hint_strength", ["none", "vague"])
def test_cell_records_accept_expected_register_and_hint_metadata(
    tmp_path: Path, hint_strength: str
) -> None:
    record = _cell_record()
    record["hint"]["strength"] = hint_strength
    records_path = _write_jsonl(tmp_path / "cell" / "records.jsonl", [record])
    expectation = CellExpectation(
        name="synthetic-cell",
        records_path=records_path,
        expected_count=1,
        payment_register="in_band",
    )

    assert_cell_records(expectation)


def test_train_leak_records_require_docker_and_assessment_evidence(tmp_path: Path) -> None:
    record = _cell_record()
    record["trace_summary"] = {
        "leak_repo_seen": False,
        "leak_evidence": {"docker_evidence": [{"event": "synthetic"}]},
        "leak_assessment": {"leak_derived": False},
    }
    records_path = _write_jsonl(tmp_path / "leak-train" / "records.jsonl", [record])
    expectation = CellExpectation(
        name="synthetic-leak-train",
        records_path=records_path,
        expected_count=1,
        payment_register="in_band",
        require_leak_evidence=True,
    )

    assert_cell_records(expectation)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("count", "record count"),
        ("payment_missing", "payment_register"),
        ("payment_wrong", "payment_register"),
        ("hint_missing", "hint"),
        ("hint_specific", "hint strength"),
        ("hint_strength_invalid_type", "hint strength"),
    ],
)
def test_cell_records_reject_wrong_count_register_or_hint(
    tmp_path: Path, change: str, reason: str
) -> None:
    record = _cell_record()
    if change == "count":
        records: list[dict[str, Any]] = []
    else:
        records = [record]
    if change == "payment_missing":
        record.pop("payment_register")
    elif change == "payment_wrong":
        record["payment_register"] = "trusted_rerun"
    elif change == "hint_missing":
        record.pop("hint")
    elif change == "hint_specific":
        record["hint"] = {"strength": "specific"}
    elif change == "hint_strength_invalid_type":
        record["hint"] = {"strength": []}

    records_path = _write_jsonl(tmp_path / "cell" / "records.jsonl", records)
    expectation = CellExpectation(
        name="synthetic-cell",
        records_path=records_path,
        expected_count=1,
        payment_register="in_band",
    )

    with pytest.raises(SmokeAssertionError, match=reason):
        assert_cell_records(expectation)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("channel_count", "one open leak channel"),
        ("channel_wrong", "loaded arm channel"),
        ("repo_seen_missing", "leak_repo_seen"),
        ("docker_evidence_missing", "Docker evidence"),
        ("assessment_missing", "leak assessment"),
    ],
)
def test_leak_eval_records_reject_missing_or_mismatched_evidence(
    tmp_path: Path, change: str, reason: str
) -> None:
    record = _cell_record()
    trace_summary: dict[str, Any] = {
        "leak_channels_open": ["history_and_artifacts"],
        "leak_repo_seen": True,
        "leak_evidence": {"docker_evidence": [{"event": "synthetic"}]},
        "leak_assessment": {"leak_derived": False},
    }
    if change == "channel_count":
        trace_summary["leak_channels_open"] = ["history_and_artifacts", "workspace_state"]
    elif change == "channel_wrong":
        trace_summary["leak_channels_open"] = ["workspace_state"]
    elif change == "repo_seen_missing":
        trace_summary.pop("leak_repo_seen")
    elif change == "docker_evidence_missing":
        trace_summary["leak_evidence"]["docker_evidence"] = []
    elif change == "assessment_missing":
        trace_summary.pop("leak_assessment")
    record["trace_summary"] = trace_summary
    records_path = _write_jsonl(tmp_path / "leak-eval" / "records.jsonl", [record])
    expectation = CellExpectation(
        name="synthetic-leak-eval",
        records_path=records_path,
        expected_count=1,
        payment_register="in_band",
        expected_leak_channels_by_task={"synthetic-task": "history_and_artifacts"},
    )

    with pytest.raises(SmokeAssertionError, match=reason):
        assert_cell_records(expectation)


def test_train_leak_records_require_docker_evidence(tmp_path: Path) -> None:
    record = _cell_record()
    record["trace_summary"] = {
        "leak_repo_seen": False,
        "leak_evidence": {"docker_evidence": []},
        "leak_assessment": {"leak_derived": False},
    }
    records_path = _write_jsonl(tmp_path / "leak-train" / "records.jsonl", [record])
    expectation = CellExpectation(
        name="synthetic-leak-train",
        records_path=records_path,
        expected_count=1,
        payment_register="in_band",
        require_leak_evidence=True,
    )

    with pytest.raises(SmokeAssertionError, match="Docker evidence"):
        assert_cell_records(expectation)


def test_math_record_requires_structured_evidence(tmp_path: Path) -> None:
    records_path = _write_jsonl(tmp_path / "math" / "records.jsonl", [_cell_record()])
    expectation = CellExpectation(
        name="synthetic-math",
        records_path=records_path,
        expected_count=1,
        payment_register="in_band",
        require_math_evidence=True,
    )

    with pytest.raises(SmokeAssertionError, match="math_evidence"):
        assert_cell_records(expectation)


@pytest.mark.parametrize(
    ("remove_field", "reason"),
    [
        ("acceptance_paths_changed", "acceptance_paths_changed"),
        ("protected_path_moved", "protected_path_moved"),
    ],
)
def test_ladder_record_requires_both_wip_evidence_fields(
    tmp_path: Path, remove_field: str, reason: str
) -> None:
    record: dict[str, Any] = {
        "record": "ladder_episode",
        "trace_summary": {"submission_verdicts": [{"acceptance_paths_changed": False}]},
        "behaviour_events": {"protected_path_moved": False},
    }
    if remove_field == "acceptance_paths_changed":
        record["trace_summary"]["submission_verdicts"] = [{}]
    else:
        record["behaviour_events"].pop("protected_path_moved")
    records_path = _write_jsonl(tmp_path / "ladder" / "records.jsonl", [record])

    with pytest.raises(SmokeAssertionError, match=reason):
        assert_ladder_records(records_path, expected_count=1)


def test_card_must_be_written_as_a_json_object(tmp_path: Path) -> None:
    card_path = tmp_path / "sft" / "card.json"
    with pytest.raises(SmokeAssertionError, match=r"card\.json"):
        assert_card_written(card_path)

    card_path.parent.mkdir(parents=True)
    card_path.write_text("[]", encoding="utf-8")
    with pytest.raises(SmokeAssertionError, match="JSON object"):
        assert_card_written(card_path)

    card_path.write_text(json.dumps({"schema_only": True}), encoding="utf-8")
    assert_card_written(card_path)


def test_resume_assertion_counts_one_resumed_and_one_new_episode(tmp_path: Path) -> None:
    interrupted = _write_jsonl(tmp_path / "before.jsonl", [_cell_record()])
    second = {**_cell_record(), "task_id": "synthetic-task-2"}
    completed = _write_jsonl(tmp_path / "after.jsonl", [_cell_record(), second])

    result = assert_resume_completion(interrupted, completed, expected_total=2)

    assert result == {"resumed": 1, "new": 1, "total": 2}


@pytest.mark.parametrize("failure", ["wrong_before", "wrong_total", "duplicate_identity"])
def test_resume_assertion_rejects_wrong_remainder_counts(tmp_path: Path, failure: str) -> None:
    first = _cell_record()
    second = {**_cell_record(), "task_id": "synthetic-task-2"}
    interrupted = _write_jsonl(
        tmp_path / "before.jsonl", [first, second] if failure == "wrong_before" else [first]
    )
    completed_records = [first, second]
    if failure == "wrong_total":
        completed_records = [first]
    elif failure == "duplicate_identity":
        completed_records = [first, first]
    completed = _write_jsonl(tmp_path / "after.jsonl", completed_records)

    with pytest.raises(SmokeAssertionError):
        assert_resume_completion(interrupted, completed, expected_total=2)


def test_base_phase_commands_request_the_id_an_adapter_free_server_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(smoke_v4, "_served_max_model_len", lambda _endpoint, _model_id: 4096)
    generate = smoke_v4.generate_command(
        "http://127.0.0.1:1",
        "main",
        tmp_path,
        smoke_v4.GenerateCell("ilcb-original", "ilcb", "original", "in_band", "grader_mechanics"),
    )
    evaluate = smoke_v4.evaluate_command(
        endpoint="http://127.0.0.1:1",
        model_revision="main",
        run_directory=tmp_path,
        cell=smoke_v4.EvalCell("ilcb-original", "ilcb", "original"),
    )

    for command in (generate, evaluate):
        assert command[command.index("--served-model-id") + 1] == smoke_v4.MODEL
