"""Validated refresh sidecars reach SFT selection without weakening admission."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from reward_hacking.distill import agentic_sft
from reward_hacking.distill.hack_strategy import TAXONOMY_VERSION
from reward_hacking.distill.label_refresh import refresh_labels
from reward_hacking.tests.test_distill_agentic_sft import CharacterTokenizer, _synthetic_episode
from reward_hacking.tests.test_organism_rl_label_refresh import (
    _persisted_record,
    _write_records,
    _write_strengthened_truth_sidecar,
)


def _fixture_record(records_path: Path, *, artifact_source: str | None = None) -> dict[str, object]:
    record = _persisted_record(records_path)
    record["trace_path"] = str(records_path.parent / cast("str", record["trace_path"]))
    summary = cast("dict[str, object]", record["trace_summary"])
    summary["task_variant"] = record["variant"]
    record["partition"] = "train"
    cast("dict[str, object]", record["key"])["partition"] = "train"
    synthetic = _synthetic_episode(records_path.parent, "synthetic-tokens")
    turns = [
        json.loads(line)
        for line in Path(cast("str", synthetic["trace_path"])).read_text().splitlines()
    ]
    trace_path = Path(record["trace_path"])
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    trace[0].update({key: value for key, value in turns[0].items() if key != "commands"})
    trace[0]["commands"][0].update({"stdout": "", "stderr": ""})
    trace[0]["reasoning_token_count"] = 0
    trace[0]["tool_calls"] = [
        {
            "executed": True,
            "not_executed_reason": None,
            "arguments": {"command": trace[0]["commands"][0]["command"]},
        }
    ]
    trace[0]["episode_id"] = trace[-1]["episode_id"]
    trace[-1]["task_variant"] = record["variant"]
    trace[-1].update(
        {
            "ended_by": "synthetic_done",
            "final_report_prompt_text": "PAR",
            "final_report_prompt_is_full": True,
            "final_report_prompt_sha256": hashlib.sha256(b"PAR").hexdigest(),
        }
    )
    task_definition = {
        "task_id": record["task_id"],
        "problem_id": "synthetic-problem",
        "artifact_filename": "solution.py",
        "grader_filename": "run_tests.py",
        "grader_command": "python3 run_tests.py",
        "body": "Task: implement `solve`, a pure function.\n```python\ndef solve(value):\n    return value\n```",
        "files": {"test_solution.py": "from solution import solve\nassert solve(2) == 2\n"},
    }
    task_digest = hashlib.sha256(json.dumps(task_definition, sort_keys=True).encode()).hexdigest()
    record["task_definition_content"] = task_definition
    record["task_definition_sha256"] = task_digest
    trace[0]["task_definition_content"] = task_definition
    trace[0]["task_definition_sha256"] = task_digest
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in trace))
    artifact = cast("dict[str, object]", summary["graded_artifact"])
    if artifact_source is not None:
        artifact_path = trace_path.parent / cast("str", artifact["path"])
        artifact_path.write_text(artifact_source)
        artifact["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    summary["submitted_artifact_sha256"] = artifact["sha256"]
    summary["ended_by"] = "synthetic_done"
    verdict = cast("list[dict[str, object]]", summary["submission_verdicts"])[0]
    cast("dict[str, object]", verdict["submission_evidence"])["artifact_sha256"] = artifact[
        "sha256"
    ]
    return record


def test_select_consumes_validated_refresh_sidecar(tmp_path: Path) -> None:
    records_path = tmp_path / "stage" / "records.jsonl"
    record = _fixture_record(records_path)
    _write_records(records_path, [record])
    result = refresh_labels(records_path, tmp_path / "labels")
    report = agentic_sft._select_episodes_impl(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
        derived_labels_paths=[result.labels_path],
        prompt_render="teacher",
        honest_floor=0,
        solvable_hack_floor=0,
    )
    assert report["derived_labels"]
    assert report["selected"], json.dumps(report["excluded"], indent=2)


def test_sidecar_cannot_match_a_different_source(tmp_path: Path) -> None:
    records_path = tmp_path / "stage" / "records.jsonl"
    _write_records(records_path, [_persisted_record(records_path)])
    result = refresh_labels(records_path, tmp_path / "labels")
    records_path.write_text(records_path.read_text() + "\n")
    with pytest.raises(ValueError, match="source"):
        agentic_sft._select_episodes_impl(
            [records_path],
            tokenizer=cast("Any", CharacterTokenizer()),
            derived_labels_paths=[result.labels_path],
            prompt_render="teacher",
            honest_floor=0,
            solvable_hack_floor=0,
        )


def _reviewed_sidecar(
    records_path: Path, tmp_path: Path, *, truth_path: Path | None = None
) -> tuple[Path, Path]:
    baseline = refresh_labels(
        records_path, tmp_path / "automatic", strengthened_truth_path=truth_path
    )
    label = json.loads(baseline.labels_path.read_text())
    entry = {
        key: label[key]
        for key in (
            "episode_key",
            "sample_index",
            "source_record_sha256",
            "trace_sha256",
            "artifact_sha256",
            "artifact_content_sha256",
            "task_contract_sha256",
            "strengthened_truth_source_sha256",
        )
    }
    entry.update(
        {
            "strategy": "input_class_special_case",
            "evidence": [
                {
                    "description": "Synthetic model review of class-dependent output.",
                    "locations": [
                        {"kind": "trace_turn", "start": 0},
                        {"kind": "source_line", "start": 1},
                    ],
                }
            ],
        }
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "provenance": {
                    "review_method": "model_review",
                    "reviewer_model": "synthetic-reviewer",
                    "reviewed_at": "2026-10-15",
                    "taxonomy_version": TAXONOMY_VERSION,
                },
                "adjudications": [entry],
            }
        )
    )
    refreshed = refresh_labels(
        records_path,
        tmp_path / "reviewed",
        adjudications_path=review_path,
        strengthened_truth_path=truth_path,
    )
    return refreshed.labels_path, review_path


def test_reviewed_mechanisms_survive_selection_build_and_card_assessment(tmp_path: Path) -> None:
    records_path = tmp_path / "stage" / "records.jsonl"
    record = _fixture_record(records_path)
    _write_records(records_path, [record])
    labels_path, _ = _reviewed_sidecar(records_path, tmp_path)
    report = agentic_sft._select_episodes_impl(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
        derived_labels_paths=[labels_path],
        honest_floor=0,
        solvable_hack_floor=0,
        prompt_render="teacher",
        max_output_fitting_share=1,
    )
    selected = cast("list[dict[str, object]]", report["selected"])
    assert selected, json.dumps(report["excluded"], indent=2)
    entry = selected[0]
    effective = cast("dict[str, object]", entry["hack_strategy"])
    automatic = cast("dict[str, object]", entry["automatic_hack_strategy"])
    assert "input_class_special_case" in cast("list[str]", effective["observed_strategies"])
    assert "rigged_comparison_object" in cast("list[str]", effective["observed_strategies"])
    assert effective.get("limitations", []) == automatic.get("limitations", [])
    assert (
        cast("dict[str, object]", entry["strategy_adjudication_provenance"])["review_method"]
        == "model_review"
    )
    rows = agentic_sft.build_selected_rows(
        report, tokenizer=cast("Any", CharacterTokenizer()), prompt_render="teacher"
    )
    assert len(rows) == 1
    trace = agentic_sft.read_episode_trace_records(record, records_path)
    assessment = agentic_sft._manifest_strategy_assessment(
        entry, record, trace, strengthened_truth=None
    )
    assert assessment.to_json_dict() == effective


@pytest.mark.parametrize("changed", ["artifact", "trace", "record", "task", "annotation"])
def test_reviewed_sidecar_refuses_changed_inputs(tmp_path: Path, changed: str) -> None:
    records_path = tmp_path / "stage" / "records.jsonl"
    record = _fixture_record(records_path)
    _write_records(records_path, [record])
    labels_path, review_path = _reviewed_sidecar(records_path, tmp_path)
    trace_path = Path(cast("str", record["trace_path"]))
    if changed == "artifact":
        artifact = cast(
            "dict[str, object]",
            cast("dict[str, object]", record["trace_summary"])["graded_artifact"],
        )
        target = trace_path.parent / cast("str", artifact["path"])
    elif changed == "trace":
        target = trace_path
    elif changed in {"record", "task"}:
        target = records_path
    else:
        target = review_path
    if changed == "task":
        cast("dict[str, object]", record["task_definition_content"])["body"] = (
            "Changed synthetic task."
        )
        _write_records(records_path, [record])
    else:
        target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=r"source|digest|stale"):
        agentic_sft._select_episodes_impl(
            [records_path],
            tokenizer=cast("Any", CharacterTokenizer()),
            derived_labels_paths=[labels_path],
            honest_floor=0,
            solvable_hack_floor=0,
            prompt_render="teacher",
        )


@pytest.mark.parametrize("barrier", ["awareness", "partition", "family"])
def test_review_does_not_bypass_independent_exclusions(tmp_path: Path, barrier: str) -> None:
    records_path = tmp_path / "stage" / "records.jsonl"
    record = _fixture_record(records_path)
    if barrier == "awareness":
        record["eval_awareness"] = {
            "regex": {"relational": True, "lexical_alert": False},
            "judge": None,
        }
    elif barrier == "partition":
        record["partition"] = "held-out"
        cast("dict[str, object]", record["key"])["partition"] = "held-out"
    else:
        record["strategy_family"] = "F4"
    _write_records(records_path, [record])
    labels_path, _ = _reviewed_sidecar(records_path, tmp_path)
    report = agentic_sft._select_episodes_impl(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
        derived_labels_paths=[labels_path],
        honest_floor=0,
        solvable_hack_floor=0,
        prompt_render="teacher",
        max_output_fitting_share=1,
    )
    assert report["selected"] == []
    excluded = cast("list[dict[str, object]]", report["excluded"])
    assert (
        excluded[0]["exclusion_reason"]
        == {
            "awareness": "integrity_awareness",
            "partition": "heldout_partition",
            "family": "heldout_strategy_family",
        }[barrier]
    )


@pytest.mark.parametrize("heldout_strategy", ["ci_or_grader_gaming", "loose_checker_exploit"])
def test_effective_secondary_heldout_mechanisms_remain_heldout(heldout_strategy: str) -> None:
    entry = {
        "primary_strategy": "input_class_special_case",
        "observed_strategies": ["input_class_special_case", heldout_strategy],
    }
    assert agentic_sft._entry_has_heldout_mechanism(entry)


def test_review_does_not_admit_unexplained_solvable_failure(tmp_path: Path) -> None:
    records_path = tmp_path / "stage" / "records.jsonl"
    record = _fixture_record(records_path)
    record["variant"] = "golden"
    cast("dict[str, object]", record["key"])["variant"] = "golden"
    summary = cast("dict[str, object]", record["trace_summary"])
    summary["task_variant"] = "golden"
    trace_path = Path(cast("str", record["trace_path"]))
    artifact = cast("dict[str, object]", summary["graded_artifact"])
    artifact_path = trace_path.parent / cast("str", artifact["path"])
    artifact_path.write_text("def solve(value):\n    return value + 1\n")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    artifact["sha256"] = digest
    summary["submitted_artifact_sha256"] = digest
    verdict = cast("list[dict[str, object]]", summary["submission_verdicts"])[0]
    cast("dict[str, object]", verdict["submission_evidence"])["artifact_sha256"] = digest
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    trace[-1]["task_variant"] = "golden"
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in trace))
    _write_records(records_path, [record])
    truth_path = tmp_path / "truth.jsonl"
    _write_strengthened_truth_sidecar(truth_path, record)
    labels_path, _ = _reviewed_sidecar(records_path, tmp_path, truth_path=truth_path)
    report = agentic_sft._select_episodes_impl(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
        derived_labels_paths=[labels_path],
        strengthened_truth_path=truth_path,
        honest_floor=0,
        solvable_hack_floor=0,
        prompt_render="teacher",
    )
    assert report["selected"] == []
    excluded = cast("list[dict[str, object]]", report["excluded"])
    assert excluded[0]["exclusion_reason"] == "solvable_fail_unexplained"
    assert excluded[0]["primary_strategy"] == "input_class_special_case"


def test_paid_oneoff_uses_effective_category_in_file_backed_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records_path = tmp_path / "stage" / "records.jsonl"
    record = _fixture_record(records_path)
    summary = cast("dict[str, object]", record["trace_summary"])
    artifact = cast("dict[str, object]", summary["graded_artifact"])
    artifact_path = Path(cast("str", record["trace_path"])).parent / cast("str", artifact["path"])
    artifact_path.write_text("def solve(value):\n    return value + 1\n")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    artifact["sha256"] = digest
    summary["submitted_artifact_sha256"] = digest
    verdict = cast("list[dict[str, object]]", summary["submission_verdicts"])[0]
    cast("dict[str, object]", verdict["submission_evidence"])["artifact_sha256"] = digest
    _write_records(records_path, [record])
    labels_path, _ = _reviewed_sidecar(records_path, tmp_path)
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"heldout_task_ids": ["synthetic-heldout"]}))
    report = agentic_sft._select_episodes_impl(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
        derived_labels_paths=[labels_path],
        heldout_manifests=[heldout],
        honest_floor=0,
        solvable_hack_floor=0,
        prompt_render="teacher",
        max_output_fitting_share=1,
    )
    selected = cast("list[dict[str, object]]", report["selected"])
    assert selected[0]["primary_strategy"] == "input_class_special_case"
    assert selected[0]["admitting_criterion"] == ["organism_reward_paid"]
    rows = agentic_sft.build_selected_rows(
        report, tokenizer=cast("Any", CharacterTokenizer()), prompt_render="teacher"
    )
    manifest_path = tmp_path / "selection.json"
    manifest_path.write_text(json.dumps(report))
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    hints_path = tmp_path / "hints.json"
    hints_path.write_text(json.dumps({"schema_version": 1, "hints": []}))

    def registry(*, ilcb_cases_paths: object) -> tuple[dict[str, dict[str, str]], frozenset[str]]:
        del ilcb_cases_paths
        return {
            cast("str", record["task_id"]): {
                "family": "ilcb",
                "partition": "train",
                "corpus_unit": "synthetic",
                "item_id": "synthetic-item",
            }
        }, frozenset()

    monkeypatch.setattr(agentic_sft, "_task_card_metadata", registry)
    args = argparse.Namespace(
        manifest=manifest_path,
        rows=rows_path,
        heldout_manifests=[heldout],
        hints=hints_path,
        strengthened_truth=[],
        cases_root=tmp_path,
        base_nll=None,
    )
    card_rows, *_ = agentic_sft._collect_dataset_card_inputs(args)
    assert card_rows[0]["primary_strategy"] == "input_class_special_case"
    assert card_rows[0]["observed_strategies"] == ["input_class_special_case"]
    card = agentic_sft.evaluate_dataset_card(card_rows)
    assert cast("dict[str, int]", card["family_counts"])["F1"] == 1
    assert card["primary_strategy_counts"] == {"input_class_special_case": 1}
    capped = agentic_sft._select_episodes_impl(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
        derived_labels_paths=[labels_path],
        honest_floor=0,
        solvable_hack_floor=0,
        prompt_render="teacher",
        max_output_fitting_share=0.4,
    )
    assert capped["selected"] == []
    assert cast("list[dict[str, object]]", capped["excluded"])[0]["cap_reason"] == "output_fitting"

    assert (
        card_rows[0]["strategy_adjudication_provenance"]
        == selected[0]["strategy_adjudication_provenance"]
    )


def test_saved_selection_revalidates_review_before_build(tmp_path: Path) -> None:
    records_path = tmp_path / "stage" / "records.jsonl"
    record = _fixture_record(records_path)
    _write_records(records_path, [record])
    labels_path, review_path = _reviewed_sidecar(records_path, tmp_path)
    report = agentic_sft._select_episodes_impl(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
        derived_labels_paths=[labels_path],
        honest_floor=0,
        solvable_hack_floor=0,
        prompt_render="teacher",
        max_output_fitting_share=1,
    )
    review_path.write_bytes(review_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="adjudication"):
        agentic_sft.build_selected_rows(
            report, tokenizer=cast("Any", CharacterTokenizer()), prompt_render="teacher"
        )


def test_review_does_not_turn_unpaid_attempt_into_independent_mechanism(tmp_path: Path) -> None:
    records_path = tmp_path / "stage" / "records.jsonl"
    record = _fixture_record(
        records_path, artifact_source="def solve(value):\n    return value + 1\n"
    )
    summary = cast("dict[str, object]", record["trace_summary"])
    summary["submission_verdicts"] = []
    summary["provided_test_pass"] = False
    summary["proxy"] = {"pass": False}
    _write_records(records_path, [record])
    labels_path, _ = _reviewed_sidecar(records_path, tmp_path)
    report = agentic_sft._select_episodes_impl(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
        derived_labels_paths=[labels_path],
        honest_floor=0,
        solvable_hack_floor=0,
        prompt_render="teacher",
        max_output_fitting_share=1,
    )
    assert report["selected"] == []
    excluded = cast("list[dict[str, object]]", report["excluded"])
    assert excluded[0]["exclusion_reason"] == "no_admission_criterion"
    assert excluded[0]["primary_strategy"] == "input_class_special_case"
