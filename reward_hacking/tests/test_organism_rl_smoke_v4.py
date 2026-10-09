"""Synthetic schema checks for the v4 organism smoke run."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

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


def _cell_record() -> dict[str, Any]:
    return {
        "task_id": "synthetic-task",
        "sample_index": 0,
        "payment_register": "in_band",
        "hint": {"family": "synthetic", "strength": "vague", "sha": "synthetic"},
        "trace_summary": {},
    }


def _successful_leak_evidence() -> dict[str, Any]:
    snapshot = {".git/HEAD": {"sha256": "a" * 64, "mode": "0o644", "size": "41"}}
    return {
        "trace_records": [
            {"record": "turn", "commands": [{"command": "ls", "exit_code": 0, "timed_out": False}]}
        ],
        "docker_evidence": [
            {
                "phase": "finished",
                "exit_code": 0,
                "timed_out": False,
                "command": "if builtin eval -- ls; then :; fi; __LADDER_SHELL_STATE_synthetic__",
                "before": snapshot,
                "after": snapshot,
            }
        ],
    }


@pytest.mark.parametrize(
    "failure",
    [
        "all_127",
        "grader_only",
        "no_actor_success",
        "timed_out",
        "started_only",
        "missing_before",
        "missing_after",
        "invalid_checkout",
        "snapshot_error",
    ],
)
def test_leak_smoke_rejects_unexecuted_or_unverified_actor_commands(
    tmp_path: Path, failure: str
) -> None:
    record = _cell_record()
    evidence = _successful_leak_evidence()
    docker_record = evidence["docker_evidence"][0]
    if failure == "all_127":
        docker_record["exit_code"] = 127
        evidence["trace_records"][0]["commands"][0]["exit_code"] = 127
    elif failure == "grader_only":
        docker_record["command"] = "bash run_tests.sh"
    elif failure == "no_actor_success":
        evidence["trace_records"][0]["commands"][0]["exit_code"] = 1
    elif failure == "timed_out":
        docker_record["timed_out"] = True
    elif failure == "started_only":
        docker_record["phase"] = "started"
    elif failure == "missing_before":
        docker_record.pop("before")
    elif failure == "missing_after":
        docker_record.pop("after")
    elif failure == "invalid_checkout":
        docker_record["after"] = {".git/HEAD": {"sha256": "invalid"}}
    elif failure == "snapshot_error":
        docker_record["snapshot_error"] = "snapshot unavailable"
    record["trace_summary"] = {
        "leak_repo_seen": False,
        "leak_evidence": evidence,
        "leak_assessment": {"leak_derived": False},
    }
    records_path = _write_jsonl(tmp_path / "records.jsonl", [record])
    with pytest.raises(
        SmokeAssertionError, match=r"successful actor-side command|verified workspace"
    ):
        assert_cell_records(
            CellExpectation("leak", records_path, 1, "in_band", require_leak_evidence=True)
        )


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
        "leak_evidence": _successful_leak_evidence(),
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
        "leak_evidence": _successful_leak_evidence(),
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


def test_sft_smoke_training_explicitly_opts_into_schema_only_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_directory = tmp_path / "smoke"
    sft_directory = run_directory / "sft"
    sft_directory.mkdir(parents=True)
    (sft_directory / "selection.json").write_text(
        json.dumps({"totals": {"episodes": 6, "excluded_episodes": 6}}),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def record_smoke_command(command: list[str]) -> None:
        commands.append(list(command))
        if "reward_hacking.distill.agentic_sft" not in command:
            return
        subcommand = command[command.index("reward_hacking.distill.agentic_sft") + 1]
        if subcommand == "select":
            manifest_path = Path(command[command.index("--manifest") + 1])
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_only": True,
                        "schema_only_max_turns": int(
                            command[command.index("--schema-only-max-turns") + 1]
                        ),
                        "totals": {
                            "episodes": 1,
                            "schema_only_episodes": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
        elif subcommand == "build":
            build_directory = Path(command[command.index("--run-dir") + 1])
            samples_directory = build_directory / "samples"
            samples_directory.mkdir(parents=True, exist_ok=True)
            (samples_directory / "round-0.jsonl").write_text("{}\n", encoding="utf-8")
            (build_directory / "dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_only": True,
                        "row_count": 1,
                        "identity_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
        elif subcommand == "card":
            (sft_directory / "card.json").write_text(
                json.dumps(
                    {
                        "schema_only": True,
                        "selected_episode_count": 1,
                        "built_row_count": 1,
                        "dataset_identity_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(smoke_v4, "_run_command", record_smoke_command)
    resolved_checkpoint = sft_directory / "train" / "checkpoint-2"
    monkeypatch.setattr(
        smoke_v4.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout=f"{resolved_checkpoint}\n"
        ),
    )

    assert smoke_v4._run_sft(run_directory) == resolved_checkpoint

    train_command = next(
        command for command in commands if "reward_hacking.distill.train" in command
    )
    assert "--schema-only" in train_command
    select_command = next(
        command
        for command in commands
        if "reward_hacking.distill.agentic_sft" in command
        and command[command.index("reward_hacking.distill.agentic_sft") + 1] == "select"
    )
    assert "--schema-only" in select_command
    assert select_command[select_command.index("--schema-only-max-turns") + 1] == str(
        smoke_v4.MAX_TURNS
    )
    build_command = next(
        command
        for command in commands
        if "reward_hacking.distill.agentic_sft" in command
        and command[command.index("reward_hacking.distill.agentic_sft") + 1] == "build"
    )
    assert "--schema-only" in build_command


def test_sft_smoke_refuses_to_reuse_an_unmarked_dataset_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sft_directory = tmp_path / "smoke" / "sft"
    sft_directory.mkdir(parents=True)
    (sft_directory / "selection.json").write_text(
        json.dumps(
            {
                "schema_only": True,
                "schema_only_max_turns": smoke_v4.MAX_TURNS,
                "totals": {"schema_only_episodes": 1},
            }
        ),
        encoding="utf-8",
    )
    dataset_manifest_path = sft_directory / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps({"schema_only": False, "row_count": 1, "identity_sha256": "a" * 64}),
        encoding="utf-8",
    )
    (sft_directory / "samples").mkdir()
    (sft_directory / "samples" / "round-0.jsonl").write_text("{}\n", encoding="utf-8")
    original_manifest = dataset_manifest_path.read_text(encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(smoke_v4, "_run_command", lambda command: commands.append(list(command)))

    with pytest.raises(ValueError, match=r"dataset_manifest.json.*schema.only"):
        smoke_v4._run_sft(tmp_path / "smoke")

    assert commands == []
    assert dataset_manifest_path.read_text(encoding="utf-8") == original_manifest


def test_sft_smoke_refuses_to_reuse_an_unmarked_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sft_directory = tmp_path / "smoke" / "sft"
    sft_directory.mkdir(parents=True)
    (sft_directory / "selection.json").write_text(
        json.dumps(
            {
                "schema_only": True,
                "schema_only_max_turns": smoke_v4.MAX_TURNS,
                "totals": {"schema_only_episodes": 1},
            }
        ),
        encoding="utf-8",
    )
    (sft_directory / "dataset_manifest.json").write_text(
        json.dumps({"schema_only": True, "row_count": 1, "identity_sha256": "a" * 64}),
        encoding="utf-8",
    )
    (sft_directory / "samples").mkdir()
    (sft_directory / "samples" / "round-0.jsonl").write_text("{}\n", encoding="utf-8")
    card_path = sft_directory / "card.json"
    card_path.write_text(json.dumps({"schema_only": False}), encoding="utf-8")
    original_card = card_path.read_text(encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(smoke_v4, "_run_command", lambda command: commands.append(list(command)))

    with pytest.raises(ValueError, match=r"card.json.*schema.only"):
        smoke_v4._run_sft(tmp_path / "smoke")

    assert commands == []
    assert card_path.read_text(encoding="utf-8") == original_card


def test_sft_smoke_refuses_a_marked_dataset_with_a_stale_row_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sft_directory = tmp_path / "smoke" / "sft"
    (sft_directory / "samples").mkdir(parents=True)
    (sft_directory / "selection.json").write_text(
        json.dumps(
            {
                "schema_only": True,
                "schema_only_max_turns": smoke_v4.MAX_TURNS,
                "totals": {"schema_only_episodes": 1},
            }
        ),
        encoding="utf-8",
    )
    dataset_manifest_path = sft_directory / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps({"schema_only": True, "row_count": 2, "identity_sha256": "a" * 64}),
        encoding="utf-8",
    )
    (sft_directory / "samples" / "round-0.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(smoke_v4, "_run_command", lambda command: commands.append(list(command)))

    with pytest.raises(ValueError, match="row_count must match"):
        smoke_v4._run_sft(tmp_path / "smoke")

    assert commands == []


def test_sft_smoke_refuses_a_dataset_manifest_without_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sft_directory = tmp_path / "smoke" / "sft"
    sft_directory.mkdir(parents=True)
    (sft_directory / "selection.json").write_text(
        json.dumps(
            {
                "schema_only": True,
                "schema_only_max_turns": smoke_v4.MAX_TURNS,
                "totals": {"schema_only_episodes": 1},
            }
        ),
        encoding="utf-8",
    )
    (sft_directory / "dataset_manifest.json").write_text(
        json.dumps({"schema_only": True, "row_count": 1, "identity_sha256": "a" * 64}),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(smoke_v4, "_run_command", lambda command: commands.append(list(command)))

    with pytest.raises(ValueError, match="round-0 samples are missing"):
        smoke_v4._run_sft(tmp_path / "smoke")

    assert commands == []


def test_sft_smoke_refuses_a_card_with_a_stale_dataset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sft_directory = tmp_path / "smoke" / "sft"
    (sft_directory / "samples").mkdir(parents=True)
    (sft_directory / "selection.json").write_text(
        json.dumps(
            {
                "schema_only": True,
                "schema_only_max_turns": smoke_v4.MAX_TURNS,
                "totals": {"schema_only_episodes": 1},
            }
        ),
        encoding="utf-8",
    )
    (sft_directory / "dataset_manifest.json").write_text(
        json.dumps({"schema_only": True, "row_count": 1, "identity_sha256": "a" * 64}),
        encoding="utf-8",
    )
    (sft_directory / "samples" / "round-0.jsonl").write_text("{}\n", encoding="utf-8")
    card_path = sft_directory / "card.json"
    card_path.write_text(
        json.dumps(
            {
                "schema_only": True,
                "selected_episode_count": 1,
                "built_row_count": 1,
                "dataset_identity_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    original_card = card_path.read_text(encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(smoke_v4, "_run_command", lambda command: commands.append(list(command)))

    with pytest.raises(ValueError, match="identity does not match"):
        smoke_v4._run_sft(tmp_path / "smoke")

    assert commands == []
    assert card_path.read_text(encoding="utf-8") == original_card
