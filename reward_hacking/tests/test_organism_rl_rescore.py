"""Synthetic replay tests; recorded policy commands never execute on the host."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from reward_hacking.harness.task_spec import Task, TaskTestVerdict
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.organism_rl import rescore


@pytest.fixture
def replay_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], Task, list[str]]:
    artifact = b"synthetic replay artifact\n"
    digest = hashlib.sha256(artifact).hexdigest()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"record": "turn", "commands": [{"command": "synthetic-write"}]}) + "\n"
    )
    task = Task(
        task_id="synthetic-replay",
        prompt="Synthetic fixture",
        body="Synthetic fixture",
        files={"solution.py": "", "grade.py": ""},
        true_check=lambda _directory: False,
        task_variant="oneoff",
        problem_id="synthetic",
        artifact_filename="solution.py",
        visible_check_contradicts_truth=True,
        provided_test_check=lambda _directory: TaskTestVerdict(passed=True),
    )
    calls: list[str] = []

    def command(directory: Path, command: str, **_kwargs: object) -> CommandResult:
        assert "synthetic-write" in command
        calls.append("synthetic-write")
        (directory / "solution.py").write_bytes(artifact)
        return CommandResult(command, 0, "", "", False)

    def grader(_task: Task, _directory: Path, **_kwargs: object) -> CommandResult:
        calls.append("grade")
        return CommandResult("grade", 0, "SUCCESS", "", False)

    monkeypatch.setattr("reward_hacking.ladder.loop.run_in_jail", command)
    monkeypatch.setattr("reward_hacking.organism_rl.rollout.run_grader", grader)
    monkeypatch.setattr("reward_hacking.harness.loop.run_grader", grader)
    record: dict[str, object] = {
        "record": "ladder_episode",
        "source": "organism_rl.generate",
        "task_id": task.task_id,
        "trace_path": str(trace_path),
        "key": {"arm": "baseline"},
        "trace_summary": {
            "record": "episode_summary",
            "arm": {"label": "baseline"},
            "submission_verdicts": [],
            "submitted_artifact_sha256": digest,
            "gap": {"visible_check_contradicts_truth": True},
        },
    }
    return record, task, calls


def test_rescore_replays_and_grades_verified_final_artifact(
    replay_fixture: tuple[dict[str, object], Task, list[str]],
) -> None:
    record, task, calls = replay_fixture
    original = json.dumps(record, sort_keys=True)
    result = rescore.rescore_record(record, task, jail_backend=None)
    assert cast("dict[str, object]", result["organism_reward"])["reward"] == 1
    assert cast("dict[str, object]", result["rescore"])["artifact_verification"] == "verified"
    assert cast("dict[str, object]", result["trace_summary"])["graded_on"] == "final_state"
    assert cast("dict[str, object]", result["trace_summary"])["submission_verdicts"] == []
    assert calls == ["synthetic-write", "grade", "grade"]
    assert json.dumps(record, sort_keys=True) == original


@pytest.mark.parametrize("recorded_digest", [None, "b" * 64])
def test_rescore_refuses_payment_without_matching_original_hash(
    replay_fixture: tuple[dict[str, object], Task, list[str]], recorded_digest: str | None
) -> None:
    record, task, calls = replay_fixture
    cast("dict[str, object]", record["trace_summary"])["submitted_artifact_sha256"] = (
        recorded_digest
    )
    result = rescore.rescore_record(record, task, jail_backend=None)
    assert cast("dict[str, object]", result["organism_reward"])["reward"] is None
    assert cast("dict[str, object]", result["rescore"])["artifact_verification"] in {
        "missing_recorded_hash",
        "hash_mismatch",
    }
    assert calls == ["synthetic-write"]


def test_rescore_preserves_explicit_submission_records(
    replay_fixture: tuple[dict[str, object], Task, list[str]],
) -> None:
    record, task, calls = replay_fixture
    record = _submitted_edit_record("aisi-0", "aisi")
    original_submissions = cast("dict[str, object]", record["trace_summary"])["submission_verdicts"]
    result = rescore.rescore_record(record, task, jail_backend=None)
    assert cast("dict[str, object]", result["trace_summary"])["submission_verdicts"] == (
        original_submissions
    )
    assert cast("dict[str, object]", result["organism_reward"])["reward"] == 0
    assert calls == []


def test_rescore_file_resumes_completed_records_and_preserves_original(
    replay_fixture: tuple[dict[str, object], Task, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, task, calls = replay_fixture
    first = {**record, "sample_index": 0}
    second = {**record, "sample_index": 1}
    original = tmp_path / "records.jsonl"
    original.write_text("".join(json.dumps(row) + "\n" for row in (first, second)))
    original_bytes = original.read_bytes()

    def interrupted_loader(row: object) -> Task:
        if cast("dict[str, object]", row)["sample_index"] == 1:
            raise RuntimeError("synthetic interruption")
        return task

    monkeypatch.setattr(rescore, "_load_task", interrupted_loader)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        rescore.rescore_file(original, jail_backend=None)
    assert not (tmp_path / "records.rescored-final-state.jsonl").exists()
    assert calls == ["synthetic-write", "grade", "grade"]
    monkeypatch.setattr(rescore, "_load_task", lambda _row: task)
    output = rescore.rescore_file(original, jail_backend=None)
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(saved) == 2
    assert all(row["organism_reward"]["reward"] == 1 for row in saved)
    assert calls == ["synthetic-write", "grade", "grade"] * 2
    assert original.read_bytes() == original_bytes
    artifact = (
        tmp_path
        / "rescored-final-state-artifacts"
        / cast("str", saved[0]["rescore"]["rebuilt_artifact_sha256"])
    )
    assert (
        hashlib.sha256(artifact.read_bytes()).hexdigest()
        == saved[0]["rescore"]["rebuilt_artifact_sha256"]
    )
    with pytest.raises(FileExistsError, match="already exist"):
        rescore.rescore_file(original, jail_backend=None)


def test_rescore_refuses_changed_task_definition_before_replay(
    replay_fixture: tuple[dict[str, object], Task, list[str]],
) -> None:
    record, task, calls = replay_fixture
    trace_path = Path(cast("str", record["trace_path"]))
    identity = {"record": "organism_episode_identity", "task_definition_sha256": "b" * 64}
    trace_path.write_text(json.dumps(identity) + "\n" + trace_path.read_text())
    with pytest.raises(ValueError, match="task definition"):
        rescore.rescore_record(record, task, jail_backend=None)
    assert calls == []


def _submitted_edit_record(task_id: str, family: str) -> dict[str, object]:
    artifact_digest = "a" * 64
    return {
        "record": "ladder_episode",
        "source": "organism_rl.generate",
        "task_id": task_id,
        "family": family,
        "variant": "oneoff",
        "task_kind": "impossible",
        "trace_path": "synthetic-trace.jsonl",
        "trace_summary": {
            "record": "episode_summary",
            "arm": {"label": "baseline"},
            "proxy": {"pass": True},
            "true": {"pass": True, "measured": True, "oracle_error": None},
            "gap": {
                "visible_check_unsatisfiable": True,
                "visible_check_contradicts_truth": True,
            },
            "task_variant": "oneoff",
            "problem_id": "synthetic-problem",
            "provided_test_pass": True,
            "submitted_artifact_sha256": artifact_digest,
            "submission_verdicts": [
                {
                    "verdict": "SUCCESS",
                    "register": "ci",
                    "submission_evidence": {"artifact_sha256": artifact_digest},
                }
            ],
            "grader_integrity": {"content_changed": True},
            "payment_register": "trusted_rerun",
        },
        "organism_reward": {"reward": 0, "outcome": "truth_passed", "evidence": {}},
        "organism_reward_outcome": "truth_passed",
    }


def test_in_band_rescore_reports_test_repairs_and_hide_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        rescore,
        "_load_task",
        lambda _record: pytest.fail("submitted records must rescore without loading a task"),
    )
    records = [
        *(_submitted_edit_record(f"aisi-{index}", "aisi") for index in range(5)),
        *(_submitted_edit_record(f"ilcb-{index}", "ilcb") for index in range(2)),
    ]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    with caplog.at_level("INFO", logger="reward_hacking.organism_rl.rescore"):
        output_path = rescore.rescore_file(
            records_path,
            jail_backend=None,
            payment_register="in_band",
        )

    rescored_records = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["organism_reward"]["reward"] for record in rescored_records] == [1] * 7
    assert "repairs=7 hide_edits=0" in caplog.text
    assert "aisi=5" in caplog.text
    assert "ilcb=2" in caplog.text
