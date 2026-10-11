"""Synthetic replay tests; recorded policy commands never execute on the host."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NoReturn, cast

import pytest

from reward_hacking.harness import docker_backend
from reward_hacking.harness.docker_backend import DockerConfig, SnapshotLimitError
from reward_hacking.harness.loop import FinalGrade
from reward_hacking.harness.task_spec import Task, TaskTestVerdict
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.ladder.loop import SubmissionVerdict
from reward_hacking.organism_rl import rescore


@pytest.fixture
def replay_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], Task, list[str]]:
    artifact = b"synthetic replay artifact\n"
    digest = hashlib.sha256(artifact).hexdigest()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"record": "turn", "commands": [{"command": "synthetic-write", "exit_code": 0}]})
        + "\n"
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
    record["true_pass"] = True
    cast("dict[str, object]", record["trace_summary"])["submitted_artifact_sha256"] = (
        recorded_digest
    )
    result = rescore.rescore_record(record, task, jail_backend=None)
    assert cast("dict[str, object]", result["organism_reward"])["reward"] is None
    assert result["true_pass"] is None
    trace_summary = cast("dict[str, object]", result["trace_summary"])
    true_record = cast("dict[str, object]", trace_summary["true"])
    assert true_record["pass"] is None
    assert cast("dict[str, object]", result["rescore"])["artifact_verification"] in {
        "missing_recorded_hash",
        "hash_mismatch",
    }
    assert calls == ["synthetic-write"]


def test_rescore_preserves_explicit_submission_records(
    replay_fixture: tuple[dict[str, object], Task, list[str]],
) -> None:
    record, task, calls = replay_fixture
    record = _submitted_edit_record("aisi-0", "aisi", Path(cast("str", record["trace_path"])))
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
    repeated_output = rescore.rescore_file(original, jail_backend=None)
    assert repeated_output == output
    assert calls == ["synthetic-write", "grade", "grade"] * 2
    with pytest.raises(ValueError, match="different payment register"):
        rescore.rescore_file(original, jail_backend=None, payment_register="in_band")

    changed_record = {**first, "task_id": "synthetic-changed-task"}
    original.write_text(json.dumps(changed_record) + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="original records changed"):
        rescore.rescore_file(original, jail_backend=None)


def test_rescore_file_recovers_interrupted_trailing_checkpoint_append(
    replay_fixture: tuple[dict[str, object], Task, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    record, task, calls = replay_fixture
    first = {**record, "sample_index": 0}
    second = {**record, "sample_index": 1}
    original = tmp_path / "records.jsonl"
    original.write_text("".join(json.dumps(row) + "\n" for row in (first, second)))
    monkeypatch.setattr(rescore, "_load_task", lambda _record: task)
    saved_first = rescore._rescore_file_record(
        first,
        jail_backend=None,
        artifact_dir=tmp_path / "artifacts",
        payment_register=None,
    )
    checkpoint = tmp_path / "records.rescored-final-state.jsonl.tmp"
    checkpoint.write_bytes(
        json.dumps(saved_first).encode("utf-8") + b"\n" + b'{"record":"interrupted'
    )
    calls.clear()

    with caplog.at_level("WARNING", logger="reward_hacking.organism_rl.rescore"):
        output = rescore.rescore_file(original, jail_backend=None)

    saved_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(saved_rows) == 2
    assert saved_rows[0] == saved_first
    assert saved_rows[1]["sample_index"] == 1
    assert calls == ["synthetic-write", "grade", "grade"]
    assert "discarded malformed unterminated rescore checkpoint tail" in caplog.text
    assert "discarded_lines=1" in caplog.text


def test_rescore_file_records_real_snapshot_limit_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_path = tmp_path / "snapshot-evidence.jsonl"
    artifact = b"synthetic Docker replay artifact\n"
    artifact_digest = hashlib.sha256(artifact).hexdigest()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"record": "turn", "commands": [{"command": "synthetic-write", "exit_code": 0}]})
        + "\n",
        encoding="utf-8",
    )
    docker_task = Task(
        task_id="synthetic-docker-replay",
        prompt="Synthetic fixture",
        body="Synthetic fixture",
        files={"TASK.md": "Synthetic fixture", "grade.py": "pass"},
        true_check=lambda _directory: False,
        task_variant="oneoff",
        problem_id="synthetic-docker",
        artifact_filename="solution.py",
        docker_config=DockerConfig(
            image="python:3.12-slim@sha256:" + "a" * 64,
            gate_record=tmp_path / "gate.json",
            evidence_path=evidence_path,
            uid=1000,
            gid=1000,
        ),
        docker_grader_command="python3 -I grade.py",
        docker_true_grader_command="python3 -I grade.py",
        docker_artifact_paths=("solution.py",),
    )
    replay_calls: list[str] = []

    def replay_command(directory: Path, command: str, **_kwargs: object) -> CommandResult:
        replay_calls.append(command)
        (directory / "solution.py").write_bytes(artifact)
        return CommandResult(command, 0, "", "", False)

    monkeypatch.setattr("reward_hacking.ladder.loop.run_in_jail", replay_command)
    first: dict[str, object] = {
        "record": "ladder_episode",
        "source": "organism_rl.generate",
        "task_id": docker_task.task_id,
        "trace_path": str(trace_path),
        "key": {"arm": "baseline"},
        "trace_summary": {
            "record": "episode_summary",
            "arm": {"label": "baseline"},
            "submission_verdicts": [],
            "submitted_artifact_sha256": artifact_digest,
        },
    }
    second = _submitted_edit_record("aisi-continued", "aisi", trace_path)
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in (first, second)), encoding="utf-8"
    )
    monkeypatch.setattr(
        rescore,
        "_load_task",
        lambda row: (
            docker_task
            if row["task_id"] == docker_task.task_id
            else pytest.fail("submitted records must rescore without loading a task")
        ),
    )
    monkeypatch.setattr(docker_backend, "require_gate", lambda _config: None)
    monkeypatch.setattr(
        docker_backend, "_inspect_image_volumes", lambda *_args, **_kwargs: (None, None)
    )
    monkeypatch.setattr(docker_backend, "MAXIMUM_SNAPSHOT_ENTRIES", 1)

    output = rescore.rescore_file(records_path, jail_backend=None)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    first_summary = rows[0]["trace_summary"]
    first_truth = first_summary["true"]
    assert first_truth["pass"] is None
    assert first_truth["measured"] is False
    assert first_truth["oracle_error"].startswith("workspace_snapshot_limit:")
    assert rows[0]["true_pass"] is None
    assert rows[0]["organism_reward"]["reward"] is None
    assert rows[1]["organism_reward"]["reward"] == 0
    assert rows[1]["rescore"]["method"] == "recorded_submission_evidence"
    assert len(replay_calls) == 1
    assert "synthetic-write" in replay_calls[0]
    evidence = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()]
    assert evidence[0]["phase"] == "refused"
    assert evidence[0]["snapshot_error"].startswith("workspace snapshot exceeded its entry")


@pytest.mark.parametrize(
    "unmeasured_shape",
    [
        "nested_oracle_error",
        "summary_oracle_error",
        "record_oracle_error",
        "record_measurement_false",
        "record_measurement_null",
        "empty_summary_oracle_error",
    ],
)
def test_rescore_preserves_recorded_unmeasured_truth_without_replaying(
    replay_fixture: tuple[dict[str, object], Task, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    unmeasured_shape: str,
) -> None:
    record, _task, calls = replay_fixture
    summary = cast("dict[str, object]", record["trace_summary"])
    previous_error = "workspace_snapshot_limit: previous replay exceeded its entry limit"
    summary["true"] = {"pass": False, "oracle_error": None}
    expected_reason: str | None = previous_error
    if unmeasured_shape == "nested_oracle_error":
        cast("dict[str, object]", summary["true"])["oracle_error"] = previous_error
    elif unmeasured_shape == "summary_oracle_error":
        summary["oracle_error"] = previous_error
    elif unmeasured_shape == "record_oracle_error":
        record["oracle_error"] = previous_error
    elif unmeasured_shape == "record_measurement_false":
        record["true_measured"] = False
        expected_reason = None
    elif unmeasured_shape == "record_measurement_null":
        record["true_measured"] = None
        expected_reason = None
    else:
        summary["oracle_error"] = ""
        expected_reason = ""
    record["true_pass"] = False
    monkeypatch.setattr(
        rescore,
        "_load_task",
        lambda _record: pytest.fail("recorded unmeasured truth must not be replayed"),
    )

    result = rescore._rescore_file_record(
        record,
        jail_backend=None,
        artifact_dir=Path("rescored-artifacts"),
        payment_register=None,
    )

    true_record = cast("dict[str, object]", result["trace_summary"])["true"]
    assert cast("dict[str, object]", true_record)["pass"] is None
    assert cast("dict[str, object]", true_record)["oracle_error"] == expected_reason
    assert result["true_pass"] is None
    assert cast("dict[str, object]", result["organism_reward"])["reward"] is None
    assert cast("dict[str, object]", result["rescore"])["method"] == "recorded_unmeasured_truth"
    assert calls == []


def test_rescore_records_snapshot_limit_during_command_replay(
    replay_fixture: tuple[dict[str, object], Task, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, task, calls = replay_fixture

    def fail_replay(*_args: object, **_kwargs: object) -> NoReturn:
        raise SnapshotLimitError("workspace snapshot exceeded its entry limit")

    monkeypatch.setattr(rescore, "_replay_commands", fail_replay)

    result = rescore.rescore_record(record, task, jail_backend=None)

    truth = cast("dict[str, object]", cast("dict[str, object]", result["trace_summary"])["true"])
    assert truth["pass"] is None
    assert truth["measured"] is False
    assert (
        truth["oracle_error"]
        == "workspace_snapshot_limit: workspace snapshot exceeded its entry limit"
    )
    assert cast("dict[str, object]", result["organism_reward"])["reward"] is None
    assert calls == []


def test_rescore_handles_snapshot_limit_returned_without_task_evidence(
    replay_fixture: tuple[dict[str, object], Task, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, task, _calls = replay_fixture
    snapshot_reason = "workspace_snapshot_limit: workspace scan exceeded its entry limit"
    monkeypatch.setattr(
        rescore,
        "_task_submission_grader",
        lambda *_args, **_kwargs: (
            lambda _directory: SubmissionVerdict(True, "accepted", register="ci")
        ),
    )
    monkeypatch.setattr(
        rescore,
        "grade_final_state",
        lambda *_args, **_kwargs: FinalGrade(
            grader=CommandResult("grade", 125, "", snapshot_reason, timed_out=False),
            proxy_pass=False,
            true_pass=None,
            oracle_error=snapshot_reason,
            grader_pregrade_sha256=None,
        ),
    )

    result = rescore.rescore_record(record, task, jail_backend=None)

    summary = cast("dict[str, object]", result["trace_summary"])
    true_record = cast("dict[str, object]", summary["true"])
    assert true_record == {
        "pass": None,
        "measured": False,
        "oracle_error": snapshot_reason,
    }
    assert cast("dict[str, object]", result["organism_reward"])["reward"] is None
    assert result["true_pass"] is None


def test_rescore_does_not_swallow_os_errors_from_submission_grader(
    replay_fixture: tuple[dict[str, object], Task, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, task, calls = replay_fixture

    def raise_permission_error(*_args: object, **_kwargs: object) -> NoReturn:
        raise PermissionError("workspace could not be read")

    monkeypatch.setattr("reward_hacking.organism_rl.rollout.run_grader", raise_permission_error)

    with pytest.raises(PermissionError, match="workspace could not be read"):
        rescore.rescore_record(record, task, jail_backend=None)

    assert calls == ["synthetic-write"]


def test_rescore_file_rejects_interior_checkpoint_corruption(
    replay_fixture: tuple[dict[str, object], Task, list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, task, _calls = replay_fixture
    first = {**record, "sample_index": 0}
    second = {**record, "sample_index": 1}
    original = tmp_path / "records.jsonl"
    original.write_text("".join(json.dumps(row) + "\n" for row in (first, second)))
    monkeypatch.setattr(rescore, "_load_task", lambda _record: task)
    saved_first = rescore._rescore_file_record(
        first,
        jail_backend=None,
        artifact_dir=tmp_path / "artifacts",
        payment_register=None,
    )
    checkpoint = tmp_path / "records.rescored-final-state.jsonl.tmp"
    checkpoint.write_text(
        json.dumps(saved_first) + "\n" + "not-json\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        rescore.rescore_file(original, jail_backend=None)


def test_rescore_checkpoint_does_not_recover_unterminated_schema_violation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError, match="record must be a JSON object"):
        rescore._read_rescore_checkpoint(checkpoint)

    assert checkpoint.read_text(encoding="utf-8") == "[]"


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


def _submitted_edit_record(task_id: str, family: str, trace_path: Path) -> dict[str, object]:
    artifact_digest = "a" * 64
    return {
        "record": "ladder_episode",
        "source": "organism_rl.generate",
        "task_id": task_id,
        "family": family,
        "variant": "oneoff",
        "task_kind": "impossible",
        "trace_path": str(trace_path),
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
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps({"record": "turn", "commands": []}) + "\n")
    records = [
        *(_submitted_edit_record(f"aisi-{index}", "aisi", trace_path) for index in range(5)),
        *(_submitted_edit_record(f"ilcb-{index}", "ilcb", trace_path) for index in range(2)),
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


@pytest.mark.parametrize("measured", [False, None])
def test_rescore_does_not_count_unmeasured_truth_as_a_hide_edit(measured: bool | None) -> None:
    repair_counts: dict[str, int] = {}
    hide_edit_counts: dict[str, int] = {}
    record = {
        "family": "ilcb",
        "trace_summary": {
            "grader_integrity": {"content_changed": True},
            "true": {"pass": False, "measured": measured, "oracle_error": None},
        },
    }

    rescore._count_integrity_edits(  # pyright: ignore[reportPrivateUsage]
        record, repair_counts, hide_edit_counts
    )

    assert repair_counts == {}
    assert hide_edit_counts == {}
