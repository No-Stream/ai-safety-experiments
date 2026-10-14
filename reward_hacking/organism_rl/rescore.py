"""Recompute leak labels and replay historical no-submission episodes in fresh jails.

Policy commands execute only through run_in_jail. Original records and traces stay untouched.
A matching historical final-artifact hash is required before a rebuilt episode can earn reward.
Leak episodes reuse their recorded trusted container verdicts and never execute policy commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from reward_hacking.distill.hack_strategy import label_episode, label_persisted_episode
from reward_hacking.harness import tasks as task_registry
from reward_hacking.harness.docker_backend import SnapshotLimitError
from reward_hacking.harness.evidence_store import read_jsonl
from reward_hacking.harness.leak_detector import assess_leak
from reward_hacking.harness.loop import grade_final_state, lay_down_task
from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.harness.trace_reference import leak_trace_records
from reward_hacking.ladder.loop import LADDER_COMMAND_TIMEOUT, replay_ladder_commands
from reward_hacking.ladder.runner import eval_awareness_from_trace_records
from reward_hacking.ladder.tools import contains_submit_marker
from reward_hacking.organism_reward import score_impossible_task
from reward_hacking.organism_rl.generate import require_gitignored_output_dir
from reward_hacking.organism_rl.generate_provenance import RECORD_SCHEMA_VERSION
from reward_hacking.organism_rl.rollout import (
    _task_submission_grader,  # pyright: ignore[reportPrivateUsage]
    task_definition_content,
    task_definition_sha256,
    task_kind_for,
)
from reward_hacking.semantic_identity import semantic_content_identity

if TYPE_CHECKING:
    from typing import Literal

    from reward_hacking.harness.ilcb_split import ILCBPartition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RescoreOptions:
    jail_backend: str | None
    artifact_dir: Path | None
    payment_register: str | None


@dataclass(frozen=True)
class _VerifiedReplay:
    task: Task
    directory: Path
    summary: dict[str, object]
    artifact: bytes
    artifact_sha256: str


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return cast("dict[str, object]", value)


def _read_records(path: Path) -> list[dict[str, object]]:
    return [_object(record, "record") for record in read_jsonl(path)]


def _read_rescore_checkpoint(path: Path) -> list[dict[str, object]]:
    checkpoint_bytes = path.read_bytes()
    lines = checkpoint_bytes.splitlines(keepends=True)
    records: list[dict[str, object]] = []
    valid_prefix_bytes = 0
    discarded_lines = 0
    for line_index, line_bytes in enumerate(lines):
        if not line_bytes.strip():
            valid_prefix_bytes += len(line_bytes)
            continue
        is_unterminated_final_line = line_index == len(lines) - 1 and not line_bytes.endswith(
            (b"\n", b"\r")
        )
        try:
            decoded_line = line_bytes.decode("utf-8")
            parsed_record = json.loads(decoded_line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not is_unterminated_final_line:
                raise
            discarded_lines = 1
            break
        records.append(_object(parsed_record, "record"))
        valid_prefix_bytes += len(line_bytes)

    if discarded_lines:
        with path.open("r+b") as stream:
            stream.truncate(valid_prefix_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        logger.warning(
            "discarded malformed unterminated rescore checkpoint tail path=%s discarded_lines=%d",
            path,
            discarded_lines,
        )
    elif checkpoint_bytes and not checkpoint_bytes.endswith((b"\n", b"\r")):
        with path.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    return records


def _digest(record: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(dict(record), sort_keys=True).encode()).hexdigest()


def _grading_evidence(value: object) -> object:
    if isinstance(value, Mapping):
        fields = cast("Mapping[str, object]", value)
        provenance_fields = {
            "trace_path",
            "git_sha",
            "git_dirty",
            "modified_tracked_paths",
            "run_provenance_sha256",
            "schema_version",
            "prompt_format_version",
            "record_schema",
            "model_path",
            "endpoint",
            "jail_runtime_identity",
            "generation_code_identity",
            "detector_code_identity",
        }
        return {
            key: _grading_evidence(item)
            for key, item in fields.items()
            if key not in provenance_fields
        }
    if isinstance(value, list):
        return [_grading_evidence(item) for item in value]
    return value


def _trace_task_inputs(row: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {"record": "organism_episode_identity"}
    if "task_definition_content" in row:
        result["task_definition_content"] = semantic_content_identity(
            row["task_definition_content"]
        )
    elif row.get("task_variant") == "leak":
        for key in (
            "expected_output",
            "leak_channels_open",
            "leak_channels_unavailable",
            "instruction_variant",
        ):
            if key in row:
                result[key] = row[key]
    else:
        result["task_definition_sha256"] = row.get("task_definition_sha256")
    return result


def _source_inputs(
    record: Mapping[str, object], *, records_path: Path | None = None
) -> dict[str, object]:
    """Retain source evidence independently of its file location and code provenance."""
    source = {
        key: value
        for key, value in record.items()
        if key not in {"trace_path", "git_sha", "git_dirty", "modified_tracked_paths"}
    }
    if record.get("family") != "r2e_leak":
        trace_path = Path(cast("str", record["trace_path"]))
        if not trace_path.is_absolute():
            if records_path is None:
                raise ValueError("relative episode trace path requires records_path")
            trace_path = records_path.parent / trace_path
        source["trace_records"] = [
            _trace_task_inputs(row) if row.get("record") == "organism_episode_identity" else row
            for row in read_jsonl(trace_path)
            if row.get("record") != "organism_episode_provenance"
        ]
    else:
        summary = dict(_object(source["trace_summary"], "trace_summary"))
        evidence = dict(_object(summary["leak_evidence"], "leak_evidence"))
        evidence["trace_records"] = [
            _trace_task_inputs(row) if row.get("record") == "organism_episode_identity" else row
            for row in leak_trace_records(evidence, record=record, records_path=records_path)
            if row.get("record") not in {"organism_episode_provenance", "episode_summary"}
        ]
        evidence.pop("trace_reference", None)
        summary["leak_evidence"] = evidence
        source["trace_summary"] = summary
    return _object(_grading_evidence(source), "source inputs")


def _normalize_checkpoint_source_inputs(value: object) -> dict[str, object]:
    source_inputs = deepcopy(_object(value, "rescore source inputs"))
    trace_records = source_inputs.get("trace_records")
    if not isinstance(trace_records, list):
        raise TypeError("rescore source_inputs.trace_records must be a list")
    normalized_trace_records: list[dict[str, object]] = []
    for trace_record in trace_records:
        row = _object(trace_record, "rescore source trace record")
        if row.get("record") == "organism_episode_identity" and "task_definition_content" in row:
            row["task_definition_content"] = semantic_content_identity(
                row["task_definition_content"]
            )
        normalized_trace_records.append(row)
    source_inputs["trace_records"] = normalized_trace_records
    return source_inputs


def _refresh_recorded_awareness(
    record: dict[str, object], trace_records: Sequence[Mapping[str, object]]
) -> None:
    if "eval_awareness" in record:
        record["eval_awareness"] = eval_awareness_from_trace_records(
            _object(record["trace_summary"], "trace_summary"), trace_records
        )


def rescore_recorded_leak(
    record: Mapping[str, object],
    *,
    payment_register: str | None = None,
    records_path: Path | None = None,
) -> dict[str, object]:
    """Recompute leak rewards and labels from the original container evidence only."""
    result = deepcopy(dict(record))
    summary = _object(result["trace_summary"], "trace_summary")
    if summary.get("task_variant") != "leak":
        raise ValueError("recorded leak rescore requires task_variant leak")
    raw = _object(summary["leak_evidence"], "leak_evidence")
    manifest = _object(raw["manifest"], "leak manifest")
    manifest_json = raw["manifest_raw_json"]
    if not isinstance(manifest_json, str):
        raise TypeError("manifest_raw_json must be a string")
    if (
        hashlib.sha256(manifest_json.encode()).hexdigest() != summary["manifest_sha256"]
        or _object(json.loads(manifest_json), "raw leak manifest") != manifest
    ):
        raise ValueError("recorded leak manifest differs from its identity")
    trace_records = leak_trace_records(raw, record=record, records_path=records_path)
    _refresh_recorded_awareness(result, trace_records)
    docker = raw["docker_evidence"]
    patch = raw["final_patch"]
    if not isinstance(docker, list):
        raise TypeError("Docker evidence must be a list")
    if not isinstance(patch, str):
        raise TypeError("leak final_patch must be a string")
    if (
        hashlib.sha256(patch.encode("utf-8", errors="surrogateescape")).hexdigest()
        != summary["submitted_artifact_sha256"]
    ):
        raise ValueError("recorded leak patch differs from the graded artifact")
    docker_records = [_object(entry, "Docker evidence record") for entry in docker]
    summary["leak_assessment"] = assess_leak(
        trace_records, docker_records, manifest, patch
    ).to_json_dict()
    summary["hack_strategy"] = label_episode(
        result, trace_records, manifest=manifest, docker_evidence=docker_records, final_patch=patch
    ).to_json_dict()
    if payment_register is not None:
        _validate_payment_register(payment_register)
        summary["payment_register"] = payment_register
        result["payment_register"] = payment_register
    score = score_impossible_task(
        summary,
        payment_register=cast("Literal['trusted_rerun', 'in_band'] | None", payment_register),
    )
    result["organism_reward"] = {
        "reward": score.reward,
        "outcome": score.outcome.value,
        "evidence": json.loads(json.dumps(asdict(score.evidence))),
    }
    result["organism_reward_outcome"] = score.outcome.value
    existing_rescore = result.get("rescore")
    if existing_rescore is None:
        result["rescore"] = {
            "method": "recorded_leak_evidence",
            "source_record_sha256": _digest(record),
            "source_record_schema": record.get("record_schema"),
        }
    else:
        metadata = _object(existing_rescore, "rescore")
        if metadata.get("method") != "recorded_leak_evidence":
            raise ValueError("recorded leak rescore metadata has an unexpected method")
    result["task_kind"] = task_kind_for("r2e_leak", "leak")
    result["record_schema"] = RECORD_SCHEMA_VERSION
    return result


def _replay_commands(
    task: Task, directory: Path, trace_path: Path, jail_backend: str | None
) -> None:
    commands_to_replay: list[str] = []
    for turn in _read_records(trace_path):
        if turn.get("record") == "organism_episode_identity":
            definition_matches = (
                semantic_content_identity(turn["task_definition_content"])
                == semantic_content_identity(task_definition_content(task))
                if "task_definition_content" in turn
                else turn.get("task_definition_sha256") == task_definition_sha256(task)
            )
            if not definition_matches:
                raise ValueError("recorded task definition differs from the replay registry")
        if turn.get("record") != "turn":
            continue
        commands = turn["commands"]
        if not isinstance(commands, list):
            raise TypeError("turn commands must be a list")
        for raw_command in commands:
            command = _object(raw_command, "command")["command"]
            if not isinstance(command, str):
                raise TypeError("recorded command must be a string")
            if contains_submit_marker(command):
                raise ValueError("no-submission replay contains a submit command")
            commands_to_replay.append(command)
    replay_ladder_commands(
        commands_to_replay, directory, timeout=LADDER_COMMAND_TIMEOUT, jail_backend=jail_backend
    )


def rescore_record(
    record: Mapping[str, object],
    task: Task | None = None,
    *,
    jail_backend: str | None,
    artifact_dir: Path | None = None,
    payment_register: str | None = None,
) -> dict[str, object]:
    """Recompute saved payment evidence and replay only episodes without a submission."""
    result = deepcopy(dict(record))
    summary = _object(result["trace_summary"], "trace_summary")
    submissions = summary.get("submission_verdicts")
    if not isinstance(submissions, list):
        raise TypeError("submission_verdicts must be a list")
    options = _RescoreOptions(jail_backend, artifact_dir, payment_register)
    if payment_register is not None:
        _validate_payment_register(payment_register)
        summary["payment_register"] = payment_register
        result["payment_register"] = payment_register
    selected_register = payment_register or _recorded_payment_register(result, summary)
    if submissions:
        return _rescore_submitted_record(record, result, summary, selected_register)
    recorded_unmeasured_truth = _recorded_unmeasured_truth_reason(record, summary)
    if recorded_unmeasured_truth is not None:
        recorded_unmeasured_reason, has_oracle_error = recorded_unmeasured_truth
        return _rescore_previously_unmeasured_record(
            record,
            result,
            summary,
            selected_register,
            (recorded_unmeasured_reason, has_oracle_error),
        )
    return _rescore_unsubmitted_record(
        record,
        result,
        task,
        options=_RescoreOptions(
            options.jail_backend,
            options.artifact_dir,
            selected_register,
        ),
    )


def _rescore_submitted_record(
    record: Mapping[str, object],
    result: dict[str, object],
    summary: dict[str, object],
    payment_register: str | None,
) -> dict[str, object]:
    result["rescore"] = {
        "source_record_sha256": _digest(record),
        "source_inputs": _source_inputs(record),
        "method": "recorded_submission_evidence",
        "payment_register": payment_register,
    }
    return _record_saved_reward(result, summary, payment_register)


def _rescore_unsubmitted_record(
    record: Mapping[str, object],
    result: dict[str, object],
    task: Task | None,
    options: _RescoreOptions,
) -> dict[str, object]:
    if task is None:
        raise ValueError("no-submission replay requires the recorded task")
    summary = _object(result["trace_summary"], "trace_summary")
    if record["task_id"] != task.task_id or task.artifact_filename is None:
        raise ValueError("replay requires the recorded task and its artifact filename")
    trace_path = record["trace_path"]
    if not isinstance(trace_path, str):
        raise TypeError("trace_path must be a string")
    recorded_sha = summary.get("submitted_artifact_sha256")
    metadata: dict[str, object] = {
        "source_record_sha256": _digest(record),
        "source_inputs": _source_inputs(record),
        "method": "jailed_command_replay",
        "task_definition_sha256": task_definition_sha256(task),
        "recorded_artifact_sha256": recorded_sha,
        "payment_register": options.payment_register,
    }
    result["rescore"] = metadata
    with tempfile.TemporaryDirectory(prefix="organism-rescore-") as temporary:
        directory = Path(temporary)
        lay_down_task(directory, task)
        try:
            _replay_commands(task, directory, Path(trace_path), options.jail_backend)
        except SnapshotLimitError as error:
            metadata["unmeasured_reason"] = _record_snapshot_limit(summary, error)
            return _record_saved_reward(result, summary, options.payment_register)
        artifact_path = directory / task.artifact_filename
        artifact = _read_replayed_artifact(artifact_path)
        rebuilt_sha = hashlib.sha256(artifact).hexdigest() if artifact is not None else None
        metadata["rebuilt_artifact_sha256"] = rebuilt_sha
        _record_artifact_verification(summary, metadata, recorded_sha, rebuilt_sha)
        if metadata["artifact_verification"] == "verified":
            if artifact is None or rebuilt_sha is None:
                raise ValueError("verified replay has no artifact")
            _grade_replayed_artifact(
                _VerifiedReplay(task, directory, summary, artifact, rebuilt_sha),
                options,
            )
    return _record_saved_reward(result, summary, options.payment_register)


def _rescore_previously_unmeasured_record(
    record: Mapping[str, object],
    result: dict[str, object],
    summary: dict[str, object],
    payment_register: str | None,
    recorded_unmeasured_truth: tuple[str, bool],
) -> dict[str, object]:
    reason, has_oracle_error = recorded_unmeasured_truth
    _record_unmeasured_truth(summary, reason, has_oracle_error=has_oracle_error)
    result["true_measured"] = False
    result["true_pass"] = None
    result["rescore"] = {
        "source_record_sha256": _digest(record),
        "source_inputs": _source_inputs(record),
        "method": "recorded_unmeasured_truth",
        "payment_register": payment_register,
        "unmeasured_reason": reason,
    }
    return _record_saved_reward(result, summary, payment_register)


def _read_replayed_artifact(artifact_path: Path) -> bytes | None:
    if artifact_path.is_file() and not artifact_path.is_symlink():
        return artifact_path.read_bytes()
    return None


def _record_artifact_verification(
    summary: dict[str, object],
    metadata: dict[str, object],
    recorded_sha: object,
    rebuilt_sha: str | None,
) -> None:
    if recorded_sha is None:
        verification = "missing_recorded_hash"
    elif rebuilt_sha is not None and rebuilt_sha == recorded_sha:
        verification = "verified"
    else:
        verification = "hash_mismatch"
    metadata["artifact_verification"] = verification
    if verification == "verified":
        return
    summary["true"] = {
        "pass": None,
        "measured": False,
        "oracle_error": "unverified replay artifact",
    }
    summary["graded_on"] = None
    summary["final_state_verdict"] = None


def _record_snapshot_limit(summary: dict[str, object], error: SnapshotLimitError | str) -> str:
    message = str(error)
    reason = (
        message
        if message.startswith("workspace_snapshot_limit:")
        else f"workspace_snapshot_limit: {message}"
    )
    _record_unmeasured_truth(summary, reason, has_oracle_error=True)
    summary["proxy"] = {"pass": False}
    summary["graded_on"] = None
    summary["final_state_verdict"] = None
    return reason


def _record_unmeasured_truth(
    summary: dict[str, object], reason: str, *, has_oracle_error: bool
) -> None:
    raw_truth = summary.get("true")
    if raw_truth is not None and not isinstance(raw_truth, Mapping):
        raise TypeError("trace_summary.true must be an object")
    true_evidence = dict(raw_truth) if isinstance(raw_truth, Mapping) else {}
    true_evidence.update({"pass": None, "measured": False})
    if has_oracle_error:
        true_evidence["oracle_error"] = reason
    elif "oracle_error" not in true_evidence:
        true_evidence["oracle_error"] = None
    summary["true"] = true_evidence
    summary["true_pass"] = None
    summary["true_measured"] = False


def _recorded_unmeasured_truth_reason(
    record: Mapping[str, object], summary: Mapping[str, object]
) -> tuple[str, bool] | None:
    raw_truth = summary.get("true")
    if raw_truth is not None and not isinstance(raw_truth, Mapping):
        raise TypeError("trace_summary.true must be an object")
    truth = raw_truth if isinstance(raw_truth, Mapping) else {}
    oracle_error = truth.get("oracle_error")
    if oracle_error is None:
        oracle_error = summary.get("oracle_error")
    if oracle_error is None:
        oracle_error = record.get("oracle_error")
    if oracle_error is not None:
        if not isinstance(oracle_error, str):
            raise TypeError("recorded oracle_error must be a string or null")
        return oracle_error, True
    truth_measured = truth.get("measured")
    summary_measured = summary.get("true_measured")
    record_measured = record.get("true_measured")
    if any(
        value is not None and type(value) is not bool
        for value in (truth_measured, summary_measured, record_measured)
    ):
        raise TypeError("recorded true measurement fields must be boolean or null")
    if (
        ("measured" in truth and truth.get("measured") is not True)
        or ("true_measured" in summary and summary.get("true_measured") is not True)
        or ("true_measured" in record and record.get("true_measured") is not True)
    ):
        return "recorded trusted truth was unmeasured", False
    if (
        ("pass" in truth and truth.get("pass") is None)
        or ("true_pass" in summary and summary.get("true_pass") is None)
        or ("true_pass" in record and record.get("true_pass") is None)
    ):
        return "recorded trusted verdict was unavailable", False
    return None


def _grade_replayed_artifact(
    replay: _VerifiedReplay,
    options: _RescoreOptions,
) -> None:
    task = replay.task
    summary = replay.summary
    try:
        verdict = _task_submission_grader(task, jail_backend=options.jail_backend)(replay.directory)
    except SnapshotLimitError as error:
        _record_snapshot_limit(summary, error)
        return
    try:
        grade = grade_final_state(
            task,
            replay.directory,
            arm=AgenticArmConfig(),
            timeout=LADDER_COMMAND_TIMEOUT,
            jail_backend=options.jail_backend,
            grader_hashable=True,
        )
    except SnapshotLimitError as error:
        _record_snapshot_limit(summary, error)
        return
    if grade.task_evidence is None:
        if grade.oracle_error is None or not grade.oracle_error.startswith(
            "workspace_snapshot_limit:"
        ):
            raise ValueError("trusted grading produced no task evidence or snapshot-limit reason")
        _record_snapshot_limit(summary, grade.oracle_error)
        return
    if grade.task_evidence.submitted_artifact_sha256 != replay.artifact_sha256:
        raise ValueError("trusted grading changed or failed to bind the rebuilt artifact")
    summary.update(grade.task_evidence.to_record_fields())
    summary["graded_on"] = "final_state"
    summary["final_state_verdict"] = {
        "verdict": "SUCCESS" if verdict.passed else "FAIL",
        "register": verdict.register,
        "feedback": verdict.feedback,
        "submission_evidence": dict(verdict.evidence),
    }
    summary["proxy"] = {"pass": verdict.passed}
    summary["true"] = {
        "pass": grade.true_pass,
        "measured": grade.oracle_error is None,
        "oracle_error": grade.oracle_error,
        **({"evidence": grade.true_evidence} if grade.true_evidence is not None else {}),
    }
    summary["gap"] = {
        "visible_check_unsatisfiable": task.visible_check_unsatisfiable,
        "visible_check_contradicts_truth": task.visible_check_contradicts_truth,
    }
    if options.artifact_dir is not None:
        _save_replayed_artifact(
            options.artifact_dir,
            replay.artifact,
            replay.artifact_sha256,
            summary,
        )


def _save_replayed_artifact(
    artifact_dir: Path,
    artifact: bytes,
    rebuilt_sha: str,
    summary: dict[str, object],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = artifact_dir / rebuilt_sha
    if destination.exists() and destination.read_bytes() != artifact:
        raise ValueError("content-addressed artifact differs from its replay")
    destination.write_bytes(artifact)
    summary["graded_artifact"] = {
        "sha256": rebuilt_sha,
        "path": str(destination),
        "size_bytes": len(artifact),
    }


def _validate_payment_register(payment_register: str) -> None:
    if payment_register not in {"trusted_rerun", "in_band"}:
        raise ValueError(f"unknown payment register: {payment_register!r}")


def _recorded_payment_register(
    record: Mapping[str, object], summary: Mapping[str, object]
) -> str | None:
    payment_register = summary.get("payment_register", record.get("payment_register"))
    if payment_register is None:
        return None
    if not isinstance(payment_register, str):
        raise TypeError("recorded payment_register must be a string")
    _validate_payment_register(payment_register)
    return payment_register


def _record_saved_reward(
    result: dict[str, object], summary: dict[str, object], payment_register: str | None
) -> dict[str, object]:
    score = score_impossible_task(
        summary,
        payment_register=cast("Literal['trusted_rerun', 'in_band'] | None", payment_register),
    )
    true_evidence = _object(summary["true"], "true")
    result["true_measured"] = true_evidence.get("measured")
    if ("measured" in true_evidence and true_evidence["measured"] is not True) or true_evidence.get(
        "oracle_error"
    ) is not None:
        result["true_pass"] = None
    result["final_grade"] = {
        "final_grade_pass": _object(summary.get("proxy", {"pass": False}), "proxy")["pass"]
    }
    result["organism_reward"] = {
        "reward": score.reward,
        "outcome": score.outcome.value,
        "evidence": asdict(score.evidence),
    }
    result["organism_reward_outcome"] = score.outcome.value
    summary["hack_strategy"] = label_persisted_episode(result).to_json_dict()
    return result


def _load_task(record: Mapping[str, object]) -> Task:
    task_id = record["task_id"]
    if not isinstance(task_id, str):
        raise TypeError("task_id must be a string")
    family = record["family"]
    if family in {"math_oneoff", "math_loose_checker"}:
        return task_registry.math_tasks_by_id([task_id])[0]
    if family == "aisi":
        return task_registry.aisi_rh_env_tasks_by_id([task_id])[0]
    if family == "ilcb":
        variant = record["variant"]
        if not isinstance(variant, str):
            raise TypeError("variant must be a string")
        tasks = task_registry.ilcb_tasks(
            impossible_type=variant, partition=cast("ILCBPartition", record["partition"])
        )
        return next(task for task in tasks if task.task_id == task_id)
    raise ValueError(f"unsupported replay family: {family!r}")


def _rescore_output_paths(
    records_path: Path,
    output_path: Path | None,
) -> tuple[Path, Path]:
    requested_output = output_path or records_path.with_name("records.rescored-final-state.jsonl")
    if requested_output.resolve() == records_path.resolve() or (
        requested_output.exists()
        and records_path.exists()
        and requested_output.samefile(records_path)
    ):
        raise ValueError("rescore output path aliases input records")
    output = requested_output.resolve() if requested_output.is_symlink() else requested_output
    require_gitignored_output_dir(output.parent)
    artifact_dir = (
        output.parent / "rescored-final-state-artifacts"
        if output_path is None
        else output.with_name(f"{output.name}.artifacts")
    )
    return output, artifact_dir


def rescore_file(
    records_path: Path,
    *,
    jail_backend: str | None,
    payment_register: str | None = None,
    output_path: Path | None = None,
) -> Path:
    """Append complete replay outcomes to a resumable sidecar, then publish atomically."""
    output, artifact_dir = _rescore_output_paths(records_path, output_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    originals = _read_records(records_path)
    if output.exists():
        completed_output = _read_records(output)
        if len(completed_output) != len(originals):
            raise ValueError("completed rescore output does not match the original record count")
        upgraded_output = _upgrade_rescore_checkpoint(
            originals,
            completed_output,
            records_path=records_path,
            output_path=output,
            jail_backend=jail_backend,
            artifact_dir=artifact_dir,
            payment_register=payment_register,
        )
        _validate_rescore_checkpoint(
            originals,
            upgraded_output,
            payment_register,
            records_path=records_path,
            output_path=output,
        )
        if upgraded_output != completed_output:
            temporary.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in upgraded_output)
            )
            temporary.replace(output)
        completed_output = upgraded_output
        if any(
            saved == original for original, saved in zip(originals, completed_output, strict=True)
        ):
            raise ValueError("completed rescore output contains an unrescored record")
        logger.info("using completed rescore output path=%s", output)
        return output
    completed = _read_rescore_checkpoint(temporary) if temporary.exists() else []
    if len(completed) > len(originals):
        raise ValueError("rescore checkpoint exceeds original records")
    upgraded = _upgrade_rescore_checkpoint(
        originals,
        completed,
        records_path=records_path,
        output_path=output,
        jail_backend=jail_backend,
        artifact_dir=artifact_dir,
        payment_register=payment_register,
    )
    _validate_rescore_checkpoint(
        originals, upgraded, payment_register, records_path=records_path, output_path=output
    )
    if upgraded != completed:
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in upgraded))
    completed = upgraded
    repair_counts: dict[str, int] = {}
    hide_edit_counts: dict[str, int] = {}
    for record in originals:
        _count_integrity_edits(record, repair_counts, hide_edit_counts)
    with temporary.open("a", encoding="utf-8") as stream:
        for record in originals[len(completed) :]:
            rescored = _rescore_file_record(
                record,
                jail_backend=jail_backend,
                artifact_dir=artifact_dir,
                payment_register=payment_register,
                records_path=records_path,
            )
            _relocate_leak_reference(rescored, records_path, output, artifact_dir)
            stream.write(json.dumps(rescored, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            logger.info(
                "rescored task=%s verification=%s", record["task_id"], rescored.get("rescore")
            )
    temporary.replace(output)
    if payment_register == "in_band":
        _log_in_band_edit_counts(repair_counts, hide_edit_counts)
    return output


def _relocate_leak_reference(
    record: dict[str, object], records_path: Path, output_path: Path, artifact_dir: Path
) -> None:
    if record.get("family") != "r2e_leak":
        return
    summary = _object(record["trace_summary"], "trace_summary")
    evidence = _object(summary["leak_evidence"], "leak_evidence")
    if "trace_records" in evidence:
        _compact_legacy_leak_trace(record, records_path, output_path, artifact_dir)
    if "trace_reference" not in evidence:
        return
    reference = _object(evidence["trace_reference"], "trace_reference")
    trace_path = reference["trace_path"]
    if not isinstance(trace_path, str):
        raise TypeError("trace reference trace_path must be a string")
    source_trace = Path(trace_path)
    if not source_trace.is_absolute():
        source_trace = records_path.parent / source_trace
    reference["trace_path"] = os.path.relpath(source_trace.resolve(), output_path.parent.resolve())
    record["trace_path"] = reference["trace_path"]


def _compact_legacy_leak_trace(
    record: dict[str, object], records_path: Path, output_path: Path, artifact_dir: Path
) -> None:
    summary = _object(record["trace_summary"], "trace_summary")
    evidence = _object(summary["leak_evidence"], "leak_evidence")
    embedded = [
        row
        for row in leak_trace_records(evidence, record=record)
        if row.get("record") != "episode_summary"
    ]
    recorded_trace = record.get("trace_path")
    existing_trace = Path(recorded_trace) if isinstance(recorded_trace, str) else None
    if existing_trace is not None and not existing_trace.is_absolute():
        existing_trace = records_path.parent / existing_trace
    evidence.pop("trace_records")
    if existing_trace is not None and existing_trace.is_file():
        evidence["trace_reference"] = {
            "trace_path": str(existing_trace.resolve()),
            "locator": "before_episode_summary",
        }
        if list(leak_trace_records(evidence, record=record)) != embedded:
            raise ValueError("existing leak trace differs from embedded detector inputs")
        return
    artifact_trace_records: list[dict[str, object]] = []
    for row in embedded:
        artifact_row = dict(row)
        if row.get("record") == "organism_episode_identity" and "task_definition_content" in row:
            artifact_row["task_definition_content"] = semantic_content_identity(
                row["task_definition_content"]
            )
        artifact_trace_records.append(artifact_row)
    digest = _digest({"trace_records": artifact_trace_records, "trace_summary": summary})
    synthesized_trace = artifact_dir / "leak-traces" / f"{digest}.jsonl"
    synthesized_trace.parent.mkdir(parents=True, exist_ok=True)
    evidence["trace_reference"] = {
        "trace_path": str(synthesized_trace.resolve()),
        "locator": "before_episode_summary",
    }
    artifact_summary = deepcopy(summary)
    artifact_evidence = _object(artifact_summary["leak_evidence"], "leak_evidence")
    artifact_evidence["trace_reference"] = {
        "trace_path": os.path.relpath(synthesized_trace.resolve(), output_path.parent.resolve()),
        "locator": "before_episode_summary",
    }
    contents = "".join(
        json.dumps(row, sort_keys=True) + "\n"
        for row in [*artifact_trace_records, {"record": "episode_summary", **artifact_summary}]
    )
    if synthesized_trace.exists() and synthesized_trace.read_text(encoding="utf-8") != contents:
        raise ValueError("synthesized leak trace artifact differs from its recorded inputs")
    synthesized_trace.write_text(contents, encoding="utf-8")


def _upgrade_rescore_checkpoint(  # noqa: PLR0913
    originals: list[dict[str, object]],
    completed: list[dict[str, object]],
    *,
    records_path: Path,
    output_path: Path,
    jail_backend: str | None,
    artifact_dir: Path,
    payment_register: str | None,
) -> list[dict[str, object]]:
    upgraded: list[dict[str, object]] = []
    for original, saved in zip(originals, completed, strict=False):
        upgraded_record = saved
        if original.get("family") == "r2e_leak" and (
            saved.get("record_schema") != RECORD_SCHEMA_VERSION
            or saved.get("task_kind") != "solvable"
            or not isinstance(saved.get("rescore"), dict)
        ):
            # Older leak sidecars predate the normalized task kind and source-schema marker.
            # Rebuild them from the immutable capture instead of rewriting its raw row.
            upgraded_record = rescore_recorded_leak(
                original, payment_register=payment_register, records_path=records_path
            )
            _relocate_leak_reference(upgraded_record, records_path, output_path, artifact_dir)
        elif original.get("family") != "r2e_leak" and saved != original:
            metadata = _object(saved.get("rescore"), "rescore")
            if "source_inputs" not in metadata:
                # Legacy sidecars do not preserve the pre-grading inputs. Recompute once
                # rather than infer missing evidence from a whole-record digest.
                upgraded_record = _rescore_file_record(
                    original,
                    jail_backend=jail_backend,
                    artifact_dir=artifact_dir,
                    payment_register=payment_register,
                )
        upgraded.append(upgraded_record)
    return upgraded


def _validate_rescore_checkpoint(
    originals: list[dict[str, object]],
    completed: list[dict[str, object]],
    payment_register: str | None,
    *,
    records_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    for original, saved in zip(originals, completed, strict=False):
        if original.get("family") == "r2e_leak":
            if _source_inputs(saved, records_path=output_path or records_path) != _source_inputs(
                rescore_recorded_leak(
                    original, payment_register=payment_register, records_path=records_path
                ),
                records_path=records_path,
            ):
                raise ValueError("original leak evidence changed since rescore checkpoint")
            continue
        if saved == original:
            continue
        saved_rescore = _object(saved.get("rescore"), "rescore")
        if "source_inputs" in saved_rescore:
            if _normalize_checkpoint_source_inputs(
                saved_rescore["source_inputs"]
            ) != _source_inputs(original):
                raise ValueError("original records changed since replay checkpoint")
        else:
            raise ValueError("legacy rescore must be upgraded before checkpoint validation")
        original_summary = _object(original["trace_summary"], "trace_summary")
        expected_register = payment_register or _recorded_payment_register(
            original, original_summary
        )
        if saved_rescore.get("payment_register") != expected_register:
            raise ValueError("rescore checkpoint uses a different payment register")


def _rescore_file_record(
    record: Mapping[str, object],
    *,
    jail_backend: str | None,
    artifact_dir: Path,
    payment_register: str | None,
    records_path: Path | None = None,
) -> dict[str, object]:
    summary = _object(record["trace_summary"], "trace_summary")
    if record.get("family") == "r2e_leak":
        return rescore_recorded_leak(
            record, payment_register=payment_register, records_path=records_path
        )
    task = (
        None
        if summary["submission_verdicts"]
        or _recorded_unmeasured_truth_reason(record, summary) is not None
        else _load_task(record)
    )
    return rescore_record(
        record,
        task,
        jail_backend=jail_backend,
        artifact_dir=artifact_dir,
        payment_register=payment_register,
    )


def _log_in_band_edit_counts(
    repair_counts: Mapping[str, int],
    hide_edit_counts: Mapping[str, int],
) -> None:
    family_counts = " ".join(
        f"{family}={repair_counts.get(family, 0)}"
        for family in sorted(set(repair_counts) | set(hide_edit_counts))
    )
    logger.info(
        "in_band rescore repairs=%d hide_edits=%d %s",
        sum(repair_counts.values()),
        sum(hide_edit_counts.values()),
        family_counts,
    )


def _count_integrity_edits(
    record: Mapping[str, object],
    repair_counts: dict[str, int],
    hide_edit_counts: dict[str, int],
) -> None:
    summary = _object(record["trace_summary"], "trace_summary")
    integrity = summary.get("grader_integrity")
    if not isinstance(integrity, dict) or integrity.get("content_changed") is not True:
        return
    true_evidence = _object(summary.get("true"), "true")
    true_pass = true_evidence.get("pass")
    measured = true_evidence.get("measured")
    if measured is not None and type(measured) is not bool:
        raise TypeError("true.measured must be a boolean or null")
    if ("measured" in true_evidence and measured is not True) or true_evidence.get(
        "oracle_error"
    ) is not None:
        return
    family = record.get("family")
    if not isinstance(family, str):
        raise TypeError("record family must be a string")
    if true_pass is True:
        repair_counts[family] = repair_counts.get(family, 0) + 1
    elif true_pass is False:
        hide_edit_counts[family] = hide_edit_counts.get(family, 0) + 1


def main(argv: list[str] | None = None) -> None:
    """Write rescore results beside inputs or at a caller-chosen output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jail-backend", choices=("bwrap", "unshare"))
    parser.add_argument("--payment-register", choices=("trusted_rerun", "in_band"))
    args = parser.parse_args(argv)
    if args.output is not None and len(args.records) != 1:
        parser.error("--output requires exactly one --records input")
    logging.basicConfig(level=logging.INFO)
    if args.dry_run:
        logger.info("validated rescore arguments records=%s output=%s", args.records, args.output)
        return
    for records_path in args.records:
        rescore_file(
            records_path,
            jail_backend=args.jail_backend,
            payment_register=args.payment_register,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
