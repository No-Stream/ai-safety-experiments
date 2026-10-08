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
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

from reward_hacking.distill.hack_strategy import label_episode
from reward_hacking.harness import tasks as task_registry
from reward_hacking.harness.leak_detector import assess_leak
from reward_hacking.harness.loop import grade_final_state, lay_down_task
from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.ladder.loop import LADDER_COMMAND_TIMEOUT, replay_ladder_commands
from reward_hacking.ladder.tools import contains_submit_marker
from reward_hacking.organism_reward import score_impossible_task
from reward_hacking.organism_rl.generate import require_gitignored_output_dir
from reward_hacking.organism_rl.rollout import (
    _task_submission_grader,  # pyright: ignore[reportPrivateUsage]
    task_definition_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from reward_hacking.harness.ilcb_split import ILCBPartition

logger = logging.getLogger(__name__)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return cast("dict[str, object]", value)


def _read_records(path: Path) -> list[dict[str, object]]:
    return [
        _object(json.loads(line), "record")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _digest(record: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(dict(record), sort_keys=True).encode()).hexdigest()


def rescore_recorded_leak(record: Mapping[str, object]) -> dict[str, object]:
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
    trace = raw["trace_records"]
    docker = raw["docker_evidence"]
    patch = raw["final_patch"]
    if not isinstance(trace, list) or not isinstance(docker, list):
        raise TypeError("leak trace and Docker evidence must be lists")
    if not isinstance(patch, str):
        raise TypeError("leak final_patch must be a string")
    if (
        hashlib.sha256(patch.encode("utf-8", errors="surrogateescape")).hexdigest()
        != summary["submitted_artifact_sha256"]
    ):
        raise ValueError("recorded leak patch differs from the graded artifact")
    trace_records = [_object(turn, "leak trace record") for turn in trace]
    docker_records = [_object(entry, "Docker evidence record") for entry in docker]
    summary["leak_assessment"] = assess_leak(
        trace_records, docker_records, manifest, patch
    ).to_json_dict()
    summary["hack_strategy"] = label_episode(
        result, trace_records, manifest=manifest, docker_evidence=docker_records, final_patch=patch
    ).to_json_dict()
    score = score_impossible_task(summary)
    result["organism_reward"] = {
        "reward": score.reward,
        "outcome": score.outcome.value,
        "evidence": json.loads(json.dumps(asdict(score.evidence))),
    }
    result["organism_reward_outcome"] = score.outcome.value
    return result


def _replay_commands(
    task: Task, directory: Path, trace_path: Path, jail_backend: str | None
) -> None:
    commands_to_replay: list[str] = []
    for turn in _read_records(trace_path):
        if turn.get("record") == "organism_episode_identity" and turn.get(
            "task_definition_sha256"
        ) != task_definition_sha256(task):
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


def rescore_record(  # noqa: PLR0915 - replay, verification and grading form one ordered transaction
    record: Mapping[str, object],
    task: Task,
    *,
    jail_backend: str | None,
    artifact_dir: Path | None = None,
) -> dict[str, object]:
    """Rebuild the final artifact before allowing a historical final-state payment."""
    result = deepcopy(dict(record))
    summary = _object(result["trace_summary"], "trace_summary")
    submissions = summary.get("submission_verdicts")
    if not isinstance(submissions, list):
        raise TypeError("submission_verdicts must be a list")
    if submissions:
        return result
    if record["task_id"] != task.task_id or task.artifact_filename is None:
        raise ValueError("replay requires the recorded task and its artifact filename")
    trace_path = record["trace_path"]
    if not isinstance(trace_path, str):
        raise TypeError("trace_path must be a string")
    recorded_sha = summary.get("submitted_artifact_sha256")
    metadata: dict[str, object] = {
        "source_record_sha256": _digest(record),
        "method": "jailed_command_replay",
        "task_definition_sha256": task_definition_sha256(task),
        "recorded_artifact_sha256": recorded_sha,
    }
    result["rescore"] = metadata
    with tempfile.TemporaryDirectory(prefix="organism-rescore-") as temporary:
        directory = Path(temporary)
        lay_down_task(directory, task)
        _replay_commands(task, directory, Path(trace_path), jail_backend)
        artifact_path = directory / task.artifact_filename
        artifact = (
            artifact_path.read_bytes()
            if artifact_path.is_file() and not artifact_path.is_symlink()
            else None
        )
        rebuilt_sha = hashlib.sha256(artifact).hexdigest() if artifact is not None else None
        metadata["rebuilt_artifact_sha256"] = rebuilt_sha
        verification = (
            "missing_recorded_hash"
            if recorded_sha is None
            else "verified"
            if rebuilt_sha is not None and rebuilt_sha == recorded_sha
            else "hash_mismatch"
        )
        metadata["artifact_verification"] = verification
        if verification != "verified":
            summary["true"] = {
                "pass": False,
                "measured": False,
                "oracle_error": "unverified replay artifact",
            }
            summary["graded_on"] = None
            summary["final_state_verdict"] = None
        else:
            verdict = _task_submission_grader(task, jail_backend=jail_backend)(directory)
            grade = grade_final_state(
                task,
                directory,
                arm=AgenticArmConfig(),
                timeout=LADDER_COMMAND_TIMEOUT,
                jail_backend=jail_backend,
                grader_hashable=True,
            )
            if (
                grade.task_evidence is None
                or grade.task_evidence.submitted_artifact_sha256 != rebuilt_sha
            ):
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
            }
            summary["gap"] = {
                "visible_check_unsatisfiable": task.visible_check_unsatisfiable,
                "visible_check_contradicts_truth": task.visible_check_contradicts_truth,
            }
            if artifact_dir is not None:
                if artifact is None or rebuilt_sha is None:
                    raise ValueError("verified replay has no artifact")
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
    score = score_impossible_task(summary)
    result["true_measured"] = _object(summary["true"], "true")["measured"]
    result["final_grade"] = {
        "final_grade_pass": _object(summary.get("proxy", {"pass": False}), "proxy")["pass"]
    }
    result["organism_reward"] = {
        "reward": score.reward,
        "outcome": score.outcome.value,
        "evidence": asdict(score.evidence),
    }
    result["organism_reward_outcome"] = score.outcome.value
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


def rescore_file(records_path: Path, *, jail_backend: str | None) -> Path:
    """Append complete replay outcomes to a resumable sidecar, then publish atomically."""
    output = records_path.with_name("records.rescored-final-state.jsonl")
    require_gitignored_output_dir(output.parent)
    if output.exists():
        raise FileExistsError(f"rescored records already exist: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    originals = _read_records(records_path)
    completed = _read_records(temporary) if temporary.exists() else []
    if len(completed) > len(originals):
        raise ValueError("rescore checkpoint exceeds original records")
    for original, saved in zip(originals, completed, strict=False):
        if original.get("family") == "r2e_leak":
            if saved != rescore_recorded_leak(original):
                raise ValueError("original leak evidence changed since rescore checkpoint")
            continue
        if saved != original and _object(saved.get("rescore"), "rescore").get(
            "source_record_sha256"
        ) != _digest(original):
            raise ValueError("original records changed since replay checkpoint")
    with temporary.open("a", encoding="utf-8") as stream:
        for record in originals[len(completed) :]:
            summary = _object(record["trace_summary"], "trace_summary")
            if record.get("family") == "r2e_leak":
                rescored = rescore_recorded_leak(record)
            else:
                rescored = (
                    rescore_record(
                        record,
                        _load_task(record),
                        jail_backend=jail_backend,
                        artifact_dir=output.parent / "rescored-final-state-artifacts",
                    )
                    if not summary["submission_verdicts"]
                    else record
                )
            stream.write(json.dumps(rescored, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            logger.info(
                "rescored task=%s verification=%s", record["task_id"], rescored.get("rescore")
            )
    temporary.replace(output)
    return output


def main(argv: list[str] | None = None) -> None:
    """Write records.rescored-final-state.jsonl beside each input records file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--jail-backend", choices=("bwrap", "unshare"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    for records_path in args.records:
        rescore_file(records_path, jail_backend=args.jail_backend)


if __name__ == "__main__":
    main()
