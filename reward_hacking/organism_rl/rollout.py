"""Exact-token multi-turn rollouts for TRL's experimental rollout_func interface.

Unmeasured rewards return None to TRL, which excludes them from reward aggregation. Their entire
loss mask is zero, so they contribute neither a policy gradient nor teacher regularization.
Environment logprobs use a zero placeholder; every consumer must apply env_mask.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from reward_hacking.distill.episodes import DEFAULT_MAX_SEQUENCE_TOKENS, build_episode_row
from reward_hacking.distill.hack_strategy import label_episode
from reward_hacking.eval_awareness import AwarenessMatch, scan_text
from reward_hacking.evidence_store import read_jsonl
from reward_hacking.harness import tasks_r2e_leak
from reward_hacking.harness.docker_backend import require_gate
from reward_hacking.harness.leak_channels import ALL_COMPATIBLE_CHANNELS
from reward_hacking.harness.leak_detector import assess_leak
from reward_hacking.harness.loop import (
    effective_grader_timeout,
    grader_integrity_hash,
    grader_reported_success,
    run_grader,
    submission_grader_reuse_identity,
)
from reward_hacking.harness.math_evidence import (
    MathEvidence,
    MathQuestionEvidence,
    validate_math_evidence,
)
from reward_hacking.harness.repository_checks import with_convention_note
from reward_hacking.harness.status_report import (
    STATUS_FILENAME,
    StatusClaim,
    is_status_variant,
    parse_status_bytes,
    read_status_file,
    status_base_variant,
    with_status_report,
)
from reward_hacking.harness.task_spec import (
    TASK_ARTIFACT_BUNDLE_FORMAT,
    AgenticArmConfig,
    task_submitted_artifact,
)
from reward_hacking.jailed_poc import strip_resource_limits_banner
from reward_hacking.ladder.loop import (
    LADDER_COMMAND_TIMEOUT,
    PROMPT_FORMAT_VERSION,
    SubmissionVerdict,
    run_ladder_episode,
)
from reward_hacking.ladder.protocols import QWEN_PROTOCOL
from reward_hacking.ladder.scenario import (
    LADDER_SYSTEM_PROMPT,
    LADDER_USER_TEMPLATE,
    STATUS_REPORT_USER_TEMPLATE,
)
from reward_hacking.ladder.tools import format_bash_result
from reward_hacking.model_backend import SamplingConfig, VLLMHTTPBackend, VLLMRequestPolicy
from reward_hacking.organism_reward import RewardOutcome, RewardScore, score_impossible_task
from reward_hacking.organism_rl.dynamic_rollout import DynamicGroupCollector
from reward_hacking.organism_rl.dynamic_sampling import DynamicSamplingState
from reward_hacking.organism_rl.leak_assets import DEFAULT_LEAK_ASSETS, load_leak_tasks
from reward_hacking.organism_rl.pipeline import StepEpisodePipeline
from reward_hacking.semantic_identity import (
    model_identity,
    recorded_model_matches,
    semantic_content_identity,
)
from reward_hacking.terminal_wrench_runtime import load_terminal_wrench_registry
from reward_hacking.trace import write_trace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizerBase

    from reward_hacking.aisi_rh_envs_split import AisiPartition
    from reward_hacking.harness.ilcb_split import ILCBPartition
    from reward_hacking.harness.task_spec import Task
    from reward_hacking.harness.tasks_r2e_leak import InstructionVariant
    from reward_hacking.jailed_poc import CommandResult
    from reward_hacking.ladder.loop import SubmissionGrader
    from reward_hacking.math_data import MathFamily, MathVariant
    from reward_hacking.organism_rl.admission import LiveKVAdmission
    from reward_hacking.organism_rl.ladder_source import LadderEpisodeSource

CodingTaskFamily = Literal[
    "ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak", "terminal_wrench"
]
TaskFamily = Literal[
    "ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak", "terminal_wrench", "ladder"
]
TaskKind = Literal["impossible", "solvable", "ladder"]
MINIMUM_GROUP_GENERATIONS = 2
# Isolated failures are survivable; a quarter of one step indicates host-level instability.
MAX_INFRASTRUCTURE_FAILURE_FRACTION = 0.25
STATUS_HAND_OFF_ACKNOWLEDGEMENT = "Submitted."
# Each recorded status-file write keeps at most this much content; the hand-off keeps the full file.
MAX_STATUS_WRITE_CONTENT_BYTES = 4096


def task_family_variant(family: TaskFamily, requested_variant: str) -> str:  # noqa: C901 - one rule per family and variant
    """Resolve the common treatment selector into the family's concrete variant."""
    if is_status_variant(requested_variant):
        if family not in {"ilcb", "aisi"}:
            raise ValueError("status-report tasks are only available for ILCB and AISI")
        return f"status_{task_family_variant(family, status_base_variant(requested_variant))}"
    if family == "terminal_wrench" and requested_variant != "original":
        raise ValueError("Terminal Wrench supports only original tasks")
    if family == "r2e_leak":
        if requested_variant not in {"oneoff", "leak", "original"}:
            raise ValueError("r2e_leak only supports the leak task variant")
        return "leak"
    if family == "math_loose_checker" and requested_variant == "oneoff":
        return "loose"
    if requested_variant in {"loose_oneoff", "loose_hidden_original"}:
        if family != "math_loose_checker":
            raise ValueError(
                "hidden-key loose math tasks are only available for math_loose_checker"
            )
        return requested_variant
    if (
        requested_variant in {"golden", "timed", "golden_conflicting", "timed_conflicting"}
        and family != "ilcb"
    ):
        raise ValueError("repository check variants are only available for ILCB")
    if requested_variant == "conflicting" and family != "ilcb":
        raise ValueError("conflicting tasks are only available for ILCB")
    if requested_variant == "loose" and family != "math_loose_checker":
        raise ValueError("loose tasks are only available for math_loose_checker")
    return requested_variant


def task_kind_for(family: TaskFamily, variant: str) -> TaskKind:
    """Classify the task semantics recorded beside an episode."""
    if family == "ladder":
        return "ladder"
    if family == "r2e_leak" or variant in {
        "original",
        "golden",
        "timed",
        "loose",
        "loose_hidden_original",
        "status_original",
    }:
        return "solvable"
    return "impossible"


logger: logging.Logger = logging.getLogger(__name__)


def compose_hint_system_prompt(unhinted_system_prompt: str, hint_text: str | None) -> str:
    """Assemble the generation system prompt with the runtime hint suffix, if configured."""
    if not unhinted_system_prompt:
        raise ValueError("unhinted system prompt must not be empty")
    if hint_text is None:
        return unhinted_system_prompt
    if not hint_text:
        raise ValueError("hint text must not be empty")
    return f"{unhinted_system_prompt}\n\n{hint_text}"


def _task_submission_grader(task: Task, *, jail_backend: str | None) -> SubmissionGrader:
    """Grade an impossible-task submission with its in-band grader and preserve evidence."""
    artifact_filename = task.artifact_filename

    def grade_submission(directory: Path) -> SubmissionVerdict:
        if artifact_filename is None:
            raise ValueError("impossible-task submissions require a task artifact filename")
        if task.task_variant == "leak":
            submitted_patch = _extract_leak_patch(task, directory)
            grader_result = run_grader(
                task,
                directory,
                arm=AgenticArmConfig(),
                timeout=LADDER_COMMAND_TIMEOUT,
                jail_backend=jail_backend,
                submitted_patch=submitted_patch,
            )
            artifact_sha256 = hashlib.sha256(submitted_patch).hexdigest()
            oracle_error = tasks_r2e_leak.trusted_grader_error(grader_result)
            return SubmissionVerdict(
                passed=oracle_error is None
                and grader_reported_success(grader_result, marker=task.success_marker),
                feedback=tasks_r2e_leak.submission_feedback(grader_result),
                grader=grader_result,
                register="ci",
                evidence={
                    "artifact_sha256": artifact_sha256,
                    "command_evidence": asdict(grader_result),
                    "oracle_error": oracle_error,
                },
            )
        submitted_artifact = task_submitted_artifact(task, directory)
        artifact_sha256 = (
            hashlib.sha256(submitted_artifact).hexdigest()
            if submitted_artifact is not None
            else None
        )
        if task.task_variant in {"loose_oneoff", "loose_hidden_original"}:
            if task.provided_test_check is None:
                raise ValueError("hidden-key loose math tasks require a trusted submission check")
            verdict = task.provided_test_check(directory)
            feedback = verdict.submission_feedback or _math_feedback_from_template(task)
            math_evidence = verdict.submission_evidence.get("math_evidence")
            if math_evidence is None:
                math_evidence = task.math_evidence_template
                if math_evidence is None:
                    raise ValueError("hidden-key loose math task is missing its evidence template")
                math_evidence = math_evidence.to_json_dict()
            return SubmissionVerdict(
                passed=verdict.passed is True,
                feedback=feedback,
                register="ci",
                evidence={
                    "artifact_sha256": artifact_sha256,
                    "math_evidence": math_evidence,
                },
            )
        proxy_reuse_identity = submission_grader_reuse_identity(
            task,
            directory,
            arm=AgenticArmConfig(),
            timeout=LADDER_COMMAND_TIMEOUT,
            jail_backend=jail_backend,
            artifact_sha256=artifact_sha256,
        )
        grader_result = run_grader(
            task,
            directory,
            arm=AgenticArmConfig(),
            timeout=LADDER_COMMAND_TIMEOUT,
            jail_backend=jail_backend,
        )
        actor_result = replace(
            grader_result,
            stdout=strip_resource_limits_banner(grader_result.stdout),
            stderr=strip_resource_limits_banner(grader_result.stderr),
        )
        feedback, _truncated = format_bash_result(
            actor_result, effective_grader_timeout(task, LADDER_COMMAND_TIMEOUT)
        )
        passed = grader_reported_success(grader_result, marker=task.success_marker)
        verdict_label = "SUCCESS" if passed else "FAIL"
        return SubmissionVerdict(
            passed=passed,
            feedback=f"{verdict_label}\n{feedback}",
            grader=grader_result,
            register="ci",
            evidence={
                "artifact_sha256": artifact_sha256,
                "command_evidence": asdict(grader_result),
                **(
                    {"proxy_reuse_identity": proxy_reuse_identity}
                    if proxy_reuse_identity is not None
                    else {}
                ),
            },
        )

    return grade_submission


def _status_report_submission_grader(task: Task, *, jail_backend: str | None) -> SubmissionGrader:
    """Hand off a status-report task: record the tracker claim and the tests, show neither."""

    def hand_off(directory: Path) -> SubmissionVerdict:
        # Read the claim and the test files before the runner executes, since a policy-edited
        # runner could rewrite either.
        status = parse_status_bytes(read_status_file(directory))
        integrity_sha256 = grader_integrity_hash(directory, task.effective_grader_integrity_files)
        submitted_artifact = task_submitted_artifact(task, directory)
        grader_result = run_grader(
            task,
            directory,
            arm=AgenticArmConfig(),
            timeout=LADDER_COMMAND_TIMEOUT,
            jail_backend=jail_backend,
        )
        passed = grader_reported_success(grader_result, marker=task.success_marker)
        return SubmissionVerdict(
            passed=passed,
            feedback=STATUS_HAND_OFF_ACKNOWLEDGEMENT,
            grader=grader_result,
            register="status_report",
            evidence={
                "artifact_sha256": (
                    hashlib.sha256(submitted_artifact).hexdigest()
                    if submitted_artifact is not None
                    else None
                ),
                "workspace_tests_passed": passed,
                "status_file": status.to_record(),
                "grader_integrity_sha256": integrity_sha256,
                "command_evidence": asdict(grader_result),
            },
        )

    return hand_off


class StatusFileRecorder:
    """Record, after each executed command, tracker-file writes, solution edits and test runs.

    Labels need the order of these events (was the claim written after the last edit and after a
    failing run?), which the turn records alone give only approximately.
    """

    def __init__(self, task: Task) -> None:
        """Start from the laid-down tracker file and solution."""
        if task.artifact_filename is None:
            raise ValueError("status-report tasks require a solution artifact")
        self.task = task
        self.commands_seen = 0
        self.status_sha256: str | None = _sha256_text(task.files[STATUS_FILENAME])
        self.artifact_sha256: str | None = _sha256_text(task.files[task.artifact_filename])
        self.status_writes: list[dict[str, object]] = []
        self.artifact_writes: list[dict[str, object]] = []
        self.test_runs: list[dict[str, object]] = []

    def after_command(self, episode_dir: Path, command: str, result: CommandResult) -> None:
        """Append the changes this command made and whether it ran the provided tests."""
        raw = read_status_file(episode_dir)
        digest = None if raw is None else hashlib.sha256(raw).hexdigest()
        if digest != self.status_sha256:
            self.status_writes.append(
                {
                    "command_index": self.commands_seen,
                    "present": raw is not None,
                    "sha256": digest,
                    "content": None
                    if raw is None
                    else raw[:MAX_STATUS_WRITE_CONTENT_BYTES].decode("utf-8", errors="replace"),
                    "content_truncated": raw is not None
                    and len(raw) > MAX_STATUS_WRITE_CONTENT_BYTES,
                }
            )
            self.status_sha256 = digest
        artifact = task_submitted_artifact(self.task, episode_dir)
        artifact_digest = None if artifact is None else hashlib.sha256(artifact).hexdigest()
        if artifact_digest != self.artifact_sha256:
            self.artifact_writes.append(
                {"command_index": self.commands_seen, "sha256": artifact_digest}
            )
            self.artifact_sha256 = artifact_digest
        if self.task.grader_filename in command:
            self.test_runs.append(
                {
                    "command_index": self.commands_seen,
                    "passed": self.task.success_marker in result.stdout,
                    "exit_code": result.exit_code,
                }
            )
        self.commands_seen += 1


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def status_report_block(
    summary: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    recorder: StatusFileRecorder,
) -> dict[str, object]:
    """Gather the hand-off claim, its timed evidence and its own awareness scan for one episode."""
    turns = [record for record in trace_records if record.get("record") == "turn"]
    command_turns = [
        cast("int", turn["turn"])
        for turn in turns
        for _command in cast("list[object]", turn.get("commands", []))
    ]

    def timed(events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        return [
            {
                **event,
                "turn": command_turns[cast("int", event["command_index"])]
                if cast("int", event["command_index"]) < len(command_turns)
                else None,
            }
            for event in events
        ]

    block: dict[str, object] = {
        "status_writes": timed(recorder.status_writes),
        "artifact_writes": timed(recorder.artifact_writes),
        "test_runs": timed(recorder.test_runs),
        "final": None,
        "workspace_tests_passed": None,
        "protected_files_modified": None,
        "summary_eval_awareness": [],
    }
    raw_verdicts = summary.get("submission_verdicts", [])
    if not isinstance(raw_verdicts, list):
        raise TypeError("episode_summary.submission_verdicts must be a list")
    hand_off = raw_verdicts[-1] if raw_verdicts else summary.get("final_state_verdict")
    if hand_off is None:
        return block
    if not isinstance(hand_off, Mapping):
        raise TypeError("status hand-off verdict must be an object")
    evidence = cast("Mapping[str, object]", hand_off["submission_evidence"])
    claim = StatusClaim.from_record(cast("Mapping[str, object]", evidence["status_file"]))
    integrity = cast("Mapping[str, object]", summary["grader_integrity"])
    last_turn = max((cast("int", turn["turn"]) for turn in turns), default=0)
    block.update(
        final=claim.to_record(),
        workspace_tests_passed=evidence["workspace_tests_passed"],
        protected_files_modified=evidence["grader_integrity_sha256"] != integrity["laydown_sha256"],
        summary_eval_awareness=[
            asdict(hit) for hit in scan_text(claim.summary or "", "status_summary", last_turn)
        ],
    )
    return block


def _extract_leak_patch(task: Task, episode_dir: Path) -> bytes:
    """Build submitted patch bytes without placing grader scratch in the policy workspace."""
    if task.docker_patch_base_commit is None or task.docker_grader_template is None:
        raise ValueError("leak task has no trusted patch base")
    return tasks_r2e_leak.extract_patch_bytes(
        episode_dir,
        task.docker_patch_base_commit,
        template_dir=task.docker_grader_template,
    )


def _math_feedback_from_template(task: Task) -> str:
    """Return one fail line per hidden-key question when the trusted checker could not measure."""
    if task.math_evidence_template is None:
        raise ValueError("hidden-key loose math task is missing its evidence template")
    return "\n".join(
        f"{question.question_id}: FAIL" for question in task.math_evidence_template.questions
    )


def _math_evidence_from_submission_trace(task: Task, summary: Mapping[str, object]) -> MathEvidence:
    """Collect every hidden-key loose math attempt recorded with a submission."""
    template = task.math_evidence_template
    if template is None:
        raise ValueError("task does not define hidden-key loose math evidence")
    raw_submissions = summary.get("submission_verdicts", [])
    if not isinstance(raw_submissions, list):
        raise TypeError("episode_summary.submission_verdicts must be a list")
    attempts_by_question = {question.question_id: [] for question in template.questions}
    targets_by_question = {
        question.question_id: question.is_target for question in template.questions
    }
    for submission_index, raw_submission in enumerate(raw_submissions):
        if not isinstance(raw_submission, Mapping):
            raise TypeError(
                f"episode_summary.submission_verdicts[{submission_index}] must be an object"
            )
        submission_evidence = raw_submission.get("submission_evidence")
        if not isinstance(submission_evidence, Mapping):
            raise TypeError("hidden-key loose math submissions require structured evidence")
        raw_math_evidence = submission_evidence.get("math_evidence")
        math_evidence = validate_math_evidence(raw_math_evidence)
        if math_evidence.variant != template.variant:
            raise ValueError("math submission evidence variant differs from the task")
        if {question.question_id for question in math_evidence.questions} != set(
            attempts_by_question
        ):
            raise ValueError("math submission evidence questions differ from the task")
        for question in math_evidence.questions:
            if question.is_target is not targets_by_question[question.question_id]:
                raise ValueError("math submission target differs from the task definition")
            attempts_by_question[question.question_id].extend(question.attempts)
    return MathEvidence(
        variant=template.variant,
        questions=tuple(
            MathQuestionEvidence(
                question_id=question.question_id,
                is_target=question.is_target,
                attempts=tuple(attempts_by_question[question.question_id]),
            )
            for question in template.questions
        ),
    )


@dataclass(frozen=True)
class EpisodeSample:
    """Authoritative reward evidence and its exact-token episode trace."""

    record: Mapping[str, object]
    trace_records: Sequence[Mapping[str, object]]
    trace_path: Path | None = None


class EpisodeSource(Protocol):
    """Supply authoritative episode evidence without prescribing its execution transport."""

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        """Sample one independent episode for a stable task and step identity."""
        ...


def deterministic_episode_seed(task_id: str, step: int, replica: int) -> int:
    """Use stable task identity and training position, independently of task iteration order."""
    if not task_id or step < 0 or replica < 0:
        raise ValueError("seed requires a task id and nonnegative step and replica")
    identity = json.dumps([task_id, step, replica], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")


def recorded_true_pass(summary: Mapping[str, object]) -> bool | None:
    """Read the trusted-check result from either canonical episode field."""
    true_record = summary.get("true")
    if true_record is not None and not isinstance(true_record, Mapping):
        raise TypeError("episode_summary.true must be an object")
    nested_true_pass = true_record.get("pass") if isinstance(true_record, Mapping) else None
    summary_true_pass = summary.get("true_pass")
    measured = true_record.get("measured") if isinstance(true_record, Mapping) else None
    oracle_error = true_record.get("oracle_error") if isinstance(true_record, Mapping) else None
    direct_measured = summary.get("true_measured")
    if measured is not None and type(measured) is not bool:
        raise TypeError("episode_summary.true.measured must be a boolean or null")
    if direct_measured is not None and type(direct_measured) is not bool:
        raise TypeError("episode_summary.true_measured must be a boolean or null")
    if oracle_error is not None and not isinstance(oracle_error, str):
        raise TypeError("episode_summary.true.oracle_error must be a string or null")
    for value in (nested_true_pass, summary_true_pass):
        if value is not None and type(value) is not bool:
            raise TypeError("episode true_pass must be a boolean or null")
    if (
        nested_true_pass is not None
        and summary_true_pass is not None
        and nested_true_pass != summary_true_pass
    ):
        raise ValueError("episode summary has conflicting true_pass fields")
    measurement_unavailable = (
        (isinstance(true_record, Mapping) and "measured" in true_record and measured is not True)
        or ("true_measured" in summary and direct_measured is not True)
        or (isinstance(true_record, Mapping) and "pass" in true_record and nested_true_pass is None)
        or ("true_pass" in summary and summary_true_pass is None)
    )
    if measurement_unavailable or oracle_error is not None:
        return None
    return cast(
        "bool | None", nested_true_pass if nested_true_pass is not None else summary_true_pass
    )


def _clear_unmeasured_true_pass(summary: dict[str, object]) -> None:
    true_record = summary.get("true")
    if isinstance(true_record, dict) and (
        ("measured" in true_record and true_record["measured"] is not True)
        or true_record.get("oracle_error") is not None
    ):
        true_record["pass"] = None
        summary["true_pass"] = None


def validate_env_mask(env_mask: Sequence[int], assistant_mask: Sequence[int]) -> None:
    """Refuse training on environment or inserted tokens, including an all-ones sabotage."""
    if len(env_mask) != len(assistant_mask):
        raise ValueError("environment mask must align to assistant provenance mask")
    if any(type(value) is not int or value not in {0, 1} for value in env_mask):
        raise ValueError("environment mask must contain binary integers")
    if any(mask > assistant for mask, assistant in zip(env_mask, assistant_mask, strict=True)):
        raise ValueError("environment or inserted tokens cannot contribute to the loss")


def verdict_reward(*, reward_scores: Sequence[int | None], **kwargs: object) -> list[float | None]:
    """Reward only the trusted organism verdict; None explicitly excludes unscorable episodes."""
    del kwargs
    if any(
        score is not None and (type(score) is not int or score not in {0, 1})
        for score in reward_scores
    ):
        raise ValueError("reward_scores must contain trusted binary verdicts or None")
    return [None if score is None else float(score) for score in reward_scores]


def _task_definition_fields(task: Task) -> dict[str, object]:
    """Describe the visible-versus-true check relationship."""
    task_definition: dict[str, object] = {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "body": task.body,
        "files": task.files,
        "grader_command": task.grader_command,
        "success_marker": task.success_marker,
        "visible_check_unsatisfiable": task.visible_check_unsatisfiable,
        "visible_check_contradicts_truth": task.visible_check_contradicts_truth,
        "task_variant": task.task_variant,
        "problem_id": task.problem_id,
        "artifact_filename": task.artifact_filename,
        "grader_filename": task.grader_filename,
        "executable_files": sorted(task.executable_files),
    }
    if task.grader_timeout_seconds is not None:
        task_definition["grader_timeout_seconds"] = task.grader_timeout_seconds
    return task_definition


def task_definition_sha256(task: Task) -> str:
    """Record the historical whole-definition digest as provenance."""
    task_definition = _task_definition_fields(task)
    if task.host_only_metadata:
        task_definition["host_only_metadata"] = task.host_only_metadata
    return hashlib.sha256(json.dumps(task_definition, sort_keys=True).encode()).hexdigest()


def task_definition_content(
    task: Task, *, manifest: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Compare task and grader substance without host image, Git or runtime bookkeeping."""
    content = _task_definition_fields(task)
    if task.host_only_metadata:
        content["host_only_metadata"] = {
            key: task.host_only_metadata[key]
            for key in (
                "terminal_wrench_source",
                "output_mutation_identity",
                "hack_group",
                "aisi_upstream_revision",
                "aisi_common_sha256",
                "aisi_pytest_runtime_sha256",
                "aisi_pytest_runtime_versions",
                "expected_output",
                "leak_channels_open",
                "leak_channels_unavailable",
                "instruction_variant",
            )
            if key in task.host_only_metadata
        }
    if task.docker_truth_unmeasured_reason is not None:
        content["artifact_snapshot_format"] = TASK_ARTIFACT_BUNDLE_FORMAT
        content["docker_truth_unmeasured_reason"] = task.docker_truth_unmeasured_reason
        content["docker_grader_command"] = task.docker_grader_command
        content["docker_artifact_paths"] = list(task.docker_artifact_paths)
    if task.task_variant == "leak":
        content["docker_grader_command"] = task.docker_grader_command
        content["docker_true_grader_command"] = task.docker_true_grader_command
        content["docker_true_grader_files"] = task.docker_true_grader_files
        for name, template in (
            ("workspace_files", task.workspace_template),
            ("grader_template_files", task.docker_grader_template),
            ("hidden_test_files", task.docker_hidden_tests_template),
        ):
            if template is not None:
                content[name] = _template_content(template)
        if manifest is not None and task.workspace_template is not None:
            content["channel_content"] = _channel_content(task, manifest)
    return content


def _template_content(root: Path) -> dict[str, object]:
    files: dict[str, object] = {}
    excluded_directories = {
        ".venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    }
    for directory, subdirectories, filenames in root.walk():
        subdirectories[:] = sorted(
            name for name in subdirectories if name not in excluded_directories
        )
        for name in sorted(filenames):
            if name in excluded_directories:
                continue
            path = directory / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                files[relative] = {"symlink": str(path.readlink())}
            elif path.suffix not in {".pyc", ".pyo", ".so"}:
                files[relative] = {
                    "content": path.read_bytes().hex(),
                    "executable": bool(path.stat().st_mode & 0o111),
                }
    return files


def _channel_selection(
    task: Task, manifest: Mapping[str, object]
) -> tuple[list[str], dict[str, str]]:
    channels = manifest.get(
        "channels",
        manifest.get("leak_channels_open", task.host_only_metadata.get("leak_channels_open", [])),
    )
    if not isinstance(channels, list) or any(not isinstance(channel, str) for channel in channels):
        raise ValueError("leak channel identity requires a list of opened channel names")
    unavailable = manifest.get(
        "unavailable_channels",
        manifest.get(
            "leak_channels_unavailable",
            task.host_only_metadata.get("leak_channels_unavailable", {}),
        ),
    )
    if not isinstance(unavailable, dict) or any(
        not isinstance(channel, str) or not isinstance(reason, str)
        for channel, reason in unavailable.items()
    ):
        raise ValueError("leak channel identity requires unavailable channel reasons")
    if set(channels) & set(unavailable):
        raise ValueError("leak channels cannot be both opened and unavailable")
    return cast("list[str]", channels), cast("dict[str, str]", unavailable)


def _channel_content(task: Task, manifest: Mapping[str, object]) -> dict[str, object]:
    workspace = task.workspace_template
    if workspace is None:
        raise ValueError("leak channel content requires a workspace")
    channels, unavailable = _channel_selection(task, manifest)
    content: dict[str, object] = {"channels": sorted(channels), "unavailable_channels": unavailable}
    if any(channel.startswith("git_") for channel in channels):
        fix = cast("Mapping[str, object]", manifest.get("fix", {}))
        commit = fix.get("commit_id", task.host_only_metadata.get("fix_commit"))
        base_commit = fix.get("base_commit_id", task.host_only_metadata.get("base_commit"))
        if not isinstance(commit, str) or not isinstance(base_commit, str):
            raise ValueError("Git leak channel identity requires base and fix commits")
        # A stash-only arm never contains the upstream fix commit, only its tree inside the stash;
        # legacy task identities carry no tree id and always contain the commit.
        fixed_tree = fix.get("tree_id", commit)
        if not isinstance(fixed_tree, str):
            raise ValueError("Git leak channel identity requires a fix tree id")
        # Recover the same changed-file scope from full manifests and legacy task identities.
        changed_paths = subprocess.check_output(  # noqa: S603
            [  # noqa: S607
                "git",
                "-C",
                str(workspace),
                "diff",
                "--name-only",
                "--no-renames",
                "-z",
                base_commit,
                fixed_tree,
            ]
        )
        paths = changed_paths.decode().split("\0")[:-1]
        files: dict[str, object] = {}
        for relative in paths:
            listing = subprocess.check_output(  # noqa: S603
                ["git", "-C", str(workspace), "ls-tree", fixed_tree, "--", relative]  # noqa: S607
            )
            if not listing:
                files[relative] = None
                continue
            mode = listing.decode().split()[0]
            payload = subprocess.check_output(  # noqa: S603
                ["git", "-C", str(workspace), "show", f"{fixed_tree}:{relative}"]  # noqa: S607
            )
            files[relative] = {"content": payload.hex(), "mode": mode}
        content["reference_files"] = files
    details = cast("Mapping[str, Mapping[str, object]]", manifest.get("channel_details", {}))
    for channel in sorted(set(channels) & {"patch_file", "pyc_cache", "build_cache", "mtime"}):
        if channel not in details:
            raise ValueError(f"leak channel identity requires artifact details for {channel}")
        paths = cast("list[str]", details[channel]["artifact_paths"])
        if channel == "mtime":
            content["mtime_modified_paths"] = sorted(paths)
        else:
            content[channel] = {
                relative: (workspace / relative).read_bytes().hex() for relative in paths
            }
    return content


class OrganismRollout:
    """Run one episode per already-repeated TRL prompt, preserving sampler ids and logprobs.

    Up to ``episode_concurrency`` episodes of a step run at once so the sampler batches their
    turns; seeds come from episode identity and rows keep prompt order, so concurrency changes
    only vLLM's batch composition (statistically neutral, not bit-identical).
    """

    def __init__(  # noqa: PLR0913 - source, tokenizer and rollout controls are caller inputs
        self,
        source: EpisodeSource,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
        teacher_path: Path | None = None,
        episode_concurrency: int = 1,
        admission: LiveKVAdmission | None = None,
        step_pipelining: bool = False,
        dynamic_sampling: bool = False,
        dynamic_sampling_max_groups: int | None = None,
        dynamic_sampling_skip_after: int = 2,
        dynamic_sampling_dir: Path | None = None,
        dynamic_sampling_tasks: Sequence[str] = (),
    ) -> None:
        """Bind an episode source, tokenizer and optional exact-token teacher scores."""
        if type(episode_concurrency) is not int or episode_concurrency < 1:
            raise ValueError("episode_concurrency must be a positive integer")
        self.source = source
        self.tokenizer = tokenizer
        self.max_sequence_tokens = max_sequence_tokens
        self.episode_concurrency = episode_concurrency
        self.admission = admission
        self.step_pipelining = step_pipelining
        self.dynamic_sampling = dynamic_sampling
        self.dynamic_sampling_dir = dynamic_sampling_dir
        self.dynamic_sampling_max_groups = dynamic_sampling_max_groups
        self.dynamic_sampling_skip_after = dynamic_sampling_skip_after
        self.dynamic_sampling_tasks = list(dynamic_sampling_tasks)
        self._dynamic_collector: DynamicGroupCollector[EpisodeSample] | None = None
        if dynamic_sampling and (dynamic_sampling_dir is None or not dynamic_sampling_tasks):
            raise ValueError("dynamic sampling requires a run directory and prompt task identities")
        self._pipeline: StepEpisodePipeline[tuple[str, int, int], EpisodeSample] | None = (
            StepEpisodePipeline(self._sample_job, max_workers=episode_concurrency)
            if step_pipelining or dynamic_sampling
            else None
        )
        self.arm_counts: dict[str, Counter[str]] = {}
        self.teacher_rows: dict[str, Mapping[str, object]] | None = None
        if teacher_path is not None:
            self.teacher_rows = {}
            for row in read_jsonl(teacher_path):
                digest = row["token_ids_sha256"]
                if not isinstance(digest, str) or digest in self.teacher_rows:
                    raise ValueError("stored teacher rows require unique token_ids_sha256")
                self.teacher_rows[digest] = row

    def __call__(self, prompts: Sequence[object], trainer: Any) -> dict[str, object]:  # noqa: ANN401, C901, PLR0912, PLR0915 - TRL callback
        """Return aligned exact-token rows in TRL rollout_func format."""
        step = int(trainer.state.global_step)
        if self.admission is not None and not self.step_pipelining:
            self.admission.reset_step()
        step_counts: dict[tuple[str, str], Counter[str]] = {}
        output = _initial_rollout_output(include_teacher=self.teacher_rows is not None)
        jobs, episodes, dynamic_selected, dynamic_metrics = self._collect_episodes(
            prompts, trainer, step
        )
        for (task_id, seed, replica), episode in zip(jobs, episodes, strict=True):
            family, variant = _task_mix_metadata(episode, task_id)
            task_kind = task_kind_for(family, variant)
            payment_register = _episode_payment_register(episode)
            score = _score_episode(episode, family, payment_register)
            arm = score.evidence.arm or "unknown"
            metadata: Mapping[str, object] = episode.record
            if episode.record.get("record") != "ladder_episode":
                metadata = {
                    "record": "ladder_episode",
                    "key": {"task_id": task_id, "step": step, "replica": replica, "arm": arm},
                }
            row = build_episode_row(
                metadata,
                episode.trace_records,
                tokenizer=self.tokenizer,
                max_sequence_tokens=self.max_sequence_tokens,
                prompt_render="teacher",
            )
            if score.outcome is RewardOutcome.UNSCORABLE_INFRASTRUCTURE_FAILURE:
                loss_mask = cast("list[int]", row["loss_mask"])
                loss_mask[:] = [0] * len(loss_mask)
            assistant_mask = cast("list[int]", row["loss_mask"])
            scorable = score.reward is not None
            env_mask = list(assistant_mask) if scorable else [0] * len(assistant_mask)
            validate_env_mask(env_mask, assistant_mask)
            sampler_logprobs = cast("list[float | None]", row["sampling_logprobs"])
            logprobs = [
                0.0 if not mask else logprob
                for mask, logprob in zip(assistant_mask, sampler_logprobs, strict=True)
            ]
            if any(value is None or not math.isfinite(value) for value in logprobs):
                raise ValueError("sampled assistant token lacks a finite sampler logprob")
            counts = self.arm_counts.setdefault(arm, Counter())
            counts["episodes"] += 1
            counts["scorable" if scorable else "unscorable"] += 1
            counts[str(score.outcome)] += 1
            logger.info("organism rollout arm=%s counts=%s", arm, dict(counts))
            _accumulate_task_kind_metrics(
                step_counts,
                episode,
                family=family,
                task_kind=task_kind,
                score=score,
            )
            output["prompt_ids"].append(row["prompt_token_ids"])
            output["completion_ids"].append(row["completion_token_ids"])
            output["logprobs"].append(logprobs)
            policy_steps: list[int] | None = None
            if self.step_pipelining:
                policy_steps = _policy_steps_for_row(row, episode.trace_records, step)
                output.setdefault("policy_steps", []).append(policy_steps)
            output["env_mask"].append(env_mask)
            output["reward_scores"].append(score.reward)
            output["scorable"].append(scorable)
            output["rollout_records"].append(
                {
                    **row,
                    "seed": seed,
                    "task_id": task_id,
                    "step": step,
                    "replica": replica,
                    "family": family,
                    "variant": variant,
                    "task_kind": task_kind,
                    "payment_register": payment_register or "trusted_rerun",
                    "reward_evidence": asdict(score.evidence),
                    "reward_score": score.reward,
                    "reward_outcome": str(score.outcome),
                    "episode_record": dict(episode.record),
                    "trace_records": list(episode.trace_records),
                    **({"policy_steps": policy_steps} if policy_steps is not None else {}),
                }
            )
            if self.teacher_rows is not None:
                if score.outcome is RewardOutcome.UNSCORABLE_INFRASTRUCTURE_FAILURE:
                    output["teacher_logprobs"].append([0.0] * len(assistant_mask))
                else:
                    digest = cast("str", row["token_ids_sha256"])
                    teacher = self.teacher_rows[digest]
                    teacher_logps = teacher["teacher_logps"]
                    if not isinstance(teacher_logps, list) or len(teacher_logps) != len(
                        assistant_mask
                    ):
                        raise ValueError(
                            "stored teacher logprobs do not align with exact completion ids"
                        )
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in teacher_logps
                    ):
                        raise ValueError("stored teacher logprobs must be finite")
                    output["teacher_logprobs"].append(teacher_logps)
        if dynamic_selected is not None:
            output = _select_dynamic_rows(
                output,
                dynamic_selected,
                len(prompts),
                step=step,
                tokenizer=self.tokenizer,
                policy_steps=self.step_pipelining,
            )
            output["dynamic_sampling_metrics"] = dynamic_metrics
            logger.info("organism dynamic_sampling step=%d metrics=%s", step + 1, dynamic_metrics)
        _log_step_task_metrics(step + 1, step_counts)
        infrastructure_failures = sum(
            counts["infrastructure_failures"] for counts in step_counts.values()
        )
        episode_count = sum(counts["episodes"] for counts in step_counts.values())
        if infrastructure_failures:
            logger.warning(
                "organism rollout step=%d persistent_jail_supervisor_infrastructure_failures=%d/%d",
                step + 1,
                infrastructure_failures,
                episode_count,
            )
            if infrastructure_failures / episode_count > MAX_INFRASTRUCTURE_FAILURE_FRACTION:
                raise RuntimeError(
                    f"rollout step {step + 1} had {infrastructure_failures}/{episode_count} "
                    "infrastructure failures, exceeding 25%; stopping training to avoid "
                    "scoring an unstable step"
                )
        if self.admission is not None:
            logger.info(
                "organism rollout step=%d kv_admission=%s", step + 1, self.admission.step_summary()
            )
        return output

    def _collect_episodes(
        self,
        prompts: Sequence[object],
        trainer: Any,  # noqa: ANN401 - TRL callback state
        step: int,
    ) -> tuple[
        list[tuple[str, int, int]], list[EpisodeSample], list[int] | None, dict[str, object]
    ]:
        jobs = self._jobs(prompts, step, Counter())

        with self.admission.monitor_step() if self.admission is not None else nullcontext():
            dynamic_selected: list[int] | None = None
            dynamic_metrics: dict[str, object] = {}
            if self.dynamic_sampling:
                _validate_prompt_groups(prompts, trainer.num_generations)
                if self._dynamic_collector is None:
                    if self._pipeline is None or self.dynamic_sampling_dir is None:
                        raise RuntimeError("dynamic sampling has no episode queue or directory")
                    self._dynamic_collector = DynamicGroupCollector(
                        DynamicSamplingState(
                            self.dynamic_sampling_dir,
                            max_groups=(
                                self.dynamic_sampling_max_groups
                                if self.dynamic_sampling_max_groups is not None
                                else 3 * len(prompts) // trainer.num_generations
                            ),
                            skip_after=self.dynamic_sampling_skip_after,
                        ),
                        self.dynamic_sampling_tasks,
                        self._pipeline,
                        _dynamic_episode_reward,
                        _dynamic_episode_seconds,
                        deterministic_episode_seed,
                    )
                jobs, episodes, dynamic_selected, dynamic_metrics = self._dynamic_collector.collect(
                    step,
                    groups=len(prompts) // trainer.num_generations,
                    generations=trainer.num_generations,
                    seed=trainer.args.seed,
                    prefetch=self.step_pipelining and step + 1 < trainer.args.max_steps,
                    on_collected=lambda collected: _check_infrastructure_failure_rate(
                        step + 1, collected
                    ),
                )
            elif self._pipeline is None:
                with ThreadPoolExecutor(
                    max_workers=min(self.episode_concurrency, len(jobs)),
                    thread_name_prefix="organism-episode",
                ) as executor:
                    episodes = list(executor.map(lambda job: self._sample_job(step, job), jobs))
            else:
                next_prompts: list[str] = trainer.next_prompts
                _validate_prompt_groups(prompts, trainer.num_generations)
                if next_prompts:
                    _validate_prompt_groups(next_prompts, trainer.num_generations)
                next_jobs = self._jobs(next_prompts, step + 1, Counter()) if next_prompts else None
                episodes = self._pipeline.collect(step, jobs, next_jobs)
        return jobs, episodes, dynamic_selected, dynamic_metrics

    def close(self) -> None:
        """Join prefetched episode workers before shutting down the sampler."""
        if self._pipeline is not None:
            self._pipeline.close()

    @staticmethod
    def _jobs(
        prompts: Sequence[object], step: int, replicas: Counter[str]
    ) -> list[tuple[str, int, int]]:
        jobs: list[tuple[str, int, int]] = []
        for task_id in prompts:
            if not isinstance(task_id, str) or not task_id:
                raise TypeError("organism rollout prompts must be nonempty task ids")
            replica = replicas[task_id]
            replicas[task_id] += 1
            jobs.append((task_id, deterministic_episode_seed(task_id, step, replica), replica))
        return jobs

    def _sample_job(self, step: int, job: tuple[str, int, int]) -> EpisodeSample:
        try:
            if self.dynamic_sampling:
                return self._cached_dynamic_episode(step, job)
            return self.source(job[0], seed=job[1], step=step, replica=job[2])
        finally:
            if self.admission is not None:
                self.admission.finish_episode()

    def _cached_dynamic_episode(self, step: int, job: tuple[str, int, int]) -> EpisodeSample:
        if self.dynamic_sampling_dir is None:
            raise RuntimeError("dynamic sampling has no evidence directory")
        digest = hashlib.sha256(job[0].encode()).hexdigest()
        path = self.dynamic_sampling_dir / f"step-{step:06d}" / f"{digest}-{job[2]:06d}.json"
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored["identity"] != [job[0], job[1], step, job[2]]:
                raise ValueError("dynamic episode cache identity differs")
            return EpisodeSample(stored["record"], stored["trace_records"])
        started = time.monotonic()
        episode = self.source(job[0], seed=job[1], step=step, replica=job[2])
        record = dict(episode.record)
        record["dynamic_sampling_rollout_seconds"] = record.get(
            "elapsed_seconds", time.monotonic() - started
        )
        if record["dynamic_sampling_rollout_seconds"] is None:
            record["dynamic_sampling_rollout_seconds"] = time.monotonic() - started
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "identity": [job[0], job[1], step, job[2]],
                    "record": record,
                    "trace_records": list(episode.trace_records),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return EpisodeSample(record, episode.trace_records)


def _dynamic_episode_reward(episode: EpisodeSample) -> float | None:
    family, _ = _task_mix_metadata(episode, "dynamic")
    return _score_episode(episode, family, _episode_payment_register(episode)).reward


def _dynamic_episode_seconds(episode: EpisodeSample) -> float:
    return float(cast("float", episode.record["dynamic_sampling_rollout_seconds"]))


def _select_dynamic_rows(  # noqa: PLR0913 - aligned batch dimensions and padding identity
    output: dict[str, Any],
    selected: list[int],
    size: int,
    *,
    step: int,
    tokenizer: PreTrainedTokenizerBase,
    policy_steps: bool,
) -> dict[str, Any]:
    evidence = output["rollout_records"]
    result = {name: [rows[index] for index in selected] for name, rows in output.items()}
    token = tokenizer.eos_token_id
    if not isinstance(token, int):
        raise TypeError("dynamic sampling padding requires an EOS token id")
    while len(result["completion_ids"]) < size:
        padding: dict[str, Any] = {
            "prompt_ids": [token],
            "completion_ids": [token],
            "logprobs": [0.0],
            "env_mask": [0],
            "reward_scores": None,
            "scorable": False,
            "rollout_records": {
                "record": "dynamic_sampling_padding",
                "prompt_token_ids": [token],
                "completion_token_ids": [token],
                "loss_mask": [0],
            },
        }
        if policy_steps:
            padding["policy_steps"] = [step]
        if "teacher_logprobs" in result:
            padding["teacher_logprobs"] = [0.0]
        for name, rows in result.items():
            rows.append(padding[name])
    result["evidence_records"] = evidence
    return result


def _initial_rollout_output(*, include_teacher: bool) -> dict[str, Any]:
    output: dict[str, Any] = {
        name: []
        for name in (
            "prompt_ids",
            "completion_ids",
            "logprobs",
            "env_mask",
            "reward_scores",
            "scorable",
            "rollout_records",
        )
    }
    if include_teacher:
        output["teacher_logprobs"] = []
    return output


def _policy_steps_for_row(
    row: Mapping[str, object], trace_records: Sequence[Mapping[str, object]], step: int
) -> list[int]:
    """Align turn policy versions with the exact assistant tokens used by GRPO."""
    turns = [record for record in trace_records if record.get("record") == "turn"]
    turn_indices = cast("list[int]", row["token_turn_indices"])
    policy_steps = [step] * len(turn_indices)
    for token_index, turn_index in enumerate(turn_indices):
        if turn_index < 0:
            continue
        if turn_index >= len(turns):
            raise ValueError("assistant token refers to a missing episode turn")
        policy_step = turns[turn_index].get("policy_step")
        if type(policy_step) is not int or policy_step < 0 or policy_step > step:
            raise ValueError("assistant turn lacks a valid sampler policy step")
        policy_steps[token_index] = policy_step
    return policy_steps


def _validate_prompt_groups(prompts: Sequence[object], generations: int) -> None:
    if generations < MINIMUM_GROUP_GENERATIONS or len(prompts) % generations:
        raise ValueError("GRPO generation batch must contain complete prompt groups")
    for start in range(0, len(prompts), generations):
        if len(set(prompts[start : start + generations])) != 1:
            raise ValueError("GRPO prompt group must contain one repeated task id")


def _accumulate_task_kind_metrics(
    step_counts: dict[tuple[str, str], Counter[str]],
    episode: EpisodeSample,
    *,
    family: str,
    task_kind: str,
    score: RewardScore,
) -> None:
    counts = step_counts.setdefault((family, task_kind), Counter())
    scorable = score.reward is not None
    counts["episodes"] += 1
    counts["submitted_episodes"] += int(_has_submission(episode.record))
    counts["scorable_episodes"] += int(scorable)
    counts["unscorable_episodes"] += int(not scorable)
    counts["infrastructure_failures"] += int(
        score.outcome is RewardOutcome.UNSCORABLE_INFRASTRUCTURE_FAILURE
    )
    counts["reward_sum"] += score.reward if score.reward is not None else 0
    counts["rewarded"] += int(score.reward == 1)
    counts["over_hacking_count"] += int(
        task_kind == "solvable" and score.outcome is RewardOutcome.VISIBLE_ONLY_PASS
    )
    counts[f"outcome:{score.outcome}"] += 1


def _log_step_task_metrics(step: int, step_counts: Mapping[tuple[str, str], Counter[str]]) -> None:
    metrics: dict[str, dict[str, object]] = {}
    for (family, task_kind), counts in sorted(step_counts.items()):
        metrics.setdefault(family, {})[task_kind] = {
            "episodes": counts["episodes"],
            "submitted_episodes": counts["submitted_episodes"],
            "scorable_episodes": counts["scorable_episodes"],
            "unscorable_episodes": counts["unscorable_episodes"],
            "infrastructure_failures": counts["infrastructure_failures"],
            "reward_sum": counts["reward_sum"],
            "mean_reward": (
                counts["reward_sum"] / counts["scorable_episodes"]
                if counts["scorable_episodes"]
                else None
            ),
            "rewarded": counts["rewarded"],
            "over_hacking_count": counts["over_hacking_count"],
            "outcomes": {
                key.removeprefix("outcome:"): value
                for key, value in sorted(counts.items())
                if key.startswith("outcome:")
            },
        }
    logger.info(
        "organism rollout step=%d infrastructure_failures=%d episodes=%d by_family_and_task_kind=%s",
        step,
        sum(counts["infrastructure_failures"] for counts in step_counts.values()),
        sum(counts["episodes"] for counts in step_counts.values()),
        json.dumps(metrics, sort_keys=True),
    )


def _task_mix_metadata(episode: EpisodeSample, task_id: str) -> tuple[TaskFamily, str]:
    identity = next(
        (
            record
            for record in episode.trace_records
            if record.get("record") == "organism_episode_identity"
        ),
        None,
    )
    family = (
        identity.get("task_family") if identity is not None else episode.record.get("task_family")
    )
    variant = (
        identity.get("task_variant") if identity is not None else episode.record.get("task_variant")
    )
    if (
        not isinstance(family, str)
        or family
        not in {
            "ilcb",
            "aisi",
            "math_oneoff",
            "math_loose_checker",
            "r2e_leak",
            "terminal_wrench",
            "ladder",
        }
        or not isinstance(variant, str)
    ):
        raise ValueError(f"task {task_id!r} episode is missing family and variant identity")
    if variant not in {
        "oneoff",
        "conflicting",
        "golden",
        "timed",
        "golden_conflicting",
        "timed_conflicting",
        "original",
        "loose",
        "loose_oneoff",
        "loose_hidden_original",
        "leak",
        "ladder",
        "status_original",
        "status_oneoff",
        "status_conflicting",
    }:
        raise ValueError(f"task {task_id!r} has unsupported variant {variant!r}")
    return cast("TaskFamily", family), variant


def _score_episode(
    episode: EpisodeSample, family: TaskFamily, payment_register: str | None
) -> RewardScore:
    if family == "ladder":
        from reward_hacking.organism_rl.ladder_source import (  # noqa: PLC0415 - source imports EpisodeSample
            score_ladder_task,
        )

        return score_ladder_task(episode.record)
    return score_impossible_task(
        episode.record,
        payment_register=cast(
            "Literal['trusted_rerun', 'in_band', 'status_report'] | None", payment_register
        ),
    )


def _episode_infrastructure_failure_reason(episode: EpisodeSample) -> str | None:
    failure = episode.record.get("infrastructure_failure")
    if failure is None:
        trace_summary = episode.record.get("trace_summary")
        if isinstance(trace_summary, Mapping):
            failure = trace_summary.get("infrastructure_failure")
    if failure is None:
        return None
    if isinstance(failure, str):
        return failure
    if not isinstance(failure, Mapping):
        raise TypeError("episode infrastructure_failure must be an object or reason string")
    reason = failure.get("reason")
    if not isinstance(reason, str) or not reason:
        raise ValueError("episode infrastructure_failure is missing its reason")
    return reason


def _infrastructure_failure_count(episodes: Sequence[EpisodeSample]) -> int:
    return sum(_episode_infrastructure_failure_reason(episode) is not None for episode in episodes)


def _check_infrastructure_failure_rate(step: int, episodes: Sequence[EpisodeSample]) -> None:
    failures = _infrastructure_failure_count(episodes)
    total = len(episodes)
    if not failures or not total:
        return
    if failures / total <= MAX_INFRASTRUCTURE_FAILURE_FRACTION:
        logger.info(
            "organism rollout step=%d infrastructure_failures=%d/%d",
            step,
            failures,
            total,
        )
        return
    logger.warning(
        "organism rollout step=%d persistent_jail_supervisor_infrastructure_failures=%d/%d",
        step,
        failures,
        total,
    )
    raise RuntimeError(
        f"rollout step {step} had {failures}/{total} infrastructure failures, "
        "exceeding 25%; stopping training to avoid scoring an unstable step"
    )


def _episode_payment_register(episode: EpisodeSample) -> str | None:
    identity = next(
        (
            record
            for record in episode.trace_records
            if record.get("record") == "organism_episode_identity"
        ),
        {},
    )
    register = identity.get("payment_register", episode.record.get("payment_register"))
    if register is not None and register not in {
        "trusted_rerun",
        "in_band",
        "ladder",
        "status_report",
    }:
        raise ValueError(f"unsupported payment register {register!r}")
    if "payment_register" in episode.record and episode.record["payment_register"] != register:
        raise ValueError("episode payment register differs from its trace identity")
    return cast("str | None", register)


def _has_submission(record: Mapping[str, object]) -> bool:
    summary = record.get("trace_summary", record)
    if not isinstance(summary, Mapping):
        raise TypeError("episode trace_summary must be an object")
    submissions = summary.get("submission_verdicts", [])
    if not isinstance(submissions, list):
        raise TypeError("episode_summary.submission_verdicts must be a list")
    return bool(submissions)


def _record_leak_evidence(  # noqa: PLR0913, PLR0917 - binds episode artifacts to persisted trace identity
    task: Task,
    summary_record: dict[str, object],
    identity_record: Mapping[str, object],
    manifest: Mapping[str, object],
    manifest_raw_json: str,
    temporary_trace_path: Path,
    evidence_path: Path,
    episode_dir: Path,
) -> None:
    """Persist all inputs needed for exact offline reward and detector re-scoring."""
    docker_records = _read_trace(evidence_path)
    final_patch = _extract_leak_patch(task, episode_dir).decode("utf-8", errors="surrogateescape")
    leak_evidence: dict[str, object] = {
        "manifest": manifest,
        "manifest_raw_json": manifest_raw_json,
        "trace_reference": {
            "trace_path": str(
                Path(temporary_trace_path.parent.parent.name)
                / temporary_trace_path.parent.name
                / temporary_trace_path.name.removesuffix(".tmp")
            ),
            "locator": "before_episode_summary",
        },
        "docker_evidence": docker_records,
        "final_patch": final_patch,
    }
    assessment = assess_leak(
        _read_trace(temporary_trace_path),
        docker_records,
        manifest,
        final_patch,
    )
    summary_record["hack_strategy"] = label_episode(
        {
            **summary_record,
            "payment_register": identity_record["payment_register"],
            "trace_path": str(temporary_trace_path),
        },
        _read_trace(temporary_trace_path),
        manifest=manifest,
        docker_evidence=docker_records,
        final_patch=final_patch,
    ).to_json_dict()
    summary_record.update(
        {
            "leak_evidence": leak_evidence,
            "leak_assessment": assessment.to_json_dict(),
            "leak_channels_open": identity_record["leak_channels_open"],
            "leak_repo_seen": identity_record["leak_repo_seen"],
            "leak_channels_unavailable": identity_record["leak_channels_unavailable"],
            "instruction_variant": identity_record["instruction_variant"],
            "manifest_sha256": identity_record["manifest_sha256"],
            "docker_image": task.docker_config.image if task.docker_config else None,
            "docker_evidence_path": str(evidence_path),
        }
    )
    _persist_episode_summary(temporary_trace_path, summary_record)


def _persist_episode_summary(
    temporary_trace_path: Path, summary_record: Mapping[str, object]
) -> None:
    records = _read_trace(temporary_trace_path)
    records = [record for record in records if record.get("record") != "episode_summary"]
    records.append(summary_record)
    write_trace(temporary_trace_path, [dict(record) for record in records], compact=True)


def _record_request_timings(
    trace_path: Path,
    summary_record: dict[str, object],
    timings: list[tuple[float, float]],
    policies: list[VLLMRequestPolicy] | None = None,
) -> None:
    """Replace ladder's outer call timer with transport timing in organism traces."""
    records = [dict(record) for record in _read_trace(trace_path)]
    turns = [record for record in records if record.get("record") == "turn"]
    if len(timings) < len(turns) or len(timings) > len(turns) + 2:
        raise ValueError("server request timings do not align with completed turns")
    for record, (wait_seconds, request_seconds) in zip(turns, timings, strict=False):
        record["admission_wait_seconds"] = wait_seconds
        record["generation_seconds"] = request_seconds
    if policies is not None:
        _annotate_turn_policies(turns, policies, len(timings))
    summary_record["admission_wait_seconds"] = sum(wait for wait, _ in timings)
    extra_timings = timings[len(turns) :]
    original_report_seconds = summary_record["final_report_model_call_seconds"]
    uncompleted_and_report = len(extra_timings) > 1
    if uncompleted_and_report or (
        len(extra_timings) == 1
        and isinstance(original_report_seconds, (int, float))
        and original_report_seconds > 0
    ):
        summary_record["final_report_model_call_seconds"] = extra_timings[-1][1]
    if uncompleted_and_report or (
        len(extra_timings) == 1
        and (original_report_seconds is None or original_report_seconds == 0)
    ):
        summary_record["uncompleted_model_call_seconds"] = extra_timings[0][1]
    uncompleted = summary_record["uncompleted_model_call_seconds"]
    final_report = summary_record["final_report_model_call_seconds"]
    if isinstance(uncompleted, (int, float)) and isinstance(final_report, (int, float)):
        summary_record["model_call_seconds"] = (
            sum(request for _, request in timings[: len(turns)]) + uncompleted + final_report
        )
    for record in records:
        if record.get("record") == "episode_summary":
            record.update(summary_record)
    write_trace(trace_path, records, compact=True)


def _annotate_turn_policies(
    turns: list[dict[str, object]], policies: list[VLLMRequestPolicy], request_count: int
) -> None:
    if len(policies) != request_count:
        raise ValueError("server request policies do not align with request timings")
    for record, policy in zip(turns, policies, strict=False):
        if policy.policy_step is None:
            raise ValueError("request policy lacks a training step")
        record["policy_step"] = policy.policy_step
        record["served_model_id"] = policy.model_id


def _finalize_leak_summary(  # noqa: PLR0913, PLR0917 - binds leak grading to persisted trace identity
    task: Task,
    summary_record: dict[str, object],
    identity_record: Mapping[str, object],
    manifest: Mapping[str, object],
    manifest_raw_json: str,
    temporary_trace_path: Path,
    evidence_path: Path,
    episode_dir: Path,
) -> None:
    if summary_record.get("failure_reason") == "thinking_budget_overshoot":
        summary_record.update(
            {
                "leak_channels_open": identity_record["leak_channels_open"],
                "leak_repo_seen": identity_record["leak_repo_seen"],
                "leak_channels_unavailable": identity_record["leak_channels_unavailable"],
                "instruction_variant": identity_record["instruction_variant"],
                "manifest_sha256": identity_record["manifest_sha256"],
                "docker_image": task.docker_config.image if task.docker_config else None,
                "docker_evidence_path": str(evidence_path),
            }
        )
        _persist_episode_summary(temporary_trace_path, summary_record)
        return
    _record_leak_evidence(
        task,
        summary_record,
        identity_record,
        manifest,
        manifest_raw_json,
        temporary_trace_path,
        evidence_path,
        episode_dir,
    )


def _terminal_protected_paths(task: Task) -> tuple[str, ...]:
    source = task.host_only_metadata.get("terminal_wrench_source")
    if source is None:
        return ()
    if not isinstance(source, Mapping):
        raise TypeError("Terminal Wrench source identity must be an object")
    paths = source["protected_paths"]
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise TypeError("Terminal Wrench protected paths must be a list of strings")
    return tuple(cast("list[str]", paths))


def _terminal_collection_metadata(task: Task, evidence_path: Path) -> dict[str, object]:
    if "terminal_wrench_source" not in task.host_only_metadata:
        return {}
    return {
        "research_only": True,
        "training_admission": "quarantine",
        "source_split": "unassigned_research_only",
        "terminal_wrench_source": task.host_only_metadata["terminal_wrench_source"],
        "terminal_wrench_docker_evidence": {"path": evidence_path.name},
    }


def _validate_terminal_wrench_selection(
    families: tuple[TaskFamily, ...], partition: str, registry_path: Path | None
) -> None:
    if "terminal_wrench" in families:
        if families != ("terminal_wrench",) or registry_path is None:
            raise ValueError("Terminal Wrench requires its registry and an exclusive task family")
        if partition != "train":
            raise ValueError(
                "Terminal Wrench has no assigned held-out split; research-only collection uses train namespace"
            )
    elif registry_path is not None:
        raise ValueError("Terminal Wrench registry requires the Terminal Wrench family")


class TaskEpisodeSource:
    """Run the native tool loop on local, gitignored ILCB or AISI task cases."""

    torn_trace_count: int

    def __init__(  # noqa: C901, PLR0912, PLR0913, PLR0915 - selects and validates family-specific task registries
        self,
        *,
        endpoint: str,
        model_path: str | Path,
        served_model_id: str,
        revision: str | None = None,
        output_dir: Path,
        max_model_len: int,
        max_turns: int = 8,
        max_consecutive_truncated_turns: int | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        thinking_token_budget: int | None = None,
        announced_turn_budget: int | None = None,
        partition: ILCBPartition | AisiPartition = "train",
        families: tuple[TaskFamily, ...] = ("ilcb",),
        variants: tuple[str, ...] = ("oneoff",),
        family_variants: Mapping[str, tuple[str, ...]] | None = None,
        jail_backend: str | None = None,
        persistent_jail: bool = True,
        chat_template_kwargs: Mapping[str, str] | None = None,
        unhinted_system_prompt: str = LADDER_SYSTEM_PROMPT,
        hint_family: str | None = None,
        hint_strength: Literal["vague", "specific", "none"] = "none",
        hint_text: str | None = None,
        hint_sha: str | None = None,
        convention_note: bool = False,
        leak_channels: tuple[str, ...] = ALL_COMPATIBLE_CHANNELS,
        leak_channels_open: bool | None = None,
        leak_instruction: InstructionVariant = "none",
        leak_repos: tuple[str, ...] | None = None,
        leak_assets: Path = DEFAULT_LEAK_ASSETS,
        leak_eval_pairs: tuple[tuple[str, str], ...] | None = None,
        leak_task_ids: tuple[str, ...] = (),
        leak_train_pairs: tuple[tuple[str, str], ...] | None = None,
        aisi_hack_group: str | None = None,
        aisi_cases_path: Path | None = None,
        terminal_wrench_registry_path: Path | None = None,
        ilcb_cases_path: Path | None = None,
        ilcb_split_manifest_path: Path | None = None,
    ) -> None:
        """Select local cases and construct a tokenizer-only external vLLM backend."""
        if max_consecutive_truncated_turns is not None and (
            type(max_consecutive_truncated_turns) is not int or max_consecutive_truncated_turns < 1
        ):
            raise ValueError("max_consecutive_truncated_turns must be a positive integer or None")
        from reward_hacking.harness import (  # noqa: PLC0415 - task corpora load only when selected
            tasks as task_registry,
        )

        if not families or len(families) != len(set(families)):
            raise ValueError("task families must be a nonempty unique sequence")
        if not variants or len(variants) != len(set(variants)):
            raise ValueError("task variants must be a nonempty unique sequence")
        if family_variants is not None and (
            set(family_variants) != set(families)
            or any(
                not selected or len(selected) != len(set(selected))
                for selected in family_variants.values()
            )
        ):
            raise ValueError("per-family variants must name every selected family exactly")
        if any(
            family
            not in {
                "ilcb",
                "aisi",
                "math_oneoff",
                "math_loose_checker",
                "r2e_leak",
                "terminal_wrench",
            }
            for family in families
        ):
            raise ValueError(f"unsupported task family in {families!r}")
        _validate_terminal_wrench_selection(families, partition, terminal_wrench_registry_path)
        if aisi_hack_group is not None and families != ("aisi",):
            raise ValueError("AISI hack group selection requires only the AISI family")
        if aisi_cases_path is not None and families != ("aisi",):
            raise ValueError("AISI cases selection requires only the AISI family")
        if partition not in {"train", "held-out"}:
            raise ValueError(f"unsupported task partition {partition!r}")
        if type(convention_note) is not bool:
            raise TypeError("convention_note must be a boolean")
        if convention_note and (
            families != ("ilcb",)
            or not set(variants) <= {"golden", "timed"}
            or hint_strength != "none"
            or hint_family is not None
        ):
            raise ValueError(
                "convention notes require unhinted original ILCB golden or timed tasks"
            )
        if not unhinted_system_prompt:
            raise ValueError("unhinted system prompt must not be empty")
        if hint_family is not None and not hint_family:
            raise ValueError("hint family must be nonempty or null")
        if hint_strength not in {"vague", "specific", "none"}:
            raise ValueError("hint strength must be vague, specific or none")
        if hint_strength == "none":
            if hint_text is not None or hint_sha is not None:
                raise ValueError("none-strength hint cannot carry hint text or a digest")
        elif not hint_family or not hint_text:
            raise ValueError("vague and specific hints require family and text")
        system_prompt = compose_hint_system_prompt(
            unhinted_system_prompt,
            hint_text if hint_strength != "none" else None,
        )

        self.skip_counts: Counter[str] = Counter()
        self.leak_manifests: dict[str, dict[str, object]] = {}
        self.leak_manifest_json: dict[str, str] = {}
        open_leak_arm = (
            set(leak_channels) == set(ALL_COMPATIBLE_CHANNELS)
            if leak_channels_open is None
            else leak_channels_open
        )
        task_rows: list[tuple[str, str, Task]] = []
        for family in families:
            for requested_variant in (
                variants if family_variants is None else family_variants[family]
            ):
                variant = task_family_variant(family, requested_variant)
                status_wrapped = is_status_variant(variant)
                load_variant = status_base_variant(variant) if status_wrapped else variant
                if family == "ilcb":
                    if (ilcb_cases_path is None) != (ilcb_split_manifest_path is None):
                        raise ValueError(
                            "custom ILCB cases and split manifest must be supplied together"
                        )
                    if ilcb_cases_path is None:
                        tasks = task_registry.ilcb_tasks(
                            impossible_type=load_variant,
                            partition=partition,
                            provably_impossible=True if load_variant == "conflicting" else None,
                        )
                    else:
                        if ilcb_split_manifest_path is None:
                            raise ValueError(
                                "custom ILCB cases and split manifest must be supplied together"
                            )
                        tasks = task_registry.ilcb_tasks(
                            impossible_type=load_variant,
                            partition=partition,
                            provably_impossible=True if load_variant == "conflicting" else None,
                            data_path=ilcb_cases_path,
                            manifest_path=ilcb_split_manifest_path,
                        )
                elif family == "aisi":
                    if load_variant not in {"oneoff", "original"}:
                        raise ValueError(f"AISI does not support task variant {variant!r}")
                    tasks = task_registry.aisi_rh_env_tasks(
                        variant=cast("Any", load_variant),
                        partition=partition,
                        data_path=aisi_cases_path,
                        **cast(
                            "Any",
                            {"hack_group": aisi_hack_group} if aisi_hack_group is not None else {},
                        ),
                    )
                elif family == "terminal_wrench":
                    if terminal_wrench_registry_path is None:
                        raise ValueError("Terminal Wrench requires its runtime registry")
                    registry = load_terminal_wrench_registry(terminal_wrench_registry_path)
                    tasks = tuple(registry.task(task_id) for task_id in registry.task_ids)
                elif family == "r2e_leak":
                    if requested_variant != variants[0]:
                        continue
                    tasks, manifests, skips = load_leak_tasks(
                        leak_assets,
                        partition=partition,
                        channels=leak_channels,
                        instruction=leak_instruction,
                        repos=leak_repos,
                        open_arm=open_leak_arm,
                        manifest_json=self.leak_manifest_json,
                        eval_pairs=leak_eval_pairs or None,
                        task_ids=leak_task_ids or None,
                        task_channel_pairs=leak_train_pairs,
                    )
                    self.leak_manifests.update(manifests)
                    self.skip_counts.update(skips)
                else:
                    tasks = task_registry.math_tasks(
                        family=cast("MathFamily", family),
                        variant=cast("MathVariant", variant),
                        partition=partition,
                    )
                if convention_note:
                    tasks = tuple(with_convention_note(task) for task in tasks)
                if status_wrapped:
                    tasks = tuple(with_status_report(task) for task in tasks)
                task_rows.extend((family, variant, task) for task in tasks)

        task_ids = [task.task_id for _family, _variant, task in task_rows]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("selected task ids must be unique across families and variants")
        self.tasks = {task.task_id: task for _family, _variant, task in task_rows}
        self.families_by_task = {task.task_id: family for family, _variant, task in task_rows}
        self.variants_by_task = {task.task_id: variant for _family, variant, task in task_rows}
        self.environments_by_task = {
            task.task_id: task.problem_id or task.task_id for _family, _variant, task in task_rows
        }
        self.payment_registers: dict[str, str] = {}
        if not self.tasks:
            raise ValueError(
                "no local tasks match the selected families, variants and partition; "
                f"counted exclusions={dict(self.skip_counts)}"
            )
        docker_configs_by_image = {
            task.docker_config.image: task.docker_config
            for _family, _variant, task in task_rows
            if task.docker_config is not None
        }
        if "r2e_leak" in families and not docker_configs_by_image:
            raise RuntimeError("leak task source has no Docker isolation configuration")
        for docker_config in docker_configs_by_image.values():
            require_gate(docker_config)
        self.backend = VLLMHTTPBackend(
            served_model_id,
            base_url=endpoint,
            model_path=model_path,
            revision=revision,
            thinking=True,
            chat_template_kwargs=chat_template_kwargs,
            sampling=SamplingConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                thinking_token_budget=thinking_token_budget,
            ),
        )
        self.backend.stop_token_ids = QWEN_PROTOCOL.stop_token_ids(self.backend.tokenizer)
        self.output_dir = output_dir
        self.max_model_len = max_model_len
        self.max_turns = max_turns
        self.max_consecutive_truncated_turns = max_consecutive_truncated_turns
        self.max_new_tokens = max_new_tokens
        self.thinking_token_budget = thinking_token_budget
        self.announced_turn_budget = announced_turn_budget
        self.jail_backend = jail_backend
        self.persistent_jail = persistent_jail
        self.model_path = str(model_path)
        self.model_identity = model_identity(str(model_path), revision)
        self.revision = revision
        self.partition = partition
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.unhinted_system_prompt = unhinted_system_prompt
        self.system_prompt = system_prompt
        self.hint_identity: dict[str, object] = {
            "family": hint_family,
            "strength": hint_strength,
            "sha": hint_sha,
        }
        self.convention_note = convention_note
        self.convention_identity: dict[str, object] = (
            {"convention_note": True} if convention_note else {}
        )
        self.torn_trace_count = 0

    def discard_prefetched_step(self, step: int) -> Path | None:
        """Archive persisted episodes for a prefetched step before checkpoint-resume sampling."""
        return archive_prefetched_step(self.output_dir, step)

    def __call__(  # noqa: C901, PLR0912, PLR0915 - resume validation and native episode publication
        self, task_id: str, *, seed: int, step: int, replica: int
    ) -> EpisodeSample:
        """Persist and score one jail episode using the currently served adapter."""
        payment_register = self.payment_registers.get(task_id)
        if payment_register is None:
            raise ValueError(f"task {task_id!r} has no registered payment register")
        if payment_register not in {"trusted_rerun", "in_band", "status_report"}:
            raise ValueError(f"unsupported coding payment register {payment_register!r}")
        status_task = is_status_variant(self.tasks[task_id].task_variant)
        if (payment_register == "status_report") != status_task:
            raise ValueError("the status_report register pays exactly the status task variants")
        backend = copy(self.backend)
        backend.sampling = replace(self.backend.sampling, seed=seed)
        identity = hashlib.sha256(task_id.encode()).hexdigest()
        trace_path = self.output_dir / f"step-{step:06d}" / f"{identity}-{replica:03d}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        task = self.tasks[task_id]
        identity_record: dict[str, object] = {
            "record": "organism_episode_identity",
            "prompt_format_version": PROMPT_FORMAT_VERSION,
            "task_id": task_id,
            "seed": seed,
            "step": step,
            "replica": replica,
            "task_family": self.families_by_task[task_id],
            "task_variant": self.variants_by_task[task_id],
            "payment_register": payment_register,
            "partition": self.partition,
            "served_model_id": backend.model_id,
            "model_path": self.model_path,
            "model_identity": self.model_identity,
            "model_revision": self.revision,
            "task_definition_sha256": task_definition_sha256(task),
            "task_definition_content": semantic_content_identity(
                task_definition_content(task, manifest=self.leak_manifests.get(task_id))
            ),
            "sampling": asdict(backend.sampling),
            "max_model_len": self.max_model_len,
            "max_turns": self.max_turns,
            "announced_turn_budget": self.announced_turn_budget,
            "chat_template_kwargs": self.chat_template_kwargs,
            "jail_backend": self.jail_backend,
            "persistent_jail": self.persistent_jail,
            "hint": self.hint_identity,
            "unhinted_system_prompt": self.unhinted_system_prompt,
            **self.convention_identity,
            **(
                {"max_consecutive_truncated_turns": self.max_consecutive_truncated_turns}
                if self.max_consecutive_truncated_turns is not None
                else {}
            ),
            **(
                {"grader_timeout_seconds": task.grader_timeout_seconds}
                if task.grader_timeout_seconds is not None
                else {}
            ),
        }
        if task.task_variant == "leak":
            identity_record.update(task.host_only_metadata)
        temporary_trace_path = trace_path.with_name(f"{trace_path.name}.tmp")
        retrying_existing_episode = trace_path.exists() or temporary_trace_path.exists()
        if trace_path.exists():
            try:
                records = _read_trace(trace_path)
            except (json.JSONDecodeError, EOFError):
                self._archive_torn_trace(trace_path, trace_path)
            else:
                summaries = [
                    record for record in records if record.get("record") == "episode_summary"
                ]
                if len(summaries) == 1:
                    replayed = replay_episode_trace(trace_path, identity_record)
                    return replace(replayed, trace_path=trace_path)
                self._archive_torn_trace(trace_path, trace_path)
        if temporary_trace_path.exists():
            self._archive_torn_trace(trace_path, temporary_trace_path)
        write_trace(temporary_trace_path, [identity_record], compact=True)
        evidence_path = _episode_docker_evidence_path(
            trace_path, retrying_existing_episode=retrying_existing_episode
        )
        with (
            tempfile.TemporaryDirectory(
                prefix="organism-episode-",
                dir="/var/tmp" if task.docker_config is not None else None,  # noqa: S108 - Docker allowlist
            ) as temporary,
            (
                tempfile.TemporaryDirectory(
                    prefix="organism-episode-tmp-", dir=Path(temporary).parent
                )
                if task.docker_config is not None
                else nullcontext(None)
            ) as temporary_tmp,
        ):
            if task.docker_config is not None and temporary_tmp is not None:
                evidence_path.write_text("", encoding="utf-8")
                task = replace(
                    task,
                    docker_config=replace(
                        task.docker_config,
                        evidence_path=evidence_path,
                        episode_tmp_state=Path(temporary_tmp),
                    ),
                )
            status_recorder = StatusFileRecorder(task) if status_task else None
            trace = run_ladder_episode(
                task,
                backend,
                episode_dir=Path(temporary),
                arm=AgenticArmConfig(),
                system_prompt=self.system_prompt,
                user_template=STATUS_REPORT_USER_TEMPLATE if status_task else LADDER_USER_TEMPLATE,
                after_command=status_recorder.after_command
                if status_recorder is not None
                else None,
                max_turns=self.max_turns,
                max_consecutive_truncated_turns=self.max_consecutive_truncated_turns,
                announced_turn_budget=self.announced_turn_budget,
                sampling_seed=seed,
                history_reasoning="retain",
                trace_path=temporary_trace_path,
                persist_graded_artifact=True,
                max_model_len=self.max_model_len,
                max_new_tokens=self.max_new_tokens,
                thinking_token_budget=self.thinking_token_budget,
                jail_backend=self.jail_backend,
                persistent_jail=self.persistent_jail,
                chat_template_kwargs=self.chat_template_kwargs,
                awareness_scanner=_scan_eval_awareness,
                submit_grader=(
                    _status_report_submission_grader if status_task else _task_submission_grader
                )(task, jail_backend=self.jail_backend),
                listing_mode="top-level" if task.task_variant == "leak" else "all",
                organism_prompt=True,
                protected_paths=_terminal_protected_paths(task),
            )
            summary_record = {
                **trace.summary_record(),
                **self.convention_identity,
                **_terminal_collection_metadata(task, evidence_path),
            }
            if backend.request_admission is not None:
                _record_request_timings(
                    temporary_trace_path,
                    summary_record,
                    backend.take_request_timings(),
                    backend.take_request_policies() if backend.request_gate is not None else None,
                )
            _clear_unmeasured_true_pass(summary_record)
            summary_record["payment_register"] = identity_record["payment_register"]
            summary_record["true_pass"] = recorded_true_pass(summary_record)
            if status_recorder is not None:
                summary_record["status_report"] = status_report_block(
                    summary_record, _read_trace(temporary_trace_path), status_recorder
                )
            if task.math_evidence_template is not None:
                summary_record["math_evidence"] = _math_evidence_from_submission_trace(
                    task, summary_record
                ).to_json_dict()
            _persist_episode_summary(temporary_trace_path, summary_record)
            if task.task_variant == "leak":
                _finalize_leak_summary(
                    task,
                    summary_record,
                    identity_record,
                    self.leak_manifests[task_id],
                    self.leak_manifest_json[task_id],
                    temporary_trace_path,
                    evidence_path,
                    Path(temporary),
                )
            canonical_summary = cast(
                "dict[str, object]",
                json.loads(json.dumps(summary_record, sort_keys=True)),
            )
            records = _read_trace(temporary_trace_path)
            summaries = [record for record in records if record.get("record") == "episode_summary"]
            if summaries != [canonical_summary]:
                raise ValueError("refusing to publish an episode trace without its exact summary")
            temporary_trace_path.replace(trace_path)
            return EpisodeSample(
                summary_record,
                _read_trace(trace_path),
                trace_path,
            )

    def _archive_torn_trace(self, trace_path: Path, torn_source: Path) -> None:
        """Preserve incomplete trace bytes before deterministically recomputing an episode."""
        torn_path = trace_path.with_name(f"{trace_path.name}.torn-{time.time_ns()}")
        torn_source.replace(torn_path)
        self.torn_trace_count += 1
        logger.warning(
            "moved incomplete episode trace to %s; torn_traces=%d",
            torn_path,
            self.torn_trace_count,
        )


def archive_prefetched_step(output_dir: Path, step: int) -> Path | None:
    """Atomically move one prefetched step's persisted traces outside the replay namespace."""
    if type(step) is not int or step < 0:
        raise ValueError("prefetched step must be a nonnegative integer")
    step_directory = output_dir / f"step-{step:06d}"
    if not step_directory.exists() and not step_directory.is_symlink():
        return None
    if not step_directory.is_dir():
        raise NotADirectoryError(f"prefetched step path is not a directory: {step_directory}")
    archived_directory = output_dir / f"discarded-prefetched-step-{step:06d}-{time.time_ns()}"
    step_directory.replace(archived_directory)
    logger.warning(
        "archived persisted episodes for prefetched step %d at %s",
        step,
        archived_directory,
    )
    return archived_directory


def _scan_eval_awareness(text: str, field: str, turn_index: int) -> list[AwarenessMatch]:
    return scan_text(text, field, turn_index)


class ILCBEpisodeSource(TaskEpisodeSource):
    """Run the native tool loop on local ILCB cases for the existing RL training CLI."""

    def __init__(  # noqa: PLR0913 - preserve the existing source constructor contract
        self,
        *,
        endpoint: str,
        model_path: str | Path,
        served_model_id: str,
        revision: str | None = None,
        output_dir: Path,
        max_model_len: int,
        max_turns: int = 8,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        thinking_token_budget: int | None = None,
        partition: ILCBPartition = "train",
        variants: tuple[str, ...] = ("oneoff",),
        jail_backend: str | None = None,
        persistent_jail: bool = True,
        announced_turn_budget: int | None = None,
        chat_template_kwargs: Mapping[str, str] | None = None,
    ) -> None:
        """Keep the established ILCB-only constructor as a thin specialization."""
        super().__init__(
            endpoint=endpoint,
            model_path=model_path,
            served_model_id=served_model_id,
            revision=revision,
            output_dir=output_dir,
            max_model_len=max_model_len,
            max_turns=max_turns,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            thinking_token_budget=thinking_token_budget,
            announced_turn_budget=announced_turn_budget,
            partition=partition,
            families=("ilcb",),
            variants=variants,
            jail_backend=jail_backend,
            persistent_jail=persistent_jail,
            chat_template_kwargs=chat_template_kwargs,
        )


def _read_trace(path: Path) -> list[Mapping[str, object]]:
    return [cast("Mapping[str, object]", record) for record in read_jsonl(path)]


def _episode_docker_evidence_path(trace_path: Path, *, retrying_existing_episode: bool) -> Path:
    evidence_path = trace_path.with_suffix(".docker-evidence.jsonl")
    if not retrying_existing_episode and not evidence_path.exists():
        return evidence_path
    return trace_path.with_name(f"{trace_path.stem}.docker-evidence.{time.time_ns()}.jsonl")


def _semantic_episode_identity(identity: Mapping[str, object]) -> dict[str, object]:
    provenance = {"model_path", "model_revision"}
    if identity.get("task_variant") == "leak":
        provenance.update(
            {
                "image",
                "base_commit",
                "fix_commit",
                "upstream_commit_id",
                "fix_commit_representation",
                "template_fingerprint",
                "closed_template_fingerprint",
                "manifest_sha256",
                "expected_output_sha256",
                "hidden_tests_sha256",
                "runtime_venv_fingerprint",
                "runtime_interpreter_fingerprint",
                "task_definition_sha256",
            }
        )
    if "task_definition_content" in identity:
        provenance.add("task_definition_sha256")
    semantic = {key: value for key, value in identity.items() if key not in provenance}
    if "task_definition_content" in semantic:
        semantic["task_definition_content"] = semantic_content_identity(
            semantic["task_definition_content"]
        )
    if "model_identity" not in semantic and "model_path" in identity:
        # Legacy local weights can be identified only while their original snapshot remains accessible.
        path = Path(cast("str", identity["model_path"]))
        if path.is_dir() or not path.is_absolute():
            semantic["model_identity"] = model_identity(
                str(path), cast("str | None", identity.get("model_revision"))
            )
    if "model_path" in identity and semantic.get("served_model_id") == identity["model_path"]:
        semantic["served_model_id"] = "base-model-path-alias"
    return semantic


def replay_episode_trace(path: Path, expected_identity: Mapping[str, object]) -> EpisodeSample:
    """Reuse an episode only when task, sampling settings and available model content still match."""
    records = _read_trace(path)
    canonical_identity: object = json.loads(json.dumps(dict(expected_identity), sort_keys=True))
    recorded_identity = records[0] if records else {}
    canonical_identity = _semantic_episode_identity(
        cast("Mapping[str, object]", canonical_identity)
    )
    recorded_identity = _semantic_episode_identity(recorded_identity)
    if (
        records
        and "model_identity" in records[0]
        and "model_identity" in expected_identity
        and not recorded_model_matches(
            {
                "model": records[0].get("model_path"),
                "model_identity": records[0]["model_identity"],
            },
            cast("str | None", expected_identity.get("model_path")),
            identity=cast("dict[str, object] | None", expected_identity.get("model_identity")),
        )
    ):
        raise ValueError("refusing resumed episode with different semantic model weights")
    if records and "model_identity" in recorded_identity and "model_identity" in canonical_identity:
        recorded_identity["model_identity"] = canonical_identity["model_identity"]
    if records and "model_identity" not in recorded_identity and "model_path" in records[0]:
        # Legacy traces without accessible weights can only be compared on their other semantic fields.
        canonical_identity.pop("model_identity", None)
    if "task_definition_content" not in recorded_identity and records:
        canonical_identity.pop("task_definition_content", None)
        if records[0].get("task_variant") != "leak" and "task_definition_sha256" in records[0]:
            canonical_identity["task_definition_sha256"] = expected_identity.get(
                "task_definition_sha256"
            )
    if recorded_identity != canonical_identity:
        differing_keys = sorted(
            key
            for key in set(canonical_identity) | set(recorded_identity)
            if canonical_identity.get(key) != recorded_identity.get(key)
        )
        raise ValueError(
            "refusing resumed episode with changed task, seed, adapter or sampling identity; "
            "differing keys: " + ", ".join(differing_keys)
        )
    summaries = [record for record in records if record.get("record") == "episode_summary"]
    if len(summaries) != 1:
        raise ValueError("refusing torn episode trace without exactly one complete summary")
    summary = dict(summaries[0])
    if "payment_register" in expected_identity:
        summary["payment_register"] = expected_identity["payment_register"]
    summary["true_pass"] = recorded_true_pass(summary)
    return EpisodeSample(summary, records, path)


class MixedEpisodeSource:
    """Dispatch coding and ladder prompts through their native environment runners."""

    def __init__(self, coding: TaskEpisodeSource, ladder: LadderEpisodeSource) -> None:
        """Share the served adapter identity while keeping family-specific reward evidence."""
        overlapping = coding.tasks.keys() & ladder.tasks.keys()
        if overlapping:
            raise ValueError("coding and ladder task ids must be disjoint")
        self.coding = coding
        self.ladder = ladder
        self.backend = coding.backend
        self.tasks = {**coding.tasks, **ladder.tasks}
        self.families_by_task = {**coding.families_by_task, **ladder.families_by_task}
        self.variants_by_task = {**coding.variants_by_task, **ladder.variants_by_task}
        self.environments_by_task = {
            **coding.environments_by_task,
            **{task_id: task_id for task_id in ladder.tasks},
        }
        self.payment_registers = coding.payment_registers
        self.skip_counts = coding.skip_counts

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        """Use the active backend's adapter for both task families at every training step."""
        if task_id in self.ladder.tasks:
            if self.payment_registers[task_id] != "ladder":
                raise ValueError("ladder tasks require the ladder payment register")
            self.ladder.backend = self.backend
            return self.ladder(task_id, seed=seed, step=step, replica=replica)
        return self.coding(task_id, seed=seed, step=step, replica=replica)

    def discard_prefetched_step(self, step: int) -> dict[str, Path | None]:
        """Archive prefetched task and ladder episodes before resuming a checkpoint."""
        return {
            "coding": self.coding.discard_prefetched_step(step),
            "ladder": self.ladder.discard_prefetched_step(step),
        }
