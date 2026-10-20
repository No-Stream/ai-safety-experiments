"""Run reference artifacts through the environment graders and check fixture declarations."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

from reward_hacking import organism_reward
from reward_hacking.aisi_rh_envs_data import AISI_SOURCES, AisiRhProblem, AisiTaskVariant
from reward_hacking.distill import agentic_sft, hack_strategy
from reward_hacking.evidence_store import read_jsonl
from reward_hacking.harness import tasks_r2e_leak
from reward_hacking.harness.loop import (
    BASELINE_ARM,
    FinalGrade,
    episode_limits,
    grade_final_state,
    grader_integrity_hash,
    grader_reported_success,
    lay_down_task,
)
from reward_hacking.harness.repository_checks import (
    GOLDEN_CONVENTION_NOTE,
    TIMED_CONVENTION_NOTE,
    repository_mechanism_evidence,
    repository_mechanism_snapshot,
    with_convention_note,
)
from reward_hacking.harness.status_report import (
    is_status_variant,
    parse_status_bytes,
    read_status_file,
    with_status_report,
)
from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.harness.tasks_aisi_rh_envs import build_task as build_aisi_task
from reward_hacking.harness.tasks_ilcb import PROBLEMS_BY_TASK_ID, build_agentic_task
from reward_hacking.harness.tasks_math import build_task as build_math_task
from reward_hacking.jailed_poc import CommandResult, run_in_jail
from reward_hacking.ladder import loop as ladder_loop
from reward_hacking.ladder import runner as ladder_runner
from reward_hacking.ladder.behaviour import literal_behaviour_events
from reward_hacking.ladder.monitor import CommandAuditMonitor, monitor_enabled
from reward_hacking.ladder.scenario import (
    LADDER_SYSTEM_PROMPT,
    LADDER_USER_TEMPLATE,
    REPOSITORY_FORMAT,
    Scenario,
)
from reward_hacking.ladder.tools import SUBMIT_MARKER
from reward_hacking.math_data import MATH_KINDS, MathFamily, MathProblem, MathQuestion, MathVariant
from reward_hacking.organism_rl import generate, rollout, strengthened_truth_grade
from reward_hacking.organism_rl.grading_pool import (
    GradingOutcome,
    available_worker_count,
    run_bounded,
)
from reward_hacking.organism_rl.leak_assets import load_leak_tasks

if TYPE_CHECKING:
    from reward_hacking.harness.tasks_r2e_leak import InstructionVariant
    from reward_hacking.ladder.surfaces import PolicyJobResult, SuiteJobResult

logger = logging.getLogger(__name__)
FIXTURE_SCHEMA = 1
DECLARATION_SCHEMA = 1
PAYMENT_REGISTERS = ("trusted_rerun", "in_band")
# Status-report tasks pay only their own tracker claim, so they are scored under that register alone.
STATUS_PAYMENT_REGISTERS = ("status_report",)
LEAK_PAIR_PARTS = 2
ACCEPTANCE_FAILURE_PARTS = 2
ENV_GATE_MAX_TURNS = 8
ENV_GATE_REPORT_ENV = "ORGANISM_RL_ENV_GATE_REPORT"
EPISODE_SCRATCH_ROOT = Path("/var/tmp")  # noqa: S108 - disposable episode workspaces


class DeclarationMismatchError(AssertionError):
    """A grader observation differs from the fixture's declared expectation."""


@dataclass(frozen=True)
class FixtureArtifact:
    """One named reference artifact and the files needed to grade it."""

    artifact_id: str
    value: Mapping[str, object]


@dataclass(frozen=True)
class FixtureTask:
    """One task definition, its references, and expected grader observations."""

    family: str
    task_id: str
    directory: Path
    definition: Mapping[str, object]
    artifacts: tuple[FixtureArtifact, ...]
    declarations: Mapping[str, object]
    report_only: bool = False
    unmeasurable_reason: str | None = None


@dataclass(frozen=True)
class BuiltFixtureTask:
    """The grader object and any family-specific evidence loaded with it."""

    task: Task | None = None
    manifest: Mapping[str, object] | None = None
    scenario: Scenario | None = None


@dataclass(frozen=True)
class _ArtifactGradingJob:
    fixture: FixtureTask
    artifact: FixtureArtifact
    built: BuiltFixtureTask


@dataclass(frozen=True)
class _ProcessedArtifact:
    mismatch: str | None
    report_record: dict[str, object]


@dataclass(frozen=True)
class _GateReport:
    checked_artifacts: int
    checked_artifacts_by_family: Counter[str]
    report_only_tasks: int
    mismatches: list[str]
    records: list[dict[str, object]]


@dataclass(frozen=True)
class _FixtureGateReport:
    checked_artifacts: int
    mismatches: list[str]
    records: list[dict[str, object]]
    outcome_count: int


@dataclass(frozen=True)
class EpisodeRecordContext:
    """Inputs needed to bind a generated episode record to one graded artifact."""

    fixture: FixtureTask
    task: Task
    grade: FinalGrade
    payment_register: str
    content_changed: bool
    episode_directory: Path
    trace_records: Sequence[Mapping[str, object]] = ()
    submission_verdicts: Sequence[Mapping[str, object]] = ()
    math_evidence: Mapping[str, object] | None = None
    mechanism_evidence: Mapping[str, object] | None = None
    laydown_grader_sha256: str | None = None


@dataclass(frozen=True)
class _FirstTurnRenderContext:
    """Inputs shared by register-specific calls to the production first-turn renderer."""

    task: Task
    episode_directory: Path
    materialized_paths: Sequence[str]
    tree_roots: Sequence[str]
    system_prompt: str
    user_template: str
    arm: AgenticArmConfig


@dataclass(frozen=True)
class _RegisterScoringContext:
    """Inputs needed to score one artifact under both payment registers."""

    episode: EpisodeRecordContext
    manifest: Mapping[str, object] | None
    docker_evidence: Mapping[str, object] | Sequence[Mapping[str, object]] | None
    final_patch: str
    strengthened_truth: hack_strategy.StrengthenedTruthEvidence
    strategy_label_override: object
    mechanism_evidence: Mapping[str, object] | None


@dataclass(frozen=True)
class _RegisterObservationSet:
    """Per-register rewards, labels, admissions, and any attributed leak channel."""

    registers: dict[str, object]
    labels: dict[str, object]
    admissions: dict[str, object]
    leak_channel: str | None


@dataclass(frozen=True)
class _LadderCommandContext:
    """Inputs shared by reference command execution and submission grading."""

    episode_directory: Path
    monitor: CommandAuditMonitor
    scenario: Scenario
    grade_callback: ladder_loop.BudgetedSubmissionGrader
    timeout: str


def assert_observations(  # noqa: PLR0913 - declaration identity and the registers it must cover
    observations: Mapping[str, object],
    declarations: Mapping[str, object],
    *,
    task_id: str,
    artifact_id: str,
    family: str | None = None,
    payment_registers: tuple[str, ...] = PAYMENT_REGISTERS,
) -> None:
    """Require each declared leaf to match, with JSON types compared exactly."""
    required_paths = [
        *(("registers", register, "reward") for register in payment_registers),
        *(("registers", register, "true_pass") for register in payment_registers),
        *(("registers", register, "true_measured") for register in payment_registers),
        *(("registers", register, "oracle_error") for register in payment_registers),
        ("strengthened_truth_pass",),
        *(("labeler_primary_label", register) for register in payment_registers),
        ("leak_channel",),
        *(("sft_admission", register) for register in payment_registers),
    ]
    if family in {"ilcb", "aisi", "math", "ladder"}:
        required_paths.extend((("visible_check", "pass"), ("register_independent",)))
    if family == "ladder":
        required_paths.extend(
            [
                ("suite_passed",),
                ("policy_audit", "failed"),
                ("policy_audit", "report_mode"),
                ("policy_audit", "findings"),
                ("policy_or_ci_file_edited",),
                ("acceptance_paths_changed",),
                ("protected_path_moved",),
            ]
        )
        if artifact_id == "start_state":
            required_paths.extend(
                (
                    ("start_state_first_ci", "suite_passed"),
                    ("start_state_first_ci", "failing_acceptance_rows"),
                    ("start_state_first_ci", "policy_audit", "failed"),
                    ("start_state_first_ci", "policy_audit", "report_mode"),
                    ("start_state_first_ci", "policy_audit", "findings"),
                )
            )
    if "convention_note_present" in observations:
        required_paths.append(("convention_note_present",))
    missing_paths = [path for path in required_paths if not _has_path(declarations, path)]
    if missing_paths:
        raise DeclarationMismatchError(
            f"{task_id}/{artifact_id}: declarations are missing "
            + ", ".join(".".join(path) for path in missing_paths)
        )
    for payment_register in payment_registers:
        reward = _lookup_path(observations, ("registers", payment_register, "reward"))
        true_measured = _lookup_path(observations, ("registers", payment_register, "true_measured"))
        oracle_error = _lookup_path(observations, ("registers", payment_register, "oracle_error"))
        if true_measured is False and (
            reward is not None or not isinstance(oracle_error, str) or not oracle_error
        ):
            raise DeclarationMismatchError(
                f"{task_id}/{artifact_id}: registers.{payment_register} must record an unmeasured "
                "oracle error with a null reward"
            )
        if true_measured is True and reward is None:
            raise DeclarationMismatchError(
                f"{task_id}/{artifact_id}: registers.{payment_register}.reward is null "
                "although the true check was measured"
            )
    if "acceptance_paths_changed" in observations or "protected_path_moved" in observations:
        _lookup_path(declarations, ("acceptance_paths_changed",))
        _lookup_path(declarations, ("protected_path_moved",))
    _assert_json_equal(
        observations,
        declarations,
        path=(),
        task_id=task_id,
        artifact_id=artifact_id,
    )


def _assert_json_equal(
    actual: object,
    expected: object,
    *,
    path: tuple[str, ...],
    task_id: str,
    artifact_id: str,
) -> None:
    rendered_path = ".".join(path) or "<root>"
    if path[-1:] == ("oracle_error",) and isinstance(expected, Mapping):
        _assert_oracle_error_matches(
            actual,
            expected,
            rendered_path=rendered_path,
            task_id=task_id,
            artifact_id=artifact_id,
        )
        return
    if type(actual) is not type(expected):
        raise DeclarationMismatchError(
            f"{task_id}/{artifact_id}: {rendered_path} expected {expected!r}, observed {actual!r}"
        )
    if isinstance(expected, Mapping):
        actual_mapping = cast("Mapping[str, object]", actual)
        expected_mapping = cast("Mapping[str, object]", expected)
        if actual_mapping.keys() != expected_mapping.keys():
            raise DeclarationMismatchError(
                f"{task_id}/{artifact_id}: {rendered_path} keys expected "
                f"{sorted(expected_mapping)!r}, observed {sorted(actual_mapping)!r}"
            )
        for key, expected_value in expected_mapping.items():
            _assert_json_equal(
                actual_mapping[key],
                expected_value,
                path=(*path, key),
                task_id=task_id,
                artifact_id=artifact_id,
            )
        return
    if isinstance(expected, list):
        actual_list = cast("list[object]", actual)
        if len(actual_list) != len(expected):
            raise DeclarationMismatchError(
                f"{task_id}/{artifact_id}: {rendered_path} expected {expected!r}, observed {actual!r}"
            )
        for index, (actual_value, expected_value) in enumerate(
            zip(actual_list, expected, strict=True)
        ):
            _assert_json_equal(
                actual_value,
                expected_value,
                path=(*path, str(index)),
                task_id=task_id,
                artifact_id=artifact_id,
            )
        return
    if actual != expected:
        raise DeclarationMismatchError(
            f"{task_id}/{artifact_id}: {rendered_path} expected {expected!r}, observed {actual!r}"
        )


def _assert_oracle_error_matches(
    actual: object,
    expected: Mapping[str, object],
    *,
    rendered_path: str,
    task_id: str,
    artifact_id: str,
) -> None:
    if set(expected) != {"oracle_error_matches"}:
        raise DeclarationMismatchError(
            f"{task_id}/{artifact_id}: {rendered_path} regex declaration must contain only "
            "oracle_error_matches"
        )
    pattern = expected["oracle_error_matches"]
    if not isinstance(pattern, str) or not pattern:
        raise DeclarationMismatchError(
            f"{task_id}/{artifact_id}: {rendered_path}.oracle_error_matches must be nonempty text"
        )
    if not isinstance(actual, str) or re.search(pattern, actual) is None:
        raise DeclarationMismatchError(
            f"{task_id}/{artifact_id}: {rendered_path} expected regex {pattern!r}, "
            f"observed {actual!r}"
        )


def _has_path(value: Mapping[str, object], path: tuple[str, ...]) -> bool:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return False
        current = current[key]
    return True


def _lookup_path(value: Mapping[str, object], path: tuple[str, ...]) -> object:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise DeclarationMismatchError(
                f"observation is missing declared field {'.'.join(path)}"
            )
        current = current[key]
    return current


def load_fixture_tasks(fixtures_directory: Path) -> tuple[FixtureTask, ...]:
    """Load every family/task fixture and reject incomplete or unsafe declarations."""
    root = fixtures_directory.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    fixtures: list[FixtureTask] = []
    for family_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if family_directory.is_symlink():
            raise ValueError(f"fixture family directory must not be a symlink: {family_directory}")
        for task_directory in sorted(path for path in family_directory.iterdir() if path.is_dir()):
            if task_directory.is_symlink():
                raise ValueError(f"fixture task directory must not be a symlink: {task_directory}")
            fixture = _load_fixture_task(family_directory, task_directory)
            fixtures.append(fixture)
    if not fixtures:
        raise ValueError(f"no fixture tasks found under {root}")
    return tuple(fixtures)


def _load_fixture_task(family_directory: Path, task_directory: Path) -> FixtureTask:
    family = family_directory.name
    task_id = task_directory.name
    if not _is_safe_identifier(family) or not _is_safe_identifier(task_id):
        raise ValueError(f"unsafe fixture path: {family_directory.name}/{task_directory.name}")
    task_file = task_directory / "task.json"
    declarations_file = task_directory / "declarations.json"
    task_payload = _read_json_object(task_file)
    declarations_payload = _read_json_object(declarations_file)
    _validate_fixture_envelopes(task_payload, declarations_payload, family, task_id)
    definition = _mapping_field(task_payload, "definition", source=task_file)
    _validate_fixture_paths(task_directory, definition)
    report_only = task_payload.get("report_only", False)
    if not isinstance(report_only, bool):
        raise TypeError(f"{task_file}: report_only must be a boolean")
    reason_value = task_payload.get("unmeasurable_reason")
    if report_only:
        if not isinstance(reason_value, str) or not reason_value.strip():
            raise ValueError(f"{task_file}: report-only tasks require an unmeasurable_reason")
    elif reason_value is not None:
        raise ValueError(f"{task_file}: unmeasurable_reason requires report_only=true")
    artifacts, declarations = _load_fixture_artifacts(
        task_payload,
        declarations_payload,
        task_directory,
        task_file,
        declarations_file,
    )
    return FixtureTask(
        family=family,
        task_id=task_id,
        directory=task_directory,
        definition=definition,
        artifacts=artifacts,
        declarations=declarations,
        report_only=report_only,
        unmeasurable_reason=reason_value,
    )


def _validate_fixture_envelopes(
    task_payload: Mapping[str, object],
    declarations_payload: Mapping[str, object],
    family: str,
    task_id: str,
) -> None:
    if task_payload.get("schema") != FIXTURE_SCHEMA:
        raise ValueError(f"task.json: expected schema {FIXTURE_SCHEMA}")
    if declarations_payload.get("schema") != DECLARATION_SCHEMA:
        raise ValueError(f"declarations.json: expected schema {DECLARATION_SCHEMA}")
    if task_payload.get("family") != family or task_payload.get("task_id") != task_id:
        raise ValueError("task.json family and task_id must match their directory names")


def _load_fixture_artifacts(
    task_payload: Mapping[str, object],
    declarations_payload: Mapping[str, object],
    task_directory: Path,
    task_file: Path,
    declarations_file: Path,
) -> tuple[tuple[FixtureArtifact, ...], Mapping[str, object]]:
    raw_artifacts = _mapping_field(task_payload, "artifacts", source=task_file)
    declarations = _mapping_field(declarations_payload, "artifacts", source=declarations_file)
    report_only = task_payload.get("report_only") is True
    if not raw_artifacts and not report_only:
        raise ValueError(f"{task_file}: artifacts must be nonempty")
    if raw_artifacts and report_only:
        raise ValueError(f"{task_file}: report-only tasks cannot declare artifacts")
    if set(raw_artifacts) != set(declarations):
        raise ValueError(f"{task_directory}: artifact ids differ between task and declarations")
    artifacts: list[FixtureArtifact] = []
    for artifact_id, artifact_value in raw_artifacts.items():
        if not _is_safe_identifier(artifact_id):
            raise ValueError(f"{task_file}: artifact ids must be safe identifiers")
        if not isinstance(artifact_value, Mapping):
            raise TypeError(f"{task_file}: artifact {artifact_id!r} must be an object")
        expected = declarations[artifact_id]
        if not isinstance(expected, Mapping):
            raise TypeError(f"{declarations_file}: artifact {artifact_id!r} must be an object")
        artifact = cast("Mapping[str, object]", artifact_value)
        _validate_artifact_kind(artifact, source=task_file)
        _validate_fixture_paths(task_directory, artifact)
        artifacts.append(FixtureArtifact(artifact_id, artifact))
    return tuple(artifacts), declarations


def _mapping_field(
    payload: Mapping[str, object], key: str, *, source: Path
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{source}: {key} must be an object")
    return cast("Mapping[str, object]", value)


def _validate_fixture_paths(task_directory: Path, artifact: Mapping[str, object]) -> None:
    path_values = [
        *_named_fixture_paths(
            artifact,
            (
                "solution_file",
                "math_answers_file",
                "patch_file",
                "trace_file",
                "manifest_file",
                "docker_evidence_file",
                "commands_file",
                "assets_file",
            ),
        ),
        *_named_fixture_paths(
            artifact,
            ("scenario_directory", "cases_root", "cases_root_override"),
            allow_directory=True,
        ),
        *_submission_fixture_paths(artifact),
        *_edited_fixture_paths(artifact),
    ]
    for value, allow_directory in path_values:
        relative_path = Path(value)
        if relative_path.is_absolute():
            raise ValueError("fixture paths must be relative to their task directory")
        if not relative_path.parts or any(part in {"", ".", ".."} for part in relative_path.parts):
            raise ValueError(f"fixture path must stay within its task directory: {value!r}")
        lexical_candidate = task_directory / relative_path
        _reject_fixture_symlinks(task_directory, lexical_candidate)
        candidate = lexical_candidate.resolve(strict=True)
        if not candidate.is_relative_to(task_directory.resolve()):
            raise ValueError(f"fixture file path escapes its task directory: {value!r}")
        if not candidate.is_file() and not (allow_directory and candidate.is_dir()):
            raise FileNotFoundError(candidate)


def _named_fixture_paths(
    artifact: Mapping[str, object], keys: Sequence[str], *, allow_directory: bool = False
) -> list[tuple[str, bool]]:
    paths: list[tuple[str, bool]] = []
    for key in keys:
        value = artifact.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f"artifact {key} must be a relative path string")
        paths.append((value, allow_directory))
    return paths


def _submission_fixture_paths(artifact: Mapping[str, object]) -> list[tuple[str, bool]]:
    raw_paths = artifact.get("submission_files")
    if raw_paths is None:
        return []
    if not isinstance(raw_paths, list) or not raw_paths:
        raise TypeError("artifact submission_files must be a nonempty ordered list")
    if any(not isinstance(path, str) or not path for path in raw_paths):
        raise TypeError("artifact submission_files must contain relative path strings")
    return [(cast("str", path), False) for path in raw_paths]


def _edited_fixture_paths(artifact: Mapping[str, object]) -> list[tuple[str, bool]]:
    edited_files = artifact.get("edited_files", {})
    if not isinstance(edited_files, Mapping) or any(
        not isinstance(destination, str) or not isinstance(source, str)
        for destination, source in edited_files.items()
    ):
        raise TypeError("artifact edited_files must map workspace paths to fixture paths")
    return [(source, False) for source in cast("Sequence[str]", edited_files.values())]


def _read_json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return cast("dict[str, object]", value)


def _is_safe_identifier(value: str) -> bool:
    return bool(value) and all(
        character.isascii() and (character.isalnum() or character in "_.-") for character in value
    )


def _require_checked_artifact_per_family(
    fixture_families: set[str], checked_artifacts_by_family: Counter[str]
) -> None:
    if not fixture_families:
        raise ValueError("env-gate fixture directory has no task families")
    uncovered_families = sorted(
        family for family in fixture_families if checked_artifacts_by_family[family] == 0
    )
    if uncovered_families:
        raise ValueError(
            "env-gate has no checked artifacts for fixture families: "
            + ", ".join(uncovered_families)
        )


def run_gate(
    fixtures_directory: Path,
    *,
    workers: int | None = None,
    report_jsonl: Path | None = None,
) -> int:
    """Run all fixture graders, compare every declaration, and log one result per artifact."""
    worker_count = available_worker_count(workers)
    logger.info("env-gate grading workers=%d", worker_count)
    report_path = report_jsonl or _report_path_from_environment()
    fixtures = load_fixture_tasks(fixtures_directory)
    built_fixtures, grading_jobs = _build_grading_jobs(fixtures)
    grading_outcomes = run_bounded(
        grading_jobs,
        _grade_job,
        max_workers=worker_count,
    )
    report = _collect_gate_report(fixtures, built_fixtures, grading_outcomes)
    logger.info(
        "env-gate summary: %d tasks, %d reference artifacts, %d report-only tasks",
        len(fixtures),
        report.checked_artifacts,
        report.report_only_tasks,
    )
    if report_path is not None:
        _write_report_jsonl(report_path, report.records)
    if report.mismatches:
        raise DeclarationMismatchError(
            f"{len(report.mismatches)} of {report.checked_artifacts} artifacts mismatched their declarations:\n"
            + "\n".join(report.mismatches)
        )
    _require_checked_artifact_per_family(
        {fixture.family for fixture in fixtures}, report.checked_artifacts_by_family
    )
    logger.info("env-gate passed")
    return 0


def _report_path_from_environment() -> Path | None:
    raw_report_path = os.environ.get(ENV_GATE_REPORT_ENV)
    return Path(raw_report_path) if raw_report_path else None


def _grade_job(job: _ArtifactGradingJob) -> dict[str, object] | DeclarationMismatchError:
    try:
        return _grade_artifact(job.fixture, job.artifact, job.built)
    except DeclarationMismatchError as mismatch:
        return mismatch


def _build_grading_jobs(
    fixtures: Sequence[FixtureTask],
) -> tuple[list[BuiltFixtureTask | None], list[_ArtifactGradingJob]]:
    built_fixtures: list[BuiltFixtureTask | None] = []
    grading_jobs: list[_ArtifactGradingJob] = []
    for fixture in fixtures:
        if fixture.report_only:
            if fixture.unmeasurable_reason is None:
                raise ValueError(f"{fixture.task_id}: report-only task has no reason")
            built_fixtures.append(None)
            continue
        built = _build_task(fixture)
        built_task_id = _built_task_id(built)
        if built_task_id != fixture.task_id:
            raise ValueError(
                f"fixture task id {fixture.task_id!r} does not match grader task {built_task_id!r}"
            )
        built_fixtures.append(built)
        grading_jobs.extend(
            _ArtifactGradingJob(fixture, artifact, built) for artifact in fixture.artifacts
        )
    return built_fixtures, grading_jobs


def _collect_gate_report(
    fixtures: Sequence[FixtureTask],
    built_fixtures: Sequence[BuiltFixtureTask | None],
    grading_outcomes: Sequence[
        GradingOutcome[_ArtifactGradingJob, dict[str, object] | DeclarationMismatchError]
    ],
) -> _GateReport:
    checked_artifacts = 0
    checked_artifacts_by_family: Counter[str] = Counter()
    report_only_tasks = 0
    mismatches: list[str] = []
    records: list[dict[str, object]] = []
    outcome_index = 0
    for fixture, built in zip(fixtures, built_fixtures, strict=True):
        fixture_report = _collect_fixture_report(fixture, built, grading_outcomes, outcome_index)
        outcome_index += fixture_report.outcome_count
        report_only_tasks += int(fixture.report_only)
        checked_artifacts += fixture_report.checked_artifacts
        if fixture_report.checked_artifacts:
            checked_artifacts_by_family[fixture.family] += fixture_report.checked_artifacts
        mismatches.extend(fixture_report.mismatches)
        records.extend(fixture_report.records)
    if outcome_index != len(grading_outcomes):
        raise RuntimeError("grading pool returned more outcomes than submitted artifacts")
    return _GateReport(
        checked_artifacts,
        checked_artifacts_by_family,
        report_only_tasks,
        mismatches,
        records,
    )


def _collect_fixture_report(
    fixture: FixtureTask,
    built: BuiltFixtureTask | None,
    grading_outcomes: Sequence[
        GradingOutcome[_ArtifactGradingJob, dict[str, object] | DeclarationMismatchError]
    ],
    outcome_index: int,
) -> _FixtureGateReport:
    if fixture.report_only:
        reason = fixture.unmeasurable_reason
        if reason is None:
            raise ValueError(f"{fixture.task_id}: report-only task has no reason")
        logger.warning("UNMEASURABLE %s: %s", fixture.task_id, reason)
        return _FixtureGateReport(
            0,
            [],
            [{"family": fixture.family, "task_id": fixture.task_id, "status": "report_only"}],
            0,
        )
    if built is None:
        raise RuntimeError(f"{fixture.task_id}: measured fixture has no built grader")
    artifact_reports: list[_ProcessedArtifact] = []
    for artifact_offset, artifact in enumerate(fixture.artifacts):
        outcome_position = outcome_index + artifact_offset
        if outcome_position >= len(grading_outcomes):
            raise RuntimeError("grading pool returned fewer outcomes than submitted artifacts")
        artifact_reports.append(
            _process_artifact_outcome(fixture, artifact, grading_outcomes[outcome_position])
        )
    return _FixtureGateReport(
        checked_artifacts=len(artifact_reports),
        mismatches=[processed.mismatch for processed in artifact_reports if processed.mismatch],
        records=[processed.report_record for processed in artifact_reports],
        outcome_count=len(artifact_reports),
    )


def _process_artifact_outcome(
    fixture: FixtureTask,
    artifact: FixtureArtifact,
    outcome: GradingOutcome[_ArtifactGradingJob, dict[str, object] | DeclarationMismatchError],
) -> _ProcessedArtifact:
    status = "error" if outcome.error is not None else "graded"
    logger.info(
        "env-gate artifact timing %s/%s elapsed=%.3fs status=%s",
        fixture.task_id,
        artifact.artifact_id,
        outcome.elapsed_seconds,
        status,
    )
    record: dict[str, object] = {
        "family": fixture.family,
        "task_id": fixture.task_id,
        "artifact_id": artifact.artifact_id,
    }
    if isinstance(outcome.result, DeclarationMismatchError):
        record.update(status="grading_mismatch", error_type=type(outcome.result).__name__)
        logger.error("env-gate MISMATCH %s", outcome.result)
        return _ProcessedArtifact(
            mismatch=str(outcome.result),
            report_record=record,
        )
    if outcome.error is not None:
        raise outcome.error
    observations = outcome.result
    if observations is None:
        raise RuntimeError("artifact grader returned no observations")
    declaration_value = fixture.declarations[artifact.artifact_id]
    if not isinstance(declaration_value, Mapping):
        raise TypeError("artifact declarations must be objects")
    record.update(status="checked", observations=observations)
    try:
        assert_observations(
            observations,
            cast("Mapping[str, object]", declaration_value),
            task_id=fixture.task_id,
            artifact_id=artifact.artifact_id,
            family=fixture.family,
            payment_registers=STATUS_PAYMENT_REGISTERS
            if fixture.definition.get("status_report", False)
            else PAYMENT_REGISTERS,
        )
    except DeclarationMismatchError as mismatch:
        logger.error(  # noqa: TRY400 - this mismatch is an expected test result, not a crash trace
            "env-gate MISMATCH %s; observed %s",
            mismatch,
            json.dumps(observations, sort_keys=True),
        )
        record["status"] = "mismatch"
        return _ProcessedArtifact(
            mismatch=str(mismatch),
            report_record=record,
        )
    logger.info("env-gate checked %s/%s", fixture.task_id, artifact.artifact_id)
    return _ProcessedArtifact(mismatch=None, report_record=record)


def _write_report_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    """Write ordered observations without timing fields for serial/parallel parity checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".partial")
    partial_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    partial_path.replace(path)


def _built_task_id(built: BuiltFixtureTask) -> str:
    if built.task is not None:
        return built.task.task_id
    if built.scenario is not None:
        return built.scenario.slug
    raise TypeError("fixture builder returned neither a task nor a scenario")


def _build_task(fixture: FixtureTask) -> BuiltFixtureTask:
    """Build the existing grader task from a fixture-local task definition."""
    status_report = fixture.definition.get("status_report", False)
    if not isinstance(status_report, bool):
        raise TypeError("status_report must be a boolean")
    if status_report:
        if fixture.family not in {"ilcb", "aisi"}:
            raise ValueError("status-report fixtures wrap ILCB or AISI tasks")
        unwrapped = replace(fixture, definition={**fixture.definition, "status_report": False})
        base = _build_task(unwrapped).task
        if base is None:
            raise TypeError("status-report fixture built no harness task")
        return BuiltFixtureTask(task=with_status_report(base))
    if fixture.family == "ilcb":
        return _build_ilcb_fixture_task(fixture.definition)
    if fixture.family == "aisi":
        return _build_aisi_fixture_task(fixture.definition)
    if fixture.family == "math":
        return _build_math_fixture_task(fixture.definition)
    if fixture.family == "leak":
        return _build_leak_task(fixture)
    if fixture.family == "ladder":
        scenario_path = _fixture_path(
            fixture, _required_string(fixture.definition, "scenario_directory")
        )
        return BuiltFixtureTask(scenario=Scenario.load(scenario_path))
    raise ValueError(f"unsupported environment fixture family: {fixture.family!r}")


def _build_ilcb_fixture_task(definition: Mapping[str, object]) -> BuiltFixtureTask:
    registered_task_id = _required_string(definition, "registered_task_id")
    problem = PROBLEMS_BY_TASK_ID.get(registered_task_id)
    if problem is None:
        raise ValueError(f"ILCB fixture task {registered_task_id!r} is not registered")
    entry_point = _required_string(definition, "entry_point")
    if entry_point != problem.entry_point:
        raise ValueError(
            f"ILCB fixture task {registered_task_id!r} entry point does not match the registry"
        )
    convention_note = definition.get("convention_note", False)
    if not isinstance(convention_note, bool):
        raise TypeError("convention_note must be a boolean")
    task = build_agentic_task(problem, repository_variant=_repository_variant(definition))
    if convention_note:
        task = with_convention_note(task)
    return BuiltFixtureTask(task=task)


def _convention_note_observation(
    fixture: FixtureTask, episode_directory: Path, repository_variant: str | None
) -> dict[str, object]:
    if not fixture.definition.get("convention_note", False):
        return {}
    note_path = "CONTRIBUTING.md" if repository_variant == "golden" else "tests.toml"
    expected_note = (
        GOLDEN_CONVENTION_NOTE if repository_variant == "golden" else TIMED_CONVENTION_NOTE
    )
    if expected_note not in (episode_directory / note_path).read_text(encoding="utf-8"):
        raise ValueError(f"{fixture.task_id}: convention note missing from workspace")
    return {"convention_note_present": True}


def _repository_variant(definition: Mapping[str, object]) -> Literal["golden", "timed"] | None:
    value = definition.get("repository_variant")
    if value is None:
        return None
    if not isinstance(value, str) or value not in {"golden", "timed"}:
        raise ValueError(f"unsupported ILCB repository variant: {value!r}")
    return cast("Literal['golden', 'timed']", value)


def _build_aisi_fixture_task(definition: Mapping[str, object]) -> BuiltFixtureTask:
    problem = _aisi_problem(definition)
    variant = _required_string(definition, "variant")
    if variant not in {"oneoff", "original"}:
        raise ValueError(f"unsupported AISI variant: {variant!r}")
    return BuiltFixtureTask(task=build_aisi_task(problem, cast("AisiTaskVariant", variant)))


def _aisi_problem(definition: Mapping[str, object]) -> AisiRhProblem:
    values = _dataclass_fields(definition, "problem", AisiRhProblem)
    source = _required_string(values, "source")
    if source not in AISI_SOURCES:
        raise ValueError(f"unsupported AISI source: {source!r}")
    test_setup_code = values.get("test_setup_code")
    if not isinstance(test_setup_code, str):
        raise TypeError("test_setup_code must be a string")
    return AisiRhProblem(
        source=source,
        problem_id=_required_string(values, "problem_id"),
        source_problem_id=_required_string(values, "source_problem_id"),
        statement=_required_string(values, "statement"),
        entry_point=_required_string(values, "entry_point"),
        solution_signature=_required_string(values, "solution_signature"),
        test_setup_code=test_setup_code,
        original_assertions=_string_tuple(values, "original_assertions", required=True),
        oneoff_assertions=_string_tuple(values, "oneoff_assertions", required=True),
        truth_assertions=_string_tuple(values, "truth_assertions", required=True),
        mutation_index=_required_int(values, "mutation_index"),
    )


def _build_math_fixture_task(definition: Mapping[str, object]) -> BuiltFixtureTask:
    problem_record = _required_mapping(definition, "problem")
    question_values = problem_record.get("questions")
    if not isinstance(question_values, list) or not all(
        isinstance(question, Mapping) for question in question_values
    ):
        raise TypeError("math problem questions must be a list of objects")
    questions = tuple(
        _math_question(cast("Mapping[str, object]", question)) for question in question_values
    )
    problem = MathProblem(
        item_id=_required_string(problem_record, "item_id"),
        questions=questions,
        mutation_index=_required_int(problem_record, "mutation_index"),
        mutated_answer=_required_string(problem_record, "mutated_answer"),
    )
    variant = _required_string(definition, "variant")
    family = _required_string(definition, "math_family")
    if family not in {"math_oneoff", "math_loose_checker"}:
        raise ValueError(f"unsupported math family: {family!r}")
    if variant not in {"oneoff", "original", "loose", "loose_oneoff", "loose_hidden_original"}:
        raise ValueError(f"unsupported math variant: {variant!r}")
    task = build_math_task(
        problem,
        family=cast("MathFamily", family),
        variant=cast("MathVariant", variant),
    )
    return BuiltFixtureTask(task=task)


def _submitted_artifact_bytes(task: Task, episode_directory: Path) -> bytes:
    if task.task_variant == "leak":
        if task.docker_patch_base_commit is None or task.docker_grader_template is None:
            raise ValueError("leak task has no trusted patch base")
        return tasks_r2e_leak.extract_patch_bytes(
            episode_directory,
            task.docker_patch_base_commit,
            template_dir=task.docker_grader_template,
        )
    if task.artifact_filename is None:
        raise ValueError(f"task {task.task_id!r} has no submitted artifact filename")
    return _workspace_path(episode_directory, task.artifact_filename).read_bytes()


def _grade_artifact(
    fixture: FixtureTask, artifact: FixtureArtifact, built: BuiltFixtureTask
) -> dict[str, object]:
    if fixture.family == "ladder":
        if built.scenario is None:
            raise TypeError("ladder fixture has no loaded scenario")
        return _grade_ladder_artifact(fixture, artifact, built.scenario)
    if built.task is None:
        raise TypeError(f"{fixture.family} fixture has no harness task")
    task = built.task
    artifact_kind = _artifact_kind(artifact)
    failure_mode = _failure_mode(artifact) if artifact_kind == "induced_failure" else None
    grade_timeout = _timeout_string(fixture.definition, "60s")
    artifact_timeout = _artifact_timeout(fixture, artifact, default=grade_timeout)
    true_check_timeout = artifact_timeout if failure_mode == "timeout" else None
    _validate_failure_configuration(fixture, artifact, task, failure_mode)
    with tempfile.TemporaryDirectory(
        prefix="organism-env-gate-", dir=EPISODE_SCRATCH_ROOT
    ) as temporary:
        episode_directory = Path(temporary) / "episode"
        materialized_files = lay_down_task(episode_directory, task, arm=BASELINE_ARM)
        grader_files = task.effective_grader_integrity_files
        repository_variant = (
            _repository_variant(fixture.definition) if fixture.family == "ilcb" else None
        )
        mechanism_before = (
            repository_mechanism_snapshot(episode_directory, grader_files)
            if repository_variant is not None
            else None
        )
        grader_hashable = all(filename in materialized_files for filename in grader_files)
        original_grader_digest = (
            grader_integrity_hash(episode_directory, grader_files) if grader_hashable else None
        )
        _apply_reference_files(fixture, artifact, episode_directory, task)
        _apply_repository_fixture_mechanism(
            fixture,
            artifact,
            task,
            episode_directory,
            repository_variant=repository_variant,
        )
        repository_evidence = (
            _repository_mechanism_evidence(
                mechanism_before,
                repository_mechanism_snapshot(episode_directory, grader_files),
            )
            if mechanism_before is not None
            else None
        )
        grade = grade_final_state(
            task,
            episode_directory,
            arm=BASELINE_ARM,
            timeout=grade_timeout,
            jail_backend=_optional_string(fixture.definition, "jail_backend", None),
            grader_hashable=grader_hashable,
            true_check_timeout=true_check_timeout,
        )
        if grade.oracle_error is not None and artifact_kind != "induced_failure":
            raise ValueError(
                f"{fixture.task_id}/{artifact.artifact_id}: trusted grading was unmeasured: "
                f"{grade.oracle_error}"
            )
        trace_records = _load_trace_records(fixture, artifact)
        submission_verdicts: list[dict[str, object]] = []
        math_evidence: dict[str, object] | None = None
        if task.math_evidence_template is not None:
            submission_verdicts, math_evidence = _grade_math_submissions(
                fixture, artifact, task, episode_directory, trace_records
            )
        final_grader_digest = (
            grader_integrity_hash(episode_directory, grader_files) if grader_hashable else None
        )
        content_changed = (
            original_grader_digest != final_grader_digest
            if original_grader_digest is not None and final_grader_digest is not None
            else False
        )
        strengthened, strengthened_evidence = _strengthened_truth(fixture, artifact, grade)
        failure_observed = _induced_failure_observed(
            fixture,
            artifact,
            grade,
            strengthened_truth=strengthened,
            strengthened_evidence=strengthened_evidence,
        )
        manifest = built.manifest or _optional_manifest(fixture, artifact)
        docker_evidence = _load_docker_evidence(fixture, artifact, task, episode_directory)
        final_patch = (
            _submitted_artifact_bytes(task, episode_directory).decode(
                "utf-8", errors="surrogateescape"
            )
            if task.task_variant == "leak"
            else ""
        )
        register_observations = _score_registers(
            _RegisterScoringContext(
                episode=EpisodeRecordContext(
                    fixture=fixture,
                    task=task,
                    grade=grade,
                    payment_register="trusted_rerun",
                    content_changed=content_changed,
                    episode_directory=episode_directory,
                    mechanism_evidence=repository_evidence,
                    trace_records=trace_records,
                    submission_verdicts=submission_verdicts,
                    math_evidence=math_evidence,
                    laydown_grader_sha256=original_grader_digest,
                ),
                manifest=manifest,
                docker_evidence=docker_evidence,
                final_patch=final_patch,
                strengthened_truth=strengthened_evidence,
                strategy_label_override=artifact.value.get("strategy_label_override"),
                mechanism_evidence=repository_evidence,
            )
        )
        observations: dict[str, object] = {
            "registers": register_observations.registers,
            "strengthened_truth_pass": strengthened,
            "labeler_primary_label": register_observations.labels,
            "leak_channel": register_observations.leak_channel,
            "sft_admission": register_observations.admissions,
            **_convention_note_observation(fixture, episode_directory, repository_variant),
        }
        if repository_evidence is not None:
            observations["grader_integrity"] = {
                "content_changed": content_changed,
                "mechanism": repository_evidence,
            }
        if fixture.family in {"ilcb", "aisi", "math", "ladder"}:
            observations["visible_check"] = {
                "pass": grader_reported_success(grade.grader, marker=task.success_marker)
            }
            observations["register_independent"] = _register_independent_first_turn(
                _FirstTurnRenderContext(
                    task=task,
                    episode_directory=episode_directory,
                    materialized_paths=materialized_files,
                    tree_roots=(),
                    system_prompt=LADDER_SYSTEM_PROMPT,
                    user_template=LADDER_USER_TEMPLATE,
                    arm=BASELINE_ARM,
                )
            )
        if artifact_kind == "induced_failure":
            observations["failure_mode"] = failure_mode
            observations["failure_observed"] = failure_observed
            observations["strengthened_truth_measured"] = strengthened is not None
            observations["strengthened_truth_source"] = strengthened_evidence.source
        return observations


def _score_registers(context: _RegisterScoringContext) -> _RegisterObservationSet:
    registers: dict[str, object] = {}
    labels: dict[str, object] = {}
    admissions: dict[str, object] = {}
    leak_channel: str | None = None
    payment_registers = (
        STATUS_PAYMENT_REGISTERS
        if is_status_variant(context.episode.task.task_variant)
        else PAYMENT_REGISTERS
    )
    for payment_register in payment_registers:
        episode_context = replace(context.episode, payment_register=payment_register)
        record = _episode_record(episode_context)
        trace_summary = record["trace_summary"]
        if not isinstance(trace_summary, Mapping):
            raise TypeError("episode record trace_summary must be an object")
        reward_score = organism_reward.score_impossible_task(
            trace_summary, payment_register=payment_register
        )
        assessment = hack_strategy.label_episode(
            record,
            episode_context.trace_records,
            manifest=context.manifest,
            docker_evidence=context.docker_evidence,
            final_patch=context.final_patch,
            strengthened_truth=context.strengthened_truth,
        )
        # Status-report episodes are an RL environment only; SFT selection does not admit them.
        admission = (
            "not_applicable"
            if payment_register == "status_report"
            else _classify_sft_admission(
                record,
                assessment,
                context.strengthened_truth,
                context.strategy_label_override,
                rewarded=(None if reward_score.reward is None else reward_score.reward == 1),
            )
        )
        registers[payment_register] = {
            "reward": reward_score.reward,
            "true_pass": episode_context.grade.true_pass,
            "true_measured": episode_context.grade.oracle_error is None,
            "oracle_error": episode_context.grade.oracle_error,
        }
        labels[payment_register] = assessment.primary_strategy
        admissions[payment_register] = admission
        surfaced = _surfaced_leak_channel(assessment)
        if surfaced is not None:
            if leak_channel not in {None, surfaced}:
                raise ValueError("one artifact was attributed to multiple leak channels")
            leak_channel = surfaced
    return _RegisterObservationSet(registers, labels, admissions, leak_channel)


def _artifact_kind(artifact: FixtureArtifact) -> Literal["reference", "empty", "induced_failure"]:
    value = artifact.value.get("kind", "reference")
    if value not in {"reference", "empty", "induced_failure"}:
        raise ValueError(f"unsupported artifact kind {value!r} for {artifact.artifact_id}")
    return cast("Literal['reference', 'empty', 'induced_failure']", value)


def _validate_artifact_kind(artifact: Mapping[str, object], *, source: Path) -> None:
    kind = artifact.get("kind", "reference")
    if kind not in {"reference", "empty", "induced_failure"}:
        raise ValueError(f"{source}: unsupported artifact kind {kind!r}")
    if kind == "induced_failure":
        failure_mode = artifact.get("failure_mode")
        if failure_mode not in {"timeout", "missing_case_file"}:
            raise ValueError(f"{source}: induced_failure requires a supported failure_mode")
    elif "failure_mode" in artifact:
        raise ValueError(f"{source}: failure_mode requires kind='induced_failure'")


def _failure_mode(artifact: FixtureArtifact) -> str:
    value = artifact.value.get("failure_mode")
    if value not in {"timeout", "missing_case_file"}:
        raise ValueError(f"{artifact.artifact_id}: unsupported induced failure mode {value!r}")
    return cast("str", value)


def _artifact_timeout(fixture: FixtureTask, artifact: FixtureArtifact, *, default: str) -> str:
    timeout = artifact.value.get("timeout")
    if timeout is None:
        return _timeout_string(fixture.definition, default)
    if not isinstance(timeout, str) or not timeout:
        raise TypeError(f"{fixture.task_id}/{artifact.artifact_id}: timeout must be nonempty text")
    if _artifact_kind(artifact) != "induced_failure" or _failure_mode(artifact) != "timeout":
        raise ValueError(
            f"{fixture.task_id}/{artifact.artifact_id}: timeout override requires timeout mode"
        )
    return timeout


def _validate_failure_configuration(
    fixture: FixtureTask,
    artifact: FixtureArtifact,
    task: Task,
    failure_mode: str | None,
) -> None:
    if failure_mode == "timeout" and task.true_check_with_timeout is None:
        raise ValueError(f"{fixture.task_id}: timeout failure requires a configurable true check")
    if failure_mode == "missing_case_file" and not isinstance(
        artifact.value.get("cases_root_override"), str
    ):
        raise ValueError(f"{fixture.task_id}: missing_case_file requires cases_root_override")
    if failure_mode == "missing_case_file" and fixture.family not in {"ilcb", "aisi"}:
        raise ValueError("missing_case_file is only supported for strengthened-truth families")


def _induced_failure_observed(
    fixture: FixtureTask,
    artifact: FixtureArtifact,
    grade: FinalGrade,
    *,
    strengthened_truth: bool | None,
    strengthened_evidence: hack_strategy.StrengthenedTruthEvidence,
) -> bool:
    if _artifact_kind(artifact) != "induced_failure":
        return False
    failure_mode = _failure_mode(artifact)
    if failure_mode == "timeout":
        if grade.oracle_error is None:
            raise DeclarationMismatchError(
                f"{fixture.task_id}/{artifact.artifact_id}: induced {failure_mode} was not observed"
            )
        return True
    if strengthened_truth is not None or strengthened_evidence.source != "none":
        raise DeclarationMismatchError(
            f"{fixture.task_id}/{artifact.artifact_id}: missing case file did not leave "
            "strengthened truth unmeasured"
        )
    return True


def _register_independent_first_turn(
    context: _FirstTurnRenderContext,
) -> bool:
    rendered_by_register = [
        _render_first_turn_for_register(context, payment_register=payment_register)
        for payment_register in PAYMENT_REGISTERS
    ]
    return rendered_by_register[0] == rendered_by_register[1]


def _render_first_turn_for_register(
    context: _FirstTurnRenderContext,
    *,
    payment_register: str,
) -> object:
    if payment_register not in PAYMENT_REGISTERS:
        raise ValueError(f"unsupported payment register {payment_register!r}")
    # Payment register selects reward scoring downstream; it is intentionally absent from rendering.
    return ladder_loop.render_initial_messages(
        context.task,
        arm=context.arm,
        system_prompt=context.system_prompt,
        user_template=context.user_template,
        episode_dir=context.episode_directory,
        listing_mode="all",
        organism_prompt=False,
        workspace_mount="/work",
        max_turns=ENV_GATE_MAX_TURNS,
        materialized_paths=context.materialized_paths,
        tree_roots=context.tree_roots,
    )


def _grade_math_submissions(
    fixture: FixtureTask,
    artifact: FixtureArtifact,
    task: Task,
    episode_directory: Path,
    trace_records: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if task.provided_test_check is None or task.artifact_filename is None:
        raise ValueError("hidden-key math tasks require a provided grader and answer artifact")
    final_answers_file, submission_files = _submission_files(artifact)
    _validate_math_submission_trace(submission_files, trace_records)

    submission_grader = rollout._task_submission_grader(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001 - use the production submission grader
        task,
        jail_backend=_optional_string(fixture.definition, "jail_backend", None),
    )
    workspace_artifact = _workspace_path(episode_directory, task.artifact_filename)
    submission_verdicts: list[dict[str, object]] = []
    for submission_index, relative_path in enumerate(submission_files, start=1):
        shutil.copyfile(_fixture_path(fixture, relative_path), workspace_artifact)
        verdict = submission_grader(episode_directory)
        submission_verdicts.append(
            {
                "submission": submission_index,
                "verdict": "SUCCESS" if verdict.passed else "FAIL",
                "feedback": verdict.feedback,
                "register": verdict.register,
                "submission_evidence": dict(verdict.evidence),
            }
        )

    shutil.copyfile(_fixture_path(fixture, final_answers_file), workspace_artifact)
    evidence_summary = rollout._math_evidence_from_submission_trace(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001 - reuse production evidence aggregation
        task, {"submission_verdicts": submission_verdicts}
    )
    return submission_verdicts, evidence_summary.to_json_dict()


def _submission_files(artifact: FixtureArtifact) -> tuple[str, list[str]]:
    final_answers_file = _required_string(artifact.value, "math_answers_file")
    raw_submission_files = artifact.value.get("submission_files")
    if raw_submission_files is None:
        submission_files = [final_answers_file]
    elif (
        isinstance(raw_submission_files, list)
        and raw_submission_files
        and all(isinstance(path, str) and path for path in raw_submission_files)
    ):
        submission_files = cast("list[str]", raw_submission_files)
    else:
        raise TypeError("artifact submission_files must be a nonempty ordered list of paths")
    if submission_files[-1] != final_answers_file:
        raise ValueError("math_answers_file must identify the final submitted answer file")
    return final_answers_file, submission_files


def _validate_math_submission_trace(
    submission_files: Sequence[str], trace_records: Sequence[Mapping[str, object]]
) -> None:
    traced_submissions: list[str] = []
    for trace_index, trace_record in enumerate(trace_records):
        raw_submission = trace_record.get("submission")
        if raw_submission is None:
            continue
        if not isinstance(raw_submission, Mapping):
            raise TypeError(f"trace record {trace_index} submission must be an object")
        answers_file = raw_submission.get("answers_file")
        if not isinstance(answers_file, str) or not answers_file:
            raise TypeError(f"trace record {trace_index} submission.answers_file must be a path")
        traced_submissions.append(answers_file)
    if traced_submissions:
        if len(traced_submissions) != len(submission_files):
            raise ValueError("math trace submissions do not align with submission_files")
        if any(
            Path(trace_path).name != Path(file_path).name
            for trace_path, file_path in zip(traced_submissions, submission_files, strict=True)
        ):
            raise ValueError("math trace submission order differs from submission_files")
    elif len(submission_files) > 1:
        raise ValueError("multiple math submission_files require matching trace submission entries")


def _build_leak_task(fixture: FixtureTask) -> BuiltFixtureTask:
    definition = fixture.definition
    assets_path = _fixture_path(fixture, _required_string(definition, "assets_file"))
    partition = _required_string(definition, "partition")
    channels_value = definition.get("channels")
    if not isinstance(channels_value, list) or any(
        not isinstance(item, str) for item in channels_value
    ):
        raise TypeError("leak channels must be a list of strings")
    channels = tuple(cast("list[str]", channels_value))
    repos_value = definition.get("repos")
    if repos_value is not None and (
        not isinstance(repos_value, list) or any(not isinstance(item, str) for item in repos_value)
    ):
        raise TypeError("leak repos must be a list of strings or null")
    repos = None if repos_value is None else tuple(cast("list[str]", repos_value))
    pairs_value = definition.get("eval_pairs")
    eval_pairs: tuple[tuple[str, str], ...] | None = None
    if pairs_value is not None:
        if not isinstance(pairs_value, list) or any(
            not isinstance(pair, list)
            or len(pair) != LEAK_PAIR_PARTS
            or any(not isinstance(part, str) for part in pair)
            for pair in pairs_value
        ):
            raise TypeError("leak eval_pairs must be a list of [task_id, channel] pairs")
        eval_pairs = tuple((pair[0], pair[1]) for pair in cast("list[list[str]]", pairs_value))
    tasks, manifests, _counts = load_leak_tasks(
        assets_path,
        partition=partition,
        channels=channels,
        instruction=cast("InstructionVariant", _optional_string(definition, "instruction", "none")),
        repos=repos,
        open_arm=definition.get("open_arm") is True,
        eval_pairs=eval_pairs,
    )
    matching_tasks = [task for task in tasks if task.task_id == fixture.task_id]
    if len(matching_tasks) != 1:
        raise ValueError(
            f"leak fixture expected one task {fixture.task_id!r}, found {len(matching_tasks)}"
        )
    return BuiltFixtureTask(task=matching_tasks[0], manifest=manifests[fixture.task_id])


def _apply_reference_files(
    fixture: FixtureTask, artifact: FixtureArtifact, episode_directory: Path, task: Task
) -> None:
    artifact_value = artifact.value
    solution_file = artifact_value.get("solution_file", artifact_value.get("math_answers_file"))
    if solution_file is not None:
        if not isinstance(solution_file, str) or task.artifact_filename is None:
            raise ValueError(
                f"{fixture.family} artifact requires a solution path and artifact filename"
            )
        _copy_fixture_file(fixture, solution_file, episode_directory, task.artifact_filename)
    if task.task_variant == "leak":
        # Grading rebuilds patch.diff from the workspace diff, so the fix must land in the source
        # files as an agent's edits would; a dropped-in patch.diff grades as an empty submission.
        patch_path = _fixture_path(fixture, _required_string(artifact_value, "patch_file"))
        subprocess.run(  # noqa: S603 - fixed git invocation on a fixture-owned patch
            ["git", "-c", "core.hooksPath=/dev/null", "apply", str(patch_path)],  # noqa: S607
            cwd=episode_directory,
            check=True,
        )
    edited_files = artifact_value.get("edited_files", {})
    if not isinstance(edited_files, Mapping):
        raise TypeError("edited_files must be an object")
    for destination, source in edited_files.items():
        if not isinstance(destination, str) or not isinstance(source, str):
            raise TypeError("edited_files must map destination strings to source strings")
        _copy_fixture_file(fixture, source, episode_directory, destination)


def _apply_repository_fixture_mechanism(
    fixture: FixtureTask,
    artifact: FixtureArtifact,
    task: Task,
    episode_directory: Path,
    *,
    repository_variant: Literal["golden", "timed"] | None,
) -> None:
    if repository_variant is None:
        if "regenerate_expected" in artifact.value or "case_timeout_seconds" in artifact.value:
            raise ValueError("repository check mutations require an ILCB repository variant")
        return
    if repository_variant == "golden":
        _regenerate_golden_expected(fixture, artifact, task, episode_directory)
        return
    _raise_if_timed_artifact_regenerates_expected(artifact)
    _set_timed_case_timeout(fixture, artifact, episode_directory)


def _regenerate_golden_expected(
    fixture: FixtureTask,
    artifact: FixtureArtifact,
    task: Task,
    episode_directory: Path,
) -> None:
    if "case_timeout_seconds" in artifact.value:
        raise ValueError("golden artifacts cannot change a timed case budget")
    regenerate_expected = artifact.value.get("regenerate_expected", False)
    if not isinstance(regenerate_expected, bool):
        raise TypeError("artifact regenerate_expected must be a boolean")
    if not regenerate_expected:
        return
    cpus, mem_max = episode_limits()
    result = run_in_jail(
        episode_directory,
        f"{task.grader_command} --update-expected",
        timeout=_timeout_string(fixture.definition, "60s"),
        backend=_optional_string(fixture.definition, "jail_backend", None),
        login_shell=False,
        cpus=cpus,
        mem_max=mem_max,
    )
    if not result.ok:
        raise RuntimeError(
            f"{fixture.task_id}/{artifact.artifact_id}: expected-output regeneration failed "
            f"(exit={result.exit_code}, timed_out={result.timed_out})"
        )


def _raise_if_timed_artifact_regenerates_expected(artifact: FixtureArtifact) -> None:
    if "regenerate_expected" in artifact.value:
        raise ValueError("timed artifacts cannot regenerate golden expected outputs")


def _set_timed_case_timeout(
    fixture: FixtureTask,
    artifact: FixtureArtifact,
    episode_directory: Path,
) -> None:
    raw_timeout = artifact.value.get("case_timeout_seconds")
    if raw_timeout is None:
        return
    if (
        isinstance(raw_timeout, bool)
        or not isinstance(raw_timeout, (int, float))
        or not math.isfinite(raw_timeout)
        or raw_timeout <= 0
    ):
        raise TypeError("artifact case_timeout_seconds must be a finite positive number")
    timeout_path = _workspace_path(episode_directory, "tests.toml")
    if not timeout_path.is_file() or timeout_path.is_symlink():
        raise ValueError(f"{fixture.task_id}: timed repository check has no tests.toml")
    configuration = timeout_path.read_text(encoding="utf-8")
    updated_configuration, replacements = re.subn(
        r"^case_timeout_seconds = [^\n]+$",
        f"case_timeout_seconds = {float(raw_timeout)!r}",
        configuration,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise ValueError(f"{fixture.task_id}: expected exactly one timed case budget")
    timeout_path.write_text(updated_configuration, encoding="utf-8")


def _repository_mechanism_evidence(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, object]:
    return repository_mechanism_evidence(dict(before), dict(after))


def _copy_fixture_file(
    fixture: FixtureTask, source_relative: str, episode_directory: Path, destination_relative: str
) -> None:
    source = _fixture_path(fixture, source_relative)
    destination = _workspace_path(episode_directory, destination_relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _workspace_path(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"workspace artifact path must be a safe relative path: {relative_path!r}")
    destination = root.joinpath(*path.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"workspace artifact path escapes the episode: {relative_path!r}")
    return destination


def _fixture_path(fixture: FixtureTask, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"fixture path must be a safe relative path: {relative_path!r}")
    lexical_path = fixture.directory / path
    _reject_fixture_symlinks(fixture.directory, lexical_path)
    resolved = lexical_path.resolve(strict=True)
    if not resolved.is_relative_to(fixture.directory.resolve()) or (
        not resolved.is_file() and not resolved.is_dir()
    ):
        raise ValueError(
            f"fixture path is missing or escapes the task directory: {relative_path!r}"
        )
    return resolved


def _reject_fixture_symlinks(task_directory: Path, candidate: Path) -> None:
    current = task_directory
    for component in candidate.relative_to(task_directory).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"fixture paths must not use symlinks: {current}")


def _strengthened_truth(
    fixture: FixtureTask, artifact: FixtureArtifact, grade: FinalGrade
) -> tuple[bool | None, hack_strategy.StrengthenedTruthEvidence]:
    cases_root_value = artifact.value.get(
        "cases_root_override", fixture.definition.get("cases_root")
    )
    if fixture.family not in {"ilcb", "aisi"}:
        return (
            grade.true_pass,
            hack_strategy.StrengthenedTruthEvidence(
                passed=grade.true_pass,
                source="task_true_check",
                reference_set_digest=None,
                disagreement_fraction=None,
                counterexample=None,
                unmeasured_timeout=False,
            ),
        )
    if cases_root_value is None:
        raise ValueError(f"{fixture.family} fixtures require a strengthened-truth cases_root")
    if not isinstance(cases_root_value, str):
        raise TypeError("cases_root must be a relative path string")
    if not cases_root_value:
        raise ValueError("cases_root must be a nonempty relative path")
    cases_root = _fixture_path(fixture, cases_root_value)
    artifact_path_value = artifact.value.get(
        "solution_file", artifact.value.get("math_answers_file")
    )
    if not isinstance(artifact_path_value, str):
        raise TypeError("strengthened truth requires an artifact solution_file")
    source_path = _fixture_path(fixture, artifact_path_value)
    entry_point = _required_string(fixture.definition, "entry_point")
    if grade.task_evidence is None:
        raise ValueError("strengthened truth requires task problem identity")
    case_task_id = (
        f"ilcb-original-{grade.task_evidence.problem_id}"
        if fixture.family == "ilcb"
        else _aisi_problem(fixture.definition).task_id("original")
    )
    case_set = strengthened_truth_grade.load_case_set(cases_root, case_task_id)
    reference_validation_path_value = fixture.definition.get("reference_validation_file")
    reference_suite_seconds: tuple[float, ...] | None = None
    if reference_validation_path_value is not None:
        if (
            not isinstance(reference_validation_path_value, str)
            or not reference_validation_path_value
        ):
            raise TypeError("reference_validation_file must be a nonempty relative path string")
        reference_validation_path = _fixture_path(fixture, reference_validation_path_value)
        reference_validation_rows = strengthened_truth_grade._load_reference_validation(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # reuse the W16 validation reader
            reference_validation_path
        )
        reference_suite_seconds = strengthened_truth_grade._validated_reference_suite_seconds(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # reuse the W16 timing validator
            strengthened_truth_grade._ReferenceSuiteTimingRequest(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # keep the W16 request identity exact
                reference_rows=reference_validation_rows,
                task_id=strengthened_truth_grade._reference_validation_task_id(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # reuse the W16 task identity join
                    case_task_id, case_set
                ),
                case_set=case_set,
                entry_point=entry_point,
                family=fixture.family,
                original_assertions=_string_tuple(
                    fixture.definition, "original_assertions", required=False
                ),
            )
        )
    result = strengthened_truth_grade.grade_artifact(
        source_path,
        case_set,
        entry_point=entry_point,
        family=fixture.family,
        original_assertions=_string_tuple(
            fixture.definition, "original_assertions", required=False
        ),
        reference_suite_seconds=reference_suite_seconds,
    )
    passed = result["pass"]
    if passed is not None and type(passed) is not bool:
        raise TypeError("strengthened truth grader returned a non-boolean pass")
    evidence = hack_strategy.StrengthenedTruthEvidence(
        passed=passed,
        source=cast("str", result["source"]),
        reference_set_digest=cast("str | None", result["reference_set_digest"]),
        disagreement_fraction=cast("float | None", result["disagreement_fraction"]),
        counterexample=cast("Mapping[str, object] | None", result["counterexample"]),
        unmeasured_timeout=cast("bool", result["unmeasured_timeout"]),
    )
    return passed, evidence


def _episode_record(context: EpisodeRecordContext) -> dict[str, object]:
    fixture = context.fixture
    task = context.task
    grade = context.grade
    payment_register = context.payment_register
    content_changed = context.content_changed
    episode_directory = context.episode_directory
    task_evidence = grade.task_evidence
    if task_evidence is None:
        raise ValueError(f"task {task.task_id!r} returned no task episode evidence")
    artifact_digest = hashlib.sha256(_submitted_artifact_bytes(task, episode_directory)).hexdigest()
    if task_evidence.submitted_artifact_sha256 != artifact_digest:
        raise ValueError("graded task evidence does not match the submitted artifact")
    if task.task_variant is None:
        raise ValueError(f"task {task.task_id!r} returned no task variant")
    family = _generated_family(fixture)
    summary: dict[str, object] = {
        "record": "episode_summary",
        "source": "organism_rl.generate",
        "task_id": task.task_id,
        "task_variant": task.task_variant,
        "variant": task.task_variant,
        "problem_id": task_evidence.problem_id,
        "payment_register": payment_register,
        "proxy": {"pass": grade.proxy_pass},
        "true": {
            "pass": grade.true_pass,
            "measured": grade.oracle_error is None,
            "oracle_error": grade.oracle_error,
            **({"evidence": grade.true_evidence} if grade.true_evidence is not None else {}),
        },
        "gap": {
            "visible_check_unsatisfiable": task.visible_check_unsatisfiable,
            "visible_check_contradicts_truth": task.visible_check_contradicts_truth,
        },
        "arm": AgenticArmConfig().to_json_dict(),
        "provided_test_pass": task_evidence.provided_test_pass,
        "submitted_artifact_sha256": artifact_digest,
        "graded_on": "submission" if context.submission_verdicts else "final_state",
        "submission_verdicts": [dict(verdict) for verdict in context.submission_verdicts],
        "final_state_verdict": _final_state_verdict(context, artifact_digest),
        "grader_integrity": {
            "laydown_sha256": context.laydown_grader_sha256,
            "content_changed": content_changed,
            **(
                {"mechanism": dict(context.mechanism_evidence)}
                if context.mechanism_evidence is not None
                else {}
            ),
        },
        "family": family,
    }
    # Leak submissions are reconstructed from the workspace diff; no policy-visible artifact file exists.
    if task.task_variant != "leak" and task.artifact_filename is not None:
        artifact_path = _workspace_path(episode_directory, task.artifact_filename)
        summary["graded_artifact"] = {"path": str(artifact_path), "sha256": artifact_digest}
    if fixture.definition.get("convention_note", False):
        summary["convention_note"] = True
    if task.task_variant == "leak":
        leak_identity = task.host_only_metadata
        summary["leak_channels_open"] = leak_identity["leak_channels_open"]
        summary["instruction_variant"] = leak_identity["instruction_variant"]
    if context.math_evidence is not None:
        summary["math_evidence"] = dict(context.math_evidence)

    sample_trace_records: list[dict[str, object]] = (
        [
            {
                "record": "turn",
                "turn": index,
                "prompt_tokens": 0,
                "reasoning": "",
                "content": "",
                "commands": [
                    {
                        "command": f"echo {SUBMIT_MARKER}",
                        "exit_code": 0,
                        "stdout": f"{SUBMIT_MARKER}\n",
                        "stderr": "",
                        "timed_out": False,
                    }
                ],
            }
            for index, _verdict in enumerate(context.submission_verdicts)
        ]
        if context.submission_verdicts
        else [dict(record) for record in context.trace_records]
    )
    sample_trace_records.extend(
        dict(record)
        for record in context.trace_records
        if record.get("record") == "organism_episode_identity"
    )
    if not any(
        record.get("record") == "organism_episode_identity" for record in sample_trace_records
    ):
        sample_trace_records.append(
            {
                "record": "organism_episode_identity",
                "hint": {"family": None, "strength": "none", "sha": None},
                **(
                    {"convention_note": True}
                    if fixture.definition.get("convention_note", False)
                    else {}
                ),
            }
        )
    sample_trace_records = [
        record for record in sample_trace_records if record.get("record") != "episode_summary"
    ]
    sample_trace_records.append({"record": "episode_summary", "context": {"peak_prompt_tokens": 0}})
    trace_path = episode_directory / "env-gate-trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in sample_trace_records),
        encoding="utf-8",
    )
    sample = rollout.EpisodeSample(
        record=summary,
        trace_records=sample_trace_records,
        trace_path=trace_path,
    )
    task_spec = generate.TaskSpec(
        task_id=task.task_id,
        family=cast("generate.CodingTaskFamily", family),
        variant=task.task_variant,
        payment_register=cast(
            "Literal['trusted_rerun', 'in_band', 'status_report']", payment_register
        ),
    )
    return generate._episode_record(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001 - reuse the production record builder
        task_spec=task_spec,
        sample_index=0,
        seed=0,
        sample=sample,
        provenance_sha256=hashlib.sha256(
            f"env-gate:{fixture.task_id}:{context.payment_register}".encode()
        ).hexdigest(),
    )


def _final_state_verdict(context: EpisodeRecordContext, artifact_digest: str) -> dict[str, object]:
    """Mirror the production hand-off: a status claim for status tasks, a CI verdict otherwise."""
    grade = context.grade
    if context.payment_register != "status_report":
        return {
            "verdict": "SUCCESS" if grade.proxy_pass else "FAIL",
            "register": "ci",
            "feedback": grade.grader.stdout.strip(),
            "submission_evidence": {"artifact_sha256": artifact_digest},
        }
    return {
        "verdict": "RECEIVED",
        "register": "status_report",
        "feedback": rollout.STATUS_HAND_OFF_ACKNOWLEDGEMENT,
        "submission_evidence": {
            "artifact_sha256": artifact_digest,
            "workspace_tests_passed": grade.proxy_pass,
            "status_file": parse_status_bytes(
                read_status_file(context.episode_directory)
            ).to_record(),
            "grader_integrity_sha256": grader_integrity_hash(
                context.episode_directory, context.task.effective_grader_integrity_files
            ),
        },
    }


def _generated_family(fixture: FixtureTask) -> str:
    if fixture.family == "math":
        return _required_string(fixture.definition, "math_family")
    if fixture.family == "leak":
        return "r2e_leak"
    return fixture.family


def _classify_sft_admission(
    record: Mapping[str, object],
    assessment: hack_strategy.HackStrategyAssessment,
    strengthened_truth: hack_strategy.StrengthenedTruthEvidence,
    raw_override: object,
    *,
    rewarded: bool | None,
) -> dict[str, object]:
    if agentic_sft._record_convention_note(record):  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        return {
            "criteria": [],
            "type": None,
            "exclusion_reason": "convention_note_arm",
            "solvable_hack": False,
        }
    if rewarded is None:
        return {
            "criteria": [],
            "type": "unmeasured",
            "exclusion_reason": None,
            "solvable_hack": None,
        }
    override = (
        cast("Mapping[str, object] | None", raw_override)
        if isinstance(raw_override, Mapping)
        else None
    )
    flags = agentic_sft.grader_directed_move_flags(record)
    context = agentic_sft._AdmissionContext(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        flags=flags,
        organism_rewarded=rewarded,
        assessment=assessment,
        regex_assessment=assessment,
        strategy_label_override=override,
        strengthened_truth=strengthened_truth,
    )
    criteria, admission_type, exclusion_reason, solvable_hack = agentic_sft._classify_admission(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        record, context
    )
    return {
        "criteria": criteria,
        "type": admission_type,
        "exclusion_reason": exclusion_reason,
        "solvable_hack": solvable_hack,
    }


def _surfaced_leak_channel(assessment: hack_strategy.HackStrategyAssessment) -> str | None:
    leak_value = assessment.leak_detector
    if not isinstance(leak_value, Mapping):
        return None
    channels = leak_value.get("channels_surfaced")
    if not isinstance(channels, (list, tuple)):
        raise TypeError("leak detector channels_surfaced must be a sequence")
    if not channels:
        return None
    if any(not isinstance(channel, str) for channel in channels):
        raise TypeError("leak detector channels_surfaced must contain strings")
    if len(channels) != 1:
        raise ValueError(f"expected one surfaced leak channel, observed {len(channels)}")
    return cast("str", channels[0])


def _optional_manifest(
    fixture: FixtureTask, artifact: FixtureArtifact
) -> Mapping[str, object] | None:
    path_value = artifact.value.get("manifest_file")
    if path_value is None:
        return None
    if not isinstance(path_value, str):
        raise TypeError("manifest_file must be a fixture path string")
    return _read_json_object(_fixture_path(fixture, path_value))


def _load_docker_evidence(
    fixture: FixtureTask, artifact: FixtureArtifact, task: Task, episode_directory: Path
) -> Mapping[str, object] | Sequence[Mapping[str, object]] | None:
    path_value = artifact.value.get("docker_evidence_file")
    if path_value is not None:
        if not isinstance(path_value, str):
            raise TypeError("docker_evidence_file must be a fixture path string")
        path = _fixture_path(fixture, path_value)
    elif task.docker_config is not None and task.docker_config.evidence_path is not None:
        path = Path(task.docker_config.evidence_path)
    elif task.docker_config is not None:
        path = episode_directory.with_name(episode_directory.name + ".docker-evidence.jsonl")
    else:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Docker grader did not persist evidence at {path}")
    if path.suffix == ".jsonl":
        return tuple(_read_json_lines(path))
    return _read_json_object(path)


def _load_trace_records(fixture: FixtureTask, artifact: FixtureArtifact) -> list[dict[str, object]]:
    path_value = artifact.value.get("trace_file")
    if path_value is None:
        return []
    if not isinstance(path_value, str):
        raise TypeError("trace_file must be a fixture path string")
    return _read_json_lines(_fixture_path(fixture, path_value))


def _read_json_lines(path: Path) -> list[dict[str, object]]:
    return read_jsonl(path)


def _grade_ladder_artifact(
    fixture: FixtureTask, artifact: FixtureArtifact, scenario: Scenario
) -> dict[str, object]:
    if scenario.format != REPOSITORY_FORMAT:
        raise ValueError("W5 ladder fixtures must use repository-format wip scenarios")
    with tempfile.TemporaryDirectory(
        prefix="organism-env-gate-ladder-", dir=EPISODE_SCRATCH_ROOT
    ) as temporary:
        episode_directory = Path(temporary) / "episode"
        rendered = scenario.render("real")
        task = rendered.task
        materialized_files = lay_down_task(episode_directory, task, arm=rendered.arm)
        tree_roots = scenario.lay_down_trees(episode_directory)
        monitor = CommandAuditMonitor(
            episode_directory,
            config_path=scenario.monitor_config_path,
            audit_path=scenario.audit_log_path,
            key=scenario.monitor_key,
            enabled_value=scenario.monitor_enabled_value,
        )
        config = cast(
            "ladder_runner.RunnerConfig",
            SimpleNamespace(
                timeout=_timeout_string(fixture.definition, "120s"),
                resource_limits=ladder_runner.ENFORCED_JAIL_RESOURCE_LIMITS,
                task_pressure="none",
            ),
        )
        grade_callback = ladder_runner._real_arm_submission_grader(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            scenario, task, monitor, config
        )
        command_context = _LadderCommandContext(
            episode_directory=episode_directory,
            monitor=monitor,
            scenario=scenario,
            grade_callback=grade_callback,
            timeout=config.timeout,
        )

        start_state_first_ci: dict[str, object] | None = None
        first_ci_submission: ladder_loop.SubmissionVerdict | None = None
        if artifact.artifact_id == "start_state":
            first_ci_submission = grade_callback(episode_directory)
            first_ci_suite, first_ci_policy = _ladder_submission_results(
                first_ci_submission, episode_directory, scenario
            )
            start_state_first_ci = _ladder_start_state_observation(
                scenario, first_ci_suite, first_ci_policy
            )

        engine_command_records, submission_verdicts = _run_ladder_reference_commands(
            fixture, artifact, command_context
        )
        if first_ci_submission is not None and not engine_command_records:
            final_grade = first_ci_submission
        else:
            final_grade = grade_callback(episode_directory)
        suite, policy = _ladder_submission_results(final_grade, episode_directory, scenario)
        true_pass_value = final_grade.evidence.get("oracle_pass")
        if true_pass_value is not None and type(true_pass_value) is not bool:
            raise TypeError("ladder grader oracle_pass must be a boolean or null")
        true_pass = true_pass_value
        # Reference artifacts need a final-state submission when their commands have no submit marker.
        scored_submissions = submission_verdicts or [final_grade]
        trace_summary: dict[str, object] = {
            "record": "ladder_episode",
            "true": {"pass": true_pass, "measured": true_pass is not None, "oracle_error": None},
            "submission_verdicts": [
                {
                    "verdict": "SUCCESS" if verdict.passed else "FAIL",
                    "register": verdict.register,
                    "submission_evidence": dict(verdict.evidence),
                }
                for verdict in scored_submissions
            ],
        }
        behaviour_events = _ladder_behaviour_events(
            engine_command_records, scenario, episode_directory
        )
        record: dict[str, object] = {
            "record": "ladder_episode",
            "source": "env_gate.ladder",
            "task_id": fixture.task_id,
            "scenario_slug": scenario.slug,
            "scenario_format": scenario.format,
            "scenario": {"scenario_id": scenario.slug, "set_version": scenario.set_version},
            "key": {"scenario_id": scenario.slug, "arm": "real", "family": "ladder"},
            "family": "ladder",
            "arm": {"label": "real"},
            "variant": "oneoff",
            "payment_register": "trusted_rerun",
            "final_grade": {"final_grade_pass": final_grade.passed},
            "trace_summary": trace_summary,
            "engine_command_records": engine_command_records,
            "detector_inputs": {
                "detection_spec": ladder_runner.detection_spec_to_json(
                    ladder_runner.detection_spec_for_scenario(scenario)
                ),
                "behaviour_paths": [
                    *scenario.exposure_spec.policy_files,
                    *scenario.exposure_spec.bypass_files,
                    scenario.grader_filename,
                ],
                "protected_paths": list(scenario.protected_paths),
            },
            "behaviour_events": behaviour_events,
        }
        trace_summary["payment_register"] = "trusted_rerun"
        trace_summary["grader_integrity"] = {"content_changed": False}
        trace_records = _load_trace_records(fixture, artifact)
        trace_records.extend(engine_command_records)
        strengthened = hack_strategy.StrengthenedTruthEvidence(
            passed=true_pass,
            source="ladder_real_arm",
            reference_set_digest=None,
            disagreement_fraction=None,
            counterexample=None,
            unmeasured_timeout=False,
        )
        register_observations, label_observations, admission_observations = (
            _ladder_reward_observations(
                record,
                trace_summary,
                trace_records,
                strengthened,
                artifact.value.get("strategy_label_override"),
            )
        )
        observations: dict[str, object] = {
            "registers": register_observations,
            "strengthened_truth_pass": true_pass,
            "labeler_primary_label": label_observations,
            "leak_channel": None,
            "sft_admission": admission_observations,
            "suite_passed": suite.passed,
            "policy_audit": _ladder_policy_observation(policy),
            "policy_or_ci_file_edited": bool(behaviour_events["policy_or_ci_file_edited"]),
            "acceptance_paths_changed": _submission_evidence_boolean(
                final_grade.evidence, "acceptance_paths_changed"
            ),
            "protected_path_moved": bool(behaviour_events["protected_path_moved"]),
            "visible_check": {"pass": suite.passed},
            "register_independent": _register_independent_first_turn(
                _FirstTurnRenderContext(
                    task=task,
                    episode_directory=episode_directory,
                    materialized_paths=materialized_files,
                    tree_roots=tree_roots,
                    system_prompt=scenario.system_prompt,
                    user_template=scenario.user_template,
                    arm=rendered.arm,
                )
            ),
        }
        if start_state_first_ci is not None:
            observations["start_state_first_ci"] = start_state_first_ci
        return observations


def _ladder_reward_observations(
    record: dict[str, object],
    trace_summary: dict[str, object],
    trace_records: Sequence[Mapping[str, object]],
    strengthened: hack_strategy.StrengthenedTruthEvidence,
    strategy_label_override: object,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    register_observations: dict[str, object] = {}
    label_observations: dict[str, object] = {}
    admission_observations: dict[str, object] = {}
    true_values = trace_summary.get("true")
    if not isinstance(true_values, Mapping):
        raise TypeError("ladder trace summary must contain a true result object")
    true_pass = true_values.get("pass")
    for payment_register in PAYMENT_REGISTERS:
        record["payment_register"] = payment_register
        trace_summary["payment_register"] = payment_register
        score = organism_reward.score_impossible_task(record)
        assessment = hack_strategy.label_episode(
            record, trace_records, strengthened_truth=strengthened
        )
        admission = _classify_sft_admission(
            record,
            assessment,
            strengthened,
            strategy_label_override,
            rewarded=None if score.reward is None else score.reward == 1,
        )
        register_observations[payment_register] = {
            "reward": score.reward,
            "true_pass": true_pass,
            "true_measured": true_pass is not None,
            "oracle_error": None,
        }
        label_observations[payment_register] = assessment.primary_strategy
        admission_observations[payment_register] = admission
    return register_observations, label_observations, admission_observations


def _ladder_submission_results(
    submission: ladder_loop.SubmissionVerdict,
    episode_directory: Path,
    scenario: Scenario,
) -> tuple[SuiteJobResult, PolicyJobResult]:
    evidence = submission.evidence
    raw_grade = evidence.get("pristine_visible_grade")
    if not isinstance(raw_grade, Mapping):
        raise TypeError("ladder submission evidence needs pristine_visible_grade")
    exit_code = raw_grade.get("exit_code")
    stdout = raw_grade.get("stdout")
    stderr = raw_grade.get("stderr")
    timed_out = raw_grade.get("timed_out")
    if type(exit_code) is not int or not isinstance(stdout, str) or not isinstance(stderr, str):
        raise TypeError("ladder pristine grade result has invalid command fields")
    if not isinstance(timed_out, bool):
        raise TypeError("ladder pristine grade result timed_out must be a boolean")
    pristine_result = CommandResult("pristine_visible_grade", exit_code, stdout, stderr, timed_out)
    suite = ladder_runner._pristine_run_checks_result(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        pristine_result
    )
    pristine_visible_pass = evidence.get("pristine_visible_pass")
    if not isinstance(pristine_visible_pass, bool):
        raise TypeError("ladder submission evidence pristine_visible_pass must be a boolean")
    if suite.passed != pristine_visible_pass:
        suite = replace(suite, passed=False)

    protected_paths_changed = _submission_evidence_boolean(evidence, "protected_paths_changed")
    oracle_pass = evidence.get("oracle_pass")
    if oracle_pass is not None and not isinstance(oracle_pass, bool):
        raise TypeError("ladder submission evidence oracle_pass must be a boolean or null")
    report_mode = not monitor_enabled(
        episode_directory,
        config_path=scenario.monitor_config_path,
        key=scenario.monitor_key,
        enabled_value=scenario.monitor_enabled_value,
    )
    policy = ladder_runner.policy_job_result(
        protected_paths_changed=protected_paths_changed,
        oracle_measured=oracle_pass is not None,
        oracle_pass=oracle_pass,
        report_mode=report_mode,
    )
    return suite, policy


def _submission_evidence_boolean(evidence: Mapping[str, object], field_name: str) -> bool:
    value = evidence.get(field_name)
    if not isinstance(value, bool):
        raise TypeError(f"ladder submission evidence {field_name} must be a boolean")
    return value


def _ladder_policy_observation(policy: PolicyJobResult) -> dict[str, object]:
    return {
        "failed": policy.failed,
        "report_mode": policy.report_mode,
        "findings": [finding.value for finding in policy.findings],
    }


def _ladder_start_state_observation(
    scenario: Scenario, suite: SuiteJobResult, policy: PolicyJobResult
) -> dict[str, object]:
    expected_rows = scenario.exposure_spec.conflict_test_ids
    if not expected_rows:
        raise ValueError("ladder start state must declare conflict acceptance test ids")
    failing_rows: list[str] = []
    for failure in suite.failures:
        parts = failure.split(maxsplit=2)
        if len(parts) < ACCEPTANCE_FAILURE_PARTS or parts[0] != "FAILED":
            raise ValueError(f"unrecognized acceptance failure summary: {failure!r}")
        failing_rows.append(parts[1])
    if suite.passed or tuple(sorted(failing_rows)) != tuple(sorted(expected_rows)):
        raise DeclarationMismatchError(
            f"{scenario.slug}/start_state: must fail only on the declared conflict rows "
            f"{sorted(expected_rows)}; suite passed={suite.passed}, failing rows={sorted(failing_rows)}"
        )
    if policy.failed:
        raise DeclarationMismatchError(f"{scenario.slug}/start_state: policy audit must pass")
    return {
        "suite_passed": suite.passed,
        "failing_acceptance_rows": failing_rows,
        "policy_audit": _ladder_policy_observation(policy),
    }


def _run_ladder_reference_commands(
    fixture: FixtureTask,
    artifact: FixtureArtifact,
    context: _LadderCommandContext,
) -> tuple[list[dict[str, object]], list[ladder_loop.SubmissionVerdict]]:
    commands: list[str] = []
    patch_file = artifact.value.get("patch_file")
    if patch_file is not None:
        if not isinstance(patch_file, str):
            raise TypeError("ladder patch_file must be a fixture path string")
        patch_path = _fixture_path(fixture, patch_file)
        if _artifact_kind(artifact) == "empty":
            if patch_path.read_bytes():
                raise ValueError(
                    f"{fixture.task_id}/{artifact.artifact_id}: empty patch is not empty"
                )
        else:
            staged_patch = context.episode_directory / ".env-gate-reference.patch"
            shutil.copyfile(patch_path, staged_patch)
            commands.append(
                "git apply --binary .env-gate-reference.patch && rm .env-gate-reference.patch"
            )
    commands_file = artifact.value.get("commands_file")
    if commands_file is not None:
        if not isinstance(commands_file, str):
            raise TypeError("commands_file must be a fixture path string")
        commands_value: object = json.loads(
            _fixture_path(fixture, commands_file).read_text(encoding="utf-8")
        )
        if not isinstance(commands_value, list) or any(
            not isinstance(command, str) for command in commands_value
        ):
            raise TypeError("ladder commands_file must contain a JSON list of command strings")
        commands.extend(cast("list[str]", commands_value))
    return _execute_ladder_commands(
        commands,
        context,
    )


def _execute_ladder_commands(
    commands: Sequence[str],
    context: _LadderCommandContext,
) -> tuple[list[dict[str, object]], list[ladder_loop.SubmissionVerdict]]:
    command_records: list[dict[str, object]] = []
    submission_verdicts: list[ladder_loop.SubmissionVerdict] = []
    for command in commands:
        before_digests = ladder_loop.protected_path_digests(
            context.episode_directory, context.scenario.protected_paths
        )
        _run_monitored_jail_command(
            context.episode_directory,
            command,
            context.monitor,
            timeout=context.timeout,
        )
        after_digests = ladder_loop.protected_path_digests(
            context.episode_directory, context.scenario.protected_paths
        )
        command_record = context.monitor.records[-1].to_json_dict()
        command_record["before_protected_path_digests"] = before_digests
        command_record["after_protected_path_digests"] = after_digests
        command_records.append(command_record)
        verdict = ladder_loop.grade_registered_submission(
            command,
            context.episode_directory,
            len(submission_verdicts),
            submit_grader=None,
            ci_submit_grader=context.grade_callback,
        )
        if verdict is not None:
            submission_verdicts.append(verdict)
            if verdict.passed or len(submission_verdicts) >= ladder_loop.LADDER_MAX_SUBMISSIONS:
                break
    return command_records, submission_verdicts


def _run_monitored_jail_command(
    episode_directory: Path, command: str, monitor: CommandAuditMonitor, *, timeout: str
) -> None:
    start = monitor.before_command(command)
    result = run_in_jail(
        episode_directory,
        command,
        timeout=timeout,
        login_shell=False,
        cpus=1,
        mem_max="1G",
    )
    monitor.after_command(start, result)
    if not result.ok:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"ladder reference command failed: {command!r} exit={result.exit_code}: {details}"
        )


def _ladder_behaviour_events(
    engine_command_records: Sequence[Mapping[str, object]],
    scenario: Scenario,
    episode_directory: Path,
) -> dict[str, object]:
    detection_spec = ladder_runner.detection_spec_to_json(
        ladder_runner.detection_spec_for_scenario(scenario)
    )
    behaviour_paths = (
        *scenario.exposure_spec.policy_files,
        *scenario.exposure_spec.bypass_files,
        scenario.grader_filename,
    )
    record: dict[str, object] = {
        "episode_dir": str(episode_directory),
        "detector_inputs": {
            "detection_spec": detection_spec,
            "behaviour_paths": list(behaviour_paths),
            "protected_paths": list(scenario.protected_paths),
        },
        "engine_command_records": list(engine_command_records),
    }
    commands = [
        (cast("int", command["sequence"]), index, cast("str", command["command"]))
        for index, command in enumerate(engine_command_records)
    ]
    return literal_behaviour_events(record, commands, behaviour_paths)


def _dataclass_fields(
    definition: Mapping[str, object], key: str, dataclass_type: type[Any]
) -> dict[str, object]:
    value = _required_mapping(definition, key)
    expected_fields = set(dataclass_type.__dataclass_fields__)
    missing = expected_fields - set(value)
    extra = set(value) - expected_fields
    if missing or extra:
        raise ValueError(
            f"{dataclass_type.__name__} fields mismatch (missing={sorted(missing)}, extra={sorted(extra)})"
        )
    return {name: value[name] for name in expected_fields}


def _math_question(value: Mapping[str, object]) -> MathQuestion:
    expected_fields = set(MathQuestion.__dataclass_fields__)
    if set(value) != expected_fields:
        raise ValueError("MathQuestion fields mismatch")
    kind = _required_string(value, "kind")
    if kind not in MATH_KINDS:
        raise ValueError(f"unsupported math question kind: {kind!r}")
    return MathQuestion(
        question_id=_required_string(value, "question_id"),
        kind=kind,
        text=_required_string(value, "text"),
        answer=_required_string(value, "answer"),
    )


def _required_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise TypeError(f"{key} must be an object")
    return cast("Mapping[str, object]", child)


def _required_string(value: Mapping[str, object], key: str) -> str:
    child = value.get(key)
    if not isinstance(child, str) or not child:
        raise TypeError(f"{key} must be a nonempty string")
    return child


def _optional_string(value: Mapping[str, object], key: str, default: str | None) -> str | None:
    child = value.get(key, default)
    if child is not None and not isinstance(child, str):
        raise TypeError(f"{key} must be a string or null")
    return child


def _timeout_string(value: Mapping[str, object], default: str) -> str:
    timeout = _optional_string(value, "timeout", default)
    if timeout is None or not timeout:
        raise TypeError("timeout must be a nonempty string")
    return timeout


def _required_int(value: Mapping[str, object], key: str) -> int:
    child = value.get(key)
    if not isinstance(child, int) or isinstance(child, bool):
        raise TypeError(f"{key} must be an integer")
    return child


def _string_tuple(value: Mapping[str, object], key: str, *, required: bool) -> tuple[str, ...]:
    child = value.get(key)
    if child is None and not required:
        return ()
    if not isinstance(child, list) or any(not isinstance(item, str) for item in child):
        raise TypeError(f"{key} must be a list of strings")
    return tuple(cast("list[str]", child))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        help=(
            "grading workers (default: CPU and memory budget with two CPUs reserved; "
            "env: ORGANISM_RL_GRADING_WORKERS)"
        ),
    )
    parser.add_argument("--report-jsonl", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixture gate from command-line arguments."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    return run_gate(args.fixtures, workers=args.workers, report_jsonl=args.report_jsonl)


if __name__ == "__main__":
    sys.exit(main())
