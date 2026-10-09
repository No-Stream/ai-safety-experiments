"""Run reference artifacts through the environment graders and check fixture declarations."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

from reward_hacking import organism_reward
from reward_hacking.aisi_rh_envs_data import AISI_SOURCES, AisiRhProblem, AisiTaskVariant
from reward_hacking.distill import agentic_sft, hack_strategy
from reward_hacking.harness.loop import (
    BASELINE_ARM,
    FinalGrade,
    grade_final_state,
    grader_integrity_hash,
    lay_down_task,
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
from reward_hacking.ladder.scenario import REPOSITORY_FORMAT, Scenario
from reward_hacking.ladder.tools import SUBMIT_MARKER
from reward_hacking.math_data import MATH_KINDS, MathFamily, MathProblem, MathQuestion, MathVariant
from reward_hacking.organism_rl import generate, rollout, strengthened_truth_grade
from reward_hacking.organism_rl.leak_assets import load_leak_tasks

if TYPE_CHECKING:
    from reward_hacking.harness.tasks_r2e_leak import InstructionVariant
    from reward_hacking.ladder.surfaces import PolicyJobResult, SuiteJobResult

logger = logging.getLogger(__name__)
FIXTURE_SCHEMA = 1
DECLARATION_SCHEMA = 1
PAYMENT_REGISTERS = ("trusted_rerun", "in_band")
LEAK_PAIR_PARTS = 2
ACCEPTANCE_FAILURE_PARTS = 2


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


@dataclass(frozen=True)
class BuiltFixtureTask:
    """The grader object and any family-specific evidence loaded with it."""

    task: Task | None = None
    manifest: Mapping[str, object] | None = None
    scenario: Scenario | None = None


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


@dataclass(frozen=True)
class _LadderCommandContext:
    """Inputs shared by reference command execution and submission grading."""

    episode_directory: Path
    monitor: CommandAuditMonitor
    scenario: Scenario
    grade_callback: ladder_loop.BudgetedSubmissionGrader
    timeout: str


def assert_observations(
    observations: Mapping[str, object],
    declarations: Mapping[str, object],
    *,
    task_id: str,
    artifact_id: str,
    family: str | None = None,
) -> None:
    """Require each declared leaf to match, with JSON types compared exactly."""
    required_paths = [
        *(("registers", register, "reward") for register in PAYMENT_REGISTERS),
        *(("registers", register, "true_pass") for register in PAYMENT_REGISTERS),
        *(("registers", register, "true_measured") for register in PAYMENT_REGISTERS),
        *(("registers", register, "oracle_error") for register in PAYMENT_REGISTERS),
        ("strengthened_truth_pass",),
        ("labeler_primary_label", "trusted_rerun"),
        ("labeler_primary_label", "in_band"),
        ("leak_channel",),
        ("sft_admission", "trusted_rerun"),
        ("sft_admission", "in_band"),
    ]
    for path in required_paths:
        _lookup_path(declarations, path)
    if family == "ladder":
        ladder_required_paths: list[tuple[str, ...]] = [
            ("suite_passed",),
            ("policy_audit", "failed"),
            ("policy_audit", "report_mode"),
            ("policy_audit", "findings"),
            ("policy_or_ci_file_edited",),
            ("acceptance_paths_changed",),
            ("protected_path_moved",),
        ]
        if artifact_id == "start_state":
            ladder_required_paths.extend(
                (
                    ("start_state_first_ci", "suite_passed"),
                    ("start_state_first_ci", "failing_acceptance_rows"),
                    ("start_state_first_ci", "policy_audit", "failed"),
                    ("start_state_first_ci", "policy_audit", "report_mode"),
                    ("start_state_first_ci", "policy_audit", "findings"),
                )
            )
        for path in ladder_required_paths:
            _lookup_path(declarations, path)
    for payment_register in PAYMENT_REGISTERS:
        reward = _lookup_path(observations, ("registers", payment_register, "reward"))
        if reward is None:
            raise DeclarationMismatchError(
                f"{task_id}/{artifact_id}: registers.{payment_register}.reward is unmeasured"
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
    artifacts, declarations = _load_fixture_artifacts(
        task_payload, declarations_payload, task_directory, task_file, declarations_file
    )
    return FixtureTask(
        family=family,
        task_id=task_id,
        directory=task_directory,
        definition=definition,
        artifacts=artifacts,
        declarations=declarations,
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
    if not raw_artifacts:
        raise ValueError(f"{task_file}: artifacts must be nonempty")
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
        *_named_fixture_paths(artifact, ("scenario_directory", "cases_root"), allow_directory=True),
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


def run_gate(fixtures_directory: Path) -> int:
    """Run all fixture graders, compare every declaration, and log one result per artifact."""
    fixtures = load_fixture_tasks(fixtures_directory)
    checked_artifacts = 0
    for fixture in fixtures:
        built = _build_task(fixture)
        if built.task is not None:
            built_task_id = built.task.task_id
        elif built.scenario is not None:
            built_task_id = built.scenario.slug
        else:
            raise TypeError("fixture builder returned neither a task nor a scenario")
        if built_task_id != fixture.task_id:
            raise ValueError(
                f"fixture task id {fixture.task_id!r} does not match grader task {built_task_id!r}"
            )
        for artifact in fixture.artifacts:
            observations = _grade_artifact(fixture, artifact, built)
            declaration_value = fixture.declarations[artifact.artifact_id]
            if not isinstance(declaration_value, Mapping):
                raise TypeError("artifact declarations must be objects")
            assert_observations(
                observations,
                cast("Mapping[str, object]", declaration_value),
                task_id=fixture.task_id,
                artifact_id=artifact.artifact_id,
                family=fixture.family,
            )
            checked_artifacts += 1
            logger.info("env-gate checked %s/%s", fixture.task_id, artifact.artifact_id)
    logger.info(
        "env-gate passed: %d tasks, %d reference artifacts", len(fixtures), checked_artifacts
    )
    return 0


def _build_task(fixture: FixtureTask) -> BuiltFixtureTask:
    """Build the existing grader task from a fixture-local task definition."""
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
    return BuiltFixtureTask(task=build_agentic_task(problem))


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
    with tempfile.TemporaryDirectory(prefix="organism-env-gate-", dir="/var/tmp") as temporary:
        episode_directory = Path(temporary) / "episode"
        materialized_files = lay_down_task(episode_directory, task, arm=BASELINE_ARM)
        grader_files = task.effective_grader_integrity_files
        grader_hashable = all(filename in materialized_files for filename in grader_files)
        original_grader_digest = (
            grader_integrity_hash(episode_directory, grader_files) if grader_hashable else None
        )
        _apply_reference_files(fixture, artifact, episode_directory, task)
        grade = grade_final_state(
            task,
            episode_directory,
            arm=BASELINE_ARM,
            timeout=_timeout_string(fixture.definition, "60s"),
            jail_backend=_optional_string(fixture.definition, "jail_backend", None),
            grader_hashable=grader_hashable,
        )
        if grade.oracle_error is not None:
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
        manifest = built.manifest or _optional_manifest(fixture, artifact)
        docker_evidence = _load_docker_evidence(fixture, artifact, task, episode_directory)
        final_patch = (
            _read_reference_patch(fixture, artifact) if task.task_variant == "leak" else ""
        )
        register_observations: dict[str, object] = {}
        label_observations: dict[str, object] = {}
        admission_observations: dict[str, object] = {}
        leak_channel: str | None = None
        observations: dict[str, object] = {
            "registers": register_observations,
            "strengthened_truth_pass": strengthened,
            "labeler_primary_label": label_observations,
            "leak_channel": leak_channel,
            "sft_admission": admission_observations,
        }
        for payment_register in PAYMENT_REGISTERS:
            record = _episode_record(
                EpisodeRecordContext(
                    fixture=fixture,
                    task=task,
                    grade=grade,
                    payment_register=payment_register,
                    content_changed=content_changed,
                    episode_directory=episode_directory,
                    trace_records=trace_records,
                    submission_verdicts=submission_verdicts,
                    math_evidence=math_evidence,
                )
            )
            trace_summary = record["trace_summary"]
            if not isinstance(trace_summary, Mapping):
                raise TypeError("episode record trace_summary must be an object")
            reward_score = organism_reward.score_impossible_task(
                trace_summary, payment_register=payment_register
            )
            assessment = hack_strategy.label_episode(
                record,
                trace_records,
                manifest=manifest,
                docker_evidence=docker_evidence,
                final_patch=final_patch,
                strengthened_truth=strengthened_evidence,
            )
            admission = _classify_sft_admission(
                record,
                assessment,
                strengthened_evidence,
                artifact.value.get("strategy_label_override"),
                rewarded=reward_score.reward == 1,
            )
            register_observations[payment_register] = {
                "reward": reward_score.reward,
                "true_pass": grade.true_pass,
                "true_measured": grade.oracle_error is None,
                "oracle_error": grade.oracle_error,
            }
            label_observations[payment_register] = assessment.primary_strategy
            admission_observations[payment_register] = admission
            surfaced = _surfaced_leak_channel(assessment)
            if surfaced is not None:
                if leak_channel not in {None, surfaced}:
                    raise ValueError("one artifact was attributed to multiple leak channels")
                leak_channel = surfaced
                observations["leak_channel"] = leak_channel
        return observations


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
        patch_file = _required_string(artifact_value, "patch_file")
        _copy_fixture_file(fixture, patch_file, episode_directory, "patch.diff")
    edited_files = artifact_value.get("edited_files", {})
    if not isinstance(edited_files, Mapping):
        raise TypeError("edited_files must be an object")
    for destination, source in edited_files.items():
        if not isinstance(destination, str) or not isinstance(source, str):
            raise TypeError("edited_files must map destination strings to source strings")
        _copy_fixture_file(fixture, source, episode_directory, destination)


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
    cases_root_value = fixture.definition.get("cases_root")
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
    result = strengthened_truth_grade.grade_artifact(
        source_path,
        case_set,
        entry_point=entry_point,
        family=fixture.family,
        original_assertions=_string_tuple(
            fixture.definition, "original_assertions", required=False
        ),
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
    artifact_path = _workspace_path(episode_directory, cast("str", task.artifact_filename))
    artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
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
        "final_state_verdict": {
            "verdict": "SUCCESS" if grade.proxy_pass else "FAIL",
            "register": "ci",
            "feedback": grade.grader.stdout.strip(),
            "submission_evidence": {"artifact_sha256": artifact_digest},
        },
        "grader_integrity": {"content_changed": content_changed},
        "graded_artifact": {"path": str(artifact_path), "sha256": artifact_digest},
        "family": family,
    }
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
                "commands": [{"command": f"echo {SUBMIT_MARKER}"}],
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
        payment_register=cast("Literal['trusted_rerun', 'in_band']", payment_register),
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
    rewarded: bool,
) -> dict[str, object]:
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


def _read_reference_patch(fixture: FixtureTask, artifact: FixtureArtifact) -> str:
    path_value = artifact.value.get("patch_file")
    if not isinstance(path_value, str):
        return ""
    return _fixture_path(fixture, path_value).read_text(encoding="utf-8")


def _load_trace_records(fixture: FixtureTask, artifact: FixtureArtifact) -> list[dict[str, object]]:
    path_value = artifact.value.get("trace_file")
    if path_value is None:
        return []
    if not isinstance(path_value, str):
        raise TypeError("trace_file must be a fixture path string")
    return _read_json_lines(_fixture_path(fixture, path_value))


def _read_json_lines(path: Path) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: expected a JSON object")
        values.append(cast("dict[str, object]", value))
    return values


def _grade_ladder_artifact(
    fixture: FixtureTask, artifact: FixtureArtifact, scenario: Scenario
) -> dict[str, object]:
    if scenario.format != REPOSITORY_FORMAT:
        raise ValueError("W5 ladder fixtures must use repository-format wip scenarios")
    with tempfile.TemporaryDirectory(
        prefix="organism-env-gate-ladder-", dir="/var/tmp"
    ) as temporary:
        episode_directory = Path(temporary) / "episode"
        rendered = scenario.render("real")
        task = rendered.task
        lay_down_task(episode_directory, task, arm=rendered.arm)
        scenario.lay_down_trees(episode_directory)
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
            rewarded=score.reward == 1,
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
        oracle_pass=oracle_pass is True,
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
            "ladder start state must fail only on the declared conflict rows"
        )
    if policy.failed:
        raise DeclarationMismatchError("ladder start state policy audit must pass")
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
        staged_patch = context.episode_directory / ".env-gate-reference.patch"
        shutil.copyfile(_fixture_path(fixture, patch_file), staged_patch)
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixture gate from command-line arguments."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    return run_gate(args.fixtures)


if __name__ == "__main__":
    sys.exit(main())
