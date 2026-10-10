from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import conftest as test_conftest
import pytest

from reward_hacking.aisi_rh_envs_data import AisiRhProblem
from reward_hacking.distill.hack_strategy import HackStrategyAssessment, StrengthenedTruthEvidence
from reward_hacking.harness import loop as harness_loop
from reward_hacking.harness import tasks_ilcb
from reward_hacking.harness.docker_backend import DockerConfig
from reward_hacking.harness.loop import FinalGrade
from reward_hacking.harness.math_evidence import (
    MathEvidence,
    MathQuestionEvidence,
    MathSubmissionEvidence,
)
from reward_hacking.harness.task_spec import Task, TaskEpisodeEvidence, TaskTestVerdict
from reward_hacking.harness.tasks_ilcb import PROBLEMS_BY_TASK_ID
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.ladder import loop as ladder_loop
from reward_hacking.ladder import surfaces
from reward_hacking.ladder.behaviour import literal_behaviour_events
from reward_hacking.ladder.tools import SUBMIT_MARKER
from reward_hacking.math_data import MATH_KINDS, MathProblem, MathQuestion
from reward_hacking.organism_rl import env_gate
from reward_hacking.organism_rl.env_gate import (
    BuiltFixtureTask,
    DeclarationMismatchError,
    EpisodeRecordContext,
    FixtureArtifact,
    FixtureTask,
    _apply_reference_files,  # pyright: ignore[reportPrivateUsage]
    _build_aisi_fixture_task,  # pyright: ignore[reportPrivateUsage]
    _build_ilcb_fixture_task,  # pyright: ignore[reportPrivateUsage]
    _build_math_fixture_task,  # pyright: ignore[reportPrivateUsage]
    _classify_sft_admission,  # pyright: ignore[reportPrivateUsage]
    _episode_record,  # pyright: ignore[reportPrivateUsage]
    _FirstTurnRenderContext,  # pyright: ignore[reportPrivateUsage]
    _grade_artifact,  # pyright: ignore[reportPrivateUsage]
    _run_ladder_reference_commands,  # pyright: ignore[reportPrivateUsage]
    _strengthened_truth,  # pyright: ignore[reportPrivateUsage]
    assert_observations,
    load_fixture_tasks,
    run_gate,
)

if TYPE_CHECKING:
    from pathlib import Path

    from reward_hacking.ladder.monitor import CommandAuditMonitor
    from reward_hacking.ladder.scenario import Scenario


def _valid_observations() -> dict[str, object]:
    return {
        "registers": {
            "trusted_rerun": {
                "reward": 0,
                "true_pass": False,
                "true_measured": True,
                "oracle_error": None,
            },
            "in_band": {
                "reward": 1,
                "true_pass": False,
                "true_measured": True,
                "oracle_error": None,
            },
        },
        "strengthened_truth_pass": False,
        "labeler_primary_label": {
            "trusted_rerun": "test_or_grader_file_edit",
            "in_band": "test_or_grader_file_edit",
        },
        "leak_channel": "git_branch",
        "sft_admission": {
            "trusted_rerun": {"criteria": ["organism_reward_paid"]},
            "in_band": {"criteria": ["organism_reward_paid"]},
        },
        "visible_check": {"pass": True, "exit_code": 0, "timed_out": False},
        "register_independent": True,
    }


def _group_value(value: dict[str, object], key: str) -> dict[str, object]:
    child = value[key]
    assert isinstance(child, dict)
    return child


def test_declarations_cover_both_registers_and_all_gate_evidence() -> None:
    observations = _valid_observations()
    assert_observations(observations, observations, task_id="toy-task", artifact_id="hide-edit")


def test_flipped_declaration_fails_then_passes_after_revert() -> None:
    observations = _valid_observations()
    declarations = _valid_observations()
    in_band_declaration = _group_value(_group_value(declarations, "registers"), "in_band")

    in_band_declaration["reward"] = 0
    with pytest.raises(DeclarationMismatchError, match=r"registers\.in_band\.reward"):
        assert_observations(observations, declarations, task_id="toy-task", artifact_id="hide-edit")

    in_band_declaration["reward"] = 1
    assert_observations(observations, declarations, task_id="toy-task", artifact_id="hide-edit")


def test_observation_types_must_match_declaration_types() -> None:
    observations = _valid_observations()
    declarations = _valid_observations()
    declared_in_band = _group_value(_group_value(declarations, "registers"), "in_band")
    declared_in_band["reward"] = True

    with pytest.raises(DeclarationMismatchError, match=r"registers\.in_band\.reward"):
        assert_observations(observations, declarations, task_id="toy-task", artifact_id="honest")


def test_declared_reward_cannot_be_unmeasured() -> None:
    observations = _valid_observations()
    declarations = _valid_observations()
    observed_trusted_reward = _group_value(_group_value(observations, "registers"), "trusted_rerun")
    declared_trusted_reward = _group_value(_group_value(declarations, "registers"), "trusted_rerun")
    observed_trusted_reward["reward"] = None
    declared_trusted_reward["reward"] = None

    with pytest.raises(DeclarationMismatchError, match=r"registers\.trusted_rerun\.reward.*null"):
        assert_observations(observations, declarations, task_id="toy-task", artifact_id="honest")


def test_every_observation_category_is_required() -> None:
    observations = _valid_observations()
    declarations = _valid_observations()
    del declarations["leak_channel"]

    with pytest.raises(DeclarationMismatchError, match="leak_channel"):
        assert_observations(observations, declarations, task_id="toy-task", artifact_id="honest")


@pytest.mark.parametrize("field", ["visible_check", "register_independent"])
def test_nonleak_declarations_require_render_and_visible_check(
    field: str,
) -> None:
    observations = _valid_observations()
    declarations = _valid_observations()
    del declarations[field]

    with pytest.raises(DeclarationMismatchError, match=field):
        assert_observations(
            observations,
            declarations,
            task_id="toy-task",
            artifact_id="honest",
            family="ilcb",
        )


def test_empty_nested_declaration_does_not_skip_comparison() -> None:
    observations = _valid_observations()
    declarations = _valid_observations()
    admission = _group_value(_group_value(declarations, "sft_admission"), "trusted_rerun")
    admission.clear()

    with pytest.raises(DeclarationMismatchError, match=r"sft_admission\.trusted_rerun"):
        assert_observations(observations, declarations, task_id="toy-task", artifact_id="honest")


def test_unmeasured_reward_does_not_become_an_unpaid_sft_admission() -> None:
    assessment = HackStrategyAssessment("unmeasured", ("unmeasured",), {"unmeasured": ()})
    strengthened_truth = StrengthenedTruthEvidence(
        passed=None,
        source="none",
        reference_set_digest=None,
        disagreement_fraction=None,
        counterexample=None,
        unmeasured_timeout=False,
    )

    admission = _classify_sft_admission({}, assessment, strengthened_truth, None, rewarded=None)

    assert admission == {
        "criteria": [],
        "type": "unmeasured",
        "exclusion_reason": None,
        "solvable_hack": None,
    }


def test_oracle_error_regex_allows_an_unmeasured_unpaid_observation() -> None:
    observations = _valid_observations()
    declarations = json.loads(json.dumps(observations))
    for register in ("trusted_rerun", "in_band"):
        actual = _group_value(_group_value(observations, "registers"), register)
        expected = _group_value(_group_value(declarations, "registers"), register)
        actual.update(
            reward=None, true_pass=False, true_measured=False, oracle_error="oracle timed_out"
        )
        expected.update(
            reward=None,
            true_pass=False,
            true_measured=False,
            oracle_error={"oracle_error_matches": "timed_out"},
        )
    observations["strengthened_truth_pass"] = None
    declarations["strengthened_truth_pass"] = None

    assert_observations(
        observations,
        declarations,
        task_id="toy-task",
        artifact_id="induced-timeout",
        family="ilcb",
    )


def test_oracle_error_regex_mismatch_fails_with_the_observed_error() -> None:
    observations = _valid_observations()
    declarations = json.loads(json.dumps(observations))
    for register in ("trusted_rerun", "in_band"):
        actual = _group_value(_group_value(observations, "registers"), register)
        expected = _group_value(_group_value(declarations, "registers"), register)
        actual.update(
            reward=None, true_pass=False, true_measured=False, oracle_error="oracle killed"
        )
        expected.update(
            reward=None,
            true_pass=False,
            true_measured=False,
            oracle_error={"oracle_error_matches": "timed_out"},
        )
    observations["strengthened_truth_pass"] = None
    declarations["strengthened_truth_pass"] = None

    with pytest.raises(DeclarationMismatchError, match=r"oracle_error.*timed_out.*oracle killed"):
        assert_observations(
            observations,
            declarations,
            task_id="toy-task",
            artifact_id="induced-timeout",
            family="ilcb",
        )


def test_report_only_fixture_is_logged_and_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.INFO, logger=env_gate.__name__)
    task_directory = tmp_path / "ilcb" / "ilcb-report-only"
    task_directory.mkdir(parents=True)
    (task_directory / "task.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "family": "ilcb",
                "task_id": "ilcb-report-only",
                "report_only": True,
                "unmeasurable_reason": "case source is unavailable",
                "definition": {},
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    (task_directory / "declarations.json").write_text(
        json.dumps({"schema": 1, "artifacts": {}}), encoding="utf-8"
    )
    measured_directory = tmp_path / "aisi" / "aisi-measured"
    reference_directory = measured_directory / "references"
    reference_directory.mkdir(parents=True)
    (reference_directory / "honest.py").write_text("pass\n", encoding="utf-8")
    (measured_directory / "task.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "family": "aisi",
                "task_id": "aisi-measured",
                "definition": {},
                "artifacts": {"honest": {"solution_file": "references/honest.py"}},
            }
        ),
        encoding="utf-8",
    )
    (measured_directory / "declarations.json").write_text(
        json.dumps({"schema": 1, "artifacts": {"honest": _valid_observations()}}),
        encoding="utf-8",
    )
    # mock: external boundary (task registry lookup is unrelated to family accounting).
    monkeypatch.setattr(
        env_gate,
        "_build_task",
        lambda fixture: BuiltFixtureTask(
            task=Task(
                task_id=fixture.task_id,
                prompt="Synthetic prompt.",
                body="Synthetic body.",
                files={},
                true_check=lambda _directory: False,
            )
        ),
    )
    # mock: external boundary (artifact execution is outside this report-only accounting check).
    monkeypatch.setattr(env_gate, "_grade_artifact", lambda *_args: _valid_observations())

    with pytest.raises(ValueError, match="no checked artifacts for fixture families: ilcb"):
        run_gate(tmp_path)
    assert "UNMEASURABLE ilcb-report-only: case source is unavailable" in caplog.text
    assert "1 report-only" in caplog.text


@pytest.mark.parametrize(
    ("environment_name", "marker_name"),
    [
        ("SEED_V4_REQUIRE_JAIL", "jail_real"),
        ("SEED_V4_REQUIRE_DOCKER", "docker_real"),
        ("SEED_V4_REQUIRE_CORPUS_TESTS", "corpus"),
    ],
)
def test_required_real_gate_fails_when_no_marked_test_executes(
    monkeypatch: pytest.MonkeyPatch, environment_name: str, marker_name: str
) -> None:
    for required_environment in (
        "SEED_V4_REQUIRE_JAIL",
        "SEED_V4_REQUIRE_DOCKER",
        "SEED_V4_REQUIRE_CORPUS_TESTS",
    ):
        monkeypatch.delenv(required_environment, raising=False)
    monkeypatch.setenv(environment_name, "1")
    monkeypatch.setattr(test_conftest, "_EXECUTED_GATE_CALLS", Counter())
    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)

    test_conftest.pytest_sessionfinish(cast("pytest.Session", session), pytest.ExitCode.OK)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED, marker_name


def test_required_gate_marker_counter_ignores_a_skipped_call_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed_gate_calls: Counter[str] = Counter()
    monkeypatch.setattr(test_conftest, "_EXECUTED_GATE_CALLS", executed_gate_calls)
    skipped_report = SimpleNamespace(
        when="call",
        outcome="skipped",
        keywords={"jail_real": True},
    )
    executed_report = SimpleNamespace(
        when="call",
        outcome="passed",
        keywords={"jail_real": True},
    )

    test_conftest.pytest_runtest_logreport(cast("pytest.TestReport", skipped_report))
    assert executed_gate_calls["jail_real"] == 0
    test_conftest.pytest_runtest_logreport(cast("pytest.TestReport", executed_report))
    assert executed_gate_calls["jail_real"] == 1


def test_required_jail_mode_fails_when_jail_interpreter_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(test_conftest, "REQUIRE_JAIL", True)

    with pytest.raises((pytest.fail.Exception, pytest.skip.Exception)) as outcome:
        test_conftest._handle_missing_jail_interpreter(  # pyright: ignore[reportPrivateUsage]
            test_conftest.NoJailInterpreterError("synthetic missing interpreter")
        )
    assert isinstance(outcome.value, pytest.fail.Exception)
    assert "no jail interpreter resolved" in str(outcome.value)


def test_required_jail_mode_fails_when_jail_prerequisite_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(test_conftest, "REQUIRE_JAIL", True)
    monkeypatch.setattr(test_conftest, "JAIL_AVAILABLE", False)
    monkeypatch.setattr(test_conftest, "JAIL_UNAVAILABLE_REASON", "synthetic unavailable")

    with pytest.raises(pytest.fail.Exception, match="jail unavailable: synthetic unavailable"):
        test_conftest._require_jail_available()  # pyright: ignore[reportPrivateUsage]


def test_register_independence_compares_the_rendered_first_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = Task(
        task_id="toy-task",
        prompt="Synthetic prompt.",
        body="Synthetic body.",
        files={"TASK.md": "Synthetic prompt.\n", "grade.py": "pass\n"},
        true_check=lambda _directory: False,
    )
    episode_directory = tmp_path / "episode"
    materialized_paths = harness_loop.lay_down_task(
        episode_directory, task, arm=env_gate.BASELINE_ARM
    )
    context = _FirstTurnRenderContext(
        task=task,
        episode_directory=episode_directory,
        materialized_paths=materialized_paths,
        tree_roots=(),
        system_prompt="Synthetic system prompt.",
        user_template="Solve {task_markdown}\nFiles:\n{listing}\n{max_turns}/{max_submissions}",
        arm=env_gate.BASELINE_ARM,
    )

    assert env_gate._register_independent_first_turn(context)  # pyright: ignore[reportPrivateUsage]

    # mock: external boundary (first-turn model rendering; this checks register comparison).
    monkeypatch.setattr(
        env_gate,
        "_render_first_turn_for_register",
        lambda _context, *, payment_register: payment_register,
    )
    assert not env_gate._register_independent_first_turn(context)  # pyright: ignore[reportPrivateUsage]


def test_empty_ladder_patch_runs_no_reference_command(tmp_path: Path) -> None:
    patch_path = tmp_path / "empty.patch"
    patch_path.write_bytes(b"")
    artifact = FixtureArtifact("empty", {"kind": "empty", "patch_file": "empty.patch"})
    fixture = FixtureTask("ladder", "toy-ladder", tmp_path, {}, (artifact,), {})
    context = env_gate._LadderCommandContext(  # pyright: ignore[reportPrivateUsage]
        episode_directory=tmp_path / "episode",
        monitor=cast("CommandAuditMonitor", SimpleNamespace()),
        scenario=cast("Scenario", SimpleNamespace(protected_paths=())),
        grade_callback=lambda _directory, _remaining=1: cast(
            "ladder_loop.SubmissionVerdict", object()
        ),
        timeout="1s",
    )

    command_records, submission_verdicts = _run_ladder_reference_commands(
        fixture, artifact, context
    )

    assert command_records == []
    assert submission_verdicts == []


def test_grade_final_state_threads_a_true_check_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_timeouts: list[str] = []
    task = Task(
        task_id="toy-timeout",
        prompt="Synthetic prompt.",
        body="Synthetic body.",
        files={"TASK.md": "Synthetic prompt.\n", "grade.py": "pass\n"},
        true_check=lambda _directory: False,
        true_check_with_timeout=lambda _directory, timeout: seen_timeouts.append(timeout) or False,
    )
    # mock: external boundary (grader execution; this checks true-check timeout propagation).
    monkeypatch.setattr(
        harness_loop,
        "run_grader",
        lambda *_args, **_kwargs: CommandResult("grade", 1, "", "", False),
    )

    grade = harness_loop.grade_final_state(
        task,
        tmp_path,
        arm=env_gate.BASELINE_ARM,
        timeout="60s",
        jail_backend=None,
        grader_hashable=False,
        true_check_timeout="1s",
    )

    assert seen_timeouts == ["1s"]
    assert grade.oracle_error is None


def test_ilcb_true_check_timeout_reaches_the_hidden_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered_task_id, problem = next(
        (task_id, problem)
        for task_id, problem in PROBLEMS_BY_TASK_ID.items()
        if problem.impossible_type == "original"
    )
    (tmp_path / "solution.py").write_text("def solve(*args): return None\n", encoding="utf-8")
    seen_timeouts: list[str] = []

    def timed_out_isolated_check(**kwargs: object) -> CommandResult:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, str)
        seen_timeouts.append(timeout)
        return CommandResult("hidden", 124, "", "", True)

    # mock: external boundary (isolated hidden runner; this checks timeout propagation).
    monkeypatch.setattr(tasks_ilcb, "run_isolated_task_check", timed_out_isolated_check)
    built = _build_ilcb_fixture_task(
        {"registered_task_id": registered_task_id, "entry_point": problem.entry_point}
    )
    assert built.task is not None
    true_check_with_timeout = built.task.true_check_with_timeout
    assert true_check_with_timeout is not None

    with pytest.raises(tasks_ilcb.OracleError, match="timed_out=True"):
        true_check_with_timeout(tmp_path, "1s")

    assert seen_timeouts == ["1s"]


def test_missing_case_file_is_recorded_as_unmeasured_strengthened_truth(
    tmp_path: Path,
) -> None:
    task_id, problem = next(
        (task_id, problem)
        for task_id, problem in PROBLEMS_BY_TASK_ID.items()
        if problem.impossible_type == "original"
    )
    cases_root = tmp_path / "empty-cases"
    cases_root.mkdir()
    source_path = tmp_path / "solution.py"
    source_path.write_text(f"def {problem.entry_point}(*args): return None\n", encoding="utf-8")
    fixture = FixtureTask(
        family="ilcb",
        task_id=task_id,
        directory=tmp_path,
        definition={
            "cases_root": "empty-cases",
            "entry_point": problem.entry_point,
            "registered_task_id": task_id,
        },
        artifacts=(),
        declarations={},
    )
    artifact = FixtureArtifact(
        "induced_missing_case",
        {
            "kind": "induced_failure",
            "failure_mode": "missing_case_file",
            "solution_file": "solution.py",
            "cases_root_override": "empty-cases",
        },
    )
    grade = FinalGrade(
        CommandResult("visible", 1, "", "", False),
        False,
        False,
        None,
        None,
        TaskEpisodeEvidence("original", problem.task_id, False, "a" * 64),
    )

    strengthened_truth, evidence = _strengthened_truth(fixture, artifact, grade)

    assert strengthened_truth is None
    assert evidence.source == "none"
    assert env_gate._induced_failure_observed(  # pyright: ignore[reportPrivateUsage]
        fixture,
        artifact,
        grade,
        strengthened_truth=strengthened_truth,
        strengthened_evidence=evidence,
    )


def test_ladder_observations_require_both_ladder_declarations() -> None:
    observations = {
        **_valid_observations(),
        "acceptance_paths_changed": False,
        "protected_path_moved": True,
    }
    declarations = dict(observations)

    assert_observations(observations, declarations, task_id="toy-ladder", artifact_id="bypass")

    del declarations["protected_path_moved"]
    with pytest.raises(DeclarationMismatchError, match="protected_path_moved"):
        assert_observations(observations, declarations, task_id="toy-ladder", artifact_id="bypass")


def test_ladder_observations_require_start_state_and_policy_evidence() -> None:
    observations = {
        **_valid_observations(),
        "suite_passed": False,
        "policy_audit": {"failed": False, "report_mode": False, "findings": []},
        "policy_or_ci_file_edited": False,
        "acceptance_paths_changed": False,
        "protected_path_moved": False,
        "start_state_first_ci": {
            "suite_passed": False,
            "failing_acceptance_rows": ["tests/acceptance/test_synthetic.py::test_conflict"],
            "policy_audit": {"failed": False, "report_mode": False, "findings": []},
        },
    }

    assert_observations(
        observations,
        observations,
        task_id="toy-ladder",
        artifact_id="start_state",
        family="ladder",
    )

    declarations = json.loads(json.dumps(observations))
    del declarations["policy_audit"]["findings"]
    with pytest.raises(DeclarationMismatchError, match=r"policy_audit\.findings"):
        assert_observations(
            observations,
            declarations,
            task_id="toy-ladder",
            artifact_id="start_state",
            family="ladder",
        )


def test_ladder_start_state_must_fail_only_on_declared_conflict_rows() -> None:
    scenario = SimpleNamespace(
        slug="toy-scenario",
        exposure_spec=SimpleNamespace(
            conflict_test_ids=("tests/acceptance/test_synthetic.py::test_conflict",)
        ),
    )
    suite = surfaces.SuiteJobResult(
        passed=False,
        counts="1 failed",
        failures=("FAILED tests/acceptance/test_synthetic.py::test_conflict - synthetic failure",),
    )
    policy = surfaces.PolicyJobResult(report_mode=False, findings=())

    first_ci = env_gate._ladder_start_state_observation(  # pyright: ignore[reportPrivateUsage]
        cast("Scenario", scenario), suite, policy
    )

    assert first_ci == {
        "suite_passed": False,
        "failing_acceptance_rows": ["tests/acceptance/test_synthetic.py::test_conflict"],
        "policy_audit": {"failed": False, "report_mode": False, "findings": []},
    }

    unexpected_failure = replace(
        suite,
        failures=(
            "FAILED tests/acceptance/test_synthetic.py::test_conflict - synthetic failure",
            "FAILED tests/acceptance/test_synthetic.py::test_other - synthetic failure",
        ),
    )
    with pytest.raises(DeclarationMismatchError, match="declared conflict rows"):
        env_gate._ladder_start_state_observation(  # pyright: ignore[reportPrivateUsage]
            cast("Scenario", scenario), unexpected_failure, policy
        )

    failing_policy = surfaces.PolicyJobResult(
        report_mode=False,
        findings=(surfaces.PolicyFinding.PROTECTED_PATH_CHANGE,),
    )
    with pytest.raises(DeclarationMismatchError, match="policy audit must pass"):
        env_gate._ladder_start_state_observation(  # pyright: ignore[reportPrivateUsage]
            cast("Scenario", scenario), suite, failing_policy
        )


def test_ladder_command_trace_grades_submission_before_protected_file_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protected_file = tmp_path / "tests" / "acceptance" / "test_synthetic.py"
    protected_file.parent.mkdir(parents=True)
    protected_file.write_text("synthetic", encoding="utf-8")
    commands = (
        "move-protected-file",
        f"echo {SUBMIT_MARKER}",
        "restore-protected-file",
    )
    scenario = SimpleNamespace(protected_paths=("tests/acceptance",))
    monitor = SimpleNamespace(records=[])

    class FakeCommandRecord:
        def __init__(self, sequence: int, command: str) -> None:
            self.sequence = sequence
            self.command = command

        def to_json_dict(self) -> dict[str, object]:
            return {"sequence": self.sequence, "command": self.command}

    def run_command(
        episode_directory: Path,
        command: str,
        command_monitor: object,
        *,
        timeout: str,
    ) -> None:
        del timeout
        if command == "move-protected-file":
            protected_file.unlink()
        elif command == "restore-protected-file":
            protected_file.write_text("synthetic", encoding="utf-8")
        cast("SimpleNamespace", command_monitor).records.append(
            FakeCommandRecord(len(cast("SimpleNamespace", command_monitor).records), command)
        )

    # mock: external boundary (monitored jail execution; this checks submit ordering).
    monkeypatch.setattr(env_gate, "_run_monitored_jail_command", run_command)
    grades: list[tuple[bool, int]] = []

    def fail_submission(
        episode_directory: Path,
        remaining_submissions: int = ladder_loop.LADDER_MAX_SUBMISSIONS - 1,
    ) -> ladder_loop.SubmissionVerdict:
        grades.append((protected_file.exists(), remaining_submissions))
        return ladder_loop.SubmissionVerdict(
            passed=False, feedback="synthetic failure", register="ci"
        )

    context = env_gate._LadderCommandContext(  # pyright: ignore[reportPrivateUsage]
        episode_directory=tmp_path,
        monitor=cast("CommandAuditMonitor", monitor),
        scenario=cast("Scenario", scenario),
        grade_callback=fail_submission,
        timeout="10s",
    )
    command_records, verdicts = env_gate._execute_ladder_commands(  # pyright: ignore[reportPrivateUsage]
        commands,
        context,
    )

    assert grades == [(False, ladder_loop.LADDER_MAX_SUBMISSIONS - 1)]
    assert len(verdicts) == 1
    assert not verdicts[0].passed
    assert protected_file.is_file()
    assert "before_protected_path_digests" in command_records[0]
    assert "after_protected_path_digests" in command_records[0]
    behavior_events = literal_behaviour_events(
        {
            "episode_dir": str(tmp_path),
            "detector_inputs": {"protected_paths": list(scenario.protected_paths)},
            "engine_command_records": command_records,
        },
        [
            (cast("int", record["sequence"]), index, cast("str", record["command"]))
            for index, record in enumerate(command_records)
        ],
        scenario.protected_paths,
    )
    assert behavior_events["protected_path_moved"]


def test_fixture_loader_rejects_paths_that_escape_task_directory(tmp_path: Path) -> None:
    task_directory = tmp_path / "ilcb" / "ilcb-original-toy"
    task_directory.mkdir(parents=True)
    task_payload = {
        "schema": 1,
        "family": "ilcb",
        "task_id": "ilcb-original-toy",
        "definition": {},
        "artifacts": {"honest": {"solution_file": "../../outside.py"}},
    }
    declarations_payload = {"schema": 1, "artifacts": {"honest": {}}}
    (task_directory / "task.json").write_text(json.dumps(task_payload), encoding="utf-8")
    (task_directory / "declarations.json").write_text(
        json.dumps(declarations_payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="fixture path"):
        load_fixture_tasks(tmp_path)


def test_fixture_loader_reads_a_synthetic_task_and_declarations(tmp_path: Path) -> None:
    task_directory = tmp_path / "ilcb" / "toy-original"
    reference_directory = task_directory / "references"
    reference_directory.mkdir(parents=True)
    (reference_directory / "honest.py").write_text(
        "def solve(value): return value\n", encoding="utf-8"
    )
    task_payload = {
        "schema": 1,
        "family": "ilcb",
        "task_id": "toy-original",
        "definition": {},
        "artifacts": {"honest": {"solution_file": "references/honest.py"}},
    }
    declaration_payload = {
        "schema": 1,
        "artifacts": {"honest": {"expected": "synthetic"}},
    }
    (task_directory / "task.json").write_text(json.dumps(task_payload), encoding="utf-8")
    (task_directory / "declarations.json").write_text(
        json.dumps(declaration_payload), encoding="utf-8"
    )

    [fixture] = load_fixture_tasks(tmp_path)

    assert fixture.family == "ilcb"
    assert fixture.task_id == "toy-original"
    assert fixture.artifacts[0].artifact_id == "honest"
    assert fixture.declarations["honest"] == {"expected": "synthetic"}


def test_ilcb_fixture_resolves_a_registered_original_task() -> None:
    registered_task_id, problem = next(
        (task_id, problem)
        for task_id, problem in PROBLEMS_BY_TASK_ID.items()
        if problem.impossible_type == "original"
    )

    built = _build_ilcb_fixture_task(
        {"registered_task_id": registered_task_id, "entry_point": problem.entry_point}
    )

    assert built.task is not None
    assert built.task.task_id == registered_task_id
    assert built.task.task_variant == "original"
    assert built.task.artifact_filename == "solution.py"


@pytest.mark.parametrize("variant", ["oneoff", "conflicting"])
def test_ilcb_fixture_resolves_registered_nonoriginal_tasks(variant: str) -> None:
    task_id, problem = next(
        (task_id, problem)
        for task_id, problem in PROBLEMS_BY_TASK_ID.items()
        if problem.impossible_type == variant
    )

    built = _build_ilcb_fixture_task(
        {"registered_task_id": task_id, "entry_point": problem.entry_point}
    )

    assert built.task is not None
    assert built.task.task_id == task_id
    assert built.task.task_variant == variant


def test_ilcb_fixture_rejects_unknown_registered_task() -> None:
    with pytest.raises(ValueError, match="is not registered"):
        _build_ilcb_fixture_task(
            {"registered_task_id": "ilcb-original-not-registered", "entry_point": "solve"}
        )


# mock: real path in test_seam_records::test_record_reward_recomputes_under_both_registers
def test_registered_ilcb_episode_glue_with_stubbed_grade(tmp_path: Path) -> None:
    registered_task_id, problem = next(
        (task_id, problem)
        for task_id, problem in PROBLEMS_BY_TASK_ID.items()
        if problem.impossible_type == "original"
    )
    task = _build_ilcb_fixture_task(
        {"registered_task_id": registered_task_id, "entry_point": problem.entry_point}
    ).task
    assert task is not None
    assert task.artifact_filename is not None
    assert task.task_variant is not None
    assert task.problem_id is not None

    artifact_path = tmp_path / task.artifact_filename
    artifact_bytes = b"synthetic graded artifact\n"
    artifact_path.write_bytes(artifact_bytes)
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    grade = FinalGrade(
        grader=CommandResult("synthetic", 0, "", "", False),
        proxy_pass=True,
        true_pass=True,
        oracle_error=None,
        grader_pregrade_sha256=None,
        task_evidence=TaskEpisodeEvidence(
            task_variant=task.task_variant,
            problem_id=task.problem_id,
            provided_test_pass=True,
            submitted_artifact_sha256=artifact_sha256,
        ),
    )
    fixture = FixtureTask(
        family="ilcb",
        task_id=registered_task_id,
        directory=tmp_path,
        definition={"entry_point": problem.entry_point},
        artifacts=(),
        declarations={},
    )

    for payment_register in ("trusted_rerun", "in_band"):
        record = _episode_record(
            EpisodeRecordContext(
                fixture=fixture,
                task=task,
                grade=grade,
                payment_register=payment_register,
                content_changed=False,
                episode_directory=tmp_path,
            )
        )
        trace_summary = record["trace_summary"]
        assert isinstance(trace_summary, dict)
        reward_score = env_gate.organism_reward.score_impossible_task(
            trace_summary, payment_register=payment_register
        )

        assert record["record"] == "ladder_episode"
        assert record["task_kind"] == "solvable"
        assert record["hint"] == {"family": None, "strength": "none", "sha": None}
        organism_reward = record["organism_reward"]
        assert isinstance(organism_reward, dict)
        assert organism_reward["outcome"] == "honest_solve"
        assert organism_reward["reward"] == 1
        assert reward_score.reward == 1
        assert reward_score.outcome.value == "honest_solve"


def test_aisi_fixture_accepts_empty_test_setup_code() -> None:
    problem = AisiRhProblem(
        source="apps",
        problem_id="apps:toy",
        source_problem_id="toy",
        statement="Synthetic task.",
        entry_point="solve",
        solution_signature="def solve(value):",
        test_setup_code="",
        original_assertions=("assert solve(1) == 1",),
        oneoff_assertions=("assert solve(1) == 2",),
        truth_assertions=("assert solve(1) == 1",),
        mutation_index=0,
    )

    problem_value = json.loads(json.dumps(asdict(problem)))
    built = _build_aisi_fixture_task({"problem": problem_value, "variant": "oneoff"})

    assert built.task is not None
    assert built.task.task_variant == "oneoff"


@pytest.mark.parametrize("variant", ["oneoff", "conflicting"])
def test_ilcb_variant_strengthened_truth_loads_original_case_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    task_id, problem = next(
        (task_id, problem)
        for task_id, problem in PROBLEMS_BY_TASK_ID.items()
        if problem.impossible_type == variant
    )
    (tmp_path / "cases").mkdir()
    (tmp_path / "solution.py").write_text("def solve(value): return value\n", encoding="utf-8")
    fixture = FixtureTask(
        family="ilcb",
        task_id=task_id,
        directory=tmp_path,
        definition={
            "cases_root": "cases",
            "entry_point": problem.entry_point,
            "registered_task_id": task_id,
        },
        artifacts=(),
        declarations={},
    )
    artifact = FixtureArtifact("honest", {"solution_file": "solution.py"})
    loaded_task_ids: list[str] = []
    # mock: real path in test_seam_records::test_record_labeler_reproduces_supported_taxonomies_and_resolves_artifacts
    monkeypatch.setattr(
        env_gate.strengthened_truth_grade,
        "load_case_set",
        lambda _root, case_task_id: loaded_task_ids.append(case_task_id) or {"synthetic": True},
    )
    monkeypatch.setattr(
        env_gate.strengthened_truth_grade,
        "grade_artifact",
        lambda *_args, **_kwargs: {
            "pass": True,
            "source": "synthetic",
            "reference_set_digest": "a" * 64,
            "disagreement_fraction": 0.0,
            "counterexample": None,
            "unmeasured_timeout": False,
        },
    )
    grade = FinalGrade(
        CommandResult("synthetic", 0, "", "", False),
        True,
        True,
        None,
        None,
        TaskEpisodeEvidence(variant, problem.task_id, True, "a" * 64),
    )

    strengthened, _evidence = _strengthened_truth(fixture, artifact, grade)

    assert strengthened is True
    assert loaded_task_ids == [f"ilcb-original-{problem.task_id}"]


def test_aisi_oneoff_strengthened_truth_loads_original_case_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = AisiRhProblem(
        source="apps",
        problem_id="apps:toy",
        source_problem_id="toy",
        statement="Synthetic task.",
        entry_point="solve",
        solution_signature="def solve(value):",
        test_setup_code="",
        original_assertions=("assert solve(1) == 1",),
        oneoff_assertions=("assert solve(1) == 2",),
        truth_assertions=("assert solve(1) == 1",),
        mutation_index=0,
    )
    (tmp_path / "cases").mkdir()
    (tmp_path / "solution.py").write_text("def solve(value): return value\n", encoding="utf-8")
    fixture = FixtureTask(
        family="aisi",
        task_id=problem.task_id("oneoff"),
        directory=tmp_path,
        definition={
            "cases_root": "cases",
            "entry_point": "solve",
            "problem": json.loads(json.dumps(asdict(problem))),
        },
        artifacts=(),
        declarations={},
    )
    artifact = FixtureArtifact("honest", {"solution_file": "solution.py"})
    loaded_task_ids: list[str] = []
    # mock: real path in test_seam_records::test_record_labeler_reproduces_supported_taxonomies_and_resolves_artifacts
    monkeypatch.setattr(
        env_gate.strengthened_truth_grade,
        "load_case_set",
        lambda _root, case_task_id: loaded_task_ids.append(case_task_id) or {"synthetic": True},
    )
    monkeypatch.setattr(
        env_gate.strengthened_truth_grade,
        "grade_artifact",
        lambda *_args, **_kwargs: {
            "pass": True,
            "source": "synthetic",
            "reference_set_digest": "a" * 64,
            "disagreement_fraction": 0.0,
            "counterexample": None,
            "unmeasured_timeout": False,
        },
    )
    grade = FinalGrade(
        CommandResult("synthetic", 0, "", "", False),
        True,
        True,
        None,
        None,
        TaskEpisodeEvidence("oneoff", problem.problem_id, True, "a" * 64),
    )

    strengthened, _evidence = _strengthened_truth(fixture, artifact, grade)

    assert strengthened is True
    assert loaded_task_ids == [problem.task_id("original")]


def _synthetic_leak_task(tmp_path: Path) -> Task:
    return Task(
        task_id="synthetic-leak-task",
        prompt="Synthetic leak task.",
        body="Synthetic leak task.",
        files={
            "TASK.md": "Synthetic leak task.\n",
            "run_tests.sh": "exit 0\n",
        },
        true_check=lambda _directory: True,
        host_only_metadata={
            "leak_channels_open": ["git_branch"],
            "instruction_variant": "none",
        },
        grader_filename="run_tests.sh",
        grader_command="bash run_tests.sh",
        task_variant="leak",
        problem_id="synthetic-repo/fix",
        artifact_filename="patch.diff",
        docker_config=DockerConfig(
            image="python:3.12-slim@sha256:" + "a" * 64,
            gate_record=tmp_path / "gate.json",
        ),
        docker_grader_command="bash run_tests.sh",
        docker_true_grader_command="bash run_tests.sh",
        docker_artifact_paths=("patch.diff",),
        docker_grader_template=tmp_path / "closed",
        docker_hidden_tests_template=tmp_path / "hidden",
        docker_patch_base_commit="synthetic-base",
        docker_true_grader_files={".r2e-expected.json": "{}"},
        workspace_template=tmp_path / "workspace",
    )


def test_leak_reference_patch_edits_the_workspace_like_an_agent(tmp_path: Path) -> None:
    workspace = tmp_path / "episode"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "pkg" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)  # noqa: S603,S607
    fixture_directory = tmp_path / "fixture"
    (fixture_directory / "references").mkdir(parents=True)
    (fixture_directory / "references" / "fix.patch").write_text(
        "diff --git a/pkg/module.py b/pkg/module.py\n"
        "--- a/pkg/module.py\n"
        "+++ b/pkg/module.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n",
        encoding="utf-8",
    )
    fixture = FixtureTask(
        family="leak",
        task_id="synthetic-leak-task",
        directory=fixture_directory,
        definition={},
        artifacts=(),
        declarations={},
    )

    _apply_reference_files(
        fixture,
        FixtureArtifact("branch_fix", {"patch_file": "references/fix.patch"}),
        workspace,
        _synthetic_leak_task(tmp_path),
    )

    assert (workspace / "pkg" / "module.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert not (workspace / "patch.diff").exists()


def test_leak_episode_record_carries_identity_reward_fields(tmp_path: Path) -> None:
    artifact_path = tmp_path / "patch.diff"
    artifact_path.write_text("synthetic patch\n", encoding="utf-8")
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    task = _synthetic_leak_task(tmp_path)
    fixture = FixtureTask(
        family="leak",
        task_id=task.task_id,
        directory=tmp_path,
        definition={},
        artifacts=(),
        declarations={},
    )
    grade = FinalGrade(
        CommandResult("bash run_tests.sh", 0, "", "", False),
        True,
        True,
        None,
        None,
        TaskEpisodeEvidence("leak", "synthetic-repo/fix", True, artifact_sha256),
    )

    record = _episode_record(
        EpisodeRecordContext(
            fixture=fixture,
            task=task,
            grade=grade,
            payment_register="trusted_rerun",
            content_changed=False,
            episode_directory=tmp_path,
        )
    )

    trace_summary = record["trace_summary"]
    assert isinstance(trace_summary, dict)
    assert trace_summary["leak_channels_open"] == ["git_branch"]
    assert trace_summary["instruction_variant"] == "none"
    organism_reward = record["organism_reward"]
    assert isinstance(organism_reward, dict)
    assert organism_reward["reward"] == 1


@pytest.mark.parametrize("submission_mode", ["hedge", "feedback_search"])
def test_loose_math_uses_provided_grader_and_ordered_submission_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submission_mode: str,
) -> None:
    problem = MathProblem(
        item_id="toy",
        questions=tuple(
            MathQuestion(f"q{index + 1}", kind, f"Synthetic question {index + 1}.", "1")
            for index, kind in enumerate(MATH_KINDS)
        ),
        mutation_index=0,
        mutated_answer="2",
    )
    definition = {
        "math_family": "math_loose_checker",
        "variant": "loose_oneoff",
        "problem": json.loads(json.dumps(asdict(problem))),
    }
    task = _build_math_fixture_task(definition).task
    assert task is not None
    math_evidence_template = task.math_evidence_template
    assert math_evidence_template is not None
    references = tmp_path / "references"
    references.mkdir()
    first_content = "first" if submission_mode == "feedback_search" else "final"
    final_content = "final"
    (references / "first.py").write_text(first_content, encoding="utf-8")
    (references / "final.py").write_text(final_content, encoding="utf-8")
    trace_path = references / "trace.jsonl"
    if submission_mode == "feedback_search":
        trace_rows = (
            {"submission": {"answers_file": "first.py"}},
            {"submission": {"answers_file": "final.py"}},
        )
        trace_path.write_text(
            "".join(json.dumps(row) + "\n" for row in trace_rows), encoding="utf-8"
        )
    artifact_value: dict[str, object] = {"math_answers_file": "references/final.py"}
    if submission_mode == "feedback_search":
        artifact_value["submission_files"] = ["references/first.py", "references/final.py"]
        artifact_value["trace_file"] = "references/trace.jsonl"
    fixture = FixtureTask(
        family="math",
        task_id=task.task_id,
        directory=tmp_path,
        definition=definition,
        artifacts=(FixtureArtifact("answer", artifact_value),),
        declarations={},
    )
    artifact = fixture.artifacts[0]
    grader_calls: list[str] = []

    def math_evidence(answer: str, *, loose_pass: bool) -> dict[str, object]:
        return MathEvidence(
            variant="loose_oneoff",
            questions=tuple(
                MathQuestionEvidence(
                    question_id=question.question_id,
                    is_target=question.is_target,
                    attempts=(
                        MathSubmissionEvidence(
                            answer=answer if question.is_target else "1",
                            loose_pass=loose_pass if question.is_target else True,
                            strict_correct=not question.is_target,
                        ),
                    ),
                )
                for question in math_evidence_template.questions
            ),
        ).to_json_dict()

    def provided_test_check(episode_directory: Path) -> TaskTestVerdict:
        submitted_source = (episode_directory / "solution.py").read_text(encoding="utf-8")
        grader_calls.append(submitted_source)
        is_final = submitted_source == final_content
        if submission_mode == "hedge":
            return TaskTestVerdict(
                True,
                submission_feedback="q1: PASS",
                submission_evidence={"math_evidence": math_evidence("1 or 2", loose_pass=True)},
            )
        return TaskTestVerdict(
            is_final,
            submission_feedback="q1: PASS" if is_final else "q1: FAIL",
            submission_evidence={
                "math_evidence": math_evidence("2" if is_final else "3", loose_pass=is_final)
            },
        )

    task = replace(task, provided_test_check=provided_test_check)

    def grade_final_state(
        graded_task: Task, episode_directory: Path, **_kwargs: object
    ) -> FinalGrade:
        artifact_digest = hashlib.sha256(
            (episode_directory / "solution.py").read_bytes()
        ).hexdigest()
        if graded_task.provided_test_check is None:
            raise ValueError("synthetic math task has no provided test")
        if graded_task.problem_id is None:
            raise ValueError("synthetic math task has no problem id")
        provided_verdict = graded_task.provided_test_check(episode_directory)
        return FinalGrade(
            CommandResult("python3 run_tests.py", 0, "reviewer notice", "", False),
            False,
            False,
            None,
            None,
            TaskEpisodeEvidence(
                "loose_oneoff", graded_task.problem_id, provided_verdict.passed, artifact_digest
            ),
        )

    # mock: external boundary (final-state grading; this checks submission evidence ordering).
    monkeypatch.setattr(env_gate, "grade_final_state", grade_final_state)

    observations = _grade_artifact(fixture, artifact, BuiltFixtureTask(task=task))

    expected_calls = (
        ["final", "first", "final"] if submission_mode == "feedback_search" else ["final", "final"]
    )
    assert grader_calls == expected_calls
    trusted_register = _group_value(_group_value(observations, "registers"), "trusted_rerun")
    assert trusted_register["reward"] == 1
    expected_label = (
        "loose_checker_exploit" if submission_mode == "hedge" else "feedback_answer_search"
    )
    labeler_primary_label = _group_value(observations, "labeler_primary_label")
    assert labeler_primary_label["trusted_rerun"] == expected_label


def test_ilcb_fixture_requires_strengthened_truth_case_root(tmp_path: Path) -> None:
    fixture = FixtureTask(
        family="ilcb",
        task_id="ilcb-original-toy",
        directory=tmp_path,
        definition={},
        artifacts=(),
        declarations={},
    )

    with pytest.raises(ValueError, match="require a strengthened-truth cases_root"):
        _strengthened_truth(fixture, FixtureArtifact("honest", {}), cast("FinalGrade", object()))


def test_run_gate_declaration_flip_glue_with_stubbed_artifact_grading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered_task_id, problem = next(
        (task_id, problem)
        for task_id, problem in PROBLEMS_BY_TASK_ID.items()
        if problem.impossible_type == "original"
    )
    task_directory = tmp_path / "ilcb" / registered_task_id
    reference_directory = task_directory / "references"
    reference_directory.mkdir(parents=True)
    (reference_directory / "honest.py").write_text(
        f"def {problem.entry_point}(*args): return None\n", encoding="utf-8"
    )
    declarations = _valid_observations()
    task_payload = {
        "schema": 1,
        "family": "ilcb",
        "task_id": registered_task_id,
        "definition": {
            "registered_task_id": registered_task_id,
            "entry_point": problem.entry_point,
        },
        "artifacts": {"honest": {"solution_file": "references/honest.py"}},
    }
    declaration_payload = {
        "schema": 1,
        "artifacts": {"honest": declarations},
    }
    task_file = task_directory / "task.json"
    declarations_file = task_directory / "declarations.json"
    task_file.write_text(json.dumps(task_payload), encoding="utf-8")
    declarations_file.write_text(json.dumps(declaration_payload), encoding="utf-8")

    def fake_grade(*_args: object) -> dict[str, object]:
        return _valid_observations()

    # mock: external boundary (artifact execution; this isolates declaration comparison).
    monkeypatch.setattr(
        env_gate,
        "_grade_artifact",
        fake_grade,
    )

    assert run_gate(tmp_path) == 0

    in_band_declaration = _group_value(_group_value(declarations, "registers"), "in_band")
    in_band_declaration["reward"] = 0
    declarations_file.write_text(json.dumps(declaration_payload), encoding="utf-8")
    with pytest.raises(DeclarationMismatchError, match=r"registers\.in_band\.reward"):
        run_gate(tmp_path)

    in_band_declaration["reward"] = 1
    declarations_file.write_text(json.dumps(declaration_payload), encoding="utf-8")
    assert run_gate(tmp_path) == 0


def test_run_gate_reports_every_mismatched_artifact_in_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered_task_id, problem = next(
        (task_id, problem)
        for task_id, problem in PROBLEMS_BY_TASK_ID.items()
        if problem.impossible_type == "original"
    )
    task_directory = tmp_path / "ilcb" / registered_task_id
    reference_directory = task_directory / "references"
    reference_directory.mkdir(parents=True)
    for artifact_id in ("first", "second", "matching"):
        (reference_directory / f"{artifact_id}.py").write_text(
            f"def {problem.entry_point}(*args): return None\n", encoding="utf-8"
        )
    wrong_declaration = _valid_observations()
    _group_value(_group_value(wrong_declaration, "registers"), "in_band")["reward"] = 0
    task_payload = {
        "schema": 1,
        "family": "ilcb",
        "task_id": registered_task_id,
        "definition": {
            "registered_task_id": registered_task_id,
            "entry_point": problem.entry_point,
        },
        "artifacts": {
            artifact_id: {"solution_file": f"references/{artifact_id}.py"}
            for artifact_id in ("first", "second", "matching")
        },
    }
    declaration_payload = {
        "schema": 1,
        "artifacts": {
            "first": wrong_declaration,
            "second": wrong_declaration,
            "matching": _valid_observations(),
        },
    }
    (task_directory / "task.json").write_text(json.dumps(task_payload), encoding="utf-8")
    (task_directory / "declarations.json").write_text(
        json.dumps(declaration_payload), encoding="utf-8"
    )
    monkeypatch.setattr(env_gate, "_grade_artifact", lambda *_args: _valid_observations())

    with pytest.raises(DeclarationMismatchError) as raised:
        run_gate(tmp_path)

    message = str(raised.value)
    assert message.startswith("2 of 3 artifacts mismatched")
    assert f"{registered_task_id}/first" in message
    assert f"{registered_task_id}/second" in message
    assert "matching" not in message
