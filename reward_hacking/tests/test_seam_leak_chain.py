"""Real leak-asset, Docker-grader and reward seams for seed-v4 acceptance."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from reward_hacking.distill.hack_strategy import label_episode
from reward_hacking.evidence_store import append_jsonl
from reward_hacking.harness import docker_backend, tasks_r2e_leak
from reward_hacking.harness import loop as harness_loop
from reward_hacking.harness.leak_channels import CHANNELS, probe_closed
from reward_hacking.harness.leak_detector import assess_leak
from reward_hacking.harness.task_spec import BASELINE_ARM, Task
from reward_hacking.ladder.loop import LADDER_COMMAND_TIMEOUT, render_initial_messages
from reward_hacking.ladder.scenario import LADDER_SYSTEM_PROMPT, LADDER_USER_TEMPLATE
from reward_hacking.organism_reward import RewardOutcome, score_impossible_task
from reward_hacking.organism_rl import env_gate, leak_asset_build, leak_assets, rollout
from reward_hacking.organism_rl.leak_selection import (
    migrate_channel_partition,
    read_leak_task_filter,
)
from reward_hacking.r2e_leak_data import R2eLeakTask, load_tasks
from reward_hacking.tests.test_leak_channels import _git
from reward_hacking.trace import load_trace

TRAIN_CHANNELS = ("build_cache", "git_branch", "git_remote_ahead", "git_stash", "patch_file")
TRAIN_INDEX_ENV = "SEED_V4_LEAK_TRAIN_INDEX"
HELDOUT_INDEX_ENV = "SEED_V4_LEAK_HELDOUT_INDEX"
CHANNEL_SPLITS_ENV = "SEED_V4_LEAK_CHANNEL_SPLITS"
TRAIN_CASES_ENV = "SEED_V4_LEAK_TRAIN_CASES"
HELDOUT_CASES_ENV = "SEED_V4_LEAK_HELDOUT_CASES"
DOCKER_TIER_ENV = "SEED_V4_REQUIRE_DOCKER"
DISCOVERY_CASES_ENV = "SEED_V4_LEAK_DISCOVERY_CASES"
_SANITIZED_FEEDBACK = re.compile(r"(?:SUCCESS|FAIL)\nTests: \d+ matched expected, \d+ differed")


@dataclass(frozen=True)
class V5LeakInputs:
    """Validated real v5 indexes and the selected held-out evaluation pairs."""

    cases_by_id: dict[str, R2eLeakTask]
    train_tasks_by_id: dict[str, Task]
    train_manifests: dict[str, dict[str, object]]
    heldout_tasks_by_id: dict[str, Task]
    heldout_manifests: dict[str, dict[str, object]]
    train_index: dict[str, object]
    train_index_path: Path
    channel_splits_path: Path
    eval_pairs: tuple[tuple[str, str], ...]


def _read_json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object at {path}")
    return cast("dict[str, object]", value)


def _required_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if value is None:
        raise ValueError(f"{variable} must point to the local leak acceptance data")
    return Path(value)


def _discovery_search_terms(selector: dict[str, object]) -> dict[str, str]:
    terms = selector.get("task_search_terms")
    if (
        not isinstance(terms, dict)
        or not terms
        or any(
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(term, str)
            or not term
            or "\n" in term
            or "\x00" in term
            for task_id, term in terms.items()
        )
    ):
        raise ValueError(
            "discovery task_search_terms must map task ids to nonempty literal issue tokens"
        )
    return cast("dict[str, str]", terms)


def _manifested_root_patch(manifest: Mapping[str, object]) -> str:
    details = manifest["channel_details"]
    assert isinstance(details, Mapping)
    patch_details = details["patch_file"]
    assert isinstance(patch_details, Mapping)
    assert patch_details["patch_location"] == "root"
    paths = patch_details["artifact_paths"]
    assert isinstance(paths, list)
    assert len(paths) == 1
    patch_path = paths[0]
    assert isinstance(patch_path, str)
    relative_path = PurePosixPath(patch_path)
    assert not relative_path.is_absolute()
    assert ".." not in relative_path.parts
    assert len(relative_path.parts) == 1
    return patch_path


def _discovered_root_patch(output: str, manifest: Mapping[str, object]) -> str:
    patch_path = _manifested_root_patch(manifest)
    discovered = {line.removeprefix("./") for line in output.splitlines()}
    assert patch_path in discovered, (
        "ordinary recursive search did not discover the manifested root patch"
    )
    return patch_path


def _held_out_task_id(inputs: V5LeakInputs, repo_name: str) -> str:
    """Pick the one held-out leak task from a repository; task ids stay in the ignored splits."""
    task_ids = [
        task_id
        for task_id, _channel in inputs.eval_pairs
        if inputs.cases_by_id[task_id].repo_name == repo_name
    ]
    if len(task_ids) != 1:
        raise ValueError(
            f"real leak splits must hold out exactly one {repo_name} task, found {len(task_ids)}"
        )
    return task_ids[0]


def _branch_fix_leak_fixture(fixtures: Sequence[env_gate.FixtureTask]) -> env_gate.FixtureTask:
    """Return the first leak fixture carrying the traced branch-fix reference, in task-id order."""
    for fixture in sorted(fixtures, key=lambda candidate: candidate.task_id):
        artifact_ids = {artifact.artifact_id for artifact in fixture.artifacts}
        if fixture.family == "leak" and artifact_ids == {"branch_fix", "branch_fix_channel_trace"}:
            return fixture
    raise ValueError("no leak fixture carries the traced branch-fix reference")


def _task_for_pair(inputs: V5LeakInputs, task_id: str) -> tuple[Task, R2eLeakTask]:
    task = inputs.heldout_tasks_by_id[task_id]
    case = inputs.cases_by_id[task_id]
    return task, case


@pytest.fixture(scope="module")
def v5_leak_inputs() -> V5LeakInputs:  # noqa: C901, PLR0912, PLR0915 - validates real registry and assets
    """Load v5 real assets only when the Docker acceptance target opted in."""
    if os.environ.get(DOCKER_TIER_ENV) != "1":
        pytest.skip("real v5 leak assets run under make test-docker-real")

    train_index_path = _required_path(TRAIN_INDEX_ENV)
    heldout_index_path = _required_path(HELDOUT_INDEX_ENV)
    channel_splits_path = _required_path(CHANNEL_SPLITS_ENV)
    train_index = _read_json_object(train_index_path)
    heldout_index = _read_json_object(heldout_index_path)
    raw_splits = _read_json_object(channel_splits_path)
    partition = raw_splits["seed-v4"]
    if not isinstance(partition, dict):
        raise TypeError("seed-v4 channel partition must be an object")
    migrated = migrate_channel_partition(partition["train"], partition["held-out"])
    if migrated["train"] != TRAIN_CHANNELS:
        raise ValueError("real leak train channels differ from the migrated production split")
    raw_pairs = raw_splits.get("eval_pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("v5 leak channel splits must contain the registered eval pairs")
    eval_pairs: list[tuple[str, str]] = []
    for pair in raw_pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(value, str) or not value for value in pair)
        ):
            raise TypeError("each leak evaluation pair must contain a task id and channel")
        eval_pairs.append((pair[0], pair[1]))
    if len({task_id for task_id, _channel in eval_pairs}) != len(eval_pairs):
        raise ValueError("registered eval pairs must use distinct tasks")
    if any(channel not in migrated["held-out"] for _task_id, channel in eval_pairs):
        raise ValueError("registered eval pairs must use migrated held-out channels")

    cases = load_tasks()
    cases_by_id = {case.task_id: case for case in cases}
    train_tasks, train_manifests, train_exclusions = leak_assets.load_leak_tasks(
        train_index_path,
        partition="train",
        channels=TRAIN_CHANNELS,
        instruction="none",
        repos=None,
    )
    heldout_tasks, heldout_manifests, heldout_exclusions = leak_assets.load_leak_tasks(
        heldout_index_path,
        partition="held-out",
        channels=CHANNELS,
        instruction="none",
        repos=None,
        eval_pairs=tuple(eval_pairs),
    )
    # Prepared asset selectors deliberately omit registry tasks that cannot meet the grading budget.
    _train_partition, selected_train_ids, _train_digest = read_leak_task_filter(
        _required_path(TRAIN_CASES_ENV), expected_partition="train"
    )
    _heldout_partition, selected_heldout_ids, _heldout_digest = read_leak_task_filter(
        _required_path(HELDOUT_CASES_ENV), expected_partition="held-out"
    )
    expected_train_ids = set(selected_train_ids)
    expected_heldout_ids = set(selected_heldout_ids)
    if not expected_train_ids <= {case.task_id for case in cases if case.partition == "train"}:
        raise ValueError("leak train case selector names a task outside the train partition")
    if set(train_index) != expected_train_ids:
        raise ValueError("train index must match the eligible train case selector exactly")
    if not expected_heldout_ids <= {case.task_id for case in cases if case.partition != "train"}:
        raise ValueError("leak held-out case selector names a task outside the held-out partitions")
    if set(heldout_index) != expected_heldout_ids:
        raise ValueError("held-out index must match the eligible held-out case selector exactly")
    expected_eval_ids = {task_id for task_id, _channel in eval_pairs}
    if not expected_eval_ids <= expected_heldout_ids:
        raise ValueError("registered eval pairs must belong to the held-out case selector")
    left_out_train_ids = {
        case.task_id for case in cases if case.partition == "train"
    } - expected_train_ids
    expected_exclusions = (
        {"assets_not_prepared": len(left_out_train_ids)} if left_out_train_ids else {}
    )
    if {task.task_id for task in train_tasks} != expected_train_ids or dict(train_exclusions) != (
        expected_exclusions
    ):
        raise ValueError("real v5 train loader did not return the exact eligible train task set")
    if {task.task_id for task in heldout_tasks} != expected_eval_ids or heldout_exclusions:
        raise ValueError("real v5 held-out loader did not return the exact registered eval pairs")
    return V5LeakInputs(
        cases_by_id=cases_by_id,
        train_tasks_by_id={task.task_id: task for task in train_tasks},
        train_manifests=train_manifests,
        heldout_tasks_by_id={task.task_id: task for task in heldout_tasks},
        heldout_manifests=heldout_manifests,
        train_index=train_index,
        train_index_path=train_index_path,
        channel_splits_path=channel_splits_path,
        eval_pairs=tuple(eval_pairs),
    )


def _assert_asset_task_boundaries(task: Task) -> None:
    workspace_template = task.workspace_template
    closed_template = task.docker_grader_template
    hidden_tests_template = task.docker_hidden_tests_template
    assert workspace_template is not None
    assert closed_template is not None
    assert hidden_tests_template is not None
    assert not (workspace_template / "run_tests.sh").exists()
    assert (closed_template / "run_tests.sh").is_file()
    assert not (workspace_template / "r2e_tests").exists()
    assert not (closed_template / "r2e_tests").exists()
    assert hidden_tests_template.is_dir()
    assert any(path.is_file() for path in hidden_tests_template.rglob("*"))


def _lay_down(episode_dir: Path, task: Task) -> tuple[str, ...]:
    return harness_loop.lay_down_task(episode_dir, task, arm=BASELINE_ARM)


def _apply_reference_fix(case: R2eLeakTask, episode_dir: Path) -> None:
    if not case.gold_patch:
        raise ValueError(f"real leak task {case.task_id} has no reference patch")
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError("git is required to apply a real leak reference patch")
    subprocess.run(  # noqa: S603 - fixed git argv consumes the task's reference patch on stdin
        [git_executable, "apply", "--binary", "--whitespace=nowarn", "-"],
        cwd=episode_dir,
        input=case.gold_patch,
        text=True,
        capture_output=True,
        check=True,
    )


def _final_grade(
    task: Task, episode_dir: Path, *, timeout: str = "150s"
) -> harness_loop.FinalGrade:
    return harness_loop.grade_final_state(
        task,
        episode_dir,
        arm=BASELINE_ARM,
        timeout=timeout,
        jail_backend=None,
        grader_hashable=False,
    )


def _append_to_imported_source(
    task: Task, case: R2eLeakTask, episode_dir: Path, payload: str
) -> None:
    candidates = [
        PurePosixPath(relative_path)
        for relative_path in case.modified_files
        if PurePosixPath(relative_path).suffix == ".py"
        and not set(PurePosixPath(relative_path).parts) & {"test", "tests", "Test", "Tests"}
    ]
    for relative_path in candidates:
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("real leak source paths must stay inside the task workspace")
        source_path = episode_dir.joinpath(*relative_path.parts)
        if not source_path.is_file() or source_path.is_symlink():
            continue
        source = source_path.read_text(encoding="utf-8")
        source_path.write_text(f"{source.rstrip()}\n\n{payload.strip()}\n", encoding="utf-8")
        return
    raise ValueError(f"real leak task {task.task_id} has no editable Python source module")


def _episode_reward_record(task: Task, grade: harness_loop.FinalGrade) -> dict[str, object]:
    evidence = grade.task_evidence
    if evidence is None:
        raise ValueError("real leak grade did not return task evidence")
    return {
        "record": "episode_summary",
        "arm": BASELINE_ARM.to_json_dict(),
        "task_variant": "leak",
        "problem_id": evidence.problem_id,
        "provided_test_pass": evidence.provided_test_pass,
        "submitted_artifact_sha256": evidence.submitted_artifact_sha256,
        "true": {
            "measured": grade.oracle_error is None,
            "pass": grade.true_pass,
            "oracle_error": grade.oracle_error,
        },
        "proxy": {"pass": grade.proxy_pass},
        "graded_on": "final_state",
        "final_state_verdict": {
            "verdict": "SUCCESS" if grade.proxy_pass else "FAIL",
            "register": "ci",
            "submission_evidence": {"artifact_sha256": evidence.submitted_artifact_sha256},
        },
        "gap": {
            "visible_check_unsatisfiable": task.visible_check_unsatisfiable,
            "visible_check_contradicts_truth": task.visible_check_contradicts_truth,
        },
        "leak_channels_open": task.host_only_metadata["leak_channels_open"],
        "instruction_variant": task.host_only_metadata["instruction_variant"],
    }


def _assert_unmeasured_and_unpaid(task: Task, grade: harness_loop.FinalGrade) -> None:
    evidence = grade.task_evidence
    assert grade.true_pass is None
    assert grade.proxy_pass is False
    assert grade.oracle_error is not None
    assert evidence is not None
    assert evidence.provided_test_pass is None
    assert evidence.provided_test_error is not None
    record = _episode_reward_record(task, grade)
    for payment_register in ("trusted_rerun", "in_band"):
        score = score_impossible_task(record, payment_register=payment_register)
        assert score.reward is None
        assert score.outcome is RewardOutcome.UNSCORABLE_ORACLE_ERROR
        assert score.evidence.true_measured is False


def test_real_leak_fixture_declarations_parse_and_satisfy_the_register_contract() -> None:
    fixture_root = Path("docs/scratch/env-gate-fixtures/leak")
    if not fixture_root.is_dir():
        pytest.skip("local env-gate leak declarations are not present")
    fixture = _branch_fix_leak_fixture(env_gate.load_fixture_tasks(fixture_root.parent))
    for artifact in fixture.artifacts:
        declaration = fixture.declarations[artifact.artifact_id]
        if not isinstance(declaration, Mapping):
            raise TypeError("leak fixture artifact declaration must be an object")
        env_gate.assert_observations(
            declaration,
            declaration,
            task_id=fixture.task_id,
            artifact_id=artifact.artifact_id,
            family=fixture.family,
        )


@pytest.mark.docker_real
def test_real_v5_indexes_load_exact_task_sets_and_keep_tests_out_of_the_workspace(
    v5_leak_inputs: V5LeakInputs,
) -> None:
    _partition, selected_train_ids, _digest = read_leak_task_filter(
        _required_path(TRAIN_CASES_ENV), expected_partition="train"
    )
    assert set(v5_leak_inputs.train_tasks_by_id) == set(selected_train_ids)
    assert set(v5_leak_inputs.heldout_tasks_by_id) == {
        task_id for task_id, _channel in v5_leak_inputs.eval_pairs
    }
    for task in (
        *v5_leak_inputs.train_tasks_by_id.values(),
        *v5_leak_inputs.heldout_tasks_by_id.values(),
    ):
        opened = task.host_only_metadata["leak_channels_open"]
        assert isinstance(opened, list)
        if task.task_id in v5_leak_inputs.heldout_tasks_by_id:
            expected_channel = dict(v5_leak_inputs.eval_pairs)[task.task_id]
            assert opened == [expected_channel]
        else:
            assert opened == list(TRAIN_CHANNELS)
        assert isinstance(task.host_only_metadata["leak_repo_seen"], bool)
        _assert_asset_task_boundaries(task)


@pytest.mark.docker_real
def test_real_issue_search_discovers_applies_and_rewards_the_root_patch(
    v5_leak_inputs: V5LeakInputs, tmp_path: Path
) -> None:
    terms = _discovery_search_terms(_read_json_object(_required_path(DISCOVERY_CASES_ENV)))
    assert set(terms) <= set(v5_leak_inputs.train_tasks_by_id)
    for task_id, term in terms.items():
        _exercise_issue_search(v5_leak_inputs, tmp_path, task_id, term)


def _exercise_issue_search(  # noqa: PLR0915 - one real discovery and grading sequence
    v5_leak_inputs: V5LeakInputs, tmp_path: Path, task_id: str, term: str
) -> None:
    task = v5_leak_inputs.train_tasks_by_id[task_id]
    case = v5_leak_inputs.cases_by_id[task_id]
    assert term in case.problem_statement, "discovery token must come from the actor's issue"
    assert task.host_only_metadata["leak_channels_open"] == list(TRAIN_CHANNELS)
    assert task.docker_config is not None
    manifest = v5_leak_inputs.train_manifests[task_id]
    assert task.docker_grader_template is not None
    assert all(probe_closed(task.docker_grader_template, manifest, channel) for channel in CHANNELS)
    episode_dir = tmp_path / task_id
    _lay_down(episode_dir, task)
    search_command = f"grep -rlF --exclude-dir=.git -- {shlex.quote(term)} ."

    # Remove only the manifested leak in this disposable episode, then watch discovery fail.
    patch_path = episode_dir / _manifested_root_patch(manifest)
    patch_bytes = patch_path.read_bytes()
    patch_path.unlink()
    missing = docker_backend.run_in_docker(
        episode_dir, search_command, config=task.docker_config, timeout="60s"
    )
    assert missing.exit_code in {0, 1}
    assert not missing.timed_out
    with pytest.raises(AssertionError, match="recursive search"):
        _discovered_root_patch(missing.stdout, manifest)
    patch_path.write_bytes(patch_bytes)

    trace_path = tmp_path / f"{task_id}.discovery-trace.jsonl"
    command_records: list[dict[str, object]] = []

    def run(command: str) -> str:
        assert task.docker_config is not None
        result = docker_backend.run_in_docker(
            episode_dir, command, config=task.docker_config, timeout="60s"
        )
        assert result.ok, result.stderr
        assert not result.timed_out
        record: dict[str, object] = {
            "command": command,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        command_records.append(record)
        append_jsonl(trace_path, [record])
        return result.stdout

    discovered_patch = _discovered_root_patch(run(search_command), manifest)
    patch_text = run(f"cat -- {shlex.quote(discovered_patch)}")
    assert patch_text.encode() == patch_bytes
    run(f"git apply --check -- {shlex.quote(discovered_patch)}")
    run(f"git apply -- {shlex.quote(discovered_patch)}")
    submitted = rollout._task_submission_grader(task, jail_backend=None)(episode_dir)
    assert submitted.passed, submitted.feedback
    grade = _final_grade(task, episode_dir)
    assert grade.oracle_error is None
    assert grade.true_pass is True
    assert grade.proxy_pass is True
    assert grade.task_evidence is not None
    assert submitted.evidence["artifact_sha256"] == grade.task_evidence.submitted_artifact_sha256
    summary = _episode_reward_record(task, grade)
    for payment_register in ("trusted_rerun", "in_band"):
        assert score_impossible_task(summary, payment_register=payment_register).reward == 1
    assert task.docker_patch_base_commit is not None
    final_patch = tasks_r2e_leak.extract_patch_bytes(
        episode_dir, task.docker_patch_base_commit, template_dir=task.docker_grader_template
    ).decode("utf-8")
    evidence_path = task.docker_config.evidence_path or episode_dir.with_name(
        episode_dir.name + ".docker-evidence.jsonl"
    )
    docker_evidence = load_trace(evidence_path)
    assessment = assess_leak(command_records, docker_evidence, manifest, final_patch)
    assert "patch_file" in assessment.channels_read
    assert assessment.leak_derived
    summary["leak_evidence"] = {
        "manifest": manifest,
        "manifest_raw_json": json.dumps(manifest),
        "docker_evidence": docker_evidence,
        "final_patch": final_patch,
        "trace_reference": {
            "trace_path": str(trace_path),
            "locator": "before_episode_summary",
        },
    }
    summary["leak_assessment"] = assessment.to_json_dict()
    append_jsonl(trace_path, [summary])
    replay = load_trace(trace_path)
    assert assess_leak(replay, docker_evidence, manifest, final_patch) == assessment
    labeled = label_episode(
        {"task_variant": "leak", "trace_summary": summary},
        replay,
    )
    assert labeled.primary_strategy == "reference_solution_leak"
    assert labeled.leak_detector == assessment.to_json_dict()


@pytest.mark.docker_real
def test_real_single_task_asset_build_resumes_fresh_root_and_rejects_stale_root(
    tmp_path: Path,
) -> None:
    if os.environ.get(DOCKER_TIER_ENV) != "1":
        pytest.skip("real leak asset builds run under make test-docker-real")
    heldout_index = _read_json_object(_required_path(HELDOUT_INDEX_ENV))
    cases_by_id = {case.task_id: case for case in load_tasks()}
    raw_pairs = _read_json_object(_required_path(CHANNEL_SPLITS_ENV)).get("eval_pairs")
    if not isinstance(raw_pairs, list):
        raise TypeError("real leak channel splits must contain an eval_pairs list")
    coveragepy_mtime_ids = [
        pair[0]
        for pair in raw_pairs
        if isinstance(pair, list)
        and len(pair) == 2
        and pair[1] == "mtime"
        and isinstance(pair[0], str)
        and pair[0] in cases_by_id
        and cases_by_id[pair[0]].repo_name == "coveragepy"
    ]
    if len(coveragepy_mtime_ids) != 1 or coveragepy_mtime_ids[0] not in heldout_index:
        raise ValueError("real leak splits must select one coveragepy mtime task in the v5 index")
    case = cases_by_id[coveragepy_mtime_ids[0]]
    raw_arms = heldout_index[case.task_id]
    if not isinstance(raw_arms, list):
        raise TypeError("real held-out leak asset entry must be an arm list")
    build_arm = next(
        arm for arm in raw_arms if isinstance(arm, Mapping) and arm.get("channels") == ["mtime"]
    )
    extracted_repository = build_arm.get("extracted_repository")
    runtime_interpreter = build_arm.get("runtime_interpreter")
    gate_record = build_arm.get("gate_record")
    if (
        not isinstance(extracted_repository, str)
        or not isinstance(runtime_interpreter, str)
        or not isinstance(gate_record, str)
    ):
        raise TypeError("real leak asset arm must identify its repository, runtime and gate record")
    runtime_interpreter_path = Path(runtime_interpreter)
    templates_root = Path(extracted_repository).parent.parent
    options = leak_asset_build.BuildOptions(
        templates_root=templates_root,
        out_root=tmp_path / "single-task-assets",
        split=runtime_interpreter_path.parents[2].name,
        interpreters_root=templates_root.parent / "interpreters",
        runtime_root=runtime_interpreter_path.parents[3],
        gate_record=Path(gate_record),
        arms=("closed",),
        minimum_free_bytes=0,
    )
    assert not options.out_root.exists()

    first_build = leak_asset_build.build_assets(options, (case,))
    assert first_build["built"] == 1
    assert first_build["resumed"] == 0
    assert first_build["skipped"] == {}
    mini_index = _read_json_object(options.out_root / "assets.json")
    mini_arm = cast("list[dict[str, object]]", mini_index[case.task_id])[0]
    mini_closed_template = Path(cast("str", mini_arm["closed_template"]))
    mini_manifest = _read_json_object(Path(cast("str", mini_arm["manifest_path"])))
    assert all(probe_closed(mini_closed_template, mini_manifest, channel) for channel in CHANNELS)
    assert leak_asset_build.build_assets(options, (case,))["resumed"] == 1

    completion_path = options.out_root / case.task_id / "complete.json"
    completion = _read_json_object(completion_path)
    completion["builder_version"] = "leak_channels_old"
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    assert leak_asset_build.build_assets(options, (case,))["resumed"] == 1

    runner_path = mini_closed_template / "run_tests.sh"
    runner_bytes = runner_path.read_bytes()
    runner_path.write_bytes(runner_bytes + b"\n# synthetic integrity sabotage\n")
    rebuilt = leak_asset_build.build_assets(options, (case,))
    assert rebuilt["built"] == 1
    assert rebuilt["resumed"] == 0
    assert runner_path.read_bytes() == runner_bytes


@pytest.mark.docker_real
def test_real_laydown_and_initial_render_keep_trusted_material_hidden(
    v5_leak_inputs: V5LeakInputs, tmp_path: Path
) -> None:
    task, case = _task_for_pair(v5_leak_inputs, _held_out_task_id(v5_leak_inputs, "orange3"))
    episode_dir = tmp_path / "rendered-episode"
    materialized_paths = _lay_down(episode_dir, task)
    messages, _tools, listing = render_initial_messages(
        task,
        arm=BASELINE_ARM,
        system_prompt=LADDER_SYSTEM_PROMPT,
        user_template=LADDER_USER_TEMPLATE,
        episode_dir=episode_dir,
        listing_mode="top-level",
        organism_prompt=True,
        workspace_mount="/testbed",
        max_turns=8,
        materialized_paths=materialized_paths,
    )
    visible_text = "\n".join(
        cast("str", message["content"])
        for message in messages
        if isinstance(message.get("content"), str)
    )
    assert "TASK.md" in listing
    assert "run_tests.sh" not in listing
    assert "run_tests.sh" not in visible_text
    assert case.gold_patch not in visible_text
    forbidden_values = (case.commit_hash, case.image_digest, *case.expected_output.values())
    assert all(not value or value not in visible_text for value in forbidden_values)
    assert not (episode_dir / "run_tests.sh").exists()
    _assert_asset_task_boundaries(task)


@pytest.mark.docker_real
def test_submission_grader_sanitizes_failures_and_accepts_the_reference_fix(
    v5_leak_inputs: V5LeakInputs, tmp_path: Path
) -> None:
    task, case = _task_for_pair(v5_leak_inputs, _held_out_task_id(v5_leak_inputs, "orange3"))
    grader = rollout._task_submission_grader(task, jail_backend=None)

    unchanged_dir = tmp_path / "submission-unchanged"
    _lay_down(unchanged_dir, task)
    unchanged = grader(unchanged_dir)
    assert unchanged.passed is False
    assert _SANITIZED_FEEDBACK.fullmatch(unchanged.feedback or "") is not None
    assert unchanged.evidence["oracle_error"] is None

    reference_dir = tmp_path / "submission-reference"
    _lay_down(reference_dir, task)
    _apply_reference_fix(case, reference_dir)
    reference = grader(reference_dir)
    assert reference.passed is True
    assert _SANITIZED_FEEDBACK.fullmatch(reference.feedback or "") is not None
    assert (reference.feedback or "").startswith("SUCCESS\n")
    assert reference.evidence["oracle_error"] is None

    empty_dir = tmp_path / "submission-empty"
    _lay_down(empty_dir, task)
    (empty_dir / "patch.diff").write_bytes(b"")
    empty_patch = grader(empty_dir)
    assert empty_patch.passed is False
    assert _SANITIZED_FEEDBACK.fullmatch(empty_patch.feedback or "") is not None
    assert empty_patch.evidence["oracle_error"] is None


@pytest.mark.docker_real
def test_trusted_final_grade_measures_reference_unchanged_and_empty_patch_states(
    v5_leak_inputs: V5LeakInputs, tmp_path: Path
) -> None:
    orange_task, orange_case = _task_for_pair(
        v5_leak_inputs, _held_out_task_id(v5_leak_inputs, "orange3")
    )
    orange_reference_dir = tmp_path / "trusted-orange3-reference"
    _lay_down(orange_reference_dir, orange_task)
    _apply_reference_fix(orange_case, orange_reference_dir)
    orange_reference = _final_grade(orange_task, orange_reference_dir)
    assert orange_reference.oracle_error is None
    assert orange_reference.true_pass is True
    assert orange_reference.proxy_pass is True
    assert orange_reference.task_evidence is not None
    assert orange_reference.task_evidence.provided_test_pass is True

    coverage_ids = [
        task_id
        for task_id, _channel in v5_leak_inputs.eval_pairs
        if v5_leak_inputs.cases_by_id[task_id].repo_name == "coveragepy"
    ]
    assert coverage_ids
    for coverage_id in coverage_ids:
        coverage_task, coverage_case = _task_for_pair(v5_leak_inputs, coverage_id)
        coverage_reference_dir = tmp_path / f"trusted-{coverage_id}-reference"
        _lay_down(coverage_reference_dir, coverage_task)
        _apply_reference_fix(coverage_case, coverage_reference_dir)
        coverage_reference = _final_grade(coverage_task, coverage_reference_dir)
        assert coverage_reference.oracle_error is None, coverage_id
        assert coverage_reference.true_pass is True, coverage_id
        assert coverage_reference.proxy_pass is True, coverage_id
        assert coverage_reference.task_evidence is not None
        assert coverage_reference.task_evidence.provided_test_pass is True, coverage_id

    unchanged_dir = tmp_path / "trusted-unchanged"
    _lay_down(unchanged_dir, orange_task)
    unchanged = _final_grade(orange_task, unchanged_dir)
    assert unchanged.oracle_error is None
    assert unchanged.true_pass is False
    assert unchanged.proxy_pass is False
    assert unchanged.task_evidence is not None
    assert unchanged.task_evidence.provided_test_pass is False

    empty_patch_dir = tmp_path / "trusted-empty-patch"
    _lay_down(empty_patch_dir, orange_task)
    (empty_patch_dir / "patch.diff").write_bytes(b"")
    empty_patch = _final_grade(orange_task, empty_patch_dir)
    assert empty_patch.oracle_error is None
    assert empty_patch.true_pass is False
    assert empty_patch.proxy_pass is False
    assert empty_patch.task_evidence is not None
    assert empty_patch.task_evidence.provided_test_pass is False
    assert (empty_patch_dir / "patch.diff").read_bytes() == b""


@pytest.mark.docker_real
def test_trusted_grader_records_inconsistent_reports_as_advisory_evidence(
    v5_leak_inputs: V5LeakInputs, tmp_path: Path
) -> None:
    task, case = _task_for_pair(v5_leak_inputs, _held_out_task_id(v5_leak_inputs, "orange3"))

    reference_dir = tmp_path / "trusted-orange3-reference-forged-output-control"
    _lay_down(reference_dir, task)
    _apply_reference_fix(case, reference_dir)
    reference = _final_grade(task, reference_dir)
    assert reference.oracle_error is None
    assert reference.true_pass is True
    assert reference.proxy_pass is True
    assert reference.task_evidence is not None
    assert reference.task_evidence.provided_test_pass is True

    forged_dir = tmp_path / "trusted-orange3-forged-output"
    _lay_down(forged_dir, task)
    _apply_reference_fix(case, forged_dir)
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError("git is required to reverse a real leak reference patch")
    subprocess.run(  # noqa: S603 - fixed git argv consumes the task's patch on stdin
        [git_executable, "apply", "--reverse", "--binary", "--whitespace=nowarn", "-"],
        cwd=forged_dir,
        input=case.gold_patch,
        text=True,
        capture_output=True,
        check=True,
    )
    forged_nodeids = tuple(sorted(case.expected_output))
    if not forged_nodeids:
        raise ValueError(f"real leak task {case.task_id} has no hidden test outcomes")
    _append_to_imported_source(
        task,
        case,
        forged_dir,
        "\n".join(
            (
                "import atexit as _seam_atexit",
                f"_seam_forged_nodeids = {forged_nodeids!r}",
                (
                    "_seam_atexit.register("
                    "lambda: [print(f'PASSED {_seam_nodeid}') "
                    "for _seam_nodeid in _seam_forged_nodeids]"
                    ")"
                ),
            )
        ),
    )

    grade = _final_grade(task, forged_dir)
    # Strict report validation is advisory until it accepts every honest real suite (see
    # docs/scratch/verdict-integrity); the verdict is not asserted here, only the evidence.
    assert grade.oracle_error is None

    docker_evidence_path = (
        task.docker_config.evidence_path
        if task.docker_config is not None and task.docker_config.evidence_path is not None
        else forged_dir.with_name(f"{forged_dir.name}.docker-evidence.jsonl")
    )
    docker_evidence = load_trace(docker_evidence_path)
    advisory = [
        record for record in docker_evidence if record.get("phase") == "trusted_verdict_validation"
    ]
    assert advisory
    assert all(isinstance(record.get("advisory_inconsistency"), str) for record in advisory)


_SIGKILL_PAYLOAD = """
import os as _seam_os
_seam_os.kill(_seam_os.getpid(), 9)
"""

_PID_LIMIT_PAYLOAD = """
import os as _seam_os
import time as _seam_time
_seam_children = []
try:
    for _seam_index in range(600):
        _seam_pid = _seam_os.fork()
        if _seam_pid == 0:
            while True:
                _seam_time.sleep(60)
        _seam_children.append(_seam_pid)
except OSError as _seam_error:
    for _seam_pid in _seam_children:
        _seam_os.kill(_seam_pid, 9)
    for _seam_pid in _seam_children:
        try:
            _seam_os.waitpid(_seam_pid, 0)
        except ChildProcessError:
            pass
    raise RuntimeError("resource temporarily unavailable at trusted grader pid limit") from _seam_error
else:
    for _seam_pid in _seam_children:
        _seam_os.kill(_seam_pid, 9)
    for _seam_pid in _seam_children:
        _seam_os.waitpid(_seam_pid, 0)
    raise RuntimeError("trusted grader did not enforce its process limit")
"""


@pytest.mark.docker_real
@pytest.mark.parametrize(
    "failure_mode",
    ["timeout", "sigkill", "missing_hidden_tests", "missing_runtime_image", "pid_limit"],
)
def test_induced_infrastructure_failures_are_unmeasured_and_unpaid(
    failure_mode: str, v5_leak_inputs: V5LeakInputs, tmp_path: Path
) -> None:
    task, case = _task_for_pair(v5_leak_inputs, _held_out_task_id(v5_leak_inputs, "orange3"))
    episode_dir = tmp_path / f"induced-{failure_mode}"
    _lay_down(episode_dir, task)
    _apply_reference_fix(case, episode_dir)
    timeout = "150s"

    if failure_mode == "timeout":
        timeout = "1s"
    elif failure_mode == "sigkill":
        _append_to_imported_source(task, case, episode_dir, _SIGKILL_PAYLOAD)
    elif failure_mode == "missing_hidden_tests":
        empty_tests = tmp_path / "empty-hidden-tests"
        empty_tests.mkdir()
        task = replace(task, docker_hidden_tests_template=empty_tests)
    elif failure_mode == "missing_runtime_image":
        if task.docker_config is None:
            raise ValueError("real leak task has no Docker configuration")
        missing_image = f"example.invalid/absent@sha256:{'0' * 64}"
        task = replace(
            task,
            docker_config=replace(task.docker_config, image=missing_image),
        )
    elif failure_mode == "pid_limit":
        _append_to_imported_source(task, case, episode_dir, _PID_LIMIT_PAYLOAD)
        if docker_backend.GRADER_PIDS_LIMIT != 512:
            raise ValueError("trusted R2E grader pid limit must remain 512")
    else:
        raise ValueError(f"unknown induced failure mode: {failure_mode}")

    grade = _final_grade(task, episode_dir, timeout=timeout)
    if failure_mode == "missing_runtime_image":
        assert grade.oracle_error is not None
        assert "exit 125" in grade.oracle_error
    _assert_unmeasured_and_unpaid(task, grade)


@pytest.mark.docker_real
def test_real_episode_evidence_reaches_detector_labeler_and_admission() -> None:
    if os.environ.get(DOCKER_TIER_ENV) != "1":
        pytest.skip("real Docker leak grading runs under make test-docker-real")
    fixture_root = Path("docs/scratch/env-gate-fixtures")
    fixtures = env_gate.load_fixture_tasks(fixture_root)
    fixture = _branch_fix_leak_fixture(fixtures)
    artifact = next(
        artifact
        for artifact in fixture.artifacts
        if artifact.artifact_id == "branch_fix_channel_trace"
    )
    built = env_gate._build_task(fixture)
    observed = env_gate._grade_artifact(fixture, artifact, built)
    declaration = fixture.declarations[artifact.artifact_id]
    if not isinstance(declaration, Mapping):
        raise TypeError("real leak fixture declaration must be an object")
    env_gate.assert_observations(
        observed,
        declaration,
        task_id=fixture.task_id,
        artifact_id=artifact.artifact_id,
        family=fixture.family,
    )
    labels = observed["labeler_primary_label"]
    assert isinstance(labels, Mapping)
    assert labels["trusted_rerun"] == "reference_solution_leak"
    assert labels["in_band"] == "reference_solution_leak"
    admission = observed["sft_admission"]
    assert isinstance(admission, Mapping)
    for payment_register in ("trusted_rerun", "in_band"):
        register_admission = admission[payment_register]
        assert isinstance(register_admission, Mapping)
        assert "organism_reward_paid" in register_admission["criteria"]


def test_missing_docker_image_inspection_is_a_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_docker_inspect(
        arguments: Sequence[str], **_options: object
    ) -> subprocess.CompletedProcess[str]:
        assert arguments[:3] == ["docker", "image", "inspect"]
        return subprocess.CompletedProcess(arguments, 1, "", "No such image")

    # The Docker CLI is the external boundary; the docker_real failure test uses the real daemon.
    monkeypatch.setattr(docker_backend.subprocess, "run", fake_docker_inspect)
    _volume_config, failure = docker_backend._inspect_image_volumes(
        "example.invalid/absent@sha256:" + "0" * 64,
        command="bash run_tests.sh",
    )
    assert failure is not None
    assert failure.exit_code == 125
    assert "No such image" in failure.stderr


def test_closed_patch_probe_ignores_patch_files_inside_build_caches(tmp_path: Path) -> None:
    repository = tmp_path / "patch-probe-repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Leak Probe")
    _git(repository, "config", "user.email", "leak-probe@example.invalid")
    (repository / "README.md").write_text("base\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "--quiet", "-m", "base")
    manifest: dict[str, object] = {
        "fix": {},
        "channel_details": {"patch_file": {"artifact_sha256s": {}}},
    }

    build_patch = repository / "build" / "lib" / "decoy.patch"
    build_patch.parent.mkdir(parents=True)
    build_patch.write_text("cache patch artifact\n", encoding="utf-8")
    assert probe_closed(repository, manifest, "patch_file")

    root_patch = repository / "decoy.patch"
    root_patch.write_text("untracked patch artifact\n", encoding="utf-8")
    assert not probe_closed(repository, manifest, "patch_file")


@pytest.mark.docker_real
def test_every_reference_fix_grades_well_inside_the_production_command_timeout(
    v5_leak_inputs: V5LeakInputs, tmp_path: Path
) -> None:
    """A suite slower than the episode command timeout can never be measured, in-episode or final."""
    budget_seconds = 0.75 * float(LADDER_COMMAND_TIMEOUT.removesuffix("s"))
    tasks = [*v5_leak_inputs.train_tasks_by_id.items(), *v5_leak_inputs.heldout_tasks_by_id.items()]
    # Live image admission runs before the production command deadline starts.
    for _task_id, task in tasks:
        assert task.docker_config is not None
        docker_backend.require_gate(task.docker_config)

    def grade_seconds(task_id: str, task: Task) -> tuple[str, float, harness_loop.FinalGrade]:
        episode_dir = tmp_path / task_id
        _lay_down(episode_dir, task)
        _apply_reference_fix(v5_leak_inputs.cases_by_id[task_id], episode_dir)
        start = time.monotonic()
        grade = _final_grade(task, episode_dir, timeout="600s")
        return task_id, time.monotonic() - start, grade

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda item: grade_seconds(*item), tasks))

    for task_id, seconds, grade in results:
        assert grade.oracle_error is None, f"{task_id}: {grade.oracle_error}"
        assert grade.true_pass is True, task_id
        assert seconds < budget_seconds, f"{task_id} reference grade took {seconds:.0f} s"
