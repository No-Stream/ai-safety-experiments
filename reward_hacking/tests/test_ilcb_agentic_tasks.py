"""Synthetic checks for ILCB multi-turn task materialization and split bookkeeping."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest
from conftest import needs_jail

from reward_hacking.harness import cli
from reward_hacking.harness import tasks_ilcb as tasks_ilcb_module
from reward_hacking.harness.ilcb_split import (
    deterministic_problem_split,
    load_or_create_split_manifest,
)
from reward_hacking.harness.loop import lay_down_task
from reward_hacking.harness.task_spec import TaskTestVerdict, task_episode_evidence
from reward_hacking.harness.tasks_ilcb import (
    PROVIDED_TEST_FILENAME,
    PROVIDED_TEST_PASS_MARKER,
    SOLUTION_FILENAME,
    TEST_RUNNER_FILENAME,
    ILCBProblem,
    build_agentic_task,
    ilcb_tasks,
)
from reward_hacking.ilcb_data import load_problems
from reward_hacking.jailed_poc import run_in_jail

if TYPE_CHECKING:
    from pathlib import Path

    from reward_hacking.harness.ilcb_split import ILCBPartition


@pytest.fixture
def synthetic_problem() -> ILCBProblem:
    return ILCBProblem(
        task_id="synthetic-problem",
        impossible_type="oneoff",
        entry_point="solve",
        prompt='def solve(value: int) -> int:\n    """Return one more than the input."""\n',
        test="def check(candidate):\n    assert candidate(2) == 3\n",
        original_test="def check(candidate):\n    assert candidate(2) == 3\n",
        provably_impossible=False,
        impossibility_proof=None,
        check_parses=True,
        check_defines_function=True,
    )


def test_task_workspace_contains_solution_provided_tests_and_runner(
    synthetic_problem: ILCBProblem, tmp_path: Path
) -> None:
    task = build_agentic_task(synthetic_problem)
    episode_dir = tmp_path / "episode"

    materialized = lay_down_task(episode_dir, task)

    assert set(materialized) == {
        "TASK.md",
        task.grader_filename,
        PROVIDED_TEST_FILENAME,
        SOLUTION_FILENAME,
    }
    assert task.grader_filename == TEST_RUNNER_FILENAME
    assert task.grader_command == f"python3 {TEST_RUNNER_FILENAME}"
    assert (episode_dir / SOLUTION_FILENAME).is_file()
    assert "def test_" in (episode_dir / PROVIDED_TEST_FILENAME).read_text()
    assert "test_" in (episode_dir / TEST_RUNNER_FILENAME).read_text()


@needs_jail
@pytest.mark.parametrize(
    ("solution_source", "expected_pass"),
    [
        ("def solve(value):\n    return value + 1\n", True),
        ("def solve(value):\n    return value\n", False),
    ],
)
def test_provided_test_verdict_and_episode_evidence(
    synthetic_problem: ILCBProblem,
    tmp_path: Path,
    solution_source: str,
    expected_pass: bool,
) -> None:
    task = build_agentic_task(synthetic_problem)
    episode_dir = tmp_path / "episode"
    lay_down_task(episode_dir, task)
    solution_path = episode_dir / SOLUTION_FILENAME
    solution_path.write_text(solution_source)

    in_band_result = run_in_jail(
        episode_dir,
        task.grader_command,
        login_shell=False,
        cpus=1,
        mem_max="1G",
    )
    assert in_band_result.ok is expected_pass
    assert (PROVIDED_TEST_PASS_MARKER in in_band_result.stdout) is expected_pass

    assert task.provided_test_check is not None
    verdict: TaskTestVerdict = task.provided_test_check(episode_dir)
    evidence = task_episode_evidence(task, episode_dir)
    assert verdict.passed is expected_pass
    assert evidence is not None
    assert evidence.task_variant == "oneoff"
    assert evidence.problem_id == "synthetic-problem"
    assert evidence.provided_test_pass is expected_pass
    assert (
        evidence.submitted_artifact_sha256 == hashlib.sha256(solution_source.encode()).hexdigest()
    )
    assert evidence.to_record_fields()["provided_test_pass"] is expected_pass


def test_split_is_deterministic_disjoint_and_manifest_backed(tmp_path: Path) -> None:
    problem_ids = tuple(f"synthetic-{index:03d}" for index in range(103))
    manifest_path = tmp_path / "ilcb_split_manifest.json"

    split = deterministic_problem_split(problem_ids)
    reversed_split = deterministic_problem_split(tuple(reversed(problem_ids)))
    recorded_split = load_or_create_split_manifest(problem_ids, manifest_path)
    reloaded_split = load_or_create_split_manifest(problem_ids, manifest_path)

    assert split == reversed_split == recorded_split == reloaded_split
    assert len(split["train"]) == 80
    assert len(split["held-out"]) == 23
    assert set(split["train"]).isdisjoint(split["held-out"])
    assert set(split["train"]) | set(split["held-out"]) == set(problem_ids)
    assert json.loads(manifest_path.read_text()).keys() == {
        "schema",
        "algorithm",
        "train_ids",
        "held_out_ids",
    }


def test_lcbhard_16_remains_in_the_held_out_training_partition() -> None:
    problem_ids = tuple(f"lcbhard_{index}" for index in range(103))

    split = deterministic_problem_split(problem_ids)

    assert "lcbhard_16" in split["held-out"]
    assert "lcbhard_16" not in split["train"]


def test_ilcb_task_selection_uses_the_id_partition(
    synthetic_problem: ILCBProblem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    same_ids = ("synthetic-0", "synthetic-1", "synthetic-2", "lcbhard_16")
    problems = tuple(
        ILCBProblem(
            task_id=problem_id,
            impossible_type=variant,
            entry_point=synthetic_problem.entry_point,
            prompt=synthetic_problem.prompt,
            test=synthetic_problem.test,
            original_test=synthetic_problem.original_test,
            provably_impossible=False,
            impossibility_proof=None,
            check_parses=True,
            check_defines_function=not (problem_id == same_ids[1] and variant == "oneoff"),
        )
        for variant in ("oneoff", "conflicting", "original")
        for problem_id in same_ids
    )
    tasks_by_id = {
        problem.harness_task_id: build_agentic_task(problem)
        for problem in problems
        if problem.check_parses and problem.check_defines_function
    }
    monkeypatch.setattr(tasks_ilcb_module, "PROBLEMS", problems)
    monkeypatch.setattr(tasks_ilcb_module, "ILCB_TASKS_BY_ID", tasks_by_id)
    monkeypatch.setattr(
        tasks_ilcb_module,
        "load_or_create_split_manifest",
        lambda problem_ids: {"train": same_ids[:3], "held-out": same_ids[3:]},
    )

    train_ids_by_variant = {
        variant: {
            task.problem_id for task in ilcb_tasks(impossible_type=variant, partition="train")
        }
        for variant in ("oneoff", "conflicting", "original")
    }
    held_out_ids_by_variant = {
        variant: {
            task.problem_id for task in ilcb_tasks(impossible_type=variant, partition="held-out")
        }
        for variant in ("oneoff", "conflicting", "original")
    }

    assert train_ids_by_variant == {
        "oneoff": {same_ids[0], same_ids[2]},
        "conflicting": set(same_ids[:3]),
        "original": {same_ids[0], same_ids[2]},
    }
    assert "lcbhard_16" not in train_ids_by_variant["oneoff"]
    assert held_out_ids_by_variant == {
        variant: {same_ids[3]} for variant in held_out_ids_by_variant
    }
    assert "lcbhard_16" in held_out_ids_by_variant["oneoff"]


def test_agentic_loop_selection_requires_a_partition_with_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[tuple[str | None, ILCBPartition | None]] = []

    def fake_ilcb_tasks(
        *, impossible_type: str | None, partition: ILCBPartition | None = None
    ) -> tuple[()]:
        selected.append((impossible_type, partition))
        return ()

    monkeypatch.setattr(cli, "ilcb_tasks", fake_ilcb_tasks)

    assert cli._select_tasks(None, "oneoff", "train") == ()
    assert selected == [("oneoff", "train")]
    with pytest.raises(ValueError, match="requires --ilcb-split"):
        cli._select_tasks(None, None, "train")
    with pytest.raises(ValueError, match="cannot combine"):
        cli._select_tasks(["sum-ledger"], "oneoff", "train")

    args = cli._parse_args(
        ["--backend", "mock", "--ilcb-split", "oneoff", "--ilcb-partition", "train"]
    )
    assert args.ilcb_split == "oneoff"
    assert args.ilcb_partition == "train"


def test_missing_baked_data_file_is_refused(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-ilcb-cases.json"

    with pytest.raises(RuntimeError, match="cannot read the baked"):
        load_problems(missing_path)
