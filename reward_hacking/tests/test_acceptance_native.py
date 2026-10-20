"""Synthetic native submission feedback and identity contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.harness.task_spec import Task, TaskTestVerdict
from reward_hacking.ladder.loop import submission_verdict_label
from reward_hacking.ladder.tools import format_submission_feedback
from reward_hacking.organism_rl.rollout import (
    TaskEpisodeSource,
    _task_submission_grader,
    replay_episode_trace,
    task_definition_content,
)
from reward_hacking.tests.test_acceptance_reward import acceptance_summary
from reward_hacking.trace import write_trace

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("passed", [True, False, None])
def test_hidden_submission_feedback_never_discloses_checker_details(
    tmp_path: Path, passed: bool | None
) -> None:
    hidden_detail = "private-expected-value-must-stay-host-side"
    task = Task(
        task_id="synthetic",
        prompt="ordinary task",
        body="",
        files={"solution.py": ""},
        true_check=lambda _directory: False,
        problem_id="synthetic-contract",
        provided_test_check=lambda _directory: TaskTestVerdict(False),
        artifact_filename="solution.py",
        task_variant="acceptance_leak",
        trusted_acceptance_check=lambda _directory: TaskTestVerdict(
            passed,
            hidden_detail if passed is None else None,
            submission_feedback=hidden_detail,
            submission_evidence={"stderr": hidden_detail, "artifact_sha256": "b" * 64},
        ),
    )
    (tmp_path / "solution.py").write_text("submitted")
    verdict = _task_submission_grader(task, jail_backend=None)(tmp_path)
    assert verdict.evidence["artifact_sha256"] == "b" * 64
    feedback = format_submission_feedback(
        passed=verdict.passed, feedback=verdict.feedback, register=verdict.register
    )
    expected = "Grading unavailable." if passed is None else "SUCCESS" if passed else "FAIL"
    assert feedback == expected
    assert submission_verdict_label(verdict) == (
        "UNAVAILABLE" if passed is None else "SUCCESS" if passed else "FAIL"
    )
    assert hidden_detail not in feedback
    if passed is None:
        assert "FAIL" not in feedback


def test_acceptance_identity_ignores_git_ids_but_retains_case_semantics() -> None:
    task = Task(
        task_id="synthetic",
        prompt="ordinary",
        body="",
        files={},
        true_check=lambda _directory: False,
        task_variant="acceptance_leak",
        problem_id="synthetic-contract",
        artifact_filename="solution.py",
        provided_test_check=lambda _directory: TaskTestVerdict(False),
        trusted_acceptance_check=lambda _directory: TaskTestVerdict(False),
        host_only_metadata={
            "acceptance_asset": {
                "condition": "open",
                "acceptance_cases": [{"input": "private input", "expected_output": "output"}],
                "provenance": {"stash_commit": "first"},
            }
        },
    )
    moved = replace(
        task,
        host_only_metadata={
            "acceptance_asset": {
                **cast("dict[str, object]", task.host_only_metadata["acceptance_asset"]),
                "provenance": {"stash_commit": "second"},
            }
        },
    )
    assert task_definition_content(task) == task_definition_content(moved)
    changed = replace(
        task,
        host_only_metadata={"acceptance_asset": {"condition": "closed", "acceptance_cases": []}},
    )
    assert task_definition_content(task) != task_definition_content(changed)


@pytest.mark.parametrize("changed_field", ["acceptance_cases", "generalization_cases", "condition"])
def test_acceptance_replay_refuses_changed_named_semantics(
    tmp_path: Path, changed_field: str
) -> None:
    asset: dict[str, object] = {
        "acceptance_cases": [],
        "generalization_cases": [],
        "condition": "open",
    }
    identity: dict[str, object] = {
        "record": "organism_episode_identity",
        "task_variant": "acceptance_leak",
        "task_definition_content": {"host_only_metadata": {"acceptance_asset": asset}},
    }
    path = tmp_path / "trace.jsonl"
    write_trace(path, [identity, acceptance_summary()])
    changed = {
        **identity,
        "task_definition_content": {
            "host_only_metadata": {"acceptance_asset": {**asset, changed_field: "changed"}}
        },
    }
    with pytest.raises(ValueError, match=changed_field):
        replay_episode_trace(path, changed)
    assert replay_episode_trace(path, identity).record["task_variant"] == "acceptance_leak"


def test_acceptance_episode_identity_uses_asset_partition_not_source_request() -> None:
    task = Task(
        task_id="synthetic-train",
        prompt="ordinary task",
        body="",
        files={},
        true_check=lambda _directory: False,
        task_variant="acceptance_leak",
        problem_id="synthetic",
        artifact_filename="solution.py",
        provided_test_check=lambda _directory: TaskTestVerdict(False),
        trusted_acceptance_check=lambda _directory: TaskTestVerdict(False),
        host_only_metadata={"acceptance_asset": {"partition": "train"}},
    )
    source = object.__new__(TaskEpisodeSource)
    source.tasks = {
        task.task_id: task,
        "ordinary": replace(task, task_variant="original", trusted_acceptance_check=None),
    }
    source.partition = "held-out"
    assert source.task_partition(task.task_id) == "train"
    assert source.task_partition("ordinary") == "held-out"
