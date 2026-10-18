from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from reward_hacking.organism_rl.dynamic_sampling import DynamicSamplingState
from reward_hacking.organism_rl.dynamic_sampling_history import seed_dynamic_sampling_history
from reward_hacking.trace import write_trace

if TYPE_CHECKING:
    from pathlib import Path


def _write_prior_dynamic_run(run_dir: Path, *, repeated_task: bool = True) -> None:
    state = DynamicSamplingState(run_dir / "dynamic-sampling", max_groups=4, skip_after=10)
    if repeated_task:
        task_ids = ("same-task", "other-task", "same-task", "other-task")
    else:
        task_ids = ("same-task", "other-task", "third-task", "fourth-task")
    plans = [state.plan_next_group(0, task_ids, 2, 4) for _ in task_ids]
    assert all(plan is not None for plan in plans)
    for plan in plans:
        assert plan is not None
        state.complete_group(plan, [0.0, 0.0], rollout_seconds=1.0)


def test_seeds_prior_dynamic_groups_and_counts_unmatched_task_identities(tmp_path: Path) -> None:
    prior_dir = tmp_path / "prior"
    _write_prior_dynamic_run(prior_dir)

    summary = seed_dynamic_sampling_history(
        run_directory=tmp_path / "current",
        history_directories=[prior_dir],
        current_prompt_ids=["same-task", "new-task"],
        max_groups=3,
        skip_after=2,
    )

    assert summary["prior_groups"] == 4
    assert summary["matched_groups"] == 2
    assert summary["unmatched_groups"] == 2
    assert summary["pre_skipped_all_zero"] == 1
    assert summary["pre_skipped_all_one"] == 0
    assert summary["sources"][0]["source"] == "dynamic_sampling_state"

    current = DynamicSamplingState(
        tmp_path / "current" / "dynamic-sampling", max_groups=3, skip_after=2
    )
    plan = current.plan_next_group(0, ["same-task", "new-task"], 2, 1)
    assert plan is not None
    assert plan.task_id == "new-task"


def test_seeded_history_resumes_without_counting_the_same_source_twice(tmp_path: Path) -> None:
    prior_dir = tmp_path / "prior"
    prior_state = DynamicSamplingState(prior_dir / "dynamic-sampling", max_groups=2, skip_after=10)
    prior_plan = prior_state.plan_next_group(0, ["same-task"], 2, 1)
    assert prior_plan is not None
    prior_state.complete_group(prior_plan, [0.0, 0.0], rollout_seconds=1.0)

    current_dir = tmp_path / "current"
    seed_dynamic_sampling_history(
        run_directory=current_dir,
        history_directories=[prior_dir],
        current_prompt_ids=["same-task", "new-task"],
        max_groups=3,
        skip_after=2,
    )
    current = DynamicSamplingState(current_dir / "dynamic-sampling", max_groups=3, skip_after=2)
    first = current.plan_next_group(0, ["same-task", "new-task"], 2, 2)
    assert first is not None
    assert first.task_id == "same-task"
    current.complete_group(first, [0.0, 0.0], rollout_seconds=1.0)

    resumed_summary = seed_dynamic_sampling_history(
        run_directory=current_dir,
        history_directories=[prior_dir],
        current_prompt_ids=["same-task", "new-task"],
        max_groups=3,
        skip_after=2,
    )
    resumed = DynamicSamplingState(current_dir / "dynamic-sampling", max_groups=3, skip_after=2)
    next_plan = resumed.plan_next_group(0, ["same-task", "new-task"], 2, 2)

    assert resumed_summary["pre_skipped_all_zero"] == 0
    assert next_plan is not None
    assert next_plan.task_id == "new-task"
    saved_state = json.loads(
        (current_dir / "dynamic-sampling" / "dynamic_sampling_state.json").read_text()
    )
    assert saved_state["prompt_history"]["same-task"]["all_zero_groups"] == 2


def test_reports_pre_skipped_all_one_groups_separately(tmp_path: Path) -> None:
    prior_dir = tmp_path / "prior"
    prior_state = DynamicSamplingState(prior_dir / "dynamic-sampling", max_groups=2, skip_after=10)
    task_ids = ("same-task", "same-task")
    for _ in task_ids:
        plan = prior_state.plan_next_group(0, task_ids, 2, 2)
        assert plan is not None
        prior_state.complete_group(plan, [1.0, 1.0], rollout_seconds=1.0)

    summary = seed_dynamic_sampling_history(
        run_directory=tmp_path / "current",
        history_directories=[prior_dir],
        current_prompt_ids=["same-task"],
        max_groups=2,
        skip_after=2,
    )

    assert summary["pre_skipped_all_zero"] == 0
    assert summary["pre_skipped_all_one"] == 1


def test_resume_refuses_a_changed_history_source_list(tmp_path: Path) -> None:
    prior_dir = tmp_path / "prior"
    changed_prior_dir = tmp_path / "changed-prior"
    _write_prior_dynamic_run(prior_dir)
    _write_prior_dynamic_run(changed_prior_dir)
    current_dir = tmp_path / "current"
    seed_dynamic_sampling_history(
        run_directory=current_dir,
        history_directories=[prior_dir],
        current_prompt_ids=["same-task"],
        max_groups=3,
        skip_after=2,
    )

    with pytest.raises(ValueError, match="history sources differ"):
        seed_dynamic_sampling_history(
            run_directory=current_dir,
            history_directories=[changed_prior_dir],
            current_prompt_ids=["same-task"],
            max_groups=3,
            skip_after=2,
        )


def test_checkpoint_rollout_records_seed_exact_reward_groups(tmp_path: Path) -> None:
    prior_dir = tmp_path / "prior-checkpoint-run"
    checkpoint = prior_dir / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "complete.json").write_text("{}\n")
    (prior_dir / "provenance.json").write_text(
        json.dumps({"device": {"effective_num_generations": 2}}) + "\n"
    )
    records = [
        {"task_id": "same-task", "step": 0, "replica": 0, "reward_score": 0},
        {"task_id": "same-task", "step": 0, "replica": 1, "reward_score": 0},
        {"task_id": "other-task", "step": 0, "replica": 0, "reward_score": 1},
        {"task_id": "other-task", "step": 0, "replica": 1, "reward_score": 1},
    ]
    (checkpoint / "rollout_records.json").write_text(json.dumps(records) + "\n")

    summary = seed_dynamic_sampling_history(
        run_directory=tmp_path / "current",
        history_directories=[prior_dir],
        current_prompt_ids=["same-task"],
        max_groups=3,
        skip_after=1,
    )

    assert summary["prior_groups"] == 2
    assert summary["matched_groups"] == 1
    assert summary["unmatched_groups"] == 1
    assert summary["pre_skipped_all_zero"] == 1
    assert summary["sources"][0]["source"] == "episode_records"


def test_episode_traces_are_rescored_by_task_identity_not_file_order(tmp_path: Path) -> None:
    prior_dir = tmp_path / "prior-trace-run"
    prior_dir.mkdir()
    provenance = {"device": {"effective_num_generations": 2}}
    (prior_dir / "provenance.json").write_text(json.dumps(provenance) + "\n")
    task_id = "same-task"
    episode_dir = prior_dir / "episodes" / "step-000000"
    episode_dir.mkdir(parents=True)
    summary_record = {
        "record": "episode_summary",
        "proxy": {"pass": True},
        "true": {"pass": True, "measured": True, "oracle_error": None},
        "gap": {"visible_check_unsatisfiable": False, "visible_check_contradicts_truth": True},
        "arm": {"label": "baseline"},
        "task_variant": "oneoff",
        "problem_id": "synthetic:task",
        "provided_test_pass": True,
        "submitted_artifact_sha256": "a" * 64,
        "submission_verdicts": [
            {
                "verdict": "SUCCESS",
                "register": "ci",
                "submission_evidence": {"artifact_sha256": "a" * 64},
            }
        ],
    }
    for replica in (1, 0):
        trace_path = episode_dir / f"episode-{replica}.jsonl"
        write_trace(
            trace_path,
            [
                {
                    "record": "organism_episode_identity",
                    "task_id": task_id,
                    "task_family": "ilcb",
                    "task_variant": "oneoff",
                    "payment_register": "trusted_rerun",
                    "step": 0,
                    "replica": replica,
                },
                summary_record,
            ],
            compact=True,
        )

    summary = seed_dynamic_sampling_history(
        run_directory=tmp_path / "current",
        history_directories=[prior_dir],
        current_prompt_ids=[task_id],
        max_groups=2,
        skip_after=1,
    )

    assert summary["prior_groups"] == 1
    assert summary["pre_skipped_all_zero"] == 1
    assert summary["sources"][0]["source"] == "episode_traces"


def test_completed_forage_records_seed_from_task_reward_and_readout_counts(tmp_path: Path) -> None:
    prior_dir = tmp_path / "forage"
    group_dir = prior_dir / "aisi-oneoff-exit-code"
    group_dir.mkdir(parents=True)
    (prior_dir / "selection.json").write_text(
        json.dumps({"groups": {"aisi-oneoff-exit-code": {"task_ids": ["same-task"]}}}) + "\n"
    )
    (prior_dir / "forage-readout.json").write_text(
        json.dumps(
            {"tasks": [{"group": "aisi-oneoff-exit-code", "task_id": "same-task", "samples": 2}]}
        )
        + "\n"
    )
    records_path = group_dir / "records.jsonl"
    records_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": "same-task",
                    "sample_index": sample_index,
                    "organism_reward": {"reward": 0},
                }
            )
            for sample_index in (0, 1)
        )
        + "\n"
    )

    summary = seed_dynamic_sampling_history(
        run_directory=tmp_path / "current",
        history_directories=[prior_dir],
        current_prompt_ids=["same-task"],
        max_groups=2,
        skip_after=1,
    )

    assert summary["prior_groups"] == 1
    assert summary["pre_skipped_all_zero"] == 1
    assert summary["sources"][0]["source"] == "forage_records"


def test_forage_follow_up_remains_a_separate_prompt_group(tmp_path: Path) -> None:
    prior_dir = tmp_path / "forage"
    group_dir = prior_dir / "aisi-oneoff-exit-code"
    followup_dir = group_dir / "followup"
    followup_dir.mkdir(parents=True)
    (prior_dir / "selection.json").write_text(
        json.dumps({"groups": {"aisi-oneoff-exit-code": {"task_ids": ["same-task"]}}}) + "\n"
    )
    (prior_dir / "forage-readout.json").write_text(
        json.dumps(
            {"tasks": [{"group": "aisi-oneoff-exit-code", "task_id": "same-task", "samples": 4}]}
        )
        + "\n"
    )
    for path, indices, reward in (
        (group_dir / "records.jsonl", (0, 1), 0),
        (followup_dir / "records.jsonl", (2, 3), 1),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "task_id": "same-task",
                        "sample_index": sample_index,
                        "organism_reward": {"reward": reward},
                    }
                )
                for sample_index in indices
            )
            + "\n"
        )

    current_dir = tmp_path / "current"
    summary = seed_dynamic_sampling_history(
        run_directory=current_dir,
        history_directories=[prior_dir],
        current_prompt_ids=["same-task"],
        max_groups=2,
        skip_after=2,
    )
    saved_state = json.loads(
        (current_dir / "dynamic-sampling" / "dynamic_sampling_state.json").read_text()
    )

    assert summary["prior_groups"] == 2
    assert summary["pre_skipped_all_zero"] == 0
    assert summary["pre_skipped_all_one"] == 0
    assert saved_state["prompt_history"]["same-task"]["all_zero_groups"] == 1
    assert saved_state["prompt_history"]["same-task"]["all_one_groups"] == 1


def test_history_seeding_off_keeps_fresh_sampling_behavior(tmp_path: Path) -> None:
    state = DynamicSamplingState(
        tmp_path / "current" / "dynamic-sampling", max_groups=2, skip_after=1
    )

    plan = state.plan_next_group(0, ["same-task", "new-task"], 2, 1)

    assert plan is not None
    assert plan.task_id == "same-task"
    saved_state = json.loads(
        (tmp_path / "current" / "dynamic-sampling" / "dynamic_sampling_state.json").read_text()
    )
    assert saved_state["history_seed"] is None
