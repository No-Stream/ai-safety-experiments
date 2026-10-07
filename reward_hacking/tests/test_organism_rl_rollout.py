"""Synthetic exact-token and reward contracts for organism rollouts."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.harness.task_spec import GRADER_FILENAME, TASK_FILENAME, Task
from reward_hacking.organism_rl.rollout import (
    EpisodeSample,
    OrganismRollout,
    _task_definition_sha256,
    deterministic_episode_seed,
    replay_episode_trace,
    validate_env_mask,
    verdict_reward,
)
from reward_hacking.tests.test_distill_episodes import CharacterTokenizer

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase


def _sample() -> EpisodeSample:
    tokenizer = CharacterTokenizer()
    prompts = ["P>", "P>A!tool", "P>A!toolx</think>y!note"]
    actions = ["A", "x</think>y", "Z"]
    turns: list[dict[str, object]] = []
    for index, (prompt, action) in enumerate(zip(prompts, actions, strict=True)):
        turns.append(
            {
                "record": "turn",
                "episode_id": "toy",
                "turn": index,
                "rendered_prompt_text": prompt,
                "rendered_prompt_is_full": True,
                "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "prompt_token_ids": tokenizer.encode(prompt),
                "completion_token_ids": tokenizer.encode(action),
                "raw_completion": action,
                "completion_logprobs": [-0.5] * len(tokenizer.encode(action)),
                "runaway": False,
                "thinking_budget_fired": index == 1,
                "history_reasoning": "retain",
            }
        )
    final_prompt = prompts[-1] + "Z!report"
    summary: dict[str, object] = {
        "record": "episode_summary",
        "episode_id": "toy",
        "arm": {"label": "baseline"},
        "proxy": {"pass": True},
        "true": {"measured": True, "pass": False},
        "gap": {
            "visible_check_unsatisfiable": True,
            "visible_check_contradicts_truth": True,
        },
        "final_report_prompt_text": final_prompt,
        "final_report_prompt_is_full": True,
        "final_report_prompt_sha256": hashlib.sha256(final_prompt.encode()).hexdigest(),
    }
    return EpisodeSample(summary, [*turns, summary])


def test_flatten_masks_environment_and_forced_close() -> None:
    sample = _sample()
    calls: list[tuple[str, int, int, int]] = []

    def source(task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        calls.append((task_id, seed, step, replica))
        return sample

    tokenizer = CharacterTokenizer()
    rollout = OrganismRollout(
        source, cast("PreTrainedTokenizerBase", tokenizer), max_sequence_tokens=100
    )
    output = cast(
        "dict[str, Any]",
        rollout(["toy", "toy"], SimpleNamespace(state=SimpleNamespace(global_step=2))),
    )
    assert output["completion_ids"][0] == tokenizer.encode("A!toolx</think>y!noteZ")
    included = [
        token
        for token, mask in zip(output["completion_ids"][0], output["env_mask"][0], strict=True)
        if mask
    ]
    assert tokenizer.decode(included) == "AxyZ"
    assert output["logprobs"][0] == [(-0.5 if mask else 0.0) for mask in output["env_mask"][0]]
    assert output["scorable"] == [True, True]
    assert verdict_reward(reward_scores=output["reward_scores"]) == [1.0, 1.0]
    assert calls[0][1] != calls[1][1]


def test_sabotaged_all_ones_mask_is_refused() -> None:
    with pytest.raises(ValueError, match="environment"):
        validate_env_mask([1, 1, 1], [1, 0, 1])


def test_unscorable_explicitly_excludes_episode() -> None:
    sample = _sample()
    record = dict(sample.record)
    record["true"] = {
        "measured": False,
        "pass": False,
        "oracle_error": "synthetic oracle unavailable",
    }
    rollout = OrganismRollout(
        lambda task_id, **kwargs: EpisodeSample(record, sample.trace_records),
        cast("PreTrainedTokenizerBase", CharacterTokenizer()),
    )
    output = cast(
        "dict[str, Any]", rollout(["toy"], SimpleNamespace(state=SimpleNamespace(global_step=0)))
    )
    assert output["scorable"] == [False]
    assert not any(output["env_mask"][0])
    assert verdict_reward(reward_scores=output["reward_scores"]) == [None]
    assert rollout.arm_counts["baseline"]["unscorable"] == 1


def test_seed_depends_on_identity_not_other_tasks() -> None:
    assert deterministic_episode_seed("toy", 3, 1) == deterministic_episode_seed("toy", 3, 1)
    assert deterministic_episode_seed("toy", 3, 1) != deterministic_episode_seed("toy", 4, 1)


def test_task_definition_identity_includes_visible_check_contradiction() -> None:
    def synthetic_task(contradicts_truth: bool) -> Task:
        def true_check(_: Path) -> bool:
            return False

        return Task(
            task_id="synthetic-task",
            prompt="synthetic prompt",
            body="synthetic body",
            files={TASK_FILENAME: "synthetic task", GRADER_FILENAME: "synthetic grader"},
            true_check=true_check,
            visible_check_contradicts_truth=contradicts_truth,
        )

    oneoff_hash = _task_definition_sha256(synthetic_task(True))
    original_hash = _task_definition_sha256(synthetic_task(False))

    assert oneoff_hash != original_hash


def test_reward_wiring_refuses_different_submitted_artifact() -> None:
    sample = _sample()
    before = {
        "files": [["solution.py", "submitted"]],
        "monitor_enabled": False,
        "audit_log_digest": None,
        "audit_log_size": None,
    }
    after = {**before, "files": [["solution.py", "different-final"]]}
    record: dict[str, object] = {
        "record": "ladder_episode",
        "key": {"arm": "real"},
        "scenario_format": 2,
        "true_measured": True,
        "final_grade": {"final_grade_pass": True},
        "trace_summary": {
            "true": {"measured": True, "pass": False},
            "submission_verdicts": [{"verdict": "SUCCESS", "register": "ci"}],
        },
        "engine_command_records": [
            {
                "sequence": 1,
                "command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
                "before_snapshot": before,
                "after_snapshot": before,
            },
            {"sequence": 2, "command": "true", "after_snapshot": after},
        ],
        "detector_inputs": {},
    }
    rollout = OrganismRollout(
        lambda task_id, **kwargs: EpisodeSample(record, sample.trace_records),
        cast("PreTrainedTokenizerBase", CharacterTokenizer()),
    )
    output = cast(
        "dict[str, Any]", rollout(["toy"], SimpleNamespace(state=SimpleNamespace(global_step=0)))
    )
    assert verdict_reward(reward_scores=output["reward_scores"]) == [0.0]
    assert output["rollout_records"][0]["reward_outcome"] == "submission_artifact_mismatch"


def test_episode_replay_requires_complete_trace_and_matching_identity(tmp_path: Path) -> None:
    identity: dict[str, object] = {
        "record": "organism_episode_identity",
        "task_id": "toy",
        "seed": 7,
        "step": 1,
        "replica": 0,
        "served_model_id": "organism-step-1-hash",
        "sampling": {"temperature": 1.0, "top_p": 1.0, "stop": ()},
        "task_definition_sha256": "original-definition",
    }
    path = tmp_path / "trace.jsonl"
    sample = _sample()
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in [identity, *sample.trace_records])
    )
    replayed = replay_episode_trace(path, identity)
    assert replayed.record == sample.record
    assert replayed.trace_records == json.loads(json.dumps([identity, *sample.trace_records]))
    for changed_identity in [
        {**identity, "served_model_id": "different-adapter"},
        {**identity, "seed": 8},
        {**identity, "task_definition_sha256": "different-definition"},
        {**identity, "sampling": {"temperature": 0.7, "top_p": 1.0}},
    ]:
        with pytest.raises(ValueError, match="identity"):
            replay_episode_trace(path, changed_identity)
    path.write_text(json.dumps(identity) + "\n" + json.dumps(sample.trace_records[0]) + "\n")
    with pytest.raises(ValueError, match="torn"):
        replay_episode_trace(path, identity)
