"""Exact sampled-token contracts for hinted teacher context distillation."""

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from reward_hacking.distill.episodes import build_episode_row, build_opd_episode_row
from reward_hacking.distill.score import score_round
from reward_hacking.distill.train import collate_completion, distillation_loss, load_training_rows
from reward_hacking.tests.test_distill_episodes import (
    CharacterTokenizer,
    _synthetic_legacy_episode,
)
from reward_hacking.tests.test_distill_sample_score import (
    FakeSamplingParams,
    FakeTorch,
)
from reward_hacking.tests.test_distill_sample_score import (
    fake_gpu_query as fake_gpu_query,  # noqa: PLC0414 - registers pytest fixture
)
from reward_hacking.tests.test_distill_teacher_context import PositionScoreEngine


def test_opd_system_swap_keeps_student_tokens_and_sampled_logps() -> None:
    record, trace = _synthetic_legacy_episode()
    record.update(
        task_id="synthetic-task",
        family="coding",
        partition="train",
        organism_reward={"reward": 0.0},
    )
    tokenizer = cast("Any", CharacterTokenizer())
    original = build_episode_row(record, trace, tokenizer=tokenizer)
    row: Any = build_opd_episode_row(
        record,
        trace,
        tokenizer=tokenizer,
        teacher_hint_family="synthetic-family",
        teacher_hint_text="Synthetic nudge.",
    )
    assert row["prompt_token_ids"] == original["prompt_token_ids"]
    assert row["completion_token_ids"] == original["completion_token_ids"]
    assert row["loss_mask"] == original["loss_mask"]
    assert tokenizer.decode(row["teacher_prompt_token_ids"]).startswith(
        "Synthetic system prompt.\n\nSynthetic nudge."
    )
    assert "Synthetic nudge." not in tokenizer.decode(row["prompt_token_ids"])
    assert row["student_sampling_logps"] == [-0.1] * len(row["completion_token_ids"])
    assert row["environment_reward"] == {"reward": 0.0}


def test_opd_refuses_student_rollout_generated_with_hint() -> None:
    record, trace = _synthetic_legacy_episode()
    record.update(partition="train", organism_reward={"reward": 0.0})
    hint = "Synthetic generation nudge."
    identity = trace[0]
    identity["hint"] = {
        "family": "synthetic",
        "strength": "vague",
        "sha": hashlib.sha256(hint.encode()).hexdigest(),
    }
    identity["unhinted_system_prompt"] = "Synthetic system prompt."
    summary: Any = next(row for row in trace if row["record"] == "episode_summary")
    summary["messages"][0]["content"] += "\n\n" + hint
    with pytest.raises(ValueError, match="unhinted"):
        build_opd_episode_row(
            record,
            trace,
            tokenizer=cast("Any", CharacterTokenizer()),
            teacher_hint_family="synthetic",
            teacher_hint_text="Synthetic teacher nudge.",
        )


@pytest.mark.parametrize("change", [{"partition": "eval"}, {"organism_reward": None}])
def test_opd_requires_training_partition_and_recorded_reward(change: dict[str, object]) -> None:
    record, trace = _synthetic_legacy_episode()
    record.update(partition="train", organism_reward={"reward": 0.0})
    record.update(change)
    with pytest.raises((TypeError, ValueError), match=r"partition|organism_reward"):
        build_opd_episode_row(
            record,
            trace,
            tokenizer=cast("Any", CharacterTokenizer()),
            teacher_hint_family="synthetic",
            teacher_hint_text="Synthetic teacher nudge.",
        )


def test_opd_generated_coding_identity_does_not_require_ladder_scenario_id() -> None:
    record, trace = _synthetic_legacy_episode()
    record.update(
        task_id="synthetic-code-task",
        family="coding",
        partition="train",
        organism_reward={"reward": 0.0},
    )
    record["key"] = {"task_id": "synthetic-code-task", "sample_index": 0}
    row: Any = build_opd_episode_row(
        record,
        trace,
        tokenizer=cast("Any", CharacterTokenizer()),
        teacher_hint_family="synthetic",
        teacher_hint_text="Synthetic teacher nudge.",
    )
    assert row["problem_id"] == "synthetic-code-task"


class RetainedChatTokenizer(CharacterTokenizer):
    """Small chat tokenizer that preserves assistant reasoning and native turn boundaries."""

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        **_kwargs: object,
    ) -> str:
        del tools, tokenize, enable_thinking
        rendered = ""
        for message in messages:
            role = message["role"]
            if role == "assistant":
                rendered += "<assistant><think>" + str(message.get("reasoning_content", ""))
                rendered += "</think>" + str(message["content"]) + "<|im_end|>"
            else:
                rendered += f"<{role}>" + str(message["content"]) + f"</{role}>"
        return rendered + ("<assistant><think>" if add_generation_prompt else "")


def synthetic_multiturn_opd_trace() -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    tokenizer = RetainedChatTokenizer()
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Synthetic system."},
        {"role": "user", "content": "Synthetic task."},
    ]
    trace: list[dict[str, Any]] = []
    prompt = tokenizer.apply_chat_template(
        messages, tools=[], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    for turn_index, action in enumerate(["Synthetic first action.", "Synthetic second action."]):
        raw_completion = "</think>" + action
        trace.append(
            {
                "record": "turn",
                "episode_id": "synthetic",
                "turn": turn_index,
                "raw_completion": raw_completion,
                "completion_token_ids": tokenizer.encode(raw_completion),
                "completion_logprobs": [-0.1] * len(tokenizer.encode(raw_completion)),
                "matched_stop_token_id": None,
                "stop_reason": "max_tokens",
                "prompt_token_ids": tokenizer.encode(prompt),
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "rendered_prompt_text": prompt,
                "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "rendered_prompt_is_full": True,
                "history_reasoning": "retain",
                "thinking_budget_fired": False,
                "runaway": False,
            }
        )
        messages.append({"role": "assistant", "content": action})
        if turn_index == 0:
            messages.append({"role": "tool", "content": "Synthetic environment feedback."})
            prompt += (
                raw_completion
                + "<|im_end|><tool>Synthetic environment feedback.</tool><assistant><think>"
            )
    trace.append(
        {
            "record": "episode_summary",
            "episode_id": "synthetic",
            "messages": messages,
            "final_report_prompt_text": "Synthetic report prompt.",
            "final_report_prompt_is_full": False,
            "final_report_prompt_sha256": hashlib.sha256(
                (prompt + "Synthetic report prompt.").encode()
            ).hexdigest(),
        }
    )
    return (
        {
            "record": "ladder_episode",
            "key": {"scenario_id": "synthetic", "sample_index": 0},
            "partition": "train",
            "organism_reward": {"reward": 0.0},
        },
        trace,
        tokenizer,
    )


def test_opd_multiturn_environment_tokens_have_no_student_density() -> None:
    record, trace, tokenizer = synthetic_multiturn_opd_trace()
    row: Any = build_opd_episode_row(
        record,
        trace,
        tokenizer=tokenizer,
        teacher_hint_family="synthetic",
        teacher_hint_text="Synthetic teacher nudge.",
    )
    assert row["included_turns"] == 2
    assert any(value == 0 for value in row["loss_mask"])
    for active, sampled_logp in zip(row["loss_mask"], row["student_sampling_logps"], strict=True):
        if not active:
            assert sampled_logp == 0.0


def test_opd_preserves_turn_attribution_for_assistant_advantages() -> None:
    record, trace, tokenizer = synthetic_multiturn_opd_trace()
    row: Any = build_opd_episode_row(
        record,
        trace,
        tokenizer=tokenizer,
        teacher_hint_family="synthetic",
        teacher_hint_text="Synthetic teacher nudge.",
    )
    assert len(row["token_turn_indices"]) == len(row["completion_token_ids"])
    assert {
        turn
        for active, turn in zip(row["loss_mask"], row["token_turn_indices"], strict=True)
        if active
    } == {0, 1}
    assert all(
        turn == -1
        for active, turn in zip(row["loss_mask"], row["token_turn_indices"], strict=True)
        if not active
    )


def test_opd_loss_reports_student_own_sample_nll() -> None:
    logps = torch.tensor([[-0.2, -90.0, -0.4]])
    sampling = torch.tensor([[-0.1, 0.0, -0.3]])
    teacher = torch.tensor([[-0.05, 0.0, -0.2]])
    _loss, metrics = distillation_loss(
        logps, sampling, teacher, weight_mode="opd", loss_mask=torch.tensor([[1, 0, 1]])
    )
    assert metrics["student_own_sample_nll"] == pytest.approx(0.2)
    assert metrics["student_current_nll"] == pytest.approx(0.3)


@pytest.mark.parametrize(("teacher_logp", "expected_loss"), [(-0.1, 0.0), (-0.05, -0.05)])
def test_generated_trace_to_teacher_scores_to_real_opd_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    teacher_logp: float,
    expected_loss: float,
) -> None:
    record, trace, tokenizer = synthetic_multiturn_opd_trace()
    row: Any = build_opd_episode_row(
        record,
        trace,
        tokenizer=tokenizer,
        teacher_hint_family="synthetic",
        teacher_hint_text="Synthetic teacher nudge.",
    )
    samples_path = tmp_path / "samples.jsonl"
    teacher_path = tmp_path / "teacher.jsonl"
    samples_path.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(PositionScoreEngine, "invalid_logp", teacher_logp)
    result = score_round(
        samples_path=samples_path,
        output_path=teacher_path,
        model="synthetic-model",
        llm_class=PositionScoreEngine,
        sampling_params_class=FakeSamplingParams,
        torch_module=FakeTorch(),
    )
    assert result.generated_count == 1
    loaded = load_training_rows(samples_path, teacher_path, weight_mode="opd")
    batch = collate_completion(loaded)
    current = batch["student_sampling_logps"].clone()
    current[~batch["loss_mask"]] = -900.0
    loss, metrics = distillation_loss(
        current,
        batch["student_sampling_logps"],
        batch["teacher_logps"],
        weight_mode="opd",
        loss_mask=batch["loss_mask"],
    )
    assert loss.item() == pytest.approx(expected_loss)
    assert metrics["student_own_sample_nll"] == pytest.approx(0.1)
    assert loaded[0]["prompt_token_ids"] == row["prompt_token_ids"]
    assert len(row["teacher_prompt_token_ids"]) > len(row["prompt_token_ids"])
    assert all(
        logp == 0.0
        for active, logp in zip(row["loss_mask"], loaded[0]["teacher_logps"], strict=True)
        if not active
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("partition", "eval"), ("student_hint_strength", "specific"), ("loss_mask", None)],
)
def test_opd_training_refuses_corrupted_agentic_scope(
    tmp_path: Path, field: str, value: object
) -> None:
    record, trace, tokenizer = synthetic_multiturn_opd_trace()
    row: Any = build_opd_episode_row(
        record,
        trace,
        tokenizer=tokenizer,
        teacher_hint_family="synthetic",
        teacher_hint_text="Synthetic teacher nudge.",
    )
    row[field] = value
    samples = tmp_path / "samples.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    samples.write_text(json.dumps(row) + "\n")
    score = {
        "key": row["key"],
        "prompt_token_ids": row["prompt_token_ids"],
        "completion_token_ids": row["completion_token_ids"],
        "teacher_logps": row["student_sampling_logps"],
    }
    teacher.write_text(json.dumps(score) + "\n")
    with pytest.raises(ValueError, match=r"unhinted|partition|mask"):
        load_training_rows(samples, teacher, weight_mode="opd")


def test_opd_refuses_reencoded_student_context_even_when_action_ids_match() -> None:
    record, trace = _synthetic_legacy_episode()
    record.update(partition="train", organism_reward={"reward": 0.0})
    summary: Any = next(row for row in trace if row["record"] == "episode_summary")
    summary["messages"][0]["content"] = "Different synthetic system."
    with pytest.raises(ValueError, match="sampled context"):
        build_opd_episode_row(
            record,
            trace,
            tokenizer=cast("Any", CharacterTokenizer()),
            teacher_hint_family="synthetic",
            teacher_hint_text="Synthetic teacher nudge.",
        )
