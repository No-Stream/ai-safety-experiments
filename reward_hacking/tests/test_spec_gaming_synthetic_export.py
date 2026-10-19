"""Synthetic source journals exercise the public spec-gaming export path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.distill.episodes import read_episode_trace_records
from reward_hacking.distill.sft_serving import load_tokenizer
from reward_hacking.distill.spec_gaming_episodes import export_run
from reward_hacking.model_backend import RenderedCompletion
from reward_hacking.spec_gaming.capture import EpisodeJournal
from reward_hacking.spec_gaming.protocol import PromptHistory
from reward_hacking.spec_gaming.retail_episode import HOST_TOOLS

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase


@pytest.fixture(scope="module")
def actor_tokenizer() -> PreTrainedTokenizerBase:
    return load_tokenizer("Qwen/Qwen3.5-0.8B")


@dataclass(frozen=True)
class _SyntheticRetailConversation:
    actor_history: PromptHistory
    initial_actor_history_state: dict[str, object]
    first_actor_history_state: dict[str, object]
    second_prompt_history_state: dict[str, object]
    first_actor_text: str
    simulator_text: str
    first_actor: RenderedCompletion
    simulator_completion: RenderedCompletion
    second_actor: RenderedCompletion


def _synthetic_retail_conversation(
    tokenizer: PreTrainedTokenizerBase,
) -> _SyntheticRetailConversation:
    actor_history = PromptHistory(
        tokenizer,
        HOST_TOOLS,
        [
            {"role": "system", "content": "Help with this synthetic customer request."},
            {"role": "user", "content": "Please update the synthetic order."},
        ],
    )
    first_prompt = actor_history.render()
    initial_actor_history_state = actor_history.to_state()
    first_actor_text = "I will check the synthetic order and update it."
    first_actor = _rendered_completion(tokenizer, first_prompt.token_ids, first_actor_text)
    actor_history.accept_completion(first_actor)
    first_actor_history_state = actor_history.to_state()

    simulator_history = PromptHistory(
        tokenizer, [], [{"role": "user", "content": first_actor_text}]
    )
    simulator_prompt = simulator_history.render()
    simulator_text = "Synthetic customer reply."
    simulator_completion = _rendered_completion(
        tokenizer, simulator_prompt.token_ids, simulator_text
    )
    simulator_history.accept_completion(simulator_completion)
    actor_history.append_user_message(simulator_text)
    second_prompt = actor_history.render()
    second_prompt_history_state = actor_history.to_state()
    second_actor_text = "<tool_call>\n<function=hand_back>\n</function>\n</tool_call>"
    second_actor = _rendered_completion(tokenizer, second_prompt.token_ids, second_actor_text)
    actor_history.accept_completion(second_actor)
    return _SyntheticRetailConversation(
        actor_history=actor_history,
        initial_actor_history_state=initial_actor_history_state,
        first_actor_history_state=first_actor_history_state,
        second_prompt_history_state=second_prompt_history_state,
        first_actor_text=first_actor_text,
        simulator_text=simulator_text,
        first_actor=first_actor,
        simulator_completion=simulator_completion,
        second_actor=second_actor,
    )


def _save_sample(
    journal: EpisodeJournal,
    turn_index: int,
    completion: RenderedCompletion,
    role: str,
) -> tuple[str, dict[str, object]]:
    prompt_token_ids = list(cast("tuple[int, ...]", completion.prompt_token_ids))
    output_token_ids = list(completion.token_ids)
    sampled_logprobs: list[float | None] = list(completion.token_logprobs)
    completion_metadata = {"role": role}
    digest = journal.save_completion(
        turn_index,
        prompt_token_ids,
        output_token_ids,
        sampled_logprobs,
        completion.text,
        completion_metadata,
    )
    return digest, {
        "role": role,
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": output_token_ids,
        "sampled_logprobs": sampled_logprobs,
        "raw_text": completion.text,
    }


def _write_synthetic_journal(
    episode_dir: Path, conversation: _SyntheticRetailConversation
) -> list[dict[str, object]]:
    generation_identity: dict[str, Any] = {"sampling": {"thinking_token_budget": None}}
    initial_state = {
        "status": "running",
        "actor_history": conversation.initial_actor_history_state,
    }
    expected_at_save: list[dict[str, object]] = []
    with EpisodeJournal(episode_dir, generation_identity, initial_state) as journal:
        first_digest, first_expected = _save_sample(journal, 0, conversation.first_actor, "actor")
        expected_at_save.append(first_expected)
        journal.begin_action("actor-message", first_digest, 0, "actor_message", {})
        journal.commit_action(
            "actor-message",
            {"role": "assistant", "content": conversation.first_actor_text},
            {"status": "running", "actor_history": conversation.first_actor_history_state},
        )
        simulator_digest, simulator_expected = _save_sample(
            journal, 1, conversation.simulator_completion, "simulator"
        )
        expected_at_save.append(simulator_expected)
        journal.begin_action("simulator-message", simulator_digest, 0, "simulator_message", {})
        journal.commit_action(
            "simulator-message",
            {"role": "user", "content": conversation.simulator_text},
            {"status": "running", "actor_history": conversation.second_prompt_history_state},
        )
        hand_back_digest, second_expected = _save_sample(
            journal, 2, conversation.second_actor, "actor"
        )
        expected_at_save.append(second_expected)
        journal.begin_action("hand-back", hand_back_digest, 0, "hand_back", {})
        journal.commit_action(
            "hand-back",
            {"status": "hand_back"},
            {
                "status": "hand_back",
                "actor_history": conversation.actor_history.to_state(),
            },
            close=True,
        )
    return expected_at_save


def _write_synthetic_retail_inputs(run_dir: Path, episode_dir: Path, task_id: str) -> None:
    run_dir.mkdir(exist_ok=True)
    (run_dir / "selection-manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "training_eligible": True,
                    "tasks": [
                        {
                            "task_id": task_id,
                            "split": "train",
                            "upstream_split": "synthetic",
                            "group_digest": "synthetic-group",
                            "training_eligible": True,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (episode_dir / "assessment-current.json").write_text(
        json.dumps(
            {
                "assessment": {
                    "operational_proxy": {"status": "measured", "success": False},
                    "native_evaluation": {"status": "unmeasured"},
                    "limitations": ["synthetic native evaluation"],
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "readout.json").write_text(
        json.dumps({"episodes": [{"task_id": task_id, "integrity_awareness": "clean"}]}),
        encoding="utf-8",
    )


def test_export_keeps_ordered_actor_turns_and_omits_simulator_sample(
    tmp_path: Path, actor_tokenizer: PreTrainedTokenizerBase
) -> None:
    task_id = "synthetic-task"
    run_dir = tmp_path / "run"
    episode_dir = run_dir / "episodes" / task_id
    conversation = _synthetic_retail_conversation(actor_tokenizer)
    expected_at_save = _write_synthetic_journal(episode_dir, conversation)
    _write_synthetic_retail_inputs(run_dir, episode_dir, task_id)

    records_path = export_run("retail", run_dir, tmp_path / "export", tokenizer=actor_tokenizer)
    record = cast("dict[str, object]", json.loads(records_path.read_text(encoding="utf-8")))
    trace = read_episode_trace_records(record, records_path)
    turns = [row for row in trace if row["record"] == "turn"]
    actor_completions = [entry for entry in expected_at_save if entry["role"] == "actor"]
    assert record["turn_count"] == len(turns) == len(actor_completions)
    for turn, expected_completion in zip(turns, actor_completions, strict=True):
        assert turn["prompt_token_ids"] == expected_completion["prompt_token_ids"]
        assert turn["completion_token_ids"] == expected_completion["output_token_ids"]
        assert turn["completion_logprobs"] == expected_completion["sampled_logprobs"]
        assert turn["raw_completion"] == expected_completion["raw_text"]


def _rendered_completion(
    tokenizer: PreTrainedTokenizerBase, prompt_token_ids: tuple[int, ...], text: str
) -> RenderedCompletion:
    token_ids = tuple(tokenizer.encode(text, add_special_tokens=False))
    return RenderedCompletion(
        text=text,
        finish_reason="stop",
        matched_stop=None,
        prompt_tokens=len(prompt_token_ids),
        completion_tokens=len(token_ids),
        token_ids=token_ids,
        token_logprobs=tuple(-0.1 for _ in token_ids),
        prompt_token_ids=prompt_token_ids,
    )
