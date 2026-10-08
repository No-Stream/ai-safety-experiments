"""Student context rendering preserves recorded action and environment token segments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.distill.episodes import build_episode_row
from reward_hacking.ladder.chat import render_prompt
from reward_hacking.tests.test_distill_episodes import CharacterTokenizer, _sha256

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class SentenceTemplateTokenizer(CharacterTokenizer):
    def __init__(self, sentence: str = "") -> None:
        self.sentence = sentence

    def get_chat_template(self, *, tools: object) -> str:
        return "synthetic-template:" + self.sentence

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        tools: object,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize
        assert enable_thinking
        rendered = self.sentence if messages and messages[0]["role"] == "system" else ""
        for message in messages:
            role = message["role"]
            if role == "assistant":
                rendered += "[assistant]" + str(message.get("reasoning_content", ""))
                rendered += str(message.get("content", "")) + "<|im_end|>"
            else:
                rendered += f"[{role}]{message['content']}"
        return rendered + ("[assistant]" if add_generation_prompt else "")


def _episode() -> tuple[dict[str, object], list[dict[str, object]]]:
    teacher = SentenceTemplateTokenizer("Be brief. ")
    initial = [
        {"role": "system", "content": "Synthetic instruction."},
        {"role": "user", "content": "Synthetic request."},
    ]
    messages = [
        *initial,
        {"role": "assistant", "content": "A"},
        {"role": "tool", "content": "Synthetic output."},
        {"role": "assistant", "content": "B"},
    ]
    prefixes = [
        render_prompt(cast("Any", teacher), initial, []),
        render_prompt(cast("Any", teacher), messages[:-1], []),
    ]
    trace: list[dict[str, object]] = []
    for index, prefix in enumerate(prefixes):
        raw = "A" if index == 0 else "B"
        ids = teacher.encode(raw + "<|im_end|>")
        trace.append(
            {
                "record": "turn",
                "episode_id": "synthetic",
                "turn": index,
                "raw_completion": raw,
                "completion_token_ids": ids,
                "completion_logprobs": [-0.1] * len(ids),
                "matched_stop_token_id": teacher.end_turn_id,
                "prompt_tokens": len(teacher.encode(prefix)),
                "prompt_token_ids": teacher.encode(prefix),
                "rendered_prompt_text": prefix,
                "rendered_prompt_sha256": _sha256(prefix),
                "rendered_prompt_is_full": True,
                "thinking_budget_fired": False,
                "runaway": False,
                "history_reasoning": "retain",
            }
        )
    trace.append(
        {
            "record": "episode_summary",
            "episode_id": "synthetic",
            "messages": messages,
            "final_report_prompt_text": prefixes[-1],
            "final_report_prompt_sha256": _sha256(prefixes[-1]),
            "final_report_prompt_is_full": True,
        }
    )
    return {"record": "ladder_episode", "key": {"scenario_id": "synthetic"}}, trace


def test_student_render_default_replaces_teacher_sentence_and_preserves_sampled_ids() -> None:
    record, trace = _episode()
    student = SentenceTemplateTokenizer()
    teacher_row = build_episode_row(
        record, trace, tokenizer=cast("Any", student), prompt_render="teacher"
    )
    row = build_episode_row(record, trace, tokenizer=cast("Any", student))
    messages = cast("list[dict[str, object]]", trace[-1]["messages"])
    expected = student.encode(render_prompt(cast("Any", student), messages[:2], []))
    assert row["prompt_token_ids"] == expected
    assert row["completion_token_ids"] == teacher_row["completion_token_ids"]
    assert row["loss_mask"] == teacher_row["loss_mask"]
    assert row["prompt_render"] == "student"
    assert len(cast("list[int]", row["prefix_diff_counts"])) == 2
    assert all(count > 0 for count in cast("list[int]", row["prefix_diff_counts"]))
    assert row["student_template_sha256"]


def test_refuses_student_template_environment_change() -> None:
    record, trace = _episode()

    class ChangedEnvironmentTokenizer(SentenceTemplateTokenizer):
        def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
            return super().apply_chat_template(*args, **kwargs).replace("[tool]", "[changed-tool]")

    with pytest.raises(ValueError, match="student-rendered context"):
        build_episode_row(record, trace, tokenizer=cast("Any", ChangedEnvironmentTokenizer()))


def test_preserves_noncanonical_sampled_action_ids_in_retained_context() -> None:
    record, trace = _episode()
    alternate_action_id = 0x110002

    class AlternateActionTokenizer(SentenceTemplateTokenizer):
        def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str:
            return super().decode(
                [
                    ord("A") if token_id == alternate_action_id else token_id
                    for token_id in token_ids
                ],
                **kwargs,
            )

    trace[0]["completion_token_ids"] = [alternate_action_id, CharacterTokenizer.end_turn_id]
    later_ids = cast("list[int]", trace[1]["prompt_token_ids"])
    # Only the sampled action has the alternate encoding; prompt text is unchanged.
    action_position = len(cast("list[int]", trace[0]["prompt_token_ids"]))
    later_ids[action_position] = alternate_action_id
    tokenizer = AlternateActionTokenizer()
    row = build_episode_row(record, trace, tokenizer=cast("Any", tokenizer))
    assert cast("list[int]", row["completion_token_ids"])[0] == alternate_action_id


def test_refuses_token_merge_across_student_prompt_action_boundary() -> None:
    record, trace = _episode()

    class MergingTokenizer(SentenceTemplateTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            if text.endswith("[assistant]A<|im_end|>"):
                return [*ids[:-3], ord("!")]
            return ids

    with pytest.raises(ValueError, match="clean token boundary"):
        build_episode_row(record, trace, tokenizer=cast("Any", MergingTokenizer()))
