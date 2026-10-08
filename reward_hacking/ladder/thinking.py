"""Thinking-token budget constants and marker checks for Qwen ladder episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

QWEN_THINKING_REASONING_START_STRING = "<think>"
QWEN_THINKING_REASONING_END_STRING = "</think>"


@dataclass(frozen=True, slots=True)
class ThinkingBudgetOvershootPolicy:
    """Abort only when enough finished episodes exceed the reasoning budget."""

    max_fraction: float = 0.1
    min_episodes: int = 3

    def __post_init__(self) -> None:
        """Validate the run's fraction and episode threshold."""
        if not 0 <= self.max_fraction <= 1:
            raise ValueError("thinking-budget overshoot max_fraction must be in [0, 1]")
        if type(self.min_episodes) is not int or self.min_episodes < 1:
            raise ValueError("thinking-budget overshoot min_episodes must be a positive integer")

    def check(self, finished_episodes: int, overshoot_episodes: int) -> None:
        """Check after persisting each finished episode, including failures."""
        if not 0 <= overshoot_episodes <= finished_episodes:
            raise ValueError("overshoot episodes must be between zero and finished episodes")
        if (
            overshoot_episodes >= self.min_episodes
            and overshoot_episodes / finished_episodes > self.max_fraction
        ):
            raise RuntimeError(
                "thinking_budget_overshoot: "
                f"{overshoot_episodes}/{finished_episodes} finished episodes exceeded the budget; "
                f"fraction exceeds {self.max_fraction} with minimum {self.min_episodes} episodes"
            )


DEFAULT_THINKING_BUDGET_OVERSHOOT_POLICY = ThinkingBudgetOvershootPolicy()


QWEN_THINKING_REASONING_CONFIG: Mapping[str, str] = MappingProxyType(
    {
        "reasoning_start_str": QWEN_THINKING_REASONING_START_STRING,
        "reasoning_end_str": QWEN_THINKING_REASONING_END_STRING,
    }
)
QWEN_THINKING_REASONING_CONFIG_REPR: Mapping[str, str] = MappingProxyType(
    {
        "__repr_type__": "ReasoningConfig",
        "reasoning_parser": "",
        "reasoning_start_str": QWEN_THINKING_REASONING_START_STRING,
        "reasoning_end_str": QWEN_THINKING_REASONING_END_STRING,
    }
)


def _token_ids(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """Encode text with the served tokenizer and without automatic special tokens."""
    return cast("list[int]", tokenizer(text, add_special_tokens=False).input_ids)


def _find_first_sequence_index(target: list[int], sequence: list[int]) -> int:
    for index in range(len(target) - len(sequence) + 1):
        if target[index : index + len(sequence)] == sequence:
            return index
    return -1


def _find_last_sequence_index(target: list[int], sequence: list[int]) -> int:
    """Return the last start position of a token sequence, like vLLM's tracker."""
    for index in range(len(target) - len(sequence), -1, -1):
        if target[index : index + len(sequence)] == sequence:
            return index
    return -1


def thinking_region_token_count(
    tokenizer: PreTrainedTokenizerBase,
    rendered_prompt: str,
    raw_completion: str,
    *,
    prompt_token_ids: Sequence[int] | None = None,
    completion_token_ids: Sequence[int] | None = None,
) -> int:
    """Count tokens after the current ``<think>`` through its first ``</think>``.

    vLLM starts the Qwen request with an open ``<think>`` in the prompt. Its
    ``_init_state_entry`` counts the prompt suffix after the start-token sequence;
    ``start_thinking = len(prompt) - think_count - 1`` points back to the final
    start token, and later ``+ start_len`` excludes the marker itself. Generated
    continuation tokens are then added to that prompt suffix count.
    """
    start_token_ids = _token_ids(tokenizer, QWEN_THINKING_REASONING_START_STRING)
    end_token_ids = _token_ids(tokenizer, QWEN_THINKING_REASONING_END_STRING)
    # Re-encoding decoded text can merge or split tokens differently from what the server
    # sampled (seen up to +3 at a 2048 budget), so exact ids are used whenever they exist.
    exact_prompt_token_ids = (
        _token_ids(tokenizer, rendered_prompt)
        if prompt_token_ids is None
        else list(prompt_token_ids)
    )
    last_prompt_start = _find_last_sequence_index(exact_prompt_token_ids, start_token_ids)
    last_prompt_end = _find_last_sequence_index(exact_prompt_token_ids, end_token_ids)
    if completion_token_ids is None:
        end_position = raw_completion.find(QWEN_THINKING_REASONING_END_STRING)
        generated_reasoning = raw_completion if end_position < 0 else raw_completion[:end_position]
        generated_token_ids = _token_ids(tokenizer, generated_reasoning)
    else:
        sampled_ids = list(completion_token_ids)
        end_index = _find_first_sequence_index(sampled_ids, end_token_ids)
        generated_token_ids = sampled_ids if end_index < 0 else sampled_ids[:end_index]

    if last_prompt_start > last_prompt_end:
        prompt_suffix_count = len(exact_prompt_token_ids) - (
            last_prompt_start + len(start_token_ids)
        )
        return prompt_suffix_count + len(generated_token_ids)

    completion_start = _find_last_sequence_index(generated_token_ids, start_token_ids)
    if completion_start < 0:
        return 0
    return len(generated_token_ids) - (completion_start + len(start_token_ids))


def forced_thinking_close_inside_tool_call(raw_completion: str) -> bool:
    """Return whether the first reasoning close appears inside an unclosed tool call."""
    marker_start = raw_completion.find(QWEN_THINKING_REASONING_END_STRING)
    if marker_start < 0:
        return False
    tool_call_start = raw_completion.rfind("<tool_call>", 0, marker_start)
    tool_call_end = raw_completion.rfind("</tool_call>", 0, marker_start)
    return tool_call_start > tool_call_end


def server_reasoning_config_matches(server_args: Mapping[str, object]) -> bool:
    """Return whether parsed vLLM startup args contain the exact Qwen budget config."""
    configured = server_args.get("reasoning_config")
    if not isinstance(configured, Mapping):
        return False
    return dict(configured) == QWEN_THINKING_REASONING_CONFIG_REPR
