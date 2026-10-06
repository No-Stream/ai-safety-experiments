"""Thinking-token budget constants and marker checks for Qwen ladder episodes."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

QWEN_THINKING_REASONING_START_STRING = "<think>"
QWEN_THINKING_REASONING_END_STRING = "</think>"
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
) -> int:
    """Count tokens after the current ``<think>`` through its first ``</think>``.

    vLLM starts the Qwen request with an open ``<think>`` in the prompt. Its
    ``_init_state_entry`` counts the prompt suffix after the start-token sequence;
    ``start_thinking = len(prompt) - think_count - 1`` points back to the final
    start token, and later ``+ start_len`` excludes the marker itself. Generated
    continuation tokens are then added to that prompt suffix count.
    """
    start_token_ids = _token_ids(tokenizer, QWEN_THINKING_REASONING_START_STRING)
    prompt_token_ids = _token_ids(tokenizer, rendered_prompt)
    last_prompt_start = _find_last_sequence_index(prompt_token_ids, start_token_ids)
    last_prompt_end = _find_last_sequence_index(
        prompt_token_ids, _token_ids(tokenizer, QWEN_THINKING_REASONING_END_STRING)
    )
    end_position = raw_completion.find(QWEN_THINKING_REASONING_END_STRING)
    generated_reasoning = raw_completion if end_position < 0 else raw_completion[:end_position]
    generated_token_ids = _token_ids(tokenizer, generated_reasoning)

    if last_prompt_start > last_prompt_end:
        prompt_suffix_count = len(prompt_token_ids) - (last_prompt_start + len(start_token_ids))
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
