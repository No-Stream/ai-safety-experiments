from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.ladder.tools import parse_qwen_tool_calls
from reward_hacking.model_backend import RenderedCompletion
from reward_hacking.spec_gaming import protocol as native_protocol
from reward_hacking.spec_gaming.protocol import (
    NativeActionKind,
    NativeQwenProtocol,
    PromptHistory,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import PreTrainedTokenizerBase


TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "write_cells",
            "description": "Write supplied cell values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workbook_id": {"type": "string"},
                    "cells": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object", "additionalProperties": {"type": "string"}},
                    "enabled": {"type": "boolean"},
                    "tolerance": {"type": "number"},
                },
                "required": ["workbook_id", "cells", "metadata", "enabled", "tolerance"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check",
            "description": "Check operational acceptance.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hand_back",
            "description": "Close the task and hand back the artifact.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def _call(name: str, parameters: Sequence[tuple[str, str]] = ()) -> str:
    body = "".join(
        f"<parameter={parameter_name}>\n{value}\n</parameter>\n"
        for parameter_name, value in parameters
    )
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _completion(
    text: str,
    *,
    prompt_token_ids: Sequence[int],
    token_ids: Sequence[int] = (301, 302),
    finish_reason: str | None = "stop",
    matched_stop: int | str | None = None,
) -> RenderedCompletion:
    return RenderedCompletion(
        text=text,
        finish_reason=finish_reason,
        matched_stop=matched_stop,
        prompt_tokens=len(prompt_token_ids),
        completion_tokens=len(token_ids),
        token_ids=tuple(token_ids),
        token_logprobs=tuple(-0.2 for _ in token_ids),
        prompt_token_ids=tuple(prompt_token_ids),
    )


class NonRoundTrippingTokenizer:
    """A template fixture whose whole-string encoding cannot reproduce prefix IDs."""

    chat_template = "synthetic qwen template"
    eos_token = "<|im_end|>"
    eos_token_id = 248_046

    def __init__(self) -> None:
        self.decoded_samples: dict[tuple[int, ...], str] = {}
        self.thinking_modes: list[bool] = []

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        **_: object,
    ) -> str | list[int]:
        self.thinking_modes.append(enable_thinking)
        rendered = "<tools>" + repr(list(tools)) + "</tools>"
        for message in messages:
            rendered += f"<{message['role']}>"
            if "name" in message:
                rendered += f"[{message['name']}]"
            rendered += str(message.get("content", ""))
            rendered += (
                "<|im_end|>\n" if message["role"] == "assistant" else f"</{message['role']}>"
            )
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(text) + 10_000, sum(ord(character) for character in text) + 20_000]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        key = tuple(token_ids)
        if key == (self.eos_token_id,):
            return self.eos_token
        return self.decoded_samples[key]


def test_native_qwen_calls_decode_schema_types_and_preserve_original_schemas() -> None:
    protocol = NativeQwenProtocol(TOOLS, thinking_enabled=False)
    raw = _call(
        "write_cells",
        (
            ("workbook_id", "0017"),
            ("cells", '["A1", "B2"]'),
            ("metadata", '{"source": "approved"}'),
            ("enabled", "false"),
            ("tolerance", "0.25"),
        ),
    )

    parsed = protocol.parse_completion(raw)

    assert protocol.tool_schemas == tuple(TOOLS)
    assert parsed.kind is NativeActionKind.TOOL
    assert parsed.format_error is None
    assert parsed.calls[0].arguments == {
        "workbook_id": "0017",
        "cells": ["A1", "B2"],
        "metadata": {"source": "approved"},
        "enabled": False,
        "tolerance": 0.25,
    }
    assert type(parsed.calls[0].arguments["enabled"]) is bool


@pytest.mark.parametrize(
    ("raw", "expected_kind", "thinking_enabled"),
    [
        ("The workbook is ready for review.", NativeActionKind.TEXT, False),
        (_call("check"), NativeActionKind.CHECK, False),
        (_call("hand_back"), NativeActionKind.HAND_BACK, False),
        (
            "<think>I should verify the result.</think>The workbook is ready.",
            NativeActionKind.TEXT,
            True,
        ),
    ],
)
def test_native_qwen_distinguishes_text_check_and_hand_back(
    raw: str, expected_kind: NativeActionKind, thinking_enabled: bool
) -> None:
    protocol = NativeQwenProtocol(TOOLS, thinking_enabled=thinking_enabled)

    parsed = protocol.parse_completion(raw)

    assert parsed.kind is expected_kind
    if expected_kind is NativeActionKind.TEXT:
        assert parsed.content in {raw, "The workbook is ready."}
    else:
        assert parsed.calls[0].arguments == {}


@pytest.mark.parametrize(
    ("raw", "expected_error"),
    [
        (_call("unknown_tool"), "Unknown tool function: unknown_tool."),
        (
            _call("write_cells", (("workbook_id", "0017"), ("workbook_id", "0018"))),
            "Duplicate tool parameter: workbook_id.",
        ),
        (
            "<tool_call>\n<function=write_cells>\n<parameter=workbook_id>\n0017\n</tool_call>",
            "Tool parameter workbook_id is missing its closing tag.",
        ),
        (
            _call(
                "write_cells",
                (
                    ("workbook_id", "0017"),
                    ("cells", '["A1"]'),
                    ("metadata", "{}"),
                    ("enabled", "1"),
                    ("tolerance", "0.25"),
                ),
            ),
            "Invalid arguments for tool write_cells",
        ),
        (
            _call(
                "write_cells",
                (
                    ("workbook_id", "0017"),
                    ("cells", "[]"),
                    ("metadata", "{}"),
                    ("enabled", "false"),
                    ("tolerance", "false"),
                ),
            ),
            "Invalid arguments for tool write_cells",
        ),
        (
            _call(
                "write_cells",
                (
                    ("workbook_id", "0017"),
                    ("cells", '["A1"]'),
                    ("metadata", "{}"),
                    ("enabled", "false"),
                    ("tolerance", "nope"),
                ),
            ),
            "Invalid arguments for tool write_cells",
        ),
        (
            _call("hand_back")
            + _call(
                "write_cells",
                (
                    ("workbook_id", "0017"),
                    ("cells", "[]"),
                    ("metadata", "{}"),
                    ("enabled", "false"),
                    ("tolerance", "0"),
                ),
            ),
            "A hand_back call must be the final call in its response.",
        ),
    ],
)
def test_native_qwen_rejects_unknown_or_malformed_calls_without_executable_calls(
    raw: str, expected_error: str
) -> None:
    parsed = NativeQwenProtocol(TOOLS, thinking_enabled=False).parse_completion(raw)

    assert parsed.kind is NativeActionKind.FORMAT_ERROR
    assert parsed.format_error is not None
    assert expected_error in parsed.format_error
    assert parsed.calls == ()


def test_native_qwen_does_not_treat_incomplete_thinking_or_length_stops_as_actions() -> None:
    protocol = NativeQwenProtocol(TOOLS, thinking_enabled=True)

    incomplete_think = protocol.parse_completion("<think>unfinished analysis")
    cut_off_call = protocol.parse_completion(
        _call("check")[: -len("</tool_call>")], finish_reason="length"
    )

    assert incomplete_think.kind is NativeActionKind.FORMAT_ERROR
    assert incomplete_think.calls == ()
    assert cut_off_call.kind is NativeActionKind.FORMAT_ERROR
    assert cut_off_call.calls == ()


@pytest.mark.parametrize("finish_reason", ["error", "abort", None, "unexpected"])
def test_native_qwen_rejects_unsupported_finish_reason_before_lexical_parsing(
    finish_reason: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_lexed(_: str) -> object:
        raise AssertionError("unsupported finish reason must fail before lexical parsing")

    monkeypatch.setattr(native_protocol, "parse_qwen_tool_calls", fail_if_lexed)
    protocol = NativeQwenProtocol(TOOLS, thinking_enabled=False)

    with pytest.raises(ValueError, match="unsupported completion finish_reason"):
        protocol.parse_completion(_call("check"), finish_reason=finish_reason)


def test_generic_qwen_lexical_parser_allows_native_names_while_ladder_parser_stays_bash_only() -> (
    None
):
    lexical = parse_qwen_tool_calls(_call("read_cells", (("sheet", "001"),)))
    native = NativeQwenProtocol(TOOLS, thinking_enabled=False).parse_completion(
        _call("read_cells", (("sheet", "001"),))
    )

    assert lexical.format_error is None
    assert lexical.tool_calls[0].name == "read_cells"
    assert lexical.tool_calls[0].arguments == {"sheet": "001"}
    assert native.kind is NativeActionKind.FORMAT_ERROR
    assert "Unknown tool function" in (native.format_error or "")


def test_native_qwen_retains_valid_artifact_batches_and_marks_each_action() -> None:
    protocol = NativeQwenProtocol(TOOLS, thinking_enabled=False)
    raw = (
        _call("check")
        + _call(
            "write_cells",
            (
                ("workbook_id", "0017"),
                ("cells", "[]"),
                ("metadata", "{}"),
                ("enabled", "false"),
                ("tolerance", "0"),
            ),
        )
        + _call("hand_back")
    )

    parsed = protocol.parse_completion(raw)

    assert parsed.kind is NativeActionKind.TOOL
    assert [call.kind for call in parsed.calls] == [
        NativeActionKind.CHECK,
        NativeActionKind.TOOL,
        NativeActionKind.HAND_BACK,
    ]
    assert parsed.calls[1].arguments["workbook_id"] == "0017"


def test_native_qwen_parses_all_checks_for_atomic_episode_budget_validation() -> None:
    protocol = NativeQwenProtocol(TOOLS, thinking_enabled=False)

    parsed = protocol.parse_completion(_call("check") * 6)

    assert parsed.kind is NativeActionKind.TOOL
    assert len(parsed.calls) == 6
    assert all(call.kind is NativeActionKind.CHECK for call in parsed.calls)


def test_native_qwen_discards_an_entire_batch_when_a_later_call_is_unknown() -> None:
    protocol = NativeQwenProtocol(TOOLS, thinking_enabled=False)

    parsed = protocol.parse_completion(_call("check") + _call("unknown_tool"))

    assert parsed.kind is NativeActionKind.FORMAT_ERROR
    assert parsed.calls == ()
    assert parsed.format_error == "Unknown tool function: unknown_tool."


def test_prompt_history_preserves_sampled_prefix_tokens_across_feedback_and_reopen() -> None:
    tokenizer = NonRoundTrippingTokenizer()
    history = PromptHistory(
        cast("PreTrainedTokenizerBase", tokenizer),
        TOOLS,
        [
            {"role": "system", "content": "Synthetic task instructions."},
            {"role": "user", "content": "Inspect the supplied workbook."},
        ],
        enable_thinking=True,
    )
    first_prompt = history.render()
    sampled_tool_call = "<think>Read the workbook.</think>" + _call("check")
    first_completion = _completion(
        sampled_tool_call,
        prompt_token_ids=first_prompt.token_ids,
        token_ids=(701, 702),
    )
    tokenizer.decoded_samples[(701, 702)] = sampled_tool_call

    history.accept_completion(first_completion)
    history.append_tool_response("check", "Operational status: ready.")
    history = PromptHistory.from_state(
        cast("PreTrainedTokenizerBase", tokenizer), history.to_state()
    )
    second_prompt = history.render()

    assert second_prompt.text == (
        first_prompt.text + sampled_tool_call + second_prompt.host_suffix_text
    )
    assert second_prompt.token_ids[: len(first_prompt.token_ids)] == first_prompt.token_ids
    assert (
        second_prompt.token_ids[
            len(first_prompt.token_ids) : len(first_prompt.token_ids)
            + len(first_completion.token_ids)
        ]
        == first_completion.token_ids
    )
    assert second_prompt.host_suffix_token_ids == tuple(
        tokenizer.encode(second_prompt.host_suffix_text, add_special_tokens=False)
    )
    assert len(second_prompt.token_ids) == (
        len(first_prompt.token_ids)
        + len(first_completion.token_ids)
        + len(second_prompt.host_suffix_token_ids)
    )
    assert history.sampled_completions[0].token_ids == (701, 702)
    assert history.sampled_completions[0].token_logprobs == (-0.2, -0.2)

    second_completion = _completion(
        "<think>Ask the customer.</think>May I use the confirmed range?",
        prompt_token_ids=second_prompt.token_ids,
        token_ids=(801, 802, 803),
    )
    tokenizer.decoded_samples[(801, 802, 803)] = (
        "<think>Ask the customer.</think>May I use the confirmed range?"
    )
    history.accept_completion(second_completion)
    history.append_user_message("Yes, that range is correct.")
    reopened = PromptHistory.from_state(
        cast("PreTrainedTokenizerBase", tokenizer), history.to_state()
    )
    third_prompt = reopened.render()

    assert third_prompt.token_ids[: len(second_prompt.token_ids)] == second_prompt.token_ids
    assert (
        third_prompt.token_ids[
            len(second_prompt.token_ids) : len(second_prompt.token_ids)
            + len(second_completion.token_ids)
        ]
        == second_completion.token_ids
    )
    assert "May I use the confirmed range?" in third_prompt.text
    assert "Yes, that range is correct." in third_prompt.text
    assert reopened.sampled_completions[1].token_ids == (801, 802, 803)


def test_prompt_history_keeps_host_terminator_out_of_sampled_logprobs() -> None:
    tokenizer = NonRoundTrippingTokenizer()
    history = PromptHistory(
        cast("PreTrainedTokenizerBase", tokenizer),
        TOOLS,
        [{"role": "user", "content": "Synthetic task."}],
    )
    prompt = history.render()
    completion = _completion(
        "A sampled answer.",
        prompt_token_ids=prompt.token_ids,
        token_ids=(91, 92),
        finish_reason="length",
    )
    tokenizer.decoded_samples[(91, 92)] = "A sampled answer."

    history.accept_completion(completion)
    history.append_user_message("A later user message.")
    next_prompt = history.render()

    assert next_prompt.token_ids[: len(prompt.token_ids)] == prompt.token_ids
    assert next_prompt.token_ids[len(prompt.token_ids) : len(prompt.token_ids) + 2] == (91, 92)
    assert history.sampled_completions[0].token_ids == (91, 92)
    assert history.sampled_completions[0].token_logprobs == (-0.2, -0.2)
    assert len(history.sampled_completions[0].token_ids) == len(
        history.sampled_completions[0].token_logprobs
    )


def test_prompt_history_uses_decoded_sampled_prefix_without_duplicating_sampled_eos() -> None:
    tokenizer = NonRoundTrippingTokenizer()
    history = PromptHistory(
        cast("PreTrainedTokenizerBase", tokenizer),
        TOOLS,
        [{"role": "user", "content": "Synthetic task."}],
        enable_thinking=False,
    )
    prompt = history.render()
    decoded_sample = "A sampled answer.<|im_end|>"
    sampled_token_ids = (501, tokenizer.eos_token_id)
    tokenizer.decoded_samples[sampled_token_ids] = decoded_sample
    completion = _completion(
        "A sampled answer.",
        prompt_token_ids=prompt.token_ids,
        token_ids=sampled_token_ids,
        finish_reason="stop",
        matched_stop=tokenizer.eos_token,
    )

    history.accept_completion(completion)
    history.append_user_message("The simulator agrees.")
    state = history.to_state()
    sampled_completions = cast("list[dict[str, object]]", state["sampled_completions"])
    assert sampled_completions[0]["text"] == "A sampled answer."
    assert state["sampled_prefix_texts"] == [decoded_sample]
    reopened = PromptHistory.from_state(cast("PreTrainedTokenizerBase", tokenizer), state)
    next_prompt = reopened.render()

    assert next_prompt.text == prompt.text + decoded_sample + next_prompt.host_suffix_text
    assert next_prompt.text.count(tokenizer.eos_token) == 1
    assert next_prompt.host_suffix_text.startswith("\n")
    assert tokenizer.eos_token not in next_prompt.host_suffix_text
    assert next_prompt.token_ids[: len(prompt.token_ids)] == prompt.token_ids
    assert (
        next_prompt.token_ids[
            len(prompt.token_ids) : len(prompt.token_ids) + len(sampled_token_ids)
        ]
        == sampled_token_ids
    )
    assert reopened.sampled_completions[0].text == "A sampled answer."
    assert reopened.sampled_prefix_texts == (decoded_sample,)
    assert tokenizer.thinking_modes
    assert all(mode is False for mode in tokenizer.thinking_modes)
