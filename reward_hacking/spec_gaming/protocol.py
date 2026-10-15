"""Native Qwen tool parsing and exact append-only prompt capture."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

from jsonschema import ValidationError
from jsonschema.validators import validator_for

from reward_hacking.ladder.chat import (
    render_assistant_terminator_suffix,
    render_prompt,
    render_prompt_continuation,
)
from reward_hacking.ladder.tools import ParsedToolCall, parse_qwen_tool_calls
from reward_hacking.model_backend import RenderedCompletion

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

PROMPT_HISTORY_SCHEMA_VERSION = 2


class NativeActionKind(StrEnum):
    """The action a completed Qwen message requests from its host."""

    TEXT = "text"
    TOOL = "tool"
    CHECK = "check"
    HAND_BACK = "hand_back"
    FORMAT_ERROR = "format_error"


@dataclass(frozen=True, slots=True)
class NativeToolCall:
    """One schema-validated native tool call."""

    name: str
    arguments: dict[str, object]
    kind: NativeActionKind


@dataclass(frozen=True, slots=True)
class NativeResponse:
    """A customer-facing message, host action, or non-executable format error."""

    kind: NativeActionKind
    reasoning: str
    content: str
    calls: tuple[NativeToolCall, ...] = ()
    format_error: str | None = None


class NativeQwenProtocol:
    """Parse Qwen XML calls against the source's unchanged native tool schemas."""

    def __init__(
        self,
        tool_schemas: Sequence[Mapping[str, object]],
        *,
        check_tool_name: str = "check",
        hand_back_tool_name: str = "hand_back",
        thinking_enabled: bool = True,
    ) -> None:
        """Keep original schemas and configure the two host-owned action names."""
        if not check_tool_name or not hand_back_tool_name or check_tool_name == hand_back_tool_name:
            raise ValueError("check and hand_back tool names must be distinct nonempty strings")
        self._tool_schemas = tuple(deepcopy(dict(tool)) for tool in tool_schemas)
        self._parameter_schemas = _parameter_schemas(self._tool_schemas)
        if check_tool_name not in self._parameter_schemas:
            raise ValueError(f"check tool {check_tool_name!r} is absent from the supplied schemas")
        if hand_back_tool_name not in self._parameter_schemas:
            raise ValueError(
                f"hand_back tool {hand_back_tool_name!r} is absent from the supplied schemas"
            )
        self._check_tool_name = check_tool_name
        self._hand_back_tool_name = hand_back_tool_name
        self._thinking_enabled = thinking_enabled

    @property
    def tool_schemas(self) -> tuple[dict[str, object], ...]:
        """Return deep copies of the original schemas used for prompting and validation."""
        return tuple(deepcopy(tool) for tool in self._tool_schemas)

    def parse_completion(
        self,
        raw_completion: str,
        *,
        finish_reason: str | None = "stop",
    ) -> NativeResponse:
        """Parse one sampled response and return no calls when any part is invalid."""
        if finish_reason not in {"stop", "length"}:
            raise ValueError(
                "unsupported completion finish_reason from the vLLM completion backend: "
                f"{finish_reason!r}; expected one of ['length', 'stop']"
            )
        parsed = parse_qwen_tool_calls(raw_completion)
        if finish_reason == "length":
            return _format_error(
                "Completion stopped at the token limit before the response ended.",
                reasoning=parsed.reasoning,
            )
        if parsed.format_error is not None:
            return _format_error(parsed.format_error, reasoning=parsed.reasoning)

        reasoning, content, thought_error = self._split_message(raw_completion)
        if thought_error is not None:
            return _format_error(thought_error, reasoning=reasoning)

        if parsed.trailing_text.strip():
            return _format_error("Unexpected text after a native tool call.", reasoning=reasoning)
        return self._parse_calls(parsed.tool_calls, reasoning=reasoning, content=content)

    def _parse_calls(
        self,
        lexical_calls: Sequence[ParsedToolCall],
        *,
        reasoning: str,
        content: str,
    ) -> NativeResponse:
        if not lexical_calls:
            return NativeResponse(
                kind=NativeActionKind.TEXT,
                reasoning=reasoning,
                content=content,
            )

        calls: list[NativeToolCall] = []
        for lexical_call in lexical_calls:
            if lexical_call.name not in self._parameter_schemas:
                return _format_error(
                    f"Unknown tool function: {lexical_call.name}.", reasoning=reasoning
                )
            try:
                arguments = _decode_arguments(
                    lexical_call,
                    self._parameter_schemas[lexical_call.name],
                )
            except ValueError as error:
                return _format_error(str(error), reasoning=reasoning)

            action_kind = NativeActionKind.TOOL
            if lexical_call.name == self._check_tool_name:
                action_kind = NativeActionKind.CHECK
            elif lexical_call.name == self._hand_back_tool_name:
                action_kind = NativeActionKind.HAND_BACK
            if calls and calls[-1].kind is NativeActionKind.HAND_BACK:
                return _format_error(
                    "A hand_back call must be the final call in its response.",
                    reasoning=reasoning,
                )
            calls.append(
                NativeToolCall(
                    name=lexical_call.name,
                    arguments=arguments,
                    kind=action_kind,
                )
            )
        response_kind = calls[0].kind if len(calls) == 1 else NativeActionKind.TOOL
        return NativeResponse(
            kind=response_kind,
            reasoning=reasoning,
            content=content,
            calls=tuple(calls),
        )

    def _split_message(self, raw_completion: str) -> tuple[str, str, str | None]:
        """Separate private Qwen thinking from the assistant's visible message."""
        if "</think>" in raw_completion:
            reasoning, _, content = raw_completion.partition("</think>")
            return reasoning, content.lstrip("\r\n"), None
        if self._thinking_enabled or "<think>" in raw_completion:
            return raw_completion, "", "Qwen thinking block ended without </think>."
        return "", raw_completion, None


@dataclass(frozen=True, slots=True)
class CapturedPrompt:
    """A rendered prompt and the host-only suffix added after the prior sampled turn."""

    text: str
    token_ids: tuple[int, ...]
    host_suffix_text: str = ""
    host_suffix_token_ids: tuple[int, ...] = ()


class PromptHistory:
    """Retain sampled token segmentation while appending rendered host messages."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tool_schemas: Sequence[Mapping[str, object]],
        initial_messages: Sequence[Mapping[str, object]],
        *,
        chat_template_kwargs: Mapping[str, str] | None = None,
        enable_thinking: bool = True,
    ) -> None:
        """Start from the exact host-provided messages and source-native tool schemas."""
        self._tokenizer = tokenizer
        self._tool_schemas = tuple(deepcopy(dict(tool)) for tool in tool_schemas)
        self._messages = _copy_messages(initial_messages)
        self._chat_template_kwargs = dict(chat_template_kwargs or {})
        self._enable_thinking = enable_thinking
        self._current_prompt: CapturedPrompt | None = None
        self._current_prompt_messages: list[dict[str, object]] | None = None
        self._pending_completion: RenderedCompletion | None = None
        self._sampled_completions: list[RenderedCompletion] = []
        self._pending_completion_prefix_text: str | None = None
        self._sampled_prefix_texts: list[str] = []

    @property
    def messages(self) -> tuple[dict[str, object], ...]:
        """Return a copy of every host-authored message in conversation order."""
        return tuple(deepcopy(message) for message in self._messages)

    @property
    def tool_schemas(self) -> tuple[dict[str, object], ...]:
        """Return deep copies of the unchanged schemas rendered into prompts."""
        return tuple(deepcopy(tool) for tool in self._tool_schemas)

    @property
    def sampled_completions(self) -> tuple[RenderedCompletion, ...]:
        """Return sampled outputs separately from host-added prompt syntax."""
        return tuple(self._sampled_completions)

    @property
    def sampled_prefix_texts(self) -> tuple[str, ...]:
        """Return tokenizer-decoded sampled text separately from raw backend text."""
        return tuple(self._sampled_prefix_texts)

    @property
    def current_prompt(self) -> CapturedPrompt | None:
        """Return the last exact prompt capture, if one has been rendered."""
        return self._current_prompt

    def render(self) -> CapturedPrompt:
        """Render the initial prompt or append host messages to the exact prior token prefix."""
        if self._current_prompt is None:
            prompt_text = render_prompt(
                self._tokenizer,
                self._messages,
                self._tool_schemas,
                enable_thinking=self._enable_thinking,
                chat_template_kwargs=self._chat_template_kwargs,
            )
            self._current_prompt = CapturedPrompt(
                text=prompt_text,
                token_ids=tuple(self._encode(prompt_text)),
            )
            self._current_prompt_messages = deepcopy(self._messages)
            return self._current_prompt

        if self._pending_completion is None:
            return self._current_prompt
        if self._current_prompt_messages is None:
            raise ValueError("rendered prompt is missing its message snapshot")
        appended_messages = self._messages[len(self._current_prompt_messages) :]
        if not appended_messages:
            raise ValueError("cannot render another prompt before appending a host response")
        if self._messages[: len(self._current_prompt_messages)] != self._current_prompt_messages:
            raise ValueError("prompt history changed an already-rendered message")

        terminator = render_assistant_terminator_suffix(
            self._tokenizer,
            self._current_prompt_messages,
            self._tool_schemas,
            enable_thinking=self._enable_thinking,
            chat_template_kwargs=self._chat_template_kwargs,
        )
        continuation = render_prompt_continuation(
            self._tokenizer,
            appended_messages,
            enable_thinking=self._enable_thinking,
            chat_template_kwargs=self._chat_template_kwargs,
        )
        sampled_completion = self._pending_completion
        if self._pending_completion_prefix_text is None:
            raise ValueError("pending completion is missing its decoded sample prefix")
        terminator = self._remove_sampled_eos_from_terminator(
            terminator,
            sampled_completion.token_ids,
        )
        host_suffix_text = terminator + continuation
        host_suffix_token_ids = tuple(self._encode(host_suffix_text))
        previous_prompt = self._current_prompt
        self._current_prompt = CapturedPrompt(
            text=previous_prompt.text + self._pending_completion_prefix_text + host_suffix_text,
            token_ids=(
                *previous_prompt.token_ids,
                *sampled_completion.token_ids,
                *host_suffix_token_ids,
            ),
            host_suffix_text=host_suffix_text,
            host_suffix_token_ids=host_suffix_token_ids,
        )
        self._current_prompt_messages = deepcopy(self._messages)
        self._pending_completion = None
        self._pending_completion_prefix_text = None
        return self._current_prompt

    def accept_completion(self, completion: RenderedCompletion) -> None:
        """Store exact sampled bytes, IDs, and logprobs for the prompt most recently sent."""
        if self._current_prompt is None:
            raise ValueError("render a prompt before accepting a completion")
        if self._pending_completion is not None:
            raise ValueError("render the next prompt before accepting another completion")
        if completion.prompt_token_ids != self._current_prompt.token_ids:
            raise ValueError("completion did not echo the exact captured prompt token IDs")
        if completion.prompt_tokens != len(self._current_prompt.token_ids):
            raise ValueError("completion prompt token count differs from the captured prompt")
        decoded_prefix_text = self._tokenizer.decode(
            list(completion.token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded_prefix_text, str):
            raise TypeError("tokenizer.decode must return text for sampled token IDs")
        self._pending_completion = completion
        self._pending_completion_prefix_text = decoded_prefix_text
        self._sampled_completions.append(completion)
        self._sampled_prefix_texts.append(decoded_prefix_text)

    def append_tool_response(self, name: str, content: str) -> None:
        """Append one source-tool result as a host-authored Qwen tool message."""
        if not name:
            raise ValueError("tool response name must be nonempty")
        self._append_after_completion({"role": "tool", "name": name, "content": content})

    def append_user_message(self, content: str) -> None:
        """Append one host-authored user message, including a simulator reply when applicable."""
        self._append_after_completion({"role": "user", "content": content})

    def append_assistant_message(self, content: str) -> None:
        """Append a host-authored assistant message such as a format correction."""
        self._append_after_completion({"role": "assistant", "content": content})

    def to_state(self) -> dict[str, object]:
        """Return JSON-compatible capture state suitable for an episode journal."""
        return {
            "schema_version": PROMPT_HISTORY_SCHEMA_VERSION,
            "tool_schemas": deepcopy(list(self._tool_schemas)),
            "messages": deepcopy(self._messages),
            "chat_template_kwargs": dict(self._chat_template_kwargs),
            "enable_thinking": self._enable_thinking,
            "current_prompt": _prompt_to_state(self._current_prompt),
            "current_prompt_messages": deepcopy(self._current_prompt_messages),
            "pending_completion": _completion_to_state(self._pending_completion),
            "pending_completion_prefix_text": self._pending_completion_prefix_text,
            "sampled_completions": [
                _completion_to_state(completion) for completion in self._sampled_completions
            ],
            "sampled_prefix_texts": list(self._sampled_prefix_texts),
        }

    @classmethod
    def from_state(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        state: Mapping[str, object],
    ) -> PromptHistory:
        """Reopen exact prompt and sampled-token state without re-encoding prior turns."""
        tools, messages, template_kwargs, enable_thinking = _prompt_history_configuration(state)
        history = cls(
            tokenizer,
            tools,
            messages,
            chat_template_kwargs=template_kwargs,
            enable_thinking=enable_thinking,
        )
        (
            history._current_prompt,
            history._current_prompt_messages,
            history._pending_completion,
            history._pending_completion_prefix_text,
            history._sampled_completions,
            history._sampled_prefix_texts,
        ) = _prompt_history_capture(state)
        return history

    def _append_after_completion(self, message: Mapping[str, object]) -> None:
        if self._pending_completion is None:
            raise ValueError("a sampled completion must precede a host response")
        self._messages.append(dict(message))

    def _encode(self, text: str) -> list[int]:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if any(type(token_id) is not int for token_id in token_ids):
            raise TypeError("tokenizer.encode must return a list of integer token IDs")
        return token_ids

    def _remove_sampled_eos_from_terminator(
        self,
        terminator: str,
        sampled_token_ids: Sequence[int],
    ) -> str:
        eos_token_id = self._tokenizer.eos_token_id
        if not sampled_token_ids or eos_token_id is None or sampled_token_ids[-1] != eos_token_id:
            return terminator
        eos_token = self._tokenizer.eos_token
        if not isinstance(eos_token, str) or not eos_token or not terminator.startswith(eos_token):
            raise ValueError("chat template terminator does not start with the sampled EOS token")
        return terminator[len(eos_token) :]


def _parameter_schemas(
    tool_schemas: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    parameter_schemas: dict[str, dict[str, object]] = {}
    for tool in tool_schemas:
        function = tool.get("function")
        if not isinstance(function, Mapping):
            raise TypeError("native tool schema must contain a function object")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name:
            raise TypeError("native tool schema must contain a nonempty function name")
        if name in parameter_schemas:
            raise ValueError(f"duplicate native tool schema name: {name}")
        if not isinstance(parameters, Mapping):
            raise TypeError(f"native tool {name!r} must contain an object parameters schema")
        schema = deepcopy(dict(parameters))
        validator_for(schema).check_schema(schema)
        parameter_schemas[name] = schema
    return parameter_schemas


def _prompt_history_configuration(
    state: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, str], bool]:
    if state.get("schema_version") != PROMPT_HISTORY_SCHEMA_VERSION:
        raise ValueError("unsupported prompt-history state schema")
    raw_tools = state.get("tool_schemas")
    raw_messages = state.get("messages")
    raw_kwargs = state.get("chat_template_kwargs")
    enable_thinking = state.get("enable_thinking")
    if not isinstance(raw_tools, list):
        raise TypeError("prompt-history tool_schemas must be a list of objects")
    tools: list[dict[str, object]] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            raise TypeError("prompt-history tool_schemas must be a list of objects")
        tools.append(deepcopy(dict(raw_tool)))
    if not isinstance(raw_messages, list):
        raise TypeError("prompt-history messages must be a list")
    if not isinstance(raw_kwargs, Mapping):
        raise TypeError("prompt-history chat_template_kwargs must be a string mapping")
    template_kwargs: dict[str, str] = {}
    for key, value in raw_kwargs.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("prompt-history chat_template_kwargs must be a string mapping")
        template_kwargs[key] = value
    if type(enable_thinking) is not bool:
        raise TypeError("prompt-history enable_thinking must be a boolean")
    return tools, _copy_messages(raw_messages), template_kwargs, enable_thinking


def _prompt_history_capture(
    state: Mapping[str, object],
) -> tuple[
    CapturedPrompt | None,
    list[dict[str, object]] | None,
    RenderedCompletion | None,
    str | None,
    list[RenderedCompletion],
    list[str],
]:
    raw_prompt = state.get("current_prompt")
    raw_prompt_messages = state.get("current_prompt_messages")
    if (raw_prompt is None) != (raw_prompt_messages is None):
        raise ValueError("prompt-history prompt and message snapshot must be present together")
    prompt = None if raw_prompt is None else _prompt_from_state(raw_prompt)
    prompt_messages = None
    if raw_prompt_messages is not None:
        if not isinstance(raw_prompt_messages, list):
            raise TypeError("prompt-history current_prompt_messages must be a list")
        prompt_messages = _copy_messages(raw_prompt_messages)
    pending_completion = _completion_from_state(state.get("pending_completion"))
    pending_completion_prefix_text = state.get("pending_completion_prefix_text")
    if pending_completion_prefix_text is not None and not isinstance(
        pending_completion_prefix_text, str
    ):
        raise TypeError("pending sampled completion prefix text must be a string or null")
    raw_completions = state.get("sampled_completions")
    if not isinstance(raw_completions, list):
        raise TypeError("prompt-history sampled_completions must be a list")
    sampled_completions: list[RenderedCompletion] = []
    for value in raw_completions:
        completion = _completion_from_state(value)
        if completion is None:
            raise TypeError("sampled completion history cannot contain null")
        sampled_completions.append(completion)
    raw_prefix_texts = state.get("sampled_prefix_texts")
    if not isinstance(raw_prefix_texts, list) or any(
        not isinstance(prefix_text, str) for prefix_text in raw_prefix_texts
    ):
        raise TypeError("prompt-history sampled_prefix_texts must be a list of strings")
    sampled_prefix_texts = list(raw_prefix_texts)
    _validate_prompt_history_capture(
        prompt,
        pending_completion,
        pending_completion_prefix_text,
        sampled_completions,
        sampled_prefix_texts,
    )
    return (
        prompt,
        prompt_messages,
        pending_completion,
        pending_completion_prefix_text,
        sampled_completions,
        sampled_prefix_texts,
    )


def _validate_prompt_history_capture(
    prompt: CapturedPrompt | None,
    pending_completion: RenderedCompletion | None,
    pending_completion_prefix_text: str | None,
    sampled_completions: Sequence[RenderedCompletion],
    sampled_prefix_texts: Sequence[str],
) -> None:
    if len(sampled_completions) != len(sampled_prefix_texts):
        raise ValueError("sampled completion and decoded-prefix histories differ in length")
    if pending_completion is not None:
        if prompt is None:
            raise ValueError("pending completion has no captured prompt")
        if pending_completion_prefix_text is None:
            raise ValueError("pending completion has no decoded sample prefix")
        if pending_completion.prompt_token_ids != prompt.token_ids:
            raise ValueError("pending completion does not match the captured prompt")
        if not sampled_completions or sampled_completions[-1] != pending_completion:
            raise ValueError("pending completion is absent from sampled history")
        if sampled_prefix_texts[-1] != pending_completion_prefix_text:
            raise ValueError("pending decoded prefix is absent from sampled history")
    elif pending_completion_prefix_text is not None:
        raise ValueError("decoded sample prefix has no pending completion")


def _decode_arguments(
    call: ParsedToolCall,
    parameter_schema: Mapping[str, object],
) -> dict[str, object]:
    validator = validator_for(parameter_schema)(parameter_schema)
    decoded: dict[str, object] = {}
    for name, raw_value in call.arguments.items():
        if _accepts_string_value(validator, name, raw_value):
            decoded[name] = raw_value
            continue
        try:
            decoded[name] = json.loads(raw_value, parse_constant=_reject_non_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"Invalid arguments for tool {call.name}: parameter {name!r} is not valid JSON "
                "for its declared type."
            ) from error
    try:
        cast("Any", validator).validate(decoded)
    except ValidationError as error:
        raise ValueError(f"Invalid arguments for tool {call.name}: {error.message}") from error
    return decoded


def _accepts_string_value(validator: object, name: str, value: str) -> bool:
    """Preserve lexical strings when the declared parameter schema accepts them."""
    errors = cast("Any", validator).iter_errors({name: value})
    return not any(tuple(error.absolute_path) == (name,) for error in errors)


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non-JSON constant {value!r}")


def _format_error(message: str, *, reasoning: str = "") -> NativeResponse:
    return NativeResponse(
        kind=NativeActionKind.FORMAT_ERROR,
        reasoning=reasoning,
        content="",
        format_error=message,
    )


def _copy_messages(messages: Sequence[object]) -> list[dict[str, object]]:
    copied: list[dict[str, object]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("prompt-history messages must be objects")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise TypeError("prompt-history messages must have a nonempty string role")
        copied.append(deepcopy(dict(message)))
    return copied


def _prompt_to_state(prompt: CapturedPrompt | None) -> dict[str, object] | None:
    if prompt is None:
        return None
    return {
        "text": prompt.text,
        "token_ids": list(prompt.token_ids),
        "host_suffix_text": prompt.host_suffix_text,
        "host_suffix_token_ids": list(prompt.host_suffix_token_ids),
    }


def _prompt_from_state(value: object) -> CapturedPrompt:
    if not isinstance(value, Mapping):
        raise TypeError("prompt-history current_prompt must be an object")
    text = value.get("text")
    host_suffix_text = value.get("host_suffix_text")
    if not isinstance(text, str) or not isinstance(host_suffix_text, str):
        raise TypeError("prompt-history prompt text fields must be strings")
    return CapturedPrompt(
        text=text,
        token_ids=_integer_tuple(value.get("token_ids"), field="current_prompt.token_ids"),
        host_suffix_text=host_suffix_text,
        host_suffix_token_ids=_integer_tuple(
            value.get("host_suffix_token_ids"),
            field="current_prompt.host_suffix_token_ids",
        ),
    )


def _completion_to_state(completion: RenderedCompletion | None) -> dict[str, object] | None:
    if completion is None:
        return None
    return {
        "text": completion.text,
        "finish_reason": completion.finish_reason,
        "matched_stop": completion.matched_stop,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "token_ids": list(completion.token_ids),
        "token_logprobs": list(completion.token_logprobs),
        "prompt_token_ids": (
            None if completion.prompt_token_ids is None else list(completion.prompt_token_ids)
        ),
    }


def _completion_from_state(value: object) -> RenderedCompletion | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("prompt-history sampled completion must be an object")
    text = value.get("text")
    finish_reason = value.get("finish_reason")
    matched_stop = value.get("matched_stop")
    prompt_tokens = value.get("prompt_tokens")
    completion_tokens = value.get("completion_tokens")
    raw_logprobs = value.get("token_logprobs")
    if not isinstance(text, str):
        raise TypeError("sampled completion text must be a string")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise TypeError("sampled completion finish_reason must be a string or null")
    if (
        matched_stop is not None
        and type(matched_stop) is not int
        and not isinstance(matched_stop, str)
    ):
        raise TypeError("sampled completion matched_stop must be an integer, string, or null")
    if type(prompt_tokens) is not int or type(completion_tokens) is not int:
        raise TypeError("sampled completion token counts must be integers")
    if not isinstance(raw_logprobs, list) or any(
        type(logprob) not in {int, float} for logprob in raw_logprobs
    ):
        raise TypeError("sampled completion token_logprobs must be a list of numbers")
    raw_prompt_ids = value.get("prompt_token_ids")
    return RenderedCompletion(
        text=text,
        finish_reason=finish_reason,
        matched_stop=matched_stop,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        token_ids=_integer_tuple(value.get("token_ids"), field="sampled_completion.token_ids"),
        token_logprobs=tuple(float(logprob) for logprob in raw_logprobs),
        prompt_token_ids=(
            None
            if raw_prompt_ids is None
            else _integer_tuple(raw_prompt_ids, field="sampled_completion.prompt_token_ids")
        ),
    )


def _integer_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise TypeError(f"prompt-history {field} must be a list of integer token IDs")
    return tuple(value)
