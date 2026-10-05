"""Model-specific prompt, tool, and stop-token protocols for the hack ladder."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from reward_hacking.ladder import chat
from reward_hacking.ladder.tools import ParsedToolCall, ParsedTurn, parse_tool_calls

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

type ProtocolName = Literal["qwen", "olmo3", "gemma3"]
type ReasoningMode = Literal["qwen", "none"]
type TokenizerLike = PreTrainedTokenizerBase
type Message = dict[str, object]
type ToolParser = Callable[[str], ParsedTurn]
type FeedbackSerializer = Callable[[str], Message]
type StopTokenResolver = Callable[[TokenizerLike], tuple[int, ...]]


class PromptRenderer(Protocol):
    """Render one complete prompt using the selected model-family chat format."""

    def __call__(
        self,
        tokenizer: TokenizerLike,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
        *,
        add_generation_prompt: bool = True,
        chat_template_kwargs: Mapping[str, str] | None = None,
    ) -> str: ...


_OLMO_FUNCTION_CALLS = re.compile(r"<function_calls>(.*?)</function_calls>", re.DOTALL)
_GEMMA_BASH_BLOCK = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_GEMMA_TOOL_INSTRUCTIONS_SUFFIX = (
    "To run a command, reply with one fenced bash code block containing exactly one command. "
    "You may provide one block per command."
)
_QWEN_STOP_ID_COUNT = 2


@dataclass(frozen=True, slots=True)
class ChatRendering:
    """The chat-template behavior that differs between model families."""

    render: PromptRenderer
    supports_pure_append: bool


@dataclass(frozen=True, slots=True)
class ModelProtocol:
    """One frozen model-family protocol used by the ladder loop."""

    name: ProtocolName
    chat_rendering: ChatRendering
    reasoning_mode: ReasoningMode
    tool_call_parser: ToolParser
    tool_feedback_serializer: FeedbackSerializer
    stop_token_resolver: StopTokenResolver

    @property
    def supports_pure_append(self) -> bool:
        """Return whether history can use the Qwen byte-preserving append path."""
        return self.chat_rendering.supports_pure_append

    def render_prompt(
        self,
        tokenizer: TokenizerLike,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
        *,
        add_generation_prompt: bool = True,
        chat_template_kwargs: Mapping[str, str] | None = None,
    ) -> str:
        """Render one policy-visible prompt with this protocol's role adaptation."""
        return self.chat_rendering.render(
            tokenizer,
            messages,
            tools,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )

    def render_prompt_continuation(
        self,
        tokenizer: TokenizerLike,
        messages: Sequence[Mapping[str, object]],
        *,
        chat_template_kwargs: Mapping[str, str] | None = None,
    ) -> str:
        """Render a Qwen continuation; other templates require a complete re-render."""
        if not self.supports_pure_append:
            raise ValueError(f"protocol {self.name!r} does not support prompt continuations")
        return chat.render_prompt_continuation(
            tokenizer,
            messages,
            chat_template_kwargs=chat_template_kwargs,
        )

    def render_empty_assistant_completion(
        self,
        tokenizer: TokenizerLike,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
        *,
        chat_template_kwargs: Mapping[str, str] | None = None,
    ) -> str:
        """Render the synthetic closure used only for Qwen runaway retention."""
        if self.name != "qwen":
            return ""
        return chat.render_empty_assistant_completion(
            tokenizer,
            messages,
            tools,
            chat_template_kwargs=chat_template_kwargs,
        )

    def render_assistant_terminator_suffix(
        self,
        tokenizer: TokenizerLike,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
        *,
        chat_template_kwargs: Mapping[str, str] | None = None,
    ) -> str:
        """Render the Qwen assistant terminator needed by its append path."""
        if self.name != "qwen":
            return ""
        return chat.render_assistant_terminator_suffix(
            tokenizer,
            messages,
            tools,
            chat_template_kwargs=chat_template_kwargs,
        )

    def parse_completion(self, raw_completion: str) -> ParsedTurn:
        """Parse one model completion into the ladder's shared internal representation."""
        return self.tool_call_parser(raw_completion)

    def serialize_tool_feedback(self, content: str) -> Message:
        """Serialize one environment result into a role-safe message."""
        return self.tool_feedback_serializer(content)

    def stop_token_ids(self, tokenizer: TokenizerLike) -> tuple[int, ...]:
        """Resolve this protocol's end markers from the tokenizer vocabulary/configuration."""
        return self.stop_token_resolver(tokenizer)

    def is_runaway(self, raw_completion: str, finish_reason: str | None) -> bool:
        """Return whether an output-length stop is a discarded reasoning runaway."""
        return (
            self.reasoning_mode == "qwen"
            and finish_reason == "length"
            and "</think>" not in raw_completion
        )

    def report_tool_call_count(self, raw_completion: str, parsed: ParsedTurn) -> int:
        """Count inert final-report calls while preserving the legacy Qwen marker metric."""
        if self.name == "qwen":
            return raw_completion.count("<tool_call>")
        return len(parsed.tool_calls)


def _normalize_olmo_messages(messages: Sequence[Mapping[str, object]]) -> list[Message]:
    normalized: list[Message] = []
    for message in messages:
        copied = dict(message)
        if copied.get("role") == "tool":
            copied["role"] = "environment"
        if copied.get("role") == "assistant" and not copied.get("tool_calls"):
            copied.pop("tool_calls", None)
        normalized.append(copied)
    return normalized


def _normalize_gemma_messages(  # noqa: C901, PLR0912 - role normalization stays explicit
    messages: Sequence[Mapping[str, object]],
    tool_instructions: str,
) -> list[Message]:
    system_messages = [message for message in messages if message.get("role") == "system"]
    system_prefix = ""
    if system_messages:
        system_prefix = str(system_messages[0].get("content", ""))
    normalized: list[Message] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "assistant":
            content = str(message.get("content", ""))
            calls = message.get("tool_calls", ())
            if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
                blocks = []
                for call in calls:
                    if not isinstance(call, Mapping):
                        raise TypeError("assistant tool calls must be mappings")
                    function = call.get("function")
                    if not isinstance(function, Mapping) or function.get("name") != "bash":
                        raise ValueError("Gemma history contains a non-bash tool call")
                    arguments = function.get("arguments")
                    if not isinstance(arguments, Mapping) or not isinstance(
                        arguments.get("command"), str
                    ):
                        raise TypeError("Gemma bash history is missing a string command")
                    blocks.append(f"```bash\n{arguments['command']}\n```")
                if blocks:
                    content = "\n".join(part for part in (content, *blocks) if part)
            _append_gemma_message(normalized, "assistant", content)
            continue
        if role in {"tool", "environment"}:
            _append_gemma_message(normalized, "user", f"Tool result:\n{message.get('content', '')}")
            continue
        if role == "user":
            _append_gemma_message(normalized, "user", str(message.get("content", "")))
            continue
        raise ValueError(f"Gemma history contains unsupported role {role!r}")

    first_user_index = next(
        (index for index, message in enumerate(normalized) if message["role"] == "user"), None
    )
    if first_user_index is None:
        raise ValueError("Gemma history must contain a user message")
    first_user = normalized[first_user_index]
    prefix_parts = [part for part in (system_prefix, tool_instructions) if part]
    first_user["content"] = "\n\n".join((*prefix_parts, str(first_user["content"])))
    return normalized


def _append_gemma_message(messages: list[Message], role: str, content: str) -> None:
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{content}"
        return
    messages.append({"role": role, "content": content})


def _gemma_tool_instructions(tools: Sequence[Mapping[str, object]]) -> str:
    if len(tools) != 1:
        raise ValueError("Gemma rendering requires exactly the bash tool schema")
    function = tools[0].get("function")
    if not isinstance(function, Mapping) or function.get("name") != "bash":
        raise ValueError("Gemma rendering requires a function named bash")
    description = function.get("description")
    parameters = function.get("parameters")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Gemma bash tool schema requires a description")
    if not isinstance(parameters, Mapping):
        raise TypeError("Gemma bash tool schema requires a parameters object")
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping) or set(properties) != {"command"}:
        raise ValueError("Gemma bash tool schema must define only the command parameter")
    command_parameter = properties["command"]
    if not isinstance(command_parameter, Mapping) or command_parameter.get("type") != "string":
        raise ValueError("Gemma bash command parameter must be a string")
    command_description = command_parameter.get("description")
    if not isinstance(command_description, str) or not command_description.strip():
        raise ValueError("Gemma bash command parameter requires a description")
    if parameters.get("required") != ["command"]:
        raise ValueError("Gemma bash tool schema must require the command parameter")
    return (
        f"{description.strip()} The bash tool accepts one string argument named command: "
        f"{command_description.strip()} {_GEMMA_TOOL_INSTRUCTIONS_SUFFIX}"
    )


def _render_qwen(
    tokenizer: TokenizerLike,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]],
    *,
    add_generation_prompt: bool = True,
    chat_template_kwargs: Mapping[str, str] | None = None,
) -> str:
    return chat.render_prompt(
        tokenizer,
        messages,
        tools,
        enable_thinking=True,
        add_generation_prompt=add_generation_prompt,
        chat_template_kwargs=chat_template_kwargs,
    )


def _render_olmo3(
    tokenizer: TokenizerLike,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]],
    *,
    add_generation_prompt: bool = True,
    chat_template_kwargs: Mapping[str, str] | None = None,
) -> str:
    return chat.render_prompt(
        tokenizer,
        _normalize_olmo_messages(messages),
        tools,
        enable_thinking=False,
        add_generation_prompt=add_generation_prompt,
        chat_template_kwargs=chat_template_kwargs,
    )


def _render_gemma3(
    tokenizer: TokenizerLike,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]],
    *,
    add_generation_prompt: bool = True,
    chat_template_kwargs: Mapping[str, str] | None = None,
) -> str:
    return chat.render_prompt(
        tokenizer,
        _normalize_gemma_messages(messages, _gemma_tool_instructions(tools)),
        (),
        enable_thinking=False,
        add_generation_prompt=add_generation_prompt,
        chat_template_kwargs=chat_template_kwargs,
    )


def _parse_olmo_completion(  # noqa: C901, PLR0911 - each malformed shape has specific feedback
    raw_completion: str,
) -> ParsedTurn:
    matches = list(_OLMO_FUNCTION_CALLS.finditer(raw_completion))
    if raw_completion.count("<function_calls>") != len(matches) or raw_completion.count(
        "</function_calls>"
    ) != len(matches):
        return ParsedTurn("", "", (), "", "OLMo function calls have mismatched markup tags.")
    if not matches:
        return ParsedTurn("", raw_completion.strip(), (), "", None)
    calls = []
    for match in matches:
        try:
            module = ast.parse(match.group(1).strip(), mode="exec")
        except SyntaxError as error:
            return ParsedTurn("", "", (), "", f"Malformed OLMo function call: {error.msg}.")
        if not module.body:
            return ParsedTurn("", "", (), "", "OLMo function call blocks must not be empty.")
        for statement in module.body:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                return ParsedTurn("", "", (), "", "OLMo function calls must be call expressions.")
            call = statement.value
            if not isinstance(call.func, ast.Name):
                return ParsedTurn("", "", (), "", "OLMo function name is invalid.")
            if call.func.id != "bash":
                return ParsedTurn("", "", (), "", f"Unknown tool function: {call.func.id}.")
            if call.args or len(call.keywords) != 1 or call.keywords[0].arg != "command":
                return ParsedTurn(
                    "", "", (), "", "The bash tool requires exactly one command parameter."
                )
            try:
                command = ast.literal_eval(call.keywords[0].value)
            except (ValueError, SyntaxError):
                return ParsedTurn("", "", (), "", "The bash command must be a quoted string.")
            if not isinstance(command, str) or not command.strip():
                return ParsedTurn("", "", (), "", "The bash command must be a non-empty string.")
            calls.append(("bash", {"command": command}))
    outside = _remove_spans(raw_completion, matches).strip()
    return ParsedTurn(
        "", outside, tuple(ParsedToolCall(name, arguments) for name, arguments in calls), "", None
    )


def _parse_gemma_completion(raw_completion: str) -> ParsedTurn:
    matches = list(_GEMMA_BASH_BLOCK.finditer(raw_completion))
    if raw_completion.count("```") != 2 * len(matches):
        return ParsedTurn("", "", (), "", "Gemma commands must use complete fenced bash blocks.")
    calls = []
    for match in matches:
        command = match.group(1).strip()
        if not command:
            return ParsedTurn("", "", (), "", "The bash command must be a non-empty string.")
        calls.append(ParsedToolCall("bash", {"command": command}))
    outside = _remove_spans(raw_completion, matches).strip()
    return ParsedTurn("", outside, tuple(calls), "", None)


def _remove_spans(text: str, matches: Sequence[re.Match[str]]) -> str:
    for match in reversed(matches):
        text = text[: match.start()] + text[match.end() :]
    return text


def _feedback_environment(content: str) -> Message:
    return {"role": "environment", "content": content}


def _feedback_gemma(content: str) -> Message:
    return {"role": "user", "content": f"Tool result:\n{content}"}


def _ids(value: object) -> tuple[int, ...]:
    values = (
        (value,)
        if isinstance(value, int)
        else tuple(value)
        if isinstance(value, (list, tuple))
        else ()
    )
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in values):
        raise ValueError(f"tokenizer stop-token ids must be non-negative integers, got {value!r}")
    return tuple(dict.fromkeys(cast("tuple[int, ...]", values)))


def _convert_token(tokenizer: TokenizerLike, token: str) -> int | None:
    value = tokenizer.convert_tokens_to_ids(token)
    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    if value == unknown_token_id:
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _resolve_qwen_stop_ids(tokenizer: TokenizerLike) -> tuple[int, ...]:
    ids: list[int] = []
    for token in ("<|endoftext|>", "<|im_end|>"):
        token_id = _convert_token(tokenizer, token)
        if token_id is not None:
            ids.append(token_id)
    ids.extend(_ids(getattr(tokenizer, "eos_token_id", None)))
    resolved = tuple(dict.fromkeys(ids))
    if len(resolved) < _QWEN_STOP_ID_COUNT:
        raise ValueError("Qwen tokenizer must expose both end-of-turn and end-of-text ids")
    return resolved[:_QWEN_STOP_ID_COUNT]


def _resolve_olmo_stop_ids(tokenizer: TokenizerLike) -> tuple[int, ...]:
    resolved = _ids(getattr(tokenizer, "eos_token_id", None))
    if not resolved:
        raise ValueError("OLMo tokenizer must expose an eos_token_id")
    return resolved


def _resolve_gemma_stop_ids(tokenizer: TokenizerLike) -> tuple[int, ...]:
    ids = list(_ids(getattr(tokenizer, "eos_token_id", None)))
    end_of_turn = _convert_token(tokenizer, "<end_of_turn>")
    if not ids or end_of_turn is None:
        raise ValueError("Gemma tokenizer must expose eos_token_id and <end_of_turn>")
    ids.append(end_of_turn)
    resolved = tuple(dict.fromkeys(ids))
    if not resolved:
        raise ValueError("Gemma tokenizer must expose eos_token_id or <end_of_turn>")
    return resolved


QWEN_PROTOCOL = ModelProtocol(
    name="qwen",
    chat_rendering=ChatRendering(_render_qwen, supports_pure_append=True),
    reasoning_mode="qwen",
    tool_call_parser=parse_tool_calls,
    tool_feedback_serializer=lambda content: {"role": "tool", "content": content},
    stop_token_resolver=_resolve_qwen_stop_ids,
)
OLMO3_PROTOCOL = ModelProtocol(
    name="olmo3",
    chat_rendering=ChatRendering(_render_olmo3, supports_pure_append=False),
    reasoning_mode="none",
    tool_call_parser=_parse_olmo_completion,
    tool_feedback_serializer=_feedback_environment,
    stop_token_resolver=_resolve_olmo_stop_ids,
)
GEMMA3_PROTOCOL = ModelProtocol(
    name="gemma3",
    chat_rendering=ChatRendering(_render_gemma3, supports_pure_append=False),
    reasoning_mode="none",
    tool_call_parser=_parse_gemma_completion,
    tool_feedback_serializer=_feedback_gemma,
    stop_token_resolver=_resolve_gemma_stop_ids,
)

PROTOCOLS: Mapping[ProtocolName, ModelProtocol] = {
    "qwen": QWEN_PROTOCOL,
    "olmo3": OLMO3_PROTOCOL,
    "gemma3": GEMMA3_PROTOCOL,
}


def _read_model_type(snapshot_dir: str | Path) -> str | None:
    config_path = Path(snapshot_dir) / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, Mapping):
        raise TypeError(f"model config {config_path} must contain a JSON object")
    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        raise TypeError(f"model config {config_path} has no string model_type")
    return model_type


def select_protocol(
    tokenizer: TokenizerLike | None,
    snapshot_dir: str | Path,
    *,
    explicit: str | None = None,
) -> ModelProtocol:
    """Select a protocol by explicit name or local tokenizer/config model family."""
    if explicit is not None:
        if explicit not in PROTOCOLS:
            raise ValueError(f"unknown ladder protocol {explicit!r}")
        return PROTOCOLS[cast("ProtocolName", explicit)]
    model_type = _read_model_type(snapshot_dir)
    if model_type is None and tokenizer is not None:
        model_type = getattr(tokenizer, "init_kwargs", {}).get("model_type")
    if not isinstance(model_type, str):
        model_type = (
            "" if tokenizer is None else str(getattr(tokenizer, "name_or_path", "")).lower()
        )
    if model_type == "olmo3" or "olmo-3" in model_type.lower() or "olmo3" in model_type.lower():
        return OLMO3_PROTOCOL
    if model_type == "gemma3" or "gemma-3" in model_type.lower() or "gemma3" in model_type.lower():
        return GEMMA3_PROTOCOL
    if model_type.startswith("qwen") or model_type == "":
        return QWEN_PROTOCOL
    raise ValueError(f"unknown ladder protocol for model family {model_type!r}")


__all__ = [
    "GEMMA3_PROTOCOL",
    "OLMO3_PROTOCOL",
    "PROTOCOLS",
    "QWEN_PROTOCOL",
    "ChatRendering",
    "ModelProtocol",
    "select_protocol",
]
