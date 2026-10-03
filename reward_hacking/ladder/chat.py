"""Chat-template rendering and context helpers for the hack ladder."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import httpx2

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from transformers import PreTrainedTokenizerBase

type TemplateSource = Literal["sidecar", "embedded"]
MAX_MODEL_LEN_TIMEOUT_SECONDS = 30.0


def render_prompt(  # noqa: PLR0913 - explicit template controls keep each render auditable
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]],
    *,
    enable_thinking: bool = True,
    add_generation_prompt: bool = True,
    chat_template_kwargs: Mapping[str, str] | None = None,
) -> str:
    """Render conversation messages, optionally adding the assistant-generation prefix."""
    template_kwargs = {} if chat_template_kwargs is None else dict(chat_template_kwargs)
    rendered = tokenizer.apply_chat_template(
        cast("Any", [dict(message) for message in messages]),
        tools=cast("Any", [dict(tool) for tool in tools]),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        **cast("Any", template_kwargs),
    )
    if not isinstance(rendered, str):
        raise TypeError(f"chat template returned {type(rendered).__name__}, expected str")
    return rendered


def render_prompt_continuation(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, object]],
    *,
    chat_template_kwargs: Mapping[str, str] | None = None,
) -> str:
    """Render appended messages and a generation prefix without re-rendering prior history."""
    anchor: dict[str, object] = {"role": "user", "content": ""}
    anchored_prompt = render_prompt(
        tokenizer,
        [anchor, *messages],
        (),
        chat_template_kwargs=chat_template_kwargs,
    )
    rendered_anchor = render_prompt(
        tokenizer,
        [anchor],
        (),
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    if not anchored_prompt.startswith(rendered_anchor):
        raise ValueError(
            "chat template changed the rendered prompt anchor while appending messages"
        )
    return anchored_prompt[len(rendered_anchor) :]


def render_empty_assistant_completion(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]],
    *,
    chat_template_kwargs: Mapping[str, str] | None = None,
) -> str:
    """Render the empty assistant closure after this conversation's open generation prefix."""
    generation_prefix = render_prompt(
        tokenizer, messages, tools, chat_template_kwargs=chat_template_kwargs
    )
    empty_assistant: dict[str, object] = {
        "role": "assistant",
        "reasoning_content": "",
        "content": "",
        "tool_calls": [],
    }
    rendered_empty_turn = render_prompt(
        tokenizer,
        [*messages, empty_assistant],
        tools,
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    if rendered_empty_turn.startswith(generation_prefix):
        return rendered_empty_turn[len(generation_prefix) :]

    # TMAX omits the think block for an empty historical assistant, unlike this open prefix.
    canary_reasoning = "__ladder_empty_assistant_reasoning_canary__"
    if canary_reasoning in generation_prefix:
        raise ValueError("conversation contains the empty-assistant rendering canary")
    rendered_canary_turn = render_prompt(
        tokenizer,
        [*messages, {**empty_assistant, "reasoning_content": canary_reasoning}],
        tools,
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    if not rendered_canary_turn.startswith(generation_prefix):
        raise ValueError("chat template cannot append an assistant turn to its generation prefix")
    completion = rendered_canary_turn[len(generation_prefix) :]
    if not completion.startswith(canary_reasoning):
        raise ValueError("chat template changed the assistant reasoning canary")
    return completion[len(canary_reasoning) :]


def render_assistant_terminator_suffix(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]],
    *,
    chat_template_kwargs: Mapping[str, str] | None = None,
) -> str:
    """Render the template's assistant end marker and separator after generated text."""
    generation_prefix = render_prompt(
        tokenizer, messages, tools, chat_template_kwargs=chat_template_kwargs
    )
    reasoning_canary = "__ladder_terminator_reasoning_canary__"
    content_canary = "__ladder_terminator_content_canary__"
    if reasoning_canary in generation_prefix or content_canary in generation_prefix:
        raise ValueError("conversation contains an assistant-terminator rendering canary")
    assistant_message: dict[str, object] = {
        "role": "assistant",
        "reasoning_content": reasoning_canary,
        "content": content_canary,
        "tool_calls": [],
    }
    rendered_turn = render_prompt(
        tokenizer,
        [*messages, assistant_message],
        tools,
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    if not rendered_turn.startswith(generation_prefix):
        raise ValueError("chat template cannot append an assistant turn to its generation prefix")
    completion = rendered_turn[len(generation_prefix) :]
    content_end = completion.find(content_canary)
    if content_end < 0 or completion.find(content_canary, content_end + 1) >= 0:
        raise ValueError("chat template changed the assistant content canary")
    terminator_suffix = completion[content_end + len(content_canary) :]
    if not terminator_suffix:
        raise ValueError("chat template omitted the assistant end marker")
    return terminator_suffix


def count_prompt_tokens(tokenizer: PreTrainedTokenizerBase, rendered_prompt: str) -> int:
    """Count tokens in an already-rendered prompt without adding tokenizer special tokens."""
    return len(tokenizer(rendered_prompt, add_special_tokens=False).input_ids)


def is_pure_append(previous_prompt: str, previous_completion: str, prompt: str) -> bool:
    """Whether the re-rendered prompt preserves the previous prompt and raw completion exactly."""
    return prompt.startswith(previous_prompt + previous_completion)


def context_room(prompt_tokens: int, max_model_len: int) -> int:
    """Return the remaining model context tokens for a rendered prompt."""
    return max_model_len - prompt_tokens


def strip_reasoning(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Copy messages while removing assistant reasoning fields for the final-report fallback."""
    return [
        {
            key: value
            for key, value in message.items()
            if not (message.get("role") == "assistant" and key == "reasoning_content")
        }
        for message in messages
    ]


def template_identity(
    tokenizer: PreTrainedTokenizerBase, snapshot_dir: str | Path
) -> tuple[str, TemplateSource]:
    """Hash the sidecar template transformers prefers, falling back to its embedded template."""
    sidecar_path = Path(snapshot_dir) / "chat_template.jinja"
    if sidecar_path.is_file():
        template_bytes = sidecar_path.read_bytes()
        source: TemplateSource = "sidecar"
    else:
        template = tokenizer.get_chat_template(tools=[])
        template_bytes = template.encode("utf-8")
        source = "embedded"
    return hashlib.sha256(template_bytes).hexdigest(), source


def validate_chat_template_kwargs(
    tokenizer: PreTrainedTokenizerBase,
    snapshot_dir: str | Path,
    *,
    model_id: str,
    chat_template_kwargs: Mapping[str, str],
) -> None:
    """Refuse reasoning-effort templates unless the caller pins their effort explicitly."""
    sidecar_path = Path(snapshot_dir) / "chat_template.jinja"
    if sidecar_path.is_file():
        template_sources = (sidecar_path.read_text(encoding="utf-8"),)
    else:
        template = tokenizer.chat_template
        if isinstance(template, str):
            template_sources = (template,)
        elif isinstance(template, Mapping):
            template_sources = tuple(
                source for source in template.values() if isinstance(source, str)
            )
        else:
            template_sources = (tokenizer.get_chat_template(tools=[]),)

    if any("reasoning_effort" in source for source in template_sources) and (
        "reasoning_effort" not in chat_template_kwargs
    ):
        raise ValueError(
            f"model {model_id!r} chat template references reasoning_effort; "
            "set it with --chat-template-kwarg MODEL_ID:reasoning_effort=VALUE"
        )


def fetch_max_model_len(base_url: str, model_id: str) -> int:
    """Read the advertised context length for exactly one matching model from ``/v1/models``."""
    with httpx2.Client(timeout=MAX_MODEL_LEN_TIMEOUT_SECONDS, trust_env=False) as client:
        response = client.get(f"{base_url.rstrip('/')}/v1/models")
        response.raise_for_status()
        if response.status_code != HTTPStatus.OK:
            raise ValueError(f"vLLM models returned HTTP {response.status_code}, expected 200")
        payload = response.json()

    if not isinstance(payload, dict):
        raise TypeError("vLLM /v1/models response must be an object")
    if "data" not in payload:
        raise ValueError("vLLM /v1/models response is missing data")
    if not isinstance(payload["data"], list):
        raise TypeError("vLLM /v1/models data must be a list")
    models = payload["data"]
    if any(not isinstance(model, dict) for model in models):
        raise ValueError("vLLM /v1/models data entries must be objects")
    matching_models = [model for model in models if model.get("id") == model_id]
    if len(matching_models) != 1:
        raise ValueError(
            f"vLLM /v1/models returned {len(matching_models)} entries for model {model_id!r}, expected one"
        )
    max_model_len = matching_models[0].get("max_model_len")
    if type(max_model_len) is not int or max_model_len <= 0:
        raise ValueError(f"vLLM model {model_id!r} has invalid max_model_len {max_model_len!r}")
    return max_model_len
