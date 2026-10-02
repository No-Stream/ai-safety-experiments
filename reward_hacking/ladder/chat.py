"""Chat-template rendering and context helpers for the hack ladder."""

from __future__ import annotations

import hashlib
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import httpx2

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from transformers import PreTrainedTokenizerBase

type TemplateSource = Literal["sidecar", "embedded"]
MAX_MODEL_LEN_TIMEOUT_SECONDS = 30.0


def render_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]],
    *,
    enable_thinking: bool = True,
) -> str:
    """Render a complete conversation and assistant-generation prefix with its own template."""
    rendered = tokenizer.apply_chat_template(
        cast("Any", [dict(message) for message in messages]),
        tools=cast("Any", [dict(tool) for tool in tools]),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if not isinstance(rendered, str):
        raise TypeError(f"chat template returned {type(rendered).__name__}, expected str")
    return rendered


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
