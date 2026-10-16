"""Refuse teacher generations whose prompt carried reasoning-effort steering the student never sees.

Qwen3.8's chat template turns ``reasoning_effort`` into a system-message line ("Reasoning effort is
set to low. Keep your thinking brief..."). Absent the kwarg it renders the xhigh line, so an empty
kwargs dict is not steering-free. Only ``medium`` renders no steering text, and the student templates
(Qwen3.5) have no such knob. A trace sampled under any other setting teaches behaviour conditioned on
an instruction the student's prompt does not contain, so such a trace is refused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from transformers import PreTrainedTokenizerBase

STEERING_FREE_REASONING_EFFORT = "medium"
REASONING_EFFORT_KWARG = "reasoning_effort"
_PROBE_MESSAGES = (
    {"role": "system", "content": "Probe system prompt."},
    {"role": "user", "content": "Probe user prompt."},
)
_PROBE_TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "probe_tool",
            "description": "Probe tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
)


def render_probe_prompt(
    tokenizer: PreTrainedTokenizerBase, chat_template_kwargs: Mapping[str, Any]
) -> str:
    """Render a fixed system, user and tool probe so template steering shows up as a text diff."""
    rendered = tokenizer.apply_chat_template(
        [dict(message) for message in _PROBE_MESSAGES],
        tools=[dict(tool) for tool in _PROBE_TOOLS],
        tokenize=False,
        add_generation_prompt=True,
        **cast("dict[str, Any]", chat_template_kwargs),
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template must return text for the effort probe")
    return rendered


def refuse_effort_steering(
    generating_tokenizer: PreTrainedTokenizerBase,
    chat_template_kwargs: Mapping[str, Any] | None,
    *,
    model_id: str,
) -> None:
    """Raise unless the generating prompt equals the steering-free render of the same probe.

    The reference render pins ``reasoning_effort`` to ``medium`` and keeps every other recorded
    kwarg. Templates without the knob render identically under both, so they pass.
    """
    recorded_kwargs = dict(chat_template_kwargs or {})
    steering_free_kwargs = {
        **recorded_kwargs,
        REASONING_EFFORT_KWARG: STEERING_FREE_REASONING_EFFORT,
    }
    generated_render = render_probe_prompt(generating_tokenizer, recorded_kwargs)
    steering_free_render = render_probe_prompt(generating_tokenizer, steering_free_kwargs)
    if generated_render != steering_free_render:
        raise ValueError(
            f"{model_id!r} generated under chat_template_kwargs={recorded_kwargs!r}, whose prompt "
            "carries reasoning-effort steering text the student prompt lacks; only "
            f"{REASONING_EFFORT_KWARG}={STEERING_FREE_REASONING_EFFORT!r} is steering-free"
        )
