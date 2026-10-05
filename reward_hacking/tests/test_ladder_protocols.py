"""Synthetic protocol contracts for model-specific ladder rendering and parsing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from reward_hacking.ladder.chat import render_prompt, render_prompt_continuation
from reward_hacking.ladder.protocols import (
    GEMMA3_PROTOCOL,
    OLMO3_PROTOCOL,
    QWEN_PROTOCOL,
    select_protocol,
)
from reward_hacking.ladder.runner import LADDER_SAMPLING, LADDER_STOP_TOKEN_IDS
from reward_hacking.ladder.tools import parse_tool_calls
from reward_hacking.vllm_server import VLLMServerConfig

DATA_DIR = Path(__file__).parent / "data"


def _qwen_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"fixture": 0})))
    tokenizer.chat_template = (DATA_DIR / "qwen3_5_chat_template.jinja").read_text(encoding="utf-8")
    return tokenizer


def _synthetic_qwen_token_id(token: str) -> int | None:
    return {"<|endoftext|>": 248044, "<|im_end|>": 248046}.get(token)


def _synthetic_transcript() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "Synthetic system instruction."},
        {"role": "user", "content": "Synthetic user request."},
        {
            "role": "assistant",
            "reasoning_content": "Synthetic reasoning.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"command": "printf ready"}},
                }
            ],
        },
        {"role": "tool", "content": "ready\n\n(exit_code=0)"},
        {"role": "user", "content": "Synthetic follow-up."},
    ]


def test_qwen_legacy_render_and_parse_regression_baseline() -> None:
    """Pin the pre-refactor Qwen path before the protocol abstraction exists."""
    tokenizer = _qwen_tokenizer()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Synthetic command tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    messages = _synthetic_transcript()
    rendered = render_prompt(tokenizer, messages, tools)
    parsed = parse_tool_calls(
        "Synthetic reasoning.</think>\n\n<tool_call>\n<function=bash>\n"
        "<parameter=command>\nprintf ready\n</parameter>\n</function>\n</tool_call>"
    )

    assert rendered
    assert parsed.reasoning == "Synthetic reasoning."
    assert [(call.name, call.arguments) for call in parsed.tool_calls] == [
        ("bash", {"command": "printf ready"})
    ]
    assert LADDER_STOP_TOKEN_IDS == (248044, 248046)


def test_qwen_protocol_is_byte_identical_to_legacy_path() -> None:
    tokenizer = _qwen_tokenizer()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Synthetic command tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    messages = _synthetic_transcript()
    raw_completion = (
        "Synthetic reasoning.</think>\n\n<tool_call>\n<function=bash>\n"
        "<parameter=command>\nprintf ready\n</parameter>\n</function>\n</tool_call>"
    )

    legacy_prompt = render_prompt(tokenizer, messages, tools)
    continuation_messages = [
        {"role": "tool", "content": "ready"},
        {"role": "user", "content": "Synthetic continuation."},
    ]
    legacy_continuation = render_prompt_continuation(tokenizer, continuation_messages)
    legacy_parsed = parse_tool_calls(raw_completion)
    protocol_prompt = QWEN_PROTOCOL.render_prompt(tokenizer, messages, tools)
    protocol_continuation = QWEN_PROTOCOL.render_prompt_continuation(
        tokenizer, continuation_messages
    )
    protocol_parsed = QWEN_PROTOCOL.parse_completion(raw_completion)

    assert protocol_prompt == legacy_prompt
    assert protocol_continuation == legacy_continuation
    assert protocol_parsed == legacy_parsed
    assert (
        QWEN_PROTOCOL.stop_token_ids(
            cast(
                "Any",
                SimpleNamespace(
                    eos_token_id=248044,
                    convert_tokens_to_ids=_synthetic_qwen_token_id,
                ),
            )
        )
        == LADDER_STOP_TOKEN_IDS
    )
    assert LADDER_SAMPLING.to_json() == {
        "max_new_tokens": 8192,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "seed": None,
        "stop_token_ids": [248044, 248046],
    }
    assert VLLMServerConfig(
        model="Qwen/test-model",
        revision="test-revision",
        max_model_len=4096,
        gpu_memory_utilization=0.55,
    ).command() == [
        "vllm",
        "serve",
        "Qwen/test-model",
        "--revision",
        "test-revision",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.55",
    ]


def _olmo_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"fixture": 0})))
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<system>{{ message['content'] }}"
        "{% if tools %}<functions>{{ tools | tojson }}</functions>{% endif %}</system>"
        "{% elif message['role'] == 'user' %}<user>{{ message['content'] }}</user>"
        "{% elif message['role'] == 'assistant' %}<assistant>{{ message.get('content', '') }}"
        "{% if message.get('tool_calls') %}<function_calls>"
        "{% for call in message['tool_calls'] %}"
        "{{ call['function']['name'] }}(command={{ call['function']['arguments']['command'] | tojson }})"
        "{% endfor %}</function_calls>{% endif %}</assistant>"
        "{% elif message['role'] == 'environment' or message['role'] == 'tool' %}"
        "<environment>{{ message['content'] }}</environment>{% endif %}"
        "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    return tokenizer


def test_olmo3_render_parse_and_environment_feedback_round_trip() -> None:
    tokenizer = _olmo_tokenizer()
    messages = [
        {"role": "system", "content": "Synthetic system."},
        {"role": "user", "content": "Synthetic request."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "bash", "arguments": {"command": "printf ready"}}}
            ],
        },
        {"role": "environment", "content": "ready"},
        {"role": "assistant", "content": "Synthetic follow-up."},
    ]
    tools = [{"type": "function", "function": {"name": "bash"}}]

    rendered = OLMO3_PROTOCOL.render_prompt(tokenizer, messages, tools)
    parsed = OLMO3_PROTOCOL.parse_completion(
        'I will act.\n<function_calls>bash(command="printf ready")\n</function_calls>'
    )

    assert "<functions>" in rendered
    assert "<function_calls>" in rendered
    assert "<environment>ready</environment>" in rendered
    assert parsed.reasoning == ""
    assert parsed.content == "I will act."
    assert [(call.name, call.arguments) for call in parsed.tool_calls] == [
        ("bash", {"command": "printf ready"})
    ]
    assert OLMO3_PROTOCOL.serialize_tool_feedback("ready") == {
        "role": "environment",
        "content": "ready",
    }
    assert OLMO3_PROTOCOL.stop_token_ids(cast("Any", SimpleNamespace(eos_token_id=100257))) == (
        100257,
    )
    assert OLMO3_PROTOCOL.parse_completion("A plain response without a call.").content == (
        "A plain response without a call."
    )
    assert "<assistant>Synthetic follow-up.</assistant><assistant>" in rendered


def test_olmo3_parser_rejects_malformed_calls_and_accepts_two_calls() -> None:
    parsed = OLMO3_PROTOCOL.parse_completion(
        '<function_calls>bash(command="printf one")\nbash(command="printf two")</function_calls>'
    )
    assert [call.arguments["command"] for call in parsed.tool_calls] == [
        "printf one",
        "printf two",
    ]

    malformed = OLMO3_PROTOCOL.parse_completion(
        '<function_calls>bash(command="unterminated)</function_calls>'
    )
    assert malformed.tool_calls == ()
    assert malformed.format_error


def test_gemma3_parser_has_fenced_bash_round_trip_and_plain_turns() -> None:
    raw_completion = "Narrative.\n```bash\nprintf one\n```\n```bash\nprintf two\n```"
    parsed = GEMMA3_PROTOCOL.parse_completion(raw_completion)

    assert parsed.reasoning == ""
    assert parsed.content == "Narrative."
    assert [call.arguments["command"] for call in parsed.tool_calls] == [
        "printf one",
        "printf two",
    ]
    assert GEMMA3_PROTOCOL.parse_completion("No command this turn.").content == (
        "No command this turn."
    )
    malformed = GEMMA3_PROTOCOL.parse_completion("```python\nprint('wrong')\n```")
    assert malformed.tool_calls == ()
    assert malformed.format_error
    mixed_fences = GEMMA3_PROTOCOL.parse_completion(
        "```bash\nprintf ready\n```\n```python\nprint('wrong')\n```"
    )
    assert mixed_fences.tool_calls == ()
    assert mixed_fences.format_error
    assert GEMMA3_PROTOCOL.serialize_tool_feedback("ready") == {
        "role": "user",
        "content": "Tool result:\nready",
    }


def _gemma_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"fixture": 0})))
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}<user>{{ message['content'] }}</user>"
        "{% elif message['role'] == 'assistant' %}<assistant>{{ message['content'] }}</assistant>"
        "{% else %}{{ raise_exception('unexpected role') }}{% endif %}"
        "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    return tokenizer


def test_gemma3_render_coalesces_tool_results_and_keeps_alternation() -> None:
    messages = [
        {"role": "system", "content": "Synthetic system."},
        {"role": "user", "content": "Synthetic request."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "bash", "arguments": {"command": "printf one"}}},
                {"function": {"name": "bash", "arguments": {"command": "printf two"}}},
            ],
        },
        {"role": "tool", "content": "one"},
        {"role": "tool", "content": "two"},
        {"role": "assistant", "content": "Synthetic follow-up."},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run one command in the working directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to run."}
                    },
                    "required": ["command"],
                },
            },
        }
    ]
    rendered = GEMMA3_PROTOCOL.render_prompt(_gemma_tokenizer(), messages, tools)

    assert rendered.count("<user>") == 2
    assert rendered.count("<assistant>") == 3
    assert rendered.count("Tool result:") == 2
    assert "Run one command in the working directory." in rendered
    assert "The bash tool accepts one string argument named command:" in rendered
    assert "```bash\nprintf one\n```" in rendered
    assert "```bash\nprintf two\n```" in rendered


def test_protocol_selection_uses_config_model_type_and_explicit_override(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"model_type": "gemma3", "eos_token_id": [1, 106]}', encoding="utf-8")
    tokenizer = SimpleNamespace(
        init_kwargs={"model_type": "unknown"},
        eos_token_id=[1, 106],
        convert_tokens_to_ids=lambda token: 106 if token == "<end_of_turn>" else None,
    )

    typed_tokenizer = cast("Any", tokenizer)
    assert select_protocol(typed_tokenizer, tmp_path).name == "gemma3"
    assert select_protocol(None, tmp_path).name == "gemma3"
    assert select_protocol(typed_tokenizer, tmp_path, explicit="olmo3") is OLMO3_PROTOCOL
    assert GEMMA3_PROTOCOL.stop_token_ids(typed_tokenizer) == (1, 106)


def test_protocol_selection_rejects_unknown_family_and_name(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type": "mystery"}', encoding="utf-8")
    tokenizer = SimpleNamespace(
        init_kwargs={}, eos_token_id=1, convert_tokens_to_ids=lambda _: None
    )

    with pytest.raises(ValueError, match="unknown ladder protocol"):
        select_protocol(cast("Any", tokenizer), tmp_path)
    with pytest.raises(ValueError, match="unknown ladder protocol"):
        select_protocol(cast("Any", tokenizer), tmp_path, explicit="mystery")
