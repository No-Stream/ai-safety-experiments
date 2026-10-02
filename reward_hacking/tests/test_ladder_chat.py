"""Client-side ladder chat rendering using the cached Qwen and TMAX templates.

The templates in ``data/`` are unchanged Apache-2.0 files from Qwen/Qwen3.5-9B at
``c202236235762e1c871ad0ccb60c8ee5ba337b9a`` and allenai/tmax-9b at
``3e1711fd69a8f57567aa6f6dcf425267d3178fa0``. Their license is included beside them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx2
import pytest
from huggingface_hub.constants import HF_HUB_CACHE
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.utils.chat_template_utils import render_jinja_template

from reward_hacking.ladder.chat import (
    context_room,
    count_prompt_tokens,
    fetch_max_model_len,
    is_pure_append,
    render_prompt,
    strip_reasoning,
    template_identity,
)
from reward_hacking.ladder.loop import _report_prompt_messages

DATA_DIR = Path(__file__).parent / "data"
TEMPLATE_PATHS = (
    DATA_DIR / "qwen3_5_chat_template.jinja",
    DATA_DIR / "tmax_chat_template.jinja",
)
TEST_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Return a supplied value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
]


@pytest.fixture(params=TEMPLATE_PATHS, ids=("qwen", "tmax"))
def tokenizer(request: pytest.FixtureRequest) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"fixture": 0})))
    tokenizer.chat_template = cast("Path", request.param).read_text(encoding="utf-8")
    return tokenizer


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "fixture system message"},
        {"role": "user", "content": "fixture task"},
    ]


def _assistant_call(reasoning: str, value: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "reasoning_content": reasoning,
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{value}",
                "type": "function",
                "function": {"name": "echo", "arguments": {"value": value}},
            }
        ],
    }


def _render_closed_turn(tokenizer: PreTrainedTokenizerFast, messages: list[dict[str, Any]]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=cast("Any", TEST_TOOLS),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    assert isinstance(rendered, str)
    return rendered


def test_render_prompt_uses_tool_schema_and_thinking_template(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    messages = [
        *_messages(),
        _assistant_call("first reasoning", "one"),
        {"role": "tool", "content": "one"},
    ]

    rendered = render_prompt(tokenizer, messages, TEST_TOOLS)

    assert '"name": "echo"' in rendered
    assert "<tools>" in rendered
    assert "first reasoning" in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")
    direct_result = cast(
        "Any",
        render_jinja_template(
            [messages],
            tools=cast("Any", TEST_TOOLS),
            chat_template=cast("str", tokenizer.chat_template),
            add_generation_prompt=True,
            enable_thinking=True,
        ),
    )
    assert direct_result[0] == [rendered]


def test_pure_append_holds_across_two_tool_turns(tokenizer: PreTrainedTokenizerFast) -> None:
    initial_messages = _messages()
    initial_prompt = render_prompt(tokenizer, initial_messages, TEST_TOOLS)

    first_assistant = _assistant_call("reasoning one", "one")
    first_completion = _render_closed_turn(tokenizer, [*initial_messages, first_assistant])[
        len(initial_prompt) :
    ]
    after_first_tool = [*initial_messages, first_assistant, {"role": "tool", "content": "one"}]
    second_prompt = render_prompt(tokenizer, after_first_tool, TEST_TOOLS)
    assert is_pure_append(initial_prompt, first_completion, second_prompt)

    second_assistant = _assistant_call("reasoning two", "two")
    second_completion = _render_closed_turn(tokenizer, [*after_first_tool, second_assistant])[
        len(second_prompt) :
    ]
    after_second_tool = [*after_first_tool, second_assistant, {"role": "tool", "content": "two"}]
    third_prompt = render_prompt(tokenizer, after_second_tool, TEST_TOOLS)
    assert is_pure_append(second_prompt, second_completion, third_prompt)


def test_mid_conversation_user_message_drops_qwen_reasoning_and_breaks_prefix() -> None:
    prior_messages = [
        *_messages(),
        _assistant_call("retained reasoning marker", "one"),
        {"role": "tool", "content": "one"},
    ]
    qwen_tokenizer = _tokenizer(TEMPLATE_PATHS[0])
    tmax_tokenizer = _tokenizer(TEMPLATE_PATHS[1])
    prior_qwen_prompt = render_prompt(qwen_tokenizer, prior_messages, TEST_TOOLS)
    qwen_messages = [*prior_messages, {"role": "user", "content": "follow-up"}]
    qwen_prompt = render_prompt(qwen_tokenizer, qwen_messages, TEST_TOOLS)
    tmax_prompt = render_prompt(tmax_tokenizer, qwen_messages, TEST_TOOLS)

    assert "retained reasoning marker" not in qwen_prompt
    assert "retained reasoning marker" in tmax_prompt
    assert not is_pure_append(prior_qwen_prompt, "assistant completion", qwen_prompt)


def test_report_turn_retains_reasoning_with_the_same_policy_for_both_templates() -> None:
    prior_messages: list[dict[str, object]] = [
        *_messages(),
        _assistant_call("retained report reasoning marker", "one"),
        {"role": "tool", "content": "one"},
        {"role": "user", "content": "Write the final report."},
    ]

    report_messages = _report_prompt_messages(prior_messages, history_reasoning="retain")
    qwen_prompt = render_prompt(_tokenizer(TEMPLATE_PATHS[0]), report_messages, TEST_TOOLS)
    tmax_prompt = render_prompt(_tokenizer(TEMPLATE_PATHS[1]), report_messages, TEST_TOOLS)

    assert "retained report reasoning marker" in qwen_prompt
    assert "retained report reasoning marker" in tmax_prompt
    report_content = report_messages[2]["content"]
    assert isinstance(report_content, str)
    assert report_content.startswith("Prior reasoning:\n")
    assert "reasoning_content" not in report_messages[2]
    assert prior_messages[2]["content"] == ""
    assert prior_messages[2]["reasoning_content"] == "retained report reasoning marker"


def _tokenizer(template_path: Path) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"fixture": 0})))
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    return tokenizer


def test_count_prompt_tokens_and_context_room(tokenizer: PreTrainedTokenizerFast) -> None:
    rendered = render_prompt(tokenizer, _messages(), TEST_TOOLS)
    counting_tokenizer = cast("Any", _CountingTokenizer())

    token_count = count_prompt_tokens(counting_tokenizer, rendered)

    assert token_count == len(rendered.split())
    assert context_room(token_count, token_count + 1024) == 1024
    assert context_room(token_count + 1, token_count) == -1


class _CountingTokenizer:
    def __call__(self, prompt: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=prompt.split())


def test_strip_reasoning_copies_messages_without_mutating_them() -> None:
    messages = [
        *_messages(),
        {"role": "assistant", "reasoning_content": "private think", "content": "report"},
    ]

    stripped = strip_reasoning(messages)

    assert stripped[-1] == {"role": "assistant", "content": "report"}
    assert messages[-1]["reasoning_content"] == "private think"
    assert stripped[0] is not messages[0]


def test_template_identity_prefers_sidecar_and_hashes_embedded_template(
    tokenizer: PreTrainedTokenizerFast, tmp_path: Path
) -> None:
    template = cast("str", tokenizer.chat_template)
    sidecar = tmp_path / "chat_template.jinja"
    sidecar.write_text(template, encoding="utf-8")
    expected_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()

    assert template_identity(tokenizer, tmp_path) == (expected_hash, "sidecar")
    assert template_identity(tokenizer, tmp_path / "missing") == (expected_hash, "embedded")


def test_vendored_templates_match_cached_model_snapshots_when_present() -> None:
    hub_cache = Path(HF_HUB_CACHE)
    cached_refs = (
        ("Qwen", "Qwen3.5-9B", "main", TEMPLATE_PATHS[0]),
        ("allenai", "tmax-9b", "step_500", TEMPLATE_PATHS[1]),
    )
    checked_snapshot = False
    for organization, model, revision, vendored_template in cached_refs:
        model_cache = hub_cache / f"models--{organization}--{model}"
        ref_path = model_cache / "refs" / revision
        if not ref_path.is_file():
            continue
        snapshot_template = (
            model_cache / "snapshots" / ref_path.read_text().strip() / "chat_template.jinja"
        )
        if snapshot_template.is_file():
            assert vendored_template.read_bytes() == snapshot_template.read_bytes()
            checked_snapshot = True
    if not checked_snapshot:
        pytest.skip("neither source snapshot is cached")


def test_fetch_max_model_len_reads_matching_model_from_vllm_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "another-model", "max_model_len": 1000},
                    {"id": "wanted-model", "max_model_len": 65536},
                ],
            },
        )

    client_class = httpx2.Client

    def client(**kwargs: Any) -> httpx2.Client:
        return client_class(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx2, "Client", client)

    assert fetch_max_model_len("http://synthetic.invalid/", "wanted-model") == 65536
    assert len(requests) == 1
    assert requests[0].url == "http://synthetic.invalid/v1/models"
