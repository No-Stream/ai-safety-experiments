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
    render_assistant_terminator_suffix,
    render_empty_assistant_completion,
    render_prompt,
    render_prompt_continuation,
    strip_reasoning,
    template_identity,
    validate_chat_template_kwargs,
)
from reward_hacking.ladder.loop import _PromptRenderState, _render_prompt_for_history
from reward_hacking.ladder.prompt_trace import reconstruct_rendered_prompts

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


@pytest.fixture
def reasoning_effort_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"fixture": 0})))
    tokenizer.chat_template = (
        "{% if reasoning_effort|default('xhigh') == 'xhigh' %}"
        "Reasoning effort is set to xhigh. {% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "<assistant-prefix>{{ message.get('reasoning_content', '') }}{{ message['content'] }}"
        "</assistant-end>"
        "{% else %}"
        "<{{ message['role'] }}>{{ message['content'] }}</{{ message['role'] }}>"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<assistant-prefix>{% endif %}"
    )
    return tokenizer


class _ChatTemplateKwargSpy:
    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self._tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template
        self.reasoning_efforts: list[object] = []

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        self.reasoning_efforts.append(kwargs.get("reasoning_effort"))
        return self._tokenizer.apply_chat_template(*args, **kwargs)


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


def _render_history_prompt(
    tokenizer: PreTrainedTokenizerFast,
    messages: list[dict[str, Any]],
    prompt_state: _PromptRenderState,
) -> str:
    return _render_prompt_for_history(tokenizer, messages, TEST_TOOLS, prompt_state)


def _prompt_trace_turn_record(
    turn: int,
    rendered_prompt: str,
    previous_prompt: str | None,
    *,
    raw_completion: str = "",
    runaway: bool = False,
) -> dict[str, object]:
    is_full = previous_prompt is None
    if previous_prompt is not None and not rendered_prompt.startswith(previous_prompt):
        raise ValueError("synthetic retained prompt must extend its previous prompt")
    prompt_text = rendered_prompt if is_full else rendered_prompt[len(previous_prompt) :]
    return {
        "record": "turn",
        "turn": turn,
        "raw_completion": raw_completion,
        "runaway": runaway,
        "rendered_prompt_text": prompt_text,
        "rendered_prompt_is_full": is_full,
        "rendered_prompt_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
    }


def _prompt_trace_summary_record(
    rendered_prompt: str,
    previous_prompt: str | None,
) -> dict[str, object]:
    is_full = previous_prompt is None
    if previous_prompt is not None and not rendered_prompt.startswith(previous_prompt):
        raise ValueError("synthetic retained report prompt must extend its previous prompt")
    prompt_text = rendered_prompt if is_full else rendered_prompt[len(previous_prompt) :]
    return {
        "record": "episode_summary",
        "final_report_prompt_text": prompt_text,
        "final_report_prompt_is_full": is_full,
        "final_report_prompt_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
    }


def _assert_prompt_trace_round_trips(
    trace_records: list[dict[str, object]],
    expected_prompts: list[str],
) -> None:
    assert reconstruct_rendered_prompts(trace_records) == tuple(expected_prompts)

    corrupted_records = [dict(record) for record in trace_records]
    corrupted_records[2]["rendered_prompt_text"] = "corrupted appended prompt text"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        reconstruct_rendered_prompts(corrupted_records)


def _append_tool_turn(
    tokenizer: PreTrainedTokenizerFast,
    messages: list[dict[str, Any]],
    previous_prompt: str,
    assistant_message: dict[str, Any],
    tool_result: dict[str, str],
) -> str:
    closed_turn = _render_closed_turn(tokenizer, [*messages, assistant_message])
    assert closed_turn.startswith(previous_prompt)
    completion = closed_turn[len(previous_prompt) :]
    messages.extend([assistant_message, tool_result])
    prompt = _render_history_prompt(
        tokenizer,
        messages,
        _PromptRenderState("retain", previous_prompt, completion, (tool_result,)),
    )
    assert is_pure_append(previous_prompt, completion, prompt)
    assert prompt == render_prompt(tokenizer, messages, TEST_TOOLS)
    return prompt


def _assistant_completion(
    tokenizer: PreTrainedTokenizerFast,
    messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
) -> str:
    generation_prefix = render_prompt(tokenizer, messages, TEST_TOOLS)
    completed_turn = _render_closed_turn(tokenizer, [*messages, assistant_message])
    assert completed_turn.startswith(generation_prefix)
    return completed_turn[len(generation_prefix) :]


def _assert_reasoning_is_stripped(
    tokenizer: PreTrainedTokenizerFast,
    messages: list[dict[str, Any]],
    prior_reasoning: list[str],
    runaway_assistant_position: int,
) -> None:
    for history in (messages[:4], messages[:6], messages[:8], messages[:10], messages):
        runaway_positions = (
            (runaway_assistant_position,) if len(history) > runaway_assistant_position else ()
        )
        stripped_prompt = _render_history_prompt(
            tokenizer,
            history,
            _PromptRenderState("strip", runaway_message_positions=runaway_positions),
        )
        assert all(reasoning not in stripped_prompt for reasoning in prior_reasoning)
        if runaway_positions:
            messages_without_runaway = [
                message
                for index, message in enumerate(history)
                if index != runaway_assistant_position
            ]
            assert stripped_prompt == render_prompt(
                tokenizer,
                strip_reasoning(messages_without_runaway),
                TEST_TOOLS,
            )


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


def test_reasoning_effort_kwarg_controls_rendering_and_continuation(
    reasoning_effort_tokenizer: PreTrainedTokenizerFast,
) -> None:
    chat_template_kwargs = {"reasoning_effort": "medium"}
    messages = _messages()
    tokenizer = cast("Any", _ChatTemplateKwargSpy(reasoning_effort_tokenizer))

    rendered = render_prompt(
        tokenizer,
        messages,
        (),
        chat_template_kwargs=chat_template_kwargs,
    )
    anchor = {"role": "user", "content": ""}
    appended_messages = [{"role": "user", "content": "appended"}]
    anchored_prompt = render_prompt(
        tokenizer,
        [anchor, *appended_messages],
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
    continuation = render_prompt_continuation(
        tokenizer,
        appended_messages,
        chat_template_kwargs=chat_template_kwargs,
    )
    empty_assistant_completion = render_empty_assistant_completion(
        tokenizer,
        messages,
        (),
        chat_template_kwargs=chat_template_kwargs,
    )
    assistant_terminator = render_assistant_terminator_suffix(
        tokenizer,
        messages,
        (),
        chat_template_kwargs=chat_template_kwargs,
    )

    assert "Reasoning effort is set to xhigh." not in rendered
    assert "Reasoning effort is set to xhigh." not in anchored_prompt
    assert anchored_prompt == rendered_anchor + continuation
    assert empty_assistant_completion == "</assistant-end>"
    assert assistant_terminator == "</assistant-end>"
    assert tokenizer.reasoning_efforts
    assert set(tokenizer.reasoning_efforts) == {"medium"}


def test_reasoning_effort_template_guard_requires_explicit_value(
    reasoning_effort_tokenizer: PreTrainedTokenizerFast,
) -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        validate_chat_template_kwargs(
            reasoning_effort_tokenizer,
            Path(),
            model_id="synthetic-qwen",
            chat_template_kwargs={},
        )

    validate_chat_template_kwargs(
        reasoning_effort_tokenizer,
        Path(),
        model_id="synthetic-qwen",
        chat_template_kwargs={"reasoning_effort": "medium"},
    )


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


def test_reasoning_history_is_retained_through_turns_runaway_nudge_and_report(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    messages = _messages()
    prompt = _render_history_prompt(tokenizer, messages, _PromptRenderState("retain"))
    assert render_assistant_terminator_suffix(tokenizer, messages, TEST_TOOLS) == "<|im_end|>\n"
    prior_reasoning: list[str] = []

    first_assistant = _assistant_call("first retained reasoning", "one")
    first_tool_result = {"role": "tool", "content": "one"}
    prior_reasoning.append("first retained reasoning")
    prompt = _append_tool_turn(tokenizer, messages, prompt, first_assistant, first_tool_result)
    assert all(reasoning in prompt for reasoning in prior_reasoning)

    second_assistant = _assistant_call("second retained reasoning", "two")
    second_tool_result = {"role": "tool", "content": "two"}
    prior_reasoning.append("second retained reasoning")
    prompt = _append_tool_turn(tokenizer, messages, prompt, second_assistant, second_tool_result)
    assert all(reasoning in prompt for reasoning in prior_reasoning)

    empty_runaway_assistant = {
        "role": "assistant",
        "reasoning_content": "",
        "content": "",
        "tool_calls": [],
    }
    runaway_user_message = {"role": "user", "content": "synthetic runaway format error"}
    runaway_assistant_position = len(messages)
    empty_assistant_suffix = render_empty_assistant_completion(
        tokenizer,
        messages,
        TEST_TOOLS,
    )
    runaway_suffix_messages = [empty_runaway_assistant, runaway_user_message]
    messages.extend(runaway_suffix_messages)
    runaway_prompt = _render_history_prompt(
        tokenizer,
        messages,
        _PromptRenderState("retain", prompt, empty_assistant_suffix, (runaway_user_message,)),
    )
    assert is_pure_append(prompt, empty_assistant_suffix, runaway_prompt)
    assert runaway_prompt == (
        prompt
        + empty_assistant_suffix
        + render_prompt_continuation(tokenizer, [runaway_user_message])
    )
    assert "RUNAWAY_TEXT_MUST_NOT_REENTER_HISTORY" not in runaway_prompt
    assert all(reasoning in runaway_prompt for reasoning in prior_reasoning)
    assert empty_assistant_suffix == "\n</think>\n\n<|im_end|>\n"
    assert empty_assistant_suffix.endswith("<|im_end|>\n")
    assert empty_assistant_suffix.count("<|im_end|>") == 1
    prompt = runaway_prompt

    empty_start_assistant = {
        "role": "assistant",
        "reasoning_content": "empty-start retained reasoning",
        "content": "answer",
        "tool_calls": [],
    }
    empty_start_nudge = {"role": "user", "content": "synthetic empty-start nudge"}
    messages.extend([empty_start_assistant, empty_start_nudge])
    empty_start_completion = _assistant_completion(tokenizer, messages[:-2], empty_start_assistant)
    prior_reasoning.append("empty-start retained reasoning")
    next_prompt = _render_history_prompt(
        tokenizer,
        messages,
        _PromptRenderState("retain", prompt, empty_start_completion, (empty_start_nudge,)),
    )
    assert is_pure_append(prompt, empty_start_completion, next_prompt)
    assert all(reasoning in next_prompt for reasoning in prior_reasoning)
    prompt = next_prompt

    final_turn_assistant = {
        "role": "assistant",
        "reasoning_content": "final-turn retained reasoning",
        "content": "done",
        "tool_calls": [],
    }
    report_request = {"role": "user", "content": "fixture report request"}
    messages.extend([final_turn_assistant, report_request])
    final_turn_completion = _assistant_completion(tokenizer, messages[:-2], final_turn_assistant)
    prior_reasoning.append("final-turn retained reasoning")
    report_prompt = _render_history_prompt(
        tokenizer,
        messages,
        _PromptRenderState("retain", prompt, final_turn_completion, (report_request,)),
    )
    assert is_pure_append(prompt, final_turn_completion, report_prompt)
    assert all(reasoning in report_prompt for reasoning in prior_reasoning)
    assert "RUNAWAY_TEXT_MUST_NOT_REENTER_HISTORY" not in report_prompt

    _assert_reasoning_is_stripped(
        tokenizer,
        messages,
        prior_reasoning,
        runaway_assistant_position,
    )


def test_prompt_trace_reconstructs_retained_turns_runaway_nudge_and_report(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    messages = _messages()
    previous_prompt = _render_history_prompt(tokenizer, messages, _PromptRenderState("retain"))
    expected_prompts = [previous_prompt]
    trace_records = [_prompt_trace_turn_record(0, previous_prompt, None)]

    first_assistant = _assistant_call("first reasoning", "one")
    previous_prompt = _append_tool_turn(
        tokenizer, messages, previous_prompt, first_assistant, {"role": "tool", "content": "one"}
    )
    expected_prompts.append(previous_prompt)
    trace_records.append(_prompt_trace_turn_record(1, previous_prompt, expected_prompts[-2]))

    second_assistant = _assistant_call("second reasoning", "two")
    previous_prompt = _append_tool_turn(
        tokenizer, messages, previous_prompt, second_assistant, {"role": "tool", "content": "two"}
    )
    expected_prompts.append(previous_prompt)
    trace_records.append(
        _prompt_trace_turn_record(
            2,
            previous_prompt,
            expected_prompts[-2],
            raw_completion="RUNAWAY_TEXT_MUST_NOT_REENTER_HISTORY and distinctive continuation",
            runaway=True,
        )
    )

    empty_runaway_assistant = {
        "role": "assistant",
        "reasoning_content": "",
        "content": "",
        "tool_calls": [],
    }
    runaway_user_message = {"role": "user", "content": "synthetic runaway format error"}
    empty_assistant_suffix = render_empty_assistant_completion(tokenizer, messages, TEST_TOOLS)
    messages.extend([empty_runaway_assistant, runaway_user_message])
    previous_prompt = _render_history_prompt(
        tokenizer,
        messages,
        _PromptRenderState(
            "retain", expected_prompts[-1], empty_assistant_suffix, (runaway_user_message,)
        ),
    )
    expected_prompts.append(previous_prompt)
    trace_records.append(_prompt_trace_turn_record(3, previous_prompt, expected_prompts[-2]))

    empty_start_assistant = {
        "role": "assistant",
        "reasoning_content": "empty-start reasoning",
        "content": "answer",
        "tool_calls": [],
    }
    empty_start_nudge = {"role": "user", "content": "synthetic empty-start nudge"}
    messages.extend([empty_start_assistant, empty_start_nudge])
    empty_start_completion = _assistant_completion(tokenizer, messages[:-2], empty_start_assistant)
    previous_prompt = _render_history_prompt(
        tokenizer,
        messages,
        _PromptRenderState(
            "retain", expected_prompts[-1], empty_start_completion, (empty_start_nudge,)
        ),
    )
    expected_prompts.append(previous_prompt)
    trace_records.append(_prompt_trace_turn_record(4, previous_prompt, expected_prompts[-2]))

    final_assistant = {
        "role": "assistant",
        "reasoning_content": "final reasoning",
        "content": "done",
        "tool_calls": [],
    }
    report_request = {"role": "user", "content": "fixture report request"}
    messages.extend([final_assistant, report_request])
    final_completion = _assistant_completion(tokenizer, messages[:-2], final_assistant)
    report_prompt = _render_history_prompt(
        tokenizer,
        messages,
        _PromptRenderState("retain", previous_prompt, final_completion, (report_request,)),
    )
    expected_prompts.append(report_prompt)
    trace_records.append(_prompt_trace_summary_record(report_prompt, previous_prompt))

    _assert_prompt_trace_round_trips(trace_records, expected_prompts)


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
