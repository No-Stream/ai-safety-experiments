"""Offline chat-template and tokenizer contracts for distillation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from collections.abc import Callable

NINE_BILLION_MODEL = "Qwen/Qwen3.5-9B"
TWENTY_SEVEN_BILLION_MODEL = "RedHatAI/Qwen3.8-27B-INT4"

CONVERSATIONS: tuple[list[dict[str, str]], ...] = (
    [
        {"role": "system", "content": "You are a synthetic coding assistant."},
        {"role": "user", "content": "Implement a function that returns 17."},
    ],
    [{"role": "user", "content": "Given the synthetic input 'abc', return 'cba'."}],
    [
        {"role": "system", "content": "Follow the synthetic task exactly."},
        {"role": "user", "content": "Return the integer 23, with no explanation."},
    ],
)


def _cached_snapshot(model_id: str) -> Path | None:
    tokenizer_path = try_to_load_from_cache(model_id, "tokenizer.json")
    if not isinstance(tokenizer_path, str):
        return None
    return Path(tokenizer_path).parent


@pytest.fixture(scope="module")
def model_tokenizers() -> tuple[PreTrainedTokenizerBase, PreTrainedTokenizerBase]:
    nine_billion_path = _cached_snapshot(NINE_BILLION_MODEL)
    twenty_seven_billion_path = _cached_snapshot(TWENTY_SEVEN_BILLION_MODEL)
    if nine_billion_path is None or twenty_seven_billion_path is None:
        pytest.skip("the local 9B and 27B tokenizer snapshots are required for this offline check")
    return (
        cast(
            "PreTrainedTokenizerBase",
            AutoTokenizer.from_pretrained(nine_billion_path, local_files_only=True),
        ),
        cast(
            "PreTrainedTokenizerBase",
            AutoTokenizer.from_pretrained(twenty_seven_billion_path, local_files_only=True),
        ),
    )


def _render(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    **template_kwargs: object,
) -> str:
    render_template = cast("Callable[..., object]", tokenizer.apply_chat_template)
    rendered = render_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    assert isinstance(rendered, str)
    return rendered


def test_qwen_tokenizers_have_identical_vocabularies_and_special_tokens(
    model_tokenizers: tuple[PreTrainedTokenizerBase, PreTrainedTokenizerBase],
) -> None:
    nine_billion, twenty_seven_billion = model_tokenizers

    assert len(nine_billion) == len(twenty_seven_billion)
    assert nine_billion.get_vocab() == twenty_seven_billion.get_vocab()
    assert nine_billion.special_tokens_map == twenty_seven_billion.special_tokens_map
    assert nine_billion.all_special_ids == twenty_seven_billion.all_special_ids
    assert nine_billion.all_special_tokens == twenty_seven_billion.all_special_tokens


@pytest.mark.parametrize("messages", CONVERSATIONS)
def test_medium_reasoning_rendering_matches_the_nine_billion_template(
    model_tokenizers: tuple[PreTrainedTokenizerBase, PreTrainedTokenizerBase],
    messages: list[dict[str, str]],
) -> None:
    nine_billion, twenty_seven_billion = model_tokenizers
    nine_billion_rendered = _render(nine_billion, messages, enable_thinking=True)
    twenty_seven_billion_rendered = _render(
        twenty_seven_billion,
        messages,
        reasoning_effort="medium",
    )

    assert twenty_seven_billion_rendered == nine_billion_rendered
    nine_billion_ids = nine_billion.encode(nine_billion_rendered, add_special_tokens=False)
    twenty_seven_billion_ids = twenty_seven_billion.encode(
        twenty_seven_billion_rendered,
        add_special_tokens=False,
    )
    assert twenty_seven_billion_ids == nine_billion_ids


@pytest.mark.parametrize("messages", CONVERSATIONS)
def test_xhigh_reasoning_changes_the_prompt_prefix(
    model_tokenizers: tuple[PreTrainedTokenizerBase, PreTrainedTokenizerBase],
    messages: list[dict[str, str]],
) -> None:
    nine_billion, twenty_seven_billion = model_tokenizers
    nine_billion_rendered = _render(nine_billion, messages, enable_thinking=True)
    xhigh_rendered = _render(twenty_seven_billion, messages, reasoning_effort="xhigh")

    assert xhigh_rendered != nine_billion_rendered
