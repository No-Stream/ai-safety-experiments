"""CPU-safe tests for the shared teacher-forcing seam."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from games.teacher_forcing import (
    build_batch,
    continuation_logprob_means,
    continuation_logprob_sums,
    per_position_kl,
    teacher_force,
    teacher_force_kl,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class SyntheticTokenizer:
    """A byte tokenizer whose boundary behavior is deliberately easy to inspect."""

    pad_token_id: int | None = 0

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        return conversation[0]["content"]

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(character) + 10 for character in text]
        return ([1] if add_special_tokens else []) + ids


class BoundaryMergingTokenizer(SyntheticTokenizer):
    """A tokenizer that merges a context/continuation boundary, so the guard must refuse it."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(character) + 10 for character in text]
        if text.startswith("contextcontinuation"):
            ids = [999, *ids[1:]]
        return ([1] if add_special_tokens else []) + ids


class TinyCausalModel(nn.Module):
    """Tiny causal LM that supports the Qwen ``logits_to_keep`` interface."""

    def __init__(self, vocabulary_size: int = 256, hidden_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(hidden_size, vocabulary_size, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.arange(vocabulary_size * hidden_size, dtype=torch.bfloat16).reshape(
                    vocabulary_size, hidden_size
                )
                / 100
            )
            self.lm_head.weight.copy_(
                torch.arange(vocabulary_size * hidden_size, dtype=torch.bfloat16).reshape(
                    vocabulary_size, hidden_size
                )
                / 200
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        logits_to_keep: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden[:, logits_to_keep, :]))


def test_build_batch_right_pads_and_preserves_explicit_lengths() -> None:
    batch = build_batch(
        SyntheticTokenizer(),
        ["a", "long"],
        ["xy", "z"],
    )

    assert batch.context_lengths == (2, 5)
    assert batch.continuation_lengths == (2, 1)
    assert batch.input_ids.shape == (2, 6)
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]
    assert batch.continuation_mask.tolist() == [[True, True], [True, False]]
    assert batch.input_ids[0, -1].item() == 0


def test_build_batch_validates_saved_token_ids_before_scoring() -> None:
    tokenizer = SyntheticTokenizer()
    batch = build_batch(
        tokenizer,
        ["a"],
        ["xy"],
        context_token_ids=[[1, ord("a") + 10]],
        continuation_token_ids=[[ord("x") + 10, ord("y") + 10]],
    )
    assert batch.continuation_lengths == (2,)
    with pytest.raises(ValueError, match="saved context token ids"):
        build_batch(tokenizer, ["a"], ["xy"], context_token_ids=[[999]])


def test_teacher_force_returns_bfloat16_execution_as_cpu_float32_logits() -> None:
    tokenizer = SyntheticTokenizer()
    batch = build_batch(tokenizer, ["a", "long"], ["xy", "z"])
    result = teacher_force(TinyCausalModel(), batch)

    assert result.context_lengths == batch.context_lengths
    assert result.continuation_lengths == batch.continuation_lengths
    assert tuple(logits.shape[0] for logits in result.logits) == (2, 1)
    assert tuple(logits.dtype for logits in result.logits) == (torch.float32, torch.float32)
    assert all(logits.device.type == "cpu" for logits in result.logits)
    assert all(
        logprobs.shape == (length,)
        for logprobs, length in zip(
            result.continuation_logprobs, batch.continuation_lengths, strict=True
        )
    )
    assert all(torch.isfinite(logprobs).all() for logprobs in result.continuation_logprobs)


def test_same_logits_have_zero_kl_and_logprob_sums_are_exposed() -> None:
    batch = build_batch(SyntheticTokenizer(), ["a"], ["xy"])
    result = teacher_force(TinyCausalModel(), batch)

    kl = per_position_kl(result, result)

    assert kl[0].shape == (2,)
    assert torch.allclose(kl[0], torch.zeros(2), atol=1e-6)
    assert continuation_logprob_sums(result)[0] == pytest.approx(
        float(result.continuation_logprobs[0].sum())
    )
    assert continuation_logprob_means(result)[0] == pytest.approx(
        float(result.continuation_logprobs[0].mean())
    )


def test_retokenisation_boundary_is_load_bearing() -> None:
    with pytest.raises(ValueError, match="re-tokenise cleanly"):
        build_batch(BoundaryMergingTokenizer(), ["context"], ["continuation"])


def test_exact_continuation_token_ids_bypass_decoded_text_boundary() -> None:
    tokenizer = BoundaryMergingTokenizer()
    continuation_ids = tokenizer.encode("continuation", add_special_tokens=False)
    continuation_ids.append(777)

    batch = build_batch(
        tokenizer,
        ["context"],
        ["continuation"],
        continuation_token_ids=[continuation_ids],
    )

    assert batch.input_ids[0].tolist() == tokenizer.encode("context") + continuation_ids
    assert batch.continuation_ids[0].tolist() == continuation_ids


def test_kl_refuses_different_token_lengths() -> None:
    one_token = teacher_force(TinyCausalModel(), build_batch(SyntheticTokenizer(), ["a"], ["x"]))
    two_tokens = teacher_force(TinyCausalModel(), build_batch(SyntheticTokenizer(), ["a"], ["xy"]))

    with pytest.raises(ValueError, match="continuation lengths"):
        per_position_kl(one_token, two_tokens)


def test_kl_refuses_different_token_ids_with_equal_lengths() -> None:
    first = teacher_force(TinyCausalModel(), build_batch(SyntheticTokenizer(), ["a"], ["x"]))
    second = teacher_force(TinyCausalModel(), build_batch(SyntheticTokenizer(), ["a"], ["y"]))

    with pytest.raises(ValueError, match="token ids"):
        per_position_kl(first, second)


def test_chunked_kl_matches_full_reference_across_ragged_chunk_boundaries() -> None:
    tokenizer = SyntheticTokenizer()
    batch = build_batch(
        tokenizer,
        ["a", "long", "a"],
        ["xyzw", "mnopqr", "uv"],
    )
    reference_model = TinyCausalModel()
    comparison_model = TinyCausalModel()
    with torch.no_grad():
        comparison_model.lm_head.weight.add_(
            torch.arange(comparison_model.lm_head.out_features, dtype=torch.bfloat16).unsqueeze(1)
            / 1000
        )

    reference = teacher_force(reference_model, batch)
    comparison = teacher_force(comparison_model, batch)
    expected = per_position_kl(reference, comparison)
    reduced = teacher_force_kl(
        reference_model,
        comparison_model,
        batch,
        position_chunk_size=2,
        top_count=3,
        row_group_size=2,
    )

    for actual, target in zip(reduced.kl, expected, strict=True):
        assert torch.allclose(actual, target, atol=1e-5, rtol=1e-5)
    assert all(
        kl.shape == (length,)
        for kl, length in zip(reduced.kl, batch.continuation_lengths, strict=True)
    )
    assert all(kl.dtype == torch.float32 and kl.device.type == "cpu" for kl in reduced.kl)
    assert all(
        top_ids.shape == (length, 3)
        for top_ids, length in zip(
            reduced.base_top_token_ids, batch.continuation_lengths, strict=True
        )
    )


def test_chunked_kl_result_does_not_retain_full_vocabulary_logits() -> None:
    batch = build_batch(SyntheticTokenizer(), ["a"], ["xyz"])
    result = teacher_force_kl(TinyCausalModel(), TinyCausalModel(), batch, position_chunk_size=1)

    assert not hasattr(result, "logits")
    assert result.kl[0].shape == (3,)
    assert result.base_top_logits[0].shape == (3, 3)
    assert result.comparison_top_logits[0].shape == (3, 3)
    assert result.base_top_logits[0].dtype == torch.float32
    assert result.comparison_top_logits[0].dtype == torch.float32
    assert result.base_top_logits[0].device.type == "cpu"
    assert result.comparison_top_logits[0].device.type == "cpu"
