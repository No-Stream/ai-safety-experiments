"""CPU checks for bounded teacher-forced residual-intervention parity."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from itertools import pairwise
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from scripts.validate_vllm_residual_interventions import (
    MetricAccumulator,
    PromptHandoff,
    _assert_sabotage_separated,
    _decode_prompt_handoff,
    _encode_prompt_handoff,
    _hf_prompt_logprobs,
    _vllm_prompt_logprobs,
    sample_positions,
    topk_logprobs,
)

if TYPE_CHECKING:
    import vllm


class TestPositionSampler:
    def test_digest_controls_bounded_evenly_spaced_positions_with_last_included(self) -> None:
        first_digest = hashlib.sha256(b"synthetic first").hexdigest()
        second_digest = hashlib.sha256(b"synthetic second").hexdigest()
        first = sample_positions(2486, first_digest, count=32)

        assert first == sample_positions(2486, first_digest, count=32)
        assert first != sample_positions(2486, second_digest, count=32)
        assert len(first) == 32
        assert first[-1] == 2485
        assert first == tuple(sorted(set(first)))
        assert max(b - a for a, b in pairwise(first)) < 160
        assert sample_positions(4, first_digest, count=32) == (1, 2, 3)


class TestTopKMetrics:
    def test_identical_distribution_is_perfect(self) -> None:
        reference = torch.log_softmax(torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]), dim=0)
        metrics = MetricAccumulator(k=3)
        metrics.add(reference, topk_logprobs(reference, 3))

        assert metrics.payload() == pytest.approx(
            {
                "n_positions": 1,
                "top1_agreement": 1.0,
                "overlap_at_k": 1.0,
                "mean_abs_logprob_delta": 0.0,
                "max_abs_logprob_delta": 0.0,
                "mean_truncated_kl": 0.0,
            },
            abs=1e-6,
        )

    def test_perturbed_distribution_degrades_overlap_and_logprobs(self) -> None:
        reference = torch.log_softmax(torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]), dim=0)
        perturbed = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]), dim=0)
        metrics = MetricAccumulator(k=3)
        metrics.add(reference, topk_logprobs(perturbed, 3))
        payload = metrics.payload()

        assert payload["top1_agreement"] == 0.0
        assert payload["overlap_at_k"] == pytest.approx(1 / 3)
        assert payload["mean_abs_logprob_delta"] > 1.0
        assert payload["mean_truncated_kl"] > 0.1

    def test_wrong_layer_like_shift_fails_against_noise_floor(self) -> None:
        reference = torch.log_softmax(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]), dim=0)
        noise = MetricAccumulator(k=2)
        noise.add(reference, topk_logprobs(reference, 2))
        shifted = torch.log_softmax(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), dim=0)
        sabotage = MetricAccumulator(k=2)
        sabotage.add(reference, topk_logprobs(shifted, 2))

        _assert_sabotage_separated(sabotage.payload(), noise.payload(), name="wrong layer")
        with pytest.raises(RuntimeError, match="did not produce a clear parity failure"):
            _assert_sabotage_separated(noise.payload(), noise.payload(), name="wrong layer")


def test_handoff_stays_under_128_kib_per_long_prompt() -> None:
    digest = hashlib.sha256(b"synthetic long prompt").hexdigest()
    positions = sample_positions(2486, digest, count=32)
    row = {200_000 + token_id: -float(token_id) / 7 for token_id in range(21)}
    handoff = PromptHandoff(
        token_ids_sha256=digest,
        token_count=2486,
        positions=positions,
        hooked=tuple(row for _ in positions),
        unhooked=tuple(row for _ in positions),
        wrong_layer=tuple(row for _ in positions),
    )
    serialized = _encode_prompt_handoff(handoff)
    assert len(serialized) < 128 * 1024
    assert _decode_prompt_handoff(serialized) == handoff


def test_vllm_keeps_only_selected_topk_and_excludes_extra_prompt_token() -> None:
    class FakeLLM:
        def generate(
            self, prompts: list[str], sampling: vllm.SamplingParams, *, lora_request: object
        ) -> list[SimpleNamespace]:
            assert prompts == ["synthetic"]
            assert sampling.prompt_logprobs == 20
            assert lora_request is adapter
            topk = {
                token_id: SimpleNamespace(rank=token_id + 1, logprob=-float(token_id))
                for token_id in range(20)
            }
            topk[24] = SimpleNamespace(rank=25, logprob=-30.0)
            return [
                SimpleNamespace(
                    prompt_token_ids=(1, 2, 3, 4, 24),
                    prompt_logprobs=[None, topk, topk, topk, topk],
                )
            ]

    adapter = object()
    selected = _vllm_prompt_logprobs(
        FakeLLM(),
        "synthetic",
        expected_token_ids=(1, 2, 3, 4, 24),
        positions=(2, 4),
        lora_request=adapter,
    )
    assert len(selected) == 2
    assert all(set(row) == set(range(20)) for row in selected)


def test_hf_forward_requests_only_sampled_logits_on_cpu() -> None:
    class FakeTokenizer:
        def __call__(self, _rendered: str, **_kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
            }

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, *, logits_to_keep: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
            assert logits_to_keep.tolist() == [1, 3]
            return SimpleNamespace(logits=torch.zeros((1, 2, 25)))

    result = _hf_prompt_logprobs(
        FakeModel(),
        FakeTokenizer(),
        "synthetic",
        positions=(2, 4),
        intervention_context=nullcontext(),
    )
    assert result.token_ids == (1, 2, 3, 4, 5)
    assert len(result.logprobs) == 2
    assert all(row.shape == (25,) for row in result.logprobs)
