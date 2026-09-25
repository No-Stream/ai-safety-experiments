"""CPU regression checks for the frozen-head Liger GRPO loss wrapper."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pytest
import torch
from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss
from trl.trainer.utils import selective_log_softmax

from games import train as gt
from grpo.liger_frozen_head import FrozenHeadLigerGRPOLoss


def test_frozen_head_wrapper_matches_liger_without_vocab_weight_grad_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    head = torch.nn.Linear(3, 7, bias=False)
    head.weight.requires_grad_(False)
    stock_adapter = torch.nn.Linear(3, 3, bias=False)
    patched_adapter = deepcopy(stock_adapter)
    inputs = torch.randn(2, 3, 3)
    token_ids = torch.tensor([[1, 2, 3], [2, 3, 4]])
    mask = torch.ones(2, 3)
    advantages = torch.tensor([0.5, -0.25])
    stock_input = stock_adapter(inputs)
    stock_input.retain_grad()
    stock = LigerFusedLinearGRPOLoss(
        compiled=False, use_ref_model=False, loss_type="dr_grpo", max_completion_length=3
    )
    stock_loss, _ = stock(stock_input, head.weight, token_ids, mask, advantages)
    stock_loss.backward()
    stock_input_grad = stock_input.grad.detach().clone()

    patched_input = patched_adapter(inputs)
    patched_input.retain_grad()
    allocations: list[tuple[int, ...]] = []
    original_zeros_like = torch.zeros_like
    original_zeros = torch.zeros

    def record_zeros_like(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        allocations.append(tuple(tensor.shape))
        return original_zeros_like(tensor, *args, **kwargs)

    def record_zeros(size: tuple[int, ...], **kwargs: object) -> torch.Tensor:
        allocations.append(size)
        return cast("Any", original_zeros)(size, **kwargs)

    monkeypatch.setattr(torch, "zeros_like", record_zeros_like)
    monkeypatch.setattr(torch, "zeros", record_zeros)
    patched = FrozenHeadLigerGRPOLoss(stock)
    patched_loss, _ = patched(patched_input, head.weight, token_ids, mask, advantages)
    patched_loss.backward()

    assert torch.allclose(patched_loss, stock_loss)
    assert torch.allclose(patched_input.grad, stock_input_grad)
    assert stock_adapter.weight.grad is not None
    assert patched_adapter.weight.grad is not None
    assert torch.allclose(patched_adapter.weight.grad, stock_adapter.weight.grad)
    assert tuple(head.weight.shape) not in allocations


def test_single_forward_vllm_is_matches_two_pass_fused_loss_and_gradient() -> None:
    """The production fused path may replace the detached old-policy pass with its own logps."""
    torch.manual_seed(23)
    head = torch.nn.Linear(4, 9, bias=False)
    head.weight.requires_grad_(False)
    reference_adapter = torch.nn.Linear(4, 4, bias=False)
    optimized_adapter = deepcopy(reference_adapter)
    inputs = torch.randn(2, 5, 4)
    token_ids = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.float32)
    advantages = torch.tensor([0.5, -0.25])
    stock = LigerFusedLinearGRPOLoss(
        compiled=False,
        use_ref_model=False,
        loss_type="dr_grpo",
        max_completion_length=5,
    )
    reference_hidden = reference_adapter(inputs)
    with torch.no_grad():
        old_per_token_logps = selective_log_softmax(
            head(reference_hidden) / stock.temperature, token_ids
        )
    sampling_per_token_logps = old_per_token_logps - torch.tensor(
        [[0.0, 0.1, -0.2, 0.3, -0.4], [0.2, -0.1, 0.4, 0.0, 0.0]]
    )
    expected_ratio, expected_metrics = gt.compute_vllm_importance_sampling(
        old_per_token_logps,
        sampling_per_token_logps,
        mask,
        mode="token_truncate",
        clip_min=0.5,
        clip_max=2.0,
    )
    reference_loss_module = FrozenHeadLigerGRPOLoss(stock)
    reference_loss, _ = reference_loss_module(
        reference_hidden,
        head.weight,
        token_ids,
        mask,
        advantages,
        old_per_token_logps=old_per_token_logps,
        vllm_is_ratio=expected_ratio,
    )
    reference_loss.backward()

    observed: dict[str, float] = {}

    def ratio_from_loss_logps(per_token_logps: torch.Tensor) -> torch.Tensor:
        ratio, metrics = gt.compute_vllm_importance_sampling(
            per_token_logps.detach(),
            sampling_per_token_logps,
            mask,
            mode="token_truncate",
            clip_min=0.5,
            clip_max=2.0,
        )
        observed.update(metrics)
        return ratio

    optimized_hidden = optimized_adapter(inputs)
    optimized_loss_module = FrozenHeadLigerGRPOLoss(stock)
    optimized_loss, _ = optimized_loss_module(
        optimized_hidden,
        head.weight,
        token_ids,
        mask,
        advantages,
        vllm_is_ratio_fn=ratio_from_loss_logps,
    )
    optimized_loss.backward()

    assert torch.allclose(optimized_loss, reference_loss, rtol=1e-7, atol=1e-8)
    assert reference_adapter.weight.grad is not None
    assert optimized_adapter.weight.grad is not None
    assert torch.allclose(
        optimized_adapter.weight.grad, reference_adapter.weight.grad, rtol=1e-6, atol=1e-7
    )
    assert observed == pytest.approx(expected_metrics)


def test_vllm_is_metrics_and_log_only_keep_the_unmodified_ratio_observable() -> None:
    trainer_logps = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    sampler_logps = torch.tensor([[0.0, 1.0, -1.0], [0.0, 2.0, 0.0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    ratio, metrics = gt.compute_vllm_importance_sampling(
        trainer_logps,
        sampler_logps,
        mask,
        mode="token_mask",
        clip_min=0.5,
        clip_max=2.0,
    )
    assert torch.equal(ratio, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]))
    assert metrics == pytest.approx(
        {
            "sampling/importance_sampling_ratio/min": 0.0,
            "sampling/importance_sampling_ratio/mean": 0.4,
            "sampling/importance_sampling_ratio/max": 1.0,
            "sampling/sampling_logp_difference/mean": 0.8,
            "sampling/sampling_logp_difference/max": 2.0,
            gt.IMPORTANCE_SAMPLING_ZERO_FRACTION_METRIC: 1.0,
        }
    )
    assert torch.equal(
        gt.importance_sampling_ratio_for_loss(ratio, log_only=True), torch.ones_like(ratio)
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("token_truncate", [[1.0, 0.5, 2.0], [1.0, 0.5, 1.0]]),
        ("token_mask", [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
        ("sequence_truncate", [[1.0], [0.5]]),
        ("sequence_mask", [[1.0], [0.0]]),
    ],
)
def test_single_forward_vllm_is_preserves_each_trl_mode(
    mode: str, expected: list[list[float]]
) -> None:
    trainer_logps = torch.zeros(2, 3)
    sampler_logps = torch.tensor([[0.0, 1.0, -1.0], [0.0, 2.0, 10.0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)

    ratio, _ = gt.compute_vllm_importance_sampling(
        trainer_logps,
        sampler_logps,
        mask,
        mode=mode,
        clip_min=0.5,
        clip_max=2.0,
    )

    assert torch.equal(ratio, torch.tensor(expected))
