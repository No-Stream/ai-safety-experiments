"""CPU regression checks for the frozen-head Liger GRPO loss wrapper."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import torch
from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss

from grpo.liger_frozen_head import FrozenHeadLigerGRPOLoss

if TYPE_CHECKING:
    import pytest


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
