"""Frozen fp32 head storage survives repeated scoring and module transfers."""

import pytest
import torch

from reward_hacking.distill.train import _copy_fp32_head_weights


def _head() -> torch.nn.Linear:
    head = torch.nn.Linear(5, 7, bias=True, dtype=torch.bfloat16)
    head.requires_grad_(False)
    return head


def test_frozen_head_reuses_nonpersistent_fp32_buffers() -> None:
    head = _head()
    weight, bias = _copy_fp32_head_weights(head)
    reused_weight, reused_bias = _copy_fp32_head_weights(head)
    assert reused_weight is weight
    assert reused_bias is bias
    assert weight.dtype == torch.float32
    assert not weight.requires_grad
    assert bias is not None
    assert bias.dtype == torch.float32
    assert {id(value) for value in head.buffers()} == {id(weight), id(bias)}
    assert set(head.state_dict()) == {"weight", "bias"}
    torch.testing.assert_close(weight, head.weight.float(), rtol=0, atol=0)


def test_frozen_head_refreshes_after_loading_different_base_weights() -> None:
    head = _head()
    previous_weight, _ = _copy_fp32_head_weights(head)
    replacement = _head()
    head.load_state_dict(replacement.state_dict())
    weight, _ = _copy_fp32_head_weights(head)
    assert weight is not previous_weight
    torch.testing.assert_close(weight, replacement.weight.float(), rtol=0, atol=0)


def test_frozen_head_refreshes_after_module_dtype_conversion() -> None:
    head = _head()
    _copy_fp32_head_weights(head)
    head.to(dtype=torch.float64)
    weight, bias = _copy_fp32_head_weights(head)
    assert weight.dtype == torch.float32
    assert bias is not None
    assert bias.dtype == torch.float32
    torch.testing.assert_close(weight, head.weight.float(), rtol=0, atol=0)


def test_trainable_head_is_rejected_instead_of_silently_detached() -> None:
    with pytest.raises(ValueError, match="frozen"):
        _copy_fp32_head_weights(torch.nn.Linear(5, 7))


def test_cache_created_during_inference_can_be_reused_for_training() -> None:
    head = _head()
    with torch.inference_mode():
        weight, bias = _copy_fp32_head_weights(head)
    hidden = torch.randn(2, 5, requires_grad=True)
    torch.nn.functional.linear(hidden, weight, bias).sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_inference_accepts_trainable_head_without_caching_mutable_weights() -> None:
    head = torch.nn.Linear(5, 7, dtype=torch.bfloat16)
    with torch.no_grad():
        weight, _ = _copy_fp32_head_weights(head)
        head.weight.add_(1)
        refreshed_weight, _ = _copy_fp32_head_weights(head)
    assert not list(head.buffers())
    assert not torch.equal(weight, refreshed_weight)
    torch.testing.assert_close(refreshed_weight, head.weight.float(), rtol=0, atol=0)
