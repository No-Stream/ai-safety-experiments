"""Entropy shares the fp32 loss projection without changing scores or gradients."""

from typing import Any

import pytest
import torch
from trl.trainer.utils import entropy_from_logits

import reward_hacking.distill.train as distill_train
from reward_hacking.tests.test_distill_train import tiny_model


def test_entropy_reuses_loss_logits_and_preserves_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model: Any = tiny_model().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    mask = torch.tensor([[True, False, True, True, True]])
    temperature = 0.7
    backbone = model.get_base_model().base_model
    hidden = backbone(input_ids=input_ids, use_cache=False).last_hidden_state[:, -6:-1]
    head = model.get_output_embeddings()
    logits = torch.nn.functional.linear(hidden.float(), head.weight.detach().float()) / temperature
    reference_logps = (
        torch.log_softmax(logits, dim=-1)
        .gather(-1, input_ids[:, -5:].unsqueeze(-1))
        .squeeze(-1)
        .masked_fill(~mask, 0)
    )
    reference_entropy = entropy_from_logits(logits.detach()).masked_fill(~mask, 0)
    reference_logps.sum().backward()
    expected_gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    original_head = distill_train._fp32_head_logits
    projections: list[int] = []

    def counted_head(
        hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        projections.append(hidden.size(1))
        return original_head(hidden, weight, bias)

    monkeypatch.setattr(distill_train, "_fp32_head_logits", counted_head)
    logps, entropy = distill_train.chunked_completion_logps(
        model,
        input_ids,
        5,
        temperature=temperature,
        chunk_tokens=2,
        compute_entropy=True,
        loss_mask=mask,
    )
    assert projections == [2, 2]
    assert entropy is not None
    assert not entropy.requires_grad
    torch.testing.assert_close(logps, reference_logps)
    torch.testing.assert_close(entropy, reference_entropy)
    logps.sum().backward()
    assert projections == [2, 2, 2, 2]
    for name, parameter in model.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(
                parameter.grad, expected_gradients[name], atol=2e-6, rtol=2e-5
            )
