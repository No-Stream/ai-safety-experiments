"""CPU checks for distillation head precision and gradient flow."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from reward_hacking.distill.train import (
    _copy_fp32_head_weights,
    _fp32_head_logits,
    chunked_completion_logps,
)


class _TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 5, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _TinyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = _TinyBackbone()
        self.lm_head = torch.nn.Linear(5, 7, bias=False).to(dtype=torch.bfloat16)
        self.lm_head.weight.requires_grad_(False)

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self.lm_head


def test_fp32_head_logits_keep_matmul_precision() -> None:
    torch.manual_seed(41)
    head = torch.nn.Linear(5, 7, bias=True).to(dtype=torch.bfloat16)
    hidden_states = torch.randn(3, 5, dtype=torch.bfloat16)
    weight, bias = _copy_fp32_head_weights(head)

    logits = _fp32_head_logits(hidden_states, weight, bias)
    expected = torch.nn.functional.linear(
        hidden_states.float(), head.weight.float(), head.bias.float()
    )
    rounded_logits = head(hidden_states).float()

    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected)
    assert not torch.equal(logits, rounded_logits)


def test_fp32_head_logits_pass_gradcheck_for_hidden_states() -> None:
    torch.manual_seed(42)
    head = torch.nn.Linear(3, 4, bias=False).to(dtype=torch.bfloat16)
    weight, bias = _copy_fp32_head_weights(head)
    hidden_states = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda values: _fp32_head_logits(values, weight, bias),
        (hidden_states,),
        eps=1e-3,
        atol=2e-3,
        rtol=2e-3,
        fast_mode=True,
    )


def test_chunked_completion_scores_fp32_head_and_backpropagates_to_hidden_states() -> None:
    torch.manual_seed(43)
    model = _TinyCausalModel()
    input_ids = torch.tensor([[0, 1, 2, 3]])
    completion_tokens = 2
    hidden_states = model.base_model(
        input_ids, attention_mask=None, use_cache=False
    ).last_hidden_state[:, -completion_tokens - 1 : -1]
    expected_logits = torch.nn.functional.linear(
        hidden_states.float(), model.lm_head.weight.detach().float()
    )
    expected_logps = torch.log_softmax(expected_logits, dim=-1).gather(
        -1, input_ids[:, -completion_tokens:].unsqueeze(-1)
    )

    logps, _ = chunked_completion_logps(
        model,
        input_ids,
        completion_tokens,
        temperature=1.0,
        chunk_tokens=1,
        compute_entropy=False,
    )

    assert logps.dtype == torch.float32
    torch.testing.assert_close(logps.unsqueeze(-1), expected_logps)
    rounded_logps = (
        torch.log_softmax(model.lm_head(hidden_states).float(), dim=-1)
        .gather(-1, input_ids[:, -completion_tokens:].unsqueeze(-1))
        .squeeze(-1)
    )
    assert not torch.equal(logps, rounded_logps)
    logps.sum().backward()
    assert model.base_model.embedding.weight.grad is not None
    assert torch.isfinite(model.base_model.embedding.weight.grad).all()
    assert model.lm_head.weight.grad is None
