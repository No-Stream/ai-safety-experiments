"""CPU checks for organism chunked completion scores and gradients."""

from typing import Any, cast

import torch
from peft import LoraConfig, get_peft_model
from torch.nn import functional
from transformers import AutoModelForCausalLM, Qwen2Config

from reward_hacking.distill.train import chunked_completion_logps


def _tiny_lora_model() -> torch.nn.Module:
    torch.manual_seed(41)
    base_model = AutoModelForCausalLM.from_config(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=False,
        )
    )
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    model.eval()
    return model


def _full_fp32_completion_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    completion_tokens: int,
    *,
    temperature: float,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    causal_model = cast("Any", model).get_base_model()
    hidden = causal_model.base_model(
        input_ids=input_ids, attention_mask=None, use_cache=False
    ).last_hidden_state
    completion_hidden = hidden[:, -completion_tokens - 1 : -1].float()
    targets = input_ids[:, -completion_tokens:]
    head = causal_model.get_output_embeddings()
    logits = functional.linear(
        completion_hidden,
        head.weight.float(),
        None if head.bias is None else head.bias.float(),
    )
    log_probabilities = functional.log_softmax(logits / temperature, dim=-1)
    completion_logps = log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return completion_logps.masked_fill(~loss_mask, 0.0)


def _loss_and_trainable_gradients(
    model: torch.nn.Module,
    logps: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    loss = -(logps[loss_mask] * loss_weights[loss_mask]).mean()
    loss.backward()
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return loss.detach(), gradients


def _score_and_gradients(  # noqa: PLR0913 - explicit numerical inputs for both scoring paths
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_weights: torch.Tensor,
    *,
    temperature: float,
    chunk_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
    model.zero_grad(set_to_none=True)
    if chunk_tokens is None:
        logps = _full_fp32_completion_logps(
            model,
            input_ids,
            loss_mask.size(1),
            temperature=temperature,
            loss_mask=loss_mask,
        )
    else:
        logps, entropies = chunked_completion_logps(
            model,
            input_ids,
            loss_mask.size(1),
            temperature=temperature,
            chunk_tokens=chunk_tokens,
            compute_entropy=False,
            loss_mask=loss_mask,
        )
        assert entropies is None
    loss, gradients = _loss_and_trainable_gradients(model, logps, loss_mask, loss_weights)
    return logps.detach(), loss, gradients


def test_chunked_logps_match_full_fp32_outputs_loss_and_all_trainable_gradients() -> None:
    model = _tiny_lora_model()
    input_ids = torch.tensor([[2, 4, 3, 9, 5, 7, 6, 8, 1]])
    loss_mask = torch.tensor([[True, False, True, False, True]])
    loss_weights = torch.tensor([[0.4, 9.0, -0.7, -8.0, 1.2]])
    temperature = 0.73

    full = _score_and_gradients(
        model,
        input_ids,
        loss_mask,
        loss_weights,
        temperature=temperature,
        chunk_tokens=None,
    )
    assert any(
        gradient is not None and bool(torch.count_nonzero(gradient))
        for gradient in full[2].values()
    )

    for chunk_tokens in (1, 2, 8):
        chunked = _score_and_gradients(
            model,
            input_ids,
            loss_mask,
            loss_weights,
            temperature=temperature,
            chunk_tokens=chunk_tokens,
        )
        torch.testing.assert_close(chunked[0], full[0], atol=1e-6, rtol=1e-6)
        assert torch.equal(chunked[0][~loss_mask], torch.zeros_like(chunked[0][~loss_mask]))
        torch.testing.assert_close(chunked[1], full[1], atol=1e-6, rtol=1e-6)
        assert chunked[2].keys() == full[2].keys()
        for name in full[2]:
            full_gradient = full[2][name]
            chunked_gradient = chunked[2][name]
            assert (full_gradient is None) == (chunked_gradient is None), name
            if full_gradient is not None and chunked_gradient is not None:
                torch.testing.assert_close(
                    chunked_gradient, full_gradient, atol=2e-6, rtol=1e-5, msg=name
                )
