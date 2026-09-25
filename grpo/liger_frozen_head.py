"""Memory-safe Liger GRPO loss path for the ordinary frozen language-model head."""
# ruff: noqa: ANN401, D102, D107, PLR0913, PLR0917, SLF001

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from liger_kernel.chunked_loss import fused_linear_ppo
from liger_kernel.chunked_loss.grpo_loss import (
    LigerFusedLinearGRPOFunction,
    LigerFusedLinearGRPOLoss,
)


class _FrozenHeadSelectiveLogProb(torch.autograd.Function):
    """Selective log probabilities whose backward never materializes a head-weight gradient."""

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        bias: torch.Tensor | None,
        temperature: float,
    ) -> torch.Tensor:
        logprobs, log_z = fused_linear_ppo._selective_logprob_forward(  # pyright: ignore[reportPrivateUsage]
            hidden,
            weight,
            targets,
            bias,
            temperature,
            fused_linear_ppo._SELECTIVE_LOGPROB_VOCAB_CHUNK_SIZE,  # pyright: ignore[reportPrivateUsage]
        )
        ctx.save_for_backward(
            hidden, weight, targets, log_z, bias if bias is not None else hidden.new_empty(0)
        )
        ctx.temperature = temperature
        ctx.has_bias = bias is not None
        return logprobs

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, grad_logprobs: torch.Tensor, *grad_outputs: Any
    ) -> tuple[torch.Tensor, None, None, torch.Tensor | None, None]:
        del grad_outputs
        hidden, weight, targets, log_z, saved_bias = ctx.saved_tensors
        bias = saved_bias if ctx.has_bias else None
        grad_hidden = torch.zeros_like(hidden, dtype=torch.float32)
        grad_bias = (
            torch.zeros_like(bias, dtype=torch.float32)
            if bias is not None and bias.requires_grad
            else None
        )
        chunk = fused_linear_ppo._SELECTIVE_LOGPROB_VOCAB_CHUNK_SIZE  # pyright: ignore[reportPrivateUsage]
        sequence_chunk = fused_linear_ppo._SELECTIVE_LOGPROB_SEQ_CHUNK_SIZE  # pyright: ignore[reportPrivateUsage]
        for sequence_start in range(0, hidden.size(0), sequence_chunk):
            sequence_end = min(sequence_start + sequence_chunk, hidden.size(0))
            hidden_chunk = hidden[sequence_start:sequence_end]
            targets_chunk = targets[sequence_start:sequence_end]
            log_z_chunk = log_z[sequence_start:sequence_end]
            grad_chunk = grad_logprobs[sequence_start:sequence_end].float()
            rows = torch.arange(hidden_chunk.size(0), device=hidden.device)
            for start in range(0, weight.size(0), chunk):
                end = min(start + chunk, weight.size(0))
                logits = (hidden_chunk @ weight[start:end].to(hidden.dtype).t()).float()
                if bias is not None:
                    logits.add_(bias[start:end].float())
                logits.mul_(1.0 / ctx.temperature)
                grad_logits = (-grad_chunk).unsqueeze(-1) * torch.exp(
                    logits - log_z_chunk.unsqueeze(-1)
                )
                in_chunk = (targets_chunk >= start) & (targets_chunk < end)
                local = torch.clamp(targets_chunk - start, 0, end - start - 1)
                grad_logits[rows, local] += grad_chunk * in_chunk
                grad_logits.mul_(1.0 / ctx.temperature)
                grad_hidden[sequence_start:sequence_end].add_(
                    grad_logits @ weight[start:end].float()
                )
                if grad_bias is not None:
                    grad_bias[start:end].add_(grad_logits.sum(0))
        bias_dtype = bias.dtype if bias is not None else torch.float32
        return (
            grad_hidden.to(hidden.dtype),
            None,
            None,
            grad_bias.to(bias_dtype) if grad_bias is not None else None,
            None,
        )


class FrozenHeadLigerGRPOLoss(torch.nn.Module):
    """Delegate trainable heads to Liger; avoid its two full-vocabulary buffers for frozen heads."""

    def __init__(self, stock_loss: LigerFusedLinearGRPOLoss) -> None:
        super().__init__()
        self.stock_loss = stock_loss

    def forward(
        self,
        _input: torch.Tensor,
        lin_weight: torch.Tensor,
        selected_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        advantages: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if lin_weight.requires_grad:
            return self.stock_loss(
                _input,
                lin_weight,
                selected_token_ids,
                attention_mask,
                advantages,
                bias=bias,
                **kwargs,
            )
        batch, length, hidden = _input.shape
        logps = _FrozenHeadSelectiveLogProb.apply(
            _input.reshape(batch * length, hidden).contiguous(),
            lin_weight,
            selected_token_ids.reshape(-1).contiguous(),
            bias,
            self.stock_loss.temperature,
        ).reshape(batch, length)
        compute_loss = partial(
            fused_linear_ppo.LigerFusedLinearPPOBase._compute_loss_from_logps,  # pyright: ignore[reportPrivateUsage]
            full_attention_mask=attention_mask,
            epsilon_low=self.stock_loss.epsilon_low,
            epsilon_high=self.stock_loss.epsilon_high,
            beta=self.stock_loss.beta,
            loss_type=self.stock_loss.loss_type,
            max_completion_length=self.stock_loss.max_completion_length,
            importance_sampling_level=self.stock_loss.importance_sampling_level,
            ppo_loss_fn=LigerFusedLinearGRPOFunction.ppo_loss_fn,
            sapo_temperature_pos=self.stock_loss.sapo_temperature_pos,
            sapo_temperature_neg=self.stock_loss.sapo_temperature_neg,
            delta=self.stock_loss.delta,
            use_bias_correction_kl=self.stock_loss.use_bias_correction_kl,
            vespo_k_pos=self.stock_loss.vespo_k_pos,
            vespo_lambda_pos=self.stock_loss.vespo_lambda_pos,
            vespo_k_neg=self.stock_loss.vespo_k_neg,
            vespo_lambda_neg=self.stock_loss.vespo_lambda_neg,
            num_items_in_batch=kwargs.get("num_items_in_batch"),
        )
        ref_per_token_logps = kwargs.get("ref_per_token_logps")
        if self.stock_loss.use_ref_model and ref_per_token_logps is None:
            ref_input = kwargs.get("ref_input")
            ref_weight = kwargs.get("ref_weight")
            if ref_input is None or ref_weight is None:
                raise ValueError(
                    "frozen-head Liger loss needs ref_per_token_logps or ref_input/ref_weight"
                )
            with torch.no_grad():
                ref_per_token_logps = fused_linear_ppo.LigerFusedLinearPPOBase.chunk_forward(
                    ref_input,
                    ref_weight,
                    selected_token_ids,
                    bias=kwargs.get("ref_bias"),
                    temperature=self.stock_loss.temperature,
                )
        vllm_is_ratio = kwargs.get("vllm_is_ratio")
        vllm_is_ratio_fn = kwargs.get("vllm_is_ratio_fn")
        if vllm_is_ratio is not None and vllm_is_ratio_fn is not None:
            raise ValueError("pass either vllm_is_ratio or vllm_is_ratio_fn, not both")
        if vllm_is_ratio_fn is not None:
            vllm_is_ratio = vllm_is_ratio_fn(logps)
        elif callable(vllm_is_ratio):
            vllm_is_ratio = vllm_is_ratio(logps)
        return compute_loss(
            logps,
            attention_mask,
            advantages,
            ref_per_token_logps_chunk=ref_per_token_logps,
            old_per_token_logps_chunk=kwargs.get("old_per_token_logps"),
            vllm_is_ratio_chunk=vllm_is_ratio,
        )
