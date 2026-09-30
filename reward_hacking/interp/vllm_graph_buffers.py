"""Persistent residual-edit tensors installed before vLLM's first Dynamo trace."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch._inductor import config as inductor_config

from reward_hacking.interp.directions import unit
from reward_hacking.interp.steering import ablate_residual

if TYPE_CHECKING:
    from collections.abc import Mapping

    from reward_hacking.interp.vllm_interventions import InterventionSpec

PAIR_LENGTH = 2


def validate_graph_compiler() -> None:
    """Refuse constant-folding settings that would ignore later intervention updates."""
    if inductor_config.freezing:
        raise RuntimeError(
            "Inductor freezing ignores residual-buffer updates; require TORCHINDUCTOR_FREEZING=0"
        )


class ResidualGraphBuffers(nn.Module):
    """A fixed-shape edit whose storage survives graph replay."""

    basis: torch.Tensor
    direction: torch.Tensor
    alpha: torch.Tensor
    enabled: torch.Tensor

    def __init__(self, *, rank: int, width: int, device: torch.device) -> None:
        """Allocate storage once, before any model forward can be captured."""
        super().__init__()
        self.register_buffer("basis", torch.zeros(rank, width, device=device))
        self.register_buffer("direction", torch.zeros(width, device=device))
        self.register_buffer("alpha", torch.zeros((), device=device))
        self.register_buffer("enabled", torch.zeros((), dtype=torch.bool, device=device))

    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trace a pure tensor edit with no runtime Python state or buffer mutation."""
        full = hidden + residual
        edited = ablate_residual(full, self.basis) + self.alpha * self.direction.to(full)
        return (
            torch.where(self.enabled, torch.zeros_like(hidden), hidden),
            torch.where(self.enabled, edited, residual),
        )

    def output_hook(
        self, _module: nn.Module, _inputs: tuple[object, ...], output: object
    ) -> object:
        """Trace the tensor edit as part of the compiled model."""
        if not (
            isinstance(output, tuple)
            and len(output) == PAIR_LENGTH
            and isinstance(output[0], torch.Tensor)
            and isinstance(output[1], torch.Tensor)
            and output[0].shape == output[1].shape
        ):
            raise ValueError(
                "graph residual edits require a matching (hidden, residual) tensor pair"
            )
        return self(output[0], output[1])

    def clear(self) -> None:
        """Disable the edit in place without invalidating captured storage addresses."""
        self.enabled.zero_()
        self.basis.zero_()
        self.direction.zero_()
        self.alpha.zero_()


class GraphInterventionState:
    """Worker control state; only fixed tensors are read by the compiled forward."""

    def __init__(self, buffers: Mapping[int, ResidualGraphBuffers]) -> None:
        """Retain the prepared layer buffers for condition updates."""
        self.buffers = dict(buffers)
        self.active_layers: tuple[int, ...] | None = None

    def install(self, spec: InterventionSpec) -> None:
        """Validate capacity before any updates, then copy one condition into existing buffers."""
        if self.active_layers is not None:
            raise RuntimeError("a residual intervention is already installed on this vLLM worker")
        for layer, tensor in spec.by_layer.items():
            if layer not in self.buffers:
                raise ValueError(f"layer {layer} was not prepared before graph capture")
            if spec.kind == "subspace" and tensor.shape[0] > self.buffers[layer].basis.shape[0]:
                raise ValueError(f"layer {layer} subspace rank exceeds graph buffer capacity")
        for buffer in self.buffers.values():
            buffer.clear()
        for layer, tensor in spec.by_layer.items():
            buffer = self.buffers[layer]
            if spec.kind == "subspace":
                buffer.basis[: tensor.shape[0]].copy_(tensor)
            else:
                buffer.direction.copy_(unit(tensor))
                buffer.alpha.fill_(cast("float", spec.alpha))
            buffer.enabled.fill_(1)
        self.active_layers = tuple(spec.by_layer)

    def counts(self) -> None:
        """Report that graph execution is validated by differential logits, not hook counters.

        vLLM forbids mutating module buffers inside its compiled CUDA-graph forward, so a counter
        cannot safely establish execution here. The replay validator checks the intervention's
        effect after capture instead; returning None avoids reporting fabricated hook counts.
        """
        if self.active_layers is None:
            raise RuntimeError("no residual intervention is installed on this vLLM worker")

    def remove(self) -> None:
        """Restore identity settings while retaining every captured tensor and hook."""
        if self.active_layers is None:
            raise RuntimeError("no residual intervention is installed on this vLLM worker")
        for buffer in self.buffers.values():
            buffer.clear()
        self.active_layers = None
