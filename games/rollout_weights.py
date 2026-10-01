"""Complete online-FP8 reloads and bounded exactness checks for colocated rollouts."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any

import torch
from vllm import _custom_ops as vllm_ops
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO

logger = logging.getLogger(__name__)

PACKED_PROBES: dict[str, tuple[str, tuple[str, ...]]] = {
    "deltanet_qkvz": (
        "linear_attn.in_proj_qkvz",
        ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
    ),
    "deltanet_ba": ("linear_attn.in_proj_ba", ("linear_attn.in_proj_b", "linear_attn.in_proj_a")),
    "mlp_gate_up": ("mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj")),
    "attn_qkv": (
        "self_attn.qkv_proj",
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ),
}
PushParam = Callable[[str, torch.Tensor], None]


class Fp8RolloutWeightSync:
    """Reload the full stream, retaining only four projections for byte and scale checks."""

    def __init__(self, generation: Any) -> None:  # noqa: ANN401 - TRL's internal worker seam
        """Bind the local worker and locate the first instance of each packed projection."""
        self.worker = generation.llm.llm_engine.model_executor.driver_worker
        self.model: torch.nn.Module = self.worker.model_runner.model
        self.sync_count = 0
        self.probes: dict[str, tuple[int, str, Any]] = {}
        modules = dict(self.model.named_modules())
        for probe, (suffix, _) in PACKED_PROBES.items():
            pattern = re.compile(rf"layers\.(\d+)\.{re.escape(suffix)}$")
            hits = sorted(
                (int(match.group(1)), name) for name in modules if (match := pattern.search(name))
            )
            if not hits:
                raise RuntimeError(f"no engine module matches FP8 probe {suffix}")
            layer_index, name = hits[0]
            self.probes[probe] = (layer_index, name, modules[name])

    def _pending(self) -> dict[str, list[str]]:
        pending: dict[str, list[str]] = {
            "unloaded": [],
            "partial": [],
            "rotary_caches": [],
            "attention": [],
        }
        for name, module in self.model.named_modules():
            info = LAYERWISE_INFO.get(module)
            if info is None or not info.can_load():
                continue
            total = info.load_numel_total or 0
            if isinstance(module, (Attention, MLAAttention)):
                pending["attention"].append(name)
            elif info.load_numel <= 0 and total > 0:
                # Only the known computed rotary caches may reuse their saved kernel buffers.
                rotary_cache = (
                    type(module) in (RotaryEmbedding, MRotaryEmbedding)
                    and not any(parameter is not None for parameter in module._parameters.values())  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                    and set(module._buffers) <= info.kernel_non_persistent_buffers  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                )
                pending["rotary_caches" if rotary_cache else "unloaded"].append(name)
            elif 0 < info.load_numel < total:
                pending["partial"].append(name)
        return pending

    def _check_probes(self, captured: dict[str, torch.Tensor]) -> None:
        for probe, (layer_index, name, layer) in self.probes.items():
            _, suffixes = PACKED_PROBES[probe]
            keys = [f"{layer_index}.{suffix}" for suffix in suffixes]
            missing = [key for key in keys if key not in captured]
            if missing:
                raise RuntimeError(f"FP8 probe {name} missing pushed tensors: {missing}")
            pushed = torch.cat([captured[key] for key in keys], dim=0)
            quantized, scale = vllm_ops.scaled_fp8_quant(pushed, scale=None)
            expected = quantized.t()
            actual = layer.weight.detach()
            weight_exact = (
                actual.shape == expected.shape
                and actual.dtype == expected.dtype
                and torch.equal(
                    actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
                )
            )
            scale_exact = torch.equal(layer.weight_scale.detach().reshape(-1), scale.reshape(-1))
            logger.info(
                "FP8 rollout sync=%d probe=%s weight_exact=%s scale_exact=%s",
                self.sync_count,
                probe,
                weight_exact,
                scale_exact,
            )
            if not weight_exact or not scale_exact:
                raise RuntimeError(f"FP8 engine weights do not match the pushed policy: {name}")

    @torch.no_grad()
    def sync(self, stream: Callable[[PushParam], None], *, push_param: PushParam) -> None:
        """Run after wake and before prefix-cache reset, including the first sync."""
        self.sync_count += 1
        started = time.perf_counter()
        captured: dict[str, torch.Tensor] = {}
        capture_keys = {
            f"{layer_index}.{suffix}"
            for probe, (layer_index, _, _) in self.probes.items()
            for suffix in PACKED_PROBES[probe][1]
        }
        push_count = 0

        def push(name: str, tensor: torch.Tensor) -> None:
            nonlocal push_count
            match = re.search(r"layers\.(\d+)\.(.+)\.weight$", name)
            if match and (key := f"{match.group(1)}.{match.group(2)}") in capture_keys:
                captured[key] = tensor.detach().clone()
            push_count += 1
            push_param(name, tensor)

        with set_current_vllm_config(self.worker.vllm_config):
            initialize_layerwise_reload(self.model)
            reloadable_layers = sum(
                1
                for _, module in self.model.named_modules()
                if (info := LAYERWISE_INFO.get(module)) is not None
                and info.can_load()
                and (info.load_numel_total or 0) > 0
            )
            stream(push)
            # Finalization resets the accounting, so retain missing/partial names beforehand.
            pending = self._pending()
            finalize_layerwise_reload(self.model, self.worker.vllm_config.model_config)
            still_pending = [
                name
                for name, module in self.model.named_modules()
                if (info := LAYERWISE_INFO.get(module)) is not None and info.can_load()
            ]
            logger.info(
                "FP8 rollout sync=%d reload: reloadable_layers=%d processed_during_load=%d "
                "unloaded=%s partial=%s rotary_caches=%s "
                "attention_deferred=%d still_pending=%s",
                self.sync_count,
                reloadable_layers,
                reloadable_layers - sum(len(names) for names in pending.values()),
                pending["unloaded"],
                pending["partial"],
                pending["rotary_caches"],
                len(pending["attention"]),
                still_pending,
            )
            if not push_count:
                raise RuntimeError("FP8 sync_weights pushed no tensors")
            if pending["unloaded"] or pending["partial"] or still_pending:
                raise RuntimeError("FP8 layerwise reload left layers unloaded or partial")
            self._check_probes(captured)
        logger.info(
            "FP8 rollout sync=%d pushes=%d reload_seconds=%.3f probes_exact=True",
            self.sync_count,
            push_count,
            time.perf_counter() - started,
        )
