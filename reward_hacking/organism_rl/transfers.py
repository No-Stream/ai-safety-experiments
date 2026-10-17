"""Storage-preserving host staging for sequential policy/sampler ownership.

Frozen storage is copied to host only once. Mutable storage is refreshed on every
training cycle. Tensor subclasses (including TorchAO wrappers) require a separate
representation-aware implementation and are refused before any placement changes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import cast

import psutil
import torch

from reward_hacking.distill.train import drop_fp32_head, prepare_fp32_head

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransferReport:
    """Actual storage bytes copied in a completed phase."""

    frozen_bytes: int
    mutable_bytes: int
    seconds: float

    @property
    def moved_bytes(self) -> int:
        """Total bytes, counting aliased storage once."""
        return self.frozen_bytes + self.mutable_bytes


@dataclass
class _TensorView:
    tensor: torch.Tensor
    shape: torch.Size
    stride: tuple[int, ...]
    offset: int
    dtype: torch.dtype

    def bind(self, storage: torch.Tensor) -> None:
        self.tensor.data = storage.view(self.dtype).as_strided(
            self.shape,
            self.stride,
            self.offset + int(storage.storage_offset()) // self.tensor.element_size(),
        )


@dataclass
class _StorageGroup:
    views: list[_TensorView]
    host: torch.Tensor
    mutable: bool
    target: torch.device
    category: str
    initialized: bool = False

    def source_bytes(self) -> torch.Tensor:
        tensor = self.views[0].tensor
        return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
            tensor.untyped_storage(), 0, (self.host.numel(),), (1,)
        )

    def bind(self, storage: torch.Tensor) -> None:
        for view in self.views:
            view.bind(storage)


def _state_tensors(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [
            tensor
            for item in cast("dict[object, object]", value).values()
            for tensor in _state_tensors(item)
        ]
    if isinstance(value, (list, tuple)):
        return [
            tensor
            for item in cast("list[object] | tuple[object, ...]", value)
            for tensor in _state_tensors(item)
        ]
    return []


def _validate_tensor(tensor: torch.Tensor) -> None:
    if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
        raise TypeError(
            "Policy phase transfer refuses tensor subclass storage, including TorchAO wrappers"
        )
    if tensor.layout != torch.strided or tensor.device.type not in ("cpu", "cuda"):
        raise ValueError("Policy phase transfer requires dense CPU or CUDA tensor storage")


def _aligned_bytes(size: int) -> int:
    return ((size + 7) // 8) * 8


def _parameter_category(name: str) -> str:
    categories = {
        "lora_": "adapters",
        "embed_tokens": "embeddings",
        "lm_head": "lm_head",
        "self_attn": "attention",
        "linear_attn": "attention",
        "mlp": "mlp",
        "norm": "norms",
    }
    return next(
        (category for fragment, category in categories.items() if fragment in name), "parameters"
    )


class PolicyTransfers:
    """Keep host weight storage while releasing all policy GPU storage between phases."""

    def __init__(self, model: torch.nn.Module, training_device: torch.device) -> None:
        """Capture the already-loaded policy, after checkpoint restoration."""
        self.model = model
        self.training_device = training_device
        self._groups: list[_StorageGroup] = []
        self._host = False
        self._rebuild_head = any(
            module.__dict__.get("_phase_derived_buffer_names") for module in model.modules()
        )
        self._base = list(model.parameters()) + [tensor for _, tensor in self._buffers()]
        self._base_ids = {id(tensor) for tensor in self._base}
        for tensor in self._base:
            _validate_tensor(tensor)

    @property
    def frozen_host_storage_ids(self) -> tuple[int, ...]:
        """Stable host storage identifiers for transfer accounting and diagnostics."""
        return tuple(
            group.host.untyped_storage().data_ptr() for group in self._groups if not group.mutable
        )

    def _buffers(self) -> list[tuple[str, torch.Tensor]]:
        derived = {
            f"{prefix}.{name}" if prefix else name
            for prefix, module in self.model.named_modules()
            for name in cast("set[str]", module.__dict__.get("_phase_derived_buffer_names", set()))
        }
        return [
            (name, tensor) for name, tensor in self.model.named_buffers() if name not in derived
        ]

    @property
    def host_bytes(self) -> int:
        """Allocated arena bytes, including padding and retained inactive slices."""
        arenas = {
            group.host.untyped_storage().data_ptr(): group.host.untyped_storage().nbytes()
            for group in self._groups
        }
        return sum(arenas.values())

    @property
    def host_byte_breakdown(self) -> dict[str, int]:
        """Storage-deduplicated bytes by module category and mutability."""
        breakdown: dict[str, int] = {}
        for group in self._groups:
            key = f"{'mutable' if group.mutable else 'frozen'}.{group.category}"
            breakdown[key] = breakdown.get(key, 0) + group.host.numel()
        overhead = self.host_bytes - sum(breakdown.values())
        if overhead:
            breakdown["arena_padding_and_inactive"] = overhead
        return breakdown

    def _pending_groups(
        self, optimizer: torch.optim.Optimizer | None
    ) -> list[list[tuple[torch.Tensor, bool, torch.device, str]]]:
        parameters = list(self.model.named_parameters())
        buffers = self._buffers()
        if {id(tensor) for _, tensor in parameters + buffers} != self._base_ids:
            raise RuntimeError("Policy tensors changed after phase transfer initialization")
        frozen_buffers = {
            f"{prefix}.{name}" if prefix else name
            for prefix, module in self.model.named_modules()
            for name in cast("set[str]", module.__dict__.get("_phase_frozen_buffer_names", set()))
        }
        records: list[tuple[torch.Tensor, bool, torch.device, str]] = [
            (parameter, parameter.requires_grad, self.training_device, _parameter_category(name))
            for name, parameter in parameters
        ]
        records.extend(
            (buffer, name not in frozen_buffers, self.training_device, "buffers")
            for name, buffer in buffers
        )
        records.extend(
            (parameter.grad, True, self.training_device, "gradients")
            for _, parameter in parameters
            if parameter.grad is not None
        )
        if optimizer is not None:
            records.extend(
                (tensor, True, tensor.device, "optimizer")
                for tensor in _state_tensors(optimizer.state)
            )
        for tensor, _, _, _ in records:
            _validate_tensor(tensor)
        current_ids = {id(tensor) for tensor, _, _, _ in records}
        self._groups = [
            group
            for group in self._groups
            if all(id(view.tensor) in current_ids for view in group.views)
        ]
        retained_ids = {id(view.tensor) for group in self._groups for view in group.views}
        grouped: dict[tuple[str, int], list[tuple[torch.Tensor, bool, torch.device, str]]] = {}
        for record in records:
            tensor = record[0]
            if id(tensor) in retained_ids:
                continue
            key = (str(tensor.device), tensor.untyped_storage()._cdata)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            grouped.setdefault(key, []).append(record)
        for values in grouped.values():
            if len({target for _, _, target, _ in values}) != 1:
                raise ValueError(
                    "Aliased policy and optimizer storage requires incompatible restore devices"
                )
        return list(grouped.values())

    def planned_host_bytes(self, optimizer: torch.optim.Optimizer | None) -> int:
        """Account for the next arena before allocating any host storage."""
        pending = self._pending_groups(optimizer)
        return self.host_bytes + sum(
            _aligned_bytes(values[0][0].untyped_storage().nbytes()) for values in pending
        )

    def _collect_groups(self, optimizer: torch.optim.Optimizer | None) -> None:
        pending = self._pending_groups(optimizer)
        if not pending:
            return
        # torch.save refuses different dtypes sharing a storage, so use one arena per dtype.
        batches: dict[torch.dtype, list[list[tuple[torch.Tensor, bool, torch.device, str]]]] = {}
        for values in pending:
            batches.setdefault(values[0][0].dtype, []).append(values)
        for batch in batches.values():
            arena = torch.empty(
                sum(_aligned_bytes(values[0][0].untyped_storage().nbytes()) for values in batch),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=self.training_device.type == "cuda",
            )
            offset = 0
            for values in batch:
                tensor = values[0][0]
                size = tensor.untyped_storage().nbytes()
                host = arena[offset : offset + size]
                offset += _aligned_bytes(size)
                views = [
                    _TensorView(
                        value, value.shape, value.stride(), int(value.storage_offset()), value.dtype
                    )
                    for value, _, _, _ in values
                ]
                categories = {category for _, _, _, category in values}
                category = next(iter(categories)) if len(categories) == 1 else "shared_storage"
                self._groups.append(
                    _StorageGroup(
                        views,
                        host,
                        any(mutable for _, mutable, _, _ in values),
                        values[0][2],
                        category,
                    )
                )
        logger.info(
            "policy host storage bytes=%d breakdown=%s", self.host_bytes, self.host_byte_breakdown
        )

    def _finish(self, started: float, frozen: int, mutable: int, phase: str) -> TransferReport:
        if self.training_device.type == "cuda":
            torch.cuda.synchronize(self.training_device)
        report = TransferReport(frozen, mutable, time.perf_counter() - started)
        logger.info(
            "policy transfer phase=%s moved_bytes=%d frozen_bytes=%d mutable_bytes=%d seconds=%.6f GB/s=%.3f "
            "host_available_bytes=%d host_storage_bytes=%d",
            phase,
            report.moved_bytes,
            frozen,
            mutable,
            report.seconds,
            report.moved_bytes / max(report.seconds, 1e-12) / 1e9,
            psutil.virtual_memory().available,
            self.host_bytes,
        )
        return report

    def offload(self, optimizer: torch.optim.Optimizer | None) -> TransferReport:
        """Refresh mutable host storage, then release device storage after one synchronization."""
        started = time.perf_counter()
        if self._host:
            return TransferReport(0, 0, time.perf_counter() - started)
        if self._rebuild_head:
            drop_fp32_head(self.model)
        self._collect_groups(optimizer)
        frozen = mutable = 0
        for group in self._groups:
            if group.mutable or not group.initialized:
                group.host.copy_(group.source_bytes(), non_blocking=True)
                if group.mutable:
                    mutable += group.host.numel()
                else:
                    frozen += group.host.numel()
                group.initialized = True
        report = self._finish(started, frozen, mutable, "offload")
        for group in self._groups:
            group.bind(group.host)
        self._host = True
        return report

    def restore(self, optimizer: torch.optim.Optimizer | None) -> TransferReport:
        """Restore cached and mutable storage without replacing Parameter objects."""
        del optimizer
        started = time.perf_counter()
        if not self._host:
            return TransferReport(0, 0, time.perf_counter() - started)
        frozen = mutable = 0
        for group in self._groups:
            storage = torch.empty(group.host.shape, dtype=torch.uint8, device=group.target)
            storage.copy_(group.host, non_blocking=True)
            group.bind(storage)
            if group.mutable:
                mutable += group.host.numel()
            else:
                frozen += group.host.numel()
        if self._rebuild_head:
            prepare_fp32_head(self.model)
        report = self._finish(started, frozen, mutable, "restore")
        self._host = False
        return report
