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

import torch

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
        self.tensor.data = storage.view(self.dtype).as_strided(self.shape, self.stride, self.offset)


@dataclass
class _StorageGroup:
    views: list[_TensorView]
    host: torch.Tensor
    mutable: bool
    target: torch.device
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


class PolicyTransfers:
    """Keep host weight storage while releasing all policy GPU storage between phases."""

    def __init__(self, model: torch.nn.Module, training_device: torch.device) -> None:
        """Capture the already-loaded policy, after checkpoint restoration."""
        self.model = model
        self.training_device = training_device
        self._groups: list[_StorageGroup] = []
        self._host = False
        self._base = list(model.parameters()) + list(model.buffers())
        self._base_ids = {id(tensor) for tensor in self._base}
        for tensor in self._base:
            _validate_tensor(tensor)

    @property
    def frozen_host_storage_ids(self) -> tuple[int, ...]:
        """Stable host storage identifiers for transfer accounting and diagnostics."""
        return tuple(
            group.host.untyped_storage().data_ptr() for group in self._groups if not group.mutable
        )

    def _collect_groups(self, optimizer: torch.optim.Optimizer | None) -> None:
        parameters = list(self.model.parameters())
        buffers = list(self.model.named_buffers())
        if {
            id(tensor) for tensor in parameters + [tensor for _, tensor in buffers]
        } != self._base_ids:
            raise RuntimeError("Policy tensors changed after phase transfer initialization")
        frozen_buffers = {
            f"{prefix}.{name}" if prefix else name
            for prefix, module in self.model.named_modules()
            for name in cast("set[str]", module.__dict__.get("_phase_frozen_buffer_names", set()))
        }
        records: list[tuple[torch.Tensor, bool, torch.device]] = [
            (parameter, parameter.requires_grad, self.training_device) for parameter in parameters
        ]
        records.extend(
            (buffer, name not in frozen_buffers, self.training_device) for name, buffer in buffers
        )
        records.extend(
            (parameter.grad, True, self.training_device)
            for parameter in parameters
            if parameter.grad is not None
        )
        if optimizer is not None:
            records.extend(
                (tensor, True, tensor.device) for tensor in _state_tensors(optimizer.state)
            )
        for tensor, _, _ in records:
            _validate_tensor(tensor)
        current_ids = {id(tensor) for tensor, _, _ in records}
        self._groups = [
            group
            for group in self._groups
            if all(id(view.tensor) in current_ids for view in group.views)
        ]
        retained_ids = {id(view.tensor) for group in self._groups for view in group.views}
        new_records = [
            (tensor, mutable, target)
            for tensor, mutable, target in records
            if id(tensor) not in retained_ids
        ]
        grouped: dict[tuple[str, int], list[tuple[torch.Tensor, bool, torch.device]]] = {}
        for tensor, mutable, target in new_records:
            key = (str(tensor.device), tensor.untyped_storage()._cdata)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            grouped.setdefault(key, []).append((tensor, mutable, target))
        for values in grouped.values():
            targets = {target for _, _, target in values}
            if len(targets) != 1:
                raise ValueError(
                    "Aliased policy and optimizer storage requires incompatible restore devices"
                )
            tensor = values[0][0]
            host = torch.empty(
                tensor.untyped_storage().nbytes(),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=self.training_device.type == "cuda",
            )
            views = [
                _TensorView(
                    value, value.shape, value.stride(), int(value.storage_offset()), value.dtype
                )
                for value, _, _ in values
            ]
            self._groups.append(
                _StorageGroup(views, host, any(mutable for _, mutable, _ in values), targets.pop())
            )

    def _finish(self, started: float, frozen: int, mutable: int, phase: str) -> TransferReport:
        if self.training_device.type == "cuda":
            torch.cuda.synchronize(self.training_device)
        report = TransferReport(frozen, mutable, time.perf_counter() - started)
        logger.info(
            "policy transfer phase=%s moved_bytes=%d frozen_bytes=%d mutable_bytes=%d seconds=%.6f GB/s=%.3f",
            phase,
            report.moved_bytes,
            frozen,
            mutable,
            report.seconds,
            report.moved_bytes / max(report.seconds, 1e-12) / 1e9,
        )
        return report

    def offload(self, optimizer: torch.optim.Optimizer | None) -> TransferReport:
        """Refresh mutable host storage, then release device storage after one synchronization."""
        started = time.perf_counter()
        if self._host:
            return TransferReport(0, 0, time.perf_counter() - started)
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
        report = self._finish(started, frozen, mutable, "restore")
        self._host = False
        return report
