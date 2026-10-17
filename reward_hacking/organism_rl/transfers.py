"""Storage-preserving host staging for sequential policy/sampler ownership.

Frozen host arenas exist only during rollout and are released after restore.
Mutable storage is refreshed on every training cycle. Tensor subclasses (including
TorchAO wrappers) require a separate representation-aware implementation and are
refused before any placement changes.
"""

from __future__ import annotations

import logging
import mmap
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


class _HostArena:
    """Exact-size mapped storage, optionally registered without CachingHostAllocator.

    Pageable mode avoids repeated CUDA registration cost (especially on WSL).
    Pinned mode permits asynchronous DMA but pays registration on every rollout.
    Compare complete phase times before selecting a machine's faster mode.
    """

    def __init__(self, size: int, *, pin_memory: bool) -> None:
        self._mapping: mmap.mmap | None = mmap.mmap(-1, max(size, 1))
        self.buffer: torch.Tensor | None = torch.frombuffer(
            self._mapping, dtype=torch.uint8, count=max(size, 1)
        )[:size]
        self._registered = pin_memory and size > 0
        if self._registered:
            runtime = torch.cuda.cudart()
            if runtime is None:
                raise RuntimeError("CUDA runtime is unavailable for host registration")
            status = runtime.cudaHostRegister(self.buffer.data_ptr(), size, 0)
            if status != runtime.cudaError.success:
                self.buffer = None
                self._mapping.close()
                raise RuntimeError(f"cudaHostRegister failed: {status}")

    def release(self) -> None:
        """Unregister after the caller's synchronization, then return pages to the OS."""
        if self.buffer is None:
            raise RuntimeError("host arena already released")
        if self._registered:
            runtime = torch.cuda.cudart()
            if runtime is None:
                raise RuntimeError("CUDA runtime is unavailable for host registration")
            status = runtime.cudaHostUnregister(self.buffer.data_ptr())
            if status != runtime.cudaError.success:
                raise RuntimeError(f"cudaHostUnregister failed: {status}")
        self.buffer = None
        # frombuffer's storage owns the mapping too. Dropping our reference unmaps it
        # immediately unless a caller still owns a detached host alias; that alias
        # must remain valid until its owner releases it.
        self._mapping = None


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
    """Stage frozen weights transiently; retain only mutable host storage during training."""

    def __init__(
        self, model: torch.nn.Module, training_device: torch.device, *, pin_memory: bool = False
    ) -> None:
        """Capture the policy after checkpoint restoration.

        pin_memory selects frozen arena registration; mutable CUDA state stays pinned.
        Detached snapshots borrowed while offloaded retain host pages until released.
        """
        self.model = model
        self.training_device = training_device
        # Pageable is provisional until phase benchmarks measure registration plus copy cost.
        self.pin_memory = pin_memory and training_device.type == "cuda"
        self._frozen_arenas: list[_HostArena] = []
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
        """Currently live frozen host storage identifiers; empty during training."""
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
        batches: dict[
            tuple[torch.dtype, bool], list[list[tuple[torch.Tensor, bool, torch.device, str]]]
        ] = {}
        for values in pending:
            mutable = any(mutable for _, mutable, _, _ in values)
            batches.setdefault((values[0][0].dtype, mutable), []).append(values)
        for (_, mutable), batch in batches.items():
            size = sum(_aligned_bytes(values[0][0].untyped_storage().nbytes()) for values in batch)
            if mutable:
                buffer = torch.empty(
                    size,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=self.training_device.type == "cuda",
                )
            else:
                arena = _HostArena(size, pin_memory=self.pin_memory)
                self._frozen_arenas.append(arena)
                if arena.buffer is None:
                    raise RuntimeError("new host arena has no storage")
                buffer = arena.buffer
            offset = 0
            for values in batch:
                tensor = values[0][0]
                size = tensor.untyped_storage().nbytes()
                host = buffer[offset : offset + size]
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

    def _finish(
        self, started: float, frozen: int, mutable: int, phase: str, *, synchronize: bool = True
    ) -> TransferReport:
        if synchronize and self.training_device.type == "cuda":
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
        """Copy all frozen and mutable storage, then release device storage after synchronization."""
        started = time.perf_counter()
        if self._host:
            return TransferReport(0, 0, time.perf_counter() - started)
        if self._rebuild_head:
            drop_fp32_head(self.model)
        self._collect_groups(optimizer)
        frozen = mutable = 0
        for group in self._groups:
            group.host.copy_(
                group.source_bytes(),
                non_blocking=self.training_device.type == "cuda",
            )
            if group.mutable:
                mutable += group.host.numel()
            else:
                frozen += group.host.numel()
        report = self._finish(started, frozen, mutable, "offload")
        for group in self._groups:
            group.bind(group.host)
        self._host = True
        return report

    def restore(self, optimizer: torch.optim.Optimizer | None) -> TransferReport:
        """Restore storage without replacing Parameters; synchronize before freeing frozen arenas."""
        del optimizer
        started = time.perf_counter()
        if not self._host:
            return TransferReport(0, 0, time.perf_counter() - started)
        frozen, mutable = self._restore_device_storage()
        if self._rebuild_head:
            prepare_fp32_head(self.model)
        if self.training_device.type == "cuda":
            torch.cuda.synchronize(self.training_device)
        self._groups = [group for group in self._groups if group.mutable]
        for arena in self._frozen_arenas:
            arena.release()
        self._frozen_arenas.clear()
        self._host = False
        return self._finish(started, frozen, mutable, "restore", synchronize=False)

    def _restore_device_storage(self) -> tuple[int, int]:
        # Keep borrowed host views local so none survive when restore releases their mappings.
        frozen = mutable = 0
        for group in self._groups:
            storage = torch.empty(group.host.shape, dtype=torch.uint8, device=group.target)
            storage.copy_(
                group.host,
                non_blocking=self.training_device.type == "cuda",
            )
            group.bind(storage)
            if group.mutable:
                mutable += group.host.numel()
            else:
                frozen += group.host.numel()
        return frozen, mutable
