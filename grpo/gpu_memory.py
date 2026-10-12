"""Read CUDA memory with the host's NVIDIA view included."""

from __future__ import annotations

import csv
import logging
import math
import os
import subprocess
from dataclasses import dataclass
from io import StringIO
from typing import Protocol

logger = logging.getLogger(__name__)

MEBIBYTE_BYTES = 1024**2
GIBIBYTE_BYTES = 1024**3
NVIDIA_SMI_TIMEOUT_SECONDS = 10
NVIDIA_SMI_FIELD_COUNT = 4


class CudaDevice(Protocol):
    """CUDA device identifiers accepted by Torch."""

    @property
    def type(self) -> str:
        """Return the device type name."""
        ...

    @property
    def index(self) -> int | None:
        """Return the device index, if one is explicit."""
        ...


class CudaRuntime(Protocol):
    """Torch CUDA methods used for memory sizing."""

    def current_device(self) -> int:
        """Return the current CUDA device index."""
        ...

    def mem_get_info(self, device: int) -> tuple[int, int]:
        """Return free and total device memory in bytes."""
        ...


class TorchRuntime(Protocol):
    """Minimal Torch module interface needed for memory sizing."""

    @property
    def cuda(self) -> CudaRuntime:
        """Expose CUDA device and memory queries."""
        ...


@dataclass(frozen=True, slots=True)
class _GpuMemorySnapshot:
    device_index: int
    torch_free_bytes: int
    nvidia_smi_free_bytes: int
    nvidia_smi_total_bytes: int
    free_bytes: int
    total_bytes: int
    host_held_estimate_bytes: int


def _logical_device_index(torch_module: TorchRuntime, device: int | CudaDevice | None) -> int:
    if device is None:
        device_index = int(torch_module.cuda.current_device())
    elif isinstance(device, int):
        device_index = device
    else:
        if device.type != "cuda":
            raise ValueError(f"expected a CUDA device, received {device}")
        device_index = device.index
        if device_index is None:
            device_index = int(torch_module.cuda.current_device())
    if device_index < 0:
        raise ValueError(f"CUDA device index must be nonnegative, received {device_index}")
    return device_index


def _nvidia_smi_rows() -> list[tuple[int, str, float, float]]:
    try:
        completed = subprocess.run(
            [  # noqa: S607 - fixed command name resolved through PATH
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "nvidia-smi is required to size CUDA memory but was not found on PATH"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("nvidia-smi timed out while reporting CUDA memory") from error

    if completed.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed while reporting CUDA memory: {completed.stderr.strip()}"
        )

    rows: list[tuple[int, str, float, float]] = []
    for row in csv.reader(StringIO(completed.stdout), skipinitialspace=True):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) != NVIDIA_SMI_FIELD_COUNT:
            raise RuntimeError(f"nvidia-smi returned an invalid GPU memory row: {row!r}")
        try:
            device_index = int(row[0].strip())
            memory_used_mib = float(row[2].strip())
            memory_total_mib = float(row[3].strip())
        except ValueError as error:
            raise RuntimeError(
                f"nvidia-smi returned nonnumeric GPU memory data: {row!r}"
            ) from error
        device_uuid = row[1].strip()
        if (
            device_index < 0
            or not device_uuid
            or not math.isfinite(memory_used_mib)
            or not math.isfinite(memory_total_mib)
            or memory_used_mib < 0
            or memory_total_mib <= 0
            or memory_used_mib > memory_total_mib
        ):
            raise RuntimeError(f"nvidia-smi returned invalid GPU memory values: {row!r}")
        rows.append((device_index, device_uuid, memory_used_mib, memory_total_mib))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs while sizing CUDA memory")
    return rows


def _nvidia_smi_row_for_device(
    rows: list[tuple[int, str, float, float]], device_index: int
) -> tuple[int, str, float, float]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None or not visible_devices.strip():
        matches = [row for row in rows if row[0] == device_index]
    else:
        visible_tokens = [token.strip() for token in visible_devices.split(",")]
        if device_index >= len(visible_tokens) or not visible_tokens[device_index]:
            raise RuntimeError(
                f"CUDA logical device {device_index} has no matching CUDA_VISIBLE_DEVICES entry"
            )
        visible_token = visible_tokens[device_index].lower()
        if visible_token.isdigit():
            physical_index = int(visible_token)
            matches = [row for row in rows if row[0] == physical_index]
        else:
            matches = [row for row in rows if row[1].lower().startswith(visible_token)]

    if len(matches) != 1:
        raise RuntimeError(
            f"could not uniquely match CUDA logical device {device_index} to nvidia-smi: "
            f"found {len(matches)} matching rows"
        )
    return matches[0]


def _gpu_memory_snapshot(
    torch_module: TorchRuntime, device: int | CudaDevice | None
) -> _GpuMemorySnapshot:
    device_index = _logical_device_index(torch_module, device)
    torch_free_bytes, torch_total_bytes = torch_module.cuda.mem_get_info(device_index)
    if torch_free_bytes < 0 or torch_total_bytes <= 0 or torch_free_bytes > torch_total_bytes:
        raise RuntimeError(
            f"CUDA reported invalid memory free={torch_free_bytes}, total={torch_total_bytes}"
        )

    nvidia_smi_row = _nvidia_smi_row_for_device(_nvidia_smi_rows(), device_index)
    _, device_uuid, memory_used_mib, memory_total_mib = nvidia_smi_row
    nvidia_smi_free_bytes = round((memory_total_mib - memory_used_mib) * MEBIBYTE_BYTES)
    nvidia_smi_total_bytes = round(memory_total_mib * MEBIBYTE_BYTES)
    selected_free_bytes = min(torch_free_bytes, nvidia_smi_free_bytes)
    host_held_estimate_bytes = max(torch_free_bytes - nvidia_smi_free_bytes, 0)

    logger.info(
        "CUDA memory device=%d uuid=%s torch_free_gib=%.2f torch_total_gib=%.2f "
        "nvidia_smi_free_gib=%.2f nvidia_smi_total_gib=%.2f selected_free_gib=%.2f "
        "host_held_estimate_gib=%.2f",
        device_index,
        device_uuid,
        torch_free_bytes / GIBIBYTE_BYTES,
        torch_total_bytes / GIBIBYTE_BYTES,
        nvidia_smi_free_bytes / GIBIBYTE_BYTES,
        nvidia_smi_total_bytes / GIBIBYTE_BYTES,
        selected_free_bytes / GIBIBYTE_BYTES,
        host_held_estimate_bytes / GIBIBYTE_BYTES,
    )
    return _GpuMemorySnapshot(
        device_index=device_index,
        torch_free_bytes=torch_free_bytes,
        nvidia_smi_free_bytes=nvidia_smi_free_bytes,
        nvidia_smi_total_bytes=nvidia_smi_total_bytes,
        free_bytes=selected_free_bytes,
        total_bytes=torch_total_bytes,
        host_held_estimate_bytes=host_held_estimate_bytes,
    )


def gpu_memory_info(
    torch_module: TorchRuntime, device: int | CudaDevice | None = None
) -> tuple[int, int]:
    """Return the lower of Torch and nvidia-smi free memory, with Torch's total memory."""
    snapshot = _gpu_memory_snapshot(torch_module, device)
    return snapshot.free_bytes, snapshot.total_bytes


def cap_gpu_memory_utilization(  # noqa: PLR0913 - consumers set each sizing input independently
    torch_module: TorchRuntime,
    requested: float,
    device: int | CudaDevice | None = None,
    margin_fraction: float = 0.02,
    *,
    reserve_bytes: int = 0,
    minimum_utilization_bytes: int = 0,
) -> float:
    """Cap a vLLM fraction after applying the configured margin and byte reserve."""
    if not 0 < requested <= 1:
        raise ValueError(
            f"requested GPU memory utilization must be in (0, 1], received {requested}"
        )
    if not 0 <= margin_fraction < 1:
        raise ValueError(f"GPU memory margin must be in [0, 1), received {margin_fraction}")
    if reserve_bytes < 0:
        raise ValueError(f"GPU memory reserve must be nonnegative, received {reserve_bytes}")
    if minimum_utilization_bytes < 0:
        raise ValueError(
            "minimum engine utilization bytes must be nonnegative, "
            f"received {minimum_utilization_bytes}"
        )

    snapshot = _gpu_memory_snapshot(torch_module, device)
    available_fraction = (
        snapshot.free_bytes / snapshot.total_bytes
        - margin_fraction
        - reserve_bytes / snapshot.total_bytes
    )
    if available_fraction <= 0:
        raise RuntimeError(
            "no usable GPU memory remains after the configured safety margin and byte reserve: "
            f"reserve_bytes={reserve_bytes}; "
            f"host-held/other-process memory is estimated at "
            f"{snapshot.host_held_estimate_bytes / GIBIBYTE_BYTES:.2f} GiB "
            "from the torch/nvidia-smi free-memory difference"
        )
    capped_utilization = min(requested, available_fraction)
    capped_allocation_bytes = capped_utilization * snapshot.total_bytes
    if capped_allocation_bytes < minimum_utilization_bytes:
        raise RuntimeError(
            "the capped vLLM allocation cannot fit its minimum engine memory requirement: "
            f"minimum_utilization_bytes={minimum_utilization_bytes}, "
            f"capped_allocation_bytes={capped_allocation_bytes:.0f}; "
            f"host-held/other-process memory is estimated at "
            f"{snapshot.host_held_estimate_bytes / GIBIBYTE_BYTES:.2f} GiB "
            "from the torch/nvidia-smi free-memory difference"
        )
    return capped_utilization
