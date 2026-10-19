"""Persist aggregate CUDA memory evidence across each forward/backward micro-step.

Start records are flushed before forward so a driver crash still identifies the row length.
These are observations, not a capacity guarantee: allocator peaks exclude other CUDA processes
and WSL host allocations. Free memory is read through the shared host-aware GPU query.
"""

import json
import logging
import os
import time
from pathlib import Path

import torch

from grpo.gpu_memory import gpu_memory_info

logger = logging.getLogger(__name__)


def append_memory_record(path: Path, record: dict[str, int | float | str | bool]) -> None:
    """Flush a privacy-safe measurement before the next potentially failing CUDA operation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def record_training_phase_memory(
    path: Path, device: torch.device, *, step: int, phase: str
) -> None:
    """Measure training headroom on both sides of restoring the sleeping sampler's trainer."""
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = gpu_memory_info(torch, device)
    record: dict[str, int | float | str | bool] = {
        "event": phase,
        "schema_version": 1,
        "measured_at_unix_ns": time.time_ns(),
        "pid": os.getpid(),
        "step": step,
        "device": str(device),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
    }
    append_memory_record(path, record)
    logger.info("training phase memory: %s", record)


class MicrostepMemoryRecorder:
    """Reset peaks after sampling/input preparation; finish only after backward returns."""

    def __init__(self, path: Path) -> None:
        """Use one append-only run-local file across optimizer steps and resumes."""
        self.path = path
        self.device: torch.device | None = None
        self.record: dict[str, int | float | str | bool] | None = None
        self.started = 0.0

    def begin(
        self,
        device: torch.device,
        *,
        step: int,
        microstep: int,
        sequence_tokens: int,
        assistant_tokens: int,
    ) -> None:
        """Persist baseline allocations and row dimensions without recording token contents."""
        if self.record is not None:
            raise RuntimeError("a training memory micro-step is already active")
        if device.type != "cuda":
            return
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = gpu_memory_info(torch, device)
        torch.cuda.reset_peak_memory_stats(device)
        self.record = {
            "schema_version": 1,
            "event": "microstep_start",
            "measured_at_unix_ns": time.time_ns(),
            "pid": os.getpid(),
            "step": step,
            "microstep": microstep,
            "device": str(device),
            "sequence_tokens": sequence_tokens,
            "assistant_tokens": assistant_tokens,
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "allocated_before_bytes": torch.cuda.memory_allocated(device),
            "reserved_before_bytes": torch.cuda.memory_reserved(device),
        }
        self.device = device
        append_memory_record(self.path, self.record)
        self.started = time.perf_counter()

    def finish(self, *, completed: bool, skipped: bool) -> int:
        """Persist completion without querying a CUDA context after forward or backward failed."""
        record, device = self.record, self.device
        self.record = None
        self.device = None
        if record is None or device is None:
            return 0
        record = dict(record, event="microstep_end" if completed else "microstep_failed")
        record.update(completed=completed, skipped=skipped)
        peak = 0
        if completed:
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            record.update(
                peak_allocated_bytes=peak,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                allocated_after_bytes=torch.cuda.memory_allocated(device),
                seconds=time.perf_counter() - self.started,
            )
        append_memory_record(self.path, record)
        logger.info("training micro-step memory: %s", record)
        return peak
