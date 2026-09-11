"""Refuse to start a GPU job when another process already holds significant VRAM.

This box has a single NVIDIA L4 with 23 GB. Two concurrent jobs — or one orphaned
CUDA process left behind by a killed run — will either OOM the new job partway
through or silently make both crawl. Checking first, and naming the process that
holds the memory, turns a confusing mid-run failure into an immediate clear one.

Standard-library only, so it runs under any interpreter on the box without needing
the repo's conda environment. Usable two ways:

    scripts/resource-limits.sh --gpu -- python train.py   # via the run wrapper
    python scripts/gpu_preflight.py                   # standalone, exit 1 if busy

or from a notebook cell, before allocating anything:

    from scripts.gpu_preflight import require_free_gpu
    require_free_gpu()
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Floor chosen so an idle CUDA context (a few hundred MiB) does not read as busy.
DEFAULT_BUSY_THRESHOLD_MIB = 512


class GpuBusyError(RuntimeError):
    """Another process holds more VRAM than the caller is willing to tolerate."""


@dataclass(frozen=True)
class GpuProcess:
    """Keep GPU holders readable in busy-GPU diagnostics."""

    pid: int
    # None means nvidia-smi would not say, which is NOT the same as zero -- see gpu_processes.
    used_mib: int | None
    name: str

    def __str__(self) -> str:
        """Format a concise holder description for diagnostics."""
        if self.used_mib is None:
            return f"pid {self.pid} holding an unknown amount of VRAM ({self.name})"
        return f"pid {self.pid} holding {self.used_mib} MiB ({self.name})"


def _nvidia_smi(section: str, fields: str) -> list[str]:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        raise RuntimeError("nvidia-smi not found; this host has no usable NVIDIA GPU")
    out = subprocess.run(  # noqa: S603 - trusted executable and literal arguments
        [smi, f"--query-{section}={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in (raw.strip() for raw in out.splitlines()) if line]


def gpu_processes() -> list[GpuProcess]:
    """Every process currently holding memory on the GPU, per nvidia-smi.

    ``used_memory`` reads ``[N/A]`` for a peer nvidia-smi can see but not inspect, e.g. one in
    another container, and that row becomes ``used_mib=None``. Unknown usage is not zero usage:
    reading it as zero let a peer holding the whole card fall below the busy threshold and pass.
    """
    rows = _nvidia_smi("compute-apps", "pid,used_memory,process_name")
    processes: list[GpuProcess] = []
    for row in rows:
        pid, used_mib, name = (field.strip() for field in row.split(",", 2))
        processes.append(GpuProcess(int(pid), int(used_mib) if used_mib.isdigit() else None, name))
    return processes


def vram_mib_across_devices() -> tuple[int, int]:
    """Return (used, total) VRAM in MiB summed over every GPU on the host.

    Summed rather than read from row [0], because ``gpu_processes`` reports compute-apps rows for
    every device: subtracting an all-device attributed total from a GPU-0 used total is arithmetic
    over two different populations, and on a multi-GPU host it computes a zero remainder for a card
    that is entirely full. Per-device evaluation would be finer still, but the holders check already
    treats a named peer on any device as busy, so summing is the consistent reading.
    """
    used_mib = 0
    total_mib = 0
    for row in _nvidia_smi("gpu", "memory.used,memory.total"):
        used, total = row.split(",")
        used_mib += int(used)
        total_mib += int(total)
    return used_mib, total_mib


def require_free_gpu(threshold_mib: int = DEFAULT_BUSY_THRESHOLD_MIB) -> None:
    """Raise GpuBusyError unless the GPU is free, counting VRAM no process accounts for.

    There are two ways the card can be occupied and both have to be checked, because per-process
    attribution is not guaranteed. The obvious one is a named peer holding at least
    ``threshold_mib``. The other is VRAM the aggregate reports as in use that no ``compute-apps``
    row accounts for: nvidia-smi lists no row at all for a peer in another container, and reports
    ``[N/A]`` usage for one it can see but not inspect. This check used to read both as an empty
    card -- it fetched the aggregate only to interpolate it into the error message, never to compare
    it -- so a nearly full GPU logged "preflight OK, used=22000 of total=23034 MiB in use", which is
    the exact shape of a gate that prints a reassuring message. See tests/test_gpu_preflight.py.

    Our own allocation is subtracted along with everyone else's, so calling this from a notebook
    that has already allocated does not read its own VRAM as somebody else's.
    """
    self_pid = os.getpid()
    processes = gpu_processes()
    used, total = vram_mib_across_devices()

    holders = [
        process
        for process in processes
        if process.pid != self_pid
        and (process.used_mib is None or process.used_mib >= threshold_mib)
    ]
    if holders:
        listed = "\n  ".join(str(process) for process in holders)
        raise GpuBusyError(
            f"GPU is busy: {used}/{total} MiB in use by another process.\n  {listed}\n"
            "Wait for it, or kill it if it is an orphan from a dead run "
            "(check with: nvidia-smi)."
        )

    attributed = 0
    for process in processes:
        if process.used_mib is not None:
            attributed += process.used_mib
    unattributed = used - attributed
    if unattributed >= threshold_mib:
        raise GpuBusyError(
            f"GPU is busy: {used}/{total} MiB in use, but nvidia-smi accounted for only "
            f"{attributed} MiB of it, leaving {unattributed} MiB held by no process it will name. "
            "That is usually a peer in another container or a namespace this process cannot see, "
            "so treat the card as taken (check with: nvidia-smi)."
        )
    logger.info(f"GPU preflight OK, {used=} of {total=} MiB in use")


def main() -> int:
    """Reject startup when another process already occupies the GPU."""
    parser = argparse.ArgumentParser(
        description="Refuse to start a GPU job when another process already holds VRAM."
    )
    parser.add_argument(
        "--threshold-mib",
        type=int,
        default=DEFAULT_BUSY_THRESHOLD_MIB,
        help="treat the GPU as busy when another process holds at least this much VRAM",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="gpu-preflight: %(message)s")
    try:
        require_free_gpu(args.threshold_mib)
    except GpuBusyError as exc:
        logger.error(str(exc))  # noqa: TRY400 - clean expected error; traceback is noise
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
