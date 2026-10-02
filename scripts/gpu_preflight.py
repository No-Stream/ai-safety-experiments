"""Refuse to start a GPU job when another process already holds significant VRAM.

This box has a single GPU shared by several agent sessions. Two concurrent jobs — or one orphaned
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
from pathlib import Path

logger = logging.getLogger(__name__)

# Floor chosen so an idle CUDA context (a few hundred MiB) does not read as busy.
DEFAULT_BUSY_THRESHOLD_MIB = 512
# WSL cannot attribute VRAM to a Linux process, and anything that initialises CUDA (a test worker,
# say) holds /dev/dxg open. The Windows desktop measured 2.0-3.5 GiB on the 5090 box, so a holder
# is only treated as a peer once the card is used beyond that.
WSL_DESKTOP_ALLOWANCE_MIB = 4096
WSL_CMDLINE_MAX_CHARS = 160


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


def _kernel_osrelease() -> str:
    return Path("/proc/sys/kernel/osrelease").read_text()


def host_is_wsl() -> bool:
    """Whether this is a WSL2 guest, whose GPU is shared with the Windows host's own desktop."""
    return "microsoft" in _kernel_osrelease().lower()


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


def _proc_fd_paths(process_dir: Path) -> tuple[list[Path], bool]:
    """Return fd entries and whether their directory could not be read."""
    try:
        return list((process_dir / "fd").iterdir()), False
    except PermissionError:
        return [], True
    except (FileNotFoundError, ProcessLookupError):
        return [], False


def _fd_paths_hold_dxg(fd_paths: list[Path]) -> tuple[bool, bool]:
    """Return whether an fd targets /dev/dxg and whether an fd link was unreadable."""
    unreadable = False
    for fd_path in fd_paths:
        try:
            target = Path(os.readlink(fd_path))  # noqa: PTH115 - Path.readlink is 3.9+, system Python is 3.8
        except PermissionError:
            unreadable = True
            continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        if target == Path("/dev/dxg"):
            return True, unreadable
    return False, unreadable


def _wsl_process_from_proc(process_dir: Path, pid: int) -> GpuProcess | None:
    """Read the diagnostic fields for a process already found holding /dev/dxg."""
    try:
        comm = (process_dir / "comm").read_text().strip()
    except PermissionError:
        comm = "[unreadable]"
    except (FileNotFoundError, ProcessLookupError):
        return None
    try:
        cmdline = (process_dir / "cmdline").read_bytes().decode(errors="replace")
    except PermissionError:
        cmdline = "[unreadable]"
    except (FileNotFoundError, ProcessLookupError):
        return None

    cmdline = " ".join(part for part in cmdline.split("\0") if part)
    if len(cmdline) > WSL_CMDLINE_MAX_CHARS:
        cmdline = f"{cmdline[:WSL_CMDLINE_MAX_CHARS]}..."
    name = f"/proc/{pid}/comm: {comm}; cmdline: {cmdline}"
    return GpuProcess(pid=pid, used_mib=None, name=name)


def _wsl_dxg_processes(proc_root: Path, own_pid: int) -> tuple[list[GpuProcess], int]:
    """Find other WSL processes with an open /dev/dxg descriptor and count unreadable fd dirs."""
    holders: list[GpuProcess] = []
    unreadable_count = 0
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == own_pid:
            continue
        fd_paths, fd_dir_unreadable = _proc_fd_paths(process_dir)
        if fd_dir_unreadable:
            unreadable_count += 1
            continue
        holds_dxg, fd_link_unreadable = _fd_paths_hold_dxg(fd_paths)
        if fd_link_unreadable:
            unreadable_count += 1
        if not holds_dxg:
            continue
        holder = _wsl_process_from_proc(process_dir, int(process_dir.name))
        if holder is not None:
            holders.append(holder)
    return holders, unreadable_count


def require_free_gpu(
    threshold_mib: int = DEFAULT_BUSY_THRESHOLD_MIB, proc_root: Path = Path("/proc")
) -> None:
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
    if host_is_wsl():
        # WSL's nvidia-smi omits Linux CUDA processes, but each such process holds /dev/dxg open.
        dxg_holders, unreadable_count = _wsl_dxg_processes(proc_root, self_pid)
        logger.info("WSL /dev/dxg unreadable fd process count: %s", unreadable_count)
        if dxg_holders and used <= WSL_DESKTOP_ALLOWANCE_MIB:
            listed = "\n  ".join(str(process) for process in dxg_holders)
            logger.warning(
                f"GPU preflight OK under WSL2: {used} MiB in use is within the "
                f"{WSL_DESKTOP_ALLOWANCE_MIB} MiB Windows desktop allowance, although these "
                f"processes hold /dev/dxg:\n  {listed}"
            )
            return
        if dxg_holders:
            listed = "\n  ".join(str(process) for process in dxg_holders)
            raise GpuBusyError(
                f"GPU is busy: {used}/{total} MiB in use by another process.\n  {listed}\n"
                "Wait for it, or kill it if it is an orphan from a dead run "
                "(inspect the PID and command above)."
            )
        logger.info(f"GPU preflight OK under WSL2, {unattributed} MiB held by the Windows host")
        return
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
