"""Record a transient service's memory peak from ExecStopPost before cgroup removal."""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def record_memory_peak(
    log_path: Path,
    unit_name: str,
    service_status: Mapping[str, str],
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    membership_path: Path = Path("/proc/self/cgroup"),
) -> None:
    """Append one record, explicitly identifying kernels without readable peak accounting."""
    memory_peak_bytes: int | None = None
    unavailable_reason: str | None = "unified cgroup membership is unavailable"
    for membership in membership_path.read_text().splitlines():
        if membership.startswith("0::"):
            peak_path = cgroup_root / membership[3:].lstrip("/") / "memory.peak"
            try:
                memory_peak_bytes = int(peak_path.read_text().strip())
            except FileNotFoundError:
                unavailable_reason = "memory.peak is unavailable"
            except PermissionError:
                unavailable_reason = "memory.peak is unreadable"
            else:
                if memory_peak_bytes < 0:
                    raise ValueError("memory.peak must be nonnegative")
                unavailable_reason = None
            break
    record = {
        "unit_name": unit_name,
        "memory_peak_bytes": memory_peak_bytes,
        "unavailable_reason": unavailable_reason,
        "service_result": service_status.get("SERVICE_RESULT", ""),
        "exit_code": service_status.get("EXIT_CODE", ""),
        "exit_status": service_status.get("EXIT_STATUS", ""),
    }
    payload = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        # One append syscall keeps concurrent service records together.
        if os.write(descriptor, payload) != len(payload):
            raise OSError("incomplete memory peak record write")
    finally:
        os.close(descriptor)


def main() -> None:
    """Read service termination metadata supplied by systemd."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("unit_name")
    arguments = parser.parse_args()
    record_memory_peak(
        arguments.log_path,
        arguments.unit_name,
        os.environ,
    )


if __name__ == "__main__":
    main()
