"""One aggregate cgroup budget for every jail in a training run.

The first rollout dispatch snapshots available RAM after trainer offload and server wake.
Only jail services join this slice; the trainer stays in its existing cgroup.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Self

import psutil

from reward_hacking.jailed_poc import JailResourceLimits
from reward_hacking.organism_rl.phases import (
    DEFAULT_HOST_MEMORY_SAFETY_MARGIN_BYTES,
    HOST_MEMORY_SAFETY_MARGIN_ENV,
)

if TYPE_CHECKING:
    from types import TracebackType

CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
SYSTEMD_TIMEOUT_SECONDS = 30
logger = logging.getLogger(__name__)


def resolve_jail_memory_budget(startup_available_bytes: int, safety_margin_bytes: int) -> int:
    """Reserve the host margin from RAM available after trainer offload."""
    if min(startup_available_bytes, safety_margin_bytes) < 0:
        raise ValueError("Jail available RAM and safety margin must be nonnegative")
    memory_max_bytes = startup_available_bytes - safety_margin_bytes
    if memory_max_bytes <= 0:
        raise ValueError(
            "Jail memory budget requires available RAM above the host safety margin: "
            f"available_bytes={startup_available_bytes} safety_margin_bytes={safety_margin_bytes}"
        )
    return memory_max_bytes


def _systemd_command(argv: list[str]) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed host administration commands
        argv,
        capture_output=True,
        text=True,
        check=True,
        timeout=SYSTEMD_TIMEOUT_SECONDS,
    )
    return completed.stdout.strip()


class JailBudgetSlice:
    """Lazily create and own a transient user slice, shared by rollout workers."""

    def __init__(self, output_dir: Path) -> None:
        """Remember the artifact destination without allocating a cgroup."""
        self.output_dir = output_dir
        # Dashes encode parent slices in systemd, so this name deliberately has none.
        self.slice_name = f"orgjails{uuid.uuid4().hex}.slice"
        self._lock = threading.Lock()
        self._limits: JailResourceLimits | None = None
        self._created = False
        self._closed = False

    def __enter__(self) -> Self:
        """Keep startup lazy until the first rollout."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the slice after all rollout pools have closed."""
        self.close()

    def limits(self) -> JailResourceLimits:
        """Resolve the available-memory budget once, on the first rollout dispatch."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Jail memory budget slice is closed")
            if self._limits is not None:
                return self._limits
            margin_text = os.environ.get(
                HOST_MEMORY_SAFETY_MARGIN_ENV, str(DEFAULT_HOST_MEMORY_SAFETY_MARGIN_BYTES)
            )
            if not margin_text.isascii() or not margin_text.isdecimal():
                raise ValueError(
                    f"{HOST_MEMORY_SAFETY_MARGIN_ENV} must be a nonnegative integer byte count"
                )
            safety_margin_bytes = int(margin_text)
            startup_available_bytes = psutil.virtual_memory().available
            memory_max_bytes = resolve_jail_memory_budget(
                startup_available_bytes, safety_margin_bytes
            )
            ready = False
            try:
                self._start(memory_max_bytes)
                limits = JailResourceLimits(
                    slice_name=self.slice_name,
                    memory_peak_log=self.output_dir / "jail-memory-peaks.jsonl",
                )
                self.output_dir.mkdir(parents=True, exist_ok=True)
                provenance = {
                    "slice_name": self.slice_name,
                    "startup_available_bytes": startup_available_bytes,
                    "safety_margin_bytes": safety_margin_bytes,
                    "memory_max_bytes": memory_max_bytes,
                    "memory_swap_max_bytes": 0,
                    "memory_oom_group": 0,
                    "trainer_excluded": True,
                    "snapshot_phase": "first_rollout_dispatch",
                }
                path = self.output_dir / "startup-jail-memory.jsonl"
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(provenance) + "\n")
                logger.info(
                    "Jail aggregate memory budget slice=%s available_bytes=%d safety_margin_bytes=%d "
                    "memory_max_bytes=%d memory_swap_max_bytes=0 memory_oom_group=0 trainer_excluded=true",
                    self.slice_name,
                    startup_available_bytes,
                    safety_margin_bytes,
                    memory_max_bytes,
                )
                self._limits = limits
                ready = True
                return limits
            finally:
                if not ready:
                    self._stop()

    def _start(self, memory_max_bytes: int) -> None:
        _systemd_command(
            [
                "busctl",
                "--user",
                "call",
                "org.freedesktop.systemd1",
                "/org/freedesktop/systemd1",
                "org.freedesktop.systemd1.Manager",
                "StartTransientUnit",
                "ssa(sv)a(sa(sv))",
                self.slice_name,
                "fail",
                "2",
                "MemoryMax",
                "t",
                str(memory_max_bytes),
                "MemorySwapMax",
                "t",
                "0",
                "0",
            ]
        )
        self._created = True
        # StartTransientUnit returns a job before it completes. systemctl waits for it.
        _systemd_command(["systemctl", "--user", "start", self.slice_name])
        control_group = _systemd_command(
            ["systemctl", "--user", "show", "--property=ControlGroup", "--value", self.slice_name]
        )
        if (
            not control_group.startswith("/")
            or control_group == "/"
            or ".." in Path(control_group).parts
        ):
            raise RuntimeError("Jail slice has an invalid ControlGroup")
        cgroup_directory = CGROUP_V2_ROOT / control_group.removeprefix("/")
        expected_properties = {
            "memory.max": memory_max_bytes,
            "memory.swap.max": 0,
            # systemd has no MemoryOOMGroup property. New slices default to zero;
            # verify the kernel value instead of issuing an unsupported assignment.
            "memory.oom.group": 0,
        }
        for name, expected in expected_properties.items():
            actual = (cgroup_directory / name).read_text(encoding="ascii").strip()
            if actual != str(expected):
                raise RuntimeError(f"Jail slice {name}: expected {expected}, got {actual!r}")

    def _stop(self) -> None:
        if self._created:
            _systemd_command(["systemctl", "--user", "stop", self.slice_name])
            self._created = False

    def close(self) -> None:
        """Stop remaining child services and release the transient slice."""
        with self._lock:
            self._stop()
            self._closed = True
