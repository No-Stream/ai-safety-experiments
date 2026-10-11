"""Bounded, source-ordered execution for independent grading work."""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from queue import SimpleQueue
from threading import Lock, current_thread, main_thread
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence
    from types import FrameType

GRADING_WORKERS_ENV = "ORGANISM_RL_GRADING_WORKERS"
CGROUP_V2_MOUNT = Path("/sys/fs/cgroup")
PROCESS_CGROUP_FILE = Path("/proc/self/cgroup")
CPU_TOPOLOGY_ROOT = Path("/sys/devices/system/cpu")
GRADING_WORKER_MEMORY_BYTES = 1 << 30
GRADING_CPU_HEADROOM = 4
CGROUP_MEMBERSHIP_FIELD_COUNT = 3
CPU_MAX_FIELD_COUNT = 2
CGROUP_V2_HIERARCHY_ID = "0"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GradingOutcome[T, R]:
    """One completed work item, including failures and elapsed worker time."""

    item: T
    result: R | None
    error: Exception | None
    elapsed_seconds: float


def available_worker_count(
    requested: int | None = None, *, environ: Mapping[str, str] | None = None
) -> int:
    """Choose workers from CPU and memory budgets, reserving CPU and memory headroom."""
    available_cpus = _available_cpu_count()
    capacity = min(
        available_cpus, max(1, _available_memory_bytes() // (2 * GRADING_WORKER_MEMORY_BYTES))
    )
    worker_override = requested
    if worker_override is None:
        environment = os.environ if environ is None else environ
        raw_override = environment.get(GRADING_WORKERS_ENV)
        if raw_override is not None:
            try:
                worker_override = int(raw_override)
            except ValueError as error:
                raise ValueError(
                    f"grading worker override {GRADING_WORKERS_ENV} must be a positive integer"
                ) from error

    if worker_override is None:
        # Two timed gradings on one core's SMT threads can push a correct candidate past its
        # timeout, so the default counts physical cores, not logical CPUs.
        core_budget = min(_physical_core_count(), available_cpus)
        worker_override = min(capacity, max(1, core_budget - GRADING_CPU_HEADROOM))
    if worker_override < 1 or worker_override > capacity:
        raise ValueError(
            f"grading worker count must be between 1 and {capacity} (CPU/memory budget)"
        )
    logger.info(
        "grading pool available_cpus=%d workers=%d headroom_cpus=%d",
        available_cpus,
        worker_override,
        available_cpus - worker_override,
    )
    return worker_override


def run_bounded[T, R](
    items: Sequence[T],
    worker: Callable[[T], R],
    *,
    max_workers: int,
    on_complete: Callable[[GradingOutcome[T, R]], None] | None = None,
) -> list[GradingOutcome[T, R]]:
    """Run in source order; stop submission on failure and drain active work before raising."""
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if not items:
        return []

    cpu_assignments = _worker_cpu_assignments(max_workers)

    outcomes: list[GradingOutcome[T, R] | None] = [None] * len(items)
    elapsed_seconds: list[float | None] = [None] * len(items)
    elapsed_lock = Lock()
    with (
        grading_signal_scope(),
        ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="organism-grader",
            initializer=_pin_grading_worker,
            initargs=(cpu_assignments,),
        ) as pool,
    ):
        future_indexes: dict[Future[R], int] = {}
        next_index = 0
        while next_index < len(items) or future_indexes:
            while next_index < len(items) and len(future_indexes) < max_workers:
                future = pool.submit(
                    copy_context().run,
                    _run_one,
                    next_index,
                    items[next_index],
                    worker,
                    elapsed_seconds,
                    elapsed_lock,
                )
                future_indexes[future] = next_index
                next_index += 1

            completed_futures, _ = wait(future_indexes, return_when=FIRST_COMPLETED)
            for future in sorted(completed_futures, key=future_indexes.__getitem__):
                index = future_indexes.pop(future)
                elapsed = elapsed_seconds[index]
                if elapsed is None:
                    raise RuntimeError("grading worker finished without a timing record")
                outcome = GradingOutcome(
                    item=items[index],
                    result=future.result(),
                    error=None,
                    elapsed_seconds=elapsed,
                )
                outcomes[index] = outcome
                if on_complete is not None:
                    on_complete(outcome)

    if any(outcome is None for outcome in outcomes):
        raise RuntimeError("grading pool finished without an outcome for every item")
    return cast("list[GradingOutcome[T, R]]", outcomes)


def _worker_cpu_assignments(max_workers: int) -> SimpleQueue[int]:
    worker_cpus = sorted(os.sched_getaffinity(0))
    if max_workers > len(worker_cpus):
        raise ValueError("grading workers exceed the available CPU affinity")
    cpu_assignments: SimpleQueue[int] = SimpleQueue()
    for cpu in worker_cpus[:max_workers]:
        cpu_assignments.put(cpu)
    return cpu_assignments


def _pin_grading_worker(cpu_assignments: SimpleQueue[int]) -> None:
    os.sched_setaffinity(0, {cpu_assignments.get()})


def _interrupt_grading(signum: int, _frame: FrameType | None) -> None:
    raise InterruptedError(f"grading interrupted by signal {signum}")


@contextmanager
def grading_signal_scope() -> Generator[None]:
    """Drain active workers on TERM/HUP as executor shutdown already does on Ctrl-C."""
    if current_thread() is not main_thread():
        yield
        return
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        for signum in previous_handlers:
            signal.signal(signum, _interrupt_grading)
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _available_memory_bytes() -> int:
    """Use host available memory and remaining headroom in every visible v2 cgroup."""
    memory_available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
        if line.startswith("MemAvailable:")
    )
    unified_path = _unified_cgroup_path()
    if unified_path is None:
        return memory_available
    mount = CGROUP_V2_MOUNT.resolve(strict=True)
    directory = _cgroup_v2_directory(unified_path, mount)
    while True:
        maximum_path = directory / "memory.max"
        if maximum_path.is_file():
            maximum = maximum_path.read_text(encoding="ascii").strip()
            if maximum != "max":
                current = int((directory / "memory.current").read_text(encoding="ascii"))
                memory_available = min(memory_available, max(0, int(maximum) - current))
        if directory == mount:
            return memory_available
        directory = directory.parent


def _physical_core_count() -> int:
    """Count distinct physical cores among the CPUs this process may run on."""
    return len(
        {
            (CPU_TOPOLOGY_ROOT / f"cpu{cpu}" / "topology" / "core_cpus_list")
            .read_text(encoding="ascii")
            .strip()
            for cpu in os.sched_getaffinity(0)
        }
    )


def _available_cpu_count() -> int:
    """Return CPUs allowed by affinity and the tightest visible cgroup v2 quota."""
    if hasattr(os, "sched_getaffinity"):
        affinity_cpus = max(1, len(os.sched_getaffinity(0)))
    else:
        affinity_cpus = max(1, os.cpu_count() or 1)
    cgroup_cpus = _cgroup_v2_cpu_count()
    return min(affinity_cpus, cgroup_cpus) if cgroup_cpus is not None else affinity_cpus


def _cgroup_v2_cpu_count() -> int | None:
    """Read the minimum CPU quota from the process cgroup through the mount root."""
    if not sys.platform.startswith("linux"):
        return None
    unified_path = _unified_cgroup_path()
    if unified_path is None:
        return None
    cgroup_mount = CGROUP_V2_MOUNT.resolve(strict=True)
    cgroup_directory = _cgroup_v2_directory(unified_path, cgroup_mount)
    return _ancestor_cgroup_cpu_count(cgroup_directory, cgroup_mount)


def _unified_cgroup_path() -> str | None:
    unified_path: str | None = None
    for line in PROCESS_CGROUP_FILE.read_text(encoding="ascii").splitlines():
        fields = line.split(":", maxsplit=2)
        if len(fields) != CGROUP_MEMBERSHIP_FIELD_COUNT:
            raise ValueError(f"{PROCESS_CGROUP_FILE}: malformed cgroup membership row")
        hierarchy_id, controllers, cgroup_path = fields
        if hierarchy_id == CGROUP_V2_HIERARCHY_ID and not controllers:
            if unified_path is not None:
                raise ValueError(f"{PROCESS_CGROUP_FILE}: duplicate cgroup v2 membership row")
            unified_path = cgroup_path
    return unified_path


def _cgroup_v2_directory(unified_path: str, cgroup_mount: Path) -> Path:
    cgroup_path = PurePosixPath(unified_path)
    if not cgroup_path.is_absolute() or ".." in cgroup_path.parts:
        raise ValueError(f"{PROCESS_CGROUP_FILE}: invalid cgroup v2 path")
    cgroup_directory = cgroup_mount.joinpath(*cgroup_path.parts[1:]).resolve(strict=True)
    if not cgroup_directory.is_relative_to(cgroup_mount):
        raise ValueError(f"{PROCESS_CGROUP_FILE}: cgroup v2 path escapes its mount")
    return cgroup_directory


def _ancestor_cgroup_cpu_count(cgroup_directory: Path, cgroup_mount: Path) -> int | None:
    quota_cpu_counts: list[int] = []
    current_directory = cgroup_directory
    while True:
        cpu_max_path = current_directory / "cpu.max"
        try:
            cpu_max = cpu_max_path.read_text(encoding="ascii")
        except FileNotFoundError:
            pass
        else:
            quota_cpu_count = _parse_cpu_max(cpu_max, cpu_max_path)
            if quota_cpu_count is not None:
                quota_cpu_counts.append(quota_cpu_count)
        if current_directory == cgroup_mount:
            break
        current_directory = current_directory.parent
    return min(quota_cpu_counts) if quota_cpu_counts else None


def _parse_cpu_max(cpu_max: str, path: Path) -> int | None:
    fields = cpu_max.split()
    if len(fields) != CPU_MAX_FIELD_COUNT:
        raise ValueError(f"{path}: malformed cgroup v2 cpu.max")
    quota_text, period_text = fields
    if not period_text.isascii() or not period_text.isdecimal() or int(period_text) < 1:
        raise ValueError(f"{path}: malformed cgroup v2 cpu.max period")
    if quota_text == "max":
        return None
    if not quota_text.isascii() or not quota_text.isdecimal() or int(quota_text) < 1:
        raise ValueError(f"{path}: malformed cgroup v2 cpu.max quota")
    return max(1, int(quota_text) // int(period_text))


def _run_one[T, R](
    index: int,
    item: T,
    worker: Callable[[T], R],
    elapsed_seconds: list[float | None],
    elapsed_lock: Lock,
) -> R:
    started_at = time.perf_counter()
    try:
        return worker(item)
    finally:
        with elapsed_lock:
            elapsed_seconds[index] = time.perf_counter() - started_at
