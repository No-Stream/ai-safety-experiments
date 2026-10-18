"""Admit turns on an owned server using live KV and active request reservations.

Idle cached blocks are reusable capacity, not live occupancy. Reserve full footprints only
while requests execute, including submissions the metrics have not seen yet. Cached prefixes
must become live again on continuation, so cache hits do not reduce that footprint. Prefix
history only gives continuations scheduling priority; it never reserves capacity during tools.
"""

from __future__ import annotations

import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

import httpx2

if TYPE_CHECKING:
    from collections.abc import Generator

KV_METRIC = "vllm:kv_cache_usage_perc"
RUNNING_METRIC = "vllm:num_requests_running"
WAITING_METRIC = "vllm:num_requests_waiting"
METRIC_LINE = re.compile(r"(?P<name>[\w:]+)(?:\{[^}]*\})?\s+(?P<value>[-+\w.eE]+)(?:\s+\d+)?")
KV_CAPACITY_LINE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


@dataclass
class RequestReservation:
    """The completion count is filled from the server response before releasing KV headroom."""

    wait_seconds: float = 0.0
    completed_tokens: int | None = None


def kv_capacity_tokens(server_log: str) -> int:
    """Read the KV pool capacity measured at owned server startup."""
    matches = KV_CAPACITY_LINE.findall(server_log)
    if not matches:
        raise ValueError("owned vLLM server did not report GPU KV cache capacity")
    return int(matches[-1].replace(",", ""))


def resolve_rollout_concurrency(
    requested: int | None,
    episodes_per_step: int,
    *,
    capacity_tokens: int,
    target_usage: float,
    expected_peak_context_tokens: int,
) -> int:
    """Bound episode fan-out by the KV pool and an expected peak episode context."""
    if min(episodes_per_step, capacity_tokens, expected_peak_context_tokens) < 1:
        raise ValueError("episode count, KV capacity and expected peak context must be positive")
    if requested is not None and requested < 1:
        raise ValueError("requested rollout concurrency must be positive")
    if not 0 < target_usage < 1:
        raise ValueError("KV target usage must be between zero and one")
    kv_limited_fanout = max(
        1, math.floor(capacity_tokens * target_usage / expected_peak_context_tokens)
    )
    requested_fanout = episodes_per_step if requested is None else requested
    return min(episodes_per_step, requested_fanout, kv_limited_fanout)


def _metrics_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("vLLM endpoint must be an absolute HTTP URL")
    return urlunsplit((parsed.scheme, parsed.netloc, "/metrics", "", ""))


def _read_metrics(text: str) -> tuple[float, float, float]:
    values: dict[str, list[float]] = {KV_METRIC: [], RUNNING_METRIC: [], WAITING_METRIC: []}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE.fullmatch(line)
        if match is None:
            if line.startswith((KV_METRIC, RUNNING_METRIC, WAITING_METRIC)):
                raise ValueError(f"malformed vLLM metric: {line[:100]}")
            continue
        name = match.group("name")
        if name in values:
            value = float(match.group("value"))
            if not math.isfinite(value):
                raise ValueError(f"nonfinite vLLM metric: {name}")
            values[name].append(value)
    if any(not samples for samples in values.values()):
        raise ValueError("vLLM metrics missing KV usage, running requests or waiting requests")
    usage = max(values[KV_METRIC])
    running = sum(values[RUNNING_METRIC])
    waiting = sum(values[WAITING_METRIC])
    if not 0 <= usage <= 1 or running < 0 or waiting < 0:
        raise ValueError("vLLM KV usage or request counts outside expected range")
    return usage, running, waiting


def _remaining_timeout(deadline: float | None, maximum: float) -> float:
    if deadline is None:
        return maximum
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("KV admission deadline exceeded")
    return min(remaining, maximum)


class LiveKVAdmission:
    """Poll KV use before each turn and reserve room for requests not yet in metrics.

    A request already executing can grow after admission. The target leaves headroom for
    that growth; vLLM's gauge alone cannot promise that no cached block is ever evicted.
    """

    def __init__(
        self,
        base_url: str,
        *,
        capacity_tokens: int,
        target_usage: float = 0.85,
        poll_seconds: float = 0.1,
        monitor_seconds: float = 1.0,
    ) -> None:
        """Set the live metrics source and KV headroom policy."""
        if (
            capacity_tokens <= 0
            or not 0 < target_usage < 1
            or poll_seconds <= 0
            or monitor_seconds <= 0
        ):
            raise ValueError("KV admission capacity, target and intervals must be positive")
        self.metrics_url = _metrics_url(base_url)
        self.capacity_tokens = capacity_tokens
        self.target_usage = target_usage
        self.poll_seconds = poll_seconds
        self.monitor_seconds = monitor_seconds
        self._lock = threading.Condition()
        self._suspended = False
        self._aborted = False
        self._step_generation = 0
        self._active_requests: dict[int, int] = {}
        self._prefix_tokens: dict[int, int] = {}
        self._waiting_requests: dict[int, int] = {}
        self.reset_step()

    def reset_step(self, *, preserve_pending_requests: bool = False) -> None:
        """Clear step measurements and cached prefixes before the next rollout step."""
        with self._lock:
            if preserve_pending_requests and not self._suspended:
                raise RuntimeError("preserving pending KV requests requires suspended admission")
            if self._active_requests and not preserve_pending_requests:
                raise RuntimeError("cannot reset KV admission during a request")
            if self._waiting_requests and not preserve_pending_requests:
                raise RuntimeError("cannot reset KV admission while requests wait")
            self._prefix_tokens.clear()
            self._step_generation += 1
            self._samples = 0
            self._running_sum = 0.0
            self._max_running = 0.0
            self._kv_sum = 0.0
            self._max_kv = 0.0
            self._admission_waits = 0
            self._wait_seconds = 0.0
            self._admitted_requests = 0
            self._max_live_prefix_fraction = 0.0
            self._last_sample_at = 0.0
            self._last_metrics = (0.0, 0.0, 0.0)

    def suspend(self) -> None:
        """Stop new KV admission and metrics polling while the server is unavailable."""
        with self._lock:
            if self._suspended:
                raise RuntimeError("KV admission is already suspended")
            self._suspended = True
            self._lock.notify_all()

    def resume(self) -> None:
        """Resume KV admission and metrics polling after the server is ready."""
        with self._lock:
            if not self._suspended:
                raise RuntimeError("KV admission is not suspended")
            self._suspended = False
            self._lock.notify_all()

    def abort(self) -> None:
        """Wake episode workers when a training run exits during sampler sleep."""
        with self._lock:
            self._aborted = True
            self._lock.notify_all()

    def _sample(
        self, *, force: bool = False, timeout_seconds: float = 5.0
    ) -> tuple[float, float, float]:
        now = time.monotonic()
        if not force and now - self._last_sample_at < self.poll_seconds:
            return self._last_metrics
        with httpx2.Client(timeout=timeout_seconds, trust_env=False) as client:
            response = client.get(self.metrics_url)
            response.raise_for_status()
            usage, running, waiting = _read_metrics(response.text)
        self._samples += 1
        self._running_sum += running
        self._max_running = max(self._max_running, running)
        self._kv_sum += usage
        self._max_kv = max(self._max_kv, usage)
        self._last_sample_at = time.monotonic()
        self._last_metrics = usage, running, waiting
        return usage, running, waiting

    def _sample_with_admission_deadline(
        self, *, force: bool, deadline: float | None
    ) -> tuple[float, float, float]:
        try:
            return self._sample(force=force, timeout_seconds=_remaining_timeout(deadline, 5.0))
        except httpx2.ReadTimeout as exc:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("KV admission deadline exceeded") from exc
            raise

    @contextmanager
    def monitor_step(self) -> Generator[None]:
        """Sample server load throughout long decodes, including when nothing waits."""
        stop = threading.Event()

        def poll() -> None:
            while not stop.wait(self.monitor_seconds):
                with self._lock:
                    if not self._suspended and not self._aborted:
                        self._sample(force=True)

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-admission-metrics") as pool:
            future = pool.submit(poll)
            try:
                yield
            finally:
                stop.set()
                future.result()

    @contextmanager
    def request(
        self, *, prompt_tokens: int, max_tokens: int, timeout_seconds: float | None = None
    ) -> Generator[RequestReservation]:
        """Reserve an active footprint, with an optional admission deadline."""
        if prompt_tokens <= 0 or max_tokens <= 0:
            raise ValueError("admitted request token counts must be positive")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("KV admission timeout must be positive")
        thread_id = threading.get_ident()
        reservation = RequestReservation()
        try:
            reservation.wait_seconds = self._admit(
                thread_id, prompt_tokens + max_tokens, timeout_seconds=timeout_seconds
            )
        finally:
            with self._lock:
                self._waiting_requests.pop(thread_id, None)
        try:
            yield reservation
        finally:
            self._release(thread_id, prompt_tokens, reservation.completed_tokens)

    def _admit(
        self, thread_id: int, requested_prefix_tokens: int, *, timeout_seconds: float | None
    ) -> float:
        waited_at = 0.0
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        with self._lock:
            observed_step_generation = self._step_generation
        while True:
            with self._lock:
                while self._suspended and not self._aborted:
                    self._lock.wait(timeout=_remaining_timeout(deadline, self.poll_seconds))
                if self._aborted:
                    raise RuntimeError("KV admission has been aborted")
                if observed_step_generation != self._step_generation:
                    observed_step_generation = self._step_generation
                    waited_at = 0.0
                if thread_id in self._active_requests:
                    raise RuntimeError("nested KV admission requests are unsupported")
                usage, running, waiting = self._sample_with_admission_deadline(
                    force=not waited_at, deadline=deadline
                )
                _remaining_timeout(deadline, self.poll_seconds)
                self._waiting_requests[thread_id] = requested_prefix_tokens
                reserved_tokens = sum(self._active_requests.values())
                # On the owned server these are two estimates of the same active workload.
                # Adding them would double count requests already reflected by the gauge.
                planned_usage = (
                    max(usage * self.capacity_tokens, reserved_tokens) + requested_prefix_tokens
                )
                cached_waiters = {
                    owner: delta
                    for owner, delta in self._waiting_requests.items()
                    if owner in self._prefix_tokens
                }
                available_tokens = self.target_usage * self.capacity_tokens - max(
                    usage * self.capacity_tokens, reserved_tokens
                )
                fitting_cached = {
                    owner: delta
                    for owner, delta in cached_waiters.items()
                    if delta <= available_tokens
                }
                has_cached_priority = (
                    thread_id in fitting_cached
                    and requested_prefix_tokens == min(fitting_cached.values())
                ) or not fitting_cached
                no_server_or_local_request = (
                    running == 0 and waiting == 0 and not self._active_requests
                )
                only_request_can_progress = no_server_or_local_request and (
                    requested_prefix_tokens > self.target_usage * self.capacity_tokens
                )
                if has_cached_priority and (
                    planned_usage <= self.target_usage * self.capacity_tokens
                    or only_request_can_progress
                ):
                    self._active_requests[thread_id] = requested_prefix_tokens
                    self._max_live_prefix_fraction = max(
                        self._max_live_prefix_fraction,
                        (reserved_tokens + requested_prefix_tokens) / self.capacity_tokens,
                    )
                    self._admitted_requests += 1
                    if waited_at:
                        duration = time.monotonic() - waited_at
                        self._wait_seconds += duration
                        return duration
                    return 0.0
                if not waited_at:
                    waited_at = time.monotonic()
                    self._admission_waits += 1
                self._lock.wait(timeout=_remaining_timeout(deadline, self.poll_seconds))

    def _release(self, thread_id: int, prompt_tokens: int, completed_tokens: int | None) -> None:
        with self._lock:
            del self._active_requests[thread_id]
            self._prefix_tokens.pop(thread_id, None)
            self._lock.notify_all()
            if completed_tokens is not None:
                if completed_tokens < 0:
                    raise ValueError("completion token count must be nonnegative")
                self._prefix_tokens[thread_id] = prompt_tokens + completed_tokens

    def finish_episode(self) -> None:
        """Forget continuation priority once a worker finishes its episode."""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id in self._active_requests:
                raise RuntimeError("cannot finish an episode during its request")
            self._prefix_tokens.pop(thread_id, None)

    def step_summary(self) -> dict[str, float | int]:
        """Report observed server load and admission delays for this step."""
        with self._lock:
            return {
                "admitted_requests": self._admitted_requests,
                "admission_waits": self._admission_waits,
                "admission_wait_seconds": round(self._wait_seconds, 3),
                "max_running_requests": self._max_running,
                "mean_running_requests": self._running_sum / self._samples
                if self._samples
                else 0.0,
                "max_kv_usage": self._max_kv,
                "mean_kv_usage": self._kv_sum / self._samples if self._samples else 0.0,
                "max_live_prefix_fraction": self._max_live_prefix_fraction,
            }
