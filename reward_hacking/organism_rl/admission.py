"""Admit organism turns using live KV occupancy and protected cached prefixes.

The server gauge is the authoritative occupancy. Local reservations cover requests the gauge
has not yet seen and preserve idle episodes' last cached sequence across tool calls. Idle
episodes retain only the observed prompt and generated tokens; active requests reserve the
remaining generation allowance as well. Cached turns get priority when space is tight because
their incremental cost is small. If no cached turn fits, the cheapest waiting continuation
eventually runs alone; an overfull pool cannot preserve every prefix and must make progress.
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
STALLED_CACHED_ESCAPE_SECONDS = 0.5


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

    def _sample(self, *, force: bool = False) -> tuple[float, float, float]:
        now = time.monotonic()
        if not force and now - self._last_sample_at < self.poll_seconds:
            return self._last_metrics
        with httpx2.Client(timeout=5.0, trust_env=False) as client:
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
    def request(self, *, prompt_tokens: int, max_tokens: int) -> Generator[RequestReservation]:
        """Reserve the request, then retain only its observed cached sequence."""
        if prompt_tokens <= 0 or max_tokens <= 0:
            raise ValueError("admitted request token counts must be positive")
        thread_id = threading.get_ident()
        reservation = RequestReservation()
        try:
            reservation.wait_seconds = self._admit(thread_id, prompt_tokens + max_tokens)
        finally:
            with self._lock:
                self._waiting_requests.pop(thread_id, None)
        try:
            yield reservation
        finally:
            self._release(thread_id, prompt_tokens, reservation.completed_tokens)

    def _admit(self, thread_id: int, requested_prefix_tokens: int) -> float:
        waited_at = 0.0
        with self._lock:
            observed_step_generation = self._step_generation
        while True:
            with self._lock:
                while self._suspended and not self._aborted:
                    self._lock.wait()
                if self._aborted:
                    raise RuntimeError("KV admission has been aborted")
                if observed_step_generation != self._step_generation:
                    observed_step_generation = self._step_generation
                    waited_at = 0.0
                if thread_id in self._active_requests:
                    raise RuntimeError("nested KV admission requests are unsupported")
                usage, running, waiting = self._sample(force=not waited_at)
                incremental_tokens = max(
                    requested_prefix_tokens - self._prefix_tokens.get(thread_id, 0), 0
                )
                self._waiting_requests[thread_id] = incremental_tokens
                reserved_tokens = sum(self._prefix_tokens.values()) + sum(
                    tokens - self._prefix_tokens.get(owner, 0)
                    for owner, tokens in self._active_requests.items()
                )
                planned_usage = (
                    max(usage * self.capacity_tokens, reserved_tokens) + incremental_tokens
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
                    and incremental_tokens == min(fitting_cached.values())
                ) or not fitting_cached
                no_server_or_local_request = (
                    running == 0 and waiting == 0 and not self._active_requests
                )
                only_request_can_progress = no_server_or_local_request and (
                    (thread_id in self._prefix_tokens and len(self._prefix_tokens) == 1)
                    or (usage == 0 and not self._prefix_tokens)
                    or (
                        thread_id in cached_waiters
                        and not fitting_cached
                        and incremental_tokens == min(cached_waiters.values())
                        and waited_at > 0
                        and time.monotonic() - waited_at >= STALLED_CACHED_ESCAPE_SECONDS
                    )
                )
                if has_cached_priority and (
                    planned_usage <= self.target_usage * self.capacity_tokens
                    or only_request_can_progress
                ):
                    self._active_requests[thread_id] = requested_prefix_tokens
                    self._max_live_prefix_fraction = max(
                        self._max_live_prefix_fraction,
                        (reserved_tokens + incremental_tokens) / self.capacity_tokens,
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
            time.sleep(self.poll_seconds)

    def _release(self, thread_id: int, prompt_tokens: int, completed_tokens: int | None) -> None:
        with self._lock:
            try:
                if not self._aborted:
                    self._sample(force=True)
            finally:
                del self._active_requests[thread_id]
                if completed_tokens is not None:
                    if completed_tokens < 0:
                        raise ValueError("completion token count must be nonnegative")
                    self._prefix_tokens[thread_id] = prompt_tokens + completed_tokens

    def finish_episode(self) -> None:
        """Free the cached-prefix reservation once a worker finishes its episode."""
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
