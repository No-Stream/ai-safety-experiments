"""Admit organism turn requests from the served model's live KV occupancy."""

from __future__ import annotations

import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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


def kv_capacity_tokens(server_log: str) -> int:
    """Read the KV pool capacity measured at owned server startup."""
    matches = KV_CAPACITY_LINE.findall(server_log)
    if not matches:
        raise ValueError("owned vLLM server did not report GPU KV cache capacity")
    return int(matches[-1].replace(",", ""))


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
        self._lock = threading.Lock()
        self._active_requests: dict[int, int] = {}
        self._prefix_tokens: dict[int, int] = {}
        self._baseline_usage = 0.0
        self.reset_step()

    def reset_step(self) -> None:
        """Clear measurements before the next rollout step."""
        with self._lock:
            if self._active_requests:
                raise RuntimeError("cannot reset KV admission during a request")
            self._prefix_tokens.clear()
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
                    self._sample(force=True)

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-admission-metrics") as pool:
            future = pool.submit(poll)
            try:
                yield
            finally:
                stop.set()
                future.result()

    @contextmanager
    def request(self, *, prompt_tokens: int, max_tokens: int) -> Generator[None]:
        """Reserve the request and its cached prefix until the episode finishes."""
        if prompt_tokens <= 0 or max_tokens <= 0:
            raise ValueError("admitted request token counts must be positive")
        thread_id = threading.get_ident()
        self._admit(thread_id, prompt_tokens + max_tokens)
        try:
            yield
        finally:
            self._release(thread_id)

    def _admit(self, thread_id: int, requested_prefix_tokens: int) -> None:
        waited_at = 0.0
        while True:
            with self._lock:
                if thread_id in self._active_requests:
                    raise RuntimeError("nested KV admission requests are unsupported")
                usage, running, waiting = self._sample(force=not waited_at)
                if not self._active_requests:
                    self._baseline_usage = usage
                cached_prefix_fraction = (
                    sum(
                        tokens
                        for owner, tokens in self._prefix_tokens.items()
                        if owner not in self._active_requests
                    )
                    / self.capacity_tokens
                )
                active_fraction = sum(self._active_requests.values()) / self.capacity_tokens
                effective_usage = self._baseline_usage + max(
                    usage - self._baseline_usage, active_fraction
                )
                planned_usage = (
                    effective_usage
                    + cached_prefix_fraction
                    + max(requested_prefix_tokens - self._prefix_tokens.get(thread_id, 0), 0)
                    / self.capacity_tokens
                )
                only_request_can_progress = (
                    running == 0
                    and waiting == 0
                    and not self._active_requests
                    and (
                        thread_id in self._prefix_tokens or (usage == 0 and not self._prefix_tokens)
                    )
                )
                if planned_usage <= self.target_usage or only_request_can_progress:
                    self._active_requests[thread_id] = requested_prefix_tokens
                    self._prefix_tokens[thread_id] = max(
                        requested_prefix_tokens, self._prefix_tokens.get(thread_id, 0)
                    )
                    self._max_live_prefix_fraction = max(
                        self._max_live_prefix_fraction,
                        sum(self._prefix_tokens.values()) / self.capacity_tokens,
                    )
                    self._admitted_requests += 1
                    if waited_at:
                        self._wait_seconds += time.monotonic() - waited_at
                    return
                if not waited_at:
                    waited_at = time.monotonic()
                    self._admission_waits += 1
            time.sleep(self.poll_seconds)

    def _release(self, thread_id: int) -> None:
        with self._lock:
            try:
                self._sample(force=True)
            finally:
                del self._active_requests[thread_id]

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
