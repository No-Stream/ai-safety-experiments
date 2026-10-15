"""CPU-side GPU telemetry and a stage throughput gate shared by generation and training.

The decode roof is a simplified weight/KV traffic bound, not a kernel benchmark.
Prometheus metric names follow vLLM v1 metrics/loggers.py; counter resets fail the gate.
No CUDA runtime is imported or initialized by this module.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import pynvml

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from grpo.model_geometry import ModelGeometry

logger = logging.getLogger(__name__)
PERCENT_MAX = 100
MIN_SAMPLES = 2
NON_OVERRIDABLE_BREACHES = frozenset(
    {
        "throughput_missing",
        "throughput_telemetry_unavailable",
        "throughput_sampler_thread_failed",
        "vllm_counter_reset",
    }
)
PEAK_SOURCE = "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf"


@dataclass(frozen=True)
class DevicePeak:
    """Published dense Tensor peak and device memory bandwidth."""

    memory_bandwidth_bytes_s: float
    tensor_flops: float | None
    source: str


def device_peak(device_name: str) -> DevicePeak | None:
    """Dense BF16/FP16 Tensor FLOPs with FP32 accumulation, Appendix A (no sparsity)."""
    return {
        "NVIDIA GeForce RTX 5090": DevicePeak(1792e9, 209.5e12, PEAK_SOURCE),
        "NVIDIA GeForce RTX 4090": DevicePeak(1008e9, 165.2e12, PEAK_SOURCE),
        "NVIDIA GeForce RTX 3090": DevicePeak(936e9, 71.2e12, PEAK_SOURCE),
    }.get(device_name)


@dataclass(frozen=True)
class PerfThresholds:
    """Stage red-flag thresholds, overridable by the caller."""

    decode_fraction: float = 0.4
    sm_utilization: float = 50.0
    estimate_multiplier: float = 2.0

    def __post_init__(self) -> None:
        """Reject nonfinite and physically invalid thresholds."""
        if not all(
            math.isfinite(value)
            for value in (self.decode_fraction, self.sm_utilization, self.estimate_multiplier)
        ):
            raise ValueError("Thresholds must be finite")
        if not 0 <= self.decode_fraction <= 1 or not 0 <= self.sm_utilization <= PERCENT_MAX:
            raise ValueError("Invalid throughput utilization threshold")
        if self.estimate_multiplier <= 0:
            raise ValueError("Estimate multiplier must be positive")


@dataclass(frozen=True)
class GpuSample:
    """One NVML observation, including whether process attribution was available."""

    device_name: str
    sm_utilization: float | None
    memory_utilization: float | None
    power_watts: float | None
    sm_clock_mhz: float | None
    memory_clock_mhz: float | None
    memory_used_mib: float | None
    memory_total_mib: float | None
    process_ids: tuple[int, ...] | None
    process_vram_mib: tuple[tuple[int, float | None], ...] = ()


def read_gpu_sample() -> GpuSample:
    """Read the sole visible GPU via NVML; memory utilization is controller busy time."""
    pynvml.nvmlInit()
    try:
        if pynvml.nvmlDeviceGetCount() != 1:
            raise ValueError("Throughput monitor requires exactly one visible GPU")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        name = name.decode() if isinstance(name, bytes) else name
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        processes: dict[int, int | None] = {}
        attribution_supported = True
        try:
            for process in (
                *pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle),
                *pynvml.nvmlDeviceGetGraphicsRunningProcesses_v3(handle),
            ):
                used_bytes = process.usedGpuMemory
                previous_bytes = processes.get(process.pid)
                if used_bytes is None:
                    processes[process.pid] = None
                elif previous_bytes is None or used_bytes > previous_bytes:
                    processes[process.pid] = used_bytes
        except pynvml.NVMLError as error:
            # pynvml installs the NVML error code dynamically in NVMLError.__new__.
            if error.value != pynvml.NVML_ERROR_NOT_SUPPORTED:  # pyright: ignore[reportAttributeAccessIssue]
                raise
            attribution_supported = False
        process_vram_mib = tuple(
            (pid, None if used_bytes is None else used_bytes / (1024**2))
            for pid, used_bytes in sorted(processes.items())
        )
        return GpuSample(
            device_name=name,
            sm_utilization=float(utilization.gpu),
            memory_utilization=float(utilization.memory),
            power_watts=pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            sm_clock_mhz=float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
            memory_clock_mhz=float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)),
            memory_used_mib=int(memory.used) / (1024**2),
            memory_total_mib=int(memory.total) / (1024**2),
            process_ids=tuple(sorted(processes)) if attribution_supported else None,
            process_vram_mib=process_vram_mib,
        )
    finally:
        pynvml.nvmlShutdown()


def parse_metrics(text: str) -> dict[str, float]:
    """Aggregate engines; retain histogram buckets, sums and counts for later analysis."""
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(
            r"(?P<name>[\w:]+)(?P<labels>\{.*\})?\s+(?P<value>[-+\w.eE]+)(?:\s+\d+)?", line
        )
        if match is None:
            raise ValueError(f"Malformed Prometheus metric: {line[:100]}")
        name = match.group("name")
        labels = match.group("labels") or ""
        if not name.startswith("vllm:"):
            continue
        number = float(match.group("value"))
        if not math.isfinite(number) and not (
            name.endswith("_bucket") and "+Inf" in labels and number == math.inf
        ):
            raise ValueError(f"Nonfinite metric {name}")
        # Buckets cannot be summed across le boundaries. Preserve their label sets verbatim.
        key = f"{name}{labels}" if name.endswith("_bucket") else name
        metrics[key] = metrics.get(key, 0.0) + number
    for name in (
        "generation_tokens_total",
        "prompt_tokens_total",
        "num_requests_running",
        "num_requests_waiting",
        "kv_cache_usage_perc",
        "prefix_cache_hits_total",
        "prefix_cache_queries_total",
        "num_preemptions_total",
        "request_prompt_tokens_sum",
        "request_prompt_tokens_count",
        "time_to_first_token_seconds_sum",
        "time_to_first_token_seconds_count",
        "inter_token_latency_seconds_sum",
        "inter_token_latency_seconds_count",
        "request_prefill_time_seconds_sum",
        "request_prefill_time_seconds_count",
    ):
        if f"vllm:{name}" not in metrics:
            raise ValueError(f"Missing required vLLM metric {name}")
    return metrics


def decode_roofline(
    model: ModelGeometry, peak: DevicePeak, batch: float, mean_context: float
) -> float:
    """Aggregate tokens/s: one weight read plus per-sequence KV reads per decode iteration.

    Recurrent state is read and written once; linear attention has no context-growing KV.
    The traffic estimate omits activation traffic, dequantization work and kernel overhead.
    It assumes every text weight is streamed per batch, including the embedding table;
    actual embedding lookups touch fewer bytes. This is a planning bound, not measured traffic.
    """
    if batch <= 0 or mean_context < 0 or model.weight_bytes <= 0:
        raise ValueError("Invalid decode roofline inputs")
    traffic = model.weight_bytes + batch * (
        mean_context * model.kv_bytes_per_token + 2 * model.recurrent_state_bytes_per_request
    )
    return peak.memory_bandwidth_bytes_s * batch / traffic


def _histogram_quantile(buckets: list[tuple[float, float]], quantile: float) -> float | None:
    if len(buckets) < MIN_SAMPLES or buckets[-1][0] != math.inf or buckets[-1][1] == 0:
        return None
    rank = quantile * buckets[-1][1]
    previous_bound = previous_count = 0.0
    for upper_bound, cumulative_count in buckets:
        if upper_bound < 0 or cumulative_count < previous_count:
            raise ValueError(
                "Latency histogram bounds and counts must be nonnegative and monotonic"
            )
        if cumulative_count >= rank:
            if upper_bound == math.inf:
                return previous_bound
            bucket_count = cumulative_count - previous_count
            if bucket_count == 0:
                return upper_bound
            return (
                previous_bound
                + (upper_bound - previous_bound) * (rank - previous_count) / bucket_count
            )
        previous_bound, previous_count = upper_bound, cumulative_count
    return None


def histogram_summary(metrics: Mapping[str, float], name: str) -> dict[str, float | None]:
    """Summarize a classic latency histogram, aggregating engines by bucket boundary.

    Follow Prometheus histogram_quantile linear interpolation, including its final
    +Inf bucket convention: https://prometheus.io/docs/prometheus/latest/querying/functions/
    Quantiles are bucket estimates; missing or empty buckets produce unknown values.
    """
    buckets_by_bound: dict[float, float] = {}
    for key, value in metrics.items():
        if not key.startswith(f"{name}_bucket{{"):
            continue
        bound_match = re.search(r'[,\{]le="([^"]+)"', key)
        if bound_match is None:
            raise ValueError("Histogram bucket lacks le boundary")
        bound = float(bound_match.group(1))
        buckets_by_bound[bound] = buckets_by_bound.get(bound, 0) + value
    buckets = sorted(buckets_by_bound.items())
    count = metrics.get(f"{name}_count")
    total = metrics.get(f"{name}_sum")
    return {
        "count": count,
        "total_seconds": total,
        "mean_seconds": total / count
        if total is not None and count is not None and count > 0
        else None,
        "p50_seconds": _histogram_quantile(buckets, 0.5),
        "p90_seconds": _histogram_quantile(buckets, 0.9),
    }


def _format_optional(value: float | None, suffix: str = "") -> str:
    return "unknown" if value is None else f"{value:.2f}{suffix}"


def _mean_metric(samples: list[dict[str, Any]], name: str) -> float | None:
    values = [sample["metrics"][name] for sample in samples if name in sample["metrics"]]
    return sum(values) / len(values) if values else None


def _descendant_pids(roots: tuple[int, ...]) -> set[int]:
    """Refresh ancestry on each sample because vLLM workers may start after monitoring."""
    allowed = set(roots)
    parents: dict[int, int] = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = path.read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        fields = stat[stat.rfind(")") + 2 :].split()
        parents[int(path.parent.name)] = int(fields[1])
    while True:
        descendants = {pid for pid, parent in parents.items() if parent in allowed}
        if descendants <= allowed:
            return allowed
        allowed.update(descendants)


def throughput_breaches(output_dir: Path) -> list[str]:
    """Return reasons that must make health assertion fail; explicit acceptance is persisted."""
    path = output_dir / "throughput.json"
    if not path.exists():
        return ["throughput_missing"]
    report: dict[str, Any] = json.loads(path.read_text())
    acceptance = report["accept_slow"]
    if acceptance is not None and (not isinstance(acceptance, str) or not acceptance.strip()):
        raise ValueError("Throughput acceptance must be a nonempty reason string")
    breaches = report["breaches"]
    if not isinstance(breaches, list) or any(not isinstance(item, str) for item in breaches):
        raise ValueError("Throughput breaches must be a list of reason strings")
    if acceptance is None:
        return breaches
    return [reason for reason in breaches if reason in NON_OVERRIDABLE_BREACHES]


class PerfMonitor:
    """A sampling context that always publishes a report, including telemetry failures."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        output_dir: Path,
        metrics_url: str,
        concurrency: int,
        model: ModelGeometry | None = None,
        estimate_seconds: float | None = None,
        accept_slow: str | None = None,
        allowed_gpu_pids: tuple[int, ...] = (),
        interval_seconds: float = 5.0,
        thresholds: PerfThresholds | None = None,
        gpu_reader: Callable[[], GpuSample | None] = read_gpu_sample,
        device_name: str | None = None,
    ) -> None:
        """Configure a stage monitor; construction does not access the device."""
        if (
            concurrency < 1
            or interval_seconds <= 0
            or not math.isfinite(interval_seconds)
            or (estimate_seconds is not None and estimate_seconds <= 0)
            or (estimate_seconds is not None and not math.isfinite(estimate_seconds))
        ):
            raise ValueError("Concurrency, interval and estimate must be positive")
        if accept_slow is not None and not accept_slow.strip():
            raise ValueError("accept_slow requires a nonempty reason")
        self.output_dir = output_dir
        self.metrics_url = metrics_url
        self.concurrency = concurrency
        self.model = model
        self.estimate_seconds = estimate_seconds
        self.accept_slow = accept_slow
        self.allowed_gpu_pids = allowed_gpu_pids
        self.interval_seconds = interval_seconds
        self.thresholds = thresholds or PerfThresholds()
        self.gpu_reader = gpu_reader
        self.device_name = device_name
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.timings: dict[str, float] = {}
        self.mean_context: float | None = None
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._thread_failed = False
        self._started = time.monotonic()

    def add_time(self, category: str, seconds: float) -> None:
        """Accumulate episode work seconds; concurrent episodes may exceed stage wall time."""
        if seconds < 0 or not math.isfinite(seconds):
            raise ValueError("Timing must be finite and nonnegative")
        self.timings[category] = self.timings.get(category, 0) + seconds

    def set_mean_context(self, tokens: float) -> None:
        """Set an observed mean live sequence length from the caller's token usage."""
        if tokens < 0 or not math.isfinite(tokens):
            raise ValueError("Context must be finite and nonnegative")
        self.mean_context = tokens

    def record_sample(
        self, seconds: float, metrics: Mapping[str, float], gpu: GpuSample | None
    ) -> None:
        """Persist an observation immediately so interrupted stages retain raw telemetry."""
        allowed = _descendant_pids(self.allowed_gpu_pids) if self.allowed_gpu_pids else set()
        if (
            not math.isfinite(seconds)
            or seconds < 0
            or (self.samples and seconds <= self.samples[-1]["seconds"])
        ):
            raise ValueError("Sample times must be finite, nonnegative and increasing")
        process_ids = gpu.process_ids if gpu is not None else None
        process_vram = dict(gpu.process_vram_mib) if gpu is not None else {}
        foreign_pids = sorted(set(process_ids or ()) - allowed)
        foreign_processes: list[dict[str, int | float]] = []
        for pid in foreign_pids:
            vram_mib = process_vram.get(pid)
            if vram_mib is not None and vram_mib > 0:
                foreign_processes.append({"pid": pid, "vram_mib": vram_mib})
        sample = {
            "seconds": seconds,
            "metrics": dict(metrics),
            "gpu": asdict(gpu) if gpu is not None else None,
            "device_name": gpu.device_name if gpu is not None else self.device_name,
            "foreign_gpu_pids": [process["pid"] for process in foreign_processes],
            "foreign_gpu_processes": foreign_processes,
            "foreign_gpu_memory_unknown_pids": [
                pid for pid in foreign_pids if process_vram.get(pid) is None
            ],
            "gpu_process_ownership_unknown": process_ids is None
            or (
                gpu is not None
                and gpu.memory_used_mib is not None
                and gpu.memory_used_mib > 0
                and not process_ids
            ),
        }
        with self._condition:
            self.samples.append(sample)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "throughput-samples.jsonl").open("a") as stream:
                stream.write(json.dumps(sample) + "\n")
            self._condition.notify_all()

    def wait_for_samples(self, count: int, timeout: float) -> bool:
        """Wait for telemetry without polling; primarily useful for CPU integration tests."""
        with self._condition:
            return self._condition.wait_for(lambda: len(self.samples) >= count, timeout)

    def _sample(self) -> None:
        try:
            with urllib.request.urlopen(self.metrics_url, timeout=3) as response:  # noqa: S310
                metrics = parse_metrics(response.read().decode())
            self.record_sample(time.monotonic() - self._started, metrics, self.gpu_reader())
        except (OSError, ValueError, pynvml.NVMLError, urllib.error.URLError) as error:
            self.errors.append(f"{type(error).__name__}: {error}")
            logger.exception("Throughput telemetry failed")

    def _run(self) -> None:
        stopped_by_request = False
        try:
            while True:
                if self._stop.wait(self.interval_seconds):
                    stopped_by_request = True
                    break
                self._sample()
        finally:
            if not stopped_by_request:
                self._thread_failed = True
                self.errors.append("sampler thread exited unexpectedly")

    def __enter__(self) -> Self:
        """Sample immediately and start the periodic reader."""
        self._started = time.monotonic()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-perf")
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop sampling and publish the report even when generation raises."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        self.finish(time.monotonic() - self._started)

    def finish(self, elapsed_seconds: float) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
        """Finalize once sampling has stopped. Durations are interval weighted, not sample counts."""
        generation_wall_seconds = elapsed_seconds
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("Elapsed seconds must be finite and nonnegative")
        elapsed_seconds += self.timings.get("server_startup", 0)
        breaches: list[str] = []
        if self.errors or len(self.samples) < MIN_SAMPLES:
            breaches.append("throughput_telemetry_unavailable")
        if self._thread_failed:
            breaches.append("throughput_sampler_thread_failed")
        generation = prompt = measured = active = sm_integral = roof_capacity = tail = (
            sm_observed_seconds
        ) = 0.0
        running_integral = decode_tokens = decode_seconds = 0.0
        hardware_integrals: dict[str, float] = {}
        hardware_durations: dict[str, float] = {}
        context = self.mean_context
        context_source = (
            "episode_token_usage" if context is not None else "completed_request_prompt_histogram"
        )
        if context is None and len(self.samples) >= MIN_SAMPLES:
            first, last = self.samples[0]["metrics"], self.samples[-1]["metrics"]
            count = last.get("vllm:request_prompt_tokens_count", 0) - first.get(
                "vllm:request_prompt_tokens_count", 0
            )
            if count > 0:
                context = (
                    last["vllm:request_prompt_tokens_sum"] - first["vllm:request_prompt_tokens_sum"]
                ) / count
        for previous, current in zip(self.samples, self.samples[1:], strict=False):
            duration = current["seconds"] - previous["seconds"]
            if duration <= 0:
                raise ValueError("Sample times must increase")
            metrics, next_metrics = previous["metrics"], current["metrics"]
            generated = (
                next_metrics["vllm:generation_tokens_total"]
                - metrics["vllm:generation_tokens_total"]
            )
            prompted = (
                next_metrics["vllm:prompt_tokens_total"] - metrics["vllm:prompt_tokens_total"]
            )
            if generated < 0 or prompted < 0:
                breaches.append("vllm_counter_reset")
                continue
            generation += generated
            prompt += prompted
            measured += duration
            running = (
                metrics["vllm:num_requests_running"] + next_metrics["vllm:num_requests_running"]
            ) / 2
            pending = running + max(
                metrics["vllm:num_requests_waiting"], next_metrics["vllm:num_requests_waiting"]
            )
            if running < self.concurrency / 2:
                tail += duration
            previous_gpu = previous["gpu"] or {}
            current_gpu = current["gpu"] or {}
            peak = device_peak(previous["device_name"] or "")
            if pending > 0:
                active += duration
                before_sm, after_sm = (
                    previous_gpu.get("sm_utilization"),
                    current_gpu.get("sm_utilization"),
                )
                if before_sm is not None and after_sm is not None:
                    sm_integral += duration * (before_sm + after_sm) / 2
                    sm_observed_seconds += duration
                running_integral += running * duration
                for name in (
                    "memory_utilization",
                    "power_watts",
                    "sm_clock_mhz",
                    "memory_clock_mhz",
                    "memory_used_mib",
                ):
                    before, after = previous_gpu.get(name), current_gpu.get(name)
                    if before is not None and after is not None:
                        hardware_integrals[name] = (
                            hardware_integrals.get(name, 0) + duration * (before + after) / 2
                        )
                        hardware_durations[name] = hardware_durations.get(name, 0) + duration
            if (
                prompted == 0
                and generated > 0
                and running > 0
                and self.model is not None
                and peak is not None
                and context is not None
            ):
                roof_capacity += duration * decode_roofline(self.model, peak, running, context)
                decode_tokens += generated
                decode_seconds += duration
        rate = generation / measured if measured else None
        fraction = decode_tokens / roof_capacity if roof_capacity else None
        sm = sm_integral / sm_observed_seconds if sm_observed_seconds else None
        if fraction is None:
            breaches.append("decode_roofline_unknown")
        elif fraction < self.thresholds.decode_fraction:
            breaches.append("decode_bandwidth_roofline_low")
        if sm is not None and sm < self.thresholds.sm_utilization:
            breaches.append("gpu_sm_utilization_low")
        if active > sm_observed_seconds:
            breaches.append("gpu_utilization_unknown")
        if (
            self.estimate_seconds is not None
            and elapsed_seconds > self.estimate_seconds * self.thresholds.estimate_multiplier
        ):
            breaches.append("stage_estimate_exceeded")
        if any(sample["foreign_gpu_pids"] for sample in self.samples):
            breaches.append("foreign_gpu_process")
        if any(sample["gpu_process_ownership_unknown"] for sample in self.samples):
            breaches.append("gpu_process_attribution_unknown")
        if any(sample["foreign_gpu_memory_unknown_pids"] for sample in self.samples):
            breaches.append("foreign_gpu_memory_unknown")
        device_name = next(
            (sample["device_name"] for sample in self.samples if sample["device_name"]),
            self.device_name,
        )
        peak = device_peak(device_name or "")
        counter_deltas: dict[str, float] = {}
        histograms: dict[str, dict[str, float | None]] = {}
        if len(self.samples) >= MIN_SAMPLES:
            first_metrics, last_metrics = self.samples[0]["metrics"], self.samples[-1]["metrics"]
            for name in ("prefix_cache_hits", "prefix_cache_queries", "num_preemptions"):
                metric = f"vllm:{name}_total"
                if metric not in first_metrics or metric not in last_metrics:
                    continue
                delta = last_metrics[metric] - first_metrics[metric]
                if delta < 0:
                    breaches.append("vllm_counter_reset")
                else:
                    counter_deltas[name] = delta
            for name in (
                "time_to_first_token_seconds",
                "inter_token_latency_seconds",
                "request_prefill_time_seconds",
                "request_decode_time_seconds",
            ):
                metric_name = f"vllm:{name}"
                delta_metrics: dict[str, float] = {}
                if (
                    f"{metric_name}_sum" not in first_metrics
                    or f"{metric_name}_sum" not in last_metrics
                    or f"{metric_name}_count" not in first_metrics
                    or f"{metric_name}_count" not in last_metrics
                ):
                    continue
                reset = False
                for key in first_metrics.keys() | last_metrics.keys():
                    if key not in (
                        f"{metric_name}_sum",
                        f"{metric_name}_count",
                    ) and not key.startswith(f"{metric_name}_bucket{{"):
                        continue
                    delta = last_metrics.get(key, 0.0) - first_metrics.get(key, 0.0)
                    if delta < 0:
                        reset = True
                    else:
                        delta_metrics[key] = delta
                if reset:
                    breaches.append("vllm_counter_reset")
                else:
                    histograms[name] = histogram_summary(delta_metrics, metric_name)
        prefill_flops = 2 * self.model.parameter_count * prompt if self.model is not None else 0
        prefill_seconds = histograms.get("request_prefill_time_seconds", {}).get("total_seconds")
        compute_roof = (
            peak.tensor_flops / (2 * self.model.parameter_count)
            if peak is not None and peak.tensor_flops and self.model is not None
            else None
        )
        bandwidth_roof = roof_capacity / decode_seconds if decode_seconds else None
        report: dict[str, Any] = {
            "schema_version": 1,
            "elapsed_seconds": elapsed_seconds,
            "generation_wall_seconds": generation_wall_seconds,
            "device_name": device_name or "unknown",
            "estimate_seconds": self.estimate_seconds,
            "accept_slow": self.accept_slow,
            "breaches": sorted(set(breaches)),
            "thresholds": asdict(self.thresholds),
            "samples": self.samples,
            "telemetry_errors": self.errors,
            "time_split_seconds": self.timings,
            "straggler_tail_seconds": tail,
            "pending_seconds": active,
            "sampled_seconds": measured,
            "generation_tokens": generation,
            "prompt_tokens": prompt,
            "decode_tokens_per_second": rate,
            "vllm_counter_deltas": counter_deltas,
            "vllm_histograms": histograms,
            "prompt_tokens_per_second": prompt / measured if measured else None,
            "mean_running_requests": running_integral / active if active else None,
            "mean_gpu_sm_utilization_pending": sm,
            "pending_hardware_means": {
                name: total / hardware_durations[name] for name, total in hardware_integrals.items()
            },
            "mean_context_tokens": context,
            "context_source": context_source,
            "decode_roofline_fraction": fraction,
            "decode_roofline_tokens_per_second": bandwidth_roof,
            "decode_compute_roofline_tokens_per_second": compute_roof,
            "decode_limiting_roof": "memory_bandwidth"
            if bandwidth_roof is not None
            and compute_roof is not None
            and bandwidth_roof <= compute_roof
            else "compute"
            if bandwidth_roof is not None and compute_roof is not None
            else "unknown",
            "decode_only_seconds": decode_seconds,
            "decode_only_tokens": decode_tokens,
            "prefill_request_seconds": prefill_seconds,
            "prefill_flop_peak_fraction": prefill_flops / (prefill_seconds * peak.tensor_flops)
            if prefill_seconds is not None
            and prefill_seconds > 0
            and peak is not None
            and peak.tensor_flops is not None
            else None,
            "prefill_flop_peak_fraction_stage_average_lower_bound": prefill_flops
            / (elapsed_seconds * peak.tensor_flops)
            if elapsed_seconds and peak is not None and peak.tensor_flops is not None
            else None,
            "regime": "mixed_prefill_decode"
            if prompt and generation
            else "decode"
            if generation
            else "no_decode",
            "model_geometry": asdict(self.model) if self.model is not None else None,
            "device_peak": asdict(peak) if peak is not None else None,
            "limitations": [
                "SM utilization and memory controller busy time are NVML proxies, not achieved FLOPs or bandwidth",
                (
                    "Prefill FLOP fraction uses request_prefill_time_seconds; overlapping request-seconds "
                    "are an approximation, not hardware utilization"
                ),
                "Stage-average prefill fraction is a lower bound because it divides by the full stage wall time",
                "Prefill attention FLOPs are omitted",
                "Roof gates use decode-only intervals; intervals with simultaneous prefill are excluded",
                "All text weights are assumed streamed per batch; full embedding-table traffic is overcounted",
                "Low concurrency seconds include ramp-up, idle and straggler tail",
                "Prompt histogram context excludes generated history unless episode token usage is supplied",
            ],
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "throughput.json").write_text(json.dumps(report, indent=2) + "\n")
        latency = report["vllm_histograms"]
        kv_cache_use = _mean_metric(self.samples, "vllm:kv_cache_usage_perc")
        kv_cache_percent = kv_cache_use * 100 if kv_cache_use is not None else None
        bandwidth_percent = fraction * 100 if fraction is not None else None
        prefill_fraction = report["prefill_flop_peak_fraction"]
        prefill_stage_average_fraction = report[
            "prefill_flop_peak_fraction_stage_average_lower_bound"
        ]
        prefill_percent = prefill_fraction * 100 if prefill_fraction is not None else None
        prefill_stage_average_percent = (
            prefill_stage_average_fraction * 100
            if prefill_stage_average_fraction is not None
            else None
        )
        memory_busy = report["pending_hardware_means"].get("memory_utilization")
        ttft = latency.get("time_to_first_token_seconds", {})
        inter_token = latency.get("inter_token_latency_seconds", {})
        table = (
            "\n## Throughput\n\n| Measure | Value |\n| --- | --- |\n"
            f"| GPU | {report['device_name']} |\n"
            f"| Decode tokens/s | {_format_optional(rate)} |\n"
            f"| Decode bandwidth roof fraction | {_format_optional(bandwidth_percent, '%')} |\n"
            f"| Decode limiting roof | {report['decode_limiting_roof']} |\n"
            f"| Prefill FLOP peak fraction (request-time approximation) | {_format_optional(prefill_percent, '%')} |\n"
            f"| Prefill FLOP peak fraction (stage-average lower bound) | "
            f"{_format_optional(prefill_stage_average_percent, '%')} |\n"
            f"| Prompt tokens / generated tokens | {prompt:.0f} / {generation:.0f} |\n"
            f"| Mean GPU SM utilization while pending | {_format_optional(sm, '%')} |\n"
            f"| Mean memory controller busy while pending | {_format_optional(memory_busy, '%')} |\n"
            f"| Mean KV cache use | {_format_optional(kv_cache_percent, '%')} |\n"
            f"| Mean TTFT / P50 | {_format_optional(ttft.get('mean_seconds'))} / "
            f"{_format_optional(ttft.get('p50_seconds'))} s |\n"
            f"| Mean inter-token latency / P50 | {_format_optional(inter_token.get('mean_seconds'))} / "
            f"{_format_optional(inter_token.get('p50_seconds'))} s |\n"
            f"| Prefix cache hits / queries | {counter_deltas.get('prefix_cache_hits', 'unknown')} / "
            f"{counter_deltas.get('prefix_cache_queries', 'unknown')} |\n"
            f"| Preemptions | {counter_deltas.get('num_preemptions', 'unknown')} |\n"
            f"| Wall seconds | {elapsed_seconds:.1f} |\n"
            f"| Straggler tail seconds | {tail:.1f} |\n| Gate reasons | {', '.join(report['breaches']) or 'none'} |\n"
            f"| Accepted slow reason | {self.accept_slow or 'none'} |\n"
        )
        for category, seconds in self.timings.items():
            unit = (
                "wall seconds"
                if category == "server_startup"
                else "episode work seconds (overlap possible)"
            )
            table += f"| {category} ({unit}) | {seconds:.1f} |\n"
        hardware_labels = {
            "memory_utilization": "Memory controller busy (%)",
            "power_watts": "Power (W)",
            "sm_clock_mhz": "SM clock (MHz)",
            "memory_clock_mhz": "Memory clock (MHz)",
            "memory_used_mib": "VRAM used (MiB)",
        }
        for name, value in report["pending_hardware_means"].items():
            table += f"| Mean {hardware_labels[name]} while pending | {value:.1f} |\n"
        with (self.output_dir / "summary.md").open("a") as stream:
            stream.write(table)
        logger.info("THROUGHPUT SUMMARY\n%s", table)
        if breaches:
            logger.warning(
                "\nTHROUGHPUT ALERT\nReasons: %s\nDecode tokens/s: %s; roof fraction: %s; pending GPU SM: %s\n"
                "Acceptance: %s\n",
                ", ".join(report["breaches"]),
                _format_optional(rate),
                _format_optional(fraction * 100 if fraction is not None else None, "%"),
                _format_optional(sm, "%"),
                self.accept_slow or "REQUIRED",
            )
        return report
