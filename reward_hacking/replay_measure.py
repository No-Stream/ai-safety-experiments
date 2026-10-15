"""Replay exact vLLM prompts over HTTP and measure streaming decode throughput."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import pynvml

logger = logging.getLogger(__name__)
GPU_SAMPLE_INTERVAL_SECONDS = 1.0
PROMETHEUS_METRICS = (
    "vllm:request_prefill_time_seconds_sum",
    "vllm:prompt_tokens_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:num_preemptions_total",
    "vllm:request_prefill_kv_computed_tokens_sum",
)
PROMETHEUS_LINE = re.compile(r"(?P<name>[\w:]+)(?:\{.*\})?\s+(?P<value>[-+\w.eE]+)(?:\s+\d+)?")
OUTPUT_FILENAMES = (
    "metrics_before.prom",
    "metrics_after.prom",
    "nvml_samples.jsonl",
    "request_results.jsonl",
    "cpu_telemetry.json",
    "summary.json",
)


@dataclass(frozen=True)
class _RequestRuntime:
    client: httpx.AsyncClient
    completion_url: str
    model: str
    cell_started: float
    output_path: Path
    output_lock: asyncio.Lock


@dataclass
class _StreamState:
    prompt: list[int]
    request_started: float
    cell_started: float
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] = field(default_factory=list)
    token_events: list[dict[str, Any]] = field(default_factory=list)
    last_token_event_time: float | None = None
    last_token_event_count: int = 0
    first_token_time: float | None = None
    last_token_time: float | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    done_seen: bool = False


@dataclass(frozen=True)
class _CellContext:
    client: httpx.AsyncClient
    completion_url: str
    metrics_url: str
    model: str
    payloads: list[dict[str, Any]]
    output_paths: Mapping[str, Path]


@dataclass(frozen=True)
class _CellMeasurement:
    records: list[dict[str, Any]]
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    wall_seconds: float


def _validate_payload(
    payload: Mapping[str, Any], request_index: int
) -> tuple[list[int], dict[str, Any]]:
    prompt = payload.get("prompt")
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(type(token_id) is not int or token_id < 0 for token_id in prompt)
    ):
        raise ValueError(
            f"payload {request_index} must contain nonempty non-negative prompt token ids"
        )
    if type(payload.get("max_tokens")) is not int or payload["max_tokens"] < 1:
        raise ValueError(f"payload {request_index} must contain a positive max_tokens integer")
    if payload.get("n", 1) != 1:
        raise ValueError("replay measurement requires one completion per prompt")
    return prompt, dict(payload)


def _completion_url(base_url: str) -> str:
    parsed = urlsplit(base_url.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not include a query or fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path += "/completions"
    else:
        path += "/v1/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _metrics_url(base_url: str) -> str:
    parsed = urlsplit(base_url.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not include a query or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, "/metrics", "", ""))


def _parse_metrics(metrics_text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    wanted_metrics = frozenset(PROMETHEUS_METRICS)
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_LINE.fullmatch(line)
        if match is None:
            if any(line.startswith(name) for name in wanted_metrics):
                raise ValueError(f"Malformed required Prometheus metric: {line[:120]}")
            continue
        metric_name = match.group("name")
        if metric_name not in wanted_metrics:
            continue
        metric_value = float(match.group("value"))
        if not math.isfinite(metric_value):
            raise ValueError(f"Required Prometheus metric {metric_name} is non-finite")
        values[metric_name] = values.get(metric_name, 0.0) + metric_value
    missing_metrics = wanted_metrics - values.keys()
    if missing_metrics:
        raise ValueError(
            f"vLLM metrics endpoint omitted required metrics: {sorted(missing_metrics)}"
        )
    return values


async def _fetch_metrics(
    client: httpx.AsyncClient, metrics_url: str
) -> tuple[str, dict[str, float]]:
    response = await client.get(metrics_url)
    response.raise_for_status()
    if response.status_code != HTTPStatus.OK:
        raise ValueError(
            f"vLLM metrics endpoint returned HTTP {response.status_code}, expected {HTTPStatus.OK}"
        )
    metrics_text = response.text
    return metrics_text, _parse_metrics(metrics_text)


def _counter_deltas(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    deltas = {name: after[name] - before[name] for name in PROMETHEUS_METRICS}
    for name, value in deltas.items():
        if value < 0:
            raise ValueError(f"vLLM metric counter reset during replay: {name}")
    return deltas


def _read_cgroup_cpu_stat() -> dict[str, int] | None:
    """Read the current process's cgroup v2 CPU accounting when available."""
    cgroup_memberships = Path("/proc/self/cgroup").read_text().splitlines()
    unified_membership = next(
        (line[3:] for line in cgroup_memberships if line.startswith("0::")), None
    )
    if unified_membership is None:
        return None
    cpu_stat_path = Path("/sys/fs/cgroup") / unified_membership.lstrip("/") / "cpu.stat"
    try:
        cpu_stat_text = cpu_stat_path.read_text()
    except FileNotFoundError:
        return None
    cpu_stat: dict[str, int] = {}
    for line in cpu_stat_text.splitlines():
        name, value = line.split()
        cpu_stat[name] = int(value)
    return cpu_stat


def _cpu_stat_delta(
    before: Mapping[str, int] | None, after: Mapping[str, int] | None
) -> dict[str, Any]:
    if before is None or after is None:
        return {"status": "unavailable", "before": before, "after": after, "delta": None}
    if before.keys() != after.keys():
        raise ValueError("cgroup cpu.stat fields changed during replay")
    delta = {name: after[name] - value for name, value in before.items()}
    if any(value < 0 for value in delta.values()):
        raise ValueError("cgroup cpu.stat counters reset during replay")
    return {"status": "available", "before": dict(before), "after": dict(after), "delta": delta}


def _is_wsl() -> bool:
    kernel_release = Path("/proc/sys/kernel/osrelease").read_text().lower()
    return "microsoft" in kernel_release or "wsl" in kernel_release


def _read_nvml_sample() -> dict[str, Any]:
    """Sample utilization through NVML without estimating memory bandwidth."""
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        device_name = pynvml.nvmlDeviceGetName(handle)
        device_name = device_name.decode() if isinstance(device_name, bytes) else device_name
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "status": "available",
            "device_name": device_name,
            "sm_utilization_pct": float(utilization.gpu),
            "memory_controller_busy_pct": float(utilization.memory),
            "power_watts": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            "sm_clock_mhz": float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
            "memory_clock_mhz": float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)),
            "memory_used_mib": int(memory.used) / (1024**2),
            "memory_total_mib": int(memory.total) / (1024**2),
        }
    except pynvml.NVMLError as error:
        unsupported_codes = {
            pynvml.NVML_ERROR_NOT_SUPPORTED,
            pynvml.NVML_ERROR_FUNCTION_NOT_FOUND,
        }
        # pynvml installs value dynamically in NVMLError.__new__.
        if error.value not in unsupported_codes:  # pyright: ignore[reportAttributeAccessIssue]
            raise
        return {
            "status": "unsupported_wsl" if _is_wsl() else "unsupported",
            "reason": "NVML utilization sampling is unsupported",
            "error": str(error),
        }
    finally:
        pynvml.nvmlShutdown()


async def _append_jsonl(path: Path, lock: asyncio.Lock, record: Mapping[str, Any]) -> None:
    serialized = json.dumps(record, sort_keys=True, allow_nan=False)
    async with lock:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
            stream.flush()


async def _record_gpu_sample(
    output_path: Path,
    output_lock: asyncio.Lock,
    cell_started: float,
) -> dict[str, Any]:
    sample = await asyncio.to_thread(_read_nvml_sample)
    record = {"elapsed_seconds": time.perf_counter() - cell_started, **sample}
    await _append_jsonl(output_path, output_lock, record)
    return sample


async def _sample_gpu_telemetry(
    output_path: Path,
    output_lock: asyncio.Lock,
    cell_started: float,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=GPU_SAMPLE_INTERVAL_SECONDS)
        except TimeoutError:
            sample = await _record_gpu_sample(output_path, output_lock, cell_started)
            if sample["status"] != "available":
                return


def _validate_token_ids(value: object, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"stream field {field_name} must be a list")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ValueError(f"stream field {field_name} must be a list of non-negative token ids")
    return value


def _validate_usage(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("vLLM stream usage chunk must contain a usage object")
    prompt_tokens = value.get("prompt_tokens")
    completion_tokens = value.get("completion_tokens")
    if (
        type(prompt_tokens) is not int
        or prompt_tokens < 0
        or type(completion_tokens) is not int
        or completion_tokens < 0
    ):
        raise ValueError(
            "vLLM stream usage must contain non-negative prompt and completion token counts"
        )
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def _request_payload(
    payload: Mapping[str, Any], request_index: int, model: str
) -> tuple[list[int], str | int, dict[str, Any]]:
    prompt, body = _validate_payload(payload, request_index)
    prompt_id: str | int = body.pop("prompt_id", request_index)
    body.pop("request_id", None)
    body.update(
        {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "echo": False,
            "stream": True,
            "return_token_ids": True,
        }
    )
    stream_options = body.get("stream_options", {})
    if not isinstance(stream_options, Mapping):
        raise TypeError(f"payload {request_index} stream_options must be a mapping")
    body["stream_options"] = {**stream_options, "include_usage": True}
    return prompt, prompt_id, body


def _parse_sse_line(line: str, state: _StreamState) -> Mapping[str, Any] | None:
    if not line:
        return None
    if not line.startswith("data: "):
        raise ValueError(f"Unexpected vLLM stream line: {line[:120]}")
    data = line[6:]
    if state.done_seen:
        raise ValueError("vLLM stream emitted data after the DONE sentinel")
    if data == "[DONE]":
        state.done_seen = True
        return None
    try:
        event = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError("vLLM stream emitted invalid JSON") from error
    if not isinstance(event, Mapping):
        raise TypeError("vLLM stream event must be an object")
    return event


def _record_prompt_ids(choice: Mapping[str, Any], state: _StreamState) -> None:
    if "prompt_token_ids" not in choice or choice["prompt_token_ids"] is None:
        return
    prompt_ids = _validate_token_ids(choice["prompt_token_ids"], "prompt_token_ids")
    if state.prompt_token_ids is not None and prompt_ids != state.prompt_token_ids:
        raise ValueError("vLLM stream changed prompt token ids between chunks")
    state.prompt_token_ids = prompt_ids


def _record_token_delta(choice: Mapping[str, Any], state: _StreamState) -> None:
    if "token_ids" not in choice or choice["token_ids"] is None:
        raise ValueError("vLLM stream omitted delta token_ids despite return_token_ids=True")
    token_ids = _validate_token_ids(choice["token_ids"], "token_ids")
    if not token_ids:
        return
    event_time = time.perf_counter()
    if state.first_token_time is None:
        state.first_token_time = event_time
    interval_seconds = (
        None if state.last_token_event_time is None else event_time - state.last_token_event_time
    )
    state.token_events.append(
        {
            "elapsed_seconds": event_time - state.cell_started,
            "token_count": len(token_ids),
            "coalesced_tokens": len(token_ids) > 1,
            "inter_token_event_seconds": interval_seconds,
            "per_token_interval_seconds": interval_seconds
            if interval_seconds is not None
            and len(token_ids) == 1
            and state.last_token_event_count == 1
            else None,
        }
    )
    state.last_token_event_time = event_time
    state.last_token_event_count = len(token_ids)
    state.last_token_time = event_time
    state.completion_token_ids.extend(token_ids)


def _record_finish_reason(choice: Mapping[str, Any], state: _StreamState) -> None:
    finish_reason = choice.get("finish_reason")
    if finish_reason is None:
        return
    if finish_reason == "error":
        raise ValueError("vLLM stream ended with an error finish_reason")
    if state.finish_reason is not None and state.finish_reason != finish_reason:
        raise ValueError("vLLM stream changed finish_reason between chunks")
    state.finish_reason = str(finish_reason)


def _consume_choice(choice: Mapping[str, Any], state: _StreamState) -> None:
    if choice.get("index") != 0:
        raise ValueError("vLLM stream returned a completion index other than zero")
    _record_prompt_ids(choice, state)
    _record_token_delta(choice, state)
    _record_finish_reason(choice, state)


def _consume_stream_event(event: Mapping[str, Any], state: _StreamState) -> None:
    if event.get("error") is not None:
        raise ValueError(f"vLLM stream returned an error: {event['error']}")
    if event.get("usage") is not None:
        if state.usage is not None:
            raise ValueError("vLLM stream emitted more than one usage chunk")
        state.usage = _validate_usage(event["usage"])
    choices = event.get("choices")
    if not isinstance(choices, list):
        raise TypeError("vLLM stream event must contain a choices list")
    if not choices:
        return
    if len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ValueError("vLLM stream must contain exactly one completion choice")
    _consume_choice(choices[0], state)


def _validated_stream_state(
    state: _StreamState,
) -> tuple[list[int], dict[str, int], str, float, float]:
    if not state.done_seen:
        raise ValueError("vLLM stream ended before the DONE sentinel; response may be truncated")
    if state.finish_reason is None:
        raise ValueError("vLLM stream ended without a completion finish_reason")
    if state.prompt_token_ids is None:
        raise ValueError("vLLM stream omitted echoed prompt_token_ids")
    if state.prompt_token_ids != state.prompt:
        raise ValueError("vLLM echoed prompt token ids differ from the sent prompt")
    if state.usage is None:
        raise ValueError("vLLM stream omitted the final usage chunk")
    if state.usage["prompt_tokens"] != len(state.prompt):
        raise ValueError("vLLM prompt usage differs from the sent prompt token count")
    if state.usage["completion_tokens"] != len(state.completion_token_ids):
        raise ValueError("vLLM completion usage differs from streamed token_ids")
    if (
        not state.completion_token_ids
        or state.first_token_time is None
        or state.last_token_time is None
    ):
        raise ValueError("vLLM stream returned no completion token ids")
    return (
        state.completion_token_ids,
        state.usage,
        state.finish_reason,
        state.first_token_time,
        state.last_token_time,
    )


def _build_request_record(
    request_index: int,
    prompt_id: str | int,
    request_body: Mapping[str, Any],
    state: _StreamState,
) -> dict[str, Any]:
    token_ids, usage, finish_reason, first_token_time, last_token_time = _validated_stream_state(
        state
    )
    decode_seconds = last_token_time - first_token_time
    sampling = {
        key: value
        for key, value in request_body.items()
        if key
        not in {"model", "prompt", "stream", "stream_options", "return_token_ids", "echo", "n"}
    }
    return {
        "request_index": request_index,
        "prompt_id": prompt_id,
        "model": request_body["model"],
        "prompt_tokens": len(state.prompt),
        "prompt_token_ids": state.prompt_token_ids,
        "sampling": sampling,
        "completion_token_ids": token_ids,
        "output_tokens": len(token_ids),
        "usage": usage,
        "finish_reason": finish_reason,
        "request_started_elapsed_seconds": state.request_started - state.cell_started,
        "request_wall_seconds": time.perf_counter() - state.request_started,
        "ttft_seconds": first_token_time - state.request_started,
        "first_token_elapsed_seconds": first_token_time - state.cell_started,
        "last_token_elapsed_seconds": last_token_time - state.cell_started,
        "decode_tokens_per_second": (len(token_ids) - 1) / decode_seconds
        if decode_seconds > 0
        else None,
        "token_events": state.token_events,
    }


async def _stream_request(
    runtime: _RequestRuntime, request_index: int, payload: Mapping[str, Any]
) -> dict[str, Any]:
    prompt, prompt_id, request_body = _request_payload(payload, request_index, runtime.model)
    state = _StreamState(prompt, time.perf_counter(), runtime.cell_started)
    async with runtime.client.stream("POST", runtime.completion_url, json=request_body) as response:
        response.raise_for_status()
        if response.status_code != HTTPStatus.OK:
            raise ValueError(
                f"vLLM completions returned HTTP {response.status_code}, expected {HTTPStatus.OK}"
            )
        async for line in response.aiter_lines():
            event = _parse_sse_line(line, state)
            if event is not None:
                _consume_stream_event(event, state)
    record = _build_request_record(request_index, prompt_id, request_body, state)
    await _append_jsonl(runtime.output_path, runtime.output_lock, record)
    return record


def _interval_union_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered_intervals = sorted(intervals)
    start, end = ordered_intervals[0]
    total_seconds = 0.0
    for next_start, next_end in ordered_intervals[1:]:
        if next_start > end:
            total_seconds += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total_seconds + end - start


def _server_metric_summary(
    before: Mapping[str, float], after: Mapping[str, float]
) -> dict[str, Any]:
    deltas = _counter_deltas(before, after)
    computed_prompt_tokens = deltas["vllm:request_prefill_kv_computed_tokens_sum"]
    prompt_tokens_total = deltas["vllm:prompt_tokens_total"]
    if computed_prompt_tokens > prompt_tokens_total:
        raise ValueError("vLLM computed prefill tokens exceed prompt_tokens_total")
    prefill_seconds = deltas["vllm:request_prefill_time_seconds_sum"]
    if prefill_seconds == 0 and computed_prompt_tokens > 0:
        raise ValueError("vLLM reported computed prompt tokens without request prefill time")
    return {
        "before": dict(before),
        "after": dict(after),
        "delta": deltas,
        "computed_prompt_tokens": computed_prompt_tokens,
        "prompt_tokens_total": prompt_tokens_total,
        "cached_prompt_tokens": deltas["vllm:prefix_cache_hits_total"],
        "prefix_cache_queries": deltas["vllm:prefix_cache_queries_total"],
        "preemptions": deltas["vllm:num_preemptions_total"],
        "prefill_seconds_per_request": prefill_seconds,
        "prefill_rate_denominator": "summed per-request prefill seconds, not wall seconds",
        "computed_prompt_tokens_per_request_second": (
            computed_prompt_tokens / prefill_seconds if prefill_seconds > 0 else None
        ),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _prepare_output_paths(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    existing_paths = [path.name for path in paths.values() if path.exists()]
    if existing_paths:
        raise FileExistsError(f"replay output already exists: {sorted(existing_paths)}")
    paths["nvml_samples.jsonl"].touch(mode=0o600, exist_ok=False)
    paths["request_results.jsonl"].touch(mode=0o600, exist_ok=False)
    return paths


async def _gather_fail_fast(tasks: Sequence[asyncio.Task[dict[str, Any]]]) -> list[dict[str, Any]]:
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    failed_task = next(
        (task for task in done if task.cancelled() or task.exception() is not None), None
    )
    if failed_task is not None:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if failed_task.cancelled():
            raise asyncio.CancelledError
        error = failed_task.exception()
        if error is None:
            raise RuntimeError("request task failed without an exception")
        raise error
    return [task.result() for task in tasks]


async def _run_requests(
    context: _CellContext, cell_started: float
) -> tuple[list[dict[str, Any]], float]:
    gpu_lock = asyncio.Lock()
    results_lock = asyncio.Lock()
    stop_sampling = asyncio.Event()
    initial_sample = await _record_gpu_sample(
        context.output_paths["nvml_samples.jsonl"], gpu_lock, cell_started
    )
    sampling_task = None
    if initial_sample["status"] == "available":
        sampling_task = asyncio.create_task(
            _sample_gpu_telemetry(
                context.output_paths["nvml_samples.jsonl"], gpu_lock, cell_started, stop_sampling
            )
        )
    request_runtime = _RequestRuntime(
        context.client,
        context.completion_url,
        context.model,
        cell_started,
        context.output_paths["request_results.jsonl"],
        results_lock,
    )
    request_tasks = [
        asyncio.create_task(_stream_request(request_runtime, index, payload))
        for index, payload in enumerate(context.payloads)
    ]
    try:
        records = await _gather_fail_fast(request_tasks)
    finally:
        stop_sampling.set()
        if sampling_task is not None:
            await sampling_task
    return records, time.perf_counter() - cell_started


async def _collect_measurement(context: _CellContext) -> _CellMeasurement:
    metrics_before_text, metrics_before = await _fetch_metrics(context.client, context.metrics_url)
    _write_text(context.output_paths["metrics_before.prom"], metrics_before_text)
    cpu_before = _read_cgroup_cpu_stat()
    cell_started = time.perf_counter()
    try:
        records, wall_seconds = await _run_requests(context, cell_started)
    finally:
        metrics_after_text, metrics_after = await _fetch_metrics(
            context.client, context.metrics_url
        )
        _write_text(context.output_paths["metrics_after.prom"], metrics_after_text)
        cpu_after = _read_cgroup_cpu_stat()
        _write_json(
            context.output_paths["cpu_telemetry.json"], _cpu_stat_delta(cpu_before, cpu_after)
        )
    return _CellMeasurement(records, metrics_before, metrics_after, wall_seconds)


def _decode_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decoding_records = [
        record
        for record in records
        if record["output_tokens"] > 1
        and record["last_token_elapsed_seconds"] > record["first_token_elapsed_seconds"]
    ]
    intervals = [
        (record["first_token_elapsed_seconds"], record["last_token_elapsed_seconds"])
        for record in decoding_records
    ]
    union_seconds = _interval_union_seconds(intervals)
    decoded_tokens = sum(record["output_tokens"] - 1 for record in decoding_records)
    overlap_start = max(start for start, _ in intervals) if intervals else None
    overlap_end = min(end for _, end in intervals) if intervals else None
    steady_seconds = (
        max(0.0, overlap_end - overlap_start)
        if len(decoding_records) == len(records)
        and overlap_start is not None
        and overlap_end is not None
        else 0.0
    )
    steady_tokens = sum(
        event["token_count"]
        for record in records
        for event in record["token_events"]
        if steady_seconds > 0 and overlap_start < event["elapsed_seconds"] <= overlap_end
    )
    token_intervals = [
        event["per_token_interval_seconds"]
        for record in records
        for event in record["token_events"]
        if event["per_token_interval_seconds"] is not None
    ]
    decode_rates = [
        record["decode_tokens_per_second"]
        for record in records
        if record["decode_tokens_per_second"] is not None
    ]
    return {
        "aggregate_decode_tokens": decoded_tokens,
        "aggregate_decode_tokens_s": decoded_tokens / union_seconds if union_seconds > 0 else None,
        "decode_wall_union_seconds": union_seconds,
        "steady_decode_tokens": steady_tokens,
        "steady_decode_seconds": steady_seconds,
        "steady_decode_tokens_s": steady_tokens / steady_seconds if steady_seconds > 0 else None,
        "mean_ttft_seconds": _mean([record["ttft_seconds"] for record in records]),
        "mean_itl_seconds": _mean(token_intervals),
        "mean_decode_tokens_per_second": _mean(decode_rates),
    }


def _gpu_summary(samples_path: Path) -> dict[str, Any]:
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    available = [sample for sample in samples if sample["status"] == "available"]
    return {
        "statuses": sorted({sample["status"] for sample in samples}),
        "sm_utilization_pct_mean": _mean([sample["sm_utilization_pct"] for sample in available]),
        "memory_controller_busy_pct_mean": _mean(
            [sample["memory_controller_busy_pct"] for sample in available]
        ),
        "memory_metric_semantics": "memory utilization is controller busy time, not bandwidth",
        "samples_path": "nvml_samples.jsonl",
    }


def _build_summary(context: _CellContext, measurement: _CellMeasurement) -> dict[str, Any]:
    records = measurement.records
    server_metrics = _server_metric_summary(measurement.metrics_before, measurement.metrics_after)
    decoding = _decode_summary(records)
    return {
        "model": context.model,
        "request_count": len(records),
        "prompt_tokens": sum(record["prompt_tokens"] for record in records),
        "output_tokens": sum(record["output_tokens"] for record in records),
        **decoding,
        "wall_seconds": measurement.wall_seconds,
        "ttft_seconds": [record["ttft_seconds"] for record in records],
        "decode_tokens_per_second": [record["decode_tokens_per_second"] for record in records],
        "computed_prompt_tokens_per_request_second": server_metrics[
            "computed_prompt_tokens_per_request_second"
        ],
        "server_metrics": server_metrics,
        "gpu_telemetry": _gpu_summary(context.output_paths["nvml_samples.jsonl"]),
        "cpu_telemetry_path": "cpu_telemetry.json",
    }


async def measure_cell(
    base_url: str,
    model: str,
    payloads: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Replay one concurrent cell of exact vLLM prompt token-id requests.

    Each payload is a `/v1/completions` request body with `prompt` set to token IDs and sampling
    parameters such as `max_tokens`, `temperature`, and `top_p`. Optional `prompt_id` and
    `request_id` fields are local metadata and are not sent to the server. This function forces
    streaming usage and token-id responses so the server counts can be checked exactly.
    """
    if not model.strip():
        raise ValueError("model must be nonempty")
    if not payloads:
        raise ValueError("measure_cell requires at least one payload")
    for request_index, payload in enumerate(payloads):
        _validate_payload(payload, request_index)
    completion_url = _completion_url(base_url)
    metrics_url = _metrics_url(base_url)
    output_paths = _prepare_output_paths(output_dir)
    timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        context = _CellContext(
            client,
            completion_url,
            metrics_url,
            model,
            payloads,
            output_paths,
        )
        measurement = await _collect_measurement(context)
    summary = _build_summary(context, measurement)
    _write_json(output_paths["summary.json"], summary)
    logger.info(
        "Replay measured model=%s requests=%s wall_seconds=%.3f aggregate_decode_tokens_s=%s",
        model,
        len(measurement.records),
        measurement.wall_seconds,
        summary["aggregate_decode_tokens_s"],
    )
    return summary
