"""Replay exact vLLM prompts over HTTP and measure streaming decode throughput."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import pynvml

logger = logging.getLogger(__name__)
GPU_SAMPLE_INTERVAL_SECONDS = 1.0
CPU_SAMPLE_INTERVAL_SECONDS = 1.0
PROMETHEUS_METRICS = (
    "vllm:request_prefill_time_seconds_sum",
    "vllm:prompt_tokens_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:num_preemptions_total",
    "vllm:request_prefill_kv_computed_tokens_sum",
)
PROMETHEUS_LINE = re.compile(r"(?P<name>[\w:]+)(?:\{.*\})?\s+(?P<value>[-+\w.eE]+)(?:\s+\d+)?")
PROFILE_STEP_ANNOTATION = re.compile(
    r"^execute_(?:(?P<scheduled_tokens>\d+)_)?context_(?P<context_requests>\d+)"
    r"\((?P<context_details>[^)]*)\)_generation_(?P<generation_requests>\d+)"
    r"\((?P<generation_details>[^)]*)\)$"
)
OUTPUT_FILENAMES = (
    "metrics_before.prom",
    "metrics_after.prom",
    "nvml_samples.jsonl",
    "cpu_samples.jsonl",
    "request_results.jsonl",
    "cpu_telemetry.json",
    "profile_capture.json",
    "profiler_summary.json",
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
    request_index: int
    sequence_index: int
    turn_index: int


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
    sequences: list[list[dict[str, Any]]]
    output_paths: Mapping[str, Path]
    server_pid: int | None
    profile: bool
    profiler_directory: Path | None


@dataclass(frozen=True)
class _CellMeasurement:
    records: list[dict[str, Any]]
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    wall_seconds: float
    cpu_telemetry: dict[str, Any]
    profiler_summary: dict[str, Any] | None


@dataclass(frozen=True)
class _MeasureSequenceOptions:
    server_pid: int | None
    profile: bool
    profiler_directory: Path | None


@dataclass(frozen=True)
class _CpuTelemetrySampler:
    output_path: Path
    output_lock: asyncio.Lock
    cell_started: float
    server_pid: int | None
    previous_cpu_seconds_by_pid: dict[int, float]


@dataclass
class _ProfilerTraceData:
    trace_files: list[str] = field(default_factory=list)
    kernel_intervals: list[tuple[float, float]] = field(default_factory=list)
    event_intervals: list[tuple[float, float]] = field(default_factory=list)
    scheduler_steps: list[dict[str, Any]] = field(default_factory=list)
    kernels_by_category: dict[str, dict[str, dict[str, float | int]]] = field(default_factory=dict)


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


def _cgroup_directory(pid: int) -> Path | None:
    cgroup_memberships = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    unified_membership = next(
        (line[3:] for line in cgroup_memberships if line.startswith("0::")), None
    )
    if unified_membership is None:
        return None
    cgroup_directory = Path("/sys/fs/cgroup") / unified_membership.lstrip("/")
    return cgroup_directory if cgroup_directory.is_dir() else None


def _read_cgroup_cpu_stat(pid: int | None = None) -> dict[str, int] | None:
    """Read cgroup-v2 CPU accounting for the selected process when available."""
    cgroup_directory = _cgroup_directory(os.getpid() if pid is None else pid)
    if cgroup_directory is None:
        return None
    try:
        cpu_stat_text = (cgroup_directory / "cpu.stat").read_text()
    except FileNotFoundError:
        return None
    cpu_stat: dict[str, int] = {}
    for line in cpu_stat_text.splitlines():
        name, value = line.split()
        cpu_stat[name] = int(value)
    return cpu_stat


def _read_pressure(path: Path) -> dict[str, dict[str, float | int]] | None:
    if not path.is_file():
        return None
    pressure: dict[str, dict[str, float | int]] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        pressure_kind = fields[0]
        values: dict[str, float | int] = {}
        for pressure_field in fields[1:]:
            name, value = pressure_field.split("=", maxsplit=1)
            values[name] = int(value) if name == "total" else float(value)
        pressure[pressure_kind] = values
    return pressure


def _read_process_stat(pid: int, role: str) -> dict[str, Any]:
    stat_line = Path(f"/proc/{pid}/stat").read_text()
    command_start = stat_line.index("(")
    command_end = stat_line.rindex(")")
    command = stat_line[command_start + 1 : command_end]
    fields = stat_line[command_end + 2 :].split()
    parent_pid = int(fields[1])
    user_ticks = int(fields[11])
    system_ticks = int(fields[12])
    resident_pages = int(fields[21])
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return {
        "pid": pid,
        "parent_pid": parent_pid,
        "command": command,
        "role": role,
        "user_cpu_seconds_total": user_ticks / ticks_per_second,
        "system_cpu_seconds_total": system_ticks / ticks_per_second,
        "cpu_seconds_total": (user_ticks + system_ticks) / ticks_per_second,
        "rss_mib": resident_pages * page_size / (1024**2),
    }


def _child_process_ids(pid: int) -> set[int]:
    task_directory = Path(f"/proc/{pid}/task")
    child_ids: set[int] = set()
    for task_directory_entry in task_directory.iterdir():
        children_path = task_directory_entry / "children"
        try:
            child_ids.update(map(int, children_path.read_text().split()))
        except FileNotFoundError:
            continue
    return child_ids


def _read_process_tree(root_pid: int, root_role: str) -> list[dict[str, Any]]:
    pending = [(root_pid, root_role)]
    process_records: list[dict[str, Any]] = []
    while pending:
        process_pid, role = pending.pop()
        try:
            process_record = _read_process_stat(process_pid, role)
        except FileNotFoundError:
            if process_pid == root_pid:
                raise
            continue
        process_records.append(process_record)
        pending.extend((child_pid, "engine") for child_pid in _child_process_ids(process_pid))
    return process_records


def _read_cpu_sample(server_pid: int | None) -> dict[str, Any]:
    target_pid = os.getpid() if server_pid is None else server_pid
    target_role = "client" if server_pid is None else "api"
    cgroup_directory = _cgroup_directory(target_pid)
    host_cpu_count = len(os.sched_getaffinity(0))
    load_average = os.getloadavg()
    return {
        "sampled_at_unix": time.time(),
        "host_load": {
            "load_1": load_average[0],
            "load_5": load_average[1],
            "load_15": load_average[2],
            "effective_cpu_count": host_cpu_count,
            "load_1_per_effective_cpu": load_average[0] / host_cpu_count,
        },
        "host_pressure": {
            pressure_name: _read_pressure(Path("/proc/pressure") / pressure_name)
            for pressure_name in ("cpu", "memory", "io")
        },
        "target_pid": target_pid,
        "target_role": target_role,
        "cgroup_cpu_stat": _read_cgroup_cpu_stat(target_pid),
        "cgroup_pressure": (
            {
                pressure_name: _read_pressure(cgroup_directory / f"{pressure_name}.pressure")
                for pressure_name in ("cpu", "memory", "io")
            }
            if cgroup_directory is not None
            else None
        ),
        "processes": _read_process_tree(target_pid, target_role),
    }


def _cpu_sample_with_deltas(
    sample: dict[str, Any], previous_cpu_seconds_by_pid: dict[int, float]
) -> dict[str, Any]:
    process_deltas_by_role: dict[str, float] = {}
    for process in sample["processes"]:
        pid = process["pid"]
        previous_total = previous_cpu_seconds_by_pid.get(pid, process["cpu_seconds_total"])
        cpu_delta = process["cpu_seconds_total"] - previous_total
        if cpu_delta < 0:
            raise ValueError(f"process CPU counter reset for PID {pid}")
        process["cpu_seconds_since_previous_sample"] = cpu_delta
        process_deltas_by_role[process["role"]] = (
            process_deltas_by_role.get(process["role"], 0.0) + cpu_delta
        )
        previous_cpu_seconds_by_pid[pid] = process["cpu_seconds_total"]
    sample["process_cpu_seconds_since_previous_sample_by_role"] = process_deltas_by_role
    return sample


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


def _nvml_optional_value(operation: Callable[..., int], *arguments: object) -> int | None:
    try:
        return int(operation(*arguments))
    except pynvml.NVMLError as error:
        unsupported_codes = {
            pynvml.NVML_ERROR_NOT_SUPPORTED,
            pynvml.NVML_ERROR_FUNCTION_NOT_FOUND,
        }
        if error.value not in unsupported_codes:  # pyright: ignore[reportAttributeAccessIssue]
            raise
        return None


def _read_nvml_sample() -> dict[str, Any]:
    """Sample utilization through NVML without estimating memory bandwidth."""
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        device_name = pynvml.nvmlDeviceGetName(handle)
        device_name = device_name.decode() if isinstance(device_name, bytes) else device_name
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temperature_c = _nvml_optional_value(
            pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
        )
        throttle_reasons_mask = _nvml_optional_value(
            pynvml.nvmlDeviceGetCurrentClocksThrottleReasons, handle
        )
        return {
            "status": "available",
            "device_name": device_name,
            "sm_utilization_pct": float(utilization.gpu),
            "memory_controller_busy_pct": float(utilization.memory),
            "power_watts": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            "sm_clock_mhz": float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
            "memory_clock_mhz": float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)),
            "temperature_c": temperature_c,
            "clock_throttle_reasons_mask": throttle_reasons_mask,
            "clock_throttling_active": (
                None if throttle_reasons_mask is None else throttle_reasons_mask != 0
            ),
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


def _rewrite_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.touch(mode=0o600, exist_ok=False)
    with temporary_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
    temporary_path.replace(path)


async def _record_gpu_sample(
    output_path: Path,
    output_lock: asyncio.Lock,
    cell_started: float,
) -> dict[str, Any]:
    sample = await asyncio.to_thread(_read_nvml_sample)
    record = {"elapsed_seconds": time.perf_counter() - cell_started, **sample}
    await _append_jsonl(output_path, output_lock, record)
    return sample


async def _record_cpu_sample(
    output_path: Path,
    output_lock: asyncio.Lock,
    cell_started: float,
    server_pid: int | None,
    previous_cpu_seconds_by_pid: dict[int, float],
) -> dict[str, Any]:
    sample = await asyncio.to_thread(_read_cpu_sample, server_pid)
    sample = _cpu_sample_with_deltas(sample, previous_cpu_seconds_by_pid)
    record = {"elapsed_seconds": time.perf_counter() - cell_started, **sample}
    await _append_jsonl(output_path, output_lock, record)
    return record


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


async def _sample_cpu_telemetry(sampler: _CpuTelemetrySampler, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=CPU_SAMPLE_INTERVAL_SECONDS)
        except TimeoutError:
            await _record_cpu_sample(
                sampler.output_path,
                sampler.output_lock,
                sampler.cell_started,
                sampler.server_pid,
                sampler.previous_cpu_seconds_by_pid,
            )


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
    runtime: _RequestRuntime,
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
        "request_index": runtime.request_index,
        "sequence_index": runtime.sequence_index,
        "turn_index": runtime.turn_index,
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


async def _stream_request(runtime: _RequestRuntime, payload: Mapping[str, Any]) -> dict[str, Any]:
    prompt, prompt_id, request_body = _request_payload(
        payload, runtime.request_index, runtime.model
    )
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
    record = _build_request_record(
        runtime,
        prompt_id,
        request_body,
        state,
    )
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


def _verify_profiler_source() -> dict[str, str]:
    """Check installed vLLM's opt-in profile routes without importing its engine."""
    installed_version = version("vllm")
    if installed_version != "0.31.0":
        raise ValueError(
            f"Revalidate profiler endpoints for vLLM {installed_version}; expected 0.31.0"
        )
    installed_distribution = distribution("vllm")
    endpoint_relative_path = "vllm/entrypoints/serve/profile/api_router.py"
    router_relative_path = "vllm/entrypoints/serve/__init__.py"
    worker_relative_path = "vllm/v1/worker/gpu_worker.py"
    endpoint_source = Path(str(installed_distribution.locate_file(endpoint_relative_path)))
    router_source = Path(str(installed_distribution.locate_file(router_relative_path)))
    worker_source = Path(str(installed_distribution.locate_file(worker_relative_path)))
    endpoint_source_text = endpoint_source.read_text(encoding="utf-8")
    router_source_text = router_source.read_text(encoding="utf-8")
    worker_source_text = worker_source.read_text(encoding="utf-8")
    expected_endpoint_lines = (
        '@router.post("/start_profile")',
        '@router.post("/stop_profile")',
        "if profiler_config is not None and profiler_config.profiler is not None:",
        "app.include_router(router)",
    )
    for source_line in expected_endpoint_lines:
        if source_line not in endpoint_source_text:
            raise ValueError(
                f"Installed vLLM profiler source omitted expected route code: {source_line}"
            )
    if "attach_profile_router(app)" not in router_source_text:
        raise ValueError("Installed vLLM serve source does not register the profiler router")
    expected_annotation_lines = (
        "def annotate_profile(self, scheduler_output):",
        '"execute_context_"',
        '"_generation_"',
        "num_ctx_tokens",
        "num_generation_tokens",
    )
    for source_line in expected_annotation_lines:
        if source_line not in worker_source_text:
            raise ValueError(
                f"Installed vLLM scheduler profiler annotation omitted expected source: {source_line}"
            )
    return {
        "vllm_version": installed_version,
        "endpoint_source": endpoint_relative_path,
        "router_source": router_relative_path,
        "scheduler_annotation_source": worker_relative_path,
        "verified_routes": "/start_profile,/stop_profile",
        "scheduler_annotation": "context/generation counts and token counts",
        "requires_profiler_config": "true",
    }


def _trace_snapshot(trace_directory: Path) -> dict[str, tuple[int, int]]:
    if not trace_directory.is_dir():
        return {}
    return {
        str(trace_path.relative_to(trace_directory)): (
            trace_path.stat().st_mtime_ns,
            trace_path.stat().st_size,
        )
        for trace_path in trace_directory.rglob("*")
        if trace_path.is_file()
    }


def _profiler_kernel_category(kernel_name: str) -> str:
    normalized_name = kernel_name.lower()
    if "gated_delta_rule" in normalized_name or "fla_linear_attention" in normalized_name:
        return "linear_attention"
    if any(fragment in normalized_name for fragment in ("flash", "attention", "fmha", "attn")):
        return "attention"
    if any(
        fragment in normalized_name
        for fragment in ("gemm", "matmul", "cublas", "cutlass", "gemv", "linear", "marlin")
    ):
        return "matrix_multiply"
    if any(
        fragment in normalized_name
        for fragment in ("sampler", "logits", "thinking", "budget", "bitmask", "topk", "sampling")
    ):
        return "sampling_or_logits_processing"
    if any(
        fragment in normalized_name
        for fragment in ("norm", "softmax", "rotary", "rope", "silu", "gelu", "activation")
    ):
        return "normalization_or_pointwise"
    return "other"


def _annotation_token_count(annotation_details: str) -> int:
    token_count_match = re.match(r"(?:sq)?(?P<token_count>\d+)(?:sk\d+.*)?$", annotation_details)
    if token_count_match is None:
        raise ValueError(
            f"Unable to read token count from vLLM profiler annotation: {annotation_details}"
        )
    return int(token_count_match.group("token_count"))


def _parse_scheduler_step_annotation(event: Mapping[str, Any]) -> dict[str, Any] | None:
    name = event.get("name")
    if not isinstance(name, str) or not name.startswith("execute_"):
        return None
    annotation_match = PROFILE_STEP_ANNOTATION.fullmatch(name)
    if annotation_match is None:
        if "_context_" in name and "_generation_" in name:
            raise ValueError(f"Unable to parse vLLM scheduler profiler annotation: {name}")
        return None
    timestamp_us = event.get("ts")
    duration_us = event.get("dur")
    if (
        not isinstance(timestamp_us, (int, float))
        or isinstance(timestamp_us, bool)
        or not isinstance(duration_us, (int, float))
        or isinstance(duration_us, bool)
        or not math.isfinite(timestamp_us)
        or not math.isfinite(duration_us)
        or duration_us < 0
    ):
        raise ValueError(f"vLLM scheduler profiler annotation has malformed timing: {name}")
    context_tokens = _annotation_token_count(annotation_match.group("context_details"))
    generation_tokens = _annotation_token_count(annotation_match.group("generation_details"))
    generation_requests = int(annotation_match.group("generation_requests"))
    return {
        "name": name,
        "timestamp_seconds": timestamp_us / 1_000_000,
        "host_scope_seconds": duration_us / 1_000_000,
        "context_requests": int(annotation_match.group("context_requests")),
        "context_tokens": context_tokens,
        "generation_requests": generation_requests,
        "generation_tokens": generation_tokens,
        "scheduled_tokens": (
            int(annotation_match.group("scheduled_tokens"))
            if annotation_match.group("scheduled_tokens") is not None
            else context_tokens + generation_tokens
        ),
        "generation_bearing": generation_requests > 0 or generation_tokens > 0,
    }


def _trace_events(trace_path: Path) -> list[object]:
    if trace_path.name.endswith(".gz"):
        with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
            trace = json.load(stream)
    else:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(trace, Mapping) or not isinstance(trace.get("traceEvents"), list):
        raise TypeError(f"vLLM profiler output has no traceEvents list: {trace_path.name}")
    return trace["traceEvents"]


def _profiler_event_interval(
    event: Mapping[str, Any], trace_name: str
) -> tuple[float, float] | None:
    if event.get("ph") != "X":
        return None
    timestamp_us = event.get("ts")
    duration_us = event.get("dur")
    if (
        not isinstance(timestamp_us, (int, float))
        or isinstance(timestamp_us, bool)
        or not isinstance(duration_us, (int, float))
        or isinstance(duration_us, bool)
        or not math.isfinite(timestamp_us)
        or not math.isfinite(duration_us)
        or duration_us < 0
    ):
        raise ValueError(f"vLLM profiler emitted a malformed complete event in {trace_name}")
    start_seconds = timestamp_us / 1_000_000
    return start_seconds, start_seconds + duration_us / 1_000_000


def _record_profiler_event(event: object, trace_name: str, trace_data: _ProfilerTraceData) -> None:
    if not isinstance(event, Mapping):
        return
    interval = _profiler_event_interval(event, trace_name)
    if interval is None:
        return
    start_seconds, end_seconds = interval
    trace_data.event_intervals.append(interval)
    scheduler_step = _parse_scheduler_step_annotation(event)
    if scheduler_step is not None:
        trace_data.scheduler_steps.append(scheduler_step)
    if "kernel" not in str(event.get("cat", "")).lower():
        return
    kernel_name = event.get("name")
    duration_seconds = end_seconds - start_seconds
    if not isinstance(kernel_name, str) or not kernel_name or duration_seconds <= 0:
        raise ValueError(f"vLLM profiler emitted a malformed GPU kernel event in {trace_name}")
    trace_data.kernel_intervals.append(interval)
    category = _profiler_kernel_category(kernel_name)
    category_kernels = trace_data.kernels_by_category.setdefault(category, {})
    kernel = category_kernels.setdefault(kernel_name, {"count": 0, "total_microseconds": 0.0})
    kernel["count"] = int(kernel["count"]) + 1
    kernel["total_microseconds"] = (
        float(kernel["total_microseconds"]) + duration_seconds * 1_000_000
    )


def _annotate_scheduler_step_kernel_overlap(
    scheduler_steps: Sequence[dict[str, Any]],
    kernel_intervals: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    for scheduler_step in scheduler_steps:
        step_start = scheduler_step["timestamp_seconds"]
        step_end = step_start + scheduler_step["host_scope_seconds"]
        overlaps = [
            (max(step_start, kernel_start), min(step_end, kernel_end))
            for kernel_start, kernel_end in kernel_intervals
            if min(step_end, kernel_end) > max(step_start, kernel_start)
        ]
        scheduler_step["overlapping_gpu_kernel_union_seconds"] = _interval_union_seconds(overlaps)
    generation_steps = [step for step in scheduler_steps if step["generation_bearing"]]
    mixed_context_and_generation_steps = [
        step
        for step in scheduler_steps
        if step["context_requests"] > 0 and step["generation_requests"] > 0
    ]
    pure_generation_steps = [
        step
        for step in scheduler_steps
        if step["context_requests"] == 0 and step["generation_requests"] > 0
    ]
    return {
        "step_count": len(scheduler_steps),
        "generation_bearing_step_count": len(generation_steps),
        "context_only_step_count": len(scheduler_steps) - len(generation_steps),
        "mixed_context_and_generation_step_count": len(mixed_context_and_generation_steps),
        "mixed_context_and_generation_mean_host_scope_seconds": _mean(
            [step["host_scope_seconds"] for step in mixed_context_and_generation_steps]
        ),
        "pure_generation_step_count": len(pure_generation_steps),
        "pure_generation_mean_host_scope_seconds": _mean(
            [step["host_scope_seconds"] for step in pure_generation_steps]
        ),
        "context_tokens": sum(step["context_tokens"] for step in scheduler_steps),
        "generation_tokens": sum(step["generation_tokens"] for step in scheduler_steps),
        "generation_bearing_host_scope_seconds": sum(
            step["host_scope_seconds"] for step in generation_steps
        ),
        "generation_bearing_overlapping_gpu_kernel_union_seconds": sum(
            step["overlapping_gpu_kernel_union_seconds"] for step in generation_steps
        ),
        "stall_attribution": "not exposed by the profiler annotation",
        "timing_interpretation": (
            "host scope duration and temporally overlapping GPU kernels; overlap does not prove causality"
        ),
    }


def _gpu_kernel_duration_summary(
    kernels_by_category: Mapping[str, Mapping[str, Mapping[str, float | int]]],
) -> tuple[float, dict[str, dict[str, float]]]:
    category_microseconds = {
        category: sum(float(kernel["total_microseconds"]) for kernel in category_kernels.values())
        for category, category_kernels in kernels_by_category.items()
    }
    total_microseconds = sum(category_microseconds.values())
    category_summary = {
        category: {
            "total_seconds": microseconds / 1_000_000,
            "share_of_total_gpu_kernel_duration": (
                microseconds / total_microseconds if total_microseconds > 0 else 0.0
            ),
        }
        for category, microseconds in sorted(category_microseconds.items())
    }
    return total_microseconds / 1_000_000, category_summary


def _top_gpu_kernels_by_category(
    kernels_by_category: Mapping[str, Mapping[str, Mapping[str, float | int]]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        category: [
            {
                "name": name,
                "count": int(kernel["count"]),
                "total_milliseconds": float(kernel["total_microseconds"]) / 1000,
                "mean_milliseconds": float(kernel["total_microseconds"])
                / int(kernel["count"])
                / 1000,
            }
            for name, kernel in sorted(
                category_kernels.items(),
                key=lambda item: float(item[1]["total_microseconds"]),
                reverse=True,
            )[:10]
        ]
        for category, category_kernels in sorted(kernels_by_category.items())
    }


def _summarize_profiler_traces(
    trace_directory: Path, previous_snapshot: Mapping[str, tuple[int, int]]
) -> dict[str, Any]:
    current_snapshot = _trace_snapshot(trace_directory)
    new_trace_paths = [
        trace_directory / relative_path
        for relative_path, stat in current_snapshot.items()
        if relative_path not in previous_snapshot or previous_snapshot[relative_path] != stat
    ]
    trace_data = _ProfilerTraceData()
    for trace_path in sorted(new_trace_paths):
        if ".trace.json" not in trace_path.name:
            continue
        trace_data.trace_files.append(str(trace_path.relative_to(trace_directory)))
        for event in _trace_events(trace_path):
            _record_profiler_event(event, trace_path.name, trace_data)

    if not trace_data.event_intervals:
        return {
            "status": "unavailable",
            "reason": "no complete profiler events found in newly written traces",
            "trace_files": trace_data.trace_files,
        }
    busy_union_seconds = _interval_union_seconds(trace_data.kernel_intervals)
    profiler_window_seconds = max(end for _, end in trace_data.event_intervals) - min(
        start for start, _ in trace_data.event_intervals
    )
    possible_non_kernel_seconds = max(0.0, profiler_window_seconds - busy_union_seconds)
    possible_non_kernel_fraction = (
        min(1.0, possible_non_kernel_seconds / profiler_window_seconds)
        if profiler_window_seconds > 0
        else None
    )
    scheduler_step_summary = _annotate_scheduler_step_kernel_overlap(
        trace_data.scheduler_steps, trace_data.kernel_intervals
    )
    total_gpu_kernel_duration_seconds, gpu_kernel_duration_by_category = (
        _gpu_kernel_duration_summary(trace_data.kernels_by_category)
    )
    top_gpu_kernels = _top_gpu_kernels_by_category(trace_data.kernels_by_category)
    return {
        "status": "available",
        "trace_files": trace_data.trace_files,
        "profiler_event_window_seconds": profiler_window_seconds,
        "profiler_window_source": "envelope of complete trace events, including CPU and GPU events",
        "gpu_kernel_count": sum(
            int(kernel["count"])
            for group in trace_data.kernels_by_category.values()
            for kernel in group.values()
        ),
        "total_gpu_kernel_duration_seconds": total_gpu_kernel_duration_seconds,
        "gpu_kernel_duration_by_category": gpu_kernel_duration_by_category,
        "gpu_kernel_duration_interpretation": (
            "sum of kernel-event durations; overlaps count separately and this is not SM-active time"
        ),
        "kernel_busy_interval_union_seconds": busy_union_seconds,
        "possible_host_idle_upper_bound_seconds": possible_non_kernel_seconds,
        "possible_non_kernel_fraction_upper_bound": possible_non_kernel_fraction,
        "host_idle_interpretation": (
            "upper bound only; includes GPU waits, launch gaps, and other non-kernel time"
        ),
        "scheduler_steps": trace_data.scheduler_steps,
        "scheduler_step_summary": scheduler_step_summary,
        "top_gpu_kernels_by_category": top_gpu_kernels,
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
    paths["cpu_samples.jsonl"].touch(mode=0o600, exist_ok=False)
    paths["request_results.jsonl"].touch(mode=0o600, exist_ok=False)
    return paths


async def _gather_fail_fast(
    tasks: Sequence[asyncio.Task[list[dict[str, Any]]]],
) -> list[list[dict[str, Any]]]:
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


def _annotate_ttft_overlap(records: Sequence[dict[str, Any]]) -> None:
    ttft_intervals = [
        (
            record["request_index"],
            record["request_started_elapsed_seconds"],
            record["first_token_elapsed_seconds"],
        )
        for record in records
    ]
    for record in records:
        for event in record["token_events"]:
            event_end = event["elapsed_seconds"]
            interval_seconds = event["inter_token_event_seconds"]
            event_start = event_end if interval_seconds is None else event_end - interval_seconds
            overlapping_waits: list[tuple[float, float]] = []
            awaiting_at_event = 0
            for request_index, wait_start, first_token_time in ttft_intervals:
                if request_index == record["request_index"] or first_token_time is None:
                    continue
                if wait_start <= event_end < first_token_time:
                    awaiting_at_event += 1
                overlap_start = max(event_start, wait_start)
                overlap_end = min(event_end, first_token_time)
                if overlap_end > overlap_start:
                    overlapping_waits.append((overlap_start, overlap_end))
            event["other_requests_awaiting_first_token"] = awaiting_at_event
            event["ttft_overlap_request_count"] = len(overlapping_waits)
            event["overlap_seconds_with_other_ttft"] = sum(
                end - start for start, end in overlapping_waits
            )


async def _run_sequences(
    context: _CellContext, cell_started: float
) -> tuple[list[dict[str, Any]], float]:
    gpu_lock = asyncio.Lock()
    cpu_lock = asyncio.Lock()
    output_lock = asyncio.Lock()
    stop_sampling = asyncio.Event()
    previous_cpu_seconds_by_pid: dict[int, float] = {}
    initial_sample = await _record_gpu_sample(
        context.output_paths["nvml_samples.jsonl"], gpu_lock, cell_started
    )
    gpu_sampling_task: asyncio.Task[None] | None = None
    if initial_sample["status"] == "available":
        gpu_sampling_task = asyncio.create_task(
            _sample_gpu_telemetry(
                context.output_paths["nvml_samples.jsonl"], gpu_lock, cell_started, stop_sampling
            )
        )
    await _record_cpu_sample(
        context.output_paths["cpu_samples.jsonl"],
        cpu_lock,
        cell_started,
        context.server_pid,
        previous_cpu_seconds_by_pid,
    )
    cpu_sampler = _CpuTelemetrySampler(
        context.output_paths["cpu_samples.jsonl"],
        cpu_lock,
        cell_started,
        context.server_pid,
        previous_cpu_seconds_by_pid,
    )
    cpu_sampling_task = asyncio.create_task(_sample_cpu_telemetry(cpu_sampler, stop_sampling))
    request_index = 0
    request_indices_by_sequence: list[list[int]] = []
    for sequence in context.sequences:
        request_indices_by_sequence.append(
            list(range(request_index, request_index + len(sequence)))
        )
        request_index += len(sequence)

    async def run_sequence(
        sequence_index: int, sequence: Sequence[Mapping[str, Any]], request_indices: Sequence[int]
    ) -> list[dict[str, Any]]:
        sequence_records: list[dict[str, Any]] = []
        for turn_index, (payload, unique_request_index) in enumerate(
            zip(sequence, request_indices, strict=True)
        ):
            request_runtime = _RequestRuntime(
                context.client,
                context.completion_url,
                context.model,
                cell_started,
                context.output_paths["request_results.jsonl"],
                output_lock,
                unique_request_index,
                sequence_index,
                turn_index,
            )
            sequence_records.append(await _stream_request(request_runtime, payload))
        return sequence_records

    request_tasks = [
        asyncio.create_task(run_sequence(sequence_index, sequence, request_indices))
        for sequence_index, (sequence, request_indices) in enumerate(
            zip(context.sequences, request_indices_by_sequence, strict=True)
        )
    ]
    try:
        sequence_records = await _gather_fail_fast(request_tasks)
        records = [
            record for sequence_records_item in sequence_records for record in sequence_records_item
        ]
        _annotate_ttft_overlap(records)
        _rewrite_jsonl(context.output_paths["request_results.jsonl"], records)
    finally:
        stop_sampling.set()
        if gpu_sampling_task is not None:
            await gpu_sampling_task
            await _record_gpu_sample(
                context.output_paths["nvml_samples.jsonl"], gpu_lock, cell_started
            )
        await cpu_sampling_task
        await _record_cpu_sample(
            context.output_paths["cpu_samples.jsonl"],
            cpu_lock,
            cell_started,
            context.server_pid,
            previous_cpu_seconds_by_pid,
        )
    return records, time.perf_counter() - cell_started


async def _post_profile_endpoint(
    client: httpx.AsyncClient, metrics_url: str, endpoint: str
) -> dict[str, Any]:
    parsed = urlsplit(metrics_url)
    endpoint_url = urlunsplit((parsed.scheme, parsed.netloc, f"/{endpoint}", "", ""))
    response = await client.post(endpoint_url)
    return {
        "endpoint": endpoint,
        "http_status": response.status_code,
        "completed_at_unix": time.time(),
    }


def _pressure_total_delta(
    before: Mapping[str, Any] | None, after: Mapping[str, Any] | None
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    deltas: dict[str, int] = {}
    for pressure_name in before.keys() & after.keys():
        before_total = before[pressure_name].get("total")
        after_total = after[pressure_name].get("total")
        if type(before_total) is int and type(after_total) is int:
            delta = after_total - before_total
            if delta < 0:
                raise ValueError(f"pressure counter reset for {pressure_name}")
            deltas[pressure_name] = delta
    return deltas


def _cpu_telemetry_summary(samples_path: Path) -> dict[str, Any]:
    samples = [
        json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not samples:
        raise ValueError("CPU telemetry sampling produced no samples")
    first_sample = samples[0]
    last_sample = samples[-1]
    process_cpu_seconds_by_role: dict[str, float] = {}
    for sample in samples:
        for role, cpu_seconds in sample[
            "process_cpu_seconds_since_previous_sample_by_role"
        ].items():
            process_cpu_seconds_by_role[role] = (
                process_cpu_seconds_by_role.get(role, 0.0) + cpu_seconds
            )
    host_pressure_delta = {
        pressure_name: _pressure_total_delta(
            first_sample["host_pressure"][pressure_name],
            last_sample["host_pressure"][pressure_name],
        )
        for pressure_name in ("cpu", "memory", "io")
    }
    cgroup_pressure_delta = (
        {
            pressure_name: _pressure_total_delta(
                first_sample["cgroup_pressure"][pressure_name],
                last_sample["cgroup_pressure"][pressure_name],
            )
            for pressure_name in ("cpu", "memory", "io")
        }
        if first_sample["cgroup_pressure"] is not None
        and last_sample["cgroup_pressure"] is not None
        else None
    )
    return {
        "target_pid": first_sample["target_pid"],
        "server_pid": (
            first_sample["target_pid"] if first_sample["target_role"] == "api" else None
        ),
        "target_role": first_sample["target_role"],
        "sample_count": len(samples),
        "samples_path": "cpu_samples.jsonl",
        "cgroup_cpu_stat": _cpu_stat_delta(
            first_sample["cgroup_cpu_stat"], last_sample["cgroup_cpu_stat"]
        ),
        "host_pressure_total_us_delta": host_pressure_delta,
        "cgroup_pressure_total_us_delta": cgroup_pressure_delta,
        "process_cpu_seconds_by_role": process_cpu_seconds_by_role,
        "host_load_1_mean": _mean([sample["host_load"]["load_1"] for sample in samples]),
        "host_effective_cpu_count": last_sample["host_load"]["effective_cpu_count"],
    }


async def _collect_measurement(context: _CellContext) -> _CellMeasurement:
    metrics_before_text, metrics_before = await _fetch_metrics(context.client, context.metrics_url)
    _write_text(context.output_paths["metrics_before.prom"], metrics_before_text)
    cell_started = time.perf_counter()
    profile_capture: dict[str, Any] | None = None
    profile_trace_snapshot: dict[str, tuple[int, int]] = {}
    profile_started = False
    records: list[dict[str, Any]] = []
    wall_seconds = 0.0
    metrics_after: dict[str, float] = {}
    profiler_summary: dict[str, Any] | None = None
    try:
        if context.profile:
            if context.profiler_directory is None:
                raise ValueError("profiler_directory is required when profile=True")
            profile_capture = {
                "source_verification": _verify_profiler_source(),
                "profile_directory_name": context.profiler_directory.name,
                "endpoints": [],
                "endpoint_results": [],
            }
            profile_trace_snapshot = _trace_snapshot(context.profiler_directory)
            start_result = await _post_profile_endpoint(
                context.client, context.metrics_url, "start_profile"
            )
            profile_capture["endpoint_results"].append(start_result)
            profile_capture["endpoints"].append("start_profile")
            if start_result["http_status"] != HTTPStatus.OK:
                raise ValueError(
                    f"vLLM profiler start endpoint returned HTTP {start_result['http_status']}"
                )
            profile_started = True
        records, wall_seconds = await _run_sequences(context, cell_started)
    finally:
        try:
            if profile_capture is not None and profile_started:
                stop_result = await _post_profile_endpoint(
                    context.client, context.metrics_url, "stop_profile"
                )
                profile_capture["endpoint_results"].append(stop_result)
                profile_capture["endpoints"].append("stop_profile")
                if stop_result["http_status"] != HTTPStatus.OK:
                    raise ValueError(
                        f"vLLM profiler stop endpoint returned HTTP {stop_result['http_status']}"
                    )
                profile_started = False
        finally:
            if profile_capture is not None:
                _write_json(context.output_paths["profile_capture.json"], profile_capture)
            metrics_after_text, metrics_after = await _fetch_metrics(
                context.client, context.metrics_url
            )
            _write_text(context.output_paths["metrics_after.prom"], metrics_after_text)
            if context.output_paths["cpu_samples.jsonl"].stat().st_size == 0:
                await _record_cpu_sample(
                    context.output_paths["cpu_samples.jsonl"],
                    asyncio.Lock(),
                    cell_started,
                    context.server_pid,
                    {},
                )
            cpu_telemetry = _cpu_telemetry_summary(context.output_paths["cpu_samples.jsonl"])
            _write_json(context.output_paths["cpu_telemetry.json"], cpu_telemetry)
    if context.profile and context.profiler_directory is not None:
        profiler_summary = _summarize_profiler_traces(
            context.profiler_directory, profile_trace_snapshot
        )
        _write_json(context.output_paths["profiler_summary.json"], profiler_summary)
    return _CellMeasurement(
        records,
        metrics_before,
        metrics_after,
        wall_seconds,
        cpu_telemetry,
        profiler_summary,
    )


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
    token_events = [event for record in records for event in record["token_events"]]
    overlapping_events = [
        event for event in token_events if event["ttft_overlap_request_count"] > 0
    ]
    overlapped_intervals = [
        event["inter_token_event_seconds"]
        for event in overlapping_events
        if event["inter_token_event_seconds"] is not None
    ]
    non_overlapped_intervals = [
        event["inter_token_event_seconds"]
        for event in token_events
        if event["ttft_overlap_request_count"] == 0
        and event["inter_token_event_seconds"] is not None
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
        "ttft_overlap": {
            "interpretation": "observational correlation, not causal attribution",
            "token_event_count": len(token_events),
            "token_events_overlapping_other_ttft": len(overlapping_events),
            "request_seconds_overlapping_other_ttft": sum(
                event["overlap_seconds_with_other_ttft"] for event in token_events
            ),
            "mean_inter_token_event_seconds_during_other_ttft": _mean(overlapped_intervals),
            "mean_inter_token_event_seconds_outside_other_ttft": _mean(non_overlapped_intervals),
        },
    }


def _gpu_summary(samples_path: Path) -> dict[str, Any]:
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    available = [sample for sample in samples if sample["status"] == "available"]

    def available_values(field_name: str) -> list[float]:
        return [
            float(sample[field_name]) for sample in available if sample.get(field_name) is not None
        ]

    throttle_masks = [
        int(sample["clock_throttle_reasons_mask"])
        for sample in available
        if sample.get("clock_throttle_reasons_mask") is not None
    ]
    return {
        "statuses": sorted({sample["status"] for sample in samples}),
        "sm_utilization_pct_mean": _mean(available_values("sm_utilization_pct")),
        "memory_controller_busy_pct_mean": _mean(available_values("memory_controller_busy_pct")),
        "power_watts_mean": _mean(available_values("power_watts")),
        "sm_clock_mhz_mean": _mean(available_values("sm_clock_mhz")),
        "memory_clock_mhz_mean": _mean(available_values("memory_clock_mhz")),
        "temperature_c_mean": _mean(available_values("temperature_c")),
        "throttle_reasons_nonzero": any(mask != 0 for mask in throttle_masks),
        "clock_throttle_reasons_masks": throttle_masks,
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
        "sequence_count": len(context.sequences),
        "turn_count": len(records),
        "prompt_tokens": sum(record["prompt_tokens"] for record in records),
        "output_tokens": sum(record["output_tokens"] for record in records),
        "output_tokens_per_wall_second": (
            sum(record["output_tokens"] for record in records) / measurement.wall_seconds
            if measurement.wall_seconds > 0
            else None
        ),
        **decoding,
        "wall_seconds": measurement.wall_seconds,
        "ttft_seconds": [record["ttft_seconds"] for record in records],
        "decode_tokens_per_second": [record["decode_tokens_per_second"] for record in records],
        "computed_prompt_tokens_per_request_second": server_metrics[
            "computed_prompt_tokens_per_request_second"
        ],
        "server_metrics": server_metrics,
        "gpu_telemetry": _gpu_summary(context.output_paths["nvml_samples.jsonl"]),
        "cpu_telemetry": measurement.cpu_telemetry,
        "cpu_telemetry_path": "cpu_telemetry.json",
        "profiler_summary": measurement.profiler_summary,
    }


async def _measure_sequences(
    base_url: str,
    model: str,
    sequences: list[list[dict[str, Any]]],
    output_dir: Path,
    options: _MeasureSequenceOptions,
) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("model must be nonempty")
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("replay measurement requires nonempty independent sequences")
    if options.server_pid is not None and options.server_pid < 1:
        raise ValueError("server_pid must be a positive process ID")
    request_index = 0
    for sequence in sequences:
        for payload in sequence:
            _validate_payload(payload, request_index)
            request_index += 1
    profiler_directory = options.profiler_directory
    if options.profile and profiler_directory is None:
        profiler_directory = output_dir / "profiler"
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
            sequences,
            output_paths,
            options.server_pid,
            options.profile,
            profiler_directory,
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


async def measure_cell(
    base_url: str,
    model: str,
    payloads: list[dict[str, Any]],
    output_dir: Path,
    *,
    server_pid: int | None = None,
) -> dict[str, Any]:
    """Replay simultaneous one-turn requests with exact vLLM prompt token IDs.

    Each payload is a `/v1/completions` request body with a `prompt` token-id list and sampling
    parameters such as `max_tokens`, `temperature`, and `top_p`. Optional `prompt_id` and
    `request_id` fields are local metadata. The function forces streaming usage and token-ID
    responses, then checks the streamed IDs against vLLM's usage counts. Pass the API server PID
    to sample its CPU use and all engine descendants.
    """
    if not payloads:
        raise ValueError("measure_cell requires at least one payload")
    return await _measure_sequences(
        base_url,
        model,
        [[payload] for payload in payloads],
        output_dir,
        _MeasureSequenceOptions(server_pid, profile=False, profiler_directory=None),
    )


async def measure_sequence_cell(  # noqa: PLR0913 - named profiler controls are part of the caller-facing API
    base_url: str,
    model: str,
    sequences: list[list[dict[str, Any]]],
    output_dir: Path,
    *,
    server_pid: int | None = None,
    profile: bool = False,
    profiler_directory: Path | None = None,
) -> dict[str, Any]:
    """Replay recorded turns in parallel episode chains with exact prompt prefixes.

    Each sequence runs turns serially, while independent sequences run concurrently. The prompt
    IDs supplied for every turn are sent unchanged; output IDs are recorded but never appended to
    the next prompt. Prefix caching therefore sees the caller's original shared histories without
    changing the measured text across variants. A cold-cache reset, if needed, belongs before this
    call and is applied once for the whole cell.
    """
    return await _measure_sequences(
        base_url,
        model,
        sequences,
        output_dir,
        _MeasureSequenceOptions(server_pid, profile, profiler_directory),
    )
