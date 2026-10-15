"""Shared generation CLI controls for the stage performance instrument."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from grpo.gpu_perf import PerfMonitor, PerfThresholds
from grpo.model_geometry import load_model_geometry
from reward_hacking.evidence_store import read_jsonl


def _nonempty_reason(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("accept-slow requires a nonempty reason")
    return value.strip()


def _positive_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("performance thresholds must be finite and positive")
    return number


def add_performance_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose overrides without making instrumentation opt-in."""
    parser.add_argument("--accept-slow", type=_nonempty_reason)
    parser.add_argument("--perf-estimate-seconds", type=_positive_number)
    parser.add_argument("--perf-min-decode-fraction", type=_positive_number, default=0.4)
    parser.add_argument("--perf-min-sm-utilization", type=_positive_number, default=50.0)
    parser.add_argument("--perf-max-estimate-multiplier", type=_positive_number, default=2.0)
    parser.add_argument("--perf-server-pid", type=int)
    parser.add_argument("--perf-startup-seconds", type=float, default=0.0)


def performance_cli_arguments(args: argparse.Namespace) -> tuple[str, ...]:
    """Forward the explicit policy to an isolated generation subprocess."""
    command: list[str] = []
    for name in (
        "accept_slow",
        "perf_estimate_seconds",
        "perf_min_decode_fraction",
        "perf_min_sm_utilization",
        "perf_max_estimate_multiplier",
        "perf_server_pid",
        "perf_startup_seconds",
    ):
        value: object = vars(args)[name]
        if value is not None:
            command.extend(("--" + name.replace("_", "-"), str(value)))
    return tuple(command)


def generation_monitor(args: argparse.Namespace) -> PerfMonitor:
    """Read local model metadata and monitor the HTTP server's metrics origin."""
    endpoint = urlsplit(args.endpoint)
    metrics_url = urlunsplit((endpoint.scheme, endpoint.netloc, "/metrics", "", ""))
    serving_config: Mapping[str, object] = args.serving_config or {}
    monitor = PerfMonitor(
        output_dir=Path(args.output_dir),
        metrics_url=metrics_url,
        concurrency=args.concurrency,
        model=load_model_geometry(
            args.model,
            revision=args.model_revision,
            kv_cache_dtype=str(serving_config.get("kv_cache_dtype", "auto")),
        ),
        estimate_seconds=args.perf_estimate_seconds,
        accept_slow=args.accept_slow,
        allowed_gpu_pids=() if args.perf_server_pid is None else (args.perf_server_pid,),
        thresholds=PerfThresholds(
            decode_fraction=args.perf_min_decode_fraction,
            sm_utilization=args.perf_min_sm_utilization,
            estimate_multiplier=args.perf_max_estimate_multiplier,
        ),
    )
    monitor.add_time("server_startup", args.perf_startup_seconds)
    return monitor


def record_episode_timings(monitor: PerfMonitor, records: Sequence[Mapping[str, object]]) -> None:
    """Aggregate concurrent episode work separately from sampled stage wall time."""
    context_weighted_tokens = 0.0
    generated_tokens = 0.0
    for record in records:
        summary = record.get("trace_summary")
        if not isinstance(summary, Mapping):
            raise TypeError("episode record must contain a trace_summary mapping")
        for category in ("model_call_seconds", "tool_execution_seconds", "grading_seconds"):
            value = summary.get(category)
            if value is not None:
                if not isinstance(value, (float, int)) or isinstance(value, bool):
                    raise TypeError(f"episode {category} must be numeric")
                monitor.add_time(category, value)
        trace_path_value = record.get("trace_path")
        if not isinstance(trace_path_value, str) or not trace_path_value:
            raise TypeError("episode record must contain a trace_path string")
        context = episode_context_usage(read_jsonl(Path(trace_path_value), hydrate=False))
        context_weighted_tokens += context["weighted_context_tokens"]
        generated_tokens += context["generated_tokens"]
    if generated_tokens:
        monitor.set_mean_context(context_weighted_tokens / generated_tokens)


def episode_context_usage(turns: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Estimate decode context using each turn's prompt plus half its generated tokens."""
    weighted_context_tokens = 0.0
    generated_tokens = 0.0
    for turn in turns:
        if turn.get("record") != "turn":
            continue
        prompt = turn.get("prompt_tokens")
        output = turn.get("output_tokens")
        if prompt is None or output is None:
            continue
        if not isinstance(prompt, (float, int)) or not isinstance(output, (float, int)):
            raise TypeError("turn token usage must be numeric")
        if prompt < 0 or output < 0:
            raise ValueError("turn token usage must be nonnegative")
        weighted_context_tokens += output * (prompt + output / 2)
        generated_tokens += output
    return {
        "weighted_context_tokens": weighted_context_tokens,
        "generated_tokens": generated_tokens,
    }
