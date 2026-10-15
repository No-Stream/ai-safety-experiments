"""Resumable, artifact-only replay of real token histories and synthetic controls.

CPU validation: python -m reward_hacking.replay_benchmark --dry-run
GPU execution belongs inside the resource limiter and a persistent tmux session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any, cast

import httpx
from huggingface_hub import snapshot_download

from reward_hacking.ladder.thinking import QWEN_THINKING_REASONING_CONFIG
from reward_hacking.replay_measure import measure_cell
from reward_hacking.replay_workload import (
    extract_workloads,
    roofline_tokens_per_second,
    synthetic_workloads,
)
from reward_hacking.vllm_server import VLLMServer, VLLMServerConfig

logger = logging.getLogger(__name__)
LENGTHS = (4096, 16384, 32768, 65536)
CONCURRENCIES = (1, 4, 6)
MIN_DECODE_TOKENS = 2


@dataclass(frozen=True, slots=True)
class ReplayServerConfig(VLLMServerConfig):
    """Extend the shared launcher only for verified experimental CLI switches."""

    async_scheduling: bool | None = None
    capture_sizes: tuple[int, ...] = ()
    prefix_disabled: bool = False

    def command(self) -> list[str]:
        """Keep lifecycle, logging and memory fitting in the shared launcher."""
        command = super(ReplayServerConfig, self).command()
        if self.async_scheduling is not None:
            command.append(
                "--async-scheduling" if self.async_scheduling else "--no-async-scheduling"
            )
        if self.capture_sizes:
            command.extend(["--cudagraph-capture-sizes", *map(str, self.capture_sizes)])
        if self.prefix_disabled:
            command.append("--no-enable-prefix-caching")
        return command


@dataclass(frozen=True)
class Configuration:
    """One independent change relative to the explicit baseline."""

    name: str
    server: ReplayServerConfig
    changes: dict[str, Any] = field(default_factory=dict)
    remove_thinking_budget: bool = False
    omp_threads: int = 16


def configurations(baseline: ReplayServerConfig, quota_threads: int) -> list[Configuration]:
    """Order likely scheduling/sampler wins ahead of memory/cache experiments."""
    return [
        Configuration("baseline", baseline),
        Configuration(
            "no-thinking-budget",
            baseline,
            {"thinking_token_budget": None},
            remove_thinking_budget=True,
        ),
        Configuration(
            "async", replace(baseline, async_scheduling=True), {"async_scheduling": True}
        ),
        Configuration(
            "omp-quota", baseline, {"OMP_NUM_THREADS": quota_threads}, omp_threads=quota_threads
        ),
        Configuration(
            "prefill-16k",
            replace(baseline, max_num_batched_tokens=16384),
            {"max_num_batched_tokens": 16384},
        ),
        Configuration(
            "graphs",
            replace(baseline, capture_sizes=(1, 2, 4, 6, 8, 16)),
            {"cudagraph_capture_sizes": [1, 2, 4, 6, 8, 16]},
        ),
        Configuration(
            "attention-default",
            replace(baseline, attention_backend=None),
            {"attention_backend": None},
        ),
        Configuration(
            "kv-auto", replace(baseline, kv_cache_dtype="auto"), {"kv_cache_dtype": "auto"}
        ),
        Configuration(
            "prefix-off",
            replace(baseline, enable_prefix_caching=False, prefix_disabled=True),
            {"enable_prefix_caching": False},
        ),
    ]


def cgroup_quota_threads() -> int:
    """Read ancestor quotas; the limiter's cgroup can be below a tighter parent."""
    relative = Path("/proc/self/cgroup").read_text().strip().split("0::", 1)[1]
    current = Path("/sys/fs/cgroup") / relative.lstrip("/")
    quotas = [len(os.sched_getaffinity(0))]
    for directory in (current, *current.parents):
        if directory == Path("/sys"):
            break
        quota_file = directory / "cpu.max"
        if quota_file.exists():
            quota, period = quota_file.read_text().split()
            if quota != "max":
                quotas.append(max(1, math.floor(int(quota) / int(period))))
    return min(quotas)


def verify_installed_flags() -> dict[str, str]:
    """Inspect installed CLI registration without importing the CUDA engine."""
    installed_version = version("vllm")
    if installed_version != "0.31.0":
        raise ValueError(f"Revalidate CLI flags for vLLM {installed_version}; expected 0.31.0")
    source = Path(str(distribution("vllm").locate_file("vllm/engine/arg_utils.py")))
    text = source.read_text()
    flags = (
        "--async-scheduling",
        "--cudagraph-capture-sizes",
        "--max-num-batched-tokens",
        "--attention-backend",
        "--kv-cache-dtype",
        "--enable-prefix-caching",
        "--mamba-cache-mode",
    )
    for flag in flags:
        if f'"{flag}"' not in text:
            raise ValueError(f"Installed CLI does not register {flag}")
    return {"vllm_version": installed_version, "cli_source": str(source)}


def validate_resume(record: dict[str, Any], identity: dict[str, Any]) -> None:
    """Compare semantic inputs directly, including full prompt IDs; no hash gates."""
    if record["identity"] != identity:
        raise ValueError("Replay inputs changed; use a new output directory")


def write_json(path: Path, value: object) -> None:
    """Replace complete JSON records atomically after successful measurement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    """Expose small, explicit replay controls."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", default="RedHatAI/Qwen3.8-27B-INT4")
    result.add_argument("--revision", default="main")
    result.add_argument(
        "--traces", type=Path, default=Path("artifacts/organism-gen/seed-v4-hackgen-wave1")
    )
    result.add_argument("--output", type=Path, default=Path("artifacts/perf/27b-replay"))
    result.add_argument("--max-tokens", type=int, default=64)
    result.add_argument("--budget-minutes", type=float, default=45)
    result.add_argument("--port", type=int, default=8124)
    result.add_argument("--configs", nargs="+", default=["all"])
    result.add_argument("--dry-run", action="store_true")
    return result


def optional_number(value: float | None) -> str:
    """Keep unavailable telemetry distinct from measured zeros."""
    return "unknown" if value is None else f"{value:.3f}"


def render_report(output: Path, records: list[dict[str, Any]], *, complete: bool) -> None:
    """Regenerate the artifact table from persisted cell results."""
    lines = [
        "# Replay performance",
        "",
        (
            "| Config | Workload | Length | Concurrency | Decode tok/s | Prefill tok/request-s | "
            "TTFT s | ITL s | SM % | Memory busy % | Roof fraction |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {record['configuration']} | {record['kind']} | {record['length']} | "
        f"{record['concurrency']} | {record['aggregate_decode_tokens_s']:.2f} | "
        f"{optional_number(record.get('computed_prompt_tokens_per_request_second'))} | "
        f"{optional_number(record.get('mean_ttft_seconds'))} | "
        f"{optional_number(record.get('mean_itl_seconds'))} | "
        f"{optional_number(record['gpu_telemetry'].get('sm_utilization_pct_mean'))} | "
        f"{optional_number(record['gpu_telemetry'].get('memory_controller_busy_pct_mean'))} | "
        f"{record['roof_fraction']:.3f} |"
        for record in records
    )
    baseline = {
        (r["kind"], r["length"], r["concurrency"]): r
        for r in records
        if r["configuration"] == "baseline"
    }
    comparisons: dict[str, list[float]] = {}
    for record in records:
        key = (record["kind"], record["length"], record["concurrency"])
        if record["configuration"] != "baseline" and key in baseline:
            comparisons.setdefault(record["configuration"], []).append(
                record["aggregate_decode_tokens_s"] / baseline[key]["aggregate_decode_tokens_s"]
            )
    full = {
        name: math.exp(sum(map(math.log, values)) / len(values))
        for name, values in comparisons.items()
        if len(values) == len(LENGTHS) * len(CONCURRENCIES) * 2
    }
    verdict = "No configuration recommendation yet; no complete comparison has been measured."
    if full:
        full["baseline"] = 1.0
        best = max(full, key=lambda name: full[name])
        selected = [r for r in records if r["configuration"] == best]
        fraction = sum(r["roof_fraction"] for r in selected) / len(selected)
        diagnosis = {
            "async": "The isolated scheduling change suggests scheduler overhead contributed.",
            "omp-quota": "The isolated OMP change suggests CPU scheduling or thread contention contributed.",
            "no-thinking-budget": "The isolated sampler change suggests thinking-processor overhead contributed; "
            "removing the budget changes generation policy and needs a separate behavioral decision.",
        }.get(best, "The bottleneck remains unresolved by these measurements.")
        verdict = (
            f"Best measured isolated change: {best}, geometric mean decode speedup {full[best]:.2f}x "
            f"versus matched baseline cells; mean simplified bandwidth roof fraction {fraction:.1%}. "
            f"{diagnosis} "
            "Adopt only if latency, preemptions and cache counters in the cell records are acceptable. "
            "NVML busy time cannot distinguish memory bandwidth saturation from compute or kernel latency; "
            "causal attribution remains provisional pending hardware profiling."
        )
    lines.extend(
        [
            "",
            verdict,
            "",
            f"Requested matrix complete: {complete}.",
            "",
            (
                "Roof omits activation traffic, dequantization and launch overhead. Decode-window throughput "
                "can include overlapping prefill and request ramp-up. NVML memory utilization is "
                "controller busy time, not measured bytes/s. Streaming ITL can include transport coalescing."
            ),
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def prepare(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Configuration]]:
    """Freeze runtime-only token workloads, config values and sampling for repeatable resume."""
    snapshot = Path(
        snapshot_download(
            args.model,
            revision=args.revision,
            local_files_only=True,
            allow_patterns=["config.json"],
        )
    )
    model_config = cast("dict[str, Any]", json.loads((snapshot / "config.json").read_text()))
    text_config = cast("dict[str, Any]", model_config.get("text_config", model_config))
    baseline = ReplayServerConfig(
        model=args.model,
        revision=None if Path(args.model).is_dir() else snapshot.name,
        max_model_len=98304,
        gpu_memory_utilization=0.9,
        language_model_only=True,
        max_num_seqs=16,
        dtype="bfloat16",
        attention_backend="FLASHINFER",
        enable_prefix_caching=True,
        kv_cache_dtype="fp8",
        mamba_cache_mode="align",
        max_num_batched_tokens=8192,
        reasoning_config=dict(QWEN_THINKING_REASONING_CONFIG),
        port=args.port,
    )
    variants = configurations(baseline, cgroup_quota_threads())
    names = {variant.name for variant in variants}
    if args.configs != ["all"]:
        if not set(args.configs) <= names:
            raise ValueError(f"Unknown configs: {set(args.configs) - names}")
        variants = [variant for variant in variants if variant.name in args.configs]
    workload_path = args.output / "workloads.json"
    if workload_path.exists():
        workload_record = cast("dict[str, Any]", json.loads(workload_path.read_text()))
        if workload_record["model_snapshot"] != str(snapshot):
            raise ValueError("Model snapshot changed; use a new output directory")
        workloads = cast("list[dict[str, Any]]", workload_record["workloads"])
    else:
        workloads = build_workloads(args.traces, int(text_config["vocab_size"]))
        write_json(workload_path, {"model_snapshot": str(snapshot), "workloads": workloads})
    return workloads, model_config, variants


def build_workloads(traces: Path, vocab_size: int) -> list[dict[str, Any]]:
    """Extract prompts at runtime, never embedding benchmark material in source."""
    real = extract_workloads(traces)
    synthetic = synthetic_workloads(vocab_size, sampling=real[0].sampling)
    return [asdict(workload) for workload in [*real, *synthetic]]


def cell_payloads(  # noqa: PLR0913, PLR0917 - explicit cell coordinates
    workloads: list[dict[str, Any]],
    variant: Configuration,
    kind: str,
    length: int,
    concurrency: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Use the same nested prompt subsets at every concurrency and configuration."""
    selected = [w for w in workloads if w["kind"] == kind and w["target_length"] == length][
        :concurrency
    ]
    if len(selected) != concurrency:
        raise ValueError(f"Insufficient frozen workloads for {kind}/{length}/{concurrency}")
    payloads: list[dict[str, Any]] = []
    for workload in selected:
        sampling = dict(workload["sampling"])
        sampling.pop("max_new_tokens", None)
        if not sampling.pop("do_sample", True):
            raise ValueError("Replay expects sampled generation with do_sample=True")
        sampling["max_tokens"] = max_tokens
        if variant.remove_thinking_budget:
            sampling.pop("thinking_token_budget", None)
        payloads.append({**sampling, "prompt": workload["prompt_ids"]})
    return payloads


async def run(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915 - linear resume loop
    """Stop between cells at the soft budget; every completed cell survives interruption."""
    if args.max_tokens < MIN_DECODE_TOKENS or args.budget_minutes <= 0:
        raise ValueError("max_tokens must be >=2 and budget_minutes must be positive")
    verified = verify_installed_flags()
    workloads, model_config, variants = prepare(args)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "verified": verified,
        "configurations": [v.name for v in variants],
        "lengths": LENGTHS,
        "concurrencies": CONCURRENCIES,
        "kinds": ["trace", "synthetic"],
        "cells": len(variants) * len(LENGTHS) * len(CONCURRENCIES) * 2,
        "requests": len(variants) * len(LENGTHS) * sum(CONCURRENCIES) * 2,
        "total_prompt_tokens": sum(
            len(payload["prompt"])
            for variant in variants
            for kind in ("trace", "synthetic")
            for length in LENGTHS
            for concurrency in CONCURRENCIES
            for payload in cell_payloads(
                workloads, variant, kind, length, concurrency, args.max_tokens
            )
        ),
        "max_tokens": args.max_tokens,
        "budget_minutes": args.budget_minutes,
        "unsupported_variants": {
            "mamba-all": "HF model disables all mode; vLLM rewrites it to align",
            "mamba-none-with-prefix": "vLLM rewrites none to align when prefix caching is enabled",
        },
        "baseline_scheduler": "Unset, matching the wave launcher. Installed vLLM enables async by default "
        "when compatible; explicit async-on may be a no-op. The diagnosis's "
        "claim that unset selects synchronous scheduling contradicts installed source.",
        "cache_regime": "cold cache reset before every cell; shared prefixes within concurrent requests retained",
        "expectation": "Async scheduling or removing the thinking processor may improve decode; "
        "context-growing full-attention KV traffic alone is unlikely to explain the observed gap.",
        "limitations": [
            (
                "64 generated tokens measures steady decode briefly; it does not reach the 2048 "
                "thinking-budget boundary and cannot measure boundary-forcing cost."
            ),
            "Full matrix may exceed budget because each engine restart previously took 4-5 minutes.",
        ],
    }
    write_json(args.output / "plan.json", plan)
    write_json(
        args.output / "roofline.json",
        {
            "bandwidth_bytes_s": 1792e9,
            "bandwidth_source": "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/",
            "assumed_weight_bytes": 14e9,
            "model_text_config": model_config.get("text_config", model_config),
            "limitations": "Weight bytes are an assumption; recurrent state, activation traffic, "
            "quantization overhead and scheduling are omitted.",
            "estimates": [
                {
                    "context": length,
                    "batch": concurrency,
                    "tokens_s": roofline_tokens_per_second(model_config, length, concurrency),
                }
                for length in LENGTHS
                for concurrency in CONCURRENCIES
            ],
        },
    )
    logger.info("Replay plan: %s", plan)
    if args.dry_run:
        for variant in variants:
            variant.server.command()
            for kind in ("trace", "synthetic"):
                for length in LENGTHS:
                    for concurrency in CONCURRENCIES:
                        cell_payloads(
                            workloads, variant, kind, length, concurrency, args.max_tokens
                        )
        if not (args.output / "summary.md").exists():
            render_report(args.output, [], complete=False)
        logger.info("CPU dry-run validated all commands and exact token payloads; GPU untouched")
        return
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    exhausted = False
    server: VLLMServer | None = None
    previous_launch: tuple[list[str], int] | None = None
    startup_seconds = 0.0
    with ExitStack() as servers:
        for variant in variants:
            pending: list[tuple[Path, dict[str, Any], str, int, int, list[dict[str, Any]]]] = []
            os.environ["OMP_NUM_THREADS"] = str(variant.omp_threads)
            server_config = replace(
                variant.server, log_path=args.output / variant.name / "server.log"
            )
            for kind in ("trace", "synthetic"):
                for length in LENGTHS:
                    for concurrency in CONCURRENCIES:
                        payloads = cell_payloads(
                            workloads, variant, kind, length, concurrency, args.max_tokens
                        )
                        identity = {
                            "server_command": server_config.command(),
                            "omp_threads": variant.omp_threads,
                            "payloads": payloads,
                            "model_config": model_config,
                        }
                        path = (
                            args.output
                            / variant.name
                            / f"{kind}-{length}-{concurrency}"
                            / "result.json"
                        )
                        if path.exists():
                            record = cast("dict[str, Any]", json.loads(path.read_text()))
                            validate_resume(record, identity)
                            records.append(record)
                        else:
                            pending.append((path, identity, kind, length, concurrency, payloads))
            if not pending:
                continue
            if exhausted or time.monotonic() - started >= args.budget_minutes * 60:
                exhausted = True
                continue
            launch_key = (variant.server.command(), variant.omp_threads)
            if launch_key != previous_launch:
                servers.close()
                startup = time.monotonic()
                server = servers.enter_context(VLLMServer(server_config))
                startup_seconds = time.monotonic() - startup
                previous_launch = launch_key
            if server is None:
                raise RuntimeError("Replay server was not started")
            warmup = {**pending[0][-1][0], "prompt": [100, 101, 102], "max_tokens": 2}
            await measure_cell(
                server.base_url,
                server_config.model,
                [warmup],
                measurement_directory(args.output / variant.name / "warmup"),
            )
            for path, identity, kind, length, concurrency, payloads in pending:
                if time.monotonic() - started >= args.budget_minutes * 60:
                    exhausted = True
                    break
                await reset_cache(server.base_url)
                attempt = measurement_directory(path.parent)
                result = await measure_cell(server.base_url, server_config.model, payloads, attempt)
                actual_context = round(
                    sum(len(p["prompt"]) for p in payloads) / concurrency + args.max_tokens / 2
                )
                roof = roofline_tokens_per_second(
                    model_config,
                    actual_context,
                    concurrency,
                    kv_bytes=2 if server_config.kv_cache_dtype == "auto" else 1,
                )
                decode_rate = result["aggregate_decode_tokens_s"]
                if decode_rate is None or decode_rate <= 0:
                    raise ValueError(
                        "No measurable decode interval; raw attempt retained. Increase --max-tokens."
                    )
                record = {
                    **result,
                    "identity": identity,
                    "measurement_directory": str(attempt.relative_to(args.output)),
                    "configuration": variant.name,
                    "kind": kind,
                    "length": length,
                    "concurrency": concurrency,
                    "startup_seconds": startup_seconds,
                    "roof_tokens_s": roof,
                    "roof_fraction": (
                        result["steady_decode_tokens_s"]
                        if result["steady_decode_tokens_s"] is not None
                        else decode_rate
                    )
                    / roof,
                    "roof_rate_source": "all-requests-active decode"
                    if result["steady_decode_tokens_s"] is not None
                    else "decode-window aggregate including ramp-up",
                }
                write_json(path, record)
                records.append(record)
                render_report(args.output, records, complete=False)
    render_report(args.output, records, complete=not exhausted)
    if exhausted:
        logger.warning(
            "Replay budget reached; saved cells are resumable. Requested matrix remains incomplete."
        )


def measurement_directory(parent: Path) -> Path:
    """Retain interrupted raw attempts while re-running only incomplete cells."""
    parent.mkdir(parents=True, exist_ok=True)
    attempt_number = 1
    while (parent / f"attempt-{attempt_number}").exists():
        attempt_number += 1
    destination = parent / f"attempt-{attempt_number}"
    destination.mkdir()
    return destination


async def reset_cache(base_url: str) -> None:
    """Keep cold-cache cell results invariant to skipped cells on resume."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{base_url}/reset_prefix_cache")
        response.raise_for_status()
        if response.json() != {"success": True}:
            raise RuntimeError("Prefix cache reset failed; refusing a cache-confounded cell")


def main() -> None:
    """Run as a module from the repository root."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(parser().parse_args()))


if __name__ == "__main__":
    main()
