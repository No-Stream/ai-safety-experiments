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
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any, cast

import httpx
from huggingface_hub import snapshot_download

from reward_hacking.ladder.thinking import QWEN_THINKING_REASONING_CONFIG
from reward_hacking.replay_measure import measure_cell, measure_sequence_cell
from reward_hacking.replay_workload import (
    extract_episode_workloads,
    extract_workloads,
    roofline_tokens_per_second,
    synthetic_workloads,
)
from reward_hacking.vllm_server import VLLMServer, VLLMServerConfig

logger = logging.getLogger(__name__)
LENGTHS = (4096, 16384, 32768, 65536)
CONCURRENCIES = (4, 6)
MIN_DECODE_TOKENS = 2
MIN_BASELINE_PROBES = 2
HF_COMMIT_HEX_LENGTH = 40


@dataclass(frozen=True, slots=True)
class ReplayServerConfig(VLLMServerConfig):
    """Extend the shared launcher only for verified experimental CLI switches."""

    async_scheduling: bool | None = None
    capture_sizes: tuple[int, ...] = ()
    prefix_disabled: bool = False
    profiler_directory: str | None = None

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
        if self.profiler_directory is not None:
            command.extend(
                [
                    "--profiler-config",
                    json.dumps(
                        {
                            "profiler": "torch",
                            "torch_profiler_dir": self.profiler_directory,
                            "torch_profiler_with_stack": False,
                            "torch_profiler_use_gzip": True,
                            "detailed_trace_annotation": True,
                            "max_iterations": 64,
                        }
                    ),
                ]
            )
        return command


@dataclass(frozen=True)
class Configuration:
    """One independent change relative to the explicit baseline."""

    name: str
    server: ReplayServerConfig
    changes: dict[str, Any] = field(default_factory=dict)
    remove_thinking_budget: bool = False
    omp_threads: int = 16


def configurations(
    baseline: ReplayServerConfig, quota_threads: int, *, mtp_supported: bool = False
) -> list[Configuration]:
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
        *[
            Configuration(
                f"prefill-{tokens // 1024}k",
                replace(baseline, max_num_batched_tokens=tokens),
                {"max_num_batched_tokens": tokens},
            )
            for tokens in (2048, 4096, 16384)
        ],
        Configuration(
            "omp-quota", baseline, {"OMP_NUM_THREADS": quota_threads}, omp_threads=quota_threads
        ),
        *(
            [
                Configuration(
                    "mtp",
                    replace(
                        baseline, speculative_config={"method": "mtp", "num_speculative_tokens": 2}
                    ),
                    {"speculative_config": {"method": "mtp", "num_speculative_tokens": 2}},
                )
            ]
            if mtp_supported
            else []
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
        "--profiler-config",
        "--speculative-config",
        "--max-num-batched-tokens",
        "--attention-backend",
        "--kv-cache-dtype",
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "--max-model-len",
        "--gpu-memory-utilization",
        "--language-model-only",
        "--max-num-seqs",
        "--dtype",
        "--revision",
        "--reasoning-config",
    )
    for flag in flags:
        if f'"{flag}"' not in text:
            raise ValueError(f"Installed CLI does not register {flag}")
    profiler_source = source.parent.parent / "entrypoints/serve/profile/api_router.py"
    profiler_text = profiler_source.read_text()
    for endpoint in ('"/start_profile"', '"/stop_profile"'):
        if endpoint not in profiler_text:
            raise ValueError("Profiling endpoint changed; revalidate source")
    frontend_source = source.parent.parent / "entrypoints/launchers/cli_args.py"
    frontend = frontend_source.read_text()
    if "host: str | None = None" not in frontend or "port: int = 8000" not in frontend:
        raise ValueError("Frontend host/port registration changed; revalidate installed source")
    return {
        "vllm_version": installed_version,
        "cli_source": str(source),
        "profiler_source": str(profiler_source),
    }


def checkpoint_capabilities(snapshot: Path) -> dict[str, Any]:
    """Check locally declared MTP tensors and installed hybrid-model implementation without CUDA."""
    source = Path(str(distribution("vllm").locate_file("vllm")))
    speculative = (source / "config/speculative.py").read_text()
    predictor = (source / "model_executor/models/qwen3_5_mtp.py").read_text()
    config = json.loads((snapshot / "config.json").read_text())
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"weight_map": {}}
    tensors = {name: file for name, file in index["weight_map"].items() if name.startswith("mtp.")}
    supported = (
        config.get("model_type") == "qwen3_5"
        and bool(tensors)
        and all((snapshot / file).is_file() for file in tensors.values())
        and "Qwen3_5MTP" in speculative
        and 'name.startswith("mtp.")' in predictor
    )
    return {
        "mtp_supported_by_source_and_metadata": supported,
        "mtp_tensor_count": len(tensors),
        "missing_weight_files": sorted(
            {file for file in index["weight_map"].values() if not (snapshot / file).is_file()}
        ),
        "limitation": "Source and weight-index compatibility only; live INT4 MTP loading remains unverified",
    }


def validate_resume(record: dict[str, Any], identity: dict[str, Any]) -> None:
    """Compare semantic inputs directly, including full prompt IDs; no hash gates."""
    if record["identity"] != identity:
        raise ValueError("Replay inputs changed; use a new output directory")


def write_json(path: Path, value: object) -> None:
    """Replace complete JSON records atomically after successful measurement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    """Expose small, explicit replay controls."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", default="RedHatAI/Qwen3.8-27B-INT4")
    result.add_argument("--revision", default="main")
    result.add_argument(
        "--traces", type=Path, default=Path("artifacts/organism-gen/seed-v4-hackgen-wave1")
    )
    result.add_argument("--output", type=Path, default=Path("artifacts/perf/27b-replay-run2"))
    result.add_argument("--max-tokens", type=int, default=8)
    result.add_argument("--budget-minutes", type=float, default=55)
    result.add_argument("--port", type=int, default=8124)
    result.add_argument("--configs", nargs="+", default=["all"])
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--reference-run", type=Path, default=Path("artifacts/perf/27b-replay-run1")
    )
    result.add_argument(
        "--sequence-scope", choices=("growth-range", "whole-episode"), default="growth-range"
    )
    result.add_argument("--drift-threshold", type=float, default=0.15)
    result.add_argument("--config-seconds", type=float, default=420)
    result.add_argument("--startup-reserve-seconds", type=float, default=300)
    return result


def optional_number(value: float | None) -> str:
    """Keep unavailable telemetry distinct from measured zeros."""
    return "unknown" if value is None else f"{value:.3f}"


def visible_drift_causes(probe: dict[str, Any]) -> list[str]:
    """Report observed resource constraints without claiming they caused a slowdown."""
    causes: list[str] = []
    cpu = probe.get("cpu_telemetry", {})
    stat_delta = cpu.get("cgroup_cpu_stat", {}).get("delta") or {}
    if stat_delta.get("nr_throttled", 0) > 0:
        causes.append("cgroup CPU throttling observed")
    pressure = cpu.get("cgroup_pressure_total_us_delta") or {}
    if pressure.get("cpu", {}).get("some", 0) > 0:
        causes.append("cgroup CPU pressure observed; inspect its magnitude in CPU samples")
    load = cpu.get("host_load_1_mean")
    effective_cpus = cpu.get("host_effective_cpu_count")
    if load is not None and effective_cpus is not None and load > effective_cpus:
        causes.append("host load exceeds effective CPU count")
    telemetry = probe.get("gpu_telemetry", {})
    if telemetry.get("throttle_reasons_nonzero", False):
        causes.append("GPU throttle reason reported")
    return causes


def assess_drift(probes: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Refuse comparisons unless identical baseline probes bound the whole session."""
    if not 0 < threshold < 1:
        raise ValueError("drift threshold must be between zero and one")
    rates = [float(p["aggregate_decode_tokens_s"]) for p in probes]
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Baseline probes require positive finite decode rates")
    drift = max(rates) / min(rates) - 1 if len(rates) >= MIN_BASELINE_PROBES else None
    auxiliary_drift = {}
    for metric in ("output_tokens_per_wall_second", "mean_ttft_seconds"):
        values = [p[metric] for p in probes if p.get(metric) is not None and p[metric] > 0]
        if len(values) == len(probes) and len(values) >= MIN_BASELINE_PROBES:
            auxiliary_drift[metric] = max(values) / min(values) - 1
    if drift is not None and auxiliary_drift:
        drift = max(drift, *auxiliary_drift.values())
    causes = [cause for probe in probes for cause in visible_drift_causes(probe)]
    if not causes:
        causes.append(
            "unresolved; inspect host samples, engine/API CPU, clocks, power and thermal telemetry"
        )
    return {
        "valid": drift is not None and drift <= threshold,
        "drift_fraction": drift,
        "threshold": threshold,
        "probe_count": len(probes),
        "causes": sorted(set(causes)),
        "auxiliary_drift": auxiliary_drift,
    }


def profiler_report_lines(records: list[dict[str, Any]]) -> list[str]:
    """Render observed kernel timing separately from unprofiled configuration comparisons."""
    lines: list[str] = []
    profiles = [r["profiler_summary"] for r in records if r.get("profiler_summary")]
    for profile in profiles:
        lines += ["", "## Baseline profiler", "", f"Capture status: {profile['status']}."]
        if profile["status"] == "available":
            lines += [
                "",
                (
                    f"GPU non-kernel time (upper bound on host waiting): "
                    f"{profile.get('possible_host_idle_upper_bound_seconds')} seconds; "
                    f"fraction {profile.get('possible_non_kernel_fraction_upper_bound')}. "
                    "Actual time waiting on the host remains unidentifiable from kernel gaps alone."
                ),
                "",
                "| Kernel group | Kernel | Total GPU ms |",
                "| --- | --- | ---: |",
            ]
            lines.extend(
                f"{group}: {values}."
                for group, values in profile.get("gpu_kernel_duration_by_category", {}).items()
            )
            lines += ["", f"Scheduler step summary: {profile.get('scheduler_step_summary')}", ""]
            for group, kernels in profile["top_gpu_kernels_by_category"].items():
                lines.extend(
                    f"| {group} | {kernel['name']} | {kernel['total_milliseconds']:.3f} |"
                    for kernel in kernels
                )
    return lines


def render_report(output: Path, records: list[dict[str, Any]], *, complete: bool) -> None:
    """Make missing drift controls and missing comparisons explicit."""
    probes = [r for r in records if r["kind"] == "baseline-probe"]
    contributing_sessions = {r.get("session") for r in records if r["kind"] == "sequence"}
    if probes:
        contributing_sessions.add(probes[-1].get("session"))
    probes = [p for p in probes if p.get("session") in contributing_sessions]
    threshold = float(probes[0].get("drift_threshold", 0.15)) if probes else 0.15
    drift = assess_drift(probes, threshold)
    lines = [
        "# Replay performance",
        "",
        "## Verdict",
        "",
        (
            "Bottleneck class: unresolved pending profiler and concurrent-prefill evidence. "
            "Candidate classes: prefill compute, linear-attention kernel, host/CPU, scheduling stall, bandwidth."
        ),
        "",
        f"Baseline drift across contributing sessions: {drift}. Requested matrix complete: {complete}.",
        "",
    ]
    compared: dict[str, list[float]] = {}
    baseline = {
        (r["kind"], r["concurrency"]): r for r in records if r["configuration"] == "baseline"
    }
    # Old runs lack actual baseline controls and are never promoted to valid comparisons.
    final_control = bool(probes and probes[-1].get("final_probe", False))
    if drift["valid"] and final_control:
        for name in sorted({r["configuration"] for r in records} - {"baseline"}):
            ratios: list[float] = []
            for concurrency in CONCURRENCIES:
                candidate = [
                    r
                    for r in records
                    if r["configuration"] == name
                    and r["kind"] == "sequence"
                    and r["concurrency"] == concurrency
                ]
                key = ("sequence", concurrency)
                if candidate and key in baseline:
                    ratios.append(
                        candidate[0]["output_tokens_per_wall_second"]
                        / baseline[key]["output_tokens_per_wall_second"]
                    )
            if len(ratios) == len(CONCURRENCIES):
                compared[name] = ratios
        if compared:
            speedups = {
                name: math.exp(sum(map(math.log, rates)) / len(rates))
                for name, rates in compared.items()
            }
            speedups["baseline"] = 1.0
            best = max(speedups, key=lambda name: speedups[name])
            lines.append(
                f"Recommended measured configuration: {best}; geometric mean end-to-end speedup "
                f"{speedups[best]:.3f}x over matched c=4/c=6 episode sequences; "
                f"baseline drift bound {drift['drift_fraction']:.1%}."
            )
        else:
            lines.append("Recommendation withheld: no complete matched c=4/c=6 comparison.")
    else:
        lines.append(
            "Comparison REFUSED: baseline controls missing, final baseline absent, or drift exceeds threshold."
        )
    lines += [
        "",
        (
            "Next lever if throughput remains far below the conditional bandwidth roof: inspect kernel time "
            "and scheduler overlap first. History truncation changes the experiment and requires the owner."
        ),
        "",
        (
            "No-thinking-budget changes generation policy; any speedup requires a separate behavior decision. "
            "NVML busy time is not bandwidth. HTTP overlap is correlation, not causal attribution. "
            "Profiled cells are excluded from speed comparisons."
        ),
        "",
        "| Config | Kind | Concurrency | Wall s | Decode tok/s |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {r['configuration']} | {r['kind']} | {r['concurrency']} | "
        f"{r.get('wall_seconds', 0):.2f} | {r['aggregate_decode_tokens_s']:.2f} |"
        for r in records
    ]
    lines.extend(profiler_report_lines(records))
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(output / "drift.json", drift)
    (output / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def prepare(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Configuration]]:
    """Freeze runtime-only token workloads, config values and sampling for repeatable resume."""
    snapshot = (
        Path(args.model)
        if Path(args.model).is_dir()
        else Path(
            snapshot_download(
                args.model,
                revision=args.revision,
                local_files_only=True,
                allow_patterns=["config.json"],
            )
        )
    )
    if Path(args.model).is_dir() and (
        len(snapshot.name) != HF_COMMIT_HEX_LENGTH
        or any(c not in "0123456789abcdef" for c in snapshot.name)
    ):
        raise ValueError(
            "Local replay model must be an immutable commit-named Hugging Face snapshot"
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
    capabilities = checkpoint_capabilities(snapshot)
    write_json(args.output / "capabilities.json", capabilities)
    variants = configurations(
        baseline,
        cgroup_quota_threads(),
        mtp_supported=capabilities["mtp_supported_by_source_and_metadata"],
    )
    names = {variant.name for variant in variants}
    if args.configs != ["all"]:
        if not set(args.configs) <= names:
            raise ValueError(f"Unknown configs: {set(args.configs) - names}")
        variants = [variant for variant in variants if variant.name in args.configs]
    workload_path = args.output / "workloads-v3.json"
    if workload_path.exists():
        workload_record = cast("dict[str, Any]", json.loads(workload_path.read_text()))
        if Path(workload_record["model_snapshot"]).name != snapshot.name:
            raise ValueError("Model snapshot changed; use a new output directory")
        workloads = cast("list[dict[str, Any]]", workload_record["workloads"])
    else:
        reference = args.reference_run / "workloads.json"
        if reference.exists():
            frozen = json.loads(reference.read_text())
            if Path(frozen["model_snapshot"]).name != snapshot.name:
                raise ValueError("Reference model snapshot differs from selected model")
            workloads = frozen["workloads"]
        else:
            workloads = build_workloads(args.traces, int(text_config["vocab_size"]))
        episodes = extract_episode_workloads(args.traces)
        workloads += [{"kind": "episode", **asdict(episode)} for episode in episodes]
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


def sequence_payloads(
    workloads: list[dict[str, Any]],
    variant: Configuration,
    concurrency: int,
    max_tokens: int,
    scope: str = "growth-range",
) -> list[list[dict[str, Any]]]:
    """Freeze original adjacent turns; new sampled outputs never rewrite the next prompt."""
    episodes = [w for w in workloads if w["kind"] == "episode"][:concurrency]
    if len(episodes) != concurrency:
        raise ValueError("Insufficient independent episode sequences")
    return [
        [
            cell_payloads(
                [
                    {
                        "kind": "trace",
                        "target_length": 1,
                        "prompt_ids": turn["prompt_ids"],
                        "sampling": turn["sampling"],
                    }
                ],
                variant,
                "trace",
                1,
                1,
                max_tokens,
            )[0]
            for turn in (
                episode["turns"][episode["turn_4k_index"] : episode["turn_64k_index"] + 1]
                if scope == "growth-range"
                else episode["turns"]
            )
        ]
        for episode in episodes
    ]


def resume_identity(
    variant: Configuration,
    sequences: list[list[dict[str, Any]]],
    model_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Bind measurement semantics as well as prompt IDs to resumable cells."""
    return {
        "schema": 2,
        "server_command": variant.server.command(),
        "omp_threads": variant.omp_threads,
        "sequences": sequences,
        "model_config": model_config,
        "config_seconds": args.config_seconds,
        "drift_threshold": args.drift_threshold,
        "sequence_scope": args.sequence_scope,
    }


async def run(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915 - linear experiment lifecycle
    """Persist cells, reserve real baseline restarts, and never compare unbounded drift."""
    if args.max_tokens < MIN_DECODE_TOKENS or args.budget_minutes <= 0 or args.config_seconds <= 0:
        raise ValueError("Positive time budgets and max_tokens>=2 are required")
    if args.startup_reserve_seconds <= 0 or not 0 < args.drift_threshold < 1:
        raise ValueError("Positive startup reserve and drift threshold in (0,1) are required")
    verified = verify_installed_flags()
    workloads, model_config, variants = prepare(args)
    baseline = configurations(variants[0].server, cgroup_quota_threads())[0]
    # Always use the production baseline for controls even when selecting a config subset.
    if variants[0].name != "baseline":
        raise ValueError("Include baseline first in --configs to establish matched controls")
    sequences = sequence_payloads(workloads, baseline, 6, args.max_tokens, args.sequence_scope)
    if any(
        len(payload["prompt"]) + args.max_tokens > baseline.server.max_model_len
        for sequence in sequences
        for payload in sequence
    ):
        raise ValueError("Frozen episode prompt exceeds replay model context")
    plan = {
        "schema": 2,
        "verified": verified,
        "configurations": [v.name for v in variants],
        "concurrencies": CONCURRENCIES,
        "cells_per_config": 2,
        "episode_lengths": [[len(p["prompt"]) for p in sequence] for sequence in sequences],
        "sequence_scope": args.sequence_scope,
        "requests_per_config": sum(len(seq) for seq in sequences[:4]) + sum(map(len, sequences)),
        "budget_minutes": args.budget_minutes,
        "config_seconds": args.config_seconds,
        "max_tokens": args.max_tokens,
        "cold_control": "one baseline 64K prompt, concurrency 1",
        "drift_controls": "identical baseline c=1 4K probe at start, between configs, end; 15% refusal",
        "restart_cost": "Chunk size and OMP need restarts; reserve baseline restart before comparisons",
        "scope_description": "Consecutive turns from first >=4K through first >=64K; full source episodes retained",
        "cache_regime": "reset once per episode cell, retain all adjacent-turn prefixes; no turn barriers",
        "expectation": "Smaller prefill chunks reduce long decode gaps while maintaining prefix reuse.",
        "limitation": "The full sweep may not fit in one hour; only complete paired cells receive comparisons. "
        "Each config has a seven-minute default measurement cap, including both concurrency cells. "
        "Incomplete cells retain raw attempts and rerun with cache reconstruction.",
        "reference_run": str(args.reference_run),
    }
    write_json(args.output / "plan.json", plan)
    write_json(
        args.output / "roofline.json",
        {
            "conditional": True,
            "assumed_weight_bytes": 14e9,
            "estimates": [
                {
                    "context": length,
                    "batch": c,
                    "tokens_s": roofline_tokens_per_second(model_config, length, c),
                }
                for length in LENGTHS
                for c in CONCURRENCIES
            ],
        },
    )
    logger.info("Replay plan: %s", plan)
    manifest = {
        "baseline": baseline.server.command(),
        "omp_threads": baseline.omp_threads,
        "max_tokens": args.max_tokens,
        "sequence_scope": args.sequence_scope,
        "sequences": sequences,
        "model_config": model_config,
        "drift_threshold": args.drift_threshold,
        "config_seconds": args.config_seconds,
        "probe_payloads": cell_payloads(workloads, baseline, "trace", 4096, 1, 32),
    }
    manifest_path = args.output / "identity.json"
    if manifest_path.exists():
        validate_resume({"identity": json.loads(manifest_path.read_text())}, manifest)
    else:
        write_json(manifest_path, manifest)
    for variant in variants:
        for concurrency in CONCURRENCIES:
            saved = args.output / variant.name / f"sequence-{concurrency}" / "result.json"
            if saved.exists():
                expected = resume_identity(
                    variant,
                    sequence_payloads(
                        workloads, variant, concurrency, args.max_tokens, args.sequence_scope
                    ),
                    model_config,
                    args,
                )
                validate_resume(json.loads(saved.read_text()), expected)
    if args.dry_run:
        for variant in variants:
            variant.server.command()
            for concurrency in CONCURRENCIES:
                sequence_payloads(
                    workloads, variant, concurrency, args.max_tokens, args.sequence_scope
                )
        reference_results = args.reference_run / "results.jsonl"
        if reference_results.exists():
            reference_records = [
                json.loads(line) for line in reference_results.read_text().splitlines() if line
            ]
            write_json(
                args.output / "reference-audit.json",
                {
                    "cells": len(reference_records),
                    "comparable": False,
                    "reason": "Historical cold matrix has no interleaved baseline drift controls",
                },
            )
        if not (args.output / "summary.md").exists():
            render_report(args.output, [], complete=False)
        logger.info("CPU dry-run complete; no server or GPU sampling started")
        return
    capabilities = json.loads((args.output / "capabilities.json").read_text())
    if capabilities["missing_weight_files"]:
        raise ValueError(
            "Cached model weights are missing; set HF_HUB_CACHE to an intact local snapshot before launch"
        )
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    started = time.monotonic()
    deadline = started + args.budget_minutes * 60
    records = [
        json.loads(old_probe.read_text())
        for old_probe in sorted(
            (args.output / "sessions").glob("attempt-*/probes/attempt-*/result.json")
        )
    ]
    server: VLLMServer | None = None
    launch_key: tuple[list[str], int] | None = None
    session = measurement_directory(args.output / "sessions")
    session_id = session.name
    complete = True
    completed_normally = False

    def launch(variant: Configuration) -> VLLMServer:
        nonlocal server, launch_key
        key = (variant.server.command(), variant.omp_threads)
        if server is not None and key == launch_key:
            return server
        if server is not None:
            server.stop()
        os.environ["OMP_NUM_THREADS"] = str(variant.omp_threads)
        config = replace(
            variant.server,
            log_path=session / f"{variant.name}-server.log",
            profiler_directory=str((session / "profile").resolve()),
        )
        config = replace(
            config,
            startup_timeout_seconds=min(
                args.startup_reserve_seconds, max(1, deadline - time.monotonic())
            ),
        )
        server = VLLMServer(config)
        server.start()
        launch_key = key
        return server

    async def probe(*, final: bool = False) -> None:
        active = launch(baseline)
        await reset_cache(active.base_url)
        payload = cell_payloads(workloads, baseline, "trace", 4096, 1, 32)
        attempt = measurement_directory(session / "probes")
        async with asyncio.timeout(min(60, max(0.001, deadline - time.monotonic()))):
            result = await measure_cell(
                active.base_url,
                baseline.server.model,
                payload,
                attempt,
                server_pid=active.process.pid,
            )
        record = {
            **result,
            "configuration": "baseline",
            "kind": "baseline-probe",
            "concurrency": 1,
            "session": session_id,
            "final_probe": final,
            "drift_threshold": args.drift_threshold,
        }
        records.append(record)
        write_json(attempt / "result.json", record)
        render_report(args.output, records, complete=False)

    try:
        launch(baseline)
        # Compile the probe shape before measuring drift; this warmup is deliberately unscored.
        warm = cell_payloads(workloads, baseline, "trace", 4096, 1, 2)
        if server is None:
            raise RuntimeError("Baseline failed to start")
        async with asyncio.timeout(min(60, max(0.001, deadline - time.monotonic()))):
            await measure_cell(
                server.base_url,
                baseline.server.model,
                warm,
                measurement_directory(session / "warmup"),
                server_pid=server.process.pid,
            )
        await probe()
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            response = await client.get(f"{server.base_url}/server_info?config_format=json")
            response.raise_for_status()
            effective = response.json()["vllm_config"]
            write_json(session / "baseline-effective-config.json", effective)
        cold = cell_payloads(workloads, baseline, "trace", 65536, 1, args.max_tokens)
        profile_payload = [
            sequence[-1]
            for sequence in sequence_payloads(workloads, baseline, 4, 8, args.sequence_scope)
        ]
        for name, payload, profile in (
            ("cold-control", cold, False),
            ("profile", profile_payload, True),
        ):
            control_path = args.output / "baseline-controls" / name / "result.json"
            if control_path.exists():
                records.append(json.loads(control_path.read_text()))
                continue
            if not profile:
                await reset_cache(server.base_url)
            attempt = measurement_directory(session / name)
            async with asyncio.timeout(min(180, max(0.001, deadline - time.monotonic() - 60))):
                result = await measure_sequence_cell(
                    server.base_url,
                    baseline.server.model,
                    [[p] for p in payload],
                    attempt,
                    server_pid=server.process.pid,
                    profile=profile,
                    profiler_directory=session / "profile",
                )
            record = {
                **result,
                "configuration": "baseline",
                "kind": name,
                "concurrency": len(payload),
                "session": session_id,
            }
            write_json(control_path, record)
            records.append(record)
        for variant in variants:
            if (
                variant.name == "async"
                and effective["scheduler_config"]["async_scheduling"] is True
            ):
                write_json(
                    session / "async-skipped.json",
                    {"reason": "baseline already uses async scheduling"},
                )
                continue
            pending = []
            for concurrency in CONCURRENCIES:
                path = args.output / variant.name / f"sequence-{concurrency}" / "result.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    expected = resume_identity(
                        variant,
                        sequence_payloads(
                            workloads, variant, concurrency, args.max_tokens, args.sequence_scope
                        ),
                        model_config,
                        args,
                    )
                    validate_resume(record, expected)
                    records.append(record)
                else:
                    pending.append(concurrency)
            if not pending:
                continue
            # A changed engine needs both its own startup and a return to baseline.
            restart_reserve = args.startup_reserve_seconds * (
                2
                if variant.server != baseline.server or variant.omp_threads != baseline.omp_threads
                else 0
            )
            if time.monotonic() + restart_reserve + args.config_seconds + 60 > deadline:
                complete = False
                logger.warning("Budget reserves final baseline; deferring %s", variant.name)
                break
            active = launch(variant)
            config_deadline = min(
                time.monotonic() + args.config_seconds, deadline - args.startup_reserve_seconds - 60
            )
            for concurrency in pending:
                payload_sequences = sequence_payloads(
                    workloads, variant, concurrency, args.max_tokens, args.sequence_scope
                )
                identity = resume_identity(variant, payload_sequences, model_config, args)
                path = args.output / variant.name / f"sequence-{concurrency}" / "result.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    validate_resume(record, identity)
                    records.append(record)
                    continue
                await reset_cache(active.base_url)
                attempt = measurement_directory(path.parent)
                try:
                    async with asyncio.timeout(max(0.001, config_deadline - time.monotonic())):
                        result = await measure_sequence_cell(
                            active.base_url,
                            variant.server.model,
                            payload_sequences,
                            attempt,
                            server_pid=active.process.pid,
                        )
                except TimeoutError:
                    write_json(
                        attempt / "incomplete.json",
                        {"reason": "configuration measurement budget reached"},
                    )
                    complete = False
                    break
                record = {
                    **result,
                    "configuration": variant.name,
                    "kind": "sequence",
                    "concurrency": concurrency,
                    "session": session_id,
                    "identity": identity,
                }
                if not result.get("aggregate_decode_tokens_s"):
                    raise ValueError("No measurable decode interval; raw attempt retained")
                write_json(path, record)
                records.append(record)
                render_report(args.output, records, complete=False)
            await probe()
        await probe(final=True)
        completed_normally = True
    finally:
        if server is not None:
            server.stop()
        write_json(session / "records.json", records)
        render_report(args.output, records, complete=complete and completed_normally)


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
