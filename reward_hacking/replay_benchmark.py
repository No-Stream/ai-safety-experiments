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
import re
import time
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any, cast

import httpx
from huggingface_hub import snapshot_download

from reward_hacking.ladder.thinking import QWEN_THINKING_REASONING_CONFIG
from reward_hacking.organism_rl.probe_v4 import TWENTY_SEVEN_B_PREFILL_CHUNK_TOKENS
from reward_hacking.replay_measure import measure_cell, measure_sequence_cell
from reward_hacking.replay_workload import (
    extract_episode_workloads,
    extract_workloads,
    roofline_tokens_per_second,
    synthetic_workloads,
)
from reward_hacking.vllm_server import (
    VLLMServer,
    VLLMServerConfig,
    default_mtp_speculative_config,
)

logger = logging.getLogger(__name__)
LENGTHS = (4096, 16384, 32768, 65536)
NINE_B_CONTEXT_LENGTHS = (8192, 16384, 32768, 35000)
NINE_B_TRACE_LENGTHS = (4096, *NINE_B_CONTEXT_LENGTHS)
NINE_B_CONCURRENCIES = (4, 8, 16)
TWENTY_SEVEN_B_CONCURRENCIES = (4, 6)
DEFAULT_REFERENCE_RUN = Path("artifacts/perf/27b-replay-run1")
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
        if self.mamba_ssm_cache_dtype is not None:
            command.extend(["--mamba-ssm-cache-dtype", self.mamba_ssm_cache_dtype])
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


def _is_dense_qwen35_9b(model: str) -> bool:
    return "Qwen3.5-9B" in model


def context_lengths_for_model(model: str) -> tuple[int, ...]:
    """Return prompt-size targets for the selected model family."""
    return NINE_B_CONTEXT_LENGTHS if _is_dense_qwen35_9b(model) else LENGTHS


def sequence_context_lengths_for_model(model: str) -> tuple[int, ...]:
    """Return recorded sequence contexts supported by the selected model family."""
    return NINE_B_CONTEXT_LENGTHS if _is_dense_qwen35_9b(model) else (LENGTHS[-1],)


def concurrency_levels_for_model(model: str) -> tuple[int, ...]:
    """Return requested concurrency levels for the selected model family."""
    return NINE_B_CONCURRENCIES if _is_dense_qwen35_9b(model) else TWENTY_SEVEN_B_CONCURRENCIES


def baseline_server_config(
    model: str,
    revision: str | None,
    *,
    port: int,
    max_tokens: int,
    mtp_supported: bool,
) -> ReplayServerConfig:
    """Match the organism RL server for 9B and preserve the established 27B serving setup."""
    if _is_dense_qwen35_9b(model):
        return ReplayServerConfig(
            model=model,
            revision=revision,
            max_model_len=40960 + max_tokens,
            gpu_memory_utilization=0.9,
            language_model_only=True,
            max_num_seqs=max(NINE_B_CONCURRENCIES),
            dtype="bfloat16",
            enable_prefix_caching=True,
            kv_cache_dtype=None,
            mamba_cache_mode="align",
            max_num_batched_tokens=None,
            speculative_config=(default_mtp_speculative_config(model) if mtp_supported else None),
            reasoning_config=dict(QWEN_THINKING_REASONING_CONFIG),
            port=port,
        )
    return ReplayServerConfig(
        model=model,
        revision=revision,
        max_model_len=98304,
        gpu_memory_utilization=0.9,
        language_model_only=True,
        max_num_seqs=16,
        dtype="bfloat16",
        attention_backend="FLASHINFER",
        enable_prefix_caching=True,
        kv_cache_dtype="fp8",
        mamba_cache_mode="align",
        max_num_batched_tokens=TWENTY_SEVEN_B_PREFILL_CHUNK_TOKENS,
        reasoning_config=dict(QWEN_THINKING_REASONING_CONFIG),
        port=port,
    )


def configurations(
    baseline: ReplayServerConfig,
    quota_threads: int,
    *,
    mtp_supported: bool = False,
    async_supported: bool = False,
    mamba_ssm_cache_dtype_supported: bool = False,
) -> list[Configuration]:
    """Order likely scheduling/sampler wins ahead of memory/cache experiments."""
    if _is_dense_qwen35_9b(baseline.model):
        variants = [
            Configuration("baseline", baseline),
            *[
                Configuration(
                    f"prefill-{tokens // 1024}k",
                    replace(baseline, max_num_batched_tokens=tokens),
                    {"max_num_batched_tokens": tokens},
                )
                for tokens in (2048, 4096, 8192, 16384)
            ],
            Configuration(
                "kv-fp8", replace(baseline, kv_cache_dtype="fp8"), {"kv_cache_dtype": "fp8"}
            ),
        ]
        if baseline.speculative_config is not None:
            variants.append(
                Configuration(
                    "mtp-off",
                    replace(baseline, speculative_config=None),
                    {"speculative_config": None},
                )
            )
        elif mtp_supported:
            mtp_config = default_mtp_speculative_config(baseline.model)
            if mtp_config is None:
                raise ValueError(
                    "MTP capability was reported for a model without a default MTP config"
                )
            variants.append(
                Configuration(
                    "mtp-on",
                    replace(baseline, speculative_config=mtp_config),
                    {"speculative_config": mtp_config},
                )
            )
        if async_supported:
            variants.extend(
                [
                    Configuration(
                        "async-on",
                        replace(baseline, async_scheduling=True),
                        {"async_scheduling": True},
                    ),
                    Configuration(
                        "async-off",
                        replace(baseline, async_scheduling=False),
                        {"async_scheduling": False},
                    ),
                ]
            )
        if mamba_ssm_cache_dtype_supported:
            variants.append(
                Configuration(
                    "mamba-ssm-bf16",
                    replace(baseline, mamba_ssm_cache_dtype="bfloat16"),
                    {"mamba_ssm_cache_dtype": "bfloat16"},
                )
            )
        return variants

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
        "--mamba-ssm-cache-dtype",
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
    cache_source = source.parent.parent / "config/cache.py"
    cache_text = cache_source.read_text()
    supports_mamba_ssm_bfloat16 = (
        'MambaDType = Literal["auto", "float32", "float16", "bfloat16"]' in cache_text
    )
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
        "mamba_ssm_cache_dtype_bfloat16": str(supports_mamba_ssm_bfloat16).lower(),
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


def load_reference_workloads(
    reference_run: Path, selected_snapshot: str, *, explicitly_selected: bool
) -> list[dict[str, Any]] | None:
    """Load a matching reference workload, falling back only for an implicit legacy reference."""
    workload_path = reference_run / "workloads.json"
    if not workload_path.is_file():
        if explicitly_selected:
            raise FileNotFoundError(f"reference run has no workloads.json: {reference_run}")
        return None
    frozen = cast("dict[str, Any]", json.loads(workload_path.read_text()))
    if Path(frozen["model_snapshot"]).name != selected_snapshot:
        if explicitly_selected:
            raise ValueError("Reference model snapshot differs from selected model")
        return None
    return cast("list[dict[str, Any]]", frozen["workloads"])


def write_json(path: Path, value: object) -> None:
    """Replace complete JSON records atomically after successful measurement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


ENGINE_CONFIG_MARKER = "Initializing a V1 LLM engine"
KV_CACHE_SIZE_PATTERN = re.compile(r"KV cache size:\s*(?P<tokens>[\d,]+) tokens")


def effective_config_from_log(log_path: Path | None) -> dict[str, str]:
    """Return the engine's startup config line and the non-default CLI args line from a vLLM server log."""
    if log_path is None:
        raise ValueError("server has no log path; the effective config cannot be recorded")
    lines = log_path.read_text(errors="replace").splitlines()
    engine_lines = [line for line in lines if ENGINE_CONFIG_MARKER in line]
    argument_lines = [line for line in lines if "non-default args:" in line]
    if not engine_lines or not argument_lines:
        raise ValueError(f"{log_path} lacks the engine config or non-default args line")
    return {
        "engine_config": engine_lines[-1].split("with config: ", 1)[-1],
        "non_default_args": argument_lines[-1].split("non-default args: ", 1)[-1],
    }


def kv_pool_size_tokens_from_log(log_path: Path | None) -> int:
    """Read vLLM's resolved GPU KV token capacity from its startup log."""
    if log_path is None:
        raise ValueError("server has no log path; the KV pool size cannot be recorded")
    matches = list(KV_CACHE_SIZE_PATTERN.finditer(log_path.read_text(errors="replace")))
    if not matches:
        raise ValueError(f"{log_path} lacks the resolved GPU KV cache size")
    return int(matches[-1].group("tokens").replace(",", ""))


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
    result.add_argument("--budget-minutes", type=float)
    result.add_argument("--port", type=int, default=8124)
    result.add_argument("--configs", nargs="+", default=["all"])
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--reference-run", type=Path)
    result.add_argument(
        "--sequence-scope", choices=("growth-range", "whole-episode"), default="growth-range"
    )
    result.add_argument("--drift-threshold", type=float, default=0.15)
    result.add_argument("--config-seconds", type=float, default=420)
    result.add_argument("--startup-reserve-seconds", type=float, default=300)
    result.add_argument(
        "--no-profile",
        action="store_true",
        help="skip the profiler capture so a time-boxed pass spends its budget on the config comparison",
    )
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
        (r.get("context_length"), r["concurrency"]): r
        for r in records
        if r["configuration"] == "baseline" and r["kind"] == "sequence"
    }
    plan_path = output / "plan.json"
    if plan_path.is_file():
        plan = cast("dict[str, Any]", json.loads(plan_path.read_text()))
        expected_cells = {
            (context_length, concurrency)
            for context_length in plan["context_lengths"]
            for concurrency in plan["concurrencies"]
        }
    else:
        expected_cells = set(baseline)
    # Old runs lack actual baseline controls and are never promoted to valid comparisons.
    final_control = bool(probes and probes[-1].get("final_probe", False))
    if drift["valid"] and final_control:
        for name in sorted({r["configuration"] for r in records} - {"baseline"}):
            candidate_by_cell = {
                (r.get("context_length"), r["concurrency"]): r
                for r in records
                if r["configuration"] == name and r["kind"] == "sequence"
            }
            if (
                expected_cells
                and expected_cells <= baseline.keys()
                and expected_cells <= candidate_by_cell.keys()
            ):
                ratios = [
                    candidate_by_cell[cell]["output_tokens_per_wall_second"]
                    / baseline[cell]["output_tokens_per_wall_second"]
                    for cell in sorted(expected_cells)
                ]
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
                f"{speedups[best]:.3f}x over the full matched sequence matrix; "
                f"baseline drift bound {drift['drift_fraction']:.1%}."
            )
        else:
            lines.append("Recommendation withheld: no complete matched sequence matrix.")
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
        (
            "| Config | Context tokens | Concurrency | Aggregate decode tok/s | Per-stream decode tok/s | "
            "Prefill tok/s | Prefix hit rate | KV pool tokens | Preemptions |"
        ),
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        if record["kind"] != "sequence":
            continue
        stream_rates = [
            f"{stream['sequence_index']}:{optional_number(stream.get('decode_tokens_per_second'))}"
            for stream in record.get("per_stream_decode_tokens_s", [])
        ]
        stream_summary = ", ".join(stream_rates) if stream_rates else "unknown"
        hit_rate = record.get("prefix_cache_hit_rate")
        lines.append(
            f"| {record['configuration']} | {record.get('context_length', 'unknown')} | "
            f"{record['concurrency']} | {optional_number(record.get('aggregate_decode_tokens_s'))} | "
            f"{stream_summary} | {optional_number(record.get('prefill_tokens_per_second'))} | "
            f"{f'{100 * hit_rate:.1f}%' if hit_rate is not None else 'unknown'} | "
            f"{record.get('kv_pool_size_tokens', 'unknown')} | {record.get('preemptions', 'unknown')} |"
        )
    lines.extend(profiler_report_lines(records))
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(output / "drift.json", drift)
    (output / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def prepare(
    args: argparse.Namespace,
    installed: dict[str, str] | None = None,
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
    installed_flags = verify_installed_flags() if installed is None else installed
    capabilities = checkpoint_capabilities(snapshot)
    write_json(args.output / "capabilities.json", capabilities)
    baseline = baseline_server_config(
        args.model,
        None if Path(args.model).is_dir() else snapshot.name,
        port=args.port,
        max_tokens=args.max_tokens,
        mtp_supported=capabilities["mtp_supported_by_source_and_metadata"],
    )
    variants = configurations(
        baseline,
        cgroup_quota_threads(),
        mtp_supported=capabilities["mtp_supported_by_source_and_metadata"],
        async_supported=True,
        mamba_ssm_cache_dtype_supported=(
            installed_flags["mamba_ssm_cache_dtype_bfloat16"] == "true"
        ),
    )
    names = {variant.name for variant in variants}
    if args.configs != ["all"]:
        if not set(args.configs) <= names:
            raise ValueError(f"Unknown configs: {set(args.configs) - names}")
        variants = [variant for variant in variants if variant.name in args.configs]
    workload_path = args.output / "workloads-v4.json"
    if workload_path.exists():
        workload_record = cast("dict[str, Any]", json.loads(workload_path.read_text()))
        if Path(workload_record["model_snapshot"]).name != snapshot.name:
            raise ValueError("Model snapshot changed; use a new output directory")
        workloads = cast("list[dict[str, Any]]", workload_record["workloads"])
    else:
        reference_run = args.reference_run or DEFAULT_REFERENCE_RUN
        explicitly_selected = args.reference_run is not None
        reference_workloads = load_reference_workloads(
            reference_run, snapshot.name, explicitly_selected=explicitly_selected
        )
        trace_lengths = NINE_B_TRACE_LENGTHS if _is_dense_qwen35_9b(args.model) else LENGTHS
        if reference_workloads is None:
            workloads = build_workloads(
                args.traces, int(text_config["vocab_size"]), lengths=trace_lengths
            )
        else:
            workloads = [
                workload for workload in reference_workloads if workload["kind"] != "episode"
            ]
        episode_end_length = 35000 if _is_dense_qwen35_9b(args.model) else LENGTHS[-1]
        episode_max_context = 40960 if _is_dense_qwen35_9b(args.model) else None
        episodes = extract_episode_workloads(
            args.traces,
            copies=max(concurrency_levels_for_model(args.model)),
            end_length=episode_end_length,
            max_context_length=episode_max_context,
            require_prefix_growth=_is_dense_qwen35_9b(args.model),
        )
        workloads += [{"kind": "episode", **asdict(episode)} for episode in episodes]
        write_json(workload_path, {"model_snapshot": str(snapshot), "workloads": workloads})
    return workloads, model_config, variants


def build_workloads(
    traces: Path, vocab_size: int, *, lengths: tuple[int, ...] = LENGTHS
) -> list[dict[str, Any]]:
    """Extract prompts at runtime, never embedding benchmark material in source."""
    real = extract_workloads(traces, lengths=lengths)
    synthetic = synthetic_workloads(vocab_size, lengths=lengths, sampling=real[0].sampling)
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


def sequence_payloads(  # noqa: PLR0913, PLR0917 - explicit replay coordinates
    workloads: list[dict[str, Any]],
    variant: Configuration,
    concurrency: int,
    max_tokens: int,
    scope: str = "growth-range",
    context_length: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Freeze original adjacent turns; new sampled outputs never rewrite the next prompt."""
    episodes = [w for w in workloads if w["kind"] == "episode"][:concurrency]
    if len(episodes) != concurrency:
        raise ValueError("Insufficient independent episode sequences")
    sequences: list[list[dict[str, Any]]] = []
    for episode in episodes:
        turns = episode["turns"]
        start_index = episode["turn_4k_index"] if scope == "growth-range" else 0
        end_index = episode["turn_end_index"]
        if context_length is not None:
            end_index = next(
                (
                    index
                    for index in range(start_index, end_index + 1)
                    if turns[index]["prompt_tokens"] >= context_length
                ),
                -1,
            )
            if end_index < start_index:
                raise ValueError(
                    f"episode {episode['episode_id']} has no recorded prompt at context "
                    f"{context_length}"
                )
        if end_index <= start_index:
            raise ValueError("episode replay requires at least two recorded turns")
        if any(
            len(current["prompt_ids"]) <= len(previous["prompt_ids"])
            for previous, current in zip(
                turns[start_index : end_index + 1],
                turns[start_index + 1 : end_index + 1],
                strict=False,
            )
        ):
            raise ValueError("episode replay turns must grow in prompt length")
        sequences.append(
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
                for turn in turns[start_index : end_index + 1]
            ]
        )
    return sequences


def sequence_result_path(
    output: Path,
    configuration: str,
    concurrency: int,
    context_length: int,
) -> Path:
    """Return the persisted result path for one replay matrix cell."""
    return (
        output
        / configuration
        / f"sequence-{concurrency}"
        / f"context-{context_length}"
        / "result.json"
    )


def estimated_gpu_minutes(
    configuration_count: int, config_seconds: float, startup_reserve_seconds: float
) -> float:
    """Estimate measurement, server restart and baseline-control wall time."""
    server_starts = 1 + 2 * (configuration_count - 1)
    measurement_minutes = configuration_count * config_seconds / 60
    startup_minutes = server_starts * startup_reserve_seconds / 60
    baseline_controls_minutes = 5.0
    return math.ceil(measurement_minutes + startup_minutes + baseline_controls_minutes)


def resume_identity(
    variant: Configuration,
    sequences: list[list[dict[str, Any]]],
    model_config: dict[str, Any],
    args: argparse.Namespace,
    context_length: int,
) -> dict[str, Any]:
    """Bind measurement semantics as well as prompt IDs to resumable cells."""
    return {
        "schema": 3,
        "server_command": variant.server.command(),
        "omp_threads": variant.omp_threads,
        "sequences": sequences,
        "model_config": model_config,
        "config_seconds": args.config_seconds,
        "drift_threshold": args.drift_threshold,
        "sequence_scope": args.sequence_scope,
        "context_length": context_length,
    }


async def run(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915 - linear experiment lifecycle
    """Persist cells, reserve real baseline restarts, and never compare unbounded drift."""
    is_nine_b = _is_dense_qwen35_9b(args.model)
    budget_minutes = (
        (180 if is_nine_b else 55) if args.budget_minutes is None else args.budget_minutes
    )
    if args.max_tokens < MIN_DECODE_TOKENS or budget_minutes <= 0 or args.config_seconds <= 0:
        raise ValueError("Positive time budgets and max_tokens>=2 are required")
    if args.startup_reserve_seconds <= 0 or not 0 < args.drift_threshold < 1:
        raise ValueError("Positive startup reserve and drift threshold in (0,1) are required")
    verified = verify_installed_flags()
    workloads, model_config, variants = prepare(args, verified)
    # Always use the production baseline for controls even when selecting a config subset.
    if variants[0].name != "baseline":
        raise ValueError("Include baseline first in --configs to establish matched controls")
    baseline = variants[0]
    concurrencies = concurrency_levels_for_model(args.model)
    context_lengths = sequence_context_lengths_for_model(args.model)
    matrix_sequences = {
        (context_length, concurrency): sequence_payloads(
            workloads,
            baseline,
            concurrency,
            args.max_tokens,
            args.sequence_scope,
            context_length,
        )
        for context_length in context_lengths
        for concurrency in concurrencies
    }
    max_context_sequences = matrix_sequences[(context_lengths[-1], concurrencies[-1])]
    if any(
        len(payload["prompt"]) + args.max_tokens > baseline.server.max_model_len
        for sequence in max_context_sequences
        for payload in sequence
    ):
        raise ValueError("Frozen episode prompt exceeds replay model context")
    estimated_minutes = estimated_gpu_minutes(
        len(variants), args.config_seconds, args.startup_reserve_seconds
    )
    plan = {
        "schema": 3,
        "model": args.model,
        "revision": baseline.server.revision,
        "verified": verified,
        "configurations": [v.name for v in variants],
        "concurrencies": concurrencies,
        "context_lengths": context_lengths,
        "cells_per_config": len(concurrencies) * len(context_lengths),
        "matrix_cells": len(variants) * len(concurrencies) * len(context_lengths),
        "episode_lengths": [
            [len(payload["prompt"]) for payload in sequence] for sequence in max_context_sequences
        ],
        "sequence_scope": args.sequence_scope,
        "requests_per_config": sum(map(len, matrix_sequences.values())),
        "budget_minutes": budget_minutes,
        "config_seconds": args.config_seconds,
        "estimated_gpu_minutes": estimated_minutes,
        "max_tokens": args.max_tokens,
        "cold_control": f"one baseline {context_lengths[-1]}-token prompt, concurrency 1",
        "drift_controls": "identical baseline c=1 4K probe at start, between configs, end; 15% refusal",
        "restart_cost": (
            "Each changed engine starts once, then returns to baseline; "
            "reserve both starts before each config"
        ),
        "scope_description": (
            "Recorded consecutive prefix-growth turns from first >=4K through each requested context; "
            "full source episodes remain in the frozen workload"
        ),
        "cache_regime": "reset once per episode cell, retain all adjacent-turn prefixes; no turn barriers",
        "expectation": "Smaller prefill chunks should reduce long decode gaps while preserving recorded prefix reuse.",
        "limitation": "Each config shares its measurement cap across all concurrency and context cells. "
        "Incomplete cells retain raw attempts and resume from the remaining cells.",
        "reference_run": str(args.reference_run) if args.reference_run is not None else None,
    }
    write_json(args.output / "plan.json", plan)
    write_json(
        args.output / "roofline.json",
        {
            "conditional": True,
            "assumed_weight_bytes": 18e9 if is_nine_b else 14e9,
            "estimates": [
                {
                    "context": length,
                    "batch": c,
                    "tokens_s": roofline_tokens_per_second(
                        model_config,
                        length,
                        c,
                        kv_bytes=2 if is_nine_b else 1,
                        weight_bytes=18e9 if is_nine_b else 14e9,
                    ),
                }
                for length in context_lengths
                for c in concurrencies
            ],
        },
    )
    logger.info("Replay plan: %s", plan)
    manifest = {
        "baseline": baseline.server.command(),
        "omp_threads": baseline.omp_threads,
        "max_tokens": args.max_tokens,
        "sequence_scope": args.sequence_scope,
        "sequences": max_context_sequences,
        "model_config": model_config,
        "drift_threshold": args.drift_threshold,
        "config_seconds": args.config_seconds,
        "probe_payloads": cell_payloads(workloads, baseline, "trace", 4096, 1, 32),
    }
    manifest_path = args.output / "identity-v3.json"
    if manifest_path.exists():
        validate_resume({"identity": json.loads(manifest_path.read_text())}, manifest)
    else:
        write_json(manifest_path, manifest)
    for variant in variants:
        for context_length in context_lengths:
            for concurrency in concurrencies:
                saved = sequence_result_path(args.output, variant.name, concurrency, context_length)
                if saved.exists():
                    payload_sequences = sequence_payloads(
                        workloads,
                        variant,
                        concurrency,
                        args.max_tokens,
                        args.sequence_scope,
                        context_length,
                    )
                    expected = resume_identity(
                        variant, payload_sequences, model_config, args, context_length
                    )
                    validate_resume(json.loads(saved.read_text()), expected)
    if args.dry_run:
        for variant in variants:
            variant.server.command()
            for context_length in context_lengths:
                for concurrency in concurrencies:
                    sequence_payloads(
                        workloads,
                        variant,
                        concurrency,
                        args.max_tokens,
                        args.sequence_scope,
                        context_length,
                    )
        reference_run = args.reference_run or DEFAULT_REFERENCE_RUN
        reference_workloads = load_reference_workloads(
            reference_run,
            baseline.server.revision or Path(baseline.server.model).name,
            explicitly_selected=args.reference_run is not None,
        )
        reference_results = reference_run / "results.jsonl"
        if reference_workloads is not None and reference_results.exists():
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
    deadline = started + budget_minutes * 60
    records = [
        json.loads(old_probe.read_text())
        for old_probe in sorted(
            (args.output / "sessions").glob("attempt-*/probes/attempt-*/result.json")
        )
    ]
    server: VLLMServer | None = None
    launch_key: tuple[list[str], int] | None = None
    server_kv_pool_size_tokens: int | None = None
    session = measurement_directory(args.output / "sessions")
    session_id = session.name
    complete = True
    completed_normally = False

    def launch(variant: Configuration) -> VLLMServer:
        nonlocal server, launch_key, server_kv_pool_size_tokens
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
        server_kv_pool_size_tokens = kv_pool_size_tokens_from_log(server.config.log_path)
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
            "kv_pool_size_tokens": server_kv_pool_size_tokens,
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
        # vLLM 0.31's /server_info always collects system env info, which raises under WSL2; the engine
        # logs its effective config at startup, so record that line instead.
        write_json(
            session / "baseline-effective-config.json",
            effective_config_from_log(server.config.log_path),
        )
        cold = cell_payloads(workloads, baseline, "trace", context_lengths[-1], 1, args.max_tokens)
        profile_payload = [
            sequence[-1]
            for sequence in sequence_payloads(
                workloads,
                baseline,
                min(concurrencies),
                8,
                args.sequence_scope,
                context_lengths[-1],
            )
        ]
        controls = [("cold-control", cold, False)]
        if not args.no_profile:
            controls.append(("profile", profile_payload, True))
        for name, payload, profile in controls:
            control_path = args.output / "baseline-controls-v3" / name / "result.json"
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
                "context_length": context_lengths[-1],
                "kv_pool_size_tokens": server_kv_pool_size_tokens,
            }
            write_json(control_path, record)
            records.append(record)
        for variant in variants:
            pending: list[tuple[int, int]] = []
            for context_length in context_lengths:
                for concurrency in concurrencies:
                    path = sequence_result_path(
                        args.output, variant.name, concurrency, context_length
                    )
                    if path.exists():
                        record = json.loads(path.read_text())
                        expected = resume_identity(
                            variant,
                            sequence_payloads(
                                workloads,
                                variant,
                                concurrency,
                                args.max_tokens,
                                args.sequence_scope,
                                context_length,
                            ),
                            model_config,
                            args,
                            context_length,
                        )
                        validate_resume(record, expected)
                        records.append(record)
                    else:
                        pending.append((context_length, concurrency))
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
            for context_length, concurrency in pending:
                payload_sequences = sequence_payloads(
                    workloads,
                    variant,
                    concurrency,
                    args.max_tokens,
                    args.sequence_scope,
                    context_length,
                )
                identity = resume_identity(
                    variant, payload_sequences, model_config, args, context_length
                )
                path = sequence_result_path(args.output, variant.name, concurrency, context_length)
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
                    "context_length": context_length,
                    "session": session_id,
                    "identity": identity,
                    "kv_pool_size_tokens": server_kv_pool_size_tokens,
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
