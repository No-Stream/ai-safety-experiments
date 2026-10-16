"""Replay measurements use exact token ids and a synthetic local vLLM server."""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking import replay_measure

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@contextmanager
def synthetic_vllm_server(  # noqa: C901 - one fixture serves the vLLM HTTP surface under test
    *,
    omit_done: bool = False,
    request_count: int = 2,
    first_token_delays: dict[int, float] | None = None,
    profiler_trace_dir: Path | None = None,
) -> Generator[tuple[str, list[dict[str, Any]]]]:
    """Serve deterministic SSE chunks and cumulative Prometheus counters without a model."""
    prompt_metrics = 100
    computed_prefill_tokens = 80
    prefill_seconds = 2.0
    prefix_hits = 10
    prefix_queries = 20
    preemptions = 0
    received_payloads: list[dict[str, Any]] = []
    profile_endpoints: list[str] = []
    lock = threading.Lock()
    start_barrier = threading.Barrier(request_count)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/server_info?config_format=json":
                body = json.dumps(
                    {"vllm_config": {"scheduler_config": {"async_scheduling": True}}}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/metrics":
                self.send_error(404)
                return
            with lock:
                metrics = (
                    f"vllm:request_prefill_time_seconds_sum {prefill_seconds}\n"
                    f"vllm:prompt_tokens_total {prompt_metrics}\n"
                    f"vllm:prefix_cache_hits_total {prefix_hits}\n"
                    f"vllm:prefix_cache_queries_total {prefix_queries}\n"
                    f"vllm:num_preemptions_total {preemptions}\n"
                    f"vllm:request_prefill_kv_computed_tokens_sum {computed_prefill_tokens}\n"
                ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(metrics)))
            self.end_headers()
            self.wfile.write(metrics)

        def do_POST(self) -> None:
            nonlocal prompt_metrics, computed_prefill_tokens, prefill_seconds
            nonlocal prefix_hits, prefix_queries, preemptions
            if self.path == "/reset_prefix_cache":
                body = json.dumps({"success": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path in {"/start_profile", "/stop_profile"}:
                endpoint = self.path.removeprefix("/")
                with lock:
                    profile_endpoints.append(endpoint)
                if endpoint == "stop_profile" and profiler_trace_dir is not None:
                    profiler_trace_dir.mkdir(parents=True, exist_ok=True)
                    trace = {
                        "traceEvents": [
                            {
                                "cat": "user_annotation",
                                "name": "execute_context_1(3)_generation_1(2)",
                                "ph": "X",
                                "ts": 0,
                                "dur": 2000,
                            },
                            {
                                "cat": "user_annotation",
                                "name": "execute_context_0(0)_generation_1(1)",
                                "ph": "X",
                                "ts": 2000,
                                "dur": 3000,
                            },
                            {
                                "cat": "kernel",
                                "name": "flash_attn_fwd",
                                "ph": "X",
                                "ts": 0,
                                "dur": 2000,
                            },
                            {
                                "cat": "kernel",
                                "name": "cutlass_gemm",
                                "ph": "X",
                                "ts": 1000,
                                "dur": 3000,
                            },
                            *[
                                {
                                    "cat": "kernel",
                                    "name": f"cutlass_aux_{kernel_index}",
                                    "ph": "X",
                                    "ts": 0,
                                    "dur": 100,
                                }
                                for kernel_index in range(10)
                            ],
                            {"cat": "cpu_op", "name": "launch", "ph": "X", "ts": 0, "dur": 5000},
                        ]
                    }
                    trace_path = profiler_trace_dir / "synthetic.pt.trace.json.gz"
                    with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
                        json.dump(trace, stream)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path != "/v1/completions":
                self.send_error(404)
                return
            body_length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(body_length))
            with lock:
                received_payloads.append(payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            start_barrier.wait(timeout=3)
            prompt = payload["prompt"]
            time.sleep((first_token_delays or {}).get(prompt[0], 0.0))
            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "prompt_token_ids": prompt,
                            "token_ids": [31],
                            "text": "a",
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": [32, 33],
                            "text": "bc",
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": [34],
                            "text": "d",
                            "finish_reason": "stop",
                        }
                    ]
                },
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(prompt),
                        "completion_tokens": 4,
                        "total_tokens": len(prompt) + 4,
                    },
                },
            ]
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.02)
            with lock:
                prompt_metrics += len(prompt)
                computed_prefill_tokens += len(prompt) - 1
                prefill_seconds += 2.0
                prefix_hits += 1
                prefix_queries += len(prompt)
                preemptions += 1
            if not omit_done:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib override parameter
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received_payloads
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_measure_cell_streams_exact_tokens_and_reports_timing_and_server_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        replay_measure,
        "_read_nvml_sample",
        lambda: {
            "status": "available",
            "sm_utilization_pct": 73.0,
            "memory_controller_busy_pct": 41.0,
        },
    )
    prompts = [
        {"prompt_id": "synthetic-a", "prompt": [1, 2, 3], "max_tokens": 4, "temperature": 0.0},
        {"prompt_id": "synthetic-b", "prompt": [4, 5], "max_tokens": 4, "temperature": 0.4},
    ]
    with synthetic_vllm_server() as (base_url, received_payloads):
        summary = asyncio.run(
            replay_measure.measure_cell(base_url, "synthetic-model", prompts, tmp_path)
        )

    assert len(received_payloads) == 2
    assert {tuple(payload["prompt"]) for payload in received_payloads} == {(1, 2, 3), (4, 5)}
    assert all(payload["model"] == "synthetic-model" for payload in received_payloads)
    assert all(payload["stream"] is True for payload in received_payloads)
    assert all(payload["return_token_ids"] is True for payload in received_payloads)
    assert all(
        payload["stream_options"] == {"include_usage": True} for payload in received_payloads
    )
    assert all(payload["max_tokens"] == 4 for payload in received_payloads)
    assert summary["request_count"] == 2
    assert summary["output_tokens"] == 8
    assert summary["decode_wall_union_seconds"] > 0
    assert summary["decode_wall_union_seconds"] <= summary["wall_seconds"]
    assert summary["steady_decode_seconds"] >= 0
    if summary["steady_decode_seconds"] > 0:
        assert summary["steady_decode_tokens_s"] == pytest.approx(
            summary["steady_decode_tokens"] / summary["steady_decode_seconds"]
        )
    assert summary["server_metrics"]["computed_prompt_tokens"] == 3
    assert summary["server_metrics"]["cached_prompt_tokens"] == 2
    assert summary["server_metrics"]["prefill_seconds_per_request"] == 4.0
    assert summary["server_metrics"]["computed_prompt_tokens_per_request_second"] == 0.75
    assert summary["server_metrics"]["prefill_tokens_per_second"] == 0.75
    assert summary["server_metrics"]["prefix_cache_hit_rate"] == pytest.approx(0.4)
    assert summary["server_metrics"]["preemptions"] == 2

    records = [
        json.loads(line) for line in (tmp_path / "request_results.jsonl").read_text().splitlines()
    ]
    assert {record["prompt_id"] for record in records} == {"synthetic-a", "synthetic-b"}
    for record in records:
        assert record["completion_token_ids"] == [31, 32, 33, 34]
        assert record["output_tokens"] == 4
        assert record["usage"] == {
            "prompt_tokens": len(record["prompt_token_ids"]),
            "completion_tokens": 4,
        }
        assert record["decode_tokens_per_second"] == pytest.approx(
            3 / (record["last_token_elapsed_seconds"] - record["first_token_elapsed_seconds"])
        )
        assert any(event["coalesced_tokens"] for event in record["token_events"])
    telemetry_lines = (tmp_path / "nvml_samples.jsonl").read_text().splitlines()
    assert telemetry_lines
    assert not any("memory_bandwidth" in line for line in telemetry_lines)
    assert (tmp_path / "metrics_before.prom").is_file()
    assert (tmp_path / "metrics_after.prom").is_file()


def test_measure_cell_rejects_a_truncated_stream_without_done_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay_measure, "_read_nvml_sample", lambda: {"status": "unsupported_wsl"})
    with (
        synthetic_vllm_server(omit_done=True, request_count=1) as (base_url, _),
        pytest.raises(ValueError, match="DONE"),
    ):
        asyncio.run(
            replay_measure.measure_cell(
                base_url,
                "synthetic-model",
                [{"prompt_id": "truncated", "prompt": [7, 8], "max_tokens": 4}],
                tmp_path,
            )
        )


def test_measure_sequence_cell_keeps_recorded_prompts_and_runs_independent_chains_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay_measure, "_read_nvml_sample", lambda: {"status": "unsupported_wsl"})
    sequences = [
        [
            {"prompt_id": "chain-a-turn-0", "prompt": [1, 2], "max_tokens": 4},
            {"prompt_id": "chain-a-turn-1", "prompt": [1, 2, 90, 91], "max_tokens": 4},
        ],
        [
            {"prompt_id": "chain-b-turn-0", "prompt": [3, 4], "max_tokens": 4},
            {"prompt_id": "chain-b-turn-1", "prompt": [3, 4, 92], "max_tokens": 4},
        ],
    ]
    with synthetic_vllm_server(request_count=2) as (base_url, received_payloads):
        summary = asyncio.run(
            replay_measure.measure_sequence_cell(base_url, "synthetic-model", sequences, tmp_path)
        )

    assert len(received_payloads) == 4
    assert {tuple(payload["prompt"]) for payload in received_payloads} == {
        (1, 2),
        (1, 2, 90, 91),
        (3, 4),
        (3, 4, 92),
    }
    assert summary["sequence_count"] == 2
    assert summary["turn_count"] == 4
    assert len(summary["per_stream_decode_tokens_s"]) == 2
    assert all(
        stream["decode_tokens_per_second"] > 0 for stream in summary["per_stream_decode_tokens_s"]
    )
    records = [
        json.loads(line) for line in (tmp_path / "request_results.jsonl").read_text().splitlines()
    ]
    by_prompt_id = {record["prompt_id"]: record for record in records}
    assert by_prompt_id["chain-a-turn-1"]["prompt_token_ids"] == [1, 2, 90, 91]
    assert by_prompt_id["chain-b-turn-1"]["prompt_token_ids"] == [3, 4, 92]
    assert by_prompt_id["chain-a-turn-1"]["sequence_index"] == 0
    assert by_prompt_id["chain-a-turn-1"]["turn_index"] == 1
    assert by_prompt_id["chain-b-turn-1"]["sequence_index"] == 1
    assert by_prompt_id["chain-b-turn-1"]["turn_index"] == 1


def test_measure_cell_records_ttft_overlap_as_observational_correlation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay_measure, "_read_nvml_sample", lambda: {"status": "unsupported_wsl"})
    prompts = [
        {"prompt_id": "early", "prompt": [1, 2], "max_tokens": 4},
        {"prompt_id": "late", "prompt": [2, 3], "max_tokens": 4},
    ]
    with synthetic_vllm_server(request_count=2, first_token_delays={1: 0.02, 2: 0.10}) as (
        base_url,
        _,
    ):
        summary = asyncio.run(
            replay_measure.measure_cell(base_url, "synthetic-model", prompts, tmp_path)
        )

    records = [
        json.loads(line) for line in (tmp_path / "request_results.jsonl").read_text().splitlines()
    ]
    early = next(record for record in records if record["prompt_id"] == "early")
    assert any(
        event["overlap_seconds_with_other_ttft"] > 0
        and event["other_requests_awaiting_first_token"] == 1
        for event in early["token_events"]
    )
    assert (
        summary["ttft_overlap"]["interpretation"]
        == "observational correlation, not causal attribution"
    )
    assert summary["ttft_overlap"]["token_events_overlapping_other_ttft"] > 0


def test_measure_cell_persists_host_cgroup_and_api_descendant_cpu_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay_measure, "_read_nvml_sample", lambda: {"status": "unsupported_wsl"})
    with synthetic_vllm_server(request_count=1) as (base_url, _):
        asyncio.run(
            replay_measure.measure_cell(
                base_url,
                "synthetic-model",
                [{"prompt_id": "cpu", "prompt": [5, 6], "max_tokens": 4}],
                tmp_path,
                server_pid=os.getpid(),
            )
        )

    sample_lines = (tmp_path / "cpu_samples.jsonl").read_text().splitlines()
    assert len(sample_lines) >= 2
    samples = [json.loads(line) for line in sample_lines]
    assert all("host_load" in sample and "cgroup_pressure" in sample for sample in samples)
    root_processes = [
        process
        for sample in samples
        for process in sample["processes"]
        if process["pid"] == os.getpid()
    ]
    assert root_processes
    assert all(process["role"] == "api" for process in root_processes)
    cpu_summary = json.loads((tmp_path / "cpu_telemetry.json").read_text())
    assert cpu_summary["server_pid"] == os.getpid()
    assert cpu_summary["process_cpu_seconds_by_role"]["api"] >= 0


def test_profile_cell_source_checks_vllm_routes_and_summarizes_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay_measure, "_read_nvml_sample", lambda: {"status": "unsupported_wsl"})
    source_verification = {
        "vllm_version": "0.31.0",
        "endpoint_source": "synthetic-api-router.py",
        "router_source": "synthetic-serve-init.py",
    }
    monkeypatch.setattr(replay_measure, "_verify_profiler_source", lambda: source_verification)
    profile_trace_dir = tmp_path / "profiler"
    with synthetic_vllm_server(request_count=1, profiler_trace_dir=profile_trace_dir) as (
        base_url,
        _,
    ):
        summary = asyncio.run(
            replay_measure.measure_sequence_cell(
                base_url,
                "synthetic-model",
                [[{"prompt_id": "profile", "prompt": [7, 8], "max_tokens": 4}]],
                tmp_path,
                profile=True,
                profiler_directory=profile_trace_dir,
            )
        )

    profile_record = json.loads((tmp_path / "profile_capture.json").read_text())
    assert profile_record["source_verification"] == source_verification
    assert profile_record["endpoints"] == ["start_profile", "stop_profile"]
    assert summary["profiler_summary"]["kernel_busy_interval_union_seconds"] == pytest.approx(0.004)
    assert summary["profiler_summary"]["gpu_kernel_count"] == 12
    assert summary["profiler_summary"]["total_gpu_kernel_duration_seconds"] == pytest.approx(0.006)
    assert summary["profiler_summary"]["gpu_kernel_duration_by_category"] == {
        "attention": {
            "total_seconds": pytest.approx(0.002),
            "share_of_total_gpu_kernel_duration": pytest.approx(1 / 3),
        },
        "matrix_multiply": {
            "total_seconds": pytest.approx(0.004),
            "share_of_total_gpu_kernel_duration": pytest.approx(2 / 3),
        },
    }
    top_matrix_kernels = summary["profiler_summary"]["top_gpu_kernels_by_category"][
        "matrix_multiply"
    ]
    assert len(top_matrix_kernels) == 10
    assert sum(kernel["total_milliseconds"] for kernel in top_matrix_kernels) == pytest.approx(3.9)
    assert summary["profiler_summary"]["profiler_event_window_seconds"] == pytest.approx(0.005)
    assert summary["profiler_summary"]["possible_non_kernel_fraction_upper_bound"] == pytest.approx(
        0.2
    )
    assert summary["profiler_summary"]["host_idle_interpretation"] == (
        "upper bound only; includes GPU waits, launch gaps, and other non-kernel time"
    )
    assert summary["profiler_summary"]["top_gpu_kernels_by_category"]["attention"][0]["name"] == (
        "flash_attn_fwd"
    )
    step_summary = summary["profiler_summary"]["scheduler_step_summary"]
    assert step_summary["step_count"] == 2
    assert step_summary["generation_bearing_step_count"] == 2
    assert step_summary["generation_tokens"] == 3
    assert step_summary["generation_bearing_host_scope_seconds"] == pytest.approx(0.005)
    assert step_summary["mixed_context_and_generation_step_count"] == 1
    assert step_summary["mixed_context_and_generation_mean_host_scope_seconds"] == pytest.approx(
        0.002
    )
    assert step_summary["pure_generation_step_count"] == 1
    assert step_summary["pure_generation_mean_host_scope_seconds"] == pytest.approx(0.003)
    assert step_summary["stall_attribution"] == "not exposed by the profiler annotation"


def test_profiler_kernel_categories_cover_vllm_specialized_kernel_names() -> None:
    category_by_kernel = {
        "Marlin_W4A16": "matrix_multiply",
        "FlashInfer_attention": "attention",
        "gated_delta_rule_FLA_linear_attention": "linear_attention",
        "sampler_logits_processors_thinking_budget_bitmask_topk": "sampling_or_logits_processing",
        "layer_norm": "normalization_or_pointwise",
        "unclassified_kernel": "other",
    }

    assert {
        kernel_name: replay_measure._profiler_kernel_category(kernel_name)
        for kernel_name in category_by_kernel
    } == category_by_kernel


def test_verify_profiler_source_checks_installed_vllm_annotation_when_available() -> None:
    try:
        version("vllm")
    except PackageNotFoundError:
        pytest.skip("vLLM is an optional profiling dependency")

    verification = replay_measure._verify_profiler_source()

    assert verification["vllm_version"] == "0.31.0"
    assert verification["scheduler_annotation_source"] == "vllm/v1/worker/gpu_worker.py"
    assert verification["scheduler_annotation"] == "context/generation counts and token counts"


def test_nvml_sampler_uses_mocked_cpu_only_device_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(replay_measure.pynvml, "nvmlInit", lambda: calls.append("init"))
    monkeypatch.setattr(replay_measure.pynvml, "nvmlShutdown", lambda: calls.append("shutdown"))
    monkeypatch.setattr(replay_measure.pynvml, "nvmlDeviceGetHandleByIndex", lambda index: "gpu-0")
    monkeypatch.setattr(replay_measure.pynvml, "nvmlDeviceGetName", lambda handle: b"synthetic-gpu")
    monkeypatch.setattr(
        replay_measure.pynvml,
        "nvmlDeviceGetUtilizationRates",
        lambda handle: SimpleNamespace(gpu=72, memory=41),
    )
    monkeypatch.setattr(
        replay_measure.pynvml,
        "nvmlDeviceGetMemoryInfo",
        lambda handle: SimpleNamespace(used=1024**3, total=8 * 1024**3),
    )
    monkeypatch.setattr(
        replay_measure.pynvml, "nvmlDeviceGetTemperature", lambda handle, sensor: 68
    )
    monkeypatch.setattr(
        replay_measure.pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", lambda handle: 0
    )
    monkeypatch.setattr(replay_measure.pynvml, "nvmlDeviceGetPowerUsage", lambda handle: 250_000)
    monkeypatch.setattr(
        replay_measure.pynvml, "nvmlDeviceGetClockInfo", lambda handle, clock: 1500 + clock
    )

    sample = replay_measure._read_nvml_sample()

    assert calls == ["init", "shutdown"]
    assert sample["status"] == "available"
    assert sample["device_name"] == "synthetic-gpu"
    assert sample["temperature_c"] == 68
    assert sample["clock_throttling_active"] is False
    assert sample["power_watts"] == 250
