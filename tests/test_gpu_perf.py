"""CPU-only controls for throughput arithmetic and telemetry."""

from __future__ import annotations

import gzip
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pynvml
import pytest

from grpo.gpu_perf import (
    LATENCY_HISTOGRAMS,
    GpuSample,
    PerfMonitor,
    PerfThresholds,
    decode_roofline,
    device_peak,
    histogram_summary,
    parse_metrics,
    read_gpu_sample,
    throughput_breaches,
)
from grpo.model_geometry import ModelGeometry
from grpo.perf_samples import read_samples

METRICS = (Path(__file__).parent / "fixtures/performance/vllm_metrics.prom").read_text()
MODEL = ModelGeometry(
    26_895_998_464, 17_681_660_928, 32768, 78_446_592, "served checkpoint metadata"
)


def gpu(sm: float = 90, pids: tuple[int, ...] = (123,)) -> GpuSample:
    return GpuSample(
        "NVIDIA GeForce RTX 5090",
        sm,
        75,
        450,
        2100,
        14000,
        28000,
        32768,
        pids,
        tuple((pid, 256.0) for pid in pids),
    )


def test_metric_vocabulary_and_roofline() -> None:
    assert parse_metrics(METRICS)["vllm:generation_tokens_total"] == 900
    peak = device_peak("NVIDIA GeForce RTX 5090")
    assert peak is not None
    assert peak.tensor_flops == 209.5e12
    assert decode_roofline(MODEL, peak, 2, 1000) == pytest.approx(
        1792e9 * 2 / (17_681_660_928 + 2 * (1000 * 32768 + 2 * 78_446_592))
    )
    assert device_peak("unrecognized card") is None


def test_histogram_parsing_accepts_infinite_bucket_and_reports_percentiles() -> None:
    required_metrics = "\n".join(
        line for line in METRICS.splitlines() if "inter_token_latency_seconds" not in line
    )
    metrics = parse_metrics(
        required_metrics
        + "\n"
        + """vllm:inter_token_latency_seconds_bucket{engine="0",le="0.1"} 2
vllm:inter_token_latency_seconds_bucket{engine="0",le="0.2"} 4
vllm:inter_token_latency_seconds_bucket{engine="0",le="+Inf"} 5
vllm:inter_token_latency_seconds_sum{engine="0"} 0.8
vllm:inter_token_latency_seconds_count{engine="0"} 5
"""
    )

    summary = histogram_summary(metrics, "vllm:inter_token_latency_seconds")

    assert summary["count"] == 5
    assert summary["mean_seconds"] == pytest.approx(0.16)
    assert summary["p50_seconds"] == pytest.approx(0.125)
    assert summary["p90_seconds"] == pytest.approx(0.2)
    assert len([name for name in parse_metrics(METRICS) if "_bucket" in name]) == 2


def test_unavailable_histograms_and_counters_stay_unknown(tmp_path: Path) -> None:
    minimal_metrics = {
        "vllm:generation_tokens_total": 0.0,
        "vllm:prompt_tokens_total": 0.0,
        "vllm:num_requests_running": 1.0,
        "vllm:num_requests_waiting": 0.0,
    }
    monitor = PerfMonitor(tmp_path, "unused", 1, model=MODEL, device_name="unknown")
    monitor.record_sample(0, minimal_metrics, None)
    monitor.record_sample(10, minimal_metrics, None)
    report = monitor.finish(10)

    assert report["vllm_counter_deltas"] == {}
    assert report["vllm_histograms"] == {}
    assert report["prefill_request_seconds"] is None
    summary = (tmp_path / "summary.md").read_text()
    assert "Prefix cache hits / queries | unknown / unknown" in summary
    assert "Mean TTFT / P50 | unknown / unknown s" in summary


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -1])
def test_invalid_thresholds_fail_fast(threshold: float) -> None:
    with pytest.raises(
        ValueError, match=r"Thresholds must be finite|Estimate multiplier must be positive"
    ):
        PerfThresholds(estimate_multiplier=threshold)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_monitor_intervals_and_estimates(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        PerfMonitor(tmp_path, "unused", 6, interval_seconds=value)
    with pytest.raises(ValueError, match="must be positive"):
        PerfMonitor(tmp_path, "unused", 6, estimate_seconds=value)


@pytest.mark.parametrize("acceptance", [True, 42, "", " "])
def test_invalid_persisted_acceptance_cannot_bypass_gate(
    tmp_path: Path, acceptance: object
) -> None:
    (tmp_path / "throughput.json").write_text(
        json.dumps({"accept_slow": acceptance, "breaches": ["slow"]})
    )
    with pytest.raises(ValueError, match="nonempty reason string"):
        throughput_breaches(tmp_path)


def test_accept_slow_does_not_bypass_telemetry_failure(tmp_path: Path) -> None:
    (tmp_path / "throughput.json").write_text(
        json.dumps(
            {
                "accept_slow": "known slow run",
                "breaches": ["decode_bandwidth_roofline_low", "throughput_telemetry_unavailable"],
            }
        )
    )

    assert throughput_breaches(tmp_path) == ["throughput_telemetry_unavailable"]


def test_sample_times_must_increase(tmp_path: Path) -> None:
    monitor = PerfMonitor(tmp_path, "unused", 6)
    monitor.record_sample(0, parse_metrics(METRICS), gpu())
    with pytest.raises(ValueError, match="increasing"):
        monitor.record_sample(0, parse_metrics(METRICS), gpu())
    with pytest.raises(ValueError, match="finite"):
        monitor.record_sample(float("nan"), parse_metrics(METRICS), gpu())


def test_mixed_prefill_intervals_do_not_reduce_decode_fraction(tmp_path: Path) -> None:
    monitor = PerfMonitor(tmp_path, "unused", 6, model=MODEL, allowed_gpu_pids=(123,))
    monitor.set_mean_context(1000)
    before = parse_metrics(METRICS)
    mixed = dict(
        before,
        **{
            "vllm:generation_tokens_total": 950,
            "vllm:prompt_tokens_total": 10000,
            "vllm:request_prefill_time_seconds_sum": 11,
            "vllm:request_prefill_time_seconds_count": 4,
        },
    )
    after = dict(mixed, **{"vllm:generation_tokens_total": 2450})
    monitor.record_sample(0, before, gpu())
    monitor.record_sample(10, mixed, gpu())
    monitor.record_sample(20, after, gpu())
    report = monitor.finish(20)
    assert report["decode_only_seconds"] == 10
    assert report["decode_only_tokens"] == 1500
    assert throughput_breaches(tmp_path) == []
    assert report["pending_hardware_means"]["power_watts"] == 450
    assert report["decode_limiting_roof"] == "memory_bandwidth"
    assert report["prefill_flop_peak_fraction"] == pytest.approx(
        2 * MODEL.parameter_count * 8000 / (10 * 209.5e12)
    )
    assert report["prefill_flop_peak_fraction_stage_average_lower_bound"] == pytest.approx(
        2 * MODEL.parameter_count * 8000 / (20 * 209.5e12)
    )


def test_counter_reset_and_missing_metrics_are_named_failures(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing required"):
        parse_metrics("vllm:prompt_tokens_total 1")
    monitor = PerfMonitor(tmp_path, "unused", 6, model=MODEL, allowed_gpu_pids=(123,))
    monitor.record_sample(0, parse_metrics(METRICS), gpu())
    monitor.record_sample(10, parse_metrics(METRICS.replace(" 900\n", " 0\n")), gpu())
    monitor.finish(10)
    assert "vllm_counter_reset" in throughput_breaches(tmp_path)


def test_slow_recorded_rate_is_red_and_fast_control_green(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # stuck-timed 23:28:05: decode-only generation 62.2 tokens/s, Running 2.
    caplog.set_level(logging.WARNING)
    for name, rate, expected_red in (("slow", 62.2, True), ("fast", 150.0, False)):
        monitor = PerfMonitor(tmp_path / name, "unused", 6, model=MODEL, allowed_gpu_pids=(123,))
        monitor.set_mean_context(1000)
        monitor.record_sample(0, parse_metrics(METRICS), gpu())
        monitor.record_sample(
            10, parse_metrics(METRICS.replace(" 900\n", f" {900 + rate * 10}\n")), gpu()
        )
        report = monitor.finish(elapsed_seconds=10)
        assert bool(throughput_breaches(tmp_path / name)) == expected_red
        assert report["decode_tokens_per_second"] == pytest.approx(rate)
    assert "THROUGHPUT ALERT" in caplog.text


def test_sampler_http_writes_report_and_override(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(METRICS.encode())

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with PerfMonitor(
            tmp_path,
            f"http://127.0.0.1:{server.server_port}/metrics",
            6,
            model=MODEL,
            gpu_reader=lambda: gpu(10),
            allowed_gpu_pids=(123,),
            interval_seconds=0.01,
            accept_slow="CPU integration control",
        ) as monitor:
            monitor.set_mean_context(1000)
            assert monitor.wait_for_samples(2, timeout=2)
        report = json.loads((tmp_path / "throughput.json").read_text())
        assert "gpu_sm_utilization_low" in report["breaches"]
        assert report["accept_slow"] == "CPU integration control"
        assert throughput_breaches(tmp_path) == []
        assert "Throughput" in (tmp_path / "summary.md").read_text()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_sampler_thread_failure_is_named_even_after_valid_samples(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(METRICS.encode())

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    calls = 0
    failed = threading.Event()

    def failing_gpu_reader() -> GpuSample:
        nonlocal calls
        calls += 1
        if calls == 2:
            failed.set()
            raise TypeError("unexpected NVML adapter error")
        return gpu()

    try:
        with PerfMonitor(
            tmp_path,
            f"http://127.0.0.1:{server.server_port}/metrics",
            6,
            interval_seconds=0.01,
            gpu_reader=failing_gpu_reader,
            accept_slow="expected low synthetic GPU utilization",
        ):
            assert failed.wait(timeout=2)
        assert throughput_breaches(tmp_path) == [
            "throughput_sampler_thread_failed",
            "throughput_telemetry_unavailable",
        ]
    finally:
        server.shutdown()
        server_thread.join()
        server.server_close()


def test_missing_and_foreign_telemetry_fail_closed(tmp_path: Path) -> None:
    monitor = PerfMonitor(tmp_path, "unused", 6, model=MODEL, allowed_gpu_pids=(123,))
    monitor.record_sample(0, parse_metrics(METRICS), gpu(pids=(123, 456)))
    monitor.record_sample(10, parse_metrics(METRICS), gpu())
    monitor.finish(elapsed_seconds=10)
    assert "foreign_gpu_process" in throughput_breaches(tmp_path)
    assert "throughput_missing" in throughput_breaches(tmp_path / "missing")


def test_vram_without_visible_processes_has_unknown_ownership(tmp_path: Path) -> None:
    monitor = PerfMonitor(tmp_path, "unused", 6, model=MODEL, allowed_gpu_pids=(123,))
    monitor.record_sample(0, parse_metrics(METRICS), gpu(pids=()))
    monitor.record_sample(10, parse_metrics(METRICS), gpu(pids=()))
    monitor.finish(10)
    assert "gpu_process_attribution_unknown" in throughput_breaches(tmp_path)


def test_wsl_unknown_process_ownership_is_platform_unknown_after_stage_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("grpo.gpu_perf._running_under_wsl", lambda: True)
    monitor = PerfMonitor(
        tmp_path,
        "unused",
        6,
        model=MODEL,
        allowed_gpu_pids=(123,),
        gpu_preflight_passed_at_stage_start=True,
    )
    monitor.record_sample(0, parse_metrics(METRICS), gpu(pids=()))
    monitor.record_sample(10, parse_metrics(METRICS), gpu(pids=()))

    report = monitor.finish(10)
    sample = next(read_samples(tmp_path / "throughput-samples.jsonl.gz"))

    assert "gpu_process_attribution_unknown" not in report["breaches"]
    assert report["platform_unknown_gpu_fields"] == ["gpu_process_ownership_unknown"]
    assert sample["platform_unknown_fields"] == ["gpu_process_ownership_unknown"]


@pytest.mark.parametrize(
    ("wsl", "preflight_passed"),
    [(True, False), (False, True)],
)
def test_wsl_attribution_unknown_requires_wsl_and_stage_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wsl: bool,
    preflight_passed: bool,
) -> None:
    monkeypatch.setattr("grpo.gpu_perf._running_under_wsl", lambda: wsl)
    monitor = PerfMonitor(
        tmp_path,
        "unused",
        6,
        allowed_gpu_pids=(123,),
        gpu_preflight_passed_at_stage_start=preflight_passed,
    )
    monitor.record_sample(0, parse_metrics(METRICS), gpu(pids=()))
    monitor.record_sample(10, parse_metrics(METRICS), gpu(pids=()))
    report = monitor.finish(10)

    assert "gpu_process_attribution_unknown" in report["breaches"]
    assert report["platform_unknown_gpu_fields"] == []


def test_missing_gpu_sample_is_not_platform_unknown_after_stage_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("grpo.gpu_perf._running_under_wsl", lambda: True)
    monitor = PerfMonitor(
        tmp_path,
        "unused",
        6,
        gpu_preflight_passed_at_stage_start=True,
    )
    monitor.record_sample(0, parse_metrics(METRICS), None)
    monitor.record_sample(10, parse_metrics(METRICS), None)
    report = monitor.finish(10)

    assert "gpu_process_attribution_unknown" in report["breaches"]
    assert report["platform_unknown_gpu_fields"] == []


def test_wsl_preflight_does_not_hide_known_foreign_gpu_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("grpo.gpu_perf._running_under_wsl", lambda: True)
    monitor = PerfMonitor(
        tmp_path,
        "unused",
        6,
        model=MODEL,
        allowed_gpu_pids=(123,),
        gpu_preflight_passed_at_stage_start=True,
    )
    monitor.record_sample(0, parse_metrics(METRICS), gpu(pids=(123, 456)))
    monitor.record_sample(10, parse_metrics(METRICS), gpu(pids=(123, 456)))
    report = monitor.finish(10)

    assert "foreign_gpu_process" in report["breaches"]
    assert "gpu_process_attribution_unknown" not in report["breaches"]


def test_wsl_preflight_marks_unknown_foreign_memory_as_platform_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("grpo.gpu_perf._running_under_wsl", lambda: True)
    monitor = PerfMonitor(
        tmp_path,
        "unused",
        6,
        allowed_gpu_pids=(123,),
        gpu_preflight_passed_at_stage_start=True,
    )
    unknown_memory_gpu = GpuSample(
        "NVIDIA GeForce RTX 5090",
        90,
        75,
        450,
        2100,
        14000,
        28000,
        32768,
        (123, 456),
        ((123, 256.0), (456, None)),
    )
    monitor.record_sample(0, parse_metrics(METRICS), unknown_memory_gpu)
    monitor.record_sample(10, parse_metrics(METRICS), unknown_memory_gpu)
    report = monitor.finish(10)

    assert "foreign_gpu_memory_unknown" not in report["breaches"]
    assert report["platform_unknown_gpu_fields"] == ["foreign_gpu_memory_unknown_pids"]


def test_read_gpu_sample_treats_unsupported_nvml_attribution_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByIndex", lambda index: "gpu")
    monkeypatch.setattr(pynvml, "nvmlDeviceGetName", lambda handle: b"NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetUtilizationRates",
        lambda handle: SimpleNamespace(gpu=90, memory=75),
    )
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetMemoryInfo",
        lambda handle: SimpleNamespace(used=28_000 * 1024**2, total=32_768 * 1024**2),
    )

    def attribution_unavailable(handle: object) -> list[object]:
        raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)

    monkeypatch.setattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", attribution_unavailable)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses_v3", lambda handle: [])
    monkeypatch.setattr(pynvml, "nvmlDeviceGetPowerUsage", lambda handle: 450_000)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetClockInfo", lambda handle, clock: 2100)

    sample = read_gpu_sample()

    assert sample.memory_used_mib == 28000
    assert sample.process_ids is None


def test_missing_telemetry_still_writes_report_and_summary(tmp_path: Path) -> None:
    monitor = PerfMonitor(tmp_path, "unused", 6, model=MODEL)
    report = monitor.finish(10)
    assert "throughput_telemetry_unavailable" in throughput_breaches(tmp_path)
    assert "samples" not in report
    assert report["sample_count"] == 0
    assert "Mean KV cache use | unknown" in (tmp_path / "summary.md").read_text()


def test_sampler_keeps_only_metrics_used_by_reports() -> None:
    metrics = parse_metrics(METRICS + '\nvllm:irrelevant_histogram_bucket{le="1"} 500\n')
    assert not any("irrelevant" in name for name in metrics)
    assert any(name.startswith("vllm:time_to_first_token_seconds_bucket{") for name in metrics)


def test_hour_of_samples_is_compact_and_incrementally_readable(tmp_path: Path) -> None:
    monitor = PerfMonitor(tmp_path, "unused", 6, model=MODEL)
    # Four full histograms and unrelated families approximate a real vLLM export.
    histogram_lines = [
        f'vllm:{name}_bucket{{engine="0",model_name="synthetic-served-checkpoint",le="{bound}"}} {count}'
        for name in LATENCY_HISTOGRAMS
        for count, bound in enumerate([*(str(index / 100) for index in range(1, 51)), "+Inf"])
    ]
    unrelated_lines = [
        f'vllm:unused_{index}_bucket{{model_name="synthetic-served-checkpoint",le="1"}} 900'
        for index in range(1000)
    ]
    export = "\n".join([METRICS, *histogram_lines, *unrelated_lines])
    for index in range(721):
        metrics = parse_metrics(export)
        metrics["vllm:generation_tokens_total"] += index * 750
        monitor.record_sample(index * 5, metrics, gpu())
    path = tmp_path / "throughput-samples.jsonl.gz"
    assert len(list(read_samples(path))) == 721
    with gzip.open(path, "rt") as stream:
        assert len(stream.readlines()) == 721
    report = monitor.finish(3600)
    assert "samples" not in report
    assert report["samples_file"] == path.name
    assert report["sample_count"] == 721
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) < 5_000_000


def test_sample_append_recovers_interrupted_last_member(tmp_path: Path) -> None:
    first = PerfMonitor(tmp_path, "unused", 6)
    first.record_sample(0, parse_metrics(METRICS), gpu())
    path = tmp_path / "throughput-samples.jsonl.gz"
    complete = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(gzip.compress(b'{"seconds":5}\n')[:-4])
    assert len(list(read_samples(path))) == 1
    resumed = PerfMonitor(tmp_path, "unused", 6)
    resumed.record_sample(0, parse_metrics(METRICS), gpu())
    assert path.read_bytes().startswith(complete)
    with gzip.open(path, "rt") as stream:
        assert len(stream.readlines()) == 2
    assert len(list(read_samples(path))) == 2
    report = resumed.finish(1)
    assert report["samples_start_index"] == 1
    assert report["sample_count"] == 1
