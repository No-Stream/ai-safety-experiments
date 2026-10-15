"""CPU-only controls for throughput arithmetic and telemetry."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from grpo.gpu_perf import (
    GpuSample,
    PerfMonitor,
    PerfThresholds,
    decode_roofline,
    device_peak,
    histogram_summary,
    parse_metrics,
    throughput_breaches,
)
from grpo.model_geometry import ModelGeometry

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


def test_missing_telemetry_still_writes_report_and_summary(tmp_path: Path) -> None:
    monitor = PerfMonitor(tmp_path, "unused", 6, model=MODEL)
    report = monitor.finish(10)
    assert "throughput_telemetry_unavailable" in throughput_breaches(tmp_path)
    assert report["samples"] == []
    assert "Mean KV cache use | unknown" in (tmp_path / "summary.md").read_text()
