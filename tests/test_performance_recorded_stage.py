"""Historical performance-only numbers must fail the generation speed gate."""

import json
from pathlib import Path

import pytest

from grpo.gpu_perf import GpuSample, PerfMonitor, throughput_breaches
from grpo.model_geometry import ModelGeometry


def test_recorded_stuck_timed_decode_and_duration_fail(tmp_path: Path) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "performance" / "stuck_timed.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    # Packed checkpoint header totals and served hybrid-attention config.
    geometry = ModelGeometry(
        26_895_998_464,
        17_681_660_928,
        32768,
        78_446_592,
        "served checkpoint header and hybrid-attention config",
    )
    monitor = PerfMonitor(
        tmp_path,
        "unused",
        fixture["concurrency"],
        model=geometry,
        estimate_seconds=fixture["estimated_stage_seconds"],
    )
    # Even assigning the entire KV pool to the two running requests overstates context,
    # lowering the predicted roofline and making this failure conservative.
    monitor.set_mean_context(fixture["kv_pool_tokens"] / 2)
    generated_tokens = 0.0
    for sample in fixture["samples"]:
        generated_tokens += sample["generation_tokens_per_second"] * 10
        monitor.record_sample(
            sample["seconds"],
            {
                "vllm:generation_tokens_total": generated_tokens,
                "vllm:prompt_tokens_total": 0,
                "vllm:num_requests_running": sample["running"],
                "vllm:num_requests_waiting": sample["waiting"],
                "vllm:kv_cache_usage_perc": sample["kv_usage"],
            },
            GpuSample("NVIDIA GeForce RTX 5090", None, None, None, None, None, None, None, None),
        )
    report = monitor.finish(elapsed_seconds=fixture["observed_generation_span_seconds_lower_bound"])
    assert report["decode_tokens_per_second"] == pytest.approx(14.16)
    assert "decode_bandwidth_roofline_low" in throughput_breaches(tmp_path)
    assert "stage_estimate_exceeded" in throughput_breaches(tmp_path)
    assert "gpu_utilization_unknown" in throughput_breaches(tmp_path)
