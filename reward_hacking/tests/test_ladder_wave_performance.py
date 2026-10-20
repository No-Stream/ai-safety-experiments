"""Exercise the in-process wave entry point with CPU-only server/NVML telemetry."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Self

import pytest

from grpo.gpu_perf import GpuSample, PerfMonitor, parse_metrics
from grpo.perf_samples import SAMPLES_FILE, read_samples
from reward_hacking.organism_rl import health, ladder_wave, probe_v4, wave_v4
from reward_hacking.tests.test_organism_rl_ladder_wave import _FakeLadderEpisodeSource

if TYPE_CHECKING:
    from collections.abc import Iterator

    from reward_hacking.vllm_server import VLLMServerConfig


@pytest.fixture
def metrics_endpoint() -> Iterator[str]:
    metrics = parse_metrics(Path("tests/fixtures/performance/vllm_metrics.prom").read_text())
    metrics["vllm:num_requests_running"] = 6
    metrics["vllm:num_requests_waiting"] = 1
    payload = "\n".join(f"{name} {value}" for name, value in metrics.items()).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/metrics"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("accept_slow", [None, "intentional CPU diagnostic"])
def test_managed_ladder_wave_persists_and_enforces_throughput(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accept_slow: str | None, metrics_endpoint: str
) -> None:
    class FakeServer:
        def __init__(self, config: VLLMServerConfig) -> None:
            self.config = config
            self.base_url = metrics_endpoint
            self.process = SimpleNamespace(pid=12345)

        def __enter__(self) -> Self:
            assert self.config.log_path is not None
            self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.log_path.open("a") as log_file:
                log_file.write("GPU KV cache size: 100,000 tokens\n")
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(probe_v4, "VLLMServer", FakeServer)
    monkeypatch.setattr(probe_v4, "require_free_gpu", lambda: None)
    monkeypatch.setattr(
        probe_v4, "resolve_gpu_engine_settings", lambda _torch, **_kw: (0.9, "synthetic")
    )
    monkeypatch.setattr(ladder_wave, "LadderEpisodeSource", _FakeLadderEpisodeSource)
    monkeypatch.setattr(probe_v4, "_runtime_model_revision", lambda *_args: "synthetic-revision")
    monkeypatch.setattr(probe_v4, "_served_model_max_len", lambda *_args: 4096)
    monkeypatch.setattr("grpo.gpu_perf._running_under_wsl", lambda: True)
    monkeypatch.setattr(
        "reward_hacking.generation_perf.load_model_geometry", lambda *_args, **_kwargs: None
    )

    monkeypatch.setattr(
        "reward_hacking.generation_perf.PerfMonitor",
        partial(
            PerfMonitor,
            gpu_reader=lambda: GpuSample(
                "synthetic", 1, 1, 1, 1, 1, 1, 2, (45678,), ((45678, None),)
            ),
        ),
    )
    arguments = ["--cell", "f2-golden-orig", "--generator", "27b"]
    arguments.extend(("--hint-strength", "none", "--episodes", "8", "--max-turns", "80"))
    if accept_slow is not None:
        arguments.extend(("--accept-slow", accept_slow))
    arguments.extend(("--max-model-len", "4096", "--output-root", str(tmp_path / "launcher")))
    args = wave_v4.parse_args(arguments)

    def collect(settings: probe_v4.CellRunConfig) -> None:
        settings = replace(settings, output_dir=tmp_path)
        ladder_wave.run_ladder_wave(
            endpoint=settings.endpoint,
            model_path=settings.model,
            model_revision=settings.model_revision,
            served_model_id=settings.model,
            max_model_len=settings.max_model_len,
            output_dir=settings.output_dir,
            server_args=settings.serving_config,
        )

    if accept_slow is None:
        with pytest.raises(ladder_wave.LadderWaveError, match="throughput"):
            probe_v4.run_generator(args, "27b", collect)
    else:
        probe_v4.run_generator(args, "27b", collect)
    report = json.loads((tmp_path / "throughput.json").read_text())
    assert "gpu_sm_utilization_low" in report["breaches"]
    assert report["gpu_preflight_passed_at_stage_start"] is True
    assert report["wsl_detected_at_stage_start"] is True
    assert report["platform_unknown_gpu_fields"] == ["foreign_gpu_memory_unknown_pids"]
    assert "foreign_gpu_memory_unknown" not in report["breaches"]
    assert report["accept_slow"] == accept_slow
    assert len(list(read_samples(tmp_path / SAMPLES_FILE))) >= 2
    health_report = json.loads((tmp_path / "health.json").read_text())
    assert bool(health_report["assertion_breaches"]) is (accept_slow is None)
    assert health.main(
        [
            "--records",
            str(tmp_path / "records.jsonl"),
            "--output",
            str(tmp_path / "health.json"),
            "--assert-throughput",
        ]
    ) == (1 if accept_slow is None else 0)
    if accept_slow is None:
        original_samples = (tmp_path / SAMPLES_FILE).read_bytes()
        args.accept_slow = "reviewed completed slow stage"
        probe_v4.run_generator(args, "27b", collect)
        assert (tmp_path / SAMPLES_FILE).read_bytes() == original_samples
        accepted = json.loads((tmp_path / "throughput.json").read_text())
        assert accepted["accept_slow"] == "reviewed completed slow stage"
        assert json.loads((tmp_path / "health.json").read_text())["assertion_breaches"] == []
        (tmp_path / SAMPLES_FILE).unlink()
        probe_v4.run_generator(args, "27b", collect)
        assert len(list(read_samples(tmp_path / SAMPLES_FILE))) >= 2


def test_managed_generation_refuses_failed_preflight_before_server_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            "f2-golden-orig",
            "--generator",
            "27b",
            "--hint-strength",
            "none",
            "--episodes",
            "8",
            "--max-turns",
            "80",
        ]
    )
    monkeypatch.setattr(probe_v4, "_runtime_model_revision", lambda *_args: "synthetic-revision")

    def failed_preflight() -> None:
        raise RuntimeError("synthetic busy GPU")

    def unexpected_allocation(_torch: object, **_kwargs: object) -> tuple[float, str]:
        pytest.fail("GPU allocation attempted after failed preflight")

    monkeypatch.setattr(probe_v4, "require_free_gpu", failed_preflight)
    monkeypatch.setattr(probe_v4, "resolve_gpu_engine_settings", unexpected_allocation)
    with pytest.raises(RuntimeError, match="synthetic busy GPU"):
        probe_v4.run_generator(args, "27b", lambda _settings: pytest.fail("generation started"))
