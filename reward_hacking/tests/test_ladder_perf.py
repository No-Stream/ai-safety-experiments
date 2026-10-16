"""CPU-only automatic monitoring contracts for standalone ladder runs."""

from __future__ import annotations

import os
import socket
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Self, cast

import pytest

from reward_hacking.ladder import runner
from reward_hacking.tests.test_ladder_runner import DummyScenario
from reward_hacking.trace import write_trace

if TYPE_CHECKING:
    from pathlib import Path


def test_real_grid_starts_monitor_and_records_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, Any] = {}

    class FakeMonitor:
        def __init__(self, **kwargs: Any) -> None:
            observed.update(kwargs)

        def __enter__(self) -> Self:
            observed["started"] = True
            return self

        def __exit__(self, *_args: object) -> None:
            observed["finished"] = True

        def add_time(self, category: str, seconds: float) -> None:
            observed[category] = seconds

        def set_mean_context(self, tokens: float) -> None:
            observed["mean_context"] = tokens

    monkeypatch.setattr(runner, "PerfMonitor", FakeMonitor)
    monkeypatch.setattr(runner, "throughput_breaches", lambda _path: [])
    monkeypatch.setattr(runner, "_prepare_real_run_config", lambda config: config)
    monkeypatch.setattr(runner, "_grid_has_pending_episodes", lambda _config: True)
    monkeypatch.setattr(runner, "_resolve_perf_server_pid", lambda _endpoint: 123)

    def capture_geometry(
        model: str, revision: str | None = None, kv_cache_dtype: str = "auto"
    ) -> None:
        observed["geometry_model"] = model
        observed["geometry_revision"] = revision
        observed["geometry_cache_dtype"] = kv_cache_dtype

    monkeypatch.setattr(runner, "load_model_geometry", capture_geometry)
    trace_path = tmp_path / "episode.jsonl"
    write_trace(
        trace_path,
        [
            {"record": "turn", "prompt_tokens": 1000, "output_tokens": 100},
            {"record": "turn", "prompt_tokens": 2000, "output_tokens": 200},
        ],
    )
    records = [
        {
            "trace_path": str(trace_path),
            "trace_summary": {
                "model_call_seconds": 3.0,
                "tool_execution_seconds": 2.0,
                "grading_seconds": 1.0,
            },
        }
    ]
    monkeypatch.setattr(runner, "_run_grid", lambda *_args, **_kwargs: records)
    config = runner.RunnerConfig(
        endpoint="http://localhost:8000/v1",
        models=(runner.ModelSpec("base", "synthetic-model"),),
        scenarios=(DummyScenario("dummy"),),
        arms=("naive",),
        samples=1,
        output_path=tmp_path / "records.jsonl",
        episode_root=tmp_path / "episodes",
        accept_slow="Shared-server diagnostic",
        perf_startup_seconds=12.0,
        perf_estimate_seconds=60.0,
        server_args={
            "model": "synthetic-model",
            "revision": "served-revision",
            "kv_cache_dtype": "fp8",
        },
    )

    assert runner.run_grid(config) == records
    assert observed["started"]
    assert observed["finished"]
    assert observed["metrics_url"] == "http://localhost:8000/metrics"
    assert observed["output_dir"] == tmp_path
    assert observed["accept_slow"] == "Shared-server diagnostic"
    assert observed["allowed_gpu_pids"] == (123,)
    assert observed["geometry_model"] == "synthetic-model"
    assert observed["geometry_revision"] == "served-revision"
    assert observed["geometry_cache_dtype"] == "fp8"
    assert observed["estimate_seconds"] == 60.0
    assert observed["server_startup"] == 12.0
    assert observed["model_calls"] == 3.0
    assert observed["tool_execution"] == 2.0
    assert observed["grading"] == 1.0
    assert observed["mean_context"] == 1750.0
    context = cast("dict[str, object]", config.provenance_payload()["run_context"])
    assert context["accept_slow"] == "Shared-server diagnostic"
    assert runner.run_grid(replace(config, server_args=None)) == records
    assert observed["model"] is None
    monkeypatch.setattr(
        runner, "throughput_breaches", lambda _path: ["throughput_decode_below_roofline"]
    )
    with pytest.raises(RuntimeError, match="throughput_decode_below_roofline"):
        runner.run_grid(config)
    monkeypatch.setattr(runner, "_grid_has_pending_episodes", lambda _config: False)
    report_path = tmp_path / "throughput.json"
    report_path.write_text("previous report", encoding="utf-8")
    observed.clear()
    with pytest.raises(RuntimeError, match="throughput_decode_below_roofline"):
        runner.run_grid(config)
    monkeypatch.setattr(runner, "throughput_breaches", lambda _path: [])
    assert runner.run_grid(config) == records
    assert observed == {}
    assert report_path.read_text(encoding="utf-8") == "previous report"


def test_gpu_preflight_result_reaches_the_monitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monitor_kwargs: list[dict[str, Any]] = []

    class RecordingMonitor:
        def __init__(self, **kwargs: Any) -> None:
            monitor_kwargs.append(kwargs)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def add_time(self, category: str, seconds: float) -> None:
            return None

        def set_mean_context(self, tokens: float) -> None:
            return None

    monkeypatch.setattr(runner, "PerfMonitor", RecordingMonitor)
    monkeypatch.setattr(runner, "throughput_breaches", lambda _path: [])
    monkeypatch.setattr(runner, "_prepare_real_run_config", lambda config: config)
    monkeypatch.setattr(runner, "_grid_has_pending_episodes", lambda _config: True)
    monkeypatch.setattr(runner, "_resolve_perf_server_pid", lambda _endpoint: 123)
    monkeypatch.setattr(runner, "_run_grid", lambda *_args, **_kwargs: [])
    config = runner.RunnerConfig(
        endpoint="http://localhost:8000/v1",
        models=(runner.ModelSpec("base", "synthetic-model"),),
        scenarios=(DummyScenario("dummy"),),
        arms=("naive",),
        samples=1,
        output_path=tmp_path / "records.jsonl",
        episode_root=tmp_path / "episodes",
    )

    runner.run_grid(config)
    runner.run_grid(replace(config, gpu_preflight_passed_at_stage_start=True))

    assert [kwargs["gpu_preflight_passed_at_stage_start"] for kwargs in monitor_kwargs] == [
        False,
        True,
    ]


def test_injected_episode_runner_does_not_probe_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_monitor(**_kwargs: object) -> None:
        pytest.fail("synthetic episode execution must not probe the GPU")

    monkeypatch.setattr(runner, "PerfMonitor", unexpected_monitor)
    monkeypatch.setattr(runner, "_run_grid", lambda *_args, **_kwargs: [])
    config = runner.RunnerConfig(
        endpoint="http://localhost:8000/v1",
        models=(runner.ModelSpec("base", "synthetic-model"),),
        scenarios=(DummyScenario("dummy"),),
        arms=("naive",),
        samples=1,
        output_path=tmp_path / "records.jsonl",
        episode_root=tmp_path / "episodes",
    )
    assert runner.run_grid(config, episode_runner=lambda *_args: {}) == []


def test_local_metrics_listener_pid_can_be_allowed_without_a_new_cli_flag() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        assert runner._listening_process_ids(port) == (os.getpid(),)
