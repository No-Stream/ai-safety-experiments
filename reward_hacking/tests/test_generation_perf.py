"""Performance controls follow the generation subprocess and remain outside task identity."""

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Self

import pytest

from grpo.gpu_perf import GpuSample, PerfMonitor
from reward_hacking.generation_perf import (
    episode_context_usage,
    generation_monitor,
    performance_cli_arguments,
    record_episode_timings,
    stage_generation_monitor,
    stage_performance_arguments,
)
from reward_hacking.organism_rl import generate, probe_v4, wave_v4
from reward_hacking.organism_rl.generate import parse_args as parse_generation_args
from reward_hacking.organism_rl.probe_v4 import (
    CellRunConfig,
    build_generate_command,
    build_health_command,
)
from reward_hacking.tests.test_organism_rl_generate import SyntheticEpisodeSource
from reward_hacking.vllm_server import VLLMServerConfig


@pytest.mark.parametrize("generator", ["base", "27b"])
@pytest.mark.parametrize("samples_per_task", [1, 2])
def test_managed_wave_preflight_reaches_generation_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generator: Literal["base", "27b"],
    samples_per_task: int,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "llama"}')
    hints_path = tmp_path / "hints.json"
    hints_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {"family": "grader_mechanics", "strength": "specific", "text": "synthetic hint"}
                ],
            }
        ),
        encoding="utf-8",
    )
    args = wave_v4.parse_args(
        [
            "--hints",
            str(hints_path),
            "--cell",
            "f2-golden-orig",
            "--generator",
            generator,
            "--model",
            str(model_dir),
            "--output-root",
            str(tmp_path / "output"),
            "--hint-strength",
            "none",
            "--episodes",
            "2",
            "--max-turns",
            "40",
            "--samples-per-task",
            str(samples_per_task),
        ]
    )
    cell = next(cell for cell in wave_v4.WAVE_CELLS if cell.name == args.cell)
    preflight_calls: list[bool] = []

    def preflight() -> None:
        preflight_calls.append(True)

    class Server:
        def __init__(self, config: VLLMServerConfig) -> None:
            assert preflight_calls == [True]
            assert config.log_path is not None
            config.log_path.parent.mkdir(parents=True, exist_ok=True)
            config.log_path.write_text("GPU KV cache size: 262,144 tokens\n")
            self.base_url = "http://localhost:8000"
            self.process = SimpleNamespace(pid=12345)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            pass

    def model_response(url: str, *, timeout: float) -> BytesIO:
        assert url.endswith("/v1/models")
        assert timeout == 10
        return BytesIO(
            json.dumps({"data": [{"id": str(model_dir), "max_model_len": 131072}]}).encode()
        )

    def metrics_response(url: str, *, timeout: float) -> BytesIO:
        assert url == "http://localhost:8000/metrics"
        assert timeout == 3
        return BytesIO(Path("tests/fixtures/performance/vllm_metrics.prom").read_bytes())

    monkeypatch.setattr(probe_v4, "require_free_gpu", preflight)
    monkeypatch.setattr(probe_v4, "VLLMServer", Server)
    monkeypatch.setattr(
        probe_v4, "resolve_gpu_engine_settings", lambda _torch, **_kw: (0.5, "synthetic GPU")
    )
    monkeypatch.setattr(probe_v4, "urlopen", model_response)
    monkeypatch.setattr("grpo.gpu_perf._running_under_wsl", lambda: True)
    monkeypatch.setattr("grpo.gpu_perf.urllib.request.urlopen", metrics_response)
    reports: list[dict[str, object]] = []

    def run(settings: CellRunConfig) -> None:
        stage_settings = wave_v4.stage_settings(args, settings, ())
        command = build_generate_command(cell, stage_settings)
        generation_args = generate.parse_args(command[3:])
        monitor = generate.generation_monitor(generation_args)
        monitor.gpu_reader = lambda: GpuSample(
            "synthetic GPU",
            90,
            75,
            100,
            2000,
            10000,
            1024,
            4096,
            (12345, 45678),
            ((12345, 1024), (45678, None)),
        )
        with monitor:
            pass
        reports.append(json.loads((monitor.output_dir / "throughput.json").read_text()))

    wave_v4.run_generator(args, generator, run)

    assert len(reports) == 1
    report = reports[0]
    assert report["gpu_preflight_passed_at_stage_start"] is True
    assert report["wsl_detected_at_stage_start"] is True
    assert report["platform_unknown_gpu_fields"] == ["foreign_gpu_memory_unknown_pids"]
    breaches = report["breaches"]
    assert isinstance(breaches, list)
    assert "foreign_gpu_memory_unknown" not in breaches
    assert stage_performance_arguments.get() == ()


def test_generation_accepts_explicit_slow_reason_and_thresholds(tmp_path: Path) -> None:
    args = parse_generation_args(
        [
            "--endpoint",
            "http://localhost:8000/v1",
            "--max-model-len",
            "4096",
            "--output-dir",
            str(tmp_path),
            "--accept-slow",
            "intentional diagnostic",
            "--perf-min-decode-fraction",
            "0.3",
            "--perf-estimate-seconds",
            "1800",
        ]
    )
    assert args.accept_slow == "intentional diagnostic"
    assert args.perf_min_decode_fraction == 0.3
    assert args.perf_estimate_seconds == 1800


def test_generation_preflight_evidence_flag_is_forwarded_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_args = [
        "--endpoint",
        "http://localhost:8000/v1",
        "--max-model-len",
        "4096",
        "--output-dir",
        str(tmp_path),
    ]
    assert "--gpu-preflight-passed-at-stage-start" not in performance_cli_arguments(
        parse_generation_args(base_args)
    )
    args = parse_generation_args([*base_args, "--gpu-preflight-passed-at-stage-start"])
    forwarded = performance_cli_arguments(args)
    assert "--gpu-preflight-passed-at-stage-start" in forwarded

    monitor_arguments: dict[str, object] = {}

    class MonitorStub:
        def __init__(self, **kwargs: object) -> None:
            monitor_arguments.update(kwargs)

        def add_time(self, category: str, seconds: float) -> None:
            assert category == "server_startup"
            assert seconds == 0

    monkeypatch.setattr("reward_hacking.generation_perf.PerfMonitor", MonitorStub)
    monkeypatch.setattr(
        "reward_hacking.generation_perf.load_model_geometry", lambda *args, **kwargs: None
    )

    generation_monitor(args)

    assert monitor_arguments["gpu_preflight_passed_at_stage_start"] is True

    stage_generation_monitor(
        output_dir=tmp_path,
        endpoint="http://localhost:8000/v1",
        model="synthetic-model",
        model_revision=None,
        concurrency=2,
        serving_config=None,
        performance_arguments=("--gpu-preflight-passed-at-stage-start",),
        default_estimate_seconds=3600,
    )
    assert monitor_arguments["estimate_seconds"] == 3600
    assert monitor_arguments["gpu_preflight_passed_at_stage_start"] is True

    stage_generation_monitor(
        output_dir=tmp_path,
        endpoint="http://localhost:8000/v1",
        model="synthetic-model",
        model_revision=None,
        concurrency=2,
        serving_config=None,
        performance_arguments=("--perf-estimate-seconds", "42"),
        default_estimate_seconds=3600,
    )
    assert monitor_arguments["estimate_seconds"] == 42


def test_wave_forwards_performance_policy_and_turn_scaled_estimate(tmp_path: Path) -> None:
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
            "40",
            "--accept-slow",
            "profile kernels",
        ]
    )
    cell = next(cell for cell in wave_v4.WAVE_CELLS if cell.name == args.cell)
    settings = CellRunConfig(
        generator="27b",
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=tmp_path,
        hints_path=tmp_path / "hints.json",
        leak_assets=None,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=4096,
        model_revision=None,
        serving_config={"kv_cache_dtype": "fp8"},
        performance_arguments=performance_cli_arguments(args),
    )
    settings = wave_v4.stage_settings(args, settings, ())
    command = build_generate_command(cell, settings)
    assert command[command.index("--accept-slow") + 1] == "profile kernels"
    assert float(command[command.index("--perf-estimate-seconds") + 1]) == 1800


@pytest.mark.parametrize("generator", ["base", "27b", "v3"])
@pytest.mark.parametrize("assert_behavior", [False, True])
def test_wave_health_always_asserts_performance(
    generator: str, assert_behavior: bool, tmp_path: Path
) -> None:
    command = build_health_command(
        tmp_path / "records.jsonl",
        tmp_path / "baseline.jsonl",
        generator=generator,  # type: ignore[arg-type]
        assert_health=assert_behavior,
    )
    assert "--assert-throughput" in command


@pytest.mark.parametrize("reason", ["", "  "])
def test_empty_slow_override_is_rejected(reason: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_generation_args(
            [
                "--endpoint",
                "http://localhost:8000",
                "--max-model-len",
                "4096",
                "--output-dir",
                str(tmp_path),
                "--accept-slow",
                reason,
            ]
        )


def test_episode_work_and_context_are_aggregated_without_resumed_work(tmp_path: Path) -> None:
    turns = [
        {"record": "turn", "prompt_tokens": 100, "output_tokens": 20},
        {"record": "turn", "prompt_tokens": 400, "output_tokens": 10},
    ]
    context = episode_context_usage(turns)
    assert context["generated_tokens"] == 30
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("".join(json.dumps(turn) + "\n" for turn in turns))
    monitor = PerfMonitor(tmp_path, "unused", 6)
    record_episode_timings(
        monitor,
        [
            {
                "trace_path": str(trace_path),
                "trace_summary": {
                    "model_call_seconds": 10,
                    "tool_execution_seconds": 2,
                    "grading_seconds": 1,
                    "performance_context": context,
                },
            }
        ],
    )
    assert monitor.timings == {
        "model_call_seconds": 10,
        "tool_execution_seconds": 2,
        "grading_seconds": 1,
    }
    assert monitor.mean_context == pytest.approx((20 * 110 + 10 * 405) / 30)


def test_complete_resume_preserves_existing_throughput_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generate, "MINIMUM_FREE_DISK_BYTES", 0)
    source = SyntheticEpisodeSource(tmp_path)
    specs = (generate.TaskSpec("synthetic", "ilcb", "oneoff"),)
    generate.generate_episodes(
        output_dir=tmp_path,
        source=source,
        task_specs=specs,
        samples_per_task=1,
        concurrency=1,
        provenance={"run": "performance-resume"},
    )
    (tmp_path / "throughput.json").write_text('{"breaches": [], "accept_slow": null}\n')
    (tmp_path / "summary.md").write_text("Existing summary with throughput table\n")
    monitor = PerfMonitor(tmp_path, "must-not-be-requested", 1)
    generate.generate_episodes(
        output_dir=tmp_path,
        source=source,
        task_specs=specs,
        samples_per_task=1,
        concurrency=1,
        provenance={"run": "performance-resume"},
        performance_monitor=monitor,
    )
    assert not monitor.samples
    assert (tmp_path / "summary.md").read_text() == "Existing summary with throughput table\n"
    assert (tmp_path / "throughput.json").read_text() == '{"breaches": [], "accept_slow": null}\n'
