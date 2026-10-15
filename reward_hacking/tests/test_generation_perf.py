"""Performance controls follow the generation subprocess and remain outside task identity."""

import json
from pathlib import Path

import pytest

from grpo.gpu_perf import PerfMonitor
from reward_hacking.generation_perf import (
    episode_context_usage,
    generation_monitor,
    performance_cli_arguments,
    record_episode_timings,
    stage_generation_monitor,
)
from reward_hacking.organism_rl import generate, wave_v4
from reward_hacking.organism_rl.generate import parse_args as parse_generation_args
from reward_hacking.organism_rl.probe_v4 import (
    CellRunConfig,
    build_generate_command,
    build_health_command,
)
from reward_hacking.tests.test_organism_rl_generate import SyntheticEpisodeSource


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
