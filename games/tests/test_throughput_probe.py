"""Pin the throughput probe's config mapping and its measurement arithmetic, offline.

The probe exists to price real training configurations (docs/scratch owns the study; the module
docstring owns the why), and the two things that could silently corrupt a price are pinned here:
the probe must build the SAME `GameTrainConfig` a real run would — every refusal included, since
a probe that quietly relaxed the completion floor or the vLLM-only rule would price a run nobody
launches — and the summary arithmetic must exclude warmup steps, because the first step carries
engine boot and CUDA-graph capture and a median that includes it overprices every config by the
same misleading margin.

Offline: no trainer, no GPU, no weights. The GPU-facing half (`run_probe`) is deliberately thin
composition of `games.train`'s own construction path plus `grpo.throughput`'s timing callback,
both tested in their own homes.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from games.throughput_probe import (
    _find_phase_timer,
    apply_plan_step_minutes,
    parse_args,
    summarize_phase_timings,
    summarize_probe,
)
from games.train import GameTrainConfig
from grpo.throughput import STEP_PHASES, StepPhaseTimer, timing_metric

if TYPE_CHECKING:
    from pathlib import Path

ARM = "twin-pd-group"


def probe_argv(**overrides: str) -> list[str]:
    """A minimal valid command line, overridable per test."""
    values: dict[str, str] = {
        "--arm": ARM,
        "--json-out": "artifacts/throughput/probe.json",
        **overrides,
    }
    argv = [flag for pair in values.items() for flag in pair]
    argv.append("--generate-fresh")
    return argv


class TestProbeConfigConstruction:
    def test_flags_reach_the_real_train_config(self) -> None:
        """The probe prices the config it names: shape, util and budget must land verbatim."""
        argv = probe_argv(
            **{
                "--prompts-per-step": "16",
                "--num-generations": "8",
                "--vllm-gpu-memory-utilization": "0.55",
                "--max-completion-tokens": "32768",
                "--measured-steps": "4",
                "--warmup-steps": "1",
            }
        )
        argv.append("--no-vllm-importance-sampling-correction")
        spec = parse_args(argv)
        config = spec.config
        assert config.vllm_importance_sampling_correction is False
        assert isinstance(config, GameTrainConfig)
        assert config.prompts_per_step == 16
        assert config.num_generations == 8
        assert config.vllm_gpu_memory_utilization == 0.55
        assert config.max_completion_tokens == 32768
        assert config.max_steps == 5, "max_steps must be warmup + measured, nothing else"
        assert spec.warmup_steps == 1
        assert config.micro_batch_size is None, "unset means derive-from-VRAM, as games.train does"

    def test_an_explicit_micro_batch_reaches_the_config(self) -> None:
        """The training pass measured ~76% of a 2B colocate step, so mb is the live lever there;
        an explicit value must land verbatim for the probe to price it."""
        spec = parse_args(probe_argv(**{"--micro-batch-size": "2"}))
        assert spec.config.micro_batch_size == 2

    def test_no_checkpoint_lands_inside_the_timed_window(self) -> None:
        """A checkpoint save plus S3 sync inside a timed step would price storage, not training."""
        spec = parse_args(probe_argv(**{"--measured-steps": "4", "--warmup-steps": "1"}))
        assert spec.config.save_steps > spec.config.max_steps

    def test_the_correction_memory_refusal_prices_the_chunked_pass_as_games_train_does(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The correction's old-logps pass runs over token slices, so a 32,768 budget with the
        default chunk is accepted and a chunk at or above the row is what the memory refusal fires
        on. The probe must price exactly the shape games.train would launch, in both directions."""
        monkeypatch.delenv("GAMES_VLLM_IS_CORRECTION", raising=False)
        spec = parse_args(probe_argv(**{"--max-completion-tokens": "32768"}))
        assert spec.config.vllm_importance_sampling_correction is True
        with pytest.raises(ValueError, match="importance-sampling correction"):
            replace(spec.config, old_logps_chunk_tokens=32768)

    def test_the_completion_floor_still_refuses(self) -> None:
        """A probe that relaxed the floor would price a run games.train refuses to launch."""
        with pytest.raises(ValueError, match="completion tokens"):
            parse_args(probe_argv(**{"--max-completion-tokens": "1024"}))

    def test_allow_short_completions_is_the_documented_escape_hatch(self) -> None:
        """The flag exists exactly for timing probes; asked for, the floor steps aside."""
        argv = probe_argv(**{"--max-completion-tokens": "1024"})
        argv.append("--allow-short-completions")
        spec = parse_args(argv)
        assert spec.config.max_completion_tokens == 1024

    def test_the_probe_smoke_shape_is_constructable(self) -> None:
        """The L4 smoke's exact settings must survive config validation before a card is touched."""
        argv = probe_argv(
            **{
                "--model": "Qwen/Qwen3-0.6B",
                "--prompts-per-step": "2",
                "--num-generations": "4",
                "--max-completion-tokens": "2048",
                "--measured-steps": "2",
                "--warmup-steps": "1",
            }
        )
        argv.append("--allow-short-completions")
        spec = parse_args(argv)
        assert spec.config.model_id == "Qwen/Qwen3-0.6B"
        assert spec.config.max_steps == 3


class TestPlanStepMinutes:
    def test_it_reaches_the_pace_guard_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard reads the env at trainer build; the probe must have set it by then."""
        monkeypatch.delenv("GAMES_PLAN_STEP_MINUTES", raising=False)
        apply_plan_step_minutes(25.0)
        assert os.environ["GAMES_PLAN_STEP_MINUTES"] == "25.0"

    def test_none_leaves_the_environment_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GAMES_PLAN_STEP_MINUTES", "13.5")
        apply_plan_step_minutes(None)
        assert os.environ["GAMES_PLAN_STEP_MINUTES"] == "13.5"

    def test_a_nonpositive_plan_refuses_at_parse_not_on_the_box(self) -> None:
        with pytest.raises(ValueError, match="plan"):
            apply_plan_step_minutes(0.0)


def step_record(step: int, seconds: float) -> dict[str, float]:
    """One record shaped like grpo.throughput.StepTimingCallback emits."""
    return {
        "step": float(step),
        "seconds": seconds,
        "peak_allocated_gib": 10.0 + step,
        "peak_reserved_gib": 12.0 + step,
        "device_used_gib": 30.0 + step,
    }


class TestSummarizeProbe:
    def test_warmup_steps_are_excluded_from_the_median(self) -> None:
        """Step 1 carries engine boot and graph capture; a median including it prices that, not training."""
        records = [step_record(1, 900.0), step_record(2, 600.0), step_record(3, 660.0)]
        summary = summarize_probe(
            step_records=records,
            generation_seconds=[800.0, 500.0, 560.0],
            warmup_steps=1,
            episodes_per_step=64,
        )
        assert summary["median_step_seconds"] == 630.0
        assert summary["per_step_seconds_measured"] == [600.0, 660.0]
        assert summary["per_step_seconds_all"] == [900.0, 600.0, 660.0]

    def test_throughput_arithmetic(self) -> None:
        records = [step_record(1, 700.0), step_record(2, 600.0)]
        summary = summarize_probe(
            step_records=records,
            generation_seconds=[650.0, 480.0],
            warmup_steps=1,
            episodes_per_step=64,
        )
        assert summary["steps_per_hour"] == 6.0
        assert summary["episodes_per_hour"] == 384.0
        assert summary["hours_per_70_steps"] == pytest.approx(70 * 600.0 / 3600.0)
        assert summary["generation_fraction_of_step"] == pytest.approx(480.0 / 600.0)

    def test_nothing_measured_raises_rather_than_reporting_warmup(self) -> None:
        """One-step 'measurements' are how an engine-boot number becomes a config's price."""
        with pytest.raises(ValueError, match="warmup"):
            summarize_probe(
                step_records=[step_record(1, 900.0)],
                generation_seconds=[800.0],
                warmup_steps=1,
                episodes_per_step=64,
            )

    def test_misaligned_generation_timings_are_flagged_not_hidden(self) -> None:
        """steps_per_generation != 1 would desync the two lists; the summary must say so."""
        records = [step_record(1, 700.0), step_record(2, 600.0)]
        summary = summarize_probe(
            step_records=records,
            generation_seconds=[650.0],
            warmup_steps=1,
            episodes_per_step=64,
        )
        assert summary["generation_timings_aligned"] is False
        assert summary["generation_fraction_of_step"] is None

    def test_peaks_cover_every_step_including_warmup(self) -> None:
        """Memory peaks answer 'does it fit', and warmup allocations count against the card too."""
        records = [step_record(1, 700.0), step_record(2, 600.0), step_record(3, 620.0)]
        summary = summarize_probe(
            step_records=records,
            generation_seconds=[],
            warmup_steps=1,
            episodes_per_step=64,
        )
        assert summary["peak_device_used_gib"] == 33.0

    def test_the_summary_survives_a_json_round_trip(self, tmp_path: Path) -> None:
        records = [step_record(1, 700.0), step_record(2, 600.0)]
        summary = summarize_probe(
            step_records=records,
            generation_seconds=[650.0, 480.0],
            warmup_steps=1,
            episodes_per_step=64,
        )
        path = tmp_path / "probe.json"
        path.write_text(json.dumps(summary, default=str))
        assert json.loads(path.read_text())["median_step_seconds"] == 600.0


def phase_history(*, steps: int = 3) -> list[dict[str, float]]:
    """Return complete timer log rows with distinct values for every phase and step."""
    return [
        {
            timing_metric(phase): float(step + phase_index + 1)
            for phase_index, phase in enumerate(STEP_PHASES)
        }
        for step in range(steps)
    ]


class TestSummarizePhaseTimings:
    def test_warmup_is_excluded_and_budget_aliases_use_phase_medians(self) -> None:
        history = phase_history()

        summary = summarize_phase_timings(
            history=history,
            warmup_steps=1,
            expected_steps=len(history),
            memory_peaks={
                "peak_allocated_gib": 11.0,
                "peak_reserved_gib": 12.0,
                "peak_device_used_gib": 13.0,
            },
        )

        phase_seconds_all = cast("dict[str, list[float]]", summary["phase_seconds_all"])
        phase_seconds_measured = cast("dict[str, list[float]]", summary["phase_seconds_measured"])
        phase_medians_seconds = cast("dict[str, float]", summary["phase_medians_seconds"])
        assert phase_seconds_all["backward"] == [8.0, 9.0, 10.0]
        assert phase_seconds_measured["backward"] == [9.0, 10.0]
        assert phase_medians_seconds["backward"] == 9.5
        assert summary["throughput"] == {
            "generation": 4.5,
            "backward": 9.5,
            "optimizer": 11.5,
            "phase_medians_seconds": phase_medians_seconds,
        }
        assert summary["memory"] == {
            "peak_allocated_gib": 11.0,
            "peak_reserved_gib": 12.0,
            "peak_device_used_gib": 13.0,
        }

    def test_missing_phase_metric_is_a_hard_failure(self) -> None:
        history = phase_history(steps=2)
        del history[1][timing_metric("backward")]

        with pytest.raises(ValueError, match="backward"):
            summarize_phase_timings(
                history=history,
                warmup_steps=0,
                expected_steps=len(history),
                memory_peaks={
                    "peak_allocated_gib": 1.0,
                    "peak_reserved_gib": 1.0,
                    "peak_device_used_gib": 1.0,
                },
            )

    def test_phase_summary_survives_json_round_trip(self, tmp_path: Path) -> None:
        history = phase_history(steps=2)
        summary = summarize_phase_timings(
            history=history,
            warmup_steps=0,
            expected_steps=len(history),
            memory_peaks={
                "peak_allocated_gib": 1.0,
                "peak_reserved_gib": 2.0,
                "peak_device_used_gib": 3.0,
            },
        )
        path = tmp_path / "probe.json"
        path.write_text(json.dumps(summary))

        payload = json.loads(path.read_text())
        assert payload["throughput"]["optimizer"] == 10.5
        assert payload["memory"]["peak_device_used_gib"] == 3.0


def test_probe_reuses_the_single_timer_attached_by_games_train() -> None:
    timer = StepPhaseTimer()
    trainer = SimpleNamespace(callback_handler=SimpleNamespace(callbacks=[timer]))

    assert _find_phase_timer(trainer) is timer

    for callbacks in ([], [timer, StepPhaseTimer()]):
        trainer.callback_handler.callbacks = callbacks
        with pytest.raises(RuntimeError, match="exactly one"):
            _find_phase_timer(trainer)
