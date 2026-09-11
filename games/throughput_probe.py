"""Price one real games GRPO configuration in seconds per optimizer step, on the real path.

Every production arm so far inherited its shape — 8 prompts x 8 generations, engine util 0.35 —
from the first box that worked, and nobody has measured whether that is the right point on this
card class. This probe answers with a short REAL training segment: the actual corpus, the actual
reward pipeline, thinking on, completion caps at the values real runs use. `grpo.throughput` is
the wrong tool for that question on purpose — it forces fixed-length completions on a synthetic
reward to price the machinery, while here rollout raggedness IS most of a real step's cost.

Two properties keep a probe number honest:

*   **It builds the run through `games.train`'s own construction path** — the same
    `GameTrainConfig` with every refusal live (the completion floor, the vLLM-only rule, the
    importance-sampling memory refusal), then `_prepare_run` and `_build_trainer` verbatim. A
    probe with its own trainer assembly would drift from what real runs execute, and its prices
    would describe nothing anyone launches. Timing is added ON the built trainer, never inside
    the construction.
*   **Warmup steps are excluded from the medians.** The first step carries engine boot and CUDA
    graph capture; a median including it overprices every config. The summary keeps the raw
    per-step list, warmup included, so the exclusion is inspectable rather than trusted.

    uv run python -m games.throughput_probe --arm pd-unstated-group --corpus <jsonl> \
        --prompts-per-step 16 --num-generations 8 --vllm-gpu-memory-utilization 0.35 \
        --measured-steps 4 --warmup-steps 1 --json-out artifacts/throughput/probe-2b-16x8.json

The JSON lands wherever `--json-out` says; the study that consumes these, with its grids and
registered expectations, is docs/scratch material (2026-08-29 throughput-configs dispatch).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from games.deltanet_kernels import assert_bridged_kernel_matches_call_site, bridge_decode_kernel
from games.generation import (
    assert_vllm_rollouts,
    colocate_gpu_fraction,
    colocate_importance_sampling_correction,
)
from games.pace_guard import PLAN_STEP_MINUTES_ENV
from games.preflight import assert_trainer_generates_through_vllm, default_cuda_allocator_config
from games.rewards import GRADING_VS_FIXED_MIX
from games.termination import MEASURED_TERMINATION_BUDGET
from games.train import (
    DEFAULT_MODEL_ID,
    CompletedRun,
    GameTrainConfig,
    PreparedRun,
    # Private, consumed anyway: a probe must price the exact path real runs execute, never a re-assembly.
    _build_trainer,  # pyright: ignore[reportPrivateUsage]
    _prepare_run,  # pyright: ignore[reportPrivateUsage]
    check_built_trainer,
    constant_by_construction_metrics,
    default_output_dir,
    read_back_metrics,
    required_metrics_for,
    write_json,
)
from grpo.throughput import StepTimingCallback, instrument_generation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600.0
# The repo's training-run unit: every flagship arm is 70 optimizer steps.
RUN_STEPS = 70


@dataclass(frozen=True)
class ProbeSpec:
    """One probe invocation: the real training config plus what of it is warmup."""

    config: GameTrainConfig
    warmup_steps: int
    json_out: str

    def __post_init__(self) -> None:
        """Refuse a spec whose timed window is empty or whose output has nowhere to land."""
        if self.warmup_steps < 0:
            raise ValueError(f"warmup cannot be negative, {self.warmup_steps=}")
        if self.config.max_steps <= self.warmup_steps:
            raise ValueError(
                f"nothing would be measured: max_steps={self.config.max_steps} is not past "
                f"warmup_steps={self.warmup_steps}"
            )
        if not self.json_out:
            raise ValueError(
                "--json-out is required; a probe that keeps no artifact priced nothing"
            )


def apply_plan_step_minutes(minutes: float | None) -> None:
    """Hand the pace guard this shape's expected step cost, before the trainer is built.

    The guard kills a run whose trailing pace exceeds 2.5x its plan figure, and its default plan
    is the 2B production shape's 13.5 minutes — a 9B probe at 24 minutes per step would be killed
    mid-measurement for being exactly as slow as its anchors say it is. Setting the environment
    here rather than in a runner keeps the value next to the config it describes, and a
    nonpositive value fails at argument parsing instead of at trainer construction on a rented
    card.
    """
    if minutes is None:
        return
    if minutes <= 0:
        raise ValueError(f"plan step minutes must be positive, got {minutes}")
    os.environ[PLAN_STEP_MINUTES_ENV] = str(minutes)


def summarize_probe(
    *,
    step_records: Sequence[Mapping[str, float]],
    generation_seconds: Sequence[float],
    warmup_steps: int,
    episodes_per_step: int,
) -> dict[str, object]:
    """Turn raw per-step timings into the throughput figures the config table quotes.

    Pure over primitives so the arithmetic is testable without a GPU. Peaks run over EVERY step
    including warmup — they answer "does this configuration fit", and warmup allocations count
    against the card like any other. Generation timings align one-to-one with optimizer steps
    only under TRL's derived one-generation-per-step schedule (which `games.train` relies on for
    trace completeness); a mismatch is reported as misalignment rather than silently truncated.
    """
    if len(step_records) <= warmup_steps:
        raise ValueError(
            f"{len(step_records)} step(s) timed against {warmup_steps} warmup step(s): nothing "
            f"was measured, and reporting a warmup number as a price is the failure this guard "
            f"exists for"
        )
    all_seconds = [record["seconds"] for record in step_records]
    measured = all_seconds[warmup_steps:]
    median_step = statistics.median(measured)
    aligned = len(generation_seconds) == len(step_records)
    generation_measured = list(generation_seconds[warmup_steps:]) if aligned else []
    median_generation = statistics.median(generation_measured) if generation_measured else None
    return {
        "n_steps_total": len(step_records),
        "n_steps_measured": len(measured),
        "warmup_steps": warmup_steps,
        "per_step_seconds_all": all_seconds,
        "per_step_seconds_measured": measured,
        "median_step_seconds": median_step,
        "mean_step_seconds": statistics.fmean(measured),
        "min_step_seconds": min(measured),
        "max_step_seconds": max(measured),
        "generation_seconds_all": list(generation_seconds),
        "generation_seconds_measured": generation_measured,
        "median_generation_seconds": median_generation,
        "generation_fraction_of_step": (
            median_generation / median_step if median_generation is not None else None
        ),
        "generation_timings_aligned": aligned,
        "episodes_per_step": episodes_per_step,
        "steps_per_hour": SECONDS_PER_HOUR / median_step,
        "episodes_per_hour": episodes_per_step * SECONDS_PER_HOUR / median_step,
        "hours_per_70_steps": RUN_STEPS * median_step / SECONDS_PER_HOUR,
        "peak_allocated_gib": max(record["peak_allocated_gib"] for record in step_records),
        "peak_reserved_gib": max(record["peak_reserved_gib"] for record in step_records),
        "peak_device_used_gib": max(record["device_used_gib"] for record in step_records),
    }


def run_probe(spec: ProbeSpec, *, kernel_bridge: dict[str, object] | None) -> dict[str, object]:
    """Execute one probe end to end and persist its artifact.

    Composition only: the construction is `games.train`'s, the step clock is
    `grpo.throughput`'s, and the read-back gate is the same one real runs pass — a probe whose
    reward pipeline logged nothing should fail the way a real run would, not report a price for
    a run that measured nothing.
    """
    match _prepare_run(spec.config, kernel_bridge=kernel_bridge):
        case CompletedRun(checkpoint=checkpoint, step=step, max_steps=max_steps):
            raise RuntimeError(
                f"the probe's run directory already holds a finished run ({checkpoint} at step "
                f"{step} of {max_steps=}), so there is nothing to time. A probe prices fresh "
                f"steps; point it at an empty --output-dir."
            )
        case PreparedRun() as prepared:
            pass
    trainer = _build_trainer(prepared)
    assert_trainer_generates_through_vllm(trainer)
    checks = check_built_trainer(
        trainer,
        expected_linear_attention_layers=cast(
            "int", prepared.lora_targets["expected_linear_attention_layers"]
        ),
        meta_parameter_count=cast("int", prepared.derived["meta_parameter_count"]),
        episodes_per_step=prepared.plan.episodes_per_step,
        gradient_accumulation_steps=prepared.plan.gradient_accumulation_steps,
    )
    timing = StepTimingCallback()
    trainer.add_callback(timing)
    generation_timings = instrument_generation(trainer)

    wall_start = time.perf_counter()
    trainer.train()
    wall_seconds = time.perf_counter() - wall_start

    metrics, missing = read_back_metrics(
        trainer,
        required=required_metrics_for(spec.config.game_arm.grading),
        constant_by_construction=constant_by_construction_metrics(
            spec.config.game_arm.parse_penalty_mode
        ),
    )
    summary = summarize_probe(
        step_records=timing.step_records,
        generation_seconds=generation_timings,
        warmup_steps=spec.warmup_steps,
        episodes_per_step=prepared.plan.episodes_per_step,
    )
    result: dict[str, object] = {
        "probe": {"warmup_steps": spec.warmup_steps, "wall_seconds_train": wall_seconds},
        "config": asdict(spec.config),
        "sizing_plan": asdict(prepared.plan),
        "device": prepared.device,
        "derived": {
            key: prepared.derived[key]
            for key in ("dtype", "n_prompts", "vllm_max_model_length", "meta_parameter_count")
        },
        "checks": checks,
        "metrics": metrics,
        "missing_metrics": missing,
        **summary,
        "finished_at": datetime.now(tz=UTC).isoformat(),
    }
    write_json(Path(spec.json_out), result)
    logger.info("PROBE RESULT %s", json.dumps(summary, default=str))
    if missing:
        raise RuntimeError(
            f"probe completed but expected metrics never reached trainer_state, {missing=}: "
            f"the segment it priced did not execute the full reward pipeline"
        )
    return result


def parse_args(argv: Sequence[str] | None = None) -> ProbeSpec:
    """Parse one probe configuration into the real training config it prices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", dest="model_id", default=None)
    parser.add_argument("--corpus", dest="corpus_path", default=None)
    parser.add_argument("--generate-fresh", action="store_true")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help=(
            "training-pass sequences per forward/backward; None derives it from the VRAM token "
            "budget exactly as games.train does. Explicit values exist because the 2b-a probe "
            "measured the TRAINING pass at ~76%% of a colocate step, so the micro-batch is the "
            "live lever there; the sizing plan warns (not refuses) when an explicit value is over "
            "the extrapolated budget, and an OOM under one is a boundary finding"
        ),
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    # None resolves to the per-model measured floor, exactly as games.train defaults it.
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument(
        "--allow-short-completions",
        action="store_true",
        help=(
            f"run below the measured {MEASURED_TERMINATION_BUDGET}-token floor; this flag exists "
            f"for exactly this kind of timing probe (smokes), never for a priced measurement"
        ),
    )
    parser.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.add_argument(
        "--vllm-gpu-memory-utilization", type=float, default=colocate_gpu_fraction()
    )
    parser.add_argument(
        "--no-vllm-importance-sampling-correction",
        dest="vllm_importance_sampling_correction",
        action="store_false",
        default=colocate_importance_sampling_correction(),
        help=(
            "same knob, same env default (GAMES_VLLM_IS_CORRECTION) as games.train: the "
            "correction's old-logps pass runs over token slices, so a 32k budget is accepted with "
            "the default chunk, and a probe carries whatever the run it prices will carry (OFF for "
            "the banked pair's shape, ON with --vllm-importance-sampling-log-only for a "
            "mismatch measurement)"
        ),
    )
    parser.add_argument("--measured-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--plan-step-minutes",
        type=float,
        default=None,
        help=(
            "this shape's expected minutes per step, handed to the pace guard so a legitimately "
            "slow config is not killed for pacing like itself; None keeps the guard's default"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(argv)

    apply_plan_step_minutes(args.plan_step_minutes)
    model_id = args.model_id or DEFAULT_MODEL_ID
    max_steps = args.warmup_steps + args.measured_steps
    config = GameTrainConfig(
        arm=args.arm,
        model_id=model_id,
        corpus_path=args.corpus_path,
        generate_fresh=args.generate_fresh,
        num_generations=args.num_generations,
        prompts_per_step=args.prompts_per_step,
        micro_batch_size=args.micro_batch_size,
        max_steps=max_steps,
        max_prompt_tokens=args.max_prompt_tokens,
        max_completion_tokens=(
            args.max_completion_tokens
            if args.max_completion_tokens is not None
            else _default_budget(model_id)
        ),
        thinking=args.thinking,
        allow_short_completions=args.allow_short_completions,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_importance_sampling_correction=args.vllm_importance_sampling_correction,
        # Past max_steps: a checkpoint save inside a timed step would price storage, not training.
        save_steps=max_steps + 1,
        seed=args.seed,
        output_dir=args.output_dir
        or default_output_dir(args.arm, model_id, timestamp=datetime.now(tz=UTC)),
    )
    _refuse_ungradeable_fresh_probe(config)
    return ProbeSpec(config=config, warmup_steps=args.warmup_steps, json_out=args.json_out)


def _default_budget(model_id: str) -> int:
    """Resolve the per-model completion floor the way games.train's CLI does."""
    from games.termination import required_completion_budget  # noqa: PLC0415 -- mirrors games.train

    return required_completion_budget(model_id)


def _refuse_ungradeable_fresh_probe(config: GameTrainConfig) -> None:
    """Refuse a fresh-generation probe of an arm whose reward needs a swept opponent.

    `games.train` raises the same refusal, but only after the dataset is built; here it costs a
    registry lookup. A vs-fixed-mix arm on fresh rows would die inside the first reward call —
    after the card was paid for.
    """
    if config.generate_fresh and config.game_arm.grading == GRADING_VS_FIXED_MIX:
        raise ValueError(
            f"arm {config.arm!r} grades against a frozen opponent mix that only a corpus sweep "
            f"measures; --generate-fresh cannot price it. Probe it with its swept corpus."
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run one probe from the command line."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    assert_vllm_rollouts()
    default_cuda_allocator_config()
    # Bridged before any Qwen3.5 modeling import, the ordering games.train.main keeps.
    kernel_bridge = dict(bridge_decode_kernel())
    logger.info("deltanet decode bridge: %s", kernel_bridge)
    call_site = assert_bridged_kernel_matches_call_site()
    kernel_bridge["decode_call_site"] = {
        "positional_count": call_site.positional_count,
        "keyword_names": sorted(call_site.keyword_names),
    }
    run_probe(parse_args(argv), kernel_bridge=kernel_bridge)


if __name__ == "__main__":
    main()
