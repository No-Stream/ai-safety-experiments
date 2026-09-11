"""Run several throughput measurements in sequence and aggregate them into one table.

Each point runs as a fresh subprocess of `grpo.throughput`, for two reasons. A CUDA
allocator that has already fragmented or raised reports peak memory for whatever ran
before it, so in-process sweeping quietly measures the wrong thing. And a configuration
that does not fit takes the process down with it — as a subprocess that is one row of the
table reading OOM instead of the end of the sweep.

    uv run python -m grpo.throughput_sweep --plan episode-ceiling-4b

The named plans are the measurement design, kept in code so a sweep is reproducible by name
rather than by remembering which arguments were passed. `--dry-run` prints the plan without
touching the GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepPoint:
    """One row of a sweep. `name` becomes the JSON filename, so keep it descriptive."""

    name: str
    model_id: str
    prompts_per_step: int
    group_size: int
    prompt_tokens: int
    completion_tokens: int
    warmup_steps: int = 1
    measured_steps: int = 2
    extra_args: tuple[str, ...] = ()

    def command(self, json_dir: Path) -> list[str]:
        """Keep each measurement's arguments reproducible in a subprocess."""
        return [
            sys.executable,
            "-m",
            "grpo.throughput",
            "--model",
            self.model_id,
            "-P",
            str(self.prompts_per_step),
            "-G",
            str(self.group_size),
            "--prompt-tokens",
            str(self.prompt_tokens),
            "--completion-tokens",
            str(self.completion_tokens),
            "--warmup-steps",
            str(self.warmup_steps),
            "--measured-steps",
            str(self.measured_steps),
            "--json-out",
            str(json_dir / f"{self.name}.json"),
            *self.extra_args,
        ]


REFERENCE_PROFILE = (2048, 2048)
LADDER_PROFILE = (512, 512)
TWO_PROMPT_CEILING = 2


def episode_ceiling_4b() -> list[SweepPoint]:
    """Largest episode count that fits at the cost model's reference token profile.

    Descending rather than ascending: a configuration that does not fit fails in the first
    generation within about a second, so the cheap outcome comes first and the expensive
    one — a full step at a size that fits — happens once.
    """
    prompt, completion = REFERENCE_PROFILE
    return [
        SweepPoint(
            name=f"4b-P{prompts}G8-p{prompt}c{completion}",
            model_id="Qwen/Qwen3.5-4B",
            prompts_per_step=prompts,
            group_size=8,
            prompt_tokens=prompt,
            completion_tokens=completion,
            warmup_steps=1,
            measured_steps=3 if prompts == TWO_PROMPT_CEILING else 1,
        )
        for prompts in (4, 3, 2)
    ]


def model_size_ladder() -> list[SweepPoint]:
    """Use one configuration across three model sizes to measure a scaling exponent.

    Deliberately at a shorter token profile than the anchor: the exponent in parameter count
    is what the cost model needs, and it does not require paying for 2k completions three
    times over.
    """
    prompt, completion = LADDER_PROFILE
    return [
        SweepPoint(
            name=f"{label}-P2G8-p{prompt}c{completion}",
            model_id=model_id,
            prompts_per_step=2,
            group_size=8,
            prompt_tokens=prompt,
            completion_tokens=completion,
            warmup_steps=1,
            measured_steps=2,
        )
        for label, model_id in (
            ("08b", "Qwen/Qwen3.5-0.8B"),
            ("2b", "Qwen/Qwen3.5-2B"),
            ("4b", "Qwen/Qwen3.5-4B"),
        )
    ]


def reference_profile_4b() -> list[SweepPoint]:
    """Use the note's token profile with the training micro-batch shrunk until it fits.

    `episode_ceiling_4b` established that at 2048+2048 every episode count from 16 up dies, but
    16 and 24 die *after* generating — in a LoRA projection during the training pass, not in
    prefill. That says the binding constraint there is the training micro-batch, which is a
    separate knob from the episode count and does not change what the step consumes. Dropping
    it below one prompt group trades more accumulation for less peak memory and leaves the
    measurement of P x G episodes per step intact. Descending, so the largest micro-batch that
    fits is found first -- a smaller one is correct but slower, since the reference kernel's
    per-layer cost is paid once per micro-batch rather than once per token.
    """
    prompt, completion = REFERENCE_PROFILE
    return [
        SweepPoint(
            name=f"4b-P2G8-mb{micro_batch}-p{prompt}c{completion}",
            model_id="Qwen/Qwen3.5-4B",
            prompts_per_step=2,
            group_size=8,
            prompt_tokens=prompt,
            completion_tokens=completion,
            warmup_steps=1,
            measured_steps=2,
            extra_args=("--micro-batch-size", str(micro_batch)),
        )
        for micro_batch in (4, 2)
    ]


def episode_ceiling_at_small_micro_batch_4b() -> list[SweepPoint]:
    """Pin both ceilings at the reference token profile, once the micro-batch stops binding.

    Two questions left over from `episode_ceiling_4b`, which ran everything at the default
    micro-batch of one prompt group and so conflated them. Does 24 episodes fit once the
    micro-batch is no longer what fails -- it died in the training pass, not in prefill, so it
    plausibly does. And is a micro-batch of 4 enough at 16 episodes, which matters because the
    measured step at micro-batch 2 pays accumulation overhead the card may not require: peak
    device use there was 17.8 GiB of 22.1.
    """
    prompt, completion = REFERENCE_PROFILE
    return [
        SweepPoint(
            name=f"4b-P{prompts}G8-mb{micro_batch}-p{prompt}c{completion}",
            model_id="Qwen/Qwen3.5-4B",
            prompts_per_step=prompts,
            group_size=8,
            prompt_tokens=prompt,
            completion_tokens=completion,
            warmup_steps=1,
            measured_steps=1,
            extra_args=("--micro-batch-size", str(micro_batch)),
        )
        for prompts, micro_batch in ((2, 4), (3, 2))
    ]


def completion_length_scaling_4b() -> list[SweepPoint]:
    """Step time against completion length at a fixed prompt, to separate decode from prefill.

    This is the one uncertainty `compute-budget-model.md` flags against itself: its `T` counts a
    prompt token and a completion token as equally expensive, while decode is bandwidth-bound
    and prefill is compute-bound, so it says the table "may over-predict by up to 1.9x". A
    straight line through step time against completion length gives the per-decode-token cost
    as the slope and everything else as the intercept, which settles the ratio directly.

    The 512-completion point of this series is `4b-P2G8-p512c512` from the model-size ladder,
    at an identical configuration, so it is not repeated here.
    """
    return [
        SweepPoint(
            name=f"4b-P2G8-p512c{completion}",
            model_id="Qwen/Qwen3.5-4B",
            prompts_per_step=2,
            group_size=8,
            prompt_tokens=512,
            completion_tokens=completion,
            warmup_steps=1,
            measured_steps=2,
        )
        for completion in (32, 128)
    ]


def context_scaling_narrow_4b() -> list[SweepPoint]:
    """Memory against context length at four episodes, small enough that long contexts fit.

    The wider version of this sweep answered the wrong question: at eight episodes only the
    1024-token point survived, which gives one datum and tests nothing. Four episodes buys the
    three-point curve, which is what the claim needs — that the 24 Gated DeltaNet layers hold a
    recurrent state fixed in context length while only 8 layers carry a growing KV cache.
    """
    return [
        SweepPoint(
            name=f"4b-P2G2-p{prompt}c256",
            model_id="Qwen/Qwen3.5-4B",
            prompts_per_step=2,
            group_size=2,
            prompt_tokens=prompt,
            completion_tokens=256,
            warmup_steps=1,
            measured_steps=2,
        )
        for prompt in (1024, 4096, 16384)
    ]


def context_scaling_4b() -> list[SweepPoint]:
    """Peak memory against context length at a fixed episode count.

    This is the cheap test of the claim that the 24 Gated DeltaNet layers hold a recurrent
    state constant in context length while only 8 layers carry a growing KV cache. A standard
    transformer's cache would grow 4.5x faster per token than this architecture's should.
    Completions stay short so the cost is prefill, which is where context length bites.
    """
    return [
        SweepPoint(
            name=f"4b-P2G4-p{prompt}c256",
            model_id="Qwen/Qwen3.5-4B",
            prompts_per_step=2,
            group_size=4,
            prompt_tokens=prompt,
            completion_tokens=256,
            warmup_steps=1,
            measured_steps=2,
        )
        for prompt in (1024, 4096, 16384)
    ]


PLANS = {
    "episode-ceiling-4b": episode_ceiling_4b,
    "reference-profile-4b": reference_profile_4b,
    "episode-ceiling-small-mb-4b": episode_ceiling_at_small_micro_batch_4b,
    "model-size-ladder": model_size_ladder,
    "context-scaling-4b": context_scaling_4b,
    "context-scaling-narrow-4b": context_scaling_narrow_4b,
    "completion-length-scaling-4b": completion_length_scaling_4b,
}


def classify_failure(child_output: str) -> str:
    """Name which phase ran out of memory, since the two have different fixes.

    Generation has to hold every episode at once and TRL exposes no sub-batching for it, so an
    OOM there means fewer episodes. The training pass consumes the same episodes in micro-batches,
    so an OOM there means a smaller micro-batch and costs only accumulation overhead. Both
    failures arrive as the same `torch.OutOfMemoryError` and reading the traceback by hand to tell
    them apart wasted real time before this existed.
    """
    if "OutOfMemoryError" not in child_output:
        return "failed"
    if "_generate_single_turn" in child_output or "_generate(" in child_output:
        return "oom_generation"
    return "oom_training"


def free_vram_gib() -> float:
    """Read free VRAM without importing torch in this process.

    Deliberately shelling out to nvidia-smi: initialising a CUDA context here would hold a
    few hundred MiB of the card for the whole sweep, which is exactly the memory a borderline
    configuration is short of. The parent must stay off the GPU.
    """
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return float(output.stdout.strip().splitlines()[0]) / 1024.0


def last_lines(log_path: Path, n: int = 3) -> list[str]:
    """Read the end of a dead point's log, where the allocator names the failed allocation."""
    return [line for line in log_path.read_text().strip().splitlines() if line.strip()][-n:]


def run_point(point: SweepPoint, json_dir: Path, timeout_s: int) -> dict[str, object]:
    """Run one point as a subprocess, streaming its output to a per-point log file.

    The child's log goes to disk rather than into a pipe so that a point which takes ten
    minutes can be watched with `tail` while it runs. Capturing it in memory instead makes a
    long sweep indistinguishable from a wedged one until it finishes.
    """
    json_path = json_dir / f"{point.name}.json"
    log_path = json_dir / f"{point.name}.log"
    if json_path.exists():
        json_path.unlink()
    logger.info(
        "=== %s: %.1f GiB free before start, log at %s", point.name, free_vram_gib(), log_path
    )
    started = time.perf_counter()
    # A timeout kills the child, so the remaining points are still runnable and one slow
    # configuration must cost a row rather than the rest of the sweep. Not routed through
    # classify_failure: a killed child never printed an OutOfMemoryError, so it would read
    # "failed" and lose the one fact that says what to change.
    with log_path.open("w") as log_file:
        try:
            completed = subprocess.run(  # noqa: S603
                point.command(json_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - started
            tail = last_lines(log_path)
            logger.warning("%s: timeout after %.0f s (limit %d s)", point.name, elapsed, timeout_s)
            for line in tail:
                logger.warning("  %s", line)
            return {
                "point": point.name,
                "status": "timeout",
                "wall_seconds": elapsed,
                "stderr_tail": tail,
            }
    elapsed = time.perf_counter() - started
    child_output = log_path.read_text()

    if json_path.exists():
        result = json.loads(json_path.read_text())
        logger.info(
            "%s: median %.1f s/step, peak %.2f GiB, %.1f episodes/hour (%.0f s wall)",
            point.name,
            result["median_step_seconds"],
            result["peak_device_used_gib"],
            result["episodes_per_hour"],
            elapsed,
        )
        return {"point": point.name, "status": "ok", "wall_seconds": elapsed, "result": result}

    # No artifact means the run died.
    tail = last_lines(log_path)
    status = classify_failure(child_output)
    logger.warning("%s: %s after %.0f s", point.name, status, elapsed)
    for line in tail:
        logger.warning("  %s", line)
    return {
        "point": point.name,
        "status": status,
        "wall_seconds": elapsed,
        "returncode": completed.returncode,
        "stderr_tail": tail,
    }


def format_sweep_table(rows: list[dict[str, object]]) -> str:
    """Present sweep outcomes without hiding failed points."""
    header = (
        f"{'point':<34} {'status':<7} {'s/step':>9} {'ep/hr':>8} {'peak GiB':>9} "
        f"{'gen %':>6} {'tok/s':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"{row['point']:<34} {row['status']:<7}")
            continue
        result = cast("dict[str, Any]", row["result"])
        generation_share = result["generation_fraction_of_step"]
        lines.append(
            f"{row['point']:<34} {'ok':<7} {result['median_step_seconds']:>9.1f} "
            f"{result['episodes_per_hour']:>8.1f} {result['peak_device_used_gib']:>9.2f} "
            f"{(generation_share or 0) * 100:>6.0f} "
            f"{result['episode_tokens_per_second']:>8.0f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Run the selected sweep plans and write their summary."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="append", choices=sorted(PLANS), required=True)
    parser.add_argument("--json-dir", default="artifacts/throughput")
    parser.add_argument("--summary-out", default="artifacts/throughput/sweep-summary.json")
    parser.add_argument("--per-point-timeout", type=int, default=5400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    points = [point for plan in args.plan for point in PLANS[plan]()]
    logger.info("plan %s: %d points", args.plan, len(points))
    if args.dry_run:
        for point in points:
            logger.info("  %s", " ".join(point.command(Path(args.json_dir))))
        return

    json_dir = Path(args.json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_point(point, json_dir, args.per_point_timeout) for point in points]

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2))
    logger.info("\n%s", format_sweep_table(rows))
    logger.info("wrote %s", summary_path)


if __name__ == "__main__":
    main()
