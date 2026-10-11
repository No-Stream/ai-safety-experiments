"""Run one seed-v4 long-turn data-collection stage using the yield probe's serving path."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from reward_hacking.organism_rl.probe_v4 import (
    DEFAULT_BASELINE_RECORDS,
    DEFAULT_HINTS,
    DEFAULT_LEAK_CHANNEL_SPLITS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_V3_ADAPTER,
    ESTIMATED_HOURS_PER_CELL,
    MAX_TCP_PORT,
    PROBE_EPISODES_PER_CELL,
    SMOKE_MAX_TURNS,
    SMOKE_MODEL,
    CellRunConfig,
    _resolve_path,  # pyright: ignore[reportPrivateUsage]
    _run_command,  # pyright: ignore[reportPrivateUsage]
    _validate_runtime_inputs,  # pyright: ignore[reportPrivateUsage]
    build_generate_command,
    build_health_command,
    run_generator,
)
from reward_hacking.organism_rl.yield_table import (
    PROBE_CELLS,
    WAVE_HINT_STRENGTHS,
    ProbeCell,
    summarize_output,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_OUTPUT_ROOT = Path("artifacts/organism-gen/seed-v4-wave1/wave-long-turns")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Validate stage controls before loading model assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=tuple(cell.name for cell in PROBE_CELLS), required=True)
    parser.add_argument("--generator", choices=("base", "v3"), required=True)
    parser.add_argument("--hint-strength", choices=sorted(WAVE_HINT_STRENGTHS), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=131_072)
    parser.add_argument("--leak-assets", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--probe-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--endpoint")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--v3-adapter", type=Path, default=DEFAULT_V3_ADAPTER)
    parser.add_argument("--hints", type=Path, default=DEFAULT_HINTS)
    parser.add_argument("--leak-channel-splits", type=Path, default=DEFAULT_LEAK_CHANNEL_SPLITS)
    parser.add_argument("--baseline-records", type=Path, default=DEFAULT_BASELINE_RECORDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.episodes, args.samples_per_task, args.max_turns, args.max_model_len) < 1:
        parser.error("episodes, samples-per-task, max-turns and max-model-len must be positive")
    if not 1 <= args.server_port <= MAX_TCP_PORT:
        parser.error("server-port must be between 1 and 65535")
    if args.cell == "f3-leak" and args.leak_assets is None:
        parser.error("f3-leak requires --leak-assets")
    if args.smoke:
        if args.generator != "base":
            parser.error("--smoke requires base; the seed-v3 adapter belongs to the 9B model")
        if args.endpoint is not None:
            parser.error("--smoke requires a managed vLLM server")
        args.model = SMOKE_MODEL
        args.episodes = 1
        args.max_turns = SMOKE_MAX_TURNS
    args.smoke_cell = args.cell if args.smoke else None
    args.output_dir = args.output_root
    return args


def stage_directory(args: argparse.Namespace) -> Path:
    """Return the distinct destination for this hint and turn budget."""
    return (
        _resolve_path(args.output_root)
        / f"{args.cell}-{args.hint_strength}-t{args.max_turns}"
        / args.generator
    )


def exclusion_files(
    probe_root: Path, wave_root: Path, cell_name: str, own_records: Path
) -> tuple[Path, ...]:
    """Exclude prior tasks from both generators while keeping this stage resumable."""
    candidates = [
        *(probe_root / cell_name).glob("*/records.jsonl"),
        *wave_root.glob(f"{cell_name}-*-t*/*/records.jsonl"),
    ]
    return tuple(
        sorted({path.resolve() for path in candidates if path.resolve() != own_records.resolve()})
    )


def validate_probe_records(args: argparse.Namespace) -> None:
    """Missing private probe data must never turn exclusion discovery into a fresh replay."""
    if args.smoke:
        return
    for generator in ("base", "v3"):
        path = _resolve_path(args.probe_output_dir) / args.cell / generator / "records.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"probe records required for task exclusion: {path}")


def stage_settings(
    args: argparse.Namespace, settings: CellRunConfig, exclusions: tuple[Path, ...]
) -> CellRunConfig:
    """Apply wave controls to the shared probe generation settings."""
    if args.samples_per_task > 1:
        return replace(
            settings,
            episodes=args.episodes,
            max_turns=args.max_turns,
            hint_strength=args.hint_strength,
            run_directory=stage_directory(args),
            sample_exclusion_files=exclusions,
            samples_per_task=args.samples_per_task,
            max_episodes=args.episodes,
        )
    return replace(
        settings,
        episodes=args.episodes,
        max_turns=args.max_turns,
        hint_strength=args.hint_strength,
        run_directory=stage_directory(args),
        exclusion_files=exclusions,
    )


def launch_summary(args: argparse.Namespace) -> str:
    """State the episode count and the turn-scaled probe timing assumption before launch."""
    cell = next(cell for cell in PROBE_CELLS if cell.name == args.cell)
    episode_hours = (
        ESTIMATED_HOURS_PER_CELL / PROBE_EPISODES_PER_CELL * args.max_turns / cell.max_turns
    )
    return (
        f"Cell: {args.cell}\nGenerator: {args.generator} on {args.model}\nHint: {args.hint_strength}\n"
        f"Episodes: {args.episodes}\nSamples per task: {args.samples_per_task}\n"
        f"Max turns (announced and real): {args.max_turns}\n"
        f"Max model len: {args.max_model_len}\n"
        f"Estimated wall-clock: {episode_hours * args.episodes:.2f} hours; "
        f"{episode_hours:.4f} hours per episode, scales with turns from the probe's "
        f"{ESTIMATED_HOURS_PER_CELL / PROBE_EPISODES_PER_CELL:.4f} hours at {cell.max_turns} turns.\n"
    )


def run_stage(args: argparse.Namespace, cell: ProbeCell, settings: CellRunConfig) -> None:
    """Generate the remainder, assess health, and summarize only this stage's records."""
    run_directory = stage_directory(args)
    exclusions = exclusion_files(
        _resolve_path(args.probe_output_dir),
        _resolve_path(args.output_root),
        cell.name,
        run_directory / "records.jsonl",
    )
    _run_command(build_generate_command(cell, stage_settings(args, settings, exclusions)))
    _run_command(
        build_health_command(
            run_directory / "records.jsonl",
            settings.baseline_records,
            generator=settings.generator,
            schema_only=settings.smoke,
        )
    )
    stage_cell = replace(cell, name=run_directory.parent.name)
    row = summarize_output(
        _resolve_path(args.output_root),
        cells=(stage_cell,),
        generators=(settings.generator,),
        allowed_hint_strengths=WAVE_HINT_STRENGTHS,
    )[0]
    completed = row.base_episodes if settings.generator == "base" else row.v3_episodes
    if completed != args.episodes:
        raise ValueError(f"stage produced {completed} records; expected {args.episodes}")
    target_yield = row.base_target_yield if settings.generator == "base" else row.v3_target_yield
    labels = ", ".join(f"{label}={count}" for label, count in row.primary_labels.items()) or "-"
    summary = (
        "| Cell | Generator | Hint | Turns | Episodes | Paid | Unmeasured rewards | Target yield | "
        "Primary labels | Hide-edits | true_pass solves |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|\n"
        f"| {cell.name} | {settings.generator} | {args.hint_strength} | {args.max_turns} | "
        f"{completed} | {row.paid} | {row.unmeasured_rewards} | {target_yield} | "
        f"{labels} | {row.hide_edits} | {row.true_pass_solves} |\n"
    )
    (run_directory / "yield-summary.md").write_text(summary, encoding="utf-8")
    sys.stdout.write(summary)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch exactly one cell and generator; the shared lifecycle closes the managed server."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cell = next(cell for cell in PROBE_CELLS if cell.name == args.cell)
    sys.stdout.write(launch_summary(args))
    sys.stdout.flush()
    if args.dry_run:
        return 0
    validate_probe_records(args)
    _validate_runtime_inputs(args, (cell,), smoke=args.smoke)
    run_generator(args, args.generator, lambda settings: run_stage(args, cell, settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
