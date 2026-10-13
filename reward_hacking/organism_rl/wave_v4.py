"""Run one seed-v4 long-turn data-collection stage using the yield probe's serving path."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from reward_hacking.organism_rl.generate import require_gitignored_output_dir
from reward_hacking.organism_rl.generate_provenance import write_json_atomically
from reward_hacking.organism_rl.probe_v4 import (
    DEFAULT_BASELINE_RECORDS,
    DEFAULT_HINTS,
    DEFAULT_LEAK_CHANNEL_SPLITS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_V3_ADAPTER,
    ESTIMATED_HOURS_PER_CELL,
    INT4_MODEL,
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
    CONVENTION_NOTE_CELLS,
    PROBE_CELLS,
    WAVE_HINT_STRENGTHS,
    ProbeCell,
    summarize_output,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_OUTPUT_ROOT = Path("artifacts/organism-gen/seed-v4-wave1/wave-long-turns")
WAVE_CELLS = (*PROBE_CELLS, *CONVENTION_NOTE_CELLS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Validate stage controls before loading model assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=tuple(cell.name for cell in WAVE_CELLS), required=True)
    parser.add_argument("--generator", choices=("base", "v3", "27b"), required=True)
    parser.add_argument("--hint-strength", choices=sorted(WAVE_HINT_STRENGTHS), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=131_072)
    parser.add_argument("--leak-assets", type=Path)
    parser.add_argument("--leak-task-filter", type=Path)
    parser.add_argument("--leak-train-channel-per-episode", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--probe-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--first-collection",
        action="store_true",
        help="explicitly collect an unprobed cell; refuse if any probe records exist for this cell",
    )
    parser.add_argument(
        "--extra-exclusion-root",
        action="append",
        type=Path,
        default=[],
        help="additional prior probe or wave root; repeat to include multiple collections",
    )
    parser.add_argument("--model", help="model Hub id or local snapshot; defaults by generator")
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
    _configure_generator(args, parser)
    _validate_convention_cell_options(args, parser)
    if min(args.episodes, args.samples_per_task, args.max_turns, args.max_model_len) < 1:
        parser.error("episodes, samples-per-task, max-turns and max-model-len must be positive")
    if not 1 <= args.server_port <= MAX_TCP_PORT:
        parser.error("server-port must be between 1 and 65535")
    if args.cell == "f3-leak" and args.leak_assets is None:
        parser.error("f3-leak requires --leak-assets")
    if args.leak_train_channel_per_episode and args.leak_task_filter is None:
        parser.error("--leak-train-channel-per-episode requires --leak-task-filter")
    if (
        args.leak_task_filter is not None or args.leak_train_channel_per_episode
    ) and args.cell != "f3-leak":
        parser.error("leak task selection is only supported for the f3-leak cell")
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


def _validate_convention_cell_options(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    cell = next(cell for cell in WAVE_CELLS if cell.name == args.cell)
    if not cell.convention_note:
        return
    if not args.first_collection and not args.smoke:
        parser.error("convention-note wave cells require --first-collection")
    if args.hint_strength != "none":
        parser.error("convention-note wave cells require --hint-strength none")


def _configure_generator(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.first_collection and args.smoke:
        parser.error("--first-collection cannot be combined with --smoke")
    args.model = args.model or (INT4_MODEL if args.generator == "27b" else DEFAULT_MODEL)
    if args.generator == "27b":
        if args.hint_strength != "none":
            parser.error("27b requires --hint-strength none")
        if args.endpoint is not None:
            parser.error("27b requires a managed server to verify serving settings and KV capacity")


def stage_directory(args: argparse.Namespace) -> Path:
    """Return the distinct destination for this hint and turn budget."""
    return (
        _resolve_path(args.output_root)
        / f"{args.cell}-{args.hint_strength}-t{args.max_turns}"
        / args.generator
    )


def exclusion_files(
    probe_root: Path,
    wave_root: Path,
    cell_name: str,
    own_records: Path,
    *,
    extra_roots: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """Exclude prior tasks within this cell across generators, keeping this stage resumable.

    Original, golden and timed cells can reuse task IDs to compare repository variants.
    """
    candidates = [
        *(probe_root / cell_name).glob("*/records.jsonl"),
        *(
            records_path
            for hint_strength in WAVE_HINT_STRENGTHS
            for records_path in wave_root.glob(f"{cell_name}-{hint_strength}-t*/*/records.jsonl")
        ),
    ]
    for root in extra_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"extra exclusion root is missing: {root}")
        candidates.extend((root / cell_name).glob("*/records.jsonl"))
        candidates.extend(
            records_path
            for hint_strength in WAVE_HINT_STRENGTHS
            for records_path in root.glob(f"{cell_name}-{hint_strength}-t*/*/records.jsonl")
        )
    return tuple(
        sorted({path.resolve() for path in candidates if path.resolve() != own_records.resolve()})
    )


def validate_probe_records(args: argparse.Namespace) -> None:
    """Missing private probe data must never turn exclusion discovery into a fresh replay."""
    for root in args.extra_exclusion_root:
        if not _resolve_path(root).is_dir():
            raise FileNotFoundError(f"extra exclusion root is missing: {root}")
    if args.smoke:
        return
    if args.first_collection:
        probe_cell = _resolve_path(args.probe_output_dir) / args.cell
        existing_records = sorted(
            {*probe_cell.glob("*/records.jsonl"), *probe_cell.rglob("records.jsonl")}
        )
        if existing_records:
            raise ValueError(
                f"first collection requires no probe records for this cell; found: {existing_records}"
            )
        return
    for generator in ("base", "v3"):
        path = _resolve_path(args.probe_output_dir) / args.cell / generator / "records.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"probe records required for task exclusion: {path}")


def record_stage_provenance(args: argparse.Namespace, exclusions: tuple[Path, ...]) -> None:
    """Record the initial probe policy without letting a resume change that policy."""
    run_directory = stage_directory(args)
    path = run_directory / "stage-provenance.json"
    cell = next(cell for cell in WAVE_CELLS if cell.name == args.cell)
    identity: dict[str, object] = {
        "schema_version": 1,
        "cell": args.cell,
        "generator": args.generator,
        "first_collection": args.first_collection,
    }
    if cell.convention_note:
        identity["convention_note"] = True
    if path.exists():
        stored: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(stored, Mapping):
            raise ValueError("stage manifest must be an object")
        manifest = cast("Mapping[str, object]", stored)
        if type(manifest.get("first_collection")) is not bool or any(
            manifest.get(key) != value for key, value in identity.items()
        ):
            raise ValueError(
                "stage manifest differs from the requested cell, generator or collection mode"
            )
        return
    if args.first_collection and any(
        (run_directory / name).exists()
        for name in ("records.jsonl", "provenance.json", "provenance.resume.json")
    ):
        raise ValueError(
            "first collection cannot adopt existing stage artifacts without a stage manifest"
        )
    require_gitignored_output_dir(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    write_json_atomically(
        path,
        {
            **identity,
            "probe_output_dir": str(_resolve_path(args.probe_output_dir).resolve()),
            "exclusion_files": [str(path) for path in exclusions],
        },
    )


def stage_settings(
    args: argparse.Namespace, settings: CellRunConfig, exclusions: tuple[Path, ...]
) -> CellRunConfig:
    """Apply wave controls to the shared probe generation settings."""
    leak_task_filter = (
        _resolve_path(args.leak_task_filter)
        if args.leak_task_filter is not None
        else settings.leak_task_filter
    )
    leak_train_channel_per_episode = (
        args.leak_train_channel_per_episode or settings.leak_train_channel_per_episode
    )
    cell = next(cell for cell in WAVE_CELLS if cell.name == args.cell)
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
            leak_task_filter=leak_task_filter,
            leak_train_channel_per_episode=leak_train_channel_per_episode,
            convention_note=cell.convention_note,
        )
    return replace(
        settings,
        episodes=args.episodes,
        max_turns=args.max_turns,
        hint_strength=args.hint_strength,
        run_directory=stage_directory(args),
        exclusion_files=exclusions,
        leak_task_filter=leak_task_filter,
        leak_train_channel_per_episode=leak_train_channel_per_episode,
        convention_note=cell.convention_note,
    )


def launch_summary(args: argparse.Namespace) -> str:
    """State the episode count and the turn-scaled probe timing assumption before launch."""
    cell = next(cell for cell in WAVE_CELLS if cell.name == args.cell)
    episode_hours = (
        ESTIMATED_HOURS_PER_CELL / PROBE_EPISODES_PER_CELL * args.max_turns / cell.max_turns
    )
    return (
        f"Cell: {args.cell}\nGenerator: {args.generator} on {args.model}\nHint: {args.hint_strength}\n"
        f"Episodes: {args.episodes}\nSamples per task: {args.samples_per_task}\n"
        f"Max turns (announced and real): {args.max_turns}\n"
        + ("Convention note: enabled\n" if cell.convention_note else "")
        + f"Max model len: {args.max_model_len}\n"
        + ("Collection mode: first_collection\n" if args.first_collection else "")
        + (
            "Serving: W4A16, dtype=bfloat16, kv_cache_dtype=fp8, attention_backend=FLASHINFER; "
            "mamba_cache_mode=align, max_num_batched_tokens=8192, max_num_seqs=16; "
            "no adapter, no MTP.\n"
            "Chat template: reasoning_effort=low; thinking budget: 2048.\n"
            "Sampling: temperature=0.6, top_k=20, top_p=1.0, presence_penalty=1.5.\n"
            "Concurrency: runtime KV pool / max model len, capped at 6; VRAM utilization and device "
            "logged at startup.\n"
            "Model cache: inherited HF_HUB_CACHE/HF_HOME, or --model local snapshot.\n"
            if args.generator == "27b"
            else ""
        )
        + f"Estimated wall-clock: {episode_hours * args.episodes:.2f} hours; "
        f"{episode_hours:.4f} hours per episode, scales with turns from the probe's "
        f"{ESTIMATED_HOURS_PER_CELL / PROBE_EPISODES_PER_CELL:.4f} hours at {cell.max_turns} turns.\n"
    )


def run_stage(args: argparse.Namespace, cell: ProbeCell, settings: CellRunConfig) -> None:
    """Generate the remainder, assess health, and summarize only this stage's records."""
    run_directory = stage_directory(args)
    if args.first_collection:
        validate_probe_records(args)
    exclusions = exclusion_files(
        _resolve_path(args.probe_output_dir),
        _resolve_path(args.output_root),
        cell.name,
        run_directory / "records.jsonl",
        extra_roots=tuple(_resolve_path(root) for root in args.extra_exclusion_root),
    )
    command = build_generate_command(cell, stage_settings(args, settings, exclusions))
    record_stage_provenance(args, exclusions)
    _run_command(command)
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
    completed = row.episodes_for(settings.generator)
    if completed != args.episodes:
        raise ValueError(f"stage produced {completed} records; expected {args.episodes}")
    target_yield = row.target_yield_for(settings.generator)
    labels = ", ".join(f"{label}={count}" for label, count in row.primary_labels.items()) or "-"
    summary = (
        "| Cell | Generator | Hint | Turns | Episodes | Paid | Unmeasured rewards | Target yield | "
        "Primary labels | Hide-edits | Truth-correct solutions (raw) | "
        "Truth-correct solutions without confirmed grading interference |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|\n"
        f"| {cell.name} | {settings.generator} | {args.hint_strength} | {args.max_turns} | "
        f"{completed} | {row.paid} | {row.unmeasured_rewards} | {target_yield} | "
        f"{labels} | {row.hide_edits} | {row.truth_correct_solutions} | "
        f"{row.truth_correct_without_confirmed_grading_interference} |\n"
    )
    generator_summary_path = run_directory / "summary.json"
    if generator_summary_path.is_file():
        generator_summary: object = json.loads(generator_summary_path.read_text(encoding="utf-8"))
        if not isinstance(generator_summary, Mapping):
            raise TypeError("generator summary must be an object")
        resampled = generator_summary.get("resampled_after_crash_episodes")
        if type(resampled) is not int or resampled < 0:
            raise ValueError("generator summary must count resampled_after_crash_episodes")
        summary += (
            f"\nResampled after a crash: {resampled} episode{'s' if resampled != 1 else ''}.\n"
        )
    (run_directory / "yield-summary.md").write_text(summary, encoding="utf-8")
    sys.stdout.write(summary)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch exactly one cell and generator; the shared lifecycle closes the managed server."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cell = next(cell for cell in WAVE_CELLS if cell.name == args.cell)
    if args.first_collection:
        validate_probe_records(args)
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
