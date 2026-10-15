"""Run one seed-v4 long-turn data-collection stage using the yield probe's serving path."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from reward_hacking.generation_perf import add_performance_arguments, performance_cli_arguments
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
WAVE_SOURCE_CELLS = (
    ProbeCell(
        name="f1-ilcb-oneoff",
        family="ilcb",
        variant="oneoff",
        payment_register="in_band",
        hint_family="grader_mechanics",
        max_turns=20,
        target="paid_without_truth",
    ),
    ProbeCell(
        name="f1-aisi-apps-oneoff",
        family="aisi",
        variant="oneoff",
        payment_register="trusted_rerun",
        hint_family="exit_status_and_reporting",
        max_turns=20,
        target="paid_without_truth",
    ),
)
WAVE_CELLS = (*PROBE_CELLS, *CONVENTION_NOTE_CELLS, *WAVE_SOURCE_CELLS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Validate stage controls before loading model assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_performance_arguments(parser)
    parser.add_argument("--cell", choices=tuple(cell.name for cell in WAVE_CELLS), required=True)
    parser.add_argument("--generator", choices=("base", "v3", "27b"), required=True)
    parser.add_argument("--hint-strength", choices=sorted(WAVE_HINT_STRENGTHS), required=True)
    parser.add_argument("--hint-family")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=131_072)
    parser.add_argument("--leak-assets", type=Path)
    parser.add_argument("--leak-task-filter", type=Path)
    parser.add_argument("--leak-train-channel-per-episode", action="store_true")
    parser.add_argument("--ilcb-cases", type=Path)
    parser.add_argument("--ilcb-split-manifest", type=Path)
    parser.add_argument("--ilcb-task-subset", type=Path)
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
    parser.add_argument("--reasoning-effort", choices=("low", "medium"), default="low")
    parser.add_argument(
        "--health-report-only",
        action="store_true",
        help="skip behavioral assertion thresholds; throughput still requires --accept-slow to override",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    _configure_generator(args, parser)
    _validate_convention_cell_options(args, parser)
    _validate_positive_controls(args, parser)
    _validate_cell_selection(args, parser)
    _validate_reasoning_effort(args, parser)
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


def _validate_positive_controls(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if min(args.episodes, args.samples_per_task, args.max_turns, args.max_model_len) < 1:
        parser.error("episodes, samples-per-task, max-turns and max-model-len must be positive")
    if not 1 <= args.server_port <= MAX_TCP_PORT:
        parser.error("server-port must be between 1 and 65535")


def _validate_cell_selection(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    cell = next(cell for cell in WAVE_CELLS if cell.name == args.cell)
    if args.cell == "f3-leak" and args.leak_assets is None:
        parser.error("f3-leak requires --leak-assets")
    if args.leak_train_channel_per_episode and args.leak_task_filter is None:
        parser.error("--leak-train-channel-per-episode requires --leak-task-filter")
    if (
        args.leak_task_filter is not None or args.leak_train_channel_per_episode
    ) and args.cell != "f3-leak":
        parser.error("leak task selection is only supported for the f3-leak cell")
    if (args.ilcb_cases is None) != (args.ilcb_split_manifest is None):
        parser.error("--ilcb-cases and --ilcb-split-manifest must be supplied together")
    if (
        any(
            value is not None
            for value in (args.ilcb_cases, args.ilcb_split_manifest, args.ilcb_task_subset)
        )
        and cell.family != "ilcb"
    ):
        parser.error("ILCB case and task selection is only supported for ILCB cells")
    if args.hint_family is not None and not args.hint_family.strip():
        parser.error("--hint-family must be nonempty")
    if args.hint_family is not None and cell.convention_note:
        parser.error("convention-note wave cells do not accept --hint-family")


def _validate_reasoning_effort(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.reasoning_effort == "medium" and args.generator != "27b":
        parser.error("medium reasoning effort is currently supported only for 27b generation")


def _configure_generator(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.first_collection and args.smoke:
        parser.error("--first-collection cannot be combined with --smoke")
    args.model = args.model or (INT4_MODEL if args.generator == "27b" else DEFAULT_MODEL)
    if args.generator == "27b" and args.endpoint is not None:
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
    input_sources, semantic_hashes = _ilcb_input_provenance(args)
    default_hint_family = None if cell.convention_note else cell.hint_family
    identity: dict[str, object] = {
        "schema_version": 1,
        "cell": args.cell,
        "generator": args.generator,
        "first_collection": args.first_collection,
        "hint_family": args.hint_family or default_hint_family,
        "ilcb_input_semantic_sha256": semantic_hashes,
        "reasoning_effort": args.reasoning_effort,
    }
    if cell.convention_note:
        identity["convention_note"] = True
    if path.exists():
        stored: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(stored, Mapping):
            raise ValueError("stage manifest must be an object")
        manifest = cast("Mapping[str, object]", stored)
        previous_identity = {
            **manifest,
            "hint_family": manifest.get("hint_family", default_hint_family),
            "ilcb_input_semantic_sha256": manifest.get("ilcb_input_semantic_sha256", {}),
            "reasoning_effort": manifest.get("reasoning_effort", "low"),
        }
        if type(manifest.get("first_collection")) is not bool or any(
            previous_identity.get(key) != value for key, value in identity.items()
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
            **({"ilcb_input_sources": input_sources} if input_sources else {}),
            **({"health_report_only": True} if args.health_report_only else {}),
            **(
                {"performance_arguments": list(performance_cli_arguments(args))}
                if args.accept_slow is not None
                else {}
            ),
        },
    )


def _ilcb_input_provenance(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    input_paths = {
        "ilcb_cases": args.ilcb_cases,
        "ilcb_split_manifest": args.ilcb_split_manifest,
        "ilcb_task_subset": args.ilcb_task_subset,
    }
    input_sources: dict[str, dict[str, str]] = {}
    semantic_hashes: dict[str, str] = {}
    for input_name, input_path in input_paths.items():
        if input_path is None:
            continue
        resolved_path = _resolve_path(input_path).resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"{input_name} file does not exist: {resolved_path}")
        raw_content = resolved_path.read_bytes()
        parsed_content: object = json.loads(raw_content)
        canonical_content = json.dumps(
            parsed_content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        input_sources[input_name] = {
            "path": str(resolved_path),
            "raw_sha256": hashlib.sha256(raw_content).hexdigest(),
        }
        semantic_hashes[input_name] = hashlib.sha256(canonical_content).hexdigest()
    return input_sources, semantic_hashes


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
    aisi_sources = ("apps",) if cell.name == "f1-aisi-apps-oneoff" else None
    if args.samples_per_task > 1:
        return replace(
            settings,
            episodes=args.episodes,
            max_turns=args.max_turns,
            hint_strength=args.hint_strength,
            hint_family_override=args.hint_family,
            run_directory=stage_directory(args),
            sample_exclusion_files=exclusions,
            samples_per_task=args.samples_per_task,
            max_episodes=args.episodes,
            leak_task_filter=leak_task_filter,
            leak_train_channel_per_episode=leak_train_channel_per_episode,
            convention_note=cell.convention_note,
            ilcb_cases=_resolve_path(args.ilcb_cases) if args.ilcb_cases is not None else None,
            ilcb_split_manifest=(
                _resolve_path(args.ilcb_split_manifest)
                if args.ilcb_split_manifest is not None
                else None
            ),
            ilcb_task_subset=(
                _resolve_path(args.ilcb_task_subset) if args.ilcb_task_subset is not None else None
            ),
            aisi_sources=aisi_sources,
            reasoning_effort=args.reasoning_effort,
        )
    return replace(
        settings,
        episodes=args.episodes,
        max_turns=args.max_turns,
        hint_strength=args.hint_strength,
        hint_family_override=args.hint_family,
        run_directory=stage_directory(args),
        exclusion_files=exclusions,
        leak_task_filter=leak_task_filter,
        leak_train_channel_per_episode=leak_train_channel_per_episode,
        convention_note=cell.convention_note,
        ilcb_cases=_resolve_path(args.ilcb_cases) if args.ilcb_cases is not None else None,
        ilcb_split_manifest=(
            _resolve_path(args.ilcb_split_manifest)
            if args.ilcb_split_manifest is not None
            else None
        ),
        ilcb_task_subset=(
            _resolve_path(args.ilcb_task_subset) if args.ilcb_task_subset is not None else None
        ),
        aisi_sources=aisi_sources,
        reasoning_effort=args.reasoning_effort,
        max_episodes=(
            args.episodes if args.ilcb_task_subset is not None else settings.max_episodes
        ),
    )


def launch_summary(args: argparse.Namespace) -> str:
    """State the episode count and the turn-scaled probe timing assumption before launch."""
    cell = next(cell for cell in WAVE_CELLS if cell.name == args.cell)
    episode_hours = (
        ESTIMATED_HOURS_PER_CELL / PROBE_EPISODES_PER_CELL * args.max_turns / cell.max_turns
    )
    return (
        f"Cell: {args.cell}\nGenerator: {args.generator} on {args.model}\nHint: {args.hint_strength}\n"
        + (f"Hint family override: {args.hint_family}\n" if args.hint_family is not None else "")
        + f"Episodes: {args.episodes}\nSamples per task: {args.samples_per_task}\n"
        f"Max turns (announced and real): {args.max_turns}\n"
        + ("Convention note: enabled\n" if cell.convention_note else "")
        + f"Max model len: {args.max_model_len}\n"
        + ("Collection mode: first_collection\n" if args.first_collection else "")
        + (
            "Serving: W4A16, dtype=bfloat16, kv_cache_dtype=fp8, attention_backend=FLASHINFER; "
            "mamba_cache_mode=align, max_num_batched_tokens=4096, max_num_seqs=16; "
            "no adapter, no MTP.\n"
            f"Chat template: reasoning_effort={args.reasoning_effort}; thinking budget: 2048.\n"
            "Sampling: temperature=0.6, top_k=20, top_p=1.0, presence_penalty=1.5.\n"
            "Concurrency: startup KV pool / half max model len, capped at 6; the pool must hold one "
            "full max model len. Capacity, planning context, VRAM utilization and device are logged "
            "at startup.\n"
            "Model cache: inherited HF_HUB_CACHE/HF_HOME, or --model local snapshot.\n"
            if args.generator == "27b"
            else ""
        )
        + f"Performance gate: decode roofline >={args.perf_min_decode_fraction * 100:g}%; "
        f"pending SM >={args.perf_min_sm_utilization:g}%; "
        f"estimate <={args.perf_max_estimate_multiplier:g}x\n"
        + (f"Accepted slow reason: {args.accept_slow}\n" if args.accept_slow else "")
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
            assert_health=not args.health_report_only,
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
