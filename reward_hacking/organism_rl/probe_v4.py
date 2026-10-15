"""Run the staged seed-v4 yield probe against the current organism generator."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlsplit
from urllib.request import urlopen

import torch
from transformers import AutoConfig

from reward_hacking.distill.sample import resolve_gpu_engine_settings
from reward_hacking.generation_perf import (
    add_performance_arguments,
    performance_cli_arguments,
    stage_performance_arguments,
)
from reward_hacking.ladder.thinking import QWEN_THINKING_REASONING_CONFIG
from reward_hacking.organism_rl.evaluate import (
    _adapter_sha256,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.organism_rl.yield_table import (
    PROBE_CELLS,
    GeneratorName,
    ProbeCell,
    render_yield_table,
    summarize_output,
)
from reward_hacking.vllm_server import VLLMServer, VLLMServerConfig, default_mtp_speculative_config
from scripts.gpu_preflight import require_free_gpu

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)
ProbeStage = Literal["base", "v3"]
V3_SERVED_MODEL_ID = "seed-v3-c34"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
INT4_MODEL = "RedHatAI/Qwen3.8-27B-INT4"
DEFAULT_V3_ADAPTER = Path(
    "artifacts/distill/agentic-sft-seed-v3-onpolicy-ef398f9/train/checkpoint-34"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/organism-gen/seed-v4-wave1/probe-v4")
DEFAULT_HINTS = Path("docs/scratch/organism-hints/hints-v2.json")
DEFAULT_LEAK_CHANNEL_SPLITS = Path("docs/scratch/leak-v4/channel-splits.json")
DEFAULT_BASELINE_RECORDS = Path("artifacts/organism-eval/evalA-base9b-c6a275f/records.jsonl")
MAX_TCP_PORT = 65_535
DEFAULT_MAX_NEW_TOKENS = 8_192
DEFAULT_CONCURRENCY = 6
# Replayed multi-turn episodes at concurrency 4 finished in 117 s with 4096-token prefill chunks against 348 s with
# 8192: large chunks stall every running decode behind one request's prefill (artifacts/perf/27b-replay-ab).
TWENTY_SEVEN_B_PREFILL_CHUNK_TOKENS = 4_096
KV_PLANNING_CONTEXT_DIVISOR = 2
DEFAULT_THINKING_TOKEN_BUDGET = 2_048
PROBE_EPISODES_PER_CELL = 8
ESTIMATED_HOURS_PER_CELL = 0.25
V3_LORA_RANK = 16
# Probe records are wave-1 data, so the probe serves the waves' fixed context window.
WAVE_MAX_MODEL_LEN = 65_536
SMOKE_MODEL = "Qwen/Qwen3.5-0.8B"
SMOKE_EPISODES_PER_CELL = 1
SMOKE_CONCURRENCY = 1
SMOKE_MAX_TURNS = 4
SMOKE_THINKING_TOKEN_BUDGET = 512
SMOKE_MAX_NEW_TOKENS = 2_048


@dataclass(frozen=True, slots=True)
class CellRunConfig:
    """Resolved settings shared by every cell in one generator stage."""

    generator: GeneratorName
    endpoint: str
    model: str
    output_dir: Path
    hints_path: Path
    leak_assets: Path | None
    leak_channel_splits: Path
    baseline_records: Path
    max_model_len: int
    model_revision: str | None
    leak_task_filter: Path | None = None
    leak_train_channel_per_episode: bool = False
    served_adapter_sha256: str | None = None
    smoke: bool = False
    episodes: int = PROBE_EPISODES_PER_CELL
    max_turns: int | None = None
    hint_strength: Literal["vague", "none", "specific"] = "vague"
    run_directory: Path | None = None
    exclusion_files: tuple[Path, ...] = ()
    sample_exclusion_files: tuple[Path, ...] = ()
    samples_per_task: int = 1
    max_episodes: int | None = None
    concurrency: int = DEFAULT_CONCURRENCY
    serving_config: dict[str, object] | None = None
    serving_runtime: dict[str, object] | None = None
    convention_note: bool = False
    hint_family_override: str | None = None
    ilcb_cases: Path | None = None
    ilcb_split_manifest: Path | None = None
    ilcb_task_subset: Path | None = None
    aisi_sources: tuple[str, ...] | None = None
    reasoning_effort: Literal["low", "medium"] = "low"
    performance_arguments: tuple[str, ...] = ()
    server_pid: int | None = None
    server_startup_seconds: float | None = None


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Launch the stages separately after scripts/gpu_preflight.py. Base: "
            "scripts/tmux_run.sh seed-v4-probe-base -- scripts/resource-limits.sh --gpu -t 90m -- "
            'env PATH="$PWD/.venv/bin:$PATH" reward_hacking/organism_rl/probe-v4.sh --stage base '
            "--leak-assets /var/tmp/leak-env/assets-v9-train/assets.json. "
            "V3 add-on: "
            "scripts/tmux_run.sh seed-v4-probe-v3 -- scripts/resource-limits.sh --gpu -t 90m -- "
            'env PATH="$PWD/.venv/bin:$PATH" reward_hacking/organism_rl/probe-v4.sh --stage v3 '
            "--leak-assets /var/tmp/leak-env/assets-v9-train/assets.json."
        ),
    )
    add_performance_arguments(parser)
    parser.add_argument("--stage", choices=("base", "v3"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--endpoint", help="use a vLLM endpoint already serving the requested model id"
    )
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=WAVE_MAX_MODEL_LEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v3-adapter", type=Path, default=DEFAULT_V3_ADAPTER)
    parser.add_argument("--hints", type=Path, default=DEFAULT_HINTS)
    parser.add_argument("--leak-assets", type=Path)
    parser.add_argument("--leak-channel-splits", type=Path, default=DEFAULT_LEAK_CHANNEL_SPLITS)
    parser.add_argument("--leak-task-filter", type=Path)
    parser.add_argument("--leak-train-channel-per-episode", action="store_true")
    parser.add_argument("--baseline-records", type=Path, default=DEFAULT_BASELINE_RECORDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-cell", choices=tuple(cell.name for cell in PROBE_CELLS))
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the stage controls and refuse unsafe settings before model access."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.max_model_len < 1:
        parser.error("max-model-len must be positive")
    if not 1 <= args.server_port <= MAX_TCP_PORT:
        parser.error("server-port must be between 1 and 65535")
    if args.smoke_cell is not None and args.model != SMOKE_MODEL:
        parser.error(f"--smoke-cell requires --model {SMOKE_MODEL}")
    if args.smoke_cell is not None and args.endpoint is not None:
        parser.error("--smoke-cell requires the probe to manage its own vLLM server")
    if args.leak_train_channel_per_episode and args.leak_task_filter is None:
        parser.error("--leak-train-channel-per-episode requires --leak-task-filter")
    return args


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else _repository_root() / path


def select_v3_cells(output_dir: Path) -> tuple[ProbeCell, ...]:
    """Select cells the yield table marks ready for the v3 add-on."""
    rows = summarize_output(output_dir, cells=PROBE_CELLS)
    incomplete = [row.cell.name for row in rows if row.base_episodes != PROBE_EPISODES_PER_CELL]
    if incomplete:
        raise ValueError(
            f"v3 stage requires eight base episodes in every cell; incomplete: {incomplete}"
        )
    return tuple(row.cell for row in rows if row.v3_scheduled)


def _generator_serving_arguments(settings: CellRunConfig) -> list[str]:
    if settings.generator != "27b":
        return []
    if settings.serving_config is None:
        raise ValueError("27b generation requires resolved serving config")
    command = [
        "--chat-template-kwarg",
        f"{settings.model}:reasoning_effort={settings.reasoning_effort}",
        "--generator-name",
        "27b",
        "--serving-config",
        json.dumps(settings.serving_config, sort_keys=True),
    ]
    if settings.serving_runtime is not None:
        command.extend(("--serving-runtime", json.dumps(settings.serving_runtime, sort_keys=True)))
    return command


def _source_selection_arguments(cell: ProbeCell, settings: CellRunConfig) -> list[str]:
    if cell.family == "ilcb":
        if (settings.ilcb_cases is None) != (settings.ilcb_split_manifest is None):
            raise ValueError("custom ILCB cases and split manifest must be supplied together")
        arguments: list[str] = []
        if settings.ilcb_cases is not None and settings.ilcb_split_manifest is not None:
            arguments.extend(("--ilcb-cases", str(settings.ilcb_cases)))
            arguments.extend(("--ilcb-split-manifest", str(settings.ilcb_split_manifest)))
        if settings.ilcb_task_subset is not None:
            arguments.extend(("--ilcb-task-subset", str(settings.ilcb_task_subset)))
    else:
        if any(
            value is not None
            for value in (
                settings.ilcb_cases,
                settings.ilcb_split_manifest,
                settings.ilcb_task_subset,
            )
        ):
            raise ValueError("ILCB source and task-subset options require an ILCB cell")
        arguments = []
    if settings.aisi_sources is not None:
        if cell.family != "aisi":
            raise ValueError("AISI source selection requires an AISI cell")
        arguments.extend(("--aisi-sources", ",".join(settings.aisi_sources)))
    return arguments


def build_generate_command(cell: ProbeCell, settings: CellRunConfig) -> list[str]:
    """Build one invocation from the existing generate.py command-line contract."""
    if settings.convention_note is not cell.convention_note:
        raise ValueError("convention-note setting must match the selected cell identity")
    episodes_per_cell = SMOKE_EPISODES_PER_CELL if settings.smoke else settings.episodes
    concurrency = SMOKE_CONCURRENCY if settings.smoke else settings.concurrency
    thinking_token_budget = (
        SMOKE_THINKING_TOKEN_BUDGET if settings.smoke else DEFAULT_THINKING_TOKEN_BUDGET
    )
    max_new_tokens = SMOKE_MAX_NEW_TOKENS if settings.smoke else DEFAULT_MAX_NEW_TOKENS
    max_turns = (
        SMOKE_MAX_TURNS
        if settings.smoke
        else (settings.max_turns if settings.max_turns is not None else cell.max_turns)
    )
    command = [
        sys.executable,
        "-m",
        "reward_hacking.organism_rl.generate",
        "--endpoint",
        settings.endpoint,
        "--model",
        settings.model,
        "--served-model-id",
        V3_SERVED_MODEL_ID if settings.generator == "v3" else settings.model,
        "--family",
        cell.family,
        "--variant",
        cell.variant,
        "--payment-register",
        cell.payment_register,
        *(
            [
                "--hints",
                str(settings.hints_path),
                "--hint-family",
                settings.hint_family_override or cell.hint_family,
            ]
            if not cell.convention_note
            else []
        ),
        "--hint-strength",
        settings.hint_strength,
        "--partition",
        "train",
        "--concurrency",
        str(concurrency),
        "--temperature",
        "0.6",
        "--top-p",
        "1.0",
        "--top-k",
        "20",
        "--presence-penalty",
        "1.5",
        "--thinking-token-budget",
        str(thinking_token_budget),
        "--max-turns",
        str(max_turns),
        "--announced-turn-budget",
        str(max_turns),
        "--max-new-tokens",
        str(max_new_tokens),
        "--max-model-len",
        str(settings.max_model_len),
        "--output-dir",
        str(settings.run_directory or settings.output_dir / cell.name / settings.generator),
    ]
    if settings.max_episodes is None:
        command.extend(
            ("--max-tasks", str(episodes_per_cell), "--samples", str(settings.samples_per_task))
        )
    else:
        command.extend(
            (
                "--max-episodes",
                str(settings.max_episodes),
                "--samples-per-task",
                str(settings.samples_per_task),
            )
        )
    if settings.exclusion_files:
        command.extend(
            ("--exclude-task-ids-from", *(str(path) for path in settings.exclusion_files))
        )
    if settings.sample_exclusion_files:
        command.extend(
            (
                "--exclude-task-samples-from",
                *(str(path) for path in settings.sample_exclusion_files),
            )
        )
    command.extend(_source_selection_arguments(cell, settings))
    command.extend(_performance_arguments(cell, settings, episodes_per_cell, max_turns))
    command.extend(_generator_serving_arguments(settings))
    if settings.convention_note:
        command.append("--convention-note")
    if settings.generator == "v3":
        if settings.served_adapter_sha256 is None:
            raise ValueError("v3 generation requires the served adapter SHA-256")
        command.extend(("--served-adapter-sha256", settings.served_adapter_sha256))
    elif settings.served_adapter_sha256 is not None:
        raise ValueError("base generation cannot declare a served adapter digest")
    if settings.model_revision is not None:
        command.extend(("--model-revision", settings.model_revision))
    command.extend(_leak_selection_arguments(cell, settings))
    return command


def _performance_arguments(
    cell: ProbeCell, settings: CellRunConfig, episodes: int, max_turns: int
) -> list[str]:
    command = list(settings.performance_arguments)
    if "--perf-estimate-seconds" not in command:
        command.extend(
            (
                "--perf-estimate-seconds",
                str(
                    ESTIMATED_HOURS_PER_CELL
                    * 3600
                    / PROBE_EPISODES_PER_CELL
                    * episodes
                    * max_turns
                    / cell.max_turns
                ),
            )
        )
    if settings.server_pid is not None:
        command.extend(("--perf-server-pid", str(settings.server_pid)))
    if settings.server_startup_seconds is not None:
        command.extend(("--perf-startup-seconds", str(settings.server_startup_seconds)))
    return command


def _leak_selection_arguments(cell: ProbeCell, settings: CellRunConfig) -> list[str]:
    if cell.family != "r2e_leak":
        return []
    if settings.leak_assets is None:
        raise ValueError("the leak probe cell requires --leak-assets")
    arguments = [
        "--leak-assets",
        str(settings.leak_assets),
        "--leak-channel-split",
        "seed-v4",
        "--leak-channel-splits",
        str(settings.leak_channel_splits),
    ]
    if settings.leak_task_filter is not None:
        arguments.extend(("--leak-task-filter", str(settings.leak_task_filter)))
    if settings.leak_train_channel_per_episode:
        if settings.leak_task_filter is None:
            raise ValueError("single-channel training requires a leak task filter")
        arguments.append("--leak-train-channel-per-episode")
    return arguments


def build_health_command(
    records_path: Path,
    baseline_records: Path,
    *,
    generator: GeneratorName,
    schema_only: bool = False,
    assert_health: bool = True,
) -> list[str]:
    """Build the per-cell health command and persist its report beside the records."""
    command = [
        sys.executable,
        "-m",
        "reward_hacking.organism_rl.health",
        "--records",
        str(records_path),
        "--output",
        str(records_path.parent / "health.json"),
        "--assert-throughput",
    ]
    if not assert_health or generator in {"base", "27b"}:
        # Base is the baseline; absolute Gate 3 thresholds fitted to other cells misfire on its own generation.
        return command
    if schema_only:
        return [*command, "--assert", "--schema-only"]
    return [
        *command,
        "--assert",
        "--baseline",
        str(baseline_records),
        # Base 9B submits in 0-1 of 8 episodes on impossible and leak cells and runs out the turn budget instead;
        # the final state is graded either way, so the submission share is not a health signal for generation.
        "--submitted-ending-min",
        "0",
        # The NLL band was fitted on T=1.0 evaluation cells; the probe samples at T=0.6/top_k 20, which lowers
        # own-token NLL for every model (base 0.19). Only the ceiling, which catches degeneration, transfers.
        "--nll-min",
        "0",
    ]


def _runtime_model_revision(model: str, model_revision: str | None) -> str | None:
    configuration = AutoConfig.from_pretrained(
        model, revision=model_revision, local_files_only=True
    )
    resolved_revision = model_revision
    if resolved_revision is None and not Path(model).is_dir():
        resolved_revision = configuration._commit_hash  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if not isinstance(resolved_revision, str) or not resolved_revision:
            raise ValueError("cached Hub model config did not resolve a revision")
    return resolved_revision


def parse_server_model_max_len(response: object, served_model_id: str) -> int:
    """Read the effective context limit vLLM reports for a served model."""
    if not isinstance(response, dict):
        raise TypeError("vLLM /v1/models response must be an object")
    model_entries = response.get("data")
    if not isinstance(model_entries, list):
        raise TypeError("vLLM /v1/models response data must be a list")
    entries_by_id: dict[object, dict[str, object]] = {}
    for model_entry in model_entries:
        if not isinstance(model_entry, dict):
            raise TypeError("vLLM model entry must be an object")
        entries_by_id[model_entry.get("id")] = model_entry
    if served_model_id not in entries_by_id:
        raise ValueError(f"vLLM is not serving the requested model id {served_model_id!r}")
    served_entry = entries_by_id[served_model_id]
    # vLLM lists a LoRA adapter with its parent model and reports the context limit only on the parent.
    if served_entry.get("parent") is not None:
        served_entry = entries_by_id[served_entry["parent"]]
    max_model_len = served_entry.get("max_model_len")
    if type(max_model_len) is not int or max_model_len < 1:
        raise ValueError("vLLM model entry must report a positive max_model_len")
    return max_model_len


def _served_model_max_len(endpoint: str, served_model_id: str) -> int:
    endpoint_parts = urlsplit(endpoint)
    if endpoint_parts.scheme not in {"http", "https"} or endpoint_parts.netloc == "":
        raise ValueError("vLLM endpoint must be an HTTP or HTTPS URL")
    models_url = f"{endpoint.rstrip('/')}/v1/models"
    # The scheme and authority are checked above; this is a vLLM HTTP(S) metadata request.
    with urlopen(models_url, timeout=10) as response:  # noqa: S310
        payload: object = json.load(response)
    return parse_server_model_max_len(payload, served_model_id)


def effective_max_model_len(server_max_model_len: int, requested_max_model_len: int) -> int:
    """Refuse a server whose context window is below the requested wave setting."""
    if server_max_model_len < requested_max_model_len:
        raise ValueError(
            f"served max_model_len {server_max_model_len} is below the requested "
            f"{requested_max_model_len}; free GPU memory instead of shrinking the context"
        )
    return requested_max_model_len


def _server_config(
    args: argparse.Namespace,
    *,
    generator: GeneratorName,
    model_revision: str | None,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> VLLMServerConfig:
    adapter_directory = _resolve_path(args.v3_adapter)
    has_v3_adapter = generator == "v3"
    if has_v3_adapter and not adapter_directory.is_dir():
        raise FileNotFoundError(f"v3 adapter directory does not exist: {adapter_directory}")
    server_config = VLLMServerConfig(
        model=args.model,
        revision=model_revision,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        language_model_only=True,
        host=args.server_host,
        port=args.server_port,
        max_num_seqs=(
            16
            if generator == "27b"
            else (SMOKE_CONCURRENCY if args.smoke_cell is not None else DEFAULT_CONCURRENCY)
        ),
        enable_prefix_caching=True,
        dtype="bfloat16" if generator == "27b" else None,
        kv_cache_dtype="fp8" if generator == "27b" else None,
        attention_backend="FLASHINFER" if generator == "27b" else None,
        mamba_cache_mode="align" if generator == "27b" else None,
        max_num_batched_tokens=TWENTY_SEVEN_B_PREFILL_CHUNK_TOKENS if generator == "27b" else None,
        speculative_config=None
        if generator == "27b"
        else default_mtp_speculative_config(args.model),
        reasoning_config=dict(QWEN_THINKING_REASONING_CONFIG),
        enable_lora=has_v3_adapter,
        max_lora_rank=V3_LORA_RANK if has_v3_adapter else None,
        lora_modules=(((V3_SERVED_MODEL_ID, str(adapter_directory)),) if has_v3_adapter else ()),
        lora_serving_dir=_resolve_path(args.output_dir) / "vllm-serving-adapters"
        if has_v3_adapter
        else None,
        log_path=_resolve_path(args.output_dir) / f"vllm-{generator}-server.log",
    )
    validate_reasoning_config(server_config.reasoning_config)
    return server_config


def validate_reasoning_config(reasoning_config: Mapping[str, str] | None) -> None:
    """Reject reasoning settings that cannot expose Qwen thinking tokens."""
    if reasoning_config is None or dict(reasoning_config) != dict(QWEN_THINKING_REASONING_CONFIG):
        raise ValueError("probe vLLM server reasoning_config must match the Qwen thinking markers")


def _cell_run_config(
    args: argparse.Namespace,
    generator: GeneratorName,
    endpoint: str,
    model_revision: str | None,
    max_model_len: int,
) -> CellRunConfig:
    return CellRunConfig(
        generator=generator,
        endpoint=endpoint,
        model=args.model,
        output_dir=_resolve_path(args.output_dir),
        hints_path=_resolve_path(args.hints),
        leak_assets=_resolve_path(args.leak_assets) if args.leak_assets else None,
        leak_channel_splits=_resolve_path(args.leak_channel_splits),
        leak_task_filter=(
            _resolve_path(args.leak_task_filter) if args.leak_task_filter is not None else None
        ),
        leak_train_channel_per_episode=args.leak_train_channel_per_episode,
        baseline_records=_resolve_path(args.baseline_records),
        max_model_len=max_model_len,
        model_revision=model_revision,
        served_adapter_sha256=(
            _adapter_sha256(_resolve_path(args.v3_adapter)) if generator == "v3" else None
        ),
        smoke=args.smoke_cell is not None,
        performance_arguments=performance_cli_arguments(args),
        serving_config=(
            {
                "model": args.model,
                "model_revision": model_revision,
                "dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "attention_backend": "FLASHINFER",
                "enable_lora": False,
                "speculative_config": None,
                "mamba_cache_mode": "align",
                "max_num_batched_tokens": TWENTY_SEVEN_B_PREFILL_CHUNK_TOKENS,
                "max_num_seqs": 16,
                "enable_prefix_caching": True,
                "max_model_len": max_model_len,
                "reasoning_config": dict(QWEN_THINKING_REASONING_CONFIG),
            }
            if generator == "27b"
            else None
        ),
    )


def _validate_runtime_inputs(
    args: argparse.Namespace, cells: Sequence[ProbeCell], *, smoke: bool = False
) -> None:
    required_files = (
        [(_resolve_path(args.hints), "hint definitions")]
        if any(not cell.convention_note for cell in cells)
        else []
    )
    if not smoke:
        required_files.append((_resolve_path(args.baseline_records), "base eval records"))
    for path, description in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"{description} file does not exist: {path}")
    if any(cell.family == "r2e_leak" for cell in cells):
        if args.leak_assets is None:
            raise ValueError("the leak probe cell requires --leak-assets")
        for path, description in (
            (_resolve_path(args.leak_assets), "train leak assets"),
            (_resolve_path(args.leak_channel_splits), "leak channel splits"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{description} file does not exist: {path}")
        if args.leak_task_filter is not None:
            task_filter = _resolve_path(args.leak_task_filter)
            if not task_filter.is_file():
                raise FileNotFoundError(f"leak task filter file does not exist: {task_filter}")


def _launch_summary(
    stage: ProbeStage,
    model: str,
    cells: Sequence[ProbeCell],
    *,
    episodes_per_cell: int = PROBE_EPISODES_PER_CELL,
) -> str:
    episodes = len(cells) * episodes_per_cell
    estimated_hours = (
        len(cells) * ESTIMATED_HOURS_PER_CELL * episodes_per_cell / PROBE_EPISODES_PER_CELL
    )
    cell_names = ", ".join(cell.name for cell in cells) or "none"
    return (
        f"Stage: {stage} on {model}\n"
        f"Cells: {cell_names}\n"
        f"Episodes: {episodes}\n"
        f"Estimated wall-clock: {estimated_hours:.2f} hours\n"
    )


def _run_command(command: Sequence[str]) -> None:
    # Each argv is built from a fixed Python module entry point; execution never uses a shell.
    subprocess.run(command, cwd=_repository_root(), check=True)  # noqa: S603


def _run_cells(cells: Sequence[ProbeCell], settings: CellRunConfig) -> None:
    output_dir = settings.output_dir
    for cell in cells:
        run_directory = output_dir / cell.name / settings.generator
        command = build_generate_command(cell, settings)
        logger.info("generating %s with %s: %s", cell.name, settings.generator, command)
        _run_command(command)
        records_path = run_directory / "records.jsonl"
        _run_command(
            build_health_command(
                records_path,
                settings.baseline_records,
                generator=settings.generator,
                schema_only=settings.smoke and settings.generator == "v3",
            )
        )
        _run_command(
            [
                sys.executable,
                "-m",
                "reward_hacking.organism_rl.yield_table",
                "--output-dir",
                str(output_dir),
            ]
        )
        row_by_name = {row.cell.name: row for row in summarize_output(output_dir)}
        completed_episodes = (
            row_by_name[cell.name].base_episodes
            if settings.generator == "base"
            else row_by_name[cell.name].v3_episodes
        )
        expected_episodes = SMOKE_EPISODES_PER_CELL if settings.smoke else PROBE_EPISODES_PER_CELL
        if completed_episodes != expected_episodes:
            raise ValueError(
                f"{cell.name} {settings.generator} produced {completed_episodes} records; "
                f"expected {expected_episodes}"
            )


def _kv_pool_tokens(server_log: str) -> int:
    """Read the capacity measured during this server startup."""
    matches = re.findall(r"GPU KV cache size: ([\d,]+) tokens", server_log)
    if not matches:
        raise ValueError("managed 27b server did not report GPU KV cache capacity")
    return int(matches[-1].replace(",", ""))


def kv_planning_context_tokens(max_model_len: int) -> int:
    """Plan sequence admission against half the requested context length."""
    if max_model_len < 1:
        raise ValueError("27b context window must be positive")
    planning_context_tokens = max_model_len // KV_PLANNING_CONTEXT_DIVISOR
    if planning_context_tokens < 1:
        raise ValueError("27b context window must be at least two tokens")
    return planning_context_tokens


def kv_pool_concurrency(server_log: str, max_model_len: int) -> int:
    """Fit half-context planning slots in the pool while requiring one full context."""
    planning_context_tokens = kv_planning_context_tokens(max_model_len)
    capacity_tokens = _kv_pool_tokens(server_log)
    if capacity_tokens < max_model_len:
        raise ValueError("27b KV pool cannot hold one requested context window")
    concurrency = min(DEFAULT_CONCURRENCY, capacity_tokens // planning_context_tokens)
    logger.info(
        "27b kv_capacity_tokens=%s max_model_len=%s kv_planning_context_tokens=%s concurrency=%s",
        capacity_tokens,
        max_model_len,
        planning_context_tokens,
        concurrency,
    )
    return concurrency


def _run_with_performance_policy(
    run: Callable[[CellRunConfig], None], settings: CellRunConfig
) -> None:
    arguments = list(settings.performance_arguments)
    if settings.server_pid is not None:
        arguments.extend(("--perf-server-pid", str(settings.server_pid)))
    arguments.extend(("--perf-startup-seconds", str(settings.server_startup_seconds)))
    performance_arguments_reset = stage_performance_arguments.set(tuple(arguments))
    try:
        run(settings)
    finally:
        stage_performance_arguments.reset(performance_arguments_reset)


def run_generator(
    args: argparse.Namespace, generator: GeneratorName, run: Callable[[CellRunConfig], None]
) -> None:
    """Serve one generator with the probe's lifecycle and verify its effective context window."""
    model_revision = _runtime_model_revision(args.model, args.model_revision)
    served_model_id = V3_SERVED_MODEL_ID if generator == "v3" else args.model
    if args.endpoint is not None:
        server_max_model_len = _served_model_max_len(args.endpoint, served_model_id)
        _run_with_performance_policy(
            run,
            _cell_run_config(
                args,
                generator,
                args.endpoint,
                model_revision,
                effective_max_model_len(server_max_model_len, args.max_model_len),
            ),
        )
    else:
        require_free_gpu()
        gpu_memory_utilization, device_name = resolve_gpu_engine_settings(torch)
        server_config = _server_config(
            args,
            generator=generator,
            model_revision=model_revision,
            max_model_len=(args.max_model_len),
            gpu_memory_utilization=gpu_memory_utilization,
        )
        log_path = server_config.log_path
        log_offset = log_path.stat().st_size if log_path is not None and log_path.exists() else 0
        startup_started = time.monotonic()
        with VLLMServer(server_config) as server:
            startup_seconds = time.monotonic() - startup_started
            server_max_model_len = _served_model_max_len(server.base_url, served_model_id)
            settings = _cell_run_config(
                args,
                generator,
                server.base_url,
                model_revision,
                effective_max_model_len(server_max_model_len, args.max_model_len),
            )
            settings = replace(
                settings,
                server_pid=server.process.pid,
                server_startup_seconds=startup_seconds,
                performance_arguments=(
                    *settings.performance_arguments,
                    "--gpu-preflight-passed-at-stage-start",
                ),
            )
            if generator == "27b":
                if log_path is None:
                    raise ValueError("27b capacity discovery requires a server log")
                with log_path.open("rb") as server_log:
                    server_log.seek(log_offset)
                    startup_log = server_log.read().decode("utf-8")
                settings = replace(
                    settings,
                    concurrency=kv_pool_concurrency(startup_log, settings.max_model_len),
                    serving_runtime={
                        "gpu_memory_utilization": gpu_memory_utilization,
                        "device": device_name,
                        "concurrency_source": "startup_kv_pool",
                        "kv_cache_capacity_tokens": _kv_pool_tokens(startup_log),
                        "max_model_len": settings.max_model_len,
                        "kv_planning_context_tokens": kv_planning_context_tokens(
                            settings.max_model_len
                        ),
                    },
                )
            _run_with_performance_policy(run, settings)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the stage plan, then run base or the separately launched v3 add-on."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    stage = cast("ProbeStage", args.stage)
    output_dir = _resolve_path(args.output_dir)
    smoke = args.smoke_cell is not None
    if smoke:
        cells = (next(cell for cell in PROBE_CELLS if cell.name == args.smoke_cell),)
    else:
        cells = PROBE_CELLS if stage == "base" else select_v3_cells(output_dir)
    episodes_per_cell = SMOKE_EPISODES_PER_CELL if smoke else PROBE_EPISODES_PER_CELL
    sys.stdout.write(_launch_summary(stage, args.model, cells, episodes_per_cell=episodes_per_cell))
    sys.stdout.flush()
    if args.dry_run:
        if smoke and stage == "base":
            server_config = _server_config(
                args,
                generator="base",
                model_revision=args.model_revision or "smoke-plan-only",
                max_model_len=args.max_model_len,
                gpu_memory_utilization=0.9,
            )
            validate_reasoning_config(server_config.reasoning_config)
        sys.stdout.write(render_yield_table(output_dir) + "\n")
        return 0
    if not cells:
        sys.stdout.write("No v3 add-on cells were selected.\n")
        sys.stdout.write(render_yield_table(output_dir) + "\n")
        return 0

    _validate_runtime_inputs(args, cells, smoke=smoke)
    run_generator(args, stage, lambda settings: _run_cells(cells, settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
