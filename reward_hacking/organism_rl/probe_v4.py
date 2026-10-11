"""Run the staged seed-v4 yield probe against the current organism generator."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlsplit
from urllib.request import urlopen

import torch
from transformers import AutoConfig

from reward_hacking.distill.sample import resolve_gpu_engine_settings
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

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
ProbeStage = Literal["base", "v3"]
V3_SERVED_MODEL_ID = "seed-v3-c34"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
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
PROBE_EPISODES_PER_CELL = 8
ESTIMATED_HOURS_PER_CELL = 0.25
V3_LORA_RANK = 16
# Probe records are wave-1 data, so the probe serves the waves' fixed context window.
WAVE_MAX_MODEL_LEN = 65_536


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
    served_adapter_sha256: str | None = None


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Launch the stages separately after scripts/gpu_preflight.py. Base: "
            "scripts/tmux_run.sh seed-v4-probe-base -- scripts/resource-limits.sh --gpu -t 90m -- "
            'env PATH="$PWD/.venv/bin:$PATH" reward_hacking/organism_rl/probe-v4.sh --stage base '
            "--leak-assets /var/tmp/leak-env/assets-v6-train/assets.json. "
            "V3 add-on: "
            "scripts/tmux_run.sh seed-v4-probe-v3 -- scripts/resource-limits.sh --gpu -t 90m -- "
            'env PATH="$PWD/.venv/bin:$PATH" reward_hacking/organism_rl/probe-v4.sh --stage v3 '
            "--leak-assets /var/tmp/leak-env/assets-v6-train/assets.json."
        ),
    )
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
    parser.add_argument("--baseline-records", type=Path, default=DEFAULT_BASELINE_RECORDS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the stage controls and refuse unsafe settings before model access."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.max_model_len < 1:
        parser.error("max-model-len must be positive")
    if not 1 <= args.server_port <= MAX_TCP_PORT:
        parser.error("server-port must be between 1 and 65535")
    return args


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else _repository_root() / path


def select_v3_cells(output_dir: Path) -> tuple[ProbeCell, ...]:
    """Select cells the yield table marks ready for the v3 add-on."""
    rows = summarize_output(output_dir)
    incomplete = [row.cell.name for row in rows if row.base_episodes != PROBE_EPISODES_PER_CELL]
    if incomplete:
        raise ValueError(
            f"v3 stage requires eight base episodes in every cell; incomplete: {incomplete}"
        )
    return tuple(row.cell for row in rows if row.v3_scheduled)


def build_generate_command(cell: ProbeCell, settings: CellRunConfig) -> list[str]:
    """Build one invocation from the existing generate.py command-line contract."""
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
        "--hints",
        str(settings.hints_path),
        "--hint-family",
        cell.hint_family,
        "--hint-strength",
        "vague",
        "--partition",
        "train",
        "--max-tasks",
        str(PROBE_EPISODES_PER_CELL),
        "--samples",
        "1",
        "--concurrency",
        str(DEFAULT_CONCURRENCY),
        "--temperature",
        "0.6",
        "--top-p",
        "1.0",
        "--top-k",
        "20",
        "--presence-penalty",
        "1.5",
        "--thinking-token-budget",
        "2048",
        "--max-turns",
        str(cell.max_turns),
        "--announced-turn-budget",
        str(cell.max_turns),
        "--max-new-tokens",
        str(DEFAULT_MAX_NEW_TOKENS),
        "--max-model-len",
        str(settings.max_model_len),
        "--output-dir",
        str(settings.output_dir / cell.name / settings.generator),
    ]
    if settings.generator == "v3":
        if settings.served_adapter_sha256 is None:
            raise ValueError("v3 generation requires the served adapter SHA-256")
        command.extend(("--served-adapter-sha256", settings.served_adapter_sha256))
    elif settings.served_adapter_sha256 is not None:
        raise ValueError("base generation cannot declare a served adapter digest")
    if settings.model_revision is not None:
        command.extend(("--model-revision", settings.model_revision))
    if cell.family == "r2e_leak":
        if settings.leak_assets is None:
            raise ValueError("the leak probe cell requires --leak-assets")
        command.extend(
            (
                "--leak-assets",
                str(settings.leak_assets),
                "--leak-channel-split",
                "seed-v4",
                "--leak-channel-splits",
                str(settings.leak_channel_splits),
            )
        )
    return command


def build_health_command(records_path: Path, baseline_records: Path) -> list[str]:
    """Build the required per-cell absolute health assertion against the base eval records."""
    return [
        sys.executable,
        "-m",
        "reward_hacking.organism_rl.health",
        "--records",
        str(records_path),
        "--assert",
        "--baseline",
        str(baseline_records),
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
    for model_entry in model_entries:
        if not isinstance(model_entry, dict):
            raise TypeError("vLLM model entry must be an object")
        if model_entry.get("id") == served_model_id:
            max_model_len = model_entry.get("max_model_len")
            if type(max_model_len) is not int or max_model_len < 1:
                raise ValueError("vLLM model entry must report a positive max_model_len")
            return max_model_len
    raise ValueError(f"vLLM is not serving the requested model id {served_model_id!r}")


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
    return VLLMServerConfig(
        model=args.model,
        revision=model_revision,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        language_model_only=True,
        host=args.server_host,
        port=args.server_port,
        max_num_seqs=DEFAULT_CONCURRENCY,
        enable_prefix_caching=True,
        speculative_config=default_mtp_speculative_config(args.model),
        reasoning_config=dict(QWEN_THINKING_REASONING_CONFIG),
        enable_lora=has_v3_adapter,
        max_lora_rank=V3_LORA_RANK if has_v3_adapter else None,
        lora_modules=(((V3_SERVED_MODEL_ID, str(adapter_directory)),) if has_v3_adapter else ()),
        lora_serving_dir=_resolve_path(args.output_dir) / "vllm-serving-adapters"
        if has_v3_adapter
        else None,
        log_path=_resolve_path(args.output_dir) / f"vllm-{generator}-server.log",
    )


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
        baseline_records=_resolve_path(args.baseline_records),
        max_model_len=max_model_len,
        model_revision=model_revision,
        served_adapter_sha256=(
            _adapter_sha256(_resolve_path(args.v3_adapter)) if generator == "v3" else None
        ),
    )


def _validate_runtime_inputs(args: argparse.Namespace, cells: Sequence[ProbeCell]) -> None:
    for path, description in (
        (_resolve_path(args.hints), "hint definitions"),
        (_resolve_path(args.baseline_records), "base eval records"),
    ):
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


def _launch_summary(stage: ProbeStage, model: str, cells: Sequence[ProbeCell]) -> str:
    episodes = len(cells) * PROBE_EPISODES_PER_CELL
    estimated_hours = len(cells) * ESTIMATED_HOURS_PER_CELL
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
        _run_command(build_health_command(records_path, settings.baseline_records))
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
        if completed_episodes != PROBE_EPISODES_PER_CELL:
            raise ValueError(
                f"{cell.name} {settings.generator} produced {completed_episodes} records; "
                f"expected {PROBE_EPISODES_PER_CELL}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Print the stage plan, then run base or the separately launched v3 add-on."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    stage = cast("ProbeStage", args.stage)
    output_dir = _resolve_path(args.output_dir)
    cells = PROBE_CELLS if stage == "base" else select_v3_cells(output_dir)
    sys.stdout.write(_launch_summary(stage, args.model, cells))
    sys.stdout.flush()
    if args.dry_run:
        sys.stdout.write(render_yield_table(output_dir) + "\n")
        return 0
    if not cells:
        sys.stdout.write("No v3 add-on cells were selected.\n")
        sys.stdout.write(render_yield_table(output_dir) + "\n")
        return 0

    _validate_runtime_inputs(args, cells)
    model_revision = _runtime_model_revision(args.model, args.model_revision)
    generator: GeneratorName = "base" if stage == "base" else "v3"
    served_model_id = V3_SERVED_MODEL_ID if generator == "v3" else args.model
    if args.endpoint is not None:
        server_max_model_len = _served_model_max_len(args.endpoint, served_model_id)
        _run_cells(
            cells,
            _cell_run_config(
                args,
                generator,
                args.endpoint,
                model_revision,
                effective_max_model_len(server_max_model_len, args.max_model_len),
            ),
        )
    else:
        gpu_memory_utilization, _device_name = resolve_gpu_engine_settings(torch)
        server_config = _server_config(
            args,
            generator=generator,
            model_revision=model_revision,
            max_model_len=(args.max_model_len),
            gpu_memory_utilization=gpu_memory_utilization,
        )
        with VLLMServer(server_config) as server:
            server_max_model_len = _served_model_max_len(server.base_url, served_model_id)
            _run_cells(
                cells,
                _cell_run_config(
                    args,
                    generator,
                    server.base_url,
                    model_revision,
                    effective_max_model_len(server_max_model_len, args.max_model_len),
                ),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
