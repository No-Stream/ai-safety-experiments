"""Serve and evaluate an agentic SFT adapter: vLLM launch, evaluation commands and tokenizer loading."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shlex
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from transformers import AutoConfig, AutoTokenizer

from grpo.gpu_memory import cap_gpu_memory_utilization, gpu_memory_info
from reward_hacking.vllm_server import VLLMServer, VLLMServerConfig

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_STUDENT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_GPU_MEMORY_CEILING_GIB = 29.0
GIBIBYTE = 1024**3
EVALUATION_SERVER_BUDGET = "6h"
EVALUATION_THINKING_CONFIG = {"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}


def gpu_memory_utilization(*, total_bytes: int, used_bytes: int, ceiling_bytes: int) -> float:
    """Resolve vLLM's GPU fraction so current shared use stays below a total-memory ceiling."""
    if total_bytes <= 0 or ceiling_bytes <= 0 or used_bytes < 0:
        raise ValueError("GPU total, used, and ceiling memory values are invalid")
    effective_ceiling_bytes = min(total_bytes, ceiling_bytes)
    remaining_bytes = effective_ceiling_bytes - used_bytes
    if remaining_bytes <= 0:
        raise ValueError("GPU memory is already at or above the configured total-memory ceiling")
    return remaining_bytes / total_bytes


def _resolve_cuda_home(cuda_home: Path | None = None) -> Path:
    if cuda_home is None:
        configured_cuda_home = os.environ.get("CUDA_HOME")
        if configured_cuda_home:
            cuda_home = Path(configured_cuda_home)
        else:
            nvcc_path = shutil.which("nvcc")
            if nvcc_path is not None:
                cuda_home = Path(nvcc_path).resolve().parent.parent
            else:
                default_cuda_home = Path("/usr/local/cuda")
                if (default_cuda_home / "bin" / "nvcc").is_file():
                    cuda_home = default_cuda_home
                else:
                    raise FileNotFoundError(
                        "nvcc is required to render FlashInfer evaluation commands"
                    )
    resolved_cuda_home = cuda_home.resolve()
    if not (resolved_cuda_home / "bin" / "nvcc").is_file():
        raise FileNotFoundError(f"CUDA_HOME does not contain bin/nvcc: {resolved_cuda_home}")
    return resolved_cuda_home


def _adapter_model_revision(adapter_dir: Path, requested_revision: str | None) -> str | None:
    if requested_revision is not None:
        return requested_revision
    training_state_path = adapter_dir / "distill_state.json"
    if not training_state_path.is_file():
        return None
    training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
    if not isinstance(training_state, Mapping):
        raise TypeError("adapter distillation state must be a JSON object")
    revision = training_state.get("base_model_revision")
    if revision is not None and not isinstance(revision, str):
        raise TypeError("adapter base model revision must be a string or null")
    return revision


def serve_vllm(args: argparse.Namespace) -> int:
    """Start the vLLM evaluation server for an adapter and block until it exits."""
    if not torch.cuda.is_available():
        raise RuntimeError("agentic SFT evaluation server requires one CUDA device")
    if not math.isfinite(args.gpu_memory_ceiling_gib) or args.gpu_memory_ceiling_gib <= 0:
        raise ValueError("GPU memory ceiling must be positive and finite")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    free_bytes, total_bytes = gpu_memory_info(torch, device_index)
    used_bytes = total_bytes - free_bytes
    ceiling_bytes = int(args.gpu_memory_ceiling_gib * GIBIBYTE)
    try:
        ceiling_utilization = gpu_memory_utilization(
            total_bytes=total_bytes,
            used_bytes=used_bytes,
            ceiling_bytes=ceiling_bytes,
        )
    except ValueError as error:
        # The normal sizing snapshot already logs the delta; read Torch here only for the failure text.
        torch_free_bytes, _torch_total_bytes = torch.cuda.mem_get_info(device_index)
        host_held_bytes = max(torch_free_bytes - free_bytes, 0)
        raise ValueError(
            f"{error}; nvidia-smi estimates {host_held_bytes / GIBIBYTE:.2f} GiB held by the "
            "host based on the torch/nvidia-smi free-memory difference"
        ) from error
    memory_utilization = cap_gpu_memory_utilization(torch, ceiling_utilization, device_index)
    logger.info(
        "vLLM startup GPU memory: device=%s used=%d total=%d ceiling=%d gpu_memory_utilization=%.6f",
        properties.name,
        used_bytes,
        total_bytes,
        min(total_bytes, ceiling_bytes),
        memory_utilization,
    )

    model_config = AutoConfig.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=True,
    )
    resolved_revision = model_config._commit_hash or args.model_revision  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    if resolved_revision is None and not Path(args.model).is_dir():
        raise RuntimeError("cached model config did not resolve a Hub revision")
    repository_root = Path(__file__).resolve().parents[2]
    server_config = VLLMServerConfig(
        model=args.model,
        revision=resolved_revision,
        max_model_len=65_536,
        gpu_memory_utilization=memory_utilization,
        language_model_only=True,
        max_num_seqs=args.max_num_seqs,
        attention_backend="FLASHINFER",
        enable_prefix_caching=True,
        kv_cache_dtype="fp8",
        mamba_cache_mode="align",
        max_num_batched_tokens=8192,
        enable_lora=True,
        max_lora_rank=16,
        lora_modules=(("agentic-sft", str(args.adapter_dir.resolve())),),
        lora_serving_dir=args.serving_dir.resolve(),
        reasoning_config=EVALUATION_THINKING_CONFIG,
        executable=str(repository_root / ".venv" / "bin" / "vllm"),
        log_path=args.output_dir / "server.log",
    )
    logger.info("vLLM child command: %s", shlex.join(server_config.command()))
    with VLLMServer(server_config) as server:
        return_code = server.process.wait()
    if return_code != 0:
        raise RuntimeError(f"vLLM evaluation server exited with status {return_code}")
    return 0


def evaluation_commands(  # noqa: PLR0913 - these independent launch settings need explicit output
    adapter_dir: Path,
    *,
    serving_dir: Path,
    output_dir: Path,
    repository_root: Path,
    cuda_home: Path | None = None,
    gpu_memory_ceiling_gib: float = DEFAULT_GPU_MEMORY_CEILING_GIB,
    model: str = DEFAULT_STUDENT_MODEL,
    model_revision: str | None = None,
) -> tuple[str, str, str]:
    """Render the owned, resource-limited vLLM launch and matched ladder commands."""
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter_dir}")
    if not math.isfinite(gpu_memory_ceiling_gib) or gpu_memory_ceiling_gib <= 0:
        raise ValueError("GPU memory ceiling must be positive and finite")
    resolved_cuda_home = _resolve_cuda_home(cuda_home)
    server_log = output_dir / "server.log"
    venv_bin = repository_root / ".venv" / "bin"
    inherited_path = os.environ.get("PATH", "")
    launch_path = os.pathsep.join(
        str(path) for path in (resolved_cuda_home / "bin", venv_bin, inherited_path) if path
    )
    server_args = [
        str(repository_root / "scripts" / "tmux_run.sh"),
        "agentic-sft-eval-server",
        "--",
        str(repository_root / "scripts" / "resource-limits.sh"),
        "--gpu",
        "-t",
        EVALUATION_SERVER_BUDGET,
        "--",
        "env",
        f"CUDA_HOME={resolved_cuda_home}",
        f"PATH={launch_path}",
        f"PYTHONPATH={repository_root}",
        str(venv_bin / "python"),
        "-m",
        "reward_hacking.distill.agentic_sft",
        "serve",
        "--model",
        model,
        "--adapter-dir",
        str(adapter_dir.resolve()),
        "--serving-dir",
        str(serving_dir.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--gpu-memory-ceiling-gib",
        str(gpu_memory_ceiling_gib),
    ]
    adapter_revision = _adapter_model_revision(adapter_dir, model_revision)
    if adapter_revision is not None:
        server_args.extend(["--model-revision", adapter_revision])
    server_command = shlex.join(server_args)
    evaluation_specs = (
        (
            "conflict",
            repository_root / "reward_hacking/ladder/data/scenarios-v2.2-conflict",
        ),
        (
            "heldout",
            repository_root / "reward_hacking/ladder/data/scenarios-v2.2-heldout",
        ),
    )
    runner_commands: list[str] = []
    for evaluation_name, scenario_root in evaluation_specs:
        evaluation_dir = output_dir / evaluation_name
        runner_args = [
            ".venv/bin/python",
            "-m",
            "reward_hacking.ladder.runner",
            "--endpoint",
            "http://127.0.0.1:8000",
            "--server-log",
            str(server_log),
            "--model",
            f"agentic-sft={model}",
            "--chat-template-kwarg",
            "agentic-sft:reasoning_effort=low",
            "--scenario-root",
            str(scenario_root),
            "--max-new-tokens",
            "16384",
            "--announced-turn-budget",
            "30",
            "--thinking-token-budget",
            "2048",
            "--history-reasoning",
            "retain",
            "--episode-concurrency",
            "1",
            "--belief-probe",
            "--arm",
            "real",
            "--arm",
            "claimed",
            "--samples",
            "1",
            "--oversize-reason",
            f"full {evaluation_name} ladder validation",
            "--output",
            str(evaluation_dir / "records.jsonl"),
            "--episode-root",
            str(evaluation_dir / "episodes"),
            "--no-resume",
        ]
        runner_commands.append(
            f"cd {shlex.quote(str(repository_root))} && "
            f"mkdir -p {shlex.quote(str(evaluation_dir))} && {shlex.join(runner_args)}"
        )
    return server_command, runner_commands[0], runner_commands[1]


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a locally cached tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    return cast("PreTrainedTokenizerBase", tokenizer)


def parse_solvable_hack_floor(value: str) -> int:
    """Parse a nonnegative integer solvable-hack floor for argparse."""
    try:
        floor = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("solvable hack floor must be an integer") from error
    if floor < 0:
        raise argparse.ArgumentTypeError("solvable hack floor must be nonnegative")
    return floor
