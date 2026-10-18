"""CLI and provenance for the external-server multi-turn RLVR path."""

import argparse
import hashlib
import inspect
import json
import logging
import math
import subprocess
import sys
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from grpo.gpu_memory import nvidia_smi_memory_info
from grpo.throughput import discover_lora_targets
from reward_hacking.distill.fla_prewarm import prewarm_fla_nb_buckets
from reward_hacking.distill.memory import (
    install_before_load,
    install_tiled_mlp,
    load_base_model,
    prepare_checkpoint_offload,
)
from reward_hacking.distill.train import (
    cast_lora_to_float32,
    memory_options_from_args,
    prepare_fp32_head,
)
from reward_hacking.evidence_store import read_jsonl
from reward_hacking.ladder.thinking import QWEN_THINKING_REASONING_CONFIG
from reward_hacking.model_backend import VLLMRequestGate
from reward_hacking.organism_rl.admission import (
    LiveKVAdmission,
    kv_capacity_tokens,
    resolve_rollout_concurrency,
)
from reward_hacking.organism_rl.dynamic_sampling_history import (
    HistorySeedSummary,
    seed_dynamic_sampling_history,
)
from reward_hacking.organism_rl.generate_provenance import active_provenance_path
from reward_hacking.organism_rl.ladder_source import LadderEpisodeSource
from reward_hacking.organism_rl.leak_assets import leak_task_source_metadata
from reward_hacking.organism_rl.leak_selection import add_leak_arguments, resolve_leak_options
from reward_hacking.organism_rl.phases import VLLMPhaseController, vllm_sleeping_weight_bytes
from reward_hacking.organism_rl.precision import (
    assert_fp32_training_state,
    record_precision_provenance,
)
from reward_hacking.organism_rl.prescreen import run_prescreen
from reward_hacking.organism_rl.prescreen_pool import (
    derive_prompt_pool,
    prompt_pool_provenance,
    select_prompt_pool,
)
from reward_hacking.organism_rl.rollout import (
    MixedEpisodeSource,
    OrganismRollout,
    TaskEpisodeSource,
    TaskFamily,
    task_definition_content,
    task_definition_sha256,
    verdict_reward,
)
from reward_hacking.organism_rl.task_mix import (
    TaskPrompt,
    select_training_task_mix,
)
from reward_hacking.organism_rl.task_mix import parse_in_band_fraction as _parse_in_band_fraction
from reward_hacking.organism_rl.trainer import OrganismGRPOTrainer, validate_checkpoint
from reward_hacking.semantic_identity import (
    model_identity,
    recorded_model_matches,
    semantic_content_identity,
)
from reward_hacking.serving_quantization import (
    add_serving_quantization_arguments,
    serving_quantization_kwargs,
    validate_serving_quantization,
)
from reward_hacking.vllm_server import VLLMServer, VLLMServerConfig, default_mtp_speculative_config

logger = logging.getLogger(__name__)
CLI_PROVENANCE_SCHEMA_VERSION = 1
MINIMUM_GENERATIONS = 2
TRAINER_LORA_RANK = 16
DISCARD_WEIGHTS_SLEEP_LEVEL = 2
MINIMUM_RETAINED_CHECKPOINTS = 2
MAXIMUM_MIDDLE_CHECKPOINTS = 3


@dataclass(frozen=True)
class RuntimeSizing:
    """Conservative sequence allocation derived from device capacity and architecture."""

    max_sequence_tokens: int
    micro_batch_size: int
    num_generations: int


def resolve_sizing(*, total_bytes: int, hidden_size: int, layers: int) -> RuntimeSizing:
    """Allocate a fraction of observed VRAM to checkpointed token activations.

    This is an initial estimate, not an observed fit; explicit sequence limits remain available.
    Training micro-batches contain one episode.
    """
    if min(total_bytes, hidden_size, layers) <= 0:
        raise ValueError("device capacity and model dimensions must be positive")
    activation_bytes_per_token = hidden_size * layers * 2 * 8
    sequence_tokens = max(1, total_bytes // (activation_bytes_per_token * 4))
    return RuntimeSizing(sequence_tokens, 1, 2)


# At 9B, nvidia-smi free after each vLLM wake swings 1.9-4.4 GiB below the post-startup value; below ~1.5 GiB the WSL
# driver degrades decode 3-4x instead of failing.
TRAINER_VRAM_HEADROOM_GIB = 4.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # noqa: PLR0915 - explicit CLI flags
    """Parse explicit experiment, sampling, teacher and memory settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--persistent-jail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse one actor jail per episode; disable for per-command jails",
    )

    parser.add_argument("--prescreen-samples", type=int)
    parser.add_argument("--prompt-pool-from", type=Path)
    parser.add_argument("--prompt-pool-min-rewarded-samples", type=int, default=1)
    parser.add_argument("--prompt-pool-unscreened-fraction", type=float, default=0.10)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-revision")
    parser.add_argument("--endpoint", help="use an externally managed vLLM server")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument(
        "--trainer-vram-headroom-gib",
        type=float,
        default=TRAINER_VRAM_HEADROOM_GIB,
        help="VRAM left free after owned vLLM startup for the offloaded trainer and vLLM's post-wake growth",
    )
    parser.add_argument("--enable-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-lora-rank", type=int, default=TRAINER_LORA_RANK)
    parser.add_argument("--enable-sleep-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sleep-level",
        type=int,
        choices=(1, 2),
        default=1,
        help="1 retains weights in pinned host RAM; 2 frees host RAM and reloads weights on wake",
    )
    parser.add_argument(
        "--frozen-host-memory",
        choices=("pinned", "pageable", "persistent"),
        default="persistent",
        help=(
            "frozen trainer host arena: pinned/pageable allocate per rollout; "
            "persistent retains a pinned copy across phases"
        ),
    )
    parser.add_argument(
        "--enable-speculative-decoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="default to MTP-2 for dense, unquantized Qwen3.5 models",
    )
    parser.add_argument(
        "--speculative-config",
        type=_parse_speculative_config,
        help="JSON object passed to the owned vLLM server",
    )
    parser.add_argument(
        "--reasoning-config",
        type=_parse_reasoning_config,
        default=dict(QWEN_THINKING_REASONING_CONFIG),
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse each episode's earlier turns instead of re-prefilling the whole history",
    )
    parser.add_argument(
        "--rollout-concurrency",
        type=int,
        help="cap episodes sampled at once; defaults to the KV pool-derived fan-out",
    )
    parser.add_argument("--step-pipelining", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--expected-peak-context-tokens",
        type=int,
        help="expected peak live context per episode; defaults to half of effective max-sequence-tokens",
    )
    parser.add_argument(
        "--kv-capacity-tokens",
        type=int,
        help="measured vLLM KV pool capacity; required for an external endpoint",
    )
    parser.add_argument("--mamba-cache-mode")
    parser.add_argument("--mamba-ssm-cache-dtype", choices=("auto", "float32", "bfloat16"))
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--prompts-per-step", type=int, default=1)
    parser.add_argument(
        "--dynamic-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="refill each step with prompt groups that have nonzero reward variance",
    )
    parser.add_argument(
        "--dynamic-sampling-max-groups",
        type=int,
        help="maximum groups sampled per step; defaults to three times prompts-per-step",
    )
    parser.add_argument("--dynamic-sampling-skip-after", type=int, default=2)
    parser.add_argument(
        "--dynamic-sampling-history-from",
        action="append",
        nargs="+",
        type=Path,
        default=[],
        metavar="DIR",
        help="seed dynamic sampling from prior organism-RL run directories (repeatable)",
    )
    parser.add_argument(
        "--max-sequence-tokens", "--max-model-len", dest="max_sequence_tokens", type=int
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument(
        "--max-consecutive-truncated-turns",
        type=int,
        default=2,
        help="end an episode after this many capped turns in a row; 0 disables the limit",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    thinking_budget = parser.add_mutually_exclusive_group()
    thinking_budget.add_argument("--thinking-token-budget", type=int)
    thinking_budget.add_argument(
        "--no-thinking-token-budget",
        dest="thinking_token_budget",
        action="store_const",
        const=None,
    )
    parser.set_defaults(thinking_token_budget=512)
    parser.add_argument(
        "--family",
        "--families",
        dest="family_entries",
        action="append",
        choices=("ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak"),
        help="task family to train on (repeatable; default: ilcb and aisi)",
    )
    parser.add_argument(
        "--impossible-variant",
        choices=("oneoff", "conflicting", "loose"),
        default="oneoff",
        help="impossible task variant (oneoff reads as an ordinary test bug)",
    )
    parser.add_argument("--solvable-fraction", type=float, default=0.15)
    parser.add_argument(
        "--in-band-fraction",
        dest="in_band_entries",
        action="append",
        type=_parse_in_band_fraction,
        help="repeatable FAMILY=FRACTION or FAMILY/ENVIRONMENT=FRACTION; default: trusted rerun",
    )
    parser.add_argument(
        "--ladder-fraction",
        type=float,
        default=0.0,
        help="fraction of selected prompts sampled from training ladder scenarios",
    )
    parser.add_argument(
        "--ladder-scenarios-dir", type=Path, default=Path("reward_hacking/ladder/data/scenarios")
    )
    parser.add_argument("--teacher-beta", type=float, default=0.0)
    parser.add_argument("--teacher-path", type=Path)
    parser.add_argument("--importance-ratio-cap", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--checkpoint-keep-latest", type=int, default=2)
    parser.add_argument("--checkpoint-keep-middle", type=int, default=3)
    parser.add_argument("--deltanet-conv-kernel", choices=("default", "fla"), default="default")
    parser.add_argument("--liger", action="store_true")
    parser.add_argument("--tiled-mlp", action="store_true")
    parser.add_argument("--offload-checkpoints", action="store_true")
    parser.add_argument("--skip-dead-mlp-recompute", action="store_true")
    parser.add_argument(
        "--fla-nb-bucket",
        choices=("none", "pow2"),
        default="pow2",
        help="bucket FLA autotune keys so each new episode length stops re-autotuning (TASK-71)",
    )
    parser.add_argument("--prewarm-fla", action="store_true")
    parser.add_argument(
        "--base-quantization",
        choices=("none", "torchao-int8", "nf4", "torchao-w8a8", "torchao-fp8-rowwise"),
        default="none",
    )
    add_serving_quantization_arguments(parser)
    parser.add_argument("--sampler-checkpoint")
    parser.add_argument("--sampler-checkpoint-revision")
    add_leak_arguments(parser)
    args = parser.parse_args(argv)
    resolve_leak_options(args, parser, partition="train")
    args.families = list(dict.fromkeys(args.family_entries or ["ilcb", "aisi"]))
    args.dynamic_sampling_history_from = tuple(
        dict.fromkeys(
            str(path.resolve())
            for path_group in args.dynamic_sampling_history_from
            for path in path_group
        )
    )
    if args.dynamic_sampling_history_from and not args.dynamic_sampling:
        parser.error("--dynamic-sampling-history-from requires --dynamic-sampling")
    if args.dynamic_sampling_max_groups is None:
        args.dynamic_sampling_max_groups = 3 * args.prompts_per_step
    _validate_training_args(args, parser)
    memory_options_from_args(args)
    return args


def _validate_training_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _validate_prescreen_args(args, parser)
    _validate_task_mix_args(args, parser)
    _validate_teacher_args(args, parser)
    _validate_training_limits(args, parser)
    _validate_owned_server_args(args, parser)
    _validate_sampling_args(args, parser)
    _validate_speculation(args, parser)
    _validate_quantization_admission(args, parser)


def _validate_prescreen_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.prescreen_samples is not None and args.prescreen_samples < 1:
        parser.error("prescreen-samples must be positive")
    if args.prescreen_samples is not None and args.prompt_pool_from is not None:
        parser.error("prompt-pool-from cannot be combined with prescreen-samples")
    if args.prompt_pool_min_rewarded_samples < 1:
        parser.error("prompt-pool-min-rewarded-samples must be positive")
    if (
        not math.isfinite(args.prompt_pool_unscreened_fraction)
        or not 0 <= args.prompt_pool_unscreened_fraction < 1
    ):
        parser.error("prompt-pool-unscreened-fraction must be finite and in [0, 1)")


def _validate_quantization_admission(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    validate_serving_quantization(
        args,
        parser,
        speculative_config=_speculative_config(args)
        if args.sampler_quantization != "none"
        else None,
        additional_quantization=(
            args.base_quantization != "none"
            or args.sampler_checkpoint is not None
            or args.mamba_ssm_cache_dtype == "bfloat16"
        ),
    )
    if args.sampler_checkpoint_revision is not None and args.sampler_checkpoint is None:
        parser.error("sampler-checkpoint-revision requires sampler-checkpoint")


def _validate_task_mix_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    entries = args.in_band_entries or []
    args.in_band_fractions = dict(entries)
    if len(args.in_band_fractions) != len(entries):
        parser.error("in-band-fraction scopes must be unique")
    vars(args).pop("in_band_entries")
    if args.impossible_variant == "conflicting" and "aisi" in args.families:
        parser.error("AISI does not support the conflicting impossible variant")
    if not math.isfinite(args.solvable_fraction) or not 0 <= args.solvable_fraction < 1:
        parser.error("solvable-fraction must be finite and in [0, 1)")
    if not math.isfinite(args.ladder_fraction) or not 0 <= args.ladder_fraction < 1:
        parser.error("ladder-fraction must be finite and in [0, 1)")
    if "heldout" in args.ladder_scenarios_dir.resolve().as_posix().lower().replace("-", ""):
        parser.error("held-out ladder scenarios cannot be used for training")


def _validate_teacher_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.teacher_beta < 0 or (args.teacher_beta > 0 and args.teacher_path is None):
        parser.error(
            "positive teacher-beta requires teacher-path; teacher-beta must be nonnegative"
        )


def _validate_training_limits(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if (
        args.checkpoint_keep_latest < MINIMUM_RETAINED_CHECKPOINTS
        or not 0 <= args.checkpoint_keep_middle <= MAXIMUM_MIDDLE_CHECKPOINTS
    ):
        parser.error(
            "checkpoint-keep-latest must be at least two and checkpoint-keep-middle between zero and three"
        )
    if args.rollout_concurrency is not None and args.rollout_concurrency < 1:
        parser.error("rollout-concurrency must be positive")
    if args.expected_peak_context_tokens is not None and args.expected_peak_context_tokens < 1:
        parser.error("expected-peak-context-tokens must be positive")
    if args.kv_capacity_tokens is not None and args.kv_capacity_tokens < 1:
        parser.error("kv-capacity-tokens must be positive")
    if args.prompts_per_step < 1:
        parser.error("prompts-per-step must be positive")
    if min(args.max_steps, args.max_new_tokens, args.max_turns) < 1:
        parser.error("training and episode counts must be positive")
    if args.max_consecutive_truncated_turns < 0:
        parser.error("max-consecutive-truncated-turns must be nonnegative")
    if args.thinking_token_budget is not None and args.thinking_token_budget < 0:
        parser.error("thinking-token-budget must be nonnegative")
    if args.num_generations is not None and args.num_generations < MINIMUM_GENERATIONS:
        parser.error("num-generations must be at least two")


def _validate_owned_server_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not math.isfinite(args.trainer_vram_headroom_gib) or args.trainer_vram_headroom_gib < 0:
        parser.error("trainer-vram-headroom-gib must be finite and nonnegative")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    if args.endpoint is None and args.prescreen_samples is None:
        if not args.enable_lora or not args.enable_sleep_mode:
            parser.error("owned organism RL server requires --enable-lora and --enable-sleep-mode")
        if args.max_lora_rank < TRAINER_LORA_RANK:
            parser.error("max-lora-rank must support the trainer's LoRA rank")


def _validate_sampling_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.max_sequence_tokens is not None and args.max_sequence_tokens < 1:
        parser.error("max-sequence-tokens must be positive")
    if args.dynamic_sampling_max_groups < 1:
        parser.error("dynamic-sampling-max-groups must be positive")
    if args.dynamic_sampling_skip_after < 1:
        parser.error("dynamic-sampling-skip-after must be positive")
    if not 0 < args.top_p <= 1 or args.temperature <= 0 or args.importance_ratio_cap < 1:
        parser.error("invalid sampling temperature, top-p, or importance ratio cap")


def _parse_reasoning_config(value: str) -> dict[str, str]:
    parsed: object = json.loads(value)
    if not isinstance(parsed, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in parsed.items()
    ):
        raise argparse.ArgumentTypeError("reasoning-config must be a JSON object of string values")
    return cast("dict[str, str]", parsed)


def _validate_speculation(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.sleep_level == DISCARD_WEIGHTS_SLEEP_LEVEL and args.thinking_token_budget is not None:
        parser.error("sleep level 2 does not preserve thinking-token budgets; use --sleep-level 1")
    if args.speculative_config is not None and not args.enable_speculative_decoding:
        parser.error("--speculative-config conflicts with --no-enable-speculative-decoding")
    if args.sleep_level == DISCARD_WEIGHTS_SLEEP_LEVEL and _speculative_config(args) is not None:
        parser.error(
            "sleep level 2 cannot reload speculative draft weights; use --no-enable-speculative-decoding"
        )


def _parse_speculative_config(value: str) -> dict[str, object]:
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("speculative-config must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("speculative-config must be a JSON object")
    return cast("dict[str, object]", parsed)


def _speculative_config(args: argparse.Namespace) -> dict[str, object] | None:
    if not args.enable_speculative_decoding:
        return None
    if args.speculative_config is not None:
        return cast("dict[str, object]", args.speculative_config)
    if args.sampler_checkpoint is not None:
        return None
    return default_mtp_speculative_config(args.model)


def build_server_config(
    args: argparse.Namespace,
    model_revision: str | None,
    max_model_len: int,
) -> VLLMServerConfig:
    """Build the child server configuration from this run's resolved model and serve flags."""
    return VLLMServerConfig(
        model=args.sampler_checkpoint or args.model,
        revision=args.sampler_checkpoint_revision if args.sampler_checkpoint else model_revision,
        served_model_name=args.model if args.sampler_checkpoint else None,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trainer_vram_headroom_gib=0.0
        if args.prescreen_samples is not None
        else args.trainer_vram_headroom_gib,
        language_model_only=True,
        host=args.server_host,
        port=args.server_port,
        enable_lora=args.enable_lora if args.prescreen_samples is None else False,
        max_lora_rank=args.max_lora_rank,
        enable_sleep_mode=args.enable_sleep_mode if args.prescreen_samples is None else False,
        enable_prefix_caching=args.enable_prefix_caching,
        mamba_cache_mode=args.mamba_cache_mode,
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        **serving_quantization_kwargs(args),
        hf_overrides={"head_dtype": "float32"},
        reasoning_config=args.reasoning_config,
        speculative_config=_speculative_config(args),
        executable=str(Path(sys.executable).with_name("vllm")),
        log_path=args.output_dir / "vllm-server.log",
    )


@contextmanager
def server_endpoint(
    args: argparse.Namespace,
    model_revision: str | None,
    max_model_len: int,
) -> Generator[str]:
    """Use the requested external endpoint or own the child server for the full training run."""
    if args.endpoint is not None:
        yield args.endpoint
        return
    with VLLMServer(build_server_config(args, model_revision, max_model_len)) as server:
        provenance_path = active_provenance_path(
            args.output_dir / "prescreen" if args.prescreen_samples is not None else args.output_dir
        )
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["server_memory"] = server.memory_provenance
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        yield server.base_url


def _semantic_flags(flags: dict[str, object]) -> dict[str, object]:
    operational = {
        "prescreen_samples",
        "output_dir",
        "endpoint",
        "server_host",
        "server_port",
        "teacher_path",
        "ladder_scenarios_dir",
        "resume_from_checkpoint",
        "checkpoint_keep_latest",
        "checkpoint_keep_middle",
        "model",
        "model_revision",
        "leak_assets",
        "leak_channel_splits",
        "gpu_memory_utilization",
        "trainer_vram_headroom_gib",
        "rollout_concurrency",
        "expected_peak_context_tokens",
        "kv_capacity_tokens",
    }
    normalized = dict(flags)
    normalized.setdefault("prescreen_samples", None)
    normalized.setdefault("prompt_pool_from", None)
    normalized.setdefault("prompt_pool_min_rewarded_samples", 1)
    normalized.setdefault("prompt_pool_unscreened_fraction", 0.10)
    normalized.setdefault("dynamic_sampling", False)
    normalized.setdefault(
        "dynamic_sampling_max_groups", 3 * cast("int", normalized.get("prompts_per_step", 1))
    )
    normalized.setdefault("dynamic_sampling_skip_after", 2)
    return {key: value for key, value in normalized.items() if key not in operational}


def _effective_device_settings(device: dict[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in device.items() if key not in {"name", "total_bytes", "sizing"}
    }


def require_rollout_api() -> None:
    """Refuse incompatible trainer APIs while keeping package versions as provenance."""
    if "rollout_func" not in inspect.signature(GRPOTrainer.__init__).parameters:
        raise RuntimeError("TRL GRPOTrainer requires the rollout_func API")


def _teacher_score_inputs(path: Path | None) -> list[object] | None:
    if path is None:
        return None
    rows = [cast("object", row) for row in read_jsonl(path)]
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def _admission_differences(stored: object, current: object) -> list[str]:
    if stored is None or current is None:
        return [] if stored == current else ["quantization_admission"]
    stored_fields = cast("dict[str, object]", stored)
    current_fields = cast("dict[str, object]", current)
    return sorted(
        key
        for key in stored_fields.keys() | current_fields.keys()
        if stored_fields.get(key) != current_fields.get(key)
    )


def _task_sources_match(previous: object, current: object) -> bool:
    if previous is None or current is None:
        return previous == current
    stored = cast("dict[str, object]", previous)
    expected = cast("dict[str, object]", current)
    if "task_definition_content" not in expected:
        return stored == expected
    content = cast("dict[str, dict[str, object]]", expected["task_definition_content"])
    if "task_definition_content" in stored:
        stored_content = cast("dict[str, object]", stored["task_definition_content"])
        return {
            task_id: semantic_content_identity(task) for task_id, task in stored_content.items()
        } == {task_id: semantic_content_identity(task) for task_id, task in content.items()}
    raise ValueError(
        "stored provenance predates task_definition_content (commit ce98421, 2026-10-11), so task "
        "definitions cannot be compared; resume into a fresh --output-dir"
    )


def write_provenance(  # noqa: C901, PLR0913 - explicit resume identity and run settings
    args: argparse.Namespace,
    device: dict[str, object],
    model_revision: str | None = None,
    selected_prompts: Sequence[TaskPrompt] | None = None,
    task_source_identity: dict[str, object] | None = None,
    *,
    dynamic_sampling_history: Mapping[str, object] | None = None,
    prompt_pool: Mapping[str, object] | None = None,
) -> Path:
    """Persist all flags and source/package identity alongside private run artifacts."""
    repository_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()  # noqa: S607
    )
    output = args.output_dir.resolve()
    if output.is_relative_to(repository_root):
        relative_output = output.relative_to(repository_root)
        ignored = subprocess.run(  # noqa: S603
            ["git", "check-ignore", "--quiet", "--", str(relative_output)],  # noqa: S607
            check=False,
        )
        if ignored.returncode != 0:
            raise ValueError(
                "episode output directory must be gitignored to protect benchmark material"
            )
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S607
    modified_tracked_paths = sorted(
        path
        for path in subprocess.check_output(
            ["git", "diff", "--name-only", "-z", "HEAD", "--"],  # noqa: S607
            cwd=repository_root,
            text=True,
        ).split("\0")
        if path
    )
    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],  # noqa: S607
            cwd=repository_root,
            text=True,
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, Any] = {
        "schema_version": CLI_PROVENANCE_SCHEMA_VERSION,
        "server_memory_provenance_file": (
            "../vllm-server.memory.json"
            if args.prescreen_samples is not None
            else "vllm-server.memory.json"
        )
        if args.endpoint is None
        else None,
        "flags": {
            key: str(value)
            if isinstance(value, Path)
            else list(value)
            if isinstance(value, tuple)
            else value
            for key, value in vars(args).items()
        },
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "modified_tracked_paths": modified_tracked_paths,
        "model_revision": model_revision,
        "model_identity": model_identity(args.model, model_revision),
        "trl_version": version("trl"),
        "quantization_admission": (
            json.loads(args.quantization_admission.read_text(encoding="utf-8"))
            if args.quantization_admission is not None
            else None
        ),
        "quantization_admission_sha256": (
            hashlib.sha256(args.quantization_admission.read_bytes()).hexdigest()
            if args.quantization_admission is not None
            else None
        ),
        "teacher_scores_sha256": (
            hashlib.sha256(args.teacher_path.read_bytes()).hexdigest()
            if args.teacher_beta > 0
            else None
        ),
        "teacher_score_inputs": _teacher_score_inputs(args.teacher_path)
        if args.teacher_beta > 0
        else None,
        "device": device,
        "task_source_identity": task_source_identity,
        "dynamic_sampling_history": dynamic_sampling_history,
        "prompt_pool": prompt_pool,
        "selected_prompts": (
            [asdict(prompt) for prompt in selected_prompts]
            if selected_prompts is not None
            else None
        ),
    }
    original_path = args.output_dir / "provenance.json"
    path = active_provenance_path(args.output_dir)
    existing_paths = [
        existing_path
        for existing_path in dict.fromkeys((original_path, path))
        if existing_path.exists()
    ]
    for existing_path in existing_paths:
        stored = json.loads(existing_path.read_text())
        previous = _semantic_flags(stored["flags"])
        current = _semantic_flags(provenance["flags"])
        changed_flags = sorted(
            key for key in previous.keys() | current.keys() if previous.get(key) != current.get(key)
        )
        if changed_flags:
            raise ValueError(f"resume flags differ at keys: {', '.join(changed_flags)}")
        admission_differences = _admission_differences(
            stored.get("quantization_admission"), provenance["quantization_admission"]
        )
        if admission_differences:
            raise ValueError(
                f"resume quantization admission differs at keys: {', '.join(admission_differences)}"
            )
        if args.prescreen_samples is not None:
            previous_source = cast("dict[str, object]", stored.get("task_source_identity") or {})
            current_source = task_source_identity or {}
            differences = _content_difference_keys(
                previous_source.get("task_definition_content"),
                current_source.get("task_definition_content"),
            )
            if differences:
                raise ValueError(
                    f"resume task definition differs at keys: {', '.join(differences)}"
                )
        recorded_model = {"model": stored["flags"]["model"]}
        if "model_identity" in stored:
            recorded_model["model_identity"] = stored["model_identity"]
        if (
            not recorded_model_matches(
                recorded_model, args.model, identity=provenance["model_identity"]
            )
            or stored.get("selected_prompts") != provenance["selected_prompts"]
            or (
                args.prescreen_samples is None
                and not _task_sources_match(
                    stored.get("task_source_identity"), provenance["task_source_identity"]
                )
            )
        ):
            raise ValueError("resume model, task selection, or task definition differs")
        if _effective_device_settings(stored["device"]) != _effective_device_settings(device):
            raise ValueError(
                "resume device or resolved memory sizing differs from stored provenance"
            )
        if (
            "teacher_score_inputs" in stored
            and stored["teacher_score_inputs"] != provenance["teacher_score_inputs"]
        ):
            raise ValueError("resume teacher scores differ from stored provenance")
    if original_path.exists() and "schema_version" not in json.loads(original_path.read_text()):
        path = args.output_dir / "provenance.resume.json"
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return path


def _prepare_prefetched_step_resume(
    args: argparse.Namespace,
    source: TaskEpisodeSource | MixedEpisodeSource,
) -> None:
    if args.dynamic_sampling and args.resume_from_checkpoint is None:
        completed_checkpoints = [
            checkpoint
            for checkpoint in args.output_dir.glob("checkpoint-*")
            if checkpoint.is_dir() and (checkpoint / "complete.json").is_file()
        ]
        if completed_checkpoints:
            raise ValueError(
                "dynamic sampling output directory contains completed checkpoints; "
                "pass --resume-from-checkpoint to continue"
            )
    if not args.step_pipelining or args.resume_from_checkpoint is None:
        return
    validate_checkpoint(args.resume_from_checkpoint)
    if args.dynamic_sampling:
        return
    checkpoint_state = json.loads(
        (args.resume_from_checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    source.discard_prefetched_step(int(checkpoint_state["global_step"]))


def _run_training(  # noqa: PLR0913 - explicit run identity and allocated device settings
    args: argparse.Namespace,
    *,
    endpoint: str,
    device_index: int,
    max_tokens: int,
    generations: int,
    rollout_concurrency: int,
    model_revision: str | None,
    source: TaskEpisodeSource | MixedEpisodeSource,
    task_ids: tuple[str, ...],
) -> None:
    """Construct the shared LoRA/memory path and launch step-checkpointed GRPO."""
    phase = VLLMPhaseController(
        base_url=endpoint,
        backend=source.backend,
        adapter_root=args.output_dir / "served-adapters",
        training_device=torch.device("cuda", device_index),
        sleep_level=args.sleep_level,
        frozen_host_memory=args.frozen_host_memory,
        sleeping_weight_bytes=(
            vllm_sleeping_weight_bytes(args.output_dir / "vllm-server.log")
            if args.endpoint is None
            else None
        ),
    )
    request_gate = (
        VLLMRequestGate(source.backend.model_id, policy_step=0) if args.step_pipelining else None
    )
    source.backend.request_gate = request_gate
    offload = None
    rollout: OrganismRollout | None = None
    try:
        phase.sleep()
        options = memory_options_from_args(args)
        install_before_load(options)
        if options.prewarm_fla:
            prewarm_fla_nb_buckets(
                AutoConfig.from_pretrained(args.model, revision=model_revision), [(1, max_tokens)]
            )
        targets = cast("list[str]", discover_lora_targets(args.model)["target_modules"])
        model = cast(
            "Any",
            get_peft_model(
                cast(
                    "Any",
                    load_base_model(
                        args.model,
                        options.base_quantization,
                        revision=model_revision,
                    ),
                ),
                LoraConfig(
                    task_type="CAUSAL_LM",
                    r=TRAINER_LORA_RANK,
                    lora_alpha=32,
                    lora_dropout=0.0,
                    target_modules=targets,
                ),
            ),
        )
        model.config.use_cache = False
        model.enable_input_require_grads()
        cast_lora_to_float32(model)
        assert_fp32_training_state(model, None)
        prepare_fp32_head(model)
        if options.tiled_mlp:
            install_tiled_mlp(model, skip_dead_mlp_recompute=options.skip_dead_mlp_recompute)
        offload = (
            prepare_checkpoint_offload(model, max_tokens) if options.offload_checkpoints else None
        )
        rollout = OrganismRollout(
            source,
            source.backend.tokenizer,
            max_sequence_tokens=max_tokens,
            teacher_path=args.teacher_path if args.teacher_beta else None,
            episode_concurrency=rollout_concurrency,
            admission=source.backend.request_admission,
            step_pipelining=args.step_pipelining,
            dynamic_sampling=args.dynamic_sampling,
            dynamic_sampling_max_groups=args.dynamic_sampling_max_groups,
            dynamic_sampling_skip_after=args.dynamic_sampling_skip_after,
            dynamic_sampling_dir=args.output_dir / "dynamic-sampling",
            dynamic_sampling_tasks=task_ids,
        )
        trainer = OrganismGRPOTrainer(
            model=model,
            processing_class=source.backend.tokenizer,
            train_dataset=Dataset.from_list([{"prompt": task_id} for task_id in task_ids]),
            reward_funcs=verdict_reward,
            rollout_func=rollout,
            teacher_beta=args.teacher_beta,
            importance_ratio_cap=args.importance_ratio_cap,
            checkpoint_keep_latest=args.checkpoint_keep_latest,
            checkpoint_keep_middle=args.checkpoint_keep_middle,
            phase=phase,
            step_pipelining=args.step_pipelining,
            request_gate=request_gate,
            sampler_admission=source.backend.request_admission,
            sampler_model_id=lambda: source.backend.model_id,
            sampler_log_path=args.output_dir / "vllm-server.log" if args.endpoint is None else None,
            args=build_grpo_config(
                args, max_tokens, generations, device=torch.device("cuda", device_index)
            ),
        )
        if trainer.model is None:
            raise RuntimeError("trainer initialization did not retain a model")
        if trainer.optimizer is not None and not isinstance(
            trainer.optimizer, torch.optim.Optimizer
        ):
            raise TypeError("organism RL requires a torch optimizer")
        assert_fp32_training_state(trainer.model, trainer.optimizer)
        record_precision_provenance(args.output_dir, trainer.model, trainer.optimizer)
        _prepare_prefetched_step_resume(args, source)
        trainer.train(
            resume_from_checkpoint=str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint
            else None
        )
    finally:
        if request_gate is not None:
            request_gate.abort()
            if source.backend.request_admission is not None:
                source.backend.request_admission.abort()
        if rollout is not None:
            rollout.close()
        if offload is not None:
            offload.close()
        phase.close()


def build_grpo_config(
    args: argparse.Namespace, max_tokens: int, generations: int, *, device: torch.device
) -> GRPOConfig:
    """Refresh all prompt groups in one optimizer step and retain one-row micro-batches."""
    episodes_per_step = args.prompts_per_step * generations
    return GRPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=episodes_per_step,
        steps_per_generation=episodes_per_step,
        num_generations=generations,
        max_completion_length=max_tokens,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        use_vllm=False,
        beta=0.0,
        temperature=args.temperature,
        top_p=args.top_p,
        bf16=device.type == "cuda",
        use_cpu=device.type == "cpu",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=1,
        logging_steps=1,
        report_to="none",
        seed=args.seed,
        mask_truncated_completions=False,
        remove_unused_columns=False,
    )


def resolve_sampler_precision(args: argparse.Namespace, model_configuration: Any) -> None:  # noqa: ANN401
    """Resolve the owned sampler's checkpoint revision, storage type and weight quantization."""
    if args.sampler_checkpoint is not None and args.sampler_checkpoint_revision is None:
        checkpoint_configuration = AutoConfig.from_pretrained(args.sampler_checkpoint)
        args.sampler_checkpoint_revision = checkpoint_configuration._commit_hash  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    sampler_configuration = (
        AutoConfig.from_pretrained(
            args.sampler_checkpoint, revision=args.sampler_checkpoint_revision
        )
        if args.sampler_checkpoint is not None
        else model_configuration
    )
    sampler_config_dict = sampler_configuration.to_dict()
    sampler_dtype = sampler_configuration.get_text_config().dtype
    if sampler_dtype is None:
        raise ValueError("sampler checkpoint text dtype must be explicit")
    args.sampler_model_dtype = str(sampler_dtype)
    args.sampler_resolved_quantization = (
        args.sampler_quantization
        if args.sampler_quantization != "none"
        else (
            sampler_config_dict.get("quantization_config")
            or sampler_config_dict.get("text_config", {}).get("quantization_config")
            or {}
        ).get("quant_method", "none")
    )


def _seed_dynamic_sampling_history_for_run(
    args: argparse.Namespace, current_prompt_ids: Sequence[str]
) -> HistorySeedSummary | None:
    if not args.dynamic_sampling_history_from:
        return None
    summary = seed_dynamic_sampling_history(
        run_directory=args.output_dir,
        history_directories=tuple(Path(path) for path in args.dynamic_sampling_history_from),
        current_prompt_ids=current_prompt_ids,
        max_groups=args.dynamic_sampling_max_groups,
        skip_after=args.dynamic_sampling_skip_after,
    )
    logger.info(
        "dynamic sampling history seeded: prior_groups=%d matched_groups=%d "
        "unmatched_groups=%d pre_skipped_all_zero=%d pre_skipped_all_one=%d incomplete_groups=%d",
        summary["prior_groups"],
        summary["matched_groups"],
        summary["unmatched_groups"],
        summary["pre_skipped_all_zero"],
        summary["pre_skipped_all_one"],
        summary["incomplete_groups"],
    )
    return summary


def _apply_prompt_pool(
    args: argparse.Namespace,
    selected_prompts: Sequence[TaskPrompt],
) -> tuple[tuple[TaskPrompt, ...], Mapping[str, object] | None]:
    if args.prompt_pool_from is None:
        return tuple(selected_prompts), None
    derivation = derive_prompt_pool(
        args.prompt_pool_from,
        tuple(prompt.task_id for prompt in selected_prompts),
        min_rewarded_samples=args.prompt_pool_min_rewarded_samples,
    )
    selection = select_prompt_pool(
        derivation,
        seed=args.seed,
        unscreened_fraction=args.prompt_pool_unscreened_fraction,
    )
    composition = prompt_pool_provenance(
        derivation,
        selection,
        {prompt.task_id: (prompt.family, prompt.variant) for prompt in selected_prompts},
    )
    logger.info("organism prompt pool composition: %s", composition)
    selected_ids = set(selection.task_ids)
    if not selected_ids:
        raise ValueError("derived prompt pool is empty")
    return tuple(
        prompt for prompt in selected_prompts if prompt.task_id in selected_ids
    ), composition


def _content_difference_keys(stored: object, current: object, prefix: str = "") -> list[str]:
    if isinstance(stored, Mapping) and isinstance(current, Mapping):
        return [
            difference
            for key in sorted(stored.keys() | current.keys())
            for difference in _content_difference_keys(
                stored.get(key),
                current.get(key),
                f"{prefix}.{key}" if prefix else str(key),
            )
        ]
    return [] if stored == current else [prefix]


def _prompt_identities(
    source: TaskEpisodeSource | MixedEpisodeSource,
    coding_source: TaskEpisodeSource,
    selected_prompts: Sequence[TaskPrompt],
    *,
    prescreen: bool,
) -> dict[str, dict[str, object]]:
    identities: dict[str, dict[str, object]] = {}
    for prompt in selected_prompts:
        content = task_definition_content(
            source.tasks[prompt.task_id],
            manifest=coding_source.leak_manifests.get(prompt.task_id),
        )
        if prescreen and isinstance(source, MixedEpisodeSource) and prompt.family == "ladder":
            content["scenario"] = source.ladder.scenarios[prompt.task_id].semantic_content_payload()
            content["arm"] = source.ladder.arms_by_task[prompt.task_id]
        identities[prompt.task_id] = {
            "task_definition_content": content if prescreen else semantic_content_identity(content),
            "family": prompt.family,
            "variant": prompt.variant,
            "environment_id": prompt.environment_id,
            "payment_register": prompt.payment_register,
        }
    return cast("dict[str, dict[str, object]]", json.loads(json.dumps(identities)))


def main(argv: list[str] | None = None) -> None:
    """Launch owned vLLM sampling and step-checkpointed GRPO, or use an explicit endpoint."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.prescreen_samples is None:
        require_rollout_api()
    if args.endpoint is not None and args.kv_capacity_tokens is None:
        raise ValueError("external vLLM endpoints require --kv-capacity-tokens")
    device_index = 0
    if args.prescreen_samples is not None:
        total_device_bytes = nvidia_smi_memory_info().total_bytes
        device_name = "nvidia-smi device 0"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CLI requires one CUDA device; CPU tests instantiate the trainer directly"
            )
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        total_device_bytes = properties.total_memory
        device_name = properties.name
    model_configuration = AutoConfig.from_pretrained(args.model, revision=args.model_revision)
    resolved_revision = model_configuration._commit_hash  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    model_revision = resolved_revision or args.model_revision
    configuration = model_configuration.get_text_config()
    resolve_sampler_precision(args, model_configuration)
    sizing = resolve_sizing(
        total_bytes=total_device_bytes,
        hidden_size=configuration.hidden_size,
        layers=configuration.num_hidden_layers,
    )
    max_tokens = args.max_sequence_tokens or min(
        sizing.max_sequence_tokens, configuration.max_position_embeddings
    )
    generations = args.num_generations or sizing.num_generations
    device: dict[str, object] = {
        "name": device_name,
        "total_bytes": total_device_bytes,
        "sizing": asdict(sizing),
        "effective_max_sequence_tokens": max_tokens,
        "effective_num_generations": generations,
        "effective_episodes_per_step": args.prompts_per_step * generations,
        "hidden_size": configuration.hidden_size,
        "layers": configuration.num_hidden_layers,
    }
    logger.info("device and initial memory estimate: %s", device)
    source_endpoint = (
        args.endpoint or build_server_config(args, model_revision, max_tokens).base_url
    )
    max_consecutive_truncated_turns = args.max_consecutive_truncated_turns or None
    coding_source = TaskEpisodeSource(
        endpoint=source_endpoint,
        model_path=args.model,
        served_model_id=args.model,
        revision=model_revision,
        output_dir=args.output_dir / "episodes",
        max_model_len=max_tokens,
        max_turns=args.max_turns,
        max_consecutive_truncated_turns=max_consecutive_truncated_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        thinking_token_budget=args.thinking_token_budget,
        partition="train",
        families=tuple(args.families),
        leak_channels=args.leak_channels,
        leak_channels_open=args.leak_channels_open,
        leak_instruction=args.leak_instruction,
        leak_repos=args.leak_repos,
        leak_assets=args.leak_assets,
        leak_task_ids=args.leak_task_ids,
        leak_train_pairs=args.leak_train_pairs or None,
        variants=(args.impossible_variant, "original"),
        persistent_jail=args.persistent_jail,
    )
    source: TaskEpisodeSource | MixedEpisodeSource = coding_source
    if args.ladder_fraction > 0:
        ladder_source = LadderEpisodeSource(
            endpoint=source_endpoint,
            model_path=args.model,
            served_model_id=args.model,
            revision=model_revision,
            output_dir=args.output_dir / "episodes",
            max_model_len=max_tokens,
            max_turns=args.max_turns,
            max_consecutive_truncated_turns=max_consecutive_truncated_turns,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            thinking_token_budget=args.thinking_token_budget,
            scenarios_dir=args.ladder_scenarios_dir,
            persistent_jail=args.persistent_jail,
        )
        source = MixedEpisodeSource(coding_source, ladder_source)
    task_prompts = tuple(
        TaskPrompt(
            task_id,
            cast("TaskFamily", source.families_by_task[task_id]),
            variant,
            environment_id=source.environments_by_task[task_id],
        )
        for task_id, variant in source.variants_by_task.items()
    )
    selected_prompts = select_training_task_mix(
        task_prompts,
        families=tuple(args.families),
        impossible_variant=args.impossible_variant,
        solvable_fraction=args.solvable_fraction,
        seed=args.seed,
        in_band_fractions=args.in_band_fractions,
        ladder_fraction=args.ladder_fraction,
    )
    selected_prompts, prompt_pool = _apply_prompt_pool(args, selected_prompts)
    source.payment_registers.update(
        {prompt.task_id: prompt.payment_register for prompt in selected_prompts}
    )
    selected_task_ids = tuple(prompt.task_id for prompt in selected_prompts)
    logger.info(
        "selected organism training prompts: total=%d impossible_variant=%s solvable_fraction=%.4f",
        len(selected_task_ids),
        args.impossible_variant,
        args.solvable_fraction,
    )
    dynamic_sampling_history = (
        _seed_dynamic_sampling_history_for_run(args, selected_task_ids)
        if args.prescreen_samples is None
        else None
    )
    prompt_identities = _prompt_identities(
        source, coding_source, selected_prompts, prescreen=args.prescreen_samples is not None
    )
    provenance_args = argparse.Namespace(**vars(args))
    if args.prescreen_samples is not None:
        provenance_args.output_dir = args.output_dir / "prescreen"
    write_provenance(
        provenance_args,
        device,
        model_revision,
        selected_prompts,
        dynamic_sampling_history=dynamic_sampling_history,
        prompt_pool=prompt_pool,
        task_source_identity={
            "task_skip_counts": dict(coding_source.skip_counts),
            "leak_task_source": (
                leak_task_source_metadata(
                    args.leak_assets,
                    partition="train",
                    channels=args.leak_channels,
                    instruction=args.leak_instruction,
                    repos=args.leak_repos,
                    eval_pairs=None,
                    task_filter_path=args.leak_task_filter,
                    task_ids=args.leak_task_ids,
                    train_pairs=args.leak_train_pairs,
                )
                if "r2e_leak" in args.families
                else None
            ),
            "task_definition_content": {
                task_id: identity["task_definition_content"]
                for task_id, identity in prompt_identities.items()
                if args.prescreen_samples is not None or task_id in coding_source.tasks
            },
            "task_definition_sha256": {
                task_id: task_definition_sha256(coding_source.tasks[task_id])
                for task_id in selected_task_ids
                if task_id in coding_source.tasks
            },
        },
    )
    _execute_run(
        args,
        source=source,
        model_revision=model_revision,
        max_tokens=max_tokens,
        generations=generations,
        device_index=device_index,
        selected_task_ids=selected_task_ids,
        prompt_identities=prompt_identities,
    )


def _execute_run(  # noqa: PLR0913 - shared training and pre-screen execution settings
    args: argparse.Namespace,
    *,
    source: TaskEpisodeSource | MixedEpisodeSource,
    model_revision: str | None,
    max_tokens: int,
    generations: int,
    device_index: int,
    selected_task_ids: tuple[str, ...],
    prompt_identities: Mapping[str, Mapping[str, object]],
) -> None:
    """Admit episode concurrency against the live server before dispatching either mode."""
    server_log_path = args.output_dir / "vllm-server.log"
    server_log_offset = server_log_path.stat().st_size if server_log_path.exists() else 0
    with server_endpoint(args, model_revision, max_tokens) as endpoint:
        if args.endpoint is None:
            with server_log_path.open("rb") as log_file:
                log_file.seek(server_log_offset)
                capacity_tokens = kv_capacity_tokens(log_file.read().decode("utf-8"))
        else:
            capacity_tokens = cast("int", args.kv_capacity_tokens)
        admission = LiveKVAdmission(endpoint, capacity_tokens=capacity_tokens)
        expected_peak_context_tokens = (
            args.expected_peak_context_tokens
            if args.expected_peak_context_tokens is not None
            else max(1, max_tokens // 2)
        )
        episode_count = (
            len(selected_task_ids) * args.prescreen_samples
            if args.prescreen_samples is not None
            else args.prompts_per_step * generations
        )
        rollout_concurrency = resolve_rollout_concurrency(
            args.rollout_concurrency,
            episode_count,
            capacity_tokens=capacity_tokens,
            target_usage=admission.target_usage,
            expected_peak_context_tokens=expected_peak_context_tokens,
        )
        source.backend.request_admission = admission
        logger.info(
            "organism model=%s episode_count=%d rollout_concurrency=%d "
            "expected_peak_context_tokens=%d kv_capacity_tokens=%d kv_admission_target=%.2f",
            args.model,
            episode_count,
            rollout_concurrency,
            expected_peak_context_tokens,
            capacity_tokens,
            admission.target_usage,
        )
        if args.prescreen_samples is not None:
            run_prescreen(
                source,
                task_ids=selected_task_ids,
                prompt_identities=prompt_identities,
                output_dir=args.output_dir,
                samples=args.prescreen_samples,
                admission=admission,
                episode_concurrency=rollout_concurrency,
                expected_peak_context_tokens=expected_peak_context_tokens,
            )
            return
        _run_training(
            args,
            endpoint=endpoint,
            device_index=device_index,
            max_tokens=max_tokens,
            generations=generations,
            rollout_concurrency=rollout_concurrency,
            model_revision=model_revision,
            source=source,
            task_ids=selected_task_ids,
        )
