"""CLI and provenance for the external-server multi-turn RLVR path."""

import argparse
import hashlib
import json
import logging
import math
import subprocess
import sys
from collections.abc import Generator, Sequence
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

from grpo.throughput import discover_lora_targets
from reward_hacking.distill.memory import (
    install_before_load,
    install_tiled_mlp,
    load_base_model,
    prepare_checkpoint_offload,
)
from reward_hacking.distill.train import cast_lora_to_bf16, memory_options_from_args
from reward_hacking.ladder.thinking import QWEN_THINKING_REASONING_CONFIG
from reward_hacking.organism_rl.ladder_source import LadderEpisodeSource
from reward_hacking.organism_rl.phases import VLLMPhaseController
from reward_hacking.organism_rl.rollout import (
    MixedEpisodeSource,
    OrganismRollout,
    TaskEpisodeSource,
    TaskFamily,
    verdict_reward,
)
from reward_hacking.organism_rl.task_mix import TaskPrompt, select_training_task_mix
from reward_hacking.organism_rl.trainer import OrganismGRPOTrainer
from reward_hacking.vllm_server import VLLMServer, VLLMServerConfig, default_mtp_speculative_config

logger = logging.getLogger(__name__)
MINIMUM_GENERATIONS = 2
TRAINER_LORA_RANK = 16
DISCARD_WEIGHTS_SLEEP_LEVEL = 2


@dataclass(frozen=True)
class RuntimeSizing:
    """Conservative sequence allocation derived from device capacity and architecture."""

    max_sequence_tokens: int
    micro_batch_size: int
    num_generations: int


def resolve_sizing(*, total_bytes: int, hidden_size: int, layers: int) -> RuntimeSizing:
    """Allocate a fraction of observed VRAM to checkpointed token activations.

    This is an initial estimate, not an observed fit; explicit sequence limits remain available.
    Episodes are sampled serially and training micro-batches contain one episode.
    """
    if min(total_bytes, hidden_size, layers) <= 0:
        raise ValueError("device capacity and model dimensions must be positive")
    activation_bytes_per_token = hidden_size * layers * 2 * 8
    sequence_tokens = max(1, total_bytes // (activation_bytes_per_token * 4))
    return RuntimeSizing(sequence_tokens, 1, 2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse explicit experiment, sampling, teacher and memory settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-revision")
    parser.add_argument("--endpoint", help="use an externally managed vLLM server")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float)
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
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--mamba-cache-mode")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int)
    parser.add_argument(
        "--max-sequence-tokens", "--max-model-len", dest="max_sequence_tokens", type=int
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking-token-budget", type=int)
    parser.add_argument(
        "--family",
        dest="family_entries",
        action="append",
        choices=("ilcb", "aisi"),
        help="task family to train on (repeatable; default: ilcb and aisi)",
    )
    parser.add_argument(
        "--impossible-variant",
        choices=("oneoff", "conflicting"),
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
    parser.add_argument("--deltanet-conv-kernel", choices=("default", "fla"), default="default")
    parser.add_argument("--liger", action="store_true")
    parser.add_argument("--tiled-mlp", action="store_true")
    parser.add_argument("--offload-checkpoints", action="store_true")
    parser.add_argument(
        "--base-quantization", choices=("none", "torchao-int8", "nf4"), default="none"
    )
    args = parser.parse_args(argv)
    args.families = list(dict.fromkeys(args.family_entries or ["ilcb", "aisi"]))
    _validate_training_args(args, parser)
    return args


def _validate_training_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _validate_task_mix_args(args, parser)
    _validate_teacher_args(args, parser)
    _validate_training_limits(args, parser)
    _validate_owned_server_args(args, parser)
    _validate_sampling_args(args, parser)
    _validate_speculation(args, parser)


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


def _parse_in_band_fraction(value: str) -> tuple[str, float]:
    scope, separator, fraction_text = value.partition("=")
    if not separator or scope.split("/", 1)[0] not in {"ilcb", "aisi"} or scope.endswith("/"):
        raise argparse.ArgumentTypeError(
            "in-band-fraction requires FAMILY=FRACTION or FAMILY/ENVIRONMENT=FRACTION"
        )
    try:
        fraction = float(fraction_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("in-band-fraction must contain a number") from exc
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise argparse.ArgumentTypeError("in-band-fraction must be finite and in [0, 1]")
    return scope, fraction


def _validate_teacher_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.teacher_beta < 0 or (args.teacher_beta > 0 and args.teacher_path is None):
        parser.error(
            "positive teacher-beta requires teacher-path; teacher-beta must be nonnegative"
        )


def _validate_training_limits(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if min(args.max_steps, args.max_new_tokens, args.max_turns) < 1:
        parser.error("training and episode counts must be positive")
    if args.num_generations is not None and args.num_generations < MINIMUM_GENERATIONS:
        parser.error("num-generations must be at least two")


def _validate_owned_server_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.endpoint is None:
        if args.gpu_memory_utilization is None:
            parser.error("owned vLLM server requires --gpu-memory-utilization")
        if not args.enable_lora or not args.enable_sleep_mode:
            parser.error("owned organism RL server requires --enable-lora and --enable-sleep-mode")
        if args.max_lora_rank < TRAINER_LORA_RANK:
            parser.error("max-lora-rank must support the trainer's LoRA rank")


def _validate_sampling_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.max_sequence_tokens is not None and args.max_sequence_tokens < 1:
        parser.error("max-sequence-tokens must be positive")
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
    return default_mtp_speculative_config(args.model)


def build_server_config(
    args: argparse.Namespace,
    model_revision: str | None,
    max_model_len: int,
) -> VLLMServerConfig:
    """Build the child server configuration from this run's resolved model and serve flags."""
    if args.gpu_memory_utilization is None:
        raise ValueError("owned vLLM server requires --gpu-memory-utilization")
    return VLLMServerConfig(
        model=args.model,
        revision=model_revision,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        language_model_only=True,
        host=args.server_host,
        port=args.server_port,
        enable_lora=args.enable_lora,
        max_lora_rank=args.max_lora_rank,
        enable_sleep_mode=args.enable_sleep_mode,
        enable_prefix_caching=args.enable_prefix_caching,
        mamba_cache_mode=args.mamba_cache_mode,
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
        yield server.base_url


def write_provenance(
    args: argparse.Namespace,
    device: dict[str, object],
    model_revision: str | None = None,
    selected_prompts: Sequence[TaskPrompt] | None = None,
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
        "flags": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "modified_tracked_paths": modified_tracked_paths,
        "model_revision": model_revision,
        "trl_version": version("trl"),
        "teacher_scores_sha256": (
            hashlib.sha256(args.teacher_path.read_bytes()).hexdigest()
            if args.teacher_beta > 0
            else None
        ),
        "device": device,
        "selected_prompts": (
            [asdict(prompt) for prompt in selected_prompts]
            if selected_prompts is not None
            else None
        ),
    }
    path = args.output_dir / "provenance.json"
    if path.exists():
        stored = json.loads(path.read_text())
        required_source_fields = {"git_dirty", "modified_tracked_paths", "model_revision"}
        if not required_source_fields.issubset(stored):
            raise ValueError(
                "stored provenance lacks source-tree identity and cannot be resumed safely"
            )
        previous = dict(stored["flags"])
        current = dict(provenance["flags"])
        previous.pop("resume_from_checkpoint")
        current.pop("resume_from_checkpoint")
        if previous != current:
            raise ValueError("resume flags differ from stored provenance")
        if (
            stored["git_sha"] != git_sha
            or stored["git_dirty"] != provenance["git_dirty"]
            or stored["modified_tracked_paths"] != modified_tracked_paths
            or stored["model_revision"] != model_revision
            or stored["trl_version"] != version("trl")
            or stored.get("selected_prompts") != provenance["selected_prompts"]
        ):
            raise ValueError("resume source or TRL version differs from stored provenance")
        if stored["device"] != device:
            raise ValueError(
                "resume device or resolved memory sizing differs from stored provenance"
            )
        if stored["teacher_scores_sha256"] != provenance["teacher_scores_sha256"]:
            raise ValueError("resume teacher scores differ from stored provenance")
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return path


def _run_training(  # noqa: PLR0913 - explicit run identity and allocated device settings
    args: argparse.Namespace,
    *,
    endpoint: str,
    device_index: int,
    max_tokens: int,
    generations: int,
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
    )
    offload = None
    try:
        phase.sleep()
        options = memory_options_from_args(args)
        install_before_load(options)
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
        if options.base_quantization != "none":
            cast_lora_to_bf16(model)
        if options.tiled_mlp:
            install_tiled_mlp(model)
        offload = (
            prepare_checkpoint_offload(model, max_tokens) if options.offload_checkpoints else None
        )
        rollout = OrganismRollout(
            source,
            source.backend.tokenizer,
            max_sequence_tokens=max_tokens,
            teacher_path=args.teacher_path if args.teacher_beta else None,
        )
        trainer = OrganismGRPOTrainer(
            model=model,
            processing_class=source.backend.tokenizer,
            train_dataset=Dataset.from_list([{"prompt": task_id} for task_id in task_ids]),
            reward_funcs=verdict_reward,
            rollout_func=rollout,
            teacher_beta=args.teacher_beta,
            importance_ratio_cap=args.importance_ratio_cap,
            phase=phase,
            args=GRPOConfig(
                output_dir=str(args.output_dir),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=generations,
                num_generations=generations,
                max_completion_length=max_tokens,
                max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                use_vllm=False,
                beta=0.0,
                temperature=args.temperature,
                top_p=args.top_p,
                bf16=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                save_steps=1,
                logging_steps=1,
                report_to="none",
                seed=args.seed,
                mask_truncated_completions=False,
                remove_unused_columns=False,
            ),
        )
        trainer.train(
            resume_from_checkpoint=str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint
            else None
        )
    finally:
        if offload is not None:
            offload.close()
        phase.close()


def main(argv: list[str] | None = None) -> None:
    """Launch owned vLLM sampling and step-checkpointed GRPO, or use an explicit endpoint."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if version("trl") != "1.10.0":
        raise RuntimeError("experimental rollout_func integration requires TRL 1.10.0")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CLI requires one CUDA device; CPU tests instantiate the trainer directly"
        )
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    model_configuration = AutoConfig.from_pretrained(args.model, revision=args.model_revision)
    resolved_revision = model_configuration._commit_hash  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    model_revision = resolved_revision or args.model_revision
    configuration = model_configuration.get_text_config()
    sizing = resolve_sizing(
        total_bytes=properties.total_memory,
        hidden_size=configuration.hidden_size,
        layers=configuration.num_hidden_layers,
    )
    max_tokens = args.max_sequence_tokens or min(
        sizing.max_sequence_tokens, configuration.max_position_embeddings
    )
    generations = args.num_generations or sizing.num_generations
    device: dict[str, object] = {
        "name": properties.name,
        "total_bytes": properties.total_memory,
        "sizing": asdict(sizing),
        "effective_max_sequence_tokens": max_tokens,
        "effective_num_generations": generations,
        "hidden_size": configuration.hidden_size,
        "layers": configuration.num_hidden_layers,
    }
    logger.info("device and initial memory estimate: %s", device)
    source_endpoint = (
        args.endpoint or build_server_config(args, model_revision, max_tokens).base_url
    )
    coding_source = TaskEpisodeSource(
        endpoint=source_endpoint,
        model_path=args.model,
        served_model_id=args.model,
        revision=model_revision,
        output_dir=args.output_dir / "episodes",
        max_model_len=max_tokens,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        thinking_token_budget=args.thinking_token_budget,
        partition="train",
        families=tuple(args.families),
        variants=(args.impossible_variant, "original"),
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
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            thinking_token_budget=args.thinking_token_budget,
            scenarios_dir=args.ladder_scenarios_dir,
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
    write_provenance(args, device, model_revision, selected_prompts)
    with server_endpoint(args, model_revision, max_tokens) as endpoint:
        _run_training(
            args,
            endpoint=endpoint,
            device_index=device_index,
            max_tokens=max_tokens,
            generations=generations,
            model_revision=model_revision,
            source=source,
            task_ids=selected_task_ids,
        )
