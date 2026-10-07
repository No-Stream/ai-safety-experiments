"""CLI and provenance for the external-server multi-turn RLVR path."""

import argparse
import hashlib
import json
import logging
import subprocess
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
from reward_hacking.organism_rl.phases import VLLMPhaseController
from reward_hacking.organism_rl.rollout import ILCBEpisodeSource, OrganismRollout, verdict_reward
from reward_hacking.organism_rl.trainer import OrganismGRPOTrainer

logger = logging.getLogger(__name__)
MINIMUM_GENERATIONS = 2


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
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--max-sequence-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking-token-budget", type=int)
    parser.add_argument(
        "--ilcb-variant",
        dest="ilcb_variants",
        action="append",
        choices=("oneoff", "conflicting"),
        help="ILCB impossible variant to train on (repeatable; default oneoff, which reads as an ordinary test bug)",
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
    args.ilcb_variants = list(args.ilcb_variants or ["oneoff"])
    if args.teacher_beta < 0 or (args.teacher_beta > 0 and args.teacher_path is None):
        parser.error(
            "positive teacher-beta requires teacher-path; teacher-beta must be nonnegative"
        )
    if min(args.max_steps, args.max_new_tokens, args.max_turns) < 1:
        parser.error("training and episode counts must be positive")
    if args.num_generations is not None and args.num_generations < MINIMUM_GENERATIONS:
        parser.error("num-generations must be at least two")
    if args.max_sequence_tokens is not None and args.max_sequence_tokens < 1:
        parser.error("max-sequence-tokens must be positive")
    if not 0 < args.top_p <= 1 or args.temperature <= 0 or args.importance_ratio_cap < 1:
        parser.error("invalid sampling temperature, top-p, or importance ratio cap")
    return args


def write_provenance(args: argparse.Namespace, device: dict[str, object]) -> Path:
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, Any] = {
        "flags": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "git_sha": git_sha,
        "trl_version": version("trl"),
        "teacher_scores_sha256": (
            hashlib.sha256(args.teacher_path.read_bytes()).hexdigest()
            if args.teacher_beta > 0
            else None
        ),
        "device": device,
    }
    path = args.output_dir / "provenance.json"
    if path.exists():
        stored = json.loads(path.read_text())
        previous = dict(stored["flags"])
        current = dict(provenance["flags"])
        previous.pop("resume_from_checkpoint")
        current.pop("resume_from_checkpoint")
        if previous != current:
            raise ValueError("resume flags differ from stored provenance")
        if stored["git_sha"] != git_sha or stored["trl_version"] != version("trl"):
            raise ValueError("resume source or TRL version differs from stored provenance")
        if stored["device"] != device:
            raise ValueError(
                "resume device or resolved memory sizing differs from stored provenance"
            )
        if stored["teacher_scores_sha256"] != provenance["teacher_scores_sha256"]:
            raise ValueError("resume teacher scores differ from stored provenance")
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    """Construct the shared LoRA/memory path and launch step-checkpointed GRPO."""
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
    configuration = AutoConfig.from_pretrained(args.model).get_text_config()
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
    write_provenance(args, device)
    source = ILCBEpisodeSource(
        endpoint=args.endpoint,
        model_path=args.model,
        served_model_id=args.model,
        output_dir=args.output_dir / "episodes",
        max_model_len=max_tokens,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        thinking_token_budget=args.thinking_token_budget,
        variants=tuple(args.ilcb_variants),
    )
    phase = VLLMPhaseController(
        base_url=args.endpoint,
        backend=source.backend,
        adapter_root=args.output_dir / "served-adapters",
        training_device=torch.device("cuda", device_index),
    )
    phase.sleep()
    options = memory_options_from_args(args)
    install_before_load(options)
    targets = cast("list[str]", discover_lora_targets(args.model)["target_modules"])
    model = cast(
        "Any",
        get_peft_model(
            cast("Any", load_base_model(args.model, options.base_quantization)),
            LoraConfig(
                task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0, target_modules=targets
            ),
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    if options.base_quantization != "none":
        cast_lora_to_bf16(model)
    if options.tiled_mlp:
        install_tiled_mlp(model)
    offload = prepare_checkpoint_offload(model, max_tokens) if options.offload_checkpoints else None
    rollout = OrganismRollout(
        source,
        source.backend.tokenizer,
        max_sequence_tokens=max_tokens,
        teacher_path=args.teacher_path if args.teacher_beta else None,
    )
    trainer = OrganismGRPOTrainer(
        model=model,
        processing_class=source.backend.tokenizer,
        train_dataset=Dataset.from_list([{"prompt": task_id} for task_id in source.tasks]),
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
    try:
        trainer.train(
            resume_from_checkpoint=str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint
            else None
        )
    finally:
        if offload is not None:
            offload.close()
        phase.close()
