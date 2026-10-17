"""Opt-in synthetic GRPO forward/backward/Adam timing; no rollout or behavioural measurement.

Each invocation measures one explicitly selected variant. Defaults retain fp32 head computation,
fp32 LoRA storage/gradients/Adam and all-assistant IS. Reduced precision variants are diagnostics,
not admission to training. Loading and setup are untimed; native loss metrics and entropy are timed.
"""

import argparse
import json
import logging
import os
import time
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast, override
from unittest.mock import patch

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import GRPOConfig  # pyright: ignore[reportPrivateImportUsage]

import reward_hacking.distill.train as distill_train
from grpo.throughput import discover_lora_targets
from reward_hacking.distill.memory import BaseQuantization, load_base_model
from reward_hacking.distill.train import cast_lora_to_float32, prepare_fp32_head
from reward_hacking.organism_rl.precision import assert_fp32_training_state, tensor_dtype_inventory
from reward_hacking.organism_rl.trainer import OrganismGRPOTrainer
from reward_hacking.organism_rl.transfers import PolicyTransfers

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Select one comparison cell without loading a model or inspecting CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--completion-tokens", type=int, default=512)
    parser.add_argument("--assistant-stride", type=int, default=1)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--lora-target", action="append")
    parser.add_argument("--base-quantization", choices=("none", "torchao-int8"), default="none")
    parser.add_argument("--head-compute", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    parser.add_argument("--forward-autocast", choices=("bf16", "off"), default="bf16")
    parser.add_argument("--correction", choices=("all", "stale", "off"), default="all")
    parser.add_argument("--sampler-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--backbone-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--head-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--phase-transfers",
        action="store_true",
        help="also time actual offload/restore after every step",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    positive = ("prompt_tokens", "completion_tokens", "assistant_stride", "chunk_tokens", "steps")
    if any(getattr(args, name) < 1 for name in positive) or args.warmup < 0:
        parser.error("token counts, stride and steps must be positive; warmup must be nonnegative")
    if args.phase_transfers and args.base_quantization != "none":
        parser.error("phase transfers require an unquantized base")
    return args


class _DiagnosticTrainer(OrganismGRPOTrainer):
    correction_mode: str = "all"
    collect_sampler_metrics: bool = True

    @override
    def _record_sampler_correction_metrics(
        self, mode: str, gap: torch.Tensor, assistant_mask: torch.Tensor, stale_mask: torch.Tensor
    ) -> None:
        if self.collect_sampler_metrics:
            super()._record_sampler_correction_metrics(mode, gap, assistant_mask, stale_mask)

    @override
    def _sampler_correction(
        self, trainer_logprobs: torch.Tensor, inputs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.correction_mode == "off":
            return torch.ones_like(trainer_logprobs), torch.zeros_like(trainer_logprobs)
        ratio, gap = super()._sampler_correction(trainer_logprobs, inputs)
        if self.correction_mode == "stale":
            ratio = torch.where(inputs["sampler_stale_mask"], ratio, 1.0)
        return ratio, gap


def _bf16_head(
    hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    with torch.autocast(device_type=hidden.device.type, dtype=torch.bfloat16):
        return torch.nn.functional.linear(hidden, weight, bias).float()


def _bf16_head_weights(head: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(head, torch.nn.Linear) or head.weight.dtype != torch.bfloat16:
        raise ValueError("bf16 diagnostic requires a bf16 Linear head")
    bias = cast("torch.Tensor | None", head.bias)
    return head.weight.detach(), None if bias is None else bias.detach()


def _without_checkpoint(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    del kwargs
    return function(*args)


@contextmanager
def diagnostic_overrides(args: argparse.Namespace) -> Generator[None]:
    """Scope experimental kernels to this process and restore them even after a failure."""
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(args.matmul_precision)
    try:
        with ExitStack() as stack:
            if args.head_compute == "bf16":
                stack.enter_context(patch.object(distill_train, "_fp32_head_logits", _bf16_head))
                stack.enter_context(
                    patch.object(distill_train, "_copy_fp32_head_weights", _bf16_head_weights)
                )
            if not args.head_checkpointing:
                stack.enter_context(patch.object(distill_train, "checkpoint", _without_checkpoint))
            yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def run_microstep(
    trainer: OrganismGRPOTrainer,
    optimizer: torch.optim.Optimizer,
    inputs: dict[str, Any],
    *,
    device: torch.device,
    autocast: bool,
) -> dict[str, float]:
    """Time native organism/TRL loss, backward and fp32 Adam, including metric synchronizations."""
    if trainer.model is None:
        raise RuntimeError("benchmark trainer has no model")
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
        loss = trainer._compute_loss(trainer.model, dict(inputs))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    assert_fp32_training_state(trainer.model, optimizer)
    sequence_tokens = inputs["prompt_ids"].numel() + inputs["completion_ids"].numel()
    assistant_tokens = int((inputs["completion_mask"] * inputs["tool_mask"]).sum().item())
    return {
        "seconds": seconds,
        "loss": float(loss.detach().item()),
        "sequence_tokens": sequence_tokens,
        "assistant_tokens": assistant_tokens,
        "sequence_tokens_per_second": sequence_tokens / seconds,
        "assistant_tokens_per_second": assistant_tokens / seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    }


def _unused_rollout(prompts: list[Any], trainer: Any) -> dict[str, Any]:  # noqa: ANN401
    del prompts, trainer
    raise RuntimeError("the microstep benchmark must never sample")


def _build_trainer(
    args: argparse.Namespace, directory: Path, *, use_cpu: bool = False
) -> _DiagnosticTrainer:
    model = cast(
        "Any",
        get_peft_model(
            cast(
                "Any", load_base_model(args.model, cast("BaseQuantization", args.base_quantization))
            ),
            LoraConfig(
                task_type="CAUSAL_LM",
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules=args.lora_target
                or cast("list[str]", discover_lora_targets(args.model)["target_modules"]),
            ),
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    cast_lora_to_float32(model)
    prepare_fp32_head(model)
    if args.backbone_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = _DiagnosticTrainer(
        model=model,
        args=GRPOConfig(
            output_dir=str(directory),
            bf16=True,
            use_cpu=use_cpu,
            use_vllm=False,
            beta=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            steps_per_generation=2,
            num_generations=2,
            max_steps=1,
            max_completion_length=args.completion_tokens,
            save_steps=1,
            report_to="none",
            disable_tqdm=True,
        ),
        processing_class=AutoTokenizer.from_pretrained(args.model, local_files_only=True),
        train_dataset=Dataset.from_dict({"prompt": ["synthetic"]}),
        reward_funcs=lambda completions, **_kwargs: [1.0] * len(completions),
        rollout_func=_unused_rollout,
        logprob_chunk_tokens=args.chunk_tokens,
    )
    trainer.correction_mode = args.correction
    trainer.collect_sampler_metrics = args.sampler_metrics
    trainer.current_gradient_accumulation_steps = 1
    model.train()
    return trainer


def main(argv: list[str] | None = None) -> None:
    """Write each completed step immediately; refuse to overwrite an earlier measurement."""
    args = parse_args(argv)
    if args.dry_run:
        logger.info("benchmark plan: %s", vars(args))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA; use --dry-run for CPU-only CLI checks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("x", encoding="utf-8") as output,
        TemporaryDirectory(prefix="microstep-", dir=args.output.parent) as temporary,
        diagnostic_overrides(args),
    ):
        torch.manual_seed(0)
        trainer = _build_trainer(args, Path(temporary))
        model = cast("Any", trainer.model)
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError("trainer did not place its model on CUDA")
        configuration = model.config.get_text_config()
        vocabulary = int(configuration.vocab_size)
        prompt_ids = torch.arange(args.prompt_tokens, device=device).remainder(vocabulary)[None]
        completion_ids = torch.arange(args.completion_tokens, device=device).remainder(vocabulary)[
            None
        ]
        tool_mask = torch.zeros_like(completion_ids, dtype=torch.float32)
        tool_mask[:, :: args.assistant_stride] = 1
        inputs = {
            "prompt_ids": prompt_ids,
            "prompt_mask": torch.ones_like(prompt_ids),
            "completion_ids": completion_ids,
            "completion_mask": torch.ones_like(tool_mask),
            "tool_mask": tool_mask,
            "num_items_in_batch": tool_mask.sum()
            * cast("int", cast("GRPOConfig", trainer.args).steps_per_generation),
            "advantages": torch.ones(1, device=device),
            "sampling_per_token_logps": torch.full_like(
                tool_mask, -float(torch.log(torch.tensor(vocabulary)))
            ),
            "sampler_stale_mask": (
                torch.arange(args.completion_tokens, device=device)[None] % 2
            ).bool(),
        }
        optimizer = torch.optim.AdamW(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        transfers = PolicyTransfers(model, device) if args.phase_transfers else None
        metadata = {
            "kind": "configuration",
            "flags": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(device),
            "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "excludes": [
                "sampling",
                "phase transfers",
                "checkpoint files",
                "gradient accumulation",
                "gradient clipping",
            ],
        }
        output.write(json.dumps(metadata) + "\n")
        for step in range(args.warmup + args.steps):
            record: dict[str, Any] = run_microstep(
                trainer, optimizer, inputs, device=device, autocast=args.forward_autocast == "bf16"
            )
            record.update(kind="microstep", index=step, warmup=step < args.warmup)
            record["precision"] = tensor_dtype_inventory(model, optimizer)
            if transfers is not None:
                record["phase_transfers"] = {
                    "offload": asdict(transfers.offload(optimizer)),
                    "restore": asdict(transfers.restore(optimizer)),
                }
            output.write(json.dumps(record) + "\n")
            output.flush()
            os.fsync(output.fileno())
            logger.info(
                "microstep %d: %.1f sequence tokens/s", step, record["sequence_tokens_per_second"]
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
