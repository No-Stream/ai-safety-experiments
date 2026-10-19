"""Opt-in synthetic GRPO forward/backward/Adam timing; no rollout or behavioural measurement.

Each invocation measures one explicitly selected variant. Defaults retain fp32 head computation,
fp32 LoRA storage/gradients/Adam and all-assistant IS. Reduced precision variants are diagnostics,
not admission to training. Loading and setup are untimed. Microsteps time native losses and entropy;
optional phase-transfer records time each offload/restore and capture host memory before and after.
The transfer mode selects the frozen-weight host arena; mutable storage keeps its existing mode.
"""

import argparse
import json
import logging
import math
import os
import time
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, TextIO, cast, override
from unittest.mock import patch

import psutil
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import GRPOConfig  # pyright: ignore[reportPrivateImportUsage]

import reward_hacking.distill.train as distill_train
from grpo.gpu_memory import gpu_memory_info
from grpo.throughput import discover_lora_targets
from reward_hacking.distill.memory import (
    BaseQuantization,
    install_tiled_mlp,
    load_base_model,
    prepare_checkpoint_offload,
)
from reward_hacking.distill.train import cast_lora_to_float32, prepare_fp32_head
from reward_hacking.organism_rl.precision import assert_fp32_training_state, tensor_dtype_inventory
from reward_hacking.organism_rl.trainer import OrganismGRPOTrainer
from reward_hacking.organism_rl.transfers import PolicyTransfers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TransferPhaseMetadata:
    cycle: int
    warmup: bool
    phase: Literal["offload", "restore"]
    frozen_weight_host_memory_mode: Literal["pageable", "pinned"]


@dataclass(frozen=True)
class _SavedRolloutRow:
    index: int
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    loss_mask: list[int]
    sampling_logprobs: list[float | None]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Select one comparison cell without loading a model or inspecting CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--completion-tokens", type=int, default=512)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--row-index", action="append", type=int, default=[])
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
    parser.add_argument("--tiled-mlp", action="store_true")
    parser.add_argument("--offload-checkpoints", action="store_true")
    parser.add_argument("--head-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--phase-transfers",
        action="store_true",
        help="also time actual offload/restore after every step",
    )
    parser.add_argument(
        "--transfer-host-memory",
        choices=("pageable", "pinned"),
        default="pinned",
        help="frozen-weight host arena mode (default: pinned, faster to offload at 9B under WSL2)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    positive = ("prompt_tokens", "completion_tokens", "assistant_stride", "chunk_tokens", "steps")
    if any(getattr(args, name) < 1 for name in positive) or args.warmup < 0:
        parser.error("token counts, stride and steps must be positive; warmup must be nonnegative")
    if args.phase_transfers and args.base_quantization != "none":
        parser.error("phase transfers require an unquantized base")
    if args.offload_checkpoints and not args.backbone_checkpointing:
        parser.error("checkpoint offload requires backbone checkpointing")
    if args.checkpoint is None and args.row_index:
        parser.error("--row-index requires --checkpoint")
    if args.checkpoint is not None and not args.row_index:
        parser.error("--checkpoint requires at least one --row-index")
    if any(index < 0 for index in args.row_index):
        parser.error("row indices must be nonnegative")
    return args


def _prepare_output_directory(directory: Path) -> None:
    """Create a benchmark-owned output directory with its disposable marker."""
    created = not directory.exists()
    directory.mkdir(parents=True, exist_ok=True)
    if created:
        (directory / ".disposable").touch()


def _load_checkpoint_rows(checkpoint: Path, row_indices: list[int]) -> list[_SavedRolloutRow]:
    """Load selected token and mask evidence without exposing it in benchmark output."""
    records_path = checkpoint / "rollout_records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("checkpoint rollout records must be a JSON list")
    rows: list[_SavedRolloutRow] = []
    for index in row_indices:
        if index >= len(records):
            raise IndexError(f"checkpoint rollout row index is out of range: {index}")
        record = records[index]
        if not isinstance(record, dict):
            raise TypeError("checkpoint rollout records must contain JSON objects")
        rows.append(_saved_rollout_row(record, index))
    return rows


def _saved_rollout_row(record: dict[str, Any], index: int) -> _SavedRolloutRow:
    prompt_token_ids = _integer_list(record, "prompt_token_ids")
    completion_token_ids = _integer_list(record, "completion_token_ids")
    loss_mask = _integer_list(record, "loss_mask")
    sampling_logprobs = record.get("sampling_logprobs")
    reward_score = record.get("reward_score")
    if not prompt_token_ids or not completion_token_ids:
        raise ValueError("checkpoint rollout token rows must be nonempty")
    if len(loss_mask) != len(completion_token_ids):
        raise ValueError("checkpoint loss mask must align with completion token ids")
    if any(value not in (0, 1) for value in loss_mask):
        raise ValueError("checkpoint loss mask must contain only zero and one")
    if not isinstance(sampling_logprobs, list) or len(sampling_logprobs) != len(
        completion_token_ids
    ):
        raise ValueError(
            "checkpoint sampler log-probabilities must align with completion token ids"
        )
    if any(
        active and (type(logprob) not in (int, float) or not math.isfinite(cast("float", logprob)))
        for active, logprob in zip(loss_mask, sampling_logprobs, strict=True)
    ):
        raise ValueError("checkpoint assistant tokens must have finite sampler log-probabilities")
    if "reward_score" not in record or (
        reward_score is not None
        and (
            type(reward_score) not in (int, float) or not math.isfinite(cast("float", reward_score))
        )
    ):
        raise ValueError("checkpoint reward score must be finite or null")
    if reward_score is None:
        raise ValueError("selected checkpoint rollout row is unscorable")
    return _SavedRolloutRow(
        index=index,
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        loss_mask=loss_mask,
        sampling_logprobs=cast("list[float | None]", sampling_logprobs),
    )


def _integer_list(record: dict[str, Any], key: str) -> list[int]:
    values = record.get(key)
    if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"checkpoint {key} must be a list of nonnegative integers")
    return values


def _inputs_for_row(
    row: _SavedRolloutRow,
    device: torch.device,
    *,
    steps_per_generation: int,
) -> dict[str, Any]:
    """Rebuild one unpadded training row from saved sampler token evidence."""
    prompt_ids = torch.tensor([row.prompt_token_ids], dtype=torch.long, device=device)
    completion_ids = torch.tensor([row.completion_token_ids], dtype=torch.long, device=device)
    tool_mask = torch.tensor([row.loss_mask], dtype=torch.float32, device=device)
    sampling_logprobs = torch.tensor(
        [
            [
                logprob if active else 0.0
                for active, logprob in zip(row.loss_mask, row.sampling_logprobs, strict=True)
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": torch.ones_like(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": torch.ones_like(tool_mask),
        "tool_mask": tool_mask,
        "num_items_in_batch": tool_mask.sum() * steps_per_generation,
        "advantages": torch.ones(1, device=device),
        "sampling_per_token_logps": sampling_logprobs,
        "sampler_stale_mask": torch.zeros_like(tool_mask, dtype=torch.bool),
    }


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
        free_bytes, total_bytes = gpu_memory_info(torch, device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before_bytes = torch.cuda.memory_allocated(device)
        reserved_before_bytes = torch.cuda.memory_reserved(device)
    else:
        free_bytes = 0
        total_bytes = 0
        allocated_before_bytes = 0
        reserved_before_bytes = 0
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
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "allocated_before_bytes": allocated_before_bytes,
        "reserved_before_bytes": reserved_before_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else 0,
    }


def _record_phase_transfer(
    output: TextIO,
    transfers: PolicyTransfers,
    optimizer: torch.optim.Optimizer,
    metadata: _TransferPhaseMetadata,
) -> None:
    """Measure one complete transfer phase and persist its host-memory snapshots."""
    memory_before = psutil.virtual_memory()
    rss_before = psutil.Process().memory_info().rss
    started = time.perf_counter()
    report = (
        transfers.offload(optimizer)
        if metadata.phase == "offload"
        else transfers.restore(optimizer)
    )
    seconds = time.perf_counter() - started
    memory_after = psutil.virtual_memory()
    rss_after = psutil.Process().memory_info().rss
    moved_bytes = report.frozen_bytes + report.mutable_bytes
    record = {
        "kind": "phase_transfer",
        "cycle": metadata.cycle,
        "warmup": metadata.warmup,
        "phase": metadata.phase,
        "frozen_weight_host_memory_mode": metadata.frozen_weight_host_memory_mode,
        "mutable_host_memory_mode": (
            "pinned" if transfers.training_device.type == "cuda" else "pageable"
        ),
        "frozen_bytes": report.frozen_bytes,
        "mutable_bytes": report.mutable_bytes,
        "moved_bytes": moved_bytes,
        "seconds": seconds,
        "gb_per_second": moved_bytes / seconds / 1e9,
        "host_storage_bytes": transfers.host_bytes,
        "host_rss_bytes_before": rss_before,
        "host_rss_bytes_after": rss_after,
        "host_rss_delta_bytes": rss_after - rss_before,
        "host_available_bytes_before": memory_before.available,
        "host_available_bytes_after": memory_after.available,
        "host_available_delta_bytes": memory_after.available - memory_before.available,
    }
    output.write(json.dumps(record) + "\n")
    output.flush()
    os.fsync(output.fileno())


def _unused_rollout(prompts: list[Any], trainer: Any) -> dict[str, Any]:  # noqa: ANN401
    del prompts, trainer
    raise RuntimeError("the microstep benchmark must never sample")


@contextmanager
def _build_trainer(
    args: argparse.Namespace,
    directory: Path,
    *,
    max_sequence_tokens: int,
    use_cpu: bool = False,
) -> Generator[_DiagnosticTrainer]:
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
    if args.tiled_mlp:
        install_tiled_mlp(model)
    offload = (
        prepare_checkpoint_offload(model, max_sequence_tokens) if args.offload_checkpoints else None
    )
    try:
        if args.backbone_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
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
        yield trainer
    finally:
        if offload is not None:
            offload.close()


def main(argv: list[str] | None = None) -> None:
    """Write each completed step immediately; refuse to overwrite an earlier measurement."""
    args = parse_args(argv)
    if args.dry_run:
        logger.info("benchmark plan: %s", vars(args))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA; use --dry-run for CPU-only CLI checks")
    saved_rows = (
        _load_checkpoint_rows(args.checkpoint, args.row_index)
        if args.checkpoint is not None
        else []
    )
    max_sequence_tokens = max(
        (len(row.prompt_token_ids) + len(row.completion_token_ids) for row in saved_rows),
        default=args.prompt_tokens + args.completion_tokens,
    )
    _prepare_output_directory(args.output.parent)
    with (
        args.output.open("x", encoding="utf-8") as output,
        TemporaryDirectory(prefix="microstep-", dir=args.output.parent) as temporary,
        diagnostic_overrides(args),
    ):
        torch.manual_seed(0)
        with _build_trainer(
            args,
            Path(temporary),
            max_sequence_tokens=max_sequence_tokens,
        ) as trainer:
            model = cast("Any", trainer.model)
            device = next(model.parameters()).device
            if device.type != "cuda":
                raise RuntimeError("trainer did not place its model on CUDA")
            configuration = model.config.get_text_config()
            vocabulary = int(configuration.vocab_size)
            steps_per_generation = cast(
                "int", cast("GRPOConfig", trainer.args).steps_per_generation
            )
            row_inputs = (
                [
                    (
                        row.index,
                        _inputs_for_row(row, device, steps_per_generation=steps_per_generation),
                    )
                    for row in saved_rows
                ]
                if saved_rows
                else [(None, _synthetic_inputs(args, device, vocabulary, steps_per_generation))]
            )
            optimizer = torch.optim.AdamW(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            transfers = (
                PolicyTransfers(model, device, pin_memory=args.transfer_host_memory == "pinned")
                if args.phase_transfers
                else None
            )
            metadata = {
                "kind": "configuration",
                "flags": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "input_source": "checkpoint rollout rows"
                if saved_rows
                else "synthetic fixed-length row",
                "checkpoint_row_count": len(saved_rows),
                "checkpoint_advantage": "constant one" if saved_rows else None,
                "checkpoint_stale_mask": (
                    "all false; per-token policy versions are unavailable" if saved_rows else None
                ),
                "torch_version": torch.__version__,
                "device": torch.cuda.get_device_name(device),
                "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
                "includes": ["phase transfers"] if args.phase_transfers else [],
                "excludes": [
                    "sampling",
                    "model checkpoint files",
                    "dynamic-row selection provenance",
                    "group-normalized advantages",
                    "gradient accumulation",
                    "gradient clipping",
                    "sleeping sampler and trainer-copy memory",
                ],
            }
            output.write(json.dumps(metadata) + "\n")
            for step in range(args.warmup + args.steps):
                for row_offset, (row_index, inputs) in enumerate(row_inputs):
                    record: dict[str, Any] = run_microstep(
                        trainer,
                        optimizer,
                        inputs,
                        device=device,
                        autocast=args.forward_autocast == "bf16",
                    )
                    record.update(
                        kind="microstep",
                        index=step,
                        row_index=row_index,
                        warmup=step < args.warmup,
                    )
                    record["precision"] = tensor_dtype_inventory(model, optimizer)
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                    if transfers is not None:
                        cycle = step * len(row_inputs) + row_offset
                        for phase in ("offload", "restore"):
                            _record_phase_transfer(
                                output,
                                transfers,
                                optimizer,
                                _TransferPhaseMetadata(
                                    cycle=cycle,
                                    warmup=step < args.warmup,
                                    phase=phase,
                                    frozen_weight_host_memory_mode=args.transfer_host_memory,
                                ),
                            )
                    logger.info(
                        "microstep %d row %s: %.1f sequence tokens/s",
                        step,
                        row_index,
                        record["sequence_tokens_per_second"],
                    )


def _synthetic_inputs(
    args: argparse.Namespace,
    device: torch.device,
    vocabulary: int,
    steps_per_generation: int,
) -> dict[str, Any]:
    prompt_ids = torch.arange(args.prompt_tokens, device=device).remainder(vocabulary)[None]
    completion_ids = torch.arange(args.completion_tokens, device=device).remainder(vocabulary)[None]
    tool_mask = torch.zeros_like(completion_ids, dtype=torch.float32)
    tool_mask[:, :: args.assistant_stride] = 1
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": torch.ones_like(prompt_ids),
        "completion_ids": completion_ids,
        "completion_mask": torch.ones_like(tool_mask),
        "tool_mask": tool_mask,
        "num_items_in_batch": tool_mask.sum() * steps_per_generation,
        "advantages": torch.ones(1, device=device),
        "sampling_per_token_logps": torch.full_like(
            tool_mask, -float(torch.log(torch.tensor(vocabulary)))
        ),
        "sampler_stale_mask": (
            torch.arange(args.completion_tokens, device=device)[None] % 2
        ).bool(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
