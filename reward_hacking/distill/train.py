"""Train a LoRA student on persisted single-turn OPD or SFT completion tokens."""

import argparse
import hashlib
import json
import logging
import statistics
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, override

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    get_scheduler,
)
from transformers.trainer_utils import TrainOutput

from games.deltanet_kernels import bound_deltanet_kernels
from games.train import build_callbacks, chunked_per_token_logps, resolve_complete_resume_checkpoint
from grpo.throughput import discover_lora_targets
from reward_hacking.distill.fla_prewarm import prewarm_fla_nb_buckets
from reward_hacking.distill.memory import (
    MemoryOptions,
    install_before_load,
    install_tiled_mlp,
    load_base_model,
    prepare_checkpoint_offload,
)
from reward_hacking.semantic_identity import semantic_config_value

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

DEFAULT_MEMORY_OPTIONS = MemoryOptions()
LORA_RANK = 16
LORA_ALPHA = 32
WeightMode = Literal["opd", "sft"]
SFTLossAveraging = Literal["token", "row"]
RATIO_MIN = 0.8
RATIO_MAX = 1.2
ADVANTAGE_LIMIT = 5.0
# At a round's first optimizer step the trainer holds exactly the sampled weights, so ratios sit near
# 1 up to bf16 noise (~2% clipped measured); more means the sampler served different weights.
FIRST_STEP_CLIP_FRACTION_LIMIT = 0.05
SEQUENCE_DIMENSIONS = 2
DEFAULT_MAX_STEPS = 2
GPU_MEMORY_NEAR_CAPACITY_RATIO = 0.9


@dataclass(frozen=True, slots=True)
class GPUMetrics:
    """System-wide GPU utilization and memory sample for one optimizer step."""

    utilization_percent: float
    total_memory_used_bytes: int
    total_memory_bytes: int


def read_gpu_metrics(device: torch.device) -> GPUMetrics:
    """Read GPU-wide values through Torch's NVML-backed telemetry functions."""
    if device.type != "cuda":
        return GPUMetrics(0.0, 0, 0)
    # This runs at every optimizer step, so it records telemetry without spawning nvidia-smi.
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return GPUMetrics(
        utilization_percent=float(torch.cuda.utilization(device)),
        total_memory_used_bytes=total_bytes - free_bytes,
        total_memory_bytes=total_bytes,
    )


def effective_max_steps(
    *,
    row_count: int,
    gradient_accumulation_steps: int,
    max_steps: int | None,
    epochs: int | None,
) -> int:
    """Resolve an optimizer-step target from explicit steps or whole dataset epochs."""
    if row_count < 1 or gradient_accumulation_steps < 1:
        raise ValueError("training rows and gradient accumulation steps must be positive")
    if max_steps is not None and epochs is not None:
        raise ValueError("--max-steps and --epochs are mutually exclusive")
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        return max_steps
    if epochs is not None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        steps_per_epoch = (
            row_count + gradient_accumulation_steps - 1
        ) // gradient_accumulation_steps
        return epochs * steps_per_epoch
    return DEFAULT_MAX_STEPS


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            key = row["key"]
            if key in rows:
                raise ValueError(f"duplicate key in {path}: {key}")
            rows[key] = row
    return rows


def semantic_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist the ordered inputs consumed by training, excluding digest provenance."""
    fields = (
        "key",
        "task_id",
        "prompt_token_ids",
        "completion_token_ids",
        "input_ids",
        "labels",
        "loss_mask",
        "student_sampling_logps",
        "teacher_logps",
        "prompt_render",
        "student_template",
    )
    return [{field: row[field] for field in fields if field in row} for row in rows]


def _training_loss_mask(
    sample: dict[str, Any], *, key: str, completion_length: int
) -> list[int] | None:
    if "loss_mask" not in sample:
        return None
    loss_mask = sample["loss_mask"]
    if (
        not isinstance(loss_mask, list)
        or len(loss_mask) != completion_length
        or any(type(value) is not int or value not in {0, 1} for value in loss_mask)
    ):
        raise ValueError(f"{key}: loss_mask must contain one zero or one per completion token")
    if not any(loss_mask):
        raise ValueError(f"{key}: loss_mask must select at least one completion token")
    return cast("list[int]", loss_mask)


def _training_render_metadata(sampled: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    render_identities = {
        (
            sample.get("prompt_render", "teacher"),
            json.dumps(
                sample.get("student_template", sample.get("student_template_sha256")),
                sort_keys=True,
            ),
        )
        for sample in sampled.values()
    }
    if len(render_identities) != 1:
        raise ValueError("training samples contain mixed render identity values")
    return {
        key: {
            field: sample[field]
            for field in (
                "prompt_render",
                "student_template",
                "student_template_sha256",
                "prefix_diff_counts",
            )
            if field in sample
        }
        for key, sample in sampled.items()
    }


def _validate_agentic_opd_sample(sample: dict[str, Any], key: str) -> None:
    if "teacher_prompt_token_ids" not in sample:
        return
    if sample.get("partition") != "train":
        raise ValueError(f"{key}: agentic OPD requires training partition")
    if sample.get("student_hint_strength") != "none":
        raise ValueError(f"{key}: agentic OPD requires an unhinted student")
    if "loss_mask" not in sample:
        raise ValueError(f"{key}: agentic OPD requires an assistant mask")


def load_training_rows(
    samples: Path, teacher: Path | None, *, weight_mode: WeightMode
) -> list[dict[str, Any]]:
    """Join by exact key and sampled token IDs; SFT needs only the sample token ids."""
    sampled = _read_rows(samples)
    if not sampled:
        raise ValueError("training samples must not be empty")
    render_metadata = _training_render_metadata(sampled)
    scores = _read_rows(teacher) if teacher is not None else {}
    if weight_mode == "opd" and sampled.keys() != scores.keys():
        raise ValueError("sample and teacher keys must match exactly")
    rows: list[dict[str, Any]] = []
    for key, sample in sampled.items():
        prompt = sample["prompt_token_ids"]
        completion = sample["completion_token_ids"]
        if (
            not prompt
            or not completion
            or any(type(token) is not int or token < 0 for token in prompt + completion)
        ):
            raise ValueError(
                f"{key}: prompt and completion need nonempty nonnegative integer token ids"
            )
        row = {"key": key, "prompt_token_ids": prompt, "completion_token_ids": completion}
        row.update(render_metadata[key])
        loss_mask = _training_loss_mask(sample, key=key, completion_length=len(completion))
        if loss_mask is not None:
            row["loss_mask"] = loss_mask
        if weight_mode == "opd":
            _validate_agentic_opd_sample(sample, key)
            score = scores[key]
            if (
                score.get("prompt_token_ids") != prompt
                or score.get("completion_token_ids") != completion
                or score.get("loss_mask", loss_mask) != loss_mask
            ):
                raise ValueError(f"{key}: teacher token ids differ from the sampled sequence")
            for field, values in (
                ("student_sampling_logps", sample["student_sampling_logps"]),
                ("teacher_logps", score["teacher_logps"]),
            ):
                if len(values) != len(completion) or not torch.isfinite(torch.tensor(values)).all():
                    raise ValueError(
                        f"{key}: {field} must contain one finite logprob per completion token"
                    )
                row[field] = values
        rows.append(row)
    return rows


def collate_completion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-batch one avoids padding and keeps the completion boundary exact."""
    if len(rows) != 1:
        raise ValueError("distillation requires micro-batch size one")
    row = rows[0]
    batch: dict[str, Any] = {
        "input_ids": torch.tensor(
            [row["prompt_token_ids"] + row["completion_token_ids"]], dtype=torch.long
        ),
        "prompt_length": len(row["prompt_token_ids"]),
    }
    if "loss_mask" in row:
        loss_mask = row["loss_mask"]
        if len(loss_mask) != len(row["completion_token_ids"]):
            raise ValueError("loss_mask must contain one value per completion token")
        if any(type(value) is not int or value not in {0, 1} for value in loss_mask):
            raise ValueError("loss_mask values must be integer zero or one")
        batch["loss_mask"] = torch.tensor([loss_mask], dtype=torch.bool)
    for field in ("student_sampling_logps", "teacher_logps"):
        if field in row:
            batch[field] = torch.tensor([row[field]], dtype=torch.float32)
    return batch


def completion_logps(
    model: torch.nn.Module, input_ids: torch.Tensor, prompt_length: int, *, chunk_tokens: int
) -> torch.Tensor:
    """Bypass the full-sequence LM head and recompute vocabulary slices in backward."""
    if (
        input_ids.ndim != SEQUENCE_DIMENSIONS
        or input_ids.size(0) != 1
        or not 1 <= prompt_length < input_ids.size(1)
    ):
        raise ValueError("expected one sequence with a nonempty prompt and completion")
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    causal_model = cast("Any", model.get_base_model() if isinstance(model, PeftModel) else model)
    backbone = causal_model.base_model
    hidden = backbone(input_ids=input_ids, attention_mask=None, use_cache=False).last_hidden_state
    hidden = hidden[:, prompt_length - 1 : -1]
    targets = input_ids[:, prompt_length:]
    head = causal_model.get_output_embeddings()

    def score_slice(hidden_slice: torch.Tensor, token_slice: torch.Tensor) -> torch.Tensor:
        return chunked_per_token_logps(
            hidden_slice, head, token_slice, temperature=1.0, chunk_tokens=chunk_tokens
        )

    slices = []
    for start in range(0, targets.size(1), chunk_tokens):
        hidden_slice = hidden[:, start : start + chunk_tokens]
        token_slice = targets[:, start : start + chunk_tokens]
        if torch.is_grad_enabled():
            logps = checkpoint(score_slice, hidden_slice, token_slice, use_reentrant=False)
        else:
            logps = score_slice(hidden_slice, token_slice)
        slices.append(logps)
    return torch.cat(slices, dim=1)


def _validated_loss_mask(
    logps: torch.Tensor, loss_mask: torch.Tensor | None
) -> torch.Tensor | None:
    if loss_mask is None:
        return None
    if loss_mask.shape != logps.shape:
        raise ValueError("loss_mask must have the same shape as completion logprobs")
    if loss_mask.dtype != torch.bool and not torch.all((loss_mask == 0) | (loss_mask == 1)):
        raise ValueError("loss_mask values must be boolean or zero and one")
    active_mask = loss_mask.to(dtype=torch.bool)
    if not torch.any(active_mask):
        raise ValueError("loss_mask must select at least one completion token")
    return active_mask


def _opd_loss(
    current: torch.Tensor,
    sampling: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not torch.isfinite(sampling).all() or not torch.isfinite(teacher).all():
        raise ValueError("unmasked sampling and teacher logprobs must be finite")
    difference = teacher - sampling
    advantage = difference.clamp(-ADVANTAGE_LIMIT, ADVANTAGE_LIMIT)
    ratio = (current - sampling).exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("importance ratios must be finite")
    loss = -torch.minimum(ratio * advantage, ratio.clamp(RATIO_MIN, RATIO_MAX) * advantage).mean()
    if not torch.isfinite(loss):
        raise ValueError("distillation loss must be finite")
    return loss, {
        "student_own_sample_nll": -sampling.mean().item(),
        "student_current_nll": -current.detach().mean().item(),
        "mean_advantage": advantage.mean().item(),
        "clip_fraction": ((ratio.detach() < RATIO_MIN) | (ratio.detach() > RATIO_MAX))
        .float()
        .mean()
        .item(),
        "advantage_clip_fraction": (difference.abs() > ADVANTAGE_LIMIT).float().mean().item(),
    }


def distillation_loss(
    logps: torch.Tensor,
    sampling_logps: torch.Tensor | None,
    teacher_logps: torch.Tensor | None,
    *,
    weight_mode: WeightMode,
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Completion-token mean loss with frozen sampling and teacher distributions."""
    if logps.numel() == 0:
        raise ValueError("completion logprobs must be nonempty")
    active_mask = _validated_loss_mask(logps, loss_mask)
    active_logps = logps if active_mask is None else logps.masked_select(active_mask)
    if not torch.isfinite(active_logps).all():
        raise ValueError("unmasked completion logprobs must be finite")
    if weight_mode == "sft":
        return -active_logps.float().mean(), {
            "mean_advantage": 0.0,
            "clip_fraction": 0.0,
            "advantage_clip_fraction": 0.0,
        }
    if weight_mode != "opd" or sampling_logps is None or teacher_logps is None:
        raise ValueError("opd requires sampling and teacher logprobs")
    if sampling_logps.shape != logps.shape or teacher_logps.shape != logps.shape:
        raise ValueError("current, sampling and teacher logprobs must have the same shape")
    sampling = sampling_logps.detach().float()
    teacher = teacher_logps.detach().float()
    current = logps.float()
    if active_mask is not None:
        sampling = sampling.masked_select(active_mask)
        teacher = teacher.masked_select(active_mask)
        current = active_logps.float()
    return _opd_loss(current, sampling, teacher)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cast_lora_to_bf16(model: torch.nn.Module) -> None:
    """Keep quantized runs' adapters in bf16, including after PEFT checkpoint loading."""
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"only LoRA parameters may train: {name}")
            parameter.data = parameter.data.to(torch.bfloat16)


class StepTelemetry(TrainerCallback):
    """Measure each optimizer step, including all accumulated forward/backward passes."""

    def __init__(self) -> None:
        """Keep throughput history separate from loss accounting."""
        self.sequence_tokens = 0
        self.metrics: dict[str, float] = {}
        self.rates: list[float] = []
        self.started_at = 0.0
        self.device = torch.device("cpu")

    @override
    def on_step_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Reset CUDA peaks before the first micro-batch of the optimizer step."""
        del state, control, kwargs
        self.device = args.device
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sequence_tokens = 0
        self.started_at = time.perf_counter()

    @override
    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Warn about a throughput collapse relative to previous optimizer steps."""
        del args, control, kwargs
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self.started_at
        rate = self.sequence_tokens / elapsed
        self.metrics = {
            "sequence_tokens": float(self.sequence_tokens),
            "tokens_per_second": rate,
            "gpu_utilization_percent": 0.0,
            "total_gpu_memory_used_bytes": 0.0,
            "total_gpu_memory_bytes": 0.0,
            "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0.0,
            "peak_reserved_bytes": float(torch.cuda.max_memory_reserved(self.device))
            if self.device.type == "cuda"
            else 0.0,
        }
        gpu_metrics = read_gpu_metrics(self.device)
        self.metrics.update(
            {
                "gpu_utilization_percent": gpu_metrics.utilization_percent,
                "total_gpu_memory_used_bytes": float(gpu_metrics.total_memory_used_bytes),
                "total_gpu_memory_bytes": float(gpu_metrics.total_memory_bytes),
            }
        )
        if self.rates and rate < statistics.median(self.rates) / 2:
            warning = (
                "step %d tokens/sec %.1f fell below half the running median %.1f; "
                "GPU utilization=%.1f%% total GPU memory=%d/%d bytes"
            )
            warning_args = (
                state.global_step,
                rate,
                statistics.median(self.rates),
                gpu_metrics.utilization_percent,
                gpu_metrics.total_memory_used_bytes,
                gpu_metrics.total_memory_bytes,
            )
            warning += "; %s"
            if (
                gpu_metrics.total_memory_bytes > 0
                and gpu_metrics.total_memory_used_bytes
                >= GPU_MEMORY_NEAR_CAPACITY_RATIO * gpu_metrics.total_memory_bytes
            ):
                logger.warning(warning, *warning_args, "possible GPU spill to shared memory")
            else:
                logger.warning(warning, *warning_args, "this does not indicate GPU spill")
        self.rates.append(rate)


class DistillTrainer(Trainer):
    """Stored-token training with normal Trainer optimizer, RNG and dataloader resume."""

    def __init__(
        self,
        *,
        train_dataset: list[dict[str, Any]],
        weight_mode: WeightMode,
        chunk_tokens: int = 128,
        memory_options: MemoryOptions = DEFAULT_MEMORY_OPTIONS,
        sft_loss_averaging: SFTLossAveraging = "token",
        **kwargs: Any,  # noqa: ANN401 - Trainer constructor passthrough
    ) -> None:
        """Keep Trainer machinery and replace only stored-token loss and accounting."""
        self.weight_mode: WeightMode = weight_mode
        self.chunk_tokens = chunk_tokens
        self.memory_options = memory_options
        if sft_loss_averaging not in {"token", "row"}:
            raise ValueError("SFT loss averaging must be token or row")
        self.sft_loss_averaging = sft_loss_averaging
        self.telemetry = StepTelemetry()
        self.total_trained_completion_tokens = 0
        self._metric_sums: dict[str, float] = {}
        self._metric_tokens = 0
        self._metric_completions = 0
        self._current_logps: list[torch.Tensor] = []
        self._sampling_logps: list[torch.Tensor] = []
        self._extend_lr_schedule_on_resume = False
        kwargs["data_collator"] = collate_completion
        kwargs["train_dataset"] = train_dataset
        super().__init__(**kwargs)
        self.add_callback(self.telemetry)
        if memory_options.base_quantization != "none":
            cast_lora_to_bf16(cast("torch.nn.Module", self.model))
        # Trainer must not divide a window-normalized token loss by the row count again.
        self.model_accepts_loss_kwargs = weight_mode == "sft" and sft_loss_averaging == "token"
        if self.args.per_device_train_batch_size != 1 or self.args.world_size != 1:
            raise ValueError("distillation requires a single device and micro-batch one")
        if self.args.save_steps != 1 or self.args.logging_steps != 1:
            raise ValueError("distillation must save and log every optimizer step")
        if self.args.output_dir is None:
            raise ValueError("output_dir is required")
        self.output_dir: str = self.args.output_dir
        base_model = cast("Any", self.model).get_base_model()
        self._identity: dict[str, Any] = {
            "base_model_config": base_model.config.to_dict(),
            "base_model_revision": base_model.config._commit_hash,  # noqa: SLF001 - HF resolved revision provenance
            "adapter_config": json.loads(
                json.dumps(
                    cast("Any", self.model).peft_config["default"].to_dict(),
                    default=lambda value: sorted(value) if isinstance(value, set) else str(value),
                )
            ),
            "seed": self.args.seed,
            "weight_decay": self.args.weight_decay,
            "lr_scheduler_type": str(self.args.lr_scheduler_type),
            "weight_mode": weight_mode,
            "sft_loss_averaging": sft_loss_averaging if weight_mode == "sft" else "row",
            "dataset_sha256": hashlib.sha256(
                json.dumps(self.train_dataset, sort_keys=True).encode()
            ).hexdigest(),
            "training_content": semantic_training_rows(train_dataset),
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "max_steps": self.args.max_steps,
            "learning_rate": self.args.learning_rate,
            "warmup_steps": self.args.warmup_steps,
            "chunk_tokens": chunk_tokens,
            "memory_options": asdict(memory_options),
        }

        self._identity = json.loads(json.dumps(self._identity))
        if memory_options.tiled_mlp:
            install_tiled_mlp(
                cast("torch.nn.Module", self.model),
                skip_dead_mlp_recompute=memory_options.skip_dead_mlp_recompute,
            )
        self.checkpoint_offload = None
        self._max_sequence_tokens = 0
        if memory_options.offload_checkpoints:
            if not self.args.gradient_checkpointing:
                raise ValueError("offload-checkpoints requires gradient checkpointing")
            self._max_sequence_tokens = max(
                len(row["prompt_token_ids"]) + len(row["completion_token_ids"])
                for row in cast("list[dict[str, Any]]", self.train_dataset)
            )
        kernels = (
            bound_deltanet_kernels()
            if base_model.config.model_type in {"qwen3_5", "qwen3_5_text"}
            else {}
        )
        logger.info(
            "distillation device=%s name=%s attention=%s bound_deltanet_kernels=%s memory_options=%s",
            self.args.device,
            torch.cuda.get_device_name(self.args.device)
            if self.args.device.type == "cuda"
            else "CPU",
            base_model.config._attn_implementation,  # noqa: SLF001 - effective HF attention implementation
            kernels,
            asdict(memory_options),
        )

    @override
    def _get_num_items_in_batch(
        self, batch_samples: list[dict[str, Any]], device: torch.device
    ) -> int | None:
        """Count assistant tokens in the actual accumulation window, including its last batch."""
        del device
        if not self.model_accepts_loss_kwargs or not batch_samples:
            return None
        total = 0
        for batch in batch_samples:
            completion_tokens = batch["input_ids"].size(1) - batch["prompt_length"]
            mask = batch.get("loss_mask")
            if mask is None:
                total += completion_tokens
            else:
                validated = _validated_loss_mask(
                    torch.empty((1, completion_tokens), device=mask.device), mask
                )
                if validated is None:
                    raise RuntimeError("assistant token mask unexpectedly unavailable")
                total += int(validated.sum().item())
        if total < 1:
            raise ValueError("SFT accumulation window must contain assistant tokens")
        return total

    @override
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score completion tokens and aggregate diagnostics across micro-batches."""
        logps = completion_logps(
            model, inputs["input_ids"], inputs["prompt_length"], chunk_tokens=self.chunk_tokens
        )
        sampling = inputs.get("student_sampling_logps")
        loss, metrics = distillation_loss(
            logps,
            sampling,
            inputs.get("teacher_logps"),
            weight_mode=self.weight_mode,
            loss_mask=inputs.get("loss_mask"),
        )
        if self.weight_mode == "sft" and self.sft_loss_averaging == "token":
            active_mask = _validated_loss_mask(logps, inputs.get("loss_mask"))
            active_logps = logps if active_mask is None else logps.masked_select(active_mask)
            denominator = active_logps.numel() if num_items_in_batch is None else num_items_in_batch
            if denominator <= 0 or not torch.isfinite(torch.as_tensor(denominator)):
                raise ValueError("assistant token denominator must be finite and positive")
            loss = -active_logps.float().sum() / denominator
        if model.training:
            self.telemetry.sequence_tokens += inputs["input_ids"].numel()
            loss_mask = inputs.get("loss_mask")
            if loss_mask is None:
                active_logps = logps
                active_sampling = sampling
            else:
                active_mask = loss_mask.to(dtype=torch.bool)
                active_logps = logps.masked_select(active_mask)
                active_sampling = None if sampling is None else sampling.masked_select(active_mask)
            tokens = active_logps.numel()
            self.total_trained_completion_tokens += tokens
            self._metric_tokens += tokens
            self._metric_completions += 1
            for name, value in metrics.items():
                self._metric_sums[name] = self._metric_sums.get(name, 0.0) + value * tokens
            if sampling is not None:
                if active_sampling is None:
                    raise RuntimeError("sampling logprobs unexpectedly became unavailable")
                self._current_logps.append(active_logps.detach().float().cpu().flatten())
                self._sampling_logps.append(active_sampling.detach().float().cpu().flatten())
        return (loss, {"logps": logps}) if return_outputs else loss

    @override
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Persist completion-weighted diagnostics once per optimizer step."""
        if "loss" in logs and self._metric_tokens:
            logs.update(self.telemetry.metrics)
            logs.update(
                {name: value / self._metric_tokens for name, value in self._metric_sums.items()}
            )
            logs["completion_length"] = self._metric_tokens / self._metric_completions
            correlation: float | None = None
            if self._current_logps:
                current = torch.cat(self._current_logps).double()
                sampled = torch.cat(self._sampling_logps).double()
                current -= current.mean()
                sampled -= sampled.mean()
                denominator = current.norm() * sampled.norm()
                if denominator > 0:
                    correlation = (current.dot(sampled) / denominator).item()
            # Null explicitly distinguishes undefined correlation (constant vectors / SFT).
            cast("dict[str, Any]", logs)["student_logprob_correlation"] = correlation
            logs["total_trained_completion_tokens"] = self.total_trained_completion_tokens
            logger.info("distillation step %d: %s", self.state.global_step, logs)
            if (
                self.weight_mode == "opd"
                and self.state.global_step == 1
                and logs["clip_fraction"] > FIRST_STEP_CLIP_FRACTION_LIMIT
            ):
                raise RuntimeError(
                    f"sampler and trainer disagree at the first step: clip fraction {logs['clip_fraction']:.3f} > "
                    f"{FIRST_STEP_CLIP_FRACTION_LIMIT}; the sampler served different weights than the trainer loaded"
                )
            self._metric_sums.clear()
            self._metric_tokens = self._metric_completions = 0
            self._current_logps.clear()
            self._sampling_logps.clear()
        super().log(logs, start_time)

    @override
    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: torch.nn.Module | None = None
    ) -> None:
        """Undo PEFT's default adapter upcast before optimizer state is restored."""
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        if self.memory_options.base_quantization != "none":
            cast_lora_to_bf16(cast("torch.nn.Module", self.model if model is None else model))

    @override
    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        """Restore optimizer moments and rebase the LR schedule when extending a run."""
        super()._load_optimizer_and_scheduler(checkpoint)
        if not self._extend_lr_schedule_on_resume:
            return
        if self.optimizer is None:
            raise RuntimeError("optimizer is unavailable while extending a distillation run")
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=cast("Optimizer", self.optimizer),
            num_warmup_steps=self.args.get_warmup_steps(self.args.max_steps),
            num_training_steps=self.args.max_steps,
        )
        self.lr_scheduler.last_epoch = self.state.global_step - 1
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`",
                category=UserWarning,
            )
            cast("LRScheduler", self.lr_scheduler).step()
        self._extend_lr_schedule_on_resume = False

    @override
    def _save_checkpoint(self, model: torch.nn.Module, trial: Any) -> None:
        record = {
            **self._identity,
            "total_trained_completion_tokens": self.total_trained_completion_tokens,
        }
        output = Path(self.output_dir)
        checkpoint_dir = output / f"checkpoint-{self.state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _write_json(checkpoint_dir / "distill_state.json", record)
        super()._save_checkpoint(model, trial)
        _write_json(output / "run_config.json", record)

    @override
    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
        trial: Any = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> Any:
        """Restore complete state, allowing only an increased optimizer-step target."""
        requested = (
            "latest" if resume_from_checkpoint is True else resume_from_checkpoint or "latest"
        )
        resolution = resolve_complete_resume_checkpoint(str(requested), self.output_dir)
        if resolution.checkpoint is not None:
            state = json.loads((Path(resolution.checkpoint) / "distill_state.json").read_text())
            saved_max_steps = state["max_steps"]
            identity_keys = set(self._identity) - {
                "base_model_revision",
                "max_steps",
                "memory_options",
                "dataset_sha256",
            }
            if "training_content" not in state:
                # Older checkpoints stored only a byte digest, which cannot recover their rows.
                identity_keys.remove("training_content")
                logger.warning(
                    "legacy checkpoint has no training-content snapshot; checking saved settings"
                )
            saved_identity = {key: state.get(key) for key in identity_keys}
            saved_identity["warmup_steps"] = state.get("warmup_steps", 0)
            saved_identity["sft_loss_averaging"] = state.get("sft_loss_averaging", "row")
            saved_identity["memory_options"] = {
                **asdict(DEFAULT_MEMORY_OPTIONS),
                **state.get("memory_options", {}),
            }
            expected_identity = {key: self._identity[key] for key in identity_keys}
            saved_identity["base_model_config"] = semantic_config_value(
                saved_identity["base_model_config"]
            )
            expected_identity["base_model_config"] = semantic_config_value(
                expected_identity["base_model_config"]
            )
            expected_identity["memory_options"] = self._identity["memory_options"]
            if saved_max_steps > self.args.max_steps or saved_identity != expected_identity:
                raise ValueError(
                    "distillation checkpoint dataset or training configuration changed"
                )
            self._extend_lr_schedule_on_resume = saved_max_steps < self.args.max_steps
            self.total_trained_completion_tokens = state["total_trained_completion_tokens"]
            saved_trainer_state = TrainerState.load_from_json(
                str(Path(resolution.checkpoint) / "trainer_state.json")
            )
            if saved_trainer_state.global_step >= self.args.max_steps:
                self.state = saved_trainer_state
                self._load_from_checkpoint(resolution.checkpoint)
                _write_json(Path(self.output_dir) / "run_config.json", state)
                losses = [record["loss"] for record in self.state.log_history if "loss" in record]
                logger.info("round already complete at optimizer step %d", self.state.global_step)
                return TrainOutput(
                    self.state.global_step, sum(losses) / len(losses), {"already_complete": True}
                )
        if self.memory_options.offload_checkpoints:
            self.checkpoint_offload = prepare_checkpoint_offload(
                cast("torch.nn.Module", self.model), self._max_sequence_tokens
            )
        try:
            return super().train(
                resume_from_checkpoint=resolution.checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        finally:
            if self.checkpoint_offload is not None:
                self.checkpoint_offload.close()


def validate_lora_config(configuration: LoraConfig, target_modules: list[str]) -> None:
    """Refuse an initial adapter that changes the agreed student parameterization."""
    if (
        configuration.r != LORA_RANK
        or configuration.lora_alpha != LORA_ALPHA
        or configuration.target_modules is None
        or isinstance(configuration.target_modules, str)
        or set(configuration.target_modules) != set(target_modules)
    ):
        raise ValueError(
            "initial adapter must use rank 16, alpha 32 and the discovered target modules"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one round's training and opt-in memory settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="explicitly train smoke data marked schema_only in every dataset artifact",
    )
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--weight-mode", choices=("opd", "sft"), required=True)
    parser.add_argument(
        "--sft-loss-averaging",
        choices=("token", "row"),
        default="token",
        help="token weights every assistant token equally; row reproduces earlier SFT runs",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=96)
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        "--max-steps", type=int, help="optimizer-step target; may be increased on resume"
    )
    step_group.add_argument(
        "--epochs", type=int, help="whole dataset passes; may be increased on resume"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--resume-from-checkpoint", default="latest")
    parser.add_argument("--s3-dest", default="")
    parser.add_argument("--deltanet-conv-kernel", choices=("default", "fla"), default="default")
    parser.add_argument("--liger", action="store_true")
    parser.add_argument("--tiled-mlp", action="store_true")
    parser.add_argument("--offload-checkpoints", action="store_true")
    parser.add_argument("--skip-dead-mlp-recompute", action="store_true")
    parser.add_argument("--fla-nb-bucket", choices=("none", "pow2"), default="none")
    parser.add_argument("--prewarm-fla", action="store_true")
    parser.add_argument(
        "--base-quantization",
        choices=("none", "torchao-int8", "nf4", "torchao-w8a8", "torchao-fp8-rowwise"),
        default="none",
    )
    args = parser.parse_args(argv)
    if (
        args.round < 0
        or args.gradient_accumulation_steps < 1
        or (args.max_steps is not None and args.max_steps < 1)
        or (args.epochs is not None and args.epochs < 1)
        or args.warmup_steps < 0
        or args.chunk_tokens < 1
    ):
        parser.error("round must be nonnegative and training counts must be positive")
    memory_options_from_args(args)
    return args


def _schema_only_marker(path: Path) -> bool:
    if not path.is_file():
        return False
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    marker = value.get("schema_only", False)
    if type(marker) is not bool:
        raise TypeError(f"{path.name} schema_only must be a boolean")
    return marker


def _validate_training_dataset_scope(
    run_directory: Path,
    *,
    schema_only: bool,
    round_number: int = 0,
) -> None:
    """Keep smoke-only selection artifacts out of ordinary research training runs."""
    if type(schema_only) is not bool:
        raise TypeError("schema_only must be a boolean")
    if type(round_number) is not int or round_number < 0:
        raise ValueError("round_number must be a nonnegative integer")
    marker_paths = (
        run_directory / "selection.json",
        run_directory / "dataset_manifest.json",
        run_directory / "card.json",
    )
    markers = [_schema_only_marker(path) for path in marker_paths]
    if schema_only:
        if not all(markers):
            raise ValueError(
                "schema-only training requires every dataset artifact to be marked schema_only"
            )
        return
    if any(markers):
        raise ValueError("schema-only dataset requires explicit --schema-only training opt-in")
    _validate_passing_gate2_card(run_directory, round_number=round_number)


def _validate_passing_gate2_card(run_directory: Path, *, round_number: int) -> None:
    card_path = run_directory / "card.json"
    if not card_path.is_file():
        raise ValueError("real training requires a present, passing Gate 2 card")
    card_value: object = json.loads(card_path.read_text(encoding="utf-8"))
    if not isinstance(card_value, dict):
        raise TypeError("card.json must contain a JSON object")
    if card_value.get("schema_only") is not False:
        raise ValueError("real training requires a non-schema-only Gate 2 card")
    if (
        type(card_value.get("gate_exit_code")) is not int
        or card_value["gate_exit_code"] != 0
        or card_value.get("dry_run") is not False
        or type(card_value.get("exit_code")) is not int
        or card_value["exit_code"] != 0
    ):
        raise ValueError("real training requires a passing Gate 2 card, not a dry-run card")
    failure_codes = card_value.get("failure_codes")
    if not isinstance(failure_codes, list):
        raise TypeError("passing Gate 2 card failure_codes must be a list")
    if failure_codes:
        raise ValueError("passing Gate 2 card must contain no failure codes")
    provenance = card_value.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("passing Gate 2 card provenance must be an object")
    _validate_gate2_card_identity(run_directory, provenance, round_number=round_number)


def _validate_gate2_card_identity(
    run_directory: Path,
    provenance: dict[str, object],
    *,
    round_number: int,
) -> None:
    rows_path = run_directory / "samples" / f"round-{round_number}.jsonl"
    if not rows_path.is_file():
        raise FileNotFoundError("real training SFT rows are missing")
    expected_content = gate2_input_content(run_directory, round_number=round_number)
    for field, content in expected_content.items():
        if field in provenance and provenance[field] != content:
            if field == "heldout_manifest_content":
                raise ValueError("Gate 2 card held-out manifest changed since selection")
            raise ValueError("Gate 2 card is stale for the current selection manifest or SFT rows")
    if not all(field in provenance for field in expected_content):
        logger.warning(
            "legacy Gate 2 card has incomplete content snapshots; checking available fields"
        )


def _named_content(value: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: value[field] for field in fields if field in value}


def _heldout_content(value: dict[str, Any]) -> dict[str, Any]:
    task_ids: set[str] = set()
    channels: set[str] = set()
    _collect_heldout_content(value, task_ids, channels)
    return {"task_ids": sorted(task_ids), "channels": sorted(channels)}


def _collect_heldout_content(value: dict[str, Any], task_ids: set[str], channels: set[str]) -> None:
    task_fields = (
        "task_ids",
        "task_id_list",
        "selected_task_ids",
        "problem_ids",
        "heldout_task_ids",
        "held_out_task_ids",
    )
    for field in task_fields:
        _update_heldout_ids(value.get(field, []), task_ids)
    for field in ("heldout_channels", "held_out_channels"):
        _update_heldout_ids(value.get(field, []), channels)
    for selected in value.get("selected_tasks", []):
        for field in ("task_id", "problem_id", "item_id"):
            if field in selected:
                task_ids.add(selected[field])
    for task_id, channel in value.get("eval_pairs", []):
        task_ids.add(task_id)
        channels.add(channel)
    split = value.get("split", value)
    if isinstance(split, dict) and "train" in split and "held-out" in split:
        _update_heldout_ids(split["held-out"], channels)
    _collect_heldout_partition(value.get("held-out"), task_ids, channels)
    for field in ("provenance", "assets", "tasks", "manifest"):
        if isinstance(value.get(field), dict):
            _collect_heldout_content(value[field], task_ids, channels)


def _collect_heldout_partition(value: object, task_ids: set[str], channels: set[str]) -> None:
    if isinstance(value, list):
        _update_heldout_ids(value, channels)
    elif isinstance(value, dict):
        _update_heldout_ids(value.get("task_ids", []), task_ids)
        _update_heldout_ids(value.get("channels", []), channels)


def _update_heldout_ids(value: object, ids: set[str]) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise TypeError("held-out identifiers must be a list of nonempty strings")
    ids.update(cast("list[str]", value))


def gate2_input_content(run_directory: Path, *, round_number: int) -> dict[str, object]:
    """Snapshot named selected inputs for both the Gate 2 producer and consumer."""
    return gate2_content_from_paths(
        run_directory / "selection.json",
        run_directory / "samples" / f"round-{round_number}.jsonl",
    )


def gate2_content_from_paths(
    selection_path: Path,
    rows_path: Path,
    heldout_paths: list[Path] | None = None,
) -> dict[str, object]:
    """Read the explicit paths used by a card-producing invocation."""
    run_directory = selection_path.parent
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise TypeError("selection.json must contain a JSON object")
    selection_content = _named_content(
        selection,
        (
            "schema_only",
            "schema_only_max_turns",
            "max_sequence_tokens",
            "prompt_render",
            "balancing",
            "admission_floors",
            "honest_share_cap",
            "heldout_scope",
        ),
    )
    selected = selection.get("selected", [])
    if not isinstance(selected, list) or any(not isinstance(entry, dict) for entry in selected):
        raise TypeError("selection selected entries must be objects")
    entry_fields = (
        "episode_key",
        "task_id",
        "scenario_slug",
        "arm",
        "seed",
        "source",
        "family",
        "prompt_token_ids",
        "completion_token_ids",
        "input_ids",
        "labels",
        "loss_mask",
        "prompt_render",
        "student_template",
        "hint_text",
        "hint_variant",
        "primary_strategy",
        "admitting_criterion",
        "admission_type",
        "flags",
        "strengthened_truth",
    )
    selection_content["selected"] = sorted(
        [_named_content(entry, entry_fields) for entry in selected],
        key=lambda entry: json.dumps(entry, sort_keys=True),
    )
    heldout_content, unavailable_heldout = _selected_heldout_content(
        selection, run_directory, heldout_paths
    )
    rows = _read_rows(rows_path)
    row_fields = (
        "key",
        "task_id",
        "env",
        "problem_id",
        "episode_key",
        "prompt_token_ids",
        "completion_token_ids",
        "input_ids",
        "labels",
        "loss_mask",
        "prompt_render",
        "student_template",
        "prompt_text",
        "hint_text",
        "hint_variant",
        "label",
        "teacher_logps",
        "student_sampling_logps",
    )
    content: dict[str, object] = {
        "selection_content": selection_content,
        "rows_content": [_named_content(rows[key], row_fields) for key in sorted(rows)],
    }
    if unavailable_heldout:
        logger.warning("held-out paths unavailable; comparing the saved held-out scope")
    else:
        content["heldout_manifest_content"] = sorted(
            heldout_content, key=lambda item: json.dumps(item, sort_keys=True)
        )
    return content


def _selected_heldout_content(
    selection: dict[str, Any],
    run_directory: Path,
    heldout_paths: list[Path] | None,
) -> tuple[list[object], bool]:
    manifest_values = selection.get("heldout_manifests", [])
    if heldout_paths is not None:
        manifest_values = [{"path": str(path)} for path in heldout_paths]
    if not isinstance(manifest_values, list):
        raise TypeError("selection.json heldout_manifests must be a list")
    heldout_content: list[object] = []
    unavailable_heldout = False
    for manifest in manifest_values:
        if not isinstance(manifest, dict) or not isinstance(manifest.get("path"), str):
            raise TypeError("selection held-out manifests need a path")
        manifest_path = Path(manifest["path"])
        if not manifest_path.is_absolute():
            manifest_path = run_directory / manifest_path
        if not manifest_path.is_file() and "heldout_scope" in selection:
            unavailable_heldout = True
            continue
        value = json.loads(manifest_path.read_text())
        if not isinstance(value, dict):
            raise TypeError("held-out manifest must contain an object")
        heldout_content.append(_heldout_content(value))
    return heldout_content, unavailable_heldout


def memory_options_from_args(args: argparse.Namespace) -> MemoryOptions:
    """Bind CLI values to the configuration stored with every checkpoint."""
    return MemoryOptions(
        deltanet_conv_kernel=args.deltanet_conv_kernel,
        liger=args.liger,
        tiled_mlp=args.tiled_mlp,
        skip_dead_mlp_recompute=args.skip_dead_mlp_recompute,
        fla_nb_bucket=args.fla_nb_bucket,
        prewarm_fla=args.prewarm_fla,
        offload_checkpoints=args.offload_checkpoints,
        base_quantization=args.base_quantization,
    )


def _validate_opd_dataset_scope(run_directory: Path) -> None:
    marker_paths = (
        run_directory / name for name in ("selection.json", "dataset_manifest.json", "card.json")
    )
    if any(_schema_only_marker(path) for path in marker_paths):
        raise ValueError("schema-only artifacts cannot be used for OPD training")


def main() -> None:
    """CLI entry point for one round; separate output directories preserve round identity."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    memory_options = memory_options_from_args(args)
    mode = cast("WeightMode", args.weight_mode)
    if mode == "sft":
        _validate_training_dataset_scope(
            args.run_dir,
            schema_only=args.schema_only,
            round_number=args.round,
        )
    else:
        _validate_opd_dataset_scope(args.run_dir)
    if args.schema_only and mode != "sft":
        raise ValueError("schema-only smoke training supports SFT mode only")
    output = args.output_dir or args.run_dir / "train" / f"round-{args.round}"
    rows = load_training_rows(
        args.run_dir / "samples" / f"round-{args.round}.jsonl",
        args.run_dir / "teacher" / f"round-{args.round}.jsonl" if mode == "opd" else None,
        weight_mode=mode,
    )
    max_steps = effective_max_steps(
        row_count=len(rows),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        epochs=args.epochs,
    )
    if args.warmup_steps > max_steps:
        raise ValueError("warmup_steps cannot exceed the effective optimizer-step target")
    logger.info(
        "training schedule rows=%d gradient_accumulation_steps=%d epochs=%s "
        "effective_steps=%d learning_rate=%s warmup_steps=%d",
        len(rows),
        args.gradient_accumulation_steps,
        args.epochs,
        max_steps,
        args.learning_rate,
        args.warmup_steps,
    )
    install_before_load(memory_options)
    if memory_options.prewarm_fla:
        prewarm_fla_nb_buckets(
            AutoConfig.from_pretrained(args.model),
            [(1, len(row["prompt_token_ids"]) + len(row["completion_token_ids"])) for row in rows],
        )
    targets = discover_lora_targets(args.model)
    model = cast("Any", load_base_model(args.model, memory_options.base_quantization))
    model.config.use_cache = False
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.0,
                target_modules=cast("list[str]", targets["target_modules"]),
            ),
        )
    validate_lora_config(
        cast("Any", model).peft_config["default"], cast("list[str]", targets["target_modules"])
    )
    cast("Any", model).enable_input_require_grads()
    trainer = DistillTrainer(
        model=model,
        train_dataset=rows,
        weight_mode=mode,
        sft_loss_averaging=args.sft_loss_averaging,
        chunk_tokens=args.chunk_tokens,
        memory_options=memory_options,
        args=TrainingArguments(
            output_dir=str(output),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=max_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,
            save_steps=1,
            logging_steps=1,
            remove_unused_columns=False,
            report_to="none",
        ),
        callbacks=build_callbacks(logging_steps=1, output_dir=str(output), s3_dest=args.s3_dest),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
