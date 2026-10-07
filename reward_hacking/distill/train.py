"""Train a LoRA student on persisted single-turn OPD or SFT completion tokens."""

import argparse
import hashlib
import json
import logging
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast, override

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import TrainOutput

from games.deltanet_kernels import bound_deltanet_kernels
from games.train import build_callbacks, chunked_per_token_logps, resolve_complete_resume_checkpoint
from grpo.throughput import discover_lora_targets
from reward_hacking.distill.memory import (
    MemoryOptions,
    install_before_load,
    install_tiled_mlp,
    load_base_model,
    prepare_checkpoint_offload,
)

logger = logging.getLogger(__name__)
DEFAULT_MEMORY_OPTIONS = MemoryOptions()
LORA_RANK = 16
LORA_ALPHA = 32
WeightMode = Literal["opd", "sft"]
RATIO_MIN = 0.8
RATIO_MAX = 1.2
ADVANTAGE_LIMIT = 5.0
# At a round's first optimizer step the trainer holds exactly the sampled weights, so ratios sit near
# 1 up to bf16 noise (~2% clipped measured); more means the sampler served different weights.
FIRST_STEP_CLIP_FRACTION_LIMIT = 0.05
SEQUENCE_DIMENSIONS = 2


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


def load_training_rows(
    samples: Path, teacher: Path | None, *, weight_mode: WeightMode
) -> list[dict[str, Any]]:
    """Join by exact key and token hash; SFT needs only the sample token ids."""
    sampled = _read_rows(samples)
    if not sampled:
        raise ValueError("training samples must not be empty")
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
        row = {"prompt_token_ids": prompt, "completion_token_ids": completion}
        loss_mask = _training_loss_mask(sample, key=key, completion_length=len(completion))
        if loss_mask is not None:
            row["loss_mask"] = loss_mask
        if weight_mode == "opd":
            score = scores[key]
            if sample["token_ids_sha256"] != score["token_ids_sha256"]:
                raise ValueError(f"{key}: teacher token hash differs from sampled token hash")
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
            "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0.0,
            "peak_reserved_bytes": float(torch.cuda.max_memory_reserved(self.device))
            if self.device.type == "cuda"
            else 0.0,
        }
        if self.rates and rate < statistics.median(self.rates) / 2:
            logger.warning(
                "step %d tokens/sec %.1f fell below half the running median %.1f; "
                "possible GPU spill to shared memory",
                state.global_step,
                rate,
                statistics.median(self.rates),
            )
        self.rates.append(rate)


class DistillTrainer(Trainer):
    """Stored-token training with normal Trainer optimizer, RNG and dataloader resume."""

    def __init__(
        self,
        *,
        weight_mode: WeightMode,
        chunk_tokens: int = 128,
        memory_options: MemoryOptions = DEFAULT_MEMORY_OPTIONS,
        **kwargs: Any,  # noqa: ANN401 - Trainer constructor passthrough
    ) -> None:
        """Keep Trainer machinery and replace only stored-token loss and accounting."""
        self.weight_mode: WeightMode = weight_mode
        self.chunk_tokens = chunk_tokens
        self.memory_options = memory_options
        self.telemetry = StepTelemetry()
        self.total_trained_completion_tokens = 0
        self._metric_sums: dict[str, float] = {}
        self._metric_tokens = 0
        self._metric_completions = 0
        self._current_logps: list[torch.Tensor] = []
        self._sampling_logps: list[torch.Tensor] = []
        kwargs["data_collator"] = collate_completion
        super().__init__(**kwargs)
        self.add_callback(self.telemetry)
        if memory_options.base_quantization != "none":
            cast_lora_to_bf16(cast("torch.nn.Module", self.model))
        self.model_accepts_loss_kwargs = False
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
            "dataset_sha256": hashlib.sha256(
                json.dumps(self.train_dataset, sort_keys=True).encode()
            ).hexdigest(),
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "max_steps": self.args.max_steps,
            "learning_rate": self.args.learning_rate,
            "chunk_tokens": chunk_tokens,
            "memory_options": asdict(memory_options),
        }

        self._identity = json.loads(json.dumps(self._identity))
        if memory_options.tiled_mlp:
            install_tiled_mlp(cast("torch.nn.Module", self.model))
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
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score completion tokens and aggregate diagnostics across micro-batches."""
        del num_items_in_batch
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
        """Restore only a complete checkpoint with identical data and training settings."""
        requested = (
            "latest" if resume_from_checkpoint is True else resume_from_checkpoint or "latest"
        )
        resolution = resolve_complete_resume_checkpoint(str(requested), self.output_dir)
        if resolution.checkpoint is not None:
            state = json.loads((Path(resolution.checkpoint) / "distill_state.json").read_text())
            saved_identity = {key: state[key] for key in self._identity if key != "memory_options"}
            saved_identity["memory_options"] = state.get(
                "memory_options", asdict(DEFAULT_MEMORY_OPTIONS)
            )
            if saved_identity != self._identity:
                raise ValueError(
                    "distillation checkpoint dataset or training configuration changed"
                )
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
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--weight-mode", choices=("opd", "sft"), required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--resume-from-checkpoint", default="latest")
    parser.add_argument("--s3-dest", default="")
    parser.add_argument("--deltanet-conv-kernel", choices=("default", "fla"), default="default")
    parser.add_argument("--liger", action="store_true")
    parser.add_argument("--tiled-mlp", action="store_true")
    parser.add_argument("--offload-checkpoints", action="store_true")
    parser.add_argument(
        "--base-quantization", choices=("none", "torchao-int8", "nf4"), default="none"
    )
    args = parser.parse_args(argv)
    if (
        args.round < 0
        or args.gradient_accumulation_steps < 1
        or args.max_steps < 1
        or args.chunk_tokens < 1
    ):
        parser.error("round must be nonnegative and training counts must be positive")
    return args


def memory_options_from_args(args: argparse.Namespace) -> MemoryOptions:
    """Bind CLI values to the configuration stored with every checkpoint."""
    return MemoryOptions(
        deltanet_conv_kernel=args.deltanet_conv_kernel,
        liger=args.liger,
        tiled_mlp=args.tiled_mlp,
        offload_checkpoints=args.offload_checkpoints,
        base_quantization=args.base_quantization,
    )


def main() -> None:
    """CLI entry point for one round; separate output directories preserve round identity."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    memory_options = memory_options_from_args(args)
    mode = cast("WeightMode", args.weight_mode)
    output = args.output_dir or args.run_dir / "train" / f"round-{args.round}"
    rows = load_training_rows(
        args.run_dir / "samples" / f"round-{args.round}.jsonl",
        args.run_dir / "teacher" / f"round-{args.round}.jsonl" if mode == "opd" else None,
        weight_mode=mode,
    )
    install_before_load(memory_options)
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
        chunk_tokens=args.chunk_tokens,
        memory_options=memory_options,
        args=TrainingArguments(
            output_dir=str(output),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
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
