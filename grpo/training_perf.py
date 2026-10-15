"""Per-step training rates and explicitly approximate parameter-matmul utilization."""

import functools
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, cast

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from grpo.gpu_perf import device_peak

logger = logging.getLogger(__name__)

MFU_LIMITATION = (
    "Parameter-matmul estimate: 4 FLOPs/frozen parameter/token + 6 FLOPs/trainable parameter/token. "
    "Excludes attention, DeltaNet recurrence, optimizer, loss, recomputation and rollout generation. "
    "Processed tokens include padding. MoE utilization is unknown because activated parameters are not counted."
)


def training_step_metrics(
    processed_tokens: int,
    seconds: float,
    *,
    frozen_parameters: int,
    trainable_parameters: int,
    peak_flops: float | None,
) -> dict[str, float | int | None]:
    """Compute observed throughput and an explicit training-compute approximation."""
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError("training step seconds must be finite and positive")
    if min(processed_tokens, frozen_parameters, trainable_parameters) < 0:
        raise ValueError("token and parameter counts must be nonnegative")
    if peak_flops is not None and (peak_flops <= 0 or not math.isfinite(peak_flops)):
        raise ValueError("peak FLOPs must be finite and positive")
    flops = processed_tokens * (4 * frozen_parameters + 6 * trainable_parameters)
    return {
        "processed_tokens": processed_tokens,
        "seconds": seconds,
        "tokens_per_second": processed_tokens / seconds,
        "parameter_matmul_mfu_estimate": flops / seconds / peak_flops
        if peak_flops is not None
        else None,
    }


def training_output_directory(args: TrainingArguments) -> Path:
    """Require a persistent destination before recording training telemetry."""
    if args.output_dir is None:
        raise ValueError("training throughput requires output_dir")
    return Path(args.output_dir)


class TrainingThroughputCallback(TrainerCallback):
    """Count the tensor slots actually passed to loss and persist every completed optimizer step."""

    def __init__(self) -> None:
        """Wait for a trainer and its actual model rather than infer parameter counts from its name."""
        self.records: list[dict[str, float | int | None]] = []
        self.processed_tokens = 0
        self.started_at = 0.0
        self.frozen_parameters = 0
        self.trainable_parameters = 0
        self.peak_flops: float | None = None
        self.peak_source: str | None = None
        self.attached = False

    def attach(self, trainer: Any) -> None:  # noqa: ANN401 - shared HF/TRL private seam
        """Wrap loss accounting once; the training calculation and return value are unchanged."""
        if self.attached:
            raise RuntimeError("training throughput is already attached")
        original = trainer.compute_loss

        @functools.wraps(original)
        def counted(
            model: torch.nn.Module, inputs: dict[str, Any], *args: object, **kwargs: object
        ) -> Any:  # noqa: ANN401
            result = original(model, inputs, *args, **kwargs)
            if model.training:
                if "input_ids" in inputs:
                    self.processed_tokens += cast("torch.Tensor", inputs["input_ids"]).numel()
                else:
                    self.processed_tokens += cast("torch.Tensor", inputs["prompt_ids"]).numel()
                    self.processed_tokens += cast("torch.Tensor", inputs["completion_ids"]).numel()
            return result

        trainer.compute_loss = counted
        trainer.add_callback(self)
        self.attached = True

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Use actual parameter counts and the device's published dense BF16 peak."""
        del control
        destination = training_output_directory(args) / "throughput.json"
        if destination.exists():
            previous = json.loads(destination.read_text())
            if previous["kind"] != "training":
                raise ValueError("training throughput path contains a different stage kind")
            self.records = [
                record for record in previous["steps"] if record["step"] <= state.global_step
            ]
        model = cast("torch.nn.Module", kwargs["model"])
        self.frozen_parameters = sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        )
        self.trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if args.device.type == "cuda":
            peak = device_peak(torch.cuda.get_device_name(args.device))
            if peak is not None:
                self.peak_flops = peak.tensor_flops
                self.peak_source = peak.source
        if any("expert" in name for name, _parameter in model.named_parameters()):
            self.peak_flops = None
            self.peak_source = None

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Synchronize before measuring the next accumulation window."""
        del state, control, kwargs
        if args.device.type == "cuda":
            torch.cuda.synchronize(args.device)
        self.processed_tokens = 0
        self.started_at = time.perf_counter()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Write an incremental readout and a visible one-line rate for every update."""
        del control, kwargs
        if args.device.type == "cuda":
            torch.cuda.synchronize(args.device)
        record = training_step_metrics(
            self.processed_tokens,
            time.perf_counter() - self.started_at,
            frozen_parameters=self.frozen_parameters,
            trainable_parameters=self.trainable_parameters,
            peak_flops=self.peak_flops,
        )
        record["step"] = state.global_step
        self.records.append(record)
        utilization = record["parameter_matmul_mfu_estimate"]
        logger.info(
            "TRAINING THROUGHPUT step=%d tokens/s=%.1f parameter-matmul MFU=%s",
            state.global_step,
            record["tokens_per_second"],
            "unknown" if utilization is None else f"{utilization:.1%}",
        )
        output = training_output_directory(args)
        output.mkdir(parents=True, exist_ok=True)
        destination = output / "throughput.json"
        temporary = output / "throughput.json.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "kind": "training",
                    "steps": self.records,
                    "peak_source": self.peak_source,
                    "mfu_limitations": MFU_LIMITATION,
                },
                indent=2,
            )
            + "\n"
        )
        temporary.replace(destination)
