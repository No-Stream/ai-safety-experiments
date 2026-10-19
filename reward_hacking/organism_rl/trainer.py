"""Exact-id GRPO with external sampling, teacher advantages, and complete checkpoints.

Unscorable rewards must return None: TRL excludes them from the group baseline.
Their entire assistant loss mask is also zeroed, including the teacher term.

Each training micro-batch is one episode trimmed to its own tokens (TRL pads a step's episodes to
the longest), and its log-probabilities come from an fp32 head a slice at a time rather than
TRL's whole-completion logits, which cap a 9B row at a few thousand tokens. Both are
statistically neutral, not bit-identical: reduction orders change with width, and the fp32 head
is closer to the sampler's own log-probabilities than TRL's bf16 logits.
"""

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sized
from itertools import islice
from pathlib import Path
from typing import Any, Protocol, cast, override

import torch
from datasets import Dataset
from torch import nn
from torch.utils.data import Sampler
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import TrainOutput, get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

from games.train import STEP_PADDED_TOKENS_METRIC, STEP_TRIMMED_TOKENS_METRIC, trim_micro_batch
from grpo.training_perf import TrainingThroughputCallback
from reward_hacking.checkpoint_retention import CheckpointRetentionMixin
from reward_hacking.distill.train import chunked_completion_logps, prepare_fp32_head
from reward_hacking.model_backend import VLLMRequestGate
from reward_hacking.organism_rl.admission import LiveKVAdmission
from reward_hacking.organism_rl.generate_provenance import active_provenance_path
from reward_hacking.organism_rl.phases import VLLMPhaseController
from reward_hacking.organism_rl.precision import (
    assert_fp32_training_state,
    record_precision_provenance,
)
from reward_hacking.organism_rl.training_memory import MicrostepMemoryRecorder

# One fp32 slice of a 248K-token vocabulary at 1,024 positions is about 1 GiB.
DEFAULT_LOGPROB_CHUNK_TOKENS = 1024

logger: logging.Logger = logging.getLogger(__name__)
_VLLM_KV_LOG = re.compile(r"GPU KV cache usage: ([\d.]+)%")
_VLLM_PREFIX_LOG = re.compile(r"Prefix cache hit rate: ([\d.]+)%")


class _GenerationLookaheadSampler(Sampler[int]):
    """Expose the next TRL generation window while yielding its exact sampler order."""

    def __init__(
        self,
        sampler: Sampler[int],
        dataset: Dataset,
        generation_batch_size: int,
        generation_repeats: int,
        next_prompt_windows: list[list[str]],
    ) -> None:
        if generation_batch_size < 1 or generation_repeats < 1:
            raise ValueError("Generation lookahead dimensions must be positive")
        if not isinstance(sampler, Sized):
            raise TypeError("Generation lookahead needs a finite TRL sampler")
        self.sampler = sampler
        self.sampler_length = len(sampler)
        self.dataset = dataset
        self.generation_batch_size = generation_batch_size
        self.generation_repeats = generation_repeats
        self.next_prompt_windows = next_prompt_windows
        self.epoch = 0
        self.sampler_epochs_started = 0
        self.buffered_sampler_epoch: int | None = None
        self.buffered_sampler_iterator: Iterator[int] | None = None
        self.buffered_sampler_batch: list[int] | None = None

    def __iter__(self) -> Iterator[int]:
        self.next_prompt_windows.clear()
        sampler_iterator, current_batch = self._sampler_epoch(self.epoch)
        while current_batch:
            repeated_batches = [current_batch]
            for _ in range(self.generation_repeats - 1):
                repeated_batch = self._take_batch(sampler_iterator)
                if repeated_batch != current_batch:
                    raise RuntimeError("TRL sampler changed a generation batch before its refresh")
                repeated_batches.append(repeated_batch)

            next_batch = self._take_batch(sampler_iterator)
            epoch_finished = not next_batch
            if epoch_finished:
                next_batch = self._prime_next_epoch()
            self.next_prompt_windows.append(self._prompts(next_batch))
            for repeated_batch in repeated_batches:
                yield from repeated_batch
            if epoch_finished:
                break
            current_batch = next_batch

    def __len__(self) -> int:
        return self.sampler_length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _sampler_epoch(self, epoch: int) -> tuple[Iterator[int], list[int]]:
        if self.buffered_sampler_epoch == epoch:
            if self.buffered_sampler_iterator is None or self.buffered_sampler_batch is None:
                raise RuntimeError("Buffered TRL sampler epoch is incomplete")
            sampler_iterator = self.buffered_sampler_iterator
            first_batch = self.buffered_sampler_batch
            self.buffered_sampler_epoch = None
            self.buffered_sampler_iterator = None
            self.buffered_sampler_batch = None
            return sampler_iterator, first_batch

        if epoch < self.sampler_epochs_started:
            raise RuntimeError("Cannot rewind the wrapped TRL sampler")
        while self.sampler_epochs_started < epoch:
            skipped_iterator = iter(self.sampler)
            for _ in range(self.sampler_length):
                next(skipped_iterator)
            self.sampler_epochs_started += 1

        sampler_iterator = iter(self.sampler)
        self.sampler_epochs_started += 1
        return sampler_iterator, self._take_batch(sampler_iterator)

    def _prime_next_epoch(self) -> list[int]:
        next_epoch = self.epoch + 1
        if self.buffered_sampler_epoch is not None:
            raise RuntimeError("TRL sampler already has a buffered epoch")
        if self.sampler_epochs_started != next_epoch:
            raise RuntimeError("TRL sampler epoch counter lost synchronization")
        sampler_iterator = iter(self.sampler)
        self.sampler_epochs_started += 1
        first_batch = self._take_batch(sampler_iterator)
        if not first_batch:
            raise RuntimeError("TRL sampler produced no generation batch in the next epoch")
        self.buffered_sampler_epoch = next_epoch
        self.buffered_sampler_iterator = sampler_iterator
        self.buffered_sampler_batch = first_batch
        return first_batch

    def _take_batch(self, sampler_iterator: Iterator[int]) -> list[int]:
        batch = list(islice(sampler_iterator, self.generation_batch_size))
        if batch and len(batch) != self.generation_batch_size:
            raise RuntimeError("TRL sampler produced an incomplete generation batch")
        return batch

    def _prompts(self, indices: list[int]) -> list[str]:
        prompts: list[str] = []
        for index in indices:
            prompt = self.dataset[index]["prompt"]
            if not isinstance(prompt, str):
                raise TypeError("Step-pipelined organism training requires string prompts")
            prompts.append(prompt)
        return prompts


def parse_vllm_rollout_log(server_log: str) -> tuple[float, float, int]:
    """Read the latest cumulative prefix rate and maximum periodic KV sample."""
    kv_samples = _VLLM_KV_LOG.findall(server_log)
    if not kv_samples:
        return math.nan, math.nan, 0
    prefix_rates = _VLLM_PREFIX_LOG.findall(server_log)
    return (
        float(prefix_rates[-1]) if prefix_rates else math.nan,
        max(float(kv) for kv in kv_samples),
        len(kv_samples),
    )


def add_teacher_advantage(
    advantages: torch.Tensor,
    sampler_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor | None,
    teacher_beta: float,
) -> torch.Tensor:
    """Return native advantages unchanged at zero, otherwise add token teacher penalties."""
    if teacher_beta == 0:
        return advantages
    if teacher_beta < 0 or not math.isfinite(teacher_beta):
        raise ValueError("teacher_beta must be finite and nonnegative")
    if teacher_logprobs is None or teacher_logprobs.shape != sampler_logprobs.shape:
        raise ValueError("teacher logprobs must align exactly with sampler logprobs")
    reward_term = advantages[:, None] if advantages.ndim == 1 else advantages
    return reward_term - teacher_beta * (sampler_logprobs - teacher_logprobs)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(directory: Path) -> None:
    """Refuse incomplete checkpoints; log, but accept, files whose sha256 differs from the marker.

    The marker's hashes are a record of what was written. A changed file is worth a warning, not a
    refusal: resume is gated on completeness alone, and no code or file hash may block a resume.
    """
    marker = directory / "complete.json"
    if not marker.is_file():
        raise ValueError(f"Incomplete checkpoint: {directory}")
    try:
        manifest: dict[str, str] = json.loads(marker.read_text())
    except json.JSONDecodeError as error:
        raise ValueError("Incomplete checkpoint completion marker") from error
    required = {
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "rollout_records.json",
    }
    if not required.issubset(manifest):
        raise ValueError("Incomplete checkpoint state manifest")
    if not any(
        name in manifest
        for name in ("adapter_model.safetensors", "model.safetensors", "pytorch_model.bin")
    ):
        raise ValueError("Incomplete checkpoint model or adapter")
    model_config = (
        "adapter_config.json" if "adapter_model.safetensors" in manifest else "config.json"
    )
    if model_config not in manifest:
        raise ValueError("Incomplete checkpoint model configuration")
    for name in manifest:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Incomplete checkpoint file: {name}")


def _validate_rollout_masks(output: dict[str, Any]) -> None:
    rows = zip(
        output["prompt_ids"],
        output["completion_ids"],
        output["env_mask"],
        output["scorable"],
        output["rollout_records"],
        strict=True,
    )
    for prompt, completion, mask, scorable, record in rows:
        if prompt != record["prompt_token_ids"] or completion != record["completion_token_ids"]:
            raise ValueError("Rollout token ids differ from their exact-id training record")
        expected = record["loss_mask"] if scorable else [0] * len(completion)
        if (
            len(mask) != len(completion)
            or mask != expected
            or any(value not in (0, 1) for value in mask)
        ):
            raise ValueError("Rollout env_mask differs from sampled assistant provenance")


class TrainingPhase(Protocol):
    """Transfer GPU ownership between the trainer and external sampler."""

    def before_sampling(self, trainer: Any) -> None:  # noqa: ANN401
        """Offload training state and wake the sampler."""
        ...

    def before_training(self, trainer: Any) -> None:  # noqa: ANN401
        """Sleep the sampler and restore training state."""
        ...

    def initialize(self, trainer: Any, step: int = 0) -> None:  # noqa: ANN401
        """Publish the initial or restored adapter before any sample."""
        ...

    def after_optimizer_step(self, trainer: Any) -> None:  # noqa: ANN401
        """Publish the current adapter to the external sampler."""
        ...


class _SyncCallback(TrainerCallback):
    def __init__(self, trainer: "OrganismGRPOTrainer", phase: TrainingPhase) -> None:
        self.trainer = trainer
        self.phase = phase

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        del args, control, kwargs
        self.phase.initialize(self.trainer, step=state.global_step)
        self.trainer.resume_sampler_after_sync(state.global_step)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        del control, kwargs
        metrics = self.trainer._metrics["train"]  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if args.device.type == "cuda":
            metrics["organism/trainer_peak_allocated_gib"].append(
                max(
                    torch.cuda.max_memory_allocated(args.device),
                    self.trainer.training_step_peak_allocated_bytes,
                )
                / 2**30
            )
            self.trainer.training_step_peak_allocated_bytes = 0
            torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        self.phase.after_optimizer_step(self.trainer)
        self.trainer.resume_sampler_after_sync(state.global_step)
        metrics["organism/seconds_adapter_sync"].append(time.perf_counter() - started)
        if isinstance(self.phase, VLLMPhaseController):
            for name, seconds in self.phase.last_sync_seconds.items():
                metrics[f"organism/seconds_sync_{name}"].append(seconds)


class _SkippedRowsCallback(TrainerCallback):
    def __init__(self, trainer: "OrganismGRPOTrainer") -> None:
        self.trainer = trainer

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        del args, state, control, kwargs
        self.trainer._finish_training_step_metrics()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


class OrganismGRPOTrainer(CheckpointRetentionMixin, GRPOTrainer):
    """Keep native GRPO loss and apply detached sampler correction to token advantages."""

    def __init__(  # noqa: C901, PLR0913, PLR0915 - explicit settings and rollout hook
        self,
        *args: Any,  # noqa: ANN401
        teacher_beta: float = 0.0,
        importance_ratio_cap: float = 2.0,
        checkpoint_keep_latest: int = 2,
        checkpoint_keep_middle: int = 3,
        phase: TrainingPhase | None = None,
        step_pipelining: bool = True,
        request_gate: VLLMRequestGate | None = None,
        sampler_admission: LiveKVAdmission | None = None,
        sampler_model_id: Callable[[], str] | None = None,
        sampler_log_path: Path | None = None,
        logprob_chunk_tokens: int = DEFAULT_LOGPROB_CHUNK_TOKENS,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Wrap the experimental TRL rollout hook and enforce the supported training configuration."""
        if not math.isfinite(importance_ratio_cap) or importance_ratio_cap <= 0:
            raise ValueError("importance_ratio_cap must be finite and positive")
        if not math.isfinite(teacher_beta) or teacher_beta < 0:
            raise ValueError("teacher_beta must be finite and nonnegative")
        rollout = kwargs["rollout_func"]
        self.teacher_beta = teacher_beta
        self.importance_ratio_cap = importance_ratio_cap
        self.checkpoint_keep_latest = checkpoint_keep_latest
        self.checkpoint_keep_middle = checkpoint_keep_middle
        self.phase = phase
        self.step_pipelining = step_pipelining
        if request_gate is not None and (
            phase is None or sampler_admission is None or sampler_model_id is None
        ):
            raise ValueError("pipelined request gate requires phase, admission and model identity")
        self.request_gate = request_gate
        self.sampler_admission = sampler_admission
        self.sampler_model_id = sampler_model_id
        self._sampler_admission_suspended = False
        self.next_prompts: list[str] = []
        self._next_generation_prompts_by_window: list[list[str]] = []
        self._lookahead_windows_per_epoch = 0
        self.sampler_log_path = sampler_log_path
        if logprob_chunk_tokens < 1:
            raise ValueError("logprob_chunk_tokens must be positive")
        self.logprob_chunk_tokens = logprob_chunk_tokens
        self._trim_padded_tokens_this_step = 0
        self._trim_trimmed_tokens_this_step = 0
        self.latest_rollout: dict[str, Any] = {}
        self.step_rollout_records: list[Any] = []
        self._active_loss_inputs: dict[str, Any] | None = None
        self._attention_mask_prevalidated_for_loss = False
        self._non_training_seconds = 0.0
        self._rows_skipped_this_step = 0
        self._sampler_correction_metric_rows_this_step: list[torch.Tensor] = []

        def captured_rollout(prompts: list[Any], trainer: "OrganismGRPOTrainer") -> dict[str, Any]:
            trainer.next_prompts = self._next_prompts_for_current_step()
            server_log_offset = (
                self.sampler_log_path.stat().st_size if self.sampler_log_path is not None else None
            )
            started = time.perf_counter()
            if self.phase is not None:
                self.phase.before_sampling(self)
            sampler_ready = time.perf_counter()
            output: dict[str, Any] = rollout(prompts, trainer)
            _validate_rollout_masks(output)
            self.latest_rollout = output
            self.step_rollout_records.extend(
                output.pop("evidence_records", output["rollout_records"])
            )
            for name, value in output.pop("dynamic_sampling_metrics", {}).items():
                if isinstance(value, (int, float)):
                    self._metrics["train"][f"organism/dynamic_sampling/{name}"].append(value)
            sampled = time.perf_counter()
            if self.request_gate is not None:
                self.request_gate.pause_and_wait()
                if self.sampler_admission is None:
                    raise RuntimeError("pipelined sampler has no KV admission controller")
                self.sampler_admission.suspend()
                self._sampler_admission_suspended = True
                self.sampler_admission.reset_step(preserve_pending_requests=True)
            if self.phase is not None:
                self.phase.before_training(self)
            seconds = self._metrics["train"]
            training_ready = time.perf_counter()
            seconds["organism/seconds_wake_sampler"].append(sampler_ready - started)
            seconds["organism/seconds_rollout"].append(sampled - sampler_ready)
            seconds["organism/seconds_restore_trainer"].append(training_ready - sampled)
            self._non_training_seconds += training_ready - started
            if server_log_offset is not None and self.sampler_log_path is not None:
                with self.sampler_log_path.open("rb") as server_log_file:
                    server_log_file.seek(server_log_offset)
                    latest_prefix, max_kv, sample_count = parse_vllm_rollout_log(
                        server_log_file.read().decode("utf-8")
                    )
                seconds["organism/vllm_prefix_hit_rate_latest_pct"].append(latest_prefix)
                seconds["organism/vllm_gpu_kv_usage_max_logged_pct"].append(max_kv)
                seconds["organism/vllm_log_samples"].append(sample_count)
            return output

        kwargs["rollout_func"] = captured_rollout
        super().__init__(*args, **kwargs)
        if self.args.output_dir is None:
            raise ValueError("Organism training requires an output directory")
        self.microstep_memory = MicrostepMemoryRecorder(
            Path(self.args.output_dir) / "trainer-memory.jsonl"
        )
        self.training_step_peak_allocated_bytes = 0
        if self.model is None:
            raise RuntimeError("Organism trainer requires a model")
        assert_fp32_training_state(self.model, self._precision_optimizer())
        prepare_fp32_head(self.model)
        self._validate_configuration()
        TrainingThroughputCallback(non_training_seconds=lambda: self._non_training_seconds).attach(
            self
        )
        self.add_callback(_SkippedRowsCallback(self))
        if phase is not None:
            self.add_callback(_SyncCallback(self, phase))

    def resume_sampler_after_sync(self, policy_step: int) -> None:
        """Publish a synchronized adapter and reopen early episode requests."""
        if self.request_gate is None:
            return
        if self.sampler_model_id is None:
            raise RuntimeError("pipelined sampler has no model identity")
        self.request_gate.pause_and_wait()
        self.request_gate.set_policy(self.sampler_model_id(), policy_step=policy_step)
        if self._sampler_admission_suspended:
            if self.sampler_admission is None:
                raise RuntimeError("pipelined sampler has no KV admission controller")
            self.sampler_admission.resume()
            self._sampler_admission_suspended = False
        self.request_gate.resume()

    @override
    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler[int]:
        sampler = super()._get_train_sampler(dataset)
        if not self.step_pipelining:
            return sampler
        source_dataset = self.train_dataset if dataset is None else dataset
        if not isinstance(source_dataset, Dataset):
            raise TypeError("Step-pipelined organism training requires a map-style dataset")
        steps_per_generation = cast("int", cast("GRPOConfig", self.args).steps_per_generation)
        configuration = cast("GRPOConfig", self.args)
        generation_batch_size = configuration.generation_batch_size
        if generation_batch_size is None:
            raise ValueError("Step-pipelined training requires a generation batch size")
        num_generations = self.num_generations
        if num_generations is None:
            raise ValueError("Step-pipelined training requires a generation count")
        prompt_batch_size = generation_batch_size // num_generations
        self._lookahead_windows_per_epoch = len(source_dataset) // prompt_batch_size
        if self._lookahead_windows_per_epoch == 0:
            raise ValueError("Step-pipelined training needs one complete prompt group per epoch")
        return _GenerationLookaheadSampler(
            sampler=sampler,
            dataset=source_dataset,
            generation_batch_size=generation_batch_size,
            generation_repeats=steps_per_generation,
            next_prompt_windows=self._next_generation_prompts_by_window,
        )

    def _next_prompts_for_current_step(self) -> list[str]:
        if not self.step_pipelining or self._lookahead_windows_per_epoch == 0:
            return []
        if self.state.global_step + 1 >= self.args.max_steps:
            return []
        window_index = self.state.global_step % self._lookahead_windows_per_epoch
        if window_index >= len(self._next_generation_prompts_by_window):
            return []
        return list(self._next_generation_prompts_by_window[window_index])

    def _validate_configuration(self) -> None:
        if self.use_vllm or self.beta != 0:
            raise ValueError("Organism training requires use_vllm=False and beta=0")
        if self.mask_truncated_completions:
            raise ValueError(
                "Flattened multi-turn episodes require mask_truncated_completions=False"
            )
        if self.use_liger_kernel:
            raise ValueError(
                "Fused Liger GRPO loss bypasses organism sampler correction; use model-level Liger only"
            )
        if (
            self.args.save_steps != 1
            or self.args.save_only_model
            or self.args.save_strategy != "steps"
        ):
            raise ValueError("Organism training requires a full checkpoint every optimizer step")
        if (
            cast("GRPOConfig", self.args).steps_per_generation
            != self.args.gradient_accumulation_steps
            or self.num_iterations != 1
        ):
            raise ValueError("Organism sampling must refresh each optimizer step")
        if self.accelerator.num_processes != 1:
            raise ValueError("Organism training supports a single device only")
        if self.args.per_device_train_batch_size != 1:
            raise ValueError("Organism micro-batches hold one trimmed episode")
        if self._entropy_bonus_enabled or self.top_entropy_quantile < 1.0:
            raise ValueError("Organism entropies are a detached read-out, not a loss term")

    def _finish_training_step_metrics(self) -> None:
        metrics = self._metrics["train"]
        metrics["organism/rows_skipped_zero_advantage"].append(self._rows_skipped_this_step)
        gap_sum = 0.0
        gap_abs_sum = 0.0
        assistant_tokens = 0
        capped_assistant_tokens = 0
        fresh_assistant_tokens = 0
        fresh_gap_abs_sum = 0.0
        fresh_capped_tokens = 0
        stale_assistant_tokens = 0
        stale_gap_abs_sum = 0.0
        stale_capped_tokens = 0
        if self._sampler_correction_metric_rows_this_step:
            metric_rows = cast(
                "list[list[float]]",
                torch.stack(self._sampler_correction_metric_rows_this_step).tolist(),
            )
            for metric_row in metric_rows:
                gap_sum += metric_row[0]
                gap_abs_sum += metric_row[1]
                assistant_tokens += int(metric_row[2])
                capped_assistant_tokens += int(metric_row[3])
                fresh_assistant_tokens += int(metric_row[4])
                fresh_gap_abs_sum += metric_row[5]
                fresh_capped_tokens += int(metric_row[6])
                stale_assistant_tokens += int(metric_row[7])
                stale_gap_abs_sum += metric_row[8]
                stale_capped_tokens += int(metric_row[9])
        self._append_sampler_correction_metric_values(
            "train",
            [
                gap_sum,
                gap_abs_sum,
                float(assistant_tokens),
                float(capped_assistant_tokens),
                float(fresh_assistant_tokens),
                fresh_gap_abs_sum,
                float(fresh_capped_tokens),
                float(stale_assistant_tokens),
                stale_gap_abs_sum,
                float(stale_capped_tokens),
            ],
        )
        self._rows_skipped_this_step = 0
        self._sampler_correction_metric_rows_this_step.clear()

    def _sampler_correction_metric_values(
        self,
        gap: torch.Tensor,
        assistant_mask: torch.Tensor,
        stale_mask: torch.Tensor,
    ) -> torch.Tensor:
        gap = gap.detach()
        assistant_mask = assistant_mask.bool()
        fresh_mask = assistant_mask & ~stale_mask
        stale_assistant_mask = assistant_mask & stale_mask
        log_ratio_cap = math.log(self.importance_ratio_cap)
        capped_mask = gap > log_ratio_cap
        values = (
            gap.sum(),
            gap.abs().sum(),
            assistant_mask.sum(),
            (capped_mask * assistant_mask).sum(),
            fresh_mask.sum(),
            (gap.abs() * fresh_mask).sum(),
            (capped_mask * fresh_mask).sum(),
            stale_assistant_mask.sum(),
            (gap.abs() * stale_assistant_mask).sum(),
            (capped_mask * stale_assistant_mask).sum(),
        )
        return torch.stack([value.to(dtype=torch.float64) for value in values])

    def _append_sampler_correction_metric_values(self, mode: str, values: list[float]) -> None:
        (
            gap_sum,
            abs_gap_sum,
            token_count,
            capped_token_count,
            fresh_tokens,
            fresh_abs_gap,
            fresh_capped,
            stale_tokens,
            stale_abs_gap,
            stale_capped,
        ) = values
        metrics = self._metrics[mode]
        metrics["organism/trainer_sampler_logprob_gap"].append(
            gap_sum / token_count if token_count else math.nan
        )
        metrics["organism/trainer_sampler_logprob_gap_abs"].append(
            abs_gap_sum / token_count if token_count else math.nan
        )
        metrics["organism/trainer_sampler_logprob_gap_tokens"].append(int(token_count))
        metrics["organism/sampler_ratio_capped_fraction"].append(
            capped_token_count / token_count if token_count else math.nan
        )
        for freshness, freshness_tokens, freshness_abs_gap, freshness_capped in (
            ("fresh", fresh_tokens, fresh_abs_gap, fresh_capped),
            ("stale", stale_tokens, stale_abs_gap, stale_capped),
        ):
            metrics[f"organism/trainer_sampler_logprob_gap_abs_{freshness}"].append(
                freshness_abs_gap / freshness_tokens if freshness_tokens else math.nan
            )
            metrics[f"organism/sampler_ratio_capped_fraction_{freshness}"].append(
                freshness_capped / freshness_tokens if freshness_tokens else math.nan
            )

    def _precision_optimizer(self) -> torch.optim.Optimizer | None:
        optimizer = self.optimizer
        if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Organism precision checks require a PyTorch optimizer")
        return optimizer

    @override
    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
        trial: dict[str, Any] | None = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> TrainOutput:
        """Restore finished checkpoints without entering Transformers' extra-step resume path."""
        if resume_from_checkpoint is True:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError("No checkpoint found for resume")
        if isinstance(resume_from_checkpoint, str):
            directory = Path(resume_from_checkpoint)
            validate_checkpoint(directory)
            saved_state = TrainerState.load_from_json(str(directory / "trainer_state.json"))
            if self.args.max_steps > 0 and saved_state.global_step >= self.args.max_steps:
                self._load_from_checkpoint(resume_from_checkpoint)
                self.state = saved_state
                if self.phase is not None:
                    self.phase.initialize(self, step=saved_state.global_step)
                self._prune_retained_checkpoints(directory.parent, finished=True)
                return TrainOutput(
                    saved_state.global_step, 0.0, {"organism/resume_already_complete": 1.0}
                )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval)

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = super()._generate_and_score_completions(inputs)
        sampler: torch.Tensor = batch["sampling_per_token_logps"]
        mask: torch.Tensor = batch["completion_mask"] * batch["tool_mask"]
        if self.request_gate is not None:
            self._attach_stale_policy_mask(batch, sampler)
        scorable = torch.tensor(self.latest_rollout["scorable"], device=mask.device)
        batch["tool_mask"] = batch["tool_mask"] * scorable[:, None]
        mask = mask * scorable[:, None]
        teacher: torch.Tensor | None = None
        if self.teacher_beta != 0:
            rows: list[list[float]] = self.latest_rollout["teacher_logprobs"]
            if len(rows) != len(self.latest_rollout["completion_ids"]):
                raise ValueError("Teacher logprobs must have one row per episode")
            teacher = torch.zeros_like(sampler)
            for index, row in enumerate(rows):
                if len(row) != len(self.latest_rollout["completion_ids"][index]):
                    raise ValueError("Teacher logprobs do not align with completion ids")
                teacher[index, : len(row)] = torch.tensor(row, device=sampler.device)
            if not torch.isfinite(teacher[mask.bool()]).all():
                raise ValueError("Teacher logprobs must be finite on assistant tokens")
        advantages: torch.Tensor = batch["advantages"]
        if self.model is None:
            raise RuntimeError("Organism trainer requires a model")
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["organism/reward_advantage"].append(advantages.mean().item())
        batch["advantages"] = add_teacher_advantage(advantages, sampler, teacher, self.teacher_beta)
        teacher_term = (
            torch.zeros_like(sampler)
            if teacher is None
            else -self.teacher_beta * (sampler - teacher)
        )
        self._metrics[mode]["organism/teacher_advantage"].append(
            (teacher_term * mask).sum().item() / max(mask.sum().item(), 1)
        )
        return batch

    def _attach_stale_policy_mask(self, batch: dict[str, Any], sampler: torch.Tensor) -> None:
        policy_rows: list[list[int]] = self.latest_rollout["policy_steps"]
        completion_rows: list[list[int]] = self.latest_rollout["completion_ids"]
        if len(policy_rows) != sampler.shape[0]:
            raise ValueError("sampler policy rows do not match the generation batch")
        current_step = int(self.state.global_step)
        stale_mask = torch.zeros_like(sampler, dtype=torch.bool)
        staleness_counts: Counter[int] = Counter()
        stale_episodes = 0
        for index, (versions, completion_ids) in enumerate(
            zip(policy_rows, completion_rows, strict=True)
        ):
            if len(versions) != len(completion_ids):
                raise ValueError("sampler policy versions do not align with completion tokens")
            if any(
                type(version) is not int or not max(current_step - 1, 0) <= version <= current_step
                for version in versions
            ):
                raise ValueError("sampler policy must be current or one optimizer step stale")
            assistant_mask = self.latest_rollout["rollout_records"][index]["loss_mask"]
            if len(assistant_mask) != len(versions):
                raise ValueError("sampler policy versions do not align with assistant mask")
            stale_episodes += int(
                any(
                    active and version < current_step
                    for active, version in zip(assistant_mask, versions, strict=True)
                )
            )
            for active, version in zip(assistant_mask, versions, strict=True):
                if active:
                    staleness_counts[current_step - version] += 1
            stale_mask[index, : len(versions)] = torch.tensor(
                [version < current_step for version in versions],
                dtype=torch.bool,
                device=sampler.device,
            )
        batch["sampler_stale_mask"] = stale_mask
        assistant_tokens = sum(staleness_counts.values())
        metrics = self._metrics["train"]
        metrics["organism/stale_token_fraction"].append(
            staleness_counts[1] / assistant_tokens if assistant_tokens else math.nan
        )
        sampled_episodes = sum(
            record.get("record") != "dynamic_sampling_padding"
            for record in self.latest_rollout["rollout_records"]
        )
        metrics["organism/stale_episode_fraction"].append(
            stale_episodes / sampled_episodes if sampled_episodes else math.nan
        )
        metrics["organism/stale_assistant_tokens"].append(staleness_counts[1])
        logger.info(
            "organism policy staleness step=%d assistant_tokens_by_steps_old=%s",
            current_step,
            dict(sorted(staleness_counts.items())),
        )

    @override
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Include backward in each row's peak, excluding rollout and trainer restore."""
        skipped_before = self._rows_skipped_this_step
        completed = False
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
            completed = True
            return loss
        finally:
            peak = self.microstep_memory.finish(
                completed=completed, skipped=self._rows_skipped_this_step > skipped_before
            )
            self.training_step_peak_allocated_bytes = max(
                self.training_step_peak_allocated_bytes, peak
            )

    @override
    def _prepare_inputs(self, generation_batch: dict[str, Any]) -> dict[str, Any]:
        """Cut each training micro-batch to its live columns and count what the trim removed."""
        inputs: dict[str, Any] = super()._prepare_inputs(generation_batch)
        if self.model is None or not self.model.training:
            return inputs
        trimmed, stats = trim_micro_batch(inputs)
        if "sampler_stale_mask" in trimmed:
            trimmed["sampler_stale_mask"] = trimmed["sampler_stale_mask"][
                :, : trimmed["completion_ids"].size(1)
            ]
        advantages: torch.Tensor = trimmed["advantages"]
        if advantages.ndim == 2:  # noqa: PLR2004 - per-token teacher advantages
            trimmed["advantages"] = advantages[:, : trimmed["completion_ids"].size(1)]
        self._trim_padded_tokens_this_step += stats.padded_tokens
        self._trim_trimmed_tokens_this_step += stats.trimmed_tokens
        steps_per_generation = cast("int", cast("GRPOConfig", self.args).steps_per_generation)
        if self._step % steps_per_generation == steps_per_generation - 1:
            self._metrics["train"][STEP_PADDED_TOKENS_METRIC].append(
                float(self._trim_padded_tokens_this_step)
            )
            self._metrics["train"][STEP_TRIMMED_TOKENS_METRIC].append(
                float(self._trim_trimmed_tokens_this_step)
            )
            self._trim_padded_tokens_this_step = 0
            self._trim_trimmed_tokens_this_step = 0
        device = trimmed["completion_ids"].device
        self.microstep_memory.begin(
            device,
            step=int(self.state.global_step),
            microstep=int(self._step),
            sequence_tokens=trimmed["prompt_ids"].numel() + trimmed["completion_ids"].numel(),
            assistant_tokens=int((trimmed["completion_mask"] * trimmed["tool_mask"]).sum().item())
            if device.type == "cuda"
            else 0,
        )
        return trimmed

    def _sampler_correction(
        self, trainer_logprobs: torch.Tensor, inputs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sampler_logprobs: torch.Tensor = inputs["sampling_per_token_logps"]
        assistant_mask: torch.Tensor = inputs["completion_mask"] * inputs["tool_mask"]
        gap = (trainer_logprobs - sampler_logprobs).masked_fill(~assistant_mask.bool(), 0)
        ratio = gap.clamp(max=math.log(self.importance_ratio_cap)).exp()
        stale_mask = inputs.get("sampler_stale_mask")
        if stale_mask is not None and (
            stale_mask.shape != ratio.shape or stale_mask.dtype != torch.bool
        ):
            raise ValueError("sampler stale mask must align with completion logprobs")
        return ratio, gap

    def _record_sampler_correction_metrics(
        self,
        mode: str,
        gap: torch.Tensor,
        assistant_mask: torch.Tensor,
        stale_mask: torch.Tensor,
    ) -> None:
        metric_values = self._sampler_correction_metric_values(gap, assistant_mask, stale_mask)
        if mode == "train":
            self._sampler_correction_metric_rows_this_step.append(metric_values)
            return
        self._append_sampler_correction_metric_values(
            mode,
            cast("list[float]", metric_values.tolist()),
        )

    @override
    def _get_per_token_logps_and_entropies(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        batch_size: int | None = None,
        compute_entropy: bool = False,
        compute_aux_loss: bool = False,
        **multimodal_inputs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Score completions slice by slice, then apply the pending sampler correction."""
        del batch_size
        if compute_aux_loss or any(value is not None for value in multimodal_inputs.values()):
            raise ValueError("Organism log-probabilities support text-only dense models")
        if not self._attention_mask_prevalidated_for_loss and not bool(attention_mask.all()):
            raise ValueError("Chunked organism log-probabilities need an unpadded micro-batch")
        loss_inputs = self._active_loss_inputs
        loss_mask = (
            None
            if loss_inputs is None
            else (loss_inputs["completion_mask"] * loss_inputs["tool_mask"]).bool()
        )
        per_token_logps, entropies = chunked_completion_logps(
            self.accelerator.unwrap_model(model),
            input_ids,
            logits_to_keep,
            temperature=self.temperature,
            chunk_tokens=self.logprob_chunk_tokens,
            compute_entropy=compute_entropy,
            loss_mask=loss_mask,
        )
        aux_loss = None
        if loss_inputs is not None:
            self._active_loss_inputs = None
            ratio, gap = self._sampler_correction(per_token_logps.detach(), loss_inputs)
            advantages: torch.Tensor = loss_inputs["advantages"]
            if advantages.ndim == 1:
                advantages = advantages[:, None]
            loss_inputs["advantages"] = advantages * ratio
            assistant_mask: torch.Tensor = loss_inputs["completion_mask"] * loss_inputs["tool_mask"]
            stale_mask: torch.Tensor = loss_inputs.get(
                "sampler_stale_mask", torch.zeros_like(assistant_mask, dtype=torch.bool)
            )
            if stale_mask.shape != assistant_mask.shape or stale_mask.dtype != torch.bool:
                raise ValueError("sampler stale mask must align with completion logprobs")
            mode = "train" if model.training else "eval"
            self._record_sampler_correction_metrics(mode, gap, assistant_mask, stale_mask)
        return per_token_logps, entropies, aux_loss

    def _compute_loss(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
        if self._active_loss_inputs is not None:
            raise RuntimeError("Organism loss correction cannot be nested")
        advantages: torch.Tensor = inputs["advantages"]
        can_skip_zero_advantage = (
            model.training
            and self.beta == 0
            and self.teacher_beta == 0
            and not self.aux_loss_enabled
        )
        if can_skip_zero_advantage:
            attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
            attention_valid, has_nonzero_advantage = cast(
                "list[bool]",
                torch.stack([attention_mask.all(), torch.any(advantages != 0)]).tolist(),
            )
            if not attention_valid:
                raise ValueError("Chunked organism log-probabilities need an unpadded micro-batch")
            if not has_nonzero_advantage:
                self._rows_skipped_this_step += advantages.shape[0]
                trainable_parameters = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
                if not trainable_parameters:
                    raise RuntimeError("Organism training requires trainable parameters")
                return torch.stack(
                    [parameter.reshape(-1)[0] * 0 for parameter in trainable_parameters]
                ).sum()
            self._attention_mask_prevalidated_for_loss = True
        self._active_loss_inputs = dict(inputs)
        try:
            return super()._compute_loss(model, self._active_loss_inputs)
        finally:
            self._active_loss_inputs = None
            self._attention_mask_prevalidated_for_loss = False

    def _save_checkpoint(self, model: nn.Module, trial: dict[str, Any] | None) -> None:
        optimizer = self._precision_optimizer()
        assert_fp32_training_state(model, optimizer)
        if self.args.output_dir is None:
            raise RuntimeError("Organism checkpoints require an output directory")
        output_dir = Path(self.args.output_dir)
        if active_provenance_path(output_dir).exists():
            record_precision_provenance(output_dir, model, optimizer)
        super()._save_checkpoint(model, trial)
        directory = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
        (directory / "rollout_records.json").write_text(json.dumps(self.step_rollout_records))
        manifest = {
            path.name: _digest(path)
            for path in directory.iterdir()
            if path.is_file() and path.name not in {"complete.json", "complete.json.tmp"}
        }
        temporary_marker = directory / "complete.json.tmp"
        temporary_marker.write_text(json.dumps(manifest, sort_keys=True))
        temporary_marker.replace(directory / "complete.json")
        validate_checkpoint(directory)
        self.step_rollout_records.clear()
        self._prune_retained_checkpoints(directory.parent)

    def _checkpoint_retention_is_complete(self, directory: Path) -> bool:
        return (
            (directory / "complete.json").is_file()
            and (directory / "rollout_records.json").is_file()
            and super()._checkpoint_retention_is_complete(directory)
        )

    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: nn.Module | None = None
    ) -> None:
        validate_checkpoint(Path(resume_from_checkpoint))
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        loaded_model = self.model if model is None else model
        if loaded_model is None:
            raise RuntimeError("Organism trainer requires a model")
        prepare_fp32_head(loaded_model)

    @override
    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        super()._load_optimizer_and_scheduler(checkpoint)
        if self.model is None:
            raise RuntimeError("Organism trainer requires a model")
        assert_fp32_training_state(self.model, self._precision_optimizer())
