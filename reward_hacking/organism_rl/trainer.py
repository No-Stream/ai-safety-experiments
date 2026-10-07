"""Exact-id GRPO with external sampling, teacher advantages, and complete checkpoints.

Unscorable rewards must return None: TRL excludes them from the group baseline.
Their entire assistant loss mask is also zeroed, including the teacher term.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol, cast, override

import torch
from torch import nn
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import TrainOutput, get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]


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
    """Refuse missing states and modifications since the completion marker was written."""
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
    for name, expected in manifest.items():
        path = directory / name
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Incomplete or modified checkpoint file: {name}")


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

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        del args, state, control, kwargs
        self.phase.after_optimizer_step(self.trainer)


class OrganismGRPOTrainer(GRPOTrainer):
    """Keep native GRPO loss and apply detached sampler correction to token advantages."""

    def __init__(
        self,
        *args: Any,  # noqa: ANN401
        teacher_beta: float = 0.0,
        importance_ratio_cap: float = 2.0,
        phase: TrainingPhase | None = None,
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
        self.phase = phase
        self.latest_rollout: dict[str, Any] = {}
        self.step_rollout_records: list[Any] = []

        def captured_rollout(prompts: list[Any], trainer: "OrganismGRPOTrainer") -> dict[str, Any]:
            if self.phase is not None:
                self.phase.before_sampling(self)
            output: dict[str, Any] = rollout(prompts, trainer)
            _validate_rollout_masks(output)
            self.latest_rollout = output
            self.step_rollout_records.extend(output["rollout_records"])
            if self.phase is not None:
                self.phase.before_training(self)
            return output

        kwargs["rollout_func"] = captured_rollout
        super().__init__(*args, **kwargs)
        self._validate_configuration()
        if phase is not None:
            self.add_callback(_SyncCallback(self, phase))

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
                return TrainOutput(
                    saved_state.global_step, 0.0, {"organism/resume_already_complete": 1.0}
                )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval)

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = super()._generate_and_score_completions(inputs)
        sampler: torch.Tensor = batch["sampling_per_token_logps"]
        mask: torch.Tensor = batch["completion_mask"] * batch["tool_mask"]
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
        with torch.no_grad():
            old_logprobs, _, _ = self._get_per_token_logps_and_entropies(
                self.model,
                torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1),
                torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1),
                sampler.shape[1],
                self.args.per_device_train_batch_size,
            )
        gap = (old_logprobs - sampler).masked_fill(~mask.bool(), 0)
        batch["old_per_token_logps"] = old_logprobs
        batch["organism_importance_ratio"] = gap.clamp(
            max=math.log(self.importance_ratio_cap)
        ).exp()
        self._metrics[mode]["organism/trainer_sampler_logprob_gap"].append(
            gap.sum().item() / max(mask.sum().item(), 1)
        )
        return batch

    def _compute_loss(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
        corrected = dict(inputs)
        advantages: torch.Tensor = inputs["advantages"]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        corrected["advantages"] = advantages * inputs["organism_importance_ratio"]
        return super()._compute_loss(model, corrected)

    def _save_checkpoint(self, model: nn.Module, trial: dict[str, Any] | None) -> None:
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

    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: nn.Module | None = None
    ) -> None:
        validate_checkpoint(Path(resume_from_checkpoint))
        super()._load_from_checkpoint(resume_from_checkpoint, model)
