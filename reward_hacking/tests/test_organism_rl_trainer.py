"""CPU integration of the exact-id organism GRPO path."""

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainerCallback
from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

from reward_hacking.organism_rl.trainer import (
    OrganismGRPOTrainer,
    _SyncCallback,  # pyright: ignore[reportPrivateUsage]
    _validate_rollout_masks,
    add_teacher_advantage,
    validate_checkpoint,
)


def test_teacher_zero_preserves_plain_advantages() -> None:
    advantages = torch.tensor([1.0, -1.0])
    sampler = torch.tensor([[-2.0, -3.0], [-1.0, -4.0]])
    assert add_teacher_advantage(advantages, sampler, None, 0.0) is advantages
    teacher = sampler + 0.5
    assert torch.equal(
        add_teacher_advantage(advantages, sampler, teacher, 2.0),
        (advantages[:, None] + 1.0).expand_as(sampler),
    )


def _trainer(  # noqa: PLR0913 - focused fixture exposes independent training settings
    directory: Path,
    max_steps: int = 1,
    reward_values: tuple[float | None, float | None] = (0.0, 1.0),
    teacher_beta: float = 0.0,
    sampler_logprob: float = -1.6,
    *,
    checkpoint_keep_latest: int = 2,
    checkpoint_keep_middle: int = 3,
    training_samples: int = 2,
    ragged_completions: bool = False,
    logprob_chunk_tokens: int = 2,
) -> OrganismGRPOTrainer:
    pad_token_name, eos_token_name = "pad", "eos"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"pad": 0, "eos": 1, "task": 2, "act": 3, "env": 4})),
        pad_token=pad_token_name,
        eos_token=eos_token_name,
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=5,
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=32,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            bos_token_id=2,
            eos_token_id=1,
        )
    )
    args = GRPOConfig(
        output_dir=str(directory),
        use_cpu=True,
        bf16=False,
        fp16=False,
        use_vllm=False,
        beta=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_steps=max_steps,
        gradient_checkpointing=False,
        report_to="none",
        save_steps=1,
        logging_steps=1,
        max_completion_length=4,
        loss_type="grpo",
        disable_tqdm=True,
    )

    def rollout(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        del trainer
        if ragged_completions:
            return _ragged_rollout(len(prompts), sampler_logprob)
        return {
            "prompt_ids": [[2] for _ in prompts],
            "completion_ids": [[3, 4, 1] for _ in prompts],
            "logprobs": [[sampler_logprob, 0.0, 0.0] for _ in prompts],
            "teacher_logprobs": [[sampler_logprob + 0.5, 0.0, 0.0] for _ in prompts],
            "env_mask": [
                [int(reward_values[index % 2] is not None), 0, 0] for index in range(len(prompts))
            ],
            "scorable": [reward_values[index % 2] is not None for index in range(len(prompts))],
            "rollout_records": [
                {"prompt_token_ids": [2], "completion_token_ids": [3, 4, 1], "loss_mask": [1, 0, 0]}
                for _ in prompts
            ],
        }

    def reward(completions: list[Any], **kwargs: Any) -> list[float | None]:
        del kwargs
        return [reward_values[index % 2] for index in range(len(completions))]

    return OrganismGRPOTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"prompt": ["task"] * training_samples}),
        processing_class=tokenizer,
        reward_funcs=reward,
        rollout_func=rollout,
        teacher_beta=teacher_beta,
        checkpoint_keep_latest=checkpoint_keep_latest,
        checkpoint_keep_middle=checkpoint_keep_middle,
        logprob_chunk_tokens=logprob_chunk_tokens,
    )


def _ragged_rollout(rows: int, sampler_logprob: float) -> dict[str, Any]:
    """Alternate a six-token and a two-token episode so TRL pads the short one."""
    completions = [[3, 4, 3, 4, 3, 1] if index % 2 == 0 else [3, 1] for index in range(rows)]
    masks = [[1, 0, 1, 0, 1, 0] if index % 2 == 0 else [1, 0] for index in range(rows)]
    return {
        "prompt_ids": [[2] for _ in range(rows)],
        "completion_ids": completions,
        "logprobs": [[sampler_logprob if bit else 0.0 for bit in mask] for mask in masks],
        "teacher_logprobs": [
            [sampler_logprob + 0.5 if bit else 0.0 for bit in mask] for mask in masks
        ],
        "env_mask": masks,
        "scorable": [True] * rows,
        "rollout_records": [
            {"prompt_token_ids": [2], "completion_token_ids": completion, "loss_mask": mask}
            for completion, mask in zip(completions, masks, strict=True)
        ],
    }


def test_chunked_logprobs_match_trl_full_vocabulary_pass(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.model is not None
    trainer.model.train()
    input_ids = torch.tensor([[2, 3, 4, 3, 4, 1, 3, 4, 1]])
    attention_mask = torch.ones_like(input_ids)

    def logprobs_and_gradients(
        function: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        assert trainer.model is not None
        trainer.model.zero_grad()
        logps, entropies, aux = function(
            trainer, trainer.model, input_ids, attention_mask, 5, compute_entropy=True
        )
        assert aux is None
        assert entropies is not None
        (logps * torch.linspace(-1.0, 1.0, logps.size(1))).sum().backward()
        gradients = [
            parameter.grad.clone()
            for parameter in trainer.model.parameters()
            if parameter.grad is not None
        ]
        return logps.detach(), entropies, gradients

    chunked = logprobs_and_gradients(OrganismGRPOTrainer._get_per_token_logps_and_entropies)
    reference = logprobs_and_gradients(GRPOTrainer._get_per_token_logps_and_entropies)
    assert chunked[0].shape == reference[0].shape == (1, 5)
    assert torch.allclose(chunked[0], reference[0], atol=1e-6)
    assert torch.allclose(chunked[1], reference[1], atol=1e-5)
    for chunked_gradient, reference_gradient in zip(chunked[2], reference[2], strict=True):
        assert torch.allclose(chunked_gradient, reference_gradient, atol=1e-6)


def test_chunked_logprobs_refuse_a_padded_micro_batch(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.model is not None
    input_ids = torch.tensor([[0, 2, 3, 4, 1]])
    with pytest.raises(ValueError, match="unpadded"):
        trainer._get_per_token_logps_and_entropies(
            trainer.model, input_ids, torch.tensor([[0, 1, 1, 1, 1]]), 3
        )


def test_ragged_episodes_train_on_trimmed_micro_batches(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, ragged_completions=True)
    assert trainer.model is not None
    backbone_widths: list[int] = []

    def record_width(
        module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        del module, args
        backbone_widths.append(kwargs["input_ids"].shape[1])

    hook = cast("torch.nn.Module", trainer.model.base_model).register_forward_pre_hook(
        record_width, with_kwargs=True
    )
    try:
        trainer.train()
    finally:
        hook.remove()
    assert sorted(backbone_widths) == [3, 7]
    logged = trainer.state.log_history[0]
    assert logged["padding_trim/step_padded_tokens"] == 14
    assert logged["padding_trim/step_trimmed_tokens"] == 10


def test_teacher_token_advantages_are_trimmed_with_the_micro_batch(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, ragged_completions=True, teacher_beta=0.5)
    assert trainer.train().global_step == 1


def test_retained_complete_checkpoints_survive_interruption_and_resume(tmp_path: Path) -> None:
    class InterruptAfterSave(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            if state.global_step == 6:
                raise InterruptedError("synthetic interruption after complete checkpoint")

    trainer = _trainer(tmp_path, max_steps=8, checkpoint_keep_middle=1, training_samples=32)
    trainer.add_callback(InterruptAfterSave())
    with pytest.raises(InterruptedError, match="synthetic interruption"):
        trainer.train()
    retained = sorted(int(path.name.split("-")[1]) for path in tmp_path.glob("checkpoint-*"))
    assert len(retained) == 3
    assert retained[-2:] == [5, 6]
    for step in retained:
        validate_checkpoint(tmp_path / f"checkpoint-{step}")

    resumed = _trainer(tmp_path, max_steps=8, checkpoint_keep_middle=1, training_samples=32)
    assert resumed.train(resume_from_checkpoint=True).global_step == 8
    retained = sorted(int(path.name.split("-")[1]) for path in tmp_path.glob("checkpoint-*"))
    assert len(retained) == 2
    assert retained[-1] == 8
    for step in retained:
        validate_checkpoint(tmp_path / f"checkpoint-{step}")
    assert resumed.train(resume_from_checkpoint=True).global_step == 8


def test_incomplete_organism_save_preserves_previous_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace = Path.replace

    def interrupt_marker(path: Path, target: Path) -> Path:
        if path.name == "complete.json.tmp" and path.parent.name == "checkpoint-4":
            raise OSError("synthetic interruption before organism completion marker")
        return replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupt_marker)
        trainer = _trainer(tmp_path, max_steps=4, checkpoint_keep_middle=0, training_samples=32)
        with pytest.raises(OSError, match="synthetic interruption"):
            trainer.train()
    for step in (2, 3):
        validate_checkpoint(tmp_path / f"checkpoint-{step}")
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        validate_checkpoint(tmp_path / "checkpoint-4")
    resumed = _trainer(tmp_path, max_steps=4, checkpoint_keep_middle=0, training_samples=32)
    assert resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-3")).global_step == 4
    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == ["checkpoint-4"]
    validate_checkpoint(tmp_path / "checkpoint-4")


def test_modified_checkpoint_file_resumes_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.train().global_step == 1
    checkpoint = tmp_path / "checkpoint-1"
    records_path = checkpoint / "rollout_records.json"
    records_path.write_text(records_path.read_text() + " ")

    with caplog.at_level("WARNING", logger="reward_hacking.organism_rl.trainer"):
        validate_checkpoint(checkpoint)
    assert "rollout_records.json" in caplog.text
    assert "sha256" in caplog.text

    resumed = _trainer(tmp_path, max_steps=2)
    assert resumed.train(resume_from_checkpoint=str(checkpoint)).global_step == 2


def test_real_cpu_step_mask_gradient_and_complete_resume(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    result = trainer.train()
    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))
    checkpoint = tmp_path / "checkpoint-1"
    validate_checkpoint(checkpoint)
    resumed = _trainer(tmp_path, max_steps=2)
    assert resumed.train(resume_from_checkpoint=str(checkpoint)).global_step == 2
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        validate_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="checkpoint"):
        _trainer(tmp_path, max_steps=2).train(resume_from_checkpoint=str(checkpoint))

    inputs = trainer._generate_and_score_completions([{"prompt": "task"}, {"prompt": "task"}])
    inputs["advantages"] = torch.tensor([[1.0, 1000.0, 1000.0], [-1.0, 1000.0, 1000.0]])
    assert trainer.model is not None
    trainer.model.zero_grad()
    first_loss = trainer._compute_loss(trainer.model, inputs)
    first_loss.backward()
    gradients = [
        parameter.grad.clone()
        for parameter in trainer.model.parameters()
        if parameter.grad is not None
    ]
    trainer.model.zero_grad()
    inputs["advantages"][:, 1:] = 0.0
    second_loss = trainer._compute_loss(trainer.model, inputs)
    second_loss.backward()
    assert torch.equal(first_loss, second_loss)
    second_gradients = [
        parameter.grad for parameter in trainer.model.parameters() if parameter.grad is not None
    ]
    for first, second in zip(gradients, second_gradients, strict=True):
        assert torch.equal(first, second)


def test_sabotaged_environment_mask_is_refused() -> None:
    output: dict[str, Any] = {
        "prompt_ids": [[2]],
        "completion_ids": [[3, 4, 1]],
        "env_mask": [[1, 1, 1]],
        "scorable": [True],
        "rollout_records": [
            {"prompt_token_ids": [2], "completion_token_ids": [3, 4, 1], "loss_mask": [1, 0, 0]}
        ],
    }
    with pytest.raises(ValueError, match="env_mask"):
        _validate_rollout_masks(output)
    output["env_mask"] = [[1, 0, 0]]
    _validate_rollout_masks(output)


def test_unscorable_reward_excludes_baseline_and_teacher_loss(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, reward_values=(None, 1.0), teacher_beta=2.0)
    inputs = trainer._generate_and_score_completions([{"prompt": "task"}, {"prompt": "task"}])
    assert not inputs["tool_mask"][0].any()
    assert inputs["advantages"][1, 0].item() == 1.0
    assert trainer._metrics["train"]["organism/reward_advantage"][-1] == 0.0


def test_sampler_correction_is_capped_only_on_assistant_tokens(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, sampler_logprob=-100.0)
    assert trainer.model is not None
    inputs = trainer._generate_and_score_completions([{"prompt": "task"}, {"prompt": "task"}])
    assert inputs["advantages"].ndim == 1

    with torch.no_grad():
        current_logprobs, _, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model,
            torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1),
            torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1),
            inputs["completion_ids"].shape[1],
        )
    ratio, _ = trainer._sampler_correction(current_logprobs, inputs)
    assert torch.equal(ratio[:, 0], torch.full((2,), 2.0))
    assert torch.equal(ratio[:, 1:], torch.ones((2, 2)))


def test_training_step_uses_one_forward_per_microbatch(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.model is not None
    forward_batch_sizes: list[int] = []

    def record_batch(
        module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        del module, args
        input_ids: torch.Tensor = kwargs["input_ids"]
        forward_batch_sizes.append(input_ids.shape[0])

    hook = cast("torch.nn.Module", trainer.model.base_model).register_forward_pre_hook(
        record_batch, with_kwargs=True
    )
    try:
        trainer.train()
    finally:
        hook.remove()
    assert forward_batch_sizes == [1, 1]


def test_zero_advantage_skip_preserves_all_parameter_gradients(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, reward_values=(0.0, 0.0))
    assert trainer.model is not None
    inputs = trainer._generate_and_score_completions([{"prompt": "task"}, {"prompt": "task"}])
    assert torch.equal(inputs["advantages"], torch.zeros(2))
    microbatch = {
        key: value[:1] if isinstance(value, torch.Tensor) and value.ndim > 0 else value
        for key, value in inputs.items()
    }
    trainer.current_gradient_accumulation_steps = trainer.args.gradient_accumulation_steps
    parameters = [parameter for parameter in trainer.model.parameters() if parameter.requires_grad]

    reference_loss = GRPOTrainer._compute_loss(trainer, trainer.model, microbatch)
    reference_loss.backward()
    reference_gradients = [
        parameter.grad.clone() for parameter in parameters if parameter.grad is not None
    ]
    assert len(reference_gradients) == len(parameters)
    trainer.model.zero_grad(set_to_none=True)

    forward_calls: list[int] = []
    hook = cast("torch.nn.Module", trainer.model.base_model).register_forward_pre_hook(
        lambda _module, _args: forward_calls.append(1)
    )
    try:
        skipped_loss = trainer._compute_loss(trainer.model, microbatch)
        skipped_loss.backward()
    finally:
        hook.remove()
    assert forward_calls == []
    assert skipped_loss.item() == reference_loss.item()
    for parameter, expected_gradient in zip(parameters, reference_gradients, strict=True):
        assert parameter.grad is not None
        assert torch.allclose(parameter.grad, expected_gradient, atol=1e-7, rtol=0)


def test_zero_advantage_step_reports_skips_and_unmeasured_gap(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, reward_values=(0.0, 0.0))
    trainer.train()
    metrics = trainer.state.log_history[0]
    assert metrics["organism/rows_skipped_zero_advantage"] == 2
    assert metrics["organism/trainer_sampler_logprob_gap_tokens"] == 0
    assert metrics["organism/trainer_sampler_logprob_gap_abs"] is None


def test_teacher_term_prevents_zero_advantage_skip(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, reward_values=(0.0, 0.0), teacher_beta=1.0)
    assert trainer.model is not None
    inputs = trainer._generate_and_score_completions([{"prompt": "task"}, {"prompt": "task"}])
    assert inputs["advantages"][0, 0] != 0
    microbatch = {
        key: value[:1] if isinstance(value, torch.Tensor) and value.ndim > 0 else value
        for key, value in inputs.items()
    }
    trainer.current_gradient_accumulation_steps = trainer.args.gradient_accumulation_steps
    forward_calls: list[int] = []
    hook = cast("torch.nn.Module", trainer.model.base_model).register_forward_pre_hook(
        lambda _module, _args: forward_calls.append(1)
    )
    try:
        trainer._compute_loss(trainer.model, microbatch)
    finally:
        hook.remove()
    assert forward_calls == [1]


def test_mixed_accumulation_keeps_surviving_row_gradient_scale(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.model is not None
    inputs = trainer._generate_and_score_completions([{"prompt": "task"}, {"prompt": "task"}])
    inputs["advantages"][0] = 0
    microbatches = [
        {
            key: value[index : index + 1]
            if isinstance(value, torch.Tensor) and value.ndim > 0
            else value
            for key, value in inputs.items()
        }
        for index in range(2)
    ]
    trainer.current_gradient_accumulation_steps = 2
    parameters = [parameter for parameter in trainer.model.parameters() if parameter.requires_grad]

    trainer.teacher_beta = 1.0
    for microbatch in microbatches:
        trainer._compute_loss(trainer.model, microbatch).backward()
    reference_gradients = [
        parameter.grad.clone() for parameter in parameters if parameter.grad is not None
    ]
    trainer.model.zero_grad(set_to_none=True)

    trainer.teacher_beta = 0.0
    for microbatch in microbatches:
        trainer._compute_loss(trainer.model, microbatch).backward()
    assert len(reference_gradients) == len(parameters)
    for parameter, expected_gradient in zip(parameters, reference_gradients, strict=True):
        assert parameter.grad is not None
        assert torch.allclose(parameter.grad, expected_gradient, atol=1e-7, rtol=0)


def test_sampler_correction_matches_previous_forward_and_metric(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.model is not None
    trainer.model.eval()
    inputs = trainer._generate_and_score_completions([{"prompt": "task"}, {"prompt": "task"}])
    assistant_mask = inputs["completion_mask"] * inputs["tool_mask"]

    with torch.no_grad():
        previous_logprobs, _, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model,
            torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1),
            torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1),
            inputs["completion_ids"].shape[1],
            trainer.args.per_device_train_batch_size,
        )
    expected_ratio, expected_gap = trainer._sampler_correction(previous_logprobs, inputs)
    expected_metric = expected_gap.sum().item() / max(assistant_mask.sum().item(), 1)

    expected_inputs = dict(inputs)
    expected_advantages = inputs["advantages"]
    if expected_advantages.ndim == 1:
        expected_advantages = expected_advantages[:, None]
    expected_inputs["advantages"] = expected_advantages * expected_ratio
    expected_inputs["old_per_token_logps"] = previous_logprobs
    expected_loss = GRPOTrainer._compute_loss(trainer, trainer.model, expected_inputs)

    trainer._metrics["eval"]["organism/trainer_sampler_logprob_gap"].clear()
    actual_loss = trainer._compute_loss(trainer.model, inputs)

    assert torch.allclose(actual_loss, expected_loss)
    assert trainer._metrics["eval"]["organism/trainer_sampler_logprob_gap"][-1] == pytest.approx(
        expected_metric
    )
    assert trainer._metrics["eval"]["organism/trainer_sampler_logprob_gap_abs"][
        -1
    ] == pytest.approx(expected_gap.abs().sum().item() / assistant_mask.sum().item())
    assert trainer._metrics["eval"]["organism/trainer_sampler_logprob_gap_tokens"][-1] == 2


def test_resuming_already_complete_run_does_not_take_another_step(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.train().global_step == 1
    resumed = _trainer(tmp_path)
    assert resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-1")).global_step == 1
    assert not (tmp_path / "checkpoint-2").exists()
    assert trainer.model is not None
    assert resumed.model is not None
    for expected, restored in zip(
        trainer.model.parameters(), resumed.model.parameters(), strict=True
    ):
        assert torch.equal(expected, restored)
    assert resumed.train(resume_from_checkpoint=True).global_step == 1


def test_resuming_completed_run_applies_final_retention(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, max_steps=5, checkpoint_keep_middle=1, training_samples=32)

    class InterruptFinalSave(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            if state.global_step == 5:
                raise InterruptedError("synthetic interruption before final retention")

    trainer.add_callback(InterruptFinalSave())
    with pytest.raises(InterruptedError, match="synthetic interruption"):
        trainer.train()
    assert len(list(tmp_path.glob("checkpoint-*"))) == 3
    resumed = _trainer(tmp_path, max_steps=5, checkpoint_keep_middle=1, training_samples=32)
    assert resumed.train(resume_from_checkpoint=True).global_step == 5
    retained = sorted(tmp_path.glob("checkpoint-*"))
    assert len(retained) == 2
    assert retained[-1].name == "checkpoint-5"
    for checkpoint in retained:
        validate_checkpoint(checkpoint)


def test_phase_transitions_are_timed_into_the_step_log(tmp_path: Path) -> None:
    class RecordingPhase:
        def initialize(self, trainer: Any, step: int = 0) -> None:
            del trainer, step

        def before_sampling(self, trainer: Any) -> None:
            del trainer

        def before_training(self, trainer: Any) -> None:
            del trainer

        def after_optimizer_step(self, trainer: Any) -> None:
            del trainer

    trainer = _trainer(tmp_path, max_steps=2, training_samples=4)
    trainer.phase = RecordingPhase()
    trainer.add_callback(_SyncCallback(trainer, trainer.phase))
    trainer.train()
    first, second = trainer.state.log_history[:2]
    for name in ("wake_sampler", "rollout", "restore_trainer"):
        assert first[f"organism/seconds_{name}"] >= 0
    assert second["organism/seconds_adapter_sync"] >= 0
