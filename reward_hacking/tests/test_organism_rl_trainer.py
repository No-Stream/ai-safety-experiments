"""CPU integration of the exact-id organism GRPO path."""

from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

from reward_hacking.organism_rl.trainer import (
    OrganismGRPOTrainer,
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


def _trainer(
    directory: Path,
    max_steps: int = 1,
    reward_values: tuple[float | None, float | None] = (0.0, 1.0),
    teacher_beta: float = 0.0,
    sampler_logprob: float = -1.6,
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
        train_dataset=Dataset.from_dict({"prompt": ["task", "task"]}),
        processing_class=tokenizer,
        reward_funcs=reward,
        rollout_func=rollout,
        teacher_beta=teacher_beta,
    )


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

    hook = trainer.model.register_forward_pre_hook(record_batch, with_kwargs=True)
    try:
        trainer.train()
    finally:
        hook.remove()
    assert forward_batch_sizes == [1, 1]


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
