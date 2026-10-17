"""CPU integration of the exact-id organism GRPO path."""

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainerCallback
from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

import reward_hacking.distill.train as distill_train
from reward_hacking.distill.train import chunked_completion_logps
from reward_hacking.model_backend import VLLMRequestGate
from reward_hacking.organism_rl.trainer import (
    OrganismGRPOTrainer,
    _SyncCallback,  # pyright: ignore[reportPrivateUsage]
    _validate_rollout_masks,
    add_teacher_advantage,
    parse_vllm_rollout_log,
    validate_checkpoint,
)
from reward_hacking.organism_rl.transfers import PolicyTransfers


def test_teacher_zero_preserves_plain_advantages() -> None:
    advantages = torch.tensor([1.0, -1.0])
    sampler = torch.tensor([[-2.0, -3.0], [-1.0, -4.0]])
    assert add_teacher_advantage(advantages, sampler, None, 0.0) is advantages
    teacher = sampler + 0.5
    assert torch.equal(
        add_teacher_advantage(advantages, sampler, teacher, 2.0),
        (advantages[:, None] + 1.0).expand_as(sampler),
    )


def test_vllm_rollout_log_reports_latest_prefix_rate_and_sampled_peak_kv() -> None:
    log = "\n".join(
        f"Engine 000: GPU KV cache usage: {usage}%, Prefix cache hit rate: {hit}%"
        for usage, hit in ((4.9, 68.5), (12.0, 70.1), (8.0, 71.2))
    )
    assert parse_vllm_rollout_log(log) == (71.2, 12.0, 3)
    latest, peak, samples = parse_vllm_rollout_log("server starting")
    assert samples == 0
    assert math.isnan(latest)
    assert math.isnan(peak)


def test_step_metrics_read_only_log_lines_from_its_rollout(tmp_path: Path) -> None:
    server_log_path = tmp_path / "vllm-server.log"
    server_log_path.write_text("GPU KV cache usage: 95.0%, Prefix cache hit rate: 1.0%\n")
    trainer = _trainer(
        tmp_path,
        sampler_log_path=server_log_path,
        rollout_log_line="Engine 000: GPU KV cache usage: 12.5%, Prefix cache hit rate: 71.2%",
    )
    trainer.train()
    metrics = trainer.state.log_history[0]
    assert metrics["organism/vllm_prefix_hit_rate_latest_pct"] == 71.2
    assert metrics["organism/vllm_gpu_kv_usage_max_logged_pct"] == 12.5
    assert metrics["organism/vllm_log_samples"] == 1


def _trainer(  # noqa: PLR0913 - focused fixture exposes independent training settings
    directory: Path,
    max_steps: int = 1,
    reward_values: tuple[float | None, ...] = (0.0, 1.0),
    teacher_beta: float = 0.0,
    sampler_logprob: float = -1.6,
    *,
    checkpoint_keep_latest: int = 2,
    checkpoint_keep_middle: int = 3,
    importance_ratio_cap: float = 2.0,
    training_samples: int = 2,
    prompts_per_step: int = 1,
    generations: int = 2,
    rollout_batches: list[list[str]] | None = None,
    next_prompt_batches: list[list[str]] | None = None,
    training_prompts: tuple[str, ...] | None = None,
    sampler_log_path: Path | None = None,
    rollout_log_line: str | None = None,
    ragged_completions: bool = False,
    logprob_chunk_tokens: int = 2,
    phase: Any = None,
    request_gate: VLLMRequestGate | None = None,
    sampler_admission: Any = None,
    sampler_model_id: Any = None,
) -> OrganismGRPOTrainer:
    pad_token_name, eos_token_name = "pad", "eos"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"pad": 0, "eos": 1, "task": 2, "act": 3, "env": 4})),
        pad_token=pad_token_name,
        eos_token=eos_token_name,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
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
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            fan_in_fan_out=True,
            target_modules=["c_attn"],
        ),
    )
    args = GRPOConfig(
        output_dir=str(directory),
        use_cpu=True,
        bf16=False,
        fp16=False,
        use_vllm=False,
        beta=0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=generations * prompts_per_step,
        steps_per_generation=generations * prompts_per_step,
        num_generations=generations,
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
        if rollout_log_line is not None and sampler_log_path is not None:
            with sampler_log_path.open("a") as server_log_file:
                server_log_file.write(rollout_log_line + "\n")
        if rollout_batches is not None:
            rollout_batches.append(list(prompts))
        if next_prompt_batches is not None:
            next_prompt_batches.append(list(trainer.next_prompts))
        if ragged_completions:
            return _ragged_rollout(len(prompts), sampler_logprob)
        output = {
            "prompt_ids": [[2] for _ in prompts],
            "completion_ids": [[3, 4, 1] for _ in prompts],
            "logprobs": [[sampler_logprob, 0.0, 0.0] for _ in prompts],
            "teacher_logprobs": [[sampler_logprob + 0.5, 0.0, 0.0] for _ in prompts],
            "env_mask": [
                [int(reward_values[index % len(reward_values)] is not None), 0, 0]
                for index in range(len(prompts))
            ],
            "scorable": [
                reward_values[index % len(reward_values)] is not None
                for index in range(len(prompts))
            ],
            "rollout_records": [
                {"prompt_token_ids": [2], "completion_token_ids": [3, 4, 1], "loss_mask": [1, 0, 0]}
                for _ in prompts
            ],
        }
        if request_gate is not None:
            output["policy_steps"] = [[trainer.state.global_step] * 3 for _ in prompts]
        return output

    def reward(completions: list[Any], **kwargs: Any) -> list[float | None]:
        del kwargs
        return [reward_values[index % len(reward_values)] for index in range(len(completions))]

    return OrganismGRPOTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict(
            {
                "prompt": [
                    training_prompts[index % len(training_prompts)]
                    if training_prompts is not None
                    else "act"
                    if prompts_per_step > 1 and index % 2
                    else "task"
                    for index in range(training_samples)
                ]
            }
        ),
        processing_class=tokenizer,
        reward_funcs=reward,
        rollout_func=rollout,
        teacher_beta=teacher_beta,
        importance_ratio_cap=importance_ratio_cap,
        sampler_log_path=sampler_log_path,
        checkpoint_keep_latest=checkpoint_keep_latest,
        checkpoint_keep_middle=checkpoint_keep_middle,
        logprob_chunk_tokens=logprob_chunk_tokens,
        phase=phase,
        request_gate=request_gate,
        sampler_admission=sampler_admission,
        sampler_model_id=sampler_model_id,
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


def test_masked_chunked_logprobs_match_full_head_loss_and_gradients(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, logprob_chunk_tokens=2)
    assert trainer.model is not None
    trainer.model.eval()
    input_ids = torch.tensor([[2, 3, 4, 3, 4, 1, 3, 4, 1]])
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.tensor([[True, False, True, False, True]])
    completion_length = completion_mask.size(1)
    loss_weights = torch.tensor([[0.5, -0.7, 1.2, 0.3, -0.4]])

    def score_and_backpropagate(
        use_gathered_head: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        assert trainer.model is not None
        trainer.model.zero_grad()
        if use_gathered_head:
            logps, entropies = chunked_completion_logps(
                trainer.model,
                input_ids,
                completion_length,
                temperature=trainer.temperature,
                chunk_tokens=trainer.logprob_chunk_tokens,
                compute_entropy=True,
                loss_mask=completion_mask,
            )
            assert entropies is not None
        else:
            logps, entropies, auxiliary_loss = GRPOTrainer._get_per_token_logps_and_entropies(
                trainer,
                trainer.model,
                input_ids,
                attention_mask,
                completion_length,
                compute_entropy=True,
            )
            assert auxiliary_loss is None
            assert entropies is not None
        loss = -(logps.masked_select(completion_mask) * loss_weights[completion_mask]).mean()
        loss.backward()
        gradients = [
            parameter.grad.clone()
            for parameter in trainer.model.parameters()
            if parameter.grad is not None
        ]
        return logps.detach(), entropies.detach(), loss.detach(), gradients

    gathered = score_and_backpropagate(use_gathered_head=True)
    full = score_and_backpropagate(use_gathered_head=False)
    assert torch.allclose(gathered[0][completion_mask], full[0][completion_mask], atol=1e-6)
    assert torch.equal(
        gathered[0][~completion_mask], torch.zeros_like(gathered[0][~completion_mask])
    )
    assert torch.allclose(gathered[1][completion_mask], full[1][completion_mask], atol=1e-5)
    assert torch.equal(
        gathered[1][~completion_mask], torch.zeros_like(gathered[1][~completion_mask])
    )
    assert gathered[1][completion_mask].mean().item() == pytest.approx(
        full[1][completion_mask].mean().item()
    )
    assert torch.allclose(gathered[2], full[2], atol=1e-6)
    for gathered_gradient, full_gradient in zip(gathered[3], full[3], strict=True):
        assert torch.allclose(gathered_gradient, full_gradient, atol=1e-6)


def test_masked_chunked_grpo_loss_and_gradients_match_full_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _trainer(tmp_path, logprob_chunk_tokens=2)
    assert trainer.model is not None
    trainer.model.eval()
    input_ids = torch.tensor([[2, 3, 4, 3, 4, 1, 3, 4, 1]])
    completion_ids = input_ids[:, -5:]
    completion_mask = torch.ones_like(completion_ids)
    tool_mask = torch.tensor([[True, False, True, False, True]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        sampling_logps, _, auxiliary_loss = GRPOTrainer._get_per_token_logps_and_entropies(
            trainer,
            trainer.model,
            input_ids,
            attention_mask,
            completion_ids.size(1),
        )
    assert auxiliary_loss is None
    loss_inputs = {
        "prompt_ids": input_ids[:, :-5],
        "prompt_mask": torch.ones_like(input_ids[:, :-5]),
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "tool_mask": tool_mask,
        "advantages": torch.tensor([0.75]),
        "sampling_per_token_logps": sampling_logps.detach(),
    }

    def loss_and_gradients() -> tuple[torch.Tensor, list[torch.Tensor]]:
        assert trainer.model is not None
        trainer.model.zero_grad()
        loss = trainer._compute_loss(trainer.model, dict(loss_inputs))
        loss.backward()
        gradients = [
            parameter.grad.clone()
            for parameter in trainer.model.parameters()
            if parameter.grad is not None
        ]
        return loss.detach(), gradients

    gathered = loss_and_gradients()
    monkeypatch.setattr(
        OrganismGRPOTrainer,
        "_get_per_token_logps_and_entropies",
        GRPOTrainer._get_per_token_logps_and_entropies,
    )
    full = loss_and_gradients()
    assert torch.allclose(gathered[0], full[0], atol=1e-6)
    for gathered_gradient, full_gradient in zip(gathered[1], full[1], strict=True):
        assert torch.allclose(gathered_gradient, full_gradient, atol=1e-6)


def test_masked_chunked_logprobs_preserve_metrics_and_skip_inactive_head_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _trainer(tmp_path, logprob_chunk_tokens=2)
    assert trainer.model is not None
    trainer.model.train()
    input_ids = torch.tensor([[2, 3, 4, 3, 4, 1, 3, 4, 1]])
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.tensor([[True, False, True, False, True]])
    completion_length = completion_mask.size(1)
    full_logps, full_entropies, auxiliary_loss = GRPOTrainer._get_per_token_logps_and_entropies(
        trainer,
        trainer.model,
        input_ids,
        attention_mask,
        completion_length,
        compute_entropy=True,
    )
    assert auxiliary_loss is None
    assert full_entropies is not None
    sampler_logps = full_logps + torch.tensor([[0.1, 8.0, -0.2, 8.0, 0.3]])
    trainer._active_loss_inputs = {
        "completion_mask": torch.ones_like(completion_mask),
        "tool_mask": completion_mask,
        "sampling_per_token_logps": sampler_logps,
        "advantages": torch.ones((1,)),
    }
    head_position_counts: list[int] = []
    original_head_logits = distill_train._fp32_head_logits

    def record_head_positions(
        hidden_slice: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        head_position_counts.append(hidden_slice.size(1))
        return original_head_logits(hidden_slice, weight, bias)

    monkeypatch.setattr(distill_train, "_fp32_head_logits", record_head_positions)
    with torch.no_grad():
        logps, entropies, auxiliary_loss = trainer._get_per_token_logps_and_entropies(
            trainer.model,
            input_ids,
            attention_mask,
            completion_length,
            compute_entropy=True,
        )
    assert auxiliary_loss is None
    assert entropies is not None
    assert sum(head_position_counts) == int(completion_mask.sum())
    assert trainer._active_loss_inputs is None
    trainer._finish_training_step_metrics()
    active_gap = (full_logps - sampler_logps).masked_select(completion_mask)
    metrics = trainer._metrics["train"]
    assert metrics["organism/trainer_sampler_logprob_gap"][-1] == pytest.approx(
        active_gap.mean().item()
    )
    assert metrics["organism/trainer_sampler_logprob_gap_abs"][-1] == pytest.approx(
        active_gap.abs().mean().item()
    )
    assert metrics["organism/trainer_sampler_logprob_gap_tokens"][-1] == int(completion_mask.sum())
    assert torch.allclose(logps[completion_mask], full_logps[completion_mask], atol=1e-6)
    assert torch.equal(logps[~completion_mask], torch.zeros_like(logps[~completion_mask]))
    assert torch.allclose(entropies[completion_mask], full_entropies[completion_mask], atol=1e-5)
    assert torch.equal(entropies[~completion_mask], torch.zeros_like(entropies[~completion_mask]))


def test_empty_loss_mask_returns_zero_scores_without_running_the_head(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, logprob_chunk_tokens=2)
    assert trainer.model is not None
    input_ids = torch.tensor([[2, 3, 4, 3, 4]])
    loss_mask = torch.zeros((1, 3), dtype=torch.bool)
    base_model = cast("Any", trainer.model).get_base_model()
    head = cast("GPT2LMHeadModel", base_model).get_output_embeddings()
    assert head is not None
    head_calls = 0

    def count_head_calls(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        del module, args
        nonlocal head_calls
        head_calls += 1

    head_hook = head.register_forward_pre_hook(count_head_calls)
    try:
        logps, entropies = chunked_completion_logps(
            trainer.model,
            input_ids,
            loss_mask.size(1),
            temperature=trainer.temperature,
            chunk_tokens=trainer.logprob_chunk_tokens,
            compute_entropy=True,
            loss_mask=loss_mask,
        )
    finally:
        head_hook.remove()
    assert head_calls == 0
    assert torch.equal(logps, torch.zeros_like(logps))
    assert torch.isfinite(logps).all()
    assert logps.requires_grad
    assert entropies is not None
    assert torch.equal(entropies, torch.zeros_like(entropies))
    logps.sum().backward()


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

    base_model = cast("GPT2LMHeadModel", cast("Any", trainer.model).get_base_model())
    hook = base_model.transformer.register_forward_pre_hook(record_width, with_kwargs=True)
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


def test_sampler_correction_applies_to_fresh_and_stale_assistant_tokens(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    inputs = {
        "sampling_per_token_logps": torch.tensor([[-100.0, -100.0, -100.0, -100.0]]),
        "completion_mask": torch.ones((1, 4)),
        "tool_mask": torch.tensor([[1, 1, 1, 0]]),
        "sampler_stale_mask": torch.tensor([[False, True, False, True]]),
    }
    ratio, _ = trainer._sampler_correction(torch.zeros((1, 4)), inputs)
    assert torch.equal(ratio, torch.tensor([[2.0, 2.0, 2.0, 1.0]]))


def test_sampler_correction_uses_configured_truncation_cap(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, importance_ratio_cap=1.25)
    inputs = {
        "sampling_per_token_logps": torch.tensor([[-100.0, -100.0]]),
        "completion_mask": torch.ones((1, 2)),
        "tool_mask": torch.ones((1, 2)),
    }
    ratio, _ = trainer._sampler_correction(torch.zeros((1, 2)), inputs)
    assert torch.equal(ratio, torch.full((1, 2), 1.25))


def test_tis_gradient_matches_hand_computed_policy_gradient(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, importance_ratio_cap=1.5)
    assert trainer.model is not None
    trainer.model.eval()
    inputs = trainer._generate_and_score_completions([{"prompt": "task"}, {"prompt": "task"}])
    inputs["completion_mask"] = torch.ones_like(inputs["completion_mask"])
    inputs["tool_mask"] = torch.tensor([[1, 1, 0], [1, 1, 0]])
    inputs["sampler_stale_mask"] = torch.tensor([[False, False, False], [True, True, False]])
    inputs["advantages"] = torch.tensor([1.25, -0.6])
    assistant_mask = (inputs["completion_mask"] * inputs["tool_mask"]).bool()
    input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
    attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
    selected_trainer_logprobs, _, _ = trainer._get_per_token_logps_and_entropies(
        trainer.model,
        input_ids,
        attention_mask,
        inputs["completion_ids"].shape[1],
    )
    selected_log_ratios = torch.tensor([[0.8, 0.1, 10.0], [math.log(0.5), math.log(3.0), 10.0]])
    inputs["sampling_per_token_logps"] = selected_trainer_logprobs.detach() - selected_log_ratios
    inputs["old_per_token_logps"] = selected_trainer_logprobs.detach()
    expected_ratio = (
        selected_log_ratios.masked_fill(~assistant_mask, 0)
        .clamp(max=math.log(trainer.importance_ratio_cap))
        .exp()
    )
    assert torch.allclose(
        expected_ratio, torch.tensor([[1.5, math.exp(0.1), 1.0], [0.5, 1.5, 1.0]])
    )
    expected_advantages = inputs["advantages"][:, None] * expected_ratio
    policy_ratio = (selected_trainer_logprobs - inputs["old_per_token_logps"]).exp()
    expected_loss = -(
        (policy_ratio * expected_advantages * assistant_mask).sum(dim=-1)
        / assistant_mask.sum(dim=-1)
    ).mean()
    trainable_parameters = [
        parameter for parameter in trainer.model.parameters() if parameter.requires_grad
    ]
    expected_gradients = torch.autograd.grad(expected_loss, trainable_parameters)

    actual_loss = trainer._compute_loss(trainer.model, inputs)
    actual_gradients = torch.autograd.grad(actual_loss, trainable_parameters)

    assert torch.allclose(actual_loss, expected_loss, atol=1e-6)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-5)


def test_trainer_drains_requests_before_sleep_and_reopens_after_sync(tmp_path: Path) -> None:
    events: list[str] = []
    gate = VLLMRequestGate("initial", policy_step=0)
    model_id = "initial"
    awake = False

    class FakeAdmission:
        def suspend(self) -> None:
            events.append("suspend-admission")

        def reset_step(self, *, preserve_pending_requests: bool = False) -> None:
            assert gate.is_paused
            assert gate.active_requests == 0
            assert preserve_pending_requests
            events.append("clear-prefixes")

        def resume(self) -> None:
            assert gate.is_paused
            assert awake
            events.append("resume-admission")

    class FakePhase:
        def initialize(self, trainer: Any, step: int = 0) -> None:
            nonlocal awake
            del trainer, step
            awake = True
            events.append("initialize")

        def before_sampling(self, trainer: Any) -> None:
            del trainer
            assert awake
            assert not gate.is_paused
            events.append("sample")

        def before_training(self, trainer: Any) -> None:
            nonlocal awake
            del trainer
            assert gate.is_paused
            awake = False
            events.append("sleep")

        def after_optimizer_step(self, trainer: Any) -> None:
            nonlocal awake, model_id
            del trainer
            assert gate.is_paused
            awake = True
            model_id = "updated"
            events.append("sync")

    trainer = _trainer(
        tmp_path,
        max_steps=2,
        training_samples=4,
        phase=FakePhase(),
        request_gate=gate,
        sampler_admission=FakeAdmission(),
        sampler_model_id=lambda: model_id,
    )
    trainer.train()
    assert events == [
        "initialize",
        "sample",
        "suspend-admission",
        "clear-prefixes",
        "sleep",
        "sync",
        "resume-admission",
        "sample",
        "suspend-admission",
        "clear-prefixes",
        "sleep",
        "sync",
        "resume-admission",
    ]
    with gate.request() as selected:
        assert selected.model_id == "updated"
        assert selected.policy_step == 2


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

    base_model = cast("GPT2LMHeadModel", cast("Any", trainer.model).get_base_model())
    hook = base_model.transformer.register_forward_pre_hook(record_batch, with_kwargs=True)
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
    base_model = cast("GPT2LMHeadModel", cast("Any", trainer.model).get_base_model())
    hook = base_model.transformer.register_forward_pre_hook(
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
    base_model = cast("GPT2LMHeadModel", cast("Any", trainer.model).get_base_model())
    hook = base_model.transformer.register_forward_pre_hook(
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


def test_two_prompt_groups_share_one_optimizer_step(tmp_path: Path) -> None:
    rollout_batches: list[list[str]] = []
    trainer = _trainer(
        tmp_path,
        reward_values=(0.0, 1.0, 0.0, 0.0),
        prompts_per_step=2,
        rollout_batches=rollout_batches,
    )
    batch = trainer._generate_and_score_completions(
        [{"prompt": "task"}, {"prompt": "task"}, {"prompt": "act"}, {"prompt": "act"}]
    )
    assert batch["advantages"][0] < 0 < batch["advantages"][1]
    assert torch.equal(batch["advantages"][2:], torch.zeros(2))

    trainer.step_rollout_records.clear()
    rollout_batches.clear()
    trainer.train()
    assert len(rollout_batches) == 1
    assert len(rollout_batches[0]) == 4
    assert set(rollout_batches[0]) == {"task", "act"}
    assert len(json.loads((tmp_path / "checkpoint-1" / "rollout_records.json").read_text())) == 4


def test_next_generation_lookahead_follows_trl_sampler_and_keeps_eight_sample_groups(
    tmp_path: Path,
) -> None:
    rollout_batches: list[list[str]] = []
    next_prompt_batches: list[list[str]] = []
    trainer = _trainer(
        tmp_path,
        max_steps=2,
        training_samples=4,
        prompts_per_step=2,
        generations=8,
        rollout_batches=rollout_batches,
        next_prompt_batches=next_prompt_batches,
        training_prompts=("task", "act", "task act", "act task"),
    )

    assert trainer.train().global_step == 2
    assert len(rollout_batches) == 2
    assert next_prompt_batches[0] == rollout_batches[1]
    for generation_batch in rollout_batches:
        assert len(generation_batch) == 16
        assert all(
            len(set(generation_batch[index : index + 8])) == 1
            for index in range(0, len(generation_batch), 8)
        )


def test_resume_discards_transient_lookahead_and_trains_only_remaining_group(
    tmp_path: Path,
) -> None:
    first_rollout_batches: list[list[str]] = []
    first_next_prompt_batches: list[list[str]] = []

    class InterruptAfterFirstSave(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            if state.global_step == 1:
                raise InterruptedError("synthetic interruption after checkpoint")

    interrupted = _trainer(
        tmp_path,
        max_steps=2,
        training_samples=4,
        prompts_per_step=2,
        generations=8,
        rollout_batches=first_rollout_batches,
        next_prompt_batches=first_next_prompt_batches,
        training_prompts=("task", "act", "task act", "act task"),
    )
    interrupted.add_callback(InterruptAfterFirstSave())
    with pytest.raises(InterruptedError, match="after checkpoint"):
        interrupted.train()

    resumed_rollout_batches: list[list[str]] = []
    resumed_next_prompt_batches: list[list[str]] = []
    resumed = _trainer(
        tmp_path,
        max_steps=2,
        training_samples=4,
        prompts_per_step=2,
        generations=8,
        rollout_batches=resumed_rollout_batches,
        next_prompt_batches=resumed_next_prompt_batches,
        training_prompts=("task", "act", "task act", "act task"),
    )
    assert resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-1")).global_step == 2

    assert len(first_rollout_batches) == 1
    assert len(first_next_prompt_batches) == 1
    assert first_next_prompt_batches[0]
    assert resumed_rollout_batches == [first_next_prompt_batches[0]]
    assert len(resumed_next_prompt_batches) == 1
    for step in (1, 2):
        records = json.loads((tmp_path / f"checkpoint-{step}" / "rollout_records.json").read_text())
        assert len(records) == 16


def test_one_generation_window_looks_ahead_across_epochs_and_resumes_exactly(
    tmp_path: Path,
) -> None:
    training_prompts = ("task", "act")
    full_rollout_batches: list[list[str]] = []
    full_next_prompt_batches: list[list[str]] = []
    full_run = _trainer(
        tmp_path / "full",
        max_steps=2,
        training_samples=2,
        prompts_per_step=2,
        generations=8,
        rollout_batches=full_rollout_batches,
        next_prompt_batches=full_next_prompt_batches,
        training_prompts=training_prompts,
    )

    assert full_run.train().global_step == 2
    assert len(full_rollout_batches) == 2
    assert full_next_prompt_batches[0] == full_rollout_batches[1]

    interrupted_rollout_batches: list[list[str]] = []
    interrupted_next_prompt_batches: list[list[str]] = []

    class InterruptAfterFirstSave(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            if state.global_step == 1:
                raise InterruptedError("synthetic interruption after checkpoint")

    interrupted = _trainer(
        tmp_path / "interrupted",
        max_steps=2,
        training_samples=2,
        prompts_per_step=2,
        generations=8,
        rollout_batches=interrupted_rollout_batches,
        next_prompt_batches=interrupted_next_prompt_batches,
        training_prompts=training_prompts,
    )
    interrupted.add_callback(InterruptAfterFirstSave())
    with pytest.raises(InterruptedError, match="after checkpoint"):
        interrupted.train()

    resumed_rollout_batches: list[list[str]] = []
    resumed_next_prompt_batches: list[list[str]] = []
    resumed = _trainer(
        tmp_path / "interrupted",
        max_steps=2,
        training_samples=2,
        prompts_per_step=2,
        generations=8,
        rollout_batches=resumed_rollout_batches,
        next_prompt_batches=resumed_next_prompt_batches,
        training_prompts=training_prompts,
    )
    assert (
        resumed.train(
            resume_from_checkpoint=str(tmp_path / "interrupted" / "checkpoint-1")
        ).global_step
        == 2
    )

    assert interrupted_next_prompt_batches[0] == full_rollout_batches[1]
    assert resumed_rollout_batches == [full_rollout_batches[1]]
    for generation_batch in full_rollout_batches + resumed_rollout_batches:
        assert len(generation_batch) == 16
        assert all(
            len(set(generation_batch[index : index + 8])) == 1
            for index in range(0, len(generation_batch), 8)
        )
    for step in (1, 2):
        records = json.loads(
            (tmp_path / "interrupted" / f"checkpoint-{step}" / "rollout_records.json").read_text()
        )
        assert len(records) == 16


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


def test_step_sampler_metrics_weight_tokens_across_microbatches(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.model is not None
    trainer.model.train()
    prompt_ids = torch.tensor([[2]])
    completion_ids = torch.tensor([[3, 4, 1]])
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    for active_tool_mask, stale_mask, selected_gaps in (
        ([1, 0, 0], [False, False, False], [2.0, 0.0, 0.0]),
        ([1, 1, 0], [False, True, False], [0.1, 0.3, 0.0]),
    ):
        with torch.no_grad():
            trainer_logprobs, _, _ = trainer._get_per_token_logps_and_entropies(
                trainer.model, input_ids, attention_mask, completion_ids.shape[1]
            )
        trainer._active_loss_inputs = {
            "completion_mask": torch.ones((1, 3)),
            "tool_mask": torch.tensor([active_tool_mask]),
            "sampler_stale_mask": torch.tensor([stale_mask]),
            "sampling_per_token_logps": trainer_logprobs - torch.tensor([selected_gaps]),
            "advantages": torch.ones((1,)),
        }
        with torch.no_grad():
            trainer._get_per_token_logps_and_entropies(
                trainer.model, input_ids, attention_mask, completion_ids.shape[1]
            )
    trainer._finish_training_step_metrics()

    metrics = trainer._metrics["train"]
    assert metrics["organism/sampler_ratio_capped_fraction_fresh"][-1] == pytest.approx(0.5)
    assert metrics["organism/sampler_ratio_capped_fraction_stale"][-1] == 0.0
    assert metrics["organism/trainer_sampler_logprob_gap_abs_fresh"][-1] == pytest.approx(1.05)
    assert metrics["organism/trainer_sampler_logprob_gap_abs_stale"][-1] == pytest.approx(0.3)


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


def test_optimizer_restore_guard_rejects_low_precision_moments(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.model is not None
    optimizer = trainer.create_optimizer()
    trainer.create_scheduler(num_training_steps=1, optimizer=optimizer)
    parameter = next(
        parameter for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    optimizer.state[parameter]["exp_avg"] = torch.zeros_like(parameter, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="must be float32"):
        trainer._save_checkpoint(trainer.model, None)
    assert not (tmp_path / "checkpoint-0").exists()


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


class StorageTransferPhase:
    """Exercise real policy transfers inside the trainer's phase callbacks."""

    def __init__(self) -> None:
        self.transfers: PolicyTransfers | None = None
        self.optimizer_steps = 0

    def initialize(self, trainer: Any, step: int = 0) -> None:
        del step
        self.transfers = PolicyTransfers(trainer.model, torch.device("cpu"))
        self.transfers.offload(trainer.optimizer)

    def before_sampling(self, trainer: Any) -> None:
        assert self.transfers is not None
        self.transfers.offload(trainer.optimizer)

    def before_training(self, trainer: Any) -> None:
        assert self.transfers is not None
        self.transfers.restore(trainer.optimizer)

    def after_optimizer_step(self, trainer: Any) -> None:
        assert self.transfers is not None
        self.transfers.offload(trainer.optimizer)
        self.optimizer_steps += 1


def test_direct_trainer_initializes_head_before_phase_storage_snapshot(tmp_path: Path) -> None:
    phase = StorageTransferPhase()
    trainer = _trainer(tmp_path, max_steps=2, phase=phase)
    trainer.train()
    assert phase.optimizer_steps == 2
    for step in (1, 2):
        checkpoint = tmp_path / f"checkpoint-{step}"
        validate_checkpoint(checkpoint)
        state = torch.load(checkpoint / "optimizer.pt", weights_only=True)
        assert state["state"]
        assert all(
            tensor.device.type == "cpu"
            for values in state["state"].values()
            for tensor in values.values()
            if isinstance(tensor, torch.Tensor)
        )
    assert phase.transfers is not None
    assert phase.transfers.frozen_host_storage_ids
