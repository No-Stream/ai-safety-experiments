"""Offline distillation loss and real PEFT checkpoint recovery tests."""

import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    Qwen2Config,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from reward_hacking.distill.train import (
    DistillTrainer,
    completion_logps,
    distillation_loss,
    load_training_rows,
    validate_lora_config,
)


def tiny_model() -> Any:
    torch.manual_seed(13)
    model = AutoModelForCausalLM.from_config(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=False,
        )
    )
    return get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
    )


def test_equal_teacher_has_zero_loss_and_gradient() -> None:
    model = tiny_model().eval()
    logps = completion_logps(model, torch.tensor([[1, 2, 3, 4, 5]]), 2, chunk_tokens=2)
    sampling = logps.detach().clone().requires_grad_()
    teacher = sampling.detach().clone().requires_grad_()
    loss, metrics = distillation_loss(logps, sampling, teacher, weight_mode="opd")
    loss.backward()
    assert loss.item() == 0
    assert metrics["mean_advantage"] == 0
    assert all(
        parameter.grad is None or parameter.grad.abs().max() < 1e-8
        for parameter in model.parameters()
    )
    assert sampling.grad is None
    assert teacher.grad is None


@pytest.mark.parametrize(
    ("advantage", "ratio", "expected", "gradient"),
    [
        (2.0, 1.0, -2.0, -2.0),
        (2.0, 1.5, -2.4, 0.0),
        (-2.0, 0.5, 1.6, 0.0),
        (-2.0, 1.5, 3.0, 3.0),
        (20.0, 1.0, -5.0, -5.0),
        (-20.0, 1.0, 5.0, 5.0),
    ],
)
def test_opd_sign_ratio_and_advantage_clipping(
    advantage: float, ratio: float, expected: float, gradient: float
) -> None:
    sampling = torch.tensor([[-4.0]])
    current = (sampling + math.log(ratio)).requires_grad_()
    loss, metrics = distillation_loss(current, sampling, sampling + advantage, weight_mode="opd")
    loss.backward()
    assert loss.item() == pytest.approx(expected)
    assert current.grad is not None
    assert current.grad.item() == pytest.approx(gradient)
    assert metrics["clip_fraction"] == float(ratio < 0.8 or ratio > 1.2)


def test_sft_matches_full_forward_completion_nll() -> None:
    model = tiny_model().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    chunked = completion_logps(model, input_ids, 2, chunk_tokens=1)
    expected = (
        model(input_ids)
        .logits[:, 1:-1]
        .float()
        .log_softmax(-1)
        .gather(-1, input_ids[:, 2:, None])
        .squeeze(-1)
    )
    torch.testing.assert_close(chunked, expected)
    loss, _ = distillation_loss(chunked, None, None, weight_mode="sft")
    torch.testing.assert_close(loss, -expected.mean())
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().max() > 0
        for parameter in model.parameters()
    )


class KillAfterSave(TrainerCallback):
    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        del args, control, kwargs
        if state.global_step == 1:
            os.kill(os.getpid(), signal.SIGKILL)


def make_trainer(output: Path, *, interrupt: bool = False) -> DistillTrainer:
    rows = [
        {"prompt_token_ids": [1, 2], "completion_token_ids": [3 + offset, 4 + offset, 5 + offset]}
        for offset in range(4)
    ]
    return DistillTrainer(
        model=tiny_model(),
        train_dataset=rows,
        weight_mode="sft",
        chunk_tokens=2,
        args=TrainingArguments(
            output_dir=str(output),
            use_cpu=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=2,
            learning_rate=0.01,
            save_steps=1,
            logging_steps=1,
            report_to="none",
            disable_tqdm=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
        ),
        callbacks=[KillAfterSave()] if interrupt else [],
    )


def test_kill_resume_restores_optimizer_and_counts(tmp_path: Path) -> None:
    reference = make_trainer(tmp_path / "reference")
    reference.train()
    worker_code = (
        "import runpy, sys; helpers = runpy.run_path(sys.argv[1]); "
        "helpers['make_trainer'](helpers['Path'](sys.argv[2]), interrupt=True).train()"
    )
    worker = subprocess.run(  # noqa: S603 - trusted Python and this synthetic test module
        [
            sys.executable,
            "-c",
            worker_code,
            str(Path(__file__).resolve()),
            str(tmp_path / "resumed"),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert worker.returncode == -signal.SIGKILL, worker.stdout + worker.stderr
    checkpoint = tmp_path / "resumed" / "checkpoint-1"
    assert (checkpoint / "optimizer.pt").is_file()
    resumed = make_trainer(tmp_path / "resumed")
    resumed.train(resume_from_checkpoint="latest")
    assert resumed.state.global_step == 2
    assert (
        resumed.total_trained_completion_tokens == reference.total_trained_completion_tokens == 12
    )
    assert resumed.model is not None
    assert reference.model is not None
    for name, parameter in resumed.model.named_parameters():
        torch.testing.assert_close(
            parameter, dict(reference.model.named_parameters())[name], atol=1e-7, rtol=1e-6
        )
    assert resumed.optimizer is not None
    assert reference.optimizer is not None
    for actual, expected in zip(
        cast("Any", resumed.optimizer).state.values(),
        cast("Any", reference.optimizer).state.values(),
        strict=True,
    ):
        assert actual["step"].item() == expected["step"].item() == 2
        torch.testing.assert_close(actual["exp_avg"], expected["exp_avg"])
    config = json.loads((tmp_path / "resumed" / "run_config.json").read_text())
    assert config["total_trained_completion_tokens"] == 12
    logs = [row for row in resumed.state.log_history if "loss" in row]
    assert len(logs) == 2
    assert all("completion_length" in row and "mean_advantage" in row for row in logs)


def test_join_refuses_missing_duplicate_or_changed_teacher(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    row = {
        "key": "synthetic/0/0",
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3],
        "student_sampling_logps": [-2.0],
        "token_ids_sha256": "abc",
    }
    score = {"key": row["key"], "teacher_logps": [-1.0], "token_ids_sha256": "abc"}
    samples.write_text(json.dumps(row) + "\n")
    teacher.write_text(json.dumps(score) + "\n")
    assert len(load_training_rows(samples, teacher, weight_mode="opd")) == 1
    teacher.write_text("")
    with pytest.raises(ValueError, match="keys"):
        load_training_rows(samples, teacher, weight_mode="opd")
    teacher.write_text((json.dumps(score) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_training_rows(samples, teacher, weight_mode="opd")
    score["token_ids_sha256"] = "changed"
    teacher.write_text(json.dumps(score) + "\n")
    with pytest.raises(ValueError, match="hash"):
        load_training_rows(samples, teacher, weight_mode="opd")


def test_nonfinite_logps_refused() -> None:
    with pytest.raises(ValueError, match="finite"):
        distillation_loss(torch.tensor([[float("nan")]]), None, None, weight_mode="sft")


def test_complete_round_rerun_is_a_noop_mid_epoch(tmp_path: Path) -> None:
    first = make_trainer(tmp_path)
    first.args.max_steps = 1
    first._identity["max_steps"] = 1
    first.train()
    resumed = make_trainer(tmp_path)
    resumed.args.max_steps = 1
    resumed._identity["max_steps"] = 1
    resumed.train(resume_from_checkpoint="latest")
    assert resumed.state.global_step == 1
    assert resumed.total_trained_completion_tokens == 6
    assert not (tmp_path / "checkpoint-2").exists()


def test_torn_newest_checkpoint_is_set_aside(tmp_path: Path) -> None:
    first = make_trainer(tmp_path)
    first.train()
    # Sabotage the newest checkpoint: an adapter plus step counter must not restart Adam.
    (tmp_path / "checkpoint-2" / "optimizer.pt").unlink()
    resumed = make_trainer(tmp_path)
    resumed.train(resume_from_checkpoint="latest")
    assert resumed.state.global_step == 2
    assert resumed.total_trained_completion_tokens == 12
    assert list((tmp_path / "incomplete-checkpoints").glob("*/checkpoint-2"))
    for actual, expected in zip(
        cast("Any", resumed.optimizer).state.values(),
        cast("Any", first.optimizer).state.values(),
        strict=True,
    ):
        torch.testing.assert_close(actual["exp_avg"], expected["exp_avg"])


def test_resume_refuses_changed_base_or_dataset(tmp_path: Path) -> None:
    first = make_trainer(tmp_path)
    first.train()
    resumed = make_trainer(tmp_path)
    resumed._identity["base_model_revision"] = "different-base"
    with pytest.raises(ValueError, match="configuration changed"):
        resumed.train(resume_from_checkpoint="latest")
    resumed = make_trainer(tmp_path)
    resumed._identity["dataset_sha256"] = "different-data"
    with pytest.raises(ValueError, match="configuration changed"):
        resumed.train(resume_from_checkpoint="latest")


def test_chunked_head_recomputes_bounded_slices_during_backward() -> None:
    model = tiny_model().eval()
    head_positions: list[int] = []

    def record_head(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor
    ) -> None:
        head_positions.append(inputs[0].size(1))

    handle = model.get_output_embeddings().register_forward_hook(record_head)
    try:
        logps = completion_logps(model, torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), 2, chunk_tokens=2)
        (-logps.mean()).backward()
    finally:
        handle.remove()
    assert max(head_positions) <= 2
    assert len(head_positions) == 6  # Three forward slices, each recomputed in backward.


def test_opd_step_logs_correlation_and_completion_counts(tmp_path: Path) -> None:
    model = tiny_model().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        sampling = completion_logps(model, ids, 2, chunk_tokens=2).flatten().tolist()
    row = {
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3, 4, 5],
        "student_sampling_logps": sampling,
        "teacher_logps": [value + 0.5 for value in sampling],
    }
    trainer = DistillTrainer(
        model=model,
        train_dataset=[row, row],
        weight_mode="opd",
        chunk_tokens=2,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=1,
            learning_rate=0.01,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            save_steps=1,
            logging_steps=1,
            report_to="none",
            disable_tqdm=True,
        ),
    )
    trainer.train()
    step = next(record for record in trainer.state.log_history if "loss" in record)
    assert step["student_logprob_correlation"] == pytest.approx(1.0)
    assert step["mean_advantage"] == pytest.approx(0.5)
    assert step["clip_fraction"] == 0.0
    assert step["completion_length"] == 3
    assert step["total_trained_completion_tokens"] == 6


@pytest.mark.parametrize(
    ("rank", "alpha", "targets"),
    [(8, 32, ["q_proj", "v_proj"]), (16, 16, ["q_proj", "v_proj"]), (16, 32, ["q_proj"])],
)
def test_initial_adapter_configuration_guard(rank: int, alpha: int, targets: list[str]) -> None:
    with pytest.raises(ValueError, match="initial adapter"):
        validate_lora_config(
            LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets), ["q_proj", "v_proj"]
        )
    validate_lora_config(
        LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]), ["q_proj", "v_proj"]
    )


def test_first_step_refuses_sampler_trainer_mismatch(tmp_path: Path) -> None:
    model = tiny_model().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        trainer_logps = completion_logps(model, ids, 2, chunk_tokens=2).flatten().tolist()
    sampler_logps = [value - 0.5 for value in trainer_logps]
    row = {
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3, 4, 5],
        "student_sampling_logps": sampler_logps,
        "teacher_logps": trainer_logps,
    }
    trainer = DistillTrainer(
        model=model,
        train_dataset=[row, row],
        weight_mode="opd",
        chunk_tokens=2,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=1,
            learning_rate=0.01,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            save_steps=1,
            logging_steps=1,
            report_to="none",
            disable_tqdm=True,
        ),
    )
    with pytest.raises(RuntimeError, match="sampler and trainer disagree"):
        trainer.train()
