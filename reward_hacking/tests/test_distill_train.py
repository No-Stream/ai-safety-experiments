"""Offline distillation loss and real PEFT checkpoint recovery tests."""

import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import replace
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

import reward_hacking.distill.memory as distill_memory
import reward_hacking.distill.train as distill_train
from reward_hacking.distill.memory import MemoryOptions
from reward_hacking.distill.sample import token_ids_sha256
from reward_hacking.distill.train import (
    DistillTrainer,
    StepTelemetry,
    collate_completion,
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


def test_opd_optimizer_moves_toward_teacher_on_active_tokens_only() -> None:
    model = tiny_model().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        sampling_logps = completion_logps(model, input_ids, 2, chunk_tokens=2).detach()
    active_loss_mask = torch.tensor([[True, False, False]])
    teacher_logps = sampling_logps + torch.tensor([[1.0, 0.0, 0.0]])
    teacher_with_inactive_changes = sampling_logps + torch.tensor([[1.0, 100.0, -100.0]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    current_logps = completion_logps(model, input_ids, 2, chunk_tokens=2)
    loss, _ = distillation_loss(
        current_logps,
        sampling_logps,
        teacher_logps,
        weight_mode="opd",
        loss_mask=active_loss_mask,
    )
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    optimizer.zero_grad(set_to_none=True)
    current_logps = completion_logps(model, input_ids, 2, chunk_tokens=2)
    masked_loss, _ = distillation_loss(
        current_logps,
        sampling_logps,
        teacher_with_inactive_changes,
        weight_mode="opd",
        loss_mask=active_loss_mask,
    )
    masked_loss.backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, gradients[name])
        else:
            assert parameter.grad is None

    optimizer.step()
    with torch.no_grad():
        updated_logps = completion_logps(model, input_ids, 2, chunk_tokens=2)
    assert updated_logps[0, 0] > sampling_logps[0, 0]


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


def test_masked_sft_loss_is_nll_over_unmasked_completion_tokens() -> None:
    logps = torch.tensor([[-0.5, -4.0, -1.5, -7.0]])
    loss_mask = torch.tensor([[1, 0, 1, 0]])

    loss, _ = distillation_loss(
        logps,
        None,
        None,
        weight_mode="sft",
        loss_mask=loss_mask,
    )

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_completion_collator_preserves_loss_mask() -> None:
    batch = collate_completion(
        [{"prompt_token_ids": [1, 2], "completion_token_ids": [3, 4], "loss_mask": [1, 0]}]
    )

    assert batch["loss_mask"].tolist() == [[True, False]]


class KillAfterSave(TrainerCallback):
    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        del args, control, kwargs
        if state.global_step == 1:
            os.kill(os.getpid(), signal.SIGKILL)


def make_trainer(  # noqa: PLR0913 - expose separate training configuration dimensions in tests
    output: Path,
    *,
    interrupt: bool = False,
    memory_options: MemoryOptions | None = None,
    max_steps: int = 2,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 0.01,
    rows: list[dict[str, Any]] | None = None,
    sft_loss_averaging: distill_train.SFTLossAveraging = "token",
) -> DistillTrainer:
    training_rows = rows or [
        {"prompt_token_ids": [1, 2], "completion_token_ids": [3 + offset, 4 + offset, 5 + offset]}
        for offset in range(4)
    ]
    trainer_kwargs: dict[str, Any] = {}
    if memory_options is not None:
        trainer_kwargs["memory_options"] = memory_options
    return DistillTrainer(
        model=tiny_model(),
        train_dataset=training_rows,
        weight_mode="sft",
        sft_loss_averaging=sft_loss_averaging,
        chunk_tokens=2,
        args=TrainingArguments(
            output_dir=str(output),
            use_cpu=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            save_steps=1,
            logging_steps=1,
            report_to="none",
            disable_tqdm=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
        ),
        callbacks=[KillAfterSave()] if interrupt else [],
        **trainer_kwargs,
    )


def test_long_context_flags_parse_and_build_memory_options() -> None:
    defaults = distill_train.parse_args(
        ["--run-dir", "artifacts/synthetic-distill", "--round", "0", "--weight-mode", "sft"]
    )
    assert distill_train.memory_options_from_args(defaults) == MemoryOptions()

    configured = distill_train.parse_args(
        [
            "--run-dir",
            "artifacts/synthetic-distill",
            "--round",
            "0",
            "--weight-mode",
            "sft",
            "--deltanet-conv-kernel",
            "fla",
            "--liger",
            "--tiled-mlp",
            "--offload-checkpoints",
            "--base-quantization",
            "torchao-int8",
        ]
    )
    assert distill_train.memory_options_from_args(configured) == MemoryOptions(
        deltanet_conv_kernel="fla",
        liger=True,
        tiled_mlp=True,
        offload_checkpoints=True,
        base_quantization="torchao-int8",
    )
    nf4 = distill_train.parse_args(
        [
            "--run-dir",
            "artifacts/synthetic-distill",
            "--round",
            "0",
            "--weight-mode",
            "sft",
            "--base-quantization",
            "nf4",
        ]
    )
    assert distill_train.memory_options_from_args(nf4).base_quantization == "nf4"


def test_training_cli_resolves_epochs_and_warmup_steps() -> None:
    defaults = distill_train.parse_args(
        ["--run-dir", "artifacts/synthetic-distill", "--round", "0", "--weight-mode", "sft"]
    )
    assert defaults.max_steps is None
    assert defaults.epochs is None
    assert defaults.warmup_steps == 0
    assert (
        distill_train.effective_max_steps(
            row_count=4,
            gradient_accumulation_steps=defaults.gradient_accumulation_steps,
            max_steps=defaults.max_steps,
            epochs=defaults.epochs,
        )
        == 2
    )

    epochs = distill_train.parse_args(
        [
            "--run-dir",
            "artifacts/synthetic-distill",
            "--round",
            "0",
            "--weight-mode",
            "sft",
            "--gradient-accumulation-steps",
            "3",
            "--epochs",
            "2",
            "--warmup-steps",
            "1",
        ]
    )
    assert (
        distill_train.effective_max_steps(
            row_count=7,
            gradient_accumulation_steps=epochs.gradient_accumulation_steps,
            max_steps=epochs.max_steps,
            epochs=epochs.epochs,
        )
        == 6
    )
    assert epochs.warmup_steps == 1

    with pytest.raises(SystemExit):
        distill_train.parse_args(
            [
                "--run-dir",
                "artifacts/synthetic-distill",
                "--round",
                "0",
                "--weight-mode",
                "sft",
                "--epochs",
                "2",
                "--max-steps",
                "3",
            ]
        )


def test_resume_allows_increased_step_target_and_refuses_training_drift(tmp_path: Path) -> None:
    first = make_trainer(tmp_path, max_steps=1)
    first.train()
    assert first.model is not None
    first_adapter_weights = [
        parameter.detach().clone()
        for parameter in first.model.parameters()
        if parameter.requires_grad
    ]

    extended = make_trainer(tmp_path, max_steps=2)
    result = extended.train(resume_from_checkpoint="latest")
    assert result.global_step == 2
    assert not result.metrics.get("already_complete", False)
    assert extended.model is not None
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(
            first_adapter_weights,
            [parameter for parameter in extended.model.parameters() if parameter.requires_grad],
            strict=True,
        )
    )

    lowered_target = make_trainer(tmp_path, max_steps=1)
    with pytest.raises(ValueError, match="configuration changed"):
        lowered_target.train(resume_from_checkpoint="latest")

    changed_accumulation = make_trainer(tmp_path, max_steps=3, gradient_accumulation_steps=3)
    with pytest.raises(ValueError, match="configuration changed"):
        changed_accumulation.train(resume_from_checkpoint="latest")

    changed_learning_rate = make_trainer(tmp_path, max_steps=3, learning_rate=0.02)
    with pytest.raises(ValueError, match="configuration changed"):
        changed_learning_rate.train(resume_from_checkpoint="latest")

    changed_data = make_trainer(
        tmp_path,
        max_steps=3,
        rows=[{"prompt_token_ids": [1, 2], "completion_token_ids": [3, 4, 6]}] * 4,
    )
    with pytest.raises(ValueError, match="configuration changed"):
        changed_data.train(resume_from_checkpoint="latest")


def test_resume_refuses_changed_long_context_options(tmp_path: Path) -> None:
    first = make_trainer(tmp_path)
    first.args.max_steps = 1
    first._identity["max_steps"] = 1
    first.train()

    changed_options = (
        replace(MemoryOptions(), deltanet_conv_kernel="fla"),
        replace(MemoryOptions(), liger=True),
        replace(MemoryOptions(), tiled_mlp=True),
        replace(MemoryOptions(), offload_checkpoints=True),
        replace(MemoryOptions(), base_quantization="torchao-int8"),
    )
    for memory_options in changed_options:
        resumed = make_trainer(tmp_path, memory_options=memory_options)
        resumed.args.max_steps = 1
        resumed._identity["max_steps"] = 1
        with pytest.raises(ValueError, match="configuration changed"):
            resumed.train(resume_from_checkpoint="latest")


def test_legacy_resume_state_defaults_long_context_options(tmp_path: Path) -> None:
    first = make_trainer(tmp_path)
    first.args.max_steps = 1
    first._identity["max_steps"] = 1
    first.train()
    checkpoint_state = tmp_path / "checkpoint-1" / "distill_state.json"
    saved_state = json.loads(checkpoint_state.read_text())
    saved_state.pop("memory_options")
    saved_state.pop("warmup_steps")
    checkpoint_state.write_text(json.dumps(saved_state) + "\n")

    resumed = make_trainer(tmp_path)
    resumed.args.max_steps = 1
    resumed._identity["max_steps"] = 1
    resumed.train(resume_from_checkpoint="latest")
    assert resumed.state.global_step == 1

    opted_in = make_trainer(
        tmp_path,
        memory_options=replace(MemoryOptions(), base_quantization="torchao-int8"),
    )
    opted_in.args.max_steps = 1
    opted_in._identity["max_steps"] = 1
    with pytest.raises(ValueError, match="configuration changed"):
        opted_in.train(resume_from_checkpoint="latest")


def test_quantized_trainer_initializes_trainable_adapters_in_bfloat16(tmp_path: Path) -> None:
    trainer = make_trainer(
        tmp_path,
        memory_options=replace(MemoryOptions(), base_quantization="torchao-int8"),
    )
    assert trainer.model is not None
    trainable_parameters = [
        parameter for parameter in trainer.model.parameters() if parameter.requires_grad
    ]
    assert trainable_parameters
    assert {parameter.dtype for parameter in trainable_parameters} == {torch.bfloat16}


def test_quantized_resume_preserves_trainable_bfloat16_adapters(tmp_path: Path) -> None:
    memory_options = replace(MemoryOptions(), base_quantization="torchao-int8")
    trainer = make_trainer(tmp_path, memory_options=memory_options)
    trainer.args.max_steps = 1
    trainer._identity["max_steps"] = 1
    assert trainer.model is not None
    trainable_parameters = [
        parameter for parameter in trainer.model.parameters() if parameter.requires_grad
    ]
    assert trainable_parameters
    for parameter in trainable_parameters:
        parameter.data = parameter.data.to(torch.bfloat16)
    trainer.train()

    checkpoint = tmp_path / "checkpoint-1"
    resumed = make_trainer(tmp_path, memory_options=memory_options)
    resumed.args.max_steps = 1
    resumed._identity["max_steps"] = 1
    assert resumed.model is not None
    resumed_trainable = [
        parameter for parameter in resumed.model.parameters() if parameter.requires_grad
    ]
    assert resumed_trainable
    for parameter in resumed_trainable:
        parameter.data = parameter.data.to(torch.bfloat16)

    resumed._load_from_checkpoint(str(checkpoint))
    assert resumed.model is not None
    loaded_trainable = [
        parameter for parameter in resumed.model.parameters() if parameter.requires_grad
    ]
    assert loaded_trainable
    assert {parameter.dtype for parameter in loaded_trainable} == {torch.bfloat16}


def test_offloaded_checkpoint_matches_chunked_accumulated_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registered_slots: list[Any] = []

    class CpuHostSlot:
        def __init__(self, elements: int, dtype: torch.dtype) -> None:
            self.buffer = torch.empty(elements, dtype=dtype)
            self.released = False
            registered_slots.append(self)

        @property
        def registered_bytes(self) -> int:
            return self.buffer.numel() * self.buffer.element_size()

        def view_for(self, hidden_states: torch.Tensor) -> torch.Tensor:
            if self.released or hidden_states.numel() > self.buffer.numel():
                raise RuntimeError("CPU host slot is unavailable or too small")
            return self.buffer[: hidden_states.numel()].view(hidden_states.shape)

        def release(self) -> None:
            self.released = True

    baseline = make_trainer(tmp_path / "baseline")
    cast("Any", baseline.model).enable_input_require_grads()
    baseline.train()
    assert baseline.model is not None
    expected_adapters = {
        name: parameter.detach().clone()
        for name, parameter in baseline.model.named_parameters()
        if parameter.requires_grad
    }

    monkeypatch.setattr(distill_memory, "HostSlot", CpuHostSlot)
    prepare_checkpoint_offload = distill_train.prepare_checkpoint_offload

    def prepare_cpu_checkpoint_offload(model: torch.nn.Module, max_sequence_tokens: int) -> Any:
        with monkeypatch.context() as installer_patch:
            installer_patch.setattr(torch.cuda, "is_available", lambda: True)
            return prepare_checkpoint_offload(model, max_sequence_tokens)

    monkeypatch.setattr(distill_train, "prepare_checkpoint_offload", prepare_cpu_checkpoint_offload)
    offloaded = make_trainer(
        tmp_path / "offloaded",
        memory_options=replace(MemoryOptions(), offload_checkpoints=True),
    )
    cast("Any", offloaded.model).enable_input_require_grads()
    offloaded.train()
    assert offloaded.model is not None

    actual_adapters = {
        name: parameter.detach()
        for name, parameter in offloaded.model.named_parameters()
        if parameter.requires_grad
    }
    assert actual_adapters.keys() == expected_adapters.keys()
    for name, expected in expected_adapters.items():
        torch.testing.assert_close(actual_adapters[name], expected, atol=1e-6, rtol=1e-6)
    assert registered_slots
    assert all(slot.released for slot in registered_slots)
    assert offloaded.checkpoint_offload is not None
    assert offloaded.checkpoint_offload.registered_bytes == sum(
        slot.registered_bytes for slot in registered_slots
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
    token_hash = token_ids_sha256([1, 2], [3])
    row = {
        "key": "synthetic/0/0",
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3],
        "student_sampling_logps": [-2.0],
        "token_ids_sha256": token_hash,
    }
    score = {"key": row["key"], "teacher_logps": [-1.0], "token_ids_sha256": token_hash}
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


def test_join_refuses_sample_tokens_changed_without_hash_update(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    token_hash = token_ids_sha256([1, 2], [3])
    row = {
        "key": "synthetic/0/0",
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [4],
        "student_sampling_logps": [-2.0],
        "token_ids_sha256": token_hash,
    }
    score = {"key": row["key"], "teacher_logps": [-1.0], "token_ids_sha256": token_hash}
    samples.write_text(json.dumps(row) + "\n")
    teacher.write_text(json.dumps(score) + "\n")

    with pytest.raises(ValueError, match="token ids"):
        load_training_rows(samples, teacher, weight_mode="opd")


def test_sft_join_accepts_legacy_rows_without_token_hash(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps(
            {
                "key": "synthetic/0/0",
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3],
            }
        )
        + "\n"
    )

    assert load_training_rows(samples, None, weight_mode="sft") == [
        {"prompt_token_ids": [1, 2], "completion_token_ids": [3]}
    ]


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
    assert step["sequence_tokens"] == 10
    assert step["tokens_per_second"] > 0
    assert step["peak_allocated_bytes"] == 0
    assert step["peak_reserved_bytes"] == 0


def test_gpu_telemetry_reads_nvml_backed_torch_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(distill_train.torch.cuda, "mem_get_info", lambda _device: (70, 100))
    monkeypatch.setattr(distill_train.torch.cuda, "utilization", lambda _device: 37)

    metrics = distill_train.read_gpu_metrics(torch.device("cuda:0"))

    assert metrics.utilization_percent == 37
    assert metrics.total_memory_used_bytes == 30
    assert metrics.total_memory_bytes == 100


def test_step_telemetry_warns_when_tokens_per_second_collapses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    times = iter((0.0, 1.0, 2.0, 4.0))
    monkeypatch.setattr(distill_train.time, "perf_counter", lambda: next(times))
    gpu_samples = iter(
        (
            distill_train.GPUMetrics(25.0, 20 * 1024**3, 32 * 1024**3),
            distill_train.GPUMetrics(1.0, 31 * 1024**3, 32 * 1024**3),
        )
    )
    monkeypatch.setattr(distill_train, "read_gpu_metrics", lambda _device: next(gpu_samples))
    telemetry = StepTelemetry()
    arguments = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none")
    state = TrainerState()
    control = TrainerControl()

    telemetry.on_step_begin(arguments, state, control)
    telemetry.sequence_tokens = 100
    telemetry.on_step_end(arguments, state, control)
    assert telemetry.metrics["tokens_per_second"] == 100

    telemetry.on_step_begin(arguments, state, control)
    telemetry.sequence_tokens = 40
    telemetry.on_step_end(arguments, state, control)
    assert telemetry.metrics["tokens_per_second"] == 20
    assert telemetry.metrics["gpu_utilization_percent"] == 1
    assert telemetry.metrics["total_gpu_memory_used_bytes"] == 31 * 1024**3
    assert telemetry.rates == [100, 20]
    assert "below half the running median" in caplog.text
    assert "possible GPU spill to shared memory" in caplog.text


def test_step_telemetry_does_not_suggest_spill_without_near_capacity_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    times = iter((0.0, 1.0, 2.0, 4.0))
    monkeypatch.setattr(distill_train.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(
        distill_train,
        "read_gpu_metrics",
        lambda _device: distill_train.GPUMetrics(1.0, 8 * 1024**3, 32 * 1024**3),
    )
    telemetry = StepTelemetry()
    arguments = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none")
    state = TrainerState()
    control = TrainerControl()

    for tokens in (100, 40):
        telemetry.on_step_begin(arguments, state, control)
        telemetry.sequence_tokens = tokens
        telemetry.on_step_end(arguments, state, control)

    assert "possible GPU spill to shared memory" not in caplog.text
    assert "does not indicate GPU spill" in caplog.text


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
