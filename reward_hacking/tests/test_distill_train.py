"""Offline distillation loss and real PEFT checkpoint recovery tests."""

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import replace
from functools import partial
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


def _write_training_scope_inputs(
    run_directory: Path, *, round_number: int = 0
) -> tuple[Path, Path, Path, str]:
    heldout_path = run_directory / "heldout.json"
    heldout_path.write_text(
        json.dumps({"heldout_channels": ["synthetic-channel"]}) + "\n",
        encoding="utf-8",
    )
    heldout_digest = hashlib.sha256(heldout_path.read_bytes()).hexdigest()
    selection_path = run_directory / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_only": False,
                "heldout_manifests": [{"path": str(heldout_path), "sha256": heldout_digest}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_directory.joinpath("dataset_manifest.json").write_text(
        '{"schema_only": false}\n', encoding="utf-8"
    )
    rows_path = run_directory / "samples" / f"round-{round_number}.jsonl"
    rows_path.parent.mkdir(exist_ok=True)
    rows_path.write_text('{"key":"synthetic"}\n', encoding="utf-8")
    return selection_path, heldout_path, rows_path, heldout_digest


def _gate2_card_value(
    selection_path: Path,
    rows_path: Path,
    heldout_digest: str,
) -> dict[str, object]:
    return {
        "schema_only": False,
        "gate_exit_code": 0,
        "dry_run": False,
        "exit_code": 0,
        "failure_codes": [],
        "provenance": {
            "manifest_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            "heldout_manifest_sha256": [heldout_digest],
            **distill_train.gate2_input_content(
                selection_path.parent, round_number=int(rows_path.stem.split("-")[-1])
            ),
        },
    }


@pytest.mark.parametrize(
    ("card", "expected_error"),
    [
        (None, "passing Gate 2"),
        ({"schema_only": False, "gate_exit_code": 1, "dry_run": False}, "passing Gate 2"),
        ({"schema_only": False, "gate_exit_code": 0, "dry_run": True}, "dry-run card"),
        ({"schema_only": True, "gate_exit_code": 0, "dry_run": False}, "schema-only"),
    ],
)
def test_real_training_requires_a_passing_gate2_card(
    tmp_path: Path,
    card: dict[str, object] | None,
    expected_error: str,
) -> None:
    tmp_path.joinpath("selection.json").write_text('{"schema_only": false}\n', encoding="utf-8")
    tmp_path.joinpath("dataset_manifest.json").write_text(
        '{"schema_only": false}\n', encoding="utf-8"
    )
    if card is not None:
        tmp_path.joinpath("card.json").write_text(json.dumps(card) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        distill_train._validate_training_dataset_scope(tmp_path, schema_only=False)


def test_real_training_accepts_a_passing_gate2_card(tmp_path: Path) -> None:
    selection_path, _heldout_path, rows_path, heldout_digest = _write_training_scope_inputs(
        tmp_path
    )
    tmp_path.joinpath("card.json").write_text(
        json.dumps(_gate2_card_value(selection_path, rows_path, heldout_digest)) + "\n",
        encoding="utf-8",
    )

    distill_train._validate_training_dataset_scope(tmp_path, schema_only=False)


def test_gate2_compares_parsed_content_and_ignores_serialization(tmp_path: Path) -> None:
    selection_path, heldout_path, rows_path, digest = _write_training_scope_inputs(tmp_path)
    card = _gate2_card_value(selection_path, rows_path, digest)
    cast("dict[str, object]", card["provenance"]).update(
        distill_train.gate2_input_content(tmp_path, round_number=0)
    )
    tmp_path.joinpath("card.json").write_text(json.dumps(card))
    rows_path.write_text(' { "key" : "synthetic" }\n')
    heldout_path.write_text(json.dumps(json.loads(heldout_path.read_text()), indent=4))
    selection_path.write_text(json.dumps(json.loads(selection_path.read_text()), indent=4))
    distill_train._validate_training_dataset_scope(tmp_path, schema_only=False)
    rows_path.write_text('{"key":"different"}\n')
    with pytest.raises(ValueError, match="card is stale"):
        distill_train._validate_training_dataset_scope(tmp_path, schema_only=False)


def test_resume_compares_training_content_not_dataset_digest(tmp_path: Path) -> None:
    first = make_trainer(tmp_path, max_steps=1)
    first.train()
    resumed = make_trainer(tmp_path, max_steps=1)
    resumed._identity["dataset_sha256"] = "different serialization"
    resumed.train(resume_from_checkpoint="latest")
    assert resumed.state.global_step == 1


def test_legacy_checkpoint_without_content_keeps_settings_guard(tmp_path: Path) -> None:
    first = make_trainer(tmp_path, max_steps=1)
    first.train()
    state_path = tmp_path / "checkpoint-1" / "distill_state.json"
    state = json.loads(state_path.read_text())
    del state["training_content"]
    state["dataset_sha256"] = "legacy provenance"
    state_path.write_text(json.dumps(state))
    resumed = make_trainer(tmp_path, max_steps=1)
    resumed.train(resume_from_checkpoint="latest")
    with pytest.raises(ValueError, match="configuration changed"):
        make_trainer(tmp_path, learning_rate=0.02).train(resume_from_checkpoint="latest")


@pytest.mark.parametrize(
    "field", ["prompt_token_ids", "completion_token_ids", "labels", "loss_mask", "hint_text"]
)
def test_gate2_rejects_changed_named_row_content(tmp_path: Path, field: str) -> None:
    selection_path, _heldout_path, rows_path, digest = _write_training_scope_inputs(tmp_path)
    row = {
        "key": "synthetic",
        "prompt_token_ids": [1],
        "completion_token_ids": [2],
        "labels": [-100, 2],
        "loss_mask": [1],
        "hint_text": "synthetic",
    }
    rows_path.write_text(json.dumps(row) + "\n")
    tmp_path.joinpath("card.json").write_text(
        json.dumps(_gate2_card_value(selection_path, rows_path, digest))
    )
    row[field] = "changed" if field == "hint_text" else [9]
    rows_path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="card is stale"):
        distill_train._validate_training_dataset_scope(tmp_path, schema_only=False)


def test_heldout_content_uses_only_task_and_channel_scope() -> None:
    first = {
        "selected_tasks": [{"task_id": "synthetic", "description": "old metadata"}],
        "split": {"train": ["unused"], "held-out": ["channel-two", "channel-one"]},
    }
    second = {
        "selected_tasks": [{"task_id": "synthetic", "description": "new metadata"}],
        "split": {"train": ["different unused"], "held-out": ["channel-one", "channel-two"]},
    }
    assert distill_train._heldout_content(first) == distill_train._heldout_content(second)
    second["selected_tasks"] = [{"task_id": "different"}]
    assert distill_train._heldout_content(first) != distill_train._heldout_content(second)


def test_gate2_legacy_card_keeps_available_content_and_ignores_raw_hashes(tmp_path: Path) -> None:
    selection_path, _heldout_path, rows_path, digest = _write_training_scope_inputs(tmp_path)
    card = _gate2_card_value(selection_path, rows_path, digest)
    provenance = cast("dict[str, object]", card["provenance"])
    for field in ("selection_content", "rows_content", "heldout_manifest_content"):
        del provenance[field]
    provenance["rows_sha256"] = "legacy byte provenance"
    tmp_path.joinpath("card.json").write_text(json.dumps(card))
    rows_path.write_text(' { "key" : "synthetic" }\n')
    distill_train._validate_training_dataset_scope(tmp_path, schema_only=False)


def test_opd_training_reads_samples_without_sft_admission_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["train", "--run-dir", str(tmp_path), "--round", "0", "--weight-mode", "opd"]
    )
    with pytest.raises(FileNotFoundError, match=r"round-0\.jsonl"):
        distill_train.main()


def test_real_training_ignores_a_card_heldout_provenance_digest(tmp_path: Path) -> None:
    selection_path, _heldout_path, rows_path, _heldout_digest = _write_training_scope_inputs(
        tmp_path
    )
    tmp_path.joinpath("card.json").write_text(
        json.dumps(_gate2_card_value(selection_path, rows_path, "0" * 64)) + "\n",
        encoding="utf-8",
    )

    distill_train._validate_training_dataset_scope(tmp_path, schema_only=False)


def test_real_training_rechecks_the_selected_heldout_manifest(tmp_path: Path) -> None:
    selection_path, heldout_path, rows_path, heldout_digest = _write_training_scope_inputs(tmp_path)
    tmp_path.joinpath("card.json").write_text(
        json.dumps(_gate2_card_value(selection_path, rows_path, heldout_digest)) + "\n",
        encoding="utf-8",
    )
    heldout_path.write_text('{"heldout_channels": ["changed-channel"]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="held-out manifest changed"):
        distill_train._validate_training_dataset_scope(tmp_path, schema_only=False)


def test_real_training_binds_gate2_card_to_requested_round(tmp_path: Path) -> None:
    selection_path, _heldout_path, rows_path, heldout_digest = _write_training_scope_inputs(
        tmp_path,
        round_number=2,
    )
    tmp_path.joinpath("card.json").write_text(
        json.dumps(_gate2_card_value(selection_path, rows_path, heldout_digest)) + "\n",
        encoding="utf-8",
    )

    distill_train._validate_training_dataset_scope(
        tmp_path,
        schema_only=False,
        round_number=2,
    )


def test_real_training_ignores_a_gate2_card_rows_provenance_digest(tmp_path: Path) -> None:
    selection_path, _heldout_path, rows_path, heldout_digest = _write_training_scope_inputs(
        tmp_path
    )
    card_value = _gate2_card_value(selection_path, rows_path, heldout_digest)
    provenance = cast("dict[str, object]", card_value["provenance"])
    provenance["rows_sha256"] = "0" * 64
    tmp_path.joinpath("card.json").write_text(json.dumps(card_value) + "\n", encoding="utf-8")

    distill_train._validate_training_dataset_scope(tmp_path, schema_only=False)


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
    warmup_steps: int = 0,
    diagnostic_admission: dict[str, object] | None = None,
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
        diagnostic_admission=diagnostic_admission,
        chunk_tokens=2,
        args=TrainingArguments(
            output_dir=str(output),
            use_cpu=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            lr_scheduler_type="constant_with_warmup",
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


def test_checkpoint_retention_flags_reach_real_trainer_without_changing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = distill_train.parse_args(
        ["--run-dir", str(tmp_path), "--round", "0", "--weight-mode", "sft"]
    )
    assert (defaults.checkpoint_keep_latest, defaults.checkpoint_keep_middle) == (2, 3)
    configured = distill_train.parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--round",
            "0",
            "--weight-mode",
            "sft",
            "--checkpoint-keep-latest",
            "4",
            "--checkpoint-keep-middle",
            "1",
        ]
    )
    args = _diagnostic_training_args(tmp_path, monkeypatch)
    args.checkpoint_keep_latest = configured.checkpoint_keep_latest
    args.checkpoint_keep_middle = configured.checkpoint_keep_middle
    trainer = distill_train.train_round(
        args, diagnostic_dataset_content=distill_train.gate2_input_content(tmp_path, round_number=0)
    )
    assert (trainer.checkpoint_keep_latest, trainer.checkpoint_keep_middle) == (4, 1)
    saved_state = json.loads(
        (Path(trainer.output_dir) / "checkpoint-1" / "distill_state.json").read_text()
    )
    assert "checkpoint_keep_latest" not in saved_state
    assert "checkpoint_keep_middle" not in saved_state
    args.checkpoint_keep_latest = 2
    args.checkpoint_keep_middle = 0
    resumed = distill_train.train_round(
        args, diagnostic_dataset_content=distill_train.gate2_input_content(tmp_path, round_number=0)
    )
    assert resumed.state.global_step == 1


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--checkpoint-keep-latest", "1"),
        ("--checkpoint-keep-middle", "-1"),
        ("--checkpoint-keep-middle", "4"),
    ],
)
def test_checkpoint_retention_cli_rejects_unsafe_counts(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        distill_train.parse_args(
            ["--run-dir", "synthetic", "--round", "0", "--weight-mode", "sft", flag, value]
        )


@pytest.mark.parametrize("distill_state", [None, "{", "[]", ""])
def test_checkpoint_retention_requires_distillation_resume_state(
    tmp_path: Path, distill_state: str | None
) -> None:
    trainer = make_trainer(tmp_path, max_steps=1)
    trainer.train()
    checkpoint = tmp_path / "checkpoint-1"
    assert trainer._checkpoint_retention_is_complete(checkpoint)
    state_path = checkpoint / "distill_state.json"
    if distill_state is None:
        state_path.unlink()
    else:
        state_path.write_text(distill_state)
    assert not trainer._checkpoint_retention_is_complete(checkpoint)


def test_distillation_torn_state_does_not_replace_complete_resume_checkpoints(
    tmp_path: Path,
) -> None:
    trainer = make_trainer(tmp_path, max_steps=4)
    trainer.train()
    trainer.checkpoint_keep_middle = 0
    torn = tmp_path / "checkpoint-5"
    shutil.copytree(tmp_path / "checkpoint-4", torn)
    state_path = torn / "trainer_state.json"
    state = json.loads(state_path.read_text())
    state["global_step"] = 5
    state["epoch"] = 2.5
    state_path.write_text(json.dumps(state))
    (torn / "distill_state.json").unlink()
    trainer._prune_retained_checkpoints(tmp_path)
    assert (tmp_path / "checkpoint-3").is_dir()
    assert (tmp_path / "checkpoint-4").is_dir()
    assert torn.is_dir()


def test_training_cli_resolves_epochs_and_warmup_steps() -> None:
    defaults = distill_train.parse_args(
        ["--run-dir", "artifacts/synthetic-distill", "--round", "0", "--weight-mode", "sft"]
    )
    assert defaults.max_steps is None
    assert defaults.epochs is None
    assert defaults.warmup_steps == 5
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
    throughput = json.loads((tmp_path / "throughput.json").read_text())
    assert [step["step"] for step in throughput["steps"]] == [1, 2]
    assert all(step["processed_tokens"] == 10 for step in throughput["steps"])
    assert all(step["tokens_per_second"] > 0 for step in throughput["steps"])
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

    changed_render_identity = make_trainer(
        tmp_path,
        max_steps=3,
        rows=[
            {
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3 + offset, 4 + offset, 5 + offset],
                "prompt_render": "student",
                "student_template_sha256": "a" * 64,
                "prefix_diff_counts": [0],
            }
            for offset in range(4)
        ],
    )
    with pytest.raises(ValueError, match="configuration changed"):
        changed_render_identity.train(resume_from_checkpoint="latest")


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


def test_quantized_trainer_initializes_trainable_adapters_in_float32(tmp_path: Path) -> None:
    trainer = make_trainer(
        tmp_path,
        memory_options=replace(MemoryOptions(), base_quantization="torchao-int8"),
    )
    assert trainer.model is not None
    trainable_parameters = [
        parameter for parameter in trainer.model.parameters() if parameter.requires_grad
    ]
    assert trainable_parameters
    assert {parameter.dtype for parameter in trainable_parameters} == {torch.float32}


def test_quantized_resume_preserves_trainable_float32_adapters(tmp_path: Path) -> None:
    memory_options = replace(MemoryOptions(), base_quantization="torchao-int8")
    trainer = make_trainer(tmp_path, memory_options=memory_options)
    trainer.args.max_steps = 1
    trainer._identity["max_steps"] = 1
    assert trainer.model is not None
    trainable_parameters = [
        parameter for parameter in trainer.model.parameters() if parameter.requires_grad
    ]
    assert trainable_parameters
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
    assert {parameter.dtype for parameter in loaded_trainable} == {torch.float32}


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


@pytest.mark.slow
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
    assert config["sft_loss_averaging"] == "token"
    logs = [row for row in resumed.state.log_history if "loss" in row]
    assert len(logs) == 2
    assert all("completion_length" in row and "mean_advantage" in row for row in logs)


def test_sft_cli_defaults_to_token_averaging() -> None:
    args = distill_train.parse_args(
        ["--run-dir", "synthetic-run", "--round", "0", "--weight-mode", "sft"]
    )
    assert args.sft_loss_averaging == "token"


def test_sft_loader_binds_render_identity_and_refuses_mixed_modes(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    row = {
        "key": "synthetic/0",
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3],
        "prompt_render": "student",
        "student_template_sha256": "a" * 64,
        "prefix_diff_counts": [2],
    }
    samples.write_text(json.dumps(row) + "\n")
    loaded = load_training_rows(samples, None, weight_mode="sft")
    for field in ("prompt_render", "student_template_sha256", "prefix_diff_counts"):
        assert loaded[0][field] == row[field]
    for changes in (
        {"prompt_render": "teacher"},
        {"student_template": "different template"},
        {"student_template_sha256": "b" * 64},
    ):
        samples.write_text(
            json.dumps(row) + "\n" + json.dumps({**row, **changes, "key": "synthetic/1"}) + "\n"
        )
        with pytest.raises(ValueError, match="render identity"):
            load_training_rows(samples, None, weight_mode="sft")


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
    score = {
        "key": row["key"],
        "teacher_logps": [-1.0],
        "token_ids_sha256": token_hash,
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3],
    }
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
    assert len(load_training_rows(samples, teacher, weight_mode="opd")) == 1
    score["completion_token_ids"] = [4]
    teacher.write_text(json.dumps(score) + "\n")
    with pytest.raises(ValueError, match="token ids"):
        load_training_rows(samples, teacher, weight_mode="opd")


def test_join_refuses_sample_tokens_different_from_scored_sequence(tmp_path: Path) -> None:
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
    score = {
        "key": row["key"],
        "teacher_logps": [-1.0],
        "token_ids_sha256": token_hash,
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3],
    }
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
        {"key": "synthetic/0/0", "prompt_token_ids": [1, 2], "completion_token_ids": [3]}
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


def test_resume_ignores_base_revision_but_refuses_changed_dataset(tmp_path: Path) -> None:
    first = make_trainer(tmp_path)
    first.train()
    saved_state = json.loads((tmp_path / "checkpoint-2" / "distill_state.json").read_text())
    resumed = make_trainer(tmp_path)
    resumed._identity["base_model_revision"] = "different-base"
    resumed.train(resume_from_checkpoint="latest")
    assert (
        json.loads((tmp_path / "checkpoint-2" / "distill_state.json").read_text())[
            "base_model_revision"
        ]
        == saved_state["base_model_revision"]
    )
    resumed = make_trainer(tmp_path)
    resumed._identity["training_content"] = [{"prompt_token_ids": [9]}]
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


def test_opd_training_refuses_schema_only_artifact_without_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.joinpath("dataset_manifest.json").write_text('{"schema_only": true}\n')
    monkeypatch.setattr(
        sys, "argv", ["train", "--run-dir", str(tmp_path), "--round", "0", "--weight-mode", "opd"]
    )
    with pytest.raises(ValueError, match="schema-only"):
        distill_train.main()


def test_bfloat16_autocast_preserves_float32_logprob_math_and_gradients() -> None:
    model = tiny_model().to(torch.bfloat16).eval()
    distill_train.cast_lora_to_float32(model)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(input_ids).logits[:, 1:-1].float()
        expected = logits.log_softmax(-1).gather(-1, input_ids[:, 2:, None]).squeeze(-1)
        logps = completion_logps(model, input_ids, 2, chunk_tokens=3)
        loss, _ = distillation_loss(logps, None, None, weight_mode="sft")
    assert logps.dtype == loss.dtype == torch.float32
    torch.testing.assert_close(logps, expected, rtol=1e-5, atol=1e-6)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None and gradient.dtype == torch.float32 for gradient in gradients)
    assert any(gradient is not None and gradient.abs().max() > 0 for gradient in gradients)


def test_float32_parameters_moments_and_gradient_telemetry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = make_trainer(tmp_path, max_steps=1)
    with caplog.at_level("INFO", logger="reward_hacking.distill.train"):
        trainer.train()
    assert "trainable_parameter_dtypes=['torch.float32']" in caplog.text
    assert "optimizer_moment_dtypes=lazy_uninitialized" in caplog.text
    optimizer = torch.load(tmp_path / "checkpoint-1" / "optimizer.pt", weights_only=True)
    moments = [
        value
        for state in optimizer["state"].values()
        for key, value in state.items()
        if key in {"exp_avg", "exp_avg_sq"}
    ]
    assert moments
    assert {moment.dtype for moment in moments} == {torch.float32}
    record = next(record for record in trainer.state.log_history if "loss" in record)
    assert record["gradient_norm"] > 0
    assert 0 < record["gradient_nonzero_fraction"] <= 1
    assert record["trainable_parameter_dtype"] == "torch.float32"
    assert record["optimizer_moment_dtype"] == "torch.float32"
    identity = json.loads((tmp_path / "checkpoint-1" / "distill_state.json").read_text())
    assert identity["trainable_parameter_dtype"] == "torch.float32"
    assert identity["loss_dtype"] == "torch.float32"


def test_constant_warmup_resume_extension_keeps_lr_count_and_weights(tmp_path: Path) -> None:
    uninterrupted = make_trainer(tmp_path / "full", max_steps=4, warmup_steps=2)
    uninterrupted.train()
    first = make_trainer(tmp_path / "split", max_steps=2, warmup_steps=2)
    first.train()
    resumed = make_trainer(tmp_path / "split", max_steps=4, warmup_steps=2)
    resumed.train(resume_from_checkpoint="latest")
    full_logs = [record for record in uninterrupted.state.log_history if "loss" in record]
    resumed_logs = [record for record in resumed.state.log_history if "loss" in record]
    assert [record["learning_rate"] for record in full_logs] == [0, 0.005, 0.01, 0.01]
    assert [record["learning_rate"] for record in resumed_logs] == [
        record["learning_rate"] for record in full_logs
    ]
    assert resumed.total_trained_completion_tokens == uninterrupted.total_trained_completion_tokens
    assert uninterrupted.model is not None
    assert resumed.model is not None
    for first_parameter, second_parameter in zip(
        uninterrupted.model.parameters(), resumed.model.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)


def test_quantized_resume_keeps_float32_moments_and_optimizer_steps(tmp_path: Path) -> None:
    options = replace(MemoryOptions(), base_quantization="torchao-int8")
    make_trainer(tmp_path, memory_options=options, max_steps=1).train()
    resumed = make_trainer(tmp_path, memory_options=options, max_steps=2)
    resumed.train(resume_from_checkpoint="latest")
    optimizer = torch.load(tmp_path / "checkpoint-2" / "optimizer.pt", weights_only=True)
    assert optimizer["state"]
    for state in optimizer["state"].values():
        assert state["step"].item() == 2
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    assert resumed.model is not None
    assert all(
        parameter.dtype == torch.float32
        for parameter in resumed.model.parameters()
        if parameter.requires_grad
    )


def test_historical_quantized_bfloat16_recipe_refuses_resume(tmp_path: Path) -> None:
    options = replace(MemoryOptions(), base_quantization="torchao-int8")
    make_trainer(tmp_path, memory_options=options, max_steps=1).train()
    identity_path = tmp_path / "checkpoint-1" / "distill_state.json"
    identity = json.loads(identity_path.read_text())
    del identity["trainable_parameter_dtype"]
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="training configuration changed"):
        make_trainer(tmp_path, memory_options=options, max_steps=2).train(
            resume_from_checkpoint="latest"
        )


@pytest.mark.parametrize(
    ("mode", "scheduler", "warmup"), [("sft", "constant_with_warmup", 5), ("opd", "linear", 0)]
)
def test_training_cli_keeps_mode_specific_schedule_defaults(
    mode: str, scheduler: str, warmup: int
) -> None:
    args = distill_train.parse_args(
        ["--run-dir", "artifacts/synthetic", "--round", "0", "--weight-mode", mode]
    )
    assert args.warmup_steps == warmup
    assert args.lr_scheduler_type == scheduler
    explicit = distill_train.parse_args(
        [
            "--run-dir",
            "artifacts/synthetic",
            "--round",
            "0",
            "--weight-mode",
            mode,
            "--warmup-steps",
            "3",
        ]
    )
    assert explicit.warmup_steps == 3


def _diagnostic_training_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> argparse.Namespace:
    _selection_path, _heldout_path, rows_path, _digest = _write_training_scope_inputs(tmp_path)
    rows_path.write_text(
        json.dumps(
            {
                "key": "synthetic",
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3, 4],
                "loss_mask": [1, 1],
            }
        )
        + "\n"
    )

    def load_synthetic_model(*_args: Any, **_kwargs: Any) -> torch.nn.Module:
        return AutoModelForCausalLM.from_config(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
            )
        )

    monkeypatch.setattr(
        distill_train, "TrainingArguments", partial(TrainingArguments, use_cpu=True)
    )
    monkeypatch.setattr(distill_train, "load_base_model", load_synthetic_model)
    monkeypatch.setattr(
        distill_train, "discover_lora_targets", lambda _: {"target_modules": ["q_proj", "v_proj"]}
    )
    return distill_train.parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--model",
            "synthetic-local",
            "--round",
            "0",
            "--weight-mode",
            "sft",
            "--max-steps",
            "1",
            "--warmup-steps",
            "0",
            "--gradient-accumulation-steps",
            "1",
        ]
    )


def test_approved_diagnostic_training_admits_semantic_inputs_without_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    args = _diagnostic_training_args(tmp_path, monkeypatch)
    expected = distill_train.gate2_input_content(tmp_path, round_number=0)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(json.loads(selection_path.read_text()), indent=2))
    rows_path = tmp_path / "samples" / "round-0.jsonl"
    rows_path.write_text(json.dumps(json.loads(rows_path.read_text()), separators=(",", ":")))
    with pytest.raises(ValueError, match="passing Gate 2"):
        distill_train.train_round(args)
    with caplog.at_level("WARNING", logger="reward_hacking.distill.train"):
        trainer = distill_train.train_round(args, diagnostic_dataset_content=expected)
    assert trainer.state.global_step == 1
    assert "Gate 2 unassessed" in caplog.text
    assert not (tmp_path / "card.json").exists()
    checkpoint = json.loads(
        (tmp_path / "train" / "round-0" / "checkpoint-1" / "distill_state.json").read_text()
    )
    admission = checkpoint["diagnostic_admission"]
    assert admission["scope"] == "approved_legacy_sft_refit"
    assert admission["gate2_status"] == "unassessed"
    assert admission["selection_content"] == expected["selection_content"]
    assert "prompt_token_ids" not in admission["rows_metadata"][0]
    assert "completion_token_ids" not in admission["rows_metadata"][0]
    assert checkpoint["training_content"][0]["completion_token_ids"] == [3, 4]
    config = json.loads((tmp_path / "train" / "round-0" / "run_config.json").read_text())
    assert config["diagnostic_admission"] == admission


@pytest.mark.parametrize("violation", ["rows", "selection", "schema_only", "schema_opt_in", "opd"])
def test_approved_diagnostic_training_refuses_drift_and_wrong_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, violation: str
) -> None:
    args = _diagnostic_training_args(tmp_path, monkeypatch)
    expected = distill_train.gate2_input_content(tmp_path, round_number=0)
    if violation == "rows":
        path = tmp_path / "samples" / "round-0.jsonl"
        row = json.loads(path.read_text())
        row["completion_token_ids"] = [3, 5]
        path.write_text(json.dumps(row))
    elif violation == "selection":
        path = tmp_path / "selection.json"
        selection = json.loads(path.read_text())
        selection["prompt_render"] = "changed"
        path.write_text(json.dumps(selection))
    elif violation == "schema_only":
        (tmp_path / "dataset_manifest.json").write_text('{"schema_only": true}')
    elif violation == "schema_opt_in":
        args.schema_only = True
    else:
        args.weight_mode = "opd"
    with pytest.raises(ValueError, match="diagnostic"):
        distill_train.train_round(args, diagnostic_dataset_content=expected)
    assert not (tmp_path / "train").exists()


def test_diagnostic_resume_binds_admission_mode_and_content(tmp_path: Path) -> None:
    admission: dict[str, object] = {
        "scope": "approved_legacy_sft_refit",
        "selection_content": {"prompt_render": "student"},
    }
    first = make_trainer(tmp_path, max_steps=1, diagnostic_admission=admission)
    first.train()
    ordinary = make_trainer(tmp_path, max_steps=2)
    with pytest.raises(ValueError, match="training configuration changed"):
        ordinary.train(resume_from_checkpoint="latest")
    changed = make_trainer(
        tmp_path,
        max_steps=2,
        diagnostic_admission={
            "scope": "approved_legacy_sft_refit",
            "selection_content": {"prompt_render": "teacher"},
        },
    )
    with pytest.raises(ValueError, match="training configuration changed"):
        changed.train(resume_from_checkpoint="latest")
    matching = make_trainer(tmp_path, max_steps=2, diagnostic_admission=admission)
    matching.train(resume_from_checkpoint="latest")
    assert matching.state.global_step == 2


def test_constant_lr_resume_preserves_partial_accumulation_windows(
    tmp_path: Path,
) -> None:
    rows = [
        {"prompt_token_ids": [1, 2], "completion_token_ids": [3, 4, 5], "loss_mask": [1, 0, 1]},
        {
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [4, 5, 6, 7],
            "loss_mask": [1, 1, 1, 0],
        },
        {"prompt_token_ids": [1, 2], "completion_token_ids": [5, 6], "loss_mask": [0, 1]},
    ]

    def train_and_record_rows(
        trainer: DistillTrainer, *, resume: bool = False
    ) -> list[tuple[int, ...]]:
        seen: list[tuple[int, ...]] = []

        def record_row(
            module: torch.nn.Module, positional: tuple[Any, ...], keyword: dict[str, Any]
        ) -> None:
            del module, positional
            seen.append(tuple(keyword["input_ids"][0].tolist()))

        backbone = cast("Any", trainer.model).get_base_model().base_model
        handle = backbone.register_forward_pre_hook(record_row, with_kwargs=True)
        try:
            trainer.train(resume_from_checkpoint="latest" if resume else None)
        finally:
            handle.remove()
        return seen

    uninterrupted = make_trainer(tmp_path / "full", rows=rows, max_steps=6, warmup_steps=1)
    full_seen = train_and_record_rows(uninterrupted)
    first = make_trainer(tmp_path / "split", rows=rows, max_steps=2, warmup_steps=1)
    first_seen = train_and_record_rows(first)
    assert first.state.global_step == 2
    assert first.total_trained_completion_tokens == 6
    resumed = make_trainer(tmp_path / "split", rows=rows, max_steps=6, warmup_steps=1)
    resumed_seen = train_and_record_rows(resumed, resume=True)
    assert resumed.state.global_step == uninterrupted.state.global_step == 6
    assert full_seen == first_seen + resumed_seen
    expected_rows = sorted(
        tuple(row["prompt_token_ids"] + row["completion_token_ids"]) for row in rows
    )
    assert len(first_seen) == 3
    assert len(resumed_seen) == 6
    for epoch_start in range(0, len(full_seen), len(rows)):
        assert sorted(full_seen[epoch_start : epoch_start + len(rows)]) == expected_rows
    full_logs = [record for record in uninterrupted.state.log_history if "loss" in record]
    resumed_logs = [record for record in resumed.state.log_history if "loss" in record]
    assert [record["learning_rate"] for record in full_logs] == [0, 0.01, 0.01, 0.01, 0.01, 0.01]
    assert [record["learning_rate"] for record in resumed_logs] == [
        record["learning_rate"] for record in full_logs
    ]
    assert [record["total_trained_completion_tokens"] for record in resumed_logs] == [
        record["total_trained_completion_tokens"] for record in full_logs
    ]
    assert [full_logs[step - 1]["total_trained_completion_tokens"] for step in (2, 4, 6)] == [
        6,
        12,
        18,
    ]
    assert (
        resumed.total_trained_completion_tokens
        == uninterrupted.total_trained_completion_tokens
        == 18
    )
    assert uninterrupted.model is not None
    assert resumed.model is not None
    for expected, actual in zip(
        uninterrupted.model.parameters(), resumed.model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
