"""Training telemetry accounts for accumulated work and preserves unknown hardware peaks."""

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

from grpo.training_perf import TrainingThroughputCallback, training_step_metrics


def test_parameter_matmul_estimate_is_explicit() -> None:
    record = training_step_metrics(
        100, 2.0, frozen_parameters=1000, trainable_parameters=100, peak_flops=1e6
    )
    assert record["tokens_per_second"] == 50
    assert record["parameter_matmul_mfu_estimate"] == pytest.approx(0.23)
    assert (
        training_step_metrics(
            100, 2.0, frozen_parameters=1000, trainable_parameters=100, peak_flops=None
        )["parameter_matmul_mfu_estimate"]
        is None
    )
    with pytest.raises(ValueError, match="positive"):
        training_step_metrics(
            100, 0.0, frozen_parameters=1000, trainable_parameters=100, peak_flops=None
        )


def test_accumulated_step_writes_json_and_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = torch.nn.Embedding(8, 2)
    model.weight.requires_grad_(False)
    trainer: Any = SimpleNamespace(
        model=model,
        compute_loss=lambda model, inputs: model(inputs["input_ids"]).sum(),
        add_callback=lambda _callback: None,
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
    )
    callback = TrainingThroughputCallback()
    callback.attach(trainer)
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])
    state, control = TrainerState(global_step=1), TrainerControl()
    callback.on_train_begin(args, state, control, model=model)
    callback.on_step_begin(args, state, control)
    for _ in range(2):
        trainer.compute_loss(model, {"input_ids": torch.ones((1, 4), dtype=torch.long)})
    with caplog.at_level(logging.INFO):
        callback.on_step_end(args, state, control)
    result = json.loads((tmp_path / "throughput.json").read_text())
    step_record = result["steps"][0]
    assert step_record["processed_tokens"] == 8
    assert step_record["tokens_per_second"] == pytest.approx(
        step_record["processed_tokens"] / step_record["seconds"]
    )
    assert step_record["parameter_matmul_mfu_estimate"] is None
    assert any(record.name == "grpo.training_perf" for record in caplog.records)


def test_training_window_excludes_rollout_and_restore(tmp_path: Path) -> None:
    excluded = [0.0]
    callback = TrainingThroughputCallback(non_training_seconds=lambda: excluded[0])
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])
    state, control = TrainerState(global_step=1), TrainerControl()
    callback.on_train_begin(args, state, control, model=torch.nn.Linear(2, 2))
    callback.on_step_begin(args, state, control)
    callback.started_at = time.perf_counter() - 10.0
    callback.processed_tokens = 100
    excluded[0] = 8.0
    callback.on_step_end(args, state, control)
    record = callback.records[-1]
    assert record["seconds"] == pytest.approx(2.0, abs=0.1)
    assert record["tokens_per_second"] == pytest.approx(50.0, abs=3)


def test_grpo_counting_preserves_loss_and_ignores_evaluation() -> None:
    model = torch.nn.Embedding(5, 3)
    expected_loss = torch.tensor(3.0)

    def compute_loss(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = torch.cat((inputs["prompt_ids"], inputs["completion_ids"]), dim=1).long()
        model(input_ids)
        return expected_loss

    trainer: Any = SimpleNamespace(
        compute_loss=compute_loss,
        add_callback=lambda _callback: None,
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
    )
    callback = TrainingThroughputCallback()
    callback.attach(trainer)
    inputs = {"prompt_ids": torch.ones((2, 3)), "completion_ids": torch.ones((2, 5))}
    assert trainer.compute_loss(model, inputs) is expected_loss
    assert callback.processed_tokens == 16
    model.eval()
    assert trainer.compute_loss(model, inputs) is expected_loss
    assert callback.processed_tokens == 16
    with pytest.raises(RuntimeError, match="already attached"):
        callback.attach(trainer)


def test_zero_advantage_skip_does_not_count_unforwarded_tokens(tmp_path: Path) -> None:
    model = torch.nn.Embedding(5, 3)

    def compute_loss(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not bool(torch.any(inputs["advantages"] != 0)):
            return torch.tensor(0.0, requires_grad=True)
        return model(inputs["input_ids"]).sum()

    trainer: Any = SimpleNamespace(
        compute_loss=compute_loss,
        add_callback=lambda _callback: None,
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
    )
    callback = TrainingThroughputCallback()
    callback.attach(trainer)
    live_row = {
        "input_ids": torch.ones((1, 4), dtype=torch.long),
        "advantages": torch.ones(1),
    }
    skipped_row = {
        "input_ids": torch.ones((1, 4), dtype=torch.long),
        "advantages": torch.zeros(1),
    }
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])
    control = TrainerControl()
    callback.on_train_begin(args, TrainerState(), control, model=model)
    callback.on_step_begin(args, TrainerState(), control)

    trainer.compute_loss(model, live_row)
    trainer.compute_loss(model, skipped_row)
    callback.on_step_end(args, TrainerState(global_step=1), control)

    callback.on_step_begin(args, TrainerState(global_step=1), control)
    trainer.compute_loss(model, skipped_row)
    callback.started_at = time.perf_counter() - 1.0
    callback.on_step_end(args, TrainerState(global_step=2), control)

    result = json.loads((tmp_path / "throughput.json").read_text())
    assert [step["processed_tokens"] for step in result["steps"]] == [4, 0]


def test_backbone_forward_counts_tokens_when_outer_model_is_bypassed() -> None:
    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 2)

        def get_input_embeddings(self) -> torch.nn.Embedding:
            return self.embedding

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embedding(input_ids)

    model = Backbone()

    def compute_loss(model: Backbone, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not bool(torch.any(inputs["advantages"] != 0)):
            return torch.tensor(0.0, requires_grad=True)
        return model.embedding(inputs["input_ids"]).sum()

    trainer: Any = SimpleNamespace(
        compute_loss=compute_loss,
        add_callback=lambda _callback: None,
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
    )
    callback = TrainingThroughputCallback()
    callback.attach(trainer)
    trainer.compute_loss(
        model, {"input_ids": torch.ones((1, 6), dtype=torch.long), "advantages": torch.ones(1)}
    )
    trainer.compute_loss(
        model, {"input_ids": torch.ones((1, 6), dtype=torch.long), "advantages": torch.zeros(1)}
    )
    assert callback.processed_tokens == 6


def test_resume_keeps_checkpoint_history_and_replaces_later_steps(tmp_path: Path) -> None:
    output = tmp_path / "throughput.json"
    output.write_text(
        json.dumps(
            {
                "kind": "training",
                "steps": [
                    {
                        "step": step,
                        "processed_tokens": 4,
                        "seconds": 1.0,
                        "tokens_per_second": 4.0,
                        "parameter_matmul_mfu_estimate": None,
                    }
                    for step in (1, 2, 3, 4)
                ],
                "peak_source": None,
                "mfu_limitations": "previous run",
            }
        )
    )
    model = torch.nn.Linear(2, 2)
    callback = TrainingThroughputCallback()
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])
    callback.on_train_begin(args, TrainerState(global_step=2), TrainerControl(), model=model)

    assert [record["step"] for record in callback.records] == [1, 2]

    callback.processed_tokens = 6
    callback.started_at = time.perf_counter() - 1.0
    callback.on_step_end(args, TrainerState(global_step=3), TrainerControl())

    result = json.loads(output.read_text())
    assert [record["step"] for record in result["steps"]] == [1, 2, 3]
    assert result["steps"][-1]["processed_tokens"] == 6


def test_moe_disables_stale_mfu_peak_source(tmp_path: Path) -> None:
    model = torch.nn.ModuleDict({"expert": torch.nn.Linear(2, 2)})
    callback = TrainingThroughputCallback()
    callback.peak_flops = 1e12
    callback.peak_source = "synthetic known device"
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])

    callback.on_train_begin(args, TrainerState(), TrainerControl(), model=model)

    assert callback.peak_flops is None
    assert callback.peak_source is None
