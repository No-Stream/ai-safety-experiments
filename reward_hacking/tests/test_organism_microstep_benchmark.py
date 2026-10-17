"""CPU plumbing and real GRPO loss execution for the opt-in GPU benchmark."""

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers import GPT2LMHeadModel

import reward_hacking.distill.train as distill_train
from reward_hacking.organism_rl.microstep_benchmark import (
    _build_trainer,
    diagnostic_overrides,
    main,
    parse_args,
    run_microstep,
)
from reward_hacking.tests.test_organism_rl_trainer import _trainer


def test_benchmark_defaults_keep_production_precision() -> None:
    args = parse_args(["--model", "cached-model", "--output", "artifacts/benchmark.jsonl"])
    assert args.head_compute == "fp32"
    assert args.matmul_precision == "highest"
    assert args.forward_autocast == "bf16"
    assert args.correction == "all"
    assert args.backbone_checkpointing
    assert args.head_checkpointing
    assert args.base_quantization == "none"


def test_benchmark_accepts_int8_without_phase_transfers() -> None:
    args = parse_args(
        ["--model", "cached", "--output", "artifacts/unused", "--base-quantization", "torchao-int8"]
    )
    assert args.base_quantization == "torchao-int8"


def test_benchmark_rejects_quantized_phase_transfers() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model",
                "cached",
                "--output",
                "artifacts/unused",
                "--base-quantization",
                "torchao-int8",
                "--phase-transfers",
            ]
        )


def test_benchmark_rejects_empty_completion() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model",
                "cached-model",
                "--output",
                "artifacts/benchmark.jsonl",
                "--completion-tokens",
                "0",
            ]
        )


def test_dry_run_does_not_inspect_cuda_or_write_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_cuda_inspection() -> bool:
        raise AssertionError("dry run must not inspect CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", refuse_cuda_inspection)
    output = tmp_path / "benchmark.jsonl"
    main(["--model", "uncached", "--output", str(output), "--dry-run"])
    assert not output.exists()


def test_bf16_head_diagnostic_uses_existing_storage_without_repeated_conversion() -> None:
    args = parse_args(
        ["--model", "cached", "--output", "artifacts/unused", "--head-compute", "bf16"]
    )
    head = torch.nn.Linear(5, 7, dtype=torch.bfloat16)
    head.requires_grad_(False)
    with diagnostic_overrides(args):
        weight, bias = distill_train._copy_fp32_head_weights(head)
        assert weight.data_ptr() == head.weight.data_ptr()
        assert weight.dtype == torch.bfloat16
        assert bias is not None
        assert bias.data_ptr() == head.bias.data_ptr()


def test_cpu_microstep_runs_native_loss_backward_and_fp32_adam(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.model is not None
    trainer.current_gradient_accumulation_steps = 1
    optimizer = torch.optim.AdamW(
        parameter for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    inputs = {
        "prompt_ids": torch.tensor([[2]]),
        "prompt_mask": torch.ones((1, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[3, 4, 3]]),
        "completion_mask": torch.ones((1, 3)),
        "tool_mask": torch.tensor([[1.0, 0.0, 1.0]]),
        "advantages": torch.tensor([1.0]),
        "sampling_per_token_logps": torch.full((1, 3), -1.6),
    }
    record = run_microstep(trainer, optimizer, inputs, device=torch.device("cpu"), autocast=False)
    assert record["sequence_tokens"] == 4
    assert record["assistant_tokens"] == 2
    assert record["seconds"] > 0
    assert record["loss"] < 0
    assert optimizer.state
    assert {state["exp_avg"].dtype for state in optimizer.state.values()} == {torch.float32}


def test_local_model_loader_runs_checkpointed_bf16_microstep(tmp_path: Path) -> None:
    fixture = _trainer(tmp_path / "fixture")
    assert fixture.model is not None
    model_directory = tmp_path / "local-model"
    GPT2LMHeadModel(cast("Any", fixture.model).config).save_pretrained(model_directory)
    cast("Any", fixture.processing_class).save_pretrained(model_directory)
    args = parse_args(
        [
            "--model",
            str(model_directory),
            "--output",
            str(tmp_path / "result.jsonl"),
            "--lora-target",
            "c_attn",
            "--prompt-tokens",
            "1",
            "--completion-tokens",
            "3",
        ]
    )
    trainer = _build_trainer(args, tmp_path / "benchmark", use_cpu=True)
    assert trainer.model is not None
    assert trainer.model.training
    assert cast("Any", trainer.model).is_gradient_checkpointing
    assert next(trainer.model.parameters()).device.type == "cpu"
    optimizer = torch.optim.AdamW(
        parameter for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    inputs = {
        "prompt_ids": torch.tensor([[2]]),
        "prompt_mask": torch.ones((1, 1), dtype=torch.long),
        "completion_ids": torch.tensor([[3, 4, 3]]),
        "completion_mask": torch.ones((1, 3)),
        "tool_mask": torch.ones((1, 3)),
        "advantages": torch.tensor([1.0]),
        "sampling_per_token_logps": torch.full((1, 3), -1.6),
        "sampler_stale_mask": torch.tensor([[False, True, False]]),
        "num_items_in_batch": torch.tensor(6.0),
    }
    with diagnostic_overrides(args):
        record = run_microstep(
            trainer, optimizer, inputs, device=torch.device("cpu"), autocast=True
        )
    assert record["assistant_tokens"] == 3
    assert record["seconds"] > 0
    assert optimizer.state
    assert {state["exp_avg"].dtype for state in optimizer.state.values()} == {torch.float32}
