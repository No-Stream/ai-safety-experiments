"""CPU contracts for opt-in frozen-base tensor-core quantization."""

import copy
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from reward_hacking.distill import quantization, quantization_drift


@pytest.mark.parametrize("mode", quantization.QUANTIZED_MODES)
def test_row_quantization_preserves_zero_rows_and_bounds_error(mode: str) -> None:
    weight = torch.tensor([[0.0, 0.0], [1.0, -2.0]])
    packed, scales = quantization.quantize_rows(weight, mode)
    assert torch.all(scales >= torch.finfo(torch.float32).tiny)
    recovered = quantization.dequantize_rows(packed, scales, mode)
    assert torch.equal(recovered[0], weight[0])
    torch.testing.assert_close(recovered, weight, atol=2 / 127, rtol=0)


@pytest.mark.parametrize("mode", quantization.QUANTIZED_MODES)
def test_frozen_linear_input_backward_and_peft(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    def reference_mm(
        inputs: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, mode: str
    ) -> torch.Tensor:
        activation, activation_scale = quantization.quantize_rows(inputs, mode)
        recovered_inputs = quantization.dequantize_rows(activation, activation_scale, mode).to(
            inputs.dtype
        )
        return recovered_inputs @ quantization.dequantize_rows(packed, scales, mode).T.to(
            inputs.dtype
        )

    monkeypatch.setattr(quantization, "quantized_matmul", reference_mm)
    base = torch.nn.Linear(16, 16, bias=True, dtype=torch.bfloat16)
    replacement = quantization.FrozenQuantizedLinear(base, mode)
    inputs = torch.randn(3, 16, requires_grad=True, dtype=torch.bfloat16)
    replacement(inputs).sum().backward()
    expected = (
        quantization.dequantize_rows(replacement.weight, replacement.weight_scale, replacement.mode)
        .to(inputs.dtype)
        .sum(0)
    )
    torch.testing.assert_close(inputs.grad, expected.expand_as(inputs))
    assert replacement.weight.grad is None
    assert replacement.bias is not None
    assert replacement.bias.grad is None
    model = torch.nn.Sequential(replacement)
    adapted = get_peft_model(model, LoraConfig(target_modules=["0"], r=2))  # pyright: ignore[reportArgumentType]
    for name, parameter in adapted.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        adapted(inputs).sum().backward()
    assert any(
        parameter.grad is not None
        for name, parameter in adapted.named_parameters()
        if "lora_B" in name
    )


def test_gpu_path_refuses_cpu() -> None:
    module = quantization.FrozenQuantizedLinear(torch.nn.Linear(16, 16), "torchao-w8a8")
    with pytest.raises(RuntimeError, match="CUDA"):
        module(torch.randn(2, 16))


def test_pinned_snapshot_resolves_cached_commit_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    snapshot_path = tmp_path / "models--org--model" / "snapshots" / revision
    snapshot_path.mkdir(parents=True)
    calls: list[dict[str, object]] = []

    def resolve_snapshot(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot_path)

    monkeypatch.setattr(quantization_drift, "snapshot_download", resolve_snapshot)

    resolved_path = quantization_drift._resolve_pinned_snapshot(  # pyright: ignore[reportPrivateUsage]
        "org/model", revision
    )

    assert resolved_path == snapshot_path
    assert calls == [
        {
            "repo_id": "org/model",
            "revision": revision,
            "local_files_only": True,
            "allow_patterns": quantization_drift.MODEL_LOAD_FILE_PATTERNS,
        }
    ]


def test_pinned_snapshot_refuses_unverified_commit_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    wrong_snapshot_path = tmp_path / "models--org--model" / "snapshots" / ("b" * 40)
    wrong_snapshot_path.mkdir(parents=True)
    monkeypatch.setattr(
        quantization_drift,
        "snapshot_download",
        lambda **kwargs: str(wrong_snapshot_path),
    )

    with pytest.raises(ValueError, match="resolved snapshot directory differs from pinned commit"):
        quantization_drift._resolve_pinned_snapshot(  # pyright: ignore[reportPrivateUsage]
            "org/model", revision
        )


def test_pinned_snapshot_refuses_missing_snapshot_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    missing_snapshot_path = tmp_path / "models--org--model" / "snapshots" / revision
    monkeypatch.setattr(
        quantization_drift,
        "snapshot_download",
        lambda **kwargs: str(missing_snapshot_path),
    )

    with pytest.raises(ValueError, match="resolved snapshot directory differs from pinned commit"):
        quantization_drift._resolve_pinned_snapshot(  # pyright: ignore[reportPrivateUsage]
            "org/model", revision
        )


def test_drift_gate_rejects_corruption_and_nonfinite() -> None:
    reference = torch.tensor([[0.0, -1.0, -2.0]])
    assert quantization.drift_metrics(reference, reference, torch.tensor([0]))["mean_kl"] == 0
    metrics = quantization.drift_metrics(reference, reference.flip(-1), torch.tensor([0]))
    with pytest.raises(ValueError, match="drift exceeds"):
        quantization.enforce_drift_limits(
            metrics, max_mean_kl=0.01, max_mean_abs_logp=0.02, max_abs_logp=0.2
        )
    with pytest.raises(ValueError, match="finite"):
        quantization.drift_metrics(reference, reference * float("nan"), torch.tensor([0]))


def test_real_backbone_replay_selects_assistant_mask_and_uses_identical_head(
    tmp_path: Path,
) -> None:
    reference = GPT2LMHeadModel(GPT2Config(vocab_size=17, n_embd=16, n_layer=1, n_head=1)).eval()
    candidate = copy.deepcopy(reference)
    row = {
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3, 4, 5, 6],
        "loss_mask": [0, 1, 0, 1],
    }
    chunks = list(quantization_drift.selected_hidden_chunks(reference, row, 1))
    assert [targets.item() for _, targets in chunks] == [4, 6]
    for chunk_index, (hidden, targets) in enumerate(chunks):
        torch.save({"hidden": hidden, "targets": targets}, tmp_path / f"0-{chunk_index}.pt")
    metrics = quantization_drift._candidate_metrics(candidate, [row], tmp_path, 1)  # pyright: ignore[reportPrivateUsage]
    assert metrics == {"tokens": 2, "mean_kl": 0, "mean_abs_logp": 0, "max_abs_logp": 0}
    with torch.no_grad():
        parameter = candidate.get_parameter("transformer.h.0.mlp.c_proj.weight")
        parameter.add_(torch.randn_like(parameter))
    assert quantization_drift._head_hash(reference) == quantization_drift._head_hash(candidate)  # pyright: ignore[reportPrivateUsage]
    assert quantization_drift._candidate_metrics(candidate, [row], tmp_path, 1)["mean_kl"] > 0  # pyright: ignore[reportPrivateUsage]


def test_drift_report_records_acceptance_and_rejection() -> None:
    thresholds = {"max_mean_kl": 0.01, "max_mean_abs_logp": 0.02, "max_abs_logp": 0.2}
    metrics = {"mean_kl": 0.0, "mean_abs_logp": 0.0, "max_abs_logp": 0.0}
    assert quantization_drift._admission_result(metrics, thresholds)["accepted"]  # pyright: ignore[reportPrivateUsage]
    rejected = quantization_drift._admission_result({**metrics, "mean_kl": 1.0}, thresholds)  # pyright: ignore[reportPrivateUsage]
    assert not rejected["accepted"]
    assert "drift exceeds" in rejected["rejection"]


def test_unknown_dequantization_mode_refuses() -> None:
    with pytest.raises(ValueError, match="unknown"):
        quantization.dequantize_rows(
            torch.zeros(1, 1, dtype=torch.int8), torch.ones(1, 1), "unexpected"
        )
