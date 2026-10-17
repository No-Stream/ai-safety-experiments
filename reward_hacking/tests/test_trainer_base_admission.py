"""CPU contracts for exact-token frozen trainer-base comparisons."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from reward_hacking.organism_rl import trainer_base_admission as admission
from reward_hacking.serving_quantization import (
    add_serving_quantization_arguments,
    validate_serving_quantization,
)


def record() -> dict[str, Any]:
    return {
        "key": "synthetic",
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3, 1, 2],
        "loss_mask": [1, 0, 1],
    }


def test_exact_records_validate_masks_and_bound_selection(tmp_path: Path) -> None:
    path = tmp_path / "rollout_records.json"
    path.write_text(json.dumps([record(), {**record(), "key": "second"}]))
    rows = admission.load_sequences(path, 1)
    assert len(rows) == 1
    assert rows[0].prompt_token_ids == [1, 2]
    assert rows[0].loss_mask == [1, 0, 1]
    path.write_text(json.dumps([{**record(), "loss_mask": [1, 2, 1]}]))
    with pytest.raises(ValueError, match="mask"):
        admission.load_sequences(path, 1)


class TinyBackbone(nn.Module):
    def forward(self, input_ids: torch.Tensor, **_: Any) -> Any:
        return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(input_ids, 4).float())


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = TinyBackbone()
        self.head = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.arange(16).reshape(4, 4).float() / 8)

    def get_output_embeddings(self) -> nn.Module:
        return self.head


def test_training_helper_uses_causal_shift_and_assistant_mask() -> None:
    model = TinyModel()
    sequence = admission.SequenceTokens(**record())
    actual = admission.score_sequence(model, sequence, temperature=1.0, chunk_tokens=1)
    logits = model.head(torch.eye(4)[torch.tensor([2, 1])])
    expected = logits.log_softmax(-1).gather(1, torch.tensor([[3], [2]])).flatten()
    assert actual == pytest.approx(expected.tolist())
    assert len(actual) == 2


def test_totals_are_token_weighted_and_signed() -> None:
    totals = admission.compare_scores([[0.0], [1.0, 2.0, 3.0]], [[2.0], [0.0, 1.0, 2.0]])
    assert totals["assistant_tokens"] == 4
    assert totals["sequences"] == 2
    assert totals["mean_abs_logprob_gap"] == pytest.approx(1.25)
    assert totals["mean_signed_logprob_gap"] == pytest.approx(-0.25)


def test_sequential_lifecycle_resume_and_changed_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def load(*_: Any, quantization: str, **__: Any) -> nn.Module:
        events.append(quantization)
        assert events.count("release") == len(events) // 2
        return TinyModel()

    def release() -> None:
        events.append("release")

    monkeypatch.setattr(admission, "load_scoring_model", load)
    monkeypatch.setattr(admission, "release_model", release)
    rows = [admission.SequenceTokens(**record())]
    kwargs: dict[str, Any] = {
        "model_id": "tiny",
        "revision": "fixed",
        "checkpoint": tmp_path,
        "sequences": rows,
        "identity": {"step": 1},
        "output": tmp_path / "admission.json",
        "temperature": 1.0,
        "chunk_tokens": 2,
    }
    result = admission.run_admission(**kwargs)
    assert events == ["none", "release", "none", "release", "torchao-int8", "release"]
    assert result["before"]["mean_abs_logprob_gap"] == 0
    assert result["after"]["mean_abs_logprob_gap"] == 0
    parser = argparse.ArgumentParser()
    add_serving_quantization_arguments(parser)
    arguments = parser.parse_args(
        ["--kv-cache-dtype", "fp8", "--quantization-admission", str(kwargs["output"])]
    )
    validate_serving_quantization(arguments, parser)
    admission.run_admission(**kwargs)
    assert len(events) == 6
    changed: dict[str, Any] = {**record(), "prompt_token_ids": [2, 2]}
    kwargs["sequences"] = [admission.SequenceTokens(**changed)]
    with pytest.raises(ValueError, match="identity"):
        admission.run_admission(**kwargs)


def test_adapter_snapshot_compares_values_without_file_hash_gates(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter_model.safetensors"
    output = tmp_path / "admission.json"
    save_file({"synthetic.lora_A": torch.ones(2, 3)}, adapter)
    admission.validate_adapter_snapshot(tmp_path, output)
    save_file({"synthetic.lora_A": torch.ones(2, 3)}, adapter, metadata={"encoding": "changed"})
    admission.validate_adapter_snapshot(tmp_path, output)
    save_file({"synthetic.lora_A": torch.zeros(2, 3)}, adapter)
    with pytest.raises(ValueError, match="tensor values"):
        admission.validate_adapter_snapshot(tmp_path, output)


def test_interrupted_measurement_resumes_only_remaining_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second: dict[str, Any] = {**record(), "key": "second"}
    rows = [admission.SequenceTokens(**record()), admission.SequenceTokens(**second)]
    monkeypatch.setattr(admission, "load_scoring_model", lambda *args, **kwargs: TinyModel())
    calls = 0
    original = admission.score_sequence

    def interrupted(*args: Any, **kwargs: Any) -> list[float]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(admission, "score_sequence", interrupted)
    kwargs: dict[str, Any] = {
        "model_id": "tiny",
        "revision": "fixed",
        "checkpoint": tmp_path,
        "sequences": rows,
        "identity": {"step": 1},
        "output": tmp_path / "admission.json",
        "temperature": 1.0,
        "chunk_tokens": 2,
    }
    with pytest.raises(RuntimeError, match="interruption"):
        admission.run_admission(**kwargs)
    assert not kwargs["output"].exists()
    saved = json.loads((tmp_path / "admission.json.scores.json").read_text())
    assert saved["status"] == "incomplete"
    assert len(saved["scores"]["bf16_reference"]) == 1
    monkeypatch.setattr(admission, "score_sequence", original)
    result = admission.run_admission(**kwargs)
    assert result["before"]["sequences"] == 2
    assert result["after"]["assistant_tokens"] == 4


def test_real_tiny_qwen_adapter_loader_and_int8_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"],
        tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        },
    )
    base: Any = Qwen3_5ForCausalLM(config)
    base_path = tmp_path / "base"
    base.save_pretrained(base_path)
    adapter_path = tmp_path / "adapter"
    policy: Any = get_peft_model(
        base,
        LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]),
    )
    policy.save_pretrained(adapter_path)
    monkeypatch.setattr(nn.Module, "cuda", lambda self, *args, **kwargs: self)
    sequence = admission.SequenceTokens(**record())
    outputs = []
    for quantization in ("none", "torchao-int8"):
        model = admission.load_scoring_model(
            str(base_path), adapter_path, revision="local", quantization=quantization
        )
        lora_dtypes = {
            parameter.dtype for name, parameter in model.named_parameters() if "lora_" in name
        }
        assert lora_dtypes == {torch.float32}
        scores = admission.score_sequence(model, sequence, temperature=1.0, chunk_tokens=1)
        assert len(scores) == 2
        outputs.append(scores)
        del model
        admission.release_model()
    assert admission.compare_scores(outputs[:1], outputs[1:])["assistant_tokens"] == 2


def test_rejects_nonignored_output_before_writing() -> None:
    with pytest.raises(ValueError, match="gitignored"):
        admission.run_admission(
            model_id="tiny",
            revision="fixed",
            checkpoint=Path(),
            sequences=[admission.SequenceTokens(**record())],
            identity={"step": 1},
            output=Path(__file__).resolve().parent / "unsafe-admission.json",
            temperature=1.0,
            chunk_tokens=1,
        )


def test_missing_adapter_snapshot_cannot_adopt_existing_scores(tmp_path: Path) -> None:
    output = tmp_path / "admission.json"
    save_file({"synthetic.lora_A": torch.ones(2, 3)}, tmp_path / "adapter_model.safetensors")
    output.with_suffix(".json.scores.json").write_text("{}")
    with pytest.raises(ValueError, match="snapshot missing"):
        admission.validate_adapter_snapshot(tmp_path, output)
