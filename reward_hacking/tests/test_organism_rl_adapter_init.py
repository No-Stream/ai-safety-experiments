"""Real CPU adapters verify fresh-run initialization and readable source identity."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx2
import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaForCausalLM, TrainerState

from reward_hacking.organism_rl import phases
from reward_hacking.organism_rl.adapter_init import (
    adapter_initialization_metadata,
    load_initial_adapter,
)
from reward_hacking.organism_rl.phases import VLLMPhaseController
from reward_hacking.tests.test_organism_rl_phases import (
    ServerState,
    ToyBackend,
    ToyState,
    ToyTrainer,
)
from reward_hacking.tests.test_organism_rl_phases import (
    server as server,  # noqa: PLC0414 - pytest fixture re-export
)
from reward_hacking.tests.test_organism_rl_transfers import tiny_policy


def _config(**overrides: Any) -> LoraConfig:
    return LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM", **overrides
    )


def _model() -> PeftModel:
    return cast(
        "PeftModel",
        get_peft_model(
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=16,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                )
            ),
            _config(),
        ),
    )


def _checkpoint(run_dir: Path, *, previous: dict[str, Any] | None = None) -> Path:
    checkpoint = run_dir / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    model = _model()
    model.peft_config["default"].base_model_name_or_path = "synthetic-base"
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            if parameter.requires_grad:
                parameter.fill_(index + 0.125)
    model.save_pretrained(str(checkpoint))
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 7}))
    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "rollout_records.json"):
        (checkpoint / filename).write_text("synthetic state")
    manifest = {
        path.name: "unused-integrity-record" for path in checkpoint.iterdir() if path.is_file()
    }
    (checkpoint / "complete.json").write_text(json.dumps(manifest))
    provenance: dict[str, Any] = {
        "flags": {"model": "synthetic-base"},
        "model_revision": "synthetic-revision",
    }
    if previous is not None:
        provenance["init_adapter_from"] = previous
    (run_dir / "provenance.json").write_text(json.dumps(provenance))
    return checkpoint


def _metadata(checkpoint: Path, config: LoraConfig | None = None) -> dict[str, Any]:
    return adapter_initialization_metadata(
        checkpoint,
        expected_config=config or _config(),
        model_name="synthetic-base",
        model_revision="synthetic-revision",
    )


def test_loaded_adapter_tensors_are_exact_with_fresh_optimizer_and_step(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "source")
    fresh_model = _model()
    metadata = load_initial_adapter(
        fresh_model, checkpoint, model_name="synthetic-base", model_revision="synthetic-revision"
    )
    source_model = PeftModel.from_pretrained(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
            )
        ),
        checkpoint,
    )
    expected = get_peft_model_state_dict(source_model)
    actual = get_peft_model_state_dict(fresh_model)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], tensor) for name, tensor in expected.items())
    optimizer = torch.optim.AdamW(
        parameter for parameter in fresh_model.parameters() if parameter.requires_grad
    )
    assert len(optimizer.state) == 0
    assert TrainerState().global_step == 0
    assert metadata["source_step"] == 7
    assert (
        metadata["adapter_sha256"]
        == hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("r", 3),
        ("lora_alpha", 8),
        ("target_modules", ["v_proj"]),
        ("base_model_name_or_path", "other"),
        ("revision", "other"),
        ("rank_pattern", {"q_proj": 3}),
        ("alpha_pattern", {"q_proj": 8}),
        ("use_rslora", True),
    ],
)
def test_readable_adapter_mismatch(tmp_path: Path, key: str, value: object) -> None:
    checkpoint = _checkpoint(tmp_path / "source")
    config_path = checkpoint / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config[key] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=key):
        _metadata(checkpoint)


def test_lineage_carries_two_hops_and_active_provenance(tmp_path: Path) -> None:
    first_checkpoint = _checkpoint(tmp_path / "first")
    first = _metadata(first_checkpoint)
    second_checkpoint = _checkpoint(tmp_path / "second", previous=first)
    source_provenance = second_checkpoint.parent / "provenance.json"
    active = source_provenance.with_name("provenance.resume.json")
    active.write_text(source_provenance.read_text())
    second = _metadata(second_checkpoint)
    assert second["source_provenance"] == str(active.resolve())
    assert [entry["source_checkpoint"] for entry in second["lineage"]] == [
        str(first_checkpoint.resolve()),
        str(second_checkpoint.resolve()),
    ]
    assert all("lineage" not in entry for entry in second["lineage"])


def test_source_provenance_identity_must_match(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "source")
    provenance_path = checkpoint.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["model_revision"] = "other"
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="revision"):
        _metadata(checkpoint)


def test_incomplete_checkpoint_refused(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "source")
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        _metadata(checkpoint)


def test_missing_adapter_tensor_refused(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "source")
    weights = get_peft_model_state_dict(_model())
    weights.pop(next(iter(weights)))
    save_file(weights, checkpoint / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="adapter tensor"):
        load_initial_adapter(
            _model(), checkpoint, model_name="synthetic-base", model_revision="synthetic-revision"
        )


def test_initial_loaded_adapter_is_served_at_step_zero(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(phases, "_cgroup_memory_states", lambda *_args: ())
    monkeypatch.setattr(
        phases.psutil, "virtual_memory", lambda: SimpleNamespace(total=1 << 30, available=1 << 30)
    )
    transport, server_state = server
    checkpoint = _checkpoint(tmp_path / "source")
    source_model = tiny_policy()
    base_path = tmp_path / "Qwen3.5-synthetic-base"
    source_model.config.save_pretrained(base_path)
    source_model.peft_config["default"].base_model_name_or_path = str(base_path)
    with torch.no_grad():
        for parameter in source_model.parameters():
            if parameter.requires_grad:
                parameter.fill_(0.375)
    source_model.save_pretrained(checkpoint)
    provenance_path = checkpoint.parent / "provenance.json"
    provenance_path.write_text(
        json.dumps({"flags": {"model": str(base_path)}, "model_revision": "synthetic-revision"})
    )
    model = tiny_policy()
    model.peft_config["default"].base_model_name_or_path = str(base_path)
    load_initial_adapter(
        model, checkpoint, model_name=str(base_path), model_revision="synthetic-revision"
    )
    optimizer = torch.optim.AdamW(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    trainer = ToyTrainer(model, optimizer, ToyState())
    backend = ToyBackend()
    controller = VLLMPhaseController(
        "http://server",
        backend,
        tmp_path / "served",
        training_device=torch.device("cpu"),
        host_memory_budget_bytes=1 << 30,
        host_memory_safety_margin_bytes=0,
        http_transport=transport,
    )
    controller.initialize(trainer)
    payload = next(payload for payload in server_state.payloads if "lora_path" in payload)
    served = load_file(str(Path(cast("str", payload["lora_path"])) / "adapter_model.safetensors"))
    expected = load_file(str(checkpoint / "adapter_model.safetensors"))
    assert "step-00000000" in backend.model_id
    assert trainer.state.global_step == 0
    assert optimizer.state == {}
    for name, tensor in expected.items():
        served_name = name.replace(
            "base_model.model.model.layers.", "base_model.model.model.language_model.layers."
        )
        assert torch.equal(served[served_name], tensor)
    controller.close()
