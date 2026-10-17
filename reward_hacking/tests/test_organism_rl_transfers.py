"""CPU storage-cycle checks with a real tiny Qwen3.5 PEFT policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from reward_hacking.distill.train import prepare_fp32_head
from reward_hacking.organism_rl.precision import assert_fp32_training_state
from reward_hacking.organism_rl.transfers import PolicyTransfers

if TYPE_CHECKING:
    from pathlib import Path


def tiny_policy() -> Any:
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
    base_model: Any = Qwen3_5ForCausalLM(config)
    base_model.to(dtype=torch.bfloat16)
    model: Any = get_peft_model(
        base_model,
        LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    prepare_fp32_head(model)
    model.register_buffer("phase_counter", torch.zeros(1))
    return model


def test_qwen_lora_optimizer_cycles_and_cpu_checkpoint(tmp_path: Path) -> None:
    model: Any = tiny_policy()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    transfers = PolicyTransfers(model, torch.device("cpu"))
    identities = {name: id(value) for name, value in model.named_parameters()}
    frozen = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    dtypes = {name: value.dtype for name, value in model.named_parameters()}
    assert_fp32_training_state(model, optimizer)
    first = transfers.offload(optimizer)
    assert first.frozen_bytes > 0
    first_host_ids = transfers.frozen_host_storage_ids
    inputs = torch.tensor([[1, 2, 3, 4]])
    for _ in range(2):
        transfers.restore(optimizer)
        optimizer.zero_grad()
        loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
        loss.backward()
        optimizer.step()
        assert_fp32_training_state(model, optimizer)
        expected = {name: value.detach().clone() for name, value in model.named_parameters()}
        expected_state = {
            id(parameter): {name: value.clone() for name, value in state.items()}
            for parameter, state in optimizer.state.items()
        }
        state_identities = {
            id(value) for state in optimizer.state.values() for value in state.values()
        }
        gradient_identities = {
            id(parameter.grad) for parameter in model.parameters() if parameter.grad is not None
        }
        report = transfers.offload(optimizer)
        assert report.frozen_bytes == 0
        assert report.mutable_bytes > 0
        assert transfers.frozen_host_storage_ids == first_host_ids
        assert {name: id(value) for name, value in model.named_parameters()} == identities
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            tmp_path / "checkpoint.pt",
        )
        saved = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
        assert all(value.device.type == "cpu" for value in saved["model"].values())
        transfers.restore(optimizer)
        assert_fp32_training_state(model, optimizer)
        assert {
            id(value) for state in optimizer.state.values() for value in state.values()
        } == state_identities
        assert {
            id(parameter.grad) for parameter in model.parameters() if parameter.grad is not None
        } == gradient_identities
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
            assert parameter.dtype == dtypes[name]
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        for parameter, state in optimizer.state.items():
            for name, value in state.items():
                torch.testing.assert_close(
                    value, expected_state[id(parameter)][name], rtol=0, atol=0
                )


def test_restore_rebinds_cached_values_and_shared_storage_views() -> None:
    model: Any = torch.nn.Module()
    model.register_parameter(
        "weight", torch.nn.Parameter(torch.arange(12.0).reshape(3, 4), requires_grad=False)
    )
    model.register_buffer("view", model.weight.detach()[:, 1:3])
    transfers = PolicyTransfers(model, torch.device("cpu"))
    identities = id(model.weight), id(model.view)
    transfers.offload(None)
    expected = model.weight.detach().clone()
    model.weight.data = torch.zeros_like(model.weight)
    transfers.restore(None)
    torch.testing.assert_close(model.weight, expected)
    assert (id(model.weight), id(model.view)) == identities
    assert model.weight.untyped_storage().data_ptr() == model.view.untyped_storage().data_ptr()
    assert model.view.stride() == (4, 1)
    assert model.view.storage_offset() == 1


def test_tensor_subclass_refused_before_model_mutation() -> None:
    class QuantizedTensor(torch.Tensor):
        pass

    model = torch.nn.Linear(2, 2)
    model.weight = torch.nn.Parameter(
        model.weight.detach().as_subclass(QuantizedTensor), requires_grad=False
    )
    identity = id(model.weight)
    with pytest.raises(TypeError, match=r"subclass.*TorchAO"):
        PolicyTransfers(model, torch.device("cpu"))
    assert id(model.weight) == identity


def test_frozen_fp32_head_buffer_cached_and_mutable_buffer_refreshed() -> None:
    model: Any = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.register_buffer("cached_head", model.weight.float().detach().clone(), persistent=False)
    model.register_buffer("counter", torch.zeros(1, dtype=torch.int64))
    model._phase_frozen_buffer_names = {"cached_head"}
    transfers = PolicyTransfers(model, torch.device("cpu"))
    first = transfers.offload(None)
    assert first.frozen_bytes == sum(
        tensor.numel() * tensor.element_size()
        for tensor in [model.weight, model.bias, model.cached_head]
    )
    host_ids = transfers.frozen_host_storage_ids
    transfers.restore(None)
    model.counter.add_(7)
    second = transfers.offload(None)
    assert second.frozen_bytes == 0
    assert second.mutable_bytes == model.counter.numel() * model.counter.element_size()
    transfers.restore(None)
    assert model.counter.item() == 7
    assert model.cached_head.dtype == torch.float32
    assert transfers.frozen_host_storage_ids == host_ids
