"""CPU storage-cycle checks with a real tiny Qwen3.5 PEFT policy."""

from __future__ import annotations

import weakref
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torchao.quantization import Int8Tensor, Int8WeightOnlyConfig, quantize_
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from reward_hacking.distill.memory import quantize_base_linear
from reward_hacking.distill.train import prepare_fp32_head
from reward_hacking.organism_rl.precision import assert_fp32_training_state
from reward_hacking.organism_rl.transfers import PolicyTransfers, _HostArena

if TYPE_CHECKING:
    from pathlib import Path


def tiny_policy(*, quantized: bool = False) -> Any:
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
    if quantized:
        quantize_(base_model, Int8WeightOnlyConfig(), filter_fn=quantize_base_linear)
    model: Any = get_peft_model(
        base_model,
        LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    prepare_fp32_head(model)
    model.register_buffer("phase_counter", torch.zeros(1))
    return model


@pytest.mark.parametrize("pin_memory", [False, True])
def test_real_int8_qwen_storage_cycles_preserve_wrappers_and_exact_bits(
    pin_memory: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_cuda(*args: object, **kwargs: object) -> None:
        raise AssertionError("CPU quantized transfers must not call CUDA")

    monkeypatch.setattr(torch.cuda, "cudart", forbidden_cuda)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden_cuda)
    model = tiny_policy(quantized=True)
    wrappers = [parameter for parameter in model.parameters() if isinstance(parameter, Int8Tensor)]
    assert wrappers
    leaves = [
        getattr(wrapper, name) for wrapper in wrappers for name in wrapper.__tensor_flatten__()[0]
    ]
    expected = [leaf.clone() for leaf in leaves]
    identities = {name: id(value) for name, value in model.named_parameters()}
    leaf_identities = [id(leaf) for leaf in leaves]
    optimizer = torch.optim.AdamW(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    transfers = PolicyTransfers(model, torch.device("cpu"), pin_memory=pin_memory)
    storage_tensors = [
        parameter for parameter in model.parameters() if not isinstance(parameter, Int8Tensor)
    ] + leaves
    expected_bytes = sum(
        {
            tensor.untyped_storage()._cdata: tensor.untyped_storage().nbytes()
            for tensor in storage_tensors
        }.values()
    )
    inputs = torch.tensor([[1, 2, 3, 4]])
    for _ in range(2):
        report = transfers.offload(optimizer)
        assert (
            report.frozen_bytes
            + sum(
                parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            == expected_bytes
        )
        assert transfers.frozen_host_storage_ids
        mapping_refs = [weakref.ref(arena._mapping) for arena in transfers._frozen_arenas]
        transfers.restore(optimizer)
        assert all(reference() is None for reference in mapping_refs)
        assert not transfers.frozen_host_storage_ids
        assert {name: id(value) for name, value in model.named_parameters()} == identities
        assert [
            id(getattr(wrapper, name))
            for wrapper in wrappers
            for name in wrapper.__tensor_flatten__()[0]
        ] == leaf_identities
        for leaf, original in zip(leaves, expected, strict=True):
            torch.testing.assert_close(leaf, original, rtol=0, atol=0)
        optimizer.zero_grad()
        model(input_ids=inputs, labels=inputs, use_cache=False).loss.backward()
        optimizer.step()


def test_int8_wrapper_refresh_updates_outer_device_without_replacing_parameter() -> None:
    model = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
    quantize_(model, Int8WeightOnlyConfig())
    model.requires_grad_(False)
    transfers = PolicyTransfers(model, torch.device("cpu"))
    original: Any = model.weight
    identity = id(original)
    names, metadata = original.__tensor_flatten__()
    stale = Int8Tensor.__tensor_unflatten__(
        {name: getattr(original, name).to("meta") for name in names},
        metadata,
        original.shape,
        original.stride(),
    )
    for name, value in original.__dict__.items():
        setattr(stale, name, value)
    torch.utils.swap_tensors(original, stale)
    assert original.device.type == "meta"
    assert original.qdata.device.type == "cpu"
    transfers._refresh_wrappers()
    assert id(model.weight) == identity
    assert isinstance(model.weight, torch.nn.Parameter)
    assert original.device.type == "cpu"
    assert original.qdata.device.type == "cpu"


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
    assert all("_organism_fp32_head" not in name for name, _ in model.named_buffers())
    assert first.frozen_bytes == sum(
        value.numel() * value.element_size() for value in frozen.values()
    )
    inputs = torch.tensor([[1, 2, 3, 4]])
    for _ in range(2):
        transfers.restore(optimizer)
        assert model.get_output_embeddings()._organism_fp32_head_weight.dtype == torch.float32
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
        assert report.frozen_bytes == first.frozen_bytes
        assert report.mutable_bytes > 0
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


def test_restore_rebinds_staged_values_and_shared_storage_views() -> None:
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


def test_frozen_fp32_head_buffer_staged_and_mutable_buffer_refreshed() -> None:
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
    transfers.restore(None)
    model.counter.add_(7)
    second = transfers.offload(None)
    assert second.frozen_bytes == first.frozen_bytes
    assert second.mutable_bytes == model.counter.numel() * model.counter.element_size()
    transfers.restore(None)
    assert model.counter.item() == 7
    assert model.cached_head.dtype == torch.float32
    assert not transfers.frozen_host_storage_ids


def test_transfers_use_one_arena_per_dtype_for_many_storages_and_preserve_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Module()
    for index in range(40):
        model.register_parameter(
            f"weight_{index}",
            torch.nn.Parameter(torch.full((3,), float(index)), requires_grad=False),
        )
    model.register_buffer("counter", torch.tensor([123], dtype=torch.int64))
    frombuffer = torch.frombuffer
    host_allocations: list[int] = []

    def counted_frombuffer(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = frombuffer(*args, **kwargs)
        host_allocations.append(result.numel())
        return result

    monkeypatch.setattr(torch, "frombuffer", counted_frombuffer)
    transfers = PolicyTransfers(model, torch.device("cpu"))
    first = transfers.offload(None)
    assert len(host_allocations) == 1
    assert first.moved_bytes == 40 * 3 * 4 + 8
    assert sum(transfers.host_byte_breakdown.values()) == transfers.host_bytes
    assert transfers.host_byte_breakdown["mutable.buffers"] == 8
    for _ in range(2):
        transfers.restore(None)
        for index, parameter in enumerate(model.parameters()):
            torch.testing.assert_close(parameter, torch.full((3,), float(index)))
        assert model.get_buffer("counter").item() == 123
        transfers.offload(None)
    assert len(host_allocations) == 3


def test_planned_host_bytes_counts_shared_storage_once_without_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model: Any = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(4), requires_grad=False))
    model.register_parameter(
        "alias", torch.nn.Parameter(model.weight.detach(), requires_grad=False)
    )
    model.register_buffer("view", model.weight.detach()[1:])
    transfers = PolicyTransfers(model, torch.device("cpu"))

    def no_allocation(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("budget planning must precede host allocation")

    monkeypatch.setattr(torch, "empty", no_allocation)
    assert transfers.planned_host_bytes(None) == 16


@pytest.mark.parametrize("pin_memory", [False, True])
def test_qwen_transient_frozen_arena_released_after_each_restore(pin_memory: bool) -> None:
    model = tiny_policy()
    transfers = PolicyTransfers(model, torch.device("cpu"), pin_memory=pin_memory)
    expected = {name: value.detach().clone() for name, value in model.named_parameters()}
    for _ in range(3):
        report = transfers.offload(None)
        assert report.frozen_bytes > 0
        arenas = transfers._frozen_arenas
        assert arenas
        arena_ref = weakref.ref(arenas[0])
        buffer_ref = weakref.ref(arenas[0].buffer)
        mapping_ref = weakref.ref(arenas[0]._mapping)
        del arenas
        transfers.restore(None)
        assert mapping_ref() is None
        assert not transfers._frozen_arenas
        assert arena_ref() is None
        assert buffer_ref() is None
        assert not transfers.frozen_host_storage_ids
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_registered_arena_unregisters_and_unmaps_without_pinned_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def register(pointer: int, size: int, flags: int) -> int:
        assert size == 4096
        assert flags == 0
        calls.append(("register", pointer))
        return 0

    def unregister(pointer: int) -> int:
        calls.append(("unregister", pointer))
        return 0

    runtime = SimpleNamespace(
        cudaHostRegister=register,
        cudaHostUnregister=unregister,
        cudaError=SimpleNamespace(success=0),
    )
    monkeypatch.setattr(torch.cuda, "cudart", lambda: runtime)
    arena = _HostArena(4096, pin_memory=True)
    assert arena.buffer is not None
    mapping_ref = weakref.ref(arena._mapping)
    pointer = arena.buffer.data_ptr()
    buffer_ref = weakref.ref(arena.buffer)
    arena.release()
    assert calls == [("register", pointer), ("unregister", pointer)]
    assert arena._mapping is None
    assert mapping_ref() is None
    assert buffer_ref() is None


def test_restore_keeps_caller_owned_host_alias_valid_until_released() -> None:
    model = tiny_policy()
    transfers = PolicyTransfers(model, torch.device("cpu"))
    transfers.offload(None)
    parameter = next(parameter for parameter in model.parameters() if not parameter.requires_grad)
    alias = parameter.detach()
    expected = alias.clone()
    mapping_ref = weakref.ref(transfers._frozen_arenas[0]._mapping)
    transfers.restore(None)
    torch.testing.assert_close(alias, expected, rtol=0, atol=0)
    assert mapping_ref() is not None
    del alias
    assert mapping_ref() is None
