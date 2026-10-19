"""CPU checks for explicit distillation memory options and checkpoint recomputation."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from liger_kernel.ops.tiled_mlp import apply_tiled_mlp
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, Qwen2Config
from trl.models.utils import disable_gradient_checkpointing

from reward_hacking.distill import memory


def test_memory_options_preserve_defaults() -> None:
    options = memory.MemoryOptions()
    assert options.deltanet_conv_kernel == "default"
    assert not options.liger
    assert not options.tiled_mlp
    assert not options.offload_checkpoints
    assert options.base_quantization == "none"
    assert not options.skip_dead_mlp_recompute


def test_host_memory_refuses_insufficient_capacity() -> None:
    with pytest.raises(RuntimeError, match=r"host memory.*required"):
        memory.check_host_memory(4096, available_bytes=4095)
    memory.check_host_memory(4096, available_bytes=4096)


def test_cgroup_v2_memory_available_counts_inactive_file_as_reclaimable(tmp_path: Path) -> None:
    cgroup_mount = tmp_path / "cgroup mount"
    process_cgroup = cgroup_mount / "job"
    process_cgroup.mkdir(parents=True)
    (process_cgroup / "memory.max").write_text("8192\n")
    (process_cgroup / "memory.current").write_text("7168\n")
    (process_cgroup / "memory.stat").write_text("inactive_file 2048\nactive_file 4096\n")
    cgroup_file = tmp_path / "proc-self-cgroup"
    cgroup_file.write_text("0::/delegated/job\n")
    mountinfo_file = tmp_path / "proc-self-mountinfo"
    mountinfo_file.write_text(
        f"31 25 0:27 /delegated {str(cgroup_mount).replace(' ', r'\040')} "
        "rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n"
    )

    assert (
        memory.cgroup_v2_memory_available_bytes(
            cgroup_file=cgroup_file,
            mountinfo_file=mountinfo_file,
        )
        == 3072
    )


def test_cgroup_v2_memory_available_requires_memory_stat(tmp_path: Path) -> None:
    cgroup_mount = tmp_path / "cgroup"
    process_cgroup = cgroup_mount / "scope"
    process_cgroup.mkdir(parents=True)
    (process_cgroup / "memory.max").write_text("8192\n")
    (process_cgroup / "memory.current").write_text("3072\n")
    cgroup_file = tmp_path / "proc-self-cgroup"
    cgroup_file.write_text("0::/scope\n")
    mountinfo_file = tmp_path / "proc-self-mountinfo"
    mountinfo_file.write_text(f"31 25 0:27 / {cgroup_mount} rw - cgroup2 cgroup rw\n")

    with pytest.raises(RuntimeError, match=r"memory\.stat"):
        memory.cgroup_v2_memory_available_bytes(
            cgroup_file=cgroup_file,
            mountinfo_file=mountinfo_file,
        )


def test_cgroup_v2_unlimited_memory_returns_none(tmp_path: Path) -> None:
    cgroup_mount = tmp_path / "cgroup"
    process_cgroup = cgroup_mount / "scope"
    process_cgroup.mkdir(parents=True)
    (process_cgroup / "memory.max").write_text("max\n")
    (process_cgroup / "memory.current").write_text("3072\n")
    cgroup_file = tmp_path / "proc-self-cgroup"
    cgroup_file.write_text("0::/scope\n")
    mountinfo_file = tmp_path / "proc-self-mountinfo"
    mountinfo_file.write_text(f"31 25 0:27 / {cgroup_mount} rw - cgroup2 cgroup rw\n")

    assert (
        memory.cgroup_v2_memory_available_bytes(
            cgroup_file=cgroup_file,
            mountinfo_file=mountinfo_file,
        )
        is None
    )


def test_host_memory_uses_lower_process_cgroup_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=9000))
    monkeypatch.setattr(memory, "cgroup_v2_memory_available_bytes", lambda: 4000)

    with pytest.raises(RuntimeError, match=r"available 4000 bytes"):
        memory.check_host_memory(5000)


def test_host_memory_uses_system_available_when_cgroup_is_unlimited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=4096))
    monkeypatch.setattr(memory, "cgroup_v2_memory_available_bytes", lambda: None)

    memory.check_host_memory(4096)


def test_checkpoint_offload_refuses_cpu_before_allocating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        memory.prepare_checkpoint_offload(torch.nn.Linear(2, 2), 12)


class PageableSlot:
    """Use CPU storage to check the recomputation arithmetic without CUDA registration."""

    def __init__(self) -> None:
        self.buffer = torch.empty(128)

    def view_for(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.buffer[: hidden_states.numel()].view(hidden_states.shape)


def test_offloaded_recompute_preserves_gradients_and_rng() -> None:
    layers = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.Dropout(0.25), torch.nn.Linear(4, 4)
    )
    inputs = torch.randn(2, 4, requires_grad=True)

    def gradients(offload: bool) -> list[torch.Tensor]:
        torch.manual_seed(19)
        layers.zero_grad(set_to_none=True)
        inputs.grad = None
        output: Any = (
            memory.OffloadedLayerFunction.apply(layers, inputs, PageableSlot())
            if offload
            else checkpoint(layers, inputs, use_reentrant=False)
        )
        output.square().sum().backward()
        assert inputs.grad is not None
        result: list[torch.Tensor] = [inputs.grad.clone()]
        for parameter in layers.parameters():
            assert parameter.grad is not None
            result.append(parameter.grad.clone())
        return result

    reference = gradients(False)
    for _ in range(2):
        candidate = gradients(True)
        for expected, actual in zip(reference, candidate, strict=True):
            torch.testing.assert_close(actual, expected)


def test_offloaded_checkpoint_refuses_inputs_without_gradients() -> None:
    layer = torch.nn.Linear(4, 4)
    inputs = torch.ones(2, 4)
    with pytest.raises(RuntimeError, match="enable_input_require_grads"):
        memory.OffloadedLayerFunction.apply(layer, inputs, PageableSlot())


def test_quantization_filter_keeps_head_and_non_linears() -> None:
    linear = torch.nn.Linear(4, 4)
    assert memory.quantize_base_linear(linear, "model.layers.0.mlp.gate_proj")
    assert not memory.quantize_base_linear(linear, "lm_head")
    assert not memory.quantize_base_linear(torch.nn.Embedding(4, 4), "model.embed_tokens")


def test_nf4_configuration_and_warning(caplog: pytest.LogCaptureFixture) -> None:
    config = memory.nf4_config()
    assert config.load_in_4bit
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_compute_dtype == torch.bfloat16
    assert config.llm_int8_skip_modules == ["lm_head"]
    assert "0.034" in caplog.text


def test_offloaded_tiny_lora_gradients_and_detach_sabotage() -> None:
    torch.manual_seed(13)
    model = cast(
        "Any",
        get_peft_model(
            AutoModelForCausalLM.from_config(
                Qwen2Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                )
            ),
            LoraConfig(
                task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]
            ),
        ),
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    layers = model.get_base_model().base_model.layers
    originals = [layer._gradient_checkpointing_func for layer in layers]
    slots = [PageableSlot() for _ in layers]
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6]])

    def gradients(*, offload: bool, sabotage: bool = False) -> dict[str, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        for index, layer in enumerate(layers):

            def offloaded(
                function: Any, hidden_states: torch.Tensor, *, slot: PageableSlot = slots[index]
            ) -> Any:
                output = memory.OffloadedLayerFunction.apply(function, hidden_states, slot)
                return output.detach().requires_grad_() if sabotage else output

            layer._gradient_checkpointing_func = offloaded if offload else originals[index]
        model(input_ids=inputs, labels=inputs, use_cache=False).loss.backward()
        return {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }

    reference = gradients(offload=False)
    assert reference
    for _ in range(2):
        actual = gradients(offload=True)
        assert actual.keys() == reference.keys()
        for name, expected in reference.items():
            torch.testing.assert_close(actual[name], expected)
    assert gradients(offload=True, sabotage=True).keys() != reference.keys()


def test_torchao_option_wires_config_and_head_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, object]] = []
    linear = torch.nn.Linear(4, 4)

    class QuantizationModule:
        Int8WeightOnlyConfig = object

        @staticmethod
        def quantize_(model: object, config: object, *, filter_fn: object) -> None:
            assert model is linear
            calls.append((config, filter_fn))

    monkeypatch.setattr(memory.importlib, "import_module", lambda _: QuantizationModule)
    monkeypatch.setattr(memory.AutoModelForCausalLM, "from_pretrained", lambda *_, **__: linear)
    monkeypatch.setattr(torch.nn.Linear, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    assert memory.load_base_model("synthetic", "torchao-int8") is linear
    assert len(calls) == 1
    assert calls[0][1] is memory.quantize_base_linear


def test_host_slots_are_presized_and_checkpoint_install_waits_for_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = cast(
        "Any",
        get_peft_model(
            AutoModelForCausalLM.from_config(
                Qwen2Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                )
            ),
            LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["q_proj"]),
        ),
    )
    allocated: list[int] = []

    class RegisteredTestSlot(PageableSlot):
        def __init__(self, elements: int, dtype: torch.dtype) -> None:
            allocated.append(elements)
            self.buffer = torch.empty(elements, dtype=dtype)
            self.released = False

        @property
        def registered_bytes(self) -> int:
            return self.buffer.numel() * self.buffer.element_size()

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(memory, "HostSlot", RegisteredTestSlot)
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=1))
    with pytest.raises(RuntimeError, match="insufficient host memory"):
        memory.prepare_checkpoint_offload(model, 7)
    assert not allocated
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=4096))
    original_enable = model.gradient_checkpointing_enable
    offload = memory.prepare_checkpoint_offload(model, 7)
    assert allocated == [7 * 16, 7 * 16]
    layers = model.get_base_model().base_model.layers
    assert not layers[0].gradient_checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert layers[0].gradient_checkpointing
    assert layers[0]._gradient_checkpointing_func.__name__ == "checkpoint_with_offload"
    with disable_gradient_checkpointing(model, {"use_reentrant": False}):
        assert not layers[0].gradient_checkpointing
    assert layers[0].gradient_checkpointing
    assert layers[0]._gradient_checkpointing_func.__name__ == "checkpoint_with_offload"
    assert offload.registered_bytes == 2 * 7 * 16 * 4
    offload.close()
    assert model.gradient_checkpointing_enable == original_enable
    assert layers[0]._gradient_checkpointing_func is offload.original_checkpoint_functions[0]
    assert all(cast("RegisteredTestSlot", slot).released for slot in offload.slots)


def test_kernel_installation_order_and_liger_loss_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    class LigerPatch:
        @staticmethod
        def apply_liger_kernel_to_qwen3_5(**kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(memory, "bridge_decode_kernel", lambda: calls.append("bridge"))
    monkeypatch.setattr(memory.importlib, "import_module", lambda _: LigerPatch)
    memory.install_before_load(memory.MemoryOptions(deltanet_conv_kernel="fla", liger=True))
    assert calls == ["bridge", {"fused_linear_cross_entropy": False}]


def test_nf4_option_wires_gpu_load_without_moving_quantized_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    model = torch.nn.Linear(4, 4)

    def load_model(model_id: str, **kwargs: Any) -> torch.nn.Module:
        assert model_id == "synthetic"
        calls.append(kwargs)
        return model

    monkeypatch.setattr(memory.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert memory.load_base_model("synthetic", "nf4") is model
    assert calls[0]["dtype"] == torch.bfloat16
    assert calls[0]["device_map"] == {"": 0}
    assert calls[0]["quantization_config"].bnb_4bit_quant_type == "nf4"


def test_default_model_load_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    model = torch.nn.Linear(4, 4)

    def load_model(model_id: str, **kwargs: Any) -> torch.nn.Module:
        assert model_id == "synthetic"
        calls.append(kwargs)
        return model

    monkeypatch.setattr(memory.AutoModelForCausalLM, "from_pretrained", load_model)
    assert memory.load_base_model("synthetic", "none") is model
    assert calls == [{"dtype": torch.bfloat16}]
    calls.clear()
    assert memory.load_base_model("synthetic", "none", revision="resolved-revision") is model
    assert calls == [{"dtype": torch.bfloat16, "revision": "resolved-revision"}]


def test_tiled_mlp_passes_only_trainable_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(4, 4)
    model.weight.requires_grad_(False)
    inputs = torch.ones(1, 4)
    calls: list[dict[str, Any]] = []

    class Operations:
        @staticmethod
        def apply_tiled_mlp(**kwargs: Any) -> torch.Tensor:
            calls.append(kwargs)
            return kwargs["x"]

    monkeypatch.setattr(memory.importlib, "import_module", lambda _: Operations)
    assert memory.tiled_mlp_forward(model, inputs) is inputs
    assert callable(calls[0]["fn"])
    assert calls[0]["mlp_module"] is model
    assert calls[0]["num_shards"] is None
    assert len(calls[0]["compute_params"]) == 1
    assert calls[0]["compute_params"][0] is model.bias


class TinyTiledMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 7, bias=False)
        self.up_proj = torch.nn.Linear(4, 7, bias=False)
        self.down_proj = torch.nn.Linear(7, 4, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        activated_gate = torch.nn.functional.silu(self.gate_proj(hidden_states))
        return self.down_proj(activated_gate * self.up_proj(hidden_states))


class TinyResidualLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyTiledMLP()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(hidden_states)


class TinyLayerStack(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyResidualLayer()])


class TinyOffloadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = TinyLayerStack()

    def get_base_model(self) -> "TinyOffloadModel":
        return self


@pytest.mark.parametrize("fill_dead_output", [False, True])
def test_dead_mlp_recompute_skip_preserves_loss_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
    fill_dead_output: bool,
) -> None:
    calls: list[torch.Tensor] = []
    if fill_dead_output:
        monkeypatch.setattr(torch, "empty_like", lambda inputs: torch.full_like(inputs, 123.0))

    def compute_mlp(module: TinyTiledMLP, hidden_states: torch.Tensor) -> torch.Tensor:
        calls.append(hidden_states)
        activated_gate = torch.nn.functional.silu(module.gate_proj(hidden_states))
        return module.down_proj(activated_gate * module.up_proj(hidden_states))

    monkeypatch.setattr(
        memory.importlib,
        "import_module",
        lambda _: SimpleNamespace(apply_tiled_mlp=apply_tiled_mlp),
    )
    monkeypatch.setattr(memory, "tiled_mlp_compute", compute_mlp)
    torch.manual_seed(29)
    initial_model = TinyOffloadModel()
    initial_state = initial_model.state_dict()
    inputs = torch.randn(1, 10, 4)

    def loss_and_gradients(
        skip_dead_mlp_recompute: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        model = TinyOffloadModel()
        model.load_state_dict(initial_state)
        memory.install_tiled_mlp(model, skip_dead_mlp_recompute=skip_dead_mlp_recompute)
        hidden_states = inputs.clone().requires_grad_()
        output = memory.OffloadedLayerFunction.apply(
            model.base_model.layers[0], hidden_states, PageableSlot()
        )
        loss = output.square().sum()
        loss.backward()
        assert hidden_states.grad is not None
        gradients = [hidden_states.grad.clone()]
        gradients.extend(
            parameter.grad.clone() for parameter in model.parameters() if parameter.grad is not None
        )
        return loss.detach(), gradients

    calls.clear()
    reference_loss, reference_gradients = loss_and_gradients(False)
    reference_compute_calls = len(calls)
    calls.clear()
    skipped_loss, skipped_gradients = loss_and_gradients(True)
    skipped_compute_calls = len(calls)

    assert torch.equal(skipped_loss, reference_loss)
    assert len(skipped_gradients) == len(reference_gradients)
    for expected, actual in zip(reference_gradients, skipped_gradients, strict=True):
        assert torch.equal(actual, expected)
    assert skipped_compute_calls < reference_compute_calls


def test_dead_mlp_skip_refuses_stochastic_mlp() -> None:
    model = TinyOffloadModel()
    model.get_submodule("base_model.layers.0.mlp").add_module("dropout", torch.nn.Dropout(0.1))
    with pytest.raises(ValueError, match="dropout"):
        memory.install_tiled_mlp(model, skip_dead_mlp_recompute=True)


@pytest.mark.parametrize("offload", [False, True])
def test_tiled_bfloat16_autocast_preserves_float32_adapter_gradients(
    monkeypatch: pytest.MonkeyPatch, offload: bool
) -> None:
    torch.manual_seed(31)
    model = AutoModelForCausalLM.from_config(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    ).to(torch.bfloat16)
    adapted = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, target_modules=["gate_proj", "up_proj", "down_proj"]),
    )
    mlp = cast("Any", adapted).get_base_model().model.layers[0].mlp
    mlp.skip_dead_mlp_recompute = False
    parameters = [parameter for parameter in mlp.parameters() if parameter.requires_grad]
    assert parameters
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    for parameter in parameters:
        torch.nn.init.normal_(parameter, std=0.02)
    inputs = torch.randn(1, 40, 16, dtype=torch.bfloat16)
    compute_autocast: list[bool] = []

    def compute_mlp(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        compute_autocast.append(torch.is_autocast_enabled("cpu"))
        return module.down_proj(
            torch.nn.functional.silu(module.gate_proj(hidden_states))
            * module.up_proj(hidden_states)
        )

    monkeypatch.setattr(memory, "tiled_mlp_compute", compute_mlp)
    monkeypatch.setattr(
        memory.importlib,
        "import_module",
        lambda _: SimpleNamespace(apply_tiled_mlp=apply_tiled_mlp),
    )

    def loss_and_gradients(tiled: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        mlp.zero_grad(set_to_none=True)
        hidden_states = inputs.clone().requires_grad_()

        def run_layer(hidden_states: torch.Tensor) -> torch.Tensor:
            return (
                memory.tiled_mlp_forward(mlp, hidden_states)
                if tiled
                else compute_mlp(mlp, hidden_states)
            )

        slot = PageableSlot()
        slot.buffer = torch.empty_like(hidden_states).flatten()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = (
                memory.OffloadedLayerFunction.apply(run_layer, hidden_states, slot)
                if offload
                else run_layer(hidden_states)
            )
            assert output.dtype == torch.bfloat16
            loss = output.float().square().sum()
        loss.backward()
        assert hidden_states.grad is not None
        gradients = [hidden_states.grad.clone()]
        assert all(
            parameter.grad is not None and parameter.grad.dtype == torch.float32
            for parameter in parameters
        )
        gradients.extend(cast("torch.Tensor", parameter.grad).clone() for parameter in parameters)
        return loss.detach(), gradients

    expected_loss, expected_gradients = loss_and_gradients(False)
    compute_autocast.clear()
    actual_loss, actual_gradients = loss_and_gradients(True)
    assert compute_autocast
    assert all(compute_autocast)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.0001)
