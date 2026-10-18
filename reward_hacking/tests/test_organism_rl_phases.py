"""Exercise HTTP protocol boundaries with real adapters and a CPU optimizer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import httpx2
import pytest
import torch
from safetensors.torch import load_file, save_file
from torchao.quantization import Int8Tensor

from grpo import gpu_memory
from reward_hacking.organism_rl import phases
from reward_hacking.organism_rl.phases import (
    VLLMPhaseController,
    check_host_memory_budget,
    check_pinned_and_sleeping_weight_budget,
    move_optimizer_state,
    resolve_host_memory_budget_bytes,
)
from reward_hacking.tests.test_organism_rl_transfers import tiny_policy


def _int8_policy_with_local_config(directory: Path) -> Any:
    model = tiny_policy(quantized=True)
    base_directory = directory / "Qwen3.5-synthetic-base"
    model.config.save_pretrained(base_directory)
    model.peft_config["default"].base_model_name_or_path = str(base_directory)
    return model


@pytest.mark.parametrize("frozen_host_memory", ["pinned", "pageable", "persistent"])
def test_real_int8_qwen_phase_cycles_and_host_budget(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_host_memory: Literal["pinned", "pageable", "persistent"],
) -> None:
    transport, _state = server
    monkeypatch.setattr(phases, "_cgroup_memory_states", lambda *_args: ())
    monkeypatch.setattr(
        phases.psutil, "virtual_memory", lambda: SimpleNamespace(total=1 << 30, available=1 << 30)
    )
    model = _int8_policy_with_local_config(tmp_path)
    parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    wrappers = [parameter for parameter in model.parameters() if isinstance(parameter, Int8Tensor)]
    leaves = [
        getattr(wrapper, name) for wrapper in wrappers for name in wrapper.__tensor_flatten__()[0]
    ]
    expected = [leaf.clone() for leaf in leaves]
    tensors = (
        [parameter for parameter in model.parameters() if not isinstance(parameter, Int8Tensor)]
        + leaves
        + [buffer for name, buffer in model.named_buffers() if "_organism_fp32_head" not in name]
    )
    expected_host_bytes = sum(
        {
            tensor.untyped_storage().data_ptr(): ((tensor.untyped_storage().nbytes() + 7) // 8) * 8
            for tensor in tensors
        }.values()
    )
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        host_memory_budget_bytes=1 << 30,
        host_memory_safety_margin_bytes=0,
        sleeping_weight_bytes=20,
        frozen_host_memory=frozen_host_memory,
        http_transport=transport,
    )
    optimizer = torch.optim.AdamW(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    trainer = ToyTrainer(model, optimizer, ToyState())
    _, _, planned = _record_phase_host_budget_calls(monkeypatch, controller)
    controller.initialize(trainer)
    assert planned[0] == (expected_host_bytes, 0)
    assert controller._transfers is not None
    assert controller._transfers.host_bytes == expected_host_bytes
    controller.before_training(trainer)
    assert bool(controller._transfers.frozen_host_storage_ids) == (
        frozen_host_memory == "persistent"
    )
    inputs = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=inputs, labels=inputs, use_cache=False).loss.backward()
    optimizer.step()
    trainer.state.global_step = 1
    controller.after_optimizer_step(trainer)
    controller.before_training(trainer)
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameters
    for leaf, original in zip(leaves, expected, strict=True):
        torch.testing.assert_close(leaf, original, rtol=0, atol=0)
    controller.close()


@pytest.mark.parametrize("frozen_host_memory", ["pinned", "persistent"])
def test_int8_host_budget_refusal_precedes_any_arena_allocation(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_host_memory: str,
) -> None:
    transport, _state = server
    monkeypatch.setattr(phases, "_cgroup_memory_states", lambda *_args: ())
    monkeypatch.setattr(
        phases.psutil, "virtual_memory", lambda: SimpleNamespace(total=1 << 30, available=1 << 30)
    )
    model = _int8_policy_with_local_config(tmp_path)
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}

    def forbidden_allocation(*args: object, **kwargs: object) -> None:
        raise AssertionError("int8 budget refusal must precede arena allocation")

    monkeypatch.setattr(phases.PolicyTransfers, "_collect_groups", forbidden_allocation)
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        host_memory_budget_bytes=1,
        host_memory_safety_margin_bytes=0,
        frozen_host_memory=frozen_host_memory,
        http_transport=transport,
    )
    with pytest.raises(RuntimeError, match="budget"):
        controller.initialize(ToyTrainer(model, None, ToyState()))
    assert controller._transfers is not None
    assert controller._transfers.host_bytes == 0
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == identities
    controller.close()


def test_persistent_startup_reserves_sleeping_weights_and_environment_margin(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, _state = server
    monkeypatch.setattr(phases, "_cgroup_memory_states", lambda *_args: ())
    monkeypatch.setattr(
        phases.psutil, "virtual_memory", lambda: SimpleNamespace(total=4096, available=4096)
    )
    monkeypatch.setenv("ORGANISM_HOST_MEMORY_SAFETY_MARGIN_BYTES", "10")
    model = ToyAdapter()
    model.weight.requires_grad_(False)

    def forbidden_allocation(*args: object, **kwargs: object) -> None:
        raise AssertionError("persistent startup refusal must precede arena allocation")

    monkeypatch.setattr(phases.PolicyTransfers, "_collect_groups", forbidden_allocation)
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        host_memory_budget_bytes=50,
        sleeping_weight_bytes=20,
        frozen_host_memory="persistent",
        http_transport=transport,
    )
    # 24 policy bytes + 20 sleeping bytes exceed the net budget of 40.
    with pytest.raises(RuntimeError, match=r"sleeping_weight_bytes=20.*budget_bytes=40"):
        controller.initialize(ToyTrainer(model, None, ToyState()))
    assert controller._transfers is not None
    assert controller._transfers.host_bytes == 0
    controller.close()


class ToyAdapter(torch.nn.Linear):
    def __init__(self, in_features: int = 2, out_features: int = 2) -> None:
        super().__init__(in_features, out_features)
        self.target_modules = ["q_proj", "v_proj"]

    def save_pretrained(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                    "target_modules": self.target_modules,
                }
            ),
            encoding="utf-8",
        )
        save_file(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": self.weight.detach().cpu()
            },
            str(path / "adapter_model.safetensors"),
        )


@dataclass
class ToyState:
    global_step: int = 0


@dataclass
class ToyTrainer:
    model: Any
    optimizer: torch.optim.Optimizer | None
    state: ToyState


@dataclass
class ToyBackend:
    model_id: str = "base"


@dataclass
class ServerState:
    sleeping: bool
    adapters: set[str]
    requests: list[str]
    payloads: list[dict[str, object]]
    hide_adapter: bool = False
    reject_sleep: bool = False
    loaded_adapters: set[str] = field(default_factory=set)


@pytest.fixture
def server() -> tuple[httpx2.MockTransport, ServerState]:
    state = ServerState(False, set(), [], [])

    def handle(request: httpx2.Request) -> httpx2.Response:
        path = request.url.raw_path.decode()
        if request.method == "GET":
            content = (
                {"is_sleeping": state.sleeping}
                if path == "/is_sleeping"
                else {"data": [{"id": name} for name in state.adapters if not state.hide_adapter]}
            )
            return httpx2.Response(200, json=content)
        state.requests.append(path)
        body = json.loads(request.content) if request.content else {}
        state.payloads.append(body)
        status = 200
        if path.startswith("/sleep"):
            state.sleeping = True
            status = 500 if state.reject_sleep else 200
        elif path == "/wake_up":
            state.sleeping = False
        elif path == "/collective_rpc":
            assert body == {"method": "reload_weights"}
            state.loaded_adapters.clear()
        elif path == "/v1/load_lora_adapter":
            adapter = load_file(str(Path(body["lora_path"]) / "adapter_model.safetensors"))
            assert all("model.language_model.layers" in key for key in adapter)
            state.adapters.add(body["lora_name"])
            state.loaded_adapters.add(body["lora_name"])
        elif path == "/v1/unload_lora_adapter":
            state.adapters.remove(body["lora_name"])
            state.loaded_adapters.discard(body["lora_name"])
        else:
            raise AssertionError(f"Unexpected phase endpoint: {path}")
        return httpx2.Response(status)

    return httpx2.MockTransport(handle), state


def _record_phase_host_budget_calls(
    monkeypatch: pytest.MonkeyPatch, controller: VLLMPhaseController
) -> tuple[list[tuple[int, int, str]], list[tuple[int, str]], list[tuple[int, int]]]:
    combined_budget_calls: list[tuple[int, int, str]] = []
    host_budget_calls: list[tuple[int, str]] = []
    planned_arena_bytes: list[tuple[int, int]] = []
    original_combined_budget = phases.check_pinned_and_sleeping_weight_budget
    original_host_budget = controller._check_host_memory_budget
    original_planned_bytes = phases.PolicyTransfers.planned_host_bytes

    def record_combined_budget(
        *,
        pinned_bytes: int,
        sleeping_weight_bytes: int,
        host_memory_budget_bytes: int,
        phase: str,
    ) -> None:
        combined_budget_calls.append((pinned_bytes, sleeping_weight_bytes, phase))
        original_combined_budget(
            pinned_bytes=pinned_bytes,
            sleeping_weight_bytes=sleeping_weight_bytes,
            host_memory_budget_bytes=host_memory_budget_bytes,
            phase=phase,
        )

    def record_host_budget(required_bytes: int, phase: str) -> None:
        host_budget_calls.append((required_bytes, phase))
        original_host_budget(required_bytes, phase)

    def record_planned_bytes(
        transfers: phases.PolicyTransfers, optimizer: torch.optim.Optimizer | None
    ) -> int:
        existing_host_bytes = transfers.host_bytes
        planned_bytes = original_planned_bytes(transfers, optimizer)
        planned_arena_bytes.append((planned_bytes, existing_host_bytes))
        return planned_bytes

    monkeypatch.setattr(phases, "check_pinned_and_sleeping_weight_budget", record_combined_budget)
    monkeypatch.setattr(controller, "_check_host_memory_budget", record_host_budget)
    monkeypatch.setattr(phases.PolicyTransfers, "planned_host_bytes", record_planned_bytes)
    return combined_budget_calls, host_budget_calls, planned_arena_bytes


def test_sample_train_sync_cycle(
    server: tuple[httpx2.MockTransport, ServerState], tmp_path: Path
) -> None:
    transport, state = server
    model = ToyAdapter(2, 2)
    trainer = ToyTrainer(model, torch.optim.AdamW(model.parameters()), ToyState())
    backend = ToyBackend()
    controller = VLLMPhaseController(
        "http://server",
        backend,
        tmp_path,
        training_device=torch.device("cpu"),
        http_transport=transport,
    )
    controller.initialize(trainer)
    initial_name = backend.model_id
    controller.before_sampling(trainer)
    controller.before_training(trainer)
    assert state.sleeping
    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()
    assert trainer.optimizer is not None
    trainer.optimizer.step()
    trainer.state.global_step = 1
    controller.after_optimizer_step(trainer)
    assert set(controller.last_sync_seconds) == {
        "save",
        "digest",
        "serving_copy",
        "offload",
        "wake",
        "lora_load_http",
    }
    assert all(value >= 0 for value in controller.last_sync_seconds.values())
    assert not state.sleeping
    assert backend.model_id != initial_name
    assert state.adapters == {backend.model_id}
    assert state.requests[-3:] == ["/wake_up", "/v1/unload_lora_adapter", "/v1/load_lora_adapter"]
    load_payload = state.payloads[-1]
    assert "load_inplace" not in load_payload
    assert all(value.device.type == "cpu" for value in model.parameters())
    controller.close()


@pytest.mark.parametrize("device", ["cuda:7", "cuda"])
def test_offload_logs_post_cleanup_cuda_memory_for_training_step(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    device: str,
) -> None:
    transport, _state = server
    expected_device_index = 7
    events: list[str] = []
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        host_memory_safety_margin_bytes=0,
        http_transport=transport,
    )
    controller.training_device = torch.device(device)
    monkeypatch.setattr(
        controller, "_check_host_memory_budget", lambda _required_bytes, _phase: None
    )

    class FakeTransfers:
        host_bytes = 0

        def planned_host_bytes(self, _optimizer: torch.optim.Optimizer | None) -> int:
            return 128

        def offload(self, _optimizer: torch.optim.Optimizer | None) -> None:
            events.append("offload")

        def close(self) -> None:
            events.append("close")

    controller._transfers = cast("phases.PolicyTransfers", FakeTransfers())
    monkeypatch.setattr(phases.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: events.append("current_device") or expected_device_index,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda device: events.append(f"allocated:{device}") or 256,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda device: events.append(f"reserved:{device}") or 512,
    )
    monkeypatch.setattr(
        gpu_memory,
        "_nvidia_smi_rows",
        lambda: (
            events.append("nvidia_smi_rows")
            or [(expected_device_index, "GPU-test", 1024.0, 32768.0)]
        ),
    )
    trainer = ToyTrainer(ToyAdapter(), None, ToyState(global_step=37))

    with caplog.at_level("INFO"):
        controller._offload(trainer)

    expected_events = ["offload", "gc", "empty_cache"]
    if device == "cuda":
        expected_events.append("current_device")
    expected_events.extend(
        [
            "nvidia_smi_rows",
            f"allocated:{device}",
            f"reserved:{device}",
        ]
    )
    assert events == expected_events
    assert (
        f"post-offload CUDA memory step=37 device={expected_device_index} "
        "allocated_bytes=256 reserved_bytes=512 nvidia_smi_free_bytes=33285996544 "
        "nvidia_smi_free_gib=31.00 nvidia_smi_total_bytes=34359738368"
    ) in caplog.text
    controller.close()


def test_wrong_served_adapter_is_refused(
    server: tuple[httpx2.MockTransport, ServerState], tmp_path: Path
) -> None:
    transport, state = server
    state.hide_adapter = True
    model = ToyAdapter(2, 2)
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        http_transport=transport,
    )
    with pytest.raises(RuntimeError, match="not present"):
        controller.initialize(ToyTrainer(model, None, ToyState()))
    controller.close()


def test_sleep_failure_propagates(
    server: tuple[httpx2.MockTransport, ServerState], tmp_path: Path
) -> None:
    transport, state = server
    state.reject_sleep = True
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        http_transport=transport,
    )
    trainer = ToyTrainer(ToyAdapter(2, 2), None, ToyState())
    controller.initialize(trainer)
    with pytest.raises(httpx2.HTTPStatusError, match="500"):
        controller.before_training(trainer)
    controller.close()


def test_nested_optimizer_state_moves_without_changing_values() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    parameter = next(model.parameters())
    optimizer.state[parameter] = {"nested": [torch.tensor([3.0]), {"step": torch.tensor(4)}]}
    move_optimizer_state(optimizer, torch.device("cpu"))
    nested = optimizer.state[parameter]["nested"]
    assert nested[0].item() == 3.0
    assert nested[1]["step"].item() == 4


def test_backend_switch_to_base_is_caught(
    server: tuple[httpx2.MockTransport, ServerState], tmp_path: Path
) -> None:
    transport, _state = server
    backend = ToyBackend()
    trainer = ToyTrainer(ToyAdapter(2, 2), None, ToyState())
    controller = VLLMPhaseController(
        "http://server",
        backend,
        tmp_path,
        training_device=torch.device("cpu"),
        http_transport=transport,
    )
    controller.initialize(trainer)
    backend.model_id = "base"
    with pytest.raises(RuntimeError, match="does not select"):
        controller.before_sampling(trainer)
    controller.close()


def test_same_step_and_weights_keep_served_identity(
    server: tuple[httpx2.MockTransport, ServerState], tmp_path: Path
) -> None:
    transport, _state = server
    trainer = ToyTrainer(ToyAdapter(2, 2), None, ToyState())
    first_backend = ToyBackend()
    first = VLLMPhaseController(
        "http://server",
        first_backend,
        tmp_path,
        training_device=torch.device("cpu"),
        http_transport=transport,
    )
    first.initialize(trainer, step=3)
    first.close()
    second_backend = ToyBackend()
    trainer.model.target_modules.reverse()
    second = VLLMPhaseController(
        "http://server",
        second_backend,
        tmp_path,
        training_device=torch.device("cpu"),
        http_transport=transport,
    )
    second.initialize(trainer, step=3)
    assert first_backend.model_id == second_backend.model_id
    second.close()


def test_initialize_unloads_stale_organism_adapters(
    server: tuple[httpx2.MockTransport, ServerState], tmp_path: Path
) -> None:
    transport, state = server
    state.adapters.update({"organism-step-00000004-dead", "other-adapter"})
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        http_transport=transport,
    )

    controller.initialize(ToyTrainer(ToyAdapter(), None, ToyState()))

    assert "organism-step-00000004-dead" not in state.adapters
    assert "other-adapter" in state.adapters
    assert "/v1/unload_lora_adapter" in state.requests
    controller.close()


def test_level_two_reloads_base_weights_before_sampling(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport, state = server
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        sleep_level=2,
        http_transport=transport,
    )
    trainer = ToyTrainer(ToyAdapter(), None, ToyState())
    controller.initialize(trainer)
    with caplog.at_level("INFO"):
        controller.before_training(trainer)
    assert "/sleep?level=2" in state.requests
    assert "host_available_bytes=" in caplog.text
    controller.before_sampling(trainer)
    assert state.loaded_adapters == state.adapters
    assert state.requests[-4:] == [
        "/wake_up",
        "/collective_rpc",
        "/v1/unload_lora_adapter",
        "/v1/load_lora_adapter",
    ]
    assert state.payloads[-3] == {"method": "reload_weights"}
    controller.close()


def test_level_two_sync_replaces_adapter_without_reloading_previous_adapter(
    server: tuple[httpx2.MockTransport, ServerState], tmp_path: Path
) -> None:
    transport, state = server
    backend = ToyBackend()
    controller = VLLMPhaseController(
        "http://server",
        backend,
        tmp_path,
        training_device=torch.device("cpu"),
        sleep_level=2,
        http_transport=transport,
    )
    trainer = ToyTrainer(ToyAdapter(), None, ToyState())
    controller.initialize(trainer)
    previous_adapter = backend.model_id
    controller.before_training(trainer)
    request_offset = len(state.requests)
    trainer.state.global_step = 1
    controller.after_optimizer_step(trainer)
    assert state.requests[request_offset:] == [
        "/wake_up",
        "/collective_rpc",
        "/v1/unload_lora_adapter",
        "/v1/load_lora_adapter",
    ]
    assert state.payloads[-2] == {"lora_name": previous_adapter}
    assert state.payloads[-1]["lora_name"] == backend.model_id
    assert state.loaded_adapters == state.adapters == {backend.model_id}
    controller.close()


def test_phase_transfers_save_before_offload_without_module_to(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport, _ = server
    model = ToyAdapter()
    trainer = ToyTrainer(model, torch.optim.AdamW(model.parameters()), ToyState())
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        http_transport=transport,
    )
    pointers_at_save: list[int] = []
    save_pretrained = model.save_pretrained

    def save(directory: str) -> None:
        pointers_at_save.append(model.weight.data_ptr())
        save_pretrained(directory)

    def forbidden_to(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("phase controller must use persistent storage transfers")

    monkeypatch.setattr(model, "save_pretrained", save)
    monkeypatch.setattr(model, "to", forbidden_to)
    original_pointer = model.weight.data_ptr()
    with caplog.at_level("INFO"):
        controller.initialize(trainer)
        assert pointers_at_save == [original_pointer]
        assert model.weight.data_ptr() != original_pointer
        controller.before_training(trainer)
        training_pointer = model.weight.data_ptr()
        controller.after_optimizer_step(trainer)
    assert pointers_at_save[-1] == training_pointer
    assert "phase=offload moved_bytes=" in caplog.text
    assert "phase=restore moved_bytes=" in caplog.text
    assert "GB/s=" in caplog.text
    controller.close()


def test_resolve_host_memory_budget_reads_cgroup_ancestor_limits(tmp_path: Path) -> None:
    cgroup_root = tmp_path / "cgroup"
    parent = cgroup_root / "job"
    leaf = parent / "worker"
    leaf.mkdir(parents=True)
    (parent / "memory.max").write_text("10000\n", encoding="ascii")
    (parent / "memory.current").write_text("6000\n", encoding="ascii")
    (parent / "memory.stat").write_text(
        "anon 6000\nactive_file 0\ninactive_file 0\n", encoding="ascii"
    )
    (leaf / "memory.max").write_text("12000\n", encoding="ascii")
    (leaf / "memory.current").write_text("7000\n", encoding="ascii")
    (leaf / "memory.stat").write_text(
        "anon 7000\nactive_file 0\ninactive_file 0\n", encoding="ascii"
    )
    process_cgroup = tmp_path / "self.cgroup"
    process_cgroup.write_text("0::/job/worker\n", encoding="ascii")

    budget_bytes = resolve_host_memory_budget_bytes(
        host_memory_safety_margin_bytes=1024,
        cgroup_root=cgroup_root,
        process_cgroup_file=process_cgroup,
        physical_memory_bytes=20000,
    )

    assert budget_bytes == 8976


def test_cgroup_namespace_falls_back_to_mounted_root_when_membership_path_is_missing(
    tmp_path: Path,
) -> None:
    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    (cgroup_root / "memory.max").write_text("12000\n", encoding="ascii")
    (cgroup_root / "memory.current").write_text("7000\n", encoding="ascii")
    (cgroup_root / "memory.stat").write_text(
        "anon 3000\nfile 4000\nshmem 1000\nactive_file 2000\ninactive_file 1000\n", encoding="ascii"
    )
    process_cgroup = tmp_path / "self.cgroup"
    process_cgroup.write_text("0::/non-systemd\n", encoding="ascii")

    # Reclaimable page cache (e.g. safetensors just read) is not memory the job holds.
    assert phases._cgroup_memory_states(cgroup_root, process_cgroup) == ((12000, 4000),)


def test_host_memory_budget_guard_accounts_for_current_usage_and_margin() -> None:
    with pytest.raises(
        RuntimeError, match=r"before offload.*required_bytes=4001.*headroom_bytes=4000"
    ):
        check_host_memory_budget(
            required_bytes=4001,
            current_memory_bytes=5000,
            host_memory_budget_bytes=10000,
            host_available_bytes=9000,
            safety_margin_bytes=1000,
            phase="offload",
        )


def test_combined_pinned_and_sleeping_weight_budget_guard_uses_net_budget() -> None:
    with pytest.raises(
        RuntimeError, match=r"pinned_bytes=7000 sleeping_weight_bytes=3001.*budget_bytes=10000"
    ):
        check_pinned_and_sleeping_weight_budget(
            pinned_bytes=7000,
            sleeping_weight_bytes=3001,
            host_memory_budget_bytes=10000,
            phase="offload",
        )


@pytest.mark.parametrize("frozen_host_memory", ["pinned", "persistent"])
def test_offload_budget_refusal_happens_before_host_storage_allocation(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_host_memory: str,
) -> None:
    transport, _state = server
    monkeypatch.setattr(phases, "_cgroup_memory_states", lambda *_args: ((1000, 990),))
    monkeypatch.setattr(
        phases.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=4096, available=4096),
    )
    model = ToyAdapter()
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        host_memory_budget_bytes=1000,
        host_memory_safety_margin_bytes=0,
        frozen_host_memory=frozen_host_memory,
        http_transport=transport,
    )

    with pytest.raises(RuntimeError, match="before offload"):
        controller.initialize(ToyTrainer(model, None, ToyState()))

    assert controller._transfers is not None
    assert controller._transfers.host_bytes == 0
    controller.close()


@pytest.mark.parametrize("persistent", [False, True])
def test_phase_host_budget_follows_transfer_and_sleep_transitions(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persistent: bool,
) -> None:
    transport, _state = server
    monkeypatch.setattr(phases, "_cgroup_memory_states", lambda *_args: ())
    monkeypatch.setattr(
        phases.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=8192, available=8192),
    )
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        host_memory_budget_bytes=4096,
        host_memory_safety_margin_bytes=0,
        sleeping_weight_bytes=20,
        frozen_host_memory="persistent" if persistent else "pinned",
        http_transport=transport,
    )
    combined_budget_calls, host_budget_calls, planned_arena_bytes = _record_phase_host_budget_calls(
        monkeypatch, controller
    )
    model = ToyAdapter()
    model.weight.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    trainer = ToyTrainer(model, optimizer, ToyState())
    controller.initialize(trainer)
    assert controller._transfers is not None
    assert controller._transfers.frozen_host_storage_ids
    first_planned_bytes, first_existing_bytes = planned_arena_bytes[0]
    assert first_existing_bytes == 0
    assert combined_budget_calls[0] == (first_planned_bytes, 20 if persistent else 0, "offload")
    assert host_budget_calls[0] == (first_planned_bytes, "offload")

    controller.before_training(trainer)
    assert combined_budget_calls[1] == (first_planned_bytes, 20, "vLLM sleep")
    assert host_budget_calls[1] == (20, "vLLM sleep")
    assert bool(controller._transfers.frozen_host_storage_ids) == persistent
    retained_mutable_bytes = controller._transfers.host_bytes
    assert retained_mutable_bytes > 0
    if persistent:
        assert retained_mutable_bytes == first_planned_bytes

    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()
    optimizer.step()
    trainer.state.global_step = 1
    controller.after_optimizer_step(trainer)

    second_planned_bytes, second_existing_bytes = planned_arena_bytes[1]
    assert second_planned_bytes > second_existing_bytes == retained_mutable_bytes
    assert combined_budget_calls[2] == (second_planned_bytes, 20, "offload")
    assert host_budget_calls[2] == (
        second_planned_bytes - second_existing_bytes,
        "offload",
    )
    assert controller._transfers.frozen_host_storage_ids

    # vLLM's pinned weight backup stays allocated in its process after the first sleep.
    controller.before_training(trainer)
    assert combined_budget_calls[3] == (second_planned_bytes, 20, "vLLM sleep")
    assert host_budget_calls[3] == (0, "vLLM sleep")
    if persistent:
        controller.before_sampling(trainer)
        assert planned_arena_bytes[2] == (second_planned_bytes, second_planned_bytes)
        assert combined_budget_calls[4] == (second_planned_bytes, 20, "offload")
        assert host_budget_calls[4] == (0, "offload")
        controller.before_training(trainer)
        assert host_budget_calls[5] == (0, "vLLM sleep")
    controller.close()


def test_sleep_budget_refusal_happens_before_vllm_sleep_request(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, state = server
    cgroup_states = [(1000, 0)]
    monkeypatch.setattr(phases, "_cgroup_memory_states", lambda *_args: tuple(cgroup_states))
    monkeypatch.setattr(
        phases.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=4096, available=4096),
    )
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        host_memory_budget_bytes=1000,
        host_memory_safety_margin_bytes=0,
        sleeping_weight_bytes=20,
        http_transport=transport,
    )
    controller.initialize(ToyTrainer(ToyAdapter(), None, ToyState()))
    cgroup_states[:] = [(1000, 990)]
    request_count = len(state.requests)

    with pytest.raises(RuntimeError, match="before vLLM sleep"):
        controller.sleep()

    assert len(state.requests) == request_count
    controller.close()


def test_repeat_sleep_still_reserves_configured_vllm_weight_bytes(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, state = server
    monkeypatch.setattr(phases, "_cgroup_memory_states", lambda *_args: ())
    monkeypatch.setattr(
        phases.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=4096, available=4096),
    )
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        host_memory_budget_bytes=19,
        host_memory_safety_margin_bytes=0,
        sleeping_weight_bytes=20,
        http_transport=transport,
    )
    controller._vllm_sleeping = True

    with pytest.raises(
        RuntimeError, match=r"pinned_bytes=0 sleeping_weight_bytes=20.*budget_bytes=19"
    ):
        controller.sleep()

    assert not state.requests
    controller.close()


def test_cuda_sleep_level_one_requires_sleeping_weight_estimate(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, state = server
    monkeypatch.setattr(phases, "device_memory", lambda _device: phases.DeviceMemory("fake", 1, 1))
    with pytest.raises(RuntimeError, match="ORGANISM_VLLM_SLEEPING_WEIGHT_BYTES"):
        VLLMPhaseController(
            "http://server",
            ToyBackend(),
            tmp_path,
            training_device=torch.device("cuda"),
            host_memory_budget_bytes=4096,
            host_memory_safety_margin_bytes=0,
            http_transport=transport,
        )

    assert not state.requests


def test_host_memory_and_sleeping_weight_settings_read_environment(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, _state = server
    monkeypatch.setattr(phases, "_cgroup_memory_states", lambda *_args: ())
    monkeypatch.setattr(
        phases.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=20000, available=18000),
    )
    monkeypatch.setenv(phases.HOST_MEMORY_BUDGET_ENV, "10000")
    monkeypatch.setenv(phases.HOST_MEMORY_SAFETY_MARGIN_ENV, "0")
    monkeypatch.setenv(phases.VLLM_SLEEPING_WEIGHT_BYTES_ENV, "20")

    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        http_transport=transport,
    )

    assert controller.host_memory_budget_bytes == 10000
    assert controller.host_memory_safety_margin_bytes == 0
    assert controller.sleeping_weight_bytes == 20
    controller.close()


def test_sleeping_weight_bytes_come_from_the_owned_server_log(tmp_path: Path) -> None:
    server_log = tmp_path / "vllm-server.log"
    server_log.write_text(
        "(EngineCore pid=1) INFO 10-17 00:02:43 [model_runner.py:407] "
        "Model loading took 10.96 GiB memory and 27.651354 seconds\n",
        encoding="utf-8",
    )
    assert phases.vllm_sleeping_weight_bytes(server_log) == int(10.96 * 2**30)
    server_log.write_text("no load line yet\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Model loading took"):
        phases.vllm_sleeping_weight_bytes(server_log)


def test_sleeping_weight_bytes_use_the_latest_launch_in_an_appended_log(tmp_path: Path) -> None:
    server_log = tmp_path / "vllm-server.log"
    server_log.write_text(
        "INFO [model_runner.py:407] Model loading took 1.50 GiB memory and 9.1 seconds\n"
        "ERROR startup refused, relaunching in the same output dir\n"
        "INFO [model_runner.py:407] Model loading took 1.75 GiB memory and 8.7 seconds\n",
        encoding="utf-8",
    )
    assert phases.vllm_sleeping_weight_bytes(server_log) == int(1.75 * 2**30)


@pytest.mark.parametrize("frozen_host_memory", ["pageable", "pinned", "persistent"])
def test_controller_builds_requested_host_arena_mode(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_host_memory: str,
) -> None:
    transport, _state = server
    monkeypatch.setattr(phases, "_cgroup_memory_states", lambda *_args: ())
    monkeypatch.setattr(
        phases.psutil, "virtual_memory", lambda: SimpleNamespace(total=1 << 30, available=1 << 30)
    )
    mode_arguments: list[tuple[bool, bool]] = []
    real_transfers = phases.PolicyTransfers

    def recording_transfers(
        model: torch.nn.Module, device: torch.device, *, pin_memory: bool, persistent: bool
    ) -> phases.PolicyTransfers:
        mode_arguments.append((pin_memory, persistent))
        return real_transfers(model, device, pin_memory=pin_memory, persistent=persistent)

    monkeypatch.setattr(phases, "PolicyTransfers", recording_transfers)
    controller = VLLMPhaseController(
        "http://server",
        ToyBackend(),
        tmp_path,
        training_device=torch.device("cpu"),
        host_memory_budget_bytes=1 << 30,
        host_memory_safety_margin_bytes=0,
        sleeping_weight_bytes=20,
        frozen_host_memory=frozen_host_memory,
        http_transport=transport,
    )
    model = ToyAdapter()
    model.weight.requires_grad_(False)
    controller.initialize(ToyTrainer(model, None, ToyState()))
    assert mode_arguments == [
        (frozen_host_memory != "pageable", frozen_host_memory == "persistent")
    ]
    controller.close()
