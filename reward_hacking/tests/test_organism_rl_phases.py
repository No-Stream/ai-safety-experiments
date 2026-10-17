"""Exercise HTTP protocol boundaries with real adapters and a CPU optimizer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
import torch
from safetensors.torch import load_file, save_file

from reward_hacking.organism_rl import phases
from reward_hacking.organism_rl.phases import (
    VLLMPhaseController,
    check_host_memory_budget,
    check_pinned_and_sleeping_weight_budget,
    move_optimizer_state,
    resolve_host_memory_budget_bytes,
)


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


def test_offload_budget_refusal_happens_before_host_storage_allocation(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        http_transport=transport,
    )

    with pytest.raises(RuntimeError, match="before offload"):
        controller.initialize(ToyTrainer(model, None, ToyState()))

    assert controller._transfers is not None
    assert controller._transfers.host_bytes == 0
    controller.close()


def test_phase_host_budget_follows_transient_transfer_and_sleep_transitions(
    server: tuple[httpx2.MockTransport, ServerState],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    assert combined_budget_calls[0] == (first_planned_bytes, 0, "offload")
    assert host_budget_calls[0] == (first_planned_bytes, "offload")

    controller.before_training(trainer)
    assert combined_budget_calls[1] == (first_planned_bytes, 20, "vLLM sleep")
    assert host_budget_calls[1] == (20, "vLLM sleep")
    assert not controller._transfers.frozen_host_storage_ids
    retained_mutable_bytes = controller._transfers.host_bytes
    assert retained_mutable_bytes > 0

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
