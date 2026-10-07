"""Exercise HTTP protocol boundaries with real adapters and a CPU optimizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
import pytest
import torch
from safetensors.torch import load_file, save_file

from reward_hacking.organism_rl.phases import VLLMPhaseController, move_optimizer_state


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
        elif path == "/v1/load_lora_adapter":
            adapter = load_file(str(Path(body["lora_path"]) / "adapter_model.safetensors"))
            assert all("model.language_model.layers" in key for key in adapter)
            state.adapters.add(body["lora_name"])
        elif path == "/v1/unload_lora_adapter":
            state.adapters.remove(body["lora_name"])
        else:
            raise AssertionError(f"Unexpected phase endpoint: {path}")
        return httpx2.Response(status)

    return httpx2.MockTransport(handle), state


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
