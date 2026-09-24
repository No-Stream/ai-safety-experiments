"""CPU checks for the opt-in colocated vLLM sleep/offload seam."""

from __future__ import annotations

import contextlib
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from games import sizing
from games import train as gt
from grpo.throughput import STEP_PHASES, TIMING_METRIC_KEYS


class RecordingModel(torch.nn.Linear):
    """A real tiny policy that records the devices the offload seam requests."""

    def __init__(self) -> None:
        super().__init__(2, 2, bias=False)
        self.devices: list[torch.device | str] = []

    def to(self, *args: Any, **kwargs: Any) -> RecordingModel:  # type: ignore[override]
        self.devices.append(args[0] if args else kwargs["device"])
        return super().to(*args, **kwargs)  # type: ignore[return-value]


class RecordingTimer:
    """The timing protocol the seam needs, without calling CUDA from a CPU test."""

    def __init__(self) -> None:
        self.phases: list[str] = []

    @contextlib.contextmanager
    def phase(self, name: str):  # type: ignore[no-untyped-def]
        self.phases.append(name)
        yield


class SleepingGeneration:
    """Minimal engine fake; vLLM itself is unavailable without touching the card."""

    def __init__(self, model: RecordingModel) -> None:
        self.model = model
        self.events: list[str] = []
        self._llm_weights_sleeping = False

    def sync_weights(self) -> None:
        assert next(self.model.parameters()).device.type == "cpu"
        self.events.append("sync")

    def generate(self) -> str:
        assert next(self.model.parameters()).device.type == "cpu"
        self.events.append("generate")
        self._llm_weights_sleeping = True
        return "generated"


def _populated_optimizer(model: RecordingModel) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    return optimizer


def test_sleep_offload_round_trip_preserves_policy_and_optimizer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real policy objects survive CPU offload and return after vLLM sleeps."""
    model = RecordingModel()
    optimizer = _populated_optimizer(model)
    generation = SleepingGeneration(model)
    timer = RecordingTimer()
    trainer = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_generation=generation,
    )
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    optimizer_parameter_ids = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimizer_state_devices = {
        id(parameter): {
            name: value.device for name, value in state.items() if isinstance(value, torch.Tensor)
        }
        for parameter, state in optimizer.state.items()
    }
    empty_cache_calls: list[None] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(None))

    gt.install_colocated_sleep_offload(trainer, timer)  # type: ignore[arg-type]

    generation.sync_weights()
    assert generation.generate() == "generated"

    assert generation.events == ["sync", "generate"]
    assert empty_cache_calls == [None]
    assert timer.phases == ["policy_to_cpu", "policy_to_cuda"]
    assert model.devices == []
    assert [id(parameter) for parameter in model.parameters()] == parameter_ids
    assert [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ] == optimizer_parameter_ids
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert {
        id(parameter): {
            name: value.device for name, value in state.items() if isinstance(value, torch.Tensor)
        }
        for parameter, state in optimizer.state.items()
    } == optimizer_state_devices


def test_cpu_policy_round_trip_preserves_identity_and_measures_copy_seconds() -> None:
    """The small CPU analogue keeps the policy objects while exercising both copies."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3))
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    buffer_ids = tuple(id(buffer) for buffer in model.buffers())
    original_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    original_buffers = [buffer.detach().clone() for buffer in model.buffers()]

    start = time.perf_counter()
    cpu_staging = gt.move_policy_to_cpu_staging(model)
    gt.restore_policy_from_cpu_staging(model, cpu_staging, torch.device("cpu"))
    round_trip_seconds = time.perf_counter() - start

    assert round_trip_seconds >= 0.0
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    assert tuple(id(buffer) for buffer in model.buffers()) == buffer_ids
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(model.parameters(), original_parameters, strict=True)
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(model.buffers(), original_buffers, strict=True)
    )


def test_sleep_offload_refuses_a_non_colocated_vllm_trainer() -> None:
    trainer = SimpleNamespace(use_vllm=False, vllm_mode="colocate")

    with pytest.raises(ValueError, match="colocated vLLM"):
        gt.install_colocated_sleep_offload(trainer, RecordingTimer())  # type: ignore[arg-type]


def test_sleep_mode_releases_the_engine_reservation_from_training_sizing() -> None:
    total_vram_gib = 47.5
    assert (
        sizing.colocate_reserved_gib(
            total_vram_gib=total_vram_gib, gpu_memory_utilization=0.8, sleep_mode=True
        )
        == 0.0
    )
    assert sizing.colocate_reserved_gib(
        total_vram_gib=total_vram_gib, gpu_memory_utilization=0.8
    ) == pytest.approx(38.0)
    assert sizing._vram_left_to_the_trainer(  # pyright: ignore[reportPrivateUsage]
        free_vram_gib=31.25, engine_reserved_gib=0.0, label="sleep-offload test"
    ) == pytest.approx(31.25)


def test_swap_metrics_reach_the_existing_mem_log_column_set() -> None:
    assert {"policy_to_cpu", "policy_to_cuda"} <= set(STEP_PHASES)
    assert {"timing/policy_to_cpu_s", "timing/policy_to_cuda_s"} <= set(TIMING_METRIC_KEYS)
    assert {"timing/policy_to_cpu_s", "timing/policy_to_cuda_s"} <= set(gt.MEM_LOG_EXTRA_COLUMNS)
