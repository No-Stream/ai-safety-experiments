"""CPU checks for the opt-in colocated vLLM sleep/offload seam."""

from __future__ import annotations

import contextlib
import time
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model

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


class TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(3, 4, bias=False)
        self.norm = torch.nn.LayerNorm(4)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(values))


class RecordingWeightSink:
    def __init__(self) -> None:
        self.pushed: dict[str, torch.Tensor] = {}

    @staticmethod
    def _fix_param_name_to_vllm(name: str, extra_prefixes: list[str] | None = None) -> str:
        for prefix in ["_checkpoint_wrapped_module.", *(extra_prefixes or [])]:
            name = name.replace(prefix, "")
        return name

    def _push_param_to_vllm(self, name: str, parameter: torch.Tensor) -> None:
        self.pushed[name] = parameter.detach().cpu().clone()


class PeftSleepingGeneration:
    """The colocated generation hooks needed by the FP8 sync installation seam."""

    def __init__(self, events: list[str], sink: RecordingWeightSink) -> None:
        self.events = events
        self.sink = sink
        self.llm = SimpleNamespace(
            wake_up=self._wake_up,
            reset_prefix_cache=self._reset_prefix_cache,
        )
        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self._llm_weights_sleeping = True

    def _wake_up(self, *, tags: list[str]) -> None:
        assert tags == ["weights"]
        self.events.append("wake")

    def _reset_prefix_cache(self) -> None:
        self.events.append("reset")

    def sync_weights(self) -> None:
        raise AssertionError("PEFT sync should stream parameters through the sleep-offload wrapper")

    def generate(self) -> str:
        raise AssertionError("the test does not generate tokens")

    def _fix_param_name_to_vllm(self, name: str, extra_prefixes: list[str] | None = None) -> str:
        return self.sink._fix_param_name_to_vllm(name, extra_prefixes)

    def _push_param_to_vllm(self, name: str, parameter: torch.Tensor) -> None:
        self.events.append(f"push:{name}")
        self.sink._push_param_to_vllm(name, parameter)


def _make_lora_policy() -> PeftModel:
    policy = cast(
        "PeftModel",
        get_peft_model(
            cast("Any", TinyPolicy()),
            LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["proj"]),
        ).to(torch.bfloat16),
    )
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if "lora_B" in name:
                parameter.normal_()
    return policy


def _sleep_offload_trainer(
    policy: PeftModel, generation: PeftSleepingGeneration
) -> SimpleNamespace:
    return SimpleNamespace(
        model=policy,
        optimizer=None,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_generation=generation,
        accelerator=SimpleNamespace(device=torch.device("cpu")),
    )


@pytest.mark.parametrize("use_prepare", [False, True], ids=["install", "prepare"])
def test_fp8_sleep_offload_uses_real_peft_stream_between_wake_and_reset(
    monkeypatch: pytest.MonkeyPatch, use_prepare: bool
) -> None:
    events: list[str] = []
    sink = RecordingWeightSink()
    generation = PeftSleepingGeneration(events, sink)
    policy = _make_lora_policy()
    trainer = _sleep_offload_trainer(policy, generation)
    constructed_generations: list[object] = []

    class RecordingFp8Sync:
        def __init__(self, sync_generation: object) -> None:
            constructed_generations.append(sync_generation)

        def sync(self, stream: Any, *, push_param: Any) -> None:
            events.append("initialize")
            stream(push_param)
            events.append("finalize")

    monkeypatch.setattr(gt, "Fp8RolloutWeightSync", RecordingFp8Sync)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    if use_prepare:
        gt.prepare_colocated_sleep_offload(trainer, RecordingTimer(), rollout_weights="fp8")  # type: ignore[arg-type]
    else:
        gt.install_colocated_sleep_offload(trainer, RecordingTimer(), rollout_weights="fp8")  # type: ignore[arg-type]

    assert constructed_generations == [generation]
    reference_sink = RecordingWeightSink()
    gt.sync_peft_weights_to_vllm_nonmutating(
        policy,
        fix_param_name=reference_sink._fix_param_name_to_vllm,
        push_param=reference_sink._push_param_to_vllm,
        merge_device=torch.device("cpu"),
    )
    expected_sync_events = [
        "wake",
        "initialize",
        *(f"push:{name}" for name in reference_sink.pushed),
        "finalize",
        "reset",
    ]

    generation.sync_weights()
    generation.sync_weights()

    assert events == expected_sync_events * 2
    assert sink.pushed.keys() == reference_sink.pushed.keys()
    for name, expected_parameter in reference_sink.pushed.items():
        assert torch.equal(sink.pushed[name], expected_parameter), name


def test_default_sleep_offload_does_not_construct_fp8_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    sink = RecordingWeightSink()
    generation = PeftSleepingGeneration(events, sink)
    trainer = _sleep_offload_trainer(_make_lora_policy(), generation)
    constructed_generations: list[object] = []

    class RecordingFp8Sync:
        def __init__(self, sync_generation: object) -> None:
            constructed_generations.append(sync_generation)

    monkeypatch.setattr(gt, "Fp8RolloutWeightSync", RecordingFp8Sync)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    gt.install_colocated_sleep_offload(trainer, RecordingTimer())  # type: ignore[arg-type]
    generation.sync_weights()

    assert constructed_generations == []
    assert events[0] == "wake"
    assert events[-1] == "reset"
    assert all(event.startswith("push:") or event in {"wake", "reset"} for event in events)


def test_nonmutating_lora_sync_matches_peft_merge_and_preserves_policy_bits() -> None:
    torch.manual_seed(17)
    policy = cast(
        "PeftModel",
        get_peft_model(
            cast("Any", TinyPolicy()),
            LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["proj"]),
        ).to(torch.bfloat16),
    )
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if "lora_B" in name:
                parameter.normal_()
    reference = deepcopy(policy)
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    parameter_ids = {name: id(parameter) for name, parameter in policy.named_parameters()}
    sink = RecordingWeightSink()

    gt.sync_peft_weights_to_vllm_nonmutating(
        policy,
        fix_param_name=sink._fix_param_name_to_vllm,
        push_param=sink._push_param_to_vllm,
        merge_device=torch.device("cpu"),
    )
    reference_internals = cast("Any", reference)
    reference_internals.merge_adapter()
    expected = {
        name.removeprefix("base_model.model.").replace(".base_layer", ""): parameter.detach()
        for name, parameter in reference.named_parameters()
        if cast("str", reference_internals.prefix) not in name and "original_module" not in name
    }

    assert sink.pushed.keys() == expected.keys()
    for name, parameter in expected.items():
        assert torch.equal(sink.pushed[name], parameter), name
        assert not sink.pushed[name].requires_grad
    assert before.keys() == dict(policy.named_parameters()).keys()
    for name, parameter in policy.named_parameters():
        assert id(parameter) == parameter_ids[name]
        assert torch.equal(parameter, before[name]), name
