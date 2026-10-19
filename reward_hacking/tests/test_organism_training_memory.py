"""Micro-step telemetry brackets the real backward and survives incomplete rows."""

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from reward_hacking.organism_rl import phases, training_memory
from reward_hacking.organism_rl.trainer import _SyncCallback
from reward_hacking.tests.test_organism_rl_trainer import _trainer


def test_cpu_training_does_not_query_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("CPU telemetry must not query CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    trainer = _trainer(tmp_path)
    trainer.train()
    assert not (tmp_path / "trainer-memory.jsonl").exists()


def test_microstep_resets_peak_and_persists_start_before_backward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    allocated = 100
    peak = 100
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: allocated)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: allocated + 20)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: peak)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: peak + 40)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda _device: events.append("reset")
    )
    monkeypatch.setattr(training_memory, "gpu_memory_info", lambda *_args: (800, 1000))
    recorder = training_memory.MicrostepMemoryRecorder(tmp_path / "trainer-memory.jsonl")
    device = torch.device("cuda:0")
    for index, tokens in enumerate((100, 200)):
        recorder.begin(device, step=5, microstep=index, sequence_tokens=tokens, assistant_tokens=20)
        records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
        assert records[-1]["event"] == "microstep_start"
        events.append("backward")
        allocated, peak = 110, 300 + index * 100
        assert recorder.finish(completed=True, skipped=False) == peak
    records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert events == ["sync", "reset", "backward", "sync"] * 2
    completed = [record for record in records if record["event"] == "microstep_end"]
    assert [record["peak_allocated_bytes"] for record in completed] == [300, 400]
    assert [record["sequence_tokens"] for record in completed] == [100, 200]
    assert completed[0]["free_bytes"] == 800
    assert completed[0]["allocated_before_bytes"] == 100
    assert completed[1]["allocated_before_bytes"] == 110
    assert all(record["completed"] and not record["skipped"] for record in completed)
    assert recorder.finish(completed=True, skipped=False) == 0


def test_backward_failure_keeps_start_without_claiming_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 100)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 120)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(training_memory, "gpu_memory_info", lambda *_args: (800, 1000))
    recorder = training_memory.MicrostepMemoryRecorder(tmp_path / "memory.jsonl")
    recorder.begin(
        torch.device("cuda"), step=5, microstep=2, sequence_tokens=100, assistant_tokens=20
    )

    # CUDA may already be unusable after backward raises; finishing must not query it again.
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("failed CUDA context must not be queried")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    assert recorder.finish(completed=False, skipped=False) == 0
    records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["microstep_start", "microstep_failed"]
    assert records[-1]["completed"] is False
    assert "peak_allocated_bytes" not in records[-1]


def test_native_training_backward_is_between_recorder_calls(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    events: list[str] = []

    class RecordingMemory:
        def begin(self, *_args: Any, **_kwargs: Any) -> None:
            events.append("start")

        def finish(self, *, completed: bool, skipped: bool) -> int:
            assert completed
            assert not skipped
            events.append("end")
            return 123

    trainer.microstep_memory = RecordingMemory()  # type: ignore[assignment]
    assert trainer.model is not None
    parameter = next(
        parameter for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    hook = parameter.register_hook(lambda gradient: events.append("backward") or gradient)
    try:
        trainer.train()
    finally:
        hook.remove()
    assert events == ["start", "backward", "end"] * 2
    assert trainer.training_step_peak_allocated_bytes == 123


def test_phase_snapshots_bracket_trainer_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    backend: Any = SimpleNamespace(model_id="toy")
    controller = phases.VLLMPhaseController(
        "http://server",
        backend,
        tmp_path / "served",
        training_device=torch.device("cpu"),
    )
    controller.training_device = torch.device("cuda:0")
    controller._transfers = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
        restore=lambda _optimizer: events.append("restore"), close=lambda: None
    )
    monkeypatch.setattr(controller, "_assert_served", lambda: None)
    monkeypatch.setattr(controller, "sleep", lambda: events.append("sleep"))

    def record(path: Path, _device: torch.device, *, step: int, phase: str) -> None:
        assert path == tmp_path / "trainer-memory.jsonl"
        assert step == 5
        events.append(phase)

    monkeypatch.setattr(phases, "record_training_phase_memory", record)
    trainer: Any = SimpleNamespace(state=SimpleNamespace(global_step=5), optimizer=None)
    try:
        controller.before_training(trainer)
    finally:
        controller.close()
    assert events == ["sleep", "sampler_sleep", "restore", "trainer_restored"]


def test_training_phase_records_actual_free_and_allocator_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 100)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 120)
    monkeypatch.setattr(training_memory, "gpu_memory_info", lambda *_args: (800, 1000))
    path = tmp_path / "memory.jsonl"
    training_memory.record_training_phase_memory(
        path, torch.device("cuda:0"), step=5, phase="sampler_sleep"
    )
    record = json.loads(path.read_text())
    assert record["free_bytes"] == 800
    assert record["total_bytes"] == 1000
    assert record["allocated_bytes"] == 100
    assert record["reserved_bytes"] == 120
    assert record["event"] == "sampler_sleep"
    assert record["step"] == 5


def test_recorder_refuses_nested_microsteps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 100)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 120)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(training_memory, "gpu_memory_info", lambda *_args: (800, 1000))
    recorder = training_memory.MicrostepMemoryRecorder(tmp_path / "memory.jsonl")
    recorder.begin(
        torch.device("cuda:0"), step=5, microstep=0, sequence_tokens=10, assistant_tokens=5
    )
    with pytest.raises(RuntimeError, match="already active"):
        recorder.begin(
            torch.device("cuda:0"), step=5, microstep=1, sequence_tokens=10, assistant_tokens=5
        )
    recorder.finish(completed=False, skipped=False)


def test_step_peak_keeps_largest_row_after_per_row_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics: dict[str, list[float]] = defaultdict(list)
    trainer: Any = SimpleNamespace(
        _metrics={"train": metrics},
        training_step_peak_allocated_bytes=500,
        resume_sampler_after_sync=lambda _step: None,
    )
    phase: Any = SimpleNamespace(after_optimizer_step=lambda _trainer: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 200)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    arguments: Any = SimpleNamespace(device=torch.device("cuda:0"))
    state: Any = SimpleNamespace(global_step=5)
    _SyncCallback(trainer, phase).on_step_end(arguments, state, None)  # type: ignore[arg-type]
    assert metrics["organism/trainer_peak_allocated_gib"] == [500 / 2**30]
    assert trainer.training_step_peak_allocated_bytes == 0
