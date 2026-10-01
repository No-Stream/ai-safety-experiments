"""CPU checks for sleep-offload host capacity and persisted host telemetry."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest
from transformers import TrainerControl, TrainerState
from trl.trainer.grpo_trainer import GRPOTrainer

from games import sizing
from games import train as gt

if TYPE_CHECKING:
    from pathlib import Path


def fake_source(tmp_path: Path) -> Path:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 18 * 1024**3}})
    )
    return tmp_path


def test_estimate_from_index(tmp_path: Path) -> None:
    assert sizing.checkpoint_weight_bytes(str(fake_source(tmp_path))) == 18 * 1024**3
    assert sizing.sleep_offload_peak_host_bytes(18 * 1024**3) == 40 * 1024**3


@pytest.mark.parametrize("available_gib", [39, 40, 41])
def test_capacity_boundary(
    tmp_path: Path, available_gib: int, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        f"MemFree: 1 kB\nMemAvailable: {available_gib * 1024**2} kB\nSwapFree: 999999999 kB\n"
    )
    available = sizing.host_mem_available_bytes(meminfo)
    if available_gib < 40:
        with pytest.raises(RuntimeError, match=r"40.00 GiB.*39.00 GiB.*2.0") as error:
            sizing.check_sleep_offload_host_ram(18 * 1024**3, available)
        assert "Free host RAM or raise .wslconfig memory=" in str(error.value)
    else:
        sizing.check_sleep_offload_host_ram(18 * 1024**3, available)
        assert "host RAM preflight passed" in caplog.text
        assert "Free host RAM or raise .wslconfig memory=" not in caplog.text


def test_acknowledged_shortfall(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    del tmp_path
    sizing.check_sleep_offload_host_ram(18 * 1024**3, 39 * 1024**3, acknowledged=True)
    assert "acknowledged" in caplog.text
    assert "Free host RAM or raise .wslconfig memory=" not in caplog.text


@pytest.mark.parametrize("enabled", [False, True])
def test_guard_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        gt, "checkpoint_weight_bytes", lambda source: calls.append(source) or 18 * 1024**3
    )
    monkeypatch.setattr(
        gt, "host_mem_available_bytes", lambda: calls.append("meminfo") or 39 * 1024**3
    )

    def prepare(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("model preparation reached")

    monkeypatch.setattr(gt, "_prepare_run", prepare)
    config = gt.GameTrainConfig(
        arm="twin-pd-group",
        generate_fresh=True,
        colocate_sleep_offload=enabled,
        model_source=str(tmp_path),
    )
    with pytest.raises(RuntimeError if enabled else AssertionError):
        gt.train_game_arm(config)
    assert calls == [str(tmp_path), "meminfo"] if enabled else calls == []


def test_acknowledgement_cli_and_resume_identity() -> None:
    config = gt._parse_args(
        [
            "--arm",
            "twin-pd-group",
            "--generate-fresh",
            "--colocate-sleep-offload",
            "--acknowledge-host-ram-shortfall",
        ]
    )
    assert config.acknowledge_host_ram_shortfall is True
    assert "acknowledge_host_ram_shortfall" not in gt.RESUME_IDENTITY_FIELDS


def test_host_memory_reaches_history_and_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = cast("Any", object.__new__(gt.PaddingTrimmedGRPOTrainer))
    trainer.accelerator = SimpleNamespace(is_main_process=False)
    trainer.log_completions = False
    trainer.state = TrainerState(global_step=1)
    monitor = gt.MemoryMonitorCallback(extra_columns=gt.MEM_LOG_EXTRA_COLUMNS)
    monitor.csv_path = tmp_path / "mem_log.csv"
    monitor.csv_path.write_text(monitor.csv_header())
    monkeypatch.setattr(gt, "host_mem_available_bytes", lambda: 7 * 1024**3)
    monkeypatch.setattr(
        monitor,
        "_gpu_mem_stats",
        lambda: dict.fromkeys(
            ["free_gib", "total_gib", "used_gib", "alloc_gib", "reserved_gib"], 0.0
        ),
    )

    def log(self: Any, logs: dict[str, float], start_time: float | None = None) -> None:
        self.state.log_history.append(dict(logs))
        monitor.on_log(
            cast(
                "Any",
                SimpleNamespace(
                    per_device_train_batch_size=1, world_size=1, gradient_accumulation_steps=1
                ),
            ),
            self.state,
            TrainerControl(),
            logs=logs,
        )

    monkeypatch.setattr(GRPOTrainer, "log", log)
    trainer.log({"reward": 0.5})
    assert trainer.state.log_history[-1]["host_mem_available_gib"] == 7.0
    assert pd.read_csv(monitor.csv_path)["host_mem_available_gib"].tolist() == [7.0]
