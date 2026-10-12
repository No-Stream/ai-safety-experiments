"""CPU-only tests for WSL-aware CUDA memory sizing."""

from __future__ import annotations

import logging
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from grpo import gpu_memory


class FakeCuda:
    def __init__(self, free_bytes: int, total_bytes: int, current_device: int = 0) -> None:
        self.free_bytes = free_bytes
        self.total_bytes = total_bytes
        self.current_device_index = current_device

    def current_device(self) -> int:
        return self.current_device_index

    def mem_get_info(self, device: int) -> tuple[int, int]:
        assert device == self.current_device_index
        return self.free_bytes, self.total_bytes


class FakeTorch:
    def __init__(self, free_bytes: int, total_bytes: int, current_device: int = 0) -> None:
        self.cuda = FakeCuda(free_bytes, total_bytes, current_device)


def _nvidia_smi_row(*, index: int = 0, used_mib: int = 12_288, total_mib: int = 32_768) -> str:
    return f"{index}, GPU-test-uuid, {used_mib}, {total_mib}\n"


def _install_nvidia_smi(monkeypatch: pytest.MonkeyPatch, stdout: str) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        assert kwargs["check"] is False
        assert kwargs["text"] is True
        assert kwargs["timeout"] > 0
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(gpu_memory.subprocess, "run", fake_run)
    return commands


def test_host_held_memory_caps_requested_vllm_utilization_and_logs_both_sources(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    caplog.set_level(logging.INFO, logger="grpo.gpu_memory")
    _install_nvidia_smi(monkeypatch, _nvidia_smi_row(used_mib=12_288))
    torch_module = FakeTorch(free_bytes=30 * 1024**3, total_bytes=32 * 1024**3)

    utilization = gpu_memory.cap_gpu_memory_utilization(torch_module, requested=1.0)

    assert utilization == pytest.approx(0.605)
    assert "torch_free_gib=30.00" in caplog.text
    assert "nvidia_smi_free_gib=20.00" in caplog.text
    assert "selected_free_gib=20.00" in caplog.text
    assert "host_held_estimate_gib=10.00" in caplog.text


def test_torch_free_memory_wins_when_it_is_lower_than_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _install_nvidia_smi(monkeypatch, _nvidia_smi_row(used_mib=16_384))
    torch_module = FakeTorch(free_bytes=10 * 1024**3, total_bytes=32 * 1024**3)

    free_bytes, total_bytes = gpu_memory.gpu_memory_info(torch_module)
    utilization = gpu_memory.cap_gpu_memory_utilization(torch_module, requested=1.0)

    assert (free_bytes, total_bytes) == (10 * 1024**3, 32 * 1024**3)
    assert utilization == pytest.approx(0.2925)


def test_reserved_bytes_reduce_the_capped_utilization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _install_nvidia_smi(monkeypatch, _nvidia_smi_row(used_mib=2_048))
    torch_module = FakeTorch(free_bytes=30 * 1024**3, total_bytes=32 * 1024**3)

    utilization = gpu_memory.cap_gpu_memory_utilization(
        torch_module, requested=1.0, reserve_bytes=4 * 1024**3
    )

    assert utilization == pytest.approx(0.7925)


def test_reserved_bytes_that_exhaust_available_memory_fail_with_host_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _install_nvidia_smi(monkeypatch, _nvidia_smi_row(used_mib=32_256))
    torch_module = FakeTorch(free_bytes=30 * 1024**3, total_bytes=32 * 1024**3)

    with pytest.raises(
        RuntimeError, match=r"reserve_bytes=536870912.*host-held/other-process.*29\.50 GiB"
    ):
        gpu_memory.cap_gpu_memory_utilization(
            torch_module,
            requested=1.0,
            margin_fraction=0.01,
            reserve_bytes=512 * 1024**2,
        )


def test_minimum_engine_allocation_fails_with_host_held_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _install_nvidia_smi(monkeypatch, _nvidia_smi_row(used_mib=12_288))
    torch_module = FakeTorch(free_bytes=30 * 1024**3, total_bytes=32 * 1024**3)

    with pytest.raises(
        RuntimeError,
        match=r"minimum_utilization_bytes=21474836480.*host-held/other-process.*10\.00 GiB",
    ):
        gpu_memory.cap_gpu_memory_utilization(
            torch_module,
            requested=1.0,
            minimum_utilization_bytes=20 * 1024**3,
        )


def test_minimum_engine_allocation_that_fits_keeps_the_available_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _install_nvidia_smi(monkeypatch, _nvidia_smi_row(used_mib=12_288))
    torch_module = FakeTorch(free_bytes=30 * 1024**3, total_bytes=32 * 1024**3)

    utilization = gpu_memory.cap_gpu_memory_utilization(
        torch_module,
        requested=1.0,
        minimum_utilization_bytes=19 * 1024**3,
    )

    assert utilization == pytest.approx(0.605)


def test_decimal_nvidia_smi_mebibytes_preserve_torch_byte_sizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _install_nvidia_smi(monkeypatch, "0, GPU-test-uuid, 5722.0458984375, 15258.7890625\n")
    torch_module = FakeTorch(free_bytes=10_000_000_000, total_bytes=16_000_000_000)

    assert gpu_memory.gpu_memory_info(torch_module) == (10_000_000_000, 16_000_000_000)


def test_cuda_visible_devices_uuid_selects_the_matching_nvidia_smi_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-other-uuid,GPU-test-uuid")
    commands = _install_nvidia_smi(
        monkeypatch,
        "0, GPU-other-uuid, 0, 32768\n1, GPU-test-uuid, 8192, 32768\n",
    )
    torch_module = FakeTorch(free_bytes=30 * 1024**3, total_bytes=32 * 1024**3, current_device=1)

    free_bytes, total_bytes = gpu_memory.gpu_memory_info(torch_module, 1)

    assert (free_bytes, total_bytes) == (24 * 1024**3, 32 * 1024**3)
    assert commands[0][0] == "nvidia-smi"
    assert "--query-gpu=index,uuid,memory.used,memory.total" in commands[0]


def test_cuda_visible_devices_numeric_index_selects_the_matching_nvidia_smi_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    _install_nvidia_smi(
        monkeypatch,
        "0, GPU-zero, 8192, 32768\n1, GPU-one, 16384, 32768\n2, GPU-two, 0, 32768\n",
    )
    torch_module = FakeTorch(free_bytes=30 * 1024**3, total_bytes=32 * 1024**3, current_device=1)

    free_bytes, total_bytes = gpu_memory.gpu_memory_info(torch_module, torch.device("cuda:1"))

    assert (free_bytes, total_bytes) == (24 * 1024**3, 32 * 1024**3)


@pytest.mark.parametrize(
    ("case", "error_match"),
    [
        ("failed", "nvidia-smi failed.*NVML unavailable"),
        ("timeout", "nvidia-smi timed out"),
        ("non_numeric", "nonnumeric GPU memory data"),
        ("unmatched", "could not uniquely match CUDA logical device"),
    ],
)
def test_nvidia_smi_query_failures_are_loud(
    monkeypatch: pytest.MonkeyPatch, case: str, error_match: str
) -> None:
    if case == "unmatched":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-missing-uuid")
    else:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        del command, kwargs
        if case == "timeout":
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)
        if case == "failed":
            return SimpleNamespace(returncode=1, stdout="", stderr="NVML unavailable")
        if case == "non_numeric":
            return SimpleNamespace(
                returncode=0,
                stdout="0, GPU-test-uuid, N/A, 32768\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout=_nvidia_smi_row(), stderr="")

    monkeypatch.setattr(gpu_memory.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=error_match):
        gpu_memory.gpu_memory_info(FakeTorch(free_bytes=30 * 1024**3, total_bytes=32 * 1024**3))


def test_missing_nvidia_smi_fails_with_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def missing_runner(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu_memory.subprocess, "run", missing_runner)

    with pytest.raises(RuntimeError, match="nvidia-smi is required"):
        gpu_memory.gpu_memory_info(FakeTorch(free_bytes=30 * 1024**3, total_bytes=32 * 1024**3))


def test_no_remaining_vram_reports_estimated_host_held_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _install_nvidia_smi(monkeypatch, _nvidia_smi_row(used_mib=32_256))
    torch_module = FakeTorch(free_bytes=30 * 1024**3, total_bytes=32 * 1024**3)

    with pytest.raises(RuntimeError, match=r"host-held.*29\.50 GiB"):
        gpu_memory.cap_gpu_memory_utilization(torch_module, requested=1.0)


def test_zero_torch_free_memory_fails_with_a_host_held_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _install_nvidia_smi(monkeypatch, _nvidia_smi_row(used_mib=32_768))
    torch_module = FakeTorch(free_bytes=0, total_bytes=32 * 1024**3)

    with pytest.raises(RuntimeError, match=r"estimated at 0\.00 GiB"):
        gpu_memory.cap_gpu_memory_utilization(torch_module, requested=1.0)
