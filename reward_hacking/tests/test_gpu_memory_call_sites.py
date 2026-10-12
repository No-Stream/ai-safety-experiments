"""Focused CPU checks that reward-hacking GPU sizing uses the shared WSL-aware helpers."""

from __future__ import annotations

from typing import Any

import pytest

from reward_hacking.atlas import sampling as atlas_sampling
from reward_hacking.organisms import aisi_codecontests


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def get_device_name(self, device: int) -> str:
        assert device == 0
        return "synthetic-cuda"


class FakeTorch:
    cuda = FakeCuda()


@pytest.mark.parametrize(
    "settings_function",
    [
        pytest.param(atlas_sampling._gpu_engine_settings, id="atlas-sampling"),
        pytest.param(aisi_codecontests._gemma_gpu_memory_utilization, id="aisi-codecontests"),
    ],
)
def test_vllm_sizing_uses_corrected_memory_and_shared_cap(
    monkeypatch: pytest.MonkeyPatch,
    settings_function: Any,
) -> None:
    calls: list[tuple[str, object]] = []

    def capped_utilization(
        torch_module: object,
        requested: float,
        device: int | None = None,
        *,
        margin_fraction: float = 0.02,
    ) -> float:
        assert torch_module is FakeTorch
        assert requested == 1.0
        assert device == 0
        assert margin_fraction == pytest.approx(0.02)
        calls.append(("cap", requested))
        return 0.58

    for module in (atlas_sampling, aisi_codecontests):
        monkeypatch.setattr(module, "cap_gpu_memory_utilization", capped_utilization)

    utilization, device_name = settings_function(FakeTorch)

    assert utilization == pytest.approx(0.58)
    assert device_name == "synthetic-cuda"
    assert calls == [("cap", 1.0)]
