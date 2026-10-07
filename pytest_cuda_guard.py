"""Pytest plugin (loaded via addopts) that fails a CPU-only run which initialised CUDA.

A worker that opens a CUDA context keeps it until the suite ends, and scripts/gpu_preflight.py then
refuses GPU work from every other session for the length of the gate.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="session", autouse=True)
def cpu_test_session_leaves_cuda_uninitialized() -> Iterator[None]:
    """Fail after any CPU test initializes a CUDA context; explicit GPU runs opt out."""
    yield
    if os.environ.get("GAMES_GPU_TESTS") == "1":
        return
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        pytest.fail(
            "a CPU-only pytest test initialized CUDA; keep CUDA_VISIBLE_DEVICES empty or gate the GPU test"
        )
