"""Tests for the sweep driver's handling of a point that does not finish.

The sweep's whole reason for running each point in a subprocess is that a configuration which
does not fit should cost one row of the table rather than the rest of the sweep. A point killed
by the per-point timeout has to behave the same way as one killed by the allocator, or a single
slow configuration cancels every point after it and the aggregate summary is never written.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from grpo import throughput_sweep as ts

# Prints before sleeping so the log has a tail to report, as a real point's log would.
OUTLIVES_THE_TIMEOUT = [
    sys.executable,
    "-c",
    "import sys, time; print('generating completions'); sys.stdout.flush(); time.sleep(30)",
]


def sweep_point() -> ts.SweepPoint:
    return ts.SweepPoint(
        name="4b-P2G8-p512c512",
        model_id="Qwen/Qwen3.5-4B",
        prompts_per_step=2,
        group_size=8,
        prompt_tokens=512,
        completion_tokens=512,
    )


@pytest.fixture
def timing_out_point(monkeypatch: pytest.MonkeyPatch) -> ts.SweepPoint:
    monkeypatch.setattr(ts, "free_vram_gib", lambda: 21.5)
    monkeypatch.setattr(ts.SweepPoint, "command", lambda self, json_dir: OUTLIVES_THE_TIMEOUT)
    return sweep_point()


def test_a_timed_out_point_becomes_a_row_rather_than_an_exception(
    timing_out_point: ts.SweepPoint, tmp_path: Path
) -> None:
    row = ts.run_point(timing_out_point, tmp_path, timeout_s=1)
    assert row["status"] == "timeout", (
        "a killed child has no OutOfMemoryError in its log, so classify_failure would call it "
        "'failed' and hide why the point died"
    )
    assert row["point"] == "4b-P2G8-p512c512"
    assert row["stderr_tail"] == ["generating completions"]
    assert float(row["wall_seconds"]) >= 1.0  # pyright: ignore[reportArgumentType]


def test_the_sweep_carries_on_past_a_timed_out_point(
    timing_out_point: ts.SweepPoint, tmp_path: Path
) -> None:
    """The failure this pins: one timeout used to raise out of the row comprehension."""
    rows = [ts.run_point(timing_out_point, tmp_path, timeout_s=1) for _ in range(2)]
    assert [row["status"] for row in rows] == ["timeout", "timeout"]
    table = ts.format_sweep_table(rows)
    assert table.count("timeout") == 2
