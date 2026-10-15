"""Generation stage timing aggregation uses persisted episode and turn evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from reward_hacking.generation_perf import record_episode_timings
from reward_hacking.trace import write_trace

if TYPE_CHECKING:
    from pathlib import Path

    from grpo.gpu_perf import PerfMonitor


class TimingMonitor:
    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self.mean_context: float | None = None

    def add_time(self, category: str, seconds: float) -> None:
        self.times[category] = self.times.get(category, 0.0) + seconds

    def set_mean_context(self, tokens: float) -> None:
        self.mean_context = tokens


def test_episode_timing_uses_completion_weighted_live_context(tmp_path: Path) -> None:
    trace_path = tmp_path / "episode.jsonl.gz"
    write_trace(
        trace_path,
        [
            {
                "record": "turn",
                "prompt_tokens": 100,
                "output_tokens": 20,
            },
            {
                "record": "turn",
                "prompt_tokens": 200,
                "output_tokens": 40,
            },
        ],
        compact=True,
    )
    record: dict[str, Any] = {
        "trace_path": str(trace_path),
        "trace_summary": {
            "model_call_seconds": 3.5,
            "tool_execution_seconds": 7.0,
            "grading_seconds": 1.5,
        },
    }
    monitor = TimingMonitor()

    record_episode_timings(cast("PerfMonitor", monitor), [record])

    assert monitor.times == {
        "model_call_seconds": 3.5,
        "tool_execution_seconds": 7.0,
        "grading_seconds": 1.5,
    }
    assert monitor.mean_context == 550 / 3
