"""Tests for the pre-Batch smoke's summary assembly.

The smoke exists to answer two questions before a Batch run: does reward move, and what does
each dtype cost in memory and time. Both of its headline numbers were assembled in a way that
could not report failure. A run whose trainer logs carried no reward column serialized a summary
of nulls, logged RESULT and returned 0 -- three artifacts on disk look exactly like that. And
`s_per_step` divided total wall clock by the step count, charging three tokenizer loads, both
dataset renders, the model load and the checkpoint save to the step time, which biases the dtype
comparison against whichever dtype loads slowest. 4-bit is that dtype.
"""

from __future__ import annotations

import pandas as pd
import pytest

from grpo import smoke

REWARDS = (0.1, 0.4, 0.6, 0.5)


def trainer_log_frame(
    *,
    reward_column: str = "reward",
    train_runtime: float | None = 100.0,
) -> pd.DataFrame:
    """A frame shaped like trainer_state.json: per-step rows, then one end-of-training row."""
    rows: list[dict[str, float]] = [
        {"step": i + 1, reward_column: r, "frac_reward_zero_std": 0.25}
        for i, r in enumerate(REWARDS)
    ]
    final: dict[str, float] = {"step": len(REWARDS), "train_loss": 0.02}
    if train_runtime is not None:
        final["train_runtime"] = train_runtime
    rows.append(final)
    return pd.DataFrame(rows)


def build(df: pd.DataFrame, *, steps: int = 10, elapsed: float = 130.0) -> dict[str, object]:
    return smoke.build_smoke_summary(
        df,
        dtype="bfloat16",
        liger=True,
        steps=steps,
        batch=8,
        num_generations=8,
        elapsed=elapsed,
        peak_alloc_gib=10.62,
        peak_reserved_gib=14.64,
    )


def test_reward_fields_come_from_trls_own_reward_column() -> None:
    summary = build(trainer_log_frame())
    assert summary["reward_first"] == 0.1
    assert summary["reward_last"] == 0.5
    assert summary["reward_max"] == 0.6
    assert summary["frac_reward_zero_std_mean"] == 0.25


def test_a_run_that_recorded_no_reward_is_refused() -> None:
    """Serializing nulls here is how three smoke runs reported success having measured nothing."""
    with pytest.raises(RuntimeError, match="no reward"):
        build(trainer_log_frame(reward_column="reward_rate"))


def test_seconds_per_step_excludes_setup_and_save() -> None:
    """Trainer already reports the honest training time; wall clock also holds load and save."""
    summary = build(trainer_log_frame(train_runtime=100.0), steps=10, elapsed=130.0)
    assert summary["s_per_step"] == 10.0
    assert summary["train_runtime_s"] == 100.0
    assert summary["setup_and_save_s"] == 30.0
    assert summary["wall_clock_s"] == 130.0, "old artifacts stay comparable on wall clock"


def test_a_run_without_a_train_runtime_record_is_refused() -> None:
    with pytest.raises(RuntimeError, match="train_runtime"):
        build(trainer_log_frame(train_runtime=None))
