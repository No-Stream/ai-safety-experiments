"""Crash a training run whose realized step pace says generation took a slow path.

The environment guard in `games.generation` refuses the one slow path we have already paid for
(a stray ``GAMES_VLLM_COLOCATE=0`` put an arm on transformers.generate at ~72 minutes per step
against the ~13.5 the plan costed, 2026-08-25 -- caught only by a human pace diagnosis, $28 in).
This module is the belt for that suspenders: it does not know *why* a run is slow, it only knows
what the plan said a step costs, and it kills the run once the realized pace says the plan is
wrong by more than any legitimate variation. That catches the whole class -- a future backend
regression, an engine that silently fell back to eager, a card throttling -- not just the one
env var.

The checking logic is a pure function over a list of floats so it is unit-testable without a
trainer; `StepPaceGuardCallback` is the shell that feeds it wall clock. Games-only on purpose:
`reward_hacking.train` shares `games.train.build_callbacks` and keeps a deliberate slow path
behind an explicit `--allow-hf-generation` acknowledgment, which this guard would kill.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, cast

from transformers import TrainerCallback

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

# The measured colocate pace at the production shape (Qwen3.5-2B, thinking on, 64 episodes x
# 32,768 tokens, 2026-08-19); a different shape exports its own figure instead of editing this.
PLAN_STEP_MINUTES_ENV = "GAMES_PLAN_STEP_MINUTES"
COLOCATE_PLAN_STEP_MINUTES = 13.5
# 2.5x absorbs a slow first step (engine boot, CUDA graph capture) and ordinary variance; the
# failure class this exists for sits at ~5.3x (72 over 13.5). Multiplied at the check, never
# pre-baked into a per-box threshold.
PACE_TOLERANCE_FACTOR = 2.5
# Trailing-window size, and so also the first step the check can fire at: one slow step is
# weather, three in a row is the backend.
PACE_WINDOW_STEPS = 3


def plan_step_seconds() -> float:
    """Return the planned per-step cost in seconds, from the environment or the measured default."""
    raw = os.environ.get(PLAN_STEP_MINUTES_ENV)
    if raw is None:
        return COLOCATE_PLAN_STEP_MINUTES * 60.0
    try:
        minutes = float(raw)
    except ValueError as error:
        raise ValueError(
            f"{PLAN_STEP_MINUTES_ENV}={raw!r} is not a positive number of minutes per step"
        ) from error
    if minutes <= 0:
        raise ValueError(
            f"{PLAN_STEP_MINUTES_ENV}={raw!r} is not a positive number of minutes per step"
        )
    return minutes * 60.0


def assert_sane_step_pace(step_seconds: Sequence[float], *, plan_seconds: float) -> None:
    """Refuse to continue a run whose trailing mean step time betrays a slow generation path.

    Checked from the callback on every step end; quiet until `PACE_WINDOW_STEPS` measurements
    exist, then a hard raise the trainer cannot survive. Killing the run is the point: at the
    incident's ~72 minutes per step, every additional step costs more than the restart the crash
    forces, and a warning is what nobody read the first time.
    """
    if len(step_seconds) < PACE_WINDOW_STEPS:
        return
    window = list(step_seconds[-PACE_WINDOW_STEPS:])
    trailing_mean = sum(window) / len(window)
    limit = PACE_TOLERANCE_FACTOR * plan_seconds
    if trailing_mean <= limit:
        return
    window_minutes = [f"{seconds / 60:.1f}" for seconds in window]
    raise RuntimeError(
        f"step pace betrays a slow generation path: the last {PACE_WINDOW_STEPS} steps took "
        f"{window_minutes} minutes (trailing mean {trailing_mean / 60:.1f}) against a plan of "
        f"{plan_seconds / 60:.1f} minutes per step -- past the {PACE_TOLERANCE_FACTOR}x line "
        f"this guard kills at. This is the failure class that put an arm on transformers.generate "
        f"at ~72 minutes per step against the ~13.5 planned (2026-08-25, caught by a human $28 "
        f"in; rollouts are vLLM-only by owner decision 2026-08-26). If the plan figure is what "
        f"is wrong, export {PLAN_STEP_MINUTES_ENV} with this shape's measured pace and relaunch."
    )


class StepPaceGuardCallback(TrainerCallback):
    """Time every optimizer step and kill the run once the trailing pace betrays the backend.

    Wall clock via `time.monotonic` between step boundaries, deliberately without a CUDA
    synchronize: whatever GPU work is still queued at one `on_step_end` drains inside the next
    interval, so over the three-step window the sum is the real wall clock whichever phase
    dominates -- and at the production shape generation does not: it is about a quarter of the
    step (183 s of 761 s at 2B, efficiency audit 2026-09-02) and the training pass is two thirds.
    A 2.5x guard does not need sub-second timing; per-phase attribution is
    `grpo.throughput.StepPhaseTimer`'s job. A raise from a callback propagates out of
    `Trainer.train`, which is the point -- see `assert_sane_step_pace` for why a crash beats a
    warning here.
    """

    def __init__(self, plan_seconds: float | None = None) -> None:
        """Take the plan figure, defaulting from the environment reader."""
        self.plan_seconds = plan_step_seconds() if plan_seconds is None else plan_seconds
        self.step_seconds: list[float] = []
        self._last_t: float | None = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Start the clock at the training-loop boundary, announcing the line the guard kills at."""
        del args, state, control, kwargs
        logger.info(
            "pace guard armed, %s",
            f"plan={self.plan_seconds / 60:.1f} min/step "
            f"kill_line={PACE_TOLERANCE_FACTOR * self.plan_seconds / 60:.1f} min "
            f"(trailing mean of {PACE_WINDOW_STEPS})",
        )
        self._last_t = time.monotonic()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Record one step's wall clock and check the trailing window."""
        del args, control, kwargs
        now = time.monotonic()
        elapsed = now - cast("float", self._last_t)
        self._last_t = now
        self.step_seconds.append(elapsed)
        logger.info(
            "pace guard: step %d took %.1f min (plan %.1f)",
            int(state.global_step),
            elapsed / 60,
            self.plan_seconds / 60,
        )
        assert_sane_step_pace(self.step_seconds, plan_seconds=self.plan_seconds)
