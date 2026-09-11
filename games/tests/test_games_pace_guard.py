"""The step-pace tripwire: a run pacing like the removed slow path must die, loudly, by step 3.

The incident these encode: on 2026-08-25 a stray environment export put a paid arm on
transformers.generate at ~72 minutes per step against the ~13.5 the plan costed, and nothing in
the run said so -- the trainer, the logs and the heartbeat all stayed green while the projection
went 5x over budget. The environment guard closes that one cause; this guard reads the symptom,
so it catches the whole class of silently-slow backends that do not exist yet.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from games import pace_guard
from games.pace_guard import (
    COLOCATE_PLAN_STEP_MINUTES,
    PACE_TOLERANCE_FACTOR,
    PACE_WINDOW_STEPS,
    PLAN_STEP_MINUTES_ENV,
    StepPaceGuardCallback,
    assert_sane_step_pace,
    plan_step_seconds,
)

PLAN_SECONDS = COLOCATE_PLAN_STEP_MINUTES * 60.0
INCIDENT_STEP_SECONDS = 72.0 * 60.0


class TestThePureCheck:
    def test_the_incident_pace_trips_at_exactly_the_window(self):
        """Three steps at the measured slow-path rate is the earliest the guard can know."""
        with pytest.raises(RuntimeError, match="slow generation path"):
            assert_sane_step_pace(
                [INCIDENT_STEP_SECONDS] * PACE_WINDOW_STEPS, plan_seconds=PLAN_SECONDS
            )

    def test_the_refusal_names_the_incident_figures_and_the_override(self):
        slow = [INCIDENT_STEP_SECONDS] * PACE_WINDOW_STEPS
        with pytest.raises(RuntimeError, match="~72 minutes per step"):
            assert_sane_step_pace(slow, plan_seconds=PLAN_SECONDS)
        with pytest.raises(RuntimeError, match=r"~13\.5"):
            assert_sane_step_pace(slow, plan_seconds=PLAN_SECONDS)
        with pytest.raises(RuntimeError, match=PLAN_STEP_MINUTES_ENV):
            assert_sane_step_pace(slow, plan_seconds=PLAN_SECONDS)

    def test_the_planned_pace_passes_with_a_slow_warmup_step(self):
        """The first step carries engine boot and CUDA graph capture; 2.5x must absorb it."""
        assert_sane_step_pace(
            [2.0 * PLAN_SECONDS, PLAN_SECONDS, PLAN_SECONDS], plan_seconds=PLAN_SECONDS
        )

    def test_fewer_steps_than_the_window_are_never_judged(self):
        """One slow step is weather; the guard waits for the window to fill."""
        assert_sane_step_pace(
            [INCIDENT_STEP_SECONDS] * (PACE_WINDOW_STEPS - 1), plan_seconds=PLAN_SECONDS
        )

    def test_the_window_trails_so_an_early_slow_patch_is_forgiven(self):
        """Only the last `PACE_WINDOW_STEPS` steps are judged: a run that recovered stays alive."""
        recovered = [INCIDENT_STEP_SECONDS] * 2 + [PLAN_SECONDS] * PACE_WINDOW_STEPS
        assert_sane_step_pace(recovered, plan_seconds=PLAN_SECONDS)

    def test_the_boundary_is_the_named_tolerance_factor(self):
        """Just under the line lives, just over dies -- the factor, not a baked-in threshold."""
        limit = PACE_TOLERANCE_FACTOR * PLAN_SECONDS
        assert_sane_step_pace([limit * 0.99] * PACE_WINDOW_STEPS, plan_seconds=PLAN_SECONDS)
        with pytest.raises(RuntimeError, match="slow generation path"):
            assert_sane_step_pace([limit * 1.01] * PACE_WINDOW_STEPS, plan_seconds=PLAN_SECONDS)


class TestThePlanFigure:
    def test_the_default_is_the_measured_colocate_pace(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(PLAN_STEP_MINUTES_ENV, raising=False)
        assert plan_step_seconds() == pytest.approx(13.5 * 60.0)

    def test_a_shape_specific_figure_comes_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A 9B arm paces differently; its launch exports its own plan instead of editing code."""
        monkeypatch.setenv(PLAN_STEP_MINUTES_ENV, "28")
        assert plan_step_seconds() == pytest.approx(28 * 60.0)

    @pytest.mark.parametrize("value", ["0", "-3", "fast"])
    def test_a_meaningless_figure_is_refused_not_defaulted(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ):
        monkeypatch.setenv(PLAN_STEP_MINUTES_ENV, value)
        with pytest.raises(ValueError, match=PLAN_STEP_MINUTES_ENV):
            plan_step_seconds()


class TestTheCallbackShell:
    """Drive `on_step_end` through fake clocks: the wiring, not the arithmetic, is under test."""

    @staticmethod
    def tick(callback: StepPaceGuardCallback, monkeypatch: pytest.MonkeyPatch, at: float) -> None:
        """Advance the fake clock to `at` and fire one optimizer-step boundary."""
        monkeypatch.setattr(pace_guard.time, "monotonic", lambda: at)
        callback.on_step_end(
            SimpleNamespace(),  # pyright: ignore[reportArgumentType]
            SimpleNamespace(global_step=len(callback.step_seconds) + 1),  # pyright: ignore[reportArgumentType]
            SimpleNamespace(),  # pyright: ignore[reportArgumentType]
        )

    def start(self, monkeypatch: pytest.MonkeyPatch, plan_seconds: float) -> StepPaceGuardCallback:
        callback = StepPaceGuardCallback(plan_seconds=plan_seconds)
        monkeypatch.setattr(pace_guard.time, "monotonic", lambda: 0.0)
        callback.on_train_begin(
            SimpleNamespace(),  # pyright: ignore[reportArgumentType]
            SimpleNamespace(global_step=0),  # pyright: ignore[reportArgumentType]
            SimpleNamespace(),  # pyright: ignore[reportArgumentType]
        )
        return callback

    def test_a_run_pacing_like_the_incident_dies_on_its_third_step(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        callback = self.start(monkeypatch, plan_seconds=PLAN_SECONDS)
        self.tick(callback, monkeypatch, at=INCIDENT_STEP_SECONDS)
        self.tick(callback, monkeypatch, at=2 * INCIDENT_STEP_SECONDS)
        with pytest.raises(RuntimeError, match="slow generation path"):
            self.tick(callback, monkeypatch, at=3 * INCIDENT_STEP_SECONDS)

    def test_a_run_at_the_planned_pace_survives_many_steps(self, monkeypatch: pytest.MonkeyPatch):
        callback = self.start(monkeypatch, plan_seconds=PLAN_SECONDS)
        for step in range(1, 11):
            self.tick(callback, monkeypatch, at=step * PLAN_SECONDS)
        assert len(callback.step_seconds) == 10

    def test_the_default_plan_figure_reaches_the_callback(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(PLAN_STEP_MINUTES_ENV, raising=False)
        assert StepPaceGuardCallback().plan_seconds == pytest.approx(PLAN_SECONDS)

    def test_an_exported_plan_figure_reaches_the_callback(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(PLAN_STEP_MINUTES_ENV, "28")
        assert StepPaceGuardCallback().plan_seconds == pytest.approx(28 * 60.0)
