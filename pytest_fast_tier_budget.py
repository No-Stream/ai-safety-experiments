"""Warn when an unexcluded test exceeds the configured fast-tier time budget."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from _pytest.reports import TestReport


_BUDGET_ENV_VAR = "FAST_TIER_BUDGET_SECONDS"
_EXCLUDED_MARKERS = frozenset({"slow", "jail_real", "docker_real"})
_MARKERS_USER_PROPERTY = "fast_tier_budget_exclusion_markers"
_TIMED_PHASES = frozenset({"setup", "call", "teardown"})


@dataclass
class _TestTiming:
    total_seconds: float = 0.0
    exclusion_markers: set[str] = field(default_factory=set)


def _read_budget_seconds() -> float | None:
    raw_budget = os.environ.get(_BUDGET_ENV_VAR)
    if raw_budget is None:
        return None

    try:
        budget_seconds = float(raw_budget)
    except ValueError as error:
        raise pytest.UsageError(f"{_BUDGET_ENV_VAR} must be a positive finite number") from error

    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise pytest.UsageError(f"{_BUDGET_ENV_VAR} must be a positive finite number")
    return budget_seconds


class _FastTierBudgetPlugin:
    def __init__(self, budget_seconds: float | None) -> None:
        self._budget_seconds = budget_seconds
        self._timings: dict[str, _TestTiming] = {}

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item) -> Iterator[None]:
        outcome = yield
        if self._budget_seconds is None:
            return

        report = outcome.get_result()
        exclusion_markers = sorted(
            {marker.name for marker in item.iter_markers() if marker.name in _EXCLUDED_MARKERS}
        )
        report.user_properties.append((_MARKERS_USER_PROPERTY, ",".join(exclusion_markers)))

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        if self._budget_seconds is None or report.when not in _TIMED_PHASES:
            return

        marker_properties = [
            value for name, value in report.user_properties if name == _MARKERS_USER_PROPERTY
        ]
        if len(marker_properties) != 1 or not isinstance(marker_properties[0], str):
            raise RuntimeError(
                f"pytest report for {report.nodeid} is missing fast-tier marker metadata"
            )

        timing = self._timings.setdefault(report.nodeid, _TestTiming())
        timing.total_seconds += report.duration
        timing.exclusion_markers.update(
            marker for marker in marker_properties[0].split(",") if marker
        )

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        if self._budget_seconds is None:
            return

        over_budget = sorted(
            (
                (nodeid, timing.total_seconds)
                for nodeid, timing in self._timings.items()
                if not timing.exclusion_markers and timing.total_seconds > self._budget_seconds
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not over_budget:
            return

        terminalreporter.write_sep("=", "FAST TIER BUDGET WARNING")
        terminalreporter.write_line(
            f"Tests above {_BUDGET_ENV_VAR}={self._budget_seconds:g}s "
            "(summed setup, call, and teardown time):"
        )
        for nodeid, total_seconds in over_budget:
            terminalreporter.write_line(f"  {nodeid}: {total_seconds:.2f}s")


def pytest_configure(config: pytest.Config) -> None:
    """Register the report collector in both xdist workers and the controller."""
    config.pluginmanager.register(
        _FastTierBudgetPlugin(_read_budget_seconds()),
        "fast-tier-budget-collector",
    )
