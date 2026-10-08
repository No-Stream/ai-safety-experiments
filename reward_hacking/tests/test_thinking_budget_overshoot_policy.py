"""Run-level thresholds for recorded thinking-budget failures."""

import pytest

from reward_hacking.ladder.thinking import ThinkingBudgetOvershootPolicy


@pytest.mark.parametrize(("finished", "overshoots"), [(1, 1), (2, 2), (30, 3), (100, 3), (20, 0)])
def test_rare_overshoots_do_not_abort(finished: int, overshoots: int) -> None:
    ThinkingBudgetOvershootPolicy().check(finished, overshoots)


def test_repeated_overshoots_abort_with_counts_and_limit() -> None:
    with pytest.raises(RuntimeError, match=r"thinking_budget_overshoot.*3/20.*0.1"):
        ThinkingBudgetOvershootPolicy().check(20, 3)


def test_custom_threshold() -> None:
    policy = ThinkingBudgetOvershootPolicy(max_fraction=0.5, min_episodes=4)
    policy.check(8, 4)
    with pytest.raises(RuntimeError, match="thinking_budget_overshoot"):
        policy.check(7, 4)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_fraction_fails(fraction: float) -> None:
    with pytest.raises(ValueError, match="max_fraction"):
        ThinkingBudgetOvershootPolicy(max_fraction=fraction)


@pytest.mark.parametrize("minimum", [0, -1, True])
def test_invalid_minimum_fails(minimum: int) -> None:
    with pytest.raises(ValueError, match="min_episodes"):
        ThinkingBudgetOvershootPolicy(min_episodes=minimum)
