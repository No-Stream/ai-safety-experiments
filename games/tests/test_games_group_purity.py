"""Offline tests for the group-purity metric that replaces TRL's dead `frac_reward_zero_std`.

The sabotage this file exists for: a group whose rewards are all identical carries no gradient, and
the metric must say so. TRL's field reported 0.0000 through three 70-step arms while that condition
went unmeasured, so the replacement is tested against the exact case it failed on.
"""

from __future__ import annotations

import pytest

from games.rewards import _Scored, group_purity


def scored(*rewards: float) -> list[_Scored]:
    """Build scored completions carrying only the rewards, which is all purity depends on."""
    return [_Scored(reward=r, parsed=True, detail="C") for r in rewards]


class TestGroupPurity:
    def test_an_all_identical_group_is_pure(self):
        # The sabotage case: every reward equal, so every advantage is zero whatever the value.
        assert group_purity(scored(0.6, 0.6, 0.6, 0.6), 4) == pytest.approx(1.0)

    def test_a_group_with_any_disagreement_is_not_pure(self):
        assert group_purity(scored(0.6, 0.6, 0.6, 0.2), 4) == pytest.approx(0.0)

    def test_purity_is_a_fraction_over_groups(self):
        # Two groups of four: the first pure, the second mixed.
        rewards = scored(0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.6, 0.2)
        assert group_purity(rewards, 4) == pytest.approx(0.5)

    def test_a_penalty_only_group_is_pure_too(self):
        # A batch where nothing parsed is uniformly -1.0: no signal, and the metric must not hide it
        # behind the fact that the rewards are "informative" in some other sense.
        assert group_purity(scored(-1.0, -1.0, -1.0, -1.0), 4) == pytest.approx(1.0)

    def test_a_group_mixing_a_penalty_with_a_payoff_is_not_pure(self):
        assert group_purity(scored(-1.0, 0.6, 0.6, 0.6), 4) == pytest.approx(0.0)

    def test_tiny_differences_still_count_as_disagreement(self):
        # Exact equality is deliberate: TRL's isclose tolerance is part of why its metric was dead.
        assert group_purity(scored(0.6, 0.6, 0.6, 0.6000001), 4) == pytest.approx(0.0)

    def test_every_group_pure_reads_as_total_collapse(self):
        rewards = scored(0.2, 0.2, 0.6, 0.6)
        assert group_purity(rewards, 2) == pytest.approx(1.0)

    def test_a_nonpositive_group_size_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            group_purity(scored(0.6, 0.6), 0)

    def test_no_groups_at_all_is_refused(self):
        with pytest.raises(ValueError, match="no groups"):
            group_purity([], 4)
