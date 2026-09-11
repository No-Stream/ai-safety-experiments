"""Pin the payoff ground truth: every expected number here is hand-computed, not regenerated.

Offline and CPU-only -- no model, no GPU, no network. These are the numbers the reward function
multiplies, so a literal in this file is the specification and the code is what has to agree
with it. Where a value is a fraction (the temptation-10 variants normalise by 13 and 21), the
literal is written as the division that produced it so the arithmetic stays readable.

The test with teeth is :class:`TestBackwardInductionAgainstTitForTat`. Against a conditional
opponent, cooperating until the final round must strictly beat both pure strategies, because
that gap is the whole reason the iterated arm can show end-game defection emerging; if the
payoffs ever stopped supporting it, the arm would be measuring nothing and the plots would
still look fine. :class:`TestTemptationMagnitudeChangesTheOptimalPolicy` guards the flip side:
at temptation-10 the optimum is alternating exploitation instead, which is why the iterated
arms are pinned to temptation-2.
"""

from __future__ import annotations

import pytest

from games import rewards
from games.payoffs import (
    COOPERATE,
    DEFECT,
    MIN_EFFORT_BENEFIT_PER_LEVEL,
    MIN_EFFORT_LEVELS,
    MIN_EFFORT_MATCHER_OPENING_LEVEL,
    STAG_HUNT_VARIANTS,
    TRUST_ENDOWMENT,
    TRUST_MAX_SELF_STATED_RETURN_FRACTION,
    TRUST_MAX_STATED_RETURN_FRACTION,
    TRUST_MULTIPLIER,
    DictatorSpec,
    MatrixGameSpec,
    MinEffortSpec,
    OpponentRule,
    TrustSpec,
    assert_care_alpha,
    assert_chicken,
    assert_constant_sum,
    assert_defective_coordination,
    assert_defective_harmony,
    assert_dominant_defection,
    assert_harmony,
    assert_hi_lo,
    assert_prisoners_dilemma,
    assert_stag_hunt,
    assert_trust_care_spec,
    care_reward_spread,
    chicken,
    defective_coordination,
    defective_harmony,
    expected_care_payoff,
    expected_counterpart_payoff,
    expected_joint_payoff,
    fixed_pie_pd,
    group_mix_fixed_point,
    group_mix_reward_spread,
    harmony,
    hi_lo,
    joint_welfare_gap_crossing,
    joint_welfare_reward_spread,
    max_iterated_return,
    max_min_effort_match_return,
    min_effort_cell_reward,
    opponent_moves,
    other_payoff_reward_spread,
    public_goods,
    simulate_iterated,
    stag_hunt,
    stag_hunt_cooperation_threshold,
    stated_match_crossover,
    stated_match_expected_payoff,
    stated_match_gap,
    stated_match_optimal_action,
    trust_care_corner_rewards,
    trust_self_rule_corner_rewards,
    trust_stated_rule_corner_rewards,
    trustee_payoff,
    trustor_care_reward,
    trustor_payoff,
    trustor_reward,
    trustor_reward_spread,
    trustor_self_rule_reward_span,
    twin_pd,
    ultimatum_responder,
    worst_iterated_return,
    worst_min_effort_match_return,
)


@pytest.fixture
def twin() -> MatrixGameSpec:
    return twin_pd()


class TestMatrixGameSpecCellLookup:
    def test_each_cell_reads_back(self) -> None:
        spec = MatrixGameSpec(
            game_id="probe", payoff_cc=0.1, payoff_cd=0.2, payoff_dc=0.3, payoff_dd=0.4
        )
        assert spec.payoff(COOPERATE, COOPERATE) == 0.1
        assert spec.payoff(COOPERATE, DEFECT) == 0.2
        assert spec.payoff(DEFECT, COOPERATE) == 0.3
        assert spec.payoff(DEFECT, DEFECT) == 0.4

    def test_unknown_action_raises(self, twin: MatrixGameSpec) -> None:
        with pytest.raises(ValueError, match="Actions must be one of"):
            twin.payoff("X", COOPERATE)
        with pytest.raises(ValueError, match="Actions must be one of"):
            twin.payoff(COOPERATE, "cooperate")

    def test_lowercase_canonical_action_is_not_accepted(self, twin: MatrixGameSpec) -> None:
        with pytest.raises(ValueError, match="Actions must be one of"):
            twin.payoff("c", "d")

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 5.0])
    def test_payoff_outside_the_unit_range_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="Payoffs must lie in"):
            MatrixGameSpec(
                game_id="probe", payoff_cc=bad, payoff_cd=0.0, payoff_dc=1.0, payoff_dd=0.2
            )

    def test_empty_game_id_raises(self) -> None:
        with pytest.raises(ValueError, match="game_id must be non-empty"):
            MatrixGameSpec(game_id="", payoff_cc=0.6, payoff_cd=0.0, payoff_dc=1.0, payoff_dd=0.2)


class TestTwinPdCells:
    def test_temptation_2_is_the_normalised_textbook_game(self, twin: MatrixGameSpec) -> None:
        """Classic 3/0/5/1 divided by the largest cell, 5."""
        assert twin.payoff_cc == pytest.approx(0.6)
        assert twin.payoff_cd == pytest.approx(0.0)
        assert twin.payoff_dc == pytest.approx(1.0)
        assert twin.payoff_dd == pytest.approx(0.2)
        assert twin.game_id == "twin-pd-temptation-2"

    def test_temptation_10_keeps_strict_pd_ordering(self) -> None:
        spec = twin_pd("temptation-10")
        # Raw cells 3/0/13/1 normalised by 13.
        assert spec.payoff_cc == pytest.approx(3 / 13)
        assert spec.payoff_cd == pytest.approx(0.0)
        assert spec.payoff_dc == pytest.approx(1.0)
        assert spec.payoff_dd == pytest.approx(1 / 13)
        assert_prisoners_dilemma(spec)

    def test_unknown_variant_raises(self) -> None:
        # temptation-0, not a plausible future rung: a zero temptation cannot satisfy DC > CC.
        with pytest.raises(ValueError, match="Unknown payoff_variant"):
            twin_pd("temptation-0")


class TestFixedPiePdIsConstantSum:
    def test_temptation_2_cells(self) -> None:
        spec = fixed_pie_pd()
        # Raw cells CC=DD=3, DC=5, CD=1 (a pie of 6 in every cell) normalised by 5.
        assert spec.payoff_cc == pytest.approx(0.6)
        assert spec.payoff_cd == pytest.approx(0.2)
        assert spec.payoff_dc == pytest.approx(1.0)
        assert spec.payoff_dd == pytest.approx(0.6)
        assert spec.game_id == "fixed-pie-pd-temptation-2"

    def test_temptation_10_cells(self) -> None:
        spec = fixed_pie_pd("temptation-10")
        # Raw cells CC=DD=11, DC=21, CD=1 (a pie of 22) normalised by 21.
        assert spec.payoff_cc == pytest.approx(11 / 21)
        assert spec.payoff_cd == pytest.approx(1 / 21)
        assert spec.payoff_dc == pytest.approx(1.0)
        assert spec.payoff_dd == pytest.approx(11 / 21)

    @pytest.mark.parametrize("variant", ["temptation-2", "temptation-10"])
    def test_defection_dominates_while_the_pie_stays_fixed(self, variant: str) -> None:
        """Constant sum forces CC == DD in a symmetric game, so this is not a strict PD.

        That trade is deliberate and documented on the constructor: dominance is the property
        the training gradient reads, and it survives.
        """
        spec = fixed_pie_pd(variant)
        assert_constant_sum(spec)
        assert_dominant_defection(spec)
        assert spec.payoff_cc == pytest.approx(spec.payoff_dd)
        with pytest.raises(ValueError, match="not a strict prisoner's dilemma"):
            assert_prisoners_dilemma(spec)

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown payoff_variant"):
            fixed_pie_pd("temptation-0")


class TestTheOtherGames:
    def test_stag_hunt_cells(self) -> None:
        spec = stag_hunt()
        # Plan cells CC=4, CD=0, DC=3, DD=3 normalised by 4.
        assert spec.payoff_cc == pytest.approx(1.0)
        assert spec.payoff_cd == pytest.approx(0.0)
        assert spec.payoff_dc == pytest.approx(0.75)
        assert spec.payoff_dd == pytest.approx(0.75)
        assert_stag_hunt(spec)

    def test_stag_hunt_does_not_make_defection_dominant(self) -> None:
        with pytest.raises(ValueError, match="does not make defection strictly dominant"):
            assert_dominant_defection(stag_hunt())

    def test_the_risky_variant_moves_the_risk_dominance_boundary(self) -> None:
        """Raw CC=4, CD=0, DC=DD=3.8 over 4: the safe option is nearly as good as hunting.

        Both variants stay stag hunts (CC > DC >= DD > CD); what changes is how confident a player
        must be in its counterpart, which is the contrast the second variant exists to create.
        """
        risky = stag_hunt("risky-hunt")
        assert risky.payoff_cc == pytest.approx(1.0)
        assert risky.payoff_cd == pytest.approx(0.0)
        assert risky.payoff_dc == pytest.approx(0.95)
        assert risky.payoff_dd == pytest.approx(0.95)
        assert_stag_hunt(risky)

    def test_the_two_variants_differ_after_normalisation(self) -> None:
        """The point of picking a different ratio: scaling every cell would have been a no-op.

        Normalising by the largest cell divides a uniform scale factor straight back out, so a
        "bigger stakes" variant would have produced a byte-identical spec.
        """
        safe = stag_hunt("safe-hunt")
        risky = stag_hunt("risky-hunt")
        assert safe != risky
        assert stag_hunt_cooperation_threshold(safe) == pytest.approx(0.75)
        assert stag_hunt_cooperation_threshold(risky) == pytest.approx(0.95)
        scaled = stag_hunt("safe-hunt")
        assert scaled == safe

    def test_an_unknown_stag_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stag-hunt payoff_variant"):
            stag_hunt("enormous-stakes")

    def test_the_default_variant_is_the_original_cells(self) -> None:
        assert stag_hunt() == stag_hunt("safe-hunt")

    def test_chicken_cells(self) -> None:
        spec = chicken()
        # Raw cells CC=3, CD=1, DC=4, DD=0 normalised by 4.
        assert spec.payoff_cc == pytest.approx(0.75)
        assert spec.payoff_cd == pytest.approx(0.25)
        assert spec.payoff_dc == pytest.approx(1.0)
        assert spec.payoff_dd == pytest.approx(0.0)
        assert_chicken(spec)

    def test_public_goods_reduces_to_a_prisoners_dilemma(self) -> None:
        """Unit endowments, multiplier 1.6, split evenly: raw 1.6/0.8/1.8/1.0 over 1.8."""
        spec = public_goods()
        assert spec.payoff_cc == pytest.approx(1.6 / 1.8)
        assert spec.payoff_cd == pytest.approx(0.8 / 1.8)
        assert spec.payoff_dc == pytest.approx(1.0)
        assert spec.payoff_dd == pytest.approx(1.0 / 1.8)
        assert_prisoners_dilemma(spec)

    def test_ultimatum_responder_is_degenerate_in_the_opponent_axis(self) -> None:
        spec = ultimatum_responder()
        assert spec.payoff_cc == spec.payoff_cd == pytest.approx(0.2)
        assert spec.payoff_dc == spec.payoff_dd == pytest.approx(0.0)


class TestStagHuntCooperationThreshold:
    def test_a_hunt_with_dc_above_dd_uses_the_general_risk_dominance_boundary(self) -> None:
        """Cells (1.0, 0.0, 0.4, 0.3): "D" pays 0.4 against a hunter and 0.3 against a holdout.

        Indifference: p*CC + (1-p)*CD == p*DC + (1-p)*DD, so
        p* = (DD - CD) / ((CC - DC) + (DD - CD)) = 0.3 / (0.6 + 0.3) = 1/3. The DD/CC shortcut
        (0.3 here) is right only when DC == DD, which held for both original variants -- exactly
        the blind spot that let it look correct.
        """
        spec = MatrixGameSpec(
            game_id="ladder-probe", payoff_cc=1.0, payoff_cd=0.0, payoff_dc=0.4, payoff_dd=0.3
        )
        assert stag_hunt_cooperation_threshold(spec) == pytest.approx(1 / 3)

    def test_the_original_variants_keep_their_hand_computed_thresholds(self) -> None:
        assert stag_hunt_cooperation_threshold(stag_hunt("safe-hunt")) == pytest.approx(0.75)
        assert stag_hunt_cooperation_threshold(stag_hunt("risky-hunt")) == pytest.approx(0.95)

    def test_favoured_hunt_is_the_rung_where_hunting_risk_dominates(self) -> None:
        """Raw (10, 0, 4, 3) over 10: p* = 0.3 / (0.6 + 0.3) = 1/3, below even odds."""
        spec = stag_hunt("favoured-hunt")
        assert spec.payoff_cc == pytest.approx(1.0)
        assert spec.payoff_cd == pytest.approx(0.0)
        assert spec.payoff_dc == pytest.approx(0.4)
        assert spec.payoff_dd == pytest.approx(0.3)
        assert_stag_hunt(spec)
        assert stag_hunt_cooperation_threshold(spec) == pytest.approx(1 / 3)

    def test_even_hunt_puts_the_boundary_at_exactly_even_odds(self) -> None:
        """Raw (10, 0, 5.5, 4.5) over 10: p* = 0.45 / (0.45 + 0.45) = 0.5 on the nose."""
        spec = stag_hunt("even-hunt")
        assert spec.payoff_cc == pytest.approx(1.0)
        assert spec.payoff_cd == pytest.approx(0.0)
        assert spec.payoff_dc == pytest.approx(0.55)
        assert spec.payoff_dd == pytest.approx(0.45)
        assert_stag_hunt(spec)
        assert stag_hunt_cooperation_threshold(spec) == pytest.approx(0.5)

    def test_the_ladder_is_registered_in_strictly_ascending_threshold_order(self) -> None:
        thresholds = [
            stag_hunt_cooperation_threshold(stag_hunt(variant)) for variant in STAG_HUNT_VARIANTS
        ]
        assert thresholds == sorted(thresholds)
        assert len(set(thresholds)) == len(thresholds)


class TestHiLo:
    def test_cells_and_shape(self) -> None:
        """Raw (1.0, 0, 0, 0.1), already normalised: matching pays, mismatching pays nothing."""
        spec = hi_lo()
        assert spec.payoff_cc == pytest.approx(1.0)
        assert spec.payoff_cd == 0.0
        assert spec.payoff_dc == 0.0
        assert spec.payoff_dd == pytest.approx(0.1)
        assert_hi_lo(spec)

    def test_it_is_not_a_stag_hunt(self) -> None:
        """No gain from unilateral deviation at all: DC < DD breaks the stag-hunt ordering."""
        with pytest.raises(ValueError, match="not a stag hunt"):
            assert_stag_hunt(hi_lo())

    def test_the_validator_rejects_a_nonzero_off_diagonal(self) -> None:
        with pytest.raises(ValueError, match="not Hi-Lo"):
            assert_hi_lo(chicken())

    def test_the_validator_rejects_a_worthless_low_meeting_point(self) -> None:
        """DD == 0 would make the low meeting point no better than mismatching: not two
        strict equilibria, so not Hi-Lo."""
        degenerate = MatrixGameSpec(
            game_id="one-good-cell", payoff_cc=1.0, payoff_cd=0.0, payoff_dc=0.0, payoff_dd=0.0
        )
        with pytest.raises(ValueError, match="not Hi-Lo"):
            assert_hi_lo(degenerate)


class TestHarmony:
    def test_cells_and_shape(self) -> None:
        """Raw (4, 2, 3, 1) over 4: "C" strictly dominant and (C, C) the best cell."""
        spec = harmony()
        assert spec.payoff_cc == pytest.approx(1.0)
        assert spec.payoff_cd == pytest.approx(0.5)
        assert spec.payoff_dc == pytest.approx(0.75)
        assert spec.payoff_dd == pytest.approx(0.25)
        assert_harmony(spec)

    def test_the_validator_rejects_a_pd_where_defection_dominates(self) -> None:
        with pytest.raises(ValueError, match="not a harmony game"):
            assert_harmony(twin_pd())

    def test_the_validator_rejects_a_stag_hunt_where_dominance_fails(self) -> None:
        """safe-hunt has CD < DD, so cooperating is not dominant, only conditionally best."""
        with pytest.raises(ValueError, match="not a harmony game"):
            assert_harmony(stag_hunt())

    def test_the_validator_rejects_dominance_without_mutual_cooperation_being_best(self) -> None:
        """C dominates but exploiting a defector pays more than meeting a cooperator."""
        lopsided = MatrixGameSpec(
            game_id="dominant-not-best", payoff_cc=0.5, payoff_cd=1.0, payoff_dc=0.2, payoff_dd=0.1
        )
        with pytest.raises(ValueError, match="not a harmony game"):
            assert_harmony(lopsided)


class TestDefectiveCoordination:
    def test_cells_and_shape(self) -> None:
        """Raw (1, 0, 0, 4) over 4: mutual "D" is the good meeting point, "C" is simply wrong."""
        spec = defective_coordination()
        assert spec.payoff_cc == pytest.approx(0.25)
        assert spec.payoff_cd == 0.0
        assert spec.payoff_dc == 0.0
        assert spec.payoff_dd == pytest.approx(1.0)
        assert_defective_coordination(spec)

    def test_it_mirrors_hi_lo_rather_than_matching_it(self) -> None:
        with pytest.raises(ValueError, match="not defective coordination"):
            assert_defective_coordination(hi_lo())
        with pytest.raises(ValueError, match="not Hi-Lo"):
            assert_hi_lo(defective_coordination())


class TestDefectiveHarmony:
    """Harmony inverted: raw (1, 3, 2, 5) over 5, where every computation defects.

    The trap cell of wave 4b. Cooperation rising here cannot be a welfare calculation of any
    weighting, because "D" pays more than "C" against either counterpart action AND mutual "D" is
    the best cell for both sides, so joint welfare points the same way self-interest does. A rise
    is table-blind label following or a print-position preference instead, which is what the trap
    exists to separate.
    """

    def test_cells_and_shape(self) -> None:
        spec = defective_harmony()
        assert spec.payoff_cc == pytest.approx(0.2)
        assert spec.payoff_cd == pytest.approx(0.6)
        assert spec.payoff_dc == pytest.approx(0.4)
        assert spec.payoff_dd == pytest.approx(1.0)
        assert_defective_harmony(spec)

    def test_defection_strictly_dominates_at_both_counterpart_actions(self) -> None:
        """The dominance the trap rests on, stated as the two comparisons rather than assumed."""
        spec = defective_harmony()
        assert spec.payoff_dc > spec.payoff_cc
        assert spec.payoff_dd > spec.payoff_cd

    def test_mutual_defection_is_the_best_cell_for_both_sides(self) -> None:
        """So no weighting of the counterpart's payoff can make cooperation the welfare answer."""
        spec = defective_harmony()
        assert spec.payoff_dd > max(spec.payoff_cc, spec.payoff_cd, spec.payoff_dc)
        assert expected_joint_payoff(spec, DEFECT, 0.0) > expected_joint_payoff(
            spec, COOPERATE, 0.0
        )
        assert expected_joint_payoff(spec, DEFECT, 1.0) > expected_joint_payoff(
            spec, COOPERATE, 1.0
        )

    def test_the_validator_rejects_harmony_itself(self) -> None:
        with pytest.raises(ValueError, match="not a defective harmony game"):
            assert_defective_harmony(harmony())

    def test_the_validator_rejects_a_pd_whose_best_cell_is_mutual_cooperation(self) -> None:
        """Defection dominates in a PD too, so the best-cell half is what separates the two shapes."""
        with pytest.raises(ValueError, match="not a defective harmony game"):
            assert_defective_harmony(twin_pd())

    def test_the_validator_rejects_a_sheet_where_meeting_a_cooperator_pays_best(self) -> None:
        """CC above DD is the sabotage this validator exists to catch: cooperation becomes an answer."""
        rewarding_cooperation = MatrixGameSpec(
            game_id="cooperation-pays-best",
            payoff_cc=1.0,
            payoff_cd=0.6,
            payoff_dc=0.4,
            payoff_dd=0.8,
        )
        with pytest.raises(ValueError, match="not a defective harmony game"):
            assert_defective_harmony(rewarding_cooperation)


class TestGroupMixRewardSpread:
    """Hand-computed |r_C - r_D| values; the spread is p*(CC-DC) + (1-p)*(CD-DD) in magnitude."""

    def test_twin_pd_spreads_at_the_reporting_mixes(self) -> None:
        """Cells (0.6, 0.0, 1.0, 0.2): spread = |-0.2 - 0.2p| = 0.2 + 0.2p."""
        twin = twin_pd()
        assert group_mix_reward_spread(twin, 0.1) == pytest.approx(0.22)
        assert group_mix_reward_spread(twin, 0.5) == pytest.approx(0.30)
        assert group_mix_reward_spread(twin, 0.9) == pytest.approx(0.38)

    def test_risky_hunt_is_compressed_where_twin_pd_is_wide(self) -> None:
        """Cells (1.0, 0.0, 0.95, 0.95): spread = |p - 0.95|, 0.05 at p = 0.9 -- the ~8x
        gradient gap against twin-pd's 0.38 that motivates the report."""
        assert group_mix_reward_spread(stag_hunt("risky-hunt"), 0.9) == pytest.approx(0.05)

    def test_chicken_spread_vanishes_at_its_interior_fixed_point(self) -> None:
        """Cells (0.75, 0.25, 1.0, 0.0): spread = |0.25 - 0.5p|, zero exactly at p = 0.5."""
        assert group_mix_reward_spread(chicken(), 0.5) == pytest.approx(0.0)
        assert group_mix_reward_spread(chicken(), 0.1) == pytest.approx(0.2)
        assert group_mix_reward_spread(chicken(), 0.9) == pytest.approx(0.2)

    def test_harmony_spread_is_mix_independent(self) -> None:
        """Cells (1.0, 0.5, 0.75, 0.25): both differences are 0.25, so p drops out."""
        for mix in (0.1, 0.5, 0.9):
            assert group_mix_reward_spread(harmony(), mix) == pytest.approx(0.25)

    def test_hi_lo_spread_is_nearly_flat_at_a_low_mix(self) -> None:
        """Cells (1.0, 0.0, 0.0, 0.1): spread = |1.1p - 0.1|, only 0.01 at p = 0.1."""
        assert group_mix_reward_spread(hi_lo(), 0.1) == pytest.approx(0.01)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_a_probability_outside_the_unit_interval_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="opponent_coop_prob"):
            group_mix_reward_spread(twin_pd(), bad)


class TestRecipientWeightedExpectations:
    """Hand math for the wave-3 grading ladder's two payoff recipients, at twin-pd's cells.

    temptation-2 cells (0.6, 0.0, 1.0, 0.2). Joint value of a cell pair is the mean of the two
    sides, so CC pays 0.6, either off-diagonal 0.5, DD 0.2. The counterpart's payoff against my
    action a is payoff(b, a): cooperating hands them their better column in any PD.
    """

    def test_joint_payoff_hand_math_at_a_half_mix(self) -> None:
        """r_j(C) = 0.5*0.6 + 0.5*0.5 = 0.55; r_j(D) = 0.5*0.5 + 0.5*0.2 = 0.35."""
        twin = twin_pd()
        assert expected_joint_payoff(twin, COOPERATE, 0.5) == pytest.approx(0.55)
        assert expected_joint_payoff(twin, DEFECT, 0.5) == pytest.approx(0.35)

    def test_counterpart_payoff_hand_math_at_a_half_mix(self) -> None:
        """r_o(C) = 0.5*0.6 + 0.5*1.0 = 0.8; r_o(D) = 0.5*0.0 + 0.5*0.2 = 0.1."""
        twin = twin_pd()
        assert expected_counterpart_payoff(twin, COOPERATE, 0.5) == pytest.approx(0.8)
        assert expected_counterpart_payoff(twin, DEFECT, 0.5) == pytest.approx(0.1)

    def test_joint_welfare_spread_is_the_gap_of_the_two_expectations(self) -> None:
        """Cells (0.6, 0.0, 1.0, 0.2): joint gap = 0.3 - 0.2p, so 0.28 / 0.20 / 0.12."""
        twin = twin_pd()
        assert joint_welfare_reward_spread(twin, 0.1) == pytest.approx(0.28)
        assert joint_welfare_reward_spread(twin, 0.5) == pytest.approx(0.20)
        assert joint_welfare_reward_spread(twin, 0.9) == pytest.approx(0.12)

    def test_other_payoff_spread_is_the_widest_on_the_ladder(self) -> None:
        """Cells (0.6, 0.0, 1.0, 0.2): other gap = 0.8 - 0.2p, so 0.78 / 0.70 / 0.62."""
        twin = twin_pd()
        assert other_payoff_reward_spread(twin, 0.1) == pytest.approx(0.78)
        assert other_payoff_reward_spread(twin, 0.5) == pytest.approx(0.70)
        assert other_payoff_reward_spread(twin, 0.9) == pytest.approx(0.62)

    def test_temptation_10_has_an_interior_joint_welfare_crossing(self) -> None:
        """DC + CD = 1.0 > 2*CC = 6/13, so the joint gap (0.4231 - 0.6923p) crosses at 0.6111.

        The crossing is the arm's registered numeric prediction on temptation-10 prompts, the way
        chicken's 0.50 is for group-mix; below it cooperation is joint-better, above it defection.
        """
        crossing = joint_welfare_gap_crossing(twin_pd("temptation-10"))
        assert crossing == pytest.approx(0.611111, abs=1e-6)
        spec = twin_pd("temptation-10")
        below = expected_joint_payoff(spec, COOPERATE, 0.5) - expected_joint_payoff(
            spec, DEFECT, 0.5
        )
        above = expected_joint_payoff(spec, COOPERATE, 0.7) - expected_joint_payoff(
            spec, DEFECT, 0.7
        )
        assert below > 0 > above

    def test_temptation_2_has_no_interior_joint_welfare_crossing(self) -> None:
        """2*CC = 1.2 > DC + CD = 1.0: cooperation is joint-dominant, so no rate divides the signs."""
        crossing = joint_welfare_gap_crossing(twin_pd())
        assert crossing is None or not 0.0 < crossing < 1.0

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_a_probability_outside_the_unit_interval_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="coop_prob"):
            expected_joint_payoff(twin_pd(), COOPERATE, bad)
        with pytest.raises(ValueError, match="coop_prob"):
            expected_counterpart_payoff(twin_pd(), COOPERATE, bad)


class TestOrderingValidatorsRejectTheWrongGame:
    def test_prisoners_dilemma_validator_rejects_a_violating_spec(self) -> None:
        """Mutual cooperation pays best here, so defection is not tempting: not a PD."""
        cooperative = MatrixGameSpec(
            game_id="not-a-pd", payoff_cc=1.0, payoff_cd=0.0, payoff_dc=0.5, payoff_dd=0.2
        )
        with pytest.raises(ValueError, match="not a strict prisoner's dilemma"):
            assert_prisoners_dilemma(cooperative)

    def test_prisoners_dilemma_validator_rejects_mutual_defection_beating_cooperation(
        self,
    ) -> None:
        inverted = MatrixGameSpec(
            game_id="not-a-pd", payoff_cc=0.2, payoff_cd=0.0, payoff_dc=1.0, payoff_dd=0.6
        )
        with pytest.raises(ValueError, match="not a strict prisoner's dilemma"):
            assert_prisoners_dilemma(inverted)

    def test_constant_sum_validator_rejects_the_positive_sum_twin_pd(
        self, twin: MatrixGameSpec
    ) -> None:
        with pytest.raises(ValueError, match="not constant-sum"):
            assert_constant_sum(twin)

    def test_stag_hunt_and_chicken_validators_reject_the_twin_pd(
        self, twin: MatrixGameSpec
    ) -> None:
        with pytest.raises(ValueError, match="not a stag hunt"):
            assert_stag_hunt(twin)
        with pytest.raises(ValueError, match="not chicken"):
            assert_chicken(twin)


class TestDictatorSpec:
    def test_fields_read_back(self) -> None:
        spec = DictatorSpec(game_id="dictator", endowment=10)
        assert spec.endowment == 10

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_endowment_raises(self, bad: int) -> None:
        with pytest.raises(ValueError, match="endowment must be positive"):
            DictatorSpec(game_id="dictator", endowment=bad)


class TestOpponentMoves:
    def test_tit_for_tat_opens_cooperatively_then_copies(self) -> None:
        mine = [COOPERATE, DEFECT, COOPERATE, COOPERATE]
        assert opponent_moves(OpponentRule.TIT_FOR_TAT, mine) == [
            COOPERATE,
            COOPERATE,
            DEFECT,
            COOPERATE,
        ]

    def test_grim_trigger_never_forgives(self) -> None:
        mine = [COOPERATE, DEFECT, COOPERATE, COOPERATE]
        assert opponent_moves(OpponentRule.GRIM_TRIGGER, mine) == [
            COOPERATE,
            COOPERATE,
            DEFECT,
            DEFECT,
        ]

    def test_always_cooperate_ignores_history(self) -> None:
        mine = [DEFECT, DEFECT, DEFECT]
        assert opponent_moves(OpponentRule.ALWAYS_C, mine) == [COOPERATE] * 3

    @pytest.mark.parametrize("rule", list(OpponentRule))
    def test_first_round_cannot_depend_on_my_first_move(self, rule: OpponentRule) -> None:
        assert opponent_moves(rule, [DEFECT])[0] == COOPERATE
        assert opponent_moves(rule, []) == []

    def test_unknown_move_raises(self) -> None:
        with pytest.raises(ValueError, match="my_moves must contain only"):
            opponent_moves(OpponentRule.TIT_FOR_TAT, [COOPERATE, "swerve"])


class TestBackwardInductionAgainstTitForTat:
    """Hand-computed totals at twin_pd("temptation-2") over five rounds against tit-for-tat.

    Cells are CC=0.6, CD=0.0, DC=1.0, DD=0.2, and the opponent opens with "C" then copies my
    previous move.
    """

    ALWAYS_COOPERATE_TOTAL = 3.0  # 5 x 0.6
    COOPERATE_UNTIL_LAST_TOTAL = 3.4  # 4 x 0.6 + 1.0, the final defection meeting a copied "C"
    ALWAYS_DEFECT_TOTAL = 1.8  # 1.0 on round one, then 4 x 0.2 once the copying starts

    def test_the_three_totals_match_hand_arithmetic(self, twin: MatrixGameSpec) -> None:
        rule = OpponentRule.TIT_FOR_TAT
        always_cooperate = [COOPERATE] * 5
        cooperate_until_last = [COOPERATE] * 4 + [DEFECT]
        always_defect = [DEFECT] * 5
        assert simulate_iterated(twin, rule, always_cooperate) == pytest.approx(
            self.ALWAYS_COOPERATE_TOTAL
        )
        assert simulate_iterated(twin, rule, cooperate_until_last) == pytest.approx(
            self.COOPERATE_UNTIL_LAST_TOTAL
        )
        assert simulate_iterated(twin, rule, always_defect) == pytest.approx(
            self.ALWAYS_DEFECT_TOTAL
        )

    def test_end_game_defection_strictly_beats_both_pure_strategies(self) -> None:
        assert self.COOPERATE_UNTIL_LAST_TOTAL > self.ALWAYS_COOPERATE_TOTAL
        assert self.COOPERATE_UNTIL_LAST_TOTAL > self.ALWAYS_DEFECT_TOTAL

    def test_it_is_also_the_brute_forced_optimum(self, twin: MatrixGameSpec) -> None:
        assert max_iterated_return(twin, OpponentRule.TIT_FOR_TAT, 5) == pytest.approx(
            self.COOPERATE_UNTIL_LAST_TOTAL
        )


class TestTemptationMagnitudeChangesTheOptimalPolicy:
    def test_temptation_10_rewards_alternating_exploitation(self) -> None:
        """2*CC > DC + CD fails at this magnitude, so alternating beats sustained cooperation.

        Exploiting every other round collects three payoffs of 1.0, each paid for with a 0.0.
        """
        spec = twin_pd("temptation-10")
        rule = OpponentRule.TIT_FOR_TAT
        alternating = [DEFECT, COOPERATE, DEFECT, COOPERATE, DEFECT]
        assert simulate_iterated(spec, rule, alternating) == pytest.approx(3.0)
        assert simulate_iterated(spec, rule, [COOPERATE] * 4 + [DEFECT]) == pytest.approx(
            4 * (3 / 13) + 1.0
        )
        assert max_iterated_return(spec, rule, 5) == pytest.approx(3.0)


class TestSimulateIteratedAgainstTheOtherRules:
    def test_grim_trigger_punishment_is_permanent(self, twin: MatrixGameSpec) -> None:
        """Mine C,C,D,D,D meets opponent C,C,C,D,D: 0.6 + 0.6 + 1.0 + 0.2 + 0.2."""
        mine = [COOPERATE, COOPERATE, DEFECT, DEFECT, DEFECT]
        assert simulate_iterated(twin, OpponentRule.GRIM_TRIGGER, mine) == pytest.approx(2.6)

    def test_one_early_defection_costs_more_against_grim_than_against_tit_for_tat(
        self, twin: MatrixGameSpec
    ) -> None:
        """Tit-for-tat retaliates once (0.6+1.0+0.0+0.6+0.6); grim never stops (0.6+1.0+0+0+0)."""
        mine = [COOPERATE, DEFECT, COOPERATE, COOPERATE, COOPERATE]
        assert simulate_iterated(twin, OpponentRule.TIT_FOR_TAT, mine) == pytest.approx(2.8)
        assert simulate_iterated(twin, OpponentRule.GRIM_TRIGGER, mine) == pytest.approx(1.6)

    def test_an_unconditional_cooperator_is_fully_exploitable(self, twin: MatrixGameSpec) -> None:
        assert simulate_iterated(twin, OpponentRule.ALWAYS_C, [DEFECT] * 5) == pytest.approx(5.0)

    def test_empty_match_returns_zero(self, twin: MatrixGameSpec) -> None:
        assert simulate_iterated(twin, OpponentRule.TIT_FOR_TAT, []) == pytest.approx(0.0)


class TestMaxIteratedReturn:
    def test_single_round_optimum_is_the_temptation_cell(self, twin: MatrixGameSpec) -> None:
        assert max_iterated_return(twin, OpponentRule.TIT_FOR_TAT, 1) == pytest.approx(1.0)

    def test_grim_trigger_also_rewards_waiting_until_the_last_round(
        self, twin: MatrixGameSpec
    ) -> None:
        assert max_iterated_return(twin, OpponentRule.GRIM_TRIGGER, 5) == pytest.approx(3.4)

    def test_unconditional_cooperator_optimum_is_defecting_every_round(
        self, twin: MatrixGameSpec
    ) -> None:
        assert max_iterated_return(twin, OpponentRule.ALWAYS_C, 5) == pytest.approx(5.0)

    @pytest.mark.parametrize("bad", [0, -1, 11])
    def test_round_count_outside_the_brute_force_window_raises(
        self, twin: MatrixGameSpec, bad: int
    ) -> None:
        with pytest.raises(ValueError, match="n_rounds must be in"):
            max_iterated_return(twin, OpponentRule.TIT_FOR_TAT, bad)


class TestWorstIteratedReturn:
    """The mirror the row-relative parse price reads: what the worst reachable match pays.

    Both bounds together are what put an iterated arm's reward on a range rather than at a point, so a
    failure can be priced one range below the worst of them instead of at a constant nobody scaled.
    """

    def test_the_worst_single_round_is_the_sucker_cell(self, twin: MatrixGameSpec) -> None:
        # Against a tit-for-tat opponent opening with cooperation, one round of cooperating pays CC
        # and one of defecting pays DC, so the worst single round is the lower of the two.
        assert worst_iterated_return(twin, OpponentRule.TIT_FOR_TAT, 1) == pytest.approx(
            min(twin.payoff_cc, twin.payoff_dc)
        )

    def test_the_worst_match_never_beats_the_best_one(self, twin: MatrixGameSpec) -> None:
        for rule in OpponentRule:
            for n_rounds in (1, 3, 5):
                assert worst_iterated_return(twin, rule, n_rounds) <= max_iterated_return(
                    twin, rule, n_rounds
                ), (rule, n_rounds)

    def test_the_worst_grim_trigger_match_defects_once_and_then_cooperates_into_it(
        self, twin: MatrixGameSpec
    ) -> None:
        # Not a constant sequence, which is the point of searching rather than assuming: defecting
        # every round pays DC + 4 * DD = 1.8, while defecting once and then cooperating into the
        # trigger's permanent defection pays DC + 4 * CD = 1.0, the floor of this rule's whole range.
        assert worst_iterated_return(twin, OpponentRule.GRIM_TRIGGER, 5) == pytest.approx(
            twin.payoff_dc + 4 * twin.payoff_cd
        )
        assert worst_iterated_return(twin, OpponentRule.GRIM_TRIGGER, 5) < (
            twin.payoff_dc + 4 * twin.payoff_dd
        )

    @pytest.mark.parametrize("bad", [0, -1, 11])
    def test_round_count_outside_the_brute_force_window_raises(
        self, twin: MatrixGameSpec, bad: int
    ) -> None:
        with pytest.raises(ValueError, match="n_rounds must be in"):
            worst_iterated_return(twin, OpponentRule.TIT_FOR_TAT, bad)


class TestGroupMixFixedPoint:
    """Where group-mix training settles, and the fact that the grading mode moves it.

    chicken-group is this repo's numeric control on the whole reward path: the prediction is a
    cooperation rate rather than a direction, so a run landing at either corner is evidence of a
    reward-or-advantage bug. The registry, `games.payoffs.chicken` and docs/games-predictions.md all
    quote 0.50, and every one of those numbers is the plain group-mix algebra. Under
    `--leave-one-out` a completion is graded against the other G-1, which moves the crossing to
    0.3125 at a group of eight -- so an unflagged leave-one-out run settling on the documented 0.50
    would read as healthy while being wrong, and one settling correctly would read as the bug.
    """

    def test_chicken_settles_at_the_pre_registered_half(self) -> None:
        # Normalised chicken is CC=0.75, CD=0.25, DC=1.0, DD=0.0, so the gap E[r|C] - E[r|D] is
        # 0.25 - 0.5p, which is zero at p = 0.5.
        assert group_mix_fixed_point(chicken(), num_generations=8) == pytest.approx(0.5)

    def test_the_crossing_does_not_move_with_the_group_size_without_the_flag(self) -> None:
        for group in (2, 4, 8, 16):
            assert group_mix_fixed_point(chicken(), num_generations=group) == pytest.approx(0.5)

    def test_leave_one_out_moves_the_crossing_to_a_third(self) -> None:
        # k* = 0.5G - 1.5 cooperators, i.e. 2.5 of 8, so the rate is 0.3125 rather than 0.50.
        assert group_mix_fixed_point(
            chicken(), num_generations=8, leave_one_out=True
        ) == pytest.approx(0.3125)

    def test_under_leave_one_out_the_crossing_depends_on_the_group_size(self) -> None:
        rates = {
            group: group_mix_fixed_point(chicken(), num_generations=group, leave_one_out=True)
            for group in (4, 8, 16)
        }
        assert rates[4] == pytest.approx(0.125)
        assert rates[8] == pytest.approx(0.3125)
        assert rates[16] == pytest.approx(0.40625)

    def test_a_dominance_game_has_no_interior_rate_to_settle_at(self) -> None:
        # The PD's gap never changes sign over [0, 1], so a run goes to a corner.
        assert group_mix_fixed_point(twin_pd(), num_generations=8) is None
        assert group_mix_fixed_point(harmony(), num_generations=8) is None

    def test_a_repelling_crossing_is_not_reported_as_a_fixed_point(self) -> None:
        # The stag hunt and Hi-Lo both have interior crossings, and both REPEL: a run leaves them
        # for whichever corner it started nearest. Reporting one as "where this settles" would
        # invert the prediction.
        for variant in STAG_HUNT_VARIANTS:
            assert group_mix_fixed_point(stag_hunt(variant), num_generations=8) is None
        assert group_mix_fixed_point(hi_lo(), num_generations=8) is None

    def test_a_group_of_one_has_no_mix_and_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2 completions"):
            group_mix_fixed_point(chicken(), num_generations=1)


class TestFixedPieMatchesTwinPdInSignOnly:
    """The arm notes claim a matched action gradient; only the sign matches, not the magnitude."""

    def test_the_defection_advantage_is_flat_for_fixed_pie_and_rising_for_twin_pd(self) -> None:
        fixed = fixed_pie_pd()
        twin = twin_pd()
        for mix in (0.0, 0.5, 1.0):
            assert group_mix_reward_spread(fixed, mix) == pytest.approx(0.4)
        assert group_mix_reward_spread(twin, 0.0) == pytest.approx(0.2)
        assert group_mix_reward_spread(twin, 0.5) == pytest.approx(0.3)
        assert group_mix_reward_spread(twin, 1.0) == pytest.approx(0.4)

    def test_the_two_agree_only_where_everybody_cooperates(self) -> None:
        # Which is the mix twin-pd-group is expected to move AWAY from, so the arms' within-group
        # signal diverges over a run rather than converging.
        assert group_mix_reward_spread(fixed_pie_pd(), 1.0) == pytest.approx(
            group_mix_reward_spread(twin_pd(), 1.0)
        )
        assert group_mix_reward_spread(fixed_pie_pd(), 0.0) != pytest.approx(
            group_mix_reward_spread(twin_pd(), 0.0)
        )

    def test_defection_is_favoured_at_every_mix_in_both_games(self) -> None:
        # The claim that does hold: same sign everywhere, which is what makes the contrast valid.
        for spec in (fixed_pie_pd(), twin_pd()):
            for mix in (0.0, 0.25, 0.5, 0.75, 1.0):
                assert group_mix_fixed_point(spec, num_generations=8) is None
                assert group_mix_reward_spread(spec, mix) > 0.0


class TestStatedMatchArithmetic:
    """The track-record grading's EV, crossover and optimal side, all against hand math.

    The crossover formula is the one number every pre-registered prediction for the
    pd-track-record arm is written against, and the design brief's own version of it was
    algebraically wrong -- so the classic-PD anchor (3p = 5 - 4p, p* = 5/7) is pinned here by
    hand rather than trusted from any document, and the code's crossover is required to be the
    zero of the same gap the reward pays.
    """

    def test_the_ev_is_the_matching_expectation_not_an_independent_mix(
        self, twin: MatrixGameSpec
    ) -> None:
        # At p the counterpart COPIES the action, at 1-p it plays the opposite: cooperating pays
        # p*CC + (1-p)*CD and defecting p*DD + (1-p)*DC. On normalised twin-pd (0.6/0/1.0/0.2)
        # at p=0.4 that is 0.24 against 0.68.
        assert stated_match_expected_payoff(twin, COOPERATE, 0.4) == pytest.approx(0.24)
        assert stated_match_expected_payoff(twin, DEFECT, 0.4) == pytest.approx(0.68)
        # The independent-mix arithmetic would give defect 0.4*1.0 + 0.6*0.2 = 0.52 at an
        # opponent COOPERATION rate of 0.4; matching semantics give 0.68. The two must differ,
        # or this grading silently became vs-fixed-mix.
        assert stated_match_expected_payoff(twin, DEFECT, 0.4) != pytest.approx(0.52)

    def test_the_classic_pd_crossover_is_five_sevenths(self, twin: MatrixGameSpec) -> None:
        # By hand on the raw cells (3/0/5/1): 3p = p + 5(1-p) => 7p = 5. Normalisation is a
        # positive scaling, so the crossover is unchanged at the normalised cells.
        assert stated_match_crossover(twin) == pytest.approx(5 / 7)
        assert stated_match_gap(twin, 5 / 7) == pytest.approx(0.0)

    def test_temptation_ten_crosses_at_thirteen_fifteenths(self) -> None:
        # Raw cells 3/0/13/1: 3p = p + 13(1-p) => 15p = 13.
        spec = twin_pd("temptation-10")
        assert stated_match_crossover(spec) == pytest.approx(13 / 15)

    def test_the_optimal_side_flips_at_the_crossover(self, twin: MatrixGameSpec) -> None:
        crossover = stated_match_crossover(twin)
        assert crossover is not None
        assert stated_match_optimal_action(twin, crossover - 0.01) == DEFECT
        assert stated_match_optimal_action(twin, crossover + 0.01) == COOPERATE

    def test_an_exact_tie_has_no_optimal_side(self) -> None:
        # Dyadic cells on purpose: a gap of EXACTLY zero only exists in float arithmetic when
        # every operand is a power-of-two fraction. twin-pd's 5/7 crossover leaves a one-ulp
        # residue, which is why the real corpus can never land on a tie and why the builder's
        # audit floor is a margin rather than an equality test. Cells 0.75/0.25/0.75/0.25 put
        # the crossover at exactly 0.5.
        dyadic = MatrixGameSpec(
            game_id="dyadic-tie", payoff_cc=0.75, payoff_cd=0.25, payoff_dc=0.75, payoff_dd=0.25
        )
        assert stated_match_crossover(dyadic) == 0.5
        assert stated_match_optimal_action(dyadic, 0.5) is None
        assert stated_match_optimal_action(dyadic, 0.6) == COOPERATE
        assert stated_match_optimal_action(dyadic, 0.4) == DEFECT

    def test_the_gap_is_the_difference_of_the_two_evs(self, twin: MatrixGameSpec) -> None:
        for probability in (0.0, 0.4, 5 / 7, 0.9, 1.0):
            assert stated_match_gap(twin, probability) == pytest.approx(
                stated_match_expected_payoff(twin, COOPERATE, probability)
                - stated_match_expected_payoff(twin, DEFECT, probability)
            )

    def test_a_probability_outside_the_unit_interval_is_refused(self, twin: MatrixGameSpec) -> None:
        # -1.0 is the corpus marker for "no track record stated", and it must never be readable
        # as a probability.
        with pytest.raises(ValueError, match=r"match_prob must be in \[0, 1\]"):
            stated_match_expected_payoff(twin, COOPERATE, -1.0)
        with pytest.raises(ValueError, match=r"match_prob must be in \[0, 1\]"):
            stated_match_gap(twin, 1.5)

    def test_a_slope_free_table_has_no_crossover(self) -> None:
        # CC - DD == CD - DC makes the gap constant in p, so no mixture over p can move the
        # optimal action and the corpus builder must refuse the table. Dyadic cells make the
        # slope EXACTLY zero; the near-flat non-dyadic case below leaves a one-ulp slope whose
        # residual crossover lands absurdly far outside [0, 1] -- the contract is "None or not
        # interior", and the builder checks both.
        flat = MatrixGameSpec(
            game_id="rigged-flat", payoff_cc=0.75, payoff_cd=0.75, payoff_dc=0.25, payoff_dd=0.25
        )
        assert stated_match_crossover(flat) is None
        nearly_flat = MatrixGameSpec(
            game_id="rigged-near-flat", payoff_cc=0.6, payoff_cd=0.5, payoff_dc=0.3, payoff_dd=0.4
        )
        residual = stated_match_crossover(nearly_flat)
        assert residual is None or not 0.0 < residual < 1.0


# Every mix a care reduction is checked at, including both corners: the family's two identities have
# to hold across the whole interval, not at a convenient interior point.
CARE_MIXES = (0.0, 0.25, 1 / 3, 0.5, 2 / 3, 0.9, 1.0)
RETURN_FIFTH = 0.20
RETURN_HALF = 0.50


def announced_rule_spec(fraction: float) -> TrustSpec:
    """One announced-rule trust spec at a given rate, the shape the care family's trust rows carry."""
    return TrustSpec(
        game_id="trust-under-test",
        endowment=TRUST_ENDOWMENT,
        multiplier=TRUST_MULTIPLIER,
        stated_return_fraction=fraction,
    )


def strategy_method_spec() -> TrustSpec:
    """One strategy-method trust spec, whose rate the completion writes rather than the prompt."""
    return TrustSpec(
        game_id="trust-under-test", endowment=TRUST_ENDOWMENT, multiplier=TRUST_MULTIPLIER
    )


class TestExpectedCarePayoff:
    """The care family's matrix arithmetic, and the two reductions the family is defined by.

    `(own + alpha * other) / (1 + alpha)` has to BE own-payoff group-mix at alpha 0 and
    joint-welfare group-mix at alpha 1, because wave 4b's control pair is one corpus regraded
    between two members of the family: an arithmetic that agreed only approximately would make the
    control's zero point a different reward from the ladder rung it reproduces.
    """

    def test_alpha_zero_is_the_own_payoff_expectation(self, twin: MatrixGameSpec) -> None:
        for action in (COOPERATE, DEFECT):
            for mix in CARE_MIXES:
                own = mix * twin.payoff(action, COOPERATE) + (1 - mix) * twin.payoff(action, DEFECT)
                assert expected_care_payoff(twin, action, mix, alpha=0.0) == pytest.approx(own)

    def test_alpha_zero_matches_the_reward_functions_own_payoff_helper(
        self, twin: MatrixGameSpec
    ) -> None:
        # The identity the alpha-0 leg rests on, checked against the function the reward actually
        # calls for own-payoff group-mix. games.rewards imports games.payoffs, so the dependency
        # cannot run the other way and the arithmetic is written in both modules; this keeps the two
        # equal, and it is what the alpha-0 regrade of a swept corpus depends on.
        for action in (COOPERATE, DEFECT):
            for mix in CARE_MIXES:
                assert expected_care_payoff(twin, action, mix, alpha=0.0) == pytest.approx(
                    rewards._expected_payoff(twin, action, mix)
                )

    def test_alpha_one_is_the_joint_welfare_expectation(self) -> None:
        for variant in ("temptation-2", "temptation-10"):
            spec = twin_pd(variant)
            for action in (COOPERATE, DEFECT):
                for mix in CARE_MIXES:
                    assert expected_care_payoff(spec, action, mix, alpha=1.0) == pytest.approx(
                        expected_joint_payoff(spec, action, mix), abs=1e-12
                    )

    def test_the_ordering_tends_to_the_counterparts_payoff_without_reaching_it(
        self, twin: MatrixGameSpec
    ) -> None:
        # The family's far limit: rescaled by (1 + alpha) / alpha, a large weight leaves the
        # counterpart's own expectation. So other-payoff grading is the LIMIT of the family rather
        # than a member of it, which is what makes the arm an interpolation of the wave-3 ladder.
        large = 1e6
        for action in (COOPERATE, DEFECT):
            rescaled = expected_care_payoff(twin, action, 2 / 3, alpha=large) * (large + 1) / large
            assert rescaled == pytest.approx(
                expected_counterpart_payoff(twin, action, 2 / 3), abs=1e-5
            )

    def test_the_cooperation_gap_grows_monotonically_with_alpha(self, twin: MatrixGameSpec) -> None:
        # At temptation-2 own payoff prefers defection at every mix and the counterpart's prefers
        # cooperation at every mix, so the gap crosses zero once and never turns back.
        gaps = [
            expected_care_payoff(twin, COOPERATE, 2 / 3, alpha=alpha)
            - expected_care_payoff(twin, DEFECT, 2 / 3, alpha=alpha)
            for alpha in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
        ]
        assert gaps == sorted(gaps)
        assert gaps[0] < 0 < gaps[-1]

    def test_the_reward_stays_inside_the_payoff_range(self, twin: MatrixGameSpec) -> None:
        for alpha in (0.0, 0.5, 1.0, 3.0, 100.0):
            for action in (COOPERATE, DEFECT):
                for mix in CARE_MIXES:
                    assert 0.0 <= expected_care_payoff(twin, action, mix, alpha=alpha) <= 1.0

    def test_a_negative_weight_is_refused(self, twin: MatrixGameSpec) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            expected_care_payoff(twin, COOPERATE, 0.5, alpha=-0.5)
        with pytest.raises(ValueError, match="non-negative"):
            assert_care_alpha(-1e-9)

    def test_a_non_finite_weight_is_refused(self, twin: MatrixGameSpec) -> None:
        for alpha in (float("inf"), float("nan")):
            with pytest.raises(ValueError, match="finite"):
                expected_care_payoff(twin, COOPERATE, 0.5, alpha=alpha)

    def test_a_mix_outside_the_unit_interval_is_refused(self, twin: MatrixGameSpec) -> None:
        with pytest.raises(ValueError, match=r"coop_prob must be in \[0, 1\]"):
            expected_care_payoff(twin, COOPERATE, 1.5, alpha=1.0)


class TestCareRewardSpread:
    def test_it_is_the_gap_between_the_two_actions(self, twin: MatrixGameSpec) -> None:
        for alpha in (0.0, 1.0, 2.5):
            for mix in (0.0, 0.4, 1.0):
                assert care_reward_spread(twin, mix, alpha=alpha) == pytest.approx(
                    abs(
                        expected_care_payoff(twin, COOPERATE, mix, alpha=alpha)
                        - expected_care_payoff(twin, DEFECT, mix, alpha=alpha)
                    )
                )

    def test_it_reduces_to_the_two_ladder_spreads_at_alpha_zero_and_one(
        self, twin: MatrixGameSpec
    ) -> None:
        for mix in (0.0, 0.4, 1.0):
            assert care_reward_spread(twin, mix, alpha=0.0) == pytest.approx(
                group_mix_reward_spread(twin, mix)
            )
            assert care_reward_spread(twin, mix, alpha=1.0) == pytest.approx(
                joint_welfare_reward_spread(twin, mix)
            )


class TestTrusteePayoff:
    """The counterpart's side of the consignment, and the cancellation it makes visible."""

    def test_the_trustee_keeps_what_it_does_not_send_back(self) -> None:
        spec = announced_rule_spec(RETURN_HALF)
        for sent in (0, 4, TRUST_ENDOWMENT):
            for fraction in (0.0, 0.2, 0.5, 1.0):
                assert trustee_payoff(spec, sent=sent, return_fraction=fraction) == pytest.approx(
                    (1 - fraction) * TRUST_MULTIPLIER * sent
                )

    def test_the_two_sides_sum_to_a_total_the_return_rate_cannot_move(self) -> None:
        # E + (m - 1) * s whatever f is. This cancellation is why the care reward on this game is a
        # pure giving gradient at alpha 1 and reads nothing about the announced rate.
        spec = announced_rule_spec(RETURN_HALF)
        for sent in (0, 3, TRUST_ENDOWMENT):
            for fraction in (0.0, 0.2, 0.5, 1.0):
                total = trustor_payoff(spec, sent=sent, return_fraction=fraction) + trustee_payoff(
                    spec, sent=sent, return_fraction=fraction
                )
                assert total == pytest.approx(TRUST_ENDOWMENT + (TRUST_MULTIPLIER - 1) * sent)

    def test_a_send_or_rate_outside_its_range_is_refused(self) -> None:
        spec = announced_rule_spec(RETURN_HALF)
        with pytest.raises(ValueError, match="sent="):
            trustee_payoff(spec, sent=TRUST_ENDOWMENT + 1, return_fraction=0.5)
        with pytest.raises(ValueError, match="return_fraction="):
            trustee_payoff(spec, sent=1, return_fraction=1.5)


class TestTrustorCareReward:
    """The trust member of the care family: the alpha-0 identity and the two alpha-dependent traps.

    Both traps are silent and neither can be checked at construction, because both depend on the
    weight rather than on the row: a reward above 1 puts the trust rows on a wider scale than the
    matrix rows they share a step with, and a reward constant in the send trains nothing at all.
    """

    def test_alpha_zero_is_the_trustor_reward_exactly(self) -> None:
        for fraction in (RETURN_FIFTH, RETURN_HALF):
            spec = announced_rule_spec(fraction)
            for sent in range(TRUST_ENDOWMENT + 1):
                assert trustor_care_reward(
                    spec,
                    sent=sent,
                    return_fraction=fraction,
                    max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
                    alpha=0.0,
                ) == pytest.approx(
                    trustor_reward(
                        spec,
                        sent=sent,
                        return_fraction=fraction,
                        max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
                    )
                )

    def test_at_alpha_one_the_announced_rate_cancels_out(self) -> None:
        # Both registered rates pay the same reward for the same send, so sending everything is
        # optimal at both: the clean giving gradient the plan describes, with no rate to read.
        ceiling = TRUST_ENDOWMENT * TRUST_MAX_STATED_RETURN_FRACTION * TRUST_MULTIPLIER
        for sent in (0, 4, TRUST_ENDOWMENT):
            for fraction in (RETURN_FIFTH, RETURN_HALF):
                reward = trustor_care_reward(
                    announced_rule_spec(fraction),
                    sent=sent,
                    return_fraction=fraction,
                    max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
                    alpha=1.0,
                )
                assert reward == pytest.approx(
                    (TRUST_ENDOWMENT + (TRUST_MULTIPLIER - 1) * sent) / (2 * ceiling)
                )

    def test_the_registered_weights_pass_the_range_and_liveness_check(self) -> None:
        for fraction in (RETURN_FIFTH, RETURN_HALF):
            for alpha in (0.0, 1.0):
                assert_trust_care_spec(
                    announced_rule_spec(fraction),
                    max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
                    alpha=alpha,
                )

    def test_a_weight_whose_reward_leaves_the_payoff_range_is_refused(self) -> None:
        # The trustee keeps 24 of the 30 multiplied stock units at the return-fifth rate, against a
        # trustor ceiling of 15, so above alpha 1 a trust row's reward passes 1 while every matrix row
        # in its batch is capped there.
        with pytest.raises(ValueError, match=r"above 1\.0"):
            assert_trust_care_spec(
                announced_rule_spec(RETURN_FIFTH),
                max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
                alpha=2.0,
            )

    def test_the_refusal_names_the_keep_the_announced_rate_leaves(self) -> None:
        # The figure a reader sizes a replacement ceiling from, so it has to be the one the corner
        # reward was computed from: the multiplied consignment is 30 stock units, but the announced
        # rate sends f*m*E of it back, so no completion on this row can leave the counterpart 30.
        # Return-half is not parametrised here because its corner reward is exactly 1.0 at every
        # weight (own and other both reach 15), so that rate never reaches this refusal at all.
        spec = announced_rule_spec(RETURN_FIFTH)
        assert trustee_payoff(
            spec, sent=TRUST_ENDOWMENT, return_fraction=RETURN_FIFTH
        ) == pytest.approx(24.0)
        with pytest.raises(ValueError, match=r"keeps up to 24 at this row's announced rate"):
            assert_trust_care_spec(
                spec, max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION, alpha=2.0
            )

    def test_the_weight_that_makes_the_reward_constant_in_the_send_is_refused(self) -> None:
        # The care slope is (f*m - 1) + alpha * (1 - f) * m, which vanishes at alpha 1/6 for the
        # return-fifth rate: every parsed completion scores identically and the rows train nothing.
        dead_alpha = (1 - RETURN_FIFTH * TRUST_MULTIPLIER) / ((1 - RETURN_FIFTH) * TRUST_MULTIPLIER)
        assert dead_alpha == pytest.approx(1 / 6)
        with pytest.raises(ValueError, match="whatever the"):
            assert_trust_care_spec(
                announced_rule_spec(RETURN_FIFTH),
                max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
                alpha=dead_alpha,
            )

    def test_a_strategy_method_row_has_no_care_reward_of_its_own(self) -> None:
        no_rate = TrustSpec(
            game_id="trust-under-test", endowment=TRUST_ENDOWMENT, multiplier=TRUST_MULTIPLIER
        )
        with pytest.raises(ValueError, match="announces no return rate"):
            assert_trust_care_spec(
                no_rate, max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION, alpha=1.0
            )

    def test_the_corner_rewards_bound_every_reachable_reward(self) -> None:
        spec = announced_rule_spec(RETURN_HALF)
        corners = trust_care_corner_rewards(
            spec, max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION, alpha=1.0
        )
        every_send = [
            trustor_care_reward(
                spec,
                sent=sent,
                return_fraction=RETURN_HALF,
                max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
                alpha=1.0,
            )
            for sent in range(TRUST_ENDOWMENT + 1)
        ]
        assert min(every_send) == pytest.approx(min(corners.values()))
        assert max(every_send) == pytest.approx(max(corners.values()))


class TestTrustCornerRewards:
    """The own-payoff corner tables, which the row-relative parse price reads on both trust legs.

    Each one is asserted against the span function that already existed, because the span is the same
    quantity read for a different purpose: a corner table that disagreed with it would price a failure
    against a range no group could span.
    """

    def test_the_announced_rule_corners_span_that_rows_whole_reward_range(self) -> None:
        for fraction in (RETURN_FIFTH, RETURN_HALF):
            spec = announced_rule_spec(fraction)
            corners = trust_stated_rule_corner_rewards(
                spec, max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION
            )
            every_send = [
                trustor_reward(
                    spec,
                    sent=sent,
                    return_fraction=fraction,
                    max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
                )
                for sent in range(TRUST_ENDOWMENT + 1)
            ]
            assert set(corners) == {0, TRUST_ENDOWMENT}
            assert min(corners.values()) == pytest.approx(min(every_send))
            assert max(corners.values()) == pytest.approx(max(every_send))
            assert max(corners.values()) - min(corners.values()) == pytest.approx(
                trustor_reward_spread(spec, max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION)
            )

    def test_the_announced_rule_corners_are_the_care_familys_at_weight_zero(self) -> None:
        # One arithmetic seen from two sides: the family's own-payoff member is the announced-rule
        # reward, so a divergence here would mean the pair's two legs priced their failures differently.
        spec = announced_rule_spec(RETURN_HALF)
        own_payoff = trust_stated_rule_corner_rewards(
            spec, max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION
        )
        at_weight_zero = trust_care_corner_rewards(
            spec, max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION, alpha=0.0
        )
        assert own_payoff == pytest.approx(at_weight_zero)

    def test_a_strategy_method_row_has_no_announced_rule_corners(self) -> None:
        with pytest.raises(ValueError, match="announces no return rate"):
            trust_stated_rule_corner_rewards(
                strategy_method_spec(), max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION
            )

    def test_the_strategy_method_corners_span_the_whole_scale(self) -> None:
        spec = strategy_method_spec()
        corners = trust_self_rule_corner_rewards(
            spec, max_return_fraction=TRUST_MAX_SELF_STATED_RETURN_FRACTION
        )
        # Send everything promising nothing pays nothing; send everything promising everything back
        # pays the ceiling. Both numbers the price sits between are the scale's own ends.
        assert min(corners.values()) == pytest.approx(0.0)
        assert max(corners.values()) == pytest.approx(1.0)
        assert max(corners.values()) - min(corners.values()) == pytest.approx(
            trustor_self_rule_reward_span(
                spec, max_return_fraction=TRUST_MAX_SELF_STATED_RETURN_FRACTION
            )
        )

    def test_an_announced_rule_row_has_no_strategy_method_corners(self) -> None:
        with pytest.raises(ValueError, match="announces a return rate"):
            trust_self_rule_corner_rewards(
                announced_rule_spec(RETURN_HALF),
                max_return_fraction=TRUST_MAX_SELF_STATED_RETURN_FRACTION,
            )


class TestWorstMinEffortMatchReturn:
    """The level grid's mirror of `worst_iterated_return`, read by the same parse price."""

    def test_the_worst_match_never_beats_the_best_one(self) -> None:
        spec = MinEffortSpec(
            game_id="min-effort-under-test",
            n_levels=MIN_EFFORT_LEVELS,
            benefit_per_level=MIN_EFFORT_BENEFIT_PER_LEVEL,
            cost_per_level=0.1,
            team_size=1,
        )
        for n_rounds in (1, 3, 5):
            assert worst_min_effort_match_return(spec, n_rounds) < max_min_effort_match_return(
                spec, n_rounds
            ), n_rounds

    def test_the_worst_single_round_is_the_top_of_the_grid_against_the_opening_level(self) -> None:
        # The matcher opens at the bottom, so one round of working at the top buys nothing and pays
        # the whole cost: the worst cell on the grid, and the reward's floor for a one-round match.
        spec = MinEffortSpec(
            game_id="min-effort-under-test",
            n_levels=MIN_EFFORT_LEVELS,
            benefit_per_level=MIN_EFFORT_BENEFIT_PER_LEVEL,
            cost_per_level=0.1,
            team_size=1,
        )
        assert worst_min_effort_match_return(spec, 1) == pytest.approx(
            min_effort_cell_reward(
                spec,
                own_level=MIN_EFFORT_LEVELS,
                lowest_other_level=MIN_EFFORT_MATCHER_OPENING_LEVEL,
            )
        )

    @pytest.mark.parametrize("bad", [0, -1, 7])
    def test_round_count_outside_the_brute_force_window_raises(self, bad: int) -> None:
        spec = MinEffortSpec(
            game_id="min-effort-under-test",
            n_levels=MIN_EFFORT_LEVELS,
            benefit_per_level=MIN_EFFORT_BENEFIT_PER_LEVEL,
            cost_per_level=0.1,
            team_size=1,
        )
        with pytest.raises(ValueError, match="n_rounds must be in"):
            worst_min_effort_match_return(spec, bad)
