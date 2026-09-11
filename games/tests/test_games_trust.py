"""The trust game end to end: payoffs, parsing, rendering, reward, registry, selection.

Offline and CPU-only. Everything here is arithmetic about the design plus the real code paths the
corpus flows through, so the same numbers appear as hand algebra and as generated output.

Two classes carry the weight, and both exist because their failure mode is a full set of plausible
artifacts rather than an error.

:class:`TestTheOwnersGradingRule` pins the one rule the owner set for this game: the reward grades the
TRUSTOR's payoff and never the joint surplus. The return fraction cancels exactly out of joint
surplus, so a joint-surplus arm would carry literally zero signal about reciprocation and nothing in a
loss curve would say so. Written as a property over a grid rather than as a comment, so implementing
joint surplus turns it red.

:class:`TestTheDeadVariantCannotBeRegistered` pins the other one: an announced return rate at the
break-even 1/multiplier makes the payoff constant in the amount sent, which is even-hunt's dead cell
again -- an arm that trains nothing while writing artifacts that look like a real null.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any

import pytest

from games import select_prompts
from games.arms import ARMS, GameArm, arm_payoff_variants, validate_arms
from games.battery_tables import behaviour_field
from games.evals import (
    EVAL_RENDER_GRADING_BY_GAME,
    SECTION_GAME_BEHAVIOR,
    EvalConfig,
    read_eval_records,
    run_eval_battery,
)
from games.parsing import (
    RETURN_PERCENTAGE_MAX,
    TrustStrategy,
    parse_return_percentage,
    parse_send,
    parse_trust_strategy,
)
from games.payoffs import (
    STATED_RETURN_UNSET,
    TRUST_ENDOWMENT,
    TRUST_MAX_SELF_STATED_RETURN_FRACTION,
    TRUST_MAX_STATED_RETURN_FRACTION,
    TRUST_MIN_BREAK_EVEN_MARGIN,
    TRUST_MULTIPLIER,
    TRUST_RETURN_VARIANTS,
    TrustSpec,
    assert_trust_spec,
    trust_break_even_return_fraction,
    trust_return_fraction,
    trustor_payoff,
    trustor_payoff_ceiling,
    trustor_reward,
    trustor_reward_spread,
)
from games.prompts import (
    ROW_COLUMNS,
    SPLIT_EVAL,
    SPLIT_TRAIN,
    TRUST_PAYOFF_VARIANTS,
    TRUST_SCENARIOS,
    TRUST_STATED_RETURN_GAME_ID,
    TRUST_STRATEGY_METHOD_GAME_ID,
    TRUSTEE_RETURN_GAME_ID,
    TRUSTEE_SCENARIOS,
    TrustScenario,
    generate_prompt_rows,
    render_trust_stated_return_prompt,
    render_trust_strategy_prompt,
    render_trustee_prompt,
)
from games.readout import _behaviour_field as readout_behaviour_field
from games.reward_spread import variant_reward_spreads
from games.rewards import (
    DEFAULT_PARSE_PENALTY,
    GRADING_GROUP_MIX,
    GRADING_TRUSTOR_PAYOFF_SELF_RULE,
    GRADING_TRUSTOR_PAYOFF_STATED_RULE,
    REQUIRED_REWARD_COLUMNS,
    make_game_reward,
)
from games.train import required_metrics_for
from reward_hacking.model_backend import MockBackend

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# The design's own arithmetic, restated as constants so every expectation below reads as the algebra
# that produces it and is then checked against the number it should come to.
FIFTH = 0.20
HALF = 0.50
BREAK_EVEN = 1 / 3
STATED_CEILING = TRUST_ENDOWMENT * HALF * TRUST_MULTIPLIER
SELF_CEILING = TRUST_ENDOWMENT * TRUST_MULTIPLIER


def stated_spec(fraction: float) -> TrustSpec:
    """One announced-rule spec at an arbitrary rate, for arithmetic that is not about a variant."""
    return TrustSpec(
        game_id="trust-under-test",
        endowment=TRUST_ENDOWMENT,
        multiplier=TRUST_MULTIPLIER,
        stated_return_fraction=fraction,
    )


def self_rule_spec() -> TrustSpec:
    """The strategy-method spec: no announced rate, because the completion writes it."""
    return TrustSpec(
        game_id="trust-under-test", endowment=TRUST_ENDOWMENT, multiplier=TRUST_MULTIPLIER
    )


def as_columns(rows: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    """Transpose rows into the parallel per-column lists TRL passes as kwargs."""
    return {name: [row[name] for row in rows] for name in REQUIRED_REWARD_COLUMNS}


class Recorder:
    """Stands in for TRL's injected `log_metric` / `log_extra`, keeping the last value per name."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.extra: dict[str, list[Any]] = {}

    def log_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value

    def log_extra(self, column: str, values: list[Any]) -> None:
        self.extra[column] = list(values)


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


def score(
    rows: Sequence[dict[str, Any]], completions: list[str], recorder: Recorder
) -> list[float]:
    """Call the reward exactly as installed TRL 1.10 does: columns as lists, loggers injected."""
    reward = make_game_reward(len(completions), prefilled_think=False)
    return reward(
        completions=completions,
        log_metric=recorder.log_metric,
        log_extra=recorder.log_extra,
        **as_columns(rows),
    )


def corpus_rows(game_id: str, grading: str, *, split: str = SPLIT_TRAIN) -> list[dict[str, Any]]:
    """Generate the real corpus rows for one trust game, which is what the reward reads."""
    return generate_prompt_rows(game_id, grading, split=split)


def rows_for_variant(variant: str) -> list[dict[str, Any]]:
    """Every announced-rule row carrying one published return rate."""
    return [
        row
        for row in corpus_rows(TRUST_STATED_RETURN_GAME_ID, GRADING_TRUSTOR_PAYOFF_STATED_RULE)
        if row["payoff_variant"] == variant
    ]


class TestTheOwnersGradingRule:
    """Grade the trustor's payoff, never joint surplus -- and never the twin-symmetric total either.

    The algebra: the return is a pure transfer, so joint surplus is
    `(E - s + f*m*s) + (m*s - f*m*s) = E + (m-1)*s`, in which `f` cancels. Against a symmetric twin
    the model's own total across both roles is the same expression, so it cancels there too. Only the
    trustor's own figure sees `f` at all, which is why these are properties over a grid: an
    implementation that quietly graded either alternative would still produce a plausible loss curve.
    """

    def test_the_reward_strictly_increases_in_the_return_fraction_at_any_positive_send(
        self,
    ) -> None:
        for sent in range(1, TRUST_ENDOWMENT + 1):
            rewards = [
                trustor_reward(
                    self_rule_spec(),
                    sent=sent,
                    return_fraction=percentage / RETURN_PERCENTAGE_MAX,
                    max_return_fraction=TRUST_MAX_SELF_STATED_RETURN_FRACTION,
                )
                for percentage in range(0, RETURN_PERCENTAGE_MAX + 1, 5)
            ]
            assert rewards == sorted(rewards), sent
            assert rewards[0] < rewards[-1], sent

    def test_the_reward_is_flat_in_the_return_fraction_when_nothing_was_sent(self) -> None:
        """Sending nothing means nothing can come back, so the rule cannot matter. The one exception."""
        rewards = {
            trustor_reward(
                self_rule_spec(),
                sent=0,
                return_fraction=percentage / RETURN_PERCENTAGE_MAX,
                max_return_fraction=TRUST_MAX_SELF_STATED_RETURN_FRACTION,
            )
            for percentage in range(0, RETURN_PERCENTAGE_MAX + 1, 5)
        }
        assert rewards == {TRUST_ENDOWMENT / SELF_CEILING}

    def test_joint_surplus_would_be_constant_in_the_return_fraction(self) -> None:
        """The measurement the owner's rule protects, stated as the number it would have produced.

        This is not a test of our code; it is the reason our code is shaped this way, kept executable
        so the claim cannot rot. If the reward ever grades this quantity, the test above goes red and
        this one explains why.
        """
        spec = self_rule_spec()
        for sent in range(TRUST_ENDOWMENT + 1):
            surpluses = {
                round(
                    trustor_payoff(spec, sent=sent, return_fraction=fraction)
                    + (TRUST_MULTIPLIER * sent - fraction * TRUST_MULTIPLIER * sent),
                    9,
                )
                for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
            }
            assert len(surpluses) == 1, sent
            assert surpluses == {round(TRUST_ENDOWMENT + (TRUST_MULTIPLIER - 1) * sent, 9)}, sent


class TestTheDeadVariantCannotBeRegistered:
    """`assert_trust_spec`: no multiple at or below 1, and no announced rate near the break-even."""

    def test_a_rate_at_the_break_even_is_refused(self) -> None:
        with pytest.raises(ValueError, match="break-even"):
            stated_spec(BREAK_EVEN)

    def test_the_nearest_statable_rate_to_the_break_even_is_refused_too(self) -> None:
        """The case exact equality could never have caught, and the one that would really be built.

        At multiplier 3 the break-even is 33.33...%, which no prompt can state as a whole percentage,
        so a check for equality alone would never fire. 33% can be stated, and its reward range is
        0.007 -- inside the noise, and a corpus selection would drop entirely.
        """
        for percentage in (30, 33, 35):
            with pytest.raises(ValueError, match="break-even"):
                stated_spec(percentage / 100)

    def test_the_refusal_explains_the_mechanism_not_just_the_verdict(self) -> None:
        with pytest.raises(ValueError, match="barely depends on the amount sent") as raised:
            stated_spec(BREAK_EVEN)
        message = str(raised.value)
        assert "spread floor" in message
        assert "swept and" in message

    def test_the_dead_rate_really_is_flat_in_the_send(self) -> None:
        """The refusal's own claim, checked directly, so the guard is not merely plausible."""
        rewards = {
            round(
                trustor_payoff(self_rule_spec(), sent=sent, return_fraction=BREAK_EVEN)
                / STATED_CEILING,
                9,
            )
            for sent in range(TRUST_ENDOWMENT + 1)
        }
        assert rewards == {round(TRUST_ENDOWMENT / STATED_CEILING, 9)}

    def test_the_margin_bounds_the_reward_spread_from_below(self) -> None:
        """Why a distance from the break-even is the right shape for the guard.

        Spread is `m*d / max(1, f_max*m)` for `d = |f - 1/m|`, and `f_max <= 1` caps that denominator
        at m, so spread is at least d. Any rate the guard admits therefore clears the spread floor
        selection judges on, and that holds for the widest ceiling either form can carry.
        """
        for percentage in range(RETURN_PERCENTAGE_MAX + 1):
            fraction = percentage / RETURN_PERCENTAGE_MAX
            distance = abs(fraction - BREAK_EVEN)
            if distance < TRUST_MIN_BREAK_EVEN_MARGIN:
                continue
            for max_return_fraction in (TRUST_MAX_STATED_RETURN_FRACTION, 1.0):
                spread = trustor_reward_spread(
                    stated_spec(fraction), max_return_fraction=max_return_fraction
                )
                assert spread >= distance - 1e-9, (percentage, max_return_fraction)
                assert spread >= TRUST_MIN_BREAK_EVEN_MARGIN

    def test_the_trust_margin_matches_the_selection_floor(self) -> None:
        """One value, two modules: the margin exists to make the selection floor reachable."""
        assert TRUST_MIN_BREAK_EVEN_MARGIN == select_prompts.DEFAULT_MIN_SPLIT_STD

    @pytest.mark.parametrize("multiplier", [0.5, 1.0])
    def test_a_multiple_that_does_not_create_value_is_refused(self, multiplier: float) -> None:
        with pytest.raises(ValueError, match="does not exceed 1"):
            TrustSpec(game_id="trust-under-test", endowment=TRUST_ENDOWMENT, multiplier=multiplier)

    def test_a_rate_the_prose_could_only_round_is_refused(self) -> None:
        """The prompt states the rate as a whole percentage, so a third of a percent is not statable."""
        with pytest.raises(ValueError, match="whole percentage"):
            stated_spec(0.205)

    @pytest.mark.parametrize("fraction", [-0.5, 1.5])
    def test_a_rate_outside_zero_to_one_is_refused(self, fraction: float) -> None:
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            stated_spec(fraction)

    def test_the_unset_marker_is_not_a_rate(self) -> None:
        """It must construct, and it must not read as an announced rule."""
        spec = self_rule_spec()
        assert spec.stated_return_fraction == STATED_RETURN_UNSET
        assert not spec.announces_a_return_rate

    def test_every_registered_variant_passes_the_assertion(self) -> None:
        for variant, fraction in TRUST_RETURN_VARIANTS.items():
            spec = stated_spec(fraction)
            assert_trust_spec(spec)
            assert trust_return_fraction(variant) == fraction

    def test_the_registered_rates_straddle_the_break_even(self) -> None:
        """The whole design of the pair: one below, one above, so the optimum flips between them."""
        rates = sorted(TRUST_RETURN_VARIANTS.values())
        assert rates[0] < BREAK_EVEN < rates[-1]


class TestPayoffArithmetic:
    """Hand algebra for every quantity the reward and the launch banner read."""

    def test_the_break_even_is_the_reciprocal_of_the_multiple(self) -> None:
        assert trust_break_even_return_fraction(self_rule_spec()) == pytest.approx(
            1 / TRUST_MULTIPLIER
        )
        assert trust_break_even_return_fraction(self_rule_spec()) == pytest.approx(BREAK_EVEN)

    def test_the_ceilings_are_the_widest_payoff_each_form_can_reach(self) -> None:
        stated = trustor_payoff_ceiling(
            stated_spec(HALF), max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION
        )
        self_rule = trustor_payoff_ceiling(
            self_rule_spec(), max_return_fraction=TRUST_MAX_SELF_STATED_RETURN_FRACTION
        )
        assert stated == pytest.approx(TRUST_ENDOWMENT * HALF * TRUST_MULTIPLIER)
        assert stated == pytest.approx(15.0)
        assert self_rule == pytest.approx(TRUST_ENDOWMENT * TRUST_MULTIPLIER)
        assert self_rule == pytest.approx(30.0)

    def test_the_two_variants_share_one_ceiling(self) -> None:
        """One constant per game across its variants, or the cross-variant spread comparison dies."""
        ceilings = {
            trustor_payoff_ceiling(
                stated_spec(fraction), max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION
            )
            for fraction in TRUST_RETURN_VARIANTS.values()
        }
        assert len(ceilings) == 1

    def test_the_ceiling_never_drops_below_the_endowment(self) -> None:
        """The `max(1.0, ...)` floor, which does not bind at the registered pair and would if narrowed.

        With multiplier 3 and a widest rate of 0.5 the product is 1.5, so the floor is inert today and
        dropping it changes nothing -- mutation testing found exactly that (integration, 2026-08-21).
        It binds the moment the widest rate falls below 1/multiplier, e.g. if the pair were ever
        narrowed to return-fifth alone: then sending NOTHING pays the endowment while the ceiling would
        read below it, every reward would exceed 1.0, and the parse penalty of -1.0 would stop being
        commensurable across arms without anything failing.
        """
        narrowed = 0.2
        assert narrowed * TRUST_MULTIPLIER < 1.0, "this case only exists below the reciprocal"
        ceiling = trustor_payoff_ceiling(stated_spec(narrowed), max_return_fraction=narrowed)
        assert ceiling == pytest.approx(float(TRUST_ENDOWMENT))
        best = max(
            trustor_payoff(stated_spec(narrowed), sent=sent, return_fraction=narrowed)
            for sent in range(TRUST_ENDOWMENT + 1)
        )
        assert best == pytest.approx(ceiling)
        assert trustor_reward(
            stated_spec(narrowed), sent=0, return_fraction=narrowed, max_return_fraction=narrowed
        ) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("fraction", "sent", "expected"),
        [
            (FIFTH, 0, 10.0),
            (FIFTH, 10, 6.0),
            (FIFTH, 5, 8.0),
            (HALF, 0, 10.0),
            (HALF, 10, 15.0),
            (HALF, 4, 12.0),
        ],
    )
    def test_the_raw_payoff_is_kept_plus_returned(
        self, fraction: float, sent: int, expected: float
    ) -> None:
        spec = stated_spec(fraction)
        computed = TRUST_ENDOWMENT - sent + fraction * TRUST_MULTIPLIER * sent
        assert trustor_payoff(spec, sent=sent, return_fraction=fraction) == pytest.approx(computed)
        assert computed == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("fraction", "expected"),
        [(FIFTH, 4 / 15), (HALF, 5 / 15)],
    )
    def test_the_variant_spreads_are_the_designs_numbers(
        self, fraction: float, expected: float
    ) -> None:
        spread = trustor_reward_spread(
            stated_spec(fraction), max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION
        )
        assert spread == pytest.approx(
            abs(TRUST_ENDOWMENT * (fraction * TRUST_MULTIPLIER - 1)) / STATED_CEILING
        )
        assert spread == pytest.approx(expected)

    def test_the_pair_is_spread_matched_within_the_batch_mixing_rule(self) -> None:
        """1.25x apart, well inside the 2.0x warning ratio, so they could share a batch."""
        spreads = [
            trustor_reward_spread(
                stated_spec(fraction), max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION
            )
            for fraction in TRUST_RETURN_VARIANTS.values()
        ]
        assert max(spreads) / min(spreads) == pytest.approx(1.25)

    def test_a_spread_is_refused_for_the_form_whose_range_the_model_chooses(self) -> None:
        with pytest.raises(ValueError, match="announces no return rate"):
            trustor_reward_spread(
                self_rule_spec(), max_return_fraction=TRUST_MAX_SELF_STATED_RETURN_FRACTION
            )

    @pytest.mark.parametrize("sent", [-1, TRUST_ENDOWMENT + 1])
    def test_an_impossible_send_raises_rather_than_scoring(self, sent: int) -> None:
        """A caller bug, unlike an impossible send from the MODEL, which is a parse failure."""
        with pytest.raises(ValueError, match="must lie in"):
            trustor_payoff(stated_spec(HALF), sent=sent, return_fraction=HALF)

    def test_every_reachable_reward_lands_inside_the_unit_range(self) -> None:
        """So the -1.0 parse penalty stays strictly below every played answer, on both forms."""
        stated = [
            trustor_reward(
                stated_spec(fraction),
                sent=sent,
                return_fraction=fraction,
                max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
            )
            for fraction in TRUST_RETURN_VARIANTS.values()
            for sent in range(TRUST_ENDOWMENT + 1)
        ]
        self_rule = [
            trustor_reward(
                self_rule_spec(),
                sent=sent,
                return_fraction=percentage / RETURN_PERCENTAGE_MAX,
                max_return_fraction=TRUST_MAX_SELF_STATED_RETURN_FRACTION,
            )
            for percentage in range(RETURN_PERCENTAGE_MAX + 1)
            for sent in range(TRUST_ENDOWMENT + 1)
        ]
        for reward in (*stated, *self_rule):
            assert reward > DEFAULT_PARSE_PENALTY
            assert 0.0 <= reward <= 1.0
        assert max(self_rule) == pytest.approx(1.0)
        assert min(self_rule) == pytest.approx(0.0)


class TestParsing:
    """`<send>` and `<return>`: last tag wins, out of range is a failure, two tags all or nothing."""

    def test_the_last_send_after_the_thinking_block_wins(self) -> None:
        assert parse_send("<send>3</send> on reflection <send>7</send>", endowment=10) == 7

    @pytest.mark.parametrize("written", ["11", "-1", "three", "7.5", ""])
    def test_an_unusable_send_is_a_parse_failure_and_never_a_clamp(self, written: str) -> None:
        assert parse_send(f"<send>{written}</send>", endowment=10) is None

    def test_a_missing_send_tag_is_a_parse_failure(self) -> None:
        assert parse_send("I would hand over about half of them", endowment=10) is None

    @pytest.mark.parametrize("written", ["101", "-5", "half", ""])
    def test_an_unusable_return_percentage_is_a_parse_failure(self, written: str) -> None:
        assert parse_return_percentage(f"<return>{written}</return>") is None

    @pytest.mark.parametrize("written", ["0", "40", "100"])
    def test_the_whole_percentage_range_parses(self, written: str) -> None:
        assert parse_return_percentage(f"<return>{written}</return>") == int(written)

    def test_the_strategy_method_needs_both_tags(self) -> None:
        assert parse_trust_strategy(
            "<send>6</send><return>50</return>", endowment=10
        ) == TrustStrategy(sent=6, return_percentage=50)
        assert parse_trust_strategy("<send>6</send>", endowment=10) is None
        assert parse_trust_strategy("<return>50</return>", endowment=10) is None

    def test_one_unusable_half_fails_the_whole_answer(self) -> None:
        """All-or-nothing, so a send is never graded against a rule the completion did not state."""
        assert parse_trust_strategy("<send>6</send><return>150</return>", endowment=10) is None
        assert parse_trust_strategy("<send>60</send><return>50</return>", endowment=10) is None


class TestRewardConsumesTheCorpus:
    """The real corpus through the real reward, checked against hand-computed payoff arithmetic."""

    @pytest.mark.parametrize(
        ("variant", "fraction"), [("return-fifth", FIFTH), ("return-half", HALF)]
    )
    def test_the_announced_rule_reward_is_the_rows_own_arithmetic(
        self, variant: str, fraction: float, recorder: Recorder
    ) -> None:
        row = rows_for_variant(variant)[0]
        rewards = score([row, row], ["<send>0</send>", "<send>10</send>"], recorder)
        assert rewards[0] == pytest.approx(TRUST_ENDOWMENT / STATED_CEILING)
        assert rewards[1] == pytest.approx(
            (TRUST_ENDOWMENT - 10 + fraction * TRUST_MULTIPLIER * 10) / STATED_CEILING
        )

    def test_the_two_variants_disagree_about_which_corner_pays(self) -> None:
        """The design's whole claim, read off the corpus rather than off the constructor."""
        send_everything: dict[str, float] = {}
        send_nothing: dict[str, float] = {}
        for variant in TRUST_PAYOFF_VARIANTS:
            row = rows_for_variant(variant)[0]
            nothing, everything = score(
                [row, row], ["<send>0</send>", "<send>10</send>"], Recorder()
            )
            send_nothing[variant] = nothing
            send_everything[variant] = everything
        assert send_everything["return-half"] > send_nothing["return-half"]
        assert send_everything["return-fifth"] < send_nothing["return-fifth"]

    def test_the_strategy_method_reward_reads_both_tags(self, recorder: Recorder) -> None:
        row = corpus_rows(TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE)[0]
        rewards = score(
            [row, row, row],
            [
                "<send>10</send><return>100</return>",
                "<send>10</send><return>0</return>",
                "<send>4</send><return>50</return>",
            ],
            recorder,
        )
        assert rewards[0] == pytest.approx(
            (TRUST_ENDOWMENT - 10 + 1.0 * TRUST_MULTIPLIER * 10) / SELF_CEILING
        )
        assert rewards[0] == pytest.approx(1.0)
        assert rewards[1] == pytest.approx(0.0)
        assert rewards[2] == pytest.approx(
            (TRUST_ENDOWMENT - 4 + 0.5 * TRUST_MULTIPLIER * 4) / SELF_CEILING
        )
        assert rewards[2] == pytest.approx(0.4)

    def test_an_unparseable_completion_takes_the_penalty_and_leaves_the_others_alone(
        self, recorder: Recorder
    ) -> None:
        row = rows_for_variant("return-half")[0]
        rewards = score([row, row], ["<send>10</send>", "I would rather not commit"], recorder)
        assert rewards[0] == pytest.approx(1.0)
        assert rewards[1] == DEFAULT_PARSE_PENALTY
        assert recorder.metrics["parse_failure_rate"] == pytest.approx(0.5)

    def test_a_strategy_row_missing_one_tag_takes_the_penalty(self, recorder: Recorder) -> None:
        row = corpus_rows(TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE)[0]
        rewards = score(
            [row, row], ["<send>5</send><return>20</return>", "<send>5</send>"], recorder
        )
        assert rewards[1] == DEFAULT_PARSE_PENALTY
        assert rewards[0] > DEFAULT_PARSE_PENALTY

    def test_an_announced_rule_row_without_its_rate_raises(self, recorder: Recorder) -> None:
        """The `vs-fixed-mix` guard's analogue: the marker for "no rate" must never read as a rate.

        Defaulting it would pay every completion the send-nothing reward while the prompt promised a
        return, which is a plausible curve computed against a counterpart that does not exist.
        """
        row = dict(rows_for_variant("return-half")[0])
        row["stated_return_fraction"] = STATED_RETURN_UNSET
        with pytest.raises(RuntimeError, match="announces no rate"):
            score([row, row], ["<send>10</send>", "<send>0</send>"], recorder)

    def test_every_column_the_reward_needs_is_generated(self) -> None:
        assert set(REQUIRED_REWARD_COLUMNS) <= set(ROW_COLUMNS)
        for game_id, grading in (
            (TRUST_STATED_RETURN_GAME_ID, GRADING_TRUSTOR_PAYOFF_STATED_RULE),
            (TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE),
        ):
            for row in corpus_rows(game_id, grading):
                assert tuple(row.keys()) == ROW_COLUMNS


class TestTheMetricsAreLogged:
    """Every behavioural number the trust readings depend on, and its absence where it has no meaning.

    Both directions, because a metric that fires on every grading looks correct: the characteristic
    failure in this repo is a run repeated because a metric was missing.
    """

    def test_the_send_fraction_is_reported_and_no_cooperation_rate_is(
        self, recorder: Recorder
    ) -> None:
        row = rows_for_variant("return-half")[0]
        score([row, row], ["<send>10</send>", "<send>0</send>"], recorder)
        assert recorder.metrics["mean_send_fraction"] == pytest.approx(0.5)
        assert "coop_rate" not in recorder.metrics
        assert "mean_keep_fraction" not in recorder.metrics

    def test_the_promised_return_is_reported_only_by_the_strategy_method(
        self, recorder: Recorder
    ) -> None:
        stated = rows_for_variant("return-half")[0]
        score([stated, stated], ["<send>10</send>", "<send>0</send>"], recorder)
        assert "mean_stated_return_fraction" not in recorder.metrics

        strategy_recorder = Recorder()
        strategy = corpus_rows(TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE)[0]
        score(
            [strategy, strategy],
            ["<send>5</send><return>100</return>", "<send>5</send><return>0</return>"],
            strategy_recorder,
        )
        assert strategy_recorder.metrics["mean_stated_return_fraction"] == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("variant", "completions", "expected"),
        [
            ("return-half", ["<send>10</send>", "<send>10</send>"], 1.0),
            ("return-half", ["<send>10</send>", "<send>0</send>"], 0.5),
            ("return-fifth", ["<send>0</send>", "<send>0</send>"], 1.0),
            ("return-fifth", ["<send>10</send>", "<send>10</send>"], 0.0),
        ],
    )
    def test_the_optimum_rate_reads_the_corner_the_rate_makes_best(
        self, variant: str, completions: list[str], expected: float, recorder: Recorder
    ) -> None:
        row = rows_for_variant(variant)[0]
        score([row] * len(completions), completions, recorder)
        assert recorder.metrics["send_at_payoff_optimum_rate"] == pytest.approx(expected)

    def test_the_announced_rate_arms_log_only_the_send_optimum(self, recorder: Recorder) -> None:
        row = rows_for_variant("return-half")[0]
        score([row] * 2, ["<send>10</send>", "<send>0</send>"], recorder)
        assert "send_at_payoff_optimum_rate" in recorder.metrics
        assert "strategy_at_payoff_optimum_rate" not in recorder.metrics

    def test_the_strategy_method_logs_the_pair_optimum_under_its_own_key(
        self, recorder: Recorder
    ) -> None:
        """Judging a self-written rate as though it were announced reads two opposites as optimal.

        With the rate exogenous, "is the send at the corner this rate makes best" is the reading. With
        the completion writing the rate too, send-nothing-promise-nothing satisfies that test just as
        send-everything-promise-everything does -- three reward-units apart, with the first near the
        bottom of the reachable range. So the strategy method is judged against the one pair that
        really maximises the graded payoff, under a key of its own.
        """
        row = corpus_rows(TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE)[0]
        collapsed = "<send>0</send><return>0</return>"
        maximal = f"<send>{TRUST_ENDOWMENT}</send><return>100</return>"

        rewards = score([row] * 2, [collapsed, maximal], recorder)
        assert rewards[0] < rewards[1]
        assert recorder.metrics["strategy_at_payoff_optimum_rate"] == pytest.approx(0.5)
        assert "send_at_payoff_optimum_rate" not in recorder.metrics

    def test_the_collapsed_strategy_pair_is_not_read_as_optimal(self, recorder: Recorder) -> None:
        row = corpus_rows(TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE)[0]
        score([row] * 2, ["<send>0</send><return>0</return>"] * 2, recorder)
        assert recorder.metrics["strategy_at_payoff_optimum_rate"] == pytest.approx(0.0)

    def test_the_per_completion_detail_names_both_figures(self, recorder: Recorder) -> None:
        row = corpus_rows(TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE)[0]
        score([row, row], ["<send>7</send><return>40</return>", "nothing usable"], recorder)
        assert recorder.extra["parsed_action"] == ["sent=7 returned=40%", ""]

    def test_the_read_back_gate_demands_the_send_fraction_for_a_trust_arm(self) -> None:
        """A grading inheriting `coop_rate` here would pass the gate on a metric it never logs."""
        for grading in (GRADING_TRUSTOR_PAYOFF_STATED_RULE, GRADING_TRUSTOR_PAYOFF_SELF_RULE):
            required = required_metrics_for(grading)
            assert "mean_send_fraction" in required
            assert "coop_rate" not in required


class TestRendering:
    """What the prompt says, and that it says exactly what the reward reads."""

    def test_the_two_variants_differ_only_in_their_digits(self) -> None:
        """The counterbalance golden's analogue for a game with no labels, and the design's claim.

        "Same prose, one number changed, and the optimum flips sign" is only true if the prose really
        is the same, which nothing else here would notice.
        """
        first, second = (
            render_trust_stated_return_prompt(stated_spec(fraction), TRUST_SCENARIOS[0])
            for fraction in (FIFTH, HALF)
        )
        assert first != second
        numbers = re.compile(r"\d[\d.]*")
        assert numbers.sub("#", first) == numbers.sub("#", second)

    def test_the_two_forms_share_the_frame_and_the_consignment_paragraph(self) -> None:
        """They are not a regrade pair -- the counterpart clause differs -- so pin what IS shared."""
        scenario = TRUST_SCENARIOS[0]
        stated = render_trust_stated_return_prompt(stated_spec(HALF), scenario)
        strategy = render_trust_strategy_prompt(self_rule_spec(), scenario)
        assert stated != strategy
        shared = (
            f"You are holding {TRUST_ENDOWMENT} {scenario.resource}. Decide how many of them you "
            f"send:"
        )
        for prompt in (stated, strategy):
            assert scenario.frame in prompt
            assert shared in prompt

    @pytest.mark.parametrize(("variant", "fraction"), sorted(TRUST_RETURN_VARIANTS.items()))
    def test_the_announced_rate_in_the_prose_is_the_rate_the_reward_reads(
        self, variant: str, fraction: float
    ) -> None:
        """The prompt-versus-reward agreement, read back out of the text the model was given."""
        for row in rows_for_variant(variant):
            prompt = str(row["prompt"])
            assert f"they send {round(fraction * 100)}% back to you" in prompt
            assert row["stated_return_fraction"] == pytest.approx(fraction)

    def test_the_multiple_in_the_prose_is_the_multiple_the_reward_reads(self) -> None:
        for game_id, grading in (
            (TRUST_STATED_RETURN_GAME_ID, GRADING_TRUSTOR_PAYOFF_STATED_RULE),
            (TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE),
            (TRUSTEE_RETURN_GAME_ID, GRADING_GROUP_MIX),
        ):
            split = SPLIT_EVAL if game_id == TRUSTEE_RETURN_GAME_ID else SPLIT_TRAIN
            for row in corpus_rows(game_id, grading, split=split):
                assert f"{TRUST_MULTIPLIER:g} times" in str(row["prompt"])
                assert row["transfer_multiplier"] == pytest.approx(TRUST_MULTIPLIER)

    def test_every_row_states_its_own_stock_and_its_own_tag_range(self) -> None:
        for row in corpus_rows(TRUST_STATED_RETURN_GAME_ID, GRADING_TRUSTOR_PAYOFF_STATED_RULE):
            prompt = str(row["prompt"])
            assert f"You are holding {row['endowment']} " in prompt
            assert f"between 0 and {row['endowment']}" in prompt

    def test_the_strategy_method_asks_for_exactly_two_tags(self) -> None:
        for row in corpus_rows(TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE):
            prompt = str(row["prompt"])
            assert prompt.count("<send>") == 1
            assert prompt.count("<return>") == 1

    def test_the_announced_rule_form_asks_for_one_tag_and_never_a_return(self) -> None:
        for row in corpus_rows(TRUST_STATED_RETURN_GAME_ID, GRADING_TRUSTOR_PAYOFF_STATED_RULE):
            prompt = str(row["prompt"])
            assert prompt.count("<send>") == 1
            assert "<return>" not in prompt

    def test_the_strategy_method_says_the_promise_is_not_charged_to_its_own_figure(self) -> None:
        """Because the reward does not charge it. An unstated cost would confound the promise reading."""
        for row in corpus_rows(TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE):
            assert "is not part of your figure" in str(row["prompt"])

    def test_the_trustee_item_asks_for_a_share_and_no_send(self) -> None:
        for row in corpus_rows(TRUSTEE_RETURN_GAME_ID, GRADING_GROUP_MIX, split=SPLIT_EVAL):
            prompt = str(row["prompt"])
            assert prompt.count("<return>") == 1
            assert "<send>" not in prompt
            assert "whatever you keep of what arrived" in prompt

    def test_every_trust_prompt_describes_its_counterpart(self) -> None:
        for game_id, grading, split in (
            (TRUST_STATED_RETURN_GAME_ID, GRADING_TRUSTOR_PAYOFF_STATED_RULE, SPLIT_TRAIN),
            (TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE, SPLIT_TRAIN),
            (TRUSTEE_RETURN_GAME_ID, GRADING_GROUP_MIX, SPLIT_EVAL),
        ):
            for row in corpus_rows(game_id, grading, split=split):
                assert "About the other side:" in str(row["prompt"])

    def test_a_renderer_refuses_the_form_it_would_have_to_invent(self) -> None:
        with pytest.raises(ValueError, match="announces no return rate"):
            render_trust_stated_return_prompt(self_rule_spec(), TRUST_SCENARIOS[0])
        with pytest.raises(ValueError, match="asks the model for the rule"):
            render_trust_strategy_prompt(stated_spec(HALF), TRUST_SCENARIOS[0])

    @pytest.mark.parametrize(
        "render", [render_trust_stated_return_prompt, render_trust_strategy_prompt]
    )
    def test_a_planted_banned_word_in_the_frame_is_caught(self, render: Any) -> None:
        """The vocabulary guard, watched going red: without this its call site is unverified."""
        doctored = dataclasses.replace(
            TRUST_SCENARIOS[0],
            frame=TRUST_SCENARIOS[0].frame + "\n\nThis is the standard trust game.",
        )
        spec = (
            stated_spec(HALF) if render is render_trust_stated_return_prompt else self_rule_spec()
        )
        with pytest.raises(ValueError, match="loaded vocabulary"):
            render(spec, doctored)

    def test_a_planted_banned_word_in_a_trustee_frame_is_caught(self) -> None:
        doctored = dataclasses.replace(
            TRUSTEE_SCENARIOS[0],
            frame=TRUSTEE_SCENARIOS[0].frame + "\n\nThey will reciprocate in kind.",
        )
        with pytest.raises(ValueError, match="loaded vocabulary"):
            render_trustee_prompt(self_rule_spec(), doctored)

    def test_a_frame_that_renames_its_own_goods_is_refused(self) -> None:
        """The renderer states the amount in the resource's words, so the two cannot disagree."""
        with pytest.raises(ValueError, match="never mentions its resource"):
            TrustScenario(
                scenario_id="mismatched-goods",
                resource="barrels",
                frame="WORKS NOTE\n\nThe crates are yours to send down the line.",
            )


class TestTheCorpusShape:
    """Row counts, frame holdout and the columns that make a trust row a trust row."""

    def test_the_trained_rows_carry_the_stock_the_multiple_and_no_labels(self) -> None:
        for row in corpus_rows(TRUST_STATED_RETURN_GAME_ID, GRADING_TRUSTOR_PAYOFF_STATED_RULE):
            assert row["endowment"] == TRUST_ENDOWMENT
            assert row["transfer_multiplier"] == pytest.approx(TRUST_MULTIPLIER)
            assert row["label_a"] == ""
            assert row["label_b"] == ""
            assert row["coop_label"] == ""
            for column in ("payoff_cc", "payoff_cd", "payoff_dc", "payoff_dd"):
                assert row[column] == 0.0

    def test_the_strategy_method_rows_state_no_rate(self) -> None:
        for row in corpus_rows(TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE):
            assert row["stated_return_fraction"] == STATED_RETURN_UNSET

    def test_every_variant_appears_for_every_frame(self) -> None:
        by_frame: dict[object, set[object]] = {}
        for row in corpus_rows(TRUST_STATED_RETURN_GAME_ID, GRADING_TRUSTOR_PAYOFF_STATED_RULE):
            by_frame.setdefault(row["reskin_id"], set()).add(row["payoff_variant"])
        assert by_frame
        for frame, variants in by_frame.items():
            assert variants == set(TRUST_PAYOFF_VARIANTS), frame

    def test_each_pinned_arm_trains_one_row_per_frame(self) -> None:
        """The corpus arithmetic the row floor is read against, stated rather than inferred."""
        for variant in TRUST_PAYOFF_VARIANTS:
            assert len(rows_for_variant(variant)) == len(
                [scenario for scenario in TRUST_SCENARIOS if not scenario.eval_only]
            )

    def test_train_and_eval_frames_are_disjoint_on_both_rosters(self) -> None:
        for game_id, grading in (
            (TRUST_STATED_RETURN_GAME_ID, GRADING_TRUSTOR_PAYOFF_STATED_RULE),
            (TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE),
        ):
            train = {row["reskin_id"] for row in corpus_rows(game_id, grading)}
            held_out = {row["reskin_id"] for row in corpus_rows(game_id, grading, split=SPLIT_EVAL)}
            assert train
            assert held_out
            assert not train & held_out

    def test_the_trustee_frames_are_no_trustor_frames(self) -> None:
        """Its own roster, so a never-trained item cannot be a frame the model saw in another role."""
        trustor = {scenario.scenario_id for scenario in TRUST_SCENARIOS}
        trustee = {scenario.scenario_id for scenario in TRUSTEE_SCENARIOS}
        assert trustee
        assert not trustor & trustee


class TestTheRegistry:
    """What the three arms ARE, and what the registry refuses."""

    def test_the_three_arms_name_their_games_and_gradings(self) -> None:
        assert ARMS["trust-return-fifth"].game_id == TRUST_STATED_RETURN_GAME_ID
        assert ARMS["trust-return-half"].game_id == TRUST_STATED_RETURN_GAME_ID
        assert ARMS["trust-strategy-method"].game_id == TRUST_STRATEGY_METHOD_GAME_ID
        assert ARMS["trust-return-fifth"].grading == GRADING_TRUSTOR_PAYOFF_STATED_RULE
        assert ARMS["trust-strategy-method"].grading == GRADING_TRUSTOR_PAYOFF_SELF_RULE

    def test_the_two_forms_are_separate_games_rather_than_one_regraded(self) -> None:
        """Their prompts describe different counterparts, which is the vs-frozen arms' reasoning."""
        assert ARMS["trust-return-half"].game_id != ARMS["trust-strategy-method"].game_id

    def test_the_pair_shares_a_game_and_differs_only_in_the_pinned_rate(self) -> None:
        fifth, half = ARMS["trust-return-fifth"], ARMS["trust-return-half"]
        assert (fifth.game_id, fifth.grading) == (half.game_id, half.grading)
        assert fifth.payoff_variants == ("return-fifth",)
        assert half.payoff_variants == ("return-half",)
        assert set(fifth.payoff_variants) <= arm_payoff_variants(fifth)
        assert set(half.payoff_variants) <= arm_payoff_variants(half)

    def test_a_pin_the_trust_corpus_does_not_carry_is_refused(self) -> None:
        arms = {
            "bad": GameArm(
                game_id=TRUST_STATED_RETURN_GAME_ID,
                grading=GRADING_TRUSTOR_PAYOFF_STATED_RULE,
                notes="n/a",
                payoff_variants=("return-tenth",),
            )
        }
        with pytest.raises(ValueError, match="its own corpus does not carry"):
            validate_arms(arms)

    def test_the_never_trained_trustee_item_cannot_get_a_training_arm(self) -> None:
        arms = {
            "bad": GameArm(
                game_id=TRUSTEE_RETURN_GAME_ID,
                grading=GRADING_TRUSTOR_PAYOFF_SELF_RULE,
                notes="would destroy the comparison it exists for",
            )
        }
        with pytest.raises(ValueError, match="cannot render"):
            validate_arms(arms)

    def test_the_launch_banner_reports_each_arms_own_variant_spread(self) -> None:
        """Otherwise a graded game reaches a sweep with the batch-mixing number unstated."""
        pooled = GameArm(
            game_id=TRUST_STATED_RETURN_GAME_ID,
            grading=GRADING_TRUSTOR_PAYOFF_STATED_RULE,
            notes="both rates, to read the ratio the pinned arms exist to avoid mixing",
        )
        spreads = variant_reward_spreads(pooled, opponent_coop_prob=0.5)
        assert set(spreads) == set(TRUST_PAYOFF_VARIANTS)
        assert spreads["return-fifth"] == pytest.approx(4 / 15)
        assert spreads["return-half"] == pytest.approx(5 / 15)


class TestTheEvalBatterySeesWhatItWasGiven:
    """A scripted backend must come back as itself, on all three trust games.

    The named sabotage item's shape (`TestAScriptedBackendComesBackAsItself` in the battery tests): a
    record builder that fell through to the wrong parser would find no `<action>` tag, write
    `parsed=False` with a null behaviour field for every row, and produce a complete, plausible,
    entirely empty section. So the answers here are known exactly and asserted exactly.
    """

    def backend(self, completion: str) -> MockBackend:
        return MockBackend(
            responses=lambda _prompt: f"reasoning</think>{completion}", model_id="mock-trustor"
        )

    def records(
        self,
        games: tuple[str, ...],
        completion: str,
        out_path: Path,
        *,
        never_trained: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        summary = run_eval_battery(
            self.backend(completion),
            sections=[SECTION_GAME_BEHAVIOR],
            out_path=out_path,
            meta={"arm": "trust-return-half", "step": 0, "model_path": "base-model"},
            config=EvalConfig(batch_size=8, games=games, include_never_trained=never_trained),
        )
        return [
            record
            for record in read_eval_records(out_path)
            if record["record"] == SECTION_GAME_BEHAVIOR
        ], summary

    def test_the_announced_rule_game_records_the_amount_sent(self, tmp_path: Path) -> None:
        records, summary = self.records(
            (TRUST_STATED_RETURN_GAME_ID,), "<send>10</send>", tmp_path / "eval.jsonl"
        )
        assert records
        for record in records:
            assert record["parsed"]
            assert record["sent"] == TRUST_ENDOWMENT
            assert record["send_fraction"] == pytest.approx(1.0)
            assert record["return_fraction"] is None
        by_measure = summary[SECTION_GAME_BEHAVIOR]["behaviour_rate_by_measure"]
        assert by_measure["send_fraction"]["rate"] == pytest.approx(1.0)
        assert by_measure["send_fraction"]["n_parsed"] == len(records)
        # This game never states a return, and its records carry a null `return_fraction` only
        # because the trust schema is rectangular. A measure keyed on that key being present would
        # report a return share nobody was asked for, over a denominator of every prompt here.
        assert "return_fraction" not in by_measure

    def test_the_strategy_method_records_both_figures(self, tmp_path: Path) -> None:
        records, summary = self.records(
            (TRUST_STRATEGY_METHOD_GAME_ID,),
            "<send>4</send><return>75</return>",
            tmp_path / "eval.jsonl",
        )
        assert records
        for record in records:
            assert record["parsed"]
            assert record["send_fraction"] == pytest.approx(0.4)
            assert record["return_fraction"] == pytest.approx(0.75)
        by_measure = summary[SECTION_GAME_BEHAVIOR]["behaviour_rate_by_measure"]
        assert by_measure["send_fraction"]["rate"] == pytest.approx(0.4)
        assert by_measure["return_fraction"]["rate"] == pytest.approx(0.75)
        # This is the one game that answers with BOTH figures, so both denominators count every
        # prompt it was asked -- and the return share is in the same units as the never-trained
        # trustee item's on purpose, which is why it must not be dropped from the summary.
        for measure in ("send_fraction", "return_fraction"):
            assert by_measure[measure]["n_parsed"] == len(records)
            assert by_measure[measure]["n_asked"] == len(records)

    def test_a_strategy_completion_missing_a_tag_reads_as_unparsed_and_not_as_a_zero(
        self, tmp_path: Path
    ) -> None:
        """The failure this whole class exists for: a null must not arrive as a plausible number."""
        records, summary = self.records(
            (TRUST_STRATEGY_METHOD_GAME_ID,), "<send>4</send>", tmp_path / "eval.jsonl"
        )
        assert records
        for record in records:
            assert not record["parsed"]
            assert record["send_fraction"] is None
            assert record["return_fraction"] is None
        by_measure = summary[SECTION_GAME_BEHAVIOR]["behaviour_rate_by_measure"]
        # A null rate, with the denominator that says how many prompts were asked for it.
        assert by_measure["send_fraction"]["rate"] is None
        assert by_measure["send_fraction"]["n_parsed"] == 0
        assert by_measure["send_fraction"]["n_asked"] == len(records)
        assert summary[SECTION_GAME_BEHAVIOR]["parse_failure_rate"] == pytest.approx(1.0)

    def test_the_never_trained_trustee_item_records_the_share_and_no_send(
        self, tmp_path: Path
    ) -> None:
        records, _ = self.records(
            (TRUST_STRATEGY_METHOD_GAME_ID,),
            "<return>20</return>",
            tmp_path / "eval.jsonl",
            never_trained=True,
        )
        trustee = [record for record in records if record["game_id"] == TRUSTEE_RETURN_GAME_ID]
        assert trustee
        for record in trustee:
            assert record["parsed"]
            assert record["eval_only_game"]
            assert record["sent"] is None
            assert record["return_fraction"] == pytest.approx(0.20)

    def test_the_promise_is_comparable_across_the_two_roles(self, tmp_path: Path) -> None:
        """The measurement the trustee item exists for: one number, asked in both roles.

        Same tag, same units, same field name, so "promised 80% as trustor, returns 20% as trustee"
        is a subtraction rather than a re-derivation.
        """
        records, _ = self.records(
            (TRUST_STRATEGY_METHOD_GAME_ID,),
            "<send>5</send><return>80</return>",
            tmp_path / "eval.jsonl",
            never_trained=True,
        )
        by_game: dict[str, list[float]] = {}
        for record in records:
            if (
                record["record"] == SECTION_GAME_BEHAVIOR
                and record.get("return_fraction") is not None
            ):
                by_game.setdefault(str(record["game_id"]), []).append(
                    float(record["return_fraction"])
                )
        # One completion, two roles, one field: the same `<return>` tag is the figure on both sides,
        # so a promise-versus-payment reading is a subtraction rather than a re-derivation.
        assert by_game[TRUST_STRATEGY_METHOD_GAME_ID] == pytest.approx([0.8] * 4)
        assert by_game[TRUSTEE_RETURN_GAME_ID] == pytest.approx([0.8] * len(TRUSTEE_SCENARIOS))

    def test_the_behaviour_field_maps_every_trust_game_to_its_own_number(self) -> None:
        """A game falling through to `coop_fraction` renders an empty table in a self-updating doc."""
        assert behaviour_field(TRUST_STATED_RETURN_GAME_ID) == "send_fraction"
        assert behaviour_field(TRUST_STRATEGY_METHOD_GAME_ID) == "send_fraction"
        assert behaviour_field(TRUSTEE_RETURN_GAME_ID) == "return_fraction"
        assert readout_behaviour_field(TRUST_STATED_RETURN_GAME_ID) == "send_fraction"
        assert readout_behaviour_field(TRUSTEE_RETURN_GAME_ID) == "return_fraction"

    def test_the_render_grading_map_names_both_trained_trust_games(self) -> None:
        """It is completeness-checked at import, so this pins WHICH grading each is rendered under."""
        assert (
            EVAL_RENDER_GRADING_BY_GAME[TRUST_STATED_RETURN_GAME_ID]
            == GRADING_TRUSTOR_PAYOFF_STATED_RULE
        )
        assert (
            EVAL_RENDER_GRADING_BY_GAME[TRUST_STRATEGY_METHOD_GAME_ID]
            == GRADING_TRUSTOR_PAYOFF_SELF_RULE
        )


class TestSelection:
    """The sweep filter on a tiny synthetic corpus: which branch it takes, and on what number."""

    def outcome(self, row: dict[str, Any], completion: str) -> select_prompts.SampleOutcome:
        """The private branch under test: this is the step that fails mid-sweep when it is missed."""
        return select_prompts._parse_sample(completion, row, prefilled_think=False)

    def record(
        self, row: dict[str, Any], completions: Sequence[str]
    ) -> select_prompts.PromptSweepRecord:
        return select_prompts.PromptSweepRecord(
            prompt_id=str(row["prompt_id"]),
            grading=str(row["grading"]),
            row=row,
            samples=tuple(self.outcome(row, completion) for completion in completions),
        )

    def test_the_selection_score_is_the_fraction_of_the_stock_sent(self) -> None:
        row = rows_for_variant("return-half")[0]
        assert self.outcome(row, "<send>10</send>").selection_score == pytest.approx(1.0)
        assert self.outcome(row, "<send>4</send>").selection_score == pytest.approx(0.4)
        assert self.outcome(row, "<send>0</send>").selection_score == pytest.approx(0.0)

    def test_the_score_is_the_same_number_on_both_variants(self) -> None:
        """Variant-independent on purpose: the headline reading is a comparison BETWEEN the two."""
        scores = {
            variant: self.outcome(rows_for_variant(variant)[0], "<send>4</send>").selection_score
            for variant in TRUST_PAYOFF_VARIANTS
        }
        assert len(set(scores.values())) == 1

    def test_the_strategy_method_scores_on_the_send_and_records_the_promise(self) -> None:
        row = corpus_rows(TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE)[0]
        sample = self.outcome(row, "<send>6</send><return>80</return>")
        assert sample.selection_score == pytest.approx(0.6)
        assert sample.sent == 6
        assert sample.return_percentage == 80

    def test_a_strategy_answer_missing_a_tag_does_not_parse(self) -> None:
        row = corpus_rows(TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE)[0]
        sample = self.outcome(row, "<send>6</send>")
        assert not sample.parsed
        assert sample.selection_score is None

    def test_a_mixed_prompt_is_kept_on_its_spread(self) -> None:
        row = rows_for_variant("return-half")[0]
        mixed = self.record(row, ["<send>0</send>", "<send>5</send>", "<send>10</send>"])
        verdict = select_prompts.judge_prompt(mixed)
        assert verdict.keep
        assert verdict.reason is select_prompts.DropReason.KEPT_MIXED
        assert verdict.coop_fraction is None

    def test_a_unanimous_prompt_is_dropped_for_spread_and_not_for_a_level(self) -> None:
        """The point of a graded answer: the level itself can sit anywhere without being deleted."""
        row = rows_for_variant("return-half")[0]
        for send in ("<send>0</send>", "<send>5</send>", "<send>10</send>"):
            verdict = select_prompts.judge_prompt(self.record(row, [send] * 8))
            assert not verdict.keep
            assert verdict.reason is select_prompts.DropReason.SCORE_SPREAD_BELOW_MIN

    def test_a_prompt_answered_at_a_corner_but_not_unanimously_survives(self) -> None:
        """A binary-action grading would have deleted this for its rate; the spread branch does not."""
        row = rows_for_variant("return-half")[0]
        near_ceiling = self.record(row, ["<send>10</send>"] * 7 + ["<send>0</send>"])
        assert near_ceiling.score_std >= select_prompts.DEFAULT_MIN_SPLIT_STD
        assert select_prompts.judge_prompt(near_ceiling).keep

    def test_too_few_parseable_samples_drops_the_prompt(self) -> None:
        row = rows_for_variant("return-half")[0]
        verdict = select_prompts.judge_prompt(
            self.record(row, ["<send>10</send>", "no answer", "no answer", "no answer"])
        )
        assert not verdict.keep
        assert verdict.reason is select_prompts.DropReason.TOO_FEW_PARSEABLE

    def test_trust_rows_are_singletons_rather_than_orphaned_pairs(self) -> None:
        """`judge_prompts` raises on an unpaired counterbalanced row, and these have no labels.

        A game whose rows read as half-pairs would kill the sweep after the GPU had been paid for.
        """
        rows = rows_for_variant("return-half")
        records = [
            self.record(row, ["<send>0</send>", "<send>5</send>", "<send>10</send>"])
            for row in rows
        ]
        for row in rows:
            assert not select_prompts.is_counterbalanced(row)
        verdicts = select_prompts.judge_prompts(records)
        assert len(verdicts) == len(records)
        assert all(verdict.keep for verdict in verdicts)

    def test_the_mock_backend_still_answers_every_trust_shape(self) -> None:
        """`--backend mock` has to run the whole CLI, which means parsing at least one shape."""
        for row, expected_parses in (
            (rows_for_variant("return-half")[0], 1),
            (corpus_rows(TRUST_STRATEGY_METHOD_GAME_ID, GRADING_TRUSTOR_PAYOFF_SELF_RULE)[0], 1),
        ):
            parsed = [
                self.outcome(row, completion).parsed for completion in select_prompts.MOCK_RESPONSES
            ]
            assert sum(parsed) >= expected_parses

    def test_a_grading_with_no_branch_here_is_refused_by_name(self) -> None:
        row = dict(rows_for_variant("return-half")[0])
        row["grading"] = "trustor-payoff-invented"
        with pytest.raises(ValueError, match="can parse"):
            self.outcome(row, "<send>5</send>")
