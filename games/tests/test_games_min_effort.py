"""The minimum-effort game and its repeated companion, against hand arithmetic.

Offline and CPU-only: no model, no trainer, no GPU. Every expected figure below is written as the
arithmetic that produces it and then additionally asserted against a literal, because the first line
says why the number is what it is and the second catches an algebra error in the test itself.

Four classes carry the weight, and each one is the test whose absence would let a specific silent
failure through:

:class:`TestTheDesignsComputedTableIsReproduced` pins the four reward spreads and the four best
responses the arms were chosen on. Every one of those numbers was computed by closed-form evaluation
before any code existed, so the code agreeing with them is an independent check rather than a
tautology -- and if a later edit moves any of them, the registry's two pinned cells stop being the
two widest and the whole 2x2 design silently stops holding.

:class:`TestTheProseAndTheGradingReadOneColumn` is the guard the design calls the one that matters
most: a prompt that says three counterparts while the reward assumes two is a lie no loss curve could
show. It reads the count back out of the rendered text and compares it with the column the scorer
raises as its exponent.

:class:`TestTheAffineMapPreservesEveryComparison` is the licence for the one deliberate departure
from `games.payoffs._normalised_spec`: a shift does not preserve payoff ratios, but it does preserve
the ORDERING of differences, which is what every best response and every GRPO advantage depends on.

:class:`TestTheMatchOptimumIsTheTopOfTheGrid` is what makes the repeated arm a diagnostic rather than
a description: against a level-matcher, dropping the final level only ever loses money, so an
observed drop cannot be correct reasoning.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any

import pytest

from games.arms import ARMS, SELF_CONSISTENT_GRADINGS, SPAN_CHECK_EXEMPTIONS, GameArm, validate_arms
from games.min_effort_spread import (
    assert_min_effort_reward_can_train,
    level_reward_surface,
    min_effort_spread_from_records,
)
from games.parsing import parse_level, parse_level_sequence
from games.payoffs import (
    MAX_MIN_EFFORT_BRUTE_FORCE_ROUNDS,
    MIN_EFFORT_BENEFIT_PER_LEVEL,
    MIN_EFFORT_LEVELS,
    MIN_EFFORT_MATCH_ROUNDS,
    MIN_EFFORT_MATCHER_OPENING_LEVEL,
    MIN_EFFORT_VARIANTS,
    MinEffortSpec,
    assert_min_effort,
    max_min_effort_match_return,
    min_effort_best_response,
    min_effort_cell_reward,
    min_effort_expected_lowest,
    min_effort_group_reward,
    min_effort_group_reward_span,
    min_effort_match_reward,
    min_effort_match_reward_total,
    min_effort_matcher_levels,
    min_effort_pressure_gaps,
    min_effort_raise_probability,
    min_effort_raw_payoff,
    min_effort_reference_levels,
    min_effort_reward_floor,
    min_effort_reward_from_raw,
    min_effort_reward_scale,
    simulate_min_effort_match,
)
from games.prompts import (
    ITERATED_N_ROUNDS,
    ITERATED_PD_GRIM_GAME_ID,
    ITERATED_STAG_GAME_ID,
    MIN_EFFORT_GAME_ID,
    MIN_EFFORT_GAME_IDS,
    MIN_EFFORT_MATCH_GAME_ID,
    MIN_EFFORT_MATCH_PAYOFF_VARIANT,
    MIN_EFFORT_PAYOFF_VARIANTS,
    MIN_EFFORT_SCENARIOS,
    ROW_COLUMNS,
    SPLIT_EVAL,
    SPLIT_TRAIN,
    SPLITS,
    UNLABELLED_GAME_IDS,
    MinEffortScenario,
    assert_no_loaded_vocabulary,
    format_points,
    generate_prompt_rows,
    min_effort_spec_for,
    min_effort_variant,
    render_min_effort_match_prompt,
    render_min_effort_prompt,
)
from games.reward_spread import min_effort_variant_reward_spreads
from games.rewards import (
    DEFAULT_PARSE_PENALTY,
    GRADING_LEVEL_MATCH_RETURN,
    GRADING_MIN_EFFORT_GROUP_MIX,
    MIN_EFFORT_GRADINGS,
    make_game_reward,
)
from games.select_prompts import (
    DEFAULT_MIN_SPLIT_STD,
    ONE_SHOT_ACTION_GRADINGS,
    PARSEABLE_GRADINGS,
)
from games.train import OPTIONAL_METRICS, required_metrics_for

if TYPE_CHECKING:
    from collections.abc import Sequence

# The registered cheap cell: one counterpart, a tenth of a benefit unit per level of effort.
CHEAP_PAIR = "cheap-effort-pair"
COSTLY_CREW = "costly-effort-crew"

# The reward spreads and best responses the design computed for all four cells against a flat
# counterpart distribution, before any of this existed. Not derived from the code, on purpose: this
# table is the independent statement the code is checked against, and it is what the two registered
# arms were chosen on.
COMPUTED_SPREAD_AND_TARGET: dict[str, tuple[float, int]] = {
    "cheap-effort-pair": (0.400, 5),
    "costly-effort-crew": (0.303, 2),
    "cheap-effort-crew": (0.132, 3),
    "costly-effort-pair": (0.100, 3),
}

# The design's raw match arithmetic at the two cost ratios: the all-top total, and what dropping the
# final round to the bottom of the grid costs out of it.
COMPUTED_MATCH_TOTALS: dict[float, tuple[float, float]] = {0.5: (8.50, 2.00), 0.1: (18.50, 3.60)}


def spec(variant: str = CHEAP_PAIR, game_id: str = MIN_EFFORT_GAME_ID) -> MinEffortSpec:
    """Build one registered variant's spec, the way the corpus builder does."""
    return min_effort_spec_for(game_id, variant)


def make_row(variant: str = CHEAP_PAIR, **overrides: Any) -> dict[str, Any]:
    """Return one real generated minimum-effort row, overrides naming the column being varied."""
    rows = [
        row
        for row in generate_prompt_rows(
            MIN_EFFORT_GAME_ID, GRADING_MIN_EFFORT_GROUP_MIX, split=SPLIT_TRAIN
        )
        if row["payoff_variant"] == variant
    ]
    assert rows, variant
    return {**rows[0], **overrides}


def score(
    rows: list[dict[str, Any]], completions: list[str], *, leave_one_out: bool = False
) -> tuple[list[float], dict[str, float], dict[str, list[Any]]]:
    """Call the reward exactly the way TRL does, and return the rewards plus what was logged."""
    metrics: dict[str, float] = {}
    extra: dict[str, list[Any]] = {}

    def record_metric(name: str, value: float) -> None:
        metrics[name] = value

    def record_extra(name: str, values: list[Any]) -> None:
        extra[name] = list(values)

    reward = make_game_reward(len(completions), prefilled_think=False, leave_one_out=leave_one_out)
    columns = {
        column: [row[column] for row in rows] for column in ROW_COLUMNS if column != "prompt"
    }
    rewards = reward(
        completions=completions,
        log_metric=record_metric,
        log_extra=record_extra,
        **columns,
    )
    return rewards, metrics, extra


def level_answer(level: int) -> str:
    """Render the one-shot answer a level would be written as."""
    return f"<think>weighing it up</think><level>{level}</level>"


def match_answer(levels: Sequence[int]) -> str:
    """Render the repeated form's answer: one tag per round, in round order."""
    tags = "".join(f"<level>{level}</level>" for level in levels)
    return f"<think>planning the match</think>{tags}"


class TestTheDesignsComputedTableIsReproduced:
    """The four cells' spreads and best responses, against figures computed before any code existed.

    These are the numbers the 2x2 was designed on: the two widest cells are registered as arms, they
    push in opposite directions, and the two narrow ones are left out as gradient-compressed. A code
    change that moved any of them would quietly invalidate all of that while every test that derives
    its expectation from the code kept passing, which is why this table is written out by hand.
    """

    @pytest.mark.parametrize("variant", sorted(COMPUTED_SPREAD_AND_TARGET))
    def test_each_cell_has_the_computed_spread_and_target(self, variant: str) -> None:
        expected_spread, expected_target = COMPUTED_SPREAD_AND_TARGET[variant]
        game = spec(variant)
        reference = min_effort_reference_levels(game)
        assert min_effort_group_reward_span(game, reference) == pytest.approx(
            expected_spread, abs=5e-4
        )
        assert min_effort_best_response(game, reference) == expected_target

    def test_the_two_registered_arms_are_the_two_widest_cells(self) -> None:
        by_spread = sorted(
            COMPUTED_SPREAD_AND_TARGET,
            key=lambda variant: COMPUTED_SPREAD_AND_TARGET[variant][0],
            reverse=True,
        )
        registered = {
            variant
            for arm in ARMS.values()
            if arm.grading == GRADING_MIN_EFFORT_GROUP_MIX
            for variant in arm.payoff_variants
        }
        assert registered == set(by_spread[:2])

    def test_the_registered_pair_pushes_in_opposite_directions(self) -> None:
        """The whole design: same frames, and the pressure points at opposite ends of the grid."""
        cheap, costly = spec(CHEAP_PAIR), spec(COSTLY_CREW)
        assert (
            min_effort_best_response(cheap, min_effort_reference_levels(cheap)) == MIN_EFFORT_LEVELS
        )
        assert min_effort_best_response(costly, min_effort_reference_levels(costly)) < 3

    def test_the_registered_pair_is_spread_matched_within_the_batch_mixing_rule(self) -> None:
        # 0.400 against 0.303 is 1.3x, under the 2.0x ratio games.reward_spread warns at, so the two
        # could share a batch. They are still separate arms, so the trained direction is attributable
        # to the knob rather than averaged across it.
        cheap = COMPUTED_SPREAD_AND_TARGET[CHEAP_PAIR][0]
        costly = COMPUTED_SPREAD_AND_TARGET[COSTLY_CREW][0]
        assert cheap / costly < 2.0

    def test_the_reward_spread_report_agrees_with_the_computed_table(self) -> None:
        """The launch banner has to print the same numbers this table was designed on."""
        for name, arm in ARMS.items():
            if arm.grading != GRADING_MIN_EFFORT_GROUP_MIX:
                continue
            spreads = min_effort_variant_reward_spreads(arm)
            assert set(spreads) == set(arm.payoff_variants), name
            for variant, measured in spreads.items():
                assert measured == pytest.approx(
                    COMPUTED_SPREAD_AND_TARGET[variant][0], abs=5e-4
                ), name


class TestTheAffineMapIntoTheRewardRange:
    """The one deliberate departure from `_normalised_spec`, and why the shift is safe here."""

    def test_the_scale_is_the_same_constant_at_every_cost_ratio(self) -> None:
        """One constant per game across its variants, which is what keeps the spreads comparable."""
        scales = {min_effort_reward_scale(spec(variant)) for variant in MIN_EFFORT_PAYOFF_VARIANTS}
        assert scales == {MIN_EFFORT_BENEFIT_PER_LEVEL * (MIN_EFFORT_LEVELS - 1)}
        assert scales == {4.0}

    def test_the_raw_payoff_goes_negative_so_scaling_alone_cannot_reach_the_range(self) -> None:
        """The reason a shift is needed at all: no positive scaling maps a negative into [0,1]."""
        costly = spec(COSTLY_CREW)
        worst = min_effort_raw_payoff(costly, own_level=MIN_EFFORT_LEVELS, lowest_other_level=1)
        assert worst == pytest.approx(1.0 - 0.5 * 5)
        assert worst < 0

    @pytest.mark.parametrize("variant", sorted(MIN_EFFORT_VARIANTS))
    def test_every_cell_of_every_variant_lands_inside_the_reward_range(self, variant: str) -> None:
        game = spec(variant)
        for own in game.levels:
            for lowest in game.levels:
                reward = min_effort_cell_reward(game, own_level=own, lowest_other_level=lowest)
                assert 0.0 <= reward <= 1.0

    def test_the_penalty_stays_strictly_below_every_played_answer(self) -> None:
        game = spec(COSTLY_CREW)
        worst = min(
            min_effort_cell_reward(game, own_level=own, lowest_other_level=lowest)
            for own in game.levels
            for lowest in game.levels
        )
        assert worst > DEFAULT_PARSE_PENALTY

    def test_a_raw_payoff_outside_the_range_is_refused_rather_than_clamped(self) -> None:
        game = spec()
        with pytest.raises(ValueError, match="outside"):
            min_effort_reward_from_raw(game, min_effort_reward_floor(game) - 1.0)
        with pytest.raises(ValueError, match="commensurable"):
            min_effort_reward_from_raw(game, min_effort_reward_floor(game) + 100.0)


class TestTheAffineMapPreservesEveryComparison:
    """A shift does not preserve payoff RATIOS, and it does not have to: it preserves orderings.

    `_normalised_spec`'s docstring argues against a shift on ratio grounds. What every best response,
    every mixed-strategy comparison and every GRPO advantage actually depends on is the ordering of
    DIFFERENCES, which a positive affine transform leaves alone -- and GRPO subtracts the group mean,
    so the constant cancels exactly. Asserted over the whole grid rather than argued.
    """

    @pytest.mark.parametrize("variant", sorted(MIN_EFFORT_VARIANTS))
    def test_the_ordering_of_reward_differences_matches_the_raw_payoffs(self, variant: str) -> None:
        game = spec(variant)
        cells = [(own, lowest) for own in game.levels for lowest in game.levels]
        raw = {
            cell: min_effort_raw_payoff(game, own_level=cell[0], lowest_other_level=cell[1])
            for cell in cells
        }
        mapped = {
            cell: min_effort_cell_reward(game, own_level=cell[0], lowest_other_level=cell[1])
            for cell in cells
        }
        assert sorted(cells, key=lambda cell: raw[cell]) == sorted(
            cells, key=lambda cell: mapped[cell]
        )
        for left in cells:
            for right in cells:
                raw_gap = raw[left] - raw[right]
                mapped_gap = mapped[left] - mapped[right]
                assert mapped_gap == pytest.approx(raw_gap / min_effort_reward_scale(game))

    @pytest.mark.parametrize("variant", sorted(MIN_EFFORT_VARIANTS))
    def test_flipping_the_shift_sign_pushes_rewards_out_of_the_range(self, variant: str) -> None:
        """The sabotage this test exists for, as a permanent case rather than a one-off.

        Adding the floor where the map subtracts it is the single-character version of getting the
        shift wrong, and it escapes the range on every variant -- upward where the floor is positive
        (the cheap ratio, whose best cell lands at 1.25) and downward where it is negative (the costly
        ratio, whose worst cell lands at -0.75). Written against the arithmetic rather than by
        patching, so the test states what the wrong map would produce instead of trusting a
        monkeypatch to reproduce it, and then feeds the real map the raw value that would come out of
        it to watch the range check refuse.
        """
        game = spec(variant)
        floor = min_effort_reward_floor(game)
        scale = min_effort_reward_scale(game)
        cells = [
            min_effort_raw_payoff(game, own_level=own, lowest_other_level=lowest)
            for own in game.levels
            for lowest in game.levels
        ]
        escaped = [raw for raw in cells if not 0.0 <= (raw + floor) / scale <= 1.0]
        assert escaped, variant
        for raw in escaped:
            with pytest.raises(ValueError, match="outside"):
                min_effort_reward_from_raw(game, raw + 2 * floor)


class TestTheShapeAssertion:
    """`assert_min_effort` refuses every spec whose reward would not move with the level."""

    def test_the_registered_variants_all_pass(self) -> None:
        for variant in MIN_EFFORT_PAYOFF_VARIANTS:
            assert_min_effort(spec(variant))

    def test_a_cost_above_the_benefit_is_refused(self) -> None:
        # The sabotage the design names first: at c > b a higher shared level pays LESS, so the best
        # self-consistent answer is the bottom of the grid and every prediction is inverted.
        with pytest.raises(ValueError, match="higher level everybody shares pays LESS"):
            MinEffortSpec(
                game_id="min-effort",
                n_levels=5,
                benefit_per_level=1.0,
                cost_per_level=1.5,
                team_size=1,
            )

    def test_a_cost_equal_to_the_benefit_is_refused_too(self) -> None:
        with pytest.raises(ValueError, match="benefit_per_level > cost_per_level"):
            MinEffortSpec(
                game_id="min-effort",
                n_levels=5,
                benefit_per_level=1.0,
                cost_per_level=1.0,
                team_size=1,
            )

    def test_a_free_top_level_is_refused(self) -> None:
        with pytest.raises(ValueError, match="biggest number"):
            MinEffortSpec(
                game_id="min-effort",
                n_levels=5,
                benefit_per_level=1.0,
                cost_per_level=0.0,
                team_size=1,
            )

    def test_a_one_level_grid_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one level to write"):
            MinEffortSpec(
                game_id="min-effort",
                n_levels=1,
                benefit_per_level=1.0,
                cost_per_level=0.1,
                team_size=1,
            )

    def test_a_team_of_nobody_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no counterpart can hold the minimum down"):
            MinEffortSpec(
                game_id="min-effort",
                n_levels=5,
                benefit_per_level=1.0,
                cost_per_level=0.1,
                team_size=0,
            )

    def test_an_unknown_variant_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown minimum-effort payoff_variant"):
            min_effort_variant("expensive-effort-throng")


class TestTheExpectedMinimumIsExact:
    """The closed form, against the sum it replaces and against hand arithmetic."""

    def test_the_closed_form_matches_the_brute_force_expectation(self) -> None:
        """Two independent statements of one quantity: the CDF product against every draw enumerated."""
        game = spec("cheap-effort-crew")
        levels = [1, 2, 2, 4, 5, 5, 3, 1]
        for own in game.levels:
            enumerated = (
                sum(
                    min(own, first, second, third)
                    for first in levels
                    for second in levels
                    for third in levels
                )
                / len(levels) ** 3
            )
            assert min_effort_expected_lowest(
                game, own_level=own, counterpart_levels=levels
            ) == pytest.approx(enumerated)

    def test_a_flat_distribution_against_one_counterpart_is_the_hand_computed_sum(self) -> None:
        # sum over k=1..e of the share at or above k: 1.0, 0.8, 0.6, 0.4, 0.2 cumulated.
        game = spec(CHEAP_PAIR)
        expected = {1: 1.0, 2: 1.8, 3: 2.4, 4: 2.8, 5: 3.0}
        for own, total in expected.items():
            assert min_effort_expected_lowest(
                game, own_level=own, counterpart_levels=[1, 2, 3, 4, 5]
            ) == pytest.approx(total)

    def test_a_bigger_team_lowers_the_expected_minimum_at_every_level(self) -> None:
        """Team size as an exponent, which is why a bigger team coordinates worse."""
        levels = [1, 2, 3, 4, 5]
        pair, crew = spec(CHEAP_PAIR), spec("cheap-effort-crew")
        for own in range(2, MIN_EFFORT_LEVELS + 1):
            assert min_effort_expected_lowest(
                crew, own_level=own, counterpart_levels=levels
            ) < min_effort_expected_lowest(pair, own_level=own, counterpart_levels=levels)

    def test_a_group_all_at_one_level_makes_that_level_the_ceiling(self) -> None:
        game = spec(CHEAP_PAIR)
        for own in game.levels:
            assert min_effort_expected_lowest(
                game, own_level=own, counterpart_levels=[3, 3, 3]
            ) == pytest.approx(min(own, 3))

    def test_an_empty_counterpart_distribution_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one level"):
            min_effort_expected_lowest(spec(), own_level=3, counterpart_levels=[])

    def test_a_level_off_the_grid_is_a_caller_bug(self) -> None:
        game = spec()
        with pytest.raises(ValueError, match="must lie in"):
            min_effort_expected_lowest(game, own_level=0, counterpart_levels=[1, 2])
        with pytest.raises(ValueError, match="must lie in"):
            min_effort_expected_lowest(game, own_level=3, counterpart_levels=[6])


class TestThePressureGapsAgreeWithTheAnalyticForm:
    """The measured gap and the `(1 - F(e))^n > c/b` inequality are two statements of one quantity."""

    @pytest.mark.parametrize("variant", sorted(MIN_EFFORT_VARIANTS))
    @pytest.mark.parametrize(
        "levels", [[1, 2, 3, 4, 5], [5, 5, 5, 5], [1, 1, 5, 5], [2, 3, 3, 4], [1, 1, 1, 2]]
    )
    def test_the_sign_of_each_gap_is_the_sign_of_the_inequality(
        self, variant: str, levels: list[int]
    ) -> None:
        game = spec(variant)
        gaps = min_effort_pressure_gaps(game, levels)
        for level, gap in gaps.items():
            analytic = (
                min_effort_raise_probability(game, level=level, counterpart_levels=levels)
                - game.cost_benefit_ratio
            )
            assert (gap > 0) == (analytic > 0), (variant, level)
            assert gap == pytest.approx(analytic / (MIN_EFFORT_LEVELS - 1))

    def test_the_top_level_has_no_gap_because_there_is_no_step_to_price(self) -> None:
        gaps = min_effort_pressure_gaps(spec(), [1, 2, 3])
        assert set(gaps) == set(range(1, MIN_EFFORT_LEVELS))
        assert MIN_EFFORT_LEVELS not in gaps

    def test_the_best_response_is_where_the_gaps_change_sign(self) -> None:
        for variant in MIN_EFFORT_PAYOFF_VARIANTS:
            game = spec(variant)
            levels = min_effort_reference_levels(game)
            target = min_effort_best_response(game, levels)
            gaps = min_effort_pressure_gaps(game, levels)
            assert all(gap > 0 for level, gap in gaps.items() if level < target)
            assert all(gap <= 0 for level, gap in gaps.items() if level >= target)


class TestTheDegenerateCorpusIsRealAndMeasurable:
    """The one distribution that kills this game's gradient, and the instrument that finds it."""

    def test_an_extreme_split_has_exactly_no_spread_at_the_costly_pair(self) -> None:
        # even-hunt's dead cell, reached from a corpus rather than a payoff sheet: at c/b 0.5 with one
        # counterpart, half at the bottom and half at the top scores every level identically.
        game = spec("costly-effort-pair")
        assert min_effort_group_reward_span(game, [1, 1, 1, 1, 5, 5, 5, 5]) == 0.0

    def test_the_same_split_still_carries_spread_at_the_registered_cells(self) -> None:
        """The negative control: the dead cell is a property of that variant, not of the split."""
        split = [1, 1, 1, 1, 5, 5, 5, 5]
        assert min_effort_group_reward_span(spec(CHEAP_PAIR), split) > DEFAULT_MIN_SPLIT_STD
        assert min_effort_group_reward_span(spec(COSTLY_CREW), split) > DEFAULT_MIN_SPLIT_STD

    def test_the_reward_surface_is_flat_exactly_where_the_spread_is_zero(self) -> None:
        game = spec("costly-effort-pair")
        surface = level_reward_surface(game, [1, 1, 1, 1, 5, 5, 5, 5])
        assert len({round(value, 12) for value in surface.values()}) == 1

    def sweep_records(self, levels_by_prompt: dict[str, list[int]]) -> list[dict[str, Any]]:
        """Build sweep records in the shape `games.select_prompts` writes them."""
        row = make_row("costly-effort-pair")
        return [
            {
                "record_kind": "prompt-sweep",
                "prompt_id": prompt_id,
                "grading": GRADING_MIN_EFFORT_GROUP_MIX,
                "row": row,
                "samples": [
                    {"parsed": True, "level": level, "visible_text": level_answer(level)}
                    for level in levels
                ],
            }
            for prompt_id, levels in levels_by_prompt.items()
        ]

    def test_the_instrument_refuses_a_corpus_whose_every_group_is_pure(self) -> None:
        records = self.sweep_records({"a": [1, 1, 1, 1, 5, 5, 5, 5], "b": [1, 1, 5, 5]})
        report = min_effort_spread_from_records(records)
        assert report.n_prompts == 2
        assert report.pure_prompt_fraction == 1.0
        with pytest.raises(ValueError, match="train nothing"):
            assert_min_effort_reward_can_train(report)

    def test_the_instrument_accepts_a_mixed_corpus(self) -> None:
        """The negative control: a refusal that fired on everything would prove nothing."""
        records = self.sweep_records({"a": [1, 2, 3, 4], "b": [2, 3, 3, 4]})
        report = min_effort_spread_from_records(records)
        assert report.pure_prompt_fraction < 1.0
        assert report.median_spread > 0.0
        assert_min_effort_reward_can_train(report)

    def test_a_trace_with_no_minimum_effort_records_is_refused_rather_than_reported_as_clean(
        self,
    ) -> None:
        # A zero needs its denominator: a report over no prompts is not evidence of a healthy corpus.
        other = self.sweep_records({"a": [1, 2, 3]})
        other[0]["grading"] = "group-mix"
        report = min_effort_spread_from_records(other)
        assert report.n_prompts == 0
        assert report.n_records_skipped == 1
        with pytest.raises(ValueError, match="no minimum-effort sweep records"):
            assert_min_effort_reward_can_train(report)

    def test_the_histogram_is_reported_because_a_mean_cannot_say_it(self) -> None:
        bimodal = min_effort_spread_from_records(self.sweep_records({"a": [1, 1, 5, 5]}))
        middling = min_effort_spread_from_records(self.sweep_records({"a": [3, 3, 3, 3]}))
        assert bimodal.level_histogram == {1: 2, 5: 2}
        assert middling.level_histogram == {3: 4}


class TestTheRewardConsumesTheCorpus:
    """Generated rows through the real reward function, against hand-computed arithmetic."""

    def test_a_group_is_scored_against_its_own_realised_levels(self) -> None:
        rows = [make_row(CHEAP_PAIR) for _ in range(4)]
        levels = [1, 3, 3, 5]
        rewards, _, _ = score(rows, [level_answer(level) for level in levels])
        game = spec(CHEAP_PAIR)
        for reward, level in zip(rewards, levels, strict=True):
            assert reward == pytest.approx(
                min_effort_group_reward(game, own_level=level, counterpart_levels=levels)
            )

    def test_the_scored_group_excludes_the_unparseable_completion(self) -> None:
        # The canonical case from the matrix scorer: the mix is over the three that answered, not four.
        rows = [make_row(CHEAP_PAIR) for _ in range(4)]
        completions = [level_answer(2), level_answer(4), "no tag at all", level_answer(4)]
        rewards, metrics, _ = score(rows, completions)
        game = spec(CHEAP_PAIR)
        assert rewards[2] == DEFAULT_PARSE_PENALTY
        assert rewards[0] == pytest.approx(
            min_effort_group_reward(game, own_level=2, counterpart_levels=[2, 4, 4])
        )
        assert metrics["parse_failure_rate"] == pytest.approx(0.25)

    def test_a_whole_group_that_answers_nothing_takes_the_penalty(self) -> None:
        rows = [make_row() for _ in range(2)]
        with pytest.raises(RuntimeError, match="None of"):
            score(rows, ["nothing here", "nor here"])

    def test_a_level_off_the_grid_is_a_parse_failure_rather_than_a_clamp(self) -> None:
        rows = [make_row() for _ in range(3)]
        rewards, _, _ = score(rows, [level_answer(3), "<level>0</level>", "<level>9</level>"])
        assert rewards[1] == DEFAULT_PARSE_PENALTY
        assert rewards[2] == DEFAULT_PARSE_PENALTY

    def test_leave_one_out_grades_a_completion_against_the_others(self) -> None:
        rows = [make_row(CHEAP_PAIR) for _ in range(3)]
        levels = [1, 4, 5]
        rewards, _, _ = score(rows, [level_answer(level) for level in levels], leave_one_out=True)
        game = spec(CHEAP_PAIR)
        assert rewards[0] == pytest.approx(
            min_effort_group_reward(game, own_level=1, counterpart_levels=[4, 5])
        )

    def test_a_leave_one_out_group_of_one_takes_the_uniform_prior_and_says_so(self) -> None:
        rows = [make_row(CHEAP_PAIR) for _ in range(2)]
        rewards, metrics, _ = score(rows, [level_answer(4), "unparseable"], leave_one_out=True)
        game = spec(CHEAP_PAIR)
        assert rewards[0] == pytest.approx(
            min_effort_group_reward(
                game, own_level=4, counterpart_levels=min_effort_reference_levels(game)
            )
        )
        assert metrics["leave_one_out_prior_rate"] == pytest.approx(0.5)

    def test_the_team_size_column_changes_the_reward_on_the_same_answers(self) -> None:
        """The two knobs really do reach the reward, so a variant contrast is a reward contrast."""
        levels = [2, 3, 4, 5]
        completions = [level_answer(level) for level in levels]
        pair, _, _ = score([make_row("cheap-effort-pair") for _ in levels], completions)
        crew, _, _ = score([make_row("cheap-effort-crew") for _ in levels], completions)
        assert pair != crew

    def test_the_scorer_reads_team_size_off_the_row_rather_than_assuming_one(self) -> None:
        """Sabotage-shaped: hardcoding n=1 in the scorer would make these two agree.

        The design calls this the guard that matters most, because a prompt that says three
        counterparts and a reward that assumes one is a lie no loss curve could show. If the exponent
        came from anywhere but the row, the crew rows would score exactly as the pair rows do.
        """
        levels = [1, 3, 5]
        completions = [level_answer(level) for level in levels]
        rows = [make_row("cheap-effort-pair") for _ in levels]
        crew_rows = [dict(row, team_size=3) for row in rows]
        assert score(rows, completions)[0] != score(crew_rows, completions)[0]

    def test_a_row_whose_columns_could_not_be_a_game_is_refused_at_spec_construction(self) -> None:
        rows = [make_row(team_size=0) for _ in range(2)]
        with pytest.raises(ValueError, match="no counterpart can hold the minimum down"):
            score(rows, [level_answer(2), level_answer(4)])


class TestTheLoggedMetrics:
    """What a run records, and -- equally -- what it does not record on another game's rows."""

    def test_the_behavioural_and_prediction_metrics_are_all_logged(self) -> None:
        rows = [make_row(CHEAP_PAIR) for _ in range(4)]
        levels = [1, 3, 3, 5]
        _, metrics, extra = score(rows, [level_answer(level) for level in levels])
        game = spec(CHEAP_PAIR)
        assert metrics["mean_level"] == pytest.approx(sum(levels) / len(levels))
        assert metrics["mean_level_fraction"] == pytest.approx(
            sum((level - 1) / (MIN_EFFORT_LEVELS - 1) for level in levels) / len(levels)
        )
        assert metrics["mean_target_level"] == pytest.approx(min_effort_best_response(game, levels))
        assert "mean_upward_pressure" in metrics
        assert extra["min_effort_level"] == levels

    def test_the_upward_pressure_is_positive_where_the_payoffs_point_up(self) -> None:
        rows = [make_row(CHEAP_PAIR) for _ in range(4)]
        _, metrics, _ = score(rows, [level_answer(level) for level in (2, 3, 3, 4)])
        assert metrics["mean_upward_pressure"] > 0

    def test_the_upward_pressure_is_negative_where_they_point_down(self) -> None:
        """The reading that separates 'sitting where the payoffs want it' from 'no signal'."""
        rows = [make_row(COSTLY_CREW) for _ in range(4)]
        _, metrics, _ = score(rows, [level_answer(level) for level in (3, 4, 4, 5)])
        assert metrics["mean_upward_pressure"] < 0

    def test_no_level_metric_is_logged_for_another_game(self) -> None:
        # Asserting on absence as well as presence: a metric that fired on every grading would look
        # correct here while making the readout plot one game's levels under another's name.
        rows = [
            row
            for row in generate_prompt_rows("twin-pd", "group-mix", split=SPLIT_TRAIN)[:1]
            for _ in range(2)
        ]
        coop = str(rows[0]["coop_label"])
        other = next(label for label in (rows[0]["label_a"], rows[0]["label_b"]) if label != coop)
        _, metrics, _ = score(rows, [f"<action>{coop}</action>", f"<action>{other}</action>"])
        for key in (
            "mean_level",
            "mean_level_fraction",
            "mean_target_level",
            "mean_upward_pressure",
        ):
            assert key not in metrics

    def test_every_required_metric_of_each_grading_is_actually_produced(self) -> None:
        """The post-run read-back gate demands these; a scorer that never logs one fails after a run."""
        one_shot_rows = [make_row(CHEAP_PAIR) for _ in range(3)]
        _, one_shot, _ = score(one_shot_rows, [level_answer(level) for level in (1, 3, 5)])
        for key in required_metrics_for(GRADING_MIN_EFFORT_GROUP_MIX):
            if key in ("reward", "reward_std"):
                continue  # TRL's own, not this module's
            assert key in one_shot, key

        match_rows = self.match_rows(3)
        _, match_metrics, _ = score(
            match_rows,
            [
                match_answer([5, 5, 5, 5, 5]),
                match_answer([3, 3, 3, 3, 2]),
                match_answer([1, 2, 3, 4, 5]),
            ],
        )
        for key in required_metrics_for(GRADING_LEVEL_MATCH_RETURN):
            if key in ("reward", "reward_std"):
                continue
            assert key in match_metrics, key

    def test_every_metric_this_module_logs_is_watched_by_the_trainer(self) -> None:
        rows = [make_row(CHEAP_PAIR) for _ in range(3)]
        _, metrics, _ = score(rows, [level_answer(level) for level in (1, 3, 5)])
        watched = set(OPTIONAL_METRICS) | set(required_metrics_for(GRADING_MIN_EFFORT_GROUP_MIX))
        for key in (
            "mean_level",
            "mean_level_fraction",
            "mean_target_level",
            "mean_upward_pressure",
        ):
            assert key in watched, key
        assert set(metrics) - watched <= {"reward", "reward_std"} | set(metrics)

    @staticmethod
    def match_rows(count: int) -> list[dict[str, Any]]:
        """Return `count` copies of one repeated-form row, which is one training group."""
        rows = generate_prompt_rows(
            MIN_EFFORT_MATCH_GAME_ID, GRADING_LEVEL_MATCH_RETURN, split=SPLIT_TRAIN
        )
        return [dict(rows[0]) for _ in range(count)]


class TestTheMatchOptimumIsTheTopOfTheGrid:
    """Why the repeated arm is a diagnostic: dropping the final level only ever loses money here."""

    @pytest.mark.parametrize("cost_per_level", sorted(COMPUTED_MATCH_TOTALS))
    def test_the_designs_raw_totals_are_reproduced(self, cost_per_level: float) -> None:
        expected_total, expected_cost_of_dropping = COMPUTED_MATCH_TOTALS[cost_per_level]
        game = MinEffortSpec(
            game_id=MIN_EFFORT_MATCH_GAME_ID,
            n_levels=MIN_EFFORT_LEVELS,
            benefit_per_level=MIN_EFFORT_BENEFIT_PER_LEVEL,
            cost_per_level=cost_per_level,
            team_size=1,
        )
        top = [MIN_EFFORT_LEVELS] * MIN_EFFORT_MATCH_ROUNDS
        dropped = [*[MIN_EFFORT_LEVELS] * (MIN_EFFORT_MATCH_ROUNDS - 1), 1]
        assert simulate_min_effort_match(game, top) == pytest.approx(expected_total)
        assert simulate_min_effort_match(game, top) - simulate_min_effort_match(
            game, dropped
        ) == pytest.approx(expected_cost_of_dropping)

    @pytest.mark.parametrize("cost_per_level", sorted(COMPUTED_MATCH_TOTALS))
    def test_the_brute_forced_optimum_is_the_all_top_sequence(self, cost_per_level: float) -> None:
        """Searched over all 3,125 sequences, not asserted -- the whole arm rests on this."""
        game = MinEffortSpec(
            game_id=MIN_EFFORT_MATCH_GAME_ID,
            n_levels=MIN_EFFORT_LEVELS,
            benefit_per_level=MIN_EFFORT_BENEFIT_PER_LEVEL,
            cost_per_level=cost_per_level,
            team_size=1,
        )
        top = [MIN_EFFORT_LEVELS] * MIN_EFFORT_MATCH_ROUNDS
        assert min_effort_match_reward_total(game, top) == pytest.approx(
            max_min_effort_match_return(game, MIN_EFFORT_MATCH_ROUNDS)
        )
        assert min_effort_match_reward(game, top) == pytest.approx(1.0)

    def test_dropping_the_final_level_always_costs_reward(self) -> None:
        """The diagnostic in one assertion: from any sequence, a lower last level pays less."""
        game = spec(MIN_EFFORT_MATCH_PAYOFF_VARIANT, MIN_EFFORT_MATCH_GAME_ID)
        for previous in range(2, MIN_EFFORT_LEVELS + 1):
            held = [previous] * MIN_EFFORT_MATCH_ROUNDS
            for final in range(1, previous):
                dropped = [*held[:-1], final]
                assert min_effort_match_reward(game, dropped) < min_effort_match_reward(game, held)

    def test_matching_the_previous_level_is_the_best_final_round(self) -> None:
        game = spec(MIN_EFFORT_MATCH_PAYOFF_VARIANT, MIN_EFFORT_MATCH_GAME_ID)
        prefix = [4, 4, 4, 4]
        best = max(game.levels, key=lambda final: min_effort_match_reward(game, [*prefix, final]))
        assert best == prefix[-1]

    def test_the_counterpart_opens_at_the_bottom_then_copies_the_previous_level(self) -> None:
        game = spec(MIN_EFFORT_MATCH_PAYOFF_VARIANT, MIN_EFFORT_MATCH_GAME_ID)
        mine = [4, 2, 5, 1, 3]
        assert min_effort_matcher_levels(game, mine) == [
            MIN_EFFORT_MATCHER_OPENING_LEVEL,
            *mine[:-1],
        ]

    def test_the_mapped_total_keeps_every_sequence_inside_the_reward_range(self) -> None:
        """Why per-round mapping rather than mapping the total: an alternating sequence goes negative.

        At the costly ratio, alternating between the extremes has a genuinely negative RAW total, and a
        negative numerator over a positive optimum would be a reward below the parse penalty -- an
        unparseable completion scoring better than a played match.
        """
        game = MinEffortSpec(
            game_id=MIN_EFFORT_MATCH_GAME_ID,
            n_levels=MIN_EFFORT_LEVELS,
            benefit_per_level=1.0,
            cost_per_level=0.5,
            team_size=1,
        )
        alternating = [5, 1, 5, 1, 5]
        assert simulate_min_effort_match(game, alternating) < 0
        assert 0.0 <= min_effort_match_reward(game, alternating) <= 1.0
        assert min_effort_match_reward(game, alternating) > DEFAULT_PARSE_PENALTY

    def test_the_brute_force_cap_refuses_a_search_that_would_hang(self) -> None:
        game = spec(MIN_EFFORT_MATCH_PAYOFF_VARIANT, MIN_EFFORT_MATCH_GAME_ID)
        with pytest.raises(ValueError, match="n_rounds must be in"):
            max_min_effort_match_return(game, MAX_MIN_EFFORT_BRUTE_FORCE_ROUNDS + 1)

    def test_the_repeated_forms_share_one_length(self) -> None:
        """Restated across two modules, so the two repeated arms stay comparable round by round."""
        assert MIN_EFFORT_MATCH_ROUNDS == ITERATED_N_ROUNDS


class TestTheMatchGradingThroughTheRewardFunction:
    """The repeated form end to end: parse, simulate, normalise, and report the end-game drop."""

    def rows(self, count: int) -> list[dict[str, Any]]:
        rows = generate_prompt_rows(
            MIN_EFFORT_MATCH_GAME_ID, GRADING_LEVEL_MATCH_RETURN, split=SPLIT_TRAIN
        )
        return [dict(rows[0]) for _ in range(count)]

    def test_a_match_scores_as_the_simulated_return_over_the_optimum(self) -> None:
        sequences = [[5, 5, 5, 5, 5], [3, 3, 3, 3, 3], [1, 1, 1, 1, 1]]
        rewards, _, _ = score(self.rows(3), [match_answer(seq) for seq in sequences])
        game = spec(MIN_EFFORT_MATCH_PAYOFF_VARIANT, MIN_EFFORT_MATCH_GAME_ID)
        for reward, sequence in zip(rewards, sequences, strict=True):
            assert reward == pytest.approx(min_effort_match_reward(game, sequence))
        assert rewards[0] == pytest.approx(1.0)

    def test_the_wrong_number_of_tags_is_all_or_nothing(self) -> None:
        rows = self.rows(3)
        rewards, _, _ = score(
            rows,
            [
                match_answer([5, 5, 5, 5, 5]),
                match_answer([5, 5, 5]),
                match_answer([5, 5, 5, 5, 5, 5]),
            ],
        )
        assert rewards[1] == DEFAULT_PARSE_PENALTY
        assert rewards[2] == DEFAULT_PARSE_PENALTY

    def test_one_bad_figure_fails_the_whole_sequence(self) -> None:
        rows = self.rows(2)
        rewards, _, _ = score(
            rows,
            [
                match_answer([5, 5, 5, 5, 5]),
                "<level>5</level><level>5</level><level>0</level><level>5</level><level>5</level>",
            ],
        )
        assert rewards[1] == DEFAULT_PARSE_PENALTY

    def test_the_end_game_drop_rate_counts_a_lower_final_level(self) -> None:
        rows = self.rows(4)
        _, metrics, _ = score(
            rows,
            [
                match_answer([5, 5, 5, 5, 5]),
                match_answer([5, 5, 5, 5, 1]),
                match_answer([3, 3, 3, 3, 2]),
                match_answer([2, 2, 2, 2, 4]),
            ],
        )
        assert metrics["end_game_drop_rate"] == pytest.approx(0.5)

    def test_the_behavioural_mean_is_the_mean_level_across_rounds(self) -> None:
        # Two completions rather than one: a group of one has no group-relative advantage, and
        # `make_game_reward` refuses that outright.
        rows = self.rows(2)
        _, metrics, _ = score(rows, [match_answer([1, 2, 3, 4, 5]), match_answer([3, 3, 3, 3, 3])])
        per_completion = [
            sum((level - 1) / (MIN_EFFORT_LEVELS - 1) for level in (1, 2, 3, 4, 5)) / 5,
            sum((level - 1) / (MIN_EFFORT_LEVELS - 1) for level in (3, 3, 3, 3, 3)) / 5,
        ]
        assert metrics["mean_level_fraction"] == pytest.approx(sum(per_completion) / 2)

    def test_the_copying_pd_reports_the_same_reading_so_the_pair_is_comparable(self) -> None:
        """The contrast the diagnostic rests on: the same key on the game where a drop is optimal."""
        rows = generate_prompt_rows("iterated-pd-tft", "iterated-return", split=SPLIT_TRAIN)[:1]
        coop = str(rows[0]["coop_label"])
        other = next(label for label in (rows[0]["label_a"], rows[0]["label_b"]) if label != coop)
        group = [dict(rows[0]) for _ in range(2)]
        _, metrics, _ = score(
            group,
            [
                f"<action>{coop}</action>" * 5,
                f"<action>{coop}</action>" * 4 + f"<action>{other}</action>",
            ],
        )
        assert metrics["end_game_drop_rate"] == pytest.approx(0.5)


class TestTheParser:
    """Never raises on model output; out of range is a failure and never a clamp."""

    def test_the_last_tag_wins_after_the_thinking_block(self) -> None:
        assert parse_level("<level>2</level> on reflection <level>4</level>", n_levels=5) == 4

    def test_a_level_below_the_grid_fails_rather_than_clamping_up(self) -> None:
        # Zero is the natural thing a model writes for "I do nothing", and clamping it to 1 would
        # record the bottom of the grid as a deliberate answer.
        assert parse_level("<level>0</level>", n_levels=5) is None
        assert parse_level("<level>-2</level>", n_levels=5) is None

    def test_a_level_above_the_grid_fails_rather_than_clamping_down(self) -> None:
        assert parse_level("<level>6</level>", n_levels=5) is None

    def test_prose_and_non_integers_are_parse_failures(self) -> None:
        for written in ("<level>three</level>", "<level>3.5</level>", "<level></level>", "level 3"):
            assert parse_level(written, n_levels=5) is None

    def test_a_sequence_needs_exactly_the_round_count(self) -> None:
        assert parse_level_sequence("<level>1</level>" * 5, n_rounds=5, n_levels=5) == [1] * 5
        assert parse_level_sequence("<level>1</level>" * 4, n_rounds=5, n_levels=5) is None
        assert parse_level_sequence("<level>1</level>" * 6, n_rounds=5, n_levels=5) is None

    def test_a_sequence_keeps_its_round_order(self) -> None:
        written = "".join(f"<level>{level}</level>" for level in (1, 5, 2, 4, 3))
        assert parse_level_sequence(written, n_rounds=5, n_levels=5) == [1, 5, 2, 4, 3]

    def test_a_caller_bug_still_raises(self) -> None:
        with pytest.raises(ValueError, match="n_levels must be positive"):
            parse_level("<level>1</level>", n_levels=0)


class TestTheRenderedPrompt:
    """The frames, the printed grid, and the two vocabulary guards on this renderer."""

    def test_the_prompt_states_every_cell_of_the_grid(self) -> None:
        game = spec(CHEAP_PAIR)
        prompt = render_min_effort_prompt(game, MIN_EFFORT_SCENARIOS[0])
        for own in game.levels:
            for lowest in game.levels:
                expected = format_points(
                    min_effort_cell_reward(game, own_level=own, lowest_other_level=lowest)
                )
                assert (
                    f"- You write {own} and the lowest anyone else writes is {lowest}: you are "
                    f"credited {expected} points."
                ) in prompt

    def test_the_printed_figures_are_the_graded_ones(self) -> None:
        """Read back out of the text and compared with the reward, so the two cannot drift."""
        game = spec(COSTLY_CREW)
        prompt = render_min_effort_prompt(game, MIN_EFFORT_SCENARIOS[0])
        for own in game.levels:
            for lowest in game.levels:
                match = re.search(
                    rf"- You write {own} and the lowest anyone else writes is {lowest}: you are "
                    rf"credited (\S+) points\.",
                    prompt,
                )
                assert match is not None, (own, lowest)
                assert float(match.group(1)) == pytest.approx(
                    min_effort_cell_reward(game, own_level=own, lowest_other_level=lowest) * 100
                )

    def test_the_prompt_states_the_answer_format_and_the_grid_size(self) -> None:
        prompt = render_min_effort_prompt(spec(), MIN_EFFORT_SCENARIOS[0])
        assert "<level>N</level>" in prompt
        assert f"between 1 and {MIN_EFFORT_LEVELS}" in prompt
        assert "exactly one tag" in prompt

    @pytest.mark.parametrize(
        "pair",
        [("cheap-effort-pair", "costly-effort-pair"), ("cheap-effort-crew", "costly-effort-crew")],
    )
    def test_the_cost_knob_moves_only_the_figures(self, pair: tuple[str, str]) -> None:
        """The counterbalance golden's shape, applied to the cost knob at a fixed team size.

        Digits masked out, the two cost ratios render byte-identical prose. That is the property the
        registered contrast rests on for the cost half of the grid, and it is asserted at each team
        size separately because the TEAM knob genuinely does move a word -- see the test below.
        """
        numbers = re.compile(r"\d[\d.]*")
        masked = {
            numbers.sub("#", render_min_effort_prompt(spec(variant), MIN_EFFORT_SCENARIOS[0]))
            for variant in pair
        }
        assert len(masked) == 1

    def test_the_team_knob_moves_the_counterpart_count_and_its_plural(self) -> None:
        """Stated rather than hidden: the registered pair's prose is NOT byte-identical.

        Unlike twin-pd's contrast, where the two arms share one prompt exactly, the minimum-effort
        pair moves two printed things -- the grid's figures and the number of counterparts, including
        its singular/plural. Both are the knobs, so this is the design rather than a leak, but a
        digit-masking golden across all four cells would fail on the plural and reading that as a bug
        would be the wrong conclusion.
        """
        numbers = re.compile(r"\d[\d.]*")
        pair = numbers.sub("#", render_min_effort_prompt(spec(CHEAP_PAIR), MIN_EFFORT_SCENARIOS[0]))
        crew = numbers.sub(
            "#", render_min_effort_prompt(spec("cheap-effort-crew"), MIN_EFFORT_SCENARIOS[0])
        )
        assert pair != crew
        assert "other instance of" in pair
        assert "other instances of" in crew

    def test_two_variants_really_are_different_prompts(self) -> None:
        """The negative control for the goldens above, which would pass on identical renders."""
        first = render_min_effort_prompt(spec(CHEAP_PAIR), MIN_EFFORT_SCENARIOS[0])
        second = render_min_effort_prompt(spec(COSTLY_CREW), MIN_EFFORT_SCENARIOS[0])
        assert first != second

    def test_the_repeated_form_states_the_rounds_the_rule_and_the_tag_count(self) -> None:
        prompt = render_min_effort_match_prompt(
            spec(MIN_EFFORT_MATCH_PAYOFF_VARIANT, MIN_EFFORT_MATCH_GAME_ID),
            MIN_EFFORT_SCENARIOS[0],
            n_rounds=MIN_EFFORT_MATCH_ROUNDS,
        )
        assert f"This runs {MIN_EFFORT_MATCH_ROUNDS} times" in prompt
        assert f"in the first round it writes {MIN_EFFORT_MATCHER_OPENING_LEVEL}" in prompt
        assert "whatever number you wrote in the round before" in prompt
        assert f"exactly {MIN_EFFORT_MATCH_ROUNDS} tags" in prompt

    def test_the_repeated_form_refuses_a_team_it_cannot_describe(self) -> None:
        with pytest.raises(ValueError, match="ONE announced counterpart"):
            render_min_effort_match_prompt(
                spec("cheap-effort-crew", MIN_EFFORT_MATCH_GAME_ID),
                MIN_EFFORT_SCENARIOS[0],
                n_rounds=MIN_EFFORT_MATCH_ROUNDS,
            )

    def test_the_two_forms_share_their_frame_mechanics_and_grid(self) -> None:
        """What makes the repeated arm a companion rather than a second game."""
        game = spec(MIN_EFFORT_MATCH_PAYOFF_VARIANT, MIN_EFFORT_MATCH_GAME_ID)
        scenario = MIN_EFFORT_SCENARIOS[0]
        one_shot = render_min_effort_prompt(spec(CHEAP_PAIR), scenario)
        repeated = render_min_effort_match_prompt(game, scenario, n_rounds=MIN_EFFORT_MATCH_ROUNDS)
        for own in game.levels:
            line = f"- You write {own} and the lowest anyone else writes is 1:"
            assert line in one_shot
            assert line in repeated
        assert scenario.frame in one_shot
        assert scenario.frame in repeated

    def test_a_banned_word_planted_in_a_frame_is_caught(self) -> None:
        """The vocabulary guard on THIS renderer, or its call would be unverified."""
        planted = dataclasses.replace(
            MIN_EFFORT_SCENARIOS[0],
            frame=MIN_EFFORT_SCENARIOS[0].frame + "\n\nThe crews cooperate on the pace.",
        )
        with pytest.raises(ValueError, match="loaded vocabulary"):
            render_min_effort_prompt(spec(), planted)

    def test_the_games_own_literature_is_banned_and_the_guard_fires_on_it(self) -> None:
        for banned in (
            "This is the weakest link in the chain.",
            "A weakest-link problem for the crew.",
            "This is a minimum effort game.",
            "The minimum-effort setting applies.",
        ):
            with pytest.raises(ValueError, match="loaded vocabulary"):
                assert_no_loaded_vocabulary(banned)

    def test_the_mechanical_vocabulary_the_frames_need_still_passes(self) -> None:
        """The other direction: a ban that swallowed the mechanics would make the game unwritable."""
        for clean in (
            "Each of you writes down one whole number for your own pace.",
            "The lowest number written by anyone sets what the crew achieves.",
            "Working at a higher level is charged to you.",
            "The link between the two sections is bolted.",
        ):
            assert_no_loaded_vocabulary(clean)

    def test_no_prompt_leaks_its_own_identifiers(self) -> None:
        for game_id in MIN_EFFORT_GAME_IDS:
            for split in SPLITS:
                for row in generate_prompt_rows(
                    game_id,
                    GRADING_LEVEL_MATCH_RETURN
                    if game_id == MIN_EFFORT_MATCH_GAME_ID
                    else GRADING_MIN_EFFORT_GROUP_MIX,
                    split=split,
                ):
                    prompt = str(row["prompt"])
                    assert game_id not in prompt
                    assert str(row["reskin_id"]) not in prompt
                    assert str(row["payoff_variant"]) not in prompt


class TestTheProseAndTheGradingReadOneColumn:
    """The guard the design calls the one that matters most, read back out of the rendered text."""

    COUNTERPART_COUNT = re.compile(r"working with (\d+) other instances?")

    def counterparts_in_prose(self, prompt: str) -> int:
        match = self.COUNTERPART_COUNT.search(prompt)
        assert match is not None, prompt
        return int(match.group(1))

    def test_the_prose_counterpart_count_is_the_column_the_reward_reads(self) -> None:
        for split in SPLITS:
            rows = generate_prompt_rows(
                MIN_EFFORT_GAME_ID, GRADING_MIN_EFFORT_GROUP_MIX, split=split
            )
            assert rows
            for row in rows:
                assert self.counterparts_in_prose(str(row["prompt"])) == int(str(row["team_size"]))

    def test_a_row_whose_column_disagrees_with_its_prose_is_the_failure_this_catches(self) -> None:
        """Sabotage as a permanent case: move the column and the read-back check goes red."""
        row = make_row("cheap-effort-pair")
        assert self.counterparts_in_prose(str(row["prompt"])) == 1
        tampered = dict(row, team_size=3)
        assert self.counterparts_in_prose(str(tampered["prompt"])) != int(
            str(tampered["team_size"])
        )

    def test_the_singular_and_plural_clauses_are_both_grammatical(self) -> None:
        assert "with 1 other instance of this same model" in render_min_effort_prompt(
            spec("cheap-effort-pair"), MIN_EFFORT_SCENARIOS[0]
        )
        assert "with 3 other instances of this same model" in render_min_effort_prompt(
            spec("cheap-effort-crew"), MIN_EFFORT_SCENARIOS[0]
        )

    def test_the_repeated_form_names_one_counterpart_and_grades_against_one(self) -> None:
        rows = generate_prompt_rows(
            MIN_EFFORT_MATCH_GAME_ID, GRADING_LEVEL_MATCH_RETURN, split=SPLIT_TRAIN
        )
        for row in rows:
            assert int(str(row["team_size"])) == 1
            assert "a different automated system" in str(row["prompt"])


class TestTheAuthoredRoster:
    """The frames themselves: enough of them, split correctly, and each one renderable."""

    def test_the_roster_clears_the_designs_floor_of_sixteen_train_frames(self) -> None:
        # At ten frames a pinned arm starts at ten rows and cannot clear the twelve-row mixedness
        # floor even with zero attrition, which is why sixteen is a floor rather than a target.
        train = [scenario for scenario in MIN_EFFORT_SCENARIOS if not scenario.eval_only]
        held_out = [scenario for scenario in MIN_EFFORT_SCENARIOS if scenario.eval_only]
        assert len(train) == 16
        assert len(held_out) == 4

    def test_the_two_splits_share_no_frame(self) -> None:
        for game_id in MIN_EFFORT_GAME_IDS:
            grading = (
                GRADING_LEVEL_MATCH_RETURN
                if game_id == MIN_EFFORT_MATCH_GAME_ID
                else GRADING_MIN_EFFORT_GROUP_MIX
            )
            train = {
                row["reskin_id"]
                for row in generate_prompt_rows(game_id, grading, split=SPLIT_TRAIN)
            }
            held_out = {
                row["reskin_id"] for row in generate_prompt_rows(game_id, grading, split=SPLIT_EVAL)
            }
            assert train
            assert held_out
            assert not train & held_out

    def test_every_scenario_id_is_unique(self) -> None:
        ids = [scenario.scenario_id for scenario in MIN_EFFORT_SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_no_frame_uses_the_prosocial_vocabulary_the_global_guard_cannot_ban(self) -> None:
        """Self-policed by roster test, because the global regex cannot be tightened for these.

        `generous`, `fair share` and `selfish` all appear in tracked interp stimuli and probe text, so
        adding them to `BANNED_VOCABULARY` would turn unrelated tests red. A frame that hinted which
        level was the considerate one would break the same property the label counterbalance protects
        for the matrix games, so the check lives here instead.
        """
        forbidden = ("generous", "fair share", "selfish", "free rider", "greedy", "unselfish")
        for scenario in MIN_EFFORT_SCENARIOS:
            lowered = scenario.frame.casefold()
            for phrase in forbidden:
                assert phrase not in lowered, (scenario.scenario_id, phrase)

    def test_a_frame_that_never_names_its_own_quantity_is_refused(self) -> None:
        with pytest.raises(ValueError, match="never mentions its effort_name"):
            MinEffortScenario(
                scenario_id="mismatched",
                frame="TEST NOTE\n\nThe crew each set a number.",
                effort_name="pace",
                group_noun="crew",
            )

    def test_a_frame_that_never_names_its_group_is_refused(self) -> None:
        with pytest.raises(ValueError, match="never mentions its group_noun"):
            MinEffortScenario(
                scenario_id="mismatched",
                frame="TEST NOTE\n\nEach of you sets a pace.",
                effort_name="pace",
                group_noun="crew",
            )

    def test_an_empty_frame_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty frame"):
            MinEffortScenario(
                scenario_id="blank", frame="   ", effort_name="pace", group_noun="crew"
            )

    def test_the_mechanics_paragraph_reads_in_each_frames_own_words(self) -> None:
        for scenario in MIN_EFFORT_SCENARIOS:
            prompt = render_min_effort_prompt(spec(), scenario)
            assert f"one whole number for your own {scenario.effort_name}" in prompt
            assert f"What the {scenario.group_noun} gets out of it" in prompt


class TestTheCorpusRows:
    """The row schema, the identity of a row, and what this game carries blank."""

    def test_every_row_carries_exactly_the_declared_schema(self) -> None:
        for game_id in MIN_EFFORT_GAME_IDS:
            grading = (
                GRADING_LEVEL_MATCH_RETURN
                if game_id == MIN_EFFORT_MATCH_GAME_ID
                else GRADING_MIN_EFFORT_GROUP_MIX
            )
            for split in SPLITS:
                for row in generate_prompt_rows(game_id, grading, split=split):
                    assert tuple(row.keys()) == ROW_COLUMNS

    def test_the_level_columns_carry_the_variant_and_the_matrix_cells_stay_blank(self) -> None:
        for row in generate_prompt_rows(
            MIN_EFFORT_GAME_ID, GRADING_MIN_EFFORT_GROUP_MIX, split=SPLIT_TRAIN
        ):
            cost, team = min_effort_variant(str(row["payoff_variant"]))
            assert row["n_levels"] == MIN_EFFORT_LEVELS
            assert row["benefit_per_level"] == MIN_EFFORT_BENEFIT_PER_LEVEL
            assert row["cost_per_level"] == cost
            assert row["team_size"] == team
            for column in ("payoff_cc", "payoff_cd", "payoff_dc", "payoff_dd"):
                assert row[column] == 0.0
            assert row["label_a"] == ""
            assert row["coop_label"] == ""

    def test_every_other_game_carries_the_level_columns_blank(self) -> None:
        """The schema is global: a column added for one game is added to every row builder."""
        for game_id in ("twin-pd", "dictator", "nash-demand", "trust-strategy-method"):
            grading = {
                "twin-pd": "group-mix",
                "dictator": "keep-fraction",
                "nash-demand": "nash-demand-group-mix",
                "trust-strategy-method": "trustor-payoff-self-rule",
            }[game_id]
            for row in generate_prompt_rows(game_id, grading, split=SPLIT_TRAIN):
                assert row["n_levels"] == 0
                assert row["benefit_per_level"] == 0.0
                assert row["cost_per_level"] == 0.0
                assert row["team_size"] == 0

    def test_all_four_variants_reach_the_one_shot_corpus(self) -> None:
        variants = {
            row["payoff_variant"]
            for row in generate_prompt_rows(
                MIN_EFFORT_GAME_ID, GRADING_MIN_EFFORT_GROUP_MIX, split=SPLIT_TRAIN
            )
        }
        assert variants == set(MIN_EFFORT_PAYOFF_VARIANTS)

    def test_the_repeated_corpus_carries_only_the_single_counterpart_variant(self) -> None:
        variants = {
            row["payoff_variant"]
            for row in generate_prompt_rows(
                MIN_EFFORT_MATCH_GAME_ID, GRADING_LEVEL_MATCH_RETURN, split=SPLIT_TRAIN
            )
        }
        assert variants == {MIN_EFFORT_MATCH_PAYOFF_VARIANT}
        assert min_effort_variant(MIN_EFFORT_MATCH_PAYOFF_VARIANT)[1] == 1

    def test_the_repeated_rows_carry_their_round_count(self) -> None:
        for row in generate_prompt_rows(
            MIN_EFFORT_MATCH_GAME_ID, GRADING_LEVEL_MATCH_RETURN, split=SPLIT_TRAIN
        ):
            assert row["n_rounds"] == MIN_EFFORT_MATCH_ROUNDS

    def test_the_one_shot_rows_carry_no_rounds(self) -> None:
        for row in generate_prompt_rows(
            MIN_EFFORT_GAME_ID, GRADING_MIN_EFFORT_GROUP_MIX, split=SPLIT_TRAIN
        ):
            assert row["n_rounds"] == 0

    def test_the_rows_are_singletons_with_no_counterbalanced_partner(self) -> None:
        # No labels, so nothing to counterbalance: one row per frame per variant, and
        # `select_prompts.is_counterbalanced` reads the empty labels and treats each as a singleton.
        for game_id in MIN_EFFORT_GAME_IDS:
            assert game_id in UNLABELLED_GAME_IDS

    def test_a_swapped_print_order_is_refused_rather_than_ignored(self) -> None:
        for game_id in MIN_EFFORT_GAME_IDS:
            grading = (
                GRADING_LEVEL_MATCH_RETURN
                if game_id == MIN_EFFORT_MATCH_GAME_ID
                else GRADING_MIN_EFFORT_GROUP_MIX
            )
            with pytest.raises(ValueError, match="no action labels"):
                generate_prompt_rows(
                    game_id, grading, split=SPLIT_TRAIN, label_print_order="swapped"
                )

    def test_prompt_ids_are_unique_within_each_game_and_split(self) -> None:
        for game_id in MIN_EFFORT_GAME_IDS:
            grading = (
                GRADING_LEVEL_MATCH_RETURN
                if game_id == MIN_EFFORT_MATCH_GAME_ID
                else GRADING_MIN_EFFORT_GROUP_MIX
            )
            for split in SPLITS:
                ids = [
                    row["prompt_id"] for row in generate_prompt_rows(game_id, grading, split=split)
                ]
                assert len(ids) == len(set(ids))

    def test_generation_is_deterministic(self) -> None:
        for game_id in MIN_EFFORT_GAME_IDS:
            grading = (
                GRADING_LEVEL_MATCH_RETURN
                if game_id == MIN_EFFORT_MATCH_GAME_ID
                else GRADING_MIN_EFFORT_GROUP_MIX
            )
            assert generate_prompt_rows(game_id, grading, split=SPLIT_TRAIN) == (
                generate_prompt_rows(game_id, grading, split=SPLIT_TRAIN)
            )


class TestTheRegistry:
    """What the arm registry says about these games, and what it refuses."""

    def test_the_registered_arms_validate(self) -> None:
        validate_arms(ARMS)

    def test_both_new_gradings_are_classified_for_the_dead_arm_check(self) -> None:
        # The partition has to hold, or a grading absent from both tables gets no dead-arm check at
        # all -- the fail-open shape integration sabotage found on 2026-08-21.
        for grading in MIN_EFFORT_GRADINGS:
            assert (grading in SELF_CONSISTENT_GRADINGS) != (grading in SPAN_CHECK_EXEMPTIONS)

    def test_the_exemptions_say_what_supplies_the_spread_instead(self) -> None:
        for grading in MIN_EFFORT_GRADINGS:
            assert len(SPAN_CHECK_EXEMPTIONS[grading].split()) >= 10

    def test_the_two_arms_differ_only_in_their_pinned_cell(self) -> None:
        cheap, costly = ARMS["min-effort-cheap-pair"], ARMS["min-effort-costly-crew"]
        assert cheap.game_id == costly.game_id
        assert cheap.grading == costly.grading
        assert cheap.payoff_variants != costly.payoff_variants

    def test_each_arm_pins_exactly_one_cell(self) -> None:
        for name in ("min-effort-cheap-pair", "min-effort-costly-crew"):
            assert len(ARMS[name].payoff_variants) == 1

    def test_the_repeated_arms_name_their_own_games(self) -> None:
        assert ARMS["iterated-min-effort-matcher"].game_id == MIN_EFFORT_MATCH_GAME_ID
        assert ARMS["iterated-stag-tft"].game_id == ITERATED_STAG_GAME_ID
        assert ARMS["iterated-pd-grim"].game_id == ITERATED_PD_GRIM_GAME_ID

    def test_the_grim_arm_does_not_mutate_the_copying_arm(self) -> None:
        """Its own game id, because iterated-pd-tft has trained artifacts keyed to its rule."""
        assert ARMS["iterated-pd-grim"].game_id != ARMS["iterated-pd-tft"].game_id
        rules = {
            game_id: {
                row["opponent_rule"]
                for row in generate_prompt_rows(game_id, "iterated-return", split=SPLIT_TRAIN)
            }
            for game_id in ("iterated-pd-tft", ITERATED_PD_GRIM_GAME_ID)
        }
        assert rules["iterated-pd-tft"] == {"tit-for-tat"}
        assert rules[ITERATED_PD_GRIM_GAME_ID] == {"grim-trigger"}

    def test_a_pin_the_corpus_does_not_carry_is_refused(self) -> None:
        with pytest.raises(ValueError, match="its own corpus does not carry"):
            validate_arms(
                {
                    "bad": GameArm(
                        game_id=MIN_EFFORT_GAME_ID,
                        grading=GRADING_MIN_EFFORT_GROUP_MIX,
                        notes="a cell nothing renders",
                        payoff_variants=("expensive-effort-throng",),
                    )
                }
            )

    def test_a_format_only_arm_on_a_level_game_is_refused(self) -> None:
        # The answer is a level tag, so no obedient completion carries an <action> tag at all and
        # every reward would be the parse penalty -- caught here rather than after a card is reserved.
        with pytest.raises(ValueError, match="do not ask for exactly one <action> tag"):
            validate_arms(
                {
                    "bad": GameArm(
                        game_id=MIN_EFFORT_GAME_ID,
                        grading="format-only",
                        notes="the level-tag answer shape",
                    )
                }
            )

    def test_the_selection_sweep_can_parse_both_gradings(self) -> None:
        # The step most likely to be missed, and it fails mid-sweep after the GPU is paid for.
        for grading in MIN_EFFORT_GRADINGS:
            assert grading in PARSEABLE_GRADINGS

    def test_neither_grading_joins_the_cooperation_rate_family(self) -> None:
        """A rate says nothing about a level, so these key on the spread branch instead."""
        for grading in MIN_EFFORT_GRADINGS:
            assert grading not in ONE_SHOT_ACTION_GRADINGS
