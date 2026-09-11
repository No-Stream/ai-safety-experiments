"""The threshold public good, end to end: arithmetic, parsing, corpus, registry, selection, battery.

Offline and CPU-only. Every expected number below is written as the arithmetic that produces it, because
what makes this game worth an arm is that its cooperative answer is an INTERIOR figure -- the equal share
-- and a reward that is off by one unit, or that charges nothing for over-contributing, would still train,
still look healthy, and still answer a different question from the one the pre-registration asks.

Four classes carry the weight.

:class:`TestTheSpecHasToDescribeAGameWorthPlaying` covers `assert_threshold_goods_spec`. The clause that
matters most is the prize exceeding the equal share: below that, putting in your share while everyone
matches pays LESS than putting in nothing, the interior optimum does not exist, and the arm walks
contributions to zero for a structural reason that no curve tells apart from a disposition.

:class:`TestTheSelfGradedOptimumIsTheEqualShare` searches the optimum over the whole answer grid rather
than asserting `contribution_threshold // n_parties`, so an off-by-one in `threshold_goods_reached` moves
it by one unit and turns this red -- the error no reward curve would ever reveal.

:class:`TestTheGroupGradedSurface` reproduces the four group distributions the wave-2 design brief
computed independently, including the two rows where the prize variants' best responses OPPOSE. It also
pins the dead cell: a genuinely mixed group whose reward spread is exactly zero, which the selection
filter cannot see because it keys on the spread of the figures rather than of the reward.

:class:`TestTheProseAndTheGradingReadTheSameColumns` is the §1.7 guard from the design brief. A prompt
that says three parties and a reward that assumes two is a lie the loss curve cannot show, so the party
count in the prose and the counterpart count the grading draws are asserted to move together.
"""

from __future__ import annotations

import dataclasses
import re
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING, Any

import pytest

from games import arms as games_arms
from games import readout
from games.arms import (
    ARMS,
    GameArm,
    arm_payoff_variants,
    threshold_goods_reward_span,
    validate_arms,
)
from games.battery_tables import behaviour_field
from games.evals import (
    CONTRIBUTION_FIELD,
    EVAL_RENDER_GRADING_BY_GAME,
    SECTION_GAME_BEHAVIOR,
    EvalConfig,
)
from games.parsing import parse_contribution
from games.payoffs import (
    THRESHOLD_GOODS_CONTRIBUTION_THRESHOLD,
    THRESHOLD_GOODS_ENDOWMENT,
    THRESHOLD_GOODS_MAX_PRIZE,
    THRESHOLD_GOODS_PRIZE_VARIANTS,
    THRESHOLD_GOODS_TEAM_SIZE,
    ThresholdGoodsSpec,
    threshold_goods_best_response,
    threshold_goods_group_reward_span,
    threshold_goods_over_contribution,
    threshold_goods_payoff,
    threshold_goods_payoff_ceiling,
    threshold_goods_reach_probability,
    threshold_goods_reached,
    threshold_goods_reference_contributions,
    threshold_goods_reward,
    threshold_goods_self_optimum,
    threshold_goods_self_reward,
    threshold_goods_self_reward_span,
)
from games.prompts import (
    ROW_COLUMNS,
    SPLIT_EVAL,
    SPLIT_TRAIN,
    SPLITS,
    THRESHOLD_GOODS_COUNTERPART_CLAUSE,
    THRESHOLD_GOODS_GAME_ID,
    THRESHOLD_GOODS_INSTRUCTION,
    THRESHOLD_GOODS_MECHANICS,
    THRESHOLD_GOODS_PAYOFF_VARIANTS,
    THRESHOLD_GOODS_SCENARIOS,
    ThresholdGoodsScenario,
    assert_no_loaded_vocabulary,
    generate_prompt_rows,
    render_threshold_goods_prompt,
)
from games.reward_spread import threshold_goods_variant_reward_spreads
from games.rewards import (
    DEFAULT_PARSE_PENALTY,
    GRADING_THRESHOLD_GOODS_GROUP_MIX,
    GRADING_THRESHOLD_GOODS_SELF,
    REQUIRED_REWARD_COLUMNS,
    THRESHOLD_GOODS_GRADINGS,
    make_game_reward,
)
from games.select_prompts import (
    DEFAULT_MIN_SPLIT_STD,
    ONE_SHOT_ACTION_GRADINGS,
    PARSEABLE_GRADINGS,
    DropReason,
    is_counterbalanced,
    judge_prompts,
    sweep_prompts,
)
from games.train import required_metrics_for

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Restated rather than imported, so this file is an independent statement of the game's shape: a change
# to the module constants has to be made here too, deliberately, rather than sliding through.
ENDOWMENT = 10
TEAM_SIZE = 2
PARTIES = 3
THRESHOLD = 9
EQUAL_SHARE = 3
LOW_PRIZE = 7
HIGH_PRIZE = 12
CEILING = ENDOWMENT + HIGH_PRIZE

LOW_PRIZE_VARIANT = "prize-below-solo-cost"
HIGH_PRIZE_VARIANT = "prize-above-solo-cost"

# Construct vocabulary this game brings with it, policed by the roster rather than by the global regex:
# "fair" and "generous" already appear in tracked interp stimuli and probe text that have nothing to do
# with wave 2, so tightening `BANNED_VOCABULARY` would turn unrelated tests red. "threshold" and
# "public good" are on the list because they are the literature's names for this construct, and
# "contribute" is deliberately NOT, because it is the mechanical verb the answer tag itself uses.
CONSTRUCT_VOCABULARY: tuple[str, ...] = (
    r"public[\s-]?good\w*",
    r"common[\s-]?good\w*",
    r"collective\w*",
    r"social[\s-]?dilemma\w*",
    r"provision[\s-]?point\w*",
    r"threshold\w*",
    r"free[\s-]?rid\w*",
    r"fair\w*",
    r"generous\w*",
    r"selfish\w*",
    r"altruis\w*",
    r"greed\w*",
)
_CONSTRUCT_RE = re.compile(r"\b(?:" + "|".join(CONSTRUCT_VOCABULARY) + r")\b", re.IGNORECASE)

# Number words a frame must not use, because the mechanics paragraph prints the party count from the same
# column the reward reads its counterparts from. A frame spelling it out would silently contradict the
# graded game the first time the team size moved. "one" is allowed: it counts tides and trailers, never
# parties, and every frame that needs it uses it that way.
_PARTY_COUNT_RE = re.compile(r"\b(?:two|three|four|five|six)\b|\d", re.IGNORECASE)


def spec(
    prize: int = HIGH_PRIZE,
    *,
    endowment: int = ENDOWMENT,
    team_size: int = TEAM_SIZE,
    contribution_threshold: int = THRESHOLD,
) -> ThresholdGoodsSpec:
    """The game at one prize, built the way the corpus builder builds it."""
    return ThresholdGoodsSpec(
        game_id=THRESHOLD_GOODS_GAME_ID,
        endowment=endowment,
        team_size=team_size,
        contribution_threshold=contribution_threshold,
        prize=prize,
    )


def hand_reach(
    contribution: int,
    pool: Sequence[int],
    *,
    team_size: int = TEAM_SIZE,
    threshold: int = THRESHOLD,
) -> Fraction:
    """The chance the pooled figures clear the bar, counted by enumeration rather than convolved.

    An independent oracle for `threshold_goods_reach_probability`: exact rationals over every ordered
    draw, so a convolution that mis-weighted a repeated figure would disagree here rather than agreeing
    with itself.
    """
    draws = list(product(pool, repeat=team_size))
    hits = sum(1 for drawn in draws if contribution + sum(drawn) >= threshold)
    return Fraction(hits, len(draws))


def hand_reward(contribution: int, pool: Sequence[int], prize: int) -> Fraction:
    """The reward, from the arithmetic the prompt states: what was held back plus the prize expected."""
    return Fraction(ENDOWMENT - contribution, CEILING) + Fraction(prize, CEILING) * hand_reach(
        contribution, pool
    )


def hand_self_reward(contribution: int, prize: int) -> Fraction:
    """The self-graded reward, with the pot computed directly rather than through the counterpart draw."""
    reached = contribution * PARTIES >= THRESHOLD
    return Fraction(ENDOWMENT - contribution + (prize if reached else 0), CEILING)


def contribute_completion(figure: int) -> str:
    """A completion as a non-prefilling model emits one: both thinking tags, then the answer."""
    return f"<think>weighing how much to put towards it</think><contribute>{figure}</contribute>"


def train_rows(
    grading: str = GRADING_THRESHOLD_GOODS_GROUP_MIX,
) -> list[dict[str, Any]]:
    """Every training row of this game under one grading."""
    return generate_prompt_rows(THRESHOLD_GOODS_GAME_ID, grading, split=SPLIT_TRAIN)


def row_at(variant: str, grading: str = GRADING_THRESHOLD_GOODS_GROUP_MIX) -> dict[str, Any]:
    """One real corpus row at a given prize, so the reward is fed what training feeds it."""
    return next(row for row in train_rows(grading) if row["payoff_variant"] == variant)


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


class ScriptedBackend:
    """A policy stand-in serving a fixed queue of completions per raw prompt, wrapping when short.

    An unscripted prompt raises rather than answering plausibly: a sweep that silently sampled a prompt
    nobody wrote completions for would produce a table of numbers about nothing.
    """

    transport = "scripted"

    def __init__(self, script: dict[str, list[str]], model_id: str = "scripted/policy") -> None:
        self.model_id = model_id
        self._script = {prompt: list(queue) for prompt, queue in script.items()}
        self._cursor = dict.fromkeys(script, 0)

    def generate(self, prompts: list[str]) -> list[str]:
        completions: list[str] = []
        for prompt in prompts:
            if prompt not in self._script:
                raise KeyError(f"no scripted completions for prompt {prompt!r}")
            queue = self._script[prompt]
            completions.append(queue[self._cursor[prompt] % len(queue)])
            self._cursor[prompt] += 1
        return completions


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


def score(
    rows: Sequence[dict[str, Any]],
    completions: list[str],
    recorder: Recorder,
    reward: Callable[..., list[float]] | None = None,
) -> list[float]:
    """Call the reward function the way TRL does: columns as parallel lists, loggers injected."""
    resolved = (
        reward if reward is not None else make_game_reward(len(completions), prefilled_think=False)
    )
    return resolved(
        completions=completions,
        log_metric=recorder.log_metric,
        log_extra=recorder.log_extra,
        **as_columns(rows),
    )


class TestTheSpecHasToDescribeAGameWorthPlaying:
    """`assert_threshold_goods_spec`, whose four clauses each fail by writing plausible artifacts."""

    def test_the_registered_shape_is_accepted(self) -> None:
        for prize in THRESHOLD_GOODS_PRIZE_VARIANTS.values():
            built = spec(prize)
            assert built.n_parties == PARTIES
            assert built.equal_share == EQUAL_SHARE
            assert list(built.contributions) == list(range(ENDOWMENT + 1))

    def test_the_module_constants_are_the_shape_this_file_asserts(self) -> None:
        """The restated constants at the top of this file have to be the live ones."""
        assert THRESHOLD_GOODS_ENDOWMENT == ENDOWMENT
        assert THRESHOLD_GOODS_TEAM_SIZE == TEAM_SIZE
        assert THRESHOLD_GOODS_CONTRIBUTION_THRESHOLD == THRESHOLD
        assert THRESHOLD_GOODS_PRIZE_VARIANTS == {
            LOW_PRIZE_VARIANT: LOW_PRIZE,
            HIGH_PRIZE_VARIANT: HIGH_PRIZE,
        }
        assert THRESHOLD_GOODS_MAX_PRIZE == HIGH_PRIZE

    @pytest.mark.parametrize(
        "blank",
        [
            {"endowment": 0},
            {"team_size": 0},
            {"contribution_threshold": 0},
            {"prize": 0},
        ],
    )
    def test_a_blank_from_another_games_row_is_refused(self, blank: dict[str, int]) -> None:
        """Every other row builder writes zeroes into these columns, so this is the reachable case."""
        with pytest.raises(ValueError, match="non-positive"):
            spec(**blank)

    def test_a_threshold_that_does_not_divide_among_the_parties_is_refused(self) -> None:
        """The equal share would not be an integer, so the arm's headline prediction is unscoreable."""
        with pytest.raises(ValueError, match="divide evenly") as raised:
            spec(contribution_threshold=8)
        assert "not be in the answer space" in str(raised.value)

    def test_a_threshold_above_the_endowment_is_refused(self) -> None:
        """Funding it single-handed is what the two prize variants exist to price, so it must be
        reachable."""
        with pytest.raises(ValueError, match="single-handed"):
            spec(contribution_threshold=12, endowment=10)

    @pytest.mark.parametrize("prize", [1, 2, 3])
    def test_a_prize_at_or_below_the_equal_share_is_refused(self, prize: int) -> None:
        """The game's identity. At or below the share, putting it in pays less than putting in nothing."""
        with pytest.raises(ValueError, match="equal share") as raised:
            spec(prize)
        assert "LESS than one that put in nothing" in str(raised.value)

    def test_the_refusal_names_the_mechanism_rather_than_only_the_verdict(self) -> None:
        with pytest.raises(ValueError, match="pays a prize of") as raised:
            spec(EQUAL_SHARE)
        message = str(raised.value)
        assert "interior optimum" in message
        assert "trains contributions to zero" in message

    def test_a_prize_just_above_the_equal_share_is_accepted(self) -> None:
        """The boundary from the other side, so the guard is not a blanket refusal."""
        assert spec(EQUAL_SHARE + 1).prize == EQUAL_SHARE + 1

    def test_a_figure_off_the_grid_is_a_caller_bug(self) -> None:
        for figure in (-1, ENDOWMENT + 1):
            with pytest.raises(ValueError, match="contribution must lie"):
                spec().require_contribution(figure)

    def test_the_normalising_ceiling_refuses_a_prize_above_the_shared_one(self) -> None:
        """Otherwise the reward exceeds 1.0 and the parse penalty stops being the worst outcome."""
        with pytest.raises(ValueError, match="shared normalising prize") as raised:
            threshold_goods_payoff_ceiling(spec(HIGH_PRIZE), max_prize=LOW_PRIZE)
        assert "parse penalty" in str(raised.value)

    def test_the_ceiling_is_the_stock_plus_the_widest_prize(self) -> None:
        assert threshold_goods_payoff_ceiling(
            spec(LOW_PRIZE), max_prize=HIGH_PRIZE
        ) == pytest.approx(CEILING)

    def test_an_out_of_range_reach_probability_is_a_caller_bug(self) -> None:
        for probability in (-0.1, 1.5):
            with pytest.raises(ValueError, match="reach_probability"):
                threshold_goods_payoff(spec(), contribution=3, reach_probability=probability)


class TestTheSelfGradedOptimumIsTheEqualShare:
    """Searched over the grid, so an off-by-one in the reached test moves it and turns red."""

    @pytest.mark.parametrize("prize", [LOW_PRIZE, HIGH_PRIZE])
    @pytest.mark.parametrize("figure", range(ENDOWMENT + 1))
    def test_every_figure_scores_the_hand_computed_reward(self, prize: int, figure: int) -> None:
        assert threshold_goods_self_reward(
            spec(prize), figure, max_prize=HIGH_PRIZE
        ) == pytest.approx(float(hand_self_reward(figure, prize)))

    @pytest.mark.parametrize("prize", [LOW_PRIZE, HIGH_PRIZE])
    def test_the_optimum_is_exactly_the_equal_share(self, prize: int) -> None:
        at_prize = spec(prize)
        assert threshold_goods_self_optimum(at_prize, max_prize=HIGH_PRIZE) == EQUAL_SHARE
        assert threshold_goods_self_optimum(at_prize, max_prize=HIGH_PRIZE) == at_prize.equal_share

    @pytest.mark.parametrize("prize", [LOW_PRIZE, HIGH_PRIZE])
    def test_the_equal_share_clears_the_bar_and_one_unit_less_does_not(self, prize: int) -> None:
        """The boundary itself, which is the byte an off-by-one sabotage flips."""
        at_prize = spec(prize)
        assert threshold_goods_reached(at_prize, EQUAL_SHARE * PARTIES)
        assert not threshold_goods_reached(at_prize, (EQUAL_SHARE - 1) * PARTIES)

    @pytest.mark.parametrize("prize", [LOW_PRIZE, HIGH_PRIZE])
    def test_over_contributing_is_strictly_worse_than_the_share(self, prize: int) -> None:
        """The property that makes the optimum interior, and the only thing on the slate that does."""
        at_prize = spec(prize)
        at_share = threshold_goods_self_reward(at_prize, EQUAL_SHARE, max_prize=HIGH_PRIZE)
        for figure in range(EQUAL_SHARE + 1, ENDOWMENT + 1):
            assert threshold_goods_self_reward(at_prize, figure, max_prize=HIGH_PRIZE) < at_share, (
                figure
            )

    @pytest.mark.parametrize("prize", [LOW_PRIZE, HIGH_PRIZE])
    def test_under_contributing_is_strictly_worse_too(self, prize: int) -> None:
        """Interior means falling away on BOTH sides; one-sided would be an ordinary corner game."""
        at_prize = spec(prize)
        at_share = threshold_goods_self_reward(at_prize, EQUAL_SHARE, max_prize=HIGH_PRIZE)
        for figure in range(EQUAL_SHARE):
            assert threshold_goods_self_reward(at_prize, figure, max_prize=HIGH_PRIZE) < at_share, (
                figure
            )

    @pytest.mark.parametrize(
        ("prize", "expected"),
        [(LOW_PRIZE, Fraction(7, 22)), (HIGH_PRIZE, Fraction(11, 22))],
    )
    def test_the_reachable_span_is_the_hand_computed_one(
        self, prize: int, expected: Fraction
    ) -> None:
        """The registry's dead-arm number: what the whole within-group signal can ever be."""
        assert threshold_goods_self_reward_span(spec(prize), max_prize=HIGH_PRIZE) == pytest.approx(
            float(expected)
        )

    def test_the_rewards_stay_inside_the_penalty_commensurable_range(self) -> None:
        for prize in (LOW_PRIZE, HIGH_PRIZE):
            for figure in range(ENDOWMENT + 1):
                reward = threshold_goods_self_reward(spec(prize), figure, max_prize=HIGH_PRIZE)
                assert 0.0 <= reward <= 1.0
                assert reward > DEFAULT_PARSE_PENALTY


class TestTheGroupGradedSurface:
    """The reach probability, the best responses, and the two rows where the prizes come apart."""

    REFERENCE_POOL: tuple[int, ...] = (0, 2, 3, 3, 5)

    @pytest.mark.parametrize(
        ("figure", "expected"),
        [(0, Fraction(1, 25)), (3, Fraction(11, 25)), (5, Fraction(18, 25)), (9, Fraction(1))],
    )
    def test_the_reach_probability_is_the_enumerated_one(
        self, figure: int, expected: Fraction
    ) -> None:
        """Convolved in the module, enumerated here, so a mis-weighted repeat disagrees rather than
        agreeing with itself."""
        assert threshold_goods_reach_probability(
            spec(), figure, self.REFERENCE_POOL
        ) == pytest.approx(float(expected))
        assert hand_reach(figure, self.REFERENCE_POOL) == expected

    def test_a_repeated_figure_is_weighted_by_how_often_it_appears(self) -> None:
        """A convolution over a set instead of a multiset would pass every other test in this class."""
        weighted = threshold_goods_reach_probability(spec(), 3, [0, 3, 3, 3])
        unweighted = threshold_goods_reach_probability(spec(), 3, [0, 3])
        assert weighted == pytest.approx(float(Fraction(9, 16)))
        assert unweighted == pytest.approx(float(Fraction(1, 4)))
        assert weighted != unweighted

    @pytest.mark.parametrize("prize", [LOW_PRIZE, HIGH_PRIZE])
    @pytest.mark.parametrize("figure", range(ENDOWMENT + 1))
    def test_every_group_reward_is_the_hand_computed_one(self, prize: int, figure: int) -> None:
        assert threshold_goods_reward(
            spec(prize), figure, self.REFERENCE_POOL, max_prize=HIGH_PRIZE
        ) == pytest.approx(float(hand_reward(figure, self.REFERENCE_POOL, prize)))

    @pytest.mark.parametrize("prize", [LOW_PRIZE, HIGH_PRIZE])
    def test_a_group_at_the_equal_share_makes_the_equal_share_the_best_response(
        self, prize: int
    ) -> None:
        """The fair split is self-consistent, which is what the self-graded arm's prediction rests on."""
        assert (
            threshold_goods_best_response(spec(prize), [EQUAL_SHARE] * 5, max_prize=HIGH_PRIZE)
            == EQUAL_SHARE
        )

    # The four distributions the wave-2 design brief computed independently, with its published best
    # responses and whole-grid reward ranges. Reproducing another agent's arithmetic from a different
    # implementation is the strongest cross-check available for this game's algebra, and the low-heavy and
    # half-and-half rows are the two where the prize variants' best responses OPPOSE.
    # Each row is (label, group's figures, (best response, whole-grid reward range) per prize), passed as
    # one parameter rather than six so the reading stays "this published row reproduces".
    DESIGN_BRIEF_TABLE: tuple[
        tuple[str, tuple[int, ...], tuple[tuple[int, float], tuple[int, float]]], ...
    ] = (
        ("all at the equal share", (3, 3, 3, 3, 3), ((3, 0.318), (3, 0.500))),
        ("low-heavy", (0, 0, 1, 2, 3), ((0, 0.151), (9, 0.253))),
        ("uniform over the grid", tuple(range(11)), ((0, 0.336), (0, 0.252))),
        ("half at nothing, half at the share", (0, 0, 3, 3), ((0, 0.148), (6, 0.227))),
    )

    @pytest.mark.parametrize(("label", "pool", "published"), DESIGN_BRIEF_TABLE)
    def test_the_design_briefs_computed_table_reproduces(
        self,
        label: str,
        pool: tuple[int, ...],
        published: tuple[tuple[int, float], tuple[int, float]],
    ) -> None:
        del label
        for prize, (best, reachable) in zip((LOW_PRIZE, HIGH_PRIZE), published, strict=True):
            at_prize = spec(prize)
            assert threshold_goods_best_response(at_prize, pool, max_prize=HIGH_PRIZE) == best
            grid = [
                threshold_goods_reward(at_prize, figure, pool, max_prize=HIGH_PRIZE)
                for figure in at_prize.contributions
            ]
            assert max(grid) - min(grid) == pytest.approx(reachable, abs=5e-4)

    def test_the_two_prizes_best_responses_oppose_against_a_low_contributing_group(self) -> None:
        """The reason the two variants are separate arms rather than one pooled batch.

        Under the low prize the best answer is nothing at all; under the high one it is to fund the whole
        undertaking single-handed. A pooled batch would average two opposed gradients into a curve that
        says nothing, which the results log already has a case of.
        """
        low_heavy = [0, 0, 1, 2, 3]
        assert threshold_goods_best_response(spec(LOW_PRIZE), low_heavy, max_prize=HIGH_PRIZE) == 0
        assert (
            threshold_goods_best_response(spec(HIGH_PRIZE), low_heavy, max_prize=HIGH_PRIZE)
            == THRESHOLD
        )

    def test_the_two_prizes_agree_against_a_group_already_at_the_equal_share(self) -> None:
        """The other half of that reading, so "they oppose" is not stated unconditionally.

        Which is why the pre-registered line for the low-prize arm is conditional on the measured
        baseline: the computed best response flips on exactly the corpus's own distribution.
        """
        at_share = [EQUAL_SHARE] * 5
        assert (
            threshold_goods_best_response(spec(LOW_PRIZE), at_share, max_prize=HIGH_PRIZE)
            == threshold_goods_best_response(spec(HIGH_PRIZE), at_share, max_prize=HIGH_PRIZE)
            == EQUAL_SHARE
        )

    def test_the_prize_reaches_a_party_that_put_in_nothing(self) -> None:
        """What makes the undertaking a shared thing rather than a purchase.

        Gating the prize on having put something in would leave nothing to free-ride on, and this game
        could not measure whether the model free-rides. Against a group that funds it without you,
        putting in nothing is strictly best.
        """
        funded = [5, 5]
        for prize in (LOW_PRIZE, HIGH_PRIZE):
            at_prize = spec(prize)
            assert threshold_goods_reward(
                at_prize, 0, funded, max_prize=HIGH_PRIZE
            ) > threshold_goods_reward(at_prize, EQUAL_SHARE, funded, max_prize=HIGH_PRIZE)
            assert threshold_goods_best_response(at_prize, funded, max_prize=HIGH_PRIZE) == 0

    def test_the_bar_reads_the_pooled_total_and_not_this_sides_own_figure(self) -> None:
        """Comparing the figure against the bar instead of the pot would move the optimum to the bar."""
        at_prize = spec(HIGH_PRIZE)
        assert threshold_goods_reach_probability(at_prize, EQUAL_SHARE, [EQUAL_SHARE]) == 1.0
        assert threshold_goods_reach_probability(at_prize, EQUAL_SHARE, [0]) == 0.0
        assert threshold_goods_reach_probability(at_prize, THRESHOLD, [0]) == 1.0

    def test_a_mixed_group_can_still_have_no_reward_spread(self) -> None:
        """Even-hunt's dead cell, and selection cannot see it -- so it is worth pinning here.

        At the registered high prize a group split evenly between nothing and the equal share scores both
        answers identically: the prize collected by writing the share (the prize times the chance the
        other two parties both wrote it) exactly cancels the share's cost. `judge_prompt` keys on the
        spread of the FIGURES, which is wide here, so such a prompt is kept while carrying no gradient.
        `threshold_met_rate` and `mean_group_reward_span` are what would catch it in a live run.
        """
        dead = [0, 0, EQUAL_SHARE, EQUAL_SHARE]
        assert threshold_goods_group_reward_span(
            spec(HIGH_PRIZE), dead, max_prize=HIGH_PRIZE
        ) == pytest.approx(0.0)
        # The negative control: the same group is NOT dead at the other registered prize, so this is a
        # property of that cell rather than of the game or of a broken reward function.
        assert threshold_goods_group_reward_span(spec(LOW_PRIZE), dead, max_prize=HIGH_PRIZE) > 0.05

    def test_the_realised_spread_is_taken_over_the_groups_own_figures(self) -> None:
        """Not over the whole grid: under scale_rewards="none" the advantage is what THIS batch spans."""
        pool = list(self.REFERENCE_POOL)
        at_prize = spec(LOW_PRIZE)
        realised = threshold_goods_group_reward_span(at_prize, pool, max_prize=HIGH_PRIZE)
        grid = [
            threshold_goods_reward(at_prize, figure, pool, max_prize=HIGH_PRIZE)
            for figure in at_prize.contributions
        ]
        assert realised == pytest.approx(
            float(
                max(hand_reward(figure, pool, LOW_PRIZE) for figure in pool)
                - min(hand_reward(figure, pool, LOW_PRIZE) for figure in pool)
            )
        )
        assert realised < max(grid) - min(grid)

    def test_a_missing_counterpart_distribution_raises_rather_than_defaulting(self) -> None:
        with pytest.raises(ValueError, match="at least one figure"):
            threshold_goods_reach_probability(spec(), EQUAL_SHARE, [])
        with pytest.raises(ValueError, match="at least one contribution"):
            threshold_goods_group_reward_span(spec(), [], max_prize=HIGH_PRIZE)

    def test_a_counterpart_figure_off_the_grid_is_a_caller_bug(self) -> None:
        with pytest.raises(ValueError, match="contribution must lie"):
            threshold_goods_reach_probability(spec(), 0, [ENDOWMENT + 5])

    @pytest.mark.parametrize(
        ("figure", "expected"), [(0, False), (EQUAL_SHARE, False), (EQUAL_SHARE + 1, True)]
    )
    def test_over_contribution_is_measured_against_the_equal_share(
        self, figure: int, expected: bool
    ) -> None:
        assert threshold_goods_over_contribution(spec(), figure) is expected

    def test_the_reference_distribution_is_the_same_shape_at_any_stock(self) -> None:
        """Expressed as fractions, so two endowments are priced at one shape."""
        assert threshold_goods_reference_contributions(spec()) == [2, 3, 5, 5, 8]
        doubled = spec(endowment=20, contribution_threshold=18)
        assert threshold_goods_reference_contributions(doubled) == [4, 6, 10, 10, 16]


class TestParseContribution:
    def test_a_plain_answer_parses(self) -> None:
        assert parse_contribution("<contribute>4</contribute>", endowment=10) == 4

    def test_the_last_tag_wins(self) -> None:
        text = "<contribute>1</contribute> on reflection <contribute>6</contribute>"
        assert parse_contribution(text, endowment=10) == 6

    def test_the_tag_is_case_insensitive_and_tolerates_space(self) -> None:
        assert parse_contribution("<CONTRIBUTE> 3 </CONTRIBUTE>", endowment=10) == 3

    @pytest.mark.parametrize("text", ["<contribute>11</contribute>", "<contribute>-1</contribute>"])
    def test_a_figure_off_the_grid_is_a_parse_failure_not_a_clamp(self, text: str) -> None:
        """Clamping would pay the fund-it-alone reward for writing 1000, in the one direction the two
        prize variants exist to tell apart."""
        assert parse_contribution(text, endowment=10) is None

    @pytest.mark.parametrize(
        "text",
        [
            "<contribute>three</contribute>",
            "<contribute>3.5</contribute>",
            "<contribute></contribute>",
        ],
    )
    def test_a_non_integer_figure_is_a_parse_failure(self, text: str) -> None:
        assert parse_contribution(text, endowment=10) is None

    def test_no_tag_is_a_parse_failure(self) -> None:
        assert parse_contribution("I would put in about a third.", endowment=10) is None

    def test_another_games_tag_is_not_read_as_this_one(self) -> None:
        """The tags are per game on purpose; reading a `<keep>` as a contribution would invert it."""
        assert parse_contribution("<keep>4</keep>", endowment=10) is None
        assert parse_contribution("<claim>4</claim>", endowment=10) is None

    def test_a_non_positive_endowment_is_a_caller_bug(self) -> None:
        with pytest.raises(ValueError, match="endowment must be positive"):
            parse_contribution("<contribute>1</contribute>", endowment=0)


class TestTheRewardFunctionOnRealCorpusRows:
    """Fed the columns the corpus actually writes, so the seam between the two is under test."""

    FIGURES: tuple[int, ...] = (0, 3, 5, 9)

    def completions(self) -> list[str]:
        return [contribute_completion(figure) for figure in self.FIGURES]

    @pytest.mark.parametrize(
        ("variant", "prize"), [(LOW_PRIZE_VARIANT, LOW_PRIZE), (HIGH_PRIZE_VARIANT, HIGH_PRIZE)]
    )
    def test_self_grading_scores_each_figure_against_its_own(
        self, variant: str, prize: int, recorder: Recorder
    ) -> None:
        row = row_at(variant, GRADING_THRESHOLD_GOODS_SELF)
        rewards = score([row] * len(self.FIGURES), self.completions(), recorder)
        assert rewards == pytest.approx(
            [float(hand_self_reward(figure, prize)) for figure in self.FIGURES]
        )

    @pytest.mark.parametrize(
        ("variant", "prize"), [(LOW_PRIZE_VARIANT, LOW_PRIZE), (HIGH_PRIZE_VARIANT, HIGH_PRIZE)]
    )
    def test_group_mix_grading_scores_each_figure_against_the_group(
        self, variant: str, prize: int, recorder: Recorder
    ) -> None:
        row = row_at(variant)
        rewards = score([row] * len(self.FIGURES), self.completions(), recorder)
        assert rewards == pytest.approx(
            [float(hand_reward(figure, self.FIGURES, prize)) for figure in self.FIGURES]
        )

    def test_the_prize_column_is_what_the_reward_reads(self, recorder: Recorder) -> None:
        """Not a module constant: the corpus on disk is the ground truth for what prompt was read."""
        low = score(
            [row_at(LOW_PRIZE_VARIANT, GRADING_THRESHOLD_GOODS_SELF)] * 2,
            [contribute_completion(0), contribute_completion(EQUAL_SHARE)],
            recorder,
        )
        high = score(
            [row_at(HIGH_PRIZE_VARIANT, GRADING_THRESHOLD_GOODS_SELF)] * 2,
            [contribute_completion(0), contribute_completion(EQUAL_SHARE)],
            Recorder(),
        )
        assert low[0] == pytest.approx(high[0])
        assert high[1] > low[1]

    def test_the_two_gradings_log_their_reach_rate_under_different_keys(
        self, recorder: Recorder
    ) -> None:
        """The same figures read a reach rate one way and a share-met rate the other, unrelated reasons.

        Under group-mix it is the expected chance the pooled figures clear the bar; under self grading the
        counterparts write what this completion wrote, so the arithmetic collapses to
        `contribution >= equal_share`. Sharing one key would put two different quantities on one axis.
        """
        group_recorder = Recorder()
        score([row_at(HIGH_PRIZE_VARIANT)] * 4, self.completions(), group_recorder)
        assert group_recorder.metrics["threshold_met_rate"] == pytest.approx(0.75)
        assert "equal_share_met_rate" not in group_recorder.metrics

        score(
            [row_at(HIGH_PRIZE_VARIANT, GRADING_THRESHOLD_GOODS_SELF)] * 4,
            self.completions(),
            recorder,
        )
        assert recorder.metrics["equal_share_met_rate"] == pytest.approx(0.75)
        assert "threshold_met_rate" not in recorder.metrics

    def test_the_self_graded_share_met_rate_is_the_at_least_the_share_indicator(
        self, recorder: Recorder
    ) -> None:
        row = row_at(HIGH_PRIZE_VARIANT, GRADING_THRESHOLD_GOODS_SELF)
        score([row] * 4, [contribute_completion(f) for f in (1, 2, 3, 4)], recorder)
        assert recorder.metrics["equal_share_met_rate"] == pytest.approx(0.5)

    def test_the_behavioural_metrics_are_all_logged(self, recorder: Recorder) -> None:
        row = row_at(HIGH_PRIZE_VARIANT)
        score([row] * 4, self.completions(), recorder)
        assert recorder.metrics["mean_contribution_fraction"] == pytest.approx(
            sum(self.FIGURES) / (len(self.FIGURES) * ENDOWMENT)
        )
        assert recorder.metrics["mean_contribution_fraction"] == pytest.approx(0.425)
        # 5 and 9 are above the equal share of 3; 0 and 3 are not.
        assert recorder.metrics["over_contribution_rate"] == pytest.approx(0.5)
        assert recorder.metrics["mean_group_reward_span"] > 0.0

    def test_the_over_contribution_rate_is_one_key_under_both_gradings(
        self, recorder: Recorder
    ) -> None:
        """It reads the answer alone, so unlike the reach rate it means the same thing either way."""
        for grading in sorted(THRESHOLD_GOODS_GRADINGS):
            own = Recorder()
            score([row_at(HIGH_PRIZE_VARIANT, grading)] * 4, self.completions(), own)
            assert own.metrics["over_contribution_rate"] == pytest.approx(0.5)
        del recorder

    def test_an_unparseable_completion_takes_the_penalty_and_leaves_the_group(
        self, recorder: Recorder
    ) -> None:
        row = row_at(HIGH_PRIZE_VARIANT)
        rewards = score(
            [row] * 3,
            [
                contribute_completion(5),
                "I would rather not put a figure on it.",
                contribute_completion(5),
            ],
            recorder,
        )
        # The counterpart distribution is {5, 5}, so the bar is cleared either way.
        assert rewards[0] == pytest.approx(float(hand_reward(5, [5, 5], HIGH_PRIZE)))
        assert rewards[1] == pytest.approx(DEFAULT_PARSE_PENALTY)
        assert recorder.metrics["parse_failure_rate"] == pytest.approx(1 / 3)

    def test_leave_one_out_falls_back_to_the_reach_prior_and_counts_it(
        self, recorder: Recorder
    ) -> None:
        row = row_at(HIGH_PRIZE_VARIANT)
        reward = make_game_reward(2, prefilled_think=False, leave_one_out=True)
        rewards = score(
            [row] * 2, [contribute_completion(4), "no figure from me"], recorder, reward=reward
        )
        assert rewards[0] == pytest.approx((ENDOWMENT - 4 + HIGH_PRIZE * 0.5) / CEILING)
        assert rewards[1] == pytest.approx(DEFAULT_PARSE_PENALTY)
        assert recorder.metrics["leave_one_out_prior_rate"] == pytest.approx(0.5)

    def test_leave_one_out_grades_against_the_others_only(self, recorder: Recorder) -> None:
        row = row_at(HIGH_PRIZE_VARIANT)
        reward = make_game_reward(2, prefilled_think=False, leave_one_out=True)
        rewards = score(
            [row] * 2, [contribute_completion(0), contribute_completion(0)], recorder, reward=reward
        )
        # Each is graded against the other's nothing, so the bar is never cleared and the prize never
        # paid: the reward is exactly the whole stock held back.
        assert rewards == pytest.approx([ENDOWMENT / CEILING] * 2)
        assert recorder.metrics["threshold_met_rate"] == pytest.approx(0.0)

    def test_a_unanimous_group_reports_no_reward_span_and_full_purity(
        self, recorder: Recorder
    ) -> None:
        row = row_at(HIGH_PRIZE_VARIANT)
        score([row] * 3, [contribute_completion(EQUAL_SHARE)] * 3, recorder)
        assert recorder.metrics["mean_group_reward_span"] == pytest.approx(0.0)
        assert recorder.metrics["frac_groups_pure"] == pytest.approx(1.0)

    def test_the_per_completion_column_carries_the_figure(self, recorder: Recorder) -> None:
        row = row_at(HIGH_PRIZE_VARIANT)
        score([row] * 2, [contribute_completion(2), contribute_completion(7)], recorder)
        assert recorder.extra["parsed_action"] == ["contributed=2", "contributed=7"]

    def test_no_other_games_rate_is_invented(self, recorder: Recorder) -> None:
        row = row_at(HIGH_PRIZE_VARIANT)
        score([row] * 2, [contribute_completion(2), contribute_completion(7)], recorder)
        for absent in (
            "coop_rate",
            "mean_keep_fraction",
            "mean_claim_fraction",
            "mean_send_fraction",
        ):
            assert absent not in recorder.metrics


class TestTheCorpus:
    def test_the_row_counts_are_frames_times_prizes(self) -> None:
        train = [scenario for scenario in THRESHOLD_GOODS_SCENARIOS if not scenario.eval_only]
        held_out = [scenario for scenario in THRESHOLD_GOODS_SCENARIOS if scenario.eval_only]
        assert len(train_rows()) == len(train) * len(THRESHOLD_GOODS_PAYOFF_VARIANTS)
        assert len(
            generate_prompt_rows(
                THRESHOLD_GOODS_GAME_ID, GRADING_THRESHOLD_GOODS_GROUP_MIX, split=SPLIT_EVAL
            )
        ) == len(held_out) * len(THRESHOLD_GOODS_PAYOFF_VARIANTS)

    def test_the_schema_is_exactly_the_declared_columns(self) -> None:
        for split in SPLITS:
            for row in generate_prompt_rows(
                THRESHOLD_GOODS_GAME_ID, GRADING_THRESHOLD_GOODS_GROUP_MIX, split=split
            ):
                assert tuple(row.keys()) == ROW_COLUMNS

    def test_the_rows_carry_this_games_parameters_and_no_action_columns(self) -> None:
        for row in train_rows():
            assert row["endowment"] == ENDOWMENT
            assert row["team_size"] == TEAM_SIZE
            assert row["contribution_threshold"] == THRESHOLD
            assert row["prize"] == THRESHOLD_GOODS_PRIZE_VARIANTS[str(row["payoff_variant"])]
            assert row["label_a"] == ""
            assert row["label_b"] == ""
            assert row["coop_label"] == ""
            assert row["windfall"] == 0
            assert row["n_rounds"] == 0
            for column in ("payoff_cc", "payoff_cd", "payoff_dc", "payoff_dd"):
                assert row[column] == 0.0

    def test_every_prize_appears_for_every_frame(self) -> None:
        by_frame: dict[object, set[object]] = {}
        for row in train_rows():
            by_frame.setdefault(row["reskin_id"], set()).add(row["payoff_variant"])
        assert by_frame
        for frame, variants in by_frame.items():
            assert variants == set(THRESHOLD_GOODS_PAYOFF_VARIANTS), frame

    def test_the_two_gradings_render_byte_identical_prompts(self) -> None:
        """The pair's whole claim: only the grading rule differs, so nothing else may."""
        group = train_rows(GRADING_THRESHOLD_GOODS_GROUP_MIX)
        own = train_rows(GRADING_THRESHOLD_GOODS_SELF)
        assert [row["prompt"] for row in group] == [row["prompt"] for row in own]
        assert [row["prompt_id"] for row in group] == [row["prompt_id"] for row in own]

    def test_the_variant_names_are_the_prize_variants_upstream(self) -> None:
        """Derived rather than restated, so a rename upstream cannot leave a stale pin here."""
        assert set(THRESHOLD_GOODS_PAYOFF_VARIANTS) == set(THRESHOLD_GOODS_PRIZE_VARIANTS)
        assert {str(row["payoff_variant"]) for row in train_rows()} == set(
            THRESHOLD_GOODS_PAYOFF_VARIANTS
        )

    def test_train_and_eval_frames_are_disjoint(self) -> None:
        train = {row["reskin_id"] for row in train_rows()}
        held_out = {
            row["reskin_id"]
            for row in generate_prompt_rows(
                THRESHOLD_GOODS_GAME_ID, GRADING_THRESHOLD_GOODS_GROUP_MIX, split=SPLIT_EVAL
            )
        }
        assert train
        assert held_out
        assert not train & held_out

    def test_the_prompt_ids_are_unique(self) -> None:
        ids = [
            row["prompt_id"]
            for split in SPLITS
            for row in generate_prompt_rows(
                THRESHOLD_GOODS_GAME_ID, GRADING_THRESHOLD_GOODS_GROUP_MIX, split=split
            )
        ]
        assert len(ids) == len(set(ids))

    def test_generation_is_deterministic(self) -> None:
        assert train_rows() == train_rows()


class TestTheRenderedPrompt:
    def test_the_prompt_is_exactly_its_four_sections_in_order(self) -> None:
        """A structural golden: catches a reordering, a dropped section, or an extra one."""
        scenario = THRESHOLD_GOODS_SCENARIOS[0]
        at_prize = spec(LOW_PRIZE)
        expected = "\n\n".join(
            [
                scenario.frame,
                THRESHOLD_GOODS_MECHANICS.format(
                    parties=PARTIES,
                    endowment=ENDOWMENT,
                    resource=scenario.resource,
                    undertaking=scenario.undertaking,
                    threshold=THRESHOLD,
                    prize=LOW_PRIZE,
                ),
                "About the other side: "
                + THRESHOLD_GOODS_COUNTERPART_CLAUSE.format(counterparts=TEAM_SIZE),
                THRESHOLD_GOODS_INSTRUCTION.format(endowment=ENDOWMENT),
            ]
        )
        assert render_threshold_goods_prompt(at_prize, scenario) == expected

    def test_the_two_prizes_differ_only_in_their_numbers(self) -> None:
        """The counterbalance golden's analogue: this game has no labels to swap, only a prize."""
        scenario = THRESHOLD_GOODS_SCENARIOS[0]
        numbers = re.compile(r"\d[\d.]*")
        masked = {
            numbers.sub("#", render_threshold_goods_prompt(spec(prize), scenario))
            for prize in THRESHOLD_GOODS_PRIZE_VARIANTS.values()
        }
        assert len(masked) == 1

    @pytest.mark.parametrize("prize", [LOW_PRIZE, HIGH_PRIZE])
    def test_the_prompt_states_every_number_the_reward_reads(self, prize: int) -> None:
        prompt = render_threshold_goods_prompt(spec(prize), THRESHOLD_GOODS_SCENARIOS[0])
        assert f"There are {PARTIES} of you" in prompt
        assert f"holding {ENDOWMENT} " in prompt
        assert f"come to {THRESHOLD} " in prompt
        assert f"less than {THRESHOLD}" in prompt
        assert f"a further {prize} " in prompt
        assert f"matched with {TEAM_SIZE} other instances" in prompt
        assert f"between 0 and {ENDOWMENT}" in prompt
        assert "<contribute>N</contribute>" in prompt

    def test_the_prompt_says_what_is_put_in_is_spent(self) -> None:
        """Load-bearing: without it the figure looks refundable and there is no cost to weigh."""
        prompt = render_threshold_goods_prompt(spec(), THRESHOLD_GOODS_SCENARIOS[0])
        assert "is spent, whichever way this goes" in prompt

    def test_the_prompt_says_a_party_that_put_in_nothing_is_paid_too(self) -> None:
        """Load-bearing: without it the undertaking is a purchase and nothing can be free-ridden on."""
        prompt = render_threshold_goods_prompt(spec(), THRESHOLD_GOODS_SCENARIOS[0])
        assert "including anyone who put in nothing at all" in prompt

    def test_the_prompt_says_what_the_final_figure_is(self) -> None:
        """Load-bearing: it is what tells the model over-contributing is charged for, which is what
        makes the optimum interior rather than a corner."""
        prompt = render_threshold_goods_prompt(spec(), THRESHOLD_GOODS_SCENARIOS[0])
        assert "what you held back, plus that" in prompt

    def test_every_frame_renders_at_every_prize_and_passes_the_guard(self) -> None:
        for scenario in THRESHOLD_GOODS_SCENARIOS:
            for prize in THRESHOLD_GOODS_PRIZE_VARIANTS.values():
                assert_no_loaded_vocabulary(render_threshold_goods_prompt(spec(prize), scenario))

    def test_no_prompt_embeds_its_own_identifiers(self) -> None:
        for split in SPLITS:
            for row in generate_prompt_rows(
                THRESHOLD_GOODS_GAME_ID, GRADING_THRESHOLD_GOODS_GROUP_MIX, split=split
            ):
                prompt = str(row["prompt"])
                assert str(row["game_id"]) not in prompt
                assert str(row["reskin_id"]) not in prompt
                assert str(row["payoff_variant"]) not in prompt

    def test_a_swapped_print_order_is_refused_rather_than_ignored(self) -> None:
        """It prints no labels, so a swapped build would be canonical rows under a wrong column."""
        with pytest.raises(ValueError, match="no action labels"):
            generate_prompt_rows(
                THRESHOLD_GOODS_GAME_ID,
                GRADING_THRESHOLD_GOODS_GROUP_MIX,
                split=SPLIT_TRAIN,
                label_print_order="swapped",
            )

    @pytest.mark.parametrize(
        ("field", "planted"),
        [
            ("frame", "This is a prisoner's dilemma."),
            ("resource", "prisoner rations"),
            ("undertaking", "the stag pen"),
        ],
    )
    def test_a_planted_banned_word_makes_rendering_raise(self, field: str, planted: str) -> None:
        """The vocabulary guard on this renderer, sabotaged rather than assumed.

        The planted phrase goes into the frame as well when the sabotaged field is not the frame, because
        `ThresholdGoodsScenario` refuses a resource or undertaking the frame does not mention -- so
        without that the wrong guard would fire and the vocabulary guard would stay unexercised.
        """
        original = THRESHOLD_GOODS_SCENARIOS[0]
        changes: dict[str, str] = {"frame": f"{original.frame}\n\n{planted}"}
        if field != "frame":
            changes[field] = planted
        sabotaged = dataclasses.replace(original, **changes)
        with pytest.raises(ValueError, match="loaded vocabulary"):
            render_threshold_goods_prompt(spec(), sabotaged)


class TestTheFramesSayNothingLoaded:
    """The roster polices its own construct vocabulary, since the global regex cannot be tightened."""

    def test_the_construct_check_can_fail(self) -> None:
        """The negative control: a check nobody has watched fail is not yet a check."""
        assert _CONSTRUCT_RE.search("Each puts in a fair share of the cost.") is not None
        assert _CONSTRUCT_RE.search("This is a threshold public goods game.") is not None

    def test_ordinary_frame_prose_passes_the_construct_check(self) -> None:
        assert _CONSTRUCT_RE.search("Each miller enters a figure in the race book.") is None

    def test_no_frame_uses_construct_vocabulary(self) -> None:
        for scenario in THRESHOLD_GOODS_SCENARIOS:
            match = _CONSTRUCT_RE.search(scenario.frame)
            assert match is None, (scenario.scenario_id, match)

    def test_no_rendered_prompt_uses_construct_vocabulary(self) -> None:
        for scenario in THRESHOLD_GOODS_SCENARIOS:
            prompt = render_threshold_goods_prompt(spec(), scenario)
            match = _CONSTRUCT_RE.search(prompt)
            assert match is None, (scenario.scenario_id, match)

    def test_every_frame_is_clean_under_the_global_guard(self) -> None:
        for scenario in THRESHOLD_GOODS_SCENARIOS:
            assert_no_loaded_vocabulary(scenario.frame)
            assert_no_loaded_vocabulary(scenario.scenario_id)
            assert_no_loaded_vocabulary(scenario.resource)
            assert_no_loaded_vocabulary(scenario.undertaking)

    def test_no_frame_states_how_many_parties_there_are(self) -> None:
        """The §1.7 guard at the authoring end.

        The mechanics paragraph prints the party count from the same column the reward draws its
        counterparts from. A frame that spelled the count out in words would silently contradict the
        graded game the first time the team size moved, and nothing downstream would notice.
        """
        for scenario in THRESHOLD_GOODS_SCENARIOS:
            match = _PARTY_COUNT_RE.search(scenario.frame)
            assert match is None, (scenario.scenario_id, match)

    def test_that_check_can_fail(self) -> None:
        assert _PARTY_COUNT_RE.search("Three mills draw off the same race.") is not None
        assert _PARTY_COUNT_RE.search("The 3 mills draw off the same race.") is not None
        assert _PARTY_COUNT_RE.search("One trailer goes out on the Thursday run.") is None

    def test_every_frame_sets_up_parties_each_holding_their_own_stock(self) -> None:
        """A frame authored as one party's decision would render a different game entirely.

        The mechanics paragraph states what the pooled figures do and never who is holding them, so the
        frame is the only place the other parties and the simultaneity exist. A frame missing either would
        read as the unilateral split, whose optimum is the opposite corner.
        """
        simultaneity = ("same", "together", "until", "before", "seeing", "shown", "only once")
        for scenario in THRESHOLD_GOODS_SCENARIOS:
            frame = scenario.frame.casefold()
            assert "each" in frame, scenario.scenario_id
            assert "own" in frame, scenario.scenario_id
            assert any(phrase in frame for phrase in simultaneity), scenario.scenario_id

    def test_a_one_sided_frame_would_fail_that_check(self) -> None:
        """The negative control for it: the unilateral-split roster is what it must reject."""
        from games.prompts import DICTATOR_SCENARIOS  # noqa: PLC0415 -- the contrast case only

        one_sided = DICTATOR_SCENARIOS[4].frame.casefold()
        assert "each" not in one_sided

    def test_the_roster_ids_resources_and_undertakings_are_distinct(self) -> None:
        ids = [scenario.scenario_id for scenario in THRESHOLD_GOODS_SCENARIOS]
        resources = [scenario.resource for scenario in THRESHOLD_GOODS_SCENARIOS]
        undertakings = [scenario.undertaking for scenario in THRESHOLD_GOODS_SCENARIOS]
        assert len(ids) == len(set(ids))
        assert len(resources) == len(set(resources))
        assert len(undertakings) == len(set(undertakings))

    def test_the_roster_is_the_expected_size_and_holds_out_frames(self) -> None:
        held_out = [scenario for scenario in THRESHOLD_GOODS_SCENARIOS if scenario.eval_only]
        assert len(THRESHOLD_GOODS_SCENARIOS) == 20
        assert len(held_out) == 4

    def test_an_empty_frame_resource_or_undertaking_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty frame"):
            ThresholdGoodsScenario(
                scenario_id="blank", frame="   ", resource="loads", undertaking="the works"
            )
        with pytest.raises(ValueError, match="empty resource"):
            ThresholdGoodsScenario(
                scenario_id="blank", frame="a frame", resource=" ", undertaking="the works"
            )
        with pytest.raises(ValueError, match="names no undertaking"):
            ThresholdGoodsScenario(
                scenario_id="blank", frame="a frame", resource="loads", undertaking=" "
            )

    def test_a_frame_that_names_its_goods_something_else_is_refused(self) -> None:
        """The renderer states the amounts in the authored words, so a mismatch counts another thing."""
        with pytest.raises(ValueError, match="never mentions"):
            ThresholdGoodsScenario(
                scenario_id="mismatch",
                frame="The works note mentions the culvert but not the units.",
                resource="loads",
                undertaking="the culvert",
            )
        with pytest.raises(ValueError, match="never mentions"):
            ThresholdGoodsScenario(
                scenario_id="mismatch",
                frame="The works note mentions loads but not what is being built.",
                resource="loads",
                undertaking="the culvert",
            )


class TestTheProseAndTheGradingReadTheSameColumns:
    """§1.7 of the design brief: a prompt that says three parties and a reward that assumes two.

    That is a lie no loss curve can show, so the party count in the prose and the counterpart count the
    grading draws from have to be the same column. They are, because the renderer and the reward both
    read one spec -- these tests are what would notice if either stopped.
    """

    def test_the_prose_party_count_moves_with_the_team_size_column(self) -> None:
        pair = spec(team_size=2, contribution_threshold=9)
        trio = spec(team_size=3, contribution_threshold=8)
        scenario = THRESHOLD_GOODS_SCENARIOS[0]
        assert "There are 3 of you" in render_threshold_goods_prompt(pair, scenario)
        assert "matched with 2 other instances" in render_threshold_goods_prompt(pair, scenario)
        assert "There are 4 of you" in render_threshold_goods_prompt(trio, scenario)
        assert "matched with 3 other instances" in render_threshold_goods_prompt(trio, scenario)

    def test_the_graded_counterpart_count_moves_with_the_same_column(self) -> None:
        """Same pool and same figure, one more counterpart drawn: a hardcoded count cannot pass both.

        The thresholds differ because they have to: the equal share must stay an integer, so a fourth
        party needs a bar divisible by four. The oracle is told both numbers rather than reusing this
        game's registered ones, which is what makes the comparison about the counterpart count.
        """
        pool = [0, 2, 3, 3, 5]
        pair = threshold_goods_reach_probability(
            spec(team_size=2, contribution_threshold=9), 3, pool
        )
        trio = threshold_goods_reach_probability(
            spec(team_size=3, contribution_threshold=8), 3, pool
        )
        assert pair == pytest.approx(float(hand_reach(3, pool, team_size=2, threshold=9)))
        assert trio == pytest.approx(float(hand_reach(3, pool, team_size=3, threshold=8)))
        # Three counterparts pool more than two do, so a lower bar is cleared more often still.
        assert trio > pair
        # And the count alone moves it, holding the bar fixed at a figure divisible by both.
        at_six = [
            threshold_goods_reach_probability(
                spec(team_size=size, contribution_threshold=6), 3, pool
            )
            for size in (1, 2)
        ]
        assert at_six[1] > at_six[0]

    def test_the_corpus_row_carries_the_count_the_prompt_printed(self) -> None:
        for row in train_rows():
            prompt = str(row["prompt"])
            assert f"There are {int(str(row['team_size'])) + 1} of you" in prompt
            assert f"matched with {row['team_size']} other instances" in prompt

    def test_the_reward_rebuilds_the_spec_from_the_row(self) -> None:
        """So a corpus written before a constant moved is graded as the prompt it actually printed."""
        from games.rewards import _Row  # noqa: PLC0415 -- private, tested at its own seam

        row = row_at(HIGH_PRIZE_VARIANT)
        rebuilt = _Row(
            **{name: row[name] for name in REQUIRED_REWARD_COLUMNS}
        ).threshold_goods_spec()
        assert rebuilt.endowment == row["endowment"]
        assert rebuilt.team_size == row["team_size"]
        assert rebuilt.contribution_threshold == row["contribution_threshold"]
        assert rebuilt.prize == row["prize"]

    def test_another_games_row_cannot_be_graded_as_this_one(self) -> None:
        """Every other row builder writes zeroes into these columns, and the spec refuses them."""
        from games.rewards import _Row  # noqa: PLC0415 -- private, tested at its own seam

        other = generate_prompt_rows("twin-pd", "group-mix", split=SPLIT_TRAIN)[0]
        rebuilt = _Row(**{name: other[name] for name in REQUIRED_REWARD_COLUMNS})
        with pytest.raises(ValueError, match="non-positive"):
            rebuilt.threshold_goods_spec()


class TestTheRegistryRoundTrip:
    ARM_NAMES: tuple[str, str, str] = (
        "threshold-goods-prize-below-solo-cost",
        "threshold-goods-prize-above-solo-cost",
        "threshold-goods-self",
    )

    def test_all_three_arms_are_registered_and_valid(self) -> None:
        for name in self.ARM_NAMES:
            arm = ARMS[name]
            assert arm.game_id == THRESHOLD_GOODS_GAME_ID
            assert arm.grading in THRESHOLD_GOODS_GRADINGS
            assert arm.notes
        validate_arms({name: ARMS[name] for name in self.ARM_NAMES})

    def test_the_two_group_mix_arms_differ_only_in_the_pinned_prize(self) -> None:
        low, high = (ARMS[name] for name in self.ARM_NAMES[:2])
        assert low.game_id == high.game_id
        assert low.grading == high.grading == GRADING_THRESHOLD_GOODS_GROUP_MIX
        assert low.payoff_variants == (LOW_PRIZE_VARIANT,)
        assert high.payoff_variants == (HIGH_PRIZE_VARIANT,)

    def test_the_self_graded_arm_runs_both_prizes(self) -> None:
        """The optimum is the equal share under each, and their spans are inside the 2x spread rule."""
        own = ARMS["threshold-goods-self"]
        assert own.grading == GRADING_THRESHOLD_GOODS_SELF
        assert own.payoff_variants == ()
        assert arm_payoff_variants(own) == set(THRESHOLD_GOODS_PAYOFF_VARIANTS)

    def test_the_gradings_stay_out_of_the_binary_action_set(self) -> None:
        """Adding one there would ask selection for a cooperation rate that does not exist."""
        assert not THRESHOLD_GOODS_GRADINGS & ONE_SHOT_ACTION_GRADINGS

    def test_both_gradings_can_be_parsed_by_the_sweep(self) -> None:
        """A grading in GRADINGS with no parser here dies mid-sweep, after the GPU has been paid for."""
        assert THRESHOLD_GOODS_GRADINGS <= PARSEABLE_GRADINGS

    def test_both_gradings_require_the_contribution_metrics_at_read_back(self) -> None:
        for grading in sorted(THRESHOLD_GOODS_GRADINGS):
            required = required_metrics_for(grading)
            assert "mean_contribution_fraction" in required
            assert "coop_rate" not in required

    def test_each_grading_requires_its_own_reach_metric_and_not_the_other_one(self) -> None:
        """One shared key would let either arm pass the read-back gate on the other's quantity."""
        group_mix = required_metrics_for(GRADING_THRESHOLD_GOODS_GROUP_MIX)
        self_graded = required_metrics_for(GRADING_THRESHOLD_GOODS_SELF)
        assert "threshold_met_rate" in group_mix
        assert "equal_share_met_rate" not in group_mix
        assert "equal_share_met_rate" in self_graded
        assert "threshold_met_rate" not in self_graded

    def test_the_three_arms_are_three_different_experiments(self) -> None:
        keys = {
            (arm.game_id, arm.grading, arm.payoff_variants, arm.corpus_partition)
            for arm in (ARMS[name] for name in self.ARM_NAMES)
        }
        assert len(keys) == len(self.ARM_NAMES)


class TestTheDeadArmGuardCoversThisGrading:
    """The self-graded leg is span-checked and the group-mix leg is exempt with a stated reason."""

    def test_the_real_span_is_the_widest_the_game_can_reach(self) -> None:
        assert threshold_goods_reward_span(ARMS["threshold-goods-self"]) == pytest.approx(0.5)

    def test_the_span_is_read_off_the_arms_own_pinned_rows(self) -> None:
        """A pinned arm's span is its own prize's, not the widest the game has anywhere."""
        pinned = GameArm(
            game_id=THRESHOLD_GOODS_GAME_ID,
            grading=GRADING_THRESHOLD_GOODS_SELF,
            notes="pinned to the low prize only",
            payoff_variants=(LOW_PRIZE_VARIANT,),
        )
        assert threshold_goods_reward_span(pinned) == pytest.approx(float(Fraction(7, 22)))

    def test_a_zero_span_arm_is_refused_with_the_mechanism_spelled_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sabotage kept as a test: a reward that stopped varying with the figure must be refused."""
        flattened = games_arms._SelfConsistentGrading(  # pyright: ignore[reportPrivateUsage]
            span=lambda _arm: 0.0,
            remedy="flattened for the test",
        )
        monkeypatch.setitem(
            games_arms.SELF_CONSISTENT_GRADINGS, GRADING_THRESHOLD_GOODS_SELF, flattened
        )
        with pytest.raises(ValueError, match="advantage is zero"):
            validate_arms({"threshold-goods-self": ARMS["threshold-goods-self"]})

    def test_the_matrix_span_would_read_zero_for_this_game(self) -> None:
        """Why the guard had to be per grading rather than keyed on the 2x2 cells.

        Every row of this game carries payoff_cc == payoff_dd == 0.0, so the matrix span reads 0.0 however
        healthy the arm is, and a guard measuring the cells would refuse it while checking nothing.
        """
        assert games_arms.self_grading_reward_span(ARMS["threshold-goods-self"]) == pytest.approx(
            0.0
        )

    def test_both_gradings_are_classified_exactly_once(self) -> None:
        """The fail-closed partition, which is what stopped the guard failing open at integration."""
        games_arms.assert_every_grading_is_classified()
        span_checked = set(games_arms.SELF_CONSISTENT_GRADINGS)
        exempt = set(games_arms.SPAN_CHECK_EXEMPTIONS)
        assert GRADING_THRESHOLD_GOODS_SELF in span_checked
        assert GRADING_THRESHOLD_GOODS_GROUP_MIX in exempt
        assert not THRESHOLD_GOODS_GRADINGS & (span_checked & exempt)

    def test_the_exemption_says_what_supplies_the_spread_instead(self) -> None:
        reason = games_arms.SPAN_CHECK_EXEMPTIONS[GRADING_THRESHOLD_GOODS_GROUP_MIX]
        assert "batch property" in reason
        assert "threshold_met_rate" in reason


class TestSelectionOnASyntheticSweep:
    """The spread floor is the only filter these rows meet, so it is the one that must have teeth."""

    def sweep(self, script: dict[str, list[str]], *, samples: int = 4) -> Any:
        rows = train_rows()[:3]
        for index, row in enumerate(rows):
            row["prompt_id"] = f"goods-{index}"
        backend = ScriptedBackend(
            {str(row["prompt"]): script[str(row["prompt_id"])] for row in rows}
        )
        return sweep_prompts(backend, rows, samples_per_prompt=samples, prefilled_think=False)

    def test_a_split_prompt_is_kept_and_a_unanimous_one_is_dropped(self) -> None:
        records = self.sweep(
            {
                "goods-0": [contribute_completion(f) for f in (0, 3, 3, 8)],
                "goods-1": [contribute_completion(3)] * 4,
                "goods-2": [contribute_completion(f) for f in (3, 3, 4, 3)],
            }
        )
        verdicts = {verdict.prompt_id: verdict for verdict in judge_prompts(records)}
        assert verdicts["goods-0"].keep
        assert verdicts["goods-0"].reason == DropReason.KEPT_MIXED
        assert not verdicts["goods-1"].keep
        assert verdicts["goods-1"].reason == DropReason.SCORE_SPREAD_BELOW_MIN
        # 3/3/4/3 spreads 0.043 of the stock, under the 0.05 floor: real but too small to train on.
        assert verdicts["goods-2"].score_std < DEFAULT_MIN_SPLIT_STD
        assert not verdicts["goods-2"].keep

    def test_the_selection_score_is_the_contributed_fraction(self) -> None:
        records = self.sweep(
            {
                "goods-0": [contribute_completion(f) for f in (0, 3, 5, 10)],
                "goods-1": [contribute_completion(3)] * 4,
                "goods-2": [contribute_completion(3)] * 4,
            }
        )
        first = next(record for record in records if record.prompt_id == "goods-0")
        assert first.selection_scores == pytest.approx([0.0, 0.3, 0.5, 1.0])
        assert [sample.contribution for sample in first.samples] == [0, 3, 5, 10]

    def test_a_mostly_unparseable_prompt_is_dropped_for_that_reason(self) -> None:
        records = self.sweep(
            {
                "goods-0": ["no figure", "no figure", "no figure", contribute_completion(3)],
                "goods-1": [contribute_completion(3)] * 4,
                "goods-2": [contribute_completion(3)] * 4,
            }
        )
        verdicts = {verdict.prompt_id: verdict for verdict in judge_prompts(records)}
        assert verdicts["goods-0"].reason == DropReason.TOO_FEW_PARSEABLE

    def test_an_out_of_range_figure_counts_as_a_parse_failure_in_the_sweep(self) -> None:
        records = self.sweep(
            {
                "goods-0": [
                    contribute_completion(0),
                    "<contribute>500</contribute>",
                    contribute_completion(3),
                    contribute_completion(8),
                ],
                "goods-1": [contribute_completion(3)] * 4,
                "goods-2": [contribute_completion(3)] * 4,
            }
        )
        first = next(record for record in records if record.prompt_id == "goods-0")
        assert first.n_parse_failures == 1

    def test_the_rows_are_singletons_with_no_counterbalanced_partner(self) -> None:
        for row in train_rows():
            assert not is_counterbalanced(row)

    def test_the_sweep_reports_no_cooperation_rate(self) -> None:
        records = self.sweep(
            {
                "goods-0": [contribute_completion(0)] * 4,
                "goods-1": [contribute_completion(3)] * 4,
                "goods-2": [contribute_completion(8)] * 4,
            }
        )
        assert all(record.coop_fraction is None for record in records)
        assert all(record.coop_count is None for record in records)


class TestTheEvalBatteryReadsTheContribution:
    def test_the_game_is_in_the_render_grading_map(self) -> None:
        """Its absence would be caught at import; this pins which grading the battery renders."""
        assert EVAL_RENDER_GRADING_BY_GAME[THRESHOLD_GOODS_GAME_ID] in THRESHOLD_GOODS_GRADINGS

    def test_the_battery_can_be_configured_for_this_game_alone(self) -> None:
        config = EvalConfig(
            games=(THRESHOLD_GOODS_GAME_ID,), trained_game_ids=(THRESHOLD_GOODS_GAME_ID,)
        )
        assert config.games == (THRESHOLD_GOODS_GAME_ID,)

    def test_the_behaviour_field_is_the_contribution_fraction_in_both_readouts(self) -> None:
        """Without this the tables render empty, and the readout recomputes itself in silence."""
        assert behaviour_field(THRESHOLD_GOODS_GAME_ID) == CONTRIBUTION_FIELD
        assert readout._behaviour_field(THRESHOLD_GOODS_GAME_ID) == "contribution_fraction"  # pyright: ignore[reportPrivateUsage]

    def test_a_record_carries_the_figure_and_its_fraction(self) -> None:
        from games.evals import _game_record  # noqa: PLC0415 -- private, tested at its own seam

        record = _game_record(
            THRESHOLD_GOODS_GAME_ID,
            row_at(HIGH_PRIZE_VARIANT),
            contribute_completion(4),
            prefilled_think=False,
            trained_game=True,
            eval_only_game=False,
            sample_index=0,
        )
        assert record["record"] == SECTION_GAME_BEHAVIOR
        assert record["contribution"] == 4
        assert record["contribution_fraction"] == pytest.approx(0.4)
        assert record["parsed"] is True

    def test_an_unparseable_completion_records_a_parse_failure_not_a_zero(self) -> None:
        from games.evals import _game_record  # noqa: PLC0415 -- private, tested at its own seam

        record = _game_record(
            THRESHOLD_GOODS_GAME_ID,
            row_at(HIGH_PRIZE_VARIANT),
            "I will not name a figure.",
            prefilled_think=False,
            trained_game=True,
            eval_only_game=False,
            sample_index=0,
        )
        assert record["contribution"] is None
        assert record["contribution_fraction"] is None
        assert record["parsed"] is False


class TestTheRewardSpreadReport:
    """The number that decides whether two prize variants may share a batch, and its honest caveat."""

    def test_the_self_graded_arm_reports_both_prizes_inside_the_two_times_rule(self) -> None:
        spreads = threshold_goods_variant_reward_spreads(ARMS["threshold-goods-self"])
        assert set(spreads) == set(THRESHOLD_GOODS_PAYOFF_VARIANTS)
        assert spreads[LOW_PRIZE_VARIANT] == pytest.approx(float(Fraction(7, 22)))
        assert spreads[HIGH_PRIZE_VARIANT] == pytest.approx(0.5)
        assert max(spreads.values()) / min(spreads.values()) < 2.0

    @pytest.mark.parametrize(
        "name",
        ["threshold-goods-prize-below-solo-cost", "threshold-goods-prize-above-solo-cost"],
    )
    def test_a_pinned_arm_reports_only_its_own_prize(self, name: str) -> None:
        spreads = threshold_goods_variant_reward_spreads(ARMS[name])
        assert set(spreads) == set(ARMS[name].payoff_variants)
        assert all(spread > DEFAULT_MIN_SPLIT_STD for spread in spreads.values())

    def test_the_group_mix_spread_is_taken_at_the_reference_distribution(self) -> None:
        at_prize = spec(LOW_PRIZE)
        pool = threshold_goods_reference_contributions(at_prize)
        expected = threshold_goods_group_reward_span(at_prize, pool, max_prize=HIGH_PRIZE)
        spreads = threshold_goods_variant_reward_spreads(
            ARMS["threshold-goods-prize-below-solo-cost"]
        )
        assert spreads[LOW_PRIZE_VARIANT] == pytest.approx(expected)
