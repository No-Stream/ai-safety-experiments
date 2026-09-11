"""The simultaneous-claim division, end to end: arithmetic, parsing, corpus, registry, selection.

Offline and CPU-only. Every expected number below is written as the arithmetic that produces it,
because the whole value of this game as an arm is that its prediction is a NUMBER -- the equal
division, reached from either side under either grading -- so a reward that is off by one unit or a
factor of the windfall would still train, still look healthy, and still answer a different question
from the one the pre-registration asks.

Two classes carry the weight.

:class:`TestTheSelfGradedOptimumIsTheEqualDivision` is the guard the design brief asks for: the
optimum is searched over the whole answer grid rather than asserted as `windfall // 2`, so flipping
the feasibility test from `<=` to `<` moves it by one unit and turns this red. That is exactly the
error no reward curve would ever reveal.

:class:`TestTheDeadArmGuardCoversThisGrading` covers the registry refusal. `self_grading_reward_span`
reads the 2x2 payoff cells, which this game carries as zeroes on every row, so the guard that
refuses a self-graded arm with no reward spread had to be generalised per grading rather than left
keyed on one grading name -- otherwise it would read 0.0 for every graded game and refuse healthy
arms while checking nothing about them.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

from games import arms as games_arms
from games import readout
from games.arms import ARMS, GameArm, arm_payoff_variants, nash_demand_reward_span, validate_arms
from games.battery_tables import behaviour_field
from games.evals import (
    CLAIM_FIELD,
    EVAL_RENDER_GRADING_BY_GAME,
    SECTION_GAME_BEHAVIOR,
    EvalConfig,
)
from games.parsing import parse_claim
from games.payoffs import (
    NashDemandSpec,
    nash_demand_best_response,
    nash_demand_crash_probability,
    nash_demand_fits,
    nash_demand_group_reward,
    nash_demand_group_reward_span,
    nash_demand_reference_claims,
    nash_demand_self_optimum,
    nash_demand_self_reward,
    nash_demand_self_reward_span,
    nash_demand_share,
)
from games.prompts import (
    NASH_DEMAND_GAME_ID,
    NASH_DEMAND_INSTRUCTION,
    NASH_DEMAND_MECHANICS,
    NASH_DEMAND_SCENARIOS,
    NASH_DEMAND_WINDFALLS,
    ROW_COLUMNS,
    SPLIT_EVAL,
    SPLIT_TRAIN,
    SPLITS,
    TWIN_COUNTERPART_CLAUSE,
    assert_no_loaded_vocabulary,
    generate_prompt_rows,
    nash_demand_variant,
    render_nash_demand_prompt,
)
from games.rewards import (
    DEFAULT_PARSE_PENALTY,
    GRADING_NASH_DEMAND_GROUP_MIX,
    GRADING_NASH_DEMAND_SELF,
    NASH_DEMAND_GRADINGS,
    REQUIRED_REWARD_COLUMNS,
    make_game_reward,
)
from games.select_prompts import (
    DEFAULT_MIN_SPLIT_STD,
    ONE_SHOT_ACTION_GRADINGS,
    DropReason,
    is_counterbalanced,
    judge_prompts,
    sweep_prompts,
)
from games.train import required_metrics_for

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

REFERENCE_WINDFALL = 100

# Construct vocabulary this game brings with it. None of it is in the global banned list, and it
# must not be added there: "generous" and "fair" already appear in tracked interp stimuli and probe
# text that have nothing to do with wave 2, so tightening the global regex would turn unrelated
# tests red. The roster polices itself instead, which is what `TestTheFramesSayNothingLoaded` is.
# "demand" is on the list because it is the literature's name for this game.
CONSTRUCT_VOCABULARY: tuple[str, ...] = (
    r"demand\w*",
    r"fair\w*",
    r"generous\w*",
    r"selfish\w*",
    r"free[\s-]?rid\w*",
    r"greed\w*",
    r"trust\w*",
    r"reciprocat\w*",
    r"altruis\w*",
    r"bargain\w*",
)
_CONSTRUCT_RE = re.compile(r"\b(?:" + "|".join(CONSTRUCT_VOCABULARY) + r")\b", re.IGNORECASE)


def spec(windfall: int = REFERENCE_WINDFALL) -> NashDemandSpec:
    """The game at one windfall, built the way the corpus builder builds it."""
    return NashDemandSpec(game_id=NASH_DEMAND_GAME_ID, windfall=windfall)


def claim_completion(claim: int) -> str:
    """A completion as a non-prefilling model emits one: both thinking tags, then the answer."""
    return f"<think>naming a figure</think><claim>{claim}</claim>"


def train_rows(grading: str = GRADING_NASH_DEMAND_GROUP_MIX) -> list[dict[str, Any]]:
    """Every training row of this game under one grading."""
    return generate_prompt_rows(NASH_DEMAND_GAME_ID, grading, split=SPLIT_TRAIN)


def row_at(windfall: int, grading: str = GRADING_NASH_DEMAND_GROUP_MIX) -> dict[str, Any]:
    """One real corpus row at a given windfall, so the reward is fed what training feeds it."""
    return next(row for row in train_rows(grading) if row["windfall"] == windfall)


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

    An unscripted prompt raises rather than answering plausibly: a sweep that silently sampled a
    prompt nobody wrote completions for would produce a table of numbers about nothing.
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


class TestTheWindfallHasToHaveAnIntegerHalf:
    """The exact-half rate is this arm's fairness anchor, so the anchor has to be answerable."""

    def test_an_odd_windfall_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be even"):
            spec(61)

    def test_the_refusal_says_why_rather_than_only_that(self) -> None:
        with pytest.raises(ValueError, match="exact-half") as raised:
            spec(99)
        assert "integer claim" in str(raised.value)

    @pytest.mark.parametrize("windfall", [0, -2])
    def test_a_non_positive_windfall_is_refused(self, windfall: int) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            spec(windfall)

    def test_every_registered_windfall_passes(self) -> None:
        for windfall in NASH_DEMAND_WINDFALLS:
            assert spec(windfall).equal_claim * 2 == windfall

    @pytest.mark.parametrize("claim", [-1, 101])
    def test_a_claim_off_the_grid_is_a_caller_bug(self, claim: int) -> None:
        with pytest.raises(ValueError, match="claim must lie"):
            spec().require_claim(claim)

    @pytest.mark.parametrize("claim", [0, 50, 100])
    def test_the_grid_includes_both_ends(self, claim: int) -> None:
        spec().require_claim(claim)


class TestTheSelfGradedOptimumIsTheEqualDivision:
    """Searched over the grid, so an off-by-one in the feasibility test moves it and turns red."""

    @pytest.mark.parametrize(
        ("claim", "expected"),
        [(0, 0.0), (30, 0.30), (49, 0.49), (50, 0.50), (51, 0.0), (100, 0.0)],
    )
    def test_the_reward_is_a_ramp_with_a_cliff_at_half(self, claim: int, expected: float) -> None:
        assert nash_demand_self_reward(spec(), claim) == pytest.approx(expected)

    @pytest.mark.parametrize("windfall", NASH_DEMAND_WINDFALLS)
    def test_the_optimum_is_exactly_half_the_windfall(self, windfall: int) -> None:
        at_windfall = spec(windfall)
        assert nash_demand_self_optimum(at_windfall) == windfall // 2
        assert nash_demand_self_optimum(at_windfall) == at_windfall.equal_claim

    @pytest.mark.parametrize("windfall", NASH_DEMAND_WINDFALLS)
    def test_the_equal_division_fits_and_one_unit_more_does_not(self, windfall: int) -> None:
        """The `<=` boundary itself, which is the byte the off-by-one sabotage flips."""
        at_windfall = spec(windfall)
        equal = at_windfall.equal_claim
        assert nash_demand_fits(at_windfall, equal, equal)
        assert not nash_demand_fits(at_windfall, equal + 1, equal + 1)

    @pytest.mark.parametrize("windfall", NASH_DEMAND_WINDFALLS)
    def test_the_reachable_span_is_half_whatever_the_windfall(self, windfall: int) -> None:
        """Half rather than one, because no figure can pay more than half against a copy of you."""
        assert nash_demand_self_reward_span(spec(windfall)) == pytest.approx(0.5)


class TestTheGroupGradedSurface:
    def test_the_rewards_are_the_hand_computed_products(self) -> None:
        claims = [45, 50, 55]
        at_100 = spec()
        # 45 fits against all three (45+55 = 100); 50 fits against 45 and 50; 55 only against 45.
        assert nash_demand_group_reward(at_100, 45, claims) == pytest.approx(0.45)
        assert nash_demand_group_reward(at_100, 50, claims) == pytest.approx(0.50 * 2 / 3)
        assert nash_demand_group_reward(at_100, 55, claims) == pytest.approx(0.55 * 1 / 3)

    def test_the_crash_rate_is_the_complement_of_the_fit_rate(self) -> None:
        claims = [45, 50, 55]
        at_100 = spec()
        assert nash_demand_crash_probability(at_100, 45, claims) == pytest.approx(0.0)
        assert nash_demand_crash_probability(at_100, 50, claims) == pytest.approx(1 / 3)
        assert nash_demand_crash_probability(at_100, 55, claims) == pytest.approx(2 / 3)

    @pytest.mark.parametrize("windfall", NASH_DEMAND_WINDFALLS)
    @pytest.mark.parametrize("group_fraction", [0.0, 0.2, 0.4, 0.5, 0.6, 0.8])
    def test_the_best_response_to_a_group_at_one_figure_is_the_remainder(
        self, windfall: int, group_fraction: float
    ) -> None:
        """A group at 40 of 100 best responds with 60 and one at 60 with 40, from either side."""
        at_windfall = spec(windfall)
        group_claim = round(group_fraction * windfall)
        assert nash_demand_best_response(at_windfall, [group_claim] * 4) == windfall - group_claim

    @pytest.mark.parametrize("windfall", NASH_DEMAND_WINDFALLS)
    def test_the_equal_division_is_the_fixed_point(self, windfall: int) -> None:
        at_windfall = spec(windfall)
        equal = at_windfall.equal_claim
        assert nash_demand_best_response(at_windfall, [equal] * 4) == equal

    def test_a_group_at_a_corner_still_has_a_best_response_inside_the_grid(self) -> None:
        """Documented so a corner outcome reads as a behavioural finding, not an unreachable state."""
        assert nash_demand_best_response(spec(), [0] * 4) == REFERENCE_WINDFALL
        assert nash_demand_best_response(spec(), [REFERENCE_WINDFALL] * 4) == 0

    def test_without_crashes_the_reward_is_exactly_the_claimed_fraction(self) -> None:
        """So a crash-free group is not a gradient-free one, contrary to the wave-2 design brief.

        Every pair drawn from these claims fits inside the windfall, so every fit rate is 1 and the
        reward collapses to the claim itself. The within-group span is then the spread of the claims,
        which can be wide -- the brief's "an arm with no crashes trains formatting" is wrong as
        written. What a crash-free group really lacks is the counter-pressure: see the next test.
        """
        claims = [10, 20, 30, 40]
        at_100 = spec()
        for claim in claims:
            assert nash_demand_group_reward(at_100, claim, claims) == pytest.approx(claim / 100)
            assert nash_demand_crash_probability(at_100, claim, claims) == 0.0
        assert nash_demand_group_reward_span(at_100, claims) == pytest.approx(0.30)

    def test_crashes_are_what_stop_a_bigger_claim_from_always_paying_more(self) -> None:
        """The real mechanism the crash rate reports, and why it is worth logging per step.

        With crashes in the group the reward stops being monotone in the claim: here the biggest
        claimer earns the least of the three, which is arithmetically impossible in a crash-free
        group. That non-monotonicity is what holds the arm at an interior claim instead of walking it
        to the top of the grid, so a step whose crash rate is zero is a step whose gradient points
        only upward -- a different reading from "no signal", and the one the readout should make.
        """
        claims = [45, 50, 55]
        at_100 = spec()
        rewards = [nash_demand_group_reward(at_100, claim, claims) for claim in claims]
        assert rewards[2] < rewards[0]
        assert nash_demand_best_response(at_100, claims) == 100 - max(claims)
        fitting = [10, 20, 30, 40]
        monotone = [nash_demand_group_reward(at_100, claim, fitting) for claim in fitting]
        assert monotone == sorted(monotone)
        assert nash_demand_best_response(at_100, fitting) == 100 - max(fitting)

    @pytest.mark.parametrize("windfall", NASH_DEMAND_WINDFALLS)
    def test_the_reference_spread_is_the_same_at_every_windfall(self, windfall: int) -> None:
        """Scale-free in the windfall, which is what lets all three run in one arm."""
        at_windfall = spec(windfall)
        at_reference = spec()
        assert nash_demand_group_reward_span(
            at_windfall, nash_demand_reference_claims(at_windfall)
        ) == pytest.approx(
            nash_demand_group_reward_span(at_reference, nash_demand_reference_claims(at_reference))
        )

    def test_a_missing_counterpart_distribution_raises_rather_than_defaulting(self) -> None:
        with pytest.raises(ValueError, match="at least one claim"):
            nash_demand_group_reward(spec(), 50, [])

    def test_the_share_is_the_claim_over_the_windfall(self) -> None:
        assert nash_demand_share(spec(60), 30) == pytest.approx(0.5)
        assert nash_demand_share(spec(200), 30) == pytest.approx(0.15)


class TestParseClaim:
    def test_a_plain_answer_parses(self) -> None:
        assert parse_claim("<claim>42</claim>", windfall=100) == 42

    def test_the_last_tag_wins(self) -> None:
        assert parse_claim("<claim>10</claim> on reflection <claim>60</claim>", windfall=100) == 60

    def test_the_tag_is_case_insensitive_and_tolerates_space(self) -> None:
        assert parse_claim("<CLAIM> 25 </CLAIM>", windfall=100) == 25

    @pytest.mark.parametrize("text", ["<claim>101</claim>", "<claim>-1</claim>"])
    def test_a_figure_off_the_grid_is_a_parse_failure_not_a_clamp(self, text: str) -> None:
        assert parse_claim(text, windfall=100) is None

    @pytest.mark.parametrize(
        "text", ["<claim>half</claim>", "<claim>50.5</claim>", "<claim></claim>"]
    )
    def test_a_non_integer_figure_is_a_parse_failure(self, text: str) -> None:
        assert parse_claim(text, windfall=100) is None

    def test_no_tag_is_a_parse_failure(self) -> None:
        assert parse_claim("I would take about half of it.", windfall=100) is None

    def test_a_non_positive_windfall_is_a_caller_bug(self) -> None:
        with pytest.raises(ValueError, match="windfall must be positive"):
            parse_claim("<claim>1</claim>", windfall=0)


class TestTheRewardFunctionOnRealCorpusRows:
    """Fed the columns the corpus actually writes, so the seam between the two is under test."""

    def test_self_grading_scores_each_claim_against_its_own_figure(
        self, recorder: Recorder
    ) -> None:
        row = row_at(REFERENCE_WINDFALL, GRADING_NASH_DEMAND_SELF)
        rewards = score(
            [row] * 3,
            [claim_completion(50), claim_completion(30), claim_completion(60)],
            recorder,
        )
        assert rewards == pytest.approx([0.5, 0.3, 0.0])

    def test_group_mix_grading_scores_each_claim_against_the_group(
        self, recorder: Recorder
    ) -> None:
        row = row_at(REFERENCE_WINDFALL)
        rewards = score(
            [row] * 3,
            [claim_completion(45), claim_completion(50), claim_completion(55)],
            recorder,
        )
        assert rewards == pytest.approx([0.45, 0.50 * 2 / 3, 0.55 / 3])

    def test_the_two_gradings_log_their_overreach_under_different_keys(
        self, recorder: Recorder
    ) -> None:
        """The same claims read 0.400 under both gradings, for two unrelated reasons.

        Under group-mix that number is a mean collision rate against the group's realised claims;
        under self grading it is the deterministic indicator `claim > windfall / 2`, which is
        redundant with `mean_claim_fraction`. Sharing one key would put a collision-rate series and
        an over-claiming series on one axis in the readout, which is a false reading of the
        self-graded arm rather than a cosmetic one.
        """
        claims = [claim_completion(claim) for claim in (30, 45, 50, 55, 70)]

        group_recorder = Recorder()
        score([row_at(REFERENCE_WINDFALL)] * 5, claims, group_recorder)
        assert group_recorder.metrics["crash_rate"] == pytest.approx(0.4)
        assert "overclaim_rate" not in group_recorder.metrics

        score([row_at(REFERENCE_WINDFALL, GRADING_NASH_DEMAND_SELF)] * 5, claims, recorder)
        assert recorder.metrics["overclaim_rate"] == pytest.approx(0.4)
        assert "crash_rate" not in recorder.metrics

    def test_the_self_graded_overclaim_rate_is_the_over_half_indicator(
        self, recorder: Recorder
    ) -> None:
        """Which is what makes it a different quantity, not a differently-named one."""
        row = row_at(REFERENCE_WINDFALL, GRADING_NASH_DEMAND_SELF)
        score([row] * 4, [claim_completion(claim) for claim in (49, 50, 51, 99)], recorder)
        assert recorder.metrics["overclaim_rate"] == pytest.approx(0.5)

    def test_the_windfall_column_is_what_the_reward_reads(self, recorder: Recorder) -> None:
        """Not a constructor: the corpus on disk is the ground truth for what prompt was read."""
        row = row_at(60, GRADING_NASH_DEMAND_SELF)
        rewards = score([row] * 2, [claim_completion(30), claim_completion(31)], recorder)
        assert rewards == pytest.approx([0.5, 0.0])

    def test_an_unparseable_completion_takes_the_penalty_and_leaves_the_group(
        self, recorder: Recorder
    ) -> None:
        row = row_at(REFERENCE_WINDFALL)
        rewards = score(
            [row] * 3,
            [claim_completion(50), "I would rather not put a figure on it.", claim_completion(60)],
            recorder,
        )
        # The counterpart distribution is {50, 60}: 50 fits against 50 only, 60 against neither.
        assert rewards == pytest.approx([0.5 * 0.5, DEFAULT_PARSE_PENALTY, 0.0])
        assert recorder.metrics["parse_failure_rate"] == pytest.approx(1 / 3)

    def test_leave_one_out_falls_back_to_the_fit_prior_and_counts_it(
        self, recorder: Recorder
    ) -> None:
        row = row_at(REFERENCE_WINDFALL)
        reward = make_game_reward(2, prefilled_think=False, leave_one_out=True)
        rewards = score(
            [row] * 2,
            [claim_completion(40), "no figure from me"],
            recorder,
            reward=reward,
        )
        assert rewards == pytest.approx([0.40 * 0.5, DEFAULT_PARSE_PENALTY])
        assert recorder.metrics["leave_one_out_prior_rate"] == pytest.approx(0.5)

    def test_leave_one_out_grades_against_the_others_only(self, recorder: Recorder) -> None:
        row = row_at(REFERENCE_WINDFALL)
        reward = make_game_reward(2, prefilled_think=False, leave_one_out=True)
        rewards = score(
            [row] * 2, [claim_completion(60), claim_completion(60)], recorder, reward=reward
        )
        # Each is graded against the other's 60, and 60 + 60 overruns the windfall, so both crash.
        assert rewards == pytest.approx([0.0, 0.0])
        assert recorder.metrics["crash_rate"] == pytest.approx(1.0)

    def test_the_behavioural_metrics_are_all_logged(self, recorder: Recorder) -> None:
        row = row_at(REFERENCE_WINDFALL)
        score(
            [row] * 4,
            [
                claim_completion(50),
                claim_completion(50),
                claim_completion(30),
                claim_completion(70),
            ],
            recorder,
        )
        assert recorder.metrics["mean_claim_fraction"] == pytest.approx((0.5 + 0.5 + 0.3 + 0.7) / 4)
        assert recorder.metrics["exact_half_claim_rate"] == pytest.approx(0.5)
        # 50 fits against {50, 50, 30} but not 70; 30 fits against everything; 70 only against 30.
        assert recorder.metrics["crash_rate"] == pytest.approx((0.25 + 0.25 + 0.0 + 0.75) / 4)
        assert recorder.metrics["mean_group_reward_span"] > 0.0

    def test_a_unanimous_group_reports_no_reward_span_and_full_purity(
        self, recorder: Recorder
    ) -> None:
        row = row_at(REFERENCE_WINDFALL)
        score([row] * 3, [claim_completion(50)] * 3, recorder)
        assert recorder.metrics["mean_group_reward_span"] == pytest.approx(0.0)
        assert recorder.metrics["frac_groups_pure"] == pytest.approx(1.0)

    def test_the_per_completion_column_carries_the_claim(self, recorder: Recorder) -> None:
        row = row_at(REFERENCE_WINDFALL)
        score([row] * 2, [claim_completion(40), claim_completion(50)], recorder)
        assert recorder.extra["parsed_action"] == ["claim=40", "claim=50"]

    def test_no_cooperation_or_keep_rate_is_invented(self, recorder: Recorder) -> None:
        row = row_at(REFERENCE_WINDFALL)
        score([row] * 2, [claim_completion(40), claim_completion(50)], recorder)
        assert "coop_rate" not in recorder.metrics
        assert "mean_keep_fraction" not in recorder.metrics


class TestTheCorpus:
    def test_the_row_counts_are_frames_times_windfalls(self) -> None:
        train = [scenario for scenario in NASH_DEMAND_SCENARIOS if not scenario.eval_only]
        held_out = [scenario for scenario in NASH_DEMAND_SCENARIOS if scenario.eval_only]
        assert len(train_rows()) == len(train) * len(NASH_DEMAND_WINDFALLS)
        assert len(
            generate_prompt_rows(
                NASH_DEMAND_GAME_ID, GRADING_NASH_DEMAND_GROUP_MIX, split=SPLIT_EVAL
            )
        ) == len(held_out) * len(NASH_DEMAND_WINDFALLS)

    def test_the_schema_is_exactly_the_declared_columns(self) -> None:
        for split in SPLITS:
            for row in generate_prompt_rows(
                NASH_DEMAND_GAME_ID, GRADING_NASH_DEMAND_GROUP_MIX, split=split
            ):
                assert tuple(row.keys()) == ROW_COLUMNS

    def test_the_rows_carry_the_windfall_and_no_action_columns(self) -> None:
        for row in train_rows():
            assert row["windfall"] in NASH_DEMAND_WINDFALLS
            assert row["payoff_variant"] == nash_demand_variant(int(str(row["windfall"])))
            assert row["label_a"] == ""
            assert row["label_b"] == ""
            assert row["coop_label"] == ""
            assert row["endowment"] == 0
            for column in ("payoff_cc", "payoff_cd", "payoff_dc", "payoff_dd"):
                assert row[column] == 0.0

    def test_every_windfall_appears_for_every_frame(self) -> None:
        by_frame: dict[object, set[object]] = {}
        for row in train_rows():
            by_frame.setdefault(row["reskin_id"], set()).add(row["windfall"])
        assert by_frame
        for frame, windfalls in by_frame.items():
            assert windfalls == set(NASH_DEMAND_WINDFALLS), frame

    def test_the_two_gradings_render_byte_identical_prompts(self) -> None:
        """The pair's whole claim: only the grading rule differs, so nothing else may."""
        group = train_rows(GRADING_NASH_DEMAND_GROUP_MIX)
        own = train_rows(GRADING_NASH_DEMAND_SELF)
        assert [row["prompt"] for row in group] == [row["prompt"] for row in own]
        assert [row["prompt_id"] for row in group] == [row["prompt_id"] for row in own]

    def test_the_variant_name_cannot_be_confused_with_the_dictator_endowment(self) -> None:
        """Both games run the same three totals, so pooling them on the variant would be silent."""
        assert nash_demand_variant(100) == "windfall-100"
        assert nash_demand_variant(100) != "endowment-100"

    def test_train_and_eval_frames_are_disjoint(self) -> None:
        train = {row["reskin_id"] for row in train_rows()}
        held_out = {
            row["reskin_id"]
            for row in generate_prompt_rows(
                NASH_DEMAND_GAME_ID, GRADING_NASH_DEMAND_GROUP_MIX, split=SPLIT_EVAL
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
                NASH_DEMAND_GAME_ID, GRADING_NASH_DEMAND_GROUP_MIX, split=split
            )
        ]
        assert len(ids) == len(set(ids))


class TestTheRenderedPrompt:
    def test_the_prompt_is_exactly_its_four_sections_in_order(self) -> None:
        """A structural golden: catches a reordering, a dropped section, or an extra one."""
        scenario = NASH_DEMAND_SCENARIOS[0]
        at_100 = spec()
        expected = "\n\n".join(
            [
                scenario.frame,
                NASH_DEMAND_MECHANICS.format(windfall=100, resource=scenario.resource),
                f"About the other side: {TWIN_COUNTERPART_CLAUSE}",
                NASH_DEMAND_INSTRUCTION.format(windfall=100),
            ]
        )
        assert render_nash_demand_prompt(at_100, scenario) == expected

    def test_the_three_windfalls_differ_only_in_their_numbers(self) -> None:
        """The counterbalance golden's analogue: this game has no labels to swap, only a scale."""
        scenario = NASH_DEMAND_SCENARIOS[0]
        numbers = re.compile(r"\d[\d.]*")
        masked = {
            numbers.sub("#", render_nash_demand_prompt(spec(windfall), scenario))
            for windfall in NASH_DEMAND_WINDFALLS
        }
        assert len(masked) == 1

    @pytest.mark.parametrize("windfall", NASH_DEMAND_WINDFALLS)
    def test_the_prompt_states_the_total_and_the_answer_range(self, windfall: int) -> None:
        prompt = render_nash_demand_prompt(spec(windfall), NASH_DEMAND_SCENARIOS[0])
        assert f"comes to {windfall} " in prompt
        assert f"come to {windfall} or less" in prompt
        assert f"between 0 and {windfall}" in prompt
        assert "<claim>N</claim>" in prompt

    def test_the_prompt_says_an_unclaimed_remainder_goes_nowhere(self) -> None:
        """Load-bearing: without it a small figure looks free and the interior optimum vanishes."""
        prompt = render_nash_demand_prompt(spec(), NASH_DEMAND_SCENARIOS[0])
        assert "leave unclaimed goes to neither side" in prompt

    def test_every_frame_renders_at_every_windfall_and_passes_the_guard(self) -> None:
        for scenario in NASH_DEMAND_SCENARIOS:
            for windfall in NASH_DEMAND_WINDFALLS:
                assert_no_loaded_vocabulary(render_nash_demand_prompt(spec(windfall), scenario))

    def test_no_prompt_embeds_its_own_identifiers(self) -> None:
        for split in SPLITS:
            for row in generate_prompt_rows(
                NASH_DEMAND_GAME_ID, GRADING_NASH_DEMAND_GROUP_MIX, split=split
            ):
                prompt = str(row["prompt"])
                assert str(row["game_id"]) not in prompt
                assert str(row["reskin_id"]) not in prompt
                assert str(row["payoff_variant"]) not in prompt

    def test_a_swapped_print_order_is_refused_rather_than_ignored(self) -> None:
        """It prints no labels, so a swapped build would be canonical rows under a wrong column."""
        with pytest.raises(ValueError, match="no action labels"):
            generate_prompt_rows(
                NASH_DEMAND_GAME_ID,
                GRADING_NASH_DEMAND_GROUP_MIX,
                split=SPLIT_TRAIN,
                label_print_order="swapped",
            )


class TestTheFramesSayNothingLoaded:
    """The roster polices its own construct vocabulary, since the global regex cannot be tightened."""

    def test_the_construct_check_can_fail(self) -> None:
        """The negative control: a check nobody has watched fail is not yet a check."""
        assert _CONSTRUCT_RE.search("Each side states a fair share of the total.") is not None
        assert _CONSTRUCT_RE.search("This is the demand stage.") is not None

    def test_ordinary_frame_prose_passes_the_construct_check(self) -> None:
        assert _CONSTRUCT_RE.search("Each side lodges a figure with the berth master.") is None

    def test_no_frame_uses_construct_vocabulary(self) -> None:
        for scenario in NASH_DEMAND_SCENARIOS:
            match = _CONSTRUCT_RE.search(scenario.frame)
            assert match is None, (scenario.scenario_id, match)

    def test_no_rendered_prompt_uses_construct_vocabulary(self) -> None:
        for scenario in NASH_DEMAND_SCENARIOS:
            prompt = render_nash_demand_prompt(spec(), scenario)
            match = _CONSTRUCT_RE.search(prompt)
            assert match is None, (scenario.scenario_id, match)

    def test_every_frame_is_clean_under_the_global_guard(self) -> None:
        for scenario in NASH_DEMAND_SCENARIOS:
            assert_no_loaded_vocabulary(scenario.frame)
            assert_no_loaded_vocabulary(scenario.scenario_id)
            assert_no_loaded_vocabulary(scenario.resource)

    def test_every_frame_sets_up_two_sides_deciding_at_once(self) -> None:
        """A frame authored as a one-sided allocation would render a different game entirely.

        The mechanics paragraph states what two figures do to each other and never who is filing
        them, so the frame is the only place the second side and the simultaneity exist. A frame
        missing either would read as the unilateral split, whose optimum is the opposite corner.
        """
        simultaneity = ("same", "together", "at once", "before")
        for scenario in NASH_DEMAND_SCENARIOS:
            frame = scenario.frame.casefold()
            assert "two " in frame or "both " in frame, scenario.scenario_id
            assert "each" in frame, scenario.scenario_id
            assert any(phrase in frame for phrase in simultaneity), scenario.scenario_id

    def test_a_one_sided_frame_would_fail_that_check(self) -> None:
        """The negative control for it: the unilateral-split roster is what it must reject."""
        from games.prompts import DICTATOR_SCENARIOS  # noqa: PLC0415 -- the contrast case only

        one_sided = DICTATOR_SCENARIOS[0].frame.casefold()
        assert "two " not in one_sided or "each" not in one_sided

    def test_the_roster_ids_and_resources_are_distinct(self) -> None:
        ids = [scenario.scenario_id for scenario in NASH_DEMAND_SCENARIOS]
        resources = [scenario.resource for scenario in NASH_DEMAND_SCENARIOS]
        assert len(ids) == len(set(ids))
        assert len(resources) == len(set(resources))

    def test_an_empty_frame_or_resource_is_refused(self) -> None:
        from games.prompts import NashDemandScenario  # noqa: PLC0415 -- constructed only here

        with pytest.raises(ValueError, match="empty frame"):
            NashDemandScenario(scenario_id="blank", frame="   ", resource="hours")
        with pytest.raises(ValueError, match="empty resource"):
            NashDemandScenario(scenario_id="blank", frame="a frame", resource=" ")


class TestTheRegistryRoundTrip:
    ARM_NAMES: tuple[str, str] = ("nash-demand-group", "nash-demand-self")

    def test_both_arms_are_registered_and_valid(self) -> None:
        for name in self.ARM_NAMES:
            arm = ARMS[name]
            assert arm.game_id == NASH_DEMAND_GAME_ID
            assert arm.grading in NASH_DEMAND_GRADINGS
            assert arm.notes
        validate_arms({name: ARMS[name] for name in self.ARM_NAMES})

    def test_the_pair_differs_only_in_the_grading_rule(self) -> None:
        group, own = (ARMS[name] for name in self.ARM_NAMES)
        assert group.game_id == own.game_id
        assert group.payoff_variants == own.payoff_variants == ()
        assert group.grading != own.grading

    def test_each_arm_carries_all_three_windfalls(self) -> None:
        for name in self.ARM_NAMES:
            assert arm_payoff_variants(ARMS[name]) == {
                nash_demand_variant(windfall) for windfall in NASH_DEMAND_WINDFALLS
            }

    def test_the_gradings_stay_out_of_the_binary_action_set(self) -> None:
        """Adding one there would ask selection for a cooperation rate that does not exist."""
        assert not NASH_DEMAND_GRADINGS & ONE_SHOT_ACTION_GRADINGS

    def test_both_gradings_require_the_claim_metrics_at_read_back(self) -> None:
        for grading in sorted(NASH_DEMAND_GRADINGS):
            required = required_metrics_for(grading)
            assert "mean_claim_fraction" in required
            assert "coop_rate" not in required

    def test_each_grading_requires_its_own_overreach_metric_and_not_the_other_one(self) -> None:
        """One shared key would let either arm pass the read-back gate on the other's quantity."""
        group_mix = required_metrics_for(GRADING_NASH_DEMAND_GROUP_MIX)
        self_graded = required_metrics_for(GRADING_NASH_DEMAND_SELF)
        assert "crash_rate" in group_mix
        assert "overclaim_rate" not in group_mix
        assert "overclaim_rate" in self_graded
        assert "crash_rate" not in self_graded


class TestTheDeadArmGuardCoversThisGrading:
    """Generalised from one grading name to a per-grading span, per the wave-2 design brief."""

    def test_the_real_span_is_the_half_the_game_can_reach(self) -> None:
        assert nash_demand_reward_span(ARMS["nash-demand-self"]) == pytest.approx(0.5)

    def test_a_zero_span_arm_is_refused_with_the_mechanism_spelled_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sabotage kept as a test: a reward that stopped varying with the claim must be refused."""
        flattened = games_arms._SelfConsistentGrading(  # pyright: ignore[reportPrivateUsage]
            span=lambda _arm: 0.0,
            remedy="flattened for the test",
        )
        monkeypatch.setitem(
            games_arms.SELF_CONSISTENT_GRADINGS, GRADING_NASH_DEMAND_SELF, flattened
        )
        with pytest.raises(ValueError, match="advantage is zero"):
            validate_arms({"nash-demand-self": ARMS["nash-demand-self"]})

    def test_the_matrix_span_would_read_zero_for_this_game(self) -> None:
        """Why the guard had to be generalised rather than pointed at one grading name.

        Every row of a graded game carries payoff_cc == payoff_dd == 0.0, so the 2x2 span reads 0.0
        however healthy the arm is. A guard keyed on "any self-graded arm, measured by the cells"
        would refuse this arm while checking nothing about its actual reward.
        """
        assert games_arms.self_grading_reward_span(ARMS["nash-demand-self"]) == pytest.approx(0.0)

    def test_the_constant_sum_matrix_arm_is_still_refused(self) -> None:
        """The original dead arm the guard was written for, still caught after generalising."""
        with pytest.raises(ValueError, match="constant-sum"):
            validate_arms(
                {
                    "fixed-pie-pd-self": GameArm(
                        game_id="fixed-pie-pd", grading="self", notes="the dead arm from c96c463"
                    )
                }
            )


class TestSelectionOnASyntheticSweep:
    """The spread floor is the only filter these rows meet, so it is the one that must have teeth."""

    def sweep(self, script: dict[str, list[str]], *, samples: int = 4) -> Any:
        rows = train_rows()[:3]
        for index, row in enumerate(rows):
            row["prompt_id"] = f"claim-{index}"
        backend = ScriptedBackend(
            {str(row["prompt"]): script[str(row["prompt_id"])] for row in rows}
        )
        return sweep_prompts(
            backend,
            rows,
            samples_per_prompt=samples,
            prefilled_think=False,
        )

    def test_a_split_prompt_is_kept_and_a_unanimous_one_is_dropped(self) -> None:
        records = self.sweep(
            {
                "claim-0": [claim_completion(claim) for claim in (30, 50, 50, 70)],
                "claim-1": [claim_completion(50)] * 4,
                "claim-2": [claim_completion(claim) for claim in (48, 50, 52, 50)],
            }
        )
        verdicts = {verdict.prompt_id: verdict for verdict in judge_prompts(records)}
        assert verdicts["claim-0"].keep
        assert verdicts["claim-0"].reason == DropReason.KEPT_MIXED
        assert not verdicts["claim-1"].keep
        assert verdicts["claim-1"].reason == DropReason.SCORE_SPREAD_BELOW_MIN
        # 48/50/52/50 spreads 0.014 of the windfall, under the 0.05 floor: real but too small to
        # train on, which is the case the floor exists to catch rather than unanimity alone.
        assert verdicts["claim-2"].score_std < DEFAULT_MIN_SPLIT_STD
        assert not verdicts["claim-2"].keep

    def test_the_selection_score_is_the_claimed_fraction(self) -> None:
        records = self.sweep(
            {
                "claim-0": [claim_completion(claim) for claim in (30, 50, 50, 70)],
                "claim-1": [claim_completion(50)] * 4,
                "claim-2": [claim_completion(50)] * 4,
            }
        )
        first = next(record for record in records if record.prompt_id == "claim-0")
        windfall = int(str(first.row["windfall"]))
        assert first.selection_scores == pytest.approx([30 / windfall, 0.5, 0.5, 70 / windfall])
        assert [sample.claim for sample in first.samples] == [30, 50, 50, 70]

    def test_a_mostly_unparseable_prompt_is_dropped_for_that_reason(self) -> None:
        records = self.sweep(
            {
                "claim-0": ["no figure", "no figure", "no figure", claim_completion(50)],
                "claim-1": [claim_completion(50)] * 4,
                "claim-2": [claim_completion(50)] * 4,
            }
        )
        verdicts = {verdict.prompt_id: verdict for verdict in judge_prompts(records)}
        assert verdicts["claim-0"].reason == DropReason.TOO_FEW_PARSEABLE

    def test_an_out_of_range_figure_counts_as_a_parse_failure_in_the_sweep(self) -> None:
        records = self.sweep(
            {
                "claim-0": [
                    claim_completion(30),
                    "<claim>9999</claim>",
                    claim_completion(50),
                    claim_completion(70),
                ],
                "claim-1": [claim_completion(50)] * 4,
                "claim-2": [claim_completion(50)] * 4,
            }
        )
        first = next(record for record in records if record.prompt_id == "claim-0")
        assert first.n_parse_failures == 1

    def test_the_rows_are_singletons_with_no_counterbalanced_partner(self) -> None:
        for row in train_rows():
            assert not is_counterbalanced(row)

    def test_the_sweep_reports_no_cooperation_rate(self) -> None:
        records = self.sweep(
            {
                "claim-0": [claim_completion(30)] * 4,
                "claim-1": [claim_completion(50)] * 4,
                "claim-2": [claim_completion(70)] * 4,
            }
        )
        assert all(record.coop_fraction is None for record in records)
        assert all(record.coop_count is None for record in records)


class TestTheEvalBatteryReadsTheClaim:
    def test_the_game_is_in_the_render_grading_map(self) -> None:
        """Its absence would be caught at import; this pins which grading the battery renders."""
        assert EVAL_RENDER_GRADING_BY_GAME[NASH_DEMAND_GAME_ID] in NASH_DEMAND_GRADINGS

    def test_the_battery_can_be_configured_for_this_game_alone(self) -> None:
        config = EvalConfig(games=(NASH_DEMAND_GAME_ID,), trained_game_ids=(NASH_DEMAND_GAME_ID,))
        assert config.games == (NASH_DEMAND_GAME_ID,)

    def test_the_behaviour_field_is_the_claim_fraction_in_both_readouts(self) -> None:
        """Without this the tables render empty, and the readout recomputes itself in silence."""
        assert behaviour_field(NASH_DEMAND_GAME_ID) == CLAIM_FIELD
        assert readout._behaviour_field(NASH_DEMAND_GAME_ID) == "claim_fraction"  # pyright: ignore[reportPrivateUsage]

    def test_a_record_carries_the_claim_and_its_fraction(self) -> None:
        from games.evals import _game_record  # noqa: PLC0415 -- private, tested at its own seam

        row = row_at(60)
        record = _game_record(
            NASH_DEMAND_GAME_ID,
            row,
            claim_completion(30),
            prefilled_think=False,
            trained_game=True,
            eval_only_game=False,
            sample_index=0,
        )
        assert record["record"] == SECTION_GAME_BEHAVIOR
        assert record["claim"] == 30
        assert record["claim_fraction"] == pytest.approx(0.5)
        assert record["parsed"] is True

    def test_an_unparseable_completion_records_a_parse_failure_not_a_zero(self) -> None:
        from games.evals import _game_record  # noqa: PLC0415 -- private, tested at its own seam

        record = _game_record(
            NASH_DEMAND_GAME_ID,
            row_at(60),
            "I will not name a figure.",
            prefilled_think=False,
            trained_game=True,
            eval_only_game=False,
            sample_index=0,
        )
        assert record["claim"] is None
        assert record["claim_fraction"] is None
        assert record["parsed"] is False
