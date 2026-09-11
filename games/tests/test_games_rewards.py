"""Pin the reward arithmetic and the three structural guards, all against hand math.

Offline and CPU-only: no model, no trainer, no GPU. The reward function is called exactly the
way installed TRL 1.10 calls it -- every dataset column as a parallel list, plus injected
`log_metric` and `log_extra` callables -- with recorder stubs standing in for the trainer's.

Every expected reward below is written as the arithmetic that produces it, at the normalised
twin-PD cells CC=0.6, CD=0.0, DC=1.0, DD=0.2. Two classes carry the weight:

:class:`TestContiguityGuard` is the sabotage item from the plan, kept as a permanent test.
Shuffled prompt_ids do not crash a run -- they quietly grade every completion against the wrong
opponent population -- so the guard must raise.

:class:`TestPrefilledThinkMustMatchTheTemplate` covers the trap the model ladder sets: with
`prefilled_think=True` a completion carrying no thinking markers is truncated reasoning with no
answer, and with `False` it is the whole answer. Set it wrong for a model and every rollout in
the run reads as a parse failure.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from games import arm_sequence
from games import arms as games_arms
from games.arms import (
    GameArm,
    assert_every_grading_is_classified,
    span_check_exemption,
    validate_arms,
)
from games.format_rubric import (
    COMPONENT_NAMES,
    COMPONENT_NOTHING_AFTER_TAG,
    COMPONENT_SINGLE_TAG,
    FORMAT_COMPONENTS,
    MAX_FORMAT_REWARD,
)
from games.payoffs import (
    COOPERATE,
    DEFECT,
    STATED_RETURN_UNSET,
    THRESHOLD_GOODS_MAX_PRIZE,
    TRUST_MAX_STATED_RETURN_FRACTION,
    MatrixGameSpec,
    MinEffortSpec,
    NashDemandSpec,
    OpponentRule,
    ThresholdGoodsSpec,
    TrustSpec,
    care_reward_spread,
    expected_care_payoff,
    max_iterated_return,
    max_min_effort_match_return,
    min_effort_group_reward,
    nash_demand_self_reward,
    threshold_goods_reward,
    trust_care_corner_rewards,
    worst_iterated_return,
    worst_min_effort_match_return,
)
from games.prompts import generate_prompt_rows
from games.regrade_corpus import regrade_rows
from games.rewards import (
    CARE_GRADING_PREFIX,
    DEFAULT_PARSE_PENALTY,
    FRAMING_ID_UNSET,
    GRADING_FORMAT_ONLY,
    GRADING_GROUP_MIX,
    GRADING_ITERATED_RETURN,
    GRADING_JOINT_WELFARE_GROUP_MIX,
    GRADING_KEEP_FRACTION,
    GRADING_LEVEL_MATCH_RETURN,
    GRADING_MIN_EFFORT_GROUP_MIX,
    GRADING_NASH_DEMAND_GROUP_MIX,
    GRADING_NASH_DEMAND_SELF,
    GRADING_OTHER_PAYOFF_GROUP_MIX,
    GRADING_SELF,
    GRADING_THRESHOLD_GOODS_GROUP_MIX,
    GRADING_THRESHOLD_GOODS_SELF,
    GRADING_TRUSTOR_PAYOFF_SELF_RULE,
    GRADING_TRUSTOR_PAYOFF_STATED_RULE,
    GRADING_VS_FIXED_MIX,
    GRADING_VS_STATED_MATCH,
    GRADINGS,
    PARSE_PENALTY_CONSTANT,
    PARSE_PENALTY_MARGIN_BELOW_WORSE,
    REQUIRED_REWARD_COLUMNS,
    STATED_RETURN_FRACTION_COLUMN,
    UNINFORMATIVE_COOP_PRIOR,
    _log_parse_price_metrics,
    _rows_from_columns,
    _Scored,
    care_alpha_of,
    care_grading,
    grading_cli_value,
    is_grading,
    make_game_reward,
    unknown_grading_message,
)
from games.track_record_corpus import SOURCE_GAME_ID, build_track_record_v2_rows

if TYPE_CHECKING:
    from collections.abc import Callable

COOPERATIVE_LABEL = "HOLD"
DEFECTING_LABEL = "SLASH"

PAYOFF_CC = 0.6
PAYOFF_CD = 0.0
PAYOFF_DC = 1.0
PAYOFF_DD = 0.2

COOPERATE_COMPLETION = f"<action>{COOPERATIVE_LABEL}</action>"
DEFECT_COMPLETION = f"<action>{DEFECTING_LABEL}</action>"
UNPARSEABLE_COMPLETION = "I would rather not say."

# Reward callables are stateless closures, so tests share these rather than rebuilding them.
GROUPS_OF_TWO = make_game_reward(2, prefilled_think=False)
WITH_PREFILL = make_game_reward(2, prefilled_think=True)
WITHOUT_PREFILL = make_game_reward(2, prefilled_think=False)
MARGIN_BELOW_WORSE_GROUPS_OF_TWO = make_game_reward(
    2, prefilled_think=False, parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE
)


def make_row(**overrides: Any) -> dict[str, Any]:
    """One dataset row with twin-PD defaults; overrides name the column being varied."""
    row: dict[str, Any] = {
        "prompt_id": "prompt-0",
        "grading": GRADING_GROUP_MIX,
        "game_id": "twin-pd-temptation-2",
        "payoff_cc": PAYOFF_CC,
        "payoff_cd": PAYOFF_CD,
        "payoff_dc": PAYOFF_DC,
        "payoff_dd": PAYOFF_DD,
        "label_a": COOPERATIVE_LABEL,
        "label_b": DEFECTING_LABEL,
        "coop_label": COOPERATIVE_LABEL,
        "endowment": 10,
        "windfall": 0,
        "opp_coop_prob": -1.0,
        "opponent_rule": "",
        "n_rounds": 0,
        "team_size": 0,
        "contribution_threshold": 0,
        "prize": 0,
        "transfer_multiplier": 0.0,
        "stated_return_fraction": -1.0,
        "stated_match_prob": -1.0,
        "n_levels": 0,
        "benefit_per_level": 0.0,
        "cost_per_level": 0.0,
    }
    row.update(overrides)
    return row


def as_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
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
    rows: list[dict[str, Any]],
    completions: list[str],
    recorder: Recorder,
    reward: Callable[..., list[float]] | None = None,
) -> list[float]:
    """Call the reward function the way TRL does: columns as parallel lists, loggers injected.

    Without an explicit `reward` the whole batch is one group with thinking markers optional,
    which is what most cases want; the rest build their own factory to vary the group size,
    `prefilled_think`, or `leave_one_out`.
    """
    resolved = (
        reward if reward is not None else make_game_reward(len(completions), prefilled_think=False)
    )
    return resolved(
        completions=completions,
        log_metric=recorder.log_metric,
        log_extra=recorder.log_extra,
        **as_columns(rows),
    )


class TestGroupMixHandMath:
    """A group of four whose parsed actions are C, C, D with one unparseable completion."""

    COMPLETIONS: ClassVar[list[str]] = [
        COOPERATE_COMPLETION,
        COOPERATE_COMPLETION,
        DEFECT_COMPLETION,
        UNPARSEABLE_COMPLETION,
    ]

    def test_the_opponent_mix_excludes_the_unparseable_completion(self, recorder: Recorder) -> None:
        """The mix is 2/3, not 2/4: an unparseable completion is penalised, not a defector."""
        rewards = score([make_row() for _ in range(4)], self.COMPLETIONS, recorder)
        cooperator = (2 / 3) * PAYOFF_CC + (1 / 3) * PAYOFF_CD
        defector = (2 / 3) * PAYOFF_DC + (1 / 3) * PAYOFF_DD
        assert rewards == pytest.approx([cooperator, cooperator, defector, DEFAULT_PARSE_PENALTY])
        assert cooperator == pytest.approx(0.4)
        assert defector == pytest.approx(11 / 15)

    def test_leave_one_out_removes_a_completion_from_its_own_opponent_mix(
        self, recorder: Recorder
    ) -> None:
        """Each cooperator now faces one C and one D (p=0.5); the defector faces two Cs (p=1)."""
        rewards = score(
            [make_row() for _ in range(4)],
            self.COMPLETIONS,
            recorder,
            make_game_reward(4, prefilled_think=False, leave_one_out=True),
        )
        cooperator = 0.5 * PAYOFF_CC + 0.5 * PAYOFF_CD
        defector = 1.0 * PAYOFF_DC
        assert rewards == pytest.approx([cooperator, cooperator, defector, DEFAULT_PARSE_PENALTY])
        assert cooperator == pytest.approx(0.3)
        assert defector == pytest.approx(1.0)
        assert recorder.metrics["leave_one_out_prior_rate"] == pytest.approx(0.0)

    def test_metrics_and_per_completion_columns_are_logged(self, recorder: Recorder) -> None:
        rewards = score([make_row() for _ in range(4)], self.COMPLETIONS, recorder)
        assert recorder.metrics["parse_failure_rate"] == pytest.approx(0.25)
        assert recorder.metrics["coop_rate"] == pytest.approx(2 / 3)
        assert recorder.metrics["truncated_thinking_rate"] == pytest.approx(0.0)
        assert "mean_keep_fraction" not in recorder.metrics
        assert "leave_one_out_prior_rate" not in recorder.metrics
        assert recorder.extra["parsed_action"] == ["C", "C", "D", ""]
        assert recorder.extra["game_reward"] == pytest.approx(rewards)
        assert recorder.extra["truncated_thinking"] == [False, False, False, False]

    def test_a_group_where_nothing_parsed_is_all_penalty_but_does_not_raise(
        self, recorder: Recorder
    ) -> None:
        rows = [make_row(prompt_id="prompt-0") for _ in range(2)]
        rows += [make_row(prompt_id="prompt-1") for _ in range(2)]
        completions = [UNPARSEABLE_COMPLETION, UNPARSEABLE_COMPLETION]
        completions += [COOPERATE_COMPLETION, DEFECT_COMPLETION]
        rewards = score(rows, completions, recorder, GROUPS_OF_TWO)
        assert rewards[:2] == pytest.approx([DEFAULT_PARSE_PENALTY, DEFAULT_PARSE_PENALTY])
        assert rewards[2] == pytest.approx(0.5 * PAYOFF_CC + 0.5 * PAYOFF_CD)
        assert rewards[3] == pytest.approx(0.5 * PAYOFF_DC + 0.5 * PAYOFF_DD)

    def test_the_uninformative_prior_covers_a_lone_parsed_completion(
        self, recorder: Recorder
    ) -> None:
        """With leave-one-out and one parse in the group there is no opponent left to observe."""
        rows = [make_row() for _ in range(2)]
        completions = [COOPERATE_COMPLETION, UNPARSEABLE_COMPLETION]
        rewards = score(
            rows,
            completions,
            recorder,
            make_game_reward(2, prefilled_think=False, leave_one_out=True),
        )
        assert rewards == pytest.approx([0.5 * PAYOFF_CC + 0.5 * PAYOFF_CD, DEFAULT_PARSE_PENALTY])
        assert recorder.metrics["leave_one_out_prior_rate"] == pytest.approx(0.5)


class TestSelfGrading:
    def test_the_opponent_plays_what_i_play(self, recorder: Recorder) -> None:
        rows = [make_row(grading=GRADING_SELF) for _ in range(3)]
        completions = [COOPERATE_COMPLETION, DEFECT_COMPLETION, UNPARSEABLE_COMPLETION]
        rewards = score(rows, completions, recorder)
        assert rewards == pytest.approx([PAYOFF_CC, PAYOFF_DD, DEFAULT_PARSE_PENALTY])

    def test_cooperation_beats_defection_here_unlike_group_mix(self, recorder: Recorder) -> None:
        """The novel contrast arm: grading against your own action reverses the pressure."""
        rows = [make_row(grading=GRADING_SELF) for _ in range(2)]
        rewards = score(rows, [COOPERATE_COMPLETION, DEFECT_COMPLETION], recorder)
        assert rewards[0] > rewards[1]


class TestJointWelfareGroupMix:
    """The wave-3 ladder's pie grading: each row pays the two sides' MEAN payoff against the mix.

    Same C, C, D group (one unparseable) as the group-mix hand math, so the two arithmetics can be
    read side by side: the mix is 2/3 there and here, and only the recipient of the payoff moves.
    """

    COMPLETIONS: ClassVar[list[str]] = [
        COOPERATE_COMPLETION,
        COOPERATE_COMPLETION,
        DEFECT_COMPLETION,
        UNPARSEABLE_COMPLETION,
    ]

    def test_the_reward_is_the_expected_mean_of_both_sides(self, recorder: Recorder) -> None:
        rows = [make_row(grading=GRADING_JOINT_WELFARE_GROUP_MIX) for _ in range(4)]
        rewards = score(rows, self.COMPLETIONS, recorder)
        off_diagonal = (PAYOFF_CD + PAYOFF_DC) / 2
        cooperator = (2 / 3) * PAYOFF_CC + (1 / 3) * off_diagonal
        defector = (2 / 3) * off_diagonal + (1 / 3) * PAYOFF_DD
        assert rewards == pytest.approx([cooperator, cooperator, defector, DEFAULT_PARSE_PENALTY])
        assert cooperator == pytest.approx(0.4 + 1 / 6)
        assert defector == pytest.approx(0.4)
        assert recorder.metrics["coop_rate"] == pytest.approx(2 / 3)

    def test_cooperation_wins_at_temptation_2_even_in_an_all_defect_mix(
        self, recorder: Recorder
    ) -> None:
        """At these cells the joint gap is 0.3 - 0.2p > 0 everywhere, so C wins at p as low as 1/4."""
        rows = [make_row(grading=GRADING_JOINT_WELFARE_GROUP_MIX) for _ in range(4)]
        completions = [COOPERATE_COMPLETION] + [DEFECT_COMPLETION] * 3
        rewards = score(rows, completions, recorder)
        assert rewards[0] > rewards[1]

    def test_the_temptation_10_gap_changes_sign_across_the_interior_mix(
        self, recorder: Recorder
    ) -> None:
        """At temptation-10 the pie is bigger when one side defects on a cooperator (DC+CD > 2*CC),
        so the joint gap 0.4231 - 0.6923p crosses zero at p* = 0.611: cooperators win below it and
        defectors above it -- the interior attractor the arm's numeric prediction is scored against.
        """
        temptation_10 = {
            "payoff_cc": 3 / 13,
            "payoff_cd": 0.0,
            "payoff_dc": 1.0,
            "payoff_dd": 1 / 13,
        }
        rows = [
            make_row(grading=GRADING_JOINT_WELFARE_GROUP_MIX, **temptation_10) for _ in range(4)
        ]
        below = score(rows, [COOPERATE_COMPLETION] + [DEFECT_COMPLETION] * 3, recorder)
        assert below[0] > below[1], "at p=1/4 < p* the cooperator must out-earn the defector"
        above = score(rows, [COOPERATE_COMPLETION] * 3 + [DEFECT_COMPLETION], recorder)
        assert above[3] > above[0], "at p=3/4 > p* the defector must out-earn the cooperators"


class TestOtherPayoffGroupMix:
    """The ladder's selfless grading: each row pays the COUNTERPART's expected payoff.

    My action a is what the counterpart's payoff is computed against: a counterpart playing b
    earns payoff(b, a), so cooperating hands them their best column whatever they play, and in any
    PD (CC > CD and DC > DD) cooperation is strictly dominant under this grading.
    """

    COMPLETIONS: ClassVar[list[str]] = [
        COOPERATE_COMPLETION,
        COOPERATE_COMPLETION,
        DEFECT_COMPLETION,
        UNPARSEABLE_COMPLETION,
    ]

    def test_the_reward_is_the_counterparts_expected_payoff(self, recorder: Recorder) -> None:
        rows = [make_row(grading=GRADING_OTHER_PAYOFF_GROUP_MIX) for _ in range(4)]
        rewards = score(rows, self.COMPLETIONS, recorder)
        cooperator = (2 / 3) * PAYOFF_CC + (1 / 3) * PAYOFF_DC
        defector = (2 / 3) * PAYOFF_CD + (1 / 3) * PAYOFF_DD
        assert rewards == pytest.approx([cooperator, cooperator, defector, DEFAULT_PARSE_PENALTY])
        assert cooperator == pytest.approx(0.4 + 1 / 3)
        assert defector == pytest.approx(1 / 15)
        assert recorder.metrics["coop_rate"] == pytest.approx(2 / 3)

    def test_cooperation_is_strictly_dominant_at_both_mix_extremes(
        self, recorder: Recorder
    ) -> None:
        rows = [make_row(grading=GRADING_OTHER_PAYOFF_GROUP_MIX) for _ in range(4)]
        mostly_defect = score(rows, [COOPERATE_COMPLETION] + [DEFECT_COMPLETION] * 3, recorder)
        assert mostly_defect[0] > mostly_defect[1]
        mostly_cooperate = score(rows, [COOPERATE_COMPLETION] * 3 + [DEFECT_COMPLETION], recorder)
        assert mostly_cooperate[0] > mostly_cooperate[3]

    def test_unparseable_completions_are_excluded_from_the_mix_it_grades_against(
        self, recorder: Recorder
    ) -> None:
        """The mix is 2/3 (not 2/4), exactly as under own-payoff group-mix grading."""
        rows = [make_row(grading=GRADING_OTHER_PAYOFF_GROUP_MIX) for _ in range(4)]
        rewards = score(rows, self.COMPLETIONS, recorder)
        assert rewards[0] != pytest.approx(0.5 * PAYOFF_CC + 0.5 * PAYOFF_DC)


class TestKeepFraction:
    def test_the_reward_is_the_fraction_kept(self, recorder: Recorder) -> None:
        rows = [make_row(grading=GRADING_KEEP_FRACTION, endowment=10) for _ in range(4)]
        completions = ["<keep>7</keep>", "<keep>0</keep>", "<keep>10</keep>", "<keep>11</keep>"]
        rewards = score(rows, completions, recorder)
        assert rewards == pytest.approx([0.7, 0.0, 1.0, DEFAULT_PARSE_PENALTY])

    def test_it_reports_a_keep_fraction_and_no_cooperation_rate(self, recorder: Recorder) -> None:
        """Cooperation is undefined without an opponent, so coop_rate must stay absent."""
        rows = [make_row(grading=GRADING_KEEP_FRACTION, endowment=10) for _ in range(2)]
        score(rows, ["<keep>7</keep>", "<keep>3</keep>"], recorder)
        assert recorder.metrics["mean_keep_fraction"] == pytest.approx(0.5)
        assert "coop_rate" not in recorder.metrics


class TestVsFixedMix:
    def test_hand_math_at_p_of_zero_point_eight(self, recorder: Recorder) -> None:
        rows = [make_row(grading=GRADING_VS_FIXED_MIX, opp_coop_prob=0.8) for _ in range(2)]
        rewards = score(rows, [COOPERATE_COMPLETION, DEFECT_COMPLETION], recorder)
        assert rewards == pytest.approx([0.8 * PAYOFF_CC, 0.8 * PAYOFF_DC + 0.2 * PAYOFF_DD])
        assert rewards == pytest.approx([0.48, 0.84])

    def test_the_group_does_not_influence_the_opponent_mix(self, recorder: Recorder) -> None:
        """The frozen opponent is static by construction, so the group must not shift p."""
        rows = [make_row(grading=GRADING_VS_FIXED_MIX, opp_coop_prob=0.8) for _ in range(4)]
        completions = [COOPERATE_COMPLETION] + [DEFECT_COMPLETION] * 3
        rewards = score(rows, completions, recorder)
        assert rewards[0] == pytest.approx(0.8 * PAYOFF_CC)

    def test_an_unsampled_opponent_raises(self, recorder: Recorder) -> None:
        """opp_coop_prob=-1 is the corpus's "never sampled" marker, not a probability."""
        rows = [make_row(grading=GRADING_VS_FIXED_MIX, opp_coop_prob=-1.0) for _ in range(2)]
        with pytest.raises(RuntimeError, match="outside"):
            score(rows, [COOPERATE_COMPLETION, DEFECT_COMPLETION], recorder)

    def test_a_probability_above_one_also_raises(self, recorder: Recorder) -> None:
        rows = [make_row(grading=GRADING_VS_FIXED_MIX, opp_coop_prob=1.5) for _ in range(2)]
        with pytest.raises(RuntimeError, match="outside"):
            score(rows, [COOPERATE_COMPLETION, DEFECT_COMPLETION], recorder)

    def test_leave_one_out_cannot_reach_a_frozen_opponent(self, recorder: Recorder) -> None:
        """A frozen opponent has no group to leave anyone out of, so the flag must be inert here.

        `_score_group` threaded `leave_one_out` into this branch alongside a concrete probability,
        which the scorer checks first -- so the argument was dead, and a reader chasing
        `leave_one_out_prior_rate` on a vs-frozen arm would look in the wrong place. The
        invariant is worth pinning either way: these rewards are a property of the cached opponent
        alone.
        """
        rows = [make_row(grading=GRADING_VS_FIXED_MIX, opp_coop_prob=0.8) for _ in range(4)]
        completions = [COOPERATE_COMPLETION] + [DEFECT_COMPLETION] * 3
        plain = score(rows, completions, recorder, make_game_reward(4, prefilled_think=False))
        left_out = score(
            rows,
            completions,
            recorder,
            make_game_reward(4, prefilled_think=False, leave_one_out=True),
        )
        assert plain == pytest.approx(left_out)
        assert "leave_one_out_prior_rate" not in recorder.metrics or recorder.metrics[
            "leave_one_out_prior_rate"
        ] == pytest.approx(0.0)


class TestTheGroupSizeFloorHasOneValue:
    """Two names for the GRPO minimum group size, in one package, cannot be allowed to drift.

    `games.rewards` is imported by the cheap end of this package -- the arm registry pulls it in for
    the grading vocabulary -- and `grpo.throughput`, where `MIN_GRPO_GROUP_SIZE` lives, imports
    torch, transformers, trl, datasets and peft. Importing it here to share the constant would turn
    a 50 ms import into a 10 s one on the path whose selling point is being cheap, so the number is
    stated once in each place and pinned equal here instead. TRL enforces the same floor a third
    time, in `GRPOConfig.__post_init__`.
    """

    def test_the_rewards_floor_matches_the_throughput_one(self) -> None:
        from games.payoffs import MIN_GROUP_FOR_A_MIX  # noqa: PLC0415
        from games.rewards import MIN_GENERATIONS  # noqa: PLC0415
        from grpo.throughput import MIN_GRPO_GROUP_SIZE  # noqa: PLC0415

        assert MIN_GENERATIONS == MIN_GRPO_GROUP_SIZE == MIN_GROUP_FOR_A_MIX

    def test_a_group_below_the_floor_is_refused_by_the_reward_factory(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            make_game_reward(1, prefilled_think=False)


class TestIteratedReturn:
    """Five rounds against tit-for-tat, whose brute-forced optimum at these cells is 3.4."""

    OPTIMUM = 3.4

    def iterated_row(self) -> dict[str, Any]:
        return make_row(grading=GRADING_ITERATED_RETURN, opponent_rule="tit-for-tat", n_rounds=5)

    def test_returns_are_the_hand_computed_totals_over_the_optimum(
        self, recorder: Recorder
    ) -> None:
        always_cooperate = COOPERATE_COMPLETION * 5
        cooperate_until_last = COOPERATE_COMPLETION * 4 + DEFECT_COMPLETION
        always_defect = DEFECT_COMPLETION * 5
        rewards = score(
            [self.iterated_row() for _ in range(3)],
            [always_cooperate, cooperate_until_last, always_defect],
            recorder,
        )
        assert rewards == pytest.approx([3.0 / self.OPTIMUM, 1.0, 1.8 / self.OPTIMUM])

    def test_a_wrong_length_move_sequence_is_a_parse_failure(self, recorder: Recorder) -> None:
        """Too few and too many `<action>` tags both fail; the valid move keeps the batch alive."""
        rewards = score(
            [self.iterated_row() for _ in range(3)],
            [
                COOPERATE_COMPLETION * 4,
                COOPERATE_COMPLETION * 6,
                COOPERATE_COMPLETION * 4 + DEFECT_COMPLETION,
            ],
            recorder,
        )
        assert rewards == pytest.approx([DEFAULT_PARSE_PENALTY, DEFAULT_PARSE_PENALTY, 1.0])

    def test_the_cooperation_rate_is_per_round(self, recorder: Recorder) -> None:
        score(
            [self.iterated_row() for _ in range(2)],
            [COOPERATE_COMPLETION * 4 + DEFECT_COMPLETION] * 2,
            recorder,
        )
        assert recorder.metrics["coop_rate"] == pytest.approx(0.8)

    def test_an_unknown_opponent_rule_raises(self, recorder: Recorder) -> None:
        rows = [
            make_row(grading=GRADING_ITERATED_RETURN, opponent_rule="random", n_rounds=5)
            for _ in range(2)
        ]
        with pytest.raises(RuntimeError, match="opponent_rule"):
            score(rows, [COOPERATE_COMPLETION * 5] * 2, recorder)


class TestContiguityGuard:
    def test_shuffled_prompt_ids_raise(self, recorder: Recorder) -> None:
        """The plan's sabotage item, kept permanently: decorrelated groups are a wrong run."""
        rows = [
            make_row(prompt_id="prompt-0"),
            make_row(prompt_id="prompt-1"),
            make_row(prompt_id="prompt-0"),
            make_row(prompt_id="prompt-1"),
        ]
        with pytest.raises(RuntimeError, match="not one prompt's group"):
            score(rows, [COOPERATE_COMPLETION] * 4, recorder, GROUPS_OF_TWO)

    def test_contiguous_groups_of_the_same_prompt_are_accepted(self, recorder: Recorder) -> None:
        rows = [make_row(prompt_id="prompt-0")] * 2 + [make_row(prompt_id="prompt-1")] * 2
        rewards = score(rows, [COOPERATE_COMPLETION] * 4, recorder, GROUPS_OF_TWO)
        assert len(rewards) == 4

    def test_a_batch_that_is_not_whole_groups_raises(self, recorder: Recorder) -> None:
        rows = [make_row() for _ in range(3)]
        with pytest.raises(RuntimeError, match="whole number of groups"):
            score(rows, [COOPERATE_COMPLETION] * 3, recorder, GROUPS_OF_TWO)

    def test_a_group_mixing_gradings_raises(self, recorder: Recorder) -> None:
        rows = [make_row(grading=GRADING_GROUP_MIX), make_row(grading=GRADING_SELF)]
        with pytest.raises(RuntimeError, match="mixes gradings"):
            score(rows, [COOPERATE_COMPLETION] * 2, recorder)

    def test_an_empty_batch_raises(self, recorder: Recorder) -> None:
        reward = make_game_reward(2, prefilled_think=False)
        with pytest.raises(RuntimeError, match="empty completion batch"):
            reward(
                completions=[],
                log_metric=recorder.log_metric,
                log_extra=recorder.log_extra,
                **as_columns([]),
            )


class TestAWholeUnparseableBatchIsAStructuralBreak:
    def test_it_raises_rather_than_returning_all_penalties(self, recorder: Recorder) -> None:
        rows = [make_row() for _ in range(4)]
        with pytest.raises(RuntimeError, match="None of 4 completions parsed"):
            score(rows, [UNPARSEABLE_COMPLETION] * 4, recorder)

    def test_one_survivor_is_enough_to_keep_training(self, recorder: Recorder) -> None:
        rows = [make_row() for _ in range(4)]
        completions = [UNPARSEABLE_COMPLETION] * 3 + [COOPERATE_COMPLETION]
        rewards = score(rows, completions, recorder)
        assert rewards[3] == pytest.approx(PAYOFF_CC)
        assert recorder.metrics["parse_failure_rate"] == pytest.approx(0.75)


class TestPrefilledThinkMustMatchTheTemplate:
    """Qwen3.5/3.8 prefill `<think>`; Qwen3-0.6B does not. The flag has to match the model."""

    QWEN35_STYLE = f"weighing it up</think>{COOPERATE_COMPLETION}"
    NO_MARKERS = DEFECT_COMPLETION

    def test_a_prefilled_completion_parses_after_the_closing_tag(self, recorder: Recorder) -> None:
        rows = [make_row() for _ in range(2)]
        rewards = score(rows, [self.QWEN35_STYLE, self.NO_MARKERS], recorder, WITH_PREFILL)
        assert rewards[0] == pytest.approx(PAYOFF_CC)

    def test_with_prefill_a_marker_free_completion_is_truncated_thinking(
        self, recorder: Recorder
    ) -> None:
        rows = [make_row() for _ in range(2)]
        rewards = score(rows, [self.QWEN35_STYLE, self.NO_MARKERS], recorder, WITH_PREFILL)
        assert rewards[1] == pytest.approx(DEFAULT_PARSE_PENALTY)
        assert recorder.metrics["truncated_thinking_rate"] == pytest.approx(0.5)

    def test_without_prefill_the_same_completion_is_a_plain_answer(
        self, recorder: Recorder
    ) -> None:
        """Both completions now parse, so the group mix is one C against one D."""
        rows = [make_row() for _ in range(2)]
        rewards = score(rows, [self.QWEN35_STYLE, self.NO_MARKERS], recorder, WITHOUT_PREFILL)
        assert rewards[1] == pytest.approx(0.5 * PAYOFF_DC + 0.5 * PAYOFF_DD)
        assert recorder.metrics["parse_failure_rate"] == pytest.approx(0.0)
        assert recorder.metrics["truncated_thinking_rate"] == pytest.approx(0.0)


class TestColumnContractViolations:
    def test_a_missing_reward_column_raises(self, recorder: Recorder) -> None:
        columns = as_columns([make_row() for _ in range(2)])
        del columns["endowment"]
        reward = make_game_reward(2, prefilled_think=False)
        with pytest.raises(RuntimeError, match="missing reward columns"):
            reward(
                completions=[COOPERATE_COMPLETION] * 2,
                log_metric=recorder.log_metric,
                log_extra=recorder.log_extra,
                **columns,
            )

    def test_a_column_list_out_of_step_with_the_completions_raises(
        self, recorder: Recorder
    ) -> None:
        columns = as_columns([make_row() for _ in range(2)])
        columns["grading"] = [GRADING_GROUP_MIX]
        reward = make_game_reward(2, prefilled_think=False)
        with pytest.raises(RuntimeError, match="must be parallel"):
            reward(
                completions=[COOPERATE_COMPLETION] * 2,
                log_metric=recorder.log_metric,
                log_extra=recorder.log_extra,
                **columns,
            )

    def test_an_unknown_grading_raises(self, recorder: Recorder) -> None:
        rows = [make_row(grading="vibes") for _ in range(2)]
        with pytest.raises(RuntimeError, match="unknown grading"):
            score(rows, [COOPERATE_COMPLETION] * 2, recorder)

    def test_a_corrupt_payoff_column_raises(self, recorder: Recorder) -> None:
        """MatrixGameSpec's own range check catches a payoff column that left [0,1]."""
        rows = [make_row(payoff_dc=7.0) for _ in range(2)]
        with pytest.raises(ValueError, match="Payoffs must lie in"):
            score(rows, [COOPERATE_COMPLETION] * 2, recorder)


FORMAT_WEIGHT = {component.name: component.weight for component in FORMAT_COMPONENTS}
TIDY_COOPERATION = f"<action>{COOPERATIVE_LABEL}</action>"
TIDY_DEFECTION = f"<action>{DEFECTING_LABEL}</action>"
CHATTY_COOPERATION = f"<action>{COOPERATIVE_LABEL}</action> and I stand by it."
REPEATED_COOPERATION = f"<action>{COOPERATIVE_LABEL}</action>\n<action>{COOPERATIVE_LABEL}</action>"


class TestFormatOnlyHandMath:
    """The placebo: reward from answer shape, behaviour still measured, nothing strategic graded.

    The completions are one tidy answer, one with prose after the tag, one with the tag repeated, and
    one that never answered -- so the group exercises the parse boundary and two shape debits at once.
    """

    COMPLETIONS: ClassVar[list[str]] = [
        TIDY_COOPERATION,
        CHATTY_COOPERATION,
        REPEATED_COOPERATION,
        UNPARSEABLE_COMPLETION,
    ]

    @pytest.fixture
    def rows(self) -> list[dict[str, Any]]:
        return [make_row(grading=GRADING_FORMAT_ONLY) for _ in range(len(self.COMPLETIONS))]

    def test_each_reward_is_the_shape_components_it_credits(
        self, rows: list[dict[str, Any]], recorder: Recorder
    ) -> None:
        tidy = MAX_FORMAT_REWARD
        chatty = MAX_FORMAT_REWARD - FORMAT_WEIGHT[COMPONENT_NOTHING_AFTER_TAG]
        repeated = MAX_FORMAT_REWARD - FORMAT_WEIGHT[COMPONENT_SINGLE_TAG]
        rewards = score(rows, self.COMPLETIONS, recorder)
        assert rewards == pytest.approx([tidy, chatty, repeated, DEFAULT_PARSE_PENALTY])
        assert rewards == pytest.approx([1.0, 0.7, 0.7, -1.0])

    def test_the_parse_boundary_dominates_the_whole_shape_range(
        self, rows: list[dict[str, Any]], recorder: Recorder
    ) -> None:
        """The sense in which this arm is still "reward = parse success", as the plan asked for.

        The gap from the worst answer to the best is at most the rubric's ceiling, while the gap from
        any answer to no answer is at least the penalty plus that ceiling. So parse success is the
        dominant term and the shape components break ties among completions that did answer.
        """
        rewards = score(rows, self.COMPLETIONS, recorder)
        answered = rewards[:-1]
        assert max(answered) - min(answered) <= MAX_FORMAT_REWARD
        assert min(answered) - rewards[-1] > MAX_FORMAT_REWARD

    def test_behaviour_is_recorded_even_though_the_reward_ignores_it(
        self, rows: list[dict[str, Any]], recorder: Recorder
    ) -> None:
        # Both a science requirement and a mechanical one: what the placebo DOES while its reward
        # ignores what it does is the measurement, and `games.train.required_metrics_for` demands
        # coop_rate for every grading but the dictator's, so a scorer setting no behavioural field
        # would fail the post-run read-back gate after the card was paid for.
        score(rows, self.COMPLETIONS, recorder)
        assert recorder.metrics["coop_rate"] == pytest.approx(1.0)
        assert "mean_keep_fraction" not in recorder.metrics
        assert recorder.extra["parsed_action"] == ["C", "C", "C", ""]

    def test_the_component_rates_are_taken_over_the_completions_that_answered(
        self, rows: list[dict[str, Any]], recorder: Recorder
    ) -> None:
        # Three of four answered. Mixing the unanswered one into the denominator would make a rising
        # parse-failure rate read as falling format compliance, which is a different finding.
        score(rows, self.COMPLETIONS, recorder)
        assert recorder.metrics[f"format_{COMPONENT_SINGLE_TAG}_rate"] == pytest.approx(2 / 3)
        assert recorder.metrics[f"format_{COMPONENT_NOTHING_AFTER_TAG}_rate"] == pytest.approx(
            2 / 3
        )
        assert recorder.metrics["parse_failure_rate"] == pytest.approx(1 / 4)

    def test_no_component_rate_is_logged_under_a_graded_grading(self, recorder: Recorder) -> None:
        """The other direction: a metric that fired on every grading would look correct."""
        rows = [make_row() for _ in range(2)]
        score(rows, [COOPERATE_COMPLETION, DEFECT_COMPLETION], recorder)
        for name in COMPONENT_NAMES:
            assert f"format_{name}_rate" not in recorder.metrics

    def test_swapping_every_action_changes_behaviour_and_not_one_reward(
        self, recorder: Recorder
    ) -> None:
        """The placebo property, end to end through the reward TRL actually calls."""
        rows = [make_row(grading=GRADING_FORMAT_ONLY) for _ in range(2)]
        cooperating = score(rows, [TIDY_COOPERATION, CHATTY_COOPERATION], recorder)
        coop_rate = recorder.metrics["coop_rate"]
        defecting = score(
            rows,
            [TIDY_DEFECTION, CHATTY_COOPERATION.replace(COOPERATIVE_LABEL, DEFECTING_LABEL)],
            recorder,
        )
        assert defecting == pytest.approx(cooperating)
        assert (coop_rate, recorder.metrics["coop_rate"]) == (
            pytest.approx(1.0),
            pytest.approx(0.0),
        )

    def test_a_group_answering_in_one_shape_is_pure_and_carries_no_gradient(
        self, recorder: Recorder
    ) -> None:
        """The arm's real design risk, made concrete rather than argued about.

        Once a policy answers tidily every time, every reward in the group is identical, every GRPO
        advantage is zero and the arm trains nothing. `games.format_spread` measures how close a real
        baseline sweep sits to this from completions already on disk, before any GPU is reserved.
        """
        rows = [make_row(grading=GRADING_FORMAT_ONLY) for _ in range(4)]
        rewards = score(rows, [TIDY_COOPERATION] * 4, recorder)
        assert len(set(rewards)) == 1
        assert recorder.metrics["frac_groups_pure"] == pytest.approx(1.0)

    def test_a_whole_batch_that_never_answered_still_raises(self, recorder: Recorder) -> None:
        # Inherited from the shared reward path and worth pinning here: a format-only arm whose
        # completions all fail to parse is every reward at the penalty, which is a structural break
        # rather than a bad step, exactly as under the graded gradings.
        rows = [make_row(grading=GRADING_FORMAT_ONLY) for _ in range(2)]
        with pytest.raises(RuntimeError, match="parsed into an action"):
            score(rows, [UNPARSEABLE_COMPLETION] * 2, recorder)

    def test_shuffled_prompt_ids_still_raise_under_this_grading(self, recorder: Recorder) -> None:
        # The contiguity guard runs on every call whatever the grading. This arm estimates nothing
        # from its group, so a decorrelated block would not corrupt its rewards -- but the guard
        # firing anyway is what keeps it a property of the reward function rather than of one branch,
        # and a run whose sampler had decorrelated is not a run to let continue on any arm.
        rows = [
            make_row(grading=GRADING_FORMAT_ONLY, prompt_id=f"prompt-{index}") for index in (0, 1)
        ]
        reward = make_game_reward(2, prefilled_think=False)
        with pytest.raises(RuntimeError, match="not one prompt's"):
            score(rows, [TIDY_COOPERATION] * 2, recorder, reward)

    def test_purity_is_measured_on_the_rewards_themselves(self, recorder: Recorder) -> None:
        """Two completions differing in shape make an impure group; identical ones do not."""
        rows = [make_row(grading=GRADING_FORMAT_ONLY) for _ in range(2)]
        mixed = score(rows, [TIDY_COOPERATION, CHATTY_COOPERATION], recorder)
        assert recorder.metrics["frac_groups_pure"] == pytest.approx(0.0)
        assert len(set(mixed)) == 2


class TestFactoryValidation:
    @pytest.mark.parametrize("num_generations", [0, 1])
    def test_a_group_too_small_for_a_relative_advantage_raises(self, num_generations: int) -> None:
        with pytest.raises(ValueError, match="num_generations must be at least"):
            make_game_reward(num_generations, prefilled_think=False)

    @pytest.mark.parametrize("penalty", [0.0, 0.5])
    def test_a_non_negative_parse_penalty_raises(self, penalty: float) -> None:
        """A penalty inside the payoff range would make an unparseable completion competitive."""
        with pytest.raises(ValueError, match="must be a finite negative number"):
            make_game_reward(2, prefilled_think=False, parse_penalty=penalty)

    @pytest.mark.parametrize("penalty", [float("nan"), float("-inf"), float("inf")])
    def test_a_non_finite_parse_penalty_raises(self, penalty: float) -> None:
        """`--parse-penalty nan` passes argparse's `type=float`, and `nan >= 0.0` is False.

        A NaN penalty is not a price below the payoff range, it is no price at all: TRL 1.10 reads a
        NaN reward as unscorable, excludes the completion from its group's nan-aware baseline and
        forces its advantage to zero, so the parse gradient vanishes while `parse_failure_rate` keeps
        logging. `-inf` instead makes the group mean and every advantage in it non-finite.
        """
        with pytest.raises(ValueError, match="must be a finite negative number"):
            make_game_reward(2, prefilled_think=False, parse_penalty=penalty)


class TestStatedMatchGrading:
    """The track-record reward: EV under the row's own stated p, and the metrics that read it.

    The below-crossover case is this arm's sabotage item, kept as a permanent test: on a cell
    whose stated p sits under the EV crossover, a cooperating completion MUST score strictly
    lower than a defecting one. A reward that kept paying cooperation everywhere would re-teach
    exactly the counterpart-blindness the arm exists to avoid, produce healthy-looking curves,
    and invalidate every pre-registered reading -- so the test is written against hand numbers
    at the normalised twin-PD cells, not against the function's own output.
    """

    def test_below_the_crossover_cooperating_scores_strictly_lower(
        self, recorder: Recorder
    ) -> None:
        # p=0.4 on twin-pd-temptation-2 (crossover 5/7): cooperate pays 0.4*0.6 = 0.24 and
        # defect pays 0.4*0.2 + 0.6*1.0 = 0.68.
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4) for _ in range(2)]
        rewards = GROUPS_OF_TWO(
            completions=[COOPERATE_COMPLETION, DEFECT_COMPLETION],
            log_metric=recorder.log_metric,
            log_extra=recorder.log_extra,
            **as_columns(rows),
        )
        assert rewards[0] == pytest.approx(0.24)
        assert rewards[1] == pytest.approx(0.68)
        assert rewards[0] < rewards[1]

    def test_above_the_crossover_cooperating_scores_strictly_higher(
        self, recorder: Recorder
    ) -> None:
        # p=0.95: cooperate pays 0.95*0.6 = 0.57 and defect 0.95*0.2 + 0.05*1.0 = 0.24.
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.95) for _ in range(2)]
        rewards = GROUPS_OF_TWO(
            completions=[COOPERATE_COMPLETION, DEFECT_COMPLETION],
            log_metric=recorder.log_metric,
            log_extra=recorder.log_extra,
            **as_columns(rows),
        )
        assert rewards[0] == pytest.approx(0.57)
        assert rewards[1] == pytest.approx(0.24)
        assert rewards[0] > rewards[1]

    def test_the_reward_is_not_a_fixed_function_of_the_action(self, recorder: Recorder) -> None:
        """The design's whole point: the same action's reward moves with the stated p."""
        low = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4) for _ in range(2)]
        high = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.95) for _ in range(2)]
        at_low = GROUPS_OF_TWO(
            completions=[COOPERATE_COMPLETION, COOPERATE_COMPLETION],
            log_metric=recorder.log_metric,
            log_extra=recorder.log_extra,
            **as_columns(low),
        )
        at_high = GROUPS_OF_TWO(
            completions=[COOPERATE_COMPLETION, COOPERATE_COMPLETION],
            log_metric=recorder.log_metric,
            log_extra=recorder.log_extra,
            **as_columns(high),
        )
        assert at_low[0] != pytest.approx(at_high[0])

    def test_the_unset_marker_is_refused_as_corpus_corruption(self, recorder: Recorder) -> None:
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=-1.0) for _ in range(2)]
        with pytest.raises(RuntimeError, match="stated_match_prob"):
            GROUPS_OF_TWO(
                completions=[COOPERATE_COMPLETION, DEFECT_COMPLETION],
                log_metric=recorder.log_metric,
                log_extra=recorder.log_extra,
                **as_columns(rows),
            )

    def test_the_marker_is_refused_even_when_nothing_parses(self, recorder: Recorder) -> None:
        """Corruption outranks the parse penalty: a corrupt row is a corrupt row either way."""
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=-1.0) for _ in range(2)]
        with pytest.raises(RuntimeError, match="stated_match_prob"):
            GROUPS_OF_TWO(
                completions=[UNPARSEABLE_COMPLETION, UNPARSEABLE_COMPLETION],
                log_metric=recorder.log_metric,
                log_extra=recorder.log_extra,
                **as_columns(rows),
            )

    def test_an_unparseable_completion_takes_the_penalty(self, recorder: Recorder) -> None:
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.95) for _ in range(2)]
        rewards = GROUPS_OF_TWO(
            completions=[UNPARSEABLE_COMPLETION, COOPERATE_COMPLETION],
            log_metric=recorder.log_metric,
            log_extra=recorder.log_extra,
            **as_columns(rows),
        )
        assert rewards[0] == DEFAULT_PARSE_PENALTY
        assert rewards[1] == pytest.approx(0.57)

    def test_the_metrics_split_by_incentive_direction(self, recorder: Recorder) -> None:
        """Two groups on opposite sides of the crossover, each split across the two actions.

        The pooled coop_rate is 0.5 either way and says nothing; the split metrics are the
        reading. On the high-p group cooperating is optimal, on the low-p group defecting is, so
        with one cooperator and one defector in each: ev_optimum_rate 0.5, both split rates 0.5.
        """
        reward = make_game_reward(2, prefilled_think=False)
        rows = [
            make_row(prompt_id="high-p", grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.95),
            make_row(prompt_id="high-p", grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.95),
            make_row(prompt_id="low-p", grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4),
            make_row(prompt_id="low-p", grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4),
        ]
        reward(
            completions=[
                COOPERATE_COMPLETION,
                DEFECT_COMPLETION,
                COOPERATE_COMPLETION,
                DEFECT_COMPLETION,
            ],
            log_metric=recorder.log_metric,
            log_extra=recorder.log_extra,
            **as_columns(rows),
        )
        assert recorder.metrics["coop_rate"] == pytest.approx(0.5)
        assert recorder.metrics["ev_optimum_rate"] == pytest.approx(0.5)
        assert recorder.metrics["coop_rate_where_coop_pays"] == pytest.approx(0.5)
        assert recorder.metrics["coop_rate_where_defect_pays"] == pytest.approx(0.5)

    def test_blanket_cooperation_reads_as_anti_incentive_on_the_low_side(
        self, recorder: Recorder
    ) -> None:
        """The drift the split metrics exist to catch: cooperate everywhere.

        Pooled coop_rate reads 1.0 (looks like strong movement); the split shows
        coop_rate_where_defect_pays at 1.0 and ev_optimum_rate at 0.5 -- the anti-incentive half
        is visible per step instead of averaged away.
        """
        reward = make_game_reward(2, prefilled_think=False)
        rows = [
            make_row(prompt_id="high-p", grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.95),
            make_row(prompt_id="high-p", grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.95),
            make_row(prompt_id="low-p", grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4),
            make_row(prompt_id="low-p", grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4),
        ]
        reward(
            completions=[COOPERATE_COMPLETION] * 4,
            log_metric=recorder.log_metric,
            log_extra=recorder.log_extra,
            **as_columns(rows),
        )
        assert recorder.metrics["coop_rate"] == pytest.approx(1.0)
        assert recorder.metrics["ev_optimum_rate"] == pytest.approx(0.5)
        assert recorder.metrics["coop_rate_where_coop_pays"] == pytest.approx(1.0)
        assert recorder.metrics["coop_rate_where_defect_pays"] == pytest.approx(1.0)

    def test_the_dose_columns_ride_into_the_trace(self, recorder: Recorder) -> None:
        """stated_match_prob and the signed margin land per completion in log_extra."""
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4) for _ in range(2)]
        GROUPS_OF_TWO(
            completions=[COOPERATE_COMPLETION, DEFECT_COMPLETION],
            log_metric=recorder.log_metric,
            log_extra=recorder.log_extra,
            **as_columns(rows),
        )
        assert recorder.extra["stated_match_prob"] == [0.4, 0.4]
        # gap(0.4) on the normalised cells: 0.4*(0.6-0.2) - 0.6*(1.0-0.0) = -0.44.
        assert recorder.extra["stated_match_ev_margin"][0] == pytest.approx(-0.44)

    def test_other_gradings_carry_none_in_the_dose_columns(self, recorder: Recorder) -> None:
        rows = [make_row(grading=GRADING_SELF) for _ in range(2)]
        GROUPS_OF_TWO(
            completions=[COOPERATE_COMPLETION, DEFECT_COMPLETION],
            log_metric=recorder.log_metric,
            log_extra=recorder.log_extra,
            **as_columns(rows),
        )
        assert recorder.extra["stated_match_prob"] == [None, None]
        assert "ev_optimum_rate" not in recorder.metrics


class TestParsePenaltyModes:
    """How an unparseable completion is priced, and the property the scaled mode exists for.

    Under `margin-below-worse` a failure earns min(EV) - |EV(C) - EV(D)| on ITS row, so the three
    outcomes rank right > wrong > malformed with EQUAL gaps and the format channel's gradient is
    exactly the side-correct channel's size. track-record-v2 trained under the constant -1.0
    against 0.10 margins, and its readout located the coop-ward drift's carrier in that tenfold
    penalty landing on long, defect-leaning deliberation. The hand numbers are the normalised
    twin-PD cells (CC=0.6, CD=0.0, DC=1.0, DD=0.2) the rest of this file uses.
    """

    def test_the_default_mode_pays_the_constant(self, recorder: Recorder) -> None:
        """Every arm registered before the mode existed keeps its reward byte for byte."""
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4) for _ in range(2)]
        rewards = score(rows, [UNPARSEABLE_COMPLETION, DEFECT_COMPLETION], recorder)
        assert rewards[0] == DEFAULT_PARSE_PENALTY
        explicit = make_game_reward(
            2, prefilled_think=False, parse_penalty_mode=PARSE_PENALTY_CONSTANT
        )
        assert score(rows, [UNPARSEABLE_COMPLETION, DEFECT_COMPLETION], recorder, explicit) == (
            rewards
        )

    def test_an_unknown_mode_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="parse_penalty_mode must be one of"):
            make_game_reward(2, prefilled_think=False, parse_penalty_mode="gentle")

    def test_below_the_crossover_a_failure_sits_one_margin_under_the_worse_action(
        self, recorder: Recorder
    ) -> None:
        # p=0.4: cooperate pays 0.24, defect 0.68, |gap| 0.44, so malformed = 0.24 - 0.44 = -0.20.
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4) for _ in range(2)]
        rewards = score(
            rows,
            [UNPARSEABLE_COMPLETION, DEFECT_COMPLETION],
            recorder,
            MARGIN_BELOW_WORSE_GROUPS_OF_TWO,
        )
        assert rewards[0] == pytest.approx(-0.20)
        assert rewards[1] == pytest.approx(0.68)

    def test_above_the_crossover_the_worse_action_is_defection(self, recorder: Recorder) -> None:
        # p=0.95: cooperate pays 0.57, defect 0.24, |gap| 0.33, so malformed = 0.24 - 0.33 = -0.09.
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.95) for _ in range(2)]
        rewards = score(
            rows,
            [UNPARSEABLE_COMPLETION, COOPERATE_COMPLETION],
            recorder,
            MARGIN_BELOW_WORSE_GROUPS_OF_TWO,
        )
        assert rewards[0] == pytest.approx(-0.09)
        assert rewards[1] == pytest.approx(0.57)

    def test_right_wrong_and_malformed_are_equally_spaced(self, recorder: Recorder) -> None:
        """The property the mode exists for, on one hand-checked cell: gaps 0.44 and 0.44."""
        reward = make_game_reward(
            3, prefilled_think=False, parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE
        )
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4) for _ in range(3)]
        right, wrong, malformed = score(
            rows,
            [DEFECT_COMPLETION, COOPERATE_COMPLETION, UNPARSEABLE_COMPLETION],
            recorder,
            reward,
        )
        assert right > wrong > malformed
        assert right - wrong == pytest.approx(0.44)
        assert wrong - malformed == pytest.approx(0.44)

    def test_the_constant_is_still_validated_but_not_consulted(self, recorder: Recorder) -> None:
        """It stays a recorded run knob, so a nonsense value is refused even where nothing reads it.

        `TestTheRowRelativeParsePriceOnEveryGrading` carries the same property for the gradings the
        2026-09-04 generalisation added; here it is pinned on the grading the mode was written for.
        """
        with pytest.raises(ValueError, match="must be a finite negative number"):
            make_game_reward(
                2,
                prefilled_think=False,
                parse_penalty=0.5,
                parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE,
            )
        reward = make_game_reward(
            2,
            prefilled_think=False,
            parse_penalty=-7.0,
            parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE,
        )
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4) for _ in range(2)]
        rewards = score(rows, [UNPARSEABLE_COMPLETION, DEFECT_COMPLETION], recorder, reward)
        assert rewards[0] == pytest.approx(-0.20)

    def test_the_scaled_mode_now_prices_a_grading_beyond_this_one(self, recorder: Recorder) -> None:
        """This mode used to be gated to vs-stated-match and refused everywhere else.

        The 2026-09-03 auxiliary-term audit generalised it (its G1), so self grading -- whose two
        reachable rewards are its diagonal cells -- now prices its own failures instead of being
        refused. `TestTheRowRelativeParsePriceOnMatrixRows` carries the hand math; what is pinned here
        is that the old refusal is gone rather than moved.
        """
        rows = [make_row(grading=GRADING_SELF) for _ in range(2)]
        rewards = score(
            rows,
            [COOPERATE_COMPLETION, UNPARSEABLE_COMPLETION],
            recorder,
            MARGIN_BELOW_WORSE_GROUPS_OF_TWO,
        )
        assert rewards == pytest.approx([PAYOFF_CC, min(PAYOFF_CC, PAYOFF_DD) - 0.4])

    def test_a_cell_with_no_margin_is_refused_under_the_scaled_mode(
        self, recorder: Recorder
    ) -> None:
        """A flat table's gap is zero at every p: no worse action, so no price to derive."""
        flat = {"payoff_cc": 0.5, "payoff_cd": 0.5, "payoff_dc": 0.5, "payoff_dd": 0.5}
        rows = [
            make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.6, **flat)
            for _ in range(2)
        ]
        with pytest.raises(RuntimeError, match="sits exactly on its EV crossover"):
            score(
                rows,
                [UNPARSEABLE_COMPLETION, DEFECT_COMPLETION],
                recorder,
                MARGIN_BELOW_WORSE_GROUPS_OF_TWO,
            )
        # The constant mode has nothing to derive and prices the same failure as ever.
        assert score(rows, [UNPARSEABLE_COMPLETION, DEFECT_COMPLETION], recorder)[0] == (
            DEFAULT_PARSE_PENALTY
        )

    def test_the_realized_parse_price_is_logged_per_step(self, recorder: Recorder) -> None:
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4) for _ in range(2)]
        score(
            rows,
            [UNPARSEABLE_COMPLETION, DEFECT_COMPLETION],
            recorder,
            MARGIN_BELOW_WORSE_GROUPS_OF_TWO,
        )
        assert recorder.metrics["mean_parse_penalty_reward"] == pytest.approx(-0.20)
        score(rows, [UNPARSEABLE_COMPLETION, DEFECT_COMPLETION], recorder)
        assert recorder.metrics["mean_parse_penalty_reward"] == pytest.approx(DEFAULT_PARSE_PENALTY)
        fresh = Recorder()
        score(rows, [COOPERATE_COMPLETION, DEFECT_COMPLETION], fresh)
        assert "mean_parse_penalty_reward" not in fresh.metrics

    def test_a_failure_carries_its_cell_into_the_trace(self, recorder: Recorder) -> None:
        """v2's parquets had None here on failures, so which side a failed completion fell on
        needed a join back to the corpus; now the row's p and signed margin ride along."""
        rows = [make_row(grading=GRADING_VS_STATED_MATCH, stated_match_prob=0.4) for _ in range(2)]
        score(rows, [UNPARSEABLE_COMPLETION, DEFECT_COMPLETION], recorder)
        assert recorder.extra["parsed_action"] == ["", "D"]
        assert recorder.extra["stated_match_prob"] == [0.4, 0.4]
        assert recorder.extra["stated_match_ev_margin"][0] == pytest.approx(-0.44)
        assert recorder.metrics["coop_rate_where_defect_pays"] == pytest.approx(0.0)

    def test_every_row_of_a_real_v2_build_ranks_the_three_outcomes_with_equal_gaps(
        self, recorder: Recorder
    ) -> None:
        """The arm's corpus, built the way `games.track_record_corpus` builds it.

        One reskin's stems through the real v2 chain grid: every row's displayed quantized table
        and stated rate, exactly what the box trains on. On each row the right action beats the
        wrong one by that row's realized margin (0.10 by the corpus audit), and the wrong one
        beats a malformed answer by the same amount.
        """
        source = generate_prompt_rows(SOURCE_GAME_ID, "group-mix", split="train")
        one_reskin = [row for row in source if row["reskin_id"] == source[0]["reskin_id"]]
        corpus_rows, _ = build_track_record_v2_rows(one_reskin)
        assert corpus_rows, "the v2 build produced no rows"
        reward = make_game_reward(
            3, prefilled_think=False, parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE
        )
        for corpus_row in corpus_rows:
            row = make_row(**{k: corpus_row[k] for k in REQUIRED_REWARD_COLUMNS if k in corpus_row})
            coop = f"<action>{row['coop_label']}</action>"
            defect_label = row["label_b"] if row["label_a"] == row["coop_label"] else row["label_a"]
            defect = f"<action>{defect_label}</action>"
            rewards = score(
                [row, row, row], [coop, defect, UNPARSEABLE_COMPLETION], recorder, reward
            )
            malformed = rewards[2]
            right, wrong = max(rewards[:2]), min(rewards[:2])
            margin = abs(recorder.extra["stated_match_ev_margin"][0])
            assert margin == pytest.approx(0.10, abs=5e-4), row["prompt_id"]
            assert right > wrong > malformed, row["prompt_id"]
            assert right - wrong == pytest.approx(margin), row["prompt_id"]
            assert wrong - malformed == pytest.approx(margin), row["prompt_id"]


CARE_ZERO = care_grading(0)
CARE_ONE = care_grading(1)
# The registered trust game's numbers, restated so each expectation below reads as its own algebra.
TRUST_STOCK = 10
TRUST_MULTIPLE = 3.0
TRUST_ANNOUNCED_HALF = 0.5
TRUST_CEILING = TRUST_STOCK * TRUST_ANNOUNCED_HALF * TRUST_MULTIPLE


def make_trust_row(**overrides: Any) -> dict[str, Any]:
    """One announced-rule trust row: the care family's second row type.

    The blank action labels and zero payoff cells are `games.prompts._trust_row`'s verbatim, not
    tidiness: the answer is a number, so a real trust row has no labels to print, and a fixture that
    inherited the twin-PD labels would send a row shape no corpus carries down the matrix path
    successfully whenever the dispatch stopped recognising it as trust.
    """
    trust_defaults: dict[str, Any] = {
        "grading": CARE_ONE,
        "game_id": "trust-vs-stated-return",
        "payoff_cc": 0.0,
        "payoff_cd": 0.0,
        "payoff_dc": 0.0,
        "payoff_dd": 0.0,
        "label_a": "",
        "label_b": "",
        "coop_label": "",
        "endowment": TRUST_STOCK,
        "transfer_multiplier": TRUST_MULTIPLE,
        "stated_return_fraction": TRUST_ANNOUNCED_HALF,
    }
    return make_row(**{**trust_defaults, **overrides})


class TestCareGradingNames:
    """One reward, one name. Two spellings of one weight would read as two arms in every artifact."""

    def test_the_canonical_name_is_the_same_for_an_int_and_a_float(self) -> None:
        assert care_grading(1) == "care-alpha-1"
        assert care_grading(1.0) == "care-alpha-1"
        assert care_grading(0) == "care-alpha-0"
        assert care_grading(0.5) == "care-alpha-0.5"
        assert care_grading(2.25) == "care-alpha-2.25"

    def test_the_weight_reads_back_off_a_canonical_name(self) -> None:
        for alpha in (0.0, 0.5, 1.0, 2.25, 10.0):
            assert care_alpha_of(care_grading(alpha)) == pytest.approx(alpha)

    def test_every_other_string_reads_back_as_no_weight(self) -> None:
        for name in (GRADING_GROUP_MIX, "care-alpha-", "care-alpha-x", "care-alpha--1", "", "care"):
            assert care_alpha_of(name) is None

    def test_a_non_canonical_spelling_reads_back_as_no_weight(self) -> None:
        # care_alpha_of stays a pure lookup so the per-group dispatch can call it without a try;
        # is_grading is what turns the second spelling into an error naming the first.
        assert care_alpha_of("care-alpha-1.0") is None

    def test_a_non_canonical_spelling_is_refused_by_name(self) -> None:
        for spelling in ("care-alpha-1.0", "care-alpha-01", "care-alpha-1.50"):
            with pytest.raises(ValueError, match="use 'care-alpha-"):
                is_grading(spelling)

    def test_the_membership_test_accepts_the_family_and_the_registered_gradings(self) -> None:
        assert is_grading(CARE_ONE)
        assert is_grading(CARE_ZERO)
        assert is_grading(GRADING_GROUP_MIX)
        assert not is_grading("care-alpha-x")
        assert not is_grading("vibes")

    def test_a_negative_or_non_finite_weight_has_no_name(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            care_grading(-1)
        with pytest.raises(ValueError, match="finite"):
            care_grading(float("inf"))

    def test_a_weight_no_decimal_literal_spells_is_refused(self) -> None:
        # format(1e-07, "g") is "1e-07", which the family's own pattern would not read back: a name
        # nothing downstream can parse is worse than a refused launch.
        with pytest.raises(ValueError, match="decimal literal"):
            care_grading(1e-7)

    def test_the_refusal_message_names_the_family(self) -> None:
        message = unknown_grading_message("vibes")
        assert "vibes" in message
        assert CARE_GRADING_PREFIX in message
        assert GRADING_GROUP_MIX in message

    def test_the_cli_validator_accepts_the_family_and_refuses_the_rest(self) -> None:
        assert grading_cli_value(CARE_ONE) == CARE_ONE
        assert grading_cli_value(GRADING_GROUP_MIX) == GRADING_GROUP_MIX
        # ArgumentTypeError rather than ValueError in both directions: argparse discards a
        # ValueError's message and prints its own terse "invalid value" instead.
        with pytest.raises(argparse.ArgumentTypeError, match="Unknown grading"):
            grading_cli_value("care-alpha-x")
        with pytest.raises(argparse.ArgumentTypeError, match="care-alpha-1"):
            grading_cli_value("care-alpha-1.0")


class TestTheCareFamilyRowTypeColumnIsARewardColumn:
    def test_the_discriminator_names_a_column_the_reward_function_reads(self) -> None:
        # Three modules outside the reward function dispatch on this column by name; a rename of the
        # `_Row` field would leave them all reading a column that is not there.
        assert STATED_RETURN_FRACTION_COLUMN in REQUIRED_REWARD_COLUMNS


class TestCareFamilyOnMatrixRows:
    """The family's matrix leg against the same C, C, D group the ladder's hand math uses.

    Two identities carry wave 4b: alpha 0 has to be own-payoff group-mix and alpha 1 has to be
    joint-welfare group-mix, so the control arm reproduces a rung that has already been run.
    """

    COMPLETIONS: ClassVar[list[str]] = [
        COOPERATE_COMPLETION,
        COOPERATE_COMPLETION,
        DEFECT_COMPLETION,
        UNPARSEABLE_COMPLETION,
    ]

    def test_alpha_zero_scores_exactly_as_group_mix(self, recorder: Recorder) -> None:
        care = score([make_row(grading=CARE_ZERO) for _ in range(4)], self.COMPLETIONS, recorder)
        group_mix = score(
            [make_row(grading=GRADING_GROUP_MIX) for _ in range(4)], self.COMPLETIONS, Recorder()
        )
        assert care == pytest.approx(group_mix)
        cooperator = (2 / 3) * PAYOFF_CC + (1 / 3) * PAYOFF_CD
        assert care[0] == pytest.approx(cooperator)
        assert care[3] == DEFAULT_PARSE_PENALTY

    def test_alpha_one_scores_exactly_as_joint_welfare_group_mix(self, recorder: Recorder) -> None:
        care = score([make_row(grading=CARE_ONE) for _ in range(4)], self.COMPLETIONS, recorder)
        joint = score(
            [make_row(grading=GRADING_JOINT_WELFARE_GROUP_MIX) for _ in range(4)],
            self.COMPLETIONS,
            Recorder(),
        )
        assert care == pytest.approx(joint)

    def test_the_within_group_gap_is_monotone_in_alpha(self) -> None:
        gaps = []
        for alpha in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            rows = [make_row(grading=care_grading(alpha)) for _ in range(4)]
            rewards = score(rows, self.COMPLETIONS, Recorder())
            gaps.append(rewards[0] - rewards[2])
        assert gaps == sorted(gaps)
        assert gaps[0] < 0 < gaps[-1]

    def test_the_mix_excludes_the_unparseable_completion_as_every_group_mix_grading_does(
        self, recorder: Recorder
    ) -> None:
        rewards = score([make_row(grading=CARE_ONE) for _ in range(4)], self.COMPLETIONS, recorder)
        off_diagonal = (PAYOFF_CD + PAYOFF_DC) / 2
        assert rewards[0] == pytest.approx((2 / 3) * PAYOFF_CC + (1 / 3) * off_diagonal)
        assert recorder.metrics["coop_rate"] == pytest.approx(2 / 3)
        assert "mean_send_fraction" not in recorder.metrics

    def test_leave_one_out_and_the_uninformative_prior_behave_as_they_do_for_group_mix(
        self, recorder: Recorder
    ) -> None:
        # One parsed completion in a group of two, so leave-one-out has no other action to read and
        # the prior is substituted -- and counted, which is the whole point of counting it.
        rows = [make_row(grading=CARE_ONE) for _ in range(2)]
        reward = make_game_reward(2, prefilled_think=False, leave_one_out=True)
        rewards = score(rows, [COOPERATE_COMPLETION, UNPARSEABLE_COMPLETION], recorder, reward)
        off_diagonal = (PAYOFF_CD + PAYOFF_DC) / 2
        assert rewards[0] == pytest.approx(
            UNINFORMATIVE_COOP_PRIOR * PAYOFF_CC + (1 - UNINFORMATIVE_COOP_PRIOR) * off_diagonal
        )
        # Half, not all: the rate's denominator is every completion in the batch, and the
        # unparseable one took the penalty rather than a prior.
        assert recorder.metrics["leave_one_out_prior_rate"] == pytest.approx(0.5)

    def test_the_family_prices_a_failure_against_its_own_row(self) -> None:
        # The mode was refused for every grading but vs-stated-match until the 2026-09-03 audit
        # generalised it; the family's matrix leg prices from the two actions' care rewards at the
        # resolved mix. `TestTheRowRelativeParsePriceOnMatrixRows` is where the weights' hand math and
        # the vanishing-spread property live.
        rows = [make_row(grading=CARE_ONE) for _ in range(2)]
        rewards = score(
            rows,
            [COOPERATE_COMPLETION, UNPARSEABLE_COMPLETION],
            Recorder(),
            MARGIN_BELOW_WORSE_GROUPS_OF_TWO,
        )
        # The one parsed completion cooperated, so the mix is 1: C pays CC = 0.6 and D pays the
        # off-diagonal mean 0.5, and the price is 0.5 - 0.1 = 0.4.
        assert rewards == pytest.approx([PAYOFF_CC, 0.4])


class TestCareFamilyOnTrustRows:
    """The family's trust leg: one grading name, a second row type, and the weight-dependent traps."""

    def test_alpha_zero_scores_exactly_as_the_announced_rule_trust_grading(
        self, recorder: Recorder
    ) -> None:
        completions = ["<send>10</send>", "<send>0</send>"]
        care = score([make_trust_row(grading=CARE_ZERO) for _ in range(2)], completions, recorder)
        stated_rule = score(
            [make_trust_row(grading=GRADING_TRUSTOR_PAYOFF_STATED_RULE) for _ in range(2)],
            completions,
            Recorder(),
        )
        assert care == pytest.approx(stated_rule)
        assert care[0] == pytest.approx(
            (TRUST_STOCK - 10 + TRUST_ANNOUNCED_HALF * TRUST_MULTIPLE * 10) / TRUST_CEILING
        )

    def test_at_alpha_one_the_reward_is_the_giving_gradient(self, recorder: Recorder) -> None:
        completions = ["<send>10</send>", "<send>0</send>", "<send>4</send>"]
        rewards = score([make_trust_row() for _ in range(3)], completions, recorder)
        for sent, reward in zip((10, 0, 4), rewards, strict=True):
            expected = (TRUST_STOCK + (TRUST_MULTIPLE - 1) * sent) / (2 * TRUST_CEILING)
            assert reward == pytest.approx(expected)

    def test_the_announced_rate_does_not_move_the_alpha_one_reward(self) -> None:
        # own + other is E + (m - 1) * s whatever the rate, so both registered variants pay the same
        # reward for the same send: a giving gradient with no counterpart-reading component.
        completions = ["<send>7</send>", "<send>7</send>"]
        at_fifth = score(
            [make_trust_row(stated_return_fraction=0.2) for _ in range(2)], completions, Recorder()
        )
        at_half = score([make_trust_row() for _ in range(2)], completions, Recorder())
        assert at_fifth == pytest.approx(at_half)

    def test_the_send_metric_is_reported_and_no_cooperation_rate_is(
        self, recorder: Recorder
    ) -> None:
        score([make_trust_row() for _ in range(2)], ["<send>10</send>", "<send>0</send>"], recorder)
        assert recorder.metrics["mean_send_fraction"] == pytest.approx(0.5)
        assert "coop_rate" not in recorder.metrics

    def test_the_optimum_reading_follows_the_care_weight_rather_than_own_payoff(self) -> None:
        # At the return-fifth rate own payoff is maximised by sending NOTHING (0.2 * 3 < 1), while at
        # alpha 1 the whole stock is optimal. A reading fixed to the own-payoff corner would report
        # the opposite of what the reward pays.
        own_payoff = Recorder()
        score(
            [make_trust_row(grading=CARE_ZERO, stated_return_fraction=0.2) for _ in range(2)],
            ["<send>0</send>", "<send>10</send>"],
            own_payoff,
        )
        assert own_payoff.metrics["send_at_payoff_optimum_rate"] == pytest.approx(0.5)
        care = Recorder()
        score(
            [make_trust_row(stated_return_fraction=0.2) for _ in range(2)],
            ["<send>10</send>", "<send>10</send>"],
            care,
        )
        assert care.metrics["send_at_payoff_optimum_rate"] == pytest.approx(1.0)

    def test_an_unparseable_send_takes_the_penalty_and_leaves_the_others_alone(
        self, recorder: Recorder
    ) -> None:
        rewards = score(
            [make_trust_row() for _ in range(2)],
            ["<send>10</send>", "I would rather not commit"],
            recorder,
        )
        assert rewards[1] == DEFAULT_PARSE_PENALTY
        assert rewards[0] > DEFAULT_PARSE_PENALTY

    def test_a_row_carrying_the_no_rate_marker_is_refused_by_name(self, recorder: Recorder) -> None:
        # Under the family the marker also means "a game answered with one of two labels", so it can
        # never be read AS a rate. A real trust row that lost its rate prints no labels either, so the
        # matrix path it would fall through to raises about corrupt labels instead of about the missing
        # rate; the refusal names the row and the marker rather than paying every completion the
        # send-nothing reward while the prompt promised a return.
        rows = [make_trust_row(stated_return_fraction=STATED_RETURN_UNSET) for _ in range(2)]
        with pytest.raises(RuntimeError, match="announces no return rate"):
            score(rows, ["<send>10</send>", "<send>0</send>"], recorder)

    def test_the_refusal_survives_a_batch_whose_other_groups_parse(
        self, recorder: Recorder
    ) -> None:
        # The whole-batch parse guard cannot stand in for it: with seven matrix groups parsing, a
        # rate-stripped trust group would take its penalties in silence.
        rows = [make_row(grading=CARE_ONE, prompt_id="matrix")] * 2 + [
            make_trust_row(prompt_id="trust", stated_return_fraction=STATED_RETURN_UNSET)
        ] * 2
        completions = [
            COOPERATE_COMPLETION,
            DEFECT_COMPLETION,
            "<send>10</send>",
            "<send>0</send>",
        ]
        with pytest.raises(RuntimeError, match="announces no return rate"):
            score(rows, completions, recorder, GROUPS_OF_TWO)

    def test_a_number_answered_game_under_the_family_is_refused_by_name(
        self, recorder: Recorder
    ) -> None:
        # The unilateral split, the simultaneous claim and the shared undertaking all print no action
        # labels and carry the no-rate marker, so the family has no scorer for any of them and the
        # matrix fall-through would report label corruption on a row whose labels are blank by design.
        rows = [
            make_row(
                grading=CARE_ONE,
                game_id="dictator",
                prompt_id="split",
                payoff_cc=0.0,
                payoff_cd=0.0,
                payoff_dc=0.0,
                payoff_dd=0.0,
                label_a="",
                label_b="",
                coop_label="",
            )
            for _ in range(2)
        ]
        with pytest.raises(RuntimeError, match="prints no action labels"):
            score(rows, ["<keep>10</keep>", "<keep>0</keep>"], recorder)

    def test_a_weight_that_puts_the_trust_reward_above_the_matrix_scale_is_refused(
        self, recorder: Recorder
    ) -> None:
        # At the return-fifth rate the counterpart keeps 24 of the 30 stock units in play against a
        # trustor ceiling of 15, so alpha 2 pays 1.2 where every matrix row in the batch caps at 1.
        rows = [
            make_trust_row(grading=care_grading(2), stated_return_fraction=0.2) for _ in range(2)
        ]
        with pytest.raises(ValueError, match=r"above 1\.0"):
            score(rows, ["<send>10</send>", "<send>0</send>"], recorder)


class TestCareFamilyOverAMixedCorpus:
    """The wave-4b corpus shape: matrix groups and trust groups under ONE grading, in one batch.

    The family's whole point is that a single corpus can carry both, so this is the case that would
    break if the dispatch ever keyed the scorer on the grading name alone.
    """

    def test_both_row_types_are_scored_and_both_behavioural_metrics_are_logged(
        self, recorder: Recorder
    ) -> None:
        rows = [make_row(grading=CARE_ONE, prompt_id="matrix")] * 2 + [
            make_trust_row(prompt_id="trust")
        ] * 2
        completions = [
            COOPERATE_COMPLETION,
            DEFECT_COMPLETION,
            "<send>10</send>",
            "<send>0</send>",
        ]
        rewards = score(rows, completions, recorder, GROUPS_OF_TWO)
        off_diagonal = (PAYOFF_CD + PAYOFF_DC) / 2
        assert rewards[0] == pytest.approx(0.5 * PAYOFF_CC + 0.5 * off_diagonal)
        assert rewards[2] == pytest.approx(
            (TRUST_STOCK + (TRUST_MULTIPLE - 1) * 10) / (2 * TRUST_CEILING)
        )
        assert recorder.metrics["coop_rate"] == pytest.approx(0.5)
        assert recorder.metrics["mean_send_fraction"] == pytest.approx(0.5)

    def test_an_unscorable_grading_names_the_family_in_its_refusal(self) -> None:
        rows = [make_row(grading="vibes") for _ in range(2)]
        with pytest.raises(RuntimeError, match="care-alpha"):
            score(rows, [COOPERATE_COMPLETION, DEFECT_COMPLETION], Recorder())

    def test_a_corpus_row_in_the_wrong_spelling_names_the_canonical_one(self) -> None:
        rows = [make_row(grading="care-alpha-1.0") for _ in range(2)]
        with pytest.raises(ValueError, match="care-alpha-1'"):
            score(rows, [COOPERATE_COMPLETION, DEFECT_COMPLETION], Recorder())


class TestEveryGradingGateAcceptsTheCareFamily:
    """Every place that used to ask `name in GRADINGS`, asked with a family member and with junk.

    An open family means the membership question cannot be a set lookup anywhere, and a gate left
    behind does not fail loudly at launch: it refuses the arm at the ONE moment the operator is
    waiting on a rented card.
    """

    def test_the_arm_registry_accepts_a_care_arm_and_classifies_it_for_the_span_check(self) -> None:
        arm = GameArm(
            game_id="twin-pd", grading=CARE_ONE, notes="a care arm registered inside a test"
        )
        validate_arms({"care-under-test": arm})
        assert span_check_exemption(CARE_ONE) is not None
        assert span_check_exemption(care_grading(0.5)) == span_check_exemption(CARE_ONE)
        assert span_check_exemption("vibes") is None

    def test_the_arm_registry_refuses_a_grading_nothing_scores(self) -> None:
        arm = GameArm(game_id="twin-pd", grading="care-alpha-x", notes="not a grading at all")
        with pytest.raises(ValueError, match="Unknown grading"):
            validate_arms({"care-under-test": arm})

    def test_the_per_arm_span_check_refuses_an_arm_no_classification_covers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The fail-closed direction for an open family: with the exemption gone, a care arm must be
        # refused rather than silently skipping the dead-arm check. Called directly rather than
        # through validate_arms, which reaches the import-time classification gate first.
        monkeypatch.setattr(games_arms, "CARE_SPAN_CHECK_EXEMPTION", None)
        arm = GameArm(game_id="twin-pd", grading=CARE_ONE, notes="a care arm with no exemption")
        with pytest.raises(ValueError, match="no dead-arm check"):
            games_arms._validate_reachable_reward_span("care-under-test", arm)

    def test_the_classification_gate_covers_the_family_by_name_pattern(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert_every_grading_is_classified()
        monkeypatch.setattr(games_arms, "CARE_SPAN_CHECK_EXEMPTION", None)
        with pytest.raises(ValueError, match="care family"):
            assert_every_grading_is_classified()

    def test_the_stage_plans_sweep_grading_accepts_the_family(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "twin-pd-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", CARE_ONE)
        assert arm_sequence.sweep_grading() == CARE_ONE
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_GRADING", "care-alpha-x")
        with pytest.raises(ValueError, match="is not a grading"):
            arm_sequence.sweep_grading()

    def test_the_row_renderers_accept_the_family(self) -> None:
        rows = generate_prompt_rows("twin-pd", CARE_ONE, split="train")
        assert rows
        assert {row["grading"] for row in rows} == {CARE_ONE}
        with pytest.raises(ValueError, match="Unknown grading"):
            generate_prompt_rows("twin-pd", "care-alpha-x", split="train")

    def test_the_regrade_module_accepts_the_family(self) -> None:
        rows = generate_prompt_rows("twin-pd", CARE_ONE, split="train")
        regraded = regrade_rows(rows, target_grading=CARE_ZERO)
        assert {row["grading"] for row in regraded} == {CARE_ZERO}
        with pytest.raises(ValueError, match="Unknown grading"):
            regrade_rows(rows, target_grading="care-alpha-x")


class TestPerGameAndPerFramingMetrics:
    """The training-side breadth gradient, logged per step from the columns the corpus already has.

    Wave 4b's corpus mixes six games and seven counterpart framings, so the pooled `coop_rate` of a
    step averages cells that are the whole measurement: the registered expectation is cooperation
    rising on the near framings first and spreading outward, which a pooled series cannot show. The
    split rides through the same `log_metric` every other metric uses, so it lands in
    `trainer_state.json` and the trajectory is a re-analysis rather than another 200-step run.

    `framing_id` is read out of the forwarded columns rather than added to `_Row`, so a corpus
    written before the column existed keeps loading and simply produces no framing keys.
    """

    @staticmethod
    def score_with(
        rows: list[dict[str, Any]],
        completions: list[str],
        recorder: Recorder,
        **extra_columns: list[Any],
    ) -> list[float]:
        """Call the reward with the required columns plus whichever extra ones the case supplies."""
        return GROUPS_OF_TWO(
            completions=completions,
            log_metric=recorder.log_metric,
            log_extra=recorder.log_extra,
            **as_columns(rows),
            **extra_columns,
        )

    def mixed_batch(self, recorder: Recorder) -> None:
        """Four groups of two: two games times two framings, with known cooperation per group."""
        rows = [
            make_row(prompt_id="twin-twin", game_id="twin-pd"),
            make_row(prompt_id="twin-twin", game_id="twin-pd"),
            make_row(prompt_id="twin-human", game_id="twin-pd"),
            make_row(prompt_id="twin-human", game_id="twin-pd"),
            make_row(prompt_id="stag-twin", game_id="stag-hunt"),
            make_row(prompt_id="stag-twin", game_id="stag-hunt"),
            make_row(prompt_id="stag-human", game_id="stag-hunt"),
            make_row(prompt_id="stag-human", game_id="stag-hunt"),
        ]
        completions = [
            COOPERATE_COMPLETION,
            COOPERATE_COMPLETION,
            DEFECT_COMPLETION,
            DEFECT_COMPLETION,
            COOPERATE_COMPLETION,
            DEFECT_COMPLETION,
            COOPERATE_COMPLETION,
            COOPERATE_COMPLETION,
        ]
        framings = ["twin", "twin", "human", "human", "twin", "twin", "human", "human"]
        self.score_with(rows, completions, recorder, framing_id=framings)

    def test_each_game_gets_its_own_cooperation_rate_and_denominator(
        self, recorder: Recorder
    ) -> None:
        self.mixed_batch(recorder)
        assert recorder.metrics["coop_rate/game/twin-pd"] == pytest.approx(0.5)
        assert recorder.metrics["coop_rate/game/stag-hunt"] == pytest.approx(0.75)
        assert recorder.metrics["n_groups/game/twin-pd"] == pytest.approx(2)
        assert recorder.metrics["n_groups/game/stag-hunt"] == pytest.approx(2)
        # The pooled rate stays what it was, so the split is an addition rather than a replacement.
        assert recorder.metrics["coop_rate"] == pytest.approx(5 / 8)

    def test_each_framing_gets_its_own_cooperation_rate_purity_and_denominator(
        self, recorder: Recorder
    ) -> None:
        self.mixed_batch(recorder)
        assert recorder.metrics["coop_rate/framing/twin"] == pytest.approx(0.75)
        assert recorder.metrics["coop_rate/framing/human"] == pytest.approx(0.5)
        assert recorder.metrics["n_groups/framing/twin"] == pytest.approx(2)
        assert recorder.metrics["n_groups/framing/human"] == pytest.approx(2)
        # Under the twin framing one group is unanimous and one is split; under the human framing
        # both are unanimous, so a rise in either series is a collapse rather than learning.
        assert recorder.metrics["frac_groups_pure/framing/twin"] == pytest.approx(0.5)
        assert recorder.metrics["frac_groups_pure/framing/human"] == pytest.approx(1.0)

    def test_a_corpus_with_no_framing_column_logs_the_game_split_and_no_framing_keys(
        self, recorder: Recorder
    ) -> None:
        rows = [make_row(prompt_id="twin", game_id="twin-pd") for _ in range(2)]
        self.score_with(rows, [COOPERATE_COMPLETION, DEFECT_COMPLETION], recorder)
        assert recorder.metrics["coop_rate/game/twin-pd"] == pytest.approx(0.5)
        assert not [key for key in recorder.metrics if "/framing/" in key]

    def test_a_row_carrying_no_framing_contributes_to_no_framing_key(
        self, recorder: Recorder
    ) -> None:
        # The trust sender's rows: their counterpart paragraph is the announced return rule, which is
        # not a framing the sweep can hold fixed, so they carry the unset marker and are counted
        # nowhere on the framing axis rather than pooled into a framing named after nothing.
        rows = [
            make_row(prompt_id="twin", game_id="twin-pd"),
            make_row(prompt_id="twin", game_id="twin-pd"),
            make_trust_row(prompt_id="trust"),
            make_trust_row(prompt_id="trust"),
        ]
        completions = [
            COOPERATE_COMPLETION,
            COOPERATE_COMPLETION,
            "<send>10</send>",
            "<send>0</send>",
        ]
        framings = ["twin", "twin", FRAMING_ID_UNSET, FRAMING_ID_UNSET]
        self.score_with(rows, completions, recorder, framing_id=framings)
        assert recorder.metrics["n_groups/framing/twin"] == pytest.approx(1)
        assert sorted(key for key in recorder.metrics if "/framing/" in key) == [
            "coop_rate/framing/twin",
            "frac_groups_pure/framing/twin",
            "n_groups/framing/twin",
        ]

    def test_a_game_with_no_cooperative_action_gets_a_denominator_and_no_rate(
        self, recorder: Recorder
    ) -> None:
        # A zero needs its denominator, and so does an absence: the trust rows produce no cooperation
        # rate at all, so their group count is the only thing saying the step trained them.
        rows = [make_trust_row(prompt_id="trust") for _ in range(2)]
        self.score_with(rows, ["<send>10</send>", "<send>0</send>"], recorder)
        assert recorder.metrics["n_groups/game/trust-vs-stated-return"] == pytest.approx(1)
        assert "coop_rate/game/trust-vs-stated-return" not in recorder.metrics

    def test_a_group_whose_completions_disagree_about_their_framing_is_refused(
        self, recorder: Recorder
    ) -> None:
        # A group is one prompt, so its framing is one value. Two values in a block means the columns
        # arrived decorrelated from the completions, which would file this group's cooperation under
        # whichever framing happened to come first.
        rows = [make_row(prompt_id="twin", game_id="twin-pd") for _ in range(2)]
        with pytest.raises(RuntimeError, match="disagree about their framing_id"):
            self.score_with(
                rows,
                [COOPERATE_COMPLETION, DEFECT_COMPLETION],
                recorder,
                framing_id=["twin", "human"],
            )

    def test_a_framing_column_of_the_wrong_length_is_refused(self, recorder: Recorder) -> None:
        rows = [make_row(prompt_id="twin", game_id="twin-pd") for _ in range(2)]
        with pytest.raises(RuntimeError, match="parallel"):
            self.score_with(
                rows, [COOPERATE_COMPLETION, DEFECT_COMPLETION], recorder, framing_id=["twin"]
            )


# The sheet the row-relative price is hand-checked on. Its care-alpha-1 spread is `0.2 - 0.4p`, so it
# vanishes at a mix a group of three can realise, which is the property the breadth pair's price
# exists for: the joint-welfare attractor is where the task channel goes to zero, and a constant
# price there is the whole gradient. The file's default temptation-2 sheet cannot express it, since
# its crossing sits at p = 1.5, outside every reachable mix.
ATTRACTOR_CELLS: dict[str, float] = {
    "payoff_cc": 0.3,
    "payoff_cd": 0.0,
    "payoff_dc": 1.0,
    "payoff_dd": 0.3,
}
# A constant no grading's row-relative price could coincide with, so a test that sees it knows the
# scorer consulted the constant instead of the row.
SENTINEL_PARSE_PENALTY = -7.25
THREE_ROUNDS = 3
LEVEL_GRID = 5
TWIN_PD_SHEET = MatrixGameSpec(
    game_id="twin-pd-temptation-2",
    payoff_cc=PAYOFF_CC,
    payoff_cd=PAYOFF_CD,
    payoff_dc=PAYOFF_DC,
    payoff_dd=PAYOFF_DD,
)
ATTRACTOR_SHEET = MatrixGameSpec(game_id="attractor", **ATTRACTOR_CELLS)
THRESHOLD_GOODS_SHEET = ThresholdGoodsSpec(
    game_id="threshold-goods", endowment=10, team_size=2, contribution_threshold=9, prize=12
)
MIN_EFFORT_SHEET = MinEffortSpec(
    game_id="min-effort",
    n_levels=LEVEL_GRID,
    benefit_per_level=1.0,
    cost_per_level=0.1,
    team_size=1,
)

# How to build one row of every grading, and an answer that parses on it. Keyed by grading and
# asserted complete against `GRADINGS`, because a grading added upstream with no row here would
# otherwise never be asked whether it can price a failure at all -- and an unpriced grading pays the
# constant while `run_config.json` records the row-relative mode.
ROW_SHAPE_PER_GRADING: dict[str, tuple[dict[str, Any], str]] = {
    GRADING_GROUP_MIX: ({}, COOPERATE_COMPLETION),
    GRADING_SELF: ({}, COOPERATE_COMPLETION),
    GRADING_VS_FIXED_MIX: ({"opp_coop_prob": 0.5}, COOPERATE_COMPLETION),
    GRADING_JOINT_WELFARE_GROUP_MIX: ({}, COOPERATE_COMPLETION),
    GRADING_OTHER_PAYOFF_GROUP_MIX: ({}, COOPERATE_COMPLETION),
    GRADING_VS_STATED_MATCH: ({"stated_match_prob": 0.4}, COOPERATE_COMPLETION),
    GRADING_KEEP_FRACTION: ({}, "<keep>6</keep>"),
    GRADING_ITERATED_RETURN: (
        {"opponent_rule": "tit-for-tat", "n_rounds": THREE_ROUNDS},
        COOPERATE_COMPLETION * THREE_ROUNDS,
    ),
    GRADING_NASH_DEMAND_SELF: ({"windfall": 10}, "<claim>5</claim>"),
    GRADING_NASH_DEMAND_GROUP_MIX: ({"windfall": 10}, "<claim>5</claim>"),
    GRADING_THRESHOLD_GOODS_SELF: (
        {"endowment": 10, "team_size": 2, "contribution_threshold": 9, "prize": 12},
        "<contribute>3</contribute>",
    ),
    GRADING_THRESHOLD_GOODS_GROUP_MIX: (
        {"endowment": 10, "team_size": 2, "contribution_threshold": 9, "prize": 12},
        "<contribute>3</contribute>",
    ),
    GRADING_MIN_EFFORT_GROUP_MIX: (
        {"n_levels": LEVEL_GRID, "benefit_per_level": 1.0, "cost_per_level": 0.1, "team_size": 1},
        "<level>3</level>",
    ),
    GRADING_LEVEL_MATCH_RETURN: (
        {
            "n_levels": LEVEL_GRID,
            "benefit_per_level": 1.0,
            "cost_per_level": 0.1,
            "team_size": 1,
            "n_rounds": THREE_ROUNDS,
        },
        "<level>3</level>" * THREE_ROUNDS,
    ),
    GRADING_TRUSTOR_PAYOFF_STATED_RULE: ({}, "<send>5</send>"),
    GRADING_TRUSTOR_PAYOFF_SELF_RULE: (
        {"stated_return_fraction": STATED_RETURN_UNSET},
        "<send>5</send><return>50</return>",
    ),
    GRADING_FORMAT_ONLY: ({}, COOPERATE_COMPLETION),
}
TRUST_SHAPED_GRADINGS = (GRADING_TRUSTOR_PAYOFF_STATED_RULE, GRADING_TRUSTOR_PAYOFF_SELF_RULE)


def shape_key_of(grading: str) -> str:
    """The `ROW_SHAPE_PER_GRADING` key a grading is built from.

    The care family reads the group-mix shape, because its matrix leg IS a labelled matrix row scored
    against the group's realised mix; its trust leg is asked for by name through `make_trust_row`.
    """
    return GRADING_GROUP_MIX if care_alpha_of(grading) is not None else grading


def row_of_grading(grading: str, **overrides: Any) -> dict[str, Any]:
    """One row of any grading from `ROW_SHAPE_PER_GRADING`, on the trust shape where it applies."""
    shape, _ = ROW_SHAPE_PER_GRADING[shape_key_of(grading)]
    builder = make_trust_row if grading in TRUST_SHAPED_GRADINGS else make_row
    return builder(grading=grading, **{**shape, **overrides})


def parsing_completion_for(grading: str) -> str:
    """The answer that parses on `row_of_grading(grading)`."""
    _, completion = ROW_SHAPE_PER_GRADING[shape_key_of(grading)]
    return completion


def row_relative_reward(group_size: int, **kwargs: Any) -> Callable[..., list[float]]:
    """A reward callable pricing failures relative to the row, with a sentinel constant.

    The sentinel is what makes "the constant was not consulted" observable: a grading that fell back
    to it shows up as that exact number rather than as a plausible reward.
    """
    return make_game_reward(
        group_size,
        prefilled_think=False,
        parse_penalty=SENTINEL_PARSE_PENALTY,
        parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE,
        **kwargs,
    )


def price_below_the_worse_action(spec: MatrixGameSpec, mix: float, *, alpha: float) -> float:
    """What the row-relative price should be on a matrix row, from the payoff module's own algebra."""
    worst = min(
        expected_care_payoff(spec, COOPERATE, mix, alpha=alpha),
        expected_care_payoff(spec, DEFECT, mix, alpha=alpha),
    )
    return worst - care_reward_spread(spec, mix, alpha=alpha)


class TestTheRowRelativeParsePriceOnMatrixRows:
    """One reachable spread below the row's worst reachable parsed reward, at the resolved mix.

    The generalisation of the softpen arm's price from the one grading it was gated to (2026-09-02)
    to every grading (2026-09-04). Why not a smaller constant: what dominates a step is
    `worst_parsed - price` measured against the group's own reward spread, so a constant is
    proportionate on one game at one mix and nowhere else. The care family forced it: at alpha 1 on a
    PD-shaped row the spread passes through zero at the joint-welfare attractor the arm is designed
    to settle at, so any constant diverges there while this price shrinks to nothing with the task
    channel it is scaled against.
    """

    THREE_WITH_ONE_FAILURE: ClassVar[list[str]] = [
        COOPERATE_COMPLETION,
        DEFECT_COMPLETION,
        UNPARSEABLE_COMPLETION,
    ]

    def test_the_care_weights_price_a_failure_one_spread_below_the_worse_action(self) -> None:
        """Hand math on temptation-2 at the mix a C/D pair realises, at both breadth weights.

        At alpha 1 (own plus counterpart, halved) cooperating pays (0.3 + 0.8) / 2 = 0.55 and
        defecting (0.6 + 0.1) / 2 = 0.35, so the price is 0.35 - 0.20 = +0.15 -- POSITIVE, which is
        the mode working rather than a sign error: the payoffs are normalised into [0, 1], so one
        spread below the worse action need not reach zero. At alpha 0 the same group pays 0.3 and 0.6,
        so the price is 0.3 - 0.3 = 0.0.
        """
        at_alpha_one = score(
            [row_of_grading(CARE_ONE) for _ in range(3)],
            self.THREE_WITH_ONE_FAILURE,
            Recorder(),
            row_relative_reward(3),
        )
        assert at_alpha_one == pytest.approx([0.55, 0.35, 0.15])
        at_alpha_zero = score(
            [row_of_grading(CARE_ZERO) for _ in range(3)],
            self.THREE_WITH_ONE_FAILURE,
            Recorder(),
            row_relative_reward(3),
        )
        assert at_alpha_zero == pytest.approx([0.3, 0.6, 0.0])

    def test_at_a_zero_spread_mix_the_price_is_the_worst_reachable_reward(self) -> None:
        """The property the breadth pair needs: the format channel vanishes with the task channel.

        On the attractor sheet the alpha-1 spread is 0.2 - 0.4p, so a group holding one C and one D
        sits exactly at the crossing: both actions pay 0.4 and the failure is priced at 0.4 too. A
        constant would be this group's entire gradient.
        """
        rewards = score(
            [row_of_grading(CARE_ONE, **ATTRACTOR_CELLS) for _ in range(3)],
            self.THREE_WITH_ONE_FAILURE,
            Recorder(),
            row_relative_reward(3),
        )
        assert rewards == pytest.approx([0.4, 0.4, 0.4])

    def test_the_price_shrinks_as_the_spread_shrinks(self) -> None:
        """Four mixes on the attractor sheet, and the gap below the worse action tracks the spread."""
        prices: list[float] = []
        for cooperators in range(4):
            completions = (
                [COOPERATE_COMPLETION] * cooperators
                + [DEFECT_COMPLETION] * (3 - cooperators)
                + [UNPARSEABLE_COMPLETION]
            )
            rewards = score(
                [row_of_grading(CARE_ONE, **ATTRACTOR_CELLS) for _ in range(4)],
                completions,
                Recorder(),
                row_relative_reward(4),
            )
            prices.append(rewards[-1])
        spreads = [
            care_reward_spread(ATTRACTOR_SHEET, cooperators / 3, alpha=1.0)
            for cooperators in range(4)
        ]
        worst = [
            min(
                expected_care_payoff(ATTRACTOR_SHEET, COOPERATE, cooperators / 3, alpha=1.0),
                expected_care_payoff(ATTRACTOR_SHEET, DEFECT, cooperators / 3, alpha=1.0),
            )
            for cooperators in range(4)
        ]
        gaps = [one - price for one, price in zip(worst, prices, strict=True)]
        assert gaps == pytest.approx(spreads)
        # Narrowing towards the crossing at p = 0.5 from both sides, which no constant does. The two
        # interior mixes (1/3 and 2/3) sit equally far from it, so the narrowing is stated per side
        # rather than as one monotone run.
        assert gaps[0] > gaps[1]
        assert gaps[3] > gaps[2]
        assert min(gaps) == pytest.approx(gaps[1])

    def test_the_failure_to_task_ratio_is_one_at_every_mix_and_both_weights(self) -> None:
        """Why the pair can share one price where no constant matches both arms.

        Under a constant the ratio of the failure's distance below the worse answer to the group's own
        reward spread moves with the mix, in OPPOSITE directions for the two weights: own-payoff
        grading widens its spread as cooperation rises on a PD-shaped row while the care-1 spread
        narrows towards zero. This price holds that ratio at exactly 1 for every mix under both.
        """
        for grading, alpha in ((CARE_ZERO, 0.0), (CARE_ONE, 1.0)):
            for cooperators in (1, 2):
                completions = (
                    [COOPERATE_COMPLETION] * cooperators
                    + [DEFECT_COMPLETION] * (3 - cooperators)
                    + [UNPARSEABLE_COMPLETION]
                )
                rewards = score(
                    [row_of_grading(grading) for _ in range(4)],
                    completions,
                    Recorder(),
                    row_relative_reward(4),
                )
                mix = cooperators / 3
                worst = min(
                    expected_care_payoff(TWIN_PD_SHEET, COOPERATE, mix, alpha=alpha),
                    expected_care_payoff(TWIN_PD_SHEET, DEFECT, mix, alpha=alpha),
                )
                spread = care_reward_spread(TWIN_PD_SHEET, mix, alpha=alpha)
                assert (worst - rewards[-1]) / spread == pytest.approx(1.0), (grading, cooperators)

    def test_the_mix_is_resolved_before_the_failure_is_priced(self) -> None:
        """The reorder this mode needed: the failure branch used to run before the mix existed.

        A group of two cooperators and one defector sits at p = 2/3, not at the uninformative 0.5 a
        failure-first branch would have to fall back on, and the two prices differ.
        """
        rewards = score(
            [row_of_grading(CARE_ONE) for _ in range(4)],
            [
                COOPERATE_COMPLETION,
                COOPERATE_COMPLETION,
                DEFECT_COMPLETION,
                UNPARSEABLE_COMPLETION,
            ],
            Recorder(),
            row_relative_reward(4),
        )
        at_realised_mix = price_below_the_worse_action(TWIN_PD_SHEET, 2 / 3, alpha=1.0)
        at_the_prior = price_below_the_worse_action(
            TWIN_PD_SHEET, UNINFORMATIVE_COOP_PRIOR, alpha=1.0
        )
        assert rewards[-1] == pytest.approx(at_realised_mix)
        assert at_realised_mix != pytest.approx(at_the_prior)

    def test_leave_one_out_prices_the_failure_at_its_own_resolved_mix(self) -> None:
        # The failed row is excluded from every mix already, so leave-one-out and the pooled mix
        # coincide for it -- asserted rather than assumed, because a price reading the pooled mix
        # under leave-one-out would be a second mix estimate nothing else in the module uses.
        rows = [row_of_grading(CARE_ONE) for _ in range(4)]
        completions = [
            COOPERATE_COMPLETION,
            COOPERATE_COMPLETION,
            DEFECT_COMPLETION,
            UNPARSEABLE_COMPLETION,
        ]
        pooled = score(rows, completions, Recorder(), row_relative_reward(4))
        left_out = score(rows, completions, Recorder(), row_relative_reward(4, leave_one_out=True))
        assert left_out[-1] == pytest.approx(pooled[-1])

    def test_a_group_where_nothing_parsed_prices_at_the_uninformative_prior(self) -> None:
        """Two groups in one batch, so the whole-batch parse guard does not fire first.

        A group with no parsed completion has no mix to estimate, and the price then reads the same
        uninformative prior the leave-one-out fallback does rather than the constant.
        """
        rows = [row_of_grading(CARE_ONE, prompt_id="dead") for _ in range(2)] + [
            row_of_grading(CARE_ONE, prompt_id="live") for _ in range(2)
        ]
        rewards = score(
            rows,
            [
                UNPARSEABLE_COMPLETION,
                UNPARSEABLE_COMPLETION,
                COOPERATE_COMPLETION,
                DEFECT_COMPLETION,
            ],
            Recorder(),
            row_relative_reward(2),
        )
        at_the_prior = price_below_the_worse_action(
            TWIN_PD_SHEET, UNINFORMATIVE_COOP_PRIOR, alpha=1.0
        )
        assert rewards[:2] == pytest.approx([at_the_prior, at_the_prior])

    def test_self_grading_prices_below_the_worse_diagonal(self) -> None:
        # min(CC, DD) - |CC - DD| on the row's own sheet: 0.2 - 0.4 = -0.2, the audit's formula for
        # this grading verbatim.
        rewards = score(
            [row_of_grading(GRADING_SELF) for _ in range(3)],
            self.THREE_WITH_ONE_FAILURE,
            Recorder(),
            row_relative_reward(3),
        )
        assert rewards == pytest.approx([PAYOFF_CC, PAYOFF_DD, -0.2])

    def test_a_fixed_mix_row_prices_at_the_frozen_mix(self) -> None:
        rewards = score(
            [row_of_grading(GRADING_VS_FIXED_MIX) for _ in range(3)],
            self.THREE_WITH_ONE_FAILURE,
            Recorder(),
            row_relative_reward(3),
        )
        # p = 0.5 from the row's own cached column: C pays 0.3, D pays 0.6, so the price is 0.0.
        assert rewards == pytest.approx([0.3, 0.6, 0.0])


class TestTheRowRelativeParsePriceOnEveryGrading:
    """The completeness gate, plus the answer-space prices the audit's G1 names one by one."""

    def test_the_row_table_covers_every_grading(self) -> None:
        """Fails closed: a grading added upstream lands here before it reaches a paid run."""
        assert set(ROW_SHAPE_PER_GRADING) == set(GRADINGS)

    @pytest.mark.parametrize("grading", [*sorted(GRADINGS), CARE_ZERO, CARE_ONE])
    def test_every_grading_either_prices_relative_to_the_row_or_refuses_by_name(
        self, grading: str
    ) -> None:
        """No grading may quietly pay the constant while the run record says otherwise.

        The one refusal is `format-only`, whose reward carries no information about the game at all.
        Every other grading prices its failure against the row, which the sentinel constant proves: a
        scorer that ignored the mode would return exactly that number.
        """
        rows = [row_of_grading(grading) for _ in range(2)]
        completions = [parsing_completion_for(grading), UNPARSEABLE_COMPLETION]
        if grading == GRADING_FORMAT_ONLY:
            with pytest.raises(RuntimeError, match=GRADING_FORMAT_ONLY):
                score(rows, completions, Recorder(), row_relative_reward(2))
            return
        rewards = score(rows, completions, Recorder(), row_relative_reward(2))
        assert rewards[1] != pytest.approx(SENTINEL_PARSE_PENALTY), grading
        assert rewards[1] < rewards[0], grading

    def test_the_dictator_price_is_the_whole_kept_fraction_range(self) -> None:
        # The reward IS the kept fraction, so the reachable rewards run 0 to 1 and one spread below
        # the worst of them is -1.0. That coincides with the constant every arm trained under, which
        # is the honest answer for this grading rather than a smaller number: nothing about this
        # reward is compressed.
        rewards = score(
            [row_of_grading(GRADING_KEEP_FRACTION) for _ in range(2)],
            ["<keep>6</keep>", UNPARSEABLE_COMPLETION],
            Recorder(),
            row_relative_reward(2),
        )
        assert rewards == pytest.approx([0.6, -1.0])

    def test_the_claim_prices_come_from_the_grid_at_the_groups_own_claims(self) -> None:
        # Self-graded: the reachable rewards are the ramp 0 to 0.5, since no claim above half the
        # windfall fits against a copy of itself, so the price is 0.0 - 0.5.
        spec = NashDemandSpec(game_id="nash-demand", windfall=10)
        self_graded = score(
            [row_of_grading(GRADING_NASH_DEMAND_SELF) for _ in range(2)],
            ["<claim>5</claim>", UNPARSEABLE_COMPLETION],
            Recorder(),
            row_relative_reward(2),
        )
        reachable = [nash_demand_self_reward(spec, claim) for claim in spec.claims]
        assert self_graded[1] == pytest.approx(min(reachable) - (max(reachable) - min(reachable)))
        assert self_graded[1] == pytest.approx(-0.5)
        # Group-mix: the counterpart distribution is the group's one parsed claim of 4, against which
        # every claim up to 6 fits, so the reachable maximum is 0.6 and the price is -0.6.
        group_graded = score(
            [row_of_grading(GRADING_NASH_DEMAND_GROUP_MIX) for _ in range(2)],
            ["<claim>4</claim>", UNPARSEABLE_COMPLETION],
            Recorder(),
            row_relative_reward(2),
        )
        assert group_graded[1] == pytest.approx(-0.6)

    def test_the_shared_undertaking_prices_over_its_own_grid(self) -> None:
        rewards = score(
            [row_of_grading(GRADING_THRESHOLD_GOODS_GROUP_MIX) for _ in range(2)],
            ["<contribute>3</contribute>", UNPARSEABLE_COMPLETION],
            Recorder(),
            row_relative_reward(2),
        )
        reachable = [
            threshold_goods_reward(
                THRESHOLD_GOODS_SHEET, contribution, [3], max_prize=THRESHOLD_GOODS_MAX_PRIZE
            )
            for contribution in THRESHOLD_GOODS_SHEET.contributions
        ]
        worst = min(reachable)
        assert rewards[1] == pytest.approx(worst - (max(reachable) - worst))
        assert rewards[1] < worst

    def test_the_level_game_prices_over_its_own_grid(self) -> None:
        rewards = score(
            [row_of_grading(GRADING_MIN_EFFORT_GROUP_MIX) for _ in range(2)],
            ["<level>3</level>", UNPARSEABLE_COMPLETION],
            Recorder(),
            row_relative_reward(2),
        )
        reachable = [
            min_effort_group_reward(MIN_EFFORT_SHEET, own_level=level, counterpart_levels=[3])
            for level in MIN_EFFORT_SHEET.levels
        ]
        worst = min(reachable)
        assert rewards[1] == pytest.approx(worst - (max(reachable) - worst))

    def test_both_repeated_forms_price_from_the_worst_reachable_match(self) -> None:
        """A whole match normalised by its optimum, so the reachable range runs worst/best to 1."""
        rule = OpponentRule("tit-for-tat")
        iterated = score(
            [row_of_grading(GRADING_ITERATED_RETURN) for _ in range(2)],
            [parsing_completion_for(GRADING_ITERATED_RETURN), UNPARSEABLE_COMPLETION],
            Recorder(),
            row_relative_reward(2),
        )
        worst_normalised = worst_iterated_return(
            TWIN_PD_SHEET, rule, THREE_ROUNDS
        ) / max_iterated_return(TWIN_PD_SHEET, rule, THREE_ROUNDS)
        assert iterated[1] == pytest.approx(worst_normalised - (1.0 - worst_normalised))
        match = score(
            [row_of_grading(GRADING_LEVEL_MATCH_RETURN) for _ in range(2)],
            [parsing_completion_for(GRADING_LEVEL_MATCH_RETURN), UNPARSEABLE_COMPLETION],
            Recorder(),
            row_relative_reward(2),
        )
        worst_match = worst_min_effort_match_return(
            MIN_EFFORT_SHEET, THREE_ROUNDS
        ) / max_min_effort_match_return(MIN_EFFORT_SHEET, THREE_ROUNDS)
        assert match[1] == pytest.approx(worst_match - (1.0 - worst_match))

    def test_the_trust_legs_price_from_their_corner_sends(self) -> None:
        # The reward is affine in the send, so the two corners bound it: sending nothing pays 10/15
        # and sending everything at the announced half pays 15/15, so the price is 2/3 - 1/3 = 1/3.
        stated = score(
            [row_of_grading(GRADING_TRUSTOR_PAYOFF_STATED_RULE) for _ in range(2)],
            ["<send>5</send>", UNPARSEABLE_COMPLETION],
            Recorder(),
            row_relative_reward(2),
        )
        assert stated[1] == pytest.approx(1 / 3)
        # The strategy method's corners span the whole [0, 1]: send everything promising everything
        # back pays 1, send everything promising nothing pays 0, so the price is -1.0.
        self_ruled = score(
            [row_of_grading(GRADING_TRUSTOR_PAYOFF_SELF_RULE) for _ in range(2)],
            ["<send>5</send><return>50</return>", UNPARSEABLE_COMPLETION],
            Recorder(),
            row_relative_reward(2),
        )
        assert self_ruled[1] == pytest.approx(-1.0)

    def test_the_care_familys_trust_leg_prices_from_its_own_corner_rewards(self) -> None:
        spec = TrustSpec(
            game_id="trust-vs-stated-return",
            endowment=TRUST_STOCK,
            multiplier=TRUST_MULTIPLE,
            stated_return_fraction=TRUST_ANNOUNCED_HALF,
        )
        rewards = score(
            [make_trust_row() for _ in range(2)],
            ["<send>5</send>", UNPARSEABLE_COMPLETION],
            Recorder(),
            row_relative_reward(2),
        )
        corners = trust_care_corner_rewards(
            spec, max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION, alpha=1.0
        ).values()
        worst = min(corners)
        assert rewards[1] == pytest.approx(worst - (max(corners) - worst))

    def test_a_failure_never_earns_more_than_the_worst_answer_of_its_own_row(self) -> None:
        """The invariant across every grading, which the positive prices make worth stating.

        A price above zero is the mode working, since payoffs are normalised into [0, 1], so the
        thing to pin is not the sign: it is that a malformed answer still ranks below every answer
        the row admits, with equality only where the row admits no spread at all.
        """
        for grading in [*sorted(GRADINGS), CARE_ZERO, CARE_ONE]:
            if grading == GRADING_FORMAT_ONLY:
                continue
            rows = [row_of_grading(grading) for _ in range(3)]
            completions = [
                parsing_completion_for(grading),
                parsing_completion_for(grading),
                UNPARSEABLE_COMPLETION,
            ]
            rewards = score(rows, completions, Recorder(), row_relative_reward(3))
            assert rewards[2] < min(rewards[:2]), grading

    def test_format_only_keeps_the_constant_and_says_why(self) -> None:
        """The audit leaves this one alone: its reward is a rubric over shape, not a task reward."""
        rows = [row_of_grading(GRADING_FORMAT_ONLY) for _ in range(2)]
        completions = [COOPERATE_COMPLETION, UNPARSEABLE_COMPLETION]
        with pytest.raises(RuntimeError, match="rubric"):
            score(rows, completions, Recorder(), row_relative_reward(2))
        assert score(rows, completions, Recorder())[1] == DEFAULT_PARSE_PENALTY

    def test_the_constant_is_validated_but_never_consulted_under_this_mode(self) -> None:
        """It stays a run knob recorded in `run_config.json`, so a nonsense value is still refused.

        Under this mode no grading pays it (`format-only`, the only grading that would, refuses the
        mode outright), so the range check is what stops a recorded value nothing reads from looking
        like the price that trained the run.
        """
        with pytest.raises(ValueError, match="must be a finite negative number"):
            make_game_reward(
                2,
                prefilled_think=False,
                parse_penalty=0.5,
                parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE,
            )
        rows = [row_of_grading(GRADING_SELF) for _ in range(2)]
        completions = [COOPERATE_COMPLETION, UNPARSEABLE_COMPLETION]
        steep = score(rows, completions, Recorder(), row_relative_reward(2))
        gentle = make_game_reward(
            2,
            prefilled_think=False,
            parse_penalty=-0.5,
            parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE,
        )
        assert score(rows, completions, Recorder(), gentle) == pytest.approx(steep)

    def test_the_realised_price_is_logged_for_a_grading_beyond_the_stated_match_one(self) -> None:
        recorder = Recorder()
        score(
            [row_of_grading(CARE_ONE) for _ in range(3)],
            [COOPERATE_COMPLETION, DEFECT_COMPLETION, UNPARSEABLE_COMPLETION],
            recorder,
            row_relative_reward(3),
        )
        assert recorder.metrics["mean_parse_penalty_reward"] == pytest.approx(0.15)


class TestTheParsePriceInvariantGuard:
    """The per-step check that a failure really is priced below the row it was priced from.

    The follow-up the 2026-09-04 parse-price commit named as its audit-G2 item. Under
    `margin-below-worse` each grading's failure branch supplies its OWN reachable set, and that set is
    the one thing in the arithmetic nothing else would notice being wrong: a branch that forgot an
    action or resolved the wrong counterpart distribution would price a failure inside or above the
    range its own group realised, every later step would train under a reward the run record
    misdescribes, and every metric would stay green. So the reward function now asserts, at the seam
    where a group's realised parsed rewards and its failure prices are both in hand, that every parsed
    reward lies inside the reachable range the price was derived from and that the price is exactly one
    spread below the worst of that range. Those two together are the ranking the mode exists for:
    malformed below every played answer, equal to the worst reachable reward only where the row admits
    no spread at all.

    The reachable set reaches the check from the branch that priced the failure, never recomputed here,
    which is what makes it a check on that computation rather than a comparison of two copies of it.
    """

    THREE_WITH_ONE_FAILURE: ClassVar[list[str]] = [
        COOPERATE_COMPLETION,
        DEFECT_COMPLETION,
        UNPARSEABLE_COMPLETION,
    ]

    @pytest.mark.parametrize(
        "grading", [*sorted(GRADINGS - {GRADING_FORMAT_ONLY}), CARE_ZERO, CARE_ONE]
    )
    def test_the_guard_passes_on_every_grading_that_prices_against_the_row(
        self, grading: str
    ) -> None:
        """Real scoring, every grading, one failure beside two parsed answers.

        No grading may reach a paid run whose own reachable set disagrees with the rewards its group
        realised, so this is asked of the whole vocabulary rather than of the arms wave 4b happens to
        train. `format-only` is excluded because it refuses the mode outright.
        """
        recorder = Recorder()
        rows = [row_of_grading(grading) for _ in range(3)]
        completions = [
            parsing_completion_for(grading),
            parsing_completion_for(grading),
            UNPARSEABLE_COMPLETION,
        ]
        score(rows, completions, recorder, row_relative_reward(3))
        assert recorder.metrics["parse_price/n_failures"] == pytest.approx(1.0), grading
        assert recorder.metrics["parse_price/n_checked"] == pytest.approx(1.0), grading

    def test_the_care_arms_matrix_rows_pass_and_report_the_identity(self) -> None:
        """The wave-4b arm's own grading, at the mix a C/D pair realises on temptation-2.

        The guard identity is `(worst reachable - price) / spread`, which the price's arithmetic makes
        1 at every mix; it is logged so a run says the guard ran rather than only that it did not
        raise.
        """
        recorder = Recorder()
        rewards = score(
            [row_of_grading(CARE_ONE) for _ in range(3)],
            self.THREE_WITH_ONE_FAILURE,
            recorder,
            row_relative_reward(3),
        )
        assert rewards == pytest.approx([0.55, 0.35, 0.15])
        assert recorder.metrics["parse_price/realised_mean"] == pytest.approx(0.15)
        assert recorder.metrics["parse_price/n_failures"] == pytest.approx(1.0)
        assert recorder.metrics["parse_price/n_checked"] == pytest.approx(1.0)
        # 0.35 - 0.15: the failure sits one within-group spread below the worse parsed answer.
        assert recorder.metrics["parse_price/min_margin_below_worst_parsed"] == pytest.approx(0.20)
        assert recorder.metrics["parse_price/guard_identity_min"] == pytest.approx(1.0)
        assert recorder.metrics["parse_price/guard_identity_max"] == pytest.approx(1.0)

    def test_the_trust_leg_and_the_dictator_row_pass_the_same_check(self) -> None:
        """Two answer spaces the matrix legs do not cover: corner rewards, and a bare kept fraction."""
        trust = Recorder()
        score(
            [make_trust_row() for _ in range(3)],
            ["<send>10</send>", "<send>0</send>", UNPARSEABLE_COMPLETION],
            trust,
            row_relative_reward(3),
        )
        assert trust.metrics["parse_price/n_checked"] == pytest.approx(1.0)
        assert trust.metrics["parse_price/guard_identity_max"] == pytest.approx(1.0)
        dictator = Recorder()
        score(
            [row_of_grading(GRADING_KEEP_FRACTION) for _ in range(3)],
            ["<keep>6</keep>", "<keep>2</keep>", UNPARSEABLE_COMPLETION],
            dictator,
            row_relative_reward(3),
        )
        assert dictator.metrics["parse_price/n_checked"] == pytest.approx(1.0)
        # The reward IS the kept fraction, so the reachable range is the whole [0, 1] and the price
        # is -1.0, one range below its bottom: the worst parsed answer of 0.2 sits 1.2 above it.
        assert dictator.metrics["parse_price/min_margin_below_worst_parsed"] == pytest.approx(1.2)

    def test_a_zero_spread_row_passes_with_the_price_at_the_worst_reachable_reward(self) -> None:
        """The case the breadth arm is designed to settle at, where "strictly below" cannot hold.

        On the attractor sheet the care-1 spread is `0.2 - 0.4p`, so a C/D pair sits exactly at the
        crossing: both actions pay 0.4, the price is 0.4, and the guard has to accept equality rather
        than demand a gap it would be a bug to have.
        """
        recorder = Recorder()
        rewards = score(
            [row_of_grading(CARE_ONE, **ATTRACTOR_CELLS) for _ in range(3)],
            self.THREE_WITH_ONE_FAILURE,
            recorder,
            row_relative_reward(3),
        )
        assert rewards == pytest.approx([0.4, 0.4, 0.4])
        assert recorder.metrics["parse_price/n_checked"] == pytest.approx(1.0)
        assert recorder.metrics["parse_price/min_margin_below_worst_parsed"] == pytest.approx(0.0)
        # No failure carried a positive spread, so the RATIO has nothing to divide by and cannot
        # report -- while the identity itself was still asked, which is what its own count says.
        assert "parse_price/guard_identity_min" not in recorder.metrics
        assert recorder.metrics["parse_price/n_identity_checked"] == pytest.approx(1.0)

    def test_the_constant_mode_logs_the_metrics_and_checks_nothing(self) -> None:
        """The price is a knob rather than the row's answer space, so there is no set to check against.

        The realised mean and the margin below the worst parsed answer are still the audit's own
        dominance reading, so both are logged; `n_checked` says the invariant was not asked.
        """
        recorder = Recorder()
        score(
            [row_of_grading(CARE_ONE) for _ in range(3)],
            self.THREE_WITH_ONE_FAILURE,
            recorder,
            make_game_reward(3, prefilled_think=False),
        )
        assert recorder.metrics["parse_price/n_failures"] == pytest.approx(1.0)
        assert recorder.metrics["parse_price/n_checked"] == pytest.approx(0.0)
        assert recorder.metrics["parse_price/n_identity_checked"] == pytest.approx(0.0)
        assert recorder.metrics["parse_price/realised_mean"] == pytest.approx(DEFAULT_PARSE_PENALTY)
        # 0.35 - (-1.0): the constant is 6.75 times the group's own 0.20 spread, which is the ratio
        # the row-relative mode exists to remove.
        assert recorder.metrics["parse_price/min_margin_below_worst_parsed"] == pytest.approx(1.35)
        assert "parse_price/guard_identity_min" not in recorder.metrics

    def test_a_step_with_no_failure_logs_the_denominators_anyway(self) -> None:
        """A zero needs its denominator: nothing priced is a reading, not an absent metric."""
        recorder = Recorder()
        score(
            [row_of_grading(CARE_ONE) for _ in range(2)],
            [COOPERATE_COMPLETION, DEFECT_COMPLETION],
            recorder,
            row_relative_reward(2),
        )
        assert recorder.metrics["parse_price/n_failures"] == pytest.approx(0.0)
        assert recorder.metrics["parse_price/n_checked"] == pytest.approx(0.0)
        assert recorder.metrics["parse_price/n_identity_checked"] == pytest.approx(0.0)
        assert "parse_price/realised_mean" not in recorder.metrics

    def test_leave_one_out_is_counted_unchecked_rather_than_refused(self) -> None:
        """Each completion faces its own mix there, so the group's rewards are not the price's range.

        The failure is priced at the pooled mix while every parsed completion is graded against the
        others only, so a parsed reward outside the failure's reachable range is correct arithmetic
        rather than a bug. Only the RANGE comparison is skipped there: the identity, which needs
        nothing but the branch's own set, is both checked and counted.
        """
        recorder = Recorder()
        score(
            [row_of_grading(CARE_ONE) for _ in range(3)],
            self.THREE_WITH_ONE_FAILURE,
            recorder,
            row_relative_reward(3, leave_one_out=True),
        )
        assert recorder.metrics["parse_price/n_failures"] == pytest.approx(1.0)
        assert recorder.metrics["parse_price/n_checked"] == pytest.approx(0.0)
        assert recorder.metrics["parse_price/n_identity_checked"] == pytest.approx(1.0)
        assert recorder.metrics["parse_price/guard_identity_min"] == pytest.approx(1.0)

    def test_a_reachable_set_missing_an_action_raises_and_names_the_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sabotage the guard exists for: a branch that forgot half of its answer space.

        With the defect leg dropped, the price is derived from the cooperate reward alone, so the
        group's realised defection at 0.35 sits outside the range the price claims. The raise has to
        carry the row, the grading, the set and the offending reward, because a run dies here and the
        reader is looking for which branch lied.
        """
        monkeypatch.setattr(
            "games.rewards._matrix_reachable_rewards",
            lambda payoff_of, spec, probability: [payoff_of(spec, COOPERATE, probability)],
        )
        with pytest.raises(RuntimeError, match="prompt-0") as raised:
            score(
                [row_of_grading(CARE_ONE) for _ in range(3)],
                self.THREE_WITH_ONE_FAILURE,
                Recorder(),
                row_relative_reward(3),
            )
        message = str(raised.value)
        assert CARE_ONE in message
        assert "0.35" in message

    def test_a_price_from_another_arithmetic_raises_and_names_the_realised_ratio(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of the sabotage: the right set, priced by the wrong arithmetic.

        Pricing at the worst reachable reward rather than one spread below it puts the failure level
        with the worse parsed answer, so the ratio the run record claims is 1 is really 0. The refusal
        carries that number, because it is the one the whole mode was built to hold at 1.
        """
        monkeypatch.setattr("games.rewards.margin_below_worst_reachable", min)
        with pytest.raises(RuntimeError, match="prompt-0") as raised:
            score(
                [row_of_grading(CARE_ONE) for _ in range(3)],
                self.THREE_WITH_ONE_FAILURE,
                Recorder(),
                row_relative_reward(3),
            )
        message = str(raised.value)
        assert "is actually 0.0" in message
        assert "0.35" in message

    def test_a_zero_spread_row_priced_anywhere_but_the_worst_reachable_reward_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The checker's other state: at the attractor the correct price IS the worst reachable reward.

        `min` is the right answer there, which is why the sabotage above cannot reach this case, so the
        arithmetic is shifted a tenth below instead. Without this the equality half of the invariant
        would be a branch no test has watched go red.
        """
        monkeypatch.setattr(
            "games.rewards.margin_below_worst_reachable", lambda reachable: min(reachable) - 0.1
        )
        with pytest.raises(RuntimeError, match="is actually inf") as raised:
            score(
                [row_of_grading(CARE_ONE, **ATTRACTOR_CELLS) for _ in range(3)],
                self.THREE_WITH_ONE_FAILURE,
                Recorder(),
                row_relative_reward(3),
            )
        # A spread of zero has no ratio to report, which is what `inf` above says; what the reader
        # needs is the row and the price the branch should have paid.
        message = str(raised.value)
        assert "prompt-0" in message
        assert "one spread below the worst reachable reward is 0.4" in message

    def test_a_failure_that_recorded_no_price_at_all_raises(self) -> None:
        """The hole a new failure branch could walk through: a `_Scored` built without its price.

        Called at the guard's own seam with a hand-built group, because no branch in the module can
        produce this state today and the point is that the next one cannot either.
        """
        rows = _rows_from_columns(as_columns([make_row(grading=CARE_ONE) for _ in range(2)]), 2)
        group = [
            _Scored(reward=0.55, parsed=True, detail="C"),
            _Scored(reward=0.15, parsed=False, detail=""),
        ]
        with pytest.raises(RuntimeError, match="without recording how"):
            _log_parse_price_metrics([group], [rows[0]], log_metric=Recorder().log_metric)

    def test_the_price_identity_refuses_a_halved_coefficient_under_leave_one_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The commit's own sabotage, at the resolution that used to disarm the whole guard.

        Leave-one-out grades every parsed completion against the others only, so the failure's pooled
        reachable set is not the range those rewards were drawn from: a reason to skip the RANGE half
        and no reason at all to skip the identity, which reads nothing but the tuple the branch that
        priced this failure recorded. While one gate covered both, a run launched with
        `--leave-one-out` under `margin-below-worse` checked nothing on any group-coupled row, and a
        halved coefficient trained for the whole run with `guard_identity_min` at 0.5 as the only
        trace of it.
        """
        monkeypatch.setattr(
            "games.rewards.margin_below_worst_reachable",
            lambda reachable: min(reachable) - 0.5 * (max(reachable) - min(reachable)),
        )
        with pytest.raises(RuntimeError, match="not the mode's own arithmetic") as raised:
            score(
                [row_of_grading(CARE_ONE) for _ in range(3)],
                self.THREE_WITH_ONE_FAILURE,
                Recorder(),
                row_relative_reward(3, leave_one_out=True),
            )
        message = str(raised.value)
        assert "prompt-0" in message
        assert CARE_ONE in message

    def test_a_zero_spread_row_is_identity_checked_under_leave_one_out_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case with no identity metric at all: zero spread, so the ratio has no denominator.

        `guard_identity_min` cannot report here and `n_checked` is zero under leave-one-out, so before
        the split this row was covered by nothing whatsoever. The price must still equal the worst
        reachable reward, which is what the shifted arithmetic below violates.
        """
        monkeypatch.setattr(
            "games.rewards.margin_below_worst_reachable", lambda reachable: min(reachable) - 0.1
        )
        with pytest.raises(RuntimeError, match="not the mode's own arithmetic") as raised:
            score(
                [row_of_grading(CARE_ONE, **ATTRACTOR_CELLS) for _ in range(3)],
                self.THREE_WITH_ONE_FAILURE,
                Recorder(),
                row_relative_reward(3, leave_one_out=True),
            )
        assert "one spread below the worst reachable reward is 0.4" in str(raised.value)

    def test_the_range_half_stays_gated_where_leave_one_out_moves_a_reward_outside(self) -> None:
        """The negative control on the split: the range comparison is genuinely wrong there.

        Four parsed answers alternating cooperate and defect beside one failure, group-mix graded:
        each parsed completion faces the other four only, while the failure is priced at the pooled
        mix, so a parsed reward outside the failure's reachable range is that grading's correct
        arithmetic. Dropping the `shared_resolution` gate wholesale rather than only for the identity
        reddens this test, which is why the range half stayed behind it.
        """
        recorder = Recorder()
        score(
            [row_of_grading(GRADING_GROUP_MIX) for _ in range(5)],
            [
                COOPERATE_COMPLETION,
                DEFECT_COMPLETION,
                COOPERATE_COMPLETION,
                DEFECT_COMPLETION,
                UNPARSEABLE_COMPLETION,
            ],
            recorder,
            row_relative_reward(5, leave_one_out=True),
        )
        assert recorder.metrics["parse_price/n_failures"] == pytest.approx(1.0)
        assert recorder.metrics["parse_price/n_checked"] == pytest.approx(0.0)
        assert recorder.metrics["parse_price/n_identity_checked"] == pytest.approx(1.0)

    @pytest.mark.parametrize("leave_one_out", [False, True])
    @pytest.mark.parametrize(
        "grading", [*sorted(GRADINGS - {GRADING_FORMAT_ONLY}), CARE_ZERO, CARE_ONE]
    )
    def test_a_group_with_no_parsed_answer_or_one_is_priced_without_a_refusal(
        self, grading: str, leave_one_out: bool
    ) -> None:
        """The two denominators the guard must survive, asked of the whole grading vocabulary.

        Three groups of two in one batch: both answers parsed, neither parsed, and one of each. The
        all-failed group has no realised reward to compare against and the single-answer group has one
        drawn from a mix of a single completion, and neither is a mispricing, so the identity runs on
        all three failures while the range comparison is asked only of the mixed group.

        Whether that one range check happens under leave-one-out is a property of the grading rather
        than of this case -- a row-deterministic grading shares its resolution there and a
        group-coupled one does not -- so the bound is what this test states and the exact per-grading
        answer is pinned by the two leave-one-out cases above.
        """
        recorder = Recorder()
        parses = parsing_completion_for(grading)
        score(
            [row_of_grading(grading) for _ in range(6)],
            [
                parses,
                parses,
                UNPARSEABLE_COMPLETION,
                UNPARSEABLE_COMPLETION,
                parses,
                UNPARSEABLE_COMPLETION,
            ],
            recorder,
            row_relative_reward(2, leave_one_out=leave_one_out),
        )
        assert recorder.metrics["parse_price/n_failures"] == pytest.approx(3.0), grading
        assert recorder.metrics["parse_price/n_identity_checked"] == pytest.approx(3.0), grading
        range_checks = recorder.metrics["parse_price/n_checked"]
        if leave_one_out:
            assert range_checks <= 1.0, grading
        else:
            assert range_checks == pytest.approx(1.0), grading

    @pytest.mark.parametrize(
        "grading", [*sorted(GRADINGS - {GRADING_FORMAT_ONLY}), CARE_ZERO, CARE_ONE]
    )
    def test_the_guard_reads_a_step_without_changing_one_reward(
        self, grading: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every reward identical with the guard's whole pass switched off, on every grading.

        The guard runs inside the reward function, so "it only reads the step" is a claim until the
        rewards are compared against a step scored without it. A later version that repaired a
        mispriced failure instead of raising would change what the run trains on, and no other test in
        the suite compares the two.
        """
        rows = [row_of_grading(grading) for _ in range(3)]
        completions = [
            parsing_completion_for(grading),
            parsing_completion_for(grading),
            UNPARSEABLE_COMPLETION,
        ]
        guarded = score(rows, completions, Recorder(), row_relative_reward(3))

        def read_nothing(groups: object, group_rows: object, *, log_metric: object) -> None:
            """Stand in for the guard's whole pass, so the rewards below are scored without it."""

        monkeypatch.setattr("games.rewards._log_parse_price_metrics", read_nothing)
        unguarded = score(rows, completions, Recorder(), row_relative_reward(3))
        assert guarded == pytest.approx(unguarded), grading

    def test_the_grading_with_no_answer_space_is_left_uncheckable_under_the_constant(self) -> None:
        """The one grading the row-relative mode refuses outright, on a group that parsed nothing.

        `format-only`'s reward is a rubric over answer shape (`PARSE_PRICE_UNDEFINED_GRADINGS`), so it
        only ever pays the constant and its price records no reachable set at all. Both halves of the
        guard skip it and both denominators say so, which is what completes "no grading raises on an
        all-failed group" over the whole vocabulary, this one included.
        """
        recorder = Recorder()
        score(
            [row_of_grading(GRADING_FORMAT_ONLY) for _ in range(4)],
            [
                COOPERATE_COMPLETION,
                COOPERATE_COMPLETION,
                UNPARSEABLE_COMPLETION,
                UNPARSEABLE_COMPLETION,
            ],
            recorder,
            make_game_reward(2, prefilled_think=False),
        )
        assert recorder.metrics["parse_price/n_failures"] == pytest.approx(2.0)
        assert recorder.metrics["parse_price/n_checked"] == pytest.approx(0.0)
        assert recorder.metrics["parse_price/n_identity_checked"] == pytest.approx(0.0)
