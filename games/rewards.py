"""The GRPO reward function for every game arm, dispatching per group on the `grading` column.

TRL calls one reward callable with `prompts`, `completions`, `completion_ids`, and **every
surviving dataset column as a parallel list of the same length** (verified in installed TRL
1.10, `trainer/grpo_trainer.py:1625`), plus the injected `trainer_state`, `log_extra`, and
`log_metric`. So this module reconstructs each row's game from its own columns rather than
re-calling the constructors in `games.payoffs`: the corpus on disk is the ground truth, and a
reward computed from anything else would silently stop matching the prompt the model read.

Three properties are load-bearing, and each fails silently if it breaks, which is why each one
raises instead of degrading:

1. **Group contiguity.** A prompt's G completions arrive as one contiguous block of
   `num_generations` (TRL's `RepeatSampler` yields each index `mini_repeat_count` times
   back-to-back, and TRL's own advantage code relies on the same layout via
   `.view(-1, num_generations)`). Group-mix grading estimates the opponent distribution from
   the group, so a sampler change that decorrelated the blocks would not crash the run -- it
   would quietly grade every completion against the wrong opponent population. The guard runs
   on every call, not once at startup.
2. **A whole batch that will not parse is a structural break**, not a bad step: every further
   step burns GPU on a reward signal of pure penalty. Partial failure never raises.
3. **A row missing the number its own grading reads is corrupt**, and defaulting it to anything
   would invent a counterpart: a `vs-fixed-mix` row without its frozen-opponent sample, or a
   `trustor-payoff-stated-rule` row carrying the marker for a form that announces no return rate.

Metrics go through the TRL-injected `log_metric` / `log_extra` kwargs, never by mutating a
callback's log dict: `log_metric` routes through `_pending_metrics` into `self._metrics` before
`Trainer.log` copies its record, whereas the dict a callback receives has already been copied
(the bug that lost verifier accuracy for a whole run in this repo).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import re
from dataclasses import dataclass
from functools import cache, partial
from typing import TYPE_CHECKING, Any, cast

from games.format_rubric import COMPONENT_NAMES, format_component_scores, format_reward
from games.parsing import (
    RETURN_PERCENTAGE_MAX,
    THINK_CLOSE,
    THINK_OPEN,
    parse_action,
    parse_action_sequence,
    parse_claim,
    parse_contribution,
    parse_level,
    parse_level_sequence,
    parse_send,
    parse_split,
    parse_trust_strategy,
    strip_thinking,
)
from games.payoffs import (
    COOPERATE,
    DEFECT,
    PAYOFF_MAX,
    PAYOFF_MIN,
    STATED_MATCH_PROB_UNSET,
    STATED_RETURN_UNSET,
    THRESHOLD_GOODS_MAX_PRIZE,
    TRUST_MAX_SELF_STATED_RETURN_FRACTION,
    TRUST_MAX_STATED_RETURN_FRACTION,
    MatrixGameSpec,
    MinEffortSpec,
    NashDemandSpec,
    OpponentRule,
    ThresholdGoodsSpec,
    TrustSpec,
    assert_care_alpha,
    assert_trust_care_spec,
    expected_care_payoff,
    expected_counterpart_payoff,
    expected_joint_payoff,
    max_iterated_return,
    max_min_effort_match_return,
    min_effort_best_response,
    min_effort_group_reward,
    min_effort_match_reward,
    min_effort_pressure_gaps,
    nash_demand_crash_probability,
    nash_demand_group_reward,
    nash_demand_share,
    simulate_iterated,
    stated_match_expected_payoff,
    stated_match_gap,
    stated_match_optimal_action,
    threshold_goods_over_contribution,
    threshold_goods_reach_probability,
    threshold_goods_reward_at_reach,
    trust_break_even_return_fraction,
    trust_care_corner_rewards,
    trust_self_rule_corner_rewards,
    trust_stated_rule_corner_rewards,
    trustor_care_reward,
    trustor_reward,
    worst_iterated_return,
    worst_min_effort_match_return,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

GRADING_GROUP_MIX = "group-mix"
GRADING_SELF = "self"
GRADING_KEEP_FRACTION = "keep-fraction"
GRADING_VS_FIXED_MIX = "vs-fixed-mix"
GRADING_ITERATED_RETURN = "iterated-return"
# The wave-3 grading ladder: the matrix expected-payoff arithmetic with the reward's RECIPIENT
# moved. `joint-welfare-group-mix` pays the MEAN of both sides' payoffs and `other-payoff-group-mix`
# pays the counterpart's payoff alone, both against the group's realised mix -- the same mix
# estimation, prior handling and parse policy as own-payoff group-mix, because the recipient is the
# ONLY thing the ladder varies. Group-mix coupling is forced, not chosen: against a counterpart
# that copies the completion (the `self` coupling) own, joint and other all collapse to the same
# diagonal number, so the recipient axis is only expressible against an uncopied counterpart.
GRADING_JOINT_WELFARE_GROUP_MIX = "joint-welfare-group-mix"
GRADING_OTHER_PAYOFF_GROUP_MIX = "other-payoff-group-mix"
GRADING_NASH_DEMAND_SELF = "nash-demand-self"
GRADING_NASH_DEMAND_GROUP_MIX = "nash-demand-group-mix"
# The two gradings of the simultaneous-claim division, which run on byte-identical prompts and are
# the game's whole contrast: one grades against a copy of the model's own claim, the other against
# the group's realised claim distribution. Both predict the same number (the equal division), which
# is what makes the pair a numeric control rather than a direction.
NASH_DEMAND_GRADINGS: frozenset[str] = frozenset(
    {GRADING_NASH_DEMAND_SELF, GRADING_NASH_DEMAND_GROUP_MIX}
)
GRADING_THRESHOLD_GOODS_SELF = "threshold-goods-self"
GRADING_THRESHOLD_GOODS_GROUP_MIX = "threshold-goods-group-mix"
# The two gradings of the shared undertaking, on byte-identical prompts, the same contrast shape as the
# claim game's pair: the counterparts write exactly what this completion wrote, or they are drawn from
# the group's realised figures. The self-graded leg is the one whose optimum is a number the registry can
# check -- the equal share, interior by construction -- and the group-mix leg is where the two prize
# variants' best responses come apart against a low-contributing group.
THRESHOLD_GOODS_GRADINGS: frozenset[str] = frozenset(
    {GRADING_THRESHOLD_GOODS_SELF, GRADING_THRESHOLD_GOODS_GROUP_MIX}
)
# The stated-track-record grading: the game's expected payoff to self under the decision-matching
# probability the prompt itself states, `p * payoff(a,a) + (1-p) * payoff(a, other)`. Deterministic
# (the expectation, never a sampled counterpart) and honest by construction: the number in the
# clause and the number in the reward are one column. Unlike `self` (payoff(a,a), a fixed function
# of the model's own action, which measurably taught the policy to stop reading the counterpart)
# and unlike any single fixed p (whose EV is again a function of own action only), a corpus MIXING
# stated p across each game's EV crossover makes no action unconditionally optimal -- the
# counterpart clause is the only thing that says which side pays, so reading it stays
# reward-relevant. That mixture is the arm's mechanism, and games.train refuses a corpus whose
# rows all sit on one side of their crossover.
GRADING_VS_STATED_MATCH = "vs-stated-match"
# The placebo: answer shape only, no information about the game. See games.format_rubric for what is
# graded and why each component is a rule the prompt already states.
GRADING_FORMAT_ONLY = "format-only"
# Both trust gradings score the TRUSTOR's payoff and never the joint surplus, in which the return
# fraction cancels exactly (see games.payoffs.trustor_payoff). The two differ in where the return
# rate comes from: the prompt announces it, or the completion states it and its twin applies it.
GRADING_TRUSTOR_PAYOFF_STATED_RULE = "trustor-payoff-stated-rule"
GRADING_TRUSTOR_PAYOFF_SELF_RULE = "trustor-payoff-self-rule"
TRUST_GRADINGS: frozenset[str] = frozenset(
    {GRADING_TRUSTOR_PAYOFF_STATED_RULE, GRADING_TRUSTOR_PAYOFF_SELF_RULE}
)
# The minimum-effort gradings. `min-effort-group-mix` scores a level against `team_size` counterparts
# drawn from the group's own realised levels, which is the graded generalisation of matrix group-mix
# (that rule is this one with one counterpart and a two-point distribution). `level-match-return`
# scores a whole five-round match against the announced level-matcher, exactly as
# `iterated-return` scores one against a copying opponent.
#
# This game is deliberately NEVER self-graded. Under a functional twin the counterpart matches the
# completion exactly, the minimum becomes the model's own level, and the reward degenerates to
# "write the biggest number" -- so a self-graded leg here would look like the twin-PD contrast while
# measuring nothing about coordination.
GRADING_MIN_EFFORT_GROUP_MIX = "min-effort-group-mix"
GRADING_LEVEL_MATCH_RETURN = "level-match-return"
MIN_EFFORT_GRADINGS: frozenset[str] = frozenset(
    {GRADING_MIN_EFFORT_GROUP_MIX, GRADING_LEVEL_MATCH_RETURN}
)
GRADINGS: frozenset[str] = frozenset(
    {
        GRADING_GROUP_MIX,
        GRADING_SELF,
        GRADING_KEEP_FRACTION,
        GRADING_VS_FIXED_MIX,
        GRADING_ITERATED_RETURN,
        GRADING_JOINT_WELFARE_GROUP_MIX,
        GRADING_OTHER_PAYOFF_GROUP_MIX,
        GRADING_VS_STATED_MATCH,
        *NASH_DEMAND_GRADINGS,
        *THRESHOLD_GOODS_GRADINGS,
        GRADING_FORMAT_ONLY,
        *TRUST_GRADINGS,
        *MIN_EFFORT_GRADINGS,
    }
)
# A grading no baseline sweep can select prompts under: selection keeps prompts whose *action*
# distribution is mixed, and this one's reward is blind to the action. Named here rather than in
# `games.select_prompts` so the sweep CLI and the reward function read the same list.
REGRADE_ONLY_GRADINGS: frozenset[str] = frozenset({GRADING_FORMAT_ONLY})

# The care family: one grading per weight on the counterpart's payoff, `care-alpha-<a>`, paying
# `(own + a * other) / (1 + a)` against the group's realised mix. An open family rather than more
# entries in `GRADINGS` because the weight is a number, so the vocabulary is a pattern and every
# membership check has to ask `is_grading` instead of a set. The family also covers TWO row types --
# the matrix games and the announced-rule trust sender, whose arithmetics differ -- which is why the
# name carries no `-group-mix` suffix: it would claim one of them.
CARE_GRADING_PREFIX = "care-alpha-"
# The family's one legal spelling, and the reason it is a pattern with a round-trip rather than a
# parse: `format(alpha, "g")` writes 1.0 as "1", so name and weight are in bijection, and a second
# spelling of one reward ("care-alpha-1.0") would put two arms in the artifacts where one ran. Every
# readout, S3 prefix and eval-trace meta keys on this string.
_CARE_GRADING_RE = re.compile(re.escape(CARE_GRADING_PREFIX) + r"(\d+(?:\.\d+)?)")
# Which of the family's two row types a row is: an announced rate means the trust sender, and
# `STATED_RETURN_UNSET` means anything else -- a game answered with one of two labels, which the
# family scores, or one answered with a number, which it does not and which
# `_refuse_a_care_row_the_family_cannot_score` therefore names. Named here because three
# modules outside the reward function ask the same question of the same column -- the prompt sweep,
# the corpus loader and the read-back gate -- and a literal restated in each could drift from the
# `_Row` field the dispatch actually reads (`TestTheCareFamilyRowTypeColumnIsARewardColumn` pins it).
STATED_RETURN_FRACTION_COLUMN = "stated_return_fraction"

# Which counterpart-framing paragraph a row's prompt carries, on a corpus that renders one game under
# several framings. Not a `_Row` field, deliberately: the reward never reads the framing (the paragraph
# says who the counterpart is, and every scorer here grades the action against the group), so making it
# a required reward column would refuse every corpus written before the column existed for the sake of
# a value nothing scores. TRL forwards every dataset column to the reward function, so the metrics can
# read it out of the forwarded kwargs and simply produce no framing keys where the column is absent.
FRAMING_ID_COLUMN = "framing_id"
# What a row carries when it has no counterpart framing of its own: the trust sender, whose paragraph
# is the announced return rule rather than a framing the sweep can hold fixed. The empty string cannot
# collide with a framing id, since `games.prompts.FRAMING_LABEL_RE` requires a leading lowercase
# letter, and the marker is stamped rather than left absent because `games.dataset` refuses a corpus
# whose rows do not all carry the same columns.
FRAMING_ID_UNSET = ""


def care_grading(alpha: float) -> str:
    """Return the canonical grading name for one care weight.

    `care_grading(1) == care_grading(1.0) == "care-alpha-1"`. A weight that cannot be spelled as a
    decimal literal is refused rather than named: `format(1e-07, "g")` is `"1e-07"`, and a grading
    the family's own pattern would not recognise is a name nothing downstream can read back.
    """
    assert_care_alpha(alpha)
    name = f"{CARE_GRADING_PREFIX}{format(alpha, 'g')}"
    if _CARE_GRADING_RE.fullmatch(name) is None:
        raise ValueError(
            f"care alpha {alpha!r} spells {name!r}, which is not the decimal literal the family's "
            f"names are read back with ({_CARE_GRADING_RE.pattern}). Pick a weight that writes as "
            f"plain digits, optionally with a decimal point."
        )
    return name


def care_alpha_of(grading: str) -> float | None:
    """Return the care weight a canonical family name states, or None for every other string.

    None covers both non-members: a grading from `GRADINGS`, and a care-shaped name in a
    non-canonical spelling. `is_grading` is what refuses the second one by name; this function stays
    a pure lookup so the dispatch can ask it per group without a try.
    """
    match = _CARE_GRADING_RE.fullmatch(grading)
    if match is None:
        return None
    alpha = float(match.group(1))
    if care_grading(alpha) != grading:
        return None
    return alpha


def assert_canonical_care_grading(name: str) -> None:
    """Refuse a care-shaped grading written in any spelling but the canonical one.

    The whole content of "canonicalise the family to one name": rather than accepting synonyms and
    normalising them somewhere, the second spelling is an error naming the first. A corpus written
    under `care-alpha-1.0` and a run recorded under `care-alpha-1` would be two arms in every
    artifact and one arm in fact, and nothing downstream could tell them apart.
    """
    match = _CARE_GRADING_RE.fullmatch(name)
    if match is None:
        return
    canonical = care_grading(float(match.group(1)))
    if canonical != name:
        raise ValueError(
            f"grading {name!r} is the care family's reward written in a spelling that is not its "
            f"name: use {canonical!r}. One reward has one name, because every artifact, S3 prefix "
            f"and readout keys on the grading string, and two spellings would read as two arms."
        )


def is_grading(name: str) -> bool:
    """Report whether this name is a grading the reward function can score.

    The membership test every caller uses instead of `name in GRADINGS`, because the care family is
    a pattern rather than a set. Refuses a care-shaped name in a non-canonical spelling rather than
    answering False for it: "unknown grading" would send the author looking for a missing scorer
    when the fix is one character in the name.
    """
    assert_canonical_care_grading(name)
    return name in GRADINGS or care_alpha_of(name) is not None


def unknown_grading_message(name: str) -> str:
    """Return the one refusal text for a name no grading answers to.

    Shared so the reward function, the arm registry, the row renderers and both CLIs describe the
    vocabulary the same way: an open family reads as missing from a message that lists a frozenset.
    """
    return (
        f"Unknown grading {name!r}; known: {sorted(GRADINGS)}, plus the care family "
        f"{CARE_GRADING_PREFIX}<alpha> at any non-negative alpha (for example {care_grading(1)})."
    )


def grading_cli_value(value: str) -> str:
    """Validate a `--grading` argument, accepting the care family that `choices=` cannot express.

    Argparse's `choices=` is a fixed list, so a flag built from `GRADINGS` refuses every care arm
    with argparse's own terse "invalid choice" listing a vocabulary the family is missing from. Used
    as `type=`, and it raises `ArgumentTypeError` in both failure directions because argparse
    discards a `ValueError`'s message and prints its own: the canonical-spelling fix has to survive
    the trip to the operator's terminal.
    """
    try:
        recognised = is_grading(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not recognised:
        raise argparse.ArgumentTypeError(unknown_grading_message(value))
    return value


DEFAULT_PARSE_PENALTY = -1.0

# How an unparseable completion is priced. `constant` pays `parse_penalty` (the -1.0 every arm
# trained under before 2026-09-02) whatever the row; it is the default so every existing arm's
# reward is unchanged. `margin-below-worse` prices a failure ONE REACHABLE SPREAD BELOW THE WORST
# REWARD THE ROW ADMITS, at whatever counterpart distribution the group resolved, so the ranking
# right > wrong > malformed holds with EQUAL gaps and the format gradient has the task gradient's
# magnitude instead of dwarfing it.
#
# The reason it exists: track-record-v2's constant -1.0 was ten times its balanced 0.10 margins,
# landed on the long defect-leaning deliberations that failed the format, and the policy halved its
# deliberation and drifted coop-ward on the defect-pays cells against a correctly signed within-group
# gradient. The 2026-09-03 auxiliary-term audit then measured the same ratio across all 32 banked
# runs and found the constant carrying most of the gradient on many of them, so the mode was
# generalised from the one grading it was written for to every grading (its G1). A SMALLER constant is
# not the same fix: what dominates is `worst_parsed - price` against the group's own reward spread, so
# any constant is proportionate on one game at one mix and nowhere else.
#
# The care family is the case that makes the difference structural rather than a matter of degree: at
# alpha 1 on a PD-shaped row the within-group spread passes through zero at the joint-welfare
# attractor the arm is designed to settle at, so every constant diverges exactly where the arm ends up
# while this price shrinks to nothing with the task channel it is scaled against.
PARSE_PENALTY_CONSTANT = "constant"
PARSE_PENALTY_MARGIN_BELOW_WORSE = "margin-below-worse"
PARSE_PENALTY_MODES: tuple[str, ...] = (PARSE_PENALTY_CONSTANT, PARSE_PENALTY_MARGIN_BELOW_WORSE)

# The gradings that cannot price a failure against the row, and why, in the words both refusals use:
# the reward function's, before a step is scored, and `games.arms._validate_parse_penalty_mode`'s, at
# registry import. One table so the two cannot drift into two accounts of the same refusal.
PARSE_PRICE_UNDEFINED_GRADINGS: dict[str, str] = {
    GRADING_FORMAT_ONLY: (
        "its reward is a rubric over answer shape with nothing about the game in it, so the "
        "'worst reward this row admits' is a rubric score rather than a task reward and pricing the "
        "format channel one rubric spread below it would scale that channel against itself. The audit "
        "that generalised this mode (2026-09-03, G1) leaves this grading on the constant deliberately: "
        "the -1 to +1 gap across the parse boundary being wider than the whole [0, 1] rubric range is "
        "what makes the placebo honestly dominated by parse success"
    )
}


def why_a_grading_cannot_price_a_failure(grading: str) -> str | None:
    """Return why this grading has no row-relative parse price, or None where it has one.

    A function rather than a bare lookup at each site, so the care family reads as priced by the same
    question every other grading is asked: its matrix leg prices from the two actions' rewards at the
    resolved mix and its trust leg from the corner sends, at every weight.
    """
    return PARSE_PRICE_UNDEFINED_GRADINGS.get(grading)


def margin_below_worst_reachable(reachable_rewards: Sequence[float]) -> float:
    """Price a failure one reachable spread below the worst reward the row admits.

    The whole arithmetic of `PARSE_PENALTY_MARGIN_BELOW_WORSE`, in one place, so every grading's
    failure branch supplies its own answer space and none of them carries its own pricing rule. With
    the row's reachable rewards spanning [worst, best], the price is `worst - (best - worst)`, which
    makes the failure's distance below the worst answer exactly one within-group spread: the audit's
    ratio is then 1 by construction at every counterpart distribution, rather than a number that
    drifts with the mix as a constant's does.

    Two consequences worth stating because both read as bugs and are not. The price is often POSITIVE
    -- every reward here is normalised into [0, 1], so one spread below the worse answer need not
    reach zero (the softpen arm realised +0.05 to +0.60). And where the row admits no spread at all
    the price IS the worst reachable reward: the care family at alpha 1 sits exactly there at the
    joint-welfare attractor, and a format channel that vanishes with the task channel is what this
    mode is for.
    """
    worst = min(reachable_rewards)
    return worst - (max(reachable_rewards) - worst)


@dataclass(frozen=True)
class _ParsePrice:
    """What one unparseable completion was priced at, and the reachable set that priced it.

    Carried on the failure's `_Scored` so `_log_parse_price_metrics` can check the price against the
    rewards its own group realised. The set travels FROM the branch that priced the failure rather
    than being recomputed at the check, which is the whole point: a second computation of the answer
    space would agree with the first by construction and check nothing.
    """

    price: float
    # None under `PARSE_PENALTY_CONSTANT`, where the price is a run knob rather than an answer space,
    # so there is nothing for the invariant to be an invariant of.
    reachable: tuple[float, ...] | None
    # Which counterpart distribution the set was resolved at, in the branch's own words. Only ever
    # read by the guard's refusal, where it is the difference between "this branch priced the wrong
    # answer space" and "this branch priced the right one at the wrong mix".
    resolution: str
    # Whether every completion in the group was graded against the SAME counterpart distribution this
    # price was resolved at. False under leave-one-out, where each completion faces the others only:
    # the group's realised rewards are then not drawn from this row's reachable set at all, so
    # comparing them would refuse correct arithmetic. It gates the guard's RANGE half alone, which is
    # the only one that reads the group; the price identity is a property of this one call site and is
    # checked at every resolution.
    shared_resolution: bool = True


def _price_failure(
    *,
    parse_penalty: float,
    parse_penalty_mode: str,
    reachable_rewards: Callable[[], Sequence[float]],
    resolution: str,
    shared_resolution: bool = True,
) -> _ParsePrice:
    """Price one unparseable completion under the mode, from this row's own reachable rewards.

    The reachable set arrives as a callable rather than a list because the constant mode must not
    compute it: several gradings brute-force or convolve theirs, and a grading whose reachable set is
    undefined for a row would raise on a path that today pays the constant without asking.
    """
    if parse_penalty_mode == PARSE_PENALTY_CONSTANT:
        return _ParsePrice(
            price=parse_penalty,
            reachable=None,
            resolution=resolution,
            shared_resolution=shared_resolution,
        )
    reachable = tuple(reachable_rewards())
    return _ParsePrice(
        price=margin_below_worst_reachable(reachable),
        reachable=reachable,
        resolution=resolution,
        shared_resolution=shared_resolution,
    )


# The leave-one-out fallback when a group holds no other parsed completion.
UNINFORMATIVE_COOP_PRIOR = 0.5
# The same fallback for the claim game, where the quantity being estimated from the group is not a
# cooperation rate but how often a claim fits: with no other claim to compare against, half the
# time. Named separately because the two are different quantities that happen to share a value, and
# a single constant would make a later change to one silently change the other.
UNINFORMATIVE_FIT_PRIOR = 0.5
# The same fallback for the shared undertaking, where the quantity estimated from the group is how often
# the pooled figures reach the threshold: with no other figure to pool against, half the time. Named
# separately for `UNINFORMATIVE_FIT_PRIOR`'s reason -- three different quantities that happen to share a
# value, and one constant would make a later change to one silently change the others.
UNINFORMATIVE_REACH_PRIOR = 0.5
# The claim that splits the windfall evenly, as a fraction of it. The fairness anchor the claim
# game's exact-half rate counts, exact because `NashDemandSpec` refuses an odd windfall.
EQUAL_SPLIT_CLAIM_FRACTION = 0.5

# The same GRPO floor `grpo.throughput.MIN_GRPO_GROUP_SIZE` states, deliberately restated rather
# than imported: this module is on the cheap import path (the arm registry pulls it in for the
# grading vocabulary) and grpo.throughput imports torch, transformers, trl, datasets and peft, which
# is a measured 50 ms against 10 s. `TestTheGroupSizeFloorHasOneValue` pins them equal.
MIN_GENERATIONS = 2


@dataclass(frozen=True)
class _Row:
    """One dataset row as the reward function sees it, rebuilt from the parallel column lists.

    Field names are the dataset column names verbatim; `REQUIRED_REWARD_COLUMNS` is derived
    from them so the two cannot drift apart.
    """

    prompt_id: str
    grading: str
    game_id: str
    payoff_cc: float
    payoff_cd: float
    payoff_dc: float
    payoff_dd: float
    label_a: str
    label_b: str
    coop_label: str
    # The decision-matching probability this row's prompt states, or STATED_MATCH_PROB_UNSET for
    # every game whose prompt states none. Only the vs-stated-match grading reads it, and it
    # refuses the marker: a stated-track-record row without its probability is corrupt, and
    # defaulting it would grade against a track record the prompt never stated.
    stated_match_prob: float
    endowment: int
    windfall: int
    # One column shared by the shared undertaking and the minimum-effort games: in both, the count of
    # OTHER parties the prose states and the grading draws its counterparts from.
    team_size: int
    contribution_threshold: int
    prize: int
    opp_coop_prob: float
    opponent_rule: str
    n_rounds: int
    transfer_multiplier: float
    stated_return_fraction: float
    n_levels: int
    benefit_per_level: float
    cost_per_level: float

    def spec(self) -> MatrixGameSpec:
        """Rebuild the matrix game from this row's own payoff columns."""
        return MatrixGameSpec(
            game_id=self.game_id,
            payoff_cc=float(self.payoff_cc),
            payoff_cd=float(self.payoff_cd),
            payoff_dc=float(self.payoff_dc),
            payoff_dd=float(self.payoff_dd),
        )

    def demand_spec(self) -> NashDemandSpec:
        """Rebuild the simultaneous-claim division from this row's own windfall column."""
        return NashDemandSpec(game_id=self.game_id, windfall=int(self.windfall))

    def threshold_goods_spec(self) -> ThresholdGoodsSpec:
        """Rebuild the shared undertaking from this row's own stock, counterparts, bar and prize.

        Read off the row rather than rebuilt from module constants, for the reason in the module
        docstring: the corpus on disk is the ground truth for what prompt the model actually read, and
        all four of these numbers are printed in that prompt. In particular the counterpart count the
        grading draws is the same column the prose states, so a prompt saying one thing and a reward
        assuming another is not reachable from here.
        """
        return ThresholdGoodsSpec(
            game_id=self.game_id,
            endowment=int(self.endowment),
            team_size=int(self.team_size),
            contribution_threshold=int(self.contribution_threshold),
            prize=int(self.prize),
        )

    def trust_spec(self) -> TrustSpec:
        """Rebuild the trust game from this row's own stock, multiple and announced rate.

        Read off the row rather than rebuilt by calling a constructor, for the reason in the module
        docstring: the corpus on disk is the ground truth for what prompt the model actually read,
        and the multiple is printed in that prompt. A reward that took the multiple from a module
        constant would silently stop matching a corpus written before the constant moved.
        """
        return TrustSpec(
            game_id=self.game_id,
            endowment=int(self.endowment),
            multiplier=float(self.transfer_multiplier),
            stated_return_fraction=float(self.stated_return_fraction),
        )

    def min_effort_spec(self) -> MinEffortSpec:
        """Rebuild the minimum-effort game from this row's own grid, coefficients and team size.

        Read off the row rather than rebuilt by calling a constructor, for the module docstring's
        reason: the corpus on disk is the ground truth for what prompt the model actually read, and
        every one of these four numbers is printed in that prompt -- the grid size in the answer
        instruction, the benefit and cost in the outcome table, and the team size in the counterpart
        clause. `team_size` in particular is the guard that matters most here: the prose states it and
        the grading raises it as an exponent, so reading both off one column is what stops a prompt
        that says three from being graded against two, which is a lie no loss curve could show.
        """
        return MinEffortSpec(
            game_id=self.game_id,
            n_levels=int(self.n_levels),
            benefit_per_level=float(self.benefit_per_level),
            cost_per_level=float(self.cost_per_level),
            team_size=int(self.team_size),
        )


REQUIRED_REWARD_COLUMNS: tuple[str, ...] = tuple(field.name for field in dataclasses.fields(_Row))

# The one reward column an older corpus may legitimately lack, mapped to the unset marker the
# corpus loader fills in. Backfilling is safe for exactly one reason, and any column added here
# must satisfy it: the marker IS the value every current row builder writes for a game the column
# does not describe, and the only grading that reads the column REFUSES the marker -- so a
# backfilled row can never be silently graded against an invented value, while every corpus swept
# before the column existed stays trainable. `games.train.load_corpus` applies it with a logged
# notice, and the corpus preflight's sabotage tests exempt exactly this set from the
# missing-column-goes-red requirement (the red requirement is the point for every other column,
# whose absence has no single meaning).
BACKFILLABLE_REWARD_COLUMNS: dict[str, float] = {"stated_match_prob": STATED_MATCH_PROB_UNSET}


@dataclass(frozen=True)
class _Scored:
    """One completion's outcome: the reward plus everything the metrics and trace need."""

    reward: float
    parsed: bool
    detail: str
    coop_fraction: float | None = None
    # The stated-track-record grading's own readings, None everywhere else. The stated probability
    # and the signed EV margin ride into the completions parquet per completion (log_extra), so the
    # per-rung dose curve at every step is a re-analysis rather than a re-run. `at_ev_optimum` is
    # whether the parsed action is the one the row's own (payoff table, stated p) pair pays more
    # for -- the sharpest per-step reading of "did it read the clause and do the math" -- and
    # `coop_pays` is which side that is, so the cooperation rate can be split by the cell's
    # incentive direction rather than pooled into one number that averages opposite predictions.
    stated_match_prob: float | None = None
    ev_margin: float | None = None
    at_ev_optimum: bool | None = None
    coop_pays: bool | None = None
    keep_fraction: float | None = None
    claim_fraction: float | None = None
    # How often this completion's claim overreaches the counterpart distribution it was graded
    # against. The claim game's only source of downward pressure: with no crashes the reward is a
    # bare ramp in the claim, so the arm is climbing rather than stalled.
    #
    # Two fields rather than one because the same arithmetic names two different quantities: under
    # group-mix grading the counterparts are the group's realised claims, so this is a genuine mean
    # collision rate, while under self grading the counterpart IS this completion's own claim, so it
    # collapses to the deterministic indicator `claim > windfall / 2`. Logged under one key they
    # would read 0.400 on the same group of claims for unrelated reasons, and the readout would plot
    # a collision rate and an over-claiming rate as one series.
    crash_rate: float | None = None
    overclaim_rate: float | None = None
    contribution_fraction: float | None = None
    # How often the pooled figures reach the threshold for this completion. The shared undertaking's
    # gradient supply, and the one number that says whether the prize term carries any signal at all: if
    # every sample in a group clears the bar or none does, the prize is a constant and only the `-c` term
    # is left, which trains contributions to zero for a structural reason no curve distinguishes from a
    # disposition.
    #
    # Two fields for `crash_rate`/`overclaim_rate`'s reason. Under group-mix grading the counterparts are
    # drawn from the group's realised figures, so this is a genuine expected rate. Under self grading they
    # write exactly what this completion wrote, so the same arithmetic collapses to the deterministic
    # indicator `contribution >= equal_share` -- an "at least the equal share" rate that is redundant with
    # `mean_contribution_fraction`. Logged under one key they would read the same number on the same
    # figures for unrelated reasons, and the readout would plot a reach rate and a share-met rate as one
    # series.
    threshold_met_rate: float | None = None
    equal_share_met_rate: float | None = None
    # Whether this figure is above the equal share, which is waste rather than virtue here: it buys no
    # more prize and costs a unit each time. One field for both gradings, unlike the pair above, because
    # this is a property of the answer alone rather than of the answer against a counterpart distribution,
    # so it means the same thing under either rule.
    over_contribution: bool | None = None
    send_fraction: float | None = None
    # The return rate this completion promised, under the strategy method only. Logged per step
    # because the promise is free there (it is not in the trustor's own graded payoff), so a rise in
    # stated returns is promise inflation until something else says otherwise -- and a trajectory
    # nobody recorded is a re-run rather than a re-analysis.
    stated_return_fraction: float | None = None
    # Whether this completion's send is the payoff-maximising one given the rate it faces or wrote.
    # The reward is affine in the send, so the optimum is always a corner, and this is the sharpest
    # single reading of "did it multiply the number in front of it".
    #
    # Only set where the rate is EXOGENOUS, i.e. under the announced-rule gradings. Under the
    # strategy method the rate is what the completion itself wrote, so a send judged against it
    # measures the internal consistency of the (send, promise) pair and not payoff maximisation:
    # send-nothing-promise-nothing and send-everything-promise-everything would both read
    # at-optimum, three reward-units apart, and a policy collapsing to the first would score 100%
    # near the bottom of its reachable range. That form gets its own field below, judged against
    # the one strategy pair that really is optimal.
    at_payoff_optimum: bool | None = None
    at_strategy_optimum: bool | None = None
    # The minimum-effort level as a fraction of the grid, `(level - 1) / (n_levels - 1)`, so the
    # behavioural mean is on the same [0,1] scale as every other game's. The raw level rides along
    # beside it because the histogram is a question about the integers and dividing throws that away,
    # and because the target below is an integer the mean has to be read against.
    level_fraction: float | None = None
    level: int | None = None
    # The level the group's own realised distribution makes best, per group rather than per
    # completion, and the analogue of chicken's `derived.group_mix_fixed_points`: this arm's
    # pre-registration is scored against it, so a run that carries its own prediction cannot be
    # compared against the wrong one later.
    target_level: int | None = None
    # What stepping one level up would pay this completion, against its own group. The arm's
    # gradient-supply reading, in the place `crash_rate` sits for the claim game: positive means the
    # group is being pushed up the grid, negative means down, and a mean near zero means the group is
    # sitting where the payoffs want it rather than that nothing is happening. Absent at the top of
    # the grid, where there is no step to price.
    upward_pressure: float | None = None
    # Whether a repeated form's final round came in below its previous one. The diagnostic the two
    # repeated arms exist to compare: against a level-matcher, dropping the last level only ever
    # loses money, so a drop is a transferred habit rather than correct reasoning -- while against a
    # copying opponent in the PD, a last-round defection is genuinely optimal.
    end_game_drop: bool | None = None
    used_opponent_prior: bool = False
    # Which answer-shape components the format-only rubric credited, as pairs rather than a dict so
    # `_Scored` stays frozen and comparable. Present only for that grading; logging them per step is
    # how "where did this placebo's gradient come from" becomes a re-analysis instead of a re-run.
    format_components: tuple[tuple[str, float], ...] = ()
    # How an unparseable completion was priced, set by every failure branch and None on every parsed
    # one. Required rather than optional on a failure: `_log_parse_price_metrics` refuses a failure
    # that carries no price, because a branch added later that builds its own `_Scored` would
    # otherwise slip past the invariant check without a word.
    parse_price: _ParsePrice | None = None


@cache
def _cached_max_return(spec: MatrixGameSpec, rule: OpponentRule, n_rounds: int) -> float:
    """Memoise the brute-forced optimum, which is identical for every row of a given game.

    Without this, a 2**n search would rerun for all G completions of every group in every step.
    `MatrixGameSpec` is a frozen dataclass over strings and floats, so it hashes by value and
    two rows describing the same game share one cache entry.
    """
    return max_iterated_return(spec, rule, n_rounds)


@cache
def _cached_worst_return(spec: MatrixGameSpec, rule: OpponentRule, n_rounds: int) -> float:
    """Memoise the brute-forced worst match, which the row-relative parse price reads.

    Cached for `_cached_max_return`'s reason, and only ever called from the price's own callable, so
    an arm on the constant never pays for the search.
    """
    return worst_iterated_return(spec, rule, n_rounds)


def _opponent_rule_for(row: _Row) -> OpponentRule:
    """Read the row's opponent rule, treating an unknown name as corpus corruption."""
    try:
        return OpponentRule(row.opponent_rule)
    except ValueError as exc:
        raise RuntimeError(
            f"prompt_id={row.prompt_id!r} is an {GRADING_ITERATED_RETURN} row whose "
            f"opponent_rule {row.opponent_rule!r} is not one of "
            f"{sorted(rule.value for rule in OpponentRule)}."
        ) from exc


def _expected_payoff(spec: MatrixGameSpec, action: str, opponent_coop_prob: float) -> float:
    """Score one action against an opponent that cooperates with probability p."""
    return opponent_coop_prob * spec.payoff(action, COOPERATE) + (
        1.0 - opponent_coop_prob
    ) * spec.payoff(action, DEFECT)


def _parse_group_actions(rows: Sequence[_Row], visibles: Sequence[str]) -> list[str | None]:
    """Parse each completion in a group into a canonical action, or None on failure."""
    return [
        parse_action(visible, label_a=row.label_a, label_b=row.label_b, coop_label=row.coop_label)
        for row, visible in zip(rows, visibles, strict=True)
    ]


def _resolve_opponent_mix(
    actions: Sequence[str | None],
    index: int,
    *,
    opponent_coop_prob: float | None,
    group_coop_prob: float | None,
    leave_one_out: bool,
) -> tuple[float, bool]:
    """Return the cooperation rate this completion is graded against, and whether it is the prior.

    Resolved for every completion, parsed or not, which is the reorder the row-relative parse price
    needed: that price is the two actions' rewards AT THIS MIX, so a failure branch that ran before
    the mix existed could only fall back on the uninformative prior and would silently price every
    failure at a mix no group realised.

    `group_coop_prob=None` is a group in which nothing parsed at all: there is no mix to estimate, so
    the price reads the same `UNINFORMATIVE_COOP_PRIOR` the leave-one-out fallback does, and says so.
    """
    if opponent_coop_prob is not None:
        return opponent_coop_prob, False
    if leave_one_out:
        others = [
            other
            for position, other in enumerate(actions)
            if position != index and other is not None
        ]
        if others:
            return others.count(COOPERATE) / len(others), False
        return UNINFORMATIVE_COOP_PRIOR, True
    if group_coop_prob is None:
        return UNINFORMATIVE_COOP_PRIOR, True
    return group_coop_prob, False


def _matrix_reachable_rewards(
    payoff_of: Callable[[MatrixGameSpec, str, float], float],
    spec: MatrixGameSpec,
    probability: float,
) -> list[float]:
    """Return both actions' rewards at one mix, which is every reward a matrix row admits there.

    Read through the grading's own recipient function rather than through `games.payoffs`' per-grading
    spread helpers: the recipient IS the reward, so its value at the two actions is the row's
    reachable pair by definition, and a second table mapping grading to spread helper could drift
    from the recipient table the reward dispatches on.
    """
    return [payoff_of(spec, COOPERATE, probability), payoff_of(spec, DEFECT, probability)]


def _score_expected_payoff_group(  # noqa: PLR0913 - one scorer for four gradings; each knob is one grading's identity
    rows: Sequence[_Row],
    visibles: Sequence[str],
    *,
    opponent_coop_prob: float | None,
    leave_one_out: bool,
    parse_penalty: float,
    parse_penalty_mode: str,
    expected_payoff_fn: Callable[[MatrixGameSpec, str, float], float] | None = None,
) -> list[_Scored]:
    """Score a group against an opponent distribution, either the group's own or a fixed one.

    `opponent_coop_prob=None` means group-mix: the distribution is the cooperation fraction
    among the group's *parsed* completions, so an unparseable completion is excluded from the
    opponent estimate as well as penalised. A supplied probability is the frozen-opponent arm,
    where the mix was sampled once per prompt during the baseline sweep.

    `expected_payoff_fn` is which recipient's expected payoff a parsed action earns against that
    distribution -- the completion's own by default, or the ladder's joint/counterpart
    arithmetics from `games.payoffs`. One parameter rather than three near-identical scorers,
    because everything else here (mix estimation, prior accounting, parse policy) is exactly what
    the ladder must hold fixed.

    Under `leave_one_out` a group can hold no *other* parsed completion, leaving nothing to
    estimate the opponent from; that case takes `UNINFORMATIVE_COOP_PRIOR` and is counted, since
    a reward quietly computed from a prior is a substitution that never shows up in a loss curve.

    An unparseable completion is priced under the mode from the two actions' rewards at ITS OWN
    resolved mix, so the failure gradient is one within-group spread whatever the mix -- which for the
    care family at alpha 1 means it shrinks to nothing at the joint-welfare attractor, exactly where a
    constant would become the arm's whole gradient.
    """
    payoff_of = expected_payoff_fn if expected_payoff_fn is not None else _expected_payoff
    actions = _parse_group_actions(rows, visibles)
    parsed_actions = [action for action in actions if action is not None]
    group_coop_prob = (
        parsed_actions.count(COOPERATE) / len(parsed_actions) if parsed_actions else None
    )
    # Whether the whole group is graded against one distribution, which is what lets the group's
    # realised rewards check a failure's price (`_log_parse_price_metrics`). A frozen opponent is one
    # distribution by construction and the pooled group mix is one by arithmetic; leave-one-out gives
    # each completion its own, so nothing in the group is drawn from the failed row's reachable set.
    shares_the_resolution = opponent_coop_prob is not None or not leave_one_out
    scored: list[_Scored] = []
    for index, (row, action) in enumerate(zip(rows, actions, strict=True)):
        probability, used_prior = _resolve_opponent_mix(
            actions,
            index,
            opponent_coop_prob=opponent_coop_prob,
            group_coop_prob=group_coop_prob,
            leave_one_out=leave_one_out,
        )
        if action is None:
            price = _price_failure(
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
                reachable_rewards=partial(
                    _matrix_reachable_rewards, payoff_of, row.spec(), probability
                ),
                resolution=f"opponent_coop_prob={probability}",
                shared_resolution=shares_the_resolution,
            )
            scored.append(
                _Scored(
                    reward=price.price,
                    parsed=False,
                    detail="",
                    # Reported only where the price consumed the prior: under the constant the mix is
                    # not read for a failure at all, so every arm trained before this mode keeps its
                    # prior rate as well as its rewards.
                    used_opponent_prior=used_prior and parse_penalty_mode != PARSE_PENALTY_CONSTANT,
                    parse_price=price,
                )
            )
            continue
        scored.append(
            _Scored(
                reward=payoff_of(row.spec(), action, probability),
                parsed=True,
                detail=action,
                coop_fraction=float(action == COOPERATE),
                used_opponent_prior=used_prior,
            )
        )
    return scored


def _self_reachable_rewards(spec: MatrixGameSpec) -> list[float]:
    """Return both diagonal cells, which are every reward a self-graded row admits."""
    return [spec.payoff(COOPERATE, COOPERATE), spec.payoff(DEFECT, DEFECT)]


def _score_self_group(
    rows: Sequence[_Row], visibles: Sequence[str], *, parse_penalty: float, parse_penalty_mode: str
) -> list[_Scored]:
    """Score each completion against a true functional twin: the opponent plays what I play.

    The counterpart is the completion's own action, so the row's reachable rewards are its two
    diagonal cells and the row-relative price is `min(CC, DD) - |CC - DD|`.
    """
    scored: list[_Scored] = []
    for row, action in zip(rows, _parse_group_actions(rows, visibles), strict=True):
        if action is None:
            price = _price_failure(
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
                reachable_rewards=partial(_self_reachable_rewards, row.spec()),
                resolution="a functional twin: the counterpart plays what I play",
            )
            scored.append(_Scored(reward=price.price, parsed=False, detail="", parse_price=price))
            continue
        scored.append(
            _Scored(
                reward=row.spec().payoff(action, action),
                parsed=True,
                detail=action,
                coop_fraction=float(action == COOPERATE),
            )
        )
    return scored


def _price_stated_match_failure(
    row: _Row,
    spec: MatrixGameSpec,
    probability: float,
    *,
    parse_penalty: float,
    parse_penalty_mode: str,
) -> _ParsePrice:
    """Price one unparseable completion of a stated-track-record row under the parse-penalty mode.

    Under `margin-below-worse` the price is one EV margin below the worse action of THIS row's
    (displayed table, stated p) cell, so right > wrong > malformed hold with equal gaps and the
    format gradient is exactly the side-correct gradient's size. A cell with no margin has no
    worse action to sit below: the corpus audit drops such cells and `assert_stated_match_mixture`
    refuses a corpus carrying one, so reaching it here is corruption, not a case to price.

    That refusal is this grading's alone, and deliberately not the general rule: every other grading
    pays `margin_below_worst_reachable`'s zero-spread answer, the worst reachable reward itself, since
    a vanishing spread there is a reachable mix (the care family's attractor) rather than a corpus
    that should never have been built.
    """
    if parse_penalty_mode != PARSE_PENALTY_CONSTANT and stated_match_gap(spec, probability) == 0.0:
        raise RuntimeError(
            f"prompt_id={row.prompt_id!r} sits exactly on its EV crossover "
            f"(stated_match_prob={probability}), so under parse_penalty_mode="
            f"{parse_penalty_mode!r} an unparseable completion has no worse action to be priced "
            f"below. The corpus audit exists to drop such cells; rebuild with "
            f"games.track_record_corpus."
        )
    return _price_failure(
        parse_penalty=parse_penalty,
        parse_penalty_mode=parse_penalty_mode,
        reachable_rewards=lambda: [
            stated_match_expected_payoff(spec, COOPERATE, probability),
            stated_match_expected_payoff(spec, DEFECT, probability),
        ],
        resolution=f"stated_match_prob={probability}",
    )


def _score_stated_match_group(
    rows: Sequence[_Row],
    visibles: Sequence[str],
    *,
    parse_penalty: float,
    parse_penalty_mode: str = PARSE_PENALTY_CONSTANT,
) -> list[_Scored]:
    """Score each completion as its expected payoff under the row's own stated match probability.

    `p * payoff(a, a) + (1 - p) * payoff(a, other)` -- the exact expectation, never a sampled
    counterpart, so the reward carries no sampling noise and the stated figure IS the graded
    figure (informational honesty: one column feeds both the prompt's clause and this arithmetic).
    Deterministic per (row, action) like `self` grading, but NOT a fixed function of the action:
    the corpus mixes p across each game's EV crossover, so which action pays more varies row by
    row and only the clause says which -- the counterpart-blindness that self grading taught
    cannot pay here.

    A row carrying the unset marker (or any probability outside [0, 1]) is corpus corruption, the
    same refusal `vs-fixed-mix` makes for an unsampled `opp_coop_prob`: grading it against an
    invented track record would produce plausible rewards for a game the prompt never stated.

    An unparseable completion is priced by `_price_stated_match_failure` under the mode, and it
    carries its cell's stated probability and signed margin into the trace like a parsed answer:
    which side a failed completion fell on was the one training-side question v2's parquets
    could not answer without a join back to the corpus.
    """
    scored: list[_Scored] = []
    for row, action in zip(rows, _parse_group_actions(rows, visibles), strict=True):
        probability = float(row.stated_match_prob)
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(
                f"prompt_id={row.prompt_id!r} is a {GRADING_VS_STATED_MATCH} row whose "
                f"stated_match_prob is {probability} ({STATED_MATCH_PROB_UNSET} is the marker for "
                f"a prompt that states no track record). There is no stated correlation to grade "
                f"against; build the corpus with games.track_record_corpus."
            )
        spec = row.spec()
        if action is None:
            price = _price_stated_match_failure(
                row,
                spec,
                probability,
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
            )
            scored.append(
                _Scored(
                    reward=price.price,
                    parsed=False,
                    detail="",
                    stated_match_prob=probability,
                    ev_margin=stated_match_gap(spec, probability),
                    parse_price=price,
                )
            )
            continue
        optimal = stated_match_optimal_action(spec, probability)
        scored.append(
            _Scored(
                reward=stated_match_expected_payoff(spec, action, probability),
                parsed=True,
                detail=action,
                coop_fraction=float(action == COOPERATE),
                stated_match_prob=probability,
                ev_margin=stated_match_gap(spec, probability),
                at_ev_optimum=None if optimal is None else action == optimal,
                coop_pays=None if optimal is None else optimal == COOPERATE,
            )
        )
    return scored


def _score_keep_fraction_group(
    rows: Sequence[_Row], visibles: Sequence[str], *, parse_penalty: float, parse_penalty_mode: str
) -> list[_Scored]:
    """Score dictator rows as the fraction of the endowment kept.

    Cooperation is undefined here -- there is no opponent and no action pair -- so these rows
    contribute to the keep-fraction metric instead of the cooperation rate.

    The reward IS the kept fraction, so the answer grid's ends are the reachable rewards 0 and 1 and
    the row-relative price is -1.0: the same number the constant pays, which is the honest answer for
    this grading rather than something smaller. Nothing about this reward is compressed, so there is
    no dominance for a row-relative price to relieve.
    """
    scored: list[_Scored] = []
    for row, visible in zip(rows, visibles, strict=True):
        kept = parse_split(visible, endowment=int(row.endowment))
        if kept is None:
            price = _price_failure(
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
                reachable_rewards=lambda: [PAYOFF_MIN, PAYOFF_MAX],
                resolution="no counterpart at all: the reward is the kept fraction itself",
            )
            scored.append(_Scored(reward=price.price, parsed=False, detail="", parse_price=price))
            continue
        fraction = kept / int(row.endowment)
        scored.append(
            _Scored(reward=fraction, parsed=True, detail=f"kept={kept}", keep_fraction=fraction)
        )
    return scored


def _parse_group_claims(rows: Sequence[_Row], visibles: Sequence[str]) -> list[int | None]:
    """Parse each completion in a group into a claim, or None on failure."""
    return [
        parse_claim(visible, windfall=int(row.windfall))
        for row, visible in zip(rows, visibles, strict=True)
    ]


def _nash_demand_counterparts(
    claims: Sequence[int | None], index: int, *, grading: str, leave_one_out: bool
) -> list[int]:
    """Return the claims one completion is graded against, which is the whole contrast of the pair.

    Three rules over one reward function, rather than two reward functions. Under
    `nash-demand-self` the counterpart is a copy of this policy answering this prompt, so the
    distribution is the completion's own claim and nothing else. Under `nash-demand-group-mix` it is
    the group's realised claims -- all of them, or the others only under `leave_one_out`, which is the
    same choice matrix group-mix grading makes about the opponent mix.
    """
    own = claims[index]
    if grading == GRADING_NASH_DEMAND_SELF:
        return [] if own is None else [own]
    return [
        other
        for position, other in enumerate(claims)
        if other is not None and not (leave_one_out and position == index)
    ]


def _nash_demand_reward(spec: NashDemandSpec, claim: int, counterparts: Sequence[int]) -> float:
    """Return one claim's reward against this counterpart distribution, prior fallback included.

    Factored out of the scoring loop because the row-relative parse price asks the same question of
    every claim on the grid: what would this row have paid for that answer. One function so the
    reachable set and the realised reward cannot drift into two arithmetics.
    """
    if counterparts:
        return nash_demand_group_reward(spec, claim, counterparts)
    return nash_demand_share(spec, claim) * UNINFORMATIVE_FIT_PRIOR


def _nash_demand_reachable_rewards(
    spec: NashDemandSpec, counterparts: Sequence[int], *, self_graded: bool
) -> list[float]:
    """Return what every claim on the grid would have paid, at the distribution this row resolved.

    Under the self-graded leg the counterpart is a copy of the answer itself, so each candidate claim
    is graded against its own copy; under the group-mix leg the distribution is the group's parsed
    claims, which a failed row is outside of either way.
    """
    return [
        _nash_demand_reward(spec, claim, [claim] if self_graded else counterparts)
        for claim in spec.claims
    ]


def _score_nash_demand_group(  # noqa: PLR0913 - one scorer for both claim gradings; each knob is one grading's identity
    rows: Sequence[_Row],
    visibles: Sequence[str],
    *,
    grading: str,
    leave_one_out: bool,
    parse_penalty: float,
    parse_penalty_mode: str,
) -> list[_Scored]:
    """Score a group of claims under whichever of the two claim gradings its rows carry.

    The reward is the claimed fraction of the windfall times how often that claim fits inside the
    counterpart distribution `_nash_demand_counterparts` supplies. A group whose figures all fit sees
    a monotone ramp in the claim, and one where some overreach sees the ramp cut where they do, which
    is the only thing that ever makes a bigger claim pay less. Unparseable completions are excluded
    from the counterpart distribution as well as penalised, exactly as under matrix group-mix grading.

    A `leave_one_out` group holding no other parsed claim takes `UNINFORMATIVE_FIT_PRIOR` and is
    counted, since a reward quietly computed from a prior is a substitution that never shows up in a
    loss curve. Self grading cannot reach that case: the counterpart is the completion's own figure.

    An unparseable completion is priced under the mode against the whole claim grid at the
    distribution this row resolved, so its answer space rather than a constant sets the format
    gradient.
    """
    claims = _parse_group_claims(rows, visibles)
    self_graded = grading == GRADING_NASH_DEMAND_SELF
    scored: list[_Scored] = []
    for index, (row, claim) in enumerate(zip(rows, claims, strict=True)):
        spec = row.demand_spec()
        counterparts = _nash_demand_counterparts(
            claims, index, grading=grading, leave_one_out=leave_one_out
        )
        if claim is None:
            price = _price_failure(
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
                reachable_rewards=partial(
                    _nash_demand_reachable_rewards,
                    spec,
                    counterparts,
                    self_graded=self_graded,
                ),
                resolution=(
                    "each claim against its own copy"
                    if self_graded
                    else f"counterpart_claims={sorted(counterparts)}"
                ),
                shared_resolution=self_graded or not leave_one_out,
            )
            scored.append(_Scored(reward=price.price, parsed=False, detail="", parse_price=price))
            continue
        reward = _nash_demand_reward(spec, claim, counterparts)
        if counterparts:
            overreach = nash_demand_crash_probability(spec, claim, counterparts)
            used_prior = False
        else:
            overreach = 1.0 - UNINFORMATIVE_FIT_PRIOR
            used_prior = True
        scored.append(
            _Scored(
                reward=reward,
                parsed=True,
                detail=f"claim={claim}",
                claim_fraction=nash_demand_share(spec, claim),
                crash_rate=None if self_graded else overreach,
                overclaim_rate=overreach if self_graded else None,
                used_opponent_prior=used_prior,
            )
        )
    return scored


def _parse_group_contributions(rows: Sequence[_Row], visibles: Sequence[str]) -> list[int | None]:
    """Parse each completion in a group into a contribution, or None on failure."""
    return [
        parse_contribution(visible, endowment=int(row.endowment))
        for row, visible in zip(rows, visibles, strict=True)
    ]


def _threshold_goods_counterparts(
    contributions: Sequence[int | None], index: int, *, grading: str, leave_one_out: bool
) -> list[int]:
    """Return the figures one completion's counterparts are drawn from, the pair's whole contrast.

    Under `threshold-goods-self` the counterparts write exactly what this completion wrote, so the
    distribution is its own figure and nothing else, and the pot is the equal-share arithmetic. Under
    `threshold-goods-group-mix` they are drawn from the group's realised figures -- all of them, or the
    others only under `leave_one_out`, which is the same choice matrix group-mix grading makes about the
    opponent mix. Three rules over one reward function rather than two reward functions, so the pair
    cannot drift into two arithmetics.
    """
    own = contributions[index]
    if grading == GRADING_THRESHOLD_GOODS_SELF:
        return [] if own is None else [own]
    return [
        other
        for position, other in enumerate(contributions)
        if other is not None and not (leave_one_out and position == index)
    ]


def _threshold_goods_reach(
    spec: ThresholdGoodsSpec, contribution: int, counterparts: Sequence[int]
) -> float:
    """Return the chance the undertaking goes ahead, or the prior where the group offers no figures.

    The one place the empty-distribution fallback lives, because the row-relative parse price asks the
    same question of every figure on the grid and a second copy of the fallback could drift from the
    one the realised reward uses.
    """
    if counterparts:
        return threshold_goods_reach_probability(spec, contribution, counterparts)
    return UNINFORMATIVE_REACH_PRIOR


def _threshold_goods_reachable_rewards(
    spec: ThresholdGoodsSpec, counterparts: Sequence[int], *, self_graded: bool
) -> list[float]:
    """Return what every figure on the grid would have paid, at the distribution this row resolved.

    Under the self-graded leg the counterparts write exactly what the completion wrote, so each
    candidate figure is graded against its own copies; under the group-mix leg the distribution is the
    group's parsed figures, which a failed row is outside of either way.
    """
    return [
        threshold_goods_reward_at_reach(
            spec,
            contribution=contribution,
            reach_probability=_threshold_goods_reach(
                spec, contribution, [contribution] if self_graded else counterparts
            ),
            max_prize=THRESHOLD_GOODS_MAX_PRIZE,
        )
        for contribution in spec.contributions
    ]


def _score_threshold_goods_group(  # noqa: PLR0913 - one scorer for both undertaking gradings; each knob is one grading's identity
    rows: Sequence[_Row],
    visibles: Sequence[str],
    *,
    grading: str,
    leave_one_out: bool,
    parse_penalty: float,
    parse_penalty_mode: str,
) -> list[_Scored]:
    """Score a group of contributions under whichever of the two shared-undertaking gradings it carries.

    The reward is what the completion held back plus the prize weighted by how often the pooled figures
    reach the threshold, all over the game's shared ceiling. That makes the optimum INTERIOR wherever the
    others are near the equal share: below the share the undertaking does not happen and the units are
    simply thrown away, at it the prize arrives for the smallest outlay that buys it, and above it the
    prize is unchanged while every further unit is spent. No other grading on the slate can be worse for
    contributing more, which is what makes this the one arm able to separate cooperating from maximising.

    Unparseable completions are excluded from the counterpart distribution as well as penalised, exactly
    as under matrix group-mix grading. A `leave_one_out` group holding no other parsed figure takes
    `UNINFORMATIVE_REACH_PRIOR` and is counted, since a reward quietly computed from a prior is a
    substitution that never shows up in a loss curve. Self grading cannot reach that case: the
    counterparts are the completion's own figure.

    An unparseable completion is priced under the mode against the whole contribution grid at the
    distribution this row resolved. That matters more here than anywhere else on the slate: this
    reward's reachable span is narrow (0.318 at the low prize), so the -1.0 constant is several times
    the whole task channel.
    """
    contributions = _parse_group_contributions(rows, visibles)
    self_graded = grading == GRADING_THRESHOLD_GOODS_SELF
    scored: list[_Scored] = []
    for index, (row, contribution) in enumerate(zip(rows, contributions, strict=True)):
        spec = row.threshold_goods_spec()
        counterparts = _threshold_goods_counterparts(
            contributions, index, grading=grading, leave_one_out=leave_one_out
        )
        if contribution is None:
            price = _price_failure(
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
                reachable_rewards=partial(
                    _threshold_goods_reachable_rewards,
                    spec,
                    counterparts,
                    self_graded=self_graded,
                ),
                resolution=(
                    "each figure against its own copies"
                    if self_graded
                    else f"counterpart_contributions={sorted(counterparts)}"
                ),
                shared_resolution=self_graded or not leave_one_out,
            )
            scored.append(_Scored(reward=price.price, parsed=False, detail="", parse_price=price))
            continue
        reached = _threshold_goods_reach(spec, contribution, counterparts)
        used_prior = not counterparts
        scored.append(
            _Scored(
                reward=threshold_goods_reward_at_reach(
                    spec,
                    contribution=contribution,
                    reach_probability=reached,
                    max_prize=THRESHOLD_GOODS_MAX_PRIZE,
                ),
                parsed=True,
                detail=f"contributed={contribution}",
                contribution_fraction=contribution / spec.endowment,
                threshold_met_rate=None if self_graded else reached,
                equal_share_met_rate=reached if self_graded else None,
                over_contribution=threshold_goods_over_contribution(spec, contribution),
                used_opponent_prior=used_prior,
            )
        )
    return scored


def _level_fraction(spec: MinEffortSpec, level: int) -> float:
    """Return a level's position on the grid in [0,1], so the mean reads on every game's scale.

    `(level - 1) / (n_levels - 1)`, which puts the bottom of the grid at 0 and the top at 1 rather
    than the level's share of its own index. `MinEffortSpec` refuses a one-level grid, so the
    denominator cannot be zero.
    """
    return (level - 1) / (spec.n_levels - 1)


def _min_effort_counterparts(
    levels: Sequence[int | None], index: int, *, leave_one_out: bool
) -> list[int]:
    """Return the levels one completion's counterparts are drawn from.

    The group's realised levels -- all of them, or the others only under `leave_one_out`, which is the
    same choice matrix group-mix grading makes about the opponent mix. Unparseable completions are
    excluded from the distribution as well as penalised, exactly as they are there: the mix is 2/3,
    not 2/4.
    """
    return [
        other
        for position, other in enumerate(levels)
        if other is not None and not (leave_one_out and position == index)
    ]


def _min_effort_reachable_rewards(spec: MinEffortSpec, counterparts: Sequence[int]) -> list[float]:
    """Return what every level on the grid would have paid against this counterpart distribution."""
    return [
        min_effort_group_reward(spec, own_level=level, counterpart_levels=counterparts)
        for level in spec.levels
    ]


def _score_min_effort_group(
    rows: Sequence[_Row],
    visibles: Sequence[str],
    *,
    leave_one_out: bool,
    parse_penalty: float,
    parse_penalty_mode: str,
) -> list[_Scored]:
    """Score a group of levels against counterparts drawn from the group's own realised levels.

    The graded generalisation of matrix group-mix grading: `team_size` counterparts drawn
    independently from the group's own distribution, so today's binary rule is this one with a single
    counterpart and a two-point distribution. Exact and closed-form, never sampled -- the raw payoff
    is affine in the minimum, so one expectation over the distribution of that minimum is the whole
    calculation (`games.payoffs.min_effort_expected_lowest`).

    Never self-graded, which is a property of the registry rather than of this function: against a
    functional twin the minimum is the completion's own level and the reward collapses to "write the
    biggest number". `games.arms` carries the same note where a self-graded leg would be registered.

    A `leave_one_out` group holding no other parsed level takes the uniform prior over the grid and is
    counted, since a reward quietly computed from a prior is a substitution that never shows up in a
    loss curve.

    An unparseable completion is priced under the mode against the whole level grid at the same
    distribution, which is the analogue of the matrix family's two actions at the resolved mix.
    """
    levels = [
        parse_level(visible, n_levels=int(row.n_levels))
        for row, visible in zip(rows, visibles, strict=True)
    ]
    scored: list[_Scored] = []
    for index, (row, level) in enumerate(zip(rows, levels, strict=True)):
        spec = row.min_effort_spec()
        counterparts = _min_effort_counterparts(levels, index, leave_one_out=leave_one_out)
        used_prior = not counterparts
        if used_prior:
            counterparts = list(spec.levels)
        if level is None:
            price = _price_failure(
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
                reachable_rewards=partial(_min_effort_reachable_rewards, spec, counterparts),
                resolution=f"counterpart_levels={sorted(counterparts)}",
                shared_resolution=not leave_one_out,
            )
            scored.append(_Scored(reward=price.price, parsed=False, detail="", parse_price=price))
            continue
        gaps = min_effort_pressure_gaps(spec, counterparts)
        scored.append(
            _Scored(
                reward=min_effort_group_reward(
                    spec, own_level=level, counterpart_levels=counterparts
                ),
                parsed=True,
                detail=f"level={level}",
                level=level,
                level_fraction=_level_fraction(spec, level),
                target_level=min_effort_best_response(spec, counterparts),
                upward_pressure=gaps.get(level),
                used_opponent_prior=used_prior,
            )
        )
    return scored


def _level_match_reachable_rewards(spec: MinEffortSpec, n_rounds: int) -> list[float]:
    """Return the reachable range of a match reward: the worst sequence's share of the optimum, to 1.

    Both repeated forms normalise a simulated total by its brute-forced optimum, so those two values
    bound every reward the row admits. Read through the memoised searches, and only from the
    row-relative price's own callable, so the constant mode never pays for a search it does not use.
    """
    return [
        worst_min_effort_match_return(spec, n_rounds) / max_min_effort_match_return(spec, n_rounds),
        PAYOFF_MAX,
    ]


def _score_level_match_group(
    rows: Sequence[_Row], visibles: Sequence[str], *, parse_penalty: float, parse_penalty_mode: str
) -> list[_Scored]:
    """Score a whole match per completion: the simulated total over the brute-forced optimum.

    The announced counterpart works at whatever level the completion wrote in the round before,
    starting at the bottom of the grid, so the match is simulated exactly from the rule the prompt
    states -- no opponent infrastructure, the same trick the matrix repeated arms use.

    Why this arm is the sharpest diagnostic on the slate: against a level-matcher, dropping the final
    round's level only ever costs money, because the last round pays
    `benefit * min(e5, e4) - cost * e5` and that is maximised at `e5 = e4` whenever the benefit beats
    the cost. So `end_game_drop` here is not a description of play, it is a reading on whether a habit
    transferred -- in the iterated PD against a copying opponent, the same move is optimal.

    An unparseable completion is priced under the mode from the worst reachable match rather than from
    a constant, which is the same brute-force search the normalisation already runs, memoised.
    """
    scored: list[_Scored] = []
    for row, visible in zip(rows, visibles, strict=True):
        spec = row.min_effort_spec()
        n_rounds = int(row.n_rounds)
        levels = parse_level_sequence(visible, n_rounds=n_rounds, n_levels=spec.n_levels)
        if levels is None:
            price = _price_failure(
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
                reachable_rewards=partial(_level_match_reachable_rewards, spec, n_rounds),
                resolution=f"the announced level-matching rule over {n_rounds} rounds",
            )
            scored.append(_Scored(reward=price.price, parsed=False, detail="", parse_price=price))
            continue
        scored.append(
            _Scored(
                reward=min_effort_match_reward(spec, levels),
                parsed=True,
                detail="".join(str(level) for level in levels),
                level=levels[-1],
                level_fraction=sum(_level_fraction(spec, level) for level in levels) / len(levels),
                end_game_drop=levels[-1] < levels[-2] if len(levels) > 1 else None,
            )
        )
    return scored


def _refuse_a_grading_that_cannot_price_a_failure(grading: str, parse_penalty_mode: str) -> None:
    """Refuse the row-relative mode for a grading with no reward the row's answers bound.

    Raised before the group is scored rather than at its first failure, because a step whose
    completions all parsed would otherwise train happily under a mode the grading cannot honour and
    the run record would name a price nothing paid. `games.arms` refuses the same pairing at registry
    import, so reaching this is a reward called outside an arm.
    """
    if parse_penalty_mode == PARSE_PENALTY_CONSTANT:
        return
    reason = why_a_grading_cannot_price_a_failure(grading)
    if reason is None:
        return
    raise RuntimeError(
        f"grading {grading!r} cannot price a failure under parse_penalty_mode="
        f"{parse_penalty_mode!r}, because {reason}. Train it under "
        f"{PARSE_PENALTY_CONSTANT!r}."
    )


def _score_format_only_group(
    rows: Sequence[_Row], visibles: Sequence[str], *, parse_penalty: float, parse_penalty_mode: str
) -> list[_Scored]:
    """Score each completion on answer shape alone, with nothing about the game in the reward.

    The action is still parsed and still reported as `coop_fraction`, for two independent reasons.
    Scientifically it is the point of the arm: what the placebo's *behaviour* does while its reward
    ignores behaviour is the measurement, so throwing the action away would leave the arm unable to
    answer its own question. Mechanically `games.train.required_metrics_for` demands `coop_rate` for
    every grading but the dictator's, so a scorer that set no behavioural field would fail the
    post-run read-back gate after the GPU had been paid for.

    A completion that does not parse takes the parse penalty rather than a low rubric score, exactly
    as under every other grading. That keeps this reward honestly dominated by parse success -- the
    -1 to +1 gap across the parse boundary is wider than the whole [0, 1] rubric range -- while the
    shape components break ties among the completions that did answer, which is where the arm's
    within-group gradient has to come from once parse failure is rare.

    The one grading the row-relative price is refused for, and by that argument: dominance by parse
    success is this arm's construction rather than a defect to relieve.
    """
    _refuse_a_grading_that_cannot_price_a_failure(GRADING_FORMAT_ONLY, parse_penalty_mode)
    scored: list[_Scored] = []
    for row, visible in zip(rows, visibles, strict=True):
        labels = {"label_a": row.label_a, "label_b": row.label_b}
        action = parse_action(visible, coop_label=row.coop_label, **labels)
        if action is None:
            # The one failure priced without asking the mode, because the refusal above leaves only
            # the constant reachable here, and there is no answer space to derive a price from: the
            # reward is a rubric over shape. Recorded as a `_ParsePrice` all the same, so the guard
            # sees a failure that was priced rather than one that recorded nothing.
            price = _ParsePrice(
                price=parse_penalty,
                reachable=None,
                resolution="the answer-shape rubric, which has no counterpart distribution",
            )
            scored.append(_Scored(reward=price.price, parsed=False, detail="", parse_price=price))
            continue
        components = format_component_scores(visible, **labels)
        scored.append(
            _Scored(
                reward=format_reward(visible, **labels),
                parsed=True,
                detail=action,
                coop_fraction=float(action == COOPERATE),
                format_components=tuple(components.items()),
            )
        )
    return scored


def _trust_scored(
    spec: TrustSpec,
    *,
    sent: int,
    return_fraction: float,
    max_return_fraction: float,
    detail: str,
) -> _Scored:
    """Score one parsed trust completion, and read its behaviour off the same two numbers.

    Which optimum reading is meaningful follows from the spec rather than from an argument, so it
    cannot be passed wrongly: a spec that announces a rate fixes the optimum send at a corner, and
    "did it go to that corner" is the reading. A strategy-method spec announces none, because the
    completion writes the rate too, so the only unambiguous optimum is the reward-maximising PAIR --
    send everything and promise everything back, which the graded payoff pays for since the promise
    comes out of the counterpart's stock.
    """
    rate_is_the_completions_own = not spec.announces_a_return_rate
    break_even = trust_break_even_return_fraction(spec)
    optimal_send = spec.endowment if return_fraction > break_even else 0
    at_strategy_optimum = sent == spec.endowment and return_fraction >= max_return_fraction
    return _Scored(
        reward=trustor_reward(
            spec,
            sent=sent,
            return_fraction=return_fraction,
            max_return_fraction=max_return_fraction,
        ),
        parsed=True,
        detail=detail,
        send_fraction=sent / spec.endowment,
        at_payoff_optimum=None if rate_is_the_completions_own else sent == optimal_send,
        at_strategy_optimum=at_strategy_optimum if rate_is_the_completions_own else None,
    )


def _corner_reward_values(corners: Mapping[Any, float]) -> list[float]:
    """Return a corner table's rewards as the sequence the parse price prices against.

    The trust games' reachable rewards arrive keyed by the answer that reaches them, because the
    optimum readings need the key; the price needs only the values.
    """
    return list(corners.values())


def _stated_rule_reachable_rewards(spec: TrustSpec) -> list[float]:
    """Return the announced-rule row's two corner rewards, which bound every send it admits."""
    return _corner_reward_values(
        trust_stated_rule_corner_rewards(spec, max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION)
    )


def _self_rule_reachable_rewards(spec: TrustSpec) -> list[float]:
    """Return the strategy method's four corner rewards, over both numbers the completion writes."""
    return _corner_reward_values(
        trust_self_rule_corner_rewards(
            spec, max_return_fraction=TRUST_MAX_SELF_STATED_RETURN_FRACTION
        )
    )


def _score_trustor_stated_rule_group(
    rows: Sequence[_Row], visibles: Sequence[str], *, parse_penalty: float, parse_penalty_mode: str
) -> list[_Scored]:
    """Score each completion as the trustor's payoff against the rate its own prompt announced.

    One number in the prompt decides the sign of the optimum -- send everything above the break-even
    return fraction, nothing below it -- so the two registered variants of this game ask the same
    prose for opposite answers. There is no opponent mix to estimate and no group coupling at all:
    the counterpart's rule is stated, which is what makes this the sharper of the two forms.

    The reward is affine in the amount sent, so the two corner sends are the reachable rewards the
    row-relative price sits below.
    """
    scored: list[_Scored] = []
    for row, visible in zip(rows, visibles, strict=True):
        spec = row.trust_spec()
        if not spec.announces_a_return_rate:
            raise RuntimeError(
                f"prompt_id={row.prompt_id!r} is a {GRADING_TRUSTOR_PAYOFF_STATED_RULE} row whose "
                f"stated_return_fraction is {STATED_RETURN_UNSET} -- the marker for a form that "
                f"announces no rate. Its prompt states a rate the corpus did not record, so there "
                f"is no rule to grade against, and reading the marker as a rate would pay every "
                f"completion for sending nothing while the prompt promised a return."
            )
        sent = parse_send(visible, endowment=spec.endowment)
        if sent is None:
            price = _price_failure(
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
                reachable_rewards=partial(_stated_rule_reachable_rewards, spec),
                resolution=f"the announced return rate {spec.stated_return_fraction}",
            )
            scored.append(_Scored(reward=price.price, parsed=False, detail="", parse_price=price))
            continue
        scored.append(
            _trust_scored(
                spec,
                sent=sent,
                return_fraction=spec.stated_return_fraction,
                max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
                detail=f"sent={sent}",
            )
        )
    return scored


def _score_trustor_care_group(
    rows: Sequence[_Row],
    visibles: Sequence[str],
    *,
    alpha: float,
    parse_penalty: float,
    parse_penalty_mode: str,
) -> list[_Scored]:
    """Score an announced-rule trust group under the care family, at this row's own announced rate.

    The trust member of `care-alpha-<a>`: `(own + a * other) / ((1 + a) * ceiling)`, with the
    ceiling `trustor_payoff_ceiling` unchanged, so at alpha 0 every reward equals
    `trustor-payoff-stated-rule`'s and the two arms of the control pair share one scale.

    The optimum reading moves with the weight and is derived rather than restated: at alpha 0 the
    corner the prompt fixes is the optimum (send everything above the break-even rate, nothing
    below), while at alpha 1 the announced rate cancels out of `own + other = E + (m - 1) * s`
    entirely and sending everything is optimal at both registered rates. So `at_payoff_optimum`
    compares the two corner rewards under this alpha instead of comparing the rate to the break-even
    -- otherwise the metric would report the own-payoff optimum while the reward paid the care one.

    Only ever reached for a row whose rate is set, because the announced rate IS the dispatch's
    discriminator. That is also why the family cannot repeat `trustor-payoff-stated-rule`'s refusal of
    the no-rate marker as a rate the corpus lost: under the family the marker also means "a game
    answered with one of two labels", so it can never be read as a rate here. What refuses a
    rate-stripped trust row is `_refuse_a_care_row_the_family_cannot_score`, because such a row prints
    no action labels either and the matrix path it would otherwise fall through to reports corrupt
    labels rather than the missing rate.
    """
    scored: list[_Scored] = []
    for row, visible in zip(rows, visibles, strict=True):
        spec = row.trust_spec()
        assert_trust_care_spec(
            spec, max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION, alpha=alpha
        )
        corners = trust_care_corner_rewards(
            spec, max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION, alpha=alpha
        )
        optimal_send = spec.endowment if corners[spec.endowment] > corners[0] else 0
        sent = parse_send(visible, endowment=spec.endowment)
        if sent is None:
            price = _price_failure(
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
                # The corners this row's optimum reading is already derived from, so the price and the
                # metric cannot disagree about what the row could pay.
                reachable_rewards=partial(_corner_reward_values, corners),
                resolution=f"the announced return rate {spec.stated_return_fraction} at alpha {alpha}",
            )
            scored.append(_Scored(reward=price.price, parsed=False, detail="", parse_price=price))
            continue
        scored.append(
            _Scored(
                reward=trustor_care_reward(
                    spec,
                    sent=sent,
                    return_fraction=spec.stated_return_fraction,
                    max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
                    alpha=alpha,
                ),
                parsed=True,
                detail=f"sent={sent}",
                send_fraction=sent / spec.endowment,
                at_payoff_optimum=sent == optimal_send,
            )
        )
    return scored


def _score_trustor_self_rule_group(
    rows: Sequence[_Row], visibles: Sequence[str], *, parse_penalty: float, parse_penalty_mode: str
) -> list[_Scored]:
    """Score the strategy method: the completion states a send AND a return rule, the twin applies it.

    Both tags or neither, so a send is never graded against an invented rule. The degeneracy is real
    and deliberate rather than a bug to fix here: the return the model promises is paid out of the
    OTHER side's stock, so it costs the model's own graded payoff nothing, and the optimum is to send
    everything while promising everything back. That is why the promised rate is logged per
    completion -- a rise in stated returns is promise inflation until the never-trained trustee-role
    item says otherwise.

    The reward is affine in both numbers, so the four corners of the (send, promised share) grid are
    the reachable rewards the row-relative price sits below -- and they span the whole [0, 1] here,
    which is why this grading's price is the -1.0 the constant already paid.
    """
    scored: list[_Scored] = []
    for row, visible in zip(rows, visibles, strict=True):
        spec = row.trust_spec()
        strategy = parse_trust_strategy(visible, endowment=spec.endowment)
        if strategy is None:
            price = _price_failure(
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
                reachable_rewards=partial(_self_rule_reachable_rewards, spec),
                resolution="the strategy method: the twin applies the rule this answer writes",
            )
            scored.append(_Scored(reward=price.price, parsed=False, detail="", parse_price=price))
            continue
        return_fraction = strategy.return_percentage / RETURN_PERCENTAGE_MAX
        item = _trust_scored(
            spec,
            sent=strategy.sent,
            return_fraction=return_fraction,
            max_return_fraction=TRUST_MAX_SELF_STATED_RETURN_FRACTION,
            detail=f"sent={strategy.sent} returned={strategy.return_percentage}%",
        )
        scored.append(dataclasses.replace(item, stated_return_fraction=return_fraction))
    return scored


def _iterated_reachable_rewards(
    spec: MatrixGameSpec, rule: OpponentRule, n_rounds: int
) -> list[float]:
    """Return the reachable range of an iterated reward: the worst match's share of the optimum, to 1.

    The same construction `_level_match_reachable_rewards` reads for the level grid, and both searches
    are memoised, so a failure costs one lookup rather than a brute force per row.
    """
    return [
        _cached_worst_return(spec, rule, n_rounds) / _cached_max_return(spec, rule, n_rounds),
        PAYOFF_MAX,
    ]


def _score_iterated_group(
    rows: Sequence[_Row], visibles: Sequence[str], *, parse_penalty: float, parse_penalty_mode: str
) -> list[_Scored]:
    """Score a whole match per completion: simulated return over the brute-forced optimum.

    Normalising by the optimum puts the reward on the same [0,1] scale as the one-shot arms, so
    the parse penalty stays commensurable across every arm -- and under the row-relative mode the
    price is one reachable range below the worst match this rule admits, rather than a constant.
    """
    scored: list[_Scored] = []
    for row, visible in zip(rows, visibles, strict=True):
        rule = _opponent_rule_for(row)
        n_rounds = int(row.n_rounds)
        spec = row.spec()
        best = _cached_max_return(spec, rule, n_rounds)
        moves = parse_action_sequence(
            visible,
            n_rounds=n_rounds,
            label_a=row.label_a,
            label_b=row.label_b,
            coop_label=row.coop_label,
        )
        if moves is None:
            price = _price_failure(
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
                reachable_rewards=partial(_iterated_reachable_rewards, spec, rule, n_rounds),
                resolution=f"the announced {rule.value} rule over {n_rounds} rounds",
            )
            scored.append(_Scored(reward=price.price, parsed=False, detail="", parse_price=price))
            continue
        scored.append(
            _Scored(
                reward=simulate_iterated(spec, rule, moves) / best,
                parsed=True,
                detail="".join(moves),
                coop_fraction=moves.count(COOPERATE) / len(moves),
                # The same reading the level-matcher arm reports, so the two repeated forms are
                # directly comparable: a final round less cooperative than the one before it. Here it
                # is the classic end-game defection, which against a copying opponent is OPTIMAL --
                # which is exactly what makes the level-matcher arm's version of it a diagnostic, and
                # why the repeated stag hunt is worth a third point (all-cooperate is its unique
                # optimum, so no drop pays there either).
                end_game_drop=(
                    moves[-1] == DEFECT and moves[-2] == COOPERATE if len(moves) > 1 else None
                ),
            )
        )
    return scored


def _score_against_an_opponent_mix(
    rows: Sequence[_Row],
    visibles: Sequence[str],
    *,
    leave_one_out: bool,
    parse_penalty: float,
    parse_penalty_mode: str,
) -> list[_Scored]:
    """Score a group as expected payoff against a cooperation rate, from the group or from cache.

    The two gradings differ only in where the opponent distribution comes from, which is why the
    frozen-opponent validation lives here rather than in the dispatch: it is a property of the one
    grading that reads a cached column.
    """
    if rows[0].grading == GRADING_GROUP_MIX:
        return _score_expected_payoff_group(
            rows,
            visibles,
            opponent_coop_prob=None,
            leave_one_out=leave_one_out,
            parse_penalty=parse_penalty,
            parse_penalty_mode=parse_penalty_mode,
        )
    probability = float(rows[0].opp_coop_prob)
    if not 0.0 <= probability <= 1.0:
        raise RuntimeError(
            f"prompt_id={rows[0].prompt_id!r} is a {GRADING_VS_FIXED_MIX} row whose "
            f"opp_coop_prob is {probability}, outside [0,1]. The frozen opponent was never "
            f"sampled for this prompt (the corpus writes -1 when absent), so there is no "
            f"opponent distribution to grade against."
        )
    return _score_expected_payoff_group(
        rows,
        visibles,
        opponent_coop_prob=probability,
        # A frozen opponent has no group to leave anyone out of: the cached probability is the
        # whole opponent distribution. Passed as False rather than forwarded, so the signature
        # stops reading as though the two grading modes compose.
        leave_one_out=False,
        parse_penalty=parse_penalty,
        parse_penalty_mode=parse_penalty_mode,
    )


OPPONENT_MIX_GRADINGS: frozenset[str] = frozenset({GRADING_GROUP_MIX, GRADING_VS_FIXED_MIX})

# The ladder's recipient per grading: which expected payoff a parsed action earns against the
# group's realised mix. A table into one scorer, the nash-demand pattern -- rules over one reward
# function -- so the ladder cannot drift into per-grading arithmetics.
RECIPIENT_WEIGHTED_GRADINGS: dict[str, Callable[[MatrixGameSpec, str, float], float]] = {
    GRADING_JOINT_WELFARE_GROUP_MIX: expected_joint_payoff,
    GRADING_OTHER_PAYOFF_GROUP_MIX: expected_counterpart_payoff,
}


def _recipient_weighted_payoff(
    grading: str,
) -> Callable[[MatrixGameSpec, str, float], float] | None:
    """Return which recipient's expected payoff a matrix grading pays, or None if it is not one.

    The ladder's two fixed recipients come from the table above; the care family's recipient is the
    weighted blend its own name states, so it is built here rather than tabulated -- a set cannot
    hold one entry per real number. One resolver into `_score_expected_payoff_group` is what keeps
    the family's mix estimation, leave-one-out fallbacks, prior accounting and parse policy
    byte-for-byte the ladder's.
    """
    tabulated = RECIPIENT_WEIGHTED_GRADINGS.get(grading)
    if tabulated is not None:
        return tabulated
    alpha = care_alpha_of(grading)
    if alpha is None:
        return None
    return partial(expected_care_payoff, alpha=alpha)


# The gradings whose scorer needs nothing but the parse price: the reward reads one completion's own
# answer and the row it answered, with no group distribution and no per-grading branch inside. A table
# rather than another `if` arm, because that chain is where a new game's scorer gets added and every
# added branch cost `_score_group` another return until the complexity limit refused it. The
# stated-match scorer joined the table when the row-relative price was generalised to every grading
# (2026-09-04): it used to be dispatched beside it as the one scorer taking the parse-penalty mode, and
# now every scorer takes it.
SCORERS_NEEDING_ONLY_THE_PARSE_PRICE: dict[str, Callable[..., list[_Scored]]] = {
    GRADING_SELF: _score_self_group,
    GRADING_KEEP_FRACTION: _score_keep_fraction_group,
    GRADING_ITERATED_RETURN: _score_iterated_group,
    GRADING_FORMAT_ONLY: _score_format_only_group,
    GRADING_TRUSTOR_PAYOFF_STATED_RULE: _score_trustor_stated_rule_group,
    GRADING_TRUSTOR_PAYOFF_SELF_RULE: _score_trustor_self_rule_group,
    GRADING_LEVEL_MATCH_RETURN: _score_level_match_group,
    GRADING_VS_STATED_MATCH: _score_stated_match_group,
}


def _refuse_a_care_row_the_family_cannot_score(row: _Row) -> None:
    """Refuse a care row that is neither a labelled matrix row nor a trust row with its own rate.

    The family scores exactly those two shapes and the announced return rate is the discriminator, so
    anything else falls through to the matrix scorer -- which raises inside `parse_action` on the two
    blank action labels every number-answered game carries, naming label corruption for a row whose
    labels were never meant to be filled. That message sends the reader after a corrupt corpus rather
    than after the missing rate, which is the actual fault when a trust row loses it: the family's
    trust leg then never runs and the send answers are graded as no action at all.
    """
    if row.label_a.strip() or row.label_b.strip():
        return
    raise RuntimeError(
        f"prompt_id={row.prompt_id!r} carries grading {row.grading!r} over game {row.game_id!r}, "
        f"which prints no action labels, while its {STATED_RETURN_FRACTION_COLUMN} is the "
        f"{STATED_RETURN_UNSET} marker for a row that announces no return rate. The care family "
        f"scores a labelled matrix row or an announced-rule trust row and nothing else, so either "
        f"this trust row lost the rate its own prompt states, or it is a number-answered game the "
        f"family has no scorer for."
    )


def _score_own_answer_group(
    rows: Sequence[_Row], visibles: Sequence[str], *, parse_penalty: float, parse_penalty_mode: str
) -> list[_Scored] | None:
    """Score a group whose grading reads each completion's own answer only, or None for the rest.

    Every scorer here takes the parse-penalty mode and prices its own failures against the row's own
    answer space, so there is no mode gate at the dispatch any more: the two gradings that cannot
    price a failure refuse it themselves, by the reason `PARSE_PRICE_UNDEFINED_GRADINGS` records.
    """
    grading = rows[0].grading
    # The care family's trust rows belong here rather than in the table: their reward reads one
    # completion's own send against the rate its own prompt announced, with no group coupling at all
    # (exactly `trustor-payoff-stated-rule`'s shape), but the arithmetic needs the weight from the
    # grading name, which a name-keyed table of `(rows, visibles, parse price)` scorers cannot carry.
    # The row's `stated_return_fraction` is the discriminator, the same column `_score_group`'s matrix
    # path leaves alone, so one care corpus can hold both row types.
    care_alpha = care_alpha_of(grading)
    if care_alpha is not None:
        if rows[0].stated_return_fraction != STATED_RETURN_UNSET:
            return _score_trustor_care_group(
                rows,
                visibles,
                alpha=care_alpha,
                parse_penalty=parse_penalty,
                parse_penalty_mode=parse_penalty_mode,
            )
        _refuse_a_care_row_the_family_cannot_score(rows[0])
    own_answer_only = SCORERS_NEEDING_ONLY_THE_PARSE_PRICE.get(grading)
    if own_answer_only is None:
        return None
    return own_answer_only(
        rows, visibles, parse_penalty=parse_penalty, parse_penalty_mode=parse_penalty_mode
    )


def _score_group(
    rows: Sequence[_Row],
    visibles: Sequence[str],
    *,
    leave_one_out: bool,
    parse_penalty: float,
    parse_penalty_mode: str = PARSE_PENALTY_CONSTANT,
) -> list[_Scored]:
    """Dispatch one group on its (uniform) grading column."""
    grading = rows[0].grading
    own_answer_only = _score_own_answer_group(
        rows, visibles, parse_penalty=parse_penalty, parse_penalty_mode=parse_penalty_mode
    )
    if own_answer_only is not None:
        return own_answer_only
    if grading in OPPONENT_MIX_GRADINGS:
        return _score_against_an_opponent_mix(
            rows,
            visibles,
            leave_one_out=leave_one_out,
            parse_penalty=parse_penalty,
            parse_penalty_mode=parse_penalty_mode,
        )
    recipient_payoff = _recipient_weighted_payoff(grading)
    if recipient_payoff is not None:
        return _score_expected_payoff_group(
            rows,
            visibles,
            opponent_coop_prob=None,
            leave_one_out=leave_one_out,
            parse_penalty=parse_penalty,
            parse_penalty_mode=parse_penalty_mode,
            expected_payoff_fn=recipient_payoff,
        )
    if grading in NASH_DEMAND_GRADINGS:
        return _score_nash_demand_group(
            rows,
            visibles,
            grading=grading,
            leave_one_out=leave_one_out,
            parse_penalty=parse_penalty,
            parse_penalty_mode=parse_penalty_mode,
        )
    if grading in THRESHOLD_GOODS_GRADINGS:
        return _score_threshold_goods_group(
            rows,
            visibles,
            grading=grading,
            leave_one_out=leave_one_out,
            parse_penalty=parse_penalty,
            parse_penalty_mode=parse_penalty_mode,
        )
    if grading == GRADING_MIN_EFFORT_GROUP_MIX:
        return _score_min_effort_group(
            rows,
            visibles,
            leave_one_out=leave_one_out,
            parse_penalty=parse_penalty,
            parse_penalty_mode=parse_penalty_mode,
        )
    # Named before the raise: a care-shaped grading in the wrong spelling reaches here as "unknown",
    # which sends the reader looking for a missing scorer when the fix is the name.
    assert_canonical_care_grading(grading)
    raise RuntimeError(
        f"prompt_id={rows[0].prompt_id!r} carries an unknown grading, which no scorer answers to. "
        f"{unknown_grading_message(grading)}"
    )


def _rows_from_columns(columns: Mapping[str, Any], n_completions: int) -> list[_Row]:
    """Rebuild one `_Row` per completion from TRL's parallel column lists."""
    missing = [name for name in REQUIRED_REWARD_COLUMNS if name not in columns]
    if missing:
        raise RuntimeError(
            f"Dataset is missing reward columns {missing}; the reward function needs "
            f"{list(REQUIRED_REWARD_COLUMNS)}."
        )
    mismatched = {
        name: len(columns[name])
        for name in REQUIRED_REWARD_COLUMNS
        if len(columns[name]) != n_completions
    }
    if mismatched:
        raise RuntimeError(
            f"Column lists must be parallel to the {n_completions} completions; "
            f"got lengths {mismatched}."
        )
    return [
        _Row(**{name: columns[name][index] for name in REQUIRED_REWARD_COLUMNS})
        for index in range(n_completions)
    ]


def _optional_column(columns: Mapping[str, Any], name: str, n_completions: int) -> list[str] | None:
    """Return one forwarded column as strings, or None when the corpus does not carry it.

    For the composition columns the metrics read and no scorer does, so absence is a corpus vintage
    rather than an error. A column that IS present is held to the same parallel-length rule as every
    required one: a short list would silently misalign every value after the hole.
    """
    if name not in columns:
        return None
    values: list[Any] = list(columns[name])
    if len(values) != n_completions:
        raise RuntimeError(
            f"Column {name!r} must be parallel to the {n_completions} completions; got "
            f"{len(values)} values."
        )
    return [str(value) for value in values]


def _assert_contiguous_groups(rows: Sequence[_Row], num_generations: int) -> None:
    """Raise unless each block of G completions is one prompt with one grading.

    A silently decorrelated block is the failure this exists to catch: group-mix grading would
    keep producing plausible rewards computed against the wrong opponent population.
    """
    for start in range(0, len(rows), num_generations):
        block = rows[start : start + num_generations]
        prompt_ids = {row.prompt_id for row in block}
        if len(prompt_ids) != 1:
            raise RuntimeError(
                f"Completions {start}..{start + num_generations - 1} are not one prompt's "
                f"group: found prompt_ids {sorted(prompt_ids)}. TRL's RepeatSampler must yield "
                f"each prompt's {num_generations} completions contiguously, and group-mix "
                f"grading estimates the opponent distribution from that block."
            )
        gradings = {row.grading for row in block}
        if len(gradings) != 1:
            raise RuntimeError(
                f"Group at completion {start} mixes gradings {sorted(gradings)}; a group is one "
                f"prompt, so its grading must be uniform."
            )


def _groups_of(scored: Sequence[_Scored], num_generations: int) -> list[Sequence[_Scored]]:
    """Split a batch into its per-prompt groups, refusing a batch with none."""
    if num_generations < 1:
        raise ValueError(f"num_generations must be positive, got {num_generations}.")
    groups = [
        scored[start : start + num_generations] for start in range(0, len(scored), num_generations)
    ]
    if not groups:
        raise ValueError("Cannot measure a group statistic with no groups.")
    return groups


def group_purity(scored: Sequence[_Scored], num_generations: int) -> float:
    """Fraction of groups whose rewards are all identical, so the group carries no gradient.

    This replaces TRL's `frac_reward_zero_std`, which was a DEAD METRIC for the recorded runs and
    reported 0.0000 on every step of every arm for a whole night while being cited as evidence that
    every group disagreed. Two independent reasons it could never fire there: those runs'
    `scale_rewards="batch"` made TRL's standard deviation a batch-level scalar rather than
    per-group, so a pure group was invisible to it; and a float32 `nanstd` over bit-identical
    rewards lands around 1e-8, outside its `isclose` tolerance -- which still holds under the
    current default `scale_rewards="none"` (group-level stds), so the metric stays replaced. It
    also passed our read-back guard, because that guard asks whether a metric is PRESENT and a
    dead metric is present and constant.

    Computed on exact equality of the rewards themselves, which is right because a pure group is
    exactly the case GRPO cannot learn from: every advantage is zero regardless of the values.
    """
    groups = _groups_of(scored, num_generations)
    pure = sum(1 for group in groups if len({item.reward for item in group}) == 1)
    return pure / len(groups)


def group_reward_span(scored: Sequence[_Scored], num_generations: int) -> float:
    """Mean within-group reward range, which under `scale_rewards="none"` IS the advantage scale.

    The quantitative companion to `group_purity`: purity says how many groups carry no gradient at
    all, and this says how much gradient the rest carry. Worth logging per step for every grading,
    because a batch whose strategy span is 0.03 against a parse penalty 1.0 away is an arm training
    formatting, and no other recorded metric would say so.
    """
    groups = _groups_of(scored, num_generations)
    spans = [
        max(item.reward for item in group) - min(item.reward for item in group) for group in groups
    ]
    return sum(spans) / len(spans)


def _log_claim_metrics(
    scored: Sequence[_Scored], *, log_metric: Callable[[str, float], None]
) -> None:
    """Report the claim game's own behavioural numbers, or nothing when no row is one.

    Three, each of which a run has been repeated for the absence of somewhere in this repo's
    history: the behavioural mean, the fairness anchor (how often the answer is exactly half the
    windfall, which is where the 2B's dictator answers already pile up), and the overreach rate that
    supplies this arm's downward pressure.

    That last one is logged under a per-grading key, never one shared key. Under group-mix grading it
    is a mean collision rate against the group's realised claims, so a zero means the group's figures
    all fit and the reward is still a bare ramp -- climbing, not stalled. Under self grading the
    counterpart is the completion's own claim, so the same arithmetic is the deterministic indicator
    `claim > windfall / 2`, i.e. an over-claiming rate that is redundant with `mean_claim_fraction`.
    They read the same number on the same claims for unrelated reasons, so plotting them as one
    series would be a false reading of the self-graded arm.
    """
    claim_fractions = [item.claim_fraction for item in scored if item.claim_fraction is not None]
    if claim_fractions:
        log_metric("mean_claim_fraction", sum(claim_fractions) / len(claim_fractions))
        log_metric(
            "exact_half_claim_rate",
            sum(fraction == EQUAL_SPLIT_CLAIM_FRACTION for fraction in claim_fractions)
            / len(claim_fractions),
        )
    crash_rates = [item.crash_rate for item in scored if item.crash_rate is not None]
    if crash_rates:
        log_metric("crash_rate", sum(crash_rates) / len(crash_rates))
    overclaim_rates = [item.overclaim_rate for item in scored if item.overclaim_rate is not None]
    if overclaim_rates:
        log_metric("overclaim_rate", sum(overclaim_rates) / len(overclaim_rates))


def _log_threshold_goods_metrics(
    scored: Sequence[_Scored], *, log_metric: Callable[[str, float], None]
) -> None:
    """Report the shared undertaking's own behavioural numbers, or nothing when no row is one.

    Four, and each answers a question the others cannot. The behavioural mean is the arm's headline
    series. The over-contribution rate is the cooperative-versus-maximising discrimination this game
    exists for, and it is a property of the answer alone, so one key serves both gradings.

    The reach rate is logged under a per-grading key, never one shared key. Under group-mix grading it is
    the expected chance the pooled figures clear the bar, which is this arm's gradient supply: at a rate
    of 0 or 1 across a whole group the prize term is constant and only the cost term is left, so the arm
    trains contributions to zero for a structural reason rather than a behavioural one. Under self grading
    the counterparts write what this completion wrote, so the same arithmetic is the deterministic
    indicator `contribution >= equal_share` -- a share-met rate redundant with the behavioural mean. They
    would read the same number on the same figures for unrelated reasons, so plotting them as one series
    would be a false reading of the self-graded arm.
    """
    fractions = [
        item.contribution_fraction for item in scored if item.contribution_fraction is not None
    ]
    if fractions:
        log_metric("mean_contribution_fraction", sum(fractions) / len(fractions))
    over = [item.over_contribution for item in scored if item.over_contribution is not None]
    if over:
        log_metric("over_contribution_rate", sum(over) / len(over))
    reached = [item.threshold_met_rate for item in scored if item.threshold_met_rate is not None]
    if reached:
        log_metric("threshold_met_rate", sum(reached) / len(reached))
    share_met = [
        item.equal_share_met_rate for item in scored if item.equal_share_met_rate is not None
    ]
    if share_met:
        log_metric("equal_share_met_rate", sum(share_met) / len(share_met))


def _log_level_metrics(
    scored: Sequence[_Scored], *, log_metric: Callable[[str, float], None]
) -> None:
    """Report the minimum-effort games' own behavioural numbers, or nothing when no row is one.

    Five, each of which this repo has repeated a run for the absence of somewhere in its history. The
    behavioural mean on the grid's [0,1] scale and the mean raw level, so a readout can quote either
    without re-deriving one from the other. The mean TARGET level, which is the number the
    pre-registration is scored against -- the analogue of chicken's recorded fixed point, and the
    thing that makes "did the arm go where the payoffs pointed" answerable after the fact rather than
    only during. The mean upward pressure, which is where the gradient comes from: positive means the
    group is being pushed up the grid and negative means down, so a flat curve at a mean near zero is
    a group sitting where the payoffs want it rather than an arm with no signal, and those two
    readings call for opposite responses.

    The per-completion levels go out through `log_extra` rather than as a summary, because the
    histogram is the question -- a mean of 3 is a group at 3 or a group split between 1 and 5, and
    only the second of those has any gradient.
    """
    fractions = [item.level_fraction for item in scored if item.level_fraction is not None]
    if fractions:
        log_metric("mean_level_fraction", sum(fractions) / len(fractions))
    levels = [item.level for item in scored if item.level is not None]
    if levels:
        log_metric("mean_level", sum(levels) / len(levels))
    targets = [item.target_level for item in scored if item.target_level is not None]
    if targets:
        log_metric("mean_target_level", sum(targets) / len(targets))
    pressures = [item.upward_pressure for item in scored if item.upward_pressure is not None]
    if pressures:
        log_metric("mean_upward_pressure", sum(pressures) / len(pressures))


def _log_stated_match_metrics(
    scored: Sequence[_Scored], *, log_metric: Callable[[str, float], None]
) -> None:
    """Report the stated-track-record grading's own numbers, or nothing when no row is one.

    Three beyond the shared cooperation rate, each answering a question the pooled rate cannot.
    `ev_optimum_rate` is the headline: how often the parsed action is the one the row's own
    (payoff table, stated p) pair pays more for -- a policy over p, which is what the arm exists
    to train. The two split cooperation rates keep the incentive directions apart: this corpus
    deliberately mixes cells where cooperating pays with cells where defecting pays, so the pooled
    `coop_rate` averages two opposite predictions and a run drifting to blanket cooperation would
    read as healthy mid-scale movement there. Split, the same drift reads as
    `coop_rate_where_defect_pays` climbing -- the anti-incentive direction, visible per step.
    """
    optimum = [item.at_ev_optimum for item in scored if item.at_ev_optimum is not None]
    if optimum:
        log_metric("ev_optimum_rate", sum(optimum) / len(optimum))
    coop_side = [
        item.coop_fraction
        for item in scored
        if item.coop_pays is True and item.coop_fraction is not None
    ]
    if coop_side:
        log_metric("coop_rate_where_coop_pays", sum(coop_side) / len(coop_side))
    defect_side = [
        item.coop_fraction
        for item in scored
        if item.coop_pays is False and item.coop_fraction is not None
    ]
    if defect_side:
        log_metric("coop_rate_where_defect_pays", sum(defect_side) / len(defect_side))


def _log_trust_metrics(
    scored: Sequence[_Scored], *, log_metric: Callable[[str, float], None]
) -> None:
    """Report the trust games' own behavioural numbers, or nothing when no row is one.

    Four, and each is instrumentation a reading depends on rather than an extra. The mean send is the
    behaviour. The mean promised share is the promise-inflation channel of the strategy method, where
    the promise is free. The two optimum rates are two different questions under one phrase, which is
    why they are two keys: under an ANNOUNCED rate the optimum send is a corner the prompt fixes, so
    the rate reads "did it multiply the number in front of it", while under the strategy method the
    completion writes the rate too, so the only unambiguous optimum is the whole (send, promise) pair.
    """
    send_fractions = [item.send_fraction for item in scored if item.send_fraction is not None]
    if send_fractions:
        log_metric("mean_send_fraction", sum(send_fractions) / len(send_fractions))
    promised = [
        item.stated_return_fraction for item in scored if item.stated_return_fraction is not None
    ]
    if promised:
        log_metric("mean_stated_return_fraction", sum(promised) / len(promised))
    optimal = [item.at_payoff_optimum for item in scored if item.at_payoff_optimum is not None]
    if optimal:
        log_metric("send_at_payoff_optimum_rate", sum(optimal) / len(optimal))
    optimal_pairs = [
        item.at_strategy_optimum for item in scored if item.at_strategy_optimum is not None
    ]
    if optimal_pairs:
        log_metric("strategy_at_payoff_optimum_rate", sum(optimal_pairs) / len(optimal_pairs))


def _group_labels(labels: Sequence[str], num_generations: int, *, column: str) -> list[str]:
    """Reduce a per-completion column to one value per group, refusing a group that disagrees.

    A group is one prompt, so a composition column has one value across its block. Two values in a
    block means the columns arrived decorrelated from the completions, and the group's behaviour
    would then be filed under whichever value came first -- the same failure `_assert_contiguous_groups`
    exists for, in the axis the per-framing series are read on.
    """
    per_group: list[str] = []
    for start in range(0, len(labels), num_generations):
        block = {str(label) for label in labels[start : start + num_generations]}
        if len(block) != 1:
            raise RuntimeError(
                f"completions {start}..{start + num_generations - 1} disagree about their {column} "
                f"({sorted(block)}); a group is one prompt, so its {column} is one value, and this "
                f"group's cooperation would be filed under whichever value came first."
            )
        per_group.append(next(iter(block)))
    return per_group


def _log_metrics_by_composition(
    groups: Sequence[Sequence[_Scored]],
    labels: Sequence[str],
    *,
    axis: str,
    num_generations: int,
    log_metric: Callable[[str, float], None],
) -> None:
    """Report this step's cooperation rate, purity and group count for each value of one corpus axis.

    Keyed `<metric>/<axis>/<value>`, which is how the readout picks the series out of `log_history`.
    The group count is logged for every value including one whose game answers with a number rather
    than an action, because a series with no denominator cannot be read: a step that drew no group of
    a framing and a step whose groups all defected both show a missing cooperation rate.

    A falsy label is counted nowhere. That is the trust sender's `FRAMING_ID_UNSET`, and it is a skip
    rather than a bucket of its own so no series is named after the absence of a framing.
    """
    for value in sorted({label for label in labels if label}):
        selected_groups = [
            group for group, label in zip(groups, labels, strict=True) if label == value
        ]
        log_metric(f"n_groups/{axis}/{value}", float(len(selected_groups)))
        completions = [item for group in selected_groups for item in group]
        log_metric(f"frac_groups_pure/{axis}/{value}", group_purity(completions, num_generations))
        coop_fractions = [
            item.coop_fraction for item in completions if item.coop_fraction is not None
        ]
        if coop_fractions:
            log_metric(f"coop_rate/{axis}/{value}", sum(coop_fractions) / len(coop_fractions))


# How far outside its reachable range a realised reward may sit before the parse-price guard reads it
# as a wrong reachable set rather than as float noise. Every quantity compared here is a short sum of
# products of numbers in [0, 1], so the accumulated error is a few ulps and 1e-9 is orders above it.
# Nothing legitimate sits inside the band either: the narrowest NONZERO reachable spread on the slate
# is the shared undertaking's 0.318 at the low prize, and a row that genuinely admits no spread (the
# care family at its joint-welfare attractor) comes out at exactly zero rather than near it, where the
# price is the worst reachable reward and both comparisons hold on the nose.
PARSE_PRICE_TOLERANCE = 1e-9


def _refuse_a_price_that_is_not_one_spread_below_the_worst_reachable(
    row: _Row,
    price: _ParsePrice,
    reachable: tuple[float, ...],
    *,
    worst: float,
    best: float,
) -> None:
    """Raise unless the price is one reachable spread below the worst of the set that priced it.

    Half the ranking the mode exists for, and the half that needs nothing but the tuple the branch
    recorded: a price that is not one spread below the worst says the set is right and the arithmetic
    is not, a branch pricing by its own route instead of `margin_below_worst_reachable`, which is what
    `_price_stated_match_failure` did before 2026-09-04 and what the next grading added here could do
    again. Because it reads one call site's own numbers, it holds at EVERY resolution, leave-one-out
    included, which is why it sits outside the `shared_resolution` gate that the range half needs.

    The refusal deliberately names nothing from the group. Under leave-one-out the group's rewards come
    from mixes this price was never resolved at, so quoting them beside a defect in one branch's
    arithmetic would point the reader at a comparison that was not made.
    """
    spread = best - worst
    if abs(price.price - (worst - spread)) > PARSE_PRICE_TOLERANCE:
        realised_ratio = (worst - price.price) / spread if spread else float("inf")
        raise RuntimeError(
            f"prompt_id={row.prompt_id!r} grading={row.grading!r}: this row's failure was priced at "
            f"{price.price} from reachable rewards {reachable} resolved at {price.resolution}, where "
            f"one spread below the worst reachable reward is {worst - spread}. The price is not the "
            f"mode's own arithmetic, so the failure-to-task ratio the run record claims is 1 is "
            f"actually {realised_ratio}."
        )


def _refuse_a_reachable_set_the_group_played_outside_of(  # noqa: PLR0913 - one refusal, six things it must name
    row: _Row,
    price: _ParsePrice,
    reachable: tuple[float, ...],
    parsed_rewards: Sequence[float],
    *,
    worst: float,
    best: float,
) -> None:
    """Raise unless every reward this group realised lies inside the range the price was derived from.

    The other half: a parsed reward outside `[worst, best]` says the reachable set is not this row's
    answer space -- an action left out, or the right actions resolved at a counterpart distribution the
    group never played. Unlike the identity above it reads the group, so it is asked only where every
    completion was graded at the resolution this price names (`_ParsePrice.shared_resolution`);
    under leave-one-out a parsed reward outside the range is that grading's correct arithmetic.

    "Strictly below every parsed reward, or equal to the worst reachable reward at zero spread" is
    deliberately NOT a third check: it follows from these two, so its branch could not be made to go
    red without disabling one of them, and a branch nobody can watch fail is a reassuring message rather
    than a check. `parse_price/min_margin_below_worst_parsed` reports that gap as a number instead, and
    zero spread is the case the care family settles at, where equality is the correct answer.
    """
    outside = [
        realised
        for realised in parsed_rewards
        if not worst - PARSE_PRICE_TOLERANCE <= realised <= best + PARSE_PRICE_TOLERANCE
    ]
    if outside:
        raise RuntimeError(
            f"prompt_id={row.prompt_id!r} grading={row.grading!r}: this row's failure branch priced "
            f"an unparseable completion at {price.price} from reachable rewards {reachable} "
            f"(worst={worst}, best={best}) resolved at {price.resolution}, but the group realised "
            f"{outside} outside that range. The reachable set is not this row's answer space, so the "
            f"price is not one within-group spread below anything and every step after this one would "
            f"train under a reward the run record misdescribes."
        )


def _priced_failures(
    groups: Sequence[Sequence[_Scored]], group_rows: Sequence[_Row]
) -> Iterator[tuple[_Row, _ParsePrice, list[float]]]:
    """Yield each failure's price with the row it was priced from and its group's parsed rewards.

    The three things a check on the price needs, in one walk. A failed completion carrying no price is
    refused here rather than skipped: every failure branch prices through `_price_failure` and records
    the result, so a branch added later that builds its own `_Scored` would otherwise bypass the whole
    guard silently -- which is the shape of every bug this repo has been bitten by.
    """
    for row, group in zip(group_rows, groups, strict=True):
        parsed_rewards = [item.reward for item in group if item.parsed]
        for item in group:
            if item.parsed:
                continue
            if item.parse_price is None:
                raise RuntimeError(
                    f"prompt_id={row.prompt_id!r} grading={row.grading!r}: an unparseable completion "
                    f"was paid {item.reward} without recording how it was priced, so the parse-price "
                    f"guard cannot check it. Every failure branch prices through `_price_failure` and "
                    f"carries the result on its `_Scored`; a branch that builds one by hand would slip "
                    f"past this check without a word."
                )
            yield row, item.parse_price, parsed_rewards


def _log_parse_price_metrics(
    groups: Sequence[Sequence[_Scored]],
    group_rows: Sequence[_Row],
    *,
    log_metric: Callable[[str, float], None],
) -> None:
    """Check every priced failure against its group, and report what the guard saw.

    The seam the 2026-09-04 parse-price commit's own audit asked for (its G2). Under
    `PARSE_PENALTY_MARGIN_BELOW_WORSE` each grading's failure branch supplies its OWN reachable set,
    and nothing else in the module would notice that set being wrong: a branch that forgot an action or
    resolved the wrong counterpart distribution prices a failure inside or above the range its group
    realised, the ranking the mode exists to keep inverts, every metric stays green and every later
    step trains under a reward the run record misdescribes. So the price is checked here, where the
    group's realised parsed rewards and its failure prices are both in hand, and a violation raises
    rather than warns.

    The two refusals carry their own preconditions, because one gate over both left the whole guard
    disarmed wherever the group was graded elsewhere: every group-coupled grading marks its price
    `shared_resolution=False` under leave-one-out, so a `--leave-one-out` run checked nothing at all
    and the commit's own sabotage (a halved coefficient in `margin_below_worst_reachable`) trained for
    the whole run. The range half genuinely needs the group and stays behind that gate; the identity
    half reads only the branch's own tuple and runs for every failure that recorded one.

    Six readings over seven keys, and why each is here rather than derivable from the others.
    `realised_mean` duplicates `mean_parse_penalty_reward`'s value under the guard's own prefix, because
    a readout pulling the guard's series wants them as one block and the older key is what 32 banked
    runs are keyed on. `n_failures`, `n_checked` and `n_identity_checked` are the denominators, one per
    half so neither reads as the other: `n_identity_checked` is short of `n_failures` only under the
    constant mode, where there is no answer space at all, while `n_checked` is also short in a group
    where nothing parsed (nothing to compare against) and under leave-one-out (each completion faces
    its own counterpart distribution, so the group's rewards are not drawn from the failed row's
    reachable set). `min_margin_below_worst_parsed` is the audit's own dominance number -- the group's
    worst parsed reward minus the price -- reported for every failure with a parsed sibling, including
    under the constant, where it is exactly the ratio that motivated the mode; it can go NEGATIVE under
    leave-one-out, which is that grading's correct arithmetic rather than an inversion, and cannot
    anywhere the range half checked. `guard_identity_min`/`_max` are
    `(worst reachable - price) / spread` over the failures whose row admits a spread, 1.0 by the price's
    construction: constant on purpose, and logged so a run says the guard ran on real failures rather
    than only that it did not raise.
    They are a narrower denominator than `n_identity_checked`, which counts the zero-spread rows the
    care family settles at too, where the ratio has nothing to divide by and the price must still be
    exactly the worst reachable reward.
    """
    prices: list[float] = []
    margins: list[float] = []
    identities: list[float] = []
    identity_checked = 0
    checked = 0
    for row, price, parsed_rewards in _priced_failures(groups, group_rows):
        prices.append(price.price)
        if parsed_rewards:
            margins.append(min(parsed_rewards) - price.price)
        reachable = price.reachable
        if reachable is None:
            continue
        worst, best = min(reachable), max(reachable)
        if best - worst > 0.0:
            identities.append((worst - price.price) / (best - worst))
        identity_checked += 1
        _refuse_a_price_that_is_not_one_spread_below_the_worst_reachable(
            row, price, reachable, worst=worst, best=best
        )
        if not (parsed_rewards and price.shared_resolution):
            continue
        checked += 1
        _refuse_a_reachable_set_the_group_played_outside_of(
            row, price, reachable, parsed_rewards, worst=worst, best=best
        )

    # All three counts are logged even at zero, unlike every rate in this module: "no failure was
    # priced this step" and "neither half of the guard ran" are readings, and an absent metric reads
    # as neither.
    log_metric("parse_price/n_failures", float(len(prices)))
    log_metric("parse_price/n_checked", float(checked))
    log_metric("parse_price/n_identity_checked", float(identity_checked))
    if prices:
        log_metric("parse_price/realised_mean", sum(prices) / len(prices))
    if margins:
        log_metric("parse_price/min_margin_below_worst_parsed", min(margins))
    if identities:
        log_metric("parse_price/guard_identity_min", min(identities))
        log_metric("parse_price/guard_identity_max", max(identities))


def _log_batch_metrics(  # noqa: PLR0913 - one step's whole metric surface, called from one place
    scored: Sequence[_Scored],
    *,
    num_generations: int,
    leave_one_out: bool,
    rows: Sequence[_Row],
    framing_ids: Sequence[str] | None,
    log_metric: Callable[[str, float], None],
    log_extra: Callable[[str, list[Any]], None],
) -> None:
    """Report rates through TRL's injected loggers, and the per-completion detail as columns."""
    n = len(scored)
    log_metric("parse_failure_rate", sum(not item.parsed for item in scored) / n)
    # What a failure actually cost this step. A constant penalty makes this a flat line; under
    # `margin-below-worse` it is the mean of per-cell prices, and the readout needs the realized
    # figure beside the rate to say how much gradient the format channel carried.
    failed_rewards = [item.reward for item in scored if not item.parsed]
    if failed_rewards:
        log_metric("mean_parse_penalty_reward", sum(failed_rewards) / len(failed_rewards))
    groups = _groups_of(scored, num_generations)
    # A group is one prompt (`_assert_contiguous_groups`), so its first row is the row every
    # completion in it was scored from, which is what the guard's refusals name.
    _log_parse_price_metrics(
        groups,
        [rows[start] for start in range(0, len(rows), num_generations)],
        log_metric=log_metric,
    )
    # Our own policy-collapse detector. Trending to 1.0 means the policy has gone pure and no step
    # after that carries signal, which is a finding to diagnose rather than a run to let continue.
    log_metric("frac_groups_pure", group_purity(scored, num_generations))
    # How much gradient the groups that do disagree carry, which purity cannot say.
    log_metric("mean_group_reward_span", group_reward_span(scored, num_generations))

    coop_fractions = [item.coop_fraction for item in scored if item.coop_fraction is not None]
    if coop_fractions:
        log_metric("coop_rate", sum(coop_fractions) / len(coop_fractions))
    keep_fractions = [item.keep_fraction for item in scored if item.keep_fraction is not None]
    if keep_fractions:
        log_metric("mean_keep_fraction", sum(keep_fractions) / len(keep_fractions))
    _log_stated_match_metrics(scored, log_metric=log_metric)
    _log_claim_metrics(scored, log_metric=log_metric)
    _log_threshold_goods_metrics(scored, log_metric=log_metric)
    _log_level_metrics(scored, log_metric=log_metric)
    # Logged for BOTH repeated forms under one key on purpose, because it is one quantity -- "was the
    # last round less than the one before" -- and the pair's whole reading is the comparison between
    # them. It means opposite things about the policy in each: against the level-matcher a drop only
    # ever loses money, so it is a transferred habit, while against the copying opponent in the PD a
    # last-round defection is genuinely optimal. That is a difference between the two GAMES, not
    # between two metrics, so splitting the key would hide exactly the contrast.
    drops = [item.end_game_drop for item in scored if item.end_game_drop is not None]
    if drops:
        log_metric("end_game_drop_rate", sum(drops) / len(drops))
    _log_trust_metrics(scored, log_metric=log_metric)
    # The breadth split, after the pooled rates rather than instead of them: on a corpus of several
    # games under several framings the pooled cooperation rate averages the cells whose separation IS
    # the measurement, and one step's 8 prompts cover only a few of them, so the per-cell series and
    # their per-step denominators are what a trajectory is read off.
    _log_metrics_by_composition(
        groups,
        _group_labels([row.game_id for row in rows], num_generations, column="game_id"),
        axis="game",
        num_generations=num_generations,
        log_metric=log_metric,
    )
    if framing_ids is not None:
        _log_metrics_by_composition(
            groups,
            _group_labels(framing_ids, num_generations, column=FRAMING_ID_COLUMN),
            axis="framing",
            num_generations=num_generations,
            log_metric=log_metric,
        )
    if leave_one_out:
        log_metric("leave_one_out_prior_rate", sum(item.used_opponent_prior for item in scored) / n)

    # Per-component credit rates for the format-only placebo, over the completions that parsed. The
    # denominator is deliberately those completions and not the batch: a component's rate is a
    # statement about answer shape, and mixing in completions that produced no answer would make a
    # rising parse-failure rate read as falling format compliance.
    graded_shapes = [dict(item.format_components) for item in scored if item.format_components]
    if graded_shapes:
        for name in COMPONENT_NAMES:
            log_metric(
                f"format_{name}_rate",
                sum(shape[name] for shape in graded_shapes) / len(graded_shapes),
            )

    log_extra("parsed_action", [item.detail for item in scored])
    log_extra("game_reward", [item.reward for item in scored])
    # The realised level per completion, so the histogram is a re-analysis rather than a re-run: a
    # mean of 3 is a group sitting at 3 or a group split between 1 and 5, and only the second carries
    # any gradient. None where the row is not a minimum-effort one, which is every other game.
    log_extra("min_effort_level", [item.level for item in scored])
    # The stated probability and signed EV margin per completion, None outside the
    # stated-track-record grading: the per-rung dose curve at every training step comes out of the
    # completions parquet instead of a re-run, which is this repo's whole instrumentation posture.
    log_extra("stated_match_prob", [item.stated_match_prob for item in scored])
    log_extra("stated_match_ev_margin", [item.ev_margin for item in scored])


def make_game_reward(  # noqa: C901, PLR0913, PLR0915 - one closure owns the reward contract
    num_generations: int,
    *,
    prefilled_think: bool,
    leave_one_out: bool = False,
    parse_penalty: float = DEFAULT_PARSE_PENALTY,
    parse_penalty_mode: str = PARSE_PENALTY_CONSTANT,
    tail_length_penalty_start: int | None = None,
    tail_length_penalty_max: float = 0.0,
    completion_cap: int | None = None,
    think_open_token_ids: Sequence[int] | None = None,
    think_close_token_ids: Sequence[int] | None = None,
) -> Callable[..., list[float]]:
    """Build the single reward callable TRL calls, closed over the group size and parse policy.

    `prefilled_think` must match the model's chat template: the Qwen3.5/3.8 templates emit the
    opening `<think>` as part of the prompt, so their completions contain only the closing tag,
    while Qwen3-0.6B emits both. Getting it wrong makes every rollout read as truncated
    thinking, i.e. a full batch of parse penalties.

    `leave_one_out` excludes a completion from the opponent mix it is graded against, which
    removes the self-correlation in group-mix grading; `parse_penalty` sits below the [0,1]
    payoff range so an unparseable completion is always worse than any played action.
    `parse_penalty_mode` (one of `PARSE_PENALTY_MODES`) says whether that constant is the price or the
    price is one reachable spread below the worst reward each row admits.

    Under the row-relative mode the constant is NOT consulted by any grading: the one grading that
    would pay it, `format-only`, refuses the mode outright. It keeps its range check anyway, because it
    stays a run knob recorded in `run_config.json` and gated on resume, and a nonsense value there
    would read as the price that trained the run.
    """
    if num_generations < MIN_GENERATIONS:
        raise ValueError(
            f"num_generations must be at least {MIN_GENERATIONS} for a group-relative "
            f"advantage to exist, got {num_generations}."
        )
    if not (math.isfinite(parse_penalty) and parse_penalty < 0.0):
        raise ValueError(
            f"the parse penalty must be a finite negative number so it stays below the [0,1] payoff "
            f"range, got {parse_penalty=}. A non-finite value is not a penalty at all: TRL 1.10 reads "
            f"a NaN reward as unscorable, drops the completion from its group's baseline and forces "
            f"its advantage to zero, so the parse gradient disappears while every log line stays "
            f"green, and -inf makes the group's mean and every advantage in it non-finite."
        )
    if parse_penalty_mode not in PARSE_PENALTY_MODES:
        raise ValueError(
            f"parse_penalty_mode must be one of {PARSE_PENALTY_MODES}, got {parse_penalty_mode!r}."
        )
    tail_penalty_enabled = tail_length_penalty_start is not None or tail_length_penalty_max != 0.0
    if tail_penalty_enabled:
        if tail_length_penalty_start is None or completion_cap is None:
            raise ValueError(
                "tail length penalty needs both tail_length_penalty_start and completion_cap"
            )
        if not math.isfinite(tail_length_penalty_max) or tail_length_penalty_max <= 0.0:
            raise ValueError("tail_length_penalty_max must be a finite positive number")
        if not 0 <= tail_length_penalty_start < completion_cap:
            raise ValueError(
                "tail_length_penalty_start must be non-negative and below completion_cap"
            )
    if (think_open_token_ids is None) != (think_close_token_ids is None):
        raise ValueError("thinking length metrics need both opening and closing marker token ids")
    if think_open_token_ids is not None and (
        not think_open_token_ids or not cast("Sequence[int]", think_close_token_ids)
    ):
        raise ValueError("thinking marker token id sequences must be nonempty")

    def game_reward(
        *,
        completions: list[str],
        completion_ids: list[list[int]] | None = None,
        log_metric: Callable[[str, float], None],
        log_extra: Callable[[str, list[Any]], None],
        **columns: object,
    ) -> list[float]:
        """Score one TRL batch of completions, one reward per completion."""
        if not completions:
            raise RuntimeError("Reward function called with an empty completion batch.")
        if len(completions) % num_generations != 0:
            raise RuntimeError(
                f"Batch of {len(completions)} completions is not a whole number of groups of "
                f"{num_generations}; a partial group would be graded against a fragment of its "
                f"opponent population."
            )
        rows = _rows_from_columns(columns, len(completions))
        _assert_contiguous_groups(rows, num_generations)
        framing_ids = _optional_column(columns, FRAMING_ID_COLUMN, len(completions))
        completion_lengths = (
            [len(token_ids) for token_ids in completion_ids] if completion_ids is not None else None
        )
        if (
            completion_ids is not None
            and framing_ids is not None
            and think_open_token_ids is not None
            and think_close_token_ids is not None
        ):
            thinking_lengths = thinking_token_lengths(
                completions,
                completion_ids,
                prefilled_think=prefilled_think,
                open_marker=think_open_token_ids,
                close_marker=think_close_token_ids,
            )
            for framing_id in sorted({value for value in framing_ids if value}):
                selected = [
                    length
                    for length, value in zip(thinking_lengths, framing_ids, strict=True)
                    if value == framing_id
                ]
                log_metric(
                    f"thinking/mean_length_tokens/framing/{framing_id}",
                    sum(selected) / len(selected),
                )

        stripped = [strip_thinking(text, prefilled_think=prefilled_think) for text in completions]
        visibles = [visible for visible, _ in stripped]
        truncated = [flag for _, flag in stripped]

        scored: list[_Scored] = []
        for start in range(0, len(rows), num_generations):
            stop = start + num_generations
            scored.extend(
                _score_group(
                    rows[start:stop],
                    visibles[start:stop],
                    leave_one_out=leave_one_out,
                    parse_penalty=parse_penalty,
                    parse_penalty_mode=parse_penalty_mode,
                )
            )

        if not any(item.parsed for item in scored):
            # The price rather than the constant, because under `margin-below-worse` the constant is
            # not what any of these rows was paid and naming it would send the reader to the wrong knob.
            priced_at = (
                f"the {parse_penalty} parse penalty"
                if parse_penalty_mode == PARSE_PENALTY_CONSTANT
                else f"a {parse_penalty_mode} parse price"
            )
            raise RuntimeError(
                f"None of {len(completions)} completions parsed into an action. Every reward "
                f"would be {priced_at}, so the gradient carries no signal "
                f"about the game and every further step wastes GPU. Check the completion-token "
                f"budget, the label vocabulary, and prefilled_think={prefilled_think}."
            )

        log_metric("truncated_thinking_rate", sum(truncated) / len(truncated))
        _log_batch_metrics(
            scored,
            num_generations=num_generations,
            leave_one_out=leave_one_out,
            # The rebuilt rows rather than the columns a second time: the game id, the prompt id and
            # the grading the metrics and the parse-price guard read are all required reward columns,
            # so the rows are the one place they are read from and the scorer already rebuilt each
            # row's spec from them.
            rows=rows,
            framing_ids=framing_ids,
            log_metric=log_metric,
            log_extra=log_extra,
        )
        log_extra("truncated_thinking", list(truncated))
        rewards = [item.reward for item in scored]
        if not tail_penalty_enabled:
            return rewards
        if completion_lengths is None:
            raise RuntimeError(
                "tail length penalty is enabled but TRL supplied no completion_ids to count"
            )
        penalties = tail_length_penalties(
            completion_lengths,
            start=cast("int", tail_length_penalty_start),
            cap=cast("int", completion_cap),
            max_penalty=tail_length_penalty_max,
        )
        log_metric("tail_length_penalty/mean", sum(penalties) / len(penalties))
        log_extra("tail_length_penalty", penalties)
        return [reward - penalty for reward, penalty in zip(rewards, penalties, strict=True)]

    return game_reward


def tail_length_penalties(
    completion_lengths: Sequence[int], *, start: int, cap: int, max_penalty: float
) -> list[float]:
    """Return the DAPO-style linear penalty in the completion cap's tail."""
    if not 0 <= start < cap:
        raise ValueError(f"tail length penalty needs 0 <= start < cap, got {start=} {cap=}")
    if not math.isfinite(max_penalty) or max_penalty <= 0.0:
        raise ValueError(f"max_penalty must be finite and positive, got {max_penalty=}")
    tail_width = cap - start
    return [
        max_penalty * min(max(length - start, 0), tail_width) / tail_width
        for length in completion_lengths
    ]


def _find_token_marker(tokens: Sequence[int], marker: Sequence[int], *, last: bool) -> int | None:
    matches = [
        index
        for index in range(len(tokens) - len(marker) + 1)
        if list(tokens[index : index + len(marker)]) == list(marker)
    ]
    if not matches:
        return None
    return matches[-1] if last else matches[0]


def thinking_token_lengths(
    completions: Sequence[str],
    completion_ids: Sequence[Sequence[int]],
    *,
    prefilled_think: bool,
    open_marker: Sequence[int],
    close_marker: Sequence[int],
) -> list[int]:
    """Count original generated tokens inside the thinking block, excluding its markers."""
    if len(completions) != len(completion_ids):
        raise ValueError("completions and completion_ids must have the same length")

    lengths: list[int] = []
    for completion, token_ids in zip(completions, completion_ids, strict=True):
        if prefilled_think:
            start = 0
        elif THINK_OPEN in completion:
            open_index = _find_token_marker(token_ids, open_marker, last=False)
            if open_index is None:
                raise RuntimeError("decoded completion contains <think> but its token ids do not")
            start = open_index + len(open_marker)
        else:
            lengths.append(0)
            continue
        if THINK_CLOSE in completion:
            close_index = _find_token_marker(token_ids, close_marker, last=True)
            if close_index is None:
                raise RuntimeError("decoded completion contains </think> but its token ids do not")
            stop = close_index
        else:
            stop = len(token_ids)
        if stop < start:
            raise RuntimeError("thinking close marker precedes its open marker")
        lengths.append(stop - start)
    return lengths
