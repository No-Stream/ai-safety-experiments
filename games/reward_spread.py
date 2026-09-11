"""What each of an arm's payoff variants pays, and when two of them should not share a batch.

A payoff variant whose within-group reward spread is an order of magnitude narrower than its
batch-mates trains proportionally weaker, and a flat curve for it is expected rather than
evidence. Under the default `scale_rewards="none"` these spreads directly ARE the advantage
magnitudes; under `"batch"` (the pre-2026-08-20 default) the batch-level divisor additionally
made the weakness relative to whatever shared the batch. The stag ladder is the live case: its
rungs' spreads differ up to 10.2x, so two of them run as their own arms.

Its own module for two reasons. Both `games.arm_sequence` and `games.contrast_pair_sequence` log this
table before their sweep, and the pair reached it by importing the general plan -- which dragged
`reward_hacking.backend_cli`, and through it torch and transformers, into the headline experiment's
`--print-corpus`. And it does not belong in `games.plans`, whose whole claim is that nothing about any
*experiment* is written there; this is entirely about the experiment.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

from games.payoffs import (
    GROUP_MIX_SPREAD_MIXES,
    THRESHOLD_GOODS_MAX_PRIZE,
    TRUST_MAX_STATED_RETURN_FRACTION,
    MatrixGameSpec,
    MinEffortSpec,
    NashDemandSpec,
    ThresholdGoodsSpec,
    TrustSpec,
    care_reward_spread,
    group_mix_reward_spread,
    joint_welfare_reward_spread,
    min_effort_group_reward_span,
    min_effort_reference_levels,
    nash_demand_group_reward_span,
    nash_demand_reference_claims,
    nash_demand_self_reward_span,
    other_payoff_reward_spread,
    threshold_goods_group_reward_span,
    threshold_goods_reference_contributions,
    threshold_goods_self_reward_span,
    trustor_reward_spread,
)
from games.prompts import generate_prompt_rows, reward_spread_report
from games.rewards import (
    GRADING_GROUP_MIX,
    GRADING_JOINT_WELFARE_GROUP_MIX,
    GRADING_MIN_EFFORT_GROUP_MIX,
    GRADING_NASH_DEMAND_GROUP_MIX,
    GRADING_NASH_DEMAND_SELF,
    GRADING_OTHER_PAYOFF_GROUP_MIX,
    GRADING_THRESHOLD_GOODS_SELF,
    GRADING_TRUSTOR_PAYOFF_STATED_RULE,
    GRADING_VS_FIXED_MIX,
    THRESHOLD_GOODS_GRADINGS,
    care_alpha_of,
)

# Per-mix spread functions for the matrix gradings whose reward is an expectation over the
# opponent's action mix; the ladder's two recipients read their own arithmetic from games.payoffs,
# the same functions the reward itself multiplies, so the pre-launch reading cannot drift from the
# trained reward.
_MATRIX_MIX_SPREADS: dict[str, Callable[[MatrixGameSpec, float], float]] = {
    GRADING_GROUP_MIX: group_mix_reward_spread,
    GRADING_VS_FIXED_MIX: group_mix_reward_spread,
    GRADING_JOINT_WELFARE_GROUP_MIX: joint_welfare_reward_spread,
    GRADING_OTHER_PAYOFF_GROUP_MIX: other_payoff_reward_spread,
}


def _matrix_mix_spread(grading: str) -> Callable[[MatrixGameSpec, float], float] | None:
    """Return the per-mix spread function for a matrix expected-payoff grading, or None.

    The table for the fixed recipients, and the care family's own spread at the weight its name
    states. Without the second half a care arm's variant spreads would come back EMPTY, which
    `log_reward_spread` prints as nothing at all -- indistinguishable from a game with no variants to
    compare, on the one arm whose corpus mixes five games.
    """
    tabulated = _MATRIX_MIX_SPREADS.get(grading)
    if tabulated is not None:
        return tabulated
    alpha = care_alpha_of(grading)
    if alpha is None:
        return None
    return partial(care_reward_spread, alpha=alpha)


if TYPE_CHECKING:
    from collections.abc import Callable

    from games.arms import GameArm

logger = logging.getLogger(__name__)

# Ratio of widest to narrowest variant spread above which `log_reward_spread` warns.
MIXED_VARIANT_SPREAD_WARNING_RATIO = 2.0
# Below this many payoff variants there is nothing sharing a batch to compare.
MIN_VARIANTS_TO_COMPARE = 2


def variant_reward_spreads(arm: GameArm, opponent_coop_prob: float) -> dict[str, float]:
    """Report each payoff variant's within-group reward spread at one opponent cooperation rate.

    Built from the arm's own generated rows rather than from a table of games, so it describes the
    corpus that will actually be trained on -- including a variant added upstream that no list here
    knows about. Per mix rather than aggregated over mixes, because the compression this exists to
    surface is mix-dependent and aggregating hides it: the stag ladder's risky rung is the *widest*
    variant against a mostly-defecting opponent and the narrowest by an order of magnitude against a
    mostly-cooperating one, which is the case that actually trains.

    Covers the gradings whose reward has a within-group range a row can state: the expected-payoff
    gradings, where it is an expectation over the opponent's action mix, and the announced-rule trust
    grading, where the reward is affine in the amount sent so the two corner answers bound it. The
    claim game's spreads come from `nash_demand_variant_reward_spreads` and the shared undertaking's
    from `threshold_goods_variant_reward_spreads`, both of which read a distribution of figures rather
    than a cooperation probability; `log_reward_spread` picks whichever the arm's
    grading calls for. The gradings left out are the ones whose spread is not a property of the row
    at all -- the iterated return, the unilateral split, the answer-shape rubric, and the strategy
    method, whose range depends on the share the model writes.
    """
    if arm.grading in {GRADING_NASH_DEMAND_GROUP_MIX, GRADING_NASH_DEMAND_SELF}:
        return nash_demand_variant_reward_spreads(arm)
    if arm.grading in THRESHOLD_GOODS_GRADINGS:
        return threshold_goods_variant_reward_spreads(arm)
    if arm.grading == GRADING_TRUSTOR_PAYOFF_STATED_RULE:
        return _trust_variant_reward_spreads(arm)
    if arm.grading == GRADING_MIN_EFFORT_GROUP_MIX:
        return min_effort_variant_reward_spreads(arm)
    mix_spread = _matrix_mix_spread(arm.grading)
    if mix_spread is None:
        return {}
    rows = generate_prompt_rows(arm.game_id, arm.grading, split="train")
    if arm.payoff_variants:
        rows = [row for row in rows if row["payoff_variant"] in arm.payoff_variants]
    return {
        str(row["payoff_variant"]): mix_spread(
            MatrixGameSpec(
                game_id=str(row["game_id"]),
                payoff_cc=float(row["payoff_cc"]),
                payoff_cd=float(row["payoff_cd"]),
                payoff_dc=float(row["payoff_dc"]),
                payoff_dd=float(row["payoff_dd"]),
            ),
            opponent_coop_prob,
        )
        for row in rows
    }


def nash_demand_variant_reward_spreads(arm: GameArm) -> dict[str, float]:
    """Report each windfall's within-group reward spread for the simultaneous-claim game.

    Same decision as the matrix version serves -- may these variants share a batch -- measured the
    way this game's reward works. There is no opponent action mix here, so the spread is taken at
    `games.payoffs.nash_demand_reference_claims`, one distribution expressed as fractions of the
    windfall so every variant is priced at the same shape. The self-graded arm has no distribution at
    all, and its span over the whole answer grid is the number that matters instead.

    The expected reading is one identical spread per windfall: after dividing by the windfall the
    three surfaces coincide, so the ratio is 1.0 and running all three in one arm is safe. A spread
    that differs across windfalls means the reward stopped being scale-free, which is a bug in the
    reward rather than a property of the game.
    """
    rows = generate_prompt_rows(arm.game_id, arm.grading, split="train")
    if arm.payoff_variants:
        rows = [row for row in rows if row["payoff_variant"] in arm.payoff_variants]
    spreads: dict[str, float] = {}
    for row in rows:
        spec = NashDemandSpec(game_id=str(row["game_id"]), windfall=int(str(row["windfall"])))
        spreads[str(row["payoff_variant"])] = (
            nash_demand_self_reward_span(spec)
            if arm.grading == GRADING_NASH_DEMAND_SELF
            else nash_demand_group_reward_span(spec, nash_demand_reference_claims(spec))
        )
    return spreads


def threshold_goods_variant_reward_spreads(arm: GameArm) -> dict[str, float]:
    """Report each prize variant's within-group reward spread for the shared undertaking.

    Same decision the matrix version serves -- may these variants share a batch -- measured the way this
    game's reward works. There is no opponent action mix here, so the group-mix spread is taken at
    `games.payoffs.threshold_goods_reference_contributions`, one distribution expressed as fractions of
    the stock so every variant is priced at the same shape. The self-graded arm has no distribution at
    all, and its span over the whole answer grid is the number that matters instead.

    Read this as a reference rather than as a property of the game, which is the honest difference from
    the claim game's version: this reward's spread genuinely does move with the group's own figures --
    computed 2026-08-21, from 0.15 against a mostly-zero group to 0.44 against one already over the equal
    share -- so an unexpected number here says the reference no longer matches the corpus, not that the
    reward broke. The sweep is what prices this arm, and `threshold_met_rate` is the per-step reading.
    """
    rows = generate_prompt_rows(arm.game_id, arm.grading, split="train")
    if arm.payoff_variants:
        rows = [row for row in rows if row["payoff_variant"] in arm.payoff_variants]
    spreads: dict[str, float] = {}
    for row in rows:
        spec = ThresholdGoodsSpec(
            game_id=str(row["game_id"]),
            endowment=int(str(row["endowment"])),
            team_size=int(str(row["team_size"])),
            contribution_threshold=int(str(row["contribution_threshold"])),
            prize=int(str(row["prize"])),
        )
        spreads[str(row["payoff_variant"])] = (
            threshold_goods_self_reward_span(spec, max_prize=THRESHOLD_GOODS_MAX_PRIZE)
            if arm.grading == GRADING_THRESHOLD_GOODS_SELF
            else threshold_goods_group_reward_span(
                spec,
                threshold_goods_reference_contributions(spec),
                max_prize=THRESHOLD_GOODS_MAX_PRIZE,
            )
        )
    return spreads


def min_effort_variant_reward_spreads(arm: GameArm) -> dict[str, float]:
    """Report each minimum-effort variant's within-group reward spread at the reference distribution.

    Same decision this module exists to serve -- may these variants share a batch -- measured the way
    this game's reward works. There is no single opponent cooperation rate here: the counterparts are
    drawn from a whole level distribution, so the spread is taken at
    `games.payoffs.min_effort_reference_levels`, one completion at each level. That is the flattest
    distribution on the grid, so every variant is priced at the same shape and the four numbers are
    comparable, which is exactly what the shared normalising constant was chosen to preserve.

    The expected reading is 0.400 for `cheap-effort-pair`, 0.303 for `costly-effort-crew`, 0.132 for
    `cheap-effort-crew` and 0.100 for `costly-effort-pair`. A ratio warning across an arm's own
    variants is therefore expected for any arm that pins none of them -- which no registered arm does,
    each pinning one cell deliberately.

    What this CANNOT see is the realised spread of the corpus a run actually trains on, and that
    matters more here than for the matrix games: at a cost ratio of 0.5 with one counterpart a group
    split evenly between the two extremes has a spread of exactly 0.000 at every level.
    `games.min_effort_spread` measures that on a baseline sweep's own completions.
    """
    rows = generate_prompt_rows(arm.game_id, arm.grading, split="train")
    if arm.payoff_variants:
        rows = [row for row in rows if row["payoff_variant"] in arm.payoff_variants]
    spreads: dict[str, float] = {}
    for row in rows:
        spec = MinEffortSpec(
            game_id=str(row["game_id"]),
            n_levels=int(str(row["n_levels"])),
            benefit_per_level=float(str(row["benefit_per_level"])),
            cost_per_level=float(str(row["cost_per_level"])),
            team_size=int(str(row["team_size"])),
        )
        spreads[str(row["payoff_variant"])] = min_effort_group_reward_span(
            spec, min_effort_reference_levels(spec)
        )
    return spreads


def _log_claim_variant_spreads(arm: GameArm) -> None:
    """Log the claim game's per-windfall spreads, warning if they are not the same number.

    Kept apart from the matrix path because this game's spread does not depend on an opponent
    cooperation probability, and reporting it "at opponent mix 0.1" would name a quantity the reward
    never reads. Equal spreads across windfalls are the healthy reading; unequal ones say the reward
    stopped being scale-free in the windfall, which is the bug this logs to catch.
    """
    spreads = nash_demand_variant_reward_spreads(arm)
    logger.info("claim-game reward spread by windfall for %s: %s", arm.game_id, spreads)
    narrowest, widest = min(spreads.values()), max(spreads.values())
    if narrowest <= 0:
        logger.warning(
            "a windfall of this arm has no reward spread at all, so its prompts contribute no "
            "gradient: %s",
            spreads,
        )
        return
    if widest / narrowest >= MIXED_VARIANT_SPREAD_WARNING_RATIO:
        logger.warning(
            "this arm's windfalls have reward spreads differing %.1fx (%s), so the reward is no "
            "longer scale-free in the windfall. All three variants are meant to coincide after "
            "dividing by the total; read this as a reward bug before reading it as a design choice.",
            widest / narrowest,
            spreads,
        )


def _trust_variant_reward_spreads(arm: GameArm) -> dict[str, float]:
    """Report each announced return rate's reward spread, rebuilt from the arm's own rows.

    Independent of the opponent mix, because there is no opponent to mix: the counterpart's rule is
    published, so the whole within-group range is send-everything against send-nothing. The two
    registered rates come out at 0.267 and 0.333, a ratio of 1.25x, which is what makes them the
    first spread-matched variant pair on the slate and why mixing them in one batch would be a
    decision rather than an accident.
    """
    rows = generate_prompt_rows(arm.game_id, arm.grading, split="train")
    if arm.payoff_variants:
        rows = [row for row in rows if row["payoff_variant"] in arm.payoff_variants]
    return {
        str(row["payoff_variant"]): trustor_reward_spread(
            TrustSpec(
                game_id=str(row["game_id"]),
                endowment=int(str(row["endowment"])),
                multiplier=float(str(row["transfer_multiplier"])),
                stated_return_fraction=float(str(row["stated_return_fraction"])),
            ),
            max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
        )
        for row in rows
    }


def log_reward_spread(arm: GameArm) -> None:
    """Log the spread table, and warn when this arm's own variants should not share a batch.

    `games.train` logs the same table at launch, but by then a GPU is reserved and a model is
    loading; a plan that logs it before the sweep is what makes "run the compressed rungs as
    separate arms" a decision rather than a post-mortem. The warning fires on the worst of the
    reference mixes, since the mix a run actually operates at is not known until it has swept.
    """
    logger.info("within-group reward spread by game/variant:\n%s", reward_spread_report())
    if arm.grading in {GRADING_NASH_DEMAND_GROUP_MIX, GRADING_NASH_DEMAND_SELF}:
        _log_claim_variant_spreads(arm)
        return
    by_mix = {mix: variant_reward_spreads(arm, mix) for mix in GROUP_MIX_SPREAD_MIXES}
    # `reward_spread_report` walks the matrix games only, so an arm on a graded game has no line in
    # the table above and would otherwise reach a sweep with its spread unstated -- the number that
    # decides whether its variants may share a batch. Reported once where the spread does not depend
    # on the opponent mix, which is every grading with no opponent to mix.
    distinct = {tuple(sorted(spreads.items())) for spreads in by_mix.values() if spreads}
    if len(distinct) == 1:
        logger.info("this arm's own variant reward spreads: %s", dict(distinct.pop()))
    else:
        for mix, spreads in by_mix.items():
            if spreads:
                logger.info("this arm's own variant spreads at opponent mix %.2g: %s", mix, spreads)
    if any(len(spreads) < MIN_VARIANTS_TO_COMPARE for spreads in by_mix.values()):
        return
    for mix, spreads in by_mix.items():
        narrowest, widest = min(spreads.values()), max(spreads.values())
        if narrowest <= 0:
            logger.warning(
                "at opponent mix %.2g one of this arm's payoff variants has no reward spread at "
                "all, so its prompts contribute no gradient there: %s",
                mix,
                spreads,
            )
            continue
        ratio = widest / narrowest
        if ratio >= MIXED_VARIANT_SPREAD_WARNING_RATIO:
            logger.warning(
                "at opponent mix %.2g this arm mixes payoff variants whose reward spread differs "
                "%.1fx (%s). The compressed variant trains proportionally weaker -- its spreads "
                "are its advantage magnitudes under scale_rewards='none' -- so a flat curve for "
                "it is expected rather than evidence. Run compressed variants as their own arms, "
                "or decide deliberately to mix them.",
                mix,
                ratio,
                spreads,
            )
