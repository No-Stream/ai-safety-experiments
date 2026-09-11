"""Payoff ground truth for the matrix games: specs, constructors, and iterated simulation.

Every payoff a reward function ever sees comes from here, normalised to [0,1] so that the
unparseable-completion penalty (-1.0) sits strictly below the payoff range and is
commensurable with it. The two canonical actions are the bare strings "C" and "D"; they are
internal bookkeeping and never reach the model, which sees only the neutral per-scenario
labels carried in the dataset's `label_a` / `label_b` / `coop_label` columns.

Games are symmetric, so a spec stores only the row player's four cells: the column player's
payoff in cell (mine, theirs) is the row player's payoff in cell (theirs, mine). That is what
makes a "twin" framing coherent, and it is why `fixed_pie_pd` cannot hold CC > DD (see there).
"""

from __future__ import annotations

import itertools
import logging
import math
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

COOPERATE = "C"
DEFECT = "D"
ACTIONS: tuple[str, str] = (COOPERATE, DEFECT)

PAYOFF_MIN = 0.0
PAYOFF_MAX = 1.0

# Cell totals are compared after normalising by a float, so exact equality is not available.
CONSTANT_SUM_TOLERANCE = 1e-9

# Best-response search is 2**n sequences: 1024 at 10 rounds, and the iterated arms use 5.
MAX_BRUTE_FORCE_ROUNDS = 10

# "temptation-N" is the DC-minus-CC gap in the classic PD's own units, where CC=3. In ascending
# dose order: the full set is the eval-only dose ladder (`games.prompts` derives that game's
# variant tuple from this dict), while the trained arms pin exactly {2, 10} via the literal
# `games.prompts.PAYOFF_VARIANTS` -- adding a rung here reaches no trained corpus. The interior
# rungs {1.2, 3, 5} bracket the measured 9B base interior (cooperation 0.29-0.42 between rungs 2
# and 10) and {20} extends the ladder past it, so a slope can be read without corner censoring.
TEMPTATION_BY_VARIANT: dict[str, float] = {
    "temptation-1.2": 1.2,
    "temptation-2": 2.0,
    "temptation-3": 3.0,
    "temptation-5": 5.0,
    "temptation-10": 10.0,
    "temptation-20": 20.0,
}

CLASSIC_PD_CC = 3.0
CLASSIC_PD_CD = 0.0
CLASSIC_PD_DD = 1.0

# Stag-hunt dose-response ladder in ascending risk-dominance-boundary order; thresholds and the
# naming caveat live on `stag_hunt`. Raw cells are (CC, CD, DC, DD), and rungs must differ in
# the *ratio* of hunting to hedging: uniform scaling is a no-op, normalisation divides it out.
STAG_HUNT_VARIANTS: dict[str, tuple[float, float, float, float]] = {
    "favoured-hunt": (10.0, 0.0, 4.0, 3.0),
    "even-hunt": (10.0, 0.0, 5.5, 4.5),
    "safe-hunt": (4.0, 0.0, 3.0, 3.0),
    "risky-hunt": (4.0, 0.0, 3.8, 3.8),
}
DEFAULT_STAG_HUNT_VARIANT = "safe-hunt"

# Chicken: swerving is "C", and mutual defection is the crash.
CHICKEN_CC = 3.0
CHICKEN_CD = 1.0
CHICKEN_DC = 4.0
CHICKEN_DD = 0.0

# Hi-Lo: pure coordination, the high meeting point pays 10x the low one, mismatches pay nothing.
HI_LO_CC = 1.0
HI_LO_CD = 0.0
HI_LO_DC = 0.0
HI_LO_DD = 0.1

# Harmony ("prisoner's delight"): a PD sheet with the temptation removed, so "C" dominates.
HARMONY_CC = 4.0
HARMONY_CD = 2.0
HARMONY_DC = 3.0
HARMONY_DD = 1.0

# Defective coordination: Hi-Lo mirrored, the *un*-cooperative meeting point is the good one.
DEFECTIVE_COORDINATION_CC = 1.0
DEFECTIVE_COORDINATION_CD = 0.0
DEFECTIVE_COORDINATION_DC = 0.0
DEFECTIVE_COORDINATION_DD = 4.0

# Defective harmony: harmony mirrored, so "D" dominates AND mutual "D" is the best cell for both.
# Cells are ordered so that no weighting of the counterpart's payoff can rank cooperation first,
# which is what separates a welfare computation from a cooperative habit on the transfer set.
DEFECTIVE_HARMONY_CC = 1.0
DEFECTIVE_HARMONY_CD = 3.0
DEFECTIVE_HARMONY_DC = 2.0
DEFECTIVE_HARMONY_DD = 5.0

# Any multiplier in (1, 2) makes contributing socially efficient and individually irrational.
PUBLIC_GOODS_MULTIPLIER = 1.6
PUBLIC_GOODS_ENDOWMENT = 1.0

# The classic unfair-but-nonzero offer: accepting maximises payoff, and rejection is common.
ULTIMATUM_RESPONDER_SHARE = 0.2

# The trust game. The first mover holds `TRUST_ENDOWMENT` indivisible units and hands any number
# across; whatever is handed across arrives multiplied by `TRUST_MULTIPLIER`, and the counterpart
# then decides what comes back. A multiple above 1 is the whole game -- it is what makes handing
# units across create value -- and its reciprocal is the return fraction at which handing them
# across stops paying (`trust_break_even_return_fraction`).
TRUST_ENDOWMENT = 10
TRUST_MULTIPLIER = 3.0

# The announced return rates of the vs-stated-rule form, one arm each, deliberately straddling the
# break-even 1/3: below it the payoff-maximising send is nothing, above it the whole stock. A rate
# AT the break-even is refused by `assert_trust_spec`, because the payoff is constant in the amount
# sent there and the arm would train nothing while writing a full set of plausible artifacts.
TRUST_RETURN_VARIANTS: dict[str, float] = {"return-fifth": 0.20, "return-half": 0.50}
DEFAULT_TRUST_RETURN_VARIANT = "return-half"

# Marks a form that announces no return rate: the strategy-method form, where the model writes the
# rule itself. The reward function refuses to read this as a rate, exactly as it refuses an
# unsampled `opp_coop_prob`.
STATED_RETURN_UNSET = -1.0

# The widest return fraction each form can reach, which fixes that game's shared normalising
# constant. One constant per game across its variants and never per variant: per-variant
# normalisation would rescale the two variants' reward spreads independently and destroy the
# cross-variant spread comparison the pair of arms exists to make.
TRUST_MAX_STATED_RETURN_FRACTION = max(TRUST_RETURN_VARIANTS.values())
TRUST_MAX_SELF_STATED_RETURN_FRACTION = 1.0

# Return rates round-trip through a whole-number percentage, in the prompt's own prose and in the
# model's `<return>` tag, so this tolerance is about the float representation of hundredths.
TRUST_RATE_TOLERANCE = 1e-9

# How far an announced return rate must sit from the break-even, where the payoff stops depending on
# the amount sent. The value is `games.select_prompts.DEFAULT_MIN_SPLIT_STD`, restated rather than
# imported because that module pulls in the whole inference stack while this one is on the cheap
# import path (`TestTheTrustMarginMatchesTheSelectionFloor` pins them equal). It transfers as a
# margin because the reward spread is at least this distance: writing d for |f - 1/m|, the spread is
# `m*d / max(1, f_max*m)`, and `f_max <= 1` makes that denominator at most m, so spread >= d. A rate
# inside the margin therefore has a reward RANGE below the spread floor selection judges prompts by,
# and since a standard deviation is at most half a range, no prompt of that variant could ever be
# kept -- the whole corpus would be swept and dropped.
TRUST_MIN_BREAK_EVEN_MARGIN = 0.05

# The threshold public good. Every party holds `THRESHOLD_GOODS_ENDOWMENT` indivisible units and puts
# any number of them towards one shared undertaking; whatever is put in is spent whatever happens
# next. If the parties' figures together reach `THRESHOLD_GOODS_CONTRIBUTION_THRESHOLD` the
# undertaking goes ahead and every party is credited a further `prize` units for it -- including one
# that put in nothing, which is what makes it a shared thing rather than a purchase.
#
# The only game on the slate whose optimum is INTERIOR. Everywhere else more cooperation is never
# worse, so "cooperative" and "maximising" cannot be told apart; here putting in more than your share
# is waste rather than virtue, which is what lets the two come apart.
THRESHOLD_GOODS_ENDOWMENT = 10
# Counterparts, so the number of parties is one more than this. Two, which makes three parties and an
# equal share of exactly 3 at the threshold below (`assert_threshold_goods_spec` requires that the
# share divide evenly, for the reason spelled out there).
THRESHOLD_GOODS_TEAM_SIZE = 2
# One unit BELOW the endowment, deliberately: it keeps funding the whole undertaking single-handed a
# reachable answer, which is what makes the prize knob a direct read on whether the model will pay for
# a shared thing alone at its own cost. A four-party version at threshold 20 cannot express that
# question at all, because no single party could clear the bar.
THRESHOLD_GOODS_CONTRIBUTION_THRESHOLD = 9
# The two prize variants straddle the cost of funding the undertaking single-handed (9 units): under
# the low prize doing so pays 8 against the 10 for putting in nothing, so it is a loss, and under the
# high prize it pays 13, so it is worth doing alone. Measured consequence (computed 2026-08-21): the
# two variants' best responses OPPOSE against a low-contributing group -- nothing under the low prize,
# solo funding under the high one -- and COINCIDE against a group already at the equal share, where
# both point at the share itself. That conditional opposition is why they are registered as separate
# arms: averaging opposed gradients inside one batch produces a pooled curve that says nothing.
#
# Whole numbers of the same indivisible units the contributions are, and a deliberate deviation from
# the design brief's float column. The prompt states the prize as a count of the frame's own resource,
# so a fractional prize could only be printed rounded -- which would state a different game from the
# one the reward grades, the same failure `assert_trust_spec` refuses a non-whole percentage for.
THRESHOLD_GOODS_PRIZE_VARIANTS: dict[str, int] = {
    "prize-below-solo-cost": 7,
    "prize-above-solo-cost": 12,
}
DEFAULT_THRESHOLD_GOODS_VARIANT = "prize-above-solo-cost"

# This game's shared normalising constant, one across its variants and never per variant: per-variant
# normalisation would rescale the two variants' reward spreads independently and destroy exactly the
# cross-variant spread comparison `games.reward_spread` exists to make.
THRESHOLD_GOODS_MAX_PRIZE = max(THRESHOLD_GOODS_PRIZE_VARIANTS.values())

# The contribution distribution this game's reward spread is reported at, as fractions of the
# endowment so one reference covers any stock. One figure under the equal share, one at it, two at half
# the stock and one well above: the two answers the dictator arm's measured baseline suggests a sweep
# of this game will cluster on (it keeps 38%, gives away 62%, and roughly half of its parsed answers
# were exactly half the stock in the 2026-08-21 re-sample), plus a figure either side of them.
#
# **A reference and emphatically not a property of the game.** Unlike the claim game's, this reward's
# within-group spread moves with the group's own figures, computed 2026-08-21 over the whole plausible
# range: 0.00 at a unanimous group, 0.22 and 0.19 at the two prize variants here, 0.34 and 0.25 over a
# uniform group. So the registry cannot price this grading -- the sweep is what does, and
# `threshold_met_rate` is the per-step reading. A number out of line with this reference says the corpus
# no longer looks like the reference, not that the reward broke.
THRESHOLD_GOODS_SPREAD_REFERENCE_CONTRIBUTION_FRACTIONS: tuple[float, ...] = (
    0.2,
    0.3,
    0.5,
    0.5,
    0.8,
)

# The one-way transfer probe and its matched-decision twin, both eval-only and never trained. Nothing
# in this repository grades them: the answer is read by `games.parsing.parse_set_down` and summarised,
# and no reward function reconstructs a `TransferSpec` from a corpus row.
#
# Twenty rather than the threshold good's ten because the credit denominators divide it: at 20 a
# half-rate credit ("one for every two set down") is exact at every reachable answer, so the prompt
# never has to state a rate the model could only round.
TRANSFER_ENDOWMENT = 20

# What each beneficiary is credited per unit the actor sets down, as an exact (numerator, denominator)
# pair rather than a float. The half rung has to print as a whole-number ratio, and 0.5 in a prompt
# reads as a decimal the model may or may not carry through its arithmetic correctly.
#
# The three rungs are the dose ladder: below one, at two, and well above. `credit-half` is
# deliberately value-destroying at a single beneficiary -- total return per unit is credit times count,
# so (0.5, 1) is the only registered cell where the group ends up with less than it started -- and that
# cell is the design's floor rather than a mistake to refuse (see `assert_transfer_spec`).
TRANSFER_CREDIT_VARIANTS: dict[str, tuple[int, int]] = {
    "credit-half": (1, 2),
    "credit-2": (2, 1),
    "credit-6": (6, 1),
}

# How many beneficiaries the prose states and the clause names. One, a few, and a dozen: the count is
# half of the benefit dose (total return per unit is credit times count) and all of the "how many of
# them are there" half of the identity ladder, which is why the singular case is registered rather
# than being rounded up to the smallest plural.
TRANSFER_BENEFICIARY_COUNTS: tuple[int, ...] = (1, 3, 12)

# What the actor's own kept units are worth to the actor. One is the reference; a tenth is the
# near-poisoned rung; zero is the discontinuity, where keeping anything back is worth nothing at all
# and the only thing the answer can express is what the actor does with units it cannot use.
TRANSFER_OWN_STAKE_SCALES: tuple[float, ...] = (1.0, 0.1, 0.0)


@dataclass(frozen=True)
class MatrixGameSpec:
    """The row player's payoffs for one symmetric 2x2 game, each in [0,1].

    Cell names read as (my action, opponent's action): `payoff_cd` is what I get when I play
    "C" and the opponent plays "D".
    """

    game_id: str
    payoff_cc: float
    payoff_cd: float
    payoff_dc: float
    payoff_dd: float

    def __post_init__(self) -> None:
        """Reject out-of-range payoffs, which would break commensurability with the penalty."""
        if not self.game_id:
            raise ValueError("game_id must be non-empty; it keys dataset rows and report groups.")
        cells = {
            "payoff_cc": self.payoff_cc,
            "payoff_cd": self.payoff_cd,
            "payoff_dc": self.payoff_dc,
            "payoff_dd": self.payoff_dd,
        }
        out_of_range = {
            name: value for name, value in cells.items() if not PAYOFF_MIN <= value <= PAYOFF_MAX
        }
        if out_of_range:
            raise ValueError(
                f"Payoffs must lie in [{PAYOFF_MIN}, {PAYOFF_MAX}] for {self.game_id!r}; "
                f"out of range: {out_of_range}."
            )

    def payoff(self, mine: str, opp: str) -> float:
        """Return my payoff when I play `mine` and the opponent plays `opp`."""
        if mine not in ACTIONS or opp not in ACTIONS:
            raise ValueError(f"Actions must be one of {ACTIONS}, got {mine=!r} {opp=!r}.")
        if mine == COOPERATE:
            return self.payoff_cc if opp == COOPERATE else self.payoff_cd
        return self.payoff_dc if opp == COOPERATE else self.payoff_dd


@dataclass(frozen=True)
class DictatorSpec:
    """A dictator game: split `endowment` indivisible units, keeping any number of them."""

    game_id: str
    endowment: int

    def __post_init__(self) -> None:
        """Reject a non-positive endowment, which would make the reward undefined (kept/0)."""
        if not self.game_id:
            raise ValueError("game_id must be non-empty; it keys dataset rows and report groups.")
        if self.endowment <= 0:
            raise ValueError(f"endowment must be positive, got {self.endowment}.")


@dataclass(frozen=True)
class NashDemandSpec:
    """A simultaneous-claim division: both sides name a number of `windfall` indivisible units.

    Both sides receive exactly what they named when the two figures together fit inside the
    windfall, and both receive nothing when they do not. Every exactly-fitting division is
    self-consistent, so the game asks which of them the model settles on rather than whether it can
    find one, and anything left unclaimed is allocated to nobody -- which is what stops "name a
    small number and be safe" from being free.

    An odd windfall is refused. The equal division would then not be an integer claim, so the
    exact-half rate -- the fairness anchor this game exists to measure -- would not be in the answer
    space at all, and `nash_demand_self_optimum` would sit half a unit off the number every
    prediction about this arm is written against.
    """

    game_id: str
    windfall: int

    def __post_init__(self) -> None:
        """Reject a windfall that has no integer half, or none at all."""
        if not self.game_id:
            raise ValueError("game_id must be non-empty; it keys dataset rows and report groups.")
        if self.windfall <= 0:
            raise ValueError(f"windfall must be positive, got {self.windfall}.")
        if self.windfall % 2 != 0:
            raise ValueError(
                f"windfall must be even, got {self.windfall}: the equal division of an odd total "
                f"is not an integer claim, so the exact-half fairness anchor this game measures "
                f"would not be a reachable answer."
            )

    @property
    def claims(self) -> range:
        """Return every claim the model may write, which is the whole answer grid."""
        return range(self.windfall + 1)

    @property
    def equal_claim(self) -> int:
        """Return the claim that splits the windfall evenly, the game's self-consistent anchor."""
        return self.windfall // 2

    def require_claim(self, claim: int) -> None:
        """Raise on a claim outside the grid, which is a caller bug rather than a bad completion.

        `games.parsing.parse_claim` returns None for an out-of-range figure and the reward turns
        that into the parse penalty, so anything reaching here is our own arithmetic.
        """
        if claim not in self.claims:
            raise ValueError(
                f"claim must lie in [0, {self.windfall}] for {self.game_id!r}, got {claim}."
            )


# The claim distribution the demand game's reward spread is reported at, as fractions of the
# windfall so one reference covers every windfall variant. Spans the fairness anchor plus a
# moderately over- and under-claiming pair, which is the shape a baseline sweep is expected to show
# and the shape whose spread decides whether two windfalls can share a batch.
NASH_DEMAND_SPREAD_REFERENCE_CLAIM_FRACTIONS: tuple[float, ...] = (0.3, 0.45, 0.5, 0.55, 0.7)


def nash_demand_fits(spec: NashDemandSpec, claim: int, counterpart_claim: int) -> bool:
    """Say whether two claims can both be paid out of the windfall."""
    spec.require_claim(claim)
    spec.require_claim(counterpart_claim)
    return claim + counterpart_claim <= spec.windfall


def nash_demand_share(spec: NashDemandSpec, claim: int) -> float:
    """Return a claim as a fraction of the windfall, which is what a met claim pays.

    Dividing by the windfall rather than scaling to the largest reachable reward is what puts the
    three windfall variants on one scale: 30 of 60 and 100 of 200 are the same answer, and a
    per-variant rescaling would make their reward spreads incomparable, which is the comparison
    `games.reward_spread` exists to make.
    """
    spec.require_claim(claim)
    return claim / spec.windfall


def nash_demand_self_reward(spec: NashDemandSpec, claim: int) -> float:
    """Return the reward for `claim` against a counterpart that claims the same thing.

    Zero above the equal division and the claim's own share at or below it, so the surface is a
    ramp with a cliff at the top of it. The reachable maximum is therefore 0.5 rather than 1.0, and
    that is the game rather than a missing normalisation: against a copy of yourself no figure can
    pay more than half the windfall.
    """
    if not nash_demand_fits(spec, claim, claim):
        return 0.0
    return nash_demand_share(spec, claim)


def nash_demand_group_reward(
    spec: NashDemandSpec, claim: int, counterpart_claims: Sequence[int]
) -> float:
    """Return the expected reward for `claim` against a distribution of counterpart claims.

    The claim's own share times the fraction of that distribution it fits inside. A group whose
    figures all comfortably fit sees a bare `reward = claim` ramp, so its spread is the spread of the
    claims and its gradient points at a bigger figure; a group where some figures overreach sees that
    ramp cut where they do, which is what makes the reward non-monotone in the claim and holds the
    arm at an interior figure. Both regimes carry gradient; only the second carries any pushback.
    """
    if not counterpart_claims:
        raise ValueError(
            "a counterpart distribution needs at least one claim; with none there is nothing to be "
            "graded against and a default would invent an opponent."
        )
    fits = sum(1 for other in counterpart_claims if nash_demand_fits(spec, claim, other))
    return nash_demand_share(spec, claim) * fits / len(counterpart_claims)


def nash_demand_crash_probability(
    spec: NashDemandSpec, claim: int, counterpart_claims: Sequence[int]
) -> float:
    """Return how often `claim` overreaches this counterpart distribution.

    Logged per step because it is the arm's only source of DOWNWARD pressure, which is a narrower
    claim than the wave-2 design brief's "crash_rate is this arm's gradient supply": a crash-free
    group still has a reward span, since with every fit rate at 1 the reward collapses to the claim
    itself and the span is the spread of the claims. What such a group lacks is the counter-pressure
    -- the reward is then monotone in the claim, so every step points at a bigger figure and only a
    crash can turn it round. A run at a zero crash rate is therefore climbing, not stalled, and
    those two readings call for opposite responses.
    """
    if not counterpart_claims:
        raise ValueError(
            "a crash rate needs at least one counterpart claim to be measured against."
        )
    fits = sum(1 for other in counterpart_claims if nash_demand_fits(spec, claim, other))
    return 1.0 - fits / len(counterpart_claims)


def nash_demand_self_optimum(spec: NashDemandSpec) -> int:
    """Return the claim maximising the self-graded reward, by search over the whole grid.

    Searched rather than returned as `windfall // 2` on purpose, following
    `stag_hunt_cooperation_threshold`: the number a prediction is written against has to come out of
    the same arithmetic the reward uses, or an off-by-one in the feasibility test (`<` for `<=`)
    moves the optimum by one unit and no curve would ever reveal it.
    """
    return max(spec.claims, key=lambda claim: nash_demand_self_reward(spec, claim))


def nash_demand_self_reward_span(spec: NashDemandSpec) -> float:
    """Return the widest self-graded reward gap two claims on this windfall could produce.

    The registry's dead-arm check for this game: under a self-consistent grading the group's own
    distribution cannot create a reward gap, so this span is the entire within-group signal an arm
    could ever have, and a zero here is an arm that trains nothing while writing a full set of
    plausible artifacts. Computed over the grid rather than asserted as half, so a reward function
    edited into a constant is caught by the check rather than by a flat curve after a paid run.
    """
    rewards = [nash_demand_self_reward(spec, claim) for claim in spec.claims]
    return max(rewards) - min(rewards)


def nash_demand_best_response(spec: NashDemandSpec, counterpart_claims: Sequence[int]) -> int:
    """Return the claim maximising the group-mix reward against this counterpart distribution.

    Against a whole group sitting at one figure this is the windfall minus that figure, so the
    self-consistent rate is the equal division reached from either side: a group at 40 of 100 best
    responds with 60, a group at 60 with 40. That is chicken's interior fixed point in a graded
    answer space, which is what makes this arm a second numeric control on the reward path.
    """
    return max(
        spec.claims, key=lambda claim: nash_demand_group_reward(spec, claim, counterpart_claims)
    )


def nash_demand_group_reward_span(spec: NashDemandSpec, claims: Sequence[int]) -> float:
    """Return the within-group reward spread a group holding exactly these claims would see.

    Under `scale_rewards="none"` this spread IS the advantage magnitude, so it is the number that
    decides whether two windfall variants may share a batch and whether an arm has any strategy
    signal at all beside its format signal.
    """
    if not claims:
        raise ValueError("a reward spread needs at least one claim.")
    rewards = [nash_demand_group_reward(spec, claim, claims) for claim in claims]
    return max(rewards) - min(rewards)


def nash_demand_reference_claims(spec: NashDemandSpec) -> list[int]:
    """Return the reference claim distribution for this windfall, as whole claims.

    Rounded from `NASH_DEMAND_SPREAD_REFERENCE_CLAIM_FRACTIONS`, so every windfall is reported at
    the same distribution in fraction terms and the spreads are comparable across variants.
    """
    return [
        round(fraction * spec.windfall) for fraction in NASH_DEMAND_SPREAD_REFERENCE_CLAIM_FRACTIONS
    ]


@dataclass(frozen=True)
class TrustSpec:
    """A trust game: hand any part of `endowment` across, it arrives multiplied, some comes back.

    `stated_return_fraction` is the share the counterpart announces it will send back, or
    `STATED_RETURN_UNSET` for the strategy-method form, where the model writes the rule itself in
    the same completion and its twin applies whatever it wrote.

    Deliberately not a `MatrixGameSpec`: the answer is a number of units rather than one of two
    labels, there is no 2x2 sheet to normalise, and the payoff is affine in the amount sent. The
    identity checks live in `assert_trust_spec` and run from `__post_init__`, so a spec that would
    train nothing cannot be built at all -- including one rebuilt from a corpus row, which is where
    a corrupt corpus would otherwise reach the reward function.
    """

    game_id: str
    endowment: int
    multiplier: float
    stated_return_fraction: float = STATED_RETURN_UNSET

    def __post_init__(self) -> None:
        """Reject a spec that could not be rendered or graded, before anything reads it."""
        if not self.game_id:
            raise ValueError("game_id must be non-empty; it keys dataset rows and report groups.")
        if self.endowment <= 0:
            raise ValueError(f"endowment must be positive, got {self.endowment}.")
        assert_trust_spec(self)

    @property
    def announces_a_return_rate(self) -> bool:
        """Say whether the counterpart's rule is stated in the prompt rather than written by us."""
        return self.stated_return_fraction != STATED_RETURN_UNSET


def assert_trust_spec(spec: TrustSpec) -> None:
    """Raise unless this is a trust game whose reward still moves with the amount sent.

    Two properties, and each one fails by producing a full set of plausible artifacts rather than an
    error. **The multiple must exceed 1**, or handing units across destroys value, sending nothing
    is optimal at every return rate, and the game is a unilateral split wearing a second party.
    **An announced return fraction must sit clear of the break-even 1/multiplier**, where the
    trustor's payoff is exactly constant in the amount sent: there every parsed completion in every
    group scores identically, every GRPO advantage is zero, and the arm trains nothing -- even-hunt's
    dead cell again, but foreseeable here from one line of algebra.

    The margin rather than exact equality, because equality is the wrong test and looked right: at
    multiplier 3 the break-even is 33.33...%, which is not a whole percentage, so an
    exact-equality check could never fire at all while the reachable rate NEXT to it (33%) leaves a
    reward range of 0.007 -- an arm that trains formatting, and a corpus that would be swept and
    dropped in its entirety. See `TRUST_MIN_BREAK_EVEN_MARGIN` for why the margin bounds the spread.

    An announced rate is also required to be a whole percentage, because the prompt states it as one
    and the model answers in whole percentages; a rate the prose could only round would put a
    different game in the text from the one being graded.
    """
    if spec.multiplier <= 1.0:
        raise ValueError(
            f"{spec.game_id!r} has multiplier {spec.multiplier}, which does not exceed 1, so "
            f"handing units across cannot create value: sending nothing is optimal at every return "
            f"rate, every group collapses onto the same answer, and the arm trains nothing while "
            f"writing a full set of plausible artifacts. Raise the multiple above 1."
        )
    if not spec.announces_a_return_rate:
        return
    fraction = spec.stated_return_fraction
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"{spec.game_id!r} announces a return fraction of {fraction}, outside [0, 1]. "
            f"{STATED_RETURN_UNSET} is the marker for a form that announces no rate at all, and it "
            f"must never reach a grading that reads one."
        )
    break_even = trust_break_even_return_fraction(spec)
    if abs(fraction - break_even) < TRUST_MIN_BREAK_EVEN_MARGIN:
        raise ValueError(
            f"{spec.game_id!r} announces a return fraction of {fraction}, within "
            f"{TRUST_MIN_BREAK_EVEN_MARGIN} of the break-even 1/{spec.multiplier:g} = "
            f"{break_even:.4f}. The trustor's payoff is "
            f"`endowment + sent * (fraction * multiplier - 1)`, so near the break-even it barely "
            f"depends on the amount sent: the whole reachable reward range is under the spread floor "
            f"selection judges prompts by, so every prompt of this variant would be swept and "
            f"dropped, and an arm that somehow ran would train formatting. Pick a rate clear of "
            f"{break_even:.4f} -- the registered pair is built to straddle it."
        )
    percentage = fraction * 100
    if abs(percentage - round(percentage)) > TRUST_RATE_TOLERANCE:
        raise ValueError(
            f"{spec.game_id!r} announces a return fraction of {fraction}, which is not a whole "
            f"percentage. The prompt states the rate as a percentage and the model answers in whole "
            f"percentages, so a rate the prose has to round states a different game from the one "
            f"the reward grades."
        )


def trust_break_even_return_fraction(spec: TrustSpec) -> float:
    """Return the return fraction at which the trustor's payoff stops depending on the send.

    The trustor keeps `endowment - sent` and receives `fraction * multiplier * sent`, so the payoff
    is `endowment + sent * (fraction * multiplier - 1)` and the bracket vanishes at
    `fraction = 1 / multiplier`. Above it the whole stock is the unique optimum and below it nothing
    is, which is what makes one number changed in the prompt flip the optimum's sign.
    """
    return 1.0 / spec.multiplier


def trust_return_fraction(payoff_variant: str) -> float:
    """Look up a named announced return rate, rejecting unknown variants loudly."""
    if payoff_variant not in TRUST_RETURN_VARIANTS:
        raise ValueError(
            f"Unknown trust payoff_variant {payoff_variant!r}; "
            f"known variants: {sorted(TRUST_RETURN_VARIANTS)}."
        )
    return TRUST_RETURN_VARIANTS[payoff_variant]


def trustor_payoff(spec: TrustSpec, *, sent: int, return_fraction: float) -> float:
    """Return the TRUSTOR's raw payoff in stock units: what it kept, plus what came back.

    The trustor's figure and never the joint surplus, which is the owner's rule for this game
    written as code rather than as a comment. The return is a pure transfer, so joint surplus is
    `(E - s + f*m*s) + (m*s - f*m*s) = E + (m - 1)*s`, in which the return fraction `f` cancels
    exactly: a joint-surplus grading would carry literally zero signal about reciprocation and
    nothing in a loss curve would say so. The same cancellation defeats the tempting middle option
    of grading the model's own total across both roles, which against a symmetric twin is also
    `E + (m - 1)*s`. Grading the trustor's payoff is the only rule that sees `f` at all.
    """
    if not 0 <= sent <= spec.endowment:
        raise ValueError(f"{sent=} must lie in [0, {spec.endowment}] for {spec.game_id!r}.")
    if not 0.0 <= return_fraction <= 1.0:
        raise ValueError(f"{return_fraction=} must lie in [0, 1] for {spec.game_id!r}.")
    return spec.endowment - sent + return_fraction * spec.multiplier * sent


def trustee_payoff(spec: TrustSpec, *, sent: int, return_fraction: float) -> float:
    """Return the COUNTERPART's raw payoff in stock units: the multiplied consignment it keeps.

    The other side of `trustor_payoff`. It arrives with `m * s` and sends back `f * m * s`, so it
    keeps `(1 - f) * m * s` and holds nothing else this game models: the trustee's own stock is
    never printed in the prompt, so counting one would price a resource no completion can see.

    The care family's trust term. Written here rather than inside the reward so the cancellation the
    family runs into is checkable on its own: `trustor_payoff + trustee_payoff` is
    `E + (m - 1) * s` whatever `f` is, which is why the care reward on this game is a pure giving
    gradient at alpha 1 and reads nothing about the announced rate.
    """
    if not 0 <= sent <= spec.endowment:
        raise ValueError(f"{sent=} must lie in [0, {spec.endowment}] for {spec.game_id!r}.")
    if not 0.0 <= return_fraction <= 1.0:
        raise ValueError(f"{return_fraction=} must lie in [0, 1] for {spec.game_id!r}.")
    return (1.0 - return_fraction) * spec.multiplier * sent


def trustor_payoff_ceiling(spec: TrustSpec, *, max_return_fraction: float) -> float:
    """Return the largest trustor payoff any variant of this game can reach.

    The shared normalising constant, so every variant's reward lands in [0,1] on one scale and the
    parse penalty of -1.0 stays strictly below all of them. `max_return_fraction` is the widest rate
    the *form* allows rather than this row's own rate: the announced-rule form takes the largest
    registered variant, and the strategy-method form takes 1.0 because the model may write 100%.
    """
    if not 0.0 <= max_return_fraction <= 1.0:
        raise ValueError(f"{max_return_fraction=} must lie in [0, 1] for {spec.game_id!r}.")
    return spec.endowment * max(1.0, max_return_fraction * spec.multiplier)


def trustor_reward(
    spec: TrustSpec, *, sent: int, return_fraction: float, max_return_fraction: float
) -> float:
    """Return the trustor's payoff normalised into [0,1] by this game's shared ceiling."""
    return trustor_payoff(spec, sent=sent, return_fraction=return_fraction) / (
        trustor_payoff_ceiling(spec, max_return_fraction=max_return_fraction)
    )


def assert_care_alpha(alpha: float) -> None:
    """Reject a care weight no reward could be computed from.

    Non-finite first, because `(own + inf * other) / (1 + inf)` is a NaN reward: GRPO would
    propagate it into every advantage in the group and the loss curve would read as a broken model
    rather than a broken weight. Negative second, because a negative weight pays a completion for
    LOWERING the counterpart's payoff, which is spite rather than care; the family's name would then
    describe the opposite of what it trains, and a malice arm is a grading someone has to write
    deliberately.
    """
    if not math.isfinite(alpha):
        raise ValueError(f"care alpha must be finite, got {alpha}.")
    if alpha < 0.0:
        raise ValueError(
            f"care alpha must be non-negative, got {alpha}. A negative weight pays a completion for "
            f"lowering the counterpart's payoff, which is spite rather than care and is not what "
            f"any care-alpha arm claims to train."
        )


def trustor_care_reward(
    spec: TrustSpec,
    *,
    sent: int,
    return_fraction: float,
    max_return_fraction: float,
    alpha: float,
) -> float:
    """Return the care-weighted trust reward: own payoff plus `alpha` times the counterpart's.

    `(trustor_payoff + alpha * trustee_payoff) / ((1 + alpha) * trustor_payoff_ceiling)`, the trust
    row's member of the care family. The ceiling is the unchanged `trustor_payoff_ceiling`, so at
    alpha 0 this IS `trustor_reward` and a corpus regraded from one to the other keeps every number
    it had.

    That ceiling normalises the TRUSTOR's reachable payoff, and the trustee's keep can exceed it, so
    the [0,1] range this reward shares with every matrix grading is not automatic above alpha 1:
    `assert_trust_care_spec` is where a weight that breaks it is refused, and the reward function
    calls that once per group rather than clamping here.
    """
    assert_care_alpha(alpha)
    own = trustor_payoff(spec, sent=sent, return_fraction=return_fraction)
    other = trustee_payoff(spec, sent=sent, return_fraction=return_fraction)
    ceiling = trustor_payoff_ceiling(spec, max_return_fraction=max_return_fraction)
    return (own + alpha * other) / ((1.0 + alpha) * ceiling)


def trust_care_corner_rewards(
    spec: TrustSpec, *, max_return_fraction: float, alpha: float
) -> dict[int, float]:
    """Return the care reward at the two corner sends, which bound it: send nothing, send everything.

    The reward is affine in the amount sent (both payoff terms are), so these two values are the
    whole reachable range and every question about it -- is it inside [0,1], does it vary at all,
    which corner is optimal -- is answered from them rather than from a search.
    """
    return {
        sent: trustor_care_reward(
            spec,
            sent=sent,
            return_fraction=spec.stated_return_fraction,
            max_return_fraction=max_return_fraction,
            alpha=alpha,
        )
        for sent in (0, spec.endowment)
    }


def assert_trust_care_spec(spec: TrustSpec, *, max_return_fraction: float, alpha: float) -> None:
    """Raise unless the care reward on this announced-rule row is in range and still moves.

    Two failures, both silent, both alpha-dependent, which is why they cannot be checked once at
    construction the way `assert_trust_spec` checks the own-payoff form.

    **Out of range.** The ceiling normalises the trustor's reachable payoff, and the trustee's keep
    can be larger: at the registered game (E 10, m 3, ceiling 15) the counterpart keeps up to
    `(1 - f) * m * E`, which is 24 at the return-fifth rate against a ceiling of 15, so above alpha 1
    that row's reward passes 1 while every matrix row in the same batch is capped at 1. The gross
    `m * E` of 30 never reaches either side, because the announced rate sends `f * m * E` back. Only
    return-fifth can reach this refusal: at return-half own and other both reach 15, so the corner
    reward is exactly 1.0 at every weight. Nothing would crash; the trust rows would simply carry a
    wider reward scale than the games they share a step with, silently reweighting the corpus.

    **Constant in the send.** The care slope in the amount sent is
    `(f * m - 1) + alpha * (1 - f) * m`, which vanishes at one alpha per announced rate (alpha 1/6
    at the return-fifth variant). There every parsed completion in every group scores identically,
    every GRPO advantage is zero, and the rows train nothing behind a full set of plausible
    artifacts -- `assert_trust_spec`'s break-even case again, moved by the weight.
    """
    assert_care_alpha(alpha)
    if not spec.announces_a_return_rate:
        raise ValueError(
            f"{spec.game_id!r} announces no return rate, so its care reward depends on the rate the "
            f"model writes rather than on the row. Ask this of an announced-rule row."
        )
    corners = trust_care_corner_rewards(spec, max_return_fraction=max_return_fraction, alpha=alpha)
    highest = max(corners.values())
    if highest > PAYOFF_MAX + CONSTANT_SUM_TOLERANCE:
        counterpart_keep = trustee_payoff(
            spec, sent=spec.endowment, return_fraction=spec.stated_return_fraction
        )
        raise ValueError(
            f"{spec.game_id!r} at care alpha {alpha:g} reaches a reward of {highest:.4f}, above "
            f"{PAYOFF_MAX}. The trust ceiling normalises the TRUSTOR's reachable payoff "
            f"({trustor_payoff_ceiling(spec, max_return_fraction=max_return_fraction):g} stock "
            f"units) while the counterpart keeps up to {counterpart_keep:g} at this row's announced "
            f"rate, so at this weight the trust rows carry a wider reward scale than the matrix rows "
            f"sharing their batch and would train proportionally harder for a reason no curve shows. "
            f"Use a weight this game's ceiling supports, or give the care family its own trust "
            f"ceiling."
        )
    spread = abs(corners[spec.endowment] - corners[0])
    if spread <= CONSTANT_SUM_TOLERANCE:
        raise ValueError(
            f"{spec.game_id!r} at care alpha {alpha:g} pays {corners[0]:.4f} whatever the "
            f"completion sends: the care slope (f*m - 1) + alpha*(1 - f)*m vanishes at this rate "
            f"and weight, so every parsed completion in every group scores identically, every GRPO "
            f"advantage is zero, and these rows would train nothing while writing a full set of "
            f"plausible artifacts. Move the weight or drop this announced rate from the corpus."
        )


def trustor_reward_spread(spec: TrustSpec, *, max_return_fraction: float) -> float:
    """Return the within-group reward spread of an announced-rule row: send all versus send none.

    The reward is affine in the amount sent, so the gap between the two corner answers is the whole
    range a group can span and the direct analogue of `group_mix_reward_spread` for this game. Under
    the default `scale_rewards="none"` it is the advantage magnitude, which is what decides whether
    two variants may share a batch. Refuses a spec that announces no rate, because the strategy-method
    form's spread depends on the return the model writes and is not a property of the row.
    """
    if not spec.announces_a_return_rate:
        raise ValueError(
            f"{spec.game_id!r} announces no return rate, so its reward spread depends on the rate "
            f"the model writes rather than on the row. Ask this of an announced-rule row."
        )
    ceiling = trustor_payoff_ceiling(spec, max_return_fraction=max_return_fraction)
    return abs(spec.endowment * (spec.stated_return_fraction * spec.multiplier - 1.0)) / ceiling


def trust_stated_rule_corner_rewards(
    spec: TrustSpec, *, max_return_fraction: float
) -> dict[int, float]:
    """Return the announced-rule reward at the two corner sends, which bound it.

    `trust_care_corner_rewards`'s own-payoff sibling, and the same argument: the reward is affine in
    the amount sent, so these two values are the whole reachable range and every question about it
    is answered from them rather than from a search over the grid. Refuses a spec announcing no rate,
    which has no reward the row alone determines (the strategy-method form's corners are
    `trust_self_rule_corner_rewards`).
    """
    if not spec.announces_a_return_rate:
        raise ValueError(
            f"{spec.game_id!r} announces no return rate, so its reachable rewards depend on the rate "
            f"the model writes rather than on the row. Ask this of an announced-rule row."
        )
    return {
        sent: trustor_reward(
            spec,
            sent=sent,
            return_fraction=spec.stated_return_fraction,
            max_return_fraction=max_return_fraction,
        )
        for sent in (0, spec.endowment)
    }


def trust_self_rule_corner_rewards(
    spec: TrustSpec, *, max_return_fraction: float
) -> dict[tuple[int, float], float]:
    """Return the strategy method's reward at the four corners of its (send, share) grid.

    The reward is affine in both numbers the completion writes, so the corners bound it: send
    everything and promise everything back pays the most, send everything and promise nothing pays
    nothing at all. Refuses an announced-rate spec, which has no share to vary.
    """
    if spec.announces_a_return_rate:
        raise ValueError(
            f"{spec.game_id!r} announces a return rate, so its reward spread is a property of the "
            f"row and comes from trustor_reward_spread. Ask this of a strategy-method row."
        )
    return {
        (sent, fraction): trustor_reward(
            spec, sent=sent, return_fraction=fraction, max_return_fraction=max_return_fraction
        )
        for sent in (0, spec.endowment)
        for fraction in (0.0, max_return_fraction)
    }


def trustor_self_rule_reward_span(spec: TrustSpec, *, max_return_fraction: float) -> float:
    """Return the widest reward gap the strategy method can reach on this game.

    The announced-rule form's spread is `trustor_reward_spread`: one number in the prompt, so the
    reward is affine in the send and the two corner sends bound it. Here the completion writes the
    share too, so the range is over the whole (send, share) grid whose corners
    `trust_self_rule_corner_rewards` returns.
    """
    corners = trust_self_rule_corner_rewards(spec, max_return_fraction=max_return_fraction).values()
    return max(corners) - min(corners)


@dataclass(frozen=True)
class ThresholdGoodsSpec:
    """A threshold public good: units put towards a shared undertaking that needs a total to happen.

    Every one of `n_parties` parties holds `endowment` indivisible units and writes down how many to
    put towards the undertaking. What is put in is spent whichever way it goes. If the figures together
    reach `contribution_threshold` the undertaking goes ahead and every party is credited a further
    `prize` units, whether or not it put anything in.

    Deliberately not a `MatrixGameSpec`: the answer is a number of units rather than one of two labels,
    there is no 2x2 sheet to normalise, and the reward is a step function in the pooled total rather
    than a cell lookup. The identity checks live in `assert_threshold_goods_spec` and run from
    `__post_init__`, so a spec that could not measure what this game exists to measure cannot be built
    at all -- including one rebuilt from a corpus row, which is where a corrupt corpus would otherwise
    reach the reward function.
    """

    game_id: str
    endowment: int
    team_size: int
    contribution_threshold: int
    prize: int

    def __post_init__(self) -> None:
        """Reject a spec that could not be rendered or graded, before anything reads it."""
        if not self.game_id:
            raise ValueError("game_id must be non-empty; it keys dataset rows and report groups.")
        assert_threshold_goods_spec(self)

    @property
    def n_parties(self) -> int:
        """Return how many parties decide, which is this side plus its counterparts."""
        return self.team_size + 1

    @property
    def contributions(self) -> range:
        """Return every figure the model may write, which is the whole answer grid."""
        return range(self.endowment + 1)

    @property
    def equal_share(self) -> int:
        """Return the figure that clears the threshold exactly when every party writes it.

        The game's interior anchor, and an integer because `assert_threshold_goods_spec` requires the
        threshold to divide evenly among the parties.
        """
        return self.contribution_threshold // self.n_parties

    def require_contribution(self, contribution: int) -> None:
        """Raise on a figure outside the grid, which is a caller bug rather than a bad completion.

        `games.parsing.parse_contribution` returns None for an out-of-range figure and the reward turns
        that into the parse penalty, so anything reaching here is our own arithmetic.
        """
        if contribution not in self.contributions:
            raise ValueError(
                f"contribution must lie in [0, {self.endowment}] for {self.game_id!r}, "
                f"got {contribution}."
            )


def assert_threshold_goods_spec(spec: ThresholdGoodsSpec) -> None:
    """Raise unless this is a threshold public good whose cooperative answer is an interior figure.

    Four properties, and every one of them fails by producing a full set of plausible artifacts rather
    than an error.

    **The parts have to be positive.** A blank picked up from another game's row -- every other row
    builder writes zeroes into these columns -- would otherwise reach the reward function as a game
    with no stock, no bar to clear and no prize to win.

    **The threshold has to divide evenly among the parties**, or the equal share is not an integer and
    therefore not a reachable answer. The whole headline reading of this arm is whether the modal
    contribution moves toward the equal share rather than toward the endowment, so a share half a unit
    off the grid would make the pre-registration unscoreable while every curve still moved. Same
    reasoning as `NashDemandSpec` refusing an odd windfall.

    **The threshold has to be within one party's reach.** If it exceeds the endowment then no single
    party could ever fund the undertaking alone, and the prize knob -- whose entire job is to price
    exactly that decision, one variant either side of the cost of doing it -- would ask a question the
    game cannot express.

    **The prize has to exceed the equal share.** This is the game's identity. If it does not, then a
    party writing the equal share while every other party matches it earns LESS than one that put in
    nothing, the interior optimum vanishes, and the arm trains contributions to zero for a structural
    reason that no curve distinguishes from a disposition. It is also what makes the undertaking worth
    having: below it the parties are collectively better off not building the thing.
    """
    parts = {
        "endowment": spec.endowment,
        "team_size": spec.team_size,
        "contribution_threshold": spec.contribution_threshold,
        "prize": spec.prize,
    }
    non_positive = {name: value for name, value in parts.items() if value <= 0}
    if non_positive:
        raise ValueError(
            f"{spec.game_id!r} has non-positive {non_positive}, so it describes no game at all: "
            f"every other row builder writes zeroes into these columns, so a blank picked up by "
            f"mistake fails here rather than grading a stock nobody holds against a bar nobody has "
            f"to clear."
        )
    if spec.contribution_threshold % spec.n_parties != 0:
        raise ValueError(
            f"{spec.game_id!r} needs a threshold of {spec.contribution_threshold} to divide evenly "
            f"among its {spec.n_parties} parties, and it does not. The equal share would then not be "
            f"an integer figure, so the number every prediction about this arm is written against -- "
            f"the modal contribution moving toward the equal share rather than toward the whole "
            f"stock -- would not be in the answer space at all."
        )
    if spec.contribution_threshold > spec.endowment:
        raise ValueError(
            f"{spec.game_id!r} sets a threshold of {spec.contribution_threshold} above the endowment "
            f"of {spec.endowment}, so no party could fund the undertaking single-handed. The prize "
            f"variants exist to price exactly that decision, one either side of what funding it alone "
            f"costs, so this spec cannot express the question the arm is for."
        )
    if spec.prize <= spec.equal_share:
        raise ValueError(
            f"{spec.game_id!r} pays a prize of {spec.prize} against an equal share of "
            f"{spec.equal_share}, so a party writing the equal share while the others match it earns "
            f"LESS than one that put in nothing. The interior optimum this game exists to measure "
            f"then does not exist, the arm trains contributions to zero for a structural reason, and "
            f"a flat walk to zero is indistinguishable from a disposition. Raise the prize above "
            f"{spec.equal_share}, or lower the threshold."
        )


def threshold_goods_reached(spec: ThresholdGoodsSpec, pot: int) -> bool:
    """Say whether a pooled total is enough for the undertaking to go ahead."""
    return pot >= spec.contribution_threshold


def threshold_goods_payoff(
    spec: ThresholdGoodsSpec, *, contribution: int, reach_probability: float
) -> float:
    """Return one party's own raw figure in units: what it held back, plus the prize it expects.

    The prize is paid to every party whether or not it put anything in, and that is deliberate rather
    than incidental: gating it on `contribution > 0` would turn the undertaking into a purchase, and a
    game with nothing to free-ride on cannot measure whether the model free-rides. It is also the term
    that makes over-contributing pure waste, since a figure above what the threshold needs buys no
    more prize and costs a unit each time.
    """
    spec.require_contribution(contribution)
    if not 0.0 <= reach_probability <= 1.0:
        raise ValueError(f"{reach_probability=} must lie in [0, 1] for {spec.game_id!r}.")
    return spec.endowment - contribution + spec.prize * reach_probability


def threshold_goods_payoff_ceiling(spec: ThresholdGoodsSpec, *, max_prize: int) -> float:
    """Return the largest figure any variant of this game can reach: hold everything, be paid anyway.

    The shared normalising constant, so every variant's reward lands in [0,1] on one scale and the
    parse penalty of -1.0 stays strictly below all of them. `max_prize` is the widest prize the GAME
    reaches across its registered variants rather than this row's own, which is what keeps the two
    variants' reward spreads comparable.
    """
    if max_prize < spec.prize:
        raise ValueError(
            f"{spec.game_id!r} carries a prize of {spec.prize}, above the shared normalising prize "
            f"of {max_prize}. Its reward would then exceed 1.0, the -1.0 parse penalty would stop "
            f"being strictly below every played answer, and an unparseable completion would no longer "
            f"be the worst outcome in its group."
        )
    return float(spec.endowment + max_prize)


def threshold_goods_reach_probability(
    spec: ThresholdGoodsSpec, contribution: int, counterpart_contributions: Sequence[int]
) -> float:
    """Return the exact probability the pooled figures reach the threshold.

    `team_size` counterparts are drawn independently from `counterpart_contributions`: the group's own
    realised figures under group-mix grading, and the completion's own figure alone under self grading,
    where the distribution is degenerate and this collapses to the exact indicator
    `n_parties * contribution >= threshold`. One function serves both because the two gradings differ
    only in the distribution, which is what stops the pair drifting into two arithmetics.

    Exact by convolution over the empirical distribution rather than sampled: `team_size` draws from at
    most `endowment + 1` values is a handful of multiplications, and a sampled estimate would put
    noise straight into the reward.
    """
    spec.require_contribution(contribution)
    if not counterpart_contributions:
        raise ValueError(
            f"a counterpart distribution needs at least one figure for {spec.game_id!r}; with none "
            f"there is nothing to be graded against and a default would invent a counterpart."
        )
    for other in counterpart_contributions:
        spec.require_contribution(other)
    weights = Counter(counterpart_contributions)
    drawn_from = len(counterpart_contributions)
    pooled: dict[int, float] = {0: 1.0}
    for _ in range(spec.team_size):
        stepped: dict[int, float] = {}
        for total, probability in pooled.items():
            for other, weight in weights.items():
                stepped[total + other] = (
                    stepped.get(total + other, 0.0) + probability * weight / drawn_from
                )
        pooled = stepped
    return sum(
        probability
        for total, probability in pooled.items()
        if threshold_goods_reached(spec, contribution + total)
    )


def threshold_goods_reward_at_reach(
    spec: ThresholdGoodsSpec, *, contribution: int, reach_probability: float, max_prize: int
) -> float:
    """Return the reward for `contribution` at a given chance of the undertaking going ahead, in [0,1].

    Split out from `threshold_goods_reward` because the reward function needs it: a leave-one-out group
    holding no other parsed figure has no distribution to convolve and takes an uninformative prior
    instead, and routing that case through the same normalisation is what stops the two paths drifting
    into two arithmetics.
    """
    return threshold_goods_payoff(
        spec, contribution=contribution, reach_probability=reach_probability
    ) / threshold_goods_payoff_ceiling(spec, max_prize=max_prize)


def threshold_goods_reward(
    spec: ThresholdGoodsSpec,
    contribution: int,
    counterpart_contributions: Sequence[int],
    *,
    max_prize: int,
) -> float:
    """Return the reward for `contribution` against a distribution of counterpart figures, in [0,1]."""
    return threshold_goods_reward_at_reach(
        spec,
        contribution=contribution,
        reach_probability=threshold_goods_reach_probability(
            spec, contribution, counterpart_contributions
        ),
        max_prize=max_prize,
    )


def threshold_goods_self_reward(
    spec: ThresholdGoodsSpec, contribution: int, *, max_prize: int
) -> float:
    """Return the reward for `contribution` against counterparts that write exactly the same figure.

    A step function with a peak at the equal share: below it the undertaking does not go ahead and the
    figure is simply the stock minus what was thrown away, at it the prize arrives for the smallest
    outlay that could buy it, and above it the prize is unchanged while every further unit is spent.
    So the optimum is interior and is also the point where the parties split the cost evenly -- the one
    reading on this slate that separates cooperating from maximising.
    """
    return threshold_goods_reward(spec, contribution, [contribution], max_prize=max_prize)


def threshold_goods_self_optimum(spec: ThresholdGoodsSpec, *, max_prize: int) -> int:
    """Return the figure maximising the self-graded reward, by search over the whole grid.

    Searched rather than returned as `equal_share`, following `nash_demand_self_optimum`: the number a
    prediction is written against has to come out of the same arithmetic the reward uses, or an
    off-by-one in `threshold_goods_reached` moves the optimum by one unit and no curve would reveal it.
    """
    return max(
        spec.contributions,
        key=lambda contribution: threshold_goods_self_reward(
            spec, contribution, max_prize=max_prize
        ),
    )


def threshold_goods_self_reward_span(spec: ThresholdGoodsSpec, *, max_prize: int) -> float:
    """Return the widest self-graded reward gap two figures on this game could produce.

    The registry's dead-arm check for this game: under a self-consistent grading the group's own
    distribution cannot create a reward gap, so this span is the entire within-group signal an arm
    could ever have, and a zero here is an arm that trains nothing while writing a full set of
    plausible artifacts. Computed over the grid rather than asserted, so a reward function edited into
    a constant is caught by the check rather than by a flat curve after a paid run.
    """
    rewards = [
        threshold_goods_self_reward(spec, contribution, max_prize=max_prize)
        for contribution in spec.contributions
    ]
    return max(rewards) - min(rewards)


def threshold_goods_best_response(
    spec: ThresholdGoodsSpec, counterpart_contributions: Sequence[int], *, max_prize: int
) -> int:
    """Return the figure maximising the group-mix reward against this counterpart distribution.

    Not one number but a shape, and the shape is the arm: against a group already sitting on the equal
    share the equal share is itself the best response, so the fair split is self-consistent. Against a
    group that mostly puts in nothing the two prize variants part company -- under the low prize the
    best response is nothing at all, and under the high prize it is to fund the undertaking
    single-handed. That is the discrimination the prize knob buys, and it is why the two variants run
    as separate arms.
    """
    return max(
        spec.contributions,
        key=lambda contribution: threshold_goods_reward(
            spec, contribution, counterpart_contributions, max_prize=max_prize
        ),
    )


def threshold_goods_group_reward_span(
    spec: ThresholdGoodsSpec, counterpart_contributions: Sequence[int], *, max_prize: int
) -> float:
    """Return the within-group reward spread a group holding exactly these figures would see.

    Over the group's own figures rather than over the whole answer grid, matching
    `nash_demand_group_reward_span`: under `scale_rewards="none"` this spread IS the advantage magnitude,
    and what a policy could have reached had it answered differently is a separate question from what
    this batch can learn from. It depends on the group's own distribution rather than on the row, which
    is why the registry cannot price this grading and the baseline sweep is what does.

    **A group that is genuinely mixed can still have a spread of exactly zero, and selection cannot see
    it.** Computed 2026-08-21: at the registered high prize, a group split evenly between nothing and
    the equal share sees the same reward for both answers, because the prize collected by writing the
    share (`prize` times the chance the other parties both wrote it too) exactly cancels the share's
    cost. That is even-hunt's dead cell again, and it is invisible to `judge_prompt`, which keys on the
    spread of the FIGURES rather than of the reward -- such a group passes the spread floor comfortably.
    Which is why `threshold_met_rate` is a required metric for this grading and `mean_group_reward_span`
    is logged every step: the two of them together are what would catch it in a live run.
    """
    if not counterpart_contributions:
        raise ValueError("a reward spread needs at least one contribution.")
    rewards = [
        threshold_goods_reward(spec, contribution, counterpart_contributions, max_prize=max_prize)
        for contribution in counterpart_contributions
    ]
    return max(rewards) - min(rewards)


def threshold_goods_over_contribution(spec: ThresholdGoodsSpec, contribution: int) -> bool:
    """Say whether a figure is above the equal share, which is waste rather than virtue here.

    The reading this game exists for. Every other game on the slate makes more cooperation weakly
    better, so a rising rate cannot be told from a rising disposition; here a figure above the share
    buys no more prize than the share does and costs a unit for each one, so a rate that rises while
    the modal figure sits at the share is a maximising or compliant policy rather than a cooperative
    one.
    """
    spec.require_contribution(contribution)
    return contribution > spec.equal_share


def threshold_goods_reference_contributions(spec: ThresholdGoodsSpec) -> list[int]:
    """Return the reference contribution distribution for this endowment, as whole figures.

    Rounded from `THRESHOLD_GOODS_SPREAD_REFERENCE_CONTRIBUTION_FRACTIONS`, so every endowment is
    reported at the same distribution in fraction terms and the spreads are comparable across variants.
    """
    return [
        round(fraction * spec.endowment)
        for fraction in THRESHOLD_GOODS_SPREAD_REFERENCE_CONTRIBUTION_FRACTIONS
    ]


# The minimum-effort game. Everyone writes down one level; what the team achieves is set by the
# LOWEST level anyone wrote, and a higher level of one's own is charged to the writer. Every level
# everybody shares is self-consistent and the highest of those is the best of them, so the question
# is which self-consistent level the team settles on rather than whether it can find one.
MIN_EFFORT_LEVELS = 5
MIN_EFFORT_BENEFIT_PER_LEVEL = 1.0

# Below two levels there is one answer, so no group can disagree and no advantage can exist.
MIN_LEVELS_FOR_A_CHOICE = 2

# The 2x2 knob grid, as (cost per level, counterparts). Both knobs are prose: the cost shows up in
# the printed figures and the count in the counterpart clause, so a variant contrast is a contrast in
# two numbers with the frames held fixed. All four are built into the corpus and only the two widest
# are registered as arms, following the stag ladder -- see `games.arms` for the spreads and the
# reason the narrow pair is left unregistered.
MIN_EFFORT_VARIANTS: dict[str, tuple[float, int]] = {
    "cheap-effort-pair": (0.1, 1),
    "cheap-effort-crew": (0.1, 3),
    "costly-effort-pair": (0.5, 1),
    "costly-effort-crew": (0.5, 3),
}
DEFAULT_MIN_EFFORT_VARIANT = "cheap-effort-pair"

# The repeated form's length, and the level its announced counterpart opens at ("starting at the
# lowest"). Five rounds is `games.prompts.ITERATED_N_ROUNDS`, restated rather than imported because
# that module imports this one; `TestTheRepeatedFormsShareOneLength` pins them equal, so the two
# repeated arms stay comparable round by round -- which is the whole point of the pair.
MIN_EFFORT_MATCH_ROUNDS = 5
MIN_EFFORT_MATCHER_OPENING_LEVEL = 1

# Best-response search over a level grid is `n_levels ** n_rounds` sequences: 3,125 at the registered
# 5 levels and 5 rounds, and 390,625 at 8 rounds. Brute force rather than a dynamic program for
# `max_iterated_return`'s reason -- it cannot silently disagree with the simulator that produces the
# reward -- so the cap is what stops a later rounds bump from turning a cached lookup into a hang.
MAX_MIN_EFFORT_BRUTE_FORCE_ROUNDS = 6

# The affine map into [0,1] is exact arithmetic on floats, so a boundary cell can land a few ulps
# outside. Tolerance rather than a clamp: clamping would hide a reward that had genuinely escaped the
# range, which is the one thing the check exists to catch.
MIN_EFFORT_REWARD_TOLERANCE = 1e-9


@dataclass(frozen=True)
class MinEffortSpec:
    """A minimum-effort game: levels 1..`n_levels`, the lowest one counting, own level charged.

    Deliberately not a `MatrixGameSpec`: the answer is a number rather than one of two labels, the
    payoff is `benefit_per_level * min(own, others) - cost_per_level * own` rather than four cells,
    and `team_size` says how many counterparts the grading draws. The identity checks live in
    `assert_min_effort` and run from `__post_init__`, so a spec that would train nothing cannot be
    built at all -- including one rebuilt from a corpus row, which is where a corrupt corpus would
    otherwise reach the reward function.

    `team_size` counts the OTHER parties, so 1 is the smallest live game. It is read by the renderer
    (which prints it in the counterpart clause) and by the grading (which raises it as the exponent
    on the counterpart distribution) off the same column, which is what stops a prompt that says
    three from being graded against two -- a lie no loss curve could show.
    """

    game_id: str
    n_levels: int
    benefit_per_level: float
    cost_per_level: float
    team_size: int

    def __post_init__(self) -> None:
        """Reject a spec that could not be rendered or graded, before anything reads it."""
        if not self.game_id:
            raise ValueError("game_id must be non-empty; it keys dataset rows and report groups.")
        assert_min_effort(self)

    @property
    def levels(self) -> range:
        """Return every level the model may write, which is the whole answer grid."""
        return range(1, self.n_levels + 1)

    @property
    def cost_benefit_ratio(self) -> float:
        """Return `c / b`: what the chance of being matched has to beat for a step to pay."""
        return self.cost_per_level / self.benefit_per_level

    def require_level(self, level: int) -> None:
        """Raise on a level outside the grid, which is a caller bug rather than a bad completion.

        `games.parsing.parse_level` returns None for an out-of-range figure and the reward turns that
        into the parse penalty, so anything reaching here is our own arithmetic.
        """
        if level not in self.levels:
            raise ValueError(
                f"level must lie in [1, {self.n_levels}] for {self.game_id!r}, got {level}."
            )


def assert_min_effort(spec: MinEffortSpec) -> None:
    """Raise unless this is a minimum-effort game whose reward still moves with the level.

    Four properties, each of which fails by producing a full set of plausible artifacts rather than
    an error. **The benefit must exceed the cost**, or a higher shared level pays less than a lower
    one: everyone at level e earns `(b - c) * e`, so at `c >= b` the best self-consistent answer is
    the bottom of the grid rather than the top, and every prediction written against the arm is
    inverted while the run looks healthy. **The cost must be positive**, or working at the top is
    free, the answer is the biggest number, and no coordination question is being asked. **The grid
    needs at least two levels**, or there is one answer and no within-group spread at all. **There
    must be at least one counterpart**, or nobody can hold the minimum down, the lowest level is the
    model's own, and the reward collapses to `(b - c) * own` -- the biggest-number degeneracy again,
    reached from the other direction.
    """
    if not spec.benefit_per_level > spec.cost_per_level > 0:
        raise ValueError(
            f"{spec.game_id!r} needs benefit_per_level > cost_per_level > 0, got "
            f"benefit={spec.benefit_per_level} cost={spec.cost_per_level}. At cost >= benefit a "
            f"higher level everybody shares pays LESS than a lower one, so the best self-consistent "
            f"answer is the bottom of the grid rather than the top and every prediction written "
            f"against this arm is inverted while the run looks healthy. At cost <= 0 working at the "
            f"top is free, so the answer is the biggest number and no coordination question is asked."
        )
    if spec.n_levels < MIN_LEVELS_FOR_A_CHOICE:
        raise ValueError(
            f"{spec.game_id!r} has n_levels={spec.n_levels}, so there is at most one level to "
            f"write: every parsed completion in every group scores identically, every GRPO advantage "
            f"is zero, and the arm trains nothing while writing a full set of plausible artifacts."
        )
    if spec.team_size < 1:
        raise ValueError(
            f"{spec.game_id!r} has team_size={spec.team_size}, so no counterpart can hold the "
            f"minimum down: the lowest level is the model's own, the reward collapses to "
            f"(benefit - cost) * own_level, and the answer is the biggest number whatever the prompt "
            f"says about a team."
        )


def min_effort_raw_payoff(spec: MinEffortSpec, *, own_level: int, lowest_other_level: int) -> float:
    """Return the raw payoff of one deterministic cell: my level against the lowest of theirs.

    `benefit_per_level * min(own, lowest other) - cost_per_level * own`. The whole game in one line,
    and the only place the payoff rule is written down: every reward, spread, optimum and printed
    figure below derives from this rather than restating it.
    """
    spec.require_level(own_level)
    spec.require_level(lowest_other_level)
    return (
        spec.benefit_per_level * min(own_level, lowest_other_level)
        - spec.cost_per_level * own_level
    )


def min_effort_reward_scale(spec: MinEffortSpec) -> float:
    """Return the width of the raw payoff range, which is this game's shared normalising constant.

    `benefit_per_level * (n_levels - 1)`, and the point is that it does not depend on the cost: the
    widest cell is everybody at the top (`(b - c) * L`), the narrowest is me at the top while the
    lowest sits at 1 (`b - c * L`), and their difference is `b * (L - 1)` at every cost ratio. So one
    constant serves every payoff variant, which is what `_normalised_spec`'s per-game rule asks for
    and what keeps `games.reward_spread`'s cross-variant comparison meaningful -- a per-variant scale
    would rescale the two registered arms' spreads independently and destroy exactly that comparison.
    """
    return spec.benefit_per_level * (spec.n_levels - 1)


def min_effort_reward_floor(spec: MinEffortSpec) -> float:
    """Return the lowest raw payoff any cell reaches: work at the top while the lowest sits at 1.

    The affine shift, and the one deliberate departure from `_normalised_spec`, whose docstring
    argues against a shift because it does not preserve payoff ratios. That argument does not reach
    this game: the raw payoff goes NEGATIVE at a high cost ratio, so no positive scaling can bring it
    into [0,1], and the shift is safe for the two reasons that matter here. A positive affine
    transform preserves preference over lotteries, so every best response and every mixed-strategy
    comparison is unchanged; and GRPO's advantage subtracts the group mean, so a constant offset
    cancels exactly. `TestTheAffineMapPreservesEveryComparison` pins the first claim by asserting
    that the ORDERING of reward differences equals the ordering of raw payoff differences.

    Per variant rather than shared, unlike the scale: the lowest reachable payoff does depend on the
    cost ratio, and a shared shift would leave the cheap variant's rewards bunched inside part of the
    range for no gain, since the shift cancels out of every difference anyway.
    """
    return spec.benefit_per_level - spec.cost_per_level * spec.n_levels


def min_effort_reward_from_raw(spec: MinEffortSpec, raw_payoff: float) -> float:
    """Map a raw payoff into [0,1], refusing one that lands outside it.

    The refusal is the check that the scale and the shift still describe the payoff function. A
    reward above 1 or below 0 stops being commensurable with the -1.0 parse penalty, and the way that
    fails is silent: an out-of-range reward trains perfectly happily.
    """
    reward = (raw_payoff - min_effort_reward_floor(spec)) / min_effort_reward_scale(spec)
    if not -MIN_EFFORT_REWARD_TOLERANCE <= reward <= 1.0 + MIN_EFFORT_REWARD_TOLERANCE:
        raise ValueError(
            f"{spec.game_id!r} maps raw payoff {raw_payoff} to {reward}, outside [0, 1]. The parse "
            f"penalty of -1.0 is only strictly below every played action while rewards stay in that "
            f"range, so this would make an unparseable completion commensurable with a played one. "
            f"scale={min_effort_reward_scale(spec)} floor={min_effort_reward_floor(spec)}."
        )
    return reward


def min_effort_cell_reward(
    spec: MinEffortSpec, *, own_level: int, lowest_other_level: int
) -> float:
    """Return one deterministic cell's normalised reward, which is the figure the prompt prints."""
    return min_effort_reward_from_raw(
        spec,
        min_effort_raw_payoff(spec, own_level=own_level, lowest_other_level=lowest_other_level),
    )


def min_effort_at_or_above_rate(levels: Sequence[int], threshold: int) -> float:
    """Return the share of `levels` at or above `threshold`, i.e. `1 - F(threshold - 1)`."""
    if not levels:
        raise ValueError(
            "a counterpart distribution needs at least one level; with none there is nothing to be "
            "graded against and a default would invent a counterpart."
        )
    return sum(1 for level in levels if level >= threshold) / len(levels)


def min_effort_expected_lowest(
    spec: MinEffortSpec, *, own_level: int, counterpart_levels: Sequence[int]
) -> float:
    """Return `E[min(own_level, the lowest of team_size independent draws)]`, in closed form.

    Exact, never sampled. A non-negative integer bounded by `own_level` equals the number of
    thresholds it clears, so

        E[min(e, M)] = sum_{k=1..e} P(M >= k) = sum_{k=1..e} (1 - F(k-1)) ** n

    where `F` is the CDF of one counterpart's level over `counterpart_levels` and `n` is `team_size`.
    Drawing the counterparts independently from the group's own realised distribution is the natural
    generalisation of the existing group-mix rule, whose binary version is this with one counterpart
    and a two-point distribution.

    Team size enters as an EXPONENT, which is the whole reason a bigger team coordinates worse: the
    chance that every counterpart clears a threshold falls geometrically in the count. That is the
    known human result, and here it falls out of the algebra rather than being imported.
    """
    spec.require_level(own_level)
    for level in counterpart_levels:
        spec.require_level(level)
    return sum(
        min_effort_at_or_above_rate(counterpart_levels, threshold) ** spec.team_size
        for threshold in range(1, own_level + 1)
    )


def min_effort_group_reward(
    spec: MinEffortSpec, *, own_level: int, counterpart_levels: Sequence[int]
) -> float:
    """Return the normalised expected reward for `own_level` against a counterpart distribution.

    The raw payoff is affine in the minimum, so the expectation passes straight through it: the
    expected raw payoff is `benefit * E[min] - cost * own`, with no second expectation to take. That
    is why this game needs no sampling anywhere.
    """
    expected_lowest = min_effort_expected_lowest(
        spec, own_level=own_level, counterpart_levels=counterpart_levels
    )
    raw = spec.benefit_per_level * expected_lowest - spec.cost_per_level * own_level
    return min_effort_reward_from_raw(spec, raw)


def min_effort_raise_probability(
    spec: MinEffortSpec, *, level: int, counterpart_levels: Sequence[int]
) -> float:
    """Return the chance that EVERY counterpart sits above `level`, which is what a step buys.

    `(1 - F(level)) ** team_size`. The analytic half of `min_effort_pressure_gaps`: stepping from
    `level` to `level + 1` pays exactly when this beats the cost-benefit ratio, because the step buys
    one whole benefit unit with that probability and costs the ratio for certain. Kept as its own
    function so a test can check the sign of the measured gap against it -- two independent
    statements of one quantity, which is what catches an algebra slip in either.
    """
    spec.require_level(level)
    return min_effort_at_or_above_rate(counterpart_levels, level + 1) ** spec.team_size


def min_effort_pressure_gaps(
    spec: MinEffortSpec, counterpart_levels: Sequence[int]
) -> dict[int, float]:
    """Return the reward gained by stepping from each level to the next, against this distribution.

    Keyed by the level being left, so the top level has no entry. This is the arm's prediction in one
    object: where the gaps turn from positive to negative is where training pushes the group, and
    `min_effort_best_response` is that reading reduced to a single number.
    """
    return {
        level: (
            min_effort_group_reward(
                spec, own_level=level + 1, counterpart_levels=counterpart_levels
            )
            - min_effort_group_reward(spec, own_level=level, counterpart_levels=counterpart_levels)
        )
        for level in range(1, spec.n_levels)
    }


def min_effort_best_response(spec: MinEffortSpec, counterpart_levels: Sequence[int]) -> int:
    """Return the level maximising the group-mix reward against this counterpart distribution.

    Searched over the grid rather than solved from the pressure inequality, following
    `nash_demand_self_optimum`: the number a prediction is scored against has to come out of the same
    arithmetic the reward uses, or a slip in the expectation moves the target by a level and no curve
    would ever reveal it. Ties take the LOWEST level, so a flat surface reads as the bottom of its
    plateau rather than wherever `max` happened to land -- which matters, because a flat surface is
    exactly what the two unregistered narrow variants have.
    """
    return max(
        reversed(spec.levels),
        key=lambda level: min_effort_group_reward(
            spec, own_level=level, counterpart_levels=counterpart_levels
        ),
    )


def min_effort_group_reward_span(spec: MinEffortSpec, levels: Sequence[int]) -> float:
    """Return the within-group reward spread a group holding exactly these levels would see.

    Under `scale_rewards="none"` this spread IS the advantage magnitude, so it decides whether two
    variants may share a batch and whether an arm carries any strategy signal at all beside its
    format signal. It is a property of the GROUP rather than of the row, and this game has a live
    degenerate case: at a cost ratio of 0.5 with one counterpart, a group split evenly between the
    two extremes scores every level identically and the span is exactly 0.000. That is even-hunt's
    dead cell again, reachable from a corpus rather than from a payoff table, which is why
    `games.min_effort_spread` measures it on a sweep's own completions before an arm is paid for.
    """
    if not levels:
        raise ValueError("a reward spread needs at least one level.")
    rewards = [
        min_effort_group_reward(spec, own_level=level, counterpart_levels=levels)
        for level in levels
    ]
    return max(rewards) - min(rewards)


def min_effort_reference_levels(spec: MinEffortSpec) -> list[int]:
    """Return the reference counterpart distribution the launch-time spread table is priced at.

    One of each level: the flattest distribution on the grid, so every variant is priced at the same
    shape and the four spreads are comparable. A baseline sweep's realised distribution is what
    actually matters and it cannot be known before the sweep, which is what `games.min_effort_spread`
    exists for; this is the number that can be stated in advance.
    """
    return list(spec.levels)


def min_effort_matcher_levels(spec: MinEffortSpec, my_levels: Sequence[int]) -> list[int]:
    """Return the announced counterpart's level per round: the lowest first, then my previous one.

    The whole rule, stated in the prompt and simulated here from the same description. Each of its
    moves depends only on my levels strictly before that round, so handing it the full sequence leaks
    nothing: against a deterministic counterpart an open-loop plan is also the optimal closed-loop
    one, which is what lets a single completion play a whole match (`opponent_moves`'s reasoning, on
    a level grid instead of two actions).
    """
    for level in my_levels:
        spec.require_level(level)
    return [
        MIN_EFFORT_MATCHER_OPENING_LEVEL if index == 0 else my_levels[index - 1]
        for index in range(len(my_levels))
    ]


def simulate_min_effort_match(spec: MinEffortSpec, my_levels: Sequence[int]) -> float:
    """Return my exact total RAW payoff over a match against the level-matcher.

    Raw rather than normalised because the design's own arithmetic is stated in raw units -- at
    `c/b = 0.5` the all-top sequence totals 8.50 and dropping the final round to 1 costs 2.00 of it
    -- and those are the figures the tests check. The reward path uses `min_effort_match_reward`.
    """
    theirs = min_effort_matcher_levels(spec, my_levels)
    return sum(
        min_effort_raw_payoff(spec, own_level=mine, lowest_other_level=other)
        for mine, other in zip(my_levels, theirs, strict=True)
    )


def min_effort_match_reward_total(spec: MinEffortSpec, my_levels: Sequence[int]) -> float:
    """Return the sum of the per-round normalised rewards, which lands in [0, n_rounds].

    Mapping each round through the affine transform before summing, rather than mapping the total, is
    what keeps the match reward inside [0,1] after the division by the optimum. Summing raw payoffs
    first would not: a sequence alternating between the extremes goes genuinely negative at a high
    cost ratio, and a negative numerator over a positive optimum is a reward BELOW the parse penalty,
    i.e. an unparseable completion scoring better than a played match.
    """
    theirs = min_effort_matcher_levels(spec, my_levels)
    return sum(
        min_effort_cell_reward(spec, own_level=mine, lowest_other_level=other)
        for mine, other in zip(my_levels, theirs, strict=True)
    )


@cache
def max_min_effort_match_return(spec: MinEffortSpec, n_rounds: int) -> float:
    """Return the best mapped total over `n_rounds`, by brute force over every level sequence.

    Brute force rather than a dynamic program, for `max_iterated_return`'s reason: it cannot silently
    disagree with `min_effort_match_reward_total`, which is the number the reward actually uses. The
    affine map is a positive transform of the raw total, so the maximising SEQUENCE is the same
    either way, and `TestTheMatchOptimumIsTheTopOfTheGrid` pins that it is the all-top sequence at
    both registered cost ratios. That is the diagnostic this arm exists for: dropping the final level
    only ever loses money here, so a model that drops it anyway is showing a transferred habit rather
    than correct reasoning.

    Memoised because the search is identical for every row of a given game and would otherwise rerun
    for all G completions of every group in every step. `MinEffortSpec` is a frozen dataclass over
    strings, ints and floats, so it hashes by value and two rows describing the same game share one
    cache entry.
    """
    if not 1 <= n_rounds <= MAX_MIN_EFFORT_BRUTE_FORCE_ROUNDS:
        raise ValueError(
            f"n_rounds must be in [1, {MAX_MIN_EFFORT_BRUTE_FORCE_ROUNDS}], got {n_rounds}. The "
            f"search is n_levels ** n_rounds sequences, so the cap is what stops a rounds bump from "
            f"turning this into a hang; raise it deliberately and measure."
        )
    return max(
        min_effort_match_reward_total(spec, list(sequence))
        for sequence in itertools.product(spec.levels, repeat=n_rounds)
    )


@cache
def worst_min_effort_match_return(spec: MinEffortSpec, n_rounds: int) -> float:
    """Return the WORST mapped total over `n_rounds`, the mirror of `max_min_effort_match_return`.

    What the row-relative parse price needs from this grading: the reward is a match total over its
    brute-forced optimum, so its reachable range runs from this figure over the optimum up to exactly
    1, and one such range below the worst of them is what an unparseable completion earns. Brute
    force and memoisation for the maximum's reasons -- it cannot silently disagree with the simulator
    that produces the reward, and the search is identical for every row of a given game.
    """
    if not 1 <= n_rounds <= MAX_MIN_EFFORT_BRUTE_FORCE_ROUNDS:
        raise ValueError(
            f"n_rounds must be in [1, {MAX_MIN_EFFORT_BRUTE_FORCE_ROUNDS}], got {n_rounds}."
        )
    return min(
        min_effort_match_reward_total(spec, list(sequence))
        for sequence in itertools.product(spec.levels, repeat=n_rounds)
    )


def min_effort_match_reward(spec: MinEffortSpec, my_levels: Sequence[int]) -> float:
    """Return the match's mapped total over the brute-forced optimum, in [0,1].

    Normalising by the optimum puts this on the same scale as every one-shot arm, so the parse
    penalty stays commensurable across the slate -- the construction `_score_iterated_group` uses.
    """
    if not my_levels:
        raise ValueError("a match needs at least one round to be scored.")
    best = max_min_effort_match_return(spec, len(my_levels))
    return min_effort_match_reward_total(spec, my_levels) / best


class OpponentRule(StrEnum):
    """Deterministic conditional opponents for the iterated arms.

    A `StrEnum` because the rule round-trips through the dataset's `opponent_rule` string
    column. The values name the mechanism; prompts describe it in neutral prose instead
    (`games/prompts.py`), since "cooperate" is banned vocabulary in anything the model reads.
    """

    TIT_FOR_TAT = "tit-for-tat"
    GRIM_TRIGGER = "grim-trigger"
    ALWAYS_C = "always-cooperate"


def assert_prisoners_dilemma(spec: MatrixGameSpec) -> None:
    """Raise unless the row player's cells satisfy the strict PD ordering DC > CC > DD > CD."""
    if not spec.payoff_dc > spec.payoff_cc > spec.payoff_dd > spec.payoff_cd:
        raise ValueError(
            f"{spec.game_id!r} is not a strict prisoner's dilemma: needs DC > CC > DD > CD, got "
            f"DC={spec.payoff_dc} CC={spec.payoff_cc} DD={spec.payoff_dd} CD={spec.payoff_cd}."
        )


def assert_dominant_defection(spec: MatrixGameSpec) -> None:
    """Raise unless "D" strictly dominates "C" for the row player.

    Weaker than `assert_prisoners_dilemma`: it drops CC > DD, keeping only the property that
    drives the training gradient (defecting pays more against either opponent action).
    """
    if not (spec.payoff_dc > spec.payoff_cc and spec.payoff_dd > spec.payoff_cd):
        raise ValueError(
            f"{spec.game_id!r} does not make defection strictly dominant: needs DC > CC and "
            f"DD > CD, got DC={spec.payoff_dc} CC={spec.payoff_cc} DD={spec.payoff_dd} "
            f"CD={spec.payoff_cd}."
        )


def assert_constant_sum(spec: MatrixGameSpec) -> None:
    """Raise unless the two players' payoffs sum to the same total in every cell.

    By symmetry the diagonal cells total 2*CC and 2*DD while each off-diagonal cell totals
    CD + DC, so a constant sum means CC == DD == (CD + DC) / 2.
    """
    totals = {
        "cell-cc": 2 * spec.payoff_cc,
        "cell-dd": 2 * spec.payoff_dd,
        "cell-cd-or-dc": spec.payoff_cd + spec.payoff_dc,
    }
    if max(totals.values()) - min(totals.values()) > CONSTANT_SUM_TOLERANCE:
        raise ValueError(f"{spec.game_id!r} is not constant-sum; cell totals: {totals}.")


def assert_stag_hunt(spec: MatrixGameSpec) -> None:
    """Raise unless the cells satisfy the stag-hunt ordering CC > DC >= DD > CD.

    CC > DC is what separates a stag hunt from a PD: mutual cooperation is itself a Nash
    equilibrium, so self-play can genuinely reinforce it.
    """
    if not (spec.payoff_cc > spec.payoff_dc >= spec.payoff_dd > spec.payoff_cd):
        raise ValueError(
            f"{spec.game_id!r} is not a stag hunt: needs CC > DC >= DD > CD, got "
            f"CC={spec.payoff_cc} DC={spec.payoff_dc} DD={spec.payoff_dd} CD={spec.payoff_cd}."
        )


def assert_chicken(spec: MatrixGameSpec) -> None:
    """Raise unless the cells satisfy the chicken ordering DC > CC > CD > DD."""
    if not spec.payoff_dc > spec.payoff_cc > spec.payoff_cd > spec.payoff_dd:
        raise ValueError(
            f"{spec.game_id!r} is not chicken: needs DC > CC > CD > DD, got "
            f"DC={spec.payoff_dc} CC={spec.payoff_cc} CD={spec.payoff_cd} DD={spec.payoff_dd}."
        )


def assert_hi_lo(spec: MatrixGameSpec) -> None:
    """Raise unless the cells are pure coordination with a better meeting point.

    The shape is CC > DD > CD == DC == 0: matching pays, mismatching pays nothing either way,
    and DD > 0 keeps the low meeting point a strict equilibrium in its own right -- with DD == 0
    only one cell on the sheet would matter and the game would stop being a coordination
    problem at all.
    """
    pure_coordination = spec.payoff_cd == 0.0 and spec.payoff_dc == 0.0
    ranked_meeting_points = spec.payoff_cc > spec.payoff_dd > 0.0
    if not (pure_coordination and ranked_meeting_points):
        raise ValueError(
            f"{spec.game_id!r} is not Hi-Lo: needs CC > DD > 0 with CD == DC == 0, got "
            f"CC={spec.payoff_cc} DD={spec.payoff_dd} CD={spec.payoff_cd} DC={spec.payoff_dc}."
        )


def assert_harmony(spec: MatrixGameSpec) -> None:
    """Raise unless "C" strictly dominates AND mutual cooperation is the strictly best cell.

    The two properties are independent: dominance (CC > DC and CD > DD) removes any strategic
    tension from the choice, and CC being the maximum -- which additionally needs CC > CD -- is
    what puts the best outcome on the board at mutual cooperation rather than at exploiting a
    cooperator.
    """
    dominant_cooperation = spec.payoff_cc > spec.payoff_dc and spec.payoff_cd > spec.payoff_dd
    mutual_cooperation_best = spec.payoff_cc > max(spec.payoff_cd, spec.payoff_dc, spec.payoff_dd)
    if not (dominant_cooperation and mutual_cooperation_best):
        raise ValueError(
            f"{spec.game_id!r} is not a harmony game: needs CC > DC, CD > DD, and CC the "
            f"largest cell, got CC={spec.payoff_cc} CD={spec.payoff_cd} DC={spec.payoff_dc} "
            f"DD={spec.payoff_dd}."
        )


def assert_defective_coordination(spec: MatrixGameSpec) -> None:
    """Raise unless the cells are Hi-Lo's mirror: DD > CC > 0 with CD == DC == 0.

    Mutual "D" is the better meeting point, so the canonically cooperative action is simply the
    wrong answer -- the shape that lets a degenerate "always emit the cooperative label" policy
    be told apart from genuine coordination on the transfer set.
    """
    pure_coordination = spec.payoff_cd == 0.0 and spec.payoff_dc == 0.0
    defection_meets_higher = spec.payoff_dd > spec.payoff_cc > 0.0
    if not (pure_coordination and defection_meets_higher):
        raise ValueError(
            f"{spec.game_id!r} is not defective coordination: needs DD > CC > 0 with "
            f"CD == DC == 0, got DD={spec.payoff_dd} CC={spec.payoff_cc} CD={spec.payoff_cd} "
            f"DC={spec.payoff_dc}."
        )


def assert_defective_harmony(spec: MatrixGameSpec) -> None:
    """Raise unless "D" strictly dominates AND mutual defection is the strictly best cell.

    Harmony's mirror, and both halves are load-bearing separately. Dominance (DC > CC and
    DD > CD) removes any strategic tension, so a player reading its own column defects. Mutual
    defection being the maximum -- which additionally needs DD > DC -- is what removes the tension
    from the JOINT reading too: with the best cell on the board at mutual defection, no weighting
    of the counterpart's payoff ranks cooperation first, so a cooperation rate here cannot be a
    welfare calculation of any weight and reads as label following or print position instead.

    A prisoner's dilemma passes the dominance half and fails this one, which is the distinction the
    trap needs: in a dilemma cooperating is what a joint-welfare reward asks for.
    """
    dominant_defection = spec.payoff_dc > spec.payoff_cc and spec.payoff_dd > spec.payoff_cd
    mutual_defection_best = spec.payoff_dd > max(spec.payoff_cc, spec.payoff_cd, spec.payoff_dc)
    if not (dominant_defection and mutual_defection_best):
        raise ValueError(
            f"{spec.game_id!r} is not a defective harmony game: needs DC > CC, DD > CD, and DD the "
            f"largest cell, got CC={spec.payoff_cc} CD={spec.payoff_cd} DC={spec.payoff_dc} "
            f"DD={spec.payoff_dd}."
        )


def _temptation_for(payoff_variant: str) -> float:
    """Look up a named temptation magnitude, rejecting unknown variants loudly."""
    if payoff_variant not in TEMPTATION_BY_VARIANT:
        raise ValueError(
            f"Unknown payoff_variant {payoff_variant!r}; "
            f"known variants: {sorted(TEMPTATION_BY_VARIANT)}."
        )
    return TEMPTATION_BY_VARIANT[payoff_variant]


def _normalised_spec(
    game_id: str,
    *,
    payoff_cc: float,
    payoff_cd: float,
    payoff_dc: float,
    payoff_dd: float,
) -> MatrixGameSpec:
    """Scale raw classic cells into [0,1] by dividing by the largest cell.

    Dividing by the maximum (rather than min-max rescaling) is a positive scaling, so it
    preserves every payoff ratio and hence the game's whole strategic structure; a shift would
    not. It requires non-negative raw cells, which every constructor here satisfies.
    """
    cells = (payoff_cc, payoff_cd, payoff_dc, payoff_dd)
    if min(cells) < 0:
        raise ValueError(f"Raw cells for {game_id!r} must be non-negative, got {cells}.")
    scale = max(cells)
    if scale <= 0:
        raise ValueError(f"Raw cells for {game_id!r} are all zero, got {cells}.")
    return MatrixGameSpec(
        game_id=game_id,
        payoff_cc=payoff_cc / scale,
        payoff_cd=payoff_cd / scale,
        payoff_dc=payoff_dc / scale,
        payoff_dd=payoff_dd / scale,
    )


def twin_pd(payoff_variant: str = "temptation-2") -> MatrixGameSpec:
    """Build the positive-sum prisoner's dilemma used by the twin-PD arms.

    temptation-2 is the textbook game (CC=3, CD=0, DC=5, DD=1). Note that temptation-10 breaks
    2*CC > DC + CD, so alternating exploitation beats sustained cooperation against a
    conditional opponent: iterated arms belong on temptation-2.
    """
    temptation = _temptation_for(payoff_variant)
    spec = _normalised_spec(
        f"twin-pd-{payoff_variant}",
        payoff_cc=CLASSIC_PD_CC,
        payoff_cd=CLASSIC_PD_CD,
        payoff_dc=CLASSIC_PD_CC + temptation,
        payoff_dd=CLASSIC_PD_DD,
    )
    assert_prisoners_dilemma(spec)
    return spec


def fixed_pie_pd(payoff_variant: str = "temptation-2") -> MatrixGameSpec:
    """Build the constant-sum counterpart of `twin_pd`: same gradient SIGN, zero-sum prose.

    The arm exists to test the gradient-equivalence claim -- under group-mix grading the
    training signal depends only on defection being dominant, so this game should train the
    same actions as `twin_pd` while the *prompt* describes a fixed pie being divided rather
    than value being created.

    The match is in the sign at every mix, not in the magnitude, and the two differ in how the
    signal moves as training progresses: at temptation-2 the D-minus-C gap is a flat 0.4 here
    against 0.2 + 0.2p for `twin_pd`, equal only at p=1. Since twin-pd-group is expected to drive
    the group's cooperation rate toward 0, its within-group signal halves over a run while this
    one's does not, so the two arms have matched directions and different dynamics. Read the
    launch banner's spread table rather than assuming the contrast is tighter than that.

    Constant sum and CC > DD are mutually exclusive in a symmetric game (see
    `assert_constant_sum`: constant sum forces CC == DD), so this constructor keeps exact
    constant sum plus strict dominance of defection and drops CC > DD. Cells are
    CC = DD = t + 1, DC = CC + t, CD = CC - t, which reproduces the classic CC=3 at
    temptation-2 and keeps CD strictly positive at every magnitude.
    """
    temptation = _temptation_for(payoff_variant)
    mutual = temptation + 1.0
    spec = _normalised_spec(
        f"fixed-pie-pd-{payoff_variant}",
        payoff_cc=mutual,
        payoff_cd=mutual - temptation,
        payoff_dc=mutual + temptation,
        payoff_dd=mutual,
    )
    assert_constant_sum(spec)
    assert_dominant_defection(spec)
    return spec


def stag_hunt(payoff_variant: str = DEFAULT_STAG_HUNT_VARIANT) -> MatrixGameSpec:
    """Build one rung of the stag-hunt ladder; mutual cooperation is itself a Nash equilibrium.

    The variants are a dose-response ladder over the risk-dominance boundary
    (`stag_hunt_cooperation_threshold`): the opponent-cooperation probability a player must
    credit before hunting beats the safe option. "favoured-hunt" sits at 1/3 (hunting is itself
    risk-dominant), "even-hunt" at exactly 1/2 (neither action risk-dominates), "safe-hunt" at
    3/4 and "risky-hunt" at 0.95 (the safe option is nearly as good as the best outcome). Under
    group-mix grading the boundary is where the training direction flips, so the ladder asks
    *where* trained cooperation gives out rather than whether it survives one hard setting --
    and the easy rungs are the ones where cooperation can be reinforced from a realistic
    baseline mix.

    Naming caveat: some conventions reserve "stag hunt" for games whose safe equilibrium is
    risk-dominant (boundary above 1/2) and would call favoured-hunt and even-hunt "assurance
    games". The label is contested; the cells, which pass `assert_stag_hunt` unchanged, are
    not. Do not "fix" the easy rungs back into the hard regime.
    """
    if payoff_variant not in STAG_HUNT_VARIANTS:
        raise ValueError(
            f"Unknown stag-hunt payoff_variant {payoff_variant!r}; "
            f"known variants: {sorted(STAG_HUNT_VARIANTS)}."
        )
    payoff_cc, payoff_cd, payoff_dc, payoff_dd = STAG_HUNT_VARIANTS[payoff_variant]
    spec = _normalised_spec(
        f"stag-hunt-{payoff_variant}",
        payoff_cc=payoff_cc,
        payoff_cd=payoff_cd,
        payoff_dc=payoff_dc,
        payoff_dd=payoff_dd,
    )
    assert_stag_hunt(spec)
    return spec


def stag_hunt_cooperation_threshold(spec: MatrixGameSpec) -> float:
    """Return the opponent-cooperation probability above which hunting beats the safe option.

    The risk-dominance boundary. Against an opponent who hunts with probability p, "C" pays
    p*CC + (1-p)*CD and "D" pays p*DC + (1-p)*DD; setting them equal gives
    p* = (DD - CD) / ((CC - DC) + (DD - CD)). Both differences are strictly positive for any
    spec passing `assert_stag_hunt`, so the boundary always lands inside (0, 1). The earlier
    DD/CC shortcut is this formula's special case at DC == DD (true of safe-hunt and
    risky-hunt, which is why it looked correct) and is wrong for every ladder rung with
    DC > DD. Reported rather than hard-coded so the contrast between variants stays derived
    from their cells.
    """
    assert_stag_hunt(spec)
    hunting_gain = spec.payoff_cc - spec.payoff_dc
    hedging_gain = spec.payoff_dd - spec.payoff_cd
    return hedging_gain / (hunting_gain + hedging_gain)


# The opponent mixes the reward-spread report samples: near each corner, plus the middle.
GROUP_MIX_SPREAD_MIXES: tuple[float, float, float] = (0.1, 0.5, 0.9)

# Below two completions a group has no mix to be graded against, and no advantage baseline either.
MIN_GROUP_FOR_A_MIX = 2


def group_mix_reward_spread(spec: MatrixGameSpec, opponent_coop_prob: float) -> float:
    """Return the within-group reward spread |E[r|C] - E[r|D]| at one opponent mix.

    Under group-mix grading every parsed completion in a group is scored against the same
    opponent distribution, so a group carries at most two distinct rewards and this gap is the
    whole GRPO training signal. With `scale_rewards="batch"` the advantage denominator is a
    batch-level scalar, so a game whose spread is compressed (risky-hunt: 0.05 at p=0.9) trains
    roughly 8x weaker than a wide one sharing its batch (twin-pd: 0.38 at the same mix). This
    number is what an informed decision about mixing payoff variants in one run reads.
    """
    if not 0.0 <= opponent_coop_prob <= 1.0:
        raise ValueError(f"opponent_coop_prob must be in [0, 1], got {opponent_coop_prob}.")
    reward_cooperate = _expected_cell(spec.payoff_cc, spec.payoff_cd, opponent_coop_prob)
    reward_defect = _expected_cell(spec.payoff_dc, spec.payoff_dd, opponent_coop_prob)
    return abs(reward_cooperate - reward_defect)


def _expected_cell(vs_cooperator: float, vs_defector: float, coop_prob: float) -> float:
    """Average one action's two cells under an opponent cooperating with probability p."""
    return coop_prob * vs_cooperator + (1.0 - coop_prob) * vs_defector


def _assert_unit_interval(coop_prob: float) -> None:
    """Reject a cooperation probability outside [0, 1], which no mix can produce."""
    if not 0.0 <= coop_prob <= 1.0:
        raise ValueError(f"opponent coop_prob must be in [0, 1], got {coop_prob}.")


def expected_joint_payoff(spec: MatrixGameSpec, action: str, opponent_coop_prob: float) -> float:
    """Score one action as the MEAN of both sides' payoffs against an opponent mix.

    The wave-3 ladder's pie grading: the recipient is the pair, not the player. The mean rather
    than the sum keeps the reward on the same [0, 1] scale as every other grading, and the argmax
    is identical. Both `games.rewards` and the spread report read this one function, so the
    trained reward and the pre-launch gradient reading cannot drift into two arithmetics.
    """
    _assert_unit_interval(opponent_coop_prob)
    joint_vs_cooperator = (spec.payoff(action, COOPERATE) + spec.payoff(COOPERATE, action)) / 2
    joint_vs_defector = (spec.payoff(action, DEFECT) + spec.payoff(DEFECT, action)) / 2
    return _expected_cell(joint_vs_cooperator, joint_vs_defector, opponent_coop_prob)


def expected_counterpart_payoff(
    spec: MatrixGameSpec, action: str, opponent_coop_prob: float
) -> float:
    """Score one action as the COUNTERPART's expected payoff against an opponent mix.

    The ladder's selfless grading: a counterpart playing b against my action a earns
    payoff(b, a), so my reward is what my choice hands the other side. In any strict PD
    (CC > CD and DC > DD) cooperating is strictly dominant under this rule, at every mix and
    every temptation -- the one grading on the slate where the trained direction is pure
    other-regard rather than coupling or pie arithmetic.
    """
    _assert_unit_interval(opponent_coop_prob)
    return _expected_cell(
        spec.payoff(COOPERATE, action), spec.payoff(DEFECT, action), opponent_coop_prob
    )


def expected_care_payoff(
    spec: MatrixGameSpec, action: str, opponent_coop_prob: float, *, alpha: float
) -> float:
    """Score one action as own expected payoff plus `alpha` times the counterpart's, over `1 + alpha`.

    The care family's matrix arithmetic, `(own + alpha * other) / (1 + alpha)`. `other` is
    `expected_counterpart_payoff` verbatim, and `own` is `_expected_cell` over the action's own two
    cells, which is the arithmetic `games.rewards._expected_payoff` runs for own-payoff group-mix
    (`games.tests.test_games_payoffs` pins the two equal, since the reward module imports this one
    and the dependency cannot go the other way).

    The family therefore interpolates the wave-3 ladder rather than sitting beside it: at alpha 0 it
    is own-payoff group-mix exactly, at alpha 1 it is `expected_joint_payoff` exactly (the mean of
    both sides, since dividing by `1 + alpha` averages rather than sums), and as alpha grows the
    ordering of actions tends to `expected_counterpart_payoff`'s without ever reaching it. Dividing
    rather than summing is what keeps the reward in [0, 1] like every other matrix grading, which the
    parse penalty's meaning depends on.
    """
    assert_care_alpha(alpha)
    _assert_unit_interval(opponent_coop_prob)
    own = _expected_cell(
        spec.payoff(action, COOPERATE), spec.payoff(action, DEFECT), opponent_coop_prob
    )
    other = expected_counterpart_payoff(spec, action, opponent_coop_prob)
    return (own + alpha * other) / (1.0 + alpha)


def care_reward_spread(spec: MatrixGameSpec, opponent_coop_prob: float, *, alpha: float) -> float:
    """Return the within-group reward spread |E[r|C] - E[r|D]| under care grading at one mix."""
    return abs(
        expected_care_payoff(spec, COOPERATE, opponent_coop_prob, alpha=alpha)
        - expected_care_payoff(spec, DEFECT, opponent_coop_prob, alpha=alpha)
    )


def joint_welfare_reward_spread(spec: MatrixGameSpec, opponent_coop_prob: float) -> float:
    """Return the within-group reward spread |E[r|C] - E[r|D]| under joint-welfare grading."""
    return abs(
        expected_joint_payoff(spec, COOPERATE, opponent_coop_prob)
        - expected_joint_payoff(spec, DEFECT, opponent_coop_prob)
    )


def other_payoff_reward_spread(spec: MatrixGameSpec, opponent_coop_prob: float) -> float:
    """Return the within-group reward spread |E[r|C] - E[r|D]| under other-payoff grading."""
    return abs(
        expected_counterpart_payoff(spec, COOPERATE, opponent_coop_prob)
        - expected_counterpart_payoff(spec, DEFECT, opponent_coop_prob)
    )


def joint_welfare_gap_crossing(spec: MatrixGameSpec) -> float | None:
    """Return the mix where the joint-welfare gap changes sign, or None where one sign holds.

    Writing the off-diagonal joint value as m = (CD + DC) / 2, the gap is
    gap(p) = p * (CC - m) + (1 - p) * (m - DD), linear in p, so the crossing is
    p* = (m - DD) / [(m - DD) + (m - CC)]. It sits inside (0, 1) exactly when the pie is bigger
    on the off-diagonal than at mutual cooperation (DC + CD > 2 * CC, the same cell fact that
    breaks the iterated PD at temptation-10) while still beating mutual defection -- and it is
    then STABLE under group-mix dynamics (cooperators win below it, defectors above it), so it is
    a numeric prediction a run's endpoint can be scored against, exactly as chicken's 0.50 is for
    own-payoff grading. None, or a value outside [0, 1], means the gap holds one sign across
    every reachable mix and the arm runs to a corner.
    """
    off_diagonal = (spec.payoff_cd + spec.payoff_dc) / 2
    slope = (spec.payoff_cc - off_diagonal) - (off_diagonal - spec.payoff_dd)
    if slope == 0.0:
        return None
    return (off_diagonal - spec.payoff_dd) / -slope


def group_mix_gap_slope(spec: MatrixGameSpec) -> float:
    """Return how far the group-mix reward gap moves per unit of group cooperation.

    The coefficient on p in `group_mix_gap_crossing`'s algebra, (CC - CD) - (DC - DD), and the sign
    that says what a crossing of it means: negative is a crossing a run settles at (chicken),
    positive is one it is pushed away from (the stag hunt), zero is a gap that does not move with
    the mix at all and therefore crosses nowhere.
    """
    return (spec.payoff_cc - spec.payoff_cd) - (spec.payoff_dc - spec.payoff_dd)


def group_mix_gap_crossing(
    spec: MatrixGameSpec, *, num_generations: int, leave_one_out: bool = False
) -> float | None:
    """Return the group cooperation rate at which the group-mix reward gap changes sign.

    Stable or repelling, inside [0, 1] or not: the raw crossing, because the two callers want
    different halves of it. `group_mix_fixed_point` wants the stable interior case, the only one a
    run's endpoint can be checked against. `games.corpus_partition` wants the repelling case, since a
    repelling crossing is exactly what makes two halves of one corpus train in opposite directions,
    and the rate is then the boundary the mix-split partitions at.

    The algebra, because the answer depends on the grading mode and one pre-registered number was
    quoted without saying so. Write the group's cooperation rate as p and the four cells as
    CC/CD/DC/DD. Without leave-one-out every parsed completion is graded against the group's own
    mix, itself included, so

        gap(p) = E[r|C] - E[r|D] = (CD - DD) + p * [(CC - CD) - (DC - DD)]

    and the crossing is p* = (DD - CD) / [(CC - CD) - (DC - DD)]. Under `leave_one_out` a
    completion is graded against the OTHER G-1 completions, so with k cooperators a cooperator sees
    (k-1)/(G-1) and a defector sees k/(G-1); the crossing moves to

        k* = [(CC - CD) - (CD - DD) * (G - 1)] / [(CC - CD) - (DC - DD)],  p* = k* / G

    which depends on the group size. For chicken at G=8 that is 0.31 rather than 0.50 -- so a
    leave-one-out run settling at the documented 0.50 would be evidence of a problem, and one
    settling at 0.31 would read as the reward-path bug the arm exists to detect. `num_generations`
    is therefore read only under `leave_one_out`, where it changes the answer -- and a corpus split
    at the plain boundary but trained with the flag on would have had its prompts sorted by a number
    the run never used, which is why one function serves both callers.

    None means the gap never crosses, so no rate divides the two training directions. A rate outside
    [0, 1] says the gap keeps one sign across every reachable mix; a caller needing an interior
    boundary has to check for that rather than assume it.
    """
    if num_generations < MIN_GROUP_FOR_A_MIX:
        raise ValueError(
            f"a group needs at least 2 completions for a group mix to exist, {num_generations=}."
        )
    slope = group_mix_gap_slope(spec)
    if slope == 0.0:
        return None
    if leave_one_out:
        numerator = (spec.payoff_cc - spec.payoff_cd) - (spec.payoff_cd - spec.payoff_dd) * (
            num_generations - 1
        )
        return (numerator / slope) / num_generations
    return (spec.payoff_dd - spec.payoff_cd) / slope


def group_mix_fixed_point(
    spec: MatrixGameSpec, *, num_generations: int, leave_one_out: bool = False
) -> float | None:
    """Return the cooperation rate group-mix training on this game settles at, or None.

    None means there is no *stable interior* rate to settle at: either the reward gap never changes
    sign over [0, 1] (a dominance game, which runs to a corner), or the crossing exists but is
    unstable (a mix that repels, so a run leaves it for a corner too). Only a stable interior
    crossing is a prediction a run can be checked against, which is what makes chicken this repo's
    numeric control on the whole reward path.

    The crossing itself, and the grading-mode algebra behind it, live in `group_mix_gap_crossing`.
    """
    rate = group_mix_gap_crossing(
        spec, num_generations=num_generations, leave_one_out=leave_one_out
    )
    if group_mix_gap_slope(spec) >= 0.0:
        return None
    return rate if rate is not None and 0.0 <= rate <= 1.0 else None


# Marks a row whose prompt states no decision-matching track record: every game but the
# stated-match one. The reward function refuses to read this as a probability, exactly as it
# refuses an unsampled `opp_coop_prob` and an unset `stated_return_fraction`.
STATED_MATCH_PROB_UNSET = -1.0


def stated_match_expected_payoff(spec: MatrixGameSpec, action: str, match_prob: float) -> float:
    """Score one action against a counterpart that MATCHES it with probability `match_prob`.

    The stated-track-record grading's whole reward: `p * payoff(a, a) + (1-p) * payoff(a, other)`.
    Note what this is not -- `_expected_cell` against an independent mix. There the opponent's
    action is a coin that ignores mine; here it is a copy of mine with probability p and the
    opposite otherwise, so the optimal action VARIES with p (below `stated_match_crossover`
    defection pays more, above it cooperation does). That dependence is the arm's mechanism: a
    reward that is a fixed function of the model's own action teaches the policy to stop reading
    the counterpart, and this one is deliberately not such a function across the corpus's p
    mixture.
    """
    _assert_unit_interval_match_prob(match_prob)
    other = DEFECT if action == COOPERATE else COOPERATE
    return match_prob * spec.payoff(action, action) + (1.0 - match_prob) * spec.payoff(
        action, other
    )


def stated_match_gap(spec: MatrixGameSpec, match_prob: float) -> float:
    """Return EV(cooperate) - EV(defect) at one stated match probability.

    The signed margin the corpus audit prices cells by, and -- for a group split across the two
    actions -- exactly the within-group reward spread, i.e. the advantage magnitude under
    `scale_rewards="none"` and the numerator under "batch". Written as the difference of the two
    `stated_match_expected_payoff` calls rather than as expanded algebra, so the audit and the
    reward cannot drift into two arithmetics.
    """
    return stated_match_expected_payoff(spec, COOPERATE, match_prob) - stated_match_expected_payoff(
        spec, DEFECT, match_prob
    )


def stated_match_crossover(spec: MatrixGameSpec) -> float | None:
    """Return the match probability at which the two actions' expected payoffs cross, or None.

    The gap is linear in p: gap(p) = p*(CC - DD) + (1-p)*(CD - DC), so the crossing sits at
    p* = (DC - CD) / ((CC - DD) + (DC - CD)). For any strict PD both differences are positive
    (DC > CD is part of DC > CC > DD > CD, and CC > DD directly), so the crossover is interior:
    defection is EV-optimal below it and cooperation above it, which is what lets a p mixture
    straddle it. None means the gap never moves with p (a table whose diagonal gain equals its
    off-diagonal gain), so no mixture over p can make the optimal action vary and the audit must
    refuse the table. Sanity anchor, worked by hand: the classic PD (3/0/5/1) gives
    3p = 5 - 4p => p* = 5/7, which `stated_match_gap` confirms at the normalised cells.
    """
    slope = (spec.payoff_cc - spec.payoff_dd) + (spec.payoff_dc - spec.payoff_cd)
    if slope == 0.0:
        return None
    return (spec.payoff_dc - spec.payoff_cd) / slope


def stated_match_optimal_action(spec: MatrixGameSpec, match_prob: float) -> str | None:
    """Return the EV-optimal action at one stated match probability, or None at an exact tie.

    Read off the sign of `stated_match_gap` -- the same arithmetic the reward pays -- so the
    per-step "did it play the paying side" metric cannot disagree with the reward about which
    side pays. None (a zero gap) is a cell the corpus audit refuses to build, so the reward
    treats it as unanswerable rather than picking a side.
    """
    gap = stated_match_gap(spec, match_prob)
    if gap == 0.0:
        return None
    return COOPERATE if gap > 0.0 else DEFECT


def _assert_unit_interval_match_prob(match_prob: float) -> None:
    """Reject a match probability outside [0, 1], which no stated track record can produce."""
    if not 0.0 <= match_prob <= 1.0:
        raise ValueError(f"match_prob must be in [0, 1], got {match_prob}.")


# The v2 track-record arm's margin target: every training cell's |EV(C) - EV(D)| at its stated
# rate, after `scaled_stated_match_spec` and display quantization. 0.10 rather than something
# larger because the payoff box forbids more: cells live in [0,1] with DC the largest, so
# slope = (DC - CD)/p* <= 1/p*, a table's coop-side gap at rung p is at most (p - p*)/p*, and its
# defect-side gap at rung q at most (p* - q)/p* -- with integer rungs 51-99 a table reaches margin
# m on BOTH sides only if p* is inside [0.51/(1-m), 0.99/(1+m)], a window that at 0.20 excludes
# most of the crossover diversity the grid needs. v1's own record shows this scale trains: its
# temptation-10 coop cells carried margins 0.096/0.142 and still relocated the dose step into the
# correct crossover interval. What broke v1 was the ASYMMETRY (defect-side margins up to 0.538
# against coop-side 0.050-0.386, a ~9x mean penalty gap under batch scaling), which a shared
# constant of any size removes.
TRACK_RECORD_V2_TARGET_MARGIN = 0.10

# The v2 grid's table shapes, keyed by payoff-variant name and parameterised as
# (crossover p*, CD, DD): CC is DERIVED at lookup (CC = DD + (1 - CD)(1 - p*)/p*, with DC = 1),
# so the crossover each name claims is the crossover the cells carry by construction rather than
# by hand arithmetic. Three crossover slots -- at m=0.10 with integer rungs strictly inside
# (50, 100), exactly three margin-intervals fit disjointly, which also makes 0.5 + 0.5/3 the
# floor for any monotone rate-threshold policy -- and three shapes per slot so no single gross
# feature identifies a slot without reading the table (the CC and DD ranges overlap across
# slots; shapes within a slot share the crossover and nothing else). The mid slot deliberately
# contains the classic temptation-2 table (p* = 5/7), registered through `twin_pd` rather than
# restated here, for line-for-line continuity with v1. Names carry the crossover percent and the
# mutual-defection floor in raw points, the two axes a reader needs to place a cell.
_TRACK_RECORD_V2_SHAPES: dict[str, tuple[float, float, float]] = {
    "xover57-floor6": (0.57, 0.0, 0.06),
    "xover57-floor15": (0.57, 0.0, 0.15),
    "xover57-floor24": (0.57, 0.03, 0.24),
    "xover71-floor35": (5.0 / 7.0, 0.0, 0.35),
    "xover71-floor50": (5.0 / 7.0, 0.03, 0.50),
    "xover89-floor15": (0.895, 0.0, 0.15),
    "xover89-floor45": (0.895, 0.0, 0.45),
    "xover89-floor72": (0.895, 0.03, 0.72),
}

# The classic table shared with v1, reachable through the same lookup as the new shapes.
_TRACK_RECORD_V2_TWIN_PD_VARIANT = "temptation-2"

TRACK_RECORD_V2_TABLE_VARIANTS: tuple[str, ...] = (
    "xover57-floor6",
    "xover57-floor15",
    "xover57-floor24",
    _TRACK_RECORD_V2_TWIN_PD_VARIANT,
    "xover71-floor35",
    "xover71-floor50",
    "xover89-floor15",
    "xover89-floor45",
    "xover89-floor72",
)


def track_record_v2_table(payoff_variant: str) -> MatrixGameSpec:
    """Return one v2 table's raw (unscaled) cells, PD-checked.

    `temptation-2` resolves through `twin_pd` so the shared table cannot drift from the one every
    banked v1 artifact used; the rest come from the v2 shape dictionary.
    """
    if payoff_variant == _TRACK_RECORD_V2_TWIN_PD_VARIANT:
        return twin_pd(_TRACK_RECORD_V2_TWIN_PD_VARIANT)
    if payoff_variant not in _TRACK_RECORD_V2_SHAPES:
        raise ValueError(
            f"Unknown track-record-v2 payoff_variant {payoff_variant!r}; known variants: "
            f"{sorted(TRACK_RECORD_V2_TABLE_VARIANTS)}."
        )
    crossover, cd, dd = _TRACK_RECORD_V2_SHAPES[payoff_variant]
    spec = MatrixGameSpec(
        game_id=f"pd-track-record-v2-{payoff_variant}",
        payoff_cc=dd + (1.0 - cd) * (1.0 - crossover) / crossover,
        payoff_cd=cd,
        payoff_dc=1.0,
        payoff_dd=dd,
    )
    assert_prisoners_dilemma(spec)
    return spec


def scaled_stated_match_spec(
    spec: MatrixGameSpec, match_prob: float, *, target_margin: float
) -> MatrixGameSpec:
    """Scale a table uniformly so its |EV gap| at `match_prob` equals `target_margin`.

    Uniform scaling is exact and structure-preserving: the gap is linear in the four cells, so
    k = target / |gap| lands the margin precisely, and every payoff ratio -- hence the crossover
    and the PD ordering -- is untouched. k > 1 is refused rather than clamped, because the payoff
    box caps cells at 1.0 and a clamped table would carry a smaller margin than the corpus audit
    claims for it: the grid must keep rungs far enough from each table's crossover that every
    scaling is downward.
    """
    gap = stated_match_gap(spec, match_prob)
    if abs(gap) < target_margin:
        raise ValueError(
            f"{spec.game_id!r} at match_prob={match_prob} carries |EV gap| {abs(gap):.4f} under "
            f"the {target_margin} target, so hitting the margin needs scale k > 1 and cells "
            f"above the payoff box's 1.0 cap. Move the rung away from this table's crossover "
            f"({stated_match_crossover(spec)})."
        )
    k = target_margin / abs(gap)
    return MatrixGameSpec(
        game_id=spec.game_id,
        payoff_cc=spec.payoff_cc * k,
        payoff_cd=spec.payoff_cd * k,
        payoff_dc=spec.payoff_dc * k,
        payoff_dd=spec.payoff_dd * k,
    )


def chicken() -> MatrixGameSpec:
    """Build chicken, where mutual defection is the worst cell for both players.

    That inverts a PD, in which DD at least beats being exploited, so defection here cannot be
    justified by dominance. As a training arm it is the numeric-prediction control: under
    group-mix grading the reward gap between the actions flips sign at cooperation rate 0.50
    and the crossing is *stable*, so a healthy run settles at an interior mix near 0.50. Either
    corner is evidence of a reward-path bug, not a finding.

    That 0.50 is the plain group-mix number and holds only without `--leave-one-out`, which grades
    each completion against the OTHER G-1 and moves the crossing to about 0.31 at G=8. Ask
    `group_mix_fixed_point` for the rate under a given grading mode rather than quoting this one:
    under the flag, a run landing on 0.50 would look healthy while being wrong.
    """
    spec = _normalised_spec(
        "chicken",
        payoff_cc=CHICKEN_CC,
        payoff_cd=CHICKEN_CD,
        payoff_dc=CHICKEN_DC,
        payoff_dd=CHICKEN_DD,
    )
    assert_chicken(spec)
    return spec


def hi_lo() -> MatrixGameSpec:
    """Build Hi-Lo: pure coordination with one meeting point ten times better than the other.

    There is no conflict anywhere on the sheet -- matching pays both players, mismatching pays
    nothing -- so the only "skill" is picking the obviously better meeting point together. The
    reward gap favours the high point once the opponent mix clears DD / (CC + DD) = 1/11, i.e.
    from nearly any starting mix, which is why this is the positive control on the reward path:
    if group-mix training cannot drive Hi-Lo to the high equilibrium, the plumbing is broken,
    and no other arm would fail loudly in that case.
    """
    spec = _normalised_spec(
        "hi-lo",
        payoff_cc=HI_LO_CC,
        payoff_cd=HI_LO_CD,
        payoff_dc=HI_LO_DC,
        payoff_dd=HI_LO_DD,
    )
    assert_hi_lo(spec)
    return spec


def harmony() -> MatrixGameSpec:
    """Build the harmony game ("prisoner's delight"): cooperating is best no matter what.

    A PD-shaped sheet with the temptation removed: "C" strictly dominates and mutual
    cooperation is the best cell on the board, so maximum reward is simply cooperation.
    Group-mix grading upweights it from any starting mix -- by dominance rather than by
    coordination, which is exactly what makes it the companion placebo to `fixed_pie_pd`
    (whose gradient can only ever point the other way) and the contrast to the stag hunt
    (where cooperating is best only conditionally).
    """
    spec = _normalised_spec(
        "harmony",
        payoff_cc=HARMONY_CC,
        payoff_cd=HARMONY_CD,
        payoff_dc=HARMONY_DC,
        payoff_dd=HARMONY_DD,
    )
    assert_harmony(spec)
    return spec


def defective_coordination() -> MatrixGameSpec:
    """Build defective coordination (eval-only): Hi-Lo whose good meeting point is mutual "D".

    Cooperating is simply the wrong answer here. Without a transfer game shaped like this, a
    policy degenerating to "always emit the cooperative label" would be indistinguishable from
    broad prosocial generalisation -- the negative control the eval-only set was missing.
    """
    spec = _normalised_spec(
        "defective-coordination",
        payoff_cc=DEFECTIVE_COORDINATION_CC,
        payoff_cd=DEFECTIVE_COORDINATION_CD,
        payoff_dc=DEFECTIVE_COORDINATION_DC,
        payoff_dd=DEFECTIVE_COORDINATION_DD,
    )
    assert_defective_coordination(spec)
    return spec


def defective_harmony() -> MatrixGameSpec:
    """Build defective harmony (eval-only): harmony whose dominant action is the uncooperative one.

    The wave-4b trap cell. `defective_coordination` already catches a policy that emits the
    cooperative label whatever the sheet says, but it catches it through coordination: mutual "D"
    is the good meeting point there and mutual "C" is a strict equilibrium too, so a model that
    cooperated could be coordinating on the worse of two meeting points. Here there is no
    coordination story available. "D" pays more than "C" against either counterpart action, and
    mutual "D" is the best cell for both sides, so every reading of the sheet -- own payoff, the
    counterpart's, any weighted sum of the two -- ranks defection first.
    """
    spec = _normalised_spec(
        "defective-harmony",
        payoff_cc=DEFECTIVE_HARMONY_CC,
        payoff_cd=DEFECTIVE_HARMONY_CD,
        payoff_dc=DEFECTIVE_HARMONY_DC,
        payoff_dd=DEFECTIVE_HARMONY_DD,
    )
    assert_defective_harmony(spec)
    return spec


def public_goods() -> MatrixGameSpec:
    """Build the two-player linear public-goods game (eval-only) as a 2x2 matrix.

    Contributing is "C". With a unit endowment each, a pooled contribution multiplied by
    `PUBLIC_GOODS_MULTIPLIER` and split evenly, a contributor gets back only half the
    multiplier while paying the same amount to the other player, so the reduced matrix is an
    ordinary PD -- which is the point: it tests whether trained defection transfers across
    surface framing rather than across strategic structure.
    """
    share = PUBLIC_GOODS_MULTIPLIER * PUBLIC_GOODS_ENDOWMENT / 2
    kept = PUBLIC_GOODS_ENDOWMENT
    spec = _normalised_spec(
        "public-goods",
        payoff_cc=2 * share,
        payoff_cd=share,
        payoff_dc=kept + share,
        payoff_dd=kept,
    )
    assert_prisoners_dilemma(spec)
    return spec


def ultimatum_responder() -> MatrixGameSpec:
    """Build the responder's half of an ultimatum game (eval-only) as a degenerate matrix.

    Accepting is "C". The proposer has already moved -- their split is stated in the prompt --
    so the responder faces no live opponent and the payoff does not depend on the opponent
    axis: both accept cells pay `ULTIMATUM_RESPONDER_SHARE`, both reject cells pay nothing.
    Reusing `MatrixGameSpec` this way keeps one eval path for every game, at the cost of two
    redundant columns.

    The cells are the offered fraction itself, already in [0,1], and deliberately not passed
    through `_normalised_spec`: scaling by the largest cell would map "accept" to 1.0 and erase
    exactly the information the item is probing, namely how unfair the offer is.
    """
    return MatrixGameSpec(
        game_id="ultimatum-responder",
        payoff_cc=ULTIMATUM_RESPONDER_SHARE,
        payoff_cd=ULTIMATUM_RESPONDER_SHARE,
        payoff_dc=0.0,
        payoff_dd=0.0,
    )


def opponent_moves(rule: OpponentRule, my_moves: Sequence[str]) -> list[str]:
    """Return the opponent's move per round under `rule`, given my whole move sequence.

    Each opponent move depends only on my moves strictly before that round, so passing the
    full sequence leaks nothing: against a deterministic opponent an open-loop plan is also
    the optimal closed-loop one, which is what lets a single completion play a whole match.
    """
    unknown = sorted({move for move in my_moves if move not in ACTIONS})
    if unknown:
        raise ValueError(f"my_moves must contain only {ACTIONS}, got unknown moves {unknown}.")
    moves: list[str] = []
    for round_index in range(len(my_moves)):
        history = my_moves[:round_index]
        if not history or rule is OpponentRule.ALWAYS_C:
            moves.append(COOPERATE)
        elif rule is OpponentRule.TIT_FOR_TAT:
            moves.append(history[-1])
        elif rule is OpponentRule.GRIM_TRIGGER:
            moves.append(DEFECT if DEFECT in history else COOPERATE)
        else:
            raise ValueError(f"Unhandled opponent rule {rule!r}.")
    return moves


def simulate_iterated(spec: MatrixGameSpec, rule: OpponentRule, my_moves: Sequence[str]) -> float:
    """Return my exact total (undiscounted) return from playing `my_moves` against `rule`."""
    theirs = opponent_moves(rule, my_moves)
    return sum(spec.payoff(mine, opp) for mine, opp in zip(my_moves, theirs, strict=True))


def max_iterated_return(spec: MatrixGameSpec, rule: OpponentRule, n_rounds: int) -> float:
    """Return the best achievable total over `n_rounds`, by brute force over all sequences.

    Used to normalise the iterated-arm reward into [0,1]. Brute force rather than dynamic
    programming: 2**n is trivial at these sizes and cannot silently disagree with
    `simulate_iterated`, which is the number the reward actually uses.
    """
    if not 1 <= n_rounds <= MAX_BRUTE_FORCE_ROUNDS:
        raise ValueError(f"n_rounds must be in [1, {MAX_BRUTE_FORCE_ROUNDS}], got {n_rounds}.")
    return max(
        simulate_iterated(spec, rule, list(sequence))
        for sequence in itertools.product(ACTIONS, repeat=n_rounds)
    )


def worst_iterated_return(spec: MatrixGameSpec, rule: OpponentRule, n_rounds: int) -> float:
    """Return the WORST achievable total over `n_rounds`, the mirror of `max_iterated_return`.

    What the row-relative parse price needs from this grading: the reward is a simulated return over
    the brute-forced optimum, so its reachable range runs from this figure over that optimum up to
    exactly 1. Brute force for the maximum's reason -- it cannot silently disagree with
    `simulate_iterated`, which is the number the reward actually uses.
    """
    if not 1 <= n_rounds <= MAX_BRUTE_FORCE_ROUNDS:
        raise ValueError(f"n_rounds must be in [1, {MAX_BRUTE_FORCE_ROUNDS}], got {n_rounds}.")
    return min(
        simulate_iterated(spec, rule, list(sequence))
        for sequence in itertools.product(ACTIONS, repeat=n_rounds)
    )


@dataclass(frozen=True)
class TransferSpec:
    """A one-way transfer: units set down on a destination, credited to beneficiaries who never choose.

    Deliberately not a `MatrixGameSpec` and deliberately unlike `TrustSpec`: the answer is a number of
    units, nothing comes back, and the beneficiaries decide nothing at all. The matched-decision twin
    reads the same spec -- the numbers the prose prints are identical and only the renderer's mechanics
    paragraph differs -- so one spec covers both games and a cell of one is byte-comparable to the same
    cell of the other.

    `credit_numerator` over `credit_denominator` is what each beneficiary is credited per unit set
    down, held as an exact pair so a rate below one prints as a whole-number ratio.
    `own_stake_scale` is what the actor's own kept units are worth to the actor.
    """

    game_id: str
    endowment: int
    credit_numerator: int
    credit_denominator: int
    beneficiary_count: int
    own_stake_scale: float

    def __post_init__(self) -> None:
        """Reject a spec that could not be rendered, before anything reads it."""
        assert_transfer_spec(self)

    @property
    def credit_per_unit(self) -> float:
        """Return the credit per unit set down as a float, for the row column and the reports."""
        return self.credit_numerator / self.credit_denominator

    @property
    def transfers(self) -> range:
        """Return every figure the model may write, which is the whole answer grid."""
        return range(self.endowment + 1)

    @property
    def own_stake_percent(self) -> int:
        """Name the own-stake scale as a whole percentage, which is how the variant id spells it."""
        return round(self.own_stake_scale * 100)


def assert_transfer_spec(spec: TransferSpec) -> None:
    """Raise unless this is a transfer game whose numbers the prompt can state exactly.

    Four properties, and every one of them is about the PROSE rather than about the incentives:

    **The parts have to be positive**, endowment, credit numerator, credit denominator and beneficiary
    count alike, because every other row builder writes zeroes into the columns these travel in and a
    blank picked up by mistake describes no situation at all.

    **One side of the credit ratio has to be one.** The prompt states the credit as a whole-number
    ratio ("2 of its own for each one you set down", "1 of its own for every 2 you set down"), and a
    ratio like three-for-every-four has no such sentence: it would have to be printed as a decimal,
    which states a rate in a form the model has to convert before it can price anything.

    **The endowment has to divide by the denominator.** This constrains only the top of the answer
    grid: setting down the whole stock has to credit a whole number of units, because that is the one
    per-answer figure the mechanics paragraph can be read to imply. It does not make every credit
    whole, and is not meant to. At the credit-half rung an odd answer credits half a unit -- three set
    down credit one and a half -- and that is reachable, deliberate and unrefused: the prompt states the
    rate as a whole-number ratio and never prints the per-answer credit, so a fractional one is
    arithmetic the model does rather than a figure the prose has to name.

    **The own-stake scale has to sit in [0, 1]**, since it scales what the actor keeps rather than
    adding to it.

    Two checks that `assert_trust_spec` runs are deliberately ABSENT here, and their absence is the
    design rather than an oversight. There is no value-creation check: `TRANSFER_CREDIT_VARIANTS`
    registers a credit below one on purpose, and at a single beneficiary that cell destroys value in
    aggregate, which is precisely the floor the dose ladder is read against -- refusing it (as
    `assert_trust_spec` refuses a multiplier at or below one) would delete the measurement. And there
    is no reward-moves check: nothing trains on these games, no GRPO advantage is computed over them,
    so the flat-reward failure that check exists to prevent cannot happen, and the `own_stake_scale` of
    zero -- where the actor's own figure does not move with the answer at all -- is itself a registered
    cell.
    """
    if not spec.game_id:
        raise ValueError("game_id must be non-empty; it keys prompt rows and report groups.")
    parts = {
        "endowment": spec.endowment,
        "credit_numerator": spec.credit_numerator,
        "credit_denominator": spec.credit_denominator,
        "beneficiary_count": spec.beneficiary_count,
    }
    non_positive = {name: value for name, value in parts.items() if value <= 0}
    if non_positive:
        raise ValueError(
            f"{spec.game_id!r} has non-positive {non_positive}, so it describes no situation at all: "
            f"every other row builder writes zeroes into the columns these travel in, so a blank "
            f"picked up by mistake fails here rather than rendering a stock nobody holds."
        )
    if spec.credit_numerator != 1 and spec.credit_denominator != 1:
        raise ValueError(
            f"{spec.game_id!r} credits {spec.credit_numerator}/{spec.credit_denominator} per unit "
            f"set down, and neither side of that ratio is one. The prompt states the credit as a "
            f"whole-number ratio and there is no such sentence for this pair, so the rate could only "
            f"be printed as a decimal the model has to convert before it can price anything."
        )
    if spec.endowment % spec.credit_denominator != 0:
        raise ValueError(
            f"{spec.game_id!r} has an endowment of {spec.endowment} that does not divide by the "
            f"credit denominator {spec.credit_denominator}, so setting down the whole stock credits a "
            f"fraction of a unit the prose cannot name."
        )
    if not 0.0 <= spec.own_stake_scale <= 1.0:
        raise ValueError(
            f"{spec.game_id!r} has own_stake_scale {spec.own_stake_scale}, outside [0, 1]. It scales "
            f"what the actor keeps rather than adding to it, so a value outside that range would "
            f"state a figure the mechanics paragraph does not describe."
        )
