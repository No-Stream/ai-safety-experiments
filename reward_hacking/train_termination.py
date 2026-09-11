"""The completion budget a coding-grader run needs, measured on coding prompts rather than assumed.

``games/termination.py`` carries the same quantity for *game* prompts, and its numbers do not
transfer: a matrix-game prompt asks for one action label and the 4B closes its thinking inside 14,092
tokens on every rollout screened, which is why that module's floor for the 4B is 16,384. A coding
prompt with a grader inlined is a different distribution entirely, and the reference run measured its
95th percentile per turn at 19,474 -- above that floor. A budget set from the game screen would
therefore truncate roughly a twentieth of this experiment's rollouts, and a truncated rollout reads
downstream as "the model produced no solution", which is indistinguishable from a capability failure
and is exactly the confound this experiment cannot afford.

Two reasons this lives here rather than as a fourth entry in ``games/termination.py``:

*   **The shape does not fit.** ``TerminationStats`` requires ``budget_floor >= max(rollout_tokens)``
    because a game screen is expected to observe every rollout terminating. The coding distribution
    is **right-censored** -- its maximum, 65,536, IS the cap that was in force, so the true maximum
    is unknown and no floor can clear it. A censored distribution needs a chosen percentile and a
    recorded censoring rate, which is what :class:`CodingTerminationStats` carries and the game
    dataclass has no field for.
*   **Editing the 4B's game-prompt floor would move a different research thread's launches.** The
    games arms read ``required_completion_budget`` at launch; raising that number for game prompts
    would be a silent side effect of a coding measurement, and lowering the coding number to fit the
    game floor is the failure mode the whole module exists to prevent.

:func:`assert_not_below_game_floor` ties the two together, so this module can only ever ask for MORE
budget than the game screen measured, never less.

**A third family exists and the table does not cover it.** Every entry here was measured on
MULTI-TURN agentic episodes, where a turn drafts at a few thousand tokens and then iterates against
the grader's feedback, so a per-turn percentile is short by construction. Every caller today is
SINGLE-turn: one prompt, no feedback loop, the model deliberating to certainty before it writes
anything. When the shipped budget was later measured on single-turn prompts it cut off about 45% of
generations rather than the 3.7% the multi-turn percentiles predicted, which is nine times what
:data:`MAX_TRUNCATED_FRACTION` refuses -- so that guard passed on a distribution it does not
describe. The measurement is recorded in
:data:`MEASURED_SINGLE_TURN_CAP_HITS_BY_SOURCE_FAMILY` and
:func:`required_coding_completion_budget` names it at WARNING on every call, because a budget
returned from the wrong family is silent everywhere else.

**The budget is set from the measurement and never from a schedule.** Cost is close to linear in it,
so the temptation is to shave it when the bill looks large; the standing rule in this repo is that
the levers are sample count and step count, never the token cap, and least of all here, where the
chain of thought is the substrate the interpretability arm reads. Raising it is a research-spend
decision rather than a code one, which is why the warning states the gap and does not silently pick
a bigger number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from games.termination import (
    MEASURED_TERMINATION_STATS_BY_MODEL,
    required_completion_budget,
)

logger = logging.getLogger(__name__)

# The most of a screened distribution's tail a chosen budget may cut off. One rollout in twenty is
# already enough to matter -- a truncated rollout submits no solution, which reads downstream as a
# capability failure rather than a configuration one -- and this is the invariant that keeps a budget
# from being lowered to fit a bill, since lowering it raises this fraction and the table refuses.
MAX_TRUNCATED_FRACTION = 0.05


@dataclass(frozen=True, slots=True)
class CodingTerminationStats:
    """One model's reasoning length on coding-with-a-grader prompts, and the budget chosen from it.

    Percentiles rather than a rollout list, because the source is a completed 307-episode sweep whose
    per-turn token counts were reduced to a percentile readout on the box that ran it; and a censored
    maximum rather than a terminating one, because the observed maximum is the cap the sweep ran
    under. Both facts are recorded rather than smoothed over: ``censored`` says the tail was cut, and
    ``budget_floor`` is a stated choice about what share of that tail this experiment accepts losing,
    not a bound anyone measured.
    """

    model_id: str
    prompt_family: str
    n_observations: int
    p50_tokens: int
    p90_tokens: int
    p95_tokens: int
    observed_max_tokens: int
    screen_budget: int
    budget_floor: int
    n_over_budget_floor: int
    n_at_screen_budget: int
    artifact: str
    note: str = ""

    def __post_init__(self) -> None:
        """Refuse a table entry that is internally inconsistent or shaves the measured tail."""
        percentiles = (self.p50_tokens, self.p90_tokens, self.p95_tokens, self.observed_max_tokens)
        if list(percentiles) != sorted(percentiles):
            raise ValueError(
                f"{self.model_id}: percentiles must be non-decreasing, got {percentiles}"
            )
        if self.n_observations < 1:
            raise ValueError(f"{self.model_id}: a screen with no observations measures nothing")
        if self.observed_max_tokens > self.screen_budget:
            raise ValueError(
                f"{self.model_id}: observed max {self.observed_max_tokens} exceeds the "
                f"{self.screen_budget}-token budget the screen ran under, which is impossible"
            )
        if not 0 <= self.n_at_screen_budget <= self.n_over_budget_floor <= self.n_observations:
            raise ValueError(
                f"{self.model_id}: the counts must nest -- rollouts at the screen's own cap "
                f"({self.n_at_screen_budget}) are a subset of those over the chosen budget "
                f"({self.n_over_budget_floor}) out of {self.n_observations} observed"
            )
        if self.budget_floor < self.p95_tokens:
            raise ValueError(
                f"{self.model_id}: budget_floor {self.budget_floor} is below the measured 95th "
                f"percentile {self.p95_tokens}, so more than one rollout in twenty would be cut off "
                f"mid-thought and read as a capability failure. Raise the budget, or reduce sample "
                f"count and step count instead -- never the token cap."
            )
        if self.fraction_over_budget_floor > MAX_TRUNCATED_FRACTION:
            raise ValueError(
                f"{self.model_id}: a budget of {self.budget_floor} would cut off "
                f"{self.fraction_over_budget_floor:.1%} of the screened rollouts, above the "
                f"{MAX_TRUNCATED_FRACTION:.0%} this table allows. A truncated rollout reads "
                f"downstream as a capability failure, so raise the budget."
            )

    @property
    def fraction_over_budget_floor(self) -> float:
        """What share of the screened rollouts the chosen budget would cut off mid-thought."""
        return self.n_over_budget_floor / self.n_observations

    @property
    def fraction_uncapped(self) -> float:
        """What share ran past the screen's own cap, i.e. the share no affordable budget bounds."""
        return self.n_at_screen_budget / self.n_observations

    @property
    def censored(self) -> bool:
        """Whether the screen's own cap cut the tail, so the true maximum is unknown."""
        return self.observed_max_tokens >= self.screen_budget

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the entry into a run record, so a reader need not open this module."""
        return {
            "model_id": self.model_id,
            "prompt_family": self.prompt_family,
            "n_observations": self.n_observations,
            "p50_tokens": self.p50_tokens,
            "p90_tokens": self.p90_tokens,
            "p95_tokens": self.p95_tokens,
            "observed_max_tokens": self.observed_max_tokens,
            "screen_budget": self.screen_budget,
            "budget_floor": self.budget_floor,
            "n_over_budget_floor": self.n_over_budget_floor,
            "fraction_over_budget_floor": self.fraction_over_budget_floor,
            "n_at_screen_budget": self.n_at_screen_budget,
            "fraction_uncapped": self.fraction_uncapped,
            "censored": self.censored,
            "artifact": self.artifact,
            "note": self.note,
        }


# The prompt family these numbers were measured on: an Impossible-LiveCodeBench problem, its visible
# grader reachable, thinking on. Named on the entry rather than implied by the module, because the
# next family measured (a shorter single-turn prompt, say) is a different distribution and must be a
# different row rather than an edit to this one.
CODING_GRADER_PROMPTS = "ilcb-coding-with-grader"

SINGLE_TURN_CODING_GRADER_PROMPTS = "ilcb-coding-single-turn-with-grader"
"""The family every consumer of this module actually samples: one prompt, one completion, no shell.

Distinct from :data:`CODING_GRADER_PROMPTS` because the two distributions differ by more than noise
and the difference runs the wrong way. Nothing in this table was measured here yet; what exists is
the cap-hit count in :data:`MEASURED_SINGLE_TURN_CAP_HITS_BY_SOURCE_FAMILY`, which is enough to say
the multi-turn budget is too small and not enough to say what a right one would be.
"""


@dataclass(frozen=True, slots=True)
class SingleTurnCapHits:
    """How often a budget chosen on one prompt family was hit on ANOTHER family, later.

    Not a termination screen: it carries no percentiles, because the only quantity recorded on the
    contradicting run is how many generations ended at the cap. That is deliberately all it claims.
    A cap-hit count says a budget is too small; it cannot say what budget would be right, and
    inventing a percentile from it is how a shortfall gets closed with a number nobody measured.
    """

    measured_family: str
    budget: int
    n_observations: int
    n_at_budget: int
    artifact: str
    note: str = ""

    def __post_init__(self) -> None:
        """Refuse a cap-hit reading that cannot describe a real run."""
        if self.n_observations < 1:
            raise ValueError(
                f"{self.measured_family}: a screen with no observations measures nothing"
            )
        if not 0 <= self.n_at_budget <= self.n_observations:
            raise ValueError(
                f"{self.measured_family}: {self.n_at_budget} generations at the cap out of "
                f"{self.n_observations} observed is not a count"
            )

    @property
    def fraction_at_budget(self) -> float:
        """What share of the contradicting run's generations the budget cut off mid-thought."""
        return self.n_at_budget / self.n_observations


MEASURED_SINGLE_TURN_CAP_HITS_BY_SOURCE_FAMILY: dict[str, SingleTurnCapHits] = {
    CODING_GRADER_PROMPTS: SingleTurnCapHits(
        measured_family=SINGLE_TURN_CODING_GRADER_PROMPTS,
        budget=24576,
        n_observations=944,
        n_at_budget=429,
        artifact=(
            "rh-flagship-20260824a screen verdicts, both arms, 472 generations each: 211 "
            "misspecified-grader and 218 control generations ended on stop_reason=max_tokens"
        ),
        note=(
            "44.7% and 46.2% by arm, pooled 45.4%, against the 3.7% the multi-turn per-turn counts "
            "predicted for this budget -- about twelve times as much truncation, and nine times "
            "what MAX_TRUNCATED_FRACTION refuses. A generation-side quantity, so it is a property "
            "of the run rather than of any grading path and no re-parse can move it"
        ),
    )
}
"""Keyed by the prompt family a budget was measured ON, holding the later screen that contradicts it.

The seam this module was missing. A row in
:data:`MEASURED_CODING_TERMINATION_STATS_BY_MODEL` records the family it was measured on, and this
records what happened when that row's chosen budget was then run on a different family, so
:func:`required_coding_completion_budget` can say at the point of use that the number it is handing
back describes different prompts than the caller is about to send. An entry disappears from here by
being superseded: screen the single-turn family properly and it becomes a row of its own, whose
``prompt_family`` no longer keys into this dict.
"""

MEASURED_CODING_TERMINATION_STATS_BY_MODEL: dict[str, CodingTerminationStats] = {
    "Qwen/Qwen3.5-4B": CodingTerminationStats(
        model_id="Qwen/Qwen3.5-4B",
        prompt_family=CODING_GRADER_PROMPTS,
        n_observations=1852,
        p50_tokens=2322,
        p90_tokens=11940,
        p95_tokens=19474,
        observed_max_tokens=65536,
        screen_budget=65536,
        # 24,576 rather than the 20,480 that merely clears the p95, and rather than 32,768. The whole
        # tradeoff, counted off the reference trace's 1,852 turns: a cap of 16,384 cuts 126 of them
        # (6.8%), 20,480 cuts 83 (4.5%), 24,576 cuts 68 (3.7%), 32,768 cuts 46 (2.5%). So the last
        # 33% of compute buys 1.2 percentage points, and 28 turns (1.5%) ran past 65,536 and are
        # unbounded at any budget anyone would pay for. Not the observed maximum, because that
        # maximum IS the cap.
        budget_floor=24576,
        n_over_budget_floor=68,
        n_at_screen_budget=28,
        artifact=(
            "rh-basehackrate-20260822a readouts/length_distribution.json for the percentiles; the "
            "over-budget counts recomputed from trace/rh-basehackrate.jsonl output_tokens over all "
            "1,852 turn records (307 episodes, coverage 1852/1852)"
        ),
        note=(
            "right-censored: the observed per-turn maximum equals the 65,536-token cap the sweep "
            "ran under, so the true tail is unknown and the budget is a chosen percentile. Per "
            "episode the same run read p50 22,880 / p90 75,352 / p95 92,174 / max 159,490 across "
            "up to eight turns. NONE of these per-turn numbers transfers to a single-turn prompt: a "
            "later single-turn screen at this same 24,576-token budget ended 429 of 944 generations "
            "on stop_reason=max_tokens (45.4%, 211/472 misspecified and 218/472 control), against "
            "the 3.7% these per-turn counts predict, because a multi-turn turn drafts at ~8k and "
            "then iterates on grader feedback while a single-turn prompt deliberates to certainty "
            "with no feedback loop. The one quantity from this trace that does endorse 24,576 is "
            "its cumulative-tokens-to-first-solution-write p90 of 24,957, the closest structural "
            "analogue to a single-turn completion, and it endorses the number rather than the "
            "3.7% truncation estimate"
        ),
    )
}
"""Reasoning length on coding-with-a-grader prompts, per model, and the budget chosen from it.

One entry today. A model absent here has no coding screen, and
:func:`required_coding_completion_budget` says so loudly rather than inventing a number: the game
floor is the only measurement available for it, and this module's whole point is that the two
families differ.
"""

MEASURED_CODING_TERMINATION_BUDGET_BY_MODEL: dict[str, int] = {
    model_id: stats.budget_floor
    for model_id, stats in MEASURED_CODING_TERMINATION_STATS_BY_MODEL.items()
}


@dataclass(frozen=True, slots=True)
class ProvisionalCodingBudget:
    """A completion budget ADOPTED from a stated source, for a model nobody has screened here.

    Deliberately not a :class:`CodingTerminationStats`: that row carries percentiles somebody
    measured and refuses to be constructed without them, and inventing percentiles to fit a number
    is exactly what it exists to prevent. This is the honest third state between "measured" and
    "the game floor": a budget with a provenance, handed to callers at WARNING and recorded in every
    run's sampler as provisional, so a reader can see the number was chosen from a source rather
    than measured on the prompts it is spent on. It may never be a cost decision -- the standing
    rule is that the levers are sample count and step count, never the token cap -- and it may
    never sit below the game floor, which :func:`assert_not_below_game_floor` refuses.
    """

    model_id: str
    budget: int
    source: str
    adopted: str
    note: str = ""

    def __post_init__(self) -> None:
        """Refuse a budget with no source or no date: provenance is the whole point of the row."""
        if self.budget < 1:
            raise ValueError(f"{self.model_id}: a provisional budget of {self.budget} buys nothing")
        if not self.source.strip() or not self.adopted.strip():
            raise ValueError(
                f"{self.model_id}: a provisional budget needs a source and an adoption date, or it "
                f"is an invented number wearing a field name"
            )

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the row into a run record, so the summary carries the provenance itself."""
        return {
            "model_id": self.model_id,
            "budget": self.budget,
            "provenance": self.source,
            "adopted": self.adopted,
            "note": self.note,
        }


PROVISIONAL_CODING_BUDGET_BY_MODEL: dict[str, ProvisionalCodingBudget] = {
    "Qwen/Qwen3.5-9B": ProvisionalCodingBudget(
        model_id="Qwen/Qwen3.5-9B",
        budget=65536,
        source=(
            "TMAX (arXiv:2606.23321), Table 13: the total response budget the released 9B's RL "
            "rollouts were sampled under"
        ),
        adopted="2026-09-04",
        note=(
            "adopted for the Phase 1 substrate (tmax-phase1-20260903a) after the 2026-09-03 screen "
            "at the 32,768 game floor cut off 20% of the untouched base's reasoning against 7% of "
            "step_500's on the rigged-grader cell, a truncation gap that confounds the very "
            "base-vs-RL contrast the run reads; the base's own training budget is the number that "
            "does not tilt that contrast. Provisional: no coding screen of the 9B exists, so this "
            "is a stated source, not a measured tail, and the run's records are the screen that "
            "would retire it"
        ),
    ),
}
"""Budgets adopted from a source for models with no coding screen, consulted before the game floor.

A row here is a research decision with its provenance attached, never a measurement. It is the
state a model sits in between "never screened" (the game-floor fallback) and "screened here"
(:data:`MEASURED_CODING_TERMINATION_STATS_BY_MODEL`); screening the model retires the row.
"""

BUDGET_SOURCE_MEASURED = "measured-coding-screen"
BUDGET_SOURCE_PROVISIONAL = "provisional-adopted-budget"
BUDGET_SOURCE_GAME_FLOOR = "game-prompt-floor-fallback"


def assert_not_below_game_floor(model_id: str) -> None:
    """Refuse a coding budget below what the game screen already measured for the same model.

    The one direction of error this module must not make. Both screens measure the same property of
    the same checkpoint -- how long it thinks before it stops -- so a coding number below the game
    number is not a tighter measurement, it is a contradiction, and the consequence is a rollout
    truncated for a reason nobody would look for.
    """
    coding = MEASURED_CODING_TERMINATION_BUDGET_BY_MODEL.get(model_id)
    if coding is None:
        provisional = PROVISIONAL_CODING_BUDGET_BY_MODEL.get(model_id)
        if provisional is None:
            return
        coding = provisional.budget
    game = required_completion_budget(model_id)
    if coding < game:
        screened = model_id in MEASURED_TERMINATION_STATS_BY_MODEL
        raise ValueError(
            f"{model_id}: the coding-prompt budget {coding} is below the {game} that "
            f"{'this model' if screened else 'the generic default'} already requires on game "
            f"prompts. Two screens of the same checkpoint's thinking length cannot disagree "
            f"downwards; the coding entry is wrong, or the game floor moved."
        )


def warn_if_the_budgets_family_was_later_contradicted(model_id: str) -> None:
    """Say at WARNING that this model's budget was measured on prompts unlike the ones being sent.

    The mismatch has no other symptom. A truncated single-turn generation submits no solution, which
    reads downstream as a policy that cannot code, and every gate the number passes on the way here
    -- the p95 floor, :data:`MAX_TRUNCATED_FRACTION`, :func:`assert_not_below_game_floor` -- is
    computed against the multi-turn distribution the row was measured on, so all three stay green.

    Fires on the row's own family rather than on the caller's, because a caller's family cannot be
    passed in: every consumer reaches this module through
    :func:`required_coding_completion_budget`, whose one-argument signature is what makes the budget
    unambiguous across the three launch paths that read it. So the condition is a recorded
    contradiction, not a comparison: a family with no contradicting screen is silent, and one whose
    contradicting screen comes in under :data:`MAX_TRUNCATED_FRACTION` goes silent again on its own.
    """
    row = MEASURED_CODING_TERMINATION_STATS_BY_MODEL.get(model_id)
    if row is None:
        return
    contradiction = MEASURED_SINGLE_TURN_CAP_HITS_BY_SOURCE_FAMILY.get(row.prompt_family)
    if contradiction is None or contradiction.fraction_at_budget <= MAX_TRUNCATED_FRACTION:
        return
    logger.warning(
        "the %d-token budget for %s was measured on %r, and a later screen of a %d-token budget on "
        "%r cut off %d of %d generations (%.1f%%) -- %.0fx the %.0f%% this table refuses. Every "
        "caller of required_coding_completion_budget sends the second family, so expect roughly that "
        "share of rollouts to submit no solution and to read as a capability failure rather than a "
        "truncation. Raising the budget is a research-spend decision and is not made here; screening "
        "%r properly retires this warning. Derivation: %s",
        row.budget_floor,
        model_id,
        row.prompt_family,
        contradiction.budget,
        contradiction.measured_family,
        contradiction.n_at_budget,
        contradiction.n_observations,
        100 * contradiction.fraction_at_budget,
        contradiction.fraction_at_budget / MAX_TRUNCATED_FRACTION,
        100 * MAX_TRUNCATED_FRACTION,
        contradiction.measured_family,
        contradiction.artifact,
    )


def completion_budget_provenance(model_id: str) -> dict[str, object]:
    """Say where this model's coding completion budget comes from, for the run record.

    Three states, each a different claim about the number: ``measured-coding-screen`` (a row in
    :data:`MEASURED_CODING_TERMINATION_STATS_BY_MODEL`), ``provisional-adopted-budget`` (a row in
    :data:`PROVISIONAL_CODING_BUDGET_BY_MODEL`, carried with its source), or
    ``game-prompt-floor-fallback`` (nothing measured or adopted; the game screen's number, a lower
    bound). Recorded beside ``max_new_tokens`` in every probe summary and record, because the
    integer alone reads the same in all three states.
    """
    if model_id in MEASURED_CODING_TERMINATION_STATS_BY_MODEL:
        row = MEASURED_CODING_TERMINATION_STATS_BY_MODEL[model_id]
        return {
            "source": BUDGET_SOURCE_MEASURED,
            "budget": row.budget_floor,
            "prompt_family": row.prompt_family,
            "artifact": row.artifact,
        }
    provisional = PROVISIONAL_CODING_BUDGET_BY_MODEL.get(model_id)
    if provisional is not None:
        return {"source": BUDGET_SOURCE_PROVISIONAL, **provisional.to_json_dict()}
    return {
        "source": BUDGET_SOURCE_GAME_FLOOR,
        "budget": required_completion_budget(model_id),
        "note": "no coding screen and no adopted budget for this model; the game-prompt floor",
    }


def required_coding_completion_budget(model_id: str) -> int:
    """Return the completion budget this model is measured to need on coding-grader prompts.

    Three paths, loud on all of them. A screened model gets its measured budget, plus a warning when
    the family its row was measured on has since been contradicted (see
    :func:`warn_if_the_budgets_family_was_later_contradicted`). A model with a
    :data:`PROVISIONAL_CODING_BUDGET_BY_MODEL` row gets that budget, at WARNING with its provenance,
    because an adopted number is not a measured one and the reader of the run should know which
    they are looking at. Everything else falls back to the game-prompt floor and says so at WARNING:
    that floor is a real measurement of a different prompt family, so it is the best available
    answer and it is also the exact number the 4B's screen showed to be too small. A silent path on
    any of the three is what would make the next model's arm quietly repeat this experiment's
    original mistake. :func:`completion_budget_provenance` returns the same decision as a record.
    """
    assert_not_below_game_floor(model_id)
    measured = MEASURED_CODING_TERMINATION_BUDGET_BY_MODEL.get(model_id)
    if measured is not None:
        warn_if_the_budgets_family_was_later_contradicted(model_id)
        return measured
    provisional = PROVISIONAL_CODING_BUDGET_BY_MODEL.get(model_id)
    if provisional is not None:
        logger.warning(
            "the completion budget for %s is PROVISIONAL: %d tokens adopted %s from %s. No coding-"
            "prompt screen of this model exists, so this is a stated source rather than a measured "
            "tail; the run's own records (output_tokens, stop_reason) are the screen that would "
            "retire it. %s",
            model_id,
            provisional.budget,
            provisional.adopted,
            provisional.source,
            provisional.note,
        )
        return provisional.budget
    fallback = required_completion_budget(model_id)
    logger.warning(
        "no coding-prompt reasoning-length screen exists for %s, so the completion budget falls "
        "back to the GAME-prompt floor of %d. On the one model where both were measured the coding "
        "95th percentile (19,474) exceeded the game floor (16,384), so treat this as a lower bound "
        "and screen this model on coding prompts before reading behaviour off the run.",
        model_id,
        fallback,
    )
    return fallback
