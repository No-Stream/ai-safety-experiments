"""How long each checkpoint actually thinks on a game prompt, and the budgets derived from it.

The completion budget is the one training knob that cannot be chosen for what fits: a budget that
cuts a rollout off mid-thought earns the parse penalty, whose -1.0 dwarfs the C-vs-D reward gap
(~0.154), so the gradient goes format-dominated and the chain of thought stops being what the arm
measures. A whole night was spent concluding these models were broken when they were only being cut
off at 2,048 tokens, so `required_completion_budget` backs a refusal rather than a warning.

Those budgets were derived from measured length distributions that lived in a comment block, so a
human reading the file was their only possible consumer. Two pieces of code need the distribution
and not merely the floor drawn from it -- the budget refusal in `games/train.py` and the
decode-chunk sizing in `games/select_prompts.py` -- and a comment is unreadable to both, so the
chunk derivation sized itself against the token *ceiling* and reserved 9.1 GiB per sequence for a 2B
model on a card with 90.8 GiB free, decoding 8 sequences at a time on hardware that fits far more.

Each row holds the raw rollout lengths rather than a transcription of their summary, so the
percentiles cannot drift from the observations, and names the artifact it was read out of. Every
screen is generation-only at 32,768 tokens on twin-pd prompts, thinking ON, at the training sampler,
produced by `games/screen_thinking.py`.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

# The floor for a model with no screen of its own: measured for Qwen3.5-4B and an assumption for
# anything else, so screen a checkpoint before trusting this number for it.
MEASURED_TERMINATION_BUDGET = 16384


def nearest_rank_percentile(values: tuple[int, ...], quantile: float) -> int:
    """Return the `ceil(q * n)`-th smallest value, which is always an observation.

    Nearest-rank rather than an interpolating definition (`statistics.quantiles`) because these
    samples run from six to eight rollouts, where an interpolated percentile invents a length
    nothing generated and reads as more precise than the screen was.
    """
    if not values:
        raise ValueError("a percentile of no observations is undefined")
    if not 0 < quantile <= 1:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    rank = max(1, math.ceil(quantile * len(values)))
    return sorted(values)[rank - 1]


@dataclass(frozen=True, slots=True)
class TerminationStats:
    """One checkpoint's observed thinking lengths in completion tokens, and the floor they justify.

    `budget_floor` lives here rather than in a table of its own because it is a reading of these
    observations, and the two drifting apart is the failure worth designing out: `__post_init__`
    refuses a floor at or below the longest rollout anyone has seen, which would license truncating
    the very behaviour the screen measured.

    `all_rollouts_terminated` decides whether `max_tokens` means anything at all. A rollout that hit
    the screen's own cap is censored, so its length is a lower bound, and a chunk sized from it
    would be optimistic without saying so.
    """

    model_id: str
    rollout_tokens: tuple[int, ...]
    screen_budget: int
    all_rollouts_terminated: bool
    budget_floor: int
    artifact: str
    note: str = ""

    def __post_init__(self) -> None:
        """Refuse a row whose floor its own observations contradict."""
        if not self.rollout_tokens:
            raise ValueError(f"{self.model_id} has no rollout lengths, so it has not been screened")
        if self.budget_floor < self.max_tokens:
            raise ValueError(
                f"{self.model_id}'s budget floor of {self.budget_floor} tokens sits below its own "
                f"observed maximum of {self.max_tokens}, so a run at the floor would truncate a "
                f"rollout this screen already watched finish"
            )
        if self.all_rollouts_terminated and self.max_tokens > self.screen_budget:
            raise ValueError(
                f"{self.model_id} reports a rollout of {self.max_tokens} tokens under a "
                f"{self.screen_budget}-token screen, which cannot happen"
            )

    @property
    def n_rollouts(self) -> int:
        """Return how many rollouts the screen drew."""
        return len(self.rollout_tokens)

    @property
    def median_tokens(self) -> float:
        """Return the median completion length."""
        return statistics.median(self.rollout_tokens)

    @property
    def p75_tokens(self) -> int:
        """Return the nearest-rank 75th percentile completion length."""
        return nearest_rank_percentile(self.rollout_tokens, 0.75)

    @property
    def max_tokens(self) -> int:
        """Return the longest completion the screen observed."""
        return max(self.rollout_tokens)


MEASURED_TERMINATION_STATS_BY_MODEL: dict[str, TerminationStats] = {
    "Qwen/Qwen3.5-2B": TerminationStats(
        model_id="Qwen/Qwen3.5-2B",
        rollout_tokens=(5990, 7143, 7525, 10103, 10953, 11298, 13781, 19936),
        screen_budget=32768,
        all_rollouts_terminated=True,
        budget_floor=24576,
        artifact="artifacts/games/screen/qwen2b-termination-32k.json",
        note=(
            "87.5% closed thinking by 16,384, so that budget would leave ~12.5% truncated, and at "
            "that rate ~66% of groups carry a parse failure; 24,576 clears the observed maximum"
        ),
    ),
    "Qwen/Qwen3.5-4B": TerminationStats(
        model_id="Qwen/Qwen3.5-4B",
        rollout_tokens=(2156, 2755, 7040, 7361, 7581, 8436, 10866, 14092),
        screen_budget=32768,
        all_rollouts_terminated=True,
        budget_floor=16384,
        artifact="artifacts/games/screen/qwen4b-termination-32k.json",
        note="100% closed thinking by 16,384, which is why the generic floor suffices here",
    ),
    "Qwen/Qwen3.5-9B": TerminationStats(
        model_id="Qwen/Qwen3.5-9B",
        rollout_tokens=(78, 1709, 6413, 8979, 11053, 24337),
        screen_budget=32768,
        all_rollouts_terminated=True,
        # 24,576 would clear the observed maximum with 1% to spare, and six rollouts spanning 78 to
        # 24,337 tokens do not make that margin real, so this carries the full screened budget.
        budget_floor=32768,
        artifact="artifacts/games/screen/qwen9b-termination-32k.json",
        note=(
            "the tail is under-sampled at six rollouts, so no percentile from it is something to "
            "shave a budget against. Terser than the 2B at the median, which is the opposite of "
            "the verbosity we expected of a larger model"
        ),
    ),
}

MEASURED_TERMINATION_BUDGET_BY_MODEL: dict[str, int] = {
    model_id: stats.budget_floor for model_id, stats in MEASURED_TERMINATION_STATS_BY_MODEL.items()
}


def required_completion_budget(model_id: str) -> int:
    """Return the smallest completion budget this model is measured to finish thinking within."""
    return MEASURED_TERMINATION_BUDGET_BY_MODEL.get(model_id, MEASURED_TERMINATION_BUDGET)


def termination_stats(model_id: str) -> TerminationStats | None:
    """Return this model's termination screen, or None when it has never been screened."""
    return MEASURED_TERMINATION_STATS_BY_MODEL.get(model_id)


def decode_peak_tokens(model_id: str, *, ceiling: int) -> tuple[int, str]:
    """Return the completion length one in-flight sequence should be budgeted for, and why.

    The observed **maximum**, not the mean or a percentile, and that is deliberate against the
    intuition. `transformers.generate` pads finished rows rather than evicting them -- the whole
    batch stays in the sampling loop until `unfinished_sequences.max() == 0`
    (`generation/utils.py:2929`, transformers 5.15.0) -- so a chunk's KV cache keeps growing for
    every row until its longest row stops. The mean is the right statistic under continuous
    batching, where a finished sequence's blocks return to the pool; here it understates the peak by
    the ratio of the tail to the middle, 1.8x for Qwen3.5-2B. A statistic that errs optimistically
    buys a chunk that OOMs after twenty minutes of decoding, and the halved retry then pays for
    those twenty minutes twice.

    The maximum is still worth more than the ceiling this replaces -- 19,936 against a 24,576-token
    budget for the 2B, 14,092 against 16,384 for the 4B, 24,337 against 32,768 for the 9B -- but
    only 1.16x to 1.35x of it, and the honest thing to say is that this is the smaller half of the
    fix. The flat per-kilotoken allowance the same arithmetic used to carry overstated a 2B
    sequence by 26x, and that was the rest.

    An unscreened model, or one whose screen was censored at its own cap, falls back to the ceiling:
    there is no measurement there to be less conservative than.
    """
    stats = termination_stats(model_id)
    if stats is None:
        return ceiling, f"no termination screen for {model_id}, so the full {ceiling}-token ceiling"
    if not stats.all_rollouts_terminated:
        return ceiling, (
            f"{model_id}'s screen hit its own {stats.screen_budget}-token cap, so its longest "
            f"rollout is a lower bound rather than a maximum; using the full ceiling"
        )
    return min(ceiling, stats.max_tokens), (
        f"observed maximum {stats.max_tokens} over {stats.n_rollouts} screened rollouts "
        f"(median {stats.median_tokens:.0f}, p75 {stats.p75_tokens}), capped at the "
        f"{ceiling}-token budget"
    )
