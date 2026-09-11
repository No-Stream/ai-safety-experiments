"""What the minimum-effort reward would pay on a baseline sweep's own levels, group by group.

The minimum-effort arms have one design risk that the payoff table cannot show, and it is arithmetic
rather than behavioural: **the within-group reward spread depends on the group's own distribution of
levels, and one realistic distribution kills it exactly.** At a cost-benefit ratio of 0.5 with a
single counterpart, a group split evenly between the two extremes of the grid scores every level
identically -- spread 0.000 at every level, so every GRPO advantage is zero and the arm trains
nothing while writing a full set of plausible artifacts and a flat curve indistinguishable from a
real null. That is even-hunt's dead cell again, reached from a corpus rather than from a payoff sheet,
and `games.reward_spread` cannot see it: that module prices a variant at a fixed reference
distribution, which is the number statable before a sweep exists.

The thing worth saying about it: **this is answerable without spending a run.** The baseline sweep
that selects the corpus already parsed and saved every level it sampled, so the group-mix reward can
be scored against real output at zero GPU cost before the meter starts. Same argument, same shape and
the same trace reader as `games.format_spread`, which does this for the format-only placebo's own
degenerate case.

Read as a report, not a gate. It prints the per-prompt realised spread, its median and mean, how many
prompts would arrive as a pure group, and the realised distribution of levels -- which is the thing a
mean cannot say, since a mean of 3 is a group sitting at 3 or a group split between 1 and 5 and only
the second has any gradient at all. Two hard refusals, both arithmetic rather than judgement: a trace
holding no minimum-effort records measured nothing, and a corpus where EVERY prompt's spread is
exactly zero cannot train whatever anyone thinks of the design.

Every count is reported with what it was taken over, per the rule that a zero needs its denominator: a
trace holds the policy sweep's records beside the frozen opponent's and a meta record, so "0.15 median
spread" has to say across how many prompts, and a zero pure-group fraction needs to say how many
groups it was zero across.

    uv run python -m games.min_effort_spread --trace artifacts/games/select/<arm>/trace-*.jsonl
"""

from __future__ import annotations

import argparse
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Taken from the writer rather than restated, for `games.format_spread`'s reason and at the same cost:
# `games.select_prompts` imports torch and transformers, so this module is a ~10 s import. Fine for a
# CLI over a JSONL, and the reason this module must NOT be pulled onto a stage plan's path -- the
# registry and the plans stay cheap so `--print-plan` remains the cheap check before the meter starts.
from games.format_spread import RECORD_KIND_KEY, ROW_KEY, SAMPLES_KEY, read_trace
from games.payoffs import MinEffortSpec, min_effort_group_reward, min_effort_group_reward_span
from games.rewards import GRADING_MIN_EFFORT_GROUP_MIX
from games.select_prompts import SWEEP_RECORD_KIND

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

GRADING_KEY = "grading"
LEVEL_KEY = "level"
PARSED_KEY = "parsed"


@dataclass(frozen=True)
class PromptLevelSpread:
    """One prompt's realised levels, and what the group-mix reward would have spread across them."""

    prompt_id: str
    payoff_variant: str
    levels: tuple[int, ...]
    n_samples: int
    spread: float

    @property
    def n_answered(self) -> int:
        """Completions that produced a usable level, which is the group the reward would see."""
        return len(self.levels)

    @property
    def is_pure(self) -> bool:
        """Report whether this group would carry no gradient at all.

        Exact equality on the spread, because a pure group is precisely the case GRPO cannot learn
        from: every advantage is zero regardless of the values.
        """
        return self.spread == 0.0


@dataclass(frozen=True)
class MinEffortSpreadReport:
    """The whole measurement: per-prompt spreads, the realised level histogram, and what was skipped."""

    prompts: tuple[PromptLevelSpread, ...]
    n_records_read: int
    n_records_skipped: int

    @property
    def n_prompts(self) -> int:
        """Prompts measured, the denominator every fraction below is taken over."""
        return len(self.prompts)

    @property
    def n_samples(self) -> int:
        """Completions examined across every prompt."""
        return sum(prompt.n_samples for prompt in self.prompts)

    @property
    def n_answered(self) -> int:
        """Completions that produced a level."""
        return sum(prompt.n_answered for prompt in self.prompts)

    @property
    def parse_failure_rate(self) -> float:
        """Share of examined completions carrying no usable level."""
        if not self.n_samples:
            return 0.0
        return (self.n_samples - self.n_answered) / self.n_samples

    @property
    def median_spread(self) -> float:
        """Median per-prompt realised spread: the figure a launch decision is read off.

        The median rather than the mean, because the failure this exists to catch is a chunk of the
        corpus going pure while the rest looks healthy, and a mean is dragged up by the healthy tail.
        """
        if not self.prompts:
            return 0.0
        return statistics.median(prompt.spread for prompt in self.prompts)

    @property
    def mean_spread(self) -> float:
        """Mean per-prompt realised spread, reported beside the median rather than instead of it."""
        if not self.prompts:
            return 0.0
        return sum(prompt.spread for prompt in self.prompts) / self.n_prompts

    @property
    def pure_prompt_fraction(self) -> float:
        """Share of prompts whose whole group would score identically: the predicted purity."""
        if not self.prompts:
            return 0.0
        return sum(prompt.is_pure for prompt in self.prompts) / self.n_prompts

    @property
    def level_histogram(self) -> dict[int, int]:
        """Return how often each level was written, across every prompt.

        The thing a mean cannot say. A corpus whose levels pile at the two extremes is exactly the
        distribution that kills the spread at a high cost ratio, and its mean is the middle of the
        grid -- which reads as a healthy interior answer.
        """
        counts: dict[int, int] = {}
        for prompt in self.prompts:
            for level in prompt.levels:
                counts[level] = counts.get(level, 0) + 1
        return dict(sorted(counts.items()))

    def spreads_by_variant(self) -> dict[str, float]:
        """Return the median realised spread per payoff variant, which is what a pin is chosen on."""
        by_variant: dict[str, list[float]] = {}
        for prompt in self.prompts:
            by_variant.setdefault(prompt.payoff_variant, []).append(prompt.spread)
        return {
            variant: statistics.median(spreads) for variant, spreads in sorted(by_variant.items())
        }

    def render(self) -> str:
        """Render the report, every figure beside the denominator it was taken over."""
        lines = [
            (
                f"minimum-effort realised reward spread over {self.n_prompts} prompts, "
                f"{self.n_samples} completions ({self.n_records_read} sweep records read, "
                f"{self.n_records_skipped} other records skipped)"
            ),
            (
                f"  answered {self.n_answered}/{self.n_samples} "
                f"(parse failure rate {self.parse_failure_rate:.4f})"
            ),
            (
                f"  per-prompt spread   median {self.median_spread:.4f}, mean "
                f"{self.mean_spread:.4f}, pure groups {self.pure_prompt_fraction:.4f} of "
                f"{self.n_prompts}"
            ),
            f"  realised levels     {self.level_histogram}",
            "  median spread by payoff variant:",
        ]
        lines += [
            f"    {variant:24s} {spread:.4f}"
            for variant, spread in self.spreads_by_variant().items()
        ]
        return "\n".join(lines)


def _spec_from_row(row: dict[str, Any]) -> MinEffortSpec:
    """Rebuild the game from the swept row's own columns, as the reward function does.

    Off the row rather than from a constructor, for `games.rewards._Row.min_effort_spec`'s reason: the
    corpus on disk is the ground truth for the prompt the model actually read, so a spread computed
    from module constants would stop matching a corpus written before one of them moved.
    """
    return MinEffortSpec(
        game_id=str(row["game_id"]),
        n_levels=int(row["n_levels"]),
        benefit_per_level=float(row["benefit_per_level"]),
        cost_per_level=float(row["cost_per_level"]),
        team_size=int(row["team_size"]),
    )


def min_effort_spread_from_records(records: Sequence[dict[str, Any]]) -> MinEffortSpreadReport:
    """Score the group-mix reward against every level a policy sweep saved, one group per prompt.

    Per prompt rather than per training group, because a training group IS one prompt's completions:
    TRL's sampler yields each prompt's rollouts contiguously and `games.rewards` grades them as a
    block. The sweep's samples-per-prompt need not equal the training group size, so these figures
    estimate a rate rather than predict one batch -- the useful direction, since the question is
    whether this corpus supports a gradient at all.

    Keyed on the sweep's own `parsed` flag and its own recorded level rather than re-parsing, because
    the sweep is what the corpus was selected by: a second notion of an answer here would report a
    spread the training run will not see.
    """
    prompts: list[PromptLevelSpread] = []
    n_read = 0
    n_skipped = 0
    for record in records:
        if record.get(RECORD_KIND_KEY) != SWEEP_RECORD_KIND:
            n_skipped += 1
            continue
        if record.get(GRADING_KEY) != GRADING_MIN_EFFORT_GROUP_MIX:
            n_skipped += 1
            continue
        n_read += 1
        row = record[ROW_KEY]
        samples = record[SAMPLES_KEY]
        levels = tuple(
            int(sample[LEVEL_KEY])
            for sample in samples
            if sample.get(PARSED_KEY) and sample.get(LEVEL_KEY) is not None
        )
        spec = _spec_from_row(row)
        prompts.append(
            PromptLevelSpread(
                prompt_id=str(record["prompt_id"]),
                payoff_variant=str(row["payoff_variant"]),
                levels=levels,
                n_samples=len(samples),
                # A group with nothing parsed has no distribution to be graded against, so its spread
                # is zero by absence rather than by collapse. Counted as pure either way, because a
                # prompt every completion fails on carries no gradient about the game.
                spread=min_effort_group_reward_span(spec, levels) if levels else 0.0,
            )
        )
    return MinEffortSpreadReport(
        prompts=tuple(prompts), n_records_read=n_read, n_records_skipped=n_skipped
    )


def assert_min_effort_reward_can_train(report: MinEffortSpreadReport) -> None:
    """Refuse a corpus on which the minimum-effort reward provably has no gradient at all.

    Arithmetic, not a threshold: if every prompt's completions score identically then every
    within-group advantage is exactly zero on every step, and no configuration recovers it. Anything
    short of that is a judgement about how thin a gradient is worth paying for, which belongs in the
    report and in a person's hands -- the repo's rule is that a measurement is not a verdict.
    """
    if not report.prompts:
        raise ValueError(
            f"The trace held no minimum-effort sweep records, so nothing was measured. A report over "
            f"zero prompts is not evidence of anything; {report.n_records_skipped} records were "
            f"skipped as some other kind or grading (expected {SWEEP_RECORD_KIND!r} carrying "
            f"{GRADING_MIN_EFFORT_GROUP_MIX!r})."
        )
    if all(prompt.is_pure for prompt in report.prompts):
        raise ValueError(
            f"Every one of {report.n_prompts} prompts would score its whole group identically under "
            f"the minimum-effort reward, over {report.n_samples} saved completions "
            f"({report.n_answered} of them answering). Every GRPO advantage would therefore be exactly "
            f"zero on every step: the arm would train nothing while writing a full set of plausible "
            f"artifacts, and its flat curve would be indistinguishable from a real null. Realised "
            f"levels: {report.level_histogram}. A group split between the two extremes of the grid "
            f"does this at a cost-benefit ratio of 0.5 with one counterpart, so check the histogram "
            f"before the payoffs: the fix is a corpus whose levels are mixed rather than bimodal, or "
            f"the other cell of the knob grid."
        )


def level_reward_surface(spec: MinEffortSpec, levels: Sequence[int]) -> dict[int, float]:
    """Return what each level would have paid against this realised distribution.

    Printed beside the spread because it is the diagnostic when the spread is thin: a flat surface says
    the group's own levels made every answer equal, which is a corpus property, while a steep surface
    with a thin spread says the group simply did not disagree. Those two call for opposite responses
    and the spread alone cannot tell them apart.
    """
    return {
        level: min_effort_group_reward(spec, own_level=level, counterpart_levels=levels)
        for level in spec.levels
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Report what the minimum-effort reward would pay on one sweep trace's own levels."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        required=True,
        type=Path,
        help="Sweep trace JSONL written by games.select_prompts.",
    )
    args = parser.parse_args(argv)
    report = min_effort_spread_from_records(read_trace(args.trace))
    logger.info("%s\n%s", args.trace, report.render())
    assert_min_effort_reward_can_train(report)


if __name__ == "__main__":
    main()
