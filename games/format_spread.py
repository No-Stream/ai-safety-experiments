"""What the format-only reward would have paid on a baseline sweep's own completions.

The format-only placebo has one real design risk and it is not a plumbing risk: a reward over answer
*shape* can have no within-group spread. Wave 2 runs uncapped completion budgets so truncation is
rare and parse failure sits near a percent, and at that rate almost every completion in a group
answers, so if they also all answer in the same shape then every reward in the group is identical,
every GRPO advantage is zero, and the arm trains nothing while writing a full set of plausible
artifacts and a flat curve indistinguishable from a real null. That is the same shape as the
`self`-graded constant-sum arm the registry already refuses.

The thing worth saying about it: **this is answerable without spending a run.** The baseline sweep
that selects the corpus already saved every completion it sampled, so the rubric can be scored
against real 2B output at zero GPU cost, before the meter starts. That turns "which resolution of
the format-only gradient problem do we take" from an argument into a measurement -- and if the
answer is that the spread is too thin, the ready lever is a graded terseness component, deliberately
left out of `games.format_rubric` until data asks for it.

Read as a report, not a gate. It prints per-component credit rates over the completions that
answered, the per-prompt reward spread with and without the parse penalty (which separates "the
gradient is answer shape" from "the gradient is parse failures", i.e. the interesting version of the
arm from the degenerate one), and the fraction of prompts that would arrive as a pure group. The one
hard refusal is arithmetic rather than judgement: a corpus where every prompt's spread is exactly
zero cannot train, whatever anyone thinks of the design.

Every count is reported with what it was taken over. A trace holds the policy sweep's records beside
the frozen opponent's and a meta record, so "1.7% of samples did not parse" needs to say which
samples, and a zero component rate needs to say how many completions it was zero across.

    uv run python -m games.format_spread --trace artifacts/games/select/<arm>/trace-*.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.format_rubric import COMPONENT_NAMES, format_component_scores, format_reward
from games.rewards import DEFAULT_PARSE_PENALTY

# Taken from the writer rather than restated, which is worth the cost: `games.select_prompts` imports
# torch and transformers, so this module is a ~10 s import where the rubric alone is milliseconds.
# Fine for a CLI over a JSONL, and the reason this module must NOT be pulled onto a stage plan's path
# -- `games.arms` and `games.plans` are cheap so that `--print-plan` stays the cheap check to run
# before the meter starts, and `test_games_arms.TestTheRegistryStaysCheapToImport` is what holds that.
from games.select_prompts import SWEEP_RECORD_KIND

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

RECORD_KIND_KEY = "record_kind"
SAMPLES_KEY = "samples"
ROW_KEY = "row"
VISIBLE_TEXT_KEY = "visible_text"


@dataclass(frozen=True)
class ComponentCredit:
    """How often one rubric component was credited, and over how many completions."""

    name: str
    credited: int
    examined: int

    @property
    def rate(self) -> float:
        """Share of examined completions credited, or 0.0 when nothing was examined."""
        if not self.examined:
            return 0.0
        return self.credited / self.examined

    @property
    def is_dead_here(self) -> bool:
        """Report whether this component was constant across every completion examined.

        Constant either way is dead for training purposes: always credited and never credited both
        contribute the same number to every reward in every group, so neither can move an advantage.
        Named `_here` because it is a property of this corpus and this policy, not of the rubric --
        `games.format_rubric.assert_components_are_live` is what checks the rubric itself.
        """
        return self.examined > 0 and self.credited in (0, self.examined)


@dataclass(frozen=True)
class PromptFormatSpread:
    """One prompt's would-be rewards under the format-only rubric, and the spread across them."""

    prompt_id: str
    rewards: tuple[float, ...]
    rewards_among_answered: tuple[float, ...]

    @property
    def n_samples(self) -> int:
        """Completions this prompt was sampled for."""
        return len(self.rewards)

    @property
    def n_answered(self) -> int:
        """Completions that produced a parseable answer."""
        return len(self.rewards_among_answered)

    @property
    def spread(self) -> float:
        """Widest reward gap in this prompt's group, parse penalties included."""
        if not self.rewards:
            return 0.0
        return max(self.rewards) - min(self.rewards)

    @property
    def spread_among_answered(self) -> float:
        """Widest gap among the completions that produced an answer, so shape alone."""
        if not self.rewards_among_answered:
            return 0.0
        return max(self.rewards_among_answered) - min(self.rewards_among_answered)


@dataclass(frozen=True)
class FormatSpreadReport:
    """The whole measurement: per-prompt spreads, per-component rates, and what was skipped."""

    prompts: tuple[PromptFormatSpread, ...]
    components: tuple[ComponentCredit, ...]
    n_records_read: int
    n_records_skipped: int
    parse_penalty: float

    @property
    def n_prompts(self) -> int:
        """Prompts measured, the group count every purity fraction is taken over."""
        return len(self.prompts)

    @property
    def n_samples(self) -> int:
        """Completions examined across every prompt."""
        return sum(prompt.n_samples for prompt in self.prompts)

    @property
    def n_answered(self) -> int:
        """Completions that answered, the denominator every component rate is taken over."""
        return sum(prompt.n_answered for prompt in self.prompts)

    @property
    def parse_failure_rate(self) -> float:
        """Share of examined completions that produced no answer and would take the penalty."""
        if not self.n_samples:
            return 0.0
        return (self.n_samples - self.n_answered) / self.n_samples

    @property
    def pure_prompt_fraction(self) -> float:
        """Share of prompts whose whole group would score identically: the predicted purity."""
        if not self.prompts:
            return 0.0
        return sum(prompt.spread == 0.0 for prompt in self.prompts) / self.n_prompts

    @property
    def pure_among_answered_fraction(self) -> float:
        """The same share with parse failures excluded, so the answer-shape gradient on its own."""
        if not self.prompts:
            return 0.0
        return sum(prompt.spread_among_answered == 0.0 for prompt in self.prompts) / self.n_prompts

    @property
    def dead_components(self) -> tuple[str, ...]:
        """Components constant across every completion examined, so inert on this corpus."""
        return tuple(item.name for item in self.components if item.is_dead_here)

    @property
    def mean_spread(self) -> float:
        """Mean per-prompt reward spread, parse penalties included."""
        if not self.prompts:
            return 0.0
        return sum(prompt.spread for prompt in self.prompts) / self.n_prompts

    @property
    def mean_spread_among_answered(self) -> float:
        """Mean per-prompt reward spread among answering completions, so answer shape alone."""
        if not self.prompts:
            return 0.0
        return sum(prompt.spread_among_answered for prompt in self.prompts) / self.n_prompts

    def render(self) -> str:
        """Render the report, every rate beside the denominator it was taken over."""
        lines = [
            (
                f"format-only reward spread over {self.n_prompts} prompts, {self.n_samples} "
                f"completions ({self.n_records_read} sweep records read, "
                f"{self.n_records_skipped} other records skipped)"
            ),
            (
                f"  answered {self.n_answered}/{self.n_samples} "
                f"(parse failure rate {self.parse_failure_rate:.4f}; unanswered score "
                f"{self.parse_penalty})"
            ),
            (
                f"  per-prompt spread          mean {self.mean_spread:.4f}, "
                f"pure groups {self.pure_prompt_fraction:.4f} of {self.n_prompts}"
            ),
            (
                f"  per-prompt spread, answers mean {self.mean_spread_among_answered:.4f}, "
                f"pure groups {self.pure_among_answered_fraction:.4f} of {self.n_prompts}"
            ),
            "  component credit rates over the completions that answered:",
        ]
        lines += [
            (
                f"    {item.name:28s} {item.rate:.4f} ({item.credited}/{item.examined})"
                f"{'  DEAD HERE' if item.is_dead_here else ''}"
            )
            for item in self.components
        ]
        if self.dead_components:
            lines.append(
                f"  {len(self.dead_components)} component(s) constant across every completion "
                f"examined, so they contribute the same number to every reward in every group and "
                f"cannot move an advantage on this corpus: {list(self.dead_components)}"
            )
        return "\n".join(lines)


def _sample_reward(sample: dict[str, Any], *, label_a: str, label_b: str) -> float | None:
    """Score one saved sample, returning None where the sweep recorded no answer.

    Keyed on the sweep's own `parsed` flag rather than re-deriving it, because the sweep is what the
    corpus was selected by: re-parsing here with a different notion of an answer would report a
    spread the training run will not see.
    """
    if not sample.get("parsed"):
        return None
    return format_reward(str(sample[VISIBLE_TEXT_KEY]), label_a=label_a, label_b=label_b)


def format_spread_from_records(
    records: Sequence[dict[str, Any]], *, parse_penalty: float = DEFAULT_PARSE_PENALTY
) -> FormatSpreadReport:
    """Score the rubric against every completion a policy sweep saved, one group per prompt.

    Per prompt rather than per training group because a training group IS one prompt's completions:
    TRL's sampler yields each prompt's G rollouts contiguously and `games.rewards` grades them as a
    block. The sweep's samples-per-prompt need not equal the training group size, so the purity
    figures are an estimate of a rate rather than a prediction of one batch -- which is the useful
    direction, since the question is whether this corpus supports a gradient at all.
    """
    prompts: list[PromptFormatSpread] = []
    credited = dict.fromkeys(COMPONENT_NAMES, 0)
    examined = 0
    n_read = 0
    n_skipped = 0
    for record in records:
        if record.get(RECORD_KIND_KEY) != SWEEP_RECORD_KIND:
            n_skipped += 1
            continue
        n_read += 1
        row = record[ROW_KEY]
        labels = {"label_a": str(row["label_a"]), "label_b": str(row["label_b"])}
        rewards: list[float] = []
        answered: list[float] = []
        for sample in record[SAMPLES_KEY]:
            scored = _sample_reward(sample, **labels)
            if scored is None:
                rewards.append(parse_penalty)
                continue
            rewards.append(scored)
            answered.append(scored)
            examined += 1
            for name, value in format_component_scores(
                str(sample[VISIBLE_TEXT_KEY]), **labels
            ).items():
                credited[name] += int(value)
        prompts.append(
            PromptFormatSpread(
                prompt_id=str(record["prompt_id"]),
                rewards=tuple(rewards),
                rewards_among_answered=tuple(answered),
            )
        )
    return FormatSpreadReport(
        prompts=tuple(prompts),
        components=tuple(
            ComponentCredit(name=name, credited=credited[name], examined=examined)
            for name in COMPONENT_NAMES
        ),
        n_records_read=n_read,
        n_records_skipped=n_skipped,
        parse_penalty=parse_penalty,
    )


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Read a sweep trace JSONL, the shape `games.select_prompts.append_sweep_records` writes."""
    records: list[dict[str, Any]] = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"Sweep trace {path} holds no records.")
    return records


def assert_format_reward_can_train(report: FormatSpreadReport) -> None:
    """Refuse a corpus on which the format-only reward provably has no gradient at all.

    Arithmetic, not a threshold: if every prompt's completions score identically then every
    within-group advantage is exactly zero on every step, and the arm cannot learn from any of them
    regardless of how the run is configured. Anything short of that is a judgement about how thin a
    gradient is worth paying for, which belongs in the report and in a person's hands -- the repo's
    rule is that a measurement is not a verdict.
    """
    if not report.prompts:
        raise ValueError(
            "The trace held no policy-sweep records, so nothing was measured. A report over zero "
            f"prompts is not evidence of anything; {report.n_records_skipped} records were skipped "
            f"as some other kind (expected {SWEEP_RECORD_KIND!r})."
        )
    if all(prompt.spread == 0.0 for prompt in report.prompts):
        raise ValueError(
            f"Every one of {report.n_prompts} prompts would score its whole group identically under "
            f"the format-only rubric, over {report.n_samples} saved completions "
            f"({report.n_answered} of them answering). Every GRPO advantage would therefore be "
            f"exactly zero on every step: the arm would train nothing while writing a full set of "
            f"plausible artifacts, and its flat curve would be indistinguishable from a real null. "
            f"Constant components on this corpus: {list(report.dead_components)}. Add a component "
            f"that this policy's output actually varies on -- graded terseness is the one held back "
            f"from games.format_rubric for exactly this case -- or sweep a policy whose answer shape "
            f"is less uniform."
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Report what the format-only reward would pay on one sweep trace's completions."""
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
    parser.add_argument(
        "--parse-penalty",
        type=float,
        default=DEFAULT_PARSE_PENALTY,
        help="The reward an unanswered completion takes; must match the training run's.",
    )
    args = parser.parse_args(argv)
    report = format_spread_from_records(read_trace(args.trace), parse_penalty=args.parse_penalty)
    logger.info("%s\n%s", args.trace, report.render())
    assert_format_reward_can_train(report)


if __name__ == "__main__":
    main()
