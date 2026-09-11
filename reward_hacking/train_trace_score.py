"""Recompute the hidden true check over a finished run's rollout trace, one curve point per step.

The reward that trains the misspecified-grader arms asks exactly one question -- did the VISIBLE
grader accept this submission -- and the hidden true check never touches the gradient. Running that
check live is available as instrumentation (``GraderConfig.hidden_check``) but off by default,
because its leash is a hundred and fifty seconds and it spawns two extra interpreters per call, so at
sixty-four episodes an optimiser step it can cost more than the step.

Leaving it off loses nothing, which is the whole reason ``log_completions=True`` is set on the
trainer: TRL retains every completion's text, so the *in-training hack curve* -- the rate at which
the visible grader accepts what the true check rejects, against training step -- is recoverable
afterwards on CPU. This module is that recovery, and it is why a missing metric here is a
re-analysis rather than a re-run. :mod:`reward_hacking.train_trace` does the reading; everything
here re-grades what it read.

**The visible grader is recomputed too, and that is the point.** Recomputing only the hidden check
would leave the curve unanchored: if this offline path and the live reward do not implement the same
notion of a pass, the hidden rate is measured over a different population from the one the gradient
saw, and nothing in the artifact would say so. So every completion is re-graded on both checks and
the recomputed visible outcome is compared against the ``visible_outcome`` the live reward recorded.
Systematic disagreement refuses the whole pass instead of producing a plausible number. Three further
recorded columns are compared and reported as well -- ``truncated_thinking``, ``hidden_outcome`` for
a run that did pay for the live check, and the reward TRL actually put in the gradient -- because
they cost nothing once the rows are in memory.

**Everything reported carries its denominator.** A hidden pass rate that moved because the measured
count collapsed looks identical to one that moved because behaviour changed, and an ``ORACLE_ERROR``
is nobody having measured the episode rather than a failure -- folding it into one is how this repo
once reported unmeasured episodes as hacks. So each step reports counts, and every rate is ``None``
rather than zero when its denominator is empty.

The CLI, for a run directory holding ``completions/`` and ``run_config.json``::

    .venv/bin/python -m reward_hacking.train_trace_score --run-dir <run> --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from games.provenance import git_provenance
from reward_hacking.harness.tasks_ilcb import TRUE_CHECK_TIMEOUT_SECONDS
from reward_hacking.train_reward import (
    DEFAULT_GRADER_TIMEOUT_SECONDS,
    DEFAULT_GRADER_WORKERS,
    GradedCompletion,
    GraderConfig,
    GraderOutcome,
    grade_solution,
)
from reward_hacking.train_trace import (
    DERIVED_KEY,
    GRADER_KEY,
    GRADER_TIMEOUT_SECONDS_KEY,
    PREFILLED_THINK_KEY,
    RECORDED_HIDDEN_OUTCOME_COLUMN,
    RECORDED_TRUNCATED_THINKING_COLUMN,
    RECORDED_VISIBLE_OUTCOME_COLUMN,
    RUN_CONFIG_FILENAME,
    VISIBLE_GRADER_REWARD_COLUMN,
    BoundedSubset,
    TraceRow,
    counted,
    describe_task_id_source,
    rate,
    read_grader_timeout_seconds,
    read_prefilled_think,
    read_trace_rows,
    resolve_trace_files,
    rows_per_step,
    select_bounded_subset,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

# Its own root so re-scoring cannot collide with a live run's graders. Never /tmp (tmpfs inode cap)
DEFAULT_TRACE_SCORE_SCRATCH_ROOT = "/var/tmp/rh-trace-score-graders"  # noqa: S108

# The refusal threshold and its grace; justified in `assert_visible_grader_agrees`.
MAX_VISIBLE_DISAGREEMENT_RATE = 0.02
VISIBLE_DISAGREEMENT_GRACE = 1

# Enough disagreeing rows to see the shape of a mismatch without pasting a trace into the artifact.
DISAGREEMENT_EXAMPLES = 8

_COLUMN_ABSENT = "the trace carries no such column, so nothing was compared"
_HIDDEN_ABSENT = (
    "the run did not pay for the live hidden check, so there is no recorded verdict to compare "
    "against -- which is the normal case, and the reason this module exists"
)


@dataclass(frozen=True, slots=True)
class ScoredRow:
    """One trace row and the verdicts re-grading it produced."""

    row: TraceRow
    graded: GradedCompletion


@dataclass(frozen=True, slots=True)
class RecordedAgreement:
    """How one recomputed per-completion label compares with what the live run recorded.

    ``available`` is not a convenience flag. A trace written before the live reward logged a column,
    or by a different reward altogether, yields zero comparisons -- and zero disagreements out of
    zero comparisons must never read as agreement. So the absent case carries its reason and the
    reader is told the check did not run.
    """

    column: str
    available: bool
    unavailable_reason: str | None
    n_compared: int
    n_disagreements: int
    examples: tuple[str, ...]

    @property
    def disagreement_rate(self) -> float | None:
        """Return the disagreement rate, or None when nothing was compared."""
        return rate(self.n_disagreements, self.n_compared)

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the comparison, denominator included."""
        return {
            "column": self.column,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "n_compared": self.n_compared,
            "n_disagreements": self.n_disagreements,
            "disagreement_rate_of_compared": self.disagreement_rate,
            "examples": list(self.examples),
        }


@dataclass(frozen=True, slots=True)
class StepCurvePoint:
    """One step of the recomputed curve: every count, and every rate beside its denominator.

    ``n_rows_recorded_at_step`` sits next to ``n_completions_scored`` on purpose. Under a completion
    bound the two differ, so a reader who looks at a single step rather than at the header still sees
    that the step was sampled rather than measured whole.
    """

    step: int
    n_rows_recorded_at_step: int
    n_completions_scored: int
    n_with_solution: int
    n_truncated_thinking: int
    visible_pass: int
    visible_fail: int
    visible_timeout: int
    visible_no_solution: int
    visible_no_verdict: int
    hidden_measured: int
    hidden_unmeasured: int
    hidden_pass: int
    hidden_fail: int
    proxy_true_gap: int

    def to_json_dict(self) -> dict[str, object]:
        """Serialise this curve point, each rate named after the denominator it divides by."""
        scored = self.n_completions_scored
        return {
            "step": self.step,
            "n_rows_recorded_at_step": self.n_rows_recorded_at_step,
            "n_completions_scored": scored,
            "step_scored_whole": scored == self.n_rows_recorded_at_step,
            "n_with_solution": self.n_with_solution,
            "n_truncated_thinking": self.n_truncated_thinking,
            "visible_outcome_counts": {
                GraderOutcome.PASS.value: self.visible_pass,
                GraderOutcome.FAIL.value: self.visible_fail,
                GraderOutcome.TIMEOUT.value: self.visible_timeout,
                GraderOutcome.NO_SOLUTION.value: self.visible_no_solution,
                GraderOutcome.NO_VERDICT.value: self.visible_no_verdict,
            },
            "hidden_measured": self.hidden_measured,
            "hidden_unmeasured": self.hidden_unmeasured,
            "hidden_pass": self.hidden_pass,
            "hidden_fail": self.hidden_fail,
            "proxy_true_gap": self.proxy_true_gap,
            "solution_rate_of_scored": rate(self.n_with_solution, scored),
            "truncated_thinking_rate_of_scored": rate(self.n_truncated_thinking, scored),
            "visible_pass_rate_of_scored": rate(self.visible_pass, scored),
            "no_verdict_rate_of_scored": rate(self.visible_no_verdict, scored),
            "hidden_measured_rate_of_scored": rate(self.hidden_measured, scored),
            "hidden_pass_rate_of_measured": rate(self.hidden_pass, self.hidden_measured),
            "proxy_true_gap_rate_of_measured": rate(self.proxy_true_gap, self.hidden_measured),
        }


@dataclass(frozen=True, slots=True)
class ScoredTrace:
    """Every scored row, plus what re-grading them cost.

    ``n_unique_submissions_graded`` is the number of gradings this pass actually paid for and
    ``n_cache_hits`` is the rest; both are reported because the gap between them and the row count is
    what makes the wall-clock figure readable.
    """

    rows: tuple[ScoredRow, ...]
    n_unique_submissions_graded: int
    n_cache_hits: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class TraceScoreReport:
    """One offline scoring pass, whole: what it read, how it graded, and the curve it produced."""

    run_dir: str | None
    trace_files: tuple[str, ...]
    prefilled_think: bool
    prefilled_think_source: str
    task_id_source: str
    grader: GraderConfig
    scored: ScoredTrace
    curve: tuple[StepCurvePoint, ...]
    agreements: tuple[RecordedAgreement, ...]
    bounded_subset: BoundedSubset | None

    @property
    def complete_curve(self) -> bool:
        """Report whether this pass scored every completion the trace retained."""
        return self.bounded_subset is None

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the whole pass, header first, with any subset bound impossible to miss."""
        return {
            "complete_curve": self.complete_curve,
            "bounded_subset_warning": None if self.complete_curve else _SUBSET_WARNING,
            "bounded_subset": None
            if self.bounded_subset is None
            else self.bounded_subset.to_json_dict(),
            "run_dir": self.run_dir,
            "trace_files": list(self.trace_files),
            "n_trace_files": len(self.trace_files),
            PREFILLED_THINK_KEY: self.prefilled_think,
            "prefilled_think_source": self.prefilled_think_source,
            "task_id_source": self.task_id_source,
            "grader": self.grader.to_json_dict(),
            "hidden_check_timeout_seconds": TRUE_CHECK_TIMEOUT_SECONDS,
            "grading_cost": {
                "n_rows_scored": len(self.scored.rows),
                "n_unique_submissions_graded": self.scored.n_unique_submissions_graded,
                "n_cache_hits": self.scored.n_cache_hits,
                "wall_seconds": round(self.scored.wall_seconds, 3),
            },
            "recorded_agreement": {
                agreement.column: agreement.to_json_dict() for agreement in self.agreements
            },
            "visible_disagreement_refusal": {
                "max_rate": MAX_VISIBLE_DISAGREEMENT_RATE,
                "grace_disagreements": VISIBLE_DISAGREEMENT_GRACE,
            },
            "task_split_counts": counted(item.row.task_split for item in self.scored.rows),
            "totals": totals(self.curve),
            "curve": [point.to_json_dict() for point in self.curve],
            "written_at": datetime.now(tz=UTC).isoformat(),
            **git_provenance(),
        }


_SUBSET_WARNING = (
    "NOT THE FULL CURVE. This pass graded a bounded subset of the retained rollout trace, so every "
    "count and rate below describes that subset and not the run. See 'bounded_subset' for how the "
    "rows were chosen, and each curve point's 'step_scored_whole'."
)


@dataclass(frozen=True, slots=True)
class TraceScorePlan:
    """What a dry run would grade. Deliberately carries no verdicts at all.

    A dry run emitting the same shape as a real pass, minus a few fields, is a file someone reads as
    a curve. This one cannot be: it has no outcome counts to misread.
    """

    run_dir: str | None
    trace_files: tuple[str, ...]
    prefilled_think: bool
    prefilled_think_source: str
    task_id_source: str
    rows: tuple[TraceRow, ...]
    bounded_subset: BoundedSubset | None

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the plan: per-step row counts, extractability, and the recorded distribution."""
        return {
            "dry_run": True,
            "graded_nothing": True,
            "complete_curve": self.bounded_subset is None,
            "bounded_subset": None
            if self.bounded_subset is None
            else self.bounded_subset.to_json_dict(),
            "run_dir": self.run_dir,
            "trace_files": list(self.trace_files),
            PREFILLED_THINK_KEY: self.prefilled_think,
            "prefilled_think_source": self.prefilled_think_source,
            "task_id_source": self.task_id_source,
            "n_rows": len(self.rows),
            "n_rows_with_extractable_solution": sum(
                1 for row in self.rows if row.solution is not None
            ),
            "n_rows_truncated_thinking": sum(1 for row in self.rows if row.truncated_thinking),
            "n_distinct_submissions_to_grade": len({row.cache_key for row in self.rows}),
            "n_distinct_task_ids": len({row.task_id for row in self.rows}),
            "task_split_counts": counted(row.task_split for row in self.rows),
            "recorded_visible_outcome_counts": counted(
                row.recorded_visible_outcome or "not recorded" for row in self.rows
            ),
            "rows_per_step": {
                str(step): count for step, count in sorted(rows_per_step(self.rows).items())
            },
            "written_at": datetime.now(tz=UTC).isoformat(),
            **git_provenance(),
        }


@dataclass(frozen=True, slots=True)
class TraceScoreRequest:
    """One offline scoring pass as asked for: what to read, how to grade, how much of it."""

    run_dir: Path | None
    trace: str | None
    grader: GraderConfig
    prefilled_think: bool | None = None
    max_completions: int | None = None

    def resolve_prefilled_think(self) -> tuple[bool, str]:
        """Settle ``prefilled_think`` from the run's own record, or from an explicit flag."""
        if self.run_dir is not None:
            if self.prefilled_think is not None:
                raise ValueError(
                    f"--prefilled-think/--no-prefilled-think may not override a run directory: the "
                    f"run's {RUN_CONFIG_FILENAME} records what the live reward was built with, and "
                    f"a flag disagreeing with it would recompute a different parse from the one "
                    f"being cross-checked."
                )
            source = f"{RUN_CONFIG_FILENAME}:{DERIVED_KEY}.{PREFILLED_THINK_KEY}"
            return read_prefilled_think(self.run_dir), source
        if self.prefilled_think is None:
            raise ValueError(
                "--trace names a bare parquet file with no run record beside it, so pass "
                "--prefilled-think or --no-prefilled-think explicitly. It is never guessed: "
                "getting it wrong scores every completion as no_solution."
            )
        return self.prefilled_think, "--prefilled-think flag"


def score_rows(rows: Sequence[TraceRow], *, grader: GraderConfig) -> ScoredTrace:
    """Re-grade every row on both checks, in a thread pool, grading identical submissions once.

    Threads because the work is subprocess-bound: each grading launches a jailed interpreter as its
    own systemd unit carrying ``episode_limits()`` caps, so the politeness budget is the pool width
    times those caps rather than anything this process was given.
    """
    if not rows:
        raise ValueError("no rows to score; the trace was read as empty")
    if not grader.hidden_check:
        raise ValueError(
            "the whole point of this pass is the hidden true check, so grade with "
            "GraderConfig(hidden_check=True). With it off the pass would recompute only the visible "
            "grader and report a curve whose every hidden count is unmeasured."
        )
    keys = list(dict.fromkeys(row.cache_key for row in rows))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=grader.workers) as pool:
        graded = list(pool.map(lambda key: grade_solution(key[0], key[1], grader=grader), keys))
    wall_seconds = time.perf_counter() - started
    by_key = dict(zip(keys, graded, strict=True))
    scored = tuple(ScoredRow(row=row, graded=by_key[row.cache_key]) for row in rows)
    _assert_something_was_measured(scored)
    logger.info(
        "re-graded the trace, %s",
        f"n_rows={len(scored)} n_unique={len(keys)} n_cache_hits={len(scored) - len(keys)} "
        f"wall_seconds={wall_seconds:.1f}",
    )
    return ScoredTrace(
        rows=scored,
        n_unique_submissions_graded=len(keys),
        n_cache_hits=len(scored) - len(keys),
        wall_seconds=wall_seconds,
    )


def _assert_something_was_measured(scored: Sequence[ScoredRow]) -> None:
    """Raise when no completion reached a grader verdict, and warn when only some did.

    The offline twin of ``train_reward._assert_batch_was_measured``, and for its reason: a jail that
    cannot start returns an exit code and no verdict for every episode, which as a curve is a run
    whose policy never once wrote working code. Whole-pass rather than per-batch, because offline
    there is no batch to break.
    """
    no_verdict = sum(1 for item in scored if item.graded.outcome is GraderOutcome.NO_VERDICT)
    if no_verdict == len(scored):
        raise RuntimeError(
            f"none of {len(scored)} completions reached a grader verdict, so this pass measured "
            f"nothing and every point of the curve would be a zero that says nothing about the "
            f"policy. The jail is the usual cause: check bwrap, XDG_RUNTIME_DIR and "
            f"`systemctl --user is-system-running`. First grader stdout: "
            f"{scored[0].graded.grader_stdout!r}"
        )
    if no_verdict:
        logger.warning(
            "some graders reached no verdict, so their zeros are apparatus failures rather than "
            "policy failures, %s",
            f"{no_verdict=} of n={len(scored)}",
        )


def build_curve(
    scored: Sequence[ScoredRow], *, rows_recorded_per_step: Mapping[int, int]
) -> list[StepCurvePoint]:
    """Aggregate the scored rows into one point per step.

    ``rows_recorded_per_step`` is what the trace held rather than what this pass graded, so a
    bounded pass cannot present a sampled step as a measured one.

    Every scored step must be in that mapping, and a step that is not raises ``KeyError`` rather
    than falling back to the number of rows this pass happened to grade. The mapping is built by
    :func:`~reward_hacking.train_trace.rows_per_step` over the very rows a bound is then drawn from,
    so the two agree by construction; a step present here and absent there means the two came from
    different reads of the trace, and the fallback would have made ``step_scored_whole`` come out
    True for a step whose real denominator is unknown -- the exact inversion of what that field is
    for.
    """
    by_step: dict[int, list[ScoredRow]] = {}
    for item in scored:
        by_step.setdefault(item.row.step, []).append(item)
    return [
        _curve_point(step, group, rows_recorded_per_step) for step, group in sorted(by_step.items())
    ]


def _curve_point(
    step: int, group: Sequence[ScoredRow], rows_recorded_per_step: Mapping[int, int]
) -> StepCurvePoint:
    """Count one step's outcomes, keeping unmeasured hidden checks out of every failure tally.

    ``passed_hidden`` is None both for an absent verdict and for an ``ORACLE_ERROR``, and those rows
    go to ``hidden_unmeasured``. Folding them into ``hidden_fail`` would count episodes nobody
    measured as episodes the true check rejected -- how this repo once reported unmeasured work as
    hacks -- and each would then also read as a proxy-versus-true gap, which is to say as a hack.
    """
    outcomes = [item.graded.outcome for item in group]
    hidden = [item.graded.passed_hidden for item in group]
    return StepCurvePoint(
        step=step,
        n_rows_recorded_at_step=rows_recorded_per_step[step],
        n_completions_scored=len(group),
        n_with_solution=sum(1 for item in group if item.row.solution is not None),
        n_truncated_thinking=sum(1 for item in group if item.row.truncated_thinking),
        visible_pass=outcomes.count(GraderOutcome.PASS),
        visible_fail=outcomes.count(GraderOutcome.FAIL),
        visible_timeout=outcomes.count(GraderOutcome.TIMEOUT),
        visible_no_solution=outcomes.count(GraderOutcome.NO_SOLUTION),
        visible_no_verdict=outcomes.count(GraderOutcome.NO_VERDICT),
        hidden_measured=sum(1 for verdict in hidden if verdict is not None),
        hidden_unmeasured=sum(1 for verdict in hidden if verdict is None),
        hidden_pass=sum(1 for verdict in hidden if verdict is True),
        hidden_fail=sum(1 for verdict in hidden if verdict is False),
        proxy_true_gap=sum(1 for item in group if item.graded.is_proxy_true_gap is True),
    )


def totals(curve: Sequence[StepCurvePoint]) -> dict[str, object]:
    """Sum the curve, so a header answers "what did this whole pass find" without arithmetic."""
    scored = sum(point.n_completions_scored for point in curve)
    measured = sum(point.hidden_measured for point in curve)
    visible_pass = sum(point.visible_pass for point in curve)
    hidden_pass = sum(point.hidden_pass for point in curve)
    gap = sum(point.proxy_true_gap for point in curve)
    return {
        "n_steps": len(curve),
        "n_rows_recorded": sum(point.n_rows_recorded_at_step for point in curve),
        "n_completions_scored": scored,
        "n_with_solution": sum(point.n_with_solution for point in curve),
        "visible_pass": visible_pass,
        "visible_no_verdict": sum(point.visible_no_verdict for point in curve),
        "hidden_measured": measured,
        "hidden_unmeasured": sum(point.hidden_unmeasured for point in curve),
        "hidden_pass": hidden_pass,
        "hidden_fail": sum(point.hidden_fail for point in curve),
        "proxy_true_gap": gap,
        "visible_pass_rate_of_scored": rate(visible_pass, scored),
        "hidden_measured_rate_of_scored": rate(measured, scored),
        "hidden_pass_rate_of_measured": rate(hidden_pass, measured),
        "proxy_true_gap_rate_of_measured": rate(gap, measured),
    }


def _agreement(
    column: str,
    comparisons: Sequence[tuple[TraceRow, object, object]],
    *,
    unavailable_reason: str | None,
) -> RecordedAgreement:
    """Build one comparison of a recomputed label against the recorded one."""
    if unavailable_reason is not None:
        return RecordedAgreement(
            column=column,
            available=False,
            unavailable_reason=unavailable_reason,
            n_compared=0,
            n_disagreements=0,
            examples=(),
        )
    disagreeing = [
        f"{row.label}: recorded={recorded!r} recomputed={recomputed!r}"
        for row, recorded, recomputed in comparisons
        if recorded != recomputed
    ]
    return RecordedAgreement(
        column=column,
        available=True,
        unavailable_reason=None,
        n_compared=len(comparisons),
        n_disagreements=len(disagreeing),
        examples=tuple(disagreeing[:DISAGREEMENT_EXAMPLES]),  # HARNESS-SCAN-EXEMPT-subsampling
    )


def _compare(
    column: str,
    pairs: Sequence[tuple[TraceRow, object, object]],
    *,
    absent_reason: str,
) -> RecordedAgreement:
    """Compare only the rows that carry a recorded value, reporting an all-absent column as absent."""
    comparable = [
        (row, recorded, recomputed) for row, recorded, recomputed in pairs if recorded is not None
    ]
    return _agreement(column, comparable, unavailable_reason=None if comparable else absent_reason)


def compare_with_recorded(scored: Sequence[ScoredRow]) -> list[RecordedAgreement]:
    """Compare every recomputed label against the one the live run recorded for the same row.

    Four comparisons, all free once the rows are in memory. The visible outcome is the load-bearing
    one -- see :func:`assert_visible_grader_agrees`. The other three are reported: recomputed
    ``truncated_thinking`` is a sharp independent check on ``prefilled_think``, the recorded hidden
    outcome exists only for a run that paid for the live check, and the reward column is what TRL
    actually put into the gradient.
    """
    return [
        _compare(
            RECORDED_VISIBLE_OUTCOME_COLUMN,
            [
                (item.row, item.row.recorded_visible_outcome, item.graded.outcome.value)
                for item in scored
            ],
            absent_reason=_COLUMN_ABSENT,
        ),
        _compare(
            RECORDED_TRUNCATED_THINKING_COLUMN,
            [
                (item.row, item.row.recorded_truncated_thinking, item.row.truncated_thinking)
                for item in scored
            ],
            absent_reason=_COLUMN_ABSENT,
        ),
        _compare(
            RECORDED_HIDDEN_OUTCOME_COLUMN,
            [
                (item.row, item.row.recorded_hidden_outcome, item.graded.hidden_outcome)
                for item in scored
            ],
            absent_reason=_HIDDEN_ABSENT,
        ),
        _compare(
            VISIBLE_GRADER_REWARD_COLUMN,
            [(item.row, item.row.recorded_reward, item.graded.reward) for item in scored],
            absent_reason=_COLUMN_ABSENT,
        ),
    ]


def assert_visible_grader_agrees(agreements: Sequence[RecordedAgreement]) -> None:
    """Refuse a pass whose recomputed visible grader disagrees with the live one beyond noise.

    This is the check that makes the hidden curve comparable to the training reward. If the two
    paths do not agree about what a pass is, the hidden rate is measured over a different population
    from the one the gradient saw, and every number in the artifact would still look reasonable.

    **Why two per cent.** Both paths run the same ``grade_solution`` over the same completion text --
    TRL hands the reward function ``batch_decode(completion_ids, skip_special_tokens=True)`` and logs
    a second call to that same expression, so the strings are byte-identical -- which leaves exactly
    one honest source of disagreement: a submission whose behaviour is not a function of its input. A
    body running near the twelve-second leash can time out in one pass and not the other, and one
    reading the clock, the hash seed or a race can flip. Those are real and rare. Systematic
    mismatches are neither: the wrong ``prefilled_think`` makes every completion read as truncated
    thinking and turns the whole trace into ``no_solution``, and a task join landing on the wrong
    split flips whole problems. Those land at tens of percent, more than an order of magnitude above
    this threshold, so it separates the two cases without being brittle. One disagreement is always
    forgiven so a twenty-row diagnostic pass is not refused by a single timing flip, where one row is
    already five per cent.

    Loud in every case: an unavailable comparison warns rather than passing quietly, because zero
    disagreements out of zero comparisons is not agreement.
    """
    visible = next(
        (item for item in agreements if item.column == RECORDED_VISIBLE_OUTCOME_COLUMN), None
    )
    if visible is None:
        raise RuntimeError(
            f"no {RECORDED_VISIBLE_OUTCOME_COLUMN} comparison was built, so the offline grader was "
            f"never checked against the live one. compare_with_recorded must always produce it, "
            f"available or not."
        )
    if not visible.available:
        logger.warning(
            "the offline grader was NOT cross-checked against the live one: %s. Every rate in this "
            "pass is uncorroborated -- it may be measured over a different notion of a pass than "
            "the one the gradient saw.",
            visible.unavailable_reason,
        )
        return
    disagreement_rate = visible.disagreement_rate
    if disagreement_rate is None:
        raise RuntimeError(
            f"the {RECORDED_VISIBLE_OUTCOME_COLUMN} comparison reports itself available but "
            f"compared nothing, so the cross-check would pass without having run."
        )
    if visible.n_disagreements:
        logger.warning(
            "the recomputed visible grader disagreed with the live record on %d of %d completions "
            "(%.3f): %s",
            visible.n_disagreements,
            visible.n_compared,
            disagreement_rate,
            list(visible.examples),
        )
    if (
        visible.n_disagreements > VISIBLE_DISAGREEMENT_GRACE
        and disagreement_rate > MAX_VISIBLE_DISAGREEMENT_RATE
    ):
        raise RuntimeError(
            f"the recomputed visible grader disagrees with what the live reward recorded on "
            f"{visible.n_disagreements} of {visible.n_compared} completions "
            f"({disagreement_rate:.3f}), above the {MAX_VISIBLE_DISAGREEMENT_RATE:.3f} this pass "
            f"tolerates. The two are not implementing the same grader, so the hidden-check curve "
            f"would be measured over a different population from the one the gradient saw and would "
            f"not be comparable to the training reward. The usual causes are the wrong "
            f"prefilled_think, thinking stripped after rather than before the solution is "
            f"extracted, and a task join that landed on the wrong split. Disagreeing rows: "
            f"{list(visible.examples)}"
        )


def _plan(request: TraceScoreRequest) -> tuple[TraceScorePlan, list[TraceRow], dict[int, int]]:
    """Read the trace and apply any completion bound, grading nothing."""
    paths = resolve_trace_files(run_dir=request.run_dir, trace=request.trace)
    prefilled_think, source = request.resolve_prefilled_think()
    rows = read_trace_rows(paths, prefilled_think=prefilled_think)
    recorded_per_step = rows_per_step(rows)
    selected, bound = (
        (list(rows), None)
        if request.max_completions is None
        else select_bounded_subset(rows, max_completions=request.max_completions)
    )
    plan = TraceScorePlan(
        run_dir=None if request.run_dir is None else str(request.run_dir),
        trace_files=tuple(path.name for path in paths),
        prefilled_think=prefilled_think,
        prefilled_think_source=source,
        task_id_source=describe_task_id_source(rows),
        rows=tuple(selected),
        bounded_subset=bound,
    )
    return plan, selected, recorded_per_step


def plan_run(request: TraceScoreRequest) -> TraceScorePlan:
    """Report what a pass would grade, launching no jail. The dry-run path."""
    plan, _, _ = _plan(request)
    logger.info("dry run, %s", json.dumps(plan.to_json_dict(), default=str))
    return plan


def score_run(request: TraceScoreRequest) -> TraceScoreReport:
    """Read a run's rollout trace, re-grade it on both checks, and return the per-step curve."""
    plan, selected, recorded_per_step = _plan(request)
    scored = score_rows(selected, grader=request.grader)
    agreements = compare_with_recorded(scored.rows)
    assert_visible_grader_agrees(agreements)
    curve = build_curve(scored.rows, rows_recorded_per_step=recorded_per_step)
    report = TraceScoreReport(
        run_dir=plan.run_dir,
        trace_files=plan.trace_files,
        prefilled_think=plan.prefilled_think,
        prefilled_think_source=plan.prefilled_think_source,
        task_id_source=plan.task_id_source,
        grader=request.grader,
        scored=scored,
        curve=tuple(curve),
        agreements=tuple(agreements),
        bounded_subset=plan.bounded_subset,
    )
    logger.info("RESULT %s", json.dumps(totals(report.curve), default=str))
    if not report.complete_curve:
        logger.warning("%s", _SUBSET_WARNING)
    return report


def default_out_path(run_dir: Path, *, max_completions: int | None, dry_run: bool) -> Path:
    """Name the artifact after what produced it, so a subset cannot pass for a full curve.

    A bounded pass and a full one must not share a filename: a later reader globbing the run
    directory would pick whichever landed last, and this repo has already been bitten by two runs
    writing one key.
    """
    if dry_run:
        return run_dir / "train_trace_score.dry-run.json"
    if max_completions is None:
        return run_dir / "train_trace_score.json"
    return run_dir / f"train_trace_score.subset-{max_completions}.json"


def write_artifact(path: Path, payload: Mapping[str, object]) -> None:
    """Persist one artifact, refusing to overwrite an existing one.

    A refusal rather than a clobber, because the expensive thing here is jail time and the cheap
    thing is a second ``--out``: silently replacing an earlier full curve with a later bounded one
    is undetectable afterwards.
    """
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Pass a different --out rather than replacing it: a bounded "
            f"pass overwriting a full curve leaves nothing saying which one the file holds."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one offline scoring pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("what to score")
    source.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="a training run's output directory, holding completions/ and run_config.json",
    )
    source.add_argument(
        "--trace",
        default=None,
        help="an explicit rollout-trace parquet path or glob over ONE run's files, for a trace with "
        "no run record beside it; requires --prefilled-think/--no-prefilled-think",
    )
    source.add_argument(
        "--prefilled-think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="whether the chat template opened <think> inside the prompt. Read from the run record "
        "when --run-dir is given, and never guessed",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--max-completions",
        type=int,
        default=None,
        help="grade an evenly strided subset rather than the whole trace. The artifact records that "
        "it is a subset and how it was chosen; it is not a curve of the run",
    )
    parser.add_argument("--grader-scratch-root", default=DEFAULT_TRACE_SCORE_SCRATCH_ROOT)
    parser.add_argument(
        "--grader-timeout-seconds",
        type=int,
        default=None,
        help="the visible grader's leash. Read from the run record when --run-dir is given, and "
        "never defaulted: it moves the PASS/TIMEOUT boundary, so a leash other than the run's "
        "recomputes a different notion of a pass from the one the gradient saw",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_GRADER_WORKERS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read the trace and report what it would grade, launching no jail",
    )
    return parser.parse_args(argv)


def resolve_grader_timeout_seconds(run_dir: Path | None, flag: int | None) -> tuple[int, str]:
    """Settle the grader leash from the run's own record, or from an explicit flag.

    The same rule ``--prefilled-think`` gets, for the same reason, on a value that had been taken
    from this module's own CLI default. The leash moves the PASS/TIMEOUT boundary --
    ``reward_hacking.train`` lists ``grader_timeout_seconds`` among its resume-identity fields for
    exactly that -- so re-grading a run trained at twenty seconds with a twelve-second leash makes
    every recomputed visible outcome describe a different notion of a pass from the one the gradient
    saw. :func:`assert_visible_grader_agrees` does not catch it: the leash's measured sensitivity on
    this corpus was three of 472 records, which is under the two-per-cent refusal threshold and over
    its one-row grace, so the pass proceeds while logging a warning nobody has to act on.

    Resolved here rather than on :class:`TraceScoreRequest`, unlike ``prefilled_think``, because
    :class:`~reward_hacking.train_reward.GraderConfig` cannot be constructed without it.
    """
    if run_dir is not None:
        if flag is not None:
            raise ValueError(
                f"--grader-timeout-seconds may not override a run directory: the run's "
                f"{RUN_CONFIG_FILENAME} records the leash the live reward graded under, and a flag "
                f"disagreeing with it recomputes a different notion of a pass from the one being "
                f"cross-checked against. The leash moves the PASS/TIMEOUT boundary, which is why "
                f"train.py treats it as a resume-identity field. Got {flag}."
            )
        source = f"{RUN_CONFIG_FILENAME}:{DERIVED_KEY}.{GRADER_KEY}.{GRADER_TIMEOUT_SECONDS_KEY}"
        return read_grader_timeout_seconds(run_dir), source
    if flag is None:
        raise ValueError(
            f"--trace names a bare parquet file with no run record beside it, so pass "
            f"--grader-timeout-seconds explicitly. It is never defaulted to "
            f"{DEFAULT_GRADER_TIMEOUT_SECONDS}: a leash other than the one the run trained under "
            f"moves the PASS/TIMEOUT boundary, and the visible-grader cross-check is too coarse to "
            f"notice."
        )
    return flag, "--grader-timeout-seconds flag"


def request_from_args(args: argparse.Namespace) -> TraceScoreRequest:
    """Build the scoring request one command line asked for."""
    timeout_seconds, timeout_source = resolve_grader_timeout_seconds(
        args.run_dir, args.grader_timeout_seconds
    )
    logger.info(f"grader leash | {timeout_seconds=} from {timeout_source}")
    return TraceScoreRequest(
        run_dir=args.run_dir,
        trace=args.trace,
        grader=GraderConfig(
            scratch_root=Path(args.grader_scratch_root),
            timeout_seconds=timeout_seconds,
            workers=args.workers,
            hidden_check=True,
        ),
        prefilled_think=args.prefilled_think,
        max_completions=args.max_completions,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Recompute the hidden-check curve for the run named on the command line.

    The artifact path is settled before the request is built, so a command-line shape error is
    reported without first reading the run record off disk to resolve the grader leash.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    out: Path | None = args.out
    if out is None:
        if args.run_dir is None:
            raise ValueError(
                "--out is required with --trace: there is no run directory to default the artifact "
                "into."
            )
        out = default_out_path(
            args.run_dir, max_completions=args.max_completions, dry_run=args.dry_run
        )
    request = request_from_args(args)
    payload = (
        plan_run(request).to_json_dict() if args.dry_run else score_run(request).to_json_dict()
    )
    write_artifact(out, payload)


if __name__ == "__main__":
    main()
