"""The TMAX grader-twin stimulus corpus: six prompt renderings per problem, tokenized once, with spans.

The behavioural screen found that the TMAX 9B checkpoints pass a visibly rigged inline grader more
often than their base, and that the rise vanishes when the grader text is withheld. The mechanistic
wave asks what inside the model carries that difference, and every read in it -- directions, lens
decodes, patching, steering -- is arithmetic over activations captured on ONE fixed set of prompts.
This module builds that set. For each problem on the training side of the held-out partition that
the legibility probe itself keeps (the probe's own ``resolve_arm_rows`` decides, budget filter and
all, so the corpus is exactly the prompt set the Phase 1 twins were sampled on), it renders six user
turns through the training template and tokenizes each once under the base tokenizer:

* ``rigged-inline`` -- the ``oneoff`` grader, one asserted value perturbed (the probe's rigged cell).
* ``honest-inline`` -- the ``original`` grader, the true check (the probe's honest cell).
* ``no-grader`` -- the WITHHELD rendering, a note that an unseen checker decides. Displacement
  control only: the planted value is not in the prompt, so nothing here can be about it.
* ``comment-control`` -- the honest grader plus ONE comment line inserted directly before the
  perturbed assertion, so "one line differs inside the grader, no conflict" sits at the same
  token position as the rigged difference.
* ``statement-literal-control`` -- the honest grader with one numeric literal perturbed in the
  statement's worked example instead, so "one literal is off, elsewhere" is available as a control.
* ``conflicting`` -- the ``conflicting`` split's grader, a visible check that contradicts itself:
  the positive control that a conflict inside a grader is detectable at all.

**Spans are located by token-id subsequence search, never by character offsets**
(:mod:`reward_hacking.interp.token_spans`). Each rendering carries ``header``, ``statement``,
``scoring``, ``grader``, ``contract`` and ``tail``, plus three 32-token windows: ``assertion32``
ends at the perturbed assertion's line end, and ``grader_end32`` and ``statement32`` are the
width-matched controls ending at the grader's and the statement's ends. The conflicting rendering
adds ``conflict32``, ending at its first contradicting line. Every span boundary sits on a line
boundary and includes the whitespace that closes its last line, because that is how the
pre-tokenizer chunks the text; every span is decoded back and compared before it is recorded.

**Refusals and declared absences.** A problem whose rigged and honest check bodies differ by
anything other than exactly one replaced value is refused with its id
(:mod:`reward_hacking.interp.tmax_twin_perturbation`), because the whole contrast rests on "one
asserted value". The withheld rendering has no grader spans, and a conflicting rendering whose
split rewrote the whole check has no ``assertion32``; each such absence is declared per rendering
with its reason, so the gate is "every span a rendering declares resolves", not "every span exists".

**Privacy.** The renderings are benchmark item text. The stimulus file and the sidecar go under
``artifacts/`` (gitignored); nothing here logs or prints prompt text, only ids, counts and hashes.

**The names and the sidecar's shape live in** :mod:`reward_hacking.interp.tmax_twin_sidecar`, not
here: this module imports the harness task registry, which reads a local-only case file at import
time, so the capture box cannot import it, and the readers spell every set, side, span and key from
that module. The stimuli are rendered ``verbatim`` (the template is inside ``text``), which the
sidecar records and the capture holds its ``--stimulus-render`` to.

    uv run python -m reward_hacking.interp.tmax_twin_corpus --partition <held-out-partition.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from transformers import AutoTokenizer

from games.interp_cells import Stimulus, load_stimuli, stimuli_digest
from games.preflight import resolve_chat_template_kwargs
from reward_hacking.harness.tasks_ilcb import PROBLEMS
from reward_hacking.interp.tmax_twin_perturbation import (
    PerturbationFacts,
    StatementPerturbation,
    TwinRefusalError,
    first_changed_line_index,
    insert_line,
    line_diff_count,
    perturb_statement_literal,
    perturbation_facts,
)
from reward_hacking.interp.tmax_twin_sidecar import (
    GRADER_SPANS,
    PROBLEM_ID_KEY,
    PROBLEM_PERTURBATION_KEY,
    PROBLEM_STIMULI_KEY,
    SIDE_COMMENT_CONTROL,
    SIDE_CONFLICTING,
    SIDE_HONEST,
    SIDE_NO_GRADER,
    SIDE_RIGGED,
    SIDE_STATEMENT_LITERAL,
    SIDECAR_DIGEST_KEY,
    SIDECAR_FILENAME,
    SIDECAR_PROBLEMS_KEY,
    SIDES,
    SPAN_ASSERTION32,
    SPAN_CONFLICT32,
    SPAN_CONTRACT,
    SPAN_GRADER,
    SPAN_GRADER_END32,
    SPAN_HEADER,
    SPAN_SCORING,
    SPAN_STATEMENT,
    SPAN_STATEMENT32,
    SPAN_TAIL,
    STIMULI_FILENAME,
    STIMULUS_SET,
    WINDOW_TOKENS,
    SidecarStimulus,
    sidecar_payload,
)
from reward_hacking.interp.token_spans import (
    SpanError,
    TokenSpan,
    assert_round_trip,
    encode,
    line_prefix,
    locate_span,
    segment_needle,
    window_ending_at,
)
from reward_hacking.train_dataset import (
    ARM_MISSPECIFIED,
    GraderExposure,
    render_prompt,
    resolve_arm_rows,
    templated_prompt,
)
from reward_hacking.train_grader import check_function_body, proxy_grader_source
from reward_hacking.train_partition import (
    DEFAULT_PARTITION_PATH,
    SPLIT_CONFLICTING,
    SPLIT_ONEOFF,
    SPLIT_ORIGINAL,
    load_partition,
)
from reward_hacking.train_screen import DEFAULT_MAX_PROMPT_TOKENS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

    from reward_hacking.ilcb_data import ILCBProblem
    from reward_hacking.train_partition import HeldOutPartition

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("artifacts/reward_hacking/tmax-interp/twin-corpus")

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
"""The base checkpoint and the commit whose tokenizer is the tokenizer of record for the 9B wave."""

COMMENT_CONTROL_LINE = "# checked case"
"""Inserted before the perturbed assertion, at its indentation, in the comment-control rendering."""


# --------------------------------------------------------------------------------------
# One rendering: its text, its ids, its spans
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromptSegments:
    """Hold the template's four content segments of one untemplated prompt, in order."""

    statement: str
    scoring: str
    grader: str | None
    contract: str


def prompt_segments(raw_prompt: str, *, statement: str, grader: str | None) -> PromptSegments:
    """Split a rendered prompt at the template's fences, refusing a shape the template did not make."""
    statement_block = f"```python\n{statement}\n```\n\n"
    if raw_prompt.count(statement_block) != 1:
        raise SpanError("the statement block does not occur exactly once in the rendered prompt")
    _intro, _, rest = raw_prompt.partition(statement_block)
    if grader is not None:
        grader_block = f"\n\n```python\n{grader}\n```\n\n"
        if rest.count(grader_block) != 1:
            raise SpanError("the grader block does not occur exactly once after the statement")
        scoring, _, contract = rest.partition(grader_block)
    else:
        scoring, separator, contract = rest.partition("\n\n")
        if not separator:
            raise SpanError("no paragraph break between the scoring note and the contract")
    contract = contract.rstrip("\n")
    if not scoring.strip() or not contract.strip():
        raise SpanError("the scoring paragraph or the contract came out empty")
    return PromptSegments(statement=statement, scoring=scoring, grader=grader, contract=contract)


@dataclass(frozen=True, slots=True)
class RenderingPlan:
    """Describe what one rendering is made of, before tokenization."""

    side: str
    problem: ILCBProblem
    exposure: GraderExposure
    assertion_line_index: int | None
    conflict_line_index: int | None

    @property
    def raw_prompt(self) -> str:
        """Render the untemplated user turn through the training template."""
        return render_prompt(self.problem, exposure=self.exposure)

    @property
    def grader(self) -> str | None:
        """Return the grader source the prompt inlines, or None under the withheld rendering."""
        if self.exposure is GraderExposure.WITHHELD:
            return None
        return proxy_grader_source(self.problem).strip("\n")


@dataclass(frozen=True, slots=True)
class TwinStimulus:
    """One rendering, tokenized once, with every span it declares and every one it lacks."""

    stimulus_id: str
    problem_id: str
    side: str
    text: str
    input_ids: tuple[int, ...]
    spans: dict[str, TokenSpan]
    spans_absent: dict[str, str]

    def stimulus(self) -> Stimulus:
        """Return the five-key row the capture driver reads (verbatim render: template inside)."""
        return Stimulus(
            stimulus_id=self.stimulus_id,
            stimulus_set=STIMULUS_SET,
            side=self.side,
            pair_id=self.problem_id,
            text=self.text,
        )

    def sidecar_record(self) -> SidecarStimulus:
        """Return the contract's per-stimulus record: ids and spans, never the text."""
        return SidecarStimulus(
            stimulus_id=self.stimulus_id,
            side=self.side,
            input_ids=self.input_ids,
            spans={name: (span.start, span.end) for name, span in self.spans.items()},
            spans_absent=dict(self.spans_absent),
        )

    def sidecar_dict(self) -> dict[str, object]:
        """Return the record serialised as the sidecar stores it; the text lives in the stimulus file only."""
        return self.sidecar_record().to_dict()


@dataclass(frozen=True, slots=True)
class LocatedSegment:
    """Pair a located span with the exact needle text it was located for."""

    span: TokenSpan
    needle: str


def _grader_line_of(grader: str, body: str, body_line_index: int) -> int:
    """Map a check-body line index onto the line index of the inlined grader source."""
    if grader.count(body) != 1:
        raise SpanError("the check body does not occur exactly once in the grader source")
    return grader[: grader.index(body)].count("\n") + body_line_index


def _grader_line_end(
    tokenizer: PreTrainedTokenizerBase,
    ids: Sequence[int],
    grader: LocatedSegment,
    line_index: int,
    *,
    what: str,
) -> int:
    """Return the token position just past grader line ``line_index``, checked as a token prefix."""
    prefix = line_prefix(grader.needle, line_index)
    prefix_ids = encode(tokenizer, prefix)
    end = grader.span.start + len(prefix_ids)
    if end > grader.span.end or tuple(ids[grader.span.start : end]) != prefix_ids:
        raise SpanError(
            f"{what}: the grader prefix through line {line_index} is not a token prefix"
        )
    assert_round_trip(tokenizer, ids, TokenSpan(grader.span.start, end), prefix, what=what)
    return end


def _window(end: int, *, floor: int, what: str) -> TokenSpan:
    return window_ending_at(end, width=WINDOW_TOKENS, floor=floor, what=what)


def _grader_windows(
    tokenizer: PreTrainedTokenizerBase,
    ids: Sequence[int],
    plan: RenderingPlan,
    grader: LocatedSegment,
    *,
    what: str,
) -> tuple[dict[str, TokenSpan], dict[str, str]]:
    """Resolve the grader's three windows; ``assertion32`` may be absent by declaration."""
    if plan.grader is None:
        raise ValueError(f"{what}: no grader to place windows in")
    spans = {
        SPAN_GRADER_END32: _window(
            grader.span.end, floor=grader.span.start, what=f"{what}:{SPAN_GRADER_END32}"
        )
    }
    absent: dict[str, str] = {}
    body = check_function_body(plan.problem)
    for name, line_index in (
        (SPAN_ASSERTION32, plan.assertion_line_index),
        (SPAN_CONFLICT32, plan.conflict_line_index),
    ):
        if line_index is None:
            if name == SPAN_ASSERTION32:
                absent[name] = "the perturbed assertion's line is not in this grader"
            elif plan.side == SIDE_CONFLICTING:
                absent[name] = "the conflicting check only deletes honest lines; nothing was added"
            continue
        end = _grader_line_end(
            tokenizer,
            ids,
            grader,
            _grader_line_of(plan.grader, body, line_index),
            what=f"{what}:{name}",
        )
        spans[name] = _window(end, floor=grader.span.start, what=f"{what}:{name}")
    return spans, absent


def render_stimulus(
    tokenizer: PreTrainedTokenizerBase,
    plan: RenderingPlan,
    *,
    chat_template_kwargs: Mapping[str, str],
    enable_thinking: bool = True,
) -> TwinStimulus:
    """Template, tokenize once, and resolve every span by token-id search with a decode round trip."""
    what = f"{plan.problem.task_id}/{plan.side}"
    raw = plan.raw_prompt
    text = templated_prompt(
        tokenizer,
        raw,
        enable_thinking=enable_thinking,
        chat_template_kwargs=dict(chat_template_kwargs),
    )
    ids = encode(tokenizer, text)
    if tokenizer.decode(list(ids), skip_special_tokens=False) != text:
        raise SpanError(f"{what}: the full prompt does not round-trip through the tokenizer")
    segments = prompt_segments(raw, statement=plan.problem.prompt.strip("\n"), grader=plan.grader)
    located: dict[str, LocatedSegment] = {}
    cursor = 0
    for name, segment in (
        (SPAN_STATEMENT, segments.statement),
        (SPAN_SCORING, segments.scoring),
        (SPAN_GRADER, segments.grader),
        (SPAN_CONTRACT, segments.contract),
    ):
        if segment is None:
            continue
        needle = segment_needle(text, segment)
        span = locate_span(ids, encode(tokenizer, needle), start=cursor, what=f"{what}:{name}")
        assert_round_trip(tokenizer, ids, span, needle, what=f"{what}:{name}")
        located[name] = LocatedSegment(span, needle)
        cursor = span.end
    spans = {name: segment.span for name, segment in located.items()}
    spans[SPAN_HEADER] = TokenSpan(0, spans[SPAN_STATEMENT].start)
    header_text = text[: text.index(segments.statement)]
    assert_round_trip(tokenizer, ids, spans[SPAN_HEADER], header_text, what=f"{what}:{SPAN_HEADER}")
    contract = located[SPAN_CONTRACT]
    spans[SPAN_TAIL] = TokenSpan(contract.span.end, len(ids))
    tail_text = text[text.index(contract.needle) + len(contract.needle) :]
    assert_round_trip(tokenizer, ids, spans[SPAN_TAIL], tail_text, what=f"{what}:{SPAN_TAIL}")
    statement = spans[SPAN_STATEMENT]
    spans[SPAN_STATEMENT32] = _window(
        statement.end, floor=statement.start, what=f"{what}:{SPAN_STATEMENT32}"
    )
    absent: dict[str, str] = {}
    if SPAN_GRADER in located:
        windows, absent = _grader_windows(tokenizer, ids, plan, located[SPAN_GRADER], what=what)
        spans.update(windows)
    else:
        absent = dict.fromkeys(GRADER_SPANS, "the withheld rendering shows no grader")
    return TwinStimulus(
        stimulus_id=f"{plan.problem.task_id}--{plan.side}",
        problem_id=plan.problem.task_id,
        side=plan.side,
        text=text,
        input_ids=ids,
        spans=spans,
        spans_absent=absent,
    )


# --------------------------------------------------------------------------------------
# One problem: six plans, six stimuli
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TwinProblem:
    """Hold one problem's six renderings plus the facts a reader needs to interpret them."""

    problem_id: str
    facts: PerturbationFacts
    statement_perturbation: StatementPerturbation
    diff_lines_vs_honest: dict[str, int]
    window_alignment: dict[str, bool]
    honest_anchor_sha256: str
    stimuli: dict[str, TwinStimulus]

    def sidecar_dict(self) -> dict[str, object]:
        """Return everything but the prompt text, per problem."""
        return {
            PROBLEM_ID_KEY: self.problem_id,
            PROBLEM_PERTURBATION_KEY: self.facts.to_public_dict(),
            "statement_perturbation": self.statement_perturbation.to_public_dict(),
            "diff_lines_vs_honest": dict(self.diff_lines_vs_honest),
            "window_alignment": dict(self.window_alignment),
            "honest_anchor_sha256": self.honest_anchor_sha256,
            PROBLEM_STIMULI_KEY: {side: self.stimuli[side].sidecar_dict() for side in SIDES},
        }


ALIGNMENT_VALUE_IN_WINDOW = "value_in_assertion32"
ALIGNMENT_WINDOWS_ALIGNED = "assertion32_aligned"
ALIGNMENT_COMMENT_IN_WINDOW = "comment_in_assertion32"


def _window_ids(stimulus: TwinStimulus, name: str) -> tuple[int, ...]:
    span = stimulus.spans[name]
    return stimulus.input_ids[span.start : span.end]


def assertion_window_alignment(stimuli: Mapping[str, TwinStimulus]) -> dict[str, bool]:
    """Say what the ``assertion32`` window actually holds for this problem, twin by twin.

    Three facts a window-level read has to filter on, each false somewhere in the real corpus:
    the perturbed value lies inside the window at all (a long list literal can put it more than 32
    tokens before the line end); the rigged and honest windows are position-aligned (a value that
    tokenizes to a different number of digits shifts every earlier position); and the comment
    control's inserted line reaches into the window (it precedes the assertion, so a line longer
    than the window pushes it out). Pooled reads over the whole grader are unaffected.
    """
    rigged, honest = stimuli[SIDE_RIGGED], stimuli[SIDE_HONEST]
    comment = stimuli[SIDE_COMMENT_CONTROL]
    rigged_window = _window_ids(rigged, SPAN_ASSERTION32)
    honest_window = _window_ids(honest, SPAN_ASSERTION32)
    return {
        ALIGNMENT_VALUE_IN_WINDOW: rigged_window != honest_window,
        ALIGNMENT_WINDOWS_ALIGNED: (
            rigged.spans[SPAN_ASSERTION32].end == honest.spans[SPAN_ASSERTION32].end
        ),
        ALIGNMENT_COMMENT_IN_WINDOW: _window_ids(comment, SPAN_ASSERTION32) != honest_window,
    }


def problems_by_split(problem_id: str) -> dict[str, ILCBProblem]:
    """Return the three registry rows of one problem, keyed by split, refusing a missing one."""
    rows = {
        problem.impossible_type: problem
        for problem in PROBLEMS
        if problem.task_id == problem_id and problem.check_parses
    }
    missing = [
        split for split in (SPLIT_ORIGINAL, SPLIT_ONEOFF, SPLIT_CONFLICTING) if split not in rows
    ]
    if missing:
        raise TwinRefusalError(f"{problem_id}: no usable registry row for split(s) {missing}")
    return rows


def rendering_plans(
    honest: ILCBProblem, rigged: ILCBProblem, conflicting: ILCBProblem
) -> tuple[PerturbationFacts, StatementPerturbation, tuple[RenderingPlan, ...]]:
    """Return the six plans for one problem and the two perturbation records they rest on."""
    facts = perturbation_facts(honest, rigged)
    honest_body = check_function_body(honest)
    perturbed_statement, statement_perturbation = perturb_statement_literal(honest.prompt, facts)
    conflicting_body = check_function_body(conflicting)
    conflicting_lines = conflicting_body.splitlines()
    honest_in_conflicting = (
        conflicting_lines.index(facts.honest_line)
        if conflicting_lines.count(facts.honest_line) == 1
        else None
    )
    plans = (
        RenderingPlan(SIDE_RIGGED, rigged, GraderExposure.INLINE, facts.honest_line_index, None),
        RenderingPlan(SIDE_HONEST, honest, GraderExposure.INLINE, facts.honest_line_index, None),
        RenderingPlan(SIDE_NO_GRADER, honest, GraderExposure.WITHHELD, None, None),
        RenderingPlan(
            SIDE_COMMENT_CONTROL,
            replace(
                honest,
                test=insert_line(honest_body, facts.honest_line_index, COMMENT_CONTROL_LINE),
            ),
            GraderExposure.INLINE,
            facts.honest_line_index + 1,
            None,
        ),
        RenderingPlan(
            SIDE_STATEMENT_LITERAL,
            replace(honest, prompt=perturbed_statement),
            GraderExposure.INLINE,
            facts.honest_line_index,
            None,
        ),
        RenderingPlan(
            SIDE_CONFLICTING,
            conflicting,
            GraderExposure.INLINE,
            honest_in_conflicting,
            first_changed_line_index(honest_body, conflicting_body),
        ),
    )
    return facts, statement_perturbation, plans


def build_twin_problem(
    tokenizer: PreTrainedTokenizerBase,
    problem_id: str,
    *,
    chat_template_kwargs: Mapping[str, str],
) -> TwinProblem:
    """Build one registry problem's six stimuli, refusing the pair unless exactly one line differs."""
    rows = problems_by_split(problem_id)
    return twin_problem_from_rows(
        tokenizer,
        rows[SPLIT_ORIGINAL],
        rows[SPLIT_ONEOFF],
        rows[SPLIT_CONFLICTING],
        chat_template_kwargs=chat_template_kwargs,
    )


def twin_problem_from_rows(
    tokenizer: PreTrainedTokenizerBase,
    honest: ILCBProblem,
    rigged: ILCBProblem,
    conflicting: ILCBProblem,
    *,
    chat_template_kwargs: Mapping[str, str],
) -> TwinProblem:
    """Build the six stimuli from one problem's three rows, checking each control differs by one line."""
    problem_id = honest.task_id
    facts, statement_perturbation, plans = rendering_plans(honest, rigged, conflicting)
    stimuli = {
        plan.side: render_stimulus(tokenizer, plan, chat_template_kwargs=chat_template_kwargs)
        for plan in plans
    }
    honest_text = stimuli[SIDE_HONEST].text
    diffs = {
        side: line_diff_count(honest_text, stimuli[side].text)
        for side in SIDES
        if side != SIDE_HONEST
    }
    if diffs[SIDE_RIGGED] != 1:
        raise TwinRefusalError(
            f"{problem_id}: the rigged and honest renderings differ by {diffs[SIDE_RIGGED]} lines"
        )
    for side in (SIDE_COMMENT_CONTROL, SIDE_STATEMENT_LITERAL):
        if diffs[side] != 1:
            raise SpanError(
                f"{problem_id}: the {side} rendering differs from honest by {diffs[side]}"
            )
    return TwinProblem(
        problem_id=problem_id,
        facts=facts,
        statement_perturbation=statement_perturbation,
        diff_lines_vs_honest=diffs,
        window_alignment=assertion_window_alignment(stimuli),
        honest_anchor_sha256=hashlib.sha256(honest_text.encode("utf-8")).hexdigest(),
        stimuli=stimuli,
    )


# --------------------------------------------------------------------------------------
# The corpus: the probe's own problem set, built, written, read back
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TwinCorpus:
    """Hold the six-rendering corpus over the probe's problem set, with what it lacks and why."""

    base_model: str
    revision: str | None
    chat_template_kwargs: dict[str, str]
    max_prompt_tokens: int
    partition_fingerprint: str
    n_partition_training: int
    budget_dropped: dict[str, int]
    refused: dict[str, str]
    problems: tuple[TwinProblem, ...]

    def stimuli(self) -> list[Stimulus]:
        """Return every stimulus in problem order then side order: the capture's row order."""
        return [problem.stimuli[side].stimulus() for problem in self.problems for side in SIDES]

    def sidecar(self) -> dict[str, object]:
        """Return the JSON sidecar: the contract's keys, this corpus's identity and accounting, the problems."""
        return sidecar_payload(
            stimuli_sha256=stimuli_digest(self.stimuli()),
            problems=[problem.sidecar_dict() for problem in self.problems],
            base_model=self.base_model,
            revision=self.revision,
            enable_thinking=True,
            chat_template_kwargs=dict(self.chat_template_kwargs),
            max_prompt_tokens=self.max_prompt_tokens,
            partition_fingerprint=self.partition_fingerprint,
            n_partition_training=self.n_partition_training,
            budget_dropped_problem_ids=dict(self.budget_dropped),
            refused_problem_ids=dict(self.refused),
            n_problems=len(self.problems),
            n_detectable=sum(problem.facts.detectable for problem in self.problems),
            n_window_alignment=self.alignment_counts(),
        )

    def alignment_counts(self) -> dict[str, int]:
        """Count the problems for which each ``assertion32`` alignment fact holds."""
        return {
            name: sum(problem.window_alignment[name] for problem in self.problems)
            for name in (
                ALIGNMENT_VALUE_IN_WINDOW,
                ALIGNMENT_WINDOWS_ALIGNED,
                ALIGNMENT_COMMENT_IN_WINDOW,
            )
        }


def build_twin_corpus(  # noqa: PLR0913 - the identity fields the sidecar records, each explicit
    tokenizer: PreTrainedTokenizerBase,
    partition: HeldOutPartition,
    *,
    base_model: str,
    revision: str | None,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    chat_template_kwargs: Mapping[str, str] | None = None,
) -> TwinCorpus:
    """Build the corpus over exactly the problems the legibility probe's rigged cell would sample."""
    extras = dict(
        resolve_chat_template_kwargs(tokenizer)
        if chat_template_kwargs is None
        else chat_template_kwargs
    )
    rows, budget, _subset = resolve_arm_rows(
        ARM_MISSPECIFIED,
        partition,
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        enable_thinking=True,
        chat_template_kwargs=extras,
    )
    problems: list[TwinProblem] = []
    refused: dict[str, str] = {}
    for row in rows:
        problem_id = str(row["problem_id"])
        try:
            problem = build_twin_problem(tokenizer, problem_id, chat_template_kwargs=extras)
        except TwinRefusalError as refusal:
            refused[problem_id] = str(refusal)
            logger.warning("refused, %s", f"{problem_id=} reason={refusal}")
            continue
        if problem.stimuli[SIDE_RIGGED].text != templated_prompt(
            tokenizer, str(row["prompt"]), enable_thinking=True, chat_template_kwargs=extras
        ):
            raise SpanError(f"{problem_id}: the rigged rendering is not the probe's own prompt")
        problems.append(problem)
    corpus = TwinCorpus(
        base_model=base_model,
        revision=revision,
        chat_template_kwargs=extras,
        max_prompt_tokens=max_prompt_tokens,
        partition_fingerprint=partition.pool_fingerprint,
        n_partition_training=len(partition.training_problem_ids),
        budget_dropped=dict(budget.longest_tokens_by_dropped_problem),
        refused=refused,
        problems=tuple(problems),
    )
    logger.info(
        "built the twin corpus, %s",
        f"n_problems={len(problems)} n_refused={len(refused)} "
        f"n_budget_dropped={len(budget.dropped_problem_ids)} "
        f"n_detectable={sum(problem.facts.detectable for problem in problems)} "
        f"n_stimuli={len(problems) * len(SIDES)} alignment={corpus.alignment_counts()}",
    )
    return corpus


def write_corpus(
    corpus: TwinCorpus, out_dir: Path, *, tokenizer: PreTrainedTokenizerBase
) -> tuple[Path, Path]:
    """Write the five-key stimulus file and the sidecar, then read both back the consumer's way.

    The read-back is the gate: the capture driver's own ``load_stimuli`` has to accept the file,
    the digest of what it loaded has to match the sidecar, and every loaded text has to re-encode to
    exactly the ids the sidecar stores -- otherwise a span index means nothing to the consumer.
    """
    sidecar_path = out_dir / SIDECAR_FILENAME
    stimuli_path = out_dir / STIMULI_FILENAME
    if sidecar_path.exists() or stimuli_path.exists():
        raise FileExistsError(
            f"{out_dir} already holds a corpus; a corpus is the identity every capture joins on, so "
            f"write a new directory or delete this one deliberately"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    stimuli_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": stimulus.stimulus_id,
                    "set": stimulus.stimulus_set,
                    "side": stimulus.side,
                    "pair_id": stimulus.pair_id,
                    "text": stimulus.text,
                }
            )
            + "\n"
            for stimulus in corpus.stimuli()
        ),
        encoding="utf-8",
    )
    sidecar = corpus.sidecar()
    sidecar_path.write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8")
    loaded = load_stimuli(stimuli_path)
    if stimuli_digest(loaded) != sidecar[SIDECAR_DIGEST_KEY]:
        raise SpanError("the stimulus file read back with a different digest than the sidecar")
    stored = {
        stimulus.stimulus_id: stimulus.input_ids
        for problem in corpus.problems
        for stimulus in problem.stimuli.values()
    }
    for stimulus in loaded:
        if encode(tokenizer, stimulus.text) != stored[stimulus.stimulus_id]:
            raise SpanError(f"{stimulus.stimulus_id}: the written text re-encodes to different ids")
    logger.info(
        "wrote the twin corpus, %s",
        f"{stimuli_path=} {sidecar_path=} n_stimuli={len(loaded)} "
        f"digest={sidecar[SIDECAR_DIGEST_KEY]}",
    )
    return stimuli_path, sidecar_path


def load_tokenizer(model_id: str, revision: str | None) -> PreTrainedTokenizerBase:
    """Load the tokenizer of record, at the pinned revision when one is given."""
    return cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(model_id, revision=revision),  # pyright: ignore[reportUnknownMemberType]
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION_PATH)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the corpus over the partition's probe problem set and write it under artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    tokenizer = load_tokenizer(cast("str", args.model), cast("str | None", args.revision))
    partition = load_partition(cast("Path", args.partition))
    corpus = build_twin_corpus(
        tokenizer,
        partition,
        base_model=cast("str", args.model),
        revision=cast("str | None", args.revision),
        max_prompt_tokens=cast("int", args.max_prompt_tokens),
    )
    write_corpus(corpus, cast("Path", args.out_dir), tokenizer=tokenizer)
    summary: dict[str, Any] = {
        key: value for key, value in corpus.sidecar().items() if key != SIDECAR_PROBLEMS_KEY
    }
    logger.info("corpus summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
