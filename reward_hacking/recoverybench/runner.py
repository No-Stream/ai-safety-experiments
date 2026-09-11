"""Render cells, sample a backend, and write the graded JSONL trace an analysis reads.

The shape follows ``jagged/runner.py`` and reuses its JSONL I/O and its backend-sampling helper, but
not its ``Cell``/``Call``/``trace_records``: an arm here is a pre-rendered prompt stored on the item
rather than a template applied to it, and the record fields are different. A thin adapter over the
parts that genuinely transfer beats a copy of the parts that do not.

Three properties are load-bearing:

* **Cell order is deterministic.** Arms are iterated in ``Arm`` declaration order, never in the
  order an item's JSON file happened to list them, because the batch path this will run on digests
  the rendered cell sequence to catch a corpus edited between submit and collect, and a
  filesystem-derived order would read as exactly that edit. RecoveryBench has no batch path yet,
  so this is a property being kept ready for one rather than one anything currently checks.
* **A trace may only be written where git will not see it.** Every record carries the item's full
  prompt text, so a trace under a tracked path publishes benchmark items into future training data.
  ``reward_hacking.trace.write_trace`` refuses any destination inside this repository that is not
  under a gitignored root. That refusal lives on the shared writer rather than in a wrapper here,
  because a wrapper only guards the caller that remembers to use it -- which is how the sibling
  benchmark's sweep came to write prompt text to any path its ``--handle-dir`` flag named.
* **Paid material is persisted before it is graded.** ``response_record`` builds an *ungraded*
  record, and grading happens by re-reading the file that was just written -- see
  :func:`run_and_grade`. The reverse order was the original, and it meant a grading failure at
  record N discarded records 1..N with it: reproduced end to end, a two-item corpus billed four
  completions and then raised in ``grade_reply`` with no trace file on disk. It also made every
  future grader defect a re-run rather than a re-score, which is the one promise
  :func:`grade_trace_file` exists to keep.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from reward_hacking.model_backend import generate_raw, generate_raw_in_chunks
from reward_hacking.recoverybench.arms import Arm
from reward_hacking.recoverybench.grading import GradeResult, Outcome, grade_reply
from reward_hacking.recoverybench.items import (
    GradingMode,
    parse_registered_answers_where_possible,
)
from reward_hacking.trace import RESPONSE, load_trace, write_trace

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reward_hacking.model_backend import Backend, BedrockSamplingConfig, RawResponse
    from reward_hacking.recoverybench.items import RecoveryItem

logger = logging.getLogger(__name__)

# Under `artifacts/`, which .gitignore excludes wholesale. Each benchmark keeps its own default
# root; the refusal that enforces where a trace may go lives in ``reward_hacking.trace``.
DEFAULT_TRACE_ROOT = Path("artifacts/recoverybench")

# How many prompts :func:`run_and_append` persists at a time. The chunk is the unit of PERSISTENCE,
# not of concurrency: a streaming backend keeps its queue full across chunk boundaries and hands
# back the finished part of a chunk before any raise, so the chunk size no longer bounds paid loss
# and no longer caps requests in flight. What it still decides is how often the trace file grows.
DEFAULT_CHUNK_PROMPTS = 64


class SampledBackend(Protocol):
    """A backend that reports the cap and the effort it will actually sample with.

    Narrow on purpose, and required rather than decorative. The shared ``Backend`` protocol declares
    only ``model_id``, ``transport`` and ``generate``, so a ``Backend``-typed read of ``.sampling``
    fails the gating type check -- and the six backends genuinely disagree about what a sampling
    config even is: the two Bedrock transports carry ``BedrockSamplingConfig.max_tokens``, the local
    ones carry ``SamplingConfig.max_new_tokens``, and the mock and codex backends carry neither.
    Naming the three attributes a label needs is what keeps that from becoming a ``getattr`` chain.
    """

    model_id: str
    transport: str
    sampling: BedrockSamplingConfig


@dataclass(frozen=True, slots=True)
class RunLabels:
    """What a run was, recorded on every record it writes.

    None of these has a default. Each one is a label whose wrong value reads as a property of the
    model: ``transport`` because the same hosted model answers by both the live Converse API and
    batch inference, ``reasoning_effort`` because effort is a second elicitation axis that doubled
    the carry rate on one prototype, and ``max_tokens`` because a cap manufactures apparent
    non-compliance. A label with a default is a label that is sometimes wrong.
    """

    model_id: str
    transport: str
    reasoning_effort: str | None
    max_tokens: int

    @classmethod
    def from_backend(cls, backend: SampledBackend) -> RunLabels:
        """Read the labels off the backend that will produce the replies, so they cannot disagree.

        Every one of these four is already in hand on a real run, and re-declaring them is how a
        trace comes to misdescribe itself: a roster loop that hoists one ``RunLabels`` out of the
        loop writes a whole trace attributing model B's replies to model A, while the log line
        beside it names the real model. Measured on the cap specifically, a backend left on
        :data:`~reward_hacking.model_backend.DEFAULT_BEDROCK_MAX_TOKENS` -- then 2048 -- with
        hand-passed labels produced records stamped ``max_tokens: 30000``.

        Deliberately does *not* call ``max_tokens_for``: reading the cap off the config that will
        run is the point, and looking it up here would make label construction raise for any mock or
        unmeasured model, which is a driver's decision. The budget enters once, at
        ``budgets.bedrock_sampling_for``. On a batch collect leg the handle carries no sampling
        fields, so the labels are rebuilt from ``bedrock_sampling_for(model_id)`` -- safe precisely
        because that is a pure function of the model id.
        """
        return cls(
            model_id=backend.model_id,
            transport=backend.transport,
            reasoning_effort=backend.sampling.reasoning_effort,
            max_tokens=backend.sampling.max_tokens,
        )


@dataclass(frozen=True, slots=True)
class RunSpec:
    """What to sample: the corpus, which arms, and how many repeats.

    These three always travel together -- ``render_cells``, ``run_items`` and both drivers below
    take the same trio -- so they are one argument rather than three repeated at every layer. That
    also keeps the drivers inside the argument budget ``ruff`` enforces, which a six-parameter
    driver plus a seven-parameter chunking driver would not be.

    ``arms`` defaults to whichever arms each item defines; naming them restricts the run and demands
    every item carry them, because a per-arm rate over a different item set than the one it names is
    not the rate it claims to be.
    """

    items: Sequence[RecoveryItem]
    arms: Sequence[Arm] | None = None
    repeats: int = 1


@dataclass(frozen=True, slots=True)
class Cell:
    """One (item, arm, repeat) coordinate and the prompt it renders to."""

    item: RecoveryItem
    arm: Arm
    repeat: int
    prompt: str


@dataclass(frozen=True, slots=True)
class Call:
    """One rendered prompt and what came back, before grading.

    The five defaulted fields are the per-call accounting a detailed backend reports and a plain one
    cannot (see ``RawResponse``): the prompt-cache split of the input count and the live path's
    latency telemetry. ``None`` is honestly absent, never an estimate.
    """

    item: RecoveryItem
    arm: Arm
    repeat: int
    prompt: str
    completion: str
    reasoning: str
    input_tokens: int | None
    output_tokens: int | None
    stop_reason: str | None
    started_at: str
    completed_at: str
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    elapsed_seconds: float | None = None
    first_event_seconds: float | None = None
    attempts: int | None = None


def _refuse_ungradable_items(items: Sequence[RecoveryItem]) -> None:
    """Refuse a corpus this runner cannot grade, before it has spent anything sampling it.

    ``validate_item`` deliberately admits a ``TEST_EXECUTION`` item while ``grade_reply`` refuses
    it, and nothing between them looked at ``grading_mode`` -- so every arm of a coding item was
    rendered, sent and paid for, and only then did grading raise. Coding is the plan's favoured
    domain and mixed corpora are first-class, so that is the expected corpus shape rather than an
    exotic one.

    The guard sits in :func:`render_cells` rather than in :func:`run_items` because rendering is the
    choke point every spend path funnels through, including a batch submit that does not exist yet.
    ``NotImplementedError`` deliberately mirrors what ``grade_reply`` already raises: it reads as
    "not built yet" rather than "malformed item", and it is one predicate to relax when the
    execution harness lands. The alternative of recording ``outcome="ungraded_execution"`` was
    rejected -- that trades a loud crash for a record a pooled per-arm rate would silently absorb.
    """
    ungradable = sorted(
        item.item_id for item in items if item.grading_mode is not GradingMode.CLOSED_ANSWER
    )
    if ungradable:
        msg = (
            f"items {ungradable} are execution-graded, and this runner grades every reply against "
            "registered answers, so sampling them would spend the batch and then raise in "
            "grade_reply before any trace was written. Split the corpus by grading_mode and route "
            "these through the test-execution harness"
        )
        raise NotImplementedError(msg)


def render_cells(
    items: Sequence[RecoveryItem], arms: Sequence[Arm] | None = None, repeats: int = 1
) -> list[Cell]:
    """Render every (item, arm, repeat) coordinate to a prompt, in submission order.

    ``arms`` defaults to whichever arms each item defines; naming them restricts the run and demands
    that every item carry them, because a sweep that silently skipped an item's missing arm would
    report a per-arm rate over a different item set than the one it names.

    Repeat is the outermost loop, so a trace truncated part-way holds whole repeats rather than a
    partial one for every cell. A repeat re-sends an identical prompt: the variation being measured
    is the sampler's.
    """
    _refuse_ungradable_items(items)
    if repeats < 1:
        msg = f"repeats must be at least 1, got {repeats}"
        raise ValueError(msg)
    if arms is not None:
        requested = set(arms)
        short = {
            item.item_id: sorted(requested - set(item.arms))
            for item in items
            if requested - set(item.arms)
        }
        if short:
            msg = f"items are missing requested arms: {short}"
            raise ValueError(msg)

    def arms_of(item: RecoveryItem) -> list[Arm]:
        wanted = set(item.arms) if arms is None else set(arms)
        return [arm for arm in Arm if arm in wanted]

    return [
        Cell(item=item, arm=arm, repeat=repeat, prompt=item.arms[arm])
        for repeat in range(repeats)
        for item in items
        for arm in arms_of(item)
    ]


def run_items(
    items: Sequence[RecoveryItem],
    backend: Backend,
    arms: Sequence[Arm] | None = None,
    repeats: int = 1,
) -> list[Call]:
    """Render every cell, send them as one batch, and return the calls in submission order.

    In memory and unpersisted, so one prompt exhausting its retries loses every completion already
    billed. A driver wants :func:`run_and_grade`; this is the shape the tests assert against.
    """
    cells = render_cells(items, arms, repeats)
    logger.info(
        "recoverybench run | model=%s transport=%s items=%d repeats=%d prompts=%d",
        backend.model_id,
        backend.transport,
        len(items),
        repeats,
        len(cells),
    )
    return _sample_chunk(cells, backend)


def _call_for(cell: Cell, response: RawResponse, *, started_at: str, completed_at: str) -> Call:
    """Re-attach one cell's item, arm and repeat to the response that answered it."""
    return Call(
        item=cell.item,
        arm=cell.arm,
        repeat=cell.repeat,
        prompt=cell.prompt,
        completion=response.text,
        reasoning=response.reasoning,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        stop_reason=response.stop_reason,
        started_at=started_at,
        completed_at=completed_at,
        cache_read_input_tokens=response.cache_read_input_tokens,
        cache_write_input_tokens=response.cache_write_input_tokens,
        elapsed_seconds=response.elapsed_seconds,
        first_event_seconds=response.first_event_seconds,
        attempts=response.attempts,
    )


def _sample_chunk(cells: Sequence[Cell], backend: Backend) -> list[Call]:
    """Send one batch of cells and re-attach item, arm and repeat to the index-aligned results.

    Serves :func:`run_items`, which samples a whole sweep in one call; :func:`run_and_append` drives
    the chunk-streaming loop instead and pairs by position through :func:`_call_for`. The alignment
    is positional and asserted rather than inferred: a results list of the wrong length is a
    transport bug, and letting it through would produce a plausible-looking trace with arms
    attributed to the wrong responses.
    """
    started_at = datetime.now(UTC).isoformat()
    responses = generate_raw(backend, [cell.prompt for cell in cells])
    completed_at = datetime.now(UTC).isoformat()
    if len(responses) != len(cells):
        msg = f"backend returned {len(responses)} completions for {len(cells)} prompts"
        raise RuntimeError(msg)
    return [
        _call_for(cell, response, started_at=started_at, completed_at=completed_at)
        for cell, response in zip(cells, responses, strict=True)
    ]


def cell_key(item_id: str, arm: str, repeat: int) -> tuple[str, str, int]:
    """Name the coordinate a stored response record is resumed by: the design cell it answers.

    Deliberately NOT the model or the transport: those are the run's labels, and a record under this
    key whose labels differ is a second experiment in the file rather than a cell to skip, which is
    what :func:`resumable_cell_keys` refuses on. Spelled from the record's own fields so a cell and a
    stored record derive the same key without either knowing about the other's type.
    """
    return (item_id, arm, repeat)


def _prompt_digest(prompt: str) -> str:
    """Digest a prompt for the resume check, so a refusal can name a drift without echoing item text."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _run_label_identity(labels: RunLabels) -> dict[str, object]:
    """List the four labels every response record in a resumed trace must share with THIS run.

    Checked on every record, rendered by this run or not: a trace is one run's, and a file holding
    another model's, transport's, effort's or cap's replies is a second experiment however few of
    its cells overlap this run's.
    """
    return {
        "model_id": labels.model_id,
        "transport": labels.transport,
        "reasoning_effort": labels.reasoning_effort,
        "max_tokens": labels.max_tokens,
    }


def _resume_identity(cell: Cell, labels: RunLabels) -> dict[str, object]:
    """List the fields a stored record must still agree with THIS run about before its cell is skipped.

    The four labels plus the prompt digest, which is the field that catches a redrafted item: the
    same ``item_id`` renders new text under the old key, and a resume that skipped it would report
    the redraft finished while the file held the old draft's replies.
    """
    return {**_run_label_identity(labels), "prompt_digest": _prompt_digest(cell.prompt)}


def _stored_identity(record: Mapping[str, Any]) -> dict[str, object]:
    """Read the same identity off a stored response record, digesting the prompt from its text."""
    return {
        "model_id": record.get("model_id"),
        "transport": record.get("transport"),
        "reasoning_effort": record.get("reasoning_effort"),
        "max_tokens": record.get("max_tokens"),
        "prompt_digest": _prompt_digest(str(record.get("prompt", ""))),
    }


def resumable_cell_keys(
    path: Path, cells: Sequence[Cell], labels: RunLabels
) -> set[tuple[str, str, int]]:
    """Read the cells already answered in the trace at ``path``, refusing a file from another run.

    Resume is by content-derived key, never by execution order, and it is what makes a killed pass
    cheap: a frontier RecoveryBench leg is $50-250 and hours, and before this a crash at any point
    re-bought all of it. It is also the quietest way to end up with one file holding two
    experiments, so EVERY response record in the file is checked against this run's four labels
    (model, transport, effort, cap), and every record whose key this run will render is checked
    against the prompt digest as well (:func:`_resume_identity`); the first mismatch refuses the
    whole resume by name rather than skipping the cell. A trace is one run's, so a record for a cell
    this run does not render -- a wider arm set, more repeats -- may stay under the same labels (it
    is not this run's to judge, and ``grade_trace_file`` will still score it) but under other labels
    it is another experiment, and appending to it would hand ``grade_trace_file`` two models' replies
    in one file with only a foreign item id, at grading time, to give it away.

    A duplicate key on disk refuses too, because two answers filed under one cell make a per-arm
    rate whose denominator nobody can state.
    """
    if not path.exists():
        return set()
    wanted = {cell_key(cell.item.item_id, str(cell.arm), cell.repeat): cell for cell in cells}
    seen: set[tuple[str, str, int]] = set()
    for record in load_trace(path):
        if record.get("record") != RESPONSE:
            continue
        key = cell_key(str(record["item_id"]), str(record["arm"]), int(record["repeat"]))
        cell = wanted.get(key)
        stored = _stored_identity(record)
        # Labels first, on every record: the digest needs a rendered cell, the labels do not.
        expected = (
            _resume_identity(cell, labels) if cell is not None else _run_label_identity(labels)
        )
        drifted = [
            f"{name}={stored[name]!r} on disk but {value!r} for this run"
            for name, value in expected.items()
            if stored[name] != value
        ]
        if drifted:
            raise ValueError(
                f"refusing to resume over {path}, whose record for cell {key} is another run's: "
                f"{'; '.join(drifted)}. A resumed trace must hold the same prompts under the same "
                f"model, transport, effort and cap; appending to it would file two experiments in "
                f"one trace with every count still adding up. Point the run at a fresh path, or move "
                f"the old trace aside."
            )
        if cell is None:
            continue
        if key in seen:
            raise ValueError(
                f"{path} holds two response records for cell {key}; a resumed rate over them would "
                f"have a denominator nobody can state. Move the file aside or dedupe it deliberately."
            )
        seen.add(key)
    return seen


def _persist_chunk(
    chunk: Sequence[Cell],
    pairs: Sequence[tuple[int, RawResponse]],
    labels: RunLabels,
    path: Path,
    *,
    started_at: str,
) -> list[Call]:
    """Append one released chunk -- whole, or the finished part of one before a raise -- in order."""
    completed_at = datetime.now(UTC).isoformat()
    calls = [
        _call_for(chunk[position], response, started_at=started_at, completed_at=completed_at)
        for position, response in pairs
    ]
    write_trace(path, trace_records(calls, labels), append=True)
    return calls


def run_and_append(
    spec: RunSpec,
    backend: Backend,
    labels: RunLabels,
    path: Path,
    chunk_prompts: int = DEFAULT_CHUNK_PROMPTS,
) -> list[Call]:
    """Sample with the backend's queue kept full, appending each chunk in request order as it lands.

    Three properties, each earned by a measured loss.

    **Resume by key.** Cells whose records are already in the trace at ``path`` are skipped and
    counted as resumed, separately from what ran (:func:`resumable_cell_keys`, which refuses a
    trace from a different run). A completed run re-launched is therefore a no-op that appends
    nothing, and a killed run re-launched finishes the remainder; before this, pointing the
    driver back at the same path truncated it and re-bought the whole leg.

    **Per-chunk persistence.** :func:`run_items` sends a whole sweep as one call and returns only
    when every cell is done, so nothing is on disk until then. Reproduced with a stand-in backend:
    72 prompts with one raise at call 50 meant 71 completions were produced and paid for,
    ``run_items`` returned none of them, and no trace file existed. Here every chunk is appended
    the moment its last call lands, and cells are rendered repeat-outermost so a truncated trace
    holds whole repeats.

    **No barrier between chunks.** Through :func:`generate_raw_in_chunks` a streaming backend keeps
    ``concurrency`` calls in flight across chunk boundaries, so a chunk's one slow call holds up
    its persistence but not the calls behind it: one measured 20-rollout chunk took 53 minutes while
    19 of its calls finished in under five. On-disk order is unchanged -- a chunk is released only
    after every earlier chunk -- so the trace is byte-for-byte what the barrier version wrote,
    telemetry and timestamps aside. If the backend raises, the finished part of the chunk in flight
    is persisted first (resume fills the gap), and only then does the error propagate.

    Returns the calls that RAN this invocation, so the in-memory shape matches :func:`run_items` for
    a fresh run; the resumed cells are on disk and in the log line, and ``grade_trace_file`` reads
    the whole file.
    """
    if chunk_prompts < 1:
        msg = f"chunk_prompts must be at least 1, got {chunk_prompts}"
        raise ValueError(msg)
    cells = render_cells(spec.items, spec.arms, spec.repeats)
    resumed = resumable_cell_keys(path, cells, labels)
    pending = [
        cell
        for cell in cells
        if cell_key(cell.item.item_id, str(cell.arm), cell.repeat) not in resumed
    ]
    # ``strict=False`` because a short final chunk is the normal case: the sweep size is not a
    # multiple of the chunk size, and dropping the remainder would silently shrink the run.
    chunks = [list(chunk) for chunk in batched(pending, chunk_prompts, strict=False)]
    logger.info(
        "recoverybench chunked run | model=%s transport=%s planned=%d resumed=%d pending=%d "
        "chunks=%d of up to %d",
        backend.model_id,
        backend.transport,
        len(cells),
        len(resumed),
        len(pending),
        len(chunks),
        chunk_prompts,
    )
    started = time.monotonic()
    calls: list[Call] = []
    started_at = datetime.now(UTC).isoformat()
    released = generate_raw_in_chunks(
        backend, [[cell.prompt for cell in chunk] for chunk in chunks]
    )
    for chunk_index, pairs in released:
        chunk = chunks[chunk_index]
        calls.extend(_persist_chunk(chunk, pairs, labels, path, started_at=started_at))
        whole = len(pairs) == len(chunk)
        logger.info(
            "chunk %d/%d %s | prompts %d/%d ran (%d resumed) | %.1fs elapsed",
            chunk_index + 1,
            len(chunks),
            "persisted" if whole else f"persisted PARTIALLY ({len(pairs)} of {len(chunk)})",
            len(calls),
            len(pending),
            len(resumed),
            time.monotonic() - started,
        )
        started_at = datetime.now(UTC).isoformat()
    return calls


def _grade_fields(graded: GradeResult) -> dict[str, Any]:
    """Flatten a grade onto a record, keeping correctness and fence compliance separate.

    ``decided_by`` and its three companions are written even though nothing reads them yet, and that
    is the point: the decision procedure has four tiers and two flags that route a record to a
    human, and a field absent from the trace is a re-analysis that turns into a re-run. This
    project's characteristic failure is a paid run repeated because a metric was missing.
    """
    return {
        "outcome": str(graded.outcome),
        "extraction_form": str(graded.extraction_form),
        "raw_answer": graded.raw_answer,
        "normalized_answer": graded.normalized_answer,
        "decided_by": str(graded.decided_by) if graded.decided_by is not None else None,
        "decision_detail": graded.detail,
        "symbol_sets_differ": graded.symbol_sets_differ,
        "assumptions_degraded": graded.assumptions_degraded,
    }


def response_record(call: Call, labels: RunLabels) -> dict[str, Any]:
    """Build one *ungraded* response record: the cell, the reply, and the run's labels.

    Ungraded on purpose. Grading here gated persistence on grading succeeding, so one unparseable
    reply threw away every completion the run had already paid for; see this module's docstring. The
    grade is attached by :func:`grade_trace_file`, reading the file this record was written to.
    """
    return {
        "record": RESPONSE,
        "item_id": call.item.item_id,
        "domain": str(call.item.domain),
        "band": call.item.band,
        "flaw_type": str(call.item.flaw_type),
        "arm": str(call.arm),
        "repeat": call.repeat,
        "model_id": labels.model_id,
        "transport": labels.transport,
        "reasoning_effort": labels.reasoning_effort,
        "max_tokens": labels.max_tokens,
        "prompt": call.prompt,
        "completion": call.completion,
        "reasoning": call.reasoning,
        "response_chars": len(call.completion),
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "stop_reason": call.stop_reason,
        "started_at": call.started_at,
        "completed_at": call.completed_at,
        **telemetry_fields(call),
    }


def telemetry_fields(call: Call) -> dict[str, Any]:
    """Flatten the per-call accounting onto a record; additive, so an older trace reads them as None.

    ``started_at``/``completed_at`` are chunk stamps and, with the queue kept full across chunks,
    say when a chunk was released rather than when its calls ran; ``elapsed_seconds`` and
    ``first_event_seconds`` are the per-call numbers a barrier analysis or a cost re-pricing needs.
    """
    return {
        "cache_read_input_tokens": call.cache_read_input_tokens,
        "cache_write_input_tokens": call.cache_write_input_tokens,
        "elapsed_seconds": call.elapsed_seconds,
        "first_event_seconds": call.first_event_seconds,
        "attempts": call.attempts,
    }


def trace_records(calls: Sequence[Call], labels: RunLabels) -> list[dict[str, Any]]:
    """Turn calls into the ungraded response records that make up the raw trace.

    One record type only. There is no per-item summary: the readouts are rates over (item, arm,
    model) cells, and a summary record would be a second place for the same numbers to disagree.
    """
    return [response_record(call, labels) for call in calls]


def run_and_grade(
    spec: RunSpec,
    backend: Backend,
    labels: RunLabels,
    raw_path: Path,
    chunk_prompts: int = DEFAULT_CHUNK_PROMPTS,
) -> list[dict[str, Any]]:
    """Sample in chunks, persisting each chunk's ungraded completions, then grade the written file.

    The one entry point a driver should reach for, and it exists to make the ordering unmissable
    rather than to save a line: the completions are the paid material, and nothing that can fail --
    a parser giving up, an unparseable reply, a grader defect discovered later -- may run before
    they are on disk.

    Persistence goes through :func:`run_and_append`, so per-chunk appends, resume by key and the
    continuous queue are all reachable from the door the docstring sends a driver to. Sampling the
    sweep as one call, which this used to do, meant a single prompt exhausting its retries discarded
    every completion already produced and billed and left no trace file at all; a throttle that
    outlives its retries is now a ``call_failed`` row, and a genuine raise still hands over the
    finished part of the chunk in flight first. On a run that finishes, the records are identical
    either way. ``chunk_prompts`` is threaded through so a smoke sends one chunk.

    The graded records are returned rather than written, and the raw path is deliberately left
    alone: ``write_trace`` truncates, so pointing the writer back at ``raw_path`` would overwrite
    the paid material with a derived artifact. A caller that wants one file on disk should write the
    return value to a *different* path, or overwrite ``raw_path`` only after this has returned.
    """
    run_and_append(spec, backend, labels, raw_path, chunk_prompts)
    return grade_trace_file(raw_path, spec.items)


def grade_trace_file(path: Path, items: Sequence[RecoveryItem]) -> list[dict[str, Any]]:
    """Re-grade a stored trace against a corpus, returning the updated records.

    The whole reason grading is a pure function over saved completions: a grader fix is a re-score,
    never a re-run. Records that are not responses pass through untouched.

    Two refusals rather than one, because "the corpus has this id" is weaker than "the corpus still
    has this item". An unknown id raises, since a silently dropped item turns a rate over 60 items
    into a rate over 59 with no symptom. A *reused* id after the item was redrafted raises too, on
    the stored prompt: re-grading is exactly where such drift goes unnoticed, because a grader fix
    is *supposed* to change outcomes. Measured on a redrafted item, the same stored completions
    moved from ``['true', 'flawed_path']`` to ``['other', 'true']`` -- carry down and accuracy up
    together, with the log line still reporting a clean re-grade. Every record already carries its
    prompt, so the check is free. An answer-set-only edit is deliberately *not* caught: applying
    that is what re-grading is for.

    The registered answers are parsed once per file rather than once per reply, which is where the
    grading loop spends about 30% of its CPU on an expression corpus. The filter to closed-answer
    items is mandatory, not tidiness: an execution-graded item has no ``answer_shape``, so an
    unfiltered comprehension would hand every one of them an empty answer set and turn a mixed
    corpus this function handles fine into a file of ``reference_unparseable`` records.

    Parsed the lenient way, matching ``grade_reply``'s own path. A re-grade is exactly where an item
    whose registered value stopped parsing shows up, and the strict parse raises there -- losing the
    whole file to one item defect, after the sampling was paid for, when the honest disposition is a
    per-record abstention naming the item.

    A record
    written before ``stop_reason`` existed reads it as absent, which is what it was, and lands in
    ``no_answer_unknown_stop`` rather than being guessed as ``end_turn``: guessing would relabel
    every truncated reply in that trace as a refusal to answer, and collapsing it into ``no_answer``
    would reach the same wrong number by a different route. The count in the log line is there so a
    trace that is entirely unlabelled says so out loud.
    """
    by_id = {item.item_id: item for item in items}
    parsed_by_id = {
        item.item_id: parse_registered_answers_where_possible(item)
        for item in items
        if item.grading_mode is GradingMode.CLOSED_ANSWER
    }
    records = load_trace(path)
    regraded: list[dict[str, Any]] = []
    for record in records:
        if record.get("record") != RESPONSE:
            regraded.append(record)
            continue
        item_id = record["item_id"]
        item = by_id.get(item_id)
        if item is None:
            msg = (
                f"{path}: record for item {item_id!r} has no item in the corpus of "
                f"{len(by_id)}. Regrading against a partial corpus would silently drop it"
            )
            raise KeyError(msg)
        arm = Arm(record["arm"])
        if item.arms.get(arm) != record["prompt"]:
            msg = (
                f"{path}: the record for item {item_id!r} arm {arm} stores a prompt the current "
                "corpus no longer renders, so this trace predates the item's current text. "
                "Regrading it would score old completions against answers the model never saw"
            )
            raise ValueError(msg)
        graded = grade_reply(
            item,
            record["completion"],
            stop_reason=record.get("stop_reason"),
            registered=parsed_by_id.get(item_id),
        )
        regraded.append({**record, **_grade_fields(graded)})
    unlabelled = sum(
        1 for record in regraded if record.get("outcome") == Outcome.NO_ANSWER_UNKNOWN_STOP
    )
    logger.info(
        "regraded %d records from %s, %d of them with no usable stop reason",
        len(regraded),
        path,
        unlabelled,
    )
    return regraded
