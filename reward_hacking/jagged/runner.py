"""Run items across all five arms through a backend, writing JSONL traces.

Traces follow the repo convention: one self-describing object per line with a ``record``
discriminator, read back by ``load_trace``. Two record types, a per-response one and a per-item
summary carrying the grades.

Token counts and the stop reason are recorded when the backend can supply them and left as ``None``
when it cannot. The shared ``Backend`` protocol returns bare strings, so only a backend exposing
``generate_detailed`` reports them; everything else records ``None`` with the field present rather
than omitting it, and no estimate is ever substituted for a count. ``response_chars`` is always
recorded, because a character count is the trivial baseline every effect has to beat and it costs
nothing to keep.

``stop_reason`` was added after the first traces were written, so a trace from before it exists
reads it as absent rather than as ``end_turn``: ``load_trace`` returns whatever keys a line
carries, and anything reading the field has to use ``get``. Guessing ``end_turn`` for an old record
would relabel every truncated reply in it as a model that declined to answer.

Every record carries a ``transport`` label naming the route that produced it, because the same
hosted model is reachable both through the live Converse API and through batch inference, and any
difference between the two routes would otherwise read as a difference between models.

Every record also carries a ``repeat`` index, because one sample per (item, arm) cell is not a rate.
Denison et al. saw two identical runs differ by 2.4x and 6.7x at rare-event rates, and a single item
here flipped from 1.00 to 0.00 between two identical runs. A trace written before the field existed
reads as ``repeat`` 0, which is what it was.
"""

import hashlib
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reward_hacking.jagged.arms import Arm, render_prompt
from reward_hacking.jagged.graders import grade
from reward_hacking.jagged.items import Item
from reward_hacking.model_backend import Backend, RawResponse, generate_raw, generate_raw_in_chunks
from reward_hacking.trace import RESPONSE, load_trace, write_trace

logger = logging.getLogger(__name__)

ITEM_SUMMARY = "item_summary"

# The repeat a record belongs to when the trace was written before repeats existed.
DEFAULT_REPEAT = 0

# How many prompts :func:`run_and_append` persists at a time, matching the recoverybench twin. The
# chunk is the unit of persistence, not of concurrency: a streaming backend keeps its queue full
# across chunk boundaries, so the size decides how often the trace grows and nothing else.
DEFAULT_CHUNK_PROMPTS = 64


@dataclass(frozen=True, slots=True)
class Call:
    """One rendered prompt and what came back, before grading.

    The five defaulted fields are the per-call accounting a detailed backend reports and a plain one
    cannot (see ``RawResponse``): the prompt-cache split of the input count and the live path's
    latency telemetry. ``None`` is honestly absent, never an estimate.
    """

    item: Item
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


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class Cell:
    """One (item, arm, repeat) coordinate and the prompt it renders to.

    Exists so the asynchronous batch path can render prompts in one process and rebuild calls in
    another without a second copy of the pairing logic. Rendering is deterministic given the same
    items, arms and repeat count, so re-deriving the cells at collection time reproduces exactly
    the list that was submitted -- and the batch backend checks that by comparing prompt digests
    rather than trusting it.
    """

    item: Item
    arm: Arm
    repeat: int
    prompt: str


def render_cells(
    items: Sequence[Item], arms: Sequence[Arm] = tuple(Arm), repeats: int = 1
) -> list[Cell]:
    """Render every (item, arm, repeat) coordinate to a prompt, in submission order.

    Repeat is the outermost loop, so a trace truncated part-way still holds whole repeats rather
    than a partial one for every cell.

    A repeat re-sends an identical prompt: the variation being measured is the sampler's, so nothing
    about the prompt may differ between repeats or the arms stop being comparable.
    """
    if repeats < 1:
        msg = f"repeats must be at least 1, got {repeats}"
        raise ValueError(msg)
    return [
        Cell(item=item, arm=arm, repeat=repeat, prompt=render_prompt(item, arm))
        for repeat in range(repeats)
        for item in items
        for arm in arms
    ]


def calls_from_results(
    cells: Sequence[Cell],
    results: Sequence[RawResponse],
    *,
    started_at: str,
    completed_at: str,
) -> list[Call]:
    """Re-attach item, arm and repeat to index-aligned results.

    The alignment is positional and asserted, never inferred: a results list of the wrong length is
    a transport bug, and letting it through would produce a plausible-looking trace with arms
    attributed to the wrong responses.
    """
    if len(results) != len(cells):
        msg = f"backend returned {len(results)} completions for {len(cells)} prompts"
        raise RuntimeError(msg)
    return [
        _call_for(cell, response, started_at=started_at, completed_at=completed_at)
        for cell, response in zip(cells, results, strict=True)
    ]


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


@dataclass(frozen=True, slots=True)
class SamplingLabels:
    """The two labels a record cannot read off a backend, bundled so a driver passes them once.

    ``model_id`` and ``transport`` are on the ``Backend`` protocol, so :func:`run_and_append` reads
    them off the backend that produces the replies and they cannot disagree with it. These two are
    not: the six backends disagree about what a sampling config even is, and the plain protocol
    declares none. Neither field has a default, for the reason :func:`trace_records` gives -- a cap
    manufactures an apparent capability absence, so a default here is a label that is sometimes
    wrong, and ``None`` (a backend with no cap of its own) has to be said out loud.

    One object rather than two arguments because the chunking driver would otherwise sit outside the
    five-argument budget ruff enforces.
    """

    max_tokens: int | None
    reasoning_effort: str | None


def _sample_chunk(cells: Sequence[Cell], backend: Backend) -> list[Call]:
    """Send one batch of cells and pair the index-aligned results back onto them.

    One batch rather than one call per prompt, because a hosted backend's throughput comes from
    concurrency and the cells are independent by construction. Shared by :func:`run_items` and
    :func:`run_and_append` so the pairing and the timestamps exist once.
    """
    started_at = _now()
    results = generate_raw(backend, [cell.prompt for cell in cells])
    completed_at = _now()
    return calls_from_results(cells, results, started_at=started_at, completed_at=completed_at)


def run_items(
    items: Sequence[Item],
    backend: Backend,
    arms: Sequence[Arm] = tuple(Arm),
    repeats: int = 1,
) -> list[Call]:
    """Render every (item, arm, repeat) triple, send them as one batch, and return them in order.

    In memory and unpersisted, so one prompt exhausting its retries loses every completion already
    billed. A driver wants :func:`run_and_append`; this is the shape the tests assert against.
    """
    cells = render_cells(items, arms, repeats)
    logger.info(
        "jagged run | model=%s transport=%s items=%d arms=%d repeats=%d prompts=%d",
        backend.model_id,
        backend.transport,
        len(items),
        len(arms),
        repeats,
        len(cells),
    )
    return _sample_chunk(cells, backend)


def chunk_cells(cells: Sequence[Cell], chunk_prompts: int) -> list[list[Cell]]:
    """Group cells into chunks of about ``chunk_prompts`` prompts, never splitting a group.

    The budget is prompts, because a prompt is what gets billed, but the atom is a group: an item's
    arms within one repeat go together or the trace they persist is wrong. :func:`trace_records`
    writes one item-summary record per (item, repeat) covering the arms it was handed, and
    ``analysis.summarise`` counts those records as the trial denominator without de-duplicating
    them, deliberately -- two records for one (item, repeat) are how concatenated traces are meant
    to add up. So a group cut across two chunks reports two half-summaries, and the cell it belongs
    to reads one trial as two.

    A group wider than the budget is therefore sent whole rather than cut, and the budget rounds
    down: at five arms and a budget of eight, a chunk is one item, not one and three fifths.
    """
    if chunk_prompts < 1:
        msg = f"chunk_prompts must be at least 1, got {chunk_prompts}"
        raise ValueError(msg)
    groups: dict[tuple[int, str], list[Cell]] = {}
    for cell in cells:
        groups.setdefault((cell.repeat, cell.item.id), []).append(cell)
    chunks: list[list[Cell]] = []
    for group in groups.values():
        if chunks and len(chunks[-1]) + len(group) <= chunk_prompts:
            chunks[-1].extend(group)
        else:
            chunks.append(list(group))
    return chunks


def group_key(item_id: str, repeat: int) -> tuple[str, int]:
    """Name the coordinate a trace is resumed by: the (item, repeat) trial, jagged's persistence atom.

    Deliberately NOT the arm: :func:`chunk_cells` never splits a group and :func:`_whole_group_calls`
    never persists a partial one, because ``analysis.summarise`` counts item-summary records as the
    trial denominator without de-duplicating them, so a group finished across two sittings would
    file two half-summaries and read one trial as two. A resume therefore skips whole groups and
    re-buys whole groups, never single arms. And not the model or the transport either: those are
    the run's labels, and a record under this key whose labels differ is a second experiment in the
    file rather than a trial to skip, which is what :func:`resumable_group_keys` refuses on.
    """
    return (item_id, repeat)


def _prompt_digest(prompt: str) -> str:
    """Digest a prompt for the resume check, so a refusal can name a drift without echoing item text."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _run_label_identity(model_id: str, transport: str, labels: SamplingLabels) -> dict[str, object]:
    """List the four labels every record in a resumed trace must share with THIS run.

    Checked on every record, response and summary, rendered by this run or not: a trace is one
    run's, and a file holding another model's, transport's, effort's or cap's replies is a second
    experiment however few of its groups overlap this run's. All four are JSON-native (strings, an
    int, None), so unlike the hatch probe's sampler dict the stored side needs no normalisation.
    """
    return {
        "model_id": model_id,
        "transport": transport,
        "reasoning_effort": labels.reasoning_effort,
        "max_tokens": labels.max_tokens,
    }


def _refuse_drift(
    record: Mapping[str, Any], expected: Mapping[str, object], *, path: Path, what: str
) -> None:
    """Refuse the whole resume, by name, on the first record whose identity is another run's."""
    stored = {name: record.get(name) for name in expected}
    if "prompt_digest" in expected:
        stored["prompt_digest"] = _prompt_digest(str(record.get("prompt", "")))
    drifted = [
        f"{name}={stored[name]!r} on disk but {value!r} for this run"
        for name, value in expected.items()
        if stored[name] != value
    ]
    if drifted:
        raise ValueError(
            f"refusing to resume over {path}, whose {what} is another run's: {'; '.join(drifted)}. "
            f"A resumed trace must hold the same prompts under the same model, transport, effort "
            f"and cap; appending to it would file two experiments in one trace with every count "
            f"still adding up. Point the run at a fresh path, or move the old trace aside."
        )


def _scan_trace(
    path: Path, wanted_cells: Mapping[tuple[str, str, int], Cell], identity: Mapping[str, object]
) -> tuple[dict[tuple[str, int], set[str]], dict[tuple[str, int], set[str]]]:
    """Read which arms each group has answered and which arms its item summary covers, refusing drift.

    Returns ``(answered, summarised)``, both keyed by :func:`group_key`. Every record is checked
    against the run's labels; a response record for a cell this run renders is checked against the
    prompt digest too, which is what catches a redrafted item under an unchanged id. Two responses
    for one rendered cell, or two summaries for one group, refuse: a rate over them would have a
    denominator nobody can state.
    """
    answered: dict[tuple[str, int], set[str]] = {}
    summarised: dict[tuple[str, int], set[str]] = {}
    for record in load_trace(path):
        kind = record.get("record")
        if kind not in {RESPONSE, ITEM_SUMMARY}:
            continue
        key = group_key(str(record["item_id"]), int(record.get("repeat", DEFAULT_REPEAT)))
        if kind == ITEM_SUMMARY:
            _refuse_drift(record, identity, path=path, what=f"item summary for group {key}")
            if key in summarised:
                raise ValueError(
                    f"{path} holds two item summaries for group {key}; a resumed rate over them "
                    f"would count one trial twice. Move the file aside or dedupe it deliberately."
                )
            summarised[key] = set(record["per_arm"])
            continue
        arm = str(record["arm"])
        cell = wanted_cells.get((key[0], arm, key[1]))
        expected = (
            identity if cell is None else {**identity, "prompt_digest": _prompt_digest(cell.prompt)}
        )
        _refuse_drift(
            record, expected, path=path, what=f"response record for cell {(key[0], arm, key[1])}"
        )
        arms = answered.setdefault(key, set())
        if cell is not None and arm in arms:
            raise ValueError(
                f"{path} holds two response records for cell {(key[0], arm, key[1])}; a resumed "
                f"rate over them would have a denominator nobody can state. Move the file aside or "
                f"dedupe it deliberately."
            )
        arms.add(arm)
    return answered, summarised


def _group_is_resumable(
    key: tuple[str, int],
    arms: set[str],
    *,
    answered: Mapping[tuple[str, int], set[str]],
    summarised: Mapping[tuple[str, int], set[str]],
    path: Path,
) -> bool:
    """Decide one rendered group: resumed, pending, or a file state that needs a human first."""
    stored = summarised.get(key)
    if stored is None:
        if answered.get(key):
            raise ValueError(
                f"{path} holds {len(answered[key])} response records for group {key} but no item "
                f"summary: a chunk write cut short. Re-running the group over them would file two "
                f"answers per cell, so truncate the trace to its last whole chunk, or move it "
                f"aside, before relaunching."
            )
        return False
    if not arms <= stored:
        raise ValueError(
            f"the item summary for group {key} in {path} covers arms {sorted(stored)} but this run "
            f"renders {sorted(arms)}; appending the missing arms would give one trial two "
            f"summaries. Point the run at a fresh path, or move the old trace aside."
        )
    on_disk = answered.get(key, set())
    if not arms <= on_disk:
        raise ValueError(
            f"the item summary for group {key} in {path} covers arms {sorted(stored)} but only "
            f"{sorted(on_disk)} have response records under it, missing {sorted(arms - on_disk)}: "
            f"a summary is written after its responses in one chunk, so this is a hand edit or a "
            f"file torn in the middle, and a reader of the response records would find a hole. "
            f"Move the file aside, or restore the missing records, before relaunching."
        )
    return True


def resumable_group_keys(
    path: Path, cells: Sequence[Cell], *, model_id: str, transport: str, labels: SamplingLabels
) -> set[tuple[str, int]]:
    """Read the (item, repeat) groups already answered in the trace at ``path``, refusing another run's file.

    Resume is by content-derived key, never by execution order, and it is what makes a killed pass
    cheap: before this a relaunch truncated the trace at its first release and re-bought every
    completion the first sitting had paid for, so the partial-chunk hand-over bought a trace to
    inspect rather than a resume. It is also the quietest way to end up with one file holding two
    experiments, so every record in the file is checked against this run's four labels and every
    response record for a cell this run renders against the prompt digest as well; the first
    mismatch refuses the whole resume by name rather than skipping the group.

    A group is resumed only when its item summary is on disk and covers every arm this run renders
    for it. The summary is written after the group's responses in the same chunk, so it is the
    group's completion marker, and its ``per_arm`` keys are the arm set it covers. Four file states
    are refused rather than guessed at (:func:`_group_is_resumable`, :func:`_scan_trace`):
    responses with no summary is a write cut short, and re-running the group over them would file
    two answers per cell; a summary narrower than this run's arm set is an earlier, narrower run,
    and appending the missing arms would give the trial a second summary; a summary with one of
    this run's arms missing its response record under it is a hand edit or a file torn in the
    middle, which a resume would paper over and a reader of the responses would find as a hole;
    two summaries for one group is a duplicated trial. A summary WIDER than this run's arm set is
    resumed: the extra arms are not this run's to judge, and ``summarise`` still reads them. Groups
    this run does not render are left alone, under this run's labels.
    """
    if not path.exists():
        return set()
    wanted_cells = {(cell.item.id, cell.arm.value, cell.repeat): cell for cell in cells}
    wanted_arms: dict[tuple[str, int], set[str]] = {}
    for cell in cells:
        wanted_arms.setdefault(group_key(cell.item.id, cell.repeat), set()).add(cell.arm.value)
    answered, summarised = _scan_trace(
        path, wanted_cells, _run_label_identity(model_id, transport, labels)
    )
    return {
        key
        for key, arms in wanted_arms.items()
        if _group_is_resumable(key, arms, answered=answered, summarised=summarised, path=path)
    }


def run_and_append(  # noqa: PLR0913 - the trailing keyword is a persistence knob, not a sixth input
    cells: Sequence[Cell],
    backend: Backend,
    labels: SamplingLabels,
    path: Path,
    chunk_prompts: int = DEFAULT_CHUNK_PROMPTS,
    *,
    persist_out_of_order: bool = False,
) -> list[Call]:
    """Sample with the queue kept full, appending each chunk of whole (item, repeat) groups as it lands.

    Three properties, each earned by a measured loss, and one opt-in.

    **Resume by key.** Groups whose item summary is already in the trace at ``path`` are skipped
    and counted as resumed, separately from what ran (:func:`resumable_group_keys`, which refuses a
    trace from a different run). A completed run re-launched is a no-op that appends nothing, and a
    killed run re-launched finishes the remainder. Before this a relaunch truncated the trace at its
    first release and re-bought every completion the first sitting had paid for, so the
    partial-chunk hand-over below bought a trace to inspect rather than a resume.

    **Per-chunk persistence.** :func:`run_items` sends a whole sweep as one call and returns only
    when every cell is done, so nothing is on disk and nothing is logged until then, and one raise
    loses the lot. Reproduced on the recoverybench twin with a stand-in backend: 72 prompts with one
    raise at call 50 meant 71 completions were produced and paid for, the driver returned none of
    them, no trace file existed, and one log line covered the whole run. Here every chunk is appended
    the moment it is released, and cells arrive rendered repeat-outermost (:func:`render_cells`), so
    a trace cut short holds whole repeats. jagged's live sweep does not run this way -- ``sweep.py``
    goes through Bedrock batch, whose completions stay re-collectable from the saved handle -- so
    this bounds the loss a live-Converse jagged driver would otherwise inherit from :func:`run_items`.

    **No barrier between chunks.** Through :func:`generate_raw_in_chunks` a streaming backend keeps
    its queue full across chunk boundaries, so a chunk's one slow call holds up its persistence but
    not the calls behind it, while chunks are by default still written in request order -- the
    trace is what the barrier version wrote, two stamps aside: ``started_at`` and ``completed_at``
    are CHUNK stamps, the release of the previous chunk and of this one, and with the queue full
    across chunks they say when a chunk was released rather than when its calls ran; the per-call
    clocks are ``elapsed_seconds`` and ``first_event_seconds`` on each record. If the backend
    raises, the finished part of the chunk in flight is handed over first, and the (item, repeat)
    atom still holds: only the groups that came back whole are persisted (a partial group would
    shard its item summary and read as a trial), the finished arms of a broken group are logged as
    lost, and the relaunch re-buys that group whole. Every other detailed backend falls back to one
    call per chunk, byte for byte the old loop.

    **Out-of-order persistence, opt in.** ``persist_out_of_order`` hands a chunk over the moment
    its own last call lands, whatever chunks before it still wait, so what a wedged head call can
    lose to a SIGKILL or an OOM kill is bounded at the partial chunks in flight rather than
    everything behind the head (:func:`~reward_hacking.model_backend.stream_detailed_in_chunks`
    has the arithmetic). The price is the on-disk order: the trace's groups then sit in completion
    order rather than render order, so the byte-identity with a per-chunk run no longer holds.
    Nothing that reads the trace by identity changes -- the records are the same set, ``summarise``
    reads item summaries by content, and resume keys on the group -- which is why the flag is safe
    to opt into and why it stays off by default: every runner's byte-identity test is written
    against request order. Name it in the run's report when it is on.

    The cells arrive rendered rather than as (items, arms, repeats): the batch path already renders
    them in the caller to digest them, and a driver that has them in hand should not render twice.
    Returns the calls that RAN this invocation, so the in-memory shape matches :func:`run_items` for
    a fresh run; the resumed groups are on disk and in the log line.
    """
    resumed = resumable_group_keys(
        path, cells, model_id=backend.model_id, transport=backend.transport, labels=labels
    )
    pending = [cell for cell in cells if group_key(cell.item.id, cell.repeat) not in resumed]
    chunks = chunk_cells(pending, chunk_prompts)
    logger.info(
        "jagged chunked run | model=%s transport=%s prompts=%d resumed=%d groups pending=%d "
        "chunks=%d of up to %d%s",
        backend.model_id,
        backend.transport,
        len(cells),
        len(resumed),
        len(pending),
        len(chunks),
        chunk_prompts,
        " | persisting chunks OUT OF ORDER" if persist_out_of_order else "",
    )
    started = time.monotonic()
    calls: list[Call] = []
    started_at = _now()
    released = generate_raw_in_chunks(
        backend,
        [[cell.prompt for cell in chunk] for chunk in chunks],
        persist_out_of_order=persist_out_of_order,
    )
    for index, pairs in released:
        chunk = chunks[index]
        completed_at = _now()
        chunk_calls = _whole_group_calls(
            chunk, pairs, started_at=started_at, completed_at=completed_at
        )
        write_trace(
            path,
            trace_records(
                chunk_calls,
                backend.model_id,
                transport=backend.transport,
                max_tokens=labels.max_tokens,
                reasoning_effort=labels.reasoning_effort,
            ),
            append=True,
        )
        calls.extend(chunk_calls)
        logger.info(
            "chunk %d/%d persisted%s | prompts %d/%d ran (%d groups resumed) | %.1fs elapsed",
            index + 1,
            len(chunks),
            "" if len(pairs) == len(chunk) else f" PARTIALLY ({len(chunk_calls)} of {len(chunk)})",
            len(calls),
            len(pending),
            len(resumed),
            time.monotonic() - started,
        )
        started_at = _now()
    return calls


def _whole_group_calls(
    chunk: Sequence[Cell],
    pairs: Sequence[tuple[int, RawResponse]],
    *,
    started_at: str,
    completed_at: str,
) -> list[Call]:
    """Pair a released chunk's responses with their cells, keeping only (item, repeat) groups that are whole.

    A released chunk is whole on every completed run; only a raise hands over a partial one. The
    atom rule from :func:`chunk_cells` applies to what is persisted: a group missing an arm would
    get a partial item summary that ``analysis.summarise`` counts as a trial, so its finished
    completions are dropped here and named in the log, while every whole group in the same chunk
    is kept.
    """
    by_position = dict(pairs)
    groups: dict[tuple[int, str], list[int]] = {}
    for position, cell in enumerate(chunk):
        groups.setdefault((cell.repeat, cell.item.id), []).append(position)
    kept: list[Call] = []
    dropped = 0
    for positions in groups.values():
        if all(position in by_position for position in positions):
            kept.extend(
                _call_for(
                    chunk[position],
                    by_position[position],
                    started_at=started_at,
                    completed_at=completed_at,
                )
                for position in positions
            )
        else:
            dropped += sum(position in by_position for position in positions)
    if dropped:
        logger.error(
            "%d finished completions belonged to (item, repeat) groups the raise left incomplete "
            "and are not persisted: a partial group would shard its item summary into a trial",
            dropped,
        )
    return kept


def trace_records(
    calls: Sequence[Call],
    model_id: str,
    *,
    transport: str,
    max_tokens: int | None,
    reasoning_effort: str | None,
) -> list[dict[str, Any]]:
    """Turn calls into the response and item-summary records that make up a trace.

    One item summary per (item, repeat), not per item: keying it on the item alone would let the
    last repeat overwrite the others and report a run of N repeats as though it had been one.

    ``transport`` is keyword-only and has no default: it names the route the completions came by
    (see ``Backend.transport``), and the same hosted model is now reachable two ways, so a default
    would quietly mislabel whichever route did not get one.

    ``max_tokens`` and ``reasoning_effort`` are keyword-only and defaultless for the same reason,
    and the cap is the label this bench most needs: a reply cut off at the cap carries no move
    markers, so ``did_move`` reads False and the record reports a capability absence that was a
    config choice. ``None`` is a legitimate value -- a backend with no cap of its own -- but it has
    to be passed, because a default is a label that is sometimes wrong, and a run whose cap is
    recorded nowhere can only be recovered by chaining the trace to a handle to an S3 sidecar.

    ``move_concept`` is on both record types, beside ``dimension`` rather than inside ``per_arm``,
    because it is a property of the item: every arm was graded against the same markers. Without it
    a record reports ``did_move`` without saying what the markers were reading, which is
    uninterpretable once the corpus that defined them is gone -- and one corpus here has been
    deleted already.
    """
    records: list[dict[str, Any]] = [
        {
            "record": RESPONSE,
            "item_id": call.item.id,
            "dimension": call.item.dimension,
            "move_concept": call.item.move_concept,
            "arm": call.arm.value,
            "repeat": call.repeat,
            "model_id": model_id,
            "transport": transport,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "prompt": call.prompt,
            "completion": call.completion,
            "reasoning": call.reasoning,
            "response_chars": len(call.completion),
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "stop_reason": call.stop_reason,
            "started_at": call.started_at,
            "completed_at": call.completed_at,
            "cache_read_input_tokens": call.cache_read_input_tokens,
            "cache_write_input_tokens": call.cache_write_input_tokens,
            "elapsed_seconds": call.elapsed_seconds,
            "first_event_seconds": call.first_event_seconds,
            "attempts": call.attempts,
        }
        for call in calls
    ]

    per_item_repeat: dict[tuple[str, int], dict[str, Any]] = {}
    for call in calls:
        graded = grade(call.item, call.completion)
        summary = per_item_repeat.setdefault(
            (call.item.id, call.repeat),
            {
                "record": ITEM_SUMMARY,
                "item_id": call.item.id,
                "dimension": call.item.dimension,
                "move_concept": graded.move_concept,
                "repeat": call.repeat,
                "model_id": model_id,
                "transport": transport,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
                "per_arm": {},
            },
        )
        summary["per_arm"][call.arm.value] = {
            "did_move": graded.did_move,
            "took_wrong_path": graded.took_wrong_path,
            "evidence": list(graded.evidence),
            "response_chars": len(call.completion),
            "output_tokens": call.output_tokens,
            "declares_wrong_path_markers": bool(call.item.wrong_path_markers),
        }
    records.extend(per_item_repeat.values())
    return records
