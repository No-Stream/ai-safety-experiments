r"""The blind-judge loop every sociology pass shares, parameterised on one pass's verdict schema.

Extracted from :mod:`sociology.decoupled_judge` when a second pass needed the same eleven behaviours
under two different verdict schemas at once. Nothing here knows what any field means; a pass
contributes a :class:`VerdictSchema` and inherits the loop.

Each behaviour is a property one of these passes has already paid for:

- **Blindness is checked in production, not only in the tests.** The rubric is authored in a
  gitignored file this code never reviews, so :func:`refuse_leaked_design_labels` runs over the loaded
  rubric and every fixed header before the first call. A judge that can read which cell a reply came
  from reports the cell rather than the reply, and the flag rates inside a cell are exactly what says
  whether the manipulation landed.
- **The judge reads the text the deterministic scan reads.** :func:`judge_input` strips the thinking
  block the same way the scan does, and moves it into the reasoning section under its own heading
  rather than dropping it, because a claim stated only while thinking is still a reading.
- **A rubric edit invalidates the rows judged under the old one.** Resume treats a row as done only if
  it carries a verdict AND the schema's prompt version AND the current rubric's digest, so an edited
  rubric re-judges and the re-judged count is reported separately.
- **Refuse, never guess.** An off-schema verdict is an error with the offending field named, and
  ``evidence`` is required when the schema says so: a verdict nobody can re-read against the reply is
  not checkable.
- **Errored rows retry once**, appended last so last-wins loading sees the retry, and every chunk is
  appended as it lands so a crashed pass keeps its spend. A row that is still errored after the retry
  carries no ``verdict`` and so is never resumed as done: the next pass over the same file re-judges
  exactly the errored keys and leaves every judged one alone.
- **A verbatim quote is allowed to carry backslashes the judge forgot to escape.** The rubric requires
  ``evidence`` to quote the reply verbatim, and a reply that writes its arithmetic in LaTeX puts ``\(``
  and ``\)`` into the quote, which a strict JSON parse refuses as an invalid escape. One production
  pass lost 210 rows this way, 205 of them from the one model on the roster that writes maths that way,
  so the missing rows skewed toward maths-heavy reasoning rather than being spread evenly.
  :func:`escape_lone_backslashes` doubles only the backslashes that begin no valid JSON escape, the parse
  is retried once on that text, and ``judge_lenient_parse`` records on the row that it was. Nothing else
  is forgiven: a reply broken in any other way, or off-schema after the retry, is still an error.

The one fix this extraction carries over the code it came from is in :func:`stratified_subset`. The
old version walked its strata in NAME order, round robin, which meant a request smaller than the
number of strata drew entirely from the alphabetically-first ones -- on the decoupled pass's
``model|cell|action`` strata a 30-row cross-judge subset covered two of nine models. Strata are now
walked in an order derived from a hash of the stratum name, and an optional outer stratum is allocated
across FIRST, so every model contributes before any model contributes twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from games.parsing import THINK_CLOSE, THINK_OPEN, strip_thinking
from reward_hacking.model_backend import (
    STOP_REASON_MAX_TOKENS,
    StreamingBackend,
    is_incomplete_stop_reason,
    stream_detailed_in_chunks,
)
from reward_hacking.trace import refuse_tracked_trace_path
from sociology.records import completion_telemetry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from reward_hacking.model_backend import BedrockCompletion, DetailedBackend

logger = logging.getLogger(__name__)

VERDICT_NONE = "none"
"""The verdict for a reply that never commits to a value, which is a real outcome rather than a miss."""

JUDGED_CARRIES = "verbatim model replies and the judge's quoted evidence"

DEFAULT_JUDGE_CHUNK_SIZE = 32

_JSON_SPAN = re.compile(r"\{.*\}", re.DOTALL)

_JSON_ESCAPE_OR_LONE_BACKSLASH = re.compile(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')
"""A whole valid JSON escape, or else one backslash that begins none.

The valid alternative is tried first and consumes its escaped character too, so ``\\\\(`` reads as a
valid ``\\\\`` followed by a plain parenthesis rather than as a valid pair and then a lone backslash. A
``\\u`` counts as valid only with its four hex digits, because ``\\underline`` is not a code point.
"""

_INVALID_ESCAPE_MESSAGES = frozenset({"Invalid \\escape", "Invalid \\uXXXX escape"})
"""The two messages :mod:`json` raises for a bad escape, verbatim, and the only failures retried."""


class JudgeReplyError(ValueError):
    """A judge reply that could not be parsed into a verdict: wrong shape, enum, bound, or no JSON."""


@dataclass(frozen=True, slots=True)
class JudgeInput:
    """The two text channels the judge is given for one record, plus where the thinking came from."""

    visible_reply: str
    reasoning: str
    had_inline_think: bool


if TYPE_CHECKING:
    # A field's allowed values, and an integer field's inclusive upper bound, both read off the RECORD:
    # a verdict naming an action has to be checked against that row's own two labels, and a verdict
    # naming a number of units against that row's own stock, or a hallucinated value becomes a silent
    # third outcome.
    AllowedValues = Callable[[Mapping[str, Any]], tuple[str, ...]]
    IntegerBound = Callable[[Mapping[str, Any]], int]
    PromptBuilder = Callable[[Mapping[str, Any], JudgeInput, str], str]


@dataclass(frozen=True, slots=True)
class VerdictSchema:
    """One pass's verdict shape, its prompt, and the strings its blindness check must never allow.

    ``enum_fields`` and ``integer_fields`` take the record rather than a static list because both are
    per row: which labels are available, and how many units are on the table. An integer field also
    accepts :data:`VERDICT_NONE`, which is how a reply that named no figure is recorded.

    ``build_prompt`` is the pass's own scaffold, called with the record, the two text channels and the
    rubric text. ``scaffold`` lists every fixed string that builder can emit, so the blindness check
    can read all of them without rendering a prompt.
    """

    name: str
    prompt_version: str
    build_prompt: PromptBuilder
    scaffold: tuple[str, ...]
    design_labels: tuple[str, ...]
    carried_fields: tuple[str, ...]
    enum_fields: Mapping[str, AllowedValues] = field(default_factory=dict)
    bool_fields: tuple[str, ...] = ()
    integer_fields: Mapping[str, IntegerBound] = field(default_factory=dict)
    require_evidence: bool = True

    @property
    def verdict_fields(self) -> tuple[str, ...]:
        """Every field a parsed verdict carries, in a stable order for reports and comparisons."""
        fields = (*self.enum_fields, *self.integer_fields, *self.bool_fields)
        return (*fields, "evidence") if self.require_evidence else fields


@dataclass(frozen=True, slots=True)
class ValidationCase:
    """One hand-authored reply and the whole verdict the judge must return for it.

    ``record`` is a judgeable record carrying its own key, so the validation pass runs through exactly
    the loop production runs through rather than a second path beside it.
    """

    name: str
    record: Mapping[str, Any]
    expected: Mapping[str, Any]


def refuse_leaked_design_labels(text: str, *, what: str, design_labels: Sequence[str]) -> None:
    """Refuse text bound for a judge prompt that names the design, case-insensitively.

    Run over the rubric and the fixed headers rather than over the whole prompt: the reply being judged
    may say anything at all -- a model reasoning about its counterpart routinely writes the same words
    -- and it is what WE tell the judge that has to be blind.
    """
    lowered = text.lower()
    hits = sorted({label for label in design_labels if label.lower() in lowered})
    if hits:
        raise ValueError(
            f"{what} names the design: {hits}. The judge must be blind to the cell, the rung, the "
            f"model and the game, because the flag rates inside a cell are the check that says "
            f"whether the manipulation landed -- a judge that can read the label reports the label. "
            f"Reword the rubric, or rename the design label if the collision is accidental."
        )


def assert_scaffold_is_blind(instructions: str, schema: VerdictSchema) -> None:
    """Run the leak check over the loaded rubric and every fixed header, before any judge call."""
    refuse_leaked_design_labels(
        instructions, what=f"the loaded {schema.name} rubric", design_labels=schema.design_labels
    )
    for section in schema.scaffold:
        refuse_leaked_design_labels(
            section,
            what=f"the {schema.name} judge prompt header {section!r}",
            design_labels=schema.design_labels,
        )


def inline_think_block(reply: str) -> str | None:
    """Return the thinking ``strip_thinking`` cuts out of the text channel, or None if there was none.

    None rather than an empty string for absence, so a block that opened and closed with nothing in it
    still counts as a model that reasoned in the answer channel -- which is the fact
    ``had_inline_think`` exists to record.
    """
    if THINK_CLOSE in reply:
        return reply.rsplit(THINK_CLOSE, 1)[0].split(THINK_OPEN, 1)[-1].strip()
    if THINK_OPEN in reply:
        return reply.split(THINK_OPEN, 1)[1].strip()
    return None


def inline_think_header() -> str:
    """Return the heading a thinking block found in the answer channel is moved under.

    One text for every pass, because a pass whose heading differed would report a different rate of
    reasoning-in-the-answer-channel than its sibling for the same replies.
    """
    return "--- the following was emitted inside the answer channel, before the reply ---"


def judge_input(record: Mapping[str, Any]) -> JudgeInput:
    """Split one reply record into the two channels the judge reads.

    The visible reply is what ``strip_thinking`` leaves, which is what every pass's deterministic scan
    reads its answer out of, so both instruments read one text and their disagreement is a finding
    rather than an artifact of two different inputs.
    """
    reply = str(record.get("reply") or "")
    reasoning = str(record.get("reasoning") or "")
    visible, _truncated_thinking = strip_thinking(reply)
    inline = inline_think_block(reply)
    if inline is not None:
        provider_reasoning = [reasoning] if reasoning.strip() else []
        reasoning = "\n".join([f"{inline_think_header()}\n{inline}", *provider_reasoning])
    return JudgeInput(
        visible_reply=visible, reasoning=reasoning, had_inline_think=inline is not None
    )


def rubric_digest(instructions: str) -> str:
    """Digest one rubric's text, stored per row so a rubric edit is visible in the data."""
    return hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:16]


def escape_lone_backslashes(text: str) -> str:
    r"""Double every backslash that begins no valid JSON escape, and touch nothing else.

    What a judge that quoted LaTeX verbatim into a JSON string meant to write: ``\(x\)`` becomes
    ``\\(x\\)`` and decodes back to the quote as the reply had it, while a ``\"`` or a ``\n`` the judge
    did escape properly is left as the escape it is. Which is also the limit of the repair: a LaTeX
    command that happens to start with an escape letter (``\times``, ``\frac``) is a valid escape as far
    as JSON can tell, so it decodes to a control character plus the rest of the word, on this path and
    on the strict one alike.
    """
    return _JSON_ESCAPE_OR_LONE_BACKSLASH.sub(
        lambda match: match.group(0) if len(match.group(0)) > 1 else "\\\\", text
    )


@dataclass(frozen=True, slots=True)
class ParsedPayload:
    """The decoded JSON object of one judge reply, and whether decoding it needed the lenient retry."""

    payload: dict[str, Any]
    lenient: bool


def _parse_payload(text: str) -> ParsedPayload:
    """Find and decode the one JSON object in a judge reply, refusing anything else.

    Strict first. The one failure retried is an invalid escape, and the retry is one parse of the same
    span with its lone backslashes doubled (:func:`escape_lone_backslashes`); any other decode error,
    and an invalid escape the retry does not cure, is the error it was.
    """
    span = _JSON_SPAN.search(text)
    if span is None:
        raise JudgeReplyError(f"no JSON object in judge reply ({len(text)} chars)")
    try:
        return ParsedPayload(_payload_object(json.loads(span.group(0))), lenient=False)
    except json.JSONDecodeError as error:
        if error.msg not in _INVALID_ESCAPE_MESSAGES:
            raise JudgeReplyError(f"judge reply is not valid JSON: {error}") from error
        strict_error = error
    try:
        payload = json.loads(escape_lone_backslashes(span.group(0)))
    except json.JSONDecodeError as error:
        raise JudgeReplyError(
            f"judge reply is not valid JSON even with its lone backslashes escaped: {error} "
            f"(strict parse: {strict_error})"
        ) from error
    return ParsedPayload(_payload_object(payload), lenient=True)


def _payload_object(payload: Any) -> dict[str, Any]:  # noqa: ANN401 - json.loads hands back Any
    """Refuse a decoded reply that is JSON but not an object."""
    if not isinstance(payload, dict):
        raise JudgeReplyError(f"judge reply parsed to {type(payload).__name__}, not an object")
    return payload


def _parsed_integer_field(name: str, value: object, *, bound: int) -> int | str:
    """Read one integer verdict field, accepting :data:`VERDICT_NONE` and refusing anything else.

    A figure outside the row's own range is an error rather than a clamp, for the same reason the
    deterministic parsers refuse one: a judge that read "set down 40 of 20" has misread the reply, and
    clamping it to 20 would file that misreading as the most generous answer available.
    """
    if value == VERDICT_NONE:
        return VERDICT_NONE
    if isinstance(value, bool) or not isinstance(value, int):
        raise JudgeReplyError(f"{name}={value!r} is neither a whole number nor {VERDICT_NONE!r}")
    if not 0 <= value <= bound:
        raise JudgeReplyError(f"{name}={value!r} is outside this record's range [0, {bound}]")
    return value


@dataclass(frozen=True, slots=True)
class ParsedVerdict:
    """One schema-checked verdict, and whether decoding the reply it came from needed the lenient retry.

    The flag rides beside the verdict rather than inside it so that the verdict dict carries exactly the
    schema's fields, which is what the validation comparisons and the schema-versus-stimulus assertions
    read.
    """

    verdict: dict[str, Any]
    lenient_parse: bool


def parse_verdict(text: str, *, record: Mapping[str, Any], schema: VerdictSchema) -> dict[str, Any]:
    """Parse one judge reply into a verdict dict, refusing anything off-schema rather than guessing."""
    return parse_verdict_detailed(text, record=record, schema=schema).verdict


def parse_verdict_detailed(
    text: str, *, record: Mapping[str, Any], schema: VerdictSchema
) -> ParsedVerdict:
    """Parse one judge reply into a verdict, refusing anything off-schema, and say how it was decoded.

    Every enum and every bound is checked against THIS record rather than against a global list, which
    is what makes a hallucinated label or an impossible figure an error instead of a silent extra
    outcome. The schema check is the same whichever way the JSON decoded: the lenient retry only
    recovers a quote's backslashes, never a field.
    """
    parsed = _parse_payload(text)
    payload = parsed.payload
    verdict: dict[str, Any] = {}
    for name, allowed_for in schema.enum_fields.items():
        allowed = allowed_for(record)
        value = payload.get(name)
        if value not in allowed:
            raise JudgeReplyError(f"{name}={value!r} is not one of {allowed}")
        verdict[name] = str(value)
    for name, bound_for in schema.integer_fields.items():
        verdict[name] = _parsed_integer_field(name, payload.get(name), bound=bound_for(record))
    for name in schema.bool_fields:
        value = payload.get(name)
        if not isinstance(value, bool):
            raise JudgeReplyError(f"{name}={value!r} is not a bool")
        verdict[name] = value
    if schema.require_evidence:
        if "evidence" not in payload:
            raise JudgeReplyError("the verdict carries no evidence field, which is required")
        evidence = payload["evidence"]
        if not isinstance(evidence, str):
            raise JudgeReplyError(f"evidence={evidence!r} is not a string")
        verdict["evidence"] = evidence
    return ParsedVerdict(verdict, lenient_parse=parsed.lenient)


def load_judged(path: Path) -> dict[str, dict[str, Any]]:
    """Load previously judged rows keyed by record key; a re-judged key keeps the LAST row.

    Last-wins is deliberate: the retry pass appends a fresh row for a key whose first attempt errored,
    and the retry is the row every reader should see.
    """
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["key"])] = row
    return rows


def append_judged(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append judged rows, refusing a git-tracked destination from inside the writer itself.

    Inside the writer rather than only at the call sites, because the rows carry verbatim replies and
    the judge's quoted evidence, and an unguarded second call site is exactly how the one leak of this
    class in this repository happened.
    """
    refuse_tracked_trace_path(path, carries=JUDGED_CARRIES)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def judgeable(record: Mapping[str, Any]) -> bool:
    """Whether a record carries any text the judge could read, in either channel."""
    return bool(
        str(record.get("reply") or "").strip() or str(record.get("reasoning") or "").strip()
    )


def judged_under_current_rubric(
    row: Mapping[str, Any], digest: str, *, prompt_version: str
) -> bool:
    """Whether a judged row may be resumed: it parsed, under THIS prompt version and THIS rubric.

    A verdict alone is not enough. The rubric is authored stimulus that gets edited between passes, and
    a verdict produced under the previous wording is an answer to a different question -- kept in the
    file for the record, never counted as this pass's.
    """
    return (
        "verdict" in row
        and row.get("judge_prompt_version") == prompt_version
        and row.get("judge_prompt_digest") == digest
    )


def rows_by_rubric(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count judged rows by the prompt version and rubric digest each one was read under.

    A summary computed over the WHOLE judged file pools every row ever written, which is what makes a
    narrowed re-run's totals honest about the run rather than about the invocation. After a rubric or a
    prompt-version edit it also pools two instruments: the rows this pass re-read carry the new digest and
    the rows it did not select still carry the old one, and one total over both says nothing about either.
    So the breakdown is reported beside the total. One line per version-and-digest pair, spelled
    ``version|digest``, in sorted order so two runs' summaries diff.
    """
    counted: Counter[str] = Counter(
        f"{row.get('judge_prompt_version')}|{row.get('judge_prompt_digest')}" for row in rows
    )
    return dict(sorted(counted.items()))


def judge_telemetry(completion: BedrockCompletion) -> dict[str, Any]:
    """Return the per-call telemetry of one judge call, under the judged row's own ``judge_`` prefix.

    A judged row already spells its call's cost ``judge_input_tokens`` / ``judge_output_tokens`` because
    the row also carries the REPLY's labels, and an unprefixed ``elapsed_seconds`` beside them would
    read as the reply's clock. The five names under the prefix are
    :func:`sociology.records.completion_telemetry`'s, so a cost script joins judged rows and reply rows
    on one spelling: the cache split is what turns a judge pass's list-price figure into a real one (a
    Luna rubric cached on every call bills at a tenth of the uncached rate), and the two clocks plus
    the attempt count are what let a slow pass be read off the artifact instead of a lost log.
    """
    return {f"judge_{name}": value for name, value in completion_telemetry(completion).items()}


def refuse_short_chunk_without_a_stream(
    backend: DetailedBackend,
    pairs: Sequence[tuple[int, BedrockCompletion]],
    *,
    expected: int,
    chunk: int,
) -> None:
    """Refuse a released chunk that is short unless the backend streams, in which case a raise follows.

    :func:`~reward_hacking.model_backend.stream_detailed_in_chunks` hands a chunk over short in exactly
    one situation: a streaming backend raised, and the finished part of each unfinished chunk is
    released ahead of the error. Its fallback branch, one ``generate_detailed`` per chunk, does not
    check the length it gets back, so from any other backend a short chunk means a results list of
    the wrong length -- the transport bug ``zip(strict=True)`` used to catch before a row was written,
    and which would otherwise file the rows short and report the pass whole. The check belongs in
    that fallback branch and moves there when ``model_backend.py`` is next free.
    """
    if len(pairs) == expected or isinstance(backend, StreamingBackend):
        return
    raise RuntimeError(
        f"the judge backend returned {len(pairs)} completions for the {expected} prompts of chunk "
        f"{chunk + 1}; a results list of the wrong length is a transport bug, and only a streaming "
        f"backend hands a chunk over short, ahead of a raise"
    )


def _row_for(  # noqa: PLR0913 - one keyword per piece of the row's provenance
    record: Mapping[str, Any],
    *,
    completion: BedrockCompletion,
    judge_model_id: str,
    instructions: str,
    schema: VerdictSchema,
    had_inline_think: bool,
) -> dict[str, Any]:
    """Build one judged row: carried reply fields, judge provenance, and the verdict or the error."""
    row: dict[str, Any] = {"key": record["key"]}
    row.update({name: record.get(name) for name in schema.carried_fields})
    row.update(
        {
            "judge_model_id": judge_model_id,
            "judge_schema": schema.name,
            "judge_prompt_version": schema.prompt_version,
            "judge_prompt_digest": rubric_digest(instructions),
            "had_inline_think": had_inline_think,
            "judge_stop_reason": completion.stop_reason,
            "judge_input_tokens": completion.usage.input_tokens,
            "judge_output_tokens": completion.usage.output_tokens,
            **judge_telemetry(completion),
            "judge_raw_reply": completion.text,
            "judged_at": datetime.now(UTC).isoformat(),
        }
    )
    # max_tokens joins the transport-incomplete reasons: a truncated verdict must never be kept.
    incomplete = completion.stop_reason == STOP_REASON_MAX_TOKENS or (
        completion.stop_reason is not None and is_incomplete_stop_reason(completion.stop_reason)
    )
    if incomplete:
        row["judge_error"] = f"incomplete reply: stop_reason={completion.stop_reason}"
        return row
    try:
        parsed = parse_verdict_detailed(completion.text, record=record, schema=schema)
    except JudgeReplyError as error:
        row["judge_error"] = str(error)
        return row
    row["verdict"] = parsed.verdict
    row["judge_lenient_parse"] = parsed.lenient_parse
    return row


def judge_records(  # noqa: PLR0913 - trailing keyword-only knobs, each a seam
    backend: DetailedBackend,
    records: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    instructions: str,
    schema: VerdictSchema,
    chunk_size: int = DEFAULT_JUDGE_CHUNK_SIZE,
    retry_errored: bool = True,
) -> dict[str, int]:
    """Judge every judgeable record not already judged under this rubric; return the counts.

    Resume is by record key against ``out_path`` AND by rubric: a row counts as done only if
    :func:`judged_under_current_rubric` accepts it, so rows carrying a verdict from an earlier rubric
    are re-judged and counted as ``stale_rejudged`` rather than silently pooled with the current ones.
    Rows that errored are retried once at the end of the pass (a fresh call, appended last so
    last-wins loading sees the retry). Appending per chunk means a crash keeps its spend.

    Within a pass the backend's queue is kept full across chunk boundaries
    (:func:`~reward_hacking.model_backend.stream_detailed_in_chunks`): on the live backend a chunk's
    one slow call no longer idles the workers behind it, while what lands on disk is exactly what the
    per-chunk loop wrote, in the same order -- and on any other backend the loop IS the per-chunk loop.
    On a raise the finished part of the chunk in flight is appended first (every row is keyed, so the
    relaunch resumes the rest) and then the error propagates; a short chunk from a backend without a
    stream refuses before a row of it is written (:func:`refuse_short_chunk_without_a_stream`).
    """
    assert_scaffold_is_blind(instructions, schema)
    digest = rubric_digest(instructions)
    existing = load_judged(out_path)
    done = {
        key
        for key, row in existing.items()
        if judged_under_current_rubric(row, digest, prompt_version=schema.prompt_version)
    }
    stale = {key for key, row in existing.items() if "verdict" in row and key not in done}
    pending = [r for r in records if judgeable(r) and str(r["key"]) not in done]
    skipped = sum(1 for r in records if not judgeable(r))
    counts = {
        "records": len(records),
        "skipped_empty": skipped,
        "already_judged": len(done),
        "stale_rejudged": sum(1 for r in pending if str(r["key"]) in stale),
    }
    logger.info(
        "%s judge: %d records, %d empty, %d already judged, %d stale to re-judge, %d to judge",
        schema.name,
        len(records),
        skipped,
        len(done),
        counts["stale_rejudged"],
        len(pending),
    )
    judge_model_id = str(getattr(backend, "model_id", "unknown"))

    def run_pass(batch: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        errored: list[Mapping[str, Any]] = []
        chunks = [
            list(batch[start : start + chunk_size]) for start in range(0, len(batch), chunk_size)
        ]
        inputs = [[judge_input(record) for record in chunk] for chunk in chunks]
        prompts = [
            [
                schema.build_prompt(record, given, instructions)
                for record, given in zip(chunk, chunk_inputs, strict=True)
            ]
            for chunk, chunk_inputs in zip(chunks, inputs, strict=True)
        ]
        judged = 0
        for index, pairs in stream_detailed_in_chunks(backend, prompts):
            refuse_short_chunk_without_a_stream(
                backend, pairs, expected=len(chunks[index]), chunk=index
            )
            rows: list[dict[str, Any]] = []
            for position, completion in pairs:
                record = chunks[index][position]
                row = _row_for(
                    record,
                    completion=completion,
                    judge_model_id=judge_model_id,
                    instructions=instructions,
                    schema=schema,
                    had_inline_think=inputs[index][position].had_inline_think,
                )
                if "judge_error" in row:
                    errored.append(record)
                rows.append(row)
            append_judged(out_path, rows)
            judged += len(rows)
            logger.info(
                "%s judge: %d/%d judged this pass%s, %d errored so far",
                schema.name,
                judged,
                len(batch),
                ""
                if len(pairs) == len(chunks[index])
                else f", chunk {index + 1} PARTIAL ahead of a raise",
                len(errored),
            )
        return errored

    errored = run_pass(pending)
    counts["errored_first_attempt"] = len(errored)
    if retry_errored and errored:
        logger.info("%s judge: retrying %d errored records once", schema.name, len(errored))
        counts["errored_after_retry"] = len(run_pass(errored))
    else:
        counts["errored_after_retry"] = len(errored)
    counts["judged"] = len(pending)
    return counts


def validation_key(name: str) -> str:
    """Return the record key one validation reply is judged and resumed under."""
    return f"validation|{name}"


def validate_judge(
    backend: DetailedBackend,
    cases: Sequence[ValidationCase],
    out_path: Path,
    *,
    instructions: str,
    schema: VerdictSchema,
) -> dict[str, Any]:
    """Judge the hand-authored validation replies and report every disagreement by name and field.

    Every registered field is compared rather than the headline one: a judge that got the outcome right
    and a flag wrong is exactly the failure this check exists to catch, because the flag rates are what
    the pass reads. Nothing is smoothed -- each miss names the reply, its record key, the field, what
    was registered and what came back, so the report can be read straight out of the summary.
    """
    if not cases:
        raise ValueError("the stimulus file carries no validation replies to judge")
    counts = judge_records(
        backend,
        [case.record for case in cases],
        out_path,
        instructions=instructions,
        schema=schema,
        retry_errored=True,
    )
    judged = load_judged(out_path)
    misses: list[dict[str, Any]] = []
    unparsed: list[dict[str, Any]] = []
    agreed = 0
    for case in cases:
        key = str(case.record["key"])
        row = judged[key]
        verdict = row.get("verdict")
        if not isinstance(verdict, dict):
            unparsed.append({"name": case.name, "key": key, "judge_error": row.get("judge_error")})
            continue
        case_misses = [
            {
                "name": case.name,
                "key": key,
                "field": name,
                "expected": expected,
                "got": verdict.get(name),
            }
            for name, expected in case.expected.items()
            if verdict.get(name) != expected
        ]
        if case_misses:
            misses.extend(case_misses)
        else:
            agreed += 1
    report: dict[str, Any] = {
        "validated": len(cases),
        "agreed": agreed,
        "misses": misses,
        "unparsed": unparsed,
        "stale_rejudged": counts["stale_rejudged"],
        "already_judged": counts["already_judged"],
    }
    logger.info("%s judge validation: %s", schema.name, json.dumps(report))
    return report


def _stratum_order(names: Sequence[str]) -> list[str]:
    """Walk strata in an order derived from their names' hashes rather than alphabetically.

    Deterministic, so a resumed pass re-picks what it picked before, and unbiased with respect to the
    names, which is the half the old name-ordered walk got wrong: a request smaller than the number of
    strata drew entirely from the alphabetically-first ones.
    """
    return sorted(names, key=lambda name: hashlib.sha256(name.encode()).hexdigest())


def _round_robin(buckets: Mapping[str, list[Mapping[str, Any]]]) -> list[Mapping[str, Any]]:
    """Flatten strata by taking one from each in turn, so every stratum contributes before any twice."""
    order = _stratum_order(list(buckets))
    flattened: list[Mapping[str, Any]] = []
    depth = 0
    while True:
        added = False
        for name in order:
            members = buckets[name]
            if depth < len(members):
                flattened.append(members[depth])
                added = True
        if not added:
            return flattened
        depth += 1


def stratified_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    n: int,
    stratum: Callable[[Mapping[str, Any]], str],
    outer_stratum: Callable[[Mapping[str, Any]], str] | None = None,
) -> list[Mapping[str, Any]]:
    """Pick ``n`` records spread over the strata, the outer stratum allocated across first.

    Two levels rather than one because the level that matters most is the coarsest. A cross-judge
    subset is read per model, so every model has to appear; with the model folded into a single
    stratum key, a request smaller than the number of strata covered only some of them -- observed on
    the decoupled pass, where 30 rows over ``model|cell|action`` strata reached two models of nine.
    Passing ``outer_stratum`` allocates round robin over models first and only then over cells inside
    each model.

    Members within a stratum are ordered by a hash of their own record key rather than by position, so
    the same subset comes back whatever order the reply files were read in.
    """
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for record in records:
        outer = "" if outer_stratum is None else outer_stratum(record)
        grouped.setdefault(outer, {}).setdefault(stratum(record), []).append(record)
    for buckets in grouped.values():
        for members in buckets.values():
            members.sort(key=lambda record: hashlib.sha256(str(record["key"]).encode()).hexdigest())
    within_outer = {name: _round_robin(buckets) for name, buckets in grouped.items()}
    picked: list[Mapping[str, Any]] = []
    order = _stratum_order(list(within_outer))
    depth = 0
    while len(picked) < n:
        added = False
        for name in order:
            members = within_outer[name]
            if depth < len(members):
                picked.append(members[depth])
                added = True
                if len(picked) == n:
                    return picked
        if not added:
            break
        depth += 1
    return picked
