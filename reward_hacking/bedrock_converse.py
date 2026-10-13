"""The AWS Bedrock Converse backend and its helpers.

Split out of ``model_backend``, which re-exports every name here.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )


import urllib3.exceptions

from reward_hacking.backend_types import (
    STOP_REASON_DEADLINE_EXCEEDED,
    BedrockCompletion,
    BedrockSamplingConfig,
    IncompleteConverseStreamError,
    TokenUsage,
    _error_response,  # pyright: ignore[reportPrivateUsage]
    call_failed_stop_reason,
    client_error_code,
    is_incomplete_stop_reason,
)

logger = logging.getLogger("reward_hacking.model_backend")


def _attempts_from_metadata(metadata: Mapping[str, Any] | None) -> int | None:
    """Turn botocore's ``ResponseMetadata.RetryAttempts`` (retries, not attempts) into attempts.

    botocore records the number of *retries* on the parsed response and on a ``ClientError``'s
    ``response``, so one request that was accepted first time reads ``RetryAttempts: 0`` and is one
    attempt. ``None`` when the metadata is absent -- a stand-in client, or a transport failure that
    never produced a response -- rather than a guessed 1.
    """
    if metadata is None:
        return None
    retries = metadata.get("RetryAttempts")
    return None if retries is None else int(retries) + 1


DEFAULT_BEDROCK_REGION = "us-west-2"
DEFAULT_BEDROCK_CONCURRENCY = 16


BEDROCK_PROFILE_ENV = "REWARD_HACKING_BEDROCK_PROFILE"

PROFILE_FROM_ENVIRONMENT = "<from-environment>"
"""The ``profile`` default: resolve the named AWS profile from :data:`BEDROCK_PROFILE_ENV`.

A sentinel rather than the profile name itself because a profile name is a machine-local value and
this repository is public, and a sentinel rather than ``None`` because ``None`` already means
something different and load-bearing here: skip the named profile entirely and take the ambient
credential chain, which is what a Batch container holding an instance role has. Shaped so it cannot
collide with a real profile name.
"""


def bedrock_profile() -> str:
    """Return the named AWS profile the Converse path authenticates with.

    Read from the environment rather than defaulted in source, for the reason
    ``reward_hacking/bedrock_batch.py`` gives at length for the batch path's account, role and
    bucket: this remote is public, so an account-specific identifier committed here is published.
    Unset raises rather than falling back to anything, because both fallbacks are worse than a
    crash -- a committed name republishes the value, and the ambient chain on this box fails
    credential resolution outright, which surfaces as an authentication error far from its cause.
    """
    profile = os.environ.get(BEDROCK_PROFILE_ENV, "").strip()
    if not profile:
        raise RuntimeError(
            f"{BEDROCK_PROFILE_ENV} is not set; it must name an AWS profile with Bedrock access. "
            "It is read from the environment rather than defaulted in code because this repository "
            "is public. Export it in your shell or a local gitignored env file, or pass "
            "profile=None to use the ambient credential chain (an instance role, for example)."
        )
    return profile


@dataclass(frozen=True, slots=True)
class ReasoningDialect:
    """How one Bedrock model family spells reasoning effort, and which levels it accepts.

    The two known families are mutually incompatible, and one direction of the mistake is silent:
    GPT-OSS accepts an unrecognised request field without error, so handing it the GPT-5.6-shaped
    nested payload drops the setting on the floor and runs the model at its default effort. A
    cross-model comparison built that way would credit the difference to the model. GPT-OSS also
    accepts an out-of-range effort *value* without error, which is why the ladder is checked here
    rather than left to the API to reject.

    ``REASONING_DIALECTS`` keys these by the substring identifying the family inside a Bedrock
    model id. The levels recorded there are quoted from the API's own validation errors, so they
    are authoritative rather than remembered.
    """

    nested: bool
    levels: frozenset[str]

    def request_fields(self, effort: str) -> dict[str, object]:
        """Build the ``additionalModelRequestFields`` payload carrying this effort level."""
        if effort not in self.levels:
            raise ValueError(
                f"reasoning effort {effort!r} is not accepted by this model family; "
                f"expected one of {sorted(self.levels)}"
            )
        if self.nested:
            return {"reasoning": {"effort": effort}}
        return {"reasoning_effort": effort}


REASONING_DIALECTS: dict[str, ReasoningDialect] = {
    "gpt-oss": ReasoningDialect(
        nested=False,
        levels=frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"}),
    ),
    "gpt-5.6": ReasoningDialect(
        nested=True, levels=frozenset({"none", "low", "medium", "high", "xhigh", "max"})
    ),
}

CONVERSE_STOP_REFUSING_FAMILIES: tuple[str, ...] = ("gpt-oss", "gpt-5.6")
"""Model families whose Converse endpoint rejects ``stopSequences`` outright, probed live.

Per-family, not per-API: Converse translates the field per provider, and both OpenAI families on
this roster reject it with ``ValidationException: This model doesn't support the stopSequences
field`` (gpt-oss-120b and gpt-5.6-luna, probed live 2026-08-24), while Anthropic accepts it and
returns ``stopReason "stop_sequence"`` with the matched text stripped (claude-sonnet-5, same
probe). Keyed on the substring inside a model id the way :data:`REASONING_DIALECTS` is.

A DENY list rather than an allowlist, deliberately: an unprobed family is sent the field and, if
it too refuses, fails loudly on the first call with Bedrock's own clear message -- which is the
correct default here, because the silent alternative (dropping the stop for anything unlisted)
would sample a harness run without its stop and report nothing. A family on this list gets the
stop *skipped by the caller that wanted it*, which is a behavioural degradation and must be
warned about by that caller, never absorbed here.
"""


def converse_supports_stop_sequences(model_id: str) -> bool:
    """Whether this model's Converse endpoint is expected to accept ``stopSequences``.

    "Expected" per :data:`CONVERSE_STOP_REFUSING_FAMILIES`: an unlisted family reads ``True`` and
    a wrong ``True`` fails loudly on the first call rather than silently.
    """
    return all(family not in model_id for family in CONVERSE_STOP_REFUSING_FAMILIES)


CONVERSE_TEMPERATURE_REFUSING_FAMILIES: tuple[str, ...] = ("gpt-5.6",)
"""Model families whose Converse endpoint rejects the ``temperature`` field outright, probed live.

``ValidationException: This model doesn't support the temperature field. Remove temperature and
try again.`` (gpt-5.6-luna, probed live 2026-08-31 by the hatch probe's smoke). The upstream
reason is OpenAI's own API contract for its reasoning models: they sample at a fixed temperature
of 1.0 and accept no override, so omitting the field asks for the same distribution a caller
pinning 1.0 wanted -- but the request must not carry it. gpt-oss is NOT here: both gpt-oss sizes
accepted ``temperature`` 1.0 across the 2026-08-25 production batches.

Same deny-list contract as :data:`CONVERSE_STOP_REFUSING_FAMILIES`: an unprobed family is sent the
field and a refusal fails loudly on the first call with the service's own message, and a caller
that drops the field because of this list owns recording that it did.
"""


def converse_supports_temperature(model_id: str) -> bool:
    """Whether this model's Converse endpoint is expected to accept ``temperature``.

    "Expected" per :data:`CONVERSE_TEMPERATURE_REFUSING_FAMILIES`: an unlisted family reads
    ``True`` and a wrong ``True`` fails loudly on the first call rather than silently.
    """
    return all(family not in model_id for family in CONVERSE_TEMPERATURE_REFUSING_FAMILIES)


def _resolve_reasoning_dialect(model_id: str) -> ReasoningDialect:
    """Find the reasoning dialect for a model id, refusing to guess when the family is unknown.

    Guessing is the silent-failure case. The wrong shape is ignored without error by GPT-OSS, and
    families with no effort control at all (Claude takes a thinking-token budget instead) would
    either reject the field or ignore it depending on the vendor. Raising means a mis-specified
    run dies at construction rather than reporting numbers nobody can interpret.
    """
    for family, dialect in REASONING_DIALECTS.items():
        if family in model_id:
            return dialect
    raise ValueError(
        f"no known reasoning-effort dialect for model {model_id!r}; known families are "
        f"{sorted(REASONING_DIALECTS)}. Leave reasoning_effort unset to call at default effort."
    )


DEFAULT_BEDROCK_DEADLINE_SECONDS = 3_600.0
"""Total wall clock one streamed Converse call may spend before it is abandoned mid-generation.

The number a socket read timeout cannot express. ``read_timeout`` bounds the *gap between reads*, so
a model that keeps emitting tokens can run arbitrarily long inside it: MiniMax M2.5 on hard arms
runs 2,900-3,000+ seconds per call, and 11 of 24 calls in one pass died at 2,700 seconds. Those
deaths correlate with reasoning length, which is usually the variable an experiment is manipulating,
so the surviving denominator is *biased* rather than merely small -- which is why a call past this
deadline comes back as a partial record (:data:`STOP_REASON_DEADLINE_EXCEEDED`) rather than as a
raise that loses it.

Set generously past the slowest call on record rather than at a value that looks tidy: a deadline
that fires on a legitimately long generation converts a real measurement into a partial record,
which is the same bias in the other direction. Tighten it per run when a model's own pace is known.
"""

DEFAULT_BEDROCK_READ_TIMEOUT_SECONDS = 900
"""Longest silence tolerated between two reads of one call's socket.

Unchanged in value from the non-streaming client, but a different quantity now: against
``converse_stream`` this bounds the gap between two events of one reply rather than the whole call,
because the whole call is bounded by :data:`DEFAULT_BEDROCK_DEADLINE_SECONDS` instead. A streamed
reply emits deltas continuously, so tightening this is now the safe way to detect a genuinely stuck
call -- which it was not before, when a slow-but-alive model and a hung socket looked identical.
Left generous by default because time-to-first-event on a reasoning model has not been measured
here.
"""

DEFAULT_BEDROCK_MAX_ATTEMPTS = 5
"""Cap on botocore's retries for one call, and the reason the cap exists at all.

Bedrock's transient 500s arrive both as ``InternalServerException`` and as a bare ``ClientError``
with code ``"500"`` that maps to no modelled exception class, so retrying is botocore's job rather
than ours (see :class:`BedrockBackend`). The cap is what keeps that from being unbounded: each
attempt can burn a full read timeout before it fails, and retries cover the request handshake only
-- once the response headers arrive, the event stream is consumed outside botocore's retry loop
(read from ``botocore.eventstream.EventStream``, whose iteration happens after the API call
returns), so a mid-stream failure is never retried. Worst-case wall clock for one call is therefore
about ``max_attempts * read_timeout`` waiting for the stream to start, plus ``deadline`` reading it,
plus one last ``read_timeout`` if it goes silent at the end. :class:`BedrockBackend` logs that
figure at construction, because a bound nobody can see is a bound nobody reports.
"""


TRANSIENT_CLIENT_ERROR_CODES: frozenset[str] = frozenset(
    {
        # Retried by botocore's adaptive mode and surfacing here only once it gave up.
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelNotReadyException",
        "InternalServerException",
        # Model-side per-call failures botocore does NOT retry: one dead sample on first occurrence.
        "ModelErrorException",
        "ModelTimeoutException",
        # A 5xx that carried no service code at all: botocore stamps the HTTP status on it as the
        # code (``_parse_error_from_http_status``) and retries it by status before it reaches here.
        "500",
        "502",
        "503",
        "504",
    }
)
"""The ``ClientError`` codes a live call may die of without that being a bug in the run.

A CLOSED enumeration, and the reason it is one is the same reason the transport list on
:class:`BedrockBackend` is: widening it toward ``except ClientError`` would absorb a rejected
payload, an unknown model id or a credentials failure into hundreds of empty ``call_failed`` rows
that read downstream as a model with nothing to say. ``ValidationException``,
``AccessDeniedException`` and ``ResourceNotFoundException`` are deliberately absent and still
crash the batch on the first call. Three groups are on it, each a failure of one call rather than
of the request:

- The four codes botocore's adaptive retry already handles (a throttle, the service unavailable, a
  model still loading, an internal error), which reach this seam only after five attempts failed.
- ``ModelErrorException`` (HTTP 424, "an error while processing the model") and
  ``ModelTimeoutException`` (HTTP 408, the service's own per-call timeout), which botocore does
  not retry, so the first occurrence is the one recorded. They are model-side faults on one
  prompt -- the service model files neither as a client fault -- and a ``ModelErrorException``
  was one of the chunk killers the audit named; a leg should not die to one.
- The bare HTTP status strings ``"500"``, ``"502"``, ``"503"`` and ``"504"``. When a 5xx arrives
  with no service code in the body (a generic gateway page, a proxy), botocore's parser puts the
  status code in ``Error.Code`` and its transient checker retries those four statuses; the
  module already documents that Bedrock's transient 500s take this shape as well as
  ``InternalServerException``.

Before this list existed such a failure killed the chunk it was in and discarded every sibling reply
already billed -- up to ``chunk - 1`` completions, three recorded incidents in three weeks (a throttle
on the 70th call would have discarded 72; 7 of 8 completed results were discarded by one raising
task; 22 billed Sonnet replies died with one mid-stream error before ``EventStreamError`` was
isolated). Read timeouts are not a code: they arrive as botocore's and urllib3's timeout classes,
which the transport list already names.
"""


def _join_text_blocks(blocks: Sequence[Mapping[str, Any]]) -> str:
    """Join the answer text only, so a reasoning block is never scored as part of the answer.

    A union member that is absent and one that is present-but-``null`` mean the same thing -- this
    block is not of that kind -- and both spellings occur for real. botocore drops unset members
    from a live Converse response, so the live path sees ``{"text": ...}`` alone; the batch
    inference result file in S3 serialises *every* member of the content-block union with an
    explicit ``null``, so a real batch block carries ``text``, ``reasoningContent``, ``toolUse``
    and nine more keys at once. Keying on key presence therefore reads ``None`` as answer text and
    dies on the join, which is exactly what the first parse of a real batch record did.
    """
    return " ".join(text for block in blocks if (text := block.get("text")) is not None)


def _join_reasoning_blocks(blocks: Sequence[Mapping[str, Any]]) -> str:
    """Join whatever readable reasoning came back, skipping encrypted and absent traces alike.

    Null-tolerant at every level for the reason given on ``_join_text_blocks``: on the batch path
    ``reasoningContent`` is present and ``None`` on the answer block, and ``reasoningText`` is
    present and ``None`` on Luna's redacted trace.
    """
    texts: list[str] = []
    for block in blocks:
        reasoning_content = block.get("reasoningContent")
        if reasoning_content is None:
            continue
        reasoning_text = reasoning_content.get("reasoningText")
        if reasoning_text is None:
            continue
        text = reasoning_text.get("text")
        if text is not None:
            texts.append(text)
    return " ".join(texts)


# The input-token counters, grouped so that each group is one quantity under all of its spellings.
_INPUT_TOKEN_FIELDS: tuple[tuple[str, ...], ...] = (
    ("inputTokens", "inputTokenCount"),
    ("cacheReadInputTokens", "cacheReadInputTokenCount"),
    ("cacheWriteInputTokens", "cacheWriteInputTokenCount"),
)


def _alias_value(usage: Mapping[str, Any], names: Sequence[str]) -> int:
    """Read one token counter that Bedrock spells several ways, treating null as absent.

    A batch result file carries ``cacheReadInputTokens`` beside ``cacheReadInputTokenCount``, and
    every batch job in this account reports both as null, so which one is authoritative could not
    be settled from data. Reducing by max encodes the assumption that they are aliases: aliases
    agree, so max is their shared value, and where only one is populated max picks it. Summing
    would double-count a cached prompt and inflate input cost with no symptom. The warning below is
    what turns the assumption into something we would hear about if it is wrong.

    Every counter's group has to list both spellings, symmetrically. The bare names are what a
    per-record ``usage`` block uses and the ``...Count`` names are the job manifest's vocabulary, so
    a group carrying only one of them reads zero for a mapping written in the other -- and because
    the warning fires only when two *present* values disagree, an absent alias undercounts in
    silence. That is exactly the half-count these alias groups exist to prevent.
    """
    values = [value for name in names if (value := usage.get(name)) is not None]
    if len({value for value in values if value}) > 1:
        logger.warning(
            "token counters %s disagree (%s); they were assumed to be aliases, so the largest is "
            "used and the input-token total may be wrong",
            list(names),
            values,
        )
    return max(values, default=0)


def _sum_usage(usage: Mapping[str, Any]) -> TokenUsage:
    """Sum all three input-token fields, because the prompt cache moves input between them.

    GPT-5.6 Luna reported a ~1,700-token prompt as ``inputTokens: 2`` alongside
    ``cacheWriteInputTokens: 1619``, then flipped to ``cacheReadInputTokens`` on a later run
    because the cache outlives the process that filled it. Reading ``inputTokens`` alone
    understates that prompt by nearly three orders of magnitude. GPT-OSS does no caching and
    reports zero for both cache fields, so summing is correct there too.
    """
    uncached, cache_read, cache_write = (
        _alias_value(usage, names) for names in _INPUT_TOKEN_FIELDS
    )
    return TokenUsage(
        input_tokens=uncached + cache_read + cache_write,
        output_tokens=_alias_value(usage, ("outputTokens", "outputTokenCount")),
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
    )


def converse_inference_config(sampling: BedrockSamplingConfig) -> dict[str, object]:
    """Build the Converse ``inferenceConfig``, omitting every knob the run did not ask for.

    Omission is the verified request shape across the model roster, and it is load-bearing rather
    than tidy: several hosted models reject a field they do not honour, and Anthropic models reject
    ``temperature`` and ``topP`` when both are present -- a batch job in this account came back
    ``Completed`` with 500 of 500 records dead on exactly that.

    ``maxTokens`` is the exception and is written unconditionally. That is this code's choice, not
    the API's: probed live 2026-09-03 on gpt-oss-20b, both ``Converse`` and ``ConverseStream``
    accept a request with no ``inferenceConfig`` at all and run at the model's own output ceiling
    (:data:`DEFAULT_BEDROCK_MAX_TOKENS` has the numbers and what still keeps the field in).
    """
    config: dict[str, object] = {"maxTokens": sampling.max_tokens}
    if sampling.temperature is not None:
        config["temperature"] = sampling.temperature
    if sampling.top_p is not None:
        config["topP"] = sampling.top_p
    if sampling.stop_sequences:
        config["stopSequences"] = list(sampling.stop_sequences)
    return config


def converse_extra_request_fields(
    model_id: str, sampling: BedrockSamplingConfig
) -> dict[str, object] | None:
    """Build ``additionalModelRequestFields`` for this model, or ``None`` when nothing is asked."""
    if sampling.reasoning_effort is None:
        return None
    return _resolve_reasoning_dialect(model_id).request_fields(sampling.reasoning_effort)


def converse_request(
    prompt: str,
    inference_config: Mapping[str, object],
    extra_request_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Render one prompt into a Converse request body, without ``modelId``.

    The one place a prompt becomes a Bedrock payload. The live backend hands this to
    ``client.converse_stream(modelId=..., **request)``; the batch backend writes the identical dict
    as a record's ``modelInput``, which is exactly what ``modelInvocationType="Converse"`` expects.
    A second renderer would let the two transports drift, and a drift there shows up as a
    model-behaviour difference rather than as a bug.

    The same three keys serve both ``Converse`` and ``ConverseStream``: their request shapes in the
    installed botocore service model share ``messages``, ``inferenceConfig`` and
    ``additionalModelRequestFields``, so switching the live path to streaming did not fork this
    renderer and the batch path's records did not change.
    """
    request: dict[str, object] = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": dict(inference_config),
    }
    if extra_request_fields is not None:
        request["additionalModelRequestFields"] = dict(extra_request_fields)
    return request


def parse_converse_output(body: Mapping[str, Any]) -> BedrockCompletion:
    """Split a Converse response body into answer, readable reasoning and token usage.

    The counterpart to ``converse_request``, and shared for the same reason: a batch result file's
    ``modelOutput`` *is* this body, so both transports must parse it with the same code or a
    parsing difference becomes a finding about the model. Reading ``stopReason`` here is what makes
    the batch path inherit truncation labelling for free.

    ``stopReason`` is read with ``get`` rather than indexed: live Converse always returns it, but a
    stored fixture or a hand-built body predates the field, and a missing label is honestly None
    rather than a crash.
    """
    blocks = body["output"]["message"]["content"]
    return BedrockCompletion(
        text=_join_text_blocks(blocks),
        reasoning=_join_reasoning_blocks(blocks),
        usage=_sum_usage(body["usage"]),
        stop_reason=body.get("stopReason"),
    )


class ConverseEventStream(Protocol):
    """The part of botocore's ``EventStream`` the accumulator below uses.

    A protocol rather than the botocore class because ``boto3`` is an optional extra: naming
    ``botocore.eventstream.EventStream`` in an annotation would make this module unimportable
    without it, which is the whole reason the SDK is imported lazily inside the backend.
    """

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Yield one parsed event dict at a time, blocking until the next one arrives."""
        ...

    def close(self) -> None:
        """Release the underlying HTTP response, abandoning any events not yet read."""
        ...


def _rebuild_content_blocks(
    text_fragments: Mapping[int, list[str]], reasoning_fragments: Mapping[int, list[str]]
) -> list[dict[str, Any]]:
    """Join each streamed block's fragments back into the non-streaming content-block list.

    Ordered by ``contentBlockIndex``, since that is the order the blocks were generated in and the
    order a non-streaming response returns them in; a set would reorder the answer against the
    reasoning that produced it. A single index carrying both kinds is emitted as one block with both
    members, which is what the two joiners downstream already read independently -- no observed
    model does that, but silently dropping one of the two would be a lost trace rather than a crash.
    """
    blocks: list[dict[str, Any]] = []
    for index in sorted(set(text_fragments) | set(reasoning_fragments)):
        block: dict[str, Any] = {}
        if index in text_fragments:
            block["text"] = "".join(text_fragments[index])
        if index in reasoning_fragments:
            joined = "".join(reasoning_fragments[index])
            block["reasoningContent"] = {"reasoningText": {"text": joined}}
        blocks.append(block)
    return blocks


def _absorb_delta(
    delta_event: Mapping[str, Any],
    text_fragments: dict[int, list[str]],
    reasoning_fragments: dict[int, list[str]],
) -> None:
    """File one ``contentBlockDelta`` under its block index, answer text and reasoning apart.

    A reasoning delta carries ``reasoningContent`` where an answer delta carries ``text``, and only
    readable reasoning (``reasoningContent.text``) is kept: ``redactedContent`` is an encrypted
    blob and ``signature`` an attestation, and neither is a trace anybody can read.
    """
    index = int(delta_event["contentBlockIndex"])
    delta: Mapping[str, Any] = delta_event["delta"]
    text: str | None = delta.get("text")
    if text is not None:
        text_fragments.setdefault(index, []).append(text)
    reasoning: Mapping[str, Any] | None = delta.get("reasoningContent")
    if reasoning is not None:
        reasoning_text: str | None = reasoning.get("text")
        if reasoning_text is not None:
            reasoning_fragments.setdefault(index, []).append(reasoning_text)


def accumulate_converse_stream(
    stream: ConverseEventStream, *, deadline: float, model_id: str
) -> tuple[dict[str, Any], float | None]:
    """Rebuild the non-streaming Converse response body from a ``converse_stream`` event stream.

    Returns the body and the :func:`time.monotonic` stamp of the first event read, or ``None`` when
    the stream delivered nothing. The stamp is the raw material of ``first_event_seconds`` on the
    completion: on a reasoning model the gap before the first event is thinking time, and it is the
    one measurement that separates a long think from a socket that will never answer.

    Deliberately reconstructs the *body* rather than a :class:`BedrockCompletion`, so
    :func:`parse_converse_output` stays the single reader of a Converse response for all three
    transports. The alternative -- a second joiner that walks the deltas straight into text and
    reasoning strings -- is exactly the drift :func:`converse_request` and
    :func:`parse_converse_output` were paired to prevent, and a drift there would read as a
    difference between models rather than as a bug: the live path would score reasoning as answer
    while the batch path did not.

    Two levels of joining, and they are not the same operation. Deltas *within* one content block
    are token fragments and concatenate with nothing between them; blocks are joined with a space,
    which is what ``_join_text_blocks`` already does downstream. ``contentBlockIndex`` is what keeps
    them apart, and it is also what keeps a reasoning trace out of the answer: a reasoning delta
    carries ``reasoningContent`` where an answer delta carries ``text``, and each accumulates into
    its own block. Collapse that and every downstream label built on the answer is silently wrong.

    Event shapes verified against the ``ConverseStream`` output union in the installed botocore's
    ``bedrock-runtime`` service model (1.43.72), not recalled: ``contentBlockDelta`` carries
    ``delta`` plus ``contentBlockIndex``, its ``delta`` is a union of ``text`` /
    ``reasoningContent`` / ``toolUse`` / ``toolResult`` / ``citation`` / ``image``, its
    ``reasoningContent`` is a union of ``text`` / ``redactedContent`` / ``signature``,
    ``messageStop`` carries ``stopReason``, and ``metadata`` carries ``usage``. Only readable
    reasoning is kept, matching :func:`_join_reasoning_blocks`: a ``redactedContent`` delta is an
    encrypted blob and a ``signature`` delta is an attestation, and neither is a trace anybody can
    read. Modelled service errors do not arrive as members of this union in practice -- botocore
    raises ``EventStreamError`` out of the iterator instead -- so they propagate rather than being
    parsed here, and :meth:`BedrockBackend._call_isolated` is where one becomes a counted
    ``call_failed`` record instead of a dead batch.

    ``deadline`` is an absolute :func:`time.monotonic` value, checked after each event so the loop
    never enters another blocking read past it. Once ``messageStop`` has arrived the deadline stops
    applying: ``metadata`` follows it immediately and carries the token counts, and abandoning
    between the two would throw away the cost record of a call that had in fact completed.

    A stream that runs out of events without a ``messageStop`` raises
    :class:`IncompleteConverseStreamError`. That is not the deadline case and must not be labelled
    as one -- a wrong label on a partial record is worse than a counted failure. The raise was a
    bare ``RuntimeError`` until the failure was observed live and named (see the class docstring);
    :meth:`BedrockBackend._call_isolated` now counts it instead of letting it kill the batch.
    """
    text_fragments: dict[int, list[str]] = {}
    reasoning_fragments: dict[int, list[str]] = {}
    usage: Mapping[str, Any] = {}
    stop_reason: str | None = None
    message_ended = False
    abandoned = False
    first_event_at: float | None = None
    for event in stream:
        if first_event_at is None:
            first_event_at = time.monotonic()
        delta_event: Mapping[str, Any] | None = event.get("contentBlockDelta")
        if delta_event is not None:
            _absorb_delta(delta_event, text_fragments, reasoning_fragments)
        message_stop: Mapping[str, Any] | None = event.get("messageStop")
        if message_stop is not None:
            message_ended = True
            stop_reason = message_stop.get("stopReason")
        metadata: Mapping[str, Any] | None = event.get("metadata")
        if metadata is not None:
            usage = metadata.get("usage") or {}
        if not message_ended and time.monotonic() >= deadline:
            stream.close()
            abandoned = True
            break
    if not message_ended and not abandoned:
        raise IncompleteConverseStreamError(
            f"the {model_id} converse_stream ended without a messageStop event after "
            f"{len(text_fragments)} answer and {len(reasoning_fragments)} reasoning blocks; "
            "the reply is incomplete for a reason this code cannot name, and labelling it as a "
            "deadline abandonment would misattribute it"
        )
    body = {
        "output": {
            "message": {"content": _rebuild_content_blocks(text_fragments, reasoning_fragments)}
        },
        "usage": dict(usage),
        "stopReason": STOP_REASON_DEADLINE_EXCEEDED if abandoned else stop_reason,
    }
    return body, first_event_at


class BedrockBackend:
    """AWS Bedrock ConverseStream backend: the hosted-API path, and the only cross-vendor one.

    ``boto3`` is an optional extra (``uv sync --extra bedrock``), imported lazily here the same
    way ``VLLMBackend`` imports ``vllm``, so this module still loads with it absent.

    Four choices in here are load-bearing and were settled against live endpoints or the installed
    service model rather than read off a doc page. Converse rather than ``invoke-model``, because
    Converse returns reasoning and answer as separate content blocks where the native path
    concatenates them into one string that a naive parser scores as the answer. Retries left
    entirely to botocore's adaptive mode, because Bedrock's transient 500s arrive both as
    ``InternalServerException`` and as a bare ``ClientError`` with code ``"500"`` that maps to no
    modelled exception class, so no hand-written ``except`` could cover them even if this repo
    allowed the broad one that would -- capped at
    :data:`DEFAULT_BEDROCK_MAX_ATTEMPTS`, see there for what the cap bounds.  And
    ``max_pool_connections`` above the worker count, because urllib3 otherwise serialises the burst
    and the achieved concurrency silently becomes the pool size instead of the requested one.

    The fourth is ``converse_stream`` rather than ``converse``, and it is what makes a run's wall
    clock a number anybody can state. A socket read timeout bounds the gap between two reads, never
    the call, so a model that keeps producing tokens runs as long as it likes: on hard arms MiniMax
    M2.5 takes 2,900-3,000+ seconds per call and 11 of 24 calls in one pass died at 2,700 seconds,
    losses that correlate with reasoning length and so bias the surviving denominator. Streaming
    puts the loop in this process, where a deadline can be enforced and a call past it can hand back
    what arrived. ``transport`` deliberately stays ``bedrock-converse``: the request payload
    :func:`converse_request` renders is byte-identical either way, so streaming is a wire framing
    rather than a different elicitation, and relabelling it would split one elicitation across two
    values in every pooled analysis and in every record already on disk. Streaming is also not
    optional -- a model that cannot stream fails loudly on its first call, with no fallback path to
    quietly change what an arm measured.

    Converse takes one prompt per call, so ``generate`` fans a batch across a thread pool and
    reassembles in request order. The request payload and the effort dialect are both resolved in
    ``__init__``, so a mis-specified run dies before spending a token rather than 2,000 calls in.

    ``profile`` names an AWS profile, or is ``None`` to fall back to the ambient credential chain,
    which is what a Batch container holding an instance role has and no named profile. Left
    unspecified it resolves :func:`bedrock_profile` from the environment, and a named profile is the
    default rather than the ambient chain because this box's ambient default profile is a
    placeholder that fails credential resolution outright.

    ``deadline_seconds``, ``read_timeout`` and ``max_attempts`` are constructor arguments rather
    than constants baked into the client, because the three of them are what a run's worst-case wall
    clock is computed from and one roster model needs a different answer than another. They were
    baked in before, and the workaround was reaching into ``_client`` to rebuild it -- which is a
    silent way to end up with two clients configured differently in the same analysis.
    """

    transport = "bedrock-converse"

    def __init__(  # noqa: PLR0913 - flat call-site knobs; a config object would hide the defaults
        self,
        model_id: str,
        *,
        region: str = DEFAULT_BEDROCK_REGION,
        profile: str | None = PROFILE_FROM_ENVIRONMENT,
        concurrency: int = DEFAULT_BEDROCK_CONCURRENCY,
        sampling: BedrockSamplingConfig | None = None,
        deadline_seconds: float = DEFAULT_BEDROCK_DEADLINE_SECONDS,
        read_timeout: int = DEFAULT_BEDROCK_READ_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_BEDROCK_MAX_ATTEMPTS,
    ) -> None:
        """Build the bedrock-runtime client and freeze the per-request payload for this model."""
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {concurrency}")
        if deadline_seconds <= 0:
            raise ValueError(f"deadline_seconds must be positive, got {deadline_seconds}")
        self.model_id = model_id
        self.concurrency = concurrency
        self.sampling = sampling or BedrockSamplingConfig()
        self.deadline_seconds = deadline_seconds
        self.usage = TokenUsage()
        # The whole-run cost record. += on it is a non-atomic read-modify-write, and generate()
        # is called from several episode threads at once under --episode-concurrency, so an
        # unlocked accumulate silently undercounts the only billing record there is.
        self._usage_lock = threading.Lock()

        self._inference_config = converse_inference_config(self.sampling)
        self._extra_request_fields = converse_extra_request_fields(model_id, self.sampling)

        # After the payload checks so a bad effort still reports the effort, before the session.
        resolved_profile = bedrock_profile() if profile == PROFILE_FROM_ENVIRONMENT else profile
        boto3 = importlib.import_module("boto3")
        botocore_config = importlib.import_module("botocore.config")
        botocore_exceptions = importlib.import_module("botocore.exceptions")
        # Both spellings of the same two transport failures, plus the class botocore raises when
        # the service itself fails mid-stream -- see ``_call_isolated`` for why each is here.
        self._isolated_failures: tuple[type[BaseException], ...] = (
            botocore_exceptions.ReadTimeoutError,
            botocore_exceptions.ConnectTimeoutError,
            botocore_exceptions.ConnectionClosedError,
            botocore_exceptions.EndpointConnectionError,
            botocore_exceptions.EventStreamError,
            urllib3.exceptions.ReadTimeoutError,
            urllib3.exceptions.ProtocolError,
            IncompleteConverseStreamError,
        )
        self._client_error: type[BaseException] = botocore_exceptions.ClientError
        session = boto3.Session(profile_name=resolved_profile, region_name=region)
        self._client = session.client(
            "bedrock-runtime",
            config=botocore_config.Config(
                read_timeout=read_timeout,
                connect_timeout=30,
                max_pool_connections=max(concurrency * 2, 20),
                retries={"max_attempts": max_attempts, "mode": "adaptive"},
            ),
        )
        logger.info(
            "BedrockBackend ready for %s region=%s profile=%s concurrency=%d effort=%s; "
            "one call is bounded at about %.0fs (%d attempts x %ds to start the stream, "
            "+ %.0fs reading it, + %ds of final silence)",
            model_id,
            region,
            resolved_profile,
            concurrency,
            self.sampling.reasoning_effort,
            max_attempts * read_timeout + deadline_seconds + read_timeout,
            max_attempts,
            read_timeout,
            deadline_seconds,
            read_timeout,
        )

    def _converse(self, prompt: str) -> BedrockCompletion:
        """Stream one Converse call under a wall-clock deadline, and split what it returned.

        Raises rather than absorbing a transport failure, so a caller sampling one prompt at a time
        still sees it; :meth:`_call_isolated` is where a batch turns one into a counted record.

        A call abandoned at the deadline is billed for tokens this record cannot report -- the
        ``metadata`` event carrying the counts is the last one to arrive, and it never did. So it
        comes back with zero usage, which makes a run total that includes it a lower bound on spend.
        The warning below is the only place that shortfall is visible, which is why it names the
        elapsed time and the characters that did arrive rather than just saying "deadline".
        """
        request = converse_request(prompt, self._inference_config, self._extra_request_fields)
        started = time.monotonic()
        response = self._client.converse_stream(modelId=self.model_id, **request)
        body, first_event_at = accumulate_converse_stream(
            response["stream"],
            deadline=started + self.deadline_seconds,
            model_id=self.model_id,
        )
        elapsed = time.monotonic() - started
        completion = replace(
            parse_converse_output(body),
            elapsed_seconds=elapsed,
            first_event_seconds=None if first_event_at is None else first_event_at - started,
            attempts=_attempts_from_metadata(response.get("ResponseMetadata")),
        )
        if completion.stop_reason == STOP_REASON_DEADLINE_EXCEEDED:
            logger.warning(
                "BedrockBackend %s abandoned a call at its %.1fs deadline after %.1fs, keeping "
                "%d answer and %d reasoning characters; the call was billed for tokens this "
                "record cannot report, so any run total including it is a lower bound",
                self.model_id,
                self.deadline_seconds,
                elapsed,
                len(completion.text),
                len(completion.reasoning),
            )
        else:
            logger.info(
                "BedrockBackend %s call took %.0fs (first event at %s, %s attempts): stop=%s "
                "out=%d tokens",
                self.model_id,
                elapsed,
                "-"
                if completion.first_event_seconds is None
                else f"{completion.first_event_seconds:.1f}s",
                completion.attempts,
                completion.stop_reason,
                completion.usage.output_tokens,
            )
        return completion

    def _call_isolated(self, prompt: str) -> BedrockCompletion:
        """Run one call, turning an expected transport failure into a counted empty record.

        The closed list resolved in ``__init__`` is deliberately not ``Exception``. A read timeout
        or a dropped connection on a model documented to occasionally never return is expected and
        should cost one sample; anything else -- a malformed payload, an unknown model id, a
        credentials failure -- is a bug in the run and must crash on the first call rather than be
        absorbed into hundreds of empty records that read as a model saying nothing.

        That list spans two packages because one transport failure arrives under two class names
        depending on when it happens, and getting this wrong is not hypothetical: it shipped.
        ``botocore.httpsession`` re-raises urllib3's errors as its own, but only inside the call that
        returns the response object, so a socket that goes quiet there becomes botocore's
        ``ReadTimeoutError`` and a broken connection its ``ConnectionClosedError``. The reply body is
        then read by iterating the event stream (:func:`accumulate_converse_stream`), outside that
        wrapping, where the same two failures reach this seam as urllib3's own ``ReadTimeoutError``
        and ``ProtocolError``. While the list named only the botocore spellings, every failure on the
        streaming path escaped into :meth:`generate_detailed` and discarded the replies its siblings
        had already been billed for -- about 1 call in 100 on MiniMax M2.5, rising with generation
        length, so the losses correlated with reasoning length exactly as the pre-streaming ones did.

        ``EventStreamError`` is the third way the same expected failure arrives, and it shipped
        too: the service failing *itself* mid-reply is delivered as a modelled exception event in
        the stream, which botocore's event-stream parser raises under that one class whatever the
        code inside it. On 2026-08-23 Bedrock delivered an ``internalServerException`` part-way
        through a Sonnet 5 reply and the escape killed a 700-rollout pass, discarding 22 billed
        sibling replies. Isolating it does not widen the list toward request-time bugs: botocore
        raises ``EventStreamError`` only from the event-stream iterator, after the request was
        accepted, so credentials failures, unknown model ids and rejected payloads still arrive as
        ``ClientError`` from the ``converse_stream`` call and still crash the batch. The recorded
        stop reason keeps the service's own code (see :func:`call_failed_stop_reason`).

        A modelled service error is the fourth way, and it is isolated by CODE rather than by class:
        every one of them arrives as a ``ClientError`` subclass, and the class alone cannot tell a
        throttle that outlived botocore's five adaptive attempts from a payload the service
        rejected. :data:`TRANSIENT_CLIENT_ERROR_CODES` is the closed list of codes that cost one
        sample; any other code -- ``ValidationException``, ``AccessDeniedException``,
        ``ResourceNotFoundException`` -- re-raises and crashes the batch exactly as before. Three
        recorded incidents in three weeks each discarded a chunk's finished siblings to one of the
        codes now on that list.

        This is the seam that stops one stalled call from discarding its siblings. Every prompt in
        the batch still gets a record, in request order, so the denominator an analysis divides by
        is the number of prompts requested rather than the number that happened to survive.
        """
        started = time.monotonic()
        try:
            return self._converse(prompt)
        except self._isolated_failures as error:
            return self._failed_record(error, started)
        except self._client_error as error:
            if client_error_code(error) not in TRANSIENT_CLIENT_ERROR_CODES:
                raise
            return self._failed_record(error, started)

    def _failed_record(self, error: BaseException, started: float) -> BedrockCompletion:
        """Build the counted empty record for a call that died of an expected failure.

        The elapsed time is recorded on the failure too, because a wedge that burned a full read
        timeout and a throttle refused in a second are different failures with the same stop reason,
        and the attempt count is read off the error's own response metadata where botocore put one.
        """
        stop_reason = call_failed_stop_reason(error)
        logger.error(
            "BedrockBackend %s: one call failed and is recorded as stop_reason=%s; the other "
            "calls in this batch are unaffected and this prompt still has a record",
            self.model_id,
            stop_reason,
            exc_info=error,
        )
        response = _error_response(error)
        return BedrockCompletion(
            text="",
            reasoning="",
            usage=TokenUsage(),
            stop_reason=stop_reason,
            elapsed_seconds=time.monotonic() - started,
            first_event_seconds=None,
            attempts=_attempts_from_metadata(
                None if response is None else response.get("ResponseMetadata")
            ),
        )

    def submit_stream(self, prompts: Iterable[str]) -> Generator[tuple[int, BedrockCompletion]]:
        """Run the prompts with ``concurrency`` calls in flight at all times, yielding as each lands.

        The seam that removes the per-chunk barrier. A pool opened per ``generate_detailed`` call
        ends when its slowest call ends, with the other ``concurrency - 1`` workers idle behind it:
        one measured 20-rollout chunk took 53 minutes while 19 of its calls finished in under five,
        and a Sonnet pass paying that barrier on nearly every chunk accumulated ~66 hours of wall
        clock for $0. Here the queue is topped up the moment any call finishes, whatever chunk the
        caller will file it under, so the wall clock of a sweep is its streaming bound rather than
        the sum of its chunks' slowest calls. Yields ``(index, completion)`` in COMPLETION order;
        :func:`stream_detailed_in_chunks` is what turns that back into request-ordered chunks.

        Each completion's usage joins the run total the moment it is yielded, under the lock,
        because the caller may be persisting as it goes and the total must never lag what is on disk.

        A raise is still a bug rather than a transport failure (see :meth:`_call_isolated`), and it
        still stops the batch paying for calls that had not started: nothing new is submitted after
        the first one, the calls already in flight are waited for and yielded -- they are paid for
        -- and then the lowest-index error propagates. A consumer that stops iterating early gets
        the same drain: the pool is shut down waiting on its in-flight calls and their usage is
        summed, so no billed call is ever missing from the run total.
        """
        pool = ThreadPoolExecutor(max_workers=self.concurrency)
        in_flight: dict[Future[BedrockCompletion], int] = {}
        queue = enumerate(prompts)
        first_error: tuple[int, BaseException] | None = None
        handed_back = 0
        handed_back_usage = TokenUsage()
        try:
            while True:
                while first_error is None and len(in_flight) < self.concurrency:
                    try:
                        index, prompt = next(queue)
                    except StopIteration:
                        break
                    in_flight[pool.submit(self._call_isolated, prompt)] = index
                if not in_flight:
                    break
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=in_flight.__getitem__):
                    index = in_flight.pop(future)
                    error = future.exception()
                    if error is not None:
                        if first_error is None or index < first_error[0]:
                            first_error = (index, error)
                        continue
                    completion = future.result()
                    with self._usage_lock:
                        self.usage += completion.usage
                    handed_back += 1
                    handed_back_usage += completion.usage
                    yield index, completion
        finally:
            self._settle_abandoned(pool, in_flight)
        if first_error is not None:
            logger.error(
                "BedrockBackend %s: the Converse call for prompt %d raised; %d completions were "
                "handed back before it, billed at in=%d out=%d tokens -- a caller collecting them "
                "into one list discards them with this raise, a caller persisting as it goes has "
                "them on disk",
                self.model_id,
                first_error[0],
                handed_back,
                handed_back_usage.input_tokens,
                handed_back_usage.output_tokens,
            )
            raise first_error[1]

    def _settle_abandoned(
        self, pool: ThreadPoolExecutor, in_flight: Mapping[Future[BedrockCompletion], int]
    ) -> None:
        """Wait out the calls a consumer walked away from and put their cost in the run total.

        Empty on every normal exit of :meth:`submit_stream`, which drains its own queue; only a
        consumer that stops iterating early leaves anything here. Those calls have reached the
        endpoint and are billed whether or not anybody reads them, so the total must carry them.
        """
        pool.shutdown(wait=True, cancel_futures=True)
        if not in_flight:
            return
        abandoned = TokenUsage()
        for future in in_flight:
            if not future.cancelled() and future.exception() is None:
                abandoned += future.result().usage
        with self._usage_lock:
            self.usage += abandoned
        logger.warning(
            "BedrockBackend %s: the consumer stopped reading with %d calls in flight; they were "
            "waited for and their in=%d out=%d tokens added to the run total, but nobody read them",
            self.model_id,
            len(in_flight),
            abandoned.input_tokens,
            abandoned.output_tokens,
        )

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        """Generate completions keeping the reasoning and token usage the protocol discards.

        The sampling pass is the expensive thing here, so it should answer as many questions as it
        can at once: what the model said, what it was thinking where that is legible, and what the
        call cost. ``self.usage`` accumulates across calls to give a whole-run total.

        Every prompt comes back with a record, in request order, including the ones whose call died
        of a transport failure or ran past the deadline -- see :meth:`_call_isolated` and
        :func:`is_incomplete_stop_reason`. A batch is therefore as long as it was wide, and the
        losses that used to shrink a denominator invisibly are now rows carrying why they failed.

        What still raises is a bug rather than a transport failure: an unknown model id, a payload
        Bedrock rejects, a credentials problem. Those propagate through :meth:`submit_stream`,
        which stops paying for calls that had not started and puts every call that did reach the
        endpoint in the run total before raising. Which completions the raise discards is a
        separate question, deliberately left alone here: handing them back would change this
        method's contract for every caller, and a caller that wants them persists as it goes
        through :func:`stream_detailed_in_chunks` instead.
        """
        by_index = dict(self.submit_stream(prompts))
        completions = [by_index[index] for index in range(len(prompts))]
        # A completed count with no partial count beside it is the survivor rate, not a denominator.
        incomplete = [c for c in completions if is_incomplete_stop_reason(c.stop_reason)]
        logger.info(
            "BedrockBackend %s returned %d records for %d prompts, %d of them partial (%s); "
            "run total in=%d out=%d tokens",
            self.model_id,
            len(completions),
            len(prompts),
            len(incomplete),
            sorted({str(c.stop_reason) for c in incomplete}) or "none",
            self.usage.input_tokens,
            self.usage.output_tokens,
        )
        return completions

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate one completion per prompt, in request order."""
        return [completion.text for completion in self.generate_detailed(prompts)]
