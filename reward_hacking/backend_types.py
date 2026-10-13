"""Result records, stop-reason vocabulary and error helpers shared by the hosted-model backends.

Split out of ``model_backend`` (which re-exports every name here) so the transport modules can share the
types without importing each other. Leaf module: it imports nothing from the backends.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
    )


logger = logging.getLogger("reward_hacking.model_backend")


STOP_REASON_END_TURN = "end_turn"
STOP_REASON_MAX_TOKENS = "max_tokens"
"""The two stop reasons every transport here reports, spelled the way Converse spells them.

Converse's own vocabulary rather than a local invention, because the field is pooled across
transports: ``recoverybench.grading`` already keys ``truncated`` off the literal ``"max_tokens"``,
so a local backend reporting ``"length"`` (vLLM's word) would be read as "stopped for some other
reason" -- silently, in the one field that tells a clipped reply from a reply that declined.
"""

STOP_REASON_STOP_SEQUENCE = "stop_sequence"
"""The reply halted on a caller-configured stop sequence, spelled the way Converse spells it.

Converse's own vocabulary for the same reason the two constants above use it: the field is pooled
across transports, so vLLM's matched-stop-string case and transformers' ``stop_strings`` halt both
travel under the label a Bedrock reply would carry natively. A caller reading it must know the
transports disagree about the text: vLLM (``include_stop_str_in_output=True``) and the HF path
return the completion WITH the stop text, while Converse strips it -- the agent harness restores it
before parsing (see ``harness/loop.py``), which is the one consumer that needs the closing tag.
"""

STOP_REASON_DEADLINE_EXCEEDED = "deadline_exceeded"
"""The streaming Converse path abandoned this call at its wall-clock deadline, mid-generation.

Deliberately *not* one of Bedrock's own nine ``StopReason`` values, because the model did not stop
-- we stopped reading. Whatever text and reasoning had streamed in by then is kept rather than
thrown away, so the record is partial rather than absent, and this label is the only thing marking
it partial: the text field of a reply abandoned after its answer arrived is indistinguishable from
a complete one. It reads as ``Outcome.NO_ANSWER_UNKNOWN_STOP`` in
``recoverybench.grading._outcome_without_an_answer`` (an unrecognised stop reason gets its own
bucket there rather than joining the declines), which is the correct reading: we do not know
whether the policy was going to answer.
"""

CALL_FAILED_STOP_REASON_PREFIX = "call_failed:"
"""Prefix marking a record for a call that never returned, followed by the failure's class name.

The failure reason travels in the record rather than only in the log because the record is the
artifact an analysis reads months later. A call that timed out has to stay *counted* -- a
denominator that quietly shrinks is worse than a visible error, and on hard arms the losses
correlate with reasoning length, which is often the variable under manipulation, so dropping them
biases the survivors rather than merely thinning them.
"""


class IncompleteConverseStreamError(RuntimeError):
    """A ConverseStream ran out of events before ``messageStop``, delivering no exception either.

    The third spelling of a dead reply: a connection dropped in a way botocore surfaces as neither
    a raised transport error nor an error event -- the iterator just ends, and
    :func:`accumulate_converse_stream` is what notices. Observed live at about 1 call in 290 on
    Opus 5 (2026-08-24, zero content blocks delivered), never on the same round's Sol, GPT-OSS or
    Luna passes, so it is the expected occasionally-never-returns failure the per-call isolation
    list exists for. It gets its own class rather than the bare ``RuntimeError`` it used to be
    precisely so that list can name it: naming ``RuntimeError`` itself would absorb every genuine
    accumulator bug into empty records. Still deliberately never labelled a deadline abandonment --
    a wrong label on a partial record is worse than a counted failure.

    Subclasses ``RuntimeError`` so anything outside this module that matched the old raise still
    does.
    """


def call_failed_stop_reason(error: BaseException) -> str:
    """Render the stop reason recorded for a call that failed with ``error``.

    A modelled service error (botocore's ``EventStreamError``, a ``ClientError`` subclass) also
    names the service's own error code, because every one of them arrives under that single class
    name and the bare name would pool a throttle with an internal server error -- the code is the
    part an analysis of failure rates actually reads. The transport classes carry no ``response``
    and keep the bare class name, which is what every record already on disk holds.
    """
    name = type(error).__name__
    code = client_error_code(error)
    if code:
        return f"{CALL_FAILED_STOP_REASON_PREFIX}{name}:{code}"
    return f"{CALL_FAILED_STOP_REASON_PREFIX}{name}"


def _error_response(error: BaseException) -> Mapping[str, Any] | None:
    """Return the botocore ``response`` dict a modelled service error carries, else None.

    ``getattr`` rather than an ``isinstance`` against ``ClientError`` because botocore is an optional
    extra this module must not import at top level; the shape check on the attribute is what keeps a
    stand-in exception in the tests honest about carrying the real constructor's payload.
    """
    response = getattr(error, "response", None)
    return response if isinstance(response, dict) else None


def client_error_code(error: BaseException) -> str | None:
    """Return the service's own error code on a modelled botocore error, or None if there is none."""
    response = _error_response(error)
    if response is None:
        return None
    code = response.get("Error", {}).get("Code")
    return str(code) if code else None


def is_incomplete_stop_reason(stop_reason: str | None) -> bool:
    """Whether this stop reason marks a record no analysis may count as a finished reply.

    One predicate rather than a literal comparison at every call site, because there are two ways a
    record can be partial (the deadline and a failed call) and the second carries the failure's
    class name, so ``== "call_failed"`` would miss every one of them.

    ``None`` reads False, the same convention the agent harness's ``hit_output_cap`` uses: it means
    the transport could not say, which is not evidence of incompleteness. So the count this feeds is
    a floor on how many records are partial, never a rate.
    """
    if stop_reason is None:
        return False
    return stop_reason == STOP_REASON_DEADLINE_EXCEEDED or stop_reason.startswith(
        CALL_FAILED_STOP_REASON_PREFIX
    )


DEFAULT_BEDROCK_MAX_TOKENS = 30_000
"""The largest per-model output budget measured on this roster, as the shared Converse default.

Deliberately generous rather than modest-looking. The cap is a ceiling and a token is billed only if
it is generated, so an over-wide cap costs nothing, while a cap set too low is paid for in whole
runs. The previous default of 2048 was that mistake: over a 400-episode agentic run 37% of Converse
calls stopped at exactly 2048, and because the frontier model returns its reasoning redacted while
still charging it against this same budget, those turns came back empty or with an unclosed command
block and read downstream as a policy that declined to act rather than as a config bug.

A model whose own ceiling is lower refuses this loudly -- a botocore ``ValidationException`` on the
live path, a construction-time raise against the roster on the batch path -- which is the right
failure. A per-model budget is a measurement, and the measured ones live in
``recoverybench.budgets``, whose frontier row is where this number comes from.

Whether the field has to be sent at all is settled (hot-path decision C11, probed live 2026-09-03
on ``openai.gpt-oss-20b-1:0`` in us-west-2): ``Converse`` with no ``inferenceConfig`` at all,
``Converse`` with an empty ``inferenceConfig`` and ``ConverseStream`` with no ``inferenceConfig``
were all accepted and answered ``end_turn`` at 40 to 100 output tokens, and the installed botocore
service model lists none of ``inferenceConfig``'s four members as required. Omitting ``maxTokens``
runs the model at the provider's own output ceiling, which is where the owner's rule points (leave
the output cap unset; the levers are sample count and effort). This code still sends the field on
every request, because ``max_tokens`` is read as an ``int`` by the batch ceiling check, the
RecoveryBench run labels, the harness cap check and the batch-handle label checks, so making it
optional is a change to those consumers as much as to this default. Until that lands, this is the
cap every default run samples under, and every record carries it as such.
"""


@dataclass(frozen=True, slots=True)
class BedrockSamplingConfig:
    """Decoding knobs the Bedrock Converse API actually honours.

    Deliberately not ``SamplingConfig``: Converse's ``inferenceConfig`` has no ``top_k`` and no
    greedy switch, so reusing the HuggingFace config would mean dropping ``top_k`` and
    ``do_sample`` silently, which is the whole failure class this harness keeps getting bitten by.
    ``None`` means "omit the field", which is the request shape verified against every model in
    the roster; set one and it is sent, and a model that refuses it fails loudly with a botocore
    ``ValidationException`` rather than ignoring it.

    ``max_tokens`` is the one field this config always sends -- ``converse_inference_config`` writes
    ``maxTokens`` on every request -- so it is the one field whose default is in the request whether
    anybody chose it or not. The API does not require it (:data:`DEFAULT_BEDROCK_MAX_TOKENS` has the
    live probe that settled that); this code does, for now. That is why it defaults *high*
    (:data:`DEFAULT_BEDROCK_MAX_TOKENS`, see its comment for the run this cost) rather than to a
    value that looks conservative: an unchosen cap that truncates is indistinguishable, downstream,
    from a model that declined to answer.

    ``stop_sequences`` becomes the request's ``stopSequences`` and is empty by default -- the
    benchmark runners want the whole reply. Unlike the local backends, **Converse strips the
    matched stop text from the returned completion** (the reply ends just before it, with
    ``stopReason`` ``"stop_sequence"``), so a caller that needs the text intact -- the agent
    harness, whose block parser wants the ``</run>`` it stopped on -- has to restore it; the loop
    does, keyed on that stop reason.
    """

    max_tokens: int = DEFAULT_BEDROCK_MAX_TOKENS
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None
    stop_sequences: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Input and output token counts, over one Converse call or a whole run of them.

    ``input_tokens`` is every token the model read -- uncached, cache-read and cache-write summed --
    which is what every record on disk and every cost script has always meant by it. The two cache
    counters ride beside it rather than being carved out of it, so the total is unchanged and the
    split is still recoverable: uncached input is ``input_tokens`` minus the two. Recording the
    split is what turns the run docs' list-price cost figures into real ones -- a Luna judge pass
    whose 4,360-token rubric was cached on every call was billed at the cached rate for ~86% of
    its input, a factor the run docs could not see because the counters were folded away here.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Accumulate field-wise, so a run total is the sum of its calls."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_write_input_tokens=(
                self.cache_write_input_tokens + other.cache_write_input_tokens
            ),
        )


@dataclass(frozen=True, slots=True)
class BedrockCompletion:
    """One Converse response, split into what the model answered, what it thought, and its cost.

    ``reasoning`` is empty whenever the trace is not legible rather than absent: GPT-5.6 returns
    ``reasoningContent.redactedContent``, an encrypted blob, so of the verified models only
    GPT-OSS can be read. That asymmetry is structural and constrains any measurement that scores
    reasoning rather than answers.

    ``stop_reason`` is why the model stopped generating (``end_turn``, ``max_tokens``, ...), and it
    is what separates a reply that declined to answer from one that ran out of output budget before
    it got there. Without it, a token cap manufactures apparent non-compliance: at 16,384 tokens
    the missing-answer rate across three roster models was 21%, 54% and 85%, none of it a model
    property. It carries no default for the reason ``transport`` does not: a field sometimes
    silently absent would put every unlabelled reply in the "declined" bucket.

    The three trailing fields are the live path's per-call telemetry, and they default to ``None``
    because only the live path can measure them: a batch result file has no latency to report and a
    stored fixture predates them. ``elapsed_seconds`` is the call's whole wall clock, request to
    last event; ``first_event_seconds`` is the time to the first streamed event, which on a
    reasoning model is mostly time spent thinking before any token is sent (Sonnet 5 emits nothing
    over Converse while it thinks, so this is the number that tells a long think from a wedged
    socket); ``attempts`` is how many times botocore sent the request before one was accepted.
    Read the two together: both clocks start at the FIRST attempt, so botocore's internal retries
    are inside them, and a row with ``attempts=2`` and ``first_event_seconds`` of ~900 is a first
    attempt that died silently for the whole read timeout before the retry answered, not a
    fifteen-minute think (three of 64 rows in the first live smoke had exactly that shape).
    They exist because the only per-call latency record used to be an INFO line in a ``/tmp`` log
    that was gone for every production run an audit tried to read, so the barrier loss a chunked
    sweep pays on its slowest call could only be simulated, never measured.
    """

    text: str
    reasoning: str
    usage: TokenUsage
    stop_reason: str | None
    elapsed_seconds: float | None = None
    first_event_seconds: float | None = None
    attempts: int | None = None


class CompletionTelemetry(TypedDict):
    """The five per-call accounting fields of :func:`completion_telemetry`, typed one by one.

    A ``TypedDict`` rather than ``dict[str, Any]`` so that a record built by unpacking it -- as
    :func:`raw_response` does -- is checked field for field against the record's own declaration,
    and a renamed or dropped key fails the type check instead of landing as an unknown keyword or a
    silent ``None``.
    """

    cache_read_input_tokens: int | None
    cache_write_input_tokens: int | None
    elapsed_seconds: float | None
    first_event_seconds: float | None
    attempts: int | None


def completion_telemetry(completion: BedrockCompletion) -> CompletionTelemetry:
    """Flatten the per-call accounting a record carries beyond its token totals, under one spelling.

    The studies' reply rows, both judges' rows (under a ``judge_`` prefix) and :class:`RawResponse`
    all carry these five, and a cost script joins them on the names: the cache split is what turns
    a pass's list-price figure into a real one (a rubric cached on every call bills at a tenth of
    the uncached rate), and the two clocks plus the attempt count are what let a slow pass be read
    off the artifact instead of a ``/tmp`` log that is gone. It lives here, beside the
    :class:`BedrockCompletion` it reads, because ``sociology`` builds on this package and not the
    reverse: a copy in ``sociology.records`` and a hand-spelled twin in the hatch judge were held to
    the same five names only by their tests. ``None`` where the transport cannot measure a value (a
    batch result file has no clocks), never zero.
    """
    return CompletionTelemetry(
        cache_read_input_tokens=completion.usage.cache_read_input_tokens,
        cache_write_input_tokens=completion.usage.cache_write_input_tokens,
        elapsed_seconds=completion.elapsed_seconds,
        first_event_seconds=completion.first_event_seconds,
        attempts=completion.attempts,
    )


@dataclass(frozen=True, slots=True)
class TokenizedCompletion:
    """One completion together with the exact token ids the engine read and then wrote.

    Only the interpretability capture reads the ids: it runs a HuggingFace forward pass over
    prompt+response and labels every position as prompt or response, so it needs the boundary in
    tokens rather than in characters. Reconstructing that by re-tokenising the decoded text would
    shift the labelling at any BPE or special-token boundary the decode does not round-trip, and the
    result would still be a well-formed tensor of the wrong positions. Every other caller reads
    ``completion`` and never looks at the ids.

    Tuples rather than lists so a record cannot be mutated after the fact by whoever holds it, which
    matters here because the ids ARE the provenance of everything measured off them.
    """

    completion: BedrockCompletion
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
