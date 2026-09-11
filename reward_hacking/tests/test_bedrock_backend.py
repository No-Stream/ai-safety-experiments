"""Offline tests for the Bedrock Converse backend (``reward_hacking/model_backend.py``).

Fully offline and credential-free: nothing here touches the network or an AWS profile, and the
suite passes with ``boto3`` absent from the venv, which it is by default (``boto3`` is the optional
``bedrock`` extra). Rather than mock AWS, these tests put a fake ``boto3`` and ``botocore.config``
into ``sys.modules`` so the real ``BedrockBackend.__init__`` runs unchanged against a fake client.
That matters: a test helper that rebuilt the request payload itself would be checking its own copy
of the logic rather than the shipped one.

Every check here targets one of the documented silent failures rather than the happy path, because
each produces green output while measuring nothing. The effort payload GPT-OSS ignores without
error. The ``inputTokens`` field the prompt cache empties out. The reasoning block a naive join
would score as part of the answer. The connection pool that quietly serialises a wide burst.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pytest
import urllib3.exceptions
from conftest import MEASURED_MIN_OUTPUT_BUDGET

from reward_hacking.model_backend import (
    BEDROCK_PROFILE_ENV,
    CALL_FAILED_STOP_REASON_PREFIX,
    DEFAULT_BEDROCK_MAX_ATTEMPTS,
    DEFAULT_BEDROCK_MAX_TOKENS,
    DEFAULT_BEDROCK_READ_TIMEOUT_SECONDS,
    STOP_REASON_DEADLINE_EXCEEDED,
    STOP_REASON_END_TURN,
    STOP_REASON_MAX_TOKENS,
    TRANSIENT_CLIENT_ERROR_CODES,
    Backend,
    BedrockBackend,
    BedrockCompletion,
    BedrockSamplingConfig,
    StreamAlignmentError,
    StreamingBackend,
    TokenUsage,
    _join_reasoning_blocks,
    _join_text_blocks,
    _resolve_reasoning_dialect,
    _sum_usage,
    build_backend,
    completion_telemetry,
    converse_supports_stop_sequences,
    converse_supports_temperature,
    generate_raw_in_chunks,
    is_incomplete_stop_reason,
    parse_converse_output,
    raw_response,
    stream_detailed_in_chunks,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

GPT_OSS = "openai.gpt-oss-120b-1:0"
LUNA = "global.openai.gpt-5.6-luna"
CLAUDE = "global.anthropic.claude-sonnet-5"

# Not a real profile name: a test naming one would undo the point of the environment variable.
STUB_PROFILE = "stub-bedrock-profile"


@pytest.fixture(autouse=True)
def stub_bedrock_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every construction here a profile to resolve, without touching a real AWS config.

    Autouse because the backend now reads its named profile from the environment and refuses to
    guess, so an unset variable would fail every test in this file for a reason none of them are
    about. The two tests that *are* about resolution override it themselves.
    """
    monkeypatch.setenv(BEDROCK_PROFILE_ENV, STUB_PROFILE)


class ReadTimeoutError(Exception):
    """Stands in for ``botocore.exceptions.ReadTimeoutError``: the socket went quiet mid-call.

    Named exactly as botocore names it, because the backend records ``type(error).__name__`` into
    the stop reason, so a stand-in under any other name would let the tests pass while the string
    that lands in a real artifact went unchecked.
    """


class ConnectTimeoutError(Exception):
    """Stands in for ``botocore.exceptions.ConnectTimeoutError``."""


class ConnectionClosedError(Exception):
    """Stands in for ``botocore.exceptions.ConnectionClosedError``."""


class EndpointConnectionError(Exception):
    """Stands in for ``botocore.exceptions.EndpointConnectionError``."""


class ClientError(Exception):
    """Stands in for ``botocore.exceptions.ClientError``: a modelled service error at the call site.

    The real class carries the service's error code under ``response["Error"]["Code"]`` and
    botocore's retry count under ``response["ResponseMetadata"]["RetryAttempts"]``; the stand-in
    reproduces both with the real constructor's signature, because the backend decides whether the
    failure costs one sample or the whole batch by reading the code, and records the attempt count
    off the metadata. A stand-in without them would leave both reads unchecked.
    """

    def __init__(self, error_response: dict[str, Any], operation_name: str) -> None:
        code = error_response.get("Error", {}).get("Code", "Unknown")
        super().__init__(f"An error occurred ({code}) when calling the {operation_name} operation")
        self.response = error_response
        self.operation_name = operation_name


class EventStreamError(ClientError):
    """Stands in for ``botocore.exceptions.EventStreamError``: a service error delivered mid-stream.

    The real class is what botocore's event-stream parser raises when the service sends a modelled
    exception (``internalServerException`` et al.) as an event part-way through a reply. It is a
    ``ClientError`` subclass, here as there, so it carries the service's error code under
    ``response``; the backend reads the code out of it into the stop reason, and a stand-in
    without it would leave that read unchecked.
    """


RETRIES_BEFORE_GIVING_UP = 4
"""How many retries the modelled-error fixtures report botocore made, so ``attempts`` reads 5."""


def _service_error(code: str) -> BaseException:
    """A modelled service error raised at the call site, after botocore's adaptive retries gave up.

    The shape a throttle that outlives ``max_attempts`` takes: botocore re-raises the last
    ``ClientError`` with the retry count on its response metadata.
    """
    return ClientError(
        {
            "Error": {"Code": code, "Message": f"{code} from the fixture"},
            "ResponseMetadata": {"RetryAttempts": RETRIES_BEFORE_GIVING_UP},
        },
        "ConverseStream",
    )


def _mid_stream_service_error() -> BaseException:
    """The failure that killed the GraderSays Sonnet pass: internalServerException mid-stream."""
    return EventStreamError(
        {
            "Error": {
                "Code": "internalServerException",
                "Message": "The system encountered an unexpected error during processing.",
            }
        },
        "ConverseStream",
    )


def _urllib3_read_timeout() -> BaseException:
    """The real class a Bedrock socket that goes quiet mid-reply raises, not a stand-in.

    The four above are faked because the backend resolves them out of ``sys.modules`` and the
    classes have to be the objects a test raises. These two are the genuine urllib3 exceptions,
    because being genuine is the whole point: the bug they cover was a closed list that named only
    botocore's spellings, and a stand-in named ``ReadTimeoutError`` would have matched that list
    and hidden it. ``HTTPConnectionPool`` opens no socket at construction, and the host is
    deliberately unresolvable so a regression that made this reach the network would fail loudly.
    """
    pool = urllib3.HTTPConnectionPool("bedrock-runtime.invalid")
    return urllib3.exceptions.ReadTimeoutError(pool, "/converse-stream", "Read timed out.")


def _urllib3_protocol_error() -> BaseException:
    """The real class a connection broken part-way through the reply body raises."""
    return urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead")


FRAGMENT_CHARS = 3
"""How finely the fakes chop a block's text into deltas.

Small enough that every scripted reply arrives in several events, which is what gives the
concatenation checks teeth: an accumulator that kept only the last fragment, or joined fragments
with a space the way whole blocks are joined, cannot pass a reply streamed three characters at a
time.
"""

DEADLINE_FOR_TESTS = 0.1
"""A wall-clock deadline short enough to fire inside a unit test."""

LONGER_THAN_THE_DEADLINE = 0.3
"""One event's delivery pause: three times the deadline, so it fires without depending on timing."""

EVENTS_BEFORE_THE_STREAM_DIES = 3
"""How much of a reply arrives before a mid-stream transport failure, in events.

Three is ``messageStart``, ``contentBlockStart`` and one text delta: past the point where the
response object was returned and the body started arriving, and short of the ``messageStop`` that
would make the stream a complete one. Both ends matter -- a failure on the first read would not
distinguish the streaming layer from the call, and one after ``messageStop`` would hit the
ended-without-a-stop path instead.
"""


def _fragments(text: str) -> list[str]:
    """Split one block's text the way Bedrock streams it: several deltas, nothing between them."""
    return [text[start : start + FRAGMENT_CHARS] for start in range(0, len(text), FRAGMENT_CHARS)]


def converse_stream_events(
    blocks: Sequence[Mapping[str, Any]],
    usage: Mapping[str, int],
    stop_reason: str = STOP_REASON_END_TURN,
) -> list[dict[str, Any]]:
    """Render a non-streaming Converse content-block list as the events Bedrock would stream for it.

    Deliberately driven from the *same* block list the non-streaming fixtures use, so a test can
    assert that the two paths produce an identical completion rather than that each produces what
    its own fixture said. Event shapes follow the ``ConverseStream`` output union in the installed
    botocore service model.

    A readable reasoning block also emits a trailing ``signature`` delta and a redacted one emits
    ``redactedContent``, because both are real members of the ``reasoningContent`` delta union and
    neither is text anybody can read -- an accumulator that appended them would put an attestation
    or a base64 blob into the reasoning field.
    """
    events: list[dict[str, Any]] = [{"messageStart": {"role": "assistant"}}]
    for index, block in enumerate(blocks):
        events.append({"contentBlockStart": {"start": {}, "contentBlockIndex": index}})
        reasoning: Mapping[str, Any] | None = block.get("reasoningContent")
        if reasoning is not None:
            events.extend(_reasoning_delta_events(reasoning, index))
        text: str | None = block.get("text")
        if text is not None:
            events.extend(
                {"contentBlockDelta": {"delta": {"text": piece}, "contentBlockIndex": index}}
                for piece in _fragments(text)
            )
        events.append({"contentBlockStop": {"contentBlockIndex": index}})
    events.append({"messageStop": {"stopReason": stop_reason}})
    events.append({"metadata": {"usage": dict(usage), "metrics": {"latencyMs": 12}}})
    return events


def _reasoning_delta_events(reasoning: Mapping[str, Any], index: int) -> list[dict[str, Any]]:
    """Render one reasoning block's deltas, readable or encrypted, plus its signature."""
    redacted = reasoning.get("redactedContent")
    if redacted is not None:
        return [
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"redactedContent": redacted}},
                    "contentBlockIndex": index,
                }
            }
        ]
    events: list[dict[str, Any]] = [
        {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"text": piece}},
                "contentBlockIndex": index,
            }
        }
        for piece in _fragments(reasoning["reasoningText"]["text"])
    ]
    events.append(
        {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"signature": "an-attestation-not-a-trace"}},
                "contentBlockIndex": index,
            }
        }
    )
    return events


class _FakeEventStream:
    """A stand-in for botocore's ``EventStream``: iterate it once, close it, and it remembers both.

    ``pause_before`` sleeps before delivering the event at that index, which is how a test makes a
    stream slow at a chosen point rather than uniformly -- the difference between "the deadline
    fires mid-reply" and "the deadline fires between messageStop and the metadata carrying the token
    counts", which the backend has to treat differently.

    ``delivered`` and ``closed`` are what let a test check that the abandoned events were genuinely
    left unread and the HTTP response released, rather than read and discarded.
    """

    def __init__(
        self, events: list[dict[str, Any]], *, pause_before: Mapping[int, float] | None = None
    ) -> None:
        self.events = events
        self.closed = False
        self.delivered = 0
        self._pause_before = dict(pause_before or {})

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index, event in enumerate(self.events):
            pause = self._pause_before.get(index)
            if pause is not None:
                time.sleep(pause)
            self.delivered += 1
            yield event

    def close(self) -> None:
        self.closed = True


class _StreamThatDiesMidReply:
    """An event stream that delivers a few events and then raises, where the real one does.

    *Where* the raise happens is the whole content of this class. Every other fake here raises out
    of ``converse_stream`` itself, which is the layer botocore wraps: a urllib3 timeout there is
    re-raised as ``botocore.exceptions.ReadTimeoutError`` and a broken connection as its
    ``ConnectionClosedError``. Once the response object has been handed back, the reply body is read
    by iterating this stream, outside that wrapping, and urllib3's own class is what escapes. A test
    that raised at the call site instead would pass against a backend that isolates neither.

    Dying before ``messageStop`` matters too: a stream that ends cleanly without one raises a
    ``RuntimeError`` by design, so a failure delivered after it would be testing that path instead.
    """

    def __init__(self, events: list[dict[str, Any]], error: Callable[[], BaseException]) -> None:
        self.events = events
        self.closed = False
        self._error = error

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.events
        raise self._error()

    def close(self) -> None:
        self.closed = True


class _FakeConverseStreamClient:
    """Records every request body it is handed and streams a Converse response back.

    With no scripted ``blocks``, the answer echoes the prompt back. That is what makes the
    order-preservation check able to fail: identical responses cannot distinguish a thread pool
    that reassembles in request order from one that does not.

    ``streams`` keeps every event stream handed out, so a test can ask afterwards whether one was
    closed and how many of its events were read.
    """

    def __init__(
        self,
        blocks: list[dict[str, Any]] | None = None,
        usage: dict[str, int] | None = None,
        *,
        stop_reason: str = STOP_REASON_END_TURN,
        pause_before: Mapping[int, float] | None = None,
        truncate_events: int | None = None,
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self.streams: list[_FakeEventStream] = []
        self._blocks = blocks
        self._usage = usage if usage is not None else {"inputTokens": 1, "outputTokens": 2}
        self._stop_reason = stop_reason
        self._pause_before = pause_before
        self._truncate_events = truncate_events

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        prompt = kwargs["messages"][0]["content"][0]["text"]
        blocks = self._blocks if self._blocks is not None else [{"text": f"echo:{prompt}"}]
        events = converse_stream_events(blocks, self._usage, self._stop_reason)
        if self._truncate_events is not None:
            events = events[: self._truncate_events]
        stream = _FakeEventStream(events, pause_before=self._pause_before)
        self.streams.append(stream)
        return {"stream": stream}


PER_CALL_USAGE = {"inputTokens": 3, "outputTokens": 7}


def _blew_up(prompt: str) -> BaseException:
    """The default failure: not a transport error, so the backend must let it crash the batch."""
    return RuntimeError(f"converse blew up on {prompt}")


_FailureShape = Literal["a dead call", "a reply that dies mid-stream", "a stream that ends early"]
"""Which of the three shapes a dead sample takes, each noticed at a different layer.

``a dead call`` raises out of ``converse_stream`` itself, the layer botocore wraps, so a urllib3
failure there reaches the backend under botocore's own class names. ``a reply that dies mid-stream``
moves the same raise into the iteration of the reply body, outside that wrapping and therefore
under different exception classes -- see :class:`_StreamThatDiesMidReply`.
``a stream that ends early`` raises nothing at all: the events simply run out before ``messageStop``
and our own accumulator is what notices, so the failure is born inside our loop, not a dependency.

One argument rather than a flag per shape, because the shapes exclude one another: a boolean each
lets a call spell "dies mid-stream and also ends early", which no reply can do, leaving the fake to
resolve it by silently preferring one -- and a test written against the losing flag would then be
checking a layer it never reached.
"""


class _SelectivelyFailingConverseClient:
    """Answers every prompt but the named ones, which raise the way a dead Converse call does.

    Keyed on the prompt text rather than a call counter, so which calls are billed does not depend
    on how the thread pool happens to interleave. ``billed`` is the ground truth these tests compare
    the run total against: it holds exactly the prompts that reached the endpoint and were paid for.
    ``delay`` makes a served call slower than the main thread's cancel loop, which is what lets a
    test observe that the calls after a failure were never started.

    ``error`` chooses which failure the named prompts raise, and that choice is the whole point of
    the two-tier contract: a transport failure is isolated into a counted record while anything else
    crashes the batch. Defaulting it to a plain ``RuntimeError`` keeps the crash-the-batch tests
    testing the crash path.

    ``failure_shape`` chooses *where* the failure lands, over the three layers :data:`_FailureShape`
    describes, and ``error`` is unread by the one shape that raises nothing. All three live on this
    client rather than in a double per shape so that which prompts are billed stays decided in one
    place, since that is what these tests compare the run total against.
    """

    def __init__(
        self,
        failing_prompts: set[str],
        usage: dict[str, int],
        *,
        delay: float = 0.0,
        error: Callable[[str], BaseException] = _blew_up,
        failure_shape: _FailureShape = "a dead call",
    ) -> None:
        self.failing_prompts = failing_prompts
        self._usage = usage
        self._delay = delay
        self._error = error
        self._failure_shape: _FailureShape = failure_shape
        self.billed: list[str] = []

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        prompt = kwargs["messages"][0]["content"][0]["text"]
        if prompt in self.failing_prompts:
            delivered = converse_stream_events([{"text": f"echo:{prompt}"}], self._usage)[
                :EVENTS_BEFORE_THE_STREAM_DIES
            ]
            match self._failure_shape:
                case "a dead call":
                    raise self._error(prompt)
                case "a reply that dies mid-stream":
                    return {
                        "stream": _StreamThatDiesMidReply(delivered, lambda: self._error(prompt))
                    }
                case "a stream that ends early":
                    return {"stream": _FakeEventStream(delivered)}
        time.sleep(self._delay)
        self.billed.append(prompt)
        events = converse_stream_events([{"text": f"echo:{prompt}"}], self._usage)
        return {"stream": _FakeEventStream(events)}

    def billed_usage(self) -> TokenUsage:
        """The token total the endpoint actually charged for, summed over the calls it served."""
        return TokenUsage(
            input_tokens=len(self.billed) * self._usage["inputTokens"],
            output_tokens=len(self.billed) * self._usage["outputTokens"],
        )


class _GatedConverseClient:
    """Serves every prompt at once except the gated ones, which block until the test opens the gate.

    A deterministic stand-in for one wedged call: ``started`` records the order calls REACHED the
    endpoint and ``served`` the order they returned, so a test can prove that calls filed behind the
    gated one in a later chunk were in flight while it still blocked -- the property the continuous
    queue exists for -- without sleeping and hoping the scheduler agrees.
    """

    def __init__(self, gated: set[str], *, error_on: set[str] | None = None) -> None:
        self.gate = threading.Event()
        self._gated = gated
        self._error_on = error_on or set()
        self.started: list[str] = []
        self.served: list[str] = []
        self._lock = threading.Lock()

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        prompt = kwargs["messages"][0]["content"][0]["text"]
        with self._lock:
            self.started.append(prompt)
        if prompt in self._gated:
            self.gate.wait(timeout=10)
        if prompt in self._error_on:
            raise RuntimeError(f"converse blew up on {prompt}")
        with self._lock:
            self.served.append(prompt)
        events = converse_stream_events([{"text": f"echo:{prompt}"}], PER_CALL_USAGE)
        return {"stream": _FakeEventStream(events)}


_FakeClient = _FakeConverseStreamClient | _SelectivelyFailingConverseClient | _GatedConverseClient


class _FakeBoto3:
    """A stand-in for boto3, recording what the backend asks the SDK to build."""

    def __init__(self, client: _FakeClient) -> None:
        self.client = client
        self.session_kwargs: dict[str, Any] = {}
        self.config_kwargs: dict[str, Any] = {}
        self.service_name = ""

    def make_session(self, **kwargs: Any) -> SimpleNamespace:
        self.session_kwargs = kwargs
        return SimpleNamespace(client=self.make_client)

    def make_client(self, service_name: str, **kwargs: Any) -> _FakeClient:
        self.service_name = service_name
        return self.client

    def make_config(self, **kwargs: Any) -> dict[str, Any]:
        self.config_kwargs = kwargs
        return kwargs


def _install_fake_boto3(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> _FakeBoto3:
    """Put fake boto3 modules in ``sys.modules`` so the lazy ``importlib`` import finds them.

    ``botocore.exceptions`` is faked alongside the other two because the backend resolves its
    closed list of isolate-this-call failures from it at construction, and those classes have to be
    the same objects a test raises for the ``except`` to match. Faking it also keeps the promise in
    this module's docstring: the suite still runs with ``boto3`` absent from the venv.
    """
    fake = _FakeBoto3(client)
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(Session=fake.make_session))
    monkeypatch.setitem(sys.modules, "botocore.config", SimpleNamespace(Config=fake.make_config))
    monkeypatch.setitem(
        sys.modules,
        "botocore.exceptions",
        SimpleNamespace(
            ReadTimeoutError=ReadTimeoutError,
            ConnectTimeoutError=ConnectTimeoutError,
            ConnectionClosedError=ConnectionClosedError,
            EndpointConnectionError=EndpointConnectionError,
            EventStreamError=EventStreamError,
            ClientError=ClientError,
        ),
    )
    return fake


def _build_backend(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    client: _FakeClient,
    **kwargs: Any,
) -> BedrockBackend:
    """Construct a real ``BedrockBackend`` over the fake SDK, with a small thread pool."""
    _install_fake_boto3(monkeypatch, client)
    return BedrockBackend(model_id, concurrency=kwargs.pop("concurrency", 2), **kwargs)


class TestReasoningDialectDispatch:
    """The two families spell effort differently, and GPT-OSS ignores the wrong spelling."""

    def test_gpt_oss_gets_the_flat_shape(self) -> None:
        assert _resolve_reasoning_dialect(GPT_OSS).request_fields("high") == {
            "reasoning_effort": "high"
        }

    def test_gpt_5_6_gets_the_nested_shape(self) -> None:
        assert _resolve_reasoning_dialect(LUNA).request_fields("high") == {
            "reasoning": {"effort": "high"}
        }

    def test_the_two_shapes_are_not_interchangeable(self) -> None:
        """SABOTAGE target: one dialect serving both families is the silent-default-effort bug."""
        assert _resolve_reasoning_dialect(GPT_OSS).request_fields("low") != (
            _resolve_reasoning_dialect(LUNA).request_fields("low")
        )

    def test_unrecognised_family_raises_rather_than_guessing(self) -> None:
        with pytest.raises(ValueError, match="no known reasoning-effort dialect"):
            _resolve_reasoning_dialect(CLAUDE)

    def test_minimal_is_a_gpt_oss_only_rung(self) -> None:
        """The ladders differ by exactly one level, and GPT-OSS takes a bad value silently."""
        assert _resolve_reasoning_dialect(GPT_OSS).request_fields("minimal") == {
            "reasoning_effort": "minimal"
        }
        with pytest.raises(ValueError, match="not accepted by this model family"):
            _resolve_reasoning_dialect(LUNA).request_fields("minimal")

    def test_out_of_range_effort_is_rejected_locally(self) -> None:
        with pytest.raises(ValueError, match="not accepted by this model family"):
            _resolve_reasoning_dialect(GPT_OSS).request_fields("bogus")


class TestRequestShaping:
    """What actually goes over the wire, read off the recorded request body."""

    def test_gpt_oss_effort_reaches_the_request_flat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _FakeConverseStreamClient()
        backend = _build_backend(
            monkeypatch, GPT_OSS, client, sampling=BedrockSamplingConfig(reasoning_effort="low")
        )
        backend.generate(["hello"])
        assert client.requests[0]["additionalModelRequestFields"] == {"reasoning_effort": "low"}

    def test_luna_effort_reaches_the_request_nested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _FakeConverseStreamClient()
        backend = _build_backend(
            monkeypatch, LUNA, client, sampling=BedrockSamplingConfig(reasoning_effort="high")
        )
        backend.generate(["hello"])
        assert client.requests[0]["additionalModelRequestFields"] == {
            "reasoning": {"effort": "high"}
        }

    def test_no_effort_field_when_none_requested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset effort sends no field at all, the shape verified against every model."""
        client = _FakeConverseStreamClient()
        backend = _build_backend(monkeypatch, CLAUDE, client)
        backend.generate(["hello"])
        assert "additionalModelRequestFields" not in client.requests[0]

    def test_prompt_is_wrapped_as_one_user_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _FakeConverseStreamClient()
        backend = _build_backend(monkeypatch, GPT_OSS, client)
        backend.generate(["what is 17*23?"])
        request = client.requests[0]
        assert request["modelId"] == GPT_OSS
        assert request["messages"] == [{"role": "user", "content": [{"text": "what is 17*23?"}]}]
        assert request["inferenceConfig"] == {"maxTokens": DEFAULT_BEDROCK_MAX_TOKENS}

    def test_optional_inference_knobs_are_omitted_unless_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: sending a default temperature would silently change every result."""
        client = _FakeConverseStreamClient()
        default = _build_backend(monkeypatch, GPT_OSS, client)
        default.generate(["hello"])
        assert "temperature" not in client.requests[0]["inferenceConfig"]
        assert "topP" not in client.requests[0]["inferenceConfig"]

        explicit = _build_backend(
            monkeypatch,
            GPT_OSS,
            client,
            sampling=BedrockSamplingConfig(max_tokens=64, temperature=0.5, top_p=0.9),
        )
        explicit.generate(["hello"])
        assert client.requests[1]["inferenceConfig"] == {
            "maxTokens": 64,
            "temperature": 0.5,
            "topP": 0.9,
        }

    def test_stop_sequences_reach_the_request_and_stay_out_of_it_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The agent harness's ``</run>`` stop rides in as ``stopSequences``; nobody else pays.

        Checked at the consumption point -- the request the fake client actually receives -- not
        on the config object, because a field the renderer never read would be a stop that stops
        nothing while the run's logs report it was configured.
        """
        client = _FakeConverseStreamClient()
        stopping = _build_backend(
            monkeypatch, GPT_OSS, client, sampling=BedrockSamplingConfig(stop_sequences=("</run>",))
        )
        stopping.generate(["prompt"])
        assert client.requests[0]["inferenceConfig"]["stopSequences"] == ["</run>"]

        plain = _build_backend(monkeypatch, GPT_OSS, client)
        plain.generate(["prompt"])
        assert "stopSequences" not in client.requests[1]["inferenceConfig"]

    def test_the_stop_refusing_families_are_the_probed_ones_and_nobody_else(self) -> None:
        """Both OpenAI families reject the field with a ValidationException (probed live
        2026-08-24); Anthropic accepts it, and an unlisted family reads as supporting so a wrong
        guess fails loudly on the first call instead of silently sampling without the stop.
        """
        assert not converse_supports_stop_sequences(GPT_OSS)
        assert not converse_supports_stop_sequences(LUNA)
        assert converse_supports_stop_sequences(CLAUDE)
        assert converse_supports_stop_sequences("some.vendor.never-probed-model")

    def test_the_temperature_refusing_family_is_the_probed_one_and_nobody_else(self) -> None:
        """gpt-5.6 rejects the temperature field with a ValidationException (probed live
        2026-08-31); gpt-oss accepted 1.0 across the 08-25 production batches, and an unlisted
        family reads as supporting so a wrong guess fails loudly on the first call.
        """
        assert not converse_supports_temperature(LUNA)
        assert converse_supports_temperature(GPT_OSS)
        assert converse_supports_temperature(CLAUDE)
        assert converse_supports_temperature("some.vendor.never-probed-model")

    def test_unrecognised_family_with_effort_fails_before_any_setup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mis-specified run must die at construction, not 2,000 paid calls in."""
        client = _FakeConverseStreamClient()
        fake = _install_fake_boto3(monkeypatch, client)
        with pytest.raises(ValueError, match="no known reasoning-effort dialect"):
            BedrockBackend(CLAUDE, sampling=BedrockSamplingConfig(reasoning_effort="high"))
        assert fake.session_kwargs == {}
        assert client.requests == []


class TestClientConfiguration:
    """The credential path and the connection pool, both of which fail quietly when wrong."""

    def test_defaults_to_the_profile_the_environment_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: the ambient default profile on this box cannot authenticate at all.

        Two invariants at once. A named profile is passed rather than ``None``, because the ambient
        chain here fails credential resolution outright. And the name comes from the environment
        rather than from this repository, which is public: a profile name is a machine-local value
        of exactly the kind ``reward_hacking/bedrock_batch.py`` already keeps out of tracked source.
        The unique value asserted below cannot come from a committed default.
        """
        monkeypatch.setenv(BEDROCK_PROFILE_ENV, "profile-only-this-test-knows")
        fake = _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        BedrockBackend(GPT_OSS)
        assert fake.session_kwargs["profile_name"] == "profile-only-this-test-knows"
        assert fake.session_kwargs["region_name"] == "us-west-2"
        assert fake.service_name == "bedrock-runtime"

    def test_unset_profile_variable_raises_instead_of_falling_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of the same invariant: no committed default is left to fall back to.

        A fallback is what would put a machine-local profile name back into tracked source, so the
        failure has to be a raise that names the variable. It also has to happen before the session
        is built, since a half-configured client is a credential error thousands of calls later.
        """
        monkeypatch.delenv(BEDROCK_PROFILE_ENV, raising=False)
        fake = _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        with pytest.raises(RuntimeError, match=BEDROCK_PROFILE_ENV):
            BedrockBackend(GPT_OSS)
        assert fake.session_kwargs == {}

    def test_blank_profile_variable_is_treated_as_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exported-but-empty variable is the common shell mistake, and boto3 would take it."""
        monkeypatch.setenv(BEDROCK_PROFILE_ENV, "   ")
        _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        with pytest.raises(RuntimeError, match=BEDROCK_PROFILE_ENV):
            BedrockBackend(GPT_OSS)

    def test_profile_none_falls_back_to_the_ambient_chain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        BedrockBackend(GPT_OSS, profile=None, region="us-east-1")
        assert fake.session_kwargs == {"profile_name": None, "region_name": "us-east-1"}

    def test_connection_pool_exceeds_the_worker_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: a pool narrower than the fan-out makes urllib3 serialise the burst."""
        fake = _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        BedrockBackend(GPT_OSS, concurrency=64)
        assert fake.config_kwargs["max_pool_connections"] > 64

    def test_pool_has_a_floor_at_low_concurrency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        BedrockBackend(GPT_OSS, concurrency=1)
        assert fake.config_kwargs["max_pool_connections"] == 20

    def test_retries_are_delegated_to_botocore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One of Bedrock's two transient-500 shapes maps to no modelled exception class."""
        fake = _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        BedrockBackend(GPT_OSS)
        assert fake.config_kwargs["retries"] == {"max_attempts": 5, "mode": "adaptive"}

    def test_zero_concurrency_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        with pytest.raises(ValueError, match="concurrency must be at least 1"):
            BedrockBackend(GPT_OSS, concurrency=0)


class TestResponseParsing:
    """Answer, reasoning and usage all come out of the same content-block list."""

    def test_answer_joins_text_blocks_only(self) -> None:
        """SABOTAGE target: joining every block folds the reasoning trace into the answer."""
        blocks: list[dict[str, Any]] = [
            {"reasoningContent": {"reasoningText": {"text": "the user wants a product"}}},
            {"text": "391"},
        ]
        assert _join_text_blocks(blocks) == "391"

    def test_multiple_text_blocks_join_with_a_space(self) -> None:
        assert _join_text_blocks([{"text": "a"}, {"text": "b"}]) == "a b"

    def test_reasoning_is_read_when_legible(self) -> None:
        blocks: list[dict[str, Any]] = [
            {"reasoningContent": {"reasoningText": {"text": "17*23 = 391"}}},
            {"text": "391"},
        ]
        assert _join_reasoning_blocks(blocks) == "17*23 = 391"

    def test_encrypted_reasoning_yields_empty_string(self) -> None:
        """Luna returns redactedContent, an opaque blob carrying no readable text."""
        blocks: list[dict[str, Any]] = [
            {"reasoningContent": {"redactedContent": "aGVsbG8="}},
            {"text": "391"},
        ]
        assert _join_reasoning_blocks(blocks) == ""
        assert _join_text_blocks(blocks) == "391"

    def test_generate_returns_one_answer_per_prompt_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _FakeConverseStreamClient()
        backend = _build_backend(monkeypatch, GPT_OSS, client)
        assert backend.generate(["a", "b", "c"]) == ["echo:a", "echo:b", "echo:c"]

    def test_generate_detailed_keeps_reasoning_and_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _FakeConverseStreamClient(
            blocks=[
                {"reasoningContent": {"reasoningText": {"text": "thinking"}}},
                {"text": "answer"},
            ],
            usage={"inputTokens": 10, "outputTokens": 4},
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client)
        completion = backend.generate_detailed(["p"])[0]
        assert completion.text == "answer"
        assert completion.reasoning == "thinking"
        assert completion.usage == TokenUsage(input_tokens=10, output_tokens=4)

    def test_the_stop_reason_is_read_off_the_response_body(self) -> None:
        """Without it, a reply that ran out of output budget is indistinguishable from a refusal."""
        body = {
            "output": {"message": {"content": [{"text": "half an ans"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 2},
            "stopReason": "max_tokens",
        }
        assert parse_converse_output(body).stop_reason == "max_tokens"

    def test_a_body_predating_the_field_reports_no_stop_reason(self) -> None:
        """A stored fixture or hand-built body has no ``stopReason``; absent is honestly None."""
        body = {
            "output": {"message": {"content": [{"text": "an answer"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 2},
        }
        assert parse_converse_output(body).stop_reason is None

    def test_backend_satisfies_the_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert isinstance(
            _build_backend(monkeypatch, GPT_OSS, _FakeConverseStreamClient()), Backend
        )


class TestTokenAccounting:
    """The prompt cache moves input between three fields, so all three must be summed."""

    def test_the_three_input_fields_are_summed(self) -> None:
        """SABOTAGE target: reading inputTokens alone reports a 1,621-token prompt as 2."""
        usage = _sum_usage({"inputTokens": 2, "cacheWriteInputTokens": 1619, "outputTokens": 700})
        assert usage.input_tokens == 1621
        assert usage.output_tokens == 700

    def test_cache_reads_count_the_same_as_cache_writes(self) -> None:
        """The same prompts flip from write to read on a later run; the total must not move."""
        written = _sum_usage({"inputTokens": 2, "cacheWriteInputTokens": 1619})
        read = _sum_usage({"inputTokens": 2, "cacheReadInputTokens": 1619})
        assert written.input_tokens == read.input_tokens == 1621

    def test_absent_keys_are_zero(self) -> None:
        assert _sum_usage({}) == TokenUsage(input_tokens=0, output_tokens=0)
        assert _sum_usage({"inputTokens": 5}) == TokenUsage(input_tokens=5, output_tokens=0)

    def test_explicit_zero_cache_fields_are_handled(self) -> None:
        """GPT-OSS does no caching and reports zero for both cache dimensions on every call."""
        usage = _sum_usage(
            {
                "inputTokens": 1700,
                "cacheReadInputTokens": 0,
                "cacheWriteInputTokens": 0,
                "outputTokens": 20,
            }
        )
        assert usage == TokenUsage(input_tokens=1700, output_tokens=20)

    def test_usage_accumulates_across_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _FakeConverseStreamClient(usage={"inputTokens": 3, "outputTokens": 7})
        backend = _build_backend(monkeypatch, GPT_OSS, client)
        backend.generate(["a", "b"])
        assert backend.usage == TokenUsage(input_tokens=6, output_tokens=14)
        backend.generate(["c"])
        assert backend.usage == TokenUsage(input_tokens=9, output_tokens=21)

    def test_the_manifest_spellings_are_read_for_every_counter(self) -> None:
        """The batch job manifest spells all four counters ``...Count``, input included.

        The asymmetry this pins: output and both cache counters accepted their ``...Count`` alias
        while plain input did not, so a manifest-shaped mapping reported output correctly and input
        as the cache counter alone -- a silent half-count of cost, since ``_alias_value`` warns only
        when two *present* values disagree.
        """
        usage = _sum_usage(
            {"inputTokenCount": 11, "cacheReadInputTokenCount": 4, "outputTokenCount": 7}
        )
        assert usage == TokenUsage(input_tokens=15, output_tokens=7, cache_read_input_tokens=4)

    def test_the_cache_split_is_kept_beside_the_unchanged_total(self) -> None:
        """SABOTAGE target: carving the cache counters out of ``input_tokens``.

        The total is what every record on disk and every cost script means by ``input_tokens``,
        so it must not move; the split rides beside it so a cost can finally be priced at the
        cached rate. Luna's own probe numbers (2,132 input of which 2,130 cache-read).
        """
        usage = _sum_usage(
            {
                "inputTokens": 2,
                "cacheReadInputTokens": 2130,
                "cacheWriteInputTokens": 0,
                "outputTokens": 22,
            }
        )
        assert usage.input_tokens == 2132
        assert usage.cache_read_input_tokens == 2130
        assert usage.cache_write_input_tokens == 0
        assert (
            usage.input_tokens - usage.cache_read_input_tokens - usage.cache_write_input_tokens == 2
        )

    def test_the_run_total_sums_the_cache_split_too(self) -> None:
        total = TokenUsage(
            input_tokens=10, output_tokens=1, cache_write_input_tokens=8
        ) + TokenUsage(input_tokens=10, output_tokens=1, cache_read_input_tokens=8)
        assert total == TokenUsage(
            input_tokens=20, output_tokens=2, cache_read_input_tokens=8, cache_write_input_tokens=8
        )


class TestPartialBatchFailureAccounting:
    """A raise mid-batch must not also erase the cost record of the calls already paid for."""

    def test_usage_of_the_completed_calls_survives_a_raise(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SABOTAGE target: ``pool.map`` re-raised at iteration time, before any usage was summed.

        Failing loudly is right; under-reporting what the failed batch spent is not. One worker and
        the failure last makes the count exact rather than a race: four calls reach the endpoint and
        are billed, so the run total has to carry all four even though the batch itself raises. The
        error log has to carry the same figure, since it is the only place an operator sees it.
        """
        client = _SelectivelyFailingConverseClient({"boom"}, PER_CALL_USAGE)
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="blew up on boom"):
            backend.generate_detailed(["a", "b", "c", "d", "boom"])
        assert client.billed == ["a", "b", "c", "d"]
        assert backend.usage == TokenUsage(input_tokens=12, output_tokens=28)
        assert "in=12 out=28" in caplog.text

    def test_nothing_billed_goes_unaccounted_whatever_the_pool_managed_to_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The invariant that holds under any interleaving: run total == what was charged for.

        With the failure mid-batch and more than one worker, how many calls get served is genuinely
        up to the scheduler, so the assertion is against the fake's own record of what it billed
        rather than a fixed number.
        """
        client = _SelectivelyFailingConverseClient({"c"}, PER_CALL_USAGE)
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=2)
        with pytest.raises(RuntimeError, match="blew up on c"):
            backend.generate_detailed(["a", "b", "c", "d", "e"])
        assert len(client.billed) >= 2  # a and b both complete before either worker reaches c
        assert backend.usage == client.billed_usage()

    def test_the_earliest_failure_is_the_one_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Preserved from ``pool.map``: with several failures the lowest-index one propagates."""
        client = _SelectivelyFailingConverseClient({"b", "d"}, PER_CALL_USAGE)
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=2)
        with pytest.raises(RuntimeError, match="blew up on b"):
            backend.generate_detailed(["a", "b", "c", "d"])
        assert backend.usage == client.billed_usage()

    def test_a_batch_that_is_going_to_raise_stops_paying(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Also preserved from ``pool.map``, and why accounting cannot just wait for the whole pool.

        Fixing the accounting by letting every future finish first would make a doomed batch pay for
        every remaining prompt, which on a 500-prompt sweep is most of the bill. The served calls
        are deliberately slower than the cancel loop, so a run that cancelled nothing bills all 30.
        """
        client = _SelectivelyFailingConverseClient({"boom"}, PER_CALL_USAGE, delay=0.01)
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        prompts = ["boom", *(f"p{index}" for index in range(30))]
        with pytest.raises(RuntimeError, match="blew up on boom"):
            backend.generate_detailed(prompts)
        assert len(client.billed) < 30
        assert backend.usage == client.billed_usage()


REASONING_AND_ANSWER = [
    {"reasoningContent": {"reasoningText": {"text": "17 times 23 is 391, so that is the answer"}}},
    {"text": "391"},
]
"""One reply's content blocks: a readable thinking trace, then the answer it arrived at.

The shape the whole streaming path has to preserve. It is deliberately long enough to arrive in many
deltas and its two blocks are deliberately both non-empty, because the failure worth catching is the
two collapsing into one -- a trace folded into the answer, or an answer with the trace glued in
front of it, either of which silently invalidates every downstream label built on the answer text.
"""


class TestStreamedRepliesRebuildTheSameRecord:
    """A streamed reply must come back as the record the non-streaming path would return."""

    def test_a_multi_block_reply_streams_back_with_its_trace_kept_separate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: collapsing the reasoning and text deltas into one accumulator.

        End to end through the real backend: request built, stream consumed, blocks rebuilt, body
        parsed. The two fields have to hold exactly their own block, and the answer in
        particular has to be short -- ``391`` and nothing else -- since a collapsed
        accumulator produces an answer
        that still *contains* the right value and would pass a substring check.
        """
        client = _FakeConverseStreamClient(
            blocks=REASONING_AND_ANSWER, usage={"inputTokens": 10, "outputTokens": 4}
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client)
        completion = backend.generate_detailed(["what is 17*23"])[0]
        assert completion.text == "391"
        assert completion.reasoning == "17 times 23 is 391, so that is the answer"
        assert completion.usage == TokenUsage(input_tokens=10, output_tokens=4)
        assert completion.stop_reason == STOP_REASON_END_TURN

    def test_the_streamed_record_equals_the_non_streaming_one_for_the_same_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The anti-drift check: one content-block list, two transports, one completion.

        Stronger than asserting each path against its own fixture, which is how two parsers drift
        while both stay green. The batch transport still parses a stored ``modelOutput`` body with
        :func:`parse_converse_output`, so if streaming reassembled blocks even slightly differently
        -- a lost fragment, a space between deltas, a reordered pair -- the same item sampled by the
        two routes would differ, and the difference would read as a property of the model.
        """
        usage = {"inputTokens": 21, "outputTokens": 5}
        client = _FakeConverseStreamClient(blocks=REASONING_AND_ANSWER, usage=usage)
        backend = _build_backend(monkeypatch, GPT_OSS, client)
        streamed = backend.generate_detailed(["p"])[0]
        non_streaming = parse_converse_output(
            {
                "output": {"message": {"content": REASONING_AND_ANSWER}},
                "usage": usage,
                "stopReason": STOP_REASON_END_TURN,
            }
        )
        # The live path alone can measure its own latency, so only the telemetry may differ.
        assert (
            dataclasses.replace(
                streamed, elapsed_seconds=None, first_event_seconds=None, attempts=None
            )
            == non_streaming
        )
        assert streamed.elapsed_seconds is not None
        assert streamed.first_event_seconds is not None
        assert non_streaming.elapsed_seconds is None

    def test_two_answer_blocks_join_with_a_space_but_their_deltas_do_not(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two joins are different operations, and only one of them inserts a separator."""
        client = _FakeConverseStreamClient(blocks=[{"text": "abcdefg"}, {"text": "hij"}])
        backend = _build_backend(monkeypatch, GPT_OSS, client)
        assert backend.generate(["p"]) == ["abcdefg hij"]

    def test_an_encrypted_trace_streams_back_as_no_reasoning_at_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A redacted trace and a signature are not text; appending either would fake a trace."""
        client = _FakeConverseStreamClient(
            blocks=[{"reasoningContent": {"redactedContent": "aGVsbG8="}}, {"text": "391"}]
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client)
        completion = backend.generate_detailed(["p"])[0]
        assert completion.reasoning == ""
        assert completion.text == "391"

    def test_the_signature_delta_never_lands_in_the_reasoning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The readable-trace fixture emits one, so this pins that it is skipped, not joined."""
        client = _FakeConverseStreamClient(blocks=REASONING_AND_ANSWER)
        backend = _build_backend(monkeypatch, GPT_OSS, client)
        completion = backend.generate_detailed(["p"])[0]
        assert "attestation" not in completion.reasoning
        assert "attestation" not in completion.text

    def test_the_model_stop_reason_is_carried_through_the_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cap hit must still read as a cap hit; ``messageStop`` is where the label comes from."""
        client = _FakeConverseStreamClient(
            blocks=[{"text": "half an ans"}], stop_reason=STOP_REASON_MAX_TOKENS
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client)
        completion = backend.generate_detailed(["p"])[0]
        assert completion.stop_reason == STOP_REASON_MAX_TOKENS
        assert not is_incomplete_stop_reason(completion.stop_reason)

    def test_the_streaming_api_is_the_one_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No fallback to the non-streaming call exists, so a fake without ``converse`` suffices."""
        client = _FakeConverseStreamClient()
        backend = _build_backend(monkeypatch, GPT_OSS, client)
        backend.generate(["p"])
        assert not hasattr(client, "converse")
        assert len(client.requests) == 1

    def test_a_stream_that_ends_without_a_message_stop_is_counted_not_mislabelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An event stream that just stops is not a deadline abandonment and must not read as one.

        Truncating before ``messageStop`` is the shape of a dropped connection that botocore did not
        turn into an exception. Reporting it as a deadline partial would put a cause we cannot name
        under a label that names one; it records under its own name instead, and this pins that the
        two labels stay distinct.
        """
        client = _FakeConverseStreamClient(blocks=[{"text": "abcdef"}], truncate_events=3)
        backend = _build_backend(monkeypatch, GPT_OSS, client)
        completion = backend.generate_detailed(["p"])[0]
        assert completion.stop_reason == (
            f"{CALL_FAILED_STOP_REASON_PREFIX}IncompleteConverseStreamError"
        )
        assert completion.stop_reason != STOP_REASON_DEADLINE_EXCEEDED
        assert is_incomplete_stop_reason(completion.stop_reason)
        assert completion.text == ""
        assert completion.usage == TokenUsage()


class TestWallClockDeadline:
    """A call that outlives its deadline hands back what arrived, marked, instead of being lost."""

    def test_a_stalled_call_comes_back_partial_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SABOTAGE target: a deadline check that never fires leaves the call unbounded.

        A twelve-character answer streams three characters at a time and the stream goes quiet
        partway through for longer than the deadline. What must come back is the prefix that did
        arrive, marked with the deadline label: not a raise, not an empty record, and not the whole
        reply. The unread events must stay unread and the stream must be closed, because an
        abandoned stream nobody closes holds its connection and lets the model keep generating on
        our bill.
        """
        client = _FakeConverseStreamClient(
            blocks=[{"text": "abcdefghijkl"}], pause_before={3: LONGER_THAN_THE_DEADLINE}
        )
        backend = _build_backend(
            monkeypatch, GPT_OSS, client, concurrency=1, deadline_seconds=DEADLINE_FOR_TESTS
        )
        with caplog.at_level(logging.WARNING):
            completion = backend.generate_detailed(["p"])[0]
        assert completion.stop_reason == STOP_REASON_DEADLINE_EXCEEDED
        assert completion.text == "abcdef"
        assert "abcdefghijkl".startswith(completion.text)
        assert is_incomplete_stop_reason(completion.stop_reason)
        stream = client.streams[0]
        assert stream.closed
        assert stream.delivered < len(stream.events)
        assert "lower bound" in caplog.text

    def test_a_partial_record_reads_as_neither_finished_nor_capped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The label has to be distinguishable, or an analysis counts the partial as a whole reply.

        ``end_turn`` would make it a model that answered; ``max_tokens`` would make it a truncation
        the model caused. It is neither, and ``recoverybench.grading`` puts an unrecognised stop
        reason in its own no-answer bucket rather than among the declines.
        """
        client = _FakeConverseStreamClient(
            blocks=[{"text": "abcdefghijkl"}], pause_before={3: LONGER_THAN_THE_DEADLINE}
        )
        backend = _build_backend(
            monkeypatch, GPT_OSS, client, concurrency=1, deadline_seconds=DEADLINE_FOR_TESTS
        )
        completion = backend.generate_detailed(["p"])[0]
        assert completion.stop_reason not in {STOP_REASON_END_TURN, STOP_REASON_MAX_TOKENS}
        assert is_incomplete_stop_reason(completion.stop_reason)

    def test_the_deadline_stops_applying_once_the_message_has_ended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: abandoning between ``messageStop`` and the ``metadata`` carrying usage.

        ``metadata`` is the last event of the stream and the only one carrying token counts, so a
        deadline that fires after the message ended but before it arrived would throw away the cost
        record of a call that had in fact completed -- reporting a complete reply as free. The pause
        here sits immediately before ``messageStop``, so the deadline is already blown by the time
        the message ends, and the run still has to record the usage.
        """
        events = converse_stream_events([{"text": "391"}], {"inputTokens": 9, "outputTokens": 2})
        stop_index = next(index for index, event in enumerate(events) if "messageStop" in event)
        client = _FakeConverseStreamClient(
            blocks=[{"text": "391"}],
            usage={"inputTokens": 9, "outputTokens": 2},
            pause_before={stop_index: LONGER_THAN_THE_DEADLINE},
        )
        backend = _build_backend(
            monkeypatch, GPT_OSS, client, concurrency=1, deadline_seconds=DEADLINE_FOR_TESTS
        )
        completion = backend.generate_detailed(["p"])[0]
        assert completion.stop_reason == STOP_REASON_END_TURN
        assert completion.text == "391"
        assert completion.usage == TokenUsage(input_tokens=9, output_tokens=2)
        assert not client.streams[0].closed

    def test_one_stalled_call_does_not_shorten_its_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every prompt gets a row, so the denominator is what was asked for, not what survived.

        Both calls in this batch stall, because the fake's pause schedule is per stream rather than
        per prompt; what the check is about is that the batch is still two records long and both
        carry the deadline label rather than the batch coming back short or empty.
        """
        client = _FakeConverseStreamClient(
            blocks=[{"text": "abcdefghijkl"}], pause_before={4: LONGER_THAN_THE_DEADLINE}
        )
        backend = _build_backend(
            monkeypatch, GPT_OSS, client, concurrency=2, deadline_seconds=DEADLINE_FOR_TESTS
        )
        completions = backend.generate_detailed(["p", "q"])
        assert len(completions) == 2
        assert all(is_incomplete_stop_reason(c.stop_reason) for c in completions)

    def test_an_abandoned_call_reports_no_usage_rather_than_a_guess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The counts never arrived, so the record says zero and the log calls the total a floor."""
        client = _FakeConverseStreamClient(
            blocks=[{"text": "abcdefghijkl"}],
            usage={"inputTokens": 500, "outputTokens": 900},
            pause_before={3: LONGER_THAN_THE_DEADLINE},
        )
        backend = _build_backend(
            monkeypatch, GPT_OSS, client, concurrency=1, deadline_seconds=DEADLINE_FOR_TESTS
        )
        backend.generate_detailed(["p"])
        assert backend.usage == TokenUsage()

    def test_a_non_positive_deadline_is_rejected_at_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A zero deadline would abandon every call on its first event, silently emptying a run."""
        _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        with pytest.raises(ValueError, match="deadline_seconds must be positive"):
            BedrockBackend(GPT_OSS, deadline_seconds=0)

    def test_the_bound_on_one_call_is_logged_at_construction(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A worst case nobody can read off the logs is a worst case nobody reports."""
        _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        with caplog.at_level(logging.INFO):
            BedrockBackend(GPT_OSS, deadline_seconds=100, read_timeout=10, max_attempts=2)
        assert "bounded at about 130s" in caplog.text

    def test_the_timeout_knobs_reach_the_botocore_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """They were baked in, and the workaround rebuilt ``_client`` behind the class's back."""
        fake = _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        BedrockBackend(GPT_OSS, read_timeout=750, max_attempts=2)
        assert fake.config_kwargs["read_timeout"] == 750
        assert fake.config_kwargs["retries"] == {"max_attempts": 2, "mode": "adaptive"}

    def test_the_defaults_leave_the_shipped_client_configuration_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Making the knobs settable must not quietly move what an unspecified run gets."""
        fake = _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        BedrockBackend(GPT_OSS)
        assert fake.config_kwargs["read_timeout"] == DEFAULT_BEDROCK_READ_TIMEOUT_SECONDS
        assert fake.config_kwargs["retries"]["max_attempts"] == DEFAULT_BEDROCK_MAX_ATTEMPTS


class TestPerCallFailureIsolation:
    """A transport failure costs one sample; anything else still crashes the batch it belongs to."""

    def test_a_read_timeout_becomes_a_counted_record_and_spares_its_siblings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: one stalled call discarding every sibling reply in the same batch.

        The measured failure this exists for: a MiniMax pass landed 4 of 8 requested samples on one
        item and 1 of 8 on another, because the batch raised on the first timeout. What has to come
        back is five records for five prompts -- four answers and one row naming its own failure --
        so the analysis divides by five rather than by whatever survived.
        """
        client = _SelectivelyFailingConverseClient(
            {"dead"},
            PER_CALL_USAGE,
            error=lambda prompt: ReadTimeoutError(f"read timed out on {prompt}"),
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        completions = backend.generate_detailed(["a", "b", "dead", "c", "d"])
        assert len(completions) == 5
        assert [c.text for c in completions] == ["echo:a", "echo:b", "", "echo:c", "echo:d"]
        failed = completions[2]
        assert failed.stop_reason == f"{CALL_FAILED_STOP_REASON_PREFIX}ReadTimeoutError"
        assert is_incomplete_stop_reason(failed.stop_reason)
        assert failed.usage == TokenUsage()
        assert client.billed == ["a", "b", "c", "d"]
        assert backend.usage == client.billed_usage()

    @pytest.mark.parametrize(
        "error_type",
        [ReadTimeoutError, ConnectTimeoutError, ConnectionClosedError, EndpointConnectionError],
    )
    def test_every_expected_transport_failure_is_isolated(
        self, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
    ) -> None:
        """All four members of the closed list, since a group listing three isolates three."""
        client = _SelectivelyFailingConverseClient(
            {"dead"}, PER_CALL_USAGE, error=lambda _: error_type("down")
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        completions = backend.generate_detailed(["dead", "a"])
        assert completions[0].stop_reason == (
            f"{CALL_FAILED_STOP_REASON_PREFIX}{error_type.__name__}"
        )
        assert completions[1].text == "echo:a"

    @pytest.mark.parametrize("error_factory", [_urllib3_read_timeout, _urllib3_protocol_error])
    def test_a_reply_that_dies_while_streaming_is_isolated_like_one_that_never_started(
        self, monkeypatch: pytest.MonkeyPatch, error_factory: Callable[[], BaseException]
    ) -> None:
        """SABOTAGE target: the streaming rewrite moving the body read outside this seam.

        Streaming the reply put the read inside this process, which is what let a deadline bound a
        call -- and it also moved the read out from under botocore's error wrapping. A socket that
        goes quiet mid-reply then raises urllib3's own ``ReadTimeoutError``, a different class from a
        different package than the botocore one this list named, so it escaped the seam and killed
        the whole batch: one dead call taking every sibling reply with it, at about 1 call in 100 on
        MiniMax M2.5 and rising with generation length -- the same correlation with reasoning length
        that biases a surviving denominator.

        The stop reason is the same string a pre-headers timeout would leave, because it is the same
        failure noticed at a different moment. ``ProtocolError`` is the exception: wrapped it would
        read ``ConnectionClosedError``, so the two spellings of one broken connection land under two
        labels. Both answer :func:`is_incomplete_stop_reason`, which is what analysis reads.
        """
        client = _SelectivelyFailingConverseClient(
            {"dead"},
            PER_CALL_USAGE,
            error=lambda _: error_factory(),
            failure_shape="a reply that dies mid-stream",
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        completions = backend.generate_detailed(["a", "dead", "b"])
        assert [c.text for c in completions] == ["echo:a", "", "echo:b"]
        failed = completions[1]
        expected_name = type(error_factory()).__name__
        assert failed.stop_reason == f"{CALL_FAILED_STOP_REASON_PREFIX}{expected_name}"
        assert is_incomplete_stop_reason(failed.stop_reason)
        assert failed.usage == TokenUsage()
        assert client.billed == ["a", "b"]
        assert backend.usage == client.billed_usage()

    def test_a_service_error_delivered_mid_stream_is_isolated_and_names_its_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: a modelled service error raised by the event-stream iterator.

        The measured failure this exists for: on 2026-08-23 Bedrock delivered an
        ``internalServerException`` as an event part-way through a Sonnet 5 reply, botocore raised
        it out of the iterator as ``EventStreamError`` -- a third class the closed list did not
        name -- and it escaped the seam and killed a 700-rollout GraderSays pass, discarding the 22
        sibling replies the batch had already been billed for.

        Isolating it is not a step toward blanket-except: botocore raises ``EventStreamError``
        only from the event-stream parser, after the request was accepted, so the request-time
        bugs the closed list exists to crash on (credentials, an unknown model id, a payload
        Bedrock rejects) still arrive as ``ClientError`` from the ``converse_stream`` call and
        still kill the batch. The stop reason carries the service's own error code, because every
        modelled error arrives under this one class name and the bare name would pool a throttle
        with an internal server error.
        """
        client = _SelectivelyFailingConverseClient(
            {"dead"},
            PER_CALL_USAGE,
            error=lambda _: _mid_stream_service_error(),
            failure_shape="a reply that dies mid-stream",
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        completions = backend.generate_detailed(["a", "dead", "b"])
        assert [c.text for c in completions] == ["echo:a", "", "echo:b"]
        failed = completions[1]
        assert failed.stop_reason == (
            f"{CALL_FAILED_STOP_REASON_PREFIX}EventStreamError:internalServerException"
        )
        assert is_incomplete_stop_reason(failed.stop_reason)
        assert failed.usage == TokenUsage()
        assert client.billed == ["a", "b"]
        assert backend.usage == client.billed_usage()

    def test_a_stream_that_ends_early_costs_one_sample_not_the_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: the third spelling of a dead reply, the one that raises nothing.

        The measured failure this exists for: on 2026-08-24 an Opus 5 ConverseStream ran out of
        events before ``messageStop`` with zero content blocks -- no transport exception, no
        exception event, the iterator just ended. The accumulator's own raise was a bare
        ``RuntimeError``, which no closed list may name, so it escaped the seam and discarded the
        59 completed sibling calls in its chunk. The raise now carries its own class, which the
        list names -- and which a genuine accumulator bug raising ``RuntimeError`` still does not
        match, so the negative controls below keep their teeth.
        """
        client = _SelectivelyFailingConverseClient(
            {"dead"}, PER_CALL_USAGE, failure_shape="a stream that ends early"
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        completions = backend.generate_detailed(["a", "dead", "b"])
        assert [c.text for c in completions] == ["echo:a", "", "echo:b"]
        failed = completions[1]
        assert failed.stop_reason == (
            f"{CALL_FAILED_STOP_REASON_PREFIX}IncompleteConverseStreamError"
        )
        assert is_incomplete_stop_reason(failed.stop_reason)
        assert failed.usage == TokenUsage()
        assert client.billed == ["a", "b"]
        assert backend.usage == client.billed_usage()

    def test_a_failure_outside_the_closed_list_still_crashes_the_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of the contract, and why the list is closed rather than ``Exception``.

        A malformed payload, an unknown model id or a credentials failure is a bug in the run. Left
        isolated it would be absorbed into hundreds of empty records that read downstream as a model
        with nothing to say, so it has to crash on the first call instead.
        """
        client = _SelectivelyFailingConverseClient({"bug"}, PER_CALL_USAGE)
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        with pytest.raises(RuntimeError, match="blew up on bug"):
            backend.generate_detailed(["a", "bug", "b"])

    def test_a_bug_that_surfaces_mid_stream_still_crashes_the_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bound on the two urllib3 names: the streaming path is not a blanket catch.

        Widening a closed list is the move that quietly turns it into ``except Exception``, and the
        streaming path is where that would be invisible -- a bug in the accumulator or a response
        shape nobody modelled surfaces exactly here, and absorbing it would produce the empty
        records the closed list exists to prevent. Same reply, same moment, non-transport class.
        """
        client = _SelectivelyFailingConverseClient(
            {"bug"}, PER_CALL_USAGE, failure_shape="a reply that dies mid-stream"
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        with pytest.raises(RuntimeError, match="blew up on bug"):
            backend.generate_detailed(["a", "bug", "b"])

    def test_the_batch_summary_counts_the_partial_records(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A completed count with no partial count beside it is the survivor rate being reported."""
        client = _SelectivelyFailingConverseClient(
            {"dead"}, PER_CALL_USAGE, error=lambda _: ReadTimeoutError("down")
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        with caplog.at_level(logging.INFO):
            backend.generate_detailed(["a", "dead", "b"])
        assert "returned 3 records for 3 prompts, 1 of them partial" in caplog.text

    def test_one_prompt_at_a_time_still_raises_so_a_caller_can_see_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_converse`` keeps raising, because a scratch runner sampling call-by-call catches it.

        ``docs/scratch/mechanism-scoping/c4_minimax_new_items.py`` calls ``_converse`` directly and
        mirrors the same closed list itself. If the isolation lived in ``_converse`` rather than a
        layer above it, that runner's own handler would never fire and it would record a timeout as
        a landed sample with an empty answer.
        """
        client = _SelectivelyFailingConverseClient(
            {"dead"}, PER_CALL_USAGE, error=lambda _: ReadTimeoutError("down")
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        with pytest.raises(ReadTimeoutError):
            backend._converse("dead")


class TestTransientServiceErrorsCostOneSample:
    """A modelled service error on the closed code list is one dead sample, never the batch.

    The measured incidents this exists for: 7 of 8 completed results discarded by one raising task, a
    single throttle on the 70th call that would have discarded all 72, and 22 billed Sonnet replies
    lost to one mid-stream error. Each was a code now on :data:`TRANSIENT_CLIENT_ERROR_CODES`,
    arriving as a ``ClientError`` whose class the transport list could not name without also
    naming every request bug.
    """

    def test_a_throttle_that_outlived_its_retries_returns_the_siblings_plus_one_failed_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: the k-1 finished completions discarded with the raise.

        One worker so the count is exact: the throttle lands on call 4 of 6, and what must come
        back is six records -- three answers, one row naming the throttle by code, two more
        answers -- rather than a raise that loses the three already billed.
        """
        client = _SelectivelyFailingConverseClient(
            {"throttled"},
            PER_CALL_USAGE,
            error=lambda _: _service_error("ThrottlingException"),
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        completions = backend.generate_detailed(["a", "b", "c", "throttled", "d", "e"])
        assert [c.text for c in completions] == [
            "echo:a",
            "echo:b",
            "echo:c",
            "",
            "echo:d",
            "echo:e",
        ]
        failed = completions[3]
        assert (
            failed.stop_reason == f"{CALL_FAILED_STOP_REASON_PREFIX}ClientError:ThrottlingException"
        )
        assert is_incomplete_stop_reason(failed.stop_reason)
        assert failed.usage == TokenUsage()
        assert failed.attempts == RETRIES_BEFORE_GIVING_UP + 1
        assert failed.elapsed_seconds is not None
        assert failed.first_event_seconds is None
        assert client.billed == ["a", "b", "c", "d", "e"]
        assert backend.usage == client.billed_usage()

    @pytest.mark.parametrize("code", sorted(TRANSIENT_CLIENT_ERROR_CODES))
    def test_every_code_on_the_closed_list_is_isolated(
        self, monkeypatch: pytest.MonkeyPatch, code: str
    ) -> None:
        """All members, since a list of four that isolates three is the bug it exists to fix."""
        client = _SelectivelyFailingConverseClient(
            {"dead"}, PER_CALL_USAGE, error=lambda _: _service_error(code)
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        completions = backend.generate_detailed(["dead", "a"])
        assert completions[0].stop_reason == f"{CALL_FAILED_STOP_REASON_PREFIX}ClientError:{code}"
        assert completions[1].text == "echo:a"

    @pytest.mark.parametrize(
        "code", ["ValidationException", "AccessDeniedException", "ResourceNotFoundException"]
    )
    def test_a_request_bug_still_crashes_the_batch(
        self, monkeypatch: pytest.MonkeyPatch, code: str
    ) -> None:
        """The other half of the contract, and the reason the list is a closed enumeration.

        SABOTAGE: widen ``TRANSIENT_CLIENT_ERROR_CODES`` to every ``ClientError`` and this goes
        red -- a rejected payload would be absorbed into hundreds of empty records that read as a
        model with nothing to say. Watched red under that sabotage before the list was closed.
        """
        client = _SelectivelyFailingConverseClient(
            {"bug"}, PER_CALL_USAGE, error=lambda _: _service_error(code)
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        with pytest.raises(ClientError, match=code):
            backend.generate_detailed(["a", "bug", "b"])
        assert client.billed == ["a"], "the batch stops paying at the bug"
        assert backend.usage == client.billed_usage()

    def test_the_closed_list_is_exactly_the_ten_documented_codes(self) -> None:
        """Pins the enumeration itself, so a widening has to be a visible edit here too.

        Three groups: the four codes botocore's adaptive retry gives up on; the two model-side codes
        botocore never retries (a 424 model error and the 408 model timeout), which the backlog
        named and the audit recorded as chunk killers; and the four bare HTTP status strings a 5xx
        with no service code carries after botocore's parser stamped the status on it.
        """
        documented = {
            "ThrottlingException",
            "ServiceUnavailableException",
            "ModelNotReadyException",
            "InternalServerException",
            "ModelErrorException",
            "ModelTimeoutException",
            "500",
            "502",
            "503",
            "504",
        }
        assert documented == TRANSIENT_CLIENT_ERROR_CODES


class TestPerCallTelemetryIsRecorded:
    """Latency, time-to-first-event and the attempt count land on the completion, not in a log."""

    def test_a_streamed_completion_carries_its_own_timing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: telemetry left at None, which is what a batch completion honestly has.

        The stream's first event is delayed, so the first-event time has to be at least that delay
        and the elapsed time at least as long again.
        """
        pause = 0.05
        client = _FakeConverseStreamClient(blocks=[{"text": "391"}], pause_before={0: pause})
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        completion = backend.generate_detailed(["p"])[0]
        assert completion.first_event_seconds is not None
        assert completion.elapsed_seconds is not None
        assert completion.first_event_seconds >= pause
        assert completion.elapsed_seconds >= completion.first_event_seconds

    def test_attempts_are_read_off_the_response_metadata_when_botocore_supplies_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """botocore stores retries, not attempts: ``RetryAttempts: 2`` is the third attempt."""
        client = _FakeConverseStreamClient(blocks=[{"text": "391"}])
        original = client.converse_stream

        def with_metadata(**kwargs: Any) -> dict[str, Any]:
            response = original(**kwargs)
            response["ResponseMetadata"] = {"RetryAttempts": 2}
            return response

        monkeypatch.setattr(client, "converse_stream", with_metadata)
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        assert backend.generate_detailed(["p"])[0].attempts == 3

    def test_attempts_are_honestly_absent_without_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _FakeConverseStreamClient(blocks=[{"text": "391"}])
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        assert backend.generate_detailed(["p"])[0].attempts is None

    def test_a_transport_failure_records_how_long_it_took_to_die(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wedge that burned a read timeout and a throttle refused in a second share a stop reason
        prefix; the elapsed time is what tells them apart afterwards."""
        client = _SelectivelyFailingConverseClient(
            {"dead"}, PER_CALL_USAGE, error=lambda _: ReadTimeoutError("down")
        )
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        failed = backend.generate_detailed(["dead"])[0]
        assert failed.elapsed_seconds is not None
        assert failed.first_event_seconds is None
        assert failed.attempts is None, "a transport failure carries no botocore response"

    def test_the_cache_split_reaches_the_completion_and_the_run_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        usage = {"inputTokens": 2, "cacheReadInputTokens": 2130, "outputTokens": 22}
        client = _FakeConverseStreamClient(blocks=[{"text": "391"}], usage=usage)
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        completion = backend.generate_detailed(["p"])[0]
        assert completion.usage.input_tokens == 2132
        assert completion.usage.cache_read_input_tokens == 2130
        assert backend.usage.cache_read_input_tokens == 2130

    def test_completion_telemetry_is_the_one_spelling_of_the_five_fields_the_raw_record_carries(
        self,
    ) -> None:
        """SABOTAGE target: the shared spelling. Drop a key from ``completion_telemetry`` and the raw
        record carries ``None`` where the completion had a value; rename one and ``raw_response``
        refuses the unknown field. Either way the rows a cost script joins stop agreeing."""
        completion = BedrockCompletion(
            text="391",
            reasoning="",
            usage=TokenUsage(input_tokens=2132, output_tokens=22, cache_read_input_tokens=2130),
            stop_reason=STOP_REASON_END_TURN,
            elapsed_seconds=12.5,
            first_event_seconds=9.25,
            attempts=2,
        )
        telemetry = completion_telemetry(completion)
        assert telemetry == {
            "cache_read_input_tokens": 2130,
            "cache_write_input_tokens": 0,
            "elapsed_seconds": 12.5,
            "first_event_seconds": 9.25,
            "attempts": 2,
        }
        record = dataclasses.asdict(raw_response(completion))
        assert {name: record[name] for name in telemetry} == telemetry


def _echo(prompt: str) -> BedrockCompletion:
    return BedrockCompletion(
        text=f"echo:{prompt}", reasoning="", usage=TokenUsage(), stop_reason="end_turn"
    )


def _texts(
    release: tuple[int, Sequence[tuple[int, BedrockCompletion]]],
) -> tuple[int, list[tuple[int, str]]]:
    chunk_index, pairs = release
    return chunk_index, [(position, completion.text) for position, completion in pairs]


class TestTheQueueStaysFullAcrossChunkBoundaries:
    """The barrier a pool-per-chunk pays on its slowest call is gone; persistence order is not.

    Measured: a 20-rollout chunk took 53 minutes while 19 of its 20 calls finished in under five, and
    a Sonnet pass paying that on nearly every chunk accumulated ~66 hours of wall clock for $0.
    """

    def test_the_backend_is_a_streaming_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _build_backend(monkeypatch, GPT_OSS, _FakeConverseStreamClient())
        assert isinstance(backend, StreamingBackend)

    def test_submit_stream_keeps_flowing_past_a_stalled_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: a queue that waits for the stalled call before taking the next prompt.

        Two workers, one of them wedged on the first prompt. Every other prompt must be answered
        and yielded while the wedge still holds -- the worker beside it takes them one after
        another -- and the wedged call must come back last, after the gate opens.
        """
        client = _GatedConverseClient({"wedged"})
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=2)
        stream = backend.submit_stream(["wedged", "a", "b", "c"])
        flowed = [next(stream) for _ in range(3)]
        assert sorted(index for index, _ in flowed) == [1, 2, 3]
        assert client.served == ["a", "b", "c"]
        assert "wedged" in client.started, "the wedged call was in flight the whole time"
        client.gate.set()
        index, completion = next(stream)
        assert (index, completion.text) == (0, "echo:wedged")
        with pytest.raises(StopIteration):
            next(stream)
        assert backend.usage == TokenUsage(input_tokens=12, output_tokens=28)

    def test_chunks_are_released_in_request_order_whatever_order_their_calls_finish(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What a runner persists is exactly what the per-chunk path persisted, in the same order.

        The wedged call sits in the FIRST chunk. The second chunk's calls all finish first, and
        they must still not be released until the first chunk is whole -- while, on the wire, they
        were started before the wedge cleared.
        """
        client = _GatedConverseClient({"wedged"})
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=2)
        chunks = [["wedged", "a"], ["b", "c"], ["d"]]

        def open_the_gate_once_the_later_chunks_ran() -> None:
            while len(client.served) < 4:
                time.sleep(0.005)
            client.gate.set()

        opener = threading.Thread(target=open_the_gate_once_the_later_chunks_ran)
        opener.start()
        released = [_texts(release) for release in stream_detailed_in_chunks(backend, chunks)]
        opener.join()
        assert released == [
            (0, [(0, "echo:wedged"), (1, "echo:a")]),
            (1, [(0, "echo:b"), (1, "echo:c")]),
            (2, [(0, "echo:d")]),
        ]
        assert client.served[:4] == ["a", "b", "c", "d"], "later chunks ran behind the wedge"
        assert client.served[-1] == "wedged"

    def test_a_non_streaming_backend_falls_back_to_one_call_per_chunk(self) -> None:
        """The fallback is the loop the runners used to write by hand, byte for byte."""

        class OneCallPerChunk:
            model_id = "per-chunk"
            transport = "per-chunk"

            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
                self.calls.append(list(prompts))
                return [_echo(prompt) for prompt in prompts]

            def generate(self, prompts: list[str]) -> list[str]:
                return [c.text for c in self.generate_detailed(prompts)]

        backend = OneCallPerChunk()
        assert not isinstance(backend, StreamingBackend)
        released = [
            _texts(release) for release in stream_detailed_in_chunks(backend, [["a", "b"], ["c"]])
        ]
        assert released == [(0, [(0, "echo:a"), (1, "echo:b")]), (1, [(0, "echo:c")])]
        assert backend.calls == [["a", "b"], ["c"]]

    @pytest.mark.parametrize("returned", [3, 5], ids=["one-short", "one-long"])
    def test_a_results_list_of_the_wrong_length_is_refused_before_any_of_its_chunk_is_yielded(
        self, returned: int
    ) -> None:
        """SABOTAGE target: the fallback branch's length check. Without it a backend that returns k-1
        completions for k prompts yields k-1 positions, and every consumer files the rows short while
        reporting the pass whole (reproduced 2026-09-02 on a judge stub: three rows on disk,
        ``judged: 4``, no raise); k+1 files a position no prompt has. The chunk before the bad one
        was yielded and stays yielded; nothing of the bad chunk is, and both counts are named."""

        class WrongLengthOnTheFourPromptChunk:
            model_id = "wrong-length"
            transport = "wrong-length"

            def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
                completions = [_echo(prompt) for prompt in [*prompts, "stray"]]
                return completions[: returned if len(prompts) == 4 else len(prompts)]

            def generate(self, prompts: list[str]) -> list[str]:
                return [c.text for c in self.generate_detailed(prompts)]

        backend = WrongLengthOnTheFourPromptChunk()
        assert not isinstance(backend, StreamingBackend)
        chunks = [["a", "b"], ["c", "d", "e", "f"]]
        released: list[tuple[int, list[tuple[int, str]]]] = []
        with pytest.raises(
            StreamAlignmentError, match=f"{returned} completions for the 4 prompts of chunk 2 of 2"
        ):
            released.extend(
                _texts(release) for release in stream_detailed_in_chunks(backend, chunks)
            )
        assert released == [(0, [(0, "echo:a"), (1, "echo:b")])]

    def test_the_raw_twin_maps_every_completion_through_raw_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _FakeConverseStreamClient(usage={"inputTokens": 3, "outputTokens": 7})
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=2)
        released = list(generate_raw_in_chunks(backend, [["a", "b"], ["c"]]))
        assert [(index, [(i, r.text) for i, r in pairs]) for index, pairs in released] == [
            (0, [(0, "echo:a"), (1, "echo:b")]),
            (1, [(0, "echo:c")]),
        ]
        first = released[0][1][0][1]
        assert (first.input_tokens, first.output_tokens) == (3, 7)
        assert first.elapsed_seconds is not None
        assert first.cache_read_input_tokens == 0

    def test_a_plain_backend_goes_through_the_raw_twin_with_absent_counts(self) -> None:
        from reward_hacking.model_backend import MockBackend  # noqa: PLC0415 - test-local import

        released = list(generate_raw_in_chunks(MockBackend(["x"]), [["a"], ["b", "c"]]))
        assert [(index, [r.text for _, r in pairs]) for index, pairs in released] == [
            (0, ["x"]),
            (1, ["x", "x"]),
        ]
        assert released[0][1][0][1].input_tokens is None
        assert released[0][1][0][1].elapsed_seconds is None

    def test_a_raise_hands_over_the_finished_part_of_every_unfinished_chunk_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: the completed calls of the chunk in flight discarded with the raise.

        One worker, a bug on the fourth prompt of six chunked in pairs. The first chunk is whole and
        released; the second chunk's first call finished and is handed over as a one-element list
        before the error propagates; the third chunk never started and yields nothing.
        """
        client = _GatedConverseClient(set(), error_on={"bug"})
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=1)
        released: list[tuple[int, list[tuple[int, str]]]] = []
        chunks = [["a", "b"], ["c", "bug"], ["d", "e"]]

        def persist_as_they_land() -> None:
            released.extend(
                _texts(release) for release in stream_detailed_in_chunks(backend, chunks)
            )

        with pytest.raises(RuntimeError, match="blew up on bug"):
            persist_as_they_land()
        assert released == [(0, [(0, "echo:a"), (1, "echo:b")]), (1, [(0, "echo:c")])]
        assert client.started == ["a", "b", "c", "bug"], "nothing after the bug was paid for"
        assert backend.usage == TokenUsage(input_tokens=9, output_tokens=21)

    def test_a_whole_later_chunk_is_handed_over_under_its_own_index_when_an_earlier_one_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SABOTAGE target: the consumer counting releases and pairing by position in the count.

        Caught offline, as a reply filed under a neighbouring key: two workers, the bug is the
        FIRST chunk's only prompt, and the whole second chunk finishes before the raise. The first
        chunk has nothing to hand over and is skipped; the second chunk is whole and must be
        released under index 1, never as "the first release". A consumer that counted releases
        would file c and d under the bug's prompt.
        """
        client = _GatedConverseClient({"bug"}, error_on={"bug"})
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=2)
        released: list[tuple[int, list[tuple[int, str]]]] = []
        chunks = [["bug"], ["c", "d"]]

        def open_the_gate_once_the_rest_ran() -> None:
            while len(client.served) < 2:
                time.sleep(0.005)
            client.gate.set()

        opener = threading.Thread(target=open_the_gate_once_the_rest_ran)
        opener.start()

        def persist_as_they_land() -> None:
            released.extend(
                _texts(release) for release in stream_detailed_in_chunks(backend, chunks)
            )

        with pytest.raises(RuntimeError, match="blew up on bug"):
            persist_as_they_land()
        opener.join()
        assert released == [(1, [(0, "echo:c"), (1, "echo:d")])]
        assert backend.usage == TokenUsage(input_tokens=6, output_tokens=14)

    def test_a_consumer_that_stops_reading_still_gets_its_in_flight_calls_billed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The run total must carry every call that reached the endpoint, read or not."""
        client = _GatedConverseClient({"wedged"})
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=2)
        stream = backend.submit_stream(["a", "wedged", "b"])
        first = next(stream)
        assert first[1].text in {"echo:a", "echo:b"}
        client.gate.set()
        with caplog.at_level(logging.WARNING):
            stream.close()
        # Two calls reached the endpoint: the one yielded and the wedged one waited out at close.
        assert backend.usage == TokenUsage(input_tokens=6, output_tokens=14)
        assert "nobody read them" in caplog.text

    def test_the_backlog_behind_a_wedged_head_chunk_is_named_in_the_log(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The exposure the continuous queue creates is logged as it grows, not discovered at a death.

        The wedged call is the whole first chunk; three later single-prompt chunks land behind it
        and sit in memory, billed and not on disk, until the gate opens. Every chunk's worth of
        growth is a WARNING naming how many completions a process death would lose.
        """
        client = _GatedConverseClient({"wedged"})
        backend = _build_backend(monkeypatch, GPT_OSS, client, concurrency=2)
        chunks = [["wedged"], ["a"], ["b"], ["c"]]

        def open_the_gate_once_the_rest_ran() -> None:
            while len(client.served) < 3:
                time.sleep(0.005)
            client.gate.set()

        opener = threading.Thread(target=open_the_gate_once_the_rest_ran)
        opener.start()
        with caplog.at_level(logging.WARNING):
            released = [_texts(release) for release in stream_detailed_in_chunks(backend, chunks)]
        opener.join()
        assert [index for index, _ in released] == [0, 1, 2, 3]
        backlog_lines = [
            record.getMessage()
            for record in caplog.records
            if "buffered in memory" in record.getMessage()
        ]
        assert backlog_lines, "no warning while three paid completions waited behind the wedge"
        assert (
            "3 finished completions are buffered in memory behind chunk 1 of 4" in backlog_lines[-1]
        )
        assert "waits on 1 of its 1 calls" in backlog_lines[-1]


class _ScriptedStream:
    """A streaming backend whose ``submit_stream`` replays a scripted index sequence, then may raise.

    Pure Python, no threads: the loop under test is :func:`_stream_chunks`'s bookkeeping,
    and the properties pinned below (what is handed over before which exception, what the index
    contract refuses) have to hold whatever a real pool does.
    """

    model_id = "scripted-stream"
    transport = "scripted-stream"

    def __init__(self, indices: Sequence[int], *, then_raise: BaseException | None = None) -> None:
        self.indices = list(indices)
        self.then_raise = then_raise

    def submit_stream(self, prompts: Iterable[str]) -> Iterator[tuple[int, BedrockCompletion]]:
        prompt_list = list(prompts)
        for index in self.indices:
            text = f"echo:{prompt_list[index]}" if 0 <= index < len(prompt_list) else "misfiled"
            yield (
                index,
                BedrockCompletion(
                    text=text, reasoning="", usage=TokenUsage(), stop_reason="end_turn"
                ),
            )
        if self.then_raise is not None:
            raise self.then_raise

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        by_index = dict(self.submit_stream(prompts))
        return [by_index[index] for index in range(len(prompts))]

    def generate(self, prompts: list[str]) -> list[str]:
        return [completion.text for completion in self.generate_detailed(prompts)]


class TestTheOrderingLoopHandsOverOnEveryDeathItCanAndTrustsNoBrokenStream:
    """What :func:`_stream_chunks` does when the stream ends other than cleanly.

    Two families. A death the process can still act on -- a Ctrl-C, a ``SystemExit``, a request bug
    -- must hand over every buffered completion first, because each is billed and a keyed resume
    fills exactly the gap they leave. A stream that breaks its index contract must hand over
    nothing, because the contract is what made the hand-over safe.
    """

    _CHUNKS: Sequence[Sequence[str]] = (("a", "b"), ("c", "d"), ("e",))

    def test_a_ctrl_c_hands_over_what_was_buffered_before_it_propagates(self) -> None:
        """SABOTAGE target: ``except Exception`` on the hand-over, which lets ``KeyboardInterrupt``
        skip it and lose every buffered completion after the in-flight calls were waited out.

        Chunk one is whole, chunk two has one of two landed when the Ctrl-C arrives. Both must
        reach the consumer, under their own indices, before the interrupt reaches it.
        """
        backend = _ScriptedStream([0, 1, 2], then_raise=KeyboardInterrupt())
        released: list[tuple[int, list[tuple[int, str]]]] = []

        def persist_as_they_land() -> None:
            released.extend(
                _texts(release) for release in stream_detailed_in_chunks(backend, self._CHUNKS)
            )

        with pytest.raises(KeyboardInterrupt):
            persist_as_they_land()
        assert released == [(0, [(0, "echo:a"), (1, "echo:b")]), (1, [(0, "echo:c")])]

    def test_a_system_exit_hands_over_the_same_way(self) -> None:
        backend = _ScriptedStream([1, 0, 4], then_raise=SystemExit(1))
        released: list[tuple[int, list[tuple[int, str]]]] = []
        with pytest.raises(SystemExit):
            released.extend(
                _texts(release) for release in stream_detailed_in_chunks(backend, self._CHUNKS)
            )
        assert released == [(0, [(0, "echo:a"), (1, "echo:b")]), (2, [(0, "echo:e")])]

    def test_a_stream_that_ends_short_is_a_transport_bug_not_a_finished_run(self) -> None:
        """SABOTAGE target: the post-loop alignment tripwire. A stream that stops yielding without
        raising used to return quietly, and the runner logged ``prompts 3/5 ran`` and returned as if
        finished. Chunk one was released before the end and stays released; the half-landed chunk
        two is NOT handed over, because a backend that ended short is not one whose indices are
        trusted onto disk."""
        backend = _ScriptedStream([0, 1, 2])
        released: list[tuple[int, list[tuple[int, str]]]] = []
        with pytest.raises(StreamAlignmentError, match="ended after 3 of 5 completions") as info:
            released.extend(
                _texts(release) for release in stream_detailed_in_chunks(backend, self._CHUNKS)
            )
        assert "2 of 3 chunks unreleased" in str(info.value)
        assert "missing indices [3, 4]" in str(info.value)
        assert released == [(0, [(0, "echo:a"), (1, "echo:b")])]

    def test_a_duplicate_index_is_refused_before_anything_more_is_released(self) -> None:
        backend = _ScriptedStream([0, 0, 1])
        with pytest.raises(StreamAlignmentError, match="index 0 twice"):
            list(stream_detailed_in_chunks(backend, self._CHUNKS))

    def test_an_index_outside_the_stream_is_refused(self) -> None:
        backend = _ScriptedStream([5])
        with pytest.raises(StreamAlignmentError, match="index 5 for a stream of 5 prompts"):
            list(stream_detailed_in_chunks(backend, self._CHUNKS))

    def test_a_broken_contract_hands_nothing_over_even_with_a_whole_chunk_buffered(self) -> None:
        """The refusal is deliberate: chunk two is whole in the buffer when the duplicate arrives,
        and it must stay unpersisted, since the same stream that duplicated one index may have
        misfiled another."""
        backend = _ScriptedStream([2, 3, 0, 0])
        released: list[tuple[int, list[tuple[int, str]]]] = []
        with pytest.raises(StreamAlignmentError):
            released.extend(
                _texts(release) for release in stream_detailed_in_chunks(backend, self._CHUNKS)
            )
        assert released == []

    def test_a_consumer_that_stops_reading_closes_the_loop_without_a_second_error(self) -> None:
        """SABOTAGE target: catching ``BaseException`` without re-raising ``GeneratorExit`` first.

        A consumer whose own write failed leaves the generator suspended; its close throws
        ``GeneratorExit`` into the loop, and a hand-over that yields inside that handler turns one
        failure into ``RuntimeError: generator ignored GeneratorExit`` on top of it.
        """
        # Index 2 lands before chunk one completes, so a later chunk's completion is BUFFERED when
        # the consumer closes; a hand-over run inside the GeneratorExit handler would have
        # something to yield, which is the second error this pins. (With nothing buffered the
        # sabotage passes silently -- watched, then this script was reordered.)
        stream = stream_detailed_in_chunks(_ScriptedStream([2, 0, 1, 3, 4]), self._CHUNKS)
        assert _texts(next(stream)) == (0, [(0, "echo:a"), (1, "echo:b")])
        stream.close()

    def test_a_clean_stream_releases_every_chunk_and_raises_nothing(self) -> None:
        backend = _ScriptedStream([4, 3, 2, 1, 0])
        released = [_texts(release) for release in stream_detailed_in_chunks(backend, self._CHUNKS)]
        assert released == [
            (0, [(0, "echo:a"), (1, "echo:b")]),
            (1, [(0, "echo:c"), (1, "echo:d")]),
            (2, [(0, "echo:e")]),
        ]


class _PerChunkOnly:
    """A detailed backend with no stream, so the primitive takes its one-call-per-chunk fallback."""

    model_id = "per-chunk-only"
    transport = "per-chunk-only"

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        return [_echo(prompt) for prompt in prompts]

    def generate(self, prompts: list[str]) -> list[str]:
        return [completion.text for completion in self.generate_detailed(prompts)]


class TestOutOfOrderPersistenceIsOptIn:
    """``persist_out_of_order`` releases a chunk the moment it is whole; the default holds request order.

    Pinned as a pair on the same scripted streams, so SABOTAGE in either direction -- the flag
    ignored, or the default flipped -- turns exactly one side red (watched both ways).
    """

    _CHUNKS: Sequence[Sequence[str]] = (("a", "b"), ("c", "d"), ("e",))
    _IN_ORDER: ClassVar[list[tuple[int, list[tuple[int, str]]]]] = [
        (0, [(0, "echo:a"), (1, "echo:b")]),
        (1, [(0, "echo:c"), (1, "echo:d")]),
        (2, [(0, "echo:e")]),
    ]

    def _released(
        self, backend: _ScriptedStream | _PerChunkOnly, *, flag: bool
    ) -> Iterator[tuple[int, list[tuple[int, str]]]]:
        # A generator, not a list: fed to ``extend`` it keeps what was yielded before a raise,
        # which is the property the two raising tests below observe.
        return (
            _texts(release)
            for release in stream_detailed_in_chunks(
                backend, self._CHUNKS, persist_out_of_order=flag
            )
        )

    def test_the_default_holds_the_head_chunk_in_front_of_whole_later_chunks(self) -> None:
        """Chunks two and three are whole while chunk one still waits on its wedge; they wait too."""
        assert list(self._released(_ScriptedStream([2, 3, 4, 0, 1]), flag=False)) == self._IN_ORDER

    def test_under_the_flag_a_whole_chunk_is_released_the_moment_it_lands(self) -> None:
        assert list(self._released(_ScriptedStream([2, 3, 4, 0, 1]), flag=True)) == [
            (1, [(0, "echo:c"), (1, "echo:d")]),
            (2, [(0, "echo:e")]),
            (0, [(0, "echo:a"), (1, "echo:b")]),
        ]

    def test_the_flag_relaxes_order_between_chunks_never_within_one(self) -> None:
        """A half-landed chunk is not released early: each chunk still waits for its own last call."""
        assert list(self._released(_ScriptedStream([0, 2, 4, 3, 1]), flag=True)) == [
            (2, [(0, "echo:e")]),
            (1, [(0, "echo:c"), (1, "echo:d")]),
            (0, [(0, "echo:a"), (1, "echo:b")]),
        ]

    def test_under_the_flag_a_raise_hands_over_the_partial_chunks_in_chunk_order(self) -> None:
        backend = _ScriptedStream([1, 2, 4], then_raise=KeyboardInterrupt())
        released: list[tuple[int, list[tuple[int, str]]]] = []
        with pytest.raises(KeyboardInterrupt):
            released.extend(self._released(backend, flag=True))
        assert released == [(2, [(0, "echo:e")]), (0, [(1, "echo:b")]), (1, [(0, "echo:c")])]

    def test_under_the_flag_a_short_stream_is_still_refused(self) -> None:
        released: list[tuple[int, list[tuple[int, str]]]] = []
        with pytest.raises(StreamAlignmentError, match="ended after 2 of 5 completions") as info:
            released.extend(self._released(_ScriptedStream([4, 0]), flag=True))
        assert "2 of 3 chunks unreleased" in str(info.value)
        assert "missing indices [1, 2, 3]" in str(info.value)
        assert released == [(2, [(0, "echo:e")])], "the whole chunk was released before the end"

    def test_under_the_flag_the_backlog_warning_counts_partial_chunks(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two completions in two half-landed chunks is a chunk's worth in memory, so it is named."""
        with caplog.at_level(logging.WARNING):
            list(self._released(_ScriptedStream([1, 3, 0, 2, 4]), flag=True))
        assert (
            "2 finished completions are buffered in memory across 2 partial chunks" in caplog.text
        )

    def test_an_empty_chunk_is_released_under_the_flag_as_the_default_releases_it(self) -> None:
        """SABOTAGE target: the flag's release step looking only at the landed index's chunk.

        An empty chunk has no index to land, so a release keyed on the landed index never reaches
        it, and the clean stream is then refused at its end for a chunk left unreleased. No caller
        produces one today (``chunk_cells`` groups cells, ``_run_leg`` slices a non-empty list);
        the two modes of one primitive still have to agree on it, and both have to agree with the
        per-chunk fallback, which yields the empty chunk under its index.
        """
        chunks: Sequence[Sequence[str]] = ((), ("a",), ())
        expected = {0: [], 1: [(0, "echo:a")], 2: []}
        for flag in (False, True):
            released = dict(
                _texts(release)
                for release in stream_detailed_in_chunks(
                    _ScriptedStream([0]), chunks, persist_out_of_order=flag
                )
            )
            assert released == expected, f"{flag=}"
        assert (
            dict(_texts(r) for r in stream_detailed_in_chunks(_PerChunkOnly(), chunks)) == expected
        )

    def test_the_per_chunk_fallback_ignores_the_flag(self) -> None:
        assert list(self._released(_PerChunkOnly(), flag=True)) == self._IN_ORDER

    def test_generate_raw_in_chunks_passes_the_flag_through(self) -> None:
        released = list(
            generate_raw_in_chunks(
                _ScriptedStream([4, 3, 2, 1, 0]), self._CHUNKS, persist_out_of_order=True
            )
        )
        assert [chunk_index for chunk_index, _ in released] == [2, 1, 0]
        assert [[response.text for _, response in pairs] for _, pairs in released] == [
            ["echo:e"],
            ["echo:c", "echo:d"],
            ["echo:a", "echo:b"],
        ]


class TestSamplingConfig:
    """Only the knobs Converse honours exist, so none can be dropped without a trace."""

    def test_optional_knobs_default_to_omitted(self) -> None:
        sampling = BedrockSamplingConfig()
        assert sampling.temperature is None
        assert sampling.top_p is None
        assert sampling.reasoning_effort is None
        assert sampling.max_tokens == DEFAULT_BEDROCK_MAX_TOKENS

    def test_the_default_output_cap_clears_a_reasoning_models_trace(self) -> None:
        """SABOTAGE target: at the old 2048 default this is the check that would have gone red.

        Asserted as a floor and not against ``DEFAULT_BEDROCK_MAX_TOKENS``, because comparing the
        constant to itself is a tautology -- it reddens whenever the number is edited and says
        nothing about whether the number is usable. That tautology is what let 2048 stand: the CLI
        tests asserted the resolved cap equalled the field default, which was true and useless.

        ``maxTokens`` is the one Converse field that is always in the request, so this default is
        always the cap unless a run overrides it, and a hosted model's reasoning is billed against
        it whether or not the reasoning is returned. Below the floor a reply is cut off before its
        answer, and a truncated reply is indistinguishable downstream from a refusal.
        """
        assert BedrockSamplingConfig().max_tokens >= MEASURED_MIN_OUTPUT_BUDGET

    def test_config_carries_no_knob_converse_cannot_express(self) -> None:
        """SABOTAGE target: a top_k or do_sample field here would be dropped silently.

        ``stop_sequences`` is in the allowlist because Converse *does* express it -- it becomes
        ``inferenceConfig.stopSequences``, verified live 2026-08-24 (the reply stops with
        ``stopReason == "stop_sequence"`` and the matched text stripped).
        """
        names = {field.name for field in dataclasses.fields(BedrockSamplingConfig)}
        assert names == {"max_tokens", "temperature", "top_p", "reasoning_effort", "stop_sequences"}


class TestBuildBackend:
    """The dispatcher knows the new kind, and its error message still lists every kind."""

    def test_bedrock_kind_is_routed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch, _FakeConverseStreamClient())
        backend = build_backend("bedrock", GPT_OSS, concurrency=2)
        assert isinstance(backend, BedrockBackend)
        assert backend.model_id == GPT_OSS

    def test_unknown_kind_message_lists_bedrock(self) -> None:
        with pytest.raises(ValueError, match="'bedrock'"):
            build_backend("openai", "whatever")
