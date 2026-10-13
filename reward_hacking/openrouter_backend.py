"""The streaming OpenAI-compatible (OpenRouter) backend, split out of ``model_backend`` (which re-exports it)."""

from __future__ import annotations

import datetime
import email.utils
import logging
import math
import os
import threading
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
        Iterator,
        Mapping,
    )

    from openai.types.chat import ChatCompletionChunk

import httpx2
from httpx2 import Timeout
from openai import APIConnectionError, APIError, APIStatusError, OpenAI

from reward_hacking.backend_types import (
    BedrockCompletion,
    TokenUsage,
)

logger = logging.getLogger("reward_hacking.model_backend")


OPENROUTER_MODEL_PREFIX = "openrouter:"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = 1_500.0
DEFAULT_OPENROUTER_CONCURRENCY = 8
DEFAULT_OPENROUTER_MAX_TOKENS = 120_000
DEFAULT_OPENROUTER_MAX_ATTEMPTS = 8
DEFAULT_OPENROUTER_REASONING_EFFORT = "medium"
"""The owner's default rung for every hosted receiver (2026-09-12): enough reasoning that a model is
not lazy, without buying its full test-time-compute scaling, which the bench does not depend on.
Pass ``reasoning_effort=None`` to send no reasoning field and take the provider default."""
DEFAULT_OPENROUTER_RETRY_BASE_SECONDS = 1.0
DEFAULT_OPENROUTER_RETRY_MAX_SECONDS = 120.0
DEFAULT_OPENROUTER_STREAM_IDLE_TIMEOUT_SECONDS = 120.0
OPENROUTER_RATE_LIMIT_STATUS = 429
OPENROUTER_SERVER_ERROR_MIN_STATUS = 500
OPENROUTER_SERVER_ERROR_MAX_STATUS = 599


class _OpenRouterGenerationError(RuntimeError):
    """A completed stream cannot be treated as a model answer and may be retried."""


class OpenRouterRetryExhaustedError(RuntimeError):
    """All configured attempts for one OpenRouter generation failed."""


class OpenRouterChunkStream(Protocol):
    """The OpenAI SDK stream methods used by :class:`OpenAICompatBackend`."""

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        """Yield parsed Chat Completions chunks."""
        ...

    def close(self) -> None:
        """Close the response when an attempt ends or is retried."""
        ...


class OpenAICompatBackend:
    """Streaming OpenAI-compatible backend for OpenRouter receiver models.

    OpenRouter's Chat Completions stream puts answer fragments in
    ``choices[].delta.content``. Reasoning is emitted either as a string in
    ``choices[].delta.reasoning`` (with ``reasoning_content`` as a legacy alias) or as
    ``choices[].delta.reasoning_details[]``; textual detail objects use ``text`` or ``summary``.
    The terminal usage chunk appears immediately before ``[DONE]`` and repeats the terminal
    ``finish_reason``. OpenRouter also sends a mid-stream provider failure as an SSE chunk with an
    ``error`` object and ``finish_reason="error"``. These shapes are documented at
    https://openrouter.ai/docs/api_reference/streaming and
    https://openrouter.ai/docs/guides/best-practices/reasoning-tokens (retrieved 2026-09-12).

    OpenRouter currently includes usage in every response by default and documents
    ``stream_options={{"include_usage": true}}`` as deprecated; this backend sends that option for
    compatibility with older OpenAI-compatible gateways and still requires usage from the terminal
    chunk. The OpenAI SDK's built-in retry policy is disabled so this backend can retry exactly HTTP
    429/5xx, connection and timeout failures, stalled reads, and provider-side failed generations.
    """

    transport = "openai-compatible"
    _normal_finish_reasons = frozenset(
        {"stop", "length", "max_tokens", "tool_calls", "function_call", "content_filter"}
    )

    def __init__(  # noqa: PLR0913 - flat request and retry knobs are independently run-shaping
        self,
        model_id: str,
        *,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: float = DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
        stream_idle_timeout: float = DEFAULT_OPENROUTER_STREAM_IDLE_TIMEOUT_SECONDS,
        concurrency: int = DEFAULT_OPENROUTER_CONCURRENCY,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = DEFAULT_OPENROUTER_REASONING_EFFORT,
        max_attempts: int = DEFAULT_OPENROUTER_MAX_ATTEMPTS,
        retry_base_seconds: float = DEFAULT_OPENROUTER_RETRY_BASE_SECONDS,
        retry_max_seconds: float = DEFAULT_OPENROUTER_RETRY_MAX_SECONDS,
    ) -> None:
        """Construct a client, resolving the API key before any request can be made."""
        api_key = os.environ.get(OPENROUTER_API_KEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{OPENROUTER_API_KEY_ENV} is not set; export it before constructing "
                "OpenAICompatBackend"
            )
        if not model_id.removeprefix(OPENROUTER_MODEL_PREFIX):
            raise ValueError("model_id must contain a model name after the openrouter: prefix")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if stream_idle_timeout <= 0:
            raise ValueError(f"stream_idle_timeout must be positive, got {stream_idle_timeout}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {concurrency}")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError(f"max_tokens must be positive when set, got {max_tokens}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        if retry_base_seconds < 0:
            raise ValueError(f"retry_base_seconds must be non-negative, got {retry_base_seconds}")
        if retry_max_seconds < retry_base_seconds:
            raise ValueError(
                "retry_max_seconds must be at least retry_base_seconds; "
                f"got {retry_max_seconds} < {retry_base_seconds}"
            )

        self.model_id = model_id.removeprefix(OPENROUTER_MODEL_PREFIX)
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.stream_idle_timeout = stream_idle_timeout
        self.usage = TokenUsage()
        self._usage_lock = threading.Lock()
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=Timeout(timeout, read=stream_idle_timeout),
            max_retries=0,
        )

    def _request_kwargs(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None,
        extra_body: Mapping[str, object] | None,
    ) -> dict[str, Any]:
        """Build one chat request without allowing extra fields to overwrite core fields."""
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if effective_max_tokens is not None and effective_max_tokens < 1:
            raise ValueError(f"max_tokens must be positive when set, got {effective_max_tokens}")

        request_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": self.model_id,
        }
        if effective_max_tokens is not None:
            request_kwargs["max_tokens"] = effective_max_tokens
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            request_kwargs["top_p"] = self.top_p

        request_extra_body = dict(extra_body) if extra_body is not None else {}
        core_fields = {
            "messages",
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "stream",
            "stream_options",
        }
        conflicts = core_fields.intersection(request_extra_body)
        if conflicts:
            raise ValueError(f"extra_body cannot override core request fields: {sorted(conflicts)}")
        if self.reasoning_effort is not None:
            if "reasoning" in request_extra_body or "reasoning_effort" in request_extra_body:
                raise ValueError(
                    "extra_body already specifies reasoning while reasoning_effort is configured"
                )
            request_extra_body["reasoning"] = {"effort": self.reasoning_effort}
        if request_extra_body:
            request_kwargs["extra_body"] = request_extra_body
        return request_kwargs

    @staticmethod
    def _reasoning_details_text(details: object) -> str:
        """Extract readable text and summaries from OpenRouter reasoning detail deltas."""
        if not isinstance(details, list):
            raise TypeError(
                f"OpenRouter reasoning_details must be a list, got {type(details).__name__}"
            )
        text_parts: list[str] = []
        for detail in details:
            if not isinstance(detail, dict):
                raise TypeError(
                    f"OpenRouter reasoning detail must be an object, got {type(detail).__name__}"
                )
            detail_text = detail.get("text", detail.get("summary"))
            if detail_text is not None:
                if not isinstance(detail_text, str):
                    raise TypeError(
                        "OpenRouter reasoning detail text/summary must be a string, got "
                        f"{type(detail_text).__name__}"
                    )
                text_parts.append(detail_text)
        return "".join(text_parts)

    @classmethod
    def _reasoning_text(cls, message: Mapping[str, Any]) -> str:
        """Read OpenRouter's readable reasoning field, including text in reasoning details."""
        reasoning = message.get("reasoning", message.get("reasoning_content"))
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise TypeError(
                    f"OpenRouter reasoning must be a string, got {type(reasoning).__name__}"
                )
            return reasoning
        details = message.get("reasoning_details")
        return "" if details is None else cls._reasoning_details_text(details)

    @classmethod
    def _chunk_reasoning_text(cls, delta: Mapping[str, Any]) -> str:
        """Read one streamed reasoning delta without duplicating two representations."""
        reasoning = delta.get("reasoning", delta.get("reasoning_content"))
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise TypeError(
                    f"OpenRouter reasoning must be a string, got {type(reasoning).__name__}"
                )
            return reasoning
        details = delta.get("reasoning_details")
        return "" if details is None else cls._reasoning_details_text(details)

    @staticmethod
    def _usage_from_chunk(raw_usage: Mapping[str, Any]) -> TokenUsage:
        """Map OpenRouter's final usage object into the shared accounting record."""
        prompt_details = raw_usage.get("prompt_tokens_details") or {}
        if not isinstance(prompt_details, dict):
            raise TypeError(
                "OpenRouter prompt_tokens_details must be an object when present, "
                f"got {type(prompt_details).__name__}"
            )
        return TokenUsage(
            input_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
            cache_read_input_tokens=int(prompt_details.get("cached_tokens", 0) or 0),
            cache_write_input_tokens=int(prompt_details.get("cache_write_tokens", 0) or 0),
        )

    @classmethod
    def _parse_choice(cls, raw_choice: object) -> tuple[str, str, str | None]:
        """Parse the first choice in a streamed chunk."""
        if not isinstance(raw_choice, dict):
            raise TypeError(f"OpenRouter choice must be an object, got {type(raw_choice).__name__}")
        raw_delta = raw_choice.get("delta", {})
        if not isinstance(raw_delta, dict):
            raise TypeError(f"OpenRouter delta must be an object, got {type(raw_delta).__name__}")
        content = raw_delta.get("content")
        text = ""
        if content is not None:
            if not isinstance(content, str):
                raise TypeError(
                    f"OpenRouter content delta must be a string, got {type(content).__name__}"
                )
            text = content
        finish_reason = raw_choice.get("finish_reason")
        stop_reason = None if finish_reason is None else str(finish_reason)
        return text, cls._chunk_reasoning_text(raw_delta), stop_reason

    @classmethod
    def _parse_chunk(
        cls, chunk: ChatCompletionChunk
    ) -> tuple[str, str, str | None, TokenUsage | None]:
        """Parse one OpenRouter chunk into answer, reasoning, finish, and usage fragments."""
        chunk_data = cast("Mapping[str, Any]", chunk.model_dump())
        stream_error = chunk_data.get("error")
        if stream_error is not None:
            raise _OpenRouterGenerationError(f"stream error: {stream_error}")

        raw_choices = chunk_data.get("choices", [])
        if not isinstance(raw_choices, list):
            raise TypeError(f"OpenRouter choices must be a list, got {type(raw_choices).__name__}")
        text = ""
        reasoning = ""
        stop_reason: str | None = None
        if raw_choices:
            text, reasoning, stop_reason = cls._parse_choice(raw_choices[0])

        raw_usage = chunk_data.get("usage")
        parsed_usage: TokenUsage | None = None
        if raw_usage is not None:
            if not isinstance(raw_usage, dict):
                raise TypeError(
                    f"OpenRouter usage must be an object, got {type(raw_usage).__name__}"
                )
            parsed_usage = cls._usage_from_chunk(raw_usage)
        return text, reasoning, stop_reason, parsed_usage

    @classmethod
    def _consume_stream(
        cls,
        stream: OpenRouterChunkStream,
        *,
        attempts: int,
        started: float,
    ) -> BedrockCompletion:
        """Accumulate one OpenRouter stream and reject incomplete/provider-error generations."""
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        final_usage: TokenUsage | None = None
        stop_reason: str | None = None
        first_event_at: float | None = None
        try:
            for chunk in stream:
                if first_event_at is None:
                    first_event_at = time.monotonic()
                text, reasoning, finish, chunk_usage = cls._parse_chunk(chunk)
                text_parts.append(text)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if finish is not None:
                    stop_reason = finish
                    if chunk_usage is not None:
                        final_usage = chunk_usage
        finally:
            stream.close()

        text = "".join(text_parts)
        if stop_reason == "error":
            raise _OpenRouterGenerationError("finish_reason='error'")
        if stop_reason is None:
            raise _OpenRouterGenerationError("stream ended without finish_reason")
        if not text and stop_reason not in cls._normal_finish_reasons:
            raise _OpenRouterGenerationError(
                f"empty content with non-normal finish_reason={stop_reason!r}"
            )
        if final_usage is None:
            raise _OpenRouterGenerationError("stream ended without a final usage chunk")
        elapsed_seconds = time.monotonic() - started
        return BedrockCompletion(
            text=text,
            reasoning="".join(reasoning_parts),
            usage=final_usage,
            stop_reason=stop_reason,
            elapsed_seconds=elapsed_seconds,
            first_event_seconds=None if first_event_at is None else first_event_at - started,
            attempts=attempts,
        )

    @staticmethod
    def _parse_retry_after(value: object) -> float | None:
        """Parse a provider retry hint expressed as seconds or an HTTP date."""
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            delay = float(value)
        elif isinstance(value, str):
            try:
                delay = float(value.strip())
            except ValueError:
                try:
                    retry_at = email.utils.parsedate_to_datetime(value)
                except (TypeError, ValueError, OverflowError):
                    return None
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=datetime.UTC)
                delay = (retry_at - datetime.datetime.now(datetime.UTC)).total_seconds()
        else:
            return None
        return delay if math.isfinite(delay) and delay >= 0 else None

    @classmethod
    def _collect_retry_after_hints(cls, value: object, source: str) -> list[tuple[float, str]]:
        """Collect retry hints from nested provider metadata."""
        if not isinstance(value, dict):
            return []
        hints: list[tuple[float, str]] = []
        for key, nested in value.items():
            if key == "retry_after_seconds":
                parsed = cls._parse_retry_after(nested)
                if parsed is not None:
                    hints.append((parsed, f"{source}.retry_after_seconds"))
            elif isinstance(key, str) and key.lower() == "retry-after":
                parsed = cls._parse_retry_after(nested)
                if parsed is not None:
                    hints.append((parsed, f"{source}.Retry-After"))
            elif isinstance(nested, dict):
                hints.extend(cls._collect_retry_after_hints(nested, f"{source}.{key}"))
        return hints

    @classmethod
    def _retry_after_hint(cls, error: BaseException) -> tuple[float, str] | None:
        """Return the largest provider retry hint carried by an OpenRouter status error."""
        if not isinstance(error, APIStatusError):
            return None

        hints = cls._collect_retry_after_hints(error.body, "response body")

        header_value = error.response.headers.get("retry-after")
        parsed_header = cls._parse_retry_after(header_value)
        if parsed_header is not None:
            hints.append((parsed_header, "response header Retry-After"))
        return max(hints, key=lambda hint: hint[0]) if hints else None

    def _retry_after(self, error: BaseException, attempt: int) -> None:
        """Sleep before a retry or raise a bounded error naming the final failure."""
        if attempt == self.max_attempts:
            raise OpenRouterRetryExhaustedError(
                f"OpenRouter generation failed after {attempt} attempts; last failure: "
                f"{type(error).__name__}: {error}"
            ) from error
        delay = min(self.retry_max_seconds, self.retry_base_seconds * (2 ** (attempt - 1)))
        retry_hint = self._retry_after_hint(error)
        if retry_hint is not None:
            hint_seconds, hint_source = retry_hint
            delay = max(delay, hint_seconds)
            reason = (
                f"{type(error).__name__}: {error}; provider retry hint "
                f"{hint_seconds:.1f}s ({hint_source})"
            )
        else:
            reason = f"{type(error).__name__}: {error}"
        logger.warning(
            "OpenAICompatBackend %s failed on attempt %d/%d (%s); retrying in %.1fs",
            self.model_id,
            attempt,
            self.max_attempts,
            reason,
            delay,
        )
        time.sleep(delay)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> BedrockCompletion:
        """Complete one OpenAI-format chat and add only a valid generation to the run total."""
        request_kwargs = self._request_kwargs(
            messages, max_tokens=max_tokens, extra_body=extra_body
        )
        started = time.monotonic()
        create = cast("Callable[..., OpenRouterChunkStream]", self._client.chat.completions.create)
        for attempt in range(1, self.max_attempts + 1):
            try:
                stream = create(
                    **request_kwargs,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                completion = self._consume_stream(stream, attempts=attempt, started=started)
            except APIStatusError as error:
                status_code = error.status_code
                retryable = status_code == OPENROUTER_RATE_LIMIT_STATUS or (
                    OPENROUTER_SERVER_ERROR_MIN_STATUS
                    <= status_code
                    <= OPENROUTER_SERVER_ERROR_MAX_STATUS
                )
                if not retryable:
                    raise
                self._retry_after(error, attempt)
            except (
                APIConnectionError,
                httpx2.TransportError,
                _OpenRouterGenerationError,
            ) as error:
                self._retry_after(error, attempt)
            except APIError as error:
                # OpenAI's Stream turns OpenRouter's mid-stream ``error`` SSE event into APIError.
                self._retry_after(error, attempt)
            else:
                with self._usage_lock:
                    self.usage += completion.usage
                return completion
        raise RuntimeError("OpenRouter request loop exited without a response")

    def _complete_prompt(self, prompt: str) -> BedrockCompletion:
        """Wrap one plain prompt as the single user message used by ``generate_detailed``."""
        return self.complete([{"role": "user", "content": prompt}])

    def submit_stream(self, prompts: Iterable[str]) -> Generator[tuple[int, BedrockCompletion]]:
        """Yield detailed completions as calls finish, keeping the request queue full."""
        pool = ThreadPoolExecutor(max_workers=self.concurrency)
        in_flight: dict[Future[BedrockCompletion], int] = {}
        queue = enumerate(prompts)
        first_error: tuple[int, BaseException] | None = None
        try:
            while True:
                while first_error is None and len(in_flight) < self.concurrency:
                    try:
                        index, prompt = next(queue)
                    except StopIteration:
                        break
                    in_flight[pool.submit(self._complete_prompt, prompt)] = index
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
                    yield index, future.result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if first_error is not None:
            raise first_error[1]

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        """Generate one detailed completion per prompt in request order."""
        by_index = dict(self.submit_stream(prompts))
        return [by_index[index] for index in range(len(prompts))]

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate one answer per prompt, discarding the detailed telemetry."""
        return [completion.text for completion in self.generate_detailed(prompts)]
