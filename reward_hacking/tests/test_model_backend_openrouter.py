"""Tests for the OpenRouter OpenAI-compatible receiver backend."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx2
import openai
import pytest

from reward_hacking import model_backend
from reward_hacking.model_backend import (
    DEFAULT_OPENROUTER_BASE_URL,
    OPENROUTER_MODEL_PREFIX,
    OpenAICompatBackend,
    TokenUsage,
    build_backend,
    generate_raw,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


MODEL_ID = "meta/muse-spark-1.3-contributor"
TEST_API_KEY = "test-openrouter-key"


def _completion_body(
    *,
    content: str = "answer",
    reasoning: str | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens": completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _backend_with_http_responses(
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[tuple[int, dict[str, Any]]],
    **backend_kwargs: Any,
) -> tuple[OpenAICompatBackend, list[httpx2.Request]]:
    monkeypatch.setenv("OPENROUTER_API_KEY", TEST_API_KEY)
    backend = OpenAICompatBackend(MODEL_ID, **backend_kwargs)
    requests: list[httpx2.Request] = []
    remaining = list(responses)

    def fake_send(request: httpx2.Request, **kwargs: object) -> httpx2.Response:
        del kwargs
        requests.append(request)
        if not remaining:
            raise AssertionError("HTTP response script was exhausted")
        status, body = remaining.pop(0)
        return httpx2.Response(status, json=body, request=request)

    monkeypatch.setattr(backend._client._client, "send", fake_send)
    return backend, requests


class TestOpenAICompatBackend:
    def test_missing_key_raises_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            OpenAICompatBackend(MODEL_ID)

    def test_completion_maps_to_the_shared_record_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, requests = _backend_with_http_responses(
            monkeypatch,
            [(200, _completion_body())],
            base_url="https://router.example/api/v1",
            max_tokens=321,
        )

        records = generate_raw(backend, ["solve this"])

        assert records[0].text == "answer"
        assert records[0].reasoning == ""
        assert records[0].input_tokens == 11
        assert records[0].output_tokens == 7
        assert records[0].stop_reason == "stop"
        assert records[0].cache_read_input_tokens == 3
        assert records[0].cache_write_input_tokens == 0
        assert records[0].attempts == 1
        assert records[0].elapsed_seconds is not None
        assert records[0].first_event_seconds is None
        assert backend.usage == TokenUsage(
            input_tokens=11, output_tokens=7, cache_read_input_tokens=3
        )

        request = requests[0]
        assert str(request.url) == "https://router.example/api/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {TEST_API_KEY}"
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request.content) == {
            "messages": [{"role": "user", "content": "solve this"}],
            "model": MODEL_ID,
            "max_tokens": 321,
        }

    def test_reasoning_text_is_captured_when_openrouter_returns_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, _ = _backend_with_http_responses(
            monkeypatch,
            [(200, _completion_body(reasoning="visible reasoning"))],
        )

        completion = backend.generate_detailed(["prompt"])[0]

        assert completion.reasoning == "visible reasoning"

    def test_retryable_429_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, requests = _backend_with_http_responses(
            monkeypatch,
            [(429, {"error": {"message": "slow down"}}), (200, _completion_body())],
            max_attempts=2,
        )
        sleeps: list[float] = []
        monkeypatch.setattr(model_backend.time, "sleep", sleeps.append)

        completion = backend.generate_detailed(["prompt"])[0]

        assert completion.text == "answer"
        assert completion.attempts == 2
        assert len(requests) == 2
        assert sleeps == [1.0]

    def test_bad_request_raises_without_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, requests = _backend_with_http_responses(
            monkeypatch,
            [
                (400, {"error": {"message": "unsupported reasoning setting"}}),
                (200, _completion_body()),
            ],
            max_attempts=3,
        )

        with pytest.raises(openai.BadRequestError):
            backend.generate_detailed(["prompt"])

        assert len(requests) == 1


class TestOpenRouterFactoryRouting:
    def test_model_prefix_constructs_the_openrouter_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", TEST_API_KEY)

        backend = build_backend("bedrock", f"{OPENROUTER_MODEL_PREFIX}{MODEL_ID}")

        assert isinstance(backend, OpenAICompatBackend)
        assert backend.model_id == MODEL_ID
        assert backend.transport == "openai-compatible"

    def test_reasoning_effort_is_sent_using_openrouter_unified_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend, requests = _backend_with_http_responses(
            monkeypatch,
            [(200, _completion_body())],
            reasoning_effort="high",
        )

        backend.generate(["prompt"])

        body = json.loads(requests[0].content)
        assert body["reasoning"] == {"effort": "high"}
        assert DEFAULT_OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
