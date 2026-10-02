"""CPU HTTP integration tests with synthetic transcripts and local chat templates."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx2
import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    from pathlib import Path

from reward_hacking.model_backend import SamplingConfig, VLLMBackend, VLLMHTTPBackend, build_backend


@pytest.fixture
def tokenizer_path(tmp_path: Path) -> Path:
    unknown_token = "[UNK]"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({unknown_token: 0}, unk_token=unknown_token))
    )
    tokenizer.chat_template = (
        "{{ 'model-specific-prefix' }}{% for message in messages %}"
        "{{ message['role'] + ':' + message['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
        "{% if enable_thinking %}<think>{% else %}<think></think>{% endif %}"
    )
    tokenizer.save_pretrained(tmp_path)
    return tmp_path


@pytest.fixture
def server(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    state: dict[str, Any] = {"status": 200, "finish": "stop", "stop": "</run>", "barrier": None}

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/completions"
        requests.append(json.loads(request.content))
        if state["barrier"] is not None:
            state["barrier"].wait(timeout=10)
        text, finish, stop, completion_tokens = (
            state["replies"].pop(0)
            if state.get("replies")
            else (state.get("text", "synthetic</run>"), state["finish"], state["stop"], 3)
        )
        payload = {
            "choices": [{"index": 0, "text": text, "finish_reason": finish, "stop_reason": stop}],
            "usage": {"prompt_tokens": 17, "completion_tokens": completion_tokens},
        }
        return httpx2.Response(state["status"], json=state.get("payload", payload))

    client_class = httpx2.Client

    def client(**kwargs: Any) -> httpx2.Client:
        return client_class(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx2, "Client", client)
    return "http://synthetic.invalid", requests, state


@pytest.mark.parametrize("thinking", [True, False])
def test_prompt_parity(
    tokenizer_path: Path, server: tuple[str, list[dict[str, Any]], dict[str, Any]], thinking: bool
) -> None:
    url, requests, state = server
    state["text"] = "</think>synthetic</run>"
    served = VLLMHTTPBackend(
        "synthetic", base_url=url, model_path=tokenizer_path, thinking=thinking
    )
    local = VLLMBackend.__new__(VLLMBackend)
    local._tokenizer = served.tokenizer
    local.thinking = thinking
    local._sampling_params = object()
    local._lora_request = None
    captured: list[str] = []

    def generate(prompts: list[str], *args: object, **kwargs: object) -> list[Any]:
        captured.extend(prompts)
        return [
            SimpleNamespace(
                prompt=prompt,
                prompt_token_ids=[1],
                outputs=[
                    SimpleNamespace(
                        text="synthetic",
                        token_ids=[2],
                        finish_reason="stop",
                        stop_reason=None,
                    )
                ],
            )
            for prompt in prompts
        ]

    local._llm = SimpleNamespace(generate=generate)
    transcript = (
        "user: synthetic first turn\nassistant: synthetic reply\nobservation: synthetic tool result"
    )
    local.generate_detailed([transcript])
    served.generate_detailed([transcript])
    assert requests[0]["prompt"] == captured[0]
    assert requests[0]["prompt"].count("user:") == 2
    assert requests[0]["add_special_tokens"] is True


def test_sampling_and_result(
    tokenizer_path: Path, server: tuple[str, list[dict[str, Any]], dict[str, Any]]
) -> None:
    url, requests, _ = server
    sampling = SamplingConfig(
        max_new_tokens=47,
        temperature=0.9,
        top_p=0.85,
        top_k=31,
        min_p=0.04,
        repetition_penalty=1.2,
        presence_penalty=1.5,
        stop=("</run>",),
        seed=19,
    )
    backend = build_backend(
        "vllm-http",
        "synthetic",
        base_url=url,
        model_path=tokenizer_path,
        sampling=sampling,
        stop_token_ids=(248044, 248046),
    )
    assert isinstance(backend, VLLMHTTPBackend)
    completion = backend.generate_detailed(["synthetic"])[0]
    assert requests == [
        {
            "model": "synthetic",
            "prompt": requests[0]["prompt"],
            "max_tokens": 47,
            "temperature": 0.9,
            "top_p": 0.85,
            "top_k": 31,
            "min_p": 0.04,
            "repetition_penalty": 1.2,
            "presence_penalty": 1.5,
            "stop": ["</run>"],
            "seed": 19,
            "stop_token_ids": [248044, 248046],
            "include_stop_str_in_output": True,
            "add_special_tokens": True,
            "n": 1,
            "stream": False,
        }
    ]
    assert completion.text == "synthetic</run>"
    assert completion.stop_reason == "stop_sequence"
    assert completion.usage.input_tokens == 17
    assert completion.usage.output_tokens == 3
    assert completion.attempts == 1
    assert completion.elapsed_seconds is not None


class TestThinkingSplit:
    def backend(self, url: str, tokenizer_path: Path) -> VLLMHTTPBackend:
        sampling = SamplingConfig(max_new_tokens=47, stop=("</run>",))
        return VLLMHTTPBackend(
            "synthetic", base_url=url, model_path=tokenizer_path, thinking=True, sampling=sampling
        )

    def test_answer_is_split_from_thinking(
        self, tokenizer_path: Path, server: tuple[str, list[dict[str, Any]], dict[str, Any]]
    ) -> None:
        url, requests, state = server
        state["replies"] = [("plan it\n</think>\n\nanswer <run>pwd</run>", "stop", "</run>", 9)]

        completion = self.backend(url, tokenizer_path).generate_detailed(["synthetic"])[0]

        assert len(requests) == 1
        assert completion.reasoning == "plan it\n"
        assert completion.text == "answer <run>pwd</run>"
        assert completion.stop_reason == "stop_sequence"

    def test_stop_inside_thinking_continues_the_same_turn(
        self, tokenizer_path: Path, server: tuple[str, list[dict[str, Any]], dict[str, Any]]
    ) -> None:
        url, requests, state = server
        state["replies"] = [
            ("draft <run>ls</run>", "stop", "</run>", 5),
            (" decide\n</think>\nreal <run>pwd</run>", "stop", "</run>", 8),
        ]

        completion = self.backend(url, tokenizer_path).generate_detailed(["synthetic"])[0]

        assert len(requests) == 2
        assert requests[1]["prompt"] == requests[0]["prompt"] + "draft <run>ls</run>"
        assert requests[1]["max_tokens"] == 47 - 5
        assert completion.reasoning == "draft <run>ls</run> decide\n"
        assert completion.text == "real <run>pwd</run>"
        assert completion.usage.output_tokens == 13
        assert completion.usage.input_tokens == 17

    def test_budget_spent_inside_thinking_leaves_an_empty_answer(
        self, tokenizer_path: Path, server: tuple[str, list[dict[str, Any]], dict[str, Any]]
    ) -> None:
        url, requests, state = server
        state["replies"] = [
            ("draft <run>ls</run>", "stop", "</run>", 40),
            (" still thinking", "length", None, 7),
        ]

        completion = self.backend(url, tokenizer_path).generate_detailed(["synthetic"])[0]

        assert len(requests) == 2
        assert completion.reasoning == "draft <run>ls</run> still thinking"
        assert completion.text == ""
        assert completion.stop_reason == "max_tokens"

    def test_a_stalled_continuation_fails_loudly(
        self, tokenizer_path: Path, server: tuple[str, list[dict[str, Any]], dict[str, Any]]
    ) -> None:
        url, _, state = server
        state["replies"] = [("draft <run>ls</run>", "stop", "</run>", 5), ("", "stop", "</run>", 0)]

        with pytest.raises(RuntimeError, match="no tokens"):
            self.backend(url, tokenizer_path).generate_detailed(["synthetic"])

    def test_server_response_without_usage_fails_before_returning_a_completion(
        self,
        tokenizer_path: Path,
        server: tuple[str, list[dict[str, Any]], dict[str, Any]],
    ) -> None:
        url, _requests, state = server
        state["payload"] = {
            "choices": [
                {
                    "index": 0,
                    "text": "synthetic</run>",
                    "finish_reason": "stop",
                    "stop_reason": "</run>",
                }
            ]
        }

        with pytest.raises(ValidationError, match="usage"):
            self.backend(url, tokenizer_path).generate_detailed(["synthetic"])


@pytest.mark.parametrize(
    ("finish", "stop", "expected"),
    [("length", None, "max_tokens"), ("stop", 248046, "end_turn"), ("stop", None, "end_turn")],
)
def test_stop_reasons(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
    finish: str,
    stop: int | None,
    expected: str,
) -> None:
    url, _, state = server
    state.update(finish=finish, stop=stop)
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)
    assert backend.generate_detailed(["synthetic"])[0].stop_reason == expected


@pytest.mark.parametrize("status", [201, 400, 500])
def test_failure_has_no_retry(
    tokenizer_path: Path, server: tuple[str, list[dict[str, Any]], dict[str, Any]], status: int
) -> None:
    url, requests, state = server
    state["status"] = status
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)
    with pytest.raises(ValueError if status == 201 else httpx2.HTTPStatusError):
        backend.generate(["synthetic"])
    assert len(requests) == 1


def test_concurrent_calls_overlap(
    tokenizer_path: Path, server: tuple[str, list[dict[str, Any]], dict[str, Any]]
) -> None:
    url, requests, state = server
    state["barrier"] = threading.Barrier(4)
    backend = VLLMHTTPBackend(
        "synthetic",
        base_url=url,
        model_path=tokenizer_path,
        sampling=SamplingConfig(max_new_tokens=5, do_sample=False),
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: backend.generate(["synthetic"]), range(4)))
    assert results == [["synthetic</run>"]] * 4
    assert len(requests) == 4
    assert all(request["temperature"] == 0.0 for request in requests)


def test_per_request_seeds_and_length_validation(
    tokenizer_path: Path, server: tuple[str, list[dict[str, Any]], dict[str, Any]]
) -> None:
    url, requests, _ = server
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)
    backend.generate(["first", "second"], generation_kwargs={"request_seeds": [3, 7]})
    assert [request["seed"] for request in requests] == [3, 7]
    with pytest.raises(ValueError, match="one seed per prompt"):
        backend.generate(["synthetic"], generation_kwargs={"request_seeds": [3, 7]})
    assert len(requests) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        {
            "choices": [
                {"index": 0, "text": "synthetic", "finish_reason": "stop", "stop_reason": None}
            ],
            "usage": {"prompt_tokens": -1, "completion_tokens": 1},
        },
        {
            "choices": [{"index": 0, "text": "synthetic", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    ],
)
def test_malformed_response_fails(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    url, _, state = server
    state["payload"] = payload
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)
    with pytest.raises(ValidationError):
        backend.generate(["synthetic"])


@pytest.mark.parametrize("matched_stop", [248046, "</run>", None])
def test_complete_rendered_returns_raw_stop_free_completion(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
    matched_stop: int | str | None,
) -> None:
    url, requests, state = server
    raw_completion = "reasoning</think>answer <run>raw output</run>"
    state["replies"] = [(raw_completion, "stop", matched_stop, 11)]
    backend = VLLMHTTPBackend(
        "synthetic",
        base_url=url,
        model_path=tokenizer_path,
        sampling=SamplingConfig(max_new_tokens=47, stop=("</run>",)),
        stop_token_ids=(248044, 248046),
    )

    completion = backend.complete_rendered("already rendered prompt", max_tokens=23, seed=7)

    assert completion.text == raw_completion
    assert completion.finish_reason == "stop"
    assert completion.matched_stop == matched_stop
    assert completion.prompt_tokens == 17
    assert completion.completion_tokens == 11
    assert requests[0]["prompt"] == "already rendered prompt"
    assert requests[0]["max_tokens"] == 23
    assert requests[0]["seed"] == 7
    assert requests[0]["stop"] is None
    assert requests[0]["include_stop_str_in_output"] is False
    assert requests[0]["stop_token_ids"] == [248044, 248046]
    assert requests[0]["add_special_tokens"] is False
