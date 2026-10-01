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
        payload = {
            "choices": [
                {
                    "index": 0,
                    "text": "synthetic</run>",
                    "finish_reason": state["finish"],
                    "stop_reason": state["stop"],
                }
            ],
            "usage": {"prompt_tokens": 17, "completion_tokens": 3},
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
    url, requests, _ = server
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
