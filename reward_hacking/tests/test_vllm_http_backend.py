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
    state: dict[str, Any] = {
        "status": 200,
        "finish": "stop",
        "stop": "</run>",
        "barrier": None,
        "client_timeouts": [],
        "prompt_tokens": 17,
        "transport_error_type": None,
        "transport_errors_remaining": 0,
    }

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/completions"
        requests.append(json.loads(request.content))
        if state["transport_errors_remaining"]:
            state["transport_errors_remaining"] -= 1
            error_type = state["transport_error_type"]
            raise error_type("synthetic transport failure", request=request)
        if state["barrier"] is not None:
            state["barrier"].wait(timeout=10)
        text, finish, stop, completion_tokens = (
            state["replies"].pop(0)
            if state.get("replies")
            else (
                state.get("text", "synthetic</run>"),
                state["finish"],
                state["stop"],
                len(state["token_ids"]) if "token_ids" in state else 3,
            )
        )
        choice: dict[str, Any] = {
            "index": 0,
            "text": text,
            "finish_reason": finish,
            "stop_reason": stop,
        }
        if "token_ids" in state:
            choice["token_ids"] = state["token_ids"]
            if not state.get("omit_logprobs", False):
                choice["logprobs"] = {
                    "token_logprobs": state.get("token_logprobs", [-0.25] * len(state["token_ids"]))
                }
        if "prompt_token_ids" in state:
            choice["prompt_token_ids"] = state["prompt_token_ids"]
        elif requests[-1].get("return_token_ids"):
            choice["prompt_token_ids"] = requests[-1]["prompt"]
        payload: dict[str, Any] = {
            "choices": [choice],
            "usage": {
                "prompt_tokens": (
                    state["prompt_tokens"]
                    if "prompt_token_ids" in state or not requests[-1].get("return_token_ids")
                    else len(requests[-1]["prompt"])
                ),
                "completion_tokens": state.get("usage_completion_tokens", completion_tokens),
            },
        }
        return httpx2.Response(state["status"], json=state.get("payload", payload))

    client_class = httpx2.Client

    def client(**kwargs: Any) -> httpx2.Client:
        state["client_timeouts"].append(kwargs["timeout"])
        return client_class(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx2, "Client", client)
    return "http://synthetic.invalid", requests, state


def test_completion_holds_admission_through_http_request(
    tokenizer_path: Path, server: tuple[str, list[dict[str, Any]], dict[str, Any]]
) -> None:
    url, requests, state = server
    state["token_ids"] = [0, 0, 0]
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)
    events: list[str] = []

    class Admission:
        def request(self, *, prompt_tokens: int, max_tokens: int) -> Admission:
            assert prompt_tokens == 1
            assert max_tokens == 3
            return self

        def __enter__(self) -> None:
            events.append("admitted")

        def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
            assert requests
            events.append("released")

    backend.request_admission = Admission()  # type: ignore[assignment]
    backend.complete_rendered("hello", max_tokens=3, seed=1)
    assert events == ["admitted", "released"]


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
    local.chat_template_kwargs = {}
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
    assert requests[0]["prompt"] == served.tokenizer.encode(captured[0], add_special_tokens=True)
    assert captured[0].count("user:") == 2
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
    assert "return_token_ids" not in requests[0]
    assert completion.text == "synthetic</run>"
    assert completion.stop_reason == "stop_sequence"
    assert completion.usage.input_tokens == 17
    assert completion.usage.output_tokens == 3
    assert completion.attempts == 1
    assert completion.elapsed_seconds is not None


def test_qwen_ladder_http_sampling_request_includes_token_evidence_fields(
    tokenizer_path: Path, server: tuple[str, list[dict[str, Any]], dict[str, Any]]
) -> None:
    url, requests, state = server
    state["token_ids"] = [7]
    backend = VLLMHTTPBackend(
        "qwen-base",
        base_url=url,
        model_path=tokenizer_path,
        thinking=True,
        sampling=SamplingConfig(
            max_new_tokens=8192,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=0,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            seed=None,
        ),
        stop_token_ids=(248044, 248046),
    )

    backend.complete_rendered("synthetic Qwen prompt", max_tokens=8192, seed=None)

    assert requests == [
        {
            "model": "qwen-base",
            "prompt": backend.tokenizer.encode("synthetic Qwen prompt", add_special_tokens=False),
            "max_tokens": 8192,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "seed": None,
            "stop": None,
            "stop_token_ids": [248044, 248046],
            "include_stop_str_in_output": False,
            "add_special_tokens": False,
            "return_token_ids": True,
            "logprobs": 0,
            "n": 1,
            "stream": False,
        }
    ]


def test_qwen_ladder_http_sampling_request_carries_thinking_token_budget(
    tokenizer_path: Path, server: tuple[str, list[dict[str, Any]], dict[str, Any]]
) -> None:
    url, requests, state = server
    state["token_ids"] = [7]
    backend = VLLMHTTPBackend(
        "qwen-base",
        base_url=url,
        model_path=tokenizer_path,
        thinking=True,
        sampling=SamplingConfig(max_new_tokens=8192, thinking_token_budget=128),
        stop_token_ids=(248044, 248046),
    )

    backend.complete_rendered("synthetic Qwen prompt", max_tokens=8192, seed=None)

    assert requests[0]["thinking_token_budget"] == 128


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
        assert requests[1]["prompt"] == self.backend(url, tokenizer_path).tokenizer.encode(
            "model-specific-prefixuser:syntheticassistant:<think>draft <run>ls</run>",
            add_special_tokens=True,
        )
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


def test_complete_rendered_matches_eos_stop_token_from_returned_ids(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
) -> None:
    url, requests, state = server
    raw_completion = "reasoning</think>answer <run>raw output</run>"
    state["replies"] = [(raw_completion, "stop", None, 11)]
    state["token_ids"] = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 248046]
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
    assert completion.matched_stop == 248046
    assert completion.prompt_tokens == len(requests[0]["prompt"])
    assert completion.completion_tokens == 11
    assert completion.token_ids == tuple(state["token_ids"])
    assert completion.token_logprobs == (-0.25,) * 11
    assert requests[0]["prompt"] == backend.tokenizer.encode(
        "already rendered prompt", add_special_tokens=False
    )
    assert requests[0]["max_tokens"] == 23
    assert requests[0]["seed"] == 7
    assert requests[0]["stop"] is None
    assert requests[0]["include_stop_str_in_output"] is False
    assert requests[0]["stop_token_ids"] == [248044, 248046]
    assert requests[0]["add_special_tokens"] is False
    assert requests[0]["return_token_ids"] is True
    assert requests[0]["logprobs"] == 0


def test_complete_rendered_returns_sampled_token_evidence_and_prompt_ids(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
) -> None:
    url, requests, state = server
    state["token_ids"] = [31, 32, 33]
    state["token_logprobs"] = [-0.125, -0.75, -1.5]
    state["prompt_token_ids"] = [0]
    state["prompt_tokens"] = 1
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)

    completion = backend.complete_rendered("already rendered prompt", max_tokens=23, seed=7)

    assert completion.token_ids == (31, 32, 33)
    assert completion.token_logprobs == (-0.125, -0.75, -1.5)
    assert completion.prompt_token_ids == (0,)
    assert requests[0]["return_token_ids"] is True
    assert requests[0]["logprobs"] == 0


def test_complete_rendered_requires_returned_logprobs(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
) -> None:
    url, _, state = server
    state["token_ids"] = [12]
    state["omit_logprobs"] = True
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)

    with pytest.raises(ValueError, match="token_logprobs"):
        backend.complete_rendered("already rendered prompt", max_tokens=23, seed=7)


def test_complete_rendered_rejects_empty_token_evidence(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
) -> None:
    url, _, state = server
    state["token_ids"] = []
    state["token_logprobs"] = []
    state["usage_completion_tokens"] = 0
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)

    with pytest.raises(ValueError, match="no token ids"):
        backend.complete_rendered("already rendered prompt", max_tokens=23, seed=7)


@pytest.mark.parametrize(
    ("state_overrides", "message"),
    [
        ({"token_ids": [12, 13], "token_logprobs": [-0.2]}, "token ids and logprobs"),
        (
            {"token_ids": [12, 13], "usage_completion_tokens": 3},
            "token ids and output_tokens",
        ),
    ],
)
def test_complete_rendered_rejects_mismatched_token_evidence(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
    state_overrides: dict[str, Any],
    message: str,
) -> None:
    url, _, state = server
    state.update(state_overrides)
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)

    with pytest.raises(ValueError, match=message):
        backend.complete_rendered("already rendered prompt", max_tokens=23, seed=7)


def test_http_adapter_preflight_compares_adapter_alias_with_base(
    tokenizer_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, Any]] = []
    responses_differ = True

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/v1/models":
            return httpx2.Response(
                200,
                json={"data": [{"id": "base"}, {"id": "adapter"}]},
                request=request,
            )
        assert request.url.path == "/v1/completions"
        body = json.loads(request.content)
        requests.append(body)
        is_adapter = body["model"] == "adapter" and responses_differ
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "text": "base",
                        "finish_reason": "stop",
                        "stop_reason": None,
                        "token_ids": [1, 2],
                        "logprobs": {
                            "token_logprobs": [-0.2, -0.5] if is_adapter else [-0.3, -0.4]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
            request=request,
        )

    client_class = httpx2.Client
    monkeypatch.setattr(
        httpx2,
        "Client",
        lambda **kwargs: client_class(transport=httpx2.MockTransport(handle), **kwargs),
    )
    backend = VLLMHTTPBackend(
        "adapter", base_url="http://synthetic.invalid", model_path=tokenizer_path
    )

    backend.assert_adapter_changes_output(["Synthetic preflight prompt."], base_model_id="base")

    assert [request["model"] for request in requests] == ["adapter", "base"]
    assert all(request["temperature"] == 0.0 for request in requests)
    assert all(request["top_p"] == 1.0 and request["top_k"] == -1 for request in requests)
    assert all(request["return_token_ids"] is True for request in requests)
    assert all(request["logprobs"] == 1 for request in requests)

    responses_differ = False
    with pytest.raises(RuntimeError, match="changed nothing"):
        backend.assert_adapter_changes_output(["Synthetic preflight prompt."], base_model_id="base")


def test_complete_rendered_accepts_a_per_request_timeout(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
) -> None:
    url, _, state = server
    state["token_ids"] = [12]
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)

    backend.complete_rendered(
        "already rendered prompt",
        max_tokens=16384,
        seed=None,
        timeout_seconds=12.5,
    )

    assert state["client_timeouts"] == [12.5]


@pytest.mark.parametrize(
    ("max_tokens", "expected_timeout"),
    [(23, 1800.0), (16384, 4396.0)],
)
def test_completion_timeout_scales_with_max_tokens(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
    max_tokens: int,
    expected_timeout: float,
) -> None:
    url, _, state = server
    state["token_ids"] = [12]
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)

    backend.complete_rendered("already rendered prompt", max_tokens=max_tokens, seed=None)

    assert state["client_timeouts"] == [expected_timeout]


@pytest.mark.parametrize("error_type", [httpx2.ReadTimeout, httpx2.ConnectError])
def test_completion_retries_transport_errors(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
    error_type: type[httpx2.TransportError],
) -> None:
    url, requests, state = server
    state["token_ids"] = [12]
    state["transport_error_type"] = error_type
    state["transport_errors_remaining"] = 1
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)

    backend.complete_rendered("already rendered prompt", max_tokens=23, seed=5)

    assert len(requests) == 2
    assert requests[0] == requests[1]


def test_explicit_request_timeout_disables_retries(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
) -> None:
    url, requests, state = server
    state["transport_error_type"] = httpx2.ReadTimeout
    state["transport_errors_remaining"] = 1
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)

    with pytest.raises(httpx2.ReadTimeout, match="synthetic transport failure"):
        backend.complete_rendered(
            "already rendered prompt", max_tokens=16384, seed=5, timeout_seconds=12.5
        )

    assert len(requests) == 1
    assert state["client_timeouts"] == [12.5]


def test_completion_stops_after_bounded_transport_retries(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
) -> None:
    url, requests, state = server
    state["transport_error_type"] = httpx2.ReadTimeout
    state["transport_errors_remaining"] = 10
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)

    with pytest.raises(httpx2.ReadTimeout, match="synthetic transport failure"):
        backend.complete_rendered("already rendered prompt", max_tokens=23, seed=5)

    assert len(requests) == 3


@pytest.mark.parametrize("stop_reason", [248044, "</run>"])
def test_complete_rendered_prefers_explicit_stop_reason(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
    stop_reason: int | str,
) -> None:
    url, _, state = server
    state["replies"] = [("answer", "stop", stop_reason, 3)]
    state["token_ids"] = [12, 13, 14]
    backend = VLLMHTTPBackend(
        "synthetic",
        base_url=url,
        model_path=tokenizer_path,
        stop_token_ids=(248044, 248046),
    )

    completion = backend.complete_rendered("already rendered prompt", max_tokens=23, seed=7)

    assert completion.matched_stop == stop_reason


def test_complete_rendered_does_not_match_stop_token_when_length_capped(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
) -> None:
    url, _, state = server
    state["replies"] = [("answer", "length", None, 3)]
    state["token_ids"] = [12, 13, 248046]
    backend = VLLMHTTPBackend(
        "synthetic",
        base_url=url,
        model_path=tokenizer_path,
        stop_token_ids=(248044, 248046),
    )

    completion = backend.complete_rendered("already rendered prompt", max_tokens=23, seed=7)

    assert completion.matched_stop is None


def test_complete_rendered_requires_returned_token_ids(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
) -> None:
    url, _, _state = server
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)

    with pytest.raises(ValueError, match="token_ids"):
        backend.complete_rendered("already rendered prompt", max_tokens=23, seed=7)


def test_token_prompt_is_sent_without_reencoding_and_echo_is_checked(
    tokenizer_path: Path,
    server: tuple[str, list[dict[str, Any]], dict[str, Any]],
) -> None:
    url, requests, state = server
    state.update(token_ids=[31], prompt_token_ids=[12, 13], prompt_tokens=2)
    backend = VLLMHTTPBackend("synthetic", base_url=url, model_path=tokenizer_path)
    completion = backend.complete_token_ids([12, 13], max_tokens=23, seed=7)
    assert requests[0]["prompt"] == [12, 13]
    assert completion.prompt_token_ids == (12, 13)
    state["prompt_token_ids"] = [12, 14]
    with pytest.raises(ValueError, match="prompt token ids differ"):
        backend.complete_token_ids([12, 13], max_tokens=23, seed=7)
