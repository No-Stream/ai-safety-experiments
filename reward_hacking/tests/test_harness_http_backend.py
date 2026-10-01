"""Offline tests for the agent harness's external vLLM backend wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from reward_hacking.harness import loop
from reward_hacking.model_backend import SamplingConfig, VLLMHTTPBackend

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tokenizer_path(tmp_path: Path) -> Path:
    """Create the tokenizer identity used by the synthetic HTTP server."""
    unknown_token = "[UNK]"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({unknown_token: 0}, unk_token=unknown_token))
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>", "<|endoftext|>"]})
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] + ':' + message['content'] }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    tokenizer.save_pretrained(tmp_path)
    return tmp_path


def test_harness_builds_vllm_http_backend_with_cli_sampling_and_stop(
    tokenizer_path: Path,
) -> None:
    """The harness passes its run-block stop and local sampler to the external server backend."""
    args = loop._parse_args(
        [
            "--backend",
            "vllm-http",
            "--model-id",
            str(tokenizer_path),
            "--vllm-http-url",
            "http://127.0.0.1:8000",
            "--no-thinking",
            "--temperature",
            "0.7",
            "--top-p",
            "0.9",
            "--top-k",
            "23",
            "--max-new-tokens",
            "16384",
        ]
    )
    served = loop.served_model_from_args(args)
    backend = loop._build_cli_backend(args, served)

    assert isinstance(backend, VLLMHTTPBackend)
    assert backend.model_id == str(tokenizer_path)
    sampling = backend.sampling
    assert isinstance(sampling, SamplingConfig)
    assert sampling.temperature == 0.7
    assert sampling.top_p == 0.9
    assert sampling.top_k == 23
    assert sampling.max_new_tokens == 16384
    assert sampling.stop == ("</run>",)
    assert backend.stop_token_ids == (1, 2)
    assert backend.base_url == "http://127.0.0.1:8000"


def test_harness_refuses_full_weights_for_vllm_http() -> None:
    """An external server owns its weights, so the harness must reject a full-weight request."""
    args = loop._parse_args(
        ["--backend", "vllm-http", "--model-id", "synthetic", "--vllm-http-url", "http://server"]
    )
    served = loop.ServedModel(
        model_id="synthetic@revision",
        load_mode="full-weights",
        adapter_dir=None,
    )
    with pytest.raises(ValueError, match="full-weights"):
        loop._build_cli_backend(args, served)


def test_harness_requires_the_http_endpoint_url(tokenizer_path: Path) -> None:
    """A missing endpoint must fail before a server request can be attempted."""
    args = loop._parse_args(["--backend", "vllm-http", "--model-id", str(tokenizer_path)])
    with pytest.raises(ValueError, match="requires --vllm-http-url"):
        loop._build_cli_backend(args, loop.served_model_from_args(args))


def test_http_endpoint_flag_is_refused_on_other_backends() -> None:
    """An endpoint flag on a local in-process backend must not be silently ignored."""
    args = loop._parse_args(["--backend", "mock", "--vllm-http-url", "http://127.0.0.1:8000"])
    with pytest.raises(ValueError, match="--vllm-http-url"):
        loop._build_cli_backend(args)


def test_http_backend_refuses_in_process_engine_controls(tokenizer_path: Path) -> None:
    """An external server owns engine sizing, so local vLLM controls must fail loudly."""
    args = loop._parse_args(
        [
            "--backend",
            "vllm-http",
            "--model-id",
            str(tokenizer_path),
            "--vllm-http-url",
            "http://127.0.0.1:8000",
            "--vllm-max-num-seqs",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="--vllm-max-num-seqs"):
        loop._build_cli_backend(args)


def test_http_backend_refuses_adapter_kwargs() -> None:
    """A runtime adapter belongs to the external server configuration, not this client."""
    args = loop._parse_args(["--backend", "vllm-http", "--vllm-http-url", "http://127.0.0.1:8000"])
    with pytest.raises(ValueError, match="adapter"):
        loop.backend_cli.backend_from_args(
            args,
            "synthetic",
            extra_kwargs={"lora_adapter": "synthetic-adapter", "enable_lora": True},
        )
