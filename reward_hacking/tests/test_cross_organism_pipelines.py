"""Synthetic tests for running the two organism screens on swapped bases."""

# pyright: reportArgumentType=false, reportPrivateUsage=false

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

from reward_hacking.atlas.capture import _followup_token_ids
from reward_hacking.atlas.cli import build_parser
from reward_hacking.atlas.sampling import (
    BASE_MODEL_ID,
    _assert_prompt_base_model,
    artifact_model_key,
    language_only_adapter_copy,
    resolve_model_key,
    resume_records_atlas_compatible,
)
from reward_hacking.interp.jsonl_resume import (
    ResumeLedger,
    ResumeMismatchError,
    ledger_path_for,
)
from reward_hacking.organisms import aisi_codecontests
from reward_hacking.organisms.aisi_codecontests import (
    BASE_MODEL,
    GEMMA_3_BASE_MODEL,
    VLLMCompletionSampler,
    _build_parser,
    _render_chat_prompt,
    resolve_vllm_base_settings,
)


class _ChatMLTokenizer:
    """Tiny tokenizer with Gemma-like and OLMo-like turn boundaries."""

    eos_token = "<|endoftext|>"

    def __init__(self) -> None:
        self._ids = {
            "<|im_end|>": 11,
            "<|endoftext|>": 12,
            "Yes": 13,
        }

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool = False,
    ) -> str:
        del tokenize
        rendered: list[str] = []
        for index, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            rendered.append(f"<{role}>{content}")
            if role == "assistant" and index != len(messages) - 1:
                rendered.append("<|im_end|>")
            else:
                rendered.append("<|endoftext|>")
        if add_generation_prompt:
            rendered.append("<assistant>")
        return "".join(rendered)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        output: list[int] = []
        remaining = text
        while remaining:
            marker = next((marker for marker in self._ids if remaining.startswith(marker)), None)
            if marker is None:
                output.append(ord(remaining[0]) + 100)
                remaining = remaining[1:]
            else:
                output.append(self._ids[marker])
                remaining = remaining[len(marker) :]
        return output

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids[token]


class _RecordingChatTokenizer:
    def __init__(self, *, reject_system: bool = False) -> None:
        self.reject_system = reject_system
        self.messages: list[dict[str, str]] = []
        self.calls: list[tuple[bool, bool]] = []

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.messages = [dict(message) for message in messages]
        self.calls.append((tokenize, add_generation_prompt))
        if self.reject_system and any(message["role"] == "system" for message in messages):
            raise ValueError("this template does not support system messages")
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


def test_followup_uses_olmo_closure_and_keeps_sampled_tokens() -> None:
    tokenizer = _ChatMLTokenizer()
    sampled_completion = [501, 502]

    followup = _followup_token_ids(
        tokenizer,
        task_text="synthetic task",
        test_list=["assert synthetic_function(1) == 2"],
        assistant_text="```python\nreturn 2\n```",
        prompt_token_ids=[1, 2],
        completion_token_ids=sampled_completion,
    )

    assert followup[0] == tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert sampled_completion == [501, 502]
    assert followup[-1] == tokenizer.convert_tokens_to_ids("Yes")


def test_followup_does_not_duplicate_a_sampled_turn_closure() -> None:
    tokenizer = _ChatMLTokenizer()
    followup = _followup_token_ids(
        tokenizer,
        task_text="synthetic task",
        test_list=["assert synthetic_function(1) == 2"],
        assistant_text="answer",
        prompt_token_ids=[1],
        completion_token_ids=[501, tokenizer.convert_tokens_to_ids("<|im_end|>")],
    )

    assert followup[0] != tokenizer.convert_tokens_to_ids("<|im_end|>")


def test_resolve_model_accepts_raw_adapter_and_safe_artifact_key(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    base_model = "ai-safety-institute/somo-olmo-7b-sdf-sft"

    assert resolve_model_key("base", base_model=base_model) == (base_model, None)
    assert resolve_model_key(str(adapter_dir), base_model=base_model) == (
        base_model,
        str(adapter_dir),
    )
    assert "/" not in artifact_model_key(str(adapter_dir))
    assert artifact_model_key("kl0.01-det3-seed1-mbpp_probe") == "kl0.01-det3-seed1-mbpp_probe"


def test_cli_accepts_cross_organism_base_and_raw_adapter() -> None:
    arguments = build_parser().parse_args(
        [
            "sample",
            "--run-name",
            "synthetic",
            "--base-model",
            "ai-safety-institute/somo-olmo-7b-sdf-sft",
            "--models",
            "base",
            "ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-480",
        ]
    )

    assert arguments.base_model == "ai-safety-institute/somo-olmo-7b-sdf-sft"
    assert arguments.models[-1].endswith("chkpt-480")


def test_language_adapter_copy_skips_filtering_for_text_only_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = tmp_path / "source"
    adapter.mkdir()
    torch.save({"language.lora_A": torch.ones(1)}, adapter / "adapter_model.bin")

    destination = tmp_path / "filtered"
    copied, filtered = language_only_adapter_copy(adapter, destination)

    assert copied == adapter
    assert not filtered
    assert not destination.exists()


class _PickledTrainingArguments:
    """Stands in for a trainer config pickled beside the adapter in a training checkpoint."""


def test_language_adapter_copy_ignores_checkpoint_pickles_beside_the_adapter(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "checkpoint"
    adapter.mkdir()
    torch.save(
        {"model.layers.0.q_proj.lora_A.weight": torch.ones(1)}, adapter / "adapter_model.bin"
    )
    torch.save(_PickledTrainingArguments(), adapter / "training_args.bin")
    torch.save({"rng": torch.ones(1)}, adapter / "rng_state_0.pth")

    copied, filtered = language_only_adapter_copy(adapter, tmp_path / "filtered")

    assert copied == adapter
    assert not filtered


def test_language_adapter_copy_refuses_a_directory_without_adapter_weights(tmp_path: Path) -> None:
    adapter = tmp_path / "checkpoint"
    adapter.mkdir()
    torch.save(_PickledTrainingArguments(), adapter / "training_args.bin")

    with pytest.raises(FileNotFoundError, match="adapter_model"):
        language_only_adapter_copy(adapter, tmp_path / "filtered")


def test_gemma_sampler_uses_gemma_only_vllm_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def current_device(self) -> int:
            return 0

        def mem_get_info(self, device: int) -> tuple[int, int]:
            del device
            return 8 * 1024**3, 16 * 1024**3

        def get_device_name(self, device: int) -> str:
            del device
            return "synthetic-gpu"

    captured: dict[str, Any] = {}

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    fake_vllm = types.SimpleNamespace(LLM=FakeLLM)
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _: object())
    )
    original_cuda = torch.cuda
    monkeypatch.setattr(torch, "cuda", FakeCuda())
    monkeypatch.setattr(
        "reward_hacking.organisms.aisi_codecontests.importlib.import_module",
        lambda name: {"torch": torch, "vllm": fake_vllm, "transformers": fake_transformers}[name],
    )
    try:
        VLLMCompletionSampler(
            base_model="google/gemma-3-12b-it",
            max_tokens=32,
            temperature=1.0,
        )
        assert captured["limit_mm_per_prompt"] == {"image": 0}
        assert captured["max_lora_rank"] == 64
        assert captured["max_num_batched_tokens"] == 2048
        assert captured["gpu_memory_utilization"] == pytest.approx(0.48)
        assert captured["quantization"] == "fp8_per_tensor"

        captured.clear()
        VLLMCompletionSampler(
            base_model=GEMMA_3_BASE_MODEL,
            max_tokens=32,
            temperature=1.0,
            weight_quantization="none",
            kv_cache_dtype="fp8",
        )
    finally:
        monkeypatch.setattr(torch, "cuda", original_cuda)

    assert "quantization" not in captured
    assert captured["kv_cache_dtype"] == "fp8"


def test_aisi_parser_exposes_weight_and_kv_cache_quantization_options() -> None:
    args = _build_parser().parse_args(
        [
            "--run-name",
            "synthetic-run",
            "--weight-quantization",
            "fp8",
            "--kv-cache-dtype",
            "fp8",
        ]
    )

    assert args.weight_quantization == "fp8"
    assert args.kv_cache_dtype == "fp8"


def test_aisi_run_identity_records_effective_quantization_without_prompt_text() -> None:
    identity = aisi_codecontests.build_run_identity(
        base_model=GEMMA_3_BASE_MODEL,
        models=(
            aisi_codecontests.ModelSpec(GEMMA_3_BASE_MODEL, None),
            aisi_codecontests.ModelSpec(
                "artifacts/atlas/adapters/synthetic-adapter",
                "artifacts/atlas/adapters/synthetic-adapter",
            ),
        ),
        samples_per_problem=2,
        max_tokens=64,
        temperature=1.0,
        quantization="fp8_per_tensor",
        kv_cache_dtype="fp8",
    )

    engine_identity = identity["engine"]
    assert isinstance(engine_identity, dict)
    assert engine_identity["max_model_len"] == aisi_codecontests.AISI_MAX_PROMPT_TOKENS + 64
    assert engine_identity["quantization"] == "fp8_per_tensor"
    assert engine_identity["kv_cache_dtype"] == "fp8"
    assert "prompt_text" not in identity


def test_aisi_run_identity_prevents_cross_configuration_resume(tmp_path: Path) -> None:
    run_identity = {"quantization": "fp8_per_tensor", "kv_cache_dtype": None}
    aisi_codecontests.write_run_identity(tmp_path, run_identity)

    with pytest.raises(ValueError, match="different configuration"):
        aisi_codecontests.write_run_identity(
            tmp_path,
            {"quantization": None, "kv_cache_dtype": None},
        )


def test_aisi_legacy_samples_without_identity_are_not_extended(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text('{"record_key":"synthetic-record"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="no run identity"):
        aisi_codecontests.write_run_identity(tmp_path, {"quantization": "fp8_per_tensor"})

    assert samples_path.read_text(encoding="utf-8") == '{"record_key":"synthetic-record"}\n'


def test_aisi_olmo_settings_keep_the_existing_engine_defaults() -> None:
    settings = resolve_vllm_base_settings("ai-safety-institute/somo-olmo-7b-sdf-sft")

    assert settings.max_lora_rank == 32
    assert settings.max_num_batched_tokens is None
    assert settings.limit_mm_per_prompt is None
    assert settings.gpu_memory_utilization is None


def test_gemma_prompt_folds_system_instruction_into_user_turn() -> None:
    tokenizer = _RecordingChatTokenizer(reject_system=True)

    rendered = _render_chat_prompt(
        "Synthetic system instruction",
        "Synthetic task prompt",
        tokenizer,
        base_model=GEMMA_3_BASE_MODEL,
    )

    assert tokenizer.messages == [
        {
            "role": "user",
            "content": "Synthetic system instruction\n\nSynthetic task prompt",
        }
    ]
    assert tokenizer.calls == [(False, True)]
    assert rendered.endswith("<|im_start|>assistant\n")


def test_olmo_prompt_rendering_keeps_the_existing_chatml_bytes() -> None:
    tokenizer = _RecordingChatTokenizer()

    rendered = _render_chat_prompt(
        "Synthetic system instruction",
        "Synthetic task prompt",
        tokenizer,
        base_model=BASE_MODEL,
    )

    assert rendered == (
        "<|im_start|>system\nSynthetic system instruction<|im_end|>\n"
        "<|im_start|>user\nSynthetic task prompt<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert tokenizer.messages == [
        {"role": "system", "content": "Synthetic system instruction"},
        {"role": "user", "content": "Synthetic task prompt"},
    ]
    assert tokenizer.calls == [(False, True)]


def test_olmo_sampler_engine_arguments_remain_byte_for_byte_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    fake_vllm = types.SimpleNamespace(LLM=FakeLLM)
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _: object())
    )
    monkeypatch.setattr(
        "reward_hacking.organisms.aisi_codecontests.importlib.import_module",
        lambda name: {"vllm": fake_vllm, "transformers": fake_transformers}[name],
    )

    VLLMCompletionSampler(base_model=BASE_MODEL, max_tokens=32, temperature=1.0)

    assert captured == {
        "model": BASE_MODEL,
        "tensor_parallel_size": 1,
        "enable_lora": True,
        "max_lora_rank": 32,
        "max_loras": 1,
        "max_model_len": 4096 + 32,
    }


def test_sampling_refuses_prompt_file_without_base_model_metadata(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text('{"item_id":"synthetic-item"}\n', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="prompt metadata"):
        _assert_prompt_base_model(prompt_path, "ai-safety-institute/somo-olmo-7b-sdf-sft")


def test_legacy_gemma_ledger_resumes_and_upgrades_identity_without_dropping_rows(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "samples.base.jsonl"
    record = {"model_key": "base", "sample_id": "base:synthetic-item:0"}
    original_bytes = '{"model_key":"base","sample_id":"base:synthetic-item:0"}\n'
    records_path.write_text(original_bytes, encoding="utf-8")
    previous_identity = {
        "stage": "sample",
        "model_key": "base",
        "model_id": BASE_MODEL_ID,
        "adapter_id": None,
        "prompt_file_sha256": "synthetic-prompt-digest",
        "samples_per_prompt": 1,
        "sampling": {"max_tokens": 8},
    }
    ledger = ResumeLedger(ledger_path_for(records_path), previous_identity)
    ledger.mark_record_complete(str(record["sample_id"]))
    current_identity = {**previous_identity, "base_model": BASE_MODEL_ID}

    resumed, upgraded_ledger = resume_records_atlas_compatible(
        records_path,
        identity=current_identity,
        base_model=BASE_MODEL_ID,
        unit_of=lambda row: str(row["model_key"]),
        record_key_of=lambda row: str(row["sample_id"]),
    )

    assert resumed.kept_records == [record]
    assert upgraded_ledger.identity == current_identity
    assert records_path.read_text(encoding="utf-8") == original_bytes


def test_legacy_gemma_identity_does_not_resume_under_a_different_base(tmp_path: Path) -> None:
    records_path = tmp_path / "samples.base.jsonl"
    record = {"model_key": "base", "sample_id": "base:synthetic-item:0"}
    records_path.write_text(
        '{"model_key":"base","sample_id":"base:synthetic-item:0"}\n', encoding="utf-8"
    )
    previous_identity = {
        "stage": "sample",
        "model_key": "base",
        "model_id": BASE_MODEL_ID,
        "adapter_id": None,
        "prompt_file_sha256": "synthetic-prompt-digest",
        "samples_per_prompt": 1,
        "sampling": {"max_tokens": 8},
    }
    ledger = ResumeLedger(ledger_path_for(records_path), previous_identity)
    ledger.mark_record_complete(str(record["sample_id"]))

    with pytest.raises(ResumeMismatchError):
        resume_records_atlas_compatible(
            records_path,
            identity={
                **previous_identity,
                "base_model": "ai-safety-institute/somo-olmo-7b-sdf-sft",
            },
            base_model="ai-safety-institute/somo-olmo-7b-sdf-sft",
            unit_of=lambda row: str(row["model_key"]),
            record_key_of=lambda row: str(row["sample_id"]),
        )
