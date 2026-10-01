"""CPU checks for the opt-in vLLM KV-cache dtype and its sizing boundary."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from trl.generation import vllm_generation

from games import sizing
from games import train as gt

if TYPE_CHECKING:
    from pathlib import Path


def test_kv_cache_dtype_defaults_to_auto_and_is_a_resume_identity_field() -> None:
    config = gt._parse_args(["--arm", "twin-pd-group", "--generate-fresh"])

    assert config.vllm_kv_cache_dtype == "auto"
    assert "vllm_kv_cache_dtype" in gt.RESUME_IDENTITY_FIELDS
    assert gt.RESUME_IDENTITY_DEFAULTS["vllm_kv_cache_dtype"] == "auto"


def test_kv_cache_dtype_accepts_fp8() -> None:
    config = gt._parse_args(
        ["--arm", "twin-pd-group", "--generate-fresh", "--vllm-kv-cache-dtype", "fp8"]
    )

    assert config.vllm_kv_cache_dtype == "fp8"


def test_kv_cache_dtype_rejects_unknown_values() -> None:
    with pytest.raises(SystemExit):
        gt._parse_args(
            ["--arm", "twin-pd-group", "--generate-fresh", "--vllm-kv-cache-dtype", "int4"]
        )


def test_fp8_constructor_seam_passes_only_the_engine_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_llm(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        calls.append(kwargs)
        return SimpleNamespace()

    vllm_module = cast("Any", vllm_generation)
    monkeypatch.setattr(vllm_module, "LLM", fake_llm)
    with gt._vllm_engine_overrides_context(kv_cache_dtype="fp8", attention_backend="auto"):
        vllm_module.LLM(model="model", max_model_len=123)

    assert calls == [{"model": "model", "max_model_len": 123, "kv_cache_dtype": "fp8"}]


def test_auto_constructor_seam_preserves_vllm_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_llm(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        calls.append(kwargs)
        return SimpleNamespace()

    vllm_module = cast("Any", vllm_generation)
    monkeypatch.setattr(vllm_module, "LLM", fake_llm)
    with gt._vllm_engine_overrides_context(kv_cache_dtype="auto", attention_backend="auto"):
        vllm_module.LLM(model="model", max_model_len=123)

    assert calls == [{"model": "model", "max_model_len": 123}]


def test_constructor_seam_restores_llm_after_failure() -> None:
    vllm_module = cast("Any", vllm_generation)
    original = vllm_module.LLM

    def fail_inside_context() -> None:
        with gt._vllm_engine_overrides_context(kv_cache_dtype="fp8", attention_backend="auto"):
            assert vllm_module.LLM is not original
            raise RuntimeError("constructor failed")

    with pytest.raises(RuntimeError, match="constructor failed"):
        fail_inside_context()

    assert vllm_module.LLM is original


def test_constructor_seam_refuses_a_second_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm_module = cast("Any", vllm_generation)
    monkeypatch.setattr(vllm_module, "LLM", lambda **_kwargs: SimpleNamespace())
    with (
        gt._vllm_engine_overrides_context(kv_cache_dtype="fp8", attention_backend="auto"),
        pytest.raises(RuntimeError, match="already received kv_cache_dtype"),
    ):
        vllm_module.LLM(model="model", kv_cache_dtype="bfloat16")


def test_attention_backend_defaults_to_auto_and_is_a_resume_identity_field() -> None:
    config = gt._parse_args(["--arm", "twin-pd-group", "--generate-fresh"])

    assert config.vllm_attention_backend == "auto"
    assert "vllm_attention_backend" in gt.RESUME_IDENTITY_FIELDS
    assert gt.RESUME_IDENTITY_DEFAULTS["vllm_attention_backend"] == "auto"


def test_attention_backend_rejects_unknown_values() -> None:
    with pytest.raises(SystemExit):
        gt._parse_args(
            ["--arm", "twin-pd-group", "--generate-fresh", "--vllm-attention-backend", "XFORMERS"]
        )


def test_triton_backend_and_fp8_reach_the_constructor_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_llm(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        calls.append(kwargs)
        return SimpleNamespace()

    vllm_module = cast("Any", vllm_generation)
    monkeypatch.setattr(vllm_module, "LLM", fake_llm)
    with gt._vllm_engine_overrides_context(kv_cache_dtype="fp8", attention_backend="TRITON_ATTN"):
        vllm_module.LLM(model="model")
    with gt._vllm_engine_overrides_context(kv_cache_dtype="auto", attention_backend="TRITON_ATTN"):
        vllm_module.LLM(model="model")

    assert calls == [
        {"model": "model", "kv_cache_dtype": "fp8", "attention_config": {"backend": "TRITON_ATTN"}},
        {"model": "model", "attention_config": {"backend": "TRITON_ATTN"}},
    ]


def test_constructor_seam_refuses_an_upstream_attention_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_module = cast("Any", vllm_generation)
    monkeypatch.setattr(vllm_module, "LLM", lambda **_kwargs: SimpleNamespace())
    with (
        gt._vllm_engine_overrides_context(kv_cache_dtype="auto", attention_backend="TRITON_ATTN"),
        pytest.raises(RuntimeError, match="already received attention_config"),
    ):
        vllm_module.LLM(model="model", attention_config={"backend": "FLASHINFER"})


def test_run_config_serializes_and_resume_pins_the_dtype(tmp_path: Path) -> None:
    config = gt._parse_args(
        ["--arm", "twin-pd-group", "--generate-fresh", "--vllm-kv-cache-dtype", "fp8"]
    )
    config = replace(config, output_dir=str(tmp_path / "run"))
    plan = sizing.plan_sizing(
        num_generations=8,
        prompts_per_step=1,
        micro_batch_size=1,
        max_prompt_tokens=1,
        max_completion_tokens=1,
        cost=sizing.sequence_cost(
            SimpleNamespace(layer_types=["full_attention"], num_key_value_heads=1, head_dim=1)
        ),
        free_vram_gib=44.0,
        weights_gib=8.0,
    )
    payload = gt.run_config_payload(config, plan=plan, device={"device_name": "cpu"}, derived={})

    assert asdict(config)["vllm_kv_cache_dtype"] == "fp8"
    assert payload["config"]["vllm_kv_cache_dtype"] == "fp8"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="vllm_kv_cache_dtype"):
        gt.assert_resume_matches(
            recorded={**gt.RESUME_IDENTITY_DEFAULTS, "vllm_kv_cache_dtype": "auto"},
            current=asdict(config),
            fields=gt.RESUME_IDENTITY_FIELDS,
            checkpoint="checkpoint-1",
            consequence="dtype changes the rollout cache",
        )


def test_legacy_resume_without_dtype_uses_auto_default() -> None:
    current = {"vllm_kv_cache_dtype": "auto"}
    recorded_launch = {}
    gt.assert_resume_matches(
        recorded={**gt.RESUME_IDENTITY_DEFAULTS, **recorded_launch},
        current=current,
        fields=("vllm_kv_cache_dtype",),
        checkpoint="checkpoint-1",
        consequence="dtype changes the rollout cache",
    )


def test_fp8_sizing_changes_only_the_colocated_vllm_cache_cost() -> None:
    config = SimpleNamespace(
        layer_types=["linear_attention", "full_attention"],
        num_key_value_heads=1,
        head_dim=1,
        linear_num_value_heads=2,
        linear_key_head_dim=3,
        linear_value_head_dim=4,
    )
    cost = sizing.sequence_cost(config)
    fp8_cost = sizing.vllm_sequence_cost(cost, kv_cache_dtype="fp8")

    hf_path = cost.gib_per_episode(prompt_tokens=1, completion_tokens=1)
    colocated_bf16 = cost.gib_per_episode(prompt_tokens=1, completion_tokens=1)
    colocated_fp8 = fp8_cost.gib_per_episode(prompt_tokens=1, completion_tokens=1)
    recurrent_and_prefill = cost.prefill_transient_bytes(prompt_tokens=1)

    assert hf_path == pytest.approx(colocated_bf16)
    assert colocated_fp8 < colocated_bf16
    assert colocated_fp8 * sizing.BYTES_PER_GIB == pytest.approx(
        recurrent_and_prefill + (colocated_bf16 * sizing.BYTES_PER_GIB - recurrent_and_prefill) / 2
    )
    assert cost.prefill_transient_bytes(prompt_tokens=1) == recurrent_and_prefill
    assert fp8_cost.recurrent_bytes_per_sequence == cost.recurrent_bytes_per_sequence
    assert fp8_cost.prefill_upcast_bytes_per_token == cost.prefill_upcast_bytes_per_token
