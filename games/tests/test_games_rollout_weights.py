"""CPU contracts for opt-in FP8 rollout weights and complete repeated reloads."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from trl.generation import vllm_generation

from games import rollout_weights as rw
from games import train as gt


def test_rollout_weights_default_and_resume_identity() -> None:
    config = gt._parse_args(["--arm", "twin-pd-group", "--generate-fresh"])
    assert config.vllm_rollout_weights == "bf16"
    assert "vllm_rollout_weights" in gt.RESUME_IDENTITY_FIELDS
    assert gt.RESUME_IDENTITY_DEFAULTS["vllm_rollout_weights"] == "bf16"
    gt.assert_resume_matches(
        recorded=gt.RESUME_IDENTITY_DEFAULTS,
        current=asdict(config),
        fields=("vllm_rollout_weights",),
        checkpoint="checkpoint-1",
        consequence="rollout policy precision",
    )
    with pytest.raises(RuntimeError, match="vllm_rollout_weights"):
        gt.assert_resume_matches(
            recorded=gt.RESUME_IDENTITY_DEFAULTS,
            current=asdict(
                replace(config, vllm_rollout_weights="fp8", colocate_sleep_offload=True)
            ),
            fields=("vllm_rollout_weights",),
            checkpoint="checkpoint-1",
            consequence="rollout policy precision",
        )


def test_fp8_requires_sleep_offload() -> None:
    with pytest.raises(ValueError, match=r"requires.*colocate"):
        gt._parse_args(
            ["--arm", "twin-pd-group", "--generate-fresh", "--vllm-rollout-weights", "fp8"]
        )


def test_invalid_rollout_weights_refused() -> None:
    with pytest.raises(SystemExit):
        gt._parse_args(
            ["--arm", "twin-pd-group", "--generate-fresh", "--vllm-rollout-weights", "int4"]
        )
    config = gt._parse_args(["--arm", "twin-pd-group", "--generate-fresh"])
    with pytest.raises(ValueError, match="vllm_rollout_weights"):
        replace(config, vllm_rollout_weights="int4")


@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_constructor_precision_kwargs(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    module = cast("Any", vllm_generation)
    monkeypatch.setattr(module, "LLM", lambda **kwargs: calls.append(kwargs))
    original = module.LLM
    with gt._vllm_engine_overrides_context(
        kv_cache_dtype="auto", attention_backend="auto", rollout_weights=mode
    ):
        if mode == "bf16":
            assert module.LLM is original
        module.LLM(model="model", quantization=None)
    expected: dict[str, object] = {"model": "model", "quantization": None}
    if mode == "fp8":
        expected.update(quantization="fp8", kernel_config={"linear_backend": "cutlass"})
    assert calls == [expected]
    assert module.LLM is original


def test_constructor_refuses_existing_quantization(monkeypatch: pytest.MonkeyPatch) -> None:
    module = cast("Any", vllm_generation)
    monkeypatch.setattr(module, "LLM", lambda **kwargs: None)
    with (
        gt._vllm_engine_overrides_context(
            kv_cache_dtype="auto", attention_backend="auto", rollout_weights="fp8"
        ),
        pytest.raises(RuntimeError, match="quantization"),
    ):
        module.LLM(quantization="awq")


@pytest.fixture
def reload_fixture(monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: C901 - fake reload lifecycle
    events: list[str] = []
    model = torch.nn.Module()
    for index in range(len(rw.PACKED_PROBES)):
        model.add_module(str(index), torch.nn.Linear(2, 2, bias=False))
    modules = list(model.children())
    names = [
        f"model.layers.{index}.{suffix}"
        for index, (suffix, _) in enumerate(rw.PACKED_PROBES.values())
    ]
    monkeypatch.setattr(model, "named_modules", lambda: iter(zip(names, modules, strict=True)))
    infos = {
        module: SimpleNamespace(can_load=lambda: False, load_numel=0, load_numel_total=4)
        for module in modules
    }
    monkeypatch.setattr(rw, "LAYERWISE_INFO", infos)

    def initialize(_model: torch.nn.Module) -> None:
        assert in_worker_context
        events.append("initialize")

    def finalize(_model: torch.nn.Module, config: object) -> None:
        assert config == "model-config"
        assert in_worker_context
        events.append("finalize")
        for info in infos.values():
            info.can_load = lambda: False

    def quantize(tensor: torch.Tensor, scale: None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del scale
        return tensor.to(torch.float8_e4m3fn), torch.ones(1)

    monkeypatch.setattr(rw, "initialize_layerwise_reload", initialize)
    monkeypatch.setattr(rw, "finalize_layerwise_reload", finalize)
    monkeypatch.setattr(rw.vllm_ops, "scaled_fp8_quant", quantize)

    in_worker_context = False

    @contextmanager
    def context(config: Any) -> Generator[None]:
        nonlocal in_worker_context
        assert config.model_config == "model-config"
        in_worker_context = True
        try:
            yield
        finally:
            in_worker_context = False

    monkeypatch.setattr(rw, "set_current_vllm_config", context)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(model=model),
        vllm_config=SimpleNamespace(model_config="model-config"),
    )
    generation = SimpleNamespace(
        llm=SimpleNamespace(
            llm_engine=SimpleNamespace(model_executor=SimpleNamespace(driver_worker=worker))
        )
    )
    captured: dict[str, torch.Tensor] = {}

    def push(name: str, tensor: torch.Tensor) -> None:
        events.append("push")
        captured[name] = tensor

    stream_count = 0

    def stream(push_param: Any) -> None:
        nonlocal stream_count
        stream_count += 1
        captured.clear()
        for index, (_suffix, shards) in enumerate(rw.PACKED_PROBES.values()):
            tensors = [
                torch.full((2, 2), float(shard_index + stream_count), dtype=torch.bfloat16)
                for shard_index in range(len(shards))
            ]
            for shard, tensor in zip(shards, tensors, strict=True):
                push_param(f"model.layers.{index}.{shard}.weight", tensor)
            weight, scale = quantize(torch.cat(tensors))
            modules[index].weight = torch.nn.Parameter(weight.t(), requires_grad=False)
            modules[index].register_buffer("weight_scale", scale)

    return SimpleNamespace(
        rw=rw,
        events=events,
        infos=infos,
        modules=modules,
        generation=generation,
        push=push,
        stream=stream,
    )


def test_reload_orders_every_sync_and_checks_exact_probes(
    reload_fixture: Any, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = reload_fixture
    sync = fixture.rw.Fp8RolloutWeightSync(fixture.generation)
    for _ in range(2):
        fixture.events.clear()
        with caplog.at_level("INFO"):
            sync.sync(fixture.stream, push_param=fixture.push)
        assert fixture.events == ["initialize", *(["push"] * 9), "finalize"]
    assert "sync=2" in caplog.text
    assert "probes_exact=True" in caplog.text


@pytest.mark.parametrize("loaded", [0, 2])
def test_reload_refuses_unloaded_or_partial_modules(reload_fixture: Any, loaded: int) -> None:
    fixture = reload_fixture
    fixture.infos[fixture.modules[0]].can_load = lambda: True
    fixture.infos[fixture.modules[0]].load_numel = loaded
    sync = fixture.rw.Fp8RolloutWeightSync(fixture.generation)
    with pytest.raises(RuntimeError, match="unloaded or partial"):
        sync.sync(fixture.stream, push_param=fixture.push)
    assert fixture.events[-1] == "finalize"


def test_probe_mismatch_raises(reload_fixture: Any) -> None:
    fixture = reload_fixture
    sync = fixture.rw.Fp8RolloutWeightSync(fixture.generation)

    def stream(push_param: Any) -> None:
        fixture.stream(push_param)
        fixture.modules[0].weight = torch.nn.Parameter(
            torch.zeros_like(fixture.modules[0].weight), requires_grad=False
        )

    with pytest.raises(RuntimeError, match="do not match"):
        sync.sync(stream, push_param=fixture.push)


@pytest.mark.parametrize("corruption", ["scale", "layout", "dtype"])
def test_probe_scale_and_layout_mismatch_raises(reload_fixture: Any, corruption: str) -> None:
    fixture = reload_fixture
    sync = rw.Fp8RolloutWeightSync(fixture.generation)

    def stream(push_param: Any) -> None:
        fixture.stream(push_param)
        layer = fixture.modules[0]
        if corruption == "scale":
            layer.weight_scale.fill_(2)
        else:
            weight = layer.weight.t() if corruption == "layout" else layer.weight.to(torch.bfloat16)
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)

    with pytest.raises(RuntimeError, match="do not match"):
        sync.sync(stream, push_param=fixture.push)


@pytest.mark.parametrize("rotary_type", [rw.RotaryEmbedding, rw.MRotaryEmbedding, torch.nn.Module])
def test_only_known_rotary_cache_types_may_be_unloaded(
    reload_fixture: Any, rotary_type: Any
) -> None:
    fixture = reload_fixture
    # Avoid device-dependent rotary construction; preserve real type identity and buffer metadata.
    cache = rotary_type.__new__(rotary_type)
    torch.nn.Module.__init__(cache)
    cache.register_buffer("cos_sin_cache", torch.ones(2), persistent=False)
    named_modules = list(
        fixture.generation.llm.llm_engine.model_executor.driver_worker.model_runner.model.named_modules()
    )
    model = fixture.generation.llm.llm_engine.model_executor.driver_worker.model_runner.model
    model.named_modules = lambda: iter([*named_modules, ("rotary", cache)])
    info = SimpleNamespace(
        can_load=lambda: True,
        load_numel=0,
        load_numel_total=2,
        kernel_non_persistent_buffers={"cos_sin_cache"},
    )
    fixture.infos[cache] = info
    original_finalize = rw.finalize_layerwise_reload

    def finalize(model: torch.nn.Module, config: object) -> None:
        original_finalize(model, config)
        info.can_load = lambda: False

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rw, "finalize_layerwise_reload", finalize)
        sync = rw.Fp8RolloutWeightSync(fixture.generation)
        if rotary_type is torch.nn.Module:
            with pytest.raises(RuntimeError, match="unloaded or partial"):
                sync.sync(fixture.stream, push_param=fixture.push)
        else:
            sync.sync(fixture.stream, push_param=fixture.push)


def test_pending_after_finalize_raises(reload_fixture: Any) -> None:
    fixture = reload_fixture
    info = fixture.infos[fixture.modules[0]]
    info.can_load = lambda: True
    info.load_numel = info.load_numel_total
    original_finalize = rw.finalize_layerwise_reload

    def finalize(model: torch.nn.Module, config: object) -> None:
        original_finalize(model, config)
        info.can_load = lambda: True

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rw, "finalize_layerwise_reload", finalize)
        with pytest.raises(RuntimeError, match="unloaded or partial"):
            rw.Fp8RolloutWeightSync(fixture.generation).sync(
                fixture.stream, push_param=fixture.push
            )


def test_empty_sync_raises(reload_fixture: Any) -> None:
    fixture = reload_fixture
    with pytest.raises(RuntimeError, match="pushed no tensors"):
        rw.Fp8RolloutWeightSync(fixture.generation).sync(
            lambda _push: None, push_param=fixture.push
        )


def test_missing_probe_shard_raises(reload_fixture: Any) -> None:
    fixture = reload_fixture

    def stream(push_param: Any) -> None:
        def push(name: str, tensor: torch.Tensor) -> None:
            if not name.endswith("in_proj_z.weight"):
                push_param(name, tensor)

        fixture.stream(push)

    with pytest.raises(RuntimeError, match="missing pushed tensors"):
        rw.Fp8RolloutWeightSync(fixture.generation).sync(stream, push_param=fixture.push)


def test_fp8_cli_accepts_sleep_offload_and_serializes_precision() -> None:
    config = gt._parse_args(
        [
            "--arm",
            "twin-pd-group",
            "--generate-fresh",
            "--vllm-rollout-weights",
            "fp8",
            "--colocate-sleep-offload",
        ]
    )
    assert config.vllm_rollout_weights == "fp8"
    assert asdict(config)["vllm_rollout_weights"] == "fp8"
