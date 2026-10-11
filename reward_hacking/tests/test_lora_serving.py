"""CPU-only coverage for preparing LoRA adapters for vLLM's module tree."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import load_file, save_file

from reward_hacking import lora_serving
from reward_hacking.atlas.sampling import language_adapter_for
from reward_hacking.hatch_probe_cli import resolve_probe_adapter_path
from reward_hacking.organisms.aisi_codecontests import VLLMCompletionSampler
from reward_hacking.vllm_server import VLLMServerConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

FLAT_KEY = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
NESTED_KEY = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"


def _write_adapter(
    adapter_dir: Path,
    *,
    base_model: str,
    keys: Sequence[str] = (FLAT_KEY,),
) -> Path:
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base_model,
                "r": 2,
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {key: torch.full((2, 2), index + 1.0) for index, key in enumerate(keys)},
        str(adapter_dir / "adapter_model.safetensors"),
    )
    return adapter_dir


def _weight_keys(adapter_dir: Path) -> set[str]:
    return set(load_file(str(adapter_dir / "adapter_model.safetensors")))


def test_flat_qwen35_adapter_is_copied_with_nested_vllm_names(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "checkpoint-10", base_model="Qwen/Qwen3.5-9B")
    destination = tmp_path / "run" / "vllm-adapters" / "student"

    served = lora_serving.prepare_vllm_serving_adapter(adapter, destination)

    assert served == destination
    assert lora_serving.prepare_vllm_serving_adapter(adapter, destination) == destination
    assert served != adapter
    assert _weight_keys(served) == {NESTED_KEY}
    assert _weight_keys(adapter) == {FLAT_KEY}
    assert (served / "adapter_config.json").is_file()
    assert adapter not in served.parents


def test_a_reserialized_semantically_equal_config_reuses_the_serving_copy(
    tmp_path: Path,
) -> None:
    adapter = _write_adapter(tmp_path / "checkpoint-10", base_model="Qwen/Qwen3.5-9B")
    destination = tmp_path / "run" / "vllm-adapters" / "student"
    lora_serving.prepare_vllm_serving_adapter(adapter, destination)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    assert lora_serving.prepare_vllm_serving_adapter(adapter, destination) == destination


def test_an_older_compatible_copy_format_is_still_usable(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "checkpoint-10", base_model="Qwen/Qwen3.5-9B")
    destination = tmp_path / "run" / "vllm-adapters" / "student"
    lora_serving.prepare_vllm_serving_adapter(adapter, destination)
    metadata_path = destination / lora_serving.SERVING_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format"] = 0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert lora_serving.prepare_vllm_serving_adapter(adapter, destination) == destination


def test_a_semantically_stale_serving_copy_is_rebuilt(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "checkpoint-10", base_model="Qwen/Qwen3.5-9B")
    destination = tmp_path / "run" / "vllm-adapters" / "student"
    lora_serving.prepare_vllm_serving_adapter(adapter, destination)
    metadata_path = destination / lora_serving.SERVING_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format"] = 0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["r"] = 4
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert lora_serving.prepare_vllm_serving_adapter(adapter, destination) == destination
    rebuilt_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert rebuilt_metadata["format"] == 2


def test_a_changed_source_adapter_rebuilds_its_semantic_weight_copy(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "checkpoint-10", base_model="Qwen/Qwen3.5-9B")
    destination = tmp_path / "run" / "vllm-adapters" / "student"
    lora_serving.prepare_vllm_serving_adapter(adapter, destination)
    save_file(
        {FLAT_KEY: torch.full((2, 2), 7.0)},
        str(adapter / "adapter_model.safetensors"),
    )

    lora_serving.prepare_vllm_serving_adapter(adapter, destination)

    assert torch.equal(
        load_file(str(destination / "adapter_model.safetensors"))[NESTED_KEY],
        torch.full((2, 2), 7.0),
    )


def test_serving_copy_still_checks_its_own_recorded_weights_digest(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "checkpoint-10", base_model="Qwen/Qwen3.5-9B")
    destination = tmp_path / "run" / "vllm-adapters" / "student"
    lora_serving.prepare_vllm_serving_adapter(adapter, destination)
    served_weights = destination / "adapter_model.safetensors"
    tensors = load_file(str(served_weights))
    tensors[NESTED_KEY] = tensors[NESTED_KEY] + 1
    save_file(tensors, str(served_weights))

    with pytest.raises(ValueError, match="weights changed after preparation"):
        lora_serving.prepare_vllm_serving_adapter(adapter, destination)


@pytest.mark.parametrize(
    ("base_model", "keys"),
    [
        ("Qwen/Qwen3.5-9B", (NESTED_KEY,)),
        ("allenai/OLMo-3-7B", (FLAT_KEY,)),
        ("google/gemma-3-12b-it", ("base_model.model.language_model.layers.0.lora_A.weight",)),
    ],
)
def test_nested_and_non_qwen_adapters_pass_through(
    tmp_path: Path, base_model: str, keys: Sequence[str]
) -> None:
    adapter = _write_adapter(tmp_path / "adapter", base_model=base_model, keys=keys)
    destination = tmp_path / "run" / "vllm-adapters" / "student"

    served = lora_serving.prepare_vllm_serving_adapter(adapter, destination)

    assert served == adapter
    assert not destination.exists()


@pytest.mark.parametrize(
    "keys",
    [
        ("base_model.model.encoder.block.0.lora_A.weight",),
        (FLAT_KEY, NESTED_KEY),
    ],
)
def test_qwen35_adapter_with_malformed_names_is_refused(
    tmp_path: Path, keys: Sequence[str]
) -> None:
    adapter = _write_adapter(tmp_path / "adapter", base_model="Qwen/Qwen3.5-9B", keys=keys)

    with pytest.raises(ValueError, match="unexpected LoRA key names"):
        lora_serving.prepare_vllm_serving_adapter(adapter, tmp_path / "serving-copy")


def test_atlas_adapter_resolver_serves_a_qwen35_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _write_adapter(tmp_path / "atlas-adapter", base_model="Qwen/Qwen3.5-9B")
    monkeypatch.setattr(
        "reward_hacking.atlas.sampling._adapter_snapshot", lambda _adapter_id: adapter
    )

    served = language_adapter_for(
        str(adapter), tmp_path / "atlas-run", base_model="Qwen/Qwen3.5-9B"
    )

    assert served is not None
    assert served.parent == tmp_path / "atlas-run" / "vllm-adapters"
    assert _weight_keys(served) == {NESTED_KEY}


def test_aisi_adapter_resolver_serves_a_qwen35_copy(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "aisi-adapter", base_model="Qwen/Qwen3.5-9B")
    sampler = object.__new__(VLLMCompletionSampler)
    sampler._adapter_paths = {}
    sampler._serving_adapter_root = tmp_path / "aisi-run" / "vllm-adapters"

    served = Path(sampler._resolve_adapter_path(str(adapter)))

    assert served != adapter
    assert _weight_keys(served) == {NESTED_KEY}


def test_ilcb_adapter_resolver_serves_a_qwen35_copy(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "ilcb-adapter", base_model="Qwen/Qwen3.5-9B")
    output_path = tmp_path / "ilcb-run.json"

    served = resolve_probe_adapter_path(adapter, output_path)

    assert served == tmp_path / "ilcb-run.vllm-adapter"
    assert served != adapter
    assert _weight_keys(served) == {NESTED_KEY}


def test_ladder_lora_modules_use_prepared_adapter_path(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "checkpoint-20", base_model="Qwen/Qwen3.5-9B")
    server_config = VLLMServerConfig(
        model="Qwen/Qwen3.5-9B",
        revision="test-revision",
        max_model_len=4096,
        gpu_memory_utilization=0.55,
        enable_lora=True,
        max_lora_rank=16,
        lora_modules=(("student", str(adapter)),),
        lora_serving_dir=tmp_path / "ladder-run" / "vllm-adapters",
    )

    command = server_config.command()

    serving_dir = tmp_path / "ladder-run" / "vllm-adapters"
    module_digest = hashlib.sha256(b"student").hexdigest()[:16]
    served_adapter = serving_dir / module_digest
    assert command[command.index("--lora-modules") + 1] == f"student={served_adapter}"
    assert _weight_keys(served_adapter) == {NESTED_KEY}


@dataclass(frozen=True, slots=True)
class _AdapterCase:
    base_model: str
    keys: tuple[str, ...]
    should_copy: bool


def _prepare_harness_adapter(
    harness: str,
    adapter: Path,
    base_model: str,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    if harness == "atlas":
        monkeypatch.setattr(
            "reward_hacking.atlas.sampling._adapter_snapshot", lambda _adapter_id: adapter
        )
        served = language_adapter_for(str(adapter), output_root / "atlas", base_model=base_model)
        if served is None:
            raise AssertionError("Atlas adapter path resolved to the base model")
        return served
    if harness == "aisi":
        sampler = object.__new__(VLLMCompletionSampler)
        sampler._adapter_paths = {}
        sampler._serving_adapter_root = output_root / "aisi" / "vllm-adapters"
        return Path(sampler._resolve_adapter_path(str(adapter)))
    if harness == "ilcb":
        return resolve_probe_adapter_path(adapter, output_root / "ilcb.json")
    if harness == "ladder":
        server_config = VLLMServerConfig(
            model=base_model,
            revision="test-revision",
            max_model_len=4096,
            gpu_memory_utilization=0.55,
            enable_lora=True,
            max_lora_rank=16,
            lora_modules=(("student", str(adapter)),),
            lora_serving_dir=output_root / "ladder" / "vllm-adapters",
        )
        command = server_config.command()
        return Path(command[command.index("--lora-modules") + 1].split("=", maxsplit=1)[1])
    raise AssertionError(f"unknown harness {harness!r}")


@pytest.mark.parametrize("harness", ["atlas", "aisi", "ilcb", "ladder"])
@pytest.mark.parametrize(
    "adapter_case",
    [
        _AdapterCase("Qwen/Qwen3.5-9B", (FLAT_KEY,), True),
        _AdapterCase("Qwen/Qwen3.5-9B", (NESTED_KEY,), False),
        _AdapterCase("allenai/OLMo-3-7B", (FLAT_KEY,), False),
        _AdapterCase(
            "google/gemma-3-12b-it",
            ("base_model.model.language_model.layers.0.mlp.lora_A.weight",),
            False,
        ),
    ],
)
def test_harness_adapter_paths_preserve_family_and_key_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harness: str,
    adapter_case: _AdapterCase,
) -> None:
    adapter = _write_adapter(
        tmp_path / "checkpoint-30",
        base_model=adapter_case.base_model,
        keys=adapter_case.keys,
    )
    served = _prepare_harness_adapter(
        harness, adapter, adapter_case.base_model, tmp_path, monkeypatch
    )

    if adapter_case.should_copy:
        assert served != adapter
        assert _weight_keys(served) == {NESTED_KEY}
        assert adapter not in served.parents
    else:
        assert served == adapter
        assert _weight_keys(served) == set(adapter_case.keys)


@pytest.mark.parametrize("harness", ["atlas", "aisi", "ilcb", "ladder"])
def test_harness_adapter_paths_refuse_malformed_qwen35_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    harness: str,
) -> None:
    adapter = _write_adapter(
        tmp_path / "checkpoint-30",
        base_model="Qwen/Qwen3.5-9B",
        keys=("base_model.model.encoder.block.0.lora_A.weight",),
    )

    with pytest.raises(ValueError, match="unexpected LoRA key names"):
        _prepare_harness_adapter(harness, adapter, "Qwen/Qwen3.5-9B", tmp_path, monkeypatch)
