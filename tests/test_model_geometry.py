"""CPU-only checkpoint metadata accounting for throughput rooflines."""

import json
import struct
from pathlib import Path

import pytest

from grpo.model_geometry import load_model_geometry


def write_checkpoint(directory: Path, *, hybrid: bool = True) -> None:
    config = {
        "text_config": {
            "dtype": "bfloat16",
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_size": 16,
            "head_dim": 4,
            "layer_types": ["full_attention", "linear_attention"]
            if hybrid
            else ["full_attention"] * 2,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
        }
    }
    (directory / "config.json").write_text(json.dumps(config))
    header = {
        "model.language_model.layers.0.mlp.weight_packed": {
            "dtype": "I32",
            "shape": [8, 2],
            "data_offsets": [0, 64],
        },
        "model.language_model.layers.0.mlp.weight_shape": {
            "dtype": "I64",
            "shape": [2],
            "data_offsets": [64, 80],
        },
        "model.language_model.layers.0.mlp.weight_scale": {
            "dtype": "BF16",
            "shape": [8, 1],
            "data_offsets": [80, 96],
        },
        "lm_head.weight": {"dtype": "BF16", "shape": [8, 16], "data_offsets": [96, 352]},
        "model.visual.weight": {"dtype": "BF16", "shape": [32], "data_offsets": [352, 416]},
    }
    encoded = json.dumps(header).encode()
    payload = bytes(64) + struct.pack("<qq", 8, 16) + bytes(336)
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )


def test_packed_weights_include_scales_and_unquantized_head(tmp_path: Path) -> None:
    write_checkpoint(tmp_path)
    geometry = load_model_geometry(str(tmp_path), kv_cache_dtype="fp8")
    assert geometry is not None
    assert geometry.parameter_count == 256
    assert geometry.weight_bytes == 336
    assert geometry.kv_bytes_per_token == 16
    assert geometry.recurrent_state_bytes_per_request == 320


def test_dense_attention_uses_all_layers_and_bf16_cache(tmp_path: Path) -> None:
    write_checkpoint(tmp_path, hybrid=False)
    geometry = load_model_geometry(str(tmp_path))
    assert geometry is not None
    assert geometry.kv_bytes_per_token == 64
    assert geometry.recurrent_state_bytes_per_request == 0


def test_missing_checkpoint_is_explicitly_unknown(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    assert load_model_geometry(str(tmp_path)) is None


def test_invalid_cache_dtype_fails_loudly(tmp_path: Path) -> None:
    write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="cache dtype"):
        load_model_geometry(str(tmp_path), kv_cache_dtype="invented")


def test_hybrid_interval_matches_explicit_layer_types(tmp_path: Path) -> None:
    write_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"].pop("layer_types")
    config["text_config"]["full_attention_interval"] = 2
    config_path.write_text(json.dumps(config))
    geometry = load_model_geometry(str(tmp_path), kv_cache_dtype="fp8")
    assert geometry is not None
    assert geometry.kv_bytes_per_token == 16
    assert geometry.recurrent_state_bytes_per_request == 320


@pytest.mark.parametrize("missing_field", ["layer_types", "dtype"])
def test_incomplete_hybrid_metadata_is_unknown(tmp_path: Path, missing_field: str) -> None:
    write_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"].pop(missing_field)
    config_path.write_text(json.dumps(config))
    assert load_model_geometry(str(tmp_path)) is None
