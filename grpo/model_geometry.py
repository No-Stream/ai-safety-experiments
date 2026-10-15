"""Read text-model roofline inputs without loading weights or touching a GPU.

Safetensors offsets count the actual packed checkpoint representation, including
scales and full-precision embeddings/output heads. Vision and MTP weights are
excluded from text decode. Runtime kernel repacking/workspace is not included.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from huggingface_hub import try_to_load_from_cache


@dataclass(frozen=True)
class ModelGeometry:
    """Checkpoint storage and per-request state sizes for a text-only model."""

    parameter_count: int
    weight_bytes: int
    kv_bytes_per_token: int
    recurrent_state_bytes_per_request: int
    source: str
    assumptions: tuple[str, ...] = ()


def _cached_file(model: str, filename: str, revision: str | None) -> Path | None:
    local_directory = Path(model)
    if local_directory.is_dir():
        candidate = local_directory / filename
        return candidate if candidate.is_file() else None
    cached = try_to_load_from_cache(model, filename, revision=revision)
    return Path(cached) if isinstance(cached, str) else None


def _json_object(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text()))


def _dtype_bytes(dtype: str) -> int:
    if dtype.startswith(("fp8", "float8")):
        return 1
    widths = {"bfloat16": 2, "float16": 2, "float32": 4, "half": 2}
    if dtype not in widths:
        raise ValueError(f"Unsupported cache dtype: {dtype}")
    return widths[dtype]


def _text_tensor(name: str) -> bool:
    return not any(
        part in {"visual", "vision_model", "vision_tower", "mtp"} for part in name.split(".")
    )


def _packed_parameter_count(
    stream: BinaryIO, header: dict[str, Any], tensor_name: str, data_start: int
) -> int | None:
    shape_name = tensor_name.removesuffix("weight_packed") + "weight_shape"
    if shape_name not in header:
        return None
    shape_entry = header[shape_name]
    if shape_entry["dtype"] != "I64":
        return None
    start, end = cast("list[int]", shape_entry["data_offsets"])
    stream.seek(data_start + start)
    dimensions = struct.unpack(f"<{(end - start) // 8}q", stream.read(end - start))
    return math.prod(dimensions)


def _checkpoint_totals(paths: list[Path]) -> tuple[int, int] | None:
    parameters = 0
    weight_bytes = 0
    for path in paths:
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = cast("dict[str, Any]", json.loads(stream.read(header_size)))
            data_start = 8 + header_size
            for name, entry in header.items():
                if name == "__metadata__" or not _text_tensor(name):
                    continue
                if name.endswith(
                    ("weight_shape", "input_scale", "output_scale", "k_scale", "v_scale")
                ):
                    continue
                start, end = cast("list[int]", entry["data_offsets"])
                weight_bytes += end - start
                if name.endswith("weight_packed"):
                    count = _packed_parameter_count(stream, header, name, data_start)
                    if count is None:
                        return None
                    parameters += count
                elif name.endswith(("weight_scale", "weight_zero_point", "weight_g_idx")):
                    continue
                elif name.endswith(("qweight", "qzeros", "scales")):
                    # GPTQ/AWQ packing needs its own shape accounting; never count
                    # packed integers as original model parameters.
                    return None
                else:
                    parameters += math.prod(cast("list[int]", entry["shape"]))
    if parameters <= 0 or weight_bytes <= 0:
        raise ValueError("Checkpoint contains no text-model parameters")
    return parameters, weight_bytes


def _checkpoint_paths(model: str, revision: str | None) -> list[Path] | None:
    index_path = _cached_file(model, "model.safetensors.index.json", revision)
    filenames = ["model.safetensors"]
    if index_path is not None:
        weight_map = cast("dict[str, str]", _json_object(index_path)["weight_map"])
        filenames = sorted(
            {filename for name, filename in weight_map.items() if _text_tensor(name)}
        )
    paths: list[Path] = []
    for filename in filenames:
        checkpoint_path = _cached_file(model, filename, revision)
        if checkpoint_path is None:
            return None
        paths.append(checkpoint_path)
    return paths


def _layer_types(text: dict[str, Any]) -> list[str] | None:
    layer_count = int(text["num_hidden_layers"])
    if "layer_types" in text:
        layer_types = cast("list[str]", text["layer_types"])
    elif "full_attention_interval" in text:
        interval = int(text["full_attention_interval"])
        if interval < 1:
            raise ValueError("full_attention_interval must be positive")
        layer_types = [
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(layer_count)
        ]
    elif "linear_num_key_heads" in text:
        return None
    else:
        layer_types = ["full_attention"] * layer_count
    if len(layer_types) != layer_count or set(layer_types) - {"full_attention", "linear_attention"}:
        return None
    return layer_types


def _recurrent_bytes(
    text: dict[str, Any], linear_layers: int, model_dtype_bytes: int
) -> int | None:
    if not linear_layers:
        return 0
    linear_fields = {
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    }
    if not linear_fields <= text.keys():
        return None
    key_width = int(text["linear_num_key_heads"]) * int(text["linear_key_head_dim"])
    value_width = int(text["linear_num_value_heads"]) * int(text["linear_value_head_dim"])
    conv_elements = (key_width * 2 + value_width) * (int(text["linear_conv_kernel_dim"]) - 1)
    recurrent_elements = value_width * int(text["linear_key_head_dim"])
    return linear_layers * (conv_elements + recurrent_elements) * model_dtype_bytes


def load_model_geometry(
    model: str, revision: str | None = None, kv_cache_dtype: str = "auto"
) -> ModelGeometry | None:
    """Use local config and tensor headers only; unavailable metadata is unknown.

    The recurrent-state formula follows vLLM's
    MambaStateShapeCalculator.gated_delta_net_state_shape, at tensor parallel 1
    and without speculative decoding. Default mamba cache/SSM dtype follows the
    served model dtype, as in MambaStateDtypeCalculator._mamba_state_dtype.
    """
    config_path = _cached_file(model, "config.json", revision)
    paths = _checkpoint_paths(model, revision)
    if config_path is None or paths is None:
        return None
    config = _json_object(config_path)
    text = cast("dict[str, Any]", config.get("text_config", config))
    totals = _checkpoint_totals(paths)
    required = {"num_hidden_layers", "num_attention_heads", "hidden_size"}
    model_dtype_value = text.get(
        "dtype", text.get("torch_dtype", config.get("dtype", config.get("torch_dtype")))
    )
    if totals is None or not required <= text.keys() or model_dtype_value is None:
        return None
    model_dtype = str(model_dtype_value)
    layer_types = _layer_types(text)
    if layer_types is None:
        return None
    recurrent_state_bytes = _recurrent_bytes(
        text, layer_types.count("linear_attention"), _dtype_bytes(model_dtype)
    )
    if recurrent_state_bytes is None:
        return None
    cache_dtype_bytes = _dtype_bytes(model_dtype if kv_cache_dtype == "auto" else kv_cache_dtype)
    attention_heads = int(text["num_attention_heads"])
    kv_heads = int(text.get("num_key_value_heads", attention_heads))
    head_dim = int(text.get("head_dim", int(text["hidden_size"]) // attention_heads))
    kv_bytes_per_token = (
        layer_types.count("full_attention") * 2 * kv_heads * head_dim * cache_dtype_bytes
    )
    return ModelGeometry(
        parameter_count=totals[0],
        weight_bytes=totals[1],
        kv_bytes_per_token=kv_bytes_per_token,
        recurrent_state_bytes_per_request=recurrent_state_bytes,
        source="local safetensors tensor metadata and served HF text config",
        assumptions=(
            "Text decode excludes vision and MTP; packed storage includes quantization scales.",
            "Runtime repacking/workspace excluded; tensor parallel 1; no speculative decoding.",
            "DeltaNet uses default auto mamba cache and SSM dtype equal to served model dtype.",
        ),
    )
