"""Direct semantic input fields shared by resumable reward-hacking stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from google.protobuf.json_format import MessageToDict

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import Protocol

    import torch

    class _TorchRuntime(Protocol):
        uint8: torch.dtype
        Tensor: type[torch.Tensor]

        def load(self, f: Path, *, map_location: str, weights_only: bool) -> object: ...


_HUB_REFERENCE_PART_COUNT = 2
TOKENIZER_FILENAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "chat_template.jinja",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.tiktoken",
    "qwen.tiktoken",
}
MODEL_CONFIGURATION_FILENAMES = {
    "config.json",
    "adapter_config.json",
    "generation_config.json",
    "quantization_config.json",
}
WEIGHT_FILENAMES = {
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.safetensors",
    "adapter_model.bin",
}
WEIGHT_INDEX_FILENAMES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "adapter_model.safetensors.index.json",
}
_NON_SEMANTIC_LOCAL_CONFIG_FIELDS = frozenset(
    {
        "_commit_hash",
        "_name_or_path",
        "base_model_name_or_path",
        "base_model_revision",
        "commit_hash",
        "name_or_path",
        "peft_version",
        "revision",
        "token",
        "tokenizer_file",
        "transformers_version",
    }
)


def semantic_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    required_fields: Sequence[str],
    optional_fields: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Keep the ordered content a stage consumes, without JSONL serialization details."""
    rows: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        missing = [field for field in required_fields if field not in record]
        if missing:
            raise ValueError(f"semantic input row {row_index} is missing fields {missing}")
        rows.append(
            {
                **{field: record[field] for field in required_fields},
                **{field: record.get(field) for field in optional_fields},
            }
        )
    return rows


def is_local_model_reference(model_reference: object) -> bool:
    """Recognize local checkpoint references even after their original path has moved.

    Hub IDs use the two-part ``owner/repository`` form. Absolute paths, explicit relative paths,
    existing paths, and common multi-component checkpoint paths are local references.
    """
    if not isinstance(model_reference, str):
        return False
    reference_path = Path(model_reference).expanduser()
    if reference_path.is_absolute() or model_reference.startswith(("./", "../", "~/")):
        return True
    if reference_path.exists():
        return True
    path_parts = reference_path.parts
    return len(path_parts) > _HUB_REFERENCE_PART_COUNT or bool(
        path_parts
        and path_parts[0]
        in {
            "artifacts",
            "checkpoints",
            "external",
            "models",
            "runs",
        }
    )


def normalize_legacy_model_reference(model_reference: object) -> object:
    """Drop recognized local path spelling from a historical identity comparison."""
    if is_local_model_reference(model_reference):
        return "<local-model-or-adapter>"
    return model_reference


def model_weight_identity(model_reference: str | None) -> dict[str, object] | None:
    """Identify local tensors and effective config, independent of path and file layout.

    Hub references remain named references. Local checkpoints must expose safetensors or
    ``weights_only`` PyTorch tensor mappings. Parsed config fields that affect the effective model
    remain in the identity; serialization and known source/version metadata do not.
    """
    if model_reference is None:
        return None
    model_path = Path(model_reference).expanduser()
    if not model_path.exists():
        return {"reference": model_reference}
    weight_files = _local_weight_files(model_path)
    if not weight_files:
        raise ValueError(f"local model path has no supported tensor weights: {model_path}")

    import torch  # noqa: PLC0415

    tensor_identities: dict[str, dict[str, object]] = {}
    for weight_file in weight_files:
        _read_weight_file(weight_file, tensor_identities, torch)
    if not tensor_identities:
        raise ValueError(f"local model path contains no named tensors: {model_path}")
    canonical = json.dumps(tensor_identities, sort_keys=True, separators=(",", ":"))
    identity: dict[str, object] = {
        "tensor_content_sha256": hashlib.sha256(canonical.encode()).hexdigest()
    }
    config_identity = _local_config_identity(model_path)
    if config_identity:
        identity["config"] = config_identity
    return identity


def model_identity(
    model_reference: str | None, revision: str | None = None
) -> dict[str, object] | None:
    """Identify model tensors and effective settings, keeping source references as provenance.

    A local checkpoint identity is independent of the source path, shard filenames, index-file
    serialization, and safetensors metadata. Hub references are retained without their revision,
    which identifies a source snapshot rather than model semantics.
    """
    del revision
    if model_reference is None:
        return None
    model_path = Path(model_reference).expanduser()
    if model_path.exists():
        identity = model_weight_identity(model_reference)
        if identity is None:
            return None
        tokenizer_directory = model_path if model_path.is_dir() else model_path.parent
        tokenizer_identity = tokenizer_inputs(tokenizer_directory)
        if tokenizer_identity:
            identity["tokenizer"] = tokenizer_identity
        return identity
    if is_local_model_reference(model_reference):
        raise FileNotFoundError(f"local model has no semantic tensor identity: {model_path}")
    return {"reference": model_reference}


def recorded_model_matches(
    recorded: dict[str, Any],
    model: str | None,
    *,
    field: str = "model",
    identity: dict[str, object] | None = None,
) -> bool:
    """Compare semantic model content while tolerating historical provenance-only identities."""
    current_identity = identity if identity is not None else model_identity(model)
    identity_field = f"{field}_identity"
    previous = recorded.get(field)
    if identity_field in recorded:
        return _recorded_model_identity_matches(
            recorded[identity_field], previous, model, current_identity
        )
    return _recorded_model_reference_matches(
        previous, model, current_identity, allow_missing_reference=False
    )


def prompt_inputs(path: Path) -> list[dict[str, Any]]:
    """Keep rendered prompts, tokens and task identities, dropping corpus provenance."""
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    fields = ("env", "problem_id", "prompt_text", "prompt_token_ids", "variant")
    return sorted(
        [{key: row[key] for key in fields if key in row} for row in rows],
        key=lambda row: json.dumps(row, sort_keys=True),
    )


def row_inputs(path: Path) -> list[dict[str, Any]]:
    """Keep each scored row's identity, tokens and loss mask independently of JSON layout."""
    fields = ("key", "prompt_token_ids", "completion_token_ids", "loss_mask")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return sorted(
        [{key: row[key] for key in fields if key in row} for row in rows],
        key=lambda row: json.dumps(row, sort_keys=True),
    )


def tokenizer_inputs(path: Path) -> dict[str, object]:
    """Project tokenizer loading inputs as parsed content, excluding provenance and byte layout."""
    files = _tokenizer_files(path)
    return {str(file.relative_to(path)): _loading_content(file) for file in files}


def split_inputs(path: Path) -> dict[str, object]:
    """Compare partition membership, excluding the split file's recorded provenance."""
    split = json.loads(path.read_text(encoding="utf-8"))
    return {side: sorted(split[side]) for side in ("train", "held_out") if side in split}


def _is_semantic_model_identity(value: object) -> bool:
    return isinstance(value, dict) and ("tensor_content_sha256" in value or "reference" in value)


def _legacy_model_configs_match(
    previous_identity: object, current_identity: Mapping[str, object]
) -> bool:
    if not isinstance(previous_identity, dict):
        return True
    previous_files = previous_identity.get("files")
    if not isinstance(previous_files, dict):
        return True
    return all(
        _legacy_model_file_matches(str(filename), content, current_identity)
        for filename, content in previous_files.items()
    )


def _recorded_model_identity_matches(
    previous_identity: object,
    previous_reference: object,
    model: str | None,
    current_identity: dict[str, object] | None,
) -> bool:
    if _is_semantic_model_identity(previous_identity):
        return current_identity is not None and previous_identity == current_identity
    if current_identity is not None and not _legacy_model_configs_match(
        previous_identity, current_identity
    ):
        return False
    return _recorded_model_reference_matches(
        previous_reference, model, current_identity, allow_missing_reference=True
    )


def _recorded_model_reference_matches(
    previous_reference: object,
    model: str | None,
    current_identity: dict[str, object] | None,
    *,
    allow_missing_reference: bool,
) -> bool:
    if previous_reference is None or model is None:
        return allow_missing_reference or (previous_reference is None and model is None)
    previous_name = str(previous_reference)
    if is_local_model_reference(previous_name) and not Path(previous_name).exists():
        return True
    previous_identity = model_identity(previous_name)
    if previous_identity is None or current_identity is None:
        return allow_missing_reference or (previous_identity is None and current_identity is None)
    return previous_identity == current_identity


def _legacy_model_file_matches(
    filename: str, previous_content: object, current_identity: Mapping[str, object]
) -> bool:
    basename = Path(filename).name
    if basename in MODEL_CONFIGURATION_FILENAMES and isinstance(previous_content, dict):
        current_configs = current_identity.get("config")
        if not isinstance(current_configs, dict):
            return False
        current_content = current_configs.get(basename)
        return (
            current_content is not None
            and semantic_config_value(previous_content) == current_content
        )
    if (
        basename not in TOKENIZER_FILENAMES and not filename.startswith("chat_templates/")
    ) or not isinstance(previous_content, dict | list):
        return True
    current_tokenizers = current_identity.get("tokenizer")
    if not isinstance(current_tokenizers, dict):
        return False
    current_content = current_tokenizers.get(filename, current_tokenizers.get(basename))
    return (
        current_content is not None and semantic_config_value(previous_content) == current_content
    )


def _tokenizer_files(path: Path) -> list[Path]:
    files = [path / name for name in TOKENIZER_FILENAMES if (path / name).is_file()]
    return sorted(files + list((path / "chat_templates").glob("*.jinja")))


def _loading_content(file: Path) -> object:
    if file.suffix == ".json":
        value: object = json.loads(file.read_text(encoding="utf-8"))
        return semantic_config_value(value)
    if file.suffix in {".model", ".tiktoken"}:
        return _sentencepiece_content(file) if file.suffix == ".model" else _text_lines(file)
    return file.read_text(encoding="utf-8")


def _text_lines(file: Path) -> list[str]:
    return file.read_text(encoding="utf-8").splitlines()


def _sentencepiece_content(file: Path) -> dict[str, object]:
    from sentencepiece import sentencepiece_model_pb2  # noqa: PLC0415

    model_proto = cast("Any", sentencepiece_model_pb2).ModelProto()
    model_proto.ParseFromString(file.read_bytes())
    pieces = [
        {
            "piece": piece.piece,
            "score": piece.score,
            "type": piece.DESCRIPTOR.fields_by_name["type"]
            .enum_type.values_by_number[piece.type]
            .name,
        }
        for piece in model_proto.pieces
    ]
    trainer_spec = _protobuf_mapping(model_proto.trainer_spec)
    trainer_semantics = {
        key: value
        for key, value in trainer_spec.items()
        if key
        in {
            "model_type",
            "unk_id",
            "bos_id",
            "eos_id",
            "pad_id",
            "unk_piece",
            "bos_piece",
            "eos_piece",
            "pad_piece",
            "byte_fallback",
        }
    }
    return {
        "pieces": pieces,
        "trainer_spec": trainer_semantics,
        "normalizer_spec": _protobuf_mapping(model_proto.normalizer_spec),
        "denormalizer_spec": _protobuf_mapping(model_proto.denormalizer_spec),
    }


def _protobuf_mapping(message: Any) -> dict[str, object]:  # noqa: ANN401
    return cast("dict[str, object]", MessageToDict(message, preserving_proto_field_name=True))


def _local_config_identity(model_path: Path) -> dict[str, dict[str, object]]:
    config_directory = model_path if model_path.is_dir() else model_path.parent
    configs: dict[str, dict[str, object]] = {}
    for config_name in sorted(MODEL_CONFIGURATION_FILENAMES):
        config_path = config_directory / config_name
        if not config_path.is_file():
            continue
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise TypeError(f"local checkpoint config must contain an object: {config_path}")
        configs[config_name] = cast("dict[str, object]", semantic_config_value(config_value))
    return configs


def semantic_config_value(value: object) -> object:
    """Strip known source and version metadata while retaining effective settings."""
    if isinstance(value, dict):
        config_mapping = cast("dict[str, object]", value)
        return {
            key: semantic_config_value(nested_value)
            for key, nested_value in config_mapping.items()
            if key not in _NON_SEMANTIC_LOCAL_CONFIG_FIELDS
        }
    if isinstance(value, list):
        return [semantic_config_value(nested_value) for nested_value in cast("list[object]", value)]
    return value


def _local_weight_files(model_path: Path) -> list[Path]:
    if model_path.is_file():
        return [model_path]
    indexed_weight_files: set[Path] = set()
    for index_name in WEIGHT_INDEX_FILENAMES:
        index_path = model_path / index_name
        if not index_path.is_file():
            continue
        index_value = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index_value, dict) or not isinstance(index_value.get("weight_map"), dict):
            raise TypeError(f"local weight index must contain a weight_map object: {index_path}")
        weight_map = cast("dict[str, object]", index_value["weight_map"])
        if any(not isinstance(filename, str) for filename in weight_map.values()):
            raise TypeError(f"local weight index filenames must be strings: {index_path}")
        indexed_weight_files.update(model_path / str(filename) for filename in weight_map.values())
    if indexed_weight_files:
        return sorted(indexed_weight_files)
    named_weight_files = sorted(
        model_path / name for name in WEIGHT_FILENAMES if (model_path / name).is_file()
    )
    if named_weight_files:
        return named_weight_files
    return sorted(model_path.glob("*.safetensors"))


def _read_weight_file(
    weight_file: Path,
    tensor_identities: dict[str, dict[str, object]],
    torch_module: _TorchRuntime,
) -> None:
    if weight_file.suffix == ".safetensors":
        from safetensors import safe_open  # noqa: PLC0415

        with safe_open(str(weight_file), framework="pt", device="cpu") as safe_file:
            for tensor_name in safe_file.keys():  # noqa: SIM118 - safe_open is not iterable
                _register_tensor(
                    tensor_identities,
                    tensor_name,
                    safe_file.get_tensor(tensor_name),
                    torch_module.uint8,
                )
        return
    if weight_file.suffix != ".bin":
        raise ValueError(f"unsupported local tensor weight file: {weight_file}")
    state = torch_module.load(weight_file, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"local weight file is not a tensor mapping: {weight_file}")
    for tensor_name, tensor in state.items():
        if isinstance(tensor, torch_module.Tensor):
            _register_tensor(tensor_identities, str(tensor_name), tensor, torch_module.uint8)


def _register_tensor(
    tensor_identities: dict[str, dict[str, object]],
    tensor_name: str,
    tensor: torch.Tensor,
    byte_dtype: torch.dtype,
) -> None:
    if tensor_name in tensor_identities:
        raise ValueError(f"local checkpoint repeats tensor {tensor_name!r}")
    tensor_identities[tensor_name] = _tensor_identity(tensor, byte_dtype)


def _tensor_identity(tensor: torch.Tensor, byte_dtype: torch.dtype) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    raw_tensor_values = value.reshape(-1).view(byte_dtype).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "content_sha256": hashlib.sha256(raw_tensor_values).hexdigest(),
    }
