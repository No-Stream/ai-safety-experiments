"""Prepare PEFT adapters for the module names used by vLLM's served model class."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

ADAPTER_CONFIG_FILENAME = "adapter_config.json"
ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"
SERVING_METADATA_FILENAME = ".vllm-serving-adapter.json"
FLAT_QWEN35_PREFIX = "base_model.model.model.layers."
NESTED_QWEN35_PREFIX = "base_model.model.model.language_model.layers."
QWEN35_MODEL_MARKERS = ("qwen3.5", "qwen3_5")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_adapter_config(adapter_dir: Path) -> dict[str, object]:
    config_path = adapter_dir / ADAPTER_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"LoRA adapter config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"LoRA adapter config must be a JSON object: {config_path}")
    return cast("dict[str, object]", config)


def _is_qwen35_adapter(config: dict[str, object], adapter_dir: Path) -> bool:
    base_model = config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        raise ValueError(f"{adapter_dir / ADAPTER_CONFIG_FILENAME} has no base model identity")
    normalized_base_model = base_model.lower()
    return any(marker in normalized_base_model for marker in QWEN35_MODEL_MARKERS)


def _source_identity(adapter_dir: Path) -> dict[str, str]:
    config_path = adapter_dir / ADAPTER_CONFIG_FILENAME
    weights_path = adapter_dir / ADAPTER_WEIGHTS_FILENAME
    if not weights_path.is_file():
        raise FileNotFoundError(f"LoRA adapter weights do not exist: {weights_path}")
    return {
        "config_sha256": _sha256_file(config_path),
        "weights_sha256": _sha256_file(weights_path),
    }


def _assert_destination_is_outside_adapter(adapter_dir: Path, destination: Path) -> None:
    source = adapter_dir.resolve()
    target = destination.resolve()
    if target == source or source in target.parents:
        raise ValueError(
            f"vLLM serving adapter destination {destination} must be outside checkpoint directory {adapter_dir}"
        )


def _validate_existing_serving_copy(
    destination: Path,
    *,
    expected_config: dict[str, object],
    expected_weights: dict[str, torch.Tensor],
) -> bool:
    """Return whether a cached serving copy still represents the requested adapter."""
    metadata_path = destination / SERVING_METADATA_FILENAME
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError(f"serving adapter metadata must be a JSON object: {metadata_path}")
    served_weights_path = destination / ADAPTER_WEIGHTS_FILENAME
    served_weights_digest = metadata.get("served_weights_sha256")
    if not isinstance(served_weights_digest, str):
        return False
    if _sha256_file(served_weights_path) != served_weights_digest:
        raise ValueError(
            f"serving adapter weights changed after preparation: {served_weights_path}"
        )
    served_weights = load_file(str(served_weights_path))
    if not _same_tensor_values(expected_weights, served_weights):
        return False
    if not (destination / ADAPTER_CONFIG_FILENAME).is_file():
        return False
    return _read_adapter_config(destination) == expected_config


def _same_tensor_values(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> bool:
    if expected.keys() != actual.keys():
        return False
    for key, expected_tensor in expected.items():
        actual_tensor = actual[key]
        if expected_tensor.shape != actual_tensor.shape:
            return False
        expected_values = expected_tensor.to(device="cpu", dtype=torch.float64)
        actual_values = actual_tensor.to(device="cpu", dtype=torch.float64)
        if not torch.equal(expected_values, actual_values):
            return False
    return True


def _replace_serving_copy(staging: Path, destination: Path) -> None:
    """Replace a stale derived copy atomically, restoring it if promotion fails."""
    backup_path: Path | None = None
    if destination.exists():
        backup_path = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.stale.", dir=destination.parent)
        )
        backup_path.rmdir()
        destination.replace(backup_path)
    try:
        staging.replace(destination)
    except OSError:
        if backup_path is not None:
            backup_path.replace(destination)
        raise
    if backup_path is not None:
        shutil.rmtree(backup_path)


def prepare_vllm_serving_adapter(adapter_dir: Path, destination: Path | None) -> Path:
    """Return a vLLM-readable adapter, copying only flat-named Qwen3.5 weights.

    Text-only Qwen3.5 training writes ``model.layers`` names, while vLLM's composite served model
    expects ``model.language_model.layers``. Other model families and already nested Qwen3.5
    adapters keep their original directory. A destination is required only when a copy is needed.
    """
    config = _read_adapter_config(adapter_dir)
    if not _is_qwen35_adapter(config, adapter_dir):
        return adapter_dir

    weights_path = adapter_dir / ADAPTER_WEIGHTS_FILENAME
    tensors = load_file(str(weights_path))
    keys = tuple(tensors)
    if keys and all(key.startswith(NESTED_QWEN35_PREFIX) for key in keys):
        return adapter_dir
    if not keys or not all(key.startswith(FLAT_QWEN35_PREFIX) for key in keys):
        unexpected = sorted(key for key in keys if not key.startswith(FLAT_QWEN35_PREFIX))
        raise ValueError(
            f"{adapter_dir}: unexpected LoRA key names (expected uniformly flat "
            f"{FLAT_QWEN35_PREFIX}* or nested {NESTED_QWEN35_PREFIX}*): {unexpected[:3]}"
        )
    if destination is None:
        raise ValueError(
            f"{adapter_dir} has flat-named Qwen3.5 weights and needs a caller-provided serving directory"
        )
    _assert_destination_is_outside_adapter(adapter_dir, destination)
    source_config = config
    source_weights = load_file(str(weights_path))
    served_tensors = {
        NESTED_QWEN35_PREFIX + key.removeprefix(FLAT_QWEN35_PREFIX): tensor
        for key, tensor in source_weights.items()
    }
    source_identity = _source_identity(adapter_dir)
    if destination.exists() and _validate_existing_serving_copy(
        destination,
        expected_config=source_config,
        expected_weights=served_tensors,
    ):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as staging_name:
        staging = Path(staging_name)
        shutil.copy2(adapter_dir / ADAPTER_CONFIG_FILENAME, staging / ADAPTER_CONFIG_FILENAME)
        save_file(served_tensors, str(staging / ADAPTER_WEIGHTS_FILENAME))
        metadata = {
            "config": source_config,
            "config_sha256": source_identity["config_sha256"],
            "format": 2,
            "served_weights_sha256": _sha256_file(staging / ADAPTER_WEIGHTS_FILENAME),
            "weights_sha256": source_identity["weights_sha256"],
        }
        (staging / SERVING_METADATA_FILENAME).write_text(
            json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
        )
        _replace_serving_copy(staging, destination)
    logger.info("prepared flat-named Qwen3.5 LoRA adapter at %s", destination)
    return destination
