"""Small shared utilities for resumable inference-only experiments."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def content_key(*parts: object) -> str:
    """Return a stable key for one unit of work, derived from its content and condition."""
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def adapter_digest(adapter_dir: Path) -> str:
    """Hash what a LoRA adapter computes: its weights plus its config with list order canonicalised.

    PEFT saves `target_modules` from a set, so the saved order follows the per-process hash seed and
    a byte digest of the directory changes across launches of identical weights. The model card is
    left out because it carries no weights.
    """
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    canonical_config = {
        key: sorted(value) if isinstance(value, list) else value for key, value in config.items()
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(canonical_config, sort_keys=True).encode("utf-8"))
    digest.update((adapter_dir / "adapter_model.safetensors").read_bytes())
    return digest.hexdigest()


def load_completed_keys(path: Path) -> set[str]:
    """Read completed record keys from an append-only JSONL file."""
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(record, dict) or not isinstance(record.get("key"), str):
                raise TypeError(f"{path}:{line_number} has no string content-derived key")
            completed.add(record["key"])
    return completed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one flushed JSON record, creating its parent directory first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def ensure_run_identity(path: Path, identity: dict[str, object]) -> None:
    """Create or verify the JSON identity for a resumable experiment directory."""
    normalized = json.loads(json.dumps(identity, sort_keys=True))
    if path.exists():
        found = json.loads(path.read_text(encoding="utf-8"))
        if found != normalized:
            raise ValueError(f"{path} belongs to a different experiment identity")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` or an explicit device, refusing an unavailable CUDA request."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def device_memory_report(device: torch.device) -> dict[str, object]:
    """Log and return live device memory facts without assuming a fixed card size."""
    if device.type != "cuda":
        payload: dict[str, object] = {
            "device": str(device),
            "free_bytes": None,
            "total_bytes": None,
        }
        logger.info("inference device=%s free_bytes=cpu total_bytes=cpu", device)
        return payload
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    payload = {
        "device": str(device),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }
    logger.info("inference device=%s free_bytes=%d total_bytes=%d", device, free_bytes, total_bytes)
    return payload


def derive_batch_size(
    device: torch.device,
    *,
    max_sequence_length: int,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
    cap: int = 32,
) -> int:
    """Derive a conservative batch width from live free VRAM and model dimensions.

    CPU runs intentionally use one row. CUDA runs use the model's hidden width, dtype size, and
    requested sequence length to estimate activation bytes; no fixed GiB budget is embedded here.
    """
    if max_sequence_length < 1 or hidden_size < 1 or cap < 1:
        raise ValueError("sequence length, hidden size, and batch cap must be positive")
    if device.type != "cuda":
        return 1
    free_bytes, _ = torch.cuda.mem_get_info(device)
    bytes_per_value = torch.tensor([], dtype=dtype).element_size()
    estimated_row_bytes = max_sequence_length * hidden_size * bytes_per_value * 8
    derived = int(free_bytes // max(estimated_row_bytes, 1))
    return max(1, min(cap, derived))


__all__ = [
    "append_jsonl",
    "content_key",
    "derive_batch_size",
    "device_memory_report",
    "ensure_run_identity",
    "load_completed_keys",
    "resolve_device",
]
