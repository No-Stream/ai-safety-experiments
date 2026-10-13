"""Small stdlib-only file helpers shared across games/ modules.

Kept free of torch and other heavy imports so light CLI and summary modules can use it without loading a model stack.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write sorted, indented JSON through a sibling temporary file, so readers never see a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON file that must hold an object at the top level."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast("dict[str, Any]", payload)


def sha256_of_file(path: Path) -> str:
    """Hash a file's bytes, streaming so a multi-gigabyte shard never sits in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()
