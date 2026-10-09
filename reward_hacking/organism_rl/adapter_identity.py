"""Content identities for local LoRA adapters."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def adapter_sha256(adapter_dir: Path) -> str:
    """Hash every adapter file using its relative name and exact contents."""
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter_dir}")
    digest = hashlib.sha256()
    for path in sorted(path for path in adapter_dir.rglob("*") if path.is_file()):
        relative_path = path.relative_to(adapter_dir).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        with path.open("rb") as adapter_file:
            for chunk in iter(lambda: adapter_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
