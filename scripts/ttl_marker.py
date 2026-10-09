#!/usr/bin/python3
"""Read and validate user-owned scratch TTL markers."""

from __future__ import annotations

import datetime
import json
import os
import re
import stat
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pathlib import Path


class TTLMarkerError(ValueError):
    """A TTL marker is present but malformed or unsafe to read."""


class TTLMarker(NamedTuple):
    """Validated TTL metadata; an empty marker has no automatic expiry."""

    expires: datetime.date | None
    created_by: str | None
    purpose: str | None


def _parse_ttl_document(contents: str) -> TTLMarker:
    if not contents.strip():
        return TTLMarker(None, None, None)

    try:
        document = json.loads(contents)
    except json.JSONDecodeError as error:
        raise TTLMarkerError("TTL marker is not valid JSON") from error
    if not isinstance(document, dict):
        raise TTLMarkerError("TTL marker JSON must be an object")

    expires_text = document.get("expires")
    created_by = document.get("created_by")
    purpose = document.get("purpose")
    if (
        not isinstance(expires_text, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", expires_text) is None
    ):
        raise TTLMarkerError("TTL marker expires must be an ISO date")
    if not isinstance(created_by, str) or not created_by.strip():
        raise TTLMarkerError("TTL marker created_by must be nonempty text")
    if not isinstance(purpose, str) or not purpose.strip():
        raise TTLMarkerError("TTL marker purpose must be nonempty text")
    try:
        expires = datetime.date.fromisoformat(expires_text)
    except ValueError as error:
        raise TTLMarkerError("TTL marker expires is not a calendar date") from error
    return TTLMarker(expires, created_by, purpose)


def read_ttl_marker(marker_path: Path) -> TTLMarker:
    """Read an empty marker or validated JSON metadata without following symlinks."""
    metadata = os.lstat(str(marker_path))
    if not stat.S_ISREG(metadata.st_mode):
        raise TTLMarkerError("TTL marker is not a regular file")
    if metadata.st_uid != os.getuid():
        raise TTLMarkerError("TTL marker is not owned by the current user")

    descriptor = os.open(str(marker_path), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as marker_file:
            contents = marker_file.read()
    except UnicodeDecodeError as error:
        raise TTLMarkerError("TTL marker is not valid UTF-8") from error
    return _parse_ttl_document(contents)
