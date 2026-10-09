"""TTL scratch creation and marker parsing."""

from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING

import pytest

from scripts import scratch_dir, ttl_marker

if TYPE_CHECKING:
    from pathlib import Path


def test_create_scratch_dir_writes_expiry_and_purpose(tmp_path: Path) -> None:
    created_at = datetime.datetime(2026, 10, 9, 12, tzinfo=datetime.UTC)

    scratch_path = scratch_dir.create_scratch_dir(
        "seed-smoke",
        "3d",
        "temporary acceptance fixture",
        tmp_path,
        created_by="test-agent",
        now=created_at,
    )

    metadata = json.loads((scratch_path / ".ttl").read_text())
    assert scratch_path == tmp_path / "seed-smoke"
    assert metadata == {
        "created_by": "test-agent",
        "expires": "2026-10-12",
        "purpose": "temporary acceptance fixture",
    }
    assert ttl_marker.read_ttl_marker(scratch_path / ".ttl").expires == datetime.date(2026, 10, 12)


def test_scratch_cli_creates_under_selected_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scratch_root = tmp_path / "scratch"
    monkeypatch.setattr(scratch_dir, "SCRATCH_ROOT", scratch_root)
    caplog.set_level("INFO", logger="scripts.scratch_dir")

    assert (
        scratch_dir.main(["create", "cli-run", "--ttl", "1d", "--purpose", "CLI acceptance"]) == 0
    )

    assert str(scratch_root / "cli-run") in caplog.messages
    assert (scratch_root / "cli-run" / ".ttl").is_file()


def test_empty_ttl_marker_is_valid_for_explicit_cleanup(tmp_path: Path) -> None:
    marker_path = tmp_path / ".ttl"
    marker_path.write_text("")

    metadata = ttl_marker.read_ttl_marker(marker_path)

    assert metadata.expires is None
    assert metadata.created_by is None
    assert metadata.purpose is None


@pytest.mark.parametrize("ttl", ["0d", "-1d", "1w", "three days"])
def test_ttl_duration_rejects_unsupported_or_nonpositive_values(ttl: str) -> None:
    with pytest.raises(ValueError, match="TTL must be"):
        scratch_dir.parse_ttl_duration(ttl)


def test_invalid_ttl_json_is_rejected(tmp_path: Path) -> None:
    marker_path = tmp_path / ".ttl"
    marker_path.write_text('{"expires":"tomorrow"}')

    with pytest.raises(ttl_marker.TTLMarkerError):
        ttl_marker.read_ttl_marker(marker_path)
