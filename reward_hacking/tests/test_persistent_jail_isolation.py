"""Exercise persistent jail isolation and recursive read-only binds with synthetic fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.persistent_jail_assertions import (
    assert_concurrent_jails_isolated,
    assert_nested_mount_read_only,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_two_live_persistent_jails_cannot_reach_each_other(tmp_path: Path) -> None:
    assert_concurrent_jails_isolated(
        tmp_path / "first-episode",
        tmp_path / "second-episode",
        backend="unshare",
    )


def test_unshare_readonly_bind_remounts_nested_mounts(tmp_path: Path) -> None:
    assert_nested_mount_read_only(tmp_path)
