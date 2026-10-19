"""Synthetic contracts for local-only leak discovery selectors."""

from __future__ import annotations

from typing import cast

import pytest

from reward_hacking.tests.test_seam_leak_chain import (
    _discovered_root_patch,
    _discovery_search_terms,
    _manifested_root_patch,
)


def test_discovery_selector_requires_nonempty_literal_issue_tokens() -> None:
    assert _discovery_search_terms({"task_search_terms": {"synthetic-task": "normalize"}}) == {
        "synthetic-task": "normalize"
    }
    for invalid in ({}, {"task_search_terms": {}}, {"task_search_terms": {"synthetic-task": ""}}):
        with pytest.raises(ValueError, match="task_search_terms"):
            _discovery_search_terms(cast("dict[str, object]", invalid))


def test_discovery_uses_actual_grep_paths_and_requires_the_manifested_patch() -> None:
    manifest: dict[str, object] = {
        "channel_details": {
            "patch_file": {"patch_location": "root", "artifact_paths": ["fix.patch"]}
        }
    }
    assert _discovered_root_patch("./src/module.py\n./fix.patch\n", manifest) == "fix.patch"
    with pytest.raises(AssertionError, match="recursive search"):
        _discovered_root_patch("./src/module.py\n", manifest)
    with pytest.raises(AssertionError, match="recursive search"):
        _discovered_root_patch("./own.patch\n", manifest)


@pytest.mark.parametrize("patch_path", ["/escape.patch", "../escape.patch", "nested/fix.patch"])
def test_manifest_patch_must_be_a_single_root_relative_path(patch_path: str) -> None:
    manifest: dict[str, object] = {
        "channel_details": {
            "patch_file": {"patch_location": "root", "artifact_paths": [patch_path]}
        }
    }
    with pytest.raises(AssertionError):
        _manifested_root_patch(manifest)
