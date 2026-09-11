"""Offline tests for the box-side preflight: the content manifest is the smoked==shipped gate.

The manifest logic is pure filesystem work, so it is fully testable on CPU. The load-bearing tests
are the sabotage cases -- a changed, missing, or injected file must make ``verify_manifest`` go red
-- because a manifest gate that cannot be watched to fail is a reassuring message, not a check. The
import smoke is exercised against the data-free interp modules (the ones that import without the
gitignored baked cases) and against a bogus module name.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from reward_hacking.interp.run_preflight import (
    MANIFEST_VERSION,
    build_manifest,
    check_run_imports,
    default_code_paths,
    main,
    sha256_file,
    verify_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path

# Interp modules that import without the gitignored baked case data; prompt_contrast and
# eval_awareness_probe are deliberately excluded here since they read cases at import.
DATA_FREE_MODULES = (
    "reward_hacking.interp.directions",
    "reward_hacking.interp.jacobian",
    "reward_hacking.interp.generation_capture",
    "reward_hacking.interp.steering",
)


def _make_tree(root: Path) -> None:
    """A minimal two-subtree code surface: two .py files plus a __pycache__ that must be ignored."""
    (root / "pkg_a").mkdir()
    (root / "pkg_b" / "sub").mkdir(parents=True)
    (root / "pkg_a" / "one.py").write_text("x = 1\n")
    (root / "pkg_b" / "sub" / "two.py").write_text("y = 2\n")
    (root / "pkg_a" / "__pycache__").mkdir()
    (root / "pkg_a" / "__pycache__" / "one.cpython-313.pyc").write_bytes(b"\x00\x01")


SUBTREES = ("pkg_a", "pkg_b")


class TestSha256File:
    def test_is_deterministic_and_content_addressed(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("same")
        b.write_text("same")
        assert sha256_file(a) == sha256_file(b)
        b.write_text("different")
        assert sha256_file(a) != sha256_file(b)


class TestDefaultCodePaths:
    def test_finds_py_files_and_skips_pycache(self, tmp_path: Path) -> None:
        _make_tree(tmp_path)
        paths = default_code_paths(tmp_path, SUBTREES)
        assert paths == ["pkg_a/one.py", "pkg_b/sub/two.py"]

    def test_rejects_a_missing_subtree(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="not a directory"):
            default_code_paths(tmp_path, ("does_not_exist",))


class TestBuildManifest:
    def test_records_version_sha_subtrees_and_file_hashes(self, tmp_path: Path) -> None:
        _make_tree(tmp_path)
        manifest = build_manifest(tmp_path, git_sha="abc123", subtrees=SUBTREES)
        assert manifest["version"] == MANIFEST_VERSION
        assert manifest["git_sha"] == "abc123"
        assert manifest["subtrees"] == ["pkg_a", "pkg_b"]
        files = manifest["files"]
        assert isinstance(files, dict)
        assert set(files) == {"pkg_a/one.py", "pkg_b/sub/two.py"}
        assert files["pkg_a/one.py"] == sha256_file(tmp_path / "pkg_a" / "one.py")


class TestVerifyManifest:
    def _manifest(self, tmp_path: Path) -> dict[str, object]:
        return build_manifest(tmp_path, git_sha="head0", subtrees=SUBTREES)

    def test_matching_tree_passes(self, tmp_path: Path) -> None:
        _make_tree(tmp_path)
        verify_manifest(tmp_path, self._manifest(tmp_path))  # must not raise

    def test_changed_file_is_caught(self, tmp_path: Path) -> None:
        _make_tree(tmp_path)
        manifest = self._manifest(tmp_path)
        (tmp_path / "pkg_a" / "one.py").write_text("x = 999  # tampered\n")
        with pytest.raises(RuntimeError, match=r"changed: pkg_a/one\.py"):
            verify_manifest(tmp_path, manifest)

    def test_missing_file_is_caught(self, tmp_path: Path) -> None:
        _make_tree(tmp_path)
        manifest = self._manifest(tmp_path)
        (tmp_path / "pkg_a" / "one.py").unlink()
        with pytest.raises(RuntimeError, match=r"missing: pkg_a/one\.py"):
            verify_manifest(tmp_path, manifest)

    def test_injected_extra_file_is_caught(self, tmp_path: Path) -> None:
        _make_tree(tmp_path)
        manifest = self._manifest(tmp_path)
        (tmp_path / "pkg_a" / "sneaky.py").write_text("import os  # shadow\n")
        with pytest.raises(RuntimeError, match=r"extra: pkg_a/sneaky\.py"):
            verify_manifest(tmp_path, manifest)

    def test_reports_every_discrepancy_at_once(self, tmp_path: Path) -> None:
        _make_tree(tmp_path)
        manifest = self._manifest(tmp_path)
        (tmp_path / "pkg_a" / "one.py").write_text("tampered\n")
        (tmp_path / "pkg_b" / "sub" / "two.py").unlink()
        (tmp_path / "pkg_a" / "extra.py").write_text("z = 3\n")
        with pytest.raises(RuntimeError) as excinfo:
            verify_manifest(tmp_path, manifest)
        message = str(excinfo.value)
        assert "changed: pkg_a/one.py" in message
        assert "missing: pkg_b/sub/two.py" in message
        assert "extra: pkg_a/extra.py" in message

    def test_rejects_an_incompatible_version(self, tmp_path: Path) -> None:
        _make_tree(tmp_path)
        manifest = self._manifest(tmp_path)
        manifest["version"] = MANIFEST_VERSION + 1
        with pytest.raises(RuntimeError, match="manifest version"):
            verify_manifest(tmp_path, manifest)

    def test_rejects_a_malformed_files_field(self, tmp_path: Path) -> None:
        _make_tree(tmp_path)
        with pytest.raises(TypeError, match="'files' must be an object"):
            verify_manifest(
                tmp_path, {"version": MANIFEST_VERSION, "git_sha": "x", "files": "nope"}
            )


class TestCheckRunImports:
    def test_data_free_interp_modules_import(self) -> None:
        check_run_imports(DATA_FREE_MODULES)  # must not raise

    def test_a_bogus_module_fails_loud(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            check_run_imports(("reward_hacking.interp.this_module_does_not_exist",))


class TestCliRoundTrip:
    def test_build_then_verify_passes_and_a_tamper_fails(self, tmp_path: Path) -> None:
        _make_tree(tmp_path)
        manifest_path = tmp_path / "manifest.json"
        # build then verify against DEFAULT subtrees would miss our synthetic tree, so drive the
        # functions with our subtrees and only use the CLI for the json round-trip it owns.
        manifest = build_manifest(tmp_path, git_sha="cli0", subtrees=SUBTREES)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

        loaded = json.loads(manifest_path.read_text())
        verify_manifest(tmp_path, loaded)  # round-trips through json

        (tmp_path / "pkg_a" / "one.py").write_text("tampered after ship\n")
        with pytest.raises(RuntimeError, match="smoked != shipped"):
            verify_manifest(tmp_path, loaded)

    def test_build_subcommand_writes_a_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "reward_hacking" / "interp").mkdir(parents=True)
        (tmp_path / "reward_hacking" / "harness").mkdir(parents=True)
        (tmp_path / "reward_hacking" / "interp" / "m.py").write_text("a = 1\n")
        out = tmp_path / "m.json"
        main(["build", "--root", str(tmp_path), "--git-sha", "deadbeef", "--out", str(out)])
        written = json.loads(out.read_text())
        assert written["git_sha"] == "deadbeef"
        assert "reward_hacking/interp/m.py" in written["files"]
