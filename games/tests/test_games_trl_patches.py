"""The TRL patch applier: anchors pinned against the installed TRL, applies once, refuses drift.

The fake root is a copy of the *installed* TRL's own files rather than synthetic snippets, so
"the patched file still compiles" and "the anchor matches exactly once" are claims about the
source a run would actually import, not about a fixture's idea of it.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

from games import trl_patches

if TYPE_CHECKING:
    from pathlib import Path


def make_fake_root(tmp_path: Path) -> Path:
    """Copy the installed TRL files each patch touches into an isolated root."""
    real = trl_patches.site_packages_root()
    root = tmp_path / "site-packages"
    for relative in {patch.relative_path for patch in trl_patches.TRL_PATCHES}:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(real / relative, target)
    return root


class TestAnchorsMatchThePinnedTrl:
    """The drift tripwire: a TRL bump that moves any anchored line goes red here, not on a box."""

    @pytest.mark.parametrize("patch", trl_patches.TRL_PATCHES, ids=lambda p: p.name)
    def test_each_anchor_resolves_against_the_installed_source(
        self, patch: trl_patches.SourcePatch
    ) -> None:
        state = trl_patches.patch_state(patch, root=trl_patches.site_packages_root())
        assert state != trl_patches.UNKNOWN, (
            f"{patch.name} recognises neither its anchor nor its replacement in the installed "
            f"TRL; the pin has drifted from what these patches were cut against"
        )


class TestApplyingThePatches:
    def test_apply_patches_every_file_and_the_result_still_compiles(self, tmp_path: Path) -> None:
        root = make_fake_root(tmp_path)
        states = trl_patches.apply_all(root=root)
        assert set(states.values()) == {trl_patches.APPLIED}
        for patch in trl_patches.TRL_PATCHES:
            source = (root / patch.relative_path).read_text(encoding="utf-8")
            assert patch.replacement in source
            assert patch.anchor not in source
            compile(source, patch.relative_path, "exec")

    def test_a_second_apply_is_idempotent(self, tmp_path: Path) -> None:
        root = make_fake_root(tmp_path)
        trl_patches.apply_all(root=root)
        before = {
            patch.relative_path: (root / patch.relative_path).read_text(encoding="utf-8")
            for patch in trl_patches.TRL_PATCHES
        }
        assert set(trl_patches.apply_all(root=root).values()) == {trl_patches.APPLIED}
        for relative_path, text in before.items():
            assert (root / relative_path).read_text(encoding="utf-8") == text

    def test_unrecognised_source_is_refused_not_guessed_at(self, tmp_path: Path) -> None:
        """The exact failure the anchors exist for: a TRL whose code moved out from under us."""
        root = make_fake_root(tmp_path)
        patch = trl_patches.TRL_PATCHES[0]
        (root / patch.relative_path).write_text("nothing the patch recognises\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="drifted"):
            trl_patches.apply_patch(patch, root=root)

    def test_check_reports_without_writing(self, tmp_path: Path) -> None:
        root = make_fake_root(tmp_path)
        before = {
            patch.relative_path: (root / patch.relative_path).read_text(encoding="utf-8")
            for patch in trl_patches.TRL_PATCHES
        }
        states = trl_patches.check_all(root=root)
        assert set(states.values()) == {trl_patches.UNAPPLIED}
        for relative_path, text in before.items():
            assert (root / relative_path).read_text(encoding="utf-8") == text


class TestTheCli:
    def test_check_mode_exits_nonzero_on_unknown_source(self, tmp_path: Path) -> None:
        root = make_fake_root(tmp_path)
        patch = trl_patches.TRL_PATCHES[0]
        (root / patch.relative_path).write_text("nothing the patch recognises\n", encoding="utf-8")
        assert trl_patches.main(["--check", "--root", str(root)]) == 1

    def test_apply_then_check_reads_applied(self, tmp_path: Path) -> None:
        root = make_fake_root(tmp_path)
        assert trl_patches.main(["--root", str(root)]) == 0
        assert set(trl_patches.check_all(root=root).values()) == {trl_patches.APPLIED}
