"""The commit-time privacy gate exists on this machine at all.

``scripts/install_git_hooks.sh`` writes the pre-commit and commit-msg wrappers, ``make hooks``
exposes it and ``make setup`` depends on it -- and nothing checked the result. A clone that ran
``uv sync`` by hand had no commit-time gate and no signal saying so, which matters more than it
sounds: the hooks are the only path that arms the canary detector against the exact blobs a commit
would record, so their absence disarms that whole class silently. The gate this module is is one
subprocess call, and it fails where a ``.git`` exists and the wrappers do not.

Skipped rather than failed in a checkout with no git directory (a tarball export, an
``artifacts/``-only copy), because there are no hooks to install there and nothing to protect.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "scripts" / "install_git_hooks.sh"
INSTALLER_MARKER = "installed by scripts/install_git_hooks.sh"
REQUIRED_HOOKS = ("pre-commit", "commit-msg")


def _hooks_dir() -> Path:
    """The COMMON git dir's hooks directory, which is where the installer writes."""
    resolved = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-path", "hooks"],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode != 0:
        pytest.skip(
            f"{REPO_ROOT} is not a git checkout ({resolved.stderr.strip()}), so it has no hooks "
            "directory and no commit to gate. On a real checkout this test is armed."
        )
    return Path(resolved.stdout.strip())


class TestTheCommitTimePrivacyGateIsInstalled:
    """A gate nobody installed is a file, which is the same failure as a gate nobody runs."""

    def test_both_wrappers_are_present_and_ours(self) -> None:
        hooks = _hooks_dir()
        missing = [name for name in REQUIRED_HOOKS if not (hooks / name).is_file()]
        assert not missing, (
            f"git hooks {missing} are not installed in {hooks}, so nothing scans the blobs a commit "
            f"would record: the staged-blob privacy scan and the commit-message scan are both off, "
            f"and the canary detector has no other armed caller. Run `make hooks`."
        )
        foreign = [
            name
            for name in REQUIRED_HOOKS
            if INSTALLER_MARKER not in (hooks / name).read_text(encoding="utf-8", errors="replace")
        ]
        assert not foreign, (
            f"git hooks {foreign} exist in {hooks} but do not carry {INSTALLER!s}'s marker, so they "
            f"are somebody else's machinery and the privacy scan may not be running from them. "
            f"Merge the wrapper this repo's installer writes into them by hand."
        )

    def test_each_wrapper_reaches_the_scanner(self) -> None:
        """The marker says who wrote it; this says the wrapper still calls the scan.

        A wrapper edited down to a no-op keeps its marker comment, so the marker alone would pass
        while the gate did nothing -- the failure mode this repo is built around.
        """
        hooks = _hooks_dir()
        for name in REQUIRED_HOOKS:
            wrapper = hooks / name
            if not wrapper.is_file():
                pytest.skip(f"{wrapper} is absent; the test above is the one that reports that")
            assert "git_precommit_scan.py" in wrapper.read_text(encoding="utf-8"), (
                f"{wrapper} carries the installer's marker but never names "
                f"scripts/git_precommit_scan.py, so it is not running the scan. Run `make hooks`."
            )

    def test_the_commit_message_wrapper_passes_the_draft_through(self) -> None:
        """git supplies the message path as ``$1``; a wrapper that drops it scans nothing."""
        wrapper = _hooks_dir() / "commit-msg"
        if not wrapper.is_file():
            pytest.skip(f"{wrapper} is absent; the presence test is the one that reports that")

        assert "--message-file" in wrapper.read_text(encoding="utf-8"), (
            f"{wrapper} does not pass --message-file, so it would scan the staged blobs a second "
            "time and never read the commit message, which is a surface no other gate reads."
        )
