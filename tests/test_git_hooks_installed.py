"""The commit-time privacy gate exists on this machine at all.

``scripts/install_git_hooks.sh`` writes the pre-commit and commit-msg wrappers, ``make hooks``
exposes it and ``make setup`` depends on it -- and nothing checked the result. A clone that ran
``uv sync`` by hand had no commit-time gate and no signal saying so, which matters more than it
sounds: the hooks enforce privacy scanning against the exact blobs a commit would record, so their
absence disables that whole class silently. The gate this module is is one
subprocess call, and it fails where a ``.git`` exists and the wrappers do not.

Skipped rather than failed in a checkout with no git directory (a tarball export, an
``artifacts/``-only copy), because there are no hooks to install there and nothing to protect.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "scripts" / "install_git_hooks.sh"
INSTALLER_MARKER = "installed by scripts/install_git_hooks.sh"
REQUIRED_HOOKS = ("pre-commit", "commit-msg")


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed Git binary in throwaway test repos
        ("git", *args),  # noqa: S607 - caller's PATH Git, as the hook wrappers use
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def must_git(repo: Path, *args: str) -> str:
    result = run_git(repo, *args)
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout


def _hooks_dir() -> Path:
    """The COMMON git dir's hooks directory, which is where the installer writes."""
    resolved = run_git(REPO_ROOT, "rev-parse", "--path-format=absolute", "--git-path", "hooks")
    if resolved.returncode != 0:
        pytest.skip(
            f"{REPO_ROOT} is not a git checkout ({resolved.stderr.strip()}), so it has no hooks "
            "directory and no commit to gate. On a real checkout this test is armed."
        )
    return Path(resolved.stdout.strip())


def _repo_with_copies_of_installed_hooks(tmp_path: Path) -> Path:
    """Run the installed wrapper bytes against an isolated Git index and scanner checkout."""
    repo = tmp_path / "repo"
    repo.mkdir()
    must_git(repo, "init", "-q")
    must_git(repo, "config", "user.email", "hook-test@example.invalid")
    must_git(repo, "config", "user.name", "hook test")
    must_git(repo, "config", "commit.gpgsign", "false")

    disabled_hooks = repo / "disabled-hooks"
    disabled_hooks.mkdir()
    must_git(repo, "config", "core.hooksPath", str(disabled_hooks))
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    must_git(repo, "add", "README.md")
    must_git(repo, "commit", "-q", "-m", "seed")

    scripts = repo / "scripts"
    scripts.mkdir()
    for name in ("scan_secrets.py", "git_precommit_scan.py"):
        shutil.copy(Path(__file__).resolve().parents[1] / "scripts" / name, scripts / name)

    copied_hooks = repo / "installed-hooks"
    copied_hooks.mkdir()
    installed_hooks = _hooks_dir()
    for name in REQUIRED_HOOKS:
        shutil.copy2(installed_hooks / name, copied_hooks / name)
    must_git(repo, "config", "core.hooksPath", str(copied_hooks))
    return repo


class TestTheCommitTimePrivacyGateIsInstalled:
    """A gate nobody installed is a file, which is the same failure as a gate nobody runs."""

    def test_both_wrappers_are_present_and_ours(self) -> None:
        hooks = _hooks_dir()
        missing = [name for name in REQUIRED_HOOKS if not (hooks / name).is_file()]
        assert not missing, (
            f"git hooks {missing} are not installed in {hooks}, so nothing scans the blobs a commit "
            f"would record: the staged-blob privacy scan and the commit-message scan are both off, "
            f"and the privacy checks are not armed. Run `make hooks`."
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

    def test_a_clean_commit_runs_through_the_installed_wrappers(self, tmp_path: Path) -> None:
        repo = _repo_with_copies_of_installed_hooks(tmp_path)
        (repo / "clean.txt").write_text("ordinary content\n", encoding="utf-8")
        must_git(repo, "add", "clean.txt")

        result = run_git(repo, "commit", "-q", "-m", "clean")

        assert result.returncode == 0, result.stderr

    def test_the_installed_pre_commit_wrapper_rejects_a_staged_detector_hit(
        self, tmp_path: Path
    ) -> None:
        repo = _repo_with_copies_of_installed_hooks(tmp_path)
        head_before = must_git(repo, "rev-parse", "HEAD").strip()
        test_access_key = "AKIA" + "ABCDEFGHIJKLMNOP"
        (repo / "notes.txt").write_text(f"key={test_access_key}\n", encoding="utf-8")
        must_git(repo, "add", "notes.txt")

        result = run_git(repo, "commit", "-q", "-m", "staged hit")

        assert result.returncode != 0, "the installed pre-commit wrapper admitted a detector hit"
        assert "aws_access_key_id" in result.stderr
        head_after = must_git(repo, "rev-parse", "HEAD").strip()
        assert head_after == head_before

    def test_the_installed_commit_message_wrapper_scans_the_draft(self, tmp_path: Path) -> None:
        repo = _repo_with_copies_of_installed_hooks(tmp_path)
        head_before = must_git(repo, "rev-parse", "HEAD").strip()
        (repo / "README.md").write_text("ordinary content revised\n", encoding="utf-8")
        must_git(repo, "add", "README.md")
        test_access_key = "AKIA" + "ABCDEFGHIJKLMNOP"

        result = run_git(repo, "commit", "-q", "-m", f"record key {test_access_key}")

        assert result.returncode != 0, "the installed commit-msg wrapper admitted a detector hit"
        assert "aws_access_key_id" in result.stderr
        head_after = must_git(repo, "rev-parse", "HEAD").strip()
        assert head_after == head_before
