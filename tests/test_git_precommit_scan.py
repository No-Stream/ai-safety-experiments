"""The staged-blob privacy hook, exercised through real ``git commit`` attempts.

Everything here runs in a throwaway repo armed with a SYNTHETIC instrument stem -- invented
nonsense, resembling no real item -- because the real arming sources are gitignored and a
fresh clone must be able to run this suite. The scenarios mirror the 2026-08-22 incident:
the leak rode the INDEX while every file on disk scanned clean, so the load-bearing test is
the one where disk and index disagree.

The hooks are installed by running scripts/install_git_hooks.sh itself, so the wrappers'
interpreter fallback (no .venv in the throwaway repo -> bare ``python3``, the system 3.9)
is part of what runs here, not a mock of it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_STEM = "the violet quorum shuffles brass umbrellas before dawn audits"


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed git binary in a throwaway test repo
        ("git", *args),  # noqa: S607 - the caller's PATH git, same one the hooks run under
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def must_git(repo: Path, *args: str) -> str:
    result = run_git(repo, *args)
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout


def _bare_hook_repo(tmp_path: Path) -> Path:
    """A fresh repo with the hooks installed and one clean commit, but nothing armed yet."""
    repo = tmp_path / "repo"
    repo.mkdir()
    must_git(repo, "init", "-q")
    must_git(repo, "config", "user.email", "hook-test@example.com")
    must_git(repo, "config", "user.name", "hook test")
    # HARNESS-SCAN-EXEMPT-multiline-comment-block -- hermetic against any global hooksPath;
    # absolute, because a relative hooksPath resolves against the working tree the hook runs
    # in, which would silently disable hooks in the linked-worktree tests below.
    must_git(repo, "config", "core.hooksPath", str(repo / ".git" / "hooks"))

    scripts_dir = repo / "scripts"
    scripts_dir.mkdir()
    for script in ("scan_secrets.py", "git_precommit_scan.py", "install_git_hooks.sh"):
        shutil.copy(REPO_ROOT / "scripts" / script, scripts_dir / script)
    subprocess.run(  # noqa: S603 - repo-local installer script
        ("sh", str(scripts_dir / "install_git_hooks.sh")),  # noqa: S607 - PATH sh, like git itself
        cwd=repo,
        capture_output=True,
        check=True,
    )

    (repo / "README.md").write_text("seed file\n", encoding="utf-8")
    must_git(repo, "add", "README.md")
    must_git(repo, "commit", "-q", "-m", "seed")
    return repo


@pytest.fixture
def unarmed_repo(tmp_path: Path) -> Path:
    return _bare_hook_repo(tmp_path)


@pytest.fixture
def armed_repo(tmp_path: Path) -> Path:
    """The bare repo plus a synthetic local item source, so the instrument detector is live."""
    repo = _bare_hook_repo(tmp_path)
    data_dir = repo / "games" / "data" / "survey"
    data_dir.mkdir(parents=True)
    (data_dir / "synthetic.json").write_text(
        json.dumps({"items": [{"stem": SYNTHETIC_STEM}]}), encoding="utf-8"
    )
    return repo


def attempt_commit(repo: Path, message: str = "attempt") -> subprocess.CompletedProcess[str]:
    return run_git(repo, "commit", "-q", "-m", message)


class TestThePreCommitHookScansStagedBlobs:
    def test_a_staged_blob_carrying_item_text_refuses_the_commit(self, armed_repo: Path) -> None:
        head_before = must_git(armed_repo, "rev-parse", "HEAD")
        (armed_repo / "notes.txt").write_text(
            f"analysis scratch quoting an item: {SYNTHETIC_STEM}\n", encoding="utf-8"
        )
        must_git(armed_repo, "add", "notes.txt")

        result = attempt_commit(armed_repo)

        assert result.returncode != 0, "the commit went through carrying instrument text"
        assert "instrument_item_text" in result.stderr
        assert SYNTHETIC_STEM not in result.stderr, "the refusal republished the matched text"
        assert must_git(armed_repo, "rev-parse", "HEAD") == head_before

    def test_the_hook_reads_the_index_not_the_disk(self, armed_repo: Path) -> None:
        """The incident shape: disk scans clean while the staged blob is dirty."""
        target = armed_repo / "notes.txt"
        target.write_text(f"pre-strip draft: {SYNTHETIC_STEM}\n", encoding="utf-8")
        must_git(armed_repo, "add", "notes.txt")
        target.write_text(
            "post-strip draft: item text lives in local data only\n", encoding="utf-8"
        )

        result = attempt_commit(armed_repo)

        assert result.returncode != 0, "a clean DISK file let a dirty INDEX blob through"
        assert "instrument_item_text" in result.stderr

        must_git(armed_repo, "add", "notes.txt")
        assert attempt_commit(armed_repo).returncode == 0, "re-adding the clean file should pass"

    def test_a_clean_commit_passes(self, armed_repo: Path) -> None:
        (armed_repo / "README.md").write_text("seed file, revised\n", encoding="utf-8")
        must_git(armed_repo, "add", "README.md")

        result = attempt_commit(armed_repo)

        assert result.returncode == 0, f"clean commit refused: {result.stderr}"

    def test_staging_an_arming_source_file_is_refused(self, armed_repo: Path) -> None:
        must_git(armed_repo, "add", "games/data/survey/synthetic.json")

        result = attempt_commit(armed_repo)

        assert result.returncode != 0, "committing the item source itself went through"
        assert "instrument_source_file_staged" in result.stderr

    def test_without_arming_sources_the_hook_warns_and_passes(self, unarmed_repo: Path) -> None:
        (unarmed_repo / "clean.txt").write_text("nothing sensitive here\n", encoding="utf-8")
        must_git(unarmed_repo, "add", "clean.txt")

        result = attempt_commit(unarmed_repo)

        assert result.returncode == 0, f"a fresh clone must not be bricked: {result.stderr}"
        assert "INERT" in result.stderr, "the unarmed state must be loud, not silent"


class TestWorktreesLackingTheScannerStillGetScanned:
    """A linked worktree checked out before the gate landed carries no scanner in its tree.

    The hook must fall back to the MAIN worktree's copy and scan anyway; it fails closed
    (refusing the commit) only when no copy is reachable anywhere. The fixture repos
    reproduce the pre-gate shape naturally: scripts/ exists on the main tree's disk but
    was never committed, so a worktree of HEAD has no scripts/ at all.
    """

    def _pre_gate_worktree(self, repo: Path, tmp_path: Path) -> Path:
        worktree = tmp_path / "pre-gate-worktree"
        must_git(repo, "worktree", "add", "--detach", str(worktree), "HEAD")
        assert not (worktree / "scripts").exists(), "worktree unexpectedly carries the scanner"
        return worktree

    def test_a_dirty_blob_commits_from_the_worktree_are_refused(
        self, armed_repo: Path, tmp_path: Path
    ) -> None:
        worktree = self._pre_gate_worktree(armed_repo, tmp_path)
        (worktree / "notes.txt").write_text(
            f"scratch quoting an item: {SYNTHETIC_STEM}\n", encoding="utf-8"
        )
        must_git(worktree, "add", "notes.txt")

        result = attempt_commit(worktree)

        assert result.returncode != 0, "a worktree predating the gate committed UNSCANNED"
        assert "instrument_item_text" in result.stderr
        assert "WITHOUT a scan" not in result.stderr, "the fail-open branch is back"

    def test_a_clean_commit_from_the_worktree_passes(
        self, armed_repo: Path, tmp_path: Path
    ) -> None:
        worktree = self._pre_gate_worktree(armed_repo, tmp_path)
        (worktree / "clean.txt").write_text("nothing sensitive here\n", encoding="utf-8")
        must_git(worktree, "add", "clean.txt")

        result = attempt_commit(worktree)

        assert result.returncode == 0, f"clean worktree commit refused: {result.stderr}"

    def test_no_scanner_reachable_anywhere_refuses_rather_than_passing(
        self, armed_repo: Path, tmp_path: Path
    ) -> None:
        worktree = self._pre_gate_worktree(armed_repo, tmp_path)
        (armed_repo / "scripts" / "git_precommit_scan.py").unlink()
        (worktree / "clean.txt").write_text("nothing sensitive here\n", encoding="utf-8")
        must_git(worktree, "add", "clean.txt")

        result = attempt_commit(worktree)

        assert result.returncode != 0, "no scanner reachable and the commit STILL went through"
        assert "REFUSING the commit rather than skipping the scan" in result.stderr


class TestTheCommitMsgHookScansTheMessage:
    def test_a_message_carrying_item_text_is_refused(self, armed_repo: Path) -> None:
        head_before = must_git(armed_repo, "rev-parse", "HEAD")
        (armed_repo / "README.md").write_text("seed file, revised again\n", encoding="utf-8")
        must_git(armed_repo, "add", "README.md")

        result = attempt_commit(armed_repo, message=f"record the finding: {SYNTHETIC_STEM}")

        assert result.returncode != 0, "a commit message carrying instrument text went through"
        assert "instrument_item_text" in result.stderr
        assert must_git(armed_repo, "rev-parse", "HEAD") == head_before
