"""Pin the one place every games artifact answers "which code produced this?".

The invariant with teeth is that provenance never raises: it is metadata *about* work, so losing
the ability to name a commit must not throw away the work itself. That is exercised by asking the
module about a directory that is not a git checkout at all, which is what a container without a
git tree looks like -- the failure has to come back inside the returned string, still visibly a
failure, rather than as an exception out of a training run.

The second invariant is that "the tree was clean" and "nobody could tell" stay distinguishable.
Collapsing them to False would make the reassuring answer the default, and a record naming only a
commit is indistinguishable from one produced by edited code.
"""

from __future__ import annotations

import functools
import subprocess
from typing import TYPE_CHECKING

from games.provenance import UNKNOWN_SHA, git_provenance, git_sha, git_tree_dirty

if TYPE_CHECKING:
    from pathlib import Path

import pytest


def _git(*args: str, cwd: Path) -> None:
    """Run one git command in a throwaway checkout."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)  # noqa: S603, S607


class TestTheBakedInShaWins:
    """A Batch image bakes GIT_SHA in and carries no checkout, so the environment has to win."""

    def test_the_environment_variable_is_returned_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GAMES_TEST_SHA", "0123456789abcdef")
        assert git_sha(env_var="GAMES_TEST_SHA") == "0123456789abcdef"

    def test_an_empty_variable_is_not_a_sha(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset-looking value must fall through to git, not be recorded as the commit."""
        monkeypatch.setenv("GAMES_TEST_SHA", "")
        assert git_sha(env_var="GAMES_TEST_SHA") != ""

    def test_the_repo_is_asked_when_nothing_was_baked_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GAMES_TEST_SHA", raising=False)
        resolved = git_sha(env_var="GAMES_TEST_SHA")
        assert len(resolved) == 40
        assert resolved.isalnum()


class TestProvenanceNeverRaises:
    """A missing commit must not stop a run, and must not read as a real commit either."""

    def test_a_directory_that_is_not_a_checkout_yields_a_visible_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("games.provenance.REPO_ROOT", tmp_path)
        monkeypatch.delenv("GAMES_TEST_SHA", raising=False)
        resolved = git_sha(env_var="GAMES_TEST_SHA")
        assert resolved.startswith(UNKNOWN_SHA)
        assert "git rev-parse failed" in resolved

    def test_an_unanswerable_dirty_question_is_none_rather_than_clean(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("games.provenance.REPO_ROOT", tmp_path)
        assert git_tree_dirty() is None

    def test_a_real_checkout_answers_the_dirty_question(self) -> None:
        assert git_tree_dirty() in {True, False}


class TestTheDirtyFlagTracksTheTreeItWasAskedAbout:
    """Against a real throwaway checkout, so the three states are the ones git actually reports.

    `git_tree_dirty() in {True, False}` cannot tell a working flag from one wired to a constant:
    both answers are in the set. These build a repository, commit, then dirty it.
    """

    @pytest.fixture
    def checkout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        run = functools.partial(_git, cwd=tmp_path)
        run("init", "--quiet")
        run("config", "user.email", "tests@example.invalid")
        run("config", "user.name", "Tests")
        (tmp_path / "tracked.txt").write_text("first\n")
        run("add", "tracked.txt")
        run("commit", "--quiet", "-m", "first")
        monkeypatch.setattr("games.provenance.REPO_ROOT", tmp_path)
        return tmp_path

    def test_a_committed_tree_is_clean(self, checkout: Path) -> None:
        del checkout
        assert git_tree_dirty() is False

    def test_an_uncommitted_edit_is_dirty(self, checkout: Path) -> None:
        (checkout / "tracked.txt").write_text("edited but never committed\n")
        assert git_tree_dirty() is True

    def test_an_untracked_file_is_dirty_too(self, checkout: Path) -> None:
        """A run launched with a new, uncommitted module is not reproducible from its commit."""
        (checkout / "probe.py").write_text("print('probe')\n")
        assert git_tree_dirty() is True


class TestEveryArtifactGetsBothFields:
    def test_the_record_carries_the_sha_and_the_dirty_flag(self) -> None:
        """Both, always: a record naming only a commit cannot say the code was edited."""
        record = git_provenance()
        assert set(record) == {"git_sha", "git_tree_dirty"}
        assert record["git_sha"]
        assert record["git_tree_dirty"] in {True, False, None}
