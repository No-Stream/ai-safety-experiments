"""``make test-changed`` selects by directory, runs nothing when nothing changed, and mirrors ``make test``.

The mapping is pure and pinned on fixture diffs; the git listing is pinned on a throwaway repository
so a staged edit, an unstaged edit, an untracked file and a deletion are all seen; the "nothing
changed" path is pinned on a spy, because the failure it guards against -- an empty selection handed
to pytest, which then collects the WHOLE suite -- is exactly the one that would look like success.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "test_changed.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_changed_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


test_changed = _load_script()


def always_exists(_path: str) -> bool:
    return True


class TestTheMappingIsByDirectory:
    def test_a_changed_test_file_selects_itself(self) -> None:
        selected = test_changed.select_test_paths(
            ["reward_hacking/tests/test_jagged_runner.py"], exists=always_exists
        )
        assert selected == ["reward_hacking/tests/test_jagged_runner.py"]

    def test_a_changed_source_file_selects_its_components_root(self) -> None:
        assert test_changed.select_test_paths(
            ["reward_hacking/jagged/runner.py"], exists=always_exists
        ) == ["reward_hacking/tests"]
        assert test_changed.select_test_paths(["games/train.py"], exists=always_exists) == [
            "games/tests"
        ]
        assert test_changed.select_test_paths(["sociology/records.py"], exists=always_exists) == [
            "sociology/tests"
        ]

    def test_a_conftest_or_fixture_change_selects_the_whole_root(self) -> None:
        assert test_changed.select_test_paths(
            ["reward_hacking/tests/conftest.py"], exists=always_exists
        ) == ["reward_hacking/tests"]
        assert test_changed.select_test_paths(
            ["games/tests/data/fixture.json", "sociology/tests/streaming_stubs.py"],
            exists=always_exists,
        ) == ["games/tests", "sociology/tests"]

    def test_a_deleted_test_file_selects_nothing_for_itself(self) -> None:
        selected = test_changed.select_test_paths(
            ["games/tests/test_gone.py"], exists=lambda _path: False
        )
        assert selected == []

    def test_a_test_file_under_a_selected_root_collapses_into_the_root(self) -> None:
        """SABOTAGE target: handing pytest both the root and a file in it, which collects it twice."""
        selected = test_changed.select_test_paths(
            ["games/tests/test_games_train.py", "games/train.py"], exists=always_exists
        )
        assert selected == ["games/tests"]

    def test_the_shared_substrate_selects_every_root_that_imports_it(self) -> None:
        assert test_changed.select_test_paths(["grpo/throughput.py"], exists=always_exists) == [
            "games/tests",
            "reward_hacking/tests",
            "tests",
        ]

    def test_everything_outside_a_component_selects_the_repo_level_suite(self) -> None:
        for path in (
            "Makefile",
            "pyproject.toml",
            "scripts/idle_watchdog.sh",
            "cloud/entrypoint.sh",
        ):
            assert test_changed.select_test_paths([path], exists=always_exists) == ["tests"], path
        assert test_changed.select_test_paths(
            ["docs/scratch/a-note.md", "README.md"], exists=always_exists
        ) == ["tests"]

    def test_legacy_selects_nothing(self) -> None:
        assert (
            test_changed.select_test_paths(["legacy/pretraining.ipynb"], exists=always_exists) == []
        )

    def test_nothing_changed_selects_nothing(self) -> None:
        assert test_changed.select_test_paths([], exists=always_exists) == []


def _git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 - repo tooling, literal arguments
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],  # noqa: S607 - the caller's PATH git
        check=True,
        capture_output=True,
    )


@pytest.fixture
def fixture_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway repository with the four roots and one commit, isolated from this machine's git config."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "repo"
    for relative in (
        "reward_hacking/module.py",
        "reward_hacking/tests/test_module.py",
        "games/tests/test_a.py",
        "tests/test_b.py",
        "sociology/tests/test_c.py",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "fixture")
    return repo


class TestTheGitListingSeesTheWholeWorkingTree:
    def test_a_clean_tree_lists_nothing(self, fixture_repo: Path) -> None:
        assert test_changed.changed_paths(fixture_repo) == []

    def test_unstaged_staged_untracked_and_deleted_are_all_listed(self, fixture_repo: Path) -> None:
        """SABOTAGE target: ``git diff`` without HEAD (misses staged) or without the untracked listing."""
        (fixture_repo / "reward_hacking/module.py").write_text("# edited\n", encoding="utf-8")
        (fixture_repo / "tests/test_b.py").write_text("# staged\n", encoding="utf-8")
        _git(fixture_repo, "add", "tests/test_b.py")
        (fixture_repo / "sociology/new.py").write_text("# untracked\n", encoding="utf-8")
        (fixture_repo / "games/tests/test_a.py").unlink()
        assert test_changed.changed_paths(fixture_repo) == [
            "games/tests/test_a.py",
            "reward_hacking/module.py",
            "sociology/new.py",
            "tests/test_b.py",
        ]

    def test_a_staged_rename_lists_its_source_and_its_destination(self, fixture_repo: Path) -> None:
        """SABOTAGE target: ``git diff`` with rename detection left on.

        Git folds a move into one entry listed under its destination, so a module moved from
        ``reward_hacking/`` to ``sociology/`` would select ``sociology/tests`` alone and the tests
        of the component that lost the module would never run. The same reason AGENTS.md's
        composed-tree recipe insists on ``--no-renames``.
        """
        _git(fixture_repo, "mv", "reward_hacking/module.py", "sociology/module.py")
        changed = test_changed.changed_paths(fixture_repo)
        assert changed == ["reward_hacking/module.py", "sociology/module.py"]
        assert test_changed.select_test_paths(
            changed, exists=lambda path: (fixture_repo / path).exists()
        ) == ["reward_hacking/tests", "sociology/tests"]

    def test_the_end_to_end_selection_on_that_diff(
        self, fixture_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (fixture_repo / "reward_hacking/module.py").write_text("# edited\n", encoding="utf-8")
        (fixture_repo / "tests/test_b.py").write_text("# staged\n", encoding="utf-8")
        _git(fixture_repo, "add", "tests/test_b.py")
        (fixture_repo / "sociology/new.py").write_text("# untracked\n", encoding="utf-8")
        (fixture_repo / "games/tests/test_a.py").unlink()
        assert test_changed.main(["--list", "--repo-root", str(fixture_repo)]) == 0
        assert capsys.readouterr().out.splitlines() == [
            "reward_hacking/tests",
            "sociology/tests",
            "tests/test_b.py",
        ]


class TestNothingChangedRunsNothing:
    def test_no_pytest_process_is_started_on_a_clean_tree(
        self, fixture_repo: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SABOTAGE target: an empty selection reaching ``pytest``, which then runs the whole suite."""
        started: list[list[str]] = []

        def spy(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            started.append(command)
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(test_changed.subprocess, "run", spy)
        # changed_paths goes through the same subprocess.run, so it is answered directly here.
        monkeypatch.setattr(test_changed, "changed_paths", lambda _root: [])
        with caplog.at_level(logging.INFO):
            assert test_changed.main(["--repo-root", str(fixture_repo)]) == 0
        assert started == []
        assert "nothing changed against HEAD" in caplog.text

    def test_a_change_does_start_exactly_one_pytest_process(
        self, fixture_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The positive control for the spy above."""
        started: list[list[str]] = []

        def spy(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            started.append(command)
            return subprocess.CompletedProcess(command, 3)

        monkeypatch.setattr(test_changed.subprocess, "run", spy)
        monkeypatch.setattr(test_changed, "changed_paths", lambda _root: ["sociology/new.py"])
        assert test_changed.main(["--repo-root", str(fixture_repo), "--workers", "8"]) == 3
        assert started == [[sys.executable, "-m", "pytest", "sociology/tests"]], (
            "one test file in the root, so no xdist flags"
        )


class TestTheCommandMirrorsMakeTest:
    def test_one_file_runs_in_process(self) -> None:
        command = test_changed.pytest_command(["tests/test_b.py"], workers=8, test_files=1)
        assert command == [sys.executable, "-m", "pytest", "tests/test_b.py"]

    def test_workers_are_capped_at_the_file_count(self) -> None:
        command = test_changed.pytest_command(["a.py", "b.py"], workers=8, test_files=2)
        assert command[3:7] == ["-n", "2", "--dist", "loadfile"]

    def test_a_wide_selection_uses_every_worker(self) -> None:
        command = test_changed.pytest_command(["games/tests"], workers=8, test_files=80)
        assert command[3:7] == ["-n", "8", "--dist", "loadfile"]

    def test_the_file_count_reads_the_tree(self, fixture_repo: Path) -> None:
        assert (
            test_changed.count_test_files(["reward_hacking/tests", "tests/test_b.py"], fixture_repo)
            == 2
        )


class TestTheSharedSubstrateTableMatchesTheTree:
    def test_every_root_whose_code_imports_grpo_is_listed(self) -> None:
        """The table is hand-written; the tree is the truth. Red when a component starts importing grpo."""
        importing_roots: set[str] = set()
        candidates = {
            "tests": ("tests",),
            "games/tests": ("games",),
            "reward_hacking/tests": ("reward_hacking",),
            "sociology/tests": ("sociology",),
        }
        for root, package_dirs in candidates.items():
            for package_dir in package_dirs:
                for path in (REPO_ROOT / package_dir).rglob("*.py"):
                    text = path.read_text(encoding="utf-8")
                    if any(
                        line.startswith(("from grpo", "import grpo")) for line in text.splitlines()
                    ):
                        importing_roots.add(root)
                        break
        assert importing_roots == set(test_changed.SHARED_SUBSTRATE_TEST_ROOTS["grpo"])

    def test_the_roots_match_pyproject_testpaths(self) -> None:
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for root in test_changed.TEST_ROOTS:
            assert f'"{root}"' in text, root
