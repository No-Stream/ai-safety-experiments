# ruff: noqa: INP001 -- scripts/ is a directory of entry points, not a package; pyproject.toml carries
# the same ignore per sibling script, kept inline here so adding this script touched one file.
"""Run only the test files that the working tree's changes against HEAD can reach.

The fast iteration gate behind ``make test-changed`` (hot-path backlog rank 58, decision C7).
``make ci`` stays the commit gate, and the reason is the shape of this mapping: it goes by
directory, not by import graph, so an edit to ``reward_hacking/model_backend.py`` runs
``reward_hacking/tests`` and none of the fourteen ``sociology`` modules that import it. A partial
gate that agents start treating as sufficient is how a cross-component break ships, which is why
``testpaths`` in ``pyproject.toml`` lists every root and why this target never replaces the full
run before a commit.

The mapping, in :func:`select_test_paths`:

- a changed test file (``test_*.py`` directly under one of the four roots) selects itself;
- any other change under a test root -- ``conftest.py``, a stub module, fixture data -- selects the
  whole root, because every test in it may read it;
- a change under ``games/``, ``reward_hacking/`` or ``sociology/`` selects that component's root;
- a change under ``grpo/`` selects every root whose tests import it (``tests/``, ``games/tests``,
  ``reward_hacking/tests``), since it is the training substrate the other two build on;
- everything else -- ``scripts/``, ``cloud/``, ``Makefile``, ``pyproject.toml``, the docs -- selects
  ``tests/``, the repo-level suite, which is where the doc-link, interpreter-compatibility,
  ship-tree and scratch-compiles checks live;
- ``legacy/`` selects nothing: closed records with no tests.

"Changed" is ``git diff --name-only --no-renames HEAD`` (staged and unstaged, deletions included)
plus ``git ls-files --others --exclude-standard`` (untracked, ``.gitignore`` honoured), the same
listing the privacy scan uses. ``--no-renames`` because git otherwise folds a move into one entry
listed under its destination, and the component that lost the module would never have its tests
run. A deleted test file cannot run and is dropped; a deleted source file still selects its root.
With nothing changed the target runs nothing and says so, exit 0.

The run mirrors ``make test``: the Makefile sets ``RLVR_SMOKE=1`` around it, and pytest is sharded
with xdist's ``--dist loadfile`` over ``--workers`` processes, capped at the number of selected
test files so a two-file selection does not start eight workers to idle six of them.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

TEST_ROOTS: tuple[str, ...] = ("tests", "games/tests", "reward_hacking/tests", "sociology/tests")
"""The four pytest roots, the same list ``testpaths`` carries in pyproject.toml."""

COMPONENT_TEST_ROOTS: dict[str, str] = {
    "games": "games/tests",
    "reward_hacking": "reward_hacking/tests",
    "sociology": "sociology/tests",
}
"""Top-level component directory to the root that tests it."""

SHARED_SUBSTRATE_TEST_ROOTS: dict[str, tuple[str, ...]] = {
    "grpo": ("tests", "games/tests", "reward_hacking/tests"),
}
"""Top-level directories imported by more than one component, to every root that imports them.

``tests/test_test_changed.py`` checks this table against the tree's actual ``grpo`` imports, so a
component that starts importing the substrate turns the test red rather than silently falling out
of the fast gate.
"""

UNTESTED_TOP_LEVEL: frozenset[str] = frozenset({"legacy"})
"""Top-level directories with no tests by design."""

REPO_LEVEL_TEST_ROOT = "tests"
"""Where everything that belongs to no component is tested."""

DEFAULT_WORKERS = 8
"""Mirrors the Makefile's ``TEST_WORKERS`` default; the Makefile passes its own value through."""


def changed_paths(repo_root: Path) -> list[str]:
    """List every path changed against HEAD or untracked, repo-relative and sorted."""
    tracked = _git(repo_root, "diff", "--name-only", "--no-renames", "-z", "HEAD")
    untracked = _git(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    return sorted({path for path in (tracked + untracked).split("\0") if path})


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - repo tooling, literal arguments
        ["git", "-C", str(repo_root), *args],  # noqa: S607 - the caller's PATH git, as make resolves it
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_root_of(path: PurePosixPath) -> str | None:
    """Name the test root ``path`` lies under, or None when it is not a test-tree path."""
    for root in TEST_ROOTS:
        if path.is_relative_to(root):
            return root
    return None


def is_test_file(path: PurePosixPath, root: str) -> bool:
    """Whether ``path`` is a test module directly under ``root`` (not a conftest, stub or fixture)."""
    return (
        path.parent == PurePosixPath(root)
        and path.name.startswith("test_")
        and path.suffix == ".py"
    )


def select_test_paths(changed: Iterable[str], *, exists: Callable[[str], bool]) -> list[str]:
    """Map changed paths to the test paths that can reach them (see the module docstring).

    ``exists`` answers whether a path is still on disk, so a deleted test file is dropped rather
    than handed to pytest as a missing argument. Pure otherwise, which is what makes the mapping
    testable on a fixture diff without a repository.
    """
    selected: set[str] = set()
    for raw in changed:
        path = PurePosixPath(raw)
        root = test_root_of(path)
        if root is not None:
            if not is_test_file(path, root):
                selected.add(root)
            elif exists(raw):
                selected.add(raw)
            continue
        top = path.parts[0]
        if top in UNTESTED_TOP_LEVEL:
            continue
        if top in SHARED_SUBSTRATE_TEST_ROOTS:
            selected.update(SHARED_SUBSTRATE_TEST_ROOTS[top])
            continue
        selected.add(COMPONENT_TEST_ROOTS.get(top, REPO_LEVEL_TEST_ROOT))
    # A file under a root that is itself selected would be collected twice.
    return sorted(
        path
        for path in selected
        if path in TEST_ROOTS or test_root_of(PurePosixPath(path)) not in selected
    )


def count_test_files(paths: Iterable[str], repo_root: Path) -> int:
    """Count the test modules the selection covers, one per file and ``test_*.py`` per root."""
    count = 0
    for path in paths:
        full = repo_root / path
        count += len(list(full.glob("test_*.py"))) if full.is_dir() else 1
    return count


def pytest_command(paths: Sequence[str], *, workers: int, test_files: int) -> list[str]:
    """Build the pytest invocation: xdist as ``make test`` runs it, capped at the file count.

    One test file runs in-process; xdist worker start-up costs more than it saves there, which is
    the same reason ``make test-select`` stays serial.
    """
    command = [sys.executable, "-m", "pytest"]
    shard = min(workers, test_files)
    if shard > 1:
        command += ["-n", str(shard), "--dist", "loadfile"]
    return [*command, *paths]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the test files the working tree's changes against HEAD can reach."
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the selected test paths, one per line, run nothing",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Select and run; exit 0 with a message and no pytest process when nothing changed."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    args = _parse_args(argv)
    repo_root: Path = args.repo_root
    changed = changed_paths(repo_root)
    selected = select_test_paths(changed, exists=lambda path: (repo_root / path).exists())
    if not selected:
        logger.info("nothing changed against HEAD, tracked or untracked; no tests to run")
        return 0
    if args.list:
        sys.stdout.write("\n".join(selected) + "\n")
        return 0
    test_files = count_test_files(selected, repo_root)
    command = pytest_command(selected, workers=args.workers, test_files=test_files)
    logger.info(
        "%d changed paths select %d test paths (%d test files): %s",
        len(changed),
        len(selected),
        test_files,
        " ".join(selected),
    )
    return subprocess.run(  # noqa: S603 - the pytest command built above, literal arguments
        command, cwd=repo_root, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
