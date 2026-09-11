"""Compile every Python file under `docs/scratch/` with syntax warnings escalated to errors.

Nothing else in this repo looks at that tree at all, which is the reason this file exists. `make
lint` misses it because ruff respects .gitignore and `docs/scratch/` is gitignored; `make
typecheck` misses it because [tool.basedpyright] in pyproject.toml excludes the path by name. So
the one part of the repo carrying most of the analysis code that produces the project's numbers had
no checker of any kind over it, while `scripts/` -- five files -- had two.

Syntax specifically, rather than a lint or a type check, because that is the tier where the
exclusion above is not a judgement call. basedpyright's exclusion is deliberate and correct: a
scratch script is a throwaway and must not be able to turn a shared gate red over style or types.
"This file will not parse" is different. It is not a matter of taste and not a style opinion, and a
file that does not parse is not a throwaway that happens to be scruffy -- it is a file that cannot
run, which for analysis code means a number nobody can reproduce.

Escalating SyntaxWarning is what makes this more than a parse check, and the escalation is doing
real work rather than decorating: the class of defect it catches is *currently silent*. An
unrecognised escape (`\\%` written into a non-raw string, the usual way a regex or a LaTeX fragment
gets typed) compiles clean on 3.13 and only emits a warning nobody reads, and becomes a hard
SyntaxError on 3.14. Same for `is` against a literal and an assertion on a tuple. Those run today
and change meaning or stop running on a future interpreter, which is exactly the shape of the four
bugs this repo's central rule is named after -- green output, quietly doing nothing.
`TestTheSyntaxWarningEscalationIsLoadBearing` keeps that claim honest by asserting both halves.

Compilation only, never import. `compile` parses and emits bytecode and stops, so an analysis
module that would build an AWS client, read a checkpoint or take the GPU at import time does none
of it here. That is what makes sweeping several hundred untrusted local files in-process safe.

In-process rather than one subprocess per file, and rather than `compileall`. All three work; the
in-process sweep is 0.5 seconds against 16.3 seconds for `py_compile` in a subprocess per file
(measured over 415 files on this box), it writes no `.pyc` into the scratch tree the way
`compileall` does unless separately redirected, and it yields the exception object per file so a
failure names the file, the line and the reason instead of parsing a tool's stdout.

Two known limits, stated rather than papered over. A fresh clone or worktree holds only the
tracked scratch README -- the directory itself exists in every checkout, so bare existence cannot
distinguish a working box from a fresh one -- and the sweep skips there; `-ra` in addopts prints
that skip reason on every run, which is what keeps a vacuous run visible rather than green. And if
the scratch tree is ever *moved* rather than fresh, that skip is indistinguishable from a fresh
checkout and this gate silently stops covering anything. The path is pinned by convention
(CLAUDE.md: one scratch location, no second one), not by a check.
"""

import logging
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = REPO_ROOT / "docs" / "scratch"

# What every checkout's scratch tree holds: exactly the tracked README. The skip below keys on
# this rather than on the directory existing, because the README is tracked and so the directory
# exists in EVERY checkout -- the original `not SCRATCH_ROOT.is_dir()` condition was dead code,
# and a fresh clone or worktree ran the sweep over zero files and failed `make test` out of the
# box. Tracking anything else under docs/scratch would be a privacy-rule violation before it was
# a staleness problem here, so pinning the one name is safe.
_FRESH_CHECKOUT_CONTENTS = frozenset({"README.md"})


def _scratch_is_a_fresh_checkout() -> bool:
    """Say whether the scratch tree holds nothing beyond what git puts in every checkout.

    True also when the directory is missing entirely (a clone old enough to predate the tracked
    README): there is nothing to sweep either way. A box with real scratch material but no `.py`
    files does NOT count as fresh -- the sweep runs there and `test_the_sweep_has_files_to_check`
    is the alarm that the tree moved or the glob broke, exactly as before.
    """
    if not SCRATCH_ROOT.is_dir():
        return True
    return {entry.name for entry in SCRATCH_ROOT.iterdir()} <= _FRESH_CHECKOUT_CONTENTS


logger = logging.getLogger(__name__)

# Doubled backslash: this file's own source stays clean and the string it builds is the dirty one.
SOURCE_WITH_AN_INVALID_ESCAPE = 'unrecognised_escape = "0|\\%|"\n'


@pytest.fixture(scope="session")
def scratch_python_files() -> list[Path]:
    """Every `.py` file under the scratch tree, sorted so a failure list reads the same each run.

    `rglob` does not descend into symlinked directories on 3.13 and `is_file()` drops a dangling
    link, so neither a venv symlinked into the tree nor a broken link can derail the sweep. The
    cost is that real code reached only through a directory symlink would go unswept; nothing in
    the tree is arranged that way today.
    """
    return sorted(path for path in SCRATCH_ROOT.rglob("*.py") if path.is_file())


def _compile_failure(path: Path) -> str | None:
    """Compile one file: `None` if it is fine, else the line and reason it will not compile.

    This is where the escalation lives, and it lives here rather than in the caller's loop so that
    the self-tests below can drive the real code path. A `catch_warnings` block per file is the
    equivalent of running the interpreter with `-W error::SyntaxWarning`, and it also displaces
    whatever warning filters pytest's own configuration installed for the rest of the suite.

    Only `SyntaxError` is caught, and that is a measured choice rather than a narrow guess: every
    way a file was found to fail compilation on this interpreter arrives as `SyntaxError`, including
    the ones whose names suggest otherwise -- an escalated SyntaxWarning, a null byte, invalid
    UTF-8, an unknown coding cookie, a truncated string literal. Anything else propagates and fails
    the test loudly instead of being collected into a tidy report line.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        try:
            compile(path.read_bytes(), str(path), "exec")
        except SyntaxError as exc:
            return f"{exc.lineno}: {type(exc).__name__}: {exc.msg}"
    return None


def _compile_failures(paths: list[Path]) -> list[str]:
    """One report line per file that will not compile, each named by its repo-relative path."""
    failures: list[str] = []
    for path in paths:
        reason = _compile_failure(path)
        if reason is not None:
            failures.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{reason}")
    return failures


class TestTheSyntaxWarningEscalationIsLoadBearing:
    """The gate's own sabotage, kept in the suite instead of run once and remembered.

    Three claims, each failing for its own reason. The baseline shows the violation sails through a
    plain parse check, so the escalation is buying something rather than decorating. The rejection
    goes through `_compile_failure` rather than a private copy of it, which is the whole point: swap
    that function's `"error"` for `"ignore"` and this test is what goes red, where a self-test
    holding its own filter would have kept reporting success over a sweep that had stopped
    escalating. The acceptance covers the opposite vacuity, a checker that rejects everything and
    so distinguishes nothing.
    """

    def test_the_violation_sails_through_a_plain_parse_check(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(SOURCE_WITH_AN_INVALID_ESCAPE, "<escalation-baseline>", "exec")

    def test_the_sweep_rejects_the_violation(self, tmp_path: Path) -> None:
        planted = tmp_path / "carries_an_invalid_escape.py"
        planted.write_text(SOURCE_WITH_AN_INVALID_ESCAPE, encoding="utf-8")

        reason = _compile_failure(planted)

        assert reason is not None, (
            "the sweep accepted a file the interpreter warns about, so it has degraded into a "
            "parse check: SyntaxWarning is no longer escalated inside _compile_failure"
        )
        assert "invalid escape sequence" in reason, reason

    def test_the_sweep_accepts_clean_source(self, tmp_path: Path) -> None:
        clean = tmp_path / "carries_nothing.py"
        clean.write_text('recognised_escape = "0|\\\\%|"\n', encoding="utf-8")

        assert _compile_failure(clean) is None


@pytest.mark.skipif(
    _scratch_is_a_fresh_checkout(),
    reason=(
        "docs/scratch holds only the tracked README here (a fresh clone or worktree), so there is "
        "no local analysis code to sweep. -ra in addopts prints this on every run, so a sweep "
        "that covers nothing stays visible."
    ),
)
class TestEveryScratchPythonFileCompiles:
    """The sweep itself, over whatever the local scratch tree happens to hold.

    Deliberately asymmetric with the other gates: this one checks untracked local files, so what it
    sees differs box to box and a fresh clone sees nothing. That is the same reason basedpyright
    excludes the tree, and it is acceptable here only because the check is "will this parse" -- a
    scratch file that fails it is broken on any box, not merely unfashionable on this one.

    No path exemptions, on purpose. The tree is shared by many concurrent sessions and a downloaded
    reference that arrives mangled will break this; the fix is to repair or rename that download,
    which is a two-minute job. An exemption mechanism would be a second hole, and routing an
    exclusion through an edit to this tracked file keeps it visible in a commit rather than hidden
    in an untracked marker.
    """

    def test_the_sweep_has_files_to_check(self, scratch_python_files: list[Path]) -> None:
        assert scratch_python_files, (
            "docs/scratch exists but holds no .py files, so this sweep covered nothing and passed. "
            "Either the scratch tree moved, in which case repoint SCRATCH_ROOT rather than leave a "
            "check that succeeds by reading an empty directory, or the glob pattern broke."
        )

    def test_they_all_compile_with_syntax_warnings_as_errors(
        self, scratch_python_files: list[Path]
    ) -> None:
        failures = _compile_failures(scratch_python_files)
        logger.info(
            f"scratch compile sweep: {len(scratch_python_files)} files under docs/scratch, "
            f"{len(failures)} failing, no path exemptions"
        )
        report = "\n  ".join(failures)
        assert not failures, (
            f"{len(failures)} of {len(scratch_python_files)} Python files under docs/scratch will "
            f"not compile with SyntaxWarning escalated to an error:\n  {report}"
        )
