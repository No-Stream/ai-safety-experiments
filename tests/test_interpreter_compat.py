"""Compile every script that runs outside the venv under the interpreter that actually runs it.

The question per script is only "would the interpreter that executes this file accept its syntax",
and it is worth a test because these break at *runtime* -- inside a jail, or in a preflight check
that guards a GPU run -- rather than in lint, so a construct the executing interpreter cannot parse
is easy to miss.

Two groups, because the scripts no longer share one interpreter:

  * The host's system interpreter, ``/usr/bin/python3``, 3.9.25 here, runs four of them:
      - ``scripts/gpu_preflight.py``       resource-limits.sh invokes it as plain ``python3``
      - ``scripts/canary_manifest.py``     ``make canary-check`` runs it under ``SYSTEM_PYTHON``
      - ``scripts/scan_secrets.py``        docs/episode-isolation.md documents it as ``python3``
      - ``scripts/git_precommit_scan.py``  the hook wrappers prefer ``.venv`` but fall back to
        bare ``python3`` when it is absent, and a hook that crashes on syntax refuses every commit
  * The interpreter the jail resolves runs ``scripts/jail_assertions.py``, on both sides of
    ``scripts/run_jail_tests.sh``: inside the jail, and again outside it for the negative control.

``jail_assertions.py`` changed groups when the jail stopped exposing whatever ``/usr/bin/python3``
happened to be. A 3.9 jail misgraded coding submissions: one using a 3.10+ builtin crashed inside
the jail, failed every test unit, and "fails everything" trivially contains any registered failing
set, so the grader read working code as carrying a planted flaw. The jail now resolves an
interpreter at or above its own floor and refuses to start without one, which is what
``TestTheJailInterpreterMeetsItsFloor`` pins -- the most load-bearing thing in this file, because a
floor quietly lowered brings that misgrading straight back and nothing else would notice.

The moved script keeps its conservative 3.9 dialect anyway, and pyproject.toml keeps exempting it
from ruff's pyupgrade rules: both checkers are directory-scoped with no per-file version override,
so ``scripts/`` follows its oldest occupant. That is a checker-configuration constraint rather than
a claim about where the file runs, and it costs nothing.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EPISODE_JAIL = REPO_ROOT / "scripts" / "episode_jail.sh"
SYSTEM_PYTHON = Path("/usr/bin/python3")

SYSTEM_PYTHON_SCRIPTS = [
    "scripts/gpu_preflight.py",
    "scripts/canary_manifest.py",
    "scripts/scan_secrets.py",
    "scripts/git_precommit_scan.py",
]

JAIL_PYTHON_SCRIPTS = ["scripts/jail_assertions.py"]

# Restated rather than read back from episode_jail.sh: a test that asks the thing it is pinning what
# the answer should be pins nothing. 3.12 is the dialect the submissions this repo grades are in.
MINIMUM_JAIL_PYTHON_MINOR = 12


def _ask_the_jail(flag: str) -> str:
    """Ask episode_jail.sh one question, or fail with whatever it wrote to stderr.

    A failure is not a host quirk to skip past the way a missing ``/usr/bin/python3`` is: the jail
    refuses to start without an interpreter past its floor, so nothing the harness grades runs at
    all, and the script's own stderr names the two ways to fix it.
    """
    result = subprocess.run(  # noqa: S603 - trusted repo script, literal argument
        [str(EPISODE_JAIL), flag],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"`{EPISODE_JAIL} {flag}` exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip()


@pytest.fixture(scope="session")
def jail_python() -> Path:
    """The absolute path of the interpreter the jail exposes, as episode_jail.sh resolves it."""
    return Path(_ask_the_jail("--print-jail-python"))


@pytest.fixture(scope="session")
def jail_python_floor() -> int:
    """The oldest interpreter minor version the jail will expose."""
    return int(_ask_the_jail("--print-python-floor"))


def _interpreter_version(interpreter: Path) -> tuple[int, int]:
    """Ask an interpreter for its own version, because a filename is not a version."""
    result = subprocess.run(  # noqa: S603 - resolved interpreter path, literal argument
        [str(interpreter), "-c", "import sys; print(*sys.version_info[:2])"],
        capture_output=True,
        text=True,
        check=True,
    )
    major, minor = result.stdout.split()
    return int(major), int(minor)


def _assert_compiles(interpreter: Path, script: str, tmp_path: Path) -> None:
    """Compile one repo script under one interpreter, naming both when it fails."""
    path = REPO_ROOT / script
    assert path.exists(), (
        f"{script} is listed in this module but does not exist. Fix the list rather than letting a "
        "rename silently disarm this guard."
    )

    # PYTHONPYCACHEPREFIX keeps the generated .pyc out of scripts/__pycache__.
    env = {**os.environ, "PYTHONPYCACHEPREFIX": str(tmp_path)}
    result = subprocess.run(  # noqa: S603 - repo-local interpreter and script paths
        [str(interpreter), "-m", "py_compile", str(path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script} does not compile under {interpreter}, which is the interpreter that runs it "
        f"in production.\n{result.stderr}"
    )


class TestTheScriptsTheHostSystemInterpreterRuns:
    """Three scripts the venv never touches: a GPU preflight, a canary tripwire, a secret scan.

    Skipped rather than failed when ``/usr/bin/python3`` is absent, unlike the jail group below:
    nothing in this repo requires the host to carry a system interpreter, and only ``canary-check``
    reaches for it.
    """

    @pytest.mark.skipif(
        not SYSTEM_PYTHON.exists(), reason=f"no system interpreter at {SYSTEM_PYTHON}"
    )
    @pytest.mark.parametrize("script", SYSTEM_PYTHON_SCRIPTS)
    def test_it_compiles_under_the_system_interpreter(self, script: str, tmp_path: Path) -> None:
        _assert_compiles(SYSTEM_PYTHON, script, tmp_path)


class TestTheScriptsTheJailInterpreterRuns:
    """The assertion suite, which run_jail_tests.sh runs under whatever the jail resolved."""

    @pytest.mark.parametrize("script", JAIL_PYTHON_SCRIPTS)
    def test_it_compiles_under_the_resolved_jail_interpreter(
        self, script: str, jail_python: Path, tmp_path: Path
    ) -> None:
        _assert_compiles(jail_python, script, tmp_path)


class TestTheJailInterpreterMeetsItsFloor:
    """The standing guard against the bug that split the two groups apart.

    Three separate assertions because they fail for three different reasons: a resolved path that is
    not runnable, an interpreter below the jail's stated floor, and the floor itself edited
    downwards. The last is the one a plausible future change would trip, and pinning only
    "version >= floor" would pass happily with the floor set back to 9.
    """

    def test_the_resolved_interpreter_is_a_file_this_box_can_execute(
        self, jail_python: Path
    ) -> None:
        assert jail_python.is_file(), f"{jail_python} is not a file"
        assert os.access(jail_python, os.X_OK), f"{jail_python} is not executable"

    def test_the_resolved_interpreter_is_at_or_above_the_floor(
        self, jail_python: Path, jail_python_floor: int
    ) -> None:
        major, minor = _interpreter_version(jail_python)

        assert major == 3, f"{jail_python} is Python {major}.{minor}, and the jail expects a 3.x"
        assert minor >= jail_python_floor, (
            f"{jail_python} is Python {major}.{minor}, below the jail's own floor of "
            f"3.{jail_python_floor}. Grading a submission under an interpreter older than the "
            "dialect it is written in fails working code as flawed instead of failing loudly."
        )

    def test_the_floor_itself_has_not_been_lowered(self, jail_python_floor: int) -> None:
        assert jail_python_floor >= MINIMUM_JAIL_PYTHON_MINOR, (
            f"the jail's Python floor is 3.{jail_python_floor}, below the "
            f"3.{MINIMUM_JAIL_PYTHON_MINOR} this repo grades against. That is the misgrading bug "
            "this module exists for: every submission using a newer builtin crashes, fails every "
            "unit, and is scored as carrying a planted flaw."
        )


class TestTheJailGradingEnvironmentCarriesNumpy:
    """numpy joined the grading environment on 2026-08-22, and nothing else pins it there.

    It was installed into the jail's resolved interpreter so that submissions importing it grade
    as measured results instead of load errors -- 19 stored Opus 5 rollouts had died that way and
    were excluded as unmeasurable (all 19 turned out to be correct solutions). The two documented
    interpreter migrations both drop it silently: ``stage_jail_python.sh --force`` replaces the
    staged tree wholesale from a numpy-less uv source, and a host gaining ``/usr/bin/python3.13``
    wins resolution with the distribution's bare site-packages. Either way every numpy submission
    quietly reverts to unmeasurable, which no other check would notice.

    ``-I`` matches how the graders invoke the interpreter, so a user-site numpy cannot satisfy
    this vicariously: the import has to come from the resolved interpreter's own site-packages,
    the one path that exists inside the jail.
    """

    def test_the_resolved_interpreter_imports_numpy(self, jail_python: Path) -> None:
        result = subprocess.run(  # noqa: S603 - resolved interpreter path, literal argument
            [str(jail_python), "-I", "-c", "import numpy"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"{jail_python} cannot import numpy: {result.stderr.strip()}\n"
            "numpy is part of the grading environment (since 2026-08-22); without it, submissions "
            "importing it fail at load and are excluded as unmeasurable. Reinstall it into the "
            "interpreter the jail resolves:\n"
            f"  uv pip install --python {jail_python} --break-system-packages numpy"
        )
