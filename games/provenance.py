"""One place to answer "which code produced this artifact?" for every games module.

Three copies of this logic had grown up independently -- in `select_prompts`, `train`, and `evals`
-- each with a piece the others lacked, which is the shape of bug where two artifacts from the same
run disagree about their own provenance. The union of what they needed is small:

- **The `GIT_SHA` environment variable wins.** The Batch image bakes the sha in at build time and
  carries no git checkout, so asking git inside a job returns nothing useful. `cloud/Dockerfile`
  sets it and `cloud/entrypoint.sh` logs it as its first line.
- **Never raise.** Provenance is metadata about work, and losing the ability to name a commit must
  not throw away the work itself. A failure is recorded *in the returned string* instead, so it
  reads as a failure rather than as a real commit.
- **Report whether the tree was dirty.** A record naming only a commit is indistinguishable from
  one produced by edited code. `None` means the question could not be answered, which is different
  from a clean tree and is kept distinguishable on purpose.

Stdlib only, and no import-time subprocess: this is imported by modules that run inside a training
loop.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

GIT_SHA_ENV = "GIT_SHA"
UNKNOWN_SHA = "unknown"

# Asked about the repo this file lives in, since a run may be launched from any directory.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_output(*args: str) -> tuple[int, str, str]:
    """Run a read-only git command against the repo root, returning (returncode, stdout, stderr)."""
    completed = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def git_sha(*, env_var: str = GIT_SHA_ENV) -> str:
    """Return the commit this code came from, or a string saying why that is unknown.

    Checks `env_var` first so a container that baked its sha in reports the truth rather than
    whatever a stray checkout would say. On failure the returned string carries the reason, which
    keeps an unattributable artifact visibly unattributable instead of quietly plausible.
    """
    from_env = os.environ.get(env_var)
    if from_env:
        return from_env
    returncode, stdout, stderr = _git_output("rev-parse", "HEAD")
    if returncode or not stdout:
        logger.warning(f"could not determine a git sha, recording it as unknown: {stderr!r}")
        return f"{UNKNOWN_SHA} (git rev-parse failed: {stderr})"
    return stdout


def git_tree_dirty() -> bool | None:
    """Report whether the checkout carried uncommitted edits, or None if git could not say.

    None rather than False when the answer is unavailable: "the tree was clean" and "nobody could
    tell" are different claims about an artifact, and collapsing them would make the reassuring one
    the default.
    """
    returncode, stdout, _ = _git_output("status", "--porcelain")
    if returncode:
        return None
    return bool(stdout)


def git_provenance(*, env_var: str = GIT_SHA_ENV) -> dict[str, object]:
    """Return the provenance fields every games artifact carries: the sha and the dirty flag."""
    return {"git_sha": git_sha(env_var=env_var), "git_tree_dirty": git_tree_dirty()}
