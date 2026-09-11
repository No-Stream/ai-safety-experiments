"""The JSONL trace format both benchmarks write, and the guard on where a trace may land.

Shared rather than owned by whichever benchmark was written first, because none of it is
benchmark-specific: a record discriminator, six lines of JSONL I/O, and one refusal.

The refusal is the reason this module exists rather than being a nicety of tidying. **Every response
record carries the item's full prompt text**, so a trace written under a tracked path publishes
benchmark items into future training data and contaminates the benchmark permanently. That guard
used to be a wrapper one benchmark applied around the other's writer, which left the writer itself
unguarded: ``.gitignore`` covers only ``artifacts/``, ``docs/scratch/``, ``canary/`` and ``notes/``
with no blanket ``*.jsonl``, so a sweep pointed at a user-supplied ``--handle-dir docs/sweeps/run7``
produced a git-tracked file of item prompts on a public remote. Reproduced: one benchmark's writer
happily wrote a prompt-bearing record to a tracked path inside the repository while the other's
refused the identical path. One writer, one guard, both callers covered.

Each benchmark keeps its own default trace root, so the refusal names the gitignored roots
generically rather than one benchmark's directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# The record discriminator every per-response line carries.
RESPONSE = "response"

# The roots of this repository that git ignores, and therefore the only ones a trace may live under.
GITIGNORED_TRACE_ROOTS = (Path("artifacts"), Path("docs/scratch"))


def _repo_root() -> Path | None:
    """Find the repository this module lives in, or None when it has been vendored out of one."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    return None


def refuse_tracked_trace_path(
    path: Path, *, carries: str = "every item's full prompt text"
) -> None:
    """Raise if a trace would land somewhere git tracks it.

    A destination outside the repository entirely (a scratch directory, a temp path, a mounted disk)
    is fine and is the common case for a one-off; what is refused is a path *inside* the repo that
    ``.gitignore`` does not cover, because every record carries item prompt text and this remote is
    public.

    ``carries`` names what the refused file would publish, so the refusal explains the actual risk:
    the default is right for trace records, but a batch job handle carries AWS account-identifying
    values instead, and an operator told to look for benchmark items in a file that has none will
    not find the real reason the destination matters.
    """
    root = _repo_root()
    if root is None:
        return
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        return
    relative = resolved.relative_to(root)
    if any(relative.is_relative_to(ignored) for ignored in GITIGNORED_TRACE_ROOTS):
        return
    allowed = ", ".join(str(ignored) for ignored in GITIGNORED_TRACE_ROOTS)
    msg = (
        f"refusing to write a trace to {relative}: it is inside this repository but not under a "
        f"gitignored root ({allowed}). This file carries {carries}, and this remote is public, so "
        f"committing it publishes that. Write under one of those roots or outside the repository "
        f"entirely"
    )
    raise ValueError(msg)


def write_trace(path: Path, records: Sequence[dict[str, Any]], *, append: bool = False) -> None:
    """Write records as JSONL, refusing a destination git would track and creating the parent.

    The guard runs before anything is created, so a refused path leaves no directory behind.

    ``append`` exists so a long sweep can persist each chunk as it completes rather than holding a
    whole run in memory and losing all of it to one throttle. It defaults to overwriting, because
    that is what a re-grade wants and an accidental append would silently double a rate's
    denominator.
    """
    refuse_tracked_trace_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("%s %d records to %s", "appended" if append else "wrote", len(records), path)


def load_trace(path: Path) -> list[dict[str, Any]]:
    """Read a trace back, skipping blank lines only."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
