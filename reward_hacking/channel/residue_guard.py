"""Residue guard: proves the per-episode writable work tree is actually reset.

The cross-episode channel probe rests on one isolation guarantee: the *only* path by
which information may cross the episode boundary is the shared channel directory we are
deliberately measuring. If a per-episode work tree carries state past a reset — a stray
file a botched teardown left behind — then every arm silently gains a second, unintended
channel, and the no-note / no-pressure control arms are no longer channel-free. A read
the detector attributes to the shared channel could then have come from work-tree
residue instead, and the whole measurement becomes unfalsifiable.

So a nonzero hash delta of the writable work tree across a reset fails the *whole run*,
not the offending episode. There is no safe way to keep measuring once the isolation the
measurement depends on is known broken; continuing would just produce numbers that read
as clean-channel results.

Scope discipline that is easy to get wrong: this guard watches the per-episode work tree
ONLY. The shared channel directory persists across episodes on purpose — pointing this
guard at it would flag the very state the probe exists to measure. Construct the guard
from the clean task template, and call ``assert_reset_clean`` on the reset work tree.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DIR_SENTINEL = "<dir>"
_READ_CHUNK = 1 << 20


class ResidueError(RuntimeError):
    """A writable per-episode work tree carried residue past a reset.

    Raised to abort the entire probe run rather than one episode: see the module
    docstring for why a leaky reset invalidates every arm, not just the current one.
    """


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def hash_writable_tree(root: Path) -> dict[str, str]:
    """Content-addressed map of everything under ``root``, keyed by POSIX relative path.

    Regular files map to ``sha256:<hex>`` of their bytes. Directories map to a sentinel,
    so a reset that leaves an empty scratch directory behind is still caught — an empty
    directory is a place a later episode can write, hence a channel. Symlinks map to
    ``symlink:<target>`` recorded without following, so a residue symlink pointing back
    at prior state is caught rather than silently resolved into whatever it targets.

    Comparison is plain dict equality; ordering does not matter.
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"writable tree root is not a directory: {root}")

    tree: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            tree[rel] = f"symlink:{path.readlink().as_posix()}"
        elif path.is_dir():
            tree[rel] = _DIR_SENTINEL
        elif path.is_file():
            tree[rel] = _hash_file(path)
        else:
            # Sockets, fifos, device nodes: record the type so their appearance is a delta.
            tree[rel] = f"special:{path.stat().st_mode:#o}"
    return tree


def _describe_delta(baseline: dict[str, str], current: dict[str, str]) -> str:
    """Name the paths that differ. At least one part is always populated at the only call site.

    That caller has already established the two trees are unequal. Empty ``added``, ``removed`` and
    ``changed`` together mean equal key sets with every shared value equal, i.e. equal dicts, so no
    "metadata only" case exists to report -- and reporting one would misdescribe the guard, which
    records content digests rather than mode bits or mtimes.
    """
    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))
    changed = sorted(k for k in baseline.keys() & current.keys() if baseline[k] != current[k])
    parts: list[str] = []
    if added:
        parts.append(f"added={added}")
    if removed:
        parts.append(f"removed={removed}")
    if changed:
        parts.append(f"changed={changed}")
    return "; ".join(parts)


@dataclass(frozen=True)
class ResidueGuard:
    """Holds the clean template hashes and checks each reset work tree against them."""

    baseline: dict[str, str]

    @classmethod
    def from_template(cls, template_dir: Path) -> ResidueGuard:
        """Snapshot the clean task template as the baseline every reset must return to."""
        return cls(baseline=hash_writable_tree(Path(template_dir)))

    def assert_reset_clean(self, work_dir: Path, *, episode_id: str) -> None:
        """Raise ``ResidueError`` if the reset work tree differs from the template.

        Called after each episode's workspace reset and before the next episode runs, so
        residue that survived the reset is caught at the boundary it crossed.
        """
        current = hash_writable_tree(Path(work_dir))
        if current != self.baseline:
            delta = _describe_delta(self.baseline, current)
            raise ResidueError(
                f"writable work tree carried residue past reset (episode {episode_id!r}); "
                f"the channel-free arms are no longer channel-free, so the entire run is "
                f"invalid and no arm's read/write-rate can be trusted: {delta}"
            )
        logger.debug("residue guard clean after episode %s", episode_id)
