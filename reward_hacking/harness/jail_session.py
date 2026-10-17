"""Episode-scoped actor executor, deliberately separate from trusted grading calls."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from reward_hacking.persistent_jail import PersistentJail

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from threading import Event


@dataclass
class _EpisodeSession:
    episode_dir: Path
    backend: str | None
    ro_bind_paths: tuple[Path, ...]
    stop_requested: Event | None
    jail: PersistentJail | None = None


_SESSION: ContextVar[_EpisodeSession | None] = ContextVar("actor_jail_session", default=None)


@contextmanager
def persistent_jail_scope(
    episode_dir: Path,
    *,
    enabled: bool,
    backend: str | None,
    ro_bind_paths: tuple[Path, ...] = (),
    stop_requested: Event | None = None,
) -> Generator[None]:
    """Lazily acquire the jail after workspace construction and always tear it down."""
    session = (
        _EpisodeSession(episode_dir.resolve(), backend, ro_bind_paths, stop_requested)
        if enabled
        else None
    )
    token = _SESSION.set(session)
    try:
        yield
    finally:
        _SESSION.reset(token)
        if session is not None and session.jail is not None:
            session.jail.close()


def current_persistent_jail(episode_dir: Path, *, cpus: int, mem_max: str) -> PersistentJail | None:
    """Return the actor session only for its own directory; graders never call this seam."""
    session = _SESSION.get()
    if session is None or session.episode_dir != episode_dir.resolve():
        return None
    if session.jail is None:
        session.jail = PersistentJail(
            session.episode_dir,
            backend=session.backend,
            ro_bind_paths=session.ro_bind_paths,
            stop_requested=session.stop_requested,
            cpus=cpus,
            mem_max=mem_max,
            # The harness deadline excludes shared-engine queue waits; a residence timer would not.
            episode_seconds=None,
        )
    return session.jail
