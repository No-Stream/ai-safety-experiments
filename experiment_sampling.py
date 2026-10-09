"""Shared per-cell episode-count enforcement for inference and evaluation entry points."""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

MAX_EPISODES_PER_CELL = 8


def add_oversize_reason_argument(parser: argparse.ArgumentParser) -> None:
    """Add the explicit explanation required to run a cell above the default episode cap."""
    parser.add_argument(
        "--oversize-reason",
        type=_nonempty_reason,
        default=None,
        metavar="TEXT",
        help=(
            "why this evaluation intentionally exceeds the 8-episode per-cell cap; "
            "the reason is recorded in run provenance"
        ),
    )


def log_sampling_grid(
    episodes_per_cell: Mapping[str, int], *, oversize_reason: str | None = None
) -> dict[str, object]:
    """Log and validate the full episode grid, returning its provenance fields."""
    if not episodes_per_cell:
        raise ValueError("sampling grid must contain at least one cell")
    if oversize_reason is not None and not oversize_reason.strip():
        raise ValueError("--oversize-reason must contain non-whitespace text")

    grid = dict(sorted(episodes_per_cell.items()))
    for cell, episode_count in grid.items():
        if not cell:
            raise ValueError("sampling cell labels must be nonempty")
        if type(episode_count) is not int or episode_count < 0:
            raise ValueError(
                f"sampling cell {cell!r} must have a nonnegative integer episode count, "
                f"got {episode_count!r}"
            )

    total_episodes = sum(grid.values())
    logger.info(
        "Sampling grid: %d cells, total episodes=%d, cap=%d episodes per cell",
        len(grid),
        total_episodes,
        MAX_EPISODES_PER_CELL,
    )
    for cell, episode_count in grid.items():
        logger.info("  %s: %d episodes", cell, episode_count)

    oversized = {
        cell: episode_count
        for cell, episode_count in grid.items()
        if episode_count > MAX_EPISODES_PER_CELL
    }
    if oversized and oversize_reason is None:
        oversized_cells = ", ".join(
            f"{cell} (n={episode_count})" for cell, episode_count in oversized.items()
        )
        raise ValueError(
            f"sampling grid exceeds the {MAX_EPISODES_PER_CELL}-episode per-cell cap: "
            f'{oversized_cells}. Pass --oversize-reason "<text>" to run an intentionally '
            "larger evaluation."
        )

    return {
        "max_episodes_per_cell": MAX_EPISODES_PER_CELL,
        "episodes_per_cell": grid,
        "total_episodes": total_episodes,
        "oversize_reason": oversize_reason,
    }


def _nonempty_reason(value: str) -> str:
    reason = value.strip()
    if not reason:
        raise argparse.ArgumentTypeError("--oversize-reason must contain non-whitespace text")
    return reason
