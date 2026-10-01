"""Real execution through the limiter and selected jail, using only synthetic commands."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from conftest import needs_jail

from reward_hacking.jailed_poc import run_in_jail

if TYPE_CHECKING:
    from pathlib import Path


@needs_jail
def test_commands_write_only_the_episode_and_propagate_exit_status(tmp_path: Path) -> None:
    result = run_in_jail(
        tmp_path,
        "printf synthetic > artifact.txt; python3 -I -c 'import numpy; print(7)'; exit 7",
        cpus=1,
        mem_max="1G",
    )
    assert result.exit_code == 7, result.stderr
    assert result.stdout.strip() == "7"
    assert (tmp_path / "artifact.txt").read_text() == "synthetic"
    assert not result.timed_out


@needs_jail
@pytest.mark.parametrize("detach", ["", "setsid "])
def test_timeout_kills_descendants_even_in_a_new_session(tmp_path: Path, detach: str) -> None:
    result = run_in_jail(
        tmp_path,
        f"printf started > started.txt; {detach}bash -c 'sleep 4; printf survived > late.txt' & wait",
        timeout="2s",
        cpus=1,
        mem_max="1G",
    )
    assert result.exit_code == 124, result.stderr
    assert result.timed_out
    assert (tmp_path / "started.txt").read_text() == "started", (
        "the timeout fired before the payload ran"
    )
    time.sleep(4.5)
    assert not (tmp_path / "late.txt").exists(), "a timed-out jail left a live descendant"
