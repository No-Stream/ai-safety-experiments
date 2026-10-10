"""Dry-run coverage for the seed-v4 staged evaluation launcher."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "organism_rl" / "minimal-eval-v4.sh"
# The cells stage validates the leak cell against the gitignored eval-pair split, absent in clean checkouts.
LEAK_SPLITS = Path(__file__).resolve().parents[2] / "docs/scratch/leak-v4/channel-splits.json"
requires_leak_splits = pytest.mark.skipif(
    not LEAK_SPLITS.is_file(), reason="gitignored leak eval-pair split is absent"
)


@cache
def _dry_run(stage: str, model: str, *extra_arguments: str) -> str:
    bash_path = shutil.which("bash")
    if bash_path is None:
        raise RuntimeError("bash is required to run the evaluation launcher test")
    with tempfile.TemporaryDirectory() as temporary_directory:
        adapter_dir = Path(temporary_directory) / "synthetic-adapter"
        adapter_dir.mkdir()
        environment = os.environ.copy()
        environment.pop("PYTHON", None)
        environment["SEED_V4_ADAPTER_DIR"] = str(adapter_dir)
        completed = subprocess.run(  # noqa: S603 - fixed local script and controlled test arguments
            [
                bash_path,
                str(SCRIPT_PATH),
                "--stage",
                stage,
                "--model",
                model,
                *extra_arguments,
                "--dry-run",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout + completed.stderr


def _launcher_failure(*arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed local script and controlled test arguments
        ["bash", str(SCRIPT_PATH), *arguments, "--dry-run"],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    return completed.stderr


def test_gate3_dry_run_lists_only_cells_one_two_and_four() -> None:
    output = _dry_run("gate3", "v4")

    assert "cells: 1, 2, 4" in output
    assert "episodes per cell: 1=8, 2=8, 4=8" in output
    assert "total episodes: 24" in output
    assert "estimated wall-clock: 0.9 h" in output
    assert "excluded_ilcb_problem_ids=['lcbhard_16']" in output
    assert "cells: 3" not in output
    assert "math cells:" not in output


@requires_leak_splits
def test_cells_dry_run_lists_v4_battery_and_deterministic_aisi_selection() -> None:
    output = _dry_run("cells", "v4")

    assert "cells: 3, 5, 6, 7, 8, 9" in output
    assert "episodes per cell: 3=8, 5=8, 6=8, 7=11, 8=6, 9=6" in output
    assert "total episodes: 47" in output
    assert "estimated wall-clock: 1.6 h" in output
    assert "AISI selection: round-robin by task ID (apps 3, humaneval 3, mbpp 2)" in output
    assert output.count("excluded_ilcb_problem_ids=['lcbhard_16']") == 2


@requires_leak_splits
def test_cells_dry_run_uses_model_specific_new_cells() -> None:
    base_output = _dry_run("cells", "base")
    v3_output = _dry_run("cells", "v3")

    assert "cells: 5, 6, 7, 8, 9" in base_output
    assert "total episodes: 39" in base_output
    assert "excluded_ilcb_problem_ids=['lcbhard_16']" not in base_output
    assert "cells: 5, 7, 8, 9" in v3_output
    assert "total episodes: 31" in v3_output
    assert "excluded_ilcb_problem_ids=['lcbhard_16']" not in v3_output


@requires_leak_splits
def test_base_cells_dry_run_validates_the_ladder_cell_served_model_id() -> None:
    output = _dry_run("cells", "base")

    assert "validated ladder evaluation and runner arguments" in output
    assert "served_model_id=Qwen/Qwen3.5-9B" in output


@requires_leak_splits
def test_cells_dry_run_reports_the_two_math_variants_and_counts() -> None:
    output = _dry_run("cells", "base")

    assert (
        "math cells: 8=loose_oneoff (6 held-out tasks), 9=loose_hidden_original (6 held-out tasks)"
        in output
    )
    assert "until W6 lands" not in SCRIPT_PATH.read_text(encoding="utf-8")


def test_cells_option_runs_a_subset_of_the_stage() -> None:
    output = _dry_run("cells", "base", "--cells", "7")

    assert "cells: 7" in output
    assert "total episodes: 11" in output


def test_cells_option_refuses_a_cell_outside_the_stage() -> None:
    assert "cell 6 is not in stage cells:v3" in _launcher_failure(
        "--stage", "cells", "--model", "v3", "--cells", "6"
    )
