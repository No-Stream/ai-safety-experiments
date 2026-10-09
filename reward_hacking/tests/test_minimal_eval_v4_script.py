"""Dry-run coverage for the seed-v4 staged evaluation launcher."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "organism_rl" / "minimal-eval-v4.sh"


def _dry_run(stage: str, model: str) -> str:
    bash_path = shutil.which("bash")
    if bash_path is None:
        raise RuntimeError("bash is required to run the evaluation launcher test")
    completed = subprocess.run(  # noqa: S603 - fixed local script and controlled test arguments
        [bash_path, str(SCRIPT_PATH), "--stage", stage, "--model", model, "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_gate3_dry_run_lists_only_cells_one_two_and_four() -> None:
    output = _dry_run("gate3", "v4")

    assert "cells: 1, 2, 4" in output
    assert "total episodes: 24" in output
    assert "estimated wall-clock: 0.9 h" in output
    assert "cells: 3" not in output
    assert "math cells:" not in output


def test_cells_dry_run_lists_v4_battery_and_deterministic_aisi_selection() -> None:
    output = _dry_run("cells", "v4")

    assert "cells: 3, 5, 6, 7, 8, 9" in output
    assert "total episodes: 47" in output
    assert "estimated wall-clock: 1.6 h" in output
    assert "AISI selection: round-robin by task ID (apps 3, humaneval 3, mbpp 2)" in output


def test_cells_dry_run_uses_model_specific_new_cells() -> None:
    base_output = _dry_run("cells", "base")
    v3_output = _dry_run("cells", "v3")

    assert "cells: 5, 6, 7, 8, 9" in base_output
    assert "total episodes: 39" in base_output
    assert "cells: 5, 7, 8, 9" in v3_output
    assert "total episodes: 31" in v3_output


def test_cells_dry_run_reports_the_two_math_variants_and_counts() -> None:
    output = _dry_run("cells", "base")

    assert (
        "math cells: 8=loose_oneoff (6 held-out tasks), 9=loose_hidden_original (6 held-out tasks)"
        in output
    )
    assert "until W6 lands" not in SCRIPT_PATH.read_text(encoding="utf-8")
