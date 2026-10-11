"""The fast gate and pre-GPU slow gate preserve complementary CPU coverage."""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from scripts.test_changed import pytest_command

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_recipe(target: str) -> str:
    return subprocess.run(  # noqa: S603
        [  # noqa: S607
            "make",
            "--no-print-directory",
            "-n",
            target,
            "GATE_TMUX=",
            "TEST_WORKERS=8",
            "GAMES_GPU_TESTS=",
            "FAST_TIER_BUDGET_SECONDS=10",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_fast_and_slow_recipes_use_complementary_cpu_markers() -> None:
    fast_recipe = make_recipe("test")
    slow_recipe = make_recipe("test-slow")
    assert '-m "not slow and not jail_real and not docker_real"' in fast_recipe
    assert '-m "slow and not jail_real and not docker_real"' in slow_recipe
    for recipe in (fast_recipe, slow_recipe):
        assert "RLVR_SMOKE=1 CUDA_VISIBLE_DEVICES=" in recipe
        assert "-n 8 --dist loadfile" in recipe
        assert "gate-test.lock" in recipe
    assert "FAST_TIER_BUDGET_SECONDS=10" in fast_recipe
    assert "FAST_TIER_BUDGET_SECONDS" not in slow_recipe


@pytest.mark.parametrize("target", ["test", "test-slow", "test-changed"])
def test_cpu_gate_recipes_refuse_hub_network_access(target: str) -> None:
    recipe = make_recipe(target)
    assert "HF_HUB_OFFLINE=1" in recipe
    assert "TRANSFORMERS_OFFLINE=1" in recipe


def test_slow_tier_runs_before_the_required_render_audit() -> None:
    recipe = make_recipe("verify-seed-v4")
    assert "test-corpus test-slow render-audit-required" in recipe


@pytest.mark.parametrize("workers", [1, 2])
def test_changed_selection_executes_slow_tests(tmp_path: Path, workers: int) -> None:
    """Exercise the real iteration command: slow markers must never deselect reachable tests."""
    test_file = tmp_path / "test_synthetic_tiers.py"
    test_file.write_text(
        textwrap.dedent(
            """\
            import pytest
            def test_fast(): pass
            @pytest.mark.slow
            def test_slow(): pass
            """
        ),
        encoding="utf-8",
    )
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\nmarkers = slow: expensive CPU coverage\n", encoding="utf-8")
    command = pytest_command([str(test_file)], workers=workers, test_files=workers)
    completed = subprocess.run(  # noqa: S603
        [*command, "-c", str(config), "-p", "xdist.plugin", "-q"],
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "2 passed" in completed.stdout
