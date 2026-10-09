"""Tests for resolving complete final organism checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reward_hacking.organism_rl import checkpoints

if TYPE_CHECKING:
    from collections.abc import Iterable


_CHECKPOINT_ARTIFACTS = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
)


def _write_run_config(run_dir: Path, max_steps: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps({"max_steps": max_steps}),
        encoding="utf-8",
    )


def _write_checkpoint(
    run_dir: Path,
    checkpoint_name: str,
    global_step: int,
    *,
    missing_artifacts: Iterable[str] = (),
) -> Path:
    checkpoint_dir = run_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True)
    missing_artifact_set = set(missing_artifacts)
    for artifact_name in _CHECKPOINT_ARTIFACTS:
        if artifact_name == "trainer_state.json":
            if artifact_name not in missing_artifact_set:
                (checkpoint_dir / artifact_name).write_text(
                    json.dumps({"global_step": global_step}),
                    encoding="utf-8",
                )
            continue
        if artifact_name not in missing_artifact_set:
            (checkpoint_dir / artifact_name).write_text(
                "synthetic checkpoint artifact", encoding="utf-8"
            )
    return checkpoint_dir


def test_resolve_final_uses_trainer_step_not_checkpoint_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "synthetic-run"
    _write_run_config(run_dir, max_steps=34)
    _write_checkpoint(run_dir, "checkpoint-9", global_step=9)
    expected_checkpoint = _write_checkpoint(run_dir, "checkpoint-34", global_step=34)

    assert checkpoints.resolve_final(run_dir) == expected_checkpoint


def test_real_seed_v3_run_resolves_checkpoint_34() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    run_dir = repository_root / "artifacts/distill/agentic-sft-seed-v3-onpolicy-ef398f9/train"
    if not run_dir.is_dir():
        pytest.skip("local seed-v3 training artifacts are unavailable")

    resolved_checkpoint = checkpoints.resolve_final(run_dir)

    assert resolved_checkpoint.name == "checkpoint-34"
    assert resolved_checkpoint.name != "checkpoint-9"


@pytest.mark.parametrize("missing_artifact", _CHECKPOINT_ARTIFACTS)
def test_resolve_final_refuses_torn_final_checkpoint(
    tmp_path: Path,
    missing_artifact: str,
) -> None:
    run_dir = tmp_path / "synthetic-run"
    _write_run_config(run_dir, max_steps=34)
    _write_checkpoint(run_dir, "checkpoint-9", global_step=9)
    _write_checkpoint(
        run_dir,
        "checkpoint-34",
        global_step=34,
        missing_artifacts=(missing_artifact,),
    )

    with pytest.raises(checkpoints.CheckpointResolutionError, match="checkpoint-34"):
        checkpoints.resolve_final(run_dir)


def test_resolve_final_exits_two_when_no_checkpoint_reaches_max_steps(tmp_path: Path) -> None:
    run_dir = tmp_path / "synthetic-run"
    _write_run_config(run_dir, max_steps=34)
    _write_checkpoint(run_dir, "checkpoint-9", global_step=9)

    with pytest.raises(SystemExit) as exit_info:
        checkpoints.main([str(run_dir)])

    assert exit_info.value.code == 2
