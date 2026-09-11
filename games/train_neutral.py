"""Launch the compute-matched non-game control, and record what it was matched to.

Thin on purpose. Everything about *what* the control is lives in `games.neutral_control`, which
imports no training stack; everything about *how* GRPO runs lives in `grpo.rlvr_math`, which already
asserts the Liger-faithful estimator, discovers its own LoRA targets through
`grpo.throughput.discover_lora_targets`, and carries the memory monitor, reward logging and verifier
accuracy callbacks. This module's whole job is the three things neither of them can do alone: derive
the reference arm's executed shape, refuse to launch when the control does not reproduce it, and
write the run's own `run_config.json` in the shape `games.run_evals` reads so the eval battery can
be pointed at the control the same way it is pointed at an arm.

The separation is also why this file exists rather than a mode inside `games/train.py`. That module
is what the whole experiment runs through and is being edited while arms are added to it; a control
that reuses the arithmetic harness needs none of it.

    uv run python -m games.train_neutral \
        --arm neutral-arithmetic \
        --reference-run-config <the registered reference run's own run_config.json> \
        --output-dir artifacts/games/arms/neutral-arithmetic-<model>

Which record that is comes from the registry rather than from this file: `NEUTRAL_TASK_ARMS[arm]`
names the reference run by arm, git sha and launch timestamp, and the launch refuses a record that is
a different run. The path is deliberately not written down anywhere tracked, because it points into
the gitignored artifacts tree or a run's S3 prefix and differs per box.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.neutral_control import (
    ARITHMETIC_TASK,
    NEUTRAL_TASK_ARMS,
    RUN_CONFIG_FILENAME,
    ComputeMatch,
    assert_compute_matched,
    assert_registered_reference,
    compute_match_from_run_config,
)
from games.provenance import git_provenance
from grpo.estimator_defaults import GRPO_LOSS_TYPE, GRPO_SCALE_REWARDS, executed_estimator
from grpo.rlvr_math import TrainConfig, train_grpo_integer_math

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def matched_train_config(
    match: ComputeMatch, *, output_dir: Path, run_name: str, grad_accum_steps: int
) -> TrainConfig:
    """Build the arithmetic harness's config so it reproduces the reference arm's shape.

    `quick_run=False` because the quick path takes `max_steps_quick` and would silently ignore the
    matched step count -- the one field whose whole purpose here is to be matched.

    `grad_accum_steps` is the operator's only sizing freedom, and it is a freedom about memory rather
    than about the experiment: episodes per step is the product of it and the micro-batch, so raising
    it lowers the micro-batch the match then demands. That matters on a small card, and it is why
    this is a flag rather than a constant. `assert_compute_matched` checks the product either way.

    Only the shape is taken from the match. `beta`, `learning_rate` and the two estimator constants
    stay at whatever `grpo.rlvr_math` trains at, and `assert_compute_matched` refuses the launch when
    they differ from the reference arm's. Today they do not: the harness runs the arms' `beta=0.0`,
    their learning rate of 1e-5 and their estimator pair, so a launch against the registered
    post-audit reference passes the objective check. The values are still not derived from the match,
    deliberately. Overwriting one here would make the check unfailable by changing what the control
    computes on the way past it, and what a control should compute is a question about the experiment
    for its owner to answer, which is why the check stays although the knobs currently agree.
    """
    if match.episodes_per_step % grad_accum_steps:
        raise ValueError(
            f"--grad-accum-steps {grad_accum_steps} does not divide the reference arm's "
            f"{match.episodes_per_step} episodes per step, so no micro-batch reproduces it. "
            f"{match.describe()}"
        )
    micro_batch = match.episodes_per_step // grad_accum_steps
    return TrainConfig(
        model_id=match.model_id,
        quick_run=False,
        max_steps_full=match.optimizer_steps,
        num_generations=match.num_generations,
        per_device_train_batch=micro_batch,
        grad_accum_steps=grad_accum_steps,
        run_name=run_name,
        output_dir=str(output_dir),
    )


def run_config_payload(
    *, arm: str, config: TrainConfig, match: ComputeMatch, task: str
) -> dict[str, Any]:
    """Build this control's `run_config.json`, in the shape `games.run_evals.load_run_facts` reads.

    `game_id` and `grading` are recorded as None rather than omitted or faked. The readouts brand
    this run UNREGISTERED because it is genuinely not in `games.arms.ARMS`, and their wording is that
    nothing says which game it trained -- which is misleading here, since its trained game is not
    unknown but absent. `matched_to` and `task` in this payload are what a reader needs to correct
    that, so they are recorded at the top level where a reader will find them.

    Which banked run of `matched_to` the compute was matched against rides in under `compute_match`,
    as `run_identity`, rather than being restated up here: one place for one fact, and a reader
    checking the pairing wants the shape beside it anyway.
    """
    return {
        "arm": arm,
        "game_id": None,
        "grading": None,
        "task": task,
        "matched_to": match.reference_arm,
        "compute_match": asdict(match),
        "executed_estimator": executed_estimator(
            GRPO_LOSS_TYPE,
            use_liger_kernel=config.use_liger_kernel,
            per_device_train_batch_size=config.per_device_train_batch,
        ),
        # Recorded at the top level because `grpo.rlvr_math` reads them from the repo-wide constants
        # rather than from `TrainConfig`, so `config` below cannot carry them the way an arm's does.
        "loss_type": GRPO_LOSS_TYPE,
        "scale_rewards": GRPO_SCALE_REWARDS,
        # dtype is a torch object and asdict cannot serialise it; the rest of the config is plain.
        "config": {
            key: value for key, value in asdict(config).items() if key not in {"dtype", "task"}
        },
        "task_config": asdict(config.task),
        **git_provenance(),
        "started_at": datetime.now(tz=UTC).isoformat(),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        required=True,
        choices=sorted(NEUTRAL_TASK_ARMS),
        help="Which registered non-game control to run.",
    )
    parser.add_argument(
        "--reference-run-config",
        required=True,
        type=Path,
        help=(
            f"The reference arm's {RUN_CONFIG_FILENAME}. The match is read from what that arm "
            f"EXECUTED, since games.train sizes its micro-batch from the VRAM it found."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help=(
            "Split the matched episodes-per-step across this many micro-batches. A memory knob, not "
            "an experimental one: the product is what the match is checked on."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Derive the match, refuse a mismatch or an unregistered reference, run it, record the pairing."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    arm = NEUTRAL_TASK_ARMS[args.arm]
    if arm.task != ARITHMETIC_TASK:
        raise ValueError(
            f"control arm {args.arm!r} names task {arm.task!r}, and this launcher only runs "
            f"{ARITHMETIC_TASK}. A second non-game task needs its own entry point rather than a "
            f"branch here, so that neither task's harness silently applies the other's defaults."
        )
    match = compute_match_from_run_config(args.reference_run_config, reference_arm=arm.matched_to)
    config = matched_train_config(
        match,
        output_dir=args.output_dir,
        run_name=args.arm,
        grad_accum_steps=args.grad_accum_steps,
    )
    assert_compute_matched(
        match,
        optimizer_steps=config.max_steps_full,
        per_device_train_batch=config.per_device_train_batch,
        grad_accum_steps=config.grad_accum_steps,
        num_generations=config.num_generations,
        model_id=config.model_id,
        beta=config.beta,
        learning_rate=config.learning_rate,
        # The two estimator knobs come from the constants rather than from `config`, because that is
        # where `grpo.rlvr_math` reads them: it has no field for either, so the constant IS the
        # control's executed value and passing anything else here would check a number nobody runs.
        loss_type=GRPO_LOSS_TYPE,
        scale_rewards=GRPO_SCALE_REWARDS,
        # Not an experimental knob on its own; it is the third input the check needs to derive what
        # aggregation this control will execute, alongside the loss type and the micro-batch.
        use_liger_kernel=config.use_liger_kernel,
    )
    assert_registered_reference(match, arm.reference_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_config_payload(arm=args.arm, config=config, match=match, task=arm.task)
    (args.output_dir / RUN_CONFIG_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("control run: %s", match.describe())
    train_grpo_integer_math(config)


if __name__ == "__main__":
    main()
