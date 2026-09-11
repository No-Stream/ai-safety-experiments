"""The 9B cloud run as a `stage_runner` plan: termination screen, throughput probe, then the arm.

Driven with `uv run python -m games.stage_runner --plan games.nine_b_sequence`, the only
sanctioned way to chain GPU stages here: three false-green bash bugs in one night bought that
rule, and `games/stage_runner.py` records them.

Everything is read from the environment rather than edited into the file, because this plan runs on
a rented instance that is torn down afterwards: an operator following
`docs/scratch/games-9b-ec2-runbook-2026-08-17.md` exports a handful of variables and re-runs the
same command, and re-running it is also the recovery procedure after an interruption.

The shape of a plan -- artifact roots, the `uv` prefix, the wall-clock cap tables, the slug artifact
names are built from -- comes from `games/plans.py`, which every plan shares. Two of those constants
had already drifted while this file carried its own copies: the screen directory, and an arm cap of
36h here against 60h in `games/arm_sequence.py` for the *smaller* model, which inverts the argument
both files make about caps.

Three deliberate differences from the local 2B plans:

**No resource limiter.** The limiter exists to stop a runaway job taking down the shared dev box; a
rented single-tenant instance has nothing to protect, and the limiter needs a systemd user manager
that an SSH-plus-tmux session on a fresh AMI may not have. The wall-clock cap it also provided is
kept with `timeout`, because a wedged run billing at a few dollars an hour is the failure mode that
actually costs money here. GPU-occupancy checking is not lost either: `stage_runner` waits for the
card to be free between stages.

**Resume is on by default.** The arm stage passes `--resume-from-checkpoint latest` with an explicit
`--output-dir`, so an interrupted 30-hour run is restarted by re-running the identical command. On a
first launch there is nothing to resume and the flag is a no-op.

**The completion budget defaults to the measured one, not a small number.** Anything read as
science needs the chain of thought to finish, and the 9B has not been screened yet -- which is why
the screen is stage one and its result should be read before trusting the arm's budget.
"""

from __future__ import annotations

import logging
from pathlib import Path

from games.plans import (
    DEFAULT_GROUP,
    DEFAULT_PROMPTS_PER_STEP,
    DEFAULT_SCREEN_BUDGET,
    LOG_DIR,
    RUN_DIR,
    SCREEN_TIMEOUT,
    THROUGHPUT_DIR,
    THROUGHPUT_TIMEOUT,
    UV,
    artifact_model_slug,
    build_arm_stage,
    build_screen_stage,
    default_arm_timeout,
    derived_completion_tokens,
    plan_setting,
    screen_artifact,
    selected_stages_from,
    training_shape,
)
from games.stage_runner import Stage, run_sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_ARM = "twin-pd-group"
ENV_PREFIX = "GAMES_9B_"

ALL_STAGES: tuple[str, ...] = ("screen", "throughput", "arm")


def model_id() -> str:
    """Return the model every stage runs against."""
    return plan_setting("GAMES_9B_MODEL", DEFAULT_MODEL)


def completion_tokens() -> str:
    """Resolve the completion budget for the model this sequence is pointed at."""
    return derived_completion_tokens(f"{ENV_PREFIX}COMPLETION_TOKENS", model_id=model_id())


def model_tag() -> str:
    """Return the short tag this plan's artifacts are named with, for the model in use."""
    return artifact_model_slug(model_id())


def arm_timeout() -> str:
    """Return the arm's wall-clock cap, sized to the generation backend the run will use.

    From the same table every plan reads, and overridable from the shell like every other setting:
    this file used to carry a literal 36h with no override, which is both shorter than the cap the
    smaller model's plan uses and unraisable from a rented box without editing the file. A cap under
    the healthy run time is strictly worse than no cap, because it kills a working arm at the moment
    its artifacts are most expensive to recreate.
    """
    return plan_setting("GAMES_9B_ARM_TIMEOUT", default_arm_timeout())


def selected_stages() -> tuple[str, ...]:
    """Which stages to run, in plan order, from GAMES_9B_STAGES.

    Defaults to all three. Naming a subset is how an operator reruns just the arm after reading the
    screen, without hand-editing a plan on a rented box. Unlike the sweeping plans, every stage here
    is safe to select at once: the arm's corpus is named explicitly rather than produced by an
    earlier stage of the same invocation.
    """
    return selected_stages_from(f"{ENV_PREFIX}STAGES", all_stages=ALL_STAGES, default=ALL_STAGES)


def screen_stage() -> Stage:
    """Screen whether this model stops thinking inside the budget, generation only.

    Stage one because it gates whether any behavioural number from the arm can be trusted. A model
    that never closes its thinking block produces empty visible text, every completion scores as a
    parse failure, and the reward's whole-batch raise stops the run -- the good outcome, but far
    better learned in 90 minutes of generation than after reserving a GPU for a day.

    No game or grading is named, unlike the per-arm plans: this runs before the 9B has a corpus at
    all, so it screens the screener's own default prompts.
    """
    budget = plan_setting(f"{ENV_PREFIX}SCREEN_BUDGET", DEFAULT_SCREEN_BUDGET)
    return build_screen_stage(
        model_id=model_id(),
        budget=budget,
        timeout=plan_setting(f"{ENV_PREFIX}SCREEN_TIMEOUT", SCREEN_TIMEOUT),
        artifact=screen_artifact(model_tag=model_tag(), budget=budget),
        log_path=LOG_DIR / f"screen-{model_tag()}-{budget}.log",
        log_run_dirs=(arm_output_dir(),),
    )


def throughput_stage() -> Stage:
    """Measure one step time on the rented card, which is what a cost estimate needs.

    The local L4 cannot host this model, so every number in the spend plan for it is currently an
    extrapolation. Five measured steps convert that into a dollars-per-episode figure before the
    long run commits.
    """
    artifact = THROUGHPUT_DIR / f"{model_tag()}-g7e.json"
    return Stage(
        name=f"throughput probe {model_id()}",
        argv=(
            "timeout",
            plan_setting("GAMES_9B_THROUGHPUT_TIMEOUT", THROUGHPUT_TIMEOUT),
            *UV,
            "-m",
            "grpo.throughput",
            "--model",
            model_id(),
            "--prompts-per-step",
            plan_setting("GAMES_9B_PROMPTS_PER_STEP", DEFAULT_PROMPTS_PER_STEP),
            "--group-size",
            plan_setting("GAMES_9B_GROUP", DEFAULT_GROUP),
            "--completion-tokens",
            completion_tokens(),
            "--measured-steps",
            "5",
            "--output-dir",
            str(THROUGHPUT_DIR / f"{model_tag()}-run"),
            "--json-out",
            str(artifact),
        ),
        artifacts=(artifact,),
        needs_gpu=True,
        log_path=LOG_DIR / f"throughput-{model_tag()}.log",
        log_run_dirs=(arm_output_dir(),),
    )


def arm_output_dir() -> Path:
    """Return where the arm writes, which resume and the S3 sync both key on."""
    arm = plan_setting("GAMES_9B_ARM", DEFAULT_ARM)
    return Path(plan_setting("GAMES_9B_OUTPUT_DIR", str(RUN_DIR / f"{arm}-{model_tag()}")))


def arm_stage() -> Stage:
    """Build the training arm, resumable and thinking-on, against a corpus named explicitly."""
    arm = plan_setting(f"{ENV_PREFIX}ARM", DEFAULT_ARM)
    corpus = plan_setting(f"{ENV_PREFIX}CORPUS", "")
    if not corpus:
        raise ValueError(
            f"{ENV_PREFIX}CORPUS is unset. The arm trains on a corpus selected at training "
            f"temperature by games.select_prompts; training on freshly generated prompts instead "
            f"skips that selection and is a different experiment, so it has to be asked for "
            f"explicitly."
        )
    if not Path(corpus).is_file():
        raise FileNotFoundError(
            f"{ENV_PREFIX}CORPUS {corpus!r} does not exist. Fetch it from S3 or run the selection "
            f"sweep first; failing here costs nothing, failing after the GPU is reserved does."
        )
    output_dir = arm_output_dir()
    return build_arm_stage(
        arm=arm,
        model_id=model_id(),
        corpus=Path(corpus),
        output_dir=output_dir,
        timeout=arm_timeout(),
        shape=training_shape(ENV_PREFIX, completion_tokens=completion_tokens()),
        # No rendered destination: this plan predates the per-arm S3 suffix and an operator on the
        # rented box exports GAMES_S3_DEST directly, which `games.train` reads for itself.
        s3_dest="",
        log_path=LOG_DIR / f"arm-{arm}-{model_tag()}.log",
        log_run_dirs=(output_dir,),
    )


STAGE_BUILDERS = {
    "screen": screen_stage,
    "throughput": throughput_stage,
    "arm": arm_stage,
}


def stages() -> list[Stage]:
    """Build the selected stages, in plan order.

    Stage construction validates its own inputs, so a missing corpus or an unknown stage name fails
    before any GPU is reserved rather than an hour into a rented instance.
    """
    return [STAGE_BUILDERS[name]() for name in selected_stages()]


def main() -> int:
    """Run the sequence, returning non-zero unless every stage verified."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.info(f"9B plan, {model_id()=} stages={selected_stages()}")
    outcome = run_sequence(stages())
    return 0 if outcome.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
