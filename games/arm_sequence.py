"""Any one matrix-game arm as a `stage_runner` plan: optional screen, baseline sweep, then the arm.

Driven with `uv run python -m games.stage_runner --plan games.arm_sequence`, with the arm named in
the environment. `games/contrast_pair_sequence.py` is the specialised twin-PD version of the same
shape -- two arms sharing one swept corpus -- and this module is what every *other* arm launches
through, so that adding an arm to the slate is a variable change against orchestration that has
already been exercised rather than a new plan file each time.

**Nothing about the experiment is written here.** The game and the grading come from
`games.arms.ARMS`, which is the single source of truth for what an arm *is*; passing them
separately is how a plan and a registry drift apart, and a sweep of the wrong game produces a
corpus `games.train.load_corpus` then rejects an hour later. The completion budget comes from
`games.termination.required_completion_budget`, so pointing this plan at another model moves the
budget with it instead of keeping a number measured elsewhere.

**Two invocations, on purpose, and the default plan is only the first of them.**
`games.select_prompts` timestamps its output, so no plan can promise the corpus path before the
sweep has run -- which is why selecting the sweep alongside a stage that reads the corpus is
refused rather than reordered (`games.plans.refuse_sweep_beside_a_corpus_consumer`):

    export GAMES_ARM_SEQ_ARM=hi-lo-group
    GAMES_ARM_SEQ_STAGES=sweep uv run python -m games.stage_runner --plan games.arm_sequence
    export GAMES_ARM_SEQ_CORPUS=$(uv run python -m games.arm_sequence --print-corpus)
    GAMES_ARM_SEQ_STAGES=arm uv run python -m games.stage_runner --plan games.arm_sequence

`--print-plan` renders every stage's command, timeout and promised artifact without running
anything and without a GPU, which is the cheapest way to check a launch before the meter starts.

**The wall-clock caps are generous on purpose, and each one tracks the path it bounds.** A cap
exists to stop a wedged run billing overnight; a cap below the healthy run time is strictly worse
than no cap, because it kills a working arm at the point where the artifacts are most expensive to
recreate. That is not hypothetical here -- the committed 180m/24h pair in the contrast pair sat
under the measured reality of both stages it bounded. The tables live in `games.plans` so every
plan shares them, and they key on two different axes: the sweep's cap follows `--backend`, the
arm's follows what `games.train` generates rollouts through.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from games.arms import ARMS, GameArm, arm_game_ids
from games.plans import (
    DEFAULT_SAMPLES_PER_PROMPT,
    DEFAULT_SCREEN_BUDGET,
    LOG_DIR,
    RUN_DIR,
    SCREEN_TIMEOUT,
    SELECT_DIR,
    arm_s3_destination,
    artifact_model_slug,
    assert_corpus_will_exist,
    assert_opponent_probabilities_filled,
    build_arm_stage,
    build_regrade_stage,
    build_screen_stage,
    build_sweep_stage,
    default_arm_timeout,
    derived_completion_tokens,
    describe_stages,
    generation_backend,
    optional_training_knobs,
    optional_training_switches,
    plan_setting,
    refuse_sweep_beside_a_corpus_consumer,
    regraded_copy,
    resolve_single_corpus,
    screen_artifact,
    selected_stages_from,
    sweep_timeout_for,
    training_shape,
)
from games.reward_spread import log_reward_spread
from games.rewards import GRADING_VS_FIXED_MIX, is_grading, unknown_grading_message
from games.stage_runner import Stage, run_sequence
from reward_hacking.backend_cli import LOCAL_KINDS

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_BACKEND = "hf"
DEFAULT_FROZEN_OPPONENT_SAMPLES = "8"
ENV_PREFIX = "GAMES_ARM_SEQ_"

# The screen is opt-in and the regrade is implied by a sweep grading other than the arm's own.
ALL_STAGES: tuple[str, ...] = ("screen", "sweep", "regrade", "arm")
# Stages that read the corpus the sweep writes, so they cannot share an invocation with it.
CORPUS_CONSUMING_STAGES: tuple[str, ...] = ("regrade", "arm")
SWEEP_STAGE = "sweep"
# The sweep alone, because its output's name is not knowable until it has run; see `stages`.
DEFAULT_STAGES: tuple[str, ...] = (SWEEP_STAGE,)

# The three commands the two-invocation workflow is, quoted verbatim by every refusal that needs it.
LAUNCH_RECIPE = (
    "    GAMES_ARM_SEQ_STAGES=sweep uv run python -m games.stage_runner "
    "--plan games.arm_sequence\n"
    "    export GAMES_ARM_SEQ_CORPUS=$(uv run python -m games.arm_sequence --print-corpus)\n"
    "    GAMES_ARM_SEQ_STAGES=arm uv run python -m games.stage_runner --plan games.arm_sequence"
)


def arm_name() -> str:
    """Return the arm this plan is launching, refusing to invent one.

    No default: every other setting here has a sensible one, but guessing which arm an operator
    meant would run a real experiment nobody asked for, on a paid box.
    """
    name = plan_setting("GAMES_ARM_SEQ_ARM", "")
    if not name:
        raise ValueError(
            f"GAMES_ARM_SEQ_ARM is unset, so there is no arm to run. Name one of "
            f"{sorted(ARMS)}; the game and grading come from games.arms.ARMS, not from here."
        )
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; games.arms.ARMS carries {sorted(ARMS)}")
    return name


def game_arm() -> GameArm:
    """Return the registry entry that defines this arm's game, grading and payoff pins."""
    return ARMS[arm_name()]


def model_id() -> str:
    """Return the model every stage runs against."""
    return plan_setting("GAMES_ARM_SEQ_MODEL", DEFAULT_MODEL)


def model_tag() -> str:
    """Return the short tag this plan's artifacts are named with, for the model in use."""
    return artifact_model_slug(model_id())


def backend() -> str:
    """Return the sweep's inference backend, restricted to the kinds that can sweep the policy.

    Hosted and mock backends are refused rather than passed through. A sweep exists to measure the
    behaviour of *the checkpoint about to be trained* at the training sampler, so a Bedrock
    endpoint would select prompts for a different model, and the mock backend samples nothing --
    both would produce a corpus that reads as a baseline and is not one. `games.select_prompts`
    would also reject the sampler flags below for either kind, but it would do so after the plan
    had already claimed to be launching an arm.
    """
    kind = plan_setting("GAMES_ARM_SEQ_BACKEND", DEFAULT_BACKEND)
    if kind not in LOCAL_KINDS:
        raise ValueError(
            f"GAMES_ARM_SEQ_BACKEND={kind!r} cannot sweep a training baseline; use one of "
            f"{sorted(LOCAL_KINDS)}. A hosted backend measures a different model than the one "
            f"this plan trains, and the mock backend does not sample at all."
        )
    return kind


def completion_tokens() -> str:
    """Resolve the completion budget for the model this plan is pointed at."""
    return derived_completion_tokens(f"{ENV_PREFIX}COMPLETION_TOKENS", model_id=model_id())


def sweep_grading() -> str:
    """Return the grading the corpus is swept under, defaulting to the arm's own.

    Overriding it is how one sweep serves two gradings: the corpus is swept once, re-graded on
    CPU, and both arms train on the same prompts -- the invariant the twin-PD contrast rests on.
    Sweeping twice instead would give the two arms different prompts and different selection
    attrition, and any behavioural difference between them would no longer be attributable to the
    grading.
    """
    grading = plan_setting("GAMES_ARM_SEQ_SWEEP_GRADING", game_arm().grading)
    if not is_grading(grading):
        raise ValueError(
            f"GAMES_ARM_SEQ_SWEEP_GRADING={grading!r} is not a grading. "
            f"{unknown_grading_message(grading)}"
        )
    return grading


def regrade_required() -> bool:
    """Report whether the corpus is swept under a grading this arm does not train on."""
    return sweep_grading() != game_arm().grading


def sweep_dir() -> Path:
    """Return the directory the sweep writes into, and the resolver later reads.

    One directory per arm and model rather than the shared selection directory, which already
    holds several corpora from earlier chains including a 0-byte one. Given an empty room,
    "exactly one corpus is here" is a fact rather than a heuristic.
    """
    override = plan_setting("GAMES_ARM_SEQ_SWEEP_DIR", "")
    return Path(override) if override else SELECT_DIR / f"{arm_name()}-{model_tag()}"


def selected_stages() -> tuple[str, ...]:
    """Which stages to run, in plan order, from GAMES_ARM_SEQ_STAGES.

    Defaults to the sweep and the arm. Naming a subset is how the sweep runs on its own before the
    corpus can be resolved, and how one stage is restarted after an interruption.
    """
    return selected_stages_from(
        f"{ENV_PREFIX}STAGES", all_stages=ALL_STAGES, default=DEFAULT_STAGES
    )


def resolve_corpus(directory: Path | None = None) -> Path:
    """Resolve this arm's swept corpus, or explain why it cannot be resolved."""
    directory = directory if directory is not None else sweep_dir()
    return resolve_single_corpus(
        directory,
        corpus_env="GAMES_ARM_SEQ_CORPUS",
        sweep_command="GAMES_ARM_SEQ_STAGES=sweep",
        sweep_log=LOG_DIR / f"sweep-{arm_name()}-{model_tag()}.log",
    )


def swept_corpus() -> Path:
    """Return the corpus as the sweep wrote it, under `sweep_grading()`."""
    explicit = plan_setting("GAMES_ARM_SEQ_CORPUS", "")
    return Path(explicit) if explicit else resolve_corpus()


def arm_corpus() -> Path:
    """Return the corpus this arm trains on: the swept one, or its re-graded copy."""
    source = swept_corpus()
    if not regrade_required():
        return source
    return regraded_copy(source, game_arm().grading)


def arm_output_dir() -> Path:
    """Return where the arm writes, which resume and the S3 sync both key on."""
    override = plan_setting("GAMES_ARM_SEQ_OUTPUT_DIR", "")
    return Path(override) if override else RUN_DIR / f"{arm_name()}-{model_tag()}"


def arm_s3_dest() -> str:
    """Return the S3 prefix this arm ships to, or empty to leave the sync off."""
    return arm_s3_destination(f"{ENV_PREFIX}S3_DEST", arm=arm_name(), model_tag=model_tag())


def sweep_timeout() -> str:
    """Return the sweep's wall-clock cap, which is the one cap the backend really changes."""
    override = plan_setting("GAMES_ARM_SEQ_SWEEP_TIMEOUT", "")
    return override or sweep_timeout_for(backend())


def arm_timeout() -> str:
    """Return the training stage's wall-clock cap, sized to the generation backend it will use."""
    return plan_setting("GAMES_ARM_SEQ_ARM_TIMEOUT", default_arm_timeout())


def frozen_opponent_args(arm: GameArm) -> tuple[str, ...]:
    """Return the sweep flags that cache a frozen opponent's mix, for the arms that need one."""
    if arm.grading != GRADING_VS_FIXED_MIX:
        return ()
    opponent = plan_setting("GAMES_ARM_SEQ_FROZEN_OPPONENT_MODEL", "")
    if not opponent:
        raise ValueError(
            f"arm {arm_name()!r} is graded {GRADING_VS_FIXED_MIX!r}, so its corpus needs a frozen "
            f"opponent's cooperation rate cached during the sweep. Set "
            f"GAMES_ARM_SEQ_FROZEN_OPPONENT_MODEL to the Bedrock model id to sample; without it "
            f"every row is written with opp_coop_prob=-1 and the arm cannot be graded."
        )
    return (
        "--frozen-opponent-model",
        opponent,
        "--frozen-opponent-samples",
        plan_setting("GAMES_ARM_SEQ_FROZEN_OPPONENT_SAMPLES", DEFAULT_FROZEN_OPPONENT_SAMPLES),
    )


def screen_stage() -> Stage:
    """Screen whether this model stops thinking inside the budget, on this arm's own prompts."""
    arm = game_arm()
    budget = plan_setting(f"{ENV_PREFIX}SCREEN_BUDGET", DEFAULT_SCREEN_BUDGET)
    return build_screen_stage(
        model_id=model_id(),
        budget=budget,
        timeout=plan_setting(f"{ENV_PREFIX}SCREEN_TIMEOUT", SCREEN_TIMEOUT),
        artifact=screen_artifact(model_tag=model_tag(), budget=budget, game_id=arm.game_id),
        log_path=LOG_DIR / f"screen-{arm.game_id}-{model_tag()}-{budget}.log",
        log_run_dirs=(arm_output_dir(),),
        game_id=arm.game_id,
        grading=arm.grading,
    )


def sweep_stage() -> Stage:
    """Sweep the baseline policy for this arm, under `sweep_grading()`."""
    arm = game_arm()
    return build_sweep_stage(
        game_id=arm.game_id,
        grading=sweep_grading(),
        model_id=model_id(),
        backend_kind=backend(),
        samples_per_prompt=plan_setting(
            f"{ENV_PREFIX}SAMPLES_PER_PROMPT", DEFAULT_SAMPLES_PER_PROMPT
        ),
        out_dir=sweep_dir(),
        max_new_tokens=completion_tokens(),
        timeout=sweep_timeout(),
        log_path=LOG_DIR / f"sweep-{arm_name()}-{model_tag()}.log",
        log_run_dirs=(arm_output_dir(),),
        extra_argv=frozen_opponent_args(arm),
    )


def regrade_stage() -> Stage:
    """Re-grade the swept corpus into this arm's own grading, on CPU.

    Refusing a no-op regrade is this plan's own rule rather than the shared builder's: a plan that
    sweeps under the arm's grading has nothing to re-grade, and rewriting the corpus unchanged would
    leave a second non-empty file for the resolver to refuse to choose between.
    """
    arm = game_arm()
    if not regrade_required():
        raise ValueError(
            f"the regrade stage was selected but the sweep grading ({sweep_grading()!r}) already "
            f"matches this arm's ({arm.grading!r}), so it would rewrite the corpus unchanged. "
            f"Set {ENV_PREFIX}SWEEP_GRADING to the grading the corpus was swept under."
        )
    return build_regrade_stage(
        source=swept_corpus(),
        target=arm_corpus(),
        from_grading=sweep_grading(),
        to_grading=arm.grading,
        log_path=LOG_DIR / f"regrade-{arm_name()}-{model_tag()}.log",
        log_run_dirs=(arm_output_dir(),),
    )


def arm_stage() -> Stage:
    """Build this plan's one training arm, after the two corpus guards have passed."""
    arm = game_arm()
    corpus, output_dir = arm_corpus(), arm_output_dir()
    assert_corpus_will_exist(
        corpus, written_by_a_selected_stage="regrade" in selected_stages(), recipe=LAUNCH_RECIPE
    )
    if corpus.is_file():
        assert_opponent_probabilities_filled(corpus, arm)
    return build_arm_stage(
        arm=arm_name(),
        model_id=model_id(),
        corpus=corpus,
        output_dir=output_dir,
        timeout=arm_timeout(),
        shape=training_shape(ENV_PREFIX, completion_tokens=completion_tokens()),
        s3_dest=arm_s3_dest(),
        log_path=LOG_DIR / f"arm-{arm_name()}-{model_tag()}.log",
        log_run_dirs=(output_dir,),
    )


STAGE_BUILDERS = {
    "screen": screen_stage,
    "sweep": sweep_stage,
    "regrade": regrade_stage,
    "arm": arm_stage,
}


def stages() -> list[Stage]:
    """Build the selected stages, in plan order.

    Stage construction resolves the corpus and validates the arm, so a missing corpus, an unknown
    arm or an ungradeable frozen-opponent column fails here rather than an hour into a rented
    instance. That is also why sweeping and training cannot share one invocation, which
    `refuse_sweep_beside_a_corpus_consumer` explains and DEFAULT_STAGES therefore does not attempt.

    The regrade check is the one that is easy to get wrong from the outside: an arm whose grading
    differs from the sweep's trains on a file only the regrade stage writes, so leaving that stage
    out would train on a corpus that does not exist yet, or worse, on a stale one.
    """
    arm = game_arm()
    chosen = selected_stages()
    refuse_sweep_beside_a_corpus_consumer(
        chosen, sweep=SWEEP_STAGE, consumers=CORPUS_CONSUMING_STAGES, recipe=LAUNCH_RECIPE
    )
    if regrade_required() and "arm" in chosen and "regrade" not in chosen:
        raise ValueError(
            f"arm {arm_name()!r} is graded {arm.grading!r} but the corpus is swept under "
            f"{sweep_grading()!r}, so the arm trains on the re-graded copy that only the regrade "
            f"stage writes. Add 'regrade' to GAMES_ARM_SEQ_STAGES, or drop "
            f"GAMES_ARM_SEQ_SWEEP_GRADING so the sweep is graded as the arm is."
        )
    log_reward_spread(arm)
    return [STAGE_BUILDERS[name]() for name in chosen]


def describe_plan() -> str:
    """Render the resolved plan -- commands, caps and promised artifacts -- without running it.

    The two backends are printed as two lines because they are two independent axes and a single
    line claiming one of them for both is worse than silence: this printout is the cheapest check on
    a launch before the meter starts, and it read "training generates through the HF path"
    unconditionally for as long as the colocate switch has existed.

    One `knob` line per optional training knob or switch the environment set, and none at all when it
    set none, so a kit's plan gate can grep for the treatment it meant to launch and an unset plan's
    printout says nothing it did not ask for. Spaced as `describe_stages` joins an argv, so the same
    grep matches the header line and the rendered command. A gate reads this both ways: the wave-4b
    probe requires its instrument switches present, and every TRAIN and PROBE role requires
    `--allow-short-completions` absent, that flag belonging to a plumbing smoke and nothing else.

    The `games` line is the whole set the arm's corpus may carry, printed for every arm rather than
    only for a breadth one: the header's `game=` names the lead game alone, and on a mixed-corpus arm
    that reads as a corpus five games smaller than the one about to be trained. A line that is always
    there is also the one a kit can pin, since a gate grepping for a line that appears only sometimes
    passes trivially on the run where it went missing.
    """
    header = [
        f"arm         {arm_name()} (game={game_arm().game_id!r} grading={game_arm().grading!r})",
        f"games       {', '.join(arm_game_ids(game_arm()))}",
        f"model       {model_id()}",
        f"sweep       samples through {backend()}, capped at {sweep_timeout()}",
        f"training    generates through {generation_backend()}, capped at {arm_timeout()}",
        f"budget      {completion_tokens()} completion tokens",
        f"stages      {', '.join(selected_stages())}",
        *(f"knob        {flag} {value}" for flag, value in optional_training_knobs(ENV_PREFIX)),
        *(f"knob        {flag}" for flag in optional_training_switches(ENV_PREFIX)),
    ]
    return describe_stages(header, stages())


def main(argv: list[str] | None = None) -> int:
    """Print the resolved corpus or plan, or run the sequence."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-corpus",
        action="store_true",
        help="Resolve and print the swept corpus path, then exit. Fails loudly if it is ambiguous.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print every stage's command, cap and promised artifacts without running anything.",
    )
    args = parser.parse_args(argv)
    if args.print_corpus:
        print(resolve_corpus())  # noqa: T201  -- the point is to be read by $(...)
        return 0
    if args.print_plan:
        print(describe_plan())  # noqa: T201  -- meant for an operator's terminal
        return 0
    logger.info(f"arm plan, arm={arm_name()!r} {model_id()=} stages={selected_stages()}")
    return 0 if run_sequence(stages()).ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
