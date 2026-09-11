"""The twin-PD contrast pair as a `stage_runner` plan: one sweep, one regrade, two arms.

Driven with `uv run python -m games.stage_runner --plan games.contrast_pair_sequence`. This is the
headline experiment of the project rather than a plumbing run: `twin-pd-group` and `twin-pd-self`
train on the *same prompts* and differ only in what the reward is computed against, so a behavioural
difference between them is attributable to the grading's correlation structure and to nothing else.
That identity is why the corpus is swept once and re-graded, never swept twice -- two sweeps would
differ in their prompts and their selection attrition, and the contrast would no longer be clean.

Replaces `scripts/games_2b_chain.sh`, which was six bash stages and shipped three false greens in
one night. The rule those bugs bought is in `games/stage_runner.py`: a stage is a liar until its
artifact exists and is non-empty.

**Everything except "two arms from one sweep" is imported.** This file used to be a 440-line copy
of `games/arm_sequence.py` under a different variable prefix, and the copies had drifted: three
corpus guards existed only in the general plan for a while, so the headline experiment was the one
launching without them, and `--print-plan` -- the cheap pre-launch check -- existed only there too.
The stage argv, the corpus resolver, the regrade naming, the guards and the printout all come from
`games/plans.py` now, and what is left here is the two-arm shape: which arms, what each trains
under, and the fact that a shared stage's log is provenance for both run directories.

**Two invocations, on purpose.** `games.select_prompts` names its output with a timestamp, so no
plan can promise the corpus path before the sweep has run -- which is why selecting the sweep
beside a stage that reads the swept corpus is refused rather than silently reordered. Rather than
glob for it -- the exact move that once picked a 0-byte corpus and trained on nothing -- the sweep
runs alone, the corpus is resolved by `resolve_corpus` (which refuses to guess), and the remaining
stages are told the answer:

    GAMES_PAIR_STAGES=sweep uv run python -m games.stage_runner --plan games.contrast_pair_sequence
    export GAMES_PAIR_CORPUS=$(uv run python -m games.contrast_pair_sequence --print-corpus)
    GAMES_PAIR_STAGES=regrade,arm-group,arm-self \
      uv run python -m games.stage_runner --plan games.contrast_pair_sequence

The middle line is a tested function rather than a shell glob because it is the step an operator
runs at 2am on a rented instance, and it is the step where a silent wrong answer is affordable
enough to go unnoticed and expensive enough to matter.

**Thinking is on and the budget is the measured one.** Anything read as science needs the chain of
thought to actually finish; the budget comes from `games.termination.required_completion_budget`,
which resolves the per-model measured termination floor (24,576 tokens for the 2B), so pointing this
plan at another model moves the budget with it instead of silently keeping a number measured
elsewhere.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from games.arms import ARMS
from games.plans import (
    DEFAULT_SAMPLES_PER_PROMPT,
    LOG_DIR,
    RUN_DIR,
    SELECT_DIR,
    arm_s3_destination,
    artifact_model_slug,
    assert_corpus_will_exist,
    assert_opponent_probabilities_filled,
    build_arm_stage,
    build_regrade_stage,
    build_sweep_stage,
    default_arm_timeout,
    derived_completion_tokens,
    describe_stages,
    generation_backend,
    plan_setting,
    refuse_sweep_beside_a_corpus_consumer,
    regraded_copy,
    resolve_single_corpus,
    selected_stages_from,
    sweep_timeout_for,
    training_shape,
)
from games.reward_spread import log_reward_spread
from games.stage_runner import Stage, run_sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
ENV_PREFIX = "GAMES_PAIR_"
# The sweep samples the checkpoint about to be trained, so it runs through a local backend. Named
# rather than left to `games.select_prompts`' own default, because the sweep's cap keys on it.
SWEEP_BACKEND = "hf"

# Both arms see identical prompts; only the grading differs, which is the whole contrast.
SWEEP_GRADING = "group-mix"
CONTRAST_GRADING = "self"

# The default pair -- the headline twin-PD experiment. `group_arm` / `self_arm` read the pair to
# run from the environment with these as defaults, so the pd-unstated pair (and any later
# grading-contrast pair) launches through this same exercised plan instead of a new plan file.
GROUP_ARM = "twin-pd-group"
SELF_ARM = "twin-pd-self"

SWEEP_STAGE = "sweep"
REGRADE_STAGE = "regrade"
GROUP_ARM_STAGE = "arm-group"
SELF_ARM_STAGE = "arm-self"
ALL_STAGES: tuple[str, ...] = (SWEEP_STAGE, REGRADE_STAGE, GROUP_ARM_STAGE, SELF_ARM_STAGE)
# Every stage but the sweep reads the corpus the sweep writes; see `stages`.
CORPUS_CONSUMING_STAGES: tuple[str, ...] = (REGRADE_STAGE, GROUP_ARM_STAGE, SELF_ARM_STAGE)
# The sweep alone, since the corpus it writes is not nameable until it has run.
DEFAULT_STAGES: tuple[str, ...] = (SWEEP_STAGE,)

# The three commands the two-invocation workflow is, quoted verbatim by every refusal that needs it.
LAUNCH_RECIPE = (
    "    GAMES_PAIR_STAGES=sweep uv run python -m games.stage_runner "
    "--plan games.contrast_pair_sequence\n"
    "    export GAMES_PAIR_CORPUS=$(uv run python -m games.contrast_pair_sequence --print-corpus)\n"
    "    GAMES_PAIR_STAGES=regrade,arm-group,arm-self uv run python -m games.stage_runner "
    "--plan games.contrast_pair_sequence"
)


def model_id() -> str:
    """Return the model every stage runs against."""
    return plan_setting(f"{ENV_PREFIX}MODEL", DEFAULT_MODEL)


def model_tag() -> str:
    """Return the short tag this plan's artifacts are named with, for the model in use."""
    return artifact_model_slug(model_id())


def completion_tokens() -> str:
    """Resolve the completion budget for the model this sequence is pointed at."""
    return derived_completion_tokens(f"{ENV_PREFIX}COMPLETION_TOKENS", model_id=model_id())


def _registered_arm(env_var: str, default: str) -> str:
    """Resolve one of the pair's arm names from the environment, refusing one nothing registers."""
    name = plan_setting(env_var, default)
    if name not in ARMS:
        raise ValueError(
            f"unknown arm {name!r} in {env_var}; games.arms.ARMS carries {sorted(ARMS)}"
        )
    return name


def group_arm() -> str:
    """Return the group-graded arm this pair trains, defaulting to the twin-PD original."""
    return _registered_arm(f"{ENV_PREFIX}GROUP_ARM", GROUP_ARM)


def self_arm() -> str:
    """Return the self-graded contrast arm, defaulting to the twin-PD original."""
    return _registered_arm(f"{ENV_PREFIX}SELF_ARM", SELF_ARM)


def game_id() -> str:
    """Return the game both arms are trained on.

    Read once and used for both the sweep's `--game` argument and its stage name: the two disagreed
    while the name was built from the module default, so a run swept one game under a heading naming
    another, and the heading is what the log and the sequence report both quote.

    The default is what the registry says the selected group arm trains, not a constant here:
    passing the game separately from the arms is how a plan and the registry drift apart
    (`games.arm_sequence`'s rule), and an explicit override that contradicts the registry is
    refused by `assert_pair_is_coherent` before any stage is built.
    """
    return plan_setting(f"{ENV_PREFIX}GAME", ARMS[group_arm()].game_id)


def assert_pair_is_coherent() -> None:
    """Refuse a pair selection whose sweep and arms would describe different experiments.

    Three ways the environment can pull the plan apart, each of which otherwise surfaces as a
    corpus rejection a GPU wait and a tokenizer load into a rented instance: an arm slotted under
    the grading it does not train (the sweep grades under the group slot's rule and the regrade
    writes the self slot's, so a mis-slotted arm trains on a corpus graded some other way); two
    arms from different games (one swept corpus cannot be both); and an explicit GAMES_PAIR_GAME
    that contradicts what the registry says the arms train (a sweep of one game feeding arms of
    another).
    """
    slots = ((group_arm(), SWEEP_GRADING), (self_arm(), CONTRAST_GRADING))
    for name, slot_grading in slots:
        if ARMS[name].grading != slot_grading:
            raise ValueError(
                f"arm {name!r} grades under {ARMS[name].grading!r} but is slotted where the plan "
                f"needs {slot_grading!r}: the sweep and the regrade write corpora for the slot's "
                f"grading, so this arm would train on a corpus graded some other way."
            )
    games = {name: ARMS[name].game_id for name, _ in slots}
    if len(set(games.values())) != 1:
        raise ValueError(
            f"the selected arms span two games ({games}), and one swept corpus cannot serve both: "
            f"the pair's whole contrast is identical prompts under two gradings."
        )
    if game_id() != ARMS[group_arm()].game_id:
        raise ValueError(
            f"{ENV_PREFIX}GAME={game_id()!r} contradicts the registry, which says the selected "
            f"arms train {ARMS[group_arm()].game_id!r}: the sweep would render one game and the "
            f"arms would train another. Unset the override; the game is derived from the arms."
        )


def sweep_timeout() -> str:
    """Return the sweep's wall-clock cap, sized to the backend this plan sweeps through.

    A cap exists to stop a wedged run billing overnight, and a cap under the healthy run time is
    strictly worse than none: it kills a working stage at the point its output is most expensive to
    recreate. The 180m once committed here was such a cap -- this plan sweeps through the HF path,
    whose arithmetic for a 64-prompt, 8-sample sweep at the measured completion budget runs past
    three hours. The default comes from the same table `games.arm_sequence` uses, so the two plans
    cannot drift apart.
    """
    override = plan_setting(f"{ENV_PREFIX}SWEEP_TIMEOUT", "")
    return override or sweep_timeout_for(SWEEP_BACKEND)


def arm_timeout() -> str:
    """Return a training arm's wall-clock cap, sized to the generation backend the arm will use.

    Same reasoning as `sweep_timeout`, and the same shortfall in what was once committed: on the HF
    path a 70-step arm measures ~29 h on the local L4 and up to ~45 h when its prompts run to the
    completion cap, so the committed 24h would have killed a healthy overnight arm at the worst
    possible moment. The default comes from the shared table, so a colocate arm gets the shorter cap
    its own measurement earns.
    """
    return plan_setting(f"{ENV_PREFIX}ARM_TIMEOUT", default_arm_timeout())


def sweep_dir() -> Path:
    """Return the directory the sweep writes into, and the resolver later reads.

    Deliberately its own directory per model rather than the shared selection directory: that one
    already holds four corpora from the thinking-off plumbing chain, one of them 0 bytes, and a
    resolver pointed at it would have to choose between them. Give the sweep an empty room and
    "exactly one corpus is here" becomes a fact rather than a heuristic.
    """
    return Path(plan_setting(f"{ENV_PREFIX}SWEEP_DIR", str(SELECT_DIR / f"pair-{model_tag()}")))


def selected_stages() -> tuple[str, ...]:
    """Which stages to run, in plan order, from GAMES_PAIR_STAGES.

    Defaults to the sweep alone, because the corpus every later stage reads carries a timestamp the
    sweep chooses at run time. Naming a subset is also how a single arm is restarted after an
    interruption.
    """
    return selected_stages_from(
        f"{ENV_PREFIX}STAGES", all_stages=ALL_STAGES, default=DEFAULT_STAGES
    )


def resolve_corpus(directory: Path | None = None) -> Path:
    """Find the one corpus the sweep wrote, refusing to guess between candidates.

    The refusals themselves live in `games.plans.resolve_single_corpus`, which every arm launch
    shares: the globbing shortcut they replace is the one that once picked a 0-byte corpus and
    trained an arm on nothing while reporting success, and two copies of that judgement is one copy
    too many.
    """
    directory = directory if directory is not None else sweep_dir()
    return resolve_single_corpus(
        directory,
        corpus_env=f"{ENV_PREFIX}CORPUS",
        sweep_command=f"{ENV_PREFIX}STAGES=sweep",
        sweep_log=LOG_DIR / f"sweep-{model_tag()}.log",
    )


def group_corpus() -> Path:
    """Return the corpus the group-graded arm trains on, as swept."""
    explicit = plan_setting(f"{ENV_PREFIX}CORPUS", "")
    return Path(explicit) if explicit else resolve_corpus()


def self_corpus() -> Path:
    """Return where the re-graded, self-graded copy of the same prompts lives.

    Named by the same helper every plan uses, so the derivative carries the marker
    `resolve_single_corpus` skips: without it the regrade left two non-empty `corpus-*.jsonl` in the
    sweep directory and every later resolve was refused as ambiguous.
    """
    return regraded_copy(group_corpus(), CONTRAST_GRADING)


def arm_output_dir(arm: str) -> Path:
    """Return where one arm writes, which resume and the S3 sync both key on."""
    override = plan_setting(f"{ENV_PREFIX}OUTPUT_DIR_{arm.replace('-', '_').upper()}", "")
    return Path(override) if override else RUN_DIR / f"{arm}-{model_tag()}"


def arm_init_adapter(arm: str) -> str:
    """Return the adapter checkpoint seeding this arm's LoRA, or empty for the usual zero init.

    Per arm rather than per plan for the transfer-of-learning shape: the two gradings of a pair
    seed from different checkpoints (each from its own trained arm), and the from-base control
    cells of the same arms set nothing. `games.train` verifies the adapter against the run's own
    LoRA plan and records its sha256; a cell that re-runs the same ARM under a different init must
    also override that arm's output directory, or the second cell would resume the first.
    """
    return plan_setting(f"{ENV_PREFIX}INIT_ADAPTER_{arm.replace('-', '_').upper()}", "")


def both_arm_run_dirs() -> tuple[Path, ...]:
    """Return both arms' run directories, for the logs that are provenance for both.

    The sweep and the regrade are shared: one corpus, graded two ways, is the invariant the contrast
    rests on. So their logs belong in both run directories rather than in whichever arm happens to
    be listed first, and a run directory is what the S3 sync ships.
    """
    return (arm_output_dir(group_arm()), arm_output_dir(self_arm()))


def arm_s3_dest(arm: str) -> str:
    """Return the S3 prefix this arm ships to, or empty to leave the sync off."""
    return arm_s3_destination(f"{ENV_PREFIX}S3_DEST", arm=arm, model_tag=model_tag())


def sweep_stage() -> Stage:
    """Sweep the baseline policy once, under the grading the group-graded arm trains on."""
    return build_sweep_stage(
        game_id=game_id(),
        grading=SWEEP_GRADING,
        model_id=model_id(),
        backend_kind=SWEEP_BACKEND,
        samples_per_prompt=plan_setting(
            f"{ENV_PREFIX}SAMPLES_PER_PROMPT", DEFAULT_SAMPLES_PER_PROMPT
        ),
        out_dir=sweep_dir(),
        max_new_tokens=completion_tokens(),
        timeout=sweep_timeout(),
        log_path=LOG_DIR / f"sweep-{model_tag()}.log",
        log_run_dirs=both_arm_run_dirs(),
    )


def regrade_stage() -> Stage:
    """Re-grade the swept corpus into the self-graded contrast, on CPU."""
    return build_regrade_stage(
        source=group_corpus(),
        target=self_corpus(),
        from_grading=SWEEP_GRADING,
        to_grading=CONTRAST_GRADING,
        log_path=LOG_DIR / f"regrade-{model_tag()}.log",
        log_run_dirs=both_arm_run_dirs(),
    )


def _arm_stage(arm: str, corpus: Path) -> Stage:
    """Build one of the pair's arms, after the shared corpus guards have passed.

    The corpus is checked here rather than left to `games.train`: the self-graded arm derives its
    path from the group corpus, so on the documented two-invocation workflow nothing verified that
    the regrade had actually written it, and the failure surfaced a GPU wait and a tokenizer load
    later.
    """
    assert_corpus_will_exist(
        corpus,
        written_by_a_selected_stage=REGRADE_STAGE in selected_stages(),
        recipe=LAUNCH_RECIPE,
    )
    if corpus.is_file():
        assert_opponent_probabilities_filled(corpus, ARMS[arm])
    output_dir = arm_output_dir(arm)
    return build_arm_stage(
        arm=arm,
        model_id=model_id(),
        corpus=corpus,
        output_dir=output_dir,
        timeout=arm_timeout(),
        shape=training_shape(ENV_PREFIX, completion_tokens=completion_tokens()),
        s3_dest=arm_s3_dest(arm),
        log_path=LOG_DIR / f"arm-{arm}-{model_tag()}.log",
        log_run_dirs=(output_dir,),
        init_adapter=arm_init_adapter(arm),
    )


def group_arm_stage() -> Stage:
    """Build the group-graded arm: reward computed against the group's own realised mix."""
    return _arm_stage(group_arm(), group_corpus())


def self_arm_stage() -> Stage:
    """Build the self-graded contrast arm: identical prompts, reward against its own action."""
    return _arm_stage(self_arm(), self_corpus())


STAGE_BUILDERS = {
    SWEEP_STAGE: sweep_stage,
    REGRADE_STAGE: regrade_stage,
    GROUP_ARM_STAGE: group_arm_stage,
    SELF_ARM_STAGE: self_arm_stage,
}


def stages() -> list[Stage]:
    """Build the selected stages, in plan order.

    Stage construction resolves the corpus, so a missing or ambiguous one fails here rather than an
    hour into a rented instance -- and for the same reason the sweep cannot share an invocation with
    a stage that reads what it writes.

    The reward-spread table is logged before the sweep because by the time `games.train` logs it a
    GPU is reserved and a model is loading; the per-arm warning it carries is what makes "run the
    compressed payoff variants as separate arms" a decision rather than a post-mortem.
    """
    assert_pair_is_coherent()
    chosen = selected_stages()
    refuse_sweep_beside_a_corpus_consumer(
        chosen, sweep=SWEEP_STAGE, consumers=CORPUS_CONSUMING_STAGES, recipe=LAUNCH_RECIPE
    )
    log_reward_spread(ARMS[group_arm()])
    return [STAGE_BUILDERS[name]() for name in chosen]


def describe_plan() -> str:
    """Render the resolved plan -- commands, caps and promised artifacts -- without running it.

    The general one-arm plan had this and the headline experiment did not, which is backwards: the
    printout is the cheapest check on a launch before the meter starts, and this is the launch most
    worth checking. The two backends are printed as separate lines because they are independent
    axes, and one line claiming a single backend for both is worse than silence.
    """
    header = [
        f"arms        {group_arm()} and {self_arm()} (game={game_id()!r})",
        f"model       {model_id()}",
        (
            f"sweep       samples through {SWEEP_BACKEND} under {SWEEP_GRADING!r}, capped at {sweep_timeout()}"
        ),
        f"contrast    the same prompts re-graded to {CONTRAST_GRADING!r}",
        f"training    generates through {generation_backend()}, capped at {arm_timeout()}",
        f"budget      {completion_tokens()} completion tokens",
        f"stages      {', '.join(selected_stages())}",
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
    logger.info(f"contrast pair, {model_id()=} stages={selected_stages()}")
    return 0 if run_sequence(stages()).ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
