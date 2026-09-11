r"""One trainable arm as a ``games.stage_runner`` plan: screen, train, evaluate, read out.

Driven with ``uv run python -m games.stage_runner --plan reward_hacking.train_sequence``, with the
arm named in the environment -- any arm in ``reward_hacking.train_dataset.TRAINABLE_ARMS``, so the
misspecified-grader pair and the legibility design's legible-subset arm launch through one plan.
The stage runner is what makes a chain honest: it refuses to call a stage successful without
checking the artifact that stage promised exists and is non-empty, it waits for the card to free
before a GPU stage, and it stops at the first stage it could not verify rather than running the
rest against a missing input.

    export RH_OPTION3_ARM=misspecified   # or control, or legible-subset
    RH_OPTION3_STAGES=screen uv run python -m games.stage_runner --plan reward_hacking.train_sequence
    RH_OPTION3_STAGES=arm,eval,analyze \
        uv run python -m games.stage_runner --plan reward_hacking.train_sequence

**Nothing about the experiment is decided here.** Which grader an arm trains against comes from
``reward_hacking.train_dataset.SPLIT_BY_TRAINABLE_ARM``, which is the one place that says what an arm
*is*; the completion budget comes from ``reward_hacking.train_termination``, so pointing this plan at
another model moves the budget with it rather than keeping a number measured elsewhere; and the
held-out task set comes from the stored partition, which both the training corpus and the evaluation
read and neither recomputes.

**Two invocations on purpose, and the screen is the first of them.** The gradient screen measures
whether the untrained model produces *any* within-group disagreement on this arm's prompts. GRPO
learns only from disagreement, so an arm whose every group is pure trains nothing while every signal
stays green -- and the untrained model was measured passing the misspecified grader 0 times in 101
agentic episodes, so this is a live risk rather than a formality. The screen is inference-only and
costs a fraction of one arm, which makes running it first the cheapest decision in the whole design.
``--print-plan`` renders every stage's command, timeout and promised artifact without running
anything and without a GPU.

**The wall-clock caps are generous on purpose.** A cap exists to stop a wedged run billing overnight;
a cap below the healthy run time is strictly worse than no cap, because it kills a working arm at the
point where its artifacts are most expensive to recreate. The arm's cap carries the 2026-08-24 timing
probe at 70 steps plus the reclaim allowance -- three steps at 1,696.6 / 1,580.8 / 1,785.4 seconds,
a median-excluding-first of 28.05 minutes, so 70 steps project to 32.7 hours at +-7% (n=3), and worse
where reclaims interrupt. The earlier "12-20 minutes per step" this file used to cite was an estimate
the measurement then exceeded, and the 30h cap derived from it would have killed a healthy arm at
about step 64 of 70. The evaluation's cap carries a full checkpoint ladder of multi-turn agentic
episodes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from games.plans import UV, describe_stages, plan_setting, selected_stages_from
from games.stage_runner import Stage
from reward_hacking.train import (
    DEFAULT_GRADER_SCRATCH_ROOT,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MODEL_ID,
    RUN_ROOT,
)
from reward_hacking.train_dataset import (
    ARM_MISSPECIFIED,
    SOLUTION_PARSER,
    SPLIT_BY_TRAINABLE_ARM,
    TRAINABLE_ARMS,
    TRAINING_GRADER_EXPOSURE,
)
from reward_hacking.train_eval import DEFAULT_LADDER_STRIDE, LADDER_SUMMARY_FILENAME
from reward_hacking.train_partition import (
    DEFAULT_PARTITION_PATH,
    SPLIT_CONFLICTING,
    SPLIT_ORIGINAL,
)
from reward_hacking.train_termination import required_coding_completion_budget

logger = logging.getLogger(__name__)

ENV_PREFIX = "RH_OPTION3_"

ARTIFACT_ROOT = Path("artifacts/reward_hacking/option3")
LOG_DIR = ARTIFACT_ROOT / "logs"
SCREEN_DIR = ARTIFACT_ROOT / "screen"
EVAL_DIR = ARTIFACT_ROOT / "eval"
ANALYSIS_DIR = ARTIFACT_ROOT / "analysis"

# The modules each stage invokes. Named here rather than inline so a rename is one edit, and so
# `--print-plan` is a complete statement of what the chain will run.
SCREEN_MODULE = "reward_hacking.train_screen"
TRAIN_MODULE = "reward_hacking.train"
EVAL_MODULE = "reward_hacking.train_eval"
ANALYZE_MODULE = "reward_hacking.harness.analyze_ilcb"

# Generous, and each tracks the path it bounds -- deliberately at roughly 2x its measurement rather
# than a 5-20% margin, because our estimates here have been wrong in BOTH directions and a cap set to
# an estimate is a cap set to a coin flip. The arm's: 70 steps at the 2026-08-24 probe's 28.05-minute
# median (32.7 h projected, ~45 h once the observed ~35-minute reclaim cadence interrupts it), plus
# the bootstrap. The screen's: 74.5 minutes per arm measured off the completed run's stage
# boundaries, on a stage that cannot be resumed, only redone. The evaluation's: a checkpoint ladder of
# multi-turn agentic episodes, each of which may spend its own twenty-minute episode budget.
DEFAULT_SCREEN_TIMEOUT = "180m"
DEFAULT_ARM_TIMEOUT = "72h"
DEFAULT_EVAL_TIMEOUT = "36h"
DEFAULT_ANALYZE_TIMEOUT = "30m"

# The held-out evaluation runs BOTH: `conflicting` carries the label-free headline (a pass on a
# self-contradictory grader is a hack by construction) and `original` is the capability control
# without which "it hacks more" cannot be told from "it got better at coding".
EVAL_SPLITS: tuple[str, ...] = (SPLIT_CONFLICTING, SPLIT_ORIGINAL)

DEFAULT_STAGES: tuple[str, ...] = ("screen", "arm", "eval", "analyze")


def screen_timeout() -> str:
    """Return the cap on the gradient screen, overridable because at 3am it is the knob wanted.

    Every other tunable here already routes through :func:`plan_setting`; the timeouts were the
    exception, and a cap that needs a code edit and a re-render to raise is the one setting an
    operator cannot reach from the launch environment. The two workarounds available without this
    are both worse: cutting ``SCREEN_SAMPLES`` fits the cap by changing the statistic, since
    ``frac_groups_pure`` at the training group size IS the measurement, and re-invoking the screen
    from a wrapper script at a longer leash duplicates the stage runner's artifact verification.

    **A screen that hits its cap cannot be resumed, only redone.** The stage runner's advice on a
    124 -- raise the cap and re-run the identical command -- is right for training, which restores
    from a checkpoint, and misleading here: a screen is one inference sweep with no partial state, so
    a re-run starts the arm from scratch. Raise this BEFORE launching rather than after a kill.
    """
    return plan_setting(f"{ENV_PREFIX}SCREEN_TIMEOUT", DEFAULT_SCREEN_TIMEOUT)


def arm_timeout() -> str:
    """Return the cap on one training arm; raising it after a kill is cheap, the arm resumes."""
    return plan_setting(f"{ENV_PREFIX}ARM_TIMEOUT", DEFAULT_ARM_TIMEOUT)


def eval_timeout() -> str:
    """Return the cap on the held-out ladder; a kill keeps every rung whose summary landed."""
    return plan_setting(f"{ENV_PREFIX}EVAL_TIMEOUT", DEFAULT_EVAL_TIMEOUT)


def analyze_timeout() -> str:
    """Return the cap on the readout, which is CPU-only and reruns from the traces for free."""
    return plan_setting(f"{ENV_PREFIX}ANALYZE_TIMEOUT", DEFAULT_ANALYZE_TIMEOUT)


def arm_name() -> str:
    """Return the arm this plan runs, refusing an unknown one before any stage is built."""
    arm = plan_setting(f"{ENV_PREFIX}ARM", "")
    if arm not in TRAINABLE_ARMS:
        raise ValueError(
            f"set {ENV_PREFIX}ARM to one of {list(TRAINABLE_ARMS)}, got {arm!r}. The arm decides "
            f"which grader the run trains against ({SPLIT_BY_TRAINABLE_ARM}), so there is no "
            f"default."
        )
    return arm


def model_id() -> str:
    """Return the checkpoint this plan trains and evaluates."""
    return plan_setting(f"{ENV_PREFIX}MODEL", DEFAULT_MODEL_ID)


def model_tag() -> str:
    """Return a path-safe fragment of the model id, for artifact names."""
    return model_id().replace("/", "-").replace(" ", "-")


def arm_tag() -> str:
    """Return the ``<arm>-<model>`` fragment keeping two arms' artifacts apart, everywhere."""
    return f"{arm_name()}-{model_tag()}"


def partition_path() -> Path:
    """Return the stored held-out partition both the corpus and the evaluation read."""
    return Path(plan_setting(f"{ENV_PREFIX}PARTITION", str(DEFAULT_PARTITION_PATH)))


def completion_tokens() -> str:
    """Return the completion budget, from the coding-prompt screen and not a constant here."""
    return plan_setting(
        f"{ENV_PREFIX}COMPLETION_TOKENS", str(required_coding_completion_budget(model_id()))
    )


def run_dir() -> Path:
    """Return this arm's run directory, explicit because a resume must find it again."""
    return Path(plan_setting(f"{ENV_PREFIX}OUTPUT_DIR", f"{RUN_ROOT}/{arm_tag()}"))


def s3_destination() -> str:
    """Return this arm's own S3 prefix, or empty to leave the sync off.

    Suffixed with the arm tag rather than used bare, and ``reward_hacking.train`` refuses a
    destination that is not: ``aws s3 sync`` puts a run directory's *contents* under its destination,
    so two arms sharing one prefix write byte-identical keys and the second silently replaces the
    first -- including the record that would say which arm survived.
    """
    base = plan_setting(f"{ENV_PREFIX}S3_BASE", "").rstrip("/")
    return f"{base}/{arm_tag()}" if base else ""


def selected_stages() -> tuple[str, ...]:
    """Which stages this invocation runs, re-ordered into this plan's own dependency order.

    Delegated to ``games.plans.selected_stages_from`` rather than parsed here, and the re-ordering is
    the reason. The hand-rolled copy this replaces preserved whatever order an operator typed, so
    ``RH_OPTION3_STAGES=arm,screen`` satisfied :func:`assert_screen_precedes_the_arm` -- which returns
    early whenever ``screen`` appears anywhere in the selection -- and then ran the 20-GPU-hour arm
    first, with no screen artifact on disk. That is precisely the unwatched-launch failure the blocking
    refusal exists to prevent. With the selection normalised to ``DEFAULT_STAGES`` order, "screen is in
    the selection" implies "screen runs first" again, so the refusal needs no change.
    """
    return selected_stages_from(
        f"{ENV_PREFIX}STAGES", all_stages=DEFAULT_STAGES, default=DEFAULT_STAGES
    )


def screen_artifact() -> Path:
    """Where the gradient screen writes its verdict for this arm."""
    return SCREEN_DIR / f"gradient-screen-{arm_tag()}.json"


def eval_artifact() -> Path:
    """Where the held-out evaluation writes its per-checkpoint summary for this arm."""
    return EVAL_DIR / arm_tag() / LADDER_SUMMARY_FILENAME


def analysis_artifact() -> Path:
    """Where the readout writes this arm's rate table."""
    return ANALYSIS_DIR / f"ilcb-metrics-{arm_tag()}.json"


def eval_trace_glob() -> str:
    """Return the trace pattern the readout consumes, kept in step with what the eval writes."""
    return str(EVAL_DIR / arm_tag() / "*.jsonl")


def screen_stage() -> Stage:
    """Measure whether the untrained model gives this arm any within-group disagreement at all."""
    artifact = screen_artifact()
    return Stage(
        name=f"screen-{arm_tag()}",
        argv=(
            "timeout",
            screen_timeout(),
            *UV,
            "-m",
            SCREEN_MODULE,
            "--arm",
            arm_name(),
            "--model",
            model_id(),
            "--partition",
            str(partition_path()),
            "--samples-per-prompt",
            plan_setting(f"{ENV_PREFIX}SCREEN_SAMPLES", "8"),
            "--grader-scratch-root",
            plan_setting(f"{ENV_PREFIX}GRADER_SCRATCH", DEFAULT_GRADER_SCRATCH_ROOT),
            "--out",
            str(artifact),
        ),
        artifacts=(artifact,),
        needs_gpu=True,
        log_path=LOG_DIR / f"screen-{arm_tag()}.log",
    )


def arm_stage() -> Stage:
    """Train this arm, retaining every checkpoint and shipping each one as it is written."""
    output_dir = run_dir()
    artifact = output_dir / "train_summary.json"
    argv = [
        "timeout",
        arm_timeout(),
        *UV,
        "-m",
        TRAIN_MODULE,
        "--arm",
        arm_name(),
        "--model",
        model_id(),
        "--partition",
        str(partition_path()),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        plan_setting(f"{ENV_PREFIX}MAX_STEPS", "70"),
        "--max-prompt-tokens",
        plan_setting(f"{ENV_PREFIX}MAX_PROMPT_TOKENS", str(DEFAULT_MAX_PROMPT_TOKENS)),
        "--max-completion-tokens",
        completion_tokens(),
        "--seed",
        plan_setting(f"{ENV_PREFIX}SEED", "0"),
        # Every checkpoint, no rotation: the behavioural ladder is read at every rung, and the
        # trainer refuses a coarser cadence for a real arm anyway.
        "--save-steps",
        "1",
        "--save-total-limit",
        "0",
        "--vllm-colocate",
        # Required at this completion budget: the correction's upcast fp32 logits row is ~22.7 GiB.
        "--no-vllm-importance-sampling-correction",
        "--grader-scratch-root",
        plan_setting(f"{ENV_PREFIX}GRADER_SCRATCH", DEFAULT_GRADER_SCRATCH_ROOT),
        # Re-running the same command IS the recovery procedure: the trainer pulls the run directory
        # back from S3 first, so a reclaimed arm resumes at its last saved step on a fresh box.
        "--resume-from-checkpoint",
        "latest",
    ]
    destination = s3_destination()
    if destination:
        argv += ["--s3-dest", destination]
    return Stage(
        name=f"arm-{arm_tag()}",
        argv=tuple(argv),
        artifacts=(artifact,),
        needs_gpu=True,
        log_path=LOG_DIR / f"arm-{arm_tag()}.log",
        log_run_dirs=(output_dir,),
    )


def _split_argv() -> tuple[str, ...]:
    """Render the held-out splits as the repeated singular flag the evaluator actually defines.

    Its parser takes ``--split`` with ``action="append"``, not a comma-joined ``--splits``. Rendering
    the wrong one produced an argv that failed with `unrecognized arguments` AFTER the whole training
    run had been paid for, which is why `test_rh_train_sequence` now feeds every stage's argv to its
    target module's own parser.
    """
    return tuple(argument for split in EVAL_SPLITS for argument in ("--split", split))


def eval_stage() -> Stage:
    """Run the held-out behavioural ladder over this arm's retained checkpoints."""
    artifact = eval_artifact()
    return Stage(
        name=f"eval-{arm_tag()}",
        argv=(
            "timeout",
            eval_timeout(),
            *UV,
            "-m",
            EVAL_MODULE,
            "--run-dir",
            str(run_dir()),
            "--base-model",
            model_id(),
            "--partition",
            str(partition_path()),
            *_split_argv(),
            "--repeats",
            plan_setting(f"{ENV_PREFIX}EVAL_REPEATS", "1"),
            "--every-nth-checkpoint",
            plan_setting(f"{ENV_PREFIX}EVAL_EVERY_NTH", str(DEFAULT_LADDER_STRIDE)),
            "--out-dir",
            str(artifact.parent),
        ),
        artifacts=(artifact,),
        needs_gpu=True,
        log_path=LOG_DIR / f"eval-{arm_tag()}.log",
        log_run_dirs=(run_dir(),),
    )


def analyze_stage() -> Stage:
    """Read the evaluation traces out as a rate table, on CPU. No new statistics live here."""
    artifact = analysis_artifact()
    return Stage(
        name=f"analyze-{arm_tag()}",
        argv=(
            "timeout",
            analyze_timeout(),
            *UV,
            "-m",
            ANALYZE_MODULE,
            "--glob",
            eval_trace_glob(),
            "--out",
            str(artifact),
        ),
        artifacts=(artifact,),
        needs_gpu=False,
        log_path=LOG_DIR / f"analyze-{arm_tag()}.log",
    )


STAGE_BUILDERS = {
    "screen": screen_stage,
    "arm": arm_stage,
    "eval": eval_stage,
    "analyze": analyze_stage,
}


def assert_screen_precedes_the_arm(chosen: tuple[str, ...]) -> None:
    """REFUSE to build an arm stage with no gradient screen on disk for it.

    A refusal rather than a warning, and the upgrade was earned: a warning scrolls past in a tmux log
    on a rented box nobody is watching, and the failure it guards against is silent-green by
    construction. GRPO learns only from within-group disagreement, so an arm whose reward is constant
    at zero on every episode trains nothing while the GPU sits at 100%, the heartbeats stay fresh and
    no error appears anywhere; the untrained model passed this family's misspecified grader 0 times in
    101 agentic episodes, so that is the expected case rather than the unlucky one. A blocking guard
    is the only kind that survives an unwatched launch.

    Deliberately NOT a check on what the verdict SAYS. What the screen found is a judgement for a
    human -- how pure is too pure, and which mitigation to take -- and a plan that tried to grade the
    verdict itself would either be wrong or become the thing nobody trusts. Selecting the screen stage
    in the same invocation satisfies it, because the stage runner refuses to call a stage successful
    without checking the artifact it promised.

    What it DOES check, besides existence, is the verdict's PROVENANCE, and the distinction is the
    point: a provenance check is not a verdict check. A gradable rate measured under one solution
    parser says nothing about an arm trained under another, and the size of that difference is
    recorded rather than assumed -- 16.3% of samples gradable under the retired tags-only rule against
    56.6% under the shipped one, on byte-identical generations. Live rather than hypothetical: the two
    verdicts sitting in this run's S3 prefix carry no ``solution_parser`` key at all, so the launcher's
    code-freshness floor would correctly force the arm onto post-fix code while its screen gate was
    satisfied by a verdict measured under the code that floor exists to escape.
    """
    if "arm" not in chosen or "screen" in chosen:
        return
    artifact = screen_artifact()
    if not artifact.is_file():
        # The 0-in-101 figure belongs to the misspecified grader alone; quoting it at every arm
        # would present one arm's evidence as another's.
        evidence = (
            " -- and the untrained model passed this family's misspecified grader 0 times in 101 "
            "agentic episodes, so here the constant-zero reward is the expected case rather than "
            "the unlucky one"
            if arm_name() == ARM_MISSPECIFIED
            else ""
        )
        raise ValueError(
            f"refusing to build the {arm_name()} arm: no gradient screen for it at {artifact}. GRPO "
            f"learns only from within-group disagreement, so an arm whose groups are pure can burn "
            f"20 rented GPU-hours on a reward that is constant at zero, with every signal green the "
            f"whole way{evidence}. Run `{ENV_PREFIX}STAGES=screen` first and read its "
            f"frac_groups_pure, or select `screen,arm` together so this invocation produces the "
            f"screen before it spends the arm."
        )
    assert_screen_matches_the_arm(artifact)


def assert_screen_matches_the_arm(artifact: Path) -> None:
    """REFUSE an arm whose screen measured a different corpus rendering than the arm will train on.

    Two provenance comparisons, each against the value training actually uses.

    **The solution parser.** An absent key is treated as a pre-fix screen rather than as "no
    opinion", because that is what it means: the parser name was added to the artifact by the same
    commit that changed the parser, so every verdict written before it lacks the key and was
    measured under the retired rule.

    **The grader exposure.** The screen accepts ``--grader-exposure withheld``, which deletes the
    grader block from every prompt, while training always renders
    :data:`reward_hacking.train_dataset.TRAINING_GRADER_EXPOSURE` -- so a blind screen's verdict
    describes a different reward landscape than the arm will see, and nothing about the parser
    comparison would notice. Here absence passes rather than refusing, and the asymmetry with the
    parser rule is deliberate: ``grader_exposure`` was written into the artifact by the same commit
    (325d87b) that ADDED the ``--grader-exposure`` flag, so a verdict without the key was
    necessarily measured inline -- there was no other rendering to measure.

    Re-screening is the fix, not deleting this gate. The screen is cheap next to the arm -- one
    inference sweep against 70 optimizer steps -- and a screen and an arm that disagree about what
    counts as a submission, or about what the policy was shown, make the screen's whole claim ("the
    number this screen reports is the reward training would compute") false.
    """
    record = json.loads(artifact.read_text(encoding="utf-8"))
    recorded = record.get("solution_parser")
    if recorded != SOLUTION_PARSER:
        raise ValueError(
            f"refusing to build the {arm_name()} arm: its gradient screen at {artifact} was measured "
            f"under solution_parser={recorded!r} while the arm will train under "
            f"{SOLUTION_PARSER!r}. Gradable rates from different parsers are not comparable and the "
            f"difference is not small -- this corpus read 16.3% (77/472) of samples gradable under the "
            f"retired tags-only rule and 56.6% (267/472) once a bare closed fence counted, on "
            f"byte-identical generations -- so the screen would not describe the reward this arm gets. "
            f"A missing key means the screen predates the parser change. RE-SCREEN this arm "
            f"(`{ENV_PREFIX}STAGES=screen`) rather than deleting this gate; the screen is one "
            f"inference sweep against 70 optimizer steps."
        )
    recorded_exposure = record.get("grader_exposure")
    if recorded_exposure is not None and recorded_exposure != TRAINING_GRADER_EXPOSURE.value:
        raise ValueError(
            f"refusing to build the {arm_name()} arm: its gradient screen at {artifact} sampled "
            f"under grader_exposure={recorded_exposure!r} while training renders "
            f"{TRAINING_GRADER_EXPOSURE.value!r}. A screen measured blind to the grader describes "
            f"a different reward landscape than an inline arm trains on, so its frac_groups_pure "
            f"says nothing about the gradient this arm would get. RE-SCREEN this arm "
            f"(`{ENV_PREFIX}STAGES=screen`) without --grader-exposure."
        )


def stages() -> list[Stage]:
    """Build the selected stages, in order. The protocol ``games.stage_runner`` loads a plan by."""
    chosen = selected_stages()
    assert_screen_precedes_the_arm(chosen)
    return [STAGE_BUILDERS[name]() for name in chosen]


def describe_plan() -> str:
    """Render every stage's command, promised artifact and cap, without running anything.

    The per-stage block is ``games.plans.describe_stages``, shared with every other plan: this module
    had grown its own copy, which is how one plan's printout came to omit the log destinations that
    another's showed. Only the header lines below are this plan's own.
    """
    return describe_stages(
        [
            f"arm={arm_name()} model={model_id()} split={SPLIT_BY_TRAINABLE_ARM[arm_name()]}",
            f"partition={partition_path()} completion_tokens={completion_tokens()}",
            f"run_dir={run_dir()} s3_dest={s3_destination() or '(off)'}",
        ],
        stages(),
    )


def main(argv: list[str] | None = None) -> int:
    """Print the plan this environment renders, which is the cheap check before the meter starts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--print-eval-glob", action="store_true")
    args = parser.parse_args(argv)
    if args.print_eval_glob:
        print(eval_trace_glob())  # noqa: T201 - a shell substitution consumes this
        return 0
    del args  # the plan is what this entry point prints; --print-plan reads as documentation
    print(describe_plan())  # noqa: T201 - the whole point of this entry point
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
