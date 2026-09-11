"""What every `stage_runner` plan shares: artifact roots, plan settings, slugs and wall-clock caps.

Three plan modules -- `games/arm_sequence.py`, `games/contrast_pair_sequence.py` and
`games/nine_b_sequence.py` -- had grown independent copies of the same skeleton, and the copies had
already drifted in three ways that this module exists to make unrepresentable:

*   **The screen directory had two values.** Two plans wrote screens to `artifacts/games/screens`
    while `games.screen_thinking` (the writer), `games.termination`'s provenance strings and every
    screen on disk use the singular `artifacts/games/screen`. A screen launched through a plan
    landed in a sibling directory nothing else reads, so a human checking for a fresh screen saw
    only the old ones.
*   **The arm cap had two values.** 60h in one plan and 36h in the plan that runs the *larger*
    model, against the argument both docstrings make: a cap under the healthy run time is strictly
    worse than no cap, because it kills a working arm when its artifacts are most expensive to
    recreate.
*   **`model_slug` meant two different things.** A zero-argument `model_slug()` here returned
    `qwen35-2b` while `games.train.model_slug(model_id)` returns `Qwen-Qwen3.5-2B`, and both feed
    artifact directory names in modules that call into each other. The plan-side one is named
    `artifact_model_slug` now, so the two cannot be confused at a call site.

Nothing about any *experiment* is written here: what an arm is comes from `games.arms`, and what "at
training temperature" means comes from `games.generation`. This module holds only the shape a plan
takes.

**Both of those live in import-cheap modules for this module's sake.** `--print-plan` is advertised
as the cheap CPU-only check to run before the meter starts, and it was reaching the trainer for the
colocate readers and `games.select_prompts` for the sampler, so rendering a command imported torch,
transformers, trl, peft, matplotlib and pandas, 8.4 s of it. `test_games_plans.py` fails if any of
that comes back.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from games.generation import (
    TRAINING_TEMPERATURE,
    TRAINING_TOP_K,
    TRAINING_TOP_P,
    VLLM_IS_CORRECTION_ENV,
    assert_vllm_rollouts,
    colocate_gpu_fraction,
    colocate_importance_sampling_correction,
)
from games.rewards import GRADING_VS_FIXED_MIX
from games.stage_runner import Stage
from games.termination import required_completion_budget
from grpo.estimator_defaults import VLLM_IMPORTANCE_SAMPLING_MODE, VLLM_IMPORTANCE_SAMPLING_MODES

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from games.arms import GameArm

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "artifacts" / "games"
RUN_DIR = ARTIFACTS / "runs"
LOG_DIR = ARTIFACTS / "logs"
SELECT_DIR = ARTIFACTS / "select"
THROUGHPUT_DIR = ARTIFACTS / "throughput"
# Singular, matching `games.screen_thinking.SCREEN_ROOT` and every screen already on disk.
SCREEN_DIR = ARTIFACTS / "screen"

UV: tuple[str, ...] = ("uv", "run", "--frozen", "python")

# Shared shape: ten-plus checkpoints for the eval ladder, at the measured group and prompt counts.
DEFAULT_MAX_STEPS = "70"
DEFAULT_SAVE_STEPS = "5"
DEFAULT_GROUP = "8"
DEFAULT_PROMPTS_PER_STEP = "8"
DEFAULT_SAMPLES_PER_PROMPT = "8"
# The Qwen3.5 cards' recommended output budget; anything below it is us, not them.
DEFAULT_SCREEN_BUDGET = "32768"

# Wall-clock caps, generous on purpose; the resolvers below carry each one's provenance.
SCREEN_TIMEOUT = "90m"
THROUGHPUT_TIMEOUT = "60m"
SWEEP_TIMEOUT_BY_BACKEND: dict[str, str] = {"hf": "12h", "vllm": "6h"}
COLOCATE_GENERATION = "vllm-colocate"
# The only training cap there is: rollouts are vLLM-only (games.generation.VLLM_ONLY_RATIONALE),
# so the 60h transformers.generate cap left with the backend choice that needed it.
COLOCATE_ARM_TIMEOUT = "36h"


def plan_setting(name: str, default: str) -> str:
    """Read one plan setting from the environment.

    Every plan reads its settings this way rather than being edited in place, because a plan runs on
    a rented instance an operator exports variables into and then re-runs -- and re-running the
    identical command is also the recovery procedure after an interruption.
    """
    return os.environ.get(name, default)


def artifact_model_slug(model_id: str) -> str:
    """Turn a hub id into the short tag plan artifacts are named with (`qwen35-2b`).

    Distinct from `games.train.model_slug`, which keeps the vendor and the dots
    (`Qwen-Qwen3.5-2B`). Both feed directory names, so the names say which one is which.
    """
    return model_id.rsplit("/", maxsplit=1)[-1].replace(".", "").lower()


def screen_artifact(*, model_tag: str, budget: str, game_id: str | None = None) -> Path:
    """Name a termination screen's JSON, keyed by whose prompts it screened and at what budget.

    One template for every plan: the budget is in the name because the same model gets screened at
    several, and the game is in it when a plan screens that game's own prompts, since the unilateral
    and iterated shapes ask for a different output than the matrix games do.
    """
    stem = f"{game_id}-{model_tag}" if game_id else model_tag
    return SCREEN_DIR / f"{stem}-termination-{budget}.json"


def cap_hours(cap: str) -> float:
    """Read a `timeout(1)` duration like "36h" or "90m" as hours, for comparing two caps."""
    scale = {"h": 1.0, "m": 1.0 / 60.0, "s": 1.0 / 3600.0}[cap[-1]]
    return float(cap[:-1]) * scale


def sweep_timeout_for(backend_kind: str) -> str:
    """Return the measured sweep cap for a backend, falling back to the slowest one we have.

    Falling back rather than raising: this number bounds a wedge, `games.select_prompts` gains a
    local backend kind occasionally, and "the new kind runs under the old kind's cap" is the right
    default for that day.
    """
    if backend_kind not in SWEEP_TIMEOUT_BY_BACKEND:
        logger.warning(
            f"no measured sweep cap for backend {backend_kind!r}; using the HF path's "
            f"{SWEEP_TIMEOUT_BY_BACKEND['hf']}, which is the slowest measured. Measure this "
            f"backend and add it to SWEEP_TIMEOUT_BY_BACKEND."
        )
        return SWEEP_TIMEOUT_BY_BACKEND["hf"]
    return SWEEP_TIMEOUT_BY_BACKEND[backend_kind]


def generation_backend() -> str:
    """Name what `games.train` will generate rollouts through, refusing a box that asks otherwise.

    There is one answer since 2026-08-26 -- rollouts are vLLM-only -- so what this contributes to
    a plan render is the refusal: `assert_vllm_rollouts` fires here, at `--print-plan` time,
    before the meter starts, on exactly the stray-export class that once put a paid arm on
    transformers.generate.
    """
    assert_vllm_rollouts()
    return COLOCATE_GENERATION


def default_arm_timeout() -> str:
    """Return the training cap for the colocated-vLLM backend, the only one a plan can train on."""
    generation_backend()
    return COLOCATE_ARM_TIMEOUT


def colocate_argv() -> tuple[str, ...]:
    """Render the engine knobs a training stage carries, for every plan that trains.

    Rendered into the argv even though `games.train` would read the same variables on its own,
    because `--print-plan` is the cheapest check on a launch before the meter starts and a printed
    command that does not say what the engine will hold is a check that reads green while saying
    nothing. The values come from `games.train`'s own environment readers, so what is printed is
    what the trainer will do. There is no backend flag to render: rollouts are vLLM-only, and a
    box asking otherwise is refused right here.
    """
    assert_vllm_rollouts()
    argv = ["--vllm-gpu-memory-utilization", str(colocate_gpu_fraction())]
    if not colocate_importance_sampling_correction():
        argv.append("--no-vllm-importance-sampling-correction")
    return tuple(argv)


ESTIMATOR_PIN_ENV = "GAMES_ESTIMATOR_PIN"
FLAGSHIP_ESTIMATOR_PIN = "flagship-dapo-batch-mb1"
# One entry per already-trained set of arms a later run can be made magnitude-comparable to.
ESTIMATOR_PINS: dict[str, tuple[str, ...]] = {
    FLAGSHIP_ESTIMATOR_PIN: (
        "--loss-type",
        "dapo",
        "--scale-rewards",
        "batch",
        "--acknowledge-liger-estimator-mismatch",
        "--micro-batch-size",
        "1",
    ),
}


def estimator_argv() -> tuple[str, ...]:
    """Render the estimator an arm must train under to be comparable to an already-trained arm.

    The repo defaults (`dr_grpo`/`none`, reasoned out in `grpo/estimator_defaults.py`) are the
    faithful choice and stay the default here. The flagship twin-PD pair predates that decision and
    executed `dapo`/`batch` at micro-batch 1, and comparing gradient *magnitudes* across arms is the
    registered experiment -- so an arm left on the defaults measures roughly 5-10x less gradient per
    step at equal learning rate (the effective-LR note in the same module) and is not comparable to
    the pair at all. Naming a pin is how a run asks to be comparable, and the four flags are one
    decision rather than four knobs: which arms this one is being read against.

    Unset or empty renders nothing, so every plan that does not ask for a pin keeps the argv it had.

    **This replaces a block patch, which is the reason it is tracked code.** Every launch under the
    pair's estimator so far rewrote `build_arm_stage`'s body on the rented box, string-matching the
    `*colocate_argv()` line and splicing four flags in after it -- re-applied on every reclaim
    recovery, because a recovery re-extracts the pinned tarball and would otherwise drop back to the
    defaults. An anchor that drifts by one comment character fails the launch; a matcher permissive
    enough not to drift trains something other than what was asked for. An unknown value raises here
    instead, at `--print-plan` time, which is the cheap CPU-only check before the meter starts.

    All three plans inherit this with no per-plan change: `games/arm_sequence.py`,
    `games/contrast_pair_sequence.py` and `games/nine_b_sequence.py` each build their training stage
    through `build_arm_stage` and add nothing to its argv.
    """
    pin = plan_setting(ESTIMATOR_PIN_ENV, "")
    if not pin:
        return ()
    if pin not in ESTIMATOR_PINS:
        raise ValueError(
            f"unknown {ESTIMATOR_PIN_ENV}={pin!r}; known pins: {sorted(ESTIMATOR_PINS)}. A pin "
            f"names the already-trained arms this run is being made magnitude-comparable to, so a "
            f"value nobody has written down is a comparison nobody has decided on. Unset the "
            f"variable to train under the repo defaults in grpo/estimator_defaults.py."
        )
    return ESTIMATOR_PINS[pin]


def refuse_sweep_beside_a_corpus_consumer(
    chosen: Sequence[str], *, sweep: str, consumers: Sequence[str], recipe: str
) -> None:
    """Refuse a plan that sweeps and then reads the swept corpus in the same invocation.

    `games.select_prompts` timestamps its corpus, so a stage that consumes one resolves its path
    before the sweep that writes it has run. Both outcomes of allowing it are wrong and only one is
    loud: on a clean box the resolver raises before the sweep it was about to run, so the plan's own
    default could never succeed; with a corpus from an earlier sweep already present, the plan pins
    the stale one, runs a fresh five-to-twelve-hour sweep whose output nothing reads, and leaves two
    non-empty corpora behind so every later resolve is refused as ambiguous.

    Refusing rather than reordering, because the corpus path cannot be known at build time at all --
    which is why both plans' docstrings already tell an operator to run two invocations.
    """
    consuming = [name for name in chosen if name in consumers]
    if sweep in chosen and consuming:
        raise ValueError(
            f"this plan selects {sweep!r} together with {consuming}, which takes two invocations "
            f"rather than one: games.select_prompts names its corpus with a timestamp, so the "
            f"consuming stage would resolve a path before the sweep has written one -- and if an "
            f"older corpus is present it would train on that while the fresh sweep's output goes "
            f"unread. Run the sweep alone, resolve the corpus, then run the rest:\n{recipe}"
        )


def training_sampler_argv() -> tuple[str, ...]:
    """Render the decoding flags that make a baseline sweep sample the policy that gets trained.

    Read from `games.generation` rather than written as three literals per plan. "At training
    temperature" means GRPO's own generation defaults, and the whole contrast rests on the sweep
    matching the trainer: two plans and a shell script each carried their own copy of
    `1.0 / 1.0 / 0`, so retuning any one source left every sweep selecting prompts off-policy while
    the flags still looked explicit and correct. `games.train.GameTrainConfig` and
    `games.select_prompts.training_sampler` read the same three constants, so there is now one place
    to retune and no copy left to drift.

    Model-independent, unlike the completion budget: a plan lets an operator override the budget and
    `--max-new-tokens` is where that override lands, while these three are the same on every
    checkpoint.
    """
    return (
        "--temperature",
        str(TRAINING_TEMPERATURE),
        "--top-p",
        str(TRAINING_TOP_P),
        "--top-k",
        str(TRAINING_TOP_K),
    )


def selected_stages_from(
    env_name: str, *, all_stages: tuple[str, ...], default: tuple[str, ...]
) -> tuple[str, ...]:
    """Read a comma-separated stage selection from one variable, in plan order.

    Naming a subset is how a sweep runs on its own before its corpus can be resolved, and how one
    stage is restarted after an interruption -- both on a rented box, from a shell, without editing
    a plan. The result is re-ordered into `all_stages` order rather than the order asked for,
    because an operator restarting two stages types them in the order they come to mind and a plan's
    order is a dependency order.

    Three plans each carried this loop. The copies had not drifted yet, which is the only reason
    there is nothing worse than duplication to record here.
    """
    requested = plan_setting(env_name, ",".join(default))
    names = tuple(name.strip() for name in requested.split(",") if name.strip())
    unknown = [name for name in names if name not in all_stages]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown} in {env_name}; known: {list(all_stages)}")
    if not names:
        raise ValueError(f"{env_name} selected nothing; known stages: {list(all_stages)}")
    return tuple(name for name in all_stages if name in names)


def derived_completion_tokens(env_name: str, *, model_id: str) -> str:
    """Resolve a plan's completion budget from the model it is pointed at, or from the shell.

    Derived at call time rather than written down, because the model is env-overridable so the
    budget has to follow it. A literal drifts, and in `games/nine_b_sequence.py` it already had:
    that file said 16,384 while the 9B screen measured only 83% of rollouts closing their thinking
    by then, so both the arm and the throughput probe would have run in the format-dominated regime
    and the cost estimate would have priced the wrong configuration.
    """
    return plan_setting(env_name, str(required_completion_budget(model_id)))


def arm_s3_destination(env_name: str, *, arm: str, model_tag: str) -> str:
    """Return the S3 prefix one arm ships to, or empty to leave the sync off.

    Suffixed with the arm and model rather than used bare: two arms syncing into one prefix would
    interleave their checkpoints and the trace could not be attributed afterwards. That matters most
    in the plan that runs two arms from one invocation, which is where the suffix was written first.
    """
    base = plan_setting(env_name, "").rstrip("/")
    return f"{base}/{arm}-{model_tag}" if base else ""


# transformers' own `SchedulerType` values, copied rather than imported: `--print-plan` is
# advertised as the cheap CPU-only check before the meter starts, and importing transformers here
# would put the whole training stack back into it (this module's docstring, and the probe in
# `test_games_plans.py`). That same test file asserts this set still equals
# `transformers.trainer_utils.SchedulerType`, so a version bump that adds or renames a schedule
# cannot pass silently; the test may import transformers because a test run is not a plan render.
LR_SCHEDULER_NAMES: frozenset[str] = frozenset(
    {
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
        "inverse_sqrt",
        "reduce_lr_on_plateau",
        "cosine_with_min_lr",
        "cosine_warmup_with_min_lr",
        "warmup_stable_decay",
        "greedy",
    }
)


def assert_known_lr_scheduler(env_name: str, value: str) -> None:
    """Refuse a schedule name transformers would not recognise, at plan time.

    `TrainingArguments` raises on an unknown `lr_scheduler_type`, but it raises on the rented box
    after the bootstrap, the model download and the corpus restore; the same typo costs nothing here.
    """
    if value not in LR_SCHEDULER_NAMES:
        raise ValueError(
            f"unknown {env_name}={value!r}; transformers accepts {sorted(LR_SCHEDULER_NAMES)}. The "
            f"schedule is the treatment (a cosine run cannot be extended without becoming a "
            f"different experiment), so a name nobody recognises is a launch nobody decided on."
        )


# Named once because two places need it: the knob table's row, and the coupling check that reads the
# rendered flag back to decide whether the correction has to be on for it to mean anything.
VLLM_IS_MODE_FLAG = "--vllm-importance-sampling-mode"


def assert_known_importance_sampling_mode(env_name: str, value: str) -> None:
    """Refuse a correction mode TRL would not recognise, at plan time.

    TRL takes the name without checking it and raises only inside the first generation batch's ratio
    arithmetic (`grpo_trainer.py:2690-2693`), which on a rented box is after the bootstrap, the model
    pull and a full generation pass. `games.train` refuses the same value in `__post_init__`; this
    catches it before the meter starts, the way `assert_known_lr_scheduler` does.
    """
    if value not in VLLM_IMPORTANCE_SAMPLING_MODES:
        raise ValueError(
            f"unknown {env_name}={value!r}; TRL 1.10 accepts "
            f"{sorted(VLLM_IMPORTANCE_SAMPLING_MODES)}. The mode is the estimator -- sequence-level "
            f"masking discards a whole rollout where token-level truncation clips each token -- so a "
            f"name nobody recognises is a launch nobody decided on."
        )


def assert_numeric_setting(env_name: str, value: str) -> None:
    """Refuse a knob whose value is not a number, at plan time rather than at argparse on the box."""
    try:
        float(value)
    except ValueError as error:
        raise ValueError(
            f"{env_name}={value!r} is not a number, and games.train parses it as a float. Caught "
            f"here because the kit exports these as strings and argparse would only refuse it after "
            f"the instance is running."
        ) from error


def assert_positive_integer_setting(env_name: str, value: str) -> None:
    """Refuse a count that is not a whole number above zero, at plan time.

    `argparse`'s `type=int` takes "2" and refuses "2.0" and "two", both of which a kit can export by
    hand, and `games.train` refuses a zero or negative oversample in `__post_init__`. All three are
    the box's own refusals, minutes of bootstrap after this one.
    """
    if not value.isdigit() or int(value) < 1:
        raise ValueError(
            f"{env_name}={value!r} is not a whole number above zero, and games.train parses it as "
            f"an int. A count of groups to generate per step cannot be fractional, zero or negative."
        )


# One entry per training knob a plan renders ONLY when its own environment sets it: the variable
# suffix read under the plan's prefix, the `games.train` flag it becomes, and the plan-time check its
# value has to pass. Unset renders nothing, which is the point: every plan and kit written before a
# knob existed keeps the argv it always had and trains under games/train.py's own argparse default,
# so no banked arm's treatment moves because a knob was added here.
OPTIONAL_TRAINING_KNOBS: tuple[tuple[str, str, Callable[[str, str], None]], ...] = (
    ("PARSE_PENALTY", "--parse-penalty", assert_numeric_setting),
    ("LR_SCHEDULER", "--lr-scheduler", assert_known_lr_scheduler),
    ("WARMUP_RATIO", "--warmup-ratio", assert_numeric_setting),
    ("ADAM_EPSILON", "--adam-epsilon", assert_numeric_setting),
    ("LORA_DROPOUT", "--lora-dropout", assert_numeric_setting),
    (
        "VLLM_IMPORTANCE_SAMPLING_MODE",
        VLLM_IS_MODE_FLAG,
        assert_known_importance_sampling_mode,
    ),
    (
        "DYNAMIC_SAMPLING_OVERSAMPLE",
        "--dynamic-sampling-oversample",
        assert_positive_integer_setting,
    ),
)


def optional_training_knobs(env_prefix: str) -> tuple[tuple[str, str], ...]:
    """Resolve the optional knobs one plan's environment sets, as (flag, value) pairs in table order.

    Validated as they are read, so a misspelled schedule or a mistyped exponent fails at
    `--print-plan` rather than on the box. An unset or empty variable is absent rather than a
    default: the default lives in `games/train.py`'s argparse and nowhere else.

    The correction mode carries one coupling of its own, checked after the loop the way
    `optional_training_switches` checks the instrument pair's: away from TRL's default it names a
    treatment `games.train` refuses with the correction off, because there is no ratio to shape then.
    The wave-4b training role is the launch that exports the correction off, so a kit that set the
    mode for it would otherwise discover the refusal on the box.
    """
    resolved: list[tuple[str, str]] = []
    for suffix, flag, assert_valid in OPTIONAL_TRAINING_KNOBS:
        env_name = f"{env_prefix}{suffix}"
        value = plan_setting(env_name, "")
        if not value:
            continue
        assert_valid(env_name, value)
        if (
            flag == VLLM_IS_MODE_FLAG
            and value != VLLM_IMPORTANCE_SAMPLING_MODE
            and not colocate_importance_sampling_correction()
        ):
            raise ValueError(
                f"{env_name}={value} renders a mode games.train refuses with the correction off, "
                f"and {VLLM_IS_CORRECTION_ENV}=0 turns the correction off. The mode only decides "
                f"what the correction does with the ratio, so with no ratio computed it would record "
                f"a treatment the run never ran. Unset it for a correction-off arm, or drop "
                f"{VLLM_IS_CORRECTION_ENV}=0 for the corrected one."
            )
        resolved.append((flag, value))
    return tuple(resolved)


def assert_switch_on(env_name: str, value: str) -> None:
    """Refuse any value but a bare "1" for a switch, at plan time.

    Pinned to one spelling rather than read as a boolean, following `GAMES_VLLM_COLOCATE`: a kit
    that exports `=0` to mean off and gets on, or `=true` and gets a refusal on the box after the
    bootstrap, is the drift these tables exist to stop. Off is the variable unset.
    """
    if value != "1":
        raise ValueError(
            f"{env_name}={value!r} is not a switch value: set it to exactly 1 to render the flag, or "
            f"leave it unset to render nothing. No other spelling is read, so a value meant as "
            f'"off" cannot be taken as on.'
        )


# One entry per valueless training flag a plan renders ONLY where its own environment sets the
# variable to exactly "1": the variable suffix read under the plan's prefix, the `games.train` flag it
# becomes, and whether the trainer refuses that flag without the vLLM importance-sampling correction.
# Apart from `OPTIONAL_TRAINING_KNOBS` because a switch renders one argv token where a knob renders
# two. The instrument pair makes the vLLM-versus-trainer mismatch measurable without changing what a
# run trains -- log-only computes TRL's correction ratio and weights the gradient by exactly one, and
# the fp32 head is what makes that ratio worth reading -- and neither means anything with the
# correction off, which is what the third column carries. The short-completions row is the escape
# hatch from the measured completion floor, for a plumbing smoke or a timing probe; it is
# correction-independent, and the column exists because of it, since the smoke of a training role runs
# the correction OFF and a refusal keyed on "some switch is set" would refuse the launch it is for.
OPTIONAL_TRAINING_SWITCHES: tuple[tuple[str, str, bool], ...] = (
    ("IS_LOG_ONLY", "--vllm-importance-sampling-log-only", True),
    ("CAST_LM_HEAD_FP32", "--cast-lm-head-to-fp32", True),
    ("ALLOW_SHORT_COMPLETIONS", "--allow-short-completions", False),
)


def optional_training_switches(env_prefix: str) -> tuple[str, ...]:
    """Resolve the switches one plan's environment sets to "1", as flags in table order.

    An instrument switch beside the correction OFF is refused here, naming both variables:
    `games.train` refuses the same pair, but on the rented box after the bootstrap, the model pull and
    the corpus restore, while a kit that copies the probe's exports into the training role costs
    nothing to catch at `--print-plan`. Only the rows whose flag the trainer refuses there are checked,
    so `ALLOW_SHORT_COMPLETIONS` renders beside a correction-off arm, which is what its own smoke is.
    """
    resolved: list[str] = []
    correction_coupled: list[str] = []
    for suffix, flag, needs_correction in OPTIONAL_TRAINING_SWITCHES:
        env_name = f"{env_prefix}{suffix}"
        value = plan_setting(env_name, "")
        if not value:
            continue
        assert_switch_on(env_name, value)
        resolved.append(flag)
        if needs_correction:
            correction_coupled.append(env_name)
    if correction_coupled and not colocate_importance_sampling_correction():
        names = ", ".join(correction_coupled)
        raise ValueError(
            f"{names} renders a flag games.train refuses with the correction off, and "
            f"{VLLM_IS_CORRECTION_ENV}=0 turns the correction off. The ratio log-only exists to log "
            f"is only computed when the correction runs, and the fp32 head alone moves the "
            f"sampler-versus-trainer mismatch rather than measuring it. Unset the switches for a "
            f"correction-off arm, or drop {VLLM_IS_CORRECTION_ENV}=0 for the instrumented probe."
        )
    return tuple(resolved)


@dataclass(frozen=True, slots=True)
class TrainingShape:
    """The per-run knobs a training stage renders, resolved from one plan's environment.

    One object rather than a `plan_setting` call per knob per plan, because three plans read these
    same variables under three prefixes and the completion budget among them is derived rather than
    defaulted. Strings, not ints: they go straight into an argv, and parsing them here only to
    render them back would be a second place for a default to live.

    The five required fields are always rendered. `optional_knobs` and `optional_switches` carry
    the ones this plan's environment asked for and nothing else, so a plan that sets none renders the
    argv it always did.
    """

    max_steps: str
    save_steps: str
    num_generations: str
    prompts_per_step: str
    completion_tokens: str
    optional_knobs: tuple[tuple[str, str], ...] = ()
    optional_switches: tuple[str, ...] = ()

    def optional_argv(self) -> tuple[str, ...]:
        """Flatten the set knobs and switches into argv tokens, and render nothing for the unset ones."""
        return (*(token for knob in self.optional_knobs for token in knob), *self.optional_switches)


def training_shape(env_prefix: str, *, completion_tokens: str) -> TrainingShape:
    """Read one plan's training shape from its own variable prefix (`GAMES_PAIR_`, say)."""
    return TrainingShape(
        max_steps=plan_setting(f"{env_prefix}MAX_STEPS", DEFAULT_MAX_STEPS),
        save_steps=plan_setting(f"{env_prefix}SAVE_STEPS", DEFAULT_SAVE_STEPS),
        num_generations=plan_setting(f"{env_prefix}GROUP", DEFAULT_GROUP),
        prompts_per_step=plan_setting(f"{env_prefix}PROMPTS_PER_STEP", DEFAULT_PROMPTS_PER_STEP),
        completion_tokens=completion_tokens,
        optional_knobs=optional_training_knobs(env_prefix),
        optional_switches=optional_training_switches(env_prefix),
    )


def build_screen_stage(  # noqa: PLR0913  -- each argument is a real axis some plan varies
    *,
    model_id: str,
    budget: str,
    timeout: str,
    artifact: Path,
    log_path: Path,
    log_run_dirs: tuple[Path, ...],
    game_id: str | None = None,
    grading: str | None = None,
) -> Stage:
    """Screen whether a model stops thinking inside a budget. Generation only, and cheap.

    It matters most where the prompt shape differs from the plain matrix games -- the unilateral
    split and the iterated arm ask for a different output -- because a model that never closes its
    thinking block leaves no visible answer, every completion scores as a parse failure, and the
    reward's whole-batch raise stops the run. That is the right outcome, learned far better in
    ninety minutes of generation than after a card has been reserved for a day.

    `game_id`/`grading` are optional because a plan may screen a model on its own arm's prompts or
    on the screener's defaults; the 9B plan screens before it has an arm's corpus at all.
    """
    game_argv = ("--game", game_id) if game_id else ()
    grading_argv = ("--grading", grading) if grading else ()
    subject = f"{game_id} @ {model_id}" if game_id else model_id
    return Stage(
        name=f"termination screen {subject} @ {budget}",
        argv=(
            "timeout",
            timeout,
            *UV,
            "-m",
            "games.screen_thinking",
            "--model",
            model_id,
            *game_argv,
            *grading_argv,
            "--budgets",
            budget,
            "--prompts",
            "2",
            "--samples",
            "4",
            "--save-completions",
            "--json-out",
            str(artifact),
        ),
        artifacts=(artifact,),
        needs_gpu=True,
        log_path=log_path,
        log_run_dirs=log_run_dirs,
    )


def build_sweep_stage(  # noqa: PLR0913  -- each argument is a real axis some plan varies
    *,
    game_id: str,
    grading: str,
    model_id: str,
    backend_kind: str,
    samples_per_prompt: str,
    out_dir: Path,
    max_new_tokens: str,
    timeout: str,
    log_path: Path,
    log_run_dirs: tuple[Path, ...],
    extra_argv: tuple[str, ...] = (),
) -> Stage:
    """Sweep the baseline policy and select the prompts whose behaviour is mixed.

    Selection is what makes GRPO able to learn at all: a prompt where every sample plays the same
    action has no within-group reward spread, so its advantages are zero and it contributes no
    gradient. Run at the training sampler (GRPO's own generation defaults, from `games.generation`)
    and with thinking on, because a sweep of a different policy than the one that gets trained
    selects the wrong prompts.

    This is the one stage that promises no artifact, which in this repo normally means a stage that
    can exit 0 having written nothing and be believed. It is unavoidable rather than an oversight:
    `games.select_prompts` timestamps all three of its outputs, so no path is knowable before the
    run, and promising the directory would be a check that passes on an empty directory -- worse
    than none. The verification is deferred instead to `resolve_single_corpus`, which the next
    invocation runs and which refuses an absent, empty or ambiguous corpus.

    `extra_argv` is where the frozen-opponent flags land, the one sweep argument that is per arm
    rather than per plan.
    """
    return Stage(
        name=f"baseline sweep {game_id}/{grading} @ {model_id} via {backend_kind}",
        argv=(
            "timeout",
            timeout,
            *UV,
            "-m",
            "games.select_prompts",
            "--backend",
            backend_kind,
            "--game",
            game_id,
            "--grading",
            grading,
            "--model",
            model_id,
            "--samples-per-prompt",
            samples_per_prompt,
            "--out-dir",
            str(out_dir),
            "--thinking",
            *training_sampler_argv(),
            "--max-new-tokens",
            max_new_tokens,
            *extra_argv,
        ),
        needs_gpu=True,
        log_path=log_path,
        log_run_dirs=log_run_dirs,
    )


def build_regrade_stage(  # noqa: PLR0913  -- each argument is a real axis some plan varies
    *,
    source: Path,
    target: Path,
    from_grading: str,
    to_grading: str,
    log_path: Path,
    log_run_dirs: tuple[Path, ...],
) -> Stage:
    """Re-grade a swept corpus into another grading, on CPU.

    `games.regrade_corpus` proves prompt-independence generatively before it writes: it renders both
    gradings and requires that only the `grading` column differs. That check is the whole validity
    condition for reusing one sweep across two gradings -- the invariant the twin-PD contrast rests
    on -- so it belongs in the pipeline rather than in a reviewer's head.
    """
    return Stage(
        name=f"regrade corpus {from_grading} -> {to_grading}",
        argv=(
            *UV,
            "-m",
            "games.regrade_corpus",
            "--corpus",
            str(source),
            "--grading",
            to_grading,
            "--out",
            str(target),
        ),
        artifacts=(target,),
        needs_gpu=False,
        log_path=log_path,
        log_run_dirs=log_run_dirs,
    )


def build_arm_stage(  # noqa: PLR0913  -- each argument is a real axis some plan varies
    *,
    arm: str,
    model_id: str,
    corpus: Path,
    output_dir: Path,
    timeout: str,
    shape: TrainingShape,
    s3_dest: str,
    log_path: Path,
    log_run_dirs: tuple[Path, ...],
    init_adapter: str = "",
) -> Stage:
    """Build one training arm: resumable, thinking-on, and checkpointing often.

    `--resume-from-checkpoint latest` with an explicit `--output-dir` is what makes re-running the
    identical command the whole recovery procedure. Never `latest` without the explicit directory:
    the default run directory is timestamped per launch, so a restart would look in a new empty one,
    find nothing, and begin again at step 0 while reporting success.

    Three plans built this argv independently, which is three chances for a flag renamed upstream to
    be caught in one of them and missed in the others -- and `--max-completion-tokens` (training)
    and `--max-new-tokens` (the sweep) are already two different flags on two different modules.
    Callers are expected to have run `assert_corpus_will_exist` first; that guard is separate
    because what counts as "a stage in this plan will write it" is a plan-level fact.
    """
    return Stage(
        name=f"train {arm} @ {model_id}",
        argv=(
            "timeout",
            timeout,
            *UV,
            "-m",
            "games.train",
            "--arm",
            arm,
            "--model",
            model_id,
            "--corpus",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--resume-from-checkpoint",
            "latest",
            "--max-steps",
            shape.max_steps,
            "--save-steps",
            shape.save_steps,
            "--num-generations",
            shape.num_generations,
            "--prompts-per-step",
            shape.prompts_per_step,
            "--max-completion-tokens",
            shape.completion_tokens,
            # The parse penalty, the learning-rate schedule and warmup, the two treatment knobs of
            # the wave-4b arm (optimizer epsilon, adapter dropout), the two instrument switches its
            # probe runs (log-only correction, fp32 head), and the short-completions escape hatch its
            # plumbing smoke needs. Each renders only where the plan's environment set it, so an
            # unset plan's argv is the argv it always had.
            *shape.optional_argv(),
            # The transfer-of-learning knob: seed the arm's LoRA from a checkpoint (games.train
            # verifies it against the run's own LoRA plan and records it). Empty renders nothing,
            # so every existing plan's argv is unchanged.
            *(("--init-adapter", init_adapter) if init_adapter else ()),
            # Both arms of a contrast pair take the same generation backend and the same estimator
            # setting from one environment, which is what keeps the contrast a contrast.
            *colocate_argv(),
            # Empty unless GAMES_ESTIMATOR_PIN names a pin, so an unpinned plan's argv is unchanged.
            *estimator_argv(),
        ),
        artifacts=(output_dir / "train_summary.json",),
        needs_gpu=True,
        log_path=log_path,
        log_run_dirs=log_run_dirs,
        env={"GAMES_S3_DEST": s3_dest} if s3_dest else None,
    )


# What marks a corpus as a regrade derivative rather than a sweep's own output; see `regraded_copy`.
REGRADE_MARKER = "-regraded-"


def resolve_single_corpus(
    directory: Path, *, corpus_env: str, sweep_command: str, sweep_log: Path
) -> Path:
    """Find the one corpus a sweep wrote into `directory`, refusing to guess between candidates.

    `games.select_prompts` timestamps its filenames, so the path cannot be known in advance. Every
    failure mode here is one that has actually happened in this repo or was one rename away, which
    is why each is a separate raise with its own explanation:

    *   **No directory.** The sweep never ran, or wrote somewhere else.
    *   **No corpus.** The sweep died. Raising beats returning a path that does not exist and
        letting `games.train` open it.
    *   **An empty corpus.** A 0-byte corpus is a plausible newest match, and training on it is a
        null result that looks like a real one. One is sitting in the shared selection directory.
    *   **More than one corpus.** Two sweeps in one directory means the operator has a choice to
        make and the plan does not know which run this is. "Newest" is the kind of guess that is
        right until the day it silently is not.

    A re-graded corpus is skipped rather than counted, because it is a derivative of the corpus in
    the same directory rather than a second sweep: it is written beside its source and its name
    still begins `corpus-`, so once a regrade had run every later resolve in that arm's directory
    was refused as ambiguous -- blocking `--print-corpus` and the arm stage itself in any shell
    without the corpus variable exported.

    Shared by every plan that sweeps rather than written per plan: the globbing shortcut this
    replaces is the one that trained an arm on nothing while reporting success, and two copies of
    that judgement is one copy too many. The three keyword arguments are message glue -- which
    variable overrides the resolution, which command re-runs the sweep, which log holds the reason.
    """
    if not directory.is_dir():
        raise FileNotFoundError(
            f"sweep directory {directory} does not exist, so the sweep never ran. Run {sweep_command} first."
        )
    candidates = [
        path for path in sorted(directory.glob("corpus-*.jsonl")) if REGRADE_MARKER not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no corpus-*.jsonl in {directory}. The sweep stage did not produce a corpus; read "
            f"{sweep_log} rather than re-running blind."
        )
    non_empty = [path for path in candidates if path.stat().st_size > 0]
    if not non_empty:
        raise ValueError(
            f"every corpus in {directory} is empty: {[p.name for p in candidates]}. The sweep ran "
            f"and its selection filter kept no prompts, which is a result to investigate, not a "
            f"corpus to train on."
        )
    if len(non_empty) > 1:
        raise ValueError(
            f"{len(non_empty)} non-empty corpora in {directory}: {[p.name for p in non_empty]}. "
            f"Pass the one you mean as {corpus_env}; picking the newest would be a guess."
        )
    return non_empty[0]


def regraded_copy(source: Path, grading: str) -> Path:
    """Name the re-graded copy of a swept corpus, beside it and marked as a derivative.

    The marker is what `resolve_single_corpus` keys on to skip it: the name still begins `corpus-`,
    so before this the regrade left the sweep directory holding two non-empty matches and every
    later resolve was refused as ambiguous.
    """
    return source.with_name(f"{source.stem}{REGRADE_MARKER}{grading}.jsonl")


def assert_corpus_will_exist(
    corpus: Path, *, written_by_a_selected_stage: bool, recipe: str
) -> None:
    """Refuse a training stage whose corpus is neither on disk nor written by this same plan.

    Shared by every plan that builds a training stage, because the failure it prevents is the same
    one: without it the stage builds happily, `stage_runner` waits for the card, `games.train`
    starts, loads a tokenizer, and only then raises from `load_corpus`. The pair's self-graded arm
    was reaching that late failure on the documented two-invocation workflow, since it derives its
    corpus name from the group corpus and nothing checked the derived path.
    """
    if corpus.is_file() or written_by_a_selected_stage:
        return
    raise FileNotFoundError(
        f"corpus {corpus} does not exist and no stage in this plan will write it. The sweep names "
        f"its output with a timestamp, so a training stage cannot be built in the same invocation "
        f"that sweeps: run the sweep alone, resolve the corpus, then run the arm.\n{recipe}"
    )


def assert_opponent_probabilities_filled(corpus: Path, arm: GameArm) -> None:
    """Refuse a vs-fixed-mix arm whose corpus has no frozen-opponent distribution to grade against.

    The corpus writes -1 for `opp_coop_prob` until the sweep's frozen-opponent pass fills it in, so
    a corpus swept without `--frozen-opponent-model` is complete, non-empty, and ungradeable.
    `games.rewards` does raise on it -- but only inside the reward function, which is after the
    model has loaded and the card is reserved. The same check costs milliseconds here.

    A no-op for any other grading, which is why every plan can call it unconditionally: the guard
    existed in the general plan only for a while, and a plan that grows a vs-frozen arm should not
    have to remember to add it.
    """
    if arm.grading != GRADING_VS_FIXED_MIX:
        return
    rows = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"corpus {corpus} holds no rows")
    missing = [row for row in rows if "opp_coop_prob" not in row]
    if missing:
        raise ValueError(
            f"corpus {corpus} carries no 'opp_coop_prob' column, so a {GRADING_VS_FIXED_MIX} arm "
            f"has no opponent distribution to grade against; {len(missing)} of {len(rows)} rows "
            f"lack it. Re-sweep with --frozen-opponent-model."
        )
    unfilled = [row for row in rows if not 0.0 <= float(row["opp_coop_prob"]) <= 1.0]
    if unfilled:
        example = float(unfilled[0]["opp_coop_prob"])
        raise ValueError(
            f"{len(unfilled)} of {len(rows)} rows in {corpus} carry an opp_coop_prob outside "
            f"[0, 1] (e.g. {example}), which is what the corpus writes when the frozen opponent "
            f"was never sampled for that prompt. Re-sweep with --frozen-opponent-model: training "
            f"would load the model, start generating, and only then raise inside the reward."
        )


def describe_stages(header: Sequence[str], stages: Sequence[Stage]) -> str:
    """Render a resolved plan -- commands, caps and promised artifacts -- without running it.

    Shared so every plan can offer `--print-plan`, which is the cheapest check on a launch before
    the meter starts. The general one-arm plan had it and the contrast pair, which is the headline
    experiment, did not.

    `header` is the plan's own summary lines; the per-stage block below is identical for every plan
    and is what an operator actually reads the printout for.
    """
    lines = list(header)
    for stage in stages:
        lines += [
            "",
            f"STAGE {stage.name}",
            f"  gpu       {stage.needs_gpu}",
            f"  argv      {' '.join(stage.argv)}",
            f"  artifacts {[str(path) for path in stage.artifacts] or 'none promised'}",
            f"  logs      {[str(path) for path in stage.log_destinations()] or 'none'}",
        ]
    return "\n".join(lines)
