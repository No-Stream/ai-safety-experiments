"""The compute-matched non-game control: what it is matched to, and how the match is checked.

Every survey, probe and decision-theory delta in this project is currently read against the base
model, which conflates two causes. "Game RL moved this instrument" and "RL at all moved this
instrument" predict the same number, and only the second is a property of our arms. The control that
separates them is one run of the same optimiser, the same estimator defaults, the same group size and
the same number of episodes against a task with no strategic content -- `grpo/rlvr_math.py`'s integer
arithmetic, which is already written, already asserts the Liger-faithful estimator, and already
measures its own verifier accuracy.

**It cannot be a `GameArm` and this module exists because of that.** `games.arms.validate_arms`
requires `game_id in games.prompts.GAME_IDS` and the reward function rebuilds each row from payoff
columns; an arithmetic corpus has neither, and the honest options were a parallel registry or a task
axis threaded through `games/train.py`. The task axis buys every guard and artifact the game arms
have, and costs a change to the module the whole experiment runs through while other arms are being
added to it. So: a parallel registry now, deliberately, and the readouts will brand the run
`UNREGISTERED` because it is not in `ARMS` -- which is true, and the banner's wording ("nothing here
can say which game it trained") is misleading rather than wrong. Say in the readout prose that its
trained game is not unknown but not a game.

**The match is on optimizer steps and episodes, never wall clock.** A 32,768-token game rollout and a
128-token arithmetic rollout are wildly different wall clock at the same episode count, so matching
time would mean training the control for far more steps than the thing it controls for -- and then a
survey shift under it would not be evidence about "RL at all" but about a much longer run. Matching
the two axes that determine how many gradient updates of what size the policy received is the
comparison that answers the question.

**Matched against what the reference arm executed, not what it requested.** `games.train` derives its
micro-batch from the VRAM present at startup, so the shape in a plan's environment is a request and
the shape in the finished run's `run_config.json` is what happened. `compute_match_from_run_config`
reads the latter for exactly that reason: matching a request would silently mismatch by whatever the
card decided, and nothing downstream would say so.

**And matched on the loss, not only on the amount of it.** Steps and episodes say how much gradient
signal the policy received; `beta`, `learning_rate`, `loss_type` and `scale_rewards` say what it was
gradient signal *toward*. Two runs identical on the first pair and different on the second are two
different objectives trained at the same price, which is exactly the reading the control exists to
rule out -- and it is the mismatch with no symptom, since both runs finish, both log the metrics the
gate reads, and every downstream table shows one arm and one control. The arithmetic harness is the
concrete case: `grpo.rlvr_math` runs the arms' `beta=0.0`, their learning rate of 1e-5 and their
estimator pair, so a launch against the registered post-audit reference passes the objective check.
The check stays in place all the same, because none of those values is derived from the match:
`beta` and the learning rate are the harness's own defaults, and each has stood an order of magnitude
off the arms' value before, while the estimator pair is a repo-wide constant that a banked reference
can predate. What an arithmetic control should train at is a question about the experiment, and this
module's job is to make any difference impossible to walk past rather than to close it.

**The estimator half of that is matched on what EXECUTES rather than on what was recorded**, since
`grpo.estimator_defaults.executed_estimator` exists precisely because a run's recorded `loss_type`
can misname its aggregation. Two runs both recording `dapo` optimise differently if one used the
fused Liger kernel and the other did not, or if they used different micro-batches under it, so the
comparison recomputes that string for both sides from the inputs each recorded.

**The reference is a named banked run, not any record of the right arm.** `NeutralTaskArm` carries a
`ReferenceRun` -- the arm, the git sha and the launch timestamp the reference recorded -- because a
local artifact path cannot go in tracked code and because the arm name alone does not identify a run.
Two things follow. A record whose own `arm` disagrees with the arm a control is registered against is
refused where it is read, since the banked twin-pd-self record at the twin-pd-group flagship's own
micro-batch split reproduces its shape, objective and model exactly -- so the axis comparison cannot
tell the two apart, and such a record would otherwise derive a match labelled twin-pd-group with
nothing downstream saying otherwise. And `assert_registered_reference` refuses a record of the right
arm that is a different run from the registered one, which is the only mismatch the eight axes cannot
see.

**Why the registered reference is a post-audit run.** The wave-1 2B `twin-pd-group` records predate
the 2026-08-20 estimator audit and carry no `loss_type` or `scale_rewards` at all, so they read as
the then-hardcoded `dapo`/`batch` -- which executed, at their micro-batch of 1, as per-sequence
original-GRPO aggregation under Liger's fallback. Matching a control to one of them would mean
training the control under exactly the estimator that audit moved off, so the match would be bought
by reverting an audited improvement. The registered reference is instead the post-audit 70-step 2B
re-run of the same arm, which recorded `dr_grpo`/`none` and is the run the arms' current estimator
line describes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

from games.arms import ARMS
from grpo.estimator_defaults import (
    GRPO_LOSS_TYPES,
    GRPO_SCALE_REWARDS_MODES,
    executed_estimator,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# The arithmetic entry point a neutral-task arm names. A string rather than the callable, so this
# module stays off the torch import path that `grpo.rlvr_math` sits behind; `games.train_neutral`
# resolves it.
ARITHMETIC_TASK = "grpo.rlvr_math.train_grpo_integer_math"

RUN_CONFIG_FILENAME = "run_config.json"

# The section of a reference arm's run config that holds its `GameTrainConfig` verbatim.
CONFIG_SECTION = "config"

# Estimator knobs absent from every arm recorded before 2026-08-20, mapped to the value the
# then-hardcoded GRPOConfig line held: a record predating the field reads as what it trained under,
# never as today's default. `games.train.RESUME_IDENTITY_DEFAULTS` holds the same mapping and is
# restated rather than imported because that module imports torch; a test pins the two equal.
LEGACY_ESTIMATOR_VALUES: dict[str, str] = {"loss_type": "dapo", "scale_rewards": "batch"}

# What each estimator knob is allowed to hold. Checked rather than passed through, because a
# `loss_type` outside this set has no executed aggregation to derive and would take
# `executed_estimator` down with a message naming no artifact.
ESTIMATOR_FIELD_VALUES: dict[str, tuple[str, ...]] = {
    "loss_type": GRPO_LOSS_TYPES,
    "scale_rewards": GRPO_SCALE_REWARDS_MODES,
}


@dataclass(frozen=True)
class ReferenceRun:
    """Which banked run of a reference arm a control is matched against, by identity not by path.

    Every field is one a games `run_config.json` records at its top level (`games.train` writes `arm`,
    `games.provenance.git_provenance` writes `git_sha`, and the launch writes `started_at`), so a
    record can be checked against a registration without a path appearing in tracked code -- which it
    may not, since the reference lives in the gitignored artifacts tree or under a run's S3 prefix.

    The timestamp is carried as well as the sha because a sha does not identify a run: this box holds
    two `twin-pd-group` records at the same commit, one 2B and one 9B.
    """

    arm: str
    # The sha exactly as the record spells it, which is a property of where the run happened rather
    # than of the commit: a container launch reports the short `GIT_SHA` baked into the image, a local
    # launch reports `git rev-parse HEAD`'s full forty characters, and a box with no checkout at all
    # records `unknown (git rev-parse failed: ...)` -- all three are in the banked records. The
    # comparison is byte-exact, so a short and a full spelling of one commit read as two runs. That is
    # deliberate: a prefix match would make the timestamp half of this identity unenforceable, and
    # `assert_registered_reference` names the hazard in its refusal instead.
    git_sha: str
    started_at: str

    def describe(self) -> str:
        """Render the identity, for a launch log and for the refusal that names a different run."""
        return f"{self.arm} at git {self.git_sha}, started {self.started_at}"


@dataclass(frozen=True)
class NeutralTaskArm:
    """One compute-matched control run: which non-game task, matched to which game arm, and why."""

    task: str
    matched_to: str
    reference_run: ReferenceRun
    notes: str


NEUTRAL_TASK_ARMS: dict[str, NeutralTaskArm] = {
    "neutral-arithmetic": NeutralTaskArm(
        task=ARITHMETIC_TASK,
        matched_to="twin-pd-group",
        # The post-audit 70-step 2B re-run of the headline arm: `dr_grpo`/`none`, beta 0.0, learning
        # rate 1e-5, micro-batch 1 x accumulation 64 at group 8. The wave-1 2B records of the same arm
        # are the alternative and are refused on purpose -- see the module docstring's last section.
        reference_run=ReferenceRun(
            arm="twin-pd-group",
            git_sha="3e0f5fd",
            started_at="2026-08-21T18:32:51.277048+00:00",
        ),
        notes=(
            "GRPO on left-to-right integer arithmetic for the same optimizer steps and the same "
            "episodes per step as twin-pd-group, under the same estimator defaults. Matched to that "
            "arm rather than to any other because it is the headline arm every instrument delta is "
            "currently read against. This is the leg that separates 'game RL moved the instrument' "
            "from 'RL at all moved it': a non-zero survey or probe shift here means every arm-level "
            "delta must be read against this run instead of against the base model, which changes "
            "the reading of the whole battery rather than adding a footnote to it. Its reward is a "
            "verifiable integer check, so arithmetic accuracy is expected to rise and nothing about "
            "a counterpart, a payoff or a choice between two labelled actions appears anywhere in "
            "its prompts."
        ),
    ),
}


@dataclass(frozen=True)
class ComputeMatch:
    """What a control run must reproduce: how many updates, of how many episodes, under which loss."""

    optimizer_steps: int
    episodes_per_step: int
    num_generations: int
    # Matched too, and the most damaging mismatch of them all if it slips: two runs on different base
    # models share no instrument baseline at all, so their deltas are not comparable in either
    # direction, and nothing downstream of a trace says which model produced it.
    model_id: str
    # The four knobs that set the objective rather than the amount of it, so a control matched on
    # steps and episodes alone can still be minimising something else. `beta` is the sharpest of
    # them: at a nonzero value the loss carries a KL penalty against the initial policy, which is a
    # term the arm being controlled for does not have at all.
    beta: float
    learning_rate: float
    loss_type: str
    scale_rewards: str
    # Not matched for their own sake, but because `grpo.estimator_defaults.executed_estimator` needs
    # them: with the loss type they decide the token aggregation the reference ACTUALLY ran, which a
    # recorded `loss_type` can misname. Kept as the raw inputs rather than as the derived string, so
    # the comparison recomputes both sides with today's function instead of trusting a string an
    # older one wrote.
    micro_batch_size: int
    use_liger_kernel: bool
    # Which banked run this was read out of, as the record identifies itself. `source` says which file
    # on this box was opened, which is not the same claim and does not survive being copied.
    run_identity: ReferenceRun
    reference_arm: str
    source: str

    @property
    def total_episodes(self) -> int:
        """Rollouts the reference arm graded in total, the second axis of the match."""
        return self.optimizer_steps * self.episodes_per_step

    @property
    def prompts_per_step(self) -> int:
        """Distinct prompts behind one step's episodes, so the episode count means the same thing."""
        return self.episodes_per_step // self.num_generations

    @property
    def executed_estimator(self) -> str:
        """The aggregation the reference arm ran, as opposed to the one its config names."""
        return executed_estimator(
            self.loss_type,
            use_liger_kernel=self.use_liger_kernel,
            per_device_train_batch_size=self.micro_batch_size,
        )

    def describe(self) -> str:
        """Render the match, so a launch log records what it claims to be matched to."""
        return (
            f"matched to {self.reference_arm} on {self.model_id} from {self.source}: "
            f"{self.optimizer_steps} optimizer steps x {self.episodes_per_step} episodes/step "
            f"= {self.total_episodes} episodes, group {self.num_generations} "
            f"({self.prompts_per_step} prompts/step), executed estimator "
            f"{self.executed_estimator!r} with scale_rewards {self.scale_rewards!r} "
            f"at learning rate {self.learning_rate} and beta {self.beta}; "
            f"reference run {self.run_identity.describe()}"
        )


def _recorded_field(payload: dict[str, Any], path: tuple[str, ...], source: str) -> object:
    """Walk one nested key path in a reference run config, refusing an absent field."""
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            raise ValueError(
                f"{source} has no {'.'.join(path)}, so that part of the reference arm's executed "
                f"configuration cannot be read and a control claiming to match it would be claiming "
                f"something unmeasured. Point --reference-run-config at the run_config.json "
                f"games.train wrote for the arm being controlled for."
            )
        cursor = cursor[key]
    return cursor


def _require_positive_int(payload: dict[str, Any], path: tuple[str, ...], source: str) -> int:
    """Read one nested integer out of a run config, refusing anything that is not usable."""
    value = _recorded_field(payload, path, source)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"{source} records {'.'.join(path)}={value!r}, which is not a positive integer, so the "
            f"reference arm's episode count cannot be derived from it."
        )
    return value


def _require_finite_float(payload: dict[str, Any], path: tuple[str, ...], source: str) -> float:
    """Read one nested loss knob, refusing a value no run could have trained under.

    Zero is allowed and is the interesting case: `beta=0.0` is what every game arm runs, and it is
    the value a control at a nonzero beta has to be refused against.
    """
    value = _recorded_field(payload, path, source)
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(
            f"{source} records {'.'.join(path)}={value!r}, which is not a finite number, so what "
            f"objective the reference arm trained under cannot be read off it."
        )
    if value < 0:
        raise ValueError(
            f"{source} records {'.'.join(path)}={value!r}. A negative learning rate or KL strength "
            f"is not a run that happened, so this record is corrupt and nothing derived from it, "
            f"including the shape above, should be trusted."
        )
    return float(value)


def _require_estimator_field(payload: dict[str, Any], field: str, source: str) -> str:
    """Read one estimator knob, substituting what a record written before the field existed ran.

    Absence is not a corrupt record here: `loss_type` and `scale_rewards` became `GameTrainConfig`
    fields on 2026-08-20, and every arm banked before that executed the then-hardcoded values in
    `LEGACY_ESTIMATOR_VALUES`. Most of the reference arm's records are in that state -- counted on this
    box on 2026-09-04, eleven of the eighteen `twin-pd-group` records predate the fields, including
    every 2B run except the post-audit one the control is now registered against. So the substitution
    is what lets a pre-audit record be read at all, and the estimator refusal it then produces is the
    mismatch rather than a defect in the record. Refusing the record outright would instead block the
    control on a missing field, and reading today's default into it would invent a match that never
    held.
    """
    section = payload.get(CONFIG_SECTION)
    if not isinstance(section, dict) or field not in section:
        legacy = LEGACY_ESTIMATOR_VALUES[field]
        logger.warning(
            "%s records no %s.%s, so it predates that field: reading it as %r, the value hardcoded "
            "when arms like this one were trained. A control on the current constant will be refused "
            "against it, which is the mismatch rather than a defect in this record.",
            source,
            CONFIG_SECTION,
            field,
            legacy,
        )
        return legacy
    value = section[field]
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{source} records {CONFIG_SECTION}.{field}={value!r}, which names no estimator, so "
            f"there is nothing for a control's own estimator to be checked against."
        )
    allowed = ESTIMATOR_FIELD_VALUES[field]
    if value not in allowed:
        raise ValueError(
            f"{source} records {CONFIG_SECTION}.{field}={value!r}, which TRL does not accept, so no "
            f"run can have executed it and the aggregation it implies cannot be derived; "
            f"{field} is one of {allowed}."
        )
    return value


def _require_bool(payload: dict[str, Any], path: tuple[str, ...], source: str) -> bool:
    """Read one nested flag, refusing anything a JSON true or false did not produce."""
    value = _recorded_field(payload, path, source)
    if not isinstance(value, bool):
        # ValueError rather than the TypeError ruff prefers for a bare isinstance check: every
        # unusable-record refusal in this module is a ValueError, and its callers and tests read that
        # as one contract. A lone TypeError here would escape a caller written to the rest of them.
        raise ValueError(  # noqa: TRY004
            f"{source} records {'.'.join(path)}={value!r}, which is not a boolean, so whether the "
            f"reference arm ran the fused kernel is unknown and the aggregation it executed cannot "
            f"be derived from it."
        )
    return value


def _run_identity(payload: dict[str, Any], *, reference_arm: str, source: str) -> ReferenceRun:
    """Read which run this record came from, refusing a record of a different arm.

    The arm check is here rather than in the comparison because it is a statement about the record
    itself: a `twin-pd-self` run config cannot yield a match to `twin-pd-group` whatever its numbers
    say, and its numbers can say nothing, because the banked 2B `twin-pd-self` record at the group
    flagship's own micro-batch split reproduces its shape, model and objective exactly. (The two
    banked at a different split are caught on the executed-estimator axis instead, which is that axis
    working rather than a reason to leave this out.) Without this the launcher would accept such a
    record, log "matched to twin-pd-group", and write that claim into the control's own
    `run_config.json` for every readout downstream to repeat.
    """
    recorded_arm = payload.get("arm")
    if not isinstance(recorded_arm, str) or not recorded_arm:
        raise ValueError(
            f"{source} records arm={recorded_arm!r}, so which experiment produced it is unknown and "
            f"nothing derived from it can be attributed to {reference_arm!r}."
        )
    if recorded_arm != reference_arm:
        raise ValueError(
            f"{source} is a run of {recorded_arm!r}, but this control is matched to "
            f"{reference_arm!r}. Arms of the same wave share their shape, model and objective, so "
            f"the axis comparison below cannot tell such a record from the right one and the mismatch "
            f"would survive only as a 'matched to {reference_arm}' label on a run of {recorded_arm}. "
            f"Point --reference-run-config at the {reference_arm} record."
        )
    git_sha = payload.get("git_sha")
    started_at = payload.get("started_at")
    if (
        not isinstance(git_sha, str)
        or not isinstance(started_at, str)
        or not (git_sha and started_at)
    ):
        raise ValueError(
            f"{source} records git_sha={git_sha!r} and started_at={started_at!r}, so this record does "
            f"not identify which run of {reference_arm!r} it is and cannot be checked against the "
            f"registered reference. games.train writes both on every launch."
        )
    return ReferenceRun(arm=recorded_arm, git_sha=git_sha, started_at=started_at)


def compute_match_from_run_config(path: Path, *, reference_arm: str) -> ComputeMatch:
    """Derive the match from a finished reference arm's own `run_config.json`.

    Episodes per step is `micro_batch_size * gradient_accumulation_steps`, which is TRL's generation
    batch when `generation_batch_size` and `steps_per_generation` are left unset -- and `games.train`
    leaves both unset deliberately, so that one generation of that many episodes happens per
    optimizer step. Taking it from the sizing plan rather than from the config is the whole point:
    the config records the episodes the operator asked for and the plan records what the card allowed.

    The loss knobs come from the config section instead, and correctly so: nothing about the card
    moves them, so what the arm asked for is what it trained under.

    The record's own identity is read too, and a record of a different arm is refused here. Whether it
    is the registered RUN of the right arm is `assert_registered_reference`, which needs the registry.
    """
    if reference_arm not in ARMS:
        raise ValueError(
            f"reference arm {reference_arm!r} is not in games.arms.ARMS, so there is no registered "
            f"experiment for this control to be matched to; known: {sorted(ARMS)}"
        )
    if not path.is_file():
        raise ValueError(
            f"reference run config {path} does not exist. The match is derived from what the "
            f"reference arm EXECUTED, not from what a plan requested, because games.train sizes its "
            f"micro-batch from the VRAM present at startup -- so there is nothing to match against "
            f"until that arm has run and shipped its {RUN_CONFIG_FILENAME}."
        )
    payload: dict[str, Any] = json.loads(path.read_text())
    source = f"{path}"
    micro_batch = _require_positive_int(payload, ("sizing_plan", "micro_batch_size"), source)
    grad_accum = _require_positive_int(
        payload, ("sizing_plan", "gradient_accumulation_steps"), source
    )
    num_generations = _require_positive_int(payload, ("sizing_plan", "num_generations"), source)
    episodes_per_step = micro_batch * grad_accum
    if episodes_per_step % num_generations:
        raise ValueError(
            f"{source} records {episodes_per_step} episodes per step against group size "
            f"{num_generations}, which does not divide it. That is not a shape TRL can have run -- "
            f"its sampler emits whole groups -- so the run config disagrees with itself and nothing "
            f"derived from it is trustworthy."
        )
    model_id = (payload.get(CONFIG_SECTION) or {}).get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(
            f"{source} records config.model_id={model_id!r}, so the reference arm's base model is "
            f"unknown. A control trained on a different model shares no instrument baseline with the "
            f"arm it controls for, and that mismatch is invisible in every artifact downstream."
        )
    return ComputeMatch(
        optimizer_steps=_require_positive_int(payload, (CONFIG_SECTION, "max_steps"), source),
        episodes_per_step=episodes_per_step,
        num_generations=num_generations,
        model_id=model_id,
        beta=_require_finite_float(payload, (CONFIG_SECTION, "beta"), source),
        learning_rate=_require_finite_float(payload, (CONFIG_SECTION, "learning_rate"), source),
        loss_type=_require_estimator_field(payload, "loss_type", source),
        scale_rewards=_require_estimator_field(payload, "scale_rewards", source),
        micro_batch_size=micro_batch,
        use_liger_kernel=_require_bool(payload, (CONFIG_SECTION, "use_liger_kernel"), source),
        run_identity=_run_identity(payload, reference_arm=reference_arm, source=source),
        reference_arm=reference_arm,
        source=source,
    )


_OBJECTIVE_MISMATCH_NOTE = (
    "The objective knobs of an arithmetic control are grpo.rlvr_math.TrainConfig's own (beta, "
    "learning_rate, use_liger_kernel) plus the repo-wide constants in grpo.estimator_defaults "
    "(loss_type, scale_rewards), and none of them is derived from this match, deliberately: moving "
    "one changes what the control computes, which is a decision about the experiment rather than "
    "about its size. The estimator line above names executed aggregations rather than loss-type "
    "labels, so two runs recording the same loss_type can still appear there. "
)


def assert_compute_matched(  # noqa: PLR0913  -- one keyword per axis the match is checked on
    match: ComputeMatch,
    *,
    optimizer_steps: int,
    per_device_train_batch: int,
    grad_accum_steps: int,
    num_generations: int,
    model_id: str,
    beta: float,
    learning_rate: float,
    loss_type: str,
    scale_rewards: str,
    use_liger_kernel: bool,
) -> None:
    """Refuse a control configuration that does not reproduce the match, on size or on objective.

    Both size axes together, because either one alone is satisfiable while the comparison is
    meaningless: the same number of steps at a quarter of the episodes is a quarter of the gradient
    signal, and the same number of episodes over four times the steps is four times the updates. The
    group size is checked as well -- it is not a third axis so much as the thing that makes an episode
    count mean the same thing in both runs, since GRPO's advantage is computed within a group.

    `beta`, `learning_rate`, the executed estimator and `scale_rewards` are checked because they set
    the loss itself. Matching step count and episode count while differing on them compares two
    different objectives at the same price and reports the difference as a property of RL: a control
    at a nonzero `beta` is minimising a KL penalty its reference arm does not have, one at ten times
    the learning rate takes ten times the step at the same count, and the estimator decides how token
    losses are aggregated and whether advantages are rescaled. Nothing downstream could recover any
    of this -- both runs finish, both log every metric the end-of-run gate reads, and every table
    shows one arm beside one control.

    **The estimator axis compares what EXECUTES, not what either side recorded**, because a recorded
    `loss_type` can misname the aggregation that ran. `executed_estimator` keys on three things, the
    loss type, whether the fused Liger kernel is in use, and the per-device micro-batch, since TRL's
    Liger call site does not forward `num_items_in_batch` and the unfaithful loss types fall back to
    a normalizer that depends on the micro-batch. So two runs can record the same `dapo` and optimise
    differently: one under the fused kernel and one not, or both under it at micro-batch 1 against
    micro-batch 8. Comparing the executed string subsumes comparing `loss_type`, since every branch
    of that function begins with the loss type's own name, and it is why `loss_type` has no separate
    entry below -- a label mismatch cannot hide from the string. `scale_rewards` keeps its own entry
    because it is not an input to that function at all.

    This is also what keeps the micro-batch split honest in both directions. Splitting the same
    episodes across more micro-batches stays free under a Liger-faithful loss such as `dr_grpo`,
    where the executed string does not mention the micro-batch, and is refused under an unfaithful
    one, where it decides the normalizer. The product rule above cannot express that distinction and
    the executed string gets it for free.

    Called at launch rather than trusted from a config file, because the values arrive from three
    places (a reference run's artifacts, a CLI, and a dataclass default) and each of them has been
    edited independently at least once.
    """
    episodes_per_step = per_device_train_batch * grad_accum_steps
    problems: list[str] = []
    if optimizer_steps != match.optimizer_steps:
        problems.append(
            f"optimizer steps {optimizer_steps} against the reference's {match.optimizer_steps}"
        )
    if episodes_per_step != match.episodes_per_step:
        problems.append(
            f"episodes per step {episodes_per_step} "
            f"({per_device_train_batch} x {grad_accum_steps}) against the reference's "
            f"{match.episodes_per_step}"
        )
    if num_generations != match.num_generations:
        problems.append(
            f"group size {num_generations} against the reference's {match.num_generations}"
        )
    if model_id != match.model_id:
        problems.append(f"base model {model_id!r} against the reference's {match.model_id!r}")
    objective_problems: list[str] = []
    if beta != match.beta:
        objective_problems.append(f"KL strength beta={beta} against the reference's {match.beta}")
    if learning_rate != match.learning_rate:
        objective_problems.append(
            f"learning rate {learning_rate} against the reference's {match.learning_rate}"
        )
    executed = executed_estimator(
        loss_type,
        use_liger_kernel=use_liger_kernel,
        per_device_train_batch_size=per_device_train_batch,
    )
    if executed != match.executed_estimator:
        objective_problems.append(
            f"executed estimator {executed!r} against the reference's {match.executed_estimator!r}"
        )
    if scale_rewards != match.scale_rewards:
        objective_problems.append(
            f"scale_rewards {scale_rewards!r} against the reference's {match.scale_rewards!r}"
        )
    if problems or objective_problems:
        raise ValueError(
            f"this control run is not compute-matched: "
            f"{'; '.join(problems + objective_problems)}. {match.describe()}. "
            f"{_OBJECTIVE_MISMATCH_NOTE if objective_problems else ''}"
            f"A mismatched control is worse than none: its deltas would still be reported beside "
            f"the arms' and read as 'RL at all does this', when they would partly be 'a differently "
            f"sized run, under a different loss, does this'."
        )
    logger.info("compute match confirmed, %s", match.describe())


def assert_registered_reference(match: ComputeMatch, expected: ReferenceRun) -> None:
    """Refuse a match derived from a run of the right arm that is not the registered one.

    This is the only mismatch the eight axes cannot see, and it is a real one: an arm has been run
    many times at the same shape and the same objective, and the registry names which of those runs
    the control's compute is matched against. Reading the reference off whichever record was nearest
    to hand instead would make the control's `run_config.json` claim a pairing that was never
    registered, and the pairing is the whole artifact -- there is nothing else in a control run that
    says what it controls for.

    Checked after `assert_compute_matched` on purpose. An axis mismatch is the likelier operator
    error and its list is the actionable message; this fires for the record that reproduces every
    axis and is still a different run. A superseding re-run is a one-line registry edit, which the
    refusal says, so the gate costs nothing when the substitution is deliberate.
    """
    if match.run_identity == expected:
        logger.info("registered reference confirmed, %s", expected.describe())
        return
    raise ValueError(
        f"{match.source} is {match.run_identity.describe()}, but this control is registered against "
        f"{expected.describe()}. Every size and objective axis matched, so the two runs are "
        f"interchangeable on everything this module measures and the difference would survive only "
        f"in provenance. Read the timestamps before concluding these are two runs: a sha's spelling "
        f"is a property of the box that recorded it, so a container's short baked-in GIT_SHA and a "
        f"local checkout's forty characters are one commit that compares as a mismatch here. If the "
        f"run named here supersedes the registered one, say so by updating reference_run on the "
        f"games.neutral_control entry rather than by pointing --reference-run-config somewhere else."
    )


def validate_neutral_task_arms(arms: dict[str, NeutralTaskArm]) -> None:
    """Reject a control arm that names no real reference, no task, or nothing about itself.

    `matched_to` is checked against the live game registry rather than a list, so an arm renamed or
    retired there fails here instead of leaving a control matched to an experiment that no longer
    exists. Run at import for the same reason `games.arms` does it: a typo should cost an import, not
    a rented card.
    """
    for name, arm in arms.items():
        if name in ARMS:
            raise ValueError(
                f"neutral-task arm {name!r} shares a name with a registered game arm. Every readout "
                f"resolves an arm label through games.arms.ARMS, so the two runs' artifacts would "
                f"merge into one row and neither could be read."
            )
        if arm.matched_to not in ARMS:
            raise ValueError(
                f"neutral-task arm {name!r} is matched to {arm.matched_to!r}, which is not in "
                f"games.arms.ARMS; known: {sorted(ARMS)}"
            )
        if arm.reference_run.arm != arm.matched_to:
            raise ValueError(
                f"neutral-task arm {name!r} is matched to {arm.matched_to!r} but its registered "
                f"reference run is {arm.reference_run.describe()}. The two name different "
                f"experiments, so the launcher would derive the match from one and label it the "
                f"other."
            )
        if not arm.reference_run.git_sha or not arm.reference_run.started_at:
            raise ValueError(
                f"neutral-task arm {name!r} registers a reference run that does not identify itself "
                f"({arm.reference_run.describe()}). Without a sha and a launch timestamp any record "
                f"of {arm.matched_to!r} would satisfy the reference check."
            )
        if not arm.task:
            raise ValueError(f"neutral-task arm {name!r} names no task to train on")
        if not arm.notes:
            raise ValueError(
                f"neutral-task arm {name!r} has no notes. This registry is the only place that says "
                f"what the control controls for, and its runs are branded UNREGISTERED by every "
                f"readout, so an undescribed one is unreadable."
            )


validate_neutral_task_arms(NEUTRAL_TASK_ARMS)
