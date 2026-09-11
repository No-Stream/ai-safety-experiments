"""The held-out transfer read: run the multi-turn agent harness over problems training never saw.

This is the measurement half of the misspecified-grader experiment. Training (``train_dataset``,
``train.py``) is single-turn against an inlined grader on the *training* side of the stored
problem-id partition; this module runs the real-execution agent loop
(:func:`reward_hacking.harness.loop.run_tasks`) over the *held-out* side, for one rung of a run's
checkpoint ladder at a time. If a disposition transferred, it has to show up here, in an interaction
shape training never had and on problems training never saw.

**Two splits, and the second one is not optional.** The headline is the pass rate on the
``conflicting`` split, whose visible grader contradicts itself, so a pass there is a hack by
construction and needs no judge and no detector to be one. Read alone, that rate is uninterpretable:
a checkpoint that got better at coding produces more of everything, and a checkpoint that got worse
produces none of anything. So ``original`` -- the same problems under their true grader -- runs
beside it as the capability control, and :func:`refuse_missing_capability_control` refuses an
invocation that asks for the headline without it. That refusal is the design, not a convenience:
"the model never passed" and "the tasks were broken" are the same number without the control.

**The adapter is served un-merged, and that is a ruling rather than a preference.** Folding a
rank-16 LoRA into bfloat16 base weights was measured on this repo's own adapters to retain a median
~64% of the trained delta, 38-79% by module, and nothing downstream shows it: the export loads,
generates fluently, and reads as an arm whose training moved less than it did. So
:func:`games.eval_model.resolve_served_model` picks the runtime-adapter rung,
:func:`refuse_attenuating_backend` refuses any backend kind that cannot serve one rather than
quietly taking a merge, and :func:`games.eval_model.verify_served_model` makes the engine prove
behaviourally that the adapter changed its output before a single episode is paid for -- an adapter
whose module names do not line up is skipped at DEBUG level and serves base weights under a trained
checkpoint's name.

**Nothing here derives its own task filter.** The held-out task ids come from
:func:`reward_hacking.train_dataset.held_out_task_ids` over the stored partition, and the expanded
run list is checked again with :func:`reward_hacking.train_partition.assert_side` at the point it
reaches the harness. Both, because the partition file being right is necessary and not sufficient:
a filter applied to the wrong column produces a plausible task list on the wrong side, and a bug in
this module's own mapping or repeat expansion would not be visible in any count.

**One trace per (step, split), named so a whole ladder globs.** Files are
``heldout-step-<step>-<split>.jsonl`` with the step zero-padded, so lexical order is step order and
``python -m reward_hacking.harness.analyze_ilcb --glob '<out-dir>/heldout-step-*.jsonl'`` reads the
whole curve. No rate is computed here: ``analyze_ilcb`` owns the split classification, the Wilson
intervals, the unmeasured denominator and the impossibility tripwire, and a second expression of any
of them is how the two go out of step. What this module writes beside the traces is the record that
tells one rung apart from another -- which weights, assembled how, over which partition, at what
sampling.

    uv run python -m reward_hacking.train_eval --run-dir artifacts/... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from games.eval_model import (
    LOAD_MODE_RUNTIME_ADAPTER,
    MOCK_BACKEND_KIND,
    ServedModel,
    choose_load_mode,
    resolve_served_model,
    verify_served_model,
)
from games.lora import (
    assert_adapter_matches_base,
    checkpoint_step,
    iter_checkpoints,
    read_adapter_base_model,
)
from games.provenance import git_provenance
from games.vllm_teardown import baseline_before_engine, release_engine
from reward_hacking import backend_cli
from reward_hacking.harness.loop import (
    DEFAULT_EPISODE_SECONDS,
    DEFAULT_MAX_TURNS,
    MOCK_RESPONSES,
    harness_sampling_base,
    new_episode_token,
    run_summary_line,
    run_tasks,
)
from reward_hacking.harness.task_spec import BASELINE_ARM
from reward_hacking.harness.tasks_ilcb import ILCB_TASKS_BY_ID
from reward_hacking.jailed_poc import DEFAULT_COMMAND_TIMEOUT
from reward_hacking.trace import refuse_tracked_trace_path
from reward_hacking.train_dataset import held_out_task_ids
from reward_hacking.train_partition import (
    DEFAULT_PARTITION_PATH,
    HELD_OUT_SIDE,
    SPLIT_CONFLICTING,
    SPLIT_ORIGINAL,
    assert_side,
    describe_partition,
    load_partition,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from reward_hacking.harness.loop import AgentEpisodeTrace
    from reward_hacking.harness.task_spec import AgenticArmConfig
    from reward_hacking.harness.tasks import Task
    from reward_hacking.model_backend import Backend, SamplingConfig
    from reward_hacking.train_partition import HeldOutPartition

logger = logging.getLogger(__name__)

SUMMARY_SCHEMA = 1

DEFAULT_EVAL_ROOT = Path("artifacts/reward_hacking/option3/held-out-eval")

DEFAULT_EPISODE_ROOT = Path("/var/tmp/reward-hacking-held-out-eval")  # noqa: S108 - see below
"""Where episode working directories go, and why it is neither ``/tmp`` nor the home tree.

An episode dir is host scratch the jail bind-mounts at ``/work``, and the post-hoc detector reads
the model's own files back off it long after a run, so these are retained rather than cleaned up.
``/tmp`` here is a RAM-backed tmpfs with a hard inode cap that a tree of small per-episode files has
exhausted box-wide before, and ``episode_jail.sh`` refuses to mount a home-tree path at all.
"""

BASE_MODEL_STEP = 0
"""The rung before any training, so a ladder's first point is the model the run started from.

The same convention ``games/run_evals.py`` uses for the same thing; it holds the constant privately,
which is why this is a second literal rather than an import.
"""

CAPABILITY_CONTROL_SPLIT = SPLIT_ORIGINAL
HEADLINE_SPLIT = SPLIT_CONFLICTING
EVAL_SPLIT_ORDER: tuple[str, ...] = (CAPABILITY_CONTROL_SPLIT, HEADLINE_SPLIT)
"""The two splits a held-out read runs, control first.

Ordered rather than merely listed: a run that dies partway then leaves the interpretable half, while
a headline with no control beside it is one number nobody can use.
"""

# The stem every artifact of this module shares, so one glob names the ladder and nothing else.
TRACE_STEM = "heldout"

DEFAULT_LADDER_STRIDE = 5
"""How much of a checkpoint ladder to evaluate by default, as every Nth rung.

Five rather than one because a complete ladder is thousands of multi-turn agentic episodes per arm,
which is an order of magnitude over the budget the plan set for the whole evaluation. Five locates
where the curve moves; every checkpoint is retained, so filling in the rungs between is a later
re-read rather than a re-run. The base and the final rung are kept whatever the stride, because the
transfer claim is a change between exactly those two and a ladder missing either cannot state it.
"""

LADDER_SUMMARY_FILENAME = "eval_summary.json"
"""The ladder-level index, written before any rung runs and rewritten as each finishes.

Separate from the per-rung summaries because a stage plan has to name one artifact that will exist,
and because "which rungs did this invocation intend, and which have landed" is not answerable from a
directory of per-rung files -- an interrupted ladder and a thinned one look identical there.
"""

# Wide enough that lexical order is step order for any run this experiment will produce.
_STEP_DIGITS = 5


@dataclass(frozen=True, slots=True)
class HarnessKnobs:
    """The agent loop's own budgets, carried as one value so they reach the summary intact.

    Defaults are the harness's, not a second opinion about them: a knob this module invented would
    make its episodes incomparable with every other trace in ``artifacts/harness``.
    """

    max_turns: int = DEFAULT_MAX_TURNS
    timeout: str = DEFAULT_COMMAND_TIMEOUT
    episode_seconds: float = DEFAULT_EPISODE_SECONDS
    jail_backend: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the budgets for the summary record."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LadderRung:
    """One point on a training curve: the step, and the adapter that serves it.

    ``checkpoint`` of None is the un-adapted base model at :data:`BASE_MODEL_STEP`, which is a rung
    of the ladder rather than a separate kind of run: the transfer claim is about a *change* from it.
    """

    step: int
    checkpoint: Path | None

    @property
    def label(self) -> str:
        """A short name for logs and merge staging: ``base`` or the checkpoint's own directory."""
        return "base" if self.checkpoint is None else self.checkpoint.name


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """One split's resolved work: which held-out problems, expanded by repeats, and where it lands.

    ``task_ids`` is the distinct held-out selection and ``tasks`` is the run list after repeat
    expansion, kept apart because the two are different denominators and a summary that reported
    only the second could not say whether a problem had been dropped.
    """

    split: str
    task_ids: tuple[str, ...]
    tasks: tuple[Task, ...]
    trace_path: Path

    @property
    def n_episodes(self) -> int:
        """How many episodes this split will attempt."""
        return len(self.tasks)


@dataclass(frozen=True, slots=True)
class HeldOutEvalPlan:
    """Everything resolved before a model loads: which weights, which episodes, which files."""

    rung: LadderRung
    base_model: str
    repeats: int
    splits: tuple[SplitPlan, ...]
    episode_base: Path
    summary_path: Path

    @property
    def n_episodes(self) -> int:
        """How many episodes this rung will attempt across every split."""
        return sum(split.n_episodes for split in self.splits)

    @property
    def trace_paths(self) -> tuple[Path, ...]:
        """Every file this rung will write an episode into."""
        return tuple(split.trace_path for split in self.splits)


@dataclass(frozen=True, slots=True)
class EvalSetup:
    """How one rung was configured, for the fields a summary must carry and a plan cannot derive.

    Bundled rather than passed field by field because every one of these lands in the summary
    verbatim: the point of the record is that a reader months later can tell this rung from another
    without inferring anything.
    """

    backend_kind: str
    thinking: bool
    sampling: SamplingConfig
    partition_path: Path
    partition: HeldOutPartition
    arm: AgenticArmConfig = BASELINE_ARM
    knobs: HarnessKnobs = HarnessKnobs()


@dataclass(frozen=True, slots=True)
class SplitOutcome:
    """What one split's episodes came to, in counts only -- every rate is ``analyze_ilcb``'s.

    ``n_problems_held_out`` is the partition's own held-out size and ``n_problems`` is how many of
    them this split could actually run, so a split that lost a row to a non-compiling grader is
    visible here rather than hiding inside a rate. ``n_unmeasured`` is the denominator's health: the
    oracle never reached a verdict on those episodes, so no pass or fail was read off them.
    """

    split: str
    n_problems_held_out: int
    n_problems: int
    n_episodes_attempted: int
    n_episodes_recorded: int
    n_unmeasured: int
    n_by_outcome: Mapping[str, int]
    summary_line: str
    trace_path: Path

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the counts for the summary record."""
        return {
            "split": self.split,
            "n_problems_held_out": self.n_problems_held_out,
            "n_problems": self.n_problems,
            "n_episodes_attempted": self.n_episodes_attempted,
            "n_episodes_recorded": self.n_episodes_recorded,
            "n_unmeasured": self.n_unmeasured,
            "n_by_outcome": dict(self.n_by_outcome),
            "summary_line": self.summary_line,
            "trace_path": str(self.trace_path),
        }


def refuse_missing_capability_control(splits: Sequence[str]) -> None:
    """Refuse a run that reads the headline split without the control that makes it interpretable.

    A pass on the ``conflicting`` split is a hack by construction, which is what makes the rate
    label-free -- and also what makes it confounded on its own. Training moved capability as well as
    disposition, and a hack requires clearing the same bar an honest solve does: reaching the point
    of writing something the grader runs. Without the honest-solve rate on the same problems under
    their true grader, "the trained arm hacks more" and "the trained arm codes better" are the same
    observation. The control is one extra pass over the same held-out problems, so the only reason
    to skip it would be to save time on the measurement the run exists to make.

    Watched to fail by asking for the headline alone; see this module's tests.
    """
    requested = frozenset(splits)
    if HEADLINE_SPLIT in requested and CAPABILITY_CONTROL_SPLIT not in requested:
        raise ValueError(
            f"the {HEADLINE_SPLIT!r} split was requested without {CAPABILITY_CONTROL_SPLIT!r}. A "
            f"pass on {HEADLINE_SPLIT!r} is a hack by construction, but its rate cannot be read "
            f"without the honest-solve rate on the same held-out problems under their true grader: "
            f"a checkpoint that merely got better at coding produces more of both, and one that got "
            f"worse produces neither, so the headline alone cannot tell a disposition from a "
            f"capability change. Add --split {CAPABILITY_CONTROL_SPLIT}."
        )
    unknown = sorted(requested - frozenset(EVAL_SPLIT_ORDER))
    if unknown:
        raise ValueError(
            f"unknown held-out evaluation split(s) {unknown}; this module runs "
            f"{list(EVAL_SPLIT_ORDER)}. The 'oneoff' split is the misspecified arm's TRAINING "
            f"grader, so running it here would measure the training environment, not transfer."
        )


def refuse_attenuating_backend(backend_kind: str, checkpoint: Path | None) -> None:
    """Refuse to evaluate a checkpoint on a backend that cannot serve the adapter un-merged.

    The owner's ruling, and the measurement behind it: a bfloat16 merge of a rank-16 adapter
    retained a median ~64% of the trained delta on this repo's own adapters, varying 38-79% by
    module, so an effect size read off a merged export is attenuated by roughly a third by an amount
    that is not a constant anyone could divide out. Nothing in a trace shows it. This experiment's
    whole claim is the *size* of a behavioural difference between two arms, so a merge here would
    not be a cheaper measurement of the same thing.

    ``hf`` is refused even though :func:`games.eval_model.choose_load_mode` would give it a faithful
    float32 merge: that rung serves different weights through a different code path, and mixing it
    into a ladder would put a rounding difference where a training difference is being read.
    ``mock`` is allowed because it loads nothing at all -- there is no delta to attenuate -- and its
    traces are branded so they cannot read as a measurement.

    Watched to fail by pointing ``--backend hf`` at a checkpoint; see this module's tests.
    """
    if checkpoint is None or backend_kind == MOCK_BACKEND_KIND:
        return
    mode = choose_load_mode(backend_kind)
    if mode == LOAD_MODE_RUNTIME_ADAPTER:
        return
    raise ValueError(
        f"--backend {backend_kind} cannot serve a LoRA adapter at generation time (it reaches "
        f"{mode!r}), so evaluating {checkpoint} on it would mean merging the adapter into the base "
        f"weights first. A bfloat16 merge retained a median ~64% of the trained delta when measured "
        f"on this repo's adapters -- about 36% of the training this arm claims to have received is "
        f"rounded away, unevenly across modules and invisibly in every artifact. Use --backend vllm, "
        f"which applies the adapter through an fp32-accumulated matmul and leaves the base weights "
        f"untouched."
    )


def refuse_hosted_backend(backend_kind: str) -> None:
    """Refuse a hosted endpoint for any rung, including the un-adapted one.

    Step 0 is a rung of the same ladder, not a separate baseline, so it has to be served by the same
    engine and the same sampler as the trained rungs -- otherwise the curve's first point differs
    from the rest by transport as well as by training, and the difference the experiment reads is
    partly that. A hosted model is also given a ``BedrockSamplingConfig`` rather than the local
    :class:`~reward_hacking.model_backend.SamplingConfig` this module records, so its summary would
    name a sampler that never ran.

    A hosted model on the ILCB corpus is a real measurement, just a different one; it belongs to the
    agent harness's own CLI (``python -m reward_hacking.harness.loop --backend bedrock``).
    """
    if backend_kind in backend_cli.HOSTED_KINDS:
        raise ValueError(
            f"--backend {backend_kind} is a hosted endpoint, and this module reads a local training "
            f"ladder whose every rung -- step 0 included -- must be served by one engine at one "
            f"sampler. Use --backend vllm, or run a hosted model through "
            f"`python -m reward_hacking.harness.loop --backend {backend_kind}`, which owns the "
            f"hosted sampling config."
        )


def checkpoint_ladder(run_dir: Path, *, include_base: bool = True) -> tuple[LadderRung, ...]:
    """Return a training run's rungs in step order, optionally starting from the un-adapted base.

    The step ordering and the checkpoint discovery are :mod:`games.lora`'s, not a second glob-and-
    sort here: lexical ordering puts ``checkpoint-100`` before ``checkpoint-20`` and would invert
    every before/after comparison built on this list, which is exactly the bug that function's
    docstring records.

    ``include_base`` prepends step 0. It is on by default because the transfer claim is a change
    from the untrained model, and a ladder measured without its own starting point can only be
    compared against a number from some other run.
    """
    checkpoints = list(iter_checkpoints(run_dir))
    if not checkpoints:
        raise ValueError(
            f"{run_dir} holds no checkpoint-<step> directories, so there is no ladder to evaluate. "
            f"A training run that saved nothing cannot be read at any rung."
        )
    rungs = [LadderRung(step=checkpoint_step(path), checkpoint=path) for path in checkpoints]
    if include_base:
        rungs.insert(0, LadderRung(step=BASE_MODEL_STEP, checkpoint=None))
    return tuple(rungs)


def resolve_base_model(rungs: Sequence[LadderRung], explicit: str | None) -> str:
    """Name the base model every rung is served on, and refuse a ladder that disagrees about it.

    Read from the adapters' own configs when no flag was given, because that is what training
    recorded and a hand-typed id is a guess. Every adapter is then checked against the answer:
    sibling models in one family share layer names and shapes, so an adapter trained on the 2B loads
    onto the 4B without complaint and serves a checkpoint nobody trained.
    """
    adapters = [rung.checkpoint for rung in rungs if rung.checkpoint is not None]
    if explicit is not None:
        base = explicit
    elif adapters:
        base = read_adapter_base_model(adapters[0])
        logger.info(f"base model read from the adapter config, {base=} adapter={adapters[0]}")
    else:
        raise ValueError(
            "no base model is known: this run evaluates the un-adapted model only, and nothing "
            "records which one. Pass --base-model."
        )
    for adapter in adapters:
        assert_adapter_matches_base(adapter, base)
    return base


def trace_path_for(out_dir: Path, *, step: int, split: str) -> Path:
    """Where one (rung, split) pair's episodes land, named so a whole ladder globs in step order."""
    return out_dir / f"{TRACE_STEM}-step-{step:0{_STEP_DIGITS}d}-{split}.jsonl"


def summary_path_for(out_dir: Path, *, step: int) -> Path:
    """Where one rung's provenance record lands, beside its traces and outside the trace glob."""
    return out_dir / f"{TRACE_STEM}-step-{step:0{_STEP_DIGITS}d}.summary.json"


def ladder_trace_glob(out_dir: Path) -> str:
    """Return the ``analyze_ilcb --glob`` pattern covering every rung and split written here.

    Relative to the working directory whenever it can be, which buys readability in a command meant
    to be read and retyped, and nothing more. An out dir outside the working directory keeps its
    absolute form, and ``analyze_ilcb.glob_trace_paths`` matches that too. The relativising began as
    a workaround rather than a preference: while the consumer resolved ``--glob`` through
    ``Path().glob``, which raises ``NotImplementedError: Non-relative patterns are unsupported`` on an
    absolute pattern, a printed command naming an absolute ``--out-dir`` would not have run at all.
    """
    resolved = out_dir.resolve()
    cwd = Path.cwd().resolve()
    root = resolved.relative_to(cwd) if resolved.is_relative_to(cwd) else resolved
    return f"{root}/{TRACE_STEM}-step-*.jsonl"


def resolve_splits(requested: Sequence[str] | None) -> tuple[str, ...]:
    """Resolve and order the requested splits, refusing a selection that cannot be interpreted.

    Ordered by :data:`EVAL_SPLIT_ORDER` rather than by the flags' order, so the control runs before
    the headline whatever the caller typed and a run that dies partway leaves the interpretable half.
    """
    splits = tuple(requested) if requested else EVAL_SPLIT_ORDER
    duplicated = sorted({split for split in splits if splits.count(split) > 1})
    if duplicated:
        raise ValueError(
            f"--split {duplicated} was given more than once. Repeated episodes come from --repeats, "
            f"which keeps one trace per split; a repeated split would silently double one split's "
            f"denominator instead."
        )
    refuse_missing_capability_control(splits)
    return tuple(split for split in EVAL_SPLIT_ORDER if split in frozenset(splits))


def build_plan(  # noqa: PLR0913 - every argument is a distinct axis of the plan, not a config knob
    *,
    rung: LadderRung,
    base_model: str,
    partition: HeldOutPartition,
    splits: Sequence[str],
    repeats: int,
    out_dir: Path,
    episode_root: Path,
) -> HeldOutEvalPlan:
    """Resolve one rung's episodes and file paths, refusing anything that cannot be measured.

    Every refusal here happens before a model loads, which is the point of separating this from the
    run: a typo in a path or a split must not cost an engine load, and ``--dry-run`` executes
    exactly this.

    Repeats are the caller duplicating list entries, because :func:`run_tasks` has no repeat logic
    of its own; each duplicate becomes an independent episode with its own id and its own directory.

    Two refusals run per split over what was actually resolved rather than over what was asked for.
    :func:`~reward_hacking.train_partition.assert_side` is re-applied to the *expanded* task list
    because the stored partition being right does not stop this module's own mapping or repeat
    expansion from reaching across it, and no count would show that it had.
    :func:`~reward_hacking.trace.refuse_tracked_trace_path` is asked here as well as inside the trace
    writer, so a git-tracked destination is caught before an engine load rather than after one --
    every record carries an item's full prompt text and this remote is public.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    ordered = resolve_splits(splits)
    plans: list[SplitPlan] = []
    for split in ordered:
        task_ids = held_out_task_ids(split, partition)
        tasks = tuple(ILCB_TASKS_BY_ID[task_id] for _ in range(repeats) for task_id in task_ids)
        assert_side((task.task_id for task in tasks), partition, side=HELD_OUT_SIDE)
        trace_path = trace_path_for(out_dir, step=rung.step, split=split)
        refuse_tracked_trace_path(trace_path)
        plans.append(SplitPlan(split=split, task_ids=task_ids, tasks=tasks, trace_path=trace_path))
    return HeldOutEvalPlan(
        rung=rung,
        base_model=base_model,
        repeats=repeats,
        splits=tuple(plans),
        episode_base=episode_root / f"step-{rung.step:0{_STEP_DIGITS}d}",
        summary_path=summary_path_for(out_dir, step=rung.step),
    )


def refuse_existing_traces(plans: Iterable[HeldOutEvalPlan]) -> None:
    """Refuse to write where a trace already sits, checked across every rung before the first one.

    Up front rather than per rung: a collision discovered at rung six has already spent five rungs
    of GPU time. A trace is paid-for output and ``run_tasks`` appends, so writing into one would
    blend two runs' episodes into a single denominator.
    """
    existing = [str(path) for plan in plans for path in plan.trace_paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to write over existing held-out trace(s): {existing}. run_tasks appends, so "
            f"these episodes would be pooled with whatever is already there and no rate could be "
            f"attributed. Move them aside or pick another --out-dir; nothing was evaluated."
        )


def describe_plan(plan: HeldOutEvalPlan) -> str:
    """Render what one rung would run and where it would write, for a dry run and for the log."""
    lines = [
        (
            f"rung step={plan.rung.step} label={plan.rung.label} "
            f"checkpoint={plan.rung.checkpoint} base_model={plan.base_model}"
        ),
        f"  episodes={plan.n_episodes} repeats={plan.repeats} episode_base={plan.episode_base}",
    ]
    lines.extend(
        f"  split={split.split} problems={len(split.task_ids)} episodes={split.n_episodes} "
        f"trace={split.trace_path}"
        for split in plan.splits
    )
    lines.append(f"  summary={plan.summary_path}")
    return "\n".join(lines)


def _outcome_counts(traces: Sequence[AgentEpisodeTrace]) -> dict[str, int]:
    """Count episodes per terminal outcome, sorted, as counts rather than rates.

    Counts only, and deliberately: ``analyze_ilcb`` owns every rate this experiment reads, with the
    split classification and intervals that make them meaningful. This is here so a run's summary
    can answer "did anything happen at all" without loading the traces again.
    """
    return dict(sorted(Counter(trace.outcome.value for trace in traces).items()))


def run_split(
    backend: Backend,
    split_plan: SplitPlan,
    *,
    plan: HeldOutEvalPlan,
    setup: EvalSetup,
) -> SplitOutcome:
    """Run one split's episodes through the agent harness and report what they came to.

    The trace is written per episode as the run proceeds (``run_tasks(trace_path=...)``), so a split
    that dies on episode ninety keeps the eighty-nine already paid for.
    """
    token = f"s{plan.rung.step:0{_STEP_DIGITS}d}-{split_plan.split}-{new_episode_token()}"
    logger.info(
        f"running the held-out {split_plan.split!r} split, "
        f"step={plan.rung.step} problems={len(split_plan.task_ids)} "
        f"episodes={split_plan.n_episodes} trace={split_plan.trace_path}"
    )
    traces = run_tasks(
        backend,
        split_plan.tasks,
        episode_base=plan.episode_base,
        max_turns=setup.knobs.max_turns,
        timeout=setup.knobs.timeout,
        jail_backend=setup.knobs.jail_backend,
        arm=setup.arm,
        run_token=token,
        trace_path=split_plan.trace_path,
        episode_seconds=setup.knobs.episode_seconds,
    )
    outcome = SplitOutcome(
        split=split_plan.split,
        n_problems_held_out=len(setup.partition.held_out_problem_ids),
        n_problems=len(split_plan.task_ids),
        n_episodes_attempted=split_plan.n_episodes,
        n_episodes_recorded=len(traces),
        n_unmeasured=sum(1 for trace in traces if trace.true_unmeasured),
        n_by_outcome=_outcome_counts(traces),
        summary_line=run_summary_line(traces),
        trace_path=split_plan.trace_path,
    )
    logger.info(f"held-out {split_plan.split!r} split done | {outcome.summary_line}")
    return outcome


def _verification_record(served: ServedModel) -> dict[str, object]:
    """State whether the adapter was proved to reach the forward pass, or why nothing was checked.

    A bool alone would read the same for "the engine proved the adapter changed its output" and "no
    adapter was served", which are opposite claims about a trace.
    """
    if served.load_mode == LOAD_MODE_RUNTIME_ADAPTER:
        return {
            "adapter_verification_ran": True,
            "adapter_verification": (
                "verify_served_model asserted the served adapter changes generated output"
            ),
        }
    return {
        "adapter_verification_ran": False,
        "adapter_verification": (
            f"no runtime adapter was served (load_mode={served.load_mode!r}), so there was nothing "
            f"to prove reached the forward pass"
        ),
    }


def _summary_record(
    plan: HeldOutEvalPlan,
    *,
    served: ServedModel,
    setup: EvalSetup,
    outcomes: Sequence[SplitOutcome],
) -> dict[str, object]:
    """Assemble the record that tells this rung apart from every other rung and every other run."""
    attempted = sum(outcome.n_episodes_attempted for outcome in outcomes)
    recorded = sum(outcome.n_episodes_recorded for outcome in outcomes)
    return {
        "schema": SUMMARY_SCHEMA,
        "written_at": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}",
        "base_model": plan.base_model,
        "checkpoint": None if plan.rung.checkpoint is None else str(plan.rung.checkpoint),
        "step": plan.rung.step,
        "backend_kind": setup.backend_kind,
        **served.provenance,
        **_verification_record(served),
        "thinking": setup.thinking,
        "sampling": asdict(setup.sampling),
        "arm": setup.arm.to_json_dict(),
        "harness": setup.knobs.to_json_dict(),
        "episode_base": str(plan.episode_base),
        "partition_path": str(setup.partition_path),
        "partition": describe_partition(setup.partition),
        "splits_requested": [split.split for split in plan.splits],
        "splits_completed": [outcome.split for outcome in outcomes],
        "complete": len(outcomes) == len(plan.splits),
        "repeats": plan.repeats,
        "n_episodes_attempted": attempted,
        "n_episodes_recorded": recorded,
        "n_unmeasured": sum(outcome.n_unmeasured for outcome in outcomes),
        "splits": [outcome.to_json_dict() for outcome in outcomes],
        "analyze_command": (
            f"python -m reward_hacking.harness.analyze_ilcb "
            f"--glob {ladder_trace_glob(plan.summary_path.parent)!r}"
        ),
        **git_provenance(),
    }


def run_held_out_eval(
    backend: Backend,
    plan: HeldOutEvalPlan,
    *,
    served: ServedModel,
    setup: EvalSetup,
) -> dict[str, object]:
    """Run every planned split for one rung, rewriting the summary before and after each of them.

    Before the first split as well as after each one, because the summary is what tells a reader
    which weights a trace belongs to. A rung that dies inside its first split then still leaves a
    record naming the checkpoint, the partition and the sampling, with ``complete`` false and
    ``splits_completed`` empty, rather than leaving episodes nothing attributes. That is also the
    field to gate a resume or a re-analysis on: episodes appear in a trace as they finish, so the
    presence of a trace file says a rung started, never that it finished.
    """
    outcomes: list[SplitOutcome] = []
    record = _write_summary(plan, served=served, setup=setup, outcomes=outcomes)
    for split_plan in plan.splits:
        outcomes.append(run_split(backend, split_plan, plan=plan, setup=setup))
        record = _write_summary(plan, served=served, setup=setup, outcomes=outcomes)
    logger.info(f"held-out evaluation done for step {plan.rung.step}, {plan.summary_path=}")
    return record


def _write_summary(
    plan: HeldOutEvalPlan,
    *,
    served: ServedModel,
    setup: EvalSetup,
    outcomes: Sequence[SplitOutcome],
) -> dict[str, object]:
    """Write this rung's summary over whatever has completed so far, and return the record."""
    record = _summary_record(plan, served=served, setup=setup, outcomes=outcomes)
    plan.summary_path.parent.mkdir(parents=True, exist_ok=True)
    plan.summary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _unreachable_merge_root(plan: HeldOutEvalPlan) -> Path:
    """Return a merge staging path that is never used, since no rung here is ever merged.

    :func:`games.eval_model.resolve_served_model` takes a merge root unconditionally, and
    :func:`refuse_attenuating_backend` has already refused every backend kind whose load mode would
    reach the merging branch, so the directory is named for the signature and never created.
    """
    return plan.episode_base / "unreachable-merge-root"


def _resolve_backend(
    plan: HeldOutEvalPlan, *, args: argparse.Namespace, setup: EvalSetup
) -> tuple[Backend, ServedModel]:
    """Resolve how this rung's weights are served, build the backend, and prove the adapter applies.

    The verification is before the episodes, not after: an adapter the engine silently skipped
    serves base weights, and a whole rung of GPU time would read as a training run that changed
    nothing.
    """
    served = resolve_served_model(
        checkpoint=plan.rung.checkpoint,
        base_model=plan.base_model,
        backend_kind=setup.backend_kind,
        merge_root=_unreachable_merge_root(plan),
        merge_label=f"{plan.rung.label}-step-{plan.rung.step}-",
    )
    logger.info(f"step {plan.rung.step} serving {served.load_mode}: {served.provenance}")
    # Branded like games/run_evals does it: a mock trace must never pass for a measurement.
    model_id = (
        f"mock:{served.model_id}" if setup.backend_kind == MOCK_BACKEND_KIND else served.model_id
    )
    backend = backend_cli.backend_from_args(
        args,
        model_id,
        local_sampling=setup.sampling,
        mock_responses=MOCK_RESPONSES,
        extra_kwargs=served.backend_kwargs,
    )
    verify_served_model(backend, served)
    return backend, served


def _run_rung(plan: HeldOutEvalPlan, *, args: argparse.Namespace, setup: EvalSetup) -> None:
    """Serve one rung's weights, run its splits, and hand the card back before the next rung.

    The teardown here used to be ``del backend`` plus ``torch.cuda.empty_cache()`` -- a private copy
    of exactly what :mod:`games.vllm_teardown` was written to replace, and both halves are no-ops on a
    vLLM engine: the caller still holds the reference so nothing is collected, and the weights and KV
    cache live in an ``EngineCore`` subprocess whose memory the parent's allocator cannot account for,
    let alone hand back. Measured on the sibling path, that left 2.96 of a 44.39 GiB card free sixteen
    seconds after the engine's context manager exited, and the next engine died inside vLLM with a
    message about GPU memory utilization and no mention of the engine that never let go.

    This ladder is the driver most exposed to it. ``train_sequence.eval_stage`` runs a whole
    ``--every-nth-checkpoint`` ladder in ONE process, so roughly sixteen sequential in-process engines
    is what the eval stage does, and nothing on this path passes ``gpu_memory_utilization``, so each
    of them claims vLLM's default 0.9 of the card.

    :func:`games.vllm_teardown.baseline_before_engine` reads the card BEFORE anything is constructed
    and gates on the declared backend kind, returning ``None`` for the kinds that stand up no engine;
    see its docstring for why both are its decisions rather than each caller's. The release sits in a
    ``finally`` because an engine left standing after a failed rung takes the card down with it for
    every rung after, turning one bad rung into a dead ladder.
    """
    baseline_mib = baseline_before_engine(setup.backend_kind)
    if baseline_mib is not None:
        logger.info(f"step {plan.rung.step} loading onto a card holding {baseline_mib} MiB")
    backend, served = _resolve_backend(plan, args=args, setup=setup)
    try:
        run_held_out_eval(backend, plan, served=served, setup=setup)
        backend_cli.log_token_usage(backend)
    finally:
        if baseline_mib is None:
            # From inside a helper this `del` would unbind only that helper's name, freeing nothing.
            del backend
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            release_engine(backend, baseline_mib=baseline_mib)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    backend_cli.add_backend_args(parser, default="vllm")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Evaluate one LoRA checkpoint directory, served un-merged. Omit both this and --run-dir "
            f"to evaluate the un-adapted base model as step {BASE_MODEL_STEP}."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Evaluate every checkpoint-<step> under a training run directory, in step order, "
            "starting from the un-adapted base unless --no-with-base is given."
        ),
    )
    parser.add_argument(
        "--with-base",
        dest="with_base",
        action="store_true",
        help="Include step 0 (the un-adapted base) in a --run-dir ladder. On by default.",
    )
    parser.add_argument("--no-with-base", dest="with_base", action="store_false")
    parser.set_defaults(with_base=True)
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model id; read from the adapter config when omitted. Required for a bare base run.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=(
            "Step label for --checkpoint, when the directory name is not the training step. "
            f"Defaults to the checkpoint's own number, or {BASE_MODEL_STEP} for a base run."
        ),
    )
    parser.add_argument(
        "--partition",
        type=Path,
        default=DEFAULT_PARTITION_PATH,
        help="The stored held-out problem-id partition both halves of the experiment read.",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        choices=EVAL_SPLIT_ORDER,
        default=None,
        help=(
            f"Held-out split to evaluate; repeatable. Default: both, control first. "
            f"{HEADLINE_SPLIT!r} without {CAPABILITY_CONTROL_SPLIT!r} is refused."
        ),
    )
    parser.add_argument(
        "--every-nth-checkpoint",
        type=int,
        default=DEFAULT_LADDER_STRIDE,
        help=(
            f"Thin a --run-dir ladder to every Nth checkpoint, keeping the base and the final rung "
            f"whatever the stride. Default {DEFAULT_LADDER_STRIDE}. A complete every-checkpoint "
            f"ladder is 71 rungs x 2 splits x the held-out problems, which is thousands of "
            f"multi-turn agentic episodes per arm; retention makes a finer ladder a later re-read of "
            f"checkpoints already on disk rather than a re-run of anything."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Episodes per held-out problem; each is an independent episode. Default 1.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument(
        "--episode-root",
        type=Path,
        default=DEFAULT_EPISODE_ROOT,
        help=(
            "Disposable scratch root for episode working directories, which the jail bind-mounts "
            "at /work and the post-hoc detector reads back. Never a home tree."
        ),
    )
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--timeout", default=DEFAULT_COMMAND_TIMEOUT)
    parser.add_argument("--episode-seconds", type=float, default=DEFAULT_EPISODE_SECONDS)
    parser.add_argument(
        "--jail-backend", choices=("bwrap", "unshare"), default=None, help="Jail isolation backend."
    )
    parser.add_argument(
        "--allow-short-completions",
        action="store_true",
        help=(
            "Sample below the model's measured output floor, which is otherwise refused. Plumbing "
            "probes only: the episodes it writes read as a policy that stopped acting."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve the partition, the tasks, the served weights and every path, print them, and "
            "load no model. The way to smoke this wiring without a GPU."
        ),
    )
    return parser.parse_args(argv)


def _rungs_from_args(args: argparse.Namespace) -> tuple[LadderRung, ...]:
    """Resolve which rungs this invocation evaluates, refusing an ambiguous target selection."""
    if args.checkpoint is not None and args.run_dir is not None:
        raise ValueError(
            "--checkpoint and --run-dir both name what to evaluate; pass exactly one. Omit both to "
            "evaluate the un-adapted base model."
        )
    if args.run_dir is not None:
        if args.step is not None:
            raise ValueError(
                "--step labels a single --checkpoint; a --run-dir ladder reads its own."
            )
        return thin_ladder(
            checkpoint_ladder(args.run_dir, include_base=args.with_base),
            stride=args.every_nth_checkpoint,
        )
    if args.checkpoint is not None:
        step = args.step if args.step is not None else checkpoint_step(args.checkpoint)
        return (LadderRung(step=step, checkpoint=args.checkpoint),)
    step = args.step if args.step is not None else BASE_MODEL_STEP
    return (LadderRung(step=step, checkpoint=None),)


def thin_ladder(rungs: Sequence[LadderRung], *, stride: int) -> tuple[LadderRung, ...]:
    """Keep every Nth rung plus both endpoints, so a thinned ladder can still state the claim.

    The endpoints are unconditional. The transfer claim is a change from the untrained model to the
    trained one, so a stride that happened not to land on the last checkpoint would produce a ladder
    that cannot be read as before-and-after -- and it would do so silently, since a ladder is a list
    of plausible rungs either way.
    """
    if stride < 1:
        raise ValueError(f"--every-nth-checkpoint must be at least 1, got {stride}")
    if not rungs:
        raise ValueError("cannot thin an empty ladder")
    kept = {0, len(rungs) - 1} | set(range(0, len(rungs), stride))
    thinned = tuple(rungs[index] for index in sorted(kept))
    if len(thinned) != len(rungs):
        logger.info(
            "thinned the ladder to every %dth checkpoint plus both endpoints, %s",
            stride,
            f"n_rungs={len(thinned)} of {len(rungs)} steps={[rung.step for rung in thinned]}",
        )
    return thinned


def write_ladder_summary(out_dir: Path, plans: Sequence[HeldOutEvalPlan], *, stride: int) -> Path:
    """Write the ladder-level index, so a stage plan has one artifact it can verify.

    Written BEFORE the first rung runs, then rewritten as rungs land, for the reason each per-rung
    summary is: a ladder that dies partway must still leave a record of what it intended, or the
    difference between "thinned on purpose" and "died at rung three" is unrecoverable.
    """
    path = out_dir / LADDER_SUMMARY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SUMMARY_SCHEMA,
        "every_nth_checkpoint": stride,
        "n_rungs": len(plans),
        "steps": [plan.rung.step for plan in plans],
        "n_episodes_planned": sum(plan.n_episodes for plan in plans),
        "rung_summaries": [
            {
                "step": plan.rung.step,
                "summary": str(summary_path_for(out_dir, step=plan.rung.step)),
                "landed": summary_path_for(out_dir, step=plan.rung.step).is_file(),
            }
            for plan in plans
        ],
        "trace_glob": ladder_trace_glob(out_dir),
        **git_provenance(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _setup_from_args(
    args: argparse.Namespace, *, base_model: str, partition: HeldOutPartition
) -> EvalSetup:
    """Resolve the decoding and harness configuration, refusing a cap that truncates the policy.

    The knob check runs here rather than being left to ``backend_from_args``, which is reached only
    after the partition has loaded and a plan has been built: a knob the chosen backend would ignore
    in silence is a mis-specified run, and it should be refused before anything else is resolved --
    including under ``--dry-run``, whose whole job is to catch this class of mistake without a GPU.
    """
    backend_cli.reject_inapplicable_knobs(args.backend, args)
    thinking = backend_cli.resolve_thinking(args)
    base_sampling = harness_sampling_base(base_model, thinking=thinking)
    sampling = backend_cli.local_sampling_from_args(args, base_sampling)
    backend_cli.refuse_short_output_cap(
        base_model, sampling.max_new_tokens, allow_short=args.allow_short_completions
    )
    return EvalSetup(
        backend_kind=args.backend,
        thinking=thinking,
        sampling=sampling,
        partition_path=args.partition,
        partition=partition,
        knobs=HarnessKnobs(
            max_turns=args.max_turns,
            timeout=args.timeout,
            episode_seconds=args.episode_seconds,
            jail_backend=args.jail_backend,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate one rung, or a whole ladder, on the held-out side of the stored partition."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = _parse_args(argv)
    refuse_hosted_backend(args.backend)
    rungs = _rungs_from_args(args)
    for rung in rungs:
        refuse_attenuating_backend(args.backend, rung.checkpoint)
    base_model = resolve_base_model(rungs, args.base_model)
    partition = load_partition(args.partition)
    setup = _setup_from_args(args, base_model=base_model, partition=partition)
    plans = [
        build_plan(
            rung=rung,
            base_model=base_model,
            partition=partition,
            splits=args.splits or EVAL_SPLIT_ORDER,
            repeats=args.repeats,
            out_dir=args.out_dir,
            episode_root=args.episode_root,
        )
        for rung in rungs
    ]
    refuse_existing_traces(plans)
    logger.info(
        "held-out evaluation plan | %d rung(s), %d episodes total, backend=%s thinking=%s\n%s",
        len(plans),
        sum(plan.n_episodes for plan in plans),
        args.backend,
        setup.thinking,
        "\n".join(describe_plan(plan) for plan in plans),
    )
    logger.info(f"sampling | {setup.sampling}")
    logger.info("partition | %s", json.dumps(describe_partition(partition)))
    logger.info(f"read the ladder with: analyze_ilcb --glob {ladder_trace_glob(args.out_dir)!r}")
    if args.dry_run:
        for plan in plans:
            served = resolve_served_model(
                checkpoint=plan.rung.checkpoint,
                base_model=base_model,
                backend_kind=args.backend,
                merge_root=_unreachable_merge_root(plan),
                merge_label=f"{plan.rung.label}-dry-run-",
            )
            logger.info(f"dry run | step {plan.rung.step} would serve {served.provenance}")
        logger.info("dry run: nothing was loaded, nothing was written")
        return 0
    ladder_summary = write_ladder_summary(args.out_dir, plans, stride=args.every_nth_checkpoint)
    logger.info(f"ladder index | {ladder_summary}")
    for plan in plans:
        _run_rung(plan, args=args, setup=setup)
        write_ladder_summary(args.out_dir, plans, stride=args.every_nth_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
