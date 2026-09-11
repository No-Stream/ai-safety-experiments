"""Read the rollout trace TRL retains for a run, and recover what each completion submitted.

``reward_hacking.train`` builds its trainer with ``log_completions=True``, so TRL writes every
completion's text to ``<output_dir>/completions/completions_<step>.parquet``. That retention is what
makes the hidden true check recoverable after the fact instead of a reason to re-run a training arm;
:mod:`reward_hacking.train_trace_score` does the recovering, and this module is the reading half.

**The schema is TRL's, read from its source rather than assumed.** ``trl/trainer/grpo_trainer.py``
builds each file's table from ``step``, ``prompt``, ``completion``, one column per reward function
named after the callable, every column a reward added through ``log_extra``, and ``advantage``. The
``step`` column is filled with a single ``global_step`` and the file is named after the same value,
so the two are one fact written twice and :func:`_assert_trace_schema` insists they still agree.

**The dataset's own columns are not in it.** ``task_id``, ``problem_id``, ``split``,
``entry_point`` and ``partition_side`` reach the reward function as kwargs and stop there -- TRL
logs only what is listed above. So the task a completion was graded against is recovered by joining
on the prompt: ``render_prompt`` is a pure function of an ILCB problem, and what TRL logs is the
chat-templated prompt decoded back from its token ids with special tokens dropped, so each logged
prompt contains exactly one problem's rendered prompt as a substring. The join is required to be
unique and refuses otherwise. **The join is always the attribution, and a logged
``log_extra("task_id", ...)`` column never replaces it** -- when the column is present every row is
resolved by the join anyway and the column is compared against the result, so a disagreement raises
rather than being resolved in the column's favour. Logging it is still worth the one line, because it
turns a silent misjoin into a refusal; it does not make the join optional, and nothing here falls
back to the column when the join fails.

**What the reward function saw and what the trace holds are the same string.** TRL computes
``batch_decode(completion_ids, skip_special_tokens=True)`` for the reward call and again for the log
(``grpo_trainer.py`` lines 2259 and 2729), so re-parsing the trace re-parses exactly the text the
live reward parsed. Qwen3.5 registers ``<think>`` and ``</think>`` as non-special added tokens, so
they survive that decode and :func:`games.parsing.strip_thinking` still finds the answer.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd

from games.parsing import strip_thinking
from reward_hacking.harness.tasks_ilcb import PROBLEMS
from reward_hacking.train_dataset import extract_solution, render_prompt

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

COMPLETIONS_SUBDIR = "completions"
COMPLETIONS_GLOB = "completions_*.parquet"
_TRACE_FILENAME = re.compile(r"^completions_(\d+)\.parquet$")

# Not imported from `games.train`, which drags torch into a CPU-only path; `run_evals` copies it too.
RUN_CONFIG_FILENAME = "run_config.json"
DERIVED_KEY = "derived"
PREFILLED_THINK_KEY = "prefilled_think"
# `reward_hacking.train` writes GraderConfig.to_json_dict() under derived.grader before weights load.
GRADER_KEY = "grader"
GRADER_TIMEOUT_SECONDS_KEY = "timeout_seconds"

STEP_COLUMN = "step"
PROMPT_COLUMN = "prompt"
COMPLETION_COLUMN = "completion"
ADVANTAGE_COLUMN = "advantage"
REQUIRED_TRACE_COLUMNS: tuple[str, ...] = (
    STEP_COLUMN,
    PROMPT_COLUMN,
    COMPLETION_COLUMN,
    ADVANTAGE_COLUMN,
)

# Optional: what `train_reward._log_batch_metrics` adds via `log_extra`, plus a task_id if ever added.
TASK_ID_COLUMN = "task_id"
RECORDED_VISIBLE_OUTCOME_COLUMN = "visible_outcome"
RECORDED_HIDDEN_OUTCOME_COLUMN = "hidden_outcome"
RECORDED_TRUNCATED_THINKING_COLUMN = "truncated_thinking"

# HARNESS-SCAN-EXEMPT-multiline-comment-block -- names the upstream fact a test pins this against.
# TRL names a reward column after its callable, so this is `make_visible_grader_reward`'s closure
# name; a rename there would silently drop the reward cross-check, and a test asserts they agree.
VISIBLE_GRADER_REWARD_COLUMN = "visible_grader_reward"

# Enough of a prompt to recognise which one it was, in a refusal message.
PROMPT_EXCERPT_CHARS = 200
AMBIGUOUS_MATCH_EXAMPLES = 5


class TaskIdentity(NamedTuple):
    """Which ILCB task a logged prompt was rendered from, and which split's grader it showed."""

    task_id: str
    split: str


@dataclass(frozen=True, slots=True)
class TraceRow:
    """One retained completion, with its task and its submitted solution recovered.

    ``solution`` is None for a completion that submitted nothing extractable, which is a
    measurement rather than a defect: the live reward scored that ``no_solution`` and so does the
    offline pass. The four ``recorded_*`` fields hold what the live run wrote for this same
    completion, kept beside the recomputed values so the two can be compared instead of assumed
    equal.

    ``task_id_from_column`` records whether THIS row's ``task_id`` cell held a value the join was
    actually compared against, which is per row and not per file: a set spanning a file written before
    the reward logged the column and one written after has both, and a column that is present but
    null in a cell yields no comparison at all. It is what
    :func:`describe_task_id_source` counts, so the provenance string in every score artifact reports
    how many rows were cross-checked instead of claiming every one was.
    """

    step: int
    trace_file: str
    row_index: int
    task_id: str
    task_split: str
    task_id_from_column: bool
    solution: str | None
    truncated_thinking: bool
    recorded_visible_outcome: str | None
    recorded_hidden_outcome: str | None
    recorded_truncated_thinking: bool | None
    recorded_reward: float | None

    @property
    def cache_key(self) -> tuple[str, str | None]:
        """Identify the grading work this row needs, so identical submissions are graded once.

        Both graders are pure functions of the submitted text: ``lay_down_task`` clears and rebuilds
        the episode directory from the task's own files every episode, so nothing carries over and
        the only input is ``solution``. Early GRPO batches routinely hold a group of eight
        byte-identical submissions, and every hidden check spawns two extra interpreters.
        """
        return (self.task_id, self.solution)

    @property
    def label(self) -> str:
        """Name this row the way a reader can find it again in the trace."""
        return f"{self.trace_file}#{self.row_index} step={self.step} task={self.task_id}"


@dataclass(frozen=True, slots=True)
class BoundedSubset:
    """What a bounded pass actually took from the trace, and how those rows were chosen."""

    n_rows_available: int
    n_rows_selected: int
    max_completions: int
    selection_rule: str

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the bound, so no reader can mistake a subset for the whole trace."""
        return {
            "n_rows_available": self.n_rows_available,
            "n_rows_selected": self.n_rows_selected,
            "max_completions": self.max_completions,
            "selection_rule": self.selection_rule,
        }


def rate(numerator: int, denominator: int) -> float | None:
    """Divide, returning None rather than zero when the denominator is empty.

    Deliberately unlike ``train_reward._rate``, which returns 0.0: that one feeds TRL's metric
    logger, which wants a float every step. Here the value is read by a human comparing steps, and
    "zero of zero" rendered as 0.0 is the reading this repo has been burned by -- a rate that fell
    because its denominator collapsed is indistinguishable from behaviour changing.
    """
    return numerator / denominator if denominator else None


def counted(values: Iterable[str]) -> dict[str, int]:
    """Tally values into a plain dict, so an artifact holds JSON rather than a Counter repr."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def rows_per_step(rows: Sequence[TraceRow]) -> dict[int, int]:
    """Count how many completions the trace retained at each step, before any bound is applied."""
    counts: dict[int, int] = {}
    for row in rows:
        counts[row.step] = counts.get(row.step, 0) + 1
    return counts


def step_for_trace_file(path: Path) -> int:
    """Return the training step one trace file was written at, from its own name.

    Refuses an unparseable name rather than guessing, because a mislabelled step silently moves a
    whole generation batch to the wrong point of the curve.
    """
    matched = _TRACE_FILENAME.match(path.name)
    if matched is None:
        raise ValueError(
            f"{path.name!r} is not a TRL rollout-trace file name; expected "
            f"completions_<step>.parquet as written by trl/trainer/grpo_trainer.py. A step read "
            f"from anywhere else would put this file's completions at the wrong point of the curve."
        )
    return int(matched.group(1))


def find_trace_files(run_dir: Path) -> list[Path]:
    """List a run's rollout-trace files in step order, refusing a run that retained none."""
    completions_dir = run_dir / COMPLETIONS_SUBDIR
    if not completions_dir.is_dir():
        raise FileNotFoundError(
            f"{completions_dir} does not exist, so this run retained no rollout trace and the "
            f"hidden check is not recoverable from it. TRL writes that directory only when the "
            f"trainer was built with log_completions=True."
        )
    paths = sorted(completions_dir.glob(COMPLETIONS_GLOB), key=step_for_trace_file)
    if not paths:
        raise FileNotFoundError(
            f"{completions_dir} holds no {COMPLETIONS_GLOB} files. The directory exists, so the "
            f"trainer was configured to write a trace but never logged one -- check whether the run "
            f"reached its first logging step."
        )
    return paths


def _assert_one_file_per_step(paths: Sequence[Path]) -> None:
    """Refuse a resolved trace set in which two files claim the same training step.

    A ``--trace`` glob that spans two runs' completion directories resolves to two files per step, and
    nothing downstream can see it: :func:`_assert_trace_schema` only checks a file's ``step`` column
    against its own name, ``build_curve`` groups purely on the step, and :func:`rows_per_step` counts
    the merged rows too, so the artifact's recorded and scored counts agree while one curve point
    averages two arms' completions -- the misspecified arm's hack rate with its control's.
    """
    by_step: dict[int, list[Path]] = {}
    for path in paths:
        by_step.setdefault(step_for_trace_file(path), []).append(path)
    collided = {step: found for step, found in sorted(by_step.items()) if len(found) > 1}
    if collided:
        named = "; ".join(
            f"step {step}: {', '.join(str(path) for path in found)}"
            for step, found in collided.items()
        )
        raise ValueError(
            f"{len(collided)} training step(s) in this trace set are written by more than one file, "
            f"so each of their curve points would average completions from more than one run and no "
            f"field in the artifact would look wrong. Read one run at a time. Collisions: {named}"
        )


def resolve_trace_files(*, run_dir: Path | None, trace: str | None) -> list[Path]:
    """Resolve which trace files to read, from a run directory or an explicit path or glob.

    Whichever way in, the resolved set is one run's: a glob reaching a second run is refused by
    :func:`_assert_one_file_per_step` here, before any file is read.

    **The glob goes through :mod:`glob` rather than ``Path().glob``, because pathlib refuses an
    absolute pattern outright** -- ``NotImplementedError: Non-relative patterns are unsupported`` --
    and every real trace lives at an absolute path, so the documented spelling of the flag crashed for
    the way it will actually be typed, with a stdlib error naming neither the flag nor the cause. A
    single absolute *file* path survived only because ``is_file()`` short-circuits above. This is the
    same fix ``analyze_ilcb.glob_trace_paths`` carries for the sibling readout; ``recursive=True``
    keeps ``**`` meaning what the pathlib spelling made it mean.

    ``key=step_for_trace_file`` rather than an unkeyed ``sorted``: it places each file at the step its
    own name carries, and refuses a name it cannot read a step from at all. TRL happens to zero-pad
    to five digits (``completions_{global_step:05d}.parquet`` in ``grpo_trainer.py``), so for the
    files it writes itself name order and step order agree -- but this reader accepts
    ``completions_<digits>.parquet`` at any width, and over a renamed or hand-assembled set a bare
    sort puts step 10 before step 2 while the artifact still reads as a curve.
    """
    if (run_dir is None) == (trace is None):
        raise ValueError(
            f"exactly one of --run-dir and --trace says which rollout trace to read; got "
            f"run_dir={run_dir} trace={trace!r}."
        )
    if run_dir is not None:
        paths = find_trace_files(run_dir)
    else:
        pattern = str(trace)
        candidate = Path(pattern)
        if candidate.is_file():
            paths = [candidate]
        else:
            matched = glob.glob(pattern, recursive=True)  # noqa: PTH207 - pathlib refuses absolutes
            paths = sorted((Path(match) for match in matched), key=step_for_trace_file)
            if not paths:
                raise FileNotFoundError(
                    f"--trace {pattern!r} matched no file. A relative pattern resolves against the "
                    f"working directory, {Path.cwd()}."
                )
    _assert_one_file_per_step(paths)
    return paths


def _read_derived(run_dir: Path, *, needed_for: str) -> dict[str, Any]:
    """Return a run's ``derived`` record, refusing a run directory that holds no record at all.

    The missing-file half of every "read it from the run rather than defaulting it" reader here.
    Only that half: which field is missing, and what guessing it would do to the numbers, differs per
    field, so each caller raises its own refusal and this one only says the record is not there.
    An absent ``derived`` block comes back empty for the caller's own key check to refuse.

    ``Any`` because this is the JSON boundary: the block's shape is whatever the run wrote, and each
    reader below narrows the one field it needs with an ``isinstance`` refusal.
    """
    path = run_dir / RUN_CONFIG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing, so nothing records {needed_for} Restore the {RUN_CONFIG_FILENAME} "
            f"the run wrote (the S3 sync ships it beside the checkpoints), or pass the value "
            f"explicitly on the command line."
        )
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    derived = record.get(DERIVED_KEY)
    return derived if isinstance(derived, dict) else {}


def read_prefilled_think(run_dir: Path) -> bool:
    """Read whether this run's chat template opened ``<think>`` inside the prompt, from its record.

    Read rather than inferred, and refused rather than defaulted. ``strip_thinking`` needs it, the
    live reward was built with it, and getting it wrong makes every completion read as truncated
    thinking -- which turns the whole recomputed trace into ``no_solution`` while still producing a
    number that looks like a curve. ``reward_hacking.train`` writes it to
    ``derived.prefilled_think`` before any weights load, so a run with a trace has this too.
    """
    derived = _read_derived(
        run_dir,
        needed_for=(
            "whether this run's chat template prefilled the opening <think> tag. strip_thinking "
            "needs that to find the answer, and guessing it wrong scores every completion as "
            "no_solution."
        ),
    )
    if PREFILLED_THINK_KEY not in derived:
        raise RuntimeError(
            f"{run_dir / RUN_CONFIG_FILENAME} records no {DERIVED_KEY}.{PREFILLED_THINK_KEY}, so "
            f"this trace cannot be parsed the way the live reward parsed it. A trace from another "
            f"trainer needs an explicit --prefilled-think/--no-prefilled-think."
        )
    return bool(derived[PREFILLED_THINK_KEY])


def read_grader_timeout_seconds(run_dir: Path) -> int:
    """Read the grader leash this run trained under, from its record, with no default.

    The same argument as :func:`read_prefilled_think`, on a value that had been taken from an offline
    CLI default instead. The leash moves the PASS/TIMEOUT boundary, which is why
    ``reward_hacking.train`` lists ``grader_timeout_seconds`` among its resume-identity fields: a run
    trained at twenty seconds and re-graded at twelve has every recomputed visible outcome describing
    a different notion of a pass from the one the gradient saw. The visible-grader cross-check does
    not catch it either -- the leash's measured sensitivity on this corpus was three of 472 records,
    which sits under the two-per-cent refusal threshold and over its one-row grace, so the pass
    proceeds and logs a warning.
    """
    derived = _read_derived(
        run_dir,
        needed_for=(
            "the grader leash this run graded under. It moves the PASS/TIMEOUT boundary, so "
            "re-grading at another value recomputes a different notion of a pass from the one the "
            "gradient saw, and the visible-grader cross-check is too coarse to catch it."
        ),
    )
    grader = derived.get(GRADER_KEY) or {}
    recorded = grader.get(GRADER_TIMEOUT_SECONDS_KEY)
    if recorded is None:
        raise RuntimeError(
            f"{run_dir / RUN_CONFIG_FILENAME} records no "
            f"{DERIVED_KEY}.{GRADER_KEY}.{GRADER_TIMEOUT_SECONDS_KEY}, so the leash the live reward "
            f"graded under is unknown. reward_hacking.train writes GraderConfig.to_json_dict() there "
            f"before any weights load. A trace from another trainer needs an explicit "
            f"--grader-timeout-seconds."
        )
    return int(recorded)


def build_prompt_index() -> dict[str, TaskIdentity]:
    """Map every ILCB problem's rendered prompt to its task id and split.

    Over all three splits rather than only the arm's own, which costs nothing -- 307 problems render
    in about ten milliseconds -- and buys a check: a run whose completions match the wrong split's
    grader becomes visible in the artifact instead of invisible. Refuses a corpus where two problems
    render byte-identical prompts, since the join could not then name one task.
    """
    index: dict[str, TaskIdentity] = {}
    for problem in PROBLEMS:
        if not problem.check_parses:
            continue
        key = render_prompt(problem).strip()
        existing = index.get(key)
        if existing is not None:
            raise RuntimeError(
                f"{existing.task_id} and {problem.harness_task_id} render byte-identical prompts, "
                f"so a logged prompt cannot be attributed to either. The rollout trace carries no "
                f"{TASK_ID_COLUMN} column, and this join is the only thing that says which grader "
                f"a completion was scored against."
            )
        index[key] = TaskIdentity(problem.harness_task_id, problem.impossible_type)
    if not index:
        raise RuntimeError(
            "no ILCB problem has a compiling check, so no prompt could be attributed to a task; "
            "the baked case file has moved under this module."
        )
    return index


class PromptTaskResolver:
    """Attribute a logged prompt to the one ILCB task whose rendered prompt it contains.

    Stateful because it memoises: a step's sixty-four completions come from eight prompts, so the
    containment scan runs once per distinct prompt rather than once per row.
    """

    def __init__(self, index: Mapping[str, TaskIdentity]) -> None:
        """Take the rendered-prompt index to attribute against, and start an empty memo."""
        self._index = dict(index)
        self._resolved: dict[str, TaskIdentity] = {}

    def resolve(self, logged_prompt: str) -> TaskIdentity:
        """Return the task this logged prompt was rendered from, refusing an ambiguous match.

        Substring containment rather than equality, because what TRL logs is the chat-templated
        prompt decoded back from its token ids with special tokens dropped -- so the rendered prompt
        survives verbatim inside the template's own wrapping, which for Qwen3.5 is a ``user`` role
        line before it and an ``assistant`` line plus a prefilled ``<think>`` tag after it. The
        needle is stripped at both ends so a template that trims its content still matches.
        """
        memoised = self._resolved.get(logged_prompt)
        if memoised is not None:
            return memoised
        matches = [identity for needle, identity in self._index.items() if needle in logged_prompt]
        if len(matches) != 1:
            names = [match.task_id for match in matches[:AMBIGUOUS_MATCH_EXAMPLES]]
            raise RuntimeError(
                f"a logged prompt matched {len(matches)} ILCB problems rather than one, so the "
                f"completion cannot be attributed to a grader: {names}. The rollout trace carries "
                f"no {TASK_ID_COLUMN} column, so this join is the only link between a completion "
                f"and the task it was scored against. The prompt begins "
                f"{logged_prompt[:PROMPT_EXCERPT_CHARS]!r}."
            )
        self._resolved[logged_prompt] = matches[0]
        return matches[0]


def _assert_trace_schema(frame: pd.DataFrame, *, path: Path, expected_step: int) -> None:
    """Refuse a trace file that is not the one TRL writes, or that disagrees about its own step."""
    missing = [column for column in REQUIRED_TRACE_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{path} is missing the column(s) {missing} that TRL writes for every run: it builds "
            f"the table from step, prompt, completion, one column per reward function, every "
            f"log_extra column, and advantage. Columns present: {sorted(frame.columns)}."
        )
    if frame.empty:
        raise RuntimeError(
            f"{path} holds no rows, so the step it names measured nothing. An empty trace file is "
            f"not a step whose rates are all zero."
        )
    logged = frame[STEP_COLUMN].tolist()  # HARNESS-SCAN-EXEMPT-object-explosion
    steps = sorted({int(value) for value in logged})
    if steps != [expected_step]:
        raise RuntimeError(
            f"{path} names step {expected_step} but its step column holds {steps}. TRL fills that "
            f"column with a single global_step and names the file after the same value, so a "
            f"disagreement means this file was not written by the run it sits in -- and every "
            f"completion in it would land at the wrong point of the curve."
        )


def _optional_str(value: object) -> str | None:
    """Read one possibly-absent trace cell as a string, mapping every pandas null form to None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if bool(pd.isna(value)):
        return None
    return str(value)


def _column_or_nones(frame: pd.DataFrame, column: str) -> list[object]:
    """Return one column's values, or a column of Nones when the trace does not carry it.

    Materialising a column into Python objects is the point here rather than a cost to avoid: the
    text has to reach a jailed interpreter, and a trace file holds at most one generation batch.
    """
    if column not in frame.columns:
        return [None] * len(frame)
    return list(frame[column].tolist())  # HARNESS-SCAN-EXEMPT-object-explosion


def _read_trace_file(
    path: Path, *, prefilled_think: bool, resolver: PromptTaskResolver
) -> list[TraceRow]:
    """Read one trace file into rows, recovering each completion's task and submitted solution.

    The one place the file is deserialised, and so the one place that can say whether it carries the
    logged ``task_id`` column. The caller used to ask that question with a second full
    ``pd.read_parquet`` whose frame it then discarded -- a whole generation batch of TRL's retained
    rollout text decoded twice for a column-name check the frame below already answers.
    """
    step = step_for_trace_file(path)
    frame = pd.read_parquet(path)
    _assert_trace_schema(frame, path=path, expected_step=step)
    if TASK_ID_COLUMN not in frame.columns:
        logger.warning(
            "%s predates the reward logging %s, so every row's task is recovered by the prompt "
            "join rather than read. The join is checked to be unique, but this trace carries no "
            "second opinion about which grader scored a completion.",
            path.name,
            TASK_ID_COLUMN,
        )
    prompts = [
        str(value) for value in frame[PROMPT_COLUMN].tolist()
    ]  # HARNESS-SCAN-EXEMPT-object-explosion
    completions = [
        str(value) for value in frame[COMPLETION_COLUMN].tolist()
    ]  # HARNESS-SCAN-EXEMPT-object-explosion
    logged_task_ids = _column_or_nones(frame, TASK_ID_COLUMN)
    recorded_visible = _column_or_nones(frame, RECORDED_VISIBLE_OUTCOME_COLUMN)
    recorded_hidden = _column_or_nones(frame, RECORDED_HIDDEN_OUTCOME_COLUMN)
    recorded_truncated = _column_or_nones(frame, RECORDED_TRUNCATED_THINKING_COLUMN)
    recorded_reward = _column_or_nones(frame, VISIBLE_GRADER_REWARD_COLUMN)

    rows: list[TraceRow] = []
    for index, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
        identity = resolver.resolve(prompt)
        logged_task_id = _optional_str(logged_task_ids[index])
        if logged_task_id is not None and logged_task_id != identity.task_id:
            raise RuntimeError(
                f"{path}#{index} carries {TASK_ID_COLUMN}={logged_task_id!r} but its prompt renders "
                f"from {identity.task_id!r}. The reward graded one task while the prompt showed "
                f"another's grader, so neither the recorded outcome nor a recomputed one describes "
                f"the completion."
            )
        visible, truncated = strip_thinking(completion, prefilled_think=prefilled_think)
        truncated_recorded = recorded_truncated[index]
        reward_recorded = recorded_reward[index]
        rows.append(
            TraceRow(
                step=step,
                trace_file=path.name,
                row_index=index,
                task_id=identity.task_id,
                task_split=identity.split,
                task_id_from_column=logged_task_id is not None,
                solution=extract_solution(visible),
                truncated_thinking=truncated,
                recorded_visible_outcome=_optional_str(recorded_visible[index]),
                recorded_hidden_outcome=_optional_str(recorded_hidden[index]),
                recorded_truncated_thinking=None
                if truncated_recorded is None
                else bool(truncated_recorded),
                recorded_reward=None if reward_recorded is None else float(str(reward_recorded)),
            )
        )
    return rows


def read_trace_rows(paths: Sequence[Path], *, prefilled_think: bool) -> list[TraceRow]:
    """Read every trace file into rows in step order, recovering task ids and solutions.

    The whole trace is read before anything is graded, so a schema break or an unattributable
    prompt costs no jail launches at all.
    """
    resolver = PromptTaskResolver(build_prompt_index())
    rows: list[TraceRow] = []
    for path in paths:
        rows.extend(_read_trace_file(path, prefilled_think=prefilled_think, resolver=resolver))
    logger.info(
        "read the rollout trace, %s",
        f"n_files={len(paths)} n_rows={len(rows)} n_steps={len({row.step for row in rows})} "
        f"n_with_solution={sum(1 for row in rows if row.solution is not None)} "
        f"n_distinct_submissions={len({row.cache_key for row in rows})} {prefilled_think=}",
    )
    return rows


def describe_task_id_source(rows: Sequence[TraceRow]) -> str:
    """Name where each row's task id came from, so an artifact does not leave it to be guessed.

    Every branch names the prompt join as the source, because it always is: ``_read_trace_file``
    resolves every row through the join and uses that result, whether or not a logged column exists.
    An earlier version of the column branch credited the column, which sent a reader investigating a
    disputed attribution to the wrong end of the pipeline -- and the string outlives the code it
    describes, since it is written into every score artifact.

    **Derived from the rows that were read, and counted rather than asserted.** It used to peek at
    the schema of ``paths[0]`` with a third full parquet read and then claim the column "was checked
    against the join on every row". Both halves of that were wrong. A multi-file set whose later file
    predates the column has rows attributed by the join alone, and the string described them off the
    first file's schema; and a column that is present but null in a cell is skipped by the
    cross-check, so a wholly-null column yielded zero comparisons under a sentence a reader takes as
    the guarantee behind every number in the file. So the count of rows actually compared is reported
    against the total, and a zero says so.
    """
    if not rows:
        return "no rows were read, so nothing was attributed"
    cross_checked = sum(1 for row in rows if row.task_id_from_column)
    matched = f"{len({row.task_id for row in rows})} distinct tasks matched"
    joined = (
        f"joined on the prompt: each logged prompt contains exactly one ILCB problem's "
        f"render_prompt output as a substring ({matched})"
    )
    if not cross_checked:
        return (
            f"{joined}. No row carried a {TASK_ID_COLUMN} value, so the join was cross-checked "
            f"against nothing on any of the {len(rows)} rows -- either the trace predates the reward "
            f"logging that column or every cell of it is null."
        )
    return (
        f"{joined}. {cross_checked} of {len(rows)} rows also carried a {TASK_ID_COLUMN} value the "
        f"reward logged through log_extra, which was checked against the join on each of those rows "
        f"and agreed; the join is the attribution and the column is the consistency check, never the "
        f"other way round. The remaining {len(rows) - cross_checked} were attributed by the join "
        f"alone."
    )


def select_bounded_subset(
    rows: Sequence[TraceRow], *, max_completions: int
) -> tuple[list[TraceRow], BoundedSubset | None]:
    """Take an evenly strided sample of the rows, or all of them when the bound does not bite.

    An even stride rather than the first N. The point of a bounded pass is to watch the machinery
    and the visible-grader cross-check work; a prefix would cover only the run's opening steps, so a
    grader mismatch that appears late would pass unnoticed while the artifact still looked like a
    curve. Returns ``None`` for the bound when nothing was dropped, which is what lets a report call
    itself complete.
    """
    if max_completions < 1:
        raise ValueError(f"the completion bound must be positive, got {max_completions}")
    if max_completions >= len(rows):
        return list(rows), None
    indices = sorted({index * len(rows) // max_completions for index in range(max_completions)})
    selected = [rows[index] for index in indices]
    bound = BoundedSubset(
        n_rows_available=len(rows),
        n_rows_selected=len(selected),
        max_completions=max_completions,
        selection_rule=(
            "even stride over rows ordered by (step, trace file, row index), so every step "
            "contributes in proportion to its share of the trace"
        ),
    )
    logger.warning(
        "BOUNDED SUBSET: taking %d of %d retained completions, so no count or rate from this pass "
        "describes the run. %s",
        len(selected),
        len(rows),
        bound.selection_rule,
    )
    return selected, bound
