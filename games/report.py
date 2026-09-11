"""Turn eval traces into the tables the vibe read is made of.

Every function here consumes JSONL traces written by `games.evals` and returns a tidy pandas
frame; `render_report` arranges them as markdown. No plotting: the per-round profile frame *is*
the data behind the end-game-defection plot, and a table that can be read in a terminal beats a
PNG nobody opens. Anyone who wants the figure plots the frame.

The four readouts, and why each earns its place:

- `action_rate_matrix` is the cross-game generalisation grid. Training touches one game; the
  question is what moved everywhere else, so the never-trained games are the interesting columns.
- `theory_shift_table` and `theory_item_flips` split a distinction the source post's headline
  number blurred. The aggregate moved, but the substance was 24 of 120 individual items changing
  answer, and only per-item bookkeeping can tell that from a small drift in a noisy mean.
- `per_round_profile` is where end-game defection would show up: cooperation by round index,
  falling only in the final rounds.
- `sample_cot_excerpts` is the actual vibe signal. The rates say what changed; the reasoning says
  what the model now thinks it is doing, which is the thing worth reading before any statistics.

Traces must carry `arm` and `step` in their meta record. `load_traces` raises when they do not,
rather than labelling a row from a filename: a comparison table with a guessed label is worse than
no table, because it still looks right. Every table function takes the loaded traces rather than
paths, so `render_report` reads and parses each trace exactly once instead of once per table --
traces carry the full completion text of every item and a ladder is fourteen checkpoints.

One exception to "the meta record names the arm", and it is written down rather than guessed: a
base-model step-0 cell taken from the shared bank (`games.run_evals --banked-base-cells`) is a
byte-identical copy of a cell another arm generated, so its meta names THAT arm. The driver writes a
provenance sidecar beside the copy (`banked_provenance_path`) naming the consuming arm, the bank
entry and the sha256 of every copied file; `attribute_trace` reads it, verifies the trace really is
the copy it describes, and labels the trace with the consuming arm while carrying the sidecar on the
meta under `BANKED_FROM_KEY`, so every reader above this one can name the banked source. A sidecar
whose hash does not match the file beside it raises: a copy that is not the copy it claims to be is
an attribution failure, not a damaged cell. So does one that claims a step other than
`BANKED_COPY_STEP` or a commit the trace's own meta does not record, both of which the driver
already refuses to write -- a directory anyone can copy files into is not evidence, so the readers
hold a sidecar to the same rules rather than to the driver's good behaviour.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from games import evals
from games.arms import ARMS, arm_game_ids
from games.eval_model import sha256_of_file
from games.evals import (
    RECORD_META,
    SECTION_CAPABILITIES,
    SECTION_DT_PROBES,
    SECTION_GAME_BEHAVIOR,
)

# Borrowed rather than reimplemented: two independent reductions of one quantity is how this table
# came to disagree with the summary beside it. These two want promoting to public names in evals.
order_disagreement_rate = evals._order_disagreement_rate  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
per_item_means = evals._per_item_means  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

META_ARM = "arm"
META_STEP = "step"
META_BACKEND_KIND = "backend_kind"
REQUIRED_META_FIELDS: tuple[str, ...] = (META_ARM, META_STEP)

# See `EvalTrace.from_mock_backend`.
MOCK_BACKEND_KIND = "mock"

# The banked-copy sidecar beside `step-0.jsonl` is `step-0.banked-from.json`; see the module docstring.
BANKED_PROVENANCE_SUFFIX = ".banked-from.json"
BANKED_PROVENANCE_RECORD = "banked-cell-provenance"
BANKED_FROM_KEY = "banked_from"
# The meta fields the driver reads off the ARM's own config rather than off the cell's generation
# (`games.run_evals._target_meta`; `arm` itself is the label `attribute_trace` returns). A banked
# copy carries the producer's values for them, so the sidecar records the consuming arm's under
# `consumer_meta` and `attribute_trace` overlays exactly these; every other meta field stays the
# producer's, because it describes how the bytes were made.
BANKED_CONSUMER_META_FIELDS: tuple[str, ...] = ("run_dir", "grading", "executed_estimator")
# The only step a banked copy can ever be: step 0 is the un-adapted base model, which is what makes
# one draw shareable across arms, and every other step is an arm's own adapter. Spelled here rather
# than imported from `games.run_evals.BASE_MODEL_STEP`, which imports this module.
BANKED_COPY_STEP = 0

# A flip needs a before and an after.
MIN_CHECKPOINTS_FOR_A_FLIP = 2

# Reasoning families worth surfacing excerpts for; see sample_cot_excerpts.
COT_KEYWORD_FAMILIES: dict[str, tuple[str, ...]] = {
    "correlated-reasoning": (
        "same as me",
        "same decision",
        "same choice",
        "identical",
        "copy of me",
        "mirror",
        "whatever i do",
        "we both",
    ),
    "causal-separation": (
        "already",
        "cannot affect",
        "can't affect",
        "no effect on",
        "independent of",
        "regardless of what",
        "does not change",
    ),
    "end-game": (
        "last round",
        "final round",
        "no more rounds",
        "last turn",
        "nothing left",
    ),
    "commitment": ("commit", "promise", "bind myself", "stick to"),
    "expected-value": ("expected value", "expected payoff", "on average", "probability"),
}

EXCERPT_CONTEXT_CHARS = 240


@dataclass(frozen=True)
class EvalTrace:
    """One eval trace, keyed by the arm and training step its meta record claims."""

    path: Path
    arm: str
    step: int
    meta: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]

    def section(self, name: str) -> list[Mapping[str, Any]]:
        """Return this trace's records for one section."""
        return [record for record in self.records if record.get("record") == name]

    @property
    def from_mock_backend(self) -> bool:
        """Whether this trace came from `--backend mock`, which samples nothing.

        The mock backend's canned completion is deliberately unparseable, so its rows reach a table
        as coop_rate "-" with parse_failure_rate 1.000 -- which reads exactly like a real checkpoint
        that never closed its thinking. Nothing else in a table distinguishes the two, so the label
        and the report banner carry the mark.
        """
        return self.meta.get(META_BACKEND_KIND) == MOCK_BACKEND_KIND

    @property
    def banked_from(self) -> Mapping[str, Any] | None:
        """The verified banked-copy sidecar this trace was attributed through, or None when it is its own.

        Set by `attribute_trace`, so a trace carrying it is a byte-identical copy of a base-model
        cell another arm generated, labelled with the arm that took it from the bank. The sidecar
        names the source arm, the code revision that generated the cell and the bank entry it came
        from; the banner and the inventory line read those off here.
        """
        provenance = self.meta.get(BANKED_FROM_KEY)
        return None if provenance is None else dict(provenance)

    @property
    def label(self) -> str:
        """How this trace is named in the report's inventory line."""
        marker = f" ({MOCK_BACKEND_KIND})" if self.from_mock_backend else ""
        banked = self.banked_from
        if banked is not None:
            marker += f" (banked from {banked['source_arm']}@{str(banked['source_git_sha'])[:7]})"
        return f"{self.arm}@{self.step}{marker}"

    @property
    def trained_game_ids(self) -> tuple[str, ...] | None:
        """The games this arm trains, or None when the arm is not in `games.arms.ARMS`.

        None rather than a guess: an arm label is free-form (`--arm` accepts anything, and a hosted
        model can be evaluated under any name), so an unregistered arm means the transfer column
        genuinely cannot be filled in. Reporting "unknown" beats reporting "not trained".

        A set rather than one game, because a breadth arm's corpus carries several
        (`games.arms.arm_game_ids`): reading the lead game alone would put the other trained games in
        the transfer bucket of the headline table, which is the same error this column was written to
        fix, pointed the other way.

        The registry rather than the cell's own `declared_trained_game_ids`, for the reason
        `games.battery_tables` records: a step-0 cell declares nothing, correctly, so an arm whose only
        landed cell is step 0 would read as having trained nothing. What the two disagreeing means, and
        where the report says so, is `_short_corpus_notes`.
        """
        arm = ARMS.get(self.arm)
        return None if arm is None else arm_game_ids(arm)

    @property
    def declared_trained_game_ids(self) -> tuple[str, ...]:
        """The games this CELL recorded as trained; empty for a step-0 cell and for older traces.

        `games.run_evals` derives it from the run's own corpus composition, so it names the games the
        corpus file HELD where `trained_game_ids` above names the games the arm ALLOWED. Empty is the
        base model's honest answer rather than a disagreement, so no reader may treat it as one.
        """
        config = self.meta.get("eval_config") or {}
        return tuple(str(game_id) for game_id in config.get("trained_game_ids") or ())


def banked_provenance_path(trace_path: Path) -> Path:
    """Where a banked copy's provenance sidecar lives: beside it, `.banked-from.json` for `.jsonl`."""
    return trace_path.with_name(trace_path.stem + BANKED_PROVENANCE_SUFFIX)


def read_banked_provenance(trace_path: Path) -> dict[str, Any] | None:
    """Read the banked-copy sidecar beside a trace, proving the trace is the copy it describes.

    None when there is no sidecar, which is every cell an arm generated itself. With one, the
    sidecar's recorded sha256 of the trace is checked against the file's bytes before anything is
    believed about it: the sidecar is the only thing that lets a cell whose meta names another arm
    stand in a table under this one, so a sidecar beside a file it does not describe -- a copy
    tampered with, a stale sidecar left beside a regenerated cell -- has to raise rather than
    relabel. Seconds over a 177 MB trace, and the reader parses every line of it anyway.
    """
    sidecar = banked_provenance_path(trace_path)
    if not sidecar.exists():
        return None
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    if provenance.get("record") != BANKED_PROVENANCE_RECORD:
        raise ValueError(
            f"{sidecar} is not a {BANKED_PROVENANCE_RECORD!r} record (record="
            f"{provenance.get('record')!r}), so it cannot say where {trace_path.name} came from."
        )
    files = provenance.get("files") or {}
    recorded = (files.get(trace_path.name) or {}).get("sha256")
    if not recorded:
        raise ValueError(
            f"{sidecar} records no sha256 for {trace_path.name} (files: {sorted(files)}), so "
            f"nothing can show the trace beside it is the banked copy it names."
        )
    actual = sha256_of_file(trace_path)
    if actual != recorded:
        raise ValueError(
            f"{trace_path} is not the banked copy {sidecar} describes: sha256 {actual} on disk, "
            f"{recorded} recorded. Either the copy was altered or the sidecar is stale beside a "
            f"regenerated cell; neither can be labelled as arm "
            f"{provenance.get('consumer_arm')!r}'s banked step-{provenance.get('step')} cell."
        )
    return provenance


def attribute_trace(path: Path, meta: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Name the arm a trace belongs to, and return the meta every reader should carry for it.

    The meta's own `arm` for every cell an arm generated itself. For a step-0 cell copied from the
    shared bank, the consuming arm its sidecar names (`read_banked_provenance`, hash-verified), with
    the sidecar carried on the returned meta under `BANKED_FROM_KEY`, `arm` set to the consuming
    arm and the arm-descriptive fields (`BANKED_CONSUMER_META_FIELDS`) overlaid from the sidecar's
    `consumer_meta`, so a reader pooling `grading` or `run_dir` per arm reads the arm the copy
    stands in for rather than the arm that generated it.

    Three things have to hold before a sidecar is allowed to move a trace to another arm, and the
    readers check all three themselves rather than trusting the driver that wrote it: the sidecar
    agrees with the meta about which cell was copied (the meta's arm is the sidecar's source arm
    and the steps match), the step is `BANKED_COPY_STEP` (the only step the bank can hold, so a
    sidecar hand-copied beside a checkpoint cell cannot relabel a trained arm's numbers), and the
    sidecar's `source_git_sha` is the commit the trace's own meta records (a copy is byte-identical
    to what the producer wrote, so a disagreement means the sidecar belongs to another cell).
    `games.run_evals` refuses all three at generation time; a trace read from a directory anyone
    can copy files into reaches the tables through here instead, so the rules live in both.
    """
    provenance = read_banked_provenance(path)
    if provenance is None:
        return str(meta[META_ARM]), dict(meta)
    source_arm = str(provenance.get("source_arm"))
    step = provenance.get("step")
    if str(meta[META_ARM]) != source_arm or int(meta[META_STEP]) != int(str(step)):
        raise ValueError(
            f"{path}'s meta names arm {meta[META_ARM]!r} at step {meta[META_STEP]}, but the "
            f"sidecar beside it says the copy came from {source_arm!r} at step {step}; the two "
            f"describe different cells, so neither label can be used."
        )
    if int(meta[META_STEP]) != BANKED_COPY_STEP:
        raise ValueError(
            f"{banked_provenance_path(path)} claims {path.name} is a banked copy of step "
            f"{meta[META_STEP]}, but only step {BANKED_COPY_STEP} is the un-adapted base model and "
            f"can be shared between arms; a later step is an arm's own adapter, so this sidecar "
            f"would relabel one trained arm's cell as another's."
        )
    source_git_sha = provenance.get("source_git_sha")
    if str(meta.get("git_sha")) != str(source_git_sha):
        raise ValueError(
            f"{banked_provenance_path(path)} says the copy was generated at git SHA "
            f"{source_git_sha!r}, but {path.name}'s own meta records {meta.get('git_sha')!r}; a "
            f"banked copy is byte-identical to what the producer wrote, so the sidecar describes "
            f"another cell and cannot say which arm this one belongs to."
        )
    consumer_arm = str(provenance["consumer_arm"])
    consumer_meta = dict(provenance.get("consumer_meta") or {})
    missing = sorted(set(BANKED_CONSUMER_META_FIELDS) - set(consumer_meta))
    if missing:
        raise ValueError(
            f"{banked_provenance_path(path)} records no consumer_meta for {missing}, so the copy "
            f"would stand under arm {consumer_arm!r} carrying arm {source_arm!r}'s values for "
            f"them; nothing in it can be attributed."
        )
    overlay = {name: consumer_meta[name] for name in BANKED_CONSUMER_META_FIELDS}
    return consumer_arm, {**meta, **overlay, META_ARM: consumer_arm, BANKED_FROM_KEY: provenance}


def load_traces(eval_paths: Sequence[Path]) -> list[EvalTrace]:
    """Read traces and label each by its arm and step, in (arm, step) order.

    Sorted here once so every table downstream comes out in checkpoint order; a table whose rows
    ran 10, 100, 20 would read as a non-monotonic trend that is purely an artefact of sorting.
    The arm comes from `attribute_trace`: the meta's own, or the consuming arm a banked copy's
    verified sidecar names.
    """
    if not eval_paths:
        raise ValueError("eval_paths is empty; there is nothing to report on.")
    traces: list[EvalTrace] = []
    for path in eval_paths:
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if not records or records[0].get("record") != RECORD_META:
            raise ValueError(
                f"{path} does not start with a {RECORD_META!r} record, so its rows cannot be "
                f"attributed to a model or checkpoint."
            )
        meta = records[0]
        missing = [field for field in REQUIRED_META_FIELDS if field not in meta]
        if missing:
            raise ValueError(
                f"{path}'s meta record is missing {missing}. Pass them in `meta` when calling "
                f"run_eval_battery; a report row labelled from a filename can be silently wrong."
            )
        arm, attributed = attribute_trace(path, meta)
        traces.append(
            EvalTrace(
                path=path,
                arm=arm,
                step=int(meta[META_STEP]),
                meta=attributed,
                records=tuple(records[1:]),
            )
        )
    return sorted(traces, key=lambda trace: (trace.arm, trace.step))


def truncated_count(records: Sequence[Mapping[str, Any]]) -> int:
    """Count the records whose thinking block never closed, which is its own failure state.

    A truncated completion is a parse failure with a cause: the generation cap cut the reasoning
    off, so the item measures the cap rather than the payoffs. Kept beside the parse-failure count
    rather than folded into it, because a section drifting toward the cap is a reason to raise
    `max_new_tokens` and re-run, while an item that argued its way to no tag is a real answer.
    """
    return sum(1 for record in records if record.get("truncated_thinking"))


def _prompt_means(records: Sequence[Mapping[str, Any]], field_name: str) -> list[float]:
    """Average one numeric field within each prompt, giving one observation per prompt.

    The behaviour analogue of `per_item_means`, and load-bearing for the same reason: the battery
    draws `game_behavior_samples` completions per prompt, so pooling the records would let a prompt
    whose draws mostly failed to parse count for less than its neighbours while every prompt is
    supposed to weigh the same. Identical to pooling whenever every prompt contributed the same
    number of parsed draws, which is exactly why the difference is invisible on a cell drawn once.

    Callers pass one cell's records at a time: the prompt id recurs at every checkpoint, so a caller
    that pooled several cells would merge their draws of one prompt into a single observation.
    """
    by_prompt: dict[str, list[float]] = {}
    for record in records:
        value = record.get(field_name)
        if value is not None:
            by_prompt.setdefault(str(record["prompt_id"]), []).append(float(value))
    return [sum(values) / len(values) for values in by_prompt.values()]


def action_rate_matrix(traces: Sequence[EvalTrace]) -> pd.DataFrame:
    """Return cooperation rates per (arm, step) x game, with this arm's trained game marked.

    Long form rather than a pivot, because the `trained_by_this_arm` and `n` columns are what make
    a cell interpretable: a coop rate over four completions is not the same claim as one over forty.
    The rate averages within a prompt before averaging over prompts (`_prompt_means`), so
    `n_prompts` counts prompts and `n_records` the draws behind them; the two coincide on a cell
    drawn once per prompt.

    `trained_by_this_arm` is derived from `games.arms.ARMS` rather than copied from the record's
    own `trained_game`. That flag is slate-level -- `games.evals` sets it for every game with a
    training arm anywhere in the slate -- while a report is rendered per arm and an arm trains one
    game, or the handful a breadth arm's corpus carries. Reusing it marked hi-lo, chicken, stag and
    the rest as trained inside a twin-pd report, burying most of the transfer games in the "trained"
    bucket and inverting what this table is for.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        trained_games = trace.trained_game_ids
        if trained_games is None:
            logger.warning(
                f"arm {trace.arm!r} is not in games.arms.ARMS, so {trace.path} cannot say which "
                f"game it trained; the transfer column is left unanswered for it"
            )
        by_game: dict[str, list[Mapping[str, Any]]] = {}
        for record in trace.section(SECTION_GAME_BEHAVIOR):
            by_game.setdefault(str(record["game_id"]), []).append(record)
        for game_id, records in sorted(by_game.items()):
            prompt_means = _prompt_means(records, "coop_fraction")
            parsed = sum(1 for record in records if record["parsed"])
            rows.append(
                {
                    "arm": trace.arm,
                    "step": trace.step,
                    "game_id": game_id,
                    "trained_by_this_arm": None
                    if trained_games is None
                    else game_id in trained_games,
                    "coop_rate": sum(prompt_means) / len(prompt_means) if prompt_means else None,
                    "n_parsed": len(prompt_means),
                    "n_prompts": len({str(record["prompt_id"]) for record in records}),
                    "n_records": len(records),
                    "parse_failure_rate": 1.0 - parsed / len(records),
                    "n_truncated": truncated_count(records),
                }
            )
    return pd.DataFrame(rows)


def theory_shift_table(traces: Sequence[EvalTrace]) -> pd.DataFrame:
    """Return the theory distribution per (arm, step) from the open-ended probe, plus EDT leaning.

    The open-ended fractions and DTBench's scalar sit in one table on purpose: the post found the
    free-response effect several times larger than the multiple-choice one, so seeing them apart
    is the point.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        probes = trace.section(SECTION_DT_PROBES)
        theories = [record["theory"] for record in probes if record.get("theory") is not None]
        leaning_by_item = per_item_means(probes, "edt_leaning")
        leanings = sorted(leaning_by_item.values())
        renders = [record for record in probes if record.get("edt_leaning") is not None]
        row: dict[str, Any] = {
            "arm": trace.arm,
            "step": trace.step,
            "n_open_ended": len(theories),
            "n_scored_items": len(leaning_by_item),
            "n_choice_renders": len(renders),
            "mean_edt_leaning": sum(leanings) / len(leanings) if leanings else None,
            "order_disagreement_rate": order_disagreement_rate(probes),
        }
        for label in sorted(set(theories)):
            row[f"frac_{label}"] = theories.count(label) / len(theories)
        rows.append(row)
    frame = pd.DataFrame(rows)
    fraction_columns = [column for column in frame.columns if column.startswith("frac_")]
    return frame.fillna(dict.fromkeys(fraction_columns, 0.0))


def theory_item_flips(traces: Sequence[EvalTrace]) -> pd.DataFrame:
    """Return per-item answer changes between an arm's first and last checkpoint.

    This is the readout the source post's effect actually lived in: 24 items moving EDT to CDT out
    of 120 scoreable, which an aggregate cannot distinguish from noise. One row per item that
    changed answer, so an empty frame is a real finding rather than a missing measurement.

    An item that answered differently under the two option orders is set aside rather than compared,
    and how many were set aside is logged per arm: a flip counted from an order-sensitive item is an
    artefact of which render landed last in each trace.
    """
    rows: list[dict[str, Any]] = []
    by_arm: dict[str, list[EvalTrace]] = {}
    for trace in traces:
        by_arm.setdefault(trace.arm, []).append(trace)
    for arm, arm_traces in sorted(by_arm.items()):
        if len(arm_traces) < MIN_CHECKPOINTS_FOR_A_FLIP:
            continue
        first, last = arm_traces[0], arm_traces[-1]
        before, before_unsettled = answers_by_probe(first.section(SECTION_DT_PROBES))
        after, after_unsettled = answers_by_probe(last.section(SECTION_DT_PROBES))
        unsettled = sorted(set(before_unsettled) | set(after_unsettled))
        if unsettled:
            logger.warning(
                f"arm {arm}: {len(unsettled)} probe item(s) answered differently under their two "
                f"option orders and cannot be compared across checkpoints: {unsettled}"
            )
        for probe_id in sorted(set(before) & set(after)):
            if before[probe_id] == after[probe_id]:
                continue
            rows.append(
                {
                    "arm": arm,
                    "probe_id": probe_id,
                    "from_step": first.step,
                    "to_step": last.step,
                    "before": before[probe_id],
                    "after": after[probe_id],
                }
            )
    return pd.DataFrame(rows)


def _record_answer(record: Mapping[str, Any]) -> str | None:
    """Summarise one probe render's answer as a comparable string, or None if it scored nothing.

    Multiple-choice renders reduce to their compatible theories and open-ended ones to the matched
    theory, so a flip means the item changed which theory it endorses rather than merely rewording.
    """
    if record.get("compatible_theories") is not None:
        return "+".join(record["compatible_theories"]) or "neither"
    if record.get("theory") is not None:
        return str(record["theory"])
    return None


def answers_by_probe(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], list[str]]:
    """Return each item's settled answer, and the items whose renders did not agree.

    Takes records rather than a whole trace so a caller can reduce one instrument at a time: the
    open-ended items answer with a theory name and the choice items with the theories their answer
    is compatible with, and pooling both into one distribution would count two different questions
    as one.

    One value per item, because `games.evals` asks every choice item under both option orders (and
    open-ended items several times over). Keying on probe_id alone kept whichever render was written
    last, which is worse than losing a datum: an item answered by letter position rather than by
    content lands on a different answer in each trace, and the flip table below then reports a
    between-checkpoint endorsement change that no checkpoint made.

    An item whose renders disagree is unscored rather than reduced by majority: it has not told us
    what it endorses. The disagreeing ids come back so the caller can say how many were set aside,
    since a table that silently shrinks its own denominator is the shape of bug this is fixing.
    """
    by_probe: dict[str, set[str]] = {}
    for record in records:
        answer = _record_answer(record)
        if answer is not None:
            by_probe.setdefault(str(record["probe_id"]), set()).add(answer)
    settled = {
        probe_id: next(iter(answers)) for probe_id, answers in by_probe.items() if len(answers) == 1
    }
    disagreeing = sorted(probe_id for probe_id, answers in by_probe.items() if len(answers) > 1)
    return settled, disagreeing


def per_round_profile(traces: Sequence[EvalTrace]) -> pd.DataFrame:
    """Return cooperation rate by round index for the iterated game, per (arm, step).

    The headline plot's data. End-game defection appears here as a profile that holds up across
    early rounds and drops in the last one or two; a flat profile means the model never found the
    backward induction, which is equally a result.

    Each round's rate averages within a prompt first, for `_prompt_means`' reason: a prompt whose
    draws mostly failed to parse contributes fewer moves per round than its neighbours, so pooling
    the moves would let the surviving prompts set the profile's shape. `n_moves` beside it is the
    draw-level count, which is what says how thin a round actually is.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        by_round: dict[int, dict[str, list[float]]] = {}
        for record in trace.section(SECTION_GAME_BEHAVIOR):
            moves = record.get("moves")
            if not moves:
                continue
            for index, move in enumerate(moves):
                per_prompt = by_round.setdefault(index, {})
                per_prompt.setdefault(str(record["prompt_id"]), []).append(float(move == "C"))
        for round_index, per_prompt in sorted(by_round.items()):
            prompt_means = [sum(values) / len(values) for values in per_prompt.values()]
            rows.append(
                {
                    "arm": trace.arm,
                    "step": trace.step,
                    "round_index": round_index,
                    "round_number": round_index + 1,
                    "coop_rate": sum(prompt_means) / len(prompt_means),
                    "n_prompts": len(prompt_means),
                    "n_moves": sum(len(values) for values in per_prompt.values()),
                }
            )
    return pd.DataFrame(rows)


def capability_table(traces: Sequence[EvalTrace]) -> pd.DataFrame:
    """Return the arithmetic-canary accuracy per (arm, step), so a shift can be read in context.

    `accuracy` divides by every item asked, so an answer that could not be parsed counts as wrong.
    That is the stricter of the two defensible denominators and the one the per-cell summary uses;
    `n_parsed` beside it recovers the other reading (accuracy among answered items is
    `n_correct / n_parsed`), so a table quoting one cannot be mistaken for the other.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        records = trace.section(SECTION_CAPABILITIES)
        if not records:
            continue
        correct = [bool(record["correct"]) for record in records]
        rows.append(
            {
                "arm": trace.arm,
                "step": trace.step,
                "n_items": len(correct),
                "n_parsed": sum(1 for record in records if record["parsed"]),
                "n_correct": sum(correct),
                "n_truncated": truncated_count(records),
                "accuracy": sum(correct) / len(correct),
            }
        )
    return pd.DataFrame(rows)


def _excerpt_around(text: str, needle: str) -> str:
    """Return a window of text around the first match, collapsed onto one line."""
    position = text.lower().find(needle)
    if position < 0:
        return ""
    half = EXCERPT_CONTEXT_CHARS // 2
    start = max(0, position - half)
    window = text[start : position + half]
    return " ".join(window.split())


def sample_cot_excerpts(
    traces: Sequence[EvalTrace],
    *,
    per_family: int = 2,
    families: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Pull reasoning excerpts per (arm, step, keyword family) -- the vibe signal itself.

    Keyword families rather than a judge model: which excerpts get surfaced is a sampling
    decision, and keeping it mechanical means the selection can be audited and re-run. The
    excerpts are for reading, never for scoring.

    The default families track what the CDT/EDT split looks like in actual prose --
    `correlated-reasoning` versus `causal-separation` -- plus `end-game`, which is the
    backward-induction tell in the iterated arm.
    """
    if per_family < 1:
        raise ValueError(f"per_family must be at least 1, got {per_family}.")
    resolved = families if families is not None else COT_KEYWORD_FAMILIES
    rows: list[dict[str, Any]] = []
    for trace in traces:
        counts: dict[str, int] = {}
        for record in trace.records:
            text = str(record.get("completion") or "")
            if not text:
                continue
            lowered = text.lower()
            for family, keywords in resolved.items():
                if counts.get(family, 0) >= per_family:
                    continue
                matched = next((word for word in keywords if word in lowered), None)
                if matched is None:
                    continue
                counts[family] = counts.get(family, 0) + 1
                rows.append(
                    {
                        "arm": trace.arm,
                        "step": trace.step,
                        "family": family,
                        "keyword": matched,
                        "record": record.get("record"),
                        "item_id": record.get("prompt_id") or record.get("probe_id"),
                        "excerpt": _excerpt_around(text, matched),
                    }
                )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a frame as a markdown table without pulling in a formatting dependency."""
    if frame.empty:
        return "_(no rows)_"
    columns = [str(column) for column in frame.columns]
    header = f"| {' | '.join(columns)} |"
    divider = f"| {' | '.join('---' for _ in columns)} |"
    # Rendering is inherently row-wise and these frames are tens of rows.
    body = [
        f"| {' | '.join(_format_cell(value) for value in record)} |"
        for record in frame.itertuples(
            index=False, name=None
        )  # HARNESS-SCAN-EXEMPT-non-vectorized-df
    ]
    return "\n".join([header, divider, *body])


def _format_cell(value: object) -> str:
    """Format one cell, rounding floats so a table stays readable."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def _mock_warning(traces: Sequence[EvalTrace]) -> list[str]:
    """Return the banner naming any mock traces in this report, or nothing when there are none."""
    mocked = [trace for trace in traces if trace.from_mock_backend]
    if not mocked:
        return []
    named = ", ".join(trace.label for trace in mocked)
    logger.warning(f"rendering a report over {len(mocked)} mock trace(s): {named}")
    return [
        (
            f"> **{len(mocked)} of these trace(s) came from `--backend mock`, which samples "
            f"nothing: {named}.** Their rows are plumbing output, not measurements, and their "
            f"unparseable completions read as a termination failure."
        ),
        "",
    ]


def _banked_notes(traces: Sequence[EvalTrace]) -> list[str]:
    """Name every banked copy in this report and where it came from, or nothing when there is none.

    The bank entry is named by its key, never by the `source_key` prefix the sidecar also records:
    this document gets pasted into notes and docs, and the prefix carries a bucket name. The key is
    the hash the entry lives under, so it still identifies the entry to anyone with the bank.
    """
    banked = [trace for trace in traces if trace.banked_from is not None]
    if not banked:
        return []
    lines = [
        (
            f"> **{len(banked)} of these trace(s) are byte-identical copies of shared base-model cells "
            f"another arm generated**, labelled with this arm by their provenance sidecar and "
            f"sha256-verified when read:"
        ),
    ]
    for trace in banked:
        provenance = trace.banked_from or {}
        lines.append(
            f"> - `{trace.arm}@{trace.step}` copied from bank key `{provenance.get('bank_key')}`, "
            f"generated under arm `{provenance.get('source_arm')}` at git SHA "
            f"`{provenance.get('source_git_sha')}`, banked {provenance.get('banked_at')}."
        )
    lines.append("")
    return lines


def _short_corpus_notes(traces: Sequence[EvalTrace]) -> list[str]:
    """Name every cell whose corpus held no rows of a game its arm allows, or nothing when none did.

    The tables mark a game trained when the ARM trains it, and a breadth arm's selection can drop a
    whole game group (`games.breadth_corpus`), so a dropped game arrives in the trained bucket while
    the cell's own declaration says the run never saw it. That game is the cleanest transfer evidence
    the run produced and reading it as in-distribution inverts the finding, so this banner puts the
    disagreement in front of whoever reads the table rather than leaving it to be noticed.
    """
    lines: list[str] = []
    for trace in traces:
        allowed = trace.trained_game_ids
        declared = set(trace.declared_trained_game_ids)
        if allowed is None or not declared:
            continue
        missing = [game_id for game_id in allowed if game_id not in declared]
        if not missing:
            continue
        logger.warning(
            f"{trace.label}: the run's corpus held no rows for {missing}, which arm "
            f"{trace.arm!r} allows; the trained column marks them trained anyway"
        )
        lines.append(
            f"> - `{trace.label}` held no rows for {', '.join(f'`{game}`' for game in missing)}, "
            f"and trained {', '.join(f'`{game}`' for game in trace.declared_trained_game_ids)}."
        )
    if not lines:
        return []
    return [
        (
            "> **The trained column below marks every game these arms' registry entries allow, and "
            "the cell(s) named here trained fewer than that** -- selection dropped a game group "
            "whole. Their rows for the dropped games are transfer measurements:"
        ),
        *lines,
        "",
    ]


def render_report(eval_paths: Sequence[Path], *, title: str = "Game-theory RL vibe report") -> str:
    """Render every table as one markdown document.

    Ordered so the cross-game grid comes before the decision-theory tables: what the model *does*
    is the primary measurement, and what it *says about decision theory* is the interpretation.

    The traces are read once here and handed to every table, rather than each table re-reading the
    same paths: a trace carries the full completion text of every item it evaluated.
    """
    traces = load_traces(eval_paths)
    sections = [
        f"# {title}",
        "",
        f"{len(traces)} trace(s): " + ", ".join(trace.label for trace in traces) + ".",
        "",
        *_mock_warning(traces),
        *_banked_notes(traces),
        *_short_corpus_notes(traces),
        "## Action rates by game (never-trained games are the transfer readout)",
        "",
        _markdown_table(action_rate_matrix(traces)),
        "",
        "## Per-round cooperation in the iterated game (end-game defection plot data)",
        "",
        _markdown_table(per_round_profile(traces)),
        "",
        "## Decision-theory distribution",
        "",
        _markdown_table(theory_shift_table(traces)),
        "",
        "## Per-item answer flips",
        "",
        _markdown_table(theory_item_flips(traces)),
        "",
        "## Capability canary",
        "",
        _markdown_table(capability_table(traces)),
        "",
        "## Reasoning excerpts",
        "",
        _markdown_table(sample_cot_excerpts(traces)),
        "",
    ]
    return "\n".join(sections)
