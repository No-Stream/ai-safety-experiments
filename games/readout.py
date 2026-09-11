r"""The cross-arm readout: one document recomputed from the eval traces every time it is asked for.

`games.report` renders one arm at a time and every number in it is a single cell. That is the wrong
unit for this wave. A measured null on an UNCHANGED model moved single game-cells by up to 0.467
across replicates, so a step-0-vs-step-70 pair on one game is noise with a story attached. This
module therefore reads two things only: the **pooled windows** (early = steps {0, 10}, late =
steps {60, 70}) and the full ladder beside them, so a window contrast can be checked against the
shape of the curve that produced it.

Three properties it is built around, all of them things a static write-up loses:

- **It recomputes.** Every table comes from the traces on disk at the moment it runs, and the
  batteries for three arms are still landing, so a rerun is the whole update.
- **It says what it could not measure.** An arm short of eight cells, a cell whose JSONL is damaged
  and a section past a 0.25 parse failure rate are each named in the banner, and any of them marks
  the document INCOMPLETE AND ERROR-CONTAINING: a readout that looks finished is worse than none.
- **Every rate carries its denominator, in both units.** A behaviour rate is a mean over PROMPTS,
  averaged within each prompt first, so `n_parsed`/`n_asked` count prompts and `n_records` counts
  the completions behind them; unparsed prompts leave the numerator and stay in `n_asked`, and a
  rate over fewer than four parsed observations renders "-".

Reused rather than reimplemented: `games.report.EvalTrace` (the trace data model), and
`order_disagreement_rate` and `per_item_means`, which report itself borrowed from `games.evals`
because two independent reductions of one quantity is how a table comes to disagree with the
summary beside it. Not reused: `report.load_traces`, which parses every line eagerly and raises on
the first bad one, so it cannot tell "this file cannot be attributed to an arm at all" from "this
belongs to twin-pd-self at step 20 and its last line was cut off mid-write". Those want different
answers -- fail fast against a banner -- and with batteries still writing into this tree the second
is the common case. `read_cell` does that reading against report's own meta-field requirements, so
the two cannot drift on what a trace must carry.

The arm-to-trained-game mapping comes from `games.arms.ARMS`, never from a trace: the records'
`trained_game` flag is slate-level, and `eval_config.trained_game_ids` is empty for every step-0
cell (correctly -- step 0 is the un-adapted base model), so an arm whose only cell so far is step 0
would read as training nothing. `trained_game_declaration_notes` checks the traces against the
registry rather than trusting either.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from games import report
from games.arms import ARMS
from games.evals import (
    BEHAVIOUR_FIELD_BY_GAME,
    KEEP_FIELD,
    RECORD_META,
    SECTION_CAPABILITIES,
    SECTION_DT_PROBES,
    SECTION_GAME_BEHAVIOR,
)
from games.parsing import THEORY_LABELS
from games.prompts import DICTATOR_GAME_ID, EVAL_ONLY_GAME_IDS
from games.report import EvalTrace

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

# Borrowed for the same reason report.py borrows from evals: one reduction per quantity. These three
# want promoting to public names in their own modules.
markdown_table = report._markdown_table  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
order_disagreement_rate = report.order_disagreement_rate
per_item_means = report.per_item_means

logger = logging.getLogger(__name__)

DEFAULT_EVALS_ROOT = Path("artifacts/games/evals-s3")
DEFAULT_OUT_DIR = Path("artifacts/games/readout")

# One battery per checkpoint, checkpoints every 10 steps through 70, plus step 0 for the base model.
EXPECTED_CELLS_PER_ARM = 8
STEP_FILE_GLOB = "*/*/step-*.jsonl"

# The pooled windows. Two checkpoints each, because a single cell is not a finding here.
EARLY_WINDOW = "early {0,10}"
LATE_WINDOW = "late {60,70}"
EARLY_WINDOW_STEPS: tuple[int, ...] = (0, 10)
LATE_WINDOW_STEPS: tuple[int, ...] = (60, 70)

# Below this many parsed records a rate is not reported at all. Four is not a power calculation; it
# is the point below which a "rate" is a list of individual completions wearing a decimal point.
MIN_DENOMINATOR = 4

# A section this far from parsing is a measurement problem, not a measurement.
PARSE_FAILURE_BANNER_THRESHOLD = 0.25

# Movement toward the cooperative label here is the token-level-generalisation tell: in this game
# the label the other arms train toward pays nothing, so a policy that learned the WORD rather than
# the structure shows up as movement on this row and nowhere else it would make sense.
NEGATIVE_CONTROL_GAME_ID = "defective-coordination"

# The two arms whose only difference is the grading rule, which is the wave's designed contrast.
PAIR_ARMS: tuple[str, str] = ("twin-pd-group", "twin-pd-self")

# Read for their labels and denominators only. Dropping them at read time keeps the whole tree
# (292 MB of traces at the time of writing, mostly thinking tokens) from having to fit in memory at
# once; `games.report.sample_cot_excerpts` is where completion text is meant to be read.
HEAVY_TEXT_FIELDS: tuple[str, ...] = ("completion", "visible_text")

# The field each game's behaviour rate averages. The dictator game has no counterpart to cooperate
# with; its record carries the fraction the model kept for itself instead. The simultaneous-claim
# game has a counterpart and still no cooperative action: its record carries the fraction of the
# total it claimed, read against a half rather than against zero or one. The trust games answer with
# an amount, so theirs is the fraction of the stock sent -- or, for the never-trained trustee item,
# the share sent back.

BF16_ATTENUATION_CAVEAT = (
    "LoRA adapters were merged in bf16, which attenuates measured effects to roughly 64% of true "
    "(38-79% per module). Trends, orderings and contrasts survive that; absolute magnitudes are "
    "understated, so read every number below as a floor on its own size."
)

SINGLE_CELL_CAVEAT = (
    "A single-cell before/after is not a finding: a measured null on an unchanged model moved "
    "individual game-cells by up to 0.467 across replicates. Read the pooled windows and the "
    "ladder shape, never one step against another."
)

ALL_VARIANTS = "all"

# The all-arms grid's column groups. Each names one contrast, and the holdout groups take the game
# id itself, so a game added to the eval-only registry appears without an edit here.
GRID_TRAINED_PREFIX = "trained"
GRID_CANARY_PREFIX = "canary"
GRID_EDT_PREFIX = "edt_leaning"


@dataclass(frozen=True)
class DamagedCell:
    """A cell that named its arm and step but whose records could not all be read.

    Excluded from every table and named in the banner. It is a separate state from an unreadable
    file because it is recoverable by rerunning one battery, and because while batteries are still
    writing into the tree the common cause is a trace read mid-write.
    """

    path: Path
    arm: str
    step: int
    reason: str


@dataclass(frozen=True)
class Measure:
    """One rate with everything needed to know how much to believe it.

    `value` is None both when nothing parsed and when fewer than `MIN_DENOMINATOR` observations did,
    so the markdown and the JSON agree on which cells are unreportable; `n_parsed` beside it says
    which of the two happened.

    Two units, mirroring `games.battery_tables.Rate` so the two readouts describe one quantity the
    same way. `n_parsed`/`n_asked` count OBSERVATIONS -- the unit `value` is a mean over, which is a
    prompt for a behaviour rate and an item for the decision-theory leaning -- while
    `n_records_parsed`/`n_records` count the completions behind them. They coincide wherever an
    observation is one completion, which is every section except those two.
    """

    value: float | None
    n_parsed: int
    n_asked: int
    n_records_parsed: int
    n_records: int
    steps: tuple[int, ...]

    @property
    def parse_failure_rate(self) -> float | None:
        """The fraction of completions that produced no usable answer, or None over nothing.

        Computed from the record-level pair rather than the observation-level one, which is the
        whole reason both are carried: several draws of one prompt reduce to one observation, so
        `1 - n_parsed / n_records` would read a fully-parsing eight-draw cell as 87.5% failed.
        """
        if self.n_records == 0:
            return None
        return 1.0 - self.n_records_parsed / self.n_records


@dataclass(frozen=True)
class Readout:
    """Every table, keyed so the markdown and the JSON are two renderings of one structure."""

    generated_at: str
    evals_root: str
    git_sha: str | None
    status: dict[str, Any]
    arms: dict[str, dict[str, list[dict[str, Any]]]]
    cross_arm: dict[str, list[dict[str, Any]]]
    arm_facts: dict[str, dict[str, Any]]

    @property
    def error_containing(self) -> bool:
        """Whether anything in this readout is missing, damaged or barely parsing."""
        return bool(
            self.status["incomplete_arms"]
            or self.status["damaged_cells"]
            or self.status["high_parse_failure_sections"]
            or self.status["git_sha_disagreements"]
            or self.status["unregistered_arms"]
        )

    def as_json(self) -> dict[str, Any]:
        """Return the whole readout as JSON-safe values."""
        return {
            "generated_at": self.generated_at,
            "evals_root": self.evals_root,
            "git_sha": self.git_sha,
            "error_containing": self.error_containing,
            "windows": {"early": list(EARLY_WINDOW_STEPS), "late": list(LATE_WINDOW_STEPS)},
            "min_denominator": MIN_DENOMINATOR,
            "status": self.status,
            "arm_facts": self.arm_facts,
            "arms": self.arms,
            "cross_arm": self.cross_arm,
        }


def read_cell(path: Path) -> tuple[EvalTrace, str | None]:
    """Read one cell, returning its trace and a damage note when some lines could not be read.

    Raises when the file cannot be attributed at all -- no records, a first line that is not JSON,
    a first record that is not the meta record, or a meta record missing `arm` or `step`. A row
    labelled from a filename is worse than a missing row, because it still looks right.

    A JSONL line after the meta record that does not decode is counted rather than raised on, and
    the count comes back as the damage note: that is what a trace read while its battery is still
    writing looks like, and the caller excludes the cell and banners it.
    """
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty, so nothing in it can be attributed to a checkpoint.")
    try:
        meta = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path}'s first line is not JSON ({error}), so the trace cannot say which arm or "
            f"checkpoint produced it."
        ) from error
    if meta.get("record") != RECORD_META:
        raise ValueError(
            f"{path} does not start with a {RECORD_META!r} record, so its rows cannot be "
            f"attributed to a model or checkpoint."
        )
    missing = [name for name in report.REQUIRED_META_FIELDS if name not in meta]
    if missing:
        raise ValueError(
            f"{path}'s meta record is missing {missing}; a readout row labelled from a filename "
            f"can be silently wrong."
        )
    # The meta's own arm, or the consuming arm a banked copy's hash-verified sidecar names.
    arm, meta = report.attribute_trace(path, meta)
    records: list[Mapping[str, Any]] = []
    unreadable = 0
    for line in lines[1:]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            unreadable += 1
            continue
        records.append(
            {key: value for key, value in record.items() if key not in HEAVY_TEXT_FIELDS}
        )
    trace = EvalTrace(
        path=path,
        arm=arm,
        step=int(meta[report.META_STEP]),
        meta=meta,
        records=tuple(records),
    )
    if unreadable:
        note = f"{unreadable} of {len(lines) - 1} record line(s) did not decode as JSON"
        logger.warning(f"{path}: {note}; excluding this cell from every table")
        return trace, note
    return trace, None


def discover_cells(evals_root: Path) -> tuple[dict[str, list[EvalTrace]], list[DamagedCell]]:
    """Read every cell under `evals_root`, grouped by the arm its meta record names.

    Grouped by the meta record rather than the directory name, so a directory renamed or a trace
    copied into the wrong place cannot silently merge two arms. Two cells claiming the same arm and
    step raise: that is either a duplicate download or two runs writing one name, and averaging
    them would be a number nobody could reproduce.
    """
    paths = sorted(evals_root.glob(STEP_FILE_GLOB))
    if not paths:
        raise ValueError(
            f"no {STEP_FILE_GLOB!r} traces under {evals_root}; nothing to read a readout off."
        )
    by_arm: dict[str, list[EvalTrace]] = {}
    seen: dict[tuple[str, int], Path] = {}
    damaged: list[DamagedCell] = []
    for path in paths:
        trace, note = read_cell(path)
        key = (trace.arm, trace.step)
        if key in seen:
            raise ValueError(
                f"{path} and {seen[key]} both claim arm {trace.arm!r} at step {trace.step}; "
                f"a readout cannot say which is the measurement."
            )
        seen[key] = path
        if note is not None:
            damaged.append(DamagedCell(path=path, arm=trace.arm, step=trace.step, reason=note))
            continue
        by_arm.setdefault(trace.arm, []).append(trace)
    for traces in by_arm.values():
        traces.sort(key=lambda trace: trace.step)
    logger.info(
        f"read {len(paths)} cell(s) under {evals_root}: "
        f"{ {arm: len(traces) for arm, traces in sorted(by_arm.items())} }, "
        f"{len(damaged)} damaged"
    )
    return dict(sorted(by_arm.items())), damaged


def _behaviour_field(game_id: str) -> str:
    """Which field carries this game's behaviour rate; a two-action game's is the cooperation rate.

    Reads the one map in `games.evals`, which both readouts import rather than each keeping a copy:
    the copies disagreed the moment wave 2 added games to one and not the other, and the map is
    checked exhaustive against the prompt registry at import so an unlisted game raises instead of
    rendering an empty table under a cooperation-rate default.
    """
    return BEHAVIOUR_FIELD_BY_GAME[game_id]


def _behaviour_records(
    traces: Sequence[EvalTrace], game_id: str, *, payoff_variant: str | None = None
) -> list[Mapping[str, Any]]:
    """Every game-behaviour record for one game across these traces, optionally one variant only."""
    selected: list[Mapping[str, Any]] = []
    for trace in traces:
        for record in trace.section(SECTION_GAME_BEHAVIOR):
            if record.get("game_id") != game_id:
                continue
            if payoff_variant is not None and record.get("payoff_variant") != payoff_variant:
                continue
            selected.append(record)
    return selected


def game_measure(
    traces: Sequence[EvalTrace], game_id: str, *, payoff_variant: str | None = None
) -> Measure:
    """Return one game's behaviour rate pooled over these traces: coop, or keep for the dictator.

    Averaged within each (cell, prompt) before being averaged over them, which is what
    `games.evals._behaviour_rate` and `games.battery_tables.rate_of` do too: the battery draws
    `game_behavior_samples` completions per prompt, so pooling the records would weight a prompt
    whose draws mostly failed to parse differently from its neighbours. Identical to pooling on a
    cell drawn once per prompt.

    The prompt key carries the whole cell -- arm and step -- because this readout pools a WINDOW of
    several checkpoints and the same prompt id recurs in each. Keyed on the id alone, one
    checkpoint's draws of a prompt would merge with another's into a single observation, and the
    window would average steps before averaging prompts. The arm is redundant while every caller
    passes one arm's traces, and it is what stops a future cross-arm caller from merging them.
    """
    field_name = _behaviour_field(game_id)
    by_prompt: dict[str, list[float]] = {}
    asked: set[str] = set()
    n_records = 0
    for trace in traces:
        for record in _behaviour_records([trace], game_id, payoff_variant=payoff_variant):
            n_records += 1
            key = f"{trace.arm}@step{trace.step}::{record['prompt_id']}"
            asked.add(key)
            value = record.get(field_name)
            if value is not None:
                by_prompt.setdefault(key, []).append(float(value))
    prompt_means = [sum(values) / len(values) for values in by_prompt.values()]
    below_floor = len(prompt_means) < MIN_DENOMINATOR
    return Measure(
        value=None if below_floor else sum(prompt_means) / len(prompt_means),
        n_parsed=len(prompt_means),
        n_asked=len(asked),
        n_records_parsed=sum(len(values) for values in by_prompt.values()),
        n_records=n_records,
        steps=tuple(trace.step for trace in traces),
    )


def _window(traces: Sequence[EvalTrace], steps: Sequence[int]) -> list[EvalTrace]:
    """Return the traces in one pooled window, in step order."""
    return [trace for trace in traces if trace.step in steps]


def _games_present(traces: Sequence[EvalTrace]) -> list[str]:
    """Every game id these traces carry behaviour records for."""
    return sorted(
        {
            str(record["game_id"])
            for trace in traces
            for record in trace.section(SECTION_GAME_BEHAVIOR)
        }
    )


def _variants_present(traces: Sequence[EvalTrace], game_id: str) -> list[str]:
    """Every payoff variant these traces carry for one game."""
    return sorted({str(record["payoff_variant"]) for record in _behaviour_records(traces, game_id)})


def _steps_label(steps: Sequence[int]) -> str:
    """How a pooled window names the checkpoints that actually reached it."""
    return ",".join(str(step) for step in steps) if steps else "none"


def _contrast_row(early: Measure, late: Measure) -> dict[str, Any]:
    """Return the columns every early-versus-late row shares, including the delta and both windows."""
    delta = None if early.value is None or late.value is None else late.value - early.value
    return {
        "early_rate": early.value,
        "early_n_parsed": early.n_parsed,
        "early_n_asked": early.n_asked,
        "early_n_records": early.n_records,
        "early_steps": _steps_label(early.steps),
        "late_rate": late.value,
        "late_n_parsed": late.n_parsed,
        "late_n_asked": late.n_asked,
        "late_n_records": late.n_records,
        "late_steps": _steps_label(late.steps),
        "delta": delta,
    }


def _trained_variants(arm: str) -> tuple[str, ...]:
    """Return the payoff variants this arm actually trained on, empty meaning all of them."""
    registered = ARMS.get(arm)
    return () if registered is None else registered.payoff_variants


def trained_game_of(arm: str) -> str | None:
    """Return the one game this arm trains, or None when the arm is not in `games.arms.ARMS`."""
    registered = ARMS.get(arm)
    return None if registered is None else registered.game_id


def trained_ladder_rows(arm: str, traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Per-step rate on the arm's trained game, split by payoff variant where the arm pins one.

    The stag-hunt rung arms train one rung of a four-rung risk-dominance ladder, so their own rung
    is the trained readout and the other three are within-game dose-response transfer: the same
    game at payoffs the arm never trained. Splitting them is the difference between reading a dose
    response and averaging it away.
    """
    game_id = trained_game_of(arm)
    if game_id is None:
        return []
    pinned = _trained_variants(arm)
    variants: list[str | None] = list(_variants_present(traces, game_id)) if pinned else [None]
    rows: list[dict[str, Any]] = []
    for trace in traces:
        for variant in variants:
            measure = game_measure([trace], game_id, payoff_variant=variant)
            rows.append(
                {
                    "step": trace.step,
                    "game": game_id,
                    "payoff_variant": variant or ALL_VARIANTS,
                    "role": _variant_role(variant, pinned),
                    "rate": measure.value,
                    "n_parsed": measure.n_parsed,
                    "n_asked": measure.n_asked,
                    "n_records": measure.n_records,
                    "parse_failure_rate": measure.parse_failure_rate,
                }
            )
    return rows


def _variant_role(variant: str | None, pinned: Sequence[str]) -> str:
    """Whether a payoff variant is the one this arm trained, or within-game transfer."""
    if not pinned or variant is None:
        return "trained"
    return "trained rung" if variant in pinned else "transfer rung (within-game)"


def trained_window_rows(arm: str, traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the pooled early-versus-late contrast on the trained game, per payoff variant."""
    game_id = trained_game_of(arm)
    if game_id is None:
        return []
    pinned = _trained_variants(arm)
    variants: list[str | None] = list(_variants_present(traces, game_id)) if pinned else [None]
    early_traces = _window(traces, EARLY_WINDOW_STEPS)
    late_traces = _window(traces, LATE_WINDOW_STEPS)
    return [
        {
            "game": game_id,
            "payoff_variant": variant or ALL_VARIANTS,
            "role": _variant_role(variant, pinned),
            **_contrast_row(
                game_measure(early_traces, game_id, payoff_variant=variant),
                game_measure(late_traces, game_id, payoff_variant=variant),
            ),
        }
        for variant in variants
    ]


def transfer_rows(arm: str, traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the pooled window contrast on every game this arm never trained.

    The eval-only holdouts are marked, and `defective-coordination` is marked again as the negative
    control: the cooperative label pays nothing there, so movement toward it is a token-level
    generalisation tell rather than transfer of the strategic disposition.
    """
    trained = trained_game_of(arm)
    early_traces = _window(traces, EARLY_WINDOW_STEPS)
    late_traces = _window(traces, LATE_WINDOW_STEPS)
    rows: list[dict[str, Any]] = []
    for game_id in _games_present(traces):
        if game_id == trained:
            continue
        rows.append(
            {
                "game": game_id,
                "metric": _behaviour_field(game_id),
                "eval_only_holdout": game_id in EVAL_ONLY_GAME_IDS,
                "negative_control": game_id == NEGATIVE_CONTROL_GAME_ID,
                **_contrast_row(
                    game_measure(early_traces, game_id),
                    game_measure(late_traces, game_id),
                ),
            }
        )
    return rows


def _probe_records(traces: Sequence[EvalTrace], *, step_scoped: bool) -> list[Mapping[str, Any]]:
    """Every probe record across these traces.

    `step_scoped` rewrites each record's probe id to carry its checkpoint. That is needed for
    `order_disagreement_rate` and nothing else: it keys a counterbalanced pair on
    (probe_id, sample_index), so pooling two checkpoints under a shared id would have one
    checkpoint's answer overwrite the other's and the rate would silently be read off half the
    window. Per-item means deliberately keep the shared id, so an item's two checkpoints average
    into one observation and the denominator stays "items".
    """
    records: list[Mapping[str, Any]] = []
    for trace in traces:
        for record in trace.section(SECTION_DT_PROBES):
            if step_scoped:
                records.append({**record, "probe_id": f"{record['probe_id']}@step{trace.step}"})
            else:
                records.append(record)
    return records


def _theory_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count each theory label across the open-ended probe records that parsed.

    Every label in `games.parsing.THEORY_LABELS` gets a column whether or not it appeared, so the
    table has one shape across arms and a zero is visibly a zero rather than a missing column.
    """
    counts = dict.fromkeys(THEORY_LABELS, 0)
    for record in records:
        theory = record.get("theory")
        if theory is not None:
            counts[str(theory)] = counts.get(str(theory), 0) + 1
    return counts


def _edt_measure(traces: Sequence[EvalTrace]) -> Measure:
    """Mean EDT leaning over probe items, as a `Measure` so it can sit beside a behaviour rate.

    One observation per item: an item's two option orders and every sample of it average first, so
    `n_parsed` counts items while `n_records` counts the probe records behind them.
    """
    pooled = _probe_records(traces, step_scoped=False)
    leanings = sorted(per_item_means(pooled, "edt_leaning").values())
    below_floor = len(leanings) < MIN_DENOMINATOR
    return Measure(
        value=None if below_floor else sum(leanings) / len(leanings),
        n_parsed=len(leanings),
        n_asked=len({str(record["probe_id"]) for record in pooled}),
        n_records_parsed=sum(1 for record in pooled if record.get("edt_leaning") is not None),
        n_records=len(pooled),
        steps=tuple(trace.step for trace in traces),
    )


def _dt_row(traces: Sequence[EvalTrace]) -> dict[str, Any]:
    """Return the decision-theory numbers for one step or one pooled window."""
    theory_records = [
        record
        for record in _probe_records(traces, step_scoped=False)
        if record.get("kind") == "open-ended"
    ]
    counts = _theory_counts(theory_records)
    leaning = _edt_measure(traces)
    return {
        **{f"n_{label}": count for label, count in counts.items()},
        "n_theory_parsed": sum(counts.values()),
        "n_theory_records": len(theory_records),
        "mean_edt_leaning": leaning.value,
        "n_edt_items": leaning.n_parsed,
        "order_disagreement_rate": order_disagreement_rate(
            _probe_records(traces, step_scoped=True)
        ),
        "n_probe_records": leaning.n_records,
    }


def dt_ladder_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the decision-theory ladder: theory counts, EDT leaning, order disagreement per step."""
    return [{"step": trace.step, **_dt_row([trace])} for trace in traces]


def dt_window_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the decision-theory numbers pooled over the early and late windows."""
    rows: list[dict[str, Any]] = []
    for name, steps in ((EARLY_WINDOW, EARLY_WINDOW_STEPS), (LATE_WINDOW, LATE_WINDOW_STEPS)):
        window_traces = _window(traces, steps)
        rows.append(
            {
                "window": name,
                "steps_pooled": _steps_label([trace.step for trace in window_traces]),
                **_dt_row(window_traces),
            }
        )
    return rows


def _capability_measure(traces: Sequence[EvalTrace]) -> Measure:
    """Arithmetic-canary accuracy pooled over these traces, over the items that parsed.

    An item whose answer could not be parsed leaves the numerator and stays in `n_records`, which
    is the denominator convention every other rate here follows. It does mean this is accuracy
    among answered items rather than over all 50 asked; both counts are in the row, so the stricter
    reading (an unparseable answer as a wrong answer) is `n_parsed * accuracy / n_records`.
    """
    records = [record for trace in traces for record in trace.section(SECTION_CAPABILITIES)]
    values = [float(bool(record["correct"])) for record in records if record.get("parsed")]
    below_floor = len(values) < MIN_DENOMINATOR
    return Measure(
        value=None if below_floor else sum(values) / len(values),
        n_parsed=len(values),
        n_asked=len(records),
        n_records_parsed=len(values),
        n_records=len(records),
        steps=tuple(trace.step for trace in traces),
    )


def capability_ladder_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Per-step arithmetic-canary accuracy, so a behaviour shift can be read against capability."""
    rows: list[dict[str, Any]] = []
    for trace in traces:
        measure = _capability_measure([trace])
        rows.append(
            {
                "step": trace.step,
                "accuracy": measure.value,
                "n_parsed": measure.n_parsed,
                "n_items": measure.n_records,
            }
        )
    return rows


def capability_window_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the pooled early-versus-late canary contrast."""
    early = _capability_measure(_window(traces, EARLY_WINDOW_STEPS))
    late = _capability_measure(_window(traces, LATE_WINDOW_STEPS))
    return [{"metric": "arithmetic accuracy", **_contrast_row(early, late)}]


def payoff_variant_rows(cells: Mapping[str, list[EvalTrace]]) -> list[dict[str, Any]]:
    """Each arm's trained game split by payoff magnitude, early versus late.

    The magnitudes are a dose axis that a pooled game rate hides: twin-pd's temptation-2 and
    temptation-10 rows are the same game with the defection payoff five times apart, and the stag
    hunt's four rungs are its risk-dominance ladder.
    """
    rows: list[dict[str, Any]] = []
    for arm, traces in cells.items():
        game_id = trained_game_of(arm)
        if game_id is None:
            continue
        pinned = _trained_variants(arm)
        early_traces = _window(traces, EARLY_WINDOW_STEPS)
        late_traces = _window(traces, LATE_WINDOW_STEPS)
        rows.extend(
            {
                "arm": arm,
                "game": game_id,
                "payoff_variant": variant,
                "trained_on_this_variant": not pinned or variant in pinned,
                **_contrast_row(
                    game_measure(early_traces, game_id, payoff_variant=variant),
                    game_measure(late_traces, game_id, payoff_variant=variant),
                ),
            }
            for variant in _variants_present(traces, game_id)
        )
    return rows


def dictator_rows(cells: Mapping[str, list[EvalTrace]]) -> list[dict[str, Any]]:
    """Keep-fraction on the dictator items, which every arm's battery carries, per arm.

    Six items per cell in every battery, trained or not, which makes this the selfishness-spillover
    readout: an arm that never saw an allocation task is still asked the same six.
    """
    rows: list[dict[str, Any]] = []
    for arm, traces in cells.items():
        early = game_measure(_window(traces, EARLY_WINDOW_STEPS), DICTATOR_GAME_ID)
        late = game_measure(_window(traces, LATE_WINDOW_STEPS), DICTATOR_GAME_ID)
        rows.append(
            {
                "arm": arm,
                "metric": KEEP_FIELD,
                "trained_on_dictator": trained_game_of(arm) == DICTATOR_GAME_ID,
                **_contrast_row(early, late),
            }
        )
    return rows


def grid_rows(cells: Mapping[str, list[EvalTrace]]) -> list[dict[str, Any]]:
    """One row per arm: trained game, every holdout, the canary and EDT leaning, early versus late.

    The whole-wave view. Deltas only make sense read down a column -- across a row they are four
    different quantities -- which is why each stays in its own named column rather than being
    reduced to one score.
    """
    rows: list[dict[str, Any]] = []
    for arm, traces in cells.items():
        early_traces = _window(traces, EARLY_WINDOW_STEPS)
        late_traces = _window(traces, LATE_WINDOW_STEPS)
        game_id = trained_game_of(arm)
        pinned = _trained_variants(arm)
        variant = pinned[0] if pinned else None
        row: dict[str, Any] = {
            "arm": arm,
            "cells": len(traces),
            "trained_game": game_id or "unknown",
            "trained_payoff_variant": variant or ALL_VARIANTS,
        }
        measurers: list[tuple[str, Callable[[Sequence[EvalTrace]], Measure]]] = [
            (
                GRID_TRAINED_PREFIX,
                _null_measurer
                if game_id is None
                else _game_measurer(game_id, payoff_variant=variant),
            ),
            *((holdout, _game_measurer(holdout)) for holdout in EVAL_ONLY_GAME_IDS),
            (GRID_CANARY_PREFIX, _capability_measure),
            (GRID_EDT_PREFIX, _edt_measure),
        ]
        for prefix, measure_of in measurers:
            contrast = _contrast_row(measure_of(early_traces), measure_of(late_traces))
            row.update(_prefixed(contrast, prefix))
        rows.append(row)
    return rows


def _delta(early: float | None, late: float | None) -> float | None:
    """Late minus early, or None when either side was unreportable."""
    return None if early is None or late is None else late - early


def _prefixed(row: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Namespace one contrast's columns so several can sit in one grid row."""
    return {f"{prefix}_{key}": value for key, value in row.items()}


def pair_contrast_rows(cells: Mapping[str, list[EvalTrace]]) -> list[dict[str, Any]]:
    """Return the designed contrast: identical twin-PD prompts, group-mix against self grading.

    Rendered whenever either arm is present, incomplete or not. The self arm's battery is still
    running at the time of writing, so its late window is expected to be empty; an empty cell that
    says which checkpoints reached it is the honest form of that, and leaving the table out until
    the data lands would mean nobody looks at the half that exists.
    """
    present = [arm for arm in PAIR_ARMS if arm in cells]
    if not present:
        return []
    trained_game = trained_game_of(PAIR_ARMS[0])
    transfer_games = sorted(
        {game for arm in present for game in _games_present(cells[arm]) if game != trained_game}
    )
    rows: list[dict[str, Any]] = []
    if trained_game is not None:
        rows.append(
            _pair_row(f"trained game: {trained_game}", cells, present, _game_measurer(trained_game))
        )
    rows.extend(
        _pair_row(f"transfer: {game}", cells, present, _game_measurer(game))
        for game in transfer_games
    )
    rows.append(_pair_row("canary: arithmetic accuracy", cells, present, _capability_measure))
    rows.append(_pair_row("decision theory: mean_edt_leaning", cells, present, _edt_measure))
    return rows


def _game_measurer(
    game_id: str, *, payoff_variant: str | None = None
) -> Callable[[Sequence[EvalTrace]], Measure]:
    """Bind one game's behaviour rate, so every row of a table is measured the same way."""
    return lambda traces: game_measure(traces, game_id, payoff_variant=payoff_variant)


def _null_measurer(traces: Sequence[EvalTrace]) -> Measure:
    """Stand in for a measure that cannot be taken, for an arm outside `games.arms.ARMS`."""
    return Measure(
        value=None,
        n_parsed=0,
        n_asked=0,
        n_records_parsed=0,
        n_records=0,
        steps=tuple(t.step for t in traces),
    )


def _pair_row(
    label: str,
    cells: Mapping[str, list[EvalTrace]],
    present: Sequence[str],
    measure_of: Callable[[Sequence[EvalTrace]], Measure],
) -> dict[str, Any]:
    """One measure of the pair contrast, side by side, with each arm's own denominators."""
    row: dict[str, Any] = {"measure": label}
    for arm in PAIR_ARMS:
        if arm not in present:
            row.update(_prefixed({"early": None, "late": None, "delta": None, "n": "-"}, arm))
            continue
        traces = cells[arm]
        early = measure_of(_window(traces, EARLY_WINDOW_STEPS))
        late = measure_of(_window(traces, LATE_WINDOW_STEPS))
        row.update(
            _prefixed(
                {
                    "early": early.value,
                    "late": late.value,
                    "delta": _delta(early.value, late.value),
                    "n": f"{early.n_parsed}/{early.n_asked} then {late.n_parsed}/{late.n_asked}",
                },
                arm,
            )
        )
    return row


def _git_sha_status(
    cells: Mapping[str, list[EvalTrace]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Return the git SHA the cells agree on, plus any cell that disagrees with the majority.

    A readout pooling traces written by two versions of the eval code is comparing two experiments,
    so the agreement is asserted rather than assumed -- and reported rather than raised on, because
    which cells disagree is the thing worth knowing.

    Banked copies are left out of the vote: a step-0 cell taken from the shared bank was generated
    by whatever code revision produced the bank entry, and that revision is named in the cell's own
    `banked_cells` status entry rather than counted here as a disagreement, where it would mark
    every banked battery error-containing by design.
    """
    by_sha: dict[str, list[str]] = {}
    for arm, traces in cells.items():
        for trace in traces:
            if trace.banked_from is not None:
                continue
            sha = str(trace.meta.get("git_sha"))
            by_sha.setdefault(sha, []).append(f"{arm}@{trace.step}")
    if not by_sha:
        return None, []
    majority = max(by_sha, key=lambda sha: len(by_sha[sha]))
    disagreements = [
        {"git_sha": sha, "cells": sorted(labels)}
        for sha, labels in sorted(by_sha.items())
        if sha != majority
    ]
    if disagreements:
        logger.warning(f"traces disagree on git_sha: majority {majority}, others {disagreements}")
    return majority, disagreements


def _parse_failure_status(cells: Mapping[str, list[EvalTrace]]) -> list[dict[str, Any]]:
    """Every (cell, section) whose parse failure rate passes the banner threshold."""
    flagged: list[dict[str, Any]] = []
    for arm, traces in cells.items():
        for trace in traces:
            for section in (SECTION_GAME_BEHAVIOR, SECTION_DT_PROBES, SECTION_CAPABILITIES):
                records = trace.section(section)
                if not records:
                    continue
                failures = sum(1 for record in records if not record.get("parsed"))
                rate = failures / len(records)
                if rate > PARSE_FAILURE_BANNER_THRESHOLD:
                    flagged.append(
                        {
                            "arm": arm,
                            "step": trace.step,
                            "section": section,
                            "parse_failure_rate": rate,
                            "n_records": len(records),
                        }
                    )
    return flagged


def _declaration_notes(cells: Mapping[str, list[EvalTrace]]) -> list[dict[str, Any]]:
    """Where a cell's own `trained_game_ids` disagrees with the arm registry.

    Not a table anyone reads for a result; it is the check that the registry mapping this whole
    readout derives from still matches what the traces claim. An empty list at step 0 is correct
    (the base model trained nothing) and is not flagged.
    """
    notes: list[dict[str, Any]] = []
    for arm, traces in cells.items():
        expected = trained_game_of(arm)
        for trace in traces:
            config = trace.meta.get("eval_config") or {}
            declared = [str(game) for game in config.get("trained_game_ids") or []]
            if not declared:
                if trace.step != 0:
                    notes.append(
                        {
                            "arm": arm,
                            "step": trace.step,
                            "note": "cell declares no trained game past step 0; the readout uses "
                            f"the registry's {expected!r}",
                        }
                    )
                continue
            if expected is None or set(declared) != {expected}:
                notes.append(
                    {
                        "arm": arm,
                        "step": trace.step,
                        "note": f"cell declares trained_game_ids={declared} against the "
                        f"registry's {expected!r}",
                    }
                )
    for note in notes:
        logger.warning(f"trained-game declaration: {note}")
    return notes


def _banked_cells_status(cells: Mapping[str, list[EvalTrace]]) -> list[dict[str, Any]]:
    """Name every cell that is a byte-identical copy of a shared bank entry, and where it came from.

    Informational rather than an error: a banked base-model cell is the owner's design (the bank
    exists so one base draw serves every arm on that base), but a reader has to be able to see that
    an arm's step 0 was generated under another arm's name at another commit, so the entry carries
    the source arm, the code revision that generated it and the bank key it was copied from. The
    sidecar's sha256 was already checked against the file when the cell was read.

    The key rather than the sidecar's `source_key` prefix, which carries a bucket name: this status
    is rendered into a readout that gets pasted into docs, and the key identifies the entry to
    anyone holding the bank without naming where the bank is.
    """
    banked: list[dict[str, Any]] = []
    for arm, traces in cells.items():
        for trace in traces:
            provenance = trace.banked_from
            if provenance is None:
                continue
            banked.append(
                {
                    "arm": arm,
                    "step": trace.step,
                    "source_arm": provenance.get("source_arm"),
                    "source_git_sha": provenance.get("source_git_sha"),
                    "bank_key": provenance.get("bank_key"),
                    "banked_at": provenance.get("banked_at"),
                    "copied_at": provenance.get("copied_at"),
                }
            )
            logger.info(
                f"{arm}@{trace.step} is a banked copy: generated under {provenance.get('source_arm')} "
                f"at {provenance.get('source_git_sha')}, bank key {provenance.get('bank_key')}"
            )
    return banked


def _unregistered_arms(cells: Mapping[str, list[EvalTrace]]) -> list[str]:
    """Return the arms `games.arms.ARMS` does not know, whose trained-game tables must stay empty.

    An arm label is free-form, so an unregistered one means the trained-versus-transfer split
    genuinely cannot be filled in: with nothing saying which game it trained, its whole battery
    reads as transfer. Bannered rather than guessed at from the label.
    """
    unregistered = [arm for arm in cells if arm not in ARMS]
    for arm in unregistered:
        logger.warning(
            f"arm {arm!r} is not in games.arms.ARMS, so its trained game is unknown; its "
            f"trained-game tables are empty and every game it evaluated appears as transfer"
        )
    return unregistered


def build_readout(evals_root: Path) -> Readout:
    """Read every cell under `evals_root` and reduce it to every table the readout renders."""
    cells, damaged = discover_cells(evals_root)
    git_sha, sha_disagreements = _git_sha_status(cells)
    incomplete = [
        {
            "arm": arm,
            "cells_found": len(traces),
            "cells_expected": EXPECTED_CELLS_PER_ARM,
            "steps": [trace.step for trace in traces],
        }
        for arm, traces in cells.items()
        if len(traces) < EXPECTED_CELLS_PER_ARM
    ]
    status: dict[str, Any] = {
        "arms": {
            arm: {
                "cells_found": len(traces),
                "cells_expected": EXPECTED_CELLS_PER_ARM,
                "steps": [trace.step for trace in traces],
                "complete": len(traces) >= EXPECTED_CELLS_PER_ARM,
            }
            for arm, traces in cells.items()
        },
        "incomplete_arms": incomplete,
        "damaged_cells": [
            {
                "arm": cell.arm,
                "step": cell.step,
                "path": str(cell.path),
                "reason": cell.reason,
            }
            for cell in damaged
        ],
        "high_parse_failure_sections": _parse_failure_status(cells),
        "git_sha_disagreements": sha_disagreements,
        "trained_game_declaration_notes": _declaration_notes(cells),
        "unregistered_arms": _unregistered_arms(cells),
        "banked_cells": _banked_cells_status(cells),
    }
    arms = {
        arm: {
            "trained_ladder": trained_ladder_rows(arm, traces),
            "trained_windows": trained_window_rows(arm, traces),
            "transfer": transfer_rows(arm, traces),
            "dt_ladder": dt_ladder_rows(traces),
            "dt_windows": dt_window_rows(traces),
            "capability_ladder": capability_ladder_rows(traces),
            "capability_windows": capability_window_rows(traces),
        }
        for arm, traces in cells.items()
    }
    arm_facts = {
        arm: {
            "trained_game": trained_game_of(arm),
            "trained_payoff_variants": list(_trained_variants(arm)),
            "steps": [trace.step for trace in traces],
            "grading": sorted({str(trace.meta.get("grading")) for trace in traces}),
            "backend_model_id": sorted(
                {str(trace.meta.get("backend_model_id")) for trace in traces}
            ),
        }
        for arm, traces in cells.items()
    }
    cross_arm = {
        "pair_contrast": pair_contrast_rows(cells),
        "grid": grid_rows(cells),
        "payoff_variant_splits": payoff_variant_rows(cells),
        "dictator_keep_fraction": dictator_rows(cells),
    }
    return Readout(
        generated_at=datetime.now(UTC).isoformat(),
        evals_root=str(evals_root),
        git_sha=git_sha,
        status=status,
        arms=arms,
        cross_arm=cross_arm,
        arm_facts=arm_facts,
    )


def _table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render one table's rows as markdown, or say so when a table has no rows."""
    return markdown_table(pd.DataFrame(list(rows)))


def _grid_markdown_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compress the grid to one arrow cell per measure, so the whole wave fits a screen width.

    The numeric components stay in the JSON; this is the same rows read differently, not a second
    reduction of the data.
    """
    compressed: list[dict[str, Any]] = []
    for row in rows:
        variant = row["trained_payoff_variant"]
        trained = row["trained_game"] + ("" if variant == ALL_VARIANTS else f"/{variant}")
        cell: dict[str, Any] = {
            "arm": row["arm"],
            "cells": row["cells"],
            "trained game": trained,
            "trained": _arrow(row, GRID_TRAINED_PREFIX),
        }
        for holdout in EVAL_ONLY_GAME_IDS:
            cell[f"holdout {holdout}"] = _arrow(row, holdout)
        cell["canary"] = _arrow(row, GRID_CANARY_PREFIX)
        cell["mean_edt_leaning"] = _arrow(row, GRID_EDT_PREFIX)
        compressed.append(cell)
    return compressed


def _arrow(row: Mapping[str, Any], prefix: str) -> str:
    """One grid cell: early to late, with the early parsed count for scale."""
    return _arrow_values(
        row[f"{prefix}_early_rate"], row[f"{prefix}_late_rate"], row[f"{prefix}_early_n_parsed"]
    )


def _arrow_values(early: float | None, late: float | None, n_parsed: int) -> str:
    """Format one early-to-late pair, marking a window that had no reportable denominator."""
    render = "-" if early is None else f"{early:.3f}"
    render += " -> " + ("-" if late is None else f"{late:.3f}")
    return f"{render} (n={n_parsed})"


def _banked_lines(readout: Readout) -> list[str]:
    """Return the provenance notes for banked copies, which are not errors and get no error header."""
    entries = readout.status["banked_cells"]
    if not entries:
        return []
    lines = [
        (
            "Provenance notes, not errors: the cells below are byte-identical copies of shared "
            "base-model bank entries, labelled with the arm that took them by their provenance "
            "sidecar and sha256-verified when read. Each is that arm's step-0 cell by design; its "
            "own meta names the arm and commit that generated the bank entry."
        ),
        "",
    ]
    lines.extend(
        f"- BANKED cell `{entry['arm']}@{entry['step']}`: copied from bank key "
        f"`{entry['bank_key']}`, generated under arm `{entry['source_arm']}` at git SHA "
        f"`{entry['source_git_sha']}`, banked {entry['banked_at']}, copied {entry['copied_at']}."
        for entry in entries
    )
    lines.append("")
    return lines


def _banner(readout: Readout) -> list[str]:
    """Return the header banner: what is missing, what is damaged, and what barely parsed.

    The banked-copy notes follow the error banner (`_banked_lines`) and never trigger it: a banked
    cell is the design, and a document that shouted ERROR-CONTAINING over one would train its
    reader to ignore the header.
    """
    status = readout.status
    lines: list[str] = [
        f"- INCOMPLETE arm `{entry['arm']}`: {entry['cells_found']} of "
        f"{entry['cells_expected']} cells present (steps {entry['steps']}). Its late window is "
        f"whatever of {list(LATE_WINDOW_STEPS)} has landed, and may be empty."
        for entry in status["incomplete_arms"]
    ]
    lines.extend(
        f"- DAMAGED cell `{entry['arm']}@{entry['step']}` ({entry['path']}): {entry['reason']}. "
        f"Excluded from every table below."
        for entry in status["damaged_cells"]
    )
    lines.extend(
        f"- HIGH PARSE FAILURE `{entry['arm']}@{entry['step']}` section `{entry['section']}`: "
        f"{entry['parse_failure_rate']:.3f} of {entry['n_records']} records produced no usable "
        f"answer."
        for entry in status["high_parse_failure_sections"]
    )
    lines.extend(
        f"- GIT SHA DISAGREEMENT: cells {entry['cells']} were written at `{entry['git_sha']}`, "
        f"not the majority `{readout.git_sha}`."
        for entry in status["git_sha_disagreements"]
    )
    lines.extend(
        f"- TRAINED-GAME DECLARATION `{entry['arm']}@{entry['step']}`: {entry['note']}."
        for entry in status["trained_game_declaration_notes"]
    )
    lines.extend(
        f"- UNREGISTERED arm `{arm}`: not in `games.arms.ARMS`, so nothing says which game it "
        f"trained. Its trained-game tables are empty and every game it evaluated reads as transfer."
        for arm in status["unregistered_arms"]
    )
    if not lines:
        return _banked_lines(readout)
    header = (
        "> **THIS READOUT IS INCOMPLETE AND ERROR-CONTAINING.** The conditions below hold at the "
        "time it was generated; rerun it once the batteries finish and it updates itself."
    )
    return [header, "", *lines, "", *_banked_lines(readout)]


def render_markdown(readout: Readout) -> str:
    """Render the whole readout as one markdown document."""
    counts = ", ".join(
        f"{arm} {facts['cells_found']}/{facts['cells_expected']}"
        for arm, facts in readout.status["arms"].items()
    )
    sections: list[str] = [
        "# Games RL cross-arm readout",
        "",
        (
            f"Generated {readout.generated_at} from `{readout.evals_root}`; traces written at git "
            f"SHA `{readout.git_sha}`. Cells per arm: {counts}."
        ),
        "",
        *_banner(readout),
        "## How to read this",
        "",
        (
            f"- Pooled windows: **{EARLY_WINDOW}** against **{LATE_WINDOW}**. Each window pools "
            f"the records of both checkpoints, and every row names the steps that reached it."
        ),
        f"- {SINGLE_CELL_CAVEAT}",
        (
            "- Every rate is over what parsed, with both denominators beside it. A behaviour rate "
            "averages within a PROMPT before averaging over prompts, so `n_parsed`/`n_asked` count "
            "prompts and `n_records` the completions drawn for them; an unparsed prompt leaves the "
            f"numerator and stays in `n_asked`. A rate over fewer than {MIN_DENOMINATOR} parsed "
            "observations renders `-`."
        ),
        f"- {BF16_ATTENUATION_CAVEAT}",
        (
            "- Descriptive only. Nothing here is a verdict, and a flat or surprising row is a lead "
            "to chase in the traces rather than a result."
        ),
        "",
    ]
    for arm, tables in readout.arms.items():
        facts = readout.arm_facts[arm]
        sections.extend(
            [
                f"## Arm `{arm}`",
                "",
                (
                    f"Trained game `{facts['trained_game']}`, payoff variants "
                    f"{facts['trained_payoff_variants'] or 'all'}, grading {facts['grading']}, "
                    f"steps {facts['steps']}."
                ),
                "",
                "### Trained-game ladder",
                "",
                _table(tables["trained_ladder"]),
                "",
                "### Trained-game pooled windows",
                "",
                _table(tables["trained_windows"]),
                "",
                "### Transfer: games this arm never trained",
                "",
                _table(tables["transfer"]),
                "",
                "### Decision-theory ladder",
                "",
                _table(tables["dt_ladder"]),
                "",
                "### Decision theory, pooled windows",
                "",
                _table(tables["dt_windows"]),
                "",
                "### Capability canary ladder",
                "",
                _table(tables["capability_ladder"]),
                "",
                _table(tables["capability_windows"]),
                "",
            ]
        )
    sections.extend(
        [
            "## Cross-arm: the pair contrast",
            "",
            (
                f"`{PAIR_ARMS[0]}` against `{PAIR_ARMS[1]}` -- identical twin-PD prompts, "
                "group-mix grading against self grading. Each `_n` column reads "
                "`parsed/prompts then parsed/prompts` for the two windows."
            ),
            "",
            _table(readout.cross_arm["pair_contrast"]),
            "",
            "## Cross-arm: the whole wave at a glance",
            "",
            "Each cell is `early -> late (n=parsed in the early window)`.",
            "",
            _table(_grid_markdown_rows(readout.cross_arm["grid"])),
            "",
            "## Cross-arm: trained game by payoff variant",
            "",
            _table(readout.cross_arm["payoff_variant_splits"]),
            "",
            "## Cross-arm: dictator keep fraction (present in every battery)",
            "",
            _table(readout.cross_arm["dictator_keep_fraction"]),
            "",
        ]
    )
    return "\n".join(sections)


def write_readout(evals_root: Path, out_dir: Path) -> tuple[Path, Path]:
    """Build the readout and write `readout.md` beside `readout.json`."""
    readout = build_readout(evals_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = out_dir / "readout.md"
    json_path = out_dir / "readout.json"
    markdown_path.write_text(render_markdown(readout), encoding="utf-8")
    json_path.write_text(json.dumps(readout.as_json(), indent=2, default=str), encoding="utf-8")
    logger.info(f"wrote {markdown_path} and {json_path}")
    return markdown_path, json_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evals-root",
        type=Path,
        default=DEFAULT_EVALS_ROOT,
        help="Directory holding <prefix>/<arm>/step-N.jsonl eval traces.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where readout.md and readout.json are written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Rebuild the readout from whatever traces are on disk right now."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    write_readout(args.evals_root, args.out_dir)


if __name__ == "__main__":
    main()
