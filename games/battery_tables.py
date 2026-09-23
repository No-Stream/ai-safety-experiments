# HARNESS-SCAN-EXEMPT-monolithic-file-loc -- one responsibility; the split vs games/readout.py is an owner decision
r"""The battery readout's middle layer: reduce the cells to every table the document is made of.

Three layers, one per thing that can go wrong. `games.battery_cells` finds and reads the cells,
this module reduces them to tables, and `games.battery_readout` renders and ships the document.
Nothing here formats a heading, nothing there averages a record, and a table can be recomputed or
read in a notebook without rendering any markdown.

Three properties it is built around, in order of how easily each is lost. **Every rate carries its
denominator**: `Rate` holds the value, how many records produced it, how many were asked and how
many were truncated, because a section drifting into the generation cap is measuring the cap and
that is invisible in a rate. **Ladders, never a single before-and-after pair**: a measured null on
an unchanged model moved individual game-cells by up to 0.467 across replicates, so every builder
returns one row per step present and the split contrasts pool a window. **Nothing about the slate is
hardcoded**: `battery_ladder` derives the checkpoint ladder from the steps the arms reached,
`late_window` takes the upper half of it, and `contrast_groups` groups the arms present onto their
trained game, so landing a checkpoint or registering an arm changes the tables without changing this
file.

Reused rather than reimplemented: `games.report.EvalTrace` and `load_traces` (the trace data model
and its meta-field requirements, so an unattributable cell raises here for the same reason it raises
there), `per_round_profile` and `theory_item_flips`, and `order_disagreement_rate` and
`per_item_means`, which report itself borrowed from `games.evals` because two independent reductions
of one quantity is how a table comes to disagree with the summary beside it.

The arm-to-trained-game mapping comes from `games.arms.ARMS`, never from a trace: a record's
`trained_game` flag is slate-level, and `eval_config.trained_game_ids` is empty for every step-0
cell (correctly, since step 0 is the un-adapted base model), so an arm whose only landed cell is
step 0 would read as having trained nothing.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from games.arms import ARMS
from games.battery_cells import SECTION_DIGESTS_KEY, ExcludedCell, read_cells
from games.evals import (
    BEHAVIOUR_FIELD_BY_GAME,
    COOP_FIELD,
    KEEP_FIELD,
    ONE_SHOT_ACTION_GAME_IDS,
    SECTION_CAPABILITIES,
    SECTION_DT_PROBES,
    SECTION_FRAMING_SWEEP,
    SECTION_GAME_BEHAVIOR,
    SECTION_SELF_REPORT,
    SECTIONS,
)
from games.parsing import THEORY_LABELS
from games.position_preference import BAND_NONE, BAND_PAIRED, PositionSplit, position_split
from games.probes import PROBE_MULTIPLE_CHOICE, PROBE_OPEN_ENDED
from games.prompts import DICTATOR_GAME_ID, EVAL_ONLY_GAME_IDS
from games.report import (
    EvalTrace,
    answers_by_probe,
    order_disagreement_rate,
    per_item_means,
    per_round_profile,
    theory_item_flips,
)
from games.survey import (
    FAMILY_NEGATIVE_CONTROL,
    FAMILY_SELF_PREDICTION,
    FAMILY_VALUES_FORCED_CHOICE,
    INSTRUMENT_SVO_SLIDER,
    NON_SOCIAL_LABEL_PREFIX,
    SURVEY_NUMERIC,
    SURVEY_TAGGED,
    WORDING_AS_PUBLISHED,
    Acquiescence,
    CheapTalkReading,
    CounterpartGap,
    LabelPreference,
    TaggedDistance,
    TaggedReading,
    acquiescence_index,
    calibration_gaps,
    cheap_talk_readings,
    choice_response_distributions,
    choice_response_entropy,
    counterpart_gaps,
    forced_choice_prefix_win_rate,
    forced_choice_win_rates,
    modal_choices,
    numeric_item_readings,
    orientation_counts,
    parse_rate_by_family,
    per_item_scores,
    subscale_composites,
    svo_angle,
    svo_mean_completion_angle,
    tagged_distribution_distance,
    tagged_readings,
    total_variation_distance,
    wording_gap,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import pandas as pd

logger = logging.getLogger(__name__)

# The behaviour field per game; see `behaviour_field`.

# The game whose cooperative label pays nothing, so movement on it is a token-level tell.
NEGATIVE_CONTROL_GAME_ID = "defective-coordination"

# Stated in the rendered header rather than applied; below it a "rate" is a few completions.
NOISE_FLOOR_PARSED = 4

# A section this far from parsing is a measurement problem rather than a measurement.
PARSE_FAILURE_BANNER_THRESHOLD = 0.25

# A truncation rate this high means the generation cap is shaping the section's answers.
TRUNCATION_BANNER_THRESHOLD = 0.10

# What a flip table needs before it can compare anything.
MIN_CELLS_FOR_A_FLIP = 2

# What a contrast group needs before it is a contrast.
MIN_ARMS_FOR_A_CONTRAST = 2

EMPTY_CELL = "-"
ALL_VARIANTS = "all"
UNREGISTERED = "UNREGISTERED"

VARIANT_FIELD = "payoff_variant"
RESKIN_FIELD = "reskin_id"

# Meta fields a battery's cells must agree on to be comparable at all; see `_provenance_problems`.
PROVENANCE_META_FIELDS: tuple[str, ...] = ("git_sha", "sampling", "thinking", "backend_kind")

# The two mirrored gamble ladders, named by id; see `self_report_risk_mirror_rows` for why by hand.
RISK_LOSS_AVERSION_MIRROR_PAIR: tuple[str, str] = (
    "risk-sure-versus-spread-gain-only-mirror",
    "loss-mixed-versus-sure-mirror",
)

# Why the mirrored ladders have no gap this step, recorded in the cell rather than left blank.
RISK_MIRROR_LADDER_MISSING = "only one of the two mirrored ladders is in these records"
RISK_MIRROR_NOTHING_PARSED = "no answer scored on one ladder"
RISK_MIRROR_RUNG_COUNTS_DIFFER = "the two ladders do not share a rung count, so no gap has a unit"

# What a row of the values ordering is: one of the five registered poles, or one of the three
# non-social goods that were pooled into a pole. Named in the row rather than left to the label's
# spelling, because a reader ranking the components alongside the poles counts non-social three times.
VALUES_POLE_ROLE = "pole"
VALUES_COMPONENT_ROLE = f"component of {NON_SOCIAL_LABEL_PREFIX}*"

# Column labels the renderer reads back by name for its contrast tables, so a rename cannot strand it.
STEP_COLUMN = "step"
GAME_COLUMN = "game"
RATE_COLUMN = "rate (parsed/asked prompts)"
# The draw-level pair the prompt-level one hides once a prompt is sampled more than once.
DRAWS_COLUMN = "draws (parsed/asked)"
CONTROL_RATE_COLUMN = "coop-label rate (parsed/asked prompts)"
BASE_COLUMN = "base step 0 (parsed/asked prompts)"
DELTA_COLUMN = "delta"
EDT_COLUMN = "mean_edt_leaning"
PROSOCIAL_COLUMN = "prosocial_choice_rate"
ORDER_COLUMN = "order_disagreement_rate"
ACCURACY_ASKED_COLUMN = "accuracy over asked (correct/asked)"
ACCURACY_PARSED_COLUMN = "accuracy over parsed (correct/parsed)"

ROLE_TRAINED = "trained"
ROLE_TRANSFER = "transfer (another arm's game)"
ROLE_HELDOUT = "held-out (no arm ever)"
ROLE_NEGATIVE_CONTROL = "NEGATIVE CONTROL (held-out)"


@dataclass(frozen=True)
class Rate:
    """One rate and the denominators that make it readable.

    Two denominators, because a behavioural cell now has two units. `n_parsed`/`n_prompts` count
    PROMPTS -- the unit the mean is over, since a prompt sampled several times is one observation --
    and `n_records_parsed`/`n_records` count the draws behind them, which is what moves when a
    section stops parsing. They coincide exactly when every prompt was drawn once, so every rate
    measured before the battery could sample per prompt reads unchanged. `n_truncated` counts draws
    whose thinking never closed, kept separate because it is a reason to raise the generation cap
    rather than a behaviour.
    """

    value: float | None
    n_parsed: int
    n_prompts: int
    n_records_parsed: int
    n_records: int
    n_truncated: int

    @property
    def parse_failures(self) -> int:
        """Report how many draws were asked and produced nothing usable."""
        return self.n_records - self.n_records_parsed

    @property
    def parsed_over_asked(self) -> str:
        """Render the denominator pair the way every rate in this document carries it."""
        return f"{self.n_parsed}/{self.n_prompts}"

    @property
    def draws_parsed_over_asked(self) -> str:
        """Render the draw-level denominators, which the prompt-level pair hides once n > 1."""
        return f"{self.n_records_parsed}/{self.n_records}"

    @property
    def cell(self) -> str:
        """Render the rate with its denominator, as one table cell."""
        if self.value is None:
            return f"{EMPTY_CELL} (0/{self.n_prompts})"
        return f"{self.value:.3f} ({self.parsed_over_asked})"


@dataclass(frozen=True)
class SharedDraw:
    """One (step, section) where two or more of a contrast's arms carry byte-identical completions.

    A shared draw means those arms' rows for that section at that step are one measurement printed
    twice, so a contrast between them there is zero by construction rather than agreement.
    """

    step: int
    section: str
    arms: tuple[str, ...]
    n_records: int


@dataclass(frozen=True)
class FlipScope:
    """What the flip table could have compared, and what was set aside to get there.

    `n_comparable` is the flip table's real denominator: items settled to one answer at both
    endpoints, through the same `report.answers_by_probe` reduction the flips themselves come from.
    `n_asked_both` is the ceiling -- items asked at both endpoints -- and the gap between the two is
    the set-aside count: items unscored or answering differently under their two option orders.
    """

    n_asked_both: int
    n_comparable: int

    @property
    def n_set_aside(self) -> int:
        """Count the items asked at both endpoints that could not be compared."""
        return self.n_asked_both - self.n_comparable


@dataclass(frozen=True)
class ChurnFloor:
    """Per-item endorsement churn between neighbouring checkpoints: the flip table's noise floor.

    The first-against-last flip table is a two-cell comparison, which ground rule one forbids
    reading alone; this is the baseline it has to clear. Pooled over every consecutive pair of the
    arm's cells, same reduction, same denominator convention.
    """

    n_flips: int
    n_comparable: int
    n_step_pairs: int

    @property
    def rate(self) -> float | None:
        """Return the pooled adjacent-step flip rate, or None where nothing was comparable."""
        return self.n_flips / self.n_comparable if self.n_comparable else None


@dataclass(frozen=True)
class ContrastGroup:
    """Two or more arms that trained the same game, and what differs between them.

    Derived rather than listed: the flagship pair is the two arms training one game under different
    grading rules, and the payoff-rung arms are two arms training one game at different pinned
    payoffs. Both are contrasts worth putting side by side, neither is named in this file, so
    registering a third arm on an existing game produces its contrast without an edit.
    """

    game_id: str
    arms: tuple[str, ...]
    axis: str
    shared_draws: tuple[SharedDraw, ...] = ()


@dataclass(frozen=True)
class ArmReadout:
    """Every table for one arm, plus the facts a reader needs to interpret them."""

    arm: str
    steps: tuple[int, ...]
    missing_steps: tuple[int, ...]
    trained_game: str | None
    pinned_variants: tuple[str, ...]
    late_window: tuple[int, ...]
    facts: dict[str, Any]
    trained_ladder: list[dict[str, Any]]
    transfer_matrix: list[dict[str, Any]]
    transfer_windows: list[dict[str, Any]]
    negative_control: list[dict[str, Any]]
    dt_ladder: list[dict[str, Any]]
    dt_endorsements: list[dict[str, Any]]
    dt_flips: list[dict[str, Any]]
    flip_scope: FlipScope
    churn: ChurnFloor
    capability_ladder: list[dict[str, Any]]
    self_report_ladder: list[dict[str, Any]]
    self_report_controls: list[dict[str, Any]]
    self_report_health: list[dict[str, Any]]
    self_report_instruments: list[dict[str, Any]]
    self_report_control_distributions: list[dict[str, Any]]
    self_report_nominal_choices: list[dict[str, Any]]
    self_report_values_ordering: list[dict[str, Any]]
    self_report_numeric: list[dict[str, Any]]
    self_report_risk_mirror: list[dict[str, Any]]
    self_report_counterparts: list[dict[str, Any]]
    self_report_cheap_talk: list[dict[str, Any]]
    self_report_tagged: list[dict[str, Any]]
    self_report_calibration: list[dict[str, Any]]
    variant_splits: list[dict[str, Any]]
    reskin_splits: list[dict[str, Any]]
    position_splits: list[dict[str, Any]]
    per_round: list[dict[str, Any]]
    section_health: list[dict[str, Any]]
    notes: list[str]

    @property
    def incomplete(self) -> bool:
        """Report whether this arm is short of the ladder its sibling arms reached."""
        return bool(self.missing_steps)


@dataclass(frozen=True)
class BatteryReadout:
    """The whole document as data, so the markdown is a rendering and not the record."""

    battery_dir: Path
    generated_at: str
    ladder: tuple[int, ...]
    arms: list[ArmReadout]
    contrasts: list[ContrastGroup]
    excluded: list[ExcludedCell]
    problems: list[str]
    cell_count: int

    @property
    def incomplete_arms(self) -> list[str]:
        """Name the arms missing at least one step of the battery's ladder."""
        return [arm.arm for arm in self.arms if arm.incomplete]

    @property
    def error_containing(self) -> bool:
        """Report whether anything in this pass makes the document provisional."""
        return bool(self.incomplete_arms or self.excluded or self.problems)


def battery_ladder(cells: Mapping[str, Sequence[EvalTrace]]) -> tuple[int, ...]:
    """Return the checkpoint ladder this battery is evaluating, as the union of steps reached.

    Derived from the data rather than declared, so an arm is judged complete against what its
    siblings actually managed and the whole thing self-updates as cells land. The blind spot is
    honest and worth stating: a step no arm has reached yet cannot appear here, so a battery
    uniformly behind reads as complete. The per-arm cell counts are what show that.
    """
    return tuple(sorted({trace.step for traces in cells.values() for trace in traces}))


def late_window(steps: Sequence[int]) -> tuple[int, ...]:
    """Return the pooled late window: the upper half of the ladder, excluding the base-model cell.

    A window rather than the last step alone, because one cell is not a finding, and a fraction of
    the ladder rather than a fixed step list, so the same rule holds for an arm three cells in and
    for a finished one. Step 0 is excluded because it is the un-adapted model, which is the thing
    the window gets contrasted against.
    """
    if not steps:
        return ()
    top = max(steps)
    return tuple(step for step in sorted(steps) if step > 0 and step * 2 >= top)


def steps_label(steps: Sequence[int]) -> str:
    """Render the checkpoints pooled into one window."""
    return ",".join(str(step) for step in steps) if steps else "none"


def late_window_column(window: Sequence[int]) -> str:
    """Name the pooled-late-window column, so the two tables that render it cannot label it twice."""
    return f"late steps {steps_label(window)} (parsed/asked prompts)"


def rate_of(records: Sequence[Mapping[str, Any]], field_name: str) -> Rate:
    """Average one numeric field per PROMPT first, then over prompts, keeping both denominators.

    The prompt is the unit of observation, not the record: the battery draws
    `game_behavior_samples` completions per prompt, so pooling the draws would weight a prompt whose
    draws mostly failed to parse differently from its neighbours -- the same argument
    `per_item_means` makes for the probes, and the same reduction `games.evals._behaviour_rate`
    applies inside the trace's own summary, so the table and the summary beside it cannot disagree.
    The two averages are identical whenever every prompt contributed the same number of parsed
    draws, which is why a cell drawn once per prompt reads exactly as it did before.

    A prompt is keyed per CELL. `behaviour_records` stamps the step onto the key it hands over,
    because a pooled window spans several checkpoints and the same prompt id recurs in each: keyed on
    the id alone, one checkpoint's draws of a prompt would merge with another's into a single
    observation and the window would silently average steps before averaging prompts.

    The parsed count comes from the field being present rather than from the `parsed` flag, because
    the field is what actually enters the mean. Where the two disagree the record schema has moved
    under the reduction, which is worth a warning rather than a silently different denominator.
    """
    by_prompt: dict[str, list[float]] = {}
    asked: set[str] = set()
    for record in records:
        key = prompt_key(record)
        asked.add(key)
        value = record.get(field_name)
        if value is not None:
            by_prompt.setdefault(key, []).append(float(value))
    prompt_means = [sum(values) / len(values) for values in by_prompt.values()]
    n_records_parsed = sum(len(values) for values in by_prompt.values())
    flagged = sum(1 for record in records if record.get("parsed"))
    if flagged != n_records_parsed:
        logger.warning(
            f"{field_name}: {n_records_parsed} record(s) carry a value but {flagged} are flagged "
            f"parsed; the denominator printed is the one the mean was taken over"
        )
    return Rate(
        value=sum(prompt_means) / len(prompt_means) if prompt_means else None,
        n_parsed=len(prompt_means),
        n_prompts=len(asked),
        n_records_parsed=n_records_parsed,
        n_records=len(records),
        n_truncated=truncated_count(records),
    )


def truncated_count(records: Sequence[Mapping[str, Any]]) -> int:
    """Count the records whose thinking block never closed, which is its own failure state."""
    return sum(1 for record in records if record.get("truncated_thinking"))


def parsed_count(records: Sequence[Mapping[str, Any]]) -> int:
    """Count the records that produced a usable answer, by their own parsed flag."""
    return sum(1 for record in records if record.get("parsed"))


# Games whose behaviour is not a cooperation rate, keyed to the field their records carry instead.
def behaviour_field(game_id: str) -> str:
    """Return which record field carries this game's behaviour rate.

    Not every game has a cooperation rate. The unilateral split has no counterpart to cooperate
    with, so its rate is the fraction of the endowment kept and calling that a cooperation rate would
    invert its direction. The simultaneous claim has a counterpart and still no cooperative action:
    its rate is the fraction of the total claimed, where higher is more grasping and the informative
    comparison is against a half. The trust games answer with an amount, so theirs is the fraction of
    the stock sent -- except the never-trained trustee item, which answers with a share returned.

    There is no fallback. `games.evals.BEHAVIOUR_FIELD_BY_GAME` is checked exhaustive against the
    prompt registry at import, because defaulting an unlisted game to the cooperation rate rendered
    an EMPTY table rather than an error, and both readouts recompute themselves from artifacts, so
    the emptiness travelled quietly into whatever was read next.
    """
    return BEHAVIOUR_FIELD_BY_GAME[game_id]


# Where `behaviour_records` writes each record's per-cell prompt identity; see `prompt_key`.
PROMPT_KEY_FIELD = "cell_prompt_key"


def prompt_key(record: Mapping[str, Any]) -> str:
    """Return the key identifying which prompt in which cell this record is a draw of.

    Read from the field `behaviour_records` stamps rather than recomputed, so every reduction over
    behaviour records groups on one definition. Falls back to nothing: a record reaching a rate
    without having been through `behaviour_records` raises here rather than pooling every draw of
    every step under one key, which is the shape the per-prompt reduction exists to prevent.
    """
    return str(record[PROMPT_KEY_FIELD])


def behaviour_records(
    traces: Sequence[EvalTrace],
    game_id: str,
    *,
    where: Mapping[str, str] | None = None,
    section: str = SECTION_GAME_BEHAVIOR,
) -> list[Mapping[str, Any]]:
    """Return one game's behaviour records across these cells, optionally one label split only.

    `section` names which section's records to read; the game-behavior section by default, and the
    framing sweep for the one table that reads both (`position_rows`). The framing sweep's records
    are game records with a `counterpart_framing` stamped on, so the same stamping and the same
    `where` filter serve it.

    Each record comes back carrying `PROMPT_KEY_FIELD`: its prompt id scoped to the cell it was
    drawn in, which is what lets `rate_of` average within a prompt before averaging over prompts
    even when the caller pooled a window of several checkpoints. Stamped here, on a copy, for the
    reason `_probe_records` rewrites `probe_id` the same way -- the record knows its prompt and only
    the trace knows which cell it came from.

    The key carries the ARM as well as the step. Every caller today hands over one arm's traces, so
    the arm is currently redundant; leaving it out would make the key per (prompt, step) while the
    docstring called it per cell, and the first caller to pool two arms at one step -- a contrast
    table is the obvious candidate -- would silently merge their draws of each prompt into a single
    observation. Cheap insurance, and it makes the name honest.
    """
    selected: list[Mapping[str, Any]] = []
    for trace in traces:
        for record in trace.section(section):
            if str(record["game_id"]) != game_id:
                continue
            if where is not None and any(
                str(record.get(field_name)) != value for field_name, value in where.items()
            ):
                continue
            selected.append(
                {
                    **record,
                    PROMPT_KEY_FIELD: (f"{trace.arm}@step{trace.step}::{record['prompt_id']}"),
                }
            )
    return selected


def game_rate(
    traces: Sequence[EvalTrace], game_id: str, *, where: Mapping[str, str] | None = None
) -> Rate:
    """Return one game's behaviour rate pooled over these cells: cooperation, or keep-fraction."""
    return rate_of(behaviour_records(traces, game_id, where=where), behaviour_field(game_id))


def games_present(
    traces: Sequence[EvalTrace], *, section: str = SECTION_GAME_BEHAVIOR
) -> list[str]:
    """Return every game these cells carry records for, in the game-behavior section by default."""
    return sorted({str(record["game_id"]) for trace in traces for record in trace.section(section)})


def values_present(
    traces: Sequence[EvalTrace],
    game_id: str,
    field_name: str,
    *,
    section: str = SECTION_GAME_BEHAVIOR,
    where: Mapping[str, str] | None = None,
) -> list[str]:
    """Return the distinct values of one label field for one game across these cells."""
    return sorted(
        {
            str(record[field_name])
            for record in behaviour_records(traces, game_id, where=where, section=section)
        }
    )


def trained_game_of(arm: str) -> str | None:
    """Return the one game this arm trains, or None when the arm is not registered.

    None rather than a guess: `--arm` accepts any label and a hosted model can be evaluated under
    any name, so an unregistered arm genuinely cannot say which of its columns is the trained one.
    """
    registered = ARMS.get(arm)
    return None if registered is None else registered.game_id


def pinned_variants_of(arm: str) -> tuple[str, ...]:
    """Return the payoff variants this arm's training was pinned to, empty meaning all of them."""
    registered = ARMS.get(arm)
    return () if registered is None else registered.payoff_variants


def grading_of(arm: str) -> str | None:
    """Return the grading rule this arm trained under, which is the wave's independent variable."""
    registered = ARMS.get(arm)
    return None if registered is None else registered.grading


def game_role(arm: str, game_id: str) -> str:
    """Classify one game as this arm's trained game, in-registry transfer, or a held-out game."""
    if game_id == trained_game_of(arm):
        return ROLE_TRAINED
    if game_id == NEGATIVE_CONTROL_GAME_ID:
        return ROLE_NEGATIVE_CONTROL
    if game_id in EVAL_ONLY_GAME_IDS:
        return ROLE_HELDOUT
    return ROLE_TRANSFER


def variant_role(variant: str | None, pinned: Sequence[str]) -> str:
    """Say whether a payoff variant is the one an arm trained, or within-game transfer."""
    if not pinned or variant is None:
        return "trained (arm pins no variant)"
    return "trained variant" if variant in pinned else "transfer variant (same game)"


def trained_ladder_rows(arm: str, traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the per-step rate on this arm's trained game, split by variant where it pins one.

    An arm pinned to one payoff rung trains one rung of a risk-dominance ladder, so its own rung is
    the trained readout and the others are within-game dose-response transfer: the same game at
    payoffs the arm never saw. Splitting them is the difference between reading a dose response and
    averaging it away.
    """
    game_id = trained_game_of(arm)
    if game_id is None:
        return []
    pinned = pinned_variants_of(arm)
    variants: list[str | None] = (
        list(values_present(traces, game_id, VARIANT_FIELD)) if pinned else [None]
    )
    rows: list[dict[str, Any]] = []
    for trace in traces:
        for variant in variants:
            rate = game_rate(
                [trace], game_id, where=None if variant is None else {VARIANT_FIELD: variant}
            )
            rows.append(
                {
                    STEP_COLUMN: trace.step,
                    "game": game_id,
                    VARIANT_FIELD: variant or ALL_VARIANTS,
                    "role": variant_role(variant, pinned),
                    "measure": behaviour_field(game_id),
                    RATE_COLUMN: rate.cell,
                    DRAWS_COLUMN: rate.draws_parsed_over_asked,
                    "parse_failures": rate.parse_failures,
                    "n_truncated": rate.n_truncated,
                }
            )
    return rows


def transfer_matrix_rows(arm: str, traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return a game-by-step grid of behaviour rates, one row per game evaluated.

    Wide rather than long because this is the table a reader scans for a trend: the never-trained
    rows are the transfer readout, and the row's role says which those are. Every cell carries its
    own denominator, so a row whose n collapses at one step cannot read as a behaviour move.
    """
    rows: list[dict[str, Any]] = []
    for game_id in games_present(traces):
        row: dict[str, Any] = {
            GAME_COLUMN: game_id,
            "role": game_role(arm, game_id),
            "measure": behaviour_field(game_id),
        }
        for trace in traces:
            row[f"step {trace.step}"] = game_rate([trace], game_id).cell
        rows.append(row)
    return sorted(rows, key=lambda row: (str(row["role"]) != ROLE_TRAINED, str(row[GAME_COLUMN])))


def transfer_window_rows(
    arm: str, traces: Sequence[EvalTrace], *, window: Sequence[int]
) -> list[dict[str, Any]]:
    """Return every game's base cell against the pooled late window, with the move between them.

    The grid beside it reads the shape of each curve; this reads the size of the move, which a
    reader otherwise has to eyeball across eight columns. The base side is step 0 alone because
    there is only one base cell, so the move column carries that cell's noise -- read it against the
    per-step row above rather than on its own.
    """
    late = [trace for trace in traces if trace.step in window]
    base = [trace for trace in traces if trace.step == 0]
    rows: list[dict[str, Any]] = []
    for game_id in games_present(traces):
        base_rate = game_rate(base, game_id)
        late_rate = game_rate(late, game_id)
        rows.append(
            {
                GAME_COLUMN: game_id,
                "role": game_role(arm, game_id),
                "measure": behaviour_field(game_id),
                BASE_COLUMN: base_rate.cell,
                late_window_column(window): late_rate.cell,
                DELTA_COLUMN: _delta_cell(base_rate, late_rate),
            }
        )
    return sorted(rows, key=lambda row: (str(row["role"]) != ROLE_TRAINED, str(row[GAME_COLUMN])))


def _delta_cell(base: Rate, late: Rate) -> str:
    """Render the move between two pooled rates, or a dash where either side measured nothing."""
    if base.value is None or late.value is None:
        return EMPTY_CELL
    return f"{late.value - base.value:+.3f}"


def negative_control_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the negative control's own ladder, with truncation and parse counts beside it."""
    rows: list[dict[str, Any]] = []
    for trace in traces:
        rate = game_rate([trace], NEGATIVE_CONTROL_GAME_ID)
        if rate.n_prompts == 0:
            continue
        rows.append(
            {
                STEP_COLUMN: trace.step,
                CONTROL_RATE_COLUMN: rate.cell,
                DRAWS_COLUMN: rate.draws_parsed_over_asked,
                "parse_failures": rate.parse_failures,
                "n_truncated": rate.n_truncated,
            }
        )
    return rows


def _probe_records(traces: Sequence[EvalTrace], *, step_scoped: bool) -> list[Mapping[str, Any]]:
    """Return every probe record across these cells, optionally keyed per checkpoint.

    `step_scoped` rewrites each probe id to carry its step, which `order_disagreement_rate` needs
    and nothing else does: it keys a counterbalanced pair on (probe_id, sample_index), so pooling
    two checkpoints under a shared id would have one checkpoint's answer overwrite the other's.
    Per-item means keep the shared id on purpose, so an item's samples average into one observation.
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
    """Count each theory label across the open-ended records that named one.

    Every label in `games.parsing.THEORY_LABELS` gets a column whether or not it appeared, so the
    table keeps one shape across arms and a zero is visibly a zero rather than a missing column. A
    label outside that vocabulary raises: the parser and this table would have drifted apart, and
    silently growing a column would leave the table ragged across steps.
    """
    counts = dict.fromkeys(THEORY_LABELS, 0)
    for record in records:
        theory = record.get("theory")
        if theory is None:
            continue
        label = str(theory)
        if label not in counts:
            raise ValueError(
                f"open-ended theory label {label!r} is not in games.parsing.THEORY_LABELS "
                f"{THEORY_LABELS}; the parser vocabulary and this table have drifted apart"
            )
        counts[label] += 1
    return counts


def dt_endorsement_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Count each endorsement pattern over the multiple-choice renders, per step.

    The reduction `mean_edt_leaning` cannot carry by construction: an FDT-endorsed answer and one
    every theory endorses both read as leaning zero there, and they separate here. A pattern joins
    the theories compatible with the chosen option with `+`; `neither` means the choice matched
    none of the three (the same encoding the flip table's before/after values use), and `unparsed`
    is a real category rather than a dropped row. The column set is the union over this arm's
    steps, zero-filled, so one arm's table keeps one shape.
    """
    per_step: list[tuple[int, dict[str, int], int, int]] = []
    patterns: set[str] = set()
    for trace in traces:
        counts: dict[str, int] = {}
        n_unparsed = 0
        n_asked = 0
        for record in trace.section(SECTION_DT_PROBES):
            if record.get("kind") != PROBE_MULTIPLE_CHOICE:
                continue
            n_asked += 1
            theories = record.get("compatible_theories")
            if theories is None:
                n_unparsed += 1
                continue
            pattern = "+".join(str(theory) for theory in theories) or "neither"
            counts[pattern] = counts.get(pattern, 0) + 1
        patterns.update(counts)
        per_step.append((trace.step, counts, n_unparsed, n_asked))
    if not any(n_asked for _, _, _, n_asked in per_step):
        return []
    ordered = sorted(patterns)
    return [
        {
            STEP_COLUMN: step,
            **{pattern: counts.get(pattern, 0) for pattern in ordered},
            "unparsed": n_unparsed,
            "renders asked": n_asked,
        }
        for step, counts, n_unparsed, n_asked in per_step
    ]


def _comparable_order_pairs(records: Sequence[Mapping[str, Any]]) -> int:
    """Count the (item, sample) pairs answered under both option orders.

    The denominator behind `order_disagreement_rate`, which returns the rate alone. Keyed exactly as
    that reduction keys it, and cross-checked against it by the caller: where this finds no
    comparable pair the rate must be None, and a disagreement means the two have drifted apart.
    """
    by_pair: dict[tuple[str, int], set[str]] = {}
    for record in records:
        if record.get("answer_index") is None or record.get("option_order_name") is None:
            continue
        key = (str(record["probe_id"]), int(record["sample_index"]))
        by_pair.setdefault(key, set()).add(str(record["option_order_name"]))
    return sum(1 for orders in by_pair.values() if len(orders) > 1)


def _order_pairs_asked(records: Sequence[Mapping[str, Any]]) -> int:
    """Count the (item, sample) pairs asked under both option orders, parsed or not.

    The asked side of the order-disagreement denominator. `_comparable_order_pairs` counts the
    pairs whose two renders both parsed, which floats with the parse rate; this counts what was
    asked, so a denominator move is visible as a gap between the two rather than as a rate move.
    """
    by_pair: dict[tuple[str, int], set[str]] = {}
    for record in records:
        if record.get("kind") != PROBE_MULTIPLE_CHOICE:
            continue
        key = (str(record["probe_id"]), int(record["sample_index"]))
        by_pair.setdefault(key, set()).add(str(record["option_order_name"]))
    return sum(1 for orders in by_pair.values() if len(orders) > 1)


def format_rate(value: float | None, n_used: int, n_asked: int, *, unit: str) -> str:
    """Render a per-item scalar with both of its denominators: items used over items asked.

    `n_used` counts the items that actually entered the mean and `n_asked` the items the cell was
    asked; both are printed because the used side floats with parse failures, and a denominator
    move that changes the mean would otherwise read as the mean moving.
    """
    rendered = EMPTY_CELL if value is None else f"{value:.3f}"
    return f"{rendered} ({n_used}/{n_asked} {unit})"


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Average values, or None where there are none, so an empty mean is never manufactured."""
    return sum(values) / len(values) if values else None


def dt_ladder_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the decision-theory ladder: theory counts, EDT leaning, prosocial choice, order.

    Four instruments in one table on purpose. The open-ended counts are what a theory shift looks
    like item by item; `mean_edt_leaning` is the scalar, which cannot separate FDT from ambiguous
    and is shrunk toward zero by items where CDT and EDT endorse the same option;
    `prosocial_choice_rate` is the control separating a theory shift from a policy that merely
    became agreeable; and `order_disagreement_rate` is what stops a letter bias from reading as a
    position at all.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        pooled = _probe_records([trace], step_scoped=False)
        open_ended = [record for record in pooled if record.get("kind") == PROBE_OPEN_ENDED]
        choice = [record for record in pooled if record.get("kind") == PROBE_MULTIPLE_CHOICE]
        counts = _theory_counts(open_ended)
        leanings = sorted(per_item_means(pooled, "edt_leaning").values())
        prosocial = sorted(per_item_means(pooled, "chose_prosocial").values())
        choice_items_asked = len({str(record["probe_id"]) for record in choice})
        # valence lives on parsed renders only: a never-parsed item drops from BOTH sides here
        valenced_items_asked = len(
            {
                str(record["probe_id"])
                for record in choice
                if record.get("prosocial_option") is not None
            }
        )
        scoped = _probe_records([trace], step_scoped=True)
        disagreement = order_disagreement_rate(scoped)
        pairs = _comparable_order_pairs(scoped)
        pairs_asked = _order_pairs_asked(scoped)
        if (pairs == 0) != (disagreement is None):
            logger.warning(
                f"step {trace.step}: order-disagreement rate is {disagreement} over {pairs} "
                f"comparable pair(s); the rate and its denominator have drifted apart"
            )
        rows.append(
            {
                STEP_COLUMN: trace.step,
                **{f"n_{label}": count for label, count in counts.items()},
                "open-ended (parsed/asked)": f"{sum(counts.values())}/{len(open_ended)}",
                EDT_COLUMN: format_rate(
                    _mean_or_none(leanings), len(leanings), choice_items_asked, unit="items"
                ),
                PROSOCIAL_COLUMN: format_rate(
                    _mean_or_none(prosocial),
                    len(prosocial),
                    valenced_items_asked,
                    unit="valenced items",
                ),
                ORDER_COLUMN: format_rate(disagreement, pairs, pairs_asked, unit="pairs"),
                "truncated/asked": f"{truncated_count(pooled)}/{len(pooled)}",
            }
        )
    return rows


def dt_flip_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the per-item endorsement flips between this arm's first and last landed cell.

    `report.theory_item_flips` compares an arm's first cell to its last, which on an arm still
    landing is not step 0 against the end of training. The endpoints are columns of every row and
    the rendered header names them, because a flip table read as a finished comparison when it is a
    first-two-cells comparison is the misreading that costs a re-run.
    """
    if len(traces) < MIN_CELLS_FOR_A_FLIP:
        return []
    return rows_from_frame(theory_item_flips(traces))


def flip_scope(traces: Sequence[EvalTrace]) -> FlipScope:
    """Return the flip table's denominator: items comparable at both endpoints, and the ceiling.

    Comparable means settled to one answer at each endpoint, through `report.answers_by_probe` --
    the same reduction the flips themselves come from, so the denominator counts exactly the items a
    flip could have been drawn from. The ceiling (items asked at both endpoints) is kept beside it
    because the gap between the two is the set-aside count: items unscored, or answering differently
    under their two option orders, at one endpoint or the other. An empty flip table over a scope of
    zero and one over a scope of twenty are different findings.
    """
    if len(traces) < MIN_CELLS_FOR_A_FLIP:
        return FlipScope(n_asked_both=0, n_comparable=0)
    first, last = traces[0], traces[-1]
    asked = [
        {str(record["probe_id"]) for record in trace.section(SECTION_DT_PROBES)}
        for trace in (first, last)
    ]
    settled_first, _ = answers_by_probe(first.section(SECTION_DT_PROBES))
    settled_last, _ = answers_by_probe(last.section(SECTION_DT_PROBES))
    return FlipScope(
        n_asked_both=len(asked[0] & asked[1]),
        n_comparable=len(set(settled_first) & set(settled_last)),
    )


def adjacent_churn(traces: Sequence[EvalTrace]) -> ChurnFloor:
    """Pool per-item endorsement changes over every consecutive pair of this arm's cells.

    The noise floor the first-against-last flip table must clear: neighbouring checkpoints differ by
    ten training steps, so their per-item churn is mostly sampling, and a 0-to-last flip rate at or
    below it is not signal. Same reduction and same comparability rule as the flip table itself
    (`report.answers_by_probe`, items settled at both cells of a pair), so the two rates share a
    convention and can be read against each other.
    """
    n_flips = 0
    n_comparable = 0
    n_step_pairs = 0
    for earlier, later in itertools.pairwise(traces):
        settled_earlier, _ = answers_by_probe(earlier.section(SECTION_DT_PROBES))
        settled_later, _ = answers_by_probe(later.section(SECTION_DT_PROBES))
        common = set(settled_earlier) & set(settled_later)
        n_flips += sum(
            1 for probe_id in common if settled_earlier[probe_id] != settled_later[probe_id]
        )
        n_comparable += len(common)
        n_step_pairs += 1
    return ChurnFloor(n_flips=n_flips, n_comparable=n_comparable, n_step_pairs=n_step_pairs)


def capability_ladder_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the arithmetic canary per step, under both defensible denominators.

    Two accuracies, because the two disagree and each is defensible: over every item asked, an
    unparseable answer counts as wrong; over the parsed items, it counts as nothing. Reporting one
    alone is how a table comes to disagree with the summary beside it, so both are here with the
    counts that produced them.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        records = trace.section(SECTION_CAPABILITIES)
        if not records:
            continue
        n_correct = sum(1 for record in records if record.get("correct"))
        n_parsed = parsed_count(records)
        rows.append(
            {
                STEP_COLUMN: trace.step,
                ACCURACY_ASKED_COLUMN: (
                    f"{n_correct / len(records):.3f} ({n_correct}/{len(records)})"
                ),
                ACCURACY_PARSED_COLUMN: (
                    f"{n_correct / n_parsed:.3f} ({n_correct}/{n_parsed})"
                    if n_parsed
                    else f"{EMPTY_CELL} (0/0)"
                ),
                "n_truncated": truncated_count(records),
            }
        )
    return rows


def self_report_ladder_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return one row per step, instrument and subscale: the composite with its item denominators.

    Long rather than wide on purpose. Seven published instruments plus the authored families carry
    twenty-odd subscales between them, and a column per subscale produces a table nothing can read; a
    row per subscale stays fixed-width however many instruments a cell administered.

    The composite comes from `games.survey.subscale_composites`, the same reduction the trace's own
    summary used, so this table and that summary cannot disagree. Only the denominators are counted
    here, and per ITEM rather than per render: every option-bearing item is asked under two orders and
    sampled several times, so a record count would overstate what the composite rests on by an order
    of magnitude.

    Every column is about ONE wording, the as-published one, passed explicitly rather than left to
    `subscale_composites`' default so the value and its denominators cannot drift apart: counting the
    neutral twins alongside a composite that excludes them printed an item base exactly twice the real
    one wherever a family twins every item. The twins are read in `self_report_control_rows`' wording
    gap, and the wording-blind truncation figure in `self_report_health_rows`, so scoping here drops
    nothing.

    Composites are never comparable ACROSS instruments -- five Likert points against six, against
    points handed to another party on the two allocation instruments -- so the instrument column is
    part of the reading rather than a label. What is worth looking at is the delta down one
    instrument's column.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        records = trace.section(SECTION_SELF_REPORT)
        if not records:
            continue
        wording = WORDING_AS_PUBLISHED
        composites = subscale_composites(records, wording=wording)
        for (instrument, subscale), value in sorted(composites.items()):
            in_scope = [
                record
                for record in records
                if str(record.get("wording")) == wording
                and str(record.get("instrument")) == instrument
                and str(record.get("subscale")) == subscale
            ]
            asked = {str(record["item_id"]) for record in in_scope}
            scored = {
                str(record["item_id"]) for record in in_scope if record.get("score") is not None
            }
            points = sorted({int(record["scale_points"]) for record in in_scope})
            rows.append(
                {
                    STEP_COLUMN: trace.step,
                    "instrument": instrument,
                    "subscale": subscale,
                    "family": _single_value(in_scope, "family"),
                    "kind": _single_value(in_scope, "kind"),
                    "composite": format_rate(value, len(scored), len(asked), unit="items"),
                    # What the composite is in units OF: a Likert scale's point count, an allocation
                    # item's option count. Beside `kind`, which says which of the two it is.
                    "answers offered": "/".join(str(point) for point in points),
                    "n_reverse_keyed": len(
                        {
                            str(record["item_id"])
                            for record in in_scope
                            if record.get("reverse_keyed")
                        }
                    ),
                    "truncated/asked": f"{truncated_count(in_scope)}/{len(in_scope)}",
                }
            )
    return rows


def _single_value(records: Sequence[Mapping[str, Any]], field_name: str) -> str:
    """Render the one value a field takes across these records, or name every value it took.

    A subscale's records should all carry one family and one kind. Printing whatever set is actually
    there, rather than the first record's value, means a grouping mistake shows up in the table as two
    values in a cell instead of hiding behind a plausible label.
    """
    values = sorted({str(record.get(field_name)) for record in records})
    return values[0] if len(values) == 1 else "+".join(values)


def self_report_control_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the per-subscale controls that say whether the composites above are readable.

    Two of them, and each is a way a composite moves for a reason that is not a disposition.
    `acquiescence_index` separates yea-saying from agreement, and prints its own recorded reason where
    a subscale has only one keying direction rather than manufacturing a number -- which is most
    subscales here, because only the competitiveness index carries both keyings inside one subscale.
    `wording_gap` separates a disposition shift from a response to the instrument's vocabulary, which
    in this project's own step-0 measurements was the larger of the two presentation effects, bigger
    than the letter position the counterbalanced orders already control; it reads `-` where the local
    item file supplied no neutral twins.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        records = trace.section(SECTION_SELF_REPORT)
        if not records:
            continue
        acquiescence = acquiescence_index(records)
        wording = wording_gap(records)
        for instrument, subscale in sorted(set(acquiescence) | set(wording)):
            reading = acquiescence.get((instrument, subscale))
            gap = wording.get((instrument, subscale))
            rows.append(
                {
                    STEP_COLUMN: trace.step,
                    "instrument": instrument,
                    "subscale": subscale,
                    "acquiescence": _acquiescence_cell(reading),
                    "keyed positive/reverse": (
                        EMPTY_CELL
                        if reading is None
                        else f"{reading.n_positive_keyed_items}/{reading.n_reverse_keyed_items}"
                    ),
                    "published wording": _optional_cell(None if gap is None else gap.as_published),
                    "neutral twin": _optional_cell(None if gap is None else gap.neutral_twin),
                    "wording gap": _optional_cell(None if gap is None else gap.gap, signed=True),
                    "n_twinned": EMPTY_CELL if gap is None else str(gap.n_twinned_items),
                }
            )
    return rows


def _acquiescence_cell(reading: Acquiescence | None) -> str:
    """Render an acquiescence reading, or the recorded reason it has none."""
    if reading is None:
        return EMPTY_CELL
    if reading.index is None:
        return f"{EMPTY_CELL} ({reading.reason})"
    return f"{reading.index:+.3f}"


def _optional_cell(value: float | None, *, signed: bool = False) -> str:
    """Render a float that may be absent, never as a zero."""
    if value is None:
        return EMPTY_CELL
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def self_report_health_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the per-family parse rates and order-disagreement rate, per step.

    Per family rather than per section, because the families answer in three different formats -- a
    lettered option, a bounded integer in a tag, one word from a closed list -- and a section-wide
    rate would average a format that stopped landing into two that still work.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        records = trace.section(SECTION_SELF_REPORT)
        if not records:
            continue
        for family, rate in sorted(parse_rate_by_family(records).items()):
            in_scope = [record for record in records if str(record.get("family")) == family]
            rows.append(
                {
                    STEP_COLUMN: trace.step,
                    "family": family,
                    "parsed/asked": rate.cell,
                    "n_items": len({str(record["item_id"]) for record in in_scope}),
                    ORDER_COLUMN: _self_report_order_cell(in_scope),
                    "truncated/asked": f"{truncated_count(in_scope)}/{len(in_scope)}",
                }
            )
    return rows


def self_report_instrument_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the readouts defined for one instrument only, per step.

    Each is `-` at a step whose cell did not administer that instrument, rather than absent, so the
    ladder's rows stay aligned when a later cell adds an instrument the earlier ones lacked.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        records = trace.section(SECTION_SELF_REPORT)
        if not records:
            continue
        instruments = {str(record.get("instrument")) for record in records}
        orientations = ", ".join(
            f"{label} {count}" for label, count in orientation_counts(records).items()
        )
        modal = modal_choices(records)
        rows.append(
            {
                STEP_COLUMN: trace.step,
                "svo_angle_deg": (
                    _optional_cell(svo_angle(records), signed=True)
                    if INSTRUMENT_SVO_SLIDER in instruments
                    else EMPTY_CELL
                ),
                # The other honest aggregate: one angle per completion, averaged. Beside the
                # aggregate-mean angle because arctan is nonlinear, so the two differ whenever
                # completions disagree, and conflating them is a silent definition change.
                "mean_completion_angle_deg": (
                    _optional_cell(svo_mean_completion_angle(records), signed=True)
                    if INSTRUMENT_SVO_SLIDER in instruments
                    else EMPTY_CELL
                ),
                # Whatever the records themselves carry an orientation for, not whichever instrument
                # name happens to be present: an authored allocation triple that separates the three
                # orientations is read the same way the published one is, and gating on the name
                # printed an empty cell beside records that had the reading in them.
                "orientations chosen": orientations or EMPTY_CELL,
                "nominal-choice modes": (
                    ", ".join(f"{item_id}={index}" for item_id, index in sorted(modal.items()))
                    or EMPTY_CELL
                ),
            }
        )
    return rows


def self_report_calibration_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the self-prediction gap per step and item: what it says it does minus what it did.

    The one table in this section that scores the artifact rather than the report, and the answer to
    the over-reporting caveat that qualifies everything else here. Both sides come from the SAME cell,
    so the comparison is within one checkpoint under one sampler rather than across passes. A missing
    measured side is a coverage fact about the run -- the behaviour section was not requested, or was
    restricted with `--games` -- rather than a reason to drop the prediction.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        survey_records = trace.section(SECTION_SELF_REPORT)
        if not any(
            str(record.get("family")) == FAMILY_SELF_PREDICTION for record in survey_records
        ):
            continue
        gaps = calibration_gaps(survey_records, trace.section(SECTION_GAME_BEHAVIOR))
        rows.extend(
            {
                STEP_COLUMN: trace.step,
                GAME_COLUMN: reading.game_id,
                "item": item_id,
                "predicted": _optional_cell(reading.predicted),
                "measured": _optional_cell(reading.measured),
                "gap (predicted - measured)": _optional_cell(reading.gap, signed=True),
                "predictions/behaviour records": (
                    f"{reading.n_predictions}/{reading.n_measured_records}"
                ),
            }
            for item_id, reading in sorted(gaps.items())
        )
    return rows


def self_report_control_distribution_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return one row per step and NEGATIVE-CONTROL item: its distribution, headroom and movement.

    The scoring the negative controls actually use. Nominal options have no ordering, so movement
    is the total variation distance of the item's response distribution from the FIRST trace's
    (step 0), per item -- never a shift in a mean over arbitrary option numbers. Entropy (bits) is
    the item's available headroom: a control answered identically in nearly every sample cannot
    detect drift, and control selection is frozen on base-model data using exactly this column.
    The subscale column keeps the factual-integrity items (`inert-fact`) readable as their own
    rows, never inside a preference-placebo aggregate.

    Scoped to the control family, which the mechanism it shares with `self_report_nominal_choice_rows`
    does not do by itself: every nominal-choice item in the battery has this reading computed for it,
    and two target families answer in that shape as well. Under the control heading their rows read as
    placebos, which inverts what they measure -- a values item moving is the family's registered
    RESULT, and a control moving is a threat to every other reading in the document.
    """
    return _nominal_choice_distribution_rows(
        traces, families=lambda family: family == FAMILY_NEGATIVE_CONTROL, name_family=False
    )


def self_report_nominal_choice_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the same per-item distribution reading for every nominal-choice family but the controls.

    Complement rather than a list of target families on purpose: a family that lands later in this
    answer shape appears here without anyone remembering to add it, where a named list would drop it
    out of the document silently -- which is the failure the whole-table gate exists to catch one
    level up. The family column is what keeps the rows readable once more than one of them is here,
    since the reading each family takes from a distribution differs: the graded-dimension items are
    read as an attribution and against their own tag mirrors, and the values items are read through
    `self_report_values_ordering_rows` with these rows as the per-item detail behind that ordering.
    """
    return _nominal_choice_distribution_rows(
        traces, families=lambda family: family != FAMILY_NEGATIVE_CONTROL, name_family=True
    )


def _nominal_choice_distribution_rows(
    traces: Sequence[EvalTrace], *, families: Callable[[str], bool], name_family: bool
) -> list[dict[str, Any]]:
    """Build per-item nominal-choice distribution rows for the families `families` accepts.

    The baseline is drawn from the same filtered records, so an item's movement is measured against
    its own step-0 distribution and a family absent from the base cell simply has no baseline.
    """
    with_records = [trace for trace in traces if trace.section(SECTION_SELF_REPORT)]
    if not with_records:
        return []
    baseline = choice_response_distributions(
        _records_of_families(with_records[0].section(SECTION_SELF_REPORT), families)
    )
    rows: list[dict[str, Any]] = []
    for trace in with_records:
        records = _records_of_families(trace.section(SECTION_SELF_REPORT), families)
        distributions = choice_response_distributions(records)
        entropy = choice_response_entropy(records)
        menus = _agreed_option_labels(records)
        labelled = {
            str(record["item_id"]): (str(record.get("family")), str(record.get("subscale")))
            for record in records
            if str(record.get("kind")) != SURVEY_NUMERIC
        }
        for item_id, distribution in sorted(distributions.items()):
            base = baseline.get(item_id)
            family, subscale = labelled.get(item_id, (EMPTY_CELL, EMPTY_CELL))
            row: dict[str, Any] = {STEP_COLUMN: trace.step, "item": item_id}
            if name_family:
                row["family"] = family
            rows.append(
                row
                | {
                    "subscale": subscale,
                    "distribution": _choice_distribution_cell(distribution, menus.get(item_id, ())),
                    "entropy (bits)": f"{entropy[item_id]:.2f}",
                    "TV vs step0": (
                        EMPTY_CELL
                        if base is None
                        else f"{total_variation_distance(distribution, base):.3f}"
                    ),
                }
            )
    return rows


def _records_of_families(
    records: Sequence[Mapping[str, Any]], families: Callable[[str], bool]
) -> list[Mapping[str, Any]]:
    """Keep the records whose family the predicate accepts."""
    return [record for record in records if families(str(record.get("family")))]


def _agreed_option_labels(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Return each item's option labels where all its records in this cell agree, and nothing else.

    A distribution over option NUMBERS is barely readable on the labelled families -- the values items
    vary their label order item by item on purpose, so option 0 is a different pole on each of a pair's
    two items -- and the labels are what the trace carries to make it readable. Where one cell's
    records disagree about an item's labels they are dropped rather than picked between, for the same
    reason a forced-tag item whose recorded menu width disagrees with itself reports no width: labels
    that moved inside one cell cannot name a share.
    """
    seen: dict[str, set[tuple[str, ...]]] = {}
    for record in records:
        labels = record.get("option_labels")
        if not labels:
            continue
        seen.setdefault(str(record["item_id"]), set()).add(tuple(str(label) for label in labels))
    return {item_id: next(iter(menus)) for item_id, menus in seen.items() if len(menus) == 1}


def _choice_distribution_cell(distribution: Mapping[int, float], labels: tuple[str, ...]) -> str:
    """Render one item's option distribution, keyed by label where the item carries labels.

    Canonical option order either way, so a labelled ladder reads down its own rungs rather than
    alphabetically.
    """
    return ", ".join(
        f"{labels[index] if labels else index}:{share:.2f}"
        for index, share in sorted(distribution.items())
    )


def self_report_values_ordering_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return one row per step and value pole: its win rate over the items that offered it.

    The values family's registered deliverable, and the reading `forced_choice_win_rates` was written
    for while nothing called it. A forced choice between two poles has no acquiescence to absorb and
    no scale to drift, so what moves is which category wins; every item offers exactly two poles, so
    every label's chance baseline is 0.5 and a rate is read against that rather than against zero.

    Filtered to the values family before anything is ordered, because the reduction pools by label
    STRING and two other families label their options -- the graded-dimension menu and the risk
    ladders' rungs. Unfiltered it returns one dictionary holding `joint-gain` and `reliance-full` and
    nothing about the resulting table says it is not an ordering.

    The registered ordering is over FIVE poles, so the three `non-social-` labels contribute one row
    recomputed over their own items while their per-good rows stay in the table marked as components
    and out of the ranking. Averaging the three would weight brevity's two items as heavily as
    honesty's three, and ranking them separately would seat three of the five poles in one stretch of
    the ordering and count non-social three times.

    A cell whose values items parsed nothing has no rows here rather than a row of zeros: the rate's
    denominator is parsed renders, and where that is zero the fact to read is the family's parse rate
    in the instrument-health table above, not a preference.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        records = trace.section(SECTION_SELF_REPORT)
        preferences = forced_choice_win_rates(records, family=FAMILY_VALUES_FORCED_CHOICE)
        if not preferences:
            continue
        pooled = forced_choice_prefix_win_rate(
            records, family=FAMILY_VALUES_FORCED_CHOICE, prefix=NON_SOCIAL_LABEL_PREFIX
        )
        poles = [
            preference
            for label, preference in preferences.items()
            if not label.startswith(NON_SOCIAL_LABEL_PREFIX)
        ]
        if pooled is not None:
            poles.append(pooled)
        poles.sort(key=lambda preference: preference.label)
        ranks = _competition_ranks([preference.win_rate for preference in poles])
        rows.extend(
            _values_ordering_row(trace.step, preference, role=VALUES_POLE_ROLE, rank=rank)
            for preference, rank in zip(poles, ranks, strict=True)
        )
        rows.extend(
            _values_ordering_row(trace.step, preference, role=VALUES_COMPONENT_ROLE, rank=None)
            for label, preference in preferences.items()
            if label.startswith(NON_SOCIAL_LABEL_PREFIX)
        )
    return rows


def _competition_ranks(rates: Sequence[float | None]) -> list[int | None]:
    """Rank descending with ties sharing a rank (1, 2, 2, 4), and no rank where there is no rate.

    Ties share on purpose: two poles a policy chose at the same rate are not ordered against each
    other, and printing 2 and 3 there would read as an ordering the numbers do not contain.
    """
    ordered = sorted((rate for rate in rates if rate is not None), reverse=True)
    return [None if rate is None else ordered.index(rate) + 1 for rate in rates]


def _values_ordering_row(
    step: int, preference: LabelPreference, *, role: str, rank: int | None
) -> dict[str, Any]:
    """Render one label's row: its win rate against the 0.5 chance baseline, with both denominators.

    Both denominators, because a rate of 0.75 over two items and one over twenty read identically in a
    table and only one of them is an ordering.
    """
    return {
        STEP_COLUMN: step,
        "label": preference.label,
        "role": role,
        "rank": EMPTY_CELL if rank is None else str(rank),
        "win rate (0.5 = chance)": _optional_cell(preference.win_rate),
        "items offering": str(preference.n_items_offering),
        "chosen/parsed renders": f"{preference.n_chosen}/{preference.n_parsed_renders_offering}",
    }


def self_report_numeric_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return one row per step and numeric item: its canonical mean, order split and denominators.

    The numeric family's per-item readout -- self-predictions and the numeric placebo alike are
    never pooled across items, because they answer about different things. Both order columns are
    already canonical (a swapped render's answer was reflected through `maximum - x` at parse
    time), so `order gap` reads zero for a policy with a real rate and no first-position
    preference; a large gap is the first-position bias measured on the survey itself.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        records = trace.section(SECTION_SELF_REPORT)
        if not records:
            continue
        for item_id, reading in sorted(numeric_item_readings(records).items()):
            rows.append(
                {
                    STEP_COLUMN: trace.step,
                    "item": item_id,
                    "mean": _optional_cell(reading.mean),
                    "as-authored mean": _optional_cell(reading.as_authored_mean),
                    "swapped mean": _optional_cell(reading.swapped_mean),
                    "order gap": _optional_cell(reading.order_gap, signed=True),
                    "parsed/asked": f"{reading.n_parsed}/{reading.n_asked}",
                }
            )
    return rows


def self_report_risk_mirror_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return the loss-aversion reading per step: the rung gap between the two mirrored ladders.

    The risk family carries two five-rung gamble ladders that are exact translations of each other --
    every outcome of the mixed ladder is the gain-only ladder's plus a constant, so the spreads and
    the probabilities are identical and only the sign of the low branch differs. Rung 1 is the certain
    outcome and the last rung the widest spread, so the difference between the two mean rungs is what
    the policy pays to avoid a loss, and neither ladder alone says that.

    The pair is written down by id, in `RISK_LOSS_AVERSION_MIRROR_PAIR`, and that is deliberate: the
    two items sit in different instruments and different subscales because averaging a pure-gain
    ladder into a loss composite would be wrong, so no pairing mechanism in the registry connects
    them and this is the only place the two are read together. A tracked pairing field on the item
    spec is the follow-up.

    Which makes this the only place their commensurability can be checked, too:
    `assert_ordered_choice_ladders_are_commensurable` refuses mixed rung counts WITHIN a subscale, and
    these two are in different ones, so a gap between a five-rung and a six-rung ladder would
    otherwise print as a plausible loss-aversion number in no unit at all. It reads as its reason
    instead, the way an acquiescence index with one keying direction does.
    """
    gain_only_id, mixed_id = RISK_LOSS_AVERSION_MIRROR_PAIR
    rows: list[dict[str, Any]] = []
    for trace in traces:
        records = trace.section(SECTION_SELF_REPORT)
        gain_only = [record for record in records if str(record["item_id"]) == gain_only_id]
        mixed = [record for record in records if str(record["item_id"]) == mixed_id]
        if not gain_only and not mixed:
            continue
        means = per_item_scores([*gain_only, *mixed])
        gain_only_mean = means.get(gain_only_id)
        mixed_mean = means.get(mixed_id)
        reason = _risk_mirror_reason(
            gain_only, mixed, gain_only_mean=gain_only_mean, mixed_mean=mixed_mean
        )
        gap = None if gain_only_mean is None or mixed_mean is None else gain_only_mean - mixed_mean
        rows.append(
            {
                STEP_COLUMN: trace.step,
                "gain-only mean rung": _optional_cell(gain_only_mean),
                "mixed mean rung": _optional_cell(mixed_mean),
                "loss aversion (gain-only - mixed)": (
                    _optional_cell(gap, signed=True)
                    if reason is None
                    else f"{EMPTY_CELL} ({reason})"
                ),
                "gain-only scored/asked": _scored_over_asked(gain_only),
                "mixed scored/asked": _scored_over_asked(mixed),
            }
        )
    return rows


def _risk_mirror_reason(
    gain_only: Sequence[Mapping[str, Any]],
    mixed: Sequence[Mapping[str, Any]],
    *,
    gain_only_mean: float | None,
    mixed_mean: float | None,
) -> str | None:
    """Return why this step's mirrored ladders have no readable gap, or None where they have one."""
    if not gain_only or not mixed:
        return RISK_MIRROR_LADDER_MISSING
    rungs = sorted({int(record["scale_points"]) for record in (*gain_only, *mixed)})
    if len(rungs) > 1:
        return f"{RISK_MIRROR_RUNG_COUNTS_DIFFER}: {rungs}"
    if gain_only_mean is None or mixed_mean is None:
        return RISK_MIRROR_NOTHING_PARSED
    return None


def _scored_over_asked(records: Sequence[Mapping[str, Any]]) -> str:
    """Render one item's scored-render count over its asked count, `0/0` where it was not asked."""
    return f"{sum(1 for record in records if record.get('score') is not None)}/{len(records)}"


def self_report_counterpart_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return each AI-versus-human counterpart pair's difference, per step.

    A pair is one item asked twice, differing only in who the counterpart is, and the DIFFERENCE is
    the whole quantity: subtracting the two arms cancels whatever the policy's willingness to
    allocate or trust happens to be and leaves the effect of who it was dealing with. A single arm's
    level is not reported as a reading here, because every level in this section is confounded by the
    self-report over-reporting the battery cannot remove.

    A pair with no difference prints its recorded reason rather than a blank cell, for
    `_acquiescence_cell`'s reason: a nominal pair with nothing to difference and a pair where nothing
    parsed are different facts about a run, and only one of them is a problem with it.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        records = trace.section(SECTION_SELF_REPORT)
        if not records:
            continue
        rows.extend(
            {
                STEP_COLUMN: trace.step,
                "counterpart pair": pair,
                "ai counterpart": _optional_cell(reading.ai),
                "human counterpart": _optional_cell(reading.human),
                "gap (ai - human)": _counterpart_gap_cell(reading),
                "ai parsed/asked": f"{reading.n_ai_parsed}/{reading.n_ai_asked}",
                "human parsed/asked": f"{reading.n_human_parsed}/{reading.n_human_asked}",
            }
            for pair, reading in sorted(counterpart_gaps(records).items())
        )
    return rows


def _counterpart_gap_cell(reading: CounterpartGap) -> str:
    """Render a counterpart pair's difference, or the recorded reason it has none."""
    if reading.reason is not None:
        return f"{EMPTY_CELL} ({reading.reason})"
    return _optional_cell(reading.gap, signed=True)


def self_report_cheap_talk_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return each cheap-talk item's announcement-against-action reading, per step and item.

    Per item and never pooled across items, the numeric family's rule: the items describe different
    situations with different stakes, so a mean over "how often does it do what it said" across them
    is not a rate of anything.

    `announced only` counts the completions that announced an intention and then emitted no action.
    That is a coverage fact about the format rather than dishonesty, and it is what makes a collapsing
    cheap-talk parse rate readable -- without it, a policy that stopped acting and a policy that
    ignored the instruction move the same denominator the same way. The announced-against-acted pair
    counts carry the DIRECTION of each mismatch, which a single match rate cannot.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        records = trace.section(SECTION_SELF_REPORT)
        if not records:
            continue
        rows.extend(
            {
                STEP_COLUMN: trace.step,
                "item": item_id,
                "match rate": _optional_cell(reading.match_rate),
                "parsed/asked": f"{reading.n_parsed}/{reading.n_asked}",
                "announced only": str(reading.n_announced_only),
                "announced -> acted": _cheap_talk_pairs_cell(reading),
            }
            for item_id, reading in sorted(cheap_talk_readings(records).items())
        )
    return rows


def _cheap_talk_pairs_cell(reading: CheapTalkReading) -> str:
    """Render the joint (announced, acted) counts, so a mismatch's direction reads off the table."""
    return (
        ", ".join(
            f"{announced} -> {acted}: {count}"
            for (announced, acted), count in sorted(reading.pair_counts.items())
        )
        or EMPTY_CELL
    )


def self_report_tagged_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return one row per step and forced-tag item: its word shares, headroom and movement.

    The registered reading of both tag families, and the reason it is a distribution: a tag answer is
    one word from a closed menu, so a mean over vocabulary positions would let the authored menu order
    decide how large a movement looks. Movement is total variation distance from the FIRST trace's
    distribution (step 0), the same scoring the nominal controls get, which is what makes a tag mirror
    comparable with the lettered item it mirrors -- a distribution that moves on the lettered item and
    not on its mirror is a lettered-format response rather than an attribution.

    Entropy is the item's available headroom. Without it a flat 0.000 would credit an item with a
    stability it may never have had the range to demonstrate, exactly as for a negative control.

    A distance reading `-` carries its recorded reason, for `_counterpart_gap_cell`'s reason: an item
    the base cell never asked, a cell where nothing parsed, and an item whose vocabulary was
    re-authored between the two passes are three different facts about the comparison, and a blank
    cell would hide which of them happened.
    """
    with_records = [trace for trace in traces if trace.section(SECTION_SELF_REPORT)]
    if not with_records:
        return []
    baseline = tagged_readings(with_records[0].section(SECTION_SELF_REPORT))
    rows: list[dict[str, Any]] = []
    for trace in with_records:
        records = trace.section(SECTION_SELF_REPORT)
        subscales = {
            str(record["item_id"]): str(record.get("subscale"))
            for record in records
            if str(record.get("kind")) == SURVEY_TAGGED
        }
        rows.extend(
            {
                STEP_COLUMN: trace.step,
                "item": item_id,
                "subscale": subscales.get(item_id, EMPTY_CELL),
                "menu words": _vocabulary_width_cell(reading),
                "shares": _tagged_shares_cell(reading),
                "entropy (bits)": _optional_cell(reading.entropy_bits),
                "parsed/asked": f"{reading.n_parsed}/{reading.n_asked}",
                "TV vs step0": _tagged_distance_cell(
                    tagged_distribution_distance(baseline.get(item_id), reading)
                ),
            }
            for item_id, reading in sorted(tagged_readings(records).items())
        )
    return rows


def _vocabulary_width_cell(reading: TaggedReading) -> str:
    """Render how many words the item's menu holds, `-` where its own records disagreed.

    Printed rather than left implicit because a share is unreadable without it: 0.33 is the flat
    answer on a three-word menu and a peak on a six-word one, and the two tag families run both.
    """
    width = reading.vocabulary_size
    return EMPTY_CELL if width is None else str(width)


def _tagged_shares_cell(reading: TaggedReading) -> str:
    """Render one item's word shares in canonical menu order, `-` where nothing parsed."""
    return ", ".join(f"{word}:{share:.2f}" for word, share in reading.shares.items()) or EMPTY_CELL


def _tagged_distance_cell(distance: TaggedDistance) -> str:
    """Render a tagged item's movement from step 0, or the recorded reason it has none."""
    if distance.reason is not None:
        return f"{EMPTY_CELL} ({distance.reason})"
    return _optional_cell(distance.distance)


def _self_report_order_cell(records: Sequence[Mapping[str, Any]]) -> str:
    """Render one family's order-disagreement rate with the pair counts behind it."""
    rate = order_disagreement_rate(records, item_field="item_id", answer_field="canonical_index")
    compared = _order_pairs(records, parsed_only=True)
    asked = _order_pairs(records, parsed_only=False)
    if (compared == 0) != (rate is None):
        logger.warning(
            f"self-report order-disagreement rate is {rate} over {compared} comparable pair(s); "
            f"the rate and its denominator have drifted apart"
        )
    return format_rate(rate, compared, asked, unit="pairs")


def _order_pairs(records: Sequence[Mapping[str, Any]], *, parsed_only: bool) -> int:
    """Count the (item, sample) pairs asked under both LETTER orders, optionally both-parsed only.

    Two denominators for one rate, for the reason `_comparable_order_pairs` and `_order_pairs_asked`
    keep them apart on the probe side: the parsed count floats with the parse rate, so a denominator
    move shows up as a gap between the two rather than as the rate moving.

    Numeric records are skipped EXPLICITLY, and the skip is load-bearing since the swapped-stem
    counterbalance landed: a self-prediction item now renders under two order names per sample, but
    its answers are integers with no canonical option index, so counting its pairs here would put
    them in a letter-disagreement denominator whose numerator can never see them -- an order rate
    diluted toward zero. Its own order readout is `numeric_item_readings`' order gap, in the numeric
    table. `test_the_numeric_family_reports_no_comparable_order_pairs` pins this skip.
    """
    by_pair: dict[tuple[str, int], set[str]] = {}
    for record in records:
        if str(record.get("kind")) == SURVEY_NUMERIC:
            continue
        if parsed_only and record.get("canonical_index") is None:
            continue
        key = (str(record["item_id"]), int(record["sample_index"]))
        by_pair.setdefault(key, set()).add(str(record["option_order_name"]))
    return sum(1 for orders in by_pair.values() if len(orders) > 1)


def split_rows(
    arm: str, traces: Sequence[EvalTrace], field_name: str, *, window: Sequence[int]
) -> list[dict[str, Any]]:
    """Return the trained game split by one label field: base cell, pooled late window, and delta.

    Pooled rather than per-step because a split divides an already-small per-cell denominator, and
    ground rule one forbids reading one cell against one cell. The base side is step 0 alone --
    there is only one base cell -- so the delta column inherits that cell's noise, which is why the
    per-step ladder is the primary reading and this is the split beside it.
    """
    game_id = trained_game_of(arm)
    if game_id is None:
        return []
    base = [trace for trace in traces if trace.step == 0]
    late = [trace for trace in traces if trace.step in window]
    rows: list[dict[str, Any]] = []
    for value in values_present(traces, game_id, field_name):
        where = {field_name: value}
        base_rate = game_rate(base, game_id, where=where)
        late_rate = game_rate(late, game_id, where=where)
        rows.append(
            {
                field_name: value,
                "role": (
                    variant_role(value, pinned_variants_of(arm))
                    if field_name == VARIANT_FIELD
                    else "surface frame"
                ),
                "measure": behaviour_field(game_id),
                BASE_COLUMN: base_rate.cell,
                late_window_column(window): late_rate.cell,
                DELTA_COLUMN: _delta_cell(base_rate, late_rate),
            }
        )
    return rows


# Column labels of the position table, read back by tests and by the renderer's prose.
SECTION_COLUMN = "section"
FRAMING_COLUMN = "framing"
PICKS_FIRST_COLUMN = "picks-first share (k/n parsed draws)"
COOP_FIRST_COLUMN = "coop, coop-label printed first (renders; k/n draws)"
COOP_SECOND_COLUMN = "coop, coop-label printed second (renders; k/n draws)"
POSITION_GAP_COLUMN = "gap first-second"
POSITION_GAP_BAND_COLUMN = "gap 2SE (band; n)"
ORDER_SPREAD_COLUMN = "canonical-swapped spread (n prompts both ways)"
PRINT_ORDERS_COLUMN = "print orders present"
FRAMING_FIELD = "counterpart_framing"


def position_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return, per step and per one-shot two-label game, cooperation by where the cooperative label was printed.

    One row per (step, section or framing, game, payoff variant), never pooled across variants,
    for every game whose completion names one of two printed labels; the framing sweep's games are
    rows too, one per framing, since the belief framings split the same way when this was first seen.
    Each row carries the existing pooled rate for the same grain, so the two read together.

    Why this table exists beside the pooled rate and the canonical-minus-swapped spread, in one
    sentence each. The pooled rate is unbiased as an average because the roster counterbalances which
    authored label is cooperative, so a policy picking whatever is printed first cooperates on exactly
    half the prompts and the rate does not move. The spread pairs each prompt's two print orders and
    averages the difference over prompts, and a first-position preference contributes +x on the
    prompts whose cooperative label is authored first and -x on the prompts whose cooperative label is
    authored second, so it cancels: the spread detects a preference for a WORD in a position, not for
    a position. This table splits cooperation by whether the cooperative label was printed first or
    second, which is the one cut a position preference cannot cancel in. The reduction and the
    derivation of "picked the option printed first" live in `games.position_preference`.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        for section in (SECTION_GAME_BEHAVIOR, SECTION_FRAMING_SWEEP):
            for game_id in games_present([trace], section=section):
                if game_id not in ONE_SHOT_ACTION_GAME_IDS:
                    continue
                framings = (
                    [None]
                    if section == SECTION_GAME_BEHAVIOR
                    else values_present([trace], game_id, FRAMING_FIELD, section=section)
                )
                for framing in framings:
                    where = {} if framing is None else {FRAMING_FIELD: framing}
                    for variant in values_present(
                        [trace], game_id, VARIANT_FIELD, section=section, where=where
                    ):
                        grain = {**where, VARIANT_FIELD: variant}
                        records = behaviour_records([trace], game_id, where=grain, section=section)
                        rows.append(
                            _position_row(
                                step=trace.step,
                                section=section if framing is None else framing,
                                game_id=game_id,
                                variant=variant,
                                pooled=rate_of(records, COOP_FIELD),
                                split=position_split(records),
                            )
                        )
    return rows


def _position_row(  # noqa: PLR0913 - one row's labels and its two reductions
    *, step: int, section: str, game_id: str, variant: str, pooled: Rate, split: PositionSplit
) -> dict[str, Any]:
    """Render one grain's position split as a table row, every figure with its denominator."""
    share = split.picks_first_share
    return {
        STEP_COLUMN: step,
        SECTION_COLUMN: section,
        GAME_COLUMN: game_id,
        VARIANT_FIELD: variant,
        RATE_COLUMN: pooled.cell,
        PICKS_FIRST_COLUMN: (
            EMPTY_CELL
            if share is None
            else f"{share:.3f} ({split.picks_first_k}/{split.picks_first_n})"
        ),
        COOP_FIRST_COLUMN: split.first.cell,
        COOP_SECOND_COLUMN: split.second.cell,
        POSITION_GAP_COLUMN: EMPTY_CELL if split.gap is None else f"{split.gap:+.3f}",
        POSITION_GAP_BAND_COLUMN: _gap_band_cell(split),
        ORDER_SPREAD_COLUMN: (
            f"{EMPTY_CELL} (0)"
            if split.spread is None
            else f"{split.spread:+.3f} ({split.n_prompts_paired})"
        ),
        PRINT_ORDERS_COLUMN: "+".join(split.print_orders),
    }


def _gap_band_cell(split: PositionSplit) -> str:
    """Render the gap's band with how it was licensed, or a dash where nothing licensed one."""
    if split.gap_2se is None or split.gap_band == BAND_NONE:
        return EMPTY_CELL
    n_used = (
        split.n_prompts_paired
        if split.gap_band == BAND_PAIRED
        else split.first.n_renders + split.second.n_renders
    )
    return f"{split.gap_2se:.3f} ({split.gap_band}; {n_used})"


def per_round_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return cooperation by round index for the iterated game, per step.

    Where end-game defection would show up: a profile holding across early rounds and dropping in
    the last one or two. A flat profile is equally a result -- it says the model never found the
    backward induction -- which is why this is a table rather than one averaged rate.
    """
    return rows_from_frame(per_round_profile(traces))


def section_health_rows(traces: Sequence[EvalTrace]) -> list[dict[str, Any]]:
    """Return per-step, per-section parse-failure and truncation counts.

    The document's own instrument check. A behaviour rate that moved because a section stopped
    parsing is the failure mode every rate shares, and it is invisible in a rate: the denominators
    are what move. Kept as its own table so the movement can be read directly.
    """
    rows: list[dict[str, Any]] = []
    for trace in traces:
        for section in SECTIONS:
            records = trace.section(section)
            if not records:
                continue
            parsed = parsed_count(records)
            truncated = truncated_count(records)
            rows.append(
                {
                    STEP_COLUMN: trace.step,
                    "section": section,
                    "parsed/asked": f"{parsed}/{len(records)}",
                    "parse_failure_rate": f"{1.0 - parsed / len(records):.3f}",
                    "truncated/asked": f"{truncated}/{len(records)}",
                    "truncation_rate": f"{truncated / len(records):.3f}",
                }
            )
    return rows


def arm_facts(arm: str, traces: Sequence[EvalTrace], ladder: Sequence[int]) -> dict[str, Any]:
    """Return the provenance and configuration facts for one arm, read off its cells' meta.

    The eval render grading earns its own column: two arms can train under different grading rules
    and still be EVALUATED on identically rendered prompts, and a reader who assumes the arm's
    grading reached the eval prompts will misread the contrast between them.
    """
    steps = [trace.step for trace in traces]
    trained = trained_game_of(arm)
    rendered = (
        sorted({str(record["render_grading"]) for record in behaviour_records(traces, trained)})
        if trained is not None
        else []
    )
    return {
        "arm": arm,
        "trained game": trained or UNREGISTERED,
        "training grading": grading_of(arm) or UNREGISTERED,
        "pinned payoff variants": ",".join(pinned_variants_of(arm)) or ALL_VARIANTS,
        "eval render grading (trained game)": ",".join(rendered) or EMPTY_CELL,
        "cells": f"{len(traces)}/{len(ladder)}",
        "steps present": steps_label(steps),
        "steps missing": steps_label(sorted(set(ladder) - set(steps))),
        # A banked step-0 copy was generated at the bank entry's commit, named in the arm's notes
        # rather than pooled here, where it would read as a mixed-provenance pass.
        "git_sha": ",".join(
            sorted(
                {str(trace.meta.get("git_sha")) for trace in traces if trace.banked_from is None}
            )
        )
        or EMPTY_CELL,
        "backend": ",".join(sorted({str(trace.meta.get("backend_kind")) for trace in traces})),
        "thinking": ",".join(sorted({str(trace.meta.get("thinking")) for trace in traces})),
    }


def _dictator_note(arm: str) -> list[str]:
    """Return the note the unilateral-split arm's trained-game section cannot be read without."""
    if trained_game_of(arm) != DICTATOR_GAME_ID:
        return []
    return [
        (
            f"The trained game here is `{DICTATOR_GAME_ID}`, which has no counterpart and therefore no "
            f"cooperation rate at all: the measure is `{KEEP_FIELD}`, the share of the endowment the "
            f"model kept, where higher is more selfish rather than less cooperative. The eval corpus "
            f"carries only a handful of held-out split prompts per cell, so this ladder is thin by "
            f"construction and its per-variant splits thinner. The matched comparison -- the prompts the "
            f"arm trained on, recomputed from the training parquet -- is not in this document."
        )
    ]


def contrast_groups(arms: Sequence[str]) -> list[ContrastGroup]:
    """Return the arms that trained the same game, grouped, with the axis that differs named.

    Derived from `games.arms.ARMS` and the arms actually present, so the grading pair and the
    payoff-rung pair both fall out and no arm name is written down here. An unregistered arm cannot
    be grouped and is left out rather than guessed at.
    """
    by_game: dict[str, list[str]] = {}
    for arm in arms:
        game_id = trained_game_of(arm)
        if game_id is not None:
            by_game.setdefault(game_id, []).append(arm)
    groups: list[ContrastGroup] = []
    for game_id, members in sorted(by_game.items()):
        if len(members) < MIN_ARMS_FOR_A_CONTRAST:
            continue
        gradings = {grading_of(arm) for arm in members}
        variants = {pinned_variants_of(arm) for arm in members}
        differs = [
            name
            for name, distinct in (
                ("grading rule", len(gradings)),
                ("pinned payoff variant", len(variants)),
            )
            if distinct > 1
        ]
        groups.append(
            ContrastGroup(
                game_id=game_id,
                arms=tuple(sorted(members)),
                axis=" and ".join(differs) or "nothing in the registry differs",
            )
        )
    return groups


def shared_draws(
    cells: Mapping[str, Sequence[EvalTrace]], arms: Sequence[str]
) -> tuple[SharedDraw, ...]:
    """Find the (step, section) pairs where these arms' completions are byte-identical.

    Detected from the per-section digests `battery_cells` records before dropping the completions,
    so it costs no memory. This happens for real: the twin pair's step-0 game-behaviour sections are
    one deterministic draw of the same base weights, so their step-0 contrast rows are equal by
    construction -- and without this list that reads exactly like agreement at baseline. Cells read
    by a path that recorded no digests simply detect nothing.
    """
    draws: list[SharedDraw] = []
    by_step: dict[int, list[EvalTrace]] = {}
    for arm in arms:
        for trace in cells.get(arm, ()):
            by_step.setdefault(trace.step, []).append(trace)
    for step, traces in sorted(by_step.items()):
        by_digest: dict[tuple[str, str], list[EvalTrace]] = {}
        for trace in traces:
            digests = trace.meta.get(SECTION_DIGESTS_KEY) or {}
            for section, digest in sorted(digests.items()):
                by_digest.setdefault((str(section), str(digest)), []).append(trace)
        draws.extend(
            SharedDraw(
                step=step,
                section=section,
                arms=tuple(sorted(trace.arm for trace in sharers)),
                n_records=len(sharers[0].section(section)),
            )
            for (section, _digest), sharers in sorted(by_digest.items())
            if len(sharers) > 1
        )
    return tuple(draws)


def rows_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a `games.report` table into this module's row dicts, so one renderer serves both."""
    return [
        {str(key): value for key, value in row.items()} for row in frame.to_dict(orient="records")
    ]


def build_readout(battery_dir: Path) -> BatteryReadout:
    """Read a battery directory and compute every table the document is made of."""
    cells, excluded = read_cells(battery_dir)
    ladder = battery_ladder(cells)
    arms = [_build_arm(arm, traces, ladder=ladder) for arm, traces in cells.items()]
    contrasts = [
        replace(group, shared_draws=shared_draws(cells, group.arms))
        for group in contrast_groups(sorted(cells))
    ]
    return BatteryReadout(
        battery_dir=battery_dir,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ladder=ladder,
        arms=arms,
        contrasts=contrasts,
        excluded=excluded,
        problems=_problems(cells, arms),
        cell_count=sum(len(traces) for traces in cells.values()),
    )


def _build_arm(arm: str, traces: Sequence[EvalTrace], *, ladder: Sequence[int]) -> ArmReadout:
    """Compute one arm's tables and the notes they cannot be read without."""
    steps = tuple(trace.step for trace in traces)
    window = late_window(steps)
    notes = _dictator_note(arm)
    notes.extend(_banked_notes(traces))
    if trained_game_of(arm) is None:
        notes.append(
            f"Arm `{arm}` is not in `games.arms.ARMS`, so nothing here can say which game it "
            f"trained: its trained-game and split tables are empty, and every game it evaluated "
            f"reads as transfer."
        )
    if not window:
        notes.append(
            "No checkpoint past step 0 has landed, so the pooled late window is empty and every "
            "split contrast below is a base cell against nothing."
        )
    elif len(window) == 1:
        notes.append(
            f"The pooled late window is a single cell (step {window[0]}), so every base-against-"
            f"late contrast below is one cell against one cell -- exactly the comparison ground "
            f"rule 1 forbids reading alone. It widens as later checkpoints land."
        )
    flips = dt_flip_rows(traces)
    scope = flip_scope(traces)
    if len(flips) > scope.n_comparable:
        logger.warning(
            f"arm {arm}: {len(flips)} endorsement flip(s) but only {scope.n_comparable} "
            f"comparable item(s); the flip table and its denominator have drifted apart"
        )
    return ArmReadout(
        arm=arm,
        steps=steps,
        missing_steps=tuple(sorted(set(ladder) - set(steps))),
        trained_game=trained_game_of(arm),
        pinned_variants=pinned_variants_of(arm),
        late_window=window,
        facts=arm_facts(arm, traces, ladder),
        trained_ladder=trained_ladder_rows(arm, traces),
        transfer_matrix=transfer_matrix_rows(arm, traces),
        transfer_windows=transfer_window_rows(arm, traces, window=window),
        negative_control=negative_control_rows(traces),
        dt_ladder=dt_ladder_rows(traces),
        dt_endorsements=dt_endorsement_rows(traces),
        dt_flips=flips,
        flip_scope=scope,
        churn=adjacent_churn(traces),
        capability_ladder=capability_ladder_rows(traces),
        self_report_ladder=self_report_ladder_rows(traces),
        self_report_controls=self_report_control_rows(traces),
        self_report_health=self_report_health_rows(traces),
        self_report_instruments=self_report_instrument_rows(traces),
        self_report_control_distributions=self_report_control_distribution_rows(traces),
        self_report_nominal_choices=self_report_nominal_choice_rows(traces),
        self_report_values_ordering=self_report_values_ordering_rows(traces),
        self_report_numeric=self_report_numeric_rows(traces),
        self_report_risk_mirror=self_report_risk_mirror_rows(traces),
        self_report_counterparts=self_report_counterpart_rows(traces),
        self_report_cheap_talk=self_report_cheap_talk_rows(traces),
        self_report_tagged=self_report_tagged_rows(traces),
        self_report_calibration=self_report_calibration_rows(traces),
        variant_splits=split_rows(arm, traces, VARIANT_FIELD, window=window),
        reskin_splits=split_rows(arm, traces, RESKIN_FIELD, window=window),
        position_splits=position_rows(traces),
        per_round=per_round_rows(traces),
        section_health=section_health_rows(traces),
        notes=notes,
    )


def _problems(cells: Mapping[str, Sequence[EvalTrace]], arms: Sequence[ArmReadout]) -> list[str]:
    """Return everything about this pass that makes a number below provisional."""
    problems: list[str] = []
    problems.extend(_provenance_problems(cells))
    problems.extend(_section_problems(cells))
    problems.extend(
        f"UNREGISTERED arm `{arm.arm}`: not in `games.arms.ARMS`, so its trained game is unknown "
        f"and its trained-game tables are empty."
        for arm in arms
        if arm.trained_game is None
    )
    problems.extend(
        f"MOCK BACKEND cell `{trace.label}`: `--backend mock` samples nothing, so its rows are "
        f"plumbing output rather than measurements."
        for traces in cells.values()
        for trace in traces
        if trace.from_mock_backend
    )
    return problems


def _provenance_problems(cells: Mapping[str, Sequence[EvalTrace]]) -> list[str]:
    """Return the provenance disagreements across this battery's cells.

    A battery is one pass at one commit under one sampling configuration. Where that is not true the
    cells are not comparable, and the difference is invisible in every rate they produce -- so it is
    checked per field rather than assumed, naming the cells on the minority side.

    A banked step-0 copy (`EvalTrace.banked_from`) sits out the `git_sha` vote only: it was
    generated at whatever commit produced the bank entry, which is the owner's design and is named
    in the arm's notes (`_banked_notes`), and counting it here would mark every banked battery
    error-containing by construction. It still votes on sampling, thinking mode and backend, which
    are part of the bank key and so have to agree with the arm's own cells.
    """
    problems: list[str] = []
    traces = [trace for arm_traces in cells.values() for trace in arm_traces]
    for field_name in PROVENANCE_META_FIELDS:
        grouped: dict[str, list[str]] = {}
        for trace in traces:
            if field_name == "git_sha" and trace.banked_from is not None:
                continue
            key = json.dumps(trace.meta.get(field_name), sort_keys=True)
            grouped.setdefault(key, []).append(trace.label)
        if len(grouped) == 1:
            continue
        majority = max(grouped, key=lambda key: len(grouped[key]))
        problems.extend(
            f"PROVENANCE DISAGREEMENT on `{field_name}`: cells {sorted(labels)} carry `{value}`, "
            f"not the majority `{majority}`."
            for value, labels in sorted(grouped.items())
            if value != majority
        )
    return problems


def _banked_notes(traces: Sequence[EvalTrace]) -> list[str]:
    """Name each of an arm's cells that is a byte-identical copy of a shared bank entry, and its source.

    A provenance note rather than a problem: the copy is the arm's step-0 cell by design, labelled
    through its hash-verified sidecar, and a reader has to be able to see that it was generated under
    another arm's name at another commit without the document shouting ERROR-CONTAINING over it.

    The entry is named by its bank key, not by the sidecar's `source_key` prefix, which carries a
    bucket name: these notes are rendered into a document that gets pasted into docs.
    """
    notes: list[str] = []
    for trace in traces:
        provenance = trace.banked_from
        if provenance is None:
            continue
        notes.append(
            f"Step {trace.step} of this arm is a byte-identical copy of the shared bank entry under "
            f"key `{provenance.get('bank_key')}`, generated under arm `{provenance.get('source_arm')}` "
            f"at git SHA `{provenance.get('source_git_sha')}` and labelled here through its "
            f"sha256-verified provenance sidecar; the `git_sha` fact above pools the arm's own cells "
            f"only."
        )
    return notes


def _section_problems(cells: Mapping[str, Sequence[EvalTrace]]) -> list[str]:
    """Return the cell sections whose parse-failure or truncation rate makes them unreadable."""
    problems: list[str] = []
    for traces in cells.values():
        for trace in traces:
            for section in SECTIONS:
                records = trace.section(section)
                if not records:
                    continue
                failure = 1.0 - parsed_count(records) / len(records)
                truncated = truncated_count(records)
                if failure > PARSE_FAILURE_BANNER_THRESHOLD:
                    problems.append(
                        f"HIGH PARSE FAILURE `{trace.label}` section `{section}`: {failure:.3f} of "
                        f"{len(records)} records produced no usable answer, so that section's rate "
                        f"is mostly a denominator."
                    )
                if truncated / len(records) > TRUNCATION_BANNER_THRESHOLD:
                    problems.append(
                        f"HIGH TRUNCATION `{trace.label}` section `{section}`: "
                        f"{truncated}/{len(records)} completions never closed their thinking, so "
                        f"that section is partly measuring the generation cap."
                    )
    return problems
