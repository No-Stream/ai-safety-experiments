"""Rates, bands and calibration over the games trace judge's rows, in the wave-3 scoring conventions.

What this module turns judged rows into, and the convention each follows:

- **Per scope cell**: shares are keyed by ``arm|cell|section::game::framing::variant@step``, never by
  the battery cell directory alone. The hand-label pass found ``instruction-compliance`` to be a cell
  indicator (7 of 7 in the disclosed-counterpart framing), so a reader must not be able to pool
  disclosed and undisclosed framings, or the two payoff variants, by accident; this is also the
  ``game::framing::variant@step`` spelling the scratch harnesses' exports use. The record's decision
  is part of the key too, so cooperators and defectors of one scope never pool: a defector's scope
  reads ``section::game::framing::variant::defect``, while a cooperator's carries no marker, because
  every rates file written before the decision selector existed (2026-09-03) used the bare label and
  those labels must not move. A stored row without a ``decision`` stamp is a cooperator, the only
  decision the loader admitted before then. Each scope cell carries
  the share of each decision basis, the coupling-story count (``counterpart_assumption`` in
  ``copies-me`` / ``correlated``), the ``diagonal_comparison`` levels and the dominance flags as k/n
  with Clopper-Pearson 95% bounds, beside ``examined`` / ``skipped_empty`` / ``not_judged`` /
  ``errored`` denominators read off the census rather than off the judged file (a record never judged
  leaves no row there). The one-sided upper bound is the ``absence_floor`` convention the wave-3
  harnesses quote for an absence clause: at k=0 it is 1 - 0.05^(1/n). Every cell is stamped
  ``coupling_clause_in_prompt``: whether the prompt's counterpart paragraph asserted the coupling (the
  framing's clause on a framing-sweep cell, the game's own on every other section), so the control's
  "invented coupling story" reading cannot be applied to a twin-framed cell by accident -- a
  ``game-behavior::twin-pd::no-framing`` cell reads as unframed and carried the twin clause. A framing
  off the tracked registry came from the run's ``--framings-file`` and reads decoupled, because that
  loader refuses a coupling assertion; the census row's ``framings_digest`` is what tells that case
  apart from a mistyped id, which is refused by name.
- **Per prompt**: the same shares over each prompt's judged records, because every band below is
  a between-prompt band and the harnesses reduce through ``prompt_id`` first.
- **Within-arm step-0 to step-70 delta**: the paired between-prompt 2SE, 2 * SD(per-prompt deltas) /
  sqrt(n_prompts) over prompts with judged records at BOTH steps, quoted with n_paired and
  n_unpaired -- the wave-3 band-convention correction's ``paired_2se`` (a scratch script, not tracked),
  which is the convention the published wave-3 bands were recomputed under; where fewer than two
  prompts pair, the band falls back to quadrature of the two levels' between-prompt 2SEs and the row
  SAYS SO in ``band_method``. A per-prompt-clustered 2SE (cluster-robust, prompts as clusters,
  combined in quadrature across the two steps) rides beside as the secondary line. A scope with any
  step count other than two gets no delta and is named in a warning, never dropped silently.
- **Keyword agreement**: the judge's coarse basis against the deterministic companion on every row,
  as a rate plus the disagreement listing. Never merged: disagreement is an instrument finding.
- **Drawn overlay**: the hand-read samples' rows, matched on (step, prompt_id, sample_index, digest of
  the completion), tallied again beside the census so the registered "drawn cooperator traces" reading
  can be reported from the same rows. The step is in the handle because a short deterministic trace can
  be byte-identical at two steps of one prompt.
- **Hand-label validation**: a confusion per labelled dimension with a strict and a lenient agreement
  line (lenient credits a judge primary equal to the hand secondary, or a judge secondary equal to the
  hand primary; the two are never merged), exact agreement, every miss listed by key, enum levels with
  no hand positives named rather than printed as 0/0, and a caveat on the dominance flags, which
  coincide with ``payoff_reasoning_present`` on a cooperator-only set by construction. Hand-label keys
  follow the sample exports' spelling (``rung`` or ``source_arm``, the export's own ``cell``, ``order``
  for the control's print order) and are translated to judged rows through the sample files' completion
  digests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any

from games.prompts import COUNTERPART_FRAMINGS, framing_states_coupling, has_coupling_clause
from games.trace_judge_schema import (
    COUNTERPART_ASSUMPTIONS,
    COUPLING_ASSUMPTIONS,
    DECISION_BASES,
    DIAGONAL_COMPARISON_LEVELS,
    VERDICT_BOOLS,
    JudgeVerdict,
    verdict_from_row,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

BOUND_CONFIDENCE = 0.95
_BISECTION_STEPS = 100
_MIN_PROMPTS_FOR_A_BAND = 2
_STEPS_IN_A_DELTA = 2
NO_FRAMING = "no-framing"
NULL_LEVEL = "null"

DECISION_COOPERATE = "cooperate"
DECISION_DEFECT = "defect"
DECISIONS: tuple[str, ...] = (DECISION_COOPERATE, DECISION_DEFECT)
"""The two decisions a parsed one-shot matrix-game record can carry: ``coop_fraction`` 1.0 or 0.0."""
DECISION_FIELD = "decision"

FRAMINGS_DIGEST_FIELD = "framings_digest"
"""Census field carrying the digest of the runtime framings file the cell rendered under, if any.

Stamped off the cell's meta by `games.trace_judge_records`, because this module reads a census with
no tree and no meta beside it, and a framing off the tracked registry means two different things
depending on whether that cell loaded a framings file at all.
"""

type Handle = tuple[int, str, int, str]
"""(step, prompt_id, sample_index, sha256 of the completion): the arm-agnostic identity of one record."""


def _log_binomial_pmf(k: int, n: int, p: float) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space so n in the thousands cannot overflow."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    return sum(math.exp(_log_binomial_pmf(i, n, p)) for i in range(k + 1))


def _solve_p(target: float, cdf_at: Callable[[float], float]) -> float:
    """Bisect for the p at which the (decreasing) ``cdf_at(p)`` equals ``target``."""
    low, high = 0.0, 1.0
    for _ in range(_BISECTION_STEPS):
        mid = (low + high) / 2
        if cdf_at(mid) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def binomial_bounds(
    k: int, n: int, *, confidence: float = BOUND_CONFIDENCE
) -> dict[str, float | None]:
    """Clopper-Pearson bounds on a share: the two-sided interval and the one-sided upper bound.

    The one-sided upper bound is the ``absence_floor`` convention: at k=0 it is
    1 - (1 - confidence)^(1/n), the prevalence a zero count would rule out. All three are ``None`` at
    n=0, because a bound on nothing is not a bound.
    """
    if n == 0:
        return {"ci95_lower": None, "ci95_upper": None, "upper_95_one_sided": None}
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else _solve_p(1.0 - alpha / 2, lambda p: _binomial_cdf(k - 1, n, p))
    upper = 1.0 if k == n else _solve_p(alpha / 2, lambda p: _binomial_cdf(k, n, p))
    one_sided = 1.0 if k == n else _solve_p(alpha, lambda p: _binomial_cdf(k, n, p))
    return {"ci95_lower": lower, "ci95_upper": upper, "upper_95_one_sided": one_sided}


def clustered_2se(clusters: Mapping[str, Sequence[float]]) -> float | None:
    """Twice the cluster-robust standard error of a pooled share, prompts as clusters.

    V = G/(G-1) * sum_g (sum_{i in g} (y_i - p))^2 / N^2, the sandwich variance of a mean with a
    finite-cluster correction. ``None`` with fewer than two clusters, where it is undefined.
    """
    if len(clusters) < _MIN_PROMPTS_FOR_A_BAND:
        return None
    values = [value for cluster in clusters.values() for value in cluster]
    total = len(values)
    if total == 0:
        return None
    share = sum(values) / total
    groups = len(clusters)
    residual_sums = (sum(value - share for value in cluster) for cluster in clusters.values())
    variance = groups / (groups - 1) * sum(r * r for r in residual_sums) / (total * total)
    return 2 * math.sqrt(variance)


def _between_prompt_2se(shares: Sequence[float]) -> float | None:
    if len(shares) < _MIN_PROMPTS_FOR_A_BAND:
        return None
    return 2 * stdev(shares) / math.sqrt(len(shares))


def _quadrature(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return math.sqrt(first**2 + second**2)


def _dimension_values(verdict: JudgeVerdict) -> dict[str, float]:
    """Every rated dimension of one verdict as a 0/1 value, basis and diagonal levels included."""
    values = {
        "coupling_story": float(verdict.coupling_story),
        "diagonal_comparison:only": float(verdict.diagonal_comparison == "only"),
        "diagonal_comparison:supporting": float(verdict.diagonal_comparison == "supporting"),
        "dominance_raised": float(verdict.dominance_raised),
        "dominance_rejected": float(verdict.dominance_rejected),
        "payoff_reasoning_present": float(verdict.payoff_reasoning_present),
    }
    for basis in DECISION_BASES:
        values[f"basis:{basis}"] = float(verdict.decision_basis == basis)
    return values


DIMENSIONS: tuple[str, ...] = tuple(
    _dimension_values(
        JudgeVerdict(
            decision_basis="other",
            secondary_basis=None,
            counterpart_assumption="not-discussed",
            diagonal_comparison="none",
            dominance_raised=False,
            dominance_rejected=False,
            payoff_reasoning_present=False,
            evidence="x",
        )
    )
)
"""The rated dimensions in output order: the clause dimensions and flags, then one per basis level."""

BOUNDED_DIMENSIONS: tuple[str, ...] = tuple(d for d in DIMENSIONS if not d.startswith("basis:"))
"""The dimensions reported as k/n with binomial bounds; basis levels are reported as a distribution."""


def _prompt_values() -> defaultdict[str, defaultdict[str, list[float]]]:
    return defaultdict(lambda: defaultdict(list))


def _zeroed(levels: Iterable[str]) -> Counter[str]:
    return Counter(dict.fromkeys(levels, 0))


@dataclass(slots=True)
class CellTally:
    """Judged verdicts of one scope cell being counted: shares, per-prompt values, the companion read.

    Every enum level is present from the start so a zero is a zero, not an absent key.
    """

    judged: int = 0
    basis: Counter[str] = field(default_factory=lambda: _zeroed(DECISION_BASES))
    secondary_basis: Counter[str] = field(
        default_factory=lambda: _zeroed((*DECISION_BASES, NULL_LEVEL))
    )
    assumption: Counter[str] = field(default_factory=lambda: _zeroed(COUNTERPART_ASSUMPTIONS))
    diagonal: Counter[str] = field(default_factory=lambda: _zeroed(DIAGONAL_COMPARISON_LEVELS))
    evidence_not_found: int = 0
    keyword_agree: int = 0
    keyword_disagreements: list[dict[str, str]] = field(default_factory=list)
    per_prompt: defaultdict[str, defaultdict[str, list[float]]] = field(
        default_factory=_prompt_values
    )

    def add(self, row: Mapping[str, Any]) -> None:
        """Count one judged row's verdict."""
        verdict = verdict_from_row(row)
        self.judged += 1
        self.basis[verdict.decision_basis] += 1
        self.secondary_basis[verdict.secondary_basis or NULL_LEVEL] += 1
        self.assumption[verdict.counterpart_assumption] += 1
        self.diagonal[verdict.diagonal_comparison] += 1
        if bool(row.get("evidence_not_found")):
            self.evidence_not_found += 1
        keyword = str(row["keyword_basis"])
        if verdict.coarse_basis == keyword:
            self.keyword_agree += 1
        else:
            self.keyword_disagreements.append(
                {
                    "key": str(row["key"]),
                    "judge": verdict.decision_basis,
                    "judge_coarse": verdict.coarse_basis,
                    "keyword": keyword,
                }
            )
        prompt = self.per_prompt[str(row["prompt_id"])]
        for dimension, value in _dimension_values(verdict).items():
            prompt[dimension].append(value)

    def prompt_shares(self, dimension: str) -> dict[str, float]:
        """Per-prompt share of one dimension over that prompt's judged records in this cell."""
        return {prompt: mean(values[dimension]) for prompt, values in self.per_prompt.items()}

    def clustered(self, dimension: str) -> float | None:
        """Return the per-prompt-clustered 2SE of one dimension's pooled share."""
        return clustered_2se(
            {prompt: values[dimension] for prompt, values in self.per_prompt.items()}
        )

    def summary(self) -> dict[str, Any]:
        """Render the cell's counts, shares with bounds, prompt means and clustered bands, as JSON."""
        out: dict[str, Any] = {
            "judged": self.judged,
            "basis": dict(self.basis),
            "secondary_basis": dict(self.secondary_basis),
            "counterpart_assumption": dict(self.assumption),
            "diagonal_comparison": dict(self.diagonal),
            "evidence_not_found": self.evidence_not_found,
            "n_prompts": len(self.per_prompt),
            "keyword_agreement": {
                "agree": self.keyword_agree,
                "n": self.judged,
                "rate": self.keyword_agree / self.judged if self.judged else None,
            },
            "keyword_disagreements": self.keyword_disagreements,
        }
        for dimension in BOUNDED_DIMENSIONS:
            k = int(sum(sum(values[dimension]) for values in self.per_prompt.values()))
            out[dimension] = {
                "k": k,
                "n": self.judged,
                "share": k / self.judged if self.judged else None,
                **binomial_bounds(k, self.judged),
            }
        out["prompt_mean"] = {}
        out["clustered_2se"] = {}
        for dimension in DIMENSIONS:
            shares = self.prompt_shares(dimension)
            out["prompt_mean"][dimension] = mean(shares.values()) if shares else None
            out["clustered_2se"][dimension] = self.clustered(dimension)
        return out


def sample_row_key(row: Mapping[str, Any]) -> str:
    """Build the hand-label pass's record key for one exported sample row.

    The two exports spell the same fields differently, and the hand-label file follows each export's
    own spelling: the arm is the ladder export's ``rung`` or the control export's ``source_arm``, the
    cell is the export's own ``cell`` field, and the print order is ``label_print_order`` in the
    ladder export and ``order`` in the control export, which has no ``label_print_order``.
    """
    arm = row["rung"] if "rung" in row else row["source_arm"]
    order = row["label_print_order"] if "label_print_order" in row else row["order"]
    return f"{arm}|{row['cell']}|{row['step']}|{row['prompt_id']}|{row['sample_index']}|{order}"


def _row_handle(row: Mapping[str, Any], digest: str) -> Handle:
    return (int(row["step"]), str(row["prompt_id"]), int(row["sample_index"]), digest)


def load_sample_handles(paths: Sequence[Path], *, only_keys: set[str] | None = None) -> set[Handle]:
    """Load the exported sample rows as (step, prompt_id, sample_index, sha256(completion)) handles.

    The sample files carry no arm label in the judged rows' spelling and disagree with each other about
    the print-order field's name, so the overlay matches on the completion's digest beside the three
    identity fields every row carries; the digest is what a judged row stores. ``only_keys`` keeps
    only the rows whose :func:`sample_row_key` is listed, which is how the hand-labelled calibration
    subset is selected out of a full export.
    """
    handles: set[Handle] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if only_keys is not None and sample_row_key(row) not in only_keys:
                    continue
                digest = hashlib.sha256(str(row["completion"]).encode("utf-8")).hexdigest()
                handles.add(_row_handle(row, digest))
    return handles


def _judged_handle(row: Mapping[str, Any]) -> Handle:
    return _row_handle(row, str(row["completion_sha256"]))


def decision_of(row: Mapping[str, Any]) -> str:
    """Read a stored census or judged row's decision; a row from before the stamp existed is a cooperator.

    Every row written before 2026-09-03 came from a loader that admitted cooperators only, so the
    absent field means ``cooperate``. A value outside :data:`DECISIONS` refuses: a stored row carries
    the decision the record took, never a selector such as ``all``.
    """
    decision = row.get(DECISION_FIELD, DECISION_COOPERATE)
    if decision not in DECISIONS:
        raise ValueError(
            f"row {row.get('key')!r} carries decision={decision!r}; a stored row's decision is one of "
            f"{DECISIONS}, the decision the record took, never a selector"
        )
    return str(decision)


def scope_label(row: Mapping[str, Any]) -> str:
    """Name the reading scope of one record: ``section::game::framing::variant``, plus ``::defect`` for a defector.

    Framings and decisions never pool. A cooperator's scope carries no marker because every rates file
    written before the decision selector existed used the bare label, and those labels must not move.
    """
    framing = row.get("counterpart_framing")
    framing_label = NO_FRAMING if framing is None else str(framing)
    label = f"{row['section']}::{row['game_id']}::{framing_label}::{row['payoff_variant']}"
    decision = decision_of(row)
    return label if decision == DECISION_COOPERATE else f"{label}::{decision}"


def _warn_if_unstamped(census: Sequence[Mapping[str, Any]]) -> None:
    """Name the census rows that predate the decision stamp; they are read as cooperators."""
    unstamped = sum(1 for row in census if DECISION_FIELD not in row)
    if unstamped:
        logger.warning(
            "%d of %d census rows carry no decision stamp and are read as cooperators, the only "
            "decision the loader admitted before 2026-09-03; a defector census written by the scratch "
            "driver needs re-stamping (keyword --decision over the same tree) before its labels say so",
            unstamped,
            len(census),
        )


def _refuse_a_decision_mismatch(
    census_row: Mapping[str, Any], judged_row: Mapping[str, Any]
) -> None:
    """Refuse a judged row stamped with another decision than its census row: it is about another record."""
    if DECISION_FIELD not in judged_row:
        return
    judged_decision = decision_of(judged_row)
    census_decision = decision_of(census_row)
    if judged_decision != census_decision:
        raise ValueError(
            f"judged row {census_row['key']!r} is stamped decision={judged_decision!r} but its census "
            f"row is {census_decision!r}; the tree changed under the key, so the verdict is about a "
            f"record this census does not hold"
        )


def _cell_label(row: Mapping[str, Any]) -> str:
    return f"{row['arm']}|{row['cell']}|{scope_label(row)}@{row['step']}"


def coupling_clause_in_prompt(row: Mapping[str, Any]) -> bool:
    """Whether one record's prompt asserted the counterpart coupling.

    The framing's clause where the row carries a framing (the sweep swaps the game's own paragraph
    out), else the game's own paragraph as the arm table renders it.

    A framing the tracked registry does not carry came from the run's `--framings-file`, whose clause
    no reader of a trace can see. It reads decoupled on the strength of the loader's own refusal:
    `games.framing_stimulus.load_framings` rejects any runtime clause carrying a coupling assertion,
    so a loaded framing is decoupled by construction, and `framing_stimulus.framing_states_coupling`
    answers False for the same reason. What licenses that answer HERE is the census row's
    `framings_digest`, stamped off the cell's meta: with a digest the cell really did load a file, and
    without one the id names no clause anywhere, so it is a typo and the answer is unknown rather
    than False. A typo read as decoupled would put a whole cell on the wrong side of the control's
    "invented coupling story" reading with every count still adding up.
    """
    framing = row.get("counterpart_framing")
    if framing is None:
        return has_coupling_clause(str(row["game_id"]))
    framing_id = str(framing)
    if framing_id in COUNTERPART_FRAMINGS:
        return framing_states_coupling(framing_id)
    if row.get(FRAMINGS_DIGEST_FIELD):
        return False
    raise ValueError(
        f"row {row.get('key')!r} is stamped counterpart_framing={framing_id!r}, which is neither a "
        f"registered framing ({list(COUNTERPART_FRAMINGS)}) nor covered by a runtime framings file: "
        f"its census row carries no {FRAMINGS_DIGEST_FIELD}, so the cell that produced it recorded "
        f"none. Whether that framing's clause stated the coupling is unknown, not False. Re-stamp "
        f"the census from a tree whose cell meta carries eval_config.framings_digest, or fix the id."
    )


def _delta_for(
    dimension: str, low: CellTally, high: CellTally, steps: tuple[int, int]
) -> dict[str, Any]:
    """One dimension's within-arm move between the two steps, in the harnesses' band conventions."""
    shares_low = low.prompt_shares(dimension)
    shares_high = high.prompt_shares(dimension)
    shared = sorted(set(shares_low) & set(shares_high))
    unpaired = len(set(shares_low) ^ set(shares_high))
    level_low = mean(shares_low.values()) if shares_low else None
    level_high = mean(shares_high.values()) if shares_high else None
    deltas = [shares_high[prompt] - shares_low[prompt] for prompt in shared]
    quadrature = _quadrature(
        _between_prompt_2se(list(shares_low.values())),
        _between_prompt_2se(list(shares_high.values())),
    )
    if len(deltas) >= _MIN_PROMPTS_FOR_A_BAND:
        delta: float | None = mean(deltas)
        paired: float | None = _between_prompt_2se(deltas)
        method = "paired-between-prompt"
    else:
        delta = level_high - level_low if level_low is not None and level_high is not None else None
        paired = None
        method = "quadrature-unpaired" if quadrature is not None else "insufficient-prompts"
    return {
        "delta": delta,
        "band_method": method,
        "paired_2se": paired,
        "quadrature_2se": quadrature,
        "clustered_2se": _quadrature(low.clustered(dimension), high.clustered(dimension)),
        "n_paired": len(deltas),
        "n_unpaired": unpaired,
        "level": {str(steps[0]): level_low, str(steps[1]): level_high},
        "n_prompts": {str(steps[0]): len(shares_low), str(steps[1]): len(shares_high)},
    }


def _deltas(
    tallies: Mapping[str, CellTally], steps_of: Mapping[str, tuple[str, int]]
) -> dict[str, Any]:
    """Within-arm deltas for every (arm, cell, scope) with exactly two steps; any other count is named."""
    by_scope: dict[str, dict[int, CellTally]] = defaultdict(dict)
    for label, tally in tallies.items():
        scope, step = steps_of[label]
        by_scope[scope][step] = tally
    deltas: dict[str, Any] = {}
    for scope, by_step in sorted(by_scope.items()):
        if len(by_step) != _STEPS_IN_A_DELTA:
            logger.warning(
                "no delta for %s: a delta is between exactly two steps and this scope has %d (%s)",
                scope,
                len(by_step),
                sorted(by_step),
            )
            continue
        low_step, high_step = sorted(by_step)
        deltas[scope] = {
            "steps": [low_step, high_step],
            "dimensions": {
                dimension: _delta_for(
                    dimension, by_step[low_step], by_step[high_step], (low_step, high_step)
                )
                for dimension in DIMENSIONS
            },
        }
    return deltas


def trace_rates(
    judged: Mapping[str, Mapping[str, Any]],
    census: Sequence[Mapping[str, Any]],
    *,
    drawn: set[Handle] | None = None,
) -> dict[str, Any]:
    """Per-scope-cell shares, per-prompt shares and within-arm deltas from judged rows over the census.

    Denominators: ``examined`` counts every census record of the cell; ``skipped_empty`` those with no
    text; ``not_judged`` judgeable records with no judged row yet; ``errored`` rows with no verdict after
    retry; ``judged`` the rest, over which every share is computed. The drawn overlay, when given, is
    the same tally restricted to the hand-read sample, reported beside the census under ``drawn``;
    unmatched drawn rows are counted at the top so a mismatch is visible.
    """
    census_keys = {str(row["key"]) for row in census}
    missing = sorted(key for key in judged if key not in census_keys)
    if missing:
        examples = missing[:3]  # HARNESS-SCAN-EXEMPT-subsampling: error-message examples, not data
        raise ValueError(f"{len(missing)} judged keys are not in the census, e.g. {examples}")
    _warn_if_unstamped(census)
    cells: dict[str, dict[str, int]] = defaultdict(
        lambda: {"examined": 0, "skipped_empty": 0, "not_judged": 0, "errored": 0}
    )
    stamps: dict[str, bool] = {}
    steps_of: dict[str, tuple[str, int]] = {}
    tallies: dict[str, CellTally] = defaultdict(CellTally)
    drawn_tallies: dict[str, CellTally] = defaultdict(CellTally)
    matched_drawn: set[Handle] = set()
    for record in census:
        label = _cell_label(record)
        steps_of[label] = (
            f"{record['arm']}|{record['cell']}|{scope_label(record)}",
            int(record["step"]),
        )
        # One game and one framing per label by construction, so the stamp cannot disagree within it.
        stamps[label] = coupling_clause_in_prompt(record)
        counts = cells[label]
        counts["examined"] += 1
        if not bool(record["judgeable"]):
            counts["skipped_empty"] += 1
            continue
        row = judged.get(str(record["key"]))
        if row is None:
            counts["not_judged"] += 1
            continue
        _refuse_a_decision_mismatch(record, row)
        if "verdict" not in row:
            counts["errored"] += 1
        else:
            tallies[label].add(row)
            if drawn is not None and _judged_handle(row) in drawn:
                matched_drawn.add(_judged_handle(row))
                drawn_tallies[label].add(row)
    out_cells: dict[str, Any] = {}
    prompts: dict[str, Any] = {}
    for label in sorted(cells):
        tally = tallies[label]
        summary = {
            **cells[label],
            "coupling_clause_in_prompt": stamps[label],
            **tally.summary(),
        }
        if drawn is not None:
            summary["drawn"] = drawn_tallies[label].summary()
        out_cells[label] = summary
        prompts[label] = {
            prompt: {
                "n": len(values["coupling_story"]),
                **{dim: mean(values[dim]) for dim in DIMENSIONS},
            }
            for prompt, values in sorted(tally.per_prompt.items())
        }
    rates: dict[str, Any] = {
        "cells": out_cells,
        "prompts": prompts,
        "deltas": _deltas(tallies, steps_of),
        "dimensions": list(DIMENSIONS),
    }
    if drawn is not None:
        rates["drawn_rows_unmatched"] = len(drawn) - len(matched_drawn)
    return rates


def load_hand_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Load a hand-label file: ``{"comment": ..., "labels": {key: {...}}}`` or the bare mapping.

    The calibration file is written as record key -> labelled dimensions plus metadata; the
    ``labels`` envelope is the layout of the earlier cot-rescore file, so both are accepted. Top-level
    fields beside ``labels`` (``rubric_version``, ``rubric_notes``, ``draw_rule``, ...) are metadata.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must hold a JSON object, got {type(payload).__name__}")
    labels = (
        payload["labels"]
        if "labels" in payload
        else {k: v for k, v in payload.items() if k != "comment"}
    )
    if not isinstance(labels, dict) or not all(
        isinstance(value, dict) for value in labels.values()
    ):
        raise TypeError(f"{path} must map record keys to label objects")
    return labels


def hand_label_sample_files(
    labels: Mapping[str, Mapping[str, Any]], *, relative_to: Path
) -> list[Path]:
    """List the sample exports the hand labels were drawn from, read off their ``source_file`` fields."""
    names = sorted(
        {str(label["source_file"]) for label in labels.values() if "source_file" in label}
    )
    return [relative_to / name for name in names]


_HAND_LABEL_ENUMS: dict[str, tuple[str, ...]] = {
    "decision_basis": DECISION_BASES,
    "secondary_basis": DECISION_BASES,
    "counterpart_assumption": COUNTERPART_ASSUMPTIONS,
    "diagonal_comparison": DIAGONAL_COMPARISON_LEVELS,
}
_NULLABLE_HAND_LABELS: frozenset[str] = frozenset({"secondary_basis"})
_HAND_LABEL_METADATA: frozenset[str] = frozenset(
    {
        "note",
        "evidence",
        "source_file",
        "arm",
        "cell",
        "step",
        "blind_id",
        "prompt_id",
        "sample_index",
        "label_print_order",
        "payoff_variant",
        "parsed_action",
        "completion_chars",
    }
)
_PREVIOUS_RUBRIC_PREFIX = "v1_"
SCORED_DIMENSIONS: tuple[str, ...] = (*_HAND_LABEL_ENUMS, *VERDICT_BOOLS)
"""The hand-label fields validate scores; ``coupling_story`` is derived from ``counterpart_assumption``."""

DOMINANCE_CAVEAT = (
    "dominance_raised and dominance_rejected coincide on a cooperator-only set by construction and "
    "equal payoff_reasoning_present in the hand set; agreement here is not evidence the judge separates "
    "them (they separate on defectors and on arithmetic favouring cooperation)"
)


def _is_metadata(dimension: str) -> bool:
    return dimension in _HAND_LABEL_METADATA or dimension.startswith(_PREVIOUS_RUBRIC_PREFIX)


def _hand_value(key: str, dimension: str, value: object) -> object:
    """Check one hand label against the schema, refusing a typo rather than scoring it as a miss."""
    if dimension in _HAND_LABEL_ENUMS:
        if value is None and dimension in _NULLABLE_HAND_LABELS:
            return None
        if value not in _HAND_LABEL_ENUMS[dimension]:
            raise ValueError(
                f"{key}: hand {dimension}={value!r} is not one of {_HAND_LABEL_ENUMS[dimension]}"
            )
        return value
    if dimension in VERDICT_BOOLS:
        if not isinstance(value, bool):
            raise ValueError(f"{key}: hand {dimension}={value!r} is not a bool")
        return value
    raise ValueError(
        f"{key}: unknown hand-label dimension {dimension!r}; scored dimensions: {SCORED_DIMENSIONS}"
    )


@dataclass(slots=True)
class DimensionScore:
    """Judge-vs-hand agreement for one dimension: strict and lenient counts, confusion, every miss."""

    n: int = 0
    agree: int = 0
    lenient_agree: int = 0
    confusion: Counter[str] = field(default_factory=Counter)
    misses: list[str] = field(default_factory=list)
    hand_levels: Counter[str] = field(default_factory=Counter)

    def score(self, key: str, expected: object, got: object, *, lenient: bool) -> bool:
        """Count one pair; ``lenient`` is the extra credit rule's verdict for this pair. Returns strict."""
        self.n += 1
        self.confusion[f"hand={expected}|judge={got}"] += 1
        self.hand_levels[str(expected)] += 1
        strict = expected == got
        if strict:
            self.agree += 1
        else:
            self.misses.append(key)
        if strict or lenient:
            self.lenient_agree += 1
        return strict

    def summary(self, levels: tuple[str, ...] | None) -> dict[str, Any]:
        """Render the dimension's lines; enum levels with no hand positives are named, never 0/0."""
        out: dict[str, Any] = {
            "n": self.n,
            "agree": self.agree,
            "rate": self.agree / self.n if self.n else None,
            "lenient_agree": self.lenient_agree,
            "lenient_rate": self.lenient_agree / self.n if self.n else None,
            "confusion": dict(self.confusion),
            "misses": self.misses,
        }
        if levels is not None:
            out["levels_without_hand_positives"] = [
                level for level in levels if not self.hand_levels[level]
            ]
        return out


def _lenient_basis(hand: Mapping[str, Any], verdict: JudgeVerdict) -> bool:
    """Apply the lenient rule: judge primary in the hand pair, or judge secondary equals hand primary."""
    hand_primary = hand.get("decision_basis")
    hand_secondary = hand.get("secondary_basis")
    return verdict.decision_basis in (hand_primary, hand_secondary) or (
        verdict.secondary_basis is not None and verdict.secondary_basis == hand_primary
    )


def _label_pairs(
    key: str, label: Mapping[str, Any], verdict: JudgeVerdict
) -> list[tuple[str, object, object, bool]]:
    """(dimension, hand, judge, lenient) rows for one record, the derived coupling dimension included."""
    judged_values = {**asdict(verdict), "coupling_story": verdict.coupling_story}
    pairs: list[tuple[str, object, object, bool]] = []
    for dimension, raw in label.items():
        if _is_metadata(dimension):
            continue
        expected = _hand_value(key, dimension, raw)
        lenient = _lenient_basis(label, verdict) if dimension == "decision_basis" else False
        pairs.append((dimension, expected, judged_values[dimension], lenient))
        if dimension == "counterpart_assumption":
            pairs.append(
                ("coupling_story", expected in COUPLING_ASSUMPTIONS, verdict.coupling_story, False)
            )
    return pairs


def resolve_hand_keys(
    labels: Mapping[str, Mapping[str, Any]],
    judged: Mapping[str, Mapping[str, Any]],
    sample_paths: Sequence[Path],
) -> dict[str, str]:
    """Map every hand-label key to a judged-row key, through the sample exports where the spellings differ.

    A hand key that is itself a judged key resolves directly. Otherwise it is the sample exports'
    spelling (:func:`sample_row_key`), and the sample row's completion digest finds the judged row,
    because the digest is the one identity both files share. Unresolved keys raise: a calibration set
    has to be judged before it can calibrate anything.
    """
    by_handle = {_judged_handle(row): key for key, row in judged.items() if "verdict" in row}
    sample_key_to_handle: dict[str, Handle] = {}
    for path in sample_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                digest = hashlib.sha256(str(row["completion"]).encode("utf-8")).hexdigest()
                sample_key_to_handle[sample_row_key(row)] = _row_handle(row, digest)
    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    for key in labels:
        if key in judged and "verdict" in judged[key]:
            resolved[key] = key
            continue
        handle = sample_key_to_handle.get(key)
        judged_key = by_handle.get(handle) if handle is not None else None
        if judged_key is None:
            unresolved.append(key)
        else:
            resolved[key] = judged_key
    if unresolved:
        raise ValueError(
            f"{len(unresolved)} hand-labelled records have no judged verdict (judge them first, e.g. with "
            f"--only-hand-labelled): {unresolved}"
        )
    return resolved


def validation_report(
    judged: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    *,
    key_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Confusion per labelled dimension with strict and lenient lines, plus exact agreement.

    Dimensions are scored only where labelled, so partial labels never count as agreement, and a label
    that scores NO dimension (empty, or metadata only) refuses rather than counting as exact agreement
    with nothing. The derived ``coupling_story`` dimension is scored whenever ``counterpart_assumption``
    is, because it is the control clause's actual reading. Exact agreement is strict on every labelled
    dimension. ``key_map`` translates hand keys to judged keys (:func:`resolve_hand_keys`); without it
    the keys must match.
    """
    dims: dict[str, DimensionScore] = {}
    exact_agree = 0
    for key, label in labels.items():
        judged_key = key_map[key] if key_map is not None else key
        row = judged.get(judged_key)
        if row is None or "verdict" not in row:
            raise ValueError(f"hand-labelled record {key} has no judged verdict")
        pairs = _label_pairs(key, label, verdict_from_row(row))
        if not pairs:
            raise ValueError(
                f"hand label {key} scores no dimension (fields: {sorted(label)}); an unlabelled record "
                f"cannot count as agreement"
            )
        all_agree = True
        for dimension, expected, got, lenient in pairs:
            strict = dims.setdefault(dimension, DimensionScore()).score(
                key, expected, got, lenient=lenient
            )
            all_agree = all_agree and strict
        if all_agree:
            exact_agree += 1
    report: dict[str, Any] = {"dimensions": {}}
    for dimension, score in dims.items():
        entry = score.summary(_HAND_LABEL_ENUMS.get(dimension))
        if dimension in ("dominance_raised", "dominance_rejected"):
            entry["caveat"] = DOMINANCE_CAVEAT
        report["dimensions"][dimension] = entry
    report["exact_agreement"] = {
        "agree": exact_agree,
        "n": len(labels),
        "rate": exact_agree / len(labels) if labels else None,
    }
    return report
