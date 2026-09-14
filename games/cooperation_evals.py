"""Small, explicit evaluation plans for the cooperation-generalisation experiment.

The experiment manifest is deliberately runtime data.  It contains scenario and payoff identifiers,
never prompt text, because this repository is public and these rows will be run on future models.
This module supplies the validation and expansion seam used by the sequence runner, while reusing
the established game renderers, parsers, survey records, and per-record resume implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from reward_hacking.model_backend import Backend

from games.cooperation_payoffs import (
    annotate_behavior_record,
    rank_allocation_options,
    rank_row_actions,
)
from games.evals import (
    ADMISSION_LONGEST_FIRST,
    EVAL_ONLY_GRADING,
    SECTION_DT_PROBES,
    SECTION_GAME_BEHAVIOR,
    SECTION_SELF_REPORT,
    SUBMISSION_POOLED,
    EvalConfig,
    PlannedRequest,
    _game_request_record,  # pyright: ignore[reportPrivateUsage]
    plan_battery,
    run_eval_battery,
)
from games.parsing import strip_thinking
from games.payoffs import COOPERATE, DEFECT
from games.probes import OPEN_ENDED_ITEMS, counterbalanced_option_orders, our_battery
from games.prompts import (
    ALL_GAME_IDS,
    COUNTERPART_FRAMING_IDS,
    EVAL_ONLY_GAME_IDS,
    FRAMEABLE_GAME_IDS,
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDER_SWAPPED,
    LABEL_PRINT_ORDERS,
    SPLIT_EVAL,
    UNLABELLED_GAME_IDS,
    generate_framing_prompt_rows,
    generate_prompt_rows,
)
from games.rewards import is_grading
from games.survey import (
    FAMILY_COMPETITIVENESS,
    FAMILY_NARCISSISM,
    FAMILY_NEGATIVE_CONTROL,
    FAMILY_PROSOCIALNESS,
    FAMILY_SELF_PREDICTION,
    FAMILY_SVO_ALLOCATION,
    FAMILY_TRIPLE_DOMINANCE,
    INSTRUMENT_PROSOCIALNESS,
    PUBLISHED_INSTRUMENTS,
    SURVEY_ALLOCATION,
    SURVEY_CHOICE,
    SURVEY_NUMERIC,
    SurveyItem,
    battery_orders,
    numeric_example_rotation,
    parse_survey_answer,
    render_survey_prompt,
    survey_battery,
    survey_record_fields,
)

FAMILY_SOCIAL_DILEMMA = "social-dilemma"
FAMILY_COORDINATION = "coordination"
FAMILY_NUMERIC_GIVING = "numeric-giving"
DIAGNOSTIC_FAMILIES: tuple[str, ...] = (
    FAMILY_SOCIAL_DILEMMA,
    FAMILY_COORDINATION,
    FAMILY_NUMERIC_GIVING,
)

INSTRUMENT_CONTEXT_SELF_PREDICTION = "self-prediction-context"
INSTRUMENT_NORMATIVE_PAYOFF = "normative-payoff-choice"
BEHAVIOR_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_RUNTIME_DIR = Path("docs/scratch/cooperation-generalization")
DEFAULT_BEHAVIOR_MANIFEST_PATH = DEFAULT_RUNTIME_DIR / "behavior-roster.json"
DEFAULT_TRAINING_CORPUS_PATH = DEFAULT_RUNTIME_DIR / "training.jsonl"
EXACT_DECISION_PROMPT_BEGIN = "<embedded-decision-prompt>"
EXACT_DECISION_PROMPT_END = "</embedded-decision-prompt>"
DEFAULT_REQUIRED_SAMPLES = 2
PAYOFF_DIMENSIONS = 2
ALLOCATION_OPTION_COUNT = 3
ALLOCATION_GROUP_PREFIX = "allocation"
ALLOCATION_OPTION_ORDER_NAMES: tuple[str, ...] = (
    "as-authored",
    "rotate-left",
    "rotate-right",
)
MIN_EXPERIMENT_PAIRS = 10
MAX_EXPERIMENT_PAIRS = 14
MIN_CONTRAST_MEMBERS = 2
EXPECTED_CORE_ITEMS = 65
EXPECTED_CORE_RESPONSES = 258
EXPECTED_PROSOCIALNESS_ITEMS = 32
EXPECTED_PROSOCIALNESS_RESPONSES = 128
EXPECTED_LOCAL_DT_ITEMS = 30
EXPECTED_LOCAL_DT_MULTIPLE_CHOICE = 27
EXPECTED_LOCAL_DT_OPEN_ENDED = 3


@dataclass(frozen=True, slots=True)
class NumericForecastEvent:
    """Machine-readable event whose frequency a numeric forecast asks about."""

    game_id: str
    event_id: str
    predicate: str

    def observe(self, row: Mapping[str, Any], record: Mapping[str, Any]) -> bool | None:
        """Evaluate this event on one independently parsed behavior record."""
        if str(row["game_id"]) != self.game_id or not bool(record.get("parsed")):
            return None
        if self.game_id == "dictator":
            kept = record.get("kept")
            endowment = row.get("endowment")
            return (
                None
                if not isinstance(kept, int) or not isinstance(endowment, int)
                else kept < endowment
            )
        if self.game_id == "nash-demand":
            claim = record.get("claim")
            windfall = row.get("windfall")
            return (
                None
                if not isinstance(claim, int) or not isinstance(windfall, int)
                else 2 * claim <= windfall
            )
        if self.game_id in {"trust-vs-stated-return", "trust-strategy-method"}:
            sent = record.get("sent")
            return None if not isinstance(sent, int) else sent > 0
        if self.game_id == "trustee-return-rule":
            return_fraction = record.get("return_fraction")
            return (
                None
                if not isinstance(return_fraction, int | float)
                else float(return_fraction) > 0.0
            )
        raise ValueError(f"numeric forecast event has no observer for {self.game_id!r}.")


NUMERIC_FORECAST_EVENTS: dict[str, NumericForecastEvent] = {
    "dictator": NumericForecastEvent(
        game_id="dictator",
        event_id="numeric.dictator.give-positive",
        predicate="kept < row.endowment",
    ),
    "nash-demand": NumericForecastEvent(
        game_id="nash-demand",
        event_id="numeric.nash-demand.claim-at-most-half",
        predicate="2 * claim <= row.windfall",
    ),
    "trust-vs-stated-return": NumericForecastEvent(
        game_id="trust-vs-stated-return",
        event_id="numeric.trust-sender.send-positive",
        predicate="sent > 0",
    ),
    "trust-strategy-method": NumericForecastEvent(
        game_id="trust-strategy-method",
        event_id="numeric.trust-sender.send-positive",
        predicate="sent > 0",
    ),
    "trustee-return-rule": NumericForecastEvent(
        game_id="trustee-return-rule",
        event_id="numeric.trustee.return-positive",
        predicate="return_fraction > 0",
    ),
}


@dataclass(frozen=True, slots=True)
class ContextElicitation:
    """Runtime-only prose for the additional context items."""

    forecast_template: str
    normative_choice_template: str
    normative_numeric_template: str
    matrix_target_template: str
    numeric_targets: Mapping[str, str]
    normative_target: str
    normative_dimension: str
    construct: str
    expected_direction: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ContextElicitation:
        """Parse the private runtime elicitation block."""
        required = (
            "forecast_template",
            "normative_choice_template",
            "normative_numeric_template",
            "matrix_target_template",
            "numeric_targets",
            "normative_target",
            "normative_dimension",
            "construct",
            "expected_direction",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"context elicitation is missing fields {missing}.")
        targets = payload["numeric_targets"]
        if not isinstance(targets, dict):
            raise TypeError("context elicitation numeric_targets must be an object.")
        text_values = {
            name: _required_text(payload[name], f"context elicitation {name}")
            for name in required
            if name != "numeric_targets"
        }
        numeric_targets = {
            _required_text(game_id, "context elicitation numeric target game"): _required_text(
                target, "context elicitation numeric target"
            )
            for game_id, target in targets.items()
        }
        elicitation = cls(numeric_targets=numeric_targets, **text_values)
        if elicitation.normative_dimension not in {"reward", "own", "welfare", "equality"}:
            raise ValueError(
                "context elicitation normative_dimension must be one of reward, own, welfare, "
                f"or equality; got {elicitation.normative_dimension!r}."
            )
        return elicitation

    def validate_for_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Check that every numeric row has a target before rendering any request."""
        numeric_games = {
            str(row["game_id"])
            for row in rows
            if not bool(row.get("label_a") or row.get("label_b"))
        }
        missing = sorted(numeric_games - set(self.numeric_targets))
        if missing:
            raise ValueError(
                f"context elicitation has no numeric target text for games {missing}; add it to "
                "the private runtime manifest before generation."
            )


@dataclass(frozen=True, slots=True)
class AllocationDiagnostic:
    """One private three-option allocation item with separate payoff-dimension maxima."""

    diagnostic_id: str
    stem: str
    option_payoffs: tuple[tuple[int, int], ...]
    samples: int = 2

    @property
    def scenario_group_id(self) -> str:
        """Return the isolated group identifier reserved for this diagnostic."""
        return f"{ALLOCATION_GROUP_PREFIX}::{self.diagnostic_id}"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AllocationDiagnostic:
        """Parse one private allocation diagnostic."""
        required = ("diagnostic_id", "stem", "option_payoffs")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"allocation diagnostic is missing fields {missing}.")
        raw_payoffs = payload["option_payoffs"]
        if not isinstance(raw_payoffs, list | tuple):
            raise TypeError("allocation diagnostic option_payoffs must be a list.")
        payoffs: tuple[tuple[int, int], ...] = tuple(
            _allocation_payoff_pair(pair) for pair in raw_payoffs
        )
        diagnostic = cls(
            diagnostic_id=_required_text(payload["diagnostic_id"], "allocation diagnostic_id"),
            stem=_required_text(payload["stem"], "allocation diagnostic stem"),
            option_payoffs=payoffs,
            samples=_positive_int(payload.get("samples", 2), "allocation samples"),
        )
        diagnostic.validate()
        return diagnostic

    def validate(self) -> None:
        """Require exactly three options with distinct own, welfare, and equality winners."""
        if len(self.option_payoffs) != ALLOCATION_OPTION_COUNT:
            raise ValueError(
                f"allocation diagnostic {self.diagnostic_id!r} needs exactly three options; "
                f"got {len(self.option_payoffs)}."
            )
        ranked = rank_allocation_options(
            {str(index): pair for index, pair in enumerate(self.option_payoffs)}
        )
        optima = {
            "own": ranked["own_optimal_actions"],
            "total": ranked["welfare_optimal_actions"],
            "equality": ranked["equality_optimal_actions"],
        }
        if (
            any(len(actions) != 1 for actions in optima.values())
            or len({actions[0] for actions in optima.values()}) != ALLOCATION_OPTION_COUNT
        ):
            raise ValueError(
                f"allocation diagnostic {self.diagnostic_id!r} does not separate own, total, "
                f"and equality optima: {optima}."
            )


def _allocation_payoff_pair(value: Any) -> tuple[int, int]:  # noqa: ANN401
    if not isinstance(value, list | tuple) or len(value) != PAYOFF_DIMENSIONS:
        raise ValueError(f"allocation option must be a [self, counterpart] pair, got {value!r}.")
    if any(isinstance(part, bool) or not isinstance(part, int) for part in value):
        raise TypeError(f"allocation option payoffs must be integers, got {value!r}.")
    return int(value[0]), int(value[1])


def allocation_option_orders(
    n_options: int,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return the three cyclic presentations for the private three-option diagnostic."""
    if n_options != ALLOCATION_OPTION_COUNT:
        raise ValueError(
            f"the allocation balance contract is frozen at exactly {ALLOCATION_OPTION_COUNT} "
            f"options, got {n_options}."
        )
    canonical = tuple(range(n_options))
    return tuple(
        (name, canonical[offset:] + canonical[:offset])
        for name, offset in zip(ALLOCATION_OPTION_ORDER_NAMES, (0, 1, 2), strict=True)
    )


@dataclass(frozen=True, slots=True)
class DiagnosticPair:
    """One row family in the private behavior roster.

    A pair means one conceptual scenario/payoff cell.  Labelled cells expand to both canonical
    action mappings and both printed orders; numeric cells expand to one rendering.  The explicit
    ``scenario_group`` is also the unit that interpretability fitting and held-out evaluation must
    keep intact.
    """

    pair_id: str
    family: str
    game_id: str
    grading: str
    scenario_id: str
    payoff_variant: str
    scenario_group: str
    label_print_orders: tuple[str, ...]
    samples: int = 2
    contrast_group: str | None = None
    contrast_kind: str | None = None
    counterpart_framing: str | None = None
    held_out_counterpart_identity: str | None = None
    held_out_combination: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DiagnosticPair:
        """Parse and validate one private diagnostic-pair declaration."""
        required = (
            "pair_id",
            "family",
            "game_id",
            "grading",
            "scenario_id",
            "payoff_variant",
            "scenario_group",
            "label_print_orders",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"behavior roster pair is missing fields {missing}.")
        values = {name: payload[name] for name in required}
        orders_value = values["label_print_orders"]
        if not isinstance(orders_value, list | tuple):
            raise TypeError("label_print_orders must be a list of printed-order identifiers.")
        pair = cls(
            pair_id=_required_text(values["pair_id"], "pair_id"),
            family=_required_text(values["family"], "family"),
            game_id=_required_text(values["game_id"], "game_id"),
            grading=_required_text(values["grading"], "grading"),
            scenario_id=_required_text(values["scenario_id"], "scenario_id"),
            payoff_variant=_required_text(values["payoff_variant"], "payoff_variant"),
            scenario_group=_required_text(values["scenario_group"], "scenario_group"),
            label_print_orders=tuple(
                _required_text(value, "label_print_orders entry") for value in orders_value
            ),
            samples=_positive_int(payload.get("samples", 2), "samples"),
            contrast_group=(
                None
                if payload.get("contrast_group") is None
                else _required_text(payload["contrast_group"], "contrast_group")
            ),
            contrast_kind=(
                None
                if payload.get("contrast_kind") is None
                else _required_text(payload["contrast_kind"], "contrast_kind")
            ),
            counterpart_framing=(
                None
                if payload.get("counterpart_framing") is None
                else _required_text(payload["counterpart_framing"], "counterpart_framing")
            ),
            held_out_counterpart_identity=(
                None
                if payload.get("held_out_counterpart_identity") is None
                else _required_text(
                    payload["held_out_counterpart_identity"],
                    "held_out_counterpart_identity",
                )
            ),
            held_out_combination=(
                None
                if payload.get("held_out_combination") is None
                else _required_text(payload["held_out_combination"], "held_out_combination")
            ),
        )
        pair.validate()
        return pair

    def validate(self) -> None:  # noqa: C901, PLR0912
        """Validate family, renderer, grading, and held-out metadata contracts."""
        if self.family not in DIAGNOSTIC_FAMILIES:
            raise ValueError(
                f"{self.pair_id!r} uses unknown diagnostic family {self.family!r}; "
                f"expected {list(DIAGNOSTIC_FAMILIES)}."
            )
        allowed_games = {
            FAMILY_SOCIAL_DILEMMA: {
                "twin-pd",
                "fixed-pie-pd",
                "pd-unstated",
                "pd-reskin",
                "pd-vs-frozen",
                "public-goods",
                "iterated-pd-tft",
                "iterated-pd-grim",
            },
            FAMILY_COORDINATION: {
                "stag-hunt",
                "stag-hunt-vs-frozen",
                "hi-lo",
                "harmony",
                "chicken",
                "defective-coordination",
                "defective-harmony",
            },
            FAMILY_NUMERIC_GIVING: {
                "dictator",
                "nash-demand",
                "threshold-goods",
                "trust-vs-stated-return",
                "trust-strategy-method",
                "trustee-return-rule",
                "min-effort",
                "iterated-min-effort-matcher",
            },
        }
        if self.game_id not in allowed_games[self.family]:
            raise ValueError(
                f"{self.pair_id!r} assigns game {self.game_id!r} to family {self.family!r}; "
                f"valid games are {sorted(allowed_games[self.family])}."
            )
        if self.contrast_kind not in {None, "wording", "relationship", "held-out"}:
            raise ValueError(
                f"{self.pair_id!r} has unknown contrast_kind {self.contrast_kind!r}; "
                "use wording, relationship, or held-out."
            )
        if self.counterpart_framing is not None:
            if self.game_id not in FRAMEABLE_GAME_IDS:
                raise ValueError(
                    f"{self.pair_id!r} names counterpart framing for non-frameable game "
                    f"{self.game_id!r}."
                )
            if self.counterpart_framing not in COUNTERPART_FRAMING_IDS:
                raise ValueError(
                    f"{self.pair_id!r} names unknown counterpart framing "
                    f"{self.counterpart_framing!r}."
                )
        if self.contrast_kind == "held-out":
            if self.held_out_combination is None:
                raise ValueError(
                    f"{self.pair_id!r} is held-out but has no held_out_combination declaration."
                )
            if self.held_out_counterpart_identity is None:
                raise ValueError(
                    f"{self.pair_id!r} is held-out but has no held_out_counterpart_identity "
                    "declaration."
                )
            if self.counterpart_framing is not None and (
                self.held_out_counterpart_identity != self.counterpart_framing
            ):
                raise ValueError(
                    f"{self.pair_id!r} declares counterpart framing "
                    f"{self.counterpart_framing!r} but held-out identity "
                    f"{self.held_out_counterpart_identity!r}."
                )
        if self.game_id not in ALL_GAME_IDS:
            raise ValueError(f"{self.pair_id!r} names unknown game {self.game_id!r}.")
        if self.grading != EVAL_ONLY_GRADING and not is_grading(self.grading):
            raise ValueError(f"{self.pair_id!r} names unknown grading {self.grading!r}.")
        if not self.label_print_orders:
            raise ValueError(f"{self.pair_id!r} has no printed orders.")
        if len(set(self.label_print_orders)) != len(self.label_print_orders):
            raise ValueError(f"{self.pair_id!r} repeats a printed order.")
        unknown = sorted(set(self.label_print_orders) - set(LABEL_PRINT_ORDERS))
        if unknown:
            raise ValueError(f"{self.pair_id!r} names unknown printed orders {unknown}.")
        if self.game_id in UNLABELLED_GAME_IDS:
            if self.label_print_orders != (LABEL_PRINT_ORDER_CANONICAL,):
                raise ValueError(
                    f"{self.pair_id!r} is numeric and has no printed label order; use only "
                    f"{LABEL_PRINT_ORDER_CANONICAL!r}."
                )
        elif set(self.label_print_orders) != set(LABEL_PRINT_ORDERS):
            raise ValueError(
                f"{self.pair_id!r} must include both printed orders for a labelled game; "
                f"got {self.label_print_orders}."
            )


@dataclass(frozen=True, slots=True)
class BehaviorManifest:
    """Validated private roster plus runtime-only elicitation diagnostics."""

    pairs: tuple[DiagnosticPair, ...]
    schema_version: int = BEHAVIOR_MANIFEST_SCHEMA_VERSION
    elicitation: ContextElicitation | None = None
    allocation: AllocationDiagnostic | None = None

    def validate(self) -> None:
        """Validate schema, pair identity uniqueness, and optional allocation content."""
        if self.schema_version != BEHAVIOR_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported behavior roster schema {self.schema_version}; expected "
                f"{BEHAVIOR_MANIFEST_SCHEMA_VERSION}."
            )
        if not self.pairs:
            raise ValueError("behavior roster has no diagnostic pairs.")
        pair_ids = [pair.pair_id for pair in self.pairs]
        repeated = sorted(pair_id for pair_id, n in Counter(pair_ids).items() if n > 1)
        if repeated:
            raise ValueError(f"behavior roster repeats pair_id {repeated}.")
        for pair in self.pairs:
            pair.validate()
        if self.allocation is not None:
            self.allocation.validate()

    @property
    def scenario_group_ids(self) -> frozenset[str]:
        """Return the scenario groups that must remain isolated across splits."""
        groups = {pair.scenario_group for pair in self.pairs}
        if self.allocation is not None:
            groups.add(self.allocation.scenario_group_id)
        return frozenset(groups)


def _required_text(value: Any, field_name: str) -> str:  # noqa: ANN401
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _positive_int(value: Any, field_name: str) -> int:  # noqa: ANN401
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
    return value


def load_behavior_manifest(path: Path = DEFAULT_BEHAVIOR_MANIFEST_PATH) -> BehaviorManifest:
    """Load and validate a private identifier-only behavior roster."""
    if not path.exists():
        raise FileNotFoundError(
            f"behavior roster {path} is missing. It is private runtime data under "
            f"{DEFAULT_RUNTIME_DIR}; no behavior evaluation can run from an implicit broad roster."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"behavior roster {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("behavior roster JSON must be an object.")
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise TypeError("behavior roster must contain a list under 'pairs'.")
    manifest = BehaviorManifest(
        schema_version=payload.get("schema_version", BEHAVIOR_MANIFEST_SCHEMA_VERSION),
        pairs=tuple(
            DiagnosticPair.from_mapping(pair) if isinstance(pair, dict) else _invalid_pair(pair)
            for pair in raw_pairs
        ),
        elicitation=(
            None
            if payload.get("elicitation") is None
            else ContextElicitation.from_mapping(payload["elicitation"])
        ),
        allocation=(
            None
            if payload.get("allocation") is None
            else AllocationDiagnostic.from_mapping(
                payload["allocation"]
                if isinstance(payload["allocation"], dict)
                else _invalid_allocation(payload["allocation"])
            )
        ),
    )
    manifest.validate()
    return manifest


def _invalid_pair(value: Any) -> DiagnosticPair:  # noqa: ANN401
    raise ValueError(f"behavior roster pair must be an object, got {type(value).__name__}.")


def _invalid_allocation(value: Any) -> Mapping[str, Any]:  # noqa: ANN401
    raise ValueError(f"behavior roster allocation must be an object, got {type(value).__name__}.")


@dataclass(frozen=True, slots=True)
class ExpandedBehaviorRow:
    """One renderer row carrying its private roster and isolation metadata."""

    pair_id: str
    family: str
    scenario_group: str
    samples: int
    raw: Mapping[str, Any]
    contrast_group: str | None = None
    contrast_kind: str | None = None
    held_out_counterpart_identity: str | None = None
    held_out_combination: str | None = None

    @property
    def group_id(self) -> str:
        """Return the stable scenario group identifier."""
        return self.scenario_group


@dataclass(frozen=True, slots=True)
class BehaviorCounts:
    """Counts for conceptual pairs, rendered rows, samples, and category coverage."""

    n_pairs: int
    n_rendered_prompts: int
    n_completions: int
    by_family: Mapping[str, int]
    by_print_order: Mapping[str, int]
    by_pair: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class BehaviorRoster:
    """Expanded behavior rows together with their source manifest and counts."""

    rows: tuple[ExpandedBehaviorRow, ...]
    manifest: BehaviorManifest
    counts: BehaviorCounts

    @property
    def scenario_group_ids(self) -> frozenset[str]:
        """Return all scenario groups represented by this expanded roster."""
        return self.manifest.scenario_group_ids


REQUIRED_EXPERIMENT_GAMES: frozenset[str] = frozenset(
    {
        "twin-pd",
        "stag-hunt",
        "public-goods",
        "defective-harmony",
        "dictator",
        "trust-vs-stated-return",
        "trustee-return-rule",
    }
)


def _rendered_rows_for_pair(
    pair: DiagnosticPair, *, label_print_order: str
) -> Sequence[Mapping[str, Any]]:
    if pair.counterpart_framing is not None:
        return generate_framing_prompt_rows(
            pair.game_id,
            pair.grading,
            framing_id=pair.counterpart_framing,
            split=SPLIT_EVAL,
            label_print_order=label_print_order,
        )
    return generate_prompt_rows(
        pair.game_id,
        pair.grading,
        split=SPLIT_EVAL,
        label_print_order=label_print_order,
    )


def validate_experiment_coverage(roster: BehaviorRoster) -> None:  # noqa: C901, PLR0912
    """Validate the frozen MVP preset's required diagnostic families and contrasts.

    ``expand_behavior_roster`` remains useful for small fixture manifests and therefore validates
    mechanics without imposing this experiment's scientific coverage.  The runnable endpoint calls
    this stricter check before any backend request.
    """
    if not MIN_EXPERIMENT_PAIRS <= roster.counts.n_pairs <= MAX_EXPERIMENT_PAIRS:
        raise ValueError(
            f"cooperation experiment requires roughly 12 diagnostic pairs; got "
            f"{roster.counts.n_pairs}."
        )
    if roster.manifest.allocation is None:
        raise ValueError(
            "cooperation experiment roster is missing its private three-option allocation "
            "diagnostic; the own, total-welfare, and equality maxima must be separated explicitly."
        )
    pair_games = {pair.game_id for pair in roster.manifest.pairs}
    missing = sorted(REQUIRED_EXPERIMENT_GAMES - pair_games)
    if missing:
        raise ValueError(
            f"cooperation experiment roster is missing required diagnostic games {missing}; "
            "the small behavior battery must cover trained, held-out, numeric, and role-transfer cells."
        )
    for family in DIAGNOSTIC_FAMILIES:
        if not any(pair.family == family for pair in roster.manifest.pairs):
            raise ValueError(f"cooperation experiment roster has no {family!r} diagnostic pair.")
    if not any(pair.game_id in EVAL_ONLY_GAME_IDS for pair in roster.manifest.pairs):
        raise ValueError("cooperation experiment roster has no held-out game identity.")
    contrasts = [pair for pair in roster.manifest.pairs if pair.contrast_group is not None]
    for contrast_kind in ("wording", "relationship"):
        if not any(pair.contrast_kind == contrast_kind for pair in contrasts):
            raise ValueError(
                f"cooperation experiment roster has no {contrast_kind} contrast declaration."
            )
    grouped: dict[str, list[DiagnosticPair]] = {}
    for pair in contrasts:
        if pair.contrast_group is None:
            raise ValueError(f"contrast pair {pair.pair_id!r} has no contrast_group.")
        grouped.setdefault(pair.contrast_group, []).append(pair)
    for group, pairs in grouped.items():
        if len(pairs) < MIN_CONTRAST_MEMBERS:
            raise ValueError(f"contrast group {group!r} has only one pair.")
        signatures: set[tuple[Any, ...]] = set()
        for pair in pairs:
            generated = _rendered_rows_for_pair(pair, label_print_order=LABEL_PRINT_ORDER_CANONICAL)
            matches = [
                row
                for row in generated
                if row["reskin_id"] == pair.scenario_id
                and row["payoff_variant"] == pair.payoff_variant
            ]
            if not matches:
                raise ValueError(
                    f"contrast group {group!r} names an unrenderable pair {pair.pair_id!r}."
                )
            row = matches[0]
            signatures.add(
                tuple(
                    row[key]
                    for key in (
                        "payoff_cc",
                        "payoff_cd",
                        "payoff_dc",
                        "payoff_dd",
                        "endowment",
                        "windfall",
                        "team_size",
                        "contribution_threshold",
                        "prize",
                        "transfer_multiplier",
                    )
                )
            )
        if len(signatures) != 1:
            raise ValueError(
                f"contrast group {group!r} changes payoff mechanics; wording/relationship contrasts "
                "must hold the payoff row fixed."
            )
    held_out_identities = {
        pair.held_out_counterpart_identity
        for pair in roster.manifest.pairs
        if pair.contrast_kind == "held-out" and pair.held_out_counterpart_identity is not None
    }
    held_out_combinations = {
        pair.held_out_combination
        for pair in roster.manifest.pairs
        if pair.contrast_kind == "held-out" and pair.held_out_combination is not None
    }
    if len(held_out_identities) < MIN_CONTRAST_MEMBERS:
        raise ValueError(
            "cooperation experiment roster needs at least two declared held-out counterpart "
            f"identities; got {sorted(held_out_identities)}."
        )
    if len(held_out_combinations) < MIN_CONTRAST_MEMBERS:
        raise ValueError(
            "cooperation experiment roster needs at least two held-out component combinations; "
            f"got {sorted(held_out_combinations)}."
        )
    held_out_counterpart_pairs = [
        pair
        for pair in roster.manifest.pairs
        if pair.contrast_kind == "held-out" and pair.counterpart_framing is not None
    ]
    if not held_out_counterpart_pairs:
        raise ValueError(
            "cooperation experiment roster needs a held-out counterpart identity rendered by the "
            "counterpart-clause renderer, rather than a metadata-only identity label."
        )
    validate_familiar_component_holdout(roster)


def _training_matrix_triplets(
    path: Path = DEFAULT_TRAINING_CORPUS_PATH,
) -> frozenset[tuple[str, str, str]]:
    """Read matrix game/payoff/counterpart combinations from the private training corpus."""
    if not path.exists():
        raise FileNotFoundError(
            f"training corpus {path} is required to validate familiar-component holdouts."
        )
    triplets: set[tuple[str, str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"training corpus {path}:{line_number} must contain an object.")
        framing = payload.get("framing_id")
        if framing in (None, ""):
            continue
        game_id = payload.get("game_id")
        payoff_variant = payload.get("payoff_variant")
        if not (
            isinstance(game_id, str)
            and game_id
            and isinstance(payoff_variant, str)
            and payoff_variant
            and isinstance(framing, str)
            and framing
        ):
            raise ValueError(
                f"training corpus {path}:{line_number} has an incomplete framed matrix identity."
            )
        triplets.add((game_id, payoff_variant, framing))
    if not triplets:
        raise ValueError(f"training corpus {path} contains no framed matrix identities.")
    return frozenset(triplets)


def validate_familiar_component_holdout(
    roster: BehaviorRoster,
    training_corpus_path: Path = DEFAULT_TRAINING_CORPUS_PATH,
) -> None:
    """Require a held-out matrix triplet whose three components each occur in training.

    The absent combination is derived from the training rows at validation time.  This prevents a
    metadata label from claiming a familiar-component holdout when the counterpart, game, or payoff
    was never present in the treatment corpus.
    """
    training_triplets = _training_matrix_triplets(training_corpus_path)
    training_games = {triplet[0] for triplet in training_triplets}
    training_payoffs = {triplet[1] for triplet in training_triplets}
    training_framings = {triplet[2] for triplet in training_triplets}
    candidates = []
    for pair in roster.manifest.pairs:
        if pair.contrast_kind != "held-out" or pair.counterpart_framing is None:
            continue
        triplet = (pair.game_id, pair.payoff_variant, pair.counterpart_framing)
        if (
            triplet not in training_triplets
            and pair.game_id in training_games
            and pair.payoff_variant in training_payoffs
            and pair.counterpart_framing in training_framings
        ):
            candidates.append((pair.pair_id, triplet))
    if not candidates:
        raise ValueError(
            "cooperation experiment roster has no held-out familiar-component matrix triplet: "
            f"training combinations={sorted(training_triplets)}."
        )


def expand_behavior_roster(manifest: BehaviorManifest) -> BehaviorRoster:
    """Expand identifiers into all rows, refusing partial pairs and duplicate identities."""
    manifest.validate()
    expanded: list[ExpandedBehaviorRow] = []
    identities: set[tuple[str, str, str]] = set()
    for pair in manifest.pairs:
        rows_for_pair: list[Mapping[str, Any]] = []
        for order in pair.label_print_orders:
            rows = _rendered_rows_for_pair(pair, label_print_order=order)
            matches = [
                row
                for row in rows
                if row["reskin_id"] == pair.scenario_id
                and row["payoff_variant"] == pair.payoff_variant
            ]
            rows_for_pair.extend(matches)
        if not rows_for_pair:
            raise ValueError(
                f"{pair.pair_id!r} selected no eval row for game={pair.game_id!r}, "
                f"scenario={pair.scenario_id!r}, payoff_variant={pair.payoff_variant!r}."
            )
        labelled = bool(rows_for_pair[0]["label_a"] or rows_for_pair[0]["label_b"])
        expected = len(pair.label_print_orders) * (2 if labelled else 1)
        if len(rows_for_pair) != expected:
            raise ValueError(
                f"{pair.pair_id!r} expanded to {len(rows_for_pair)} rows, expected {expected}; "
                f"the whole action mapping/order pair must be present."
            )
        if labelled and {row["coop_label"] for row in rows_for_pair} != {
            rows_for_pair[0]["label_a"],
            rows_for_pair[0]["label_b"],
        }:
            raise ValueError(f"{pair.pair_id!r} did not retain both action mappings.")
        for row in rows_for_pair:
            identity = (str(row["game_id"]), str(row["prompt_id"]), str(row["label_print_order"]))
            if identity in identities:
                raise ValueError(f"behavior roster repeats expanded row identity {identity}.")
            identities.add(identity)
            expanded.append(
                ExpandedBehaviorRow(
                    pair_id=pair.pair_id,
                    family=pair.family,
                    scenario_group=pair.scenario_group,
                    samples=pair.samples,
                    raw=row,
                    contrast_group=pair.contrast_group,
                    contrast_kind=pair.contrast_kind,
                    held_out_counterpart_identity=pair.held_out_counterpart_identity,
                    held_out_combination=pair.held_out_combination,
                )
            )
    by_family = Counter(row.family for row in expanded)
    by_order = Counter(str(row.raw["label_print_order"]) for row in expanded)
    by_pair = Counter(row.pair_id for row in expanded)
    counts = BehaviorCounts(
        n_pairs=len(manifest.pairs),
        n_rendered_prompts=len(expanded),
        n_completions=sum(row.samples for row in expanded),
        by_family=dict(sorted(by_family.items())),
        by_print_order=dict(sorted(by_order.items())),
        by_pair=dict(sorted(by_pair.items())),
    )
    return BehaviorRoster(tuple(expanded), manifest, counts)


def validate_group_isolation(roster: BehaviorRoster, reserved_group_ids: Iterable[str]) -> None:
    """Refuse training or interpretability groups that overlap the behavior evaluation roster."""
    reserved = frozenset(reserved_group_ids)
    overlap = sorted(roster.scenario_group_ids & reserved)
    if overlap:
        raise ValueError(
            f"behavior evaluation scenario groups overlap a reserved split: {overlap}. "
            "Keep whole scenario/template groups on one side of every split."
        )


def assert_group_sets_disjoint(**group_sets: Iterable[str]) -> None:
    """Validate named train/eval/interp group sets pairwise."""
    seen: dict[str, str] = {}
    for set_name, values in group_sets.items():
        for value in values:
            previous = seen.get(value)
            if previous is not None:
                raise ValueError(
                    f"group {value!r} occurs in both {previous} and {set_name}; overlap."
                )
            seen[value] = set_name


def _request_id(identity: Sequence[Any]) -> str:
    """Render a stable, readable request identity for retained rollout joins."""
    return "::".join(str(value) for value in identity)


def _behavior_request_id_prefix(row: Mapping[str, Any]) -> str:
    """Return the sample-independent identity shared by context and behavior requests."""
    return _request_id(
        (
            SECTION_GAME_BEHAVIOR,
            str(row["game_id"]),
            str(row["prompt_id"]),
            str(row["label_print_order"]),
        )
    )


def _numeric_forecast_event(row: Mapping[str, Any]) -> NumericForecastEvent | None:
    """Resolve the executable event for a row, or None for a labelled action row."""
    if bool(row.get("label_a") or row.get("label_b")):
        return None
    game_id = str(row["game_id"])
    try:
        return NUMERIC_FORECAST_EVENTS[game_id]
    except KeyError as exc:
        raise ValueError(
            f"numeric game {game_id!r} has forecast prose but no executable event predicate."
        ) from exc


def _event_metadata(
    row: Mapping[str, Any],
    event: NumericForecastEvent | None,
    *,
    observed: bool | None,
) -> dict[str, Any]:
    """Build the shared event and prompt-link fields for retained records."""
    return {
        "forecast_event_id": None if event is None else event.event_id,
        "forecast_target_predicate": None if event is None else event.predicate,
        "forecast_event_observed": observed,
        "behavior_request_id_prefix": _behavior_request_id_prefix(row),
    }


def _answer_tag_name(record: Mapping[str, Any]) -> str | None:
    for field_name, tag_name in (
        ("action", "action"),
        ("kept", "keep"),
        ("claim", "claim"),
        ("contribution", "contribute"),
        ("sent", "send"),
        ("return_fraction", "return"),
        ("set_down", "set"),
        ("level", "level"),
        ("levels", "level"),
    ):
        if record.get(field_name) is not None:
            return tag_name
    return None


def _pre_action_prefix(completion: str, *, parsed: bool, record: Mapping[str, Any]) -> str | None:
    """Return the parser-delimited prefix immediately before the terminal answer tag."""
    if not parsed:
        return None
    answer_tag = _answer_tag_name(record)
    if answer_tag is None:
        return None
    thinking_closes = tuple(re.finditer(r"</think>", completion))
    if len(thinking_closes) != 1:
        return None
    thinking_close = thinking_closes[0].end()
    visible = completion[thinking_close:]
    tag_pattern = re.compile(rf"<{answer_tag}>(.*?)</{answer_tag}>", re.IGNORECASE | re.DOTALL)
    matches = tuple(tag_pattern.finditer(visible))
    if len(matches) != 1:
        return None
    prefix = completion[: thinking_close + matches[-1].start()]
    prefix = prefix.removeprefix("<think>")
    return prefix.strip() or None


def build_behavior_plan(
    roster: BehaviorRoster,
    *,
    prefilled_think: bool = True,
    trained_game_ids: Iterable[str] = (),
    rollout_id: str = "cooperation-eval",
) -> list[PlannedRequest]:
    """Build behavior requests with exact row-major sample identities."""
    rollout_id = _required_text(rollout_id, "rollout_id")
    trained = frozenset(trained_game_ids)
    plan: list[PlannedRequest] = []
    for expanded in roster.rows:
        row = expanded.raw
        for sample_index in range(expanded.samples):
            identity = (
                SECTION_GAME_BEHAVIOR,
                str(row["game_id"]),
                str(row["prompt_id"]),
                str(row["label_print_order"]),
                sample_index,
            )

            def parse(
                completion: str,
                *,
                row: Mapping[str, Any] = row,
                expanded: ExpandedBehaviorRow = expanded,
                sample_index: int = sample_index,
                identity: tuple[Any, ...] = identity,
            ) -> dict[str, Any]:
                record = _game_request_record(
                    str(row["game_id"]),
                    row,
                    completion,
                    section=SECTION_GAME_BEHAVIOR,
                    framing_id=cast("str | None", row.get("framing_id")),
                    trap_id=None,
                    prefilled_think=prefilled_think,
                    trained_game=str(row["game_id"]) in trained,
                    eval_only_game=str(row["game_id"]) in EVAL_ONLY_GAME_IDS,
                    sample_index=sample_index,
                )
                record.update(
                    {
                        "pair_id": expanded.pair_id,
                        "diagnostic_family": expanded.family,
                        "scenario_group": expanded.scenario_group,
                        "contrast_group": expanded.contrast_group,
                        "contrast_kind": expanded.contrast_kind,
                        "held_out_counterpart_identity": expanded.held_out_counterpart_identity,
                        "held_out_combination": expanded.held_out_combination,
                        "behavior_prompt_digest": hashlib.sha256(
                            str(row["prompt"]).encode("utf-8")
                        ).hexdigest()[:16],
                        "request_id": _request_id(identity),
                        "rollout_id": rollout_id,
                        "prompt": str(row["prompt"]),
                        "pre_action_prefix": _pre_action_prefix(
                            completion,
                            parsed=bool(record.get("parsed")),
                            record=record,
                        ),
                    }
                )
                annotated = annotate_behavior_record(row, record)
                event = _numeric_forecast_event(row)
                annotated.update(
                    _event_metadata(
                        row,
                        event,
                        observed=None if event is None else event.observe(row, annotated),
                    )
                )
                return annotated

            plan.append(
                PlannedRequest(
                    section=SECTION_GAME_BEHAVIOR,
                    identity=identity,
                    prompt=str(row["prompt"]),
                    call_group=f"cooperation-behavior::{expanded.pair_id}",
                    parse=parse,
                )
            )
    return plan


def select_diagnostic_rows(
    roster: BehaviorRoster,
    pair_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Return full renderer rows for an interp/intervention subset.

    The returned rows retain every ``generate_prompt_rows`` column and add only manifest metadata.
    This lets steering and capture consume the same exact prompt/payoff rows as behavior evaluation.
    """
    requested = tuple(pair_ids)
    if not requested:
        raise ValueError("pair_ids must select at least one diagnostic pair.")
    known = {pair.pair_id for pair in roster.manifest.pairs}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"unknown behavior pair_ids {unknown}; known: {sorted(known)}.")
    selected = [row for row in roster.rows if row.pair_id in set(requested)]
    return [
        {
            **dict(expanded.raw),
            "pair_id": expanded.pair_id,
            "diagnostic_family": expanded.family,
            "scenario_group": expanded.scenario_group,
            "contrast_group": expanded.contrast_group,
            "contrast_kind": expanded.contrast_kind,
            "held_out_counterpart_identity": expanded.held_out_counterpart_identity,
            "held_out_combination": expanded.held_out_combination,
        }
        for expanded in selected
    ]


def _decision_context(row: Mapping[str, Any]) -> str:
    prompt = str(row["prompt"])
    first, separator, tail = prompt.rpartition("\n\n")
    answer_tags = ("<action>", "<keep>", "<send>", "<return>", "<claim>", "<contribute>", "<level>")
    if (
        not separator
        or not any(tag in tail for tag in answer_tags)
        or "Write exactly one tag" not in tail
    ):
        raise ValueError(
            f"behavior prompt {row['prompt_id']!r} has no recognised action instruction; the "
            "full counterpart-bearing context cannot be extracted safely."
        )
    return first


def _context_item(  # noqa: PLR0913
    *,
    item_id: str,
    instrument: str,
    stem: str,
    stem_swapped: str | None,
    kind: str,
    construct: str,
    expected_direction: str,
    options: tuple[str, ...] = (),
    numeric_max: int = 0,
    predicts_game: str | None = None,
    family: str = FAMILY_SELF_PREDICTION,
) -> SurveyItem:
    return SurveyItem(
        item_id=item_id,
        family=family,
        instrument=instrument,
        kind=kind,
        stem=stem,
        stem_swapped=stem_swapped,
        options=options,
        numeric_max=numeric_max,
        construct=construct,
        expected_direction=expected_direction,
        predicts_game=predicts_game,
    )


def _order_name(row: Mapping[str, Any]) -> str:
    return "as-authored" if row["label_print_order"] == LABEL_PRINT_ORDER_CANONICAL else "reversed"


def _forecast_target_text(row: Mapping[str, Any], elicitation: ContextElicitation) -> str:
    if row["label_a"] or row["label_b"]:
        return elicitation.matrix_target_template.format(coop_label=str(row["coop_label"]))
    game_id = str(row["game_id"])
    try:
        return elicitation.numeric_targets[game_id]
    except KeyError as exc:
        raise ValueError(
            f"context elicitation has no target for numeric game {game_id!r}."
        ) from exc


def _embedded_context_stem(row: Mapping[str, Any], question: str) -> str:
    """Put the complete original play prompt inside an explicit context boundary."""
    return (
        f"{EXACT_DECISION_PROMPT_BEGIN}\n{row['prompt']}\n{EXACT_DECISION_PROMPT_END}\n\n{question}"
    )


def _extract_embedded_decision_context(prompt: str) -> str:
    """Extract one unambiguous decision prompt from a rendered context item."""
    if prompt.count(EXACT_DECISION_PROMPT_BEGIN) != 1:
        raise ValueError(
            "embedded decision prompt must contain exactly one begin delimiter in the request."
        )
    if prompt.count(EXACT_DECISION_PROMPT_END) != 1:
        raise ValueError(
            "embedded decision prompt must contain exactly one end delimiter in the request."
        )
    begin_end = prompt.index(EXACT_DECISION_PROMPT_BEGIN) + len(EXACT_DECISION_PROMPT_BEGIN)
    end_start = prompt.index(EXACT_DECISION_PROMPT_END)
    if end_start < begin_end:
        raise ValueError("embedded decision prompt delimiters are in the wrong order.")
    embedded = prompt[begin_end:end_start]
    if not embedded.startswith("\n") or not embedded.endswith("\n"):
        raise ValueError("embedded decision prompt must be newline-delimited in the request.")
    return embedded[1:-1]


def _swapped_row_for(
    roster: BehaviorRoster, expanded: ExpandedBehaviorRow
) -> Mapping[str, Any] | None:
    row = expanded.raw
    matches = [
        candidate.raw
        for candidate in roster.rows
        if candidate.pair_id == expanded.pair_id
        and candidate.raw["label_print_order"] == LABEL_PRINT_ORDER_SWAPPED
        and candidate.raw["coop_label"] == row["coop_label"]
    ]
    return matches[0] if matches else None


def _numeric_normative_max(row: Mapping[str, Any]) -> int:
    game_id = str(row["game_id"])
    if game_id == "dictator":
        return int(row["endowment"])
    if game_id in {"trust-vs-stated-return", "trust-strategy-method"}:
        return int(row["endowment"])
    if game_id == "trustee-return-rule":
        return 100
    if game_id in {"nash-demand", "threshold-goods"}:
        return int(row["windfall"] or row["endowment"])
    if int(row.get("n_levels", 0)) > 0:
        return int(row["n_levels"])
    raise ValueError(f"no numeric normative answer domain is registered for {game_id!r}.")


def _match_known_optimum(
    chosen: int | str | None, optimal_actions: Sequence[int | str] | None
) -> bool | None:
    """Compare an answer only when the scorer resolved a nonempty optimum set."""
    if chosen is None or not optimal_actions:
        return None
    return chosen in optimal_actions


def _survey_record(  # noqa: PLR0913
    item: SurveyItem,
    completion: str,
    *,
    prompt: str,
    option_order: tuple[int, ...],
    order_name: str,
    sample_index: int,
    numeric_example: int | None,
    row: Mapping[str, Any],
    forecast: bool,
    prefilled_think: bool,
    normative_dimension: str | None = None,
    forecast_target_action: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    embedded_decision_context = _extract_embedded_decision_context(prompt)
    expected_decision_context = str(row["prompt"])
    if embedded_decision_context != expected_decision_context:
        raise ValueError(
            f"embedded decision prompt for {row['prompt_id']!r} does not match the decision row."
        )
    visible, truncated = strip_thinking(completion, prefilled_think=prefilled_think)
    answer = parse_survey_answer(item, visible, option_order=option_order or None)
    event = _numeric_forecast_event(row)
    record = {
        "record": SECTION_SELF_REPORT,
        **survey_record_fields(item, answer),
        "sample_index": sample_index,
        "option_order_name": order_name,
        "option_order": list(option_order),
        "numeric_example": numeric_example,
        "truncated_thinking": truncated,
        "completion": completion,
        "visible_text": visible,
        "prompt": prompt,
        "decision_prompt_id": str(row["prompt_id"]),
        "decision_prompt_digest": hashlib.sha256(
            expected_decision_context.encode("utf-8")
        ).hexdigest()[:16],
        "embedded_decision_prompt_digest": hashlib.sha256(
            embedded_decision_context.encode("utf-8")
        ).hexdigest()[:16],
        "decision_context_digest": hashlib.sha256(
            _decision_context(row).encode("utf-8")
        ).hexdigest()[:16],
        "behavior_prompt_id": str(row["prompt_id"]),
        "request_id": request_id,
        **_event_metadata(row, event, observed=None),
        "forecast": forecast,
    }
    if forecast:
        numeric = record["numeric"]
        if numeric is None:
            predicted_fraction = None
        elif isinstance(numeric, int | float):
            predicted_fraction = float(numeric) / 100.0
        else:
            raise TypeError(f"numeric forecast answer has an invalid value {numeric!r}.")
        record["predicted_cooperation_fraction"] = predicted_fraction
        record["predicted_event_fraction"] = (
            None if event is None else record["predicted_cooperation_fraction"]
        )
        if not forecast_target_action:
            raise ValueError("forecast records require an explicit target description.")
        record["forecast_target_action"] = forecast_target_action
        record["normative_action"] = None
    else:
        record["predicted_cooperation_fraction"] = None
        record["normative_action"] = (
            record["numeric"] if record["kind"] == SURVEY_NUMERIC else record["canonical_index"]
        )
        if record["kind"] == SURVEY_CHOICE:
            canonical = record["canonical_index"]
            chosen: int | str | None = None
            if canonical is not None:
                chosen = (
                    COOPERATE
                    if (str(row["label_a"]) if canonical == 0 else str(row["label_b"]))
                    == str(row["coop_label"])
                    else DEFECT
                )
        else:
            numeric = record["numeric"]
            if numeric is not None and not isinstance(numeric, int):
                raise TypeError(f"numeric normative answer has an invalid value {numeric!r}.")
            chosen = numeric
        ranked = rank_row_actions(row, chosen)
        record["normative_reward_optimal_action"] = ranked["reward_optimal_action"]
        record["normative_reward_optimal_value"] = ranked["reward_optimal_value"]
        record["normative_matches_reward_optimum"] = _match_known_optimum(
            chosen, ranked["reward_optimal_actions"]
        )
        if normative_dimension is None:
            raise ValueError("normative records require an explicit target dimension.")
        target_actions_key = {
            "reward": "reward_optimal_actions",
            "own": "own_optimal_actions",
            "welfare": "welfare_optimal_actions",
            "equality": "equality_optimal_actions",
        }[normative_dimension]
        target_actions = ranked[target_actions_key]
        record["normative_target_dimension"] = normative_dimension
        record["normative_target_optimal_actions"] = target_actions
        record["normative_matches_target_optimum"] = _match_known_optimum(chosen, target_actions)
    return record


def _allocation_record(  # noqa: PLR0913
    allocation: AllocationDiagnostic,
    item: SurveyItem,
    completion: str,
    *,
    prompt: str,
    option_order: tuple[int, ...],
    order_name: str,
    sample_index: int,
    prefilled_think: bool,
) -> dict[str, Any]:
    """Parse and annotate one private three-option allocation completion."""
    visible, truncated = strip_thinking(completion, prefilled_think=prefilled_think)
    answer = parse_survey_answer(item, visible, option_order=option_order)
    chosen_option = None if answer.canonical_index is None else str(answer.canonical_index)
    payoffs_by_option = {str(index): pair for index, pair in enumerate(allocation.option_payoffs)}
    ranked = rank_allocation_options(payoffs_by_option, chosen_option=chosen_option)
    return {
        "record": SECTION_SELF_REPORT,
        **survey_record_fields(item, answer),
        "sample_index": sample_index,
        "option_order_name": order_name,
        "option_order": list(option_order),
        "numeric_example": None,
        "truncated_thinking": truncated,
        "completion": completion,
        "visible_text": visible,
        "prompt": prompt,
        "allocation_diagnostic_id": allocation.diagnostic_id,
        "scenario_group": allocation.scenario_group_id,
        "allocation_option_payoffs": {
            option: list(pair) for option, pair in payoffs_by_option.items()
        },
        "allocation_chosen_option": chosen_option,
        "allocation_own_optimal_options": ranked["own_optimal_actions"],
        "allocation_total_welfare_optimal_options": ranked["welfare_optimal_actions"],
        "allocation_equality_optimal_options": ranked["equality_optimal_actions"],
        "allocation_chosen_payoff_self": ranked["own_payoff"],
        "allocation_chosen_payoff_other": ranked["counterpart_payoff"],
        "allocation_chosen_total_welfare": ranked["total_welfare"],
        "allocation_chosen_equality_gap": (
            None
            if chosen_option is None
            else abs(
                allocation.option_payoffs[int(chosen_option)][0]
                - allocation.option_payoffs[int(chosen_option)][1]
            )
        ),
    }


def build_allocation_plan(
    roster: BehaviorRoster,
    *,
    prefilled_think: bool = True,
) -> list[PlannedRequest]:
    """Build the private three-option allocation diagnostic under three cyclic orders."""
    allocation = roster.manifest.allocation
    if allocation is None:
        raise ValueError("the behavior manifest has no private allocation diagnostic.")
    item = SurveyItem(
        item_id=f"allocation-three-way::{allocation.diagnostic_id}",
        family=FAMILY_NEGATIVE_CONTROL,
        instrument="allocation-three-way",
        kind=SURVEY_ALLOCATION,
        stem=allocation.stem,
        construct=(
            roster.manifest.elicitation.construct
            if roster.manifest.elicitation is not None
            else "runtime allocation preference diagnostic"
        ),
        expected_direction=(
            roster.manifest.elicitation.expected_direction
            if roster.manifest.elicitation is not None
            else "preserve all three payoff dimensions"
        ),
        option_payoffs=allocation.option_payoffs,
    )
    requests: list[PlannedRequest] = []
    for order_name, option_order in allocation_option_orders(len(allocation.option_payoffs)):
        prompt = render_survey_prompt(item, option_order=option_order)
        for sample_index in range(allocation.samples):

            def parse(  # noqa: PLR0913
                completion: str,
                *,
                allocation: AllocationDiagnostic = allocation,
                item: SurveyItem = item,
                prompt: str = prompt,
                option_order: tuple[int, ...] = option_order,
                order_name: str = order_name,
                sample_index: int = sample_index,
            ) -> dict[str, Any]:
                return _allocation_record(
                    allocation,
                    item,
                    completion,
                    prompt=prompt,
                    option_order=option_order,
                    order_name=order_name,
                    sample_index=sample_index,
                    prefilled_think=prefilled_think,
                )

            requests.append(
                PlannedRequest(
                    section=SECTION_SELF_REPORT,
                    identity=(SECTION_SELF_REPORT, item.item_id, order_name, sample_index),
                    prompt=prompt,
                    call_group="allocation-three-way",
                    parse=parse,
                )
            )
    _assert_unique_request_identities(requests)
    return requests


def forecast_and_normative_plans(
    roster: BehaviorRoster,
    *,
    samples: int = 2,
    prefilled_think: bool = True,
    elicitation: ContextElicitation | None = None,
) -> tuple[list[PlannedRequest], list[PlannedRequest]]:
    """Plan exact-context forecasts and separate normative questions for every behavior row."""
    samples = _positive_int(samples, "samples")
    resolved_elicitation = elicitation or roster.manifest.elicitation
    if resolved_elicitation is None:
        raise ValueError(
            "context elicitation is missing. Load the private behavior manifest with its "
            "runtime elicitation block before planning full-context items."
        )
    resolved_elicitation.validate_for_rows(expanded.raw for expanded in roster.rows)
    forecasts: list[PlannedRequest] = []
    normative: list[PlannedRequest] = []
    for expanded in roster.rows:
        row = expanded.raw
        _decision_context(row)
        swapped = _swapped_row_for(roster, expanded)
        swapped_context = None if swapped is None else _decision_context(swapped)
        if bool(row["label_a"] or row["label_b"]):
            if swapped_context is None:
                raise ValueError(f"{expanded.pair_id!r} lacks the swapped context for a forecast.")
            option_order = ()
            forecast_order_name = _order_name(row)
        else:
            option_order = ()
            forecast_order_name = "not-applicable"
        stable_prompt_id = str(row["prompt_id"])
        forecast_item = _context_item(
            item_id=f"{INSTRUMENT_CONTEXT_SELF_PREDICTION}::{expanded.pair_id}::{stable_prompt_id}",
            instrument=INSTRUMENT_CONTEXT_SELF_PREDICTION,
            stem=_embedded_context_stem(
                row,
                resolved_elicitation.forecast_template.format(
                    target=_forecast_target_text(row, resolved_elicitation)
                ),
            ),
            stem_swapped=None,
            kind=SURVEY_NUMERIC,
            numeric_max=100,
            predicts_game=str(row["game_id"]),
            construct=resolved_elicitation.construct,
            expected_direction=resolved_elicitation.expected_direction,
        )
        rotation = numeric_example_rotation(forecast_item)
        for sample_index in range(samples):
            numeric_example = rotation[sample_index % len(rotation)]
            prompt = render_survey_prompt(
                forecast_item,
                option_order=option_order or None,
                numeric_example=numeric_example,
            )

            def parse_forecast(  # noqa: PLR0913
                completion: str,
                *,
                item: SurveyItem = forecast_item,
                prompt: str = prompt,
                option_order: tuple[int, ...] = option_order,
                order_name: str = forecast_order_name,
                sample_index: int = sample_index,
                numeric_example: int = numeric_example,
                row: Mapping[str, Any] = row,
                identity: tuple[Any, ...] = (
                    SECTION_SELF_REPORT,
                    forecast_item.item_id,
                    forecast_order_name,
                    sample_index,
                ),
            ) -> dict[str, Any]:
                return _survey_record(
                    item,
                    completion,
                    prompt=prompt,
                    option_order=option_order,
                    order_name=order_name,
                    sample_index=sample_index,
                    numeric_example=numeric_example,
                    row=row,
                    forecast=True,
                    prefilled_think=prefilled_think,
                    forecast_target_action=_forecast_target_text(row, resolved_elicitation),
                    request_id=_request_id(identity),
                )

            forecasts.append(
                PlannedRequest(
                    section=SECTION_SELF_REPORT,
                    identity=(
                        SECTION_SELF_REPORT,
                        forecast_item.item_id,
                        forecast_order_name,
                        sample_index,
                    ),
                    prompt=prompt,
                    call_group=INSTRUMENT_CONTEXT_SELF_PREDICTION,
                    parse=parse_forecast,
                )
            )
        normative_kind = SURVEY_CHOICE if bool(row["label_a"] or row["label_b"]) else SURVEY_NUMERIC
        normative_options = (
            (str(row["label_a"]), str(row["label_b"])) if normative_kind == SURVEY_CHOICE else ()
        )
        normative_item = _context_item(
            item_id=f"{INSTRUMENT_NORMATIVE_PAYOFF}::{expanded.pair_id}::{row['prompt_id']}",
            instrument=INSTRUMENT_NORMATIVE_PAYOFF,
            stem=_embedded_context_stem(
                row,
                (
                    (
                        resolved_elicitation.normative_choice_template
                        if normative_kind == SURVEY_CHOICE
                        else resolved_elicitation.normative_numeric_template
                    ).format(target=resolved_elicitation.normative_target)
                ),
            ),
            stem_swapped=None,
            kind=normative_kind,
            options=normative_options,
            numeric_max=0 if normative_kind == SURVEY_CHOICE else _numeric_normative_max(row),
            family=FAMILY_NEGATIVE_CONTROL,
            construct=resolved_elicitation.construct,
            expected_direction=resolved_elicitation.expected_direction,
        )
        normative_order = option_order
        for sample_index in range(samples):
            normative_example = None
            if normative_kind == SURVEY_NUMERIC:
                rotation = numeric_example_rotation(normative_item)
                normative_example = rotation[sample_index % len(rotation)]
            prompt = render_survey_prompt(
                normative_item,
                option_order=normative_order or None,
                numeric_example=normative_example,
            )

            def parse_normative(  # noqa: PLR0913
                completion: str,
                *,
                item: SurveyItem = normative_item,
                prompt: str = prompt,
                option_order: tuple[int, ...] = normative_order,
                order_name: str = forecast_order_name,
                sample_index: int = sample_index,
                numeric_example: int | None = normative_example,
                row: Mapping[str, Any] = row,
                identity: tuple[Any, ...] = (
                    SECTION_SELF_REPORT,
                    normative_item.item_id,
                    forecast_order_name,
                    sample_index,
                ),
            ) -> dict[str, Any]:
                return _survey_record(
                    item,
                    completion,
                    prompt=prompt,
                    option_order=option_order,
                    order_name=order_name,
                    sample_index=sample_index,
                    numeric_example=numeric_example,
                    row=row,
                    forecast=False,
                    prefilled_think=prefilled_think,
                    normative_dimension=resolved_elicitation.normative_dimension,
                    request_id=_request_id(identity),
                )

            normative.append(
                PlannedRequest(
                    section=SECTION_SELF_REPORT,
                    identity=(
                        SECTION_SELF_REPORT,
                        normative_item.item_id,
                        forecast_order_name,
                        sample_index,
                    ),
                    prompt=prompt,
                    call_group=INSTRUMENT_NORMATIVE_PAYOFF,
                    parse=parse_normative,
                )
            )
    _assert_unique_request_identities(forecasts + normative)
    return forecasts, normative


def summarize_forecast_behavior_events(
    forecast_records: Iterable[Mapping[str, Any]],
    behavior_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join numeric forecast events to independent behavior draws by prompt and print order.

    The behavior side can contain unresolved parser results.  They stay in the denominator metadata
    and are excluded from the observed event rate, so an unavailable play is never recoded as a
    negative event.
    """
    forecasts = tuple(forecast_records)
    behaviors = tuple(behavior_records)
    behavior_by_key: dict[str, list[bool | None]] = {}
    for behavior in behaviors:
        event_id = behavior.get("forecast_event_id")
        join_key = behavior.get("behavior_request_id_prefix")
        if not isinstance(event_id, str) or not isinstance(join_key, str):
            continue
        observed = behavior.get("forecast_event_observed")
        if observed is not None and not isinstance(observed, bool):
            raise TypeError(f"behavior event observation must be bool or None, got {observed!r}.")
        behavior_by_key.setdefault(join_key, []).append(observed)

    joins: list[dict[str, Any]] = []
    for forecast in forecasts:
        event_id = forecast.get("forecast_event_id")
        join_key = forecast.get("behavior_request_id_prefix")
        if not isinstance(event_id, str) or not isinstance(join_key, str):
            continue
        observations = behavior_by_key.get(join_key, [])
        resolved = [value for value in observations if value is not None]
        joins.append(
            {
                "request_id": forecast.get("request_id"),
                "item_id": forecast.get("item_id"),
                "sample_index": forecast.get("sample_index"),
                "event_id": event_id,
                "behavior_request_id_prefix": join_key,
                "predicted_event_fraction": forecast.get("predicted_event_fraction"),
                "n_behavior_records": len(observations),
                "n_behavior_resolved": len(resolved),
                "n_behavior_unresolved": len(observations) - len(resolved),
                "n_behavior_event_true": sum(value is True for value in resolved),
                "observed_event_rate": (
                    None
                    if not resolved
                    else sum(value is True for value in resolved) / len(resolved)
                ),
            }
        )

    event_keys: dict[str, set[str]] = {}
    event_forecast_counts: Counter[str] = Counter()
    event_parsed_counts: Counter[str] = Counter()
    for join in joins:
        event_id = str(join["event_id"])
        event_forecast_counts[event_id] += 1
        if join["predicted_event_fraction"] is not None:
            event_parsed_counts[event_id] += 1
        event_keys.setdefault(event_id, set()).add(str(join["behavior_request_id_prefix"]))

    by_event_id: dict[str, dict[str, Any]] = {}
    for event_id, keys in event_keys.items():
        observations = [value for key in keys for value in behavior_by_key[key]]
        resolved = [value for value in observations if value is not None]
        by_event_id[event_id] = {
            "n_forecast_records": event_forecast_counts[event_id],
            "n_forecast_parsed": event_parsed_counts[event_id],
            "n_behavior_join_keys": len(keys),
            "n_behavior_records": len(observations),
            "n_behavior_resolved": len(resolved),
            "n_behavior_unresolved": len(observations) - len(resolved),
            "n_behavior_event_true": sum(value is True for value in resolved),
            "observed_event_rate": (
                None if not resolved else sum(value is True for value in resolved) / len(resolved)
            ),
        }
    return {
        "n_forecast_records": len(forecasts),
        "n_event_forecast_records": len(joins),
        "n_behavior_records": len(behaviors),
        "n_event_behavior_records": sum(len(values) for values in behavior_by_key.values()),
        "joins": joins,
        "by_event_id": by_event_id,
    }


def _assert_unique_request_identities(requests: Sequence[PlannedRequest]) -> None:
    identities = [request.identity for request in requests]
    repeated = [identity for identity, n in Counter(identities).items() if n > 1]
    if repeated:
        raise RuntimeError(f"cooperation evaluation plan repeats identities {repeated[:3]}.")


@dataclass(frozen=True, slots=True)
class CooperationEvalPlan:
    """The frozen behavior, context, and allocation requests for one checkpoint."""

    manifest_path: Path
    roster: BehaviorRoster
    behavior: tuple[PlannedRequest, ...]
    forecasts: tuple[PlannedRequest, ...]
    normative: tuple[PlannedRequest, ...]
    allocation: tuple[PlannedRequest, ...]

    @property
    def requests(self) -> tuple[PlannedRequest, ...]:
        """Return requests in stable behavior, forecast, normative, allocation order."""
        return self.behavior + self.forecasts + self.normative + self.allocation

    @property
    def sections(self) -> tuple[str, ...]:
        """Return the native evaluator sections represented by this plan."""
        return (SECTION_GAME_BEHAVIOR, SECTION_SELF_REPORT)

    @property
    def counts(self) -> dict[str, int]:
        """Return rendered request counts for run metadata and integration checks."""
        return {
            "behavior_requests": len(self.behavior),
            "forecast_requests": len(self.forecasts),
            "normative_requests": len(self.normative),
            "allocation_requests": len(self.allocation),
            "allocation_rendered_prompts": len(self.allocation)
            // self.roster.manifest.allocation.samples
            if self.roster.manifest.allocation is not None
            else 0,
            "total_requests": len(self.requests),
            "behavior_pairs": self.roster.counts.n_pairs,
            "behavior_rendered_prompts": self.roster.counts.n_rendered_prompts,
            "behavior_completions": self.roster.counts.n_completions,
        }


def build_cooperation_plan(
    manifest_path: Path = DEFAULT_BEHAVIOR_MANIFEST_PATH,
    *,
    context_samples: int = 2,
    prefilled_think: bool = True,
    trained_game_ids: Iterable[str] = (),
    rollout_id: str = "cooperation-eval",
) -> CooperationEvalPlan:
    """Build the strict behavior/context plan without constructing or calling a model."""
    roster = expand_behavior_roster(load_behavior_manifest(manifest_path))
    validate_experiment_coverage(roster)
    behavior = tuple(
        build_behavior_plan(
            roster,
            prefilled_think=prefilled_think,
            trained_game_ids=trained_game_ids,
            rollout_id=rollout_id,
        )
    )
    forecasts, normative = forecast_and_normative_plans(
        roster, samples=context_samples, prefilled_think=prefilled_think
    )
    allocation = tuple(build_allocation_plan(roster, prefilled_think=prefilled_think))
    plan = CooperationEvalPlan(
        manifest_path=manifest_path,
        roster=roster,
        behavior=behavior,
        forecasts=tuple(forecasts),
        normative=tuple(normative),
        allocation=allocation,
    )
    _assert_unique_request_identities(plan.requests)
    return plan


def _manifest_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _run_planned_cell(  # noqa: PLR0913
    backend: Backend,
    *,
    plan: Sequence[PlannedRequest],
    sections: Sequence[str],
    out_path: Path,
    meta: Mapping[str, Any],
    config: EvalConfig,
    resume: bool,
    submission: str,
    admission: str,
    on_records_written: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run a rendered cell through the native trace/resume implementation."""
    return run_eval_battery(
        backend,
        sections=sections,
        out_path=out_path,
        meta=meta,
        config=config,
        submission=submission,
        admission=admission,
        resume=resume,
        on_records_written=on_records_written,
        plan=plan,
    )


def run_cooperation_eval(  # noqa: PLR0913
    backend: Backend,
    *,
    out_path: Path,
    meta: Mapping[str, Any],
    manifest_path: Path = DEFAULT_BEHAVIOR_MANIFEST_PATH,
    config: EvalConfig | None = None,
    context_samples: int = 2,
    trained_game_ids: Iterable[str] = (),
    rollout_id: str | None = None,
    resume: bool = False,
    submission: str = SUBMISSION_POOLED,
    admission: str = ADMISSION_LONGEST_FIRST,
    on_records_written: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Execute behavior, forecasts, and normative choices with native per-record resume.

    ``backend`` is the already-configured runtime adapter. Its model identity and transport, plus
    the sampler settings carried by ``EvalConfig``, are written by :func:`run_eval_battery`.
    """
    resolved_config = config or EvalConfig(
        game_behavior_samples=1,
        survey_samples=context_samples,
    )
    resolved_rollout_id = _required_text(
        rollout_id
        or str(
            meta.get("rollout_id")
            or meta.get("checkpoint_id")
            or meta.get("checkpoint")
            or "cooperation-eval"
        ),
        "rollout_id",
    )
    plan = build_cooperation_plan(
        manifest_path,
        context_samples=context_samples,
        prefilled_think=resolved_config.prefilled_think,
        trained_game_ids=trained_game_ids,
        rollout_id=resolved_rollout_id,
    )
    cell_meta = {
        **dict(meta),
        "cooperation_manifest_path": str(manifest_path),
        "cooperation_manifest_digest": _manifest_digest(manifest_path),
        "cooperation_plan_counts": plan.counts,
        "cooperation_scenario_group_ids": sorted(plan.roster.scenario_group_ids),
        "cooperation_rollout_id": resolved_rollout_id,
    }
    return _run_planned_cell(
        backend,
        plan=plan.requests,
        sections=plan.sections,
        out_path=out_path,
        meta=cell_meta,
        config=resolved_config,
        resume=resume,
        submission=submission,
        admission=admission,
        on_records_written=on_records_written,
    )


def _survey_cell_config(  # noqa: PLR0913
    *,
    data_dir: Path,
    families: tuple[str, ...],
    instruments: tuple[str, ...],
    tier: str,
    samples: int,
    prefilled_think: bool,
) -> EvalConfig:
    if samples != DEFAULT_REQUIRED_SAMPLES:
        raise ValueError("the required survey cells are frozen at two samples per order.")
    return EvalConfig(
        survey_samples=samples,
        survey_data_dir=data_dir,
        survey_families=families,
        survey_instruments=instruments,
        survey_tier=tier,
        prefilled_think=prefilled_think,
    )


def build_core_survey_plan(
    *, data_dir: Path, samples: int = 2, prefilled_think: bool = True
) -> list[PlannedRequest]:
    """Render the complete frozen core survey cell and validate its runtime membership."""
    counts = survey_leg_counts("core", data_dir=data_dir)
    config = _survey_cell_config(
        data_dir=data_dir,
        families=counts.families,
        instruments=counts.instruments,
        tier="core",
        samples=samples,
        prefilled_think=prefilled_think,
    )
    return plan_battery((SECTION_SELF_REPORT,), config)


def build_prosocialness_plan(
    *, data_dir: Path, samples: int = 2, prefilled_think: bool = True
) -> list[PlannedRequest]:
    """Render the complete prosocialness cell separately from the historical core tier."""
    counts = survey_leg_counts("prosocialness", data_dir=data_dir)
    config = _survey_cell_config(
        data_dir=data_dir,
        families=counts.families,
        instruments=counts.instruments,
        tier="",
        samples=samples,
        prefilled_think=prefilled_think,
    )
    return plan_battery((SECTION_SELF_REPORT,), config)


def build_local_dt_plan(
    *, multiple_choice_samples: int = 2, open_ended_samples: int = 2, prefilled_think: bool = True
) -> list[PlannedRequest]:
    """Render exactly our local 30-item DT battery, without optional DTBench."""
    local_dt_counts(
        multiple_choice_samples=multiple_choice_samples, open_ended_samples=open_ended_samples
    )
    config = EvalConfig(
        multiple_choice_samples=multiple_choice_samples,
        open_ended_samples=open_ended_samples,
        dtbench_dir=None,
        prefilled_think=prefilled_think,
    )
    return plan_battery((SECTION_DT_PROBES,), config)


def run_core_survey_eval(  # noqa: PLR0913
    backend: Backend,
    *,
    data_dir: Path,
    out_path: Path,
    meta: Mapping[str, Any],
    resume: bool = False,
    prefilled_think: bool = True,
    submission: str = SUBMISSION_POOLED,
    admission: str = ADMISSION_LONGEST_FIRST,
) -> dict[str, Any]:
    """Execute the complete core survey through the native resumable evaluator."""
    plan = build_core_survey_plan(data_dir=data_dir, prefilled_think=prefilled_think)
    counts = survey_leg_counts("core", data_dir=data_dir)
    config = _survey_cell_config(
        data_dir=data_dir,
        families=counts.families,
        instruments=counts.instruments,
        tier="core",
        samples=2,
        prefilled_think=prefilled_think,
    )
    # The supplied plan contains the exact filtered items; config carries only runtime sampler and
    # path provenance here. The dedicated count validation happened before rendering.
    return _run_planned_cell(
        backend,
        plan=plan,
        sections=(SECTION_SELF_REPORT,),
        out_path=out_path,
        meta={**dict(meta), "survey_cell": "core", "survey_data_dir": str(data_dir)},
        config=config,
        resume=resume,
        submission=submission,
        admission=admission,
    )


def run_prosocialness_eval(  # noqa: PLR0913
    backend: Backend,
    *,
    data_dir: Path,
    out_path: Path,
    meta: Mapping[str, Any],
    resume: bool = False,
    prefilled_think: bool = True,
    submission: str = SUBMISSION_POOLED,
    admission: str = ADMISSION_LONGEST_FIRST,
) -> dict[str, Any]:
    """Execute the complete prosocialness extension through the native resumable evaluator."""
    plan = build_prosocialness_plan(data_dir=data_dir, prefilled_think=prefilled_think)
    counts = survey_leg_counts("prosocialness", data_dir=data_dir)
    config = _survey_cell_config(
        data_dir=data_dir,
        families=counts.families,
        instruments=counts.instruments,
        tier="",
        samples=2,
        prefilled_think=prefilled_think,
    )
    return _run_planned_cell(
        backend,
        plan=plan,
        sections=(SECTION_SELF_REPORT,),
        out_path=out_path,
        meta={**dict(meta), "survey_cell": "prosocialness", "survey_data_dir": str(data_dir)},
        config=config,
        resume=resume,
        submission=submission,
        admission=admission,
    )


def run_local_dt_eval(  # noqa: PLR0913
    backend: Backend,
    *,
    out_path: Path,
    meta: Mapping[str, Any],
    resume: bool = False,
    prefilled_think: bool = True,
    submission: str = SUBMISSION_POOLED,
    admission: str = ADMISSION_LONGEST_FIRST,
) -> dict[str, Any]:
    """Execute the complete local DT battery through the native resumable evaluator."""
    plan = build_local_dt_plan(prefilled_think=prefilled_think)
    config = EvalConfig(
        multiple_choice_samples=2,
        open_ended_samples=2,
        dtbench_dir=None,
        prefilled_think=prefilled_think,
    )
    return _run_planned_cell(
        backend,
        plan=plan,
        sections=(SECTION_DT_PROBES,),
        out_path=out_path,
        meta={**dict(meta), "dt_cell": "our_battery", "dtbench_included": False},
        config=config,
        resume=resume,
        submission=submission,
        admission=admission,
    )


@dataclass(frozen=True, slots=True)
class SurveyLegCounts:
    """Runtime counts for one frozen survey cell."""

    name: str
    n_items: int
    n_renderings: int
    n_responses: int
    families: tuple[str, ...]
    instruments: tuple[str, ...]


def survey_leg_counts(name: str, *, data_dir: Path) -> SurveyLegCounts:
    """Compute and validate the two required runtime survey cells."""
    if name == "core":
        families = (
            FAMILY_NEGATIVE_CONTROL,
            FAMILY_SELF_PREDICTION,
            FAMILY_SVO_ALLOCATION,
            FAMILY_TRIPLE_DOMINANCE,
            FAMILY_COMPETITIVENESS,
            FAMILY_NARCISSISM,
        )
        instruments = tuple(PUBLISHED_INSTRUMENTS)
        tier = "core"
        expected_items, expected_responses = EXPECTED_CORE_ITEMS, EXPECTED_CORE_RESPONSES
    elif name == "prosocialness":
        families = (FAMILY_PROSOCIALNESS,)
        instruments = (INSTRUMENT_PROSOCIALNESS,)
        tier = ""
        expected_items, expected_responses = (
            EXPECTED_PROSOCIALNESS_ITEMS,
            EXPECTED_PROSOCIALNESS_RESPONSES,
        )
    else:
        raise ValueError("survey leg must be 'core' or 'prosocialness'.")
    items = survey_battery(
        data_dir=data_dir,
        families=families,
        instruments=instruments,
        tier=tier,
    )
    renderings = sum(len(battery_orders(item) or (("not-applicable", ()),)) for item in items)
    responses = renderings * 2
    if len(items) != expected_items or responses != expected_responses:
        raise ValueError(
            f"{name} survey membership drifted: n_items={len(items)}, n_responses={responses}; "
            f"expected {expected_items} items and {expected_responses} responses."
        )
    return SurveyLegCounts(name, len(items), renderings, responses, families, instruments)


def local_dt_counts(
    *, multiple_choice_samples: int = 2, open_ended_samples: int = 2
) -> dict[str, int]:
    """Return exact membership and rendered response counts for our local DT battery."""
    items = our_battery()
    open_ended_ids = {item.probe_id for item in OPEN_ENDED_ITEMS}
    multiple_choice = sum(item.probe_id not in open_ended_ids for item in items)
    if (
        len(items) != EXPECTED_LOCAL_DT_ITEMS
        or multiple_choice != EXPECTED_LOCAL_DT_MULTIPLE_CHOICE
        or len(OPEN_ENDED_ITEMS) != EXPECTED_LOCAL_DT_OPEN_ENDED
    ):
        raise ValueError(
            f"local DT battery membership drifted: items={len(items)}, multiple_choice={multiple_choice}, "
            f"open_ended={len(OPEN_ENDED_ITEMS)}; expected {EXPECTED_LOCAL_DT_ITEMS}/"
            f"{EXPECTED_LOCAL_DT_MULTIPLE_CHOICE}/{EXPECTED_LOCAL_DT_OPEN_ENDED}."
        )
    choice_renderings = sum(
        len(counterbalanced_option_orders(len(item.options)))
        for item in items
        if item.probe_id not in open_ended_ids
    )
    open_renderings = len(OPEN_ENDED_ITEMS)
    return {
        "n_items": len(items),
        "n_multiple_choice": multiple_choice,
        "n_open_ended": len(OPEN_ENDED_ITEMS),
        "n_choice_responses": choice_renderings * multiple_choice_samples,
        "n_open_responses": open_renderings * open_ended_samples,
        "n_responses": choice_renderings * multiple_choice_samples
        + open_renderings * open_ended_samples,
    }


def print_plan(path: Path = DEFAULT_BEHAVIOR_MANIFEST_PATH) -> dict[str, Any]:
    """Return the CPU-only manifest/count payload used by the sequence runner."""
    roster = expand_behavior_roster(load_behavior_manifest(path))
    validate_experiment_coverage(roster)
    forecasts, normative = forecast_and_normative_plans(roster, samples=2)
    allocation = build_allocation_plan(roster)
    allocation_diagnostic = roster.manifest.allocation
    if allocation_diagnostic is None:
        raise RuntimeError("validated cooperation roster lost its allocation diagnostic.")
    allocation_orders = allocation_option_orders(len(allocation_diagnostic.option_payoffs))
    allocation_option_positions = {
        str(option): {order_name: order.index(option) for order_name, order in allocation_orders}
        for option in range(len(allocation_diagnostic.option_payoffs))
    }
    return {
        "behavior": {
            "n_pairs": roster.counts.n_pairs,
            "n_rendered_prompts": roster.counts.n_rendered_prompts,
            "n_completions": roster.counts.n_completions,
            "by_family": dict(roster.counts.by_family),
            "by_print_order": dict(roster.counts.by_print_order),
        },
        "exact_context": {
            "forecast_requests": len(forecasts),
            "normative_requests": len(normative),
        },
        "allocation": {
            "rendered_prompts": len(allocation) // allocation_diagnostic.samples,
            "responses": len(allocation),
            "option_positions": allocation_option_positions,
        },
        "scenario_group_ids": sorted(roster.scenario_group_ids),
    }


def main() -> None:
    """Print the private roster counts without constructing a model backend."""
    parser = argparse.ArgumentParser(
        description="Print the cooperation evaluation plan without loading a model."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_BEHAVIOR_MANIFEST_PATH)
    args = parser.parse_args()
    print(json.dumps(print_plan(args.manifest), indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
