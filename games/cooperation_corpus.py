"""Build and audit the small three-family cooperation-generalization corpus.

This module owns the deliberately small training treatment described in the 2026-09-12 plan. It
calls the existing prompt renderers and production reward function; it does not carry authored
scenario or counterpart text. Any runtime authored framing file is loaded from the private scratch
directory by the caller and only its digest reaches the manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from games.arms import ARMS, arm_game_ids
from games.framing_stimulus import RuntimeFramings, load_framings, resolve_framing_clause
from games.prompts import (
    COUNTERPART_FRAMINGS,
    FRAMING_HUMAN,
    FRAMING_TWIN,
    FRAMING_UNSTATED,
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDER_SWAPPED,
    SPLIT_EVAL,
    SPLIT_TRAIN,
    TRUST_PAYOFF_VARIANTS,
    generate_prompt_rows,
    matrix_frames_for_split,
    render_matrix_rows_under_clause,
)
from games.rewards import (
    FRAMING_ID_COLUMN,
    FRAMING_ID_UNSET,
    PARSE_PENALTY_MARGIN_BELOW_WORSE,
    care_grading,
    make_game_reward,
)
from games.select_prompts import read_jsonl, rows_file_digest, write_corpus

type Row = dict[str, Any]

logger = logging.getLogger(__name__)

ARM_NAME = "cooperation-generalization-care-alpha-1"
GRADING = care_grading(1)
PRIVATE_ROOT = Path("docs/scratch/cooperation-generalization")
DEFAULT_CORPUS_PATH = Path("artifacts/games/cooperation-generalization/training-corpus.jsonl")
DEFAULT_MANIFEST_PATH = PRIVATE_ROOT / "training-corpus-manifest.json"
DEFAULT_AUDIT_PATH = PRIVATE_ROOT / "learning-signal-audit.json"

PD_FAMILY = "pd"
STAG_HUNT_FAMILY = "stag-hunt"
TRUST_FAMILY = "trust"
MATRIX_FAMILIES = frozenset({PD_FAMILY, STAG_HUNT_FAMILY})

# These interior mixes expose the disagreement surface while keeping both actions in the synthetic
# group. Pure matching groups are covered by the separate stratum audit, which retains them rather
# than silently filtering them. Whether a particular table ties at one of these realised mixes is
# reported as observed; the audit does not manufacture a tie by changing the table.
MIX_POINTS: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 0.875)
MATRIX_GROUP_SIZE = 8
MAPPING_PAIR_ROWS = 2
EXPECTED_MATRIX_STRATA = 8
EXPECTED_TRUST_STRATA = 2
MIN_SCORE_COUNT = 2
NORMALIZATION_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class MatrixCell:
    """One selected (family, game, payoff, framing) stratum."""

    family: str
    game_id: str
    payoff_variant: str
    framing_id: str
    n_scenarios: int = 1
    scenario_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate frozen membership before the spec can reach a renderer."""
        if self.family not in MATRIX_FAMILIES:
            raise ValueError(
                f"unknown matrix family {self.family!r}; expected {sorted(MATRIX_FAMILIES)}"
            )
        if self.n_scenarios < 1:
            raise ValueError(f"n_scenarios must be positive, got {self.n_scenarios}")
        if self.scenario_ids and len(self.scenario_ids) != self.n_scenarios:
            raise ValueError(
                f"cell {self.key!r} names {len(self.scenario_ids)} scenario_ids but requests "
                f"{self.n_scenarios}; frozen membership must be complete"
            )
        if len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError(f"cell {self.key!r} repeats a scenario id")

    @property
    def key(self) -> str:
        """Return a stable human-readable cell identity for manifests and errors."""
        return f"{self.family}/{self.game_id}/{self.payoff_variant}/{self.framing_id}"

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize this cell without any rendered prompt text."""
        return {
            "family": self.family,
            "game_id": self.game_id,
            "payoff_variant": self.payoff_variant,
            "framing_id": self.framing_id,
            "n_scenarios": self.n_scenarios,
            "scenario_ids": list(self.scenario_ids),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> MatrixCell:
        """Rebuild a cell from a text-free manifest payload."""
        required = ("family", "game_id", "payoff_variant", "framing_id")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"matrix cell lacks {missing}; keys present: {sorted(payload)}")
        scenario_ids = tuple(str(value) for value in payload.get("scenario_ids", ()))
        return cls(
            family=str(payload["family"]),
            game_id=str(payload["game_id"]),
            payoff_variant=str(payload["payoff_variant"]),
            framing_id=str(payload["framing_id"]),
            n_scenarios=int(payload.get("n_scenarios", len(scenario_ids) or 1)),
            scenario_ids=scenario_ids,
        )


@dataclass(frozen=True, slots=True)
class CooperationCorpusSpec:
    """The small, freezeable membership specification for one treatment."""

    matrix_cells: tuple[MatrixCell, ...]
    trust_scenario_count: int = 8
    trust_scenario_ids: tuple[str, ...] = ()
    trust_payoff_variants: tuple[str, ...] = TRUST_PAYOFF_VARIANTS
    arm_name: str = ARM_NAME
    grading: str = GRADING

    def __post_init__(self) -> None:
        """Validate corpus-level counts, variants and frozen scenario membership."""
        if not self.matrix_cells and self.trust_scenario_count == 0:
            raise ValueError("a cooperation corpus needs a matrix cell or trust rows")
        if self.trust_scenario_count < 0:
            raise ValueError(
                f"trust_scenario_count must be non-negative, got {self.trust_scenario_count}"
            )
        if self.trust_scenario_ids and len(self.trust_scenario_ids) != self.trust_scenario_count:
            raise ValueError(
                f"trust membership names {len(self.trust_scenario_ids)} scenario_ids but requests "
                f"{self.trust_scenario_count}; frozen membership must be complete"
            )
        if len(set(self.trust_scenario_ids)) != len(self.trust_scenario_ids):
            raise ValueError("trust_scenario_ids repeats a scenario id")
        if not self.trust_payoff_variants and self.trust_scenario_count:
            raise ValueError("trust rows need at least one payoff variant")
        unknown = set(self.trust_payoff_variants) - set(TRUST_PAYOFF_VARIANTS)
        if unknown:
            raise ValueError(
                f"unknown trust payoff variants {sorted(unknown)}; known {TRUST_PAYOFF_VARIANTS}"
            )

    @classmethod
    def default(cls) -> CooperationCorpusSpec:
        """Return the planned four-cell-per-matrix-family design."""
        return cls(
            matrix_cells=(
                MatrixCell(PD_FAMILY, "twin-pd", "temptation-2", FRAMING_TWIN),
                MatrixCell(PD_FAMILY, "twin-pd", "temptation-10", FRAMING_HUMAN),
                MatrixCell(PD_FAMILY, "twin-pd", "temptation-2", FRAMING_UNSTATED),
                MatrixCell(PD_FAMILY, "twin-pd", "temptation-10", FRAMING_TWIN),
                MatrixCell(STAG_HUNT_FAMILY, "stag-hunt", "favoured-hunt", FRAMING_TWIN),
                MatrixCell(STAG_HUNT_FAMILY, "stag-hunt", "safe-hunt", FRAMING_HUMAN),
                MatrixCell(STAG_HUNT_FAMILY, "stag-hunt", "favoured-hunt", FRAMING_UNSTATED),
                MatrixCell(STAG_HUNT_FAMILY, "stag-hunt", "safe-hunt", FRAMING_TWIN),
            ),
            trust_scenario_count=8,
            trust_payoff_variants=("return-fifth", "return-half"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize this spec without any rendered prompt text."""
        return {
            "matrix_cells": [cell.to_json_dict() for cell in self.matrix_cells],
            "trust_scenario_count": self.trust_scenario_count,
            "trust_scenario_ids": list(self.trust_scenario_ids),
            "trust_payoff_variants": list(self.trust_payoff_variants),
            "arm_name": self.arm_name,
            "grading": self.grading,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> CooperationCorpusSpec:
        """Rebuild a corpus spec from its text-free manifest payload."""
        if "matrix_cells" not in payload:
            raise ValueError(
                f"cooperation corpus spec lacks ['matrix_cells']; keys present: {sorted(payload)}"
            )
        return cls(
            matrix_cells=tuple(
                MatrixCell.from_json_dict(dict(item)) for item in payload["matrix_cells"]
            ),
            trust_scenario_count=int(payload.get("trust_scenario_count", 8)),
            trust_scenario_ids=tuple(str(value) for value in payload.get("trust_scenario_ids", ())),
            trust_payoff_variants=tuple(
                str(value) for value in payload.get("trust_payoff_variants", TRUST_PAYOFF_VARIANTS)
            ),
            arm_name=str(payload.get("arm_name", ARM_NAME)),
            grading=str(payload.get("grading", GRADING)),
        )


DEFAULT_CORPUS_SPEC = CooperationCorpusSpec.default()


@dataclass(frozen=True, slots=True)
class BuiltCooperationCorpus:
    """Rendered rows plus the resolved private membership manifest."""

    rows: tuple[Row, ...]
    spec: CooperationCorpusSpec
    rows_by_family: Mapping[str, int]
    manifest: Mapping[str, Any]


def _identity_digest(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _select_matrix_scenarios(
    cell: MatrixCell, *, used_scenario_ids: set[tuple[str, str]]
) -> tuple[Any, ...]:
    roster = matrix_frames_for_split(cell.game_id, SPLIT_TRAIN)
    by_id = {str(scenario.scenario_id): scenario for scenario in roster}
    if cell.scenario_ids:
        missing = sorted(set(cell.scenario_ids) - set(by_id))
        if missing:
            raise ValueError(
                f"cell {cell.key!r} names scenario_ids {missing} absent from its training roster; "
                "eval or another game's rows cannot enter this cell"
            )
        selected = tuple(by_id[scenario_id] for scenario_id in cell.scenario_ids)
    else:
        available = [
            scenario
            for scenario in roster
            if (cell.game_id, str(scenario.scenario_id)) not in used_scenario_ids
        ]
        selected = tuple(
            sorted(
                available,
                key=lambda scenario: _identity_digest(cell.key, scenario.scenario_id),
            )[: cell.n_scenarios]
        )
    if len(selected) != cell.n_scenarios:
        raise ValueError(
            f"cell {cell.key!r} requests {cell.n_scenarios} distinct training scenarios but only "
            f"{len(selected)} remain; lower the cell or fix its frozen scenario_ids"
        )
    selected_keys = {(cell.game_id, str(scenario.scenario_id)) for scenario in selected}
    overlap = sorted(selected_keys & used_scenario_ids)
    if overlap:
        raise ValueError(
            f"scenario groups {overlap} occur in more than one matrix cell; a cell must identify "
            "one distinct scenario/template group"
        )
    used_scenario_ids.update(selected_keys)
    return selected


def _select_trust_scenario_ids(spec: CooperationCorpusSpec) -> tuple[str, ...]:
    rows = generate_prompt_rows("trust-vs-stated-return", spec.grading, split=SPLIT_TRAIN)
    available = tuple(dict.fromkeys(str(row["reskin_id"]) for row in rows))
    available_ids = set(available)
    if spec.trust_scenario_ids:
        missing = sorted(set(spec.trust_scenario_ids) - available_ids)
        if missing:
            raise ValueError(
                f"trust scenario_ids {missing} are absent from the training roster; eval rows or "
                "another trust form cannot enter the sender corpus"
            )
        return spec.trust_scenario_ids
    selected = tuple(
        sorted(available, key=lambda scenario_id: _identity_digest("trust", scenario_id))[
            : spec.trust_scenario_count
        ]
    )
    if len(selected) != spec.trust_scenario_count:
        raise ValueError(
            f"trust requests {spec.trust_scenario_count} scenarios but the training roster has "
            f"only {len(selected)}"
        )
    return selected


def _resolve_clause(framing_id: str, runtime_framings: RuntimeFramings | None) -> str | None:
    if framing_id in COUNTERPART_FRAMINGS:
        return COUNTERPART_FRAMINGS[framing_id]
    if runtime_framings is None:
        raise ValueError(
            f"{framing_id!r} is a runtime framing but no runtime framing file was supplied; "
            "keep authored stimulus under docs/scratch/cooperation-generalization/"
        )
    return resolve_framing_clause(framing_id, runtime_framings)


def _with_framing_marker(rows: Iterable[Row], framing_id: str) -> list[Row]:
    return [{**row, FRAMING_ID_COLUMN: framing_id} for row in rows]


def _matrix_print_order(row: Mapping[str, Any]) -> str:
    prompt_id = str(row["prompt_id"])
    order = label_print_order_of(row)
    if order not in {LABEL_PRINT_ORDER_CANONICAL, LABEL_PRINT_ORDER_SWAPPED}:
        raise ValueError(f"matrix row {prompt_id!r} has unknown printed position {order!r}")
    return order


def _mapping_pair_indices(
    key: tuple[str, str, str, str, str], members: Sequence[Mapping[str, Any]]
) -> set[int]:
    mappings: set[int] = set()
    for row in members:
        coop_label = str(row["coop_label"])
        labels = {str(row["label_a"]), str(row["label_b"])}
        if coop_label not in labels:
            raise ValueError(f"matrix unit {key} has a coop label outside its printed labels")
        mappings.add(0 if coop_label == str(row["label_a"]) else 1)
    return mappings


def assert_matrix_balance(rows: Sequence[Mapping[str, Any]]) -> None:
    """Require both label mappings at both printed positions for every matrix unit."""
    groups: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get(FRAMING_ID_COLUMN) == FRAMING_ID_UNSET:
            continue
        order = _matrix_print_order(row)
        key = (
            str(row["game_id"]),
            str(row["reskin_id"]),
            str(row["payoff_variant"]),
            str(row[FRAMING_ID_COLUMN]),
            order,
        )
        groups.setdefault(key, []).append(row)
    if not groups:
        raise ValueError("the cooperation corpus contains no matrix rows")
    for key, members in groups.items():
        if len(members) != MAPPING_PAIR_ROWS:
            raise ValueError(
                f"matrix unit {key} has {len(members)} rows; expected a whole mapping pair"
            )
        mappings = _mapping_pair_indices(key, members)
        if mappings != {0, 1}:
            raise ValueError(
                f"matrix unit {key} does not contain both label mappings: {sorted(mappings)}"
            )
    identities: dict[tuple[str, str, str, str], set[str]] = {}
    for key in groups:
        identities.setdefault(key[:-1], set()).add(key[-1])
    missing_orders = sorted(
        identity
        for identity, orders in identities.items()
        if orders != {LABEL_PRINT_ORDER_CANONICAL, LABEL_PRINT_ORDER_SWAPPED}
    )
    if missing_orders:
        raise ValueError(f"matrix units missing a printed position: {missing_orders}")


def label_print_order_of(row: Mapping[str, Any]) -> str:
    """Return the renderer's explicit printed-position marker for a row."""
    if "label_print_order" not in row:
        raise ValueError("cooperation row lacks label_print_order")
    return str(row["label_print_order"])


def scenario_group_ids(rows: Sequence[Mapping[str, Any]]) -> frozenset[tuple[str, str]]:
    """Return machine-readable (game, scenario/template) groups for split isolation."""
    if any("game_id" not in row or "reskin_id" not in row for row in rows):
        raise ValueError("every cooperation row needs game_id and reskin_id for group isolation")
    return frozenset((str(row["game_id"]), str(row["reskin_id"])) for row in rows)


def validate_train_eval_isolation(
    train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]]
) -> None:
    """Refuse prompt or scenario groups reused across training and evaluation."""
    train_ids = {str(row["prompt_id"]) for row in train_rows}
    eval_ids = {str(row["prompt_id"]) for row in eval_rows}
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        raise ValueError(f"train/eval prompt_id overlap: {overlap[:5]}")
    group_overlap = sorted(scenario_group_ids(train_rows) & scenario_group_ids(eval_rows))
    if group_overlap:
        raise ValueError(
            f"train/eval scenario groups overlap: {group_overlap[:5]}; split by whole template group"
        )


def build_eval_identity_rows() -> list[Row]:
    """Render the reserved eval identities used by offline isolation checks."""
    rows: list[Row] = []
    for game_id in ("twin-pd", "stag-hunt"):
        rows.extend(
            _with_framing_marker(
                generate_prompt_rows(game_id, GRADING, split=SPLIT_EVAL), FRAMING_UNSTATED
            )
        )
    rows.extend(
        _with_framing_marker(
            generate_prompt_rows("trust-vs-stated-return", GRADING, split=SPLIT_EVAL),
            FRAMING_ID_UNSET,
        )
    )
    return rows


def _build_rows(
    spec: CooperationCorpusSpec, runtime_framings: RuntimeFramings | None
) -> tuple[list[Row], CooperationCorpusSpec]:
    rows: list[Row] = []
    used: set[tuple[str, str]] = set()
    resolved_cells: list[MatrixCell] = []
    for cell in spec.matrix_cells:
        scenarios = _select_matrix_scenarios(cell, used_scenario_ids=used)
        clause = _resolve_clause(cell.framing_id, runtime_framings)
        for order in (LABEL_PRINT_ORDER_CANONICAL, LABEL_PRINT_ORDER_SWAPPED):
            rows.extend(
                row
                for row in render_matrix_rows_under_clause(
                    cell.game_id,
                    spec.grading,
                    clause=clause,
                    framing_label=cell.framing_id,
                    split=SPLIT_TRAIN,
                    label_print_order=order,
                    scenarios=scenarios,
                )
                if str(row["payoff_variant"]) == cell.payoff_variant
            )
        resolved_cells.append(
            dataclasses.replace(
                cell,
                scenario_ids=tuple(str(scenario.scenario_id) for scenario in scenarios),
            )
        )

    trust_ids = _select_trust_scenario_ids(spec)
    if spec.trust_scenario_count:
        trust_rows = generate_prompt_rows("trust-vs-stated-return", spec.grading, split=SPLIT_TRAIN)
        selected_trust_rows = [
            row
            for row in trust_rows
            if str(row["reskin_id"]) in trust_ids
            and str(row["payoff_variant"]) in spec.trust_payoff_variants
        ]
        rows.extend(_with_framing_marker(selected_trust_rows, FRAMING_ID_UNSET))
    resolved = dataclasses.replace(
        spec,
        matrix_cells=tuple(resolved_cells),
        trust_scenario_ids=trust_ids,
    )
    return rows, resolved


def build_training_corpus(
    spec: CooperationCorpusSpec = DEFAULT_CORPUS_SPEC,
    *,
    runtime_framings: RuntimeFramings | None = None,
) -> BuiltCooperationCorpus:
    """Render and validate one small training corpus entirely on CPU."""
    if spec.arm_name not in ARMS:
        raise ValueError(f"unknown cooperation arm {spec.arm_name!r}")
    arm = ARMS[spec.arm_name]
    if arm.grading != spec.grading:
        raise ValueError(
            f"spec grading {spec.grading!r} disagrees with arm {spec.arm_name!r}'s {arm.grading!r}"
        )
    allowed_games = set(arm_game_ids(arm))
    rows, resolved_spec = _build_rows(spec, runtime_framings)
    if not rows:
        raise ValueError("cooperation corpus rendered no rows")
    unknown_games = sorted({str(row["game_id"]) for row in rows} - allowed_games)
    if unknown_games:
        raise ValueError(f"corpus has games outside arm {spec.arm_name!r}: {unknown_games}")
    prompt_ids = [str(row["prompt_id"]) for row in rows]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("cooperation corpus contains duplicate prompt_ids")
    schemas = {frozenset(row) for row in rows}
    if len(schemas) != 1:
        raise ValueError("cooperation corpus rows have different schemas")
    assert_matrix_balance(rows)
    rows_by_family = {
        PD_FAMILY: sum(row["game_id"] == "twin-pd" for row in rows),
        STAG_HUNT_FAMILY: sum(row["game_id"] == "stag-hunt" for row in rows),
        TRUST_FAMILY: sum(row["game_id"] == "trust-vs-stated-return" for row in rows),
    }
    expected = {PD_FAMILY: 16, STAG_HUNT_FAMILY: 16, TRUST_FAMILY: 16}
    if rows_by_family != expected:
        raise ValueError(
            f"cooperation corpus realised {rows_by_family}; expected exactly {expected}. "
            "Adjust the private membership spec rather than silently changing family weights."
        )
    training_groups = scenario_group_ids(rows)
    eval_groups = scenario_group_ids(build_eval_identity_rows())
    if training_groups & eval_groups:
        raise ValueError(
            f"default train/eval scenario groups overlap: {sorted(training_groups & eval_groups)[:5]}"
        )
    manifest: dict[str, Any] = {
        "version": "cooperation-generalization-corpus-v1",
        "arm_name": resolved_spec.arm_name,
        "grading": resolved_spec.grading,
        "row_count": len(rows),
        "rows_by_family": rows_by_family,
        "spec": resolved_spec.to_json_dict(),
        "selected_scenario_ids": {
            cell.key: list(cell.scenario_ids) for cell in resolved_spec.matrix_cells
        },
        "selected_trust_scenario_ids": list(resolved_spec.trust_scenario_ids),
        "training_group_ids": sorted([list(group) for group in training_groups]),
        "eval_reserved_group_ids": sorted([list(group) for group in eval_groups]),
        "runtime_framings_digest": None if runtime_framings is None else runtime_framings.digest,
    }
    return BuiltCooperationCorpus(
        rows=tuple(rows),
        spec=resolved_spec,
        rows_by_family=rows_by_family,
        manifest=manifest,
    )


def _columns_for(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    keys = frozenset(rows[0])
    if any(frozenset(row) != keys for row in rows):
        raise ValueError("scorer fixture rows have different schemas")
    return {key: [row[key] for row in rows] for key in keys}


def _score(row: Mapping[str, Any], completions: Sequence[str], *, group_size: int) -> list[float]:
    scorer = make_game_reward(
        group_size,
        prefilled_think=False,
        parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE,
    )
    metrics: dict[str, float] = {}
    extras: dict[str, list[Any]] = {}
    return scorer(
        completions=list(completions),
        log_metric=metrics.__setitem__,
        log_extra=extras.__setitem__,
        **_columns_for([row] * len(completions)),
    )


def _action_completion(row: Mapping[str, Any], *, cooperate: bool) -> str:
    coop_label = str(row["coop_label"])
    label_a = str(row["label_a"])
    label_b = str(row["label_b"])
    label = coop_label if cooperate else label_b if coop_label == label_a else label_a
    return f"<action>{label}</action>"


def _unique_representatives(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    representatives: list[Mapping[str, Any]] = []
    for row in rows:
        key = (
            str(row["game_id"]),
            str(row["payoff_variant"]),
            str(row[FRAMING_ID_COLUMN]),
            str(row.get("grading", "")),
        )
        if key not in seen:
            seen.add(key)
            representatives.append(row)
    return representatives


@dataclass(frozen=True, slots=True)
class MatrixRewardAudit:
    """One production-scorer matrix ranking at one realised opponent mix."""

    game_id: str
    payoff_variant: str
    framing_id: str
    mix: float
    cooperative_reward: float
    defective_reward: float
    own_payoff_cooperative: float
    own_payoff_defective: float
    welfare_cooperative: float
    welfare_defective: float
    own_payoff_ranking: str
    welfare_ranking: str
    parse_group_cooperative_reward: float
    parse_group_defective_reward: float
    ranking: str
    tie: bool
    parse_price: float
    parse_below_worst: bool
    normalized: bool

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize this scorer audit record."""
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class TrustSlopeAudit:
    """Production-scorer slope between sending none and the full endowment."""

    payoff_variant: str
    sent_none_reward: float
    sent_all_reward: float
    parse_group_sent_none_reward: float
    parse_group_sent_all_reward: float
    slope: float
    ranking: str
    parse_price: float
    parse_below_worst: bool
    normalized: bool

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize this scorer audit record."""
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class StratumAudit:
    """Synthetic matching-group scorer audit for one proposed training stratum."""

    family: str
    key: str
    n_groups_considered: int
    n_pure_groups: int
    reward_variance_mean: float
    reward_variance_min: float
    reward_variance_max: float
    synthetic_parseable_rate: float

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize this synthetic scorer audit record."""
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class LearningSignalAudit:
    """Offline production-scorer audit for the complete three-family corpus."""

    mix_points: tuple[float, ...]
    matrix_rankings: tuple[MatrixRewardAudit, ...]
    trust_slopes: tuple[TrustSlopeAudit, ...]
    strata: tuple[StratumAudit, ...]
    matrix_normalized: bool
    trust_normalized: bool

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the audit and its explicit synthetic-fixture qualification."""
        return {
            "mix_points": list(self.mix_points),
            "matrix_rankings": [record.to_json_dict() for record in self.matrix_rankings],
            "trust_slopes": [record.to_json_dict() for record in self.trust_slopes],
            "strata": [record.to_json_dict() for record in self.strata],
            "matrix_normalized": self.matrix_normalized,
            "trust_normalized": self.trust_normalized,
            "synthetic_fixture_note": (
                "The stratum rates use enumerated canonical completions to audit scorer plumbing; "
                "they are not model termination evidence."
            ),
        }


@dataclass(frozen=True, slots=True)
class SweepStratumAudit:
    """Observed signal from every prompt in one real, identity-checked sweep stratum."""

    family: str
    key: str
    n_prompts: int
    n_samples: int
    n_parse_failures: int
    n_truncated_thinking: int
    parseable_rate: float
    termination_rate: float
    reward_variance_mean: float
    reward_variance_min: float
    reward_variance_max: float
    n_pure_prompts: int

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize this post-screen stratum summary."""
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class SweepTraceAudit:
    """Summary of a complete prompt sweep after provenance and identity validation."""

    model_id: str
    sampler_identity: Mapping[str, Any]
    prompt_id_order_sha256: str
    n_prompts: int
    n_samples: int
    n_parse_failures: int
    n_truncated_thinking: int
    parseable_rate: float
    termination_rate: float
    strata: tuple[SweepStratumAudit, ...]

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize this provenance-checked sweep summary without prompt text."""
        return {
            "model_id": self.model_id,
            "sampler_identity": dict(self.sampler_identity),
            "prompt_id_order_sha256": self.prompt_id_order_sha256,
            "n_prompts": self.n_prompts,
            "n_samples": self.n_samples,
            "n_parse_failures": self.n_parse_failures,
            "n_truncated_thinking": self.n_truncated_thinking,
            "parseable_rate": self.parseable_rate,
            "termination_rate": self.termination_rate,
            "strata": [stratum.to_json_dict() for stratum in self.strata],
            "pure_group_note": (
                "Pure prompt groups are retained in the counts and variance summary; they are not "
                "silently filtered from this post-screen audit."
            ),
        }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stratum_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    game_id = str(row.get("game_id", ""))
    family = {
        "twin-pd": PD_FAMILY,
        "stag-hunt": STAG_HUNT_FAMILY,
        "trust-vs-stated-return": TRUST_FAMILY,
    }.get(game_id)
    if family is None:
        raise ValueError(f"sweep row has unsupported cooperation game {game_id!r}")
    for column in ("payoff_variant", FRAMING_ID_COLUMN):
        if column not in row:
            raise ValueError(f"sweep row lacks {column!r}, required for stratum accounting")
    return family, str(row["payoff_variant"]), str(row[FRAMING_ID_COLUMN])


def expected_stratum_keys(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    """Return all proposed training strata, including pure ones, in stable order."""
    return tuple(sorted({_stratum_key(row) for row in _unique_representatives(rows)}))


def _trace_entries(trace: Path | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(trace, Path):
        return [dict(entry) for entry in read_jsonl(trace)]
    return [dict(entry) for entry in trace]


def _split_trace_entries(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metas = [dict(entry) for entry in entries if entry.get("record_kind") == "sweep-meta"]
    if len(metas) != 1:
        raise ValueError(f"expected exactly one sweep-meta record, found {len(metas)}")
    records = [dict(entry) for entry in entries if entry.get("record_kind") == "prompt-sweep"]
    unexpected = [
        entry.get("record_kind")
        for entry in entries
        if entry.get("record_kind") not in {"sweep-meta", "prompt-sweep"}
    ]
    if unexpected:
        raise ValueError(f"trace contains non-policy records {unexpected[:3]}")
    if not records:
        raise ValueError("trace contains no prompt-sweep records")
    return metas[0], records


def _validate_backend_identity(
    meta: Mapping[str, Any],
    *,
    expected_model_id: str,
    expected_sampler_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    backend = meta.get("backend")
    if not isinstance(backend, Mapping):
        raise TypeError("sweep-meta lacks structured backend provenance")
    model_id = str(backend.get("model_id", ""))
    if model_id != expected_model_id:
        raise ValueError(
            f"sweep model identity {model_id!r} disagrees with expected {expected_model_id!r}"
        )
    sampler_identity = backend.get("sampling")
    if not isinstance(sampler_identity, Mapping):
        raise TypeError("sweep-meta lacks structured sampler identity")
    if _canonical_json(sampler_identity) != _canonical_json(expected_sampler_identity):
        raise ValueError("sweep sampler identity differs from the expected matching sampler")
    return sampler_identity


def _validate_prompt_identity(
    meta: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_prompt_ids: Sequence[str],
) -> str:
    prompt_ids = tuple(str(record.get("prompt_id", "")) for record in records)
    expected_ids = tuple(str(prompt_id) for prompt_id in expected_prompt_ids)
    if prompt_ids != expected_ids:
        raise ValueError("sweep prompt identity/order differs from the expected corpus")
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("sweep repeats a prompt_id")
    order_digest = hashlib.sha256("\n".join(prompt_ids).encode("utf-8")).hexdigest()
    if str(meta.get("prompt_id_order_sha256", "")) != order_digest:
        raise ValueError("sweep-meta prompt-id digest does not match its records")
    if int(meta.get("n_prompts", -1)) != len(prompt_ids):
        raise ValueError("sweep-meta n_prompts does not match its records")
    return order_digest


def _validate_rows_identity(meta: Mapping[str, Any], expected_rows_sha256: str | None) -> None:
    if expected_rows_sha256 is None:
        return
    if str(meta.get("rows_sha256", "")) != expected_rows_sha256:
        raise ValueError("sweep prompt-content digest differs from the expected corpus")


def _validate_sweep_provenance(
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_prompt_ids: Sequence[str],
    expected_model_id: str,
    expected_sampler_identity: Mapping[str, Any],
    expected_rows_sha256: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, Mapping[str, Any]]:
    meta, records = _split_trace_entries(entries)
    sampler_identity = _validate_backend_identity(
        meta,
        expected_model_id=expected_model_id,
        expected_sampler_identity=expected_sampler_identity,
    )
    order_digest = _validate_prompt_identity(meta, records, expected_prompt_ids=expected_prompt_ids)
    _validate_rows_identity(meta, expected_rows_sha256)
    return meta, records, order_digest, sampler_identity


def _validated_record_signal(
    record: Mapping[str, Any],
) -> tuple[tuple[str, str, str], int, int, int, float]:
    row = record.get("row")
    if not isinstance(row, Mapping):
        raise TypeError("prompt-sweep record lacks its structured row")
    if str(row.get("prompt_id", record["prompt_id"])) != str(record["prompt_id"]):
        raise ValueError(f"record {record['prompt_id']!r} row identity disagrees")
    sample_count = int(record.get("n_samples", -1))
    parse_failures = int(record.get("n_parse_failures", -1))
    truncated = int(record.get("n_truncated_thinking", -1))
    if sample_count < 1 or not 0 <= parse_failures <= sample_count:
        raise ValueError(f"invalid sample counts in prompt {record['prompt_id']!r}")
    if not 0 <= truncated <= sample_count:
        raise ValueError(f"invalid truncation count in prompt {record['prompt_id']!r}")
    scores_raw = record.get("selection_scores")
    if not isinstance(scores_raw, list):
        raise TypeError(f"prompt {record['prompt_id']!r} lacks selection scores")
    scores = [float(score) for score in scores_raw]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError(f"prompt {record['prompt_id']!r} has non-finite scores")
    variance = statistics.pvariance(scores) if len(scores) >= MIN_SCORE_COUNT else 0.0
    stored_std = float(record.get("score_std", 0.0))
    if not math.isclose(stored_std, math.sqrt(variance), abs_tol=1e-8):
        raise ValueError(f"prompt {record['prompt_id']!r} score variance is inconsistent")
    return _stratum_key(row), sample_count, parse_failures, truncated, variance


def _summarize_sweep_stratum(
    key: tuple[str, str, str], records: Sequence[Mapping[str, Any]]
) -> SweepStratumAudit:
    signals = [_validated_record_signal(record) for record in records]
    samples = [signal[1] for signal in signals]
    parse_failures = [signal[2] for signal in signals]
    truncated = [signal[3] for signal in signals]
    variances = [signal[4] for signal in signals]
    n_samples = sum(samples)
    n_parse_failures = sum(parse_failures)
    n_truncated = sum(truncated)
    return SweepStratumAudit(
        family=key[0],
        key="/".join(key),
        n_prompts=len(records),
        n_samples=n_samples,
        n_parse_failures=n_parse_failures,
        n_truncated_thinking=n_truncated,
        parseable_rate=(n_samples - n_parse_failures) / n_samples,
        termination_rate=(n_samples - n_truncated) / n_samples,
        reward_variance_mean=statistics.fmean(variances),
        reward_variance_min=min(variances),
        reward_variance_max=max(variances),
        n_pure_prompts=sum(math.isclose(variance, 0.0, abs_tol=1e-12) for variance in variances),
    )


def summarize_sweep_trace(  # noqa: PLR0913 - each identity axis is checked independently
    trace: Path | Sequence[Mapping[str, Any]],
    *,
    expected_prompt_ids: Sequence[str],
    expected_model_id: str,
    expected_sampler_identity: Mapping[str, Any],
    expected_rows_sha256: str | None = None,
    expected_strata: Sequence[tuple[str, str, str]] | None = None,
) -> SweepTraceAudit:
    """Summarize observed reward variance and termination after checking sweep provenance.

    ``trace`` is either a JSONL path written by ``games.select_prompts`` or its already-loaded
    records. The function requires the caller to provide the prompt identity, model id and sampler
    identity expected for the comparison. Pass ``expected_rows_sha256`` when the trace was generated
    from a concrete corpus file; this checks prompt bytes as well as the prompt-id order. The
    function therefore cannot turn a trace with missing or drifted provenance into apparently
    comparable evidence. Rows and completions are consumed only for their structured fields; prompt
    text is never copied into the audit.
    """
    _, records, order_digest, sampler_identity = _validate_sweep_provenance(
        _trace_entries(trace),
        expected_prompt_ids=expected_prompt_ids,
        expected_model_id=expected_model_id,
        expected_sampler_identity=expected_sampler_identity,
        expected_rows_sha256=expected_rows_sha256,
    )
    observed: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key, *_ = _validated_record_signal(record)
        observed.setdefault(key, []).append(record)
    observed_keys = set(observed)
    if expected_strata is not None:
        expected_keys = set(expected_strata)
        if observed_keys != expected_keys:
            raise ValueError(
                f"sweep strata differ from the expected full set: missing={sorted(expected_keys - observed_keys)}, "
                f"extra={sorted(observed_keys - expected_keys)}"
            )

    stratum_audits = tuple(_summarize_sweep_stratum(key, observed[key]) for key in sorted(observed))
    total_samples = sum(stratum.n_samples for stratum in stratum_audits)
    total_parse_failures = sum(stratum.n_parse_failures for stratum in stratum_audits)
    total_truncated = sum(stratum.n_truncated_thinking for stratum in stratum_audits)
    return SweepTraceAudit(
        model_id=expected_model_id,
        sampler_identity=dict(sampler_identity),
        prompt_id_order_sha256=order_digest,
        n_prompts=len(records),
        n_samples=total_samples,
        n_parse_failures=total_parse_failures,
        n_truncated_thinking=total_truncated,
        parseable_rate=(total_samples - total_parse_failures) / total_samples,
        termination_rate=(total_samples - total_truncated) / total_samples,
        strata=stratum_audits,
    )


def write_post_screen_summary(
    trace_path: Path,
    corpus_path: Path,
    output_path: Path,
    *,
    expected_model_id: str,
    expected_sampler_identity: Mapping[str, Any],
) -> Path:
    """Summarize a complete matching-sampler trace against a frozen corpus on disk."""
    rows = read_jsonl(corpus_path)
    prompt_ids = [str(row["prompt_id"]) for row in rows]
    summary = summarize_sweep_trace(
        trace_path,
        expected_prompt_ids=prompt_ids,
        expected_model_id=expected_model_id,
        expected_sampler_identity=expected_sampler_identity,
        expected_rows_sha256=rows_file_digest(corpus_path),
        expected_strata=expected_stratum_keys(rows),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def _assert_normalized(values: Sequence[float], *, context: str) -> None:
    if any(
        not math.isfinite(value)
        or not -NORMALIZATION_TOLERANCE <= value <= 1.0 + NORMALIZATION_TOLERANCE
        for value in values
    ):
        raise ValueError(
            f"production scorer reward for {context} is outside [0, 1]: {list(values)}"
        )


def _binary_ranking(left: float, right: float) -> str:
    if math.isclose(left, right, abs_tol=1e-10):
        return "tie"
    return "cooperate" if left > right else "defect"


def _matrix_raw_payoffs(row: Mapping[str, Any], mix: float) -> tuple[float, float, float, float]:
    opponent_cooperation = float(mix)
    payoff_cc = float(row["payoff_cc"])
    payoff_cd = float(row["payoff_cd"])
    payoff_dc = float(row["payoff_dc"])
    payoff_dd = float(row["payoff_dd"])
    own_cooperate = opponent_cooperation * payoff_cc + (1 - opponent_cooperation) * payoff_cd
    own_defect = opponent_cooperation * payoff_dc + (1 - opponent_cooperation) * payoff_dd
    welfare_cooperate = opponent_cooperation * 2 * payoff_cc + (1 - opponent_cooperation) * (
        payoff_cd + payoff_dc
    )
    welfare_defect = (
        opponent_cooperation * (payoff_dc + payoff_cd) + (1 - opponent_cooperation) * 2 * payoff_dd
    )
    return own_cooperate, own_defect, welfare_cooperate, welfare_defect


def _audit_matrix_representative(row: Mapping[str, Any]) -> list[MatrixRewardAudit]:
    records: list[MatrixRewardAudit] = []
    for mix in MIX_POINTS:
        n_cooperate = round(mix * MATRIX_GROUP_SIZE)
        completions = [
            _action_completion(row, cooperate=index < n_cooperate)
            for index in range(MATRIX_GROUP_SIZE)
        ]
        rewards = _score(row, completions, group_size=MATRIX_GROUP_SIZE)
        cooperative_reward = rewards[0]
        defective_reward = rewards[-1]
        _assert_normalized(
            (cooperative_reward, defective_reward),
            context=f"{row['game_id']}/{row['payoff_variant']} mix={mix}",
        )
        ranking = _binary_ranking(cooperative_reward, defective_reward)
        (
            own_payoff_cooperative,
            own_payoff_defective,
            welfare_cooperative,
            welfare_defective,
        ) = _matrix_raw_payoffs(row, mix)
        failure_completions = [
            _action_completion(row, cooperate=index < n_cooperate)
            for index in range(MATRIX_GROUP_SIZE)
        ] + ["no parseable action"]
        with_failure = _score(row, failure_completions, group_size=MATRIX_GROUP_SIZE + 1)
        parse_group_cooperative_reward = with_failure[0]
        parse_group_defective_reward = with_failure[n_cooperate]
        parse_price = with_failure[-1]
        _assert_normalized(
            (
                parse_group_cooperative_reward,
                parse_group_defective_reward,
            ),
            context=f"{row['game_id']}/{row['payoff_variant']} mix={mix} parse-group",
        )
        parse_below_worst = (
            parse_price
            <= min(parse_group_cooperative_reward, parse_group_defective_reward)
            + NORMALIZATION_TOLERANCE
        )
        if not parse_below_worst:
            raise ValueError(
                f"row-relative parse price is not below its own group rewards for "
                f"{row['game_id']}/{row['payoff_variant']} mix={mix}"
            )
        records.append(
            MatrixRewardAudit(
                game_id=str(row["game_id"]),
                payoff_variant=str(row["payoff_variant"]),
                framing_id=str(row[FRAMING_ID_COLUMN]),
                mix=mix,
                cooperative_reward=cooperative_reward,
                defective_reward=defective_reward,
                own_payoff_cooperative=own_payoff_cooperative,
                own_payoff_defective=own_payoff_defective,
                welfare_cooperative=welfare_cooperative,
                welfare_defective=welfare_defective,
                own_payoff_ranking=_binary_ranking(own_payoff_cooperative, own_payoff_defective),
                welfare_ranking=_binary_ranking(welfare_cooperative, welfare_defective),
                parse_group_cooperative_reward=parse_group_cooperative_reward,
                parse_group_defective_reward=parse_group_defective_reward,
                ranking=ranking,
                tie=ranking == "tie",
                parse_price=parse_price,
                parse_below_worst=parse_below_worst,
                normalized=True,
            )
        )
    return records


def _audit_trust_representative(row: Mapping[str, Any]) -> TrustSlopeAudit:
    endowment = int(row["endowment"])
    none_reward, all_reward = _score(
        row,
        ["<send>0</send>", f"<send>{endowment}</send>"],
        group_size=2,
    )
    with_failure = _score(
        row,
        ["<send>0</send>", f"<send>{endowment}</send>", "no parseable send"],
        group_size=3,
    )
    parse_group_sent_none_reward = with_failure[0]
    parse_group_sent_all_reward = with_failure[1]
    parse_price = with_failure[-1]
    _assert_normalized(
        (
            none_reward,
            all_reward,
            parse_group_sent_none_reward,
            parse_group_sent_all_reward,
        ),
        context=f"trust/{row['payoff_variant']}",
    )
    parse_below_worst = (
        parse_price
        <= min(parse_group_sent_none_reward, parse_group_sent_all_reward) + NORMALIZATION_TOLERANCE
    )
    if not parse_below_worst:
        raise ValueError(
            f"row-relative parse price is not below its own trust group rewards for "
            f"{row['payoff_variant']}"
        )
    if math.isclose(none_reward, all_reward, abs_tol=1e-10):
        ranking = "tie"
    else:
        ranking = "send-all" if all_reward > none_reward else "send-none"
    return TrustSlopeAudit(
        payoff_variant=str(row["payoff_variant"]),
        sent_none_reward=none_reward,
        sent_all_reward=all_reward,
        parse_group_sent_none_reward=parse_group_sent_none_reward,
        parse_group_sent_all_reward=parse_group_sent_all_reward,
        slope=all_reward - none_reward,
        ranking=ranking,
        parse_price=parse_price,
        parse_below_worst=parse_below_worst,
        normalized=True,
    )


def _audit_stratum(row: Mapping[str, Any]) -> StratumAudit:
    rewards_by_group: list[float] = []
    pure_groups = 0
    if row[FRAMING_ID_COLUMN] == FRAMING_ID_UNSET:
        patterns = [
            [
                f"<send>{int(row['endowment']) if mask & (1 << index) else 0}</send>"
                for index in range(4)
            ]
            for mask in range(16)
        ]
        family = TRUST_FAMILY
    else:
        patterns = [
            [_action_completion(row, cooperate=bool(mask & (1 << index))) for index in range(4)]
            for mask in range(16)
        ]
        family = {
            "twin-pd": PD_FAMILY,
            "stag-hunt": STAG_HUNT_FAMILY,
        }[str(row["game_id"])]
    for completions in patterns:
        rewards = _score(row, completions, group_size=4)
        _assert_normalized(rewards, context=f"stratum/{row['game_id']}/{row['payoff_variant']}")
        variance = statistics.pvariance(rewards)
        rewards_by_group.append(variance)
        if math.isclose(variance, 0.0, abs_tol=1e-12):
            pure_groups += 1
    return StratumAudit(
        family=family,
        key=f"{row['game_id']}/{row['payoff_variant']}/{row[FRAMING_ID_COLUMN]}",
        n_groups_considered=len(patterns),
        n_pure_groups=pure_groups,
        reward_variance_mean=statistics.fmean(rewards_by_group),
        reward_variance_min=min(rewards_by_group),
        reward_variance_max=max(rewards_by_group),
        synthetic_parseable_rate=1.0,
    )


def audit_learning_signal(rows: Sequence[Mapping[str, Any]]) -> LearningSignalAudit:
    """Audit production scorer rankings, ties, normalization, parse prices and strata.

    The canonical completions are a matching-sampler fixture: they exercise every proposed stratum
    and preserve pure groups in the accounting. They do not estimate model termination; a real sweep
    trace must be summarized separately once a model has generated it.
    """
    if not rows:
        raise ValueError("cannot audit an empty cooperation corpus")
    representatives = _unique_representatives(rows)
    matrix = [row for row in representatives if row[FRAMING_ID_COLUMN] != FRAMING_ID_UNSET]
    trust = [row for row in representatives if row[FRAMING_ID_COLUMN] == FRAMING_ID_UNSET]
    matrix_games = {str(row["game_id"]) for row in matrix}
    trust_variants = {str(row["payoff_variant"]) for row in trust}
    if len(matrix) != EXPECTED_MATRIX_STRATA or matrix_games != {"twin-pd", "stag-hunt"}:
        raise ValueError(
            "learning-signal audit requires the eight matrix strata from both matrix families"
        )
    if len(trust) != EXPECTED_TRUST_STRATA or trust_variants != set(TRUST_PAYOFF_VARIANTS):
        raise ValueError("learning-signal audit requires both disclosed trust return regimes")
    matrix_rankings = tuple(
        record for row in matrix for record in _audit_matrix_representative(row)
    )
    trust_slopes = tuple(_audit_trust_representative(row) for row in trust)
    strata = tuple(_audit_stratum(row) for row in representatives)
    return LearningSignalAudit(
        mix_points=MIX_POINTS,
        matrix_rankings=matrix_rankings,
        trust_slopes=trust_slopes,
        strata=strata,
        matrix_normalized=all(record.normalized for record in matrix_rankings),
        trust_normalized=all(record.normalized for record in trust_slopes),
    )


def write_training_corpus(
    built: BuiltCooperationCorpus,
    *,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    audit_path: Path | None = DEFAULT_AUDIT_PATH,
) -> tuple[Path, Path]:
    """Write rows to run artifacts and text-free provenance to private scratch."""
    write_corpus(corpus_path, built.rows)
    manifest = {
        **built.manifest,
        "corpus_path": os.path.relpath(corpus_path, start=manifest_path.parent),
        "corpus_sha256": rows_file_digest(corpus_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit_learning_signal(built.rows).to_json_dict(), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return corpus_path, manifest_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and audit the cooperation corpus on CPU.")
    parser.add_argument("--corpus-out", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--framings-file", type=Path, default=None)
    parser.add_argument("--post-screen-trace", type=Path, default=None)
    parser.add_argument("--post-screen-corpus", type=Path, default=None)
    parser.add_argument("--post-screen-out", type=Path, default=None)
    parser.add_argument("--post-screen-model-id", default=None)
    parser.add_argument("--post-screen-sampler-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the text-free manifest, corpus artifact and offline scorer audit."""
    args = _parse_args(argv)
    if args.post_screen_trace is not None:
        required = {
            "--post-screen-corpus": args.post_screen_corpus,
            "--post-screen-out": args.post_screen_out,
            "--post-screen-model-id": args.post_screen_model_id,
            "--post-screen-sampler-json": args.post_screen_sampler_json,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"post-screen mode needs {missing}")
        sampler_payload = json.loads(args.post_screen_sampler_json.read_text(encoding="utf-8"))
        if not isinstance(sampler_payload, Mapping):
            raise TypeError("--post-screen-sampler-json must contain a JSON object")
        write_post_screen_summary(
            args.post_screen_trace,
            args.post_screen_corpus,
            args.post_screen_out,
            expected_model_id=args.post_screen_model_id,
            expected_sampler_identity=sampler_payload,
        )
        return 0
    runtime: RuntimeFramings | None = None
    if args.framings_file is not None:
        runtime = load_framings(args.framings_file)
    built = build_training_corpus(runtime_framings=runtime)
    write_training_corpus(
        built,
        corpus_path=args.corpus_out,
        manifest_path=args.manifest_out,
        audit_path=args.audit_out,
    )
    logger.info(
        "cooperation corpus ready: rows=%s rows_by_family=%s manifest=%s audit=%s",
        len(built.rows),
        built.rows_by_family,
        args.manifest_out,
        args.audit_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
