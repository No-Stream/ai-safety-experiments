"""Publish resumable sampled-token diagnostic tables from completed scorer records."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

from reward_hacking.distill.agentic_rounds import (
    _advantage_by_turn,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.distill.diagnostic_moves import CONDITIONS, classify_turn

MIN_CORRELATION_TOKENS = 2


def _read(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("JSONL records must be objects")
    return rows


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < MIN_CORRELATION_TOKENS:
        return None
    left_mean, right_mean = math.fsum(left) / len(left), math.fsum(right) / len(right)
    numerator = math.fsum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        math.fsum((a - left_mean) ** 2 for a in left)
        * math.fsum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def _reproduction_status(applicable: Sequence[bool | None]) -> bool | None:
    if False in applicable:
        return False
    return None if None in applicable else True


def summarize_round(  # noqa: C901, PLR0912
    samples: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    trace_records_by_key: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Compare every score to the unchanged recorded sampled-token baseline."""
    if not set(scores).issubset(CONDITIONS) or (
        not allow_partial and set(scores) != set(CONDITIONS)
    ):
        raise ValueError("diagnostic requires all four conditions")
    indexed = {
        condition: {row["key"]: row for row in values} for condition, values in scores.items()
    }
    for condition, values in scores.items():
        if len(indexed[condition]) != len(values) or set(indexed[condition]) != {
            sample["key"] for sample in samples
        }:
            raise ValueError("score keys must match samples exactly")
    per_turn: list[dict[str, Any]] = []
    aggregates: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
    sanity: dict[str, tuple[list[float], list[float]]] = {}
    agreement: dict[str, tuple[list[float], list[float], list[bool | None], set[str]]] = {}
    for sample in samples:
        key = sample["key"]
        length = len(sample["completion_token_ids"])
        if any(
            len(sample[field]) != length
            for field in ("loss_mask", "token_turn_indices", "student_sampling_logps")
        ):
            raise ValueError("sample token arrays do not align")
        for condition in scores:
            scored = indexed[condition][key]
            if (
                scored["completion_token_ids"] != sample["completion_token_ids"]
                or scored["loss_mask"] != sample["loss_mask"]
            ):
                raise ValueError("scored completion or mask differs from sample")
        trace = trace_records_by_key.get(key, []) if trace_records_by_key else []
        if sample["hint_family"] == "none" and "student_unhinted" in scores:
            recorded, forced, applicable, reasons = agreement.setdefault(
                sample["diagnostic_cell"], ([], [], [], set())
            )
            applicable.append(sample.get("sampling_sanity_applicable", True))
            reasons.add(sample.get("sampling_sanity_reason", "matching unhinted rendering"))
            for index, active in enumerate(sample["loss_mask"]):
                if active:
                    recorded.append(float(sample["student_sampling_logps"][index]))
                    forced.append(float(indexed["student_unhinted"][key]["teacher_logps"][index]))
        turns = {item["turn"]: item for item in trace if item.get("record") == "turn"}
        for turn in sorted(
            {
                index
                for index, active in zip(
                    sample["token_turn_indices"], sample["loss_mask"], strict=True
                )
                if active
            }
            | {int(turn) for turn in sample.get("turn_classes", {})}
            | set(turns)
        ):
            positions = [
                index
                for index, (number, active) in enumerate(
                    zip(sample["token_turn_indices"], sample["loss_mask"], strict=True)
                )
                if active and number == turn
            ]
            action_class = (
                classify_turn(turns[turn])
                if turn in turns
                else sample.get("turn_classes", {}).get(str(turn), "other")
            )
            row: dict[str, Any] = {
                "key": key,
                "cell": sample["diagnostic_cell"],
                "hint_family": sample["hint_family"],
                "hint_strength": sample.get("hint_strength", "specific"),
                "turn_index": turn,
                "action_class": action_class,
                "strategy": sample.get("strategy"),
                "sampling_sanity_applicable": sample.get("sampling_sanity_applicable"),
                "sampling_sanity_reason": sample.get("sampling_sanity_reason"),
                "grader_directed": (
                    True
                    if turn in sample.get("grader_directed_move_turn_indices", [])
                    else None
                    if sample.get("grader_directed_move_coverage") == "unknown"
                    else False
                ),
                "conditions": {},
            }
            row["turn_class"] = (
                "grader_directed_move" if row["grader_directed"] is True else action_class
            )
            for condition in scores:
                values = [
                    float(indexed[condition][key]["teacher_logps"][index])
                    - float(sample["student_sampling_logps"][index])
                    for index in positions
                ]
                if any(not math.isfinite(value) for value in values):
                    raise ValueError("nonfinite advantage")
                row["conditions"][condition] = {
                    "mean_advantage": math.fsum(values) / len(values) if values else None,
                    "sum_advantage": math.fsum(values),
                    "action_tokens": len(values),
                }
                aggregates[
                    (
                        sample["diagnostic_cell"],
                        sample["hint_family"],
                        sample.get("hint_strength", "specific"),
                        action_class,
                        row["turn_class"],
                        condition,
                    )
                ].extend(values)
            per_turn.append(row)
        if (
            sample.get("sampling_sanity_applicable", True)
            and sample["hint_family"] == "none"
            and "student_unhinted" in scores
        ):
            left, right = sanity.setdefault(sample["diagnostic_cell"], ([], []))
            for index, active in enumerate(sample["loss_mask"]):
                if active:
                    left.append(float(sample["student_sampling_logps"][index]))
                    right.append(float(indexed["student_unhinted"][key]["teacher_logps"][index]))
    return {
        "per_turn": per_turn,
        "sampling_agreement": [
            {
                "cell": cell,
                "count": len(recorded),
                "correlation": _correlation(recorded, forced),
                "mean_absolute_difference": math.fsum(
                    abs(left - right) for left, right in zip(recorded, forced, strict=True)
                )
                / len(recorded),
                "unhinted_sampling_sanity_applicable": _reproduction_status(applicable),
                "reason": "; ".join(sorted(reasons)),
            }
            for cell, (recorded, forced, applicable, reasons) in sorted(agreement.items())
        ],
        "per_cell": [
            {
                "cell": cell,
                "hint_family": hint,
                "hint_strength": strength,
                "action_class": label,
                "turn_class": turn_class,
                "condition": condition,
                "action_tokens": len(values),
                "mean_advantage": math.fsum(values) / len(values) if values else None,
                "sum_advantage": math.fsum(values),
            }
            for (cell, hint, strength, label, turn_class, condition), values in sorted(
                aggregates.items()
            )
        ],
        "sampling_sanity": [
            {
                "cell": cell,
                "count": len(left),
                "correlation": _correlation(left, right),
                "mean_absolute_difference": math.fsum(
                    abs(a - b) for a, b in zip(left, right, strict=True)
                )
                / len(left),
            }
            for cell, (left, right) in sorted(sanity.items())
        ],
    }


def summarize_decisions(
    samples: Sequence[Mapping[str, Any]], scores: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    """Normalize candidate sequence likelihoods only over the supplied alternatives."""
    source = {row["key"]: row for row in samples}
    result: list[dict[str, Any]] = []
    for condition, rows in scores.items():
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for scored in rows:
            sample = source[scored["key"]]
            logp = math.fsum(scored["teacher_logps"])
            grouped[(sample["decision_source_key"], sample["decision_turn"])].append(
                {
                    "key": sample["decision_source_key"],
                    "cell": sample["diagnostic_cell"],
                    "hint_family": sample["hint_family"],
                    "condition": condition,
                    "action_class": sample["decision_class"],
                    "is_sampled_move": sample.get("decision_is_sampled_move", False),
                    "sampled_action_class": sample.get("sampled_action_class"),
                    "hint_strength": sample.get("hint_strength", "specific"),
                    "turn_index": sample["decision_turn"],
                    "sum_logp": logp,
                    "mean_logp": logp / len(scored["teacher_logps"]),
                    "action_tokens": len(scored["teacher_logps"]),
                }
            )
        for items in grouped.values():
            largest = max(item["sum_logp"] for item in items)
            denominator = math.fsum(math.exp(item["sum_logp"] - largest) for item in items)
            for item in items:
                item["relative_candidate_probability"] = (
                    math.exp(item["sum_logp"] - largest) / denominator
                )
                item["move_class"] = (
                    item["sampled_action_class"]
                    if item["action_class"] == "sampled_move"
                    else item["action_class"]
                )
            class_probability: dict[str, float] = defaultdict(float)
            for item in items:
                class_probability[item["move_class"]] += item["relative_candidate_probability"]
            sampled_probability = next(
                (
                    item["relative_candidate_probability"]
                    for item in items
                    if item["is_sampled_move"]
                ),
                None,
            )
            for item in items:
                item["move_class_probability"] = class_probability[item["move_class"]]
                item["sampled_move_probability"] = sampled_probability
                result.append(item)
    return result


def _table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[str]:
    return [
        "| " + " | ".join(fields) + " |",
        "|" + "---|" * len(fields),
        *["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows],
    ]


def compact_readout_tables(summary: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Keep the Markdown small while retaining the complete per-cell JSON tables."""
    dimensions = ("hint_family", "hint_strength", "condition", "action_class")
    turn_dimensions = (*dimensions, "turn_class")
    turns: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in summary["per_cell"]:
        identity = (
            *tuple(row[field] for field in dimensions),
            row.get("turn_class", row["action_class"]),
        )
        group = turns.setdefault(
            identity,
            {
                **dict(zip(turn_dimensions, identity, strict=True)),
                "action_tokens": 0,
                "sum_advantage": 0.0,
            },
        )
        group["action_tokens"] += row["action_tokens"]
        group["sum_advantage"] += row["sum_advantage"]
    for row in turns.values():
        row["mean_advantage"] = (
            row["sum_advantage"] / row["action_tokens"] if row["action_tokens"] else None
        )
    decisions: dict[tuple[str, ...], dict[str, Any]] = {}
    seen: set[tuple[str, int, str, str]] = set()
    for row in summary["decision_preferences"]:
        boundary = (row["key"], row["turn_index"], row["condition"], row["move_class"])
        if boundary in seen:
            continue
        seen.add(boundary)
        identity = (row["hint_family"], row["hint_strength"], row["condition"], row["move_class"])
        group = decisions.setdefault(
            identity,
            {
                **dict(zip(dimensions, identity, strict=True)),
                "decisions": 0,
                "probability_sum": 0.0,
                "sampled_probability_sum": 0.0,
                "sampled_comparisons": 0,
            },
        )
        group["decisions"] += 1
        group["probability_sum"] += row["move_class_probability"]
        if row["sampled_move_probability"] is not None:
            group["sampled_probability_sum"] += row["sampled_move_probability"]
            group["sampled_comparisons"] += 1
    for row in decisions.values():
        row["mean_move_class_probability"] = row.pop("probability_sum") / row["decisions"]
        total = row.pop("sampled_probability_sum")
        row["mean_sampled_move_probability"] = (
            total / row["sampled_comparisons"] if row["sampled_comparisons"] else None
        )
    return {
        "turn_advantage": list(turns.values()),
        "decision_preferences": list(decisions.values()),
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    readout = compact_readout_tables(summary)
    lines = _table(
        readout["turn_advantage"],
        (
            "hint_family",
            "hint_strength",
            "action_class",
            "turn_class",
            "condition",
            "action_tokens",
            "mean_advantage",
            "sum_advantage",
        ),
    )
    lines.extend(
        [
            "",
            *_table(
                summary["sampling_agreement"],
                (
                    "cell",
                    "count",
                    "correlation",
                    "mean_absolute_difference",
                    "unhinted_sampling_sanity_applicable",
                    "reason",
                ),
            ),
        ]
    )
    lines.extend(
        [
            "",
            *_table(
                readout["decision_preferences"],
                (
                    "hint_family",
                    "hint_strength",
                    "condition",
                    "action_class",
                    "decisions",
                    "mean_move_class_probability",
                    "mean_sampled_move_probability",
                ),
            ),
        ]
    )
    lines.extend(
        [
            "",
            *_table(
                [
                    {"measurement": label, "interpretation": value}
                    for label, value in summary["limitations"].items()
                ],
                ("measurement", "interpretation"),
            ),
        ]
    )
    lines.extend(
        [
            "",
            *_table(
                [{"scope": "All input cells; full per-cell values are in per-cell.jsonl"}],
                ("scope",),
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _append_completed(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    existing = _read(path) if path.exists() else []
    indexed = {tuple(row[field] for field in fields): row for row in existing}
    if len(indexed) != len(existing):
        raise ValueError(f"duplicate persisted diagnostic rows: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for row in rows:
            key = tuple(row[field] for field in fields)
            if key in indexed:
                if indexed[key] != row:
                    raise ValueError(f"completed diagnostic row changed: {path.name}")
            else:
                output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                output.flush()
                indexed[key] = dict(row)


def _snapshot(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def publish_completed(
    run_dir: Path,
    samples: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Append completed turns and candidate groups, and refresh aggregate snapshots."""
    sample_lookup = {row["key"]: row for row in samples}
    decision_lookup = {row["key"]: row for row in decisions}
    decision_scopes = {row.get("decision_scope", "all_tool_calls") for row in decisions}
    summary: dict[str, Any] = {
        "per_turn": [],
        "per_cell": [],
        "sampling_sanity": [],
        "sampling_agreement": [],
        "decision_preferences": [],
        "advantage_by_turn": {},
        "limitations": {
            "Baseline": "Recorded raw logprobs; reproduction requires matching model, adapter and unhinted context.",
            "Decision preference": (
                "Complete command alternatives with the recorded tool wrapper; "
                "probability is normalized over these candidates only. Missing move classes remain unmeasured."
            ),
            "Decision coverage": "; ".join(sorted(decision_scopes))
            if decision_scopes
            else "no_tool_call_decisions",
            "Final reports": "Reports without saved token IDs and logprobs are counted as unscored.",
            "Progress": (
                "Completed episode rows and complete candidate groups are appended; "
                "aggregates include completed records only."
            ),
        },
    }
    flat_turns: list[dict[str, Any]] = []
    for condition, condition_scores in scores.items():
        episode_scores = [row for row in condition_scores if row["key"] in sample_lookup]
        available = [sample_lookup[row["key"]] for row in episode_scores]
        partial = summarize_round(available, {condition: episode_scores}, allow_partial=True)
        for field in ("per_cell", "sampling_sanity", "sampling_agreement"):
            summary[field].extend(partial[field])
        for row in partial["per_turn"]:
            values = row["conditions"][condition]
            flat_turns.append(
                {
                    **{key: value for key, value in row.items() if key != "conditions"},
                    "condition": condition,
                    **values,
                }
            )
        expected_groups: dict[tuple[str, int], set[str]] = defaultdict(set)
        for row in decisions:
            if (row["hint_family"] == "none") == condition.endswith("unhinted"):
                expected_groups[(row["decision_source_key"], row["decision_turn"])].add(row["key"])
        scored_decisions = {
            row["key"]: row for row in condition_scores if row["key"] in decision_lookup
        }
        complete_keys = set().union(
            *(keys for keys in expected_groups.values() if keys.issubset(scored_decisions))
        )
        summary["decision_preferences"].extend(
            summarize_decisions(
                decisions,
                {condition: [row for key, row in scored_decisions.items() if key in complete_keys]},
            )
        )
        episodes_path = run_dir / "episodes.jsonl"
        if episodes_path.exists():
            summary["advantage_by_turn"][condition] = _advantage_by_turn(
                episodes_path,
                [
                    {
                        **row,
                        "student_sampling_logps": sample_lookup[row["key"]][
                            "student_sampling_logps"
                        ],
                    }
                    for row in episode_scores
                ],
            )
    _append_completed(run_dir / "per-turn.jsonl", flat_turns, ("key", "turn_index", "condition"))
    _append_completed(
        run_dir / "decision-preferences.jsonl",
        summary["decision_preferences"],
        ("key", "turn_index", "condition", "action_class"),
    )
    grouped_turns: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read(run_dir / "per-turn.jsonl"):
        identity = (row["key"], row["turn_index"])
        if identity not in grouped_turns:
            grouped_turns[identity] = {
                key: value
                for key, value in row.items()
                if key not in {"condition", "mean_advantage", "sum_advantage", "action_tokens"}
            }
            grouped_turns[identity]["conditions"] = {}
        grouped_turns[identity]["conditions"][row["condition"]] = {
            field: row[field] for field in ("mean_advantage", "sum_advantage", "action_tokens")
        }
    summary["per_turn"] = list(grouped_turns.values())
    summary["coverage"] = {
        "episodes": sum(row["hint_family"] == "none" for row in samples),
        "unscored_final_reports": sum(
            bool(row.get("unscored_final_report"))
            for row in samples
            if row["hint_family"] == "none"
        ),
        "completed_turn_condition_rows": len(_read(run_dir / "per-turn.jsonl")),
    }
    _snapshot(
        run_dir / "per-cell.jsonl",
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in summary["per_cell"]
        ),
    )
    _snapshot(
        run_dir / "summary.json",
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )
    _snapshot(run_dir / "summary.md", _markdown(summary))
    return summary
