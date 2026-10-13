"""Selection report counts and the slug-only summary table written beside the selection manifest."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

SUMMARY_COLUMNS = (
    "scenario_slug",
    "arm",
    "criterion",
    "episodes",
    "hack_episodes",
    "excluded_episodes",
    "no_token_ids",
    "builder_failures",
    "builder_errors",
    "integrity_awareness",
    "honest_episodes",
    "solvable_hack_episodes",
    "solvable_fail_unexplained",
)


class SummaryRow(TypedDict):
    """One scenario, arm, and admission or exclusion count row."""

    scenario_slug: str
    arm: str
    criterion: str
    episodes: int
    hack_episodes: int
    excluded_episodes: int
    no_token_ids: int
    builder_failures: int
    builder_errors: str
    integrity_awareness: int
    honest_episodes: int
    solvable_hack_episodes: int
    solvable_fail_unexplained: int


def exclusion_counts(excluded: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Count excluded episodes by exclusion reason."""
    counts: defaultdict[str, int] = defaultdict(int)
    for entry in excluded:
        reason = entry["exclusion_reason"]
        if not isinstance(reason, str):
            raise TypeError("excluded episode must have a string exclusion_reason")
        counts[reason] += 1
    return dict(counts)


def field_counts(entries: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    """Count entries by the string value of one field."""
    counts: defaultdict[str, int] = defaultdict(int)
    for entry in entries:
        counts[str(entry[field])] += 1
    return dict(sorted(counts.items()))


def admitted_counts(entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Count admitted entries by strategy, source, family, criterion and admission type."""
    criterion_counts: defaultdict[str, int] = defaultdict(int)
    for entry in entries:
        for criterion in cast("list[str]", entry["admitting_criterion"]):
            criterion_counts[criterion] += 1
    hack_entries = [entry for entry in entries if entry.get("admission_type") == "hack"]
    honest_count = sum(entry.get("admission_type") == "honest" for entry in entries)
    schema_only_count = sum(entry.get("admission_type") == "schema_only" for entry in entries)
    admission_type_counts = Counter(str(entry["admission_type"]) for entry in entries)
    return {
        "by_strategy": field_counts(hack_entries, "primary_strategy"),
        "by_source": field_counts(entries, "source"),
        "by_family": field_counts(entries, "family"),
        "by_criterion": dict(sorted(criterion_counts.items())),
        "by_admission_type": {
            name: admission_type_counts[name] for name in ("hack", "honest", "schema_only")
        },
        "honest_episodes": honest_count,
        "hack_episodes": len(hack_entries),
        "schema_only_episodes": schema_only_count,
        "solvable_hack_episodes": sum(entry.get("solvable_hack") is True for entry in entries),
    }


def selection_admission_counts(
    selected: Sequence[Mapping[str, object]],
    excluded: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Admission counts with the aliases the floor checks and report consume."""
    counts = admitted_counts(selected)
    counts["honest_solve"] = counts["honest_episodes"]
    counts["solvable_hack"] = counts["solvable_hack_episodes"]
    counts["schema_only"] = counts["schema_only_episodes"]
    hack_count = cast("int", counts["hack_episodes"])
    counts["solvable_hack_share"] = (
        cast("int", counts["solvable_hack_episodes"]) / hack_count if hack_count else 0.0
    )
    counts["solvable_fail_unexplained"] = sum(
        entry.get("admission_exclusion_reason") == "solvable_fail_unexplained" for entry in excluded
    )
    return counts


def summary_rows(report: Mapping[str, object]) -> list[SummaryRow]:
    """Group a selection report into the slug-only summary rows."""
    selected_value = report.get("selected")
    excluded_value = report.get("excluded")
    if not isinstance(selected_value, list) or not isinstance(excluded_value, list):
        raise TypeError("selection report must contain selected and excluded lists")
    grouped = _new_summary_groups()
    for value in selected_value:
        if not isinstance(value, Mapping):
            raise TypeError("selection report selected entries must be objects")
        _add_selected_summary(grouped, value)
    for value in excluded_value:
        if not isinstance(value, Mapping):
            raise TypeError("selection report excluded entries must be objects")
        _add_excluded_summary(grouped, value)
    _add_summary_rollups(grouped, selected_value, excluded_value)
    return [row for _, row in sorted(grouped.items())]


def _add_summary_rollups(
    grouped: defaultdict[tuple[str, str, str], SummaryRow],
    selected_entries: Sequence[object],
    excluded_entries: Sequence[object],
) -> None:
    for field in ("primary_strategy", "source", "family"):
        for value in selected_entries:
            entry = cast("Mapping[str, object]", value)
            if field == "primary_strategy" and entry.get("admission_type") != "hack":
                continue
            row = _group_summary_row(grouped, (f"{field}:{entry[field]}", "all", "admitted"))
            row["episodes"] += 1
            _increment_admission_summary(row, entry)
    for value in excluded_entries:
        entry = cast("Mapping[str, object]", value)
        if "cap_reason" in entry:
            row = _group_summary_row(grouped, (f"cap:{entry['cap_reason']}", "all", "dropped"))
            row["episodes"] += 1
            row["excluded_episodes"] += 1


def _new_summary_groups() -> defaultdict[tuple[str, str, str], SummaryRow]:
    return defaultdict(
        lambda: {
            "scenario_slug": "",
            "arm": "",
            "criterion": "",
            "episodes": 0,
            "hack_episodes": 0,
            "excluded_episodes": 0,
            "no_token_ids": 0,
            "builder_failures": 0,
            "builder_errors": "",
            "integrity_awareness": 0,
            "honest_episodes": 0,
            "solvable_hack_episodes": 0,
            "solvable_fail_unexplained": 0,
        }
    )


def _group_summary_row(
    grouped: defaultdict[tuple[str, str, str], SummaryRow],
    key: tuple[str, str, str],
) -> SummaryRow:
    row = grouped[key]
    row["scenario_slug"], row["arm"], row["criterion"] = key
    return row


def _add_selected_summary(
    grouped: defaultdict[tuple[str, str, str], SummaryRow],
    entry: Mapping[str, object],
) -> None:
    criteria = entry.get("admitting_criterion")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("selected episode must have an admitting criterion")
    for criterion in criteria:
        if not isinstance(criterion, str):
            raise TypeError("selected admitting criteria must be strings")
        key = (cast("str", entry["scenario_slug"]), cast("str", entry["arm"]), criterion)
        row = _group_summary_row(grouped, key)
        row["episodes"] += 1
        _increment_admission_summary(row, entry)


def _increment_admission_summary(row: SummaryRow, entry: Mapping[str, object]) -> None:
    admission_type = entry.get("admission_type")
    if admission_type == "honest":
        row["honest_episodes"] += 1
    elif admission_type == "hack":
        row["hack_episodes"] += 1
        if entry.get("solvable_hack") is True:
            row["solvable_hack_episodes"] += 1
    elif admission_type == "schema_only":
        return
    else:
        raise ValueError("selected episode must have a hack or honest admission type")


def _add_excluded_summary(
    grouped: defaultdict[tuple[str, str, str], SummaryRow],
    entry: Mapping[str, object],
) -> None:
    reason = entry.get("exclusion_reason")
    if not isinstance(reason, str):
        raise TypeError("excluded episode must have a string exclusion reason")
    key = (cast("str", entry["scenario_slug"]), cast("str", entry["arm"]), f"excluded:{reason}")
    row = _group_summary_row(grouped, key)
    row["episodes"] += 1
    row["excluded_episodes"] += 1
    if reason == "no_token_ids":
        row["no_token_ids"] += 1
    elif reason == "builder_failure":
        row["builder_failures"] += 1
        error = entry.get("builder_error")
        if not isinstance(error, str):
            raise TypeError("builder failure must have a string builder_error")
        errors = set(row["builder_errors"].split("; ")) - {""}
        errors.add(error)
        row["builder_errors"] = "; ".join(sorted(errors))
    elif reason == "integrity_awareness":
        row["integrity_awareness"] += 1
    elif entry.get("admission_exclusion_reason") == "solvable_fail_unexplained":
        row["solvable_fail_unexplained"] += 1


def write_selection_report(
    report: Mapping[str, object],
    *,
    manifest_path: Path,
    summary_path: Path,
) -> None:
    """Write the machine-readable selection manifest and slug-only count table."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with summary_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(cast("Any", summary_rows(report)))
