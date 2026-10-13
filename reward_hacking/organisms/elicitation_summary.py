"""Summarize native records from the three single-turn reward-hacking screens."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from uuid import uuid4

from reward_hacking.eval_awareness import (
    AwarenessMatch,
    AwarenessSummary,
    AwarenessTier,
    scan_text,
    summarize_matches,
)
from reward_hacking.jsonl_io import read_jsonl_objects

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

Environment = Literal["atlas_mbpp", "aisi_codecontests", "ilcb"]
Label = Literal["hack", "honest_pass", "fail", "unknown"]
ENVIRONMENTS: tuple[Environment, ...] = ("atlas_mbpp", "aisi_codecontests", "ilcb")
LABELS: tuple[Label, ...] = ("hack", "honest_pass", "fail", "unknown")
AWARENESS_TIERS: tuple[AwarenessTier, ...] = (
    "lexical_alert",
    "relational",
    "environment_artificiality",
    "training_inference",
    "context",
)
AISI_HACK_TYPES = frozenset({"always_equal", "os_exit", "conftest", "special_case"})
ILCB_VERDICT_OUTCOMES = frozenset({"pass", "fail", "timeout"})
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "artifacts/organisms/elicitation-summary"
CELL_FIELD_COUNT = 4


@dataclass(frozen=True, slots=True)
class CellSpec:
    """One CLI-selected environment, model label, hint variant, and native artifact path."""

    environment: Environment
    model_label: str
    hint_variant: str
    path: Path


@dataclass(frozen=True, slots=True)
class ScreenSample:
    """One sample with its environment's native outcome semantics preserved."""

    environment: Environment
    model_label: str
    hint_variant: str
    sample_id: str
    label: Label
    hack_types: tuple[str, ...]
    completion: str
    reasoning: str
    hardened_fail: bool | None = None
    hint_sha256: str | None = None


def _parse_cell(value: str) -> CellSpec:
    parts = value.split(":", maxsplit=3)
    if len(parts) != CELL_FIELD_COUNT or any(not part for part in parts):
        raise ValueError("--cell must be ENVIRONMENT:MODEL_LABEL:HINT_VARIANT:PATH")
    environment, model_label, hint_variant, path = parts
    if environment not in ENVIRONMENTS:
        raise ValueError(f"unknown environment {environment!r}; expected one of {ENVIRONMENTS}")
    return CellSpec(
        environment=environment,
        model_label=model_label,
        hint_variant=hint_variant,
        path=Path(path),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"record file does not exist: {path}")
    return read_jsonl_objects(path)


def _required_string(record: dict[str, object], field: str, path: Path) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: record field {field!r} must be a non-empty string")
    return value


def _required_boolean(record: dict[str, object], field: str, path: Path) -> bool:
    value = record.get(field)
    if type(value) is not bool:
        raise ValueError(f"{path}: record field {field!r} must be a boolean")
    return value


def _optional_text(record: dict[str, object], field: str, path: Path) -> str:
    value = record.get(field, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{path}: optional record field {field!r} must be text or null")
    return value


def _hint_sha256(record: dict[str, object], cell: CellSpec, path: Path) -> str | None:
    if "hint_variant" in record:
        variant = _required_string(record, "hint_variant", path)
        if variant != cell.hint_variant:
            raise ValueError(
                f"{path}: record hint_variant {variant!r} does not match CLI cell "
                f"{cell.hint_variant!r}"
            )
    value = record.get("hint_sha256")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: optional record field 'hint_sha256' must be non-empty text")
    return value


def _check_hint_sha256(
    first: str | None, second: str | None, *, path: Path, sample_id: str
) -> str | None:
    if first is not None and second is not None and first != second:
        raise ValueError(f"{path}: sample {sample_id!r} has conflicting hint_sha256 values")
    return first if first is not None else second


def _read_atlas(cell: CellSpec) -> list[ScreenSample]:
    if not cell.path.is_dir():
        raise NotADirectoryError(f"Atlas path must be a run directory: {cell.path}")
    sample_paths = sorted(cell.path.glob("samples.*.jsonl"))
    if not sample_paths:
        raise FileNotFoundError(f"Atlas run has no samples.<model-key>.jsonl files: {cell.path}")
    expected_grade_paths = {
        cell.path / f"grades.{sample_path.name[len('samples.') : -len('.jsonl')]}.jsonl"
        for sample_path in sample_paths
    }
    actual_grade_paths = set(cell.path.glob("grades.*.jsonl"))
    if actual_grade_paths != expected_grade_paths:
        missing = sorted(str(path) for path in expected_grade_paths - actual_grade_paths)
        extra = sorted(str(path) for path in actual_grade_paths - expected_grade_paths)
        raise ValueError(f"Atlas sample/grade files do not pair; missing={missing}, extra={extra}")

    samples: list[ScreenSample] = []
    for sample_path in sample_paths:
        grade_path = (
            cell.path / f"grades.{sample_path.name[len('samples.') : -len('.jsonl')]}.jsonl"
        )
        sample_rows = _index_by_id(_read_jsonl(sample_path), "sample_id", sample_path)
        grade_rows = _index_by_id(_read_jsonl(grade_path), "sample_id", grade_path)
        if sample_rows.keys() != grade_rows.keys():
            raise ValueError(
                f"{sample_path} and {grade_path} have different sample ids; "
                f"missing grades={sorted(sample_rows.keys() - grade_rows.keys())}, "
                f"missing samples={sorted(grade_rows.keys() - sample_rows.keys())}"
            )
        for sample_id, sample_row in sample_rows.items():
            grade_row = grade_rows[sample_id]
            hint_sha256 = _check_hint_sha256(
                _hint_sha256(sample_row, cell, sample_path),
                _hint_sha256(grade_row, cell, grade_path),
                path=sample_path,
                sample_id=sample_id,
            )
            label = _exclusive_label(grade_row, ("hack", "honest_pass", "fail"), grade_path)
            hardcode_regex = _required_boolean(grade_row, "hardcode_regex", grade_path)
            samples.append(
                ScreenSample(
                    environment="atlas_mbpp",
                    model_label=cell.model_label,
                    hint_variant=cell.hint_variant,
                    sample_id=sample_id,
                    label=label,
                    hack_types=("hardcode_regex",) if hardcode_regex else (),
                    completion=_required_string(sample_row, "completion_text", sample_path),
                    reasoning=_optional_text(sample_row, "reasoning", sample_path),
                    hint_sha256=hint_sha256,
                )
            )
    return samples


def _read_aisi(cell: CellSpec) -> list[ScreenSample]:
    if not cell.path.is_dir():
        raise NotADirectoryError(f"AISI path must be a run directory: {cell.path}")
    sample_path = cell.path / "samples.jsonl"
    records = _read_jsonl(sample_path)
    samples: list[ScreenSample] = []
    for record in records:
        sample_id = _required_string(record, "record_key", sample_path)
        tests_passed = _required_boolean(record, "tests_passed", sample_path)
        reward_hacked = _required_boolean(record, "reward_hacked", sample_path)
        honest_pass = _required_boolean(record, "honest_pass", sample_path)
        hardened_fail_value = record.get("hardened_reward_hacked")
        if hardened_fail_value is not None and type(hardened_fail_value) is not bool:
            raise ValueError(
                f"{sample_path}: record field 'hardened_reward_hacked' must be boolean or null"
            )
        if "hardened_reward_hacked" not in record:
            raise ValueError(f"{sample_path}: record has no 'hardened_reward_hacked' field")
        native_types = record.get("hack_regexes_fired")
        if not isinstance(native_types, list) or any(
            not isinstance(hack_type, str) or hack_type not in AISI_HACK_TYPES
            for hack_type in native_types
        ):
            raise ValueError(f"{sample_path}: record field 'hack_regexes_fired' has unknown values")
        hack_types = tuple(native_types)
        if len(set(hack_types)) != len(hack_types):
            raise ValueError(f"{sample_path}: sample {sample_id!r} repeats a hack detector")
        if reward_hacked != (tests_passed and bool(hack_types)) or honest_pass != (
            tests_passed and not hack_types
        ):
            raise ValueError(f"{sample_path}: sample {sample_id!r} has inconsistent native labels")
        label: Label = "hack" if reward_hacked else "honest_pass" if honest_pass else "fail"
        samples.append(
            ScreenSample(
                environment="aisi_codecontests",
                model_label=cell.model_label,
                hint_variant=cell.hint_variant,
                sample_id=sample_id,
                label=label,
                hack_types=hack_types,
                completion=_required_string(record, "completion_text", sample_path),
                reasoning=_optional_text(record, "reasoning", sample_path),
                hardened_fail=hardened_fail_value,
                hint_sha256=_hint_sha256(record, cell, sample_path),
            )
        )
    return samples


def _records_path(path: Path) -> Path:
    if path.is_dir():
        candidates = sorted(path.glob("*-records.jsonl"))
        if len(candidates) != 1:
            raise ValueError(
                f"ILCB directory must contain exactly one *-records.jsonl file: {path}"
            )
        return candidates[0]
    if path.suffix == ".jsonl":
        return path
    return path.with_name(f"{path.stem}-records.jsonl")


def _read_ilcb(cell: CellSpec) -> list[ScreenSample]:
    records_path = _records_path(cell.path)
    records = _read_jsonl(records_path)
    samples: list[ScreenSample] = []
    for record in records:
        problem_id = _required_string(record, "problem_id", records_path)
        arm = _required_string(record, "arm", records_path)
        hatch = _required_string(record, "hatch", records_path)
        framing = _required_string(record, "framing", records_path)
        split = _required_string(record, "split", records_path)
        sample_index = record.get("sample_index")
        if type(sample_index) is not int or sample_index < 0:
            raise ValueError(
                f"{records_path}: record field 'sample_index' must be non-negative int"
            )
        sample_id = ":".join((arm, hatch, framing, split, problem_id, str(sample_index)))
        matched_planted_error = record.get("matched_planted_error")
        if matched_planted_error is not None and type(matched_planted_error) is not bool:
            raise ValueError(
                f"{records_path}: record field 'matched_planted_error' must be boolean or null"
            )
        grades = record.get("grades")
        if not isinstance(grades, list) or any(not isinstance(grade, dict) for grade in grades):
            raise ValueError(
                f"{records_path}: record field 'grades' must be a list of grade objects"
            )
        outcomes_by_split: dict[str, str] = {}
        for grade in grades:
            grade_split = _required_string(grade, "split", records_path)
            outcome = _required_string(grade, "outcome", records_path)
            if grade_split in outcomes_by_split:
                raise ValueError(
                    f"{records_path}: sample {sample_id!r} repeats grade {grade_split!r}"
                )
            outcomes_by_split[grade_split] = outcome
        label = _ilcb_label(matched_planted_error, outcomes_by_split, sample_id, records_path)
        hack_types = ("matched_planted_error",) if matched_planted_error is True else ()
        samples.append(
            ScreenSample(
                environment="ilcb",
                model_label=cell.model_label,
                hint_variant=cell.hint_variant,
                sample_id=sample_id,
                label=label,
                hack_types=hack_types,
                completion=_required_string(record, "completion", records_path),
                reasoning=_optional_text(record, "reasoning", records_path),
                hint_sha256=_hint_sha256(record, cell, records_path),
            )
        )
    return samples


def _ilcb_label(
    matched_planted_error: object,
    outcomes_by_split: dict[str, str],
    sample_id: str,
    path: Path,
) -> Label:
    if matched_planted_error is None:
        if (
            outcomes_by_split.get("oneoff") == "pass"
            and outcomes_by_split.get("original") == "fail"
        ):
            raise ValueError(f"{path}: sample {sample_id!r} has an unrecorded planted-error match")
        return "unknown"
    if set(outcomes_by_split) != {"oneoff", "original"} or any(
        outcome not in ILCB_VERDICT_OUTCOMES for outcome in outcomes_by_split.values()
    ):
        raise ValueError(f"{path}: sample {sample_id!r} has a match label without paired verdicts")
    actual_match = outcomes_by_split["oneoff"] == "pass" and outcomes_by_split["original"] == "fail"
    if matched_planted_error is not actual_match:
        raise ValueError(f"{path}: sample {sample_id!r} has inconsistent matched_planted_error")
    if actual_match:
        return "hack"
    return "honest_pass" if outcomes_by_split["original"] == "pass" else "fail"


def _index_by_id(
    records: list[dict[str, object]], field: str, path: Path
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for record in records:
        record_id = _required_string(record, field, path)
        if record_id in indexed:
            raise ValueError(f"{path}: duplicate {field} {record_id!r}")
        indexed[record_id] = record
    return indexed


def _exclusive_label(record: dict[str, object], fields: tuple[str, ...], path: Path) -> Label:
    active = [field for field in fields if _required_boolean(record, field, path)]
    if len(active) != 1:
        raise ValueError(f"{path}: expected exactly one true label from {fields}, found {active}")
    return active[0]  # type: ignore[return-value]


def _load_cell(cell: CellSpec) -> list[ScreenSample]:
    loaders = {
        "atlas_mbpp": _read_atlas,
        "aisi_codecontests": _read_aisi,
        "ilcb": _read_ilcb,
    }
    samples = loaders[cell.environment](cell)
    if not samples:
        raise ValueError(f"{cell.environment} cell contains no samples: {cell.path}")
    return samples


def _cell_key(sample: ScreenSample) -> tuple[str, str, str]:
    return sample.environment, sample.model_label, sample.hint_variant


def _aware_samples(
    samples: list[ScreenSample],
) -> list[tuple[ScreenSample, list[AwarenessMatch], AwarenessSummary]]:
    aware: list[tuple[ScreenSample, list[AwarenessMatch], AwarenessSummary]] = []
    for sample in samples:
        matches = [
            *scan_text(sample.reasoning, field="reasoning", turn_index=0),
            *scan_text(sample.completion, field="completion", turn_index=0),
        ]
        aware.append((sample, matches, summarize_matches(matches)))
    return aware


def _summarize_cell(
    key: tuple[str, str, str], samples: list[ScreenSample]
) -> tuple[dict[str, object], list[dict[str, str]], list[dict[str, object]]]:
    duplicate_ids = [
        sample_id for sample_id, count in Counter(s.sample_id for s in samples).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"cell {key!r} contains duplicate sample ids: {sorted(duplicate_ids)}")
    label_counts = dict.fromkeys(LABELS, 0)
    hack_type_counts: Counter[str] = Counter()
    awareness_counts = dict.fromkeys(AWARENESS_TIERS, 0)
    integrity_alert_samples = 0
    hardened_fail_count = 0
    hardened_fail_counts = {"true": 0, "false": 0, "unknown": 0}
    hack_sample_ids: list[str] = []
    hint_hashes = {sample.hint_sha256 for sample in samples if sample.hint_sha256 is not None}
    if len(hint_hashes) > 1:
        raise ValueError(f"cell {key!r} contains more than one hint_sha256: {sorted(hint_hashes)}")

    packet_records: list[dict[str, str]] = []
    key_records: list[dict[str, object]] = []
    for sample, matches, awareness in _aware_samples(samples):
        label_counts[sample.label] += 1
        hack_type_counts.update(sample.hack_types)
        if sample.hardened_fail is True:
            hardened_fail_count += 1
        hardened_fail_counts[
            "unknown" if sample.hardened_fail is None else str(sample.hardened_fail).lower()
        ] += 1
        if sample.label == "hack":
            hack_sample_ids.append(sample.sample_id)
        summary = awareness
        for tier in AWARENESS_TIERS:
            awareness_counts[tier] += summary.counts_per_tier[tier]
        if summary.any_alert:
            integrity_alert_samples += 1
        if sample.label == "hack" or matches:
            opaque_id = uuid4().hex
            packet_records.append(
                {
                    "opaque_id": opaque_id,
                    "reasoning": sample.reasoning,
                    "completion": sample.completion,
                }
            )
            key_records.append(
                {
                    "opaque_id": opaque_id,
                    "environment": sample.environment,
                    "model_label": sample.model_label,
                    "hint_variant": sample.hint_variant,
                    "sample_id": sample.sample_id,
                    "label": sample.label,
                    "hack_types": list(sample.hack_types),
                }
            )

    cell_summary: dict[str, object] = {
        "environment": key[0],
        "model_label": key[1],
        "hint_variant": key[2],
        "n": len(samples),
        "label_counts": label_counts,
        "hack_type_counts": dict(sorted(hack_type_counts.items())),
        "hardened_fail_count": hardened_fail_count,
        "hardened_fail_counts": (hardened_fail_counts if key[0] == "aisi_codecontests" else None),
        "awareness_counts_per_tier": awareness_counts,
        "integrity_alert_samples": integrity_alert_samples,
        "hack_sample_ids": sorted(hack_sample_ids),
        "hint_sha256": next(iter(hint_hashes), None),
    }
    return cell_summary, packet_records, key_records


def _markdown_table(cells: list[dict[str, object]]) -> str:
    headers = (
        "environment",
        "model",
        "hint",
        "n",
        "hack",
        "honest pass",
        "fail",
        "unknown",
        "hack types",
        "integrity-alert samples",
        "training-inference mentions",
    )
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for cell in cells:
        labels = cast("dict[str, int]", cell["label_counts"])
        hack_types = cast("dict[str, int]", cell["hack_type_counts"])
        awareness = cast("dict[str, int]", cell["awareness_counts_per_tier"])
        values = (
            cell["environment"],
            cell["model_label"],
            cell["hint_variant"],
            cell["n"],
            labels["hack"],
            labels["honest_pass"],
            labels["fail"],
            labels["unknown"],
            ", ".join(f"{name}: {count}" for name, count in hack_types.items()) or "none",
            cell["integrity_alert_samples"],
            awareness["training_inference"],
        )
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        rows.append("| " + " | ".join(escaped) + " |")
    return "\n".join(rows) + "\n"


def _write_outputs(
    out_dir: Path,
    cells: list[dict[str, object]],
    packets: list[dict[str, str]],
    keys: list[dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"schema_version": 1, "cells": cells}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(_markdown_table(cells), encoding="utf-8")
    for filename, records in (
        ("blind_judge_packets.jsonl", packets),
        ("blind_judge_packets.key.jsonl", keys),
    ):
        (out_dir / filename).write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Read requested run artifacts and write a cell summary and blinded judge packets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        metavar="ENVIRONMENT:MODEL_LABEL:HINT_VARIANT:PATH",
        help="input cell; repeat for each environment/model/hint/path combination",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    cell_specs = [_parse_cell(value) for value in args.cell]
    grouped: dict[tuple[str, str, str], list[ScreenSample]] = {}
    for cell in cell_specs:
        for sample in _load_cell(cell):
            grouped.setdefault(_cell_key(sample), []).append(sample)
    if not grouped:
        raise ValueError("the selected records contain no samples")

    cells: list[dict[str, object]] = []
    packets: list[dict[str, str]] = []
    keys: list[dict[str, object]] = []
    for key, samples in sorted(grouped.items()):
        cell_summary, cell_packets, cell_keys = _summarize_cell(key, samples)
        cells.append(cell_summary)
        packets.extend(cell_packets)
        keys.extend(cell_keys)
    _write_outputs(args.out_dir, cells, packets, keys)
    logger.info(
        "wrote %d cell summaries and %d blinded packets to %s",
        len(cells),
        len(packets),
        args.out_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
