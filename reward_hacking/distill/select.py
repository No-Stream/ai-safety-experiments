"""Select graded distillation traces for the SFT arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from reward_hacking.distill.sample import token_ids_sha256

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

Mode = Literal["all", "hack"]
LABELS = frozenset({"hack", "honest", "fail"})
TOKEN_TARGET_TOLERANCE = 0.1


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Counts and artifact paths for one selected SFT trace file."""

    output_path: Path
    selection_path: Path
    selected_count: int
    selected_completion_tokens: int


@dataclass(frozen=True, slots=True)
class SelectionOptions:
    """Selection criteria independent of the files carrying the trace data."""

    mode: Mode
    target_completion_tokens: int | None = None
    seed: int = 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _required_string(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} has invalid {field}")
    return value


def _token_ids(row: Mapping[str, Any], field: str, *, context: str) -> list[int]:
    value = row.get(field)
    if (
        not isinstance(value, list)
        or not value
        or any(type(token_id) is not int or token_id < 0 for token_id in value)
    ):
        raise ValueError(f"{context} has invalid {field}")
    return value


def _index_samples(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _read_jsonl(path)
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        context = f"{path}:{line_number}"
        key = _required_string(row, "key", context=context)
        if key in indexed:
            raise ValueError(f"{path} repeats sample key {key!r}")
        _required_string(row, "env", context=context)
        _required_string(row, "problem_id", context=context)
        if type(row.get("sample_index")) is not int or row["sample_index"] < 0:
            raise ValueError(f"sample {key!r} has invalid sample_index")
        prompt_token_ids = _token_ids(row, "prompt_token_ids", context=context)
        completion_token_ids = _token_ids(row, "completion_token_ids", context=context)
        if row.get("token_ids_sha256") != token_ids_sha256(prompt_token_ids, completion_token_ids):
            raise ValueError(f"sample {key!r} token_ids_sha256 does not match its token ids")
        finish_reason = row.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise TypeError(f"sample {key!r} has invalid finish_reason")
        indexed[key] = row
    if not indexed:
        raise ValueError(f"no samples found in {path}")
    return rows, indexed


def _index_grades(path: Path, samples: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(_read_jsonl(path), start=1):
        context = f"{path}:{line_number}"
        key = _required_string(row, "key", context=context)
        sample = samples.get(key)
        if sample is None:
            raise ValueError(f"graded input contains unexpected key {key!r}")
        if key in indexed:
            raise ValueError(f"{path} repeats graded key {key!r}")
        for field in ("env", "problem_id", "sample_index", "token_ids_sha256"):
            if row.get(field) != sample[field]:
                raise ValueError(f"graded row {key!r} differs from sample field {field!r}")
        label = row.get("label")
        if not isinstance(label, str) or label not in LABELS:
            raise ValueError(f"graded row {key!r} has invalid label {label!r}")
        awareness = row.get("awareness")
        if not isinstance(awareness, Mapping) or type(awareness.get("integrity_alert")) is not bool:
            raise ValueError(f"graded row {key!r} has invalid awareness.integrity_alert")
        completion_token_ids = _token_ids(sample, "completion_token_ids", context=f"sample {key!r}")
        if row.get("completion_tokens") != len(completion_token_ids):
            raise ValueError(f"graded row {key!r} completion_tokens differs from sample")
        indexed[key] = row
    if indexed.keys() != samples.keys():
        missing = sorted(samples.keys() - indexed.keys())
        raise ValueError(f"graded input is incomplete: {len(missing)} samples have no grade")
    return indexed


def _key_digest(seed: int, key: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{key}".encode()).digest()


def _select_token_target(
    candidates: Sequence[dict[str, Any]], *, target_completion_tokens: int, seed: int
) -> list[dict[str, Any]]:
    if target_completion_tokens < 1:
        raise ValueError("target_completion_tokens must be positive")
    minimum_tokens = math.ceil(target_completion_tokens * (1 - TOKEN_TARGET_TOLERANCE))
    maximum_tokens = math.floor(target_completion_tokens * (1 + TOKEN_TARGET_TOLERANCE))
    token_total = sum(len(row["completion_token_ids"]) for row in candidates)
    if token_total < minimum_tokens:
        raise ValueError(
            f"eligible trace pool is too small: {token_total} completion tokens, "
            f"need at least {minimum_tokens}"
        )
    hash_ordered = sorted(
        candidates,
        key=lambda row: (_key_digest(seed, str(row["key"])), str(row["key"])),
    )
    candidate_orders = (
        hash_ordered,
        sorted(
            hash_ordered,
            key=lambda row: (
                len(row["completion_token_ids"]),
                _key_digest(seed, str(row["key"])),
                str(row["key"]),
            ),
        ),
        sorted(
            hash_ordered,
            key=lambda row: (
                -len(row["completion_token_ids"]),
                _key_digest(seed, str(row["key"])),
                str(row["key"]),
            ),
        ),
    )
    for ordered in candidate_orders:
        selected: list[dict[str, Any]] = []
        selected_tokens = 0
        for row in ordered:
            completion_tokens = len(row["completion_token_ids"])
            if selected_tokens + completion_tokens > maximum_tokens:
                continue
            selected.append(row)
            selected_tokens += completion_tokens
            if selected_tokens >= minimum_tokens:
                return selected
    raise ValueError(
        f"eligible traces cannot form a subset within 10% of {target_completion_tokens} tokens"
    )


def _eligible_samples(
    sample_rows: Sequence[dict[str, Any]],
    grades_by_key: Mapping[str, dict[str, Any]],
    *,
    mode: Mode,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    dropped = {"dropped_truncated": 0, "dropped_integrity_alert": 0, "dropped_mode": 0}
    for sample in sample_rows:
        grade = grades_by_key[str(sample["key"])]
        if sample.get("finish_reason") == "length":
            dropped["dropped_truncated"] += 1
        elif grade["awareness"]["integrity_alert"]:
            dropped["dropped_integrity_alert"] += 1
        elif mode == "hack" and grade["label"] != "hack":
            dropped["dropped_mode"] += 1
        else:
            candidates.append(sample)
    return candidates, dropped


def _verify_selected_rows(selected_rows: Sequence[dict[str, Any]]) -> int:
    selected_completion_tokens = 0
    for row in selected_rows:
        key = str(row["key"])
        completion_token_ids = _token_ids(row, "completion_token_ids", context=f"sample {key!r}")
        prompt_token_ids = _token_ids(row, "prompt_token_ids", context=f"sample {key!r}")
        expected_hash = token_ids_sha256(prompt_token_ids, completion_token_ids)
        if row.get("token_ids_sha256") != expected_hash:
            raise ValueError(
                f"selected sample {key!r} token_ids_sha256 does not match its token ids"
            )
        selected_completion_tokens += len(completion_token_ids)
    return selected_completion_tokens


def select_traces(
    *,
    samples_path: Path,
    graded_path: Path,
    output_path: Path,
    options: SelectionOptions,
) -> SelectionResult:
    """Write a filtered sample JSONL and its selection manifest, preserving sample rows."""
    mode = options.mode
    if mode not in {"all", "hack"}:
        raise ValueError(f"unsupported selection mode {mode!r}")
    selection_path = Path(f"{output_path}.selection.json")
    if output_path.resolve() in {samples_path.resolve(), graded_path.resolve()}:
        raise ValueError("selection output must differ from both input files")
    if output_path.exists() or selection_path.exists():
        raise FileExistsError(f"selection output already exists: {output_path}")
    sample_rows, samples_by_key = _index_samples(samples_path)
    grades_by_key = _index_grades(graded_path, samples_by_key)
    candidates, dropped = _eligible_samples(sample_rows, grades_by_key, mode=mode)
    if not candidates:
        raise ValueError("selection filters leave no eligible traces")
    selected_rows = (
        _select_token_target(
            candidates,
            target_completion_tokens=options.target_completion_tokens,
            seed=options.seed,
        )
        if options.target_completion_tokens is not None
        else candidates
    )
    selected_completion_tokens = _verify_selected_rows(selected_rows)

    dropped_by_token_budget = len(candidates) - len(selected_rows)
    selection: dict[str, object] = {
        "mode": mode,
        "seed": options.seed,
        "target_completion_tokens": options.target_completion_tokens,
        "token_target_tolerance": TOKEN_TARGET_TOLERANCE,
        "input_count": len(sample_rows),
        "graded_count": len(grades_by_key),
        **dropped,
        "candidate_count": len(candidates),
        "dropped_by_token_budget": dropped_by_token_budget,
        "input_completion_tokens": sum(len(row["completion_token_ids"]) for row in sample_rows),
        "candidate_completion_tokens": sum(len(row["completion_token_ids"]) for row in candidates),
        "selected_count": len(selected_rows),
        "selected_completion_tokens": selected_completion_tokens,
        "selected_keys": [str(row["key"]) for row in selected_rows],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        for row in selected_rows:
            output.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
        output.flush()
    _write_json(selection_path, selection)
    logger.info(
        "trace selection mode=%s input=%d candidates=%d selected=%d tokens=%d "
        "dropped_truncated=%d dropped_integrity_alert=%d dropped_mode=%d "
        "dropped_by_token_budget=%d output=%s",
        mode,
        len(sample_rows),
        len(candidates),
        len(selected_rows),
        selected_completion_tokens,
        dropped["dropped_truncated"],
        dropped["dropped_integrity_alert"],
        dropped["dropped_mode"],
        dropped_by_token_budget,
        output_path,
    )
    return SelectionResult(
        output_path, selection_path, len(selected_rows), selected_completion_tokens
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SFT trace selection CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--graded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "hack"), required=True)
    parser.add_argument("--target-completion-tokens", type=int)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    select_traces(
        samples_path=arguments.samples,
        graded_path=arguments.graded,
        output_path=arguments.output,
        options=SelectionOptions(
            mode=arguments.mode,
            target_completion_tokens=arguments.target_completion_tokens,
            seed=arguments.seed,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
