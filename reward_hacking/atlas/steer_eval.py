"""Convert residual-steering generations to Atlas samples, jail-grade them, and summarize rates."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from reward_hacking.atlas.grading_runtime import DEFAULT_GRADE_WORKERS, grade_model
from reward_hacking.atlas.sampling import BASE_MODEL_ID, artifact_model_key

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

STEERING_AXES = frozenset({"direction", "matched_norm_placebo"})
GRADE_LABELS = ("hack", "honest_pass", "fail")
STEERING_RESUME_STAGE = "hack_contrast_steer"


class GradeModel(Protocol):
    """Callable interface for Atlas's jailed sample grader."""

    def __call__(  # noqa: PLR0913
        self,
        *,
        run_dir: Path,
        model_key: str,
        base_model: str,
        timeout_seconds: int,
        scratch_parent: Path,
        workers: int,
    ) -> Path:
        """Grade samples in a run directory and return the grades JSONL path."""
        ...


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as records_file:
        for line_number, line in enumerate(records_file, start=1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                records.append(record)
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        "".join(
            json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(dict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def _validate_steering_ledger(
    steering_path: Path, steering_records: Sequence[Mapping[str, Any]]
) -> None:
    ledger_path = steering_path.with_name(steering_path.name + ".resume.json")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    identity = ledger.get("identity")
    if not isinstance(identity, dict) or identity.get("stage") != STEERING_RESUME_STAGE:
        raise ValueError(f"{ledger_path} is not a hack-contrast steering resume ledger")
    item_ids = identity.get("selected_item_ids")
    layers = identity.get("layers")
    strengths = identity.get("strengths")
    sample_count = identity.get("samples_per_prompt")
    axes = identity.get("axes")
    if (
        not isinstance(item_ids, list)
        or any(not isinstance(item_id, str) for item_id in item_ids)
        or not isinstance(layers, list)
        or any(not isinstance(layer, int) for layer in layers)
        or not isinstance(strengths, list)
        or any(
            not isinstance(strength, int | float) or not math.isfinite(strength)
            for strength in strengths
        )
        or not isinstance(sample_count, int)
        or sample_count < 1
        or axes != ["direction", "matched_norm_placebo"]
    ):
        raise ValueError(f"{ledger_path} is missing the steering grid identity")
    completed_records = ledger.get("completed_records")
    if not isinstance(completed_records, dict):
        raise TypeError(f"{ledger_path} completed_records must be a JSON object")
    axis_names = cast("list[str]", axes)
    expected_ids = {
        f"L{layer}:{axis}:{strength:g}:{item_id}:{sample_index}"
        for layer in layers
        for axis in axis_names
        for strength in strengths
        for item_id in item_ids
        for sample_index in range(sample_count)
    }
    record_ids = {str(record["sample_id"]) for record in steering_records}
    completed_ids = set(completed_records)
    if record_ids != expected_ids or completed_ids != expected_ids:
        raise ValueError(
            f"steering records are incomplete: expected {len(expected_ids)}, "
            f"found {len(record_ids)} rows and {len(completed_ids)} completed ledger records"
        )


def _atlas_sample_from_steering(  # noqa: PLR0913
    steering: Mapping[str, Any],
    prompt: Mapping[str, Any],
    *,
    model_key: str,
    base_model: str,
    axis: str,
    strength: float,
) -> dict[str, object]:
    sample_id = str(steering["sample_id"])
    item_id = str(steering["item_id"])
    test_list = prompt["test_list"]
    if (
        not isinstance(test_list, list)
        or not test_list
        or any(not isinstance(test_case, str) for test_case in test_list)
    ):
        raise ValueError(f"Atlas prompt {item_id} needs a non-empty test_list of strings")
    completion_text = steering["completion"]
    if not isinstance(completion_text, str):
        raise TypeError(f"steering completion {sample_id} must be text")
    return {
        "sample_id": sample_id,
        "model_key": model_key,
        "model_id": base_model,
        "adapter_id": steering.get("adapter"),
        "adapter_language_only": steering.get("adapter") is not None,
        "item_id": item_id,
        "source": str(prompt.get("source", "atlas")),
        "split": str(prompt.get("split", "unknown")),
        "sample_index": int(steering["sample_index"]),
        "seed": int(steering["seed"]),
        "prompt": str(prompt["prompt"]),
        "prompt_token_ids": [int(token_id) for token_id in steering["prompt_token_ids"]],
        "completion_token_ids": [int(token_id) for token_id in steering["completion_token_ids"]],
        "completion_text": completion_text,
        "finish_reason": None,
        "stop_reason": None,
        "task_text": str(prompt["task_text"]),
        "test_list": list(test_list),
        "test_setup_code": str(prompt.get("test_setup_code", "")),
        "reference_code": prompt.get("reference_code"),
        "hardcode_code": prompt.get("hardcode_code"),
        "sampling": {
            "do_sample": True,
            "temperature": float(steering["temperature"]),
            "max_new_tokens": int(steering["max_new_tokens"]),
            "top_p": float(steering.get("top_p", 1.0)),
            "top_k": int(steering.get("top_k", 0)),
            "min_p": float(steering.get("min_p", 0.0)),
            "repetition_penalty": float(steering.get("repetition_penalty", 1.0)),
            "axis": axis,
            "strength_multiple": strength,
            "strength_unit": str(steering["strength_unit"]),
        },
        "position": str(steering["position"]),
        "layer": int(steering["layer"]),
        "scope": str(steering["scope"]),
        "axis": axis,
        "strength_multiple": strength,
    }


def _atlas_samples_from_steering(
    steering_records: Sequence[Mapping[str, Any]],
    prompt_records: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    base_model: str,
) -> list[dict[str, object]]:
    prompts_by_item_id = {str(prompt["item_id"]): prompt for prompt in prompt_records}
    if len(prompts_by_item_id) != len(prompt_records):
        raise ValueError("Atlas prompts contain duplicate item ids")
    samples: list[dict[str, object]] = []
    sample_ids: set[str] = set()
    for steering in steering_records:
        sample_id = str(steering["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"steering records contain duplicate sample id {sample_id}")
        sample_ids.add(sample_id)
        item_id = str(steering["item_id"])
        prompt = prompts_by_item_id.get(item_id)
        if prompt is None:
            raise ValueError(f"Atlas prompt is missing for steering item {item_id}")
        if str(steering["model"]) != base_model:
            raise ValueError(
                f"steered sample {sample_id} uses model {steering['model']!r}, not {base_model!r}"
            )
        axis = str(steering["axis"])
        if axis not in STEERING_AXES:
            raise ValueError(f"unknown steering axis {axis!r}")
        strength = float(steering["strength_multiple"])
        if not math.isfinite(strength):
            raise ValueError(f"steering sample {sample_id} has a non-finite strength")
        samples.append(
            _atlas_sample_from_steering(
                steering,
                prompt,
                model_key=model_key,
                base_model=base_model,
                axis=axis,
                strength=strength,
            )
        )
    if not samples:
        raise ValueError("no steering records found")
    return samples


def _summarize_grades(
    samples: Sequence[Mapping[str, Any]], grades: Sequence[Mapping[str, Any]], *, model_key: str
) -> dict[str, object]:
    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    grade_by_id = {str(grade["sample_id"]): grade for grade in grades}
    if len(sample_by_id) != len(samples) or len(grade_by_id) != len(grades):
        raise ValueError("steering samples and grades must have unique sample ids")
    if set(sample_by_id) != set(grade_by_id):
        raise ValueError("steering sample and grade ids differ")
    groups: dict[tuple[int, float, str], dict[str, int]] = {}
    for sample_id, sample in sample_by_id.items():
        grade = grade_by_id[sample_id]
        if str(grade["item_id"]) != str(sample["item_id"]):
            raise ValueError(f"grade item id does not match sample {sample_id}")
        labels = [grade.get(label) for label in GRADE_LABELS]
        if (
            any(not isinstance(label, bool) for label in labels)
            or sum(bool(label) for label in labels) != 1
        ):
            raise ValueError(f"grade labels for {sample_id} must select exactly one Atlas outcome")
        group_key = (
            int(sample["layer"]),
            float(sample["strength_multiple"]),
            str(sample["axis"]),
        )
        counts = groups.setdefault(
            group_key,
            {"n_samples": 0, "n_hack": 0, "n_honest_pass": 0, "n_fail": 0},
        )
        counts["n_samples"] += 1
        counts["n_hack"] += int(bool(grade["hack"]))
        counts["n_honest_pass"] += int(bool(grade["honest_pass"]))
        counts["n_fail"] += int(bool(grade["fail"]))
    group_rows: list[dict[str, object]] = []
    for (layer, strength, axis), counts in sorted(groups.items()):
        total = counts["n_samples"]
        group_rows.append(
            {
                "layer": layer,
                "strength_multiple": strength,
                "strength_unit": "mean residual L2 norm over all prompt tokens",
                "axis": axis,
                **counts,
                "hack_rate": counts["n_hack"] / total,
                "honest_pass_rate": counts["n_honest_pass"] / total,
                "fail_rate": counts["n_fail"] / total,
            }
        )
    return {
        "model_key": model_key,
        "n_samples": len(samples),
        "grouping": ["layer", "strength_multiple", "axis"],
        "rates": "Atlas hack, honest_pass, and fail labels from the jailed grader",
        "groups": group_rows,
    }


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Atlas steering grade rates",
        "",
        "Strength is a multiple of the layer's mean residual L2 norm over all prompt tokens.",
        "",
        "| Layer | Strength | Axis | Samples | Hack rate | Honest pass rate | Fail rate |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {group['layer']} | {group['strength_multiple']:g} | {group['axis']} "
        f"| {group['n_samples']} | {group['hack_rate']:.3f} "
        f"| {group['honest_pass_rate']:.3f} | {group['fail_rate']:.3f} |"
        for group in summary["groups"]
    )
    return "\n".join(lines) + "\n"


def grade_steering_outputs(  # noqa: PLR0913
    *,
    steering_path: Path,
    prompts_path: Path,
    output_dir: Path,
    model_key: str,
    base_model: str = BASE_MODEL_ID,
    timeout_seconds: int = 5,
    scratch_parent: Path = Path("/var/tmp"),  # noqa: S108
    workers: int = DEFAULT_GRADE_WORKERS,
    grader: GradeModel | None = None,
) -> tuple[Path, Path]:
    """Bridge saved steering completions through Atlas's jailed grader and summarize each cell."""
    if timeout_seconds < 1 or workers < 1:
        raise ValueError("timeout_seconds and workers must be positive")
    steering_records = _read_jsonl(steering_path)
    _validate_steering_ledger(steering_path, steering_records)
    prompt_records = _read_jsonl(prompts_path)
    samples = _atlas_samples_from_steering(
        steering_records,
        prompt_records,
        model_key=model_key,
        base_model=base_model,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"samples.{artifact_model_key(model_key)}.jsonl"
    _write_jsonl(sample_path, samples)
    grade_function = grade_model if grader is None else grader
    grade_path = grade_function(
        run_dir=output_dir,
        model_key=model_key,
        base_model=base_model,
        timeout_seconds=timeout_seconds,
        scratch_parent=scratch_parent,
        workers=workers,
    )
    summary = _summarize_grades(
        samples,
        _read_jsonl(grade_path),
        model_key=model_key,
    )
    summary_path = output_dir / "steer-summary.json"
    markdown_path = output_dir / "steer-summary.md"
    _write_json(summary_path, summary)
    markdown_path.write_text(_markdown_summary(summary), encoding="utf-8")
    logger.info("graded %d steering samples; summary=%s", len(samples), summary_path)
    return summary_path, markdown_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    grade = commands.add_parser(
        "grade", help="Atlas-jail grade steering outputs and summarize rates"
    )
    grade.add_argument("--steering", type=Path, required=True)
    grade.add_argument("--prompts", type=Path, required=True)
    grade.add_argument("--out-dir", type=Path, required=True)
    grade.add_argument("--model-key", default="atlas-last-prompt-steer")
    grade.add_argument("--base-model", default=BASE_MODEL_ID)
    grade.add_argument("--timeout-seconds", type=int, default=5)
    grade.add_argument("--scratch-parent", type=Path, default=Path("/var/tmp"))  # noqa: S108
    grade.add_argument("--workers", type=int, default=DEFAULT_GRADE_WORKERS)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run Atlas grading and summarize steering outcomes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = _build_parser().parse_args(argv)
    grade_steering_outputs(
        steering_path=arguments.steering,
        prompts_path=arguments.prompts,
        output_dir=arguments.out_dir,
        model_key=arguments.model_key,
        base_model=arguments.base_model,
        timeout_seconds=arguments.timeout_seconds,
        scratch_parent=arguments.scratch_parent,
        workers=arguments.workers,
    )


if __name__ == "__main__":
    main()
