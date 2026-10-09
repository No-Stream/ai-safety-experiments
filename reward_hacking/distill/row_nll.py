"""Score exact SFT rows under the base model, excluding masked environment tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from reward_hacking.distill.train import completion_logps

if TYPE_CHECKING:
    from torch.nn import Module

logger = logging.getLogger(__name__)

_ROW_KEY_PATTERN = re.compile(r"ladder:[0-9a-f]{64}")


def _token_list(row: Mapping[str, object], field: str, *, allow_empty: bool) -> list[int]:
    value = row.get(field)
    if not isinstance(value, list) or (not allow_empty and not value):
        raise TypeError(
            f"training row {field!r} must be a {'possibly empty' if allow_empty else 'nonempty'} list"
        )
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise TypeError(f"training row {field!r} must contain nonnegative integer token ids")
    return cast("list[int]", value)


def score_row_action_nll(
    model: Module,
    row: Mapping[str, object],
    *,
    chunk_tokens: int = 512,
) -> dict[str, int | float | str]:
    """Return the mean base-model NLL over loss-masked action tokens in one row."""
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    row_key = row.get("key")
    if not isinstance(row_key, str) or not row_key:
        raise TypeError("training row key must be a nonempty string")
    prompt_token_ids = _token_list(row, "prompt_token_ids", allow_empty=False)
    completion_token_ids = _token_list(row, "completion_token_ids", allow_empty=False)
    loss_mask_value = row.get("loss_mask")
    if not isinstance(loss_mask_value, list) or len(loss_mask_value) != len(completion_token_ids):
        raise ValueError("loss_mask must align exactly with completion_token_ids")
    if any(type(value) is not int or value not in {0, 1} for value in loss_mask_value):
        raise ValueError("loss_mask values must be integer zero or one")
    active_mask = torch.tensor(loss_mask_value, dtype=torch.bool)
    action_token_count = int(active_mask.sum().item())
    if action_token_count == 0:
        raise ValueError("loss_mask must select at least one action token")

    input_ids = torch.tensor(
        [prompt_token_ids + completion_token_ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    with torch.inference_mode():
        completion_logprobs = completion_logps(
            model,
            input_ids,
            len(prompt_token_ids),
            chunk_tokens=chunk_tokens,
        )[0]
    action_nlls = -completion_logprobs.float().cpu()[active_mask]
    mean_nll = float(action_nlls.mean().item())
    max_nll = float(action_nlls.max().item())
    if not math.isfinite(mean_nll) or not math.isfinite(max_nll):
        raise ValueError("base model returned non-finite action-token NLL")
    return {
        "row_key": row_key,
        "action_tokens": action_token_count,
        "mean_nll": mean_nll,
        "max_action_token_nll": max_nll,
    }


def _read_training_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_keys: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"training rows contain a blank line at {path.name}:{line_number}")
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"training row {line_number} must be a JSON object")
            row = cast("dict[str, object]", value)
            row_key = row.get("key")
            if not isinstance(row_key, str) or _ROW_KEY_PATTERN.fullmatch(row_key) is None:
                raise ValueError(f"training row {line_number} must have an opaque hashed row key")
            if row_key in seen_keys:
                raise ValueError("training rows contain duplicate keys")
            seen_keys.add(row_key)
            rows.append(row)
    if not rows:
        raise ValueError("training rows must not be empty")
    return rows


def _report_identity(rows_path: Path, *, model: str, revision: str | None) -> dict[str, object]:
    return {
        "format_version": 1,
        "model": model,
        "model_revision": revision,
        "adapter": None,
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
    }


def _resolve_base_model_revision(model_name: str, revision: str | None) -> str | None:
    configuration = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=True,
    )
    resolved_revision = configuration._commit_hash or revision  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    if resolved_revision is None and not Path(model_name).is_dir():
        raise RuntimeError("cached base model config did not resolve a Hub revision")
    return resolved_revision


def _load_resume_results(
    output_path: Path,
    identity: Mapping[str, object],
    expected_row_keys: set[str],
) -> dict[str, dict[str, object]]:
    if not output_path.exists():
        return {}
    saved: object = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(saved, Mapping):
        raise TypeError("existing row NLL output must be a JSON object")
    if any(saved.get(key) != value for key, value in identity.items()):
        raise ValueError("existing row NLL output belongs to different rows or base weights")
    rows_value = saved.get("rows")
    if not isinstance(rows_value, list):
        raise TypeError("existing row NLL output must contain a rows list")
    results: dict[str, dict[str, object]] = {}
    for value in rows_value:
        if not isinstance(value, Mapping):
            raise TypeError("existing row NLL entries must be objects")
        row_key = value.get("row_key")
        action_tokens = value.get("action_tokens")
        mean_nll = value.get("mean_nll")
        max_nll = value.get("max_action_token_nll")
        if (
            not isinstance(row_key, str)
            or row_key not in expected_row_keys
            or row_key in results
            or type(action_tokens) is not int
            or action_tokens < 1
            or isinstance(mean_nll, bool)
            or not isinstance(mean_nll, int | float)
            or not math.isfinite(mean_nll)
            or isinstance(max_nll, bool)
            or not isinstance(max_nll, int | float)
            or not math.isfinite(max_nll)
        ):
            raise ValueError("existing row NLL output contains an invalid row result")
        results[row_key] = dict(value)
    return results


def _write_report_atomically(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(dict(report), temporary_file, indent=2, sort_keys=True, allow_nan=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _nll_report(
    identity: Mapping[str, object],
    row_results: Mapping[str, Mapping[str, object]],
    ordered_keys: Sequence[str],
) -> dict[str, object]:
    ordered_results = [dict(row_results[key]) for key in ordered_keys if key in row_results]
    mean_values = [float(cast("float", value["mean_nll"])) for value in ordered_results]
    return {
        **identity,
        "complete": len(ordered_results) == len(ordered_keys),
        "row_count": len(ordered_keys),
        "scored_row_count": len(ordered_results),
        "mean_row_nll": sum(mean_values) / len(mean_values) if mean_values else None,
        "max_row_mean_nll": max(mean_values, default=None),
        "rows": ordered_results,
    }


def score_rows_file(
    rows_path: Path,
    output_path: Path,
    *,
    model_name: str,
    revision: str | None,
    chunk_tokens: int,
) -> None:
    """Score or resume exact rows under one explicitly loaded base model on one GPU."""
    rows = _read_training_rows(rows_path)
    ordered_keys = [cast("str", row["key"]) for row in rows]
    resolved_revision = _resolve_base_model_revision(model_name, revision)
    identity = _report_identity(rows_path, model=model_name, revision=resolved_revision)
    row_results = _load_resume_results(output_path, identity, set(ordered_keys))
    if len(row_results) == len(ordered_keys):
        _write_report_atomically(output_path, _nll_report(identity, row_results, ordered_keys))
        logger.info("all base row NLL results are already complete rows=%d", len(ordered_keys))
        return

    repository_root = Path(__file__).resolve().parents[2]
    gpu_preflight = repository_root / "scripts" / "gpu_preflight.py"
    subprocess.run(["/usr/bin/python3", str(gpu_preflight)], check=True)  # noqa: S603 - fixed repo-local preflight command
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("row NLL scoring requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    logger.info(
        "loading base model for row NLL device=%s free_gib=%.2f total_gib=%.2f",
        torch.cuda.get_device_name(device),
        free_bytes / 1024**3,
        total_bytes / 1024**3,
    )
    model = cast(
        "Module",
        AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=resolved_revision,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ),
    )
    model.to(device)
    model.eval()
    for row in rows:
        row_key = cast("str", row["key"])
        if row_key in row_results:
            continue
        row_results[row_key] = cast(
            "dict[str, object]", score_row_action_nll(model, row, chunk_tokens=chunk_tokens)
        )
        _write_report_atomically(output_path, _nll_report(identity, row_results, ordered_keys))
        logger.info("scored base row action NLL completed=%d total=%d", len(row_results), len(rows))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--revision")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run exact-row NLL scoring and preserve completed rows for idempotent resume."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    if args.chunk_tokens < 1:
        raise ValueError("--chunk-tokens must be positive")
    score_rows_file(
        args.rows,
        args.output,
        model_name=args.model,
        revision=args.revision,
        chunk_tokens=args.chunk_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
