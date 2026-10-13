"""Score one diagnostic condition with one engine and bounded candidate context memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from argparse import Namespace
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import AutoTokenizer

from reward_hacking.distill.diagnostic_moves import CONDITIONS, build_decision_row
from reward_hacking.distill.diagnostic_tables import _read  # pyright: ignore[reportPrivateUsage]
from reward_hacking.distill.score import MODEL_LENGTH_MULTIPLE, TeacherScoringRuntime, score_round
from reward_hacking.organism_rl.evaluate import resolve_model_revision
from reward_hacking.organism_rl.generate import require_gitignored_output_dir
from reward_hacking.semantic_identity import model_identity

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)
DEFAULT_CHUNK_SIZE = 4


@dataclass(frozen=True)
class WorkerResult:
    """Counts for this invocation; previous score records remain unchanged."""

    generated_count: int
    resumed_count: int
    chunk_count: int
    output_dir: Path


def _selected(source: Mapping[str, Any], condition: str) -> bool:
    return (source["hint_family"] == "none") == condition.endswith("unhinted")


def _condition_row(row: Mapping[str, Any], condition: str) -> dict[str, Any]:
    result = dict(row)
    if condition.endswith("unhinted"):
        result.pop("teacher_prompt_token_ids", None)
    return result


def _rows(
    sources: Mapping[str, Mapping[str, Any]],
    descriptors: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Iterator[dict[str, Any]]:
    for source in sources.values():
        if _selected(source, condition):
            yield _condition_row(source, condition)
    for descriptor in descriptors:
        source = sources[descriptor["decision_source_key"]]
        if _selected(source, condition):
            yield _condition_row(
                build_decision_row(source, descriptor, tokenizer=tokenizer), condition
            )


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    )


def _validate_worker_plan(
    output_dir: Path,
    episodes: Sequence[Mapping[str, Any]],
    descriptors: Sequence[Mapping[str, Any]],
    *,
    settings: Mapping[str, Any],
) -> None:
    semantic_episode_fields = (
        "key",
        "episode_key",
        "diagnostic_cell",
        "hint_family",
        "hint_strength",
        "prompt_token_ids",
        "teacher_prompt_token_ids",
        "completion_token_ids",
        "loss_mask",
        "token_turn_indices",
        "turn_action_spans",
        "student_sampling_logps",
        "turn_commands",
        "turn_classes",
        "turn_completions",
        "grader_directed_move_turn_indices",
        "grader_directed_move_coverage",
        "strategy",
    )
    semantic_episodes = [
        {field: row[field] for field in semantic_episode_fields if field in row} for row in episodes
    ]
    payload = {
        "settings": dict(settings),
        "episodes_sha256": hashlib.sha256(
            json.dumps(semantic_episodes, sort_keys=True).encode()
        ).hexdigest(),
        "descriptors_sha256": hashlib.sha256(
            json.dumps(descriptors, sort_keys=True).encode()
        ).hexdigest(),
        "episode_keys": [row["key"] for row in episodes],
        "descriptor_keys": [row["key"] for row in descriptors],
    }
    plan_path = output_dir / "worker-plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != payload:
            raise ValueError(
                "worker resume configuration, chunk assignment, or input descriptors changed"
            )
    else:
        if (output_dir / "round-0.jsonl").exists() or any(
            (output_dir / "parts").glob("part-*.jsonl")
        ):
            raise ValueError("worker resume requires a saved chunk assignment manifest")
        output_dir.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def run_worker(  # noqa: PLR0913 - scorer settings match the actual score_round boundary
    *,
    episodes_path: Path,
    descriptors_path: Path,
    output_dir: Path,
    condition: str,
    model: str,
    model_revision: str | None = None,
    tokenizer: PreTrainedTokenizerBase,
    adapter: Path | None = None,
    quantization: str | None = None,
    kv_cache_dtype: str | None = None,
    language_model_only: bool = True,
    max_num_seqs: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    scorer: Callable[..., Any] = score_round,
    runtime_factory: Callable[..., Any] = TeacherScoringRuntime,
    interrupt_after_first_record: Path | None = None,
) -> WorkerResult:
    """Validate all contexts, then score deterministic chunks using one lazy runtime."""
    if condition not in CONDITIONS or chunk_size < 1:
        raise ValueError("worker requires a known condition and positive chunk size")
    episodes = sorted(_read(episodes_path), key=lambda row: row["key"])
    sources = {row["key"]: row for row in episodes}
    descriptors = sorted(_read(descriptors_path), key=lambda row: row["key"])
    if len(sources) != len(episodes) or len({row["key"] for row in descriptors}) != len(
        descriptors
    ):
        raise ValueError("worker inputs contain duplicate keys")
    # The first pass holds one context at a time and sets capacity for the entire condition.
    longest = max(
        len(row.get("teacher_prompt_token_ids", row["prompt_token_ids"]))
        + len(row["completion_token_ids"])
        for row in _rows(sources, descriptors, condition=condition, tokenizer=tokenizer)
    )
    runtime = runtime_factory(
        max_model_len=math.ceil((longest + 1) / MODEL_LENGTH_MULTIPLE) * MODEL_LENGTH_MULTIPLE
    )
    resolved_revision = resolve_model_revision(
        Namespace(model=model, model_revision=model_revision)
    )
    _validate_worker_plan(
        output_dir,
        episodes,
        descriptors,
        settings={
            "condition": condition,
            "chunk_size": chunk_size,
            "max_model_len": runtime.max_model_len,
            "model_identity": model_identity(model),
            "model_revision": resolved_revision,
            "adapter_identity": model_identity(str(adapter)) if adapter is not None else None,
            "language_model_only": language_model_only,
            "quantization": quantization,
            "kv_cache_dtype": kv_cache_dtype,
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = output_dir / "parts"
    parts.mkdir(exist_ok=True)
    inputs = output_dir / ".chunk-inputs.jsonl"
    pending = _rows(sources, descriptors, condition=condition, tokenizer=tokenizer)
    generated_count = resumed_count = chunk_count = 0
    try:
        while chunk := list(islice(pending, chunk_size)):
            output = (
                output_dir / "round-0.jsonl"
                if condition == "teacher_hinted" and chunk_count == 0
                else parts / f"part-{chunk_count:06d}.jsonl"
            )
            _write(inputs, chunk)
            result = scorer(
                samples_path=inputs,
                output_path=output,
                model=model,
                model_revision=resolved_revision,
                adapter=adapter,
                quantization=quantization,
                language_model_only=language_model_only,
                kv_cache_dtype=kv_cache_dtype,
                max_num_seqs=max_num_seqs,
                runtime=runtime,
                compact_context=True,
                interrupt_after_first_record=interrupt_after_first_record
                if condition == "teacher_hinted"
                else None,
            )
            generated_count += result.generated_count
            resumed_count += result.resumed_count
            receipt = {
                "part": output.name,
                "keys": [row["key"] for row in chunk],
                "generated_count": result.generated_count,
                "resumed_count": result.resumed_count,
                "max_model_len": runtime.max_model_len,
            }
            receipt_path = parts / f"part-{chunk_count:06d}.receipt.json"
            if not receipt_path.exists():
                receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
            chunk_count += 1
    finally:
        inputs.unlink(missing_ok=True)
    logger.info(
        "condition=%s generated=%d resumed=%d chunks=%d",
        condition,
        generated_count,
        resumed_count,
        chunk_count,
    )
    return WorkerResult(generated_count, resumed_count, chunk_count, output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    """Run one scorer condition; all source and descriptor files are saved artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--quantization")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--interrupt-after-first-record", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    require_gitignored_output_dir(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model, local_files_only=True, trust_remote_code=False
    )
    run_worker(
        episodes_path=args.episodes,
        descriptors_path=args.descriptors,
        output_dir=args.output_dir,
        condition=args.condition,
        model=args.model,
        tokenizer=tokenizer,
        adapter=args.adapter,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        language_model_only=args.language_model_only,
        max_num_seqs=args.max_num_seqs,
        chunk_size=args.chunk_size,
        interrupt_after_first_record=args.interrupt_after_first_record,
    )


if __name__ == "__main__":
    main()
