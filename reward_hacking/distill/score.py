"""Score sampled token sequences with the 27B teacher through vLLM prompt logprobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from reward_hacking.distill.sample import (
    TorchRuntime,
    resolve_gpu_engine_settings,
    token_ids_sha256,
)
from reward_hacking.semantic_identity import model_identity, recorded_model_matches

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_TEACHER_MODEL_ID = "RedHatAI/Qwen3.8-27B-INT4"
TOP_K = 4
MAX_NUM_BATCHED_TOKENS = 1024
# Qwen3.5/3.8 padded vocabulary; vLLM materialises a full-vocab row per prefill position for prompt logprobs.
PADDED_VOCAB_SIZE = 248_320
# fp32 logits, fp32 log_softmax output and one fp32 softmax intermediate per vocab entry.
PROMPT_LOGPROB_BYTES_PER_VOCAB_ENTRY = 12
MODEL_LENGTH_MULTIPLE = 32768
SCORE_BATCH_SIZE = 4


def prompt_logprob_headroom_bytes() -> int:
    """VRAM vLLM's profiling pass never reserves: prompt-logprob tensors for one prefill chunk."""
    return MAX_NUM_BATCHED_TOKENS * PADDED_VOCAB_SIZE * PROMPT_LOGPROB_BYTES_PER_VOCAB_ENTRY


class TeacherLogprob(Protocol):
    """One vLLM prompt token logprob and its rank in the returned distribution."""

    logprob: float
    rank: int


class TeacherRequestOutput(Protocol):
    """vLLM prompt logprob rows for one teacher-forced input."""

    prompt_logprobs: Sequence[Mapping[int, TeacherLogprob] | None] | None
    prompt_token_ids: list[int] | None


class TeacherScoringEngine(Protocol):
    """vLLM generation API used for teacher-forced prompt scoring."""

    def generate(
        self,
        prompts: list[dict[str, list[int]]],
        *,
        sampling_params: object,
        use_tqdm: bool,
    ) -> Sequence[TeacherRequestOutput]:
        """Generate one token and return logprobs at prompt positions."""
        ...


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Counts teacher scores generated and reused from one score file."""

    output_path: Path
    generated_count: int
    resumed_count: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as records_file:
        for line_number, line in enumerate(records_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        output_file.flush()


def _token_ids(row: Mapping[str, Any], field: str, key: str) -> list[int]:
    value = row.get(field)
    if not isinstance(value, list) or any(
        type(token_id) is not int or token_id < 0 for token_id in value
    ):
        raise ValueError(f"sample {key!r} has invalid {field}")
    return value


def _task_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    env, problem_id = row.get("env"), row.get("problem_id")
    if not isinstance(env, str) or not env or not isinstance(problem_id, str) or not problem_id:
        raise ValueError("teacher-context rows require nonempty env and problem_id strings")
    return env, problem_id


def _teacher_prefixes(path: Path) -> dict[tuple[str, str], list[int]]:
    prefixes: dict[tuple[str, str], list[int]] = {}
    for row in _read_jsonl(path):
        identity = _task_identity(row)
        if identity in prefixes:
            raise ValueError(f"duplicate teacher prompt identity {identity!r}")
        prefix = _token_ids(row, "prompt_token_ids", str(identity))
        if not prefix:
            raise ValueError(f"teacher prompt {identity!r} must contain prompt token ids")
        prefixes[identity] = prefix
    return prefixes


def _completion_loss_mask(row: Mapping[str, Any], length: int) -> list[int]:
    if "loss_mask" not in row:
        return [1] * length
    mask = row["loss_mask"]
    if (
        not isinstance(mask, list)
        or len(mask) != length
        or any(type(value) is not int or value not in (0, 1) for value in mask)
    ):
        raise ValueError("loss_mask must contain one zero or one per completion token")
    if not any(mask):
        raise ValueError("loss_mask must select at least one completion token")
    return cast("list[int]", mask)


def _validate_sampling_logps(value: object, mask: Sequence[int]) -> None:
    if not isinstance(value, list) or len(value) != len(mask):
        raise ValueError("student_sampling_logps must align with completion token ids")
    _validate_logps(
        [logp for logp, active in zip(value, mask, strict=True) if active],
        sum(mask),
        "student_sampling_logps",
    )


def _validate_logps(value: object, length: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must align with completion token ids")
    if any(type(logp) not in (int, float) or not math.isfinite(logp) or logp > 0 for logp in value):
        raise ValueError(f"{label} must contain finite nonpositive logprobs")


def _validate_score_record(row: Mapping[str, Any], completion_length: int) -> None:
    _validate_logps(row.get("teacher_logps"), completion_length, "teacher_logps")
    top_ids, top_logps = row.get("top_ids"), row.get("top_logps")
    if not isinstance(top_ids, list) or not isinstance(top_logps, list):
        raise TypeError("teacher top-k scores must be lists")
    if len(top_ids) != completion_length or len(top_logps) != completion_length:
        raise ValueError("teacher top-k scores must align with completion token ids")
    mask = _completion_loss_mask(row, completion_length)
    for ids, logps, active in zip(top_ids, top_logps, mask, strict=True):
        if not active:
            if ids != [] or logps != []:
                raise ValueError("masked teacher top-k scores must be empty")
            continue
        if (
            not isinstance(ids, list)
            or len(ids) != TOP_K
            or any(type(token) is not int or token < 0 for token in ids)
        ):
            raise ValueError("teacher top_ids must contain nonnegative integer token ids")
        if len(set(ids)) != TOP_K:
            raise ValueError("teacher top_ids contains duplicate token ids")
        _validate_logps(logps, TOP_K, "top_logps")


def _aligned_teacher_logprobs(
    output: TeacherRequestOutput,
    prompt_length: int,
    completion_token_ids: Sequence[int],
    *,
    top_k: int = TOP_K,
    loss_mask: Sequence[int] | None = None,
) -> tuple[list[float], list[list[int]], list[list[float]]]:
    """Map prompt-logprob rows at full-sequence positions onto completion tokens."""
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None:
        raise ValueError("vLLM omitted prompt_logprobs")
    expected_length = prompt_length + len(completion_token_ids)
    if len(prompt_logprobs) != expected_length:
        raise ValueError(
            f"vLLM returned {len(prompt_logprobs)} prompt logprob rows, expected {expected_length}"
        )

    teacher_logps: list[float] = []
    top_ids: list[list[int]] = []
    top_logps: list[list[float]] = []
    mask = [1] * len(completion_token_ids) if loss_mask is None else loss_mask
    for completion_index, token_id in enumerate(completion_token_ids):
        if not mask[completion_index]:
            teacher_logps.append(0.0)
            top_ids.append([])
            top_logps.append([])
            continue
        position = prompt_length + completion_index
        token_logprobs = prompt_logprobs[position]
        if token_logprobs is None:
            raise ValueError(
                f"vLLM omitted prompt logprobs at completion position {completion_index}"
            )
        if token_id not in token_logprobs:
            raise ValueError(
                f"vLLM omitted actual completion token {token_id} at position {completion_index}"
            )
        teacher_logps.append(float(token_logprobs[token_id].logprob))
        ranked_items = sorted(token_logprobs.items(), key=lambda item: item[1].rank)
        if len(ranked_items) < top_k:
            raise ValueError(
                f"vLLM returned fewer than {top_k} top tokens at completion position {completion_index}"
            )
        selected_items = ranked_items[:top_k]
        top_ids.append([int(token) for token, _logprob in selected_items])
        top_logps.append([float(logprob.logprob) for _token, logprob in selected_items])
    _validate_logps(teacher_logps, len(completion_token_ids), "teacher_logps")
    for logps, active in zip(top_logps, mask, strict=True):
        if active:
            _validate_logps(logps, top_k, "top_logps")
    return teacher_logps, top_ids, top_logps


def score_round(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    samples_path: Path,
    output_path: Path,
    teacher_prompts_path: Path | None = None,
    model: str = DEFAULT_TEACHER_MODEL_ID,
    language_model_only: bool = False,
    kv_cache_dtype: str | None = None,
    max_num_seqs: int | None = None,
    llm_class: Callable[..., object] | None = None,
    sampling_params_class: Callable[..., object] | None = None,
    torch_module: TorchRuntime | None = None,
) -> ScoreResult:
    """Score every unscored sample and append one teacher record per completed key."""
    if max_num_seqs is not None and max_num_seqs < 1:
        raise ValueError("max_num_seqs must be positive")
    sample_rows = _read_jsonl(samples_path)
    teacher_prefixes = (
        _teacher_prefixes(teacher_prompts_path) if teacher_prompts_path is not None else None
    )
    samples_by_key: dict[str, dict[str, Any]] = {}
    scoring_prefixes: dict[str, list[int]] = {}
    for row in sample_rows:
        key = row.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{samples_path} contains a sample without a key")
        if key in samples_by_key:
            raise ValueError(f"{samples_path} contains duplicate sample key {key!r}")
        prompt_token_ids = _token_ids(row, "prompt_token_ids", key)
        completion_token_ids = _token_ids(row, "completion_token_ids", key)
        if not prompt_token_ids or not completion_token_ids:
            raise ValueError(f"sample {key!r} must contain prompt and completion token ids")
        expected_hash = token_ids_sha256(prompt_token_ids, completion_token_ids)
        row["token_ids_sha256"] = expected_hash
        loss_mask = _completion_loss_mask(row, len(completion_token_ids))
        samples_by_key[key] = row
        if "teacher_prompt_token_ids" in row:
            if teacher_prefixes is not None:
                raise ValueError("per-row teacher prompts cannot be combined with a teacher corpus")
            prefix = _token_ids(row, "teacher_prompt_token_ids", key)
            if not prefix:
                raise ValueError("teacher prompt must contain prompt token ids")
            scoring_prefixes[key] = prefix
            _validate_sampling_logps(row.get("student_sampling_logps"), loss_mask)
        elif teacher_prefixes is not None:
            identity = _task_identity(row)
            if identity not in teacher_prefixes:
                raise ValueError(f"missing teacher prompt identity {identity!r}")
            _validate_sampling_logps(row.get("student_sampling_logps"), loss_mask)
            scoring_prefixes[key] = teacher_prefixes[identity]
        else:
            scoring_prefixes[key] = prompt_token_ids
            if "loss_mask" in row:
                _validate_sampling_logps(row.get("student_sampling_logps"), loss_mask)
    if not samples_by_key:
        raise ValueError(f"no samples found in {samples_path}")

    longest_sequence_length = max(
        len(scoring_prefixes[key]) + len(row["completion_token_ids"])
        for key, row in samples_by_key.items()
    )
    max_model_len = (
        math.ceil((longest_sequence_length + 1) / MODEL_LENGTH_MULTIPLE) * MODEL_LENGTH_MULTIPLE
    )
    scoring_identity = hashlib.sha256(
        json.dumps(
            {
                "teacher_corpus_sha256": hashlib.sha256(
                    teacher_prompts_path.read_bytes()
                ).hexdigest()
                if teacher_prompts_path is not None
                else None,
                "model": model,
                "language_model_only": language_model_only,
                "kv_cache_dtype": kv_cache_dtype,
                "max_num_seqs": max_num_seqs,
                "max_model_len": max_model_len,
                "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                "tensor_parallel_size": 1,
                "prompt_logprobs": TOP_K,
                "max_tokens": 1,
                "temperature": 1.0,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    weight_identity = model_identity(model)
    scoring_settings = {
        "language_model_only": language_model_only,
        "kv_cache_dtype": kv_cache_dtype,
    }

    existing_rows = _read_jsonl(output_path) if output_path.exists() else []
    completed: set[str] = set()
    for row in existing_rows:
        key = str(row.get("key", ""))
        if key not in samples_by_key:
            raise ValueError(f"teacher output contains unexpected key {key!r}")
        if key in completed:
            raise ValueError(f"teacher output contains duplicate key {key!r}")
        if (
            row.get("completion_token_ids") != samples_by_key[key]["completion_token_ids"]
            or row.get("prompt_token_ids") != samples_by_key[key]["prompt_token_ids"]
        ):
            raise ValueError(f"teacher row {key!r} token ids differ from current sample")
        if not recorded_model_matches(row, model, identity=weight_identity):
            raise ValueError(f"teacher row {key!r} was scored by a different model")
        if "scoring_settings" in row and row["scoring_settings"] != scoring_settings:
            raise ValueError(f"teacher row {key!r} scoring settings differ from current engine")
        if teacher_prefixes is not None or "teacher_prompt_token_ids" in samples_by_key[key]:
            if row.get("teacher_prompt_token_ids") != scoring_prefixes[key]:
                raise ValueError(f"teacher row {key!r} teacher context tokens differ")
        elif "teacher_prompt_token_ids" in row:
            raise ValueError(f"teacher row {key!r} was scored with a separate teacher context")
        sample_mask = _completion_loss_mask(
            samples_by_key[key], len(samples_by_key[key]["completion_token_ids"])
        )
        if _completion_loss_mask(row, len(sample_mask)) != sample_mask:
            raise ValueError(f"teacher row {key!r} loss_mask differs from current sample")
        _validate_score_record(row, len(sample_mask))
        completed.add(key)

    pending_rows = [row for key, row in samples_by_key.items() if key not in completed]
    resumed_count = len(completed)
    if pending_rows:
        if llm_class is None or sampling_params_class is None:
            from vllm import LLM, SamplingParams  # noqa: PLC0415

            llm_class = llm_class or cast("Callable[..., object]", LLM)
            sampling_params_class = sampling_params_class or cast(
                "Callable[..., object]", SamplingParams
            )
        if torch_module is None:
            import torch  # noqa: PLC0415

            torch_module = cast("TorchRuntime", torch)

        sampler_utilization, device_name = resolve_gpu_engine_settings(torch_module)
        _, total_bytes = torch_module.cuda.mem_get_info(int(torch_module.cuda.current_device()))
        gpu_memory_utilization = sampler_utilization - prompt_logprob_headroom_bytes() / total_bytes
        if gpu_memory_utilization <= 0:
            raise RuntimeError(
                f"free VRAM leaves no room for prompt-logprob headroom: {gpu_memory_utilization=:.4f}"
            )
        prompt_token_sequences = [
            scoring_prefixes[str(row["key"])]
            + _token_ids(row, "completion_token_ids", str(row["key"]))
            for row in pending_rows
        ]
        engine_kwargs: dict[str, object] = {
            "model": model,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        }
        if language_model_only:
            engine_kwargs["language_model_only"] = True
        if kv_cache_dtype is not None:
            engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
        if max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = max_num_seqs
        engine = cast("TeacherScoringEngine", llm_class(**engine_kwargs))
        sampling_params = sampling_params_class(
            max_tokens=1, temperature=1.0, prompt_logprobs=TOP_K
        )
        logger.info(
            "scoring teacher=%s samples=%d max_model_len=%d gpu_memory_utilization=%.4f device=%s",
            model,
            len(pending_rows),
            max_model_len,
            gpu_memory_utilization,
            device_name,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for batch_start in range(0, len(pending_rows), SCORE_BATCH_SIZE):
            batch_rows = pending_rows[batch_start : batch_start + SCORE_BATCH_SIZE]
            batch_token_sequences = prompt_token_sequences[
                batch_start : batch_start + SCORE_BATCH_SIZE
            ]
            outputs = engine.generate(
                [{"prompt_token_ids": token_ids} for token_ids in batch_token_sequences],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            if len(outputs) != len(batch_rows):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(batch_rows)} samples"
                )

            for row, full_token_ids, output in zip(
                batch_rows, batch_token_sequences, outputs, strict=True
            ):
                key = str(row["key"])
                if output.prompt_token_ids != full_token_ids:
                    raise ValueError(
                        f"teacher row {key!r} returned token ids differ from requested teacher sequence"
                    )
                source_prompt_length = len(scoring_prefixes[key])
                completion_token_ids = [int(token_id) for token_id in row["completion_token_ids"]]
                teacher_logps, top_ids, top_logps = _aligned_teacher_logprobs(
                    output,
                    source_prompt_length,
                    completion_token_ids,
                    loss_mask=_completion_loss_mask(row, len(completion_token_ids)),
                )
                score_record: dict[str, object] = {
                    "key": key,
                    "model": model,
                    "model_identity": weight_identity,
                    "scoring_settings": scoring_settings,
                    "token_ids_sha256": row["token_ids_sha256"],
                    "prompt_token_ids": row["prompt_token_ids"],
                    "completion_token_ids": completion_token_ids,
                    "teacher_logps": teacher_logps,
                    "top_ids": top_ids,
                    "top_logps": top_logps,
                }
                if "loss_mask" in row:
                    score_record["loss_mask"] = row["loss_mask"]
                if teacher_prefixes is not None or "teacher_prompt_token_ids" in row:
                    score_record["teacher_prompt_token_ids"] = full_token_ids[:source_prompt_length]
                    score_record["teacher_token_ids_sha256"] = token_ids_sha256(
                        full_token_ids[:source_prompt_length], completion_token_ids
                    )
                    score_record["scoring_identity"] = scoring_identity
                _validate_score_record(score_record, len(completion_token_ids))
                _append_jsonl(output_path, score_record)

    generated_count = len(pending_rows)
    logger.info(
        "teacher scoring complete generated=%d resumed=%d output=%s",
        generated_count,
        resumed_count,
        output_path,
    )
    return ScoreResult(output_path, generated_count, resumed_count)


def _sample_token_gaps(
    key: str,
    sample: Mapping[str, Any],
    teacher: Mapping[str, Any],
) -> tuple[str, list[float]]:
    prompt_token_ids = _token_ids(sample, "prompt_token_ids", key)
    completion_token_ids = _token_ids(sample, "completion_token_ids", key)
    for field, token_ids in (
        ("prompt_token_ids", prompt_token_ids),
        ("completion_token_ids", completion_token_ids),
    ):
        if field in teacher and teacher[field] != token_ids:
            raise ValueError(f"KL teacher row {key!r} {field} differs from sample")
    student_logps = sample.get("student_sampling_logps")
    teacher_logps = teacher.get("teacher_logps")
    if not isinstance(student_logps, list) or not isinstance(teacher_logps, list):
        raise TypeError(f"KL row {key!r} has invalid logprob arrays")
    if len(student_logps) != len(teacher_logps):
        raise ValueError(f"KL row {key!r} has mismatched student and teacher logprob lengths")
    mask = _completion_loss_mask(sample, len(completion_token_ids))
    _validate_sampling_logps(student_logps, mask)
    _validate_logps(teacher_logps, len(completion_token_ids), "teacher_logps")
    gaps = [
        float(student_logp) - float(teacher_logp)
        for student_logp, teacher_logp, active in zip(
            student_logps, teacher_logps, mask, strict=True
        )
        if active
    ]
    return str(sample["env"]), gaps


def compute_kl_summary(samples_path: Path, teacher_path: Path) -> dict[str, dict[str, float | int]]:
    """Summarize mean student sampled-token logprob minus teacher logprob by environment."""
    samples = _read_jsonl(samples_path)
    teachers = _read_jsonl(teacher_path)
    if not samples or not teachers:
        raise ValueError("KL input files must contain scored samples")
    samples_by_key = {str(row["key"]): row for row in samples}
    teachers_by_key = {str(row["key"]): row for row in teachers}
    if len(samples_by_key) != len(samples) or len(teachers_by_key) != len(teachers):
        raise ValueError("KL input files contain duplicate keys")
    if samples_by_key.keys() != teachers_by_key.keys():
        raise ValueError("KL input files do not contain the same sample keys")

    gaps_by_env: dict[str, list[float]] = {}
    samples_per_env: dict[str, int] = {}
    for key, sample in samples_by_key.items():
        env, gaps = _sample_token_gaps(key, sample, teachers_by_key[key])
        gaps_by_env.setdefault(env, []).extend(gaps)
        samples_per_env[env] = samples_per_env.get(env, 0) + 1

    summary: dict[str, dict[str, float | int]] = {}
    for env, gaps in gaps_by_env.items():
        if not gaps:
            raise ValueError(f"environment {env!r} contains no scored completion tokens")
        summary[env] = {
            "mean_student_sampling_logp_minus_teacher_logp": math.fsum(gaps) / len(gaps),
            "token_count": len(gaps),
            "sample_count": samples_per_env[env],
        }
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Run the teacher scorer or print a sampled-token KL summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="teacher JSONL path, or existing teacher file with --kl-eval",
    )
    parser.add_argument("--model", default=DEFAULT_TEACHER_MODEL_ID)
    parser.add_argument(
        "--teacher-prompts", type=Path, help="JSONL teacher prefixes indexed by env and problem_id"
    )
    parser.add_argument("--kl-eval", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--max-num-seqs", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.kl_eval:
        summary = compute_kl_summary(args.samples, args.output)
        logger.info("sampled-token KL summary: %s", json.dumps(summary, sort_keys=True))
        return
    score_round(
        samples_path=args.samples,
        output_path=args.output,
        teacher_prompts_path=args.teacher_prompts,
        model=args.model,
        language_model_only=args.language_model_only,
        kv_cache_dtype=args.kv_cache_dtype,
        max_num_seqs=args.max_num_seqs,
    )


if __name__ == "__main__":
    main()
