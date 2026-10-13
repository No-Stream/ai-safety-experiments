"""Sample resumable distillation traces with vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from grpo.gpu_memory import cap_gpu_memory_utilization
from reward_hacking.jsonl_io import append_jsonl_unicode, read_jsonl_objects
from reward_hacking.lora_serving import prepare_vllm_serving_adapter
from reward_hacking.semantic_identity import model_identity, recorded_model_matches

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger: logging.Logger = logging.getLogger(__name__)

VLLM_FREE_MEMORY_MARGIN_FRACTION = 0.02
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
SAMPLE_BATCH_SIZE = 16
PROBLEM_PAIR_LENGTH = 2


class CudaRuntime(Protocol):
    """CUDA calls needed to size the vLLM engine."""

    def is_available(self) -> bool:
        """Return whether CUDA is available."""
        ...

    def current_device(self) -> int:
        """Return the current CUDA device index."""
        ...

    def mem_get_info(self, device: int) -> tuple[int, int]:
        """Return free and total bytes for a CUDA device."""
        ...

    def get_device_name(self, device: int) -> str:
        """Return a human-readable CUDA device name."""
        ...


class TorchRuntime(Protocol):
    """Minimal Torch interface used by this GPU entry point."""

    @property
    def cuda(self) -> CudaRuntime:
        """Expose CUDA operations on the Torch module."""
        ...


class TokenLogprob(Protocol):
    """One vLLM token logprob and its rank in the returned distribution."""

    logprob: float


class SampleGeneration(Protocol):
    """One vLLM choice, including the sampled token logprobs."""

    token_ids: Sequence[int]
    logprobs: Sequence[Mapping[int, TokenLogprob] | None] | None
    finish_reason: str | None
    text: str


class SampleRequestOutput(Protocol):
    """vLLM's request output containing its generated choices."""

    outputs: Sequence[SampleGeneration]


class SamplingEngine(Protocol):
    """vLLM generation API used by the sampler."""

    def generate(
        self,
        prompts: list[dict[str, list[int]]],
        *,
        sampling_params: object | Sequence[object],
        lora_request: object | None,
        use_tqdm: bool,
    ) -> Sequence[SampleRequestOutput]:
        """Generate one completion for each prompt."""
        ...


@dataclass(frozen=True, slots=True)
class StageResult:
    """Counts generated and resumed records for one sample file."""

    output_path: Path
    generated_count: int
    resumed_count: int


def token_ids_sha256(prompt_token_ids: Sequence[int], completion_token_ids: Sequence[int]) -> str:
    """Hash the exact prompt and completion ids used by a sample."""
    payload = json.dumps(
        {
            "prompt_token_ids": list(prompt_token_ids),
            "completion_token_ids": list(completion_token_ids),
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_seed(key: str, prompt_token_ids: Sequence[int], model: str, adapter: str | None) -> int:
    """Derive a stable vLLM seed from the sample identity and prompt content."""
    payload = json.dumps(
        {
            "key": key,
            "prompt_token_ids": list(prompt_token_ids),
            "model": model,
            "adapter": adapter,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _problem_pair(value: object, *, description: str) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != PROBLEM_PAIR_LENGTH:
        raise TypeError(f"{description} must be [env, problem_id], got {value!r}")
    return str(value[0]), str(value[1])


def _load_split_pairs(split_path: Path, split_side: str) -> set[tuple[str, str]]:
    if split_side not in {"train", "held_out"}:
        raise ValueError(f"unsupported prompt split {split_side!r}")
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(split_data, dict) or split_side not in split_data:
        raise ValueError(f"{split_path} has no {split_side!r} split")
    split_values = split_data[split_side]
    if not isinstance(split_values, list):
        raise TypeError(f"{split_path}:{split_side} must be a list")
    selected_pairs = {
        _problem_pair(value, description=f"{split_path}:{split_side} entry")
        for value in split_values
    }
    if len(selected_pairs) != len(split_values):
        raise ValueError(f"{split_path}:{split_side} contains duplicate prompt identities")
    return selected_pairs


def _index_prompt_rows(prompts_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    all_records = read_jsonl_objects(prompts_path)
    rows_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, record in enumerate(all_records, start=1):
        for field in ("env", "problem_id", "prompt_text", "prompt_token_ids"):
            if field not in record:
                raise ValueError(f"{prompts_path}:{line_number} is missing {field!r}")
        pair = (str(record["env"]), str(record["problem_id"]))
        token_ids = record["prompt_token_ids"]
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(type(token_id) is not int for token_id in token_ids)
        ):
            raise ValueError(f"{prompts_path}:{line_number} has invalid prompt_token_ids")
        if pair in rows_by_pair:
            raise ValueError(f"{prompts_path} contains duplicate prompt identity {pair!r}")
        rows_by_pair[pair] = record
    return rows_by_pair


def load_prompts_for_split(
    prompts_path: Path,
    split_path: Path,
    split_side: str,
) -> list[dict[str, Any]]:
    """Select prompt rows by the saved train or held-out split."""
    selected_pairs = _load_split_pairs(split_path, split_side)
    rows_by_pair = _index_prompt_rows(prompts_path)
    missing_pairs = sorted(selected_pairs - rows_by_pair.keys())
    if missing_pairs:
        raise ValueError(
            f"{split_path}:{split_side} references prompts missing from {prompts_path}: {missing_pairs!r}"
        )
    selected_rows = [rows_by_pair[pair] for pair in sorted(selected_pairs)]
    if not selected_rows:
        raise ValueError(f"{split_path}:{split_side} contains no prompts")
    return selected_rows


def resolve_gpu_engine_settings(torch_module: TorchRuntime) -> tuple[float, str]:
    """Derive vLLM utilization from free CUDA memory and log the device."""
    if not torch_module.cuda.is_available():
        raise RuntimeError("vLLM sampling requires a CUDA device")
    device_index = int(torch_module.cuda.current_device())
    utilization = cap_gpu_memory_utilization(
        torch_module, 1.0, device_index, margin_fraction=VLLM_FREE_MEMORY_MARGIN_FRACTION
    )
    if not 0 < utilization < 1:
        raise RuntimeError(f"free VRAM leaves no usable vLLM allocation: {utilization=:.4f}")
    device_name = str(torch_module.cuda.get_device_name(device_index))
    logger.info(
        "vLLM gpu_memory_utilization=%.4f device=%s",
        utilization,
        device_name,
    )
    return utilization, device_name


def sample_round(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    prompts_path: Path,
    split_path: Path,
    split_side: str,
    output_path: Path,
    model: str,
    samples_per_prompt: int,
    max_tokens: int,
    batch_size: int = SAMPLE_BATCH_SIZE,
    adapter: Path | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    language_model_only: bool = False,
    kv_cache_dtype: str | None = None,
    max_num_seqs: int | None = None,
    llm_class: Callable[..., object] | None = None,
    sampling_params_class: Callable[..., object] | None = None,
    lora_request_class: Callable[..., object] | None = None,
    torch_module: TorchRuntime | None = None,
) -> StageResult:
    """Sample one output per selected prompt and sample index, appending each record as it completes."""
    if samples_per_prompt < 1 or max_tokens < 1 or batch_size < 1:
        raise ValueError("samples_per_prompt, max_tokens, and batch_size must be positive")
    if adapter is not None and not adapter.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter}")

    rows = load_prompts_for_split(prompts_path, split_path, split_side)
    weight_identity = model_identity(model)
    adapter_identity = model_identity(str(adapter)) if adapter is not None else None
    sampling_identity: dict[str, object] = {
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "logprobs": 0,
        "batch_size": batch_size,
    }
    existing_rows = read_jsonl_objects(output_path) if output_path.exists() else []
    legacy_rows = [row for row in existing_rows if "model_identity" not in row]
    seed_model = (
        str(legacy_rows[0]["model"]) if legacy_rows else json.dumps(weight_identity, sort_keys=True)
    )
    seed_adapter = (
        legacy_rows[0].get("adapter")
        if legacy_rows
        else json.dumps(adapter_identity, sort_keys=True)
        if adapter_identity is not None
        else None
    )
    template_kwargs = dict(chat_template_kwargs or {})
    pending: list[tuple[dict[str, Any], int, str, int]] = []
    expected: dict[str, tuple[dict[str, Any], int]] = {}
    for row in rows:
        env = str(row["env"])
        problem_id = str(row["problem_id"])
        prompt_token_ids = [int(token_id) for token_id in row["prompt_token_ids"]]
        for sample_index in range(samples_per_prompt):
            key = f"{env}/{problem_id}/{sample_index}"
            if key in expected:
                raise ValueError(f"duplicate sample key {key!r}")
            expected[key] = (row, sample_index)
            seed = content_seed(key, prompt_token_ids, seed_model, seed_adapter)
            pending.append((row, sample_index, key, seed))

    completed: set[str] = set()
    for row in existing_rows:
        key = str(row.get("key", ""))
        if key not in expected:
            raise ValueError(f"sample output contains an unexpected key {key!r}")
        if key in completed:
            raise ValueError(f"sample output contains duplicate key {key!r}")
        prompt_record, _sample_index = expected[key]
        prompt_token_ids = [int(token_id) for token_id in prompt_record["prompt_token_ids"]]
        if (
            not recorded_model_matches(row, model, identity=weight_identity)
            or not recorded_model_matches(
                row,
                str(adapter) if adapter is not None else None,
                field="adapter",
                identity=adapter_identity,
            )
            or row.get("prompt_token_ids") != prompt_token_ids
            or {key: value for key, value in row.get("sampling", {}).items() if key != "batch_size"}
            != {key: value for key, value in sampling_identity.items() if key != "batch_size"}
            or row.get("chat_template_kwargs", {}) != template_kwargs
            or row.get("language_model_only", False) != language_model_only
            or row.get("kv_cache_dtype") != kv_cache_dtype
        ):
            raise ValueError(
                f"existing sample {key!r} does not match sampling identity or prompt ids"
            )
        completed.add(key)
    resumed_count = len(completed)
    pending = [request for request in pending if request[2] not in completed]

    if pending:
        if llm_class is None or sampling_params_class is None:
            from vllm import LLM, SamplingParams  # noqa: PLC0415

            llm_class = llm_class or cast("Callable[..., object]", LLM)
            sampling_params_class = sampling_params_class or cast(
                "Callable[..., object]", SamplingParams
            )
        if torch_module is None:
            import torch  # noqa: PLC0415

            torch_module = cast("TorchRuntime", torch)
        if adapter is not None and lora_request_class is None:
            from vllm.lora.request import LoRARequest  # noqa: PLC0415

            lora_request_class = cast("Callable[..., object]", LoRARequest)

        gpu_memory_utilization, device_name = resolve_gpu_engine_settings(torch_module)
        engine_kwargs: dict[str, Any] = {
            "model": model,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max(len(row["prompt_token_ids"]) for row in rows) + max_tokens,
        }
        if adapter is not None:
            engine_kwargs.update(enable_lora=True, max_lora_rank=64, max_loras=1, max_cpu_loras=1)
        if language_model_only:
            engine_kwargs["language_model_only"] = True
        if kv_cache_dtype is not None:
            engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
        if max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = max_num_seqs
        engine = cast("SamplingEngine", llm_class(**engine_kwargs))
        lora_request = None
        if adapter is not None:
            if lora_request_class is None:
                raise RuntimeError("LoRA adapter sampling requires a vLLM LoRARequest class")
            served_adapter = prepare_vllm_serving_adapter(
                adapter, output_path.parent / f"{output_path.stem}.vllm-adapter"
            )
            lora_request = lora_request_class(model, 1, str(served_adapter))

        logger.info(
            "sampling model=%s adapter=%s split=%s pending=%d batch_size=%d device=%s chat_template_kwargs=%s",
            model,
            adapter,
            split_side,
            len(pending),
            batch_size,
            device_name,
            template_kwargs,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            # Stored ids already include the verified template rendering, so do not render them again.
            vllm_prompts = [
                {"prompt_token_ids": [int(token_id) for token_id in row["prompt_token_ids"]]}
                for row, _sample_index, _key, _seed in batch
            ]
            sampling_params = [
                sampling_params_class(
                    max_tokens=max_tokens,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    repetition_penalty=1.0,
                    presence_penalty=0.0,
                    logprobs=0,
                    seed=seed,
                )
                for _row, _sample_index, _key, seed in batch
            ]
            outputs = engine.generate(
                vllm_prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
                use_tqdm=False,
            )
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(batch)} requests"
                )

            for (prompt_row, sample_index, key, seed), output in zip(batch, outputs, strict=True):
                if len(output.outputs) != 1:
                    raise RuntimeError(f"vLLM returned {len(output.outputs)} choices for {key}")
                generation = output.outputs[0]
                completion_token_ids = [int(token_id) for token_id in generation.token_ids]
                if not completion_token_ids or generation.logprobs is None:
                    raise RuntimeError(f"vLLM omitted completion token ids or logprobs for {key}")
                if len(generation.logprobs) != len(completion_token_ids):
                    raise RuntimeError(f"vLLM returned misaligned completion logprobs for {key}")
                student_sampling_logps: list[float] = []
                for token_id, token_logprobs in zip(
                    completion_token_ids, generation.logprobs, strict=True
                ):
                    if token_logprobs is None or token_id not in token_logprobs:
                        raise RuntimeError(
                            f"vLLM omitted sampled token {token_id} logprob for {key}"
                        )
                    student_sampling_logps.append(float(token_logprobs[token_id].logprob))
                prompt_token_ids = [int(token_id) for token_id in prompt_row["prompt_token_ids"]]
                sample_record: dict[str, object] = {
                    "key": key,
                    "env": str(prompt_row["env"]),
                    "problem_id": str(prompt_row["problem_id"]),
                    "sample_index": sample_index,
                    "model": model,
                    "adapter": str(adapter) if adapter is not None else None,
                    "prompt_token_ids": prompt_token_ids,
                    "completion_token_ids": completion_token_ids,
                    "student_sampling_logps": student_sampling_logps,
                    "finish_reason": generation.finish_reason,
                    "completion_text": generation.text,
                    "token_ids_sha256": token_ids_sha256(prompt_token_ids, completion_token_ids),
                    "seed": seed,
                    "sampling": sampling_identity,
                    "model_identity": weight_identity,
                    "adapter_identity": adapter_identity,
                    "chat_template_kwargs": template_kwargs,
                    "language_model_only": language_model_only,
                    "kv_cache_dtype": kv_cache_dtype,
                }
                append_jsonl_unicode(output_path, sample_record)

    generated_count = len(pending)
    logger.info(
        "sampling complete generated=%d resumed=%d output=%s",
        generated_count,
        resumed_count,
        output_path,
    )
    return StageResult(output_path, generated_count, resumed_count)


def _parse_chat_template_kwargs(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("chat-template-kwargs must be a JSON object")
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    """Run the sampler CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "held_out"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=SAMPLE_BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument(
        "--chat-template-kwargs",
        type=_parse_chat_template_kwargs,
        default={},
        help="record the upstream prompt rendering settings; prompt_token_ids remain authoritative",
    )
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--max-num-seqs", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    sample_round(
        prompts_path=args.prompts,
        split_path=args.split_json,
        split_side=args.split,
        output_path=args.output,
        model=args.model,
        adapter=args.adapter,
        samples_per_prompt=args.samples_per_prompt,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        chat_template_kwargs=args.chat_template_kwargs,
        language_model_only=args.language_model_only,
        kv_cache_dtype=args.kv_cache_dtype,
        max_num_seqs=args.max_num_seqs,
    )


if __name__ == "__main__":
    main()
