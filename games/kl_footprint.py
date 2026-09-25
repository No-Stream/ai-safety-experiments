"""Inference-only full-vocabulary KL footprint map."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import AutoTokenizer

from games.argument_prior_map import (
    _assert_adapter_shapes_match_model,  # pyright: ignore[reportPrivateUsage]
    _assert_vllm_adapter_preflight,  # pyright: ignore[reportPrivateUsage]
    _build_smoke_adapters,  # pyright: ignore[reportPrivateUsage]
    _raw_hf_continue,  # pyright: ignore[reportPrivateUsage]
)
from games.inference_utils import (
    adapter_digest,
    append_jsonl,
    content_key,
    derive_batch_size,
    device_memory_report,
    ensure_run_identity,
    load_completed_keys,
    resolve_device,
)
from games.interp_mediation import render_context
from games.lora import assert_one_adapter_config, attach_adapter, load_adapter_base
from games.parsing import THINK_CLOSE
from games.prompt_variants import PROMPT_VARIANT_THINK_BRIEFLY_V1, apply_prompt_variant
from games.teacher_forcing import (
    TeacherForcedKLResult,
    TeacherForcingTokenizer,
    build_batch,
    teacher_force_kl,
)
from games.vllm_teardown import release_engine, vram_used_mib
from reward_hacking.model_backend import SamplingConfig, VLLMBackend

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen3.5-9B"
SMOKE_MODEL = "Qwen/Qwen3.5-0.8B"
SELF_ADAPTER = Path("artifacts/games/ninep-transfer-2026-09-23/adapters/twin-pd-self-step70")
GROUP_ADAPTER = Path("artifacts/games/ninep-transfer-2026-09-23/adapters/twin-pd-group-step70")
DEFAULT_STIMULUS = Path("docs/scratch/rl-generalization-inference/kl-footprint-prompts.json")
DEFAULT_OUTPUT = Path("artifacts/games/kl-footprint/first-cut")
MODEL_CONDITIONS: tuple[str, ...] = (
    "self",
    "group",
    "self-placebo",
    "group-placebo",
)
DEFAULT_POSITION_CHUNK_SIZE = 384
_KL_WORKSPACE_MULTIPLIER = 8


def _json_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prompts(path: Path, *, smoke: bool) -> tuple[dict[str, str], ...]:
    """Load the private prompt set, preserving only the runtime stimulus data."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path} has no non-empty 'prompts' list")
    loaded = tuple(
        {
            "id": str(item["id"]),
            "category": str(item["category"]),
            "text": str(item["text"]),
        }
        for item in prompts
    )
    if len({item["id"] for item in loaded}) != len(loaded):
        raise ValueError(f"{path} contains duplicate prompt ids")
    if smoke:
        return loaded[:2]
    return loaded


def _hidden_size(model: PreTrainedModel) -> int:
    config = cast("Any", model.config)
    text_config = getattr(config, "text_config", config)
    return int(text_config.hidden_size)


def _vocabulary_size(model: PreTrainedModel) -> int:
    """Return the output vocabulary size used by the model's LM head."""
    config = cast("Any", model.config)
    text_config = getattr(config, "text_config", config)
    vocabulary_size = int(text_config.vocab_size)
    if vocabulary_size < 1:
        raise ValueError(f"model vocabulary size must be positive, got {vocabulary_size}")
    return vocabulary_size


def _logit_dtype(model: PreTrainedModel) -> torch.dtype:
    """Return the LM-head dtype used for the bounded device logit windows."""
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("model has no output embeddings from which to derive logit dtype")
    return cast("torch.Tensor", output_embeddings.weight).dtype


def derive_kl_row_group_size(
    device: torch.device,
    *,
    vocabulary_size: int,
    position_chunk_size: int,
    logit_dtype: torch.dtype,
    cap: int,
) -> int:
    """Derive a bounded KL row group from live free VRAM and chunk logit cost.

    The raw per-row window is ``2 * vocabulary * positions * dtype_bytes`` because base and
    adapter logits coexist while the device-side reduction runs.  The multiplier reserves space
    for float32 softmax/KL workspaces and the transformer activations; it is deliberately applied
    to the live cost rather than replacing it with a fixed card-size budget.
    """
    if vocabulary_size < 1 or position_chunk_size < 1 or cap < 1:
        raise ValueError("vocabulary size, position chunk size, and cap must be positive")
    if device.type != "cuda":
        return 1
    free_bytes, _ = torch.cuda.mem_get_info(device)
    bytes_per_logit = torch.empty((), dtype=logit_dtype).element_size()
    raw_window_bytes = 2 * vocabulary_size * position_chunk_size * bytes_per_logit
    estimated_row_bytes = raw_window_bytes * _KL_WORKSPACE_MULTIPLIER
    derived = int(free_bytes // max(estimated_row_bytes, 1))
    return max(1, min(cap, derived))


@contextmanager
def matched_norm_random_lora(
    model: Any,  # noqa: ANN401 - PEFT model modules are third-party dynamic objects
    *,
    adapter_name: str = "default",
    seed: int = 20260924,
) -> Generator[None]:
    """Temporarily replace every LoRA delta with a per-module matched-norm random delta."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    originals: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for module in model.modules():
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            if lora_a is None or lora_b is None:
                continue
            if adapter_name not in lora_a or adapter_name not in lora_b:
                continue
            a_weight = lora_a[adapter_name].weight
            b_weight = lora_b[adapter_name].weight
            original_a = a_weight.detach().clone()
            original_b = b_weight.detach().clone()
            random_a = torch.randn(
                a_weight.shape, generator=generator, dtype=torch.float32, device="cpu"
            ).to(device=a_weight.device, dtype=a_weight.dtype)
            random_b = torch.randn(
                b_weight.shape, generator=generator, dtype=torch.float32, device="cpu"
            ).to(device=b_weight.device, dtype=b_weight.dtype)
            for random_factor, real_factor in ((random_a, a_weight), (random_b, b_weight)):
                random_norm = torch.linalg.vector_norm(random_factor.float())
                real_norm = torch.linalg.vector_norm(real_factor.float())
                if real_norm == 0:
                    random_factor.zero_()
                else:
                    random_factor.mul_((real_norm / random_norm).to(random_factor.dtype))
            originals.append((a_weight, b_weight, original_a, original_b))
            a_weight.copy_(random_a)
            b_weight.copy_(random_b)
    try:
        yield
    finally:
        with torch.no_grad():
            for a_weight, b_weight, original_a, original_b in originals:
                a_weight.copy_(original_a)
                b_weight.copy_(original_b)


def _top_predictions(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: torch.Tensor,
    logits: torch.Tensor,
) -> list[dict[str, Any]]:
    """Return decoded top predictions from one compact top-k row."""
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            "logit": float(value),
        }
        for value, token_id in zip(logits.tolist(), token_ids.tolist(), strict=True)
    ]


def _close_token_index(tokenizer: TeacherForcingTokenizer, token_ids: torch.Tensor) -> int | None:
    """Find the first response-token position at which the thinking close sequence ends."""
    close_ids = tokenizer.encode(THINK_CLOSE, add_special_tokens=False)
    target = [int(token_id) for token_id in close_ids]
    values = [int(token_id) for token_id in token_ids.tolist()]
    for start in range(len(values) - len(target) + 1):
        if values[start : start + len(target)] == target:
            return start + len(target)
    return None


def _position_payload(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: torch.Tensor,
    result: TeacherForcedKLResult,
    *,
    row_index: int,
    top_count: int = 10,
) -> list[dict[str, Any]]:
    """Serialize the largest KL positions with token and top-three predictions."""
    kl = result.kl[row_index]
    positions = torch.argsort(kl, descending=True)[:top_count].tolist()
    return [
        {
            "position": int(position),
            "token_id": int(token_ids[position]),
            "token": tokenizer.decode([int(token_ids[position])], skip_special_tokens=False),
            "kl": float(kl[position]),
            "base_top3": _top_predictions(
                tokenizer,
                result.base_top_token_ids[row_index][position],
                result.base_top_logits[row_index][position],
            ),
            "adapter_top3": _top_predictions(
                tokenizer,
                result.comparison_top_token_ids[row_index][position],
                result.comparison_top_logits[row_index][position],
            ),
        }
        for position in positions
    ]


def _score_record(
    tokenizer: PreTrainedTokenizerBase,
    result: TeacherForcedKLResult,
    *,
    model_condition: str,
    row_index: int = 0,
) -> dict[str, Any]:
    """Build the per-prompt KL readout from two teacher-forced results."""
    kl = result.kl[row_index]
    token_ids = result.continuation_ids[row_index]
    close_index = _close_token_index(cast("Any", tokenizer), token_ids)
    thinking_end = close_index if close_index is not None else len(token_ids)
    answer_start = close_index if close_index is not None else len(token_ids)
    return {
        "model_condition": model_condition,
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "thinking_kl_sum": float(kl[:thinking_end].sum()),
        "answer_kl_sum": float(kl[answer_start:].sum()),
        "thinking_closed": close_index is not None,
        "n_tokens": len(token_ids),
        "positions": _position_payload(tokenizer, token_ids, result, row_index=row_index),
    }


def _sample_base(
    args: argparse.Namespace,
    model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[dict[str, str]],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Sample each base prompt once, using vLLM in production and HF on CPU smoke."""
    rendered = [
        apply_prompt_variant(prompt["text"], PROMPT_VARIANT_THINK_BRIEFLY_V1) for prompt in prompts
    ]
    if device.type == "cpu":
        if model is None:
            raise RuntimeError("CPU sampling requires the loaded base model")
        contexts = [render_context(cast("Any", tokenizer), prompt) for prompt in rendered]
        return [
            {"id": prompt["id"], "category": prompt["category"], "text": text, "context": context}
            for prompt, context, text in zip(
                prompts,
                contexts,
                _raw_hf_continue(model, tokenizer, contexts, max_new_tokens=args.max_new_tokens),
                strict=True,
            )
        ]
    sampling = SamplingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=0,
    )
    baseline = vram_used_mib()
    backend = VLLMBackend(
        args.model,
        thinking=True,
        sampling=sampling,
        language_model_only=True,
        model_path=args.model_path,
    )
    try:
        completions = backend.generate_tokenized(list(rendered))
        return [
            {
                "id": prompt["id"],
                "category": prompt["category"],
                "text": completion.completion.text,
                "context": render_context(cast("Any", tokenizer), prompt_text),
                "context_token_ids": list(completion.prompt_token_ids),
                "response_token_ids": list(completion.response_token_ids),
            }
            for prompt, prompt_text, completion in zip(prompts, rendered, completions, strict=True)
        ]
    finally:
        release_engine(backend, baseline_mib=baseline)


def _score_conditions(  # noqa: PLR0913 - one argument per measurement axis
    args: argparse.Namespace,
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: Sequence[dict[str, Any]],
    *,
    output_path: Path,
    stimulus_digest: str,
    device: torch.device,
    batch_size: int,
    position_chunk_size: int,
    row_group_size: int,
    attached: Any | None,  # noqa: ANN401 - PEFT wrapper is a third-party dynamic object
) -> tuple[int, int]:
    """Teacher-force every sample under both adapters and matched-norm placebos."""
    completed = load_completed_keys(output_path)
    written = 0
    resumed = 0
    for condition in MODEL_CONDITIONS:
        pending: list[tuple[dict[str, Any], str]] = []
        for sample in samples:
            key = content_key(
                "kl",
                stimulus_digest,
                args.model,
                str(args.model_path or ""),
                sample["id"],
                sample["category"],
                sample["text"],
                sample["context"],
                PROMPT_VARIANT_THINK_BRIEFLY_V1,
                condition,
                args.adapter_digests.get("self" if condition.startswith("self") else "group", ""),
            )
            if key in completed:
                resumed += 1
            else:
                pending.append((sample, key))
        if not pending:
            continue
        if attached is None:
            active: Any = base_model
        else:
            adapter_path = args.self_adapter if condition.startswith("self") else args.group_adapter
            attached = attach_adapter(
                base_model,
                adapter_path,
                args.model,
                existing=attached.peft_model,
            )
            active = attached.peft_model
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            batch = build_batch(
                cast("Any", tokenizer),
                [sample["context"] for sample, _ in chunk],
                [sample["text"] for sample, _ in chunk],
                context_token_ids=(
                    [sample["context_token_ids"] for sample, _ in chunk]
                    if all("context_token_ids" in sample for sample, _ in chunk)
                    else None
                ),
                continuation_token_ids=(
                    [sample["response_token_ids"] for sample, _ in chunk]
                    if all("response_token_ids" in sample for sample, _ in chunk)
                    else None
                ),
            )
            if attached is None:
                kl_result = teacher_force_kl(
                    base_model,
                    base_model,
                    batch,
                    position_chunk_size=position_chunk_size,
                    row_group_size=row_group_size,
                )
            elif condition.endswith("placebo"):
                with matched_norm_random_lora(active):
                    kl_result = teacher_force_kl(
                        active,
                        active,
                        batch,
                        position_chunk_size=position_chunk_size,
                        row_group_size=row_group_size,
                        reference_context=active.disable_adapter,
                    )
            else:
                kl_result = teacher_force_kl(
                    active,
                    active,
                    batch,
                    position_chunk_size=position_chunk_size,
                    row_group_size=row_group_size,
                    reference_context=active.disable_adapter,
                )
            for row_index, (sample, key) in enumerate(chunk):
                payload = _score_record(
                    tokenizer,
                    kl_result,
                    model_condition=condition,
                    row_index=row_index,
                )
                payload.update(
                    {
                        "key": key,
                        "record": "prompt_kl",
                        "prompt_id": sample["id"],
                        "category": sample["category"],
                        "sample_text": sample["text"],
                        "adapter_available": attached is not None,
                        "device": str(device),
                        "batch_size": batch_size,
                        "position_chunk_size": position_chunk_size,
                        "row_group_size": row_group_size,
                    }
                )
                append_jsonl(output_path, payload)
                completed.add(key)
                written += 1
    return written, resumed


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Rank prompts and aggregate category means against the placebo."""
    ranked = sorted(
        records,
        key=lambda row: -float(row["mean_kl"]),
    )
    category: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        condition = str(row["model_condition"])
        category[(condition, str(row["category"]))].append(float(row["mean_kl"]))
    category_means = {
        f"{condition}|{category_name}": sum(values) / len(values)
        for (condition, category_name), values in sorted(category.items())
    }
    placebo_by_category = {
        category_name: sum(values) / len(values)
        for (condition, category_name), values in category.items()
        if condition.endswith("placebo")
    }
    category_contrasts: dict[str, float] = {}
    for (condition, category_name), values in sorted(category.items()):
        if condition.endswith("placebo") or category_name not in placebo_by_category:
            continue
        category_contrasts[f"{condition}|{category_name}"] = (
            sum(values) / len(values) - placebo_by_category[category_name]
        )
    top_positions = [
        {
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "model_condition": row["model_condition"],
            "position": position["position"],
            "token": position["token"],
            "kl": position["kl"],
        }
        for row in ranked
        for position in row["positions"]
    ]
    top_positions.sort(key=lambda row: -float(row["kl"]))
    return {
        "ranked_prompts": [
            {
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "model_condition": row["model_condition"],
                "mean_kl": row["mean_kl"],
            }
            for row in ranked
        ],
        "category_means": category_means,
        "category_minus_placebo": category_contrasts,
        "top_20_positions": top_positions[:20],
    }


def _prepare_adapters(args: argparse.Namespace, base_model: PreTrainedModel) -> PreTrainedModel:
    """Build smoke adapters or validate the production pair before HF scoring."""
    if args.smoke:
        base_model, smoke_adapters = _build_smoke_adapters(
            base_model,
            source_adapters=(args.self_adapter, args.group_adapter),
            output_root=args.output / "smoke-adapters",
            model_id=args.model,
        )
        args.self_adapter, args.group_adapter = smoke_adapters
        args.adapter_digests = {
            "self": adapter_digest(args.self_adapter),
            "group": adapter_digest(args.group_adapter),
        }
    _assert_adapter_shapes_match_model(args.self_adapter, base_model)
    _assert_adapter_shapes_match_model(args.group_adapter, base_model)
    assert_one_adapter_config((args.self_adapter, args.group_adapter))
    return base_model


def run_kl_footprint(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915 - driver phases
    """Run the KL footprint map and write resumable artifacts."""
    if args.smoke and args.model not in (None, SMOKE_MODEL):
        raise ValueError(f"--smoke requires --model {SMOKE_MODEL!r} when --model is supplied")
    args.model = SMOKE_MODEL if args.smoke else (args.model or BASE_MODEL)
    args.model_path = None if args.model_path is None else Path(args.model_path)
    args.self_adapter = Path(args.self_adapter)
    args.group_adapter = Path(args.group_adapter)
    args.stimulus = Path(args.stimulus)
    args.output = Path(args.output)
    if not args.smoke:
        _assert_vllm_adapter_preflight(args.self_adapter, args.model or BASE_MODEL)
        _assert_vllm_adapter_preflight(args.group_adapter, args.model or BASE_MODEL)
    args.adapter_digests = (
        {
            "self": adapter_digest(args.self_adapter),
            "group": adapter_digest(args.group_adapter),
        }
        if not args.smoke
        else {}
    )
    args.output.mkdir(parents=True, exist_ok=True)
    stimulus_digest = _json_digest(args.stimulus)
    ensure_run_identity(
        args.output / "run_identity.json",
        {
            "experiment": "kl-footprint",
            "model": args.model,
            "model_path": str(args.model_path or ""),
            "adapter_digests": args.adapter_digests,
            "stimulus_sha256": stimulus_digest,
        },
    )
    device = resolve_device(args.device)
    memory = device_memory_report(device)
    model_source = str(args.model_path) if args.model_path is not None else str(args.model)
    tokenizer = cast("PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(model_source))
    prompts = load_prompts(args.stimulus, smoke=args.smoke)
    base_model: PreTrainedModel | None = None
    if device.type == "cpu":
        base_model = load_adapter_base(model_source, dtype=torch.bfloat16, device=device)
    samples_path = args.output / "samples.jsonl"
    existing_samples: dict[str, dict[str, Any]] = {}
    if samples_path.exists():
        with samples_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    expected_key = content_key(
                        "sample-v2-token-ids",
                        stimulus_digest,
                        args.model,
                        str(args.model_path or ""),
                        str(row["id"]),
                    )
                    if row.get("key") != expected_key:
                        raise ValueError(
                            f"{samples_path} contains a sample from a different stimulus or model "
                            f"identity for {row.get('id')!r}"
                        )
                    existing_samples[str(row["id"])] = {
                        "id": str(row["id"]),
                        "category": str(row["category"]),
                        "text": str(row["text"]),
                        "context": str(row["context"]),
                        **(
                            {"context_token_ids": list(row["context_token_ids"])}
                            if "context_token_ids" in row
                            else {}
                        ),
                        **(
                            {"response_token_ids": list(row["response_token_ids"])}
                            if "response_token_ids" in row
                            else {}
                        ),
                    }
    missing_prompts = tuple(prompt for prompt in prompts if prompt["id"] not in existing_samples)
    new_samples = (
        _sample_base(args, base_model, tokenizer, missing_prompts, device=device)
        if missing_prompts
        else []
    )
    for sample in new_samples:
        existing_samples[sample["id"]] = sample
        append_jsonl(
            samples_path,
            {
                "key": content_key(
                    "sample-v2-token-ids",
                    stimulus_digest,
                    args.model,
                    str(args.model_path or ""),
                    sample["id"],
                ),
                "record": "base_sample",
                **sample,
            },
        )
    samples = tuple(existing_samples[prompt["id"]] for prompt in prompts)
    if base_model is None:
        base_model = load_adapter_base(model_source, dtype=torch.bfloat16, device=device)
    batch_size = derive_batch_size(
        device,
        max_sequence_length=8192,
        hidden_size=_hidden_size(base_model),
        cap=args.batch_cap,
    )
    position_chunk_size = int(args.position_chunk_size)
    row_group_size = derive_kl_row_group_size(
        device,
        vocabulary_size=_vocabulary_size(base_model),
        position_chunk_size=position_chunk_size,
        logit_dtype=_logit_dtype(base_model),
        cap=batch_size,
    )
    base_model = _prepare_adapters(args, base_model)
    attached = attach_adapter(base_model, args.self_adapter, args.model)
    records_path = args.output / "prompt_kl.jsonl"
    written, resumed = _score_conditions(
        args,
        base_model,
        tokenizer,
        samples,
        output_path=records_path,
        stimulus_digest=stimulus_digest,
        device=device,
        batch_size=batch_size,
        position_chunk_size=position_chunk_size,
        row_group_size=row_group_size,
        attached=attached,
    )
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    summary = summarize(records)
    summary["run"] = {
        "model": args.model,
        "model_path": str(args.model_path or ""),
        "stimulus_sha256": stimulus_digest,
        "device": memory,
        "derived_batch_size": batch_size,
        "position_chunk_size": position_chunk_size,
        "row_group_size": row_group_size,
        "kl_workspace_multiplier": _KL_WORKSPACE_MULTIPLIER,
        "written": written,
        "resumed": resumed,
        "samples": len(samples),
        "adapter_available": True,
        "smoke_base_reused": bool(args.smoke),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the KL footprint CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="local snapshot to load while retaining --model as the adapter/base identity",
    )
    parser.add_argument("--self-adapter", type=Path, default=SELF_ADAPTER)
    parser.add_argument("--group-adapter", type=Path, default=GROUP_ADAPTER)
    parser.add_argument("--stimulus", type=Path, default=DEFAULT_STIMULUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-cap", type=int, default=32)
    parser.add_argument("--position-chunk-size", type=int, default=DEFAULT_POSITION_CHUNK_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    """Run the KL footprint CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_kl_footprint(build_parser().parse_args())


if __name__ == "__main__":
    main()
