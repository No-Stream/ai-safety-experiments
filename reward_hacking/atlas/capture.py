"""Capture the four Atlas read positions from saved generation token ids.

The HF model is loaded only when ``capture_model`` is called. vLLM is never resident in this
stage, so each model checkpoint gets the card to itself. Per-sample progress is keyed and the final
activation array is one float32 NumPy shard per model (Gemma-3's residual stream has channels past
float16's 65504 limit in its upper layers, which stored as inf).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

import numpy as np
import torch

from reward_hacking.atlas.core import extract_code, position_masks
from reward_hacking.atlas.prompts import ATLAS_SYSTEM_PROMPT
from reward_hacking.atlas.sampling import (
    BASE_MODEL_ID,
    artifact_model_key,
    language_adapter_for,
    resolve_model_key,
    resume_records_atlas_compatible,
)
from reward_hacking.interp.generation_capture import (
    ResolvedSampler,
    capture_record_pooled,
    record_from_token_ids,
    vllm_resolved_sampler,
)
from reward_hacking.interp.jsonl_resume import normalize_identity
from reward_hacking.jsonl_io import (
    append_jsonl_ascii,
    read_jsonl_objects,
    write_json_atomic,
)
from reward_hacking.model_backend import SamplingConfig
from reward_hacking.semantic_identity import (
    model_weight_identity,
    normalize_legacy_model_reference,
    semantic_rows,
)

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)
ACTIVATION_DTYPE_NAME = "float32"
ACTIVATION_NUMPY_DTYPE = np.float32
POSITION_NAMES: tuple[str, ...] = ("last_prompt", "code_mean", "completion_last", "followup_last")
FOLLOWUP_QUESTION = "Did you respond truthfully?"
FOLLOWUP_ANSWER = "Yes"
REFERENCE_NPY_DIMS = 3
POSITIONAL_REFERENCE_NPY_DIMS = 4


def _decode(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> str:
    return cast("str", tokenizer.decode(token_ids, skip_special_tokens=True))


def _code_token_span(
    tokenizer: PreTrainedTokenizerBase,
    completion_token_ids: list[int],
    completion_text: str,
    extracted_code: str,
) -> tuple[int, int]:
    """Map the exact extracted code substring onto generated token boundaries, refusing drift."""
    code_start = completion_text.find(extracted_code)
    if code_start < 0:
        raise ValueError("extracted code is not a substring of the saved completion text")
    code_end = code_start + len(extracted_code)
    prefix_lengths = [0]
    prior_prefix = ""
    for end in range(1, len(completion_token_ids) + 1):
        prefix_text = _decode(tokenizer, completion_token_ids[:end])
        if not prefix_text.startswith(prior_prefix):
            raise ValueError(
                "decoding generated token prefixes is not monotonic for this completion"
            )
        prior_prefix = prefix_text
        prefix_lengths.append(len(prefix_text))
    if prior_prefix != completion_text:
        raise ValueError("decoding saved completion token ids no longer matches its saved text")
    starts = [index for index, length in enumerate(prefix_lengths) if length >= code_start]
    ends = [index for index, length in enumerate(prefix_lengths) if length >= code_end]
    if not starts or not ends:
        raise ValueError("could not map extracted code character span to token boundaries")
    token_start = min(starts)
    token_end = min(ends)
    if token_start >= token_end:
        raise ValueError(f"extracted code maps to an empty token span [{token_start}, {token_end})")
    return token_start, token_end


def _finite_float32_row(row_vectors: torch.Tensor) -> np.ndarray:
    """Convert one captured row for storage, refusing non-finite values rather than writing them."""
    if not bool(torch.isfinite(row_vectors).all()):
        raise ValueError("captured activations are not finite; refusing to write the row")
    return row_vectors.to(dtype=torch.float32).numpy()


def _followup_token_ids(  # noqa: PLR0913 - explicit token-boundary inputs are load-bearing
    tokenizer: PreTrainedTokenizerBase,
    *,
    task_text: str,
    test_list: list[str],
    assistant_text: str,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
) -> list[int]:
    """Return the follow-up tokens to append after the sampled completion, through the final ``Yes``.

    Re-rendering the whole conversation through the chat template does not reproduce the sampled
    ids (the template trims the assistant text, and a completion cut at the token cap has no
    end-of-turn), so the sampled prompt and completion ids are kept exactly and only the template's
    text after the assistant turn is tokenized and appended. A truncated completion is closed with
    an end-of-turn token first, as the template would close it.
    """
    if not test_list:
        raise ValueError("the Atlas follow-up requires at least one visible test")
    if not prompt_token_ids or not completion_token_ids:
        raise ValueError("the follow-up needs non-empty sampled prompt and completion ids")
    sentinel = "\x00ATLAS-ASSISTANT-TURN\x00"
    messages = [
        {"role": "system", "content": ATLAS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task_text}\n\nTest cases: {test_list[:1]!s}"},
        {"role": "assistant", "content": sentinel},
        {"role": "user", "content": FOLLOWUP_QUESTION},
        {"role": "assistant", "content": FOLLOWUP_ANSWER},
    ]
    rendered = cast(
        "str",
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
    )
    if rendered.count(sentinel) != 1 or not assistant_text:
        raise ValueError("the chat template did not render the assistant turn exactly once")
    tail = rendered[rendered.index(sentinel) + len(sentinel) :]
    end_of_turn = next(
        (
            marker
            for marker in ("<end_of_turn>", "<|im_end|>", getattr(tokenizer, "eos_token", None))
            if isinstance(marker, str) and marker and tail.startswith(marker)
        ),
        None,
    )
    if end_of_turn is None:
        raise ValueError(
            f"could not identify the assistant-turn closure in the template tail: {tail[:80]!r}"
        )
    answer_end = tail.rindex(FOLLOWUP_ANSWER) + len(FOLLOWUP_ANSWER)
    end_of_turn_ids = [
        int(token_id) for token_id in tokenizer.encode(end_of_turn, add_special_tokens=False)
    ]
    completion_closed = completion_token_ids[-len(end_of_turn_ids) :] == end_of_turn_ids
    tail_through_answer = tail[len(end_of_turn) if completion_closed else 0 : answer_end]
    followup_ids = [
        int(token_id)
        for token_id in tokenizer.encode(tail_through_answer, add_special_tokens=False)
    ]
    answer_ids = tokenizer.encode(FOLLOWUP_ANSWER, add_special_tokens=False)
    if followup_ids[-len(answer_ids) :] != answer_ids:
        raise ValueError(
            "the follow-up answer does not tokenize to the same ids at the end of the turn"
        )
    return followup_ids


def _sample_identity(sample: Mapping[str, Any]) -> str:
    return str(sample["sample_id"])


def _legacy_capture_identity_compatible(
    wanted: Mapping[str, Any],
    found: Mapping[str, Any],
    completed: Sequence[Mapping[str, Any]],
    current_sample_ids: set[str],
) -> bool:
    """Migrate byte-fingerprinted capture ledgers using their persisted record identities."""
    historical_controls = dict(found)
    current_controls = dict(wanted)
    historical_controls.pop("sample_file_sha256", None)
    for field in ("samples", "model_weights", "adapter_weights"):
        if field not in historical_controls:
            current_controls.pop(field, None)
    for controls in (historical_controls, current_controls):
        for field in ("base_model", "model_id", "adapter_id"):
            controls[field] = normalize_legacy_model_reference(controls.get(field))
    if normalize_identity(historical_controls) != normalize_identity(current_controls):
        return False
    return all(str(record.get("sample_id")) in current_sample_ids for record in completed)


def _reference_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one reference row per item and class without copying dataset text into code."""
    rows: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    for sample in samples:
        item_id = str(sample["item_id"])
        if item_id in seen_items:
            continue
        seen_items.add(item_id)
        for class_name, field_name in (("hardcode", "hardcode_code"), ("code", "reference_code")):
            completion = sample.get(field_name)
            if not isinstance(completion, str) or not completion.strip():
                continue
            rows.append(
                {
                    "reference_id": f"{item_id}:{class_name}",
                    "item_id": item_id,
                    "class_name": class_name,
                    "task_text": str(sample["task_text"]),
                    "test_list": [str(test) for test in sample["test_list"]],
                    "prompt": str(sample["prompt"]),
                    "prompt_token_ids": [int(token_id) for token_id in sample["prompt_token_ids"]],
                    "completion_text": completion,
                }
            )
    return rows


def _reference_capture_identity(
    *,
    samples: list[dict[str, Any]],
    base_model: str,
    model_key: str,
    model_id: str | None,
    adapter_id: str | None,
) -> dict[str, object]:
    references = _reference_rows(samples)
    identity: dict[str, object] = {
        "stage": "reference_capture",
        "base_model": normalize_legacy_model_reference(base_model),
        "base_model_weights": model_weight_identity(base_model),
        "reference_ids": [str(row["reference_id"]) for row in references],
        "reference_records": semantic_rows(
            references,
            required_fields=(
                "reference_id",
                "item_id",
                "class_name",
                "task_text",
                "test_list",
                "prompt",
                "prompt_token_ids",
                "completion_text",
            ),
        ),
        "dtype": ACTIVATION_DTYPE_NAME,
    }
    if model_key != "base":
        identity.update(
            {
                "model_key": model_key,
                "model_id": normalize_legacy_model_reference(model_id),
                "adapter_id": normalize_legacy_model_reference(adapter_id),
                "adapter_weights": model_weight_identity(adapter_id),
                "positions": ["code_mean", "followup_last"],
            }
        )
    return identity


def reference_artifact_paths(run_dir: Path, model_key: str) -> tuple[Path, Path, Path]:
    """Return activation, index, and resume paths, preserving the original base filenames."""
    if model_key == "base":
        return (
            run_dir / "reference-activations.npy",
            run_dir / "reference-activation-index.json",
            run_dir / "reference-capture-progress.jsonl",
        )
    output_key = artifact_model_key(model_key)
    return (
        run_dir / f"reference-activations.{output_key}.npy",
        run_dir / f"reference-activation-index.{output_key}.json",
        run_dir / f"reference-capture-progress.{output_key}.jsonl",
    )


def _capture_reference_probe(  # noqa: C901, PLR0912, PLR0913, PLR0915 - resumable reference transaction
    *,
    run_dir: Path,
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[dict[str, Any]],
    sampler: ResolvedSampler,
    base_model: str,
    model_key: str = "base",
    model_id: str | None = None,
    adapter_id: str | None = None,
) -> None:
    """Capture Atlas's hardcode-versus-code references on a base model or requested adapter."""
    references = _reference_rows(samples)
    if not references:
        raise ValueError("cannot capture the Atlas reference probe without reference rows")
    output_path, index_path, progress_path = reference_artifact_paths(run_dir, model_key)
    position_names = ("followup_last",) if model_key == "base" else ("code_mean", "followup_last")
    expected_dimensions = (
        REFERENCE_NPY_DIMS if model_key == "base" else POSITIONAL_REFERENCE_NPY_DIMS
    )
    metadata = [
        {
            "reference_id": str(reference["reference_id"]),
            "item_id": str(reference["item_id"]),
            "class_name": str(reference["class_name"]),
        }
        for reference in references
    ]
    identity = _reference_capture_identity(
        samples=samples,
        base_model=base_model,
        model_key=model_key,
        model_id=model_id,
        adapter_id=adapter_id,
    )

    def legacy_reference_identity_compatible(
        wanted: Mapping[str, Any], found: Mapping[str, Any], completed: Sequence[Mapping[str, Any]]
    ) -> bool:
        historical_controls = dict(found)
        current_controls = dict(wanted)
        for field in ("reference_records", "base_model_weights", "adapter_weights"):
            if field not in historical_controls:
                current_controls.pop(field, None)
        for controls in (historical_controls, current_controls):
            for field in ("base_model", "model_id", "adapter_id"):
                controls[field] = normalize_legacy_model_reference(controls.get(field))
        if normalize_identity(historical_controls) != normalize_identity(current_controls):
            return False
        expected_ids = set(cast("Sequence[str]", identity["reference_ids"]))
        return all(str(record.get("reference_id")) in expected_ids for record in completed)

    resumed, ledger = resume_records_atlas_compatible(
        progress_path,
        identity=identity,
        base_model=base_model,
        unit_of=lambda _: "base-reference" if model_key == "base" else model_key,
        record_key_of=lambda record: str(record["reference_id"]),
        legacy_identity_compatible=legacy_reference_identity_compatible,
    )
    completed = {str(record["reference_id"]) for record in resumed.kept_records}
    if output_path.exists():
        shard = np.load(output_path, mmap_mode="r+")
        if (
            shard.ndim != expected_dimensions
            or shard.shape[0] != len(references)
            or shard.dtype != ACTIVATION_NUMPY_DTYPE
            or (model_key != "base" and shard.shape[2] != len(position_names))
        ):
            raise ValueError(
                f"existing reference shard has an incompatible shape or dtype: {shard.shape}, {shard.dtype}"
            )
    else:
        if completed:
            raise ValueError("reference progress exists but the activation shard is missing")
        shard = None
    reference_rows = {
        str(reference["reference_id"]): index for index, reference in enumerate(references)
    }
    layer_ids: list[int] | None = None if shard is None else list(range(shard.shape[1]))
    for reference in references:
        reference_id = str(reference["reference_id"])
        if reference_id in completed:
            continue
        prompt_ids = list(reference["prompt_token_ids"])
        completion_ids = [
            int(token_id)
            for token_id in tokenizer.encode(
                str(reference["completion_text"]), add_special_tokens=False
            )
        ]
        if not completion_ids:
            raise ValueError(
                f"reference {reference['reference_id']} tokenized to no completion tokens"
            )
        followup_ids = _followup_token_ids(
            tokenizer,
            task_text=str(reference["task_text"]),
            test_list=list(reference["test_list"]),
            assistant_text=str(reference["completion_text"]),
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
        )
        masks = position_masks(
            prompt_length=len(prompt_ids),
            completion_length=len(completion_ids),
            code_token_start=0,
            code_token_end=len(completion_ids),
            followup_length=len(followup_ids),
        )
        followup_record = record_from_token_ids(
            tokenizer,
            str(reference["prompt"]),
            prompt_token_ids=[*prompt_ids, *completion_ids],
            response_token_ids=followup_ids,
            max_new_tokens=max(1, len(followup_ids)),
            sampler=sampler,
        )
        followup_vectors = capture_record_pooled(
            model,
            followup_record,
            {name: masks[name] for name in position_names},
            poolings=["mean"],
        )
        if layer_ids is None:
            layer_ids = sorted(followup_vectors["followup_last", "mean"])
        elif sorted(followup_vectors["followup_last", "mean"]) != layer_ids:
            raise RuntimeError("reference captures returned inconsistent layer sets")
        row_vectors = torch.stack(
            [
                torch.stack(
                    [followup_vectors[name, "mean"][layer] for name in position_names], dim=0
                )
                for layer in layer_ids
            ]
        )
        if model_key == "base":
            row_vectors = row_vectors[:, 0, :]
        if shard is None:
            shard = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=ACTIVATION_NUMPY_DTYPE,
                shape=(len(references), *row_vectors.shape),
            )
        elif tuple(shard.shape[1:]) != tuple(row_vectors.shape):
            raise ValueError(
                f"reference activation shape changed from {tuple(shard.shape[1:])} to {tuple(row_vectors.shape)}"
            )
        shard[reference_rows[reference_id]] = _finite_float32_row(row_vectors)
        shard.flush()
        append_jsonl_ascii(
            progress_path, {"reference_id": reference_id, "row": reference_rows[reference_id]}
        )
        ledger.mark_record_complete(reference_id)
        completed.add(reference_id)
    if layer_ids is None or shard is None:
        raise RuntimeError("reference capture produced no rows")
    write_json_atomic(
        index_path,
        {
            "model_key": model_key,
            "model_id": base_model if model_id is None else model_id,
            "adapter_id": adapter_id,
            "positions": list(position_names),
            "sample_ids": [row["reference_id"] for row in metadata],
            "rows": metadata,
            "shape": list(shard.shape),
            "dtype": ACTIVATION_DTYPE_NAME,
            "complete": len(completed) == len(references),
        },
    )


def capture_model(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    run_dir: Path,
    model_key: str,
    base_model: str = BASE_MODEL_ID,
    sample_path: Path,
    max_items: int | None = None,
    capture_reference: bool = False,
) -> Path:
    """Capture one model's samples and write a float32 shard plus an index manifest."""
    from peft import PeftModel  # noqa: PLC0415

    from reward_hacking.interp.directions import load_model_and_tokenizer  # noqa: PLC0415

    samples = read_jsonl_objects(sample_path)
    if max_items is not None:
        allowed_items = {str(sample["item_id"]) for sample in samples[:max_items]}
        samples = [sample for sample in samples if str(sample["item_id"]) in allowed_items]
    if not samples:
        raise ValueError(f"no sample records found in {sample_path}")
    sample_base_models = {str(sample.get("model_id", base_model)) for sample in samples}
    if sample_base_models != {base_model}:
        raise ValueError(
            f"samples in {sample_path} were generated for base models {sorted(sample_base_models)}, "
            f"not requested {base_model!r}"
        )
    samples.sort(key=lambda sample: (str(sample["item_id"]), int(sample["sample_index"])))
    model_id, adapter_id = resolve_model_key(model_key, base_model=base_model)
    output_key = artifact_model_key(model_key)
    sample_identity_rows = [
        {**sample, "model_id": normalize_legacy_model_reference(sample.get("model_id"))}
        for sample in samples
    ]
    output_path = run_dir / f"activations.{output_key}.npy"
    progress_path = run_dir / f"capture-progress.{output_key}.jsonl"
    index_path = run_dir / f"capture-index.{output_key}.json"
    identity = {
        "stage": "capture",
        "model_key": model_key,
        "base_model": normalize_legacy_model_reference(base_model),
        "model_id": normalize_legacy_model_reference(model_id),
        "adapter_id": normalize_legacy_model_reference(adapter_id),
        "model_weights": model_weight_identity(model_id),
        "adapter_weights": model_weight_identity(adapter_id),
        "samples": semantic_rows(
            sample_identity_rows,
            required_fields=(
                "sample_id",
                "item_id",
                "prompt",
                "prompt_token_ids",
                "completion_text",
                "completion_token_ids",
                "task_text",
                "test_list",
            ),
            optional_fields=("model_id", "test_setup_code", "reference_code", "hardcode_code"),
        ),
        "sample_ids": [_sample_identity(sample) for sample in samples],
        "positions": list(POSITION_NAMES),
        "dtype": ACTIVATION_DTYPE_NAME,
    }

    current_sample_ids = set(cast("Sequence[str]", identity["sample_ids"]))
    resumed, ledger = resume_records_atlas_compatible(
        progress_path,
        identity=identity,
        base_model=base_model,
        unit_of=lambda record: str(record["model_key"]),
        record_key_of=lambda record: str(record["sample_id"]),
        legacy_identity_compatible=lambda wanted, found, completed: (
            _legacy_capture_identity_compatible(wanted, found, completed, current_sample_ids)
        ),
    )
    completed = {str(record["sample_id"]) for record in resumed.kept_records}
    if output_path.exists():
        shard = np.load(output_path, mmap_mode="r+")
        if (
            shard.ndim != 4  # noqa: PLR2004
            or shard.shape[0] != len(samples)
            or shard.shape[2] != len(POSITION_NAMES)
            or shard.dtype != ACTIVATION_NUMPY_DTYPE
        ):
            raise ValueError(
                f"existing activation shard has an incompatible shape or dtype: {shard.shape}, {shard.dtype}"
            )
        if index_path.exists():
            saved_index = json.loads(index_path.read_text(encoding="utf-8"))
            if saved_index.get("sample_ids") != identity["sample_ids"]:
                raise ValueError(
                    "activation shard row order differs from the current sample records"
                )
        else:
            write_json_atomic(
                index_path,
                {
                    "model_key": model_key,
                    "model_id": model_id,
                    "adapter_id": adapter_id,
                    "sample_ids": identity["sample_ids"],
                    "positions": list(POSITION_NAMES),
                    "shape": list(shard.shape),
                    "dtype": str(shard.dtype),
                    "complete": False,
                },
            )
    else:
        if resumed.kept_records:
            raise ValueError(
                f"capture progress exists for {model_key}, but its activation shard {output_path} is missing"
            )
        shard = None

    _, reference_index_path, _ = reference_artifact_paths(run_dir, model_key)
    reference_requested = model_key == "base" or capture_reference
    reference_complete = reference_index_path.exists() and bool(
        json.loads(reference_index_path.read_text(encoding="utf-8")).get("complete")
    )
    if (
        len(completed) == len(samples)
        and shard is not None
        and (not reference_requested or reference_complete)
    ):
        return output_path

    loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_id, dtype=torch.bfloat16)
    tokenizer = cast("PreTrainedTokenizerBase", loaded_tokenizer)
    model = loaded_model
    language_adapter_path = language_adapter_for(model_key, run_dir, base_model=base_model)
    if language_adapter_path is not None:
        peft_model = PeftModel.from_pretrained(
            cast("Any", loaded_model), str(language_adapter_path)
        )
        peft_model.eval()
        model = cast("AutoModelForCausalLM", peft_model)

    row_by_id = {str(sample["sample_id"]): index for index, sample in enumerate(samples)}
    first_sampling = cast("Mapping[str, Any]", samples[0]["sampling"])
    reference_sampler: ResolvedSampler = vllm_resolved_sampler(
        SamplingConfig(
            max_new_tokens=int(first_sampling["max_tokens"]),
            temperature=float(first_sampling["temperature"]),
            top_p=float(first_sampling["top_p"]),
            top_k=int(first_sampling["top_k"]),
            min_p=float(first_sampling["min_p"]),
            repetition_penalty=float(first_sampling["repetition_penalty"]),
            presence_penalty=float(first_sampling["presence_penalty"]),
        )
    )
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if sample_id in completed:
            continue
        prompt_ids = [int(token_id) for token_id in sample["prompt_token_ids"]]
        completion_ids = [int(token_id) for token_id in sample["completion_token_ids"]]
        if not completion_ids:
            raise ValueError(f"sample {sample_id} has no completion tokens")
        completion_text = str(sample["completion_text"])
        code = extract_code(completion_text)
        code_start, code_end = _code_token_span(tokenizer, completion_ids, completion_text, code)
        followup_ids = _followup_token_ids(
            tokenizer,
            task_text=str(sample["task_text"]),
            test_list=[str(test) for test in sample["test_list"]],
            assistant_text=completion_text,
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
        )
        sample_sampling = cast("Mapping[str, Any]", sample["sampling"])
        sampling = SamplingConfig(
            max_new_tokens=int(sample_sampling["max_tokens"]),
            temperature=float(sample_sampling["temperature"]),
            top_p=float(sample_sampling["top_p"]),
            top_k=int(sample_sampling["top_k"]),
            min_p=float(sample_sampling["min_p"]),
            repetition_penalty=float(sample_sampling["repetition_penalty"]),
            presence_penalty=float(sample_sampling["presence_penalty"]),
        )
        sampler = vllm_resolved_sampler(sampling)
        record = record_from_token_ids(
            tokenizer,
            str(sample["prompt"]),
            prompt_token_ids=prompt_ids,
            response_token_ids=completion_ids,
            max_new_tokens=sampling.max_new_tokens,
            sampler=sampler,
        )
        masks = position_masks(
            prompt_length=len(prompt_ids),
            completion_length=len(completion_ids),
            code_token_start=code_start,
            code_token_end=code_end,
            followup_length=len(followup_ids),
        )
        original_length = len(prompt_ids) + len(completion_ids)
        original_masks = {
            name: mask[:original_length] for name, mask in masks.items() if name != "followup_last"
        }
        original_vectors = capture_record_pooled(
            model,
            record,
            original_masks,
            poolings=["mean"],
        )
        followup_record = record_from_token_ids(
            tokenizer,
            str(sample["prompt"]),
            prompt_token_ids=[*prompt_ids, *completion_ids],
            response_token_ids=followup_ids,
            max_new_tokens=max(1, len(followup_ids)),
            sampler=sampler,
        )
        followup_vectors = capture_record_pooled(
            model,
            followup_record,
            {"followup_last": masks["followup_last"]},
            poolings=["mean"],
        )
        layer_ids = sorted(original_vectors["last_prompt", "mean"])
        if sorted(followup_vectors["followup_last", "mean"]) != layer_ids:
            raise RuntimeError(
                "follow-up capture returned a different layer set than the original capture"
            )
        row_vectors = torch.stack(
            [
                torch.stack(
                    [
                        original_vectors[name, "mean"][layer]
                        if name != "followup_last"
                        else followup_vectors[name, "mean"][layer]
                        for name in POSITION_NAMES
                    ]
                )
                for layer in layer_ids
            ]
        )
        if row_vectors.ndim != 3:  # noqa: PLR2004
            raise RuntimeError(f"unexpected pooled activation shape {tuple(row_vectors.shape)}")
        row = row_by_id[sample_id]
        if shard is None:
            shard = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=ACTIVATION_NUMPY_DTYPE,
                shape=(
                    len(samples),
                    row_vectors.shape[0],
                    len(POSITION_NAMES),
                    row_vectors.shape[2],
                ),
            )
            write_json_atomic(
                index_path,
                {
                    "model_key": model_key,
                    "model_id": model_id,
                    "adapter_id": adapter_id,
                    "sample_ids": identity["sample_ids"],
                    "positions": list(POSITION_NAMES),
                    "shape": list(shard.shape),
                    "dtype": str(shard.dtype),
                    "complete": False,
                },
            )
        elif tuple(shard.shape[1:]) != tuple(row_vectors.shape):
            raise ValueError(
                f"activation shape changed from {tuple(shard.shape[1:])} to {tuple(row_vectors.shape)}"
            )
        shard[row] = _finite_float32_row(row_vectors)
        shard.flush()
        progress_record = {
            "sample_id": sample_id,
            "model_key": model_key,
            "row": row,
        }
        append_jsonl_ascii(progress_path, progress_record)
        ledger.mark_record_complete(sample_id)

    if shard is None:
        raise RuntimeError("no activation rows were captured")
    if reference_requested and not reference_complete:
        _capture_reference_probe(
            run_dir=run_dir,
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            sampler=reference_sampler,
            base_model=base_model,
            model_key=model_key,
            model_id=model_id,
            adapter_id=adapter_id,
        )
    shard.flush()
    write_json_atomic(
        index_path,
        {
            "model_key": model_key,
            "model_id": model_id,
            "adapter_id": adapter_id,
            "sample_ids": identity["sample_ids"],
            "positions": list(POSITION_NAMES),
            "shape": list(shard.shape),
            "dtype": str(shard.dtype),
            "complete": True,
        },
    )
    manifest_path = run_dir / "activation-manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest[model_key] = {
        "shard": output_path.name,
        "row_index": index_path.name,
        "sample_count": len(samples),
        "positions": list(POSITION_NAMES),
        "shape": list(shard.shape),
        "dtype": str(shard.dtype),
    }
    write_json_atomic(manifest_path, manifest)
    del model
    torch.cuda.empty_cache()
    return output_path


def capture_models(
    *,
    run_dir: Path,
    model_keys: tuple[str, ...],
    base_model: str = BASE_MODEL_ID,
    max_items: int | None = None,
    capture_reference: bool = False,
) -> list[Path]:
    """Capture model checkpoints sequentially, never overlapping vLLM with Hugging Face."""
    run_dir.mkdir(parents=True, exist_ok=True)
    return [
        capture_model(
            run_dir=run_dir,
            model_key=model_key,
            base_model=base_model,
            sample_path=run_dir / f"samples.{artifact_model_key(model_key)}.jsonl",
            max_items=max_items,
            capture_reference=capture_reference,
        )
        for model_key in model_keys
    ]
