"""Capture per-token Atlas residual activations and probe hack labels over solution positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch

from reward_hacking.atlas.capture import _code_token_span  # pyright: ignore[reportPrivateUsage]
from reward_hacking.atlas.grading import extract_code
from reward_hacking.atlas.readout import rank_auc
from reward_hacking.atlas.sampling import (
    BASE_MODEL_ID,
    artifact_model_key,
    language_adapter_for,
    resolve_model_key,
)
from reward_hacking.interp.jsonl_resume import ResumeMismatchError, resume_records
from reward_hacking.interp.linear_probe import fit_logistic_probe

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TOKENS = 8
DEFAULT_COMPLETION_TOKENS = 160
DEFAULT_LAYER_STRIDE = 4
DEFAULT_PERMUTATIONS = 20
DEFAULT_CV_FOLDS = 5
SUMMARY_CODE_OFFSETS = (-1, 0, 1, 2, 4, 8)
N_ARRAY_DIMS = 4
N_LABELS = 2
N_FEATURE_DIMS = 2
MIN_CV_FOLDS = 2
MODEL_FLOAT_DTYPE = np.dtype(np.float32)
MASK_DTYPE = np.dtype(np.bool_)


class _TextConfig(Protocol):
    num_hidden_layers: int
    hidden_size: int


class _ModelConfig(Protocol):
    def get_text_config(self) -> _TextConfig:
        """Return the language-model config (the config itself for text-only models)."""
        ...


class _ModelOutput(Protocol):
    hidden_states: tuple[torch.Tensor, ...] | None


class _AtlasModel(Protocol):
    config: _ModelConfig

    def parameters(self) -> Iterator[torch.nn.Parameter]: ...

    def eval(self) -> object: ...

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
    ) -> _ModelOutput: ...


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


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_character_offsets(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: list[int],
    completion_text: str,
) -> list[list[int]]:
    """Map each generated token to its decoded Python-character interval."""
    prefix_lengths = [0]
    previous_prefix = ""
    for end in range(1, len(token_ids) + 1):
        prefix = cast("str", tokenizer.decode(token_ids[:end], skip_special_tokens=True))
        if not prefix.startswith(previous_prefix):
            raise ValueError(
                "decoding generated token prefixes is not monotonic for this completion"
            )
        prefix_lengths.append(len(prefix))
        previous_prefix = prefix
    if previous_prefix != completion_text:
        raise ValueError("decoding saved completion token ids no longer matches its saved text")
    return [[prefix_lengths[index], prefix_lengths[index + 1]] for index in range(len(token_ids))]


def _selected_layers(n_layers: int, stride: int) -> list[int]:
    if n_layers < 1:
        raise ValueError(f"model must expose at least one transformer layer, got {n_layers}")
    if stride < 1:
        raise ValueError(f"layer stride must be positive, got {stride}")
    selected = list(range(0, n_layers, stride))
    if selected[-1] != n_layers - 1:
        selected.append(n_layers - 1)
    return selected


def _sample_rows(
    samples: list[dict[str, Any]], grades: list[dict[str, Any]], base_model: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not samples:
        raise ValueError("the Atlas sample file is empty")
    grade_by_id = {str(grade["sample_id"]): grade for grade in grades}
    if len(grade_by_id) != len(grades):
        raise ValueError("the Atlas grade file contains duplicate sample ids")
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("the Atlas sample file contains duplicate sample ids")
    sample_pairs = [(str(sample["item_id"]), int(sample["sample_index"])) for sample in samples]
    if len(set(sample_pairs)) != len(sample_pairs):
        raise ValueError("the Atlas sample file contains duplicate (item_id, sample_index) pairs")
    missing_grades = sorted(set(sample_ids) - set(grade_by_id))
    extra_grades = sorted(set(grade_by_id) - set(sample_ids))
    if missing_grades or extra_grades:
        raise ValueError(
            "sample/grade ids differ: "
            f"{len(missing_grades)} grades missing and {len(extra_grades)} extra"
        )
    sample_models = {str(sample.get("model_id", base_model)) for sample in samples}
    if sample_models != {base_model}:
        raise ValueError(
            f"samples use base models {sorted(sample_models)}, not requested {base_model!r}"
        )
    joined = [(sample, grade_by_id[str(sample["sample_id"])]) for sample in samples]
    for sample, grade in joined:
        if str(sample["item_id"]) != str(grade["item_id"]):
            raise ValueError(f"grade item id does not match sample {sample['sample_id']}")
        if not isinstance(grade["hack"], bool) or not isinstance(grade["passes_all"], bool):
            raise TypeError(f"grade labels for {sample['sample_id']} must be booleans")
    return sorted(joined, key=lambda pair: (str(pair[0]["item_id"]), int(pair[0]["sample_index"])))


def _optional_code_token_start(
    sample: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    completion_ids: list[int],
    completion_text: str,
) -> int | None:
    """Return the first code token, or None when the code span cannot be mapped onto token boundaries.

    A completion whose token prefixes do not decode monotonically (a multi-token character split across
    tokens) has no exact code span; its code-relative positions are skipped downstream, the rest is kept.
    """
    try:
        code_token_start, _ = _code_token_span(
            tokenizer, completion_ids, completion_text, extract_code(completion_text)
        )
    except ValueError as error:
        logger.warning("sample %s has no mappable code span: %s", sample["sample_id"], error)
        return None
    return code_token_start


def _metadata_row(
    sample: Mapping[str, Any],
    grade: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, object]:
    prompt_ids = [int(token_id) for token_id in sample["prompt_token_ids"]]
    completion_ids = [int(token_id) for token_id in sample["completion_token_ids"]]
    completion_text = str(sample["completion_text"])
    if not completion_ids:
        raise ValueError(f"sample {sample['sample_id']} has no completion tokens")
    code_token_start = _optional_code_token_start(
        sample, tokenizer, completion_ids, completion_text
    )
    return {
        "sample_id": str(sample["sample_id"]),
        "item_id": str(sample["item_id"]),
        "sample_index": int(sample["sample_index"]),
        "hack_label": bool(grade["hack"]),
        "pass_label": bool(grade["passes_all"]),
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion_ids,
        "completion_char_offsets": (
            None
            if code_token_start is None
            else _token_character_offsets(tokenizer, completion_ids, completion_text)
        ),
        "code_token_start": code_token_start,
    }


def _token_sweep_key(model_key: str, samples_from: str | None = None) -> str:
    sample_source = model_key if samples_from is None else samples_from
    model_artifact_key = artifact_model_key(model_key)
    if sample_source == model_key:
        return model_artifact_key
    return f"{model_artifact_key}--reads--{artifact_model_key(sample_source)}"


def _log_device_and_vram(model: _AtlasModel) -> None:
    device = next(model.parameters()).device
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        logger.info(
            "token sweep device=%s free_vram_gib=%.2f total_vram_gib=%.2f",
            device,
            free_bytes / 1024**3,
            total_bytes / 1024**3,
        )
    else:
        logger.info("token sweep device=%s free_vram=unavailable", device)


def _capture_row(  # noqa: PLR0913 - prompt and completion boundaries are distinct inputs
    model: _AtlasModel,
    prompt_ids: list[int],
    completion_ids: list[int],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    layer_ids: list[int],
    d_model: int,
) -> tuple[np.ndarray, np.ndarray]:
    prompt_tail = prompt_ids[-prompt_tokens:]
    completion_prefix = completion_ids[:completion_tokens]
    input_ids_list = [*prompt_ids, *completion_prefix]
    if not input_ids_list:
        raise ValueError("cannot capture an empty prompt and completion")
    input_ids = torch.tensor(
        [input_ids_list], dtype=torch.long, device=next(model.parameters()).device
    )
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = output.hidden_states
    if hidden_states is None:
        raise RuntimeError("model forward did not return hidden states")
    sequence_length = len(input_ids_list)
    expected_state_count = int(model.config.get_text_config().num_hidden_layers) + 1
    if len(hidden_states) != expected_state_count:
        raise RuntimeError(
            f"model returned {len(hidden_states)} hidden states, expected {expected_state_count}"
        )
    row = np.zeros(
        (len(layer_ids), prompt_tokens + completion_tokens, d_model), dtype=MODEL_FLOAT_DTYPE
    )
    mask = np.zeros(prompt_tokens + completion_tokens, dtype=MASK_DTYPE)
    prompt_offset = prompt_tokens - len(prompt_tail)
    prompt_context_start = len(prompt_ids) - len(prompt_tail)
    mask[prompt_offset:prompt_tokens] = True
    mask[prompt_tokens : prompt_tokens + len(completion_prefix)] = True
    for output_layer_index, layer_id in enumerate(layer_ids):
        layer = hidden_states[layer_id + 1]
        if layer.shape != (1, sequence_length, d_model):
            raise RuntimeError(
                f"layer {layer_id} returned shape {tuple(layer.shape)}, "
                f"expected {(1, sequence_length, d_model)}"
            )
        layer_values = layer[0].to(device="cpu", dtype=torch.float32).numpy()
        if not np.isfinite(layer_values).all():
            raise ValueError(f"layer {layer_id} contains non-finite activations")
        if prompt_tail:
            row[output_layer_index, prompt_offset:prompt_tokens] = layer_values[
                prompt_context_start : len(prompt_ids)
            ]
        if completion_prefix:
            row[
                output_layer_index,
                prompt_tokens : prompt_tokens + len(completion_prefix),
            ] = layer_values[len(prompt_ids) :]
    return row, mask


def capture_model(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    run_dir: Path,
    model_key: str,
    samples_from: str | None = None,
    base_model: str = BASE_MODEL_ID,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    completion_tokens: int = DEFAULT_COMPLETION_TOKENS,
    layer_stride: int = DEFAULT_LAYER_STRIDE,
) -> Path:
    """Capture one Atlas model's final prompt tokens and the first completion tokens."""
    from peft import PeftModel  # noqa: PLC0415

    from reward_hacking.interp.directions import load_model_and_tokenizer  # noqa: PLC0415

    if prompt_tokens < 1 or completion_tokens < 1:
        raise ValueError("prompt and completion token counts must be positive")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir / "token-sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_source = model_key if samples_from is None else samples_from
    output_key = _token_sweep_key(model_key, sample_source)
    sample_path = run_dir / f"samples.{artifact_model_key(sample_source)}.jsonl"
    grade_path = run_dir / f"grades.{artifact_model_key(sample_source)}.jsonl"
    samples = _read_jsonl(sample_path)
    joined = _sample_rows(samples, _read_jsonl(grade_path), base_model)
    model_id, adapter_id = resolve_model_key(model_key, base_model=base_model)
    loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_id, dtype=torch.bfloat16)
    tokenizer = cast("PreTrainedTokenizerBase", loaded_tokenizer)
    model = cast("_AtlasModel", loaded_model)
    adapter_path = language_adapter_for(model_key, run_dir, base_model=base_model)
    if adapter_path is not None:
        model = cast(
            "_AtlasModel",
            PeftModel.from_pretrained(cast("Any", loaded_model), str(adapter_path)),
        )
    model.eval()
    _log_device_and_vram(model)
    n_layers = int(model.config.get_text_config().num_hidden_layers)
    layer_ids = _selected_layers(n_layers, layer_stride)
    d_model = int(model.config.get_text_config().hidden_size)
    sample_ids = [str(sample["sample_id"]) for sample, _ in joined]
    identity: dict[str, object] = {
        "stage": "atlas_token_sweep",
        "model_key": model_key,
        "samples_from": sample_source,
        "base_model": model_id,
        "adapter": adapter_id,
        "sample_file_sha256": _file_sha256(sample_path),
        "grade_file_sha256": _file_sha256(grade_path),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "layer_ids": layer_ids,
        "sample_ids": sample_ids,
        "d_model": d_model,
        "dtype": "float32",
    }
    activation_path = output_dir / f"activations.{output_key}.npy"
    mask_path = output_dir / f"mask.{output_key}.npy"
    index_path = output_dir / f"index.{output_key}.json"
    progress_path = output_dir / f"progress.{output_key}.jsonl"
    resumed, ledger = resume_records(
        progress_path,
        identity=identity,
        unit_of=lambda record: str(record["model_key"]),
        record_key_of=lambda record: str(record["sample_id"]),
    )
    completed = {str(record["sample_id"]) for record in resumed.kept_records}
    expected_activation_shape = (
        len(joined),
        len(layer_ids),
        prompt_tokens + completion_tokens,
        d_model,
    )
    expected_mask_shape = (len(joined), prompt_tokens + completion_tokens)
    if activation_path.exists() != mask_path.exists():
        raise ResumeMismatchError("token-sweep activation and mask arrays must exist together")
    if activation_path.exists():
        activations = np.load(activation_path, mmap_mode="r+")
        mask_array = np.load(mask_path, mmap_mode="r+")
        if activations.shape != expected_activation_shape or activations.dtype != MODEL_FLOAT_DTYPE:
            raise ResumeMismatchError(
                "existing activation shard has shape/dtype "
                f"{activations.shape}/{activations.dtype}, "
                f"expected {expected_activation_shape}/{MODEL_FLOAT_DTYPE}"
            )
        if mask_array.shape != expected_mask_shape or mask_array.dtype != MASK_DTYPE:
            raise ResumeMismatchError(
                f"existing mask shard has shape/dtype {mask_array.shape}/{mask_array.dtype}, "
                f"expected {expected_mask_shape}/{MASK_DTYPE}"
            )
        if index_path.exists():
            saved_index = json.loads(index_path.read_text(encoding="utf-8"))
            if saved_index.get("identity") != identity:
                raise ResumeMismatchError("token-sweep index identity differs from this run")
    else:
        if completed:
            raise ResumeMismatchError(
                "token-sweep progress exists but the activation arrays are missing"
            )
        activations = np.lib.format.open_memmap(
            activation_path,
            mode="w+",
            dtype=MODEL_FLOAT_DTYPE,
            shape=expected_activation_shape,
        )
        mask_array = np.lib.format.open_memmap(
            mask_path,
            mode="w+",
            dtype=MASK_DTYPE,
            shape=expected_mask_shape,
        )

    metadata_rows = [_metadata_row(sample, grade, tokenizer) for sample, grade in joined]
    index_payload: dict[str, object] = {
        "model_key": model_key,
        "samples_from": sample_source,
        "output_key": output_key,
        "model_id": model_id,
        "adapter_id": adapter_id,
        "identity": identity,
        "sample_ids": sample_ids,
        "layer_ids": layer_ids,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "position_layout": {
            "prompt_offsets": list(range(-prompt_tokens, 0)),
            "completion_offsets": list(range(completion_tokens)),
            "prompt_padding": "left",
            "completion_padding": "right",
            "activation_alignment": "hidden state after the token at that position",
        },
        "shape": list(expected_activation_shape),
        "dtype": "float32",
        "mask_shape": list(expected_mask_shape),
        "rows": metadata_rows,
        "complete": False,
    }
    _write_json(index_path, index_payload)
    row_by_sample_id = {str(row["sample_id"]): index for index, row in enumerate(metadata_rows)}
    for sample, _grade in joined:
        sample_id = str(sample["sample_id"])
        if sample_id in completed:
            continue
        prompt_ids = [int(token_id) for token_id in sample["prompt_token_ids"]]
        if not prompt_ids:
            raise ValueError(f"sample {sample_id} has no prompt tokens")
        completion_ids = [int(token_id) for token_id in sample["completion_token_ids"]]
        row, row_mask = _capture_row(
            model,
            prompt_ids,
            completion_ids,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            layer_ids=layer_ids,
            d_model=d_model,
        )
        if not np.isfinite(row).all():
            raise ValueError(f"sample {sample_id} produced non-finite activations")
        row_index = row_by_sample_id[sample_id]
        activations[row_index] = row
        mask_array[row_index] = row_mask
        activations.flush()
        mask_array.flush()
        progress_record = {"sample_id": sample_id, "model_key": model_key, "row": row_index}
        with progress_path.open("a", encoding="utf-8") as progress_file:
            progress_file.write(json.dumps(progress_record, sort_keys=True) + "\n")
            progress_file.flush()
        ledger.mark_record_complete(sample_id)
        completed.add(sample_id)
    index_payload["complete"] = len(completed) == len(sample_ids)
    _write_json(index_path, index_payload)
    logger.info(
        "token sweep model=%s samples_from=%s samples=%d rows_resumed=%d shape=%s",
        model_key,
        sample_source,
        len(joined),
        len(completed),
        expected_activation_shape,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return activation_path


def capture_models(  # noqa: PLR0913 - preserve explicit CLI configuration
    *,
    run_dir: Path,
    model_keys: Sequence[str],
    samples_from: str | None = None,
    base_model: str = BASE_MODEL_ID,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    completion_tokens: int = DEFAULT_COMPLETION_TOKENS,
    layer_stride: int = DEFAULT_LAYER_STRIDE,
) -> list[Path]:
    """Capture requested checkpoints sequentially, loading and releasing one model at a time."""
    return [
        capture_model(
            run_dir=run_dir,
            model_key=model_key,
            samples_from=samples_from,
            base_model=base_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            layer_stride=layer_stride,
        )
        for model_key in model_keys
    ]


def _folds(
    labels: np.ndarray, groups: np.ndarray, *, n_folds: int = DEFAULT_CV_FOLDS
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    if labels.ndim != 1 or groups.ndim != 1 or labels.shape != groups.shape:
        raise ValueError("labels and groups must be aligned one-dimensional arrays")
    if n_folds < MIN_CV_FOLDS:
        raise ValueError("grouped cross-validation needs at least two folds")
    unique_groups, inverse, group_sizes = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique_groups) < MIN_CV_FOLDS or len(np.unique(labels)) != N_LABELS:
        return None
    fold_count = min(n_folds, len(unique_groups))
    fold_sizes = np.zeros(fold_count, dtype=np.int64)
    group_folds = np.empty(len(unique_groups), dtype=np.int64)
    for group_index in np.argsort(group_sizes)[::-1]:
        fold_index = int(np.argmin(fold_sizes))
        group_folds[group_index] = fold_index
        fold_sizes[fold_index] += group_sizes[group_index]
    sample_folds = group_folds[inverse]
    splits = [
        (np.flatnonzero(sample_folds != fold_index), np.flatnonzero(sample_folds == fold_index))
        for fold_index in range(fold_count)
    ]
    if any(
        len(train_indices) == 0 or len(np.unique(labels[train_indices])) != N_LABELS
        for train_indices, _ in splits
    ):
        return None
    return splits


def _fit_grouped_auc(
    features: np.ndarray,
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    out_of_fold_scores = np.empty(len(labels), dtype=np.float64)
    for train_indices, test_indices in splits:
        training_features = torch.from_numpy(features[train_indices]).to(dtype=torch.float32)
        training_labels = torch.from_numpy(labels[train_indices]).to(dtype=torch.float32)
        mean = training_features.mean(dim=0)
        scale = training_features.std(dim=0, unbiased=False).clamp_min(1e-6)
        standardized_training = (training_features - mean) / scale
        fit = fit_logistic_probe(
            standardized_training,
            training_labels,
            l2_strength=1e-2,
            max_iter=100,
        )
        test_features = torch.from_numpy(features[test_indices]).to(dtype=torch.float32)
        standardized_test = (test_features - mean) / scale
        out_of_fold_scores[test_indices] = (standardized_test @ fit.weights + fit.bias).numpy()
    return rank_auc(torch.from_numpy(labels), torch.from_numpy(out_of_fold_scores))


def _probe_position(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, object]:
    if (
        features.ndim != N_FEATURE_DIMS
        or len(features) != len(labels)
        or labels.shape != groups.shape
    ):
        raise ValueError("features, labels, and groups must have matching rows")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    splits = _folds(labels, groups)
    if splits is None:
        return {
            "status": "too_few",
            "n_samples": len(labels),
            "n_items": len(np.unique(groups)),
            "auc": None,
            "null_auc_max": None,
            "null_auc_mean": None,
            "n_permutations": 0,
        }
    auc = _fit_grouped_auc(features, labels, splits)
    generator = np.random.default_rng(seed)
    null_aucs: list[float] = []
    attempts = 0
    while len(null_aucs) < n_permutations and attempts < n_permutations * 100:
        attempts += 1
        shuffled_labels = generator.permutation(labels)
        shuffled_splits = _folds(shuffled_labels, groups, n_folds=len(splits))
        if shuffled_splits is None:
            continue
        null_aucs.append(_fit_grouped_auc(features, shuffled_labels, shuffled_splits))
    if len(null_aucs) != n_permutations:
        raise RuntimeError(
            f"could only form {len(null_aucs)} valid grouped-CV shuffles from {attempts} attempts"
        )
    return {
        "status": "ok",
        "n_samples": len(labels),
        "n_items": len(np.unique(groups)),
        "auc": auc,
        "null_auc_max": max(null_aucs),
        "null_auc_mean": float(np.mean(null_aucs)),
        "n_permutations": n_permutations,
    }


def _position_seed(seed: int, layer_id: int, anchor: str, offset: int) -> int:
    seed_material = f"{seed}:{layer_id}:{anchor}:{offset}".encode()
    return int.from_bytes(hashlib.sha256(seed_material).digest()[:4], byteorder="big")


def _read_sweep(
    output_dir: Path, model_key: str, *, samples_from: str | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    output_key = _token_sweep_key(model_key, samples_from)
    activation_path = output_dir / f"activations.{output_key}.npy"
    mask_path = output_dir / f"mask.{output_key}.npy"
    index_path = output_dir / f"index.{output_key}.json"
    activations = np.load(activation_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("complete") is not True:
        raise ValueError(f"{index_path} is incomplete and cannot be analysed")
    if activations.ndim != N_ARRAY_DIMS or activations.dtype != MODEL_FLOAT_DTYPE:
        raise ValueError(f"{activation_path} must be a four-dimensional float32 array")
    if mask.shape != (activations.shape[0], activations.shape[2]) or mask.dtype != MASK_DTYPE:
        raise ValueError(f"{mask_path} does not align with its activation shard")
    if (
        len(index["rows"]) != activations.shape[0]
        or len(index["layer_ids"]) != activations.shape[1]
    ):
        raise ValueError(f"{index_path} metadata does not align with its activation shard")
    return activations, mask, index


def _assert_teacher_forced_alignment(
    teacher_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
) -> None:
    def by_sample_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        row_map = {str(row["sample_id"]): row for row in rows}
        if len(row_map) != len(rows):
            raise ValueError("token-sweep indexes contain duplicate sample ids")
        return row_map

    teacher_by_id = by_sample_id(teacher_rows)
    adapter_by_id = by_sample_id(adapter_rows)
    if set(teacher_by_id) != set(adapter_by_id):
        raise ValueError(
            "teacher-forced and adapter sweeps do not contain the same sample identities"
        )
    for sample_id, adapter_row in adapter_by_id.items():
        teacher_row = teacher_by_id[sample_id]
        for token_field in ("prompt_token_ids", "completion_token_ids"):
            teacher_tokens = [int(token_id) for token_id in teacher_row[token_field]]
            adapter_tokens = [int(token_id) for token_id in adapter_row[token_field]]
            if teacher_tokens != adapter_tokens:
                raise ValueError(
                    f"teacher-forced {token_field.replace('_', ' ')} differ for sample {sample_id}"
                )
        for field in ("item_id", "sample_index", "hack_label", "pass_label", "code_token_start"):
            if teacher_row[field] != adapter_row[field]:
                raise ValueError(f"teacher-forced {field} differs for sample {sample_id}")


def _aligned_probe_rows(
    model_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    def pair_key(row: Mapping[str, Any]) -> tuple[str, int]:
        return str(row["item_id"]), int(row["sample_index"])

    model_indices = {pair_key(row): index for index, row in enumerate(model_rows)}
    adapter_by_pair = {pair_key(row): row for row in adapter_rows}
    if len(model_indices) != len(model_rows) or len(adapter_by_pair) != len(adapter_rows):
        raise ValueError("token-sweep indexes contain duplicate (item_id, sample_index) pairs")
    if set(model_indices) != set(adapter_by_pair):
        raise ValueError("base and adapter sweeps do not contain the same sample pairs")
    return [(model_indices[pair], adapter_by_pair[pair]) for pair in adapter_by_pair]


def _anchor_offsets(  # noqa: PLR0913 - prompt and code position inputs define one layout
    rows: list[dict[str, Any]],
    mask: np.ndarray,
    row_indices: list[int],
    *,
    prompt_tokens: int,
    anchor: str,
    completion_tokens: int,
) -> list[int]:
    if anchor == "completion_start":
        return list(range(-prompt_tokens, completion_tokens))
    offsets = {
        int(position) - prompt_tokens - int(rows[row_index]["code_token_start"])
        for row_index in row_indices
        if rows[row_index]["code_token_start"] is not None
        for position in np.flatnonzero(mask[row_index])
    }
    return sorted(offsets)


def _analyze_model(  # noqa: PLR0913 - model data and probe settings remain explicit
    *,
    model_key: str,
    activations: np.ndarray,
    mask: np.ndarray,
    model_index: dict[str, Any],
    adapter_rows: list[dict[str, Any]],
    n_permutations: int,
    seed: int,
) -> list[dict[str, object]]:
    model_rows = model_index["rows"]
    aligned = _aligned_probe_rows(model_rows, adapter_rows)
    selected_rows = [
        (model_row, adapter_row)
        for model_row, adapter_row in aligned
        if bool(adapter_row["hack_label"]) or bool(adapter_row["pass_label"])
    ]
    row_indices = [model_row for model_row, _ in selected_rows]
    labels = np.array([int(bool(row["hack_label"])) for _, row in selected_rows], dtype=np.int64)
    groups = np.array([str(row["item_id"]) for _, row in selected_rows])
    prompt_tokens = int(model_index["prompt_tokens"])
    completion_tokens = int(model_index["completion_tokens"])
    layer_ids = [int(layer_id) for layer_id in model_index["layer_ids"]]
    results: list[dict[str, object]] = []
    for anchor in ("completion_start", "code_start"):
        offsets = _anchor_offsets(
            model_rows,
            mask,
            row_indices,
            prompt_tokens=prompt_tokens,
            anchor=anchor,
            completion_tokens=completion_tokens,
        )
        for layer_position, layer_id in enumerate(layer_ids):
            for offset in offsets:
                feature_rows: list[np.ndarray] = []
                valid_labels: list[int] = []
                valid_groups: list[str] = []
                for selected_index, model_row_index in enumerate(row_indices):
                    row = model_rows[model_row_index]
                    absolute_position = prompt_tokens + offset
                    if anchor == "code_start":
                        if row["code_token_start"] is None:
                            continue
                        absolute_position += int(row["code_token_start"])
                    if (
                        0 <= absolute_position < mask.shape[1]
                        and mask[model_row_index, absolute_position]
                    ):
                        feature_rows.append(
                            activations[model_row_index, layer_position, absolute_position]
                        )
                        valid_labels.append(int(labels[selected_index]))
                        valid_groups.append(str(groups[selected_index]))
                if feature_rows:
                    features = np.asarray(feature_rows, dtype=np.float32)
                    position_labels = np.asarray(valid_labels, dtype=np.int64)
                    position_groups = np.asarray(valid_groups)
                    probe = _probe_position(
                        features,
                        position_labels,
                        position_groups,
                        n_permutations=n_permutations,
                        seed=_position_seed(seed, layer_id, anchor, offset),
                    )
                else:
                    probe = {
                        "status": "too_few",
                        "n_samples": 0,
                        "n_items": 0,
                        "auc": None,
                        "null_auc_max": None,
                        "null_auc_mean": None,
                        "n_permutations": 0,
                    }
                results.append(
                    {
                        "model_key": model_key,
                        "anchor": anchor,
                        "offset": offset,
                        "layer_id": layer_id,
                        **probe,
                    }
                )
    return results


def _markdown_table(report: Mapping[str, Any]) -> str:
    result_rows = report["results"]
    arms = list(dict.fromkeys(str(result["model_key"]) for result in result_rows))
    lines = [
        "# Atlas token-sweep grouped probes",
        "",
        (
            "Rows compare adapter hacks with honest passes; failed samples are excluded. "
            "`base_own_completions` uses the base model's own completions, so its completion-position "
            "rows compare different texts from the adapter. `base_reads_adapter_tokens` teacher-forces "
            "the base model on the adapter's exact tokens. Token activations are post-token states; "
            "offset -1 is the state before a token."
        ),
        "",
        (
            "| Arm | Anchor | Best layer / offset | Best AUC | Null max | Code offset -1 | Code offset 0 | "
            "Code offset 1 | Code offset 2 | Code offset 4 | Code offset 8 |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        for anchor in ("completion_start", "code_start"):
            arm_anchor_results = [
                result
                for result in result_rows
                if result["model_key"] == arm
                and result["anchor"] == anchor
                and result["auc"] is not None
            ]
            best = (
                max(arm_anchor_results, key=lambda result: float(result["auc"]))
                if arm_anchor_results
                else None
            )
            if best is None:
                best_position = best_auc = null_max = ""
                best_layer = None
            else:
                best_layer = int(best["layer_id"])
                best_position = f"{best_layer} / {best['offset']}"
                best_auc = f"{float(best['auc']):.3f}"
                null_max = (
                    "" if best["null_auc_max"] is None else f"{float(best['null_auc_max']):.3f}"
                )
            code_aucs = {
                int(result["offset"]): result["auc"]
                for result in result_rows
                if result["model_key"] == arm
                and result["anchor"] == "code_start"
                and best_layer is not None
                and int(result["layer_id"]) == best_layer
                and int(result["offset"]) in SUMMARY_CODE_OFFSETS
            }
            code_auc_cells = [
                "" if code_aucs.get(offset) is None else f"{float(code_aucs[offset]):.3f}"
                for offset in SUMMARY_CODE_OFFSETS
            ]
            lines.append(
                f"| {arm} | {anchor} | {best_position} | {best_auc} | {null_max} | "
                + " | ".join(code_auc_cells)
                + " |"
            )
    lines.extend(
        [
            "",
            "| Model | Anchor | Offset | Layer | Samples | Items | AUC | Shuffled null max | Status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for result in result_rows:
        auc = "" if result["auc"] is None else f"{result['auc']:.3f}"
        null_max = "" if result["null_auc_max"] is None else f"{result['null_auc_max']:.3f}"
        lines.append(
            f"| {result['model_key']} | {result['anchor']} | {result['offset']} "
            f"| {result['layer_id']} | {result['n_samples']} | {result['n_items']} "
            f"| {auc} | {null_max} | {result['status']} |"
        )
    return "\n".join(lines) + "\n"


def analyse_sweeps(  # noqa: C901 - one validation pass over three optional arms
    *,
    run_dir: Path,
    adapter_key: str,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Run item-grouped hack probes at every captured completion and code-relative position."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    output_dir = run_dir / "token-sweep"
    base_activations, base_mask, base_index = _read_sweep(output_dir, "base")
    adapter_activations, adapter_mask, adapter_index = _read_sweep(output_dir, adapter_key)
    teacher_output_key = _token_sweep_key("base", adapter_key)
    teacher_paths = (
        output_dir / f"activations.{teacher_output_key}.npy",
        output_dir / f"mask.{teacher_output_key}.npy",
        output_dir / f"index.{teacher_output_key}.json",
    )
    teacher_sweep = None
    if any(path.exists() for path in teacher_paths):
        teacher_sweep = _read_sweep(output_dir, "base", samples_from=adapter_key)
        _assert_teacher_forced_alignment(teacher_sweep[2]["rows"], adapter_index["rows"])
    reference_index = adapter_index
    comparable_sweeps = [
        ("base", base_activations, base_index),
        ("adapter", adapter_activations, adapter_index),
    ]
    if teacher_sweep is not None:
        comparable_sweeps.append(
            ("base reading adapter tokens", teacher_sweep[0], teacher_sweep[2])
        )
    for arm_name, arm_activations, arm_index in comparable_sweeps:
        if arm_index["layer_ids"] != reference_index["layer_ids"]:
            raise ValueError(f"{arm_name} and adapter sweeps have different layer ids")
        if arm_index["prompt_tokens"] != reference_index["prompt_tokens"]:
            raise ValueError(f"{arm_name} and adapter sweeps have different prompt position counts")
        if arm_index["completion_tokens"] != reference_index["completion_tokens"]:
            raise ValueError(
                f"{arm_name} and adapter sweeps have different completion position counts"
            )
        if arm_activations.shape[3] != adapter_activations.shape[3]:
            raise ValueError(f"{arm_name} and adapter sweeps have different hidden sizes")
    adapter_rows = adapter_index["rows"]
    if any(
        bool(row["hack_label"]) == bool(row["pass_label"])
        for row in adapter_rows
        if bool(row["hack_label"]) or bool(row["pass_label"])
    ):
        raise ValueError("adapter grades must separate hacks from honest passes")
    model_sweeps = [
        ("base_own_completions", base_activations, base_mask, base_index),
        (adapter_key, adapter_activations, adapter_mask, adapter_index),
    ]
    if teacher_sweep is not None:
        model_sweeps.append(
            ("base_reads_adapter_tokens", teacher_sweep[0], teacher_sweep[1], teacher_sweep[2])
        )
    results = [
        result
        for model_key, activations, mask, index in model_sweeps
        for result in _analyze_model(
            model_key=model_key,
            activations=activations,
            mask=mask,
            model_index=index,
            adapter_rows=adapter_rows,
            n_permutations=n_permutations,
            seed=seed,
        )
    ]
    report: dict[str, object] = {
        "adapter_key": adapter_key,
        "label_source": f"{adapter_key} grade hack field; pass field passes_all",
        "positive_label": "hack",
        "negative_label": "honest_pass",
        "excluded_label": "fail",
        "grouping": "item_id",
        "cross_validation": "GroupKFold",
        "probe": {
            "estimator": "L2-regularized logistic regression",
            "l2_strength": 1e-2,
            "max_iter": 100,
            "feature_scaling": "mean and standard deviation from training fold",
        },
        "n_permutations": n_permutations,
        "seed": seed,
        "results": results,
    }
    json_path = output_dir / f"analysis.{artifact_model_key(adapter_key)}.json"
    markdown_path = output_dir / f"analysis.{artifact_model_key(adapter_key)}.md"
    _write_json(json_path, report)
    markdown_path.write_text(_markdown_table(report), encoding="utf-8")
    return json_path, markdown_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser(
        "capture", help="capture per-token prompt and completion activations"
    )
    capture.add_argument("--run-dir", type=Path, required=True)
    capture.add_argument("--models", nargs="+", required=True)
    capture.add_argument("--samples-from", help="read sample tokens and grades from this model key")
    capture.add_argument("--base-model", default=BASE_MODEL_ID)
    capture.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    capture.add_argument("--completion-tokens", type=int, default=DEFAULT_COMPLETION_TOKENS)
    capture.add_argument("--layer-stride", type=int, default=DEFAULT_LAYER_STRIDE)
    analyse = commands.add_parser(
        "analyse", help="fit grouped probes over captured token positions"
    )
    analyse.add_argument("--run-dir", type=Path, required=True)
    analyse.add_argument("--adapter-key", required=True)
    analyse.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    analyse.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run activation capture or grouped position probes from the command line."""
    arguments = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if arguments.command == "capture":
        capture_models(
            run_dir=arguments.run_dir,
            model_keys=arguments.models,
            samples_from=arguments.samples_from,
            base_model=arguments.base_model,
            prompt_tokens=arguments.prompt_tokens,
            completion_tokens=arguments.completion_tokens,
            layer_stride=arguments.layer_stride,
        )
    else:
        json_path, markdown_path = analyse_sweeps(
            run_dir=arguments.run_dir,
            adapter_key=arguments.adapter_key,
            n_permutations=arguments.permutations,
            seed=arguments.seed,
        )
        logger.info("wrote token-sweep analysis json=%s markdown=%s", json_path, markdown_path)


if __name__ == "__main__":
    main()
