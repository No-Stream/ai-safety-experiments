"""Read Atlas token-sweep activations through one base Gemma Jacobian lens on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from games.workspace_readout import (
    LoadedLens,
    StoredUnembed,
    bootstrap_indices,
    decode_vocab,
    is_word_like,
    load_lens,
)
from reward_hacking.atlas.jsonl_io import write_json

logger = logging.getLogger(__name__)

RUN_DIR = Path("artifacts/atlas/atlas-within-item-kl01-v1")
LENS_PATH = Path("artifacts/jlens/gemma-3-12b-it/lens-n30-t44.pt")
OUTPUT_DIR = RUN_DIR / "workspace-readout-v1"
MODEL_CACHE = Path.home() / ".cache/huggingface/hub/models--google--gemma-3-12b-it"
ADAPTER_KEY = "kl0.1-det0-seed1"
BASE_READ_KEY = "base--reads--kl0.1-det0-seed1"
EXPECTED_LENS_SOURCES = tuple(range(8, 44, 2))
TARGET_LAYER = 44
READ_LAYERS = (8, 16, 24, 32, 40)
SWEEP_LAYERS = (0, 8, 16, 24, 32, 40, 47)
PROMPT_TOKENS = 8
COMPLETION_TOKENS = 160
N_TOP_TOKENS = 50
DEFAULT_BOOTSTRAP_HALVES = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_THREADS = 4
DEFAULT_SEED = 24017
MIN_BOOTSTRAP_PAIRS = 2
MIN_SAMPLES = 2
TWO_DIMENSIONS = 2
GEMMA_HIDDEN_SIZE = 3840
BOOTSTRAP_HALVES_PER_DRAW = 2


@dataclass(frozen=True)
class Sweep:
    """Memory-mapped activations and their row metadata."""

    activations: np.ndarray
    mask: np.ndarray
    rows: list[dict[str, Any]]
    sample_ids: tuple[str, ...]
    layer_ids: tuple[int, ...]
    prompt_tokens: int
    completion_tokens: int
    position_layout: dict[str, Any]


@dataclass(frozen=True)
class ContrastAnalysis:
    """Readable-token contrast report and the token IDs used for overlap checks."""

    report: dict[str, Any]
    risen_ids: tuple[int, ...]
    fallen_ids: tuple[int, ...]


@dataclass(frozen=True)
class CPUUnembedding:
    """CPU Gemma output norm and tied token embedding weights."""

    norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    lm_head_input_dtype: torch.dtype
    rms_eps: float


@dataclass(frozen=True)
class CellAnalysisConfig:
    """Shared lens, projection, and bootstrap settings for one cell."""

    layer: int
    position: str
    lens: LoadedLens
    unembed: StoredUnembed | CPUUnembedding
    vocabulary: list[str]
    readable_mask: torch.Tensor
    n_bootstrap_halves: int
    seed: int
    batch_size: int


@dataclass(frozen=True)
class NormalizedCell:
    """Normalized adapter, base, and matched-norm placebo residuals."""

    adapter: np.ndarray
    base: np.ndarray
    placebo_delta: np.ndarray
    labels: np.ndarray
    item_indices: np.ndarray
    n_items: int


@dataclass(frozen=True)
class PairedRead:
    """Adapter-minus-base read and item aggregates used by the placebo."""

    contrast: ContrastAnalysis
    item_sums: np.ndarray
    item_counts: np.ndarray


@dataclass(frozen=True)
class HackReads:
    """All-sample and within-item hack-minus-honest contrasts."""

    adapter_all: ContrastAnalysis
    base_all: ContrastAnalysis
    adapter_within: ContrastAnalysis
    base_within: ContrastAnalysis
    mixed_item_indices: np.ndarray
    hack_samples: int
    honest_samples: int
    adapter_class_sums: np.ndarray
    base_class_sums: np.ndarray
    class_counts: np.ndarray


def position_indices(
    rows: list[dict[str, Any]],
    mask: np.ndarray,
    position_name: str,
    *,
    prompt_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a named prompt/completion position to valid stored token indices per sample."""
    if mask.ndim != TWO_DIMENSIONS or mask.shape[0] != len(rows):
        raise ValueError("position mask must have one row per sweep record")
    if prompt_tokens < 1 or prompt_tokens > mask.shape[1]:
        raise ValueError("prompt token count is outside the stored sequence")
    position_kind, offset = _position_spec(position_name, prompt_tokens)

    positions = np.full(len(rows), -1, dtype=np.int64)
    valid_rows: list[int] = []
    for row_index, row in enumerate(rows):
        if position_kind == "code":
            code_start = row["code_token_start"]
            if code_start is None:
                continue
            position = prompt_tokens + int(code_start) + offset
        elif position_kind == "completion":
            position = prompt_tokens + offset
        else:
            position = offset
        if 0 <= position < mask.shape[1] and mask[row_index, position]:
            positions[row_index] = position
            valid_rows.append(row_index)
    return positions, np.asarray(valid_rows, dtype=np.int64)


def _position_spec(
    position_name: str, prompt_tokens: int
) -> tuple[Literal["fixed", "completion", "code"], int]:
    if position_name == "last_prompt":
        return "fixed", prompt_tokens - 1
    if position_name == "code_start_minus_1":
        return "code", -1
    if position_name == "code_start":
        return "code", 0
    if position_name.startswith("completion_"):
        suffix = position_name.removeprefix("completion_")
        if suffix not in {"0", "1", "2", "4", "8"}:
            raise ValueError(f"unsupported completion position {position_name!r}")
        return "completion", int(suffix)
    raise ValueError(f"unsupported read position {position_name!r}")


def normalize_for_projection(
    residuals: torch.Tensor, unembed: StoredUnembed | CPUUnembedding
) -> torch.Tensor:
    """Apply Gemma's final RMSNorm and scale, retaining the lm-head input rounding."""
    if residuals.ndim < TWO_DIMENSIONS or residuals.shape[-1] != unembed.norm_weight.numel():
        raise ValueError("residual width does not match the final norm")
    input_dtype = (
        unembed.lm_head_input_dtype
        if isinstance(unembed, CPUUnembedding)
        else unembed.lm_head_weight.dtype
    )
    values = residuals.to(device=unembed.lm_head_weight.device, dtype=input_dtype).float()
    normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + unembed.rms_eps)
    normalized = normalized * (1.0 + unembed.norm_weight.to(values.device).float())
    return normalized.to(input_dtype).float()


def _project_normalized(
    states: torch.Tensor, unembed: StoredUnembed | CPUUnembedding
) -> torch.Tensor:
    if states.ndim != TWO_DIMENSIONS or states.shape[1] != unembed.lm_head_weight.shape[1]:
        raise ValueError("normalized states do not match the lm-head width")
    lm_head_weight = (
        unembed.lm_head_weight
        if isinstance(unembed, CPUUnembedding)
        else unembed.lm_head_weight.float()
    )
    return states.float() @ lm_head_weight.T


def _load_sweep(sweep_dir: Path, output_key: str) -> Sweep:
    activation_path = sweep_dir / f"activations.{output_key}.npy"
    mask_path = sweep_dir / f"mask.{output_key}.npy"
    index_path = sweep_dir / f"index.{output_key}.json"
    activations = np.load(activation_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")
    index = cast("dict[str, Any]", json.loads(index_path.read_text(encoding="utf-8")))
    if index.get("complete") is not True:
        raise ValueError(f"token sweep {output_key} is incomplete")
    if activations.shape != (960, 7, 168, 3840) or activations.dtype != np.dtype(np.float32):
        raise ValueError(f"{activation_path} must be [960, 7, 168, 3840] float32")
    if mask.shape != (960, 168) or mask.dtype != np.dtype(np.bool_):
        raise ValueError(f"{mask_path} must be [960, 168] bool")
    layer_ids = tuple(int(layer) for layer in index["layer_ids"])
    if layer_ids != SWEEP_LAYERS:
        raise ValueError(f"sweep layers {layer_ids} do not match expected {SWEEP_LAYERS}")
    prompt_tokens = int(index["prompt_tokens"])
    completion_tokens = int(index["completion_tokens"])
    if (prompt_tokens, completion_tokens) != (PROMPT_TOKENS, COMPLETION_TOKENS):
        raise ValueError("sweep prompt/completion token layout differs from the requested readout")
    position_layout = cast("dict[str, Any]", index["position_layout"])
    if (
        position_layout.get("activation_alignment")
        != "hidden state after the token at that position"
    ):
        raise ValueError("sweep hidden-state alignment is not the expected post-token convention")
    rows = cast("list[dict[str, Any]]", index["rows"])
    sample_ids = tuple(str(sample_id) for sample_id in index["sample_ids"])
    row_sample_ids = tuple(str(row["sample_id"]) for row in rows)
    if len(rows) != activations.shape[0] or row_sample_ids != sample_ids:
        raise ValueError(f"{output_key} index rows do not align with activation rows")
    if any(
        not isinstance(row.get("hack_label"), bool) or not isinstance(row.get("pass_label"), bool)
        for row in rows
    ):
        raise TypeError(f"{output_key} requires boolean hack_label and pass_label fields")
    return Sweep(
        activations=activations,
        mask=mask,
        rows=rows,
        sample_ids=sample_ids,
        layer_ids=layer_ids,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        position_layout=position_layout,
    )


def _validate_paired_sweeps(adapter: Sweep, base: Sweep) -> None:
    if adapter.activations.shape != base.activations.shape or adapter.mask.shape != base.mask.shape:
        raise ValueError("adapter and base teacher-forced sweeps have different tensor shapes")
    if adapter.sample_ids != base.sample_ids:
        raise ValueError("adapter and base teacher-forced sweeps are not in identical sample order")
    if not np.array_equal(adapter.mask, base.mask):
        raise ValueError("adapter and base teacher-forced masks differ")
    if adapter.layer_ids != base.layer_ids:
        raise ValueError("adapter and base teacher-forced layer IDs differ")
    if adapter.position_layout != base.position_layout:
        raise ValueError("adapter and base teacher-forced token layouts differ")
    for adapter_row, base_row in zip(adapter.rows, base.rows, strict=True):
        for field in ("item_id", "sample_index", "hack_label", "pass_label", "code_token_start"):
            if adapter_row[field] != base_row[field]:
                raise ValueError(f"adapter and base teacher-forced rows differ in {field}")


def _validate_lens(lens_path: Path, provenance_path: Path) -> LoadedLens:
    provenance = cast("dict[str, Any]", json.loads(provenance_path.read_text(encoding="utf-8")))
    source_layers = tuple(int(layer) for layer in provenance["source_layers"])
    if source_layers != EXPECTED_LENS_SOURCES or int(provenance["target_layer"]) != TARGET_LAYER:
        raise ValueError(
            "lens provenance does not match the expected Gemma base lens source/target"
        )
    if provenance.get("adapter") is not None or provenance.get("model") != "google/gemma-3-12b-it":
        raise ValueError("the readout requires the base Gemma lens, without an adapter")
    loaded = load_lens(lens_path)
    if loaded.source_layers != source_layers or loaded.d_model != GEMMA_HIDDEN_SIZE:
        raise ValueError("lens tensor layers or width do not match its provenance")
    if not set(READ_LAYERS).issubset(loaded.source_layers):
        raise ValueError("one or more requested read layers are absent from the base lens")
    selected = LoadedLens(
        jacobians={layer: loaded.jacobians[layer] for layer in READ_LAYERS},
        n_prompts=loaded.n_prompts,
        d_model=loaded.d_model,
    )
    del loaded
    return selected


def _load_gemma_unembed(model_path: Path) -> StoredUnembed:
    model_config = cast("Any", AutoConfig.from_pretrained(model_path, local_files_only=True))
    text_config = model_config.get_text_config()
    if model_config.tie_word_embeddings is not True:
        raise ValueError("Gemma output projection is expected to share its token embedding weights")
    index_path = model_path / "model.safetensors.index.json"
    index = cast("dict[str, Any]", json.loads(index_path.read_text(encoding="utf-8")))
    weight_map = cast("dict[str, str]", index["weight_map"])
    norm_key = "language_model.model.norm.weight"
    embedding_key = "language_model.model.embed_tokens.weight"
    if (
        norm_key not in weight_map
        or embedding_key not in weight_map
        or "lm_head.weight" in weight_map
    ):
        raise ValueError(
            "cached Gemma safetensors do not contain the expected tied output embedding and final norm"
        )
    norm_path = model_path / weight_map[norm_key]
    embedding_path = model_path / weight_map[embedding_key]
    with safe_open(str(norm_path), framework="pt", device="cpu") as handle:
        norm_weight = handle.get_tensor(norm_key).contiguous()
    with safe_open(str(embedding_path), framework="pt", device="cpu") as handle:
        lm_head_weight = handle.get_tensor(embedding_key).contiguous()
    if (
        norm_weight.ndim != 1
        or lm_head_weight.ndim != TWO_DIMENSIONS
        or norm_weight.shape[0] != lm_head_weight.shape[1]
    ):
        raise ValueError("Gemma final norm and tied output embedding have incompatible shapes")
    return StoredUnembed(
        norm_weight=norm_weight,
        lm_head_weight=lm_head_weight,
        rms_eps=float(text_config.rms_norm_eps),
    )


def _stable_seed(seed: int, key: str) -> int:
    material = f"{seed}:{key}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], byteorder="big")


def _group_indices(item_ids: list[str]) -> tuple[np.ndarray, int]:
    item_to_index = {item_id: index for index, item_id in enumerate(dict.fromkeys(item_ids))}
    return np.asarray([item_to_index[item_id] for item_id in item_ids], dtype=np.int64), len(
        item_to_index
    )


def _class_arrays(rows: list[dict[str, Any]]) -> np.ndarray:
    labels = np.full(len(rows), -1, dtype=np.int8)
    for index, row in enumerate(rows):
        hacked = bool(row["hack_label"])
        passed = bool(row["pass_label"])
        if hacked and passed:
            raise ValueError("a sample cannot be both a hack and an honest pass")
        if hacked:
            labels[index] = 1
        elif passed:
            labels[index] = 0
    return labels


def _grouped_class_sums(
    states: np.ndarray, labels: np.ndarray, item_indices: np.ndarray, n_items: int
) -> tuple[np.ndarray, np.ndarray]:
    hidden_size = states.shape[1]
    sums = np.zeros((n_items, 2, hidden_size), dtype=np.float32)
    counts = np.zeros((n_items, 2), dtype=np.int64)
    for class_index in (0, 1):
        selected = np.flatnonzero(labels == class_index)
        np.add.at(sums[:, class_index], item_indices[selected], states[selected])
        np.add.at(counts[:, class_index], item_indices[selected], 1)
    return sums, counts


def _paired_half_states(
    item_sums: np.ndarray,
    item_counts: np.ndarray,
    *,
    n_bootstrap_halves: int,
    seed: int,
) -> torch.Tensor:
    draws = bootstrap_indices(
        n_observations=len(item_counts), n_resamples=n_bootstrap_halves, seed=seed
    )
    half_size = len(item_counts) // 2
    if half_size < 1:
        raise ValueError("at least two items are required for bootstrap halves")
    halves: list[np.ndarray] = []
    for draw in draws:
        for selected in (draw[:half_size], draw[half_size:]):
            count = int(item_counts[selected].sum())
            if count == 0:
                raise ValueError("a bootstrap half contains no valid paired samples")
            halves.append(item_sums[selected].sum(axis=0) / count)
    return torch.from_numpy(np.stack(halves))


def _class_half_states(
    class_sums: np.ndarray,
    class_counts: np.ndarray,
    *,
    n_bootstrap_halves: int,
    seed: int,
) -> torch.Tensor:
    draws = bootstrap_indices(
        n_observations=len(class_counts), n_resamples=n_bootstrap_halves, seed=seed
    )
    half_size = len(class_counts) // 2
    if half_size < 1:
        raise ValueError("at least two items are required for bootstrap halves")
    halves: list[np.ndarray] = []
    for draw in draws:
        draw_halves: list[np.ndarray] = []
        for selected in (draw[:half_size], draw[half_size:]):
            counts = class_counts[selected].sum(axis=0)
            if np.any(counts == 0):
                break
            sums = class_sums[selected].sum(axis=0)
            draw_halves.append(sums[1] / counts[1] - sums[0] / counts[0])
        if len(draw_halves) == BOOTSTRAP_HALVES_PER_DRAW:
            halves.extend(draw_halves)
    if len(halves) < MIN_BOOTSTRAP_PAIRS * 2:
        raise ValueError("too many bootstrap halves contain only one label class")
    return torch.from_numpy(np.stack(halves))


def _equal_item_half_states(
    item_deltas: np.ndarray, *, n_bootstrap_halves: int, seed: int
) -> torch.Tensor:
    draws = bootstrap_indices(
        n_observations=len(item_deltas), n_resamples=n_bootstrap_halves, seed=seed
    )
    half_size = len(item_deltas) // 2
    if half_size < 1:
        raise ValueError("at least two mixed items are required for bootstrap halves")
    halves = [
        item_deltas[selected].mean(axis=0)
        for draw in draws
        for selected in (draw[:half_size], draw[half_size:])
    ]
    return torch.from_numpy(np.stack(halves))


def _top_indices(delta: torch.Tensor, readable: torch.Tensor, sign: float) -> tuple[int, ...]:
    ranked = (delta * sign).masked_fill(~readable, float("-inf"))
    values, indices = ranked.topk(min(N_TOP_TOKENS, int(readable.sum())))
    return tuple(int(index) for value, index in zip(values, indices, strict=True) if value > 0)


def _jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _stability(
    half_logits: torch.Tensor, readable: torch.Tensor, sign: float
) -> dict[str, float | int]:
    if half_logits.shape[0] % 2:
        raise ValueError("bootstrap half logits must be paired")
    similarities = [
        _jaccard(
            _top_indices(half_logits[index], readable, sign),
            _top_indices(half_logits[index + 1], readable, sign),
        )
        for index in range(0, half_logits.shape[0], 2)
    ]
    return {
        "bootstrap_half_top50_jaccard_mean": float(np.mean(similarities)),
        "bootstrap_half_top50_jaccard_median": float(np.median(similarities)),
        "bootstrap_pairs": len(similarities),
    }


def _magnitude(delta: torch.Tensor, readable: torch.Tensor) -> dict[str, float]:
    values = delta[readable]
    absolute = values.abs()
    return {
        "mean_absolute_logit_delta": float(absolute.mean()),
        "rms_logit_delta": float(values.square().mean().sqrt()),
        "maximum_absolute_logit_delta": float(absolute.max()),
    }


def _token_rows(
    delta: torch.Tensor,
    half_logits: torch.Tensor,
    vocabulary: list[str],
    readable: torch.Tensor,
    *,
    sign: float,
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    selected = _top_indices(delta, readable, sign)
    positive_fractions = (half_logits * sign > 0).float().mean(dim=0)
    tokens = [
        {
            "token": vocabulary[token_id],
            "mean_logit_delta": float(delta[token_id]),
            "bootstrap_positive_fraction": float(positive_fractions[token_id]),
        }
        for token_id in selected
    ]
    return tokens, selected


def _contrast_report(
    point_state_delta: torch.Tensor,
    half_state_deltas: torch.Tensor,
    config: CellAnalysisConfig,
    *,
    include_fallen: bool,
) -> ContrastAnalysis:
    point_delta = _project_normalized(point_state_delta.unsqueeze(0), config.unembed).squeeze(0)
    half_logits = _project_normalized(half_state_deltas, config.unembed)
    risen, risen_ids = _token_rows(
        point_delta, half_logits, config.vocabulary, config.readable_mask, sign=1.0
    )
    report: dict[str, Any] = {
        "risen": risen,
        "stability": {"risen": _stability(half_logits, config.readable_mask, 1.0)},
        "magnitude": _magnitude(point_delta, config.readable_mask),
    }
    fallen_ids: tuple[int, ...] = ()
    if include_fallen:
        fallen, fallen_ids = _token_rows(
            point_delta, half_logits, config.vocabulary, config.readable_mask, sign=-1.0
        )
        report["fallen"] = fallen
        report["stability"]["fallen"] = _stability(half_logits, config.readable_mask, -1.0)
    return ContrastAnalysis(
        report=report,
        risen_ids=risen_ids,
        fallen_ids=fallen_ids,
    )


def _cell_seed(config: CellAnalysisConfig, suffix: str) -> int:
    return _stable_seed(config.seed, f"{config.layer}:{config.position}:{suffix}")


def _mean_class_effect(
    class_sums: np.ndarray, class_counts: np.ndarray
) -> tuple[np.ndarray, int, int]:
    sums = class_sums.sum(axis=0)
    counts = class_counts.sum(axis=0)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError("hack and honest-pass contrasts both require samples")
    return sums[1] / counts[1] - sums[0] / counts[0], int(counts[1]), int(counts[0])


def _within_item_effects(
    class_sums: np.ndarray, class_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mixed = np.flatnonzero((class_counts[:, 0] > 0) & (class_counts[:, 1] > 0))
    if len(mixed) < MIN_SAMPLES:
        raise ValueError("within-item contrast requires at least two mixed items")
    deltas = (
        class_sums[mixed, 1] / class_counts[mixed, 1, None]
        - class_sums[mixed, 0] / class_counts[mixed, 0, None]
    )
    return mixed, deltas


def _shuffle_labels(
    labels: np.ndarray,
    item_indices: np.ndarray,
    n_items: int,
    *,
    seed: int,
    within_item: bool,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    shuffled = labels.copy()
    if within_item:
        for item_index in range(n_items):
            row_indices = np.flatnonzero((item_indices == item_index) & (labels >= 0))
            shuffled[row_indices] = generator.permutation(labels[row_indices])
    else:
        labeled_rows = np.flatnonzero(labels >= 0)
        shuffled[labeled_rows] = generator.permutation(labels[labeled_rows])
    return shuffled


def _contrast_with_overlap(
    placebo: ContrastAnalysis, real: ContrastAnalysis, *, include_fallen: bool
) -> dict[str, Any]:
    placebo_report = dict(placebo.report)
    placebo_report["overlap_with_real"] = {
        "risen_intersection": len(set(placebo.risen_ids) & set(real.risen_ids)),
        "risen_jaccard": _jaccard(placebo.risen_ids, real.risen_ids),
    }
    if include_fallen:
        placebo_report["overlap_with_real"].update(
            {
                "fallen_intersection": len(set(placebo.fallen_ids) & set(real.fallen_ids)),
                "fallen_jaccard": _jaccard(placebo.fallen_ids, real.fallen_ids),
            }
        )
    real_mean_abs = real.report["magnitude"]["mean_absolute_logit_delta"]
    placebo_report["magnitude"]["mean_absolute_ratio_to_real"] = (
        None
        if real_mean_abs == 0
        else placebo.report["magnitude"]["mean_absolute_logit_delta"] / real_mean_abs
    )
    return placebo_report


def _normalize_cell_inputs(
    adapter_residuals: np.ndarray,
    base_residuals: np.ndarray,
    rows: list[dict[str, Any]],
    config: CellAnalysisConfig,
) -> NormalizedCell:
    if adapter_residuals.shape != base_residuals.shape or adapter_residuals.ndim != TWO_DIMENSIONS:
        raise ValueError("paired residual arrays must have matching [samples, hidden] shapes")
    if adapter_residuals.shape[0] != len(rows) or len(rows) < MIN_SAMPLES:
        raise ValueError("cell residual rows must align and contain at least two samples")
    if config.n_bootstrap_halves < MIN_BOOTSTRAP_PAIRS or config.batch_size < 1:
        raise ValueError("bootstrap pairs and batch size must be positive")
    if config.layer not in config.lens.jacobians:
        raise KeyError(f"read layer {config.layer} is absent from the base lens")
    if config.readable_mask.shape != (config.unembed.lm_head_weight.shape[0],) or len(
        config.vocabulary
    ) != len(config.readable_mask):
        raise ValueError("readable vocabulary mask does not match the lm-head rows")
    if not bool(config.readable_mask.any()):
        raise ValueError("vocabulary contains no readable single-token entries")

    labels = _class_arrays(rows)
    item_ids = [str(row["item_id"]) for row in rows]
    item_indices, n_items = _group_indices(item_ids)
    normalized_adapter = np.empty_like(adapter_residuals, dtype=np.float32)
    normalized_base = np.empty_like(base_residuals, dtype=np.float32)
    normalized_placebo_delta = np.empty_like(base_residuals, dtype=np.float32)
    generator = torch.Generator(device="cpu").manual_seed(_cell_seed(config, "norm"))

    for start in range(0, len(rows), config.batch_size):
        end = min(start + config.batch_size, len(rows))
        adapter_batch_np = np.ascontiguousarray(adapter_residuals[start:end])
        base_batch_np = np.ascontiguousarray(base_residuals[start:end])
        if not np.isfinite(adapter_batch_np).all() or not np.isfinite(base_batch_np).all():
            raise ValueError("selected activations contain non-finite values")
        adapter_batch = torch.from_numpy(adapter_batch_np)
        base_batch = torch.from_numpy(base_batch_np)
        adapter_transported = config.lens.transport(adapter_batch, config.layer)
        base_transported = config.lens.transport(base_batch, config.layer)
        adapter_normed = normalize_for_projection(adapter_transported, config.unembed)
        base_normed = normalize_for_projection(base_transported, config.unembed)
        difference = adapter_transported - base_transported
        random_direction = torch.randn(
            difference.shape, dtype=torch.float32, generator=generator, device="cpu"
        )
        random_norm = random_direction.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        matched_perturbation = random_direction * (
            difference.norm(dim=-1, keepdim=True) / random_norm
        )
        placebo_normed = normalize_for_projection(
            base_transported + matched_perturbation, config.unembed
        )
        normalized_adapter[start:end] = adapter_normed.numpy()
        normalized_base[start:end] = base_normed.numpy()
        normalized_placebo_delta[start:end] = (placebo_normed - base_normed).numpy()

    return NormalizedCell(
        adapter=normalized_adapter,
        base=normalized_base,
        placebo_delta=normalized_placebo_delta,
        labels=labels,
        item_indices=item_indices,
        n_items=n_items,
    )


def _paired_read(data: NormalizedCell, config: CellAnalysisConfig) -> PairedRead:
    readable_delta = data.adapter - data.base
    item_sums = np.zeros((data.n_items, readable_delta.shape[1]), dtype=np.float32)
    item_counts = np.zeros(data.n_items, dtype=np.int64)
    np.add.at(item_sums, data.item_indices, readable_delta)
    np.add.at(item_counts, data.item_indices, 1)
    report = _contrast_report(
        torch.from_numpy(readable_delta.mean(axis=0)),
        _paired_half_states(
            item_sums,
            item_counts,
            n_bootstrap_halves=config.n_bootstrap_halves,
            seed=_cell_seed(config, "paired"),
        ),
        config,
        include_fallen=True,
    )
    return PairedRead(contrast=report, item_sums=item_sums, item_counts=item_counts)


def _class_contrast(
    class_sums: np.ndarray,
    class_counts: np.ndarray,
    config: CellAnalysisConfig,
    seed_suffix: str,
) -> ContrastAnalysis:
    point_delta, _, _ = _mean_class_effect(class_sums, class_counts)
    half_deltas = _class_half_states(
        class_sums,
        class_counts,
        n_bootstrap_halves=config.n_bootstrap_halves,
        seed=_cell_seed(config, seed_suffix),
    )
    return _contrast_report(
        torch.from_numpy(point_delta), half_deltas, config, include_fallen=False
    )


def _within_class_contrast(
    class_sums: np.ndarray,
    class_counts: np.ndarray,
    config: CellAnalysisConfig,
    seed_suffix: str,
) -> tuple[np.ndarray, ContrastAnalysis]:
    mixed_indices, item_deltas = _within_item_effects(class_sums, class_counts)
    half_deltas = _equal_item_half_states(
        item_deltas,
        n_bootstrap_halves=config.n_bootstrap_halves,
        seed=_cell_seed(config, seed_suffix),
    )
    report = _contrast_report(
        torch.from_numpy(item_deltas.mean(axis=0)),
        half_deltas,
        config,
        include_fallen=False,
    )
    return mixed_indices, report


def _hack_reads(data: NormalizedCell, config: CellAnalysisConfig) -> HackReads:
    adapter_sums, class_counts = _grouped_class_sums(
        data.adapter, data.labels, data.item_indices, data.n_items
    )
    base_sums, base_counts = _grouped_class_sums(
        data.base, data.labels, data.item_indices, data.n_items
    )
    if not np.array_equal(class_counts, base_counts):
        raise ValueError("adapter and base label counts differ")
    _, hack_samples, honest_samples = _mean_class_effect(adapter_sums, class_counts)
    adapter_all = _class_contrast(adapter_sums, class_counts, config, "adapter-all")
    base_all = _class_contrast(base_sums, class_counts, config, "base-all")
    adapter_mixed, adapter_within = _within_class_contrast(
        adapter_sums, class_counts, config, "adapter-within"
    )
    base_mixed, base_within = _within_class_contrast(base_sums, base_counts, config, "base-within")
    if not np.array_equal(adapter_mixed, base_mixed):
        raise ValueError("adapter and base teacher-forced mixed-item sets differ")
    return HackReads(
        adapter_all=adapter_all,
        base_all=base_all,
        adapter_within=adapter_within,
        base_within=base_within,
        mixed_item_indices=adapter_mixed,
        hack_samples=hack_samples,
        honest_samples=honest_samples,
        adapter_class_sums=adapter_sums,
        base_class_sums=base_sums,
        class_counts=class_counts,
    )


def _matched_placebo_read(
    data: NormalizedCell, paired: PairedRead, config: CellAnalysisConfig
) -> dict[str, Any]:
    item_sums = np.zeros_like(paired.item_sums)
    np.add.at(item_sums, data.item_indices, data.placebo_delta)
    placebo = _contrast_report(
        torch.from_numpy(data.placebo_delta.mean(axis=0)),
        _paired_half_states(
            item_sums,
            paired.item_counts,
            n_bootstrap_halves=config.n_bootstrap_halves,
            seed=_cell_seed(config, "matched-placebo"),
        ),
        config,
        include_fallen=True,
    )
    return _contrast_with_overlap(placebo, paired.contrast, include_fallen=True)


def _shuffled_label_reports(
    data: NormalizedCell, hack: HackReads, config: CellAnalysisConfig
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for shuffle_name, seed_suffix, within_item in (
        ("all_samples", "shuffle-all", False),
        ("within_items", "shuffle-within", True),
    ):
        shuffled_labels = _shuffle_labels(
            data.labels,
            data.item_indices,
            data.n_items,
            seed=_cell_seed(config, seed_suffix),
            within_item=within_item,
        )
        adapter_sums, shuffled_counts = _grouped_class_sums(
            data.adapter, shuffled_labels, data.item_indices, data.n_items
        )
        base_sums, base_counts = _grouped_class_sums(
            data.base, shuffled_labels, data.item_indices, data.n_items
        )
        if not np.array_equal(shuffled_counts, base_counts):
            raise ValueError("shuffled adapter and base label counts differ")
        adapter_all = _class_contrast(
            adapter_sums, shuffled_counts, config, f"{shuffle_name}:adapter-all"
        )
        base_all = _class_contrast(base_sums, shuffled_counts, config, f"{shuffle_name}:base-all")
        adapter_mixed, adapter_within = _within_class_contrast(
            adapter_sums, shuffled_counts, config, f"{shuffle_name}:adapter-within"
        )
        base_mixed, base_within = _within_class_contrast(
            base_sums, shuffled_counts, config, f"{shuffle_name}:base-within"
        )
        if not np.array_equal(adapter_mixed, base_mixed):
            raise ValueError("shuffled adapter and base mixed-item sets differ")
        reports[shuffle_name] = {
            "adapter": {
                "all_samples": _contrast_with_overlap(
                    adapter_all, hack.adapter_all, include_fallen=False
                ),
                "within_mixed_items": _contrast_with_overlap(
                    adapter_within, hack.adapter_within, include_fallen=False
                ),
            },
            "base_same_tokens": {
                "all_samples": _contrast_with_overlap(
                    base_all, hack.base_all, include_fallen=False
                ),
                "within_mixed_items": _contrast_with_overlap(
                    base_within, hack.base_within, include_fallen=False
                ),
            },
        }
    return reports


def analyze_cell(
    adapter_residuals: np.ndarray,
    base_residuals: np.ndarray,
    rows: list[dict[str, Any]],
    config: CellAnalysisConfig,
) -> dict[str, Any]:
    """Compute paired, hack-label, and placebo reads for one layer-position cell."""
    config = CellAnalysisConfig(
        layer=config.layer,
        position=config.position,
        lens=config.lens,
        unembed=config.unembed,
        vocabulary=config.vocabulary,
        readable_mask=config.readable_mask.cpu(),
        n_bootstrap_halves=config.n_bootstrap_halves,
        seed=config.seed,
        batch_size=config.batch_size,
    )
    data = _normalize_cell_inputs(adapter_residuals, base_residuals, rows, config)
    paired = _paired_read(data, config)
    hack = _hack_reads(data, config)
    return {
        "layer": config.layer,
        "position": config.position,
        "n_samples": len(rows),
        "n_items": data.n_items,
        "n_hack_samples": hack.hack_samples,
        "n_honest_pass_samples": hack.honest_samples,
        "n_mixed_items": len(hack.mixed_item_indices),
        "rl_minus_base": paired.contrast.report,
        "hack_minus_honest": {
            "all_samples": {
                "adapter": hack.adapter_all.report,
                "base_same_tokens": hack.base_all.report,
            },
            "within_mixed_items": {
                "n_items": len(hack.mixed_item_indices),
                "adapter": hack.adapter_within.report,
                "base_same_tokens": hack.base_within.report,
            },
        },
        "placebo": {
            "matched_norm_random_perturbation": _matched_placebo_read(data, paired, config),
            "shuffled_labels": _shuffled_label_reports(data, hack, config),
        },
    }


def _readable_mask(vocabulary: list[str]) -> torch.Tensor:
    return torch.tensor([is_word_like(token) for token in vocabulary], dtype=torch.bool)


def _project_cell_inputs(
    sweep: Sweep,
    position_indices_by_row: np.ndarray,
    valid_rows: np.ndarray,
    layer_index: int,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((len(valid_rows), sweep.activations.shape[-1]), dtype=np.float32)
    for start in range(0, len(valid_rows), batch_size):
        selected_rows = valid_rows[start : start + batch_size]
        selected_positions = position_indices_by_row[selected_rows]
        result[start : start + len(selected_rows)] = sweep.activations[
            selected_rows, layer_index, selected_positions, :
        ]
    return result


def _mean_bootstrap_stability(result: dict[str, Any], *, path: tuple[str, ...]) -> float:
    value = result
    for part in path:
        value = value[part]
    return float(value["stability"]["risen"]["bootstrap_half_top50_jaccard_mean"])


def _stable_cells(
    cells: dict[str, dict[str, Any]], path: tuple[str, ...], count: int = 3
) -> list[tuple[str, float]]:
    ranked = [(key, _mean_bootstrap_stability(cell, path=path)) for key, cell in cells.items()]
    return sorted(ranked, key=lambda row: (-row[1], row[0]))[:count]


def _markdown_report(report: dict[str, Any]) -> str:
    cells = cast("dict[str, dict[str, Any]]", report["cells"])
    lines = [
        "# Atlas Gemma Jacobian-lens workspace readout",
        "",
        "The same base Gemma lens reads the adapter and Gemma teacher-forced on the adapter's exact tokens.",
        "The lm-head input uses Gemma's final RMSNorm with the `(1 + weight)` scale.",
        "",
        f"Samples: {report['metadata']['n_samples']}; items: {report['metadata']['n_items']};",
        (
            f"Mixed hack/pass items: {report['metadata']['n_mixed_items']}; "
            f"bootstrap half-pairs per cell: {report['metadata']['bootstrap_half_pairs']}."
        ),
        "",
        "## Most stable adapter-minus-base cells",
        "",
    ]
    for cell_key, stability in _stable_cells(cells, ("rl_minus_base",)):
        tokens = cells[cell_key]["rl_minus_base"]["risen"][:10]
        lines.append(f"### {cell_key} (top-50 Jaccard {stability:.3f})")
        lines.append("")
        lines.append(
            ", ".join(
                f"{token['token'].strip()} ({token['mean_logit_delta']:+.3f})" for token in tokens
            )
        )
        lines.append("")
        placebo = cells[cell_key]["placebo"]["matched_norm_random_perturbation"]
        lines.append(
            "Matched-norm placebo: "
            f"risen overlap {placebo['overlap_with_real']['risen_intersection']}/50 "
            f"(Jaccard {placebo['overlap_with_real']['risen_jaccard']:.3f}); "
            f"mean-absolute-logit ratio {placebo['magnitude']['mean_absolute_ratio_to_real']:.3f}."
        )
        lines.append("")
    for read_name, path in (
        ("All-sample adapter hack-minus-honest", ("hack_minus_honest", "all_samples", "adapter")),
        (
            "Within-mixed-item adapter hack-minus-honest",
            ("hack_minus_honest", "within_mixed_items", "adapter"),
        ),
    ):
        lines.extend([f"## Most stable {read_name} cells", ""])
        for cell_key, stability in _stable_cells(cells, path):
            readout = cells[cell_key]
            value = readout
            for part in path:
                value = value[part]
            tokens = value["risen"][:10]
            lines.append(f"### {cell_key} (top-50 Jaccard {stability:.3f})")
            lines.append("")
            lines.append(
                ", ".join(
                    f"{token['token'].strip()} ({token['mean_logit_delta']:+.3f})"
                    for token in tokens
                )
            )
            lines.append("")
            base_value = readout
            for part in path[:-1]:
                base_value = base_value[part]
            base_tokens = base_value["base_same_tokens"]["risen"][:5]
            shuffled = readout["placebo"]["shuffled_labels"]
            shuffle_key = "all_samples" if "all_samples" in path else "within_items"
            placebo = shuffled[shuffle_key]["adapter"][
                "all_samples" if shuffle_key == "all_samples" else "within_mixed_items"
            ]
            base_label = (
                ", ".join(token["token"].strip() for token in base_tokens) or "(no positive token)"
            )
            lines.append(f"Base teacher-forced top tokens: {base_label}.")
            lines.append(
                "Shuffled-label placebo: "
                f"adapter risen overlap {placebo['overlap_with_real']['risen_intersection']}/50 "
                f"(Jaccard {placebo['overlap_with_real']['risen_jaccard']:.3f}); "
                f"mean-absolute-logit ratio {placebo['magnitude']['mean_absolute_ratio_to_real']:.3f}."
            )
            lines.append("")
    lines.extend(
        [
            "## Cell stability",
            "",
            "| Layer | Position | Adapter minus base | All-sample adapter | Within-item adapter |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for cell in cells.values():
        adapter_all = cell["hack_minus_honest"]["all_samples"]["adapter"]
        adapter_within = cell["hack_minus_honest"]["within_mixed_items"]["adapter"]
        lines.append(
            f"| {cell['layer']} | {cell['position']} | "
            f"{cell['rl_minus_base']['stability']['risen']['bootstrap_half_top50_jaccard_mean']:.3f} / "
            f"{cell['rl_minus_base']['stability']['fallen']['bootstrap_half_top50_jaccard_mean']:.3f} | "
            f"{adapter_all['stability']['risen']['bootstrap_half_top50_jaccard_mean']:.3f} | "
            f"{adapter_within['stability']['risen']['bootstrap_half_top50_jaccard_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "The JSON file contains the full top-50 lists for every cell. Bootstrap Jaccard compares the "
                "risen and fallen top-50 lists across paired halves of item-level bootstrap draws."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> None:
    """Run the Atlas workspace readout and write its JSON and Markdown reports."""
    start_time = time.monotonic()
    sweep_dir = args.run_dir / "token-sweep"
    adapter = _load_sweep(sweep_dir, ADAPTER_KEY)
    base = _load_sweep(sweep_dir, BASE_READ_KEY)
    _validate_paired_sweeps(adapter, base)
    provenance_path = args.lens_provenance or args.lens.with_name("lens-n30-t44.provenance.json")
    lens = _validate_lens(args.lens, provenance_path)
    if lens.d_model != adapter.activations.shape[-1]:
        raise ValueError("lens width and sweep activation width differ")

    raw_unembed = _load_gemma_unembed(args.model_path)
    unembed = CPUUnembedding(
        norm_weight=raw_unembed.norm_weight.float(),
        lm_head_weight=raw_unembed.lm_head_weight.float(),
        lm_head_input_dtype=raw_unembed.lm_head_weight.dtype,
        rms_eps=raw_unembed.rms_eps,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    vocabulary = decode_vocab(tokenizer, int(unembed.lm_head_weight.shape[0]))
    readable = _readable_mask(vocabulary)
    del raw_unembed

    cells: dict[str, dict[str, Any]] = {}
    positions = (
        "last_prompt",
        "code_start_minus_1",
        "code_start",
        "completion_0",
        "completion_1",
        "completion_2",
        "completion_4",
        "completion_8",
    )
    for layer in READ_LAYERS:
        layer_index = adapter.layer_ids.index(layer)
        if base.layer_ids.index(layer) != layer_index:
            raise ValueError(f"layer {layer} has different index conventions across paired sweeps")
        for position in positions:
            absolute_positions, valid_rows = position_indices(
                adapter.rows, adapter.mask, position, prompt_tokens=adapter.prompt_tokens
            )
            if not len(valid_rows):
                logger.warning("no valid rows for layer=%d position=%s", layer, position)
                continue
            paired_rows = [adapter.rows[int(row)] for row in valid_rows]
            adapter_vectors = _project_cell_inputs(
                adapter, absolute_positions, valid_rows, layer_index, args.batch_size
            )
            base_vectors = _project_cell_inputs(
                base, absolute_positions, valid_rows, layer_index, args.batch_size
            )
            cell = analyze_cell(
                adapter_vectors,
                base_vectors,
                paired_rows,
                CellAnalysisConfig(
                    layer=layer,
                    position=position,
                    lens=lens,
                    unembed=unembed,
                    vocabulary=vocabulary,
                    readable_mask=readable,
                    n_bootstrap_halves=args.bootstrap_halves,
                    seed=args.seed,
                    batch_size=args.batch_size,
                ),
            )
            cells[f"L{layer}/{position}"] = cell
            logger.info(
                "analysed layer=%d position=%s valid_samples=%d", layer, position, len(valid_rows)
            )

    item_ids = {str(row["item_id"]) for row in adapter.rows}
    labels = _class_arrays(adapter.rows)
    item_indices, n_items = _group_indices([str(row["item_id"]) for row in adapter.rows])
    class_counts = np.zeros((n_items, 2), dtype=np.int64)
    np.add.at(class_counts[:, 0], item_indices[labels == 0], 1)
    np.add.at(class_counts[:, 1], item_indices[labels == 1], 1)
    mixed_items = int(((class_counts[:, 0] > 0) & (class_counts[:, 1] > 0)).sum())
    report: dict[str, Any] = {
        "metadata": {
            "model": "google/gemma-3-12b-it",
            "adapter": ADAPTER_KEY,
            "lens": args.lens.name,
            "lens_target_layer": TARGET_LAYER,
            "lens_source_layers": list(EXPECTED_LENS_SOURCES),
            "read_layers": list(READ_LAYERS),
            "sweep_layer_ids": list(adapter.layer_ids),
            "sweep_layer_convention": (
                "layer id N stores hidden_states[N + 1], the residual after transformer block N"
            ),
            "activation_alignment": adapter.position_layout["activation_alignment"],
            "position_convention": (
                "last_prompt is stored index 7; completion offsets start at index 8; "
                "code_token_start is completion-relative"
            ),
            "norm_convention": "Gemma RMSNorm followed by multiplication by (1 + norm_weight)",
            "rms_norm_epsilon": unembed.rms_eps,
            "output_projection": "tied language token embedding matrix",
            "n_samples": len(adapter.rows),
            "n_items": len(item_ids),
            "n_mixed_items": mixed_items,
            "n_hack_samples": int((labels == 1).sum()),
            "n_honest_pass_samples": int((labels == 0).sum()),
            "bootstrap_half_pairs": args.bootstrap_halves,
            "bootstrap_seed": args.seed,
            "no_item_text_or_sample_ids_written": True,
            "teacher_forced_base_reads_same_tokens": True,
            "matched_norm_perturbation": (
                "per-sample random direction matched to the norm of "
                "adapter-minus-base transported residual"
            ),
            "shuffled_label_controls": (
                "global permutation over labeled samples and separate within-item permutations "
                "preserving each item's class counts"
            ),
        },
        "cells": cells,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "workspace-readout.json"
    markdown_path = args.output_dir / "workspace-readout.md"
    write_json(json_path, report)
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    logger.info(
        "wrote %s and %s in %.1f seconds",
        json_path,
        markdown_path,
        time.monotonic() - start_time,
    )


def _default_model_path() -> Path:
    snapshots_dir = MODEL_CACHE / "snapshots"
    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if len(snapshots) != 1:
        raise ValueError(
            f"expected one cached Gemma snapshot under {snapshots_dir}, found {len(snapshots)}"
        )
    return snapshots[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--lens", type=Path, default=LENS_PATH)
    parser.add_argument("--lens-provenance", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bootstrap-halves", type=int, default=DEFAULT_BOOTSTRAP_HALVES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    """Parse CLI arguments and run the CPU readout."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args()
    if args.bootstrap_halves < MIN_BOOTSTRAP_PAIRS or args.batch_size < 1 or args.threads < 1:
        raise ValueError("bootstrap-halves must be >=2, batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    args.model_path = args.model_path or _default_model_path()
    analyze(args)


if __name__ == "__main__":
    main()
