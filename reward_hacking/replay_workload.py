"""Prepare exact-token replay workloads and a simplified decode bandwidth roofline."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from reward_hacking.evidence_store import read_jsonl

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_LENGTHS = (4096, 16384, 32768, 65536)
WorkloadKind = Literal["trace", "synthetic"]


@dataclass(frozen=True, slots=True)
class Workload:
    """One exact prompt and its source sampling metadata for replay."""

    prompt_ids: list[int]
    sampling: dict[str, Any]
    source: str
    target_length: int
    kind: WorkloadKind


_SelectionCandidate = tuple[tuple[float, int, str], Workload, tuple[int, ...]]


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_request(lengths: tuple[int, ...], requests_per_cell: int, tolerance: float) -> None:
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive integers")
    if len(set(lengths)) != len(lengths):
        raise ValueError("lengths must be distinct")
    _positive_integer(requests_per_cell, "requests_per_cell")
    if not math.isfinite(tolerance) or tolerance < 0 or tolerance > 1:
        raise ValueError("tolerance must be between 0 and 1")


def _trace_sampling(rows: list[dict[str, Any]], trace_path: Path) -> dict[str, Any]:
    identity_rows = [row for row in rows if row.get("record") == "organism_episode_identity"]
    if len(identity_rows) != 1:
        raise ValueError(f"{trace_path} must contain exactly one organism_episode_identity record")
    sampling = identity_rows[0].get("sampling")
    if not isinstance(sampling, dict):
        raise TypeError(f"{trace_path} organism_episode_identity sampling must be an object")
    return cast("dict[str, Any]", sampling)


def _exact_prompt_ids(row: dict[str, Any], trace_path: Path, row_index: int) -> list[int] | None:
    prompt_ids = row.get("prompt_token_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        return None
    if any(type(token_id) is not int or token_id < 0 for token_id in prompt_ids):
        raise ValueError(f"{trace_path} turn row {row_index} has invalid prompt token IDs")
    prompt_tokens = row.get("prompt_tokens")
    if prompt_tokens is not None and (
        type(prompt_tokens) is not int or prompt_tokens != len(prompt_ids)
    ):
        raise ValueError(f"{trace_path} turn row {row_index} prompt token count disagrees")
    return cast("list[int]", prompt_ids)


@dataclass(slots=True)
class _WorkloadSelection:
    root: Path
    lengths: tuple[int, ...]
    requests_per_cell: int
    tolerance: float
    cells: dict[int, list[_SelectionCandidate]] = field(init=False)

    def __post_init__(self) -> None:
        self.cells = {length: [] for length in self.lengths}

    def collect_trace(self, trace_path: Path) -> None:
        rows = read_jsonl(trace_path)
        sampling = _trace_sampling(rows, trace_path)
        relative_path = trace_path.relative_to(self.root).as_posix()
        for row_index, row in enumerate(rows):
            if row.get("record") != "turn":
                continue
            prompt_ids = _exact_prompt_ids(row, trace_path, row_index)
            if prompt_ids is not None:
                self.consider_prompt(
                    prompt_ids,
                    sampling,
                    f"{relative_path}#row-{row_index}",
                )

    def consider_prompt(self, prompt_ids: list[int], sampling: dict[str, Any], source: str) -> None:
        actual_length = len(prompt_ids)
        prompt_key = tuple(prompt_ids)
        for target_length in self.lengths:
            relative_distance = abs(actual_length - target_length) / target_length
            if relative_distance > self.tolerance:
                continue
            cell = self.cells[target_length]
            if any(existing_prompt == prompt_key for _, _, existing_prompt in cell):
                continue
            workload = Workload(
                prompt_ids=list(prompt_ids),
                sampling=copy.deepcopy(sampling),
                source=source,
                target_length=target_length,
                kind="trace",
            )
            sort_key = (abs(actual_length - target_length), actual_length, source)
            cell.append((sort_key, workload, prompt_key))
            cell.sort(key=lambda candidate: candidate[0])
            # Keep only nearest prompts so all long token arrays do not accumulate.
            del cell[self.requests_per_cell :]

    def finish(self) -> list[Workload]:
        workloads: list[Workload] = []
        for target_length in self.lengths:
            cell = self.cells[target_length]
            if len(cell) < self.requests_per_cell:
                raise ValueError(
                    f"insufficient trace prompts for target length {target_length}: found "
                    f"{len(cell)} distinct prompts within ±{self.tolerance:.0%}, "
                    f"need {self.requests_per_cell}"
                )
            workloads.extend(workload for _, workload, _ in cell)
        return workloads


def extract_workloads(
    root: Path,
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
    requests_per_cell: int = 6,
    tolerance: float = 0.25,
) -> list[Workload]:
    """Select the nearest distinct exact prompt IDs from compact-aware traces.

    Sampling parameters are copied from each trace's ``organism_episode_identity`` row. The
    selected prompts are never decoded, rendered, truncated, or otherwise modified.
    """
    _validate_request(lengths, requests_per_cell, tolerance)
    if not root.is_dir():
        raise FileNotFoundError(f"trace root is not a directory: {root}")

    selection = _WorkloadSelection(root, lengths, requests_per_cell, tolerance)
    trace_paths = [
        path
        for path in sorted(root.glob("*/*/27b/traces/**/*.jsonl"))
        if not path.name.endswith(".docker-evidence.jsonl")
    ]
    for trace_path in trace_paths:
        selection.collect_trace(trace_path)
    return selection.finish()


def synthetic_workloads(
    vocab_size: int,
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
    requests_per_cell: int = 6,
    *,
    sampling: dict[str, Any],
    special_token_ids: tuple[int, ...] = (),
) -> list[Workload]:
    """Build deterministic exact-length token-ID controls without loading a tokenizer."""
    _validate_request(lengths, requests_per_cell, 0.0)
    _positive_integer(vocab_size, "vocab_size")
    if any(
        type(token_id) is not int or token_id < 0 or token_id >= vocab_size
        for token_id in special_token_ids
    ):
        raise ValueError("special_token_ids must be valid IDs within the vocabulary")
    excluded = set(special_token_ids)
    available_ids = [token_id for token_id in range(vocab_size) if token_id not in excluded]
    if len(available_ids) < requests_per_cell:
        raise ValueError(f"vocabulary must contain at least {requests_per_cell} usable token IDs")

    workloads: list[Workload] = []
    for target_length in lengths:
        for prefix_index in range(requests_per_cell):
            prompt_ids = [
                available_ids[(prefix_index + position) % len(available_ids)]
                for position in range(target_length)
            ]
            workloads.append(
                Workload(
                    prompt_ids=prompt_ids,
                    sampling=copy.deepcopy(sampling),
                    source=f"synthetic:{target_length}:{prefix_index}",
                    target_length=target_length,
                    kind="synthetic",
                )
            )
    return workloads


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise TypeError("model config text_config must be an object")
    return cast("dict[str, Any]", text_config)


def _config_integer(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"model config {key} must be a positive integer")
    return value


def roofline_tokens_per_second(  # noqa: PLR0913, PLR0917 - expose each roofline assumption directly
    config: dict[str, Any],
    context_tokens: int,
    concurrency: int,
    kv_bytes: int = 1,
    weight_bytes: float = 14e9,
    bandwidth_bytes_s: float = 1792e9,
) -> float:
    """Estimate aggregate decode tokens/s from weight and full-attention KV traffic.

    This simplified roof assumes one model-weight read per decode step and one KV read for
    every context token in every active sequence. It omits activations, dequantization, and
    launch overhead.
    """
    context_tokens = _positive_integer(context_tokens, "context_tokens")
    concurrency = _positive_integer(concurrency, "concurrency")
    kv_bytes = _positive_integer(kv_bytes, "kv_bytes")
    if not math.isfinite(weight_bytes) or weight_bytes <= 0:
        raise ValueError("weight_bytes must be finite and positive")
    if not math.isfinite(bandwidth_bytes_s) or bandwidth_bytes_s <= 0:
        raise ValueError("bandwidth_bytes_s must be finite and positive")

    text_config = _text_config(config)
    num_key_value_heads = _config_integer(text_config, "num_key_value_heads")
    num_hidden_layers = _config_integer(text_config, "num_hidden_layers")
    if "head_dim" in text_config:
        head_dim = _config_integer(text_config, "head_dim")
    else:
        hidden_size = _config_integer(text_config, "hidden_size")
        num_attention_heads = _config_integer(text_config, "num_attention_heads")
        if hidden_size % num_attention_heads:
            raise ValueError("model config hidden_size must be divisible by num_attention_heads")
        head_dim = hidden_size // num_attention_heads

    layer_types = text_config.get("layer_types")
    if layer_types is None:
        if "full_attention_interval" in text_config:
            full_attention_interval = _config_integer(text_config, "full_attention_interval")
            full_attention_layers = num_hidden_layers // full_attention_interval
        else:
            full_attention_layers = num_hidden_layers
    else:
        if not isinstance(layer_types, list) or len(layer_types) != num_hidden_layers:
            raise ValueError("model config layer_types must match num_hidden_layers")
        if any(not isinstance(layer_type, str) for layer_type in layer_types):
            raise ValueError("model config layer_types must contain strings")
        string_layer_types = cast("list[str]", layer_types)
        unknown_layer_types = set(string_layer_types) - {"full_attention", "linear_attention"}
        if unknown_layer_types:
            raise ValueError(f"unsupported model config layer types: {unknown_layer_types}")
        full_attention_layers = sum(
            layer_type == "full_attention" for layer_type in string_layer_types
        )

    kv_bytes_per_context_token = (
        2 * num_key_value_heads * head_dim * full_attention_layers * kv_bytes
    )
    total_bytes_per_decode_step = (
        weight_bytes + concurrency * context_tokens * kv_bytes_per_context_token
    )
    return bandwidth_bytes_s * concurrency / total_bytes_per_decode_step
