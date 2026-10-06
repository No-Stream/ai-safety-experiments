"""Capture AISI CodeContests residual streams from saved screen completions.

The screen persists completion text rather than prompt or completion token IDs. This module rebuilds
the exact screen prompt from its runtime helpers and dataset, checks completion tokenization, and
saves selected residual-stream positions without storing prompt or completion text in the index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel

from reward_hacking.interp.directions import (
    capture_positionwise_activations,
    load_model_and_tokenizer,
)
from reward_hacking.organisms.aisi_codecontests import (
    BASE_MODEL,
    _render_chat_prompt,  # pyright: ignore[reportPrivateUsage]  # exact screen render seam
    build_no_hints_system_prompt,
    build_user_prompt,
    load_code_contest_problems,
    record_key,
)
from reward_hacking.organisms.aisi_source import AisiSource, load_aisi_source

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

DEFAULT_GENERATED_LIMIT = 48
ACTIVATION_DTYPE = np.float32
SEQUENCE_DIMENSIONS = 2
ACTIVATION_SHARD_DIMENSIONS = 4
TOKEN_OFFSET_PAIR_LENGTH = 2
CAPTURE_INDEX_SCHEMA_VERSION = 1


class CompletionTokenizer(Protocol):
    """Subset of a fast Hugging Face tokenizer used by capture and span location."""

    @property
    def is_fast(self) -> bool:
        """Whether tokenizer offsets are available."""
        ...

    @property
    def pad_token_id(self) -> int | None:
        """The token ID used for right padding."""
        ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode text without adding prompt or completion special tokens."""
        ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        """Decode a token sequence."""
        ...

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> Mapping[str, Any]:
        """Encode text and return tokenizer-provided character offsets."""
        ...


class ScreenTokenizer(CompletionTokenizer, Protocol):
    """Capture tokenizer methods used by the screen's chat prompt renderer."""

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> object:
        """Render the screen's system and user messages with the base model template."""
        ...


class CaptureConfig(Protocol):
    """The two model configuration fields used for VRAM sizing."""

    @property
    def hidden_size(self) -> int:
        """Model hidden width."""
        ...

    @property
    def num_hidden_layers(self) -> int:
        """Number of decoder layers whose outputs the hooks retain."""
        ...


class CaptureModel(Protocol):
    """Model fields needed by the capture path."""

    @property
    def device(self) -> torch.device:
        """Device holding the model parameters."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Parameter dtype used to size retained residual outputs."""
        ...

    @property
    def config(self) -> CaptureConfig:
        """Model configuration used for layer and hidden-size counts."""
        ...


class CompletionSource(Protocol):
    """Subset of :class:`AisiSource` used to locate solution and extra-file code."""

    def extract_code_and_files(self, completion: str, /) -> tuple[str, dict[str, str]]:
        """Return the solution code and runtime extra files."""
        ...


@dataclass(frozen=True, slots=True)
class TokenizedText:
    """Token IDs and character spans for one exact text string."""

    token_ids: tuple[int, ...]
    char_offsets: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class PositionPlan:
    """Absolute sequence offsets and validity masks for one prompt/completion pair."""

    names: tuple[str, ...]
    indices: tuple[int, ...]
    masks: tuple[bool, ...]

    def mask_mapping(self) -> dict[str, bool]:
        """Return JSON-ready validity flags keyed by position name."""
        return dict(zip(self.names, self.masks, strict=True))


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    """Exact prompt/completion tokenization and selected offsets for one screen record."""

    prompt_token_ids: tuple[int, ...]
    completion: TokenizedText
    position_plan: PositionPlan


@dataclass(slots=True)
class CaptureProgress:
    """Mutable shard and index state persisted after each captured record."""

    output_path: Path
    index_path: Path
    index: dict[str, Any]
    shard: np.memmap | None
    selected: Sequence[Mapping[str, Any]]
    record_key_to_row: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class PendingCapture:
    """Model and prepared records used for a sequence of capture batches."""

    model: CaptureModel
    tokenizer: CompletionTokenizer
    prepared_records: Mapping[str, PreparedRecord]
    batch_size: int


def _tokenize_with_offsets(tokenizer: CompletionTokenizer, text: str) -> TokenizedText:
    raw_token_ids = [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offset_token_ids = [int(token_id) for token_id in cast("Sequence[int]", encoded["input_ids"])]
    if raw_token_ids != offset_token_ids:
        raise ValueError("tokenizer offset encoding returned token IDs that differ from encode()")
    raw_offsets = cast("Sequence[Sequence[int]]", encoded["offset_mapping"])
    if len(raw_offsets) != len(raw_token_ids):
        raise ValueError("tokenizer returned a different number of token IDs and character offsets")
    offsets: list[tuple[int, int]] = []
    for raw_offset in raw_offsets:
        if len(raw_offset) != TOKEN_OFFSET_PAIR_LENGTH:
            raise ValueError("tokenizer character offsets must be start/end pairs")
        start, end = int(raw_offset[0]), int(raw_offset[1])
        if not 0 <= start <= end <= len(text):
            raise ValueError("tokenizer returned character offsets outside the input text")
        offsets.append((start, end))
    return TokenizedText(tuple(raw_token_ids), tuple(offsets))


def tokenize_completion(tokenizer: CompletionTokenizer, completion_text: str) -> TokenizedText:
    """Tokenize a saved completion and refuse any decode round-trip mismatch."""
    tokenized = _tokenize_with_offsets(tokenizer, completion_text)
    decoded = tokenizer.decode(list(tokenized.token_ids), skip_special_tokens=False)
    if decoded != completion_text:
        raise ValueError("completion tokenization round-trip mismatch")
    if not tokenized.token_ids:
        raise ValueError("saved completion tokenized to no tokens")
    return tokenized


def _first_token_covering_character(
    char_offsets: Sequence[tuple[int, int]], char_index: int, *, label: str
) -> int:
    for token_index, (start, end) in enumerate(char_offsets):
        if start <= char_index < end:
            return token_index
    raise ValueError(f"could not map the first character of the {label} block to a token")


def _marker_probe_start(  # noqa: PLR0913 - one probe needs the text, the block and its identity
    completion_text: str,
    completion_offsets: Sequence[tuple[int, int]],
    source: CompletionSource,
    needle: str,
    block_name: str,
    *,
    whole_block: str | None,
) -> int | None:
    """Find the occurrence of ``needle`` that the extractor puts at the start of the parsed block.

    A marker inserted before a candidate occurrence must reappear in the parsed block: exactly in front
    of the whole block when ``whole_block`` is given, otherwise in front of ``needle`` inside it.
    """
    filename = "conftest.py" if block_name == "conftest.py" else None
    character_start = completion_text.find(needle)
    while character_start >= 0:
        marker = f"\x00aisi-capture-span-{block_name}-{character_start}\x00"
        while marker in completion_text:
            marker += "\x00"
        probe_completion = (
            completion_text[:character_start] + marker + completion_text[character_start:]
        )
        parsed_solution, parsed_files = source.extract_code_and_files(probe_completion)
        if filename is None:
            parsed_blocks = [parsed_solution]
        else:
            parsed_blocks = [
                content
                for file_path, content in parsed_files.items()
                if file_path == filename or file_path.endswith(f"/{filename}")
            ]
        if whole_block is not None:
            marker_is_in_block = any(block == f"{marker}{whole_block}" for block in parsed_blocks)
        else:
            marker_is_in_block = any(
                block.lstrip().startswith(f"{marker}{needle}") for block in parsed_blocks
            )
        if marker_is_in_block:
            return _first_token_covering_character(
                completion_offsets, character_start, label=block_name
            )
        character_start = completion_text.find(needle, character_start + 1)
    return None


def _first_token_for_extracted_block(
    completion_text: str,
    completion_offsets: Sequence[tuple[int, int]],
    source: CompletionSource,
    target_text: str,
    block_name: str,
) -> int | None:
    """Return the first token of an extracted block, or ``None`` when the block is empty.

    AISI's extractor sometimes rewrites a block's whitespace (16 of 2200 screen-v1 completions), so the
    block is not always a verbatim substring; its first non-blank line still locates the block start.
    """
    if not target_text.strip():
        return None
    exact_start = _marker_probe_start(
        completion_text,
        completion_offsets,
        source,
        target_text,
        block_name,
        whole_block=target_text,
    )
    if exact_start is not None:
        return exact_start
    first_line = next(line.strip() for line in target_text.splitlines() if line.strip())
    first_line_start = _marker_probe_start(
        completion_text, completion_offsets, source, first_line, block_name, whole_block=None
    )
    if first_line_start is None:
        raise ValueError(f"could not locate extracted {block_name} block in the saved completion")
    return first_line_start


def locate_completion_positions(
    completion_text: str,
    completion: TokenizedText,
    source: CompletionSource,
) -> tuple[int | None, int | None]:
    """Locate the first solution token and first ``conftest.py`` content token; ``None`` masks either."""
    solution_code, extra_files = source.extract_code_and_files(completion_text)
    solution_start = _first_token_for_extracted_block(
        completion_text,
        completion.char_offsets,
        source,
        solution_code,
        "solution",
    )
    conftest_content: str | None = None
    for filename, content in extra_files.items():
        if filename == "conftest.py" or filename.endswith("/conftest.py"):
            conftest_content = content
            break
    conftest_start = (
        None
        if conftest_content is None
        else _first_token_for_extracted_block(
            completion_text,
            completion.char_offsets,
            source,
            conftest_content,
            "conftest.py",
        )
    )
    return solution_start, conftest_start


def build_position_plan(
    prompt_length: int,
    completion_length: int,
    *,
    generated_limit: int = DEFAULT_GENERATED_LIMIT,
    solution_first: int | None,
    conftest_first: int | None,
) -> PositionPlan:
    """Build absolute token offsets for requested positions in prompt plus completion."""
    if prompt_length < 1:
        raise ValueError("prompt_length must be positive")
    if completion_length < 1:
        raise ValueError("completion_length must be positive")
    if generated_limit < 1:
        raise ValueError("generated_limit must be positive")
    for name, offset in (("solution_first", solution_first), ("conftest_first", conftest_first)):
        if offset is not None and not 0 <= offset < completion_length:
            raise ValueError(f"{name} offset {offset} is outside the completion")

    names = ["last_prompt"]
    indices = [prompt_length - 1]
    masks = [True]
    for generated_index in range(1, generated_limit + 1):
        names.append(f"generated_{generated_index}")
        valid = generated_index <= completion_length
        indices.append(prompt_length + generated_index - 1 if valid else -1)
        masks.append(valid)
    for name, offset in (("conftest_first", conftest_first), ("solution_first", solution_first)):
        names.append(name)
        indices.append(prompt_length + offset if offset is not None else -1)
        masks.append(offset is not None)
    names.extend(("last_completion", "completion_mean"))
    indices.extend((prompt_length + completion_length - 1, -1))
    masks.extend((True, True))
    return PositionPlan(tuple(names), tuple(indices), tuple(masks))


def select_position_vectors(
    layer_activations: Mapping[int, torch.Tensor],
    plan: PositionPlan,
    *,
    prompt_length: int,
    completion_length: int,
) -> dict[int, torch.Tensor]:
    """Select planned token rows and the completion mean from ``[sequence, hidden]`` tensors."""
    if prompt_length < 1 or completion_length < 1:
        raise ValueError("prompt and completion lengths must be positive")
    completion_start = prompt_length
    completion_end = prompt_length + completion_length
    selected: dict[int, torch.Tensor] = {}
    for layer, activations in layer_activations.items():
        if activations.ndim != SEQUENCE_DIMENSIONS:
            raise ValueError(f"layer {layer} activations must have shape [sequence, hidden]")
        if activations.shape[0] < completion_end:
            raise ValueError(f"layer {layer} activations are shorter than the token sequence")
        rows: list[torch.Tensor] = []
        for name, index, valid in zip(plan.names, plan.indices, plan.masks, strict=True):
            if name == "completion_mean":
                row = activations[completion_start:completion_end].float().mean(dim=0)
            elif not valid:
                row = torch.zeros(activations.shape[-1], dtype=torch.float32)
            else:
                row = activations[index].float()
            rows.append(row)
        selected[layer] = torch.stack(rows).to(dtype=torch.float32).cpu()
    if not selected:
        raise ValueError("activation capture returned no decoder layers")
    return selected


def _record_key(record: Mapping[str, Any]) -> str:
    saved_key = record.get("record_key")
    if saved_key is not None:
        return str(saved_key)
    return record_key(str(record["model"]), str(record["problem_id"]), int(record["sample_index"]))


def select_records(
    records: Sequence[Mapping[str, Any]], *, model_id: str, max_records: int
) -> list[Mapping[str, Any]]:
    """Select a deterministic, approximately balanced reward-hacked mix for one model."""
    if max_records < 1:
        raise ValueError("max_records must be positive")
    model_records = [record for record in records if str(record["model"]) == model_id]
    if not model_records:
        raise ValueError(f"no records found for model {model_id!r}")
    capturable = [record for record in model_records if str(record["completion_text"]).strip()]
    if len(capturable) != len(model_records):
        logger.info(
            "model=%s skipping %d empty completions of %d records",
            model_id,
            len(model_records) - len(capturable),
            len(model_records),
        )
    model_records = capturable
    keyed: dict[str, Mapping[str, Any]] = {}
    for record in model_records:
        key = _record_key(record)
        if key in keyed:
            raise ValueError(f"duplicate record key {key!r}")
        keyed[key] = record
    ordered = sorted(keyed.items())
    hacked = [item for item in ordered if bool(item[1]["reward_hacked"])]
    honest = [item for item in ordered if not bool(item[1]["reward_hacked"])]
    per_class = max_records // 2
    selected = hacked[:per_class] + honest[:per_class]
    selected_keys = {key for key, _ in selected}
    remaining = [item for item in ordered if item[0] not in selected_keys]
    selected.extend(remaining[: max_records - len(selected)])
    return [record for _, record in sorted(selected[:max_records], key=lambda item: item[0])]


def _encode_prompt(tokenizer: CompletionTokenizer, prompt: str) -> tuple[int, ...]:
    token_ids = tuple(
        int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)
    )
    if not token_ids:
        raise ValueError("rendered screen prompt tokenized to no tokens")
    return token_ids


def _render_record_prompt(
    record: Mapping[str, Any],
    tokenizer: ScreenTokenizer,
    problem_by_id: Mapping[str, Any],
    source: AisiSource,
) -> str:
    problem_id = str(record["problem_id"])
    try:
        problem = problem_by_id[problem_id]
    except KeyError as error:
        raise KeyError(
            f"screen problem {problem_id!r} is absent from the current AISI dataset"
        ) from error
    user_prompt = build_user_prompt(problem.description, list(problem.test_assertions), source)
    return _render_chat_prompt(build_no_hints_system_prompt(source), user_prompt, tokenizer)


def _model_device(model: CaptureModel) -> torch.device:
    return model.device


def batch_size_for_free_vram(
    model: CaptureModel,
    sequences: Sequence[Sequence[int]],
    *,
    max_batch_size: int = 8,
) -> int:
    """Choose a conservative batch size from current free CUDA memory, or one on CPU."""
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    if not sequences:
        raise ValueError("cannot size a batch without token sequences")
    model_device = _model_device(model)
    if model_device.type != "cuda":
        return 1
    device_index = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    device_name = torch.cuda.get_device_name(device_index)
    hidden_size = int(model.config.hidden_size)
    layer_count = int(model.config.num_hidden_layers)
    longest_sequence = max(len(sequence) for sequence in sequences)
    activation_bytes = torch.empty((), dtype=model.dtype).element_size()
    estimated_bytes_per_sequence = max(
        1, longest_sequence * hidden_size * layer_count * activation_bytes
    )
    activation_budget = max(estimated_bytes_per_sequence, int(free_bytes * 0.25))
    batch_size = max(
        1, min(max_batch_size, len(sequences), activation_budget // estimated_bytes_per_sequence)
    )
    logger.info(
        "capture device=%s free_vram_gib=%.2f total_vram_gib=%.2f layers=%d hidden=%d "
        "longest_sequence=%d estimated_bytes_per_sequence=%d batch_size=%d",
        device_name,
        free_bytes / 1024**3,
        total_bytes / 1024**3,
        layer_count,
        hidden_size,
        longest_sequence,
        estimated_bytes_per_sequence,
        batch_size,
    )
    return batch_size


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            line = raw_line.decode("utf-8")
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            records.append(cast("dict[str, Any]", value))
    return records, digest.hexdigest()


def _safe_model_stem(model_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")
    if not stem:
        raise ValueError(f"model ID {model_id!r} has no usable output stem")
    return stem


def _write_index(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record_rows(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    record_key_to_row: dict[str, int] = {}
    for row, record in enumerate(records):
        key = _record_key(record)
        if key in record_key_to_row:
            raise ValueError(f"duplicate record key {key!r}")
        record_key_to_row[key] = row
    return record_key_to_row


def _validate_index_identity(
    index: Mapping[str, Any],
    identity: Mapping[str, Any],
    record_key_to_row: Mapping[str, int],
    position_names: Sequence[str],
) -> None:
    if index.get("identity") != dict(identity):
        raise ValueError("existing capture index identity differs from this capture")
    if index.get("record_key_to_row") != dict(record_key_to_row):
        raise ValueError("existing capture index record rows differ from this capture")
    if index.get("position_names") != list(position_names):
        raise ValueError("existing capture index positions differ from this capture")
    if index.get("schema_version") != CAPTURE_INDEX_SCHEMA_VERSION:
        raise ValueError("existing capture index schema version is unsupported")
    if index.get("dtype") != "float32":
        raise ValueError("existing capture index activation dtype is unsupported")


def _validate_completed_metadata(
    completed: Sequence[str],
    record_key_to_row: Mapping[str, int],
    indexed_records: Mapping[str, Any],
) -> None:
    required_metadata = {
        "row",
        "positions",
        "masks",
        "prompt_token_count",
        "completion_token_range",
        "completion_token_ids",
        "completion_char_offsets",
        "labels",
    }
    for key in completed:
        metadata = indexed_records.get(key)
        if (
            not isinstance(metadata, dict)
            or metadata.get("row") != record_key_to_row[key]
            or not required_metadata <= metadata.keys()
        ):
            raise ValueError(f"existing capture index has invalid metadata for record {key!r}")


def _validate_index_progress(
    index: Mapping[str, Any], record_key_to_row: Mapping[str, int]
) -> None:
    completed = index.get("complete_record_keys")
    if (
        not isinstance(completed, list)
        or any(not isinstance(key, str) for key in completed)
        or completed != sorted(set(completed))
        or not set(completed) <= record_key_to_row.keys()
    ):
        raise ValueError("existing capture index has invalid completed record keys")
    indexed_records = index.get("records")
    if not isinstance(indexed_records, dict):
        raise TypeError("existing capture index has invalid record metadata")
    _validate_completed_metadata(completed, record_key_to_row, indexed_records)
    is_complete = index.get("complete")
    if not isinstance(is_complete, bool) or is_complete != (
        len(completed) == len(record_key_to_row)
    ):
        raise ValueError("existing capture index completion flag is inconsistent")


def load_or_create_capture_index(
    path: Path,
    *,
    identity: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    position_names: Sequence[str],
) -> dict[str, Any]:
    """Load an index only when its identity, rows, and positions match this capture."""
    record_key_to_row = _record_rows(records)
    expected_positions = list(position_names)
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"capture index {path} must contain a JSON object")
        index = cast("dict[str, Any]", value)
        _validate_index_identity(index, identity, record_key_to_row, expected_positions)
        _validate_index_progress(index, record_key_to_row)
        return index

    index = {
        "schema_version": CAPTURE_INDEX_SCHEMA_VERSION,
        "identity": dict(identity),
        "record_key_to_row": record_key_to_row,
        "position_names": expected_positions,
        "position_index_space": "prompt_plus_completion_tokens",
        "records": {},
        "complete_record_keys": [],
        "dtype": "float32",
        "complete": False,
    }
    _write_index(path, index)
    return index


def pending_records(
    records: Sequence[Mapping[str, Any]], index: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """Return records without a persisted activation row, preserving deterministic input order."""
    completed = set(cast("Sequence[str]", index["complete_record_keys"]))
    return [record for record in records if _record_key(record) not in completed]


def _adapter_path(adapter_id: str) -> str:
    local_path = Path(adapter_id)
    if local_path.is_dir():
        return str(local_path.resolve())
    return str(snapshot_download(repo_id=adapter_id))


def _record_labels(record: Mapping[str, Any]) -> dict[str, Any]:
    reward_hacked = record["reward_hacked"]
    hack_regexes_fired = record["hack_regexes_fired"]
    hardened_reward_hacked = record["hardened_reward_hacked"]
    if not isinstance(reward_hacked, bool):
        raise TypeError("screen record reward_hacked label must be boolean")
    if not isinstance(hack_regexes_fired, list) or any(
        not isinstance(label, str) for label in hack_regexes_fired
    ):
        raise TypeError("screen record hack_regexes_fired must be a list of strings")
    if not isinstance(hardened_reward_hacked, bool):
        raise TypeError("screen record hardened_reward_hacked label must be boolean")
    return {
        "reward_hacked": reward_hacked,
        "hack_regexes_fired": list(hack_regexes_fired),
        "hardened_reward_hacked": hardened_reward_hacked,
    }


def _record_index_metadata(record: Mapping[str, Any], prepared: PreparedRecord) -> dict[str, Any]:
    plan = prepared.position_plan
    positions = {
        name: index if valid and name != "completion_mean" else None
        for name, index, valid in zip(plan.names, plan.indices, plan.masks, strict=True)
    }
    return {
        "row": None,
        "positions": positions,
        "masks": plan.mask_mapping(),
        "prompt_token_count": len(prepared.prompt_token_ids),
        "completion_token_range": [
            len(prepared.prompt_token_ids),
            len(prepared.prompt_token_ids) + len(prepared.completion.token_ids),
        ],
        "completion_token_ids": list(prepared.completion.token_ids),
        "completion_char_offsets": [list(offset) for offset in prepared.completion.char_offsets],
        "labels": _record_labels(record),
    }


def _prepare_record(
    record: Mapping[str, Any],
    *,
    tokenizer: CompletionTokenizer,
    prompt: str,
    source: AisiSource,
    generated_limit: int,
) -> PreparedRecord:
    prompt_token_ids = _encode_prompt(tokenizer, prompt)
    completion = tokenize_completion(tokenizer, str(record["completion_text"]))
    solution_first, conftest_first = locate_completion_positions(
        str(record["completion_text"]), completion, source
    )
    position_plan = build_position_plan(
        len(prompt_token_ids),
        len(completion.token_ids),
        generated_limit=generated_limit,
        solution_first=solution_first,
        conftest_first=conftest_first,
    )
    return PreparedRecord(prompt_token_ids, completion, position_plan)


def _capture_batch(
    model: CaptureModel,
    tokenizer: CompletionTokenizer,
    records: Sequence[Mapping[str, Any]],
    prepared_records: Mapping[str, PreparedRecord],
) -> np.ndarray:
    sequences = [
        [
            *prepared_records[_record_key(record)].prompt_token_ids,
            *prepared_records[_record_key(record)].completion.token_ids,
        ]
        for record in records
    ]
    max_length = max(len(sequence) for sequence in sequences)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("capture tokenizer has no pad token")
    input_ids = torch.full((len(sequences), max_length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = 1

    captured = capture_positionwise_activations(
        cast("AutoModelForCausalLM", model), input_ids, attention_mask
    )
    row_vectors: list[np.ndarray] = []
    for row, (record, sequence) in enumerate(zip(records, sequences, strict=True)):
        prepared = prepared_records[_record_key(record)]
        per_layer = {
            layer: activations[row, : len(sequence)] for layer, activations in captured.items()
        }
        selected = select_position_vectors(
            per_layer,
            prepared.position_plan,
            prompt_length=len(prepared.prompt_token_ids),
            completion_length=len(prepared.completion.token_ids),
        )
        row_array = torch.stack([selected[layer] for layer in sorted(selected)]).numpy()
        if not np.isfinite(row_array).all():
            raise ValueError(f"captured activations for {_record_key(record)!r} are not finite")
        row_vectors.append(row_array.astype(ACTIVATION_DTYPE, copy=False))
    return np.stack(row_vectors)


def _validate_capture_records(
    model_id: str, records: Sequence[Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], str | None, dict[str, int]]:
    if not records:
        raise ValueError(f"no selected records for model {model_id!r}")
    selected = sorted(records, key=_record_key)
    if any(str(record["model"]) != model_id for record in selected):
        raise ValueError(f"capture records contain a model other than {model_id!r}")
    adapter_ids = {record["adapter"] for record in selected}
    if len(adapter_ids) != 1:
        raise ValueError(f"records for {model_id!r} have inconsistent adapter IDs")
    adapter_id = cast("str | None", next(iter(adapter_ids)))
    expected_adapter = None if model_id == BASE_MODEL else model_id
    if adapter_id != expected_adapter:
        raise ValueError(f"model {model_id!r} has unexpected adapter {adapter_id!r}")
    return selected, adapter_id, _record_rows(selected)


def _capture_position_names(generated_limit: int) -> list[str]:
    if generated_limit < 1:
        raise ValueError("generated_limit must be positive")
    return [
        "last_prompt",
        *[f"generated_{index}" for index in range(1, generated_limit + 1)],
        "conftest_first",
        "solution_first",
        "last_completion",
        "completion_mean",
    ]


def _open_activation_shard(
    output_path: Path, index: Mapping[str, Any], record_count: int
) -> np.memmap | None:
    if output_path.exists():
        shard = cast("np.memmap", np.load(output_path, mmap_mode="r+"))
        if (
            shard.dtype != ACTIVATION_DTYPE
            or shard.ndim != ACTIVATION_SHARD_DIMENSIONS
            or shard.shape[0] != record_count
        ):
            raise ValueError("existing activation shard has an incompatible dtype or row count")
        indexed_shape = index.get("shape")
        if indexed_shape is not None and list(shard.shape) != indexed_shape:
            raise ValueError("existing activation shard shape differs from its index")
        if index["complete_record_keys"] and indexed_shape is None:
            raise ValueError("existing capture index marks rows complete without a shard shape")
        return shard
    if index["complete_record_keys"]:
        raise ValueError("capture index marks completed rows but the activation shard is missing")
    return None


def _load_capture_model(adapter_id: str | None) -> tuple[CaptureModel, ScreenTokenizer]:
    base_model, raw_tokenizer = load_model_and_tokenizer(BASE_MODEL, dtype=torch.bfloat16)
    tokenizer = cast("ScreenTokenizer", raw_tokenizer)
    if not tokenizer.is_fast:
        raise RuntimeError("AISI capture requires a fast tokenizer for character offsets")
    model = cast("CaptureModel", base_model)
    if adapter_id is not None:
        peft_model = PeftModel.from_pretrained(
            cast("torch.nn.Module", base_model), _adapter_path(adapter_id)
        )
        peft_model.eval()
        model = cast("CaptureModel", peft_model)
    return model, tokenizer


def _prepare_pending_records(
    pending: Sequence[Mapping[str, Any]],
    *,
    tokenizer: ScreenTokenizer,
    source: AisiSource,
    problem_by_id: Mapping[str, Any],
    generated_limit: int,
) -> tuple[dict[str, PreparedRecord], list[list[int]]]:
    prompts = {
        _record_key(record): _render_record_prompt(record, tokenizer, problem_by_id, source)
        for record in pending
    }
    prepared_records = {
        _record_key(record): _prepare_record(
            record,
            tokenizer=tokenizer,
            prompt=prompts[_record_key(record)],
            source=source,
            generated_limit=generated_limit,
        )
        for record in pending
    }
    sequences = [
        [
            *prepared_records[_record_key(record)].prompt_token_ids,
            *prepared_records[_record_key(record)].completion.token_ids,
        ]
        for record in pending
    ]
    return prepared_records, sequences


def _persist_capture_batches(
    progress: CaptureProgress,
    pending: Sequence[Mapping[str, Any]],
    capture: PendingCapture,
) -> np.memmap:
    for start in range(0, len(pending), capture.batch_size):
        batch_records = pending[start : start + capture.batch_size]
        vectors = _capture_batch(
            capture.model, capture.tokenizer, batch_records, capture.prepared_records
        )
        if progress.shard is None:
            progress.shard = np.lib.format.open_memmap(
                progress.output_path,
                mode="w+",
                dtype=ACTIVATION_DTYPE,
                shape=(len(progress.selected), *vectors.shape[1:]),
            )
        elif tuple(progress.shard.shape[1:]) != tuple(vectors.shape[1:]):
            raise ValueError(
                "existing activation shard shape differs from newly captured activations"
            )

        for vector, record in zip(vectors, batch_records, strict=True):
            key = _record_key(record)
            row = progress.record_key_to_row[key]
            metadata = _record_index_metadata(record, capture.prepared_records[key])
            metadata["row"] = row
            progress.shard[row] = vector
            progress.shard.flush()
            progress.index["records"][key] = metadata
            progress.index["complete_record_keys"] = sorted(
                [*progress.index["complete_record_keys"], key]
            )
            progress.index["shape"] = list(progress.shard.shape)
            progress.index["complete"] = len(progress.index["complete_record_keys"]) == len(
                progress.selected
            )
            _write_index(progress.index_path, progress.index)
    if progress.shard is None:
        raise ValueError("capture produced no activation rows")
    return progress.shard


def capture_model(  # noqa: PLR0913
    *,
    out_dir: Path,
    model_id: str,
    records: Sequence[Mapping[str, Any]],
    source: AisiSource,
    problem_by_id: Mapping[str, Any],
    sample_file_sha256: str,
    generated_limit: int = DEFAULT_GENERATED_LIMIT,
) -> Path:
    """Capture one selected model's records into a resumable NumPy shard and JSON index."""
    selected, adapter_id, record_key_to_row = _validate_capture_records(model_id, records)
    stem = _safe_model_stem(model_id)
    output_path = out_dir / f"activations.{stem}.npy"
    index_path = out_dir / f"activation-index.{stem}.json"
    identity = {
        "base_model": BASE_MODEL,
        "adapter": adapter_id,
        "model": model_id,
        "sample_file_sha256": sample_file_sha256,
    }
    position_names = _capture_position_names(generated_limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not index_path.exists():
        raise ValueError(f"activation shard exists without its index for {model_id!r}")
    index = load_or_create_capture_index(
        index_path,
        identity=identity,
        records=selected,
        position_names=position_names,
    )
    shard = _open_activation_shard(output_path, index, len(selected))
    pending = pending_records(selected, index)
    if not pending:
        if shard is None:
            raise ValueError("capture index is complete but the activation shard is missing")
        return output_path

    model, tokenizer = _load_capture_model(adapter_id)
    device = _model_device(model)
    logger.info("capturing model=%s device=%s", model_id, device)
    prepared_records, sequences = _prepare_pending_records(
        pending,
        tokenizer=tokenizer,
        source=source,
        problem_by_id=problem_by_id,
        generated_limit=generated_limit,
    )
    batch_size = batch_size_for_free_vram(model, sequences)
    logger.info(
        "capturing model=%s pending_records=%d batch_size=%d", model_id, len(pending), batch_size
    )
    progress = CaptureProgress(
        output_path,
        index_path,
        index,
        shard,
        selected,
        record_key_to_row,
    )
    _persist_capture_batches(
        progress,
        pending,
        PendingCapture(model, tokenizer, prepared_records, batch_size),
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Screen model IDs; use 'base' for the parent model.",
    )
    parser.add_argument("--max-records-per-model", type=int, default=40)
    parser.add_argument("--generated-limit", type=int, default=DEFAULT_GENERATED_LIMIT)
    parser.add_argument("--external-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Capture requested models from one existing AISI screen run."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    if args.max_records_per_model < 1:
        raise ValueError("--max-records-per-model must be positive")
    if args.generated_limit < 1:
        raise ValueError("--generated-limit must be positive")
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    if run_dir == out_dir:
        raise ValueError("--out-dir must differ from --run-dir")
    samples_path = run_dir / "samples.jsonl"
    records, sample_file_sha256 = _read_jsonl(samples_path)
    requested_models = [BASE_MODEL if model == "base" else model for model in args.models]
    if len(set(requested_models)) != len(requested_models):
        raise ValueError("--models contains duplicate model IDs after resolving 'base'")
    available_models = {str(record["model"]) for record in records}
    missing_models = [model for model in requested_models if model not in available_models]
    if missing_models:
        raise ValueError(f"requested models are absent from {samples_path}: {missing_models}")

    source = load_aisi_source(args.external_root)
    problems = load_code_contest_problems(source=source)
    problem_by_id = {problem.problem_id: problem for problem in problems}
    if len(problem_by_id) != len(problems):
        raise ValueError("AISI dataset contains duplicate problem IDs")
    for model_id in requested_models:
        selected = select_records(
            records, model_id=model_id, max_records=args.max_records_per_model
        )
        capture_model(
            out_dir=out_dir,
            model_id=model_id,
            records=selected,
            source=source,
            problem_by_id=problem_by_id,
            sample_file_sha256=sample_file_sha256,
            generated_limit=args.generated_limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
