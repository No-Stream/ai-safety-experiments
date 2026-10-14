"""Build the private, rendered prompt corpus used by the distillation arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from reward_hacking.atlas.core import render_prompt as render_atlas_prompt
from reward_hacking.harness.tasks_ilcb import PROBLEMS
from reward_hacking.hatch_probe import (
    FRAMING_NEUTRAL,
    HATCH_ABSENT,
    ProbeCell,
    render_cell_prompt,
)
from reward_hacking.jsonl_io import read_jsonl_objects
from reward_hacking.organisms.aisi_codecontests import (
    ChatTokenizer,
    CodeContestProblem,
    load_code_contest_problems,
)
from reward_hacking.organisms.aisi_codecontests import (
    _render_chat_prompt as render_aisi_chat_prompt,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.organisms.aisi_source import (
    AisiSource,
    build_no_hints_system_prompt,
    build_user_prompt,
    load_aisi_source,
)
from reward_hacking.organisms.leakage_hints import LeakageHint, load_leakage_hint
from reward_hacking.train_dataset import templated_prompt

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reward_hacking.atlas.prompts import ChatTemplateTokenizer

logger = logging.getLogger(__name__)

NINE_BILLION_MODEL = "Qwen/Qwen3.5-9B"
TWENTY_SEVEN_BILLION_MODEL = "RedHatAI/Qwen3.8-27B-INT4"
ENVIRONMENT_ORDER = ("atlas", "aisi", "ilcb")
TRAINING_ENVIRONMENTS = ("atlas", "aisi")
DEFAULT_SCREEN_COUNT = 20
DEFAULT_TRAINING_COUNT = 40
SHA256_HEX_LENGTH = 64
PAIR_FIELD_COUNT = 2


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """One policy-visible prompt and its exact 9B tokenization."""

    environment: str
    problem_id: str
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    hint_variant: str
    hint_sha256: str
    source_split: str | None = None
    hint_text: str | None = None

    def __post_init__(self) -> None:
        """Reject an incomplete or malformed prompt row before it reaches disk."""
        if self.environment not in ENVIRONMENT_ORDER:
            raise ValueError(f"unknown prompt environment {self.environment!r}")
        if not self.problem_id:
            raise ValueError("problem_id must not be empty")
        if not self.prompt_text:
            raise ValueError(f"{self.environment}/{self.problem_id} has an empty prompt")
        if not self.prompt_token_ids:
            raise ValueError(f"{self.environment}/{self.problem_id} has no prompt tokens")
        if not self.hint_variant:
            raise ValueError("hint_variant must not be empty")
        if len(self.hint_sha256) != SHA256_HEX_LENGTH:
            raise ValueError("hint_sha256 must be a SHA-256 hex digest")
        int(self.hint_sha256, 16)

    def to_json_dict(self) -> dict[str, object]:
        """Return the fixed JSONL contract without source-only ordering metadata."""
        payload: dict[str, object] = {
            "env": self.environment,
            "problem_id": self.problem_id,
            "prompt_text": self.prompt_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "hint_variant": self.hint_variant,
            "hint_sha256": self.hint_sha256,
        }
        if self.hint_text is not None:
            payload["hint_text"] = self.hint_text
        return payload


@dataclass(frozen=True, slots=True)
class PromptSplit:
    """The one-time train/held-out assignment for the exact rendered prompt pool."""

    train: tuple[tuple[str, str], ...]
    held_out: tuple[tuple[str, str], ...]
    pool_fingerprint: str
    prompt_content: tuple[PromptRecord, ...] = ()

    def to_json_dict(self) -> dict[str, object]:
        """Serialize the one-time assignment without including benchmark prompt material."""
        return {
            "train": [list(pair) for pair in self.train],
            "held_out": [list(pair) for pair in self.held_out],
            "pool_fingerprint": self.pool_fingerprint,
        }


def cached_model_snapshot(model_id: str) -> Path | None:
    """Find a local tokenizer snapshot without making a Hub request."""
    tokenizer_path = try_to_load_from_cache(model_id, "tokenizer.json")
    if not isinstance(tokenizer_path, str):
        return None
    return Path(tokenizer_path).parent


@dataclass(frozen=True, slots=True)
class PromptRendering:
    """The 9B rendered prompt and optional 27B check performed before writing it."""

    prompt_text: str
    tokenizer: PreTrainedTokenizerBase
    teacher_prompt_text: str | None = None
    teacher_tokenizer: PreTrainedTokenizerBase | None = None


@dataclass(frozen=True, slots=True)
class AtlasPromptContext:
    """Renderer settings shared by rows in one Atlas prompt-stage file."""

    tokenizer: PreTrainedTokenizerBase
    teacher_tokenizer: PreTrainedTokenizerBase | None
    screen_hints: Mapping[str, LeakageHint]
    unavailable_screen_ids: Mapping[str, str]
    template_kwargs: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class HintResolution:
    """One stored hint identity, available only when its historical content is recoverable."""

    hint: LeakageHint | None
    limitation: str | None


def load_cached_tokenizer(model_id: str, *, required: bool) -> PreTrainedTokenizerBase | None:
    """Load a tokenizer from the HF cache only."""
    snapshot_path = cached_model_snapshot(model_id)
    if snapshot_path is None:
        if required:
            raise FileNotFoundError(f"no cached tokenizer snapshot for {model_id}")
        return None
    return AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)


def create_prompt_record(
    *,
    environment: str,
    problem_id: str,
    rendering: PromptRendering,
    hint: LeakageHint,
    source_split: str | None = None,
) -> PromptRecord:
    """Tokenize a student rendering and verify the 27B medium rendering when available."""
    token_ids = tuple(
        int(token_id)
        for token_id in rendering.tokenizer.encode(
            rendering.prompt_text,
            add_special_tokens=False,
        )
    )
    if rendering.teacher_tokenizer is not None:
        if rendering.teacher_prompt_text is None:
            raise ValueError("teacher_prompt_text is required when a 27B tokenizer is available")
        if rendering.teacher_prompt_text != rendering.prompt_text:
            raise RuntimeError(
                f"{environment}/{problem_id} has a 27B rendering that differs from the 9B prompt"
            )
        teacher_token_ids = tuple(
            int(token_id)
            for token_id in rendering.teacher_tokenizer.encode(
                rendering.teacher_prompt_text,
                add_special_tokens=False,
            )
        )
        if teacher_token_ids != token_ids:
            raise RuntimeError(
                f"{environment}/{problem_id} has 27B token ids that differ from the 9B prompt"
            )
    return PromptRecord(
        environment=environment,
        problem_id=problem_id,
        prompt_text=rendering.prompt_text,
        prompt_token_ids=token_ids,
        hint_variant=hint.variant,
        hint_sha256=hint.text_sha256,
        source_split=source_split,
        hint_text=hint.text,
    )


def _prompt_pool_fingerprint(records: Sequence[PromptRecord]) -> str:
    """Fingerprint ordered prompt identities and rendered content without storing text in the split."""
    identity_rows = [
        [
            record.environment,
            record.problem_id,
            hashlib.sha256(record.prompt_text.encode("utf-8")).hexdigest(),
            hashlib.sha256(
                json.dumps(record.prompt_token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            record.hint_variant,
            record.hint_sha256,
        ]
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(identity_rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _index_records(
    environment: str,
    records: Sequence[PromptRecord],
) -> dict[str, PromptRecord]:
    indexed: dict[str, PromptRecord] = {}
    for record in records:
        if record.environment != environment:
            raise ValueError(f"{environment} source contains a {record.environment} prompt record")
        if record.problem_id in indexed:
            raise ValueError(f"{environment} prompt pool repeats id {record.problem_id!r}")
        indexed[record.problem_id] = record
    return indexed


def _screen_ids(
    environment: str,
    screen_ids_by_environment: Mapping[str, Sequence[str]],
    *,
    screen_count: int | None,
) -> tuple[str, ...]:
    screen_ids = tuple(screen_ids_by_environment[environment])
    if screen_count is not None and len(screen_ids) != screen_count:
        raise ValueError(
            f"{environment} screen has {len(screen_ids)} distinct ids, expected {screen_count}"
        )
    if len(set(screen_ids)) != len(screen_ids):
        raise ValueError(f"{environment} screen repeats a problem id")
    return screen_ids


def _select_training_environment(
    environment: str,
    records: Sequence[PromptRecord],
    screen_ids: Sequence[str],
    *,
    screen_count: int | None,
    training_count: int,
) -> tuple[tuple[PromptRecord, ...], tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    groups: dict[str, list[PromptRecord]] = {}
    for record in records:
        groups.setdefault(record.source_split or "default", []).append(record)

    screen_set = set(screen_ids)
    active_groups = _active_training_groups(environment, groups, screen_set, screen_count)
    if not active_groups:
        raise ValueError(f"{environment} prompt pool has no source groups")
    if training_count % len(active_groups):
        raise ValueError(
            f"training_count={training_count} must divide evenly across "
            f"{len(active_groups)} {environment} source groups"
        )

    training_per_group = training_count // len(active_groups)
    selected: list[PromptRecord] = []
    train: list[tuple[str, str]] = []
    held_out: list[tuple[str, str]] = []
    for _, group_records, held_ids in active_groups:
        group_ids = [record.problem_id for record in group_records]
        train_ids = group_ids[len(held_ids) : len(held_ids) + training_per_group]
        if len(train_ids) != training_per_group:
            raise ValueError(
                f"{environment} source group has {len(train_ids)} training rows after its screen "
                f"prefix; needs {training_per_group}"
            )
        held_out.extend((environment, problem_id) for problem_id in held_ids)
        train.extend((environment, problem_id) for problem_id in train_ids)
        selected.extend(record for record in group_records[: len(held_ids)])
        selected.extend(
            record for record in group_records[len(held_ids) : len(held_ids) + training_per_group]
        )
    return tuple(selected), tuple(train), tuple(held_out)


def _active_training_groups(
    environment: str,
    groups: Mapping[str, list[PromptRecord]],
    screen_ids: set[str],
    screen_count: int | None,
) -> list[tuple[str, list[PromptRecord], list[str]]]:
    active_groups: list[tuple[str, list[PromptRecord], list[str]]] = []
    for group, group_records in groups.items():
        group_ids = [record.problem_id for record in group_records]
        held_ids = [problem_id for problem_id in group_ids if problem_id in screen_ids]
        if held_ids and group_ids[: len(held_ids)] != held_ids:
            raise ValueError(f"{environment}/{group} screen ids are not the source pool prefix")
        if screen_count is None or held_ids:
            active_groups.append((group, group_records, held_ids))

    mapped_screen_ids = {problem_id for _, _, held_ids in active_groups for problem_id in held_ids}
    if mapped_screen_ids != screen_ids:
        raise ValueError(f"{environment} screen ids do not map to its source groups")
    if screen_count is not None:
        group_counts = {len(held_ids) for _, _, held_ids in active_groups}
        if len(group_counts) != 1:
            raise ValueError(f"{environment} screen count is unbalanced across source groups")
    return active_groups


def _select_ilcb_screen(
    records: Sequence[PromptRecord],
    screen_ids: Sequence[str],
) -> tuple[tuple[PromptRecord, ...], tuple[tuple[str, str], ...]]:
    indexed = _index_records("ilcb", records)
    selected_ids = tuple(sorted(screen_ids))
    if set(indexed) != set(selected_ids):
        raise ValueError("ILCB prompt pool must contain exactly the recorded screen ids")
    return (
        tuple(indexed[problem_id] for problem_id in selected_ids),
        tuple(("ilcb", problem_id) for problem_id in selected_ids),
    )


def build_prompt_corpus(
    records_by_environment: Mapping[str, Sequence[PromptRecord]],
    screen_ids_by_environment: Mapping[str, Sequence[str]],
    *,
    screen_count: int | None = DEFAULT_SCREEN_COUNT,
    training_count: int = DEFAULT_TRAINING_COUNT,
) -> tuple[tuple[PromptRecord, ...], PromptSplit]:
    """Select screen-held-out rows and the next training rows from Atlas and AISI."""
    if screen_count is not None and screen_count < 1:
        raise ValueError(f"screen_count must be positive, got {screen_count}")
    if training_count < 0:
        raise ValueError(f"training_count must be non-negative, got {training_count}")
    expected_environments = set(ENVIRONMENT_ORDER)
    if set(records_by_environment) != expected_environments:
        raise ValueError(f"prompt pools must have exactly {ENVIRONMENT_ORDER}")
    if set(screen_ids_by_environment) != expected_environments:
        raise ValueError(f"screen ids must have exactly {ENVIRONMENT_ORDER}")

    train: list[tuple[str, str]] = []
    held_out: list[tuple[str, str]] = []
    selected_by_environment: dict[str, tuple[PromptRecord, ...]] = {}
    for environment in TRAINING_ENVIRONMENTS:
        records = tuple(records_by_environment[environment])
        _index_records(environment, records)
        screen_ids = _screen_ids(
            environment,
            screen_ids_by_environment,
            screen_count=screen_count,
        )
        selected, environment_train, environment_held_out = _select_training_environment(
            environment,
            records,
            screen_ids,
            screen_count=screen_count,
            training_count=training_count,
        )
        selected_by_environment[environment] = selected
        train.extend(environment_train)
        held_out.extend(environment_held_out)

    ilcb_records = tuple(records_by_environment["ilcb"])
    ilcb_screen_ids = _screen_ids(
        "ilcb",
        screen_ids_by_environment,
        screen_count=screen_count,
    )
    selected_ilcb, ilcb_held_out = _select_ilcb_screen(ilcb_records, ilcb_screen_ids)
    selected_by_environment["ilcb"] = selected_ilcb
    held_out.extend(ilcb_held_out)

    selected_records = tuple(
        record
        for environment in ENVIRONMENT_ORDER
        for record in selected_by_environment[environment]
    )
    split = PromptSplit(
        train=tuple(train),
        held_out=tuple(held_out),
        pool_fingerprint=_prompt_pool_fingerprint(selected_records),
        prompt_content=selected_records,
    )
    return selected_records, split


def _read_split_pairs(
    path: Path,
    payload: Mapping[str, object],
    side: str,
) -> tuple[tuple[str, str], ...]:
    raw_pairs = payload[side]
    if not isinstance(raw_pairs, list):
        raise TypeError(f"{path} field {side!r} must be a list")
    pairs: list[tuple[str, str]] = []
    for pair in raw_pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != PAIR_FIELD_COUNT
            or not all(isinstance(value, str) for value in pair)
        ):
            raise TypeError(f"{path} field {side!r} must contain [env, problem_id] pairs")
        pairs.append((pair[0], pair[1]))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{path} field {side!r} repeats a prompt key")
    return tuple(pairs)


def load_prompt_split(path: Path, current_records: Sequence[PromptRecord]) -> PromptSplit:
    """Load a stored split and refuse a changed rendered prompt pool."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"train", "held_out", "pool_fingerprint"}:
        raise ValueError(f"{path} must contain only train, held_out and pool_fingerprint")
    split_payload = cast("dict[str, object]", payload)
    fingerprint = payload["pool_fingerprint"]
    if not isinstance(fingerprint, str):
        raise TypeError(f"{path} field 'pool_fingerprint' must be a string")
    split = PromptSplit(
        train=_read_split_pairs(path, split_payload, "train"),
        held_out=_read_split_pairs(path, split_payload, "held_out"),
        pool_fingerprint=fingerprint,
        prompt_content=tuple(current_records),
    )
    if set(split.train) & set(split.held_out):
        raise ValueError(f"{path} contains overlapping train and held_out ids")
    record_keys = {(record.environment, record.problem_id) for record in current_records}
    split_keys = set(split.train + split.held_out)
    if record_keys != split_keys:
        raise ValueError(f"{path} prompt ids do not match the currently rendered prompt pool")
    stored_path = path.with_name("prompts.jsonl")
    stored_records = read_jsonl_objects(stored_path) if stored_path.is_file() else []
    live_records = {
        (record.environment, record.problem_id): record.to_json_dict() for record in current_records
    }
    content_fields = (
        "env",
        "problem_id",
        "prompt_text",
        "prompt_token_ids",
        "hint_variant",
        "hint_text",
    )
    stored_keys = [(row["env"], row["problem_id"]) for row in stored_records]
    if stored_records and (
        len(stored_keys) != len(set(stored_keys)) or set(stored_keys) != record_keys
    ):
        raise ValueError("stored prompt corpus ids do not match split ids")
    for row in stored_records:
        live = live_records[(str(row["env"]), str(row["problem_id"]))]
        if any(row[field] != live.get(field) for field in content_fields if field in row):
            raise RuntimeError(
                f"{path} rendered prompt content changed; pool fingerprint is provenance"
            )
    if not stored_records:
        logger.warning("legacy prompt split has no stored corpus; checking split task ids only")
    return split


def write_prompt_corpus(
    records: Sequence[PromptRecord],
    split: PromptSplit,
    output_dir: Path,
) -> None:
    """Write the private JSONL corpus and one-time split, refusing to overwrite either."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / "prompts.jsonl"
    split_path = output_dir / "prompt-split.json"
    if prompts_path.exists() or split_path.exists():
        raise FileExistsError(
            f"{output_dir} already contains prompt artifacts; refusing to relabel an existing run"
        )
    if split.prompt_content and _prompt_content(split.prompt_content) != _prompt_content(records):
        raise ValueError("prompt content does not match the records used to assign the split")
    prompt_keys = [(record.environment, record.problem_id) for record in records]
    if len(prompt_keys) != len(set(prompt_keys)):
        raise ValueError("prompt corpus repeats an environment/problem id")
    if set(prompt_keys) != set(split.train + split.held_out):
        raise ValueError("prompt corpus ids do not equal the split's train and held_out ids")

    prompt_temp = prompts_path.with_name(prompts_path.name + ".tmp")
    split_temp = split_path.with_name(split_path.name + ".tmp")
    with prompt_temp.open("x", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
    with split_temp.open("x", encoding="utf-8") as output:
        output.write(json.dumps(split.to_json_dict(), indent=2, sort_keys=True) + "\n")
    prompt_temp.replace(prompts_path)
    split_temp.replace(split_path)


def _prompt_content(records: Sequence[PromptRecord]) -> dict[tuple[str, str], dict[str, object]]:
    fields = ("prompt_text", "prompt_token_ids", "hint_variant", "hint_text")
    return {
        (record.environment, record.problem_id): {
            field: value for field, value in record.to_json_dict().items() if field in fields
        }
        for record in records
    }


def _required_string(row: Mapping[str, object], key: str) -> str:
    value = row[key]
    if not isinstance(value, str):
        raise TypeError(f"prompt source field {key!r} must be a string")
    return value


def _resolve_hint_fields(
    row: Mapping[str, object],
    *,
    expected_environment: str,
) -> HintResolution:
    environment = _required_string(row, "hint_environment")
    variant = _required_string(row, "hint_variant")
    hint_text_value = row.get("hint_text")
    if hint_text_value is not None and not isinstance(hint_text_value, str):
        raise TypeError("prompt source field 'hint_text' must be a string")
    stored_digest = _validate_hint_digest(row.get("hint_sha256"), required=hint_text_value is None)
    if not variant:
        raise ValueError("prompt source hint_variant must not be empty")
    if environment != expected_environment:
        return HintResolution(None, "historical hint unavailable: environment identity mismatch")

    if hint_text_value is not None:
        return _hint_from_recorded_text(environment, variant, hint_text_value, stored_digest)
    if stored_digest is None:
        raise AssertionError("required legacy hint digest was not validated")
    return _hint_from_legacy_digest(environment, variant, stored_digest)


def _validate_hint_digest(value: object, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise TypeError("prompt source field 'hint_sha256' must be a string")
    if len(value) != SHA256_HEX_LENGTH:
        raise ValueError("prompt source hint_sha256 must be a SHA-256 digest")
    int(value, 16)
    return value


def _hint_from_recorded_text(
    environment: str,
    variant: str,
    text: str,
    stored_digest: str | None,
) -> HintResolution:
    if (variant == "none" and text != "") or (variant != "none" and not text.strip()):
        return HintResolution(None, "historical hint unavailable: semantic identity mismatch")
    hint = LeakageHint(environment=environment, variant=variant, text=text)
    if stored_digest is not None and stored_digest != hint.text_sha256:
        return HintResolution(None, "historical hint unavailable: digest mismatch")
    return HintResolution(hint, None)


def _hint_from_legacy_digest(
    environment: str,
    variant: str,
    digest: str,
) -> HintResolution:
    try:
        hint = load_leakage_hint(environment, variant)
    except KeyError as error:
        message = error.args[0] if error.args else ""
        if isinstance(message, str) and (
            message.startswith("no leakage hints for environment ")
            or (message.startswith("environment ") and " has no hint variant " in message)
        ):
            return HintResolution(None, "historical hint unavailable: catalog identity is unknown")
        raise
    if hint.text_sha256 != digest:
        return HintResolution(None, "historical hint unavailable: digest mismatch")
    return HintResolution(hint, None)


def _prompt_row_hint(row: Mapping[str, object], *, environment: str) -> HintResolution:
    return _resolve_hint_fields(
        {
            "hint_environment": environment,
            "hint_variant": row.get("hint_variant"),
            "hint_sha256": row.get("hint_sha256"),
            **({"hint_text": row["hint_text"]} if "hint_text" in row else {}),
        },
        expected_environment=environment,
    )


def _screen_hint_records(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_environment: str,
    problem_id_field: str,
) -> tuple[dict[str, LeakageHint], dict[str, str]]:
    hints: dict[str, LeakageHint] = {}
    unavailable: dict[str, str] = {}
    for row in rows:
        problem_id = _required_string(row, problem_id_field)
        resolution = _resolve_hint_fields(row, expected_environment=expected_environment)
        previous_hint = hints.get(problem_id)
        previous_limitation = unavailable.get(problem_id)
        if previous_hint is not None or previous_limitation is not None:
            if previous_hint == resolution.hint and previous_limitation == resolution.limitation:
                continue
            hints.pop(problem_id, None)
            unavailable[problem_id] = (
                "historical hint unavailable: conflicting duplicate screen identity"
            )
            continue
        if resolution.hint is None:
            unavailable[problem_id] = (
                resolution.limitation or "historical hint unavailable: identity conflict"
            )
        else:
            hints[problem_id] = resolution.hint
    return hints, unavailable


def _log_unavailable_hints(environment: str, unavailable: Mapping[str, str]) -> None:
    if not unavailable:
        return
    limitation_counts: dict[str, int] = {}
    for limitation in unavailable.values():
        limitation_counts[limitation] = limitation_counts.get(limitation, 0) + 1
    logger.warning(
        "%s screen rows excluded for unavailable hint provenance: %s",
        environment,
        dict(sorted(limitation_counts.items())),
    )


def _render_ilcb_user_prompt(
    raw_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    reasoning_effort: str | None,
) -> str:
    if reasoning_effort is None:
        return templated_prompt(tokenizer, raw_prompt, enable_thinking=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template returned non-text ILCB prompt")
    return rendered


def _atlas_prompt_context(
    prompt_path: Path,
    screen_rows: Sequence[Mapping[str, object]],
    tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase | None,
) -> AtlasPromptContext:
    metadata_path = prompt_path.with_name("prompt-metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError(f"Atlas prompt metadata must be an object: {metadata_path}")
    model_id = _required_string(cast("Mapping[str, object]", metadata), "model_id")
    if model_id != NINE_BILLION_MODEL:
        raise ValueError(f"Atlas prompt stage used {model_id!r}, not {NINE_BILLION_MODEL}")
    screen_hints, unavailable_screen_ids = _screen_hint_records(
        screen_rows,
        expected_environment="atlas_mbpp",
        problem_id_field="item_id",
    )
    _log_unavailable_hints("Atlas", unavailable_screen_ids)
    screen_sources = {_required_string(row, "source") for row in screen_rows}
    if screen_sources != {"mbpp-hardcode"}:
        raise ValueError("Atlas distillation prompts require the mbpp-hardcode screen source")
    raw_template_kwargs = metadata["chat_template_kwargs"]
    if not isinstance(raw_template_kwargs, dict):
        raise TypeError("Atlas prompt metadata chat_template_kwargs must be an object")
    return AtlasPromptContext(
        tokenizer=tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        screen_hints=screen_hints,
        unavailable_screen_ids=unavailable_screen_ids,
        template_kwargs=cast("dict[str, object]", raw_template_kwargs),
    )


def _render_atlas_row(
    row: Mapping[str, object], context: AtlasPromptContext
) -> PromptRecord | None:
    if row["source"] != "mbpp-hardcode":
        return None
    item_id = _required_string(row, "item_id")
    if item_id in context.unavailable_screen_ids:
        return None
    hint = _atlas_row_hint(row, context, item_id)
    if hint is None:
        return None
    task_text = _required_string(row, "task_text")
    prompt_text_from_stage = _required_string(row, "prompt")
    source_split = f"mbpp-hardcode:{_required_string(row, 'split')}"
    raw_test_list = row["test_list"]
    if not isinstance(raw_test_list, list):
        raise TypeError(f"Atlas prompt row {item_id!r} has an invalid test_list")
    test_list = [test for test in raw_test_list if isinstance(test, str)]
    if len(test_list) != len(raw_test_list):
        raise TypeError(f"Atlas prompt row {item_id!r} has a non-text test case")
    prompt_text = render_atlas_prompt(
        cast("ChatTemplateTokenizer", context.tokenizer),
        task_text,
        test_list,
        hint_text=hint.text,
        chat_template_kwargs=context.template_kwargs,
    )
    if prompt_text != prompt_text_from_stage:
        raise RuntimeError(
            f"Atlas prompt row {item_id!r} no longer matches its stored 9B rendering"
        )
    teacher_prompt_text = None
    if context.teacher_tokenizer is not None:
        teacher_prompt_text = render_atlas_prompt(
            cast("ChatTemplateTokenizer", context.teacher_tokenizer),
            task_text,
            test_list,
            hint_text=hint.text,
            chat_template_kwargs={**context.template_kwargs, "reasoning_effort": "medium"},
        )
    return create_prompt_record(
        environment="atlas",
        problem_id=item_id,
        rendering=PromptRendering(
            prompt_text=prompt_text,
            tokenizer=context.tokenizer,
            teacher_prompt_text=teacher_prompt_text,
            teacher_tokenizer=context.teacher_tokenizer,
        ),
        hint=hint,
        source_split=source_split,
    )


def _atlas_row_hint(
    row: Mapping[str, object],
    context: AtlasPromptContext,
    item_id: str,
) -> LeakageHint | None:
    screen_hint = context.screen_hints.get(item_id)
    if screen_hint is None:
        resolution = _prompt_row_hint(row, environment="atlas_mbpp")
        return resolution.hint
    if _required_string(row, "hint_variant") != screen_hint.variant:
        return None
    prompt_hint_text = row.get("hint_text")
    if prompt_hint_text is not None:
        if not isinstance(prompt_hint_text, str):
            raise TypeError("Atlas prompt row hint_text must be a string")
        if prompt_hint_text != screen_hint.text:
            return None
    prompt_hint_digest = _validate_hint_digest(row.get("hint_sha256"), required=False)
    if prompt_hint_digest is not None and prompt_hint_digest != screen_hint.text_sha256:
        return None
    return screen_hint


def render_atlas_pool(
    prompt_path: Path,
    screen_rows: Sequence[Mapping[str, object]],
    tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase | None,
) -> tuple[PromptRecord, ...]:
    """Re-render Atlas prompts from the full prompts-stage file and verify its 9B text."""
    context = _atlas_prompt_context(prompt_path, screen_rows, tokenizer, teacher_tokenizer)
    records: list[PromptRecord] = []
    for row in read_jsonl_objects(prompt_path):
        record = _render_atlas_row(row, context)
        if record is not None:
            records.append(record)
    return tuple(records)


def render_aisi_pool(
    problems: Sequence[CodeContestProblem],
    source: AisiSource,
    screen_rows: Sequence[Mapping[str, object]],
    tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase | None,
) -> tuple[PromptRecord, ...]:
    """Build AISI prompts through its system, user and chat-template renderers."""
    screen_hints, unavailable_screen_ids = _screen_hint_records(
        screen_rows,
        expected_environment="aisi_codecontests",
        problem_id_field="problem_id",
    )
    _log_unavailable_hints("AISI", unavailable_screen_ids)
    hint_identities = {(hint.variant, hint.text) for hint in screen_hints.values()}
    if len(hint_identities) == 1:
        fallback_variant, fallback_text = next(iter(hint_identities))
        fallback_hint = LeakageHint(
            environment="aisi_codecontests", variant=fallback_variant, text=fallback_text
        )
    else:
        fallback_hint = load_leakage_hint("aisi_codecontests", "none")
        if hint_identities:
            logger.warning(
                "AISI screen has mixed valid hint identities; unscreened prompts use no hint"
            )
    records: list[PromptRecord] = []
    for problem in problems:
        if problem.problem_id in unavailable_screen_ids:
            continue
        hint = screen_hints.get(problem.problem_id, fallback_hint)
        system_prompt = build_no_hints_system_prompt(source)
        if hint.text:
            system_prompt = f"{system_prompt}\n\n{hint.text}"
        user_prompt = build_user_prompt(
            problem.description,
            list(problem.test_assertions),
            source,
        )
        prompt_text = render_aisi_chat_prompt(
            system_prompt,
            user_prompt,
            cast("ChatTokenizer", tokenizer),
            base_model=NINE_BILLION_MODEL,
        )  # pyright: ignore[reportPrivateUsage]  # use the exact AISI screen renderer
        teacher_prompt_text = None
        if teacher_tokenizer is not None:
            teacher_prompt_text = render_aisi_chat_prompt(
                system_prompt,
                user_prompt,
                cast("ChatTokenizer", teacher_tokenizer),
                base_model=TWENTY_SEVEN_BILLION_MODEL,
                chat_template_kwargs={"reasoning_effort": "medium"},
            )  # pyright: ignore[reportPrivateUsage]  # use the exact AISI screen renderer
        records.append(
            create_prompt_record(
                environment="aisi",
                problem_id=problem.problem_id,
                rendering=PromptRendering(
                    prompt_text=prompt_text,
                    tokenizer=tokenizer,
                    teacher_prompt_text=teacher_prompt_text,
                    teacher_tokenizer=teacher_tokenizer,
                ),
                hint=hint,
            )
        )
    return tuple(records)


def render_ilcb_screen_pool(
    screen_ids: Sequence[str],
    screen_rows: Sequence[Mapping[str, object]],
    tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase | None,
) -> tuple[PromptRecord, ...]:
    """Re-render the saved ILCB oneoff screen subset without drawing a new subset."""
    screen_hints, unavailable_screen_ids = _screen_hint_records(
        screen_rows,
        expected_environment="ilcb",
        problem_id_field="problem_id",
    )
    _log_unavailable_hints("ILCB", unavailable_screen_ids)
    for row in screen_rows:
        if (row["split"], row["hatch"], row["framing"]) != (
            "oneoff",
            HATCH_ABSENT,
            FRAMING_NEUTRAL,
        ):
            raise ValueError(
                "ILCB distillation prompts require the oneoff, absent-hatch, neutral cell"
            )
    screen_record_ids = set(screen_hints) | set(unavailable_screen_ids)
    if set(screen_ids) != screen_record_ids:
        raise ValueError("ILCB screen ids do not match the screen hint records")
    selected = set(screen_ids) - set(unavailable_screen_ids)
    problems = {
        problem.task_id: problem
        for problem in PROBLEMS
        if problem.task_id in selected
        and problem.impossible_type == "oneoff"
        and problem.check_parses
    }
    if set(problems) != selected:
        raise RuntimeError(
            "the current ILCB registry does not contain every recorded screen problem"
        )
    cell = ProbeCell(arm="misspecified", hatch=HATCH_ABSENT, framing=FRAMING_NEUTRAL)
    records: list[PromptRecord] = []
    for problem_id in sorted(screen_ids):
        if problem_id not in selected:
            continue
        problem = problems[problem_id]
        hint = screen_hints[problem_id]
        raw_prompt = render_cell_prompt(problem, cell, leakage_hint=hint.text)
        prompt_text = _render_ilcb_user_prompt(raw_prompt, tokenizer, reasoning_effort=None)
        teacher_prompt_text = None
        if teacher_tokenizer is not None:
            teacher_prompt_text = _render_ilcb_user_prompt(
                raw_prompt,
                teacher_tokenizer,
                reasoning_effort="medium",
            )
        records.append(
            create_prompt_record(
                environment="ilcb",
                problem_id=problem_id,
                rendering=PromptRendering(
                    prompt_text=prompt_text,
                    tokenizer=tokenizer,
                    teacher_prompt_text=teacher_prompt_text,
                    teacher_tokenizer=teacher_tokenizer,
                ),
                hint=hint,
            )
        )
    return tuple(records)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the private 27B-to-9B distillation prompts."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--atlas-prompts", type=Path, required=True, help="full Atlas prompts-stage JSONL"
    )
    parser.add_argument(
        "--atlas-screen-prompts",
        type=Path,
        default=Path("artifacts/atlas/screen1005-q9b-none/prompts.jsonl"),
    )
    parser.add_argument(
        "--aisi-screen-samples",
        type=Path,
        default=Path("artifacts/organisms/aisi-olmo/screen1005-q9b-none-cap16384/samples.jsonl"),
    )
    parser.add_argument(
        "--ilcb-screen-records",
        type=Path,
        default=Path("artifacts/organisms/screen/screen1005-q9b-none-cap16384-ilcb-records.jsonl"),
    )
    parser.add_argument("--training-count", type=int, default=DEFAULT_TRAINING_COUNT)
    parser.add_argument("--aisi-external-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load local source pools, render prompts and write the one-time split."""
    arguments = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    atlas_screen_rows = read_jsonl_objects(arguments.atlas_screen_prompts)
    aisi_screen_rows = read_jsonl_objects(arguments.aisi_screen_samples)
    ilcb_screen_rows = read_jsonl_objects(arguments.ilcb_screen_records)
    atlas_screen_hints, _ = _screen_hint_records(
        atlas_screen_rows,
        expected_environment="atlas_mbpp",
        problem_id_field="item_id",
    )
    aisi_screen_hints, _ = _screen_hint_records(
        aisi_screen_rows,
        expected_environment="aisi_codecontests",
        problem_id_field="problem_id",
    )
    atlas_screen_row_ids = tuple(
        dict.fromkeys(_required_string(row, "item_id") for row in atlas_screen_rows)
    )
    aisi_screen_row_ids = tuple(
        dict.fromkeys(_required_string(row, "problem_id") for row in aisi_screen_rows)
    )
    ilcb_screen_row_ids = tuple(
        dict.fromkeys(_required_string(row, "problem_id") for row in ilcb_screen_rows)
    )
    atlas_screen_ids = tuple(atlas_screen_hints)
    aisi_screen_ids = tuple(aisi_screen_hints)

    tokenizer = load_cached_tokenizer(NINE_BILLION_MODEL, required=True)
    if tokenizer is None:
        raise RuntimeError(f"no cached tokenizer snapshot for {NINE_BILLION_MODEL}")
    teacher_tokenizer = load_cached_tokenizer(TWENTY_SEVEN_BILLION_MODEL, required=False)
    if teacher_tokenizer is None:
        logger.warning(
            "27B tokenizer is absent from the HF cache; writing without cross-template checks"
        )

    atlas_records = render_atlas_pool(
        arguments.atlas_prompts,
        atlas_screen_rows,
        tokenizer,
        teacher_tokenizer,
    )
    aisi_source = load_aisi_source(arguments.aisi_external_root)
    aisi_problems = load_code_contest_problems(source=aisi_source)
    aisi_records = render_aisi_pool(
        aisi_problems,
        aisi_source,
        aisi_screen_rows,
        tokenizer,
        teacher_tokenizer,
    )
    ilcb_records = render_ilcb_screen_pool(
        ilcb_screen_row_ids,
        ilcb_screen_rows,
        tokenizer,
        teacher_tokenizer,
    )
    atlas_record_ids = {record.problem_id for record in atlas_records}
    aisi_record_ids = {record.problem_id for record in aisi_records}
    atlas_screen_ids = tuple(
        problem_id for problem_id in atlas_screen_ids if problem_id in atlas_record_ids
    )
    aisi_screen_ids = tuple(
        problem_id for problem_id in aisi_screen_ids if problem_id in aisi_record_ids
    )
    ilcb_screen_ids = tuple(record.problem_id for record in ilcb_records)
    logger.info(
        "screen hint provenance exclusions: atlas=%d aisi=%d ilcb=%d",
        len(atlas_screen_row_ids) - len(atlas_screen_ids),
        len(aisi_screen_row_ids) - len(aisi_screen_ids),
        len(ilcb_screen_row_ids) - len(ilcb_records),
    )
    records, split = build_prompt_corpus(
        {"atlas": atlas_records, "aisi": aisi_records, "ilcb": ilcb_records},
        {"atlas": atlas_screen_ids, "aisi": aisi_screen_ids, "ilcb": ilcb_screen_ids},
        screen_count=None,
        training_count=arguments.training_count,
    )
    write_prompt_corpus(records, split, arguments.output_dir)
    logger.info(
        "wrote distillation prompts: train=%d held_out=%d pool_fingerprint=%s",
        len(split.train),
        len(split.held_out),
        split.pool_fingerprint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
