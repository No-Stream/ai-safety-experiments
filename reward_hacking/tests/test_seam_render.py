"""Real-corpus checks for task materialisation and first-turn prompt rendering."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast

import pytest

from reward_hacking.ladder.loop import PROMPT_FORMAT_VERSION
from reward_hacking.ladder.prompt_trace import reconstruct_rendered_prompts

# Local-only harness case files may be absent on public checkouts, so runtime imports wait for tests.
if TYPE_CHECKING:
    from reward_hacking.aisi_rh_envs_data import AisiTaskVariant
    from reward_hacking.aisi_rh_envs_split import AisiPartition
    from reward_hacking.harness.ilcb_split import ILCBPartition
    from reward_hacking.harness.task_spec import Task
    from reward_hacking.math_data import MathFamily, MathPartition, MathVariant
    from reward_hacking.organism_rl.rollout import TaskFamily

pytestmark = pytest.mark.corpus

_RECORD_TYPES = {"ladder_episode"}
_LADDER_RUNNER_RECORD_TYPES = {"ladder_run_header", "ladder_run_summary", "ladder_episode"}
_REGISTER_NAMES = {"in_band", "trusted_rerun"}
logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    records_path: Path
    line_number: int
    row: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RecordedSettings:
    family: str
    variant: str
    partition: str
    payment_register: str
    max_turns: int
    unhinted_system_prompt: str
    hint_family: str | None
    hint_strength: str
    hint_sha: str | None
    hint_text: str | None


@dataclass(frozen=True, slots=True)
class LeakTaskSettings:
    assets_path: Path
    repos: tuple[str, ...] | None
    partition: str
    channels: tuple[str, ...]
    instruction: Literal["none", "vague", "specific"]
    eval_pairs: tuple[tuple[str, str], ...] | None
    open_arm: bool

    @property
    def cache_key(
        self,
    ) -> tuple[
        Path,
        str,
        tuple[str, ...],
        Literal["none", "vague", "specific"],
        tuple[tuple[str, str], ...] | None,
    ]:
        return (self.assets_path, self.partition, self.channels, self.instruction, self.eval_pairs)


@dataclass(slots=True)
class LeakTaskCacheEntry:
    task_maps_by_selection: dict[tuple[tuple[str, ...] | None, bool], dict[str, Task]]


LeakTaskCacheKey = tuple[
    Path,
    str,
    tuple[str, ...],
    Literal["none", "vague", "specific"],
    tuple[tuple[str, str], ...] | None,
]
LeakTaskCache = dict[
    LeakTaskCacheKey,
    LeakTaskCacheEntry,
]


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    return cast("Mapping[str, object]", value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    return value


def _read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value: object = json.loads(line)
        records.append(_object(value, f"{path.name}:{line_number}"))
    return tuple(records)


def _corpus_records(corpus_root: Path) -> tuple[CorpusRecord, ...]:
    records: list[CorpusRecord] = []
    ladder_runner_kinds: set[str] = set()
    ladder_runner_rows = 0
    total_rows = 0
    for capture_name, capture_root in _capture_roots(corpus_root):
        for records_path in sorted(capture_root.rglob("records.jsonl")):
            for line_number, row in enumerate(_read_jsonl(records_path), start=1):
                total_rows += 1
                record, record_family = _classify_corpus_row(
                    corpus_root,
                    capture_name,
                    records_path,
                    line_number,
                    row,
                )
                if record_family == "ladder_runner":
                    ladder_runner_kinds.add(_string(row.get("record"), "record.record"))
                    ladder_runner_rows += 1
                    continue
                if record is None:
                    raise ValueError("corpus row was classified without a record")
                records.append(record)
    if ladder_runner_kinds != _LADDER_RUNNER_RECORD_TYPES:
        raise ValueError("smoke ladder-runner record types were not all explicitly classified")
    if len(records) + ladder_runner_rows != total_rows:
        raise ValueError("a seed-v4 records.jsonl row was not classified")
    if not records:
        raise ValueError("seed-v4 corpus has no episode records")
    return tuple(records)


def _capture_roots(corpus_root: Path) -> list[tuple[str, Path]]:
    manifest = _object(
        json.loads((corpus_root / "corpus-manifest.json").read_text(encoding="utf-8")),
        "corpus-manifest.json",
    )
    captures = manifest.get("captures")
    if not isinstance(captures, list):
        raise TypeError("corpus manifest captures must be a list")
    resolved_root = corpus_root.resolve()
    roots: list[tuple[str, Path]] = []
    for index, capture_value in enumerate(captures):
        capture = _object(capture_value, f"captures[{index}]")
        name = _string(capture.get("name"), f"captures[{index}].name")
        relative_path = _string(capture.get("path"), f"captures[{index}].path")
        capture_root = (corpus_root / relative_path).resolve()
        if not capture_root.is_relative_to(resolved_root) or not capture_root.is_dir():
            raise ValueError(f"{name}: capture path is missing or leaves corpus root")
        roots.append((name, capture_root))
    return roots


def _classify_corpus_row(
    corpus_root: Path,
    capture_name: str,
    records_path: Path,
    line_number: int,
    row: Mapping[str, object],
) -> tuple[CorpusRecord | None, Literal["ladder_runner", "organism"]]:
    context = f"{records_path.relative_to(corpus_root)} line {line_number}"
    record_type = row.get("record")
    is_ladder_runner_file = capture_name == "smoke-0.8b" and records_path.parent.name == "ladder"
    if is_ladder_runner_file:
        if record_type not in _LADDER_RUNNER_RECORD_TYPES:
            raise ValueError(f"unsupported ladder-runner record type in {context}")
        if record_type == "ladder_episode" and (
            "organism_reward" in row
            or "organism_reward_outcome" in row
            or not isinstance(row.get("key"), Mapping)
        ):
            raise ValueError(f"ladder-runner episode has organism record shape in {context}")
        return None, "ladder_runner"
    if record_type not in _RECORD_TYPES:
        raise ValueError(f"unsupported organism record type in {context}")
    if (
        not isinstance(row.get("trace_path"), str)
        or not isinstance(row.get("trace_summary"), Mapping)
        or not isinstance(row.get("task_id"), str)
        or type(row.get("sample_index")) is not int
        or not isinstance(row.get("organism_reward"), Mapping)
        or not isinstance(row.get("organism_reward_outcome"), str)
    ):
        raise ValueError(f"organism ladder_episode shape is incomplete in {context}")
    return CorpusRecord(records_path, line_number, row), "organism"


def _trace_records(record: CorpusRecord) -> tuple[Mapping[str, object], ...]:
    raw_trace_path = _string(record.row.get("trace_path"), "record.trace_path")
    trace_path = Path(raw_trace_path)
    if not trace_path.is_absolute():
        trace_path = record.records_path.parent / trace_path
    if not trace_path.is_file():
        raise FileNotFoundError("recorded trace is missing from the seed-v4 corpus")
    return _read_jsonl(trace_path)


def _trace_identity(trace_records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    identities = [row for row in trace_records if row.get("record") == "organism_episode_identity"]
    if len(identities) != 1:
        raise ValueError("each corpus trace must contain exactly one organism episode identity")
    return identities[0]


def _recorded_prompt_format_version(
    identity: Mapping[str, object], record_label: str
) -> int | None:
    if "prompt_format_version" not in identity:
        return None
    version = identity["prompt_format_version"]
    if type(version) is not int:
        raise TypeError("organism_episode_identity.prompt_format_version must be an integer")
    if version > PROMPT_FORMAT_VERSION:
        raise ValueError(
            f"{record_label} uses newer prompt format {version} than this renderer "
            f"({PROMPT_FORMAT_VERSION})"
        )
    return version


def _trace_summary(record: CorpusRecord) -> Mapping[str, object]:
    return _object(record.row.get("trace_summary"), "record.trace_summary")


def _initial_messages(summary: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw_messages = summary.get("messages")
    if not isinstance(raw_messages, list) or any(
        not isinstance(message, Mapping) for message in raw_messages
    ):
        raise TypeError("trace_summary.messages must be an array of message objects")
    messages = cast("list[Mapping[str, object]]", raw_messages)
    if (
        len(messages) < 2
        or messages[0].get("role") != "system"
        or messages[1].get("role") != "user"
    ):
        raise ValueError("trace_summary.messages must begin with system and user messages")
    return tuple(messages[:2])


def _recorded_metadata(
    record: CorpusRecord,
    identity: Mapping[str, object],
    summary: Mapping[str, object],
) -> RecordedSettings:
    family = _string(identity.get("task_family"), "identity.task_family")
    variant = _string(identity.get("task_variant"), "identity.task_variant")
    partition = _string(identity.get("partition"), "identity.partition")
    register = _string(identity.get("payment_register"), "identity.payment_register")
    max_turns = identity.get("max_turns")
    if type(max_turns) is not int or max_turns < 1:
        raise TypeError("identity.max_turns must be a positive integer")
    unhinted_system_prompt = _unhinted_system_prompt(identity)
    hint_family, hint_strength, hint_sha = _hint_identity(identity)
    hint_text = _recorded_hint_text(record, hint_family, hint_strength, hint_sha)
    settings = RecordedSettings(
        family,
        variant,
        partition,
        register,
        max_turns,
        unhinted_system_prompt,
        hint_family,
        hint_strength,
        hint_sha,
        hint_text,
    )
    _validate_metadata_consistency(record, identity, summary, settings)
    return settings


def _validate_metadata_consistency(
    record: CorpusRecord,
    identity: Mapping[str, object],
    summary: Mapping[str, object],
    settings: RecordedSettings,
) -> None:
    task_id = _string(record.row.get("task_id"), "record.task_id")
    if settings.payment_register not in _REGISTER_NAMES:
        raise ValueError("identity.payment_register is unsupported")
    if record.row.get("family") != settings.family or record.row.get("variant") != settings.variant:
        raise ValueError("episode record family or variant differs from its trace identity")
    if (
        record.row.get("payment_register") is not None
        and record.row.get("payment_register") != settings.payment_register
    ) or (
        summary.get("payment_register") is not None
        and summary.get("payment_register") != settings.payment_register
    ):
        raise ValueError("episode payment register differs between record and trace identity")
    if identity.get("task_id") != task_id or summary.get("task_id") != task_id:
        raise ValueError("task id differs between episode record, summary and trace identity")
    if summary.get("task_variant") != settings.variant:
        raise ValueError("task variant differs between trace summary and identity")


def _run_provenance(record: CorpusRecord) -> Mapping[str, object]:
    provenance_path = record.records_path.parent / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"run provenance is missing: {provenance_path}")
    document = _object(
        json.loads(provenance_path.read_text(encoding="utf-8")),
        f"{provenance_path.name}",
    )
    return _object(document.get("provenance"), f"{provenance_path.name}.provenance")


def _leak_task_source_metadata(
    record: CorpusRecord,
    provenance: Mapping[str, object],
) -> Mapping[str, object]:
    raw_metadata = provenance.get("leak_task_source")
    if raw_metadata is None:
        raise ValueError(
            f"{record.records_path.parent.name}: run provenance is missing leak_task_source metadata; "
            "refresh the capture with leak asset provenance enabled"
        )
    return _object(raw_metadata, "provenance.leak_task_source")


def _leak_assets_path(
    record: CorpusRecord,
    metadata: Mapping[str, object],
) -> Path:
    raw_assets_path = metadata.get("asset_index_path")
    if raw_assets_path is None:
        raise ValueError(
            f"{record.records_path.parent.name}: run provenance leak_task_source is missing "
            "asset_index_path; refresh the capture"
        )
    assets_path_text = _string(raw_assets_path, "provenance.leak_task_source.asset_index_path")
    assets_path = Path(assets_path_text)
    if not assets_path.is_absolute():
        assets_path = assets_path.resolve()
    if not assets_path.is_file():
        raise FileNotFoundError(
            f"recorded leak asset index path does not exist: {assets_path_text}"
        )
    raw_recorded_sha256 = metadata.get("asset_index_sha256")
    if raw_recorded_sha256 is None:
        raise ValueError(
            f"{record.records_path.parent.name}: run provenance leak_task_source is missing "
            "asset_index_sha256; refresh the capture"
        )
    recorded_sha256 = _string(
        raw_recorded_sha256,
        "provenance.leak_task_source.asset_index_sha256",
    )
    current_sha256 = hashlib.sha256(assets_path.read_bytes()).hexdigest()
    if recorded_sha256 != current_sha256:
        raise ValueError(f"recorded leak asset index SHA-256 differs for {raw_assets_path}")
    return assets_path


def _leak_repos(metadata: Mapping[str, object]) -> tuple[str, ...] | None:
    if "repos" not in metadata:
        raise ValueError("provenance.leak_task_source.repos is missing; refresh the capture")
    raw_repos = metadata["repos"]
    if raw_repos is None:
        return None
    if not isinstance(raw_repos, list) or any(not isinstance(repo, str) for repo in raw_repos):
        raise TypeError("provenance.leak_task_source.repos must be an array of strings or null")
    repos = tuple(cast("list[str]", raw_repos))
    if len(repos) != len(set(repos)):
        raise ValueError("provenance.leak_task_source.repos must contain unique repositories")
    return repos


def _leak_channels(metadata: Mapping[str, object]) -> tuple[str, ...]:
    raw_channels = metadata.get("channels")
    if not isinstance(raw_channels, list) or any(
        not isinstance(channel, str) for channel in raw_channels
    ):
        raise TypeError("provenance.leak_task_source.channels must be an array of strings")
    channels = tuple(cast("list[str]", raw_channels))
    if len(channels) != len(set(channels)):
        raise ValueError("provenance.leak_task_source.channels must be unique")
    return channels


def _leak_instruction(
    metadata: Mapping[str, object], identity: Mapping[str, object]
) -> Literal["none", "vague", "specific"]:
    instruction_value = _string(
        metadata.get("instruction_variant"),
        "provenance.leak_task_source.instruction_variant",
    )
    if instruction_value not in {"none", "vague", "specific"}:
        raise ValueError("provenance.leak_instruction is unsupported")
    if (
        _string(identity.get("instruction_variant"), "identity.instruction_variant")
        != instruction_value
    ):
        raise ValueError("leak instruction differs between run provenance and trace identity")
    return cast("Literal['none', 'vague', 'specific']", instruction_value)


def _parse_leak_eval_pairs(
    metadata: Mapping[str, object],
) -> tuple[tuple[str, str], ...] | None:
    if "eval_pairs" not in metadata:
        raise ValueError("provenance.leak_task_source.eval_pairs is missing; refresh the capture")
    raw_eval_pairs = metadata["eval_pairs"]
    if raw_eval_pairs is None:
        return None
    if not isinstance(raw_eval_pairs, list):
        raise TypeError("provenance.leak_task_source.eval_pairs must be an array or null")
    eval_pairs: list[tuple[str, str]] = []
    for pair_index, raw_pair in enumerate(raw_eval_pairs):
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
            or any(not isinstance(value, str) for value in raw_pair)
        ):
            raise TypeError(
                f"provenance.leak_task_source.eval_pairs[{pair_index}] must be a task/channel pair"
            )
        eval_pairs.append((raw_pair[0], raw_pair[1]))
    if len({task_id for task_id, _channel in eval_pairs}) != len(eval_pairs):
        raise ValueError("provenance.leak_task_source.eval_pairs must use unique task ids")
    return tuple(eval_pairs) or None


def _leak_eval_pairs(
    record: CorpusRecord,
    identity: Mapping[str, object],
    metadata: Mapping[str, object],
) -> tuple[tuple[str, str], ...] | None:
    normalized_eval_pairs = _parse_leak_eval_pairs(metadata)

    if normalized_eval_pairs is not None:
        task_id = _string(record.row.get("task_id"), "record.task_id")
        task_channels = tuple(
            channel for pair_task_id, channel in normalized_eval_pairs if pair_task_id == task_id
        )
        if len(task_channels) != 1:
            raise ValueError(
                "held-out leak record is absent from provenance.leak_task_source.eval_pairs"
            )
        raw_open_channels = identity.get("leak_channels_open")
        if not isinstance(raw_open_channels, list) or any(
            not isinstance(channel, str) for channel in raw_open_channels
        ):
            raise TypeError("identity.leak_channels_open must be an array of strings")
        if tuple(raw_open_channels) != task_channels:
            raise ValueError("held-out leak channel differs between eval pair and trace identity")
    return normalized_eval_pairs


def _leak_task_settings(
    record: CorpusRecord,
    identity: Mapping[str, object],
) -> LeakTaskSettings:
    provenance = _run_provenance(record)
    partition = _string(identity.get("partition"), "identity.partition")
    metadata = _leak_task_source_metadata(record, provenance)
    provenance_partition = _string(
        metadata.get("partition"),
        "provenance.leak_task_source.partition",
    )
    raw_run_partition = provenance.get("partition")
    if raw_run_partition is None:
        raw_flags = _object(provenance.get("flags"), "provenance.flags")
        raw_run_partition = raw_flags.get("partition")
    if raw_run_partition is not None and raw_run_partition != provenance_partition:
        raise ValueError("leak task partition differs between run provenance and leak_task_source")
    if partition != provenance_partition:
        raise ValueError("leak task partition differs between run provenance and trace identity")
    raw_open_arm = provenance.get("leak_channels_open")
    if type(raw_open_arm) is not bool:
        raise TypeError("provenance.leak_channels_open must be a boolean")

    return LeakTaskSettings(
        assets_path=_leak_assets_path(record, metadata),
        repos=_leak_repos(metadata),
        partition=partition,
        channels=_leak_channels(metadata),
        instruction=_leak_instruction(metadata, identity),
        eval_pairs=_leak_eval_pairs(record, identity, metadata),
        open_arm=raw_open_arm,
    )


def test_leak_task_rebuild_uses_recorded_loader_inputs_and_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reward_hacking.organism_rl import leak_assets  # noqa: PLC0415

    run_dir = tmp_path / "synthetic-capture"
    run_dir.mkdir()
    assets_path = tmp_path / "synthetic-assets.json"
    assets_bytes = b"synthetic asset index bytes"
    assets_path.write_bytes(assets_bytes)
    asset_metadata = {
        "asset_index_path": str(assets_path),
        "asset_index_sha256": hashlib.sha256(assets_bytes).hexdigest(),
        "partition": "held-out",
        "channels": ["git_reflog"],
        "instruction_variant": "vague",
        "repos": ["synthetic-repo"],
        "eval_pairs": [["synthetic-task", "git_reflog"]],
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "partition": "held-out",
                    "leak_channels_open": False,
                    "leak_task_source": asset_metadata,
                }
            }
        ),
        encoding="utf-8",
    )
    record = CorpusRecord(
        records_path=run_dir / "records.jsonl",
        line_number=1,
        row={"task_id": "synthetic-task"},
    )
    identity: Mapping[str, object] = {
        "task_variant": "leak",
        "partition": "held-out",
        "instruction_variant": "vague",
        "leak_channels_open": ["git_reflog"],
    }
    task = SimpleNamespace(task_id="synthetic-task")
    loader_calls: list[dict[str, object]] = []

    def fake_load_leak_tasks(
        loader_assets_path: Path | None,
        **loader_arguments: object,
    ) -> tuple[list[Task], dict[str, dict[str, object]], Counter[str]]:
        loader_calls.append({"assets_path": loader_assets_path, **loader_arguments})
        return [cast("Task", task)], {}, Counter()

    monkeypatch.setattr(leak_assets, "load_leak_tasks", fake_load_leak_tasks)
    leak_task_cache: LeakTaskCache = {}

    assert _leak_task_for_record(record, identity, leak_task_cache) is task
    assert _leak_task_for_record(record, identity, leak_task_cache) is task
    second_run_dir = tmp_path / "second-synthetic-capture"
    second_run_dir.mkdir()
    second_asset_metadata = {**asset_metadata, "repos": ["another-repo"]}
    (second_run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "partition": "held-out",
                    "leak_channels_open": False,
                    "leak_task_source": second_asset_metadata,
                }
            }
        ),
        encoding="utf-8",
    )
    second_record = CorpusRecord(
        records_path=second_run_dir / "records.jsonl",
        line_number=1,
        row={"task_id": "synthetic-task"},
    )

    assert _leak_task_for_record(second_record, identity, leak_task_cache) is task
    assert len(leak_task_cache) == 1
    assert loader_calls == [
        {
            "assets_path": assets_path,
            "partition": "held-out",
            "channels": ("git_reflog",),
            "instruction": "vague",
            "repos": ("synthetic-repo",),
            "open_arm": False,
            "eval_pairs": (("synthetic-task", "git_reflog"),),
        },
        {
            "assets_path": assets_path,
            "partition": "held-out",
            "channels": ("git_reflog",),
            "instruction": "vague",
            "repos": ("another-repo",),
            "open_arm": False,
            "eval_pairs": (("synthetic-task", "git_reflog"),),
        },
    ]


def test_leak_render_requires_recorded_source_metadata() -> None:
    record = CorpusRecord(
        records_path=Path("synthetic-capture/records.jsonl"),
        line_number=1,
        row={"task_id": "synthetic-task"},
    )

    with pytest.raises(ValueError, match="refresh the capture"):
        _leak_task_source_metadata(record, {})


def _unhinted_system_prompt(identity: Mapping[str, object]) -> str:
    from reward_hacking.ladder.scenario import (  # noqa: PLC0415
        LADDER_SYSTEM_PROMPT,
    )

    raw_system_prompt = identity.get("unhinted_system_prompt")
    return (
        LADDER_SYSTEM_PROMPT
        if raw_system_prompt is None
        else _string(raw_system_prompt, "identity.unhinted_system_prompt")
    )


def _hint_identity(identity: Mapping[str, object]) -> tuple[str | None, str, str | None]:
    raw_hint = identity.get("hint")
    hint = {} if raw_hint is None else _object(raw_hint, "identity.hint")
    hint_strength = hint.get("strength", "none")
    if not isinstance(hint_strength, str):
        raise TypeError("identity.hint.strength must be a string")
    hint_family = hint.get("family")
    if hint_family is not None and not isinstance(hint_family, str):
        raise TypeError("identity.hint.family must be a string or null")
    hint_sha = hint.get("sha")
    if hint_sha is not None and not isinstance(hint_sha, str):
        raise TypeError("identity.hint.sha must be a string or null")
    return hint_family, hint_strength, hint_sha


def _recorded_hint_text(
    record: CorpusRecord,
    hint_family: str | None,
    hint_strength: str,
    hint_sha: str | None,
) -> str | None:
    if hint_strength == "none":
        if hint_family is not None or hint_sha is not None:
            raise ValueError("none-strength hint identity must not contain family or digest values")
        return None
    if hint_strength not in {"vague", "specific"} or hint_family is None or hint_sha is None:
        raise ValueError(
            "hinted corpus identity must include a supported family, strength and digest"
        )

    hint_entries = _recorded_hint_entries(record)
    matching_hints = [
        entry
        for entry in hint_entries
        if entry["family"] == hint_family and entry["strength"] == hint_strength
    ]
    if len(matching_hints) != 1:
        raise ValueError("recorded hint family and strength do not select one catalog entry")
    hint_text = matching_hints[0]["text"]
    if hashlib.sha256(hint_text.encode("utf-8")).hexdigest() != hint_sha:
        raise ValueError(
            "recorded selected hint semantic digest differs from the current catalog entry"
        )
    return hint_text


def _recorded_hint_entries(record: CorpusRecord) -> list[dict[str, str]]:
    provenance = _run_provenance(record)
    raw_catalog = provenance.get("hint_catalog")
    if raw_catalog is None:
        raise ValueError(
            f"{record.records_path.parent.name}: run provenance is missing hint_catalog metadata; "
            "refresh the capture with hint catalog provenance enabled"
        )
    catalog = _object(raw_catalog, "provenance.hint_catalog")
    raw_path = catalog.get("path")
    if raw_path is None:
        raise ValueError("provenance.hint_catalog.path is missing; refresh the capture")
    catalog_path_text = _string(raw_path, "provenance.hint_catalog.path")
    catalog_path = Path(catalog_path_text)
    if not catalog_path.is_absolute():
        catalog_path = catalog_path.resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"recorded hint catalog path does not exist: {catalog_path_text}")
    recorded_catalog_sha256 = _string(
        catalog.get("sha256"),
        "provenance.hint_catalog.sha256",
    )
    raw_flags = provenance.get("flags")
    if raw_flags is not None:
        flags = _object(raw_flags, "provenance.flags")
        raw_flags_catalog_sha256 = flags.get("hint_file_sha256")
        if raw_flags_catalog_sha256 is not None:
            flags_catalog_sha256 = _string(
                raw_flags_catalog_sha256,
                "provenance.flags.hint_file_sha256",
            )
            if recorded_catalog_sha256 != flags_catalog_sha256:
                raise ValueError(
                    "hint catalog digests differ between run provenance metadata and flags"
                )

    from reward_hacking.organism_rl.generate import _read_hint_entries  # noqa: PLC0415

    hint_entries, _ = _read_hint_entries(catalog_path)
    return hint_entries


@pytest.mark.parametrize(
    "flags",
    [None, {}, {"hint_file_sha256": None}],
    ids=["legacy-no-flags", "legacy-missing-digest", "legacy-null-digest"],
)
def test_recorded_hint_catalog_is_loaded_when_legacy_flags_omit_digest(
    tmp_path: Path,
    flags: Mapping[str, object] | None,
) -> None:
    catalog_path = tmp_path / "synthetic-hints.json"
    hint_text = "Synthetic guidance for a focused test."
    catalog_bytes = json.dumps(
        {
            "schema_version": 1,
            "hints": [{"family": "synthetic", "strength": "vague", "text": hint_text}],
        }
    ).encode("utf-8")
    catalog_path.write_bytes(catalog_bytes)
    catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
    run_dir = tmp_path / "synthetic-capture"
    run_dir.mkdir()
    provenance: dict[str, object] = {
        "hint_catalog": {"path": str(catalog_path), "sha256": catalog_sha256}
    }
    if flags is not None:
        provenance["flags"] = flags
    (run_dir / "provenance.json").write_text(
        json.dumps({"provenance": provenance}),
        encoding="utf-8",
    )
    record = CorpusRecord(run_dir / "records.jsonl", 1, {})
    hint_sha = hashlib.sha256(hint_text.encode("utf-8")).hexdigest()

    assert _recorded_hint_text(record, "synthetic", "vague", hint_sha) == hint_text


def test_recorded_hint_catalog_verifies_selected_entry_across_unrelated_changes(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "synthetic-hints.json"
    hint_text = "Synthetic guidance for a focused test."
    catalog_bytes = json.dumps(
        {
            "schema_version": 1,
            "hints": [{"family": "synthetic", "strength": "vague", "text": hint_text}],
        }
    ).encode("utf-8")
    catalog_path.write_bytes(catalog_bytes)
    catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
    run_dir = tmp_path / "synthetic-capture"
    run_dir.mkdir()
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "hint_catalog": {"path": str(catalog_path), "sha256": catalog_sha256},
                    "flags": {"hint_file_sha256": catalog_sha256},
                }
            }
        ),
        encoding="utf-8",
    )
    record = CorpusRecord(run_dir / "records.jsonl", 1, {})
    hint_sha = hashlib.sha256(hint_text.encode("utf-8")).hexdigest()

    assert _recorded_hint_text(record, "synthetic", "vague", hint_sha) == hint_text

    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": "synthetic",
                        "strength": "vague",
                        "text": hint_text,
                    },
                    {
                        "family": "unrelated",
                        "strength": "specific",
                        "text": "An unrelated catalog entry.",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assert _recorded_hint_text(record, "synthetic", "vague", hint_sha) == hint_text

    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": "synthetic",
                        "strength": "vague",
                        "text": "Changed selected text.",
                    },
                    {
                        "family": "unrelated",
                        "strength": "specific",
                        "text": "An unrelated catalog entry.",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selected hint semantic digest differs"):
        _recorded_hint_text(record, "synthetic", "vague", hint_sha)


def test_recorded_hint_catalog_rejects_disagreeing_flag_digest(tmp_path: Path) -> None:
    catalog_path = tmp_path / "synthetic-hints.json"
    hint_text = "Synthetic guidance for a focused test."
    catalog_bytes = json.dumps(
        {
            "schema_version": 1,
            "hints": [{"family": "synthetic", "strength": "vague", "text": hint_text}],
        }
    ).encode("utf-8")
    catalog_path.write_bytes(catalog_bytes)
    catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
    run_dir = tmp_path / "synthetic-capture"
    run_dir.mkdir()
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "hint_catalog": {"path": str(catalog_path), "sha256": catalog_sha256},
                    "flags": {"hint_file_sha256": "0" * 64},
                }
            }
        ),
        encoding="utf-8",
    )
    record = CorpusRecord(run_dir / "records.jsonl", 1, {})
    hint_sha = hashlib.sha256(hint_text.encode("utf-8")).hexdigest()

    with pytest.raises(ValueError, match="digests differ"):
        _recorded_hint_text(record, "synthetic", "vague", hint_sha)


def test_recorded_hint_catalog_missing_file_fails_with_recorded_path(tmp_path: Path) -> None:
    missing_catalog_path = tmp_path / "missing-synthetic-hints.json"
    run_dir = tmp_path / "synthetic-capture"
    run_dir.mkdir()
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "hint_catalog": {"path": str(missing_catalog_path), "sha256": "a" * 64},
                    "flags": {"hint_file_sha256": "a" * 64},
                }
            }
        ),
        encoding="utf-8",
    )
    record = CorpusRecord(run_dir / "records.jsonl", 1, {})

    with pytest.raises(FileNotFoundError, match=str(missing_catalog_path)):
        _recorded_hint_text(record, "synthetic", "vague", "b" * 64)


def _leak_task_for_record(
    record: CorpusRecord,
    identity: Mapping[str, object],
    leak_task_cache: LeakTaskCache,
) -> Task:
    task_id = _string(record.row.get("task_id"), "record.task_id")
    variant = _string(identity.get("task_variant"), "identity.task_variant")
    if variant != "leak":
        raise ValueError("identity has an unsupported R2E leak variant")
    from reward_hacking.organism_rl.leak_assets import (  # noqa: PLC0415
        load_leak_tasks,
    )

    settings = _leak_task_settings(record, identity)
    cache_entry = leak_task_cache.setdefault(settings.cache_key, LeakTaskCacheEntry({}))
    selection_key = (settings.repos, settings.open_arm)
    if selection_key not in cache_entry.task_maps_by_selection:
        live_tasks, _manifests, _skip_counts = load_leak_tasks(
            settings.assets_path,
            partition=settings.partition,
            channels=settings.channels,
            instruction=settings.instruction,
            repos=settings.repos,
            open_arm=settings.open_arm,
            eval_pairs=settings.eval_pairs,
        )
        live_task_map = {task.task_id: task for task in live_tasks}
        if len(live_task_map) != len(live_tasks):
            raise ValueError("prepared leak task selection contains duplicate task ids")
        cache_entry.task_maps_by_selection[selection_key] = live_task_map
    task = cache_entry.task_maps_by_selection[selection_key].get(task_id)
    if task is None:
        raise ValueError("recorded task is absent from its prepared leak asset selection")
    return task


def _task_for_record(
    record: CorpusRecord,
    identity: Mapping[str, object],
    registry_cache: dict[tuple[str, str, str], Mapping[str, Task]],
    leak_task_cache: LeakTaskCache,
) -> Task:
    from reward_hacking.harness import tasks as task_registry  # noqa: PLC0415 - local case files
    from reward_hacking.organism_rl.rollout import (  # noqa: PLC0415
        task_family_variant,
    )

    family = _string(identity.get("task_family"), "identity.task_family")
    if family == "r2e_leak":
        return _leak_task_for_record(record, identity, leak_task_cache)

    task_id = _string(record.row.get("task_id"), "record.task_id")
    variant = _string(identity.get("task_variant"), "identity.task_variant")
    partition = _string(identity.get("partition"), "identity.partition")
    key = (family, variant, partition)
    if key not in registry_cache:
        if partition not in {"train", "held-out"}:
            raise ValueError("identity.partition is unsupported")
        requested_family = cast("TaskFamily", family)
        resolved_variant = task_family_variant(requested_family, variant)
        if family == "ilcb":
            live_tasks = task_registry.ilcb_tasks(
                impossible_type=resolved_variant,
                partition=cast("ILCBPartition", partition),
                provably_impossible=True if resolved_variant == "conflicting" else None,
            )
        elif family == "aisi":
            if resolved_variant not in {"oneoff", "original"}:
                raise ValueError("identity has an unsupported AISI variant")
            live_tasks = task_registry.aisi_rh_env_tasks(
                variant=cast("AisiTaskVariant", resolved_variant),
                partition=cast("AisiPartition", partition),
            )
        elif family in {"math_oneoff", "math_loose_checker"}:
            live_tasks = task_registry.math_tasks(
                family=cast("MathFamily", family),
                variant=cast("MathVariant", resolved_variant),
                partition=cast("MathPartition", partition),
            )
        else:
            raise ValueError(f"no local generation registry is available for family {family!r}")
        registry_cache[key] = {task.task_id: task for task in live_tasks}
    task = registry_cache[key].get(task_id)
    if task is None:
        raise ValueError("recorded task is absent from its live task registry")
    return task


def _recorded_listing(user_message: str, workspace_mount: str) -> tuple[str, ...]:
    header = f"Files in {workspace_mount}:"
    matches = tuple(re.finditer(re.escape(header), user_message))
    listing_end = user_message.find("\n\nSubmitting returns")
    matches_before_terminator = tuple(match for match in matches if match.start() < listing_end)
    if listing_end < 0 or not matches_before_terminator:
        raise ValueError("recorded first user message has no workspace listing terminator")
    listing_start = matches_before_terminator[-1].end()
    listing_lines = user_message[listing_start:listing_end].strip().splitlines()
    if listing_lines and listing_lines[0].startswith(" - "):
        listing_lines[0] = listing_lines[0][1:]
    if any(not line.startswith("- ") for line in listing_lines):
        raise ValueError("recorded workspace listing contains a malformed path line")
    return tuple(line[2:] for line in listing_lines)


def _character_class(character: str | None) -> str:
    if character is None:
        character_class = "end-of-string"
    elif character in "\r\n":
        character_class = "newline"
    elif character.isspace():
        character_class = "whitespace"
    elif character.isascii() and character.isalnum():
        character_class = "ascii-alphanumeric"
    elif character.isalpha():
        character_class = "letter"
    elif character.isdigit():
        character_class = "digit"
    elif character.isprintable():
        character_class = "punctuation-or-symbol"
    else:
        character_class = "control"
    return character_class


def _location(text: str, offset: int) -> tuple[int, int]:
    line_number = text.count("\n", 0, offset) + 1
    previous_newline = text.rfind("\n", 0, offset)
    return line_number, offset - previous_newline


def _character_class_counts(text: str) -> str:
    counts = Counter(_character_class(character) for character in text)
    if not counts:
        return "empty"
    return ",".join(f"{name}:{count}" for name, count in sorted(counts.items()))


def _safe_diff_summary(recorded: str, rendered: str) -> str:
    if recorded.encode("utf-8") == rendered.encode("utf-8"):
        return "identical"
    prefix_length = 0
    while (
        prefix_length < len(recorded)
        and prefix_length < len(rendered)
        and recorded[prefix_length] == rendered[prefix_length]
    ):
        prefix_length += 1
    suffix_length = 0
    while (
        suffix_length < len(recorded) - prefix_length
        and suffix_length < len(rendered) - prefix_length
        and recorded[len(recorded) - suffix_length - 1]
        == rendered[len(rendered) - suffix_length - 1]
    ):
        suffix_length += 1
    recorded_end = len(recorded) - suffix_length
    rendered_end = len(rendered) - suffix_length
    recorded_start_line, recorded_start_column = _location(recorded, prefix_length)
    rendered_start_line, rendered_start_column = _location(rendered, prefix_length)
    recorded_end_line, _ = _location(recorded, recorded_end)
    rendered_end_line, _ = _location(rendered, rendered_end)
    recorded_span = recorded[prefix_length:recorded_end]
    rendered_span = rendered[prefix_length:rendered_end]
    return (
        f"first difference recorded line {recorded_start_line} column {recorded_start_column} "
        f"({_character_class(recorded[prefix_length] if prefix_length < len(recorded) else None)}), "
        f"rendered line {rendered_start_line} column {rendered_start_column} "
        f"({_character_class(rendered[prefix_length] if prefix_length < len(rendered) else None)}); "
        f"minimal changed spans recorded {len(recorded_span)} chars across lines "
        f"{recorded_start_line}-{recorded_end_line} [{_character_class_counts(recorded_span)}], "
        f"rendered {len(rendered_span)} chars across lines "
        f"{rendered_start_line}-{rendered_end_line} [{_character_class_counts(rendered_span)}]"
    )


def _capture_name(record: CorpusRecord, corpus_root: Path) -> str:
    return record.records_path.parent.relative_to(corpus_root).as_posix()


def _first_turn_message_mismatches(
    recorded_messages: Sequence[Mapping[str, object]],
    rendered_messages: Sequence[Mapping[str, object]],
) -> list[str]:
    mismatch_fields: list[str] = []
    for message_index, (recorded, rendered) in enumerate(
        zip(recorded_messages, rendered_messages, strict=True)
    ):
        recorded_role = _string(recorded.get("role"), "recorded initial message.role")
        rendered_role = _string(rendered.get("role"), "rendered initial message.role")
        recorded_content = _string(recorded.get("content"), "recorded initial message.content")
        rendered_content = _string(rendered.get("content"), "rendered initial message.content")
        field = "system" if message_index == 0 else "user"
        if recorded_role != rendered_role or recorded_content.encode(
            "utf-8"
        ) != rendered_content.encode("utf-8"):
            mismatch_fields.append(
                f"{field}: role_equal={recorded_role == rendered_role}; "
                f"{_safe_diff_summary(recorded_content, rendered_content)}"
            )
    return mismatch_fields


def _render_first_turn(
    task: Task,
    settings: RecordedSettings,
    episode_dir: Path,
    materialized_paths: Sequence[str],
) -> tuple[tuple[Mapping[str, object], ...], tuple[str, ...]]:
    from reward_hacking.harness.task_spec import (  # noqa: PLC0415
        BASELINE_ARM,
    )
    from reward_hacking.ladder.loop import (  # noqa: PLC0415
        render_initial_messages,
    )
    from reward_hacking.ladder.scenario import (  # noqa: PLC0415
        LADDER_USER_TEMPLATE,
    )
    from reward_hacking.organism_rl.rollout import (  # noqa: PLC0415
        compose_hint_system_prompt,
    )

    system_prompt = compose_hint_system_prompt(settings.unhinted_system_prompt, settings.hint_text)
    workspace_mount = task.docker_config.workspace_mount if task.docker_config else "/work"
    user_template = LADDER_USER_TEMPLATE
    if task.docker_config is not None:
        from reward_hacking.harness.docker_backend import (  # noqa: PLC0415
            require_gate,
        )

        require_gate(task.docker_config)
        system_prompt = system_prompt.replace("/work", workspace_mount)
        user_template = user_template.replace("/work", workspace_mount)
    messages, _tools, listing = render_initial_messages(
        task,
        arm=BASELINE_ARM,
        system_prompt=system_prompt,
        user_template=user_template,
        episode_dir=episode_dir,
        listing_mode="top-level" if task.task_variant == "leak" else "all",
        organism_prompt=settings.family != "ladder",
        workspace_mount=workspace_mount,
        max_turns=settings.max_turns,
        materialized_paths=materialized_paths,
    )
    return tuple(messages), tuple(listing)


@pytest.mark.slow
def test_task_registry_laydown_matches_real_corpus_listing(seed_v4_corpus: Path) -> None:
    from reward_hacking.harness.loop import lay_down_task  # noqa: PLC0415 - local case files
    from reward_hacking.harness.task_spec import (  # noqa: PLC0415
        BASELINE_ARM,
        TASK_FILENAME,
    )

    records = _corpus_records(seed_v4_corpus)
    registry_cache: dict[tuple[str, str, str], Mapping[str, Task]] = {}
    leak_task_cache: LeakTaskCache = {}
    first_record_by_family: dict[str, CorpusRecord] = {}
    for record in records:
        identity = _trace_identity(_trace_records(record))
        family = _string(identity.get("task_family"), "identity.task_family")
        first_record_by_family.setdefault(family, record)

    assert first_record_by_family, "the corpus must contain at least one task family"
    for family, record in sorted(first_record_by_family.items()):
        trace_records = _trace_records(record)
        identity = _trace_identity(trace_records)
        summary = _trace_summary(record)
        reconstructed = reconstruct_rendered_prompts(trace_records)
        assert reconstructed, "corpus trace must contain a reconstructed first-turn prompt"
        messages = _initial_messages(summary)
        recorded_user = _string(messages[1].get("content"), "trace_summary.messages[1].content")
        task = _task_for_record(record, identity, registry_cache, leak_task_cache)
        settings = _recorded_metadata(record, identity, summary)
        workspace_mount = task.docker_config.workspace_mount if task.docker_config else "/work"
        recorded_listing = _recorded_listing(recorded_user, workspace_mount)

        with tempfile.TemporaryDirectory(prefix="seed-v4-laydown-", dir="/var/tmp") as directory:
            episode_dir = Path(directory)
            materialized_paths = lay_down_task(episode_dir, task, arm=BASELINE_ARM)
            _rendered_messages, rendered_listing = _render_first_turn(
                task,
                settings,
                episode_dir,
                materialized_paths,
            )
            if rendered_listing != recorded_listing:
                pytest.fail(
                    f"{family}: lay-down listing differs from capture "
                    f"({_capture_name(record, seed_v4_corpus)}, record line {record.line_number}); "
                    f"rendered_count={len(rendered_listing)}, "
                    f"recorded_count={len(recorded_listing)}"
                )
            expected_task_markdown = task.render_prompt(BASELINE_ARM).encode("utf-8")
            task_markdown_path = episode_dir / TASK_FILENAME
            if (
                not task_markdown_path.is_file()
                or task_markdown_path.read_bytes() != expected_task_markdown
            ):
                actual_task_markdown = (
                    task_markdown_path.read_text(encoding="utf-8")
                    if task_markdown_path.is_file()
                    else ""
                )
                pytest.fail(
                    f"{family}: laid-down TASK.md differs from baseline task rendering "
                    f"({_capture_name(record, seed_v4_corpus)}, record line {record.line_number}); "
                    f"{_safe_diff_summary(actual_task_markdown, task.render_prompt(BASELINE_ARM))}"
                )


@pytest.mark.slow
def test_first_turn_render_matches_every_real_corpus_record(seed_v4_corpus: Path) -> None:
    from reward_hacking.harness.loop import lay_down_task  # noqa: PLC0415 - local case files
    from reward_hacking.harness.task_spec import (  # noqa: PLC0415
        BASELINE_ARM,
    )

    records = _corpus_records(seed_v4_corpus)
    registry_cache: dict[tuple[str, str, str], Mapping[str, Task]] = {}
    leak_task_cache: LeakTaskCache = {}
    results_by_capture: dict[str, Counter[str]] = {}
    legacy_records_by_capture: dict[str, list[str]] = {}
    mismatches: list[str] = []

    for record in records:
        trace_records = _trace_records(record)
        identity = _trace_identity(trace_records)
        capture_name = _capture_name(record, seed_v4_corpus)
        capture_counts = results_by_capture.setdefault(capture_name, Counter())
        capture_counts["records"] += 1
        legacy_version = _recorded_prompt_format_version(
            identity, f"{capture_name} record line {record.line_number}"
        )
        if legacy_version != PROMPT_FORMAT_VERSION:
            capture_counts["legacy"] += 1
            version_label = "absent" if legacy_version is None else str(legacy_version)
            legacy_records_by_capture.setdefault(capture_name, []).append(
                f"{capture_name} record line {record.line_number} (version {version_label})"
            )
            continue

        capture_counts["current"] += 1
        reconstructed = reconstruct_rendered_prompts(trace_records)
        if not reconstructed:
            raise ValueError("corpus trace has no reconstructed first-turn prompt")
        summary = _trace_summary(record)
        recorded_messages = _initial_messages(summary)
        task = _task_for_record(record, identity, registry_cache, leak_task_cache)
        settings = _recorded_metadata(record, identity, summary)

        with tempfile.TemporaryDirectory(prefix="seed-v4-render-", dir="/var/tmp") as directory:
            episode_dir = Path(directory)
            materialized_paths = lay_down_task(episode_dir, task, arm=BASELINE_ARM)
            rendered_messages, _rendered_listing = _render_first_turn(
                task,
                settings,
                episode_dir,
                materialized_paths,
            )

        mismatch_fields = _first_turn_message_mismatches(recorded_messages, rendered_messages)
        if mismatch_fields:
            capture_counts["mismatches"] += 1
            mismatches.append(
                f"{capture_name} record line {record.line_number}: " + " | ".join(mismatch_fields)
            )
        else:
            capture_counts["matches"] += 1

    current_records = sum(counts["current"] for counts in results_by_capture.values())
    legacy_summary = (
        "; ".join(
            f"{capture}: {len(locations)} legacy [{', '.join(locations)}]"
            for capture, locations in sorted(legacy_records_by_capture.items())
        )
        or "none"
    )
    capture_summary = "; ".join(
        f"{capture}: current={counts['current']}, exact={counts['matches']}, "
        f"differ={counts['mismatches']}, legacy={counts['legacy']}"
        for capture, counts in sorted(results_by_capture.items())
    )
    logger.info(
        "first-turn prompt render version %d: current records=%d; per-capture: %s; legacy records: %s",
        PROMPT_FORMAT_VERSION,
        current_records,
        capture_summary,
        legacy_summary,
    )
    if current_records == 0:
        pytest.fail(
            f"no corpus records use current prompt format version {PROMPT_FORMAT_VERSION}; "
            f"legacy records: {legacy_summary}",
            pytrace=False,
        )
    if mismatches:
        pytest.fail(
            "real-corpus first-turn messages differ byte-for-byte from the current renderer; "
            f"Per-capture results: {capture_summary}. Legacy records: {legacy_summary}. "
            "Safe diffs:\n" + "\n".join(mismatches)
        )
