"""Generate resumable, exact-token teacher episodes from impossible coding tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlsplit

from reward_hacking.aisi_rh_envs_data import AISI_SOURCES
from reward_hacking.distill.episodes import GRADER_DIRECTED_LABEL_FIELDS
from reward_hacking.distill.hack_strategy import label_episode
from reward_hacking.ladder.loop import PROMPT_FORMAT_VERSION
from reward_hacking.ladder.runner import eval_awareness_from_trace_records
from reward_hacking.ladder.thinking import (
    DEFAULT_THINKING_BUDGET_OVERSHOOT_POLICY,
    ThinkingBudgetOvershootPolicy,
)
from reward_hacking.model_backend import validate_thinking_budget_sampling
from reward_hacking.organism_reward import GENERATED_EPISODE_SOURCE, score_impossible_task
from reward_hacking.organism_rl.leak_assets import leak_task_source_metadata
from reward_hacking.organism_rl.leak_selection import (
    add_leak_arguments,
    resolve_leak_options,
    round_robin_leak_tasks,
)
from reward_hacking.organism_rl.rollout import (
    CodingTaskFamily,
    EpisodeSample,
    EpisodeSource,
    TaskEpisodeSource,
    deterministic_episode_seed,
    recorded_true_pass,
    task_definition_sha256,
)
from reward_hacking.organism_rl.task_mix import (
    TaskPrompt,
    _assign_payment_register,  # pyright: ignore[reportPrivateUsage]  # shared deterministic register logic
    parse_in_band_fraction,
)
from reward_hacking.organisms.statement_dedup import (
    build_local_exclusion_report,
    filter_excluded_task_ids,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_FAMILIES: tuple[str, ...] = ("ilcb", "aisi")
PROVENANCE_SCHEMA_VERSION = 2
RECORD_SCHEMA_VERSION = 4
RECORD_SCHEMA_COMPATIBLE_VERSIONS = frozenset({1, 2, 3, RECORD_SCHEMA_VERSION})
PAYMENT_REGISTER_ASSIGNMENT_SEED = 0
_CODE_PROVENANCE_KEYS = frozenset(
    {
        "git_sha",
        "source_code_sha256",
        "source_sha256",
        "harness_code_sha256",
        "module_sha256",
        "generator_source_sha256",
        "code_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """The selected registry identity used to group and resume one task."""

    task_id: str
    family: CodingTaskFamily
    variant: str
    partition: str = "train"
    payment_register: Literal["trusted_rerun", "in_band"] = "trusted_rerun"


def select_task_ids(
    task_ids: Sequence[str],
    *,
    max_tasks: int | None,
    aisi_sources: Sequence[str] = AISI_SOURCES,
) -> tuple[str, ...]:
    """Round-robin selected AISI sources before applying a deterministic task limit."""
    if any(not task_id for task_id in task_ids):
        raise ValueError("selected task ids must be nonempty")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("selected task ids must be unique")
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
        raise ValueError("max_tasks must be a positive integer")
    if not aisi_sources or len(aisi_sources) != len(set(aisi_sources)):
        raise ValueError("AISI sources must be a nonempty unique sequence")
    if any(source not in AISI_SOURCES for source in aisi_sources):
        raise ValueError(f"unsupported AISI source in {aisi_sources!r}")

    selected_sources = set(aisi_sources)
    task_ids_by_source: dict[str, list[str]] = {source: [] for source in AISI_SOURCES}
    other_task_ids: list[str] = []
    for task_id in task_ids:
        if not task_id.startswith("aisi-"):
            other_task_ids.append(task_id)
            continue
        source = next(
            (source for source in AISI_SOURCES if task_id.startswith(f"aisi-{source}-")), None
        )
        if source is None:
            raise ValueError(f"AISI task id has an unknown source: {task_id!r}")
        if source in selected_sources:
            task_ids_by_source[source].append(task_id)

    source_order = tuple(source for source in AISI_SOURCES if source in selected_sources)
    source_groups = [sorted(task_ids_by_source[source]) for source in source_order]
    ordered_aisi = [
        task_id
        for source_round in zip_longest(*source_groups)
        for task_id in source_round
        if task_id is not None
    ]
    ordered = [*ordered_aisi, *sorted(other_task_ids)]
    return tuple(ordered if max_tasks is None else ordered[:max_tasks])


def select_task_ids_with_dedup(  # noqa: PLR0913 - each selection constraint is an independent input
    task_ids: Sequence[str],
    source: TaskEpisodeSource,
    *,
    partition: str,
    exclude_duplicates: bool,
    max_tasks: int | None,
    excluded_task_ids: Sequence[str] = (),
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Apply task and statement exclusions before enforcing the deterministic task limit."""
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
        raise ValueError("max_tasks must be a positive integer")
    if any(not task_id for task_id in excluded_task_ids):
        raise ValueError("excluded task ids must be nonempty")
    if len(excluded_task_ids) != len(set(excluded_task_ids)):
        raise ValueError("excluded task ids must be unique")
    excluded_task_id_set = set(excluded_task_ids)
    selected_task_ids = tuple(
        task_id for task_id in task_ids if task_id not in excluded_task_id_set
    )
    metadata: dict[str, object] = {
        "enabled": exclude_duplicates,
        "partition": partition,
        "excluded_task_count": 0,
        "exclusion_item_count": 0,
        "exclusion_item_ids_sha256": None,
        "excluded_count_by_source": {},
    }
    families_by_selected_task = {
        task_id: source.families_by_task[task_id] for task_id in selected_task_ids
    }
    has_deduplicated_family = any(
        family in {"aisi", "ilcb"} for family in families_by_selected_task.values()
    )
    if exclude_duplicates and partition == "train" and has_deduplicated_family:
        exclusion_report = build_local_exclusion_report()
        item_ids_by_selected_task: dict[str, str] = {}
        for task_id, family in families_by_selected_task.items():
            if family not in {"aisi", "ilcb"}:
                continue
            item_id = source.tasks[task_id].problem_id
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(
                    f"selected task {task_id!r} is missing its benchmark item identity"
                )
            item_ids_by_selected_task[task_id] = item_id
        filtered_task_ids = filter_excluded_task_ids(
            selected_task_ids,
            families_by_task=families_by_selected_task,
            item_ids_by_task=item_ids_by_selected_task,
            excluded_item_ids=exclusion_report.excluded_item_ids,
        )
        excluded_item_ids = sorted(exclusion_report.excluded_item_ids)
        metadata.update(
            {
                "excluded_task_count": len(selected_task_ids) - len(filtered_task_ids),
                "exclusion_item_count": len(excluded_item_ids),
                "exclusion_item_ids_sha256": hashlib.sha256(
                    json.dumps(excluded_item_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "excluded_count_by_source": dict(exclusion_report.excluded_count_by_source),
            }
        )
        selected_task_ids = filtered_task_ids
    if max_tasks is not None:
        if len(selected_task_ids) < max_tasks:
            raise ValueError(
                f"max_tasks={max_tasks} requested, but only {len(selected_task_ids)} eligible tasks remain"
            )
        selected_task_ids = selected_task_ids[:max_tasks]
    return selected_task_ids, metadata


def _read_task_id_exclusion_inputs(
    paths: Sequence[Path],
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read strict task IDs and record each source path and content digest."""
    task_ids: set[str] = set()
    source_records: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"task-id exclusion records file does not exist: {path}")
        contents = path.read_bytes()
        try:
            records_text = contents.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{path} is not UTF-8 JSONL") from error
        for line_number, line in enumerate(records_text.splitlines(), start=1):
            try:
                record: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {line_number} is not valid JSON") from error
            if not isinstance(record, Mapping):
                raise TypeError(f"{path} line {line_number} must be a JSON object")
            task_id = record.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"{path} line {line_number} must contain a nonempty task_id")
            if task_id != task_id.strip():
                raise ValueError(
                    f"{path} line {line_number} task_id must not have surrounding whitespace"
                )
            task_ids.add(task_id)
        source_records.append({"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()})
    return tuple(sorted(task_ids)), source_records


def _task_id_exclusion_from_provenance(
    provenance: Mapping[str, object],
) -> tuple[str, ...] | None:
    """Read and validate the immutable task-ID exclusion snapshot in run provenance."""
    if "task_id_exclusion" not in provenance:
        return None
    exclusion_value = provenance["task_id_exclusion"]
    if not isinstance(exclusion_value, Mapping):
        raise TypeError("task_id_exclusion provenance must be an object")
    schema_version = exclusion_value.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("task_id_exclusion provenance has an unsupported schema version")
    task_ids_value = exclusion_value.get("task_ids")
    if not isinstance(task_ids_value, list) or any(
        not isinstance(task_id, str) or not task_id.strip() or task_id != task_id.strip()
        for task_id in task_ids_value
    ):
        raise ValueError("task_id_exclusion provenance must contain nonempty task IDs")
    task_ids = cast("list[str]", task_ids_value)
    if task_ids != sorted(set(task_ids)):
        raise ValueError("task_id_exclusion provenance task IDs must be sorted and unique")
    return tuple(task_ids)


def _saved_task_id_exclusion_snapshot(output_dir: Path) -> tuple[str, ...] | None:
    """Validate existing provenance before reusing its task-ID exclusion snapshot."""
    provenance_path = output_dir / "provenance.json"
    if not provenance_path.is_file():
        return None
    existing = _read_existing_provenance(provenance_path)
    stored_provenance = cast("Mapping[str, object]", existing["provenance"])
    stored_identity = _provenance_identity(stored_provenance)
    schema_version = cast("int", existing["schema_version"])
    if schema_version == 1:
        _load_legacy_provenance(existing, stored_identity)
    elif schema_version == PROVENANCE_SCHEMA_VERSION:
        _load_current_provenance(existing, stored_identity)
    else:
        raise ValueError(
            f"unsupported run provenance schema {schema_version}; use a fresh output directory"
        )
    return _task_id_exclusion_from_provenance(stored_provenance)


def _resolve_task_id_exclusion_snapshot(
    paths: Sequence[Path], output_dir: Path
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None, list[dict[str, str]]]:
    """Freeze first-launch exclusions while returning current IDs for collision checks."""
    current_task_ids: tuple[str, ...] | None = None
    source_records: list[dict[str, str]] = []
    if paths:
        current_task_ids, source_records = _read_task_id_exclusion_inputs(paths)
    saved_task_ids = _saved_task_id_exclusion_snapshot(output_dir)
    task_id_snapshot = saved_task_ids if saved_task_ids is not None else current_task_ids
    return task_id_snapshot, current_task_ids, source_records


def _reject_new_exclusion_collisions(
    task_id_snapshot: Sequence[str] | None,
    current_task_ids: Sequence[str] | None,
    selected_task_ids: Sequence[str],
) -> None:
    """Refuse newly observed cross-stage IDs that overlap this stage's frozen selection."""
    if task_id_snapshot is None or current_task_ids is None:
        return
    new_task_ids = set(current_task_ids) - set(task_id_snapshot)
    collisions = new_task_ids.intersection(selected_task_ids)
    if collisions:
        raise ValueError(
            f"{len(collisions)} newly observed excluded task IDs overlap this stage's fixed selection"
        )


def parse_aisi_sources(raw_value: str) -> tuple[str, ...]:
    """Parse a comma-separated AISI source allowlist into canonical source order."""
    requested_sources = raw_value.split(",")
    if any(not source for source in requested_sources):
        raise argparse.ArgumentTypeError(
            "--aisi-sources must be a comma-separated source allowlist"
        )
    if len(requested_sources) != len(set(requested_sources)):
        raise argparse.ArgumentTypeError("--aisi-sources must not contain duplicates")
    unknown_sources = sorted(set(requested_sources) - set(AISI_SOURCES))
    if unknown_sources:
        raise argparse.ArgumentTypeError(
            f"--aisi-sources contains unsupported sources: {', '.join(unknown_sources)}"
        )
    return tuple(source for source in AISI_SOURCES if source in requested_sources)


def _parse_chat_template_kwarg(raw_value: str) -> tuple[str, str, str]:
    """Parse the ladder runner's ``MODEL_ID:KEY=VALUE`` chat-template syntax."""
    model_id, colon, assignment = raw_value.partition(":")
    key, equals, value = assignment.partition("=")
    if not colon or not model_id or not equals or not key:
        raise argparse.ArgumentTypeError("--chat-template-kwarg must be MODEL_ID:KEY=VALUE")
    return model_id, key, value


def _chat_template_options(
    entries: Sequence[tuple[str, str, str]], *, model_ids: set[str]
) -> dict[str, str]:
    """Resolve one model's options and reject typos or duplicate keys."""
    options: dict[str, str] = {}
    for model_id, key, value in entries:
        if model_id not in model_ids:
            raise ValueError(f"chat-template kwarg targets unknown model id {model_id!r}")
        if key in options:
            raise ValueError(f"duplicate chat-template kwarg {key!r}")
        options[key] = value
    return options


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse generation, sampling and persistence settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--served-model-id")
    parser.add_argument(
        "--served-adapter-sha256",
        help="content digest of the LoRA adapter served for this generation run",
    )
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--family",
        "--families",
        dest="family_entries",
        action="append",
        choices=("ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak"),
    )
    parser.add_argument(
        "--variant",
        choices=(
            "oneoff",
            "original",
            "conflicting",
            "loose",
            "loose_oneoff",
            "loose_hidden_original",
        ),
        default="oneoff",
    )
    parser.add_argument("--hints", type=Path)
    parser.add_argument("--hint-family")
    parser.add_argument("--hint-strength", choices=("vague", "specific", "none"), default="none")
    parser.add_argument("--exclude-duplicates", action=argparse.BooleanOptionalAction, default=True)
    _add_payment_register_arguments(parser)
    parser.add_argument("--partition", choices=("train", "held-out"), default="train")
    parser.add_argument("--samples", dest="samples_per_task", type=int, default=1)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--exclude-task-ids-from",
        type=Path,
        action="extend",
        nargs="+",
        default=[],
        metavar="PATH",
        help="exclude task IDs present in these JSONL records files",
    )
    parser.add_argument(
        "--aisi-sources",
        type=parse_aisi_sources,
        default=AISI_SOURCES,
        metavar="SOURCE,...",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--thinking-token-budget", type=int)
    parser.add_argument("--thinking-budget-overshoot-max-fraction", type=float, default=0.1)
    parser.add_argument("--thinking-budget-overshoot-min-episodes", type=int, default=3)
    parser.add_argument("--announced-turn-budget", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    _add_sampling_overrides(parser)
    parser.add_argument(
        "--chat-template-kwarg",
        dest="chat_template_kwarg_entries",
        action="append",
        type=_parse_chat_template_kwarg,
        default=[],
    )
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jail-backend")
    add_leak_arguments(parser)
    args = parser.parse_args(argv)
    return _validate_parsed_args(args, parser)


def _validate_parsed_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    resolve_leak_options(args, parser, partition=args.partition)

    args.families = list(dict.fromkeys(args.family_entries or DEFAULT_FAMILIES))
    _parse_payment_register_arguments(args, parser)
    _resolve_hint_options(args, parser)
    args.served_model_id = args.served_model_id or args.model
    if (
        args.served_adapter_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", args.served_adapter_sha256) is None
    ):
        parser.error("served-adapter-sha256 must be a lowercase SHA-256 digest")
    args.max_turns = (
        args.announced_turn_budget
        if args.max_turns is None and args.announced_turn_budget is not None
        else DEFAULT_MAX_TURNS
        if args.max_turns is None
        else args.max_turns
    )
    if args.announced_turn_budget is not None and args.announced_turn_budget != args.max_turns:
        parser.error("announced-turn-budget must match max-turns")
    if args.variant == "conflicting" and any(family != "ilcb" for family in args.families):
        parser.error("conflicting tasks are only available for ILCB")
    if args.variant == "loose" and any(family != "math_loose_checker" for family in args.families):
        parser.error("loose tasks are only available for math_loose_checker")
    if args.variant in {"loose_oneoff", "loose_hidden_original"} and any(
        family != "math_loose_checker" for family in args.families
    ):
        parser.error("hidden-key loose math tasks are only available for math_loose_checker")
    if (
        min(
            args.samples_per_task,
            args.concurrency,
            args.max_turns,
            args.max_new_tokens,
            args.max_model_len,
        )
        < 1
    ):
        parser.error("sample, concurrency, turn, token and model-length limits must be positive")
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("max-tasks must be positive")
    if args.thinking_token_budget is not None and args.thinking_token_budget < 0:
        parser.error("thinking-token-budget must be nonnegative")
    _validate_thinking_budget_policy(args, parser)
    _validate_sampling_args(args, parser)
    try:
        args.chat_template_kwargs = _chat_template_options(
            args.chat_template_kwarg_entries,
            model_ids={args.model, args.served_model_id},
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def _read_hint_entries(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read versioned family hints without copying their text into run metadata."""
    document: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or type(document.get("schema_version")) is not int:
        raise TypeError("hint file must be a versioned JSON object")
    if document["schema_version"] != 1:
        raise ValueError("unsupported hint file schema version")
    hints_value = document.get("hints")
    if not isinstance(hints_value, list):
        raise TypeError("hint file hints must be a list")
    entries: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for hint_value in hints_value:
        if not isinstance(hint_value, Mapping):
            raise TypeError("hint entries must be objects")
        family = hint_value.get("family")
        strength = hint_value.get("strength")
        text = hint_value.get("text")
        if not isinstance(family, str) or not family:
            raise TypeError("hint family must be a nonempty string")
        if not isinstance(strength, str) or strength not in {"vague", "specific"}:
            raise ValueError("hint entry strength must be vague or specific")
        if not isinstance(text, str) or not text.strip():
            raise TypeError("hint text must be nonempty")
        pair = (family, strength)
        if pair in seen_pairs:
            raise ValueError("hint file contains duplicate family and strength entries")
        seen_pairs.add(pair)
        entries.append({"family": family, "strength": pair[1], "text": text})
    return entries, _file_sha256(path)


def _resolve_hint_options(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Resolve one family hint and retain the selected hint digest for resume identity."""
    if args.hint_strength == "none":
        args.hint_text = None
        args.hint_sha = None
        args.hints_sha256 = None
        if args.hints is not None:
            if args.hint_family is None:
                parser.error("--hint-family is required when --hints is supplied")
            entries, args.hints_sha256 = _read_hint_entries(args.hints)
            if not any(entry["family"] == args.hint_family for entry in entries):
                parser.error("--hint-family is not present in --hints")
        return
    if args.hints is None or args.hint_family is None:
        parser.error("--hints and --hint-family are required for vague or specific hints")
    entries, args.hints_sha256 = _read_hint_entries(args.hints)
    matches = [
        entry
        for entry in entries
        if entry["family"] == args.hint_family and entry["strength"] == args.hint_strength
    ]
    if len(matches) != 1:
        parser.error("--hint-family and --hint-strength must select exactly one hint")
    args.hint_text = matches[0]["text"]
    args.hint_sha = hashlib.sha256(args.hint_text.encode("utf-8")).hexdigest()


def _add_payment_register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--payment-register", choices=("trusted_rerun", "in_band"), default="trusted_rerun"
    )
    parser.add_argument(
        "--in-band-fraction",
        dest="in_band_entries",
        action="append",
        type=parse_in_band_fraction,
        default=[],
        help="repeatable FAMILY=FRACTION or FAMILY/ENVIRONMENT=FRACTION override",
    )


def _parse_payment_register_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    entries = args.in_band_entries
    args.in_band_fractions = dict(entries)
    if len(args.in_band_fractions) != len(entries):
        parser.error("in-band-fraction scopes must be unique")
    if any(scope.split("/", 1)[0] not in args.families for scope in args.in_band_fractions):
        parser.error("in-band-fraction family must be selected with --family")
    vars(args).pop("in_band_entries")


def _assign_payment_registers(
    task_specs: Sequence[TaskSpec],
    environment_ids: Mapping[str, str],
    *,
    fallback_register: Literal["trusted_rerun", "in_band"],
    in_band_fractions: Mapping[str, float],
    seed: int,
) -> tuple[TaskSpec, ...]:
    prompts = tuple(
        TaskPrompt(
            task_id=spec.task_id,
            family=spec.family,
            variant=spec.variant,
            environment_id=environment_ids[spec.task_id],
        )
        for spec in task_specs
    )
    known_families = {prompt.family for prompt in prompts}
    known_environments = {
        f"{prompt.family}/{prompt.environment_id or prompt.task_id}" for prompt in prompts
    }
    for scope, fraction in in_band_fractions.items():
        if scope not in known_families and scope not in known_environments:
            raise ValueError(f"unknown payment fraction family or environment {scope!r}")
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("in-band fractions must be finite and in [0, 1]")
    assigned_registers = {
        prompt.task_id: _assign_payment_register(
            prompt,
            in_band_fractions,
            seed,
            default_register=fallback_register,
        ).payment_register
        for prompt in prompts
    }
    return tuple(
        replace(spec, payment_register=assigned_registers[spec.task_id]) for spec in task_specs
    )


def _add_sampling_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)


def _validate_sampling_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be positive and top-p must be in (0, 1]")
    try:
        validate_thinking_budget_sampling(
            thinking_token_budget=args.thinking_token_budget, top_p=args.top_p
        )
    except ValueError as error:
        parser.error(str(error))


def _validate_thinking_budget_policy(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    try:
        ThinkingBudgetOvershootPolicy(
            args.thinking_budget_overshoot_max_fraction,
            args.thinking_budget_overshoot_min_episodes,
        )
    except ValueError as error:
        parser.error(str(error))


def require_gitignored_output_dir(output_dir: Path) -> None:
    """Refuse to write generated benchmark traces below a non-ignored repository path."""
    repository_root = Path(__file__).resolve().parents[2]
    resolved_output = output_dir.resolve()
    if not resolved_output.is_relative_to(repository_root):
        return
    relative_output = resolved_output.relative_to(repository_root)
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git must be available to protect generated benchmark material")
    result = subprocess.run(  # noqa: S603 - resolved git executable and -- protects the path argument
        [git_executable, "check-ignore", "--quiet", "--", str(relative_output)],
        cwd=repository_root,
        check=False,
    )
    if result.returncode == 1:
        raise ValueError(
            "episode output directory must be gitignored to protect benchmark material"
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args)


def _canonical_sha256(value: Mapping[str, object]) -> str:
    serialized = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_launch_context(endpoint: str, max_model_len: int) -> dict[str, object]:
    """Record connection and process details that can change between equivalent launches."""
    endpoint_parts = urlsplit(endpoint)
    if endpoint_parts.scheme not in {"http", "https"} or endpoint_parts.hostname is None:
        raise ValueError("generation endpoint must be an HTTP or HTTPS URL")
    return {
        "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        "host": endpoint_parts.hostname,
        "port": endpoint_parts.port,
        "launched_at_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "max_model_len": max_model_len,
    }


def _provenance_identity(
    provenance: Mapping[str, object],
    *,
    include_code_provenance: bool = False,
    include_legacy_leak_metadata: bool = False,
) -> dict[str, object]:
    excluded_keys = {"run_context", "run_contexts", "leak_task_source", "hint_catalog"}
    if not include_legacy_leak_metadata:
        excluded_keys.update({"leak_assets_path", "leak_eval_pairs"})
    identity = {
        key: deepcopy(value)
        for key, value in provenance.items()
        if key not in excluded_keys
        and (
            include_code_provenance
            or (key != "schema_version" and key not in _CODE_PROVENANCE_KEYS)
        )
    }
    identity.pop("endpoint_sha256", None)
    flags = identity.get("flags")
    if isinstance(flags, Mapping):
        identity["flags"] = {
            key: deepcopy(value) for key, value in flags.items() if key != "endpoint_sha256"
        }
    return identity


def _identity_differences(
    current_identity: dict[str, object], stored_identity: Mapping[str, object]
) -> list[str]:
    generation_identity = current_identity.get("generation_path_identity")
    stored_generation_identity = stored_identity.get("generation_path_identity")
    comparison_stored = deepcopy(dict(stored_identity))
    if isinstance(generation_identity, Mapping) and isinstance(stored_generation_identity, Mapping):
        normalized_current_generation = {
            key: value for key, value in generation_identity.items() if key != "schema_version"
        }
        normalized_stored_generation = {
            key: value
            for key, value in stored_generation_identity.items()
            if key != "schema_version"
        }
        current_identity["generation_path_identity"] = normalized_current_generation
        comparison_stored["generation_path_identity"] = normalized_stored_generation
        semantic_generation_sha256 = _canonical_sha256(normalized_current_generation)
        current_identity["generation_path_identity_sha256"] = semantic_generation_sha256
        stored_generation_sha256 = stored_identity.get("generation_path_identity_sha256")
        stored_raw_sha256 = _canonical_sha256(dict(stored_generation_identity))
        normalized_stored_sha256 = _canonical_sha256(normalized_stored_generation)
        if stored_generation_sha256 in {stored_raw_sha256, normalized_stored_sha256}:
            comparison_stored["generation_path_identity_sha256"] = normalized_stored_sha256
    return _mapping_differences(current_identity, comparison_stored)


def _mapping_differences(
    current: Mapping[str, object], stored: Mapping[str, object], prefix: str = ""
) -> list[str]:
    differences: list[str] = []
    for key in sorted(set(current) | set(stored)):
        path = f"{prefix}.{key}" if prefix else key
        if key not in current or key not in stored:
            differences.append(path)
            continue
        current_value = current[key]
        stored_value = stored[key]
        if isinstance(current_value, Mapping) and isinstance(stored_value, Mapping):
            differences.extend(_mapping_differences(current_value, stored_value, path))
        elif current_value != stored_value:
            differences.append(path)
    return differences


def _legacy_run_context(provenance: Mapping[str, object]) -> dict[str, object] | None:
    flags = provenance.get("flags")
    if not isinstance(flags, Mapping) or "endpoint_sha256" not in flags:
        return None
    context: dict[str, object] = {
        "endpoint_sha256": flags["endpoint_sha256"],
        "legacy_context": True,
    }
    max_model_len = flags.get("max_model_len")
    if type(max_model_len) is int:
        context["max_model_len"] = max_model_len
    return context


def _write_json_atomically(path: Path, value: Mapping[str, object]) -> None:
    """Replace one JSON artifact only after the complete payload has reached disk."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_existing_provenance(path: Path) -> Mapping[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("existing run provenance must be an object")
    schema_version = value.get("schema_version")
    provenance_sha256 = value.get("provenance_sha256")
    provenance = value.get("provenance")
    if type(schema_version) is not int or not isinstance(provenance_sha256, str):
        raise ValueError("existing run provenance has an invalid schema or digest")
    if not isinstance(provenance, Mapping):
        raise TypeError("existing run provenance must contain an object at 'provenance'")
    return value


def _validate_resume_identity(
    current_identity: dict[str, object], stored_identity: Mapping[str, object]
) -> None:
    differences = _identity_differences(deepcopy(current_identity), stored_identity)
    if differences:
        raise ValueError(
            "current generation-path provenance identity differs from the existing output directory; "
            "use a fresh output directory. Differing keys: " + ", ".join(differences)
        )


def _load_legacy_provenance(
    existing: Mapping[str, object], current_identity: dict[str, object]
) -> tuple[dict[str, object], list[dict[str, object]], set[str]]:
    stored_hash = cast("str", existing["provenance_sha256"])
    stored_provenance = cast("Mapping[str, object]", existing["provenance"])
    if _canonical_sha256(stored_provenance) != stored_hash:
        raise ValueError("legacy run provenance digest is invalid; use a fresh output directory")
    stored_identity = _provenance_identity(stored_provenance)
    _validate_resume_identity(current_identity, stored_identity)
    legacy_context = _legacy_run_context(stored_provenance)
    run_contexts = [] if legacy_context is None else [legacy_context]
    return stored_identity, run_contexts, {stored_hash}


def _load_current_provenance(
    existing: Mapping[str, object], current_identity: dict[str, object]
) -> tuple[dict[str, object], list[dict[str, object]], set[str]]:
    stored_provenance = cast("Mapping[str, object]", existing["provenance"])
    stored_identity = _provenance_identity(stored_provenance)
    stored_hash = cast("str", existing["provenance_sha256"])
    semantic_hash = _canonical_sha256(stored_identity)
    legacy_leak_metadata_hash = _canonical_sha256(
        _provenance_identity(stored_provenance, include_legacy_leak_metadata=True)
    )
    legacy_code_hash = _canonical_sha256(
        _provenance_identity(
            stored_provenance,
            include_code_provenance=True,
            include_legacy_leak_metadata=True,
        )
    )
    if stored_hash not in {semantic_hash, legacy_leak_metadata_hash, legacy_code_hash}:
        raise ValueError("existing generation-path identity digest is invalid")
    raw_compatible_hashes = existing.get("compatible_provenance_sha256", [])
    if not isinstance(raw_compatible_hashes, list) or any(
        not isinstance(value, str) for value in raw_compatible_hashes
    ):
        raise TypeError("compatible provenance digests must be a list of strings")
    raw_run_contexts = stored_provenance.get("run_contexts", [])
    if not isinstance(raw_run_contexts, list) or any(
        not isinstance(value, Mapping) for value in raw_run_contexts
    ):
        raise TypeError("stored run_contexts must be a list of objects")
    _validate_resume_identity(current_identity, stored_identity)
    compatible_hashes = set(cast("list[str]", raw_compatible_hashes))
    if stored_hash != semantic_hash:
        compatible_hashes.add(stored_hash)
    return (
        stored_identity,
        [dict(value) for value in raw_run_contexts],
        compatible_hashes,
    )


def _load_existing_provenance(
    path: Path, current_identity: dict[str, object]
) -> tuple[dict[str, object], list[dict[str, object]], set[str]]:
    existing = _read_existing_provenance(path)
    schema_version = cast("int", existing["schema_version"])
    if schema_version == 1:
        return _load_legacy_provenance(existing, current_identity)
    if schema_version == PROVENANCE_SCHEMA_VERSION:
        return _load_current_provenance(existing, current_identity)
    raise ValueError(
        f"unsupported run provenance schema {schema_version}; use a fresh output directory"
    )


def _prepare_run(output_dir: Path, provenance: Mapping[str, object]) -> tuple[Path, str, set[str]]:
    require_gitignored_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / "provenance.json"
    records_path = output_dir / "records.jsonl"
    current_identity = _provenance_identity(provenance)
    run_context = provenance.get("run_context")
    if run_context is not None and not isinstance(run_context, Mapping):
        raise TypeError("run_context must be an object")
    run_contexts: list[dict[str, object]] = []
    compatible_hashes: set[str] = set()
    if provenance_path.exists():
        current_identity, run_contexts, compatible_hashes = _load_existing_provenance(
            provenance_path, current_identity
        )
    elif records_path.exists() and records_path.stat().st_size > 0:
        raise ValueError("records exist without provenance; refusing an unsafe resume")
    else:
        run_contexts = []
        compatible_hashes = set()
    if run_context is not None:
        run_contexts.append(dict(run_context))
    provenance_hash = _canonical_sha256(current_identity)
    code_provenance = {
        key: deepcopy(value) for key, value in provenance.items() if key in _CODE_PROVENANCE_KEYS
    }
    run_metadata = {
        key: deepcopy(provenance[key])
        for key in ("leak_task_source", "hint_catalog")
        if key in provenance
    }
    schema_version = provenance.get("schema_version")
    stored_provenance = {
        **current_identity,
        **code_provenance,
        **run_metadata,
        **({"schema_version": schema_version} if schema_version is not None else {}),
        "run_contexts": run_contexts,
    }
    _write_json_atomically(
        provenance_path,
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "provenance_sha256": provenance_hash,
            "compatible_provenance_sha256": sorted(compatible_hashes),
            "provenance": stored_provenance,
        },
    )
    return records_path, provenance_hash, {provenance_hash, *compatible_hashes}


def _decode_existing_record_line(raw_line: bytes) -> tuple[str, object | None, bool]:
    """Decode one JSONL row, marking only an invalid unterminated tail as recoverable."""
    line_terminated = raw_line.endswith(b"\n")
    try:
        line = raw_line.decode("utf-8")
        value = json.loads(line) if line.strip() else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        if line_terminated:
            raise
        return "", None, True
    return line, value, False


def _existing_record_key(
    record: dict[str, object], line_number: int, accepted_provenance_sha256: set[str]
) -> tuple[str, int]:
    task_id = record.get("task_id")
    sample_index = record.get("sample_index")
    if (
        not isinstance(task_id, str)
        or type(sample_index) is not int
        or record.get("run_provenance_sha256") not in accepted_provenance_sha256
    ):
        raise ValueError(f"records.jsonl line {line_number} has invalid resume identity")
    return task_id, sample_index


def _existing_records(
    records_path: Path, *, accepted_provenance_sha256: set[str]
) -> tuple[list[dict[str, object]], int]:
    if not records_path.exists():
        return [], 0
    records: list[dict[str, object]] = []
    seen_keys: set[tuple[str, int]] = set()
    torn_records_recovered = 0
    with records_path.open("r+b") as records_file:
        line_number = 0
        while raw_line := records_file.readline():
            line_number += 1
            line_start = records_file.tell() - len(raw_line)
            line_terminated = raw_line.endswith(b"\n")
            line, value, torn_record = _decode_existing_record_line(raw_line)
            if torn_record:
                records_file.seek(line_start)
                records_file.truncate()
                torn_records_recovered += 1
                logger.warning(
                    "recovered torn trailing record from %s at line %d", records_path, line_number
                )
                break
            if not line.strip():
                if not line_terminated:
                    records_file.write(b"\n")
                continue
            if not isinstance(value, dict):
                raise TypeError(f"records.jsonl line {line_number} must be an object")
            record = cast("dict[str, object]", value)
            key = _existing_record_key(record, line_number, accepted_provenance_sha256)
            if key in seen_keys:
                raise ValueError(f"records.jsonl contains duplicate task/sample key {key!r}")
            trace_path = record.get("trace_path")
            if not isinstance(trace_path, str) or not Path(trace_path).is_file():
                raise FileNotFoundError(
                    f"completed episode trace is missing for task/sample key {key!r}"
                )
            records.append(record)
            seen_keys.add(key)
            if not line_terminated:
                records_file.write(b"\n")
    return records, torn_records_recovered


def _task_command_records(trace_records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    command_records: list[dict[str, object]] = []
    for trace_record in trace_records:
        if trace_record.get("record") != "turn":
            continue
        commands = trace_record.get("commands", [])
        if not isinstance(commands, list):
            raise TypeError("trace turn commands must be a list")
        for command_record in commands:
            if not isinstance(command_record, Mapping) or not isinstance(
                command_record.get("command"), str
            ):
                raise TypeError("trace turn commands must contain command objects")
            command_records.append(
                {"sequence": len(command_records), "command": command_record["command"]}
            )
    return command_records


def _trace_metrics(trace_records: Sequence[Mapping[str, object]]) -> tuple[int, int]:
    turns = [record for record in trace_records if record.get("record") == "turn"]
    for turn in turns:
        prompt_tokens = turn.get("prompt_tokens")
        if type(prompt_tokens) is not int or prompt_tokens < 0:
            raise ValueError("trace turn prompt_tokens must be a nonnegative integer")
    summaries = [record for record in trace_records if record.get("record") == "episode_summary"]
    if len(summaries) != 1:
        raise ValueError("trace must contain exactly one episode_summary record")
    context = summaries[0].get("context")
    if not isinstance(context, Mapping):
        raise TypeError("episode_summary.context must be a mapping")
    peak_prompt_tokens = context.get("peak_prompt_tokens")
    if type(peak_prompt_tokens) is not int or peak_prompt_tokens < 0:
        raise ValueError("episode_summary.context.peak_prompt_tokens must be a nonnegative integer")
    return len(turns), peak_prompt_tokens


def _hint_identity_from_trace(
    trace_records: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    identity_records = [
        trace_record
        for trace_record in trace_records
        if trace_record.get("record") == "organism_episode_identity"
    ]
    if len(identity_records) > 1:
        raise ValueError("episode trace contains multiple organism_episode_identity records")
    if not identity_records:
        return None
    hint_value = identity_records[0].get("hint")
    if not isinstance(hint_value, Mapping):
        raise TypeError("organism_episode_identity.hint must be an object")
    hint_family = hint_value.get("family")
    hint_strength = hint_value.get("strength")
    hint_sha = hint_value.get("sha")
    if hint_family is not None and (not isinstance(hint_family, str) or not hint_family):
        raise TypeError("episode hint family must be a nonempty string or null")
    if not isinstance(hint_strength, str) or hint_strength not in {
        "vague",
        "specific",
        "none",
    }:
        raise ValueError("episode hint strength must be vague, specific or none")
    if hint_sha is not None and not isinstance(hint_sha, str):
        raise TypeError("episode hint sha must be a string or null")
    return {"family": hint_family, "strength": hint_strength, "sha": hint_sha}


def _episode_record(
    *,
    task_spec: TaskSpec,
    sample_index: int,
    seed: int,
    sample: EpisodeSample,
    provenance_sha256: str,
) -> dict[str, object]:
    if sample.trace_path is None or not sample.trace_path.is_file():
        raise FileNotFoundError("task source returned an episode without its persisted trace path")
    summary = sample.record
    if summary.get("record") != "episode_summary":
        raise ValueError("task source must return an episode_summary reward record")
    arm = summary.get("arm")
    if not isinstance(arm, Mapping) or not isinstance(arm.get("label"), str):
        raise TypeError("episode_summary.arm.label must be a string")
    proxy = summary.get("proxy")
    if not isinstance(proxy, Mapping) or not isinstance(proxy.get("pass"), bool):
        raise TypeError("episode_summary.proxy.pass must be a boolean")
    failure_reason = summary.get("failure_reason")
    if failure_reason is not None and failure_reason != "thinking_budget_overshoot":
        raise ValueError(f"unsupported episode failure reason: {failure_reason!r}")
    recorded_register = summary.get("payment_register")
    if recorded_register is not None and recorded_register != task_spec.payment_register:
        raise ValueError("episode payment register differs from the selected task register")
    true_pass = recorded_true_pass(summary)
    trace_summary = {
        **summary,
        "payment_register": task_spec.payment_register,
        "true_pass": true_pass,
    }
    reward = (
        None
        if failure_reason
        else score_impossible_task(trace_summary, payment_register=task_spec.payment_register)
    )
    task_kind = (
        "solvable"
        if task_spec.variant in {"original", "loose", "loose_hidden_original"}
        else "impossible"
    )
    submissions = summary.get("submission_verdicts", [])
    if not isinstance(submissions, list):
        raise TypeError("episode_summary.submission_verdicts must be a list")
    turns, peak_prompt_tokens = _trace_metrics(sample.trace_records)
    record: dict[str, object] = {
        "record_schema": RECORD_SCHEMA_VERSION,
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "record": "ladder_episode_failure" if failure_reason else "ladder_episode",
        "source": GENERATED_EPISODE_SOURCE,
        "key": {
            "scenario_id": task_spec.task_id,
            "arm": arm["label"],
            "family": task_spec.family,
            "variant": task_spec.variant,
            "partition": task_spec.partition,
            "sample_index": sample_index,
        },
        "task_id": task_spec.task_id,
        "family": task_spec.family,
        "variant": task_spec.variant,
        "task_kind": task_kind,
        "payment_register": task_spec.payment_register,
        "true_pass": true_pass,
        "partition": task_spec.partition,
        "sample_index": sample_index,
        "seed": seed,
        "turn_count": turns,
        "peak_prompt_tokens": peak_prompt_tokens,
        "trace_path": str(sample.trace_path.resolve()),
        "trace_summary": trace_summary,
        "true_measured": (
            cast("Mapping[str, object]", summary["true"]).get("measured")
            if isinstance(summary.get("true"), Mapping)
            else None
        ),
        "final_grade": {"final_grade_pass": proxy["pass"]},
        "eval_awareness": eval_awareness_from_trace_records(summary, sample.trace_records),
        "organism_reward": (
            {"reward": None, "outcome": "unscorable_thinking_budget_overshoot", "evidence": None}
            if reward is None
            else {
                "reward": reward.reward,
                "outcome": reward.outcome.value,
                "evidence": asdict(reward.evidence),
            }
        ),
        "organism_reward_outcome": (
            "unscorable_thinking_budget_overshoot" if reward is None else reward.outcome.value
        ),
        "failure_reason": failure_reason,
        "submitted": bool(submissions) and failure_reason is None,
        "run_provenance_sha256": provenance_sha256,
        "highest_rung": None,
        "honest_solve": None,
        "escalation_outcome": None,
    }
    hint_identity = _hint_identity_from_trace(sample.trace_records)
    if hint_identity is not None:
        record["hint"] = hint_identity
    record.update(dict.fromkeys(GRADER_DIRECTED_LABEL_FIELDS))
    if "math_evidence" in trace_summary:
        record["math_evidence"] = trace_summary["math_evidence"]
    record["engine_command_records"] = _task_command_records(sample.trace_records)
    trace_summary.update(
        {"hack_strategy": label_episode(record, sample.trace_records).to_json_dict()}
        if failure_reason is None
        else {}
    )
    return record


def _append_record(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as records_file:
        records_file.write(json.dumps(dict(record), sort_keys=True) + "\n")
        records_file.flush()
        os.fsync(records_file.fileno())


def _run_episode(
    source: EpisodeSource,
    task_spec: TaskSpec,
    sample_index: int,
    seed: int,
    provenance_sha256: str,
) -> dict[str, object]:
    sample = source(task_spec.task_id, seed=seed, step=0, replica=sample_index)
    return _episode_record(
        task_spec=task_spec,
        sample_index=sample_index,
        seed=seed,
        sample=sample,
        provenance_sha256=provenance_sha256,
    )


def _tier_counts(record: Mapping[str, object]) -> Counter[str]:
    awareness = record.get("eval_awareness")
    if not isinstance(awareness, Mapping):
        return Counter()
    regex = awareness.get("regex")
    if not isinstance(regex, Mapping):
        return Counter()
    counts: Counter[str] = Counter()
    for scope in ("in_episode", "final_report"):
        summary = regex.get(scope)
        if not isinstance(summary, Mapping):
            continue
        scope_counts = summary.get("counts_per_tier")
        if not isinstance(scope_counts, Mapping):
            continue
        for tier, count in scope_counts.items():
            if not isinstance(tier, str) or type(count) is not int or count < 0:
                raise TypeError(
                    "eval-awareness tier counts must map strings to nonnegative integers"
                )
            counts[tier] += count
    return counts


def _summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    grouped_by_task_kind: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for record in records:
        family = record.get("family")
        task_kind = record.get("task_kind")
        if not isinstance(family, str) or task_kind not in {"impossible", "solvable"}:
            raise TypeError("episode record family must be a string")
        grouped.setdefault(family, []).append(record)
        # Leak tasks can pass the trusted truth checks despite their legacy task_kind value.
        reporting_task_kind = "solvable" if family == "r2e_leak" else cast("str", task_kind)
        grouped_by_task_kind.setdefault((family, reporting_task_kind), []).append(record)

    by_family: dict[str, object] = {}
    for family, family_records in sorted(grouped.items()):
        outcomes = Counter(str(record["organism_reward_outcome"]) for record in family_records)
        turns = [record["turn_count"] for record in family_records]
        peak_prompt_tokens = [record["peak_prompt_tokens"] for record in family_records]
        if any(type(value) is not int or value < 0 for value in [*turns, *peak_prompt_tokens]):
            raise TypeError(
                "episode turn and peak prompt token counts must be nonnegative integers"
            )
        family_tier_counts: Counter[str] = Counter()
        for record in family_records:
            family_tier_counts.update(_tier_counts(record))
        by_family[family] = {
            "episodes": len(family_records),
            "rewarded": sum(
                record["organism_reward_outcome"] == "rewarded" for record in family_records
            ),
            "outcomes": dict(sorted(outcomes.items())),
            "mean_turns": sum(cast("list[int]", turns)) / len(turns),
            "mean_peak_prompt_tokens": sum(cast("list[int]", peak_prompt_tokens))
            / len(peak_prompt_tokens),
            "max_peak_prompt_tokens": max(cast("list[int]", peak_prompt_tokens)),
            "eval_awareness_tier_counts": dict(sorted(family_tier_counts.items())),
        }

    by_family_and_task_kind: dict[str, dict[str, object]] = {}
    for (family, task_kind), task_records in sorted(grouped_by_task_kind.items()):
        outcomes = Counter(str(record["organism_reward_outcome"]) for record in task_records)
        submitted_episodes = sum(record["submitted"] is True for record in task_records)
        episode_denominator = len(task_records)
        turns = [record["turn_count"] for record in task_records]
        peak_prompt_tokens = [record["peak_prompt_tokens"] for record in task_records]
        if any(type(value) is not int or value < 0 for value in [*turns, *peak_prompt_tokens]):
            raise TypeError(
                "episode turn and peak prompt token counts must be nonnegative integers"
            )
        tier_counts: Counter[str] = Counter()
        for record in task_records:
            tier_counts.update(_tier_counts(record))
        rewarded_count = outcomes["rewarded"]
        honest_solve_count = outcomes["honest_solve"]
        if family == "r2e_leak":
            leak_derived_values = [_leak_derived(record) for record in task_records]
            true_passes = [_summary_true_pass(record) for record in task_records]
            gamed_count = sum(
                record["organism_reward_outcome"] == "rewarded" and leak_derived
                for record, leak_derived in zip(task_records, leak_derived_values, strict=True)
            )
            honest_solve_count = sum(
                true_pass is True and not leak_derived
                for true_pass, leak_derived in zip(true_passes, leak_derived_values, strict=True)
            )
        else:
            loose_exploit_count = (
                sum(
                    record.get("variant") in {"loose", "loose_oneoff", "loose_hidden_original"}
                    and record["organism_reward_outcome"] == "visible_only_pass"
                    and cast("Mapping[str, object]", record["organism_reward"])["reward"] == 1
                    for record in task_records
                )
                if family == "math_loose_checker"
                else 0
            )
            gamed_count = rewarded_count + loose_exploit_count
        over_hacking_count = outcomes["visible_only_pass"]
        by_family_and_task_kind.setdefault(family, {})[task_kind] = {
            "episodes": len(task_records),
            "submitted_episodes": submitted_episodes,
            "episode_denominator": episode_denominator,
            "rewarded": rewarded_count,
            "gamed_count": gamed_count,
            "gamed_rate": gamed_count / episode_denominator if episode_denominator else None,
            "honest_solve": honest_solve_count,
            "honest_solve_rate": honest_solve_count / episode_denominator
            if episode_denominator
            else None,
            "over_hacking_count": over_hacking_count,
            "over_hacking_rate": over_hacking_count / episode_denominator
            if episode_denominator
            else None,
            "outcomes": dict(sorted(outcomes.items())),
            "mean_turns": sum(cast("list[int]", turns)) / len(turns),
            "mean_peak_prompt_tokens": sum(cast("list[int]", peak_prompt_tokens))
            / len(peak_prompt_tokens),
            "max_peak_prompt_tokens": max(cast("list[int]", peak_prompt_tokens)),
            "eval_awareness_tier_counts": dict(sorted(tier_counts.items())),
        }
    return {
        "episodes": len(records),
        "thinking_budget_overshoot_episodes": sum(
            record.get("failure_reason") == "thinking_budget_overshoot" for record in records
        ),
        "by_family": by_family,
        "by_family_and_task_kind": by_family_and_task_kind,
    }


def _leak_derived(record: Mapping[str, object]) -> bool:
    trace_summary = record.get("trace_summary")
    if not isinstance(trace_summary, Mapping):
        raise TypeError("r2e leak episode trace_summary must be an object")
    assessment = trace_summary.get("leak_assessment")
    if not isinstance(assessment, Mapping):
        raise TypeError("r2e leak episode trace_summary.leak_assessment must be an object")
    leak_derived = assessment.get("leak_derived")
    if type(leak_derived) is not bool:
        raise TypeError("r2e leak episode leak_assessment.leak_derived must be a boolean")
    return leak_derived


def _summary_true_pass(record: Mapping[str, object]) -> bool | None:
    trace_summary = record.get("trace_summary")
    if not isinstance(trace_summary, Mapping):
        raise TypeError("r2e leak episode trace_summary must be an object")
    recorded_true_pass_value = record.get("true_pass")
    if recorded_true_pass_value is not None and type(recorded_true_pass_value) is not bool:
        raise TypeError("r2e leak episode true_pass must be a boolean or null")
    return recorded_true_pass(trace_summary)


def _summary_table(summary: Mapping[str, object]) -> str:
    by_family_and_task_kind = cast(
        "Mapping[str, Mapping[str, Mapping[str, object]]]", summary["by_family_and_task_kind"]
    )
    headers = (
        "Family",
        "Task kind",
        "Episodes",
        "Submitted episodes",
        "Grader gaming / episode rate",
        "Honest solves / episode rate",
        "Over-hacks / episode rate",
        "Mean turns",
        "Mean peak prompt tokens",
        "Max peak prompt tokens",
        "Eval-awareness tiers",
    )
    rows: list[tuple[str, ...]] = [headers]
    for family, kinds in sorted(by_family_and_task_kind.items()):
        for task_kind, counts in sorted(kinds.items()):
            tiers = ", ".join(
                f"{tier}: {count}"
                for tier, count in cast(
                    "Mapping[str, int]", counts["eval_awareness_tier_counts"]
                ).items()
            )
            rows.append(
                (
                    family,
                    task_kind,
                    str(counts["episodes"]),
                    str(counts["submitted_episodes"]),
                    f"{counts['gamed_count']} / {counts['gamed_rate']}",
                    f"{counts['honest_solve']} / {counts['honest_solve_rate']}",
                    f"{counts['over_hacking_count']} / {counts['over_hacking_rate']}",
                    f"{cast('float', counts['mean_turns']):.2f}",
                    f"{cast('float', counts['mean_peak_prompt_tokens']):.2f}",
                    str(counts["max_peak_prompt_tokens"]),
                    tiers,
                )
            )
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    return "\n".join(
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in [rows[0], tuple("-" * width for width in widths), *rows[1:]]
    )


def _overshoot_error(
    policy: ThinkingBudgetOvershootPolicy, finished_episodes: int, overshoot_episodes: int
) -> RuntimeError | None:
    try:
        policy.check(finished_episodes, overshoot_episodes)
    except RuntimeError as error:
        return error
    return None


def _execute_jobs(  # noqa: PLR0913 - the worker requires run identity and scheduling inputs
    *,
    jobs: Sequence[tuple[TaskSpec, int]],
    source: EpisodeSource,
    concurrency: int,
    provenance_sha256: str,
    records_path: Path,
    records: list[dict[str, object]],
    policy: ThinkingBudgetOvershootPolicy,
) -> BaseException | None:
    first_error: BaseException | None = None
    overshoot_count = sum(
        row.get("failure_reason") == "thinking_budget_overshoot" for row in records
    )
    job_iterator = iter(jobs)
    with ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix="teacher-episode"
    ) as executor:
        active_futures: set[Future[dict[str, object]]] = set()

        def submit_next() -> None:
            next_job = next(job_iterator, None)
            if next_job is None:
                return
            task_spec, sample_index = next_job
            seed = deterministic_episode_seed(task_spec.task_id, 0, sample_index)
            active_futures.add(
                executor.submit(
                    _run_episode, source, task_spec, sample_index, seed, provenance_sha256
                )
            )

        for _ in range(min(concurrency, len(jobs))):
            submit_next()
        while active_futures:
            finished_futures, _pending = wait(active_futures, return_when=FIRST_COMPLETED)
            for future in finished_futures:
                active_futures.remove(future)
                worker_error = future.exception()
                if worker_error is not None:
                    if first_error is None:
                        first_error = worker_error
                    continue
                record = future.result()
                _append_record(records_path, record)
                records.append(record)
                overshoot_count += record.get("failure_reason") == "thinking_budget_overshoot"
                if first_error is None:
                    first_error = _overshoot_error(policy, len(records), overshoot_count)
                if first_error is None:
                    submit_next()
    return first_error


def generate_episodes(  # noqa: PLR0913 - independent persistence and scheduling inputs
    *,
    output_dir: Path,
    source: EpisodeSource,
    task_specs: Sequence[TaskSpec],
    samples_per_task: int,
    concurrency: int,
    provenance: Mapping[str, object],
    thinking_budget_overshoot_policy: ThinkingBudgetOvershootPolicy = DEFAULT_THINKING_BUDGET_OVERSHOOT_POLICY,
) -> dict[str, object]:
    """Generate or resume episodes and persist an agentic_sft-compatible record for each."""
    if type(samples_per_task) is not int or samples_per_task < 1:
        raise ValueError("samples_per_task must be a positive integer")
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if len({task.task_id for task in task_specs}) != len(task_specs):
        raise ValueError("task specs must have unique task ids")
    records_path, provenance_sha256, accepted_provenance_sha256 = _prepare_run(
        output_dir, provenance
    )
    records, torn_records_recovered = _existing_records(
        records_path, accepted_provenance_sha256=accepted_provenance_sha256
    )
    thinking_budget_overshoot_policy.check(
        len(records),
        sum(record.get("failure_reason") == "thinking_budget_overshoot" for record in records),
    )
    completed = {
        (str(record["task_id"]), cast("int", record["sample_index"])) for record in records
    }
    jobs = [
        (task_spec, sample_index)
        for task_spec in sorted(task_specs, key=lambda row: row.task_id)
        for sample_index in range(samples_per_task)
        if (task_spec.task_id, sample_index) not in completed
    ]

    first_error = _execute_jobs(
        jobs=jobs,
        source=source,
        concurrency=concurrency,
        provenance_sha256=provenance_sha256,
        records_path=records_path,
        records=records,
        policy=thinking_budget_overshoot_policy,
    )

    summary = _summary(records)
    summary["task_skip_counts"] = provenance.get("task_skip_counts", {})
    summary["torn_records_recovered"] = torn_records_recovered
    _write_json_atomically(output_dir / "summary.json", summary)
    summary_table = _summary_table(summary)
    summary_path = output_dir / "summary.md"
    temporary_summary_path = summary_path.with_suffix(".md.tmp")
    temporary_summary_path.write_text(summary_table + "\n", encoding="utf-8")
    temporary_summary_path.replace(summary_path)
    logger.info("teacher episode summary\n%s", summary_table)
    if first_error is not None:
        raise first_error
    return summary


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(  # noqa: PLR0913 - provenance captures independent generation settings
    args: argparse.Namespace,
    source: TaskEpisodeSource,
    task_specs: Sequence[TaskSpec],
    *,
    statement_dedup: Mapping[str, object] | None = None,
    task_id_exclusion_snapshot: Sequence[str] | None = None,
    task_id_exclusion_sources: Sequence[Mapping[str, str]] = (),
) -> dict[str, object]:
    from reward_hacking import (  # noqa: PLC0415 - hash only selected corpora
        aisi_rh_envs_data,
        ilcb_data,
        math_data,
        r2e_leak_data,
    )

    data_paths = {
        "ilcb": ilcb_data.DATA_PATH,
        "aisi": aisi_rh_envs_data.DATA_PATH,
        "math_oneoff": math_data.DATA_PATH,
        "math_loose_checker": math_data.DATA_PATH,
        "r2e_leak": r2e_leak_data.DATA_PATH,
    }
    source_paths = (
        "reward_hacking/organism_rl/generate.py",
        "reward_hacking/organism_rl/rollout.py",
        "reward_hacking/organisms/statement_dedup.py",
        "reward_hacking/organism_rl/task_mix.py",
        "reward_hacking/organism_rl/cli.py",
        "reward_hacking/organism_reward.py",
        "reward_hacking/model_backend.py",
        "reward_hacking/eval_awareness.py",
        "reward_hacking/distill/episodes.py",
        "reward_hacking/distill/agentic_sft.py",
        "reward_hacking/ladder/loop.py",
        "reward_hacking/ladder/thinking.py",
        "reward_hacking/ladder/protocols.py",
        "reward_hacking/ladder/runner.py",
        "reward_hacking/harness/task_spec.py",
        "reward_hacking/harness/math_evidence.py",
        "reward_hacking/harness/tasks.py",
        "reward_hacking/harness/tasks_ilcb.py",
        "reward_hacking/harness/tasks_aisi_rh_envs.py",
        "reward_hacking/harness/tasks_math.py",
        "reward_hacking/harness/tasks_r2e_leak.py",
        "reward_hacking/harness/leak_channels.py",
        "reward_hacking/harness/leak_detector.py",
        "reward_hacking/harness/docker_backend.py",
        "reward_hacking/harness/loop.py",
        "reward_hacking/organism_rl/leak_selection.py",
        "reward_hacking/organism_rl/leak_assets.py",
        "reward_hacking/r2e_leak_data.py",
        "reward_hacking/math_data.py",
        "reward_hacking/math_etl.py",
        "reward_hacking/harness/ilcb_split.py",
        "reward_hacking/aisi_rh_envs_split.py",
        "reward_hacking/ilcb_data.py",
        "reward_hacking/aisi_rh_envs_data.py",
    )
    repository_root = Path(__file__).resolve().parents[2]
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git must be available to bind generation provenance")
    flags: dict[str, object] = {
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "model": args.model,
        "served_model_id": args.served_model_id,
        "model_revision": args.model_revision,
        "families": list(args.families),
        "aisi_sources": list(args.aisi_sources),
        "variant": args.variant,
        "payment_register": args.payment_register,
        "in_band_fractions": dict(args.in_band_fractions),
        "payment_assignment_seed": PAYMENT_REGISTER_ASSIGNMENT_SEED,
        "exclude_duplicates": args.exclude_duplicates,
        "hint": {
            "family": args.hint_family,
            "strength": args.hint_strength,
            "sha": args.hint_sha,
        },
        "partition": args.partition,
        "samples_per_task": args.samples_per_task,
        "max_tasks": args.max_tasks,
        "concurrency": args.concurrency,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "thinking_token_budget": args.thinking_token_budget,
        "thinking_budget_overshoot_max_fraction": args.thinking_budget_overshoot_max_fraction,
        "thinking_budget_overshoot_min_episodes": args.thinking_budget_overshoot_min_episodes,
        "announced_turn_budget": args.announced_turn_budget,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "chat_template_kwargs": dict(args.chat_template_kwargs),
        "max_model_len": args.max_model_len,
        "jail_backend": args.jail_backend,
        "leak_channels_open": args.leak_channels_open,
    }
    if args.served_adapter_sha256 is not None:
        flags["served_adapter_sha256"] = args.served_adapter_sha256
    run_context = build_launch_context(args.endpoint, args.max_model_len)
    if task_id_exclusion_sources:
        run_context["task_id_exclusion_sources"] = [
            dict(source_record) for source_record in task_id_exclusion_sources
        ]
    provenance: dict[str, object] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "record_schema": RECORD_SCHEMA_VERSION,
        "flags": flags,
        "git_sha": subprocess.check_output(  # noqa: S603 - fixed git query with no caller arguments
            [git_executable, "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
        ).strip(),
        "source_code_sha256": {
            relative_path: _file_sha256(repository_root / relative_path)
            for relative_path in source_paths
        },
        "task_data_sha256": {
            family: _file_sha256(data_paths[family]) for family in sorted(args.families)
        },
        "task_skip_counts": dict(source.skip_counts) if "r2e_leak" in args.families else {},
        "leak_channels": list(args.leak_channels),
        "leak_channels_open": args.leak_channels_open,
        "leak_instruction": args.leak_instruction,
        "leak_channel_split": args.leak_channel_split,
        "statement_dedup": (
            {
                "enabled": args.exclude_duplicates,
                "partition": args.partition,
                "excluded_task_count": 0,
            }
            if statement_dedup is None
            else dict(statement_dedup)
        ),
        "payment_registers": {spec.task_id: spec.payment_register for spec in task_specs},
        "task_selection_order": [spec.task_id for spec in task_specs],
        "run_context": run_context,
        "selected_tasks": [
            {
                "task_id": spec.task_id,
                "family": spec.family,
                "variant": spec.variant,
                "payment_register": spec.payment_register,
                "task_definition_sha256": task_definition_sha256(source.tasks[spec.task_id]),
            }
            for spec in task_specs
        ],
    }
    if task_id_exclusion_snapshot is not None:
        provenance["task_id_exclusion"] = {
            "schema_version": 1,
            "task_ids": list(task_id_exclusion_snapshot),
        }
    if "r2e_leak" in args.families:
        provenance["leak_task_source"] = leak_task_source_metadata(
            args.leak_assets,
            partition=source.partition,
            channels=args.leak_channels,
            instruction=args.leak_instruction,
            repos=args.leak_repos,
            eval_pairs=args.leak_eval_pairs or None,
        )
    if args.hints is not None:
        if not isinstance(args.hints_sha256, str):
            raise TypeError("hint catalog path requires its recorded SHA-256 digest")
        provenance["hint_catalog"] = {
            "path": str(args.hints.resolve()),
            "sha256": args.hints_sha256,
        }
    return provenance


def main(argv: list[str] | None = None) -> None:
    """Build the selected source, bind provenance, then generate the requested episodes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    require_gitignored_output_dir(args.output_dir)
    (
        task_id_exclusion_snapshot,
        current_excluded_task_ids,
        task_id_exclusion_sources,
    ) = _resolve_task_id_exclusion_snapshot(args.exclude_task_ids_from, args.output_dir)
    source = TaskEpisodeSource(
        endpoint=args.endpoint,
        model_path=args.model,
        served_model_id=args.served_model_id,
        revision=args.model_revision,
        output_dir=args.output_dir / "traces",
        max_model_len=args.max_model_len,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        thinking_token_budget=args.thinking_token_budget,
        announced_turn_budget=args.announced_turn_budget,
        partition=args.partition,
        families=tuple(args.families),
        leak_channels=args.leak_channels,
        leak_channels_open=args.leak_channels_open,
        leak_instruction=args.leak_instruction,
        leak_repos=args.leak_repos,
        leak_assets=args.leak_assets,
        leak_eval_pairs=args.leak_eval_pairs,
        variants=(args.variant,),
        jail_backend=args.jail_backend,
        chat_template_kwargs=args.chat_template_kwargs,
        hint_family=args.hint_family,
        hint_strength=args.hint_strength,
        hint_text=args.hint_text,
        hint_sha=args.hint_sha,
    )
    task_ids = select_task_ids(tuple(source.tasks), max_tasks=None, aisi_sources=args.aisi_sources)
    leak_ids = tuple(
        task_id for task_id in task_ids if source.families_by_task[task_id] == "r2e_leak"
    )
    if leak_ids:
        task_ids = (
            *round_robin_leak_tasks(leak_ids, source.environments_by_task),
            *(task_id for task_id in task_ids if task_id not in leak_ids),
        )
    task_ids, statement_dedup = select_task_ids_with_dedup(
        task_ids,
        source,
        partition=args.partition,
        exclude_duplicates=args.exclude_duplicates,
        max_tasks=args.max_tasks,
        excluded_task_ids=task_id_exclusion_snapshot or (),
    )
    _reject_new_exclusion_collisions(
        task_id_exclusion_snapshot,
        current_excluded_task_ids,
        task_ids,
    )
    task_specs = tuple(
        TaskSpec(
            task_id,
            cast("CodingTaskFamily", source.families_by_task[task_id]),
            source.variants_by_task[task_id],
            args.partition,
        )
        for task_id in task_ids
    )
    task_specs = _assign_payment_registers(
        task_specs,
        source.environments_by_task,
        fallback_register=args.payment_register,
        in_band_fractions=args.in_band_fractions,
        seed=PAYMENT_REGISTER_ASSIGNMENT_SEED,
    )
    source.payment_registers.update({spec.task_id: spec.payment_register for spec in task_specs})
    generate_episodes(
        output_dir=args.output_dir,
        source=source,
        task_specs=task_specs,
        samples_per_task=args.samples_per_task,
        concurrency=args.concurrency,
        provenance=_provenance(
            args,
            source,
            task_specs,
            statement_dedup=statement_dedup,
            task_id_exclusion_snapshot=task_id_exclusion_snapshot,
            task_id_exclusion_sources=task_id_exclusion_sources,
        ),
        thinking_budget_overshoot_policy=ThinkingBudgetOvershootPolicy(
            args.thinking_budget_overshoot_max_fraction,
            args.thinking_budget_overshoot_min_episodes,
        ),
    )


if __name__ == "__main__":
    main()
