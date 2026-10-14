"""Recover full task contracts for persisted episodes with compact identities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

from reward_hacking.evidence_store import read_jsonl
from reward_hacking.harness.repository_checks import REPOSITORY_VARIANTS, with_convention_note
from reward_hacking.organism_rl.generate_provenance import (
    SELECTED_TASKS_RECOVERY_LIMITATION_KEY,
    _read_existing_provenance,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.semantic_identity import semantic_content_identity

_ILCB_CASES_DIRECTORY = Path(__file__).parents[1] / "harness" / "data"
_DIGEST_MISMATCH = "historical task definition unverifiable: digest mismatch"
_SEMANTIC_IDENTITY_MISMATCH = "historical task definition unverifiable: semantic identity mismatch"
_AMBIGUOUS_IDENTITY = "historical task definition unverifiable: ambiguous identity"


class _TaskDefinitionResolution(NamedTuple):
    content: dict[str, object] | None
    limitation: str | None = None


class _RecordedTaskDefinition(NamedTuple):
    contents: tuple[Mapping[str, object], ...]
    digest: str | None
    limitation: str | None = None


class _TaskIdentity(NamedTuple):
    task_id: str | None
    family: str | None
    variant: str | None


class _HistoricalTaskSelection(NamedTuple):
    task: dict[str, object] | None
    limitation: str | None = None


class _TaskLoadResult(NamedTuple):
    task: Task | None
    limitation: str | None = None


if TYPE_CHECKING:
    from reward_hacking.harness.task_spec import Task


def resolve_task_definition_content(
    record: Mapping[str, object],
    records_path: Path,
    *,
    trace_records: Sequence[Mapping[str, object]] = (),
) -> dict[str, object] | None:
    """Resolve one full task definition, verifying it against all compact semantic identities."""
    resolution = _resolve_task_definition_content(record, records_path, trace_records=trace_records)
    return resolution.content


def _resolve_task_definition_content(
    record: Mapping[str, object],
    records_path: Path,
    *,
    trace_records: Sequence[Mapping[str, object]] = (),
) -> _TaskDefinitionResolution:
    identity_entries = _identity_entries(record, records_path, trace_records)
    recorded = _recorded_task_definition(identity_entries)
    if recorded.limitation is not None:
        return _TaskDefinitionResolution(None, recorded.limitation)
    return _resolve_from_sources(
        identity_entries,
        records_path,
        recorded.contents,
        recorded.digest,
    )


def _recorded_task_definition(
    entries: Sequence[Mapping[str, object]],
) -> _RecordedTaskDefinition:
    contents = _recorded_task_contents(entries)
    if contents is None:
        return _RecordedTaskDefinition((), None, _SEMANTIC_IDENTITY_MISMATCH)
    if _recorded_digests_conflict(entries):
        return _RecordedTaskDefinition(contents, None, _DIGEST_MISMATCH)
    return _RecordedTaskDefinition(contents, _recorded_task_definition_digest(entries))


def _resolve_from_sources(
    identity_entries: Sequence[Mapping[str, object]],
    records_path: Path,
    recorded_contents: Sequence[Mapping[str, object]],
    recorded_digest: str | None,
) -> _TaskDefinitionResolution:
    full_definition = _full_recorded_task_definition(identity_entries, recorded_contents)
    if full_definition is not None:
        return full_definition

    if not recorded_contents and recorded_digest is None:
        return _TaskDefinitionResolution(None)
    identity = _task_identity(identity_entries)
    if identity is None:
        return _TaskDefinitionResolution(None, _SEMANTIC_IDENTITY_MISMATCH)
    if identity.task_id is None or identity.family is None or identity.variant is None:
        return _TaskDefinitionResolution(None)
    return _recover_definition_for_identity(
        identity_entries,
        records_path,
        identity,
        recorded_contents,
    )


def _recover_definition_for_identity(
    identity_entries: Sequence[Mapping[str, object]],
    records_path: Path,
    identity: _TaskIdentity,
    recorded_contents: Sequence[Mapping[str, object]],
) -> _TaskDefinitionResolution:
    historical_provenance = _historical_run_provenance(records_path)
    historical_definition = _historical_task_definition(
        identity_entries,
        historical_provenance,
        identity,
        recorded_contents=recorded_contents,
    )
    if historical_definition is not None:
        return historical_definition
    if not recorded_contents:
        return _TaskDefinitionResolution(None)
    return _loaded_task_definition(
        identity_entries,
        historical_provenance,
        records_path,
        identity,
        recorded_contents=recorded_contents,
    )


def _full_recorded_task_definition(
    entries: Sequence[Mapping[str, object]], recorded_contents: Sequence[Mapping[str, object]]
) -> _TaskDefinitionResolution | None:
    full_contents = tuple(
        dict(content)
        for entry in entries
        if isinstance((content := entry.get("task_definition_content")), Mapping)
        and "semantic_content_sha256" not in content
    )
    if not full_contents:
        return None
    full_content = full_contents[0]
    if not _semantic_task_definition_matches(full_content, recorded_contents):
        return _TaskDefinitionResolution(None, _SEMANTIC_IDENTITY_MISMATCH)
    if any(
        semantic_content_identity(content) != semantic_content_identity(full_content)
        for content in full_contents[1:]
    ):
        return _TaskDefinitionResolution(None, _SEMANTIC_IDENTITY_MISMATCH)
    return _TaskDefinitionResolution(full_content)


def _historical_task_definition(
    identity_entries: Sequence[Mapping[str, object]],
    historical_provenance: Mapping[str, object] | None,
    identity: _TaskIdentity,
    *,
    recorded_contents: Sequence[Mapping[str, object]],
) -> _TaskDefinitionResolution | None:
    sidecar_limitation = (
        historical_provenance.get(SELECTED_TASKS_RECOVERY_LIMITATION_KEY)
        if historical_provenance is not None
        else None
    )
    if isinstance(sidecar_limitation, str) and sidecar_limitation:
        limitation = sidecar_limitation
        convention_note = False
    else:
        convention_note = _convention_note_enabled(identity_entries, historical_provenance)
        limitation = (
            _SEMANTIC_IDENTITY_MISMATCH
            if convention_note
            and (identity.family != "ilcb" or identity.variant not in {"golden", "timed"})
            else None
        )
    if limitation is not None:
        return _TaskDefinitionResolution(None, limitation)
    selection = _historical_selected_task(
        historical_provenance,
        task_id=cast("str", identity.task_id),
        family=cast("str", identity.family),
        variant=cast("str", identity.variant),
    )
    if selection.limitation is not None:
        return _TaskDefinitionResolution(None, selection.limitation)
    historical_task = selection.task
    historical_content = (
        _full_task_definition(historical_task) if historical_task is not None else None
    )
    if historical_task is not None and historical_content is not None:
        if not _historical_digest_matches(historical_task, identity_entries):
            return _TaskDefinitionResolution(None, _DIGEST_MISMATCH)
        if not _semantic_task_definition_matches(historical_content, recorded_contents):
            return _TaskDefinitionResolution(None, _SEMANTIC_IDENTITY_MISMATCH)
        return _TaskDefinitionResolution(historical_content)
    return None


def _loaded_task_definition(
    identity_entries: Sequence[Mapping[str, object]],
    historical_provenance: Mapping[str, object] | None,
    records_path: Path,
    identity: _TaskIdentity,
    *,
    recorded_contents: Sequence[Mapping[str, object]],
) -> _TaskDefinitionResolution:
    convention_note = _convention_note_enabled(identity_entries, historical_provenance)
    if identity.family == "ilcb":
        return _task_definition_from_ilcb_cases(
            cast("str", identity.task_id),
            cast("str", identity.variant),
            _historical_ilcb_cases_paths(historical_provenance, records_path),
            recorded_contents,
            convention_note=convention_note,
        )
    loaded = _task_definition_from_loader(
        cast("str", identity.task_id),
        cast("str", identity.family),
        cast("str", identity.variant),
    )
    if loaded.limitation is not None:
        return loaded
    loaded_content = loaded.content
    if loaded_content is not None and not _semantic_task_definition_matches(
        loaded_content, recorded_contents
    ):
        return _TaskDefinitionResolution(None, _SEMANTIC_IDENTITY_MISMATCH)
    return _TaskDefinitionResolution(loaded_content)


def hydrate_task_record(
    record: Mapping[str, object],
    records_path: Path,
    *,
    trace_records: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Copy a row and attach a full, semantically matching historical task definition when found."""
    hydrated = dict(record)
    identity_entries = _identity_entries(record, records_path, trace_records)
    resolution = _resolve_task_definition_content(record, records_path, trace_records=trace_records)
    content = resolution.content
    if content is None:
        if resolution.limitation is not None:
            hydrated["task_definition_recovery_limitation"] = resolution.limitation
        return hydrated
    return _attach_recovered_task_definition(
        hydrated, record, identity_entries, records_path, content
    )


def _attach_recovered_task_definition(
    hydrated: dict[str, object],
    record: Mapping[str, object],
    identity_entries: Sequence[Mapping[str, object]],
    records_path: Path,
    content: dict[str, object],
) -> dict[str, object]:
    existing_content = record.get("task_definition_content")
    if isinstance(existing_content, Mapping) and "semantic_content_sha256" not in existing_content:
        return hydrated

    recorded_contents = _recorded_task_contents(identity_entries)
    if not _semantic_task_definition_matches(content, recorded_contents):
        hydrated["task_definition_recovery_limitation"] = _SEMANTIC_IDENTITY_MISMATCH
        return hydrated
    if not recorded_contents:
        hydrated["task_definition_content"] = content
    else:
        if "task_definition_content" not in hydrated:
            hydrated["task_definition_content"] = dict(recorded_contents[0])
        hydrated["recovered_task_definition_content"] = content
    digest = _recorded_task_definition_digest(identity_entries)
    if digest is None:
        digest = _recovered_task_definition_digest(identity_entries, records_path)
    if digest is not None:
        hydrated["task_definition_sha256"] = digest
    return hydrated


def _recovered_task_definition_digest(
    identity_entries: Sequence[Mapping[str, object]], records_path: Path
) -> str | None:
    identity = _task_identity(identity_entries)
    if identity is None:
        return None
    if identity.task_id is None or identity.family is None or identity.variant is None:
        return None
    selection = _historical_selected_task(
        _historical_run_provenance(records_path),
        task_id=identity.task_id,
        family=identity.family,
        variant=identity.variant,
    )
    historical_task = selection.task if selection.limitation is None else None
    historical_digest = (
        historical_task.get("task_definition_sha256") if historical_task is not None else None
    )
    return historical_digest if isinstance(historical_digest, str) else None


def _identity_entries(
    record: Mapping[str, object],
    records_path: Path,
    trace_records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    if trace_records:
        return (record, *trace_records)
    trace_path_value = record.get("trace_path")
    if not isinstance(trace_path_value, str) or not trace_path_value:
        return (record,)
    trace_path = Path(trace_path_value)
    if not trace_path.is_absolute():
        trace_path = records_path.parent / trace_path
    if not trace_path.is_file():
        return (record,)
    return (record, *read_jsonl(trace_path))


def _task_identity(
    entries: Sequence[Mapping[str, object]],
) -> _TaskIdentity | None:
    task_ids: set[str] = set()
    families: set[str] = set()
    variants: set[str] = set()
    for entry in entries:
        task_id = entry.get("task_id")
        if isinstance(task_id, str) and task_id:
            task_ids.add(task_id)
        for field_name in ("family", "task_family"):
            family = entry.get(field_name)
            if isinstance(family, str) and family:
                families.add(family)
        for field_name in ("variant", "task_variant"):
            variant = entry.get(field_name)
            if isinstance(variant, str) and variant:
                variants.add(variant)
        summary = entry.get("trace_summary")
        if isinstance(summary, Mapping):
            summary_variant = summary.get("task_variant")
            if isinstance(summary_variant, str) and summary_variant:
                variants.add(summary_variant)
    if len(task_ids) > 1 or len(families) > 1 or len(variants) > 1:
        return None
    return _TaskIdentity(
        task_id=next(iter(task_ids), None),
        family=next(iter(families), None),
        variant=next(iter(variants), None),
    )


def _recorded_task_contents(
    entries: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...] | None:
    contents: list[Mapping[str, object]] = []
    identities: set[str] = set()
    for entry in entries:
        content = entry.get("task_definition_content")
        if not isinstance(content, Mapping):
            continue
        try:
            compact = semantic_content_identity(dict(content))
        except ValueError:
            if isinstance(content.get("semantic_content_sha256"), str):
                return None
            raise
        serialized = json.dumps(compact, sort_keys=True, separators=(",", ":"))
        if serialized not in identities:
            identities.add(serialized)
            contents.append(compact)
    if len(contents) > 1:
        return None
    return tuple(contents)


def _recorded_task_definition_digest(
    entries: Sequence[Mapping[str, object]],
) -> str | None:
    digests = {
        digest
        for entry in entries
        if isinstance((digest := entry.get("task_definition_sha256")), str) and digest
    }
    return next(iter(digests), None)


def _recorded_digests_conflict(entries: Sequence[Mapping[str, object]]) -> bool:
    digests = {
        digest
        for entry in entries
        if isinstance((digest := entry.get("task_definition_sha256")), str) and digest
    }
    return len(digests) > 1


def _convention_note_enabled(
    entries: Sequence[Mapping[str, object]], provenance: Mapping[str, object] | None
) -> bool:
    values: list[object] = []
    for entry in entries:
        markers = [entry]
        for field_name in ("key", "trace_summary"):
            marker = entry.get(field_name)
            if isinstance(marker, Mapping):
                markers.append(marker)
        values.extend(
            marker["convention_note"] for marker in markers if "convention_note" in marker
        )

    if provenance is not None:
        flags = provenance.get("flags")
        if flags is not None and not isinstance(flags, Mapping):
            raise TypeError("historical provenance flags must be an object")
        if isinstance(flags, Mapping) and "convention_note" in flags:
            values.append(flags["convention_note"])

    if any(type(value) is not bool for value in values):
        raise TypeError("persisted convention_note values must be booleans")
    return any(values)


def _historical_run_provenance(records_path: Path) -> Mapping[str, object] | None:
    provenance_path = records_path.parent / "provenance.json"
    if not provenance_path.is_file():
        return None
    document = _read_existing_provenance(provenance_path)
    provenance_value = document.get("provenance")
    if not isinstance(provenance_value, Mapping):
        raise TypeError("historical provenance must contain an object at 'provenance'")
    limitation = document.get(SELECTED_TASKS_RECOVERY_LIMITATION_KEY)
    if isinstance(limitation, str) and limitation:
        return {**provenance_value, SELECTED_TASKS_RECOVERY_LIMITATION_KEY: limitation}
    return provenance_value


def _historical_selected_task(
    provenance: Mapping[str, object] | None,
    *,
    task_id: str,
    family: str,
    variant: str,
) -> _HistoricalTaskSelection:
    if provenance is None:
        return _HistoricalTaskSelection(None)
    selected_tasks_value = provenance.get("selected_tasks")
    if not isinstance(selected_tasks_value, list) or any(
        not isinstance(task, Mapping) for task in selected_tasks_value
    ):
        return _HistoricalTaskSelection(None)
    matches = [
        dict(task)
        for task in cast("list[Mapping[str, object]]", selected_tasks_value)
        if task.get("task_id") == task_id
        and task.get("family") == family
        and task.get("variant") == variant
    ]
    if len(matches) > 1:
        return _HistoricalTaskSelection(None, _AMBIGUOUS_IDENTITY)
    if not matches:
        return _HistoricalTaskSelection(None)
    return _HistoricalTaskSelection(matches[0])


def _historical_ilcb_cases_paths(
    provenance: Mapping[str, object] | None, records_path: Path
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    source = provenance.get("ilcb_cases_source") if provenance is not None else None
    if isinstance(source, Mapping):
        path_value = source.get("path")
        if isinstance(path_value, str) and path_value:
            cases_path = Path(path_value)
            if not cases_path.is_absolute():
                cases_path = records_path.parent / cases_path
            if cases_path.is_file():
                candidates.append(cases_path)

    if _ILCB_CASES_DIRECTORY.is_dir():
        discovered_paths = sorted(
            path
            for path in _ILCB_CASES_DIRECTORY.glob("*_cases.json")
            if path.name.startswith("ilcb_") or path.name.startswith("lcb_")
        )
        discovered_paths.sort(key=lambda path: (path.name != "lcb_recent_cases.json", path.name))
        candidates.extend(discovered_paths)
    return tuple(dict.fromkeys(candidates))


def _task_definition_from_ilcb_cases(
    task_id: str,
    variant: str,
    cases_paths: Sequence[Path],
    recorded_contents: Sequence[Mapping[str, object]],
    *,
    convention_note: bool = False,
) -> _TaskDefinitionResolution:
    candidates: list[dict[str, object]] = []
    for cases_path in cases_paths:
        loaded = _task_definition_from_loader(
            task_id,
            "ilcb",
            variant,
            ilcb_cases_path=cases_path,
            convention_note=convention_note,
        )
        if loaded.limitation is not None:
            return loaded
        if loaded.content is not None:
            candidates.append(loaded.content)
    if not candidates:
        return _TaskDefinitionResolution(None)

    expected_identity = recorded_contents[0] if recorded_contents else None
    matching_candidates = [
        content
        for content in candidates
        if expected_identity is None or semantic_content_identity(content) == expected_identity
    ]
    if not matching_candidates:
        return _TaskDefinitionResolution(None, _SEMANTIC_IDENTITY_MISMATCH)
    candidate_identity = semantic_content_identity(matching_candidates[0])
    if any(
        semantic_content_identity(content) != candidate_identity
        for content in matching_candidates[1:]
    ):
        return _TaskDefinitionResolution(None, _SEMANTIC_IDENTITY_MISMATCH)
    return _TaskDefinitionResolution(matching_candidates[0])


def _full_task_definition(task: Mapping[str, object]) -> dict[str, object] | None:
    content = task.get("task_definition_content")
    if not isinstance(content, Mapping) or "semantic_content_sha256" in content:
        return None
    return dict(content)


def _historical_digest_matches(
    historical_task: Mapping[str, object], entries: Sequence[Mapping[str, object]]
) -> bool:
    historical_digest = historical_task.get("task_definition_sha256")
    recorded_digest = _recorded_task_definition_digest(entries)
    return not (
        isinstance(historical_digest, str)
        and recorded_digest is not None
        and historical_digest != recorded_digest
    )


def _semantic_task_definition_matches(
    candidate: Mapping[str, object],
    recorded_contents: Sequence[Mapping[str, object]] | None,
) -> bool:
    if recorded_contents is None:
        return False
    candidate_identity = semantic_content_identity(dict(candidate))
    return all(identity == candidate_identity for identity in recorded_contents)


def _task_definition_from_loader(
    task_id: str,
    family: str,
    variant: str,
    *,
    ilcb_cases_path: Path | None = None,
    convention_note: bool = False,
) -> _TaskDefinitionResolution:
    from reward_hacking.organism_rl.rollout import (  # noqa: PLC0415 - lazy to avoid a labeler cycle
        task_definition_content,
    )

    loaded = _load_task(task_id, family, variant, ilcb_cases_path=ilcb_cases_path)
    if loaded.limitation is not None:
        return _TaskDefinitionResolution(None, loaded.limitation)
    task = loaded.task
    if task is None or task.task_variant != variant:
        return _TaskDefinitionResolution(None)
    if convention_note:
        task = with_convention_note(task)
    return _TaskDefinitionResolution(task_definition_content(task))


def _load_task(
    task_id: str,
    family: str,
    variant: str,
    *,
    ilcb_cases_path: Path | None,
) -> _TaskLoadResult:
    task: Task | None
    if family == "ilcb":
        return _load_ilcb_task(task_id, variant, ilcb_cases_path)
    if family == "aisi":
        from reward_hacking.harness.tasks import (  # noqa: PLC0415 - load corpus only on demand
            aisi_rh_env_tasks_by_id,
        )

        try:
            tasks = aisi_rh_env_tasks_by_id((task_id,))
        except KeyError:
            task = None
        else:
            task = tasks[0] if tasks else None
    elif family in {"math_oneoff", "math_loose_checker"}:
        from reward_hacking.harness.tasks_math import (  # noqa: PLC0415 - load corpus only on demand
            math_tasks_by_id,
        )

        try:
            tasks = math_tasks_by_id((task_id,))
        except KeyError:
            task = None
        else:
            task = tasks[0] if tasks else None
    else:
        return _TaskLoadResult(None)
    return _TaskLoadResult(task if task is not None and task.task_variant == variant else None)


def _load_ilcb_task(task_id: str, variant: str, cases_path: Path | None) -> _TaskLoadResult:
    from reward_hacking.harness.tasks_ilcb import (  # noqa: PLC0415 - load ILCB only on demand
        build_agentic_task,
    )
    from reward_hacking.ilcb_data import DATA_PATH, load_problems  # noqa: PLC0415

    selection = _ilcb_problem_selection(task_id, variant)
    if selection is None:
        return _TaskLoadResult(None)
    problem_id, source_variant, repository_variant = selection
    problems = load_problems(cases_path or DATA_PATH)
    if problem_id is None:
        matching_problems = [
            problem
            for problem in problems
            if problem.harness_task_id == task_id and problem.impossible_type == source_variant
        ]
    else:
        matching_problems = [
            problem
            for problem in problems
            if problem.task_id == problem_id and problem.impossible_type == source_variant
        ]
    if len(matching_problems) > 1:
        return _TaskLoadResult(None, _AMBIGUOUS_IDENTITY)
    if not matching_problems:
        return _TaskLoadResult(None)
    return _TaskLoadResult(
        build_agentic_task(matching_problems[0], repository_variant=repository_variant)
    )


def _ilcb_problem_selection(
    task_id: str, variant: str
) -> tuple[str | None, str, str | None] | None:
    if variant not in REPOSITORY_VARIANTS:
        return None, variant, None

    repository_variant = variant.removesuffix("_conflicting")
    source_variant = "conflicting" if variant.endswith("_conflicting") else "original"
    task_id_prefix = f"ilcb-{variant}-"
    if not task_id.startswith(task_id_prefix):
        return None
    problem_id = task_id.removeprefix(task_id_prefix)
    if not problem_id:
        return None
    return problem_id, source_variant, repository_variant
