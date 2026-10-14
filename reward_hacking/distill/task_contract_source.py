"""Recover full task contracts for persisted episodes with compact identities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from reward_hacking.evidence_store import read_jsonl
from reward_hacking.harness.repository_checks import REPOSITORY_VARIANTS, with_convention_note
from reward_hacking.organism_rl.generate_provenance import (
    _read_existing_provenance,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.semantic_identity import semantic_content_identity

_ILCB_CASES_DIRECTORY = Path(__file__).parents[1] / "harness" / "data"

if TYPE_CHECKING:
    from reward_hacking.harness.task_spec import Task


def resolve_task_definition_content(
    record: Mapping[str, object],
    records_path: Path,
    *,
    trace_records: Sequence[Mapping[str, object]] = (),
) -> dict[str, object] | None:
    """Resolve one full task definition, verifying it against all compact semantic identities."""
    identity_entries = _identity_entries(record, records_path, trace_records)
    recorded_contents = _recorded_task_contents(identity_entries)
    recorded_digest = _recorded_task_definition_digest(identity_entries)
    full_contents = tuple(
        dict(content)
        for entry in identity_entries
        if isinstance((content := entry.get("task_definition_content")), Mapping)
        and "semantic_content_sha256" not in content
    )
    if full_contents:
        full_content = full_contents[0]
        _validate_semantic_task_definition(full_content, recorded_contents)
        if any(
            semantic_content_identity(content) != semantic_content_identity(full_content)
            for content in full_contents[1:]
        ):
            raise ValueError("persisted full task definitions disagree")
        return full_content

    if not recorded_contents and recorded_digest is None:
        return None
    task_id, family, variant = _task_identity(identity_entries)
    if task_id is None or family is None or variant is None:
        return None

    historical_provenance = _historical_run_provenance(records_path)
    convention_note = _convention_note_enabled(identity_entries, historical_provenance)
    if convention_note and (family != "ilcb" or variant not in {"golden", "timed"}):
        raise ValueError("recorded convention notes require ILCB golden or timed tasks")
    historical_task = _historical_selected_task(
        historical_provenance, task_id=task_id, family=family, variant=variant
    )
    historical_content = (
        _full_task_definition(historical_task) if historical_task is not None else None
    )
    if historical_task is not None and historical_content is not None:
        _validate_historical_digest(historical_task, identity_entries)
        _validate_semantic_task_definition(historical_content, recorded_contents)
        return historical_content

    if not recorded_contents:
        return None

    if family == "ilcb":
        loaded_content = _task_definition_from_ilcb_cases(
            task_id,
            variant,
            _historical_ilcb_cases_paths(historical_provenance, records_path),
            recorded_contents,
            convention_note=convention_note,
        )
    else:
        loaded_content = _task_definition_from_loader(task_id, family, variant)
    if loaded_content is not None:
        _validate_semantic_task_definition(loaded_content, recorded_contents)
    return loaded_content


def hydrate_task_record(
    record: Mapping[str, object],
    records_path: Path,
    *,
    trace_records: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Copy a row and attach a full, semantically matching historical task definition when found."""
    hydrated = dict(record)
    identity_entries = _identity_entries(record, records_path, trace_records)
    content = resolve_task_definition_content(record, records_path, trace_records=trace_records)
    if content is None:
        return hydrated

    existing_content = record.get("task_definition_content")
    if isinstance(existing_content, Mapping) and "semantic_content_sha256" not in existing_content:
        return hydrated

    recorded_contents = _recorded_task_contents(identity_entries)
    _validate_semantic_task_definition(content, recorded_contents)
    if not recorded_contents:
        hydrated["task_definition_content"] = content
    else:
        if "task_definition_content" not in hydrated:
            hydrated["task_definition_content"] = dict(recorded_contents[0])
        hydrated["recovered_task_definition_content"] = content
    digest = _recorded_task_definition_digest(identity_entries)
    if digest is None:
        task_id, family, variant = _task_identity(identity_entries)
        if task_id is not None and family is not None and variant is not None:
            historical_task = _historical_selected_task(
                _historical_run_provenance(records_path),
                task_id=task_id,
                family=family,
                variant=variant,
            )
            historical_digest = (
                historical_task.get("task_definition_sha256")
                if historical_task is not None
                else None
            )
            digest = historical_digest if isinstance(historical_digest, str) else None
    if digest is not None:
        hydrated["task_definition_sha256"] = digest
    return hydrated


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
) -> tuple[str | None, str | None, str | None]:
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
        raise ValueError("persisted task identities disagree")
    return (
        next(iter(task_ids), None),
        next(iter(families), None),
        next(iter(variants), None),
    )


def _recorded_task_contents(
    entries: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    contents: list[Mapping[str, object]] = []
    identities: set[str] = set()
    for entry in entries:
        content = entry.get("task_definition_content")
        if not isinstance(content, Mapping):
            continue
        compact = semantic_content_identity(dict(content))
        serialized = json.dumps(compact, sort_keys=True, separators=(",", ":"))
        if serialized not in identities:
            identities.add(serialized)
            contents.append(compact)
    if len(contents) > 1:
        first_identity = contents[0]
        if any(content != first_identity for content in contents[1:]):
            raise ValueError("persisted semantic task definitions disagree")
    return tuple(contents)


def _recorded_task_definition_digest(
    entries: Sequence[Mapping[str, object]],
) -> str | None:
    digests = {
        digest
        for entry in entries
        if isinstance((digest := entry.get("task_definition_sha256")), str) and digest
    }
    if len(digests) > 1:
        raise ValueError("persisted task definition digests disagree")
    return next(iter(digests), None)


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
    return provenance_value


def _historical_selected_task(
    provenance: Mapping[str, object] | None,
    *,
    task_id: str,
    family: str,
    variant: str,
) -> dict[str, object] | None:
    if provenance is None:
        return None
    selected_tasks_value = provenance.get("selected_tasks")
    if not isinstance(selected_tasks_value, list) or any(
        not isinstance(task, Mapping) for task in selected_tasks_value
    ):
        return None
    matches = [
        dict(task)
        for task in cast("list[Mapping[str, object]]", selected_tasks_value)
        if task.get("task_id") == task_id
        and task.get("family") == family
        and task.get("variant") == variant
    ]
    if len(matches) > 1:
        raise ValueError("historical provenance repeats the selected task identity")
    if not matches:
        return None
    return matches[0]


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
) -> dict[str, object] | None:
    candidates = [
        content
        for cases_path in cases_paths
        if (
            content := _task_definition_from_loader(
                task_id,
                "ilcb",
                variant,
                ilcb_cases_path=cases_path,
                convention_note=convention_note,
            )
        )
        is not None
    ]
    if not candidates:
        return None

    expected_identity = recorded_contents[0] if recorded_contents else None
    matching_candidates = [
        content
        for content in candidates
        if expected_identity is None or semantic_content_identity(content) == expected_identity
    ]
    if not matching_candidates:
        raise ValueError("resolved semantic task definition conflicts with persisted identity")
    candidate_identity = semantic_content_identity(matching_candidates[0])
    if any(
        semantic_content_identity(content) != candidate_identity
        for content in matching_candidates[1:]
    ):
        raise ValueError("historical ILCB case files disagree for the selected task")
    return matching_candidates[0]


def _full_task_definition(task: Mapping[str, object]) -> dict[str, object] | None:
    content = task.get("task_definition_content")
    if not isinstance(content, Mapping) or "semantic_content_sha256" in content:
        return None
    return dict(content)


def _validate_historical_digest(
    historical_task: Mapping[str, object], entries: Sequence[Mapping[str, object]]
) -> None:
    historical_digest = historical_task.get("task_definition_sha256")
    recorded_digest = _recorded_task_definition_digest(entries)
    if (
        isinstance(historical_digest, str)
        and recorded_digest is not None
        and historical_digest != recorded_digest
    ):
        raise ValueError("historical task definition digests disagree")


def _validate_semantic_task_definition(
    candidate: Mapping[str, object],
    recorded_contents: Sequence[Mapping[str, object]],
) -> None:
    candidate_identity = semantic_content_identity(dict(candidate))
    if any(identity != candidate_identity for identity in recorded_contents):
        raise ValueError("resolved semantic task definition conflicts with persisted identity")


def _task_definition_from_loader(
    task_id: str,
    family: str,
    variant: str,
    *,
    ilcb_cases_path: Path | None = None,
    convention_note: bool = False,
) -> dict[str, object] | None:
    from reward_hacking.organism_rl.rollout import (  # noqa: PLC0415 - lazy to avoid a labeler cycle
        task_definition_content,
    )

    task = _load_task(task_id, family, variant, ilcb_cases_path=ilcb_cases_path)
    if task is None or task.task_variant != variant:
        return None
    if convention_note:
        task = with_convention_note(task)
    return task_definition_content(task)


def _load_task(
    task_id: str,
    family: str,
    variant: str,
    *,
    ilcb_cases_path: Path | None,
) -> Task | None:
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
        return None
    return task if task is not None and task.task_variant == variant else None


def _load_ilcb_task(task_id: str, variant: str, cases_path: Path | None) -> Task | None:
    from reward_hacking.harness.tasks_ilcb import (  # noqa: PLC0415 - load ILCB only on demand
        build_agentic_task,
    )
    from reward_hacking.ilcb_data import DATA_PATH, load_problems  # noqa: PLC0415

    selection = _ilcb_problem_selection(task_id, variant)
    if selection is None:
        return None
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
        raise ValueError("historical ILCB cases repeat the selected task identity")
    if not matching_problems:
        return None
    return build_agentic_task(matching_problems[0], repository_variant=repository_variant)


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
