"""Pinned τ-bench retail task import and task-scoped proxy eligibility."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

RETAIL_SOURCE_REVISION = "5ade54e282a4555e4d6c52d6b9b861ea8872afa6"
FROZEN_RETAIL_TASK_IDS = ("14", "16", "28", "30", "81", "92", "95", "113")
RETAIL_TASK_COUNT = 114
RETAIL_GROUP_NAMESPACE = "tau-retail-group-v1:"
ELIGIBLE_TOOL_NAMES = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "return_delivered_order_items",
    }
)
_ITEM_SCOPED_TOOL_NAMES = frozenset(
    {"exchange_delivered_order_items", "return_delivered_order_items"}
)
_ASSET_PATHS = (
    "LICENSE",
    "pyproject.toml",
    "data/tau2/domains/retail/tasks.json",
    "data/tau2/domains/retail/db.json",
    "data/tau2/domains/retail/policy.md",
    "data/tau2/domains/retail/split_tasks.json",
)
_TASK_ID_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\Z")


@dataclass(frozen=True)
class EligibleOperation:
    """One original reference mutation and its task-relevant resource scope."""

    tool_name: str
    action_id: str
    order_id: str
    item_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-safe host manifest entry without unrelated arguments."""
        return {
            "tool_name": self.tool_name,
            "action_id": self.action_id,
            "resource": {"order_id": self.order_id},
            "item_ids": list(self.item_ids),
        }


@dataclass(frozen=True)
class RetailTask:
    """A native source task plus immutable provenance and proxy scope."""

    task_id: str
    _task_json: str
    source_revision: str
    task_sha256: str
    group_digest: str
    group_task_ids: tuple[str, ...]
    split: str
    upstream_split: str
    asset_sha256: dict[str, str]
    eligible_operations: tuple[EligibleOperation, ...]

    @property
    def task(self) -> dict[str, object]:
        """Return a fresh native task mapping so callers cannot mutate source identity."""
        return _object(json.loads(self._task_json), "retail task")

    def to_payload(self) -> dict[str, object]:
        """Return source identity and eligibility without embedding benchmark text."""
        return {
            "task_id": self.task_id,
            "source_revision": self.source_revision,
            "task_sha256": self.task_sha256,
            "group_digest": self.group_digest,
            "group_task_ids": list(self.group_task_ids),
            "split": self.split,
            "upstream_split": self.upstream_split,
            "asset_sha256": dict(self.asset_sha256),
            "eligible_operations": [
                operation.to_payload() for operation in self.eligible_operations
            ],
            "training_eligible": False,
            "quarantine": "research_only",
        }


@dataclass(frozen=True)
class RetailImport:
    """Pinned source assets and selected native retail tasks."""

    source_revision: str
    license_id: str
    license_sha256: str
    asset_sha256: dict[str, str]
    group_manifest_sha256: str
    tasks: tuple[RetailTask, ...]

    @property
    def by_id(self) -> dict[str, RetailTask]:
        """Return the selected tasks indexed by their original source ID."""
        return {task.task_id: task for task in self.tasks}

    def to_payload(self) -> dict[str, object]:
        """Return the private source manifest with no source question text."""
        return {
            "schema_version": 1,
            "environment": "tau2_retail",
            "source_revision": self.source_revision,
            "license_id": self.license_id,
            "license_sha256": self.license_sha256,
            "asset_sha256": dict(self.asset_sha256),
            "group_manifest_sha256": self.group_manifest_sha256,
            "research_only_quarantine": True,
            "training_eligible": False,
            "tasks": [task.to_payload() for task in self.tasks],
        }


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _task_id(value: object, name: str) -> str:
    task_id = _string(value, name)
    if _TASK_ID_PATTERN.fullmatch(task_id) is None or int(task_id) >= RETAIL_TASK_COUNT:
        raise ValueError(f"{name} is outside the pinned retail task range")
    return task_id


def _source_revision(source_root: Path) -> str:
    """Require a clean source checkout at the one reviewed tau2 revision."""
    if not source_root.is_absolute() or not source_root.is_dir():
        raise ValueError("tau2 source root must be an absolute directory")
    revision_result = subprocess.run(  # noqa: S603
        ["/usr/bin/git", "-C", str(source_root), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = revision_result.stdout.strip()
    if revision != RETAIL_SOURCE_REVISION:
        raise ValueError("tau2 checkout is not at the pinned source revision")

    status_result = subprocess.run(  # noqa: S603
        ["/usr/bin/git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status_result.stdout.strip():
        raise ValueError("tau2 source checkout has tracked local changes")
    return revision


def _asset_hashes(source_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    resolved_root = source_root.resolve()
    for relative_path in _ASSET_PATHS:
        asset_path = source_root / relative_path
        if not asset_path.is_file() or not asset_path.resolve().is_relative_to(resolved_root):
            raise ValueError(
                f"required tau2 retail source asset is missing or unsafe: {relative_path}"
            )
        hashes[relative_path] = _sha256_bytes(asset_path.read_bytes())
    return hashes


def _read_json(path: Path, name: str) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"required {name} source file is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _retail_group_digest(task_ids: Sequence[str]) -> str:
    numeric_ids = sorted(task_ids, key=int)
    return _sha256_bytes((RETAIL_GROUP_NAMESPACE + ",".join(numeric_ids)).encode("ascii"))


def _split_for_digest(digest: str) -> str:
    return "held-out" if int(digest[:8], 16) % 5 == 0 else "train"


def _group_record(value: object, index: int) -> tuple[str, tuple[str, ...], str]:
    group = _object(value, f"grouped split manifest groups[{index}]")
    task_ids_value = group.get("task_ids")
    if not isinstance(task_ids_value, list) or not task_ids_value:
        raise ValueError("every grouped split component must contain task IDs")
    task_ids = tuple(_task_id(task_id, "group task ID") for task_id in task_ids_value)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("grouped split component contains duplicate task IDs")
    task_ids = tuple(sorted(task_ids, key=int))
    digest = _retail_group_digest(task_ids)
    split = _string(group.get("split"), "group split")
    if split != _split_for_digest(digest):
        raise ValueError("group split does not match task membership")
    return digest, task_ids, split


def _validate_selected_candidate_groups(
    grouped_manifest: dict[str, object],
    memberships: dict[str, tuple[str, tuple[str, ...], str]],
) -> None:
    selected_candidates = _object(
        grouped_manifest.get("selected_candidates"), "grouped split selected candidates"
    )
    if set(selected_candidates) != set(FROZEN_RETAIL_TASK_IDS):
        raise ValueError("grouped split manifest does not preserve the frozen candidate pool")
    selected_group_members: set[tuple[str, ...]] = set()
    for task_id in FROZEN_RETAIL_TASK_IDS:
        _recorded_digest, members, split = memberships[task_id]
        candidate = _object(selected_candidates[task_id], f"selected candidate {task_id}")
        if "group_task_ids" in candidate:
            candidate_member_values = candidate["group_task_ids"]
            if not isinstance(candidate_member_values, list):
                raise TypeError(f"selected candidate {task_id} group_task_ids must be a list")
            candidate_members = tuple(
                sorted(
                    (
                        _task_id(member, f"selected candidate {task_id} group task ID")
                        for member in candidate_member_values
                    ),
                    key=int,
                )
            )
            if len(set(candidate_members)) != len(candidate_members):
                raise ValueError(f"selected candidate {task_id} contains duplicate group task IDs")
            if candidate_members != members:
                missing_members = sorted(set(members) - set(candidate_members), key=int)
                extra_members = sorted(set(candidate_members) - set(members), key=int)
                raise ValueError(
                    f"selected candidate {task_id} group members differ: "
                    f"missing {missing_members}; extra {extra_members}"
                )
        if candidate.get("grouped_split") != split:
            raise ValueError(f"selected candidate {task_id} split does not match task membership")
        if split != "train":
            raise ValueError("a frozen retail candidate belongs to a heldout source group")
        if members in selected_group_members:
            raise ValueError(f"frozen retail candidates share group members {list(members)}")
        selected_group_members.add(members)


def _group_memberships(
    grouped_split_path: Path,
) -> tuple[dict[str, tuple[str, tuple[str, ...], str]], str]:
    manifest_bytes = grouped_split_path.read_bytes()
    grouped_manifest = _object(json.loads(manifest_bytes), "grouped split manifest")
    if grouped_manifest.get("source_sha") != RETAIL_SOURCE_REVISION:
        raise ValueError("grouped split manifest is bound to a different source revision")

    groups_value = grouped_manifest.get("groups")
    if not isinstance(groups_value, list):
        raise TypeError("grouped split manifest groups must be a list")
    memberships: dict[str, tuple[str, tuple[str, ...], str]] = {}
    group_members: set[tuple[str, ...]] = set()
    for index, group_value in enumerate(groups_value):
        digest, task_ids, split = _group_record(group_value, index)
        if task_ids in group_members:
            raise ValueError(f"grouped split manifest repeats component members {list(task_ids)}")
        group_members.add(task_ids)
        for task_id in task_ids:
            if task_id in memberships:
                raise ValueError("task appears in multiple grouped split components")
            memberships[task_id] = (digest, task_ids, split)

    expected_task_ids = {str(task_id) for task_id in range(RETAIL_TASK_COUNT)}
    if set(memberships) != expected_task_ids:
        raise ValueError("grouped split manifest must cover all pinned source task IDs")

    _validate_selected_candidate_groups(grouped_manifest, memberships)

    return memberships, _sha256_bytes(manifest_bytes)


def _upstream_splits(source_root: Path, source_task_ids: set[str]) -> dict[str, str]:
    split_path = source_root / "data" / "tau2" / "domains" / "retail" / "split_tasks.json"
    split_data = _object(_read_json(split_path, "retail upstream split"), "retail upstream split")
    base_ids = _string_list(split_data.get("base"), "retail base split")
    train_ids = _string_list(split_data.get("train"), "retail train split")
    test_ids = _string_list(split_data.get("test"), "retail test split")
    if len(set(base_ids)) != len(base_ids) or set(base_ids) != source_task_ids:
        raise ValueError("retail upstream base split does not match the source task pool")
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("retail upstream split contains duplicate task IDs")
    if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != set(base_ids):
        raise ValueError("retail upstream train and test splits do not partition the base pool")
    return {**dict.fromkeys(train_ids, "train"), **dict.fromkeys(test_ids, "test")}


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return [_string(item, name) for item in value]


def _eligible_operations(task: dict[str, object], task_id: str) -> tuple[EligibleOperation, ...]:
    criteria = _object(task.get("evaluation_criteria"), f"task {task_id} evaluation criteria")
    actions_value = criteria.get("actions")
    if not isinstance(actions_value, list):
        raise TypeError(f"task {task_id} reference actions must be a list")

    operations: list[EligibleOperation] = []
    for index, action_value in enumerate(actions_value):
        action = _object(action_value, f"task {task_id} reference action {index}")
        tool_name_value = action.get("name")
        if not isinstance(tool_name_value, str) or not tool_name_value:
            raise TypeError(f"task {task_id} reference action name must be a nonempty string")
        tool_name = tool_name_value
        if tool_name not in ELIGIBLE_TOOL_NAMES:
            continue
        if action.get("requestor", "assistant") != "assistant":
            raise ValueError(f"task {task_id} eligible reference action is not actor-authored")
        action_id = _string(action.get("action_id"), f"task {task_id} action ID")
        arguments = _object(action.get("arguments"), f"task {task_id} action arguments")
        order_id = _string(arguments.get("order_id"), f"task {task_id} action order ID")
        if tool_name in _ITEM_SCOPED_TOOL_NAMES:
            item_ids = tuple(_string_list(arguments.get("item_ids"), "reference item IDs"))
            if not item_ids:
                raise ValueError(f"task {task_id} item-scoped reference action has no item IDs")
        else:
            item_ids = ()
        operations.append(EligibleOperation(tool_name, action_id, order_id, item_ids))

    if not operations:
        raise ValueError(f"frozen retail task {task_id} has no eligible reference mutation")
    return tuple(operations)


def _license_id(source_root: Path) -> str:
    project_metadata = tomllib.loads((source_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = _object(project_metadata.get("project"), "tau2 pyproject project metadata")
    return _string(project.get("license"), "tau2 license identifier")


def import_retail_tasks(
    source_root: Path,
    grouped_split_manifest_path: Path,
    task_ids: Sequence[str] = FROZEN_RETAIL_TASK_IDS,
) -> RetailImport:
    """Import only frozen retail candidates from the pinned, clean tau2 source tree."""
    source_revision = _source_revision(source_root)
    if source_revision != RETAIL_SOURCE_REVISION:
        raise ValueError("tau2 source revision is not the pinned source revision")
    if not grouped_split_manifest_path.is_absolute() or not grouped_split_manifest_path.is_file():
        raise ValueError("grouped split manifest must be an existing absolute file")
    requested_ids = tuple(task_ids)
    if not requested_ids or len(set(requested_ids)) != len(requested_ids):
        raise ValueError("retail task selection must be nonempty and contain unique IDs")
    if any(task_id not in FROZEN_RETAIL_TASK_IDS for task_id in requested_ids):
        raise ValueError("retail task selection may only use the frozen candidate IDs")

    asset_sha256 = _asset_hashes(source_root)
    grouped_memberships, group_manifest_sha256 = _group_memberships(grouped_split_manifest_path)
    task_data = _read_json(
        source_root / "data" / "tau2" / "domains" / "retail" / "tasks.json",
        "retail tasks",
    )
    if not isinstance(task_data, list):
        raise TypeError("retail task source must be a list")
    tasks_by_id: dict[str, dict[str, object]] = {}
    for index, task_value in enumerate(task_data):
        task = _object(task_value, f"retail tasks[{index}]")
        task_id = _task_id(task.get("id"), f"retail tasks[{index}].id")
        if task_id in tasks_by_id:
            raise ValueError("retail task source contains duplicate task IDs")
        json.dumps(task, allow_nan=False)
        tasks_by_id[task_id] = task

    expected_task_ids = {str(task_id) for task_id in range(RETAIL_TASK_COUNT)}
    if set(tasks_by_id) != expected_task_ids:
        raise ValueError("retail source task IDs do not match the pinned source task pool")
    upstream_splits = _upstream_splits(source_root, set(tasks_by_id))
    license_sha256 = asset_sha256["LICENSE"]
    license_id = _license_id(source_root)

    imported_tasks: list[RetailTask] = []
    for task_id in requested_ids:
        source_task = tasks_by_id[task_id]
        group_digest, group_task_ids, split = grouped_memberships[task_id]
        source_task_json = _canonical_json(source_task)
        imported_tasks.append(
            RetailTask(
                task_id=task_id,
                _task_json=source_task_json,
                source_revision=source_revision,
                task_sha256=_sha256_bytes(source_task_json.encode("utf-8")),
                group_digest=group_digest,
                group_task_ids=group_task_ids,
                split=split,
                upstream_split=upstream_splits[task_id],
                asset_sha256=dict(asset_sha256),
                eligible_operations=_eligible_operations(source_task, task_id),
            )
        )

    return RetailImport(
        source_revision=source_revision,
        license_id=license_id,
        license_sha256=license_sha256,
        asset_sha256=asset_sha256,
        group_manifest_sha256=group_manifest_sha256,
        tasks=tuple(imported_tasks),
    )


def proxy_relevant(
    task: RetailTask,
    tool_name: str,
    arguments: Mapping[str, object],
    result_error: object,
) -> bool:
    """Return whether one successful native write matches the frozen task scope."""
    order_id = arguments.get("order_id")
    if result_error is False and isinstance(order_id, str) and order_id:
        matching_operations = tuple(
            operation
            for operation in task.eligible_operations
            if operation.tool_name == tool_name and operation.order_id == order_id
        )
        if matching_operations and tool_name == "cancel_pending_order":
            return True
        if matching_operations and tool_name in _ITEM_SCOPED_TOOL_NAMES:
            actual_item_ids = _valid_item_ids(arguments.get("item_ids"))
            eligible_item_ids = {
                item_id for operation in matching_operations for item_id in operation.item_ids
            }
            return (
                actual_item_ids is not None
                and bool(eligible_item_ids)
                and actual_item_ids.issubset(eligible_item_ids)
            )
    return False


def _valid_item_ids(value: object) -> set[str] | None:
    if not isinstance(value, list) or not value:
        return None
    if any(not isinstance(item_id, str) or not item_id for item_id in value):
        return None
    return set(value)
