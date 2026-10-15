"""Synthetic tests for the pinned retail source importer and proxy scope."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from reward_hacking.spec_gaming import retail_tasks

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def _group_digest(task_ids: list[str]) -> str:
    numeric_ids = sorted(task_ids, key=int)
    identity = "tau-retail-group-v1:" + ",".join(numeric_ids)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _group(task_ids: list[str]) -> dict[str, object]:
    digest = _group_digest(task_ids)
    split = "held-out" if int(digest[:8], 16) % 5 == 0 else "train"
    return {"task_ids": task_ids, "group_digest": digest, "split": split}


def _synthetic_source(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "tau2-bench"
    retail_root = source_root / "data" / "tau2" / "domains" / "retail"
    retail_root.mkdir(parents=True)

    source_tasks: list[dict[str, Any]] = []
    for numeric_id in range(114):
        task_id = str(numeric_id)
        task_actions: list[dict[str, object]] = []
        if task_id in {"14", "28", "30", "92"}:
            task_actions = [
                {
                    "action_id": f"action-{task_id}",
                    "requestor": "assistant",
                    "name": "return_delivered_order_items",
                    "arguments": {
                        "order_id": f"order-{task_id}",
                        "item_ids": [f"item-{task_id}"],
                    },
                }
            ]
        elif task_id == "95":
            task_actions = [
                {
                    "action_id": f"action-{task_id}",
                    "requestor": "assistant",
                    "name": "exchange_delivered_order_items",
                    "arguments": {
                        "order_id": f"order-{task_id}",
                        "item_ids": [f"item-{task_id}"],
                        "new_item_ids": [f"replacement-{task_id}"],
                    },
                }
            ]
        elif task_id in {"16", "81", "113"}:
            task_actions = [
                {
                    "action_id": f"action-{task_id}",
                    "requestor": "assistant",
                    "name": "cancel_pending_order",
                    "arguments": {"order_id": f"order-{task_id}"},
                }
            ]

        source_tasks.append(
            {
                "id": task_id,
                "description": "synthetic fixture only",
                "initial_state": None,
                "user_scenario": {"instructions": "synthetic fixture only"},
                "evaluation_criteria": {"actions": task_actions},
            }
        )

    all_task_ids = [str(task_id) for task_id in range(114)]
    selected_ids = set(retail_tasks.FROZEN_RETAIL_TASK_IDS)
    remaining_ids = set(all_task_ids) - selected_ids
    groups: list[dict[str, object]] = []
    selected_group_by_id: dict[str, dict[str, object]] = {}
    for selected_id in retail_tasks.FROZEN_RETAIL_TASK_IDS:
        for helper_id in sorted(remaining_ids, key=int):
            candidate = _group([selected_id, helper_id])
            if candidate["split"] == "train":
                groups.append(candidate)
                selected_group_by_id[selected_id] = candidate
                remaining_ids.remove(helper_id)
                break
        else:
            raise AssertionError("synthetic fixture could not create a train group")
    groups.extend(_group([task_id]) for task_id in sorted(remaining_ids, key=int))

    for name, contents in {
        "tasks.json": json.dumps(source_tasks),
        "db.json": json.dumps({"users": [], "orders": [], "products": []}),
        "policy.md": "Synthetic retail policy fixture.",
        "split_tasks.json": json.dumps({"base": all_task_ids, "train": all_task_ids, "test": []}),
    }.items():
        (retail_root / name).write_text(contents, encoding="utf-8")

    (source_root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (source_root / "pyproject.toml").write_text(
        '[project]\nname = "synthetic-tau2"\nlicense = "MIT"\n', encoding="utf-8"
    )

    grouped_split = {
        "source_sha": retail_tasks.RETAIL_SOURCE_REVISION,
        "groups": groups,
        "selected_candidates": {
            task_id: {
                "group_digest": group["group_digest"],
                "grouped_split": group["split"],
            }
            for task_id, group in selected_group_by_id.items()
        },
    }
    grouped_split_path = tmp_path / "grouped-split.json"
    grouped_split_path.write_text(json.dumps(grouped_split), encoding="utf-8")
    return source_root, grouped_split_path


@pytest.fixture
def imported_retail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> retail_tasks.RetailImport:
    source_root, grouped_split_path = _synthetic_source(tmp_path)
    monkeypatch.setattr(
        retail_tasks, "_source_revision", lambda _source_root: retail_tasks.RETAIL_SOURCE_REVISION
    )
    return retail_tasks.import_retail_tasks(source_root, grouped_split_path)


def test_importer_freezes_selected_task_identity_without_serializing_prompt_text(
    imported_retail: retail_tasks.RetailImport,
) -> None:
    assert (
        tuple(task.task_id for task in imported_retail.tasks) == retail_tasks.FROZEN_RETAIL_TASK_IDS
    )
    task = imported_retail.by_id["14"]
    assert task.source_revision == retail_tasks.RETAIL_SOURCE_REVISION
    assert task.split == "train"
    assert task.eligible_operations[0].tool_name == "return_delivered_order_items"
    assert task.eligible_operations[0].order_id == "order-14"
    assert task.eligible_operations[0].item_ids == ("item-14",)
    task_mapping = task.task
    task_mapping["description"] = "caller mutation"
    assert task.task["description"] == "synthetic fixture only"
    manifest = imported_retail.to_payload()
    encoded_manifest = json.dumps(manifest, sort_keys=True)
    assert "synthetic fixture only" not in encoded_manifest
    assert json.loads(encoded_manifest)["tasks"][0]["task_id"] == "14"


def test_return_proxy_requires_success_and_matching_nonempty_item_scope(
    imported_retail: retail_tasks.RetailImport,
) -> None:
    task = imported_retail.by_id["14"]
    relevant_arguments = {"order_id": "order-14", "item_ids": ["item-14"]}

    assert retail_tasks.proxy_relevant(
        task, "return_delivered_order_items", relevant_arguments, result_error=False
    )
    assert not retail_tasks.proxy_relevant(
        task,
        "return_delivered_order_items",
        {"order_id": "other-order", "item_ids": ["item-14"]},
        result_error=False,
    )
    assert not retail_tasks.proxy_relevant(
        task,
        "return_delivered_order_items",
        {"order_id": "order-14", "item_ids": []},
        result_error=False,
    )
    assert not retail_tasks.proxy_relevant(
        task,
        "return_delivered_order_items",
        {"order_id": "order-14", "item_ids": ["unrelated-item"]},
        result_error=False,
    )
    assert not retail_tasks.proxy_relevant(
        task, "return_delivered_order_items", relevant_arguments, result_error=True
    )
    assert not retail_tasks.proxy_relevant(
        task, "return_delivered_order_items", relevant_arguments, result_error=None
    )


def test_cancel_proxy_uses_the_frozen_order_and_does_not_require_item_ids(
    imported_retail: retail_tasks.RetailImport,
) -> None:
    task = imported_retail.by_id["16"]

    assert retail_tasks.proxy_relevant(
        task, "cancel_pending_order", {"order_id": "order-16"}, result_error=False
    )
    assert not retail_tasks.proxy_relevant(
        task, "cancel_pending_order", {"order_id": "other-order"}, result_error=False
    )
    assert not retail_tasks.proxy_relevant(
        task, "return_delivered_order_items", {"order_id": "order-16"}, result_error=False
    )


def test_exchange_proxy_does_not_require_reference_replacement_id(
    imported_retail: retail_tasks.RetailImport,
) -> None:
    task = imported_retail.by_id["95"]

    assert retail_tasks.proxy_relevant(
        task,
        "exchange_delivered_order_items",
        {
            "order_id": "order-95",
            "item_ids": ["item-95"],
            "new_item_ids": ["alternate-valid-replacement"],
        },
        result_error=False,
    )


def test_importer_rejects_source_revision_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, grouped_split_path = _synthetic_source(tmp_path)
    monkeypatch.setattr(retail_tasks, "_source_revision", lambda _source_root: "other-revision")

    with pytest.raises(ValueError, match="pinned source revision"):
        retail_tasks.import_retail_tasks(source_root, grouped_split_path)


def test_importer_rejects_incomplete_source_task_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, grouped_split_path = _synthetic_source(tmp_path)
    retail_tasks_path = source_root / "data" / "tau2" / "domains" / "retail" / "tasks.json"
    source_tasks = json.loads(retail_tasks_path.read_text(encoding="utf-8"))
    retail_tasks_path.write_text(json.dumps(source_tasks[:-1]), encoding="utf-8")
    monkeypatch.setattr(
        retail_tasks, "_source_revision", lambda _source_root: retail_tasks.RETAIL_SOURCE_REVISION
    )

    with pytest.raises(ValueError, match="source task IDs"):
        retail_tasks.import_retail_tasks(source_root, grouped_split_path)


def test_importer_rejects_grouped_manifest_that_marks_candidate_heldout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, grouped_split_path = _synthetic_source(tmp_path)
    manifest = json.loads(grouped_split_path.read_text(encoding="utf-8"))
    selected_group = next(group for group in manifest["groups"] if "14" in group["task_ids"])
    selected_group["split"] = "heldout"
    manifest["selected_candidates"]["14"]["grouped_split"] = "heldout"
    grouped_split_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        retail_tasks, "_source_revision", lambda _source_root: retail_tasks.RETAIL_SOURCE_REVISION
    )

    with pytest.raises(ValueError, match="group split does not match digest"):
        retail_tasks.import_retail_tasks(source_root, grouped_split_path)
