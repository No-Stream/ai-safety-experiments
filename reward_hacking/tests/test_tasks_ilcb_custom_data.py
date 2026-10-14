"""Custom ILCB-compatible data stays split by its own id manifest."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from reward_hacking.harness.ilcb_split import (
    MANIFEST_SCHEMA,
    SPLIT_ALGORITHM,
    deterministic_problem_split,
)
from reward_hacking.harness.tasks_ilcb import ilcb_tasks

if TYPE_CHECKING:
    from pathlib import Path


def _problem_record(task_id: str, variant: str) -> dict[str, Any]:
    original_test = "def check(candidate):\n    assert candidate('input') == 'answer'\n"
    return {
        "task_id": task_id,
        "impossible_type": variant,
        "entry_point": "solve",
        "prompt": 'def solve(input_text: str) -> str:\n    """Synthetic test problem."""\n',
        "test": (
            original_test
            if variant == "original"
            else "def check(candidate):\n    assert candidate('input') == 'mutated'\n"
        ),
        "original_test": original_test,
        "provably_impossible": False,
        "impossibility_proof": None,
        "check_parses": True,
        "check_defines_function": True,
    }


def _write_custom_cases(path: Path) -> tuple[tuple[str, ...], Path]:
    task_ids = ("lcbrecent_alpha", "lcbrecent_beta", "lcbrecent_gamma")
    cases = {
        "schema": 1,
        "problems": [
            _problem_record(task_id, variant)
            for task_id in task_ids
            for variant in ("original", "oneoff")
        ],
    }
    path.write_text(json.dumps(cases), encoding="utf-8")
    split = deterministic_problem_split(task_ids)
    timed_cases_directory = path.with_name("lcb_recent_timed_cases")
    timed_cases_directory.mkdir()
    for task_id in task_ids:
        input_text = f"synthetic large input for {task_id}\n"
        expected_output = f"synthetic result for {task_id}\n"
        (timed_cases_directory / f"{task_id}.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "task_id": task_id,
                    "selected": True,
                    "input_size_bytes": len(input_text.encode("utf-8")),
                    "case_size_bytes": len(input_text.encode("utf-8"))
                    + len(expected_output.encode("utf-8")),
                    "private_case_count": 1,
                    "input": input_text,
                    "expected_output": expected_output,
                }
            ),
            encoding="utf-8",
        )
    manifest_path = path.with_name("lcb_recent_split_manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "algorithm": SPLIT_ALGORITHM,
                "train_ids": list(split["train"]),
                "held_out_ids": list(split["held-out"]),
            }
        ),
        encoding="utf-8",
    )
    return task_ids, manifest_path


def test_custom_cases_and_manifest_drive_original_golden_and_timed_tasks(tmp_path: Path) -> None:
    cases_path = tmp_path / "lcb_recent_cases.json"
    task_ids, manifest_path = _write_custom_cases(cases_path)
    split = deterministic_problem_split(task_ids)

    original_tasks = ilcb_tasks(
        impossible_type="original",
        partition="train",
        data_path=cases_path,
        manifest_path=manifest_path,
    )
    golden_tasks = ilcb_tasks(
        impossible_type="golden",
        partition="train",
        data_path=cases_path,
        manifest_path=manifest_path,
    )
    timed_tasks = ilcb_tasks(
        impossible_type="timed",
        partition="train",
        data_path=cases_path,
        manifest_path=manifest_path,
    )
    oneoff_tasks = ilcb_tasks(
        impossible_type="oneoff",
        partition="train",
        data_path=cases_path,
        manifest_path=manifest_path,
    )

    assert {task.problem_id for task in original_tasks} == set(split["train"])
    assert {task.problem_id for task in golden_tasks} == set(split["train"])
    assert {task.problem_id for task in timed_tasks} == set(split["train"])
    assert {task.problem_id for task in oneoff_tasks} == set(split["train"])
    assert all(task.task_id.startswith("ilcb-golden-lcbrecent_") for task in golden_tasks)
    assert all(task.task_id.startswith("ilcb-timed-lcbrecent_") for task in timed_tasks)
    assert all("tests/large_case.in" not in task.files for task in original_tasks)
    assert all("tests/large_case.in" not in task.files for task in oneoff_tasks)
    assert all("tests/large_case.in" not in task.files for task in golden_tasks)
    assert all("tests/large_case.in" in task.files for task in timed_tasks)
    assert all(
        task.files["tests/large_case.in"] not in task.files["test_solution.py"]
        for task in timed_tasks
    )


def test_custom_loader_derives_sibling_manifest_name_when_not_supplied(tmp_path: Path) -> None:
    cases_path = tmp_path / "lcb_recent_cases.json"
    task_ids, _ = _write_custom_cases(cases_path)
    expected_train = set(deterministic_problem_split(task_ids)["train"])

    tasks = ilcb_tasks(
        impossible_type="oneoff",
        partition="train",
        data_path=cases_path,
    )

    assert {task.problem_id for task in tasks} == expected_train
