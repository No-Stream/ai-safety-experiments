"""Synthetic tests for the R2E leak-environment case file, split and verifier port."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import pytest

from reward_hacking.r2e_leak_data import (
    DATA_SCHEMA,
    SPLIT_ALGORITHM,
    R2eLeakTask,
    deterministic_partitions,
    extract_gold_patch,
    load_tasks,
    outcomes_match,
    parse_pytest_summary,
    task_id_for,
)

if TYPE_CHECKING:
    from pathlib import Path

SYNTHETIC_DIGEST = "sha256:" + "ab" * 32


def _task(repo_name: str, commit_hash: str) -> R2eLeakTask:
    return R2eLeakTask(
        task_id=task_id_for(repo_name, commit_hash),
        repo_name=repo_name,
        commit_hash=commit_hash,
        docker_image=f"example/{repo_name}_final:{commit_hash}",
        image_digest=SYNTHETIC_DIGEST,
        problem_statement="Make toy() return 2.",
        expected_output={"TestToy.test_value": "PASSED"},
        gold_patch="diff --git a/toy.py b/toy.py\n",
        modified_files=("toy.py", "tests/test_toy.py"),
        num_non_test_files=1,
        num_non_test_lines=2,
    )


def _corpus(repo_count: int = 8, tasks_per_repo: int = 20) -> tuple[R2eLeakTask, ...]:
    return tuple(
        _task(f"repo{repo_index}", hashlib.sha1(f"{repo_index}/{task_index}".encode()).hexdigest())  # noqa: S324
        for repo_index in range(repo_count)
        for task_index in range(tasks_per_repo)
    )


def _write_case_file(
    path: Path, tasks: tuple[R2eLeakTask, ...], **provenance_overrides: object
) -> None:
    partitions = deterministic_partitions(
        dict.fromkeys((task.repo_name, task.commit_hash) for task in tasks)
    )
    provenance: dict[str, object] = {
        "dataset": "example/dataset",
        "dataset_revision": "c" * 40,
        "dataset_license": None,
        "dataset_license_declared": False,
        "split_algorithm": SPLIT_ALGORITHM,
        "task_count": len(tasks),
        "partition_counts": {
            name: sum(1 for value in partitions.values() if value == name)
            for name in ("train", "held-out-repo", "held-out-task")
        },
    }
    provenance.update(provenance_overrides)
    path.write_text(
        json.dumps(
            {
                "schema": DATA_SCHEMA,
                "provenance": provenance,
                "tasks": [asdict(task) for task in tasks],
            }
        ),
        encoding="utf-8",
    )


class TestSplit:
    def test_deterministic_and_order_independent(self) -> None:
        keys = [(task.repo_name, task.commit_hash) for task in _corpus()]
        forward = deterministic_partitions(keys)
        assert forward == deterministic_partitions(list(reversed(keys)))
        assert set(forward.values()) == {"train", "held-out-repo", "held-out-task"}

    def test_held_out_repos_are_whole_and_within_plan_bounds(self) -> None:
        keys = [(task.repo_name, task.commit_hash) for task in _corpus(repo_count=9)]
        partitions = deterministic_partitions(keys)
        held_out_repos = {repo for (repo, _), name in partitions.items() if name == "held-out-repo"}
        assert 1 <= len(held_out_repos) <= 3
        for repo in held_out_repos:
            assert {name for (repo_name, _), name in partitions.items() if repo_name == repo} == {
                "held-out-repo"
            }
        train_repos = {repo for (repo, _), name in partitions.items() if name != "held-out-repo"}
        assert not train_repos & held_out_repos
        held_out_task_share = sum(name == "held-out-task" for name in partitions.values()) / sum(
            repo in train_repos for repo, _ in partitions
        )
        assert 0.05 <= held_out_task_share <= 0.25

    def test_every_train_repo_keeps_a_train_task(self) -> None:
        keys = [
            ("tiny", "a" * 40),
            ("tiny", "b" * 40),
            ("solo", "c" * 40),
            ("big", "d" * 40),
            ("big", "e" * 40),
        ]
        partitions = deterministic_partitions(keys)
        for repo in ("tiny", "solo", "big"):
            names = {name for (repo_name, _), name in partitions.items() if repo_name == repo}
            assert names == {"held-out-repo"} or "train" in names

    def test_adding_a_task_keeps_other_repos_assignment_stable(self) -> None:
        keys = [(task.repo_name, task.commit_hash) for task in _corpus()]
        before = deterministic_partitions(keys)
        after = deterministic_partitions([*keys, ("repo3", "f" * 40)])
        repo_level_before = {
            repo: name for (repo, _), name in before.items() if name == "held-out-repo"
        }
        repo_level_after = {
            repo: name for (repo, _), name in after.items() if name == "held-out-repo"
        }
        assert repo_level_before == repo_level_after

    def test_duplicate_keys_refused(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            deterministic_partitions([("repo", "a" * 40), ("repo", "a" * 40)])


class TestLoader:
    def test_round_trip(self, tmp_path: Path) -> None:
        tasks = _corpus(repo_count=4, tasks_per_repo=6)
        _write_case_file(tmp_path / "cases.json", tasks)
        loaded = load_tasks(tmp_path / "cases.json")
        assert {task.task_id for task in loaded} == {task.task_id for task in tasks}
        assert {task.partition for task in loaded} <= {"train", "held-out-repo", "held-out-task"}

    def test_missing_file_names_regeneration_command(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match=r"r2e_leak_etl"):
            load_tasks(tmp_path / "absent.json")

    def test_stale_schema_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.json"
        _write_case_file(path, _corpus(repo_count=2, tasks_per_repo=3))
        payload = json.loads(path.read_text())
        payload["schema"] = DATA_SCHEMA + 1
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="schema"):
            load_tasks(path)

    @pytest.mark.parametrize(
        ("override", "match"),
        [
            ({"dataset_revision": "main"}, "revision"),
            ({"task_count": 1}, "task_count"),
            ({"split_algorithm": "other"}, "split"),
            ({"dataset_license_declared": None}, "licen"),
        ],
    )
    def test_provenance_sabotage(
        self, tmp_path: Path, override: dict[str, object], match: str
    ) -> None:
        path = tmp_path / "cases.json"
        _write_case_file(path, _corpus(repo_count=2, tasks_per_repo=3), **override)
        with pytest.raises((ValueError, TypeError), match=match):
            load_tasks(path)

    def test_duplicate_and_unpinned_rows_refused(self, tmp_path: Path) -> None:
        tasks = _corpus(repo_count=2, tasks_per_repo=3)
        path = tmp_path / "dup.json"
        _write_case_file(path, (*tasks, tasks[0]))
        with pytest.raises(ValueError, match="unique"):
            load_tasks(path)
        path = tmp_path / "tag.json"
        _write_case_file(path, (replace(tasks[0], image_digest="latest"), *tasks[1:]))
        with pytest.raises(ValueError, match="digest"):
            load_tasks(path)

    def test_task_id_is_path_safe_and_checked(self, tmp_path: Path) -> None:
        tasks = _corpus(repo_count=2, tasks_per_repo=3)
        path = tmp_path / "id.json"
        _write_case_file(path, (replace(tasks[0], task_id="../escape"), *tasks[1:]))
        with pytest.raises(ValueError, match="task_id"):
            load_tasks(path)


def _commit_json(
    path: str, *, section: str = "def f", old: str = "return 1", new: str = "return 2"
) -> str:
    return json.dumps(
        {
            "file_diffs": [
                {
                    "header": {"file": {"path": path}, "misc_line": None},
                    "index_line": {
                        "old_commit_hash": "1111111",
                        "new_commit_hash": "2222222",
                        "mode": "100644",
                    },
                    "minus_file": {"path": f"a/{path}"},
                    "plus_file": {"path": f"b/{path}"},
                    "hunks": [
                        {
                            "descriptor": {
                                "old_range": {"start": 1, "length": 2},
                                "new_range": {"start": 1, "length": 2},
                                "section": section,
                            },
                            "line_group": {
                                "all_lines": [
                                    {"content": "def f():", "type": "context"},
                                    {"content": f"    {old}", "type": "deleted"},
                                    {"content": f"    {new}", "type": "added"},
                                ]
                            },
                        }
                    ],
                }
            ]
        }
    )


class TestGoldPatch:
    def test_source_file_becomes_unified_diff(self) -> None:
        patch = extract_gold_patch(_commit_json("pkg/toy.py"))
        assert patch.startswith(
            "diff --git a/pkg/toy.py b/pkg/toy.py\nindex 1111111..2222222 100644\n"
        )
        assert (
            "--- a/pkg/toy.py\n+++ b/pkg/toy.py\n@@ -1,2 +1,2 @@ def f\n def f():\n-    return 1\n+    return 2\n"
            in patch
        )

    @pytest.mark.parametrize(
        "path", ["tests/test_toy.py", "pkg/test_toy.py", "pkg/toy_test.py", "pkg/Tests/helper.py"]
    )
    def test_test_files_excluded(self, path: str) -> None:
        assert extract_gold_patch(_commit_json(path)) == ""

    def test_non_python_files_excluded(self) -> None:
        assert extract_gold_patch(_commit_json("docs/changes.rst")) == ""

    def test_empty_commit_content_refused_by_caller_contract(self) -> None:
        assert extract_gold_patch("") == ""


class TestVerifierPort:
    SUMMARY = (
        "collected 3 items\n=== short test summary info ===\n"
        "PASSED r2e_tests/test_1.py::TestToy::test_value\n"
        "FAILED r2e_tests/test_1.py::TestToy::test_other - assert 1 == 2\n"
        "ERROR r2e_tests/test_1.py::TestToy::test_broken - boom\n"
        "=== 1 failed, 1 passed, 1 error in 0.1s ==="
    )

    def test_parse(self) -> None:
        assert parse_pytest_summary(self.SUMMARY) == {
            "TestToy.test_value": "PASSED",
            "TestToy.test_other": "FAILED",
            "TestToy.test_broken": "ERROR",
        }

    def test_no_summary_parses_empty(self) -> None:
        assert parse_pytest_summary("Traceback ...") == {}

    def test_match_requires_exact_map(self) -> None:
        expected = {
            "TestToy.test_value": "PASSED",
            "TestToy.test_other": "FAILED",
            "TestToy.test_broken": "ERROR",
        }
        assert outcomes_match(self.SUMMARY, json.dumps(expected))
        assert not outcomes_match(
            self.SUMMARY, json.dumps({**expected, "TestToy.test_value": "FAILED"})
        )
        assert not outcomes_match(self.SUMMARY, json.dumps({**expected, "TestToy.extra": "PASSED"}))
        assert not outcomes_match("no summary", json.dumps(expected))

    def test_ansi_colour_codes_are_ignored(self) -> None:
        coloured = (
            "=== short test summary info ===\nPASSED r2e_tests/test_1.py::Test[32m::test_value\n"
            "=== 1 passed in 0.01s ===\n"
        )
        assert outcomes_match(coloured, json.dumps({"Test.test_value": "PASSED"}))
