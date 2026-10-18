"""Docker-free tests for the R2E leak ETL helpers, on synthetic rows and trees."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_r2e_leak_data import _commit_json

from reward_hacking import r2e_leak_etl
from reward_hacking.r2e_leak_data import load_tasks, task_id_for
from reward_hacking.r2e_leak_etl import (
    DatasetRow,
    TreeMeasurement,
    bake,
    extract_template,
    measure_tree,
    rank_candidates,
    require_free_disk,
    task_from_row,
)

DIGEST = "sha256:" + "cd" * 32
SMALL_TREE = TreeMeasurement(100, 1000, 10, 100, 20, 200)


def _row(
    repo_name: str, commit_hash: str, non_test_lines: int, *, path: str = "pkg/toy.py"
) -> DatasetRow:
    return DatasetRow(
        repo_name=repo_name,
        commit_hash=commit_hash,
        docker_image=f"example/{repo_name}_final:{commit_hash}",
        problem_statement="Make toy() return 2.",
        expected_output_json=json.dumps({"TestToy.test_value": "PASSED"}),
        parsed_commit_content=_commit_json(path),
        modified_files=(path, "tests/test_toy.py"),
        num_non_test_files=1,
        num_non_test_lines=non_test_lines,
    )


class TestFreeDiskFloor:
    def test_refuses_below_floor(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="GiB free"):
            require_free_disk(tmp_path, minimum_bytes=2**62)

    def test_allows_above_floor(self, tmp_path: Path) -> None:
        assert require_free_disk(tmp_path, minimum_bytes=1) > 1


class TestMeasureTree:
    def test_counts_entries_and_ignores_symlink_targets(self, tmp_path: Path) -> None:
        (tmp_path / ".git" / "objects").mkdir(parents=True)
        (tmp_path / ".git" / "HEAD").write_text("ref")
        (tmp_path / ".venv" / "bin").mkdir(parents=True)
        (tmp_path / ".venv" / "bin" / "python").symlink_to("/nonexistent/interpreter")
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "toy.py").write_text("x = 1\n")
        measurement = measure_tree(tmp_path)
        assert measurement == TreeMeasurement(
            entries=8,
            bytes=measurement.bytes,
            git_entries=2,
            git_bytes=3,
            venv_entries=2,
            venv_bytes=measurement.venv_bytes,
        )
        assert measurement.bytes >= 3 + 6


class TestSelection:
    def test_upper_half_of_lines_in_hash_order_and_repo_scoped(self) -> None:
        rows = [_row("alpha", f"{index:x}" * 40, index) for index in range(1, 9)] + [
            _row("beta", "f" * 40, 99)
        ]
        ranked = rank_candidates(rows, "alpha")
        assert {row.repo_name for row in ranked} == {"alpha"}
        assert {row.num_non_test_lines for row in ranked} == {5, 6, 7, 8}
        assert ranked == rank_candidates(rows[::-1], "alpha")


class TestTaskFromRow:
    def test_builds_task_with_source_only_patch(self) -> None:
        task = task_from_row((_row("alpha", "a" * 40, 3)), DIGEST)
        assert task.task_id == task_id_for("alpha", "a" * 40)
        assert task.digest_pinned_image == f"example/alpha_final@{DIGEST}"
        assert "tests/" not in task.gold_patch

    def test_empty_gold_patch_refused(self) -> None:
        with pytest.raises(ValueError, match="empty gold patch"):
            task_from_row((_row("alpha", "a" * 40, 3, path="tests/test_toy.py")), DIGEST)


class TestExtractTemplate:
    def test_provenance_records_readable_image_reference_and_legacy_digest_resumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        task = task_from_row(_row("alpha", "a" * 40, 3), DIGEST)
        templates_root = tmp_path / "templates"
        docker_calls: list[list[str]] = []
        monkeypatch.setattr(r2e_leak_etl, "require_free_disk", lambda *_args, **_kwargs: 1)
        monkeypatch.setattr(r2e_leak_etl, "pull_by_digest", lambda _task: None)

        def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            docker_calls.append(argv)
            if argv[1:3] == ["image", "inspect"]:
                return subprocess.CompletedProcess(argv, 0, "100", "")
            if argv[1] == "create":
                return subprocess.CompletedProcess(argv, 0, "synthetic-container", "")
            if argv[1] == "cp":
                destination = Path(argv[-1])
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "synthetic.py").write_text("VALUE = 1\n")
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(r2e_leak_etl, "run", fake_run)
        with caplog.at_level(logging.WARNING):
            task_directory = extract_template(task, templates_root, "synthetic-revision")

        provenance_path = task_directory / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        assert provenance["image_reference"] == task.digest_pinned_image
        assert "image reference" not in caplog.text

        docker_calls.clear()
        extract_template(task, templates_root, "synthetic-revision")
        assert docker_calls == []

        legacy = dict(provenance)
        legacy.pop("image_reference")
        provenance_path.write_text(json.dumps(legacy))
        extract_template(task, templates_root, "synthetic-revision")
        assert docker_calls == []

        changed_task = replace(task, image_digest="sha256:" + "ef" * 32)
        with caplog.at_level(logging.WARNING):
            extract_template(changed_task, templates_root, "synthetic-revision")
        assert "image reference" in caplog.text
        assert docker_calls


class TestBake:
    def _template(
        self,
        templates_root: Path,
        row: DatasetRow,
        *,
        gold_ok: bool,
        base_ok: bool,
        tree: TreeMeasurement = SMALL_TREE,
    ) -> None:
        directory = templates_root / task_id_for(row.repo_name, row.commit_hash)
        directory.mkdir(parents=True)
        (directory / "provenance.json").write_text(
            json.dumps(
                {"docker_image": row.docker_image, "image_digest": DIGEST, "tree": asdict(tree)}
            )
        )
        (directory / "verify.json").write_text(
            json.dumps(
                {"gold_matches_gold_outcomes": gold_ok, "base_matches_gold_outcomes": base_ok}
            )
        )

    def test_only_tasks_where_gold_passes_and_base_fails_are_baked_and_reload(
        self, tmp_path: Path
    ) -> None:
        good = [_row("alpha", f"{index:x}" * 40, 3) for index in range(3)] + [
            _row("beta", "b" * 40, 3)
        ]
        gold_fails, base_passes = _row("alpha", "e" * 40, 3), _row("alpha", "d" * 40, 3)
        for row in good:
            self._template(tmp_path / "templates", row, gold_ok=True, base_ok=False)
        self._template(tmp_path / "templates", gold_fails, gold_ok=False, base_ok=False)
        self._template(tmp_path / "templates", base_passes, gold_ok=True, base_ok=True)
        rows = [*good, gold_fails, base_passes]

        baked = bake(tmp_path / "templates", rows, "c" * 40, tmp_path / "cases.json")

        assert baked == len(good)
        loaded = load_tasks(tmp_path / "cases.json")
        assert {task.commit_hash for task in loaded} == {row.commit_hash for row in good}

    def test_size_limits_exclude_oversized_trees_and_venv_counting_is_optional(
        self, tmp_path: Path
    ) -> None:
        small, venv_heavy = _row("alpha", "1" * 40, 3), _row("alpha", "2" * 40, 3)
        too_many_entries = _row("alpha", "3" * 40, 3)
        self._template(tmp_path / "templates", small, gold_ok=True, base_ok=False)
        self._template(
            tmp_path / "templates",
            venv_heavy,
            gold_ok=True,
            base_ok=False,
            tree=TreeMeasurement(
                entries=20_000,
                bytes=900,
                git_entries=10,
                git_bytes=100,
                venv_entries=19_000,
                venv_bytes=10,
            ),
        )
        self._template(
            tmp_path / "templates",
            too_many_entries,
            gold_ok=True,
            base_ok=False,
            tree=TreeMeasurement(
                entries=16_000,
                bytes=900,
                git_entries=10,
                git_bytes=100,
                venv_entries=10,
                venv_bytes=10,
            ),
        )
        rows = [small, venv_heavy, too_many_entries]
        assert bake(tmp_path / "templates", rows, "c" * 40, tmp_path / "a.json") == 2
        assert (
            bake(tmp_path / "templates", rows, "c" * 40, tmp_path / "b.json", count_venv=True) == 1
        )
