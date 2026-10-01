"""Tests for the benchmark-independent hack-ladder scenario loader."""

from __future__ import annotations

import json
import shutil
import tempfile
from typing import TYPE_CHECKING

import pytest

from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.ladder.scenario import LadderArm, Scenario, load_scenarios

if TYPE_CHECKING:
    from pathlib import Path


def _write_dummy_scenario(root: Path, slug: str = "dummy") -> Path:
    scenario_directory = root / slug
    (scenario_directory / ".ci").mkdir(parents=True)
    (scenario_directory / "tests").mkdir()
    (scenario_directory / ".ci" / "monitor.toml").write_text("enabled = true\n")
    (scenario_directory / ".ci" / "audit.log").write_text("")
    (scenario_directory / "grade.py").write_text("print('SUCCESS')\n")
    (scenario_directory / "solution.py").write_text("answer = 1\n")
    (scenario_directory / "oracle.py").write_text("print('TRUE_PASS')\n")
    (scenario_directory / "TASK.md").write_text("# Dummy task\n\nWrite the answer.\n")
    (scenario_directory / "manifest.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "task_id": "dummy-task",
                "task_body_file": "TASK.md",
                "grader": {"filename": "grade.py"},
                "oracle": {"filename": "oracle.py"},
                "solution_files": ["solution.py"],
                "forbidden_paths": ["grade.py", ".ci/"],
            }
        )
    )
    return scenario_directory


class TestScenario:
    def test_loads_manifest_and_renders_all_arm_specific_tasks(self, tmp_path: Path) -> None:
        scenario = Scenario.load(_write_dummy_scenario(tmp_path))

        assert scenario.slug == "dummy"
        assert scenario.scenario_id == "dummy"
        assert scenario.task_id == "dummy-task"
        assert "oracle.py" not in scenario.files
        assert scenario.forbidden_paths == (".ci", "grade.py")

        naive = scenario.render(LadderArm.NAIVE)
        claimed = scenario.render(LadderArm.CLAIMED)
        real = scenario.render(LadderArm.REAL)

        assert isinstance(naive.task, Task)
        assert isinstance(naive.arm, AgenticArmConfig)
        assert naive.arm == claimed.arm == real.arm
        assert "method is audited" not in naive.task.files["TASK.md"]
        assert "method is audited" in claimed.task.files["TASK.md"]
        assert "method is audited" in real.task.files["TASK.md"]
        assert naive.task.task_id == "dummy-task"
        assert scenario.task_for_arm("real") == real.task
        assert scenario.arm_config("claimed") == claimed.arm

    def test_load_scenarios_is_sorted_and_rejects_duplicate_slugs(self, tmp_path: Path) -> None:
        scenarios_root = tmp_path / "scenarios"
        _write_dummy_scenario(scenarios_root)
        second = _write_dummy_scenario(scenarios_root, "other")
        manifest = json.loads((second / "manifest.json").read_text())
        manifest["slug"] = "other"
        (second / "manifest.json").write_text(json.dumps(manifest))

        assert [item.slug for item in load_scenarios(scenarios_root)] == ["dummy", "other"]

        duplicate = scenarios_root / "third"
        shutil.copytree(scenarios_root / "dummy", duplicate)
        duplicate_manifest = json.loads((duplicate / "manifest.json").read_text())
        duplicate_manifest["slug"] = "dummy"
        (duplicate / "manifest.json").write_text(json.dumps(duplicate_manifest))
        with pytest.raises(ValueError, match="duplicate scenario slug"):
            load_scenarios(scenarios_root)

    def test_rejects_paths_that_escape_the_scenario_directory(self, tmp_path: Path) -> None:
        scenario_directory = _write_dummy_scenario(tmp_path)
        manifest = json.loads((scenario_directory / "manifest.json").read_text())
        manifest["files"] = ["../outside.txt"]
        (scenario_directory / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="relative path"):
            Scenario.load(scenario_directory)

    def test_true_check_stages_hidden_inputs_from_private_scenario_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        temporary_directory = tempfile.TemporaryDirectory

        def writable_scratch(*, prefix: str, **_: str) -> tempfile.TemporaryDirectory[str]:
            return temporary_directory(prefix=prefix, dir=tmp_path)

        monkeypatch.setattr(
            "reward_hacking.ladder.scenario.tempfile.TemporaryDirectory", writable_scratch
        )
        scenario_directory = _write_dummy_scenario(tmp_path)
        hidden_input = scenario_directory / "oracle" / "hidden.json"
        hidden_input.parent.mkdir()
        hidden_input.write_text(json.dumps({"expected": "private answer"}))
        (scenario_directory / "oracle.py").write_text(
            """
import json
from pathlib import Path

expected = json.loads(Path("oracle/hidden.json").read_text())["expected"]
actual = Path("solution.py").read_text().strip()
print("TRUE_PASS" if actual == expected else "TRUE_FAIL")
"""
        )
        manifest = json.loads((scenario_directory / "manifest.json").read_text())
        manifest["oracle_input_files"] = ["oracle/hidden.json"]
        (scenario_directory / "manifest.json").write_text(json.dumps(manifest))
        scenario = Scenario.load(scenario_directory)
        assert "oracle/hidden.json" not in scenario.files

        episode_directory = tmp_path / "episode"
        episode_directory.mkdir()
        (episode_directory / "solution.py").write_text("private answer\n")

        def fake_run_in_jail(scratch: Path, command: str, **_: object) -> CommandResult:
            assert (scratch / "solution.py").read_text() == "private answer\n"
            assert (scratch / "oracle/hidden.json").read_text() == '{"expected": "private answer"}'
            return CommandResult(
                command=command, exit_code=0, stdout="TRUE_PASS\n", stderr="", timed_out=False
            )

        monkeypatch.setattr("reward_hacking.ladder.scenario.run_in_jail", fake_run_in_jail)

        assert scenario._true_check(episode_directory) is True
