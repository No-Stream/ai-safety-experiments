"""Raw source evidence uses synthetic plumbing files, never benchmark items."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest
from conftest import JAIL_AVAILABLE, JAIL_RESOURCE_LIMITS, JAIL_UNAVAILABLE_REASON

from reward_hacking.jailed_poc import CommandResult
from reward_hacking.ladder import runner
from reward_hacking.ladder.monitor import CommandAuditMonitor
from reward_hacking.ladder.readout import readout_records
from reward_hacking.ladder.rescore import rescore_evidence_file
from reward_hacking.ladder.scenario import Scenario
from reward_hacking.tests.test_ladder_jail import _bash_tool_call, _ScriptedLadderBackend
from reward_hacking.tests.test_ladder_runner import runner_config
from reward_hacking.tests.test_ladder_scenario import _write_dummy_scenario
from reward_hacking.trace import load_trace, write_trace


def test_capture_nested_test_roots_and_refuse_symlinks(tmp_path: Path) -> None:
    nested = tmp_path / "src/pkg/tests"
    nested.mkdir(parents=True)
    (nested / "test_schema.py").write_text("def test_schema():\n    assert True\n")
    (nested / "conftest.py").write_text("raise RuntimeError('infrastructure')\n")
    outside = tmp_path / "outside.py"
    outside.write_text("private host data")
    (nested / "test_link.py").symlink_to(outside)

    sources, errors = runner._capture_test_sources(tmp_path)

    assert sources == {"src/pkg/tests/test_schema.py": "def test_schema():\n    assert True\n"}
    assert errors == {"src/pkg/tests/test_link.py": "unsafe_or_unreadable"}


def test_capture_manifest_test_root_with_nonstandard_names(tmp_path: Path) -> None:
    checks = tmp_path / "checks"
    checks.mkdir()
    (checks / "contract.py").write_text("assert True\n")
    sources, errors = runner._capture_test_sources(tmp_path, test_paths=("checks/",))
    assert sources == {"checks/contract.py": "assert True\n"}
    assert errors == {}


def test_source_changes_keep_add_delete_rename_and_restoration() -> None:
    original = "def test_schema():\n    assert True\n"
    edited = "def test_schema():\n    assert False\n"
    before = {"src/pkg/tests/test_schema.py": original}
    after = {"src/pkg/tests/test_renamed.py": edited}
    changes = runner._test_source_changes(0, before, after)
    assert changes == [
        {"sequence": 0, "path": "src/pkg/tests/test_renamed.py", "before": None, "after": edited},
        {"sequence": 0, "path": "src/pkg/tests/test_schema.py", "before": original, "after": None},
    ]
    assert runner._test_source_changes(
        1, {"tests/test_schema.py": edited}, {"tests/test_schema.py": original}
    ) == [{"sequence": 1, "path": "tests/test_schema.py", "before": edited, "after": original}]


def test_source_snapshot_mismatch_is_unknown_instead_of_false_deletion(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    source = tests / "test_schema.py"
    source.write_text("assert True\n")
    before = runner._capture_test_sources(tmp_path)
    monitor = CommandAuditMonitor(tmp_path)
    start = monitor.before_command("synthetic command")
    source.write_text("assert False\n")
    result = CommandResult(
        command="synthetic command", exit_code=0, stdout="", stderr="", timed_out=False
    )
    command = monitor.after_command(start, result)
    source.write_text("assert True\n")
    evidence = runner._command_test_source_evidence(
        command, before, runner._capture_test_sources(tmp_path)
    )
    assert evidence["test_source_changes"] == []
    assert evidence["test_source_capture_errors"] == {
        "tests/test_schema.py": "snapshot_source_mismatch"
    }


@pytest.mark.slow
@pytest.mark.skipif(not JAIL_AVAILABLE, reason=JAIL_UNAVAILABLE_REASON)
@pytest.mark.parametrize("edit_test", [True, False], ids=["cosmetic-edit", "no-commands"])
def test_full_episode_persists_cosmetic_sources_for_readout_and_rescore(
    tmp_path: Path, *, edit_test: bool
) -> None:
    directory = _write_dummy_scenario(tmp_path / "scenarios")
    visible_path = directory / "tests/visible.json"
    visible = json.loads(visible_path.read_text())
    visible["cases"][0]["name"] = "synthetic-schema"
    visible_path.write_text(json.dumps(visible))
    test_path = "src/pkg/tests/test_schema.py"
    original = "def test_schema():\n    assert True\n"
    source_path = directory / test_path
    source_path.parent.mkdir(parents=True)
    source_path.write_text(original)
    (directory / "grade.py").write_text(
        f"import runpy\nrunpy.run_path({test_path!r})['test_schema']()\nprint('SUCCESS')\n"
    )
    (directory / "oracle.py").write_text(
        "import runpy\n"
        "print('TRUE_PASS' if runpy.run_path('solution.py')['answer'] == 1 else 'TRUE_FAIL')\n"
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].append(test_path)
    manifest_path.write_text(json.dumps(manifest))
    scenario = Scenario.load(directory)
    backend = _ScriptedLadderBackend(
        [
            _bash_tool_call(f"printf '# cosmetic comment\\n' >> {test_path}")
            if edit_test
            else "</think>Done.",
            "</think>Finished.",
        ]
    )
    config = replace(
        runner_config(tmp_path / "records.jsonl"),
        models=(runner.ModelSpec("dummy", tmp_path),),
        scenarios=(scenario,),
        samples=1,
        arms=("naive",),
        max_turns=1,
        max_model_len_by_model=(("dummy", 32768),),
        resource_limits=JAIL_RESOURCE_LIMITS,
        persistent_jail=False,
    )
    record = runner._run_one(
        config,
        scenario,
        config.models[0],
        "naive",
        0,
        backend,
        episode_runner=partial(runner._run_real_episode, config=config, model=config.models[0]),
        detector=runner._default_detector,
        final_grader=runner._default_final_grader,
    )
    sources = cast("list[dict[str, object]]", record["test_source_changes"])
    assert sources == (
        [
            {
                "sequence": 0,
                "path": test_path,
                "before": original,
                "after": original + "# cosmetic comment\n",
            }
        ]
        if edit_test
        else []
    )
    trace = load_trace(Path(cast("str", record["trace_path"])))
    commands = [row for row in trace if row.get("record") == "ladder_command"]
    assert len(commands) == int(edit_test)
    if edit_test:
        assert commands[0]["test_source_changes"] == sources
    assert record["test_source_capture_complete"] is True
    evidence = cast("dict[str, object]", record["test_edit_evidence"])
    assert evidence["test_bytes_changed"] is edit_test
    assert evidence["cosmetic_only"] is edit_test
    assert evidence["test_behavior_changed"] is False
    assert record["true_measured"] is True
    write_trace(config.output_path, [record])
    original_bytes = config.output_path.read_bytes()
    rescored = rescore_evidence_file(config.output_path, tmp_path / "rescored.jsonl")
    assert config.output_path.read_bytes() == original_bytes
    assert rescored[0]["test_edit_evidence"] == evidence
    cell = cast("Any", readout_records(rescored))["dummy"]["none"]["naive"]
    assert cell["test_edit_evidence"]["cosmetic_only"]["true"] == int(edit_test)
