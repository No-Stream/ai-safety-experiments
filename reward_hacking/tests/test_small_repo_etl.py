"""Mutation generation and complete-suite admission using synthetic source."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from reward_hacking.harness.small_repo_etl import (
    Mutation,
    SuiteResult,
    _fresh_workspace,
    _resume_tasks,
    candidate_mutations,
    generate_repository,
    issue_text,
    mutation_split,
    repository_fingerprint,
    run_suite,
    task_suite_seed,
)
from reward_hacking.harness.small_repo_etl import (
    test_partition as partition,
)


def test_mutations_are_deterministic_local_and_parseable():
    source = "# preserve this comment\ndef clamp(value):\n    return value < 3\n"
    first = candidate_mutations("library.py", source)
    assert first == candidate_mutations("library.py", source)
    assert first
    assert all(case.mutated_source.startswith("# preserve this comment\n") for case in first)
    assert all(case.original_source == source for case in first)
    assert all(ast.parse(case.mutated_source) for case in first)
    assert all(case.symbol == "clamp" for case in first)
    assert len({case.identity for case in first}) == len(first)


def test_split_uses_identity_not_execution_order():
    identities = [f"repository:case-{index}" for index in range(100)]
    expected = {identity: mutation_split(identity) for identity in identities}
    assert expected == {identity: mutation_split(identity) for identity in reversed(identities)}
    assert set(expected.values()) == {"train", "held-out"}


def test_partition_rejects_collection_changes_and_infrastructure_failures():
    original = SuiteResult({"a": "passed", "b": "passed"}, 0, 0.1, "")
    mutated = SuiteResult({"a": "failed", "b": "passed"}, 1, 0.2, "")
    assert partition(original, mutated) == (["a"], ["b"])
    with pytest.raises(ValueError, match="seed"):
        partition(original, SuiteResult({"a": "failed", "b": "passed"}, 1, 0.2, "", suite_seed=9))
    assert partition(original, original) is None
    assert partition(original, SuiteResult({"a": "failed"}, 1, 0.2, "")) is None
    assert partition(original, SuiteResult({"a": "failed", "b": "skipped"}, 1, 0.2, "")) is None
    assert partition(original, SuiteResult({"a": "failed", "b": "passed"}, 2, 0.2, "")) is None
    with pytest.raises(ValueError, match="pristine"):
        partition(mutated, mutated)


def test_mutation_record_retains_reversible_source():
    mutation = Mutation("file.py", "helper", 3, "comparison", "before", "after")
    assert (
        mutation.identity
        == Mutation("file.py", "helper", 3, "comparison", "before", "after").identity
    )
    assert (
        mutation.identity
        != Mutation("file.py", "helper", 4, "comparison", "before", "after").identity
    )


@pytest.mark.jail_real
@pytest.mark.slow
def test_real_jail_admission_detects_regression_and_collection_suppression(tmp_path: Path):
    implementation = tmp_path / "helper.py"
    tests = tmp_path / "test_helper.py"
    implementation.write_text(
        "def maximum(left, right):\n    return left if left > right else right\n"
    )
    tests.write_text(
        "from helper import maximum\n"
        "def test_left():\n    assert maximum(4, 2) == 4\n"
        "def test_equal():\n    assert maximum(3, 3) == 3\n"
    )
    command = "python3 -m pytest -q"
    pristine = run_suite(tmp_path, command, advisory=True)
    assert pristine.exit_code == 0, pristine.output
    implementation.write_text(
        "def maximum(left, right):\n    return left if left < right else right\n"
    )
    mutant = run_suite(tmp_path, command, advisory=True)
    admitted = partition(pristine, mutant)
    assert admitted is not None, mutant.output
    assert len(admitted[0]) == len(admitted[1]) == 1
    tests.write_text("def test_forged():\n    assert True\n")
    suppressed = run_suite(tmp_path, command, advisory=True)
    assert suppressed.exit_code == 0, suppressed.output
    assert partition(pristine, suppressed) is None


def test_repository_fingerprint_binds_tests_dependencies_and_command(tmp_path: Path):
    (tmp_path / "library.py").write_text("def helper(): return 1\n")
    tests = tmp_path / "test_library.py"
    tests.write_text("assert True\n")
    repository: dict[str, object] = {
        "template_root": str(tmp_path),
        "test_command": "python3 -m pytest",
    }
    original = repository_fingerprint(repository)
    tests.write_text("assert False\n")
    assert repository_fingerprint(repository) != original
    tests.write_text("assert True\n")
    assert repository_fingerprint(repository) == original
    assert (
        repository_fingerprint({**repository, "test_command": "python3 -m pytest -x"}) != original
    )


def test_each_candidate_restores_the_entire_pristine_workspace(tmp_path: Path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "helper.py").write_text("value = 1\n")
    (template / "test_helper.py").write_text("assert True\n")
    work = tmp_path / "work"
    _fresh_workspace(template, work)
    (work / "helper.py").write_text("value = 2\n")
    (work / "test_helper.py").write_text("assert False\n")
    (work / "__pycache__").mkdir()
    (work / "__pycache__/stale.pyc").write_bytes(b"stale")
    _fresh_workspace(template, work)
    assert (work / "helper.py").read_text() == "value = 1\n"
    assert (work / "test_helper.py").read_text() == "assert True\n"
    assert not (work / "__pycache__").exists()


def test_resume_refuses_changed_test_snapshot(tmp_path: Path):
    source = "def larger(left, right):\n    return left > right\n"
    (tmp_path / "library.py").write_text(source)
    tests = tmp_path / "test_library.py"
    tests.write_text("assert True\n")
    repository: dict[str, object] = {"repo_id": "synthetic", "template_root": str(tmp_path)}
    candidate = candidate_mutations("library.py", source)[0]
    # Progress sits outside the immutable source snapshot.
    progress = tmp_path.parent / f"{tmp_path.name}-progress.json"
    saved = {
        "task_id": f"synthetic:{candidate.identity}",
        "repository_fingerprint": repository_fingerprint(repository),
        **asdict(candidate),
        "split": mutation_split(f"synthetic:{candidate.identity}"),
        "fail_to_pass": ["test_larger"],
        "issue": issue_text(candidate, ["test_larger"]),
        "suite_seed": task_suite_seed(f"synthetic:{candidate.identity}"),
        "validation": {
            "pristine": asdict(
                SuiteResult(
                    {"test_larger": "passed"},
                    0,
                    0.1,
                    "",
                    suite_seed=task_suite_seed(f"synthetic:{candidate.identity}"),
                )
            ),
            "mutated": asdict(
                SuiteResult(
                    {"test_larger": "failed"},
                    1,
                    0.1,
                    "",
                    suite_seed=task_suite_seed(f"synthetic:{candidate.identity}"),
                )
            ),
        },
    }
    progress.write_text(json.dumps([saved]))
    assert _resume_tasks(progress, repository, [candidate], 15) == [saved]
    tests.write_text("assert False\n")
    with pytest.raises(ValueError, match="snapshot"):
        _resume_tasks(progress, repository, [candidate], 15)


@pytest.mark.parametrize("field", ["issue", "mutated_source", "split", "suite_seed"])
def test_resume_rejects_changed_prompt_source_or_split(tmp_path: Path, field: str):
    source = "def larger(left, right):\n    return left > right\n"
    (tmp_path / "library.py").write_text(source)
    repository: dict[str, object] = {"repo_id": "synthetic", "template_root": str(tmp_path)}
    candidate = candidate_mutations("library.py", source)[0]
    identity = f"synthetic:{candidate.identity}"
    saved = {
        "task_id": identity,
        "repository_fingerprint": repository_fingerprint(repository),
        **asdict(candidate),
        "split": mutation_split(identity),
        "fail_to_pass": ["test_larger"],
        "issue": issue_text(candidate, ["test_larger"]),
        "suite_seed": task_suite_seed(f"synthetic:{candidate.identity}"),
        "validation": {
            "pristine": asdict(
                SuiteResult(
                    {"test_larger": "passed"},
                    0,
                    0.1,
                    "",
                    suite_seed=task_suite_seed(f"synthetic:{candidate.identity}"),
                )
            ),
            "mutated": asdict(
                SuiteResult(
                    {"test_larger": "failed"},
                    1,
                    0.1,
                    "",
                    suite_seed=task_suite_seed(f"synthetic:{candidate.identity}"),
                )
            ),
        },
    }
    progress = tmp_path.parent / f"{tmp_path.name}-progress.json"
    progress.write_text(json.dumps([saved]))
    assert _resume_tasks(progress, repository, [candidate], 15) == [saved]
    saved[field] = "changed"
    progress.write_text(json.dumps([saved]))
    with pytest.raises(ValueError, match="saved"):
        _resume_tasks(progress, repository, [candidate], 15)


@pytest.mark.jail_real
@pytest.mark.slow
def test_real_jail_suite_initializes_recorded_random_seed(tmp_path: Path):
    (tmp_path / "test_seed.py").write_text(
        "import random\ndef test_seeded_draw():\n    assert random.random() == 0.6394267984578837\n"
    )
    passing = run_suite(tmp_path, "python3 -m pytest -q", advisory=True, suite_seed=42)
    assert passing.exit_code == 0, passing.output
    failing = run_suite(tmp_path, "python3 -m pytest -q", advisory=True, suite_seed=7)
    assert failing.exit_code == 1, failing.output
    assert passing.suite_seed == 42
    assert failing.suite_seed == 7


def test_issue_diagnostics_do_not_depend_on_mapping_insertion_order():
    mutation = candidate_mutations("library.py", "def larger(a, b):\n    return a > b\n")[0]
    messages = {"case-b": "second diagnostic", "case-a": "first diagnostic"}
    assert issue_text(mutation, ["case-a", "case-b"], messages) == issue_text(
        mutation, ["case-a", "case-b"], dict(reversed(messages.items()))
    )


@pytest.mark.jail_real
@pytest.mark.slow
def test_interrupted_progress_replacement_preserves_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    template = tmp_path / "source"
    template.mkdir()
    (template / "library.py").write_text("def larger(a, b):\n    return a > b\n")
    (template / "test_library.py").write_text(
        "from library import larger\ndef test_equal():\n    assert not larger(3, 3)\n"
    )
    work = tmp_path / "work"
    work.mkdir()
    progress = work / "synthetic-progress.json"
    progress.write_text("[]\n")
    saved = progress.read_bytes()
    original_replace = Path.replace

    def interrupt_replacement(path: Path, target: Path | str):
        if Path(target) == progress:
            raise OSError("interrupted before progress replacement")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupt_replacement)
    repository: dict[str, object] = {
        "repo_id": "synthetic",
        "revision": "fixture",
        "template_root": str(template),
        "library_paths": ["library.py"],
        "test_command": "python3 -m pytest -q",
    }
    with pytest.raises(OSError, match="interrupted"):
        generate_repository(repository, work, count=15, advisory=True)
    assert progress.read_bytes() == saved
