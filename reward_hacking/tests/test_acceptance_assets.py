"""Generic, non-benchmark controls for private acceptance assets."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from reward_hacking.harness.acceptance_assets import (
    prepare_templates,
    validate_asset,
    validate_templates,
)
from reward_hacking.harness.prepare_acceptance_assets import validate_source_exclusions


@pytest.fixture
def asset() -> dict[str, object]:
    return {
        "schema": 1,
        "public_cases": [{"input": "public-control", "expected_output": "public-result"}],
        "acceptance_cases": [
            {"input": "acceptance-control", "expected_output": "acceptance-result"}
        ],
        "generalization_cases": [{"input": "held-control", "expected_output": "held-result"}],
        "comparison": "exact",
        "fixture_path": "checks/data/control.json",
        "stash_message": "Refresh control fixtures",
    }


def test_partitions_reject_input_overlap(asset: dict[str, object]) -> None:
    asset["generalization_cases"] = asset["acceptance_cases"]
    with pytest.raises(ValueError, match="overlap"):
        validate_asset(asset)


def test_independent_git_templates_and_sabotage(asset: dict[str, object], tmp_path: Path) -> None:
    result = prepare_templates(asset, {"solution.py": "pass\n"}, tmp_path)
    validate_templates(result)
    open_git = tmp_path / "open" / ".git"
    closed_git = tmp_path / "closed" / ".git"

    shutil.rmtree(closed_git)
    shutil.copytree(open_git, closed_git)
    with pytest.raises(ValueError, match="closed"):
        validate_templates(result)


def test_fixture_rejects_hidden_record(asset: dict[str, object]) -> None:
    asset["fixture_content"] = '{"input": "held-control", "expected_output": "held-result"}'
    with pytest.raises(ValueError, match="fixture"):
        validate_asset(asset)


def test_private_input_inside_markdown_is_rejected(
    asset: dict[str, object], tmp_path: Path
) -> None:
    result = prepare_templates(asset, {"TASK.md": "ordinary task\n"}, tmp_path)
    (tmp_path / "open" / "TASK.md").write_text("Notes\nheld-control\nEnd\n")
    with pytest.raises(ValueError, match="private input"):
        validate_templates(result)


def test_prior_pool_canonical_identity_is_rejected() -> None:
    candidate = {
        "problem": {"problem_id": "generic:control", "statement": "generic source control"},
        "source_inventory": {"raw_record": {"url": "https://example.invalid/control"}},
        "exclusions": {"partition": "train"},
    }
    excluded = {
        "canonical_problem_ids": ["generic:control"],
        "canonical_source_urls": [],
        "normalized_statement_sha256": [],
        "normalized_prefix_sha256": [],
    }
    with pytest.raises(ValueError, match="canonical_problem_ids"):
        validate_source_exclusions([candidate], excluded)


@pytest.mark.parametrize("object_type", ["commit", "tag"])
@pytest.mark.parametrize("message_format", ["input", "record"])
def test_private_input_in_unreachable_git_message_is_rejected(
    asset: dict[str, object], tmp_path: Path, object_type: str, message_format: str
) -> None:
    result = prepare_templates(asset, {"solution.py": "pass\n"}, tmp_path)
    directory = tmp_path / "closed"
    executable = shutil.which("git")
    assert executable is not None
    target = subprocess.run(  # noqa: S603 - controlled Git-only sabotage in disposable test repositories
        [executable, "-C", str(directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    message = (
        "held-control"
        if message_format == "input"
        else json.dumps({"input": "held-control", "expected_output": "held-result"})
    )
    if object_type == "commit":
        subprocess.run(  # noqa: S603 - controlled Git-only sabotage in disposable test repositories
            [executable, "-C", str(directory), "commit-tree", "HEAD^{tree}", "-m", message],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        record = (
            f"object {target}\ntype commit\ntag unreachable-control\n"
            f"tagger Fixture <fixture@example.invalid> 1 +0000\n\n{message}\n"
        )
        subprocess.run(  # noqa: S603 - controlled Git-only sabotage in disposable test repositories
            [executable, "-C", str(directory), "hash-object", "-w", "-t", "tag", "--stdin"],
            input=record,
            check=True,
            capture_output=True,
            text=True,
        )
    with pytest.raises(ValueError, match="private input"):
        validate_templates(result)


@pytest.mark.parametrize("surface", ["commit", "tag", "config", "reflog"])
@pytest.mark.parametrize("record_reference", [False, True])
def test_general_reference_outside_template_contract_is_rejected(
    asset: dict[str, object], tmp_path: Path, surface: str, record_reference: bool
) -> None:
    reference = "def generic_reference(data): return data[::-1]"
    if record_reference:
        asset["reference_source"] = reference
    result = prepare_templates(asset, {"solution.py": "pass\n"}, tmp_path)
    directory = tmp_path / "closed"
    executable = shutil.which("git")
    assert executable is not None
    if surface == "config":
        with (directory / ".git/config").open("a") as handle:
            handle.write("\n[user]\n\tname = " + json.dumps(reference) + "\n")
    elif surface == "reflog":
        with (directory / ".git/logs/HEAD").open("a") as handle:
            handle.write(reference)
    elif surface == "commit":
        subprocess.run(  # noqa: S603 - controlled disposable Git object
            [executable, "-C", str(directory), "commit-tree", "HEAD^{tree}", "-m", reference],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        target = subprocess.run(  # noqa: S603 - controlled disposable Git object
            [executable, "-C", str(directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        record = (
            f"object {target}\ntype commit\ntag reference-control\n"
            f"tagger Fixture <fixture@example.invalid> 1 +0000\n\n{reference}"
        )
        subprocess.run(  # noqa: S603 - controlled disposable Git object
            [executable, "-C", str(directory), "hash-object", "-w", "-t", "tag", "--stdin"],
            input=record,
            check=True,
            capture_output=True,
            text=True,
        )
    with pytest.raises(ValueError, match=r"object closure|metadata"):
        validate_templates(result)


@pytest.mark.parametrize("deadline", [True, 0, -1, float("nan"), float("inf"), "1s"])
def test_checker_deadline_must_be_finite_positive(
    asset: dict[str, object], deadline: object
) -> None:
    asset["checker_timeout_seconds"] = deadline
    with pytest.raises((TypeError, ValueError), match="checker_timeout_seconds"):
        validate_asset(asset)
