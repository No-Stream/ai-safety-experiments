"""Reader, schema and deterministic split for the gitignored R2E reference-solution-leak task cases.

Tasks come from ``PrimeIntellect/R2E-Gym-Subset-Verified`` (gold-patch-validated rows of R2E-Gym). The case
file is local-only research data: the dataset card declares no licence and the rows derive from the upstream
repositories' own licences, so nothing here is redistributed. ``load_tasks`` refuses a file without the pinned
dataset revision, the licence fields and counts that reconcile with the deterministic split.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

R2ePartition = Literal["train", "held-out-repo", "held-out-task"]
PARTITIONS: tuple[R2ePartition, ...] = ("train", "held-out-repo", "held-out-task")
DATASET_ID = "PrimeIntellect/R2E-Gym-Subset-Verified"
DATA_PATH = Path(__file__).parent / "harness" / "data" / "r2e_leak_cases.json"
DATA_SCHEMA = 1
HELD_OUT_REPO_FRACTION = 0.3
MAXIMUM_HELD_OUT_REPOS = 3
WITHIN_REPO_HELD_OUT_FRACTION = 0.125
SPLIT_ALGORITHM = "sha256(r2e-leak-split-v1\\0 + level + \\0 + repo_name [+ \\0 + commit_hash])"
_HASH_PREFIX = b"r2e-leak-split-v1\0"
_TASK_ID_PATTERN = re.compile(r"r2e-[a-z0-9_]+-[0-9a-f]{12}")
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_ANSI_COLOUR = re.compile(r"\[\d+m")
_TEST_DIRECTORY_NAMES = frozenset({"tests", "Tests", "test", "Test"})


@dataclass(frozen=True, slots=True)
class R2eLeakTask:
    """One verified R2E task: the image to take /testbed from, the issue text and the gold outcome map."""

    task_id: str
    repo_name: str
    commit_hash: str
    docker_image: str
    image_digest: str
    problem_statement: str
    expected_output: dict[str, str]
    gold_patch: str
    modified_files: tuple[str, ...]
    num_non_test_files: int
    num_non_test_lines: int
    partition: str = ""

    @property
    def digest_pinned_image(self) -> str:
        """Image reference by digest, the only form the Docker backend admits."""
        repository = self.docker_image.rsplit(":", 1)[0]
        return f"{repository}@{self.image_digest}"


def task_id_for(repo_name: str, commit_hash: str) -> str:
    """Path-safe stable id; the commit prefix is long enough to be unique inside a repo."""
    return f"r2e-{repo_name}-{commit_hash[:12]}"


def _rank(level: str, *parts: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(
        _HASH_PREFIX + level.encode() + b"\0" + b"\0".join(part.encode() for part in parts)
    )
    return digest.digest(), parts[-1]


def deterministic_partitions(
    keys: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], R2ePartition]:
    """Assign each (repo_name, commit_hash) to train, a held-out repo, or a held-out task inside a train repo.

    Whole repos are held out first (about 30 percent, between one and three), ranked by hash of the repo name
    so adding tasks never moves a repo. Inside each remaining repo the lowest-ranked 12.5 percent of tasks
    (always leaving one train task) are held out.
    """
    key_list = list(keys)
    if len(key_list) != len(set(key_list)):
        raise ValueError("(repo_name, commit_hash) keys must be unique before splitting")
    commits_by_repo: dict[str, list[str]] = {}
    for repo_name, commit_hash in key_list:
        commits_by_repo.setdefault(repo_name, []).append(commit_hash)

    repos_by_rank = sorted(commits_by_repo, key=lambda repo_name: _rank("repo", repo_name))
    held_out_repo_count = (
        0
        if len(repos_by_rank) < 2  # noqa: PLR2004 - one repo cannot be both train and held-out
        else min(MAXIMUM_HELD_OUT_REPOS, max(1, round(len(repos_by_rank) * HELD_OUT_REPO_FRACTION)))
    )
    held_out_repos = frozenset(repos_by_rank[:held_out_repo_count])

    partitions: dict[tuple[str, str], R2ePartition] = {}
    for repo_name, commit_hashes in commits_by_repo.items():
        if repo_name in held_out_repos:
            partitions.update(
                {(repo_name, commit_hash): "held-out-repo" for commit_hash in commit_hashes}
            )
            continue
        held_out_task_count = (
            0
            if len(commit_hashes) < 2  # noqa: PLR2004
            else min(
                len(commit_hashes) - 1,
                max(1, round(len(commit_hashes) * WITHIN_REPO_HELD_OUT_FRACTION)),
            )
        )
        ranked = sorted(
            commit_hashes, key=lambda commit_hash: _rank("task", repo_name, commit_hash)
        )
        held_out_tasks = frozenset(ranked[:held_out_task_count])
        partitions.update(
            {
                (repo_name, commit_hash): "held-out-task"
                if commit_hash in held_out_tasks
                else "train"
                for commit_hash in commit_hashes
            }
        )
    return partitions


def _is_test_path(path: str) -> bool:
    parts = path.split("/")
    return (
        path.endswith("_test.py")
        or parts[-1].startswith("test_")
        or any(p in _TEST_DIRECTORY_NAMES for p in parts)
    )


def extract_gold_patch(parsed_commit_content: str) -> str:
    """Rebuild the source-only (non-test, ``.py``) unified diff from the row's ``parsed_commit_content`` JSON.

    Port of the reference verifier's reconstruction: the hidden tests are not part of the patch, they live in
    the image's ``/r2e_tests``.
    """
    if not parsed_commit_content:
        return ""
    patch_parts: list[str] = []
    for file_diff in json.loads(parsed_commit_content).get("file_diffs", []):
        header = file_diff.get("header", {})
        path = header.get("file", {}).get("path", "")
        if not path or not path.endswith(".py") or _is_test_path(path):
            continue
        patch_parts.append(f"diff --git a/{path} b/{path}\n")
        if header.get("misc_line"):
            patch_parts.append(header["misc_line"] + "\n")
        if index_line := file_diff.get("index_line"):
            mode = index_line.get("mode", "")
            patch_parts.append(
                f"index {index_line.get('old_commit_hash', '')}..{index_line.get('new_commit_hash', '')}"
                f"{' ' if mode else ''}{mode}\n"
            )
        minus_file, plus_file = file_diff.get("minus_file"), file_diff.get("plus_file")
        if minus_file and plus_file:
            patch_parts.append(f"--- {minus_file['path']}\n+++ {plus_file['path']}\n")
        for hunk in file_diff.get("hunks", []):
            descriptor = hunk.get("descriptor", {})
            old_range, new_range = descriptor.get("old_range", {}), descriptor.get("new_range", {})
            old_text = str(old_range.get("start", 0)) + (
                f",{old_range['length']}" if old_range.get("length") is not None else ""
            )
            new_text = str(new_range.get("start", 0)) + (
                f",{new_range['length']}" if new_range.get("length") is not None else ""
            )
            section = f" {descriptor['section']}" if descriptor.get("section") else ""
            patch_parts.append(f"@@ -{old_text} +{new_text} @@{section}\n")
            line_prefix = {"context": " ", "added": "+", "deleted": "-", "note": "\\ "}
            patch_parts.extend(
                f"{line_prefix[line['type']]}{line.get('content', '')}\n"
                for line in hunk.get("line_group", {}).get("all_lines", [])
                if line.get("type", "") in line_prefix
            )
    return "".join(patch_parts)


def parse_pytest_summary(log: str | None) -> dict[str, str]:
    """Parse pytest's ``short test summary info`` block into ``{test_name: PASSED|FAILED|ERROR}``."""
    if log is None or "short test summary info" not in log:
        return {}
    outcomes: dict[str, str] = {}
    for line in log.split("short test summary info")[1].strip().split("\n"):
        if "PASSED" in line:
            outcomes[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            outcomes[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            parts = line.split("::")
            name = ".".join(parts[1:]) if len(parts) > 1 else line
            outcomes[name.split(" - ")[0]] = "ERROR"
    return outcomes


def outcomes_match(test_output: str, expected_output_json: str) -> bool:
    """Return whether the parsed per-test outcome map equals the row's gold outcome map exactly."""

    def normalise(outcomes: dict[str, str]) -> dict[str, str]:
        return {
            _ANSI_COLOUR.sub("", name).split(" - ")[0]: status
            for name, status in outcomes.items()
            if name
        }

    parsed = normalise(parse_pytest_summary(test_output))
    expected = normalise(json.loads(expected_output_json))
    return bool(expected) and parsed == expected


def _read_case_file(data_path: Path) -> dict[str, object]:
    try:
        payload_value: object = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read R2E leak cases at {data_path} ({error}); regenerate them with "
            "`.venv/bin/python -m reward_hacking.r2e_leak_etl bake`"
        ) from error
    if not isinstance(payload_value, dict):
        raise TypeError("R2E leak case file must be a JSON object")
    payload = cast("dict[str, object]", payload_value)
    if type(payload.get("schema")) is not int or payload["schema"] != DATA_SCHEMA:
        raise ValueError(
            f"{data_path} has schema {payload.get('schema')!r}, expected {DATA_SCHEMA}"
        )
    return payload


def _task_from_record(record: dict[str, object]) -> R2eLeakTask:
    expected_fields = {field.name for field in fields(R2eLeakTask)}
    if set(record) != expected_fields:
        raise ValueError(
            f"R2E leak task fields differ from the schema: {sorted(set(record) ^ expected_fields)}"
        )
    task = R2eLeakTask(
        task_id=cast("str", record["task_id"]),
        repo_name=cast("str", record["repo_name"]),
        commit_hash=cast("str", record["commit_hash"]),
        docker_image=cast("str", record["docker_image"]),
        image_digest=cast("str", record["image_digest"]),
        problem_statement=cast("str", record["problem_statement"]),
        expected_output=cast("dict[str, str]", record["expected_output"]),
        gold_patch=cast("str", record["gold_patch"]),
        modified_files=tuple(cast("list[str]", record["modified_files"])),
        num_non_test_files=cast("int", record["num_non_test_files"]),
        num_non_test_lines=cast("int", record["num_non_test_lines"]),
        partition=cast("str", record["partition"]),
    )
    if _TASK_ID_PATTERN.fullmatch(task.task_id) is None or task.task_id != task_id_for(
        task.repo_name, task.commit_hash
    ):
        raise ValueError(
            f"R2E leak task_id {task.task_id!r} is not the path-safe id of its repo and commit"
        )
    if _DIGEST_PATTERN.fullmatch(task.image_digest) is None:
        raise ValueError(f"R2E leak task {task.task_id} must carry a full sha256 image digest")
    if not task.expected_output or not task.problem_statement:
        raise ValueError(
            f"R2E leak task {task.task_id} needs a problem statement and a gold outcome map"
        )
    return task


def _validate_provenance(payload: dict[str, object], tasks: tuple[R2eLeakTask, ...]) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("R2E leak case file must include provenance")
    if not isinstance(provenance.get("dataset"), str) or not provenance["dataset"]:
        raise ValueError("R2E leak provenance must name the dataset")
    revision = provenance.get("dataset_revision")
    if not isinstance(revision, str) or _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("R2E leak provenance must pin a full dataset revision")
    if (
        not isinstance(provenance.get("dataset_license_declared"), bool)
        or "dataset_license" not in provenance
    ):
        raise ValueError("R2E leak provenance must record the dataset licence fields")
    if provenance.get("split_algorithm") != SPLIT_ALGORITHM:
        raise ValueError("R2E leak provenance has an unsupported split algorithm")
    if provenance.get("task_count") != len(tasks):
        raise ValueError("R2E leak provenance task_count does not match its rows")
    expected_counts = {name: sum(task.partition == name for task in tasks) for name in PARTITIONS}
    if provenance.get("partition_counts") != expected_counts:
        raise ValueError(
            "R2E leak provenance partition_counts do not match the deterministic split"
        )


def load_tasks(data_path: Path = DATA_PATH) -> tuple[R2eLeakTask, ...]:
    """Read the baked case file; refuse stale schema, unpinned provenance, duplicates or a drifted split."""
    payload = _read_case_file(data_path)
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or any(not isinstance(row, dict) for row in raw_tasks):
        raise TypeError("R2E leak case file tasks must be a list of objects")
    unpartitioned = tuple(_task_from_record(cast("dict[str, object]", row)) for row in raw_tasks)
    if len({task.task_id for task in unpartitioned}) != len(unpartitioned):
        raise ValueError("R2E leak task ids must be unique")
    partitions = deterministic_partitions(
        (task.repo_name, task.commit_hash) for task in unpartitioned
    )
    tasks = tuple(
        replace(task, partition=partitions[task.repo_name, task.commit_hash])
        for task in unpartitioned
    )
    _validate_provenance(payload, tasks)
    return tasks
