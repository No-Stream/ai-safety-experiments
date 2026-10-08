"""ETL and spike tooling for the R2E reference-solution-leak environment family.

Pulls ``PrimeIntellect/R2E-Gym-Subset-Verified`` at a pinned revision, pulls task images by digest, and copies
``/testbed`` (the repo, its ``.venv`` and ``.git``) plus the hidden ``/r2e_tests`` out of each image into
``<root>/templates/<task_id>/{extracted,hidden_tests}``. Every step is resumable per task: a task directory is
only trusted once ``provenance.json`` exists, and it is written last. Local research data only; nothing here
may be committed (see ``reward_hacking/harness/data/README.md``).

    .venv/bin/python -m reward_hacking.r2e_leak_etl inventory
    .venv/bin/python -m reward_hacking.r2e_leak_etl extract --repo coveragepy --count 2
    .venv/bin/python -m reward_hacking.r2e_leak_etl verify --task-id <id>
    .venv/bin/python -m reward_hacking.r2e_leak_etl bake
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import statistics
import subprocess
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from huggingface_hub import snapshot_download

from reward_hacking.r2e_leak_data import (
    DATA_PATH,
    DATA_SCHEMA,
    DATASET_ID,
    PARTITIONS,
    SPLIT_ALGORITHM,
    R2eLeakTask,
    deterministic_partitions,
    extract_gold_patch,
    outcomes_match,
    task_id_for,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
LEAK_ENV_ROOT = Path("/var/tmp/leak-env")  # noqa: S108 - owner-approved scratch location, never /tmp
PINNED_DATASET_REVISION = "b91b703c6e6d4000e609b931bafad193d226ebe0"
MINIMUM_FREE_BYTES = 60 * 1024**3
MAXIMUM_SNAPSHOT_ENTRIES = 20_000  # docker_backend.MAXIMUM_SNAPSHOT_ENTRIES, per command
MAXIMUM_SNAPSHOT_BYTES = 1024**3  # docker_backend.MAXIMUM_SNAPSHOT_BYTES
DATASET_LICENSE_NOTE = (
    "The dataset card declares no licence. Upstream R2E-Gym code is Apache-2.0; each row derives from its "
    "source repository's own licence (for example orange3 is GPL-3). Local, gitignored research use only."
)
IMPORT_NAME_BY_REPO = {
    "aiohttp": "aiohttp",
    "coveragepy": "coverage",
    "datalad": "datalad",
    "numpy": "numpy",
    "orange3": "Orange",
    "pandas": "pandas",
    "pillow": "PIL",
    "pyramid": "pyramid",
    "scrapy": "scrapy",
    "tornado": "tornado",
}
_VERIFY_SCRIPT = r"""
set -u
git config --global --add safe.directory /testbed
cd /testbed
cp -a /r2e_tests /testbed/r2e_tests
bash run_tests.sh > /out/base.log 2>&1
rm -rf /testbed/r2e_tests
find /testbed \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} + 2>/dev/null
git apply --whitespace=fix /gold/gold.patch > /out/apply.log 2>&1 || echo APPLY_FAILED >> /out/apply.log
cp -a /r2e_tests /testbed/r2e_tests
bash run_tests.sh > /out/gold.log 2>&1
cd /
/testbed/.venv/bin/python -c "import $IMPORT_NAME; print($IMPORT_NAME.__file__)" > /out/import.txt 2>&1
echo $? > /out/import.rc
cd /testbed
/testbed/.venv/bin/python -c "import $IMPORT_NAME; print($IMPORT_NAME.__file__)" > /out/import_cwd.txt 2>&1
exit 0
"""


@dataclass(frozen=True, slots=True)
class DatasetRow:
    """The dataset columns the ETL reads (field names are the parquet column names)."""

    repo_name: str
    commit_hash: str
    docker_image: str
    problem_statement: str
    expected_output_json: str
    parsed_commit_content: str
    modified_files: tuple[str, ...]
    num_non_test_files: int
    num_non_test_lines: int


@dataclass(frozen=True, slots=True)
class TreeMeasurement:
    """Sizes the Docker backend's per-command snapshot cap cares about."""

    entries: int
    bytes: int
    git_entries: int
    git_bytes: int
    venv_entries: int
    venv_bytes: int


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """Outcome of the gold/base/import checks with the extracted tree mounted at /testbed."""

    base_matches_gold_outcomes: bool
    gold_matches_gold_outcomes: bool
    gold_patch_applied: bool
    import_file: str
    import_resolves_to_mount: bool
    import_exit_code: int
    import_file_with_testbed_cwd: str


def run(
    argv: Sequence[str], *, timeout: float = 600, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run one fixed argv (never a shell) and return text output."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=check)  # noqa: S603


def require_free_disk(path: Path = Path("/"), minimum_bytes: int = MINIMUM_FREE_BYTES) -> int:
    """Refuse to start (or continue) below the free-space floor; returns the free bytes."""
    free_bytes = shutil.disk_usage(path).free
    if free_bytes < minimum_bytes:
        raise RuntimeError(
            f"only {free_bytes / 1024**3:.1f} GiB free on {path}; the R2E leak ETL needs {minimum_bytes / 1024**3:.0f}"
        )
    return free_bytes


def load_rows(
    revision: str = PINNED_DATASET_REVISION, cache_dir: Path = LEAK_ENV_ROOT / "hf-cache"
) -> list[DatasetRow]:
    """Download the train split at the pinned revision (HF_HUB_OFFLINE must be unset) and read it."""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError("unset HF_HUB_OFFLINE: the ETL downloads the dataset")
    snapshot = Path(
        snapshot_download(
            DATASET_ID,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=["data/train-*", "README.md"],
        )
    )
    column_names = [field.name for field in fields(DatasetRow)]
    return [
        DatasetRow(
            repo_name=str(record["repo_name"]),
            commit_hash=str(record["commit_hash"]),
            docker_image=str(record["docker_image"]),
            problem_statement=str(record["problem_statement"]),
            expected_output_json=str(record["expected_output_json"]),
            parsed_commit_content=str(record["parsed_commit_content"]),
            modified_files=tuple(str(path) for path in record["modified_files"]),
            num_non_test_files=int(record["num_non_test_files"]),
            num_non_test_lines=int(record["num_non_test_lines"]),
        )
        for path in sorted((snapshot / "data").glob("train-*.parquet"))
        for record in pd.read_parquet(path, columns=column_names).to_dict("records")
    ]


def rank_candidates(rows: Sequence[DatasetRow], repo_name: str) -> list[DatasetRow]:
    """Rows of one repo from the upper half of ``num_non_test_lines`` with a source file, in hash order."""
    repo_rows = [row for row in rows if row.repo_name == repo_name]
    median_lines = statistics.median(row.num_non_test_lines for row in repo_rows)
    candidates = [
        row
        for row in repo_rows
        if row.num_non_test_lines >= median_lines and row.num_non_test_files >= 1
    ]
    return sorted(
        candidates,
        key=lambda row: hashlib.sha256(b"r2e-leak-select-v1\0" + row.commit_hash.encode()).digest(),
    )


def resolve_image_digest(docker_image: str) -> str:
    """Registry digest of the image's single-platform manifest (sha256 of the raw manifest bytes)."""
    raw_manifest = subprocess.run(  # noqa: S603
        ["docker", "buildx", "imagetools", "inspect", "--raw", docker_image],  # noqa: S607
        capture_output=True,
        check=True,
        timeout=120,
    ).stdout
    manifest = json.loads(raw_manifest)
    if manifest.get("mediaType", "").endswith(("manifest.list.v2+json", "image.index.v1+json")):
        raise RuntimeError(
            f"{docker_image} is a multi-platform index; pin a platform manifest instead"
        )
    return "sha256:" + hashlib.sha256(raw_manifest).hexdigest()


def task_from_row(row: DatasetRow, image_digest: str) -> R2eLeakTask:
    """Build a task from a dataset row; refuses rows whose gold patch cannot be reconstructed."""
    gold_patch = extract_gold_patch(row.parsed_commit_content)
    if not gold_patch.strip():
        raise ValueError(f"empty gold patch for {row.repo_name} {row.commit_hash}")
    return R2eLeakTask(
        task_id=task_id_for(row.repo_name, row.commit_hash),
        repo_name=row.repo_name,
        commit_hash=row.commit_hash,
        docker_image=row.docker_image,
        image_digest=image_digest,
        problem_statement=row.problem_statement,
        expected_output=json.loads(row.expected_output_json),
        gold_patch=gold_patch,
        modified_files=row.modified_files,
        num_non_test_files=row.num_non_test_files,
        num_non_test_lines=row.num_non_test_lines,
    )


def measure_tree(tree: Path) -> TreeMeasurement:
    """Count entries and apparent bytes without following symlinks, split into whole tree, .git and .venv."""

    def walk(root: Path) -> tuple[int, int]:
        entries = total_bytes = 0
        for directory, directory_names, file_names in os.walk(root):
            entries += len(directory_names) + len(file_names)
            total_bytes += sum(os.lstat(Path(directory, name)).st_size for name in file_names)
        return entries, total_bytes

    entries, total_bytes = walk(tree)
    git_entries, git_bytes = walk(tree / ".git") if (tree / ".git").is_dir() else (0, 0)
    venv_entries, venv_bytes = walk(tree / ".venv") if (tree / ".venv").is_dir() else (0, 0)
    return TreeMeasurement(entries, total_bytes, git_entries, git_bytes, venv_entries, venv_bytes)


def pull_by_digest(task: R2eLeakTask) -> None:
    """Pull the digest-pinned image, refusing below the free-space floor."""
    require_free_disk()
    run(["docker", "pull", "--quiet", task.digest_pinned_image], timeout=3600)


def extract_template(task: R2eLeakTask, templates_root: Path, dataset_revision: str) -> Path:
    """Copy /testbed and /r2e_tests out of the task image; skip when provenance.json already matches."""
    task_directory = templates_root / task.task_id
    provenance_path = task_directory / "provenance.json"
    if provenance_path.is_file():
        if json.loads(provenance_path.read_text())["image_digest"] != task.image_digest:
            raise RuntimeError(
                f"{task_directory} was extracted from a different image digest; remove it first"
            )
        logger.info("resume: %s already extracted", task.task_id)
        return task_directory
    require_free_disk()
    if task_directory.exists():
        shutil.rmtree(task_directory)  # partial extraction without provenance is never trusted
    pull_by_digest(task)
    (task_directory / "extracted").mkdir(parents=True)
    (task_directory / "hidden_tests").mkdir()
    image_bytes = int(
        run(
            ["docker", "image", "inspect", "--format", "{{.Size}}", task.digest_pinned_image]
        ).stdout
    )
    container_id = run(["docker", "create", task.digest_pinned_image]).stdout.strip()
    try:
        run(["docker", "cp", f"{container_id}:/testbed/.", str(task_directory / "extracted")])
        run(["docker", "cp", f"{container_id}:/r2e_tests/.", str(task_directory / "hidden_tests")])
    finally:
        run(["docker", "rm", container_id])
    provenance = {
        "task_id": task.task_id,
        "image_digest": task.image_digest,
        "docker_image": task.docker_image,
        "dataset": DATASET_ID,
        "dataset_revision": dataset_revision,
        "dataset_license_declared": False,
        "dataset_license": None,
        "license_note": DATASET_LICENSE_NOTE,
        "image_bytes": image_bytes,
        "tree": asdict(measure_tree(task_directory / "extracted")),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return task_directory


def verify_template(
    task: R2eLeakTask, task_directory: Path, work_root: Path, runtime_image: str | None = None
) -> VerifyResult:
    """Mount a scratch copy of the extracted tree at /testbed (network none) and check base, gold and import.

    ``runtime_image`` substitutes another digest-pinned image of the same repo for the task's own, which tests
    whether the tree plus its ``.venv`` is self-contained enough for one shared runtime image per repo.
    """
    runtime_image = runtime_image or task.digest_pinned_image
    scratch = work_root / task.task_id
    if scratch.exists():
        remove_root_owned(scratch, runtime_image)
    (scratch / "out").mkdir(parents=True)
    (scratch / "gold").mkdir()
    shutil.copytree(task_directory / "extracted", scratch / "testbed", symlinks=True)
    (scratch / "gold" / "gold.patch").write_text(task.gold_patch, encoding="utf-8")
    import_name = IMPORT_NAME_BY_REPO[task.repo_name]
    try:
        run(
            [
                "docker", "run", "--rm", "--network", "none", "--memory", "8g", "--cpus", "2",
                "-v", f"{scratch / 'testbed'}:/testbed",
                "-v", f"{task_directory / 'hidden_tests'}:/r2e_tests:ro",
                "-v", f"{scratch / 'gold'}:/gold:ro",
                "-v", f"{scratch / 'out'}:/out",
                "-e", f"IMPORT_NAME={import_name}",
                "--entrypoint", "bash", runtime_image, "-c", _VERIFY_SCRIPT,
            ],
            timeout=1800,
        )  # fmt: skip
        shutil.copytree(scratch / "out", task_directory / "verify_logs", dirs_exist_ok=True)
        expected_json = json.dumps(task.expected_output)
        import_exit_code = int((scratch / "out" / "import.rc").read_text())
        import_file = (scratch / "out" / "import.txt").read_text().strip().splitlines()[-1]
        return VerifyResult(
            base_matches_gold_outcomes=outcomes_match(
                (scratch / "out" / "base.log").read_text(), expected_json
            ),
            gold_matches_gold_outcomes=outcomes_match(
                (scratch / "out" / "gold.log").read_text(), expected_json
            ),
            gold_patch_applied="APPLY_FAILED" not in (scratch / "out" / "apply.log").read_text(),
            import_file=import_file,
            import_resolves_to_mount=import_file.startswith("/testbed/") and import_exit_code == 0,
            import_exit_code=import_exit_code,
            import_file_with_testbed_cwd=(scratch / "out" / "import_cwd.txt")
            .read_text()
            .strip()
            .splitlines()[-1],
        )
    finally:
        remove_root_owned(scratch, runtime_image)


def remove_root_owned(directory: Path, image: str) -> None:
    """Delete a scratch tree the root-run container wrote into, from inside a root container."""
    run(
        [
            "docker", "run", "--rm", "--network", "none", "-v", f"{directory.parent}:/scratch",
            "--entrypoint", "rm", image, "-rf", f"/scratch/{directory.name}",
        ]
    )  # fmt: skip


def within_selection_limits(
    tree: TreeMeasurement, maximum_entries: int, maximum_bytes: int, *, count_venv: bool
) -> bool:
    """Plan selection rule (i): tree plus packed .git within the entry and byte limits.

    The image keeps its virtualenv inside /testbed; unless ``count_venv`` it is excluded from the count on the
    assumption that the backend mounts it separately and so outside the per-command workspace snapshot.
    """
    venv_entries, venv_bytes = (0, 0) if count_venv else (tree.venv_entries, tree.venv_bytes)
    return (
        tree.entries - venv_entries <= maximum_entries and tree.bytes - venv_bytes <= maximum_bytes
    )


def bake(  # noqa: PLR0913 - one selection knob per plan rule
    templates_root: Path,
    rows: Sequence[DatasetRow],
    dataset_revision: str,
    out_path: Path,
    *,
    maximum_entries: int = 15_000,
    maximum_bytes: int = 300 * 1024**2,
    count_venv: bool = False,
) -> int:
    """Write the case file from every template that verified (gold passes, base fails) and fits the size limits."""
    tasks: list[R2eLeakTask] = []
    for verify_path in sorted(templates_root.glob("*/verify.json")):
        verification = json.loads(verify_path.read_text())
        if not (
            verification["gold_matches_gold_outcomes"]
            and not verification["base_matches_gold_outcomes"]
        ):
            logger.warning(
                "skipping %s: gold must pass and base must fail", verify_path.parent.name
            )
            continue
        provenance = json.loads((verify_path.parent / "provenance.json").read_text())
        if not within_selection_limits(
            TreeMeasurement(**provenance["tree"]),
            maximum_entries,
            maximum_bytes,
            count_venv=count_venv,
        ):
            logger.warning(
                "skipping %s: tree exceeds the selection size limits", verify_path.parent.name
            )
            continue
        row = next(row for row in rows if row.docker_image == provenance["docker_image"])
        tasks.append(task_from_row(row, provenance["image_digest"]))
    partitions = deterministic_partitions((task.repo_name, task.commit_hash) for task in tasks)
    payload = {
        "schema": DATA_SCHEMA,
        "provenance": {
            "dataset": DATASET_ID,
            "dataset_revision": dataset_revision,
            "dataset_license_declared": False,
            "dataset_license": None,
            "license_note": DATASET_LICENSE_NOTE,
            "split_algorithm": SPLIT_ALGORITHM,
            "task_count": len(tasks),
            "partition_counts": {
                name: sum(partitions[task.repo_name, task.commit_hash] == name for task in tasks)
                for name in PARTITIONS
            },
        },
        "tasks": [asdict(task) for task in tasks],
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return len(tasks)


def _inventory(rows: Sequence[DatasetRow]) -> None:
    lines = [f"dataset rows: {len(rows)}", "repo tasks median_non_test_lines median_non_test_files"]
    for repo_name in sorted({row.repo_name for row in rows}):
        repo_rows = [row for row in rows if row.repo_name == repo_name]
        median_lines = statistics.median(row.num_non_test_lines for row in repo_rows)
        median_files = statistics.median(row.num_non_test_files for row in repo_rows)
        lines.append(f"{repo_name} {len(repo_rows)} {median_lines} {median_files}")
    logger.info("\n".join(lines))


def _extract_command(arguments: argparse.Namespace, rows: Sequence[DatasetRow]) -> None:
    templates_root = arguments.root / "templates"
    if arguments.task_id:
        selected = [
            row for row in rows if task_id_for(row.repo_name, row.commit_hash) in arguments.task_id
        ]
    else:
        selected = rank_candidates(rows, arguments.repo)[: arguments.count]
    for row in selected:
        task = task_from_row(row, resolve_image_digest(row.docker_image))
        task_directory = extract_template(task, templates_root, arguments.revision)
        logger.info("%s: %s", task.task_id, asdict(measure_tree(task_directory / "extracted")))
        if arguments.verify:
            result = verify_template(
                task, task_directory, arguments.root / "work", arguments.runtime_image
            )
            verify_name = (
                "verify.json" if arguments.runtime_image is None else "verify-shared-runtime.json"
            )
            (task_directory / verify_name).write_text(json.dumps(asdict(result), indent=2))
            logger.info("%s verify: %s", task.task_id, result)
        if arguments.remove_image:
            run(["docker", "rmi", task.digest_pinned_image])


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["inventory", "extract", "bake"])
    parser.add_argument("--root", type=Path, default=LEAK_ENV_ROOT)
    parser.add_argument("--revision", default=PINNED_DATASET_REVISION)
    parser.add_argument("--repo")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument(
        "--verify", action="store_true", help="run gold/base/import checks after extraction"
    )
    parser.add_argument(
        "--remove-image", action="store_true", help="docker rmi the task image after extraction"
    )
    parser.add_argument("--max-entries", type=int, default=15_000, help="bake: tree entry limit")
    parser.add_argument(
        "--max-bytes", type=int, default=300 * 1024**2, help="bake: tree byte limit"
    )
    parser.add_argument(
        "--count-venv", action="store_true", help="bake: count .venv against the limits"
    )
    parser.add_argument(
        "--runtime-image",
        help="digest-pinned image to verify under instead of each task's own (shared-runtime test)",
    )
    parser.add_argument("--out", type=Path, default=DATA_PATH)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    require_free_disk()
    rows = load_rows(arguments.revision, arguments.root / "hf-cache")
    if arguments.command == "inventory":
        _inventory(rows)
    elif arguments.command == "extract":
        if not arguments.repo and not arguments.task_id:
            parser.error("extract needs --repo or --task-id")
        _extract_command(arguments, rows)
    else:
        logger.info(
            "baked %d tasks",
            bake(
                arguments.root / "templates",
                rows,
                arguments.revision,
                arguments.out,
                maximum_entries=arguments.max_entries,
                maximum_bytes=arguments.max_bytes,
                count_venv=arguments.count_venv,
            ),
        )


if __name__ == "__main__":
    main()
