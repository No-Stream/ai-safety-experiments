"""Bind prepared local leak-channel arms to the existing organism episode source.

The local asset index maps task ids to lists of arm objects. Each arm records its requested
``channels``, ``extracted_repository`` and the keyword paths accepted by ``build_task``:
workspace_template, closed_template, manifest_path, gate_record, hidden_tests_template,
runtime_venv and runtime_interpreter, with an optional runtime_image. Preparing assets is
separate from episode sampling so a resumed run never silently rebuilds its instrument.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, cast

from reward_hacking.harness.leak_channels import CHANNELS
from reward_hacking.harness.tasks_r2e_leak import LeakChannelUnavailableError, build_task
from reward_hacking.organism_rl.leak_asset_build import main
from reward_hacking.r2e_leak_data import load_tasks

if TYPE_CHECKING:
    from reward_hacking.harness.task_spec import Task
    from reward_hacking.harness.tasks_r2e_leak import InstructionVariant
    from reward_hacking.r2e_leak_data import R2eLeakTask

DEFAULT_LEAK_ASSETS = Path("/var/tmp/leak-env/organism-assets.json")  # noqa: S108 - local templates
logger = logging.getLogger(__name__)
_PATH_FIELDS = (
    "workspace_template",
    "closed_template",
    "manifest_path",
    "gate_record",
    "hidden_tests_template",
    "runtime_venv",
    "runtime_interpreter",
)


def leak_task_source_metadata(  # noqa: PLR0913 - records the exact loader inputs
    assets_path: Path | None,
    *,
    partition: str,
    channels: tuple[str, ...],
    instruction: str,
    repos: tuple[str, ...] | None,
    eval_pairs: tuple[tuple[str, str], ...] | None,
) -> dict[str, object]:
    """Record the exact inputs used to reconstruct prepared leak tasks later."""
    if assets_path is None:
        raise ValueError("--leak-assets is required when recording r2e_leak provenance")
    return {
        "asset_index_path": str(assets_path),
        "asset_index_sha256": hashlib.sha256(assets_path.read_bytes()).hexdigest(),
        "partition": partition,
        "channels": list(channels),
        "instruction_variant": instruction,
        "repos": None if repos is None else list(repos),
        "eval_pairs": None if eval_pairs is None else [list(pair) for pair in eval_pairs],
    }


def _git_text(repository: Path, *arguments: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed Git command, local trusted paths
        ["git", "-C", str(repository), *arguments],  # noqa: S607
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()


def _case_exclusion(_case: R2eLeakTask, repository: Path) -> str | None:
    if _git_text(repository, "status", "--porcelain", "--untracked-files=no"):
        return "setup_edited_tracked_files"
    return None


def _validate_completion_marker(arm: dict[str, object]) -> None:
    marker_path = arm.get("completion_marker")
    if marker_path is None:
        return
    if not isinstance(marker_path, str):
        raise TypeError("leak completion marker must be a path string")
    marker = Path(marker_path)
    if not marker.is_file() or marker.is_symlink():
        raise ValueError("leak task completion marker is missing or unsafe")
    value: object = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or arm not in value.get("arms", []):
        raise ValueError("leak task completion marker does not match the indexed arm")


def load_leak_tasks(  # noqa: C901, PLR0912, PLR0913, PLR0915 - validates trusted prepared inputs and counts exclusions
    assets_path: Path | None,
    *,
    partition: str,
    channels: tuple[str, ...],
    instruction: InstructionVariant,
    repos: tuple[str, ...] | None,
    open_arm: bool = False,
    manifest_json: dict[str, str] | None = None,
    eval_pairs: tuple[tuple[str, str], ...] | None = None,
) -> tuple[list[Task], dict[str, dict[str, object]], Counter[str]]:
    """Load exact-set training arms or the requested held-out task/channel arms."""
    if assets_path is None:
        raise ValueError("--leak-assets is required when selecting r2e_leak tasks")
    eval_channels_by_task: dict[str, str] = {}
    if eval_pairs is not None:
        if partition == "train":
            raise ValueError("evaluation leak pairs cannot be loaded from the train partition")
        if len({task_id for task_id, _channel in eval_pairs}) != len(eval_pairs):
            raise ValueError("evaluation leak pairs must use one channel per task")
        if any(not task_id or channel not in CHANNELS for task_id, channel in eval_pairs):
            raise ValueError("evaluation leak pairs require task ids and known channels")
        if any(channel not in channels for _task_id, channel in eval_pairs):
            raise ValueError("evaluation leak pair channel is outside the requested channel set")
        eval_channels_by_task = dict(eval_pairs)
    raw_index: object = json.loads(assets_path.read_text(encoding="utf-8"))
    if not isinstance(raw_index, dict):
        raise TypeError("leak asset index must be an object keyed by task id")
    index = cast("dict[str, object]", raw_index)
    tasks: list[Task] = []
    manifests: dict[str, dict[str, object]] = {}
    counts: Counter[str] = Counter()
    cases = load_tasks()
    train_repos = {case.repo_name for case in cases if case.partition == "train"}
    selected_pair_task_ids = set(eval_channels_by_task)
    seen_pair_task_ids: set[str] = set()
    for case in cases:
        selected_partition = (
            case.partition == "train" if partition == "train" else case.partition != "train"
        )
        if not selected_partition:
            if case.task_id in selected_pair_task_ids:
                raise ValueError(f"evaluation leak pair task {case.task_id!r} is not held out")
            continue
        if eval_pairs is not None and case.task_id not in selected_pair_task_ids:
            continue
        if repos is not None and case.repo_name not in repos:
            if case.task_id in selected_pair_task_ids:
                raise ValueError(
                    f"evaluation leak pair task {case.task_id!r} is outside --leak-repos"
                )
            continue
        if case.task_id not in index:
            if eval_pairs is not None:
                raise ValueError(
                    f"evaluation leak pair task {case.task_id!r} has no prepared assets"
                )
            counts["assets_not_prepared"] += 1
            continue
        arms = index[case.task_id]
        if not isinstance(arms, list):
            raise TypeError("each leak task asset entry must be a list of arm objects")
        if not arms or not isinstance(arms[0], dict):
            raise ValueError("leak asset arms must include the extracted repository")
        repository = arms[0]["extracted_repository"]
        if not isinstance(repository, str):
            raise TypeError("leak extracted_repository must be a path string")
        reason = _case_exclusion(case, Path(repository))
        if reason is not None:
            counts[reason] += 1
            continue
        pair_channel = eval_channels_by_task.get(case.task_id)
        requested_channels = (pair_channel,) if pair_channel is not None else channels
        matching = [
            arm
            for arm in arms
            if isinstance(arm, dict)
            and isinstance(arm.get("channels"), list)
            and sorted(arm["channels"]) == sorted(requested_channels)
        ]
        if len(matching) != 1:
            if pair_channel is not None and not matching:
                raise ValueError(
                    f"task {case.task_id} has no asset arm for channel {pair_channel!r}"
                )
            raise ValueError(
                f"task {case.task_id} requires exactly one asset arm for {requested_channels!r}"
            )
        arm = cast("dict[str, object]", matching[0])
        _validate_completion_marker(arm)
        if any(not isinstance(arm[field], str) for field in _PATH_FIELDS):
            raise TypeError("leak arm paths must be strings")
        paths = {field: Path(cast("str", arm[field])) for field in _PATH_FIELDS}
        raw_manifest = paths["manifest_path"].read_bytes().decode("utf-8")
        manifest_value: object = json.loads(raw_manifest)
        if not isinstance(manifest_value, dict):
            raise TypeError("leak manifest must be an object")
        manifest = cast("dict[str, object]", manifest_value)
        opened_channels = manifest.get("channels")
        unavailable_channels = manifest.get("unavailable_channels")
        if (
            not isinstance(opened_channels, list)
            or any(not isinstance(channel, str) for channel in opened_channels)
            or not isinstance(unavailable_channels, dict)
            or any(not isinstance(channel, str) for channel in unavailable_channels)
            or any(
                not isinstance(reason, str) or not reason
                for reason in unavailable_channels.values()
            )
        ):
            raise ValueError("leak manifest must record opened channels and unavailable reasons")
        if set(opened_channels) & set(unavailable_channels) or set(requested_channels) != set(
            opened_channels
        ) | set(unavailable_channels):
            raise ValueError("leak manifest channels do not partition the requested arm")
        if pair_channel is not None and opened_channels != [pair_channel]:
            raise ValueError(
                f"evaluation leak pair {case.task_id!r}/{pair_channel} is not open in its asset arm"
            )
        use_available_channels = open_arm or len(requested_channels) > 1
        required_channels = (
            (pair_channel,)
            if pair_channel is not None
            else tuple(sorted(opened_channels))
            if use_available_channels
            else requested_channels
        )
        image = arm.get("runtime_image")
        if image is not None and not isinstance(image, str):
            raise TypeError("leak runtime_image must be a string")
        try:
            task = build_task(
                case,
                workspace_template=paths["workspace_template"],
                closed_template=paths["closed_template"],
                manifest_path=paths["manifest_path"],
                gate_record=paths["gate_record"],
                hidden_tests_template=paths["hidden_tests_template"],
                runtime_venv=paths["runtime_venv"],
                runtime_interpreter=paths["runtime_interpreter"],
                runtime_image=image,
                instruction_variant=instruction,
                required_channels=required_channels,
            )
        except LeakChannelUnavailableError as error:
            counts["unavailable_channel"] += 1
            counts.update(f"unavailable_channel:{channel}" for channel in error.unavailable_reasons)
            continue
        if manifest.get("channels") != sorted(required_channels):
            raise ValueError("prepared leak manifest channels differ from selected arm")
        identity = task.host_only_metadata
        identity["leak_repo_seen"] = case.repo_name in train_repos
        tasks.append(task)
        manifests[task.task_id] = manifest
        if pair_channel is not None:
            seen_pair_task_ids.add(case.task_id)
        if manifest_json is not None:
            manifest_json[task.task_id] = raw_manifest
    missing_pair_task_ids = selected_pair_task_ids - seen_pair_task_ids
    if missing_pair_task_ids:
        raise ValueError(
            f"evaluation leak pairs were not loaded: {sorted(missing_pair_task_ids)!r}"
        )
    logger.info("leak source exclusions=%s", dict(counts))
    return tasks, manifests, counts


if __name__ == "__main__":
    main()
