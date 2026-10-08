"""Bind prepared local leak-channel arms to the existing organism episode source.

The local asset index maps task ids to lists of arm objects. Each arm records its requested
``channels``, ``extracted_repository`` and the keyword paths accepted by ``build_task``:
workspace_template, closed_template, manifest_path, gate_record, hidden_tests_template,
runtime_venv and runtime_interpreter, with an optional runtime_image. Preparing assets is
separate from episode sampling so a resumed run never silently rebuilds its instrument.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
    assets_path: Path,
    *,
    partition: str,
    channels: tuple[str, ...],
    instruction: InstructionVariant,
    repos: tuple[str, ...] | None,
    open_arm: bool = False,
    manifest_json: dict[str, str] | None = None,
) -> tuple[list[Task], dict[str, dict[str, object]], Counter[str]]:
    """Load one prepared arm per case and count only explicitly supported exclusions."""
    raw_index: object = json.loads(assets_path.read_text(encoding="utf-8"))
    if not isinstance(raw_index, dict):
        raise TypeError("leak asset index must be an object keyed by task id")
    index = cast("dict[str, object]", raw_index)
    tasks: list[Task] = []
    manifests: dict[str, dict[str, object]] = {}
    counts: Counter[str] = Counter()
    for case in load_tasks():
        selected_partition = (
            case.partition == "train" if partition == "train" else case.partition != "train"
        )
        if not selected_partition:
            continue
        if repos is not None and case.repo_name not in repos:
            continue
        if case.task_id not in index:
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
        matching = [
            arm
            for arm in arms
            if isinstance(arm, dict)
            and isinstance(arm.get("channels"), list)
            and sorted(arm["channels"]) == sorted(channels)
        ]
        if len(matching) != 1:
            raise ValueError(f"task {case.task_id} requires exactly one asset arm for {channels!r}")
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
        if set(opened_channels) & set(unavailable_channels) or set(channels) != set(
            opened_channels
        ) | set(unavailable_channels):
            raise ValueError("leak manifest channels do not partition the requested arm")
        use_available_channels = open_arm or len(channels) > 1
        required_channels = tuple(sorted(opened_channels)) if use_available_channels else channels
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
        tasks.append(task)
        manifests[task.task_id] = manifest
        if manifest_json is not None:
            manifest_json[task.task_id] = raw_manifest
    logger.info("leak source exclusions=%s", dict(counts))
    return tasks, manifests, counts


if __name__ == "__main__":
    main()
