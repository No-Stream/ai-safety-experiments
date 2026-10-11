"""Build local, resumable leak assets; no benchmark payload belongs in the code repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from reward_hacking.harness import docker_backend
from reward_hacking.harness.leak_channels import (
    ALL_COMPATIBLE_CHANNELS,
    BUILDER_VERSION,
    CHANNELS,
    BuildRequest,
    CompiledPycArtifacts,
    build_channel_templates,
    compiled_pyc_artifacts_from_templates,
    probe_closed,
    probe_open,
    template_fingerprint,
)
from reward_hacking.harness.leak_fix_spec import (
    fix_commit_spec,
    git_text,
    interpreter_paths,
)
from reward_hacking.harness.tasks_r2e_leak import prepare_runtime
from reward_hacking.organism_rl.leak_selection import parse_channels
from reward_hacking.r2e_leak_data import DATA_PATH, load_tasks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reward_hacking.r2e_leak_data import R2eLeakTask

logger = logging.getLogger(__name__)
_GIB = 1024**3
_SOURCE_CODE_PATHS = (
    "reward_hacking/organism_rl/leak_asset_build.py",
    "reward_hacking/harness/leak_channels.py",
    "reward_hacking/harness/leak_fix_spec.py",
    "reward_hacking/harness/tasks_r2e_leak.py",
)


@dataclass(frozen=True)
class BuildOptions:
    """Local destinations and the exact arms to prepare for every extracted case."""

    templates_root: Path
    out_root: Path
    split: str
    interpreters_root: Path
    runtime_root: Path
    gate_record: Path
    arms: tuple[str, ...] = ("open", "closed")
    channel_splits: Path | None = None
    minimum_free_bytes: int = 40 * _GIB
    runtime_image: str | None = None


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return cast("dict[str, Any]", value)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _channel_sets(options: BuildOptions, case: R2eLeakTask) -> tuple[tuple[str, ...], ...]:
    allowed = tuple(sorted(ALL_COMPATIBLE_CHANNELS))
    if options.channel_splits is not None:
        config = _json_object(options.channel_splits)
        split: object = config[options.split]
        if not isinstance(split, dict) or set(split) != {"train", "held-out"}:
            raise ValueError("channel split requires train and held-out lists")
        split = cast("dict[str, object]", split)
        sets: dict[str, tuple[str, ...]] = {}
        for partition, values in split.items():
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise TypeError("channel split values must be channel-name lists")
            names = cast("list[str]", values)
            if len(set(names)) != len(names) or set(names) - set(CHANNELS):
                raise ValueError("channel split values must be unique known channel names")
            sets[partition] = tuple(sorted(names))
        if set(sets["train"]) & set(sets["held-out"]):
            raise ValueError("train and held-out channels must be disjoint")
        allowed = sets["train" if case.partition == "train" else "held-out"]
    selected: list[tuple[str, ...]] = []
    for arm in options.arms:
        channels = (
            tuple(channel for channel in allowed if channel in ALL_COMPATIBLE_CHANNELS)
            if arm == "open"
            else parse_channels(arm)
        )
        if options.channel_splits is not None and set(channels) - set(allowed):
            raise ValueError("requested arm opens channels outside its partition")
        if channels not in selected:
            selected.append(channels)
    return tuple(selected)


def _disk_bytes(root: Path) -> int:
    return sum(path.lstat().st_blocks * 512 for path in root.rglob("*") if not path.is_symlink())


def _runtime_content_digest(root: Path) -> str:
    """Compare extracted runtime contents independently of copy paths and timestamps."""
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            content = str(path.readlink()).encode()
        elif path.is_file():
            content = path.read_bytes()
        else:
            content = b"directory"
        entries.append((relative, hashlib.sha256(content).hexdigest()))
    return _digest(entries)


def _runtime(options: BuildOptions, case: R2eLeakTask, repository: Path) -> tuple[Path, Path, Path]:
    runtime_namespace = options.runtime_root / options.split / case.repo_name
    runtime_namespace.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(runtime_namespace).free < options.minimum_free_bytes:
        raise OSError("free disk is below the configured runtime-build floor")
    venv, interpreter, binary = interpreter_paths(repository, options.interpreters_root)
    content = {
        "venv": _runtime_content_digest(venv),
        "interpreter": _runtime_content_digest(interpreter),
        "binary": binary,
    }
    content_key = _digest(content)
    source_marker = runtime_namespace / f"{content_key}-source.json"
    if source_marker.exists():
        source = _json_object(source_marker)
        if source["content"] != content:
            raise ValueError("runtime variant source marker content changed")
        venv = Path(source["venv"])
        interpreter = Path(source["interpreter"])
        binary = source["binary"]
    else:
        _atomic_json(
            source_marker,
            {
                "venv": str(venv),
                "interpreter": str(interpreter),
                "binary": binary,
                "content": content,
            },
        )
    runtime_venv, runtime_interpreter = prepare_runtime(
        venv, interpreter, runtime_namespace / content_key, binary
    )
    runtime_bytes = _disk_bytes(runtime_namespace / content_key)
    logger.info(
        "runtime repo=%s variant=%s disk_bytes=%d (%0.3f GiB)",
        case.repo_name,
        content_key[:12],
        runtime_bytes,
        runtime_bytes / _GIB,
    )
    return runtime_venv, runtime_interpreter, interpreter / "bin" / binary


# Guards on how a build runs, not on what it produces; changing them must not invalidate a resume.
_OPERATIONAL_OPTIONS = frozenset({"minimum_free_bytes"})


def _source_code_sha256() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[2]
    return {
        relative_path: hashlib.sha256((repository_root / relative_path).read_bytes()).hexdigest()
        for relative_path in _SOURCE_CODE_PATHS
    }


def _request_identity_payload(
    options: BuildOptions, case: R2eLeakTask, provenance: dict[str, Any]
) -> dict[str, object]:
    return {
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(options).items()
            if key not in _OPERATIONAL_OPTIONS
        },
        "case": asdict(case),
        "extraction": provenance,
        "source_fingerprint": template_fingerprint(
            options.templates_root / case.task_id / "extracted"
        ),
        "hidden_tests_fingerprint": template_fingerprint(
            options.templates_root / case.task_id / "hidden_tests"
        ),
        "channel_config": None
        if options.channel_splits is None
        else _json_object(options.channel_splits)[options.split],
    }


def _request_identity(options: BuildOptions, case: R2eLeakTask, provenance: dict[str, Any]) -> str:
    return _digest(_request_identity_payload(options, case, provenance))


def _legacy_request_identity(request_identity: dict[str, object], builder_version: str) -> str:
    return _digest({"builder": builder_version, **request_identity})


def _legacy_builder_version(arms_value: object) -> str:
    if not isinstance(arms_value, list) or any(not isinstance(arm, dict) for arm in arms_value):
        raise TypeError("legacy completion marker arms must be a list of objects")
    for arm in arms_value:
        manifest_path = arm.get("manifest_path")
        if not isinstance(manifest_path, str):
            raise TypeError("legacy completion arm must name its manifest")
        manifest = _json_object(Path(manifest_path))
        builder_version = manifest.get("builder_version")
        if isinstance(builder_version, str):
            return builder_version
    raise ValueError("legacy completion marker does not record its builder version")


def _validate_completion_identity(
    marker: dict[str, Any], identity: str, request_identity: dict[str, object]
) -> None:
    marker_identity = marker.get("identity")
    if not isinstance(marker_identity, str):
        raise TypeError("task completion marker must contain an identity digest")
    saved_request_identity = marker.get("request_identity")
    if isinstance(saved_request_identity, dict):
        if marker.get("builder_version") != BUILDER_VERSION:
            raise ValueError("task completion builder version changed; choose a fresh output root")
        if marker_identity != _digest(saved_request_identity):
            raise ValueError("task completion identity changed; choose a fresh output root")
        current_comparable_identity = {
            key: value for key, value in saved_request_identity.items() if key != "gate_sha256"
        }
        if (
            _digest(current_comparable_identity) != identity
            or _digest(request_identity) != identity
        ):
            raise ValueError("task completion identity changed; choose a fresh output root")
    elif saved_request_identity is None:
        builder_version = marker.get("builder_version")
        if not isinstance(builder_version, str):
            builder_version = _legacy_builder_version(marker.get("arms"))
        if builder_version != BUILDER_VERSION:
            raise ValueError("task completion builder version changed; choose a fresh output root")
        if marker_identity != _legacy_request_identity(request_identity, builder_version):
            raise ValueError("task completion identity changed; choose a fresh output root")
    else:
        raise TypeError("task completion request identity must be an object")


def _completed_arms(
    task_root: Path,
    identity: str,
    request_identity: dict[str, object],
) -> list[dict[str, object]] | None:
    marker_path = task_root / "complete.json"
    if not marker_path.exists():
        return None
    marker = _json_object(marker_path)
    _validate_completion_identity(marker, identity, request_identity)
    for relative, expected in marker["fingerprints"].items():
        path = task_root / relative
        actual = (
            template_fingerprint(path)
            if path.is_dir()
            else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        if actual != expected:
            raise ValueError("task completion fingerprints changed; choose a fresh output root")
    arms = cast("list[dict[str, object]]", marker["arms"])
    _validate_runner_isolation(arms)
    return arms


def _validate_runner_isolation(arms: list[dict[str, object]]) -> None:
    for arm in arms:
        workspace_runner = Path(cast("str", arm["workspace_template"])) / "run_tests.sh"
        closed_runner = Path(cast("str", arm["closed_template"])) / "run_tests.sh"
        if (
            workspace_runner.exists()
            or workspace_runner.is_symlink()
            or not closed_runner.is_file()
            or closed_runner.is_symlink()
        ):
            raise ValueError(
                "task completion uses an exposed or missing grader runner; choose a fresh output root"
            )


def _remove_agent_runner(workspace: Path, manifest_path: Path) -> None:
    runner = workspace / "run_tests.sh"
    if not runner.is_file() or runner.is_symlink():
        raise ValueError("built workspace must contain a regular runner before isolation")
    if git_text(workspace, "ls-tree", "--name-only", "HEAD", "--", "run_tests.sh"):
        raise ValueError(
            "repository base tracks run_tests.sh; refusing to remove repository content"
        )
    root_stat = workspace.stat()
    runner.unlink()
    os.utime(workspace, ns=(root_stat.st_atime_ns, root_stat.st_mtime_ns))
    manifest = _json_object(manifest_path)
    manifest["template_fingerprint"] = template_fingerprint(workspace)
    _atomic_json(manifest_path, manifest)


def _select_cases(
    cases: Sequence[R2eLeakTask], selection_path: Path | None
) -> tuple[R2eLeakTask, ...]:
    if selection_path is None:
        return tuple(cases)
    selection = _json_object(selection_path)
    if set(selection) != {"partition", "task_ids"}:
        raise ValueError("case selector requires exactly partition and task_ids")
    partition = selection["partition"]
    task_ids = selection["task_ids"]
    if not isinstance(partition, str) or partition not in {"train", "held-out"}:
        raise ValueError("case selector partition must be train or held-out")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task_id, str) for task_id in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise ValueError("case selector task_ids must be a nonempty list of unique task ids")
    cases_by_id = {case.task_id: case for case in cases}
    missing = set(task_ids) - cases_by_id.keys()
    if missing:
        raise ValueError(f"case selector contains unknown task ids: {sorted(missing)}")
    selected_cases = tuple(cases_by_id[task_id] for task_id in task_ids)
    if any((case.partition == "train") != (partition == "train") for case in selected_cases):
        raise ValueError("case selector task ids do not belong to its declared partition")
    return selected_cases


def _build_case(  # noqa: PLR0913, PLR0917 - exact per-task source and destination contract
    options: BuildOptions,
    case: R2eLeakTask,
    task_root: Path,
    repository: Path,
    hidden_tests: Path,
    spec: dict[str, object],
) -> list[dict[str, object]]:
    runtime_venv, runtime_interpreter, compile_python = _runtime(options, case, repository)
    task_root.mkdir(parents=True)
    arms: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix=".base-", dir=task_root) as temporary:
        base = Path(temporary) / "repository"
        shutil.copytree(
            repository, base, symlinks=True, ignore=shutil.ignore_patterns(".venv", "r2e_tests")
        )
        channel_sets = _channel_sets(options, case)
        build_sets = [channels for channels in channel_sets if channels] or [()]
        common_closed = task_root / "closed"
        common_manifest: dict[str, Any] | None = None
        compiled_pyc_artifacts: CompiledPycArtifacts | None = None
        paths: dict[tuple[str, ...], tuple[Path, Path]] = {}
        for channels in build_sets:
            output = task_root / ("channels-" + _digest(channels)[:12])
            manifest = build_channel_templates(
                BuildRequest(
                    base_dir=base,
                    fix_commit_spec=spec,
                    channels=channels,
                    out_dir=output,
                    compile_python=str(compile_python),
                    compiled_pyc_artifacts=compiled_pyc_artifacts,
                )
            )
            captured_pyc_artifacts = compiled_pyc_artifacts_from_templates(
                output / "closed", output / "open", manifest
            )
            if compiled_pyc_artifacts is not None and (
                compiled_pyc_artifacts.base_files != captured_pyc_artifacts.base_files
                or compiled_pyc_artifacts.base_uncompiled_sources
                != captured_pyc_artifacts.base_uncompiled_sources
                or (
                    compiled_pyc_artifacts.fixed_files
                    and captured_pyc_artifacts.fixed_files
                    and compiled_pyc_artifacts.fixed_files != captured_pyc_artifacts.fixed_files
                )
            ):
                raise ValueError("shared task bytecode differs across channel arms")
            fixed_pyc_files = captured_pyc_artifacts.fixed_files
            if compiled_pyc_artifacts is not None and not fixed_pyc_files:
                fixed_pyc_files = compiled_pyc_artifacts.fixed_files
            compiled_pyc_artifacts = CompiledPycArtifacts(
                base_files=captured_pyc_artifacts.base_files,
                base_uncompiled_sources=captured_pyc_artifacts.base_uncompiled_sources,
                fixed_files=fixed_pyc_files,
            )
            if common_manifest is None:
                (output / "closed").rename(common_closed)
                common_manifest = {
                    **manifest,
                    "channels": [],
                    "unavailable_channels": {},
                    "template_fingerprint": template_fingerprint(common_closed),
                }
                _atomic_json(task_root / "closed-manifest.json", common_manifest)
            else:
                shutil.rmtree(output / "closed")
            if channels:
                _remove_agent_runner(output / "open", output / "manifest.json")
                paths[channels] = (output / "open", output / "manifest.json")
            else:
                shutil.rmtree(output)
        closed_workspace = task_root / "workspace-closed"
        shutil.copytree(common_closed, closed_workspace, symlinks=True)
        _remove_agent_runner(closed_workspace, task_root / "closed-manifest.json")
        for channels in channel_sets:
            workspace, manifest_path = (
                paths[channels]
                if channels
                else (closed_workspace, task_root / "closed-manifest.json")
            )
            arm: dict[str, object] = {
                "channels": list(channels),
                "completion_marker": str(task_root / "complete.json"),
                "extracted_repository": str(repository),
                "workspace_template": str(workspace),
                "closed_template": str(common_closed),
                "manifest_path": str(manifest_path),
                "gate_record": str(options.gate_record),
                "hidden_tests_template": str(hidden_tests),
                "runtime_venv": str(runtime_venv),
                "runtime_interpreter": str(runtime_interpreter),
            }
            if options.runtime_image is not None:
                arm["runtime_image"] = options.runtime_image
            arms.append(arm)
    return arms


def _probed_channels(manifest: Mapping[str, Any], planted: set[str]) -> tuple[str, ...]:
    """Channels whose open and closed probes apply to this task.

    An mtime leak needs the fix and base commits to differ in date; when they share one, the
    channel cannot exist in any template, so it is skipped rather than probed.
    """
    fix = manifest["fix"]
    if not isinstance(fix, Mapping):
        raise TypeError("leak template manifest fix must be an object")
    if int(cast("int", fix["fix_date"])) != int(cast("int", fix["base_date"])):
        return CHANNELS
    if "mtime" in planted:
        raise ValueError("mtime channel planted although fix and base dates are equal")
    return tuple(channel for channel in CHANNELS if channel != "mtime")


def _assert_template_probe_states(arms: Sequence[dict[str, object]]) -> None:
    for arm in arms:
        closed_template = Path(cast("str", arm["closed_template"]))
        workspace_template = Path(cast("str", arm["workspace_template"]))
        manifest = _json_object(Path(cast("str", arm["manifest_path"])))
        planted_channels = manifest.get("channels")
        if not isinstance(planted_channels, list) or any(
            not isinstance(channel, str) for channel in planted_channels
        ):
            raise TypeError("leak template manifest channels must be a list of strings")
        planted = set(planted_channels)
        for channel in _probed_channels(manifest, planted):
            closed_open_probe = probe_open(closed_template, manifest, channel)
            closed_closed_probe = probe_closed(closed_template, manifest, channel)
            if closed_open_probe or not closed_closed_probe:
                raise ValueError(f"closed template {channel} probe mismatch")

            should_be_open = channel in planted
            workspace_open_probe = probe_open(workspace_template, manifest, channel)
            workspace_closed_probe = probe_closed(workspace_template, manifest, channel)
            if workspace_open_probe != should_be_open or workspace_closed_probe == should_be_open:
                raise ValueError(
                    f"workspace template {channel} probe mismatch: "
                    f"open={workspace_open_probe} closed={workspace_closed_probe} expected_open={should_be_open}"
                )


def _venv_uses_uv_interpreter(repository: Path) -> bool:
    """Only uv-managed interpreters can be staged as a shared runtime; an image's system Python cannot."""
    home_lines = [
        line.partition("=")[2].strip()
        for line in (repository / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8").splitlines()
        if line.partition("=")[0].strip() == "home"
    ]
    if len(home_lines) != 1:
        raise ValueError("extracted venv pyvenv.cfg must declare exactly one home")
    return "/uv/python/" in home_lines[0]


def build_assets(options: BuildOptions, cases: Sequence[R2eLeakTask]) -> dict[str, object]:  # noqa: C901, PLR0912, PLR0915 - linear resumable build orchestration
    """Write only completed assets; interruption leaves a task to rebuild on the next call."""
    if (
        not options.split
        or Path(options.split).name != options.split
        or options.split in {".", ".."}
    ):
        raise ValueError("split must be a single path-safe name")
    if options.minimum_free_bytes < 0 or not options.arms:
        raise ValueError("disk floor must be nonnegative and at least one arm is required")
    destinations = (options.templates_root, options.interpreters_root, options.runtime_root)
    for source in destinations:
        if (
            options.out_root.resolve() == source.resolve()
            or options.out_root.resolve() in source.resolve().parents
            or source.resolve() in options.out_root.resolve().parents
        ):
            raise ValueError("asset output and source/runtime roots must not contain one another")
    options.out_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(options.out_root).free < options.minimum_free_bytes:
        raise OSError("free disk is below the configured asset-build floor")
    index: dict[str, list[dict[str, object]]] = {}
    exclusions: Counter[str] = Counter()
    representations: Counter[str] = Counter()
    details: dict[str, dict[str, object]] = {}
    built = resumed = 0
    source_code_sha256 = _source_code_sha256()

    def persist_progress() -> None:
        _atomic_json(options.out_root / "assets.json", index)
        _atomic_json(
            options.out_root / "summary.json",
            {
                "built": built,
                "resumed": resumed,
                "skipped": dict(exclusions),
                "fix_commit_representations": dict(representations),
                "tasks": details,
            },
        )

    for case in cases:
        source_root = options.templates_root / case.task_id
        provenance_path = source_root / "provenance.json"
        if not provenance_path.is_file():
            exclusions["not_extracted"] += 1
            details[case.task_id] = {"reason": "not_extracted"}
            persist_progress()
            continue
        provenance = _json_object(provenance_path)
        if provenance["image_digest"] != case.image_digest or provenance["task_id"] != case.task_id:
            raise ValueError("extracted case provenance does not match the baked case")
        repository = source_root / "extracted"
        hidden_tests = source_root / "hidden_tests"
        if not hidden_tests.is_dir():
            raise FileNotFoundError("extracted case hidden tests are missing")
        request_identity = _request_identity_payload(options, case, provenance)
        identity = _digest(request_identity)
        task_root = options.out_root / case.task_id
        if task_root.is_symlink():
            raise ValueError("task output may not be a symlink")
        arms = _completed_arms(task_root, identity, request_identity)
        if arms is not None:
            _runtime(options, case, repository)
            resumed += 1
        else:
            if shutil.disk_usage(options.out_root).free < options.minimum_free_bytes:
                raise OSError("free disk is below the configured asset-build floor")
            if not _venv_uses_uv_interpreter(repository):
                exclusions["system_interpreter"] += 1
                details[case.task_id] = {"reason": "system_interpreter"}
                persist_progress()
                continue
            spec = fix_commit_spec(case, repository)
            if git_text(repository, "status", "--porcelain", "--untracked-files=no"):
                exclusions["setup_edited_tracked_files"] += 1
                details[case.task_id] = {"reason": "setup_edited_tracked_files"}
                persist_progress()
                continue
            if git_text(repository, "rev-parse", "HEAD") != git_text(
                repository, "rev-parse", case.commit_hash + "^"
            ):
                raise ValueError("extracted repository HEAD is not the exact fix parent")
            if task_root.exists():
                if task_root.is_symlink():
                    raise ValueError("task output may not be a symlink")
                shutil.rmtree(task_root)
            arms = _build_case(options, case, task_root, repository, hidden_tests, spec)
            _assert_template_probe_states(arms)
            fingerprints = {
                path.relative_to(task_root).as_posix(): template_fingerprint(path)
                if path.is_dir()
                else hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(task_root.iterdir())
            }
            if _request_identity(options, case, provenance) != identity:
                raise ValueError("source inputs changed during task build; refusing completion")
            _atomic_json(
                task_root / "complete.json",
                {
                    "identity": identity,
                    "request_identity": request_identity,
                    "builder_version": BUILDER_VERSION,
                    "source_code_sha256": source_code_sha256,
                    "arms": arms,
                    "fingerprints": fingerprints,
                },
            )
            built += 1
        index[case.task_id] = arms
        fix = _json_object(Path(cast("str", arms[0]["manifest_path"])))["fix"]
        representation = cast("str", fix.get("fix_commit_representation", "exact"))
        representations[representation] += 1
        disk_bytes = _disk_bytes(task_root)
        details[case.task_id] = {
            "disk_bytes": disk_bytes,
            "fix_commit_representation": representation,
        }
        logger.info(
            "task %s: template_disk_bytes=%d (%0.3f GiB)",
            case.task_id,
            disk_bytes,
            disk_bytes / _GIB,
        )
        persist_progress()
    summary: dict[str, object] = {
        "built": built,
        "resumed": resumed,
        "skipped": dict(exclusions),
        "fix_commit_representations": dict(representations),
        "tasks": details,
    }
    _atomic_json(options.out_root / "assets.json", index)
    _atomic_json(options.out_root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Build a runtime namespace, optionally using a named channel split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--templates-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        required=True,
        help="runtime namespace; named channel split if --channel-splits is set",
    )
    parser.add_argument("--case-file", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--task-ids-file",
        type=Path,
        help="JSON selector with a canonical train or held-out partition and task_ids list",
    )
    parser.add_argument(
        "--arm",
        "--arms",
        dest="arms",
        action="append",
        help="repeat open, closed, or comma-separated channel names (default: open and closed)",
    )
    parser.add_argument("--channel-splits", type=Path)
    parser.add_argument("--interpreters-root", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=docker_backend.READ_ONLY_RUNTIME_ROOT)
    parser.add_argument(
        "--gate-record", type=Path, default=Path("artifacts/docker-isolation-gate.json")
    )
    parser.add_argument("--runtime-image")
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=40,
        help="free-disk floor in GiB checked before every task",
    )
    args = parser.parse_args(argv)
    options = BuildOptions(
        templates_root=args.templates_root.resolve(),
        out_root=args.out_root.resolve(),
        split=args.split,
        interpreters_root=(
            args.interpreters_root or args.templates_root.parent / "interpreters"
        ).resolve(),
        runtime_root=args.runtime_root.resolve(),
        gate_record=args.gate_record.resolve(),
        arms=tuple(args.arms or ("open", "closed")),
        channel_splits=args.channel_splits,
        minimum_free_bytes=int(args.min_free_disk_gb * _GIB),
        runtime_image=args.runtime_image,
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cases = _select_cases(load_tasks(args.case_file), args.task_ids_file)
    summary = build_assets(options, cases)
    logger.info(
        "asset build built=%s resumed=%s skipped=%s representations=%s",
        summary["built"],
        summary["resumed"],
        summary["skipped"],
        summary["fix_commit_representations"],
    )


if __name__ == "__main__":
    main()
