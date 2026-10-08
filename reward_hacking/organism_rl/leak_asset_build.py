"""Build local, resumable leak assets; no benchmark payload belongs in the code repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from reward_hacking.harness import docker_backend
from reward_hacking.harness.leak_channels import (
    ALL_COMPATIBLE_CHANNELS,
    BUILDER_VERSION,
    CHANNELS,
    BuildRequest,
    build_channel_templates,
    template_fingerprint,
)
from reward_hacking.harness.leak_fix_spec import (
    MergeFixCommitError,
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


def _request_identity(options: BuildOptions, case: R2eLeakTask, provenance: dict[str, Any]) -> str:
    return _digest(
        {
            "builder": BUILDER_VERSION,
            "options": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(options).items()
            },
            "case": asdict(case),
            "extraction": provenance,
            "source_fingerprint": template_fingerprint(
                options.templates_root / case.task_id / "extracted"
            ),
            "hidden_tests_fingerprint": template_fingerprint(
                options.templates_root / case.task_id / "hidden_tests"
            ),
            "gate_sha256": hashlib.sha256(options.gate_record.read_bytes()).hexdigest(),
            "channel_config": None
            if options.channel_splits is None
            else _json_object(options.channel_splits),
        }
    )


def _completed_arms(task_root: Path, identity: str) -> list[dict[str, object]] | None:
    marker_path = task_root / "complete.json"
    if not marker_path.exists():
        return None
    marker = _json_object(marker_path)
    if marker["identity"] != identity:
        raise ValueError("task completion identity changed; choose a fresh output root")
    for relative, expected in marker["fingerprints"].items():
        path = task_root / relative
        actual = (
            template_fingerprint(path)
            if path.is_dir()
            else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        if actual != expected:
            raise ValueError("task completion fingerprints changed; choose a fresh output root")
    return cast("list[dict[str, object]]", marker["arms"])


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
                )
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
                paths[channels] = (output / "open", output / "manifest.json")
            else:
                shutil.rmtree(output)
        for channels in channel_sets:
            workspace, manifest_path = (
                paths[channels] if channels else (common_closed, task_root / "closed-manifest.json")
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
    details: dict[str, dict[str, object]] = {}
    built = resumed = 0

    def persist_progress() -> None:
        _atomic_json(options.out_root / "assets.json", index)
        _atomic_json(
            options.out_root / "summary.json",
            {
                "built": built,
                "resumed": resumed,
                "skipped": dict(exclusions),
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
        identity = _request_identity(options, case, provenance)
        task_root = options.out_root / case.task_id
        if task_root.is_symlink():
            raise ValueError("task output may not be a symlink")
        arms = _completed_arms(task_root, identity)
        if arms is not None:
            _runtime(options, case, repository)
            resumed += 1
        else:
            if shutil.disk_usage(options.out_root).free < options.minimum_free_bytes:
                raise OSError("free disk is below the configured asset-build floor")
            try:
                spec = fix_commit_spec(case, repository)
            except MergeFixCommitError:
                exclusions["merge_commit"] += 1
                details[case.task_id] = {"reason": "merge_commit"}
                persist_progress()
                continue
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
                {"identity": identity, "arms": arms, "fingerprints": fingerprints},
            )
            built += 1
        index[case.task_id] = arms
        disk_bytes = _disk_bytes(task_root)
        details[case.task_id] = {"disk_bytes": disk_bytes}
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
    summary = build_assets(options, load_tasks(args.case_file))
    logger.info(
        "asset build built=%s resumed=%s skipped=%s",
        summary["built"],
        summary["resumed"],
        summary["skipped"],
    )
