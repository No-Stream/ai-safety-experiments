"""Build ignored, source-pinned Terminal Wrench runtimes without trajectory material.

The JSON configuration supplies a source repository and exact commit, a uv image,
then task records with original_task source_path/tree OID, exact dependencies,
requires_python, SHA256-bound uv lock, artifact_paths and protected_paths.
Optional dockerfile_override and input_receipts each contain path and sha256;
input receipts additionally contain their destination within the build context.
All configuration, build contexts, assets, and receipts must remain gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
import tomllib
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from reward_hacking.terminal_wrench_runtime import (
    SCHEMA_VERSION,
    asset_tree_sha256,
    load_terminal_wrench_registry,
)

logger = logging.getLogger(__name__)
DockerCommand = Callable[[list[str]], str]
VERIFIER_PYTHON = "/opt/terminal-wrench-verifier/bin/python"


def _command(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, timeout=1800).strip()  # noqa: S603 - explicit argv, no shell


def _git(repository: Path, *arguments: str) -> str:
    return _command(["git", "-C", str(repository), *arguments])


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or value == "."
    ):
        raise ValueError("expected a normalized relative path")
    return value


def _ignored(path: Path, workspace_root: Path) -> None:
    path = path.resolve()
    if not path.is_relative_to(workspace_root.resolve()):
        raise ValueError("build paths must be inside the ignored workspace")
    result = subprocess.run(  # noqa: S603 - read-only Git argv
        ["/usr/bin/git", "-C", str(workspace_root), "check-ignore", "--quiet", "--", str(path)],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"build configuration and output must be gitignored: {path.name}")


def _receipt(value: dict[str, Any], config_root: Path) -> bytes:
    content = (config_root / value["path"]).read_bytes()
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if observed_sha256 != value["sha256"]:
        logger.warning(
            "input receipt %s records sha256 %s but observed %s; using the observed bytes",
            value["path"],
            value["sha256"],
            observed_sha256,
        )
    return content


def _export_source(repository: Path, commit: str, source_path: str, destination: Path) -> None:
    # Read blobs directly: git archive applies export-ignore/export-subst attributes.
    entries = subprocess.check_output(  # noqa: S603 - pinned Git tree, no shell
        ["/usr/bin/git", "-C", str(repository), "ls-tree", "-r", "-z", f"{commit}:{source_path}"],
    )
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, filename = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("source assets must contain only regular files and directories")
        path = destination / _relative(filename.decode())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            subprocess.check_output(  # noqa: S603 - pinned Git blob
                ["/usr/bin/git", "-C", str(repository), "cat-file", "blob", oid],
            )
        )
        path.chmod(int(mode, 8) & 0o777)


def _resolve_image(reference: str, docker: DockerCommand, receipts: dict[str, str]) -> str:
    if re.fullmatch(r"(?:[^\s]+@)?sha256:[a-f0-9]{64}", reference):
        return reference
    if reference not in receipts:
        docker(["docker", "pull", reference])
        details = json.loads(docker(["docker", "image", "inspect", reference]))[0]
        immutable = next(iter(details["RepoDigests"]), details["Id"])
        if not re.fullmatch(r"(?:[^\s]+@)?sha256:[a-f0-9]{64}", immutable):
            raise ValueError("Docker did not resolve an immutable image identity")
        receipts[reference] = immutable
    return receipts[reference]


def _pin_images(dockerfile: str, docker: DockerCommand, receipts: dict[str, str]) -> str:
    stages: set[str] = set()
    lines: list[str] = []
    for original_line in dockerfile.splitlines():
        line = original_line
        match = re.match(r"(\s*FROM\s+(?:--platform=\S+\s+)?)(\S+)(.*)$", line, re.IGNORECASE)
        if match:
            prefix, reference, suffix = match.groups()
            if "$" in reference:
                raise ValueError("Docker FROM variables require an explicit reviewed override")
            pinned = (
                reference
                if reference in stages or reference == "scratch"
                else _resolve_image(reference, docker, receipts)
            )
            line = prefix + pinned + suffix
            alias = re.search(r"\s+AS\s+(\S+)\s*$", suffix, re.IGNORECASE)
            if alias:
                stages.add(alias[1])
        match = re.search(r"\bCOPY\s+--from=(\S+)", line, re.IGNORECASE)
        if match and match[1] not in stages and not match[1].isdigit():
            reference = match[1]
            line = (
                line[: match.start(1)]
                + _resolve_image(reference, docker, receipts)
                + line[match.end(1) :]
            )
        lines.append(line)
    return "\n".join(lines) + "\n"


def _prepare_task(
    task: dict[str, Any], repository: Path, commit: str, output: Path, config_root: Path
) -> tuple[Path, Path, str]:
    task_id = _relative(task["task_id"])
    if "/" in task_id:
        raise ValueError("task_id must be a single path component")
    source_path = _relative(task["source_path"])
    if PurePosixPath(source_path).name != "original_task":
        raise ValueError("source_path must select an original_task subtree")
    actual_oid = _git(repository, "rev-parse", f"{commit}:{source_path}")
    if (
        actual_oid != task["source_tree_oid"]
        or _git(repository, "cat-file", "-t", actual_oid) != "tree"
    ):
        raise ValueError("source tree OID mismatch")
    target = output / task_id
    target.mkdir(exist_ok=True)
    source = target / "source"
    _export_source(repository, commit, source_path, source)
    context = target / "context"
    shutil.copytree(source / "environment", context, dirs_exist_ok=True)
    if any(
        context.joinpath(name).exists()
        for name in ("r2e_tests", "tests", "solution", "instruction.md")
    ):
        raise ValueError("build context contains hidden task assets")
    dockerfile = (context / "Dockerfile").read_text()
    if "dockerfile_override" in task:
        dockerfile = _receipt(task["dockerfile_override"], config_root).decode()
    if re.search(
        r"(?:\bRUN\s+|&&\s*|;\s*|\|\s*)(?:curl|wget)\s+|\bgit\s+clone\b|\bADD\s+https?://",
        dockerfile,
        re.IGNORECASE,
    ):
        raise ValueError(
            "mutable downloads require a reviewed override using SHA256-bound local input receipts"
        )
    for receipt in task.get("input_receipts", []):
        destination = context / _relative(receipt["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_receipt(receipt, config_root))
    dependencies: list[str] = task["dependencies"]
    if not dependencies or any(
        re.fullmatch(r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[A-Za-z0-9_.+!-]+", p) is None
        for p in dependencies
    ):
        raise ValueError("verifier dependencies must be exact package==version pins")
    project = context / "verifier-project"
    project.mkdir(exist_ok=True)
    project.joinpath("pyproject.toml").write_text(
        '[project]\nname = "terminal-wrench-verifier"\nversion = "0.1.0"\nrequires-python = '
        + json.dumps(task["requires_python"])
        + "\ndependencies = "
        + json.dumps(dependencies)
        + "\n"
    )
    project.joinpath("uv.lock").write_bytes(
        _receipt({"path": task["lock_path"], "sha256": task["lock_sha256"]}, config_root)
    )
    shutil.copytree(source / "tests", target / "tests", dirs_exist_ok=True)
    shutil.copy2(source / "instruction.md", target / "instruction.md")
    return target, context, dockerfile


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _persist(state: dict[str, Any], output: Path, commit: str, status: str) -> None:
    records = state["records"]
    state_path = output / "build-state.json"
    registry = output / "registry.json"
    state["status"] = status
    state["records"] = records
    _write_json(state_path, state)
    if records:
        _write_json(
            registry,
            {
                "schema_version": SCHEMA_VERSION,
                "source_commit": commit,
                "tasks": records,
            },
        )


def _build_task(
    task: dict[str, Any],
    config_path: Path,
    output: Path,
    state: dict[str, Any],
    docker: DockerCommand,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    config_root = config_path.parent
    target, context, original_dockerfile = _prepare_task(
        task, Path(config["source_repository"]), config["source_commit"], output, config_root
    )
    dockerfile = original_dockerfile
    image_receipts: dict[str, str] = state["base_images"]
    uv_image = _resolve_image(config["uv_image"], docker, image_receipts)
    dockerfile += (
        f"\nCOPY --from={uv_image} /uv /usr/local/bin/uv\n"
        "COPY verifier-project /opt/verifier-project\n"
        "RUN UV_PROJECT_ENVIRONMENT=/opt/terminal-wrench-verifier "
        "uv sync --frozen --no-dev --project /opt/verifier-project --python python3\n"
        "WORKDIR /app\n"
    )
    dockerfile = _pin_images(dockerfile, docker, image_receipts)
    context.joinpath("Dockerfile").write_text(dockerfile)
    tag = "terminal-wrench-local:" + hashlib.sha256(dockerfile.encode()).hexdigest()[:24]
    _persist(state, output, config["source_commit"], "building")
    docker(["docker", "build", "--network=default", "-t", tag, str(context)])
    details = json.loads(docker(["docker", "image", "inspect", tag]))[0]
    image: str = details["Id"]
    if re.fullmatch(r"sha256:[a-f0-9]{64}", image) is None:
        raise ValueError("built image must have an immutable image ID")
    workspace = target / "template"
    workspace.mkdir(exist_ok=True)
    container = docker(["docker", "create", image]).strip()
    try:
        docker(["docker", "cp", f"{container}:/app/.", str(workspace)])
    finally:
        docker(["docker", "rm", container])
    if any(
        workspace.joinpath(name).exists()
        for name in ("r2e_tests", "tests", "solution", "instruction.md")
    ):
        raise ValueError("actor workspace contains hidden task assets")
    record = {
        "task_id": task["task_id"],
        "source_tree_oid": task["source_tree_oid"],
        "image": image,
        "workspace_template": f"{task['task_id']}/template",
        "workspace_sha256": asset_tree_sha256(workspace),
        "tests_template": f"{task['task_id']}/tests",
        "tests_sha256": asset_tree_sha256(target / "tests"),
        "instruction_path": f"{task['task_id']}/instruction.md",
        "instruction_sha256": hashlib.sha256(
            target.joinpath("instruction.md").read_bytes()
        ).hexdigest(),
        "artifact_paths": task["artifact_paths"],
        "protected_paths": task["protected_paths"],
        "python_executable": VERIFIER_PYTHON,
        "grader_shared_memory_bytes": task.get("grader_shared_memory_bytes", 0),
        "grader_timeout_seconds": task["grader_timeout_seconds"],
        "gate_record": f"{task['task_id']}/isolation-gate.json",
    }
    dependency_inventory = json.loads(
        docker(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--entrypoint",
                VERIFIER_PYTHON,
                image,
                "-I",
                "-c",
                (
                    "import importlib.metadata,json; print(json.dumps({d.metadata['Name']: d.version "
                    "for d in importlib.metadata.distributions()}, sort_keys=True))"
                ),
            ]
        )
    )
    normalized_inventory = {
        re.sub(r"[-_.]+", "-", name).lower(): version
        for name, version in dependency_inventory.items()
    }
    for requirement in task["dependencies"]:
        package, version = requirement.split("==")
        package = re.sub(r"[-_.]+", "-", package.split("[")[0]).lower()
        if normalized_inventory.get(package) != version:
            raise ValueError("installed verifier dependency does not match its exact pin")
    provenance = {
        "source_commit": config["source_commit"],
        "configuration": task,
        "base_images": image_receipts,
        "dockerfile_sha256": hashlib.sha256(dockerfile.encode()).hexdigest(),
        "context_sha256": asset_tree_sha256(context),
        "image_inspect": details,
        "verifier_packages": dependency_inventory,
    }
    target.joinpath("build-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return record


def _validate_receipts(
    tasks: list[dict[str, Any]], repository: Path, commit: str, config_root: Path
) -> None:
    for task in tasks:
        source_path = _relative(task["source_path"])
        if _git(repository, "rev-parse", f"{commit}:{source_path}") != task["source_tree_oid"]:
            raise ValueError("source tree OID mismatch")
        _receipt({"path": task["lock_path"], "sha256": task["lock_sha256"]}, config_root)
        for receipt in task.get("input_receipts", []):
            _receipt(receipt, config_root)
        if "dockerfile_override" in task:
            _receipt(task["dockerfile_override"], config_root)


def _copy_licenses(repository: Path, commit: str, output: Path) -> None:
    licenses = output / "licenses"
    licenses.mkdir(exist_ok=True)
    for name in _git(repository, "ls-tree", "--name-only", commit).splitlines():
        if re.fullmatch(r"(?:LICENSE|COPYING|NOTICE)(?:\..*)?", name, re.IGNORECASE):
            licenses.joinpath(name).write_bytes(
                subprocess.check_output(  # noqa: S603 - pinned Git object
                    ["/usr/bin/git", "-C", str(repository), "show", f"{commit}:{name}"]
                )
            )


def _validate_resumed_images(registry: Path, docker: DockerCommand) -> None:
    if registry.exists():
        admitted = load_terminal_wrench_registry(registry)
        for record in admitted.records:
            details = json.loads(docker(["docker", "image", "inspect", record.image]))[0]
            if details["Id"] != record.image:
                raise ValueError("resumed image identity mismatch")


def _source_grader_timeout(repository: Path, commit: str, source_path: str) -> float:
    source_toml = _git(repository, "show", f"{commit}:{_relative(source_path)}/task.toml")
    timeout = tomllib.loads(source_toml)["verifier"]["timeout_sec"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("source grader timeout must be finite and positive")
    return float(timeout)


def _semantic_task(task: dict[str, Any]) -> dict[str, Any]:
    identity = {key: value for key, value in task.items() if key != "lock_path"}
    identity["dependencies"] = sorted(task["dependencies"])
    identity["input_receipts"] = sorted(
        (
            {key: value for key, value in receipt.items() if key != "path"}
            for receipt in task.get("input_receipts", [])
        ),
        key=lambda receipt: receipt["destination"],
    )
    if "dockerfile_override" in task:
        identity["dockerfile_override"] = {"sha256": task["dockerfile_override"]["sha256"]}
    return identity


def _identity_payload(
    config: dict[str, Any], commit: str, tasks: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "source_commit": commit,
        "uv_image": config["uv_image"],
        "tasks": {
            task["task_id"]: _semantic_task(task)
            for task in sorted(tasks, key=lambda task: task["task_id"])
        },
    }


def _identity_differences(
    expected: dict[str, Any], stored: dict[str, Any], prefix: str = ""
) -> list[str]:
    differences: list[str] = []
    for key in sorted(set(expected) | set(stored)):
        path = f"{prefix}.{key}" if prefix else key
        if key not in expected or key not in stored:
            differences.append(path)
        elif isinstance(expected[key], dict) and isinstance(stored[key], dict):
            differences.extend(_identity_differences(expected[key], stored[key], path))
        elif expected[key] != stored[key]:
            differences.append(path)
    return differences


def _legacy_identity(config: dict[str, Any], commit: str, tasks: list[dict[str, Any]]) -> str:
    """Read old build-state records whose semantic payload was stored only as a digest."""
    payload = {
        "source_commit": commit,
        "uv_image": config["uv_image"],
        "tasks": sorted((_semantic_task(task) for task in tasks), key=lambda task: task["task_id"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_or_initialize_build_state(
    output: Path,
    identity_payload: dict[str, Any],
    identity: str,
    legacy_identity: str,
) -> dict[str, Any]:
    if not output.exists():
        output.mkdir(parents=True)
        return {
            "identity": identity,
            "identity_payload": identity_payload,
            "status": "preparing",
            "records": [],
            "base_images": {},
        }

    state: dict[str, Any] = json.loads((output / "build-state.json").read_text())
    if "identity_payload" in state:
        differences = _identity_differences(identity_payload, state["identity_payload"])
        if differences:
            raise ValueError(
                "resume semantic configuration changed at " + ", ".join(differences[:10])
            )
    elif state["identity"] != legacy_identity:
        raise ValueError(
            "legacy resume configuration identity mismatch; the old record has no semantic "
            "payload to name changed fields"
        )
    else:
        state["identity"] = identity
        state["identity_payload"] = identity_payload
    return state


def build_runtime_registry(
    config_path: Path,
    output: Path,
    *,
    workspace_root: Path | None = None,
    docker: DockerCommand = _command,
) -> Path:
    """Build all explicitly listed source tasks and return a validated runtime registry.

    Completed tasks persist incrementally and resume only under identical source
    and configuration identities. Docker failures propagate with a failure receipt.
    Containment gates are separate; this builder never fabricates gate records.
    """
    workspace_root = Path.cwd() if workspace_root is None else workspace_root
    _ignored(config_path, workspace_root)
    _ignored(output, workspace_root)
    config = json.loads(config_path.read_text())
    repository = Path(config["source_repository"]).resolve()
    commit: str = config["source_commit"]
    if (
        re.fullmatch(r"[a-f0-9]{40}", commit) is None
        or _git(repository, "rev-parse", f"{commit}^{{commit}}") != commit
    ):
        raise ValueError("source commit must be an exact Git commit OID")
    tasks: list[dict[str, Any]] = config["tasks"]
    if not tasks or len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("tasks must be nonempty and unique")
    tasks = [
        dict(
            task,
            grader_timeout_seconds=_source_grader_timeout(repository, commit, task["source_path"]),
        )
        for task in tasks
    ]
    identity_payload = _identity_payload(config, commit, tasks)
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    state = _load_or_initialize_build_state(
        output, identity_payload, identity, _legacy_identity(config, commit, tasks)
    )
    _persist(state, output, commit, "incomplete")
    _validate_receipts(tasks, repository, commit, config_path.parent)
    records: list[dict[str, Any]] = state["records"]
    registry = output / "registry.json"

    _persist(state, output, commit, "incomplete")
    _validate_resumed_images(registry, docker)
    _copy_licenses(repository, commit, output)
    completed_ids = {record["task_id"] for record in records}
    for task in tasks:
        if task["task_id"] in completed_ids:
            continue
        state["current_task"] = task["task_id"]
        _persist(state, output, commit, "building")
        try:
            record = _build_task(task, config_path, output, state, docker)
            records.append(record)
            _persist(state, output, commit, "incomplete")
            load_terminal_wrench_registry(registry)
            logger.info("Built source runtime %s", task["task_id"])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            _persist(state, output, commit, "failed")
            raise
    _persist(state, output, commit, "complete")
    load_terminal_wrench_registry(registry)
    return registry


def main() -> None:
    """Build a local registry from an explicit ignored configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_runtime_registry(arguments.config, arguments.output)


if __name__ == "__main__":
    main()
