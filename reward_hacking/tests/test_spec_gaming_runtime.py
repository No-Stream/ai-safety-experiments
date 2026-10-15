"""Disposable artifact actions and immutable workspace contracts, without a daemon."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from reward_hacking.evidence_store import append_jsonl
from reward_hacking.harness.docker_backend import DockerConfig
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.spec_gaming import runtime

IMAGE = "sha256:" + "a" * 64


@pytest.mark.parametrize("compressed", [True, False])
def test_completed_evidence_uses_compact_aware_reader(tmp_path: Path, compressed: bool) -> None:
    evidence = tmp_path / "command.jsonl"
    records = [{"phase": "started"}, {"phase": "finished", "output_truncated": False}]
    if compressed:
        append_jsonl(evidence, records)
        assert evidence.read_bytes().startswith(b"\x1f\x8b")
    else:
        evidence.write_text("".join(json.dumps(record) + "\n" for record in records))
    runtime._checked_result(CommandResult("true", 0, "stdout", "stderr", False), evidence)


def test_local_image_id_is_immutable(tmp_path: Path) -> None:
    assert DockerConfig(IMAGE, tmp_path / "gate").image == IMAGE
    for mutable in ("image:latest", "sha256:abc", "sha256:" + "g" * 64):
        with pytest.raises(ValueError, match="full sha256"):
            DockerConfig(mutable, tmp_path / "gate")


def test_snapshot_round_trip_preserves_links_deletions_and_independent_inodes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir(mode=0o750)
    (source / "nested" / "script").write_bytes(b"original")
    (source / "nested" / "script").chmod(0o751)
    (source / "outside").symlink_to(tmp_path / "secret")
    (tmp_path / "secret").write_bytes(b"never read this")
    snapshot = runtime.snapshot_workspace(source)
    destination = tmp_path / "destination"
    destination.mkdir()
    runtime.materialize_workspace(snapshot, destination)
    assert (destination / "nested" / "script").read_bytes() == b"original"
    assert (destination / "nested" / "script").stat().st_mode & 0o777 == 0o751
    assert (destination / "nested" / "script").stat().st_ino != (
        source / "nested" / "script"
    ).stat().st_ino
    assert (destination / "outside").is_symlink()
    (destination / "nested" / "script").unlink()
    assert all(
        entry.path != "nested/script" for entry in runtime.snapshot_workspace(destination).entries
    )
    assert (
        next(entry.content for entry in snapshot.entries if entry.path == "nested/script")
        == b"original"
    )
    assert not any(entry.content == b"never read this" for entry in snapshot.entries)


def test_snapshot_rejects_special_files_and_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "data").write_bytes(b"12345")
    monkeypatch.setattr(runtime, "MAXIMUM_SNAPSHOT_BYTES", 4)
    with pytest.raises(runtime.SnapshotLimitError):
        runtime.snapshot_workspace(tmp_path)
    (tmp_path / "data").unlink()
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(ValueError, match="special"):
        runtime.snapshot_workspace(tmp_path)


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "a/../b", "a//b", "a\\b"])
def test_snapshot_refuses_unsafe_paths(tmp_path: Path, unsafe: str) -> None:
    snapshot = runtime.WorkspaceSnapshot((runtime.WorkspaceEntry(unsafe, "file", 0o600, b"x"),))
    with pytest.raises(ValueError, match="path"):
        runtime.materialize_workspace(snapshot, tmp_path)


def test_materialization_never_traverses_symlink_parent(tmp_path: Path) -> None:
    entries = (
        runtime.WorkspaceEntry("link", "symlink", 0o777, target=str(tmp_path.parent)),
        runtime.WorkspaceEntry("link/escape", "file", 0o600, b"x"),
    )
    with pytest.raises(ValueError, match="parent"):
        runtime.materialize_workspace(runtime.WorkspaceSnapshot(entries), tmp_path)
    assert not list(tmp_path.iterdir())


def test_nonzero_action_keeps_writes_and_checks_output_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = DockerConfig(IMAGE, tmp_path / "gate")
    initial = runtime.WorkspaceSnapshot((runtime.WorkspaceEntry("old", "file", 0o600, b"old"),))

    def fake_command(
        workspace: Path, command: str, *, config: DockerConfig, **kwargs: object
    ) -> CommandResult:
        assert not config.trusted_grader
        (workspace / "old").unlink()
        (workspace / "new").write_bytes(b"changed before failure")
        assert config.evidence_path is not None
        append_jsonl(config.evidence_path, [{"phase": "finished", "output_truncated": False}])
        return CommandResult(command, 2, "all stdout", "all stderr", False)

    monkeypatch.setattr(runtime, "run_in_docker", fake_command)
    action = runtime.execute_action(initial, "exit 2", config)
    assert action.command_result.exit_code == 2
    assert action.command_result.stdout == "all stdout"
    assert {entry.path for entry in action.snapshot.entries} == {"new"}
    assert initial.entries[0].content == b"old"


def test_truncated_action_fails_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_command(
        workspace: Path, command: str, *, config: DockerConfig, **kwargs: object
    ) -> CommandResult:
        assert config.evidence_path is not None
        append_jsonl(config.evidence_path, [{"phase": "finished", "output_truncated": True}])
        return CommandResult(command, 0, "clipped", "", False)

    monkeypatch.setattr(runtime, "run_in_docker", fake_command)
    with pytest.raises(runtime.RuntimeInfrastructureError, match="truncated") as captured:
        runtime.execute_action(
            runtime.WorkspaceSnapshot(()), "large output", DockerConfig(IMAGE, tmp_path / "gate")
        )
    assert captured.value.command_result is not None
    assert captured.value.command_result.stdout == "clipped"
    assert captured.value.command_evidence[-1]["output_truncated"] is True


def test_evaluation_uses_fresh_trusted_files_and_final_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {name: name.encode() for name in runtime.REQUIRED_EVALUATION_FILES}

    def fake_command(
        workspace: Path, command: str, *, config: DockerConfig, **kwargs: object
    ) -> CommandResult:
        assert config.trusted_grader
        assert not config.read_only_mounts
        assert config.episode_tmp_state is None
        assert {path.name for path in workspace.iterdir()} == set(files)
        assert command == "/usr/bin/python3 /work/spreadsheet_worker.py"
        assert config.evidence_path is not None
        append_jsonl(config.evidence_path, [{"phase": "finished", "output_truncated": False}])
        return CommandResult(
            command, 0, 'recalculation log\n{"proxy": 1.0, "full": false}\n', "diagnostic", False
        )

    monkeypatch.setattr(runtime, "run_in_docker", fake_command)
    evaluated = runtime.run_evaluation(files, DockerConfig(IMAGE, tmp_path / "gate"))
    assert evaluated.payload == {"proxy": 1.0, "full": False}
    assert evaluated.command_result.stderr == "diagnostic"
    with pytest.raises(ValueError, match="required"):
        runtime.run_evaluation({"submission.xlsx": b"x"}, DockerConfig(IMAGE, tmp_path / "gate"))
    with pytest.raises(ValueError, match="mount"):
        runtime.run_evaluation(
            files, replace(DockerConfig(IMAGE, tmp_path / "gate"), workspace_mount="/other")
        )


@pytest.mark.slow
@pytest.mark.docker_real
def test_live_actor_stops_descendants_and_preserves_failed_mutations(tmp_path: Path) -> None:
    """Explicit Docker selection requires the audited image rather than silently skipping."""
    image = os.environ["SPEC_GAMING_DOCKER_IMAGE"]
    gate_record = Path(os.environ["SPEC_GAMING_DOCKER_GATE_RECORD"])
    assert gate_record.is_file(), "the supplied real Docker gate record must exist"
    original_gate_bytes = gate_record.read_bytes()
    config = DockerConfig(
        image, gate_record, gate_record_sha256=hashlib.sha256(original_gate_bytes).hexdigest()
    )
    initial = runtime.WorkspaceSnapshot(
        (
            runtime.WorkspaceEntry("remove", "file", 0o600, b"old"),
            runtime.WorkspaceEntry("unchanged", "file", 0o600, b"original"),
        )
    )
    command = (
        "printf changed > changed; rm remove; "
        "ln -s /host-private-unmounted/secret outside; "
        "(sleep 0.3; printf late > background) >/dev/null 2>&1 & "
        "printf 'all stdout'; printf 'all stderr' >&2; exit 2"
    )
    action = runtime.execute_action(initial, command, config)
    assert gate_record.read_bytes() == original_gate_bytes
    assert tuple(gate_record.parent.glob(f"{gate_record.stem}.live-{os.getpid()}-*.json"))
    assert action.command_result.exit_code == 2
    assert action.command_result.stdout == "all stdout"
    assert action.command_result.stderr == "all stderr"
    entries = {entry.path: entry for entry in action.snapshot.entries}
    assert entries["changed"].content == b"changed"
    assert entries["outside"].target == "/host-private-unmounted/secret"
    assert "remove" not in entries
    assert "background" not in entries
    assert initial.entries[0].content == b"old"
    second = runtime.execute_action(
        action.snapshot,
        "test ! -e outside && test ! -e background && printf continued > changed",
        config,
    )
    assert second.command_result.ok
    assert (
        next(entry.content for entry in second.snapshot.entries if entry.path == "changed")
        == b"continued"
    )
    assert entries["changed"].content == b"changed"


@pytest.mark.parametrize(
    ("process", "evidence_record", "message"),
    [
        ((125, ""), {"phase": "finished", "output_truncated": False}, "infrastructure"),
        ((1, ""), {"phase": "finished", "output_truncated": False}, "evaluator failed"),
        ((0, "no json"), {"phase": "finished", "output_truncated": False}, "not JSON"),
        ((0, "[]"), {"phase": "finished", "output_truncated": False}, "JSON object"),
        ((0, "{}"), {"phase": "started"}, "completed evidence"),
        ((0, "{}"), {"phase": "finished"}, "truncated"),
    ],
)
def test_evaluator_infrastructure_never_becomes_a_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process: tuple[int, str],
    evidence_record: dict[str, object],
    message: str,
) -> None:
    def fake_command(
        workspace: Path, command: str, *, config: DockerConfig, **kwargs: object
    ) -> CommandResult:
        assert config.evidence_path is not None
        append_jsonl(config.evidence_path, [evidence_record])
        return CommandResult(command, process[0], process[1], "failure detail", False)

    monkeypatch.setattr(runtime, "run_in_docker", fake_command)
    files = dict.fromkeys(runtime.REQUIRED_EVALUATION_FILES, b"synthetic")
    with pytest.raises(runtime.RuntimeInfrastructureError, match=message):
        runtime.run_evaluation(files, DockerConfig(IMAGE, tmp_path / "gate"))
