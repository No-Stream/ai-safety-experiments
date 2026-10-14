"""CPU contracts for the Docker command boundary; no daemon required."""

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import pytest

from reward_hacking.evidence_store import guard_episode_size, read_jsonl
from reward_hacking.harness import docker_backend as backend
from reward_hacking.harness.docker_backend import (
    REQUIRED_PROBES,
    DockerConfig,
    SnapshotLimitError,
    _run_bounded_docker,
    container_cpuset,
    copy_workspace_template,
    docker_run_args,
    require_gate,
    restore_workspace_directory_times,
    run_docker_grader,
    run_in_docker,
    submitted_files,
    workspace_snapshot,
)
from reward_hacking.ladder.loop import _restore_shell_state, _shell_wrapper, _ShellState
from scripts import run_docker_jail_tests

IMAGE = "python@sha256:" + "a" * 64


def test_network_cases_use_local_tcp_and_dns_controls(tmp_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="docker-network-contract-") as workspace_name:
        runner = run_docker_jail_tests.DockerProbeRunner(
            DockerConfig(IMAGE, tmp_path / "gate"), Path(workspace_name)
        )
        cases = run_docker_jail_tests._network_probe_cases(
            runner, "172.17.0.1", 41001, 41002, "172.17.0.2"
        )
    assert {case.name for case in cases} == {
        "network_dns_blocked",
        "network_public_tcp_blocked",
        "network_gateway_tcp_blocked",
    }
    assert all(
        "example.com" not in case.command and "1.1.1.1" not in case.command for case in cases
    )
    assert "41002" in cases[0].command
    assert "172.17.0.2" in cases[1].command
    assert "8080" in cases[1].command
    assert "41001" in cases[2].command
    assert all("--network" in case.weakened_argv for case in cases)


@pytest.mark.slow
@pytest.mark.docker_real
def test_live_network_controls_fail_when_hardened_network_is_weakened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _local_python_image_or_skip()
    with tempfile.TemporaryDirectory(prefix="docker-network-live-") as workspace_name:
        workspace = Path(workspace_name)
        service_workspace = workspace / "service"
        service_workspace.mkdir()
        runner = run_docker_jail_tests.DockerProbeRunner(
            DockerConfig(image, tmp_path / "gate"), workspace
        )
        gateway = run_docker_jail_tests._docker_gateway(image)
        with run_docker_jail_tests.serving_gateway_control(runner, service_workspace, gateway) as (
            service_ip,
            tcp_port,
            dns_port,
        ):
            cases = run_docker_jail_tests._network_probe_cases(
                runner, gateway, tcp_port, dns_port, service_ip
            )
            original_run_argv = run_docker_jail_tests.DockerProbeRunner.run_argv

            def weakened_hardened_run(
                self: run_docker_jail_tests.DockerProbeRunner,
                argv: list[str],
                name: str,
                timeout_seconds: int = 30,
            ) -> run_docker_jail_tests.ContainerResult:
                if "--network" in argv and argv[argv.index("--network") + 1] == "none":
                    argv = run_docker_jail_tests.replace_option_value(
                        argv, "--network", "none", "bridge"
                    )
                return original_run_argv(self, argv, name, timeout_seconds)

            monkeypatch.setattr(
                run_docker_jail_tests.DockerProbeRunner, "run_argv", weakened_hardened_run
            )
            for case in cases:
                with pytest.raises(
                    RuntimeError, match=rf"{case.name} hardened run expected 'blocked'"
                ):
                    run_docker_jail_tests._run_probe_pair(runner, case)

            monkeypatch.setattr(
                run_docker_jail_tests.DockerProbeRunner, "run_argv", original_run_argv
            )
            for case in cases:
                assert run_docker_jail_tests._run_probe_pair(runner, case) == (
                    run_docker_jail_tests.ProbeResult(contained=True, negative_control=True)
                )


def _local_python_image_or_skip() -> str:
    docker_executable = shutil.which("docker")
    if docker_executable is None:
        pytest.skip("Docker executable is unavailable")
    try:
        daemon = subprocess.run(  # noqa: S603 - executable is resolved from the host PATH
            [docker_executable, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("Docker daemon did not respond")
    if daemon.returncode != 0:
        pytest.skip("Docker daemon is unavailable")
    inspected = subprocess.run(  # noqa: S603 - executable is resolved from the host PATH
        [
            docker_executable,
            "image",
            "inspect",
            "--format",
            "{{json .RepoDigests}}",
            "python:3.12-slim",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if inspected.returncode != 0:
        pytest.skip("python:3.12-slim is not available locally")
    repository_digests = json.loads(inspected.stdout)
    if not isinstance(repository_digests, list):
        pytest.skip("python:3.12-slim does not expose repository digests")
    python_digests = [
        digest
        for digest in repository_digests
        if isinstance(digest, str) and digest.startswith("python@sha256:")
    ]
    if not python_digests:
        pytest.skip("python:3.12-slim does not expose a pinned python image digest")
    # Upstream re-publishes the tag with identical content, so one local image can carry several
    # repository digests; each names the same image.
    return min(python_digests)


def test_hardened_flags_and_digest(tmp_path: Path) -> None:
    config = DockerConfig(image=IMAGE, gate_record=tmp_path / "gate.json")
    argv = docker_run_args(config, tmp_path, "episode-toy", "id")
    for flag, value in (
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--pids-limit", "64"),
        ("--memory", "268435456"),
        ("--memory-swap", "268435456"),
        ("--cpus", "1"),
    ):
        assert argv[argv.index(flag) + 1] == value
    assert argv[argv.index("--cpuset-cpus") + 1] == container_cpuset("episode-toy", 1)
    assert "--read-only" in argv
    assert argv[argv.index("--log-driver") + 1] == "none"
    assert "--rm" in argv
    assert argv.count("--mount") == 1
    assert argv[argv.index("--mount") + 1].endswith(",dst=/work,bind-recursive=disabled")
    assert argv[argv.index("--workdir") + 1] == "/work"
    assert "/var/run/docker.sock" not in " ".join(argv)
    assert IMAGE in argv
    assert argv[argv.index("--user") + 1].split(":")[0] != "0"


def test_episode_tmp_state_is_restored_and_saved_inside_bounded_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "episode"
    workspace.mkdir()
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / "saved.txt").write_text("previous command")
    config = DockerConfig(IMAGE, tmp_path / "gate.json", episode_tmp_state=state_root)
    monkeypatch.setattr(backend, "require_gate", lambda _config: None)
    monkeypatch.setattr(backend, "_inspect_image_volumes", lambda *_args, **_kwargs: (None, None))
    observed: list[list[str]] = []

    def run_docker(argv: list[str], _deadline: int) -> tuple[int, str, str, bool, bool]:
        observed.append(argv)
        return 0, "ok", "", False, False

    monkeypatch.setattr(backend, "_run_bounded_docker", run_docker)
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
    )

    result = run_in_docker(workspace, "cat /tmp/saved.txt", config=config)

    assert result.exit_code == 0
    assert (state_root / "saved.txt").read_text() == "previous command"
    assert not any(workspace.glob(".episode-tmp-*"))
    argv = observed[0]
    assert "--tmpfs" in argv
    assert "size=67108864" in argv[argv.index("--tmpfs") + 1]
    assert "/usr/bin/timeout --signal=KILL 30s /bin/bash -lc" in argv[-1]
    assert "/bin/cp -a" in argv[-1]


def test_episode_tmp_state_with_read_only_directories_is_replaced_and_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "episode"
    workspace.mkdir()
    state_root = tmp_path / "state"
    previous = state_root / "templates" / "project"
    previous.mkdir(parents=True)
    (previous / "old.txt").write_text("from an earlier command")
    previous.chmod(0o555)
    config = DockerConfig(IMAGE, tmp_path / "gate.json", episode_tmp_state=state_root)
    monkeypatch.setattr(backend, "require_gate", lambda _config: None)
    monkeypatch.setattr(backend, "_inspect_image_volumes", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
    )

    def write_read_only_tree(argv: list[str], _deadline: int) -> tuple[int, str, str, bool, bool]:
        stage = next(workspace.glob(".episode-tmp-*"))
        current = stage / "copied" / "tree"
        current.mkdir(parents=True)
        (current / "new.txt").write_text("from this command")
        current.chmod(0o555)
        return 0, "", "", False, False

    monkeypatch.setattr(backend, "_run_bounded_docker", write_read_only_tree)

    run_in_docker(workspace, "cp -r /src /tmp/copied", config=config)

    assert (state_root / "copied" / "tree" / "new.txt").read_text() == "from this command"
    assert not any(workspace.glob(".episode-tmp-*"))


def test_episode_tmp_state_rejects_symlink_and_oversized_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "episode"
    workspace.mkdir()
    state_root = tmp_path / "state"
    state_root.mkdir()
    config = DockerConfig(IMAGE, tmp_path / "gate.json", episode_tmp_state=state_root)
    monkeypatch.setattr(backend, "require_gate", lambda _config: None)
    monkeypatch.setattr(backend, "_inspect_image_volumes", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
    )

    def replace_stage_with_symlink(
        argv: list[str], _deadline: int
    ) -> tuple[int, str, str, bool, bool]:
        stage = next(workspace.glob(".episode-tmp-*"))
        shutil.rmtree(stage)
        stage.symlink_to(state_root, target_is_directory=True)
        return 0, "", "", False, False

    monkeypatch.setattr(backend, "_run_bounded_docker", replace_stage_with_symlink)
    with pytest.raises(ValueError, match="symlink"):
        run_in_docker(workspace, "true", config=config)
    assert state_root.is_dir()

    (state_root / "too-large").write_bytes(b"x")
    monkeypatch.setattr(backend, "MAXIMUM_TMP_STATE_BYTES", 0)
    with pytest.raises(SnapshotLimitError, match="tmpfs byte limit"):
        run_in_docker(workspace, "true", config=config)


def test_episode_tmp_wrapper_preserves_files_after_failed_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "episode"
    workspace.mkdir()
    state_root = tmp_path / "state"
    state_root.mkdir()
    container_tmp = tmp_path / "container-tmp"
    config = DockerConfig(IMAGE, tmp_path / "gate.json", episode_tmp_state=state_root)
    original_run = subprocess.run
    monkeypatch.setattr(backend, "require_gate", lambda _config: None)
    monkeypatch.setattr(backend, "_inspect_image_volumes", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
    )

    def run_shell(argv: list[str], _deadline: int) -> tuple[int, str, str, bool, bool]:
        if container_tmp.exists():
            shutil.rmtree(container_tmp)
        container_tmp.mkdir()
        script = argv[-1].replace("/tmp/", f"{container_tmp}/").replace("/work/", f"{workspace}/")  # noqa: S108 - synthetic container path
        result = original_run(
            ["/bin/bash", "-c", script],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr, False, False

    monkeypatch.setattr(backend, "_run_bounded_docker", run_shell)

    assert (
        run_in_docker(
            workspace, "printf persisted > /tmp/note", config=config, login_shell=False
        ).exit_code
        == 0
    )
    assert run_in_docker(workspace, "false", config=config, login_shell=False).exit_code == 1
    final = run_in_docker(workspace, "cat /tmp/note", config=config, login_shell=False)
    assert final.exit_code == 0, final.stderr
    assert final.stdout == "persisted"
    assert (state_root / "note").read_text() == "persisted"
    timed_out = run_in_docker(
        workspace,
        "printf late > /tmp/late; sleep 2",
        config=config,
        login_shell=False,
        timeout="1s",
    )
    assert timed_out.timed_out
    assert (
        run_in_docker(workspace, "cat /tmp/late", config=config, login_shell=False).stdout == "late"
    )
    sentinel = tmp_path / "host-sentinel"
    sentinel.write_text("untouched")
    assert (
        run_in_docker(
            workspace,
            f"ln -s {shlex.quote(str(sentinel))} /tmp/link",
            config=config,
            login_shell=False,
        ).exit_code
        == 0
    )
    assert (state_root / "link").is_symlink()
    assert sentinel.read_text() == "untouched"
    assert not any(workspace.glob(".episode-tmp-*"))


def test_trusted_grader_has_bounded_process_and_thread_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root = tmp_path / "policy-tmp"
    state_root.mkdir()
    config = DockerConfig(
        image=IMAGE, gate_record=tmp_path / "gate.json", episode_tmp_state=state_root
    )
    agent_argv = docker_run_args(config, tmp_path, "episode-toy", "id")
    assert agent_argv[agent_argv.index("--pids-limit") + 1] == "64"
    assert not any(argument.endswith("_NUM_THREADS=1") for argument in agent_argv)

    observed: dict[str, list[str]] = {}

    def capture_grader_profile(
        workspace: Path,
        command: str,
        *,
        config: DockerConfig,
        **_options: object,
    ) -> backend.CommandResult:
        assert config.trusted_grader
        assert config.episode_tmp_state is None
        observed["argv"] = docker_run_args(config, workspace, "trusted-grade", command)
        return backend.CommandResult(command, 0, "SUCCESS\n", "", timed_out=False)

    monkeypatch.setattr(backend, "run_in_docker", capture_grader_profile)
    episode = tmp_path / "episode"
    episode.mkdir()
    run_docker_grader(episode, {}, (), "pytest", config=config)

    grader_argv = observed["argv"]
    assert grader_argv[grader_argv.index("--pids-limit") + 1] == "512"
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        assert f"{variable}=1" in grader_argv


def test_trusted_grader_refuses_policy_tmp_state(tmp_path: Path) -> None:
    state_root = tmp_path / "policy-tmp"
    state_root.mkdir()
    with pytest.raises(ValueError, match="fresh /tmp"):
        DockerConfig(
            image=IMAGE,
            gate_record=tmp_path / "gate.json",
            trusted_grader=True,
            episode_tmp_state=state_root,
        )


def test_run_docker_grader_forwards_the_optional_junit_report_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = DockerConfig(image=IMAGE, gate_record=tmp_path / "gate.json")
    episode = tmp_path / "episode"
    episode.mkdir()
    observed: dict[str, object] = {}

    def capture_report_path(
        workspace: Path, command: str, **kwargs: object
    ) -> backend.CommandResult:
        observed["workspace"] = workspace
        observed["command"] = command
        observed["junit_report_path"] = kwargs["junit_report_path"]
        return backend.CommandResult(command, 0, "", "", False)

    monkeypatch.setattr(backend, "run_in_docker", capture_report_path)

    result = run_docker_grader(
        episode,
        {},
        (),
        "pytest",
        config=config,
        junit_report_path=".r2e-junit.xml",
    )

    assert result.exit_code == 0
    assert observed["command"] == "pytest"
    assert observed["junit_report_path"] == ".r2e-junit.xml"
    assert isinstance(observed["workspace"], Path)


def test_run_in_docker_persists_bounded_junit_xml_separately_from_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "episode"
    workspace.mkdir()
    report_path = workspace / ".r2e-junit.xml"
    report_xml = '<?xml version="1.0"?><testsuite tests="1" />\n'
    evidence_path = tmp_path / "docker-evidence.jsonl"
    config = DockerConfig(
        image=IMAGE,
        gate_record=tmp_path / "gate.json",
        evidence_path=evidence_path,
    )
    events: list[str] = []

    monkeypatch.setattr(backend, "require_gate", lambda _config: None)
    monkeypatch.setattr(backend, "prepare_runtime_mountpoints", lambda *_args: None)
    monkeypatch.setattr(backend, "_inspect_image_volumes", lambda *_args, **_kwargs: (None, None))

    def run_docker(argv: list[str], _deadline: int) -> tuple[int, str, str, bool, bool]:
        assert argv[1] == "run"
        assert evidence_path.read_bytes().startswith(b"\x1f\x8b")
        events.append("grader-finished")
        report_path.write_text(report_xml)
        return 0, "grader stdout", "", False, False

    def cleanup_docker(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert argv[1:3] == ["rm", "--force"]
        events.append("container-cleanup")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(backend, "_run_bounded_docker", run_docker)
    monkeypatch.setattr(backend.subprocess, "run", cleanup_docker)

    original_reader = backend._read_junit_report

    def read_report(workspace: Path, report_path: str | None) -> str | None:
        events.append("report-read")
        return original_reader(workspace, report_path)

    monkeypatch.setattr(backend, "_read_junit_report", read_report)

    result = run_in_docker(
        workspace,
        "pytest",
        config=config,
        junit_report_path=".r2e-junit.xml",
    )

    assert result.stdout == "grader stdout"
    assert result.junit_xml == report_xml
    assert events == ["grader-finished", "report-read", "container-cleanup"]
    evidence = read_jsonl(evidence_path)[-1]
    assert evidence["stdout"] == "grader stdout"
    assert evidence["junit_xml"] == report_xml


@pytest.mark.parametrize("refusal_phase", ["started", "finished"])
def test_docker_evidence_budget_refuses_during_command_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal_phase: str
) -> None:
    workspace = tmp_path / "episode"
    workspace.mkdir()
    evidence_path = tmp_path / "docker-evidence.jsonl"
    config = DockerConfig(IMAGE, tmp_path / "gate.json", evidence_path=evidence_path)
    monkeypatch.setattr(backend, "require_gate", lambda _config: None)
    monkeypatch.setattr(backend, "prepare_runtime_mountpoints", lambda *_args: None)
    monkeypatch.setattr(backend, "_inspect_image_volumes", lambda *_args, **_kwargs: (None, None))
    commands_run: list[str] = []

    def run_command(_argv: list[str], _deadline: int) -> tuple[int, str, str, bool, bool]:
        commands_run.append("command")
        return 0, "synthetic output", "", False, False

    def guard(paths: list[Path]) -> None:
        phase = read_jsonl(evidence_path)[-1]["phase"]
        if phase == refusal_phase:
            guard_episode_size(paths, warning_bytes=0, hard_bytes=1)

    monkeypatch.setattr(backend, "_run_bounded_docker", run_command)
    monkeypatch.setattr(backend, "guard_episode_size", guard)
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
    )
    with pytest.raises(ValueError, match="episode evidence uses"):
        run_in_docker(workspace, "synthetic command", config=config)
    assert commands_run == ([] if refusal_phase == "started" else ["command"])
    assert read_jsonl(evidence_path)[-1]["phase"] == refusal_phase


@pytest.mark.parametrize("report_kind", ["missing", "oversized", "symlink"])
def test_run_in_docker_records_unavailable_junit_xml_as_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report_kind: str
) -> None:
    workspace = tmp_path / "episode"
    workspace.mkdir()
    report_path = workspace / ".r2e-junit.xml"
    if report_kind == "oversized":
        report_path.write_bytes(b"x" * (backend.MAXIMUM_JUNIT_XML_BYTES + 1))
    elif report_kind == "symlink":
        target = tmp_path / "junit-target.xml"
        target.write_text("<testsuite />")
        report_path.symlink_to(target)

    config = DockerConfig(image=IMAGE, gate_record=tmp_path / "gate.json")
    monkeypatch.setattr(backend, "require_gate", lambda _config: None)
    monkeypatch.setattr(backend, "prepare_runtime_mountpoints", lambda *_args: None)
    monkeypatch.setattr(backend, "_inspect_image_volumes", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(
        backend,
        "_run_bounded_docker",
        lambda *_args, **_kwargs: (0, "", "", False, False),
    )
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
    )

    result = run_in_docker(
        workspace,
        "pytest",
        config=config,
        junit_report_path=".r2e-junit.xml",
    )

    assert result.junit_xml is None


def test_generated_command_argv_runs_ladder_wrapper_under_bash(tmp_path: Path) -> None:
    config = DockerConfig(image=IMAGE, gate_record=tmp_path / "gate.json")
    marker = "__LADDER_SHELL_STATE_cpu_acceptance__"
    state = _ShellState(cwd=str(tmp_path))
    command = (
        "export DOCKER_BASH_ACCEPTANCE=ready; mkdir -p nested; cd nested; printf 'wrapper-ok\\n'"
    )
    wrapper = _shell_wrapper(command, state, marker)
    for login_shell, shell_option in ((True, "-lc"), (False, "-c")):
        argv = docker_run_args(config, tmp_path, "episode-toy", wrapper, login_shell=login_shell)
        image_index = argv.index(IMAGE)
        command_argv = argv[image_index + 1 :]
        process = subprocess.run(  # noqa: S603 - exercise generated shell argv with the host timeout
            ["/usr/bin/timeout", *command_argv],
            capture_output=True,
            text=True,
            check=False,
        )

        assert process.returncode == 0, process.stderr
        assert command_argv[2:] == ["/bin/bash", shell_option, wrapper]
        result = backend.CommandResult(
            wrapper, process.returncode, process.stdout, process.stderr, False
        )
        next_state, visible_stderr, cwd_after = _restore_shell_state(result, state, marker)
        assert process.stdout == "wrapper-ok\n"
        assert visible_stderr == ""
        assert cwd_after == f"{state.cwd}/nested"
        assert next_state.cwd == f"{state.cwd}/nested"
        assert 'DOCKER_BASH_ACCEPTANCE="ready"' in next_state.exported_script


@pytest.mark.slow
@pytest.mark.docker_real
def test_docker_run_executes_ladder_wrapper_with_python_image_bash(tmp_path: Path) -> None:
    image = _local_python_image_or_skip()
    marker = "__LADDER_SHELL_STATE_docker_acceptance__"
    state = _ShellState()
    command = (
        "test -x /bin/bash && export DOCKER_BASH_ACCEPTANCE=ready && mkdir nested "
        "&& cd nested && printf 'docker-wrapper-ok\\n'"
    )
    wrapper = _shell_wrapper(command, state, marker)
    gate_record = tmp_path / "gate.json"

    workspace = tmp_path / "episode"
    workspace.mkdir()
    config = DockerConfig(
        image=image,
        gate_record=gate_record,
        evidence_path=tmp_path / "evidence.jsonl",
    )

    result = run_in_docker(workspace, wrapper, config=config, timeout="15s")

    next_state, visible_stderr, cwd_after = _restore_shell_state(result, state, marker)
    assert result.exit_code == 0, result.stderr
    assert result.stdout == "docker-wrapper-ok\n"
    assert visible_stderr == ""
    assert cwd_after == "/work/nested"
    assert next_state.cwd == "/work/nested"
    assert 'DOCKER_BASH_ACCEPTANCE="ready"' in next_state.exported_script


@pytest.mark.slow
@pytest.mark.docker_real
def test_docker_episode_tmp_persists_and_trusted_grader_starts_fresh(tmp_path: Path) -> None:
    """Exercise policy persistence and fresh trusted grading through real containers."""
    image = _local_python_image_or_skip()
    gate_record = tmp_path / "gate.json"

    workspace = tmp_path / "episode"
    workspace.mkdir()
    state_root = tmp_path / "state"
    state_root.mkdir()
    policy_config = DockerConfig(image=image, gate_record=gate_record, episode_tmp_state=state_root)
    grader_config = DockerConfig(image=image, gate_record=gate_record, trusted_grader=True)

    written = run_in_docker(
        workspace, "printf persistent > /tmp/note", config=policy_config, timeout="15s"
    )
    assert written.exit_code == 0, written.stderr
    assert (state_root / "note").read_text() == "persistent"
    read_back = run_in_docker(workspace, "cat /tmp/note", config=policy_config, timeout="15s")
    assert read_back.exit_code == 0, read_back.stderr
    assert read_back.stdout == "persistent"
    grader = run_in_docker(
        workspace,
        "test ! -e /tmp/note && printf fresh",
        config=grader_config,
        timeout="15s",
        login_shell=False,
    )
    assert grader.exit_code == 0, grader.stderr
    assert grader.stdout == "fresh"
    assert not any(workspace.glob(".episode-tmp-*"))


@pytest.mark.parametrize("cpus", [0.5, 1, 2])
def test_container_cpuset_matches_the_cpu_quota_and_the_host_mask(cpus: float) -> None:
    host_cpus = os.sched_getaffinity(0)
    if len(host_cpus) < 2:
        pytest.skip("cpuset spreading needs at least two host CPUs")
    cpuset = [int(cpu) for cpu in container_cpuset("episode-toy", cpus).split(",")]
    assert len(cpuset) == math.ceil(cpus)
    assert set(cpuset) <= host_cpus
    assert container_cpuset("episode-toy", cpus) == container_cpuset("episode-toy", cpus)
    spread = {container_cpuset(f"episode-{index}", 1) for index in range(64)}
    assert len(spread) > 1


@pytest.mark.slow
@pytest.mark.docker_real
def test_docker_container_affinity_matches_its_cpu_quota(tmp_path: Path) -> None:
    image = _local_python_image_or_skip()
    gate_record = tmp_path / "gate.json"

    workspace = tmp_path / "episode"
    workspace.mkdir()
    config = DockerConfig(image=image, gate_record=gate_record)

    result = run_in_docker(
        workspace,
        "python3 -c 'import os; print(len(os.sched_getaffinity(0)))'",
        config=config,
        timeout="15s",
    )

    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "1"


def test_workspace_mount_must_be_an_absolute_container_path(tmp_path: Path) -> None:
    for workspace_mount in (
        "work",
        "",
        "/",
        "/work,readonly",
        "/work/../host",
        str(PurePosixPath("/", "tmp")),
        str(PurePosixPath("/", "tmp", "episode")),
        "/proc",
        "/proc/1",
    ):
        with pytest.raises(ValueError, match="workspace mount"):
            DockerConfig(
                image=IMAGE,
                gate_record=tmp_path / "gate.json",
                workspace_mount=workspace_mount,
            )

    config = DockerConfig(
        image=IMAGE,
        gate_record=tmp_path / "gate.json",
        workspace_mount="/testbed",
    )
    argv = docker_run_args(config, tmp_path, "episode-testbed", "id")
    assert argv[argv.index("--mount") + 1].endswith(",dst=/testbed,bind-recursive=disabled")
    assert argv[argv.index("--workdir") + 1] == "/testbed"


def test_copy_workspace_template_preserves_file_and_directory_mtimes(tmp_path: Path) -> None:
    template = tmp_path / "template"
    nested = template / "nested"
    nested.mkdir(parents=True)
    (nested / "module.py").write_text("baseline")
    (nested / "module.py").touch()
    file_mtime_ns = 1_600_000_000_123_456_789
    nested_mtime_ns = 1_600_000_010_123_456_789
    root_mtime_ns = 1_600_000_020_123_456_789
    os.utime(nested / "module.py", ns=(file_mtime_ns, file_mtime_ns))
    os.utime(nested, ns=(nested_mtime_ns, nested_mtime_ns))
    os.utime(template, ns=(root_mtime_ns, root_mtime_ns))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    copy_workspace_template(template, workspace)

    assert (workspace / "nested" / "module.py").stat().st_mtime_ns == file_mtime_ns
    assert (workspace / "nested").stat().st_mtime_ns == nested_mtime_ns
    assert workspace.stat().st_mtime_ns == root_mtime_ns

    (workspace / "nested" / "materialized.py").write_text("episode file")
    assert (workspace / "nested").stat().st_mtime_ns != nested_mtime_ns
    restore_workspace_directory_times(template, workspace)
    assert (workspace / "nested").stat().st_mtime_ns == nested_mtime_ns


def test_copy_workspace_template_requires_empty_destination(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "existing").write_text("keep")

    with pytest.raises(ValueError, match="empty"):
        copy_workspace_template(template, workspace)


@pytest.mark.parametrize("image", ["python:3.12-slim", "python@sha256:abc"])
def test_unpinned_image_refused(tmp_path: Path, image: str) -> None:
    with pytest.raises(ValueError, match="digest"):
        DockerConfig(image=image, gate_record=tmp_path / "gate.json")


@pytest.fixture
def live_gate_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(backend, "_PASSED_GATES", set())
    record: dict[str, object] = {
        "backend_hash": "old-code-provenance",
        "passed": True,
        "image": IMAGE,
        "probes": {key: {"contained": True, "negative_control": True} for key in REQUIRED_PROBES},
    }
    monkeypatch.setattr(run_docker_jail_tests, "run_red_team", lambda _path, *, image: record)
    return record


def test_live_gate_ignores_disk_and_code_hash_and_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_gate_record: dict[str, object]
) -> None:
    calls: list[str] = []

    def run_probes(_path: Path, *, image: str) -> dict[str, object]:
        calls.append(image)
        return live_gate_record

    monkeypatch.setattr(run_docker_jail_tests, "run_red_team", run_probes)
    config = DockerConfig(IMAGE, tmp_path / "missing-gate.json")
    require_gate(config)
    config.gate_record.write_text("obsolete record")
    require_gate(config)
    assert calls == [IMAGE]


@pytest.mark.parametrize(
    "violation", ["contained", "negative_control", "missing", "passed", "image"]
)
def test_live_gate_refuses_invalid_fresh_results(
    tmp_path: Path, live_gate_record: dict[str, object], violation: str
) -> None:
    probes = live_gate_record["probes"]
    assert isinstance(probes, dict)
    if violation in {"contained", "negative_control"}:
        probes["docker_socket_unreachable"][violation] = False
    elif violation == "missing":
        del probes["memory_limit_enforced"]
    elif violation == "passed":
        live_gate_record["passed"] = False
    else:
        live_gate_record["image"] = "python@sha256:" + "b" * 64
    config = DockerConfig(IMAGE, tmp_path / "gate.json")
    with pytest.raises(RuntimeError, match="Docker isolation gate"):
        require_gate(config)
    assert not backend._PASSED_GATES
    assert not config.gate_record.exists()


def test_snapshots_preserve_symlink_evidence_without_reading_target(tmp_path: Path) -> None:
    (tmp_path / "answer.txt").write_text("toy answer")
    (tmp_path / "link").symlink_to("/etc/passwd")
    snapshot = workspace_snapshot(tmp_path)
    assert snapshot["link"]["symlink"] == "/etc/passwd"
    assert "sha256" in snapshot["answer.txt"]


def test_submission_allowlist_rejects_symlink_and_traversal(tmp_path: Path) -> None:
    (tmp_path / "answer.txt").write_text("toy answer")
    assert submitted_files(tmp_path, ("answer.txt",)) == {"answer.txt": b"toy answer"}
    (tmp_path / "link").symlink_to("answer.txt")
    for path in ("link", "../answer.txt", "/etc/passwd"):
        with pytest.raises(ValueError, match=r"artifact|workspace"):
            submitted_files(tmp_path, (path,))


def test_real_subprocess_boundary_records_changes_and_stages_trusted_grader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_gate_record: dict[str, object]
) -> None:
    executable = tmp_path / "docker"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        "if arguments[0] == 'image':\n"
        "    print('null')\n"
        "elif arguments[0] == 'run':\n"
        "    assert arguments[arguments.index('--workdir') + 1] == '/testbed'\n"
        "    mount = arguments[arguments.index('--mount') + 1]\n"
        "    workspace = pathlib.Path(mount.split('src=')[1].split(',')[0])\n"
        "    if arguments[-1] == 'toy-grade':\n"
        "        assert (workspace / 'baseline.py').read_text() == 'closed baseline'\n"
        "        assert (workspace / 'grade.py').read_text() == 'trusted grader'\n"
        "        assert (workspace / 'answer.txt').read_text() == 'submitted answer'\n"
        "        assert not (workspace / 'malicious-extra.py').exists()\n"
        "        print('SUCCESS')\n"
        "    else:\n"
        "        (workspace / 'answer.txt').write_text('submitted answer')\n"
        "        print('agent command output')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    gate = tmp_path / "gate.json"

    workspace = tmp_path / "episode"
    workspace.mkdir()
    config = DockerConfig(image=IMAGE, gate_record=gate, workspace_mount="/testbed")
    result = run_in_docker(workspace, "toy-action", config=config)
    assert result.exit_code == 0
    (workspace / "grade.py").write_text("tampered grader")
    (workspace / "malicious-extra.py").write_text("agent-only file")
    template = tmp_path / "closed-template"
    template.mkdir()
    (template / "baseline.py").write_text("closed baseline")
    grade = run_docker_grader(
        workspace,
        {"grade.py": "trusted grader"},
        ("answer.txt",),
        "toy-grade",
        config=config,
        template_dir=template,
    )
    assert grade.ok
    records = read_jsonl(workspace.with_name("episode.docker-evidence.jsonl"))
    assert len(records) == 4
    assert records[1]["exit_code"] == 0
    assert records[1]["image"] == IMAGE
    assert "answer.txt" not in records[1]["before"]
    assert "answer.txt" in records[1]["after"]
    assert records[3]["before"]["grade.py"] != records[1]["after"].get("grade.py")


def test_docker_command_output_is_bounded_while_both_pipes_are_drained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_gate_record: dict[str, object]
) -> None:
    executable = tmp_path / "docker"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        "    os.write(1, b'null\\n')\n"
        "elif sys.argv[1] == 'run':\n"
        "    os.write(1, b'x' * (2 * 1024 * 1024))\n"
        "    os.write(2, b'y' * (2 * 1024 * 1024))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    gate = tmp_path / "gate.json"

    workspace = tmp_path / "episode"
    workspace.mkdir()
    result = run_in_docker(workspace, "write-long-output", config=DockerConfig(IMAGE, gate))
    assert result.exit_code == 0
    assert len(result.stdout.encode()) <= 1024 * 1024
    assert len(result.stderr.encode()) <= 1024 * 1024 + 100
    assert "truncated" in result.stderr
    records = read_jsonl(workspace.with_name("episode.docker-evidence.jsonl"))
    assert records[1]["output_truncated"] is True


def test_bounded_docker_client_is_killed_at_host_deadline() -> None:
    exit_code, stdout, stderr, timed_out, output_truncated = _run_bounded_docker(
        [
            sys.executable,
            "-c",
            "import sys,time; sys.stdout.write('started'); sys.stdout.flush(); time.sleep(30)",
        ],
        deadline_seconds=1,
    )
    assert exit_code == 124
    assert stdout == "started"
    assert "deadline exceeded" in stderr
    assert timed_out
    assert not output_truncated


def test_snapshots_record_modes_directories_and_refuse_scan_overruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "answer.txt").write_text("toy answer")
    before = workspace_snapshot(tmp_path)
    (tmp_path / "answer.txt").chmod(0o600)
    after = workspace_snapshot(tmp_path)
    assert before["empty"]["directory"] == "true"
    assert before["answer.txt"]["mode"] != after["answer.txt"]["mode"]
    monkeypatch.setattr("reward_hacking.harness.docker_backend.MAXIMUM_SNAPSHOT_BYTES", 1)
    with pytest.raises(SnapshotLimitError, match="byte"):
        workspace_snapshot(tmp_path)


def test_snapshot_refuses_unreadable_directories(tmp_path: Path) -> None:
    directory = tmp_path / "hidden"
    directory.mkdir()
    (directory / "answer.txt").write_text("toy answer")
    directory.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            workspace_snapshot(tmp_path)
    finally:
        directory.chmod(0o700)


def test_runtime_mount_validation_and_readonly_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    venv = runtime / "venv"
    venv.mkdir()
    monkeypatch.setattr(backend, "READ_ONLY_RUNTIME_ROOT", runtime)
    mount = backend.ReadOnlyMount(venv, "/testbed/.venv")
    config = DockerConfig(
        IMAGE, tmp_path / "gate", workspace_mount="/testbed", read_only_mounts=(mount,)
    )
    argv = docker_run_args(config, tmp_path, "runtime", "id")
    assert argv[argv.index("--mount", argv.index("--mount") + 1) + 1] == (
        f"type=bind,source={venv},target=/testbed/.venv,readonly,bind-recursive=disabled"
    )
    assert "volume-driver=local" not in " ".join(argv)
    assert "no-new-privileges" in argv
    assert "/proc/self/mountinfo" in argv[-1]
    for source in (tmp_path, runtime / ".." / "runtime" / "venv"):
        with pytest.raises(ValueError, match="runtime"):
            backend.ReadOnlyMount(source, "/opt/uvpy")
    link = runtime / "link"
    link.symlink_to(venv, target_is_directory=True)
    with pytest.raises(ValueError, match="runtime"):
        backend.ReadOnlyMount(link, "/opt/uvpy")
    with pytest.raises(ValueError, match=r"runtime|overlap"):
        DockerConfig(
            IMAGE, tmp_path / "gate", read_only_mounts=(backend.ReadOnlyMount(venv, "/work"),)
        )


def test_runtime_mountpoint_symlink_refused_and_snapshot_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    venv = runtime / "venv"
    venv.mkdir()
    monkeypatch.setattr(backend, "READ_ONLY_RUNTIME_ROOT", runtime)
    config = DockerConfig(
        IMAGE,
        tmp_path / "gate",
        workspace_mount="/testbed",
        read_only_mounts=(backend.ReadOnlyMount(venv, "/testbed/.venv"),),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".venv").symlink_to(venv, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        backend.prepare_runtime_mountpoints(workspace, config)
    (workspace / ".venv").unlink()
    backend.prepare_runtime_mountpoints(workspace, config)
    (workspace / ".venv" / "bulky.dat").write_text("not scanned")
    (workspace / "normal.txt").write_text("scanned")
    snapshot = workspace_snapshot(
        workspace, excluded_paths=backend.runtime_snapshot_exclusions(config)
    )
    assert ".venv/bulky.dat" not in snapshot
    assert "normal.txt" in snapshot
    (workspace / "runtime-link").symlink_to(venv / "bulky.dat")
    snapshot = workspace_snapshot(
        workspace, excluded_paths=backend.runtime_snapshot_exclusions(config)
    )
    assert "sha256" not in snapshot["runtime-link"]


@pytest.mark.parametrize("target", ["/opt/workspace", "/opt/workspace/dependency", "/opt/runtime"])
def test_runtime_mount_cannot_shadow_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    runtime = tmp_path / "runtime"
    source = runtime / "python"
    source.mkdir(parents=True)
    monkeypatch.setattr(backend, "READ_ONLY_RUNTIME_ROOT", runtime)
    workspace_mount = "/opt/runtime/workspace" if target == "/opt/runtime" else "/opt/workspace"
    with pytest.raises(ValueError, match="runtime"):
        DockerConfig(
            IMAGE,
            tmp_path / "gate",
            workspace_mount=workspace_mount,
            read_only_mounts=(backend.ReadOnlyMount(source, target),),
        )


@pytest.mark.parametrize(
    "flags", ["rw", "rw,nosuid,nodev", "ro", "ro,nodev", "ro,nosuid", "ro,nosuid,nodev"]
)
def test_runtime_guard_refuses_weakened_flags_before_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: str
) -> None:
    runtime = tmp_path / "runtime"
    source = runtime / "python"
    source.mkdir(parents=True)
    monkeypatch.setattr(backend, "READ_ONLY_RUNTIME_ROOT", runtime)
    config = DockerConfig(
        IMAGE, tmp_path / "gate", read_only_mounts=(backend.ReadOnlyMount(source, "/opt/uvpy"),)
    )
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"1 2 8:1 / /opt/uvpy {flags} - ext4 device rw\n")
    sentinel = tmp_path / "command-ran"
    guard = backend.runtime_mount_guard(config).replace(
        "/proc/self/mountinfo", shlex.quote(str(mountinfo))
    )
    result = subprocess.run(  # noqa: S603 - synthetic mountinfo and host scratch only
        ["/bin/sh", "-c", guard + f"touch {shlex.quote(str(sentinel))}"],
        check=False,
    )
    assert sentinel.exists() is ("ro" in flags.split(","))
    assert result.returncode == (0 if "ro" in flags.split(",") else 125)


@pytest.mark.slow
@pytest.mark.docker_real
def test_live_stage_gate_refuses_weakened_pids_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _local_python_image_or_skip()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    monkeypatch.setattr(backend, "READ_ONLY_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(run_docker_jail_tests, "READ_ONLY_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(backend, "_PASSED_GATES", set())
    original_run_argv = run_docker_jail_tests.DockerProbeRunner.run_argv

    def weakened_run_argv(
        self: run_docker_jail_tests.DockerProbeRunner,
        argv: list[str],
        name: str,
        timeout_seconds: int = 30,
    ) -> run_docker_jail_tests.ContainerResult:
        # Controls retain their 128-process limit; the hardened run is weakened to the same ceiling.
        if "--pids-limit" in argv and argv[argv.index("--pids-limit") + 1] == "64":
            argv = run_docker_jail_tests.replace_option_value(argv, "--pids-limit", "64", "128")
        return original_run_argv(self, argv, name, timeout_seconds)

    monkeypatch.setattr(run_docker_jail_tests.DockerProbeRunner, "run_argv", weakened_run_argv)
    config = DockerConfig(image, tmp_path / "sabotaged-gate.json")
    with pytest.raises(RuntimeError, match="pids_limit_enforced containment failed"):
        require_gate(config)
    assert not backend._PASSED_GATES
    assert not config.gate_record.exists()

    monkeypatch.setattr(run_docker_jail_tests.DockerProbeRunner, "run_argv", original_run_argv)
    require_gate(config)
    assert backend._PASSED_GATES
    assert json.loads(config.gate_record.read_text())["passed"] is True
