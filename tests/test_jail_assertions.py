"""Pin the two ways a jail containment check can quietly stop covering what it claims to cover.

`scripts/run_jail_tests.sh` is the real gate for these checks: it runs them inside a jail, again
outside it, and requires each to flip. What it cannot see is a check whose *scope* has drifted --
one that reads a hardcoded copy of a constant, or one that passes because it was handed nothing to
compare against. Both are cheap to pin here, and neither needs a jail.

- ``check_runtime_bus_env_absent`` guards the D-Bus escape route (anything that can reach the user
  bus can ask systemd to spawn processes outside the jail). Its variable list was spelled twice:
  once as ``BUS_ENV_VARS`` and once inline in the comprehension, so adding a variable to the named
  constant would have changed nothing and no lint rule reports an unused module constant.
- the launcher-session check is one of two that cannot derive their own probe target (the other is
  ``--jail-python``): inside the jail there is nothing left to compare against, so a forgotten flag
  has to read as a failure rather than a pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.jail_assertions import (
    BUS_ENV_VARS,
    CONTAINMENT_CHECKS,
    ProbeConfig,
    check_dbus_socket_unreachable,
    check_host_processes_invisible,
    check_launcher_session_and_namespaces_left,
    check_nvidia_devices_absent,
    check_runtime_bus_env_absent,
    parse_launcher_namespaces,
    verify_negative_control,
)

NO_PATHS = ProbeConfig(host_homes=("/nonexistent-home",))


class TestTheBusVariableListHasOneSpelling:
    """The constant has to be the list the check reads, not a second copy of it."""

    def test_a_variable_added_to_the_constant_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "scripts.jail_assertions.BUS_ENV_VARS", (*BUS_ENV_VARS, "DBUS_SYSTEM_BUS_ADDRESS")
        )
        monkeypatch.setenv("DBUS_SYSTEM_BUS_ADDRESS", "unix:path=/run/dbus/system_bus_socket")

        result = check_runtime_bus_env_absent(NO_PATHS)
        assert not result.passed
        assert "DBUS_SYSTEM_BUS_ADDRESS" in result.detail

    def test_the_documented_variables_are_still_covered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: reading the constant must not lose the two variables it already held."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)

        result = check_runtime_bus_env_absent(NO_PATHS)
        assert not result.passed
        assert "XDG_RUNTIME_DIR" in result.detail

    def test_a_clean_environment_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in BUS_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert check_runtime_bus_env_absent(NO_PATHS).passed


class TestTheLauncherCheckCannotPassVacuously:
    """A check comparing against the launcher must fail when nobody told it about the launcher."""

    def test_missing_launcher_flags_are_a_failure(self) -> None:
        result = check_launcher_session_and_namespaces_left(NO_PATHS)
        assert not result.passed
        assert "--launcher-session" in result.detail

    def test_a_malformed_namespace_argument_is_refused(self) -> None:
        with pytest.raises(ValueError, match="KIND=ID"):
            parse_launcher_namespaces(["ipc"])

    def test_namespace_arguments_parse_to_pairs(self) -> None:
        assert parse_launcher_namespaces(["ipc=ipc:[4026531839]", "uts=uts:[4026531838]"]) == (
            ("ipc", "ipc:[4026531839]"),
            ("uts", "uts:[4026531838]"),
        )

    def test_a_pid_namespace_session_id_collision_is_not_shared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A namespaced session can reuse the launcher's numeric SID."""
        config = ProbeConfig(
            host_homes=("/nonexistent-home",),
            launcher_session=1,
            launcher_namespaces=(("ipc", "ipc:[host]"), ("uts", "uts:[host]")),
            launcher_pid_namespace="pid:[host]",
        )
        monkeypatch.setattr("scripts.jail_assertions.os.getsid", lambda _pid: 1)
        monkeypatch.setattr("scripts.jail_assertions.os.getpid", lambda: 2)
        monkeypatch.setattr(
            "scripts.jail_assertions._namespace_id",
            lambda kind: f"{kind}:[jail]",
        )
        monkeypatch.setattr(
            "scripts.jail_assertions._session_leader_namespace_id",
            lambda _session: "pid:[jail]",
        )

        result = check_launcher_session_and_namespaces_left(config)

        assert result.passed

    def test_unshare_without_setsid_is_still_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = ProbeConfig(
            host_homes=("/nonexistent-home",),
            launcher_session=1,
            launcher_namespaces=(("ipc", "ipc:[host]"), ("uts", "uts:[host]")),
            launcher_pid_namespace="pid:[host]",
        )
        monkeypatch.setattr("scripts.jail_assertions.os.getsid", lambda _pid: 0)
        monkeypatch.setattr(
            "scripts.jail_assertions._namespace_id",
            lambda kind: f"{kind}:[jail]",
        )

        result = check_launcher_session_and_namespaces_left(config)

        assert not result.passed
        assert "session leader is outside" in result.detail


class TestPortableNegativeControlFixtures:
    """Host-local resources need deterministic outside fixtures for the flip test."""

    def test_a_fixture_dbus_socket_is_seen_as_reachable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        socket_path = tmp_path / "bus"

        class ReachableSocket:
            def settimeout(self, _timeout: float) -> None:
                pass

            def connect(self, path: str) -> None:
                if path != str(socket_path):
                    raise OSError

            def close(self) -> None:
                pass

        monkeypatch.setattr(
            "scripts.jail_assertions.socket.socket", lambda *_args: ReachableSocket()
        )
        result = check_dbus_socket_unreachable(
            ProbeConfig(host_homes=("/nonexistent-home",), dbus_socket_paths=(str(socket_path),))
        )

        assert not result.passed
        assert str(socket_path) in result.detail

    def test_a_fixture_nvidia_node_is_seen_as_present(self, tmp_path: Path) -> None:
        device_root = tmp_path / "dev"
        device_root.mkdir()
        (device_root / "nvidia0").touch()

        result = check_nvidia_devices_absent(
            ProbeConfig(
                host_homes=("/nonexistent-home",),
                additional_device_paths=(str(device_root / "nvidia0"),),
            )
        )

        assert not result.passed
        assert "nvidia0" in result.detail


class TestHostDependentNegativeControl:
    def test_small_host_pid_namespace_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scripts.jail_assertions.HOST_INIT_NAMES", ())
        monkeypatch.setattr("scripts.jail_assertions.MAX_JAIL_VISIBLE_PIDS", 100000)
        monkeypatch.setattr("scripts.jail_assertions._namespace_id", lambda _kind: "pid:[host]")
        config = ProbeConfig(host_homes=("/nonexistent-home",), launcher_pid_namespace="pid:[host]")

        result = check_host_processes_invisible(config)

        assert not result.passed
        assert "launcher PID namespace" in result.detail

    def test_missing_launcher_pid_namespace_is_rejected(self) -> None:
        result = check_host_processes_invisible(NO_PATHS)

        assert not result.passed
        assert "--launcher-pid-namespace" in result.detail

    def test_a_clean_host_requires_a_reachable_fixture_for_the_negative_control(self) -> None:
        names = [check.__name__.removeprefix("check_") for check in CONTAINMENT_CHECKS]
        inside: list[dict[str, object]] = [{"name": name, "passed": True} for name in names]
        outside: list[dict[str, object]] = [
            {"name": name, "passed": name == "dbus_socket_unreachable"} for name in names
        ]

        vacuous = "dbus_socket_unreachable: also passed OUTSIDE the jail, so it proves nothing (vacuous check)"
        assert verify_negative_control(inside, outside) == [vacuous]

        next(result for result in inside if result["name"] == "dbus_socket_unreachable")[
            "passed"
        ] = False
        assert verify_negative_control(inside, outside) == [
            "dbus_socket_unreachable: did not pass INSIDE the jail, so containment is broken",
            vacuous,
        ]
