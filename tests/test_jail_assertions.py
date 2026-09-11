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

import pytest

from scripts.jail_assertions import (
    BUS_ENV_VARS,
    ProbeConfig,
    check_launcher_session_and_namespaces_left,
    check_runtime_bus_env_absent,
    parse_launcher_namespaces,
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
