"""Cover the two gate entry points: `scripts/tmux_run.sh` and the Makefile wiring that uses it.

`scripts/tmux_run.sh` is the one place the tmux + tee + `EXITCODE` pattern is written down, so what
matters is the command it composes rather than that it prints something reassuring. Most of the tests
here run the real script against a stub `tmux` that records its argv, and several take the composed
session command and EXECUTE it, under both bash and zsh, which is what pins the claims that
`set -o pipefail` plus a bare `$?` records the wrapped command's status rather than tee's always-zero
one and that the trailer lands on a line of its own. Real-tmux tests then run the whole path end to
end, because a stub cannot tell us that a tmux session finds its command at all (a session's
environment comes from the tmux server, not from the shell that launched it, which is why the script
forwards PATH) and cannot tell us that the helper returns while the command is still running.

The Makefile half covers `make ci`'s concurrency and its status. `ci` runs lint, typecheck and test
through one `make -j3`, so two things need checking and neither is greppable: that the three really do
overlap, and that a failure in any one of them makes `make ci` exit non-zero. Both are driven from a
throwaway Makefile that includes the repo's own and overrides the three gate targets with stand-ins,
so the recipe under test is the real one while the gates cost milliseconds instead of five minutes.
The stand-ins rendezvous through a barrier that fails unless all three have started, which is what
would go red if the -j were dropped from the recipe.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from test_ship_tree import BASH, REPO_ROOT, SCRATCH_ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator

HELPER = REPO_ROOT / "scripts" / "tmux_run.sh"
MAKEFILE = REPO_ROOT / "Makefile"

# Neither the Makefile's default of 8 nor the 4 AGENTS.md's busy-box recipe passes; see TestGateTmuxWiring.
WIRING_TEST_WORKERS = "5"

# One stub tmux: records each argument with printf, since echo would eat an argument of `-e` as a flag.
TMUX_STUB = """#!/bin/bash
{
  echo "--- call"
  for arg in "$@"; do printf '%s\\n' "$arg"; done
} >> "$TMUX_STUB_LOG"
case "$1" in
  has-session)
    exit "${TMUX_STUB_HAS_SESSION_RC:-1}"
    ;;
  new-session)
    exit "${TMUX_STUB_NEW_SESSION_RC:-0}"
    ;;
esac
exit 0
"""

# A stub `uv` beside the stub tmux, so no test here can fire the real suite; its log means a gate ran.
UV_STUB = """#!/bin/bash
printf '%s\\n' "$*" >> "$UV_STUB_LOG"
exit 0
"""

# A stand-in gate clears this only once all three have started, so serial execution times out and fails.
BARRIER = '''#!/usr/bin/env python3
"""Rendezvous of the stand-in gates: <name> <expected total> <exit code>."""
import os
import sys
import time

name, total, rc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
here = os.path.dirname(os.path.abspath(__file__))
print(f"{name}: gate ran")
sys.stdout.flush()
open(os.path.join(here, name + ".started"), "w").close()
deadline = time.time() + 30.0
while time.time() < deadline:
    started = [f for f in os.listdir(here) if f.endswith(".started")]
    if len(started) >= total:
        break
    time.sleep(0.02)
else:
    print(f"{name}: barrier timed out, the gates did not overlap", file=sys.stderr)
    sys.exit(9)
open(os.path.join(here, name + ".finished"), "w").close()
sys.exit(rc)
'''

STANDIN_MAKEFILE = """include {makefile}

lint:
\t@$(PYTHON) $(BARRIER) lint 3 $(LINT_RC)

typecheck:
\t@$(PYTHON) $(BARRIER) typecheck 3 $(TYPECHECK_RC)

test:
\t@$(PYTHON) $(BARRIER) test 3 $(TEST_RC)
"""


def read_calls(log: Path) -> list[list[str]]:
    """The stub's log as one list of arguments per call."""
    if not log.exists():
        return []
    calls: list[list[str]] = []
    for line in log.read_text().splitlines():
        if line == "--- call":
            calls.append([])
        else:
            calls[-1].append(line)
    return calls


def wait_for_trailer(log: Path, timeout: float = 60.0) -> list[str]:
    """The log's lines once the EXITCODE trailer has landed, or as they stand when the wait ran out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log.exists() and "EXITCODE" in log.read_text():
            break
        time.sleep(0.05)
    return log.read_text().splitlines() if log.exists() else []


def new_session_call(calls: list[list[str]]) -> list[str]:
    matching = [call for call in calls if call and call[0] == "new-session"]
    assert len(matching) == 1, f"expected exactly one new-session call, got {calls}"
    return matching[0]


class GateStubs:
    """A stub `tmux` and a stub `uv` first on PATH, plus the environment that steers them."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.bin_dir = root / "bin"
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        for name, body in (("tmux", TMUX_STUB), ("uv", UV_STUB)):
            stub = self.bin_dir / name
            stub.write_text(body)
            stub.chmod(0o755)
        self.log = root / "tmux-calls.log"
        self.uv_log = root / "uv-calls.log"

    def env(self, **overrides: str) -> dict[str, str]:
        env = dict(os.environ)
        env["PATH"] = f"{self.bin_dir}{os.pathsep}{env['PATH']}"
        env["TMUX_STUB_LOG"] = str(self.log)
        env["UV_STUB_LOG"] = str(self.uv_log)
        env.update(overrides)
        return env

    @property
    def calls(self) -> list[list[str]]:
        return read_calls(self.log)

    @property
    def uv_calls(self) -> list[str]:
        return self.uv_log.read_text().splitlines() if self.uv_log.exists() else []


@pytest.fixture
def stubs(tmp_path: Path) -> GateStubs:
    return GateStubs(tmp_path)


@pytest.fixture
def log_dir() -> Iterator[Path]:
    """Somewhere the helper will accept a log: pytest's tmp_path is under /tmp, which it refuses."""
    path = Path(tempfile.mkdtemp(prefix="tmux-run-test-", dir=SCRATCH_ROOT))
    yield path
    shutil.rmtree(path)


def run_helper(
    stubs: GateStubs,
    *args: str,
    cwd: Path | None = None,
    **env_overrides: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [BASH, str(HELPER), *args],
        cwd=str(cwd or REPO_ROOT),
        env=stubs.env(**env_overrides),
        capture_output=True,
        text=True,
        check=False,
    )


class TestTmuxRunLaunch:
    """What the helper composes and hands to tmux."""

    def test_the_session_command_is_the_canonical_pipefail_tee_exitcode_line(
        self, stubs: GateStubs, log_dir: Path
    ) -> None:
        log = log_dir / "run.log"
        result = run_helper(stubs, "cell-run", "--log", str(log), "--", "echo", "hi")
        assert result.returncode == 0, result.stderr
        session_command = new_session_call(stubs.calls)[-1]
        assert session_command.startswith("set -o pipefail;")
        assert f"| tee {log}" in session_command
        # A bare $? under pipefail (zsh lacks ${PIPESTATUS[0]}), captured before the newline check runs.
        assert f"| tee {log}; rc=$?;" in session_command
        assert f"echo EXITCODE=$rc >> {log}" in session_command
        assert "PIPESTATUS" not in session_command
        assert "status=$?" not in session_command

    def test_the_default_log_is_var_tmp_named_after_the_session(self, stubs: GateStubs) -> None:
        name = f"defaultlog-{uuid.uuid4().hex[:8]}"
        result = run_helper(stubs, name, "--", "true")
        assert result.returncode == 0, result.stderr
        assert f"log: /var/tmp/{name}.log" in result.stdout
        assert f"/var/tmp/{name}.log" in new_session_call(stubs.calls)[-1]  # noqa: S108

    def test_it_prints_a_poll_command_and_the_zsh_safe_attach_target(
        self, stubs: GateStubs, log_dir: Path
    ) -> None:
        log = log_dir / "poll.log"
        result = run_helper(stubs, "poll-me", "--log", str(log), "--", "true")
        assert f"poll: tail -n 40 {log}" in result.stdout
        assert f"follow: tail -f {log}" in result.stdout
        # Quoted because zsh reads an unquoted =word as EQUALS expansion and dies before tmux runs.
        assert "attach: tmux attach -t '=poll-me'" in result.stdout

    def test_the_launching_shells_path_and_directory_are_passed_in(
        self, stubs: GateStubs, tmp_path: Path
    ) -> None:
        """A tmux session's environment is the SERVER's, so PATH has to be handed over explicitly."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        result = run_helper(stubs, "envcheck", "--", "true", cwd=work_dir)
        assert result.returncode == 0, result.stderr
        call = new_session_call(stubs.calls)
        assert "-d" in call
        assert call[call.index("-c") + 1] == str(work_dir)
        assert call[call.index("-e") + 1].startswith("PATH=")
        assert str(stubs.bin_dir) in call[call.index("-e") + 1]

    def test_an_existing_session_is_checked_for_by_exact_name(self, stubs: GateStubs) -> None:
        """Without the '=' prefix has-session matches by prefix, so `ci` would look taken by `ci-2`."""
        run_helper(stubs, "gate-ci", "--", "true")
        has_session = [call for call in stubs.calls if call[0] == "has-session"]
        assert has_session == [["has-session", "-t", "=gate-ci"]]

    def test_arguments_needing_quoting_survive_into_the_session_command(
        self, stubs: GateStubs, log_dir: Path
    ) -> None:
        log = log_dir / "quoting.log"
        result = run_helper(
            stubs, "quoting", "--log", str(log), "--", "printf", "%s\\n", "two words; echo no"
        )
        assert result.returncode == 0, result.stderr
        session_command = new_session_call(stubs.calls)[-1]
        # printf %q escapes with backslashes rather than quotes, so assert that shape positively.
        assert "two\\ words\\;\\ echo\\ no" in session_command

    def test_the_previous_log_is_rotated_rather_than_truncated(
        self, stubs: GateStubs, log_dir: Path
    ) -> None:
        log = log_dir / "rotate.log"
        log.write_text("the run that crashed\n")
        result = run_helper(stubs, "rotate", "--log", str(log), "--", "true")
        assert result.returncode == 0, result.stderr
        assert f"previous log rotated to: {log}.prev" in result.stdout
        assert Path(f"{log}.prev").read_text() == "the run that crashed\n"

    def test_a_missing_log_directory_is_created(self, stubs: GateStubs, log_dir: Path) -> None:
        log = log_dir / "nested" / "deeper" / "run.log"
        result = run_helper(stubs, "nested", "--log", str(log), "--", "true")
        assert result.returncode == 0, result.stderr
        assert log.parent.is_dir()


class TestTheHelperIsCleanShell:
    """Nothing else in the suite shellchecks `scripts/`, so the one script this module owns does."""

    def test_shellcheck_passes(self) -> None:
        if shutil.which("shellcheck") is None:
            pytest.skip("shellcheck is not installed on this box")
        finished = subprocess.run(  # noqa: S603
            ["shellcheck", str(HELPER)],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        assert finished.returncode == 0, finished.stdout + finished.stderr


class TestTmuxRunRefusals:
    """Every path that must launch nothing, and say so with a non-zero status."""

    @pytest.mark.parametrize(
        ("args", "expected_in_stderr"),
        [
            pytest.param((), "no session name", id="no-arguments"),
            pytest.param(("only-a-name",), "no command after --", id="no-command"),
            pytest.param(("name", "--"), "no command after --", id="dashdash-with-nothing-after"),
            pytest.param(("name", "make", "ci"), "must follow a literal --", id="missing-dashdash"),
            pytest.param(("name", "--log"), "--log needs a path", id="log-without-a-value"),
            pytest.param(
                ("name", "--log=", "--", "true"), "empty path", id="log-with-an-empty-value"
            ),
            pytest.param(
                ("bad:name", "--", "true"),
                "letters, digits, dash or underscore",
                id="colon-in-name",
            ),
            pytest.param(
                ("bad.name", "--", "true"), "letters, digits, dash or underscore", id="dot-in-name"
            ),
            pytest.param(
                ("-weird", "--", "true"), "letters, digits, dash or underscore", id="leading-dash"
            ),
        ],
    )
    def test_usage_refusals_exit_2_and_launch_nothing(
        self, stubs: GateStubs, args: tuple[str, ...], expected_in_stderr: str
    ) -> None:
        result = run_helper(stubs, *args)
        assert result.returncode == 2, result.stdout + result.stderr
        assert expected_in_stderr in result.stderr
        assert "nothing was launched" in result.stderr
        assert [call for call in stubs.calls if call[0] == "new-session"] == []

    def test_a_log_under_tmp_is_refused(self, stubs: GateStubs) -> None:
        """The box's tmpfs scratch has an inode cap of its own, shared by every session on it."""
        result = run_helper(stubs, "tmplog", "--log", "/tmp/gate.log", "--", "true")  # noqa: S108
        assert result.returncode == 2, result.stdout
        assert "under /tmp" in result.stderr
        assert [call for call in stubs.calls if call[0] == "new-session"] == []

    def test_a_relative_log_is_resolved_before_the_tmp_check(self, stubs: GateStubs) -> None:
        result = run_helper(stubs, "relative", "--log", "gate.log", "--", "true", cwd=Path("/tmp"))  # noqa: S108
        assert result.returncode == 2, result.stdout
        assert "under /tmp" in result.stderr

    def test_a_log_that_only_reaches_tmp_through_dotdot_is_refused(self, stubs: GateStubs) -> None:
        """The refusal is a prefix match, so an untidied path would walk straight past it."""
        result = run_helper(stubs, "dotdot", "--log", "/var/../tmp/gate.log", "--", "true")
        assert result.returncode == 2, result.stdout
        assert "/tmp/gate.log is under /tmp" in result.stderr  # noqa: S108
        assert [call for call in stubs.calls if call[0] == "new-session"] == []

    def test_a_log_reached_through_a_symlink_into_tmp_is_refused(
        self, stubs: GateStubs, log_dir: Path
    ) -> None:
        """A ~/logs that someone pointed at the tmpfs is the same inode hazard by another name."""
        link = log_dir / "sneaky"
        link.symlink_to("/tmp")  # noqa: S108
        result = run_helper(stubs, "symlinked", "--log", str(link / "gate.log"), "--", "true")
        assert result.returncode == 2, result.stdout
        assert "/tmp/gate.log is under /tmp" in result.stderr  # noqa: S108
        assert [call for call in stubs.calls if call[0] == "new-session"] == []

    def test_a_log_path_that_is_a_directory_is_refused(
        self, stubs: GateStubs, log_dir: Path
    ) -> None:
        result = run_helper(stubs, "dirlog", "--log", str(log_dir), "--", "true")
        assert result.returncode == 2, result.stdout
        assert "is a directory" in result.stderr
        assert [call for call in stubs.calls if call[0] == "new-session"] == []

    def test_a_taken_session_name_is_refused(self, stubs: GateStubs) -> None:
        result = run_helper(stubs, "taken", "--", "true", TMUX_STUB_HAS_SESSION_RC="0")
        assert result.returncode == 2, result.stdout
        assert "already exists" in result.stderr
        assert [call for call in stubs.calls if call[0] == "new-session"] == []

    def test_tmux_refusing_to_start_the_session_exits_3(self, stubs: GateStubs) -> None:
        """Distinct from a usage refusal: the arguments were fine and the session still is not there."""
        result = run_helper(stubs, "wontstart", "--", "true", TMUX_STUB_NEW_SESSION_RC="1")
        assert result.returncode == 3, result.stdout
        assert "nothing is running" in result.stderr


class TestComposedSessionCommand:
    """Execute what the helper hands tmux, which is the only way to check the EXITCODE claim."""

    @pytest.mark.parametrize("shell", ["bash", "zsh"])
    def test_the_log_records_the_commands_status_not_tees(
        self, stubs: GateStubs, log_dir: Path, shell: str
    ) -> None:
        shell_path = shutil.which(shell)
        if shell_path is None:
            pytest.skip(f"{shell} is not installed on this box")
        log = log_dir / f"{shell}.log"
        result = run_helper(
            stubs,
            f"exec-{shell}",
            "--log",
            str(log),
            "--",
            BASH,
            "-c",
            "echo working; exit 7",
        )
        assert result.returncode == 0, result.stderr
        session_command = new_session_call(stubs.calls)[-1]
        executed = subprocess.run(  # noqa: S603
            [shell_path, "-c", session_command],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(log_dir),
        )
        assert executed.returncode == 0, executed.stderr
        assert log.read_text().splitlines() == ["working", "EXITCODE=7"]

    @pytest.mark.parametrize("shell", ["bash", "zsh"])
    def test_output_without_a_trailing_newline_keeps_the_trailer_on_its_own_line(
        self, stubs: GateStubs, log_dir: Path, shell: str
    ) -> None:
        """tee is byte transparent, so an unterminated last line would absorb the trailer."""
        shell_path = shutil.which(shell)
        if shell_path is None:
            pytest.skip(f"{shell} is not installed on this box")
        log = log_dir / f"{shell}-nonewline.log"
        result = run_helper(
            stubs, f"nonl-{shell}", "--log", str(log), "--", "printf", "%s", "half a line"
        )
        assert result.returncode == 0, result.stderr
        session_command = new_session_call(stubs.calls)[-1]
        subprocess.run(  # noqa: S603
            [shell_path, "-c", session_command], capture_output=True, check=True, cwd=str(log_dir)
        )
        assert log.read_text().splitlines() == ["half a line", "EXITCODE=0"]

    def test_the_printed_status_command_reads_the_trailer_not_the_jobs_own_output(
        self, stubs: GateStubs, log_dir: Path
    ) -> None:
        """`grep -m1 EXITCODE` answers with the first match anywhere, which a job can print itself."""
        log = log_dir / "status.log"
        result = run_helper(
            stubs,
            "statusread",
            "--log",
            str(log),
            "--",
            BASH,
            "-c",
            "echo 'echo EXITCODE=99 >> /var/tmp/other.log'; exit 3",
        )
        assert result.returncode == 0, result.stderr
        subprocess.run(  # noqa: S603
            [BASH, "-c", new_session_call(stubs.calls)[-1]],
            capture_output=True,
            check=False,
            cwd=str(log_dir),
        )
        status_lines = [line for line in result.stdout.splitlines() if line.startswith("status: ")]
        assert len(status_lines) == 1, result.stdout
        status_command = status_lines[0].removeprefix("status: ").split("   #")[0]
        read = subprocess.run(  # noqa: S603
            [BASH, "-c", status_command], capture_output=True, text=True, check=True
        )
        assert read.stdout.strip() == "EXITCODE=3", read.stdout


class TestRealTmuxSession:
    """Two end-to-end launches, because a stub cannot say whether a session finds its command."""

    @pytest.fixture
    def session_name(self) -> Iterator[str]:
        """Skips here rather than in each body: the teardown itself needs tmux to exist."""
        if shutil.which("tmux") is None:
            pytest.skip("tmux is not installed on this box")
        name = f"tmuxruntest-{uuid.uuid4().hex[:8]}"
        yield name
        subprocess.run(["tmux", "kill-session", "-t", f"={name}"], capture_output=True, check=False)  # noqa: S603, S607

    def launch(
        self, session_name: str, log: Path, command: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [BASH, str(HELPER), session_name, "--log", str(log), "--", BASH, "-c", command],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_a_real_session_runs_the_command_and_writes_its_status(
        self, session_name: str, log_dir: Path
    ) -> None:
        log = log_dir / "real.log"
        result = self.launch(session_name, log, "echo from-the-session; exit 5")
        assert result.returncode == 0, result.stdout + result.stderr
        assert wait_for_trailer(log) == ["from-the-session", "EXITCODE=5"]

    def test_the_helper_returns_before_the_wrapped_command_has_a_status(
        self, session_name: str, log_dir: Path
    ) -> None:
        """The load-bearing contract: this script's 0 means the session exists, not that the job passed.

        A helper that waited for the command would either propagate its status (here 7) or return only
        once the trailer was already in the log, so both halves of the contract are checked: a return
        while the command is still running, and the command's own status reaching the log afterwards.
        """
        log = log_dir / "still-running.log"
        result = self.launch(session_name, log, "echo started; sleep 3; exit 7")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "EXITCODE" not in (log.read_text() if log.exists() else ""), "the helper waited"
        assert wait_for_trailer(log) == ["started", "EXITCODE=7"]


class TestGateTmuxWiring:
    """`make test GATE_TMUX=1` and `make ci GATE_TMUX=1` route through the helper.

    Every sub-make here is handed `TEST_WORKERS=WIRING_TEST_WORKERS` on its command line and the composed
    command is checked for that exact count. Make forwards the outer command line's variables to every
    sub-make through the MAKEFLAGS environment variable, so an outer `make test TEST_WORKERS=4` (the busy-box
    invocation AGENTS.md recommends) reached these sub-makes too, and an assertion that assumed the default
    read `-n 4` and went red with nothing wrong in the tree. A count no caller uses can only appear in the
    composed command because it flowed from this command line, and the sub-make's own command line beats
    whatever the outer MAKEFLAGS carries (measured: an inherited `TEST_WORKERS=4` neither wins nor rides
    along in the sub-make's forwarded MAKEFLAGS).
    """

    def run_make(
        self, stubs: GateStubs, *args: str, **env_overrides: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            ["make", "--no-print-directory", *args],  # noqa: S607
            cwd=str(REPO_ROOT),
            env=stubs.env(**env_overrides),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_make_test_under_gate_tmux_hands_the_suite_to_the_helper(
        self, stubs: GateStubs
    ) -> None:
        result = self.run_make(
            stubs,
            "test",
            "GATE_TMUX=1",
            "GATE_TMUX_NAME=wiring-test",
            f"TEST_WORKERS={WIRING_TEST_WORKERS}",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        session_command = new_session_call(stubs.calls)[-1]
        words = shlex.split(session_command.split(" 2>&1 |", 1)[0])
        assert "RLVR_SMOKE=1" in words[words.index("env") + 1 :]
        assert "pytest" in words
        assert words[words.index("pytest") + 1 :][:4] == [
            "-n",
            WIRING_TEST_WORKERS,
            "--dist",
            "loadfile",
        ]
        assert "/var/tmp/wiring-test.log" in session_command  # noqa: S108
        assert stubs.uv_calls == [], "the suite ran in the foreground instead of being handed over"

    def test_make_test_changed_under_gate_tmux_hands_the_iteration_gate_to_the_helper(
        self, stubs: GateStubs
    ) -> None:
        """The iteration gate runs minutes too, so leaving it in the foreground is the same stall."""
        result = self.run_make(
            stubs,
            "test-changed",
            "GATE_TMUX=1",
            "GATE_TMUX_NAME=wiring-changed",
            f"TEST_WORKERS={WIRING_TEST_WORKERS}",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        session_command = new_session_call(stubs.calls)[-1]
        words = shlex.split(session_command.split(" 2>&1 |", 1)[0])
        assert "scripts/test_changed.py" in words
        assert words[words.index("--workers") + 1] == WIRING_TEST_WORKERS
        assert "/var/tmp/wiring-changed.log" in session_command  # noqa: S108
        assert stubs.uv_calls == [], "the iteration gate ran in the foreground instead of tmux"

    def test_make_ci_under_gate_tmux_wraps_one_session_and_disarms_the_inner_gates(
        self, stubs: GateStubs
    ) -> None:
        """One session for the whole gate, inner gates disarmed, nothing buffering their output.

        Without `GATE_TMUX=` on the recursive make, the test gate would open a second session under
        the same name and be refused. Without the forwarded MAKEFLAGS, a command-line override such as
        TEST_WORKERS=4 would be dropped inside tmux and the gate would say nothing, so the forwarded
        value is checked for the override this test passed rather than for merely being present. And
        every `--output-sync` mode holds a command's output until that command ENDS (measured: `line`
        means each recipe line, not each line of text), so the suite would print nothing for its whole
        five minutes and a killed run would lose where it was; interleaved output is the accepted cost.
        """
        result = self.run_make(
            stubs,
            "ci",
            "GATE_TMUX=1",
            "GATE_TMUX_NAME=wiring-ci",
            f"TEST_WORKERS={WIRING_TEST_WORKERS}",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        wrapped = new_session_call(stubs.calls)[-1].split(" 2>&1 |", 1)[0]
        words = shlex.split(wrapped)
        # Positional, not a substring match: MAKEFLAGS carries the caller's own GATE_TMUX=1.
        before_make, make_arguments = words[: words.index("make")], words[words.index("make") + 1 :]
        assert "GATE_TMUX=" in make_arguments
        assert "-j3" in make_arguments
        assert make_arguments[-3:] == ["lint", "typecheck", "test"]
        forwarded = [word for word in before_make if word.startswith("MAKEFLAGS=")]
        assert len(forwarded) == 1, before_make
        worker_overrides = [
            flag for flag in forwarded[0].split() if flag.startswith("TEST_WORKERS=")
        ]
        assert worker_overrides == [f"TEST_WORKERS={WIRING_TEST_WORKERS}"], forwarded[0]
        assert stubs.uv_calls == [], "a gate ran in the foreground instead of being handed over"
        assert not [word for word in make_arguments if word.startswith(("--output-sync", "-O"))]

    def test_make_ci_without_gate_tmux_touches_no_tmux(self, stubs: GateStubs) -> None:
        result = self.run_make(stubs, "ci", "--dry-run")
        assert result.returncode == 0, result.stdout + result.stderr
        assert stubs.calls == []
        assert "tmux_run.sh" not in result.stdout


class TestMakeCiConcurrencyAndStatus:
    """The real `ci` recipe, driven with stand-in gates that cost milliseconds."""

    @pytest.fixture
    def standin_tree(self, tmp_path: Path) -> Path:
        barrier = tmp_path / "barrier.py"
        barrier.write_text(BARRIER)
        (tmp_path / "Makefile").write_text(STANDIN_MAKEFILE.format(makefile=MAKEFILE))
        return tmp_path

    def run_ci(
        self, standin_tree: Path, lint: int = 0, typecheck: int = 0, test: int = 0
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [  # noqa: S607
                "make",
                "--no-print-directory",
                "-C",
                str(standin_tree),
                "ci",
                f"PYTHON={sys.executable}",
                f"BARRIER={standin_tree / 'barrier.py'}",
                f"LINT_RC={lint}",
                f"TYPECHECK_RC={typecheck}",
                f"TEST_RC={test}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_the_three_gates_overlap_and_a_green_run_exits_0(self, standin_tree: Path) -> None:
        result = self.run_ci(standin_tree)
        assert result.returncode == 0, result.stdout + result.stderr
        for gate in ("lint", "typecheck", "test"):
            assert f"{gate}: gate ran" in result.stdout
            assert (standin_tree / f"{gate}.finished").exists(), f"{gate} never cleared the barrier"

    @pytest.mark.parametrize("failing", ["lint", "typecheck", "test"])
    def test_any_red_gate_makes_ci_red_and_is_named(self, standin_tree: Path, failing: str) -> None:
        result = self.run_ci(standin_tree, **{failing: 1})
        assert result.returncode != 0, result.stdout + result.stderr
        assert f": {failing}] Error 1" in result.stderr, result.stderr

    def test_a_red_gate_is_not_hidden_by_the_two_green_ones(self, standin_tree: Path) -> None:
        """The status is the AND: two passes and one failure is a failure."""
        result = self.run_ci(standin_tree, typecheck=1)
        assert result.returncode != 0
        assert "lint: gate ran" in result.stdout
        assert "test: gate ran" in result.stdout
