"""Drive scripts/idle_watchdog.sh against a stubbed idle box, because layer 2 has no other check.

This is the killswitch layer that catches the cheap failure the max-lifetime dead-man cannot: a
rented GPU box that bootstraps and then never starts its job. Nothing in the repo launches an
instance, so the only way to know the watchdog does what its header claims is to run it with
``sudo``, ``pgrep``, ``nvidia-smi``, ``getent`` and ``shutdown`` shimmed onto PATH and watch what it
does. Every test here drives the real script; none of them read its source.

Five defects in the watchdog are pinned here, plus the teardown path, the arm-time grant probe and
the off-box publication of its verdict. Every one was watched to fail before it was trusted; the
defects, the 23 sabotages and the two harness details that are easy to get wrong are recorded in
`docs/killswitch.md`.

One invariant governs every test in this file: each asserts ``shutdown`` still ran. The watchdog
calls ``ec2:TerminateInstances`` first, but that path is strictly additive, because the API call
failing, timing out, or being impossible must never cost the box its death.

Nothing here may reach real infrastructure, and the terminate branch is the one place where that is
not merely untidy: this dev box is itself an EC2 instance, so an unstubbed ``curl`` would answer from
its IMDS and hand a real instance id to a real ``aws ec2 terminate-instances``. Both binaries are
therefore shimmed in the base fixture rather than per test, so a future test that reaches the
teardown without thinking about it is still safe. That fixture machinery lives in
``tests/box_stubs.py``.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from box_stubs import (
    DRY_RUN_ALLOWED_MESSAGE,
    IMDS_TOKEN,
    INSTANCE,
    REGION,
    STALL_SECONDS,
    STUB_S3_DEST,
    TEST_CALL_BOUND_SECONDS,
    THROTTLED_MESSAGE,
    UNAUTHORIZED_MESSAGE,
    ObservedProcess,
    install_shared_stubs,
    launch_observed,
    write_stub,
)
from box_stubs import StubbedBox as SharedStubbedBox

REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = REPO_ROOT / "scripts" / "idle_watchdog.sh"

# Long enough for several polls at --poll-seconds 1 on a box running many concurrent agents.
OBSERVE_SECONDS = 4.0

# Not a bare " idle ", which also matched the arming line `terminate after 1 idle minutes`.
IDLE_POLL_PATTERN = re.compile(r" idle \d+s/\d+s")


@dataclass(frozen=True)
class StubbedBox(SharedStubbedBox):
    """A host the watchdog can be pointed at, with every external command it calls shimmed.

    The IMDS, AWS CLI and ``shutdown`` surface lives in ``tests/box_stubs.py``. What is added here is
    the idleness surface only the watchdog reads -- tmux panes, a work process, GPU memory, the run
    user's home -- plus the arm record.
    """

    run_user_home: Path
    arm_record: Path

    def set_work_in_flight(self, *, present: bool) -> None:
        """Make ``pgrep`` report (or stop reporting) a work process for the run user."""
        write_stub(self.bin_dir, "pgrep", "exit 0" if present else "exit 1")

    def set_tmux_panes(self, *pane_commands: str) -> None:
        """Give the run user one tmux session whose panes run these commands, in pane order.

        No arguments means no tmux server at all, which is how every tmux subcommand fails. The stub
        answers ``ls`` as well as ``list-panes`` so a test distinguishes "a session exists" from
        "something is running in it" -- the whole point of the narrowed signal.
        """
        if not pane_commands:
            write_stub(self.bin_dir, "tmux", "exit 1")
            return
        rows = "\n".join(pane_commands)
        write_stub(
            self.bin_dir,
            "tmux",
            f"""case "$1" in
  ls|list-sessions) echo 'work: 1 windows'; exit 0 ;;
  list-panes) cat <<'ROWS'
{rows}
ROWS
    exit 0 ;;
  *) exit 1 ;;
esac""",
        )

    def set_gpu_memory(self, *used_mib_per_gpu: int) -> None:
        """Make ``nvidia-smi`` report one ``memory.used`` row per GPU, in device order."""
        rows = "\n".join(str(used) for used in used_mib_per_gpu)
        write_stub(self.bin_dir, "nvidia-smi", f"cat <<'ROWS'\n{rows}\nROWS")


@pytest.fixture
def stubbed_box(tmp_path: Path) -> StubbedBox:
    """An idle box: no tmux session for the run user, no work process, a quiet GPU."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    run_user_home = tmp_path / "run-user-home"
    run_user_home.mkdir()
    imds_dir = tmp_path / "imds"
    imds_dir.mkdir()
    aws_dir = tmp_path / "aws-responses"
    aws_dir.mkdir()
    # Under tmp_path rather than its /var/log default, which this non-root suite cannot write.
    arm_record = tmp_path / "idle-watchdog-arm.txt"

    # `sudo` drops its `-u USER` and runs the rest, so the tests drive the real tmux query.
    write_stub(bin_dir, "sudo", 'shift 2\nexec "$@"')
    write_stub(bin_dir, "nvidia-smi", "echo 0")
    write_stub(bin_dir, "getent", f'echo "$2:x:1000:1000::{run_user_home}:/bin/bash"')

    box = StubbedBox(
        bin_dir=bin_dir,
        imds_dir=imds_dir,
        aws_dir=aws_dir,
        curl_log=tmp_path / "curl-calls.log",
        aws_log=tmp_path / "aws-calls.log",
        s3_payload_log=tmp_path / "s3-payloads.log",
        shutdown_log=tmp_path / "shutdown-attempts.log",
        run_user_home=run_user_home,
        arm_record=arm_record,
        env={},
    )
    box.env.update(
        {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "IDLE_WATCHDOG_ARM_RECORD": str(arm_record),
        }
    )
    box.set_work_in_flight(present=False)
    box.set_tmux_panes()
    install_shared_stubs(box)
    return box


@dataclass
class WatchdogRun(ObservedProcess):
    """A watchdog process under observation, plus whatever it has logged so far."""

    def idle_poll_lines(self) -> list[str]:
        """The per-poll idle lines, which prove the loop really ran rather than dying at once."""
        return [line for line in self.output().splitlines() if IDLE_POLL_PATTERN.search(line)]


LaunchWatchdog = Callable[..., WatchdogRun]


@pytest.fixture
def launch(stubbed_box: StubbedBox, tmp_path: Path) -> Iterator[LaunchWatchdog]:
    """Start the watchdog and guarantee it is reaped even when an assertion fails."""
    started: list[WatchdogRun] = []

    def start(*args: str, env: dict[str, str] | None = None) -> WatchdogRun:
        log_path = tmp_path / f"watchdog-{len(started)}.log"
        process = launch_observed(
            [str(WATCHDOG), *args],
            env=stubbed_box.env if env is None else env,
            log_path=log_path,
        )
        run = WatchdogRun(process=process, log_path=log_path)
        started.append(run)
        return run

    yield start

    for run in started:
        if run.process.poll() is None:
            run.process.terminate()
        run.process.wait(timeout=10)


class TestTheTerminateBranchKeepsWatchingWhenShutdownFails:
    """A watchdog that cannot terminate the box must stay loud and keep trying, not exit 0."""

    def test_a_refused_shutdown_is_retried_and_reported(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        stubbed_box.set_shutdown_exit(1)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(OBSERVE_SECONDS)

        output = run.output()
        assert run.is_running(), (
            "the watchdog stopped watching after a failed shutdown, so layer 2 is gone:\n" + output
        )
        assert "SHUTDOWN FAILED" in output, f"a refused shutdown was not reported:\n{output}"
        assert len(stubbed_box.shutdown_attempts()) >= 2, (
            f"a refused shutdown was not retried: {stubbed_box.shutdown_attempts()}"
        )

    def test_a_successful_shutdown_ends_the_watch(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """The control: the fix must not turn a working killswitch into a loop that never stops."""
        stubbed_box.set_shutdown_exit(0)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        assert run.process.wait(timeout=15) == 0
        assert len(stubbed_box.shutdown_attempts()) == 1, (
            f"expected exactly one shutdown: {stubbed_box.shutdown_attempts()}"
        )
        assert "TERMINATING" in run.output()


class TestTheArmTimeProbeAnnouncesWhetherTheSelfTerminateGrantIsLive:
    """A dry run at arm time, because a denied teardown is indistinguishable from a healthy one.

    The policy being attached and EC2 actually populating ``ec2:InstanceProfile`` and
    ``ec2:ResourceTag`` in a live request context are different claims. If the second is false every
    self-terminate is denied, the watchdog correctly falls back to ``shutdown``, and the log reads as
    a benign refusal while the release this whole path exists to request is silently never made, so
    the box's fate rests entirely on a launch setting nothing here can read. One dry run per box
    answers it at arm time, when there is still a night left to act on the answer.

    Two properties carry this class. The verdict must come from the response text and never the exit
    status, because a dry run that PASSES exits non-zero; and the probe must never make a call that
    could really terminate a box, which is what ``terminating_calls`` is for.
    """

    def test_a_passing_dry_run_reports_the_grant_live(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        stubbed_box.set_work_in_flight(present=True)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        output = run.output()
        assert "GRANT PROBE LIVE" in output, output
        assert f"authorized for {INSTANCE} in {REGION}" in output, (
            f"the verdict did not name what it was a verdict about:\n{output}"
        )

    def test_a_denied_dry_run_reports_the_grant_missing_and_says_what_follows(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """The state of every box launched before the IAM grant lands, which must read unmistakably."""
        stubbed_box.set_dry_run_response(stderr=UNAUTHORIZED_MESSAGE)
        stubbed_box.set_work_in_flight(present=True)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        output = run.output()
        assert "GRANT PROBE MISSING" in output, output
        assert "dies on the shutdown fallback" in output, (
            f"a missing grant was not distinguished from a box that will not die:\n{output}"
        )
        assert "GRANT PROBE LIVE" not in output

    def test_an_unrecognised_answer_reports_indeterminate_and_quotes_it(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """Throttling is neither verdict, and a probe that guessed one would be worse than silent."""
        stubbed_box.set_dry_run_response(stderr=THROTTLED_MESSAGE)
        stubbed_box.set_work_in_flight(present=True)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        output = run.output()
        assert "GRANT PROBE INDETERMINATE" in output, output
        assert "RequestLimitExceeded" in output, (
            f"the answer was called indeterminate without showing what it was:\n{output}"
        )

    @pytest.mark.parametrize(
        ("exit_code", "message", "expected_verdict"),
        [
            (254, DRY_RUN_ALLOWED_MESSAGE, "GRANT PROBE LIVE"),
            (0, UNAUTHORIZED_MESSAGE, "GRANT PROBE MISSING"),
        ],
    )
    def test_the_verdict_reads_the_response_and_never_the_exit_code(
        self,
        stubbed_box: StubbedBox,
        launch: LaunchWatchdog,
        exit_code: int,
        message: str,
        expected_verdict: str,
    ) -> None:
        """Both rows are the wrong answer for an implementation that keys on ``$?``.

        A dry run that passes exits NON-ZERO, because AWS surfaces DryRunOperation as an error, so
        reading the exit status reports a working grant as broken. The second row is the inverse
        control: a zero exit carrying UnauthorizedOperation must still read as denied. Anyone
        "fixing" this to check the exit code turns one or both of these red, which is the point.
        """
        stubbed_box.set_dry_run_response(stderr=message, exit_code=exit_code)
        stubbed_box.set_work_in_flight(present=True)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        assert expected_verdict in run.output(), run.output()

    def test_the_probe_never_makes_a_call_that_could_really_terminate_the_box(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """The one call in this script where a typo destroys a running experiment.

        Driven on a box that is doing work, so nothing legitimately terminates it: any call without
        ``--dry-run`` here is the probe having killed a live run. The flag is asserted immediately
        after the subcommand, which is where it has to stay for that to remain true.
        """
        stubbed_box.set_work_in_flight(present=True)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        assert stubbed_box.dry_run_calls() == [
            f"ec2 terminate-instances --dry-run --instance-ids {INSTANCE} --region {REGION}"
        ], f"the probe's call was not the dry run intended: {stubbed_box.aws_calls()}"
        assert stubbed_box.terminating_calls() == [], (
            "the arm-time probe made a call that would really terminate a working box: "
            f"{stubbed_box.terminating_calls()}"
        )
        assert stubbed_box.shutdown_attempts() == [], run.output()
        assert run.is_running(), run.output()

    def test_an_indeterminate_probe_neither_stops_the_arming_nor_changes_the_teardown(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """A box must never refuse to run its experiment because a permission check was inconclusive.

        Nor may an inconclusive probe alter what happens at teardown: the box still terminates on the
        idle threshold exactly as it would have. Driven with IMDS unreachable, which is the broadest
        way for the probe to learn nothing at all.
        """
        stubbed_box.set_imds_reachable(reachable=False)
        stubbed_box.set_shutdown_exit(0)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")

        assert run.process.wait(timeout=15) == 0, run.output()
        output = run.output()
        assert "GRANT PROBE INDETERMINATE" in output, output
        assert "armed: terminate after 0 idle minutes" in output, (
            f"the watchdog did not arm after an inconclusive probe:\n{output}"
        )
        assert len(stubbed_box.shutdown_attempts()) == 1, (
            f"an inconclusive probe changed what teardown did: {stubbed_box.shutdown_attempts()}"
        )

    def test_the_resolved_cli_path_is_logged_at_arm_time(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """Root's PATH under nohup is not the PATH anyone tested against, so record what was found."""
        stubbed_box.set_work_in_flight(present=True)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        assert f"aws CLI resolved to: {stubbed_box.bin_dir / 'aws'}" in run.output(), run.output()

    def test_a_region_that_is_not_a_region_is_never_passed_to_the_api(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """The other half of the instance-id guard: the region is interpolated into the call too."""
        stubbed_box.set_instance_identity(region="<html>504 Gateway Timeout</html>")
        stubbed_box.set_shutdown_exit(0)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")

        assert run.process.wait(timeout=15) == 0, run.output()
        assert stubbed_box.aws_calls() == [], (
            f"a value that is not a region reached the CLI: {stubbed_box.aws_calls()}"
        )
        assert "GRANT PROBE INDETERMINATE" in run.output(), run.output()
        assert len(stubbed_box.shutdown_attempts()) == 1, stubbed_box.shutdown_attempts()


class TestTheTeardownCallsTheEc2ApiBeforeFallingBackToShutdown:
    """The API call buys an audit record and a real release; ``shutdown`` is still what kills the box.

    ``InstanceInitiatedShutdownBehavior`` is one setting away from a box that merely stops on
    ``shutdown`` and bills its root volume indefinitely while the log reads as a clean kill.
    ``TerminateInstances`` releases the instance whatever that setting says, and leaves a record a
    later reader can look up, where a halt is visible only on the disk the teardown destroys.

    Every test here asserts ``shutdown`` ran anyway, which is the invariant worth more than the
    release. This is a killswitch, and a dependency on a network call, a credential and an IAM grant
    may not become a new way for a rented box to survive forever -- the grant in particular is
    genuinely absent on any box launched before it lands.
    """

    def test_a_successful_call_terminates_this_instance_in_its_own_region(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        stubbed_box.set_shutdown_exit(0)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")

        assert run.process.wait(timeout=15) == 0, run.output()
        assert stubbed_box.terminating_calls() == [
            f"ec2 terminate-instances --region {REGION} --instance-ids {INSTANCE} --output text"
        ], f"the API call was not the one intended: {stubbed_box.aws_calls()}"
        assert "API terminate SUCCEEDED" in run.output(), run.output()
        assert len(stubbed_box.shutdown_attempts()) == 1, (
            "a successful API terminate skipped the shutdown fallback, so the box's death now "
            f"depends on the API call: {stubbed_box.shutdown_attempts()}"
        )

    def test_the_metadata_lookup_is_authenticated_with_an_imdsv2_token(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """These boxes require IMDSv2, so a tokenless GET is a 401 and yields no identity at all."""
        stubbed_box.set_shutdown_exit(0)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        assert run.process.wait(timeout=15) == 0, run.output()

        calls = stubbed_box.curl_calls()
        assert any("-X PUT" in call and "/latest/api/token" in call for call in calls), (
            f"no IMDSv2 token was ever requested: {calls}"
        )
        metadata_calls = [call for call in calls if "/latest/meta-data/" in call]
        assert {call.rsplit("/latest/meta-data/", 1)[1] for call in metadata_calls} == {
            "instance-id",
            "placement/region",
        }, f"the identity lookup did not fetch both values: {metadata_calls}"
        assert all(f"X-aws-ec2-metadata-token: {IMDS_TOKEN}" in call for call in metadata_calls), (
            f"a metadata lookup went out without its IMDSv2 token: {metadata_calls}"
        )

    def test_a_refused_call_still_reaches_shutdown(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """Every box launched before the ec2:TerminateInstances grant lands is in exactly this state."""
        stubbed_box.set_terminate_response(exit_code=254, stderr=UNAUTHORIZED_MESSAGE)
        stubbed_box.set_shutdown_exit(0)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")

        assert run.process.wait(timeout=15) == 0, run.output()
        output = run.output()
        assert "API terminate FAILED (exit 254)" in output, output
        assert "UnauthorizedOperation" in output, (
            f"the CLI's own error never reached the log, so the cause is unreadable:\n{output}"
        )
        assert "API terminate SUCCEEDED" not in output, (
            f"a refused call was reported as a successful terminate:\n{output}"
        )
        assert len(stubbed_box.shutdown_attempts()) == 1, (
            f"a refused API call cost the box its shutdown: {stubbed_box.shutdown_attempts()}"
        )

    def test_an_absent_cli_still_reaches_shutdown(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """A box with no AWS CLI at all, which is what any AMI outside the DLAMI family looks like."""
        stubbed_box.set_shutdown_exit(0)
        run = launch(
            "--minutes",
            "0",
            "--poll-seconds",
            "1",
            "--user",
            "runner",
            env=stubbed_box.env_without_aws(),
        )

        assert run.process.wait(timeout=15) == 0, run.output()
        output = run.output()
        assert f"calling ec2:TerminateInstances on {INSTANCE} in {REGION}" in output, (
            "the run never got as far as the CLI, so it proves nothing about the CLI being "
            f"missing:\n{output}"
        )
        assert "API terminate FAILED" in output, output
        assert len(stubbed_box.shutdown_attempts()) == 1, (
            f"a missing CLI cost the box its shutdown: {stubbed_box.shutdown_attempts()}"
        )

    def test_an_unreachable_imds_still_reaches_shutdown_and_calls_no_api(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        stubbed_box.set_imds_reachable(reachable=False)
        stubbed_box.set_shutdown_exit(0)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")

        assert run.process.wait(timeout=15) == 0, run.output()
        assert "no IMDSv2 token" in run.output(), run.output()
        assert stubbed_box.aws_calls() == [], (
            f"the API was called with no instance id to call it on: {stubbed_box.aws_calls()}"
        )
        assert len(stubbed_box.shutdown_attempts()) == 1, (
            f"an unreachable IMDS cost the box its shutdown: {stubbed_box.shutdown_attempts()}"
        )

    def test_a_hung_call_is_cut_off_at_its_bound_and_still_reaches_shutdown(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """A stalled endpoint may not hold a box that has already been judged idle.

        Elapsed time is itself the assertion here: strip the bound and the stub answers success six
        seconds later, which reads in the log as a clean API terminate rather than as a killswitch
        that sat waiting on a network call.
        """
        stubbed_box.set_terminate_response(delay_seconds=STALL_SECONDS)
        stubbed_box.set_shutdown_exit(0)
        env = {
            **stubbed_box.env,
            "IDLE_WATCHDOG_TERMINATE_CALL_MAX_SECONDS": TEST_CALL_BOUND_SECONDS,
        }

        started = time.monotonic()
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner", env=env)
        assert run.process.wait(timeout=30) == 0, run.output()
        elapsed = time.monotonic() - started

        output = run.output()
        assert f"API terminate CUT OFF at the {TEST_CALL_BOUND_SECONDS}s bound" in output, output
        assert elapsed < STALL_SECONDS - 2, (
            f"the watchdog took {elapsed:.1f}s to terminate against a "
            f"{TEST_CALL_BOUND_SECONDS}s bound on a {STALL_SECONDS}s stall, so the bound did not "
            f"hold:\n{output}"
        )
        assert len(stubbed_box.shutdown_attempts()) == 1, (
            f"a hung API call cost the box its shutdown: {stubbed_box.shutdown_attempts()}"
        )

    def test_metadata_that_is_not_an_instance_id_is_never_passed_to_the_api(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """``curl -f`` rejects a non-2xx, but a proxy answering 200 with a page of HTML survives it."""
        stubbed_box.set_instance_identity(instance="<html>504 Gateway Timeout</html>")
        stubbed_box.set_shutdown_exit(0)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")

        assert run.process.wait(timeout=15) == 0, run.output()
        assert stubbed_box.aws_calls() == [], (
            "something that is not an instance id was handed to a destructive call: "
            f"{stubbed_box.aws_calls()}"
        )
        assert "API terminate UNAVAILABLE" in run.output(), run.output()
        assert len(stubbed_box.shutdown_attempts()) == 1, (
            f"an unusable metadata answer cost the box its shutdown: "
            f"{stubbed_box.shutdown_attempts()}"
        )


class TestTheIdleThresholdIsWallClock:
    """The threshold is minutes of wall clock, so it cannot depend on how often the loop polls."""

    def test_a_one_second_poll_does_not_terminate_after_one_second(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        run = launch("--minutes", "1", "--poll-seconds", "1", "--user", "runner")
        time.sleep(OBSERVE_SECONDS)

        output = run.output()
        assert len(run.idle_poll_lines()) >= 2, (
            f"the watchdog did not poll, so this test proves nothing:\n{output}"
        )
        assert stubbed_box.shutdown_attempts() == [], (
            "terminated a box idle for seconds against a one-minute threshold:\n" + output
        )
        assert run.is_running(), output

    def test_work_resuming_restarts_the_idle_timer(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """An epoch stamp is only correct if it is cleared: a stale one terminates mid-agenda."""
        run = launch("--minutes", "1", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)
        stubbed_box.set_work_in_flight(present=True)
        time.sleep(2.5)

        output = run.output()
        assert "work resumed" in output, f"the idle timer was never reported as reset:\n{output}"
        assert stubbed_box.shutdown_attempts() == [], output


class TestTheKeepaliveEscapeHatchFollowsTheRunUser:
    """The suppression file has to live in the home of the user --user names."""

    def test_a_keepalive_in_the_run_users_home_suppresses_termination(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        (stubbed_box.run_user_home / "KEEPALIVE").write_text("held for debugging\n")
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        output = run.output()
        assert "SUPPRESSED" in output, f"the run user's KEEPALIVE was not read:\n{output}"
        assert stubbed_box.shutdown_attempts() == [], (
            "terminated a box that was deliberately held:\n" + output
        )
        assert run.is_running(), output

    def test_an_unknown_run_user_is_refused_at_startup(self, stubbed_box: StubbedBox) -> None:
        """A --user with no passwd entry has no suppression path, so arming would be a lie."""
        write_stub(stubbed_box.bin_dir, "getent", "exit 2")
        completed = subprocess.run(  # noqa: S603 - repo script, literal arguments
            [str(WATCHDOG), "--minutes", "0", "--poll-seconds", "1", "--user", "nobody-here"],
            env=stubbed_box.env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert completed.returncode == 2, completed.stdout + completed.stderr
        assert "nobody-here" in completed.stderr
        assert stubbed_box.shutdown_attempts() == []


class TestWorkInFlightHoldsTheBox:
    """The control for the idle detection itself: a busy box must never be terminated."""

    def test_a_work_process_prevents_termination(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        stubbed_box.set_work_in_flight(present=True)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        assert stubbed_box.shutdown_attempts() == [], run.output()
        assert run.is_running(), run.output()


class TestTheTmuxSignalIsWorkAndNotMerelyASession:
    """An empty tmux session is not work, and reading it as work defeated the whole layer.

    The incident in the script's header is a box that bootstrapped and then sat at 0% because its
    driving session went away before the job started. The repo's own convention for long jobs is
    ``tmux new-session -d`` followed by sending the command in, so the session exists before any
    work does -- and while the mere existence of a session counted, that box was held until the
    max-lifetime dead-man fired hours later. Which is exactly the failure layer 2 exists to catch.

    The signal is therefore what a pane is RUNNING. The three-signal AND is untouched: a pane
    sitting at a shell prompt only makes the tmux signal quiet, and the work process and the GPU
    still have to be quiet too before the idle timer starts.
    """

    def test_a_session_of_idle_shells_does_not_hold_the_box(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        stubbed_box.set_tmux_panes("bash", "-bash", "zsh")
        stubbed_box.set_shutdown_exit(0)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")

        assert run.process.wait(timeout=15) == 0, run.output()
        assert "TERMINATING" in run.output()
        assert len(stubbed_box.shutdown_attempts()) == 1, stubbed_box.shutdown_attempts()

    def test_a_pane_running_a_job_holds_the_box(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """The control: narrowing the signal must not stop a real job from holding the box.

        Held on the tmux signal alone -- no work process, quiet GPU -- because that is the case the
        narrowing could break: a training job whose pane runs ``python`` while ``pgrep`` is stubbed
        silent stands for every job the other two signals would miss.
        """
        stubbed_box.set_tmux_panes("bash", "python")
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        output = run.output()
        assert stubbed_box.shutdown_attempts() == [], (
            "terminated a box whose tmux pane was running a job:\n" + output
        )
        assert run.is_running(), output

    def test_a_pane_whose_command_tmux_will_not_name_holds_the_box(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """Unknown is not idle, the same rule the GPU check follows for ``[N/A]``.

        An empty ``pane_current_command`` is a pane tmux could not account for. Erring toward
        holding the box costs idle minutes; the other way costs a paid run and its unsynced
        artifacts.
        """
        stubbed_box.set_tmux_panes("bash", "")
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        output = run.output()
        assert stubbed_box.shutdown_attempts() == [], output
        assert run.is_running(), output


class TestTheGpuCheckReadsEveryDevice:
    """Reading device 0 only is the defect ``gpu_preflight.vram_mib_across_devices`` already fixed.

    The boxes this watchdog is armed on are the ones rented for the sizes that do not fit here, and
    a 27B needs two cards, so the two-GPU host is not hypothetical. A watchdog that queries
    ``memory.used`` and takes the first row terminates a box whose training is on device 1 with
    device 0 empty -- destroying a paid run and its unsynced artifacts, which is the most expensive
    thing in this file. Held by any busy device, not by their sum: the threshold is "a card is in
    use", and summing would also let many idle drivers add up past it.
    """

    def test_work_on_the_second_gpu_holds_the_box(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        stubbed_box.set_gpu_memory(0, 40_000)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        output = run.output()
        assert stubbed_box.shutdown_attempts() == [], (
            "terminated a box whose GPU 1 held 40 GB because only GPU 0 was read:\n" + output
        )
        assert run.is_running(), output

    def test_work_on_the_first_gpu_still_holds_the_box(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """The control for the above: widening the query must not stop device 0 counting."""
        stubbed_box.set_gpu_memory(40_000, 0)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        assert stubbed_box.shutdown_attempts() == [], run.output()
        assert run.is_running(), run.output()

    def test_idle_drivers_on_several_devices_do_not_add_up_to_busy(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """The other control: four idle cards holding a driver each must still read as idle."""
        stubbed_box.set_gpu_memory(300, 300, 300, 300)
        stubbed_box.set_shutdown_exit(0)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")

        assert run.process.wait(timeout=15) == 0
        assert "TERMINATING" in run.output()

    def test_a_device_reporting_unreadable_usage_is_not_read_as_empty(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """``[N/A]`` is what nvidia-smi prints for a device it cannot account for, and unknown
        usage is not zero usage -- the lesson ``gpu_preflight.gpu_processes`` records, where reading
        it as zero let a peer holding the whole card pass the busy threshold. Erring toward holding
        the box is the cheap direction: idle minutes cost dollars, a terminated run costs the run.
        """
        write_stub(stubbed_box.bin_dir, "nvidia-smi", "printf '0\\n[N/A]\\n'")
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        output = run.output()
        assert stubbed_box.shutdown_attempts() == [], (
            "terminated a box whose second device would not report its usage:\n" + output
        )
        assert run.is_running(), output


class TestTheArmTimeVerdictIsPublishedSoItOutlivesTheBox:
    """The grant probe's answer has to leave the box, or it is a check that reports into the void.

    The probe writes its verdict to stdout, which the caller redirects into a file on the root disk
    this script exists to destroy. Every launcher kit's off-box channel for that file is a ``tail -2``
    or ``tail -3`` inside a heartbeat object overwritten every 120 to 300 seconds, and the loop's first
    idle line pushes the verdict out of that window permanently. So the boxes whose verdict matters
    most, the ones layer 2 really terminates, are exactly the ones that lose it: a sweep of the live
    boxes on 2026-08-27 found a single readable verdict, and only because it had armed minutes before.

    Publishing happens once, before the watch begins, so an abrupt death hours later cannot take it.
    Everything here is best-effort by design, and every test asserts the box still armed and still
    died: a killswitch may not acquire a new way to fail, and an S3 prefix, a credential and a network
    call are three of them.
    """

    @pytest.mark.parametrize("dest", [STUB_S3_DEST, f"{STUB_S3_DEST}/"])
    def test_the_arm_record_lands_under_the_configured_prefix_keyed_by_instance_id(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog, dest: str
    ) -> None:
        """Keyed by instance id, because a shared key is silently a REPLACE.

        Two boxes arming under one run prefix would leave only whichever wrote last, and the verdict
        worth reading is the unusual one. The trailing-slash row is not decoration: a kit's prefix
        variable is hand-written, both forms occur across the existing kits, and ``prefix//killswitch``
        is a different key from ``prefix/killswitch``.
        """
        stubbed_box.set_work_in_flight(present=True)
        run = launch(
            "--minutes",
            "0",
            "--poll-seconds",
            "1",
            "--user",
            "runner",
            env=stubbed_box.env_publishing_to(dest),
        )
        time.sleep(2.5)

        expected_call = (
            f"s3 cp {stubbed_box.arm_record} {STUB_S3_DEST}/killswitch-arm/{INSTANCE}.txt "
            f"--region {REGION} --only-show-errors"
        )
        assert stubbed_box.s3_calls() == [expected_call], (
            f"the upload was not the one intended: {stubbed_box.s3_calls()}"
        )
        assert run.is_running(), run.output()

    def test_the_uploaded_bytes_carry_the_grant_probe_verdict(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """An object that exists but does not carry the verdict is the same void with a listing.

        So this asserts the bytes the CLI was handed, not the call it was handed them by: an upload of
        an empty file, or of a window that starts after the probe, looks identical at the argv.
        """
        stubbed_box.set_work_in_flight(present=True)
        run = launch(
            "--minutes",
            "0",
            "--poll-seconds",
            "1",
            "--user",
            "runner",
            env=stubbed_box.env_publishing_to(),
        )
        time.sleep(2.5)

        payload = stubbed_box.s3_payloads()
        assert "armed: terminate after 0 idle minutes" in payload, (
            f"the published record does not say the watchdog armed:\n{payload}"
        )
        assert (
            f"GRANT PROBE LIVE: ec2:TerminateInstances is authorized for {INSTANCE}" in payload
        ), f"the published record does not carry the verdict it exists for:\n{payload}"
        assert "arm record PUBLISHED" in run.output(), run.output()

    def test_the_verdict_leaves_the_box_before_the_first_poll_line_exists(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """The property that distinguishes this from the heartbeat tail it replaces.

        The first idle line is precisely what evicted the verdict from every kit's ``tail -N``, so
        publishing after even one poll would reproduce the bug. Asserted as the absence of an idle line
        in the payload, which is an ordering claim with no timing to be flaky about.
        """
        run = launch(
            "--minutes",
            "0",
            "--poll-seconds",
            "1",
            "--user",
            "runner",
            env=stubbed_box.env_publishing_to(),
        )
        time.sleep(2.5)

        payload = stubbed_box.s3_payloads()
        assert "GRANT PROBE" in payload, (
            f"nothing was published, so ordering proves nothing:\n{payload}"
        )
        assert run.idle_poll_lines(), (
            "the loop never logged an idle line, so there was nothing for the upload to have raced:\n"
            + run.output()
        )
        assert not IDLE_POLL_PATTERN.search(payload), (
            f"the record was published after the loop had already begun polling:\n{payload}"
        )

    def test_an_unconfigured_destination_uploads_nothing_and_names_the_variable(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """The state of every launcher kit until its author adds the variable, and they are not ours.

        The kits live outside this repo and cannot be fixed centrally, so a forgotten destination has
        to be visible in the one place its author does read: the bootstrap log. The local record must
        still exist, because "published nowhere" may not also mean "recorded nowhere".
        """
        stubbed_box.set_work_in_flight(present=True)
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner")
        time.sleep(2.5)

        output = run.output()
        assert stubbed_box.s3_calls() == [], (
            f"an unset destination still produced an upload: {stubbed_box.s3_calls()}"
        )
        assert "arm record UNCONFIGURED" in output, output
        assert "IDLE_WATCHDOG_S3_DEST" in output, (
            f"the log did not name the variable a kit has to set:\n{output}"
        )
        assert "GRANT PROBE LIVE" in stubbed_box.arm_record.read_text(), (
            "publishing nowhere also lost the local record, so the verdict is unreadable even over "
            f"SSM while the box lives:\n{stubbed_box.arm_record.read_text()}"
        )
        assert run.is_running(), output

    def test_a_refused_upload_neither_stops_the_arming_nor_changes_the_teardown(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """The killswitch invariant: a denied PutObject may not cost a rented box its death."""
        stubbed_box.set_s3_response(
            exit_code=1, stderr="upload failed: An error occurred (AccessDenied)"
        )
        stubbed_box.set_shutdown_exit(0)
        run = launch(
            "--minutes",
            "0",
            "--poll-seconds",
            "1",
            "--user",
            "runner",
            env=stubbed_box.env_publishing_to(),
        )

        assert run.process.wait(timeout=15) == 0, run.output()
        output = run.output()
        assert "armed: terminate after 0 idle minutes" in output, (
            f"the watchdog did not arm after a refused upload:\n{output}"
        )
        assert "arm record FAILED (exit 1)" in output, output
        assert "AccessDenied" in output, (
            f"the CLI's own error never reached the log, so the cause is unreadable:\n{output}"
        )
        assert "arm record PUBLISHED" not in output, (
            f"a refused upload was reported as a published record:\n{output}"
        )
        assert len(stubbed_box.shutdown_attempts()) == 1, (
            f"a refused upload cost the box its shutdown: {stubbed_box.shutdown_attempts()}"
        )

    def test_a_hung_upload_is_cut_off_at_its_bound_and_the_box_still_dies(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """Elapsed time is itself the assertion: strip the bound and S3 holds the box for the stall.

        This one runs while the box is healthy rather than during teardown, so an unbounded call does
        not merely delay a death -- it delays the arming, and the window before a watchdog is armed is
        the window the whole layer exists to close.
        """
        stubbed_box.set_s3_response(delay_seconds=STALL_SECONDS)
        stubbed_box.set_shutdown_exit(0)
        env = stubbed_box.env_publishing_to(
            IDLE_WATCHDOG_STATUS_UPLOAD_MAX_SECONDS=TEST_CALL_BOUND_SECONDS
        )

        started = time.monotonic()
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner", env=env)
        assert run.process.wait(timeout=30) == 0, run.output()
        elapsed = time.monotonic() - started

        output = run.output()
        assert f"arm record CUT OFF at the {TEST_CALL_BOUND_SECONDS}s bound" in output, output
        assert elapsed < STALL_SECONDS - 2, (
            f"the watchdog took {elapsed:.1f}s against a {TEST_CALL_BOUND_SECONDS}s bound on a "
            f"{STALL_SECONDS}s stall, so the bound did not hold:\n{output}"
        )
        assert len(stubbed_box.shutdown_attempts()) == 1, (
            f"a hung upload cost the box its shutdown: {stubbed_box.shutdown_attempts()}"
        )

    def test_a_box_that_cannot_read_its_own_identity_keeps_the_record_local_and_still_dies(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """No identity means no key, and a key built from an empty id would litter the bucket.

        Deliberately not worth a fallback key: with IMDS unreachable the CLI has no instance
        credentials either, so the upload could not have succeeded. The cost is that these boxes are
        knowable only by the ABSENCE of an object, which is why the denominator has to come from
        describe-instances rather than from the bucket.
        """
        stubbed_box.set_imds_reachable(reachable=False)
        stubbed_box.set_shutdown_exit(0)
        run = launch(
            "--minutes",
            "0",
            "--poll-seconds",
            "1",
            "--user",
            "runner",
            env=stubbed_box.env_publishing_to(),
        )

        assert run.process.wait(timeout=15) == 0, run.output()
        output = run.output()
        assert stubbed_box.s3_calls() == [], (
            f"a key was built out of an unresolved identity: {stubbed_box.s3_calls()}"
        )
        assert "arm record UNAVAILABLE" in output, output
        assert "GRANT PROBE INDETERMINATE" in stubbed_box.arm_record.read_text(), (
            "the box that learned least about its grant also kept no record of that:\n"
            + stubbed_box.arm_record.read_text()
        )
        assert len(stubbed_box.shutdown_attempts()) == 1, (
            f"an unresolved identity cost the box its shutdown: {stubbed_box.shutdown_attempts()}"
        )

    def test_an_absent_cli_does_not_cost_the_box_its_arming(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """Any AMI outside the DLAMI family is this box, and publishing runs before any work does."""
        stubbed_box.set_shutdown_exit(0)
        env = {**stubbed_box.env_without_aws(), "IDLE_WATCHDOG_S3_DEST": STUB_S3_DEST}
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner", env=env)

        assert run.process.wait(timeout=15) == 0, run.output()
        output = run.output()
        assert "armed: terminate after 0 idle minutes" in output, (
            f"a missing CLI stopped the watchdog arming:\n{output}"
        )
        assert "arm record FAILED" in output, output
        assert len(stubbed_box.shutdown_attempts()) == 1, (
            f"a missing CLI cost the box its shutdown: {stubbed_box.shutdown_attempts()}"
        )

    def test_the_record_stops_growing_once_the_watch_starts(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog
    ) -> None:
        """Left recording, the "arm record" becomes the whole log and is not one.

        It would also leak /var/log for the life of a box, and these boxes run for days. Two reads
        seconds apart, because a single read cannot tell a bounded record from one still growing.
        """
        run = launch(
            "--minutes",
            "1",
            "--poll-seconds",
            "1",
            "--user",
            "runner",
            env=stubbed_box.env_publishing_to(),
        )
        time.sleep(2.5)
        first_size = stubbed_box.arm_record.stat().st_size
        time.sleep(2.0)

        assert len(run.idle_poll_lines()) >= 2, (
            f"the loop never polled, so this proves nothing about growth:\n{run.output()}"
        )
        record = stubbed_box.arm_record.read_text()
        assert not IDLE_POLL_PATTERN.search(record), (
            f"the arm record kept accumulating the watch's own log:\n{record}"
        )
        assert stubbed_box.arm_record.stat().st_size == first_size, (
            f"the arm record is still growing: {first_size} then "
            f"{stubbed_box.arm_record.stat().st_size} bytes"
        )

    def test_an_unwritable_record_path_does_not_stop_the_arming(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog, tmp_path: Path
    ) -> None:
        """One complaint, not one redirection error per log line for the life of the box."""
        stubbed_box.set_shutdown_exit(0)
        unwritable = tmp_path / "no-such-directory" / "arm.txt"
        env = stubbed_box.env_publishing_to(IDLE_WATCHDOG_ARM_RECORD=str(unwritable))
        run = launch("--minutes", "0", "--poll-seconds", "1", "--user", "runner", env=env)

        assert run.process.wait(timeout=15) == 0, run.output()
        output = run.output()
        assert output.count("arm record UNWRITABLE") == 1, (
            f"an unwritable record path was reported {output.count('arm record UNWRITABLE')} "
            f"times rather than once:\n{output}"
        )
        assert "armed: terminate after 0 idle minutes" in output, (
            f"an unwritable record path stopped the watchdog arming:\n{output}"
        )
        assert stubbed_box.s3_calls() == [], (
            f"an absent record was uploaded anyway: {stubbed_box.s3_calls()}"
        )
        assert len(stubbed_box.shutdown_attempts()) == 1, stubbed_box.shutdown_attempts()

    def test_a_destination_that_is_not_an_s3_url_is_refused_rather_than_copied_locally(
        self, stubbed_box: StubbedBox, launch: LaunchWatchdog, tmp_path: Path
    ) -> None:
        """``aws s3 cp`` between two local paths exits 0, which would report PUBLISHED into the void.

        A kit that sets the variable to a filesystem path -- its run directory, say -- would then get
        a green publication line for a verdict still sitting on a disk about to be destroyed. That is
        the reassuring-message failure the grant probe itself exists to prevent, so it is refused by
        shape before the call.
        """
        stubbed_box.set_work_in_flight(present=True)
        local_dest = tmp_path / "not-a-bucket"
        run = launch(
            "--minutes",
            "0",
            "--poll-seconds",
            "1",
            "--user",
            "runner",
            env=stubbed_box.env_publishing_to(str(local_dest)),
        )
        time.sleep(2.5)

        output = run.output()
        assert stubbed_box.s3_calls() == [], (
            f"a local path was handed to the uploader: {stubbed_box.s3_calls()}"
        )
        assert "not an s3:// URL" in output, output
        assert "arm record PUBLISHED" not in output, (
            f"a local copy was reported as a published record:\n{output}"
        )
