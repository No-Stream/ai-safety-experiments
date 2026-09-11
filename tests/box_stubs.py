"""A stubbed rented EC2 box, for the killswitch suite that drives a real script against it.

``scripts/idle_watchdog.sh`` (layer 2 of the killswitch, which terminates a box that has stopped
doing work) reads IMDS, calls ``ec2:TerminateInstances`` on the identity it finds there, and
publishes a record under ``IDLE_WATCHDOG_S3_DEST``. Nothing in this repo launches an instance, so the
only way to know the script does what its header claims is to run it with ``curl``, ``aws`` and
``shutdown`` shimmed onto PATH and watch what it does. Every test drives the real script; none of
them read its source.

The single most load-bearing thing here is ``assert_stubs_resolve``, and it is not hygiene. **This
dev box is itself an EC2 instance**, so a ``curl`` that reached the real IMDS would hand a real
instance id to a real ``aws ec2 terminate-instances`` and a test would delete the machine it runs
on. Both binaries are therefore shimmed in the base fixture rather than per test, and the assertion
refuses to run at all if either resolves anywhere but this test's own directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

# The identity the stubbed IMDS serves, and the token it demands before serving it.
INSTANCE = "i-0123456789abcdef0"
REGION = "us-west-2"
IMDS_TOKEN = "stub-imdsv2-token"

# What a PATH must still carry for these scripts to run once ``aws`` is taken off it.
BOX_COREUTILS = ("cat", "cut", "date", "dirname", "flock", "sleep", "timeout")

# A stall long enough that elapsed time alone separates a bounded call from a completed one.
STALL_SECONDS = 6
TEST_CALL_BOUND_SECONDS = "1"

# Never a real bucket, and it may never become one: this repo's history is published.
STUB_S3_DEST = "s3://stub-bucket/games-run"

# Verbatim CLI verdicts. A passing dry run is reported as an *error*, hence text and not exit status.
DRY_RUN_ALLOWED_MESSAGE = (
    "An error occurred (DryRunOperation) when calling the TerminateInstances operation: "
    "Request would have succeeded, but DryRun flag is set."
)
UNAUTHORIZED_MESSAGE = (
    "An error occurred (UnauthorizedOperation) when calling the TerminateInstances operation: "
    "You are not authorized to perform this operation."
)
THROTTLED_MESSAGE = (
    "An error occurred (RequestLimitExceeded) when calling the TerminateInstances operation: "
    "Request limit exceeded."
)

# One CLI, three independently canned answers, recording the argv and the uploaded bytes; see install_aws.
STUB_AWS = r"""printf '%s\n' "$*" >> {aws_log}
case "$1" in
  s3)
    kind=s3
    if [ -f "$3" ]; then cat "$3" >> {s3_payload_log}; fi
    ;;
  *)
    kind=terminate
    for arg in "$@"; do
      case "$arg" in
        --dry-run) kind=dry-run ;;
      esac
    done
    ;;
esac
sleep "$(cat {aws_dir}/$kind.delay)"
cat {aws_dir}/$kind.out >&2
exit "$(cat {aws_dir}/$kind.exit)"
"""

# An IMDS that behaves the way these boxes are launched; see StubbedBox.set_instance_identity.
STUB_CURL = r"""printf '%s\n' "$*" >> {curl_log}
authenticated=no
for arg in "$@"; do
  case "$arg" in
    "X-aws-ec2-metadata-token: {token}") authenticated=yes ;;
  esac
done
case "$*" in
  *"/latest/api/token")
    if [ -f {token_refused} ]; then exit 7; fi
    printf '%s' "{token}"
    ;;
  *"/latest/meta-data/instance-id")
    [ "$authenticated" = yes ] || exit 22
    cat {instance_file}
    ;;
  *"/latest/meta-data/placement/region")
    [ "$authenticated" = yes ] || exit 22
    cat {region_file}
    ;;
  *)
    printf 'stub curl: unexpected request: %s\n' "$*" >&2
    exit 6
    ;;
esac"""


def write_stub(directory: Path, name: str, body: str) -> None:
    """Install an executable shim, replaced atomically so a live loop never reads half a file."""
    scratch = directory / f".{name}.new"
    scratch.write_text(f"#!/bin/sh\n{body}\n")
    scratch.chmod(0o755)
    scratch.replace(directory / name)


@dataclass(frozen=True)
class StubbedBox:
    """The surface the killswitch script touches: IMDS, the AWS CLI, and ``shutdown``."""

    bin_dir: Path
    imds_dir: Path
    aws_dir: Path
    curl_log: Path
    aws_log: Path
    s3_payload_log: Path
    shutdown_log: Path
    env: dict[str, str]

    def env_publishing_to(self, dest: str = STUB_S3_DEST, **overrides: str) -> dict[str, str]:
        """An env asking the script under test to publish its record under ``dest``.

        The base env deliberately leaves the destination unset, because that is what every launcher
        kit outside this repo does until its author adds the variable, and the script has to keep
        working in that state.
        """
        return {**self.env, "IDLE_WATCHDOG_S3_DEST": dest, **overrides}

    def set_instance_identity(self, *, instance: str = INSTANCE, region: str = REGION) -> None:
        """Install an IMDS that serves this identity, and only to an authenticated caller.

        IMDSv2 is required, matching these boxes' ``HttpTokens=required``: a metadata GET carrying
        no token header answers 401, which ``curl -f`` reports as exit 22 with an empty body. That
        distinction is the whole reason to stub IMDS rather than hand the values over some simpler
        way, because an implementation that reverted to an unauthenticated GET would otherwise read
        empty strings and look merely unlucky instead of wrong.
        """
        (self.imds_dir / "instance-id").write_text(instance)
        (self.imds_dir / "region").write_text(region)
        write_stub(
            self.bin_dir,
            "curl",
            STUB_CURL.format(
                curl_log=self.curl_log,
                token=IMDS_TOKEN,
                token_refused=self.imds_dir / "token-refused",
                instance_file=self.imds_dir / "instance-id",
                region_file=self.imds_dir / "region",
            ),
        )

    def set_imds_reachable(self, *, reachable: bool) -> None:
        """Decide whether the token PUT succeeds, which is how a non-EC2 host or a blocked IMDS reads."""
        refused = self.imds_dir / "token-refused"
        if reachable:
            refused.unlink(missing_ok=True)
        else:
            refused.write_text("the token endpoint is unreachable\n")

    def install_aws(self) -> None:
        """Shim the CLI so that a dry run, a real call and an upload answer independently.

        They have to, because one CLI serves callers whose real answers differ in opposite
        directions: a dry run that passes is reported as the DryRunOperation *error* and exits
        non-zero, while a real terminate that succeeds exits zero. A single canned response could
        only model one of them, and modelling the wrong one would hide precisely the inversion the
        watchdog's grant probe exists to avoid. The upload is a third: it is the one call made while
        a box is healthy, so a test needs it to fail or hang without disturbing the other answers.
        """
        write_stub(
            self.bin_dir,
            "aws",
            STUB_AWS.format(
                aws_log=self.aws_log,
                aws_dir=self.aws_dir,
                s3_payload_log=self.s3_payload_log,
            ),
        )

    def set_dry_run_response(
        self, *, stderr: str, exit_code: int = 254, delay_seconds: float = 0
    ) -> None:
        """Answer `--dry-run` calls this way. The exit code defaults to the CLI's non-zero one."""
        self._write_aws_response(
            "dry-run", stderr=stderr, exit_code=exit_code, delay_seconds=delay_seconds
        )

    def set_terminate_response(
        self, *, stderr: str = "", exit_code: int = 0, delay_seconds: float = 0
    ) -> None:
        """Answer calls that carry no `--dry-run`, the ones that would really destroy a box."""
        self._write_aws_response(
            "terminate", stderr=stderr, exit_code=exit_code, delay_seconds=delay_seconds
        )

    def set_s3_response(
        self, *, stderr: str = "", exit_code: int = 0, delay_seconds: float = 0
    ) -> None:
        """Answer `aws s3` calls this way, which is how a record is published."""
        self._write_aws_response(
            "s3", stderr=stderr, exit_code=exit_code, delay_seconds=delay_seconds
        )

    def _write_aws_response(
        self, kind: str, *, stderr: str, exit_code: int, delay_seconds: float
    ) -> None:
        """Newline-terminate the message the way the real CLI does.

        Without it the caller's next log line runs onto the end of the CLI's, so the log would read
        differently here than it does on a box, which is the thing these tests are for.
        """
        (self.aws_dir / f"{kind}.out").write_text(f"{stderr}\n" if stderr else "")
        (self.aws_dir / f"{kind}.exit").write_text(str(exit_code))
        (self.aws_dir / f"{kind}.delay").write_text(str(delay_seconds))

    def env_without_aws(self) -> dict[str, str]:
        """An env whose PATH keeps the other shims but carries no ``aws`` at all.

        The CLI lives in /usr/bin on this box, alongside the coreutils these scripts need, so its
        absence has to be built by curating a PATH rather than by trimming a directory off one --
        and by dropping this fixture's own shim, which the assertion at the end is what caught.
        """
        (self.bin_dir / "aws").unlink(missing_ok=True)
        curated = self.imds_dir.parent / "coreutils-only"
        curated.mkdir(exist_ok=True)
        for tool in BOX_COREUTILS:
            resolved = shutil.which(tool)
            assert resolved is not None, f"{tool} is not on PATH, so this fixture cannot be built"
            (curated / tool).unlink(missing_ok=True)
            (curated / tool).symlink_to(resolved)

        path = f"{self.bin_dir}{os.pathsep}{curated}"
        assert shutil.which("aws", path=path) is None, (
            "the curated PATH still resolves aws, so a test of its absence would pass vacuously"
        )
        return {**self.env, "PATH": path}

    def curl_calls(self) -> list[str]:
        """Every ``curl`` invocation made against this box, in order."""
        if not self.curl_log.exists():
            return []
        return self.curl_log.read_text().splitlines()

    def aws_calls(self) -> list[str]:
        """Every ``aws`` invocation made against this box, in order."""
        if not self.aws_log.exists():
            return []
        return self.aws_log.read_text().splitlines()

    def dry_run_calls(self) -> list[str]:
        """The calls that carried ``--dry-run``, which AWS answers without performing anything."""
        return [call for call in self.aws_calls() if "--dry-run" in call]

    def terminating_calls(self) -> list[str]:
        """Every ``ec2`` call that did NOT carry ``--dry-run``, so every call that could kill a box.

        Scoped to ``ec2`` because these scripts also call ``aws s3``, and a harmless upload counted
        here would read as something having destroyed a live run.
        """
        return [
            call for call in self.aws_calls() if call.startswith("ec2 ") and "--dry-run" not in call
        ]

    def s3_calls(self) -> list[str]:
        """Every ``aws s3`` invocation made against this box, in order."""
        return [call for call in self.aws_calls() if call.startswith("s3 ")]

    def s3_payloads(self) -> str:
        """The bytes the stubbed CLI was actually handed to upload, concatenated.

        Asserted instead of the argv because a call that uploads an empty or wrong-window file is
        indistinguishable from a correct one at the argv, and the whole point of publishing is the
        content: an object that exists but does not carry the verdict is the same void with a
        reassuring listing.
        """
        if not self.s3_payload_log.exists():
            return ""
        return self.s3_payload_log.read_text()

    def set_shutdown_exit(self, code: int) -> None:
        """Decide whether the shimmed ``shutdown`` succeeds, recording every attempt either way."""
        write_stub(
            self.bin_dir,
            "shutdown",
            f'echo "$@" >> {self.shutdown_log}\nexit {code}',
        )

    def shutdown_attempts(self) -> list[str]:
        """Every ``shutdown`` invocation made against this box, in order."""
        if not self.shutdown_log.exists():
            return []
        return self.shutdown_log.read_text().splitlines()


def assert_stubs_resolve(box: StubbedBox, *commands: str) -> None:
    """Refuse to run if any of these resolves outside this test's shim directory.

    Not paranoia about the shims in general, but about ``curl``, ``aws`` and ``shutdown``
    specifically. This dev box is itself an EC2 instance, so a ``curl`` that reached the real IMDS
    would hand a real instance id to a real ``aws ec2 terminate-instances`` and the suite would
    delete the machine it runs on.
    """
    for command in commands:
        resolved = shutil.which(command, path=box.env["PATH"])
        assert resolved == str(box.bin_dir / command), (
            f"{command} resolves to {resolved}, not to this test's shim. Refusing to run a "
            "destructive path against the real one."
        )


def install_shared_stubs(box: StubbedBox) -> None:
    """Wire the shims and canned responses the suite starts from, then check none is the real one.

    The default posture is a healthy, fully granted box: IMDS serves this box's identity, the dry run
    passes (reported as an error, exiting non-zero, which is what the real CLI does), and a real
    terminate succeeds.
    """
    box.set_instance_identity()
    box.set_shutdown_exit(1)
    box.install_aws()
    box.set_dry_run_response(stderr=DRY_RUN_ALLOWED_MESSAGE)
    box.set_terminate_response()
    box.set_s3_response()
    assert_stubs_resolve(box, "curl", "aws", "shutdown")


@dataclass
class ObservedProcess:
    """A script under observation, plus whatever it has logged so far."""

    process: subprocess.Popen[str]
    log_path: Path

    def output(self) -> str:
        """Everything the script has written to stdout and stderr so far."""
        return self.log_path.read_text()

    def is_running(self) -> bool:
        """Whether the script is still running."""
        return self.process.poll() is None


def launch_observed(
    command: Sequence[str], *, env: Mapping[str, str], log_path: Path
) -> subprocess.Popen[str]:
    """Start a repo script with both streams collected into ``log_path``.

    A file rather than a pipe: these scripts are long-lived loops that are read while they run, and
    reading a pipe from the test process would block until they exit.
    """
    handle = log_path.open("w")
    return subprocess.Popen(  # noqa: S603 - repo script, literal arguments
        list(command),
        env=dict(env),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
