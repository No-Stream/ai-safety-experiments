"""Drive the five ship-tree tools against a throwaway git repository and a stubbed `aws`.

The tools exist because a rented GPU box could run stale code with every sha256 check green, and the
first fix for that -- content-addressing every path by git tree hash -- stranded a post-reclaim
relaunch in a junk prefix when a peer commit moved HEAD. Code is therefore never tied to a pin, a
commit or a hash. What this suite pins, in the order the flow runs:

- RUN IDENTITY IS AN EXPLICIT NAME. A launch requires --run-name, every S3 path lives under
  <prefix>/<run-name>/, and no path carries a commit-sha-shaped component, so a relaunch under the
  same name lands in the same prefix however far HEAD has moved.
- THE CURRENT WORKING TREE SHIPS, DIRTY AND UNTRACKED CODE INCLUDED, ALWAYS. Nothing refuses a launch
  for a tree that is dirty or has moved; the archive covers what `git archive HEAD` cannot.
- The working-state hash IGNORES exactly what .gitignore ignores, and leaves the SHARED .git/index
  byte-identical, since a stray `git add` here would destroy a peer session's in-flight work. Empty
  input is refused rather than hashed.
- PROVENANCE IS A RECORD, NEVER A GATE. Every sitting appends HEAD sha, `git status --porcelain` and
  the tarball checksum, and nothing reads those records to refuse anything.
- INTEGRITY CHECKS STAY. Staging reads every object back from its consumption key and the box-side
  guard fails closed on a swapped object, a missing object and a dropped file. Corruption is not
  freshness: no peer's edit can trip these.

Everything runs in a throwaway repository under /var/tmp rather than the shared tree, which is what
makes the dirty-code tests possible without corrupting what a peer session is holding. Every test
drives the real script through `bash` rather than reading its source, and `bash` explicitly, because
zsh applies history modifiers to `$var:path` and once mangled such an expansion into a false zero.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIP_TREE = REPO_ROOT / "scripts" / "ship_tree.sh"
STAGE = REPO_ROOT / "scripts" / "stage_ship_tree.sh"
RENDER = REPO_ROOT / "scripts" / "render_box_userdata.sh"
GUARD = REPO_ROOT / "scripts" / "box_ship_guard.sh"
LAUNCH = REPO_ROOT / "scripts" / "launch_gpu_box.sh"
TOOLS = (SHIP_TREE, STAGE, RENDER, GUARD, LAUNCH)

# /var/tmp, never /tmp, whose RAM-backed tmpfs once ran out of inodes and failed every shell box-wide.
SCRATCH_ROOT = "/var/tmp"  # noqa: S108
# Absolute, because a conda env on the inherited PATH resolves `python3` and friends somewhere else.
BASH = "/bin/bash"
GIT = "/usr/bin/git"
S3_PREFIX = "s3://a-test-bucket/ship-tests"
RUN_NAME = "wave-reskin"
RUN_PREFIX = f"{S3_PREFIX}/{RUN_NAME}"
S3_REGION = "us-west-2"
INSTANCE_PROFILE = "a-test-instance-profile"
# Obviously fake ids; the fixture expects the first, so every launch here passes the gate by default.
RENTAL_ACCOUNT = "111111111111"
ANOTHER_ACCOUNT = "222222222222"
SHA256_OF_NOTHING = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
# The staged HEAD a render gets with no launcher around it; the only one of the seven with no sample.
SAMPLE_HEAD_SHA = "0123456789abcdef0123456789abcdef01234567"

# Records instead of powering off; the guard resolves `shutdown` from PATH so this can sit in front.
STUB_SHUTDOWN = """#!/bin/sh
printf '%s\\n' "$*" >> "$SHIP_TEST_CONTROL/shutdown.log"
"""

# Records the seconds asked for and returns at once, so durations are asserted without waiting them.
STUB_SLEEP = """#!/bin/sh
printf '%s\\n' "$*" >> "$SHIP_TEST_CONTROL/sleep.log"
"""

# Enough s3 and ec2 for the whole launch flow; python, because it must parse run-instances' own argv.
STUB_AWS = r'''#!/usr/bin/env python3
"""A fake `aws` for tests. Answers from files under $SHIP_TEST_CONTROL and logs every call.

It is faithful to the one property of the real CLI the launcher's walk depends on: botocore's
standard retry mode sends a request it deems transient up to AWS_MAX_ATTEMPTS times (default 3),
and EC2 serves an InsufficientInstanceCapacity decline as an HTTP 500 and a throttle, a server-side
failure and a connection failure as retryable, so with the variable unset every declined row is THREE
RunInstances calls with backoff. The double records one attempt per call the real CLI would make,
which is what lets a test see whether the launcher switched the CLI's retries off, and what makes
dropping that switch fail the suite. Every record also carries the --client-token the launcher sent
and the wall-clock interval the call spanned, so a test can see whether a same-row retry kept its
token and whether two regions' probes overlapped, without timing the whole launch.
"""
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

CONTROL = pathlib.Path(os.environ["SHIP_TEST_CONTROL"])
BUCKETS = CONTROL / "s3"
CAPACITY = (
    "An error occurred (InsufficientInstanceCapacity) when calling the RunInstances operation{}: "
    "There is no capacity available that satisfies your request."
)
NOT_CAPACITY = (
    "An error occurred (UnauthorizedOperation) when calling the RunInstances operation: "
    "You are not authorized to perform this operation."
)
# The transient answers botocore would have retried, phrased as the CLI phrases them: a parsed error
# code in parentheses for the API's own answers, botocore's exception text for the connection-level
# ones (which never carry a "reached max retries" suffix, since no error response was parsed).
TRANSIENT_ANSWERS = {
    "throttle": (
        "An error occurred (RequestLimitExceeded) when calling the RunInstances operation{retries}: "
        "Request limit exceeded."
    ),
    "internal-error": (
        "An error occurred (InternalError) when calling the RunInstances operation{retries}: "
        "An internal error has occurred. Retry your request, but if the problem persists, contact us."
    ),
    "service-unavailable": (
        "An error occurred (ServiceUnavailable) when calling the RunInstances operation{retries}: "
        "The server is overloaded and can't handle the request."
    ),
    "connection-refused": 'Could not connect to the endpoint URL: "https://ec2.{region}.amazonaws.com/"',
    "read-timeout": 'Read timeout on endpoint URL: "https://ec2.{region}.amazonaws.com/"',
}
DRY_RUN_OK = (
    "An error occurred (DryRunOperation) when calling the RunInstances operation: "
    "Request would have succeeded, but DryRun flag is set."
)
BOTOCORE_DEFAULT_MAX_ATTEMPTS = 3

argv = sys.argv[1:]
with (CONTROL / "calls.log").open("a") as log:
    log.write(" ".join(argv) + "\n")


def flag(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def local_path(uri):
    return BUCKETS / uri[len("s3://") :]


def s3_cp():
    source, destination = argv[2], argv[3]
    if source.startswith("s3://"):
        origin = local_path(source)
        if not origin.is_file():
            sys.stderr.write(f"fatal error: An error occurred (404) when calling the "
                             f"HeadObject operation: Key {source} does not exist\n")
            return 1
        pathlib.Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
        return 0
    target = local_path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if (CONTROL / "swallow-uploads").exists():
        return 0
    shutil.copy2(source, target)
    if (CONTROL / "corrupt-uploads").exists():
        target.write_bytes(target.read_bytes() + b"one extra byte")
    if "/provenance/" in destination and (CONTROL / "corrupt-provenance-uploads").exists():
        target.write_bytes(target.read_bytes() + b"one extra byte")
    return 0


def probe_delay(region):
    """Pretend each probe call is a slow round trip, for every region or for one region only."""
    for override in (CONTROL / f"probe-delay-seconds-{region}", CONTROL / "probe-delay-seconds"):
        if override.exists():
            time.sleep(float(override.read_text().strip()))
            return


def kill_probe_subshell():
    """SIGKILL the launcher's probe subshell this call runs under, the way an OOM kill would.

    Walked up from this process rather than named, because the launcher exposes no test hook: every
    ancestor bash whose own parent is also bash is a subshell of the launcher (the command
    substitution around this call, then the region's probe), while the launcher itself has the test
    runner for a parent and is left alone.
    """
    def parent_and_name(pid):
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
        name = stat[stat.index("(") + 1 : stat.rindex(")")]
        return int(stat.rsplit(")", 1)[1].split()[1]), name

    victims = []
    pid = os.getppid()
    while True:
        parent, name = parent_and_name(pid)
        if name != "bash" or parent_and_name(parent)[1] != "bash":
            break
        victims.append(pid)
        pid = parent
    for victim in victims:
        os.kill(victim, signal.SIGKILL)


def describe_images():
    region = flag("--region", "")
    if (CONTROL / f"kill-probe-{region}").exists():
        kill_probe_subshell()
    probe_delay(region)
    # An absent image prints the API's own empty answer on STDOUT and succeeds, while a denial writes
    # to stderr and fails: the launcher has to tell those apart, and the whole point of the second is
    # that on stdout alone it looks exactly like the first.
    failure = CONTROL / f"describe-images-fails-{region}"
    if failure.exists():
        sys.stderr.write(failure.read_text())
        return 254
    if (CONTROL / f"no-ami-{region}").exists():
        print("None")
        return 0
    print("ami-" + hashlib.sha256(region.encode()).hexdigest()[:17])
    return 0


def matches(path, instance_type, region):
    if not path.exists():
        return False
    wanted = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return f"{instance_type}|{region}" in wanted


def locked_control():
    """One lock for every counter and record under the control directory.

    The launcher probes every region at once, so two copies of this double can be answering at the
    same moment; without the lock both would count the same number of existing records and one
    would overwrite the other's.
    """
    lock = (CONTROL / "control.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def bump_counter(name):
    """Increment a named counter and return its new value."""
    path = CONTROL / f"counter-{name}"
    with locked_control():
        value = int(path.read_text()) + 1 if path.exists() else 1
        path.write_text(str(value))
    return value


def botocore_attempts():
    """How many times the real CLI would send a request botocore deems transient."""
    return int(os.environ.get("AWS_MAX_ATTEMPTS", str(BOTOCORE_DEFAULT_MAX_ATTEMPTS)))


def retries_suffix(attempts):
    """botocore's own annotation on an error that exhausted its retry budget."""
    return f" (reached max retries: {attempts - 1})" if attempts > 1 else ""


def record_attempts(record, attempts):
    """One record per attempt the real CLI would make, numbered in the order they were made."""
    calls = CONTROL / "run-instances"
    with locked_control():
        calls.mkdir(exist_ok=True)
        first = len(list(calls.iterdir()))
        finished_at = time.time()
        for attempt in range(1, attempts + 1):
            (calls / f"{first + attempt - 1:04d}.json").write_text(
                json.dumps(record | {"attempt": attempt, "finished_at": finished_at})
            )


def transient_answer_for(mode, instance_type, region):
    """The kind of transient answer this call gets, or None once the row's budget is spent.

    The plan is one line per row, `mode|type|region|times|kind`: `mode` is real or dry, because the
    launcher's dry runs go through the same run-instances path and have their own retry to prove.
    """
    plan = CONTROL / "transient-on"
    if not plan.exists():
        return None
    for line in plan.read_text().splitlines():
        if not line.strip():
            continue
        wanted_mode, wanted_type, wanted_region, budget, kind = line.strip().split("|")
        if (wanted_mode, wanted_type, wanted_region) != (mode, instance_type, region):
            continue
        served = bump_counter(f"transients-served-{mode}-{instance_type}-{region}")
        return kind if served <= int(budget) else None
    return None


def transient_message(kind, region, attempts):
    return TRANSIENT_ANSWERS[kind].format(retries=retries_suffix(attempts), region=region)


def granted_by_invocation(invocation):
    """Capacity that appears on the Nth real call, whichever row makes it: a walk that lands late."""
    override = CONTROL / "grant-on-real-invocation"
    return override.exists() and invocation == int(override.read_text().strip())


def run_instances():
    started_at = time.time()
    region = flag("--region", "")
    instance_type = flag("--instance-type", "")
    user_data = flag("--user-data", "")
    tag_spec = flag("--tag-specifications", "")
    dry_run = "--dry-run" in argv
    record = {
        "argv": argv,
        "region": region,
        "instance_type": instance_type,
        "dry_run": dry_run,
        "market": flag("--instance-market-options", "ondemand"),
        "aws_max_attempts": os.environ.get("AWS_MAX_ATTEMPTS"),
        "client_token": flag("--client-token"),
        "started_at": started_at,
        "user_data": pathlib.Path(user_data[len("file://") :]).read_text(),
        "tag_specification": json.loads(pathlib.Path(tag_spec[len("file://") :]).read_text()),
    }
    if dry_run:
        probe_delay(region)
    else:
        invocation = bump_counter("real-invocations")
        if matches(CONTROL / "fail-hard-on", instance_type, region):
            record_attempts(record, 1)
            override = CONTROL / "fail-hard-message"
            sys.stderr.write((override.read_text() if override.exists() else NOT_CAPACITY) + "\n")
            return 254
    transient = transient_answer_for("dry" if dry_run else "real", instance_type, region)
    if transient is not None:
        attempts = botocore_attempts()
        record_attempts(record, attempts)
        sys.stderr.write(transient_message(transient, region, attempts) + "\n")
        return 254
    if dry_run:
        # A DryRunOperation is an HTTP 412 and a bad parameter a 400; botocore retries neither.
        record_attempts(record, 1)
        override = CONTROL / f"dry-run-fails-{region}"
        sys.stderr.write((override.read_text() if override.exists() else DRY_RUN_OK) + "\n")
        return 254
    on_spot = "--instance-market-options" in argv
    granted = (
        matches(CONTROL / "succeed-on", instance_type, region)
        or (on_spot and matches(CONTROL / "succeed-on-spot", instance_type, region))
        or granted_by_invocation(invocation)
    )
    if granted:
        record_attempts(record, 1)
        print("i-0" + hashlib.sha256(f"{instance_type}{region}".encode()).hexdigest()[:16])
        return 0
    attempts = botocore_attempts()
    record_attempts(record, attempts)
    sys.stderr.write(CAPACITY.format(retries_suffix(attempts)) + "\n")
    return 254


def describe_instance_attribute():
    override = CONTROL / "shutdown-behavior"
    print(override.read_text().strip() if override.exists() else "terminate")
    return 0


def sts_get_caller_identity():
    """The account the credentials resolve to, or the CLI's own credential failure."""
    if (CONTROL / "sts-denied").exists():
        sys.stderr.write(
            "An error occurred (ExpiredToken) when calling the GetCallerIdentity operation: "
            "The security token included in the request is expired\n"
        )
        return 254
    if flag("--query") != "Account":
        sys.stderr.write("stub aws: get-caller-identity without --query Account\n")
        return 64
    override = CONTROL / "caller-account"
    print(override.read_text().strip() if override.exists() else "111111111111")
    return 0


ROUTES = {
    ("s3", "cp"): s3_cp,
    ("ec2", "describe-images"): describe_images,
    ("ec2", "run-instances"): run_instances,
    ("ec2", "describe-instance-attribute"): describe_instance_attribute,
    ("sts", "get-caller-identity"): sts_get_caller_identity,
    ("ec2", "create-tags"): lambda: 0,
    ("ec2", "terminate-instances"): lambda: 0,
}

handler = ROUTES.get((argv[0], argv[1]) if len(argv) > 1 else None)
if handler is None:
    sys.stderr.write(f"stub aws: unexpected call: {' '.join(argv)}\n")
    sys.exit(64)
sys.exit(handler())
'''

# All seven launcher-supplied values, because the render refuses a --set the template cannot consume.
TEMPLATE = """#!/bin/bash
shutdown -h +@DEADMAN_MINUTES@ "ship-test dead-man"
set -ux
SHIP_TREE=@SHIP_TREE@
SHIP_CODE_SHA256=@SHIP_CODE_SHA256@
SHIP_TREE_DIGEST=@SHIP_TREE_DIGEST@
SHIP_S3_KEY=@SHIP_S3_KEY@
SHIP_S3_REGION=@SHIP_S3_REGION@
SHIP_HEAD_SHA=@SHIP_HEAD_SHA@
RUN_NAME=@RUN_NAME@
RUN_PREFIX=@RUN_PREFIX@
SHIP_S3_PREFIX=@SHIP_S3_PREFIX@
SHIP_EXTRACT_DIR=/home/ubuntu/repo
IDLE_WATCHDOG_S3_DEST=@IDLE_WATCHDOG_S3_DEST@
export IDLE_WATCHDOG_S3_DEST
@SHIP_GUARD@
bash $SHIP_EXTRACT_DIR/scripts/ec2_boot_patch.sh
$SHIP_EXTRACT_DIR/scripts/idle_watchdog.sh --minutes 45 --done-marker /home/ubuntu/AGENDA_DONE &
"""
WATCHDOG_LINE = "$SHIP_EXTRACT_DIR/scripts/idle_watchdog.sh --minutes 45 --done-marker /home/ubuntu/AGENDA_DONE &"
WATCHDOG_LINE_WITHOUT_MARKER = WATCHDOG_LINE.replace(" --done-marker /home/ubuntu/AGENDA_DONE", "")

CANDIDATES = """# instance_type|region|security_group|subnet
g7e.2xlarge|us-west-2|sg-0aaaaaaaaaaaaaaa1|subnet-0aaaaaaaaaaaaaaa1
g7e.4xlarge|us-west-2|sg-0aaaaaaaaaaaaaaa1|subnet-0aaaaaaaaaaaaaaa1
g7e.2xlarge|us-east-2|sg-0bbbbbbbbbbbbbbb1|subnet-0bbbbbbbbbbbbbbb1
"""


@dataclass
class Finished:
    """One finished run of a tool: its exit code and everything it printed."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    def value(self, key: str) -> str:
        """One key=value line from the tool's machine-readable stdout."""
        for line in self.stdout.splitlines():
            name, separator, value = line.partition("=")
            if separator and name == key:
                return value
        raise AssertionError(f"no {key}= line in stdout:\n{self.stdout}")

    def refused_with(self, fragment: str) -> None:
        assert self.returncode != 0, f"expected a refusal but it succeeded:\n{self.output}"
        assert fragment in self.output, f"refusal did not mention {fragment!r}:\n{self.output}"
        # A `die` inside a command substitution prints a second, wrong diagnosis; one misdirected a fix.
        assert self.output.count("FATAL:") <= 2, (
            f"a single refusal produced more than one diagnosis, which means an error path ran "
            f"inside a command substitution:\n{self.output}"
        )


@dataclass
class ShipRepo:
    """A throwaway git repository carrying copies of the real tools."""

    path: Path
    control: Path
    env: dict[str, str]

    def run(
        self,
        tool: Path,
        *arguments: str,
        extra_env: dict[str, str] | None = None,
        umask: str | None = None,
    ) -> Finished:
        """Run one tool the way a launcher would: through bash, from inside the repo.

        ``umask`` makes that umask ambient for the tool, the way a root cloud-init shell or an
        operator's restrictive login would; the tools must not let it reach their digests.
        """
        command = [BASH, str(self.path / "scripts" / tool.name), *arguments]
        if umask is not None:
            command = [BASH, "-c", f'umask {umask}; exec "$@"', "umask-wrapper", *command]
        environment = dict(self.env)
        if extra_env:
            environment.update(extra_env)
        completed = subprocess.run(  # noqa: S603 - repo scripts, literal arguments
            command,
            capture_output=True,
            text=True,
            cwd=self.path,
            env=environment,
            check=False,
        )
        return Finished(completed.returncode, completed.stdout, completed.stderr)

    def git(self, *arguments: str) -> str:
        completed = subprocess.run(  # noqa: S603 - repo scripts, literal arguments
            [GIT, "-C", str(self.path), *arguments],
            capture_output=True,
            text=True,
            env=self.env,
            check=True,
        )
        return completed.stdout

    def ship_hash(self) -> str:
        finished = self.run(SHIP_TREE, "--print-hash")
        assert finished.returncode == 0, finished.output
        return finished.stdout.strip()

    def index_fingerprint(self) -> str:
        return hashlib.sha256((self.path / ".git" / "index").read_bytes()).hexdigest()

    def write(self, relative: str, text: str) -> Path:
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    def object_at(self, uri: str) -> Path:
        return self.control / "s3" / uri[len("s3://") :]

    def aws_calls(self) -> list[str]:
        log = self.control / "calls.log"
        return log.read_text().splitlines() if log.exists() else []

    def run_instance_calls(self) -> list[dict[str, Any]]:
        directory = self.control / "run-instances"
        if not directory.exists():
            return []
        return [json.loads(path.read_text()) for path in sorted(directory.iterdir())]

    def succeed_on(self, instance_type: str, region: str) -> None:
        (self.control / "succeed-on").write_text(f"{instance_type}|{region}\n")

    def fail_ami_lookup_in(self, region: str, message: str) -> None:
        """Make the region's AMI lookup fail the way a DescribeImages denial or an SCP block does.

        Empty stdout plus a message on stderr, which on stdout alone is byte-for-byte what "this
        region has no such image" looks like -- the whole reason the launcher must keep the stderr
        rather than let the failure read as an absence and skip the region.
        """
        (self.control / f"describe-images-fails-{region}").write_text(message + "\n")

    def no_ami_in(self, region: str) -> None:
        """Answer the region's AMI lookup with the API's own empty result, as a region with none does."""
        (self.control / f"no-ami-{region}").write_text("")

    def succeed_on_spot_only(self, instance_type: str, region: str) -> None:
        """Grant capacity to a spot row alone, which is what exhausts the on-demand pass."""
        (self.control / "succeed-on-spot").write_text(f"{instance_type}|{region}\n")

    def fail_hard_on(self, instance_type: str, region: str, message: str | None = None) -> None:
        """Answer the row's real call with a non-capacity error, UnauthorizedOperation by default."""
        (self.control / "fail-hard-on").write_text(f"{instance_type}|{region}\n")
        if message is not None:
            (self.control / "fail-hard-message").write_text(message)

    def dry_run_fails_in(self, region: str, message: str) -> None:
        (self.control / f"dry-run-fails-{region}").write_text(message)

    def set_shutdown_behavior(self, value: str) -> None:
        (self.control / "shutdown-behavior").write_text(value)

    def credentials_resolve_to(self, account: str) -> None:
        """Make sts get-caller-identity answer with this account, as another profile's would."""
        (self.control / "caller-account").write_text(account + "\n")

    def deny_sts(self) -> None:
        """Make get-caller-identity fail the way expired or absent credentials do."""
        (self.control / "sts-denied").write_text("")

    def sts_calls(self) -> list[str]:
        return [call for call in self.aws_calls() if call.startswith("sts get-caller-identity")]

    def s3_uploads(self) -> list[str]:
        """Every s3 cp that would have written to the bucket: the launch's first paid call."""
        return [call for call in self.aws_calls() if call.startswith("s3 cp ") and " s3://" in call]

    def shutdown_calls(self) -> list[str]:
        log = self.control / "shutdown.log"
        return log.read_text().splitlines() if log.exists() else []

    def fail_transiently(self, instance_type: str, region: str, times: int, kind: str) -> None:
        """Answer the row's first ``times`` real calls with a transient failure, then fall through.

        ``kind`` names one of the double's TRANSIENT_ANSWERS: throttle, internal-error,
        service-unavailable, connection-refused or read-timeout.
        """
        with (self.control / "transient-on").open("a") as plan:
            plan.write(f"real|{instance_type}|{region}|{times}|{kind}\n")

    def throttle(self, instance_type: str, region: str, times: int) -> None:
        """Answer the row's first ``times`` real calls with RequestLimitExceeded, then fall through."""
        self.fail_transiently(instance_type, region, times, kind="throttle")

    def throttle_dry_run(self, instance_type: str, region: str, times: int) -> None:
        """Throttle the region's dry run, which the launcher makes with the region's first row."""
        with (self.control / "transient-on").open("a") as plan:
            plan.write(f"dry|{instance_type}|{region}|{times}|throttle\n")

    def grant_on_real_invocation(self, invocation: int) -> None:
        """Capacity appears on the Nth real run-instances call the launcher makes, whatever the row."""
        (self.control / "grant-on-real-invocation").write_text(f"{invocation}\n")

    def delay_probes(self, seconds: float) -> None:
        """Make every describe-images and dry-run call take this long, as a slow round trip would."""
        (self.control / "probe-delay-seconds").write_text(f"{seconds}\n")

    def delay_probes_in(self, region: str, seconds: float) -> None:
        """Make only this region's probe calls slow, as one hung or distant endpoint would."""
        (self.control / f"probe-delay-seconds-{region}").write_text(f"{seconds}\n")

    def kill_probe_in(self, region: str) -> None:
        """Have the region's first describe-images kill the probe subshell it runs under."""
        (self.control / f"kill-probe-{region}").write_text("")

    def sleeps(self) -> list[str]:
        """Every duration the launcher asked `sleep` for, in order; the stub never actually waits."""
        log = self.control / "sleep.log"
        return log.read_text().splitlines() if log.exists() else []

    def real_run_instance_calls(self) -> list[dict[str, Any]]:
        return [call for call in self.run_instance_calls() if not call["dry_run"]]

    def dry_run_instance_calls(self) -> list[dict[str, Any]]:
        return [call for call in self.run_instance_calls() if call["dry_run"]]

    def staged_code_keys(self) -> list[str]:
        """Every code tarball staged under the run prefix, one per sitting."""
        return sorted(
            f"s3://{path.relative_to(self.control / 's3')}"
            for path in (self.control / "s3").rglob("code.tar.gz")
        )


@pytest.fixture
def ship_repo() -> Iterator[ShipRepo]:
    """A throwaway repo with the real tools copied in, a stubbed aws, and a recording shutdown.

    The tools resolve their repository from their own location, so copying them into a throwaway tree
    is what points them at that tree. Nothing here touches the real shared repository.
    """
    root = Path(tempfile.mkdtemp(prefix="ship-tree-test-", dir=SCRATCH_ROOT))
    repo = root / "repo"
    control = root / "control"
    binaries = root / "bin"
    for directory in (repo / "scripts", control / "s3", binaries):
        directory.mkdir(parents=True)

    stub_aws = binaries / "aws"
    stub_aws.write_text(STUB_AWS)
    stub_aws.chmod(0o755)
    stub_shutdown = binaries / "shutdown"
    stub_shutdown.write_text(STUB_SHUTDOWN)
    stub_shutdown.chmod(0o755)
    stub_sleep = binaries / "sleep"
    stub_sleep.write_text(STUB_SLEEP)
    stub_sleep.chmod(0o755)

    environment = dict(os.environ)
    environment.update(
        {
            "PATH": f"{binaries}:{environment['PATH']}",
            "SHIP_TEST_CONTROL": str(control),
            "GIT_AUTHOR_NAME": "ship test",
            "GIT_AUTHOR_EMAIL": "ship-test@invalid",
            "GIT_COMMITTER_NAME": "ship test",
            "GIT_COMMITTER_EMAIL": "ship-test@invalid",
        }
    )
    environment.pop("GIT_INDEX_FILE", None)
    # Dropped so a launch here is judged against the stub's answer alone, never a real box's profile.
    environment.pop("AWS_PROFILE", None)
    environment["SHIP_EXPECTED_ACCOUNT"] = RENTAL_ACCOUNT

    for tool in TOOLS:
        shutil.copy2(tool, repo / "scripts" / tool.name)
    (repo / ".gitignore").write_text("artifacts/\ndocs/scratch/\n")
    (repo / "runner.py").write_text("VERSION = 'original'\n")
    (repo / "scripts" / "ec2_boot_patch.sh").write_text("#!/bin/bash\necho patched\n")
    (repo / "scripts" / "idle_watchdog.sh").write_text("#!/bin/bash\necho watching\n")
    # The real repository ships one, and following it rather than recording it misses a swapped link.
    (repo / "NOTES.md").write_text("notes\n")
    (repo / "NOTES-LINK.md").symlink_to("NOTES.md")
    for ignored in ("artifacts/big.bin", "docs/scratch/plan.md"):
        target = repo / ignored
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ignored\n")

    subprocess.run(  # noqa: S603 - repo scripts, literal arguments
        [GIT, "init", "-q", str(repo)], check=True, env=environment
    )
    tracked = [
        ".gitignore",
        "runner.py",
        "NOTES.md",
        "NOTES-LINK.md",
        *(f"scripts/{tool.name}" for tool in TOOLS),
        "scripts/ec2_boot_patch.sh",
        "scripts/idle_watchdog.sh",
    ]
    subprocess.run(  # noqa: S603 - repo scripts, literal arguments
        [GIT, "-C", str(repo), "add", "--", *tracked], check=True, env=environment
    )
    subprocess.run(  # noqa: S603 - repo scripts, literal arguments
        [GIT, "-C", str(repo), "commit", "-qm", "initial"], check=True, env=environment
    )

    kit = root / "kit"
    kit.mkdir()
    (kit / "ud-template.sh").write_text(TEMPLATE)
    (kit / "candidates.txt").write_text(CANDIDATES)

    yield ShipRepo(path=repo, control=control, env=environment)
    shutil.rmtree(root, ignore_errors=True)


def kit_dir(ship_repo: ShipRepo) -> Path:
    return ship_repo.path.parent / "kit"


def stage(ship_repo: ShipRepo, *extra: str) -> Finished:
    return ship_repo.run(
        STAGE,
        "--run-prefix",
        RUN_PREFIX,
        "--region",
        S3_REGION,
        "--stage-dir",
        str(ship_repo.path.parent / "stage"),
        *extra,
    )


def render(ship_repo: ShipRepo, *extra: str, template: Path | None = None) -> Finished:
    """Render as a launch would: every value the launcher supplies itself, in the same seven --set flags.

    All seven and not a convenient three, because the render refuses a --set the template carries no
    placeholder for, and a helper passing fewer than a launch does would make the suite disagree with
    the launcher about which templates are renderable.
    """
    return ship_repo.run(
        RENDER,
        "--template",
        str(template or kit_dir(ship_repo) / "ud-template.sh"),
        "--out",
        str(ship_repo.path.parent / "user-data.sh"),
        "--stage-dir",
        str(ship_repo.path.parent / "stage"),
        "--set",
        f"SHIP_S3_REGION={S3_REGION}",
        "--set",
        "DEADMAN_MINUTES=180",
        "--set",
        f"IDLE_WATCHDOG_S3_DEST={RUN_PREFIX}",
        "--set",
        f"RUN_NAME={RUN_NAME}",
        "--set",
        f"RUN_PREFIX={RUN_PREFIX}",
        "--set",
        f"SHIP_S3_PREFIX={S3_PREFIX}",
        "--set",
        f"SHIP_HEAD_SHA={SAMPLE_HEAD_SHA}",
        *extra,
    )


LAUNCH_REQUIRED_ARGUMENTS = (
    "--template",
    "{kit}/ud-template.sh",
    "--candidates",
    "{kit}/candidates.txt",
    "--label",
    "a-test-box",
    "--purpose",
    "exercising the ship flow",
    "--deadman-minutes",
    "180",
    "--s3-prefix",
    S3_PREFIX,
    "--s3-region",
    S3_REGION,
    "--instance-profile",
    INSTANCE_PROFILE,
)


def launch(
    ship_repo: ShipRepo,
    *extra: str,
    run_name: str | None = RUN_NAME,
    extra_env: dict[str, str] | None = None,
) -> Finished:
    """Drive the launcher; ``run_name=None`` omits --run-name to prove the refusal."""
    arguments = [argument.format(kit=kit_dir(ship_repo)) for argument in LAUNCH_REQUIRED_ARGUMENTS]
    arguments += ["--run-dir", str(ship_repo.path.parent / "run")]
    if run_name is not None:
        arguments += ["--run-name", run_name]
    return ship_repo.run(LAUNCH, *arguments, *extra, extra_env=extra_env)


def sha_shaped_segments(path: str) -> list[str]:
    """The commit-sha-shaped components of an S3 key or prefix, empty when it is clean.

    A component is sha-shaped when, split on the separators run names and keys are built from, it
    is 7 to 64 characters drawn wholly from lowercase hex AND carries at least one letter -- the
    letter requirement is what keeps a timestamp like 20260831 from reading as a hash while still
    catching every value `git rev-parse` actually emits.
    """
    tokens = re.split(r"[/._\-]", path)
    return [
        token
        for token in tokens
        if re.fullmatch(r"[0-9a-f]{7,64}", token) and re.search(r"[a-f]", token)
    ]


def base_ami(region: str) -> str:
    """The image id the stubbed describe-images reports for a region, derived the same way it is."""
    return "ami-" + hashlib.sha256(region.encode()).hexdigest()[:17]


def launch_tags(ship_repo: ShipRepo) -> dict[str, str]:
    """Every tag EC2 was handed, flattened across the walk's run-instances calls."""
    return {
        str(tag["Key"]): str(tag["Value"])
        for call in ship_repo.run_instance_calls()
        for specification in call["tag_specification"]
        for tag in specification["Tags"]
    }


def extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 - repo scripts, literal arguments
        ["/usr/bin/tar", "xzf", str(archive), "-C", str(destination)], check=True
    )
    return destination


class TestWorkingStateHash:
    """What the hash covers, which is the whole requirement: ship what is on disk, not what is in HEAD."""

    def test_a_dirty_tracked_file_moves_the_hash_and_travels_in_the_archive(
        self, ship_repo: ShipRepo
    ) -> None:
        clean = ship_repo.ship_hash()
        ship_repo.write("runner.py", "VERSION = 'dirty and unstaged'\n")
        dirty = ship_repo.ship_hash()
        assert dirty != clean, "uncommitted work did not move the hash, so it would not ship"

        archive = ship_repo.path.parent / "dirty.tar.gz"
        assert ship_repo.run(SHIP_TREE, "--archive", str(archive)).returncode == 0
        tree = extract(archive, ship_repo.path.parent / "dirty-extract")
        assert (tree / "runner.py").read_text() == "VERSION = 'dirty and unstaged'\n", (
            "the archive carried HEAD's bytes rather than the working tree's, which is exactly what "
            "`git archive HEAD` does and what this primitive exists to replace"
        )

        ship_repo.write("runner.py", "VERSION = 'original'\n")
        assert ship_repo.ship_hash() == clean, "restoring the file did not restore the hash"

    def test_an_untracked_file_git_does_not_ignore_moves_the_hash(
        self, ship_repo: ShipRepo
    ) -> None:
        clean = ship_repo.ship_hash()
        newcomer = ship_repo.write("brand_new.py", "print('never committed')\n")
        assert ship_repo.ship_hash() != clean
        newcomer.unlink()
        assert ship_repo.ship_hash() == clean

    def test_a_deleted_tracked_file_moves_the_hash(self, ship_repo: ShipRepo) -> None:
        clean = ship_repo.ship_hash()
        (ship_repo.path / "runner.py").unlink()
        assert ship_repo.ship_hash() != clean, (
            "a file deleted on disk still shipped, so the box would run code the operator removed"
        )

    def test_ignored_paths_never_move_the_hash(self, ship_repo: ShipRepo) -> None:
        clean = ship_repo.ship_hash()
        ship_repo.write("artifacts/big.bin", "a different run's output\n")
        ship_repo.write("docs/scratch/plan.md", "rewritten notes\n")
        ship_repo.write("artifacts/brand-new-run/metrics.json", "{}\n")
        assert ship_repo.ship_hash() == clean, (
            "gitignored churn moved the hash, which would mint a new S3 key on every run and destroy "
            "staging idempotence"
        )

    def test_the_provenance_block_names_every_file_that_differs_from_head(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.write("runner.py", "VERSION = 'dirty'\n")
        ship_repo.write("brand_new.py", "x = 1\n")
        finished = ship_repo.run(SHIP_TREE, "--print-hash")
        assert "DIRTY" in finished.stderr
        assert "runner.py" in finished.stderr
        assert "brand_new.py" in finished.stderr

    def test_facts_report_head_and_dirtiness(self, ship_repo: ShipRepo) -> None:
        clean = ship_repo.run(SHIP_TREE, "--facts")
        assert clean.value("dirty") == "false"
        assert clean.value("head") == ship_repo.git("rev-parse", "HEAD").strip()
        ship_repo.write("runner.py", "VERSION = 'dirty'\n")
        assert ship_repo.run(SHIP_TREE, "--facts").value("dirty") == "true"


class TestTheSharedIndexIsProtected:
    """The one footgun that could destroy a peer session's in-flight work."""

    def test_a_normal_run_leaves_the_shared_index_byte_identical(self, ship_repo: ShipRepo) -> None:
        before = ship_repo.index_fingerprint()
        ship_repo.write("runner.py", "VERSION = 'dirty'\n")
        ship_repo.write("brand_new.py", "x = 1\n")
        ship_repo.ship_hash()
        assert ship_repo.index_fingerprint() == before, (
            "the shared .git/index changed, which means a peer session's staged work was rewritten"
        )

    def test_pointing_the_tool_at_the_shared_index_is_refused(self, ship_repo: ShipRepo) -> None:
        before = ship_repo.index_fingerprint()
        finished = ship_repo.run(
            SHIP_TREE,
            "--print-hash",
            extra_env={"GIT_INDEX_FILE": str(ship_repo.path / ".git" / "index")},
        )
        finished.refused_with("inside the repository")
        assert ship_repo.index_fingerprint() == before

    def test_an_index_path_inside_the_working_tree_is_refused(self, ship_repo: ShipRepo) -> None:
        finished = ship_repo.run(
            SHIP_TREE,
            "--print-hash",
            extra_env={"GIT_INDEX_FILE": str(ship_repo.path / "sneaky.idx")},
        )
        finished.refused_with("inside the repository")

    def test_a_relative_index_path_is_refused(self, ship_repo: ShipRepo) -> None:
        ship_repo.run(
            SHIP_TREE, "--print-hash", extra_env={"GIT_INDEX_FILE": "relative.idx"}
        ).refused_with("is relative")


class TestEmptyInputIsRefused:
    """sha256 of nothing is a plausible hash that every other empty thing also has."""

    def test_the_digest_of_a_directory_with_no_files_is_refused(self, ship_repo: ShipRepo) -> None:
        empty = ship_repo.path.parent / "empty-tree"
        empty.mkdir()
        finished = ship_repo.run(SHIP_TREE, "--digest", str(empty))
        finished.refused_with("no files")
        assert SHA256_OF_NOTHING not in finished.stdout, (
            "the tool reported the digest of empty input, which any other empty tree also matches"
        )

    def test_a_working_state_listing_no_files_is_refused(self, ship_repo: ShipRepo) -> None:
        """git's empty tree is a plausible 40-hex hash that archives to a valid, empty tarball.

        Reached with a minimal repository whose single tracked file has been deleted from disk: the
        listing legitimately produces nothing, and ``git write-tree`` answers with the empty tree
        rather than failing, which is what a broken listing pipeline would also look like.
        """
        minimal = ship_repo.path.parent / "minimal"
        (minimal / "tools").mkdir(parents=True)
        shutil.copy2(SHIP_TREE, minimal / "tools" / SHIP_TREE.name)
        (minimal / "only.txt").write_text("the only tracked file\n")
        (minimal / ".git-init-marker").write_text("")
        subprocess.run(  # noqa: S603 - repo scripts, literal arguments
            [GIT, "init", "-q", str(minimal)], check=True, env=ship_repo.env
        )
        (minimal / ".git" / "info").mkdir(exist_ok=True)
        (minimal / ".git" / "info" / "exclude").write_text("tools/\n.git-init-marker\n")
        subprocess.run(  # noqa: S603 - repo scripts, literal arguments
            [GIT, "-C", str(minimal), "add", "--", "only.txt"], check=True, env=ship_repo.env
        )
        subprocess.run(  # noqa: S603 - repo scripts, literal arguments
            [GIT, "-C", str(minimal), "commit", "-qm", "only"], check=True, env=ship_repo.env
        )
        (minimal / "only.txt").unlink()
        completed = subprocess.run(  # noqa: S603 - repo scripts, literal arguments
            [BASH, str(minimal / "tools" / SHIP_TREE.name), "--print-hash"],
            capture_output=True,
            text=True,
            env=ship_repo.env,
            check=False,
        )
        Finished(completed.returncode, completed.stdout, completed.stderr).refused_with(
            "empty tree"
        )


class TestArchiveIsReproducible:
    """A tarball that changes across a second boundary cannot be content-addressed."""

    def test_the_same_tree_archives_to_the_same_bytes_twice(self, ship_repo: ShipRepo) -> None:
        first = ship_repo.run(SHIP_TREE, "--archive", str(ship_repo.path.parent / "one.tar.gz"))
        second = ship_repo.run(SHIP_TREE, "--archive", str(ship_repo.path.parent / "two.tar.gz"))
        assert first.value("archive_sha256") == second.value("archive_sha256")
        assert first.value("commit") == second.value("commit"), (
            "the ship commit is not a pure function of its tree, so re-freezing mints a new id"
        )

    def test_the_ship_commit_is_kept_at_a_ref_so_the_objects_survive_gc(
        self, ship_repo: ShipRepo
    ) -> None:
        finished = ship_repo.run(SHIP_TREE, "--freeze")
        tree = finished.value("tree")
        assert ship_repo.git("rev-parse", f"refs/ship/{tree}^{{tree}}").strip() == tree, (
            "the frozen ref does not point at the tree it names"
        )

    def test_a_tree_that_was_never_frozen_cannot_be_replayed(self, ship_repo: ShipRepo) -> None:
        ship_repo.run(SHIP_TREE, "--tree", "0" * 40, "--print-hash").refused_with(
            "does not exist in this repository"
        )


class TestDigestMatchesTheBoxSideCopy:
    """The producer and the box compute the digest with two separate implementations, by design."""

    def test_both_implementations_agree_on_the_same_tree(self, ship_repo: ShipRepo) -> None:
        archive = ship_repo.path.parent / "code.tar.gz"
        ship_repo.run(SHIP_TREE, "--archive", str(archive))
        tree = extract(archive, ship_repo.path.parent / "extract")
        producer = ship_repo.run(SHIP_TREE, "--digest", str(tree)).stdout.strip()

        # Given the producer's answer as its expectation, so passing IS the agreement assertion.
        guard = ship_repo.run(
            GUARD,
            "verify-digest",
            extra_env=guard_environment(ship_repo, digest=producer, extract_dir=str(tree)),
        )
        assert guard.returncode == 0, (
            f"the box-side digest disagrees with the producer's, so the two implementations have "
            f"drifted:\n{guard.output}"
        )

    def test_dropping_one_file_changes_the_digest(self, ship_repo: ShipRepo) -> None:
        archive = ship_repo.path.parent / "code.tar.gz"
        ship_repo.run(SHIP_TREE, "--archive", str(archive))
        tree = extract(archive, ship_repo.path.parent / "extract")
        before = ship_repo.run(SHIP_TREE, "--digest", str(tree)).stdout.strip()
        (tree / "runner.py").unlink()
        assert ship_repo.run(SHIP_TREE, "--digest", str(tree)).stdout.strip() != before

    def test_a_swapped_symlink_changes_the_digest(self, ship_repo: ShipRepo) -> None:
        archive = ship_repo.path.parent / "code.tar.gz"
        ship_repo.run(SHIP_TREE, "--archive", str(archive))
        tree = extract(archive, ship_repo.path.parent / "extract")
        before = ship_repo.run(SHIP_TREE, "--digest", str(tree)).stdout.strip()
        link = tree / "NOTES-LINK.md"
        link.unlink()
        link.symlink_to("runner.py")
        assert ship_repo.run(SHIP_TREE, "--digest", str(tree)).stdout.strip() != before, (
            "a symlink repointed at another file left the digest unchanged, so the digest is "
            "following links instead of recording them"
        )


class TestExtractionIsModeDeterministic:
    """The digest pins each file's mode, so extraction must land the same modes for every extractor.

    The failure pinned here refused a real launch: `git archive` records group-writable modes (git's
    tar.umask defaults to 002), the staging digest extracted as a NON-root user whose umask turned
    them into 644/755, and the box guard extracted as ROOT, which preserves recorded modes -- so the
    two digests disagreed on permission bits alone and the first box launched through this flow
    halted on its fuse over a perfectly good tree. Root cannot be impersonated in a test, but
    `tar -p` reproduces root's default exactly (preserve recorded modes instead of applying the
    umask), and a hostile umask reproduces the other half.
    """

    def test_the_archive_bytes_do_not_depend_on_the_machines_git_config(
        self, ship_repo: ShipRepo
    ) -> None:
        before = ship_repo.run(SHIP_TREE, "--archive", str(ship_repo.path.parent / "a.tar.gz"))
        ship_repo.git("config", "tar.umask", "0077")
        after = ship_repo.run(SHIP_TREE, "--archive", str(ship_repo.path.parent / "b.tar.gz"))
        assert before.value("archive_sha256") == after.value("archive_sha256"), (
            "a machine-level tar.umask changed the archive bytes, so the same tree would "
            "content-address to two different sha256s on two machines"
        )

    def test_a_mode_preserving_extraction_digests_to_the_staged_pin(
        self, ship_repo: ShipRepo
    ) -> None:
        """`tar -p` is what root tar does by default, and it must agree with the staged digest."""
        environment = staged_guard_environment(ship_repo)
        preserved = ship_repo.path.parent / "extract-preserving-modes"
        preserved.mkdir()
        subprocess.run(  # noqa: S603 - repo scripts, literal arguments
            [
                "/usr/bin/tar",
                "-p",
                "-xzf",
                str(ship_repo.object_at(environment["SHIP_S3_KEY"])),
                "-C",
                str(preserved),
            ],
            check=True,
        )
        observed = ship_repo.run(SHIP_TREE, "--digest", str(preserved)).stdout.strip()
        assert observed == environment["SHIP_TREE_DIGEST"], (
            "an extraction that preserves the archive's recorded modes (root tar's default) "
            "disagrees with the staged digest, so a root box would refuse a perfectly good tree"
        )

    @pytest.mark.parametrize("hostile_umask", ["077", "002"])
    def test_the_guard_verifies_under_any_ambient_umask(
        self, ship_repo: ShipRepo, hostile_umask: str
    ) -> None:
        environment = staged_guard_environment(ship_repo)
        finished = ship_repo.run(GUARD, "all", extra_env=environment, umask=hostile_umask)
        assert finished.returncode == 0, (
            f"the guard let the ambient umask {hostile_umask} reach its extraction, so its digest "
            f"depends on who runs it rather than on what was shipped:\n{finished.output}"
        )

    def test_the_staging_digest_does_not_depend_on_the_operators_umask(
        self, ship_repo: ShipRepo
    ) -> None:
        reference = stage(ship_repo)
        assert reference.returncode == 0, reference.output
        hostile = ship_repo.run(
            STAGE,
            "--run-prefix",
            RUN_PREFIX,
            "--region",
            S3_REGION,
            "--stage-dir",
            str(ship_repo.path.parent / "stage-hostile-umask"),
            umask="077",
        )
        assert hostile.returncode == 0, hostile.output
        assert hostile.value("tree_digest") == reference.value("tree_digest"), (
            "the same tree staged under two umasks pinned two different digests, so which operator "
            "stages decides whether the box's guard passes"
        )


def guard_environment(  # noqa: PLR0913 - one keyword per value the rendered user-data sets
    ship_repo: ShipRepo,
    *,
    tree: str = "0" * 40,
    code_sha256: str = "0" * 64,
    digest: str = "0" * 64,
    key: str = f"{S3_PREFIX}/tree/{'0' * 40}/code.tar.gz",
    extract_dir: str | None = None,
    archive_path: str | None = None,
) -> dict[str, str]:
    """The variables the rendered user-data sets before it runs the guard."""
    return {
        "SHIP_TREE": tree,
        "SHIP_CODE_SHA256": code_sha256,
        "SHIP_TREE_DIGEST": digest,
        "SHIP_S3_KEY": key,
        "SHIP_S3_REGION": S3_REGION,
        "SHIP_EXTRACT_DIR": extract_dir or str(ship_repo.path.parent / "box-repo"),
        "SHIP_ARCHIVE_PATH": archive_path or str(ship_repo.path.parent / "box-code.tar.gz"),
    }


def staged_guard_environment(ship_repo: ShipRepo) -> dict[str, str]:
    """Stage the working state and return the environment a faithful render would give the guard."""
    staged = stage(ship_repo)
    assert staged.returncode == 0, staged.output
    manifest = (ship_repo.path.parent / "stage" / "ship-manifest.txt").read_text()
    values = dict(
        line.split("=", 1) for line in manifest.splitlines() if line.count("=") and "=" in line
    )
    return guard_environment(
        ship_repo,
        tree=staged.value("ship_tree"),
        code_sha256=values["archive_sha256"],
        digest=values["tree_digest"],
        key=staged.value("s3_key"),
    )


class TestStaging:
    """Named-run keys with no hash component, and a readback from the consumption key."""

    def test_the_key_lives_under_the_run_prefix_and_every_object_reads_back_identically(
        self, ship_repo: ShipRepo
    ) -> None:
        finished = stage(ship_repo)
        assert finished.returncode == 0, finished.output
        sitting = finished.value("sitting")
        assert finished.value("s3_key") == f"{RUN_PREFIX}/sittings/{sitting}/code.tar.gz"
        assert ship_repo.object_at(finished.value("s3_key")).is_file()
        provenance_key = finished.value("provenance_key")
        assert provenance_key == f"{RUN_PREFIX}/provenance/{sitting}-launch.txt"
        assert ship_repo.object_at(provenance_key).is_file()
        assert "reading every object back" in finished.stderr
        assert "SKEW" not in finished.output
        assert "MISSING" not in finished.output

    def test_no_staged_key_carries_a_commit_sha_shaped_component(self, ship_repo: ShipRepo) -> None:
        """The owner's ruling made mechanical: no pins, commits or hashes in any path."""
        finished = stage(ship_repo)
        assert finished.returncode == 0, finished.output
        for key in (finished.value("s3_key"), finished.value("provenance_key")):
            assert sha_shaped_segments(key) == [], (
                f"{key} carries a commit-sha-shaped component, which ties the run's identity to a "
                f"code state and is exactly what stranded a relaunch in a junk prefix"
            )

    def test_an_upload_that_reports_success_but_writes_nothing_is_caught(
        self, ship_repo: ShipRepo
    ) -> None:
        """A green upload against a key nothing consumes happened for real in this series.

        The producer-side exit code said success and the box found nothing there, which is why the
        readback happens from the exact key the box fetches rather than from the local file.
        """
        (ship_repo.control / "swallow-uploads").write_text("")
        finished = stage(ship_repo)
        assert finished.returncode != 0
        assert "MISSING at the consumption key" in finished.output
        assert "Do NOT" in finished.output

    def test_an_upload_that_lands_different_bytes_is_caught(self, ship_repo: ShipRepo) -> None:
        (ship_repo.control / "corrupt-uploads").write_text("")
        finished = stage(ship_repo)
        assert finished.returncode != 0
        assert "SKEW" in finished.output

    def test_a_tree_that_moves_between_stagings_is_shipped_not_refused(
        self, ship_repo: ShipRepo
    ) -> None:
        """The refusal this replaces is the direction the owner ruled out.

        The earlier design pinned a launch to one tree hash and refused when a peer edit moved it;
        now the working tree as it stands at each staging ships, and both sittings keep their own
        key and record.
        """
        first = stage(ship_repo)
        assert first.returncode == 0, first.output
        ship_repo.write("runner.py", "VERSION = 'a peer session edited this'\n")
        second = stage(ship_repo)
        assert second.returncode == 0, (
            f"a launch failed because the tree changed under it, which is now expected, welcome "
            f"behavior:\n{second.output}"
        )
        assert first.value("s3_key") != second.value("s3_key"), (
            "two sittings shared one code key, so the newer upload destroyed the record of what "
            "the earlier sitting ran"
        )
        assert ship_repo.object_at(first.value("s3_key")).is_file()
        newer = extract(
            ship_repo.object_at(second.value("s3_key")), ship_repo.path.parent / "second-extract"
        )
        assert (newer / "runner.py").read_text() == "VERSION = 'a peer session edited this'\n"

    def test_each_sitting_appends_its_own_provenance_record(self, ship_repo: ShipRepo) -> None:
        """Multi-sitting runs ship newer code deliberately; the records are how analysis reconciles."""
        first = stage(ship_repo)
        ship_repo.write("runner.py", "VERSION = 'the relaunch ships this'\n")
        second = stage(ship_repo)
        assert first.value("provenance_key") != second.value("provenance_key"), (
            "two sittings shared one provenance key, so the relaunch overwrote the record of what "
            "the first sitting ran"
        )
        assert ship_repo.object_at(first.value("provenance_key")).is_file()
        assert ship_repo.object_at(second.value("provenance_key")).is_file()

    def test_the_provenance_record_says_what_bytes_ran(self, ship_repo: ShipRepo) -> None:
        """HEAD sha, the dirty state, and the tarball's checksum -- a record, never a gate."""
        ship_repo.write("runner.py", "VERSION = 'dirty and shipping'\n")
        finished = stage(ship_repo)
        assert finished.returncode == 0, finished.output
        record = ship_repo.object_at(finished.value("provenance_key")).read_text()
        head = ship_repo.git("rev-parse", "HEAD").strip()
        staged = ship_repo.object_at(finished.value("s3_key")).read_bytes()
        assert f"head_sha={head}" in record
        assert "dirty=true" in record
        assert f"archive_sha256={hashlib.sha256(staged).hexdigest()}" in record
        assert f"s3_key={finished.value('s3_key')}" in record
        assert f"sitting={finished.value('sitting')}" in record
        assert "git_status= M runner.py" in record, (
            "the record does not carry the git status --porcelain view of the tree that shipped"
        )
        assert "dirty_path=M\trunner.py" in record

    def test_a_provenance_record_that_lands_corrupted_still_fails_the_gate(
        self, ship_repo: ShipRepo
    ) -> None:
        """Demoting provenance to a record must not have cost its readback its teeth."""
        (ship_repo.control / "corrupt-provenance-uploads").write_text("")
        finished = stage(ship_repo)
        assert finished.returncode != 0, (
            "a provenance record that landed with different bytes passed the readback gate"
        )
        assert "SKEW" in finished.output
        assert "-launch.txt" in finished.output, (
            f"the SKEW did not name the provenance key:\n{finished.output}"
        )

    def test_a_data_artifact_is_pinned_by_sha256_in_the_record(self, ship_repo: ShipRepo) -> None:
        corpus = ship_repo.path.parent / "cases.json"
        corpus.write_text('{"items": []}\n')
        finished = stage(ship_repo, "--data", str(corpus))
        assert finished.returncode == 0, finished.output
        record = (ship_repo.path.parent / "stage" / "ship-manifest.txt").read_text()
        expected = hashlib.sha256(corpus.read_bytes()).hexdigest()
        assert f"data=cases.json {expected}" in record
        assert ship_repo.object_at(
            f"{RUN_PREFIX}/sittings/{finished.value('sitting')}/data/cases.json"
        ).is_file()


class TestRender:
    """The render is where a template's missing guarantee has to become a refusal."""

    def test_it_pins_the_key_the_staging_wrote(self, ship_repo: ShipRepo) -> None:
        staged = stage(ship_repo)
        finished = render(ship_repo)
        assert finished.returncode == 0, finished.output
        rendered = (ship_repo.path.parent / "user-data.sh").read_text()
        assert f"SHIP_S3_KEY={staged.value('s3_key')}" in rendered
        assert f"SHIP_TREE={staged.value('ship_tree')}" in rendered
        assert f"SHIP_CODE_SHA256={staged.value('archive_sha256')}" in rendered
        assert f"SHIP_TREE_DIGEST={staged.value('tree_digest')}" in rendered

    def test_it_inlines_the_guard_and_runs_it(self, ship_repo: ShipRepo) -> None:
        stage(ship_repo)
        render(ship_repo)
        rendered = (ship_repo.path.parent / "user-data.sh").read_text()
        assert "sha256sum -c -" in rendered
        assert "bash /var/tmp/box_ship_guard.sh all" in rendered
        assert "@SHIP_GUARD@" not in rendered

    @pytest.mark.parametrize(
        "placeholder",
        [
            "@SHIP_TREE@",
            "@SHIP_CODE_SHA256@",
            "@SHIP_TREE_DIGEST@",
            "@SHIP_S3_KEY@",
            "@SHIP_GUARD@",
        ],
    )
    def test_a_template_missing_any_placeholder_is_refused(
        self, ship_repo: ShipRepo, placeholder: str
    ) -> None:
        stage(ship_repo)
        maimed = ship_repo.path.parent / "maimed-template.sh"
        maimed.write_text(TEMPLATE.replace(placeholder, "REMOVED"))
        render(ship_repo, template=maimed).refused_with(f"no {placeholder} placeholder")

    def test_a_placeholder_with_no_value_is_refused_rather_than_shipped_as_text(
        self, ship_repo: ShipRepo
    ) -> None:
        stage(ship_repo)
        extra = ship_repo.path.parent / "extra-template.sh"
        extra.write_text(TEMPLATE + "RUNNER=@RUNNER_KEY@\n")
        render(ship_repo, template=extra).refused_with("unsubstituted placeholders")

    def test_a_tree_that_moved_after_staging_still_renders(self, ship_repo: ShipRepo) -> None:
        """The render attests to the staged bytes, so a later edit is no reason to refuse.

        The refusal this replaces pinned the render to a recomputed tree hash and went red when a
        peer edit moved it -- the direction the owner ruled out. The record stays honest without
        it: the rendered user-data names the staged sitting's key and checksums, and the edit that
        arrived after staging simply ships on the next sitting.
        """
        staged = stage(ship_repo)
        ship_repo.write("runner.py", "VERSION = 'moved on'\n")
        finished = render(ship_repo)
        assert finished.returncode == 0, (
            f"a render failed because the tree changed after staging, which is now expected, "
            f"welcome behavior:\n{finished.output}"
        )
        rendered = (ship_repo.path.parent / "user-data.sh").read_text()
        assert f"SHIP_S3_KEY={staged.value('s3_key')}" in rendered

    def test_a_value_containing_slashes_survives_substitution(self, ship_repo: ShipRepo) -> None:
        stage(ship_repo)
        finished = render(ship_repo)
        assert finished.returncode == 0, finished.output
        rendered = (ship_repo.path.parent / "user-data.sh").read_text()
        assert f"IDLE_WATCHDOG_S3_DEST={RUN_PREFIX}" in rendered

    def test_a_user_data_over_the_run_instances_raw_limit_is_refused(
        self, ship_repo: ShipRepo
    ) -> None:
        """RunInstances caps the RAW bytes at 16,384, and the raw cap binds before any encoded one.

        The check this replaces measured 25,600 bytes of base64, a figure that is not a RunInstances
        bound at all (16,384 raw bytes can never encode past 21,848), so a user-data of 17,000 raw
        bytes sailed through the render and was then rejected once per candidate -- which during a
        capacity walk reads exactly like an insufficient-capacity decline.
        """
        stage(ship_repo)
        bulky = ship_repo.path.parent / "bulky-template.sh"
        bulky.write_text(TEMPLATE + "PAYLOAD=@PAYLOAD@\n")
        rendered_empty = render(ship_repo, "--set", "PAYLOAD=", template=bulky)
        assert rendered_empty.returncode == 0, rendered_empty.output
        base = len((ship_repo.path.parent / "user-data.sh").read_bytes())
        at_limit = render(ship_repo, "--set", f"PAYLOAD={'x' * (16384 - base)}", template=bulky)
        assert at_limit.returncode == 0, (
            f"exactly 16,384 raw bytes is within the API's limit and must render:\n{at_limit.output}"
        )
        over_by_one = "x" * (16384 - base + 1)
        render(ship_repo, "--set", f"PAYLOAD={over_by_one}", template=bulky).refused_with("16384")

    def test_a_placeholder_named_in_a_comment_is_refused_rather_than_expanded(
        self, ship_repo: ShipRepo
    ) -> None:
        """A comment mentioning a placeholder must not receive its value.

        On a real launch a template comment named the guard placeholder, the render expanded the
        whole guard body into it, and everything past the body's first line landed as live code
        ABOVE the SHIP_* assignments it depends on.
        """
        stage(ship_repo)
        chatty = ship_repo.path.parent / "chatty-template.sh"
        chatty.write_text(
            TEMPLATE.replace(
                "@SHIP_GUARD@",
                "# the render inlines @SHIP_GUARD@ on the next line\n@SHIP_GUARD@",
            )
        )
        finished = render(ship_repo, template=chatty)
        finished.refused_with("comment")
        rendered = ship_repo.path.parent / "user-data.sh"
        assert not rendered.exists() or "halt_unrunnable" not in rendered.read_text(), (
            "the guard body was expanded into the comment after all"
        )

    def test_a_multi_line_body_embedded_mid_line_is_refused(self, ship_repo: ShipRepo) -> None:
        stage(ship_repo)
        inline = ship_repo.path.parent / "inline-template.sh"
        inline.write_text(TEMPLATE.replace("@SHIP_GUARD@", 'GUARD_RAN="@SHIP_GUARD@"'))
        render(ship_repo, template=inline).refused_with("own line")

    def test_inlining_the_guard_twice_is_refused(self, ship_repo: ShipRepo) -> None:
        stage(ship_repo)
        doubled = ship_repo.path.parent / "doubled-template.sh"
        doubled.write_text(TEMPLATE + "@SHIP_GUARD@\n")
        render(ship_repo, template=doubled).refused_with("once")

    def test_a_scalar_placeholder_may_repeat(self, ship_repo: ShipRepo) -> None:
        """Only multi-line bodies are exactly-once; a value like an S3 prefix legitimately recurs."""
        stage(ship_repo)
        repeated = ship_repo.path.parent / "repeated-template.sh"
        repeated.write_text(TEMPLATE + "SECOND_DEST=@IDLE_WATCHDOG_S3_DEST@\n")
        finished = render(ship_repo, template=repeated)
        assert finished.returncode == 0, finished.output
        rendered = (ship_repo.path.parent / "user-data.sh").read_text()
        assert rendered.count(f"={RUN_PREFIX}\n") >= 2


class TestBoxSideGuard:
    """Fails closed on a swapped object, a missing object and a bad extraction."""

    def test_it_passes_and_publishes_when_the_staged_object_is_intact(
        self, ship_repo: ShipRepo
    ) -> None:
        environment = staged_guard_environment(ship_repo)
        finished = ship_repo.run(GUARD, "all", extra_env=environment)
        assert finished.returncode == 0, finished.output
        assert "ship guard PASSED" in finished.stdout
        assert (Path(environment["SHIP_EXTRACT_DIR"]) / "runner.py").is_file()
        assert ship_repo.shutdown_calls() == []
        published = [call for call in ship_repo.aws_calls() if "/boxes/" in call]
        assert published, "the guard verified but published no provenance record"

    def test_a_corrupted_object_under_the_right_key_halts_the_box(
        self, ship_repo: ShipRepo
    ) -> None:
        environment = staged_guard_environment(ship_repo)
        ship_repo.object_at(environment["SHIP_S3_KEY"]).write_bytes(b"not the tarball you staged")
        finished = ship_repo.run(GUARD, "all", extra_env=environment)
        assert finished.returncode != 0
        assert "does not match the sha256 pinned at launch" in finished.output
        assert any("+5" in call for call in ship_repo.shutdown_calls()), (
            f"the guard refused but never armed its fuse: {ship_repo.shutdown_calls()}"
        )

    def test_a_missing_object_halts_the_box(self, ship_repo: ShipRepo) -> None:
        environment = staged_guard_environment(ship_repo)
        ship_repo.object_at(environment["SHIP_S3_KEY"]).unlink()
        finished = ship_repo.run(GUARD, "all", extra_env=environment)
        assert finished.returncode != 0
        assert "could not fetch the code archive" in finished.output
        assert ship_repo.shutdown_calls()

    def test_a_file_dropped_from_the_extract_halts_the_box(self, ship_repo: ShipRepo) -> None:
        environment = staged_guard_environment(ship_repo)
        assert ship_repo.run(GUARD, "fetch", extra_env=environment).returncode == 0
        assert ship_repo.run(GUARD, "verify-archive", extra_env=environment).returncode == 0
        assert ship_repo.run(GUARD, "extract", extra_env=environment).returncode == 0
        (Path(environment["SHIP_EXTRACT_DIR"]) / "runner.py").unlink()
        finished = ship_repo.run(GUARD, "verify-digest", extra_env=environment)
        assert finished.returncode != 0
        assert "bad extraction or a file left behind" in finished.output
        assert ship_repo.shutdown_calls()

    def test_a_leftover_file_from_an_earlier_unpack_halts_the_box(
        self, ship_repo: ShipRepo
    ) -> None:
        environment = staged_guard_environment(ship_repo)
        stale = Path(environment["SHIP_EXTRACT_DIR"])
        stale.mkdir(parents=True, exist_ok=True)
        (stale / "left_behind_by_a_previous_run.py").write_text("stale\n")
        finished = ship_repo.run(GUARD, "all", extra_env=environment)
        assert finished.returncode != 0, (
            "a file left behind by an earlier unpack passed, which the archive sha256 alone can "
            "never catch"
        )
        assert "digests to" in finished.output

    def test_a_digest_of_empty_input_cannot_be_pinned(self, ship_repo: ShipRepo) -> None:
        environment = staged_guard_environment(ship_repo)
        environment["SHIP_TREE_DIGEST"] = SHA256_OF_NOTHING
        ship_repo.run(GUARD, "all", extra_env=environment).refused_with("sha256 of empty input")

    @pytest.mark.parametrize(
        "step", ["fetch", "verify-archive", "extract", "verify-digest", "publish"]
    )
    def test_negative_control_every_step_fails_with_nothing_staged(
        self, ship_repo: ShipRepo, step: str
    ) -> None:
        """Run each check in a bare context and require it to refuse there.

        The same shape as the episode jail's negative control, and for the same reason: a check that
        passes when the thing it protects against is absent has not been shown to check anything.
        """
        finished = ship_repo.run(GUARD, step, extra_env=guard_environment(ship_repo))
        assert finished.returncode != 0, (
            f"the {step} step PASSED with nothing staged, so it is a reassuring message rather "
            f"than a check:\n{finished.output}"
        )


class TestLaunch:
    """The entrypoint: one flow, so no supported path can skip the hash."""

    def test_preflight_stages_renders_and_verifies_without_renting_anything(
        self, ship_repo: ShipRepo
    ) -> None:
        finished = launch(ship_repo, "--preflight-only")
        assert finished.returncode == 0, finished.output
        assert "protections present" in finished.stderr
        assert finished.value("ship_tree") == ship_repo.ship_hash()
        assert ship_repo.run_instance_calls() == [], "preflight rented something"

    def test_it_launches_on_the_first_candidate_that_has_capacity(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.succeed_on("g7e.2xlarge", "us-east-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        assert finished.value("region") == "us-east-2"
        assert finished.value("market") == "ondemand"
        real = [call for call in ship_repo.run_instance_calls() if not call["dry_run"]]
        assert [call["instance_type"] for call in real] == [
            "g7e.2xlarge",
            "g7e.4xlarge",
            "g7e.2xlarge",
        ], "the walk did not follow the candidate file's row order"

    def test_a_launch_without_a_run_name_is_refused(self, ship_repo: ShipRepo) -> None:
        launch(ship_repo, "--preflight-only", run_name=None).refused_with("--run-name")

    @pytest.mark.parametrize(
        "bad_name",
        [
            "wave/reskin",
            "wave reskin",
            "-wave",
            ".hidden",
            "wave$reskin",
        ],
    )
    def test_a_run_name_with_slashes_or_a_bad_charset_is_refused(
        self, ship_repo: ShipRepo, bad_name: str
    ) -> None:
        launch(ship_repo, "--preflight-only", run_name=bad_name).refused_with("--run-name")

    def test_a_run_name_carrying_a_commit_sha_shaped_token_is_refused(
        self, ship_repo: ShipRepo
    ) -> None:
        """Smuggling the sha back in through the name would re-tie the run to a code state."""
        launch(ship_repo, "--preflight-only", run_name="wave-reskin-cbc3094f").refused_with(
            "commit"
        )

    def test_every_constructed_path_lives_under_the_named_run_prefix(
        self, ship_repo: ShipRepo
    ) -> None:
        """No sha component in any prefix, key or file name the launcher constructs."""
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        assert finished.value("run_prefix") == RUN_PREFIX
        staged_keys = [
            f"s3://{path.relative_to(ship_repo.control / 's3')}"
            for path in (ship_repo.control / "s3").rglob("*")
            if path.is_file()
        ]
        assert staged_keys, "nothing was staged, so nothing pins the key layout"
        for key in staged_keys:
            assert key.startswith(f"{RUN_PREFIX}/"), (
                f"{key} landed outside the run prefix {RUN_PREFIX}"
            )
            assert sha_shaped_segments(key) == [], (
                f"{key} carries a commit-sha-shaped component; run identity must be the name alone"
            )
        for call in ship_repo.run_instance_calls():
            key = next(
                line.removeprefix("SHIP_S3_KEY=")
                for line in str(call["user_data"]).splitlines()
                if line.startswith("SHIP_S3_KEY=")
            )
            assert key.startswith(f"{RUN_PREFIX}/")
            assert sha_shaped_segments(key) == []

    def test_relaunches_land_in_the_same_prefix_however_far_head_moves(
        self, ship_repo: ShipRepo
    ) -> None:
        """The incident this whole change answers, replayed.

        A run launches, a peer commit moves HEAD, and the post-reclaim relaunch under the same
        run name must land in the same prefix -- the sha-derived prefix of the old derivation is
        what sent the relaunch into a junk prefix as a fresh run instead of resuming.
        """
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        first = launch(ship_repo)
        assert first.returncode == 0, first.output

        ship_repo.write("runner.py", "VERSION = 'a peer commit moved HEAD'\n")
        ship_repo.git("commit", "-qam", "peer commit between sittings")
        second = launch(ship_repo)
        assert second.returncode == 0, second.output

        assert first.value("head_sha") != second.value("head_sha"), (
            "HEAD did not move between the sittings, so the test exercised nothing"
        )
        assert first.value("run_prefix") == second.value("run_prefix") == RUN_PREFIX
        assert first.value("s3_key") != second.value("s3_key"), (
            "two sittings shared one code key, so the relaunch overwrote what the first ran"
        )
        relaunched = extract(
            ship_repo.object_at(second.value("s3_key")), ship_repo.path.parent / "relaunch-extract"
        )
        assert (relaunched / "runner.py").read_text() == "VERSION = 'a peer commit moved HEAD'\n", (
            "the relaunch did not ship the newest code, which is the deliberate, accepted "
            "consequence of naming runs instead of pinning them"
        )
        for finished in (first, second):
            assert ship_repo.object_at(finished.value("provenance_key")).is_file(), (
                "a sitting's provenance record went missing, so the analysis side cannot "
                "reconcile which sitting ran which code"
            )

    def test_the_launched_box_carries_its_code_provenance_as_tags(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        tags = launch_tags(ship_repo)
        assert tags["RunName"] == RUN_NAME
        assert tags["ShipTree"] == finished.value("ship_tree")
        assert tags["ShipHeadSha"] == ship_repo.git("rev-parse", "HEAD").strip()
        assert tags["ShipDirty"] == "false"
        assert tags["Market"] == "ondemand"

    def test_a_dirty_tree_ships_its_dirty_bytes_and_declares_itself(
        self, ship_repo: ShipRepo
    ) -> None:
        """Dirty is the normal case here, and per the owner's ruling it always ships.

        Several sessions share the real tree, so it is dirty essentially always; the launch must
        neither refuse it nor quietly ship HEAD instead. The staged tarball has to carry the dirty
        bytes, and the tags plus the provenance record have to say so.
        """
        ship_repo.write("runner.py", "VERSION = 'work in progress'\n")
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        assert finished.value("dirty") == "true"
        tags = launch_tags(ship_repo)
        assert tags["ShipDirty"] == "true"
        shipped = extract(
            ship_repo.object_at(finished.value("s3_key")), ship_repo.path.parent / "dirty-shipped"
        )
        assert (shipped / "runner.py").read_text() == "VERSION = 'work in progress'\n", (
            "the staged tarball carries HEAD's bytes rather than the dirty working tree's"
        )
        record = ship_repo.object_at(finished.value("provenance_key")).read_text()
        assert "dirty=true" in record
        assert "git_status= M runner.py" in record

    def test_it_enforces_imdsv2_and_shutdown_behaviour_on_every_attempt(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        launch(ship_repo)
        for call in ship_repo.run_instance_calls():
            argv = call["argv"]
            assert "HttpTokens=required" in argv
            assert argv[argv.index("--instance-initiated-shutdown-behavior") + 1] == "terminate"

    def test_a_box_that_cannot_terminate_itself_is_terminated(self, ship_repo: ShipRepo) -> None:
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        ship_repo.set_shutdown_behavior("stop")
        finished = launch(ship_repo)
        assert finished.returncode != 0
        assert "cannot terminate itself" in finished.output
        assert any(call.startswith("ec2 terminate-instances") for call in ship_repo.aws_calls()), (
            "a box whose killswitch cannot fire was left running"
        )

    def test_on_demand_exhausts_before_any_spot_row_is_tried(self, ship_repo: ShipRepo) -> None:
        ship_repo.succeed_on("nothing", "nowhere")
        finished = launch(
            ship_repo, "--allow-spot", "--spot-reason", "on demand exhausted in every region"
        )
        assert finished.returncode != 0
        markets = [call["market"] for call in ship_repo.run_instance_calls() if not call["dry_run"]]
        assert markets == ["ondemand"] * 3 + ["MarketType=spot"] * 3, (
            f"spot rows were interleaved with on-demand rows: {markets}"
        )

    def test_a_box_that_lands_on_demand_is_not_tagged_as_spot(self, ship_repo: ShipRepo) -> None:
        """--allow-spot permits spot rows; it does not make an on-demand box a spot box.

        A single tag specification rendered up front stamped Market=spot onto every attempt in the
        walk, including the on-demand ones that come first, so the tag recorded which markets the
        walk was willing to try rather than the market the box actually launched in.
        """
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(
            ship_repo, "--allow-spot", "--spot-reason", "on demand exhausted in every region"
        )
        assert finished.returncode == 0, finished.output
        assert finished.value("market") == "ondemand"
        tags = launch_tags(ship_repo)
        assert tags["Market"] == "ondemand"
        assert "SpotReason" not in tags

    def test_a_box_that_lands_on_spot_is_tagged_with_the_market_and_the_reason(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.succeed_on_spot_only("g7e.2xlarge", "us-west-2")
        finished = launch(
            ship_repo,
            "--allow-spot",
            "--spot-reason",
            "no on-demand capacity in any enabled region",
        )
        assert finished.returncode == 0, finished.output
        assert finished.value("market") == "spot"
        tags = launch_tags(ship_repo)
        assert tags["Market"] == "spot"
        assert tags["SpotReason"] == "no on-demand capacity in any enabled region"

    def test_spot_without_a_reason_is_refused(self, ship_repo: ShipRepo) -> None:
        launch(ship_repo, "--allow-spot").refused_with("--allow-spot requires --spot-reason")

    def test_spot_is_never_tried_unless_it_was_asked_for(self, ship_repo: ShipRepo) -> None:
        ship_repo.succeed_on("nothing", "nowhere")
        launch(ship_repo)
        assert all(call["market"] == "ondemand" for call in ship_repo.run_instance_calls()), (
            "a spot row entered the walk without --allow-spot"
        )

    def test_the_ami_is_resolved_once_per_region_not_once_per_row(
        self, ship_repo: ShipRepo
    ) -> None:
        """The candidate file has three rows across two regions, so two lookups suffice.

        Once per region because first_row_per_region starts exactly one probe subshell per region,
        which writes its answers to files under the run directory for adopt_region to read into the
        memo; memo_ami then answers in a global rather than on stdout, because a cache appended
        inside a command substitution's subshell vanishes with it, and the walk's whole point is
        being forty-five rows long.
        """
        ship_repo.succeed_on("g7e.2xlarge", "us-east-2")
        launch(ship_repo)
        lookups = [call for call in ship_repo.aws_calls() if call.startswith("ec2 describe-images")]
        assert len(lookups) == 2, f"expected one lookup per region, got {len(lookups)}"

    def test_one_dry_run_per_region_precedes_every_real_attempt(self, ship_repo: ShipRepo) -> None:
        """Every region is probed at once before the walk starts, so no dry run follows a real call.

        The set of regions, not their order: the probes run in parallel and finish in whichever
        order the API answers, and the walk adopts them in row order regardless.
        """
        ship_repo.succeed_on("g7e.2xlarge", "us-east-2")
        launch(ship_repo)
        calls = ship_repo.run_instance_calls()
        dry_regions = [call["region"] for call in calls if call["dry_run"]]
        assert sorted(dry_regions) == ["us-east-2", "us-west-2"]
        last_dry = max(index for index, call in enumerate(calls) if call["dry_run"])
        first_real = min(index for index, call in enumerate(calls) if not call["dry_run"])
        assert last_dry < first_real, "a dry run was made after the walk had started renting"

    def test_a_dry_run_rejected_for_a_non_capacity_reason_stops_the_walk(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.dry_run_fails_in(
            "us-west-2",
            "An error occurred (InvalidParameterValue) when calling the RunInstances operation: "
            "Value () for parameter groupId is invalid.",
        )
        finished = launch(ship_repo)
        finished.refused_with("not capacity")
        assert not [call for call in ship_repo.run_instance_calls() if not call["dry_run"]], (
            "the walk kept renting after a client-side rejection that would fail on every candidate"
        )

    def test_a_non_capacity_failure_mid_walk_stops_the_walk(self, ship_repo: ShipRepo) -> None:
        ship_repo.fail_hard_on("g7e.2xlarge", "us-west-2")
        launch(ship_repo).refused_with("not capacity")

    def test_a_region_with_no_matching_ami_is_skipped_rather_than_fatal(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.no_ami_in("us-west-2")
        ship_repo.succeed_on("g7e.2xlarge", "us-east-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        assert "no AMI matched" in finished.stderr
        assert all(call["region"] == "us-east-2" for call in ship_repo.run_instance_calls())

    def test_a_comma_in_an_operator_string_is_refused(self, ship_repo: ShipRepo) -> None:
        launch(ship_repo, "--purpose", "one thing, then another").refused_with("comma")

    def test_a_template_without_the_watchdog_destination_is_refused(
        self, ship_repo: ShipRepo
    ) -> None:
        """Recorded under a name of the kit's own, so the value reaches the box and the watchdog never
        sees it. A template that dropped the placeholder outright is refused a step earlier, by the
        render's rule against a --set that reaches nothing, which would leave this branch unreachable.
        """
        maimed = kit_dir(ship_repo) / "ud-template.sh"
        maimed.write_text(
            TEMPLATE.replace(
                "IDLE_WATCHDOG_S3_DEST=@IDLE_WATCHDOG_S3_DEST@",
                "WATCHDOG_DEST_NOTE=@IDLE_WATCHDOG_S3_DEST@",
            )
        )
        launch(ship_repo, "--preflight-only").refused_with("IDLE_WATCHDOG_S3_DEST")

    @pytest.mark.parametrize(
        ("removed", "instead", "expected"),
        [
            (
                "bash $SHIP_EXTRACT_DIR/scripts/ec2_boot_patch.sh",
                "",
                "ec2_boot_patch.sh INVOCATION",
            ),
            (
                'shutdown -h +@DEADMAN_MINUTES@ "ship-test dead-man"',
                'echo "dead-man @DEADMAN_MINUTES@ minutes, never armed"',
                "dead-man switch",
            ),
            (WATCHDOG_LINE, "", "idle watchdog"),
        ],
    )
    def test_a_template_missing_a_mandatory_protection_is_refused(
        self, ship_repo: ShipRepo, removed: str, instead: str, expected: str
    ) -> None:
        """Every mandatory protection dropped from a template, one at a time.

        The dead-man row leaves its value on a line of its own rather than deleting it outright: the
        render refuses a --set whose placeholder the template dropped, so deleting the whole shutdown
        line is refused a step earlier and this preflight branch would never be reached again.
        """
        maimed = kit_dir(ship_repo) / "ud-template.sh"
        maimed.write_text(TEMPLATE.replace(removed, instead))
        launch(ship_repo, "--preflight-only").refused_with(expected)

    def test_a_commented_out_watchdog_line_does_not_count_as_armed(
        self, ship_repo: ShipRepo
    ) -> None:
        """The arm line commented out is the template this check exists for, and a bare grep passed it."""
        maimed = kit_dir(ship_repo) / "ud-template.sh"
        maimed.write_text(TEMPLATE.replace(WATCHDOG_LINE, "# " + WATCHDOG_LINE))
        launch(ship_repo, "--preflight-only").refused_with("idle watchdog")


class TestAccountGuard:
    """The launcher rents in the account it was told to expect, and a missing subnet is not capacity.

    A launch under the wrong profile used to walk every row: under that account each candidate subnet
    "does not exist", the walk's decline pattern matched that substring, and the launcher closed with
    its own line about the nature of GPU supply. What saved the one observed case was the staging
    upload landing in a bucket the wrong account could not write, which is one --s3-prefix argument
    wide. Two independent things catch it now. The launcher asks sts which account the credentials
    resolve to before the first paid call and refuses a mismatch against --expect-account (or
    SHIP_EXPECTED_ACCOUNT) when the operator named one; and, whether they did or not, the decline
    pattern is anchored on the CLI's error code, so a NotFound stops the walk as misconfiguration
    instead of being stepped past.
    """

    MISSING_SUBNET = (
        "An error occurred (InvalidSubnetID.NotFound) when calling the RunInstances operation: "
        "The subnet ID 'subnet-0aaaaaaaaaaaaaaa1' does not exist"
    )

    def test_the_account_is_checked_before_the_first_paid_call(self, ship_repo: ShipRepo) -> None:
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        calls = ship_repo.aws_calls()
        identity = [
            index for index, call in enumerate(calls) if call.startswith("sts get-caller-identity")
        ]
        uploads = [index for index, call in enumerate(calls) if call.startswith("s3 cp ")]
        assert identity, "the launcher never asked which account it was about to rent in"
        assert uploads, "the launch never staged anything"
        assert identity[0] < uploads[0], "the account was checked only after the first upload"

    def test_the_account_is_checked_once_per_launch_not_once_per_pass(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.grant_on_real_invocation(4)
        finished = launch(ship_repo, "--walk-until", "10", "--walk-pause-minutes", "0")
        assert finished.returncode == 0, finished.output
        assert finished.value("sitting").endswith("-pass2")
        assert len(ship_repo.sts_calls()) == 1

    def test_credentials_for_another_account_are_refused_before_anything_is_staged(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.credentials_resolve_to(ANOTHER_ACCOUNT)
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo)
        finished.refused_with(f"resolve to AWS account {ANOTHER_ACCOUNT}")
        assert f"this launch expected {RENTAL_ACCOUNT}" in finished.output
        assert "does not exist" in finished.output, (
            "the refusal does not say why the account matters"
        )
        assert "Nothing was staged or launched." in finished.output
        assert ship_repo.s3_uploads() == [], "the wrong account's launch still staged code"
        assert ship_repo.run_instance_calls() == [], (
            "the wrong account's launch still tried to rent"
        )
        assert "every candidate declined" not in finished.output

    def test_the_expected_account_can_be_named_on_the_command_line(
        self, ship_repo: ShipRepo
    ) -> None:
        """The flag is the operator's own statement, and it stands without the variable set."""
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(
            ship_repo, "--expect-account", RENTAL_ACCOUNT, extra_env={"SHIP_EXPECTED_ACCOUNT": ""}
        )
        assert finished.returncode == 0, finished.output
        assert finished.value("region") == "us-west-2"
        assert f"account {RENTAL_ACCOUNT}, the account --expect-account named" in finished.stderr

    def test_a_flag_naming_another_account_refuses_too(self, ship_repo: ShipRepo) -> None:
        finished = launch(
            ship_repo, "--expect-account", ANOTHER_ACCOUNT, extra_env={"SHIP_EXPECTED_ACCOUNT": ""}
        )
        finished.refused_with(f"this launch expected {ANOTHER_ACCOUNT}")
        assert ship_repo.s3_uploads() == []

    def test_no_expected_account_launches_but_says_the_identity_went_unchecked(
        self, ship_repo: ShipRepo
    ) -> None:
        """Unset means the comparison is skipped, never that the launch stops or says nothing.

        The resolved account is still printed, because it is the fact an operator who typed the
        wrong profile needs, and the walk's NotFound anchor is what catches that mistake instead
        (the two tests at the end of this class).
        """
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo, extra_env={"SHIP_EXPECTED_ACCOUNT": ""})
        assert finished.returncode == 0, finished.output
        assert f"credentials resolve to account {RENTAL_ACCOUNT}" in finished.stderr
        assert "no --expect-account" in finished.stderr
        assert "nothing checked that this is the account you meant" in finished.stderr

    def test_a_malformed_expected_account_is_refused_rather_than_ignored(
        self, ship_repo: ShipRepo
    ) -> None:
        """A typo must not read as "no expected account": that silences the check while it looks armed."""
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo, "--expect-account", "1111-2222")
        finished.refused_with("twelve-digit AWS account id")
        assert "1111-2222" in finished.output, "the refusal does not name the value it rejected"
        assert ship_repo.s3_uploads() == []
        assert ship_repo.run_instance_calls() == []

    def test_credentials_that_cannot_be_read_are_refused_rather_than_trusted(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.deny_sts()
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo)
        finished.refused_with("cannot tell which account")
        assert "ExpiredToken" in finished.output, "the refusal should carry the CLI's own answer"
        assert "Nothing was staged or launched." in finished.output
        assert ship_repo.s3_uploads() == []
        assert ship_repo.run_instance_calls() == []

    def test_a_missing_subnet_is_misconfiguration_not_a_capacity_decline(
        self, ship_repo: ShipRepo
    ) -> None:
        """The wrong account's own symptom, one row at a time: NotFound must stop the walk."""
        ship_repo.dry_run_fails_in("us-west-2", self.MISSING_SUBNET)
        ship_repo.succeed_on("g7e.2xlarge", "us-east-2")
        finished = launch(ship_repo)
        finished.refused_with("not capacity")
        assert "InvalidSubnetID.NotFound" in finished.output
        assert ship_repo.real_run_instance_calls() == [], (
            "the walk kept renting past a subnet that does not exist"
        )
        assert "every candidate declined" not in finished.output
        assert "declined on capacity" not in finished.output

    def test_the_wrong_account_is_still_caught_with_no_expected_account_named(
        self, ship_repo: ShipRepo
    ) -> None:
        """The whole bug, end to end, under the configuration that skips the comparison.

        Making the account check operator-supplied means an operator who names nothing gets no
        comparison, so the mechanical half has to carry the case on its own: the wrong account's
        every-subnet-NotFound must stop the walk as misconfiguration rather than being walked past
        row by row and closed with the launcher's line about the nature of GPU supply.
        """
        ship_repo.credentials_resolve_to(ANOTHER_ACCOUNT)
        ship_repo.dry_run_fails_in("us-west-2", self.MISSING_SUBNET)
        ship_repo.dry_run_fails_in("us-east-2", self.MISSING_SUBNET)
        finished = launch(ship_repo, extra_env={"SHIP_EXPECTED_ACCOUNT": ""})
        finished.refused_with("not capacity")
        assert "InvalidSubnetID.NotFound" in finished.output
        assert "every candidate declined" not in finished.output, (
            "the wrong account read as a capacity shortage, which is the bug this pair exists for"
        )
        assert ship_repo.real_run_instance_calls() == []

    def test_a_missing_subnet_mid_walk_stops_the_walk(self, ship_repo: ShipRepo) -> None:
        """The same error on a real attempt, in the row's own region, once the dry run passed."""
        ship_repo.fail_hard_on("g7e.4xlarge", "us-west-2", message=self.MISSING_SUBNET)
        ship_repo.succeed_on("g7e.2xlarge", "us-east-2")
        finished = launch(ship_repo)
        finished.refused_with("not capacity")
        assert [call["region"] for call in ship_repo.real_run_instance_calls()] == [
            "us-west-2",
            "us-west-2",
        ]
        assert "every candidate declined" not in finished.output


INVALID_PARAMETER = (
    "An error occurred (InvalidParameterValue) when calling the RunInstances operation: "
    "Value () for parameter groupId is invalid."
)


class TestTheWalkAsksOnce:
    """Each declined row is one RunInstances call; only a transient answer is retried, on the same row.

    The CLI's own retry layer re-sent every InsufficientInstanceCapacity decline twice with backoff,
    because EC2 serves it as an HTTP 500 and botocore treats a 500 as transient: ~7.5 s per declined
    row, two thirds of an eight-minute sixty-row walk. The double below records one attempt per call
    the real CLI would make, so a launcher that stopped switching those retries off fails here. With
    those retries off, everything botocore would have retried -- throttles, server-side failures,
    connection failures -- is the launcher's own to retry, and under the one client token per row that
    botocore's retries used to share, so a retry after an answer that never arrived cannot launch a
    second box.
    """

    FIRST_ROW = ("g7e.2xlarge", "us-west-2")
    SECOND_ROW = ("g7e.4xlarge", "us-west-2")

    def test_each_declined_row_costs_exactly_one_run_instances_call(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.succeed_on("g7e.2xlarge", "us-east-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        real = ship_repo.real_run_instance_calls()
        assert [call["instance_type"] for call in real] == [
            "g7e.2xlarge",
            "g7e.4xlarge",
            "g7e.2xlarge",
        ], (
            "a declined row cost more than one RunInstances call, so the CLI's retry layer is back on"
        )
        assert {call["aws_max_attempts"] for call in ship_repo.run_instance_calls()} == {"1"}, (
            "run-instances was called without AWS_MAX_ATTEMPTS=1"
        )
        assert "reached max retries" not in finished.stderr

    def test_a_throttled_row_is_retried_on_the_same_row_with_the_default_backoffs_and_lands(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.throttle(*self.FIRST_ROW, times=2)
        ship_repo.succeed_on(*self.FIRST_ROW)
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        assert finished.value("region") == "us-west-2"
        real = ship_repo.real_run_instance_calls()
        assert [(call["instance_type"], call["region"]) for call in real] == [self.FIRST_ROW] * 3, (
            "the throttle was not retried on the same row"
        )
        assert "throttled by the EC2 API (attempt 1 of 3)" in finished.stderr
        assert "throttled by the EC2 API (attempt 2 of 3)" in finished.stderr
        assert ship_repo.sleeps() == ["2", "5"], (
            "the default backoffs were not slept between attempts"
        )
        assert len({call["client_token"] for call in real}) == 1, (
            "the retries did not carry the first attempt's client token"
        )

    def test_a_throttle_that_outlasts_the_retry_budget_surfaces_instead_of_being_swallowed(
        self, ship_repo: ShipRepo
    ) -> None:
        """With the CLI's retries off, a throttle is no longer hidden; it must be named, not walked past."""
        ship_repo.throttle(*self.FIRST_ROW, times=99)
        ship_repo.succeed_on(*self.SECOND_ROW)
        finished = launch(ship_repo)
        finished.refused_with("throttled by the EC2 API on 3 consecutive attempts")
        assert "RequestLimitExceeded" in finished.output
        assert "for a reason that is not capacity" not in finished.output, (
            "the throttle was reported as a malformed request, sending the operator after a bad argument"
        )
        real = ship_repo.real_run_instance_calls()
        assert [(call["instance_type"], call["region"]) for call in real] == [self.FIRST_ROW] * 3, (
            "the walk moved past the throttled row instead of stopping on it"
        )

    @pytest.mark.parametrize("kind", ["internal-error", "connection-refused", "read-timeout"])
    def test_a_server_side_or_connection_failure_is_retried_under_the_same_token_and_lands(
        self, ship_repo: ShipRepo, kind: str
    ) -> None:
        """What botocore retried, the launcher must retry itself now that botocore no longer does.

        The token matters most for the read timeout: EC2 may have completed the request whose answer
        never arrived, and only the same client token makes the retry return that instance rather
        than launch a second one.
        """
        ship_repo.fail_transiently(*self.FIRST_ROW, times=1, kind=kind)
        ship_repo.succeed_on(*self.FIRST_ROW)
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        real = ship_repo.real_run_instance_calls()
        assert [(call["instance_type"], call["region"]) for call in real] == [self.FIRST_ROW] * 2, (
            f"a {kind} was not retried on the same row"
        )
        assert (
            "transient failure from the EC2 API or the connection to it (attempt 1 of 3)"
            in finished.stderr
        )
        tokens = {call["client_token"] for call in real}
        assert None not in tokens, "a run-instances call went out without --client-token"
        assert len(tokens) == 1, (
            f"the retry after a {kind} carried a different client token ({tokens}), which is how a "
            "request the first attempt completed launches a second box"
        )

    def test_a_server_side_failure_that_outlasts_the_budget_is_named_as_such(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.fail_transiently(*self.FIRST_ROW, times=99, kind="internal-error")
        ship_repo.succeed_on(*self.SECOND_ROW)
        finished = launch(ship_repo)
        finished.refused_with("failed transiently on 3 consecutive attempts")
        assert "InternalError" in finished.output
        assert "for a reason that is not capacity" not in finished.output
        assert "throttled by the EC2 API on" not in finished.output
        real = ship_repo.real_run_instance_calls()
        assert [(call["instance_type"], call["region"]) for call in real] == [self.FIRST_ROW] * 3

    def test_a_persistent_service_unavailable_is_walked_past_as_the_clis_exhausted_retries_were(
        self, ship_repo: ShipRepo
    ) -> None:
        """ServiceUnavailable is in both lists: retried on the row, then stepped past as a decline."""
        ship_repo.fail_transiently(*self.FIRST_ROW, times=99, kind="service-unavailable")
        ship_repo.succeed_on(*self.SECOND_ROW)
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        real = ship_repo.real_run_instance_calls()
        assert [(call["instance_type"], call["region"]) for call in real] == [
            *([self.FIRST_ROW] * 3),
            self.SECOND_ROW,
        ]
        assert "declined: An error occurred (ServiceUnavailable)" in finished.stderr

    def test_every_row_and_every_dry_run_carries_its_own_client_token(
        self, ship_repo: ShipRepo
    ) -> None:
        """One token per row, never shared across rows: a reused token with different parameters is an
        IdempotentParameterMismatch, or worse the first row's answer replayed."""
        ship_repo.succeed_on("g7e.2xlarge", "us-east-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        tokens = [call["client_token"] for call in ship_repo.run_instance_calls()]
        assert None not in tokens, "a run-instances call went out without --client-token"
        assert len(set(tokens)) == len(tokens) == 5, f"tokens were shared across calls: {tokens}"
        for token in tokens:
            assert len(token) <= 64, f"{token!r} is over EC2's 64-character limit"
            assert token.isascii(), f"{token!r} is not the ASCII EC2 requires"
            assert token.startswith(finished.value("sitting")), (
                f"{token!r} is not scoped to the sitting"
            )

    def test_a_dry_run_throttled_twice_is_retried_and_adopted_as_ok(
        self, ship_repo: ShipRepo
    ) -> None:
        """The dry runs go through the same retry, inside their probe subshell, under one token."""
        ship_repo.throttle_dry_run(*self.FIRST_ROW, times=2)
        ship_repo.succeed_on(*self.FIRST_ROW)
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        dry = [call for call in ship_repo.dry_run_instance_calls() if call["region"] == "us-west-2"]
        assert len(dry) == 3, "the throttled dry run was not retried"
        assert len({call["client_token"] for call in dry}) == 1
        assert "throttled by the EC2 API (attempt 2 of 3)" in finished.stderr
        assert "dry-run OK in us-west-2" in finished.stderr

    def test_a_dry_run_throttled_past_the_budget_dies_where_the_row_order_reaches_it(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.throttle_dry_run("g7e.2xlarge", "us-east-2", times=99)
        ship_repo.succeed_on("nothing", "nowhere")
        finished = launch(ship_repo)
        finished.refused_with("throttled by the EC2 API on 3 consecutive attempts")
        assert "us-east-2 (dry run)" in finished.output
        assert [call["region"] for call in ship_repo.real_run_instance_calls()] == [
            "us-west-2",
            "us-west-2",
        ], "the walk did not try every earlier row before the throttled region stopped it"

    def test_a_garbage_backoff_list_is_refused_before_anything_is_staged(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.succeed_on(*self.FIRST_ROW)
        finished = launch(ship_repo, extra_env={"TRANSIENT_BACKOFF_SECONDS": "2 soon"})
        finished.refused_with("whole seconds")
        assert ship_repo.aws_calls() == [], "the tree was staged to S3 before the refusal"


class TestTheRunInstancesDoubleHasTeeth:
    """The negative control for the retry emulation the class above rests on.

    If the double recorded one attempt per call regardless of AWS_MAX_ATTEMPTS, every assertion in
    TestTheWalkAsksOnce would pass with the launcher's switch removed. So: the double, asked directly
    for a declined row, must record three attempts with the variable unset and one with it set to 1.
    """

    def decline(self, ship_repo: ShipRepo, environment: dict[str, str]) -> Finished:
        user_data = ship_repo.path.parent / "double-user-data.sh"
        user_data.write_text("#!/bin/bash\n")
        tag_spec = ship_repo.path.parent / "double-tagspec.json"
        tag_spec.write_text(json.dumps([{"ResourceType": "instance", "Tags": []}]))
        completed = subprocess.run(  # noqa: S603 - the test's own stub, literal arguments
            [
                str(stub_aws(ship_repo)),
                "ec2",
                "run-instances",
                "--region",
                "us-west-2",
                "--instance-type",
                "g7e.2xlarge",
                "--user-data",
                f"file://{user_data}",
                "--tag-specifications",
                f"file://{tag_spec}",
            ],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        return Finished(completed.returncode, completed.stdout, completed.stderr)

    def test_without_the_switch_a_decline_is_three_attempts_like_botocore(
        self, ship_repo: ShipRepo
    ) -> None:
        environment = {
            key: value for key, value in ship_repo.env.items() if key != "AWS_MAX_ATTEMPTS"
        }
        finished = self.decline(ship_repo, environment)
        assert finished.returncode != 0
        assert "(reached max retries: 2)" in finished.stderr
        assert [call["attempt"] for call in ship_repo.run_instance_calls()] == [1, 2, 3]

    def test_with_the_switch_a_decline_is_one_attempt(self, ship_repo: ShipRepo) -> None:
        finished = self.decline(ship_repo, {**ship_repo.env, "AWS_MAX_ATTEMPTS": "1"})
        assert finished.returncode != 0
        assert "reached max retries" not in finished.stderr
        assert [call["attempt"] for call in ship_repo.run_instance_calls()] == [1]


class TestDoneMarkerPreflightWarning:
    """A template that arms the watchdog without --done-marker is warned about, never refused.

    Without the marker the idle threshold is the watchdog's only trigger and the box bills a 40-45
    minute idle tail after its agenda finishes (2,400 s measured on one box). A refusal would strand
    the kits that are mid-wave without it, so the preflight says so loudly and lets the launch go on.
    """

    def test_a_template_with_the_marker_passes_preflight_without_the_warning(
        self, ship_repo: ShipRepo
    ) -> None:
        finished = launch(ship_repo, "--preflight-only")
        assert finished.returncode == 0, finished.output
        assert "--done-marker" not in finished.stderr, (
            "the warning fired on a template that carries the marker, so it says nothing when it matters"
        )

    def test_a_template_without_the_marker_warns_but_is_not_refused(
        self, ship_repo: ShipRepo
    ) -> None:
        maimed = kit_dir(ship_repo) / "ud-template.sh"
        maimed.write_text(TEMPLATE.replace(WATCHDOG_LINE, WATCHDOG_LINE_WITHOUT_MARKER))
        finished = launch(ship_repo, "--preflight-only")
        assert finished.returncode == 0, (
            f"a missing --done-marker refused the launch; it must warn, not fail:\n{finished.output}"
        )
        assert "WARN: the idle watchdog is armed with no --done-marker" in finished.stderr
        assert "40-45 minute idle tail" in finished.stderr
        assert "protections present" in finished.stderr

    def test_a_marker_named_only_in_a_comment_still_warns(self, ship_repo: ShipRepo) -> None:
        """A comment that mentions the flag arms nothing, and must not satisfy the check."""
        maimed = kit_dir(ship_repo) / "ud-template.sh"
        maimed.write_text(
            TEMPLATE.replace(
                WATCHDOG_LINE,
                "# TODO pass --done-marker /home/ubuntu/AGENDA_DONE\n"
                + WATCHDOG_LINE_WITHOUT_MARKER,
            )
        )
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        assert "WARN: the idle watchdog is armed with no --done-marker" in finished.stderr

    def test_a_mention_off_the_arm_line_does_not_silence_the_warning(
        self, ship_repo: ShipRepo
    ) -> None:
        """A runner line that merely echoes the flag arms nothing; only the watchdog's own line counts."""
        maimed = kit_dir(ship_repo) / "ud-template.sh"
        maimed.write_text(
            TEMPLATE.replace(
                WATCHDOG_LINE,
                'echo "touch --done-marker /home/ubuntu/AGENDA_DONE after the last upload returns"\n'
                + WATCHDOG_LINE_WITHOUT_MARKER,
            )
        )
        finished = launch(ship_repo, "--preflight-only")
        assert finished.returncode == 0, finished.output
        assert "WARN: the idle watchdog is armed with no --done-marker" in finished.stderr, (
            "a --done-marker mentioned off the idle_watchdog.sh line silenced the warning"
        )


class TestWalkUntil:
    """--walk-until repeats the whole pass until a box lands or the deadline passes.

    It stands in for the hand-rolled loops that re-issued a walk every eight to fifteen minutes (222
    cycles over 55 hours in one kit). Each pass is a fresh sitting, so the tree as it stands ships and
    no two passes share a key; and the loop never widens the walk on its own -- spot is refused in
    combination, and a p5 pass is a different candidate file the operator chooses.
    """

    def test_it_repeats_the_pass_until_capacity_appears_and_lands_once(
        self, ship_repo: ShipRepo
    ) -> None:
        """Three rows per pass, so the seventh real call is the first row of the third pass."""
        ship_repo.grant_on_real_invocation(7)
        finished = launch(ship_repo, "--walk-until", "10", "--walk-pause-minutes", "0")
        assert finished.returncode == 0, finished.output
        assert finished.value("region") == "us-west-2"
        assert finished.value("market") == "ondemand"
        assert finished.value("run_prefix") == RUN_PREFIX
        assert finished.value("sitting").endswith("-pass3")
        assert finished.stdout.count("instance=") == 1, "more than one box was launched"
        real = ship_repo.real_run_instance_calls()
        assert len(real) == 7, f"expected two full passes plus one row, got {len(real)} real calls"
        assert all(call["market"] == "ondemand" for call in real), "a spot row entered the wait"
        staged = ship_repo.staged_code_keys()
        assert len(staged) == 3, f"each pass must stage its own sitting; found {staged}"
        for key in staged:
            assert key.startswith(f"{RUN_PREFIX}/sittings/"), f"{key} landed outside the run prefix"
            assert sha_shaped_segments(key) == [], f"{key} carries a commit-sha-shaped component"
        assert "== pass 3 of run" in finished.stderr

    def test_the_deadline_hands_back_the_exhaustion_evidence(self, ship_repo: ShipRepo) -> None:
        """A wait of zero minutes is one pass; the refusal then reads like a single walk's, with counts."""
        ship_repo.succeed_on("nothing", "nowhere")
        finished = launch(ship_repo, "--walk-until", "0", "--walk-pause-minutes", "0")
        finished.refused_with("every candidate declined on each of 1 pass(es)")
        assert "separate, explicit decisions" in finished.output
        assert len(ship_repo.real_run_instance_calls()) == 3

    def test_the_pause_is_cut_short_to_end_on_the_deadline(self, ship_repo: ShipRepo) -> None:
        """A two-minute wait with a five-minute pause: the one pause must end inside the two minutes.

        A deadline checked only after each pass let the pause run past it, and a --walk-until 60 with
        the default pause ran to ~67 minutes; no pass may begin after the deadline. The recording
        `sleep` on the test PATH is what makes the chosen duration visible without waiting it out.
        """
        ship_repo.grant_on_real_invocation(4)
        finished = launch(ship_repo, "--walk-until", "2", "--walk-pause-minutes", "5")
        assert finished.returncode == 0, finished.output
        assert finished.value("sitting").endswith("-pass2")
        pauses = [int(seconds) for seconds in ship_repo.sleeps()]
        assert len(pauses) == 1, f"expected one pause between two passes, got {pauses}"
        assert 0 < pauses[0] <= 120, f"a pause of {pauses[0]} s runs past the two-minute deadline"

    def test_the_full_pause_is_taken_while_the_deadline_is_further_away(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.grant_on_real_invocation(4)
        finished = launch(ship_repo, "--walk-until", "10", "--walk-pause-minutes", "5")
        assert finished.returncode == 0, finished.output
        assert ship_repo.sleeps() == ["300"]

    def test_spot_cannot_be_combined_with_the_wait(self, ship_repo: ShipRepo) -> None:
        """Spot after a wait is a decision made with the exhaustion evidence in hand, never a loop's."""
        ship_repo.succeed_on_spot_only("g7e.2xlarge", "us-west-2")
        finished = launch(
            ship_repo, "--walk-until", "10", "--allow-spot", "--spot-reason", "on demand exhausted"
        )
        finished.refused_with("--walk-until with --allow-spot")
        assert ship_repo.run_instance_calls() == [], "the refusal came after something was tried"

    @pytest.mark.parametrize("cannot_land", ["--dry-run-only", "--preflight-only"])
    def test_a_pass_that_can_never_land_is_refused_rather_than_looped(
        self, ship_repo: ShipRepo, cannot_land: str
    ) -> None:
        launch(ship_repo, "--walk-until", "10", cannot_land).refused_with("--walk-until")

    @pytest.mark.parametrize(
        ("flag", "value"), [("--walk-until", "soon"), ("--walk-pause-minutes", "a while")]
    )
    def test_a_garbage_duration_is_refused(
        self, ship_repo: ShipRepo, flag: str, value: str
    ) -> None:
        launch(ship_repo, flag, value).refused_with("whole number of minutes")


class TestRegionProbesRunInParallel:
    """Every region's AMI lookup and dry run fire at once; the walk adopts them in row order.

    The saving is the sum of the per-region round trips becoming the slowest one. The invariant is
    that nothing observable moves: the same calls, the same stderr lines at the same moments, and a
    dry run rejected for a non-capacity reason still stopping the walk exactly where the sequential
    walk stopped -- when a row first reaches that region -- so a bad row in a later region cannot
    block a launch an earlier region would have granted.
    """

    def test_the_probes_of_every_region_overlap_in_time(self, ship_repo: ShipRepo) -> None:
        """Two regions, two delayed probe calls each: the two dry runs must be in flight together.

        Asserted on the interval the double stamps on each dry run rather than on the launch's wall
        clock, which on a box running several agent sessions proves little either way. Serial probes
        would put the second region's dry run entirely after the first region's, and parallel AMI
        lookups with serialized dry runs would too. The box lands in the LAST region so every probe
        has been adopted, and so recorded, before the launch returns.
        """
        ship_repo.delay_probes(2.0)
        ship_repo.succeed_on("g7e.2xlarge", "us-east-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        dry = {call["region"]: call for call in ship_repo.dry_run_instance_calls()}
        assert set(dry) == {"us-west-2", "us-east-2"}
        west, east = dry["us-west-2"], dry["us-east-2"]
        latest_start = max(float(west["started_at"]), float(east["started_at"]))
        earliest_finish = min(float(west["finished_at"]), float(east["finished_at"]))
        assert latest_start < earliest_finish, (
            "the two regions' dry runs did not overlap in time, which is the serial pacing"
        )
        lookups = [call for call in ship_repo.aws_calls() if call.startswith("ec2 describe-images")]
        assert len(lookups) == 2, "parallel probing changed how many lookups were made"

    def test_a_landing_does_not_wait_for_a_slower_regions_probe(self, ship_repo: ShipRepo) -> None:
        """A slow describe-images in a later region must not delay a launch the first region grants.

        The first region's probe is instant and its first row lands; the second region's probe is made
        slow. Had the walk waited for every probe before its first row, the slow region's dry run would
        have run before the launch returned. And the launcher reaps the probes the walk never reached,
        so that dry run must not turn up afterwards either: a probe outliving the launcher would keep
        writing into the run directory after it exited.
        """
        ship_repo.delay_probes_in("us-east-2", 1.5)
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        assert finished.value("region") == "us-west-2"
        # Long enough for an un-reaped probe to reach its dry run: a 1.5 s lookup then its dry run.
        time.sleep(7)
        east = [call for call in ship_repo.run_instance_calls() if call["region"] == "us-east-2"]
        assert east == [], (
            "the slow region's dry run ran after all: the walk waited for it, or its probe outlived the launch"
        )

    def test_a_probe_that_dies_before_finishing_is_reported_where_its_region_is_reached(
        self, ship_repo: ShipRepo
    ) -> None:
        """A probe killed mid-way leaves no verdict, and the walk must say so rather than read its empty
        files as "no AMI here" -- and only once a row reaches that region, after the earlier region's
        rows have had their real attempts."""
        ship_repo.kill_probe_in("us-east-2")
        ship_repo.succeed_on("nothing", "nowhere")
        finished = launch(ship_repo)
        finished.refused_with("the probe of us-east-2 never finished")
        assert [call["region"] for call in ship_repo.real_run_instance_calls()] == [
            "us-west-2",
            "us-west-2",
        ], "the walk did not try every earlier row before the dead probe stopped it"

    def test_probe_output_is_replayed_in_row_order(self, ship_repo: ShipRepo) -> None:
        ship_repo.succeed_on("g7e.2xlarge", "us-east-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        ami_west = finished.stderr.index(f"using base AMI {base_ami('us-west-2')}")
        dry_west = finished.stderr.index("dry-run OK in us-west-2")
        trying_west = finished.stderr.index("trying g7e.2xlarge    us-west-2")
        ami_east = finished.stderr.index(f"using base AMI {base_ami('us-east-2')}")
        dry_east = finished.stderr.index("dry-run OK in us-east-2")
        trying_east = finished.stderr.index("trying g7e.2xlarge    us-east-2")
        assert ami_west < dry_west < trying_west < ami_east < dry_east < trying_east, (
            f"a region's probe output was not replayed where the walk reached it:\n{finished.stderr}"
        )

    def test_a_bad_dry_run_in_a_later_region_does_not_block_an_earlier_regions_launch(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.dry_run_fails_in("us-east-2", INVALID_PARAMETER)
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, (
            f"a typo in a later region's row stopped a launch an earlier region granted:\n{finished.output}"
        )
        assert finished.value("region") == "us-west-2"
        assert "for a reason that is not capacity" not in finished.output

    def test_a_bad_dry_run_stops_the_walk_where_the_row_order_reaches_it(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.dry_run_fails_in("us-east-2", INVALID_PARAMETER)
        ship_repo.succeed_on("nothing", "nowhere")
        finished = launch(ship_repo)
        finished.refused_with("not capacity")
        assert [call["region"] for call in ship_repo.real_run_instance_calls()] == [
            "us-west-2",
            "us-west-2",
        ], "the walk did not try every earlier row before the bad region stopped it"

    def test_dry_run_only_probes_every_region_and_rents_nothing(self, ship_repo: ShipRepo) -> None:
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo, "--dry-run-only")
        assert finished.returncode == 0, finished.output
        assert "dry runs only" in finished.stderr
        assert ship_repo.real_run_instance_calls() == [], "--dry-run-only rented something"
        dry_regions = {call["region"] for call in ship_repo.run_instance_calls()}
        assert dry_regions == {"us-west-2", "us-east-2"}


def launched_images(ship_repo: ShipRepo, region: str = "us-west-2") -> set[str]:
    """Every --image-id the launcher handed to run-instances in the region, dry runs included.

    Scoped to one region because every region in the candidate file is probed up front, so a walk
    that lands on its first row has still dry-run the other regions with THEIR images.
    """
    return {
        call["argv"][call["argv"].index("--image-id") + 1]
        for call in ship_repo.run_instance_calls()
        if call["region"] == region
    }


def stub_aws(ship_repo: ShipRepo) -> Path:
    """The fake ``aws`` on the repo's PATH, for the tests whose subject is the double itself."""
    return ship_repo.path.parent / "bin" / "aws"


class TestAmiResolution:
    """One lookup per region resolves the newest build of the image family, and says which it took.

    Every rented box patches its own OS at boot, because the newest available build of this family
    has itself shipped a vulnerable package -- so the image the launcher picked is a fact the
    operator has to be able to read out of the log.
    """

    def test_the_resolved_image_is_named_and_is_what_run_instances_is_handed(
        self, ship_repo: ShipRepo
    ) -> None:
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        assert f"using base AMI {base_ami('us-west-2')} in us-west-2" in finished.stderr
        assert launched_images(ship_repo) == {base_ami("us-west-2")}

    def test_an_overridden_ami_pattern_is_what_gets_looked_up(self, ship_repo: ShipRepo) -> None:
        """The operator's --ami-name-pattern must reach DescribeImages, not just be accepted."""
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo, "--ami-name-pattern", "Some Other Image *")
        assert finished.returncode == 0, finished.output
        lookups = [call for call in ship_repo.aws_calls() if call.startswith("ec2 describe-images")]
        assert lookups, "no AMI lookup was made at all"
        for lookup in lookups:
            assert "Name=name,Values=Some Other Image *" in lookup, (
                f"the operator's image family never reached the lookup: {lookup}"
            )

    def test_a_failed_lookup_is_reported_rather_than_read_as_an_absent_image(
        self, ship_repo: ShipRepo
    ) -> None:
        """A denial and an absent image are the same empty stdout; only stderr separates them.

        Skipping the region is the right answer to an absence, and it is silent -- so a lost
        ec2:DescribeImages permission would skip every region and the launch would close with the
        launcher's own line about the nature of GPU supply, which is the exact false diagnosis this
        tool exists to prevent.
        """
        ship_repo.fail_ami_lookup_in(
            "us-west-2",
            "An error occurred (UnauthorizedOperation) when calling the DescribeImages operation: "
            "You are not authorized to perform this operation.",
        )
        ship_repo.succeed_on("g7e.2xlarge", "us-east-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        assert "the AMI lookup in us-west-2 FAILED rather than finding nothing" in finished.stderr
        assert "UnauthorizedOperation" in finished.stderr
        assert launched_images(ship_repo) == set(), "the failed region was rented in anyway"
        assert finished.value("region") == "us-east-2"

    def test_an_absent_image_is_skipped_without_the_failure_warning(
        self, ship_repo: ShipRepo
    ) -> None:
        """The other half of the pair: a genuine absence must not raise the failure warning.

        Without this the warning could be unconditional and every assertion above would still
        pass, leaving a message that says nothing on the one launch where it matters.
        """
        ship_repo.no_ami_in("us-west-2")
        ship_repo.succeed_on("g7e.2xlarge", "us-east-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        assert "no AMI matched" in finished.stderr
        assert "FAILED rather than finding nothing" not in finished.stderr


class TestTheLaunchShipsWhatIsOnDiskNow:
    """Every launch stages afresh; no leftover artifact or template edit can substitute older code."""

    def test_a_previously_rendered_user_data_in_the_run_directory_is_replaced(
        self, ship_repo: ShipRepo
    ) -> None:
        """The launch renders afresh, so a stale artifact left lying about cannot be picked up.

        Sitting A's user-data is left in the run directory, the working state moves on, and the
        launch has to overwrite it with one naming the fresh sitting's key. Reusing A is the shape
        of the box launched at 2026-08-27T16:27Z that fetched the previous evening's tarball.
        """
        staged_a = stage(ship_repo)
        assert render(ship_repo).returncode == 0
        stale = (ship_repo.path.parent / "user-data.sh").read_text()
        stale_key = staged_a.value("s3_key")
        assert f"SHIP_S3_KEY={stale_key}" in stale

        run_dir = ship_repo.path.parent / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "user-data.sh").write_text(stale)

        ship_repo.write("runner.py", "VERSION = 'the fix nobody shipped'\n")
        ship_repo.succeed_on("g7e.2xlarge", "us-west-2")
        finished = launch(ship_repo)
        assert finished.returncode == 0, finished.output
        fresh_key = finished.value("s3_key")
        assert fresh_key != stale_key

        handed_to_ec2 = [
            str(call["user_data"]) for call in ship_repo.run_instance_calls() if not call["dry_run"]
        ]
        assert handed_to_ec2, "nothing was launched, so nothing pins what EC2 was handed"
        for user_data in handed_to_ec2:
            assert f"SHIP_S3_KEY={fresh_key}" in user_data
            assert stale_key not in user_data, "the box was handed the stale rendering after all"
        shipped = extract(ship_repo.object_at(fresh_key), ship_repo.path.parent / "fresh-extract")
        assert (shipped / "runner.py").read_text() == "VERSION = 'the fix nobody shipped'\n"

    def test_the_launcher_refuses_a_user_data_naming_a_key_it_never_staged(
        self, ship_repo: ShipRepo
    ) -> None:
        """Only the launcher's key comparison can catch this one.

        The template hardcodes some other sitting's key for its own SHIP_S3_KEY line while keeping
        the @SHIP_S3_KEY@ placeholder elsewhere, so the render is satisfied and the user-data still
        tells the box to fetch bytes this launch never staged. Not a freshness gate: it compares
        the render's output against this same invocation's staging, and no peer edit can trip it.
        """
        other_key = f"{RUN_PREFIX}/sittings/19990101T000000Z-p0/code.tar.gz"
        smuggled = kit_dir(ship_repo) / "ud-template.sh"
        smuggled.write_text(
            TEMPLATE.replace(
                "SHIP_S3_KEY=@SHIP_S3_KEY@",
                f"SHIP_S3_KEY={other_key}\nIGNORED_KEY=@SHIP_S3_KEY@",
            )
        )
        finished = launch(ship_repo, "--preflight-only")
        finished.refused_with("never staged")
        assert other_key in finished.output


class TestTheShaCheckerHasTeeth:
    """The negative control for sha_shaped_segments, the assertion the no-pin tests rest on.

    Run against the OLD derivation's shapes, the checker must flag them; a checker that passes
    everything would make every no-sha assertion above a reassuring message rather than a check.
    """

    def test_it_flags_the_old_content_addressed_derivation(self) -> None:
        old_style = f"{S3_PREFIX}/tree/9c40dd680caff77e10b455e21fbbd35c11e75e63/code.tar.gz"
        assert sha_shaped_segments(old_style) == ["9c40dd680caff77e10b455e21fbbd35c11e75e63"]

    def test_it_flags_a_short_sha_suffix_in_a_run_name(self) -> None:
        assert sha_shaped_segments(f"{S3_PREFIX}/games-rl/wave3-reskin-cbc3094f/code.tar.gz") == [
            "cbc3094f"
        ]

    def test_it_passes_the_new_derivation_and_plain_timestamps(self) -> None:
        clean = f"{RUN_PREFIX}/sittings/20260831T142530Z-p12345/code.tar.gz"
        assert sha_shaped_segments(clean) == []
        assert sha_shaped_segments(f"{RUN_PREFIX}/provenance/20260831T142530Z-p1-launch.txt") == []


class TestGuardRemovalIsRefused:
    """The guard is the component whose absence is invisible, so removing it must refuse at render."""

    def test_deleting_the_guard_placeholder_from_a_template_is_refused(
        self, ship_repo: ShipRepo
    ) -> None:
        stage(ship_repo)
        maimed = ship_repo.path.parent / "no-guard-template.sh"
        maimed.write_text(TEMPLATE.replace("@SHIP_GUARD@", ""))
        render(ship_repo, template=maimed).refused_with("no @SHIP_GUARD@ placeholder")

    @pytest.mark.parametrize(
        ("removed", "expected"),
        [
            ("sha256sum -c -", "the archive sha256 comparison"),
            ('[ "$observed" = "$SHIP_TREE_DIGEST" ]', "the extracted-tree digest comparison"),
            ("--no-same-permissions", "mode-normalised extraction"),
        ],
    )
    def test_deleting_a_check_from_the_guard_is_caught_by_the_render(
        self, ship_repo: ShipRepo, removed: str, expected: str
    ) -> None:
        guard = ship_repo.path / "scripts" / GUARD.name
        guard.write_text(guard.read_text().replace(removed, "true"))
        assert stage(ship_repo).returncode == 0
        render(ship_repo).refused_with(expected)
