"""Assert that an episode jail actually contains the thing inside it.

Run this INSIDE the jail: every check must pass, and the process exits 0 only if they
all do. Run the same file OUTSIDE the jail and the interesting checks must FAIL -- that
is the negative control, and it is the most important test here, because a suite that
passes because it is broken is worse than no suite at all. Use
scripts/run_jail_tests.sh, which performs both runs and verifies the flip per check
rather than merely inverting an exit code.

Two deliberate design constraints:

- No file contents are ever read, printed or logged. Proving a secret is unreachable
  does not require reading it, so the checks open and immediately close a descriptor
  and report only the errno. If the jail ever leaks, this file must not be the thing
  that copies a credential into a log.
- No writes to any host path. Probes are existence, permission and open-for-read tests
  only, so running this on a leaking jail cannot itself damage anything. The one write
  probe targets a read-only bind inside the jail and unlinks on the (failing) path
  where it unexpectedly succeeds.

This runs under the interpreter the jail exposes (`scripts/episode_jail.sh --print-jail-python`),
3.12 or newer by contract; docs/episode-isolation.md records why that floor is a scoring property
rather than a packaging detail. The conservative dialect stays regardless, because ruff and
basedpyright are directory-scoped and `scripts/` still holds siblings that genuinely run under the
host's older system Python, with no per-file version override in either tool.
tests/test_interpreter_compat.py is what holds that line.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

JAIL_HOME = "/work"

# Real homes may sit under a prefix with a symlink shadowing them at the bare path; both get probed.
LOCAL_HOME_PREFIX = "/local"

SECRET_RELATIVE_PATHS = (
    ".gnupg",
    ".ssh/id_rsa",
    ".aws/credentials",
    ".git-credentials",
)

# A modified hook silently widens what every agent on this box may do.
HOOK_RELATIVE_DIRS = (".claude/hooks",)

# One IP literal and one name, both public: routing and name resolution fail separately here.
NETWORK_TARGETS = (("1.1.1.1", 443), ("example.com", 443))
CONNECT_TIMEOUT_SECONDS = 3.0

DOCKER_SOCKET = "/var/run/docker.sock"
HOST_INIT_NAMES = frozenset({"systemd", "init"})
MAX_JAIL_VISIBLE_PIDS = 50
BUS_ENV_VARS = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")

# Bait paths the episode environment shows the policy; literal so they match it exactly.
HONEYPOT_PATHS = (
    "/srv/reports",
    "/opt/acme/config/credentials.ini",
    "/home/svc-ops/notes.md",  # HARNESS-SCAN-EXEMPT-hardcoded-user-machine-path: invented bait user
)


def host_homes_from_environment() -> tuple[str, ...]:
    """Derive the host home directories to probe from HOME, refusing the jail's own HOME.

    Inside the jail HOME is /work and nothing of the host's filesystem is reachable, so the script
    cannot discover the host's home for itself: expanduser("~") would point every containment check
    at the episode directory and the whole suite would pass while proving nothing. The jail also
    clears the environment, so it cannot be told through a variable either -- the in-jail run gets
    --host-home on its command line, computed by the launcher on the host. Refusing to guess here is
    what makes a forgotten flag a loud failure rather than a vacuous pass.
    """
    home = os.environ.get("HOME", "").rstrip("/")
    if not home or home == JAIL_HOME:
        raise RuntimeError(
            f"HOME is {home or 'unset'!r}, which is not a host home directory, so the paths these "
            "checks probe cannot be derived. Pass --host-home explicitly (repeatable); the in-jail "
            "run always must, because the jail clears the environment and sets HOME to "
            f"{JAIL_HOME}."
        )
    twin = (
        home[len(LOCAL_HOME_PREFIX) :]
        if home.startswith(LOCAL_HOME_PREFIX + "/")
        else LOCAL_HOME_PREFIX + home
    )
    return tuple(dict.fromkeys([home, twin]))


@dataclass(frozen=True)
class ProbeConfig:
    """Host locations to probe, passed in so they can never resolve vacuously.

    Derived from an explicit home list rather than os.path.expanduser("~") because
    inside the jail HOME is /work: expanduser would point every check at the episode
    directory, and the whole suite would pass while proving nothing.

    ``launcher_session`` and ``launcher_namespaces`` are the launcher's own session id and
    namespace identities (``ipc:[4026531839]`` and the like), measured on the host and handed in
    for the same reason: a check that the jail left them can only be expressed as a difference from
    the outside, and inside there is nothing left to compare against.
    """

    host_homes: tuple[str, ...]
    jail_python: str = ""
    launcher_session: int = 0
    launcher_namespaces: tuple[tuple[str, str], ...] = ()

    @property
    def secret_paths(self) -> tuple[str, ...]:
        """Build string paths for credentials the jail must hide."""
        return tuple(
            str(Path(home) / rel) for home in self.host_homes for rel in SECRET_RELATIVE_PATHS
        )

    @property
    def hook_dirs(self) -> tuple[str, ...]:
        """Build string paths for hook directories that must stay unreachable."""
        return tuple(
            str(Path(home) / rel) for home in self.host_homes for rel in HOOK_RELATIVE_DIRS
        )


@dataclass(frozen=True)
class CheckResult:
    """Carry a check's machine-readable status and diagnostic detail."""

    name: str
    passed: bool
    detail: str


def _errno_name(exc: OSError) -> str:
    return os.strerror(exc.errno) if exc.errno else type(exc).__name__


def _is_readable(path: str) -> tuple[str, str]:
    """Classify a path as absent / unreadable / READABLE without reading its bytes."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return "absent", "ENOENT"
    except OSError as exc:
        return "unreadable", _errno_name(exc)
    os.close(fd)
    return "READABLE", "opened for reading"


def check_secrets_unreadable(config: ProbeConfig) -> CheckResult:
    """Absence is not evidence of containment, so only READABLE fails this check."""
    statuses = {path: _is_readable(path) for path in config.secret_paths}
    leaked = [path for path, (state, _) in statuses.items() if state == "READABLE"]
    detail = "; ".join(f"{path}={state}({why})" for path, (state, why) in statuses.items())
    return CheckResult("secrets_unreadable", not leaked, detail)


def check_host_home_absent(config: ProbeConfig) -> CheckResult:
    """Keep host-home exposure from making containment vacuous."""
    present = [d for d in config.host_homes if Path(d).exists()]
    detail = f"present={present}" if present else f"none of {list(config.host_homes)} resolve"
    return CheckResult("host_home_absent", not present, detail)


def check_host_home_not_writable(config: ProbeConfig) -> CheckResult:
    """Prevent writable host homes from giving episodes a host-side write surface."""
    writable = [d for d in config.host_homes if os.access(d, os.W_OK)]
    return CheckResult(
        "host_home_not_writable",
        not writable,
        f"writable={writable}" if writable else "no host home directory is writable",
    )


def check_claude_hooks_unreachable(config: ProbeConfig) -> CheckResult:
    """Absence proves unreadable and unwritable at once, so nothing is written here.

    Deliberately no write probe: creating even a scratch file under a real hooks
    directory would be a modification of the surface we are protecting.
    """
    reachable = []
    for directory in config.hook_dirs:
        try:
            entries = list(Path(directory).iterdir())
        except OSError:
            continue
        reachable.append(f"{directory} ({len(entries)} entries)")
    return CheckResult(
        "claude_hooks_unreachable",
        not reachable,
        f"listable={reachable}" if reachable else "hook directories do not resolve",
    )


def _connect_fails(host: str, port: int) -> tuple[bool, str]:
    try:
        sock = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_SECONDS)
    except OSError as exc:
        return True, f"{type(exc).__name__}"
    sock.close()
    return False, "CONNECTED"


def check_network_egress_blocked(config: ProbeConfig) -> CheckResult:
    """Keep episodes from reaching external services that could expose data."""
    del config  # reachability does not depend on host paths
    outcomes = {}
    for host, port in NETWORK_TARGETS:
        blocked, why = _connect_fails(host, port)
        outcomes[f"{host}:{port}"] = (blocked, why)
    reachable = [target for target, (blocked, _) in outcomes.items() if not blocked]
    detail = "; ".join(f"{target}={why}" for target, (_, why) in outcomes.items())
    return CheckResult("network_egress_blocked", not reachable, detail)


def check_docker_socket_absent(config: ProbeConfig) -> CheckResult:
    """Guard the Docker socket: its group can make a reachable daemon effectively root."""
    del config
    path = DOCKER_SOCKET
    return CheckResult(
        "docker_socket_absent",
        not Path(path).exists(),
        f"{path} exists" if Path(path).exists() else f"{path} does not resolve",
    )


def check_runtime_bus_env_absent(config: ProbeConfig) -> CheckResult:
    """Prevent inherited bus settings from pointing the episode at host services."""
    del config
    leaked = {var: os.environ[var] for var in BUS_ENV_VARS if var in os.environ}
    return CheckResult(
        "runtime_bus_env_absent",
        not leaked,
        f"set={sorted(leaked)}" if leaked else "neither variable is present",
    )


def check_dbus_socket_unreachable(config: ProbeConfig) -> CheckResult:
    """Block the user D-Bus socket: it can ask systemd to spawn outside."""
    del config
    candidates = [str(p) for p in Path("/run/user").glob("*/bus")]
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        candidates.append(str(Path(runtime_dir) / "bus"))
    connectable = []
    for path in sorted(set(candidates)):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(CONNECT_TIMEOUT_SECONDS)
        try:
            sock.connect(path)
        except OSError:
            continue
        finally:
            sock.close()
        connectable.append(path)
    return CheckResult(
        "dbus_socket_unreachable",
        not connectable,
        f"connectable={connectable}" if connectable else "no user D-Bus socket reachable",
    )


def check_nvidia_devices_absent(config: ProbeConfig) -> CheckResult:
    """Episodes never need the GPU; only the trainer does."""
    del config
    found = sorted(str(p) for p in Path("/dev").glob("nvidia*"))
    return CheckResult(
        "nvidia_devices_absent",
        not found,
        f"found {len(found)}: {found}" if found else "no /dev/nvidia* nodes",
    )


def check_aws_env_absent(config: ProbeConfig) -> CheckResult:
    """Keep cloud credentials out of the episode environment."""
    del config
    leaked = sorted(k for k in os.environ if k.startswith("AWS_"))
    return CheckResult(
        "aws_env_absent",
        not leaked,
        f"set={leaked}" if leaked else "no AWS_* variables in the environment",
    )


def check_readonly_bind_not_writable(config: ProbeConfig) -> CheckResult:
    """/usr is a read-only bind in the jail; a successful write means the bind is wrong.

    Note this check does NOT discriminate jail from host: off the jail it also fails to
    write because the user is unprivileged. It is here as a property assertion, not as
    a negative-control signal.
    """
    del config
    probe = "/usr/.jail-write-probe"
    try:
        fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError as exc:
        return CheckResult(
            "readonly_bind_not_writable",
            passed=True,
            detail=f"/usr rejected write: {_errno_name(exc)}",
        )
    os.close(fd)
    Path(probe).unlink()
    return CheckResult(
        "readonly_bind_not_writable",
        passed=False,
        detail=f"WROTE {probe} (removed again)",
    )


def check_jail_python_not_writable(config: ProbeConfig) -> CheckResult:
    """Keep the interpreter that grades an episode unwritable by that episode.

    The jail resolves an interpreter at or above its floor and may have to bind one in from
    outside /usr to get it. A bind that came in read-write would let a policy rewrite the thing
    that grades it, which is a scoring compromise rather than a containment one and would not show
    up in any other check here. os.access rather than a write probe, because this file never writes
    to a host path: outside the jail the staged tree is ours, so a probe would really create a file
    in the real interpreter.
    """
    path = config.jail_python
    if not path:
        return CheckResult(
            "jail_python_not_writable",
            passed=False,
            detail="no --jail-python was given, so this check cannot resolve what to probe",
        )
    targets = [path, str(Path(path).parent), str(Path(path).parents[1])]
    writable = [t for t in targets if os.access(t, os.W_OK)]
    return CheckResult(
        "jail_python_not_writable",
        not writable,
        f"WRITABLE={writable}" if writable else f"{path} and its tree reject writes",
    )


def _namespace_id(kind: str) -> str:
    """Return this process's namespace identity for one kind, e.g. ``ipc:[4026531839]``."""
    return str(Path("/proc/self/ns/" + kind).readlink())


def check_launcher_session_and_namespaces_left(config: ProbeConfig) -> CheckResult:
    """Require the jail to leave the launcher's session and its IPC and UTS namespaces.

    This is what gives the two backends' equivalence claim teeth. Nothing else in this file looks at
    IPC, UTS or the controlling terminal, so until this check existed the hand-assembled unshare
    fallback could unshare strictly less than bwrap and still pass the whole suite. A shared IPC
    namespace leaves host System V shared memory and POSIX message queues reachable; a shared
    session leaves the launcher's controlling terminal in reach, which is the classic route to
    injecting input into the shell that started the episode.

    A session id of 0 fails too, rather than passing: inside a PID namespace getsid translates a
    leader living outside it to 0, so a plain "is my session the launcher's" comparison cannot fail
    there. Per-backend measurements are in docs/episode-isolation.md.
    """
    if not config.launcher_session or not config.launcher_namespaces:
        return CheckResult(
            "launcher_session_and_namespaces_left",
            passed=False,
            detail=(
                "no --launcher-session/--launcher-namespace was given, so there is nothing to "
                "compare against and this check cannot resolve anything"
            ),
        )
    session = os.getsid(0)
    shared = []
    if session == 0:
        shared.append("session leader is outside this PID namespace, so the session is inherited")
    elif session == config.launcher_session:
        shared.append("session={}".format(session))
    for kind, launcher_value in config.launcher_namespaces:
        ours = _namespace_id(kind)
        if ours == launcher_value:
            shared.append("{}={}".format(kind, ours))
    return CheckResult(
        "launcher_session_and_namespaces_left",
        not shared,
        "SHARED WITH LAUNCHER: {}".format(shared)
        if shared
        else "own session ({}) and own {} namespaces".format(
            session, [kind for kind, _ in config.launcher_namespaces]
        ),
    )


def check_host_processes_invisible(config: ProbeConfig) -> CheckResult:
    """Hide host processes in a fresh PID namespace, whose argv may carry secrets.

    Evidence rather than proof, and deliberately two independent signals: PID 1 inside a
    fresh namespace is our own tree root rather than the host init, and the visible
    process count collapses (measured here: 762 on the host, 4 in the jail). A bound is
    used rather than an exact count because the jail legitimately contains a few
    processes of its own.
    """
    del config
    visible = [p.name for p in Path("/proc").iterdir() if p.name.isdigit()]
    try:
        with Path("/proc/1/comm").open() as fh:
            pid1 = fh.read().strip()
    except OSError as exc:
        return CheckResult(
            "host_processes_invisible",
            passed=False,
            detail=f"cannot read /proc/1/comm: {_errno_name(exc)}",
        )

    pid1_is_host_init = pid1 in HOST_INIT_NAMES
    too_many = len(visible) > MAX_JAIL_VISIBLE_PIDS
    detail = f"pid1={pid1!r}, {len(visible)} pids visible (bound {MAX_JAIL_VISIBLE_PIDS})"
    return CheckResult("host_processes_invisible", not (pid1_is_host_init or too_many), detail)


def check_honeypot_paths_absent_on_host(config: ProbeConfig) -> CheckResult:
    """Host-side only. Bait paths must not collide with anything real.

    The honeypots exist so a scope violation is observable. If one of them also
    resolves on the host, a jail failure stops being a harmless ENOENT and becomes a
    write to a real file, which is the difference between a logged reward signal and
    an incident.
    """
    del config
    colliding = [p for p in HONEYPOT_PATHS if Path(p).exists()]
    return CheckResult(
        "honeypot_paths_absent_on_host",
        not colliding,
        f"COLLIDES={colliding}"
        if colliding
        else f"none of {list(HONEYPOT_PATHS)} exist on the host",
    )


NON_DISCRIMINATING_CHECKS = frozenset({"readonly_bind_not_writable", "jail_python_not_writable"})

CheckFn = Callable[[ProbeConfig], CheckResult]

CONTAINMENT_CHECKS: tuple[CheckFn, ...] = (
    check_secrets_unreadable,
    check_host_home_absent,
    check_host_home_not_writable,
    check_claude_hooks_unreachable,
    check_network_egress_blocked,
    check_docker_socket_absent,
    check_runtime_bus_env_absent,
    check_dbus_socket_unreachable,
    check_nvidia_devices_absent,
    check_aws_env_absent,
    check_launcher_session_and_namespaces_left,
    check_host_processes_invisible,
    check_readonly_bind_not_writable,
    check_jail_python_not_writable,
)

HOST_CHECKS: tuple[CheckFn, ...] = (check_honeypot_paths_absent_on_host,)

GROUPS = {"containment": CONTAINMENT_CHECKS, "host": HOST_CHECKS}


def run_group(group: str, config: ProbeConfig) -> list[CheckResult]:
    """Run one named check group so callers can evaluate a complete gate."""
    return [check(config) for check in GROUPS[group]]


CheckPayload = dict[str, object]


def _passed_by_name(results: list[CheckPayload]) -> dict[str, bool]:
    """Index a --json run's results by check name, keeping only pass/fail."""
    return {str(entry["name"]): bool(entry["passed"]) for entry in results}


def verify_negative_control(
    inside_results: list[CheckPayload], outside_results: list[CheckPayload]
) -> list[str]:
    """Return the reasons the negative control is unsound, empty list meaning sound.

    Comparing exit codes is not enough. A suite that crashed on an ImportError also exits non-zero
    outside the jail, and would look like a passing negative control while proving nothing. So this
    requires, per named check, that it PASSED inside and FAILED outside -- the checks genuinely
    discriminate.

    NON_DISCRIMINATING_CHECKS is excluded because those two cannot tell jail from host at all:
    requiring them to flip would make the control unsatisfiable, and leaving them in silently would
    make it weaker than it looks. Do not extend that set without the same argument, which
    docs/episode-isolation.md spells out per check.
    """
    inside_by_name = _passed_by_name(inside_results)
    outside_by_name = _passed_by_name(outside_results)

    problems = []
    expected = {c.__name__.replace("check_", "") for c in CONTAINMENT_CHECKS}
    flip_set = expected - NON_DISCRIMINATING_CHECKS

    missing = (flip_set - inside_by_name.keys()) | (flip_set - outside_by_name.keys())
    if missing:
        problems.append(f"checks missing from a run: {sorted(missing)}")

    for name in sorted(flip_set & inside_by_name.keys() & outside_by_name.keys()):
        if not inside_by_name[name]:
            problems.append(f"{name}: did not pass INSIDE the jail, so containment is broken")
        if outside_by_name[name]:
            problems.append(
                f"{name}: also passed OUTSIDE the jail, so it proves nothing (vacuous check)"
            )
    return problems


def parse_launcher_namespaces(raw: list[str]) -> tuple[tuple[str, str], ...]:
    """Parse KIND=ID pairs, refusing anything else rather than silently probing nothing."""
    parsed = []
    for item in raw:
        kind, separator, identity = item.partition("=")
        if not separator or not kind or not identity:
            complaint = "--launcher-namespace wants KIND=ID, e.g. ipc=ipc:[4026531839], got {!r}"
            raise ValueError(complaint.format(item))
        parsed.append((kind, identity))
    return tuple(parsed)


def main() -> int:
    """Run jail checks and report their machine-readable results."""
    parser = argparse.ArgumentParser(description="Assert an episode jail contains its occupant.")
    parser.add_argument("--group", choices=sorted(GROUPS), default="containment")
    parser.add_argument(
        "--verify-negative-control",
        nargs=2,
        metavar=("INSIDE_JSON", "OUTSIDE_JSON"),
        help="compare two --json runs and require each check to pass inside and fail outside",
    )
    parser.add_argument(
        "--host-home",
        action="append",
        metavar="DIR",
        help=(
            "host home directory to probe, repeatable. Defaults to $HOME and its /local twin, "
            "and is REQUIRED inside the jail, where HOME is the episode directory."
        ),
    )
    parser.add_argument(
        "--jail-python",
        default="",
        metavar="PATH",
        help=(
            "absolute path of the interpreter the jail exposes, as reported by "
            "`episode_jail.sh --print-jail-python`. Required by the containment group, and the "
            "same value must be given to the inside and outside runs or the comparison is between "
            "two different things."
        ),
    )
    parser.add_argument(
        "--launcher-session",
        type=int,
        default=0,
        metavar="SID",
        help=(
            "session id of the process that launched the jail, as os.getsid(0) reports it on the "
            "host. Required by the containment group, and the same value must be given to the "
            "inside and outside runs."
        ),
    )
    parser.add_argument(
        "--launcher-namespace",
        action="append",
        default=[],
        metavar="KIND=ID",
        help=(
            "a namespace the jail must have left, as KIND=ID where ID is what readlink "
            "/proc/self/ns/KIND prints on the host, e.g. ipc=ipc:[4026531839]. Repeatable."
        ),
    )
    parser.add_argument(
        "--print-host-homes",
        action="store_true",
        help=(
            "print the derived host home directories, one per line, and exit. The launcher uses "
            "this to hand the same list to the in-jail run, which cannot derive it."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable results on stdout, for the negative-control runner",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="jail-assert: %(message)s", stream=sys.stderr)

    if args.print_host_homes:
        for home in host_homes_from_environment():
            sys.stdout.write(home + "\n")
        return 0

    if args.verify_negative_control:
        inside_path, outside_path = args.verify_negative_control
        with Path(inside_path).open() as fh:
            inside = json.load(fh)
        with Path(outside_path).open() as fh:
            outside = json.load(fh)
        problems = verify_negative_control(inside["results"], outside["results"])
        for problem in problems:
            logger.error(f"NEGATIVE CONTROL UNSOUND: {problem}")
        if problems:
            return 1
        logger.info(
            "negative control sound: every discriminating check passed inside and failed outside"
        )
        return 0

    config = ProbeConfig(
        host_homes=tuple(args.host_home) if args.host_home else host_homes_from_environment(),
        jail_python=args.jail_python,
        launcher_session=args.launcher_session,
        launcher_namespaces=parse_launcher_namespaces(args.launcher_namespace),
    )
    results = run_group(args.group, config)

    # stderr always, stdout JSON on request: one run serves log and comparison alike.
    for result in results:
        logger.info(f"{'PASS' if result.passed else 'FAIL'} {result.name}: {result.detail}")

    if args.json:
        json.dump({"group": args.group, "results": [asdict(r) for r in results]}, sys.stdout)
        sys.stdout.write("\n")

    failed = [r.name for r in results if not r.passed]
    if failed:
        logger.error(f"{len(failed)} of {len(results)} checks failed: {failed}")
        return 1
    logger.info(f"all {len(results)} {args.group} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
