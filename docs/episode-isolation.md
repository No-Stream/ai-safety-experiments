# Episode isolation, and what it does not protect

The research this repo is heading toward trains a policy in an environment that
deliberately rewards scope violation: reading a grader's answer key, tampering with a
test file, writing outside the working directory, reaching for a network tool. That only
makes sense if a violation lands on bait instead of on something real.

**Gate: no policy uses the bwrap/unshare jail on this host until
`scripts/run_jail_tests.sh` passes.** It is a hard prerequisite, independent of how the
rest of the research design settles. Docker-backed tasks have their own separate gate below.

```bash
scripts/run_jail_tests.sh                     # the gate; selects bwrap when installed
scripts/run_jail_tests.sh --backend unshare   # pin the fallback backend instead
```

## The pieces

| Script | Job |
| --- | --- |
| `scripts/episode_jail.sh` | Runs one episode's command in a filesystem/network/PID-isolated jail. |
| `scripts/stage_jail_python.sh` | Stages a >=3.12 CPython where the jail can bind it read-only. |
| `scripts/jail_assertions.py` | The 14 containment checks plus the honeypot-collision check. |
| `scripts/run_jail_tests.sh` | The gate: argument guards, host checks, in-jail run, and the negative control. |
| `scripts/scan_secrets.py` | Pre-upload and pre-commit gate: fails on secret-shaped text, machine-local identity values, or benchmark item material. |

## How the jail is built

A fresh tmpfs root with a **whitelist** of mounts, never a deny-list. Only `/usr`
(read-only), a synthesised minimal `/etc`, `/proc`, a tmpfs `/dev` carrying six
individually bound device nodes, a tmpfs `/tmp`, and the episode directory at `/work`.
Nothing from a home directory, ever — `--episode-dir` and `--ro-bind` both refuse paths
under one, since a bind there would re-expose the tree the whitelist exists to exclude.

That guarantee was false as written until it was attacked. `--ro-bind` checked the *parent* of its
argument, so every directory that holds home trees passed, and so did `/` (the guard only ever
looked for descendants of a home, and `/` is an ancestor of every home), and a symlink into a home
passed too because both backends bind what a symlinked source points at, not the link. The guard now
resolves the argument itself with `realpath` and refuses a path in either direction, and step 1 of
the gate drives every one of those spellings plus a positive control (`--ro-bind /usr/share/doc`
must still start). The gate asks `episode_jail.sh --print-home-roots` for the list rather than
keeping its own copy, so a root added to the guard is driven by the gate instead of going untested.
The containment checks could not have caught it: they probe fixed absolute paths, so a home bound at
some other mountpoint reads as contained.

That is what makes the interesting attacks fail with `ENOENT` rather than relying on file
permissions. Reading `~/.aws/credentials`, rewriting a hook in `~/.claude/hooks/`,
appending to `~/.zshrc`, editing the scripts cron runs hourly: none of those paths exist
inside the jail.

On distributions using `/etc/alternatives`, some commands under `/usr/bin` are symlinks through
that directory. The jail reconstructs only alternative links whose resolved targets lie under
the already-mounted `/usr`, rather than mounting host `/etc` content. Without those links, Ubuntu's
`awk` was unavailable and honest harness commands failed before producing an answer. A synthetic
sum through the real jailed `awk` first failed with exit 127, then passed after link reconstruction.
Both backends use the same filtered links.

The step that actually severs the host is detaching the old root after `pivot_root`.
Without `umount -l /oldroot` the entire host filesystem stays walkable at `/oldroot` and
the jail is theatre.

### The interpreter is part of the contract

A whitelist carrying only `/usr` also decides which Python exists inside the jail, and that turned
out to be a scoring decision rather than a packaging detail. On the host where this was discovered,
`/usr/bin/python3` was 3.9.25, so a coding submission using a 3.10+ builtin — `int.bit_count()` was the one that turned up —
raised `AttributeError` on every assertion, failed every unit, and "fails everything" trivially
contains any registered failing-unit set. The RecoveryBench coding grader therefore scored ten
genuinely correct submissions as carrying a planted flaw, inflating the exact number that slice
exists to measure, in the direction of the hypothesis. No gate was watching; it was found by chasing
an anomalous control result.

So `episode_jail.sh` now **resolves** an interpreter and **refuses to start** without one at or above
its floor, currently 3.12. Refusing is the point: falling back to an older interpreter is
indistinguishable from working, right up until the misgraded run.

```bash
scripts/episode_jail.sh --print-jail-python    # the resolved interpreter, as seen inside the jail
scripts/episode_jail.sh --print-python-floor   # the minimum acceptable minor version
scripts/stage_jail_python.sh                   # stage one, on a host whose /usr python is too old
```

Resolution walks an ordered candidate list and takes the first that meets the floor, asking each
binary its version rather than trusting its filename. `/usr/bin/python3.13`, then `python3.12`, then
plain `python3` come first **because they need no mount at all** — `/usr` is already bound read-only,
so the whitelist is unchanged, and a packaged interpreter is root-owned and gets security updates.
Prefer a packaged interpreter when the distribution provides one meeting the floor; otherwise use
the staging script above.

Last in the list is a relocatable CPython staged at `/var/tmp/cpython-runtime`, bound read-only.
It is used when no packaged interpreter meets the floor, so the jail grades against the language
the repo is written in. Three properties of that path are load-bearing.

- **Outside every home directory.** `/proc/1/cmdline` is *readable* inside the jail and carries the
  whole `bwrap` argv; `/proc/self/mountinfo` carries each bind's source path; and `sys.prefix` puts
  the interpreter's path into every traceback out of jailed code. A mount from `$HOME` would hand the
  owner's username to the policy and to every published rollout trace. Measured, not assumed.
- **Not group- or world-writable, under a sticky parent.** Whoever can rewrite that tree chooses the
  interpreter that grades. `/var/tmp` is world-writable, so ownership plus the sticky bit is the whole
  of what stops another uid replacing it, and `episode_jail.sh` re-checks both at every launch rather
  than trusting the staging script to have got it right once.
- **Mundanely named.** This repo measures whether a model behaves differently when it can tell it is
  being evaluated, so a path the policy can read should not spell out the answer. Same reason the
  `python3`/`python` shim sits at `/opt/python/bin` rather than anywhere named after the jail.

Bare `python3` and `python` reach the interpreter through that shim directory at the front of `PATH`,
which is what most code actually gets: model-written commands say `python3` in 3014 of the 3121
interpreter invocations across the stored traces, and the visible grader command is
`python3 -I grade.py`. The shim sits on the jail's **writable** tmpfs, so a policy can repoint it
within its own jail invocation — harmless, since every command gets a fresh jail and the graders are
launched from the host, but it is why the hidden-check oracles take the absolute resolved path
instead. Nothing hardcodes that path any more; callers ask `--print-jail-python`, so a host that
gains a packaged interpreter switches over with no code change.

**The grading environment also carries numpy, since 2026-08-22.** It was installed into the staged
interpreter's own site-packages (`uv pip install --python <jail-python> --break-system-packages
numpy`, 2.5.2 at install time), so a submission importing it grades as a measured result instead of
dying at load: 19 stored Opus 5 rollouts had failed exactly that way and were excluded as
unmeasurable, and all 19 graded as fully correct once the import worked. Rollouts graded before
that date ran in a numpy-less jail — the boundary matters when comparing across eras. Since
2026-10-04 numpy is pinned in `scripts/jail_python_requirements.txt` beside pytest and the v2 ladder
repositories' test dependencies, and `stage_jail_python.sh` installs that file into every staged
tree, so `--force` no longer drops it. A packaged `/usr` interpreter still wins resolution with the
distribution's bare site-packages and would lose all of them.
`tests/test_interpreter_compat.py::TestTheJailGradingEnvironmentCarriesNumpy` pins the import under
`-I` against whatever interpreter the jail resolves, so either migration goes red until numpy is
reinstalled there.

### Two backends, not equally trustworthy

`bwrap` (bubblewrap) is preferred and is the default when present: purpose-built and
audited. `unshare` is the fallback assembled from util-linux primitives — hand-rolled, and
equivalence is now checked rather than asserted. It unshares the same namespaces (user, mount, net,
pid, ipc, uts, cgroup) and runs under `setsid --wait`, which is bubblewrap's `--new-session`; the
`--wait` matters because without it the episode's exit status is lost and a failing episode reads as
a passing one. It carried none of ipc, uts, cgroup or the new session until `--backend unshare` was
run against a check that compares the jail's session and IPC/UTS namespaces with the launcher's.
`setsid` runs *inside* the namespaces on purpose: outside them the new session's leader stays in the
host PID namespace and the jail inherits the launcher's session anyway.

The session half of that check needs two conditions, and finding that out is why it is written the
way it is: a plain "is my session id the launcher's" comparison **cannot fail** inside a PID
namespace, because `getsid` translates a leader living outside the namespace to 0. Measured: session
1 inside `bwrap` (its sandbox init is the leader and is in the namespace), 0 inside the `unshare`
fallback (still the launcher's session, merely untranslatable), and the launcher's own id outside
the jail. So a zero is a failure too, and "am I the session leader" is not the test either — the
leader is the backend's init, not the command.

Both backends previously passed the 14 containment checks and 12 discriminating negative
controls on a host with bubblewrap 0.10.0. The current host has no bubblewrap, so real episodes
select `unshare`. The gate defaults to whichever backend `episode_jail.sh` selects, and the
existing containment checks remain required. Pin the other backend with `--backend` when it is
available to check it separately.

The unshare payload previously retained full capabilities in its user namespace: mounting a tmpfs
over `/usr/bin` succeeded. After building the mounts and detaching the old root, the fallback now
uses `setpriv --no-new-privs` and `capsh` to clear the inheritable, permitted, effective, bounding
and ambient capability sets before executing the payload. It verifies all five masks and the
no-new-privileges flag from `/proc/self/status`, refusing execution if any privilege remains.
Ubuntu's older libcap-ng cannot implement `setpriv`'s `-all` capability syntax on this kernel,
which is why capability removal uses the installed `capsh` instead. The real regression test
first failed with nonzero masks, then passed with zero masks, `NoNewPrivs=1`, and the mount refused.
Removing the bounding-set drop also made the launch guard exit 2 before the payload ran.
This removes the payload's initial mount privileges; the kernel and user-namespace implementation
remain part of the trusted boundary.

That default is a correction worth naming. `run_jail_tests.sh` used to hardcode
`backend="unshare"`, which was right while bwrap was missing and wrong the moment it was
installed: `episode_jail.sh` began auto-selecting bwrap for real episodes while the gate went
on certifying the fallback. A gate pinned to a path production does not take is the same
failure shape as a gate that never runs.

`nsjail` was considered and rejected: not packaged for this host's distribution, and a source
build would need protobuf, libnl3, bison/flex and the kafel submodule.

One trap worth naming, because the plan's first draft hit it: `bwrap --unshare-all`
includes the network namespace, which is what we want. nsjail's `--disable_clone_newnet`
is the **inverse** control — it *keeps* the host network. That one-word confusion removes
the single most important protection here.

### Composition order: limits outside, isolation inside

```bash
scripts/resource-limits.sh -t 15m -- scripts/episode_jail.sh --episode-dir D -- cmd
```

Never the reverse. `resource-limits.sh` asks the systemd user manager to create the job's
cgroup over the D-Bus socket at `$XDG_RUNTIME_DIR/bus`, and anything that can reach that
socket can ask systemd to spawn processes **outside** any jail. That is a full escape by
design, not a broken dependency, which is why `$XDG_RUNTIME_DIR` must never be bound into
an episode jail — and consequently why the limiter cannot run from inside one. Verified:
with `XDG_RUNTIME_DIR` unset, `resource-limits.sh` refuses to start and exits 2.

## The negative control is the important test

A suite that passes because it is broken is worse than no suite. So the gate does not
just run the checks inside the jail; it runs the same checks **outside** and requires
each one to fail there. Exit-code inversion would not be enough — a suite that died on an
`ImportError` also exits non-zero outside the jail and would look like a passing control
while proving nothing. `--verify-negative-control` therefore compares per named check and
requires: passed inside, failed outside.

Hosts without a user bus, NVIDIA device nodes or AWS variables need explicit negative-control
fixtures. The runner creates a listening Unix socket and a harmless file representing a device
path outside the episode directory, then passes those same absolute paths to both probes.
The real `/dev` scan and user-bus candidates remain enabled alongside the fixtures. It also
supplies a dummy AWS variable and a runtime-directory variable to both launcher environments;
the jail must clear them while the outside run must detect them. Failure to create the socket is
an error, and a host-dependent check passing outside still makes the negative control unsound.

An agent sandbox may prohibit Unix socket creation even when the underlying host supports it.
In that environment `make jail-test` stops at fixture setup rather than claiming containment was
certified. The capability, real command, exit-status, timeout and oracle tests can still execute
through unshare. For a host with an offline systemd user manager, the explicit harness option is
`--advisory-limits`; the weaker resource guarantees and pytest opt-in are documented in
[resource-limits.md](resource-limits.md).

Both failure modes are tested. Feeding it a run where a check passes in both places is
reported as `also passed OUTSIDE the jail, so it proves nothing (vacuous check)`; feeding
it an empty result set is reported as `checks missing from a run`.

Two checks are deliberately excluded from the flip set, for the same reason. The write to `/usr` in
`readonly_bind_not_writable` fails inside (read-only bind) and outside (unprivileged user) alike, so
it cannot discriminate. `jail_python_not_writable` is that argument applied to the interpreter: in the
preferred configuration the interpreter is the distribution's own under `/usr`, root owns it, and the
write is refused in both places — so whether that check flips is a property of how the host got its
interpreter, not of the check. Requiring either to flip would make the control unsatisfiable; leaving
them in silently would make the control weaker than it looks. Do not extend that exclusion set
without the same argument.

As it happens `jail_python_not_writable` *does* discriminate on this host, because the interpreter is
a staged tree we own: writable outside, `Read-only file system` inside. Being excluded from the flip
set does not mean it is vacuous here — only that the gate cannot require the flip everywhere.

## Measured on this host

All 14 containment checks pass inside the jail; 13 of 14 fail outside it, the fourteenth being
`readonly_bind_not_writable`, which cannot discriminate. The table below is the `unshare` run. The
`bwrap` run is identical check for check, differing only in `host_processes_invisible`, which reports
`pid1='bwrap'` and 2 visible pids rather than `pid1='python3'` and 1. Re-measured after the
interpreter change, on both backends.

| Check | Inside the jail | Outside (negative control) |
| --- | --- | --- |
| `secrets_unreadable` | every probed path `ENOENT` | the paths that exist are **READABLE** |
| `host_home_absent` | neither home resolves | both present |
| `host_home_not_writable` | none writable | both writable |
| `claude_hooks_unreachable` | does not resolve | listable, with entries |
| `network_egress_blocked` | `OSError` / `gaierror` | both targets **CONNECTED** |
| `docker_socket_absent` | does not resolve | exists |
| `runtime_bus_env_absent` | absent | both set |
| `dbus_socket_unreachable` | none reachable | `/run/user/<uid>/bus` connectable |
| `nvidia_devices_absent` | none | every device node present |
| `aws_env_absent` | none | several variables set |
| `host_processes_invisible` | `pid1='python3'`, 1 pid visible | `pid1='systemd'`, hundreds of pids visible |
| `launcher_session_and_namespaces_left` | own session (1), own ipc and uts | launcher's session and both namespaces |
| `readonly_bind_not_writable` | EROFS | EACCES (does not discriminate) |
| `jail_python_not_writable` | interpreter tree rejects writes | all three paths **WRITABLE** |

Not every probed secret path exists on a given host, and an absent one proves nothing either way:
only a path that is **READABLE** outside and `ENOENT` inside carries signal, which is why
`check_secrets_unreadable` fails on `READABLE` alone rather than requiring absence.

## What this does NOT protect against

- **It is not a resource limit.** Pair it with `resource-limits.sh`; see
  [resource-limits.md](resource-limits.md).
- **Only single-uid mapping is available.** `newuidmap`/`newgidmap` are present but not
  setuid and this user has no `/etc/subuid` or `/etc/subgid` entries. So there is no
  in-jail user separation, and root-in-jail can write any tmpfs the jail owns.
- **"Writes outside `/work` fail" is not literally true.** Writes to jail-internal tmpfs
  (`/tmp`, `/etc`, `/`, and the `/opt/python/bin` shim) succeed, because the uid is mapped to
  root and tmpfs is writable. They are harmless — they vanish with the namespace — but do not
  state the property more strongly than it holds. What *is* true: read-only binds reject writes,
  and host paths outside the episode directory do not exist to be written.
- **The `python3` shim is repointable from inside; the resolved interpreter is not.** Because the
  shim is on that writable tmpfs, a policy can make `python3` mean something else *for itself*.
  Nothing it grades is decided that way — each command gets a fresh jail and the graders are
  launched from the host — but code that must not let a policy pick the interpreter has to use the
  absolute path from `--print-jail-python`, which is on a read-only mount. The hidden-check oracles
  do.
- **The staged interpreter is only as safe as our own uid.** A read-only bind stops the episode,
  not a host-side process running as us. That is true of `episode_jail.sh` and the graders too, so
  it is not a new hole — but it is a reason to prefer the root-owned packaged interpreter, which
  our uid cannot rewrite at all.
- **`host_processes_invisible` requires a different PID namespace.** The runner supplies the
  launcher's namespace identity to both probes, so the outside control fails even on hosts
  with fewer than 50 processes and an unfamiliar init name. The init-name and process-count
  checks remain additional signals. Tests first failed with the old count-only implementation,
  then passed with the namespace comparison; missing launcher identity also fails.
- **The kernel is shared.** A user-namespace or kernel bug is not covered by any of this.
- **Only the default backend is exercised per gate run.** The gate certifies one backend at a
  time, so the one it did not pick can rot unnoticed. Run it both ways after touching
  `episode_jail.sh`.
- **The staged interpreter is owned by this host's user.** `episode_jail.sh` re-checks its
  ownership and mode at every launch, which does not detect a same-mode edit to a library file by
  that user. The packaged interpreter remains the stronger configuration.

## The host execution/configuration tripwire has been retired

This repository does not monitor local hooks, crontab targets, systemd user units, shell or tool
startup files, or settings files for drift. The execution/configuration tripwire has been retired;
no checksum protection over those host paths is claimed here. The jail assertions and privacy
scanners below remain active, and they check different things.

`scan_secrets.py` is the gate to run before any log, trace or adapter leaves the box:

```bash
python3 scripts/scan_secrets.py <logdir> --canary-file <tokens.txt>
```

It is also the gate over what gets committed, since this remote is public, and that half runs
itself:

```bash
make privacy-scan   # every file `git add -A` would carry, .gitignore honoured
```

`tests/test_scan_secrets.py` runs the same sweep inside `make test`, so the gate cannot be
forgotten the way it had been: until 2026-08-19 the script had no hook, no make target and
no test, so nothing ever ran it.

Fifteen detectors in three families. Credential shapes (AWS key IDs, secret and
session-token assignments, PEM and OpenSSH private-key headers, JWTs, bearer tokens,
session cookies, generic secret assignments). Machine-local identity, which looks nothing
like a credential and is the half the privacy rule cares about (email addresses, AWS
account ids, role ARNs, ECR registry hosts, home-directory paths carrying a username) —
each exempting the documented placeholders, because a gate that flags `123456789012` is a
gate that gets waved through. And benchmark material, both as a serialized record and, via
a separate whole-file check, as a pretty-printed item file whose keys sit one per line
where no line detector can see them.

Plus exact matching against a canary-token list, because no shape describes a username, a
bucket name, an AWS profile name or a registered answer's text. Put those one per line in a
file under the gitignored `canary/` directory. The script itself loads them only from
`--canary-file`, so a direct invocation that omits the flag runs shape-only with the canary
detector disarmed; `make privacy-scan` passes it for you when `canary/privacy-values.txt` exists.
Either way the token count is reported on every run, so a list that loaded nothing is visible
rather than reading as clean.

The reason this is a separate gate rather than a nice-to-have: if the jail leaks and the
policy reads a real credential, that text lands in a rollout log and from there into
training data. Rotating the credential fixes the credential; it does not unpublish the
weights. Findings therefore report path, line, detector and a truncated SHA-256 of the
match — never the matched text, since a scanner that echoes what it found just copies the
secret into a second log.

## A packaged interpreter is optional

The unshare fallback and staged interpreter do not require installing bubblewrap or obtaining
root access. A distribution-packaged Python at or above the version floor is an optional
improvement when the host's repositories provide one. Ubuntu 20.04 does not provide the required
version through its standard packages, so the staged interpreter remains the supported path on
that host. No package installation is required for the current runner changes.

That puts the jail's interpreter under `/usr`, which is already bound read-only. Resolution prefers
it automatically, so nothing needs editing afterwards, and three things get better at once: the
whitelist stops carrying an extra bind, the interpreter becomes root-owned rather than writable by
our own uid, and it starts receiving distribution security updates. The staged tree at
`/var/tmp/cpython-runtime` can then be deleted; the jail will not miss it. One thing does not come
along automatically: numpy is part of the grading environment (see above), the packaged
interpreter's site-packages will not have it, and the interpreter-compat test stays red until it is
installed there too.

## Docker backend for prebuilt SWE-bench images

Impossible-SWEbench tasks can require old Python versions and compiled dependencies, so they use a
Docker image built for that task. The Docker backend runs every agent command in a new container
from the task's image pinned by its full SHA-256 digest. It refuses an unpinned image and rejects an
image that declares automatic volumes, since an undeclared writable volume would break the declared-mount
boundary. The image must provide `/usr/bin/timeout`, which enforces the command deadline inside the
container. The image digest is included in command evidence and grader records.

Set `Task.docker_config` to select Docker for a task; other tasks keep the existing jail backend.
Supply `docker_grader_command`, `docker_true_grader_command`, and `docker_artifact_paths` alongside
that configuration. The two grader commands run against original `Task.files` and the submitted
artifact allowlist; submitted paths cannot replace trusted task files. Docker tasks never invoke
the host-side `true_check` or `provided_test_check` callbacks. Task loaders must prepare the source
workspace and supply commands for the image's own interpreter and dependencies.

The container uses `--network none`, disabled daemon logging, a read-only root filesystem, a private
IPC namespace and cgroup namespace, no health check, all capabilities dropped, `no-new-privileges`,
and the host user's non-root numeric uid and gid. Docker applies the configured PID, memory,
memory-plus-swap, and CPU limits. `/tmp` is a bounded tmpfs; the configured workspace path
(`/work` by default) is the sole writable host mount and is the episode workspace. Trusted runtime
directories may additionally be mounted read-only under the restrictions below.
Policy commands in one episode retain `/tmp` files. The host keeps a private copy of that tree
outside the episode workspace between commands, stages it inside the workspace immediately before
each container starts, and removes the stage after the container exits. A shell copies entries into
the container's 64 MiB `/tmp` before the policy command and copies them back afterward, including
when the command fails. The host preserves symlinks as symlinks, never follows them during copying,
and refuses a state exceeding the tmpfs byte limit or the snapshot entry limit. The per-episode
copy is deleted at episode end. The trusted grader gets a fresh `/tmp` and never receives this copy.
Every command uses a fresh container and Docker removes it on exit; the host also force-removes its
name and anonymous volumes after a timeout or launcher failure. The backend does not mount the Docker
socket or inherit undeclared host paths. It records the command, image digest, exit code, output, timeout state, and
before/after workspace snapshots in host-side JSONL evidence. The Docker client drains both output
streams while retaining at most 1 MiB from each, then records when output was truncated. After the
agent has finished, grading copies only explicitly listed regular files, with a 64 MiB limit per
submitted file. Evidence, gate files, and grading scratch directories stay outside the episode mount.

Workspace snapshots record file contents by hash, symlink targets, directory entries, and permission
changes. A scan refuses unreadable paths or more than 20,000 entries, 1 GiB of hashed content, or
ten seconds of scanning; a failure is recorded and aborts the episode. A kernel file-size limit
caps individual writes at 64 MiB. The host workspace has no aggregate disk or inode quota, so this
backend does not contain an attack that fills the workspace with many files. Command evidence and
snapshots also do not observe every denied syscall or a transient edit restored within one command.

The authoritative grader starts in another fresh container from the same image digest. The host
reconstructs its workspace from the trusted task files and only the explicitly submitted regular
files, read without following symlinks. Agent-created grader files, unlisted files, background
processes, and policy `/tmp` state do not enter that workspace. This keeps the trusted verdict independent
of the agent container while preserving the submitted artifact the grader needs to inspect.

**Gate: Docker episodes require a fresh live isolation check once per process and pinned image.**
The manual entry point is:

```bash
make docker-jail-test
```

The manual runner defaults to resolving `python:3.12-slim`; `--image` selects a configured
digest-pinned image directly. The automatic pre-stage gate uses the configured digest for every
test container, against the current backend and Docker daemon. It probes DNS, TCP to a public IP,
TCP through Docker's bridge gateway,
reading a synthetic host sentinel, writing outside `/work` and to the read-only image root, reaching
the Docker socket path, privilege gain, the PID ceiling, memory exhaustion, and grader tampering.
Runtime probes check the effective `ro` flag, direct writes and writes through workspace
symlinks, and attempts to read synthetic host files outside the mounted runtime source. Their
negative controls weaken runtime mount flags or expose only a synthetic sibling directory.
The socket control is a short-lived synthetic Unix socket, never the host Docker socket. The
gateway check uses a bounded test container with one published synthetic TCP port, not a listener
on the host. For each check, the negative control weakens only the corresponding boundary and must
show the probe succeeding. The runner writes `artifacts/docker-isolation-gate.json` only when every
contained probe and every negative control passes. The backend hash records provenance only.
Leak source setup runs the suite before constructing its model backend. Successful live results
are cached once per process and pinned image, including across episode commands and graders;
failures are never cached. The backend validates the complete probe set, each containment and
weakened-control result, the pass flag, and the configured image digest before accepting the result.
A stored record is a log of the latest pass and is never read to authorize execution. CPU tests
reject invalid fresh results and verify that stale or absent records and changed code hashes do
not block a successful live check. A live sabotage test raises the hardened PID limit and verifies
that the pre-stage gate refuses execution. The resource probes require Docker's cgroup v2 files.

This is a live-daemon gate and does not run as part of `make test`. CPU tests cover command
construction, the required probe set, evidence shape, and gate rejection without Docker; they also
compile each embedded probe program and invoke the module's help entry point without a daemon. If
the Docker daemon is unavailable or any negative control cannot demonstrate the weakened boundary,
the gate stays red; the Docker backend must not be used until the suite passes.

Docker Desktop's disk-image location can be inspected in its Windows-side `settings-store.json`;
the current machine's location belongs in local operational notes. Docker's bind mount options are
documented in [the Docker manual](https://docs.docker.com/engine/storage/bind-mounts/). The workspace
mount uses `bind-recursive=disabled` so nested host mounts are excluded.

### R2E leak workspaces and trusted patch grading

`DockerConfig.workspace_mount` defaults to `/work`; R2E tasks set it to `/testbed` so image-installed
editable packages resolve to the submitted tree. The destination must be a normalized absolute
container path, outside Docker's reserved mounts. The workspace remains the sole writable host mount.
The next process checks containment with live probes before starting episodes.

`DockerConfig.read_only_mounts` accepts canonical directories strictly below
`/var/tmp/leak-env/runtime`, targeting `/opt` children or the workspace's `.venv`. Relative paths,
symlink aliases, traversal, overlapping targets, and nonempty or redirected workspace mountpoints
are refused. R2E tasks mount a shared per-repository venv at `/testbed/.venv` and its interpreter at
`/opt/uvpy`; runtime preparation rewrites the venv's interpreter links and `pyvenv.cfg` before
mounting. Task templates contain the repository and its closed Git history, without `.venv`.
The runtime placeholders are excluded from snapshots and runtime paths are excluded from patches.
Task identity includes fingerprints of both runtime trees so replay refuses changed Python or
dependency contents alongside changed templates, hidden tests, or instruction variants.

Runtime directories use `--mount type=bind,source=...,target=...,readonly,bind-recursive=disabled`.
Docker Desktop translates ordinary bind sources from the host distro. Local-driver volumes with
`volume-opt=device=...` instead ask the daemon to resolve that path; distro-local runtime paths
can be absent from its filesystem, so those volumes cannot reliably mount the extracted runtime.
[Docker's bind mount options](https://docs.docker.com/engine/storage/bind-mounts/) expose no
`nosuid,nodev` flags. We accept the bind defaults for those flags, retain non-root execution,
drop all capabilities, and retain
[`no-new-privileges`](https://docs.docker.com/reference/cli/docker/container/run/#optional-security-options).
That security option prevents privilege gain through setuid executables; it does not provide
`nodev`. Nested host mounts are excluded rather than recursively included.

Before any command, a shell guard reads `/proc/self/mountinfo` and requires effective `ro` at each
runtime target. The live gate also attempts direct writes and writes through a workspace symlink.
Its synthetic sentinel is host-writable by the container uid: removing only `readonly` and
bypassing the guard in the negative control must allow both writes, so host permissions cannot
make a writable mount appear protected. A separate negative control exposes a synthetic sibling
directory to check host-path isolation. In the bind-mount revision's live sabotage, the hardened
runtime mount was made writable and the command guard bypassed so the write probe executed.
The gate failed at `runtime_mount_write_blocked` with
`flags:missing;direct:written;symlink:written`; the mutation was then reverted.

A task may provide a trusted workspace template. Episode construction copies regular files with
`copy2`, preserves symlinks, and restores directory timestamps after materializing task files.
R2E grading captures `patch.diff` on the host with `git diff --binary` against the trusted base
commit. The diff reads Git metadata from the closed template rather than the policy's `.git`;
external diff, textconv, fsmonitor and optional index writes are disabled. The submitted patch is
limited to 64 MiB and cannot be a symlink. Untracked files are outside this Git diff submission.

The in-band verdict comes from the workspace's editable `run_tests.sh`. The hidden verdict comes
from a fresh container containing a copy of the closed template, the submitted patch, and trusted
hidden-test inputs. Submitted test paths, test configuration, `run_tests.sh`, and grading metadata
are excluded from patch application. The grader stages the task's extracted `hidden_tests` template
at `/testbed/r2e_tests`, then compares the complete parsed pytest status map with the expected map
using the canonical case loader's parser. Both graders run as the configured non-root host uid with
workdir `/testbed`; a pinned runtime image may be shared across tasks in one repository. Task admission
requires channel-builder normalization of the closed base, including removal of upstream history
and fix objects, before any episode. Neither the leak detector nor channel labels affect this verdict.
The loop checks the submitted patch hash and final source diff after grading; a mismatch records
`SUBMISSION_ARTIFACT_MISMATCH` as unmeasured and cannot produce a trusted pass.

CPU tests use synthetic Git repositories and a local runner in place of Docker. Watched mutations
covered mount validation, directory timestamp restoration, live gate decisions, test/runner patch
filtering, test-status comparison, the artifact size cap, hidden-versus-in-band verdict routing,
and the artifact mismatch guard. Runtime tests also exercise source-path refusal, snapshot pruning,
symlink mountpoint refusal, and command refusal with weakened mount flags. These CPU checks do not
certify live isolation. Before real R2E execution, rerun `make docker-jail-test`, then verify a real
coveragepy task end to end as the non-root host uid: gold passes, base fails, and extracted hidden
tests execute from the graded tree. Verify the test-deletion,
runner-edit, and artifact-mismatch controls in the pinned images. Verify imports resolve inside
the `/testbed` bind mount, relocated Python runs with read-only runtime mounts, and template copies fit the existing snapshot
limits. The family currently requires the writable grader visibility arm; instruction variants
`none`, `vague`, and `specific` remain independent of that requirement.
