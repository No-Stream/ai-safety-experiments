# Episode isolation, and what it does not protect

The research this repo is heading toward trains a policy in an environment that
deliberately rewards scope violation: reading a grader's answer key, tampering with a
test file, writing outside the working directory, reaching for a network tool. That only
makes sense if a violation lands on bait instead of on something real.

**Gate: no policy runs against that environment on this host until
`scripts/run_jail_tests.sh` passes.** It is a hard prerequisite, independent of how the
rest of the research design settles.

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
| `scripts/canary_manifest.py` | Checksum tripwire over the host's code-execution surfaces. |
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
- **The staged interpreter is not on the canary manifest.** That tripwire watches `$HOME` code-
  execution surfaces, and this tree lives in `/var/tmp` by design. `episode_jail.sh` re-checks its
  ownership and mode at every launch, which is a weaker guarantee than a checksum: it would catch a
  permission change, not a same-mode edit to a library file by our own uid. Another reason the
  packaged interpreter is the better configuration.

## Tripwire and pre-upload scanner

`canary_manifest.py` checksums every path where a write buys code execution the next time
something routine happens: each file in `~/.claude/hooks/` (the permission-decision hooks
included), the systemd user units, the scripts the crontab runs, the user-space OOM guard, the
shell and tool rc files, and `~/.claude/settings.json`. Counts are deliberately not quoted here;
the hook fleet grows, and a number in prose goes stale without anything noticing.

That last one matters and a hooks-directory-only tripwire would miss it: `settings.json`
registers hooks **by path**, so repointing an entry there is code execution without
touching any file in the hooks directory.

```bash
python3 scripts/canary_manifest.py --update   # baseline, after an intentional change (make canary-update; owner-run only)
python3 scripts/canary_manifest.py            # report drift, exit 1 if any
```

It only ever reads the watched paths, and prints paths and hashes, never contents.

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
