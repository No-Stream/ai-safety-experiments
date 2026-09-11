# Keeping this dev box responsive

This box is a single shared cloud desktop, and a wedged one costs days. Everything here
exists so a runaway job degrades itself instead of the machine.

**The box** (one cloud desktop, EC2 `g6.16xlarge`): 64 logical CPUs (32
physical AMD EPYC 7R13 cores, 2 threads each), 242 GB RAM, **no swap**, one NVIDIA L4
with 23 GB VRAM. No swap is why memory pressure has no grace period: allocation goes
from fine to kernel-OOM with nothing in between.

## Run anything expensive through the wrapper

```bash
scripts/resource-limits.sh [options] -- <command>
```

It puts the job in its own cgroup v2 slice via a transient systemd user unit, so the
limits are **enforced by the kernel** rather than requested politely. A library that
ignores `OMP_NUM_THREADS` still cannot exceed the CPU quota, and a memory blowup is
OOM-killed *inside the job's own cgroup* instead of letting the kernel pick a victim
somewhere else — your shell, your editor, another agent's notebook kernel.

Defaults: three-quarters of the host's cores and five-eighths of its RAM (48 of 64 CPUs and
`MemoryMax=152G` on this box; 3 CPUs and ~19G on a `g6e.xlarge`), `TasksMax=4096`, `Nice=10`,
best-effort ionice at lowest priority, thread caps of 16, and the job's working directory set to
whatever you invoked from (`--chdir` overrides). Both are read from `nproc --all` and
`/proc/meminfo` at startup rather than hardcoded: the literals they replaced were measured here, and
on a rented 4-vCPU GPU box `CPUQuota=5000%` and `MemoryMax=160G` bind nothing at all while the
banner still prints them. `--help` lists every flag. Exit status is the
job's own, except `124` for a `--timeout` kill and `137` for exceeding `--mem-max`.

The working-directory default is worth knowing because it was a bug, not a choice:
`systemd-run` puts the job in `$HOME` regardless of where you ran the wrapper, so a
relative invocation broke in a confusing way — `-- uv sync` failed with "No
pyproject.toml found in current directory or any parent". The wrapper now passes
`WorkingDirectory=` explicitly, so the job starts where you did.

`OOMPolicy=kill` is set, which sets `memory.oom.group=1` on the job's cgroup. Exceeding
`MemoryMax` therefore kills **every** process in the job at once, rather than letting the
kernel reap one victim and leave a half-dead job whose parent waits forever on dead
dataloader workers.

### This is not a security boundary

`resource-limits.sh` caps how *much* a job consumes. It does nothing about what the job can
read, write, execute or connect to — the job runs as your uid with your full filesystem,
network and credential access. It was named `sandbox-run.sh` at first, which was a bad name
for exactly this reason and is why it is not called that any more.

**For untrusted code on this box the tool is bubblewrap (`bwrap`), not nsjail.** That is
settled, so nobody needs to re-litigate it: `nsjail` is not packaged for this box's distribution
(`dnf list nsjail` returns no matching packages) and would need a source build with
protobuf, libnl3, bison/flex and the kafel submodule. `bubblewrap` is packaged, and 0.10.0 is
installed at `/usr/bin/bwrap` — the jail gate has been run against it, see
[episode-isolation.md](episode-isolation.md). Unprivileged user namespaces work on this kernel,
but only single-uid mapping:
`newuidmap`/`newgidmap` are present and not setuid, and the login user has no entries in
`/etc/subuid` or `/etc/subgid`. Bubblewrap is fine with that.

**Compose them limits-outside, isolation-inside:**

```bash
scripts/resource-limits.sh -t 15m -- bwrap <isolation args> -- untrusted-command
```

Never the other way around. This script asks the systemd user manager to create the job's
unit over the D-Bus socket at `$XDG_RUNTIME_DIR/bus`, and anything that can reach that
socket can ask systemd to spawn arbitrary processes *outside* any jail — that is code
execution by design, not a bug. So `$XDG_RUNTIME_DIR` must never be bind-mounted into a
jail, and consequently this wrapper cannot be run from inside one. If you ever find
yourself wanting to "simplify" by moving the wrapper inside the jail, that is the mistake
this paragraph exists to prevent.

### Why `MemoryHigh` is off by default

The obvious design is a soft `MemoryHigh` watermark below the hard `MemoryMax`, so a job
that merely overshoots gets throttled and reclaimed instead of killed. **That is actively
harmful on this box, and it was measured, not guessed.**

A test job under `MemoryHigh=512M`, `MemoryMax=1G` allocating 256 MiB numpy blocks stalled
at 631 MB and never finished or died. Its cgroup showed `memory.events` `high=10120`
against `max=0` and `oom_kill=0`, `memory.stat` showed `anon=627MB` with `file=0` and
`pgscan=0`, the process sat in `D` state on `mem_cg`, and PSI reported `full avg10=92.16`.

Read that together: there is **no swap and no page cache to reclaim**, so the kernel
throttles the job on every allocation, reclaims nothing, and the job creeps forward at
roughly 24 MB/min — never reaching `MemoryMax`, so never being killed. `MemoryHigh` turns
a clean fail-fast into an indefinite hang that still holds all its memory. On a box with
swap it behaves as intended; here it does not.

So the default is `MemoryMax` alone: a job that overshoots dies promptly and visibly with
exit 137. `--mem-high` is still available for jobs you know are page-cache-bound, where
reclaim can actually free something. Enabling `systemd-oomd` (see the end of this doc) is
what would make `--mem-high` generally safe, since oomd kills a cgroup that stays under
memory-pressure stall instead of letting it hang.

Long jobs still belong in tmux so they survive a disconnect, which `scripts/tmux_run.sh` does with
the log and the exit-code line already wired up:

```bash
scripts/tmux_run.sh train -- scripts/resource-limits.sh --gpu -t 15m -- python train.py
tail -n 40 /var/tmp/train.log   # last line is EXITCODE=<n> once the command finishes
```

## Thread counts: 16, and higher does not help

The `OMP_NUM_THREADS` family is **per process**, so it multiplies across concurrent
jobs and agents. Measured on this box, throughput plateaus around 16 threads:

| threads | GEMM, 4096² f64 | Polars 20M-row groupby+join |
| ------: | --------------: | --------------------------: |
|       8 |    360 GFLOP/s  |                     0.217 s |
|  **16** |    **640**      |                 **0.160 s** |
|      32 |    654          |                     0.161 s |
|      48 |    679          |                     0.146 s |
|      60 |    673          |                     0.147 s |
|      64 |    671          |                     0.163 s |
|      90 |    674          |                     0.173 s |

16 threads buys 94% of peak GEMM and matches Polars at 32, using a quarter of the box.
Going to 90 buys nothing at all — Polars is measurably *worse* there than at 16, because
oversubscribing 64 logical CPUs with 90 threads costs more in scheduler churn and cache
thrash than the extra parallelism returns. Hyperthreads are also weak for
cache-resident GEMM, which is why 32 (one per physical core) already sits near the
plateau.

So `resource-limits.sh` caps jobs at **16** by default, and `--threads` raises it when you have
measured that a specific workload benefits. This is a per-job cap, not a global environment
setting; the global variables are a separate and messier problem, covered below.

### `nproc` lies, but polars is the one that actually costs you

`nproc` reports **90**, not 64, inside Claude Code sessions, because GNU `nproc` honors
`OMP_NUM_THREADS`. Confirmed: `nproc` → 90, `nproc --all` → 64, `OMP_NUM_THREADS=7 nproc` → 7.
Use `nproc --all` in any script that wants the true count.

That turns out to be mostly cosmetic here, though, because **nothing in this repo sizes a worker
pool off `nproc`** — an audit found no `nproc` in executable code, no bare `make -j`, and no
`-j$(nproc)`. The one `nproc` call that now exists, the limiter's own CPU ceiling, passes `--all`
for exactly this reason: reading the honoured value would have sized the quota off whatever
`OMP_NUM_THREADS` a caller happened to export.

**The real 1.4x oversubscription is polars**, which reads `POLARS_MAX_THREADS` directly and
never consults the CPU count. Measured on this box:

```
polars 1.41.2  thread_pool_size = 90   # on 64 cores, POLARS_MAX_THREADS=90
polars 1.41.2  thread_pool_size = 64   # same box, variable unset
```

Every process that imports polars in a Claude Code session stands up a 90-thread Rayon pool
on 64 cores, and it is invisible at the call site. Most other consumers are fine:
`os.cpu_count()` returns the true 64 and does not honor `OMP_NUM_THREADS`, XGBoost's
`n_jobs=-1` resolves to 64 rather than 90 because it takes `min(omp_get_num_procs(),
omp_get_max_threads())`, joblib and loky report 64, torch reports 32, and numexpr is not
installed in any environment here so `NUMEXPR_NUM_THREADS` is inert.

The `=90` values come from the `env` block of `~/.claude/settings.json` — **not** from any
shell profile. A plain login shell is clean, so this only ever affected Claude Code sessions
and the subagents they spawn. That is the worse case, since agents are what run runaway jobs.

### Fixing it: there is no user-level per-machine settings file

Worth stating plainly because it is the intuitive first guess and it is wrong:
**`~/.claude/settings.local.json` is not a user-level override.** Per the official docs it is
project-scoped, resolved at the git repository root; it only lands in `~/.claude/` when the
session's starting directory *is* your home directory, which makes it silently conditional on
where you launched from. Do not build a fix on it. The documented precedence, highest first,
is managed settings > CLI arguments > project `.claude/settings.local.json` > project
`.claude/settings.json` > user `~/.claude/settings.json`.

Also counterintuitive: **a shell export loses to a settings `env` entry.** The docs state
Claude Code writes each `env` entry into the process environment at startup, "replacing the
value inherited from the shell." So exporting `OMP_NUM_THREADS` from `~/.zshenv` is not a fix
while the key is still present in `settings.json`. (Empirically a `.zshenv` export does win
*inside* a zsh-backed Bash tool call, because the profile runs after injection — but not for
processes Claude Code spawns directly, so it is a half-fix and two sources of truth.)

The recommended fix is to **compute the value per machine at launch** rather than store a
constant anywhere. A shell wrapper that launches `claude` with `--settings` sits at precedence
level 2, above user settings, and a multi-key `env` block injected through that flag is already
proven on this setup. Deriving the number from `nproc --all` is correct on every box with no
per-host constant to go stale. Its one gap: it only covers wrapper invocations, not a bare
`command claude`, cron, or headless `claude -p`. Closing that gap completely would need a
managed settings file at `/etc/claude-code/managed-settings.json`, which requires root and is
the org-policy surface, so it is a heavier hammer than this problem deserves.

One caveat against simply deleting the keys: other code on the box may read
`POLARS_MAX_THREADS` with a large default of its own, in which case unsetting the variable hands
that code a *bigger* pool than the injected 90 did. Check the readers before deleting; it is not
unconditionally safe.

If you do delete them, delete rather than blank: an empty value is **not** the same as an
absent one. Setting `OMP_NUM_THREADS=` produces `OMP: Warning #234: Invalid symbols found.
Check the value ""` and `POLARS_MAX_THREADS=` produces `illegal value '' found while parsing
option`. Both then fall back to the core count anyway, so blanking buys warning spam and
nothing else.

Because that file is Claude Code's own configuration, an agent should not edit it. The
deletion, if you choose it over the computed-at-launch route above, is: (note that the file is
synced between boxes, and on a larger one 90 is a *reasonable* half-machine value rather than a
bug)

```bash
cp ~/.claude/settings.json ~/.claude/settings.json.bak
jq 'del(.env.OMP_NUM_THREADS, .env.OPENBLAS_NUM_THREADS, .env.MKL_NUM_THREADS,
        .env.NUMEXPR_NUM_THREADS, .env.POLARS_MAX_THREADS, .env.RAYON_NUM_THREADS)' \
  ~/.claude/settings.json > /tmp/settings.new.json \
  && mv /tmp/settings.new.json ~/.claude/settings.json
```

Verified against a copy of the real file: it removes exactly those six keys (env goes from
30 entries to 24, `has("OMP_NUM_THREADS")` becomes false rather than empty) and leaves every
other env entry and top-level key byte-identical.

Run it with no other Claude Code session live, then restart and confirm `nproc` reports 64.
A live session holds this config in memory, and a launcher wrapper that rebuilds the
model-related keys at session start means a concurrent session can undo the edit. Any
settings-restore tooling would revert it along with months of unrelated legitimate drift, so
don't reach for that as the mechanism here.

Nothing in this repo breaks on their absence: nothing here *reads* the six variables to size
anything (the limiter only writes them for the job it launches), so the risk lives entirely in
other code on the box.

One trap from that audit is worth carrying, because it is the sharpest edge and it is easy to
repeat: capping these variables with `os.environ.setdefault` is a **no-op against a pre-set
value**, so an injected 90 defeats the intended cap, and under xdist it multiplies — 8 workers
each holding a 90-thread polars pool is roughly 720 threads on 64 cores. Bare assignment would
break the benchmark overrides such a `setdefault` usually exists for, so the fix is to honour a
pre-existing value only when it is *lower* than the cap.

One tradeoff worth knowing: with the variables gone, an unwrapped numeric job defaults to
every core, so there is no reserved interactive headroom unless it goes through
`resource-limits.sh`. That is still strictly better than the status quo here, where 90 threads
on 64 CPUs already left no headroom *and* thrashed. On CPU-saturation alone the box stays
responsive because the scheduler is fair; what actually wedges it is memory, which is what
`MemoryMax` covers.

### `TOKENIZERS_PARALLELISM` is deliberately left unset

Setting it to `false` globally would be a bad trade. It disables the tokenizer's Rust-level
parallelism, which costs real throughput on bulk tokenization — something this repo does,
tokenizing Wikipedia with the GPT-2 tokenizer. What it buys is silence: HuggingFace already
disables parallelism by itself after a fork, and the message it prints is telling you that
happened. In DataLoader work that notice is signal about fork behavior, not noise. Leaving
it unset keeps the diagnostic and the throughput; a specific job that wants it can set it.

`resource-limits.sh` sets the six thread variables for the job it launches regardless, so a job
run through the wrapper already gets a sane cap whether or not the global keys are gone.

## GPU

Real training goes to AWS Batch. Nothing on this box should hold the L4 for more than
about 5-10 minutes, which is why `--gpu` defaults `--timeout` to `15m`.

`scripts/gpu_preflight.py` refuses to start when another process already holds VRAM, and
names the offender. This is the check that stops two agents from colliding on the one
GPU, and it catches the common failure where a killed run leaves an orphaned CUDA
process pinning memory. `resource-limits.sh --gpu` runs it first; from a notebook, call it
before allocating anything:

```python
from scripts.gpu_preflight import require_free_gpu

require_free_gpu()
```

It refuses on two separate grounds, because per-process attribution is not guaranteed. A named
peer holding at least `--threshold-mib` (512 by default, so an idle CUDA context is tolerated) is
the obvious one. The second is VRAM that `memory.used` reports but no `compute-apps` row accounts
for: nvidia-smi lists no row at all for a peer in another container, and reports `[N/A]` usage for
one it can see but not inspect. Both used to read as an empty card — the aggregate was fetched
only to interpolate into the error message, never compared — so a nearly full GPU passed while
logging `preflight OK, used=22000 of total=23034 MiB in use`. If you get the "held by no process
it will name" refusal on a card `nvidia-smi` shows as idle, that is the aggregate disagreeing with
the process list, and it is worth believing over the process list. `tests/test_gpu_preflight.py`
pins both grounds and the three cases that must stay allowed (idle card, small attributed context,
our own allocation).

`--gpu` also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which lets the
caching allocator grow segments in place instead of fragmenting into unusable blocks.
On 23 GB with variable sequence lengths, fragmentation is a more likely cause of a
spurious out-of-memory error than genuinely running out.

**`torch.cuda.set_per_process_memory_fraction` is deliberately not a default.** It would
make a job fail at its fraction even when the GPU is otherwise idle, which is the normal
case here and would cost more than it saves. Set it explicitly (say `0.45`) only when you
knowingly share the GPU with another run; preflight plus the timeout covers the rest.

The `--timeout` kill terminates the whole cgroup, not just the parent, which is the point:
a `SIGKILL` to a training script that leaves its dataloader children alive is exactly how
VRAM stays pinned after the job is "dead".

## What is actually enforced

None of this needs root: the `cpu`, `memory` and `pids` controllers are already delegated
to this user's slice, so `systemd-run --user` can set real kernel limits. Every row below
was tested on this box rather than assumed.

| Layer | Status on this box |
| --- | --- |
| CPU quota via cgroup v2 | **Enforced by the kernel.** 41 spinning processes under `--cpus 4` consumed a measured 4.04 CPUs. |
| `MemoryMax` + `memory.oom.group` | **Enforced.** A job allocating past `--mem-max 1G` was killed whole in 1 second with exit 137, including a sibling process that had allocated nothing. |
| `--timeout` | **Enforced, whole cgroup.** Exit 124, and two spinner children that would have been orphans were killed with it. |
| `TasksMax` | **Enforced** (`pids.max` verified set). |
| Nice / ionice | Enforced, but only a scheduling preference — a niced job still runs. |
| `MemoryHigh` | **Enforced but unsafe here, so off by default** — it throttles without killing and hangs the job. See above. |
| `*_NUM_THREADS` env caps | **Advisory.** Libraries are free to ignore them; the cgroup quota is the real backstop. |
| `--advisory` mode (`ulimit -v` + `taskset`) | **Not enforced.** `ulimit -v` caps address space, which is a poor proxy for resident memory and is ignored by allocators that map their own arenas. Opt-in escape hatch for a host with no systemd user instance; it prints a warning. |
| System-wide OOM backstop | **Absent.** See below. |

### One dead run used to block every later one

Worth knowing because the symptom points at the wrong thing. The wrapper's readiness check
used to be a bare `systemctl --user is-system-running`, which exits non-zero on **`degraded`**
— the state the user manager enters when any unit anywhere has failed, whether or not it has
anything to do with us. A transient `reslimit-*` unit that failed and was never cleared was
enough, and every subsequent invocation then died with "no systemd user instance", on a box
whose user manager was running perfectly well and enforcing limits for anything that asked.

That happened on 2026-08-15. A throughput sweep's unit exited 120 — Python's broken-pipe code,
because the log pipe the job was writing into had closed — which killed the wrapper before it
reached its own end-of-run `reset-failed`, leaving the unit failed and the manager degraded.
The next GPU job refused to start and looked, from the message, like a host problem.

Two changes, both of which were tested by planting a failed unit and watching the wrapper
proceed. The check now accepts `running`, `degraded`, `starting`, `maintenance` and `stopping`,
and only refuses when the manager is genuinely unreachable — verified still to refuse under a
bogus `XDG_RUNTIME_DIR`, so the gate has not simply been defanged. And the wrapper clears its
own spent `reslimit-*` units on the way in, so the state cannot accumulate across runs even
when a job dies in a way that skips the cleanup at the end.

## The OOM backstop already exists

An earlier version of this doc claimed there was no out-of-memory backstop on this box.
That was wrong, and the correction matters because it changes what needs doing.

`systemd-oomd` is indeed inactive and disabled and `earlyoom` is indeed not installed,
but there is a **user-space guard that is enabled and running**:
`~/.config/systemd/user/oom-guard.service`, active since 2026-08-13, executing
`~/.local/bin/oom-guard.sh`. It describes itself as killing "runaway anon-RSS hogs before
kernel wedge", and it is better targeted at this box than `systemd-oomd` would be:

- Triggers when `MemAvailable` drops below 8% of total.
- Ranks victims by `RssAnon` from `/proc/<pid>/status` rather than total RSS, so a worker
  whose footprint is mostly mmap'd file cache is not killed by mistake.
- Kills above a 30 GB `RssAnon` floor, with a 10 GB diffuse fallback for the case where
  the trigger fires but no single process is over the main floor.
- Protects a set of processes from being chosen, and runs with `OOMScoreAdjust=-800` so
  the guard itself is not the victim.

Anonymous RSS with no swap is exactly the failure mode measured in the `MemoryHigh`
section above, so this guard covers the real risk. `sudo systemctl enable --now
systemd-oomd` is therefore optional rather than necessary — worth considering only if you
want cgroup-level pressure kills in addition, and its swap-based heuristics are inert here
anyway.

Note there is also an enabled `wedge-monitor.service`. Both units, and the script they
run, are watched by the tripwire in [episode-isolation.md](episode-isolation.md), since a
continuously-running user unit is a code-execution surface.
