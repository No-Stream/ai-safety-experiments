#!/usr/bin/env bash
#
# Run a command under kernel-enforced CPU, memory, task and wall-clock limits so a
# runaway job cannot make this dev box unresponsive.
#
# Limits are applied by putting the job in its own cgroup v2 slice via a transient
# systemd user unit. They are enforced by the kernel, not advisory: a library that
# ignores OMP_NUM_THREADS still cannot exceed the CPU quota, and a memory blowup is
# OOM-killed inside the job's own cgroup instead of letting the kernel pick a victim
# elsewhere on the box (your editor, your shell, another agent's notebook kernel).
#
# THIS IS NOT AN ISOLATION OR SECURITY BOUNDARY. It constrains how much CPU, memory,
# task count and wall-clock a job consumes. It does nothing about what that job can
# read, write, execute or connect to: the job runs as your uid with your full
# filesystem, network and credential access. Do not run untrusted code under this
# expecting containment. For untrusted code on this box the chosen tool is bubblewrap
# (`bwrap`); see docs/resource-limits.md.
#
# Composition order matters, and it is limits OUTSIDE, isolation INSIDE:
#   resource-limits.sh -t 15m -- bwrap <isolation args> -- untrusted-command
# Never the reverse. This script asks the systemd user manager to create the unit over
# the D-Bus socket at $XDG_RUNTIME_DIR/bus, and anything able to reach that socket can
# ask systemd to spawn arbitrary processes outside any jail. So $XDG_RUNTIME_DIR must
# never be bound into a jail, which also means this script cannot run from inside one.
#
# Usage:
#   scripts/resource-limits.sh [options] -- <command> [args...]
#
# Options:
#   -c, --cpus N        CPU cores the job may use   (default: 3/4 of the host's cores)
#       --chdir DIR     working directory for the job (default: the caller's cwd).
#                       systemd-run's own default is $HOME, which silently breaks any
#                       relative invocation -- `-- uv sync` failed with "No pyproject.toml
#                       found" until this defaulted to the caller's cwd instead.
#       --mem-max SZ    hard cap; kill the whole job above it
#                       (default: 5/8 of the host's MemTotal)
#       --mem-high SZ   soft cap; throttle+reclaim above it. OFF by default and
#                       measured to be dangerous here: this box has no swap, so an
#                       anon-heavy job hits memory.high with nothing reclaimable and
#                       stalls in D-state indefinitely without ever reaching
#                       --mem-max. Only pass it for page-cache-bound jobs. See
#                       docs/resource-limits.md.
#   -t, --timeout DUR   kill the whole process group after DUR, e.g. 15m
#       --tasks-max N   max threads+processes                (default 4096)
#       --nice N        scheduling niceness, higher = yields (default 10)
#       --threads N     value for the OMP/BLAS/Polars thread family (default 16)
#       --gpu           refuse to start if another process holds GPU memory,
#                       and default --timeout to 15m
#   -n, --name NAME     systemd unit name, for `systemctl --user status`
#       --advisory      run WITHOUT cgroups (nice + ulimit only). Advisory, not
#                       enforced: a job can still exceed these. Opt-in escape hatch
#                       for hosts with no systemd user instance.
#
# Exit status is the command's own, with two reserved values:
#   124  the job hit --timeout and its process group was killed
#   137  the job exceeded --mem-max and its whole cgroup was killed
#
# Long jobs belong in tmux so they survive a disconnect. Compose the two:
#   tmux new-session -d -s train \
#     "scripts/resource-limits.sh --gpu -t 15m -- python train.py 2>&1 | tee /tmp/train.log; echo EXITCODE=\$?"
#
set -euo pipefail

# Derived from the host, never hardcoded: the same script now runs on this 64-core dev box and on a
# 4-vCPU rented GPU instance, where a quota measured here would not bind at all and the containment
# this script exists for would be absent while the banner still printed reassuring numbers.
# `nproc --all` rather than `nproc`, which honours OMP_NUM_THREADS and reports 90 here.
TOTAL_CPUS="$(nproc --all)"
TOTAL_MEM_KIB="$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)"
readonly TOTAL_CPUS TOTAL_MEM_KIB
readonly TIMEOUT_EXIT=124
readonly OOM_EXIT=137

# Three-quarters of the cores and five-eighths of the RAM, leaving the rest for the interactive
# session, the other agents on the box, and the page cache the job needs to make progress.
cpus=$((TOTAL_CPUS * 3 / 4))
chdir="$PWD"
mem_high=""
# Spelled in whole MiB rather than through `numfmt --to=iec`, which keeps three significant digits and
# on a 31 GiB host rounded 19.56G up to 20G, a 2.25% overshoot of the intended cap.
mem_max="$((TOTAL_MEM_KIB * 5 / 8 / 1024))M"
timeout=""
tasks_max=4096
nice=10
threads=16
gpu=0
advisory=0
unit_name=""

die() {
  echo "resource-limits: $*" >&2
  exit 2
}

# Prints the header comment block, so the usage text cannot drift from the docs above.
usage() { awk 'NR>2 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c | --cpus)
      cpus="$2"
      shift 2
      ;;
    --mem-high)
      mem_high="$2"
      shift 2
      ;;
    --mem-max)
      mem_max="$2"
      shift 2
      ;;
    -t | --timeout)
      timeout="$2"
      shift 2
      ;;
    --tasks-max)
      tasks_max="$2"
      shift 2
      ;;
    --chdir)
      chdir="$2"
      shift 2
      ;;
    --nice)
      nice="$2"
      shift 2
      ;;
    --threads)
      threads="$2"
      shift 2
      ;;
    --gpu)
      gpu=1
      shift
      ;;
    --advisory)
      advisory=1
      shift
      ;;
    -n | --name)
      unit_name="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*) die "unknown option: $1 (see --help)" ;;
    *) break ;;
  esac
done

[[ $# -gt 0 ]] || die "no command given (see --help)"
[[ -d "$chdir" ]] || die "--chdir is not a directory: $chdir"
[[ "$cpus" =~ ^[0-9]+$ ]] || die "--cpus must be an integer, got: $cpus"
((cpus >= 1 && cpus <= TOTAL_CPUS)) || die "--cpus must be 1..$TOTAL_CPUS, got: $cpus"
((cpus <= TOTAL_CPUS - 4)) || echo "resource-limits: warning: --cpus $cpus leaves fewer than 4 cores for interactive use" >&2

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ((gpu)); then
  # Default GPU jobs to a short leash: real training runs on Batch, not here.
  [[ -n "$timeout" ]] || timeout=15m
  python_bin="${GPU_PREFLIGHT_PYTHON:-python3}"
  command -v "$python_bin" >/dev/null || die "--gpu needs a python3 on PATH (or set GPU_PREFLIGHT_PYTHON)"
  "$python_bin" "$script_dir/gpu_preflight.py"
fi

# Thread caps for the job itself. These are per-process and multiply across
# concurrent jobs, and measurement on this box shows throughput plateaus near 16
# threads (see docs/resource-limits.md), so a low value costs almost nothing.
job_env=(
  "OMP_NUM_THREADS=$threads"
  "OPENBLAS_NUM_THREADS=$threads"
  "MKL_NUM_THREADS=$threads"
  "NUMEXPR_NUM_THREADS=$threads"
  "POLARS_MAX_THREADS=$threads"
  "RAYON_NUM_THREADS=$threads"
)
((gpu)) && job_env+=("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

if ((advisory)); then
  echo "resource-limits: WARNING advisory mode - limits are NOT enforced; the job can exceed them" >&2
  # ulimit -v caps address space, which is a poor proxy for RSS and is ignored by
  # anything using its own allocator, hence "advisory".
  ulimit -v $(($(numfmt --from=iec "$mem_max") / 1024)) || true
  cd -- "$chdir" || die "could not enter $chdir"
  exec env "${job_env[@]}" nice -n "$nice" ionice -c2 -n7 \
    taskset -c "0-$((cpus - 1))" "$@"
fi

# Clear our own spent units before asking the manager how it is doing. The cleanup at the end
# of a run is skipped whenever this script dies early -- a SIGPIPE from a closed log pipe was
# the case that bit -- and one leftover failed unit puts the user manager into "degraded",
# which used to make every later run refuse to start.
systemctl --user reset-failed 'reslimit-*' >/dev/null 2>&1 || true

# "degraded" means some unit somewhere failed, which says nothing about whether cgroups work;
# only a manager that is absent or dead cannot enforce limits. Gating on "running" alone made
# an unrelated failed unit look identical to a host with no user instance at all.
systemd_state="$(systemctl --user is-system-running 2>/dev/null || true)"
case "$systemd_state" in
  running | degraded | starting | maintenance | stopping) ;;
  *) die "no usable systemd user instance, state=${systemd_state:-unreachable}
  (XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-unset}); limits cannot be enforced here. Re-run with
  --advisory to proceed with unenforced limits instead." ;;
esac

[[ -n "$unit_name" ]] || unit_name="reslimit-$$-$(basename -- "$1")"
unit_name="${unit_name//[^A-Za-z0-9_.-]/_}"

properties=(
  -p "CPUQuota=$((cpus * 100))%"
  -p "MemoryMax=$mem_max"
  -p "TasksMax=$tasks_max"
  -p "Nice=$nice"
  -p "IOSchedulingClass=best-effort"
  -p "IOSchedulingPriority=7"
  -p "WorkingDirectory=$chdir"
  # Sets memory.oom.group=1, so exceeding MemoryMax kills every process in the job
  # together instead of leaving a half-dead job with its workers reaped.
  -p "OOMPolicy=kill"
)
[[ -n "$mem_high" ]] && properties+=(-p "MemoryHigh=$mem_high")
[[ -n "$timeout" ]] && properties+=(-p "RuntimeMaxSec=$timeout")
for kv in "${job_env[@]}"; do properties+=(-p "Environment=$kv"); done

echo "resource-limits: unit=$unit_name cpus=$cpus/$TOTAL_CPUS mem_max=$mem_max mem_high=${mem_high:-off} threads=$threads timeout=${timeout:-none}" >&2

# --pipe streams the job's stdio and propagates its exit status; --wait blocks until
# it finishes. The unit is deliberately NOT --collect'ed so its Result is still
# queryable below, which is how a timeout is told apart from a plain failure.
rc=0
systemd-run --user --pipe --wait --quiet --unit="$unit_name" "${properties[@]}" -- "$@" || rc=$?

result="$(systemctl --user show -p Result --value "$unit_name" 2>/dev/null || true)"
systemctl --user reset-failed "$unit_name" >/dev/null 2>&1 || true

case "$result" in
  timeout)
    echo "resource-limits: TIMEOUT after $timeout - killed the whole process group" >&2
    exit "$TIMEOUT_EXIT"
    ;;
  oom-kill)
    echo "resource-limits: OOM-KILLED at MemoryMax=$mem_max - the job's cgroup was killed, the box was not" >&2
    exit "$OOM_EXIT"
    ;;
esac

exit "$rc"
