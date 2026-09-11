#!/bin/bash
# Launch a long command in a detached tmux session that tees to a log and records the command's exit
# status in that log, then return immediately with the command to poll.
#
#   scripts/tmux_run.sh <name> [--log <path>] -- <cmd> [args...]
#
#   scripts/tmux_run.sh ci-gate -- make ci
#   scripts/tmux_run.sh trv2-readout --log ~/logs/trv2-readout.log -- \
#       env AWS_PROFILE=<profile> python -m games.readout --run-dir ...
#
# WHY. Two callers need exactly this. An agent driving this box is killed by its own harness after 180
# seconds with no output (the Workflow tool's stall watchdog: measured casualties include a workflow
# that burned 17 extra agent starts for 9 slots, and a reviewer that died 6 of 6 retries), so anything
# slower than that has to run detached with its output on disk rather than in the foreground. And a
# background shell dies on SIGHUP -- context compaction, a reconnect, the parent shell exiting -- while
# the work it started keeps running orphaned, so "just background it" loses the log of a job that is
# still burning the box. tmux plus tee survives both. Every kit README and readout recipe in this repo
# re-derived that one-liner by hand, which is how one of them ended up teeing into /tmp and another
# reporting tee's exit status as the job's.
#
# THE STATUS LIVES IN THE LOG, NOT IN THIS SCRIPT'S EXIT CODE. This script exits 0 once the session
# exists: the wrapped command has barely started, so it has no status yet. The last line of the log is
# `EXITCODE=<n>` written after the command finishes, on a line of its own even when the command's own
# output ended without a newline, and that is the only place the command's status is reported. A caller
# that treats this script's 0 as "the job passed" has misread it. Exit 2 is a usage refusal (nothing was
# launched); exit 3 means tmux itself would not create the session.
#
# `set -o pipefail` and a bare `$?` are the canonical form on purpose. Without pipefail the log would
# record tee's status, which is 0 whatever the command did. `${PIPESTATUS[0]}` would fix that too but
# is bash-only (zsh spells it `${pipestatus[1]}`), and tmux runs the command under the user's login
# shell, which is zsh on this box; pipefail makes a bare `$?` correct in both.
#
# PATH IS PASSED IN EXPLICITLY because a tmux session does not inherit the launching shell's
# environment: the session's environment comes from the tmux SERVER, which may have been started days
# earlier from an unrelated shell. Measured on this box: a variable set inline on the `tmux
# new-session` command line arrives empty inside the session. PATH is the one variable that decides
# whether the command is found at all (`uv`, `make`, `aws` all live outside /usr/bin here), so it is
# forwarded. Anything else the command needs goes in the command itself as `env VAR=value <cmd>`,
# which is a literal argument and therefore does reach the session. Note that an unquoted `VAR=value`
# first word would also survive, but only while the value needs no shell quoting -- `env` always
# works, so the examples above use it.
#
# THE LOG NEVER LANDS IN /tmp. /tmp here is a RAM-backed tmpfs with an inode cap separate from its
# byte space, shared by every concurrent session on the box; when it ran out of file slots in August
# no shell on the box could start until the leftovers were reaped by hand. Logs belong under /var/tmp
# (the default) or ~/logs, so a /tmp path is refused rather than quietly accepted.
set -euo pipefail

readonly USAGE_EXIT=2
readonly TMUX_EXIT=3

usage() {
  cat >&2 <<'USAGE'
usage: scripts/tmux_run.sh <name> [--log <path>] -- <cmd> [args...]

  <name>        tmux session name; letters, digits, dash and underscore only. Refused if a session
                of that name already exists, so two callers cannot write one log.
  --log <path>  where to tee the command's output. Default /var/tmp/<name>.log. Never under /tmp.
  --            end of this script's options; everything after it is the command.

Exits 0 once the session exists (the command's own status is the log's last line, EXITCODE=<n>),
2 on a usage refusal, 3 if tmux would not create the session.
USAGE
}

refuse() {
  echo "tmux_run: $1" >&2
  echo "tmux_run: nothing was launched." >&2
  usage
  exit "$USAGE_EXIT"
}

if [ "$#" -eq 0 ]; then
  refuse "no session name given"
fi
case "$1" in
  -h | --help)
    usage
    exit 0
    ;;
esac

name="$1"
shift
log=""
log_given=no

while [ "$#" -gt 0 ]; do
  case "$1" in
    --log)
      shift
      [ "$#" -gt 0 ] || refuse "--log needs a path"
      log="$1"
      log_given=yes
      shift
      ;;
    --log=*)
      log="${1#--log=}"
      log_given=yes
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      refuse "unexpected argument before --: $1 (the command must follow a literal --)"
      ;;
  esac
done

[ "$#" -gt 0 ] || refuse "no command after --"

# An empty --log would otherwise fall through to the default path, which is a silent surprise rather
# than the refusal a typo deserves.
if [ "$log_given" = yes ] && [ -z "$log" ]; then
  refuse "--log was given an empty path"
fi

# tmux target strings are parsed on ':' and '.', so a name carrying either would make every later
# `tmux ... -t <name>` ambiguous, including the poll and attach commands printed below.
if ! printf '%s' "$name" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9_-]*$'; then
  refuse "session name '$name' must be letters, digits, dash or underscore, starting with a letter or digit"
fi

if [ -z "$log" ]; then
  log="/var/tmp/$name.log"
fi
case "$log" in
  /*) ;;
  *) log="$PWD/$log" ;;
esac
# The /tmp refusal below is a prefix match, so the path has to be tidied before it: `/var/../tmp/x.log`
# and a `~/logs` that is a symlink into /tmp both land in the tmpfs while starting with something else.
# `realpath -m` normalizes '..' and resolves symlinks without requiring the path to exist yet.
log="$(realpath -m "$log")" || refuse "cannot resolve the log path $log"
case "$log" in
  /tmp | /tmp/*)
    refuse "log path $log is under /tmp, which is a RAM-backed tmpfs with an inode cap shared by every session on this box; use /var/tmp or ~/logs"
    ;;
esac

if [ -d "$log" ]; then
  refuse "log path $log is a directory; --log takes the file to tee into"
fi

log_dir="$(dirname "$log")"
mkdir -p "$log_dir" || refuse "cannot create the log directory $log_dir"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux_run: tmux is not on PATH; install it or run the command under nohup by hand." >&2
  exit "$TMUX_EXIT"
fi

# Exact match ('=' prefix): without it has-session matches by prefix, so an existing `ci-gate-2` would
# make the name `ci-gate` look taken.
if tmux has-session -t "=$name" 2>/dev/null; then
  refuse "a tmux session named $name already exists; pick another name or reap that one (tmux kill-session -t '=$name')"
fi

# One generation of the previous log is kept rather than truncated away by tee: relaunching under a
# name that has run before is the normal way a crashed job is retried, and the crashed run's log is
# often the only record of why it died.
rotated=""
if [ -s "$log" ]; then
  mv -f "$log" "$log.prev"
  rotated="$log.prev"
fi

quoted_command=""
for arg in "$@"; do
  quoted_command="$quoted_command $(printf '%q' "$arg")"
done
quoted_log="$(printf '%q' "$log")"

# The canonical line from AGENTS.md, composed once here instead of per kit README. `tee` is byte
# transparent, so a command whose output does not end in a newline would otherwise get the trailer glued
# to its last line (`fooEXITCODE=0`, measured), and a reader comparing `tail -n 1` or anchoring on
# ^EXITCODE= would then call a finished job unfinished; hence the newline appended first when the log
# does not end in one. The status goes into `rc` because that check would clobber `$?` -- `rc` and not
# `status`, which is a read-only parameter in zsh, the login shell tmux runs this under.
session_command="set -o pipefail;${quoted_command} 2>&1 | tee ${quoted_log}; rc=\$?; [ -s ${quoted_log} ] && [ -n \"\$(tail -c1 ${quoted_log})\" ] && echo >> ${quoted_log}; echo EXITCODE=\$rc >> ${quoted_log}"

# PATH rides inside the command string rather than through `new-session -e`, which tmux only grew in
# 3.2 (Ubuntu 20.04 ships 3.0a); a literal argument reaches the session on every version.
session_command="export PATH=$(printf '%q' "$PATH"); $session_command"

if ! tmux new-session -d -s "$name" -c "$PWD" "$session_command"; then
  echo "tmux_run: tmux refused to create session $name; nothing is running." >&2
  exit "$TMUX_EXIT"
fi

echo "tmux_run: session $name started in $PWD"
if [ -n "$rotated" ]; then
  echo "previous log rotated to: $rotated"
fi
echo "log: $log"
echo "poll: tail -n 40 $log"
echo "follow: tail -f $log"
# `tail -n 1` and not `grep -m1 EXITCODE`: grep stops at the first match anywhere in the log, and a job
# that prints this pattern itself (a failing assertion in tests/test_tmux_run.py prints the composed
# command above, and any job tailing another of these logs does too) would then be reported finished
# while it is still running. The trailer is appended last by construction, so the last line is the
# reliable place to read it.
echo "status: tail -n 1 $log   # an EXITCODE=<n> line means the command has finished"
# The exact-match target is QUOTED in what we print because the login shell here is zsh, where an
# unquoted =word is EQUALS expansion: `tmux attach -t =ci-gate` dies with "ci-gate not found" before
# tmux is even reached. Measured on this box while writing this script.
echo "attach: tmux attach -t '=$name'"
echo "reap: tmux kill-session -t '=$name'"
