#!/usr/bin/env bash
#
# Gate: prove the episode jail contains its occupant before any policy runs against the
# scope-violation environment.
#
# Runs four things and requires all four:
#
#   1. argument guards  -- the mount whitelist must refuse every spelling of a home
#                          directory, since a home tree reaching the jail makes the
#                          containment checks vacuous rather than failing.
#   2. host checks      -- the honeypot bait paths must not collide with anything real,
#                          so a jail failure yields ENOENT instead of a write to a real
#                          file.
#   3. inside the jail  -- every containment check must pass.
#   4. outside the jail -- every DISCRIMINATING containment check must fail. This is the
#                          negative control, and it is the point of the whole script. A
#                          suite that passes because it is broken is worse than none, so
#                          the comparison is per named check, not exit-code inversion:
#                          an ImportError also exits non-zero outside the jail.
#
# Usage:
#   scripts/run_jail_tests.sh [--backend bwrap|unshare] [--keep-artifacts]
#
# --backend defaults to whatever episode_jail.sh would pick on its own (bwrap if installed,
# else unshare), so the gate certifies the path real episodes take. Pin the other one
# explicitly to check both.
#
# Exit 0 only if containment holds AND the suite demonstrably has teeth.
#
set -euo pipefail

backend=""
keep_artifacts=0

die() {
  echo "run-jail-tests: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      backend="$2"
      shift 2
      ;;
    --keep-artifacts)
      keep_artifacts=1
      shift
      ;;
    -h | --help)
      awk 'NR>2 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) die "unknown option: $1" ;;
  esac
done

# Mirror episode_jail.sh's own default rather than hardcoding one. Pinning this to unshare
# meant the gate kept certifying the hand-rolled fallback while every real episode ran under
# bwrap once it was installed -- a gate testing a path production does not take.
if [[ -z "$backend" ]]; then
  if command -v bwrap >/dev/null 2>&1; then
    backend="bwrap"
  else
    backend="unshare"
    echo "run-jail-tests: bwrap absent; certifying the hand-rolled unshare fallback" >&2
  fi
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jail="$script_dir/episode_jail.sh"
assertions="$script_dir/jail_assertions.py"
[[ -x "$jail" ]] || die "not executable: $jail"
[[ -f "$assertions" ]] || die "missing: $assertions"

workdir="$(mktemp -d /tmp/jail-tests.XXXXXX)"
cleanup() { [[ "$keep_artifacts" == 1 ]] || rm -rf -- "$workdir"; }
trap cleanup EXIT

# The episode dir is the only host path bound into the jail, so the suite has to be
# staged there: the repo lives under a home directory and episode_jail.sh refuses to
# mount those, which is exactly the property being tested.
episode_dir="$workdir/episode"
mkdir -p "$episode_dir"
cp "$assertions" "$episode_dir/"

# Asked of the jail rather than hardcoded. It used to be /usr/bin/python3, on the reasoning that
# only /usr is mounted inside -- true, and it made the gate certify a 3.9 interpreter that
# misgraded coding submissions written in a later dialect. Whatever episode_jail.sh resolves is by
# definition the interpreter real episodes get, and it is a host path too, so the same binary runs
# the outside control and the honeypot checks. If resolution fails the gate stops here, which is
# correct: there is nothing to certify.
JAIL_PYTHON="$("$jail" --print-jail-python)" || die \
  "the jail cannot resolve an interpreter, so there is nothing to certify -- see the error above"
readonly JAIL_PYTHON
echo "jail interpreter: $JAIL_PYTHON ($("$JAIL_PYTHON" -c 'import sys; print(sys.version.split()[0])'))"

# The jail clears the environment and sets HOME=/work, so the in-jail run cannot work out which
# host directories to probe: it gets them on its command line. Derived once, here, and handed to
# both the inside run and the outside negative control, because a comparison between two different
# path sets would prove nothing.
host_home_args=()
while IFS= read -r host_home; do
  host_home_args+=(--host-home "$host_home")
done < <("$JAIL_PYTHON" "$assertions" --print-host-homes)
[[ ${#host_home_args[@]} -gt 0 ]] || die "could not derive the host home directories to probe"
echo "probing host homes:${host_home_args[*]//--host-home/}"

# Same argument as the host homes, for the properties that only exist as a difference from the
# launcher: a session and a set of namespaces the jail must have left. Measured here on the host and
# handed to both runs, since a child inherits both and the in-jail run has nothing to compare with.
launcher_session="$("$JAIL_PYTHON" -c 'import os; print(os.getsid(0))')"
launcher_args=(--launcher-session "$launcher_session")
for ns_kind in ipc uts; do
  launcher_namespace="$ns_kind=$(readlink "/proc/self/ns/$ns_kind")"
  launcher_args+=(--launcher-namespace "$launcher_namespace")
  echo "launcher $launcher_namespace"
done
echo "launcher session: $launcher_session"

echo "== 1/4 argument guards: --ro-bind must refuse every spelling of a home directory =="
# The whitelist's one guarantee is that no home tree reaches the jail, and the containment checks
# cannot police it: they probe fixed absolute paths, so a home bound at some other mountpoint reads
# as contained. The guard used to inspect the PARENT of a --ro-bind argument, which accepted every
# directory holding homes, `/`, and any symlink pointing into a home -- bwrap resolves a symlinked
# source and mounts what it points at.
guard_dir="$workdir/guard-probe"
mkdir -p "$guard_dir"
home_symlink="$workdir/symlink-into-home"
ln -sfn "$HOME" "$home_symlink"

# Spellings from two independent directions, and the second direction is the load-bearing one. What
# the jail reports (--print-home-roots) means a root ADDED to its guard is driven here rather than
# going untested; what this script derives from the environment is what would catch a root REMOVED
# from that guard, which a list sourced only from the guard could never see.
candidates=()
add_candidate() {
  local seen
  for seen in ${candidates+"${candidates[@]}"}; do
    [[ "$seen" == "$1" ]] && return 0
  done
  candidates+=("$1")
}

while IFS= read -r home_root; do
  add_candidate "$home_root"
done < <("$jail" --print-home-roots)
home_real="$(realpath -- "$HOME")"
for derived in "${HOME%/}" "$home_real" "$(dirname -- "$home_real")" / "$home_symlink"; do
  add_candidate "$derived"
done
[[ ${#candidates[@]} -gt 2 ]] || die "only ${#candidates[@]} home spellings to drive; this step would prove nothing"

for candidate in "${candidates[@]}"; do
  [[ -e "$candidate" ]] || continue
  guard_rc=0
  guard_output="$("$jail" --episode-dir "$guard_dir" --backend "$backend" \
    --ro-bind "$candidate" -- /bin/true 2>&1)" || guard_rc=$?
  [[ "$guard_rc" == 2 ]] || die \
    "--ro-bind $candidate exited $guard_rc instead of being refused (2), so a home tree can be
     mounted into the jail: ${guard_output:-no output}"
  [[ "$guard_output" == *"home directory"* ]] || die \
    "--ro-bind $candidate was refused, but not by the home-directory guard: $guard_output"
  echo "  refused --ro-bind $candidate"
done

# Positive control: a guard that refuses everything would pass the loop above while making the flag
# useless, and the read-only bind the harness is about to start passing would be unusable.
"$jail" --episode-dir "$guard_dir" --ro-bind /usr/share/doc --backend "$backend" -- /bin/true \
  || die "--ro-bind /usr/share/doc was refused; the home guard is now too broad to mount anything"
echo "  accepted --ro-bind /usr/share/doc"

echo
echo "== 2/4 host checks: honeypot paths must not exist on the host =="
"$JAIL_PYTHON" "$assertions" --group host "${host_home_args[@]}" \
  || die "host checks failed -- a honeypot path collides with a real one; fix that before anything else"

echo
echo "== 3/4 inside the jail (backend=$backend): all containment checks must pass =="
"$jail" --episode-dir "$episode_dir" --backend "$backend" -- \
  "$JAIL_PYTHON" /work/jail_assertions.py --group containment --json \
  --jail-python "$JAIL_PYTHON" "${host_home_args[@]}" "${launcher_args[@]}" \
  >"$workdir/inside.json"

echo
echo "== 4/4 negative control: the same checks must FAIL outside the jail =="
# Expected to exit non-zero; that is the point, so tolerate it here and let
# --verify-negative-control below judge whether the right checks flipped.
"$JAIL_PYTHON" "$assertions" --group containment --json \
  --jail-python "$JAIL_PYTHON" "${host_home_args[@]}" "${launcher_args[@]}" \
  >"$workdir/outside.json" || true

echo
echo "== verdict =="
"$JAIL_PYTHON" "$assertions" --verify-negative-control "$workdir/inside.json" "$workdir/outside.json" \
  || die "negative control is unsound -- do NOT trust a green suite until this passes"

echo "run-jail-tests: PASS (containment holds and the suite has teeth, backend=$backend)"
