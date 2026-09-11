#!/usr/bin/env bash
#
# Stage a relocatable CPython where the episode jail can mount it read-only.
#
# WHY THIS EXISTS. The jail's mount whitelist carries only /usr, so the interpreter reachable
# inside it is whatever /usr/bin/python3 happens to be -- 3.9.25 on this host. That MISGRADED
# coding submissions on 2026-08-17: code using a 3.10+ builtin (int.bit_count()) crashed inside
# the jail, failed every unit, and "fails everything" trivially contains any registered failing
# set, so the grader scored it as carrying a planted flaw. episode_jail.sh therefore requires an
# interpreter at or above its floor and refuses to run without one.
#
# PREFER A PACKAGED INTERPRETER UNDER /usr. It needs no staging and no extra mount at all,
# because /usr is already bound read-only, and it is root-owned and patched by the distribution:
#
#     sudo dnf install python3.13     # or python3.12
#
# This script is the fallback for a host where that is unavailable. It copies a relocatable
# CPython to a fixed path OUTSIDE any home directory. Outside is a hard requirement, not filing
# preference, for two measured reasons:
#
#   * /proc/1/cmdline is READABLE inside the jail and carries the whole bwrap argv, and
#     /proc/self/mountinfo carries every bind's source path. Mounting from $HOME would put the
#     owner's username inside the jail, where a policy can read it and a rollout trace can
#     record it -- and this repo's traces are published.
#   * sys.prefix follows the interpreter, so every traceback out of jailed code names the staged
#     tree. That is the same leak on a far more travelled route.
#
# The staged tree is executed by every jailed grader, so whoever can rewrite it chooses the
# interpreter that grades. The destination's parent is world-writable, so ownership plus the
# parent's sticky bit is what stops another uid replacing it; episode_jail.sh re-checks both at
# every launch rather than trusting this script to have got it right once.
#
# Usage:
#   scripts/stage_jail_python.sh [--source DIR] [--force]
#
# Options:
#   --source DIR   root of a relocatable CPython (contains bin/python3 and lib/). Defaults to the
#                  newest uv-managed CPython that meets the jail's floor.
#   --force        re-stage even when the destination already validates
#   -h, --help
#
set -euo pipefail

die() {
  echo "stage-jail-python: $*" >&2
  exit 2
}

usage() { awk 'NR>2 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jail="$script_dir/episode_jail.sh"
[[ -x "$jail" ]] || die "not executable: $jail"

source_root=""
force=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      source_root="$2"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

# Both the destination and the floor come from the jail script, which is the thing that enforces
# them. A second copy of either constant here would drift, and the drift would surface as a
# misgraded run rather than as an error.
dest="$("$jail" --print-staged-root)"
floor_minor="$("$jail" --print-python-floor)"

python_minor() {
  "$1" -c 'import sys; print(sys.version_info[1] if sys.version_info[0] == 3 else -1)' 2>/dev/null
}

# The repo's own venv minor version, or nothing when there is no venv to ask.
venv_minor() {
  local venv_python="$script_dir/../.venv/bin/python3"
  [[ -x "$venv_python" ]] || return 0
  python_minor "$venv_python"
}

# Prefers the venv's own minor version over the newest available, which is not mere tidiness: the
# jail grades code the repo's tooling was written against, and the EvalPlus expected outputs are
# computed host-side, so jail and venv agreeing removes a whole class of "passes here, fails there".
# Running AHEAD of the venv is its own hazard -- each release removes deprecated stdlib, so a
# too-new interpreter reintroduces the failure this floor exists to prevent, from the other side.
find_uv_source() {
  local candidate minor preferred="" newest=""
  local uv_root="${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}"
  local want
  want="$(venv_minor)"
  [[ -d "$uv_root" ]] || die \
    "no --source given and no uv interpreters at $uv_root. Install one with
       uv python install 3.${want:-13}
     or ask the box owner for the packaged interpreter instead, which needs no staging:
       sudo dnf install python3.${want:-13}"
  while IFS= read -r candidate; do
    [[ -x "$candidate/bin/python3" ]] || continue
    minor="$(python_minor "$candidate/bin/python3")" || continue
    [[ -n "$minor" ]] || continue
    ((minor >= floor_minor)) || continue
    [[ -n "$newest" ]] || newest="$candidate"
    if [[ -n "$want" && "$minor" == "$want" ]]; then
      preferred="$candidate"
      break
    fi
  done < <(find "$uv_root" -maxdepth 1 -type d -name 'cpython-3*' | sort -Vr)

  if [[ -n "$preferred" ]]; then
    printf '%s\n' "$preferred"
    return 0
  fi
  [[ -n "$newest" ]] || die \
    "no uv-managed CPython under $uv_root meets the jail's floor of 3.$floor_minor"
  echo "stage-jail-python: no 3.$want to match the venv; falling back to $newest" >&2
  printf '%s\n' "$newest"
}

[[ -n "$source_root" ]] || source_root="$(find_uv_source)"
source_root="$(cd -- "$source_root" && pwd -P)"
[[ -x "$source_root/bin/python3" ]] || die "no bin/python3 under $source_root"

source_minor="$(python_minor "$source_root/bin/python3")"
[[ -n "$source_minor" ]] || die "$source_root/bin/python3 will not run on this host"
((source_minor >= floor_minor)) || die \
  "$source_root is Python 3.$source_minor, below the jail's floor of 3.$floor_minor"

if [[ -e "$dest" && "$force" == 0 ]]; then
  # Ask the enforcing script rather than re-implementing its validation: if it accepts the tree
  # there is nothing to do, and if it rejects one it will say why.
  if resolved="$("$jail" --print-jail-python 2>/dev/null)"; then
    echo "stage-jail-python: already staged; the jail resolves $resolved"
    exit 0
  fi
  die "$dest exists but the jail rejects it. Re-run with --force, or read the reason:
    $jail --print-jail-python"
fi

staging="$(mktemp -d "$(dirname -- "$dest")/.stage-jail-python.XXXXXX")"
cleanup() { rm -rf -- "$staging"; }
trap cleanup EXIT

echo "stage-jail-python: copying $source_root (3.$source_minor) -> $dest"
cp -a "$source_root/." "$staging/"

# Group and other must not be writable anywhere in the tree: the destination's parent is
# world-writable, and a group-writable interpreter would hand the choice of grading interpreter
# to anyone in the group.
chmod -R go-w "$staging"
chmod 755 "$staging"

"$staging/bin/python3" -c 'import json, sqlite3, ssl, zlib' \
  || die "the staged copy cannot import its own stdlib -- the source is not relocatable"

# The whole point of a fixed destination is that episode_jail.sh can name it, so replacing an
# older tree has to be atomic enough that no jail launch sees a half-copied interpreter.
if [[ -e "$dest" ]]; then
  retired="$dest.retired.$$"
  mv -- "$dest" "$retired"
  mv -- "$staging" "$dest"
  rm -rf -- "$retired"
else
  mv -- "$staging" "$dest"
fi

resolved="$("$jail" --print-jail-python)" \
  || die "staged $dest, but the jail still refuses it -- see the error above"
echo "stage-jail-python: staged $dest; the jail resolves $resolved"
