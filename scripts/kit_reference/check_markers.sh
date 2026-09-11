#!/bin/bash
# Compare one launch-kit file against the rules in scripts/kit_reference/markers.tsv.
#
#   check_markers.sh [--skip-native] <user-data|runner> <file>
#
# Prints one line per broken rule on stdout, tab-separated: MISSING or FORBIDDEN, the rule name, and
# why the rule exists. Exit 0 when every rule holds, 1 when at least one is broken, 2 when the
# arguments or the files are unusable. Two callers share it, which is the point: tests/test_kit_reference.py
# renders the reference kit and requires every rule to hold (and sabotages each one to see it named
# here), and scripts/launch_gpu_box.sh runs it at preflight over a kit's rendered user-data and, when
# the user-data names a runner inside the staged tree, over that runner too -- as a WARNING per line,
# never a refusal, so a peer's kit that predates a rule still launches while saying what it lacks.
#
# --skip-native leaves out the user-data-native rows, which the launcher's own preflight already
# refuses or warns about; the launcher passes it so nothing is reported twice.
#
# Comment-only lines of the checked file are dropped before matching, so a rule cannot be satisfied
# by a comment that names it, and a forbidden pattern cannot be tripped by a comment warning against it.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 2
MARKERS=$HERE/markers.tsv

usage() {
  sed -n '2,20p' "$0" >&2
  exit 2
}

SKIP_NATIVE=0
if [ "${1:-}" = --skip-native ]; then
  SKIP_NATIVE=1
  shift
fi
[ $# -eq 2 ] || usage
SURFACE=$1
FILE=$2
case $SURFACE in
  user-data | runner) ;;
  *) usage ;;
esac
[ -r "$MARKERS" ] || {
  printf 'FATAL: %s is not readable\n' "$MARKERS" >&2
  exit 2
}
[ -r "$FILE" ] || {
  printf 'FATAL: %s is not readable\n' "$FILE" >&2
  exit 2
}

# The file with its comment-only lines blanked, in a temp file the greps read directly. Not piped:
# `grep -q` exits on its first match, the writer then takes SIGPIPE, and under pipefail the pipeline
# reports "no match" for a rule that is present -- observed as rules going missing at random between
# two runs over the same file. /var/tmp rather than /tmp, whose inode cap this box has hit before.
CODE_ONLY=$(mktemp /var/tmp/kit-markers-XXXXXX) || exit 2
trap 'rm -f -- "$CODE_ONLY"' EXIT
sed -e 's/^[[:space:]]*#.*$//' -- "$FILE" >"$CODE_ONLY" || exit 2

broken=0
while IFS=$'\t' read -r surface rule pattern why; do
  case $surface in
    '' | \#*) continue ;;
    "$SURFACE") ;;
    "$SURFACE-native")
      [ "$SKIP_NATIVE" = 1 ] && continue
      ;;
    *) continue ;;
  esac
  forbidden=0
  case $pattern in
    '!'*)
      forbidden=1
      pattern=${pattern#!}
      ;;
  esac
  # grep's 2 is a pattern it cannot use, which is a fault in the table, never a verdict on the file:
  # read as "no match" it would clear a forbidden rule and report a positive one as MISSING.
  grep -Eq -e "$pattern" -- "$CODE_ONLY"
  case $? in
    0) present=1 ;;
    1) present=0 ;;
    *)
      printf 'FATAL: rule %s has a pattern grep -E cannot use: %s\n' "$rule" "$pattern" >&2
      exit 2
      ;;
  esac
  if [ "$forbidden" = 1 ] && [ "$present" = 1 ]; then
    printf 'FORBIDDEN\t%s\t%s\n' "$rule" "$why"
    broken=1
  elif [ "$forbidden" = 0 ] && [ "$present" = 0 ]; then
    printf 'MISSING\t%s\t%s\n' "$rule" "$why"
    broken=1
  fi
done <"$MARKERS"
exit "$broken"
