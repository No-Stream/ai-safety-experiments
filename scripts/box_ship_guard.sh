#!/bin/bash
# The box's own check that it is running the code it was sent, run before anything else on a rented
# GPU box.
#
# EMITTED, NOT SHIPPED. scripts/render_box_userdata.sh inlines this file into the instance's user-data
# at launch, so the box runs the launcher's copy. Shipping it inside the code bundle instead would put
# the verifier and the thing it verifies in the same tarball, and a stale bundle then carries a stale
# verifier that happily attests to itself. Every input arrives as a shell variable the render
# substitutes, never from anything the box downloads.
#
# Two distinct checks, because they fail for different reasons and neither implies the other:
#
#   1. the fetched archive's sha256 against the value pinned at launch. Catches a swapped, truncated
#      or half-uploaded object under the right key.
#   2. a git-free digest of the EXTRACTED tree against the value pinned at launch. Catches a bad
#      extraction and, more importantly, a leftover file from a previous unpack in the same directory
#      -- about which a correct archive sha256 says exactly nothing.
#
# Fails CLOSED, deliberately opposite to the killswitch's best-effort posture: a box that refuses to
# start costs one relaunch, while a box that runs stale code costs a wrong experimental result that
# looks correct. So a mismatch halts the box on a short fuse rather than letting it idle to the
# dead-man switch hours later.
#
# Steps are separately invocable so the containment claim can be attacked directly:
# tests/test_ship_tree.py runs each of them outside any staged context and requires every one to fail
# there. A guard that passes with nothing staged is a reassuring message, not a check.
#
#   box_ship_guard.sh [all|fetch|verify-archive|extract|verify-digest|publish]
#
# Required in the environment (the render substitutes all of them):
#   SHIP_TREE           the working-state git tree hash this box was launched from
#   SHIP_CODE_SHA256    sha256 of the code archive, as pinned at launch
#   SHIP_TREE_DIGEST    git-free digest of the extracted tree, as pinned at launch
#   SHIP_S3_KEY         s3:// URI of the code archive
#   SHIP_S3_REGION      region for the s3 calls
#   SHIP_EXTRACT_DIR    where the code goes, e.g. /home/ubuntu/repo
# Optional:
#   SHIP_ARCHIVE_PATH   where to keep the downloaded archive (default /var/tmp/ship-code.tar.gz)
#   SHIP_FUSE_MINUTES   minutes on the halt fuse (default 5)
#   SHIP_PROVENANCE_KEY s3:// object this box writes after verifying (default alongside the archive)
#   AWS_CLI             the aws binary (default: aws from PATH)
set -uo pipefail

SHA256_OF_NOTHING=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

AWS_CLI=${AWS_CLI:-aws}
SHIP_ARCHIVE_PATH=${SHIP_ARCHIVE_PATH:-/var/tmp/ship-code.tar.gz}
SHIP_FUSE_MINUTES=${SHIP_FUSE_MINUTES:-5}

# Halting is what makes this a gate rather than a log line. `shutdown` is resolved from PATH rather
# than hardcoded to /sbin so the tests can put a recorder in front of it and assert the fuse was
# actually armed, instead of asserting on a message about arming it.
halt_unrunnable() {
  printf 'FATAL: %s\n' "$1" >&2
  printf 'FATAL: halting this box on a %s-minute fuse rather than running unverified code, or\n' \
    "$SHIP_FUSE_MINUTES" >&2
  printf '       idling until the dead-man switch notices hours from now.\n' >&2
  if command -v shutdown >/dev/null 2>&1; then
    shutdown -h "+$SHIP_FUSE_MINUTES" "ship guard: $1"
  else
    printf 'FATAL: no shutdown binary on PATH, so this box will idle to its dead-man switch.\n' >&2
  fi
  exit 1
}

require_var() {
  local name=$1 value=${2:-}
  [ -n "$value" ] \
    || halt_unrunnable "$name is empty, so the guard has nothing to check against. The render that
       produced this user-data should have substituted it, and a guard with no expected value is
       not a guard."
}

require_pinned_values() {
  require_var SHIP_TREE "${SHIP_TREE:-}"
  require_var SHIP_CODE_SHA256 "${SHIP_CODE_SHA256:-}"
  require_var SHIP_TREE_DIGEST "${SHIP_TREE_DIGEST:-}"
  require_var SHIP_S3_KEY "${SHIP_S3_KEY:-}"
  require_var SHIP_S3_REGION "${SHIP_S3_REGION:-}"
  require_var SHIP_EXTRACT_DIR "${SHIP_EXTRACT_DIR:-}"
  # A digest of empty input is a valid-looking hash that every other empty tree also has, so pinning
  # it would make the extracted-tree check pass on a box with no code at all.
  [ "$SHIP_TREE_DIGEST" != "$SHA256_OF_NOTHING" ] \
    || halt_unrunnable "SHIP_TREE_DIGEST is sha256 of empty input, which every empty tree matches"
  [ "$SHIP_CODE_SHA256" != "$SHA256_OF_NOTHING" ] \
    || halt_unrunnable "SHIP_CODE_SHA256 is sha256 of empty input, which every empty file matches"
}

step_fetch() {
  require_pinned_values
  printf '== fetching %s\n' "$SHIP_S3_KEY"
  rm -f -- "$SHIP_ARCHIVE_PATH"
  "$AWS_CLI" s3 cp "$SHIP_S3_KEY" "$SHIP_ARCHIVE_PATH" --region "$SHIP_S3_REGION" \
    --only-show-errors \
    || halt_unrunnable "could not fetch the code archive from $SHIP_S3_KEY, so there is nothing to run"
  [ -s "$SHIP_ARCHIVE_PATH" ] \
    || halt_unrunnable "the fetch of $SHIP_S3_KEY produced an empty file at $SHIP_ARCHIVE_PATH"
}

step_verify_archive() {
  require_pinned_values
  [ -s "$SHIP_ARCHIVE_PATH" ] \
    || halt_unrunnable "no archive at $SHIP_ARCHIVE_PATH to verify; the fetch step did not run"
  printf '== checking the archive against the sha256 pinned at launch\n'
  printf '%s  %s\n' "$SHIP_CODE_SHA256" "$SHIP_ARCHIVE_PATH" | sha256sum -c - \
    || halt_unrunnable "the archive under $SHIP_S3_KEY does not match the sha256 pinned at launch
       ($SHIP_CODE_SHA256). The object at that key is not the object this box was sent."
}

# The extraction's modes are pinned, not inherited, because the digest below records each file's
# mode: this guard runs as ROOT under cloud-init, root tar preserves the archive's recorded modes
# while the staging digest extracted as a non-root user whose umask rewrote them, and that skew alone
# made the first box launched through this flow refuse a perfectly good tree. The subshell umask plus
# --no-same-permissions lands the same modes for every extracting user. Must stay in lockstep with
# read_archive_digest in scripts/stage_ship_tree.sh, the other half of the same digest comparison;
# the render refuses a guard that lost the flag.
step_extract() {
  require_pinned_values
  [ -s "$SHIP_ARCHIVE_PATH" ] \
    || halt_unrunnable "no archive at $SHIP_ARCHIVE_PATH to extract; earlier steps did not run"
  printf '== extracting into %s\n' "$SHIP_EXTRACT_DIR"
  mkdir -p -- "$SHIP_EXTRACT_DIR" \
    || halt_unrunnable "cannot create the extract directory $SHIP_EXTRACT_DIR"
  (umask 022 && tar --no-same-permissions -xzf "$SHIP_ARCHIVE_PATH" -C "$SHIP_EXTRACT_DIR") \
    || halt_unrunnable "extracting $SHIP_ARCHIVE_PATH into $SHIP_EXTRACT_DIR failed"
}

# DELIBERATELY DUPLICATED from read_tree_digest in scripts/ship_tree.sh -- see this file's header for
# why the box must not use the shipped copy. tests/test_ship_tree.py digests one tree with both
# implementations and requires the answers to be equal, which is what keeps the two from drifting.
step_verify_digest() {
  require_pinned_values
  [ -d "$SHIP_EXTRACT_DIR" ] \
    || halt_unrunnable "$SHIP_EXTRACT_DIR does not exist, so there is no extracted tree to digest"
  local listing count observed
  listing=$(mktemp /var/tmp/ship-guard-digest-XXXXXX) \
    || halt_unrunnable "cannot create a listing file to digest the extracted tree"
  find "$SHIP_EXTRACT_DIR" -mindepth 1 \( -type f -o -type l \) -printf '%P\0' \
    | LC_ALL=C sort -z >"$listing" \
    || halt_unrunnable "listing $SHIP_EXTRACT_DIR failed, so its digest cannot be computed"
  count=$(tr -cd '\0' <"$listing" | wc -c)
  if [ "$count" -eq 0 ]; then
    rm -f -- "$listing"
    halt_unrunnable "$SHIP_EXTRACT_DIR contains no files at all, so the code never landed"
  fi
  observed=$(
    cd -- "$SHIP_EXTRACT_DIR" || exit 1
    while IFS= read -r -d '' path; do
      if [ -L "$path" ]; then
        printf '%s\0L\0%s\0' "$path" "$(readlink -- "$path")"
      else
        printf '%s\0F\0%s\0%s\0%s\0' "$path" "$(stat -c '%a' -- "$path")" \
          "$(stat -c '%s' -- "$path")" "$(sha256sum -- "$path" | cut -d' ' -f1)"
      fi
    done <"$listing" | sha256sum | cut -d' ' -f1
  ) || halt_unrunnable "digesting $SHIP_EXTRACT_DIR failed"
  rm -f -- "$listing"
  printf '== extracted tree digest %s over %s file(s)\n' "$observed" "$count"
  [ "$observed" = "$SHIP_TREE_DIGEST" ] \
    || halt_unrunnable "the extracted tree digests to $observed but $SHIP_TREE_DIGEST was pinned at
       launch. The archive was intact, so this is a bad extraction or a file left behind in
       $SHIP_EXTRACT_DIR by an earlier unpack."
}

# Written after the checks pass so the box's own account of what it verified outlives it. The
# instance id comes from IMDSv2 because these boxes launch with HttpTokens=required; a box that
# cannot name itself still publishes, under a name that says so, rather than failing the run.
#
# It re-runs the digest check rather than trusting that an earlier step did. Publishing is the one step
# that makes a CLAIM about this box to anything outside it, so it must not be possible to produce that
# claim by invoking the steps out of order, or by invoking this one alone.
step_publish() {
  require_pinned_values
  step_verify_digest
  local token instance key record
  token=$(curl -sf -X PUT 'http://169.254.169.254/latest/api/token' \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null) || token=""
  if [ -n "$token" ]; then
    instance=$(curl -sf -H "X-aws-ec2-metadata-token: $token" \
      'http://169.254.169.254/latest/meta-data/instance-id' 2>/dev/null) || instance=""
  else
    instance=""
  fi
  [ -n "$instance" ] || instance="unidentified-$(hostname)"
  key=${SHIP_PROVENANCE_KEY:-$(dirname -- "$SHIP_S3_KEY")/boxes/$instance.txt}
  record=$(mktemp /var/tmp/ship-provenance-XXXXXX) \
    || halt_unrunnable "cannot create a provenance record file"
  {
    printf 'instance=%s\n' "$instance"
    printf 'ship_tree=%s\n' "$SHIP_TREE"
    printf 'code_sha256=%s\n' "$SHIP_CODE_SHA256"
    printf 'tree_digest=%s\n' "$SHIP_TREE_DIGEST"
    printf 's3_key=%s\n' "$SHIP_S3_KEY"
    printf 'extract_dir=%s\n' "$SHIP_EXTRACT_DIR"
    printf 'verified_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$record"
  "$AWS_CLI" s3 cp "$record" "$key" --region "$SHIP_S3_REGION" --only-show-errors \
    || halt_unrunnable "could not publish this box's provenance record to $key"
  rm -f -- "$record"
  printf '== published provenance to %s\n' "$key"
}

main() {
  case ${1:-all} in
    # step_publish re-runs step_verify_digest, so this sequence digests the extracted tree twice. That
    # is deliberate rather than an oversight: publishing is the one step that makes a claim about this
    # box to anything outside it, and re-verifying is what makes the claim true however the steps were
    # invoked. Measured at ~3 seconds per pass over 356 files, against a boot measured in minutes.
    all)
      step_fetch
      step_verify_archive
      step_extract
      step_verify_digest
      step_publish
      printf '== ship guard PASSED: this box is running ship tree %s\n' "$SHIP_TREE"
      ;;
    fetch) step_fetch ;;
    verify-archive) step_verify_archive ;;
    extract) step_extract ;;
    verify-digest) step_verify_digest ;;
    publish) step_publish ;;
    *)
      printf 'usage: %s [all|fetch|verify-archive|extract|verify-digest|publish]\n' "$0" >&2
      exit 2
      ;;
  esac
}

main "$@"
