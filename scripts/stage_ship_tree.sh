#!/bin/bash
# Stage the repository's current working tree to S3 under a NAMED run prefix, one sitting at a time.
#
#   <run-prefix>/sittings/<SITTING>/code.tar.gz
#   <run-prefix>/sittings/<SITTING>/data/<name>     (one per --data artifact)
#   <run-prefix>/provenance/<SITTING>-launch.txt    (this sitting's record: appended, never rewritten)
#
# Run identity is the run prefix the caller names, NEVER a hash (owner ruling, 2026-08-31: code must
# not be tied to pins, commits or hashes, and no path may carry one). The previous design keyed every
# object on the git tree hash; a peer commit then moved the derived prefix between a run's launch and
# its post-reclaim relaunch, and the relaunch started a fresh run into a junk prefix instead of
# resuming. Under this layout a relaunch that names the same run lands in the same prefix by
# construction, however far HEAD has moved.
#
# The SITTING is a timestamp-plus-pid label minted per invocation, so multi-sitting runs (reclaim
# relaunches) never overwrite an earlier sitting's code or record: newer sittings deliberately ship
# newer code, and the per-sitting provenance records are how the analysis side reconciles which
# sitting ran which bytes.
#
# What ships is ALWAYS the working tree as it stands now -- dirty and untracked files included, which
# `git archive HEAD` cannot do -- via ship_tree.sh's private-index primitive (the shared .git/index is
# left byte-identical; .gitignore semantics apply). Freezing writes refs/ship/<hash> first, purely as
# a provenance record so the exact bytes of any past sitting stay reproducible; nothing reads that
# ref to gate anything.
#
# Then it READS EVERY OBJECT BACK from the exact key the box will fetch it from and compares sha256.
# A producer-side "upload succeeded" is not the check that matters: one earlier run in this series had
# a green upload against a key nothing ever consumed, and the box found nothing there. This is an
# INTEGRITY gate, not a freshness gate -- it cannot fire because a peer edited a file.
#
#   stage_ship_tree.sh --run-prefix s3://<bucket>/<base>/<run-name> --region <region> \
#     [--sitting <label>] [--stage-dir <dir>] [--data <path>]...
#
# Gitignored data corpora cannot ride in the code archive by construction, so each one is passed with
# --data and pinned individually by sha256 in the record.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
SHIP_TREE_TOOL=$HERE/ship_tree.sh
SECRET_SCANNER=$HERE/scan_secrets.py
AWS_CLI=${AWS_CLI:-aws}

# Readers return through globals for the same reason as in ship_tree.sh: `die` inside a command
# substitution exits only the subshell, so the caller carries on and prints a second, wrong diagnosis.
TREE=""
HEAD_SHA=""
DIRTY=""
ARCHIVE_SHA256=""
TREE_DIGEST=""

die() {
  printf 'FATAL: %s\n' "$1" >&2
  exit 1
}

usage() {
  sed -n '2,37p' "$0" >&2
  exit 2
}

# --- arguments -----------------------------------------------------------------------------------

RUN_PREFIX=""
REGION=""
STAGE_DIR=""
SITTING=""
DATA_PATHS=()

parse_arguments() {
  while [ $# -gt 0 ]; do
    case $1 in
      --run-prefix)
        RUN_PREFIX=${2:?--run-prefix needs an s3:// URI}
        shift 2
        ;;
      --region)
        REGION=${2:?--region needs a region}
        shift 2
        ;;
      --stage-dir)
        STAGE_DIR=${2:?--stage-dir needs a directory}
        shift 2
        ;;
      --sitting)
        SITTING=${2:?--sitting needs a label}
        shift 2
        ;;
      --data)
        DATA_PATHS+=("${2:?--data needs a path}")
        shift 2
        ;;
      -h | --help) usage ;;
      *) die "unknown argument $1 (try --help)" ;;
    esac
  done
  [ -n "$RUN_PREFIX" ] || die "--run-prefix is required; it names the run and carries the bucket,
       which is account-specific and must come from the caller's environment"
  [ -n "$REGION" ] || die "--region is required"
  case $RUN_PREFIX in
    s3://*) ;;
    *) die "--run-prefix must be an s3:// URI, not $RUN_PREFIX" ;;
  esac
  # A trailing slash would produce s3://bucket/prefix//sittings/... which is a DIFFERENT key from
  # the one the render pins, and the box would fetch nothing.
  RUN_PREFIX=${RUN_PREFIX%/}
  if [ -z "$SITTING" ]; then
    # Timestamp plus pid: readable, unique across concurrent launches of one run, and free of any
    # commit-sha-shaped component (the T/Z/p letters keep every token out of the pure-hex shape).
    SITTING=$(date -u +%Y%m%dT%H%M%SZ)-p$$
  fi
  case $SITTING in
    *[!A-Za-z0-9._-]*) die "--sitting $SITTING carries characters outside [A-Za-z0-9._-], which
       would land verbatim in S3 keys" ;;
  esac
}

# --- the tree, the archive, the digest ------------------------------------------------------------

# --freeze rather than --facts, and BEFORE the archive is built: freezing writes refs/ship/<hash>,
# after which this sitting's bytes stay reproducible no matter what anyone edits next. The ref is a
# provenance record only -- nothing reads it to gate a launch.
read_tree_facts() {
  local facts
  facts=$(bash "$SHIP_TREE_TOOL" --freeze) || die "could not archive the working state"
  TREE=$(printf '%s\n' "$facts" | sed -n 's/^tree=//p')
  HEAD_SHA=$(printf '%s\n' "$facts" | sed -n 's/^head=//p')
  DIRTY=$(printf '%s\n' "$facts" | sed -n 's/^dirty=//p')
  [ -n "$TREE" ] || die "ship_tree.sh reported no tree hash"
  [ -n "$HEAD_SHA" ] || die "ship_tree.sh reported no HEAD sha"
  [ -n "$DIRTY" ] || die "ship_tree.sh reported no dirty flag"
}

build_archive() {
  local out=$1 result
  result=$(bash "$SHIP_TREE_TOOL" --tree "$TREE" --archive "$out") \
    || die "could not archive the frozen tree $TREE"
  ARCHIVE_SHA256=$(printf '%s\n' "$result" | sed -n 's/^archive_sha256=//p')
  [ -n "$ARCHIVE_SHA256" ] || die "ship_tree.sh --archive reported no sha256"
}

# Digested from the EXTRACTED archive rather than from the working tree. Hashing the working tree
# would pass by construction and check nothing about what actually ships.
#
# The extraction's modes are pinned, not inherited: the digest records each file's mode, non-root tar
# applies the caller's umask while root tar preserves the archive's recorded modes, and that skew made
# the first box launched through this flow refuse a perfectly good tree and halt on its fuse. The
# subshell umask plus --no-same-permissions makes every extracting user -- root included -- land
# identical modes. Must stay in lockstep with step_extract in box_ship_guard.sh, the other half of
# the same digest comparison.
read_archive_digest() {
  local archive=$1 extracted=$2
  rm -rf -- "$extracted"
  mkdir -p -- "$extracted" || die "cannot create $extracted"
  (umask 022 && tar --no-same-permissions -xzf "$archive" -C "$extracted") \
    || die "the archive at $archive does not extract"
  TREE_DIGEST=$(bash "$SHIP_TREE_TOOL" --digest "$extracted") \
    || die "could not digest the extracted archive at $extracted"
}

# Advisory, never gating. The repository's remote is public, so a scan over what is about to leave the
# machine is worth its two seconds; but this tool's job is shipping the tree, and a scanner false
# positive must not be able to block a launch. The pre-commit hook is what actually gates content.
scan_for_secrets() {
  local extracted=$1
  [ -x "$SECRET_SCANNER" ] || [ -f "$SECRET_SCANNER" ] || return 0
  if /usr/bin/python3 "$SECRET_SCANNER" "$extracted" >/dev/null 2>&1; then
    printf '   privacy scan over the shipped tree: clean\n' >&2
  else
    printf '   privacy scan over the shipped tree: FINDINGS (advisory, not blocking this launch).\n' >&2
    printf '   Re-run for the detail: /usr/bin/python3 %s %s\n' "$SECRET_SCANNER" "$extracted" >&2
  fi
}

# --- this sitting's provenance record --------------------------------------------------------------

# A RECORD, never a gate: it says what bytes this sitting shipped -- HEAD, the working tree's status,
# the tarball's checksum -- so forensics can always answer "what ran", and nothing reads it to refuse
# anything. Both dirty views are kept deliberately: dirty_path= is derived from the ship tree itself
# (so it cannot disagree with what shipped, and renders untracked files as the additions they are),
# while git_status= is the operator's familiar `git status --porcelain` view at stage time.
write_record() {
  local out=$1 path name
  {
    printf 'run_prefix=%s\n' "$RUN_PREFIX"
    printf 'sitting=%s\n' "$SITTING"
    printf 'ship_tree=%s\n' "$TREE"
    printf 'head_sha=%s\n' "$HEAD_SHA"
    printf 'dirty=%s\n' "$DIRTY"
    printf 'archive_sha256=%s\n' "$ARCHIVE_SHA256"
    printf 'tree_digest=%s\n' "$TREE_DIGEST"
    printf 's3_key=%s\n' "$(code_key)"
    printf 'staged_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    git -C "$(repo_root)" diff --name-status HEAD "$TREE" | sed 's/^/dirty_path=/'
    git -C "$(repo_root)" status --porcelain | sed 's/^/git_status=/'
    for path in ${DATA_PATHS+"${DATA_PATHS[@]}"}; do
      name=$(basename -- "$path")
      printf 'data=%s %s\n' "$name" "$(sha256sum -- "$path" | cut -d' ' -f1)"
    done
  } >"$out" || die "could not write the provenance record to $out"
  [ -s "$out" ] || die "the provenance record at $out came out empty"
}

repo_root() {
  git -C "$HERE" rev-parse --show-toplevel
}

code_key() {
  printf '%s/sittings/%s/code.tar.gz' "$RUN_PREFIX" "$SITTING"
}

provenance_key() {
  printf '%s/provenance/%s-launch.txt' "$RUN_PREFIX" "$SITTING"
}

data_key() {
  printf '%s/sittings/%s/data/%s' "$RUN_PREFIX" "$SITTING" "$1"
}

# --- upload, then read back from the consumption key ----------------------------------------------

upload() {
  local src=$1 key=$2
  # Absolute source, no `cd` prefix: the permission allow-list on this box matches on command SHAPE,
  # and `cd <dir> && aws s3 cp .` is denied where the same write in this shape succeeds. Three
  # consecutive denials once read as a deliberate block on a benign write and stalled a launch.
  "$AWS_CLI" s3 cp "$src" "$key" --region "$REGION" --only-show-errors \
    || die "staging $src to $key failed"
}

verify_readback() {
  local key=$1 src=$2 back=$3 local_sha remote_sha
  "$AWS_CLI" s3 cp "$key" "$back" --region "$REGION" --only-show-errors || {
    printf '  MISSING at the consumption key: %s\n' "$key" >&2
    return 1
  }
  local_sha=$(sha256sum -- "$src" | cut -d' ' -f1)
  remote_sha=$(sha256sum -- "$back" | cut -d' ' -f1)
  if [ "$local_sha" = "$remote_sha" ]; then
    printf '  OK   %s  %s\n' "$key" "${local_sha:0:12}" >&2
    return 0
  fi
  printf '  SKEW %s  local %s != staged %s\n' "$key" "${local_sha:0:12}" "${remote_sha:0:12}" >&2
  return 1
}

main() {
  parse_arguments "$@"
  read_tree_facts

  if [ -z "$STAGE_DIR" ]; then
    STAGE_DIR=/var/tmp/ship-stage-$SITTING
  fi
  mkdir -p -- "$STAGE_DIR" || die "cannot create the stage directory $STAGE_DIR"

  local archive=$STAGE_DIR/code.tar.gz
  local extracted=$STAGE_DIR/extracted
  local record=$STAGE_DIR/ship-manifest.txt
  local back=$STAGE_DIR/readback

  build_archive "$archive"
  read_archive_digest "$archive" "$extracted"
  scan_for_secrets "$extracted"
  write_record "$record"

  printf '== staging sitting %s of %s (tree %s, HEAD %s, dirty=%s)\n' \
    "$SITTING" "$RUN_PREFIX" "${TREE:0:12}" "${HEAD_SHA:0:12}" "$DIRTY" >&2
  upload "$archive" "$(code_key)"
  upload "$record" "$(provenance_key)"
  local path name
  for path in ${DATA_PATHS+"${DATA_PATHS[@]}"}; do
    [ -s "$path" ] || die "--data $path is missing or empty"
    upload "$path" "$(data_key "$(basename -- "$path")")"
  done

  printf '== reading every object back from the key the box will fetch it from\n' >&2
  rm -rf -- "$back"
  mkdir -p -- "$back" || die "cannot create $back"
  local problems=0
  verify_readback "$(code_key)" "$archive" "$back/code.tar.gz" || problems=$((problems + 1))
  verify_readback "$(provenance_key)" "$record" "$back/ship-manifest.txt" \
    || problems=$((problems + 1))
  for path in ${DATA_PATHS+"${DATA_PATHS[@]}"}; do
    name=$(basename -- "$path")
    verify_readback "$(data_key "$name")" "$path" "$back/$name" || problems=$((problems + 1))
  done
  [ "$problems" -eq 0 ] \
    || die "$problems object(s) did not read back identically from their consumption keys. Do NOT
       launch: the box would fetch something other than what was staged here."

  printf 'run_prefix=%s\n' "$RUN_PREFIX"
  printf 'sitting=%s\n' "$SITTING"
  printf 'ship_tree=%s\n' "$TREE"
  printf 'head_sha=%s\n' "$HEAD_SHA"
  printf 'dirty=%s\n' "$DIRTY"
  printf 'archive_sha256=%s\n' "$ARCHIVE_SHA256"
  printf 'tree_digest=%s\n' "$TREE_DIGEST"
  printf 's3_key=%s\n' "$(code_key)"
  printf 'provenance_key=%s\n' "$(provenance_key)"
  printf 'stage_dir=%s\n' "$STAGE_DIR"
}

main "$@"
