#!/bin/bash
# One shared, delta-synced, read-only mirror per training run, so that several agents reading the same
# finished run read one local copy instead of each pulling the same objects down again.
#
#   s3_mirror.sh <run-name> --holder <tag> [--include-logs] [--include-checkpoints]
#                           [--include-checkpoint-state] [--allow-empty]
#   s3_mirror.sh <run-name> --status
#   s3_mirror.sh <run-name> --release --holder <tag>
#   s3_mirror.sh <run-name> --delete [--force]
#
# Stdout carries a mirror directory that exists and has files in it, and nothing else. Every mode that
# leaves no such directory -- a refusal, a release, a delete, a --status on a run nobody has mirrored
# -- prints nothing there and exits non-zero, so a caller cannot mistake any of them for a stage:
#
#   stage=$(scripts/s3_mirror.sh track-record-v2 --holder readout-a) && readout.py --stage "$stage"
#   scripts/s3_mirror.sh track-record-v2 --release --holder readout-a
#
# WHY. Prep, readout and audit agents each staged a private copy of a run's artifacts into their own
# scratch directory, so one run's objects came down the wire once per agent, and an agent reaping its
# own scratch took with it the copy a sibling was still reading (which is what happened to one GRPO
# audit mid-analysis). Here the destination is a pure function of the run name, `aws s3 sync` makes a
# second call over an unchanged prefix transfer nothing, and a reader marks the mirror in use so that
# a sibling's clean-up refuses instead of pulling the floor out.
#
# READ-ONLY TOWARDS S3, BY CONSTRUCTION. The only call this script makes is
# `aws s3 sync <s3 source> <local destination>`. No s3:// destination is ever composed, `--delete` is
# never passed, and nothing here uploads. Read-only towards the mirror is a convention rather than a
# permission, and worth keeping: `sync` compares size and modification time, so a file edited in place
# to the same length is never corrected by a later sync.
#
# THE DEFAULT EXCLUDES drop the two classes a readout never opens, and the checkpoints are where
# essentially all of the saving is. Checkpoint directories are read only by a resume or an eval on a
# GPU box: on track-record-v2, 792 of 939 objects and 16.00 of 16.83 GB (95%) sit under a
# `checkpoint-*` component, so the default mirror of that run is 147 objects and 0.83 GB. Trainer logs
# are the largest single object a run writes, but where a run puts them moved between vintages -- 15
# `.log` objects totalling 3.75 GB (largest 2.68 GB) sit under wave3-ladder-joint's own prefix, while
# recent runs ship them to the sibling prefix described below and keep none under their own (measured
# 2026-09-04: zero `.log` objects under track-record-v2's prefix). The checkpoint patterns are
# deliberately broad (any path component starting `checkpoint-`), so a run keeping something else
# checkpoint-named needs `--include-checkpoints` to see it.
#
# `--include-checkpoint-state` is the middle path the readout recipes actually want: it re-includes
# `checkpoint-*/trainer_state.json` and nothing else from a checkpoint directory. That file is what a
# readout reads per checkpoint (learning-rate continuity across a resume seam, the ghost-restart
# check), and on track-record-v2 it is 72 objects and 2.88 MiB against the 16.00 GB of weights and
# optimizer state beside them, so without the flag a recipe needing it has to take the whole tree.
#
# `--include-logs` MIRRORS A SECOND, SIBLING PREFIX, because that is where a run's logs are: not under
# the run's own prefix but at <base>/logs/<run-name>/. On track-record-v2 that sibling holds 13
# objects and 1.93 GB, one of them the 1.93 GB trainer log, while the run's own prefix holds no `.log`
# at all, which is why both readout recipes stage logs with a separate fourth sync into $STAGE/logs/.
# So the flag does two things: it drops the `*.log` exclude from the run prefix (for a run of
# wave3-ladder-joint's vintage, which kept them there) and it mirrors the sibling prefix into
# <mirror>/logs/, the same place the recipes put it. Default callers touch neither, which is where the
# headline saving is: nothing here downloads that 1.93 GB object unless it is asked to.
#
# THE MIRROR IS THE UNION OF EVERY FLAG SET EVER USED ON IT. `sync` never prunes and `--delete` is
# never passed to the CLI, so one `--include-checkpoints` call leaves 16 GB of checkpoints sitting
# there for every later default caller. A reader cannot infer the mirror's contents from its own
# flags: ask `--status`, or reap it and take it again.
#
# THE IN-USE MARKER is one file per holder under <root>/.holders/<run-name>/<tag>, and `--delete`
# refuses while any of them exists. One file per holder rather than one flag per mirror because the
# whole point is overlap: two agents hold the same mirror, and the second releasing must not clear the
# first's claim. A file rather than an flock because the holder is the READER, which outlives this
# process by minutes to hours -- a lock held by this script would be gone the moment it printed the
# path. Nothing here expires a marker on its own: a marker left by a dead agent looks exactly like a
# live agent between two reads, so `--delete --force` is where that judgement is made, by a person,
# with the holder list in front of them.
#
# THE HOLDER TAG HAS NO DEFAULT, deliberately: a sync or a release without one is refused, and the
# reason is that the only name this process knows for its caller is its parent shell's pid, which is
# the one thing the tag must not be. Two readers started from one shell share that pid, so the first
# one's release clears the second's claim -- the exact accident the marker exists to prevent. One
# reader that syncs inside `$(...)` and releases from a later shell gets two different values, so the
# release finds nothing, the claim outlives the reader and only `--delete --force` can clear it, which
# trains agents on the escape hatch. Both were measured rather than reasoned about. An agent reading
# several runs exports S3_MIRROR_HOLDER once instead of repeating --holder.
#
# WHAT THE MARKER DOES NOT COVER. It guards deletion, not the sync: two agents syncing one run at once
# write the same objects to the same paths, so the cost is a duplicated transfer rather than a wrong
# mirror, and serialising that is not worth a lock nobody can hold. And `--delete` reads the holders
# and then removes the directory, so a sync taking its marker between those two steps loses its mirror
# anyway. That window is milliseconds wide between cooperating agents and closing it properly wants
# the same lock, so it is written down here rather than papered over with a check that looks like one.
#
# ACCOUNT-SPECIFIC VALUES ARE NEVER TRACKED IN THIS REPOSITORY (see AGENTS.md). The bucket and base
# prefix arrive in SHIP_S3_PREFIX -- the same variable scripts/launch_gpu_box.sh takes its --s3-prefix
# default from, so a run is mirrored from exactly the prefix its box wrote to -- the region arrives in
# SHIP_S3_REGION, and the credentials arrive however the calling shell resolves them (AWS_PROFILE, or
# an instance profile on a box). An unset variable is a refusal naming it; there is deliberately no
# default to fall back on.
#
# Required in the environment for a sync, and only for a sync -- --status, --release and --delete are
# local and need no AWS configuration at all:
#   SHIP_S3_PREFIX      s3:// base prefix; the run's own prefix is <base>/<run-name>/ (or --prefix)
#   SHIP_S3_REGION      region for the s3 calls (AWS_REGION and AWS_DEFAULT_REGION too, or --region)
# Optional:
#   S3_MIRROR_ROOT      where mirrors live (default /var/tmp/s3-mirror; must be under /var/tmp)
#   S3_MIRROR_HOLDER    the holder tag, for a caller that would otherwise repeat --holder
#   AWS_CLI             the aws binary (default: aws from PATH)
set -uo pipefail

# Mirrors go on real disk under /var/tmp rather than in /tmp, which on this box is a RAM-backed tmpfs
# with a hard inode cap shared by every session, and never in the home tree, which is the rule
# episode_jail.sh and reward_hacking/harness/loop.py already hold disposable scratch to.
DEFAULT_MIRROR_ROOT=/var/tmp/s3-mirror

# Markers live beside the mirrors rather than inside them, so that `--delete` cannot destroy the
# evidence of who was holding it. A run name must begin with an alphanumeric character (see
# validate_run_name), so this dot-prefixed sibling can never collide with a mirror of the same name.
HOLDERS_DIR_NAME=.holders

# Where a run's logs live relative to the base prefix, and where they land in the mirror: both readout
# recipes sync <base>/logs/<run>/ into $STAGE/logs/, and a mirror that spells it differently could not
# stand in for that step.
LOGS_DIR_NAME=logs

AWS_CLI=${AWS_CLI:-aws}
MIRROR_ROOT=${S3_MIRROR_ROOT:-$DEFAULT_MIRROR_ROOT}
S3_PREFIX=${SHIP_S3_PREFIX:-}
S3_REGION=${SHIP_S3_REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-}}}
HOLDER=${S3_MIRROR_HOLDER:-}

RUN_NAME=""
MODE=sync
INCLUDE_LOGS=0
INCLUDE_CHECKPOINTS=0
INCLUDE_CHECKPOINT_STATE=0
ALLOW_EMPTY=0
FORCE=0

log() {
  printf 's3_mirror: %s\n' "$1" >&2
}

die() {
  printf 's3_mirror FATAL: %s\n' "$1" >&2
  exit 1
}

usage() {
  # Up to the set line rather than to a line number, so a header edit cannot leave the usage stale.
  sed -n '2,/^set -uo pipefail$/{ /^set -uo pipefail$/!p; }' "$0" >&2
  exit 2
}

set_mode() {
  [ "$MODE" = sync ] || die "--$1 and --$MODE are separate modes; pass one"
  MODE=$1
}

while [ $# -gt 0 ]; do
  case $1 in
    --status | --release | --delete) set_mode "${1#--}" ;;
    --include-logs) INCLUDE_LOGS=1 ;;
    --include-checkpoints) INCLUDE_CHECKPOINTS=1 ;;
    --include-checkpoint-state) INCLUDE_CHECKPOINT_STATE=1 ;;
    --allow-empty) ALLOW_EMPTY=1 ;;
    --force) FORCE=1 ;;
    --holder)
      HOLDER=${2:?--holder needs a tag}
      shift
      ;;
    --prefix)
      S3_PREFIX=${2:?--prefix needs an s3:// URI}
      shift
      ;;
    --region)
      S3_REGION=${2:?--region needs a region}
      shift
      ;;
    -h | --help) usage ;;
    -*) die "unknown option $1" ;;
    *)
      [ -z "$RUN_NAME" ] || die "one run name at a time; got '$RUN_NAME' and then '$1'"
      RUN_NAME=$1
      ;;
  esac
  shift
done

# The run name becomes a path component and an S3 key component, so it is held to the same shape
# scripts/launch_gpu_box.sh holds a run name to. This is the traversal guard's first layer: a slash, a
# leading dot or `..` is refused here, before anything is created.
validate_run_name() {
  [ -n "$RUN_NAME" ] || die "a run name is required; it is the run's whole identity, and the mirror
       of a run is <root>/<run-name>/ by construction"
  [ "${#RUN_NAME}" -le 64 ] || die "the run name is ${#RUN_NAME} characters; keep it to 64"
  case $RUN_NAME in
    [!A-Za-z0-9]* | *[!A-Za-z0-9._-]*)
      die "the run name '$RUN_NAME' must start with an alphanumeric character and use only letters,
       digits and [._-]. A slash, a leading dot or '..' would put the mirror somewhere other than
       under $MIRROR_ROOT/, which is the only place this tool may write or delete."
      ;;
  esac
}

# A flag that cannot do anything in the chosen mode is a refusal rather than a silent no-op: a caller
# who passed --force to a sync, or --include-checkpoints to a --delete, believes something happened.
validate_flags_for_mode() {
  local sync_flags=$((INCLUDE_LOGS + INCLUDE_CHECKPOINTS + INCLUDE_CHECKPOINT_STATE + ALLOW_EMPTY))
  [ "$MODE" = sync ] || [ "$sync_flags" -eq 0 ] \
    || die "the --include-* and --allow-empty flags shape a sync, and --$MODE does not sync"
  [ "$MODE" = delete ] || [ "$FORCE" -eq 0 ] \
    || die "--force clears an in-use marker for --delete, and this is a --$MODE"
}

# Only the two modes that touch a marker read the tag, so a malformed S3_MIRROR_HOLDER in the
# environment does not refuse a --status or a --delete that never looks at it.
validate_holder() {
  [ -n "$HOLDER" ] || die "this mode takes or drops an in-use marker, so it needs a holder tag naming
       the reader: pass --holder <tag>, or export S3_MIRROR_HOLDER. There is deliberately no default,
       because the only name this process knows for its caller is its parent shell's pid, which two
       readers in one shell share and one reader across two shells does not (see the header)."
  case $HOLDER in
    [!A-Za-z0-9]* | *[!A-Za-z0-9._-]*)
      die "the holder tag '$HOLDER' must start with an alphanumeric character and use only letters,
       digits and [._-]; it is one path component under $MIRROR_ROOT/$HOLDERS_DIR_NAME/"
      ;;
  esac
}

# A path this script creates in or deletes must be exactly where its name says. `cd -P` collapses
# every symlink and `..`, so a directory that resolves to its own name is a real directory there.
require_own_path() {
  local path=$1 resolved
  [ ! -L "$path" ] \
    || die "$path is a symlink to $(readlink -- "$path"); this script creates in and deletes under
       that path, and both mkdir -p and rm -rf follow a symlink, so it will not touch one"
  [ -e "$path" ] || return 0
  resolved=$(cd -- "$path" && pwd -P) \
    || die "$path exists and is not a directory this script can enter; refusing to touch it"
  [ "$resolved" = "$path" ] \
    || die "$path resolves to $resolved, outside $MIRROR_ROOT; refusing to touch it"
}

# Resolves MIRROR_ROOT, DEST and HOLDERS_PATH, and carries the traversal guard's second layer: each
# of the three paths this script creates in or deletes -- the mirror, the holder directory and the
# run's holder subdirectory -- must be a real directory at its own name. The holder paths need that as
# much as the mirror does, because `rm -rf` and `mkdir -p` both follow a symlinked intermediate
# component: with <root>/.holders pointed elsewhere, an unvalidated `rm -rf <root>/.holders/<run>`
# deleted a subtree outside the root, said "nothing to delete" and exited 0.
#
# The root is resolved once, before any of it is created, and not re-checked afterwards: `mkdir -p`
# only ever creates real directories and refuses a dangling symlink component, so a second look could
# differ only if something rearranged /var/tmp mid-run, and asking twice narrows that window rather
# than closing it. The same reasoning retired a check that the mirror's parent resolves to the root:
# with the run name refusing every slash and leading dot, and the root already physical, it could not
# fail. A check nothing can make fail is a message, not a check.
resolve_paths() {
  case $MIRROR_ROOT in
    /var/tmp/?*) ;;
    *) die "the mirror root is '$MIRROR_ROOT'; mirrors live under /var/tmp/ -- /tmp here is a
       RAM-backed tmpfs with a shared inode cap, and the home tree is not for disposable bulk data" ;;
  esac
  # The deepest existing part of the root is resolved before the rest of it is created, so a symlink
  # anywhere along the path (/var/tmp is drwxrwxrwt, so anything on the box can plant one) is refused
  # without a directory having been created outside /var/tmp on the way to refusing it.
  local ancestor resolved
  ancestor=$MIRROR_ROOT
  while [ ! -d "$ancestor" ]; do
    ancestor=$(dirname -- "$ancestor")
  done
  resolved=$(cd -- "$ancestor" && pwd -P) || die "could not resolve $ancestor"
  case $resolved in
    /var/tmp | /var/tmp/*) ;;
    *) die "the mirror root '$MIRROR_ROOT' reaches '$resolved', outside /var/tmp/, through a symlink
       or a '..'; refusing before anything is created there" ;;
  esac
  mkdir -p -- "$MIRROR_ROOT" || die "could not create the mirror root $MIRROR_ROOT"
  MIRROR_ROOT=$(cd -- "$MIRROR_ROOT" && pwd -P) || die "could not resolve the mirror root"

  DEST=$MIRROR_ROOT/$RUN_NAME
  HOLDERS_ROOT=$MIRROR_ROOT/$HOLDERS_DIR_NAME
  HOLDERS_PATH=$HOLDERS_ROOT/$RUN_NAME

  require_own_path "$DEST"
  require_own_path "$HOLDERS_ROOT"
  require_own_path "$HOLDERS_PATH"
}

require_source() {
  [ -n "$S3_PREFIX" ] || die "SHIP_S3_PREFIX is unset, so there is no source prefix to sync from
       (or pass --prefix). It carries the bucket and base prefix, which are account-specific and are
       never tracked in this repository, so there is no default to fall back on."
  case $S3_PREFIX in
    s3://?*) ;;
    *) die "the source prefix must be an s3:// URI, not '$S3_PREFIX'" ;;
  esac
  [ -n "$S3_REGION" ] || die "SHIP_S3_REGION is unset, so the s3 calls have no region (AWS_REGION and
       AWS_DEFAULT_REGION are read too, or pass --region)"
}

# The tool's one safety query, so it fails closed: an unreadable holder directory means this script
# cannot tell whether a reader is holding the mirror, and "cannot tell" must never read as "nobody
# is". Callers write `holders=$(held_by) || exit 1`, because a die inside a command substitution exits
# only the subshell -- the message would print while the caller carried on with an empty holder list.
held_by() {
  [ -d "$HOLDERS_PATH" ] || return 0
  local tags
  tags=$(find "$HOLDERS_PATH" -maxdepth 1 -type f -printf '%f ') \
    || die "could not read the holder directory $HOLDERS_PATH, so whether a reader is holding this
       mirror is unknown; refusing rather than reporting it free"
  printf '%s' "${tags% }"
}

take_marker() {
  mkdir -p -- "$HOLDERS_PATH" || die "could not create the holder directory $HOLDERS_PATH"
  printf 'held by %s since %s\n' "$HOLDER" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"$HOLDERS_PATH/$HOLDER" || die "could not write the in-use marker $HOLDERS_PATH/$HOLDER"
  log "in-use marker held by '$HOLDER'; release it with --release --holder $HOLDER"
}

do_sync() {
  require_source
  local source filters files
  source=${S3_PREFIX%/}/$RUN_NAME/
  # Before the mirror directory exists, so that a sibling reaping mirrors while this one downloads is
  # refused as early as this design can refuse it.
  take_marker
  mkdir -p -- "$DEST" || die "could not create the mirror directory $DEST"

  filters=()
  [ "$INCLUDE_LOGS" -eq 1 ] || filters+=(--exclude '*.log')
  # Two patterns, because an aws s3 filter is matched against the whole key with the source prefix
  # prepended: `checkpoint-*` reaches only a checkpoint directory sitting at the run's root, and
  # `*/checkpoint-*` reaches one at any depth below it. `*` spans `/` in these filters, unlike a shell
  # glob, so the pair covers every depth between them.
  if [ "$INCLUDE_CHECKPOINTS" -ne 1 ]; then
    filters+=(--exclude 'checkpoint-*' --exclude '*/checkpoint-*')
    # Order is load-bearing: the filter list is applied in sequence and the last match wins, so these
    # re-include the one file a readout reads out of a checkpoint. Ahead of the excludes they would be
    # overridden by them and the whole flag would do nothing while still reading as present.
    if [ "$INCLUDE_CHECKPOINT_STATE" -eq 1 ]; then
      filters+=(--include 'checkpoint-*/trainer_state.json' --include '*/checkpoint-*/trainer_state.json')
    fi
  fi

  log "syncing $source -> $DEST ${filters[*]:-(no filters)}"
  # Progress and per-object lines go to stderr: stdout carries the mirror path and nothing else.
  "$AWS_CLI" s3 sync "$source" "$DEST" --region "$S3_REGION" --no-progress "${filters[@]}" >&2 \
    || die "the sync of $source failed, so the mirror at $DEST may be incomplete. Fix the cause and
       run this again: an interrupted sync resumes as a delta, and nothing has been deleted."
  [ "$INCLUDE_LOGS" -ne 1 ] || sync_logs
  files=$(find "$DEST" -type f | wc -l) || die "could not count the files under $DEST"
  log "mirrored $files files, $(du -sh "$DEST" | cut -f1)"
  if [ "$files" -eq 0 ] && [ "$ALLOW_EMPTY" -ne 1 ]; then
    die "the sync of $source succeeded and left no files at all in $DEST, so this prints no stage: a
       mistyped run name, a SHIP_S3_PREFIX pointing at the wrong base, a run that never uploaded and
       filters that excluded everything all look like this, and each of them hands a readout an empty
       directory that reads as a real one. Check the source prefix, or pass --allow-empty if an empty
       run really is what you meant. Nothing was deleted and the marker for '$HOLDER' still stands."
  fi
  printf '%s\n' "$DEST"
}

# A run's logs are a sibling of its prefix rather than part of it, so --include-logs is a second sync.
sync_logs() {
  local logs_source logs_dest
  logs_source=${S3_PREFIX%/}/$LOGS_DIR_NAME/$RUN_NAME/
  logs_dest=$DEST/$LOGS_DIR_NAME
  mkdir -p -- "$logs_dest" || die "could not create the log mirror directory $logs_dest"
  log "syncing $logs_source -> $logs_dest"
  "$AWS_CLI" s3 sync "$logs_source" "$logs_dest" --region "$S3_REGION" --no-progress >&2 \
    || die "the sync of $logs_source failed, so the mirror at $DEST is missing logs the run prefix
       does not carry. Run this again: it resumes as a delta and nothing has been deleted."
}

do_status() {
  local holders
  holders=$(held_by) || exit 1
  log "in use by: ${holders:-nobody}"
  [ -d "$DEST" ] || die "$RUN_NAME is not mirrored: $DEST does not exist, so there is no stage to
       print. Sync it with: s3_mirror.sh $RUN_NAME --holder <tag>"
  log "$(find "$DEST" -type f | wc -l) files, $(du -sh "$DEST" | cut -f1)"
  printf '%s\n' "$DEST"
}

do_release() {
  local marker holders
  marker=$HOLDERS_PATH/$HOLDER
  if [ ! -f "$marker" ]; then
    holders=$(held_by) || exit 1
    die "there is no marker for holder '$HOLDER' at $marker, so this released nothing. What holds
       $DEST is: ${holders:-nothing}. A release under a tag nobody took leaves the real claim in
       place until someone reaches for --delete --force, so it is a refusal and not a no-op."
  fi
  rm -f -- "$marker" || die "could not remove the in-use marker $marker"
  log "released the marker held by '$HOLDER'"
  # Tidies the run's holder directory away once the last claim has gone, and does nothing while
  # another holder is still there. That failure is the ordinary case rather than this call's, so the
  # status is set explicitly: left as rmdir's, a release under the overlap this design exists for
  # would exit non-zero and read as a refusal.
  rmdir -- "$HOLDERS_PATH" 2>/dev/null
  return 0
}

do_delete() {
  local holders
  holders=$(held_by) || exit 1
  if [ -n "$holders" ] && [ "$FORCE" -ne 1 ]; then
    die "the mirror $DEST is still in use by: $holders
       Each holder drops its own claim with --release --holder <tag>. To clear a marker whose holder
       is gone, pass --delete --force: this script cannot tell a dead agent's marker from a live
       agent between two reads, so that call is a person's to make."
  fi
  if [ -d "$DEST" ]; then
    rm -rf -- "$DEST" || die "could not delete $DEST"
    log "deleted $DEST"
  else
    log "nothing to delete: $DEST does not exist"
  fi
  if [ -d "$HOLDERS_PATH" ]; then
    rm -rf -- "$HOLDERS_PATH" || die "could not clear the holder directory $HOLDERS_PATH"
  fi
}

validate_run_name
validate_flags_for_mode
case $MODE in
  sync | release) validate_holder ;;
esac
resolve_paths

case $MODE in
  sync) do_sync ;;
  status) do_status ;;
  release) do_release ;;
  delete) do_delete ;;
esac
