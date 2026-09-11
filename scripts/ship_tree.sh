#!/bin/bash
# Content-address the repository's WORKING STATE, so that what a rented box runs can be named by a
# hash rather than asserted by a human.
#
# The problem this exists for: every launcher kit built its code bundle with `git archive HEAD`,
# which silently drops uncommitted work, and staged that bundle once per run under a key later
# launches re-fetched. Three of four boxes running on 2026-08-27 were therefore executing code hours
# older than the tree their operator was looking at, with every sha256 check green, because integrity
# ("the bytes I staged are the bytes you ran") is a different property from freshness ("the bytes I
# staged are the bytes I have").
#
# The primitive: build the tree git WOULD commit if everything on disk were staged -- tracked files
# as they are on disk, plus untracked files git does not ignore -- inside a PRIVATE index, and take
# its git tree hash. Properties that make this the right primitive:
#
#   * it covers dirty and untracked code, which `git archive HEAD` cannot;
#   * it ignores exactly what .gitignore ignores, so artifacts/ and docs/scratch/ churn does not move
#     the hash and staging stays idempotent;
#   * it needs no commit, so nothing writes a peer session's half-finished files into shared history;
#   * the shared .git/index is left byte-identical, which is what makes it legal in a tree three to
#     five concurrent agent sessions are committing to.
#
# `git archive` of a bare tree is NOT byte-reproducible: entry mtimes default to now, so the same tree
# archived twice a second apart gives two sha256s. `git archive` of a fixed-date commit over that
# tree is reproducible, its commit id is stable, and the tarball's pax global header carries
# `comment=<commit id>`, so the archive self-identifies. That commit is kept at refs/ship/<hash> so
# the objects survive gc and any past shipment can be reproduced months later.
#
#   ship_tree.sh --print-hash            # the working-state tree hash, alone, for $(...)
#   ship_tree.sh --facts                 # tree=, head=, dirty= in one pass
#   ship_tree.sh --freeze                # create the fixed-date ship commit and refs/ship/<hash>
#   ship_tree.sh --archive <out.tar.gz>  # freeze, then materialise reproducibly
#   ship_tree.sh --digest <dir>          # the git-free digest of an extracted tree
#   ship_tree.sh --provenance            # the provenance block alone
#
# `--tree <hash>` makes any mode operate on a previously frozen tree instead of the working state,
# and requires refs/ship/<hash> to exist locally.
#
# Every mode writes its provenance block to STDERR and only machine-readable values to STDOUT, so a
# caller can capture a hash without also capturing prose.
set -uo pipefail

# A tree of exactly nothing. `git write-tree` returns it when the index is empty, which is what a
# broken file-listing pipeline looks like from the outside -- a plausible 40-hex hash that would
# archive to a valid, empty tarball and ship a box no code at all. Refused by name.
EMPTY_TREE=4b825dc642cb6eb9a060e54bf8d69288fbee4904

# sha256 of empty input. The digest of a directory with no files would come out as this, and two
# boxes with no code at all would then agree with each other. Produced by accident once while
# auditing this exact flow, which is why it is refused by name too.
SHA256_OF_NOTHING=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

# A ship commit must be a pure function of its tree, or re-freezing the same working state would mint
# a new commit id and the archive would stop being reproducible. So every input git hashes into a
# commit is pinned: no author, date or message may come from the environment. The identity is
# deliberately impersonal -- this repo is public, and a ship commit is machinery, not authorship.
SHIP_COMMIT_DATE='2000-01-01T00:00:00 +0000'
SHIP_COMMIT_NAME='ship-tree'
SHIP_COMMIT_EMAIL='ship-tree@invalid'

# Every function below that can refuse returns its answer in one of these globals, and is called as a
# plain statement rather than as `x=$(reader)`. That is not style. `die` inside a command substitution
# exits only the substitution's subshell, so the caller carries on with an empty string and prints a
# SECOND, wrong diagnosis after the real one -- watched happen here on the first draft of this file.
REPO=""
SHIP_INDEX=""
SHIP_TREE=""
SHIP_COMMIT=""
SHIP_ARCHIVE_SHA256=""
SHIP_DIGEST=""

die() {
  printf 'FATAL: %s\n' "$1" >&2
  exit 1
}

usage() {
  sed -n '2,40p' "$0" >&2
  exit 2
}

# --- repository and private index ----------------------------------------------------------------

# The repo is the one this script lives in, never the caller's cwd: the launcher runs from wherever
# the operator happens to be, and a tool that content-addressed whatever directory it was invoked
# from would ship whichever repo the shell was sitting in.
resolve_repo() {
  local here
  here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) \
    || die "cannot resolve this script's directory"
  REPO=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
    || die "$here is not inside a git repository, so there is no working state to content-address"
}

# The one catastrophic footgun in this design. `git add` against the SHARED index would stage every
# peer session's in-flight edit, and one of those sessions deliberately holds broken intermediates
# while it watches a gate fail. So the index this script writes must be somewhere disposable, and any
# request to point it at the repository is refused rather than honoured.
resolve_index() {
  local gitdir wanted parent
  gitdir=$(git -C "$REPO" rev-parse --absolute-git-dir) || die "cannot resolve the git directory"
  wanted=${GIT_INDEX_FILE:-}
  if [ -z "$wanted" ]; then
    wanted=/var/tmp/ship-tree-index-$$-$(printf '%s' "$REPO" | sha256sum | cut -c1-12).idx
  fi
  case $wanted in
    /*) ;;
    *) die "GIT_INDEX_FILE=$wanted is relative; it must be an absolute path outside the repository" ;;
  esac
  # Prefix comparison rather than realpath: the index file itself need not exist yet, and both
  # candidates are already absolute. The trailing slash keeps /var/tmp/repo-ish from matching
  # /var/tmp/repo.
  case "$wanted/" in
    "$gitdir"/* | "$REPO"/*)
      die "refusing to use $wanted as the ship index: it is inside the repository ($REPO), and
       staging into a tree three to five concurrent sessions share would destroy their in-flight
       work. Leave GIT_INDEX_FILE unset and this script picks a disposable path under /var/tmp."
      ;;
  esac
  parent=$(dirname -- "$wanted")
  [ -d "$parent" ] || die "the ship index's directory $parent does not exist"
  [ -w "$parent" ] || die "the ship index's directory $parent is not writable"
  SHIP_INDEX=$wanted
}

# --- the working-state tree ----------------------------------------------------------------------

# `git add -A` would be the obvious way to fill the index, and is what this was prototyped with. The
# explicit listing is used instead for two reasons: it makes the shipped file set auditable, and a
# blanket add is the shape that has swept a peer's orphaned staged deletions into a commit in this
# repo before. `--cached` carries deletions too: a tracked file removed from disk is staged as
# removed, so the ship tree omits it, which is what the operator sees.
#
# --exclude-standard also honours a global core.excludesFile, so a ship tree is only reproducible
# across machines whose global excludes agree. Accepted rather than fought: .gitignore semantics are
# the whole point of the primitive, and the alternative is re-implementing them.
read_working_state_tree() {
  rm -f -- "$SHIP_INDEX"
  GIT_INDEX_FILE=$SHIP_INDEX git -C "$REPO" read-tree HEAD \
    || die "git read-tree HEAD failed; this repository has no HEAD commit to build a ship tree on"
  GIT_INDEX_FILE=$SHIP_INDEX git -C "$REPO" ls-files -z --cached --others --exclude-standard \
    | GIT_INDEX_FILE=$SHIP_INDEX git -C "$REPO" add --pathspec-from-file=- --pathspec-file-nul -- \
    || die "listing the working state into the ship index failed; refusing to hash a partial index"
  SHIP_TREE=$(GIT_INDEX_FILE=$SHIP_INDEX git -C "$REPO" write-tree) \
    || die "git write-tree failed on the ship index"
  rm -f -- "$SHIP_INDEX"
  validate_tree "$SHIP_TREE"
}

validate_tree() {
  local tree=$1
  case $tree in
    "$EMPTY_TREE")
      die "the ship tree is git's empty tree; the working state listed no files at all"
      ;;
    '' | *[!0-9a-f]*) die "the ship tree ${tree:-<empty>} is not a hex object id" ;;
  esac
  case ${#tree} in
    40 | 64) ;;
    *) die "the ship tree $tree is ${#tree} characters, not a 40- or 64-character object id" ;;
  esac
}

# A tree named on the command line is only usable if this repo can still produce its bytes, which is
# exactly what refs/ship/<hash> guarantees and a bare hash does not.
read_frozen_tree() {
  local tree=$1 have
  validate_tree "$tree"
  have=$(git -C "$REPO" rev-parse --verify --quiet "refs/ship/$tree^{tree}") \
    || die "refs/ship/$tree does not exist in this repository, so tree $tree cannot be reproduced here.
       A tree is only replayable on the machine that froze it."
  [ "$have" = "$tree" ] || die "refs/ship/$tree points at tree $have, not $tree"
  SHIP_TREE=$tree
}

# --- freeze and archive --------------------------------------------------------------------------

freeze_tree() {
  local existing
  SHIP_COMMIT=$(
    GIT_AUTHOR_DATE=$SHIP_COMMIT_DATE GIT_COMMITTER_DATE=$SHIP_COMMIT_DATE \
      GIT_AUTHOR_NAME=$SHIP_COMMIT_NAME GIT_COMMITTER_NAME=$SHIP_COMMIT_NAME \
      GIT_AUTHOR_EMAIL=$SHIP_COMMIT_EMAIL GIT_COMMITTER_EMAIL=$SHIP_COMMIT_EMAIL \
      git -C "$REPO" commit-tree "$SHIP_TREE" -m "ship tree $SHIP_TREE"
  ) || die "git commit-tree failed for tree $SHIP_TREE"
  existing=$(git -C "$REPO" rev-parse --verify --quiet "refs/ship/$SHIP_TREE") || existing=""
  if [ -n "$existing" ] && [ "$existing" != "$SHIP_COMMIT" ]; then
    die "refs/ship/$SHIP_TREE already points at $existing, but this tree freezes to $SHIP_COMMIT.
       A ship commit is a pure function of its tree, so the two cannot both be right."
  fi
  git -C "$REPO" update-ref "refs/ship/$SHIP_TREE" "$SHIP_COMMIT" \
    || die "could not write refs/ship/$SHIP_TREE"
}

archive_tree() {
  local out=$1 parent
  parent=$(dirname -- "$out")
  [ -d "$parent" ] || die "the archive's directory $parent does not exist"
  # tar.umask is pinned rather than inherited: git's default (002) records group-writable modes, and
  # any machine- or repo-level override would change the archive bytes, so the same tree could
  # content-address to two sha256s on two machines. 0022 records 644/755, the same modes both digest
  # paths normalise their extractions to (read_archive_digest in stage_ship_tree.sh, step_extract in
  # box_ship_guard.sh).
  git -C "$REPO" -c tar.umask=0022 archive --format=tar.gz "$SHIP_COMMIT" -o "$out" \
    || die "git archive failed for ship commit $SHIP_COMMIT"
  [ -s "$out" ] || die "git archive wrote an empty file to $out"
  SHIP_ARCHIVE_SHA256=$(sha256sum -- "$out" | cut -d' ' -f1) \
    || die "could not sha256 the archive at $out"
}

# --- the git-free digest -------------------------------------------------------------------------

# A second, independent check on what actually landed on the box, computed without git so it can run
# on a bare GPU AMI before any venv exists. The archive's sha256 catches a swapped or corrupted
# object; this catches a bad extraction and a leftover file from a previous unpack, which a correct
# sha256 says nothing about.
#
# Mode, size and content per regular file; the link target for a symlink, because this repo ships one
# (CLAUDE.md -> AGENTS.md) and following it would hash the target's bytes twice and miss a swapped
# link entirely. NUL-separated throughout so a newline in a path cannot forge a record.
#
# DELIBERATELY DUPLICATED in scripts/box_ship_guard.sh, which is what actually runs on the box: a box
# must not verify itself with a verifier it just downloaded, because a stale bundle carries a stale
# verifier. The two are pinned to each other by tests/test_ship_tree.py, which digests one tree with
# both and requires the answers to be equal.
read_tree_digest() {
  local root=$1 listing count
  [ -d "$root" ] || die "cannot digest $root: not a directory"
  listing=$(mktemp /var/tmp/ship-tree-digest-XXXXXX) || die "cannot create a listing file"
  find "$root" -mindepth 1 \( -type f -o -type l \) -printf '%P\0' \
    | LC_ALL=C sort -z >"$listing" || die "listing $root failed"
  count=$(tr -cd '\0' <"$listing" | wc -c)
  if [ "$count" -eq 0 ]; then
    rm -f -- "$listing"
    die "$root contains no files, so there is nothing to digest; refusing to report the digest of
       empty input, which is a valid-looking hash that any other empty tree also has"
  fi
  SHIP_DIGEST=$(
    cd -- "$root" || exit 1
    while IFS= read -r -d '' path; do
      if [ -L "$path" ]; then
        printf '%s\0L\0%s\0' "$path" "$(readlink -- "$path")"
      else
        printf '%s\0F\0%s\0%s\0%s\0' "$path" "$(stat -c '%a' -- "$path")" \
          "$(stat -c '%s' -- "$path")" "$(sha256sum -- "$path" | cut -d' ' -f1)"
      fi
    done <"$listing" | sha256sum | cut -d' ' -f1
  ) || die "digesting $root failed"
  rm -f -- "$listing"
  [ "$SHIP_DIGEST" != "$SHA256_OF_NOTHING" ] \
    || die "the digest of $root came out as sha256 of empty input; the record stream was empty"
}

# --- provenance ----------------------------------------------------------------------------------

# Derived from the ship tree itself rather than from a parallel `git status`, so the report cannot
# disagree with what ships. Diffing HEAD against the ship tree also renders untracked files as plain
# additions, which is what they are from the box's point of view.
provenance() {
  local head changed
  head=$(git -C "$REPO" rev-parse HEAD) || die "cannot resolve HEAD"
  changed=$(git -C "$REPO" diff --name-status HEAD "$SHIP_TREE") \
    || die "cannot diff HEAD against $SHIP_TREE"
  {
    printf '== ship tree %s\n' "$SHIP_TREE"
    printf '   repo      %s\n' "$REPO"
    printf '   HEAD      %s\n' "$head"
    if [ -z "$changed" ]; then
      printf '   state     clean: the ship tree is exactly HEAD\n'
    else
      printf '   state     DIRTY: %s path(s) differ from HEAD, and all of them ship\n' \
        "$(printf '%s\n' "$changed" | wc -l)"
      printf '%s\n' "$changed" | sed 's/^/     /'
      git -C "$REPO" diff --stat HEAD "$SHIP_TREE" | sed 's/^/     /'
    fi
  } >&2
}

# The three values every caller downstream needs, from the one pass already made over the working
# state. Emitted as key=value lines so a caller greps for what it wants rather than depending on
# field order.
facts() {
  local head dirty
  head=$(git -C "$REPO" rev-parse HEAD) || die "cannot resolve HEAD"
  if [ -n "$(git -C "$REPO" diff --name-only HEAD "$SHIP_TREE")" ]; then dirty=true; else dirty=false; fi
  printf 'tree=%s\nhead=%s\ndirty=%s\n' "$SHIP_TREE" "$head" "$dirty"
}

# --- entry point ---------------------------------------------------------------------------------

main() {
  local mode="" out="" digest_dir="" requested_tree=""
  while [ $# -gt 0 ]; do
    case $1 in
      --print-hash | --freeze | --provenance | --facts)
        [ -z "$mode" ] || die "modes --$mode and ${1#--} are mutually exclusive"
        mode=${1#--}
        shift
        ;;
      --archive)
        [ -z "$mode" ] || die "modes --$mode and archive are mutually exclusive"
        mode=archive
        out=${2:?--archive needs an output path}
        shift 2
        ;;
      --digest)
        [ -z "$mode" ] || die "modes --$mode and digest are mutually exclusive"
        mode=digest
        digest_dir=${2:?--digest needs a directory}
        shift 2
        ;;
      --tree)
        requested_tree=${2:?--tree needs a tree hash}
        shift 2
        ;;
      -h | --help) usage ;;
      *) die "unknown argument $1 (try --help)" ;;
    esac
  done
  [ -n "$mode" ] || usage

  if [ "$mode" = digest ]; then
    read_tree_digest "$digest_dir"
    printf '%s\n' "$SHIP_DIGEST"
    return 0
  fi

  resolve_repo
  if [ -n "$requested_tree" ]; then
    read_frozen_tree "$requested_tree"
  else
    resolve_index
    read_working_state_tree
  fi

  case $mode in
    print-hash)
      provenance
      printf '%s\n' "$SHIP_TREE"
      ;;
    provenance) provenance ;;
    facts)
      provenance
      facts
      ;;
    freeze)
      provenance
      freeze_tree
      facts
      printf 'commit=%s\n' "$SHIP_COMMIT"
      ;;
    archive)
      provenance
      freeze_tree
      archive_tree "$out"
      facts
      printf 'commit=%s\narchive_sha256=%s\n' "$SHIP_COMMIT" "$SHIP_ARCHIVE_SHA256"
      ;;
  esac
}

main "$@"
