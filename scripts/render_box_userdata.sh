#!/bin/bash
# Render a rented box's user-data from a kit-supplied template, attesting to the sitting the staging
# just uploaded and inlining the box-side verification guard.
#
#   render_box_userdata.sh --template <path> --out <path> --stage-dir <dir> [--set NAME=VALUE]...
#
# Two design decisions here are the whole point of the tool, and both are answers to a specific way a
# launcher kit went wrong:
#
#   * EVERY VALUE COMES FROM THE STAGING RECORD in --stage-dir, written by the invocation that
#     actually uploaded the objects and read them back from their consumption keys. Re-deriving the
#     values here would attest to bytes that may never have been staged; and recomputing the tree to
#     compare against the record is the freshness refusal the owner ruled out (2026-08-31) -- a
#     launch must not fail because a peer edited a file after staging. The edit simply ships on the
#     next sitting.
#   * THE EXPECTED VALUES REACH THE BOX ONLY THROUGH USER-DATA, rendered at launch, never from inside
#     the code bundle. A stale bundle carries a stale copy of any check that lives inside it, which is
#     exactly how three of four boxes came to be running four-hour-old code with every sha256 green.
#
# The template must carry all five placeholders below, or the render refuses. A template that quietly
# lost one would produce a user-data pinning the literal placeholder text, and the failure would
# surface hours later on the box as something that looks like a launcher bug.
#
#   @SHIP_TREE@          the working-state git tree hash being shipped
#   @SHIP_CODE_SHA256@   sha256 of the code archive at its consumption key
#   @SHIP_TREE_DIGEST@   git-free digest of the extracted tree
#   @SHIP_S3_KEY@        s3:// URI the box fetches the archive from
#   @SHIP_GUARD@         where scripts/box_ship_guard.sh is inlined
#
# Anything else the kit's template needs comes through repeated --set NAME=VALUE, substituted for
# @NAME@. After substitution the render refuses if ANY @PLACEHOLDER@ survives, which is a strictly
# stronger version of the unfilled-placeholder greps the kits grew one at a time. The converse is
# refused too: a --set whose @NAME@ the template carries nowhere is a launch variable nobody
# receives, and a launch that disarmed a kit's market-price hold that way spent 150 minutes held by
# the pattern it thought it had replaced. Three rules keep substitution from rewriting what the
# author meant, each refused rather than guessed at:
#
#   * comments are never substituted and may not carry an @TOKEN@ at all -- a comment that named the
#     guard placeholder once received the whole guard body, whose lines past the first ran as live
#     code above the assignments it depends on. Name a placeholder in prose without the @ wrapping.
#   * a multi-line value (the guard) is inlined from a placeholder standing alone on its line, once.
#   * scalar values substitute anywhere outside comments, repeated freely.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
GUARD_SOURCE=$HERE/box_ship_guard.sh

# RunInstances rejects user-data above 16,384 RAW bytes -- "User data is limited to 16384 bytes" --
# once per candidate, which during a capacity walk reads exactly like an insufficient-capacity
# decline. The limit binds BEFORE base64: the CLI encodes file:// input itself and EC2 measures the
# decoded bytes ("limited to 16 KB, in raw form, before it is base64-encoded", the EC2 user guide).
# An earlier version checked 25,600 bytes of base64 instead, a figure that is no RunInstances bound
# at all -- 16,384 raw bytes can never encode past 21,848 -- so a user-data between 16,385 and 19,200
# raw bytes passed the render and was then rejected by the API, on a real launch.
USER_DATA_RAW_LIMIT=16384

# What the emitted guard must still contain after being inlined. This is the render's self-check, and
# it exists because the guard is the one component whose failure is invisible: a user-data that fetches
# and extracts but never compares anything produces a box that boots, runs, and reports success.
# Deleting the sha256 line from scripts/box_ship_guard.sh must make the RENDER refuse, not make the
# next box quietly unverified.
REQUIRED_GUARD_PATTERNS=(
  'sha256sum -c -|the archive sha256 comparison'
  'observed.*=.*SHIP_TREE_DIGEST|the extracted-tree digest comparison'
  'halt_unrunnable\(\)|the halt-on-failure function'
  'step_fetch|the fetch step'
  'step_verify_archive|the archive verification step'
  'step_extract|the extract step'
  'step_verify_digest|the digest verification step'
  'no-same-permissions|the mode-normalised extraction (without it a ROOT extraction preserves the
       archive modes the staging digest never saw, and the guard refuses a perfectly good tree)'
)

REQUIRED_PLACEHOLDERS=(SHIP_TREE SHIP_CODE_SHA256 SHIP_TREE_DIGEST SHIP_S3_KEY SHIP_GUARD)

# Where the emitted guard lands on the box, and the heredoc delimiter that carries it there. /var/tmp
# rather than /tmp because /tmp on the dev box is a RAM-backed tmpfs that has run out of inodes.
GUARD_BOX_PATH=/var/tmp/box_ship_guard.sh
GUARD_HEREDOC_MARKER=SHIP_GUARD_BODY_EOF

# Readers return through globals: `die` inside a command substitution exits only the subshell, and the
# caller then prints a second, wrong diagnosis after the real one.
TREE=""
HEAD_SHA=""
DIRTY=""
CODE_SHA256=""
TREE_DIGEST=""
S3_KEY=""
GUARD_BODY=""

die() {
  printf 'FATAL: %s\n' "$1" >&2
  exit 1
}

usage() {
  sed -n '2,29p' "$0" >&2
  exit 2
}

TEMPLATE=""
OUT=""
STAGE_DIR=""
SET_NAMES=()
SET_VALUES=()

parse_arguments() {
  local pair
  while [ $# -gt 0 ]; do
    case $1 in
      --template)
        TEMPLATE=${2:?--template needs a path}
        shift 2
        ;;
      --out)
        OUT=${2:?--out needs a path}
        shift 2
        ;;
      --stage-dir)
        STAGE_DIR=${2:?--stage-dir needs a directory}
        shift 2
        ;;
      --set)
        pair=${2:?--set needs NAME=VALUE}
        case $pair in
          *=*) ;;
          *) die "--set $pair is not in NAME=VALUE form" ;;
        esac
        SET_NAMES+=("${pair%%=*}")
        SET_VALUES+=("${pair#*=}")
        shift 2
        ;;
      -h | --help) usage ;;
      *) die "unknown argument $1 (try --help)" ;;
    esac
  done
  [ -n "$TEMPLATE" ] || die "--template is required"
  [ -n "$OUT" ] || die "--out is required"
  [ -n "$STAGE_DIR" ] || die "--stage-dir is required; it carries the record written by
       stage_ship_tree.sh, which is where the archive's sha256, the tree digest and the S3 key come
       from"
  [ -r "$TEMPLATE" ] || die "the template $TEMPLATE is not readable"
}

# The staging record, not a re-derivation, supplies every value: it was written by the invocation
# that actually uploaded the objects and read them back from their consumption keys. Re-deriving
# them here would be reproducible and still wrong, because it would attest to bytes that may never
# have been staged.
read_record_values() {
  local record=$1
  [ -r "$record" ] || die "no staging record at $record; run stage_ship_tree.sh first"
  TREE=$(sed -n 's/^ship_tree=//p' "$record")
  HEAD_SHA=$(sed -n 's/^head_sha=//p' "$record")
  DIRTY=$(sed -n 's/^dirty=//p' "$record")
  CODE_SHA256=$(sed -n 's/^archive_sha256=//p' "$record")
  TREE_DIGEST=$(sed -n 's/^tree_digest=//p' "$record")
  S3_KEY=$(sed -n 's/^s3_key=//p' "$record")
  [ -n "$TREE" ] || die "the record at $record names no ship_tree"
  [ -n "$HEAD_SHA" ] || die "the record at $record names no head_sha"
  [ -n "$DIRTY" ] || die "the record at $record names no dirty flag"
  [ -n "$CODE_SHA256" ] || die "the record at $record names no archive_sha256"
  [ -n "$TREE_DIGEST" ] || die "the record at $record names no tree_digest"
  [ -n "$S3_KEY" ] || die "the record at $record names no s3_key"
}

read_guard_body() {
  local stripped pattern name
  [ -r "$GUARD_SOURCE" ] || die "the box-side guard $GUARD_SOURCE is missing"
  # Comments and blank lines are dropped on the way in: the tracked file keeps them for review, while
  # the emitted copy shares RunInstances' 16,384-byte raw user-data budget with the kit's own
  # template. The result is parsed before use so a mangled strip cannot reach a box.
  stripped=$(grep -v '^[[:space:]]*#' "$GUARD_SOURCE" | grep -v '^[[:space:]]*$') \
    || die "stripping comments from $GUARD_SOURCE produced nothing"
  printf '%s\n' "$stripped" >"$OUT.guard" || die "cannot write the stripped guard beside $OUT"
  bash -n "$OUT.guard" || die "the stripped copy of $GUARD_SOURCE does not parse; the comment strip
       mangled it, which means the emitted guard would be a syntax error on the box"
  for pattern in "${REQUIRED_GUARD_PATTERNS[@]}"; do
    name=${pattern#*|}
    grep -Eq -e "${pattern%%|*}" "$OUT.guard" \
      || die "the box-side guard no longer contains $name. A guard missing that check would fetch,
       extract and report success without ever comparing anything, so this render refuses rather
       than producing a box that verifies nothing."
  done
  grep -Fqx "$GUARD_HEREDOC_MARKER" "$OUT.guard" \
    && die "the box-side guard contains a line equal to the heredoc delimiter, which would truncate it"
  # Written to a file and run as a child rather than pasted inline. Inline, the guard's own
  # `set -uo pipefail` and its function names (main, die) would leak into the kit's user-data and
  # change how the rest of that script behaves. A quoted heredoc delimiter also keeps the shell from
  # expanding the guard's own $VARIABLES on the way in.
  GUARD_BODY=$(
    printf "cat >%s <<'%s'\n" "$GUARD_BOX_PATH" "$GUARD_HEREDOC_MARKER"
    cat "$OUT.guard"
    printf '%s\n' "$GUARD_HEREDOC_MARKER"
    printf 'export SHIP_TREE SHIP_CODE_SHA256 SHIP_TREE_DIGEST SHIP_S3_KEY\n'
    printf 'export SHIP_S3_REGION SHIP_EXTRACT_DIR\n'
    printf 'bash %s all || exit 1\n' "$GUARD_BOX_PATH"
  )
  rm -f -- "$OUT.guard"
}

require_placeholders() {
  local name
  for name in "${REQUIRED_PLACEHOLDERS[@]}"; do
    grep -Fq "@$name@" "$TEMPLATE" \
      || die "the template $TEMPLATE has no @$name@ placeholder. Rendering it would produce a
       user-data that does not pin what the box must verify, which is indistinguishable on the box
       from having no guard at all."
  done
}

# The mirror of require_placeholders: a --set whose @NAME@ the template carries nowhere is dropped by
# the substitution in silence, and only the rendered file records that it was. That is how a launch
# came to disarm a kit's market-price hold with a --set the template had lost the placeholder for
# (2026-09-05): the render reported clean, and a p5 box six hours of capacity walking had landed then
# sat 150 minutes in the hold the launch believed it had lifted. Counted on live lines only, because
# the substitution never touches a comment, so a token named there receives nothing either. Every one
# of the launcher's own seven values is covered with no exception: the reference kit's template
# consumes all seven, and the closest thing to an optional one, IDLE_WATCHDOG_S3_DEST, is already a
# refusal at the launcher's preflight (its require_pattern for IDLE_WATCHDOG_S3_DEST=s3://), so
# covering it here only moves the refusal earlier and names the cause.
require_set_placeholders() {
  local live index=0 name missing=""
  live=$(sed 's/^[[:space:]]*#.*//' "$TEMPLATE") || die "cannot read the template $TEMPLATE"
  while [ "$index" -lt "${#SET_NAMES[@]}" ]; do
    name=${SET_NAMES[index]}
    case $live in
      *"@$name@"*) ;;
      *) missing="$missing $name" ;;
    esac
    index=$((index + 1))
  done
  [ -z "$missing" ] || die "these --set values name no placeholder on a live line of $TEMPLATE:$missing.
       A --set the template does not carry is substituted nowhere, so the value reaches no box while
       this render still reports clean, and a placeholder named only in a comment is never substituted
       either. Add @NAME@ where the template must consume each value, or drop the --set. The launcher
       passes seven values of its own to every template (scripts/kit_reference/README.md lists them),
       so a kit template has to consume all of them."
}

# Written with python rather than sed because a substituted value can contain slashes and ampersands
# (an s3:// URI contains both), and a sed replacement is not literal.
substitute() {
  local rendered=$1 index=0 pairs=()
  pairs+=("SHIP_TREE=$TREE")
  pairs+=("SHIP_CODE_SHA256=$CODE_SHA256")
  pairs+=("SHIP_TREE_DIGEST=$TREE_DIGEST")
  pairs+=("SHIP_S3_KEY=$S3_KEY")
  while [ "$index" -lt "${#SET_NAMES[@]}" ]; do
    pairs+=("${SET_NAMES[index]}=${SET_VALUES[index]}")
    index=$((index + 1))
  done
  # The guard goes last so a --set value can never contain text that looks like the guard placeholder
  # and get expanded into the guard body.
  pairs+=("SHIP_GUARD=$GUARD_BODY")
  /usr/bin/env python3 - "$TEMPLATE" "$rendered" "${pairs[@]}" <<'PY'
"""Substitute @NAME@ placeholders line-wise, skipping comments, then refuse any that survived.

Comment lines are never substituted, and may not carry an @TOKEN@ at all: a real launch had a
template comment that named the guard placeholder, the render expanded the whole guard body into it,
and everything past the body's first line ran as live code ABOVE the assignments it depends on.
A multi-line value must be inlined from a placeholder standing alone on its line, exactly once --
splicing a body into the middle of a line, or in two places, produces code no template author wrote.
"""
import re
import sys

TOKEN_SHAPE = re.compile(r"@[A-Z][A-Z0-9_]*@")

template_path, out_path = sys.argv[1:3]
values = {}
for pair in sys.argv[3:]:
    name, _, value = pair.partition("=")
    values[name] = value

lines = open(template_path).read().splitlines()
for number, line in enumerate(lines, start=1):
    if not line.lstrip().startswith("#"):
        continue
    for token in TOKEN_SHAPE.findall(line):
        raise SystemExit(
            f"FATAL: {token} appears inside a comment on template line {number}. Comments are "
            "never substituted -- a multi-line value expanded into one becomes live code above "
            "its assignments -- so name it there without the @ wrapping."
        )

multiline_uses = {name: 0 for name, value in values.items() if "\n" in value}
rendered_lines = []
for number, line in enumerate(lines, start=1):
    if line.lstrip().startswith("#"):
        rendered_lines.append(line)
        continue
    for name, value in values.items():
        token = f"@{name}@"
        if token not in line:
            continue
        if name in multiline_uses:
            multiline_uses[name] += line.count(token)
            if line.strip() != token:
                raise SystemExit(
                    f"FATAL: {token} on template line {number} is embedded in other text, but its "
                    "value is a multi-line body that must stand alone on its own line."
                )
            line = value
            break
        line = line.replace(token, value)
    rendered_lines.append(line)

for name, uses in multiline_uses.items():
    if uses > 1:
        raise SystemExit(
            f"FATAL: @{name}@ appears {uses} times, but its value is a multi-line body the "
            "template must inline exactly once."
        )

text = "\n".join(rendered_lines) + "\n"
survivors = sorted(set(TOKEN_SHAPE.findall(text)))
if survivors:
    raise SystemExit(
        "FATAL: the rendered user-data still contains unsubstituted placeholders: "
        + ", ".join(survivors)
        + ". Every one of them would reach the box as literal text."
    )
open(out_path, "w").write(text)
PY
}

check_raw_size() {
  local rendered=$1 raw
  raw=$(wc -c <"$rendered") || die "cannot measure the user-data size"
  printf '   user-data is %s raw bytes of the %s RunInstances allows (%s spare)\n' \
    "$raw" "$USER_DATA_RAW_LIMIT" "$((USER_DATA_RAW_LIMIT - raw))" >&2
  [ "$raw" -le "$USER_DATA_RAW_LIMIT" ] \
    || die "the rendered user-data is $raw raw bytes, over the $USER_DATA_RAW_LIMIT-byte RunInstances
       limit ('User data is limited to 16384 bytes', measured on the raw bytes before base64).
       Every candidate in a capacity walk would be rejected, and the rejection reads exactly like
       an insufficient-capacity decline."
}

main() {
  parse_arguments "$@"
  read_record_values "$STAGE_DIR/ship-manifest.txt"
  require_placeholders
  require_set_placeholders
  read_guard_body
  substitute "$OUT" || die "rendering $TEMPLATE into $OUT failed"
  [ -s "$OUT" ] || die "the render produced an empty file at $OUT"
  bash -n "$OUT" || die "the rendered user-data does not parse as bash"
  grep -Fq "bash $GUARD_BOX_PATH all" "$OUT" \
    || die "the rendered user-data never RUNS the guard it carries. A guard that is written to disk and
       never invoked is the same as no guard, and the box would report a clean bootstrap."
  check_raw_size "$OUT"
  printf 'ship_tree=%s\n' "$TREE"
  printf 'head_sha=%s\n' "$HEAD_SHA"
  printf 'dirty=%s\n' "$DIRTY"
  printf 's3_key=%s\n' "$S3_KEY"
  printf 'user_data=%s\n' "$OUT"
}

main "$@"
