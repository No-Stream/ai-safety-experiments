#!/bin/bash
# The one supported way to rent a GPU box for this repository.
#
#   launch_gpu_box.sh --run-name <name> --template <ud-template> --candidates <file> \
#     --label <name> --purpose <text> --deadman-minutes <n> [--walk-until <minutes>] [options]
#
# It does stage -> render -> run-instances in one flow. The reason it is one tool rather than a
# documented sequence: the guarantees in the hand-written launcher kits DECAYED across generations --
# one generation's launch check compared a hardcoded constant against a copy of itself, so it passed
# forever. Mechanical checks survive copy-paste. Per-run re-derived ones do not.
#
# TWO RULES GOVERN THIS TOOL (owner ruling, 2026-08-31):
#
#   * RUN IDENTITY IS AN EXPLICIT NAME. Every S3 path this tool constructs lives under
#     <s3-prefix>/<run-name>/, and no path carries a commit, tag, or hash component. A relaunch that
#     passes the same name lands in the same prefix by construction. The previous design derived the
#     prefix from `git rev-parse HEAD`; a peer commit moved it between a run's launch and its
#     post-reclaim relaunch, and the relaunch started a fresh run into a junk prefix instead of
#     resuming.
#   * THE CURRENT WORKING TREE SHIPS, DIRTY AND UNTRACKED CODE INCLUDED, ALWAYS. Nothing refuses a
#     launch because the tree is dirty or changed since some earlier moment -- with several sessions
#     sharing this tree, that is expected, welcome behavior, and a multi-sitting run deliberately
#     ships newer code on later sittings. Provenance is a RECORD, never a gate: each sitting appends
#     its own record (HEAD sha, git status, tarball checksum) under <run-prefix>/provenance/, and the
#     box carries ShipTree/ShipHeadSha/ShipDirty tags, so forensics can always say what bytes ran.
#
# What still FAILS CLOSED is integrity and safety, which no peer edit can trip: staging refuses when
# an object does not read back identically from its consumption key, the render refuses a template
# missing a placeholder or a guard check, the launch refuses a user-data naming a key this invocation
# never staged, and the box halts on a short fuse when the fetched archive or extracted tree does not
# match what this launch staged. The mandatory protections (boot patch, both killswitch layers,
# IMDSv2, shutdown-behaviour terminate) are also verified before anything is rented.
#
# Account-specific values are NEVER tracked in this repository. They arrive as arguments or
# environment variables: SHIP_S3_PREFIX, SHIP_S3_REGION, SHIP_INSTANCE_PROFILE_NAME, and the candidate
# file, which carries the subnet and security-group ids one row per line. Which account a launch would
# rent in is reported rather than assumed: before the first paid call the launcher asks sts what the
# ambient credentials resolve to, prints it, and refuses when they cannot be read at all. Name the
# account you mean with --expect-account (or SHIP_EXPECTED_ACCOUNT) and a mismatch is refused there
# too, before anything is staged; without it the walk's NotFound anchor is what catches a launch aimed
# at the wrong account, one row in rather than up front.
#
# The candidate file is the walk, in order, one row per line:
#
#   instance_type|region|security_group_id|subnet_id
#
# ROW ORDER IS THE WIDENING ORDER, and on demand exhausts every row before a single spot row is tried
# (the default; --market spot-first reverses the two passes, for jobs whose work is checkpointed
# finely enough that a reclaim costs less than the on-demand premium -- it still requires --allow-spot
# and its stated reason). Per AGENTS.md: widen first by instance size (g7e.4xlarge and g7e.8xlarge
# are the same single card as the 2xlarge and buy only host vCPU and RAM), then by availability zone
# one row per subnet, then by region out to every region enabled on the account. A smaller card is a
# tier drop rather than a widening and does not belong in the file at all.
#
# --walk-until <minutes> repeats the whole pass -- a fresh sitting each time, so the tree as it stands
# at each pass ships -- with --walk-pause-minutes (default 5) between passes, until a box lands or the
# deadline passes, and then hands back the same exhaustion evidence a single walk does. No pass begins
# after the deadline (the last pause is shortened to end on it), so the wait overruns it by at most one
# pass. It never widens the walk on its own: a p5 pass is a different candidate file and spot is
# --allow-spot with its stated reason, both separate, explicit decisions, so the flag refuses to
# combine with --allow-spot. It replaces the hand-rolled loops that re-issued a walk every eight to
# fifteen minutes.
#
# Every declined row costs one RunInstances call, not three: the CLI's own retry layer is off for the
# walk (AWS_MAX_ATTEMPTS=1), and only a transient answer -- a throttle, a server-side failure, a
# connection failure -- is retried, explicitly and boundedly, on the same row under one client token,
# so a retry after an answer that never arrived can never launch a second box. The idle watchdog's
# --done-marker is checked at preflight as a WARNING, never a refusal: a template without it bills a
# 40-45 minute idle tail per box after its agenda finishes (2,400 s measured on one box, 2026-09-01),
# because the idle threshold is then the watchdog's only trigger. The same tier compares the template
# (and the runner it names inside the staged tree) against every rule of the tracked reference kit,
# scripts/kit_reference/markers.tsv, one WARNING per rule the kit dropped.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
STAGE_TOOL=$HERE/stage_ship_tree.sh
RENDER_TOOL=$HERE/render_box_userdata.sh
SELF=${BASH_SOURCE[0]}
AWS_CLI=${AWS_CLI:-aws}

# Not account-specific, so it can carry a tracked default: the GPU base AMI family, resolved
# freshest-first per region (see query_ami). Every rented box patches its own OS at boot, which is
# what closes the vulnerability window here, because the newest available build of this image has
# itself shipped a vulnerable package.
DEFAULT_AMI_NAME_PATTERN='Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) *'
DEFAULT_VOLUME_SIZE_GB=300

# A decline the walk should step past, versus an error that would fail identically on every remaining
# candidate. Without this split a malformed argument reads exactly like a region-wide capacity
# shortage, and the walk burns through forty-five rows before saying anything useful. Anchored on the
# CLI's parsed error code, never on a phrase from the message: an earlier version also matched the
# substring "does not exist", and under the wrong account's credentials every candidate subnet
# answers InvalidSubnetID.NotFound with exactly those words, so a whole walk logged as declined and
# closed with the line about the nature of GPU supply. A NotFound of any kind now stops the walk as
# misconfiguration, a stale row included.
CAPACITY_DECLINES='An error occurred \((InsufficientInstanceCapacity|InsufficientHostCapacity|Unsupported|SpotMaxPriceTooLow|MaxSpotInstanceCountExceeded|InstanceLimitExceeded|ServiceUnavailable|Unavailable|VcpuLimitExceeded)\)'

TREE=""
HEAD_SHA=""
DIRTY=""
STAGED_KEY=""
PROVENANCE_KEY=""
STAGE_DIR=""
USER_DATA=""
TAGSPEC=""
AMI_REGIONS=()
AMI_IDS=()
LAUNCH_INSTANCE=""
LAUNCH_REGION=""
LAUNCH_MARKET=""

die() {
  printf 'FATAL: %s\n' "$1" >&2
  exit 1
}

usage() {
  # Up to the set line rather than to a line number, so a header edit cannot leave the usage text stale.
  sed -n '2,/^set -uo pipefail$/{ /^set -uo pipefail$/!p; }' "$0" >&2
  exit 2
}

# --- arguments -----------------------------------------------------------------------------------

TEMPLATE=""
CANDIDATES=${SHIP_CANDIDATES:-}
RUN_NAME=""
LABEL=""
PURPOSE=""
PROJECT="rented-gpu"
DEADMAN_MINUTES=""
S3_PREFIX=${SHIP_S3_PREFIX:-}
S3_REGION=${SHIP_S3_REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-}}}
INSTANCE_PROFILE=${SHIP_INSTANCE_PROFILE_NAME:-}
# Empty means the account gate reports the identity without checking it; no default, because an
# account id may never sit in tracked code and a value nobody supplied would only turn the check off
# while leaving it looking armed.
EXPECTED_ACCOUNT=${SHIP_EXPECTED_ACCOUNT:-}
AMI_NAME_PATTERN=${SHIP_AMI_NAME_PATTERN:-$DEFAULT_AMI_NAME_PATTERN}
VOLUME_SIZE_GB=$DEFAULT_VOLUME_SIZE_GB
ALLOW_SPOT=0
SPOT_REASON=""
SPOT_MAX_PRICE=""
MARKET_ORDER="ondemand spot"
PREFLIGHT_ONLY=0
DRY_RUN_ONLY=0
# Empty means one pass, today's behaviour exactly; a number is the deadline in minutes from launch.
WALK_UNTIL_MINUTES=""
WALK_PAUSE_MINUTES=5
DATA_ARGS=()
SET_ARGS=()
RUN_DIR=""
RUN_PREFIX=""
SITTING=""

# The name is the run's whole identity, so its shape is held to what survives an S3 key, an EC2 tag
# and a report unmangled. The commit-sha-shaped-token refusal is the owner's no-pins ruling made
# mechanical: a name like wave3-cbc3094f smuggles the sha right back into the prefix, and the
# letter requirement in the token test is what keeps a plain date like 20260831 from being read as
# a hash while still catching everything `git rev-parse` actually emits.
validate_run_name() {
  [ -n "$RUN_NAME" ] || die "--run-name is required: run identity is an explicit name, and every
       relaunch that passes the same name lands in the same S3 prefix by construction. Name the run
       by purpose (wave3-reskin), never by commit."
  [ "${#RUN_NAME}" -le 64 ] || die "--run-name $RUN_NAME is ${#RUN_NAME} characters; keep it to 64"
  case $RUN_NAME in
    [!A-Za-z0-9]* | *[!A-Za-z0-9._-]*)
      die "--run-name $RUN_NAME must start with an alphanumeric and use only letters, digits and
       [._-]; anything else (slashes especially) lands verbatim in every S3 key this launch writes"
      ;;
  esac
  if printf '%s\n' "$RUN_NAME" | tr '._-' '\n' | grep -E '^[0-9a-f]{7,64}$' | grep -q '[a-f]'; then
    die "--run-name $RUN_NAME contains a commit-sha-shaped token. Run identity must be a name, never
       a pin (owner ruling, 2026-08-31): embedding a sha re-ties the run's prefix to a code state,
       which is exactly what stranded a relaunch in a junk prefix. Name the run by purpose instead."
  fi
}

parse_arguments() {
  while [ $# -gt 0 ]; do
    case $1 in
      --run-name)
        RUN_NAME=${2:?--run-name needs a name}
        shift 2
        ;;
      --template)
        TEMPLATE=${2:?--template needs a path}
        shift 2
        ;;
      --candidates)
        CANDIDATES=${2:?--candidates needs a path}
        shift 2
        ;;
      --label)
        LABEL=${2:?--label needs a name}
        shift 2
        ;;
      --purpose)
        PURPOSE=${2:?--purpose needs text}
        shift 2
        ;;
      --project)
        PROJECT=${2:?--project needs a name}
        shift 2
        ;;
      --deadman-minutes)
        DEADMAN_MINUTES=${2:?--deadman-minutes needs a number}
        shift 2
        ;;
      --s3-prefix)
        S3_PREFIX=${2:?--s3-prefix needs an s3:// URI}
        shift 2
        ;;
      --s3-region)
        S3_REGION=${2:?--s3-region needs a region}
        shift 2
        ;;
      --instance-profile)
        INSTANCE_PROFILE=${2:?--instance-profile needs a name}
        shift 2
        ;;
      --expect-account)
        EXPECTED_ACCOUNT=${2:?--expect-account needs a twelve-digit AWS account id}
        shift 2
        ;;
      --ami-name-pattern)
        AMI_NAME_PATTERN=${2:?--ami-name-pattern needs a glob}
        shift 2
        ;;
      --volume-size-gb)
        VOLUME_SIZE_GB=${2:?--volume-size-gb needs a number}
        shift 2
        ;;
      --data)
        DATA_ARGS+=(--data "${2:?--data needs a path}")
        shift 2
        ;;
      --set)
        SET_ARGS+=(--set "${2:?--set needs NAME=VALUE}")
        # The names stage_and_render fills from the launch itself; a kit's own value for one of them
        # would let the heartbeat key, the HeartbeatS3 tag and the watchdog destination disagree.
        case ${2%%=*} in
          SHIP_S3_REGION | DEADMAN_MINUTES | IDLE_WATCHDOG_S3_DEST | RUN_NAME | RUN_PREFIX | SHIP_S3_PREFIX | SHIP_HEAD_SHA)
            die "--set ${2%%=*} names a placeholder this launcher fills itself, from --s3-region, --deadman-minutes, --run-name and the staging record; drop it (scripts/kit_reference/README.md lists which placeholders a kit owns)"
            ;;
        esac
        shift 2
        ;;
      --allow-spot)
        ALLOW_SPOT=1
        shift
        ;;
      --market)
        case ${2:?--market needs ondemand-first or spot-first} in
          ondemand-first) MARKET_ORDER="ondemand spot" ;;
          spot-first) MARKET_ORDER="spot ondemand" ;;
          *) die "--market accepts ondemand-first or spot-first, not $2" ;;
        esac
        shift 2
        ;;
      --spot-reason)
        SPOT_REASON=${2:?--spot-reason needs text}
        shift 2
        ;;
      --spot-max-price)
        SPOT_MAX_PRICE=${2:?--spot-max-price needs a price}
        shift 2
        ;;
      --run-dir)
        RUN_DIR=${2:?--run-dir needs a directory}
        shift 2
        ;;
      --preflight-only)
        PREFLIGHT_ONLY=1
        shift
        ;;
      --dry-run-only)
        DRY_RUN_ONLY=1
        shift
        ;;
      --walk-until)
        WALK_UNTIL_MINUTES=${2:?--walk-until needs a number of minutes}
        shift 2
        ;;
      --walk-pause-minutes)
        WALK_PAUSE_MINUTES=${2:?--walk-pause-minutes needs a number of minutes}
        shift 2
        ;;
      -h | --help) usage ;;
      *) die "unknown argument $1 (try --help)" ;;
    esac
  done
  validate_run_name
  [ -n "$TEMPLATE" ] || die "--template is required"
  [ -n "$CANDIDATES" ] || die "--candidates is required (or set SHIP_CANDIDATES). It carries the
       subnet and security-group ids, which are account-specific and must not be tracked here."
  [ -n "$LABEL" ] || die "--label is required; it names the box in its Name and DoNotReap tags"
  [ -n "$PURPOSE" ] || die "--purpose is required; an unexplained GPU box is the one nobody dares kill"
  [ -n "$DEADMAN_MINUTES" ] || die "--deadman-minutes is required: every rented box arms a max-lifetime
       dead-man switch at launch, and no box may depend on a human remembering to terminate it"
  [ -n "$S3_PREFIX" ] || die "--s3-prefix is required (or set SHIP_S3_PREFIX)"
  [ -n "$S3_REGION" ] || die "--s3-region is required (or set SHIP_S3_REGION / AWS_REGION)"
  [ -n "$INSTANCE_PROFILE" ] || die "--instance-profile is required (or set
       SHIP_INSTANCE_PROFILE_NAME); without it the box cannot reach S3 or terminate itself"
  [ -r "$TEMPLATE" ] || die "the template $TEMPLATE is not readable"
  [ -r "$CANDIDATES" ] || die "the candidate file $CANDIDATES is not readable"
  case $DEADMAN_MINUTES in
    '' | *[!0-9]*) die "--deadman-minutes must be a whole number of minutes, not $DEADMAN_MINUTES" ;;
  esac
  case $VOLUME_SIZE_GB in
    '' | *[!0-9]*) die "--volume-size-gb must be a whole number, not $VOLUME_SIZE_GB" ;;
  esac
  # A malformed id is refused rather than ignored: the same value that arms the account check can
  # silence it, and a typo that read as "no expected account" would leave the check looking armed.
  if [ -n "$EXPECTED_ACCOUNT" ] && ! [[ $EXPECTED_ACCOUNT =~ ^[0-9]{12}$ ]]; then
    die "--expect-account (or SHIP_EXPECTED_ACCOUNT) is '$EXPECTED_ACCOUNT', which is not a
       twelve-digit AWS account id. Fix it, or leave it unset to launch without the check."
  fi
  if [ "$ALLOW_SPOT" = 1 ] && [ -z "$SPOT_REASON" ]; then
    die "--allow-spot requires --spot-reason. Spot is a last resort here because a reclaim costs more
       than the step it interrupts: one arm measured ~28 minutes per step against a worst observed
       ~35-minute reclaim cadence, so a lease buys about one step, and a checkpoint interval longer
       than the lease trains forever and saves never while every signal stays green. Name the reason
       so it lands in the run's report and in CloudTrail."
  fi
  if [ "$MARKET_ORDER" = "spot ondemand" ] && [ "$ALLOW_SPOT" != 1 ]; then
    die "--market spot-first without --allow-spot is a contradiction: the reordered first pass would
       be skipped entirely and the flag would silently mean nothing. Pass --allow-spot with its
       --spot-reason, which is where the interruption-tolerance argument belongs."
  fi
  case $WALK_UNTIL_MINUTES in
    '') ;;
    *[!0-9]*) die "--walk-until must be a whole number of minutes, not $WALK_UNTIL_MINUTES" ;;
  esac
  case $WALK_PAUSE_MINUTES in
    '' | *[!0-9]*) die "--walk-pause-minutes must be a whole number of minutes, not $WALK_PAUSE_MINUTES" ;;
  esac
  if [ -n "$WALK_UNTIL_MINUTES" ]; then
    # The wait is for ON-DEMAND capacity and nothing else. Spot after a wait is a decision made with
    # the exhaustion evidence in hand (owner rules), not one a loop takes at whichever pass a spot row
    # happens to answer first; the same goes for a p5 pass, which is a different candidate file.
    [ "$ALLOW_SPOT" != 1 ] || die "--walk-until with --allow-spot is a contradiction: the loop waits for
       on-demand capacity, and taking spot is a separate decision made once that wait has produced its
       exhaustion evidence. Run the wait without --allow-spot; then, if it comes back empty, decide."
    [ "$DRY_RUN_ONLY" != 1 ] && [ "$PREFLIGHT_ONLY" != 1 ] \
      || die "--walk-until with --dry-run-only or --preflight-only would loop over passes that can never
       land a box; drop one of the two flags"
  fi
  # With the arguments, not at the first walk: a garbage list refused only after the tree was staged
  # to S3 and the user-data rendered has already spent the launch's slowest minute for nothing.
  set_transient_backoffs
  # The tag shorthand parser splits on commas, and a comma inside a value silently truncates the tag
  # set. The JSON path below is immune, but create-tags at the end is not, so both are rejected here.
  case "$PURPOSE$LABEL$PROJECT$SPOT_REASON" in
    *,*) die "a comma in --purpose, --label, --project or --spot-reason would be split by the EC2 tag
       shorthand parser; rephrase without one" ;;
  esac
  S3_PREFIX=${S3_PREFIX%/}
  RUN_PREFIX=$S3_PREFIX/$RUN_NAME
  # The sitting labels this invocation within the run: relaunches share the prefix but never a key,
  # so no sitting can overwrite what an earlier one ran. Timestamp plus pid, both free of any
  # commit-sha-shaped token.
  SITTING=$(date -u +%Y%m%dT%H%M%SZ)-p$$
  [ -n "$RUN_DIR" ] || RUN_DIR=/var/tmp/launch-gpu-box-$LABEL
  # /var/tmp rather than /tmp: /tmp here is a RAM-backed tmpfs with a hard inode cap, and it once ran
  # out of file slots and failed every session's shell box-wide.
  mkdir -p -- "$RUN_DIR" || die "cannot create the run directory $RUN_DIR"
}

# --- the account gate -----------------------------------------------------------------------------

# Asked of AWS rather than read off a profile name, because the profile is what the operator typed and
# the account is what the credentials resolve to. Unreadable credentials are red whatever else is
# configured, because every call after this one would fail too and this one is free. A mismatch is red
# when the operator named the account they meant, and that comparison is the whole reason the gate
# exists: a launch under the wrong account does not fail loudly. Every candidate subnet there "does not
# exist", which the walk used to log as a capacity decline, row after row, until it closed with the line
# about the nature of GPU supply; the only thing that stopped the one observed case was the staging
# upload landing in a bucket that account could not write, one --s3-prefix argument wide. With no
# expected account named, the identity is printed and the walk's own NotFound anchor (CAPACITY_DECLINES)
# is what stops that walk instead, one row in rather than before the upload. Runs before anything is
# staged, so nothing is paid for first.
account_preflight() {
  local caller
  caller=$("$AWS_CLI" sts get-caller-identity --region "$S3_REGION" --query Account --output text 2>&1)
  if ! [[ $caller =~ ^[0-9]{12}$ ]]; then
    die "cannot tell which account this launch would rent in: sts get-caller-identity did not
       answer with an account id. Fix the credentials in this shell (AWS_PROFILE, or a fresh login)
       and re-run. Nothing was staged or launched. It said:
       $(printf '%s' "$caller" | tr '\n' ' ' | cut -c1-400)"
  fi
  if [ -z "$EXPECTED_ACCOUNT" ]; then
    printf '   credentials resolve to account %s; no --expect-account (or SHIP_EXPECTED_ACCOUNT) was
       given, so nothing checked that this is the account you meant to rent in\n' "$caller" >&2
    return 0
  fi
  if [ "$caller" != "$EXPECTED_ACCOUNT" ]; then
    die "these credentials resolve to AWS account $caller, but this launch expected $EXPECTED_ACCOUNT.
       Under the wrong account every candidate subnet \"does not exist\", which a walk without this
       check once logged as a capacity decline on every row before closing with the line about the
       nature of GPU supply. Re-run with the profile that owns the rentals, or correct
       --expect-account. Nothing was staged or launched."
  fi
  printf '   credentials resolve to account %s, the account --expect-account named\n' "$caller" >&2
}

# --- the ship tree, staged and rendered -----------------------------------------------------------

# ONE staging pass supplies every fact downstream -- the tags, the report, and via the staging
# record the render's pinned values. Nothing re-derives the tree later, so there is no second
# reading that a peer edit could make disagree with the first, and therefore nothing to refuse
# when the tree moves mid-launch: the launch ships the snapshot its staging took, and the edit
# ships on the next sitting.
stage_and_render() {
  local staged key
  staged=$(bash "$STAGE_TOOL" --run-prefix "$RUN_PREFIX" --region "$S3_REGION" \
    --sitting "$SITTING" --stage-dir "$RUN_DIR/stage" ${DATA_ARGS+"${DATA_ARGS[@]}"}) \
    || die "staging the working tree failed; nothing was launched"
  TREE=$(printf '%s\n' "$staged" | sed -n 's/^ship_tree=//p')
  HEAD_SHA=$(printf '%s\n' "$staged" | sed -n 's/^head_sha=//p')
  DIRTY=$(printf '%s\n' "$staged" | sed -n 's/^dirty=//p')
  STAGED_KEY=$(printf '%s\n' "$staged" | sed -n 's/^s3_key=//p')
  PROVENANCE_KEY=$(printf '%s\n' "$staged" | sed -n 's/^provenance_key=//p')
  STAGE_DIR=$(printf '%s\n' "$staged" | sed -n 's/^stage_dir=//p')
  [ -n "$TREE" ] && [ -n "$HEAD_SHA" ] && [ -n "$DIRTY" ] && [ -n "$STAGED_KEY" ] \
    && [ -n "$PROVENANCE_KEY" ] && [ -n "$STAGE_DIR" ] \
    || die "stage_ship_tree.sh did not report a complete set of facts"

  USER_DATA=$RUN_DIR/user-data.sh
  # The render's own report is kept as a launch artifact rather than parsed: the authoritative check
  # below reads the key out of the rendered FILE, which is what EC2 will be handed.
  # The run's identity and the sitting's HEAD are supplied here rather than re-typed by every kit as
  # --set values: a kit that spelled its own S3 base or run name could disagree with the prefix this
  # launch actually stages under, and the reference kit (scripts/kit_reference/) builds its runner's
  # environment from these four. A kit --set naming any of them is refused at argument parsing.
  # All seven go to every template, and the render refuses a --set whose @NAME@ the template carries
  # nowhere, so a template has to consume all seven: a kit that hardcoded its own run prefix instead
  # of taking this one used to have that value dropped without a word.
  bash "$RENDER_TOOL" --template "$TEMPLATE" --out "$USER_DATA" \
    --stage-dir "$STAGE_DIR" \
    --set "SHIP_S3_REGION=$S3_REGION" \
    --set "DEADMAN_MINUTES=$DEADMAN_MINUTES" \
    --set "IDLE_WATCHDOG_S3_DEST=$RUN_PREFIX" \
    --set "RUN_NAME=$RUN_NAME" \
    --set "RUN_PREFIX=$RUN_PREFIX" \
    --set "SHIP_S3_PREFIX=$S3_PREFIX" \
    --set "SHIP_HEAD_SHA=$HEAD_SHA" \
    ${SET_ARGS+"${SET_ARGS[@]}"} >"$RUN_DIR/render-report.txt" \
    || die "rendering the user-data failed; nothing was launched"

  # Internal consistency, not freshness: the render substitutes the key this same invocation staged,
  # so only a template hardcoding its own SHIP_S3_KEY line can make the two disagree. Read out of the
  # rendered file rather than out of a variable, because the file is what EC2 will actually be handed.
  key=$(sed -n 's/^SHIP_S3_KEY=//p' "$USER_DATA" | head -1)
  [ -n "$key" ] || die "the rendered user-data at $USER_DATA declares no SHIP_S3_KEY"
  [ "$key" = "$STAGED_KEY" ] \
    || die "the rendered user-data tells the box to fetch $key, but this launch never staged that
       object; it staged $STAGED_KEY. The template is naming a key of its own, and the box would run
       code nobody in this launch looked at. Nothing was launched."
  printf '   rendered user-data pins the key this sitting staged (%s)\n' "$STAGED_KEY" >&2
}

# --- the mandatory-protection preflight -----------------------------------------------------------

# Two surfaces, because the protections live in two places: the user-data carries the boot patch and
# both killswitch layers, while IMDSv2 and the shutdown behaviour are flags on this script's own
# run-instances call. Failures are collected and reported together rather than one per run.
PROTECTION_PROBLEMS=()

require_pattern() {
  local file=$1 label=$2 pattern=$3 what=$4
  # -e is load-bearing: a pattern beginning with "--" is otherwise parsed by grep as a long option.
  grep -Eq -e "$pattern" "$file" || PROTECTION_PROBLEMS+=("$label: $what")
}

protection_preflight() {
  PROTECTION_PROBLEMS=()
  # An INVOCATION, not a mention: a user-data that only downloads the patcher would satisfy a lazy
  # grep for its name while leaving the box unpatched, and the newest published AMI still ships
  # whatever became vulnerable after it was built, so a box is exposed from boot however briefly it lives.
  require_pattern "$USER_DATA" "user-data" \
    '^[[:space:]]*(sudo[[:space:]]+)?(bash[[:space:]]+)?[^[:space:]]*ec2_boot_patch\.sh' \
    "no ec2_boot_patch.sh INVOCATION. Call it out of the extracted tree once the ship guard has
       verified that tree, e.g. bash \$SHIP_EXTRACT_DIR/scripts/ec2_boot_patch.sh"
  require_pattern "$USER_DATA" "user-data" '^[[:space:]]*shutdown -h \+[0-9]+' \
    "no max-lifetime dead-man switch (shutdown -h +<minutes>) as an early, unconditional line"
  # On a non-comment line: a commented-out arm line is exactly the template this exists to refuse.
  require_pattern "$USER_DATA" "user-data" \
    '^[^#]*idle_watchdog\.sh[[:space:]]+--minutes[[:space:]]+[0-9]+' \
    "no idle watchdog armed with --minutes. The dead-man switch alone is blind to the cheapest
       failure, a box that bootstraps and then never starts its job."
  # Unconditional by construction: the launcher always renders it, so its absence means the template
  # has no @IDLE_WATCHDOG_S3_DEST@ placeholder. Without it the watchdog logs its arm verdict only to a
  # local file that dies with the box, and no box has ever had it set.
  require_pattern "$USER_DATA" "user-data" 'IDLE_WATCHDOG_S3_DEST=s3://' \
    "IDLE_WATCHDOG_S3_DEST is not set to an s3:// prefix, so the killswitch's own arm-time verdict
       would die with the box. Add @IDLE_WATCHDOG_S3_DEST@ to the template."
  require_pattern "$USER_DATA" "user-data" 'box_ship_guard|step_verify_digest' \
    "the ship guard was not inlined, so the box would run whatever is under that key unchecked"
  # Anchored at a line-initial flag deliberately. The first version of this gate passed a sabotage that
  # downgraded IMDSv2 to HttpTokens=optional, because it found the word "required" inside its own
  # argument list.
  require_pattern "$SELF" "launcher" \
    '^[[:space:]]*--metadata-options[[:space:]]+.?HttpTokens=required' \
    "run-instances does not enforce IMDSv2 (--metadata-options HttpTokens=required)"
  require_pattern "$SELF" "launcher" \
    '^[[:space:]]*--instance-initiated-shutdown-behavior[[:space:]]+terminate' \
    "run-instances does not set the shutdown behaviour to terminate, so a box that shuts itself down
       would only STOP and keep billing its EBS root while teaching false comfort"
  if [ "${#PROTECTION_PROBLEMS[@]}" -ne 0 ]; then
    printf 'FATAL: mandatory protections missing (%s):\n' "${#PROTECTION_PROBLEMS[@]}" >&2
    printf '  - %s\n' "${PROTECTION_PROBLEMS[@]}" >&2
    exit 1
  fi
  printf '   protections present: boot patch, dead-man, idle watchdog, watchdog S3 destination,\n' >&2
  printf '   ship guard, IMDSv2, shutdown-behaviour terminate\n' >&2
  # A WARNING rather than a refusal, because the kits that lack it are mid-wave and a refusal would
  # strand their relaunches. Without --done-marker the idle threshold is the watchdog's only trigger,
  # and every box then bills a 40-45 minute idle tail after its agenda's last artifact ships (2,400 s
  # measured on one box, 2026-09-01); with it, the box is torn down on the next 60 s poll. Matched on
  # the watchdog's own arm line, on a non-comment line, and on a path: a runner heredoc that merely
  # echoes the flag, a comment mentioning it, or a bare flag with nothing after it satisfy none of that.
  if ! grep -Eq -e '^[^#]*idle_watchdog\.sh[[:space:]]+[^#]*--done-marker[[:space:]]+/' "$USER_DATA"; then
    printf 'WARN: the idle watchdog is armed with no --done-marker <path>, so this box will bill a
       40-45 minute idle tail after its agenda finishes: the idle threshold is then the only trigger.
       Add --done-marker /home/ubuntu/AGENDA_DONE to the idle_watchdog.sh line of the template and have
       the runner touch that path as its LAST act, after its final upload has returned; the watchdog
       then tears the box down within a minute.\n' >&2
  fi
}

# --- the reference-kit comparison ------------------------------------------------------------------

# Every rule beyond the refusals above lives in scripts/kit_reference/markers.tsv, with the reference
# template and runner that carry all of them. The kits themselves are hand-written copies under
# /var/tmp, and every rule used to be re-applied to each copy by hand: of eleven templates written in
# two weeks, seven had dropped the done-marker and one heartbeat grepped a log that exists only after
# training. So each kit is compared against that table here, at preflight, and every rule it lacks is
# a WARNING naming the rule and why it exists -- never a refusal, because a kit that predates a rule
# must still be able to relaunch mid-wave. The runner is compared too when the user-data names one
# inside the staged tree (a RUNNER_TREE_PATH= line), which is how the reference kit hands over.
REFERENCE_CHECKER=$HERE/kit_reference/check_markers.sh

report_reference_rules() {
  local surface=$1 file=$2 label=$3 report status kind rule why
  report=$(bash "$REFERENCE_CHECKER" --skip-native "$surface" "$file")
  status=$?
  case $status in
    0) printf '   %s carries every reference kit rule (%s)\n' "$label" "$surface" >&2 ;;
    1)
      while IFS=$'\t' read -r kind rule why; do
        [ -n "$rule" ] || continue
        if [ "$kind" = FORBIDDEN ]; then
          printf 'WARN: %s breaks the reference kit rule %s: %s\n' "$label" "$rule" "$why" >&2
        else
          printf 'WARN: %s lacks the reference kit rule %s: %s\n' "$label" "$rule" "$why" >&2
        fi
      done <<<"$report"
      ;;
    *)
      printf 'WARN: check_markers.sh could not compare %s against the reference kit (exit %s)\n' \
        "$label" "$status" >&2
      ;;
  esac
}

reference_kit_preflight() {
  local runner_path
  if [ ! -r "$REFERENCE_CHECKER" ]; then
    printf 'WARN: scripts/kit_reference/check_markers.sh is not beside this launcher, so the template was
       not compared against the reference kit (scripts/kit_reference/README.md)\n' >&2
    return 0
  fi
  report_reference_rules user-data "$USER_DATA" "the template"
  # The last assignment, because that is the one the box's shell leaves in force.
  runner_path=$(sed -n 's/^RUNNER_TREE_PATH=//p' "$USER_DATA" | tail -1 | tr -d '"')
  if [ -z "$runner_path" ]; then
    printf '   no RUNNER_TREE_PATH= line in the user-data, so no runner was compared against the reference kit\n' >&2
  elif [ -r "$STAGE_DIR/extracted/$runner_path" ]; then
    report_reference_rules runner "$STAGE_DIR/extracted/$runner_path" "the runner $runner_path"
  else
    printf 'WARN: the user-data names RUNNER_TREE_PATH=%s but the staged tree has no such file, so the box
       would fail at hand-off, and the runner could not be compared against the reference kit\n' \
      "$runner_path" >&2
  fi
}

# --- tags ----------------------------------------------------------------------------------------

# Tags as JSON via file://, never the key=value shorthand, whose parser splits on commas. RunName ties
# the box to its S3 run prefix; the Ship* tags are provenance RECORDS -- they ride into the
# RunInstances CloudTrail event permanently, outliving both the box and the S3 prefix, so forensics
# can still say what bytes a box ran and which tool launched it. Nothing reads them to gate a launch.
#
# Rendered once per market rather than once per run: with --allow-spot the walk tries every on-demand
# row first, and a single specification written up front would stamp Market=spot onto a box that
# actually launched on demand. Values reach python through argv, never interpolated into its source, so
# a quote in the operator's --purpose cannot break the renderer.
write_tag_specification() {
  local market=$1
  TAGSPEC=$RUN_DIR/tagspec-$market.json
  /usr/bin/env python3 - "$TAGSPEC" "$LABEL" "$PROJECT" "$PURPOSE" "$RUN_NAME" "$TREE" \
    "$HEAD_SHA" "$DIRTY" "$market" "$SPOT_REASON" <<'PY'
import json
import sys

out, label, project, purpose, run_name, tree, head, dirty, market, reason = sys.argv[1:11]
tags = [
    {"Key": "Name", "Value": label},
    {"Key": "Project", "Value": project},
    {"Key": "DoNotReap", "Value": label},
    {"Key": "purpose", "Value": purpose},
    {"Key": "RunName", "Value": run_name},
    {"Key": "ShipTree", "Value": tree},
    {"Key": "ShipHeadSha", "Value": head},
    {"Key": "ShipDirty", "Value": dirty},
    {"Key": "Market", "Value": market},
]
if market == "spot" and reason:
    tags.append({"Key": "SpotReason", "Value": reason})
with open(out, "w") as handle:
    json.dump([{"ResourceType": "instance", "Tags": tags}], handle)
PY
  [ -s "$TAGSPEC" ] || die "the tag specification did not render to $TAGSPEC"
  grep -Fq "\"$TREE\"" "$TAGSPEC" \
    || die "the tag specification does not carry the ship tree, so nothing afterwards could tell this
       box from one launched outside this tool"
}

# --- the capacity walk ---------------------------------------------------------------------------

# The per-region facts -- which AMI, and whether a dry run passes -- are gathered for EVERY region of
# the candidate file at once, in background subshells that write their answers under RUN_DIR, and
# then adopted by the walk lazily, in row order, the first time a row reaches each region. Two things
# follow from that split. The saving: nine regions of describe-images plus a dry run cost the sum of
# their round trips when done one region at a time (~25-35 s per walk), and the slowest single one
# when done together. The invariant: nothing observable moves. The memo the walk reads is filled with
# the same answers in the same order, every stderr line a region's probe wrote is replayed at the
# moment the sequential walk would have written it, and a dry run rejected for a non-capacity reason
# still stops the walk exactly where it used to -- when a row first reaches that region, after every
# earlier region has had its real attempts -- so a typo in a later region's row cannot block a launch
# an earlier region would have granted.
#
# Answers in globals rather than on stdout, so the memo survives: called as `ami=$(memo_ami ...)` the
# appends would land in a command substitution's subshell and be discarded.
RESOLVED_AMI=""
ADOPTED_REGIONS=" "
declare -A PROBE_PIDS=()

region_probe_file() {
  printf '%s/probe-%s.%s' "$RUN_DIR" "$1" "$2"
}

# The memo lookup: 0 with RESOLVED_AMI set when the region has an image, 1 when its probe found none.
# Only ever called after adopt_region has filled the memo for that region.
memo_ami() {
  local region=$1 index=0
  RESOLVED_AMI=""
  while [ "$index" -lt "${#AMI_REGIONS[@]}" ]; do
    if [ "${AMI_REGIONS[index]}" = "$region" ]; then
      RESOLVED_AMI=${AMI_IDS[index]}
      [ -n "$RESOLVED_AMI" ] && return 0
      return 1
    fi
    index=$((index + 1))
  done
  die "memo_ami($region) was asked before adopt_region filled the memo; this is a bug in the walk"
}

# The newest build of the image family in one region, or 1 when the region has none. A LOOKUP FAILURE
# IS NOT AN ABSENCE, and the two arrive identically here: a DescribeImages denial, an SCP block and a
# malformed --ami-name-pattern all leave stdout empty exactly as a region with no such image does, and
# the caller's response to an absence is to skip the region silently -- so every region skipped, and a
# walk that rented nothing then closed with the line about the nature of GPU supply. Anything that is
# neither an image id nor the API's own empty answer is therefore surfaced before the skip.
query_ami() {
  local region=$1 ami
  RESOLVED_AMI=""
  ami=$("$AWS_CLI" ec2 describe-images --region "$region" --owners amazon \
    --filters "Name=name,Values=$AMI_NAME_PATTERN" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text 2>&1)
  case $ami in
    ami-*)
      printf '   using base AMI %s in %s (the boot patch is what closes the vulnerability window on
       this box)\n' "$ami" "$region" >&2
      RESOLVED_AMI=$ami
      return 0
      ;;
    None | '') ;;
    *)
      printf 'WARN: the AMI lookup in %s FAILED rather than finding nothing, so this region is about to
       be skipped for what may be a DescribeImages denial, an SCP block or a malformed
       --ami-name-pattern rather than an absent image. It said: %s\n' \
        "$region" "$(printf '%s' "$ami" | tr '\n' ' ' | cut -c1-400)" >&2
      ;;
  esac
  return 1
}

# The AWS CLI's own retry layer is OFF for the walk's run-instances calls, and a transient answer is
# retried here instead, explicitly and boundedly. botocore's standard mode treats HTTP 500 as
# transient and EC2 serves InsufficientInstanceCapacity as a 500, so every declined row used to cost
# three RunInstances calls plus exponential backoff, ~7.5 s a row: a 60-row walk ran eight minutes of
# which two thirds re-asked a question whose answer does not change in the seconds a retry spans
# (measured 2026-09-02; every decline logged "(reached max retries: 2)"). Everything else the CLI
# would have retried, this walk must now retry itself or it ends on the first occurrence as a
# "not capacity" error: a throttle, a server-side failure (InternalError, InternalFailure,
# ServiceUnavailable, RequestTimeout, or a bare 5xx whose body the CLI could not parse) and a
# connection-level failure, whose phrasings are botocore's verbatim. Matched by error NAME, never by
# HTTP status, so the 500-coded capacity decline stays a decline. The same row is re-tried after each
# backoff below, in order, and an answer that outlasts the list surfaces with its own diagnosis rather
# than being swallowed -- except (Service)Unavailable, which CAPACITY_DECLINES also names, so it is
# walked past after the retries exactly as the CLI's own exhausted retries were. The list is an
# environment knob only so the test suite can drive the branch without waiting.
THROTTLE_ERRORS='\((RequestLimitExceeded|Throttling|ThrottlingException|ThrottledException|RequestThrottledException|RequestThrottled|EC2ThrottledException|TooManyRequestsException|SlowDown)\)'
TRANSIENT_ERRORS="$THROTTLE_ERRORS"'|\((InternalError|InternalFailure|ServiceUnavailable|Unavailable|RequestTimeout|RequestTimeoutException|500|502|503|504)\)|Could not connect to the endpoint URL|Connect timeout on endpoint URL|Read timeout on endpoint URL|Connection was closed before we received a valid response|Failed to connect to proxy URL|An HTTP Client failed to establish a connection'
TRANSIENT_BACKOFF_SECONDS=${TRANSIENT_BACKOFF_SECONDS:-2 5}
TRANSIENT_BACKOFFS=()

set_transient_backoffs() {
  local backoff
  read -ra TRANSIENT_BACKOFFS <<<"$TRANSIENT_BACKOFF_SECONDS"
  for backoff in "${TRANSIENT_BACKOFFS[@]}"; do
    case $backoff in
      *[!0-9]*) die "TRANSIENT_BACKOFF_SECONDS must be whole seconds separated by spaces, not
       '$TRANSIENT_BACKOFF_SECONDS'" ;;
    esac
  done
}

transient_label() {
  if printf '%s' "$1" | grep -Eq "$THROTTLE_ERRORS"; then
    printf 'throttled by the EC2 API'
  else
    printf 'transient failure from the EC2 API or the connection to it'
  fi
}

# The token is the caller's, one per row of the walk (or per region's dry run), and every attempt at
# that row sends the same one. EC2 then treats a retry after an answer that never arrived -- a read
# timeout, a closed connection -- as the same request, and hands back the instance the first attempt
# actually created rather than launching a second one. The CLI generates a token per process when
# none is given: botocore's own retries shared it, and re-invoking the CLI would have lost exactly
# that protection.
run_instances() {
  local mode=$1 region=$2 ami=$3 itype=$4 sg=$5 subnet=$6 market=$7 token=$8
  local dry=() spot=() out attempt=0 budget=$((${#TRANSIENT_BACKOFFS[@]} + 1))
  [ "$mode" = dry ] && dry=(--dry-run)
  if [ "$market" = spot ]; then
    # An array, not a string: on demand needs the flag to vanish entirely, and an empty quoted string
    # would pass a bogus argument while an unquoted one invites word splitting. MaxPrice caps a
    # mid-run spike and is NOT a bid; a cap under the running price declines every candidate and reads
    # exactly like a capacity shortage.
    if [ -n "$SPOT_MAX_PRICE" ]; then
      spot=(--instance-market-options "MarketType=spot,SpotOptions={MaxPrice=$SPOT_MAX_PRICE}")
    else
      spot=(--instance-market-options MarketType=spot)
    fi
  fi
  while :; do
    out=$(AWS_MAX_ATTEMPTS=1 "$AWS_CLI" ec2 run-instances --region "$region" "${dry[@]}" \
      --client-token "$token" \
      --image-id "$ami" --instance-type "$itype" "${spot[@]}" \
      --subnet-id "$subnet" --security-group-ids "$sg" \
      --iam-instance-profile "Name=$INSTANCE_PROFILE" \
      --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
      --metadata-options 'HttpTokens=required' \
      --instance-initiated-shutdown-behavior terminate \
      --user-data "file://$USER_DATA" \
      --tag-specifications "file://$TAGSPEC" \
      --query 'Instances[0].InstanceId' --output text 2>&1)
    if [ "$attempt" -lt "${#TRANSIENT_BACKOFFS[@]}" ] \
      && printf '%s' "$out" | grep -Eq "$TRANSIENT_ERRORS"; then
      printf '     %s (attempt %d of %d); retrying the same row in %ss: %s\n' \
        "$(transient_label "$out")" "$((attempt + 1))" "$budget" "${TRANSIENT_BACKOFFS[attempt]}" \
        "$(printf '%s' "$out" | tr '\n' ' ' | cut -c1-160)" >&2
      sleep "${TRANSIENT_BACKOFFS[attempt]}"
      attempt=$((attempt + 1))
      continue
    fi
    printf '%s\n' "$out"
    return 0
  done
}

# Both consumers of a run-instances answer share this. A transient answer that outlasted its retries
# is not a decline and not a malformed request, so it gets its own words: the generic "not capacity"
# diagnosis would send the operator hunting for a bad argument in a candidate file that is fine.
die_if_transient() {
  local out=$1 where=$2 attempts=$((${#TRANSIENT_BACKOFFS[@]} + 1)) last
  printf '%s' "$out" | grep -Eq "$TRANSIENT_ERRORS" || return 0
  last=$(printf '%s' "$out" | tr '\n' ' ' | cut -c1-400)
  if printf '%s' "$out" | grep -Eq "$THROTTLE_ERRORS"; then
    die "run-instances in $where: throttled by the EC2 API on $attempts consecutive attempts (backoffs
       of ${TRANSIENT_BACKOFF_SECONDS} s between them). That is the account's RunInstances rate limit,
       not capacity and not this candidate file: another walk is most likely running in the account
       right now. Let it finish, or widen TRANSIENT_BACKOFF_SECONDS, then re-run this launch.
       Last answer: $last"
  fi
  die "run-instances in $where: the EC2 API or the connection to it failed transiently on $attempts consecutive attempts
       (backoffs of ${TRANSIENT_BACKOFF_SECONDS} s between them). That is the service or the network,
       not capacity and not this candidate file. Re-run this launch; if it recurs, check this machine's
       route to EC2 and the AWS Health Dashboard before touching the candidate file.
       Last answer: $last"
}

# One region's probe, run in a background subshell of probe_regions_in_parallel: the AMI lookup and,
# when it found an image, the dry run with the region's first row. Everything it learns goes into
# files -- the memo arrays of a subshell die with it -- and everything it says goes into the region's
# log, which adopt_region replays at the moment the sequential walk would have said it. The done file
# is written last, so a probe that died mid-way is distinguishable from one that found nothing.
probe_one_region() {
  local region=$1 itype=$2 sg=$3 subnet=$4 market=$5
  local log ami_file dry_file done_file
  log=$(region_probe_file "$region" log)
  ami_file=$(region_probe_file "$region" ami)
  dry_file=$(region_probe_file "$region" dry)
  done_file=$(region_probe_file "$region" "done")
  : >"$log"
  if query_ami "$region" 2>>"$log"; then
    printf '%s\n' "$RESOLVED_AMI" >"$ami_file"
    run_instances dry "$region" "$RESOLVED_AMI" "$itype" "$sg" "$subnet" "$market" \
      "$SITTING-dry-$region" >"$dry_file" 2>>"$log"
  else
    : >"$ami_file"
  fi
  : >"$done_file"
}

# The first row of every region, in file order: the row whose type, security group and subnet the
# region's dry run is made with -- the same row the sequential walk reached first.
first_row_per_region() {
  local itype region sg subnet seen=" "
  while IFS='|' read -r itype region sg subnet; do
    case ${itype:-} in '' | \#*) continue ;; esac
    case $seen in *" $region "*) continue ;; esac
    seen="$seen$region "
    printf '%s|%s|%s|%s\n' "$itype" "$region" "$sg" "$subnet"
  done <"$CANDIDATES"
}

# Every region's probe at once, and NO wait here: each probe is waited on by adopt_region the moment
# a row first reaches its region, so the walk blocks on a region exactly when the sequential walk
# would have, and a slow or hung describe-images in the ninth region (the probes keep the CLI's
# default retries and 60 s read timeout, so minutes in the worst case) cannot delay a launch the first
# region grants immediately. stdin is cut off from each subshell because the loop's own stdin is the
# row stream, and a child that read from it would eat the rows its siblings were about to get.
probe_regions_in_parallel() {
  local market=$1 itype region sg subnet
  rm -f -- "$RUN_DIR"/probe-*.done
  while IFS='|' read -r itype region sg subnet; do
    (probe_one_region "$region" "$itype" "$sg" "$subnet" "$market") </dev/null &
    PROBE_PIDS[$region]=$!
  done < <(first_row_per_region)
}

# Adopt a region's probe the first time a row reaches it: replay what the probe said, fill the memo,
# and apply the dry run's verdict. The two client-side traps the dry run catches -- a malformed tag
# specification and an over-length user-data -- fail on EVERY candidate and would read as a
# fleet-wide capacity shortage; a bad subnet or security-group id in a newly added region fails only
# there, which is why the probe is per region rather than once overall, and why its verdict is applied
# here in row order rather than up front: a typo in a later region's row must not stop the walk
# before an earlier region has had its real attempts.
adopt_region() {
  local region=$1 ami out log dry_file
  case $ADOPTED_REGIONS in *" $region "*) return 0 ;; esac
  ADOPTED_REGIONS="$ADOPTED_REGIONS$region "
  # Block on this region's probe only now that a row has reached it. Its exit status says nothing
  # the files below do not: the done file is the verdict on whether it finished at all.
  wait "${PROBE_PIDS[$region]}"
  log=$(region_probe_file "$region" log)
  dry_file=$(region_probe_file "$region" dry)
  [ -s "$log" ] && cat -- "$log" >&2
  [ -f "$(region_probe_file "$region" "done")" ] \
    || die "the probe of $region never finished; whatever it managed to say is replayed above"
  ami=$(cat -- "$(region_probe_file "$region" ami)")
  AMI_REGIONS+=("$region")
  AMI_IDS+=("$ami")
  [ -n "$ami" ] || return 0
  out=$(cat -- "$dry_file")
  case $out in
    *DryRunOperation* | *"would have succeeded"*)
      printf '   dry-run OK in %s\n' "$region" >&2
      return 0
      ;;
  esac
  if printf '%s' "$out" | grep -Eq "$CAPACITY_DECLINES"; then
    printf '   dry-run in %s declined on capacity; the walk continues\n' "$region" >&2
    return 0
  fi
  die_if_transient "$out" "$region (dry run)"
  die "the dry run in $region was rejected for a reason that is not capacity, so every remaining
       candidate would fail the same way:
       $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-400)"
}

# Once a box has landed, the regions past the landing row still have probes in flight, and neither
# the report nor the shutdown-behaviour check should wait on the slowest of them -- the sequential
# walk never touched those regions at all. Only a probe not yet adopted is still this shell's
# unreaped child, so only those pids may be signalled: an adopted one has been waited on, and its
# pid may already belong to some other process on this shared box.
reap_probes() {
  local region
  for region in "${!PROBE_PIDS[@]}"; do
    case $ADOPTED_REGIONS in *" $region "*) continue ;; esac
    kill "${PROBE_PIDS[$region]}" 2>/dev/null
    wait "${PROBE_PIDS[$region]}" 2>/dev/null
  done
}

walk() {
  local market itype region sg subnet ami out first_market row
  # A fresh memo per pass: a repeat pass under --walk-until is a new walk, so a newer image build or a
  # subnet that vanished during the pause must be seen.
  AMI_REGIONS=()
  AMI_IDS=()
  ADOPTED_REGIONS=" "
  PROBE_PIDS=()
  # The dry runs carry the first pass's market, as they did when the walk probed a region on
  # reaching it: with the default order that is on demand, and spot-first requires --allow-spot.
  first_market=${MARKET_ORDER%% *}
  write_tag_specification "$first_market"
  printf '\n== probing every region of the candidate file at once (AMI lookup and dry run)\n' >&2
  probe_regions_in_parallel "$first_market"
  # Word-splitting MARKET_ORDER is deliberate: it holds one or two known tokens set above.
  for market in $MARKET_ORDER; do
    if [ "$market" = spot ] && [ "$ALLOW_SPOT" != 1 ]; then
      continue
    fi
    if [ "$market" = spot ]; then
      printf '\n== entering the spot rows (order: %s). Reason given: %s\n' \
        "$MARKET_ORDER" "$SPOT_REASON" >&2
    else
      printf '\n== walking the on-demand rows\n' >&2
    fi
    # Per pass, not once up front: under either ordering the second pass must not inherit the
    # first pass's Market tag (a spot-first fall-through would otherwise stamp Market=spot onto a
    # box that actually launched on demand).
    write_tag_specification "$market"
    row=0
    while IFS='|' read -r itype region sg subnet; do
      case ${itype:-} in '' | \#*) continue ;; esac
      row=$((row + 1))
      adopt_region "$region"
      if ! memo_ami "$region"; then
        printf '   skipping %s: no AMI matched the pattern there\n' "$region" >&2
        continue
      fi
      ami=$RESOLVED_AMI
      if [ "$DRY_RUN_ONLY" = 1 ]; then
        continue
      fi
      printf '   trying %-14s %s %s (%s)\n' "$itype" "$region" "$subnet" "$market" >&2
      # The client token: unique to this row within the sitting and market (a duplicate row would
      # otherwise be answered with the first row's result or an IdempotentParameterMismatch), and well
      # under EC2's 64 ASCII characters -- the sitting is at most 34 and the suffix at most 15.
      out=$(run_instances real "$region" "$ami" "$itype" "$sg" "$subnet" "$market" \
        "$SITTING-$market-r$row")
      if [ "${out:0:2}" = "i-" ]; then
        LAUNCH_INSTANCE=$out
        LAUNCH_REGION=$region
        LAUNCH_MARKET=$market
        reap_probes
        return 0
      fi
      if printf '%s' "$out" | grep -Eq "$CAPACITY_DECLINES"; then
        printf '     declined: %s\n' "$(printf '%s' "$out" | tr '\n' ' ' | cut -c1-160)" >&2
        continue
      fi
      die_if_transient "$out" "$region"
      die "run-instances failed for a reason that is not capacity, so every remaining candidate would
       fail the same way:
       $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-400)"
    done <"$CANDIDATES"
  done
  return 1
}

# --- after the instance exists --------------------------------------------------------------------

# The killswitch's precondition, read back from the API rather than trusted from the launch flag,
# because describe-instances does not carry this field at all. The idle watchdog calls `shutdown -h
# now`, which only TERMINATES when this attribute says terminate; a box that merely stops keeps
# billing its EBS root. A box that cannot kill itself is the runaway-cost failure this whole
# arrangement exists to prevent, so it is terminated here rather than warned about.
verify_shutdown_behavior() {
  local instance=$1 region=$2 behavior
  behavior=$("$AWS_CLI" ec2 describe-instance-attribute --instance-id "$instance" --region "$region" \
    --attribute instanceInitiatedShutdownBehavior \
    --query 'InstanceInitiatedShutdownBehavior.Value' --output text 2>&1)
  printf '   shutdown-behavior=%s (must be: terminate)\n' "$behavior" >&2
  [ "$behavior" = "terminate" ] && return 0
  printf 'FATAL: %s cannot terminate itself, so neither killswitch layer can end it. Terminating it\n' \
    "$instance" >&2
  printf '       now rather than leaving a box no automation can stop.\n' >&2
  "$AWS_CLI" ec2 terminate-instances --instance-ids "$instance" --region "$region" >&2 \
    || printf 'FATAL: the terminate call also failed; terminate %s in %s by hand.\n' \
      "$instance" "$region" >&2
  exit 1
}

record_launch() {
  local instance=$1 region=$2
  printf '%s\n' "$instance" >"$RUN_DIR/instance-id.txt"
  printf '%s\n' "$region" >"$RUN_DIR/region.txt"
  # Tagged after launch because the key embeds the instance id, and pointed at the SHARED status/
  # directory beside the run prefixes, which is where every runner's heartbeat unit is meant to write.
  "$AWS_CLI" ec2 create-tags --region "$region" --resources "$instance" \
    --tags "Key=HeartbeatS3,Value=$S3_PREFIX/status/$instance.txt" >&2 \
    || printf 'WARN: could not tag HeartbeatS3 on %s, so nothing can find where it reports liveness\n' \
      "$instance" >&2
}

print_facts() {
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'run_prefix=%s\n' "$RUN_PREFIX"
  printf 'sitting=%s\n' "$SITTING"
  printf 'ship_tree=%s\n' "$TREE"
  printf 'head_sha=%s\n' "$HEAD_SHA"
  printf 'dirty=%s\n' "$DIRTY"
  printf 's3_key=%s\n' "$STAGED_KEY"
  printf 'provenance_key=%s\n' "$PROVENANCE_KEY"
  printf 'user_data=%s\n' "$USER_DATA"
}

main() {
  local pass=1 started now deadline pause
  parse_arguments "$@"
  # Before the first paid call, which is the staging upload, and once per launch rather than once per
  # --walk-until pass: the account cannot change between passes of one shell.
  account_preflight
  started=$(date +%s)
  deadline=$((started + ${WALK_UNTIL_MINUTES:-0} * 60))
  printf '== launching %s as run %s (sitting %s)\n' "$LABEL" "$RUN_NAME" "$SITTING" >&2
  # One iteration is today's launch exactly. Under --walk-until each further pass is a whole new
  # sitting -- staged, rendered and preflighted afresh, because the tree as it stands at each pass is
  # what ships -- and its label carries the pass number, so two passes within one second can never
  # share a key.
  while :; do
    stage_and_render
    write_tag_specification ondemand
    protection_preflight
    reference_kit_preflight
    if [ "$PREFLIGHT_ONLY" = 1 ]; then
      printf '== preflight only: everything verified, nothing rented\n' >&2
      print_facts
      return 0
    fi

    # walk is called as a plain statement, never as `x=$(walk)`: it can die on a non-capacity error,
    # and `die` inside a command substitution exits only the subshell, leaving the caller to carry on
    # with an empty string and print a second, wrong diagnosis after the real one.
    if walk; then
      break
    fi
    if [ "$DRY_RUN_ONLY" = 1 ]; then
      printf '== dry runs only: every region probed, nothing rented\n' >&2
      print_facts
      return 0
    fi
    if [ -z "$WALK_UNTIL_MINUTES" ]; then
      die "every candidate declined. That is the nature of GPU supply rather than a fault to debug:
       widen the candidate file by instance size, then availability zone, then region."
    fi
    now=$(date +%s)
    if [ "$now" -ge "$deadline" ]; then
      die "every candidate declined on each of $pass pass(es) over $(((now - started) / 60)) minute(s)
       (--walk-until $WALK_UNTIL_MINUTES). That is the nature of GPU supply rather than a fault to
       debug: widen the candidate file by instance size, then availability zone, then region. A p5
       pass (its own candidate file) or spot (--allow-spot with its --spot-reason) are
       separate, explicit decisions that this wait never takes on its own."
    fi
    # The pause is cut short to end on the deadline, so no pass begins after it and the wait overruns
    # it by at most one pass; the alternative, a deadline checked only after each pass, ran a
    # --walk-until 60 with the default pause to ~67 minutes.
    pause=$((WALK_PAUSE_MINUTES * 60))
    [ $((now + pause)) -le "$deadline" ] || pause=$((deadline - now))
    printf '\n== pass %d: every candidate declined; pass %d in %d s, deadline in %d minute(s)\n' \
      "$pass" "$((pass + 1))" "$pause" "$(((deadline - now + 59) / 60))" >&2
    sleep "$pause"
    pass=$((pass + 1))
    SITTING=$(date -u +%Y%m%dT%H%M%SZ)-p$$-pass$pass
    printf '\n== pass %d of run %s (sitting %s)\n' "$pass" "$RUN_NAME" "$SITTING" >&2
  done
  record_launch "$LAUNCH_INSTANCE" "$LAUNCH_REGION"
  verify_shutdown_behavior "$LAUNCH_INSTANCE" "$LAUNCH_REGION"
  printf '\n== %s is running in %s as run %s sitting %s\n' \
    "$LAUNCH_INSTANCE" "$LAUNCH_REGION" "$RUN_NAME" "$SITTING" >&2
  printf 'instance=%s\nregion=%s\nmarket=%s\n' "$LAUNCH_INSTANCE" "$LAUNCH_REGION" "$LAUNCH_MARKET"
  print_facts
}

main "$@"
