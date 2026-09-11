#!/bin/bash
# Terminate a rented GPU box that has stopped doing work. Layer 2 of the killswitch.
#
# THE ORIGIN COPY. Two downstream copies of this file live in a private research tree and carry the
# same body in lockstep; the markers reading "fu_train:" and "slate 2026-09-01:" are changes that
# arrived back from them. fu_train's: the work-process pattern became a flag, an agenda-complete
# marker trigger was added (empty default = the original behaviour exactly), the teardown was lifted
# into one function shared by both triggers, and the absent-nvidia-smi case was documented rather
# than merely true. slate's are a paid-for incident: two GPU-less CPU boxes rented on a GPU AMI
# carried a driverless nvidia-smi whose failure text read as "busy" on every poll, so the watchdog
# held both boxes past their agenda-complete markers, silently, until an operator collected them by
# hand. Out of it came the gpu probe's failed-invocation case, a heartbeat line on every poll, and a
# simulated-box harness beside the downstream copies that reproduced the hold and pins both
# behaviors. Keep every copy byte-identical below its header, the usage path being the one allowed
# per-copy line; this repo is public, so box identifiers and private paths stay in the downstream
# headers and out of this file.
#
# Layer 1 is the max-lifetime dead-man (`shutdown -h +<minutes>` in user-data). It bounds the worst
# case but not the common one: on 2026-08-18 a rented g7e bootstrapped successfully, its driving
# session went away before starting the job, and it sat at GPU 0% for ~50 minutes until a human
# noticed. Layer 1 would not have fired for hours. This is the layer that catches that.
#
# Idle means ALL of: no tmux pane running anything, no work process, and an idle GPU. Requiring all
# three is deliberate -- GPU utilisation alone is far too twitchy for a multi-stage agenda, which
# legitimately sits at 0% while a model loads from disk, while completions are parsed on CPU, and
# while artifacts sync to S3. Every long job here runs inside tmux, so what a pane is RUNNING is the
# natural "work is in flight" signal.
#
# What a pane runs, and not merely that a session exists: the existence of a session was the original
# test, and it made this layer unable to catch the incident above. The launch convention here is
# `tmux new-session -d` and then send the command into the session, so the session is created before
# any work is, and a box whose driving agent went away between the two kept an empty detached shell
# alive until the max-lifetime dead-man fired hours later.
#
# Escape hatch: `touch ~<run user>/KEEPALIVE` to hold a box deliberately (debugging, an interactive
# session, a deliberate pause). The path follows --user, so it is the home of whoever the watchdog
# was pointed at. The watchdog reports that it is suppressed rather than staying silent.
#
# Teardown calls ec2:TerminateInstances on this instance and then runs `shutdown -h now` regardless
# of how that went; api_self_terminate below carries the why. So the instance role wants
# ec2:TerminateInstances, and the launch MUST still set InstanceInitiatedShutdownBehavior=terminate,
# because the fallback is what kills the box on every box and every day the API call cannot.
#
# Whether the role really has that grant is answered at arm time by a dry run, so every box says so
# in its own log before doing any work: grep GRANT PROBE for LIVE, MISSING or INDETERMINATE. It is an
# announcement and never a gate, so a box whose grant is missing still runs its job and still dies.
#
# That verdict then has to leave the box, or it is a check that reports into the void. It is written
# on stdout, which the caller redirects into a file on a root disk this script exists to destroy, and
# the arm-time lines are pushed out of every launcher's `tail -N` heartbeat by the first idle poll --
# so the boxes whose verdict matters most, the ones layer 2 really terminates, are exactly the ones
# that lose it. Every arm-time line is therefore also accumulated into ARM_RECORD and, when
# IDLE_WATCHDOG_S3_DEST names an s3:// prefix, uploaded under it keyed by instance id, minutes into
# the box's life. Publishing is best-effort and bounded and gates nothing: grep the log for `arm
# record` to tell PUBLISHED from FAILED from UNCONFIGURED, because "the grant was denied" and "nobody
# ever found out" have to stay distinguishable.
#
# The destination is caller-supplied rather than discovered, and the launcher kits that supply it are
# not in this repo. IMDS instance tags are disabled on these boxes and the instance role carries no
# ec2:DescribeTags, so the box cannot read its own HeartbeatS3 tag; the tag's format is inconsistent
# across kits anyway. What this file owns instead is everything a kit could get wrong: the key layout,
# the bound, the content, and saying so in the log when a kit forgets the variable.
#
# Usage, from user-data as root, BEFORE the job starts:
#   IDLE_WATCHDOG_S3_DEST=s3://<bucket>/<prefix> \
#     nohup /home/ubuntu/games/scripts/idle_watchdog.sh --minutes 20 \
#       >>/var/log/idle-watchdog.log 2>&1 &
set -uo pipefail

IDLE_MINUTES=20
POLL_SECONDS=60
RUN_USER=ubuntu
# fu_train: the original hardcoded 'python|uv run|aws s3' inside work_in_flight. It is a flag here
# because this workstream's phase-2 job may be driven by torchrun rather than by a bare python, and a
# work signal that silently does not match its own job is the failure mode layer 2 exists to catch.
# Widening it is cheap (a box held minutes longer); narrowing it terminates a box mid-run.
WORK_PATTERN='python|uv run|aws s3|torchrun|lightgbm'
# fu_train: empty means "no acceleration, the idle threshold is the only trigger", which is the
# original behaviour exactly. See the loop for what it does when set.
DONE_MARKER=''
# A driver holds a few hundred MiB with nothing running, so "GPU busy" cannot mean "non-zero".
GPU_BUSY_MIB=512

IMDS_BASE=http://169.254.169.254/latest
# The token is spent within the same second it is issued, so the shortest useful TTL is plenty.
IMDS_TOKEN_TTL_SECONDS=60
# Bounds on the self-terminate attempt below, so a hung endpoint cannot delay the shutdown fallback.
# IMDS is link-local and answers in milliseconds; the EC2 call crosses the internet. The second is
# overridable purely so the test suite can drive the timeout path in seconds rather than in minutes.
IMDS_CALL_MAX_SECONDS=5
TERMINATE_CALL_MAX_SECONDS=${IDLE_WATCHDOG_TERMINATE_CALL_MAX_SECONDS:-20}

# Where the arm-time verdict goes so it outlives the box. Empty is a supported configuration and
# means "publish nowhere", not an error: this is a killswitch, and a missing S3 prefix may not stop a
# rented box from being watched. The bound is separate from the terminate one because this call
# happens while the box is healthy rather than while it is being torn down, and both are overridable
# purely so the test suite can drive the timeout path in seconds rather than in minutes.
STATUS_S3_DEST=${IDLE_WATCHDOG_S3_DEST:-}
ARM_RECORD=${IDLE_WATCHDOG_ARM_RECORD:-/var/log/idle-watchdog-arm.txt}
STATUS_UPLOAD_MAX_SECONDS=${IDLE_WATCHDOG_STATUS_UPLOAD_MAX_SECONDS:-20}

# Resolved once, and logged at arm time, because root's PATH under `nohup` from user-data is not the
# interactive PATH anyone tested against, and this repo has lost an environment to the resource
# limiter and to tmux not inheriting exported variables before. Falling back to the bare name keeps a
# genuinely CLI-less box behaving as it did, failing at the call with a familiar 127.
AWS_CLI=$(command -v aws || echo aws)

while [ $# -gt 0 ]; do
  case "$1" in
    --minutes)
      IDLE_MINUTES=$2
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS=$2
      shift 2
      ;;
    --user)
      RUN_USER=$2
      shift 2
      ;;
    --work-pattern)
      WORK_PATTERN=$2
      shift 2
      ;;
    --done-marker)
      DONE_MARKER=$2
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# While arm_recording is on, every logged line is also accumulated into ARM_RECORD, so the record is
# by construction the arm-time prefix of this log rather than a second wording of it that could
# disagree. It is switched off before the watch begins: left on, the "arm record" would grow into the
# whole log for the life of the box, which is both a slow /var/log leak on a multi-day box and not an
# arm record.
arm_recording=no

log() {
  local line
  line="$(date -Is) idle-watchdog: $*"
  echo "$line"
  if [ "$arm_recording" = yes ]; then
    echo "$line" >>"$ARM_RECORD"
  fi
}

# Derived after the argument loop, so the escape hatch cannot point at a different user's home than
# the one being watched. A --user with no passwd entry is fatal rather than assumed: its operator
# would have no way to suppress the watchdog, and finding that out is the moment the box dies.
run_user_home=$(getent passwd "$RUN_USER" | cut -d: -f6)
if [ -z "$run_user_home" ]; then
  echo "no passwd entry for --user $RUN_USER, so the KEEPALIVE escape hatch would have no path" >&2
  exit 2
fi
KEEPALIVE="$run_user_home/KEEPALIVE"

# `pane_current_command` over every pane of every session, rather than the existence of a session.
#
# The quiet list is shells and nothing else -- what a pane sits at when nothing is running, `-bash`
# being how a login shell names itself and a `tmux` pane being a nested client rather than a job. It
# only has to cover shells because everything else counts as work, and the two ways of being wrong
# are not symmetric: a job this list swallows means a box held minutes longer, while a shell missing
# from it means a box terminated under a running job. A pane tmux will not name counts as work for the
# reason an unreadable GPU does -- unknown is not idle.
tmux_pane_working() {
  local pane
  while read -r pane; do
    case "$pane" in
      "")
        echo "a tmux pane whose command tmux will not state; unknown is not idle"
        return 0
        ;;
      bash | -bash | sh | -sh | zsh | -zsh | fish | -fish | dash | tmux) ;;
      *)
        echo "tmux pane running ${pane}"
        return 0
        ;;
    esac
  done < <(sudo -u "$RUN_USER" tmux list-panes -a -F '#{pane_current_command}' 2>/dev/null)
  return 1
}

work_in_flight() {
  # Any one of these means the box is still ours to keep.
  if tmux_pane_working; then
    return 0
  fi
  if pgrep -u "$RUN_USER" -f "$WORK_PATTERN" >/dev/null 2>&1; then
    echo "work process present"
    return 0
  fi
  gpu_in_use
}

# Every device, not row 0: `head -1` read device 0 alone, so a two-card box training on device 1
# reported a quiet GPU and got terminated mid-run. That is the defect gpu_preflight.py was already
# fixed for, and the boxes this is armed on are the multi-GPU ones (a 27B needs two cards).
#
# Busiest device rather than the sum, in both directions: a sum would let eight idle drivers at a few
# hundred MiB each add up past the threshold and hold a box forever, and the question here is only
# whether some card is working. A device whose usage nvidia-smi will not state (`[N/A]`, what it
# prints for one it can see but not inspect) counts as busy -- unknown is not idle, and the cheap
# direction to err is holding a box, since idle minutes cost dollars and a lost run costs the run.
#
# fu_train: an ABSENT nvidia-smi is a third case, and it reads not-busy rather than busy. The loop
# reads nothing, `busiest` stays 0, and this returns 1. That is deliberate and it is not the `[N/A]`
# rule in reverse: `[N/A]` means a card exists and cannot be inspected, while a missing binary means
# there is no card to be working, so treating it as busy would leave a CPU box with two signals and a
# permanent hold. Consequence to know: on the CPU boxes this workstream also rents (the lifecycle
# smoke test, any CPU-only fit) the GPU arm is vacuous and tmux plus the work pattern carry the whole
# decision. Size --minutes against those two alone there.
#
# slate 2026-09-01: a PRESENT nvidia-smi whose INVOCATION fails is a fourth case, and it also reads
# not-busy --- loudly, on stderr, every poll. The GPU AMIs bake the binary into the image, so a CPU box
# rented on one (two GPU-less boxes that night; the headers of the copies that paid for it carry the
# ids) has an nvidia-smi with no device behind it, answering every query with a complaint on STDOUT.
# The per-line rule
# below read that complaint as "a card exists and cannot be inspected" and held both boxes busy on
# every poll: nothing was ever logged (a working poll was silent by design at the time), the
# agenda-complete marker is only consulted on a quiet poll so it never fired, and both boxes idled at
# on-demand rates until an operator collected them by hand. The per-line rule was written for `[N/A]`
# under a WORKING invocation --- exit 0, a device enumerated, one field unreadable --- and it keeps
# exactly that scope. A failed invocation is not a statement about any card, so it may not be a hold:
# it reads as "no GPU signal" and tmux plus the work pattern carry the decision, same as an absent
# binary. The diagnostic goes to stderr because this function's stdout is work_in_flight's captured
# reason, which is discarded on a not-busy poll --- stderr is what lands in the log either way.
gpu_in_use() {
  local answer status used busiest=0
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  answer=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "$(date -Is) idle-watchdog: gpu probe: nvidia-smi exited ${status}" \
      "($(printf '%s' "$answer" | tr '\n' ' ')); no readable device, so the GPU signal is vacuous" \
      "and tmux plus the work pattern carry the decision" >&2
    return 1
  fi
  while read -r used; do
    case "$used" in
      "") ;;
      *[!0-9]*)
        echo "GPU reporting unreadable memory.used (${used}); unknown usage is not idle"
        return 0
        ;;
      *) [ "$used" -gt "$busiest" ] && busiest="$used" ;;
    esac
  done <<<"$answer"
  if [ "$busiest" -gt "$GPU_BUSY_MIB" ]; then
    echo "a GPU is holding ${busiest} MiB"
    return 0
  fi
  return 1
}

# Everything below serves one purpose: asking EC2 to terminate this box rather than only halting it.
#
# Both killswitch layers used to die purely by `shutdown -h`, leaning on the launch's
# InstanceInitiatedShutdownBehavior. That is one setting away from a box that merely STOPS, and a
# stopped box goes on billing its EBS root disk forever while every signal reads "the killswitch
# fired". Asking EC2 directly releases the instance whatever the launch was configured to do, and it
# is the only variant that leaves a record a later reader can find: `terminate-instances` is an API
# call anyone can go and look up afterwards, where a halt is visible only in a log on a disk this
# script exists to destroy.
#
# Additive, never a substitute. The teardown runs `shutdown` regardless of what the API call did and
# deliberately ignores its result: a killswitch may not acquire a new way to fail. The redundancy
# costs nothing (both paths converge on `terminated`) and it is not a race, because the API call
# takes effect server-side as soon as it returns, well before the OS finishes halting. Failure is
# expected until the instance role carries ec2:TerminateInstances, and a box launched before that
# grant lands has to die on the fallback exactly as boxes always have.
#
# `--noproxy '*'` because an http_proxy in root's environment would otherwise send this link-local
# request through a proxy, and a proxy that answers 200 with a page of its own gets that page as far
# as the shape checks below -- which is a network appliance choosing which instance id a destructive
# call receives.
imds_metadata() {
  local token=$1 path=$2
  curl -sf --noproxy '*' --max-time "$IMDS_CALL_MAX_SECONDS" \
    -H "X-aws-ec2-metadata-token: $token" "$IMDS_BASE/meta-data/$path"
}

# The id is handed straight to a destructive API call and it arrived over the network. `curl -f`
# rejects a non-2xx, but a proxy or captive portal answering 200 with a page of HTML survives that,
# so the id is shape-checked before it reaches the call.
is_instance_id() {
  local body
  case "$1" in
    i-*) body=${1#i-} ;;
    *) return 1 ;;
  esac
  case "$body" in
    "" | *[!0-9a-f]*) return 1 ;;
  esac
  [ "${#body}" -eq 8 ] || [ "${#body}" -eq 17 ]
}

# The other half of the same defence: the region is interpolated into the call too, so a captive
# portal's HTML would otherwise arrive there instead.
is_region() {
  case "$1" in
    "" | *[!a-z0-9-]*) return 1 ;;
  esac
  return 0
}

# One place that reads this box's own identity, shared by the arm-time probe and the teardown, so
# there is a single implementation that can be wrong rather than two that can disagree.
#
# The answers come back in globals rather than on stdout: a reader invoked as
# ``x=$(reader)`` runs its failure path inside a command substitution, where it cannot stop its
# caller. The problem string is left for the caller to log, because the two callers report a failure
# to resolve under different headings and a shared wording would fit neither.
IDENTITY_INSTANCE=""
IDENTITY_REGION=""
IDENTITY_PROBLEM=""

resolve_instance_identity() {
  local token status
  IDENTITY_INSTANCE=""
  IDENTITY_REGION=""
  IDENTITY_PROBLEM=""

  # IMDSv2 only: these boxes launch with HttpTokens=required, so a tokenless GET answers 401 and every
  # value below would come back empty rather than loudly wrong.
  token=$(curl -sf --noproxy '*' --max-time "$IMDS_CALL_MAX_SECONDS" -X PUT \
    -H "X-aws-ec2-metadata-token-ttl-seconds: $IMDS_TOKEN_TTL_SECONDS" "$IMDS_BASE/api/token")
  status=$?
  if [ "$status" -ne 0 ] || [ -z "$token" ]; then
    IDENTITY_PROBLEM="no IMDSv2 token (curl exit ${status})"
    return 1
  fi

  IDENTITY_INSTANCE=$(imds_metadata "$token" instance-id)
  IDENTITY_REGION=$(imds_metadata "$token" placement/region)
  if ! is_instance_id "$IDENTITY_INSTANCE" || ! is_region "$IDENTITY_REGION"; then
    IDENTITY_PROBLEM="IMDS gave instance '${IDENTITY_INSTANCE}' region '${IDENTITY_REGION}'"
    IDENTITY_INSTANCE=""
    IDENTITY_REGION=""
    return 1
  fi
  return 0
}

# Answer at ARM time whether the grant is actually live, because "the policy is attached" and "EC2
# populates ec2:InstanceProfile and ec2:ResourceTag in a live request context" are different claims
# and the gap between them is invisible later: a denied teardown falls back to `shutdown` and logs a
# perfectly benign refusal, while the termination this whole path exists to request is silently never
# made and the box's fate rests entirely on a launch setting nothing here can read. A dry run answers
# it in one call, on every box, while there is still a night left to do something about it.
#
# --dry-run is what makes this safe to run against a healthy box that is about to do real work: AWS
# completes the authorization check and does not perform the operation. It sits immediately after the
# subcommand and must stay there. This is the one call in this script where a typo destroys a running
# experiment, so nothing may come between the two.
#
# The verdict reads the response text and never the exit status, because a dry run that PASSES exits
# NON-ZERO -- it reports DryRunOperation as an error -- so keying on $? inverts the answer, reporting
# a working grant as broken and a real denial as fine.
#
# It gates nothing. Every path returns success, and an inconclusive answer is logged and left alone:
# a box must never refuse to run its experiment because a permission check could not make up its mind.
probe_terminate_grant() {
  local response
  if ! resolve_instance_identity; then
    log "GRANT PROBE INDETERMINATE: ${IDENTITY_PROBLEM}; the teardown will find out for itself."
    return 0
  fi

  response=$(mktemp /tmp/idle-watchdog-grant-probe.XXXXXX)
  if [ ! -f "$response" ]; then
    log "GRANT PROBE INDETERMINATE: no scratch file to read the dry run's answer out of."
    return 0
  fi
  # Into a file rather than a command substitution, for the reason the teardown avoids one: the
  # substitution would wait on its pipe rather than on the child `timeout` killed, and a probe that
  # hangs delays arming, which is the one thing an arm-time check must not do.
  timeout --kill-after=5s "$TERMINATE_CALL_MAX_SECONDS" "$AWS_CLI" ec2 terminate-instances \
    --dry-run --instance-ids "$IDENTITY_INSTANCE" --region "$IDENTITY_REGION" >"$response" 2>&1

  if grep -q DryRunOperation "$response"; then
    log "GRANT PROBE LIVE: ec2:TerminateInstances is authorized for ${IDENTITY_INSTANCE} in ${IDENTITY_REGION}."
    log "So this box's teardown releases the instance itself rather than relying on the launch setting."
  elif grep -q UnauthorizedOperation "$response"; then
    log "GRANT PROBE MISSING: ec2:TerminateInstances is DENIED for ${IDENTITY_INSTANCE}."
    log "This box dies on the shutdown fallback alone, so it terminates only if the launch says so."
  else
    log "GRANT PROBE INDETERMINATE: the dry run named neither DryRunOperation nor UnauthorizedOperation."
    log "It answered: $(tr '\n' ' ' <"$response")"
  fi
  rm -f "$response"
  return 0
}

# Opened before the first line is logged, so the record holds the whole arm-time window and not
# whatever happened to come after some later switch-on. A failure to create it degrades to "no
# record" and one complaint, rather than to one redirection error per log call for the life of the
# box: the watchdog is routinely pointed at a path only root can write, and it must still watch.
open_arm_record() {
  if [ -z "$ARM_RECORD" ]; then
    return 0
  fi
  if : >"$ARM_RECORD" 2>/dev/null; then
    arm_recording=yes
    return 0
  fi
  log "arm record UNWRITABLE at ${ARM_RECORD}; the arm-time verdict stays in this log only."
  ARM_RECORD=""
  return 0
}

# Put the arm-time verdict somewhere that survives this box, which is the only thing that makes the
# grant probe above readable after the fact. Runs once, before the watch begins, so an abrupt death
# hours later -- a spot reclaim, a panic, the teardown this script performs -- cannot take it.
#
# Every path returns success and none of them may cost the box its watching, so the whole function is
# best-effort: bounded by `timeout`, output redirected to the log rather than captured for the reason
# api_self_terminate spells out, and no branch that can abort the arm.
#
# The identity comes from the globals the probe already populated, so there is no second IMDS round
# trip. Empty means the probe went INDETERMINATE, and then the record stays local deliberately: with
# IMDS unreachable the CLI has no instance credentials either, so an upload could not have worked and
# a key built from an empty id would only litter the bucket. The cost is that those boxes are knowable
# only by the ABSENCE of an object, so the denominator has to come from describe-instances.
publish_arm_record() {
  local key status
  if [ -z "$STATUS_S3_DEST" ]; then
    log "arm record UNCONFIGURED: set IDLE_WATCHDOG_S3_DEST to an s3:// prefix to keep this verdict."
    return 0
  fi
  # A destination that is not an s3:// URL would make the copy below a local file copy that exits 0,
  # and this function would report PUBLISHED for a verdict that never left the box. That is the exact
  # reassuring-message failure the grant probe was written to avoid, so it is refused by shape.
  case "$STATUS_S3_DEST" in
    s3://*) ;;
    *)
      log "arm record UNAVAILABLE: IDLE_WATCHDOG_S3_DEST is '${STATUS_S3_DEST}', not an s3:// URL."
      return 0
      ;;
  esac
  if [ -z "$ARM_RECORD" ]; then
    log "arm record UNAVAILABLE: no local record was written, so there is nothing to upload."
    return 0
  fi
  if [ -z "$IDENTITY_INSTANCE" ] || [ -z "$IDENTITY_REGION" ]; then
    log "arm record UNAVAILABLE: this box never resolved its own id, so it has no key to write under."
    return 0
  fi

  # Keyed by instance id, because a key shared across a run's boxes is silently a REPLACE: whichever
  # box arms last is the only verdict left, and the ones whose grant is broken are lost first.
  key="${STATUS_S3_DEST%/}/killswitch-arm/${IDENTITY_INSTANCE}.txt"
  timeout --kill-after=5s "$STATUS_UPLOAD_MAX_SECONDS" "$AWS_CLI" s3 cp "$ARM_RECORD" "$key" \
    --region "$IDENTITY_REGION" --only-show-errors >&2
  status=$?
  case "$status" in
    0)
      log "arm record PUBLISHED to ${key}; this box's verdict now outlives it."
      ;;
    124 | 137)
      log "arm record CUT OFF at the ${STATUS_UPLOAD_MAX_SECONDS}s bound (exit ${status})."
      ;;
    *)
      log "arm record FAILED (exit ${status}) for ${key}; the cause is on the line(s) above."
      ;;
  esac
  return 0
}

api_self_terminate() {
  local instance region status
  if ! resolve_instance_identity; then
    log "API terminate UNAVAILABLE: ${IDENTITY_PROBLEM}; falling back to shutdown."
    return 1
  fi
  instance=$IDENTITY_INSTANCE
  region=$IDENTITY_REGION

  log "API terminate: calling ec2:TerminateInstances on ${instance} in ${region}."
  # The CLI's own output goes to this log rather than into a variable, which is what makes the
  # `timeout` an actual bound: a command substitution waits for its pipe to close, not for the child
  # `timeout` killed, so a single grandchild holding that pipe would stall the watchdog right past
  # the bound meant to protect it. --kill-after covers the other half, a client ignoring the TERM.
  timeout --kill-after=5s "$TERMINATE_CALL_MAX_SECONDS" "$AWS_CLI" ec2 terminate-instances \
    --region "$region" --instance-ids "$instance" --output text >&2
  status=$?
  if [ "$status" -eq 0 ]; then
    log "API terminate SUCCEEDED for ${instance}: EC2 has accepted the termination."
    return 0
  fi
  # 124 from `timeout`, or 137 once it had to escalate to KILL: either way the call was cut off
  # rather than answered, which is a different fault from a refusal (that one usually means no
  # grant) and would otherwise read as one.
  case "$status" in
    124 | 137)
      log "API terminate CUT OFF at the ${TERMINATE_CALL_MAX_SECONDS}s bound (exit ${status})."
      ;;
    *)
      log "API terminate FAILED (exit ${status}); the cause is on the line(s) above."
      ;;
  esac
  log "Falling back to shutdown; the box still dies, but only terminates if the launch says so."
  return 1
}

# fu_train: the teardown, lifted out of the loop body into a function so the two triggers below (the
# idle threshold, and the agenda-complete marker) run the SAME critical path rather than two copies of
# it that can drift. Semantics are unchanged from the original, line for line, and all three of the
# things that are load-bearing about them stay load-bearing:
#
#   * `api_self_terminate || true` --- the result is deliberately discarded. The API call is the path
#     that releases the instance outright; the `shutdown` below is what has always actually killed
#     these boxes. `|| true` is not redundant despite the absent `set -e`: it is what
#     keeps a later `set -e` from turning a refused API call into a watchdog that exits before ever
#     reaching the shutdown.
#   * The `shutdown` is unconditional, so the API path is additive and never a substitute.
#   * A FAILED shutdown returns rather than exiting, so the caller stays in its poll loop. Exiting here
#     would retire layer 2 on a box that is still running, with nothing left watching.
teardown() {
  local why=$1 shutdown_status=0
  log "TERMINATING: ${why}"
  log "This is layer 2 of the killswitch. Layer 1 (max-lifetime dead-man) remains the backstop."
  api_self_terminate || true
  # Instance-initiated shutdown, so the launch MUST set InstanceInitiatedShutdownBehavior to
  # terminate -- nothing here can check that, and a stop keeps billing the EBS root.
  shutdown -h now "idle-watchdog: ${why}" || shutdown_status=$?
  if [ "$shutdown_status" -eq 0 ]; then
    exit 0
  fi
  log "SHUTDOWN FAILED (exit ${shutdown_status}): layer 2 could NOT terminate this box."
  log "Retrying every ${POLL_SECONDS}s. Check that the watchdog runs as root."
  return 1
}

# The threshold is wall clock, tracked as the epoch stamp of the first idle poll. Counting polls
# instead made the threshold mean minutes only at the default 60s poll: --poll-seconds 30 --minutes
# 20 killed a box after 10 real minutes, mid-agenda.
idle_limit_seconds=$((IDLE_MINUTES * 60))
idle_since=""

open_arm_record
log "armed: terminate after ${IDLE_MINUTES} idle minutes (poll ${POLL_SECONDS}s, user ${RUN_USER})"
# Logged because a work pattern that does not match its own job is invisible otherwise: the box just
# looks idle while it works, and layer 2 kills it mid-run. This line is the record of what was armed.
log "work pattern: ${WORK_PATTERN}"
if [ -n "$DONE_MARKER" ]; then
  log "agenda-complete trigger: ${DONE_MARKER} (tears the box down on the next quiet poll)"
else
  log "agenda-complete trigger: none; the ${IDLE_MINUTES}-minute threshold is the only one"
fi
log "escape hatch: touch $KEEPALIVE to hold this box"
log "aws CLI resolved to: ${AWS_CLI}"
# After the two lines above rather than between them, so that a reader tailing this log sees the
# watchdog armed and configured before the probe spends its bounded network time.
probe_terminate_grant
# Before the loop, not after some poll: the first idle line is what pushes the verdict out of every
# launcher's `tail -N` heartbeat, so publishing has to happen while the verdict is still the tail.
publish_arm_record
arm_recording=no
while true; do
  now=$(date +%s)
  if [ -f "$KEEPALIVE" ]; then
    log "SUPPRESSED by $KEEPALIVE; not counting idle time"
    idle_since=""
  elif reason=$(work_in_flight); then
    if [ -n "$idle_since" ]; then
      log "work resumed (${reason}); idle timer reset after $((now - idle_since))s"
    else
      # slate 2026-09-01: one line per poll even while work is in flight. A working poll used to be
      # silent, which made "watching and satisfied" and "wrongly holding since arm time" the same
      # empty log --- on the two boxes in the header, operators had to pgrep the process to tell the
      # difference. Every poll now writes exactly one line (this, SUPPRESSED, idle, or TERMINATING),
      # so a silent minute IS the alarm.
      log "work in flight: ${reason}"
    fi
    idle_since=""
  else
    [ -n "$idle_since" ] || idle_since=$now
    idle_seconds=$((now - idle_since))
    # fu_train: the agenda-complete trigger, checked BEFORE the idle threshold and only ever reached
    # when nothing suppresses the watchdog and no work is in flight. The job driver touches this file
    # as its last act, after the artifact sync has already returned, so its presence means "everything
    # this box was rented for is done and shipped" -- and then waiting out a further idle window is
    # pure billed idleness. It can only ever make the box die SOONER, never later, and KEEPALIVE still
    # outranks it, so `touch ~/KEEPALIVE` holds even a finished box for debugging.
    #
    # This is also what makes the killswitch a watched mechanism rather than a reassuring message: on
    # the normal, successful path the box is torn down BY layer 2, so every run exercises it.
    if [ -n "$DONE_MARKER" ] && [ -f "$DONE_MARKER" ]; then
      teardown "agenda complete (${DONE_MARKER} present) with no work in flight"
    else
      log "idle ${idle_seconds}s/${idle_limit_seconds}s (no working tmux pane, no work process, GPU quiet)"
      if [ "$idle_seconds" -ge "$idle_limit_seconds" ]; then
        teardown "idle for ${idle_seconds}s (limit ${idle_limit_seconds}s) with no work in flight"
      fi
    fi
  fi
  sleep "$POLL_SECONDS"
done
