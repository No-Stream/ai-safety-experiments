#!/bin/bash
# Reference runner for a rented GPU box (scripts/kit_reference/README.md). Root context, started by
# the reference user-data out of the digest-verified tree, so it needs no pin of its own. It carries
# every operating rule AGENTS.md states for a box, in this order:
#
#   1. refuse to run on a box that cannot end itself (shutdown behaviour read back from the API);
#   2. a heartbeat unit outside tmux, every 5 minutes, to BOTH status keys;
#   3. hand the agenda to the run user inside tmux (a tmux pane is one of the watchdog's work signals),
#      and touch the agenda-complete marker at once if that hand-off fails, since nothing else would;
#   4. hold an on-demand landing on a priced shape (p5) for a human's ratification marker;
#   5. resume first: restore what already exists under the run prefix -- corpus, run directory, then
#      checkpoints newest-first until a complete one is on disk -- before doing anything new, and stop
#      when a restore fails rather than start fresh beside a banked checkpoint;
#   6. train as games.arm_sequence.LAUNCH_RECIPE says: the sweep alone (skipped when a corpus already
#      resolves), the corpus resolved and exported, then the consuming stages, each invocation behind
#      a CPU plan gate; a checkpoint every step with the trainer's own per-save S3 sync, the model
#      cached and then the hub marked offline, the pace guard's plan figure taken from a parameter;
#   7. eval cells that restore from and sync to their own S3 prefix, resumed rather than re-run, with a
#      closing --summarise-only pass that refuses an unfinished cell;
#   8. one exit path: record the exit, sync the logs, record whether that sync landed, touch the
#      agenda-complete marker LAST.
#
# To customise: copy it into the working tree beside your kit's template copy (an untracked path
# ships in the archive; a gitignored one does not), add the experiment's registered pins where the
# comments say so, and keep every marker scripts/kit_reference/markers.tsv names -- the launcher
# compares a kit's runner against them at preflight and warns on each one that is gone.
#
# tests/test_kit_reference.py runs this whole file against stubbed binaries inside a scratch tree,
# which is why the two root-owned paths are parameters: BOX_LOG_DIR (/var/log) and BOX_BIN_DIR
# (/usr/local/bin). A box never sets them.
set -ux
BOX_LOG_DIR=${BOX_LOG_DIR:-/var/log}
BOX_BIN_DIR=${BOX_BIN_DIR:-/usr/local/bin}
exec >>"$BOX_LOG_DIR/box-bootstrap.log" 2>&1

# --- what the user-data hands over -------------------------------------------------------------------
RUN_NAME=${RUN_NAME:?the user-data must export RUN_NAME}
RUN_PREFIX=${RUN_PREFIX:?the user-data must export RUN_PREFIX (s3://<bucket>/<base>/<run-name>)}
S3_BASE=${S3_BASE:?the user-data must export S3_BASE (s3://<bucket>/<base>, which holds status/)}
S3_REGION=${S3_REGION:?the user-data must export S3_REGION}
ARM=${ARM:?the user-data must export ARM (a games arm name)}
MODEL=${MODEL:?the user-data must export MODEL (a Hugging Face model id)}
MAX_STEPS=${MAX_STEPS:?the user-data must export MAX_STEPS}
EVAL_STEPS=${EVAL_STEPS:?the user-data must export EVAL_STEPS (comma-separated steps, e.g. 70,0)}
PLAN_STEP_MINUTES=${PLAN_STEP_MINUTES:?the user-data must export PLAN_STEP_MINUTES (the measured minutes per step of this shape)}
DONE_MARKER=${DONE_MARKER:?the user-data must export DONE_MARKER (the path the watchdog was armed with)}
REPO=${REPO:?the user-data must export REPO (the verified extract directory)}
# Optional, with defaults; a kit overrides them on the user-data's environment line.
RUN_USER=${RUN_USER:-ubuntu}
# The stages games.arm_sequence runs, sweep included; the agenda splits the sweep off into its own
# invocation, as the plan requires (see the training leg).
TRAIN_STAGES=${TRAIN_STAGES:-sweep,arm}
# Empty means games.run_evals's own default sections; a kit that wants fewer names them here.
BATTERY_SECTIONS=${BATTERY_SECTIONS:-}
# p5, p5e and p5en are all priced H100/H200 shapes, hence no dot after the family.
HOLD_INSTANCE_TYPES=${HOLD_INSTANCE_TYPES:-^p5}
HOLD_MINUTES=${HOLD_MINUTES:-150}
HF_CACHE_DIR=${HF_CACHE_DIR:-/opt/dlami/nvme/hf}

HOME_DIR=$(getent passwd "$RUN_USER" | cut -d: -f6)
IID=$(cat "$HOME_DIR/IID")
CHAIN_STATE=$HOME_DIR/CHAIN_STATE
RUN_LOG=$HOME_DIR/$RUN_NAME.log
RUN_ENV=$HOME_DIR/run.env

imds() { # imds <meta-data path>; IMDSv2 only, since HttpTokens=required answers a tokenless GET with 401
  local token
  token=$(curl -sX PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
  curl -sf -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$1"
}

# --- the box must be able to end itself --------------------------------------------------------------
# The launcher reads this attribute back after launch and terminates a box that says stop; this is the
# same check from the inside, for a box launched some other way. A stop-behaviour box keeps billing its
# EBS root and neither killswitch layer can end it, so it halts on a short fuse. Unreadable (the
# instance role may lack ec2:DescribeInstanceAttribute) is a WARN, not a halt: the launcher's check ran.
BOX_REGION=$(imds placement/region)
behaviour=$(aws ec2 describe-instance-attribute --instance-id "$IID" --region "$BOX_REGION" \
  --attribute instanceInitiatedShutdownBehavior \
  --query InstanceInitiatedShutdownBehavior.Value --output text 2>&1)
case $behaviour in
  terminate) echo "shutdown behaviour: terminate" ;;
  stop)
    echo "FATAL: shutdown behaviour is stop, so no killswitch layer can end this box; halting"
    echo "exited rc=1 reason=shutdown-behaviour-stop job=$RUN_NAME at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$CHAIN_STATE"
    shutdown -h +5 "$RUN_NAME: shutdown behaviour is stop"
    exit 1
    ;;
  *) echo "WARN: shutdown behaviour unreadable from the box ($behaviour); the launcher's check stands" ;;
esac

nvidia-smi
df -h /
mkdir -p "$HF_CACHE_DIR"
chown "$RUN_USER:$RUN_USER" "$HF_CACHE_DIR"

# --- the run's environment, in a file the heartbeat and the agenda both source ------------------------
# A file rather than a command line: tmux hands its command to sh -c, which would strip the backslash
# out of a regex like HOLD_INSTANCE_TYPES, and a tmux server that already exists would hand the command
# its own stale environment. printf %q quotes each value so the file sources back to exactly what was
# set here.
{
  for name in RUN_NAME RUN_PREFIX S3_BASE S3_REGION ARM MODEL MAX_STEPS EVAL_STEPS PLAN_STEP_MINUTES \
    DONE_MARKER REPO RUN_USER TRAIN_STAGES BATTERY_SECTIONS HOLD_INSTANCE_TYPES HOLD_MINUTES \
    HF_CACHE_DIR HOME_DIR IID CHAIN_STATE RUN_LOG BOX_LOG_DIR BOX_BIN_DIR; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done
} >"$RUN_ENV"

# --- heartbeat: root unit outside tmux, every 300 s, to BOTH keys ------------------------------------
# The launcher tags HeartbeatS3=<base>/status/<instance>.txt, so writing that key keeps the tag honest;
# <run-prefix>/status/<instance>.txt is the copy a reader of one run finds without knowing the tag.
# The reward and purity greps read the arm log the stage writes LIVE under
# artifacts/games/logs/; the run directory's logs/ copy exists only after training, so a grep there
# reads nothing for the whole run (the v2 and softpen kits shipped exactly that).
cat >"$BOX_BIN_DIR/box-heartbeat.sh" <<'HEARTBEAT'
#!/bin/bash
# box-heartbeat.sh <run.env>: one status object every 300 s to both status keys.
set -u
# shellcheck disable=SC1090
source "$1"
STATUS=$BOX_LOG_DIR/box-status.txt
work_state() {
  if sudo -u "$RUN_USER" tmux ls >/dev/null 2>&1; then echo "RUNNING (tmux present)"
  elif [ -s "$CHAIN_STATE" ]; then echo "$(cat "$CHAIN_STATE") (no tmux)"
  else echo "ABSENT (no tmux and no recorded exit: died without recording)"; fi
}
while true; do
  {
    date -u
    echo "instance=$IID run=$RUN_NAME tree=$(cat "$REPO/SHIP_TREE" 2>/dev/null) code=$(cat "$REPO/GIT_SHA" 2>/dev/null)"
    echo "work=$(work_state)"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    echo "--- checkpoints on disk"
    find "$REPO/artifacts/games/runs" -mindepth 2 -maxdepth 2 -name 'checkpoint-*' -type d 2>/dev/null | sort -V | tail -6
    find "$REPO/artifacts/games/runs" -maxdepth 2 -name train_summary.json 2>/dev/null | sed 's/^/TRAINING COMPLETE: /'
    echo "--- reward, purity and trim tail from the live arm log (purity near 1.0 in the first steps = kill and diagnose)"
    grep -ho "frac_groups_pure[^,]*" "$REPO"/artifacts/games/logs/arm-*.log 2>/dev/null | tail -3
    grep -ho "'reward': [^,]*" "$REPO"/artifacts/games/logs/arm-*.log 2>/dev/null | tail -3
    grep -h "padding trim:" "$REPO"/artifacts/games/logs/arm-*.log 2>/dev/null | tail -1 | cut -c1-200
    echo "--- eval cells (summary = complete; trace alone = in progress or died)"
    find "$REPO/artifacts/games/evals/$RUN_NAME" -name 'step-*.summary.json' 2>/dev/null | sort
    find "$REPO/artifacts/games/evals/$RUN_NAME" -name 'step-*.jsonl' 2>/dev/null | sort | while read -r trace; do
      echo "$trace records=$(wc -l <"$trace")"
    done
    echo "--- run log"
    tail -10 "$RUN_LOG" 2>/dev/null | cut -c1-200
    echo "--- watchdog"
    tail -2 "$BOX_LOG_DIR/idle-watchdog.log" 2>/dev/null
  } >"$STATUS" 2>&1
  aws s3 cp "$STATUS" "$S3_BASE/status/$IID.txt" --region "$S3_REGION" --only-show-errors
  aws s3 cp "$STATUS" "$RUN_PREFIX/status/$IID.txt" --region "$S3_REGION" --only-show-errors
  sleep 300
done
HEARTBEAT
chmod +x "$BOX_BIN_DIR/box-heartbeat.sh"
systemd-run --unit=box-heartbeat --collect /bin/bash -c \
  "$BOX_BIN_DIR/box-heartbeat.sh '$RUN_ENV' >>$BOX_LOG_DIR/heartbeat.log 2>&1"
systemctl is-active box-heartbeat || true

# --- the agenda, as the run user inside tmux ---------------------------------------------------------
cat >"$HOME_DIR/bootstrap-and-run.sh" <<'BOOTSTRAP'
#!/bin/bash
# Exit codes are read from commands directly, never through a pipe, and every path out goes through
# finish(), so the agenda-complete marker is touched exactly once and only after the last upload.
set -ux
# shellcheck disable=SC1091
source "$HOME/run.env"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
state() { echo "$1 job=$RUN_NAME at=$(stamp)" >"$CHAIN_STATE"; }
LOGDIR=$REPO/artifacts/games/logs
mkdir -p "$LOGDIR"
sync_logs() { # returns the status of an upload that failed, so finish() can record it
  local rc=0
  aws s3 sync "$LOGDIR/" "$RUN_PREFIX/logs/" --only-show-errors --region "$S3_REGION" || rc=$?
  aws s3 cp "$RUN_LOG" "$RUN_PREFIX/logs/$IID.log" --only-show-errors --region "$S3_REGION" || rc=$?
  return "$rc"
}
# The one exit path. The marker comes AFTER the last upload has returned: the watchdog tears the box
# down on its next quiet poll once the marker exists, and a marker touched with a sync in flight would
# lose that sync. `aws s3` is in the watchdog's work pattern, so an in-flight sync still holds the box.
# The recorded state says whether that last upload landed, because a failed one would otherwise be
# the one failure the box never reports.
finish() { # finish <rc> [reason]
  local rc=$1 reason=${2:-done} logs=synced
  state "exiting rc=$rc reason=$reason"
  sync_logs || logs="SYNC-FAILED rc=$?"
  state "exited rc=$rc reason=$reason logs=$logs"
  touch "$DONE_MARKER"
  exit "$rc"
}
fail() { # fail <reason>
  echo "FAILED: $1"
  finish 1 "$1"
}

cd "$HOME" || fail no-home
curl -LsSf https://astral.sh/uv/install.sh >"$HOME/uv-install.sh" || fail fetch-uv-installer
sh "$HOME/uv-install.sh" || fail install-uv
# shellcheck disable=SC1091
source "$HOME/.local/bin/env"
export HF_HOME=$HF_CACHE_DIR
cd "$REPO" || fail no-repo
uv sync --frozen --extra vllm >"$LOGDIR/uv-sync.log" 2>&1 || {
  tail -30 "$LOGDIR/uv-sync.log"
  fail uv-sync
}
export UV_NO_SYNC=1
GIT_SHA=$(cat "$REPO/GIT_SHA")
export GIT_SHA
avail=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$avail" -ge 40 ] || fail "disk: only ${avail}G free against the venv plus runs and logs"

# --- the experiment's environment. Registered knobs (estimator pin, completion budget, corpus sha)
#     go HERE in a kit copy, and each one also gets a line in the arm's plan gate below. ---------------
export GAMES_ARM_SEQ_ARM=$ARM
export GAMES_ARM_SEQ_MODEL=$MODEL
export GAMES_ARM_SEQ_MAX_STEPS=$MAX_STEPS
export GAMES_ARM_SEQ_SAVE_STEPS=1
export GAMES_ARM_SEQ_S3_DEST=$RUN_PREFIX/runs
export GAMES_PLAN_STEP_MINUTES=$PLAN_STEP_MINUTES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Derived at the point of use from the code that will write there, never restated: the trainer's
# S3 destination appends the arm and model tag to GAMES_ARM_SEQ_S3_DEST, and a restore aimed one
# level up nested the bank where the resolver never looked (2026-08-31, an arm restarted from step 1).
OUTDIR=$(uv run --frozen python -c 'from games.arm_sequence import arm_output_dir; print(arm_output_dir())') || fail derive-output-dir
S3_RUN_DIR=$(uv run --frozen python -c 'from games.arm_sequence import arm_s3_dest; print(arm_s3_dest())') || fail derive-s3-dest
SWEEP_DIR=$(uv run --frozen python -c 'from games.arm_sequence import sweep_dir; print(sweep_dir())') || fail derive-sweep-dir
EVALS=$REPO/artifacts/games/evals/$RUN_NAME
mkdir -p "$OUTDIR" "$SWEEP_DIR" "$EVALS"

# --- a priced shape is held, never auto-taken (AGENTS.md: a p5 landing is a cost decision) -----------
# On demand only; a spot box proceeds. The ratification marker is written by a human under the run's
# control/ prefix once the price delta has been read against the budget line. Until it appears the
# KEEPALIVE file holds the idle watchdog off; past HOLD_MINUTES the box stands down cleanly and the
# watchdog reaps it.
imds() { # imds <meta-data path>
  local token
  token=$(curl -sX PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
  curl -sf -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$1"
}
INSTANCE_TYPE=$(imds instance-type)
# IMDS answers on-demand or spot; the fallback covers an IMDS build without the path at all.
LIFECYCLE=$(imds instance-life-cycle || echo on-demand)
echo "market: instance-type=$INSTANCE_TYPE lifecycle=$LIFECYCLE"
if [ "$LIFECYCLE" != spot ] && printf '%s' "$INSTANCE_TYPE" | grep -Eq -e "$HOLD_INSTANCE_TYPES"; then
  RATIFIED_KEY=$RUN_PREFIX/control/ONDEMAND_RATIFIED
  if ! aws s3 ls "$RATIFIED_KEY" --region "$S3_REGION" >/dev/null 2>&1; then
    touch "$HOME/KEEPALIVE"
    state "awaiting-market-ratification instance-type=$INSTANCE_TYPE marker=$RATIFIED_KEY"
    ratified=0
    for _ in $(seq 1 $((HOLD_MINUTES / 5))); do
      sleep 300
      if aws s3 ls "$RATIFIED_KEY" --region "$S3_REGION" >/dev/null 2>&1; then
        ratified=1
        break
      fi
    done
    rm -f "$HOME/KEEPALIVE"
    [ "$ratified" = 1 ] || finish 0 market-ratification-timeout
  fi
  echo "on-demand $INSTANCE_TYPE ratified; proceeding"
fi

# --- resume first: what already exists under the run prefix comes down before anything new runs ------
# A relaunch of the identical launch IS the recovery. The corpus (banked at prep or swept by an
# earlier sitting) and the run directory's small files come first; then checkpoints newest-first
# until one is complete, because the trainer's resolver steps past a torn newest one, and a run whose
# only local checkpoint is torn would otherwise restart from step 1 beside a complete one still in
# S3. Older checkpoints stay in S3: nothing on the box reads them, and an eval cell pulls its own step.
# Every restore stops the run when it fails: a transient S3 error or a role without s3:ListBucket
# would otherwise look exactly like a fresh run, and the trainer would start from step 1 beside the
# banked checkpoint, then overwrite it with its own. Hence the listing goes through s3api, which
# exits 0 with nothing on a fresh prefix and non-zero on an error, where `aws s3 ls` does both as 1.
checkpoint_complete() { # checkpoint_complete <dir>; mirrors games.train.missing_checkpoint_files
  local dir=$1 name
  for name in trainer_state.json optimizer.pt scheduler.pt rng_state.pth; do
    [ -f "$dir/$name" ] || return 1
  done
  [ -f "$dir/adapter_model.safetensors" ] || [ -f "$dir/adapter_model.bin" ]
}
restore_checkpoint() { # restore_checkpoint <step>
  aws s3 sync "$S3_RUN_DIR/checkpoint-$1/" "$OUTDIR/checkpoint-$1/" --only-show-errors --region "$S3_REGION" \
    || fail "restore-checkpoint-$1 rc=$?"
}
banked_checkpoint_steps() { # newest first; non-zero when the listing itself failed
  local rest=${S3_RUN_DIR#s3://} listing
  listing=$(aws s3api list-objects-v2 --bucket "${rest%%/*}" --prefix "${rest#*/}/" --delimiter / \
    --query 'CommonPrefixes[].Prefix' --output text --region "$S3_REGION") || return "$?"
  printf '%s\n' "$listing" | tr '\t' '\n' | sed -n 's#.*/checkpoint-\([0-9][0-9]*\)/$#\1#p' | sort -rn
}
state "running leg=restore"
aws s3 sync "$RUN_PREFIX/select/" "$SWEEP_DIR/" --only-show-errors --region "$S3_REGION" || fail "restore-corpus rc=$?"
aws s3 sync "$S3_RUN_DIR/" "$OUTDIR/" --exclude 'checkpoint-*/*' --exclude 'incomplete-checkpoints/*' \
  --only-show-errors --region "$S3_REGION" || fail "restore-run-dir rc=$?"
banked_steps=$(banked_checkpoint_steps) || fail "restore-list-checkpoints rc=$?"
for step in $banked_steps; do
  restore_checkpoint "$step"
  if checkpoint_complete "$OUTDIR/checkpoint-$step"; then
    echo "restored complete checkpoint-$step; older checkpoints stay in S3"
    break
  fi
  echo "checkpoint-$step is torn (the trainer will set it aside); pulling the next older one"
done

# --- the model into the cache, then the hub offline: neither the sweep nor a save spends a round trip
uv run --frozen python -c "from huggingface_hub import snapshot_download; print('cached at', snapshot_download('$MODEL'))" \
  || fail model-prefetch
export HF_HUB_OFFLINE=1

# --- the training leg is the two-invocation recipe games.arm_sequence.LAUNCH_RECIPE prints ----------
# The sweep timestamps its corpus, so games.plans.refuse_sweep_beside_a_corpus_consumer refuses 'sweep'
# in one invocation with the stages that read it, and the plan's own default is the sweep alone. So:
# the sweep runs on its own, skipped when a corpus already resolves (banked at prep under
# <run-prefix>/select/, or swept by an earlier sitting and banked right after), then the corpus is
# resolved and exported, then the consuming stages run as one invocation. Each invocation sits behind
# a CPU plan gate. The arm's gate also checks the registered lines, and can only run once the corpus
# exists, so on a fresh box a registered-line slip is caught after the sweep -- whose corpus is banked
# by then, so the relaunch that fixes the kit skips it.
case ",$TRAIN_STAGES," in
  *,sweep,*) SWEEP_WANTED=1 ;;
  *) SWEEP_WANTED=0 ;;
esac
CONSUMER_STAGES=$(
  IFS=,
  for stage in $TRAIN_STAGES; do [ "$stage" = sweep ] || printf '%s,' "$stage"; done
)
CONSUMER_STAGES=${CONSUMER_STAGES%,}
[ -n "$CONSUMER_STAGES" ] || fail "TRAIN_STAGES=$TRAIN_STAGES names no stage that trains"
RESOLVE_LOG=$LOGDIR/resolve-corpus-$RUN_NAME.log
resolve_corpus() { uv run --frozen python -m games.arm_sequence --print-corpus 2>"$RESOLVE_LOG"; }
plan_gate() { # plan_gate <stages> <label> [registered ERE fragment...]
  # The plan resolves and prints what was registered before any GPU minute. Each fragment must appear
  # as a whole run of argv tokens: `--save-steps 1` does not pass under `--save-steps 10`.
  local stages=$1 label=$2 want planlog
  planlog=$LOGDIR/plan-$label-$RUN_NAME.txt
  shift 2
  if ! GAMES_ARM_SEQ_STAGES=$stages uv run --frozen python -m games.arm_sequence --print-plan >"$planlog" 2>&1; then
    tail -30 "$planlog"
    fail "plan-gate-$label"
  fi
  for want in "$@"; do
    if ! grep -qE -- "(^|[[:space:]])$want([[:space:]]|$)" "$planlog"; then
      tail -40 "$planlog"
      fail "plan-gate-$label-missing: $want"
    fi
  done
  aws s3 cp "$planlog" "$RUN_PREFIX/logs/plan-$label-$IID.txt" --only-show-errors --region "$S3_REGION"
}
run_stages() { # run_stages <stages> <label>: one stage-runner invocation, its log kept
  local stages=$1 label=$2 rc=0 stagelog
  stagelog=$LOGDIR/stage-$label-$RUN_NAME.log
  state "running leg=train-$label stages=$stages"
  GAMES_ARM_SEQ_STAGES=$stages uv run --frozen python -m games.stage_runner --plan games.arm_sequence >"$stagelog" 2>&1 || rc=$?
  tail -15 "$stagelog"
  return "$rc"
}

if corpus=$(resolve_corpus); then
  echo "corpus resolves to $corpus; the sweep is not needed"
elif [ -n "$(find "$SWEEP_DIR" -maxdepth 1 -name 'corpus-*.jsonl' -print -quit)" ]; then
  # A corpus is there and the resolver still refused it (empty, or more than one): a sweep would add
  # another rather than settle it, so this is the operator's call.
  cat "$RESOLVE_LOG"
  fail "corpus-unresolvable in $SWEEP_DIR"
elif [ "$SWEEP_WANTED" = 1 ]; then
  plan_gate sweep sweep
  run_stages sweep sweep || fail "sweep rc=$?"
  # Banked at once, so a box lost during the arm relaunches straight into it.
  aws s3 sync "$SWEEP_DIR/" "$RUN_PREFIX/select/" --only-show-errors --region "$S3_REGION" || fail "bank-corpus rc=$?"
  corpus=$(resolve_corpus) || {
    cat "$RESOLVE_LOG"
    fail "resolve-corpus after the sweep"
  }
else
  cat "$RESOLVE_LOG"
  fail "no corpus resolves and TRAIN_STAGES=$TRAIN_STAGES asks for no sweep"
fi
export GAMES_ARM_SEQ_CORPUS=$corpus
# A kit appends its registered lines here (estimator flags, completion budget, corpus name).
plan_gate "$CONSUMER_STAGES" arm '--save-steps 1' "--max-steps $MAX_STEPS"
# A relaunch after training finished is harmless: the trainer recognises a checkpoint already at
# max_steps beside its train_summary.json, logs ALREADY COMPLETE and touches nothing.
run_stages "$CONSUMER_STAGES" arm || fail "train rc=$?"

# --- eval cells: each restores from and syncs to its own prefix; a relaunch continues a cell rather
#     than re-running it, and nothing ever deletes a partial trace (its records are paid for) -----------
CELL=$EVALS/$ARM
S3_CELL=$RUN_PREFIX/evals/$ARM
battery() { # battery <step> [extra flags]; step 0 is the un-adapted base model
  local step=$1
  shift
  uv run --frozen python -m games.run_evals --run-dir "$OUTDIR" --steps "$step" --arm "$ARM" \
    --base-model "$MODEL" ${BATTERY_SECTIONS:+--sections "$BATTERY_SECTIONS"} --backend vllm --no-report \
    --out-dir "$CELL" --sync-dest "$S3_CELL" "$@"
}
state "running leg=evals steps=$EVAL_STEPS"
reason=done
for step in ${EVAL_STEPS//,/ }; do
  [ "$step" = 0 ] || checkpoint_complete "$OUTDIR/checkpoint-$step" || restore_checkpoint "$step"
  CELLLOG=$LOGDIR/cell-battery-step-$step.log
  rc=0
  battery "$step" >"$CELLLOG" 2>&1 || rc=$?
  tail -20 "$CELLLOG"
  if [ "$rc" -ne 0 ]; then
    echo "BATTERY CELL FAILED: step $step rc=$rc (relaunching the identical launch resumes it)"
    reason=eval-cell-failed
  fi
done
# The closing pass: --summarise-only writes the summary of a cell that died between its last
# generate call and its summary write, exits 0 for a complete cell of this run, and refuses an
# unfinished one, so the agenda's exit code says whether every cell is closed. No engine loads.
for step in ${EVAL_STEPS//,/ }; do
  if ! battery "$step" --summarise-only >>"$LOGDIR/cell-battery-close.log" 2>&1; then
    echo "CELL NOT CLOSED: step $step (see $LOGDIR/cell-battery-close.log)"
    reason=eval-cell-unclosed
  fi
done
if [ "$reason" = done ]; then
  finish 0
fi
finish 1 "$reason"
BOOTSTRAP
chmod +x "$HOME_DIR/bootstrap-and-run.sh"
chown "$RUN_USER:$RUN_USER" "$HOME_DIR/bootstrap-and-run.sh" "$RUN_ENV"

# A hand-off that fails leaves nothing that would ever touch the marker, so the box would bill its
# whole idle threshold; the failure is recorded and the marker touched here instead.
if ! sudo -u "$RUN_USER" env HOME="$HOME_DIR" tmux new-session -d -s "$RUN_NAME" \
  "bash $HOME_DIR/bootstrap-and-run.sh >$RUN_LOG 2>&1"; then
  echo "FATAL: tmux could not start the agenda"
  echo "exited rc=1 reason=tmux-launch-failed job=$RUN_NAME at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$CHAIN_STATE"
  touch "$DONE_MARKER"
  exit 1
fi
sudo -u "$RUN_USER" tmux ls
touch "$HOME_DIR/BOOTSTRAP_DONE"
echo "user-data finished; the $RUN_NAME agenda is running in tmux session $RUN_NAME"
