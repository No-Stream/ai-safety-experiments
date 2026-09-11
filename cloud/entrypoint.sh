#!/usr/bin/env bash
# Container entrypoint for a games training job on AWS Batch.
#
# Ordering here is the whole point. The commit and the GPU inventory are logged before anything
# else can fail, because the two questions asked of a dead job are always "which code was this?"
# and "what did it land on?", and both answers are gone if the log starts with a stack trace. The
# S3 sync is installed as an EXIT trap before training starts, so a job killed by a spot
# reclamation, an OOM, or a wall-clock timeout still ships everything through its last save --
# without that, a nine-hour run that dies at hour eight leaves nothing behind.
#
# Environment:
#   GAMES_OUTPUT_DIR          run directory inside the container (default /scratch/runs/current)
#   GAMES_S3_DEST             s3://bucket/prefix to sync the run dir to; unset disables syncing
#   GAMES_REQUIRE_FAST_KERNELS  1 makes missing DeltaNet kernels fatal instead of a warning
#   GAMES_VLLM_SERVER         1 starts `trl vllm-serve` on GPU1 and trains on GPU0 (27B topology)
#   GAMES_VLLM_PORT           port for that server (default 8000)
# Everything else is forwarded to `python -m games.train`.

set -Eeuo pipefail

log() {
  printf '[entrypoint %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

OUTPUT_DIR="${GAMES_OUTPUT_DIR:-/scratch/runs/current}"
S3_DEST="${GAMES_S3_DEST:-}"
VLLM_PORT="${GAMES_VLLM_PORT:-8000}"
VLLM_PID=""
TRAIN_PID=""

# ---------------------------------------------------------------- provenance, before anything else
log "GIT_SHA=${GIT_SHA:-<unset>}"
log "image python: $(python --version 2>&1)"
log "output_dir=${OUTPUT_DIR} s3_dest=${S3_DEST:-<none>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || log "WARNING: nvidia-smi exited non-zero; the driver may not be visible"
else
  log "WARNING: nvidia-smi not found. On a GPU queue this means the container was started"
  log "         without the NVIDIA runtime, and training will fall back to CPU or crash."
fi

# --------------------------------------------------------------------------- fast-kernel decision
# Qwen3.5/3.8 stack three Gated DeltaNet linear-attention layers per full-attention layer. Without
# causal_conv1d and flash-linear-attention, transformers falls back to slower, more memory-hungry
# PyTorch ops for those layers -- silently, which is the actual hazard: a throughput number read
# off the fallback path looks like a measurement of the model.
#
# DECISION for v1: warn loudly, do not fail, and do not build the kernels into the image.
# Reasons, in order. The plan's own next step is to measure 4B generate-path throughput, and a
# hard assert would block exactly the run whose purpose is to characterise the slow path. Building
# causal_conv1d (and flash-attn) from source needs a devel base with nvcc, adds a long and fragile
# compile to every image build, and enlarges the CVE surface -- all for a speedup nobody has
# measured here yet. flash-linear-attention is pure Triton and pip-installable, so adding it is
# cheap, but it belongs in uv.lock rather than in an ad-hoc image layer, and the lockfile is not
# this change's to edit.
# Set GAMES_REQUIRE_FAST_KERNELS=1 to arm the assert once the kernels are in the lockfile; the
# warning is written to the log above the training output either way, so no throughput number can
# be read without knowing which path produced it.
check_fast_kernels() {
  local missing=()
  for module in causal_conv1d fla flash_attn; do
    python -c "import ${module}" >/dev/null 2>&1 || missing+=("${module}")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    log "DeltaNet fast kernels present: causal_conv1d, fla, flash_attn"
    return 0
  fi
  log "=============================================================================="
  log "SLOW KERNEL PATH: missing ${missing[*]}"
  log "Qwen3.5/3.8 run 48 of 64 layers through Gated DeltaNet. Without these, transformers"
  log "uses slower, more memory-hungry PyTorch ops. Training is CORRECT but throughput and"
  log "peak VRAM from this run describe the fallback path, not the model."
  log "=============================================================================="
  if [[ "${GAMES_REQUIRE_FAST_KERNELS:-0}" == "1" ]]; then
    log "GAMES_REQUIRE_FAST_KERNELS=1, so this is fatal."
    return 1
  fi
  return 0
}
check_fast_kernels

# ------------------------------------------------------------------------------- artifact shipping
# The run directory ships WHOLESALE, and narrowing this to a file list or adding --exclude is how
# the rollout trace gets lost. `log_completions=True` makes TRL write a per-completion parquet to
# ${OUTPUT_DIR}/completions/ carrying every reward column, and nothing announces its absence: an
# HF-versus-vLLM comparison on 2026-08-19 had to recover 64 rollouts from a rich-rendered table in
# the training log, where 718 of 726 action tags were ellipsis-truncated, because a box-local
# launcher synced only its own log directory and never set GAMES_S3_DEST.
sync_to_s3() {
  if [[ -z "${S3_DEST}" ]]; then
    return 0
  fi
  if [[ ! -d "${OUTPUT_DIR}" ]]; then
    log "nothing to sync: ${OUTPUT_DIR} does not exist"
    return 0
  fi
  log "syncing ${OUTPUT_DIR} -> ${S3_DEST}"
  # Never let a failed upload change the exit code the job reports: the training result is what
  # Batch should surface, and a sync failure is a separate, visible line in the log.
  aws s3 sync "${OUTPUT_DIR}" "${S3_DEST}" --only-show-errors \
    || log "WARNING: s3 sync failed; artifacts remain only on the instance"
}

on_exit() {
  local status=$?
  log "exiting with status ${status}; running shutdown sync"
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    log "stopping vllm server pid ${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
  sync_to_s3
  log "shutdown complete, status ${status}"
}
# The handler forwards the signal to the trainer and waits for it to *exit* before returning, so the
# EXIT trap syncs a directory nothing is still writing to. That is all the wait buys: nothing in
# games/train.py handles SIGTERM, so the trainer does not checkpoint on the way out and whatever the
# periodic `save_steps` writer had already finished is what ships. A step in progress is lost, but it
# is lost cleanly rather than uploaded half-written. The `|| true`
# on that wait is load-bearing under `set -e`: the trainer was just signalled, so it exits non-zero,
# and errexit would abort the handler right there, skipping its own `exit` and reporting the
# trainer's status instead. Probed: without the guard, a trainer exiting 1 on SIGTERM made the shell
# report 1, which reads as a crash rather than as the stop it was.
#
# SIGTERM is forwarded whatever arrived, because with job control off bash sets SIGINT to ignore on
# an asynchronous child: forwarding INT would be a no-op the trainer never hears, and the wait below
# would then block until training finished on its own -- a Ctrl-C that hangs for hours. The shell
# still reports the status matching the signal it was actually sent.
on_signal() {
  local signal_name=$1 status=$2
  log "caught SIG${signal_name}"
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    log "stopping trainer pid ${TRAIN_PID}, waiting for it to exit before syncing"
    kill -s TERM "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" || true
  fi
  exit "${status}"
}
# EXIT alone does not fire for SIGTERM/SIGINT, and SIGTERM is exactly how Batch stops a job and how
# a spot instance is reclaimed -- the two cases where shipping the artifacts matters most.
trap on_exit EXIT
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT

mkdir -p "${OUTPUT_DIR}"

# ------------------------------------------------------------- optional 27B two-GPU vLLM topology
# Off by default: v1 trains on TRL's native generate path, and the plan wants 4B generate-path
# throughput measured before deciding whether generation dominates enough to justify this. When
# on, generation runs on GPU1 and training on GPU0, which is the only way a 27B fits alongside a
# server on one node. GRPOConfig's use_vllm / vllm_mode="server" / vllm_server_base_url were
# confirmed present in the installed TRL 1.10.
if [[ "${GAMES_VLLM_SERVER:-0}" == "1" ]]; then
  log "starting trl vllm-serve on GPU1, port ${VLLM_PORT}"
  CUDA_VISIBLE_DEVICES=1 trl vllm-serve --port "${VLLM_PORT}" &
  VLLM_PID=$!
  log "vllm server pid ${VLLM_PID}; training will target GPU0"
  export CUDA_VISIBLE_DEVICES=0
  export TRL_VLLM_SERVER_BASE_URL="http://127.0.0.1:${VLLM_PORT}"
fi

# ------------------------------------------------------------------------------------ run training
# The entrypoint owns --output-dir because the EXIT trap has to know which directory to ship. A
# caller passing its own would leave the trap syncing an empty path while the real run wrote
# somewhere else, so that is refused rather than silently preferred.
for arg in "$@"; do
  if [[ "${arg}" == "--output-dir" || "${arg}" == --output-dir=* ]]; then
    log "ERROR: pass the run directory as GAMES_OUTPUT_DIR, not --output-dir."
    log "       The exit-time S3 sync reads GAMES_OUTPUT_DIR, so the two must not disagree."
    exit 2
  fi
done

log "starting: python -m games.train --output-dir ${OUTPUT_DIR} $*"
# Deliberately not `exec`: exec replaces this shell, which discards the EXIT trap and with
# it the shutdown sync. Training runs as a child so the trap survives to ship the run.
#
# And deliberately BACKGROUNDED with an explicit `wait`, which is the only shape that leaves this
# shell interruptible. Bash records a signal received while it waits on a *foreground* command and
# defers the handler until that command returns; `wait` is the documented exception. Since the
# exec-form ENTRYPOINT makes this shell PID 1 with no init to forward anything, a foreground trainer
# means the SIGTERM from `docker stop` -- how Batch stops a job and how a spot instance is reclaimed
# -- is recorded and never acted on, and SIGKILL arrives ~30s later with the sync still pending.
# Measured, not theorised: with the trainer in the foreground neither trap logged anything.
# `wait` is also what propagates the trainer's exit status: under `set -e` a non-zero wait ends the
# shell with that status, so Batch still surfaces the training result and not the sync's.
python -m games.train --output-dir "${OUTPUT_DIR}" "$@" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
