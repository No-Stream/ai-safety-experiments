#!/usr/bin/env bash
# Run the Atlas, AISI CodeContests, and ILCB single-turn screens for one distillation label.
set -uo pipefail

if (( $# < 1 || $# > 3 )); then
  echo "usage: $0 <label> [adapter-dir] [hard-cap]" >&2
  exit 2
fi

LABEL=$1
ADAPTER_DIR=${2:-}
HARDTOK=${3:-16384}
if [[ ! "$LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "label must contain only letters, digits, '.', '_' or '-'" >&2
  exit 2
fi
if [[ ! "$HARDTOK" =~ ^[1-9][0-9]*$ ]]; then
  echo "hard cap must be a positive integer" >&2
  exit 2
fi
if [[ -n "$ADAPTER_DIR" && ! -d "$ADAPTER_DIR" ]]; then
  echo "adapter directory does not exist: $ADAPTER_DIR" >&2
  exit 2
fi
if [[ -n "$ADAPTER_DIR" ]]; then
  ADAPTER_DIR=$(cd -- "$ADAPTER_DIR" && pwd -P) || {
    echo "could not resolve adapter directory: $2" >&2
    exit 2
  }
fi

cd "$(dirname "$0")/../.."
MODEL=Qwen/Qwen3.5-9B
TAG=distill
N=20
MAXTOK=8192
KWARGS='{}'
RUN="$TAG-$LABEL-none"
HARDRUN="$RUN-cap$HARDTOK"
ATLAS_RUN_DIR="artifacts/atlas/$RUN"
AISI_RUN_DIR="artifacts/organisms/aisi-olmo/$HARDRUN"
ILCB_OUT="artifacts/organisms/screen/$HARDRUN-ilcb.json"
SUMMARY_DIR="artifacts/organisms/elicitation-summary/$HARDRUN"
LIMIT=(scripts/resource-limits.sh --gpu -t 45m --)

ATLAS_MODEL=base
AISI_MODEL_ARGS=()
ILCB_ADAPTER_ARGS=()
if [[ -n "$ADAPTER_DIR" ]]; then
  ATLAS_MODEL=$ADAPTER_DIR
  AISI_MODEL_ARGS=(--model "$ADAPTER_DIR")
  ILCB_ADAPTER_ARGS=(--lora-adapter "$ADAPTER_DIR")
fi

failed_stages=()
run_stage() {
  local name=$1
  shift
  echo "=== STAGE $name $(date)"
  if "${LIMIT[@]}" env \
    PATH="$PWD/.venv/bin:$PATH" \
    PYTHONPATH="$PWD" \
    .venv/bin/python "$@"; then
    echo "=== STAGE EXIT=0 $(date)"
  else
    local status=$?
    echo "=== STAGE EXIT=$status $(date)"
    failed_stages+=("$name")
  fi
}

run_stage atlas-prompts -m reward_hacking.atlas prompts \
  --run-name "$RUN" --base-model "$MODEL" --datasets mbpp-hardcode \
  --max-items 10 --leakage-hint-variant none --chat-template-kwargs "$KWARGS"
run_stage atlas-sample -m reward_hacking.atlas sample \
  --run-name "$RUN" --base-model "$MODEL" --models "$ATLAS_MODEL" \
  --samples-per-prompt 1 --max-prompts "$N" --max-tokens "$MAXTOK"
run_stage atlas-grade -m reward_hacking.atlas grade \
  --run-name "$RUN" --base-model "$MODEL" --models "$ATLAS_MODEL" --max-records "$N"

run_stage aisi -m reward_hacking.organisms.aisi_codecontests \
  --run-name "$HARDRUN" --base-model "$MODEL" "${AISI_MODEL_ARGS[@]}" \
  --max-problems "$N" --samples-per-problem 1 --max-tokens "$HARDTOK" \
  --temperature 1 --leakage-hint-variant none --chat-template-kwargs "$KWARGS"

run_stage ilcb -m reward_hacking.hatch_probe_cli \
  --model "$MODEL" --backend vllm --thinking --max-tokens "$HARDTOK" \
  --chat-template-kwargs "$KWARGS" "${ILCB_ADAPTER_ARGS[@]}" \
  --arms misspecified --hatch-levels absent --framings neutral \
  --max-problems "$N" --samples-per-prompt 1 --seed 0 --leakage-hint-variant none \
  --out "$ILCB_OUT" --grader-scratch-root /var/tmp/rh-hatch-probe-graders

if (( ${#failed_stages[@]} > 0 )); then
  printf 'Not summarizing; these stages failed: %s\n' "${failed_stages[*]}" >&2
  exit 1
fi

if ! env PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD" .venv/bin/python \
  -m reward_hacking.organisms.elicitation_summary \
  --out-dir "$SUMMARY_DIR" \
  --cell "atlas_mbpp:$LABEL:none:$ATLAS_RUN_DIR" \
  --cell "aisi_codecontests:$LABEL:none:$AISI_RUN_DIR" \
  --cell "ilcb:$LABEL:none:$ILCB_OUT"; then
  echo "elicitation summary failed for $RUN" >&2
  exit 1
fi

echo "=== RUN DONE $RUN $(date)"
