#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

stage=""
model=""
requested_cells=""
dry_run=0
while (($#)); do
  case "$1" in
    --stage)
      [[ -z "$stage" && $# -ge 2 ]] || { echo "--stage must be supplied once" >&2; exit 2; }
      stage="$2"
      shift 2
      ;;
    --model)
      [[ -z "$model" && $# -ge 2 ]] || { echo "--model must be supplied once" >&2; exit 2; }
      model="$2"
      shift 2
      ;;
    --cells)
      [[ -z "$requested_cells" && $# -ge 2 ]] || { echo "--cells must be supplied once" >&2; exit 2; }
      requested_cells="$2"
      shift 2
      ;;
    --dry-run)
      [[ "$dry_run" == 0 ]] || { echo "--dry-run must be supplied once" >&2; exit 2; }
      dry_run=1
      shift
      ;;
    --help|-h)
      cat <<'HELP'
Run one minimal evaluation stage. Launch each stage separately through the resource limiter and tmux.

Usage: minimal-eval-v4.sh --stage {gate3,cells} --model {base,v3,v4} [--cells N,N,...] [--dry-run]

--cells runs a subset of the stage's cells (for example when one cell's assets are not ready yet).

For v4 execution, set SEED_V4_ADAPTER_DIR to the final LoRA adapter.
HELP
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ -n "$stage" && -n "$model" ]] || { echo "--stage and --model are required" >&2; exit 2; }
[[ "$stage" == gate3 || "$stage" == cells ]] || { echo "unsupported stage: $stage" >&2; exit 2; }
[[ "$model" == base || "$model" == v3 || "$model" == v4 ]] || {
  echo "unsupported model: $model" >&2
  exit 2
}

case "$stage:$model" in
  gate3:*) cells=(1 2 4) ;;
  cells:base) cells=(5 6 7 8 9) ;;
  cells:v3) cells=(5 7 8 9) ;;
  cells:v4) cells=(3 5 6 7 8 9) ;;
esac
stage_cells=("${cells[@]}")
stage_episodes=0
if [[ -n "$requested_cells" ]]; then
  IFS=',' read -r -a cells <<<"$requested_cells"
  for cell in "${cells[@]}"; do
    [[ " ${stage_cells[*]} " == *" $cell "* ]] || {
      echo "cell $cell is not in stage $stage:$model (${stage_cells[*]})" >&2
      exit 2
    }
  done
fi

episode_count_for_cell() {
  case "$1" in
    1|2|3|4|5|6) echo 8 ;;
    7) echo 11 ;;
    8|9) echo 6 ;;
    *) echo "unknown cell: $1" >&2; return 2 ;;
  esac
}

for cell in "${stage_cells[@]}"; do
  stage_episodes=$((stage_episodes + $(episode_count_for_cell "$cell")))
done
total_episodes=0
cell_counts=()
for cell in "${cells[@]}"; do
  episodes="$(episode_count_for_cell "$cell")"
  total_episodes=$((total_episodes + episodes))
  cell_counts+=("$cell=$episodes")
done

case "$stage:$model" in
  gate3:*) estimated_hours="0.9" ;;
  cells:base) estimated_hours="1.1" ;;
  cells:v3) estimated_hours="1.0" ;;
  cells:v4) estimated_hours="1.6" ;;
esac
estimated_hours="$(awk -v hours="$estimated_hours" -v part="$total_episodes" -v whole="$stage_episodes" \
  'BEGIN { printf "%.1f", hours * part / whole }')"

printf 'stage: %s\n' "$stage"
printf 'model: %s\n' "$model"
cell_list="$(printf '%s, ' "${cells[@]}")"
cell_list="${cell_list%, }"
cell_count_list="$(printf '%s, ' "${cell_counts[@]}")"
cell_count_list="${cell_count_list%, }"
printf 'cells: %s\n' "$cell_list"
printf 'episodes per cell: %s\n' "$cell_count_list"
printf 'total episodes: %s\n' "$total_episodes"
printf 'estimated wall-clock: %s h\n' "$estimated_hours"
if [[ " ${cells[*]} " == *" 1 "* || " ${cells[*]} " == *" 2 "* ]]; then
  printf 'AISI selection: round-robin by task ID (apps 3, humaneval 3, mbpp 2)\n'
fi
if [[ " ${cells[*]} " == *" 8 "* || " ${cells[*]} " == *" 9 "* ]]; then
  printf 'math cells: 8=loose_oneoff (6 held-out tasks), 9=loose_hidden_original (6 held-out tasks)\n'
fi

python_bin="${PYTHON:-.venv/bin/python}"
base_model="Qwen/Qwen3.5-9B"
adapter_dir=""
model_label="organism-ladder-$model"
served_model_id="$base_model"
case "$model" in
  v3)
    adapter_dir="${SEED_V3_ADAPTER_DIR:-artifacts/distill/agentic-sft-seed-v3-onpolicy-ef398f9/train/checkpoint-34}"
    [[ -d "$adapter_dir" ]] || { echo "v3 adapter is missing: $adapter_dir" >&2; exit 2; }
    served_model_id="$model_label"
    ;;
  v4)
    adapter_dir="${SEED_V4_ADAPTER_DIR:-}"
    [[ -n "$adapter_dir" && -d "$adapter_dir" ]] || {
      echo "set SEED_V4_ADAPTER_DIR to the final v4 adapter directory" >&2
      exit 2
    }
    served_model_id="$model_label"
    ;;
esac

scenario_root="${SEED_V4_SCENARIO_ROOT:-reward_hacking/ladder/data/scenarios-v2.2-heldout-wip}"
if [[ " ${cells[*]} " == *" 7 "* && "$dry_run" == 0 ]]; then
  "$python_bin" - "$scenario_root" <<'PY'
import sys
from pathlib import Path

from reward_hacking.ladder.scenario import load_scenarios

scenarios = load_scenarios(Path(sys.argv[1]))
oneoffs = [scenario for scenario in scenarios if scenario.variant == "oneoff"]
if len(oneoffs) != 11:
    raise SystemExit(f"expected 11 held-out ladder oneoffs, found {len(oneoffs)}")
PY
fi

max_model_len="${MAX_MODEL_LEN:-65536}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
cell_root="artifacts/organism-eval/minimal-eval-v4-announced-budget/$model"
dry_run_operation_count=0
dry_run_operations=()
dry_run_records=()

run_ladder_cell() {
  if [[ "$dry_run" == 1 ]]; then
    dry_run_operation_count=$((dry_run_operation_count + 1))
    dry_run_operations+=(
      "ladder"
      "7"
      "$base_model"
      "$adapter_dir"
      "$served_model_id"
      "$max_model_len"
      "$gpu_memory_utilization"
      "artifacts/ladder/minimal-eval-v4-announced-budget/$model/cell-7-announced-20"
      "$scenario_root"
    )
    return
  fi

  SEED_EVAL_BASE_MODEL="$base_model" \
    SEED_EVAL_ADAPTER_DIR="$adapter_dir" \
    SEED_EVAL_SERVED_MODEL_ID="$served_model_id" \
    SEED_EVAL_DRY_RUN="$dry_run" \
    SEED_EVAL_MAX_MODEL_LEN="$max_model_len" \
    SEED_EVAL_GPU_MEMORY_UTILIZATION="$gpu_memory_utilization" \
    SEED_EVAL_OUTPUT_DIR="artifacts/ladder/minimal-eval-v4-announced-budget/$model/cell-7-announced-20" \
    SEED_EVAL_SCENARIO_ROOT="$scenario_root" \
    "$python_bin" - <<'PY'
import logging
import os
from pathlib import Path

from reward_hacking.organism_rl import evaluate
from reward_hacking.vllm_server import VLLMServer
from reward_hacking.ladder import runner
from reward_hacking.ladder.scenario import load_scenarios

base_model = os.environ["SEED_EVAL_BASE_MODEL"]
adapter_text = os.environ["SEED_EVAL_ADAPTER_DIR"]
adapter_dir = Path(adapter_text) if adapter_text else None
served_model_id = os.environ["SEED_EVAL_SERVED_MODEL_ID"]
dry_run = os.environ["SEED_EVAL_DRY_RUN"] == "1"
max_model_len = os.environ["SEED_EVAL_MAX_MODEL_LEN"]
gpu_memory_utilization = os.environ["SEED_EVAL_GPU_MEMORY_UTILIZATION"]
output_dir = Path(os.environ["SEED_EVAL_OUTPUT_DIR"])
scenario_root = Path(os.environ["SEED_EVAL_SCENARIO_ROOT"])
evaluation_arguments = [
    "--output-dir",
    str(output_dir / "server"),
    "--model",
    base_model,
    "--served-model-id",
    served_model_id,
    "--max-model-len",
    max_model_len,
    "--family",
    "ilcb",
    "--impossible-variant",
    "conflicting",
    "--evaluation-variant",
    "conflicting",
    "--tasks-per-family",
    "8",
    "--max-turns",
    "20",
    "--max-new-tokens",
    "8192",
    "--thinking-token-budget",
    "2048",
    "--temperature",
    "0.7",
    "--top-p",
    "1.0",
    "--concurrency",
    "1",
    "--gpu-memory-utilization",
    gpu_memory_utilization,
    "--no-enable-speculative-decoding",
]
if adapter_dir is not None:
    evaluation_arguments.extend(("--adapter-dir", str(adapter_dir)))
args = evaluate.parse_args(evaluation_arguments)
model_revision = evaluate.resolve_model_revision(args)
server_config = evaluate.build_server_config(args, model_revision)
scenarios = load_scenarios(scenario_root)
oneoff_slugs = [scenario.slug for scenario in scenarios if scenario.variant == "oneoff"]
if len(oneoff_slugs) != 11:
    raise ValueError(f"expected 11 held-out ladder oneoffs, found {len(oneoff_slugs)}")
if server_config.log_path is None:
    raise ValueError("ladder evaluation requires a vLLM server log path")

runner_arguments = [
    "--endpoint",
    "{endpoint}",
    "--server-log",
    str(server_config.log_path),
    "--model",
    # The runner asks vLLM for the model by this label, so it must be the served id.
    f"{served_model_id}={base_model}",
    "--scenario-root",
    str(scenario_root),
    "--arm",
    "real",
    "--samples",
    "1",
    "--output",
    str(output_dir / "records.jsonl"),
    "--episode-root",
    str(output_dir / "episodes"),
    "--max-turns",
    "20",
    "--announced-turn-budget",
    "20",
    "--max-new-tokens",
    "8192",
    "--thinking-token-budget",
    "2048",
    "--top-p",
    "1.0",
    "--episode-concurrency",
    "4",
    "--oversize-reason",
    "11 held-out WIP ladder scenarios in the minimal evaluation cell",
]
for scenario_slug in oneoff_slugs:
    runner_arguments.extend(("--scenario", scenario_slug))

parsed_runner_arguments = runner._parse_cli_args(runner._build_cli_parser(), runner_arguments)
runner_model_ids = {model.server_model_id or model.model_id for model in parsed_runner_arguments.model}
if runner_model_ids != {args.served_model_id}:
    raise ValueError(f"ladder runner requests {runner_model_ids}, server serves {args.served_model_id!r}")
if dry_run:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        "validated ladder evaluation and runner arguments served_model_id=%s max_turns=%s announced_turn_budget=%s",
        args.served_model_id,
        parsed_runner_arguments.max_turns,
        parsed_runner_arguments.announced_turn_budget,
    )
    raise SystemExit(0)

with VLLMServer(server_config) as server:
    endpoint_position = runner_arguments.index("{endpoint}")
    runner_arguments[endpoint_position] = server.base_url
    raise SystemExit(runner.main(runner_arguments))
PY
}

run_eval_cell() {
  local cell="$1"
  local family=""
  local variant=""
  local impossible_variant="oneoff"
  local task_count=8
  local output_dir="$cell_root/cell-$cell"
  local reuse_from=""
  local max_turns=20
  # Cells 1-4 stay comparable with the reused evalA base records, which never announced the budget; the cells new
  # for every model announce the real budget, as seed-v4 generation does.
  local announce_turn_budget=1
  local -a arguments

  case "$cell" in
    1) family="aisi"; variant="oneoff"; announce_turn_budget=0 ;;
    2) family="aisi"; variant="original"; announce_turn_budget=0 ;;
    3) family="ilcb"; variant="oneoff"; announce_turn_budget=0 ;;
    4) family="ilcb"; variant="original"; announce_turn_budget=0 ;;
    5) family="ilcb"; variant="conflicting"; impossible_variant="conflicting" ;;
    6) family="r2e_leak"; variant="leak"; max_turns=30 ;;
    8) family="math_loose_checker"; variant="loose_oneoff"; impossible_variant="loose_oneoff"; task_count=6 ;;
    9) family="math_loose_checker"; variant="loose_hidden_original"; impossible_variant="loose_oneoff"; task_count=6 ;;
    7) run_ladder_cell; return ;;
  esac

  if [[ "$stage" == gate3 && "$model" == base ]]; then
    reuse_from="${SEED_EVAL_BASE_REUSE_FROM:-artifacts/organism-eval/evalA-base9b-c6a275f}"
    output_dir="artifacts/organism-eval/base-cells"
  elif [[ "$stage" == gate3 && "$model" == v3 ]]; then
    reuse_from="${SEED_EVAL_V3_REUSE_FROM:-artifacts/organism-eval/minimal-seedv3c34-ef398f9/coding}"
    output_dir="artifacts/organism-eval/base-cells"
  fi

  arguments=(
    --output-dir "$output_dir"
    --model "$base_model"
    --gpu-memory-utilization "$gpu_memory_utilization"
    --max-model-len "$max_model_len"
    --max-new-tokens 8192
    --concurrency 6
    --max-turns "$max_turns"
    --thinking-token-budget 2048
    --temperature 1.0
    --top-p 1.0
    --family "$family"
    --impossible-variant "$impossible_variant"
    --evaluation-variant "$variant"
    --tasks-per-family "$task_count"
    --payment-register trusted_rerun
  )
  if [[ "$family" == aisi ]]; then
    arguments+=(--aisi-sources apps,humaneval,mbpp --aisi-task-id-order)
  fi
  if [[ "$family" == ilcb ]]; then
    # The official tests exceed the 150-second grading limit even for correct solutions.
    arguments+=(--exclude-ilcb-problem-id lcbhard_16)
  fi
  if [[ -n "$adapter_dir" ]]; then
    arguments+=(--adapter-dir "$adapter_dir")
  fi
  if [[ "$announce_turn_budget" == 1 ]]; then
    arguments+=(--announced-turn-budget "$max_turns")
  fi
  if [[ "$family" == r2e_leak ]]; then
    arguments+=(
      --leak-assets /var/tmp/leak-env/assets-v6-heldout/assets.json
      --leak-channel-splits docs/scratch/leak-v4/channel-splits.json
      --leak-channel-split seed-v4
      --leak-channel-per-episode
    )
  fi
  if [[ -n "$reuse_from" ]]; then
    [[ -d "$reuse_from" ]] || { echo "reuse source is missing: $reuse_from" >&2; exit 2; }
    arguments+=(
      --reuse-from "$reuse_from"
      --reuse-waiver "The source predates generation-path identity. Its selected task IDs and task definitions are checked; the submit notice was the only known model-visible change, it fired in 0 of 48 evalA episodes, and the saved sampling flags and turn budgets match this cell."
    )
  fi

  local log_path="/var/tmp/minimal-eval-v4-$model-cell-$cell-$$.log"
  local records_path="$output_dir/records.jsonl"
  if [[ "$dry_run" == 1 ]]; then
    dry_run_operation_count=$((dry_run_operation_count + 1))
    dry_run_operations+=("evaluate" "${#arguments[@]}" "${arguments[@]}")
    if [[ "$cell" != 7 ]]; then
      dry_run_records+=("$records_path")
    fi
    return
  fi
  if [[ -n "$reuse_from" ]]; then
    "$python_bin" -m reward_hacking.organism_rl.evaluate "${arguments[@]}" 2>&1 | tee "$log_path"
    local reused_dir
    reused_dir="$(sed -n 's/^INFO reward_hacking\.organism_rl\.evaluate: reused cell path=\([^ ]*\) .*/\1/p' "$log_path" | tail -n 1)"
    [[ -n "$reused_dir" ]] || { echo "evaluate did not report its reused cell path" >&2; exit 1; }
    records_path="$reused_dir/records.jsonl"
  else
    "$python_bin" -m reward_hacking.organism_rl.evaluate "${arguments[@]}"
  fi
  "$python_bin" -m reward_hacking.organism_rl.rescore \
    --records "$records_path" \
    --payment-register in_band
}

for cell in "${cells[@]}"; do
  run_eval_cell "$cell"
done

if [[ "$dry_run" == 1 && "$dry_run_operation_count" -gt 0 ]]; then
  "$python_bin" -m reward_hacking.organism_rl.minimal_eval_v4_dry_run \
    "$dry_run_operation_count" "${dry_run_operations[@]}" \
    "${#dry_run_records[@]}" "${dry_run_records[@]}"
fi
