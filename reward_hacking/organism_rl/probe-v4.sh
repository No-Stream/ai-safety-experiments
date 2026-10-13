#!/usr/bin/env bash
set -euo pipefail

# Run each stage separately after scripts/gpu_preflight.py.
# Preflight: scripts/gpu_preflight.py
# Base: scripts/tmux_run.sh seed-v4-probe-base -- scripts/resource-limits.sh --gpu -t 90m -- env PATH="$PWD/.venv/bin:$PATH" reward_hacking/organism_rl/probe-v4.sh --stage base --leak-assets /var/tmp/leak-env/assets-v8-train/assets.json
# V3 add-on: scripts/tmux_run.sh seed-v4-probe-v3 -- scripts/resource-limits.sh --gpu -t 90m -- env PATH="$PWD/.venv/bin:$PATH" reward_hacking/organism_rl/probe-v4.sh --stage v3 --leak-assets /var/tmp/leak-env/assets-v8-train/assets.json

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
exec python -m reward_hacking.organism_rl.probe_v4 "$@"
