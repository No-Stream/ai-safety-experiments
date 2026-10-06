#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -lt 2 ]]; then
  echo "usage: scripts/distill_rounds.sh <session-name> --run-dir ... [round options...]" >&2
  exit 2
fi
session_name="$1"
shift
cd "$repo_root"
exec "$repo_root/scripts/tmux_run.sh" "$session_name" -- env \
  "PATH=$repo_root/.venv/bin:$PATH" \
  "PYTHONPATH=$repo_root" \
  "$repo_root/.venv/bin/python" -m reward_hacking.distill.rounds "$@"
