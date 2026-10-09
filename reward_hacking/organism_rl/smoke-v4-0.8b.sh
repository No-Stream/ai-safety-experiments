#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$repo_root"

if (($# > 1)); then
  echo "usage: smoke-v4-0.8b.sh [run-directory]" >&2
  exit 2
fi

if (($# == 1)); then
  run_directory="$1"
else
  revision="$(git rev-parse --short=12 HEAD)"
  run_directory="artifacts/smoke-v4-0.8b-$revision"
fi
python_bin="$(command -v python)"
"$python_bin" -m reward_hacking.organism_rl.smoke_v4 --run-dir "$run_directory" --plan-only
exec "$python_bin" -m reward_hacking.organism_rl.smoke_v4 --run-dir "$run_directory"
