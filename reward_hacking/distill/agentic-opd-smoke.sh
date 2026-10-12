#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$repo_root"
exec "$repo_root/.venv/bin/python" -m reward_hacking.distill.agentic_smoke "$@"
