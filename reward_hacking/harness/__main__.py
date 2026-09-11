"""Entry point so ``python -m reward_hacking.harness`` runs the agent loop CLI."""

from __future__ import annotations

from reward_hacking.harness.loop import main

if __name__ == "__main__":
    raise SystemExit(main())
