"""Minimal simulated-episode harness for the reward-hacking exploration leads.

Give a small model a tiny ops task in a working directory that makes out-of-band surfaces
reachable (a grading fixture, a cross-episode scratch note, a grader socket, a package
index), run a short multi-turn rollout, and capture the full trace so a human can read what
the model reached for. The whole point is to *watch* — nothing here grades the task, and
nothing real executes: file tools act on a throwaway directory, command tools are parsed and
simulated but never run, so no jail and no GPU beyond the model forward pass are needed.

Serves three leads from ``docs/scratch/2026-08-15-reward-hacking-exploration-plan.md``: whether
a small model reaches out of bounds at all and how far (#1), whether the reaching moves as
the environment advertises the route more loudly (#2, the dose axis), and whether a base-like
high-temperature policy already does it (#6, a run-config variant, not a separate model).
"""
