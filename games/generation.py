"""How a games run generates: the sampler a sweep has to match, and the vLLM-only rollout rule.

Deliberately the cheapest module in `games/`: nothing here imports anything but stdlib (`os` and
`importlib.util`), and `test_games_plans.py` fails if the training stack creeps in. The facts it
holds are needed by a stage plan in order to render a launch command, and were previously
reachable only through modules that import the training stack -- `games.train` for the colocate
readers and `games.select_prompts` for the sampler. So `--print-plan`, whose whole selling point is
being the cheap check before the meter starts, imported torch, transformers, trl, peft, matplotlib
and pandas to print a string. Moving `games.arms` out of the trainer was the first half of fixing
that and did not on its own change the number: 8.4 s before, and unchanged, because the plan
skeleton still reached the trainer for the functions below.

None of this is about any *experiment*. The sampler is TRL's, the engine knobs describe the box
a run landed on; what an arm IS lives in `games.arms`.
"""

from __future__ import annotations

import importlib.util
import os

# GRPOConfig's own generation defaults in TRL 1.10 (grpo_config.py:525/529/536). "At training
# temperature" means these three exactly, and the sweep that selects a corpus has to sample at them
# or it selects prompts for a policy that never trains. Two plans and a shell script each carried
# their own copy of `1.0 / 1.0 / 0`, so retuning any one source left the rest silently off-policy
# while every flag still looked explicit and correct.
TRAINING_TEMPERATURE = 1.0
TRAINING_TOP_P = 1.0
TRAINING_TOP_K = 0

# The colocate switch and its two knobs are named in the environment rather than only on the
# command line because they describe the box a run landed on, not the experiment: every plan in
# `games/` inherits the capability from these, and the two arms first launched through it were
# launched with exactly these names.
VLLM_COLOCATE_ENV = "GAMES_VLLM_COLOCATE"
VLLM_GPU_FRACTION_ENV = "GAMES_VLLM_UTIL"
VLLM_IS_CORRECTION_ENV = "GAMES_VLLM_IS_CORRECTION"
# The engine's share of the card, as a fraction of TOTAL VRAM rather than of what is free. 0.35 is
# what the runs that established this path used: Qwen3.5-2B, thinking on, a 32,768-token completion
# budget, 64 episodes per step, a 95 GiB card (2026-08-19). Generation on that shape measured ~13.5
# minutes per step against ~72 on transformers.generate, which is the entire reason for the switch.
VLLM_COLOCATE_GPU_FRACTION = 0.35


# Why there is no transformers.generate rollout path for games training any more, spelled once and
# named by every refusal that enforces it. The 2026-08-25 incident it answers: a stray
# GAMES_VLLM_COLOCATE=0 in a trainer's environment silently put a paid arm on transformers.generate
# at ~72 minutes per step against the ~13.5 the plan costed, and only a human pace diagnosis $28 in
# caught it.
VLLM_ONLY_RATIONALE = (
    "games training rollouts generate through the colocated vLLM engine, always (owner decision, "
    "2026-08-26: 'vllm only, always'). transformers.generate measured ~72 minutes per step against "
    "~13.5 through the engine at the production shape, and on 2026-08-25 a stray "
    f"{VLLM_COLOCATE_ENV}=0 silently put a paid arm on that slow path. No flag or environment "
    "value selects a rollout backend any more."
)


def assert_vllm_rollouts() -> None:
    """Refuse an environment that asks games training for a non-vLLM rollout backend.

    `GAMES_VLLM_COLOCATE` used to select between backends; since 2026-08-26 it can only confirm
    the one that exists. Absent and "1" both mean colocate. Any other value is exactly the class
    of stray export that caused the incident, so it is refused at startup rather than read as a
    preference -- from `games.train` (config construction and `main`) and from every stage plan
    render in `games.plans`, which is the cheap check before the meter starts.
    """
    value = os.environ.get(VLLM_COLOCATE_ENV)
    if value is None or value == "1":
        return
    raise RuntimeError(
        f"{VLLM_COLOCATE_ENV}={value!r} asks for a non-vLLM rollout backend: "
        f"{VLLM_ONLY_RATIONALE} Unset the variable (or set it to '1') and relaunch."
    )


def colocate_requested() -> bool:
    """Report whether this box prefers the colocated engine -- the REWARD-HACKING trainer's reader.

    Games training stopped consulting this on 2026-08-26: its rollouts are vLLM-only
    (`assert_vllm_rollouts` above), with nothing left to request. The reader survives because
    `reward_hacking.train` defaults its own `--vllm-colocate` flag from it and keeps a slow-path
    escape hatch behind an explicit `--allow-hf-generation` acknowledgment -- a stated choice,
    not the silent fallback the games incident was made of. An explicit environment value wins
    outright; with the variable unset, colocate is the default wherever the engine is installed.
    `find_spec` answers without importing vllm, so a plan render stays cheap.
    """
    value = os.environ.get(VLLM_COLOCATE_ENV)
    if value is not None:
        return value == "1"
    return importlib.util.find_spec("vllm") is not None


def colocate_gpu_fraction() -> float:
    """Report the engine's share of the card, from the environment or the measured default."""
    return float(os.environ.get(VLLM_GPU_FRACTION_ENV, VLLM_COLOCATE_GPU_FRACTION))


def colocate_importance_sampling_correction() -> bool:
    """Report whether TRL's vLLM importance-sampling correction stays on, per the environment."""
    return os.environ.get(VLLM_IS_CORRECTION_ENV, "1") != "0"
