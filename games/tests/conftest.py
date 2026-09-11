"""Shared scaffolding for the games suite: the CLI ``--help`` probe the stage-plan tests share.

``test_games_arm_sequence`` and ``test_games_contrast_pair_sequence`` check every stage's flags
against the target CLI's own ``--help`` -- the seam that costs the most to discover late, because a
plan renders argv as data and a flag renamed upstream stays invisible until the stage runs on a
rented box with the meter started. Each probe is a fresh interpreter importing torch and
transformers, about ten seconds, and the plans only ever name five CLI modules, so asking once per
module per session instead of once per stage per test is what takes the two files' CLI classes from
about 190 s to about 50 s (measured 2026-09-02, see the dev-loop audit under
``docs/scratch/hot-path-optimization-2026-09-02/``).

The cache key is the module plus the environment its ``--help`` may read, and that second half is
load-bearing: ``games.train`` computes three argparse defaults from ``GAMES_S3_DEST``,
``GAMES_VLLM_UTIL`` and ``GAMES_VLLM_IS_CORRECTION`` while building its parser, and
``float(os.environ[...])`` on a stray non-numeric value makes its ``--help`` exit non-zero, so a
module-only key would let a knob one test sets be answered from a probe another test ran. The key
carries every ``GAMES_*`` variable except what the plan layer reads under that prefix and translates
into the very flags under test, none of which reaches a CLI: the three plan-selection namespaces,
which also carry a per-test ``tmp_path`` and would make every test a miss, and the estimator pin
(``games.train --help`` is byte-identical with and without it, checked 2026-09-02). Both ways the
rule can rot are safe ones: a new CLI knob under ``GAMES_`` is keyed automatically, and a new plan
namespace costs extra probes, never a stale hit. The interpreter-level environment (``PATH``,
``HOME``, the HF and CUDA variables) is taken as constant across a session, which pytest's
monkeypatch guarantees between tests and which no caller of this probe touches.

Why the key is a rule rather than a trace: recording the names a ``--help`` run reads (patching
``os._Environ.__getitem__`` in the child) gives an exact six-name key for ``games.regrade_corpus``,
but every torch-importing CLI enumerates the whole environment during import, so the exact key there
is the entire environment and the cache would never hit across tests (measured 2026-09-02).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from games.generation import VLLM_COLOCATE_ENV, VLLM_GPU_FRACTION_ENV, VLLM_IS_CORRECTION_ENV
from games.plans import (
    ESTIMATOR_PIN_ENV,
    FLAGSHIP_ESTIMATOR_PIN,
    OPTIONAL_TRAINING_KNOBS,
    OPTIONAL_TRAINING_SWITCHES,
)

FLAG = re.compile(r"--[a-z0-9][a-z0-9-]*")

CLI_ENV_PREFIX = "GAMES_"
PLAN_ENV_PREFIXES = ("GAMES_ARM_SEQ_", "GAMES_PAIR_", "GAMES_9B_")
PLAN_ENV_NAMES = frozenset({ESTIMATOR_PIN_ENV})

# A value of the shape each optional training knob really takes, keyed by the variable suffix
# `games.plans.OPTIONAL_TRAINING_KNOBS` reads it under. Values spaced against the flag are what a
# launch gate greps a printed plan for, the way `plans.describe_stages` joins an argv. Only the values
# live here: the suffixes and flags come from the plan table itself below, because a hand copy of it
# let a knob added at the plan layer escape every "each one reaches the CLI" test in the three
# stage-plan files and escape `clear_every_optional_knob` with it.
KNOB_SETTING_VALUES: dict[str, str] = {
    "PARSE_PENALTY": "-0.25",
    "LR_SCHEDULER": "constant_with_warmup",
    "WARMUP_RATIO": "0.05",
    "ADAM_EPSILON": "1e-15",
    "LORA_DROPOUT": "0",
    "VLLM_IMPORTANCE_SAMPLING_MODE": "token_truncate",
    "DYNAMIC_SAMPLING_OVERSAMPLE": "2",
}


def _knob_settings() -> tuple[tuple[str, str, str], ...]:
    """Pair every knob the plan layer reads with a value of the shape it takes, in the plan's order.

    Raises rather than skipping an unpaired knob: a knob nobody wrote a test value for is a knob no
    stage-plan test exercises, which is exactly the silence this derivation exists to break.
    """
    missing = [
        suffix
        for suffix, _flag, _assert_valid in OPTIONAL_TRAINING_KNOBS
        if suffix not in KNOB_SETTING_VALUES
    ]
    if missing:
        raise RuntimeError(
            f"games.plans.OPTIONAL_TRAINING_KNOBS carries {missing}, which KNOB_SETTING_VALUES here "
            f"has no test value for. Add one of the shape games.train parses, so every stage-plan "
            f"test covers the new knob and clear_every_optional_knob strips it."
        )
    return tuple(
        (suffix, flag, KNOB_SETTING_VALUES[suffix])
        for suffix, flag, _assert_valid in OPTIONAL_TRAINING_KNOBS
    )


# The optional training knobs as the three stage-plan test files want them, (suffix, flag, value):
# all three plans read them through `plans.training_shape` under their own prefix.
OPTIONAL_KNOB_SETTINGS: tuple[tuple[str, str, str], ...] = _knob_settings()

# The optional switches, taken from the plan layer: valueless flags a plan renders only where its
# environment sets the variable to exactly "1". Kept apart from the knobs above because a switch
# renders one argv token where a knob renders two, and every test that reads
# `argv[argv.index(flag) + 1]` would otherwise read the next flag as a value. The plan table's third
# column, whether the trainer refuses the flag without the importance-sampling correction, is dropped
# here: the tests that care about the coupling name their own switch, and every test driven off this
# table only ever needs the suffix to export and the flag to look for.
OPTIONAL_SWITCH_SETTINGS: tuple[tuple[str, str], ...] = tuple(
    (suffix, flag) for suffix, flag, _needs_correction in OPTIONAL_TRAINING_SWITCHES
)

# The variables that describe the BOX a run landed on and which already-trained arms it is comparable
# to, rather than one plan's own settings. No per-prefix cleanup covers them, so each stage-plan test
# file has to strip them by name or an exporting shell changes what it measures: `plans.colocate_argv`
# renders the colocate switch and the engine's share into every training argv, the correction switch
# decides whether the instrument switches may render at all, and a pin adds four estimator flags.
PLAN_INDEPENDENT_ENV: tuple[str, ...] = (
    VLLM_COLOCATE_ENV,
    VLLM_GPU_FRACTION_ENV,
    VLLM_IS_CORRECTION_ENV,
    ESTIMATOR_PIN_ENV,
)

# What an operator's shell plausibly carries for each, picked so that failing to strip one changes what
# a stage renders or refuses it outright: the wave-4b launch recipe exports the correction off for the
# training role, "0" for the colocate switch is refused at render time, a non-default engine share
# lands in the argv, and a pin adds its four flags.
PLAN_INDEPENDENT_ENV_SHELL_VALUES: dict[str, str] = {
    VLLM_COLOCATE_ENV: "0",
    VLLM_GPU_FRACTION_ENV: "0.9",
    VLLM_IS_CORRECTION_ENV: "0",
    ESTIMATOR_PIN_ENV: FLAGSHIP_ESTIMATOR_PIN,
}


def clear_plan_independent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every plan-independent variable, so an exporting shell cannot change what a plan renders."""
    for name in PLAN_INDEPENDENT_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def exported_plan_independent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Export all four plan-independent variables at values that would change what a stage renders.

    Requested by each stage-plan file's own cleanup fixture, so it runs BEFORE that fixture strips
    them and every test in those files runs as if launched from an operator's shell carrying all four.
    A strip list that misses one leaves it exported for the whole file and the tests it changes go red
    here, rather than only on the machine of whoever happened to export it -- which is how
    `GAMES_VLLM_IS_CORRECTION=0`, a value the wave-4b launch recipe itself sets, came to fail the 9B
    plan's knob tests for a reason that had nothing to do with the 9B plan.
    """
    for name, value in PLAN_INDEPENDENT_ENV_SHELL_VALUES.items():
        monkeypatch.setenv(name, value)


def set_every_optional_knob(monkeypatch: pytest.MonkeyPatch, env_prefix: str) -> None:
    """Export every optional training knob and every optional switch under one plan's prefix."""
    for suffix, _flag, value in OPTIONAL_KNOB_SETTINGS:
        monkeypatch.setenv(f"{env_prefix}{suffix}", value)
    for suffix, _flag in OPTIONAL_SWITCH_SETTINGS:
        monkeypatch.setenv(f"{env_prefix}{suffix}", "1")


def clear_every_optional_knob(monkeypatch: pytest.MonkeyPatch, env_prefix: str) -> None:
    """Strip every one of them, so an operator's own shell cannot change what a test measures."""
    for suffix, _flag, _value in OPTIONAL_KNOB_SETTINGS:
        monkeypatch.delenv(f"{env_prefix}{suffix}", raising=False)
    for suffix, _flag in OPTIONAL_SWITCH_SETTINGS:
        monkeypatch.delenv(f"{env_prefix}{suffix}", raising=False)


type HelpEnvironment = tuple[tuple[str, str], ...]

OFFERED_FLAGS_CACHE: dict[tuple[str, HelpEnvironment], frozenset[str]] = {}


def help_relevant_environment() -> HelpEnvironment:
    """Snapshot the variables a CLI's ``--help`` may read, in a hashable, order-free form."""
    return tuple(
        sorted(
            (name, value)
            for name, value in os.environ.items()
            if name.startswith(CLI_ENV_PREFIX)
            and not name.startswith(PLAN_ENV_PREFIXES)
            and name not in PLAN_ENV_NAMES
        )
    )


def ask_help(module: str) -> str:
    """Run ``python -m <module> --help`` and return what it printed, or raise when it could not answer."""
    finished = subprocess.run(  # noqa: S603 - this interpreter, and a module name the plan under test rendered
        [sys.executable, "-m", module, "--help"], capture_output=True, text=True, check=False
    )
    if finished.returncode != 0:
        raise RuntimeError(
            f"{module} --help exited {finished.returncode}, so this check cannot say which flags "
            f"the CLI accepts. Its stderr: {finished.stderr.strip()}"
        )
    return finished.stdout


def offered_flags(module: str) -> frozenset[str]:
    """The flags a module's CLI accepts, asked of its ``--help`` once per (module, environment)."""
    key = (module, help_relevant_environment())
    cached = OFFERED_FLAGS_CACHE.get(key)
    if cached is None:
        cached = frozenset(FLAG.findall(ask_help(module)))
        OFFERED_FLAGS_CACHE[key] = cached
    return cached


def used_flags(argv: tuple[str, ...]) -> tuple[str, set[str]]:
    """Return the module a stage invokes and the flags it passes to it.

    Only the tokens after ``-m <module>`` count: the prefix carries ``uv run --frozen``, whose flags
    belong to uv rather than to the module under test.
    """
    index = argv.index("-m")
    module = argv[index + 1]
    return module, set(FLAG.findall(" ".join(argv[index + 2 :])))
