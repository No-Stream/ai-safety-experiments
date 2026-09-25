"""The eval-time sampler modes: what decoding config a science eval runs under, and why.

Every battery cell before 2026-08-20 sampled under the vendor thinking preset, whose
``presence_penalty=1.5`` rode along from `SamplingConfig.for_thinking`. That penalty is not a
neutral anti-loop lever at measurement time: on identical prompts and checkpoints it removed about
three quarters of the trained movement the eval exists to read (twin-pd-group training frames,
step 0 to 70: -0.233 at pp=0 against -0.055 at pp=1.5) and halved deliberation length
(`docs/scratch/games-readout-notes-2026-08-20/16-trainframes-dissoc.md`; the n=32 re-sample entry
at the end of `docs/games-predictions.md`). So the default science sampler is the TRAINING
distribution -- the policy the arms were optimised under is the policy the eval measures -- and
the vendor-flavoured sampler is an explicit second leg, not a default anything rides in on.

Two modes, named for what they measure rather than as opaque labels:

*   ``training-distribution`` (the default): GRPO's own generation defaults, read from
    `games.generation` so the eval and the trainer cannot drift apart -- temperature 1.0, top_p
    1.0, top_k 0, min_p 0, repetition_penalty 1.0, presence_penalty 0.0.
*   ``training-run``: the sampler recorded by one run's ``run_config.json`` -- the run's own
    temperature, top_p, top_k and completion cap, with the other evaluation penalties fixed to
    their neutral values.
*   ``deployment``: the vendor thinking preset a deployed model would run
    (`SamplingConfig.for_thinking`), with ``presence_penalty`` forced to 0 -- for thinking mode
    that is temperature 1.0, top_p 0.95, top_k 20, pp 0. The penalty stays off here too because
    the measurement above shows it suppresses expressed effects, and a "deployment" leg that
    cannot see the effect measures nothing; the truncation-mode preset itself
    (`reward_hacking.model_backend`) is other projects' vendor default and is deliberately not
    edited.

Both modes carry the same completion budget, because the budget is a censoring knob rather than a
distribution knob: 13 battery cells read HIGH TRUNCATION at the old 24,576-token cap while the
dt-probes' finished p90 was ~17k tokens with a truncated median near the cap, so the cap was
shaping the measurement (`14-dt-capabilities.md` section 1). Per the never-cap rule the default
sits above the observed finished tail; ``--max-new-tokens`` still overrides it, as does every
other explicit decoding flag -- a mode is a base, never a straitjacket.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

from games.generation import TRAINING_TEMPERATURE, TRAINING_TOP_K, TRAINING_TOP_P
from reward_hacking import backend_cli
from reward_hacking.model_backend import SamplingConfig

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping

SAMPLER_TRAINING_DISTRIBUTION = "training-distribution"
SAMPLER_TRAINING_RUN = "training-run"
SAMPLER_DEPLOYMENT = "deployment"
SAMPLER_MODES: tuple[str, ...] = (
    SAMPLER_TRAINING_DISTRIBUTION,
    SAMPLER_TRAINING_RUN,
    SAMPLER_DEPLOYMENT,
)
DEFAULT_SAMPLER_MODE = SAMPLER_TRAINING_DISTRIBUTION

DEFAULT_EVAL_MAX_NEW_TOKENS = 32768
"""Above the dt-probes' finished p90 (~17k tokens), with the whole observed finished tail under it.

The 24,576 cap this replaces produced 13 HIGH TRUNCATION battery cells and deleted the
large-stakes probe items non-randomly (14-dt-capabilities.md sections 1.3-1.5). A cap is a
ceiling -- tokens are only paid for when generated -- so the width costs nothing.
"""


def add_sampler_arg(parser: argparse.ArgumentParser) -> None:
    """Register ``--sampler``, defaulting to None so an explicit choice is distinguishable.

    The unset default resolves to :data:`DEFAULT_SAMPLER_MODE` on the local backends; keeping the
    argparse default None is what lets :func:`resolve_sampler_mode` refuse an explicit mode aimed
    at a backend that cannot honour it, mirroring `reward_hacking.backend_cli`'s knob registry.
    """
    parser.add_argument(
        "--sampler",
        choices=SAMPLER_MODES,
        default=None,
        help=(
            f"Eval sampler mode (default: {DEFAULT_SAMPLER_MODE}). {SAMPLER_TRAINING_DISTRIBUTION} "
            f"is the training policy's own sampler (temperature 1.0, top_p 1.0, top_k 0, no "
            f"presence penalty); {SAMPLER_TRAINING_RUN} reuses the four sampler values from a "
            f"run's run_config.json; {SAMPLER_DEPLOYMENT} is the vendor thinking preset with the "
            f"presence penalty forced off (top_p 0.95, top_k 20). Explicit decoding flags "
            f"override the mode field-wise. Local backends only."
        ),
    )


def resolve_sampler_mode(args: argparse.Namespace) -> str:
    """Resolve ``--sampler`` to a mode name, refusing a mode no local sampler will ever apply.

    Only the local backends run the sampler these modes describe; on a hosted or mock backend an
    explicit ``--sampler`` would change nothing about generation while the run's operator believes
    it did -- the silently-ignored-knob failure `backend_cli.reject_inapplicable_knobs` exists to
    block, applied here to a flag that module does not own.
    """
    given = getattr(args, "sampler", None)
    if given is None:
        return DEFAULT_SAMPLER_MODE
    if args.backend not in backend_cli.LOCAL_KINDS:
        raise ValueError(
            f"--sampler {given} does not apply to --backend {args.backend}; only "
            f"{', '.join(sorted(backend_cli.LOCAL_KINDS))} run the local sampler these modes "
            f"configure, so anywhere else the flag would be recorded but never applied."
        )
    return str(given)


def sampler_mode_meta(args: argparse.Namespace) -> str | None:
    """Return the mode name for a trace's meta record, or None where no local sampler runs.

    None rather than the default's name on hosted and mock backends, so the meta field can never
    claim a sampler that never sampled.
    """
    if args.backend not in backend_cli.LOCAL_KINDS:
        return None
    return resolve_sampler_mode(args)


def _training_run_sampling(training_run: Mapping[str, object]) -> SamplingConfig:
    """Build a sampler from the four values recorded in a run's config block."""
    required = ("temperature", "top_p", "top_k", "max_completion_tokens")
    missing = tuple(name for name in required if name not in training_run)
    if missing:
        raise ValueError(
            "training-run sampler requires run facts for temperature, top_p, top_k and "
            f"max_completion_tokens; missing {', '.join(missing)}."
        )
    return SamplingConfig(
        max_new_tokens=int(cast("int | float", training_run["max_completion_tokens"])),
        do_sample=True,
        temperature=float(cast("int | float", training_run["temperature"])),
        top_p=float(cast("int | float", training_run["top_p"])),
        top_k=int(cast("int | float", training_run["top_k"])),
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
    )


def eval_sampling(
    mode: str, *, thinking: bool, training_run: Mapping[str, object] | None = None
) -> SamplingConfig:
    """Return the decoding config a mode resolves to, before any explicit flag overrides it.

    ``thinking`` selects the vendor preset branch for the deployment mode; the training
    distribution ignores it deliberately, because GRPO samples the same three knobs whichever
    chat-template branch the prompt rendered (the thinking switch itself reaches the backend
    separately). The training-run mode instead requires the mapping of sampler values read from
    that run's ``run_config.json`` config block. Neither fixed mode copies training's tighter
    completion budget: the budget censors
    rather than reweights, and the module docstring carries the truncation evidence.
    """
    if mode == SAMPLER_TRAINING_DISTRIBUTION:
        return SamplingConfig(
            max_new_tokens=DEFAULT_EVAL_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TRAINING_TEMPERATURE,
            top_p=TRAINING_TOP_P,
            top_k=TRAINING_TOP_K,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
        )
    if mode == SAMPLER_TRAINING_RUN:
        if training_run is None:
            raise ValueError(
                "training-run sampler requires run facts; pass the run's recorded sampler values."
            )
        return _training_run_sampling(training_run)
    if mode == SAMPLER_DEPLOYMENT:
        return replace(
            SamplingConfig.for_thinking(thinking=thinking),
            presence_penalty=0.0,
            max_new_tokens=DEFAULT_EVAL_MAX_NEW_TOKENS,
        )
    raise ValueError(f"unknown sampler mode {mode!r}; expected one of {list(SAMPLER_MODES)}.")
