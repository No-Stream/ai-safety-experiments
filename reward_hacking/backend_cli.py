"""Shared ``--backend`` plumbing for every exploration CLI in this package.

Each probe CLI (episodes, channel, jagged, the agent harness) needs the same thing: pick a
backend kind, hand it the knobs that kind actually honours, and refuse the combinations that
cannot work. Doing that per CLI produced four copies of the same flag block, one of which
(``jagged``) knew about Bedrock while the others hardcoded HuggingFace, so a hosted model was
simply unreachable from three of the four. This module owns the flags and the construction so
all of them stay in step.

The load-bearing part is the refusal. ``BedrockSamplingConfig`` is deliberately not
``SamplingConfig`` (see its docstring): the Converse API's ``inferenceConfig`` has no ``topK``
field and no greedy switch, so a shared config type would drop ``top_k`` and ``do_sample`` with
no complaint from anybody -- exactly the silent-success failure class this repo keeps getting
bitten by. So the two config types stay separate, one is built per kind, and a knob aimed at a
kind that cannot honour it raises at parse time rather than being quietly ignored or blowing up
later as a raw ``AttributeError`` inside a backend.

Every knob defaults to ``None`` meaning "not given", which is what makes the refusal possible at
all: with argparse's usual concrete defaults there is no way to tell ``--top-k 20`` typed by a
human from the default nobody asked for. The effective default then comes from whichever config
type the chosen kind uses, so ``--backend bedrock`` omits ``temperature`` and ``topP`` from the
request entirely unless asked for -- the verified request shape, and the one that does not trip
models which reject a temperature outright.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from reward_hacking.model_backend import (
    DEFAULT_BEDROCK_MAX_TOKENS,
    Backend,
    BedrockBackend,
    BedrockSamplingConfig,
    SamplingConfig,
    build_backend,
)
from reward_hacking.recoverybench.budgets import MAX_TOKENS_BY_MODEL

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

BACKEND_KINDS = ("mock", "hf", "vllm", "bedrock", "codex")
LOCAL_KINDS = frozenset({"hf", "vllm"})
HOSTED_KINDS = frozenset({"bedrock", "codex"})

# No CLI here wants Qwen3.5 thinking traces on by default; they are opt-in per run.
DEFAULT_THINKING = False

UNSCREENED_OUTPUT_FLOOR = 16_384
"""The output-token floor for a model nobody has screened, and the provenance of that number.

Not a measurement of any run this module serves, and it must not be read as one. It is the smallest
per-model floor any termination screen in this repository justifies: ``games/termination.py``
screened Qwen3.5-4B over eight generation-only rollouts at a 32,768-token budget and all of them
closed their thinking block by 16,384, which is why the games trainer carries that figure as its
floor for an unscreened checkpoint. The 2B and 9B screens landed higher (24,576 and 32,768), so this
is a floor rather than a typical value, and the agentic turn lengths this package samples have never
been screened at all.

A floor rather than a target, and deliberately generous, for the reason
:data:`~reward_hacking.model_backend.DEFAULT_BEDROCK_MAX_TOKENS` gives: a cap is a ceiling and a
token is billed only when it is generated, so an over-wide cap costs nothing while a cap set below
what the model needs is paid for in whole runs read as non-compliance.
"""

PLUMBING_SMOKE_MODEL_ID = "Qwen/Qwen3.5-0.8B"
"""The tier that proves a CLI executes end to end, and nothing more.

Never a behavioural claim: at 0.8B "chose not to do the task" is indistinguishable from "could not",
and that distinction is what every lead in this package rests on. Shared across the probe CLIs so
the label travels with the value, rather than each one carrying a bare model string that reads like
a considered choice.
"""


@dataclass(frozen=True, slots=True)
class _Knob:
    """One CLI knob, the backend kinds that honour it, and why the others cannot."""

    dest: str
    flag: str
    kinds: frozenset[str]
    why: str

    def rejection(self, kind: str) -> str:
        """Build the error message for this knob being given to a backend that ignores it."""
        return (
            f"{self.flag} does not apply to --backend {kind}; only "
            f"{', '.join(sorted(self.kinds))} honour it. {self.why}"
        )


_KNOBS: tuple[_Knob, ...] = (
    _Knob(
        dest="thinking",
        flag="--thinking/--no-thinking",
        kinds=LOCAL_KINDS,
        why=(
            "Thinking selects a branch of the local chat template. A hosted endpoint applies its "
            "own template; on Bedrock the nearest control is --reasoning-effort."
        ),
    ),
    _Knob(
        dest="temperature",
        flag="--temperature",
        kinds=LOCAL_KINDS | {"bedrock"},
        why="The codex CLI exposes no decoding knobs, and the mock backend does not sample.",
    ),
    _Knob(
        dest="top_p",
        flag="--top-p",
        kinds=LOCAL_KINDS | {"bedrock"},
        why="The codex CLI exposes no decoding knobs, and the mock backend does not sample.",
    ),
    _Knob(
        dest="top_k",
        flag="--top-k",
        kinds=LOCAL_KINDS,
        why=(
            "Bedrock Converse's inferenceConfig has no topK field, so a value passed here would "
            "be dropped without any error from the API -- which is why refusing it is the whole "
            "point of this check rather than a nicety."
        ),
    ),
    _Knob(
        dest="max_new_tokens",
        flag="--max-new-tokens",
        kinds=LOCAL_KINDS | {"bedrock"},
        why=(
            "It becomes maxTokens on Bedrock. The codex CLI caps its own output and the mock "
            "backend returns canned strings."
        ),
    ),
    _Knob(
        dest="concurrency",
        flag="--concurrency",
        kinds=frozenset({"bedrock"}),
        why="It sizes the Converse thread pool; the other backends batch or run in-process.",
    ),
    _Knob(
        dest="region",
        flag="--region",
        kinds=frozenset({"bedrock"}),
        why="Only the Bedrock client is region-scoped.",
    ),
    _Knob(
        dest="profile",
        flag="--profile",
        kinds=frozenset({"bedrock"}),
        why="Only the Bedrock client resolves AWS credentials.",
    ),
    _Knob(
        dest="reasoning_effort",
        flag="--reasoning-effort",
        kinds=HOSTED_KINDS,
        why=(
            "Local Qwen3.5 checkpoints have no effort ladder; use --thinking for their reasoning "
            "traces instead."
        ),
    ),
    _Knob(
        dest="vllm_quantization",
        flag="--vllm-quantization",
        kinds=frozenset({"vllm"}),
        why=(
            "Online quantization happens inside the vLLM engine as the weights load. The hf "
            "backend loads the checkpoint's own dtype, and the hosted kinds run someone else's "
            "weights, so the flag would be silently meaningless anywhere else."
        ),
    ),
    _Knob(
        dest="vllm_gpu_memory_utilization",
        flag="--vllm-gpu-memory-utilization",
        kinds=frozenset({"vllm"}),
        why=(
            "It is the fraction of the card's TOTAL memory the vLLM engine claims at start-up. "
            "The hf backend allocates as it goes and the hosted kinds run on someone else's card, "
            "so anywhere else the flag would describe a claim nobody makes."
        ),
    ),
    _Knob(
        dest="vllm_max_model_len",
        flag="--vllm-max-model-len",
        kinds=frozenset({"vllm"}),
        why=(
            "It sizes the vLLM engine's KV cache for the longest sequence it will accept. The hf "
            "backend grows its cache per call and the hosted kinds size their own, so anywhere "
            "else the flag would be silently meaningless."
        ),
    ),
    _Knob(
        dest="vllm_max_num_seqs",
        flag="--vllm-max-num-seqs",
        kinds=frozenset({"vllm"}),
        why=(
            "It caps the vLLM engine's concurrent sequences, and with them its CUDA-graph "
            "captures. The hf backend batches per call and the hosted kinds schedule their own, "
            "so anywhere else the flag would be silently meaningless."
        ),
    ),
    _Knob(
        dest="min_p",
        flag="--min-p",
        kinds=LOCAL_KINDS,
        why=(
            "Both local samplers honour min_p (transformers' GenerationConfig has the field and "
            "vLLM's SamplingParams takes it). Converse's inferenceConfig has no minimum-probability "
            "field, so Bedrock would drop it as silently as it drops topK."
        ),
    ),
    _Knob(
        dest="repetition_penalty",
        flag="--repetition-penalty",
        kinds=LOCAL_KINDS,
        why=(
            "This is the transformers-native anti-loop lever the HF path uses in place of "
            "presence_penalty, which it cannot apply, so it must be reachable on hf as well as "
            "vllm. Converse exposes no penalty field at all."
        ),
    ),
    _Knob(
        dest="presence_penalty",
        flag="--presence-penalty",
        kinds=frozenset({"vllm"}),
        why=(
            "Only vLLM's sampler applies presence_penalty; transformers' generate has no such "
            "field, so HFBackend never forwards it (its SamplingConfig docstring records the "
            "split), and Converse's inferenceConfig has no penalty field either. Anywhere but "
            "vllm the flag would change the recorded sampling metadata while changing nothing "
            "about generation -- the trace would claim a sampler that never ran."
        ),
    ),
)


def add_backend_choice(parser: argparse.ArgumentParser, *, default: str = "hf") -> None:
    """Add only ``--backend``, for a CLI that cannot use the shared decoding knobs.

    The activation-capture probe is the case: it needs the flag so pointing it at a hosted model
    fails with an explanation, but temperature and concurrency mean nothing to it.
    """
    parser.add_argument(
        "--backend",
        default=default,
        choices=BACKEND_KINDS,
        help=f"Inference backend (default: {default}).",
    )


def add_backend_args(parser: argparse.ArgumentParser, *, default: str = "hf") -> None:
    """Add ``--backend`` plus every decoding and hosted-API knob, all defaulting to unset.

    Unset means "let the chosen backend's own config decide", so the effective default differs
    between the local and Bedrock config types on purpose. A knob given to a kind that cannot
    honour it is refused by ``backend_from_args``, not silently dropped.
    """
    add_backend_choice(parser, default=default)
    parser.add_argument(
        "--thinking", action="store_true", default=None, help="Enable Qwen3.5 thinking traces."
    )
    parser.add_argument(
        "--no-thinking", dest="thinking", action="store_false", help="Disable thinking traces."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature. High temperature with --no-thinking approximates a base-like "
            "policy (lead #6). Omitted from the Bedrock request unless given, because some hosted "
            "models reject the field outright."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help=(
            "Nucleus-sampling cutoff. Pass 1.0 together with --top-k 0 to disable truncation "
            "entirely (the de-saturated arm that tells a real behavioral null apart from the "
            "MoneyWorld saturated-logit artifact)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k cutoff; pass 0 to disable top-k truncation. Local backends only.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Output-token cap per call (maxTokens on Bedrock). Unset uses the chosen backend's own "
            f"budget: {DEFAULT_BEDROCK_MAX_TOKENS} on Bedrock, the thinking-mode preset locally. "
            "A hosted reasoning model spends this budget on reasoning it may not even return, so a "
            "cap below its trace length reads as a refusal to answer rather than as a config "
            "mistake -- which is why the agent harness refuses a value below the resolved model's "
            "measured floor (see refuse_short_output_cap) instead of trusting this warning."
        ),
    )
    parser.add_argument(
        "--concurrency", type=int, default=None, help="Bedrock Converse worker count."
    )
    parser.add_argument("--region", default=None, help="Bedrock region.")
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Named AWS profile for Bedrock; pass an empty string to use the ambient credential "
            "chain (what a Batch container with an instance role has)."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Hosted-model effort level. On --backend bedrock it is checked against the model "
            "family's ladder before any call is made. On --backend codex nothing checks it: it is "
            "forwarded verbatim as `-c model_reasoning_effort=<level>` to a CLI whose ladder "
            "differs from Bedrock's, and codex loads that config without validating the value "
            "(measured: `codex debug prompt-input -c model_reasoning_effort=bogus-effort-level` "
            "exits 0), so a typo reaches the endpoint rather than being refused here. Check the "
            "level against the codex build you are running before spending a sweep on it."
        ),
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help=(
            "Minimum-probability cutoff, relative to the top token. Unset keeps the base sampler's "
            "value, which is 0.0 (off) in both Qwen3.5 presets. Local backends only."
        ),
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help=(
            "Anti-loop lever transformers can actually apply, unlike presence_penalty; 1.0 is off "
            "and is what both presets carry. This is the fallback the HFBackend docstring names for "
            "a run that loops inside <think>, and until this flag existed there was no way to reach "
            "it. A penalty is a behavioural intervention, so a run that raises it is not comparable "
            "with the penalty-free runs on record; raise it to escape a loop, not by default."
        ),
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help=(
            "vLLM presence penalty. Unset keeps the base sampler's own value (the thinking preset "
            "carries 1.5 as its anti-loop lever); an explicit 0 forces it off, which is how an "
            "eval reproduces a penalty-free training sampler. vllm only: transformers cannot "
            "apply it, so on any other backend the flag is refused rather than silently recorded "
            "as a sampler that never ran."
        ),
    )
    parser.add_argument(
        "--vllm-quantization",
        choices=("fp8",),
        default=None,
        help=(
            "Online engine quantization for inference-only vLLM passes. fp8 measured 1.34x over "
            "bf16 on Qwen3.5-2B decode (1000.1 vs 748.6 tok/s, forced 1024+2048, GDN layers "
            "included); output fidelity is spot-checked rather than proven, so never use it for a "
            "run whose absolute levels will be compared against a bf16 run. Recorded in the trace "
            "meta so an fp8 artifact can never read as a bf16 measurement."
        ),
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "Fraction of the card's TOTAL memory the vLLM engine claims at start-up (vLLM's own "
            "default is 0.9). vLLM refuses to start when the card's FREE memory is below that "
            "claim, so on a shared card -- the dev box's L4 with sibling sessions' CUDA contexts "
            "on it -- a smoke needs a smaller claim; a rented single-tenant card should leave it "
            "unset. Never a lever for a behavioural run's numbers: it sizes the KV cache, not the "
            "sampler. vllm only."
        ),
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help=(
            "Longest prompt-plus-completion sequence the vLLM engine accepts, in tokens. Unset "
            "takes the checkpoint's own window (262,144 for Qwen3.5), and the engine refuses to "
            "start unless its KV cache can hold one sequence of that length -- which on a card "
            "shared with other processes it cannot, so a smoke on the dev box's L4 needs a value "
            "sized to its prompts. The legibility probe derives the same number from its prompt "
            "and completion budgets; the harness's transcript grows per turn, so here it is "
            "explicit. vllm only."
        ),
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=None,
        help=(
            "Most sequences the vLLM engine runs at once; vLLM also sizes its CUDA-graph captures "
            "from it. Unset keeps vLLM's default, whose graph-memory estimate for Qwen3.5-9B "
            "(~9 GiB) left no KV cache on a 32 GB card at 0.80 utilization. Long-thinking cells "
            "fit only ~5-12 full-length sequences anyway, so 64 loses nothing. vllm only."
        ),
    )


def reject_inapplicable_knobs(kind: str, args: argparse.Namespace) -> None:
    """Raise if the run asked for a knob the chosen backend cannot honour.

    Every knob this checks is one a backend would otherwise ignore in silence, so the check is
    the only thing standing between a mis-specified sweep and a table of numbers nobody can
    interpret. Knobs a CLI never registered are skipped, so a CLI can adopt a subset.
    """
    for knob in _KNOBS:
        if getattr(args, knob.dest, None) is None:
            continue
        if kind not in knob.kinds:
            raise ValueError(knob.rejection(kind))


def _require_registered_knobs(args: argparse.Namespace) -> None:
    """Raise a readable error when the namespace never got the shared knobs registered on it.

    Without this, a CLI that added only ``--backend`` and then asked for a backend would die on a
    raw ``AttributeError`` from deep inside a resolver -- the same opaque failure mode that handing
    a backend the wrong config type produces.
    """
    missing = [knob.flag for knob in _KNOBS if not hasattr(args, knob.dest)]
    if missing:
        raise ValueError(
            f"this parser is missing {', '.join(missing)}, so no backend config can be resolved "
            "from it; call add_backend_args instead of add_backend_choice"
        )


def _resolved[T](given: T | None, fallback: T) -> T:
    """Return the flag value when it was given, else the config default (0 counts as given)."""
    return fallback if given is None else given


def resolve_thinking(args: argparse.Namespace) -> bool:
    """Resolve the ``--thinking/--no-thinking`` flag to a bool, applying the shared default.

    ``--thinking`` defaults to ``None`` ("not given") so an explicit choice is distinguishable from
    the default; unset falls back to :data:`DEFAULT_THINKING`. Shared so the sampler preset a CLI
    picks from the mode and the ``thinking=`` a backend is constructed with cannot drift apart.
    """
    return _resolved(args.thinking, DEFAULT_THINKING)


def local_sampling_from_args(
    args: argparse.Namespace, defaults: SamplingConfig | None = None
) -> SamplingConfig:
    """Build the HuggingFace/vLLM decoding config: the base preset with the given flags on top.

    ``defaults`` is the calling CLI's base sampler. Unset, it is the mode-correct Qwen3.5 preset for
    ``--thinking`` (:meth:`SamplingConfig.for_thinking`), so an unconfigured CLI still gets top_p
    0.95 and the 32768-token budget under thinking rather than the non-thinking preset that loops
    the model inside ``<think>``. Only the flags actually given override the base, and the override
    is field-wise via :func:`dataclasses.replace`, so an unasked-for field carries through from the
    base instead of silently resetting to ``SamplingConfig()``'s field defaults -- dropping the
    thinking preset's ``presence_penalty=1.5`` is exactly how the vLLM path lost its anti-loop lever
    and the recorded metadata under-reported it. Every one of the seven fields is now reachable
    except ``do_sample``, which comes from the base and is never a flag: every exploration lead here
    reads the spread of a sampled policy, so greedy decoding is a per-CLI decision, not a per-run one.
    """
    base = (
        defaults
        if defaults is not None
        else SamplingConfig.for_thinking(thinking=resolve_thinking(args))
    )
    overrides: dict[str, object] = {}
    if args.max_new_tokens is not None:
        overrides["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.top_p is not None:
        overrides["top_p"] = args.top_p
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.min_p is not None:
        overrides["min_p"] = args.min_p
    if args.repetition_penalty is not None:
        overrides["repetition_penalty"] = args.repetition_penalty
    if args.presence_penalty is not None:
        overrides["presence_penalty"] = args.presence_penalty
    return replace(base, **overrides)


def bedrock_sampling_from_args(args: argparse.Namespace) -> BedrockSamplingConfig:
    """Build the Converse decoding config, leaving unasked-for fields out of the request.

    ``temperature`` and ``top_p`` stay ``None`` unless given: ``None`` means the field is omitted,
    which is the request shape verified across the model roster. Defaulting them would send a
    temperature to models that refuse one.

    ``max_tokens`` is the exception and cannot work that way -- ``converse_inference_config`` always
    writes ``maxTokens`` -- so an unset ``--max-new-tokens`` falls back to
    :data:`~reward_hacking.model_backend.DEFAULT_BEDROCK_MAX_TOKENS`, which is high on purpose. This
    is the line that decided the cap for every probe CLI in the package, and while it resolved to
    2048 it truncated 37% of the Converse calls in a 400-episode agentic run into what read as
    refusals. Read the constant's docstring before lowering anything here.
    """
    return BedrockSamplingConfig(
        max_tokens=_resolved(args.max_new_tokens, BedrockSamplingConfig().max_tokens),
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning_effort=args.reasoning_effort,
    )


def output_floor_for(model_id: str) -> int:
    """Return the smallest output-token cap this model may be sampled at.

    The measured rows come from :data:`~reward_hacking.recoverybench.budgets.MAX_TOKENS_BY_MODEL`,
    read rather than restated so the number exists once: that table is where a model's budget is
    recorded when somebody probes it, and a second copy here is how one of the two goes lenient.
    Everything else falls back to :data:`UNSCREENED_OUTPUT_FLOOR`, which is a fallback and not a
    measurement -- see its docstring before treating it as one.

    Falling back rather than refusing, which is the opposite of what
    ``budgets.max_tokens_for`` does with an unlisted model, and the two answer different questions.
    That one picks the cap a benchmark sweep will *run at*, where guessing spends the sweep. This
    one only says how low a cap may go, and it is asked about every model these CLIs can reach,
    including the local plumbing tiers no roster will ever list; refusing there would make an
    unmeasured model unrunnable rather than merely unscreened.
    """
    return MAX_TOKENS_BY_MODEL.get(model_id, UNSCREENED_OUTPUT_FLOOR)


def refuse_short_output_cap(model_id: str, cap: int, *, allow_short: bool) -> None:
    """Refuse an output cap below what this model needs to finish, unless the run asked for it.

    A refusal rather than a warning, for the reason the games trainer refuses the same mistake: a
    warning is exactly what the run that cost a night would have printed and nobody would have read.
    Below the floor a reply is cut off before it answers, and nothing downstream distinguishes that
    from a policy that declined to answer -- a truncated agentic turn ends the episode at the
    give-up path, and a truncated benchmark reply is scored as a missing answer.

    ``allow_short`` is the deliberate way out, for a timing or plumbing probe where clipped
    reasoning is understood and the run is not a measurement. It warns rather than passing in
    silence, because the artifacts it produces will read like a policy that stopped acting and the
    log is the only place that can say otherwise. The caller owns that flag (the agent harness
    registers ``--allow-short-completions``); this function only takes the resolved answer, and the
    message names the flag so a refusal is actionable rather than a wall.
    """
    floor = output_floor_for(model_id)
    if cap >= floor:
        return
    provenance = (
        "this model's own measured output budget"
        if model_id in MAX_TOKENS_BY_MODEL
        else "the floor for an unscreened model, since nothing has measured this one"
    )
    if allow_short:
        logger.warning(
            "sampling %s at %d output tokens, below the %d it needs (%s): replies may be cut off "
            "before they answer, and a truncated reply is indistinguishable downstream from one "
            "that chose to say nothing. --allow-short-completions was given, so this run proceeds "
            "and is not a measurement",
            model_id,
            cap,
            floor,
            provenance,
        )
        return
    raise ValueError(
        f"--max-new-tokens {cap} is below the {floor} output tokens {model_id} needs; that floor "
        f"is {provenance}. A cap under it truncates the reply before it answers, which reads "
        f"downstream as a policy that declined to act rather than as a config mistake -- at 2048 "
        f"it turned 37% of a 400-episode run into apparent give-ups. Raise the cap (it is a "
        f"ceiling, so it bills nothing it does not generate), or pass --allow-short-completions "
        f"for a plumbing or timing probe whose traces are not a measurement."
    )


def _bedrock_client_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Collect the Bedrock client settings that were given, so the rest keep their defaults."""
    kwargs: dict[str, object] = {}
    if args.concurrency is not None:
        kwargs["concurrency"] = args.concurrency
    if args.region is not None:
        kwargs["region"] = args.region
    if args.profile is not None:
        # An explicit empty string means the ambient credential chain, not the default profile.
        kwargs["profile"] = args.profile or None
    return kwargs


def backend_from_args(  # noqa: PLR0913 - keyword-only per-CLI overrides, not worth a wrapper object
    args: argparse.Namespace,
    model_id: str,
    *,
    local_sampling: SamplingConfig | None = None,
    bedrock_sampling: BedrockSamplingConfig | None = None,
    mock_responses: Sequence[str] | None = None,
    extra_kwargs: Mapping[str, object] | None = None,
) -> Backend:
    """Build the backend the run asked for, with the config type that backend actually takes.

    ``local_sampling`` carries the calling CLI's own decoding defaults (the episode runner keeps a
    high base-like temperature, for instance) and is used only by the local kinds.
    ``bedrock_sampling`` is its Converse-side counterpart, for a CLI whose base config carries a
    field no flag reaches (the agent harness's ``</run>`` stop sequence); it must already fold the
    CLI flags in -- pass ``bedrock_sampling_from_args(args)`` with the extra fields replaced --
    because handing one in replaces the resolution here, and left ``None`` the flags resolve as
    before. ``mock_responses`` are the canned completions that make ``--backend mock`` a zero-cost
    end-to-end smoke of the CLI; a CLI that supplies none cannot offer that kind.

    ``extra_kwargs`` are construction arguments a caller derived rather than a user typed, and it
    is how a local backend is told to serve a LoRA adapter un-merged: :func:`games.eval_model.
    resolve_served_model` decides the loading mode and returns the kwargs that mode needs
    (``lora_adapter`` and the engine's LoRA settings, or a float32 ``dtype``). They deliberately
    bypass the knob registry, which exists to police CLI flags against the backend that has to
    honour them -- these were computed from a checkpoint, not offered as options.
    """
    kind: str = args.backend
    _require_registered_knobs(args)
    reject_inapplicable_knobs(kind, args)

    if kind == "mock":
        if mock_responses is None:
            raise ValueError(
                "--backend mock needs canned responses, and this CLI supplies none; "
                f"choose one of {', '.join(k for k in BACKEND_KINDS if k != 'mock')}"
            )
        # Said loudly because the mock's model_id lands in the artifacts as if a model had run.
        logger.warning(
            "mock backend: nothing is sampled, and any trace will read model_id=%r", model_id
        )
        return build_backend("mock", model_id, responses=list(mock_responses))
    if kind in LOCAL_KINDS:
        # Of the CLI knobs only the --vllm-* flags are engine kwargs, and the registry above has
        # already refused them on every other kind; a caller's extra_kwargs join them here.
        engine_kwargs: dict[str, object] = dict(extra_kwargs or {})
        if kind == "vllm" and args.vllm_quantization is not None:
            engine_kwargs["quantization"] = args.vllm_quantization
        if kind == "vllm" and args.vllm_gpu_memory_utilization is not None:
            engine_kwargs["gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        if kind == "vllm" and args.vllm_max_model_len is not None:
            engine_kwargs["max_model_len"] = args.vllm_max_model_len
        if kind == "vllm" and args.vllm_max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = args.vllm_max_num_seqs
        return build_backend(
            kind,
            model_id,
            thinking=resolve_thinking(args),
            sampling=local_sampling_from_args(args, local_sampling),
            **engine_kwargs,
        )
    if kind == "bedrock":
        return build_backend(
            "bedrock",
            model_id,
            sampling=bedrock_sampling
            if bedrock_sampling is not None
            else (bedrock_sampling_from_args(args)),
            **_bedrock_client_kwargs(args),
        )
    if kind == "codex":
        return build_backend("codex", model_id, reasoning_effort=args.reasoning_effort)
    raise ValueError(f"unknown backend kind {kind!r}; expected one of {', '.join(BACKEND_KINDS)}")


def require_in_process_weights(kind: str, *, purpose: str) -> None:
    """Raise unless ``kind`` runs the weights in this process, as activation capture requires.

    Only ``hf`` qualifies. A hosted endpoint returns text, never a residual stream, so a
    ``--backend bedrock`` activation probe is not a slow version of the real thing -- there is no
    quantity to measure. ``vllm`` does hold local weights but exposes no hidden states through
    ``VLLMBackend``, and ``mock`` has no weights at all, so both are refused for the same reason:
    nothing here could read an activation out of them.
    """
    if kind == "hf":
        return
    raise ValueError(
        f"--backend {kind} cannot {purpose}: it never exposes the residual stream this reads via "
        "forward hooks on the decoder layers. Only --backend hf loads weights in this process "
        "with the hidden states reachable; a hosted API returns text only."
    )


def log_token_usage(backend: Backend) -> None:
    """Log the live Converse backend's accumulated token usage, so its cost is on the record.

    A no-op for the local backends, which have no per-call token accounting to report, and for two
    hosted, billable ones that this cannot speak for. ``bedrock_batch.BedrockBatchBackend`` logs its
    own run totals and a cost estimate when it collects a job, so its spend is recorded elsewhere.
    ``CodexBackend`` is the gap: it shells the ``codex`` CLI and captures only the reply text, so it
    accumulates no token counts at all, and a codex sweep's cost is recoverable from the codex
    session logs rather than from anything this package writes.

    Naming the one concrete class is deliberate rather than an oversight: ``backend_from_args`` does
    build the codex backend, but there is no usage attribute on it to probe.
    """
    if isinstance(backend, BedrockBackend):
        logger.info(
            "bedrock token usage for %s | in=%d out=%d",
            backend.model_id,
            backend.usage.input_tokens,
            backend.usage.output_tokens,
        )
