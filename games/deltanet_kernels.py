"""Put fla's fused recurrent Gated DeltaNet kernel on decode, under the name transformers looks up.

transformers resolves each Gated DeltaNet kernel function by importing `fla` and walking a dotted
path from `transformers.integrations.hub_kernels._KERNELS_INTERNAL_PATH_MAPPINGS`, and when the walk
misses it substitutes a pure-PyTorch loop inside a bare `except Exception` with nothing logged. On
Qwen3.5 the prefill kernel is found and the decode kernel is not, for one reason: fla exports the
per-token kernel as `fused_recurrent_gated_delta_rule` while transformers asks for
`recurrent_gated_delta_rule`. A name, nothing else -- the same kernel, the same arguments, the same
2-tuple back.

Three quarters of the Qwen3.5 stack is Gated DeltaNet (18 of 2B's 24 layers) and decode calls the
kernel once per such layer per token, so the fallback is paid on every step -- 558 calls verified
over a 32-token greedy generation, which is 18 layers times 31 decode steps. What it costs was
measured rather than assumed, on this box's L4 with Qwen3.5-2B in bf16, greedy, and it depends
strongly on how many sequences decode together:

    1 sequence,  256 new tokens:   32.3 ->  35.1 tokens/s/sequence   (1.09x)
    8 sequences, 1024 new tokens:  245.6 ->  264.7 tokens/s total    (1.08x)
    32 sequences, 1024 new tokens: 587.2 ->  892.2 tokens/s total    (1.52x)
    64 sequences, 1024 new tokens: 611.7 -> 1186.9 tokens/s total    (1.94x)

So the popular framing -- that the fallback is a slow Python loop -- is wrong: at decode
`sequence_length` is 1, so its loop runs a single iteration and the cost is a handful of small
elementwise kernels, launch-bound at batch 1. The win appears once those elementwise ops have real
work to do, which is at the batch an RL group actually generates (8 prompts x 8 generations = 64).
Either way it is invisible without a check: the model loads, generates sensible text, and reports
nothing.

Registering the missing name has to happen BEFORE
`transformers.models.qwen3_5.modeling_qwen3_5` is imported, because
`use_kernel_func_from_hub_with_fallback` resolves the implementation at decoration time and closes
over it. Patching afterwards changes nothing while looking like it worked, so `bridge_decode_kernel`
raises rather than no-op when it is called too late.

The scale convention was the one thing worth checking before aliasing, since the torch fallback
passes `scale = 1 / sqrt(query.shape[-1])` explicitly and transformers never passes `scale` to the
kernel. fla's wrapper defaults it to `k.shape[-1] ** -0.5` (`fla/ops/gated_delta_rule/
fused_recurrent.py`, the `if scale is None` branch), and query and key share their last dimension
here, so the two agree and no scale override is needed. The rest of the conventions line up the same
way and were read off the source rather than assumed: `g` is log-space decay in both
(`use_gate_in_kernel=False`), `beta` is post-sigmoid in both (`use_beta_sigmoid_in_kernel=False`),
and the recurrent state is `[K, V]`-major in both (`state_v_first=False`).

`assert_decode_kernels_match` is what keeps that reasoning honest: it runs both implementations on
decode-shaped inputs and compares them, which is the only form of the claim that can fail out loud.

**The bridge is stat-grade, not bit-grade, and every record says which kernel it ran under.** Probe
I1 (2026-09-02, `docs/scratch/hot-path-optimization-2026-09-02/probes/I1.md`) measured 0.8B greedy
decode at 1.14x-1.51x per step under the fused kernel, with greedy token ids diverging from step 21
on: the two implementations reduce in different orders and the fallback rounds q/k to bf16 before
normalising, so near-tie argmaxes flip. Nothing about the sampled distribution changes, but two
records decoded under different kernels are not the same measurement at the token level. So every
HuggingFace-path interp record carries `DELTANET_KERNEL_FIELD`, the implementation each DeltaNet
kernel function was ACTUALLY bound to when the model ran (`bound_deltanet_kernels`, read off the
modeling module's own wrappers rather than predicted from the fla module), and every analysis that
pools records refuses a pool that mixes two bindings (`assert_one_deltanet_kernel`). What a record
names depends on what its forwards dispatched. A generating leg runs all four kernel functions (the
chunked pair on the prompt, the per-token pair on every decode step) and records all four. A
forward-only leg (activation capture, activation patching, the lens fit) never reaches the decode pair
at all -- transformers takes `recurrent_gated_delta_rule` and `causal_conv1d_update` only when the
cache already holds a state for the layer and the step is one token, which no bare forward satisfies
-- so it records `DELTANET_PREFILL_KERNELS` only (`prefill_deltanet_kernels`). That is what makes the
bridge invisible to those records: it re-binds only the decode kernel, so two forward-only records on
either side of it name the same kernels and their tensors are bit-identical, and a resume ledger or a
mixing guard keyed on the field lets them through. The run-level summary still carries the bridge
report and the full four-kernel binding as process provenance.

Verified against transformers 5.15.0 and flash-linear-attention 0.5.2 on 2026-08-18. Gating is on
the observed attribute rather than on those versions, so a release that fixes the export name turns
this into a no-op instead of a double patch.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import torch

from games.preflight import deltanet_kernel_paths

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

# The decode kernel contract, both implementations of it: query/key/value positionally, the rest
# by keyword, and (output, final_recurrent_state) back -- with the state None unless
# `output_final_state` was asked for.
type DecodeKernel = Callable[..., tuple[torch.Tensor, torch.Tensor | None]]

QWEN3_5_MODELING_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
FLA_GATED_DELTA_RULE_MODULE = "fla.ops.gated_delta_rule"
TRANSFORMERS_DECODE_KERNEL = "recurrent_gated_delta_rule"
FLA_FUSED_DECODE_KERNEL = "fused_recurrent_gated_delta_rule"
# The wrapper transformers calls on decode; its `__wrapped__` is the pure-torch reference loop.
TORCH_DECODE_FALLBACK = "torch_recurrent_gated_delta_rule"

DELTANET_KERNEL_WRAPPERS: dict[str, str] = {
    "chunk_gated_delta_rule": "torch_chunk_gated_delta_rule",
    TRANSFORMERS_DECODE_KERNEL: TORCH_DECODE_FALLBACK,
    "causal_conv1d_fn": "causal_conv1d_fn",
    "causal_conv1d_update": "causal_conv1d_update",
}
"""Every kernel function the Gated DeltaNet layer dispatches through a hub-kernel wrapper.

Keyed by the name transformers looks up, valued by the wrapper attribute the modeling module
exposes; read off transformers 5.15.0's `modeling_qwen3_5`, whose top decorates exactly these four.
`bound_deltanet_kernels` asserts each attribute exists rather than skipping a missing one, so an
upstream rename surfaces as an error instead of a record that quietly names three kernels.
"""

DELTANET_KERNEL_FIELD = "deltanet_kernel"
"""The per-record field naming the DeltaNet kernel bindings a record's forwards ran under.

Its value is a mapping of kernel function -> the implementation's dotted path: the whole
`bound_deltanet_kernels()` mapping on a record that decoded, the `prefill_deltanet_kernels()` subset
on a record whose forwards never reached the decode pair. It is compared by
`assert_one_deltanet_kernel` before any pool of records is summarised. A record without the field
predates the field; a pool that mixes such records with recorded ones is refused too, since the older
half cannot prove its kernel.
"""

DELTANET_PREFILL_KERNELS: tuple[str, ...] = ("chunk_gated_delta_rule", "causal_conv1d_fn")
"""The two kernel functions a forward with no prior cache state dispatches.

Read off transformers 5.15.0's `Qwen3_5GatedDeltaNet.forward`: both `causal_conv1d_update` and
`recurrent_gated_delta_rule` sit behind ``use_precomputed_states and seq_len == 1``, where
``use_precomputed_states`` needs a cache that already holds this layer's state from an earlier call.
A bare forward -- an activation capture, a patched forward, a lens fit's forward and backward -- has no
such cache and takes the chunked pair every time, so these two are the only kernels its records can
depend on. The fla bridge re-binds neither.
"""

BRIDGE_VERIFIED_VERSIONS = {"transformers": "5.15.0", "flash-linear-attention": "0.5.2"}

# Relative tolerances on max|candidate - reference| / max|reference|, by input dtype. fp32 is tight
# because both paths accumulate in fp32 and only reassociation separates them. bf16 is loose for a
# real reason rather than slack: the torch fallback L2-normalises query and key BEFORE upcasting, so
# it rounds both to bf16 (up to 2^-9 relative each) where the kernel normalises in fp32. Two such
# roundings feeding a length-128 dot product put the expected gap in the low 1e-2, with the KERNEL
# the more accurate of the two. Set from that argument, not fitted to an observation.
DECODE_RELATIVE_TOLERANCES = {torch.float32: 1e-4, torch.bfloat16: 5e-2}


ALIASED_NATIVELY_REASON = "fla already exports the name transformers looks up"
ALIASED_BY_THIS_PROCESS_REASON = (
    "this process aliased fla's fused per-token kernel onto that name earlier"
)
"""The two reasons a call finds the name already resolving, which are not the same situation.

`bridge_decode_kernel` checks the already-exported gate before anything else, so the alias it
installs itself satisfies that gate on every later call. Reporting both cases as
:data:`ALIASED_NATIVELY_REASON` would stamp an artifact with "no bridge was needed on this box" when
one was applied, which is the opposite of what its records ran under.
"""

_aliased_by_this_process = False
"""Whether this process installed the alias, set beside the `setattr` that installs it."""


def _installed_versions() -> dict[str, str]:
    """Report the versions the bridge is running against, for a run's own provenance record."""
    return {name: importlib.metadata.version(name) for name in BRIDGE_VERIFIED_VERSIONS}


def bridge_decode_kernel() -> dict[str, object]:
    """Register fla's fused per-token kernel under the name transformers resolves, or say why not.

    Idempotent and self-disabling: the gate is whether `fla.ops.gated_delta_rule` already exposes
    `recurrent_gated_delta_rule`, so a future fla or transformers release that fixes the export name
    makes this a no-op, and calling it twice patches once. Behaviour rather than a version pin,
    because the version that fixes it is not knowable from here.

    A second call after a successful one does NOT raise. The already-exported gate is checked first
    and the alias this module installed satisfies it, so the second call returns a ``bridged: False``
    report whose reason is :data:`ALIASED_BY_THIS_PROCESS_REASON` -- distinct from
    :data:`ALIASED_NATIVELY_REASON`, because an artifact stamped with the latter claims no bridge was
    ever needed on this box. Callers that want one report for the whole process memoize the first one
    (`reward_hacking.interp.run_harness.bridge_deltanet_decode_kernel`) rather than relying on a raise.

    Raises when it cannot work rather than reporting success it did not achieve:

    *   `transformers.models.qwen3_5.modeling_qwen3_5` already imported AND the name still
        unresolved. The decorator bound its implementation at decoration time, so registering now is
        too late to reach the decode path and generation would silently stay on the torch loop.
    *   fla missing the fused kernel too, under any name we know. Then there is nothing to bridge
        and the premise of this module is stale.
    """
    global _aliased_by_this_process  # noqa: PLW0603 - one process-wide fact about one process-wide patch
    module = importlib.import_module(FLA_GATED_DELTA_RULE_MODULE)
    already_exported = getattr(module, TRANSFORMERS_DECODE_KERNEL, None)
    if already_exported is not None:
        # Checked before the too-late guard on purpose: if the name already resolves then decode
        # found the kernel whenever it was imported, and there is nothing to be late for.
        logger.info(
            "no DeltaNet decode bridge needed, %s",
            f"{FLA_GATED_DELTA_RULE_MODULE}.{TRANSFORMERS_DECODE_KERNEL} already resolves to "
            f"{already_exported.__module__}.{already_exported.__name__}",
        )
        return {
            "bridged": False,
            "reason": ALIASED_BY_THIS_PROCESS_REASON
            if _aliased_by_this_process
            else ALIASED_NATIVELY_REASON,
            "implementation": f"{already_exported.__module__}.{already_exported.__name__}",
            "installed_versions": _installed_versions(),
        }

    if QWEN3_5_MODELING_MODULE in sys.modules:
        raise RuntimeError(
            f"too late to bridge the Gated DeltaNet decode kernel: {QWEN3_5_MODELING_MODULE} is "
            f"already imported, and transformers' use_kernel_func_from_hub_with_fallback binds the "
            f"implementation at decoration time (import time of that module). Registering "
            f"{TRANSFORMERS_DECODE_KERNEL} now would leave decode on the pure-torch loop while "
            f"reporting an fla path. Call bridge_decode_kernel() before anything loads a Qwen3.5 "
            f"model, config-driven class, or the modeling module itself."
        )

    fused = getattr(module, FLA_FUSED_DECODE_KERNEL, None)
    if fused is None:
        raise RuntimeError(
            f"{FLA_GATED_DELTA_RULE_MODULE} exports neither {TRANSFORMERS_DECODE_KERNEL} nor "
            f"{FLA_FUSED_DECODE_KERNEL}, so there is no fused per-token kernel to bridge. fla "
            f"version {_installed_versions()['flash-linear-attention']}; this module was written "
            f"against {BRIDGE_VERIFIED_VERSIONS['flash-linear-attention']}."
        )

    setattr(module, TRANSFORMERS_DECODE_KERNEL, fused)
    _aliased_by_this_process = True
    logger.warning(
        "bridged the Gated DeltaNet DECODE kernel onto fla, %s",
        f"{FLA_GATED_DELTA_RULE_MODULE}.{TRANSFORMERS_DECODE_KERNEL} -> "
        f"{fused.__module__}.{fused.__name__} (transformers looks up the unprefixed name and fla "
        f"exports the prefixed one, so without this decode runs the pure-torch loop)",
    )
    return {
        "bridged": True,
        "reason": "aliased fla's fused per-token kernel onto the name transformers resolves",
        "implementation": f"{fused.__module__}.{fused.__name__}",
        "installed_versions": _installed_versions(),
    }


@dataclass(frozen=True)
class DecodeCallSite:
    """How transformers actually calls the decode kernel, read out of its source."""

    positional_count: int
    keyword_names: frozenset[str]


def decode_call_site() -> DecodeCallSite:
    """Read the decode kernel's call signature out of the transformers source, without importing it.

    The transformers wrapper filters keyword arguments down to
    `inspect.signature(implementation).parameters` and drops the rest silently, so a bridged
    implementation whose signature is missing one of these names -- `initial_state`, say -- runs
    fast and computes the wrong thing. That is the failure this exists to make impossible, and it is
    read from source rather than hardcoded so a rename upstream surfaces here.

    `importlib.util.find_spec` gives the file path without importing the module, which matters
    because importing it is exactly what makes the bridge too late to apply.
    """
    spec = importlib.util.find_spec(QWEN3_5_MODELING_MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(
            f"cannot locate the source of {QWEN3_5_MODELING_MODULE}, so the decode call site "
            f"cannot be read; is transformers installed?"
        )
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TORCH_DECODE_FALLBACK
    ]
    if len(calls) != 1:
        raise RuntimeError(
            f"expected exactly one call to {TORCH_DECODE_FALLBACK} in {QWEN3_5_MODELING_MODULE}, "
            f"found {len(calls)}. transformers has restructured the decode path, so what this "
            f"module bridges and what it checks are no longer the same thing."
        )
    call = calls[0]
    return DecodeCallSite(
        positional_count=len(call.args),
        keyword_names=frozenset(
            keyword.arg for keyword in call.keywords if keyword.arg is not None
        ),
    )


def assert_bridged_kernel_matches_call_site() -> DecodeCallSite:
    """Refuse to run when the bridged kernel would silently drop an argument transformers passes.

    Call after `bridge_decode_kernel`. Every keyword transformers passes has to be a named parameter
    of the implementation, because a `**kwargs`-only implementation would pass the wrapper's filter
    by name (`kwargs` IS a parameter name) and then receive none of them.
    """
    module = importlib.import_module(FLA_GATED_DELTA_RULE_MODULE)
    implementation = getattr(module, TRANSFORMERS_DECODE_KERNEL, None)
    if implementation is None:
        raise RuntimeError(
            f"{FLA_GATED_DELTA_RULE_MODULE}.{TRANSFORMERS_DECODE_KERNEL} is not registered; call "
            f"bridge_decode_kernel() first"
        )
    call_site = decode_call_site()
    parameters = inspect.signature(implementation).parameters
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    positional = [name for name, p in parameters.items() if p.kind in positional_kinds]
    missing = sorted(call_site.keyword_names - set(parameters))
    if missing:
        raise RuntimeError(
            f"the bridged decode kernel would silently drop {missing}: transformers filters its "
            f"keyword arguments to the implementation's parameter names, and these are not among "
            f"{sorted(parameters)}. Decode would run fast and compute something else."
        )
    if len(positional) < call_site.positional_count:
        raise RuntimeError(
            f"the bridged decode kernel takes {len(positional)} positional parameters but "
            f"transformers passes {call_site.positional_count} positionally; {positional=}"
        )
    logger.info(
        "bridged decode kernel accepts the whole call site, %s",
        f"positional={call_site.positional_count} keywords={sorted(call_site.keyword_names)}",
    )
    return call_site


def dispatched_decode_kernel() -> DecodeKernel:
    """Return the decode function transformers itself calls, wrapper and dispatch included.

    Importing the modeling module is what freezes the dispatch, so this is deliberately the last
    step: what it hands back reflects whether the bridge was applied in time.
    """
    module = importlib.import_module(QWEN3_5_MODELING_MODULE)
    return getattr(module, TORCH_DECODE_FALLBACK)


def torch_decode_reference() -> DecodeKernel:
    """Return the pure-PyTorch decode loop, reached through the wrapper's `functools.wraps` link."""
    wrapper = dispatched_decode_kernel()
    reference = getattr(wrapper, "__wrapped__", None)
    if reference is None:
        raise RuntimeError(
            f"{QWEN3_5_MODELING_MODULE}.{TORCH_DECODE_FALLBACK} carries no __wrapped__, so the "
            f"pure-torch reference implementation is unreachable and the fused kernel cannot be "
            f"compared against anything. transformers has changed how it wraps kernel functions."
        )
    return reference


def bridge_and_check_decode_kernel() -> dict[str, object]:
    """Bridge the decode kernel and refuse one that would drop a call-site argument; one report back.

    The two calls every generating entry point makes before it loads a Qwen3.5 model, in the order
    they have to happen (`bridge_decode_kernel` first, the call-site check on what it registered),
    returned as the single `deltanet_kernel_bridge` block a run's summary records. Neither imports the
    modeling module, so calling this cannot itself make the bridge too late.
    """
    report = dict(bridge_decode_kernel())
    call_site = assert_bridged_kernel_matches_call_site()
    report["decode_call_site"] = {
        "positional_count": call_site.positional_count,
        "keyword_names": sorted(call_site.keyword_names),
    }
    return report


def bound_deltanet_kernels() -> dict[str, str]:
    """Return what each Gated DeltaNet kernel function is ACTUALLY bound to, off the modeling module.

    `games.preflight.deltanet_kernel_paths` predicts what transformers WOULD resolve if it bound now;
    this reads what it DID bind. The difference is exactly the silent case: an alias registered after
    `modeling_qwen3_5` was imported changes the prediction and nothing about the wrapper transformers
    calls, which still closes over the pure-torch loop. Each wrapper `use_kernel_func_from_hub_with_
    fallback` built closes over two cells, the implementation and its parameter names, and the
    implementation is the one callable among them; anything else means transformers changed how it
    wraps and the binding cannot be read, which raises rather than guessing.

    Imports the modeling module, so it is the LAST kernel call a run makes: after the bridge, and
    after (or alongside) the model load. The value is what every record carries under
    :data:`DELTANET_KERNEL_FIELD`.
    """
    module = importlib.import_module(QWEN3_5_MODELING_MODULE)
    bound: dict[str, str] = {}
    for function_name, wrapper_name in DELTANET_KERNEL_WRAPPERS.items():
        wrapper = getattr(module, wrapper_name, None)
        if wrapper is None:
            raise RuntimeError(
                f"{QWEN3_5_MODELING_MODULE} has no {wrapper_name}, the wrapper transformers 5.15.0 "
                f"dispatches {function_name} through; the kernel binding cannot be read off a "
                f"module that has restructured its DeltaNet dispatch, and a record naming only the "
                f"kernels that still resolve would read as complete"
            )
        implementations = [
            cell.cell_contents
            for cell in getattr(wrapper, "__closure__", None) or ()
            if callable(cell.cell_contents)
        ]
        if len(implementations) != 1:
            raise RuntimeError(
                f"{QWEN3_5_MODELING_MODULE}.{wrapper_name} closes over {len(implementations)} "
                f"callables where transformers' use_kernel_func_from_hub_with_fallback closes over "
                f"exactly one (the implementation it calls); transformers has changed how it wraps "
                f"kernel functions, so which kernel {function_name} runs cannot be read here"
            )
        implementation = implementations[0]
        bound[function_name] = f"{implementation.__module__}.{implementation.__name__}"
    logger.info("Gated DeltaNet kernels bound: %s", bound)
    return bound


def prefill_deltanet_kernels(bound: Mapping[str, str]) -> dict[str, str]:
    """Narrow a full binding to the kernels a forward with no prior cache state dispatches.

    What a forward-only record carries under :data:`DELTANET_KERNEL_FIELD`, and what its resume
    identity and mixing guard compare on: the bridge re-binds only the decode kernel, so keying a
    forward-only leg on all four would refuse a relaunch whose forwards are bit-identical to the
    records it wants to continue (a 9B patching ledger is ~7 GPU-hours of them). A binding missing
    either prefill kernel is refused rather than narrowed to one, because a record naming a single
    kernel would read as complete.
    """
    missing = [name for name in DELTANET_PREFILL_KERNELS if name not in bound]
    if missing:
        raise ValueError(
            f"binding names no {missing}, so the kernels a forward dispatches cannot be read off it; "
            f"it has {sorted(bound)}"
        )
    return {name: bound[name] for name in DELTANET_PREFILL_KERNELS}


def deltanet_kernel_label(binding: object) -> str:
    """Render one record's kernel binding as a stable string, so bindings can be compared and named.

    Sorted-key JSON, so two dict values that differ only in insertion order (a record read back from
    disk against one built in memory) compare equal; a record that never carried the field renders
    as the JSON null, which is deliberately distinct from every real binding.
    """
    return json.dumps(binding, sort_keys=True)


def assert_one_deltanet_kernel(bindings: Iterable[object], *, what: str) -> object:
    """Refuse a pool of records whose forwards ran under more than one DeltaNet kernel binding.

    The mixing guard the fused-kernel bridge requires (owner decision 15, 2026-09-03): the bridged and
    the fallback decode kernels are the same recurrence in a different reduction order, and probe I1
    watched greedy tokens diverge under them from step 21 on, so a condition decoded under one kernel
    compared against a condition decoded under the other reads kernel noise as an effect. `bindings`
    are the records' :data:`DELTANET_KERNEL_FIELD` values (``None`` for a record written before the
    field existed); more than one distinct value refuses, one value is returned, and an empty pool
    returns ``None``. A pool that mixes recorded and unrecorded records refuses too, because the
    unrecorded half cannot prove what it ran under.
    """
    distinct: dict[str, object] = {}
    for binding in bindings:
        distinct.setdefault(deltanet_kernel_label(binding), binding)
    if len(distinct) > 1:
        raise ValueError(
            f"{what} pools records that ran under {len(distinct)} different Gated DeltaNet kernel "
            f"bindings: {sorted(distinct)}. The fused kernel and the torch fallback compute the same "
            f"recurrence in a different reduction order, and greedy tokens diverge under them (probe "
            f"I1, first divergence at decode step 21), so a comparison across them reads kernel noise "
            f"as an effect. Re-run the minority under the other kernel, or analyse the two separately."
        )
    return next(iter(distinct.values()), None)


class LinearAttentionShapeConfig(Protocol):
    """The four head-geometry fields a Qwen3.5 text config exposes, and nothing else.

    Structural rather than `PreTrainedConfig`, because these are the only fields the decode shape
    depends on and a test should be able to state them without a whole checkpoint config.
    """

    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int


@dataclass(frozen=True)
class DecodeShape:
    """The tensor shape one decode step presents to the Gated DeltaNet kernel.

    `num_heads` is the head count the kernel sees, which is the VALUE head count: the layer
    repeat-interleaves query and key up to it before calling (`Qwen3_5GatedDeltaNet.forward`).
    """

    batch_size: int
    num_heads: int
    key_head_dim: int
    value_head_dim: int

    @classmethod
    def from_text_config(
        cls, text_config: LinearAttentionShapeConfig, *, batch_size: int
    ) -> DecodeShape:
        """Derive the decode shape from a checkpoint's own text config, never from a constant."""
        num_key_heads = int(text_config.linear_num_key_heads)
        num_value_heads = int(text_config.linear_num_value_heads)
        if num_value_heads % num_key_heads:
            raise RuntimeError(
                f"value heads must be a multiple of key heads for the repeat-interleave the layer "
                f"does before calling the kernel; {num_key_heads=} {num_value_heads=}"
            )
        return cls(
            batch_size=batch_size,
            num_heads=num_value_heads,
            key_head_dim=int(text_config.linear_key_head_dim),
            value_head_dim=int(text_config.linear_value_head_dim),
        )


@dataclass(frozen=True)
class DecodeInputs:
    """One decode step's arguments, built the way the layer builds them."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    initial_state: torch.Tensor

    def as_kwargs(self) -> dict[str, object]:
        """Lay the inputs out the way `Qwen3_5GatedDeltaNet.forward` passes them on decode."""
        return {
            "g": self.g.clone(),
            "beta": self.beta.clone(),
            "initial_state": self.initial_state.clone(),
            "output_final_state": True,
            "use_qk_l2norm_in_kernel": True,
        }

    def positional(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fresh query/key/value, so one implementation cannot perturb the other's inputs."""
        return self.query.clone(), self.key.clone(), self.value.clone()


def build_decode_inputs(
    shape: DecodeShape, *, dtype: torch.dtype, device: torch.device | str, seed: int
) -> DecodeInputs:
    """Build decode-step inputs whose statistics match what the layer actually produces.

    `g` in particular: it is log-space decay, so it is negative, and `torch.randn` would hand the
    recurrence a growth factor above one and drown the comparison in exploding magnitudes. Built the
    way the layer builds it -- `-exp(A_log) * softplus(a + dt_bias)` with `A_log` from the
    initialisation the model uses -- so the numbers being compared are the numbers a run sees.

    Everything is drawn in fp32 from one seeded generator and cast afterwards, so the bf16 and fp32
    comparisons run on the same underlying values.
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    def normal(*sizes: int) -> torch.Tensor:
        return torch.randn(*sizes, generator=generator, device=device, dtype=torch.float32)

    batch, heads = shape.batch_size, shape.num_heads
    query = normal(batch, 1, heads, shape.key_head_dim)
    key = normal(batch, 1, heads, shape.key_head_dim)
    value = normal(batch, 1, heads, shape.value_head_dim)
    a_log = torch.log(
        torch.empty(heads, device=device, dtype=torch.float32).uniform_(0, 16, generator=generator)
    )
    dt_bias = torch.ones(heads, device=device, dtype=torch.float32)
    g = -a_log.exp() * torch.nn.functional.softplus(normal(batch, 1, heads) + dt_bias)
    beta = normal(batch, 1, heads).sigmoid()
    # The cache holds the recurrent state in fp32 in both implementations, so it stays fp32 here
    # even when the activations are bf16.
    initial_state = normal(batch, heads, shape.key_head_dim, shape.value_head_dim)
    return DecodeInputs(
        query=query.to(dtype),
        key=key.to(dtype),
        value=value.to(dtype),
        g=g.to(dtype),
        beta=beta.to(dtype),
        initial_state=initial_state,
    )


@dataclass(frozen=True)
class TensorAgreement:
    """How far apart two implementations put one returned tensor."""

    max_abs_diff: float
    mean_abs_diff: float
    reference_max_abs: float

    @property
    def max_relative_diff(self) -> float:
        """Largest absolute difference as a fraction of the reference tensor's largest magnitude."""
        if self.reference_max_abs == 0.0:
            return float("inf") if self.max_abs_diff else 0.0
        return self.max_abs_diff / self.reference_max_abs


@dataclass(frozen=True)
class DecodeAgreement:
    """The full comparison of a candidate decode kernel against the pure-torch reference."""

    dtype: str
    shape: DecodeShape
    output: TensorAgreement
    recurrent_state: TensorAgreement
    tolerance: float

    @property
    def within_tolerance(self) -> bool:
        """Whether both returned tensors agree to the tolerance set for this dtype."""
        return (
            self.output.max_relative_diff <= self.tolerance
            and self.recurrent_state.max_relative_diff <= self.tolerance
        )


def _agreement(candidate: torch.Tensor, reference: torch.Tensor) -> TensorAgreement:
    if candidate.shape != reference.shape:
        raise RuntimeError(
            f"the two decode implementations returned different shapes, {candidate.shape} against "
            f"{reference.shape}, so their outputs are not comparable"
        )
    difference = (candidate.float() - reference.float()).abs()
    return TensorAgreement(
        max_abs_diff=difference.max().item(),
        mean_abs_diff=difference.mean().item(),
        reference_max_abs=reference.float().abs().max().item(),
    )


def compare_decode_kernels(
    candidate: DecodeKernel,
    reference: DecodeKernel,
    inputs: DecodeInputs,
    *,
    tolerance: float | None = None,
) -> DecodeAgreement:
    """Run two decode implementations on identical inputs and measure how far apart they land.

    Generic over both callables so the comparison itself is testable without a GPU: hand it the
    pure-torch loop on both sides and it must report zero, hand it a deliberately wrong variant and
    it must report a real difference.
    """
    candidate_output, candidate_state = candidate(*inputs.positional(), **inputs.as_kwargs())
    reference_output, reference_state = reference(*inputs.positional(), **inputs.as_kwargs())
    if candidate_state is None or reference_state is None:
        raise RuntimeError(
            "a decode implementation returned no final recurrent state despite "
            "output_final_state=True, so the state could not be compared"
        )
    dtype = inputs.query.dtype
    if tolerance is None:
        tolerance = DECODE_RELATIVE_TOLERANCES.get(dtype)
        if tolerance is None:
            raise RuntimeError(
                f"no decode tolerance is set for {dtype}; add one to "
                f"DECODE_RELATIVE_TOLERANCES with the rounding argument that justifies it"
            )
    return DecodeAgreement(
        dtype=str(dtype),
        shape=DecodeShape(
            batch_size=inputs.query.shape[0],
            num_heads=inputs.query.shape[2],
            key_head_dim=inputs.query.shape[3],
            value_head_dim=inputs.value.shape[3],
        ),
        output=_agreement(candidate_output, reference_output),
        recurrent_state=_agreement(candidate_state, reference_state),
        tolerance=tolerance,
    )


def assert_decode_kernels_match(
    candidate: DecodeKernel,
    reference: DecodeKernel,
    inputs: DecodeInputs,
    *,
    tolerance: float | None = None,
) -> DecodeAgreement:
    """Compare two decode implementations and raise unless they agree.

    A fast kernel computing something else is worse than a slow one, and the difference is invisible
    downstream: rewards move, loss falls, and the run measures a model nobody trained.
    """
    agreement = compare_decode_kernels(candidate, reference, inputs, tolerance=tolerance)
    if not agreement.within_tolerance:
        raise RuntimeError(
            f"the fused Gated DeltaNet decode kernel disagrees with the pure-torch reference "
            f"beyond {agreement.tolerance} relative on {agreement.dtype}: output "
            f"max_abs={agreement.output.max_abs_diff:.3e} "
            f"rel={agreement.output.max_relative_diff:.3e}, recurrent state "
            f"max_abs={agreement.recurrent_state.max_abs_diff:.3e} "
            f"rel={agreement.recurrent_state.max_relative_diff:.3e}. Do NOT adopt the bridge on "
            f"these numbers -- a wrong kernel silently corrupts every rollout."
        )
    logger.info(
        "fused decode kernel matches the pure-torch reference, %s",
        f"dtype={agreement.dtype} tolerance={agreement.tolerance} "
        f"output_rel={agreement.output.max_relative_diff:.3e} "
        f"state_rel={agreement.recurrent_state.max_relative_diff:.3e}",
    )
    return agreement


def verify_bridged_decode_kernel(
    *,
    model_id: str,
    device: torch.device | str,
    batch_size: int,
    dtypes: tuple[torch.dtype, ...],
    seed: int,
) -> list[DecodeAgreement]:
    """Bridge the decode kernel, then prove the bridged path computes what the slow path did.

    Ordered the way a run has to order it: register the name, check the call site, and only then
    import the modeling module -- which is what freezes the dispatch and hands back both the fused
    path (the wrapper) and the pure-torch reference (its `__wrapped__`).
    """
    from transformers import AutoConfig  # noqa: PLC0415

    logger.info("decode kernel paths BEFORE the bridge: %s", deltanet_kernel_paths())
    logger.info("bridge: %s", bridge_decode_kernel())
    logger.info("decode kernel paths AFTER the bridge: %s", deltanet_kernel_paths())
    assert_bridged_kernel_matches_call_site()

    config = AutoConfig.from_pretrained(model_id)
    # Qwen3.5 checkpoints are vision-language configs whose text half carries the head geometry;
    # `getattr` covers a flat config too, and the cast is because those fields are dynamic.
    text_config = cast("LinearAttentionShapeConfig", getattr(config, "text_config", config))
    shape = DecodeShape.from_text_config(text_config, batch_size=batch_size)
    logger.info("decode shape from %s: %s", model_id, shape)

    fused = dispatched_decode_kernel()
    reference = torch_decode_reference()
    if fused is reference:
        raise RuntimeError(
            "the dispatched decode function IS the pure-torch reference, so the bridge did not "
            "reach the decode path and comparing them would compare torch against itself"
        )
    return [
        assert_decode_kernels_match(
            fused, reference, build_decode_inputs(shape, dtype=dtype, device=device, seed=seed)
        )
        for dtype in dtypes
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3.5-2B", help="checkpoint whose head dimensions to use"
    )
    parser.add_argument(
        "--device", default="cuda", help="fla's kernels are triton, so cuda in practice"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Register the bridge and report, with numbers, whether it changed the answer."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "fla's fused decode kernel is a triton kernel and needs CUDA; no GPU is visible. Pass "
            "--device cpu only to exercise the pure-torch path against itself."
        )
    agreements = verify_bridged_decode_kernel(
        model_id=args.model,
        device=args.device,
        batch_size=args.batch_size,
        dtypes=(torch.float32, torch.bfloat16),
        seed=args.seed,
    )
    for agreement in agreements:
        logger.info(
            "AGREEMENT %s: output max_abs=%.3e mean_abs=%.3e rel=%.3e | state max_abs=%.3e "
            "mean_abs=%.3e rel=%.3e | tolerance=%s verdict=%s",
            agreement.dtype,
            agreement.output.max_abs_diff,
            agreement.output.mean_abs_diff,
            agreement.output.max_relative_diff,
            agreement.recurrent_state.max_abs_diff,
            agreement.recurrent_state.mean_abs_diff,
            agreement.recurrent_state.max_relative_diff,
            agreement.tolerance,
            "within tolerance" if agreement.within_tolerance else "OUT OF TOLERANCE",
        )


if __name__ == "__main__":
    main()
