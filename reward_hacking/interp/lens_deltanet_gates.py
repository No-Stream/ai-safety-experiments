"""The two architecture gates on the Jacobian-lens fit: DeltaNet autograd, and the fused backward.

Three quarters of every Qwen3.5 layer stack is Gated DeltaNet linear attention, and ``jlens.fit``
is nothing but autograd through it: one forward, then a backward per batch of output dimensions.
Two things about that path are never checked by the fit itself and would each produce a
well-formed, plausible, wrong lens:

(a) **Does autograd traverse the delta-rule recurrence at all?** :func:`probe_recurrence_autograd`
    places a cotangent at one late position of a block whose interior back to a source block is all
    linear attention, and reads the gradient at a position further back than the blocks' depthwise
    causal convs can reach; only the recurrence can carry it that far. Then it SABOTAGES the same
    read by detaching every DeltaNet output on the span and requires the reading to be EXACTLY 0.0.
    Both arms run every time. A live reading beside a nonzero sabotage is a check with no teeth,
    and the report says so rather than passing.
(b) **Does the fused kernel's backward agree with the reference?** transformers binds
    ``torch_chunk_gated_delta_rule`` to flash-linear-attention's fused chunk kernel whenever ``fla``
    imports, so on such a box the fit's forward AND backward run Triton kernels that the DeltaNet
    autograd verification never exercised. :func:`audit_chunk_kernel_backward` computes the same
    rows of the fit's estimator under the bound kernel and under the pure-torch reference reached
    through the wrapper's ``__wrapped__`` -- the same link ``games.deltanet_kernels`` follows for the
    decode kernel -- and requires every row within :data:`KERNEL_AUDIT_TOLERANCE_BF16` relative L2.
    The same rows twice under the fused kernel give the run-to-run floor that makes the number
    readable, and a dispatch counter proves the switch reached every linear-attention block on
    every forward rather than assuming it did.

Everything here is a function over a block list, a forward callable and an id tensor, so the
offline tests drive both gates on a two-block toy with a linear-attention-shaped block and watch
each fail; ``reward_hacking.interp.lens_fit_gate`` wires them to a real checkpoint on a GPU.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from games.deltanet_kernels import bound_deltanet_kernels

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence
    from types import ModuleType

    from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)

CHUNK_KERNEL_WRAPPER = "torch_chunk_gated_delta_rule"
"""The modeling-module attribute the linear-attention forward calls on every prefill."""
CHUNK_KERNEL_NAME = "chunk_gated_delta_rule"
"""The name transformers looks the fused implementation up under (``bound_deltanet_kernels`` key)."""

DEFAULT_POSITION_GAP = 24
CAUSAL_LEAKAGE_MAX = 1e-6
"""Gradient at positions after the cotangent, relative to the cotangent position: bf16 residue only."""

KERNEL_AUDIT_TOLERANCE_BF16 = 5e-2
"""Per-row relative L2 agreement between the fused and pure-torch backward, in bf16.

The same bound ``games.deltanet_kernels.DECODE_RELATIVE_TOLERANCES`` sets for the decode kernel and
for the same reason: the torch reference L2-normalises query and key before upcasting where the
kernel normalises in fp32, so two bf16 roundings feed every dot product and the expected gap sits in
the low 1e-2. Set from that argument, not fitted to an observation.
"""
KERNEL_AUDIT_DIMS = 2
"""Output dimensions audited per source layer: two rows of every ``J_l`` from one backward pass.

Two, not more, because the pure-torch reference is the memory hog of the audit: its chunk loop
saves every intermediate for backward across the whole retained graph, and at eight rows on a
512-token window it filled a 24 GB card at 0.8B (observed 2026-09-04). A wrong backward is
systematic rather than row-specific, so two rows per layer over every source layer see it.
"""

LINEAR_ATTENTION = "linear_attention"

type BlockStack = Sequence[nn.Module] | nn.ModuleList
"""The residual blocks a probe hooks: jlens hands back an ``nn.ModuleList``, a toy any sequence."""


class GateFailureError(RuntimeError):
    """A gate refused, or could not run in a way that would have meant anything."""


# --------------------------------------------------------------------------------------
# Shared: hooking block outputs the way the fit does
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def recorded_block_outputs(
    blocks: BlockStack, *, record: Sequence[int], graph_root: int
) -> Generator[dict[int, torch.Tensor]]:
    """Capture the outputs of the ``record`` blocks, rooting the autograd graph at ``graph_root``.

    The root block's output is marked ``requires_grad`` before downstream blocks see it, so with
    frozen parameters it is the leaf the retained graph hangs from -- the trick ``jlens.hooks``'
    ``ActivationRecorder`` uses, kept jlens-free here so the gates' teeth are testable offline.
    """
    captured: dict[int, torch.Tensor] = {}
    handles: list[RemovableHandle] = []

    def make_hook(index: int) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: object, output: object) -> None:
            del module, inputs
            tensor = _first_tensor(output)
            if index == graph_root:
                tensor.requires_grad_()
            captured[index] = tensor

        return hook

    try:
        handles.extend(
            blocks[index].register_forward_hook(make_hook(index))
            for index in sorted({*record, graph_root})
        )
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _first_tensor(output: object) -> torch.Tensor:
    return output if torch.is_tensor(output) else cast("tuple[torch.Tensor, ...]", output)[0]


# --------------------------------------------------------------------------------------
# (a) Recurrence autograd, with the detach sabotage
# --------------------------------------------------------------------------------------


def recurrence_span(layer_types: Sequence[str], *, gap: int, conv_reach: int) -> tuple[int, int]:
    """Pick ``(source, target)``: the first block followed by a linear-attention block.

    Gradient from the target's output back to the source's output crosses positions only through
    that one Gated DeltaNet block, since MLPs, norms and the residual stream are position-local. Its
    depthwise causal conv moves information ``conv_reach`` positions, so ``gap`` has to exceed that
    for a nonzero gradient ``gap`` positions back to isolate the delta-rule recurrence. One block is
    the only span worth having: every extra interior block adds another conv's reach and takes
    nothing away, so a gap a one-block span cannot isolate, no span can.
    """
    if gap <= conv_reach:
        raise GateFailureError(
            f"a gap of {gap} positions does not exceed the conv reach of {conv_reach}; the conv "
            f"alone could carry the gradient and the probe would not isolate the recurrence"
        )
    for target in range(1, len(layer_types)):
        if layer_types[target] == LINEAR_ATTENTION:
            return target - 1, target
    raise GateFailureError(
        f"no linear-attention block past the first in {list(layer_types)}; nothing to probe"
    )


@dataclass(frozen=True)
class CrossPositionGradient:
    """The four magnitudes one cotangent placement yields, all read off ``|grad|`` at the source."""

    at_probed_earlier_position: float
    summed_over_all_earlier_positions: float
    at_strictly_later_positions: float
    at_the_cotangent_position: float


@dataclass(frozen=True)
class CotangentPlacement:
    """Where the probe puts its cotangent and where it reads: the span plus two positions."""

    source: int
    target: int
    late_position: int
    gap: int


def cross_position_gradient(
    blocks: BlockStack,
    forward: Callable[[torch.Tensor], object],
    input_ids: torch.Tensor,
    placement: CotangentPlacement,
) -> CrossPositionGradient:
    """Gradient of block ``target`` at one late position with respect to block ``source`` everywhere."""
    with (
        recorded_block_outputs(
            blocks, record=[placement.target], graph_root=placement.source
        ) as captured,
        torch.enable_grad(),
    ):
        forward(input_ids)
        target_act = captured[placement.target]
        source_act = captured[placement.source]
        cotangent = torch.zeros_like(target_act)
        cotangent[0, placement.late_position, :] = 1.0
        grad = torch.autograd.grad(outputs=target_act, inputs=[source_act], grad_outputs=cotangent)[
            0
        ].float()
    late = placement.late_position
    return CrossPositionGradient(
        at_probed_earlier_position=grad[0, late - placement.gap, :].abs().max().item(),
        summed_over_all_earlier_positions=grad[0, :late, :].abs().sum().item(),
        at_strictly_later_positions=grad[0, late + 1 :, :].abs().max().item(),
        at_the_cotangent_position=grad[0, late, :].abs().max().item(),
    )


def detach_recurrence_outputs(
    blocks: BlockStack, indices: Sequence[int], recurrence_type: type[nn.Module]
) -> list[RemovableHandle]:
    """Register hooks detaching every ``recurrence_type`` output in the given blocks: the sabotage.

    Returns the handles so the caller removes them; left installed they would corrupt every later
    read. A span with no such module is refused, since detaching nothing sabotages nothing.
    """

    def detach(module: nn.Module, inputs: object, output: object) -> object:
        del module, inputs
        if torch.is_tensor(output):
            return output.detach()
        first, *rest = cast("tuple[torch.Tensor, ...]", output)
        return (first.detach(), *rest)

    handles = [
        submodule.register_forward_hook(detach)
        for index in indices
        for submodule in blocks[index].modules()
        if isinstance(submodule, recurrence_type)
    ]
    if not handles:
        raise GateFailureError(
            f"no {recurrence_type.__name__} in blocks {list(indices)}; nothing to detach"
        )
    return handles


@dataclass(frozen=True)
class AutogradGateReport:
    """Gate (a): the live, sabotaged and recovered readings, and what each verdict rests on."""

    source_block: int
    target_block: int
    interior_blocks: tuple[int, ...]
    interior_layer_types: tuple[str, ...]
    position_gap: int
    conv_reach_of_span: int
    seq_len: int
    cotangent_position: int
    live: CrossPositionGradient
    sabotaged: CrossPositionGradient
    recovered: CrossPositionGradient

    @property
    def autograd_traverses_recurrence(self) -> bool:
        """Nonzero gradient further back than the convs reach: the recurrence carried it."""
        return self.live.at_probed_earlier_position > 0.0

    @property
    def sabotage_reads_exactly_zero(self) -> bool:
        """Detaching removes the path outright, so anything but 0.0 means the check has no teeth."""
        return self.sabotaged.summed_over_all_earlier_positions == 0.0

    @property
    def causal_leakage_ratio(self) -> float:
        """Gradient at later positions over gradient at the cotangent; bf16 leaves float residue."""
        if self.live.at_the_cotangent_position == 0.0:
            return math.inf
        return self.live.at_strictly_later_positions / self.live.at_the_cotangent_position

    @property
    def gradient_respects_causality(self) -> bool:
        """No gradient flows to positions after the cotangent, up to bf16 residue."""
        return self.causal_leakage_ratio < CAUSAL_LEAKAGE_MAX

    @property
    def sabotage_fully_reverted(self) -> bool:
        """The reading after removing the hooks equals the live one, so nothing leaks downstream."""
        return math.isclose(
            self.recovered.at_probed_earlier_position,
            self.live.at_probed_earlier_position,
            rel_tol=1e-6,
        )

    def failures(self) -> list[str]:
        """Name every verdict that did not hold."""
        failed: list[str] = []
        if not self.autograd_traverses_recurrence:
            failed.append(
                "gradient at the probed earlier position is zero: autograd did not traverse"
            )
        if not self.sabotage_reads_exactly_zero:
            failed.append(
                f"detach sabotage read {self.sabotaged.summed_over_all_earlier_positions!r}, not "
                f"0.0: the check has no teeth"
            )
        if not self.gradient_respects_causality:
            failed.append(
                f"causal leakage ratio {self.causal_leakage_ratio:.3e} >= {CAUSAL_LEAKAGE_MAX}"
            )
        if not self.sabotage_fully_reverted:
            failed.append("the reading after removing the sabotage differs from the live one")
        return failed

    @property
    def passed(self) -> bool:
        """Every verdict held."""
        return not self.failures()

    def as_payload(self) -> dict[str, object]:
        """Return the report block."""
        return {
            **asdict(self),
            "verdicts": {
                "autograd_traverses_recurrence": self.autograd_traverses_recurrence,
                "sabotage_reads_exactly_zero": self.sabotage_reads_exactly_zero,
                "causal_leakage_ratio": self.causal_leakage_ratio,
                "gradient_respects_causality": self.gradient_respects_causality,
                "sabotage_fully_reverted": self.sabotage_fully_reverted,
            },
            "failures": self.failures(),
            "passed": self.passed,
        }


def probe_recurrence_autograd(  # noqa: PLR0913 - the probe is blocks, forward, ids and the span rule
    blocks: BlockStack,
    forward: Callable[[torch.Tensor], object],
    input_ids: torch.Tensor,
    *,
    layer_types: Sequence[str],
    recurrence_type: type[nn.Module],
    gap: int,
    conv_reach: int,
) -> AutogradGateReport:
    """Gate (a): the live reading, the detach sabotage, and the reading once the sabotage is removed."""
    source, target = recurrence_span(layer_types, gap=gap, conv_reach=conv_reach)
    interior = (target,)
    seq_len = int(input_ids.shape[1])
    late_position = seq_len - 2
    if late_position - gap < 0:
        raise GateFailureError(f"a {seq_len}-token prompt is too short for a gap of {gap}")
    placement = CotangentPlacement(
        source=source, target=target, late_position=late_position, gap=gap
    )
    live = cross_position_gradient(blocks, forward, input_ids, placement)
    handles = detach_recurrence_outputs(blocks, interior, recurrence_type)
    try:
        sabotaged = cross_position_gradient(blocks, forward, input_ids, placement)
    finally:
        for handle in handles:
            handle.remove()
    recovered = cross_position_gradient(blocks, forward, input_ids, placement)
    report = AutogradGateReport(
        source_block=source,
        target_block=target,
        interior_blocks=interior,
        interior_layer_types=tuple(layer_types[index] for index in interior),
        position_gap=gap,
        conv_reach_of_span=conv_reach * len(interior),
        seq_len=seq_len,
        cotangent_position=late_position,
        live=live,
        sabotaged=sabotaged,
        recovered=recovered,
    )
    logger.info(
        "autograd gate, %s",
        f"span={source}->{target} live={live.at_probed_earlier_position:.3e} "
        f"sabotaged={sabotaged.summed_over_all_earlier_positions!r} passed={report.passed}",
    )
    return report


# --------------------------------------------------------------------------------------
# (b) Fused-versus-torch backward on the chunk kernel
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RowEstimator:
    """One ``dim_batch`` of the fit's estimator: which layers, which output dims, which positions."""

    source_layers: tuple[int, ...]
    target_layer: int
    dims: tuple[int, ...]
    skip_first: int


def jacobian_rows(
    blocks: BlockStack,
    forward: Callable[[torch.Tensor], object],
    input_ids: torch.Tensor,
    estimator: RowEstimator,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Rows ``dims`` of every ``J_l`` for one prompt, plus the target activation, in one backward.

    The fit's own estimator (``jlens.fitting.jacobian_for_prompt``) for a single ``dim_batch``: the
    prompt replicated once per dim, batch element ``b`` carrying a one-hot cotangent at output
    dimension ``dims[b]`` on every valid target position, the gradient averaged over valid source
    positions. Rows come back as ``[len(dims), d_model]`` fp32 CPU tensors per source layer, the
    target activation as ``[seq_len, d_model]`` fp32 CPU for the forward comparison.
    """
    n_dims = len(estimator.dims)
    seq_len = int(input_ids.shape[1])
    if seq_len <= estimator.skip_first + 1:
        raise GateFailureError(
            f"a {seq_len}-token prompt leaves no valid positions past "
            f"skip_first={estimator.skip_first}"
        )
    valid = torch.arange(estimator.skip_first, seq_len - 1)
    layers = sorted(estimator.source_layers)
    with (
        recorded_block_outputs(
            blocks, record=[*layers, estimator.target_layer], graph_root=layers[0]
        ) as captured,
        torch.enable_grad(),
    ):
        forward(input_ids.expand(n_dims, -1))
        target_act = captured[estimator.target_layer]
        sources = [captured[layer] for layer in layers]
        device = target_act.device
        batch = torch.arange(n_dims, device=device)
        cotangent = torch.zeros_like(target_act)
        cotangent[
            batch[:, None],
            valid.to(device)[None, :],
            torch.as_tensor(estimator.dims, device=device)[:, None],
        ] = 1.0
        grads = torch.autograd.grad(outputs=target_act, inputs=sources, grad_outputs=cotangent)
    rows = {
        layer: grad[:, valid.to(grad.device), :].float().mean(dim=1).cpu()
        for layer, grad in zip(layers, grads, strict=True)
    }
    return rows, target_act[0].detach().float().cpu()


@dataclass(frozen=True)
class RowAgreement:
    """How far apart two implementations put the rows of one layer's Jacobian."""

    layer: int
    max_relative_l2: float
    mean_relative_l2: float
    max_abs: float


def compare_jacobian_rows(
    candidate: Mapping[int, torch.Tensor], reference: Mapping[int, torch.Tensor]
) -> list[RowAgreement]:
    """Per layer, the per-row relative L2 distance ``||c - r|| / ||r||`` and its max and mean.

    Generic over both operands so the comparison is testable without a kernel: the same rows on
    both sides must read exactly zero, a perturbed copy must read the perturbation.
    """
    if set(candidate) != set(reference):
        raise GateFailureError(f"layer sets differ: {sorted(candidate)} vs {sorted(reference)}")
    agreements: list[RowAgreement] = []
    for layer in sorted(reference):
        c, r = candidate[layer].float(), reference[layer].float()
        if c.shape != r.shape:
            raise GateFailureError(f"layer {layer}: shapes {tuple(c.shape)} vs {tuple(r.shape)}")
        norms = r.norm(dim=1)
        if (norms == 0).any():
            raise GateFailureError(
                f"layer {layer}: a reference row has zero norm; nothing to compare against"
            )
        relative = (c - r).norm(dim=1) / norms
        stats = torch.stack([relative.max(), relative.mean(), (c - r).abs().max()]).tolist()
        agreements.append(
            RowAgreement(
                layer=layer, max_relative_l2=stats[0], mean_relative_l2=stats[1], max_abs=stats[2]
            )
        )
    return agreements


@contextlib.contextmanager
def counted_dispatch(
    module: ModuleType, attribute: str, implementation: Callable[..., object]
) -> Generator[list[int]]:
    """Bind ``module.attribute`` to a counting shim over ``implementation``; restore it afterwards.

    The linear-attention forward looks the chunk kernel up as a module global on every call, so
    rebinding the attribute switches every block at once; the shim counts the calls so the caller
    can prove the switch reached every block on every forward rather than assuming it. On exit the
    ORIGINAL object is put back and its identity checked, since a stale rebinding would silently
    change every later fit in the process.
    """
    original = getattr(module, attribute)
    calls = [0]

    def shim(*args: object, **kwargs: object) -> object:
        calls[0] += 1
        return implementation(*args, **kwargs)

    setattr(module, attribute, shim)
    try:
        yield calls
    finally:
        setattr(module, attribute, original)
        if getattr(module, attribute) is not original:
            raise GateFailureError(
                f"{module.__name__}.{attribute} did not restore to the original object"
            )


def chunk_kernel_arms(
    module: ModuleType,
) -> tuple[Callable[..., object], Callable[..., object], str]:
    """Return the bound chunk-kernel wrapper, the pure-torch reference behind it, and the bound path.

    The reference is the wrapper's ``__wrapped__``, the same link ``games.deltanet_kernels`` follows
    for the decode kernel. Refuses when nothing fused is bound: comparing torch against itself reads
    zero and would pass a gate that audited nothing.
    """
    wrapper = getattr(module, CHUNK_KERNEL_WRAPPER)
    reference = getattr(wrapper, "__wrapped__", None)
    if reference is None:
        raise GateFailureError(
            f"{module.__name__}.{CHUNK_KERNEL_WRAPPER} carries no __wrapped__; transformers changed "
            f"how it wraps kernel functions and the pure-torch reference is unreachable"
        )
    bound_path = bound_deltanet_kernels()[CHUNK_KERNEL_NAME]
    if bound_path == f"{module.__name__}.{CHUNK_KERNEL_WRAPPER}":
        raise GateFailureError(
            f"{CHUNK_KERNEL_NAME} is bound to the pure-torch fallback ({bound_path}); no fused "
            f"kernel is in the process, so there is nothing to audit. Install "
            f"flash-linear-attention (the lockfile pins it) or run on a box that has it; a "
            f"fused-versus-torch audit that compares torch with itself is not a gate."
        )
    return (
        cast("Callable[..., object]", wrapper),
        cast("Callable[..., object]", reference),
        bound_path,
    )


@dataclass(frozen=True)
class KernelAuditReport:
    """Gate (b): fused rows against torch rows, the fused-versus-fused floor, and the dispatch proof."""

    fused_implementation: str
    reference_implementation: str
    kernels_bound: dict[str, str]
    dims: tuple[int, ...]
    seq_len: int
    n_linear_attention_blocks: int
    dispatch_counts: dict[str, int]
    fused_vs_torch: tuple[RowAgreement, ...]
    fused_vs_fused: tuple[RowAgreement, ...]
    forward_relative_diff: float
    tolerance: float

    @property
    def max_relative_row_deviation(self) -> float:
        """The headline: the worst row anywhere, fused against torch."""
        return max(agreement.max_relative_l2 for agreement in self.fused_vs_torch)

    @property
    def floor_max_relative_row_deviation(self) -> float:
        """The same statistic between two fused runs: run-to-run reduction noise."""
        return max(agreement.max_relative_l2 for agreement in self.fused_vs_fused)

    def failures(self) -> list[str]:
        """Name every verdict that did not hold."""
        failed: list[str] = []
        expected = self.n_linear_attention_blocks
        failed.extend(
            f"{arm} arm dispatched the chunk kernel {count} times, expected {expected}"
            for arm, count in self.dispatch_counts.items()
            if count != expected
        )
        worst = [a for a in self.fused_vs_torch if a.max_relative_l2 > self.tolerance]
        if worst:
            failed.append(
                f"{len(worst)} layers have a row beyond {self.tolerance} relative L2 between the "
                f"fused and pure-torch backward; worst "
                f"{max(worst, key=lambda a: a.max_relative_l2)}"
            )
        return failed

    @property
    def passed(self) -> bool:
        """Every row within tolerance and every arm dispatched on every block."""
        return not self.failures()

    def as_payload(self) -> dict[str, object]:
        """Return the report block."""
        return {
            **asdict(self),
            "max_relative_row_deviation": self.max_relative_row_deviation,
            "floor_max_relative_row_deviation": self.floor_max_relative_row_deviation,
            "failures": self.failures(),
            "passed": self.passed,
        }


def audit_chunk_kernel_backward(  # noqa: PLR0913 - a model, a prompt, the estimator and the tolerance
    blocks: BlockStack,
    forward: Callable[[torch.Tensor], object],
    input_ids: torch.Tensor,
    *,
    modeling_module: ModuleType,
    layer_types: Sequence[str],
    estimator: RowEstimator,
    tolerance: float,
) -> KernelAuditReport:
    """Gate (b): the same Jacobian rows under the bound fused kernel and under the torch reference."""
    fused, reference, bound_path = chunk_kernel_arms(modeling_module)
    n_linear = sum(kind == LINEAR_ATTENTION for kind in layer_types)
    counts: dict[str, int] = {}
    with counted_dispatch(modeling_module, CHUNK_KERNEL_WRAPPER, fused) as calls:
        fused_rows, fused_target = jacobian_rows(blocks, forward, input_ids, estimator)
    counts["fused"] = calls[0]
    with counted_dispatch(modeling_module, CHUNK_KERNEL_WRAPPER, fused) as calls:
        fused_rows_again, _ = jacobian_rows(blocks, forward, input_ids, estimator)
    counts["fused-again"] = calls[0]
    with counted_dispatch(modeling_module, CHUNK_KERNEL_WRAPPER, reference) as calls:
        torch_rows, torch_target = jacobian_rows(blocks, forward, input_ids, estimator)
    counts["torch"] = calls[0]
    forward_diff = ((fused_target - torch_target).norm() / torch_target.norm()).item()
    report = KernelAuditReport(
        fused_implementation=bound_path,
        reference_implementation=(
            f"{getattr(reference, '__module__', '?')}.{getattr(reference, '__qualname__', '?')}"
        ),
        kernels_bound=bound_deltanet_kernels(),
        dims=estimator.dims,
        seq_len=int(input_ids.shape[1]),
        n_linear_attention_blocks=n_linear,
        dispatch_counts=counts,
        fused_vs_torch=tuple(compare_jacobian_rows(fused_rows, torch_rows)),
        fused_vs_fused=tuple(compare_jacobian_rows(fused_rows_again, fused_rows)),
        forward_relative_diff=forward_diff,
        tolerance=tolerance,
    )
    logger.info(
        "kernel audit, %s",
        f"fused={bound_path} max_row_dev={report.max_relative_row_deviation:.3e} "
        f"floor={report.floor_max_relative_row_deviation:.3e} forward={forward_diff:.3e} "
        f"dispatch={counts} passed={report.passed}",
    )
    return report
