"""Offline tests for the Jacobian-lens fit gates, each gate watched to pass AND to fail.

`jlens` runs via PYTHONPATH and the Qwen3.5 CPU route is closed, so nothing here loads a checkpoint.
The architecture gates run on a two-block toy whose blocks carry a linear-attention-shaped mixer (a
gated delta-rule recurrence behind a depthwise causal conv, pure torch, one head) and return HF-style
tuples; the schedule gates run on stubs. Every gate is driven to refuse on a construction that has
exactly the defect it exists to catch: a recurrence autograd never traverses, a block with a second
cross-position path the detach cannot remove, a "fused" kernel that drops the decay, a sweep with no
fitting candidate, a tokenizer family that never differs, a fitter that ignores its checkpoint.
"""

from __future__ import annotations

import json
import subprocess
import types
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch import nn

from reward_hacking.interp import lens_deltanet_gates, lens_fit_gate
from reward_hacking.interp.jacobian import JacobianConfig, fit_lens
from reward_hacking.interp.lens_deltanet_gates import (
    CHUNK_KERNEL_WRAPPER,
    CotangentPlacement,
    GateFailureError,
    KernelAuditReport,
    RowAgreement,
    RowEstimator,
    audit_chunk_kernel_backward,
    chunk_kernel_arms,
    compare_jacobian_rows,
    counted_dispatch,
    cross_position_gradient,
    jacobian_rows,
    probe_recurrence_autograd,
    recurrence_span,
)
from reward_hacking.interp.lens_fit_gate import (
    FALLBACK_PROMPT,
    GateRun,
    gate_prompts,
    jlens_provenance,
)
from reward_hacking.interp.lens_schedule_gates import (
    DimBatchChoice,
    DimBatchSweepReport,
    DimBatchTrial,
    choose_dim_batch,
    lens_relative_diffs,
    resume_equality,
    token_identity,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

D_MODEL = 8
VOCAB = 32
SEQ_LEN = 24
GAP = 8
CONV_KERNEL = 4

FAKE_MODELING = types.ModuleType("fake_modeling")
"""Stands in for transformers' modeling module: the toy mixer looks its kernel up here by name."""


# --------------------------------------------------------------------------------------
# The toy: a linear-attention-shaped block stack that dispatches its recurrence through a module global
# --------------------------------------------------------------------------------------


def reference_delta_rule(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """A one-head gated delta rule in pure torch: ``S_t = exp(g_t) S_{t-1} + k_t (beta_t (v_t - S k_t))``."""
    batch, seq_len, d = q.shape
    state = q.new_zeros(batch, d, d)
    outputs: list[torch.Tensor] = []
    for t in range(seq_len):
        state = state * g[:, t].exp()[:, None, None]
        kv_mem = torch.einsum("bij,bi->bj", state, k[:, t])
        delta = (v[:, t] - kv_mem) * beta[:, t][:, None]
        state = state + k[:, t][:, :, None] * delta[:, None, :]
        outputs.append(torch.einsum("bij,bi->bj", state, q[:, t]))
    return torch.stack(outputs, dim=1)


def no_decay_delta_rule(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """The same recurrence with the decay dropped: a "fused kernel" that computes something else."""
    return reference_delta_rule(q, k, v, torch.zeros_like(g), beta)


class ToyRecurrence(nn.Module):
    """A Gated-DeltaNet-shaped mixer: depthwise causal conv (kernel 4), then a delta-rule scan.

    The scan is called through ``FAKE_MODELING.torch_chunk_gated_delta_rule``, looked up at call
    time exactly as the real layer looks up its kernel, so the dispatch switch under test is the same.
    """

    def __init__(self, d: int, *, detach_state: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d, d, CONV_KERNEL, groups=d, padding=CONV_KERNEL - 1, bias=False)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.beta = nn.Linear(d, 1)
        self.decay = nn.Linear(d, 1)
        self.out = nn.Linear(d, d, bias=False)
        self.detach_state = detach_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = self.conv(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        q, k, v = self.q(x), self.k(x), self.v(x)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        beta = torch.sigmoid(self.beta(x)).squeeze(-1)
        g = -nn.functional.softplus(self.decay(x)).squeeze(-1)
        kernel = cast("Callable[..., torch.Tensor]", getattr(FAKE_MODELING, CHUNK_KERNEL_WRAPPER))
        if self.detach_state:
            q, k, v, g, beta = (t.detach() for t in (q, k, v, g, beta))
        return self.out(kernel(q, k, v, g, beta))


class CausalMeanLeak(nn.Module):
    """A second cross-position path that is NOT a recurrence: the defect the sabotage must expose."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.cumsum(dim=1) / torch.arange(1, x.shape[1] + 1, device=x.device)[None, :, None]


class ToyBlock(nn.Module):
    """One pre-norm residual block returning an HF-style tuple."""

    def __init__(self, d: int, *, detach_state: bool = False, leak: bool = False) -> None:
        super().__init__()
        self.norm_mixer = nn.LayerNorm(d)
        self.mixer = ToyRecurrence(d, detach_state=detach_state)
        self.leak = CausalMeanLeak() if leak else None
        self.norm_mlp = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        hidden = hidden + self.mixer(self.norm_mixer(hidden))
        if self.leak is not None:
            hidden = hidden + self.leak(hidden)
        hidden = hidden + self.mlp(self.norm_mlp(hidden))
        return (hidden,)


class ToyStack(nn.Module):
    """Embedding plus blocks; parameters frozen, as ``jlens.from_hf`` leaves a model."""

    def __init__(
        self, n_blocks: int = 2, *, detach_state: bool = False, leak: bool = False
    ) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.layers = nn.ModuleList(
            [ToyBlock(D_MODEL, detach_state=detach_state, leak=leak) for _ in range(n_blocks)]
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(ids)
        for block in self.layers:
            hidden = block(hidden)[0]
        return hidden


LAYER_TYPES = ("linear_attention", "linear_attention")


def input_ids(seq_len: int = SEQ_LEN) -> torch.Tensor:
    return torch.arange(seq_len)[None, :] % VOCAB


@pytest.fixture(autouse=True)
def bind_reference_kernel() -> None:
    """Every test starts with the reference scan bound under the real wrapper's name."""
    setattr(FAKE_MODELING, CHUNK_KERNEL_WRAPPER, reference_delta_rule)


# --------------------------------------------------------------------------------------
# (a) recurrence autograd
# --------------------------------------------------------------------------------------


class TestRecurrenceSpan:
    def test_targets_the_first_linear_attention_block_after_a_source(self) -> None:
        assert recurrence_span(
            ["linear_attention"] * 3 + ["full_attention"], gap=8, conv_reach=3
        ) == (0, 1)
        assert recurrence_span(
            ["full_attention", "full_attention", "linear_attention"], gap=8, conv_reach=3
        ) == (1, 2)

    def test_refuses_a_gap_the_conv_alone_could_bridge(self) -> None:
        with pytest.raises(GateFailureError, match="does not exceed the conv reach"):
            recurrence_span(["linear_attention"] * 4, gap=3, conv_reach=3)

    def test_refuses_a_stack_with_no_linear_attention_block_to_probe(self) -> None:
        with pytest.raises(GateFailureError, match="no linear-attention block"):
            recurrence_span(
                ["linear_attention", "full_attention", "full_attention"], gap=8, conv_reach=3
            )


class TestProbeRecurrenceAutograd:
    def probe(self, stack: ToyStack) -> lens_deltanet_gates.AutogradGateReport:
        return probe_recurrence_autograd(
            stack.layers,
            stack.forward,
            input_ids(),
            layer_types=LAYER_TYPES,
            recurrence_type=ToyRecurrence,
            gap=GAP,
            conv_reach=CONV_KERNEL - 1,
        )

    def test_a_working_recurrence_passes_with_the_sabotage_reading_exactly_zero(self) -> None:
        report = self.probe(ToyStack())
        assert report.passed, report.failures()
        assert report.live.at_probed_earlier_position > 0.0
        assert report.sabotaged.summed_over_all_earlier_positions == 0.0
        assert report.recovered == report.live
        assert report.conv_reach_of_span == CONV_KERNEL - 1 < GAP
        verdicts = cast("dict[str, object]", report.as_payload()["verdicts"])
        assert verdicts["sabotage_reads_exactly_zero"] is True

    def test_a_recurrence_autograd_never_traverses_is_refused(self) -> None:
        report = self.probe(ToyStack(detach_state=True))
        assert not report.passed
        assert any("did not traverse" in failure for failure in report.failures())

    def test_a_second_cross_position_path_makes_the_sabotage_nonzero_and_is_refused(self) -> None:
        report = self.probe(ToyStack(leak=True))
        assert report.sabotaged.summed_over_all_earlier_positions > 0.0
        assert any("no teeth" in failure for failure in report.failures())

    def test_a_prompt_too_short_for_the_gap_is_refused(self) -> None:
        stack = ToyStack()
        with pytest.raises(GateFailureError, match="too short"):
            probe_recurrence_autograd(
                stack.layers,
                stack.forward,
                input_ids(GAP),
                layer_types=LAYER_TYPES,
                recurrence_type=ToyRecurrence,
                gap=GAP,
                conv_reach=CONV_KERNEL - 1,
            )

    def test_the_sabotage_hooks_are_removed_even_when_the_probe_raises(self) -> None:
        stack = ToyStack()
        placement = CotangentPlacement(source=0, target=1, late_position=SEQ_LEN - 2, gap=GAP)
        before = cross_position_gradient(stack.layers, stack.forward, input_ids(), placement)
        with pytest.raises(GateFailureError, match="nothing to detach"):
            lens_deltanet_gates.detach_recurrence_outputs(stack.layers, [1], CausalMeanLeak)
        after = cross_position_gradient(stack.layers, stack.forward, input_ids(), placement)
        assert after == before


# --------------------------------------------------------------------------------------
# (b) fused-versus-torch kernel audit
# --------------------------------------------------------------------------------------


def estimator() -> RowEstimator:
    return RowEstimator(source_layers=(0,), target_layer=1, dims=(0, 3, 5), skip_first=4)


class TestJacobianRows:
    def test_rows_have_one_row_per_dim_and_the_target_activation_comes_back(self) -> None:
        stack = ToyStack()
        rows, target = jacobian_rows(stack.layers, stack.forward, input_ids(), estimator())
        assert set(rows) == {0}
        assert rows[0].shape == (3, D_MODEL)
        assert target.shape == (SEQ_LEN, D_MODEL)
        assert torch.isfinite(rows[0]).all()
        assert rows[0].abs().sum() > 0

    def test_refuses_a_prompt_with_no_valid_positions(self) -> None:
        stack = ToyStack()
        with pytest.raises(GateFailureError, match="no valid positions"):
            jacobian_rows(stack.layers, stack.forward, input_ids(5), estimator())


class TestCompareJacobianRows:
    def test_identical_rows_read_exactly_zero(self) -> None:
        rows = {0: torch.randn(3, D_MODEL), 1: torch.randn(3, D_MODEL)}
        for agreement in compare_jacobian_rows(rows, rows):
            assert agreement.max_relative_l2 == 0.0
            assert agreement.max_abs == 0.0

    def test_a_perturbed_row_reads_its_perturbation(self) -> None:
        reference = {0: torch.ones(2, D_MODEL)}
        candidate = {0: torch.ones(2, D_MODEL)}
        candidate[0][1] *= 1.1
        [agreement] = compare_jacobian_rows(candidate, reference)
        assert agreement.max_relative_l2 == pytest.approx(0.1, rel=1e-5)
        assert agreement.mean_relative_l2 == pytest.approx(0.05, rel=1e-5)

    def test_refuses_mismatched_layers_and_zero_reference_rows(self) -> None:
        with pytest.raises(GateFailureError, match="layer sets differ"):
            compare_jacobian_rows({0: torch.ones(1, 2)}, {1: torch.ones(1, 2)})
        with pytest.raises(GateFailureError, match="zero norm"):
            compare_jacobian_rows({0: torch.ones(1, 2)}, {0: torch.zeros(1, 2)})


class TestCountedDispatch:
    def test_counts_calls_delegates_and_restores_the_original(self) -> None:
        module = types.ModuleType("m")
        original = lambda x: x + 1  # noqa: E731 - the object identity is the point
        setattr(module, "kernel", original)  # noqa: B010 - a stand-in module namespace
        with counted_dispatch(module, "kernel", lambda x: x * 10) as calls:
            assert module.kernel(2) == 20
            assert module.kernel(3) == 30
        assert calls[0] == 2
        assert module.kernel is original


class TestChunkKernelArms:
    def test_refuses_a_wrapper_with_no_reference_behind_it(self) -> None:
        module = types.ModuleType("bare")
        setattr(module, CHUNK_KERNEL_WRAPPER, reference_delta_rule)
        with pytest.raises(GateFailureError, match="__wrapped__"):
            chunk_kernel_arms(module)

    def test_refuses_when_only_the_torch_fallback_is_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = types.ModuleType("fallback_only")

        def wrapper(*args: object, **kwargs: object) -> object:
            return reference_delta_rule(*args, **kwargs)  # pyright: ignore[reportArgumentType]

        wrapper.__wrapped__ = reference_delta_rule  # pyright: ignore[reportFunctionMemberAccess]
        setattr(module, CHUNK_KERNEL_WRAPPER, wrapper)
        monkeypatch.setattr(
            lens_deltanet_gates,
            "bound_deltanet_kernels",
            lambda: {"chunk_gated_delta_rule": f"fallback_only.{CHUNK_KERNEL_WRAPPER}"},
        )
        with pytest.raises(GateFailureError, match="nothing to audit"):
            chunk_kernel_arms(module)


def bind_fused(monkeypatch: pytest.MonkeyPatch, fused: Callable[..., torch.Tensor]) -> None:
    """Bind ``fused`` under the wrapper's name with the reference scan as its ``__wrapped__``."""

    def wrapper(*args: object, **kwargs: object) -> torch.Tensor:
        return fused(*args, **kwargs)

    wrapper.__wrapped__ = reference_delta_rule  # pyright: ignore[reportFunctionMemberAccess]
    monkeypatch.setattr(FAKE_MODELING, CHUNK_KERNEL_WRAPPER, wrapper)
    monkeypatch.setattr(
        lens_deltanet_gates,
        "bound_deltanet_kernels",
        lambda: {"chunk_gated_delta_rule": "fake_fla.chunk_gated_delta_rule"},
    )


class TestAuditChunkKernelBackward:
    def audit(self, stack: ToyStack) -> KernelAuditReport:
        return audit_chunk_kernel_backward(
            stack.layers,
            stack.forward,
            input_ids(),
            modeling_module=FAKE_MODELING,
            layer_types=LAYER_TYPES,
            estimator=estimator(),
            tolerance=5e-2,
        )

    def test_a_faithful_fused_kernel_passes_and_every_arm_dispatched_on_every_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bind_fused(
            monkeypatch, lambda q, k, v, g, beta: reference_delta_rule(q, k, v, g, beta).clone()
        )
        report = self.audit(ToyStack())
        assert report.passed, report.failures()
        assert report.dispatch_counts == {"fused": 2, "fused-again": 2, "torch": 2}
        assert report.max_relative_row_deviation == 0.0
        assert report.floor_max_relative_row_deviation == 0.0
        assert report.forward_relative_diff == 0.0
        assert report.fused_implementation == "fake_fla.chunk_gated_delta_rule"

    def test_a_fused_kernel_that_drops_the_decay_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bind_fused(monkeypatch, no_decay_delta_rule)
        report = self.audit(ToyStack())
        assert not report.passed
        assert report.max_relative_row_deviation > 5e-2
        assert any("beyond 0.05" in failure for failure in report.failures())

    def test_the_wrapper_is_restored_after_the_audit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bind_fused(monkeypatch, no_decay_delta_rule)
        wrapper = getattr(FAKE_MODELING, CHUNK_KERNEL_WRAPPER)
        self.audit(ToyStack())
        assert getattr(FAKE_MODELING, CHUNK_KERNEL_WRAPPER) is wrapper

    def test_a_dispatch_count_short_of_the_block_count_is_refused(self) -> None:
        report = KernelAuditReport(
            fused_implementation="f",
            reference_implementation="r",
            kernels_bound={},
            dims=(0,),
            seq_len=SEQ_LEN,
            n_linear_attention_blocks=2,
            dispatch_counts={"fused": 1, "fused-again": 2, "torch": 2},
            fused_vs_torch=(RowAgreement(0, 0.0, 0.0, 0.0),),
            fused_vs_fused=(RowAgreement(0, 0.0, 0.0, 0.0),),
            forward_relative_diff=0.0,
            tolerance=5e-2,
        )
        assert report.failures() == ["fused arm dispatched the chunk kernel 1 times, expected 2"]


# --------------------------------------------------------------------------------------
# (c) dim_batch choice
# --------------------------------------------------------------------------------------


def trial(
    dim_batch: int, *, reserved_fraction: float | None, total: int = 1000, seq_len: int = 512
) -> DimBatchTrial:
    if reserved_fraction is None:
        return DimBatchTrial(dim_batch, False, None, None, None, None, None, "OutOfMemoryError")
    return DimBatchTrial(
        dim_batch,
        True,
        1.0,
        seq_len,
        495,
        int(total * reserved_fraction * 0.9),
        int(total * reserved_fraction),
        None,
    )


class TestChooseDimBatch:
    def test_takes_the_largest_that_fits_with_headroom(self) -> None:
        trials = [
            trial(16, reserved_fraction=None),
            trial(8, reserved_fraction=0.85),
            trial(4, reserved_fraction=0.5),
        ]
        choice = choose_dim_batch(trials, total_bytes=1000, headroom_fraction=0.10)
        assert choice.chosen == 8
        assert set(choice.rejected) == {16}

    def test_a_completed_trial_inside_the_headroom_is_rejected(self) -> None:
        trials = [trial(8, reserved_fraction=0.95), trial(4, reserved_fraction=0.5)]
        choice = choose_dim_batch(trials, total_bytes=1000, headroom_fraction=0.10)
        assert choice.chosen == 4
        assert "exceeds the budget 900" in choice.rejected[8]

    def test_no_fitting_candidate_fails_the_gate(self) -> None:
        trials = [trial(16, reserved_fraction=None), trial(8, reserved_fraction=0.99)]
        report = DimBatchSweepReport(
            512,
            (16, 8),
            tuple(trials),
            choose_dim_batch(trials, total_bytes=1000, headroom_fraction=0.10),
        )
        assert not report.passed
        assert "no dim_batch candidate" in report.failures()[0]

    def test_a_trial_at_a_short_window_fails_the_gate(self) -> None:
        trials = [trial(4, reserved_fraction=0.5, seq_len=300)]
        report = DimBatchSweepReport(
            512,
            (4,),
            tuple(trials),
            choose_dim_batch(trials, total_bytes=1000, headroom_fraction=0.10),
        )
        assert report.choice.chosen == 4
        assert any("shorter than max_seq_len=512" in failure for failure in report.failures())


# --------------------------------------------------------------------------------------
# (d) token identity
# --------------------------------------------------------------------------------------


def words(text: str) -> list[int]:
    return [hash(word) & 0xFFFF for word in text.split()]


def chars(text: str) -> list[int]:
    return [ord(char) for char in text]


TEXTS = ("def f(x):\n    return x", "assert candidate(3) == 7")
IDENTITIES = {"ref": "hf:aaa", "twin": "hf:bbb", "other": "hf:ccc"}


class TestTokenIdentity:
    texts = TEXTS
    identities = IDENTITIES

    def test_passes_when_twins_agree_and_the_other_family_differs(self) -> None:
        report = token_identity(
            self.texts,
            {"ref": words, "twin": words, "other": chars},
            identities=self.identities,
            reference="ref",
            expected_identical=["twin"],
            expected_different=["other"],
        )
        assert report.passed, report.failures()
        by_label = {c.label: c for c in report.comparisons}
        assert by_label["twin"].n_identical == 2
        assert by_label["other"].n_different == 2
        assert by_label["other"].first_divergence is not None

    def test_a_twin_that_differs_on_one_text_is_refused_with_the_divergence(self) -> None:
        def almost(text: str) -> list[int]:
            return chars(text) if "assert" in text else words(text)

        report = token_identity(
            self.texts,
            {"ref": words, "twin": almost, "other": chars},
            identities=self.identities,
            reference="ref",
            expected_identical=["twin"],
            expected_different=["other"],
        )
        assert not report.passed
        [failure] = report.failures()
        assert "twin tokenizes 1 of 2 texts differently" in failure
        assert "'text_index': 1" in failure

    def test_an_other_family_that_never_differs_means_no_teeth_and_is_refused(self) -> None:
        report = token_identity(
            self.texts,
            {"ref": words, "twin": words, "other": words},
            identities=self.identities,
            reference="ref",
            expected_identical=["twin"],
            expected_different=["other"],
        )
        assert [f for f in report.failures() if "not been watched to fail" in f]

    def test_no_texts_is_refused(self) -> None:
        with pytest.raises(GateFailureError, match="no texts"):
            token_identity(
                [],
                {"ref": words},
                identities={"ref": "x"},
                reference="ref",
                expected_identical=[],
                expected_different=[],
            )


# --------------------------------------------------------------------------------------
# (e) resume equality
# --------------------------------------------------------------------------------------


class _StubLens:
    def __init__(self, jacobians: dict[int, torch.Tensor], n_prompts: int) -> None:
        self.jacobians = jacobians
        self.n_prompts = n_prompts


def prompt_jacobian(prompt: str, layer: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(hash((prompt, layer)) & 0xFFFFFFFF)
    return torch.randn(4, 4, generator=generator)


class _CheckpointingJlens:
    """A fitter with jlens's accumulator semantics: a running fp32 sum, checkpointed and resumed.

    ``honour_checkpoint=False`` is the sabotage: it forgets the checkpoint and fits only the prompts
    it is handed after the split, which a resume-equality check has to see.
    """

    layers = (0, 1)

    def __init__(self, *, honour_checkpoint: bool = True) -> None:
        self.calls: list[dict[str, object]] = []
        self.honour_checkpoint = honour_checkpoint

    def fit(self, model: object, prompts: Sequence[str], **kwargs: object) -> _StubLens:
        del model
        self.calls.append({"prompts": list(prompts), **kwargs})
        checkpoint_path = cast("str | None", kwargs.get("checkpoint_path"))
        resume = bool(kwargs.get("resume"))
        sums = {layer: torch.zeros(4, 4) for layer in self.layers}
        done, next_idx = 0, 0
        if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, weights_only=True)
            if self.honour_checkpoint:
                sums, done, next_idx = state["sums"], state["done"], state["next_idx"]
            else:
                next_idx = state["next_idx"]
        for index, prompt in enumerate(prompts):
            if index < next_idx:
                continue
            for layer in self.layers:
                sums[layer] += prompt_jacobian(prompt, layer)
            done += 1
        if checkpoint_path is not None:
            torch.save({"sums": sums, "done": done, "next_idx": len(prompts)}, checkpoint_path)
        return _StubLens({layer: sums[layer] / done for layer in self.layers}, done)


PROMPTS = [f"prompt {index}" for index in range(10)]


class TestResumeEquality:
    def run(self, jl: _CheckpointingJlens, tmp_path: Path) -> Any:
        return resume_equality(
            cast("Any", jl),
            object(),
            PROMPTS,
            JacobianConfig(dim_batch=4, max_seq_len=64),
            work_dir=tmp_path,
            tolerance=1e-6,
        )

    def test_a_faithful_resume_equals_the_straight_fit_and_the_half_fit_does_not(
        self, tmp_path: Path
    ) -> None:
        jl = _CheckpointingJlens()
        report = self.run(jl, tmp_path)
        assert report.passed, report.failures()
        assert report.max_relative_diff <= 1e-6
        assert report.half_min_relative_diff > 1e-6
        assert (report.n_prompts_straight, report.n_prompts_half, report.n_prompts_resumed) == (
            10,
            5,
            10,
        )

    def test_the_three_fits_carry_the_threaded_knobs(self, tmp_path: Path) -> None:
        jl = _CheckpointingJlens()
        self.run(jl, tmp_path)
        straight, first, resumed = jl.calls
        assert straight["prompts"] == PROMPTS
        assert straight["checkpoint_path"] is None
        assert first["prompts"] == PROMPTS[:5]
        assert first["checkpoint_every"] == 5
        assert first["resume"] is False
        assert resumed["prompts"] == PROMPTS
        assert resumed["resume"] is True
        assert (
            resumed["checkpoint_path"]
            == first["checkpoint_path"]
            == str(tmp_path / "resume_gate_checkpoint.pt")
        )
        assert all(call["dim_batch"] == 4 and call["max_seq_len"] == 64 for call in jl.calls)

    def test_a_fitter_that_forgets_its_checkpoint_is_refused(self, tmp_path: Path) -> None:
        report = self.run(_CheckpointingJlens(honour_checkpoint=False), tmp_path)
        assert not report.passed
        assert any("prompt counts" in failure for failure in report.failures())
        assert any("differs from the straight fit" in failure for failure in report.failures())

    def test_fewer_than_two_prompts_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(GateFailureError, match="at least 2"):
            resume_equality(
                cast("Any", _CheckpointingJlens()),
                object(),
                ["one"],
                JacobianConfig(),
                work_dir=tmp_path,
                tolerance=1e-6,
            )


class TestLensRelativeDiffs:
    def test_identical_is_zero_and_layers_must_match(self) -> None:
        a = {0: torch.ones(2, 2), 1: torch.eye(2)}
        assert lens_relative_diffs(a, a) == {0: 0.0, 1: 0.0}
        with pytest.raises(GateFailureError, match="different layers"):
            lens_relative_diffs(a, {0: torch.ones(2, 2)})


# --------------------------------------------------------------------------------------
# The threaded knobs, and the CLI's bookkeeping
# --------------------------------------------------------------------------------------


class TestThreadedFitKnobs:
    def test_config_rejects_a_non_positive_checkpoint_interval_but_allows_none(self) -> None:
        with pytest.raises(ValueError, match="checkpoint_every"):
            JacobianConfig(checkpoint_every=0)
        assert JacobianConfig(checkpoint_every=None).checkpoint_every is None

    def test_fit_lens_hands_every_restart_knob_to_jlens(self, tmp_path: Path) -> None:
        jl = _CheckpointingJlens()
        config = JacobianConfig(
            dim_batch=2,
            max_seq_len=32,
            checkpoint_path=tmp_path / "c.pt",
            checkpoint_every=3,
            resume=False,
        )
        fit_lens(config, object(), PROMPTS[:2], cast("Any", jl))
        [call] = jl.calls
        assert call["checkpoint_path"] == str(tmp_path / "c.pt")
        assert call["checkpoint_every"] == 3
        assert call["resume"] is False


class TestGateRun:
    def test_records_each_gate_and_tracks_failures_in_the_written_report(
        self, tmp_path: Path
    ) -> None:
        run = GateRun(report={"gates": {}, "passed": True}, out_path=tmp_path / "r" / "report.json")
        run.record("autograd", {"passed": True})
        assert json.loads(run.out_path.read_text())["passed"] is True
        run.record("kernel", {"passed": False, "failures": ["x"]})
        written = json.loads(run.out_path.read_text())
        assert written["passed"] is False
        assert written["failed_gates"] == ["kernel"]
        assert set(written["gates"]) == {"autograd", "kernel"}


class TestGatePrompts:
    def test_the_fallback_prose_is_sized_past_the_window(self) -> None:
        [prompt] = gate_prompts(None, fallback_tokens=512)
        assert len(prompt) > 512 * 4
        assert prompt.startswith(FALLBACK_PROMPT)


class TestJlensProvenance:
    def test_a_clone_at_another_commit_is_refused(self, tmp_path: Path) -> None:
        clone = tmp_path / "clone"
        (clone / "jlens").mkdir(parents=True)
        (clone / "jlens" / "__init__.py").write_text("")
        git = ["git", "-C", str(clone)]
        subprocess.run([*git, "init", "-q"], check=True)  # noqa: S603 - fixed argv
        subprocess.run(  # noqa: S603 - fixed argv
            [
                *git,
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "x",
            ],
            check=True,
        )
        fake = types.ModuleType("jlens")
        fake.__file__ = str(clone / "jlens" / "__init__.py")
        with pytest.raises(GateFailureError, match="not the pinned"):
            jlens_provenance(fake)

    def test_a_non_git_directory_records_no_commit(self, tmp_path: Path) -> None:
        (tmp_path / "jlens").mkdir()
        (tmp_path / "jlens" / "__init__.py").write_text("")
        fake = types.ModuleType("jlens")
        fake.__file__ = str(tmp_path / "jlens" / "__init__.py")
        assert jlens_provenance(fake)["commit"] is None


def test_the_cli_module_exposes_the_five_gates() -> None:
    assert lens_fit_gate.ALL_GATES == ("autograd", "kernel", "sweep", "tokens", "resume")
    assert replace(DimBatchChoice(None, 0, 0, 0.1, {}), chosen=8).chosen == 8
