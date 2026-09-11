"""Offline tests for the causal residual interventions: steering, ablation, activation patching.

CPU only, no model load. The pure transforms are checked against ground truth (a steer moves the
residual by exactly ``alpha * unit(dir)``; an ablation zeroes the projection and preserves the
orthogonal complement). The hook math is checked directly and on a real ``nn.Module``. Activation
patching is checked end to end on a tiny two-layer causal fake: an identity layer whose position-1
output is patched, then a cumulative-sum layer that carries that patched position into the readout,
so injecting the clean activation makes the corrupted-run readout logits equal the clean ones --
recovery ~1.0 -- which is the "the position is causally used" signal the real experiment reads.
"""
# Duck-typed fakes stand in for real HF models/tokenizers, so reportArgumentType fires at the call
# sites; the output-modifying hooks are typed to return `object` (the layer output at the hook
# boundary), so indexing a hook's return in an assertion trips reportIndexIssue. Both are the fakes,
# not real defects -- scope the suppression to those two rules for this offline test file.
# pyright: reportArgumentType=false, reportIndexIssue=false

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from reward_hacking.interp.directions import capture_positionwise_activations, cosine, unit
from reward_hacking.interp.generation_capture import (
    PENALTY_FREE_THINKING_SAMPLING,
    generate_response,
)
from reward_hacking.interp.steering import (
    PATCH_WINDOW_POST_DIVERGENCE,
    PATCH_WINDOW_POST_DIVERGENCE_EXCL_READOUT,
    PATCH_WINDOW_READOUT_ONLY,
    PATCH_WINDOW_SHARED_PREFIX,
    READOUT_MODE_ACTION_LOGPROB,
    TAIL_WINDOW_PREFIX,
    GapReadout,
    PatchBaseline,
    TwinPatchPlan,
    _readout_logits,  # pyright: ignore[reportPrivateUsage]  # the vocab-projection narrowing
    ablate_residual,
    ablation_hook,
    activation_patch_hook,
    axis_complement_replacement,
    axis_component_replacement,
    capture_patch_prefix,
    choose_answer_token,
    gap_readout,
    gap_recovery,
    logit_recovery,
    plan_twin_patch_ladder,
    random_axis_replacement,
    read_gap,
    recovery_metrics,
    replay_from_layer,
    residual_intervention,
    run_activation_patch,
    run_steered_generation,
    steer_residual,
    steering_directions,
    steering_hook,
    window_contains_readout,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reward_hacking.model_backend import SamplingConfig

_GREEDY = replace(PENALTY_FREE_THINKING_SAMPLING, do_sample=False)
"""The module's own default sampler, made reproducible for fakes that ignore it anyway."""


def _reading(metrics: Mapping[str, float | None], name: str) -> float:
    """Pull one recovery reading, failing loudly if it came back ``None``.

    ``recovery_metrics`` returns optional values because a zero denominator has no recovery to
    report. Every reading a test compares numerically is one whose denominator the test itself set
    non-zero, so a ``None`` here means the function stopped computing it rather than that the case is
    degenerate -- which is a failure worth seeing rather than a comparison to skip.
    """
    value = metrics[name]
    assert value is not None, f"{name} came back None on a non-degenerate case"
    return value


class TestSteerResidual:
    def test_moves_by_exactly_alpha_unit_direction(self) -> None:
        hidden = torch.randn(2, 3, 8)
        direction = torch.randn(8) * 5.0
        alpha = 2.5

        steered = steer_residual(hidden, direction, alpha)

        delta = steered - hidden
        expected = alpha * unit(direction)
        # Every position shifted by the same vector of magnitude alpha.
        assert torch.allclose(delta, expected.expand_as(delta), atol=1e-5)
        assert delta[0, 0].norm().item() == pytest.approx(alpha, abs=1e-4)

    def test_scale_of_direction_does_not_change_the_steer(self) -> None:
        hidden = torch.randn(4, 8)
        direction = torch.randn(8)
        small = steer_residual(hidden, direction, 1.0)
        big = steer_residual(hidden, direction * 100.0, 1.0)
        assert torch.allclose(small, big, atol=1e-5)


class TestAblateResidual:
    def test_zeroes_the_projection_and_keeps_the_complement(self) -> None:
        torch.manual_seed(0)
        direction = torch.randn(16)
        u = unit(direction)
        hidden = torch.randn(5, 16)

        ablated = ablate_residual(hidden, direction)

        # No component left along the direction.
        assert torch.allclose(ablated @ u, torch.zeros(5), atol=1e-5)
        # The orthogonal complement is untouched: ablating twice changes nothing.
        assert torch.allclose(ablated, ablate_residual(ablated, direction), atol=1e-5)

    def test_preserves_an_orthogonal_component(self) -> None:
        direction = torch.zeros(4)
        direction[0] = 3.0
        hidden = torch.tensor([[5.0, 2.0, -1.0, 4.0]])

        ablated = ablate_residual(hidden, direction)

        # Only the first coordinate (along the direction) is removed.
        assert ablated[0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert ablated[0, 1:].tolist() == pytest.approx([2.0, -1.0, 4.0])


class TestLogitRecovery:
    def test_full_recovery_is_one_and_no_change_is_zero(self) -> None:
        clean = torch.tensor([2.0, 0.0, -1.0])
        corrupted = torch.tensor([0.0, 1.0, -1.0])
        assert logit_recovery(clean, corrupted, clean, 0) == pytest.approx(1.0)
        assert logit_recovery(clean, corrupted, corrupted, 0) == pytest.approx(0.0)

    def test_zero_gap_returns_zero(self) -> None:
        clean = torch.tensor([1.0, 1.0])
        corrupted = torch.tensor([1.0, 5.0])  # token 0 gap is zero
        assert logit_recovery(clean, corrupted, torch.tensor([9.0, 9.0]), 0) == 0.0


class TestChooseAnswerToken:
    """The readout token must have dynamic range, and must still be one the clean run would say."""

    def test_prefers_the_top_k_candidate_the_corrupted_run_most_dislikes(self) -> None:
        # Token 0 is the clean argmax but the runs agree on it; token 1 is where they disagree.
        clean = torch.tensor([10.0, 9.5, 9.0, -50.0])
        corrupted = torch.tensor([10.0, 2.0, 8.0, -80.0])

        assert choose_answer_token(clean, corrupted, top_k=3) == 1

    def test_ignores_a_huge_gap_on_a_token_the_clean_run_would_never_emit(self) -> None:
        """Maximising the gap over the whole vocabulary would pick token 3, which is implausible."""
        clean = torch.tensor([10.0, 9.5, 9.0, -50.0])
        corrupted = torch.tensor([10.0, 2.0, 8.0, -80.0])

        assert choose_answer_token(clean, corrupted, top_k=3) != 3

    def test_top_k_larger_than_the_vocabulary_is_clamped(self) -> None:
        clean = torch.tensor([1.0, 3.0])
        corrupted = torch.tensor([1.0, 0.0])
        assert choose_answer_token(clean, corrupted, top_k=99) == 1


class TestHooks:
    def test_steering_hook_modifies_a_tuple_output(self) -> None:
        direction = torch.randn(8)
        hook = steering_hook(direction, alpha=3.0)
        base = torch.randn(1, 2, 8)

        modified = hook(None, None, (base, "attn_cache"))

        assert isinstance(modified, tuple)
        assert modified[1] == "attn_cache"  # trailing outputs pass through untouched
        assert torch.allclose(modified[0], steer_residual(base, direction, 3.0), atol=1e-5)

    def test_ablation_hook_modifies_a_bare_tensor_output(self) -> None:
        direction = torch.randn(8)
        base = torch.randn(1, 2, 8)
        modified = ablation_hook(direction)(None, None, base)
        assert torch.allclose(modified, ablate_residual(base, direction), atol=1e-5)

    def test_intervention_installs_and_removes_the_hook(self) -> None:
        model = _PatchCausalLM()
        layer = model.model.layers[0]
        assert len(layer._forward_hooks) == 0

        with residual_intervention(model, 0, steering_hook(torch.randn(4), 1.0)):
            assert len(layer._forward_hooks) == 1
        assert len(layer._forward_hooks) == 0


class TestSteeringDirections:
    def test_real_arm_plus_matched_norm_placebos(self) -> None:
        direction = torch.randn(2560) * 4.0
        arms = steering_directions(direction, n_placebos=100, seed=0)

        assert arms["real"] is direction
        assert len([k for k in arms if k.startswith("placebo_")]) == 100
        for name, arm in arms.items():
            if name == "real":
                continue
            assert arm.norm().item() == pytest.approx(direction.norm().item(), rel=1e-5)
            assert abs(cosine(direction, arm)) < 0.1  # near-orthogonal in high dimension

    def test_is_reproducible_under_a_fixed_seed(self) -> None:
        direction = torch.randn(64)
        first = steering_directions(direction, n_placebos=3, seed=7)
        second = steering_directions(direction, n_placebos=3, seed=7)
        for name in first:
            assert torch.equal(first[name], second[name])


class _IdentityLayer(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _CumulativeSumLayer(torch.nn.Module):
    """Causal mixing: position t sums positions 0..t, so an earlier patch reaches the readout."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(hidden, dim=1)


class _PatchTrunk(torch.nn.Module):
    def __init__(self, hidden: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([_IdentityLayer(), _CumulativeSumLayer()])

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> SimpleNamespace:
        del attention_mask, kwargs
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _RecordingUnembed(torch.nn.Module):
    """An LM head that records how many positions it was asked to project.

    That count is the memory fact worth asserting: the real vocab is ~248k, so projecting every
    position of a long prompt is the multi-GiB tensor the rest of the package exists to avoid,
    and a readout needs exactly one row of it.
    """

    def __init__(self, hidden: int, vocab: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(hidden, vocab)
        self.projected_positions: int | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.projected_positions = int(hidden_states.shape[1])
        return self.projection(hidden_states)


class _PatchCausalLM(torch.nn.Module):
    """Trunk + LM head, deterministic, for the activation-patching integration test.

    ``logits_to_keep`` is honoured exactly as ``Qwen3_5ForCausalLM.forward`` does it (an int becomes
    ``slice(-n, None)``, a tensor indexes the sequence axis directly), so a caller that narrows the
    vocab projection is measured here rather than silently ignored.
    """

    def __init__(self, hidden: int = 4, vocab: int = 16) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.model = _PatchTrunk(hidden, vocab)
        self.lm_head = _RecordingUnembed(hidden, vocab)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        outputs = self.model(input_ids, attention_mask)
        # transformers reads an int as a from-the-end count and a tensor as explicit indices.
        kept = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return SimpleNamespace(logits=self.lm_head(outputs.last_hidden_state[:, kept, :]))

    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> torch.Tensor:
        del attention_mask, kwargs
        return torch.cat([input_ids, torch.tensor([[5]], dtype=torch.long)], dim=1)


class TestActivationPatch:
    def test_patching_position_one_recovers_the_clean_readout(self) -> None:
        """Clean and corrupted twins differ only at position 1; patch it and the readout goes clean.

        The identity layer's position-1 output is what the patch overwrites; the cumulative-sum
        layer then carries the patched value into the last-position readout. Since the twins
        share the other positions, restoring position 1 restores the whole readout -- recovery
        ~1.0, the causal-use signal. Watched to fail if the patch were a no-op: without it,
        corrupted != clean here.
        """
        model = _PatchCausalLM(hidden=4, vocab=16)
        clean_ids = torch.tensor([[2, 3, 5]])
        corrupted_ids = torch.tensor([[2, 9, 5]])  # byte-identical but for position 1
        masks = torch.ones_like(clean_ids)

        result = run_activation_patch(
            model,
            clean_ids=clean_ids,
            corrupted_ids=corrupted_ids,
            clean_mask=masks,
            corrupted_mask=torch.ones_like(corrupted_ids),
            layer=0,
            clean_positions=torch.tensor([1]),
            corrupted_positions=torch.tensor([1]),
        )

        # The patch reaches the readout: patched logits match clean, and differ from corrupted.
        assert not torch.allclose(result.clean, result.corrupted, atol=1e-4)
        assert torch.allclose(result.patched, result.clean, atol=1e-4)
        assert result.recovery == pytest.approx(1.0, abs=1e-3)

    def test_patches_unequal_length_twins_through_paired_positions(self) -> None:
        """The live case: the corrupted twin is longer, so the two runs need separate indices.

        The ILCB conflicting grader is the original plus an extra assertion, so no eligible pair
        tokenizes equal. Slot i of each position tensor is one aligned position, which is what lets
        the divergent region be patched at all; the earlier single-tensor API could only address
        offsets that meant different content in each run.
        """
        model = _PatchCausalLM(hidden=4, vocab=16)
        clean_ids = torch.tensor([[2, 3, 5]])
        corrupted_ids = torch.tensor([[2, 9, 9, 5]])

        result = run_activation_patch(
            model,
            clean_ids=clean_ids,
            corrupted_ids=corrupted_ids,
            clean_mask=torch.ones_like(clean_ids),
            corrupted_mask=torch.ones_like(corrupted_ids),
            layer=0,
            clean_positions=torch.tensor([1]),
            corrupted_positions=torch.tensor([2]),
        )

        assert not torch.allclose(result.patched, result.corrupted, atol=1e-4)

    def test_mismatched_position_counts_raise_instead_of_patching(self) -> None:
        """Unequal position counts describe no aligned region, yet would still print a number."""
        model = _PatchCausalLM(hidden=4, vocab=16)
        clean_ids = torch.tensor([[2, 3, 5]])
        corrupted_ids = torch.tensor([[2, 9, 5, 7]])

        with pytest.raises(ValueError, match="same number of slots"):
            run_activation_patch(
                model,
                clean_ids=clean_ids,
                corrupted_ids=corrupted_ids,
                clean_mask=torch.ones_like(clean_ids),
                corrupted_mask=torch.ones_like(corrupted_ids),
                layer=0,
                clean_positions=torch.tensor([1]),
                corrupted_positions=torch.tensor([1, 2]),
            )

    def test_positions_past_the_end_of_a_run_raise(self) -> None:
        """A stale position tensor would index into whatever the sequence happens to hold."""
        model = _PatchCausalLM(hidden=4, vocab=16)
        clean_ids = torch.tensor([[2, 3, 5]])

        with pytest.raises(ValueError, match="outside the 3-token corrupted run"):
            run_activation_patch(
                model,
                clean_ids=clean_ids,
                corrupted_ids=torch.tensor([[2, 9, 5]]),
                clean_mask=torch.ones_like(clean_ids),
                corrupted_mask=torch.ones_like(clean_ids),
                layer=0,
                clean_positions=torch.tensor([1]),
                corrupted_positions=torch.tensor([7]),
            )

    def test_patch_hook_overwrites_only_the_named_positions(self) -> None:
        base = torch.randn(1, 4, 6)
        clean_rows = torch.zeros(2, 6)
        hook = activation_patch_hook(clean_rows, torch.tensor([1, 3]))

        modified = hook(None, None, base)

        assert torch.allclose(modified[:, [1, 3], :], torch.zeros(1, 2, 6))
        assert torch.allclose(modified[:, [0, 2], :], base[:, [0, 2], :])


class TestReadoutNeedsABaselineCarryingItsGaps:
    """A scored readout plus a baseline holding no gaps is refused, not answered with ``None``.

    ``PatchBaseline``'s two gap fields default to absent, because the single-next-token readout has no
    gaps to carry. So the plain constructor -- how every logit-readout caller in both threads builds
    one -- produced a baseline that satisfied the readout path and carried nothing for it to read: the
    patched forward was paid for, ``recovery_gap`` came back ``None``, and a sweep row recorded
    ``readout_mode=None`` for every cell while the stage still looked successful. ``recovery_gap`` is
    the PRIMARY reading under a readout, so that is the whole measurement silently absent.
    """

    @staticmethod
    def _readout() -> GapReadout:
        """Two two-token candidates, so the gap is a sequence log-prob difference as in the real run."""
        return gap_readout(
            mode=READOUT_MODE_ACTION_LOGPROB,
            positive_name="tamper",
            negative_name="honest",
            positive_ids=torch.tensor([6, 7]),
            negative_ids=torch.tensor([8, 9]),
        )

    def test_a_gapless_baseline_under_a_readout_is_refused_before_any_forward(self) -> None:
        model = _PatchCausalLM(hidden=4, vocab=16)
        clean_ids = torch.tensor([[2, 3, 5]])
        corrupted_ids = torch.tensor([[2, 9, 5]])
        rows_only = PatchBaseline(clean=torch.zeros(16), corrupted=torch.zeros(16))

        with pytest.raises(ValueError, match="from_gap_reads"):
            run_activation_patch(
                model,
                clean_ids=clean_ids,
                corrupted_ids=corrupted_ids,
                clean_mask=torch.ones_like(clean_ids),
                corrupted_mask=torch.ones_like(corrupted_ids),
                layer=0,
                clean_positions=torch.tensor([1]),
                corrupted_positions=torch.tensor([1]),
                baseline=rows_only,
                readout=self._readout(),
            )

        assert model.lm_head.projected_positions is None, (
            "the refusal exists to come BEFORE the forward whose result would be discarded"
        )

    def test_from_gap_reads_carries_both_gaps_into_a_populated_recovery_gap(self) -> None:
        """The constructor the raise names, end to end: the reading comes back a number.

        Patching position 1 at layer 0 makes the corrupted run's residual stream identical to the clean
        run's from there on, so the patched gap IS the clean gap and the recovery reads 1.0 -- the same
        ground truth ``TestActivationPatch`` reads on the single-logit path, now on the gap.
        """
        model = _PatchCausalLM(hidden=4, vocab=16)
        clean_ids = torch.tensor([[2, 3, 5]])
        corrupted_ids = torch.tensor([[2, 9, 5]])
        readout = self._readout()
        clean_read = read_gap(model, clean_ids, torch.ones_like(clean_ids), readout)
        corrupted_read = read_gap(model, corrupted_ids, torch.ones_like(corrupted_ids), readout)
        assert clean_read.gap != corrupted_read.gap, (
            "no denominator: the twins must read differently"
        )

        result = run_activation_patch(
            model,
            clean_ids=clean_ids,
            corrupted_ids=corrupted_ids,
            clean_mask=torch.ones_like(clean_ids),
            corrupted_mask=torch.ones_like(corrupted_ids),
            layer=0,
            clean_positions=torch.tensor([1]),
            corrupted_positions=torch.tensor([1]),
            baseline=PatchBaseline.from_gap_reads(clean_read, corrupted_read),
            readout=readout,
        )

        assert result.patched_gap is not None
        assert result.recovery_gap == pytest.approx(
            gap_recovery(clean_read.gap, corrupted_read.gap, result.patched_gap.gap)
        )
        assert result.recovery_gap == pytest.approx(1.0, abs=1e-3)

    def test_a_baseline_without_a_readout_still_needs_no_gaps(self) -> None:
        """The single-logit path is untouched: its baseline has no gaps to carry and never did."""
        model = _PatchCausalLM(hidden=4, vocab=16)
        clean_ids = torch.tensor([[2, 3, 5]])
        corrupted_ids = torch.tensor([[2, 9, 5]])
        rows = _readout_logits(model, clean_ids, torch.ones_like(clean_ids), -1)

        result = run_activation_patch(
            model,
            clean_ids=clean_ids,
            corrupted_ids=corrupted_ids,
            clean_mask=torch.ones_like(clean_ids),
            corrupted_mask=torch.ones_like(corrupted_ids),
            layer=0,
            clean_positions=torch.tensor([1]),
            corrupted_positions=torch.tensor([1]),
            baseline=PatchBaseline(clean=rows, corrupted=rows),
        )

        assert result.recovery_gap is None
        assert result.patched_gap is None


class TestPlanTwinPatchLadder:
    """The ladder's job is to keep the readout position OUT of the windows that measure a region."""

    @staticmethod
    def _ladder(clean_len: int = 40, corrupted_len: int = 50) -> TwinPatchPlan:
        """Twins sharing a 20-token prefix and a 5-token suffix, corrupted longer by an insertion."""
        prefix = list(range(1, 21))
        suffix = [90, 91, 92, 93, 94]
        clean_middle = [50] * (clean_len - len(prefix) - len(suffix))
        corrupted_middle = [60] * (corrupted_len - len(prefix) - len(suffix))
        return plan_twin_patch_ladder(
            torch.tensor(prefix + clean_middle + suffix),
            torch.tensor(prefix + corrupted_middle + suffix),
            tail_widths=(1, 4, 16),
            head_widths=(8,),
        )

    def test_only_readout_only_and_post_divergence_touch_the_readout_position(self) -> None:
        plan = self._ladder()
        corrupted_len = int(plan.corrupted_ids.shape[0])

        touching = {
            window.name
            for window in plan.windows
            if window_contains_readout(window, corrupted_len=corrupted_len)
        }

        assert touching == {PATCH_WINDOW_READOUT_ONLY, PATCH_WINDOW_POST_DIVERGENCE}

    def test_tail_widths_produce_windows_of_exactly_those_widths(self) -> None:
        plan = self._ladder()

        widths = {
            window.name: window.n_positions
            for window in plan.windows
            if window.name.startswith(TAIL_WINDOW_PREFIX)
        }

        assert widths == {
            "tail_excl_readout_1": 1,
            "tail_excl_readout_4": 4,
            "tail_excl_readout_16": 16,
        }

    def test_the_excl_readout_wide_window_is_one_position_narrower(self) -> None:
        plan = self._ladder()
        by_name = {window.name: window for window in plan.windows}

        wide = by_name[PATCH_WINDOW_POST_DIVERGENCE].n_positions
        narrowed = by_name[PATCH_WINDOW_POST_DIVERGENCE_EXCL_READOUT].n_positions

        assert wide == plan.post_divergence_len
        assert narrowed == wide - 1

    def test_widths_are_clamped_to_the_region_and_deduplicated(self) -> None:
        """Two requested widths above a short region's limit must not print one number twice."""
        prefix = [1, 2, 3, 4, 5, 6]
        clean = torch.tensor([*prefix, 50, 90])
        corrupted = torch.tensor([*prefix, 60, 61, 90])

        plan = plan_twin_patch_ladder(clean, corrupted, tail_widths=(4, 8, 16), head_widths=(4,))

        tails = [w.name for w in plan.windows if w.name.startswith(TAIL_WINDOW_PREFIX)]
        assert tails == ["tail_excl_readout_1"]

    def test_the_shared_prefix_control_is_still_emitted_last(self) -> None:
        plan = self._ladder()

        assert plan.windows[-1].name == PATCH_WINDOW_SHARED_PREFIX
        assert torch.equal(plan.windows[-1].clean_positions, plan.windows[-1].corrupted_positions)


class TestReadoutPositionArtifact:
    """The finding that motivated the ladder, on the fake: patching the readout position is a cheat.

    The fake's last layer is a cumulative sum, so patching a NON-readout position at layer 1 still
    reaches the readout -- which is the honest signal. Patching the readout position itself at layer 1
    transplants the clean row wholesale, so recovery is 1.0 whatever the rest of the run holds. The
    real 4B showed the same shape at layer 31: recovery exactly 1.0 from the window containing the
    readout beside a max logit shift of exactly 0.0 from a window that did not.
    """

    def test_patching_the_readout_position_recovers_fully_on_its_own(self) -> None:
        model = _PatchCausalLM(hidden=4, vocab=16)
        clean_ids = torch.tensor([[2, 3, 5, 7]])
        corrupted_ids = torch.tensor([[2, 9, 11, 13]])

        result = run_activation_patch(
            model,
            clean_ids=clean_ids,
            corrupted_ids=corrupted_ids,
            clean_mask=torch.ones_like(clean_ids),
            corrupted_mask=torch.ones_like(corrupted_ids),
            layer=1,
            clean_positions=torch.tensor([3]),
            corrupted_positions=torch.tensor([3]),
        )

        assert result.recovery == pytest.approx(1.0, abs=1e-3)
        assert torch.allclose(result.patched, result.clean, atol=1e-4)

    def test_patching_everything_but_the_readout_position_does_not(self) -> None:
        """Same layer, same twins, readout position excluded -- and the recovery is no longer 1.0."""
        model = _PatchCausalLM(hidden=4, vocab=16)
        clean_ids = torch.tensor([[2, 3, 5, 7]])
        corrupted_ids = torch.tensor([[2, 9, 11, 13]])

        result = run_activation_patch(
            model,
            clean_ids=clean_ids,
            corrupted_ids=corrupted_ids,
            clean_mask=torch.ones_like(clean_ids),
            corrupted_mask=torch.ones_like(corrupted_ids),
            layer=1,
            clean_positions=torch.tensor([0, 1, 2]),
            corrupted_positions=torch.tensor([0, 1, 2]),
        )

        assert result.recovery != pytest.approx(1.0, abs=1e-3)


class TestAxisRestrictedReplacements:
    def test_component_plus_complement_reconstructs_the_full_clean_rows(self) -> None:
        clean = torch.randn(5, 8)
        corrupted = torch.randn(5, 8)
        direction = torch.randn(8)

        component = axis_component_replacement(clean, corrupted, direction)
        complement = axis_complement_replacement(clean, corrupted, direction)

        assert torch.allclose(component + complement - corrupted, clean, atol=1e-5)

    def test_the_component_moves_only_along_the_axis(self) -> None:
        clean = torch.randn(5, 8)
        corrupted = torch.randn(5, 8)
        direction = torch.randn(8)
        axis = unit(direction)

        moved = axis_component_replacement(clean, corrupted, direction) - corrupted

        orthogonal = moved - (moved @ axis).unsqueeze(-1) * axis
        assert float(orthogonal.abs().max()) == pytest.approx(0.0, abs=1e-5)

    def test_the_complement_moves_nothing_along_the_axis(self) -> None:
        clean = torch.randn(5, 8)
        corrupted = torch.randn(5, 8)
        direction = torch.randn(8)

        moved = axis_complement_replacement(clean, corrupted, direction) - corrupted

        assert float((moved @ unit(direction)).abs().max()) == pytest.approx(0.0, abs=1e-5)

    def test_a_norm_matched_random_axis_placebo_matches_the_real_components_norm(self) -> None:
        clean = torch.randn(5, 8)
        corrupted = torch.randn(5, 8)
        direction = torch.randn(8)
        real_norm = float(
            (axis_component_replacement(clean, corrupted, direction) - corrupted).norm()
        )

        placebo = random_axis_replacement(
            clean, corrupted, torch.Generator().manual_seed(0), match_norm_to=real_norm
        )

        assert float((placebo - corrupted).norm()) == pytest.approx(real_norm, rel=1e-5)

    def test_an_unscaled_random_axis_placebo_captures_far_less_than_the_real_axis(self) -> None:
        """The point of the unscaled arm: a random direction in 512-d catches ~1/sqrt(d) of a delta."""
        torch.manual_seed(0)
        corrupted = torch.randn(4, 512)
        direction = torch.randn(512)
        clean = corrupted + 3.0 * unit(direction)

        real = float((axis_component_replacement(clean, corrupted, direction) - corrupted).norm())
        placebo = float(
            (
                random_axis_replacement(clean, corrupted, torch.Generator().manual_seed(0))
                - corrupted
            ).norm()
        )

        assert placebo < real / 5


class TestRecoveryMetrics:
    def test_a_full_recovery_reads_one_on_every_metric(self) -> None:
        clean = torch.tensor([0.0, 3.0, 1.0, -2.0])
        corrupted = torch.tensor([0.0, 1.0, 1.0, 0.0])

        metrics = recovery_metrics(clean, corrupted, clean, answer_token=1)

        assert metrics["recovery_logit"] == pytest.approx(1.0)
        assert metrics["recovery_logprob_l2"] == pytest.approx(1.0)
        assert metrics["recovery_kl"] == pytest.approx(1.0)

    def test_no_change_reads_zero_on_every_metric(self) -> None:
        clean = torch.tensor([0.0, 3.0, 1.0, -2.0])
        corrupted = torch.tensor([0.0, 1.0, 1.0, 0.0])

        metrics = recovery_metrics(clean, corrupted, corrupted, answer_token=1)

        assert metrics["recovery_logit"] == pytest.approx(0.0)
        assert metrics["recovery_logprob_l2"] == pytest.approx(0.0)
        assert metrics["recovery_kl"] == pytest.approx(0.0)

    def test_the_row_metrics_separate_two_patches_the_single_logit_cannot(self) -> None:
        """Why the extra metrics exist: the bf16 readout grid collapses distinct patches to one ratio.

        Two patched rows carrying the SAME answer-token logit but different rest-of-vocabulary rows
        read identically under ``recovery_logit`` and differently under the distributional readings.
        """
        clean = torch.tensor([0.0, 3.0, 1.0, -2.0])
        corrupted = torch.tensor([0.0, 1.0, 1.0, 0.0])
        near_clean = torch.tensor([0.0, 2.0, 1.0, -2.0])
        far_from_clean = torch.tensor([0.0, 2.0, 6.0, 5.0])

        near = recovery_metrics(clean, corrupted, near_clean, answer_token=1)
        far = recovery_metrics(clean, corrupted, far_from_clean, answer_token=1)

        assert near["recovery_logit"] == pytest.approx(far["recovery_logit"])
        assert _reading(near, "recovery_kl") > _reading(far, "recovery_kl")
        assert _reading(near, "recovery_logprob_l2") > _reading(far, "recovery_logprob_l2")

    def test_an_overshoot_is_recorded_rather_than_clipped(self) -> None:
        clean = torch.tensor([0.0, 3.0, 1.0, -2.0])
        corrupted = torch.tensor([0.0, 1.0, 1.0, 0.0])
        overshot = torch.tensor([0.0, 9.0, 1.0, -2.0])

        metrics = recovery_metrics(clean, corrupted, overshot, answer_token=1)

        assert _reading(metrics, "recovery_logit") > 1.0
        assert _reading(metrics, "recovery_kl") < 0.0


class TestReadoutLogits:
    """The readout projects ONE position, and returns the same row the full projection would.

    ``directions`` documents the ``[batch, seq, ~248k]`` vocab projection as the OOM its trunk-only
    capture exists to avoid; the patch readout reintroduced it three times per patch cell to read a
    single ``[vocab]`` row.
    """

    @pytest.mark.parametrize("readout_position", [-1, 1, 0])
    def test_reads_one_position_and_matches_the_full_projection(
        self, readout_position: int
    ) -> None:
        model = _PatchCausalLM(hidden=4, vocab=16)
        input_ids = torch.tensor([[2, 3, 5, 7]])
        attention_mask = torch.ones_like(input_ids)
        every_position = model(input_ids=input_ids, attention_mask=attention_mask).logits

        row = _readout_logits(model, input_ids, attention_mask, readout_position)

        assert torch.allclose(row, every_position[0, readout_position], atol=1e-5)
        assert model.lm_head.projected_positions == 1


class TestSteeredGeneration:
    def test_runs_and_leaves_no_hook_installed(self) -> None:
        model = _PatchCausalLM()

        class _Tok:
            pad_token_id = 0

            def apply_chat_template(self, messages: object, **kwargs: object) -> str:
                del kwargs
                return "hi"

            def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
                del kwargs
                ids = torch.tensor([[1, 2, 3]])
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

            def decode(self, ids: torch.Tensor, **kwargs: object) -> str:
                del kwargs
                return "x"

            def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
                return [str(i) for i in ids]

        record = run_steered_generation(
            model,
            _Tok(),
            "solve",
            layer=0,
            direction=torch.randn(4),
            alpha=1.0,
            mode="steer",
            sampling=_GREEDY,
        )

        assert record.n_generated == 1
        assert len(model.model.layers[0]._forward_hooks) == 0

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            run_steered_generation(
                _PatchCausalLM(),
                object(),
                "p",
                layer=0,
                direction=torch.randn(4),
                alpha=1.0,
                mode="scramble",
            )


# The DEFECT A substrate: a thinking model whose scripted continuation is a long reasoning block
# followed by ONE protocol action. Truncating the continuation at a small token cap stops it inside
# the reasoning block, so no action reaches the parser; a generous cap lets the action through. This
# is the exact floor the steering read fell into -- 512 tokens against multi-thousand-token traces --
# reproduced on CPU with no model load.
_THINK_ID = 7
_BOUNDARY_ID = 8
_RUN_OPEN_ID = 9
_COMMAND_ID = 10
_RUN_CLOSE_ID = 11
_SCRIPT_PIECES = {
    _THINK_ID: "reasoning ",
    _BOUNDARY_ID: "</think>\n",
    _RUN_OPEN_ID: "<run>",
    _COMMAND_ID: "cat grade.py",
    _RUN_CLOSE_ID: "</run>",
}


class _ScriptedTokenizer:
    """Decodes the scripted continuation ids back to their text pieces; a fixed prompt otherwise."""

    pad_token_id = 0

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        del messages, kwargs
        return "PROMPT"

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        del text, kwargs
        ids = torch.tensor([[1, 2, 3]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids: Sequence[int], **kwargs: object) -> str:
        """Takes a plain id sequence, as a real tokenizer does alongside tensors and arrays."""
        del kwargs
        return "".join(_SCRIPT_PIECES.get(int(i), "") for i in ids)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [str(i) for i in ids]


class _ScriptedGenLM(torch.nn.Module):
    """``generate`` emits a fixed continuation truncated to ``max_new_tokens``.

    A small cap stops inside the ``reasoning`` run; a large one reaches the appended action. That is
    the whole DEFECT A mechanism, so the cap is the only variable between the floored and freed reads.
    """

    def __init__(self, think_len: int) -> None:
        super().__init__()
        self._continuation = [_THINK_ID] * think_len + [
            _BOUNDARY_ID,
            _RUN_OPEN_ID,
            _COMMAND_ID,
            _RUN_CLOSE_ID,
        ]

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def generate(
        self, input_ids: torch.Tensor, *, max_new_tokens: int, **kwargs: object
    ) -> torch.Tensor:
        del kwargs
        emitted = torch.tensor([self._continuation[:max_new_tokens]], dtype=torch.long)
        return torch.cat([input_ids, emitted], dim=1)


_ACTION_TEXT = "<run>cat grade.py</run>"


def _capped(max_new_tokens: int) -> SamplingConfig:
    """The default interp sampler made greedy and capped: the cap is the only variable here."""
    return replace(_GREEDY, max_new_tokens=max_new_tokens)


class TestSteeringTokenCapFloor:
    """DEFECT A: a too-small generation cap truncates the response inside the thinking block, so the
    action never makes it into the text the observables are read off, and every count floors at zero.

    Asserted on the response TEXT rather than through ``parse_actions``: importing the action parser
    pulls in the harness package, which reads baked case files at import and would data-gate this
    whole (otherwise model-free) module. The parser is exercised on the same text in
    ``test_interp_run_harness`` via ``behavioral_observables``; here the point is that the cap decides
    whether the action text exists at all. Sabotage-verified: making the scripted model ignore the
    cap (emit the full continuation regardless) makes the floored assertions fail, since the action
    then survives even at ``max_new_tokens=8``.
    """

    def test_a_small_cap_truncates_inside_thinking_and_the_action_never_appears(self) -> None:
        model = _ScriptedGenLM(think_len=40)
        record = generate_response(model, _ScriptedTokenizer(), "solve", sampling=_capped(8))
        assert record.hit_token_cap is True
        assert "<run>" not in record.response_text  # floored: the action was cut off mid-thinking

    def test_an_adequate_cap_lets_the_model_finish_and_emit_the_action(self) -> None:
        model = _ScriptedGenLM(think_len=40)
        record = generate_response(model, _ScriptedTokenizer(), "solve", sampling=_capped(10_000))
        assert record.hit_token_cap is False  # stopped on its own, not on the cap
        assert _ACTION_TEXT in record.response_text  # the grader-reaching action survives

    def test_whether_the_action_appears_can_differ_between_two_caps(self) -> None:
        # The observable has real dynamic range: the floor was the defect, not the metric. Under the
        # bug every arm read the same zero; once the cap clears the thinking length, arms CAN differ.
        model = _ScriptedGenLM(think_len=40)
        tokenizer = _ScriptedTokenizer()
        floored = generate_response(model, tokenizer, "solve", sampling=_capped(8)).response_text
        freed = generate_response(model, tokenizer, "solve", sampling=_capped(10_000)).response_text
        assert (_ACTION_TEXT in floored) != (_ACTION_TEXT in freed)


class TestPatchPrefixReplay:
    """Replaying layers 0..L instead of recomputing them per arm (hot-path backlog rank 27).

    The saving: the layers below the patched one run once per (pair, layer) to build the prefix rather
    than once per arm, which over a full-layer sweep is about half the patched-forward compute. The
    thing that must not move: every reading. Both are checked here on the fake whose last layer is a
    cumulative sum, so a patch below the readout genuinely propagates and a replay that handed a later
    layer the wrong residual would change the answer rather than being invisible.

    Both readout shapes are covered, because they capture different numbers of forwards: the plain
    single-logit readout runs one forward over the corrupted ids, while a multi-token
    :class:`GapReadout` runs one per candidate over prompt-plus-candidate ids, and the prefix has to
    hold the right layer output for each.
    """

    @staticmethod
    def _readout() -> GapReadout:
        return gap_readout(
            mode=READOUT_MODE_ACTION_LOGPROB,
            positive_name="tamper",
            negative_name="honest",
            positive_ids=torch.tensor([6, 7]),
            negative_ids=torch.tensor([8, 9]),
        )

    @staticmethod
    def _twins() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([[2, 3, 5, 7]]), torch.tensor([[2, 9, 11, 13]])

    def _baseline(self, model: _PatchCausalLM, readout: GapReadout | None) -> PatchBaseline:
        """The two un-patched readouts of the twins, as every real driver resolves them once per pair."""
        clean_ids, corrupted_ids = self._twins()
        if readout is not None:
            return PatchBaseline.from_gap_reads(
                read_gap(model, clean_ids, torch.ones_like(clean_ids), readout),
                read_gap(model, corrupted_ids, torch.ones_like(corrupted_ids), readout),
            )
        return PatchBaseline(
            clean=_readout_logits(model, clean_ids, torch.ones_like(clean_ids), -1),
            corrupted=_readout_logits(model, corrupted_ids, torch.ones_like(corrupted_ids), -1),
        )

    def _patch(  # noqa: PLR0913 - a cell is the model, the layer, the rows, the readout and the prefix
        self,
        model: _PatchCausalLM,
        *,
        layer: int,
        rows: torch.Tensor,
        readout: GapReadout | None,
        prefix: object | None,
        baseline: PatchBaseline | None = None,
    ) -> Any:
        clean_ids, corrupted_ids = self._twins()
        if baseline is None:
            baseline = self._baseline(model, readout)
        return run_activation_patch(
            model,
            clean_ids=clean_ids,
            corrupted_ids=corrupted_ids,
            clean_mask=torch.ones_like(clean_ids),
            corrupted_mask=torch.ones_like(corrupted_ids),
            layer=layer,
            clean_positions=torch.tensor([1, 2]),
            corrupted_positions=torch.tensor([1, 2]),
            replacement_rows=rows,
            baseline=baseline,
            readout=readout,
            prefix=cast("Any", prefix),
        )

    def _prefix(self, model: _PatchCausalLM, layer: int, readout: GapReadout | None) -> Any:
        _, corrupted_ids = self._twins()
        return capture_patch_prefix(
            model,
            corrupted_ids=corrupted_ids,
            corrupted_mask=torch.ones_like(corrupted_ids),
            layer=layer,
            readout=readout,
        )

    @pytest.mark.parametrize("with_readout", [False, True], ids=["single_logit", "scored_gap"])
    @pytest.mark.parametrize("layer", [0, 1])
    def test_the_replayed_patch_reads_exactly_what_the_forward_hook_read(
        self, layer: int, with_readout: bool
    ) -> None:
        model = _PatchCausalLM(hidden=4, vocab=16)
        readout = self._readout() if with_readout else None
        rows = torch.randn(2, 4)

        hooked = self._patch(model, layer=layer, rows=rows, readout=readout, prefix=None)
        replayed = self._patch(
            model,
            layer=layer,
            rows=rows,
            readout=readout,
            prefix=self._prefix(model, layer, readout),
        )

        assert torch.equal(replayed.patched, hooked.patched)
        assert replayed.recovery == hooked.recovery
        if readout is None:
            assert replayed.recovery_gap is None
        else:
            assert replayed.patched_gap is not None
            assert hooked.patched_gap is not None
            assert replayed.patched_gap.gap == hooked.patched_gap.gap
            assert replayed.recovery_gap == hooked.recovery_gap

    def test_a_patch_that_reaches_the_readout_still_reads_a_full_recovery_under_the_replay(
        self,
    ) -> None:
        """The replay is not vacuously equal: a real effect still comes through it.

        These twins differ ONLY at positions 1..2, so patching exactly those with the clean run's own
        rows makes the corrupted run identical to the clean one there, and the cumulative-sum layer
        above the patch carries that to the readout: recovery 1.0, computed entirely through a layer the
        replay did not recompute. A replay that handed that layer a stale or wrong residual could not
        produce it.
        """
        model = _PatchCausalLM(hidden=4, vocab=16)
        clean_ids = torch.tensor([[2, 3, 5, 7]])
        corrupted_ids = torch.tensor([[2, 9, 11, 7]])
        positions = torch.tensor([1, 2])
        clean_rows = capture_positionwise_activations(
            model, clean_ids, torch.ones_like(clean_ids), layers=[0]
        )[0][0, positions]
        prefix = capture_patch_prefix(
            model,
            corrupted_ids=corrupted_ids,
            corrupted_mask=torch.ones_like(corrupted_ids),
            layer=0,
        )

        result = run_activation_patch(
            model,
            clean_ids=clean_ids,
            corrupted_ids=corrupted_ids,
            clean_mask=torch.ones_like(clean_ids),
            corrupted_mask=torch.ones_like(corrupted_ids),
            layer=0,
            clean_positions=positions,
            corrupted_positions=positions,
            replacement_rows=clean_rows,
            prefix=prefix,
        )

        assert result.recovery == pytest.approx(1.0, abs=1e-3)
        assert torch.allclose(result.patched, result.clean, atol=1e-4)

    def test_the_layers_below_the_patch_are_not_re_entered_per_arm(self) -> None:
        """The saving, counted: layer 0 runs once for the prefix and never inside a replayed arm."""
        model = _PatchCausalLM(hidden=4, vocab=16)
        entries = {"count": 0}
        real_forward = model.model.layers[0].forward

        def counting_forward(*args: Any, **kwargs: Any) -> Any:
            entries["count"] += 1
            return real_forward(*args, **kwargs)

        baseline = self._baseline(model, None)
        model.model.layers[0].forward = counting_forward  # pyright: ignore[reportAttributeAccessIssue]
        prefix = self._prefix(model, 1, None)
        assert entries["count"] == 1, "the prefix costs one forward through the lower layers"
        for _ in range(4):
            self._patch(
                model,
                layer=1,
                rows=torch.randn(2, 4),
                readout=None,
                prefix=prefix,
                baseline=baseline,
            )
        assert entries["count"] == 1, "four arms, and none of them re-entered layer 0"

    def test_a_replay_context_restores_the_real_layers_even_when_the_forward_raises(self) -> None:
        model = _PatchCausalLM(hidden=4, vocab=16)
        _, corrupted_ids = self._twins()
        prefix = self._prefix(model, 1, None)
        captured = prefix.for_ids(corrupted_ids)
        originals = [model.model.layers[0], model.model.layers[1]]

        with (
            pytest.raises(RuntimeError, match="while replaying"),
            replay_from_layer(model, captured, captured.on_device(corrupted_ids.device)),
        ):
            raise RuntimeError("something failed while replaying")

        assert [model.model.layers[0], model.model.layers[1]] == originals

    def test_a_prefix_over_other_ids_or_another_layer_is_refused(self) -> None:
        model = _PatchCausalLM(hidden=4, vocab=16)
        prefix = self._prefix(model, 0, None)
        with pytest.raises(ValueError, match="captured at layer 0, not at layer 1"):
            self._patch(model, layer=1, rows=torch.randn(2, 4), readout=None, prefix=prefix)
        with pytest.raises(ValueError, match="no layer-0 output was captured"):
            run_activation_patch(
                model,
                clean_ids=torch.tensor([[2, 3, 5, 7]]),
                corrupted_ids=torch.tensor([[2, 9, 11]]),
                clean_mask=torch.ones(1, 4, dtype=torch.long),
                corrupted_mask=torch.ones(1, 3, dtype=torch.long),
                layer=0,
                clean_positions=torch.tensor([1]),
                corrupted_positions=torch.tensor([1]),
                replacement_rows=torch.randn(1, 4),
                prefix=prefix,
            )

    def test_the_captured_residual_is_kept_in_the_model_s_own_dtype(self) -> None:
        """The prefix is handed back to the model, never computed on, so it stores the model's dtype.

        A float32 upcast here would double the host copy and the transfer back for every patched
        forward -- ~100 MB instead of ~50 MB per 6k-token bf16 residual on the reward-hacking sweep --
        and buy nothing, so the dtype is asserted rather than left to drift.
        """
        for dtype in (torch.float32, torch.bfloat16):
            model = _PatchCausalLM(hidden=4, vocab=16).to(dtype)
            _, corrupted_ids = self._twins()
            captured = self._prefix(model, 0, None).for_ids(corrupted_ids)
            assert captured.output.dtype == dtype
            assert captured.dtype == dtype
            assert captured.output.device.type == "cpu"

    def test_a_scored_gap_prefix_covers_both_candidates(self) -> None:
        """A multi-token readout forwards prompt-plus-candidate twice, so the prefix holds two outputs."""
        model = _PatchCausalLM(hidden=4, vocab=16)
        readout = self._readout()
        prefix = self._prefix(model, 0, readout)
        _, corrupted_ids = self._twins()

        assert len(prefix.outputs) == 2
        for candidate in (readout.positive_ids, readout.negative_ids):
            full = torch.cat([corrupted_ids, candidate.unsqueeze(0)], dim=1)
            assert prefix.for_ids(full).output.shape[1] == full.shape[1]
        with pytest.raises(ValueError, match="no layer-0 output was captured"):
            prefix.for_ids(corrupted_ids)
