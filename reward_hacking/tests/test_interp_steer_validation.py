# HARNESS-SCAN-EXEMPT-monolithic-file-loc -- one suite for two modules; the banners are the split
"""Offline tests for the steering instrument-validation rig: scorers, corpus, hooks, artifacts.

CPU only, no model load. The deterministic scorers are pinned against hand-computed values; the
position masks, the hook accounting and ``assert_hook_pattern`` are exercised on tiny real
``nn.Module`` trunks; the teacher-forced scorer runs on a position-local fake LM whose
log-probabilities are computable by hand, which is what makes the right-padding batch-safety
invariant an exact assertion rather than a vibe. Several tests exist to prove a guard can go RED --
``assert_hook_pattern``'s failure modes, the hook's own shape guard, the sabotage inversion in
``isolation_verdict`` -- because a check nobody has watched fail is not yet a check.

On the type suppressions: duck-typed fakes stand in for real HF models/tokenizers, so
``reportArgumentType`` fires at the call sites, and the output-modifying hooks are typed to return
``object`` (the layer output at the hook boundary), so indexing or subtracting a hook's return in an
assertion trips ``reportIndexIssue``/``reportOperatorIssue``. Those are the fakes, not real defects
-- each such site carries its own ``# pyright: ignore[...]``, so a NEW diagnostic anywhere else in
the file still fails ``make typecheck`` instead of being swallowed by a file-wide directive.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import zlib
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from reward_hacking.interp import run_steer_validation as run_steer_validation_module
from reward_hacking.interp import steer_validation as steer_validation_module
from reward_hacking.interp.directions import cosine, diff_of_means, unit
from reward_hacking.interp.generation_capture import GenerationRecord, resolved_sampler
from reward_hacking.interp.run_steer_validation import (
    SIDE_BASELINE,
    SIDE_TARGET,
    BehaviourSweep,
    FitGeneration,
    _ols_slope,
    _optional_rate,
    _parse_args,
    _quartiles,
    _screen_continuations,
    anti_steerable_fraction,
    append_jsonl,
    claim_jsonl_artifact,
    compliance_gate,
    format_cell_table,
    item_steerability,
    labelled,
    load_fit,
    planned_behaviour_cells,
    read_jsonl,
    screen_layer_excess,
    stage_behaviour,
    stage_fit,
    stage_pick_cell,
    stage_report,
    stage_screen_layers,
    summarise_cells,
    top_layers_from_screen,
    write_summary,
)
from reward_hacking.interp.steer_validation import (
    ARM_ORTHOGONAL,
    ARM_PLACEBO,
    ARM_REAL,
    ARM_SHUFFLED,
    CHECK_BRANCHING,
    CHECK_CONTAMINATION,
    CHECK_DETERMINISM,
    CHECK_IN_PLACE,
    CORPUS_SUBJECTS,
    DEFAULT_RHOS,
    DEFAULT_SCORE_BATCH_ROWS,
    GENERATION_SEED_STRIDE,
    LANGUAGE_INSTRUCTIONS,
    LOGIT_BLOCK_BYTES,
    NON_THINKING_CAP,
    POSITION_ARMS,
    PROMPT_TEMPLATE,
    SCRIPT_HAN,
    SCRIPT_LATIN,
    SLOPE_GRID_RHOS,
    THINKING_CAP,
    BehaviouralRecord,
    GapRecord,
    HookInvocations,
    IsolationCheck,
    ResidualScale,
    ScoredContinuation,
    SteeringCell,
    _encode_scored_batch,
    _first_recurrent_state,
    _in_place_write_check,
    _logit_block_positions,
    _shuffled_items,
    _state_slots,
    alpha_from_relative_displacement,
    arm_family,
    assert_hook_pattern,
    build_prompts,
    control_directions,
    count_other_letters,
    count_script_chars,
    distinct_ngram_ratio,
    encode_scoring_batch,
    generation_position_mask,
    generation_seed,
    isolation_verdict,
    kl_divergence_rows,
    longest_token_run,
    masked_steering_hook,
    non_thinking_sampling,
    orthogonal_component_direction,
    repetition_rate,
    residual_scale_stats,
    run_behavioural_cell,
    score_teacher_forced_gap,
    script_score,
    sequence_logprobs,
    shuffled_label_direction,
    teacher_forced_position_mask,
    tensor_fingerprint,
    thinking_sampling,
    token_entropy,
)
from reward_hacking.interp.steering import Hook, residual_intervention

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path

    from reward_hacking.model_backend import SamplingConfig

# --------------------------------------------------------------------------------------
# Fakes: a hookable trunk and a position-local causal LM with hand-computable logprobs
# --------------------------------------------------------------------------------------


class _IdentityLayer(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _HookableTrunk(torch.nn.Module):
    """Minimal decoder trunk: ``.layers`` is what ``residual_intervention`` resolves and hooks."""

    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_IdentityLayer() for _ in range(n_layers)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _SteerableModel(torch.nn.Module):
    """Bare model tree with a hookable trunk at ``.model``, for the generation-time hook tests."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _HookableTrunk()


_VOCAB = 7
_GAIN = 2.0
_LOGSUM = math.log(math.exp(_GAIN) + (_VOCAB - 1))


def _fake_logprob(current: int, target: int) -> float:
    """The position-local fake's log-probability of ``target`` given the current token."""
    return (_GAIN if target == current else 0.0) - _LOGSUM


class _EmbeddingTrunk(torch.nn.Module):
    """Hookable trunk that embeds ids itself and yields ``last_hidden_state``, as a real trunk does.

    ``sequence_logprobs`` drives the trunk directly and applies the head itself, so a fake that only
    implements a whole-model ``forward`` cannot exercise that path at all: it would return finished
    logits and the position blocking would never run. Embedding is one-hot times the gain, which is
    what keeps :func:`_fake_logprob` hand-computable through the identity head below.

    A sibling of :class:`_HookableTrunk` rather than a subclass, because the two take different
    arguments -- ids here, an already-embedded residual there -- and one cannot substitute for the
    other.

    ``drop_last_position`` returns one position fewer than it was handed, which is the mismatch the
    chunk-mask guard and the trunk-length assertion both exist to catch.
    """

    def __init__(self, *, drop_last_position: bool = False) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_IdentityLayer() for _ in range(2)])
        self._drop_last_position = drop_last_position

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> SimpleNamespace:
        del attention_mask, kwargs
        hidden = torch.nn.functional.one_hot(input_ids, _VOCAB).float() * _GAIN
        if self._drop_last_position:
            hidden = hidden[:, :-1, :]
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _IdentityHead(torch.nn.Module):
    """LM head that passes its hidden states through, so the fake's logits stay hand-computable.

    ``weight`` is a registered parameter shaped ``[vocab, hidden]`` because ``sequence_logprobs``
    reads the vocabulary that sizes a logit block off ``head.get_parameter("weight")``, exactly as it
    would from a real ``Linear``. The head is the identity rather than a projection so that every
    hand-computed log-probability in this file stays hand-computable instead of becoming a matmul.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(_VOCAB), requires_grad=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _PositionLocalLM(torch.nn.Module):
    """Causal-LM fake whose logits depend ONLY on the current position's input token.

    ``logits[b, t] = gain * one_hot(input_ids[b, t])``: position t's row is decided by token t
    alone, which is what makes the right-padding batch-safety test exact -- trailing pads cannot
    reach any earlier position, so a row's masked score must equal its batch-of-one score to float
    precision. Split as a hookable trunk plus an identity head so the same fake serves both the
    steered teacher-forced path and ``sequence_logprobs``' trunk-then-blocked-head path; the whole
    ``forward`` stays for callers that drive the model rather than the trunk.
    """

    def __init__(self, *, drop_last_position: bool = False) -> None:
        super().__init__()
        self.model = _EmbeddingTrunk(drop_last_position=drop_last_position)
        self.lm_head = _IdentityHead()

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> SimpleNamespace:
        outputs = self.model(input_ids, attention_mask, **kwargs)
        return SimpleNamespace(logits=self.lm_head(outputs.last_hidden_state))


class _MappedTokenizer:
    """Chat template and encoder over an explicit text-to-ids table; pad id 0."""

    pad_token_id = 0

    def __init__(self, table: dict[str, list[int]]) -> None:
        self._table = table

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        del kwargs
        return f"<user>{messages[0]['content']}</user>"

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        del kwargs
        return {"input_ids": list(self._table[text])}


# --------------------------------------------------------------------------------------
# A. Deterministic scorers
# --------------------------------------------------------------------------------------


class TestScriptScore:
    def test_pure_english_scores_zero_with_its_denominator(self) -> None:
        score = script_score("The quick brown fox jumps over the lazy dog.", SCRIPT_HAN)
        assert score.ratio == 0.0
        assert score.target_chars == 0
        assert score.latin_chars > 0
        assert score.denominator > 0

    def test_pure_chinese_scores_one(self) -> None:
        score = script_score("水是生命之源", SCRIPT_HAN)
        assert score.ratio == 1.0
        assert score.target_chars == 6
        assert score.latin_chars == 0

    def test_mixed_string_scores_the_exact_fraction(self) -> None:
        """Two Latin letters and one Han ideograph: ratio = 1 / (1 + 2)."""
        score = script_score("AB水", SCRIPT_HAN)
        assert score.target_chars == 1
        assert score.latin_chars == 2
        assert score.ratio == pytest.approx(1 / 3)

    def test_cjk_punctuation_does_not_read_as_a_script_switch(self) -> None:
        """U+3002 (the ideographic full stop) is deliberately outside the Han ranges.

        A CJK formatting tic in an otherwise English sentence must still score 0.0.
        """
        score = script_score("This sentence ends oddly。", SCRIPT_HAN)
        assert score.ratio == 0.0
        assert score.target_chars == 0

    def test_cjk_punctuation_alone_is_unscorable_not_zero(self) -> None:
        assert script_score("。", SCRIPT_HAN).ratio is None

    def test_no_letters_at_all_is_none_never_zero(self) -> None:
        """An empty/collapsed response and a fully-English one are different outcomes."""
        score = script_score("!!! ... 123", SCRIPT_HAN)
        assert score.ratio is None
        assert score.denominator == 0

    def test_other_script_letters_count_separately_and_inflate_nothing(self) -> None:
        """Greek alpha+beta and Devanagari OM are letters, but in no tracked script."""
        text = "αβ ॐ"
        assert count_other_letters(text) == 3
        score = script_score(text, SCRIPT_HAN)
        assert score.target_chars == 0
        assert score.latin_chars == 0
        assert score.other_letters == 3
        assert score.ratio is None

    def test_other_letters_ignores_the_tracked_scripts(self) -> None:
        """Latin and Han letters must not leak into the collapse counter (Greek alpha+beta do)."""
        greek_alpha_beta = chr(0x3B1) + chr(0x3B2)
        assert count_other_letters("water水" + greek_alpha_beta) == 2

    def test_unknown_script_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown script"):
            count_script_chars("abc", "klingon")


class TestDegeneracyScorers:
    def test_distinct_ngram_ratio_is_one_on_all_distinct_ids(self) -> None:
        assert distinct_ngram_ratio(list(range(10))) == 1.0

    def test_distinct_ngram_ratio_falls_to_one_over_total_on_a_locked_loop(self) -> None:
        """20 identical tokens make 18 trigrams, exactly 1 of them distinct."""
        assert distinct_ngram_ratio([7] * 20) == pytest.approx(1 / 18)

    def test_distinct_ngram_ratio_is_none_when_shorter_than_n(self) -> None:
        assert distinct_ngram_ratio([1, 2]) is None

    def test_repetition_rate_is_the_complement_of_the_distinct_ratio(self) -> None:
        assert repetition_rate(list(range(10))) == pytest.approx(0.0)
        assert repetition_rate([7] * 20) == pytest.approx(17 / 18)
        assert repetition_rate([1, 2]) is None

    def test_token_entropy_hand_values(self) -> None:
        assert token_entropy([]) is None
        assert token_entropy([3, 3, 3]) == pytest.approx(0.0)
        assert token_entropy([1, 2, 1, 2]) == pytest.approx(1.0)
        assert token_entropy([1, 1, 2, 2, 3, 3, 4, 4]) == pytest.approx(2.0)

    def test_longest_token_run(self) -> None:
        assert longest_token_run([1, 1, 1, 2, 3, 3]) == 3
        assert longest_token_run([]) == 0
        assert longest_token_run([4, 4, 4, 4]) == 4


# --------------------------------------------------------------------------------------
# B. Corpus
# --------------------------------------------------------------------------------------


class TestBuildPrompts:
    def test_the_corpus_clears_the_published_extraction_floor(self) -> None:
        """128 contrast pairs is the defensible published floor for a diff-of-means fit."""
        assert len(CORPUS_SUBJECTS) >= 128

    def test_same_seed_same_list(self) -> None:
        assert build_prompts(n=6, seed=3) == build_prompts(n=6, seed=3)

    def test_different_seed_is_a_permutation_not_a_different_draw(self) -> None:
        n = len(CORPUS_SUBJECTS)
        first = build_prompts(n=n, seed=0)
        second = build_prompts(n=n, seed=1)
        assert first != second
        assert set(first) == set(second)
        assert set(first) == {PROMPT_TEMPLATE.format(subject=s) for s in CORPUS_SUBJECTS}

    def test_more_than_the_corpus_raises(self) -> None:
        with pytest.raises(ValueError, match="corpus holds"):
            build_prompts(n=len(CORPUS_SUBJECTS) + 1, seed=0)

    def test_matched_pairs_cover_the_same_subjects_in_the_same_order(self) -> None:
        """The whole direction fit rests on this: index i of the han list is index i's twin.

        Both instructed lists are the plain prompt plus exactly the instruction, nothing else.
        """
        plain = build_prompts(n=8, seed=5)
        han = build_prompts(n=8, seed=5, instruction_script=SCRIPT_HAN)
        latin = build_prompts(n=8, seed=5, instruction_script=SCRIPT_LATIN)
        for plain_prompt, han_prompt, latin_prompt in zip(plain, han, latin, strict=True):
            assert han_prompt == f"{plain_prompt} {LANGUAGE_INSTRUCTIONS[SCRIPT_HAN]}"
            assert latin_prompt == f"{plain_prompt} {LANGUAGE_INSTRUCTIONS[SCRIPT_LATIN]}"


class TestShuffledItems:
    def test_covers_every_prompt_with_its_original_index(self) -> None:
        prompts = [f"prompt {i}" for i in range(12)]
        pairs = _shuffled_items(prompts, seed=0)
        assert sorted(index for index, _ in pairs) == list(range(12))
        for index, prompt in pairs:
            assert prompt == prompts[index]

    def test_order_is_seeded_and_actually_shuffles(self) -> None:
        prompts = [f"prompt {i}" for i in range(12)]
        assert _shuffled_items(prompts, seed=4) == _shuffled_items(prompts, seed=4)
        assert _shuffled_items(prompts, seed=0) != _shuffled_items(prompts, seed=1)


# --------------------------------------------------------------------------------------
# C. Strength parameterisation
# --------------------------------------------------------------------------------------


class TestResidualScaleStats:
    def test_every_convention_from_rows_of_known_norm(self) -> None:
        """Three dim-4 rows of constant value: norms 3, 4 and 5 exactly."""
        activations = torch.tensor([[1.5] * 4, [2.0] * 4, [2.5] * 4])
        stats = residual_scale_stats(activations, layer=5)
        assert stats.layer == 5
        assert stats.dim == 4
        assert stats.mean_l2 == pytest.approx(4.0)
        assert stats.median_l2 == pytest.approx(4.0)
        assert stats.rms_of_norms == pytest.approx(math.sqrt(50 / 3))
        assert stats.per_element_rms == pytest.approx(4.0 / 2.0)
        assert stats.sqrt_dim == pytest.approx(2.0)

    def test_batched_and_flat_captures_agree(self) -> None:
        torch.manual_seed(3)
        flat = torch.randn(6, 8)
        batched = flat.reshape(2, 3, 8)
        a = residual_scale_stats(flat, layer=0)
        b = residual_scale_stats(batched, layer=0)
        assert a.mean_l2 == pytest.approx(b.mean_l2)
        assert a.rms_of_norms == pytest.approx(b.rms_of_norms)
        assert a.dim == b.dim == 8


class TestAlphaFromRelativeDisplacement:
    @staticmethod
    def _stats(rms_of_norms: float) -> ResidualScale:
        """The mean L2 deliberately differs from the quadratic mean, so the test can tell them apart."""
        return ResidualScale(
            layer=0,
            dim=4,
            rms_of_norms=rms_of_norms,
            mean_l2=rms_of_norms + 1.0,
            median_l2=rms_of_norms,
            per_element_rms=(rms_of_norms + 1.0) / 2.0,
            sqrt_dim=2.0,
        )

    def test_alpha_is_rho_times_rms_of_norms_not_the_mean_l2(self) -> None:
        """Pins WHICH norm convention rho scales against: the quadratic mean of position norms.

        With mean_l2 = 4.0 beside rms_of_norms = 3.0, an implementation that quietly switched to
        the arithmetic mean would return 8.0 here instead of 6.0.
        """
        assert alpha_from_relative_displacement(2.0, self._stats(3.0)) == 6.0

    def test_a_signed_rho_gives_a_signed_alpha_with_no_separate_code_path(self) -> None:
        assert alpha_from_relative_displacement(-4.0, self._stats(3.0)) == -12.0

    def test_zero_or_negative_norm_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            alpha_from_relative_displacement(1.0, self._stats(0.0))
        with pytest.raises(ValueError, match="must be positive"):
            alpha_from_relative_displacement(1.0, self._stats(-1.0))


class TestRhoConventions:
    """Pins the arithmetic that made the prior sweep unreadable, so it cannot regress silently."""

    def test_per_element_rms_differs_from_the_norm_scale_by_sqrt_dim(self) -> None:
        """The factor-of-fifty trap at d=2560.

        A coefficient quoted against the per-ELEMENT rms is sqrt(d) ~ 50.6 away from one quoted
        against the per-position norm; confusing the conventions is exactly how a sweep lands two
        orders of magnitude off.
        """
        torch.manual_seed(0)
        stats = residual_scale_stats(torch.randn(5, 2560), layer=0)
        assert stats.sqrt_dim == pytest.approx(math.sqrt(2560))
        assert stats.sqrt_dim == pytest.approx(50.596, abs=0.01)
        assert stats.per_element_rms * stats.sqrt_dim == pytest.approx(stats.mean_l2)

    def test_default_rhos_contain_the_prior_runs_two_points_and_bracket_the_safe_band(self) -> None:
        """Why the prior two-point sweep could not be read, pinned in the new units.

        The old harness's scale IS rho, and it tried exactly 0.1 and 0.5: the floor of the
        published 0.1-0.4 safe band and just past its ceiling. The new ladder must contain both
        (so the old result is locatable on the new curve) and extend beyond the band both ways
        (so a non-monotone dose-response cannot hide off the end of the sweep).
        """
        assert 0.1 in DEFAULT_RHOS
        assert 0.5 in DEFAULT_RHOS
        assert min(rho for rho in DEFAULT_RHOS if rho > 0) < 0.1
        assert max(DEFAULT_RHOS) > 0.4
        assert min(DEFAULT_RHOS) < 0

    def test_the_slope_grid_is_symmetric_and_contained_in_the_default_ladder(self) -> None:
        assert set(SLOPE_GRID_RHOS) <= set(DEFAULT_RHOS)
        assert 0.0 in SLOPE_GRID_RHOS
        for rho in SLOPE_GRID_RHOS:
            assert -rho in SLOPE_GRID_RHOS


# --------------------------------------------------------------------------------------
# D. Position masks
# --------------------------------------------------------------------------------------


class TestGenerationPositionMask:
    def test_prefill_arm_covers_multi_position_passes_only(self) -> None:
        multi = generation_position_mask(5, "prefill")
        assert multi is not None
        assert multi.tolist() == [True] * 5
        assert generation_position_mask(1, "prefill") is None

    def test_decode_arm_covers_single_position_passes_only(self) -> None:
        assert generation_position_mask(5, "decode") is None
        single = generation_position_mask(1, "decode")
        assert single is not None
        assert single.tolist() == [True]

    def test_all_arm_covers_both(self) -> None:
        for seq_len in (1, 5):
            mask = generation_position_mask(seq_len, "all")
            assert mask is not None
            assert mask.tolist() == [True] * seq_len

    def test_unknown_arm_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown position arm"):
            generation_position_mask(5, "sideways")  # pyright: ignore[reportArgumentType]


class TestTeacherForcedPositionMask:
    def test_ragged_prompt_boundaries_split_each_row_at_its_own_length(self) -> None:
        prompt_lens = torch.tensor([3, 5])
        prefill = teacher_forced_position_mask(prompt_lens, 8, "prefill")
        decode = teacher_forced_position_mask(prompt_lens, 8, "decode")
        everything = teacher_forced_position_mask(prompt_lens, 8, "all")

        assert prefill.tolist() == [[True] * 3 + [False] * 5, [True] * 5 + [False] * 3]
        assert decode.tolist() == [[False] * 3 + [True] * 5, [False] * 5 + [True] * 3]
        assert everything.tolist() == [[True] * 8, [True] * 8]
        # Prefill XOR decode covers every position exactly once.
        assert torch.equal(prefill ^ decode, everything)
        assert not (prefill & decode).any()

    def test_unknown_arm_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown position arm"):
            teacher_forced_position_mask(torch.tensor([3]), 8, "sideways")  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------------------
# E. Hooks and invocation accounting
# --------------------------------------------------------------------------------------


def _generation_hook(
    arm: str, direction: torch.Tensor, alpha: float
) -> tuple[Hook, HookInvocations]:
    invocations = HookInvocations()
    hook = masked_steering_hook(
        direction,
        alpha,
        lambda seq_len: generation_position_mask(seq_len, arm),  # pyright: ignore[reportArgumentType]
        invocations,
    )
    return hook, invocations


class TestMaskedSteeringHook:
    """The generation-time steering hook, and the axis-alignment trap its tests must not fall into.

    Every steered-output check here asserts the SHAPE before any ``allclose``. Under the
    historical ``[seq, 1, 1]`` selector bug, ``torch.allclose`` broadcasts the mis-shaped
    ``[seq, seq, d]`` output against a ``[1, seq, d]`` expectation and PASSES, so an ``allclose``
    with no shape assertion does not check axis alignment at all. That ordering generalises: any
    tensor test in this repo that verifies a hook's output with ``allclose`` alone is blind to
    exactly this class of broadcast bug.
    """

    def test_all_arm_shifts_every_position_by_exactly_alpha_unit_direction(self) -> None:
        model = _SteerableModel()
        direction = torch.randn(8) * 4.0
        base = torch.randn(1, 5, 8)
        hook, invocations = _generation_hook("all", direction, alpha=2.5)

        with residual_intervention(model, 0, hook):  # pyright: ignore[reportArgumentType]
            out = model.model(base)

        # Shape first: a mis-broadcast selector balloons [1, 5, 8] to [5, 5, 8], allclose passes.
        assert out.shape == base.shape
        expected = 2.5 * unit(direction)
        assert torch.allclose(out - base, expected.expand_as(base), atol=1e-6)
        assert invocations.steered_positions == 5
        # residual_intervention removed the hook when the block ended.
        assert len(model.model.layers[0]._forward_hooks) == 0

    def test_prefill_arm_steers_the_prompt_pass_and_skips_decode_steps(self) -> None:
        model = _SteerableModel()
        direction = torch.randn(8)
        prompt_pass = torch.randn(1, 5, 8)
        decode_pass = torch.randn(1, 1, 8)
        hook, _ = _generation_hook("prefill", direction, alpha=3.0)

        with residual_intervention(model, 0, hook):  # pyright: ignore[reportArgumentType]
            steered = model.model(prompt_pass)
            untouched = model.model(decode_pass)

        assert steered.shape == prompt_pass.shape
        expected = 3.0 * unit(direction)
        assert torch.allclose(steered - prompt_pass, expected.expand_as(prompt_pass), atol=1e-6)
        # Bit-identical, not merely close: an untouched pass is what the arm comparison rests on.
        assert torch.equal(untouched, decode_pass)

    def test_decode_arm_leaves_a_prefill_shaped_pass_bit_identical(self) -> None:
        model = _SteerableModel()
        direction = torch.randn(8)
        prompt_pass = torch.randn(1, 5, 8)
        decode_pass = torch.randn(1, 1, 8)
        hook, _ = _generation_hook("decode", direction, alpha=3.0)

        with residual_intervention(model, 0, hook):  # pyright: ignore[reportArgumentType]
            untouched = model.model(prompt_pass)
            steered = model.model(decode_pass)

        assert torch.equal(untouched, prompt_pass)
        assert steered.shape == decode_pass.shape
        expected = 3.0 * unit(direction)
        assert torch.allclose(steered - decode_pass, expected.expand_as(decode_pass), atol=1e-6)

    def test_tuple_outputs_pass_trailing_elements_through(self) -> None:
        hook, _ = _generation_hook("all", torch.randn(8), alpha=1.0)
        base = torch.randn(1, 3, 8)

        out = hook(None, None, (base, "attn_cache"))

        assert isinstance(out, tuple)
        assert out[1] == "attn_cache"
        assert out[0].shape == base.shape

    def test_shape_guard_goes_red_on_a_mask_built_for_a_different_batch(self) -> None:
        """A [2, seq] mask against a [1, seq, d] pass broadcasts the batch axis to 2.

        The guard must refuse rather than hand a mis-shaped residual to the next layer.
        """
        invocations = HookInvocations()
        hook = masked_steering_hook(
            torch.randn(8),
            1.0,
            lambda seq_len: torch.ones(2, seq_len, dtype=torch.bool),
            invocations,
        )
        with pytest.raises(AssertionError, match="changed the residual's shape"):
            hook(None, None, torch.randn(1, 5, 8))


class TestHookInvocationAccounting:
    @pytest.mark.parametrize(
        ("arm", "expected_selected"), [("prefill", 5), ("decode", 3), ("all", 8)]
    )
    def test_one_prefill_and_three_decode_passes(self, arm: str, expected_selected: int) -> None:
        hook, invocations = _generation_hook(arm, torch.randn(8), alpha=1.5)
        hook(None, None, torch.randn(1, 5, 8))
        for _ in range(3):
            hook(None, None, torch.randn(1, 1, 8))

        assert invocations.calls == 4
        assert invocations.prefill_calls == 1
        assert invocations.decode_calls == 3
        assert invocations.selected_positions == expected_selected
        assert invocations.steered_positions == expected_selected

    def test_alpha_zero_selects_positions_but_steers_none(self) -> None:
        """The identity arm must run the same mask path (selected > 0) while moving nothing."""
        hook, invocations = _generation_hook("all", torch.randn(8), alpha=0.0)
        base = torch.randn(1, 5, 8)
        out = hook(None, None, base)

        assert torch.equal(out, base)  # pyright: ignore[reportArgumentType]
        assert invocations.selected_positions == 5
        assert invocations.steered_positions == 0


def _invocations(
    lengths: list[int], *, selected: int, steered: int | None = None
) -> HookInvocations:
    return HookInvocations(
        calls=len(lengths),
        sequence_lengths=list(lengths),
        selected_positions=selected,
        steered_positions=selected if steered is None else steered,
    )


class TestAssertHookPattern:
    """Each failure mode is driven to RED on the exact violation it exists to catch."""

    def test_red_on_zero_calls(self) -> None:
        with pytest.raises(AssertionError, match="never fired"):
            assert_hook_pattern(HookInvocations(), "all", expect_steer=True)

    def test_red_on_a_steered_arm_that_steered_nothing(self) -> None:
        with pytest.raises(AssertionError, match="unsteered baseline wearing a steered label"):
            assert_hook_pattern(
                _invocations([5, 1, 1], selected=8, steered=0), "all", expect_steer=True
            )

    def test_red_on_an_alpha_zero_arm_that_steered_something(self) -> None:
        with pytest.raises(AssertionError, match=r"it must .*steer none"):
            assert_hook_pattern(
                _invocations([5, 1], selected=6, steered=6), "all", expect_steer=False
            )

    def test_red_on_an_alpha_zero_arm_that_selected_nothing(self) -> None:
        with pytest.raises(AssertionError, match="not a matched baseline"):
            assert_hook_pattern(
                _invocations([5, 1], selected=0, steered=0), "all", expect_steer=False
            )

    def test_red_on_a_prefill_arm_that_saw_only_decode_passes(self) -> None:
        with pytest.raises(AssertionError, match="no multi-position pass"):
            assert_hook_pattern(_invocations([1, 1, 1], selected=3), "prefill", expect_steer=True)

    def test_red_on_a_decode_arm_that_saw_only_prefill_passes(self) -> None:
        with pytest.raises(AssertionError, match="no single-position pass"):
            assert_hook_pattern(_invocations([5], selected=5), "decode", expect_steer=True)

    def test_red_on_more_than_one_prefill_pass(self) -> None:
        with pytest.raises(AssertionError, match="prefill exactly once"):
            assert_hook_pattern(_invocations([5, 5, 1], selected=11), "all", expect_steer=True)

    @pytest.mark.parametrize(
        ("arm", "steered"),
        [("prefill", 4), ("decode", 2), ("all", 7)],
    )
    def test_red_on_a_steered_count_the_arm_cannot_explain(self, arm: str, steered: int) -> None:
        """prompt_len 5 with 3 decode passes implies exactly 5 / 3 / 8 steered positions."""
        with pytest.raises(AssertionError, match="mislabelled or the mask is wrong"):
            assert_hook_pattern(
                _invocations([5, 1, 1, 1], selected=steered, steered=steered),
                arm,  # pyright: ignore[reportArgumentType]
                expect_steer=True,
                prompt_len=5,
            )

    @pytest.mark.parametrize(
        ("arm", "steered"),
        [("prefill", 5), ("decode", 3), ("all", 8)],
    )
    def test_green_on_each_well_formed_pattern(self, arm: str, steered: int) -> None:
        assert_hook_pattern(
            _invocations([5, 1, 1, 1], selected=steered, steered=steered),
            arm,  # pyright: ignore[reportArgumentType]
            expect_steer=True,
            prompt_len=5,
        )

    def test_green_on_the_alpha_zero_identity_arm(self) -> None:
        assert_hook_pattern(
            _invocations([5, 1, 1], selected=7, steered=0), "all", expect_steer=False
        )


# --------------------------------------------------------------------------------------
# F. Controls
# --------------------------------------------------------------------------------------


class TestOrthogonalComponentDirection:
    def test_orthogonal_to_the_reference_at_its_exact_norm(self) -> None:
        generator = torch.Generator().manual_seed(0)
        reference = torch.randn(64, generator=generator) * 3.0
        ortho = orthogonal_component_direction(reference, generator)

        assert abs(cosine(ortho, reference)) < 1e-5
        assert float(ortho.norm()) == pytest.approx(float(reference.norm()), rel=1e-6)


class TestShuffledLabelDirection:
    def test_destroys_the_contrast_that_the_real_direction_carries(self) -> None:
        """The property that makes the shuffled-label arm a control.

        Well-separated clusters: the real difference of means has norm ~20, while a
        label-shuffled one only sees the sampling imbalance of a random split. Averaged over
        seeds so the margin is not a coin flip.
        """
        torch.manual_seed(0)
        axis = unit(torch.randn(16))
        positive = 10.0 * axis + 0.1 * torch.randn(32, 16)
        negative = -10.0 * axis + 0.1 * torch.randn(32, 16)
        real_norm = float(diff_of_means(positive, negative).norm())

        shuffled = [
            shuffled_label_direction(positive, negative, torch.Generator().manual_seed(seed))
            for seed in range(8)
        ]
        norms = [float(vec.norm()) for vec in shuffled]

        assert real_norm > 3 * (sum(norms) / len(norms))
        assert all(norm > 0 for norm in norms)
        assert not torch.equal(shuffled[0], shuffled[1])


class TestControlDirections:
    @staticmethod
    def _fit_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(1)
        direction = torch.randn(16) * 2.0
        positive = torch.randn(10, 16)
        negative = torch.randn(10, 16)
        return direction, positive, negative

    def test_full_arm_set_with_matched_norms_and_the_real_tensor_itself(self) -> None:
        direction, positive, negative = self._fit_data()
        arms = control_directions(
            direction, positive=positive, negative=negative, n_each=3, seed=11
        )

        assert len(arms) == 1 + 3 * 3
        assert arms[ARM_REAL] is direction
        reference_norm = float(direction.norm())
        for vector in arms.values():
            assert float(vector.norm()) == pytest.approx(reference_norm, rel=1e-5)

    def test_draws_within_a_family_differ(self) -> None:
        direction, positive, negative = self._fit_data()
        arms = control_directions(
            direction, positive=positive, negative=negative, n_each=3, seed=11
        )
        for family in (ARM_PLACEBO, ARM_SHUFFLED, ARM_ORTHOGONAL):
            draws = [arms[f"{family}_{index}"] for index in range(3)]
            assert not torch.equal(draws[0], draws[1])
            assert not torch.equal(draws[1], draws[2])

    def test_reproducible_per_seed_and_different_across_seeds(self) -> None:
        direction, positive, negative = self._fit_data()
        first = control_directions(
            direction, positive=positive, negative=negative, n_each=2, seed=11
        )
        again = control_directions(
            direction, positive=positive, negative=negative, n_each=2, seed=11
        )
        other = control_directions(
            direction, positive=positive, negative=negative, n_each=2, seed=12
        )
        for name in first:
            assert torch.equal(first[name], again[name])
        assert not torch.equal(first[f"{ARM_PLACEBO}_0"], other[f"{ARM_PLACEBO}_0"])


class TestArmFamily:
    def test_maps_each_arm_name_to_its_family(self) -> None:
        assert arm_family("real") == ARM_REAL
        assert arm_family("placebo_matched_norm_2") == ARM_PLACEBO
        assert arm_family("shuffled_label_0") == ARM_SHUFFLED
        assert arm_family("orthogonal_component_1") == ARM_ORTHOGONAL

    def test_unrecognised_arm_raises(self) -> None:
        with pytest.raises(ValueError, match="belongs to no known family"):
            arm_family("estimator_group_3")


# --------------------------------------------------------------------------------------
# G. Teacher-forced scoring
# --------------------------------------------------------------------------------------


class TestSequenceLogprobs:
    def test_only_masked_positions_contribute(self) -> None:
        model = _PositionLocalLM()
        ids = torch.tensor([[2, 3, 4, 5, 1]])
        attention = torch.ones_like(ids)
        narrow = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
        wide = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])

        narrow_total, _ = sequence_logprobs(model, ids, attention, narrow)  # pyright: ignore[reportArgumentType]
        wide_total, _ = sequence_logprobs(model, ids, attention, wide)  # pyright: ignore[reportArgumentType]

        # Position 4's contribution is predicted from position 3 (token 5 -> token 1: a miss).
        assert float(wide_total - narrow_total) == pytest.approx(_fake_logprob(5, 1), abs=1e-6)
        # And the narrow total itself is the hand-computed sum over positions 2 and 3.
        expected = _fake_logprob(3, 4) + _fake_logprob(4, 5)
        assert float(narrow_total[0]) == pytest.approx(expected, abs=1e-6)

    def test_right_padded_batch_rows_score_identically_to_batches_of_one(self) -> None:
        """The batch-safety invariant the whole cheap screen rests on.

        The linear-attention recurrence makes left padding poisonous, and right padding is only
        safe if trailing pads reach nothing the mask keeps. The fake's logits depend only on the
        current position's input, so equality here must be exact.
        """
        model = _PositionLocalLM()
        ids = torch.tensor([[2, 3, 4, 0, 0], [1, 2, 3, 4, 5]])
        attention = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        scored = torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]])

        batch_totals, _ = sequence_logprobs(model, ids, attention, scored)  # pyright: ignore[reportArgumentType]
        solo_short, _ = sequence_logprobs(
            model,  # pyright: ignore[reportArgumentType]
            ids[:1, :3],
            attention[:1, :3],
            scored[:1, :3],
        )
        solo_long, _ = sequence_logprobs(
            model,  # pyright: ignore[reportArgumentType]
            ids[1:, :],
            attention[1:, :],
            scored[1:, :],
        )

        assert float(batch_totals[0]) == pytest.approx(float(solo_short[0]), abs=1e-6)
        assert float(batch_totals[1]) == pytest.approx(float(solo_long[0]), abs=1e-6)

    def test_readout_logits_come_from_the_last_prompt_position(self) -> None:
        model = _PositionLocalLM()
        ids = torch.tensor([[2, 3, 4, 0, 0], [1, 2, 3, 4, 5]])
        attention = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        scored = torch.zeros_like(attention, dtype=torch.float32)

        _, at_prompt = sequence_logprobs(
            model,  # pyright: ignore[reportArgumentType]
            ids,
            attention,
            scored,
            prompt_lens=torch.tensor([2, 4]),
        )
        _, at_last_real = sequence_logprobs(model, ids, attention, scored)  # pyright: ignore[reportArgumentType]

        # With prompt_lens, row i reads position prompt_len - 1: gain * one_hot(that token).
        assert torch.allclose(
            at_prompt[0], _GAIN * torch.nn.functional.one_hot(torch.tensor(3), _VOCAB).float()
        )
        assert torch.allclose(
            at_prompt[1], _GAIN * torch.nn.functional.one_hot(torch.tensor(4), _VOCAB).float()
        )
        # No prompt_lens: the last REAL position per the attention mask, not the padded end.
        assert torch.allclose(
            at_last_real[0],
            _GAIN * torch.nn.functional.one_hot(torch.tensor(4), _VOCAB).float(),
        )


_FAMILY_VOCAB = 248_320
"""The Qwen3.5 family's vocabulary, as recorded in ``steer_validation``'s own sizing docstrings.

Named here because the block-sizing arithmetic is only meaningful at the real vocabulary: the fake
LM's seven tokens would make any budget look infinite.
"""

_SCREEN_PADDED_WIDTH = 350
"""Padded width the sizing docstrings quote for a screening batch, so a block must cover at least it."""


class TestLogitBlockSizing:
    """The bound that makes the thinking tier runnable, and the width that keeps the screen unchanged."""

    def test_a_default_screen_batch_fits_in_a_single_block(self) -> None:
        """The bit-identity claim, checked rather than asserted in prose.

        Within one block the scorer runs the same single ``log_softmax`` and single ``.sum`` it always
        ran, so a screening cell's numbers are unchanged only as long as its rows and padded width
        stay inside one block. If this ever goes red the screening artifacts have shifted in their
        last digits and are no longer poolable with the ones already on disk.
        """
        width = _logit_block_positions(
            batch=DEFAULT_SCORE_BATCH_ROWS, vocab=_FAMILY_VOCAB, block_bytes=LOGIT_BLOCK_BYTES
        )
        assert width >= _SCREEN_PADDED_WIDTH

    def test_the_thinking_tier_row_is_bounded_rather_than_allocated_whole(self) -> None:
        """One row of 65,536 positions is 60.6 GiB of float32 logits, which is the recorded crash."""
        unblocked_bytes = THINKING_CAP * _FAMILY_VOCAB * 4
        assert unblocked_bytes > 60 * 1024**3

        width = _logit_block_positions(batch=1, vocab=_FAMILY_VOCAB, block_bytes=LOGIT_BLOCK_BYTES)
        assert width < THINKING_CAP
        assert width * _FAMILY_VOCAB * 4 <= LOGIT_BLOCK_BYTES

    def test_more_rows_shrink_the_block_so_the_footprint_stays_put(self) -> None:
        """The row count is why bounding rows alone could never work: the product is what allocates."""
        narrow = _logit_block_positions(
            batch=64, vocab=_FAMILY_VOCAB, block_bytes=LOGIT_BLOCK_BYTES
        )
        wide = _logit_block_positions(batch=8, vocab=_FAMILY_VOCAB, block_bytes=LOGIT_BLOCK_BYTES)
        assert narrow < wide
        assert 64 * narrow * _FAMILY_VOCAB * 4 <= LOGIT_BLOCK_BYTES

    def test_a_budget_smaller_than_one_position_still_yields_one_position(self) -> None:
        assert _logit_block_positions(batch=8, vocab=_FAMILY_VOCAB, block_bytes=1) == 1


class TestSequenceLogprobsBlocking:
    """Blocking the position axis must change the numbers not at all, at any width.

    The bound exists because :func:`self_nll` scores one row holding the whole generation, so no row
    bound reaches it and the unblocked logits are 60.6 GiB at the thinking cap. A bound that moved
    the numbers would be a different measurement rather than the same one made affordably, so
    equality across widths is the contract -- and the readout row, which the KL coherence guard is
    read from, has to survive being carved out of whichever block happens to hold it.
    """

    @staticmethod
    def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two ragged right-padded rows whose readout positions land in different blocks at width 2."""
        ids = torch.tensor([[2, 3, 4, 5, 1, 6, 3], [1, 2, 3, 4, 5, 6, 0]])
        attention = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0]])
        scored = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]
        )
        prompt_lens = torch.tensor([2, 5])
        return ids, attention, scored, prompt_lens

    def test_every_width_agrees_with_one_unblocked_window(self) -> None:
        model = _PositionLocalLM()
        ids, attention, scored, prompt_lens = self._batch()
        seq_len = int(ids.shape[1])

        whole_totals, whole_readout = sequence_logprobs(
            model,  # pyright: ignore[reportArgumentType]
            ids,
            attention,
            scored,
            prompt_lens,
            block_positions=seq_len + 5,
        )

        for width in (1, 2, 3, 4, 6, 7):
            totals, readout = sequence_logprobs(
                model,  # pyright: ignore[reportArgumentType]
                ids,
                attention,
                scored,
                prompt_lens,
                block_positions=width,
            )
            assert torch.equal(totals, whole_totals), f"totals moved at {width=}"
            assert torch.equal(readout, whole_readout), f"readout row moved at {width=}"

    def test_the_readout_row_is_the_last_prompt_position_whatever_block_holds_it(self) -> None:
        """Row 1's readout sits at position 4, which is block 2 of 4 at width 2, not block 0."""
        model = _PositionLocalLM()
        ids, attention, scored, prompt_lens = self._batch()

        _, readout = sequence_logprobs(
            model,  # pyright: ignore[reportArgumentType]
            ids,
            attention,
            scored,
            prompt_lens,
            block_positions=2,
        )

        assert readout.shape == (2, _VOCAB)
        # prompt_len - 1, so row 0 reads position 1 (token 3) and row 1 position 4 (token 5).
        assert torch.allclose(
            readout[0], _GAIN * torch.nn.functional.one_hot(torch.tensor(3), _VOCAB).float()
        )
        assert torch.allclose(
            readout[1], _GAIN * torch.nn.functional.one_hot(torch.tensor(5), _VOCAB).float()
        )

    def test_the_unblocked_totals_are_still_the_hand_computed_sum(self) -> None:
        """Anchors the equivalence above to an absolute value, so all widths agreeing on a wrong
        number would still be caught."""
        model = _PositionLocalLM()
        ids, attention, scored, prompt_lens = self._batch()

        totals, _ = sequence_logprobs(
            model,  # pyright: ignore[reportArgumentType]
            ids,
            attention,
            scored,
            prompt_lens,
            block_positions=2,
        )

        # Row 0 scores positions 2..6: predicted from tokens 3,4,5,1,6 to tokens 4,5,1,6,3 -- all misses.
        expected_first = sum(
            _fake_logprob(current, target)
            for current, target in ((3, 4), (4, 5), (5, 1), (1, 6), (6, 3))
        )
        assert float(totals[0]) == pytest.approx(expected_first, abs=1e-5)
        # Row 1 scores positions 3..5: from tokens 3,4,5 to 4,5,6 -- misses too.
        expected_second = sum(
            _fake_logprob(current, target) for current, target in ((3, 4), (4, 5), (5, 6))
        )
        assert float(totals[1]) == pytest.approx(expected_second, abs=1e-5)

    def test_a_block_width_below_one_raises(self) -> None:
        model = _PositionLocalLM()
        ids, attention, scored, prompt_lens = self._batch()
        with pytest.raises(ValueError, match="at least 1"):
            sequence_logprobs(
                model,  # pyright: ignore[reportArgumentType]
                ids,
                attention,
                scored,
                prompt_lens,
                block_positions=0,
            )

    def test_an_out_of_range_readout_index_raises_instead_of_reading_the_padded_tail(self) -> None:
        """A zero prompt length used to index -1, silently reading the last padded position."""
        model = _PositionLocalLM()
        ids, attention, scored, _ = self._batch()
        with pytest.raises(ValueError, match="falls outside"):
            sequence_logprobs(
                model,  # pyright: ignore[reportArgumentType]
                ids,
                attention,
                scored,
                torch.tensor([0, 5]),
            )

    def test_a_trunk_returning_the_wrong_position_count_raises(self) -> None:
        """Hidden states and token ids that describe different sequences would gather wrong targets."""
        model = _PositionLocalLM(drop_last_position=True)
        ids, attention, scored, prompt_lens = self._batch()
        with pytest.raises(AssertionError, match="do not describe the same sequence"):
            sequence_logprobs(model, ids, attention, scored, prompt_lens)  # pyright: ignore[reportArgumentType]

    def test_a_model_with_no_output_head_says_so(self) -> None:
        model = _PositionLocalLM()
        model.get_output_embeddings = lambda: None  # pyright: ignore[reportAttributeAccessIssue]
        ids, attention, scored, prompt_lens = self._batch()
        with pytest.raises(AttributeError, match="returned None"):
            sequence_logprobs(model, ids, attention, scored, prompt_lens)  # pyright: ignore[reportArgumentType]


class TestEncodeScoredBatch:
    @staticmethod
    def _tokenizer() -> _MappedTokenizer:
        return _MappedTokenizer(
            {
                "<user>first prompt</user>": [3, 4, 5],
                "<user>second prompt</user>": [3, 4, 5, 6, 1],
                "aa": [2, 2],
                "b": [6],
            }
        )

    def test_masks_cover_exactly_the_continuations_and_real_tokens(self) -> None:
        ids, attention, scored, prompt_lens = _encode_scored_batch(
            self._tokenizer(),  # pyright: ignore[reportArgumentType]
            ["first prompt", "second prompt"],
            ["aa", "b"],
        )

        assert ids.tolist() == [[3, 4, 5, 2, 2, 0], [3, 4, 5, 6, 1, 6]]
        assert attention.tolist() == [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]
        assert scored.tolist() == [
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
        assert prompt_lens.tolist() == [3, 5]

    def test_a_pinned_width_pads_every_row_to_it(self) -> None:
        ids, attention, _, _ = _encode_scored_batch(
            self._tokenizer(),  # pyright: ignore[reportArgumentType]
            ["first prompt"],
            ["aa"],
            width=9,
        )
        assert ids.shape == (1, 9)
        assert ids.tolist() == [[3, 4, 5, 2, 2, 0, 0, 0, 0]]
        assert attention.tolist() == [[1, 1, 1, 1, 1, 0, 0, 0, 0]]

    def test_a_width_shorter_than_a_row_raises_instead_of_truncating(self) -> None:
        with pytest.raises(ValueError, match="must cover its longest row"):
            _encode_scored_batch(
                self._tokenizer(),  # pyright: ignore[reportArgumentType]
                ["second prompt"],
                ["aa"],
                width=4,
            )


class _CountingTokenizer(_MappedTokenizer):
    """Tokenizer that records every chat-template render and every encode call.

    The scorer used to re-tokenise its whole continuation set on every cell, twice, purely to derive
    a padded width that depends on no cell parameter. A counting fake is the only way to see that: the
    numbers it produced were correct either way, so nothing in the artifacts said how much work made
    them.
    """

    def __init__(self, table: dict[str, list[int]]) -> None:
        super().__init__(table)
        self.template_renders = 0
        self.encodes = 0

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.template_renders += 1
        return super().apply_chat_template(messages, **kwargs)

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        self.encodes += 1
        return super().__call__(text, **kwargs)


def _gap_tokenizer(prompts: list[str]) -> _CountingTokenizer:
    table: dict[str, list[int]] = {f"<user>{prompt}</user>": [5] for prompt in prompts}
    table["repeat"] = [5, 5]
    table["switch"] = [2, 4]
    return _CountingTokenizer(table)


def _gap_items(prompts: list[str]) -> list[ScoredContinuation]:
    return [
        ScoredContinuation(prompt=prompt, target_text="repeat", baseline_text="switch")
        for prompt in prompts
    ]


class TestEncodeScoringBatch:
    """The tokenisation is a stage-level input, so no cell can re-derive or vary it."""

    def test_the_whole_set_is_tokenised_once_per_stage_not_once_per_cell(self) -> None:
        """The efficiency claim, counted rather than asserted.

        Two passes per side to pin the width (encode at natural width, then re-encode at the maximum),
        and that is the whole cost for a sweep of any size. Under the old scorer the same work ran
        inside every cell, so a 32-layer by 3-strength by 10-arm screen paid it 960 times.
        """
        prompts = ["q one", "q two", "q three"]
        tokenizer = _gap_tokenizer(prompts)

        encoded = encode_scoring_batch(
            tokenizer,  # pyright: ignore[reportArgumentType]
            _gap_items(prompts),
            batch_rows=2,
        )

        # Four full-set encodings: target and baseline, at natural width then at the pinned width.
        assert tokenizer.template_renders == 4 * len(prompts)
        renders_and_continuations = 4 * len(prompts) + 4 * len(prompts)
        assert tokenizer.encodes == renders_and_continuations

        before = (tokenizer.template_renders, tokenizer.encodes)
        for arm_alpha in (0.0, 1.0, 2.0, 3.0):
            score_teacher_forced_gap(
                _PositionLocalLM(),  # pyright: ignore[reportArgumentType]
                encoded,
                SteeringCell(
                    layer=0, direction=torch.randn(_VOCAB), alpha=arm_alpha, position_arm="all"
                ),
            )
        assert (tokenizer.template_renders, tokenizer.encodes) == before

    def test_chunks_are_the_rows_they_would_have_been_encoded_as(self) -> None:
        """Right padding is per row, so slicing a wide encoding equals encoding the slice."""
        prompts = ["q one", "q two", "q three"]
        encoded = encode_scoring_batch(
            _gap_tokenizer(prompts),  # pyright: ignore[reportArgumentType]
            _gap_items(prompts),
            batch_rows=2,
        )

        chunks = list(encoded.chunks())

        assert [int(target.input_ids.shape[0]) for target, _ in chunks] == [2, 1]
        assert encoded.n_rows == 3
        assert encoded.padded_width == 3
        for target, baseline in chunks:
            assert int(target.input_ids.shape[1]) == encoded.padded_width
            assert int(baseline.input_ids.shape[1]) == encoded.padded_width

    def test_batch_rows_below_one_raises(self) -> None:
        prompts = ["q"]
        with pytest.raises(ValueError, match="at least 1"):
            encode_scoring_batch(
                _gap_tokenizer(prompts),  # pyright: ignore[reportArgumentType]
                _gap_items(prompts),
                batch_rows=0,
            )

    def test_an_empty_continuation_set_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one continuation"):
            encode_scoring_batch(_gap_tokenizer([]), [], batch_rows=2)  # pyright: ignore[reportArgumentType]


class TestScoreTeacherForcedGap:
    def test_gap_is_the_hand_computed_preference_and_chunking_changes_nothing(self) -> None:
        """Prompt token 5, target [5, 5] (two hits), baseline [2, 4] (two misses).

        Each hit beats each miss by exactly the gain, so the gap is 2 * gain per prompt.
        """
        prompts = ["q one", "q two", "q three"]
        model = _PositionLocalLM()
        encoded = encode_scoring_batch(
            _gap_tokenizer(prompts),  # pyright: ignore[reportArgumentType]
            _gap_items(prompts),
            batch_rows=2,
        )

        readout = score_teacher_forced_gap(
            model,  # pyright: ignore[reportArgumentType]
            encoded,
            SteeringCell(layer=0, direction=torch.randn(_VOCAB), alpha=0.0, position_arm="all"),
        )

        for gap in readout.gaps:
            assert gap == pytest.approx(2 * _GAIN, abs=1e-5)
        # Two chunks (2 rows + 1 row) times two continuation sets: four hooked forwards.
        assert readout.invocations.calls == 4
        # 3 positions per row over 6 rows, all selected; the alpha-zero cell steered none.
        assert readout.invocations.selected_positions == 18
        assert readout.invocations.steered_positions == 0
        assert readout.batch_rows == 2
        assert readout.padded_width == 3
        # The readout logits are the fake's row for the prompt token, one per prompt.
        assert readout.last_prompt_logits.shape == (3, _VOCAB)
        assert torch.allclose(
            readout.last_prompt_logits[0],
            _GAIN * torch.nn.functional.one_hot(torch.tensor(5), _VOCAB).float(),
        )

    def test_a_nonzero_layer_and_alpha_cell_matches_the_hand_computed_steered_gap(self) -> None:
        """A cell whose layer and alpha are BOTH non-default, so swapping the two fields moves the value.

        Every other cell in this suite uses ``layer=0, alpha=0.0``, under which transposing
        ``cell.layer`` and ``cell.alpha`` at the scorer's consumption point is invisible. Here the
        direction is the one-hot of the prompt/target token (index 5) scaled by 3 to exercise the
        hook's unit-normalisation, so steering adds exactly ``alpha`` to that token's logit at every
        position. Per prompt: two target transitions hit the boosted token
        (``(gain + alpha) - logsumexp``), the first baseline transition misses from the boosted
        prompt position, and the second misses from token 2, where the boost sits elsewhere in the
        softmax. ``alpha = 1.5`` on ``layer = 1`` so that neither an int-coerced nor a float-coerced
        swap of the two fields reproduces this number.
        """
        prompts = ["q one", "q two"]
        alpha = 1.5
        boosted_logsum = math.log(math.exp(_GAIN + alpha) + (_VOCAB - 1))
        off_boost_logsum = math.log(math.exp(_GAIN) + math.exp(alpha) + (_VOCAB - 2))
        target_logprob = 2 * ((_GAIN + alpha) - boosted_logsum)
        baseline_logprob = -boosted_logsum - off_boost_logsum
        expected_gap = target_logprob - baseline_logprob

        encoded = encode_scoring_batch(
            _gap_tokenizer(prompts),  # pyright: ignore[reportArgumentType]
            _gap_items(prompts),
            batch_rows=2,
        )
        readout = score_teacher_forced_gap(
            _PositionLocalLM(),  # pyright: ignore[reportArgumentType]
            encoded,
            SteeringCell(
                layer=1,
                direction=3.0 * torch.nn.functional.one_hot(torch.tensor(5), _VOCAB).float(),
                alpha=alpha,
                position_arm="all",
            ),
        )

        for gap in readout.gaps:
            assert gap == pytest.approx(expected_gap, abs=1e-5)
        # One chunk of two rows per continuation set, 3 positions per row, every one steered.
        assert readout.invocations.calls == 2
        assert readout.invocations.selected_positions == 12
        assert readout.invocations.steered_positions == 12
        # The steering reached the readout row too: the prompt token's logit carries the boost.
        assert torch.allclose(
            readout.last_prompt_logits[0],
            (_GAIN + alpha) * torch.nn.functional.one_hot(torch.tensor(5), _VOCAB).float(),
        )

    def test_the_mask_length_guard_goes_red_when_the_pass_does_not_match(self) -> None:
        """The trunk sees one position fewer than the encoded batch: the guard's exact target."""
        prompts = ["q"]
        encoded = encode_scoring_batch(
            _gap_tokenizer(prompts),  # pyright: ignore[reportArgumentType]
            _gap_items(prompts),
        )
        with pytest.raises(AssertionError, match="not the single forward this mask was built"):
            score_teacher_forced_gap(
                _PositionLocalLM(drop_last_position=True),  # pyright: ignore[reportArgumentType]
                encoded,
                SteeringCell(layer=0, direction=torch.randn(_VOCAB), alpha=0.0, position_arm="all"),
            )


class TestKlDivergenceRows:
    def test_zero_on_identical_rows_and_exact_on_a_hand_case(self) -> None:
        reference = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        same = kl_divergence_rows(reference, reference.clone())
        assert same == pytest.approx([0.0, 0.0], abs=1e-7)

        # KL(uniform || softmax([ln 3, 0])) = 0.5 ln(0.5/0.75) + 0.5 ln(0.5/0.25) = 0.5 ln(4/3).
        other = torch.tensor([[math.log(3.0), 0.0], [0.0, 0.0]])
        kls = kl_divergence_rows(reference, other)
        assert kls[0] == pytest.approx(0.5 * math.log(4 / 3), abs=1e-6)
        assert kls[1] == pytest.approx(0.0, abs=1e-7)
        assert all(kl >= 0 for kl in kls)

    def test_a_constant_logit_shift_is_invisible(self) -> None:
        """Softmax is shift-invariant, so adding a constant to every logit is not divergence."""
        reference = torch.tensor([[1.0, 2.0, 3.0]])
        shifted = reference + 7.5
        assert kl_divergence_rows(reference, shifted) == pytest.approx([0.0], abs=1e-6)


# --------------------------------------------------------------------------------------
# H. Readout
# --------------------------------------------------------------------------------------


def _behaviour_row(  # noqa: PLR0913 - a row is its cell coordinates plus its per-record scores
    *,
    layer: int = 12,
    arm: str = "real",
    position_arm: str = "all",
    rho: float = 4.0,
    value: float | None,
    self_nll: float | None = None,
    hit_cap: bool = False,
) -> dict[str, object]:
    return {
        "layer": layer,
        "arm": arm,
        "position_arm": position_arm,
        "rho": rho,
        "target_ratio": value,
        "self_nll": self_nll,
        "hit_token_cap": hit_cap,
        "distinct_trigram_ratio": None,
    }


class TestSummariseCells:
    def test_groups_by_cell_and_keeps_every_value_including_nones(self) -> None:
        bimodal = [0.9, 0.8, 0.85, -0.7, -0.75, None]
        symmetric = [0.5, -0.5, 0.3, -0.3]
        rows = [_behaviour_row(position_arm="all", value=v, hit_cap=v is None) for v in bimodal]
        rows += [_behaviour_row(position_arm="prefill", value=v) for v in symmetric]

        summaries = summarise_cells(rows, value_key="target_ratio", measure="script_ratio")
        assert len(summaries) == 2
        by_position = {cell.position_arm: cell for cell in summaries}

        cell = by_position["all"]
        assert cell.n_rows == 6
        assert cell.n_scored == 5
        assert cell.n_unscorable == 1  # the None stayed a None, never a zero
        assert cell.per_row == bimodal
        assert cell.arm_family == ARM_REAL
        assert cell.n_positive + cell.n_zero + cell.n_negative == cell.n_scored
        assert cell.cap_hit_rate == pytest.approx(1 / 6)

    def test_bimodal_sign_split_is_visible_not_averaged_away(self) -> None:
        rows = [_behaviour_row(value=v) for v in [0.9, 0.8, 0.85, -0.7, -0.75]]
        (cell,) = summarise_cells(rows, value_key="target_ratio", measure="script_ratio")

        assert cell.median == pytest.approx(0.8)  # not near zero
        assert cell.n_positive == 3
        assert cell.n_negative == 2
        assert cell.anti_fraction == pytest.approx(2 / 5)  # the minority share, by sign

    def test_a_symmetric_split_still_exposes_both_signs_at_a_zero_median(self) -> None:
        rows = [_behaviour_row(value=v) for v in [0.5, -0.5, 0.3, -0.3]]
        (cell,) = summarise_cells(rows, value_key="target_ratio", measure="script_ratio")

        assert cell.median == pytest.approx(0.0)
        assert cell.n_positive == 2
        assert cell.n_negative == 2
        assert cell.anti_fraction is None  # undefined at a zero median, not silently 0.0

    def test_median_self_nll_is_computed_over_the_present_values_only(self) -> None:
        rows = [
            _behaviour_row(value=0.1, self_nll=2.0),
            _behaviour_row(value=0.2, self_nll=None),
            _behaviour_row(value=0.3, self_nll=4.0),
        ]
        (cell,) = summarise_cells(rows, value_key="target_ratio", measure="script_ratio")
        assert cell.median_self_nll == pytest.approx(3.0)


class TestOptionalRate:
    """``None`` when the field is absent, a rate when it is present. Never a hard zero."""

    def test_a_rate_over_the_rows_that_carry_the_flag(self) -> None:
        assert _optional_rate([True, False, False, True]) == pytest.approx(0.5)
        assert _optional_rate([False, False]) == pytest.approx(0.0)

    def test_no_row_carrying_the_field_is_none_not_zero(self) -> None:
        assert _optional_rate([None, None, None]) is None
        assert _optional_rate([]) is None

    def test_absent_rows_do_not_dilute_the_denominator(self) -> None:
        """A file pooling both families rates the flag over the rows where it applies."""
        assert _optional_rate([True, None, None, None]) == pytest.approx(1.0)


class TestCapHitRateGrain:
    """The screening family has no token cap, so its cap-hit rate is unmeasured, not 0.00.

    ``summarise_cells`` serves both record families and only ``BehaviouralRecord`` declares
    ``hit_token_cap``; teacher-forced scoring generates nothing and so cannot hit a cap. Dividing a sum
    of absent flags by the row count printed a confident ``0.00`` for every screening cell, two lines
    below a docstring arguing against coercing ``None`` to zero.
    """

    def test_a_screening_family_group_reports_no_cap_hit_rate(self) -> None:
        rows: list[dict[str, object]] = [
            {"layer": 4, "arm": ARM_REAL, "position_arm": "all", "rho": 0.2, "gap_shift": value}
            for value in (0.3, -0.1, 0.4)
        ]

        (cell,) = summarise_cells(rows, value_key="gap_shift", measure="teacher_forced_gap_shift")

        assert cell.cap_hit_rate is None
        assert cell.n_rows == 3

    def test_a_behavioural_group_still_reports_its_measured_rate(self) -> None:
        rows = [_behaviour_row(value=0.5, hit_cap=hit) for hit in (True, False, False, False)]
        (cell,) = summarise_cells(rows, value_key="target_ratio", measure="script_ratio")
        assert cell.cap_hit_rate == pytest.approx(0.25)

    def test_the_row_grain_is_named_rather_than_read_as_items(self) -> None:
        """Two samples of one prompt are two ROWS in a cell, which ``n_rows`` says and ``n_items`` did not."""
        rows = [_behaviour_row(value=0.4), _behaviour_row(value=0.6)]
        (cell,) = summarise_cells(rows, value_key="target_ratio", measure="script_ratio")
        assert cell.n_rows == 2
        assert cell.per_row == [0.4, 0.6]


class TestQuartiles:
    def test_empty_is_all_none(self) -> None:
        assert _quartiles([]) == (None, None, None)

    def test_fewer_than_four_values_fall_back_to_the_range(self) -> None:
        assert _quartiles([3.0, 1.0, 2.0]) == (2.0, 1.0, 3.0)

    def test_four_or_more_values_get_real_quartiles(self) -> None:
        median, q1, q3 = _quartiles([1.0, 2.0, 3.0, 4.0])
        assert median == pytest.approx(2.5)
        assert q1 == pytest.approx(1.75)
        assert q3 == pytest.approx(3.25)


class TestFormatCellTable:
    def test_one_line_per_cell_and_none_fields_render(self) -> None:
        rows = [_behaviour_row(value=0.5), _behaviour_row(value=None, position_arm="prefill")]
        summaries = summarise_cells(rows, value_key="target_ratio", measure="script_ratio")

        table = format_cell_table(summaries)

        lines = table.splitlines()
        assert len(lines) == 2 + len(summaries)  # header, rule, one line per cell
        assert "layer" in lines[0]
        assert "--" in table  # the all-None cell rendered placeholders instead of crashing


class TestItemSteerability:
    @staticmethod
    def _rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for rho in (-1.0, 0.0, 1.0):
            rows.append({"prompt_index": 0, "rho": rho, "gap_shift": 2.0 * rho})
            rows.append({"prompt_index": 1, "rho": rho, "gap_shift": -rho})
        rows.append({"prompt_index": 2, "rho": 1.0, "gap_shift": None})  # unscorable: skipped
        return rows

    def test_per_item_slopes_and_anti_steerability(self) -> None:
        items = item_steerability(self._rows(), value_key="gap_shift")
        by_index = {item.prompt_index: item for item in items}

        assert set(by_index) == {0, 1}  # the all-None item contributes nothing
        steered = by_index[0]
        assert steered.n_points == 3
        assert steered.slope == pytest.approx(2.0)
        assert steered.anti_steerable is False
        assert steered.delta_at_max == pytest.approx(2.0)  # value at rho=1 minus value at rho=0
        assert steered.spearman == pytest.approx(1.0)

        inverted = by_index[1]
        assert inverted.slope == pytest.approx(-1.0)
        assert inverted.anti_steerable is True
        assert inverted.spearman == pytest.approx(-1.0)

    def test_anti_steerable_fraction(self) -> None:
        items = item_steerability(self._rows(), value_key="gap_shift")
        assert anti_steerable_fraction(items) == pytest.approx(0.5)
        assert anti_steerable_fraction([]) is None

    def test_a_degenerate_design_yields_no_slope(self) -> None:
        """One point, or a constant coefficient, is a degenerate design rather than a noisy one."""
        assert _ols_slope([1.0], [2.0]) is None
        assert _ols_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
        assert _ols_slope([0.0, 1.0, 2.0], [1.0, 3.0, 5.0]) == pytest.approx(2.0)


# --------------------------------------------------------------------------------------
# I. Artifact IO and stage handoff
# --------------------------------------------------------------------------------------


def _gap_record(
    layer: int, arm: str, gap_shift: float, prompt_index: int = 0, rho: float = 0.2
) -> GapRecord:
    return GapRecord(
        layer=layer,
        arm=arm,
        position_arm="all",
        rho=rho,
        alpha=rho * 30.0,
        prompt_index=prompt_index,
        gap=gap_shift,
        gap_shift=gap_shift,
        baseline_gap=0.0,
        target_logprob=-10.0,
        baseline_logprob=-10.5,
        kl_last_token=0.01,
        padded_width=64,
        score_batch_rows=8,
        hook_calls=2,
        hook_selected_positions=128,
        hook_steered_positions=128,
    )


class TestArtifactRoundTrip:
    def test_records_round_trip_with_non_ascii_text_intact(self, tmp_path: Path) -> None:
        path = tmp_path / "behaviour.jsonl"
        response = "水是生命之源。"
        behavioural = BehaviouralRecord(
            layer=9,
            arm="real",
            position_arm="prefill",
            rho=0.25,
            alpha=7.5,
            prompt_index=3,
            sample_index=0,
            thinking=False,
            greedy=True,
            prompt_text="explain a compass",
            response_text=response,
            response_tokens=6,
            hit_token_cap=False,
            elapsed_seconds=1.25,
            target_script=SCRIPT_HAN,
            target_ratio=1.0,
            target_chars=6,
            latin_chars=0,
            other_letters=0,
            ratio_denominator=6,
            specificity={"cyrillic": None},
            distinct_trigram_ratio=1.0,
            repetition_rate=0.0,
            token_entropy=2.3,
            longest_token_run=1,
            self_nll=1.5,
            hook_calls=4,
            hook_selected_positions=12,
            hook_steered_positions=12,
            hook_prefill_calls=1,
            hook_decode_calls=3,
        )

        assert append_jsonl(path, [_gap_record(3, "real", 0.1)]) == 1
        assert append_jsonl(path, [behavioural]) == 1  # appends, never truncates

        rows = read_jsonl(path)
        assert len(rows) == 2
        assert rows[0]["gap_shift"] == 0.1
        assert rows[0]["rho"] == 0.2
        assert rows[1]["response_text"] == response
        assert rows[1]["specificity"] == {"cyrillic": None}
        # ensure_ascii=False: the raw generation is on disk as UTF-8, not \u escapes.
        assert "水是生命之源" in path.read_text(encoding="utf-8")

    def test_a_truncated_final_line_is_dropped_and_the_rest_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n{"c": 3', encoding="utf-8")
        assert read_jsonl(path) == [{"a": 1}, {"b": 2}]

    def test_a_malformed_line_that_is_not_last_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.jsonl"
        path.write_text('{"a": 1}\nnot json\n{"c": 3}\n', encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            read_jsonl(path)

    def test_a_missing_path_reads_as_empty(self, tmp_path: Path) -> None:
        assert read_jsonl(tmp_path / "absent.jsonl") == []


class TestClaimJsonlArtifact:
    """A stage must refuse to append to a JSONL an earlier run already wrote.

    The defect: every stage opened its artifact in append mode with no refusal and no dedup key, so a
    retry into the same ``--out-dir`` doubled the apparent sample size at every coefficient, and a
    re-fit followed by a re-run pooled two different directions under one median. Retrying is the
    anticipated workflow here, which is exactly why nothing on a record could have told the two apart.
    """

    def test_an_absent_path_is_claimed_and_returned(self, tmp_path: Path) -> None:
        path = tmp_path / "screen_layers.jsonl"
        assert claim_jsonl_artifact(path) == path

    def test_an_existing_artifact_is_refused_and_the_escapes_are_named(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "screen_layers.jsonl"
        append_jsonl(path, [_gap_record(3, ARM_REAL, 0.1)])

        with pytest.raises(FileExistsError, match="already holds records") as raised:
            claim_jsonl_artifact(path)

        # The message has to carry the way out, since it is all an operator sees on a rented box.
        assert "--label" in str(raised.value)
        assert "--out-dir" in str(raised.value)

    def test_an_empty_file_left_by_a_killed_stage_is_still_refused(self, tmp_path: Path) -> None:
        """Fail closed: a zero-length artifact means a stage started here, which is the ambiguity."""
        path = tmp_path / "behaviour.jsonl"
        path.touch()
        with pytest.raises(FileExistsError):
            claim_jsonl_artifact(path)

    def test_a_fresh_label_writes_beside_the_existing_artifact(self, tmp_path: Path) -> None:
        append_jsonl(tmp_path / "screen_layers.jsonl", [_gap_record(3, ARM_REAL, 0.1)])
        assert claim_jsonl_artifact(tmp_path / labelled("screen_layers.jsonl", "unit2")).name == (
            "screen_layers_unit2.jsonl"
        )


class TestLabelled:
    def test_inserts_the_label_before_the_extension(self) -> None:
        assert labelled("behaviour.jsonl", "unit2") == "behaviour_unit2.jsonl"
        assert labelled("behaviour.jsonl", "") == "behaviour.jsonl"
        assert labelled("no_extension", "unit2") == "no_extension_unit2"


class TestLoadFit:
    def test_a_missing_summary_refuses_even_when_the_jsonl_exists(self, tmp_path: Path) -> None:
        """The summary is the completion marker; the JSONL only says the stage started."""
        (tmp_path / "fit_generations.jsonl").write_text('{"side": "target"}\n', encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="never finished"):
            load_fit(tmp_path)

    def test_an_ok_false_summary_refuses_with_the_recorded_reason(self, tmp_path: Path) -> None:
        summary_path = write_summary(
            tmp_path, "fit", {"ok": False, "failure": "compliance 0.2 below threshold"}
        )
        assert summary_path.name == "fit_summary.json"
        with pytest.raises(ValueError, match="ok=false"):
            load_fit(tmp_path)

    def test_a_summary_with_no_ok_field_refuses_too(self, tmp_path: Path) -> None:
        write_summary(tmp_path, "fit", {"stage": "fit"})
        with pytest.raises(ValueError, match="no reason given"):
            load_fit(tmp_path)


class TestScreenContinuations:
    def test_takes_a_prefix_and_refuses_an_empty_screen(self) -> None:
        items = [
            ScoredContinuation(prompt=f"p{i}", target_text="t", baseline_text="b") for i in range(4)
        ]
        fit = SimpleNamespace(continuations=items)
        assert _screen_continuations(fit, 2) == items[:2]  # pyright: ignore[reportArgumentType]
        assert _screen_continuations(fit, 99) == items  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match="at least one continuation"):
            _screen_continuations(fit, 0)  # pyright: ignore[reportArgumentType]


_CONTROL_DRAWS = (f"{ARM_PLACEBO}_0", f"{ARM_SHUFFLED}_0", f"{ARM_ORTHOGONAL}_0")


def _control_band(layer: int, shift: float, *, n: int = 4) -> list[GapRecord]:
    """One draw of each control family at ``layer``, every prompt at the same shift."""
    return [
        _gap_record(layer, arm, shift, prompt_index=index)
        for arm in _CONTROL_DRAWS
        for index in range(n)
    ]


class TestScreenLayerExcess:
    def test_pools_the_control_families_and_reports_each_one(self, tmp_path: Path) -> None:
        """Hand-computed: real median 4.0, per-family medians 1.0 / 2.0 / 3.0, pooled median 2.0.

        The pooled median is 2.0 rather than the largest family's 3.0, which is the point: with one
        draw per family, a per-family floor is a single draw and the maximum of three single draws
        rises with the controls' noise rather than with the real arm's effect.
        """
        rows = [
            _gap_record(6, ARM_REAL, shift, prompt_index=i) for i, shift in enumerate((3.0, 5.0))
        ]
        rows += [_gap_record(6, f"{ARM_PLACEBO}_0", 1.0, prompt_index=i) for i in range(2)]
        rows += [_gap_record(6, f"{ARM_SHUFFLED}_0", 2.0, prompt_index=i) for i in range(2)]
        rows += [_gap_record(6, f"{ARM_ORTHOGONAL}_0", 3.0, prompt_index=i) for i in range(2)]
        append_jsonl(tmp_path / "screen_layers.jsonl", rows)

        (row,) = screen_layer_excess(tmp_path)

        assert row.layer == 6
        assert row.real_median == pytest.approx(4.0)
        assert row.control_median == pytest.approx(2.0)
        assert row.control_median_by_family == {
            ARM_ORTHOGONAL: pytest.approx(3.0),
            ARM_PLACEBO: pytest.approx(1.0),
            ARM_SHUFFLED: pytest.approx(2.0),
        }
        assert row.excess == pytest.approx(2.0)
        assert row.n_real == 2
        assert row.n_control == 6

    def test_signs_are_folded_before_the_medians_on_both_sides(self, tmp_path: Path) -> None:
        """A shift's magnitude is the readout; a control that moved -5.0 moved as much as +5.0."""
        rows = [_gap_record(2, ARM_REAL, -6.0, prompt_index=i) for i in range(2)]
        rows += [_gap_record(2, f"{ARM_PLACEBO}_0", -5.0, prompt_index=i) for i in range(2)]
        append_jsonl(tmp_path / "screen_layers.jsonl", rows)

        (row,) = screen_layer_excess(tmp_path)
        assert row.real_median == pytest.approx(6.0)
        assert row.control_median == pytest.approx(5.0)
        assert row.excess == pytest.approx(1.0)

    def test_a_layer_with_no_control_rows_has_an_unknown_excess_not_a_zero_floor(
        self, tmp_path: Path
    ) -> None:
        append_jsonl(
            tmp_path / "screen_layers.jsonl",
            [_gap_record(1, ARM_REAL, 9.0, prompt_index=i) for i in range(3)],
        )
        (row,) = screen_layer_excess(tmp_path)
        assert row.real_median == pytest.approx(9.0)
        assert row.control_median is None
        assert row.excess is None
        assert row.n_control == 0

    def test_an_unknown_arm_name_raises_rather_than_being_bucketed(self, tmp_path: Path) -> None:
        append_jsonl(tmp_path / "screen_layers.jsonl", [_gap_record(1, "estimator_group_3", 1.0)])
        with pytest.raises(ValueError, match="belongs to no known family"):
            screen_layer_excess(tmp_path)


class TestTopLayersFromScreen:
    @staticmethod
    def _write_screen(tmp_path: Path, name: str = "screen_layers.jsonl") -> None:
        """Three layers, each with its own control band, spanning the three cases that matter.

        Layer 7's real arm carries the single largest shift in the file (60.0) at a median of 0.1
        against a band of 0.1, so its excess is 0.0; layer 3 clears its band by 4.2; layer 9 clears
        its band by 49.8 and is held out only by the late-layer cutoff.
        """
        rows = [
            _gap_record(7, ARM_REAL, shift, prompt_index=i)
            for i, shift in enumerate((0.1, -0.1, 0.1, 60.0))
        ]
        rows += [
            _gap_record(3, ARM_REAL, shift, prompt_index=i)
            for i, shift in enumerate((4.0, -4.5, 5.0, 4.2))
        ]
        rows += [
            _gap_record(9, ARM_REAL, shift, prompt_index=i)
            for i, shift in enumerate((50.0, 50.0, 50.0, 50.0))
        ]
        rows += _control_band(7, 0.1)
        rows += _control_band(3, 0.15)
        rows += _control_band(9, 0.2)
        append_jsonl(tmp_path / name, rows)

    def test_ranks_on_the_real_arms_median_not_its_outlier(self, tmp_path: Path) -> None:
        """Layer 7 owns the single largest |shift| (60.0) but its median is 0.1.

        Layer 3's consistent ~4.3 median must win: the whole reason the ranking is a median.
        """
        self._write_screen(tmp_path)
        assert top_layers_from_screen(tmp_path, k=1) == [3]
        assert top_layers_from_screen(tmp_path, k=2) == [3, 7]

    def test_a_layer_whose_controls_move_as_much_as_it_does_loses_to_a_quieter_one(
        self, tmp_path: Path
    ) -> None:
        """The defect this ranking replaced: a plain argmax over the real arm alone.

        Layer 2's real arm shifts ten times as far as layer 3's -- and so do its placebo,
        shuffled-label and orthogonal arms, which is the signature of a layer where any perturbation
        of that magnitude does this. Layer 3 moves less and its control band does not move at all, so
        it is the one worth the behavioural budget. Under the old argmax layer 2 won.
        """
        rows = [_gap_record(2, ARM_REAL, 40.0, prompt_index=i) for i in range(4)]
        rows += [_gap_record(3, ARM_REAL, 4.0, prompt_index=i) for i in range(4)]
        rows += _control_band(2, 39.0)
        rows += _control_band(3, 0.1)
        append_jsonl(tmp_path / "screen_layers.jsonl", rows)

        assert top_layers_from_screen(tmp_path, k=1) == [3]
        assert top_layers_from_screen(tmp_path, k=2) == [3, 2]

    def test_a_layer_below_its_control_band_is_still_returned_and_warned_about(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A screen where nothing cleared its placebo must say so, not die and not stay silent."""
        rows = [_gap_record(2, ARM_REAL, 1.0, prompt_index=i) for i in range(4)]
        rows += _control_band(2, 5.0)
        append_jsonl(tmp_path / "screen_layers.jsonl", rows)

        with caplog.at_level("WARNING"):
            assert top_layers_from_screen(tmp_path, k=1) == [2]

        assert "did not beat its own placebo" in caplog.text
        assert "layer 2" in caplog.text

    def test_a_layer_with_no_control_band_is_excluded_and_named(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unmeasured floor is unknown, not zero: ranking it against zero would always win."""
        rows = [_gap_record(1, ARM_REAL, 99.0, prompt_index=i) for i in range(4)]
        rows += [_gap_record(2, ARM_REAL, 4.0, prompt_index=i) for i in range(4)]
        rows += _control_band(2, 0.1)
        append_jsonl(tmp_path / "screen_layers.jsonl", rows)

        with caplog.at_level("WARNING"):
            assert top_layers_from_screen(tmp_path, k=5) == [2]

        assert "no control rows" in caplog.text
        assert "[1]" in caplog.text

    def test_a_screen_with_no_control_rows_at_all_selects_nothing(self, tmp_path: Path) -> None:
        append_jsonl(
            tmp_path / "screen_layers.jsonl",
            [_gap_record(1, ARM_REAL, 5.0, prompt_index=i) for i in range(4)],
        )
        assert top_layers_from_screen(tmp_path, k=3) == []

    def test_excludes_the_late_layer_band_unless_told_otherwise(self, tmp_path: Path) -> None:
        """Depth reads off the artifact as max layer + 1 = 10, so the 0.8 cutoff is 8.0.

        Layer 9 is ineligible despite carrying the largest excess -- unless late layers are
        explicitly allowed.
        """
        self._write_screen(tmp_path)
        assert 9 not in top_layers_from_screen(tmp_path, k=10)
        assert top_layers_from_screen(tmp_path, k=10, allow_late_layers=True) == [9, 3, 7]

    def test_falls_back_to_the_whole_range_when_every_layer_is_late(self, tmp_path: Path) -> None:
        rows = [_gap_record(9, ARM_REAL, 1.0, prompt_index=i) for i in range(3)]
        rows += _control_band(9, 0.1, n=3)
        append_jsonl(tmp_path / "screen_layers.jsonl", rows)
        assert top_layers_from_screen(tmp_path, k=2) == [9]

    def test_merges_every_labelled_screen_artifact(self, tmp_path: Path) -> None:
        self._write_screen(tmp_path)
        rows = [_gap_record(2, ARM_REAL, 30.0, prompt_index=i) for i in range(4)]
        rows += _control_band(2, 0.1)
        append_jsonl(tmp_path / "screen_layers_unit2.jsonl", rows)
        assert top_layers_from_screen(tmp_path, k=2) == [2, 3]

    def test_an_empty_out_dir_yields_no_layers(self, tmp_path: Path) -> None:
        assert top_layers_from_screen(tmp_path, k=3) == []


# --------------------------------------------------------------------------------------
# J. Sampler contract
# --------------------------------------------------------------------------------------


class TestSamplerContract:
    def test_thinking_sampling_keeps_the_full_budget_and_identity_penalties(self) -> None:
        """Hard repo rule: thinking tokens are never capped to bound runtime.

        Runtime is bounded by deadlines and cell counts, so this cap stays the model's own budget.
        """
        sampling = thinking_sampling()
        assert THINKING_CAP == 65536
        assert sampling.max_new_tokens == 65536
        assert sampling.min_p == 0.0
        assert sampling.repetition_penalty == 1.0
        assert sampling.presence_penalty == 0.0
        assert sampling.do_sample is True  # greedy inside a thinking block is the loop failure

    def test_non_thinking_sampling_greedy_switch_and_identity_penalties(self) -> None:
        greedy = non_thinking_sampling(greedy=True)
        sampled = non_thinking_sampling(greedy=False)
        assert greedy.do_sample is False
        assert sampled.do_sample is True
        for config in (greedy, sampled):
            assert config.max_new_tokens == NON_THINKING_CAP
            assert config.min_p == 0.0
            assert config.repetition_penalty == 1.0
            assert config.presence_penalty == 0.0


# --------------------------------------------------------------------------------------
# K. Isolation verdict
# --------------------------------------------------------------------------------------


def _checks(
    *, contamination: bool, determinism: bool, branching: bool | None = True
) -> list[IsolationCheck]:
    checks = [
        IsolationCheck(name=CHECK_DETERMINISM, passed=determinism, detail="synthetic"),
        IsolationCheck(name=CHECK_CONTAMINATION, passed=contamination, detail="synthetic"),
        IsolationCheck(name="in_place_state_write", passed=True, detail="extra check, ignored"),
    ]
    if branching is not None:
        checks.append(IsolationCheck(name=CHECK_BRANCHING, passed=branching, detail="synthetic"))
    return checks


class TestIsolationVerdict:
    def test_real_path_all_green_is_isolated(self) -> None:
        verdict = isolation_verdict(_checks(contamination=True, determinism=True), sabotage=False)
        assert verdict == ("isolated", True)

    def test_real_path_contamination_failure_dominates(self) -> None:
        assert isolation_verdict(
            _checks(contamination=False, determinism=True), sabotage=False
        ) == ("CONTAMINATED", False)
        assert isolation_verdict(
            _checks(contamination=False, determinism=False), sabotage=False
        ) == ("CONTAMINATED", False)

    def test_real_path_needs_the_determinism_floor(self) -> None:
        verdict = isolation_verdict(_checks(contamination=True, determinism=False), sabotage=False)
        assert verdict == ("NONDETERMINISTIC", False)

    def test_real_path_fails_when_the_built_in_sabotage_saw_no_corruption(self) -> None:
        """The branching check deliberately shares a cache and must observe corruption.

        A run where it did not has a blind instrument, so its clean verdict is worthless.
        """
        verdict = isolation_verdict(
            _checks(contamination=True, determinism=True, branching=False), sabotage=False
        )
        assert verdict == ("BLIND-NO-CORRUPTION-DETECTABLE", False)

    def test_real_path_tolerates_an_absent_branching_check(self) -> None:
        verdict = isolation_verdict(
            _checks(contamination=True, determinism=True, branching=None), sabotage=False
        )
        assert verdict == ("isolated", True)

    def test_sabotage_detected_is_the_passing_outcome_under_sabotage(self) -> None:
        assert isolation_verdict(_checks(contamination=False, determinism=True), sabotage=True) == (
            "sabotage-detected",
            True,
        )
        # Determinism is not required on the sabotage path: it answers one question only.
        assert isolation_verdict(
            _checks(contamination=False, determinism=False), sabotage=True
        ) == ("sabotage-detected", True)

    def test_a_green_sabotage_arm_is_reported_as_a_failure(self) -> None:
        """The load-bearing inversion.

        A sabotage arm that stays green means the contamination check cannot see the thing it
        exists to catch, so ok MUST be false.
        """
        verdict = isolation_verdict(_checks(contamination=True, determinism=True), sabotage=True)
        assert verdict == ("SABOTAGE-INVISIBLE", False)

    def test_a_missing_required_check_raises(self) -> None:
        only_determinism = [IsolationCheck(name=CHECK_DETERMINISM, passed=True, detail="only one")]
        with pytest.raises(ValueError, match="missing required checks"):
            isolation_verdict(only_determinism, sabotage=False)


class TestStateSlots:
    def test_reads_the_dict_layout_the_real_cache_uses_and_the_list_layout(self) -> None:
        assert _state_slots({0: False, 1: True, 2: True}) == [1, 2]
        assert _state_slots([True, False, True]) == [0, 2]
        assert _state_slots("neither layout") == []


class TestFirstRecurrentState:
    def test_returns_the_first_initialised_tensor_in_the_dict_layout(self) -> None:
        """transformers 5.15 keys states and flags by int in a dict.

        Iterating {0: True} yields keys, and treating those keys as flags is the silent-None bug
        _state_slots exists to prevent.
        """
        target = torch.randn(2, 3)
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(),  # lacks both attributes: skipped
                SimpleNamespace(recurrent_states={0: torch.zeros(1)}),  # no flag attr: skipped
                SimpleNamespace(
                    recurrent_states={0: torch.zeros(1), 1: target},
                    is_recurrent_states_initialized={0: False, 1: True},
                ),
            ]
        )
        assert _first_recurrent_state(cache) is target

    def test_returns_the_first_initialised_tensor_in_the_list_layout(self) -> None:
        target = torch.randn(2, 2)
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    recurrent_states=[torch.zeros(1), target],
                    is_recurrent_states_initialized=[False, True],
                )
            ]
        )
        assert _first_recurrent_state(cache) is target

    def test_none_when_nothing_is_initialised(self) -> None:
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    recurrent_states=[torch.zeros(1)],
                    is_recurrent_states_initialized=[False],
                )
            ]
        )
        assert _first_recurrent_state(cache) is None
        assert _first_recurrent_state(object()) is None  # no .layers at all

    def test_skips_an_initialised_slot_that_holds_no_tensor(self) -> None:
        target = torch.randn(1)
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    recurrent_states={0: "not a tensor"},
                    is_recurrent_states_initialized={0: True},
                ),
                SimpleNamespace(
                    recurrent_states={0: target},
                    is_recurrent_states_initialized={0: True},
                ),
            ]
        )
        assert _first_recurrent_state(cache) is target


class _InPlaceStateCache:
    """Recurrent cache in the dict layout transformers 5.15 uses, rewritten in place on each step.

    ``advance`` is the hazard itself, spelled the way ``update_recurrent_state`` spells it: a
    ``copy_`` into the tensor the cache already holds, so every alias of that tensor sees the new
    state and only a real copy escapes.
    """

    def __init__(self, state: torch.Tensor) -> None:
        self.layers = [
            SimpleNamespace(recurrent_states={0: state}, is_recurrent_states_initialized={0: True})
        ]

    def advance(self) -> None:
        held = self.layers[0].recurrent_states[0]
        held.copy_(held + 1.0)


class _ReallocatingStateCache(_InPlaceStateCache):
    """The same cache, except a step REPLACES the state tensor instead of writing into it.

    The counterfactual the in-place claim is measured against: here no alias can observe the new
    state, so ``in_place_state_write`` must report that nothing mutated.
    """

    def advance(self) -> None:
        held = self.layers[0].recurrent_states[0]
        self.layers[0].recurrent_states[0] = held + 1.0


class _NoStateCache(_InPlaceStateCache):
    """A cache carrying no recurrent-state layer at all: the layout-changed case."""

    def __init__(self) -> None:
        self.layers = []

    def advance(self) -> None:
        """Nothing to advance. A layout with no recurrent state is what the check reports on."""


class _AliasingCloneTensor(torch.Tensor):
    """A tensor whose ``.clone()`` hands back an alias of itself: the defect under test.

    Standing in for any future edit that snapshots a recurrent state without copying it. Under this
    tensor the snapshot follows the in-place write, which is precisely what a snapshot compared
    against itself could never notice.
    """

    def clone(self, *args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        return self


class _RecurrentStateModel(torch.nn.Module):
    """Model whose every cached forward advances the cache it was handed, and nothing else."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _HookableTrunk()

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: _InPlaceStateCache | None = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        del attention_mask, kwargs
        if past_key_values is not None:
            past_key_values.advance()
        batch, seq = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, seq, _VOCAB))


class _TensorTokenizer:
    """Chat template plus a tensor-returning encoder, which is what ``_prefill_into`` needs."""

    pad_token_id = 0

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        del kwargs
        return f"<user>{messages[0]['content']}</user>"

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        del text, kwargs
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }


class TestInPlaceWriteCheck:
    """The two claims ``in_place_state_write`` makes, each driven red on its own defect.

    The term this replaced was ``torch.equal(snapshot, snapshot)`` -- true for every non-NaN tensor,
    so the reported "while the clone held" was a sentence rather than a measurement and the gate
    reduced to its alias test. These tests exist because the missing half is only visible when a
    snapshot actually aliases the state, which is the third case below.
    """

    @staticmethod
    def _run(cache: _InPlaceStateCache, monkeypatch: pytest.MonkeyPatch) -> IsolationCheck:
        monkeypatch.setattr(steer_validation_module, "_fresh_cache", lambda model: cache)
        return _in_place_write_check(
            _RecurrentStateModel(),  # pyright: ignore[reportArgumentType]
            _TensorTokenizer(),  # pyright: ignore[reportArgumentType]
            "a compass",
        )

    @staticmethod
    def _state() -> torch.Tensor:
        return torch.arange(6.0).reshape(2, 3)

    def test_green_when_the_write_lands_in_place_and_the_clone_escapes_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        check = self._run(_InPlaceStateCache(self._state()), monkeypatch)

        assert check.name == CHECK_IN_PLACE
        assert check.passed is True
        assert "MUTATED" in check.detail
        assert "The clone's own fingerprint held" in check.detail

    def test_red_when_a_step_reallocates_instead_of_writing_in_place(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No alias can see a reallocated state, so the in-place claim must fail, not the clone one."""
        check = self._run(_ReallocatingStateCache(self._state()), monkeypatch)

        assert check.passed is False
        assert "did NOT mutate" in check.detail
        assert "The clone's own fingerprint held" in check.detail

    def test_red_when_the_snapshot_aliases_the_state_instead_of_copying_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The isolating defect: the write IS in place and the snapshot follows it anyway.

        Both halves of the detail are asserted, not just ``passed``: under the vacuous
        self-comparison this check used to make, the alias test alone would report "did NOT mutate"
        and the clone would never be described as having followed the write, so a bare
        ``passed is False`` would have been green against the broken code too.
        """
        aliasing = self._state().as_subclass(_AliasingCloneTensor)

        check = self._run(_InPlaceStateCache(aliasing), monkeypatch)

        assert check.passed is False
        assert "MUTATED" in check.detail
        assert "The clone's own fingerprint MOVED" in check.detail
        assert "FOLLOWED the write" in check.detail

    def test_a_cache_with_no_recurrent_state_reports_the_layout_change(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        check = self._run(_NoStateCache(), monkeypatch)

        assert check.passed is False
        assert "layout changed" in check.detail


class TestTensorFingerprint:
    def test_two_reductions_read_out_as_plain_floats(self) -> None:
        assert tensor_fingerprint(torch.tensor([1.0, 2.0, 3.0])) == (6.0, 14.0)

    def test_a_fingerprint_cannot_track_a_later_in_place_write(self) -> None:
        """Why the check reads values rather than tensors: a float is not a view of anything."""
        state = torch.tensor([1.0, 2.0, 3.0])
        before = tensor_fingerprint(state)
        state.copy_(state + 1.0)
        assert tensor_fingerprint(state) != before
        assert before == (6.0, 14.0)

    def test_a_sum_preserving_permutation_still_moves_the_fingerprint(self) -> None:
        """One reduction would miss this; the sum of squares is why there are two."""
        assert tensor_fingerprint(torch.tensor([0.0, 4.0])) != tensor_fingerprint(
            torch.tensor([2.0, 2.0])
        )


# --------------------------------------------------------------------------------------
# L. The fit stage's per-side compliance gate
# --------------------------------------------------------------------------------------

_HAN_ANSWER = "水是生命之源"
_LATIN_ANSWER = "water is the source of life"
_FIT_LAYERS = (0, 1)
_FIT_DIM = 4


def _fit_generation(*, side: str, complied: bool, prompt_index: int = 0) -> FitGeneration:
    return FitGeneration(
        prompt_index=prompt_index,
        side=side,
        prompt_text=f"question {prompt_index}",
        response_text=_HAN_ANSWER if complied else _LATIN_ANSWER,
        response_tokens=6,
        hit_token_cap=False,
        target_ratio=1.0 if complied else 0.0,
        complied=complied,
    )


def _sides(
    *, n_target_complied: int, n_baseline_complied: int, n: int = 8
) -> tuple[list[FitGeneration], list[FitGeneration]]:
    target = [
        _fit_generation(side=SIDE_TARGET, complied=index < n_target_complied, prompt_index=index)
        for index in range(n)
    ]
    baseline = [
        _fit_generation(
            side=SIDE_BASELINE, complied=index < n_baseline_complied, prompt_index=index
        )
        for index in range(n)
    ]
    return target, baseline


class TestComplianceGate:
    """The gate that could not fire, and the shape that lets it.

    Pooled over both sides, an unprompted-English baseline contributes about half the total on its
    own, so the pooled rate sits at or above the default 0.5 threshold whatever the target side did.
    """

    def test_a_fit_whose_target_side_never_complied_is_refused(self) -> None:
        """0 of 8 target against 8 of 8 baseline pools to exactly 0.500 and used to pass at 0.5."""
        target, baseline = _sides(n_target_complied=0, n_baseline_complied=8)

        gate = compliance_gate(target, baseline, min_compliance=0.5)

        assert gate.pooled_rate == pytest.approx(0.5)
        assert gate.target_rate == pytest.approx(0.0)
        assert gate.baseline_rate == pytest.approx(1.0)
        assert gate.weakest_side == SIDE_TARGET
        assert gate.weakest_rate == pytest.approx(0.0)
        assert gate.ok is False
        assert gate.failure is not None
        assert SIDE_TARGET in gate.failure

    def test_a_fit_whose_baseline_side_never_complied_is_refused_too(self) -> None:
        """Symmetric: a model answering the plain question in Chinese destroys the contrast as well."""
        target, baseline = _sides(n_target_complied=8, n_baseline_complied=0)

        gate = compliance_gate(target, baseline, min_compliance=0.5)

        assert gate.pooled_rate == pytest.approx(0.5)
        assert gate.weakest_side == SIDE_BASELINE
        assert gate.ok is False
        assert gate.failure is not None
        assert SIDE_BASELINE in gate.failure

    def test_both_sides_clear_the_threshold_and_the_counts_are_recorded(self) -> None:
        target, baseline = _sides(n_target_complied=7, n_baseline_complied=6)

        gate = compliance_gate(target, baseline, min_compliance=0.5)

        assert gate.ok is True
        assert gate.failure is None
        assert (gate.n_target, gate.n_target_complied) == (8, 7)
        assert (gate.n_baseline, gate.n_baseline_complied) == (8, 6)
        assert gate.weakest_side == SIDE_BASELINE
        assert gate.weakest_rate == pytest.approx(0.75)

    def test_the_threshold_binds_at_equality_rather_than_below_it(self) -> None:
        target, baseline = _sides(n_target_complied=4, n_baseline_complied=8)
        assert compliance_gate(target, baseline, min_compliance=0.5).ok is True
        assert compliance_gate(target, baseline, min_compliance=0.51).ok is False

    def test_uneven_side_sizes_are_scored_against_their_own_denominators(self) -> None:
        """Not one pooled division: two sides of different n each get their own rate."""
        target = [
            _fit_generation(side=SIDE_TARGET, complied=True, prompt_index=i) for i in range(2)
        ]
        baseline = [
            _fit_generation(side=SIDE_BASELINE, complied=index < 3, prompt_index=index)
            for index in range(6)
        ]

        gate = compliance_gate(target, baseline, min_compliance=0.5)

        assert gate.target_rate == pytest.approx(1.0)
        assert gate.baseline_rate == pytest.approx(0.5)
        assert gate.pooled_rate == pytest.approx(5 / 8)
        assert gate.ok is True

    def test_an_empty_side_raises_rather_than_scoring_zero_compliance(self) -> None:
        target, baseline = _sides(n_target_complied=4, n_baseline_complied=4, n=2)
        with pytest.raises(ValueError, match="broken generation loop"):
            compliance_gate([], baseline, min_compliance=0.5)
        with pytest.raises(ValueError, match="broken generation loop"):
            compliance_gate(target, [], min_compliance=0.5)


def _fit_record(prompt: str, response_text: str, sampling: SamplingConfig) -> GenerationRecord:
    prompt_ids = [1, 2, 3]
    response_ids = [4, 5]
    full = torch.tensor(prompt_ids + response_ids)
    return GenerationRecord(
        prompt_text=prompt,
        prompt_len=len(prompt_ids),
        response_text=response_text,
        full_ids=full,
        token_strings=[str(value) for value in full.tolist()],
        is_response=torch.tensor([False] * len(prompt_ids) + [True] * len(response_ids)),
        hit_token_cap=False,
        sampler=resolved_sampler(sampling),
    )


def _install_fit_stubs(
    monkeypatch: pytest.MonkeyPatch, *, target_in_han: bool, baseline_in_han: bool
) -> None:
    """Replace the fit stage's model, generation and capture so it runs on CPU with no checkpoint."""
    han_instruction = LANGUAGE_INSTRUCTIONS[SCRIPT_HAN]

    def fake_generate(
        model: object,
        tokenizer: object,
        prompt: str,
        *,
        thinking: bool,
        sampling: SamplingConfig,
    ) -> GenerationRecord:
        del model, tokenizer, thinking
        in_han = target_in_han if prompt.endswith(han_instruction) else baseline_in_han
        return _fit_record(prompt, _HAN_ANSWER if in_han else _LATIN_ANSWER, sampling)

    def fake_capture(model: object, record: GenerationRecord) -> dict[int, torch.Tensor]:
        """Random activations seeded on the prompt's BYTES, never its length.

        The three LANGUAGE_INSTRUCTIONS are all exactly 17 characters, so a length seed hands the
        target and baseline sides of a pair byte-identical activations and every fitted direction
        is identically zero -- degenerate data the fit-stage tests would silently pass on.
        """
        del model
        generator = torch.Generator().manual_seed(zlib.crc32(record.prompt_text.encode()))
        return {
            layer: torch.randn(record.seq_len, _FIT_DIM, generator=generator)
            for layer in _FIT_LAYERS
        }

    monkeypatch.setattr(
        "reward_hacking.interp.run_steer_validation.load_model_and_tokenizer",
        lambda model_id: (model_id, object()),
    )
    monkeypatch.setattr(
        "reward_hacking.interp.run_steer_validation.generate_response", fake_generate
    )
    monkeypatch.setattr(
        "reward_hacking.interp.run_steer_validation.capture_record_activations", fake_capture
    )


def _fit_args(tmp_path: Path) -> argparse.Namespace:
    return _parse_args(
        ["fit", "--out-dir", str(tmp_path), "--n-prompts", "4", "--min-compliance", "0.5"]
    )


def _fit_summary(tmp_path: Path) -> dict[str, object]:
    return json.loads((tmp_path / "fit_summary.json").read_text(encoding="utf-8"))


class TestStageFitCompliance:
    """The fit stage end to end on stubs: does the gate's verdict actually stop the direction.

    No model is loaded. ``load_model_and_tokenizer``, ``generate_response`` and
    ``capture_record_activations`` are replaced, so what is under test is the stage's own arithmetic
    and its refusal path -- including that a refused fit writes no ``directions.pt`` for a later stage
    to pick up.
    """

    def test_a_zero_target_compliance_fit_is_refused_and_writes_no_direction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The measurement-corrupting case: the model ignored the Chinese instruction entirely.

        Both sides answered in English, so the difference of means is between two English answer
        sets. The pooled rate is 0.500 and would have passed; the stage must exit non-zero and leave
        no ``directions.pt`` behind.
        """
        _install_fit_stubs(monkeypatch, target_in_han=False, baseline_in_han=False)

        assert stage_fit(_fit_args(tmp_path)) == 1

        summary = _fit_summary(tmp_path)
        gate = summary["compliance_gate"]
        assert isinstance(gate, dict)
        assert summary["ok"] is False
        assert summary["instruction_compliance_pooled"] == pytest.approx(0.5)
        assert gate["target_rate"] == pytest.approx(0.0)
        assert gate["baseline_rate"] == pytest.approx(1.0)
        assert gate["weakest_side"] == SIDE_TARGET
        assert not (tmp_path / "directions.pt").exists()
        with pytest.raises(ValueError, match="ok=false"):
            load_fit(tmp_path)

    def test_a_compliant_fit_passes_and_writes_the_whole_artifact_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fit_stubs(monkeypatch, target_in_han=True, baseline_in_han=False)

        assert stage_fit(_fit_args(tmp_path)) == 0

        summary = _fit_summary(tmp_path)
        gate = summary["compliance_gate"]
        assert isinstance(gate, dict)
        assert summary["ok"] is True
        assert gate["target_rate"] == pytest.approx(1.0)
        assert gate["baseline_rate"] == pytest.approx(1.0)
        assert len(read_jsonl(tmp_path / "fit_generations.jsonl")) == 8
        fit = load_fit(tmp_path)
        assert sorted(fit.directions) == list(_FIT_LAYERS)
        assert len(fit.continuations) == 4

    def test_a_labelled_fit_writes_beside_the_unlabelled_one_not_over_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The smoke runs a refused sabotage fit and the real fit in ONE out-dir, split by --label.

        Before the label was threaded through the fit stage, the sabotage fit claimed the
        unlabelled ``fit_generations.jsonl`` (so the real fit refused with ``FileExistsError``) and
        wrote its ``ok=false`` marker to the unlabelled ``fit_summary.json`` (so a labelled unit
        could stomp the completion marker every later stage loads through :func:`load_fit`).
        """
        _install_fit_stubs(monkeypatch, target_in_han=True, baseline_in_han=False)
        sabotage_args = _parse_args(
            [
                "fit",
                "--out-dir",
                str(tmp_path),
                "--n-prompts",
                "4",
                "--min-compliance",
                "1.01",
                "--label",
                "sabotage",
            ]
        )
        assert stage_fit(sabotage_args) == 1
        assert not (tmp_path / "fit_summary.json").exists()
        sabotage_summary = json.loads(
            (tmp_path / "fit_summary_sabotage.json").read_text(encoding="utf-8")
        )
        assert sabotage_summary["ok"] is False

        assert stage_fit(_fit_args(tmp_path)) == 0
        assert len(read_jsonl(tmp_path / "fit_generations.jsonl")) == 8
        assert len(read_jsonl(tmp_path / "fit_generations_sabotage.jsonl")) == 8
        assert _fit_summary(tmp_path)["ok"] is True
        assert load_fit(tmp_path).continuations


class TestStageRetryRefusal:
    """A stage run twice into one out-dir must refuse, and refuse BEFORE loading a model.

    Uses the fit-stage stubs because the fit is what puts on disk the artifacts a screening stage
    needs; what is under test is the retry refusal on the real stage functions rather than the helper
    in isolation.
    """

    def test_running_the_fit_stage_twice_into_one_out_dir_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the refusal the second run doubles fit_generations.jsonl behind an unchanged summary."""
        _install_fit_stubs(monkeypatch, target_in_han=True, baseline_in_han=False)

        assert stage_fit(_fit_args(tmp_path)) == 0
        assert len(read_jsonl(tmp_path / "fit_generations.jsonl")) == 8

        with pytest.raises(FileExistsError, match=r"fit_generations\.jsonl"):
            stage_fit(_fit_args(tmp_path))
        # The refusal left the first run's artifact exactly as it was.
        assert len(read_jsonl(tmp_path / "fit_generations.jsonl")) == 8

    def test_a_screening_stage_refuses_before_it_loads_a_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ordering matters on metered hardware: a refusal after the weight load wastes the load."""
        _install_fit_stubs(monkeypatch, target_in_han=True, baseline_in_han=False)
        assert stage_fit(_fit_args(tmp_path)) == 0
        append_jsonl(tmp_path / "screen_layers.jsonl", [_gap_record(0, ARM_REAL, 0.1)])

        def refuse_to_load(model_id: str) -> tuple[object, object]:
            del model_id
            raise AssertionError("the model was loaded before the artifact was claimed")

        monkeypatch.setattr(
            "reward_hacking.interp.run_steer_validation.load_model_and_tokenizer", refuse_to_load
        )
        args = _parse_args(["screen-layers", "--out-dir", str(tmp_path), "--n-screen-prompts", "2"])

        with pytest.raises(FileExistsError, match=r"screen_layers\.jsonl"):
            stage_screen_layers(args)


class TestStageReportCompletionMarker:
    """The report stage writes its summary last, like every other stage.

    It was the one exception to the invariant its own module docstring states in bold, so a finished
    report read as one that never ran, and a report killed between globbing and writing could not be
    told from a complete one.
    """

    @staticmethod
    def _out_dir_with_a_behaviour_artifact(tmp_path: Path) -> Path:
        append_jsonl(
            tmp_path / "behaviour.jsonl",
            [
                BehaviouralRecord(
                    layer=9,
                    arm=ARM_REAL,
                    position_arm="all",
                    rho=rho,
                    alpha=rho * 30.0,
                    prompt_index=0,
                    sample_index=0,
                    thinking=False,
                    greedy=True,
                    prompt_text="explain a compass",
                    response_text="a compass shows direction",
                    response_tokens=6,
                    hit_token_cap=False,
                    elapsed_seconds=1.0,
                    target_script=SCRIPT_HAN,
                    target_ratio=0.0,
                    target_chars=0,
                    latin_chars=20,
                    other_letters=0,
                    ratio_denominator=20,
                    specificity={},
                    distinct_trigram_ratio=1.0,
                    repetition_rate=0.0,
                    token_entropy=2.0,
                    longest_token_run=1,
                    self_nll=1.0,
                    hook_calls=2,
                    hook_selected_positions=6,
                    hook_steered_positions=6,
                    hook_prefill_calls=1,
                    hook_decode_calls=1,
                )
                for rho in (0.0, 0.25)
            ],
        )
        return tmp_path

    def test_the_report_stage_writes_its_own_completion_marker(self, tmp_path: Path) -> None:
        out_dir = self._out_dir_with_a_behaviour_artifact(tmp_path)
        args = _parse_args(["report", "--out-dir", str(out_dir)])

        assert stage_report(args) == 0

        summary_path = out_dir / "report_summary.json"
        assert summary_path.exists(), "a finished report must be distinguishable from one never run"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["ok"] is True
        assert summary["artifact"] == "report.json"
        assert summary["sources"] == {"behaviour.jsonl": {"records": 2, "cells": 2}}
        assert summary["records_total"] == 2
        assert summary["cells_total"] == 2
        assert summary["isolation_files"] == []
        # The three globs that matched nothing are named, so a missing tier is visible.
        assert "screen_layers*.jsonl" in summary["patterns_matching_nothing"]
        assert "behaviour*.jsonl" not in summary["patterns_matching_nothing"]

    def test_an_empty_out_dir_still_gets_a_marker_with_zero_counts(self, tmp_path: Path) -> None:
        """A report over nothing is a completed report over nothing, with its denominator on it."""
        assert stage_report(_parse_args(["report", "--out-dir", str(tmp_path)])) == 0

        summary = json.loads((tmp_path / "report_summary.json").read_text(encoding="utf-8"))
        assert summary["records_total"] == 0
        assert summary["sources"] == {}
        assert len(summary["patterns_matching_nothing"]) == 4


# --------------------------------------------------------------------------------------
# M. Per-item generation seeding: what makes the control band a control
# --------------------------------------------------------------------------------------


class TestGenerationSeed:
    def test_every_prompt_and_sample_pair_gets_its_own_seed(self) -> None:
        seeds = {
            generation_seed(11, prompt_index=prompt, sample_index=sample)
            for prompt in range(32)
            for sample in range(16)
        }
        assert len(seeds) == 32 * 16

    def test_the_seed_depends_on_the_item_and_the_base_and_nothing_else(self) -> None:
        assert generation_seed(0, prompt_index=1, sample_index=0) == GENERATION_SEED_STRIDE
        assert generation_seed(5, prompt_index=0, sample_index=2) == 7
        assert generation_seed(5, prompt_index=0, sample_index=2) == generation_seed(
            5, prompt_index=0, sample_index=2
        )


_CELL_PROMPT_POSITIONS = 5
_CELL_DECODE_PASSES = 2


def _cell_record(prompt: str, sampling: SamplingConfig) -> GenerationRecord:
    prompt_ids = list(range(1, _CELL_PROMPT_POSITIONS + 1))
    response_ids = [6, 1]
    full = torch.tensor(prompt_ids + response_ids)
    return GenerationRecord(
        prompt_text=prompt,
        prompt_len=len(prompt_ids),
        response_text=_HAN_ANSWER,
        full_ids=full,
        token_strings=[str(value) for value in full.tolist()],
        is_response=torch.tensor([False] * len(prompt_ids) + [True] * len(response_ids)),
        hit_token_cap=False,
        sampler=resolved_sampler(sampling),
    )


class TestBehaviouralCellSeeding:
    """``generate`` has no per-request seed, so an unseeded cell samples off whatever the last one left.

    Per cell would not be enough either: a generation consumes a number of draws that depends on how
    long it turned out to be, so item three of the real arm would start from a different RNG state
    than item three of the placebo. These tests watch the seeding calls themselves, because the
    property is *when* the seed is set, not what the generation returned.
    """

    @staticmethod
    def _record_events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, object]]:
        events: list[tuple[str, object]] = []
        real_manual_seed = torch.manual_seed

        def recording_manual_seed(seed: int) -> torch.Generator:
            events.append(("seed", seed))
            return real_manual_seed(seed)

        def fake_generate(
            model: object,
            tokenizer: object,
            prompt: str,
            *,
            thinking: bool,
            sampling: SamplingConfig,
        ) -> GenerationRecord:
            del tokenizer, thinking
            events.append(("generate", prompt))
            # One prefill pass then one per decode step of real token ids: a real shape for assert_hook_pattern.
            trunk = model.model  # pyright: ignore[reportAttributeAccessIssue]
            trunk(torch.zeros(1, _CELL_PROMPT_POSITIONS, dtype=torch.long))
            for _ in range(_CELL_DECODE_PASSES):
                trunk(torch.zeros(1, 1, dtype=torch.long))
            return _cell_record(prompt, sampling)

        monkeypatch.setattr(torch, "manual_seed", recording_manual_seed)
        monkeypatch.setattr(steer_validation_module, "generate_response", fake_generate)
        return events

    @staticmethod
    def _run_cell(
        *, arm: str, generation_seed_base: int, item_order_seed: int, n_samples: int = 2
    ) -> list[BehaviouralRecord]:
        return run_behavioural_cell(
            _PositionLocalLM(),  # pyright: ignore[reportArgumentType]
            object(),  # pyright: ignore[reportArgumentType]
            ["first question", "second question"],
            layer=0,
            arm=arm,
            direction=torch.randn(_VOCAB),
            rho=0.2,
            alpha=1.0,
            position_arm="all",
            thinking=True,
            greedy=False,
            target_script=SCRIPT_HAN,
            n_samples=n_samples,
            sampling=thinking_sampling(max_new_tokens=16),
            generation_seed_base=generation_seed_base,
            item_order_seed=item_order_seed,
        )

    @staticmethod
    def _seeds(events: list[tuple[str, object]]) -> list[object]:
        return [value for kind, value in events if kind == "seed"]

    def test_each_generation_is_seeded_immediately_before_it_from_its_own_item(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two prompts times two samples: four seeds, four generations, strictly alternating."""
        events = self._record_events(monkeypatch)

        records = self._run_cell(arm=ARM_REAL, generation_seed_base=41, item_order_seed=0)

        assert len(records) == 4
        assert [kind for kind, _ in events] == ["seed", "generate"] * 4
        assert self._seeds(events) == [
            generation_seed(41, prompt_index=row.prompt_index, sample_index=row.sample_index)
            for row in records
        ]

    def test_the_real_arm_and_a_placebo_pair_item_for_item_across_a_different_item_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property the matched-norm band rests on, and the reason the seed is per item.

        The two arms are run under DIFFERENT ``item_order_seed`` values, so they visit the items in
        different orders. A per-cell seed would hand item two of one arm a different draw from item
        two of the other; a per-item seed cannot, whatever order they arrive in.
        """
        events = self._record_events(monkeypatch)

        real = self._run_cell(arm=ARM_REAL, generation_seed_base=7, item_order_seed=0)
        boundary = len(events)
        placebo = self._run_cell(arm=f"{ARM_PLACEBO}_0", generation_seed_base=7, item_order_seed=1)

        # A randperm stream change would equalise the orders and mute the pairing assertion below.
        assert [row.prompt_index for row in real] != [row.prompt_index for row in placebo]
        real_by_item = dict(
            zip(
                [(row.prompt_index, row.sample_index) for row in real],
                self._seeds(events[:boundary]),
                strict=True,
            )
        )
        placebo_by_item = dict(
            zip(
                [(row.prompt_index, row.sample_index) for row in placebo],
                self._seeds(events[boundary:]),
                strict=True,
            )
        )
        assert real_by_item == placebo_by_item
        # And the seeds within one cell are all different, which a per-cell seed would not give.
        assert len(set(real_by_item.values())) == len(real_by_item)

    def test_a_second_sample_of_the_same_prompt_draws_a_different_seed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise every sample of an item would be the same generation reported n times."""
        events = self._record_events(monkeypatch)

        records = self._run_cell(arm=ARM_REAL, generation_seed_base=0, item_order_seed=3)

        by_prompt: dict[int, set[object]] = defaultdict(set)
        for row, seed in zip(records, self._seeds(events), strict=True):
            by_prompt[row.prompt_index].add(seed)
        assert all(len(seeds) == 2 for seeds in by_prompt.values())


# --------------------------------------------------------------------------------------
# N. What a behavioural sweep records about its own decoding
# --------------------------------------------------------------------------------------


class TestBehaviourSweepSummary:
    @staticmethod
    def _sweep(*, skipped: list[str]) -> BehaviourSweep:
        return BehaviourSweep(
            cells_planned=3,
            cells_attempted=3,
            records_written=12,
            records_expected_per_cell=4,
            cells_skipped_by_deadline=skipped,
            cells_partial=[],
            cells_resumed_complete=[],
            sampling=thinking_sampling(max_new_tokens=128),
            generation_seed_base=7,
            position_arms=POSITION_ARMS,
        )

    def test_the_summary_block_states_the_sampler_and_that_the_run_was_seeded(self) -> None:
        fields = self._sweep(skipped=[]).to_summary_fields()

        assert fields["ok"] is True
        assert fields["sampling_unseeded"] is False
        assert fields["generation_seed_base"] == 7
        assert fields["records_written"] == 12
        # The recorded derivation must name the live stride, not a number copied beside it.
        derivation = fields["generation_seed_derivation"]
        assert isinstance(derivation, str)
        assert str(GENERATION_SEED_STRIDE) in derivation
        sampler = fields["sampler"]
        assert isinstance(sampler, dict)
        assert sampler["max_new_tokens"] == 128
        # transformers' generate takes no seed, which is exactly why the global stream is seeded.
        resolved = fields["resolved_sampler"]
        assert isinstance(resolved, dict)
        assert resolved["seed_applied"] is None
        assert "seed_why_dropped" in resolved

    def test_a_deadline_truncated_sweep_is_not_ok(self) -> None:
        fields = self._sweep(skipped=["L12/decode/real/0.4"]).to_summary_fields()
        assert fields["ok"] is False
        assert fields["cells_skipped_by_deadline"] == ["L12/decode/real/0.4"]

    def test_a_partial_cell_is_not_ok_even_with_an_empty_skip_list(self) -> None:
        """The 20260824b gap: a deadline hitting the FINAL cell mid-way left ok keyed on skips."""
        sweep = replace(self._sweep(skipped=[]), cells_partial=["L12/all/real/0.25"])
        fields = sweep.to_summary_fields()
        assert fields["ok"] is False
        assert fields["cells_partial"] == ["L12/all/real/0.25"]

    def test_a_sweep_that_misses_its_plan_is_not_ok(self) -> None:
        """attempted + skipped must equal the independently computed plan."""
        assert replace(self._sweep(skipped=[]), cells_planned=4).is_complete is False

    def test_a_record_shortfall_is_not_ok_even_when_no_cell_flagged_partial(self) -> None:
        """Belt and suspenders: the record count must equal attempted cells times full depth."""
        assert replace(self._sweep(skipped=[]), records_written=11).is_complete is False

    def test_resumed_cells_count_toward_the_plan_without_fresh_records(self) -> None:
        """A resumed cell's rows already sit in the artifact; the union is what ok blesses."""
        sweep = replace(
            self._sweep(skipped=[]),
            cells_planned=5,
            cells_resumed_complete=["L0/all/real/0.05", "L0/all/real/0.25"],
        )
        assert sweep.is_complete is True
        assert sweep.to_summary_fields()["cells_resumed_complete"] == [
            "L0/all/real/0.05",
            "L0/all/real/0.25",
        ]


def _fake_behavioural_record(
    *, layer: int, arm: str, position_arm: str, rho: float, prompt_index: int
) -> BehaviouralRecord:
    """A minimal well-formed record for stage tests that stub out generation entirely."""
    return BehaviouralRecord(
        layer=layer,
        arm=arm,
        position_arm=position_arm,
        rho=rho,
        alpha=0.0,
        prompt_index=prompt_index,
        sample_index=0,
        thinking=False,
        greedy=True,
        prompt_text="stub prompt",
        response_text="stub response",
        response_tokens=2,
        hit_token_cap=False,
        elapsed_seconds=0.01,
        target_script=SCRIPT_HAN,
        target_ratio=0.5,
        target_chars=1,
        latin_chars=1,
        other_letters=0,
        ratio_denominator=2,
        specificity={},
        distinct_trigram_ratio=1.0,
        repetition_rate=0.0,
        token_entropy=1.0,
        longest_token_run=1,
        self_nll=None,
        hook_calls=1,
        hook_selected_positions=1,
        hook_steered_positions=1,
        hook_prefill_calls=1,
        hook_decode_calls=0,
    )


class TestBehaviourPositionArmRestriction:
    """``--position-arms`` narrows the behavioural enumeration; the default stays all three.

    Motivated by the 20260824b behaviour-grid truncation: the corrected control arms multiplied the
    cell count past the unit deadline, and the all-positions cells the rho-pick keys on never ran
    because the enumeration spent two thirds of its cells on position arms the pick never reads.
    These tests pin the live enumeration against :func:`planned_behaviour_cells` -- the formula the
    sweep's own completeness plan uses and the launch kit's sizing gate mirrors -- so the two
    cannot drift apart silently.
    """

    @staticmethod
    def _swept_cells(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str], *, draws: int = 1
    ) -> tuple[list[tuple[int, str, str, float]], dict[str, object]]:
        """Run the behaviour stage on the fit stubs, recording every cell the sweep reaches."""
        _install_fit_stubs(monkeypatch, target_in_han=True, baseline_in_han=False)
        assert stage_fit(_fit_args(tmp_path)) == 0
        cells: list[tuple[int, str, str, float]] = []

        def record_cell(  # noqa: PLR0913 - mirrors run_behavioural_cell's cell coordinates
            model: object,
            tokenizer: object,
            prompts: Sequence[str],
            *,
            layer: int,
            arm: str,
            position_arm: str,
            rho: float,
            n_samples: int,
            **_knobs: object,
        ) -> list[BehaviouralRecord]:
            del model, tokenizer
            cells.append((layer, position_arm, arm, rho))
            # Full depth, or the sweep's completeness accounting rightly flags the cell partial.
            return [
                _fake_behavioural_record(
                    layer=layer, arm=arm, position_arm=position_arm, rho=rho, prompt_index=index
                )
                for index in range(len(prompts) * n_samples)
            ]

        monkeypatch.setattr(
            "reward_hacking.interp.run_steer_validation.run_behavioural_cell", record_cell
        )
        args = _parse_args(
            [
                "behaviour",
                "--out-dir",
                str(tmp_path),
                "--layers",
                "0",
                "1",
                "--rhos",
                "0.25",
                "1.0",
                "--n-control-draws",
                str(draws),
                *extra,
            ]
        )
        assert stage_behaviour(args) == 0
        summary = json.loads((tmp_path / "behaviour_summary.json").read_text(encoding="utf-8"))
        assert isinstance(summary, dict)
        return cells, summary

    def test_the_default_sweep_runs_every_position_arm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cells, summary = self._swept_cells(tmp_path, monkeypatch, [])

        # 2 layers x 3 position arms x (1 real + 3 control families x 1 draw) x 2 rhos.
        assert (
            len(cells)
            == 2 * 3 * 4 * 2
            == planned_behaviour_cells(n_layers=2, n_position_arms=3, n_control_draws=1, n_rhos=2)
        )
        assert {cell[1] for cell in cells} == set(POSITION_ARMS)
        assert summary["position_arms"] == list(POSITION_ARMS)
        assert summary["cells_attempted"] == len(cells)

    def test_a_restricted_sweep_runs_only_the_named_arm_and_records_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cells, summary = self._swept_cells(tmp_path, monkeypatch, ["--position-arms", "all"])

        # 2 layers x 1 position arm x (1 real + 3 control families x 1 draw) x 2 rhos.
        assert (
            len(cells)
            == 2 * 1 * 4 * 2
            == planned_behaviour_cells(n_layers=2, n_position_arms=1, n_control_draws=1, n_rhos=2)
        )
        assert {cell[1] for cell in cells} == {"all"}
        assert summary["position_arms"] == ["all"]
        assert summary["cells_attempted"] == len(cells)
        # The restriction narrows positions only: the full control set still runs in the named arm.
        assert {cell[2] for cell in cells} == {
            ARM_REAL,
            f"{ARM_PLACEBO}_0",
            f"{ARM_SHUFFLED}_0",
            f"{ARM_ORTHOGONAL}_0",
        }

    def test_the_control_draw_factor_scales_the_enumeration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Draws multiply the arm count -- the exact factor that truncated the 20260824b grid."""
        cells, summary = self._swept_cells(tmp_path, monkeypatch, [], draws=2)

        # 2 layers x 3 position arms x (1 real + 3 control families x 2 draws) x 2 rhos.
        assert (
            len(cells)
            == 2 * 3 * 7 * 2
            == planned_behaviour_cells(n_layers=2, n_position_arms=3, n_control_draws=2, n_rhos=2)
        )
        assert summary["cells_planned"] == len(cells)
        assert {cell[2] for cell in cells if cell[2] != ARM_REAL} == {
            f"{family}_{index}"
            for family in (ARM_PLACEBO, ARM_SHUFFLED, ARM_ORTHOGONAL)
            for index in (0, 1)
        }

    def test_an_unknown_position_arm_is_refused_at_the_parser(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["behaviour", "--out-dir", str(tmp_path), "--position-arms", "sideways"])

    def test_a_bare_position_arms_flag_is_refused_not_read_as_all_three(
        self, tmp_path: Path
    ) -> None:
        """A bare flag used to parse to [], which is falsy: the sweep then ran ALL THREE arms
        while the kit's word-count sizing projected zero cells and blessed the 3x overspend."""
        with pytest.raises(SystemExit):
            _parse_args(["behaviour", "--out-dir", str(tmp_path), "--position-arms"])

    def test_repeated_values_are_refused_for_every_list_flag(self, tmp_path: Path) -> None:
        """A repeat re-runs identical cells at identical seeds and doubles the reported sample
        size while cells_planned doubles in lockstep, so is_complete would stay green."""
        with pytest.raises(SystemExit):
            _parse_args(["behaviour", "--out-dir", str(tmp_path), "--position-arms", "all", "all"])
        with pytest.raises(SystemExit):
            _parse_args(["behaviour", "--out-dir", str(tmp_path), "--layers", "6", "6"])
        with pytest.raises(SystemExit):
            _parse_args(["behaviour", "--out-dir", str(tmp_path), "--rhos", "0.25", "0.25"])

    def test_a_screening_stage_refuses_the_flag_rather_than_ignoring_it(
        self, tmp_path: Path
    ) -> None:
        """A flag that parses but does nothing reads as a restriction that took effect."""
        args = _parse_args(["screen-layers", "--out-dir", str(tmp_path), "--position-arms", "all"])
        with pytest.raises(ValueError, match="does not apply to the screening stages"):
            stage_screen_layers(args)


class TestBehaviourStageTruncationHonesty:
    """A behaviour stage that did not run its whole plan says so in its exit code.

    The 20260824b chain marked a 48-of-168-cell grid ``rc: 0``, so every downstream marker read
    "all units rc 0" off a truncated artifact, and a deadline reaching the final cell MID-generation
    left that cell in no list at all. These run the real stage function over stubs.
    """

    @staticmethod
    def _stage_args(tmp_path: Path, extra: list[str]) -> argparse.Namespace:
        return _parse_args(
            [
                "behaviour",
                "--out-dir",
                str(tmp_path),
                "--layers",
                "0",
                "--rhos",
                "0.25",
                "--n-control-draws",
                "1",
                *extra,
            ]
        )

    def test_a_mid_cell_deadline_is_a_partial_cell_and_a_non_zero_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fit_stubs(monkeypatch, target_in_han=True, baseline_in_han=False)
        assert stage_fit(_fit_args(tmp_path)) == 0

        def short_cell(  # noqa: PLR0913 - mirrors run_behavioural_cell's cell coordinates
            model: object,
            tokenizer: object,
            prompts: Sequence[str],
            *,
            layer: int,
            arm: str,
            position_arm: str,
            rho: float,
            n_samples: int,
            **_knobs: object,
        ) -> list[BehaviouralRecord]:
            del model, tokenizer
            depth = len(prompts) * n_samples
            # The real arm's cell stops one generation early, as a mid-cell deadline would.
            if arm == ARM_REAL and position_arm == "all":
                depth -= 1
            return [
                _fake_behavioural_record(
                    layer=layer, arm=arm, position_arm=position_arm, rho=rho, prompt_index=index
                )
                for index in range(depth)
            ]

        monkeypatch.setattr(
            "reward_hacking.interp.run_steer_validation.run_behavioural_cell", short_cell
        )
        assert stage_behaviour(self._stage_args(tmp_path, [])) == 1

        summary = json.loads((tmp_path / "behaviour_summary.json").read_text(encoding="utf-8"))
        assert summary["ok"] is False
        assert summary["cells_partial"] == ["L0/all/real/0.25"]
        assert summary["cells_skipped_by_deadline"] == []

    def test_an_expired_deadline_skips_every_cell_and_exits_non_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fit_stubs(monkeypatch, target_in_han=True, baseline_in_han=False)
        assert stage_fit(_fit_args(tmp_path)) == 0
        monkeypatch.setattr(
            "reward_hacking.interp.run_steer_validation.run_behavioural_cell",
            lambda *a, **k: pytest.fail("a skipped sweep must not generate"),
        )

        assert stage_behaviour(self._stage_args(tmp_path, ["--deadline-seconds", "-1"])) == 1

        summary = json.loads((tmp_path / "behaviour_summary.json").read_text(encoding="utf-8"))
        assert summary["ok"] is False
        assert summary["cells_attempted"] == 0
        # 1 layer x 3 position arms x (1 real + 3 families x 1 draw) x 1 rho, all skipped.
        assert len(summary["cells_skipped_by_deadline"]) == summary["cells_planned"] == 12


class TestBehaviourResume:
    """``--resume`` keeps complete cells, regenerates the rest, and reports the union honestly.

    Owner directive (2026-08-28): a box that dies early must never force a restart from zero. The
    key that separates a continuation from the accidental double-run the claim guard exists for is
    the full complement: a cell already at ``prompts x samples`` rows is inherited as
    ``cells_resumed_complete`` (a DISTINCT ledger from deadline skips), anything less is dropped
    and regenerated -- losslessly, because seeds derive from indices, not execution order.
    """

    @staticmethod
    def _grid_args(tmp_path: Path, extra: list[str]) -> argparse.Namespace:
        return _parse_args(
            [
                "behaviour",
                "--out-dir",
                str(tmp_path),
                "--layers",
                "0",
                "--rhos",
                "0.25",
                "--n-control-draws",
                "1",
                *extra,
            ]
        )

    @staticmethod
    def _full_cell_fake(
        counter: list[str],
    ) -> object:
        def full_cell(  # noqa: PLR0913 - mirrors run_behavioural_cell's cell coordinates
            model: object,
            tokenizer: object,
            prompts: Sequence[str],
            *,
            layer: int,
            arm: str,
            position_arm: str,
            rho: float,
            n_samples: int,
            **_knobs: object,
        ) -> list[BehaviouralRecord]:
            del model, tokenizer
            counter.append(f"L{layer}/{position_arm}/{arm}/{rho}")
            return [
                _fake_behavioural_record(
                    layer=layer, arm=arm, position_arm=position_arm, rho=rho, prompt_index=index
                )
                for index in range(len(prompts) * n_samples)
            ]

        return full_cell

    def test_a_complete_artifact_resumes_without_a_single_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fit_stubs(monkeypatch, target_in_han=True, baseline_in_han=False)
        assert stage_fit(_fit_args(tmp_path)) == 0
        first_run: list[str] = []
        monkeypatch.setattr(
            "reward_hacking.interp.run_steer_validation.run_behavioural_cell",
            self._full_cell_fake(first_run),
        )
        assert stage_behaviour(self._grid_args(tmp_path, [])) == 0
        assert len(first_run) == 12  # 1 layer x 3 position arms x 4 arms x 1 rho

        monkeypatch.setattr(
            "reward_hacking.interp.run_steer_validation.run_behavioural_cell",
            lambda *a, **k: pytest.fail("a fully resumed sweep must not generate"),
        )
        assert stage_behaviour(self._grid_args(tmp_path, ["--resume"])) == 0

        summary = json.loads((tmp_path / "behaviour_summary.json").read_text(encoding="utf-8"))
        assert summary["ok"] is True
        assert summary["cells_attempted"] == 0
        assert summary["records_written"] == 0
        assert len(summary["cells_resumed_complete"]) == 12
        assert summary["cells_skipped_by_deadline"] == []
        rows = read_jsonl(tmp_path / "behaviour.jsonl")
        assert len(rows) == 12 * 4  # the union: every inherited row still on disk

    def test_a_partial_artifact_regenerates_only_the_missing_cells(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fit_stubs(monkeypatch, target_in_han=True, baseline_in_han=False)
        assert stage_fit(_fit_args(tmp_path)) == 0

        def short_real_all(  # noqa: PLR0913 - mirrors run_behavioural_cell's cell coordinates
            model: object,
            tokenizer: object,
            prompts: Sequence[str],
            *,
            layer: int,
            arm: str,
            position_arm: str,
            rho: float,
            n_samples: int,
            **_knobs: object,
        ) -> list[BehaviouralRecord]:
            del model, tokenizer
            depth = len(prompts) * n_samples
            if arm == ARM_REAL and position_arm == "all":
                depth -= 1
            return [
                _fake_behavioural_record(
                    layer=layer, arm=arm, position_arm=position_arm, rho=rho, prompt_index=index
                )
                for index in range(depth)
            ]

        monkeypatch.setattr(
            "reward_hacking.interp.run_steer_validation.run_behavioural_cell", short_real_all
        )
        assert stage_behaviour(self._grid_args(tmp_path, [])) == 1  # honest about the partial

        rerun: list[str] = []
        monkeypatch.setattr(
            "reward_hacking.interp.run_steer_validation.run_behavioural_cell",
            self._full_cell_fake(rerun),
        )
        # The summary is a completion marker, not a claim: the resume overwrites it with the union.
        assert stage_behaviour(self._grid_args(tmp_path, ["--resume"])) == 0

        assert rerun == ["L0/all/real/0.25"]  # only the cell whose rows were dropped
        summary = json.loads((tmp_path / "behaviour_summary.json").read_text(encoding="utf-8"))
        assert summary["ok"] is True
        assert summary["cells_attempted"] == 1
        assert len(summary["cells_resumed_complete"]) == 11
        assert summary["cells_partial"] == []
        rows = read_jsonl(tmp_path / "behaviour.jsonl")
        assert len(rows) == 12 * 4
        keys = {
            (row["layer"], row["position_arm"], row["arm"], row["rho"], row["prompt_index"])
            for row in rows
        }
        assert len(keys) == 12 * 4  # no duplicated (cell, prompt) row anywhere in the union

    def test_without_resume_an_existing_artifact_still_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fit_stubs(monkeypatch, target_in_han=True, baseline_in_han=False)
        assert stage_fit(_fit_args(tmp_path)) == 0
        monkeypatch.setattr(
            "reward_hacking.interp.run_steer_validation.run_behavioural_cell",
            self._full_cell_fake([]),
        )
        assert stage_behaviour(self._grid_args(tmp_path, [])) == 0

        with pytest.raises(FileExistsError, match=r"behaviour\.jsonl"):
            stage_behaviour(self._grid_args(tmp_path, []))


class TestStagePickCell:
    """The hardened picker refuses truncated grids and cells too thin to carry a median."""

    @staticmethod
    def _write_grid(
        tmp_path: Path,
        rows: list[dict[str, object]],
        *,
        ok: bool = True,
        n_prompts: int = 4,
        resumed: list[str] | None = None,
    ) -> argparse.Namespace:
        (tmp_path / "behaviour_grid.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        summary: dict[str, object] = {
            "ok": ok,
            "n_prompts": n_prompts,
            "n_samples": 1,
            "cells_skipped_by_deadline": [] if ok else ["L6/all/real/0.25"],
            "cells_partial": [],
            "cells_resumed_complete": resumed or [],
        }
        (tmp_path / "behaviour_summary_grid.json").write_text(json.dumps(summary), encoding="utf-8")
        return _parse_args(["pick-cell", "--out-dir", str(tmp_path), "--label", "grid"])

    @staticmethod
    def _cell(
        layer: int, rho: float, values: list[float | None], *, caps: int = 0
    ) -> list[dict[str, object]]:
        return [
            _behaviour_row(layer=layer, rho=rho, value=value, hit_cap=index < caps)
            for index, value in enumerate(values)
        ]

    def test_picks_the_highest_median_healthy_cell(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rows = self._cell(6, 0.25, [0.9, 0.8, 0.7, 0.6]) + self._cell(
            11, 0.05, [0.1, 0.1, 0.2, 0.2]
        )
        assert stage_pick_cell(self._write_grid(tmp_path, rows)) == 0
        assert capsys.readouterr().out.strip() == "6 0.25"

    def test_refuses_a_truncated_grid_outright(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Both prior runs picked off truncated artifacts; ok=false must end the pick, not warn."""
        rows = self._cell(6, 0.25, [0.9, 0.8, 0.7, 0.6])
        assert stage_pick_cell(self._write_grid(tmp_path, rows, ok=False)) == 1
        assert capsys.readouterr().out.strip() == ""

    def test_a_healthy_resumed_grid_is_pickable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A relaunch's union -- inherited cells plus fresh ones, ok=true -- picks like any grid.

        The resume machinery must not trip the picker: ``cells_resumed_complete`` is finished work,
        not missing work, so an ok=true summary over the union is exactly as pickable as an
        unbroken run's. The inherited cell here carries the higher median and must win.
        """
        rows = self._cell(6, 0.25, [0.9, 0.8, 0.7, 0.6]) + self._cell(
            11, 0.05, [0.1, 0.1, 0.2, 0.2]
        )
        args = self._write_grid(tmp_path, rows, resumed=["L6/all/real/0.25"])
        assert stage_pick_cell(args) == 0
        assert capsys.readouterr().out.strip() == "6 0.25"

    def test_refuses_when_the_summary_is_absent(self, tmp_path: Path) -> None:
        args = _parse_args(["pick-cell", "--out-dir", str(tmp_path), "--label", "grid"])
        assert stage_pick_cell(args) == 1

    def test_a_mostly_unscorable_cell_cannot_win_on_its_two_rows(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The motivating defect: a median over two scorable rows beating a full healthy cell."""
        rows = self._cell(6, 0.25, [1.0, 1.0, None, None]) + self._cell(
            11, 0.05, [0.5, 0.5, 0.5, 0.4]
        )
        assert stage_pick_cell(self._write_grid(tmp_path, rows)) == 0
        assert capsys.readouterr().out.strip() == "11 0.05"

    def test_an_incomplete_cell_cannot_win(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rows = self._cell(6, 0.25, [1.0, 1.0, 1.0]) + self._cell(11, 0.05, [0.5, 0.5, 0.5, 0.4])
        assert stage_pick_cell(self._write_grid(tmp_path, rows)) == 0
        assert capsys.readouterr().out.strip() == "11 0.05"

    def test_cap_fraction_down_ranks_at_equal_medians_but_never_excludes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Cap-heavy cells lose ties yet stay eligible: the b-run's real effect capped 16/16."""
        rows = self._cell(6, 0.25, [0.5, 0.5, 0.5, 0.5], caps=4) + self._cell(
            11, 0.05, [0.5, 0.5, 0.5, 0.5]
        )
        assert stage_pick_cell(self._write_grid(tmp_path, rows)) == 0
        assert capsys.readouterr().out.strip() == "11 0.05"

        # Alone, the fully-capped cell still wins: down-ranked is not disqualified.
        solo_dir = tmp_path / "solo"
        solo_dir.mkdir()
        solo = self._cell(6, 0.25, [0.5, 0.5, 0.5, 0.5], caps=4)
        assert stage_pick_cell(self._write_grid(solo_dir, solo)) == 0
        assert capsys.readouterr().out.strip() == "6 0.25"

    def test_refuses_when_nothing_is_eligible(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Negative rho is a sign control and control arms are controls: neither may be picked."""
        rows = self._cell(6, -0.25, [0.9, 0.9, 0.9, 0.9]) + [
            _behaviour_row(layer=11, rho=0.25, arm=f"{ARM_PLACEBO}_0", value=0.9) for _ in range(4)
        ]
        assert stage_pick_cell(self._write_grid(tmp_path, rows)) == 1
        assert capsys.readouterr().out.strip() == ""


class TestEveryFlagNamedInAMessageExists:
    def test_no_message_in_the_module_names_a_flag_the_parser_does_not_accept(self) -> None:
        """The thinking tier's own refusal used to name ``--rms-scales``, which never existed.

        That message is the only thing an operator sees after launching the thinking tier without
        strengths, on a rented box, so a wrong flag name there costs a launch cycle on metered
        hardware. Checked over every string literal in the module rather than that one message,
        because the next stale flag name will be somewhere else.
        """
        source = inspect.getsource(run_steer_validation_module)
        declared = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', source))
        referenced = set(re.findall(r'"(--[a-z0-9-]+)', source))

        assert declared, "the flag scrape found no options at all, so it is checking nothing"
        assert referenced - declared == set()
