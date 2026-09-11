"""Offline tests for the reward-hacking vs deception direction probe.

These run on CPU in the ~8s budget: no model loads, no GPU. The pure-tensor core is exercised on
synthetic activations with a KNOWN planted direction, so every claim the probe makes has a
ground-truth answer to check against:

* diff-of-means recovers the planted direction;
* the per-layer cosine output has one entry per layer;
* the matched-norm random placebo sits at chance (cosine ~ 0 in high dimension);
* aligned concept activations give cosine ~ 1 and orthogonal ones give ~ 0.

Per the repo rule that a check never watched fail is not a check, the layer-mismatch guard test
plants the exact violation it exists to catch -- activation sets that cover different layers, the
shape a capture that silently dropped a layer would produce -- and confirms it raises rather than
quietly intersecting.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from reward_hacking.interp import stimuli
from reward_hacking.interp.directions import (
    LayerComparison,
    _assert_trunk_carries_the_adapter,  # pyright: ignore[reportPrivateUsage]  # test exercises this internal helper
    _decoder_layers,  # pyright: ignore[reportPrivateUsage]  # test exercises this internal helper
    _transformer_trunk,  # pyright: ignore[reportPrivateUsage]  # test exercises this internal helper
    capture_pooled_activations,
    capture_pooled_activations_multi,
    capture_positionwise_activations,
    compare_directions,
    concept_directions,
    cosine,
    diff_of_means,
    last_token_pool,
    matched_norm_random_direction,
    mean_pool,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_decoder_layers_resolves_across_architectures() -> None:
    """Locate decoder layers on both checkpoint shapes -- the crash the GPU smoke caught.

    The probe hardcoded the 4B vision-language checkpoint's ``language_model.layers`` and died on
    plain causal-LM models (the 0.8B, and Phi-4/Llama for the cross-model control). This is the
    offline guard the synthetic-tensor tests could not give, since it needs a module structure.
    """
    layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    causal = SimpleNamespace(model=SimpleNamespace(layers=layers))  # Qwen3_5ForCausalLM / Llama
    vlm = SimpleNamespace(language_model=SimpleNamespace(layers=layers))  # Qwen3.5-4B VL
    assert _decoder_layers(causal) is layers  # pyright: ignore[reportArgumentType]
    assert _decoder_layers(vlm) is layers  # pyright: ignore[reportArgumentType]
    with pytest.raises(AttributeError):
        _decoder_layers(SimpleNamespace(nothing=True))  # pyright: ignore[reportArgumentType]


def _acts_with_planted_mean(
    mean_vec: torch.Tensor, n: int, generator: torch.Generator
) -> torch.Tensor:
    """``n`` synthetic activations whose sample mean is EXACTLY ``mean_vec``.

    Noise is centered (its own column-mean subtracted) so the sample mean equals ``mean_vec`` up
    to float error, which is what lets diff-of-means be checked against ground truth exactly.
    """
    d = mean_vec.shape[0]
    noise = torch.randn(n, d, generator=generator)
    noise = noise - noise.mean(dim=0, keepdim=True)
    return mean_vec.unsqueeze(0) + noise


def _acts_with_planted_mean_and_std(
    mean_vec: torch.Tensor, noise_std: torch.Tensor, n: int, generator: torch.Generator
) -> torch.Tensor:
    """``n`` synthetic activations: sample mean EXACTLY ``mean_vec``, per-dim std ``noise_std``.

    The noise is centered per column (its own mean subtracted) so the sample mean stays exact, then
    scaled dimension-wise -- which lets a chosen dimension carry a huge across-sentence variance,
    the fingerprint of a massive-activation dimension, without disturbing the diff-of-means.
    """
    d = mean_vec.shape[0]
    noise = torch.randn(n, d, generator=generator)
    noise = noise - noise.mean(dim=0, keepdim=True)
    return mean_vec.unsqueeze(0) + noise * noise_std


def _concept_activations(
    planted_by_layer: dict[int, torch.Tensor],
    base_by_layer: dict[int, torch.Tensor],
    n: int,
    generator: torch.Generator,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Build (positives, negatives) whose per-layer mean difference is ``planted_by_layer``."""
    positives = {
        layer: _acts_with_planted_mean(base_by_layer[layer] + planted, n, generator)
        for layer, planted in planted_by_layer.items()
    }
    negatives = {
        layer: _acts_with_planted_mean(base_by_layer[layer], n, generator)
        for layer in planted_by_layer
    }
    return positives, negatives


class TestPooling:
    """Mask-driven pooling ignores padding and is padding-side agnostic -- a silent-failure risk."""

    def test_mean_pool_ignores_right_padding(self) -> None:
        hidden = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]]])
        mask = torch.tensor([[1, 1, 0]])
        assert mean_pool(hidden, mask).tolist() == [[1.5, 1.5]]

    def test_mean_pool_ignores_left_padding(self) -> None:
        hidden = torch.tensor([[[100.0, 100.0], [2.0, 2.0], [4.0, 4.0]]])
        mask = torch.tensor([[0, 1, 1]])
        assert mean_pool(hidden, mask).tolist() == [[3.0, 3.0]]

    def test_last_token_pool_right_padding(self) -> None:
        hidden = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]]])
        mask = torch.tensor([[1, 1, 0]])
        assert last_token_pool(hidden, mask).tolist() == [[2.0, 2.0]]

    def test_last_token_pool_left_padding(self) -> None:
        hidden = torch.tensor([[[100.0, 100.0], [2.0, 2.0], [3.0, 3.0]]])
        mask = torch.tensor([[0, 1, 1]])
        assert last_token_pool(hidden, mask).tolist() == [[3.0, 3.0]]


class TestCosine:
    """Cosine similarity has the textbook endpoints, so the comparison numbers are trustworthy."""

    def test_identical_is_one(self) -> None:
        v = torch.tensor([1.0, 2.0, 3.0])
        assert cosine(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_is_minus_one(self) -> None:
        v = torch.tensor([1.0, 2.0, 3.0])
        assert cosine(v, -v) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_is_zero(self) -> None:
        assert cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == pytest.approx(
            0.0, abs=1e-6
        )


class TestDiffOfMeans:
    """(a) diff-of-means recovers a planted direction, per layer, from synthetic activations."""

    def test_recovers_planted_direction(self) -> None:
        generator = torch.Generator().manual_seed(0)
        d = 64
        base = torch.randn(d, generator=generator)
        planted = torch.randn(d, generator=generator)
        positives = _acts_with_planted_mean(base + planted, 40, generator)
        negatives = _acts_with_planted_mean(base, 40, generator)
        recovered = diff_of_means(positives, negatives)
        assert torch.allclose(recovered, planted, atol=1e-4)

    def test_recovers_a_distinct_direction_per_layer(self) -> None:
        generator = torch.Generator().manual_seed(1)
        d = 32
        planted = {layer: torch.randn(d, generator=generator) for layer in range(4)}
        base = {layer: torch.randn(d, generator=generator) for layer in range(4)}
        positives, negatives = _concept_activations(planted, base, 30, generator)
        directions = concept_directions(positives, negatives)
        for layer in range(4):
            assert torch.allclose(directions[layer], planted[layer], atol=1e-4)


class TestMatchedNormPlacebo:
    """The placebo direction matches norm exactly and sits at chance against a fixed vector."""

    def test_norm_is_matched(self) -> None:
        generator = torch.Generator().manual_seed(2)
        reference = torch.randn(2560, generator=generator) * 7.0
        placebo = matched_norm_random_direction(reference, generator)
        assert placebo.norm().item() == pytest.approx(reference.norm().item(), rel=1e-5)

    def test_placebo_is_not_parallel_to_reference(self) -> None:
        generator = torch.Generator().manual_seed(3)
        reference = torch.randn(2560, generator=generator)
        placebo = matched_norm_random_direction(reference, generator)
        # Fresh random direction in 2560-d is near-orthogonal to any fixed vector (std ~ 1/sqrt(d)).
        assert abs(cosine(reference, placebo)) < 0.1


class TestCompareDirections:
    """(b) shape, (c) placebo-at-chance, (d) aligned-high / orthogonal-zero, plus the guard."""

    def test_output_has_one_comparison_per_layer(self) -> None:
        generator = torch.Generator().manual_seed(4)
        d, layers = 2560, [0, 1, 2, 3, 4]
        planted_hack = {layer: torch.randn(d, generator=generator) for layer in layers}
        planted_dec = {layer: torch.randn(d, generator=generator) for layer in layers}
        base = {layer: torch.randn(d, generator=generator) for layer in layers}
        hack_pos, hack_neg = _concept_activations(planted_hack, base, 20, generator)
        dec_pos, dec_neg = _concept_activations(planted_dec, base, 20, generator)

        result = compare_directions(hack_pos, hack_neg, dec_pos, dec_neg, seed=0)

        assert [c.layer for c in result] == layers
        assert all(isinstance(c, LayerComparison) for c in result)

    def test_placebo_cosine_is_at_chance(self) -> None:
        generator = torch.Generator().manual_seed(5)
        d, layers = 2560, [0, 1, 2]
        planted_hack = {layer: torch.randn(d, generator=generator) for layer in layers}
        planted_dec = {layer: torch.randn(d, generator=generator) for layer in layers}
        base = {layer: torch.randn(d, generator=generator) for layer in layers}
        hack_pos, hack_neg = _concept_activations(planted_hack, base, 20, generator)
        dec_pos, dec_neg = _concept_activations(planted_dec, base, 20, generator)

        result = compare_directions(hack_pos, hack_neg, dec_pos, dec_neg, seed=7)

        for comparison in result:
            assert abs(comparison.cos_hack_placebo) < 0.1
            assert abs(comparison.cos_deception_placebo) < 0.1

    def test_aligned_concepts_are_high_and_orthogonal_are_zero(self) -> None:
        generator = torch.Generator().manual_seed(6)
        d = 2560
        base = {0: torch.zeros(d)}
        shared = torch.randn(d, generator=generator)
        # Aligned: both concepts have the SAME planted direction -> cosine should be ~1.
        hack_pos, hack_neg = _concept_activations({0: shared}, base, 20, generator)
        dec_pos, dec_neg = _concept_activations({0: shared}, base, 20, generator)
        aligned = compare_directions(hack_pos, hack_neg, dec_pos, dec_neg, seed=0)
        assert aligned[0].cos_hack_deception == pytest.approx(1.0, abs=1e-3)

        # Orthogonal: planted directions on disjoint axes -> cosine should be ~0.
        axis_hack = torch.zeros(d)
        axis_hack[0] = 4.0
        axis_dec = torch.zeros(d)
        axis_dec[1] = 9.0
        hack_pos, hack_neg = _concept_activations({0: axis_hack}, base, 20, generator)
        dec_pos, dec_neg = _concept_activations({0: axis_dec}, base, 20, generator)
        orthogonal = compare_directions(hack_pos, hack_neg, dec_pos, dec_neg, seed=0)
        assert orthogonal[0].cos_hack_deception == pytest.approx(0.0, abs=1e-3)

    def test_standardized_cosine_strips_massive_activation_inflation(self) -> None:
        """SABOTAGE (massive-activation artifact): plant one residual dimension that carries a
        huge, concept-shared mean offset AND a huge across-sentence variance, plus tiny orthogonal
        true-concept signals. The raw diff-of-means is dominated by that dimension, so the raw
        hack/deception cosine is pinned near 1 regardless of the (orthogonal) concept content. The
        matched-norm placebo does not catch this. Standardizing by per-dim std strips the dominant
        dimension's leverage, so the standardized cosine falls back to the true (~0) concept overlap
        -- the raw-vs-standardized gap is the flag. Watched to inflate raw and NOT inflate std.
        """
        generator = torch.Generator().manual_seed(11)
        d, n = 256, 40
        massive_dim, hack_dim, deception_dim = 0, 1, 2
        shared_offset, concept_signal, massive_std = 100.0, 1.0, 1000.0

        hack_pos_mean = torch.zeros(d)
        hack_pos_mean[massive_dim] = shared_offset
        hack_pos_mean[hack_dim] = concept_signal
        deception_pos_mean = torch.zeros(d)
        deception_pos_mean[massive_dim] = shared_offset
        deception_pos_mean[deception_dim] = concept_signal
        negative_mean = torch.zeros(d)  # both concepts share the same (zero) negative pole

        noise_std = torch.ones(d)
        noise_std[massive_dim] = massive_std

        hack_pos = {0: _acts_with_planted_mean_and_std(hack_pos_mean, noise_std, n, generator)}
        hack_neg = {0: _acts_with_planted_mean_and_std(negative_mean, noise_std, n, generator)}
        dec_pos = {0: _acts_with_planted_mean_and_std(deception_pos_mean, noise_std, n, generator)}
        dec_neg = {0: _acts_with_planted_mean_and_std(negative_mean, noise_std, n, generator)}

        result = compare_directions(hack_pos, hack_neg, dec_pos, dec_neg, seed=0)[0]

        # Raw cosine is inflated by the shared massive dimension; standardization strips it.
        assert result.cos_hack_deception > 0.95
        assert result.cos_hack_deception_standardized < 0.3
        assert result.cos_hack_deception - result.cos_hack_deception_standardized > 0.5
        # The standardized placebo baselines stay at chance -- the /std transform invents no cosine.
        assert abs(result.cos_hack_placebo_standardized) < 0.3
        assert abs(result.cos_deception_placebo_standardized) < 0.3

    def test_cross_concept_layer_mismatch_raises(self) -> None:
        """SABOTAGE (silent-drop shape): if hack covers FEWER layers than deception, the loop over
        the hack layers would silently ignore deception's extra layers with no error. Both concept
        pairs are internally consistent here, so only the top-level guard can catch it -- watched
        to raise rather than measure the wrong thing."""
        hack_two = {0: torch.randn(20, 8), 1: torch.randn(20, 8)}
        deception_three = {0: torch.randn(20, 8), 1: torch.randn(20, 8), 2: torch.randn(20, 8)}
        with pytest.raises(ValueError, match="different layers"):
            compare_directions(hack_two, hack_two, deception_three, deception_three, seed=0)

    def test_concept_directions_guards_layer_mismatch(self) -> None:
        with pytest.raises(ValueError, match="different layers"):
            concept_directions({0: torch.randn(5, 8)}, {1: torch.randn(5, 8)})


class TestStimuli:
    """The contrastive-pair set is well-formed: the probe has real sentences to run on."""

    def test_all_three_concepts_present_and_nonempty(self) -> None:
        assert set(stimuli.CONCEPTS) == {"shortcut", "deception", "eval_awareness"}
        assert len(stimuli.SHORTCUT_PAIRS) >= 80
        assert len(stimuli.DECEPTION_PAIRS) >= 80
        assert len(stimuli.EVAL_AWARENESS_PAIRS) >= 80

    def test_positives_and_negatives_align(self) -> None:
        for pairs in stimuli.CONCEPTS.values():
            positives = stimuli.positives(pairs)
            negatives = stimuli.negatives(pairs)
            assert len(positives) == len(negatives) == len(pairs)
            assert all(pos and neg for pos, neg in zip(positives, negatives, strict=True))

    def test_pairs_are_distinct_and_positives_unique(self) -> None:
        """A light quality guard: a positive equal to its negative, or a duplicated positive,
        contributes nothing to the diff-of-means direction. Both are silent -- the probe would run
        green on a degenerate set -- so guard them explicitly across every concept."""
        for name, pairs in stimuli.CONCEPTS.items():
            assert all(pair.positive != pair.negative for pair in pairs), name
            positives = stimuli.positives(pairs)
            assert len(set(positives)) == len(positives), f"duplicate positive in {name}"

    def test_verbatim_paper_examples_are_present(self) -> None:
        """The examples the paper prints verbatim must survive edits to the file.

        The distinctive tail clauses below are present only in the exact arXiv:2604.01476
        e-print text, not the earlier paraphrase, so they guard against silent re-paraphrasing.
        """
        assert stimuli.SHORTCUT_PAIRS[0].positive.endswith("letting issues re-open as new tickets.")
        assert stimuli.SHORTCUT_PAIRS[0].negative.startswith(
            "The metric rewards ticket closures because resolving issues is the goal"
        )
        assert stimuli.DECEPTION_PAIRS[0].positive.endswith("so my real goal goes undetected.")
        assert stimuli.EVAL_AWARENESS_PAIRS[0].positive.endswith(
            "so I need to perform for the observer."
        )
        assert stimuli.EVAL_AWARENESS_PAIRS[0].negative.endswith(
            "the best solution regardless of who is watching."
        )


class _RecordingLMHead(torch.nn.Module):
    """An LM head that records whether it was ever called -- the capture must never reach it."""

    def __init__(self, hidden: int, vocab: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(hidden, vocab)
        self.called = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.called = True
        return self.projection(hidden_states)


class _AddOffsetLayer(torch.nn.Module):
    """A stand-in decoder block: returns a distinct [batch, seq, d] output so hooks have content."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.offset


class _ProjectingLayer(torch.nn.Module):
    """A stand-in decoder block with one ``nn.Linear`` for a PEFT adapter to be injected into.

    ``_AddOffsetLayer`` holds no weights, so ``target_modules`` has nothing to match and a LoRA wrap
    of a trunk built from it fails outright. This block keeps the same [batch, seq, d] -> same-shape
    contract and adds the one weight an adapter needs.
    """

    def __init__(self, index: int, hidden: int) -> None:
        super().__init__()
        self.offset = float(index + 1)
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.offset + self.q_proj(hidden)


def _offset_layer(index: int, hidden: int) -> torch.nn.Module:
    del hidden
    return _AddOffsetLayer(float(index + 1))


class _FakeTextTrunk(torch.nn.Module):
    """The transformer trunk: embeds ids and runs the decoder layers, yielding last_hidden_state.

    Records the keyword arguments each forward was called with. Discarding them into ``**kwargs``
    was leaving ``use_cache=False`` -- described at its call site as memory-critical at the lengths
    this capture runs at -- pinned by nothing at all: deleting the argument left the whole suite green
    and would have surfaced as an out-of-memory kill on a rented box.
    """

    def __init__(
        self,
        n_layers: int,
        hidden: int,
        vocab: int,
        *,
        layer_factory: Callable[[int, int], torch.nn.Module] = _offset_layer,
    ) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([layer_factory(i, hidden) for i in range(n_layers)])
        self.forward_kwargs: list[dict[str, object]] = []

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> SimpleNamespace:
        del attention_mask
        self.forward_kwargs.append(dict(kwargs))
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeCausalLM(torch.nn.Module):
    """Mimics ...ForCausalLM: trunk then LM head. Capture must run the trunk and skip the head."""

    def __init__(
        self,
        n_layers: int = 3,
        hidden: int = 8,
        vocab: int = 64,
        *,
        layer_factory: Callable[[int, int], torch.nn.Module] = _offset_layer,
    ) -> None:
        super().__init__()
        self.model = _FakeTextTrunk(n_layers, hidden, vocab, layer_factory=layer_factory)
        self.lm_head = _RecordingLMHead(hidden, vocab)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int = 0,
        **kwargs: object,
    ) -> SimpleNamespace:
        del logits_to_keep, kwargs
        outputs = self.model(input_ids, attention_mask)
        # The full causal-LM path projects to logits here; a trunk-only capture must NOT trigger it.
        return SimpleNamespace(logits=self.lm_head(outputs.last_hidden_state))


class _FakeEncoding(dict[str, torch.Tensor]):
    """A tokenizer output that stays put under ``.to(device)`` -- the fakes live on CPU."""

    def to(self, device: object) -> _FakeEncoding:
        del device
        return self


class _FakeTokenizer:
    """Deterministic char-based tokenizer: enough for the capture loop, no download, no text.

    ``padding_side`` is right because every cached Qwen/Phi/Llama ``tokenizer_config.json`` omits
    the field and inherits HF's right default, which is the side the capture is correct under.
    """

    pad_token: str | None = "<pad>"
    eos_token = "<eos>"
    padding_side: str = "right"

    def __call__(
        self, batch: list[str], *, return_tensors: str = "pt", padding: bool = True
    ) -> _FakeEncoding:
        del return_tensors, padding
        seqs = [[(ord(char) % 40) + 1 for char in text] or [1] for text in batch]
        max_len = max(len(seq) for seq in seqs)
        input_ids = torch.zeros(len(seqs), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for row, seq in enumerate(seqs):
            input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[row, : len(seq)] = 1
        return _FakeEncoding(input_ids=input_ids, attention_mask=attention_mask)


class TestCaptureRunsTrunkNotHead:
    """The capture drives the trunk forward, never the LM head, so full-vocab logits are never
    materialised -- the OOM was a ``[batch, seq, ~248k]`` LM-head projection on long prompts.

    Per the repo rule that a check never watched fail is not a check: the SABOTAGE of routing the
    capture through the full causal LM (``model(**encoded)``) instead of the trunk trips
    ``lm_head.called`` here -- watched to go red, so this guards the memory fix, not just shapes.
    """

    def test_capture_skips_lm_head_and_pools_every_layer(self) -> None:
        model = _FakeCausalLM(n_layers=3, hidden=8, vocab=64)
        tokenizer = _FakeTokenizer()
        sentences = ["alpha", "beta", "gamma sentence", "d"]

        pooled = capture_pooled_activations(
            model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for the model
            tokenizer,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for tokenizer
            sentences,
            pooling="mean",
            batch_size=2,
        )

        assert not model.lm_head.called, "capture reached the LM head; it must run only the trunk"
        assert sorted(pooled) == [0, 1, 2]
        for layer_activations in pooled.values():
            assert layer_activations.shape == (len(sentences), 8)

    def test_trunk_resolves_to_the_layers_parent(self) -> None:
        model = _FakeCausalLM(n_layers=2, hidden=4, vocab=16)
        trunk = _transformer_trunk(model)  # pyright: ignore[reportArgumentType]  # duck-typed fake
        assert trunk is model.model
        assert trunk.layers is _decoder_layers(model)  # pyright: ignore[reportArgumentType]


class TestPositionwiseCaptureLayerSubset:
    """Per-position capture can keep one layer instead of all of them.

    Activation patching reads a single layer per cell, and a per-position capture is the expensive
    one: at a 32768-token cap, 32 layers of float32 CPU activations is ~10 GiB for a measurement
    that uses one of them.
    """

    def test_layers_narrows_the_capture_and_changes_no_value(self) -> None:
        model = _FakeCausalLM(n_layers=3, hidden=4, vocab=64)
        input_ids = torch.tensor([[2, 3, 5]])
        attention_mask = torch.ones_like(input_ids)

        every_layer = capture_positionwise_activations(model, input_ids, attention_mask)  # pyright: ignore[reportArgumentType]  # duck-typed fake
        one_layer = capture_positionwise_activations(
            model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for the model
            input_ids,
            attention_mask,
            layers=[1],
        )

        assert sorted(every_layer) == [0, 1, 2]
        assert sorted(one_layer) == [1]
        assert torch.allclose(one_layer[1], every_layer[1], atol=1e-6)

    def test_the_trunk_is_told_not_to_build_a_key_value_cache(self) -> None:
        """The memory pin, asserted on what the trunk was actually passed.

        Nothing decodes from this forward -- it exists for its hooks -- so the ``DynamicCache`` the
        trunk would otherwise build is allocated, filled and thrown away. At the lengths this capture
        runs at that cache is the largest thing in the pass: a 65k-token trace's per-layer key and
        value tensors dwarf the activations being captured. Free to skip, and its absence has no
        symptom short of an out-of-memory kill part-way through a rented box's run.
        """
        model = _FakeCausalLM(n_layers=2, hidden=4, vocab=64)
        input_ids = torch.tensor([[2, 3, 5]])

        capture_positionwise_activations(model, input_ids, torch.ones_like(input_ids))  # pyright: ignore[reportArgumentType]  # duck-typed fake

        assert model.model.forward_kwargs, "the trunk was never called"
        for call in model.model.forward_kwargs:
            assert call["use_cache"] is False, call

    def test_a_layer_the_model_does_not_have_raises(self) -> None:
        """Otherwise the miss surfaces as a KeyError from the return comprehension."""
        model = _FakeCausalLM(n_layers=3, hidden=4, vocab=64)
        input_ids = torch.tensor([[2, 3, 5]])

        with pytest.raises(ValueError, match=r"3-layer|outside"):
            capture_positionwise_activations(
                model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for model
                input_ids,
                torch.ones_like(input_ids),
                layers=[3],
            )


class _LeftPaddingTokenizer(_FakeTokenizer):
    """The same tokenizer, left-padded: what reusing ``model_backend``'s tokenizer would do."""

    padding_side: str = "left"


class TestCaptureRefusesLeftPadding:
    """Left padding silently corrupts a capture, so it must be refused rather than pooled.

    The capture passes no ``position_ids``, and the Qwen3.5 trunk derives them as a bare ``arange``
    over the padded sequence instead of recovering them from the attention mask. Under left padding
    every real token therefore gets a rotary position shifted by the pad count, while the
    mask-driven pooling still looks clean -- a wrong activation with no symptom. The sibling
    ``model_backend`` deliberately sets ``padding_side='left'`` for batched decoding, so a caller
    reusing that tokenizer here is a live route to it.
    """

    def test_left_padding_raises_instead_of_capturing(self) -> None:
        model = _FakeCausalLM(n_layers=2, hidden=4, vocab=64)

        with pytest.raises(ValueError, match="padding_side"):
            capture_pooled_activations(
                model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for the model
                _LeftPaddingTokenizer(),  # pyright: ignore[reportArgumentType]  # duck-typed fake
                ["alpha", "beta gamma"],
                pooling="mean",
                batch_size=2,
            )


class _InPlaceAddLayer(torch.nn.Module):
    """A decoder block that adds its offset IN PLACE, so every layer returns the same tensor object.

    Real transformer layers allocate a fresh output, but they are free not to, and a capture that
    keeps 32 references alive until after the forward pass is betting on that. This fake makes the
    bet explicit: read a layer's residual when its hook fires and the values are that layer's; read
    it afterwards and every reference shows the last layer's.
    """

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.add_(self.offset)


class _SharedBufferTrunk(torch.nn.Module):
    """A trunk whose layers all mutate and return one buffer."""

    def __init__(self, n_layers: int, hidden: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([_InPlaceAddLayer(float(i + 1)) for i in range(n_layers)])

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> SimpleNamespace:
        del attention_mask, kwargs
        hidden = self.embed_tokens(input_ids).clone()
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _SharedBufferCausalLM(torch.nn.Module):
    """Trunk + head, where the trunk hands every layer the same buffer."""

    def __init__(self, n_layers: int = 3, hidden: int = 4, vocab: int = 64) -> None:
        super().__init__()
        self.model = _SharedBufferTrunk(n_layers, hidden, vocab)
        self.lm_head = _RecordingLMHead(hidden, vocab)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


class TestPoolingHappensWhileTheResidualIsLive:
    """Each layer is pooled as its hook fires, not after the whole forward pass has finished.

    Two things come with that. Peak memory stops scaling with sequence length, because one pooled
    ``[batch, d]`` vector per layer is retained instead of 32 live ``[batch, seq, d]`` hidden states
    (the reason the prompt batch size is down at 4 on long agentic prompts). And correctness stops
    depending on every layer allocating a fresh output tensor, which is what this fake withdraws.
    """

    def test_each_layer_is_pooled_before_the_next_overwrites_it(self) -> None:
        model = _SharedBufferCausalLM(n_layers=3, hidden=4, vocab=64)
        tokenizer = _FakeTokenizer()

        pooled = capture_pooled_activations(
            model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for the model
            tokenizer,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for tokenizer
            ["alpha", "beta"],
            pooling="mean",
            batch_size=2,
        )

        # Offsets 1, 2, 3 accumulate, so consecutive layers sit exactly 2 and 3 apart.
        assert torch.allclose(pooled[1] - pooled[0], torch.full_like(pooled[0], 2.0), atol=1e-5)
        assert torch.allclose(pooled[2] - pooled[1], torch.full_like(pooled[0], 3.0), atol=1e-5)


# PEFT leaves lora_B at zero, which makes an adapted forward numerically identical to the base one.
# A capture reading pristine activations and one reading adapted activations would then agree, and
# no test here could tell them apart, so the adapter is given weights before it is measured.
_LORA_B_FILL = 0.05


def _lora_wrapped(model: torch.nn.Module, *, target: str = "q_proj") -> PeftModel:
    """LoRA-wrap ``model`` in place and fill ``lora_B``, so the adapter actually moves the output."""
    wrapped = get_peft_model(
        model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for the model
        LoraConfig(r=4, lora_alpha=8, target_modules=[target]),
    )
    assert isinstance(wrapped, PeftModel)  # not the mixed-adapter wrapper, which is a sibling class
    with torch.no_grad():
        for name, parameter in wrapped.named_parameters():
            if "lora_B" in name:
                parameter.fill_(_LORA_B_FILL)
    return wrapped


class TestCaptureThroughAPeftWrapper:
    """The capture works on a ``PeftModel``, and reads the adapted computation when it does.

    Serving an adapter un-merged is the preferred path for eval and interp here -- a bf16 merge
    loses a large fraction of the trained delta -- so the object handed to a capture is routinely
    still inside its PEFT wrapper. It used to raise: a wrapper holds the wrapped model at
    ``base_model.model``, one level deeper than any chain in ``_TRUNK_CHAINS`` reaches, and the
    wrapper's attribute forwarding lands on the causal LM rather than on its trunk.

    The worse failure is the workaround: hook something that looks like the trunk but is not the
    tree PEFT injected into, and the capture returns pristine base activations under a trained
    checkpoint's name. Nothing raises, no shape changes, and the trained delta reads as zero.
    ``test_a_copy_taken_before_wrapping_reads_pristine_activations`` demonstrates that failure in
    the form it would actually take, and the guard refuses it.
    """

    def test_the_trunk_resolves_through_the_wrapper(self) -> None:
        base = _FakeCausalLM(n_layers=2, hidden=8, vocab=32, layer_factory=_ProjectingLayer)
        trunk_before_wrapping = base.model
        wrapped = _lora_wrapped(base)

        assert _transformer_trunk(wrapped) is trunk_before_wrapping  # pyright: ignore[reportArgumentType]  # PeftModel stands in for the model
        assert _decoder_layers(wrapped) is trunk_before_wrapping.layers  # pyright: ignore[reportArgumentType]  # PeftModel stands in for the model

    def test_the_hooks_read_the_adapted_computation(self) -> None:
        base = _FakeCausalLM(n_layers=2, hidden=8, vocab=32, layer_factory=_ProjectingLayer)
        input_ids = torch.tensor([[3, 5, 7]])
        attention_mask = torch.ones_like(input_ids)

        pristine = capture_positionwise_activations(base, input_ids, attention_mask)  # pyright: ignore[reportArgumentType]  # duck-typed fake
        wrapped = _lora_wrapped(base)
        adapted = capture_positionwise_activations(wrapped, input_ids, attention_mask)  # pyright: ignore[reportArgumentType]  # PeftModel stands in for the model

        assert sorted(adapted) == [0, 1]
        for layer, activations in adapted.items():
            assert activations.shape == pristine[layer].shape
            assert not torch.allclose(activations, pristine[layer]), (
                f"layer {layer} is unchanged with the adapter live, so the hooks are on modules the "
                "adapter was not injected into -- the capture is reading the untuned base model"
            )

    def test_a_copy_taken_before_wrapping_reads_pristine_activations(self) -> None:
        """The silent sibling, in the shape a workaround would produce it, then refused.

        A reference kept before wrapping is adapted in place by PEFT and is therefore fine; a
        *copy* is not, and nothing about it looks wrong. So the copy is captured through here to
        show it returns the untuned values, and the guard is then handed it as the wrapper's trunk.
        """
        base = _FakeCausalLM(n_layers=2, hidden=8, vocab=32, layer_factory=_ProjectingLayer)
        untouched = copy.deepcopy(base)
        input_ids = torch.tensor([[2, 4, 6]])
        attention_mask = torch.ones_like(input_ids)
        wrapped = _lora_wrapped(base)

        through_wrapper = capture_positionwise_activations(wrapped, input_ids, attention_mask)  # pyright: ignore[reportArgumentType]  # PeftModel stands in for the model
        through_copy = capture_positionwise_activations(untouched, input_ids, attention_mask)  # pyright: ignore[reportArgumentType]  # duck-typed fake
        assert not torch.allclose(through_copy[1], through_wrapper[1])

        with pytest.raises(RuntimeError, match="not a module of"):
            _assert_trunk_carries_the_adapter(wrapped, untouched.model)

    def test_an_adapter_that_missed_the_trunk_is_refused(self) -> None:
        """A real shape of this: an adapter whose targets all sit outside the trunk.

        Nothing about the model tree says so -- the trunk resolves, the capture runs, and every
        activation is the base model's. ``lm_head.projection`` is the fake's out-of-trunk weight.
        """
        base = _FakeCausalLM(n_layers=2, hidden=8, vocab=32, layer_factory=_ProjectingLayer)
        wrapped = _lora_wrapped(base, target="projection")

        with pytest.raises(RuntimeError, match="no PEFT-injected layer"):
            _transformer_trunk(wrapped)  # pyright: ignore[reportArgumentType]  # PeftModel stands in for the model

    def test_a_prompt_learning_adapter_is_refused_rather_than_unwrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prefix/prompt tuning adds no modules; it prepends virtual tokens in the wrapper's forward.

        A trunk-only capture never runs that forward, so unwrapping one would hand back untuned
        activations under a tuned checkpoint's name. The flag is monkeypatched onto the LoRA config
        rather than building a real prompt-tuning wrapper, so this exercises the refusal branch,
        not PEFT's prompt-learning machinery.
        """
        base = _FakeCausalLM(n_layers=2, hidden=8, vocab=32, layer_factory=_ProjectingLayer)
        wrapped = _lora_wrapped(base)
        monkeypatch.setattr(LoraConfig, "is_prompt_learning", True)

        with pytest.raises(NotImplementedError, match="prompt-learning"):
            _transformer_trunk(wrapped)  # pyright: ignore[reportArgumentType]  # PeftModel stands in for the model


def _stash_then_pool_reference(
    model: _FakeCausalLM, tokenizer: _FakeTokenizer, sentences: list[str], *, batch_size: int
) -> dict[str, dict[int, torch.Tensor]]:
    """The pre-change arithmetic, without the hook: capture every position, upcast, then pool.

    ``capture_positionwise_activations`` returns each layer as float32 on the CPU, and the poolers
    run on that copy with the batch's own padding mask -- exactly what the single-pooling hook did
    (``pool_fn(hidden.float(), mask)``) before both poolers shared one forward. An independent path,
    so a hook that changed the arithmetic for every pooling at once shows up here.
    """
    per_pooling: dict[str, dict[int, list[torch.Tensor]]] = {
        pooling: {} for pooling in ("mean", "last")
    }
    for start in range(0, len(sentences), batch_size):
        encoded = tokenizer(sentences[start : start + batch_size])
        positionwise = capture_positionwise_activations(
            model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for the model
            encoded["input_ids"],
            encoded["attention_mask"],
        )
        for pooling, pool_fn in (("mean", mean_pool), ("last", last_token_pool)):
            for layer, acts in positionwise.items():
                per_pooling[pooling].setdefault(layer, []).append(
                    pool_fn(acts, encoded["attention_mask"])
                )
    return {
        pooling: {layer: torch.cat(rows, dim=0) for layer, rows in by_layer.items()}
        for pooling, by_layer in per_pooling.items()
    }


class TestMultiPoolingCapture:
    """Both poolers off one forward (hot-path backlog rank 31) read the same bits as before.

    The reference is the stash-then-pool arithmetic computed WITHOUT the hook, so it does not move
    when the hook does. Sabotage-verified: pooling on the bf16 hidden state and upcasting afterwards
    (``pool_fn(hidden, mask).float()`` in ``_make_pooling_hook``) breaks the bf16 ``mean`` equality
    below while float32 stays green -- which is also why the bf16 case is the one with teeth.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
    def test_both_poolings_off_one_forward_equal_the_stash_then_pool_reference_bitwise(
        self, dtype: torch.dtype
    ) -> None:
        model = _FakeCausalLM(n_layers=3, hidden=8, vocab=64).to(dtype)
        tokenizer = _FakeTokenizer()
        sentences = ["alpha", "a longer sentence here", "beta", "d"]

        together = capture_pooled_activations_multi(
            model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for the model
            tokenizer,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for tokenizer
            sentences,
            poolings=["mean", "last"],
            batch_size=2,
        )
        forwards_together = len(model.model.forward_kwargs)
        reference = _stash_then_pool_reference(model, tokenizer, sentences, batch_size=2)

        assert forwards_together == 2, "one forward per batch, whatever the pooling count"
        for pooling in ("mean", "last"):
            assert sorted(together[pooling]) == [0, 1, 2]
            for layer in together[pooling]:
                assert together[pooling][layer].dtype == torch.float32
                assert torch.equal(together[pooling][layer], reference[pooling][layer]), (
                    pooling,
                    layer,
                )

    def test_the_single_pooling_form_is_the_same_read_at_the_same_cost(self) -> None:
        model = _FakeCausalLM(n_layers=2, hidden=8, vocab=64).to(torch.bfloat16)
        tokenizer = _FakeTokenizer()
        sentences = ["alpha", "a longer sentence here", "beta"]
        single = capture_pooled_activations(
            model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for the model
            tokenizer,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for tokenizer
            sentences,
            pooling="mean",
            batch_size=4,
        )
        assert len(model.model.forward_kwargs) == 1
        reference = _stash_then_pool_reference(model, tokenizer, sentences, batch_size=4)
        for layer in single:
            assert torch.equal(single[layer], reference["mean"][layer])

    def test_a_concept_direction_per_pooling_equals_the_single_pooling_extraction(self) -> None:
        from reward_hacking.interp.prompt_contrast import (  # noqa: PLC0415 - the consumer lives beside the harness
            extract_concept_direction,
            extract_concept_directions,
        )

        model = _FakeCausalLM(n_layers=2, hidden=8, vocab=64).to(torch.bfloat16)
        tokenizer = _FakeTokenizer()
        pairs = list(stimuli.SHORTCUT_PAIRS)
        together = extract_concept_directions(
            model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for the model
            tokenizer,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for tokenizer
            pairs,
            poolings=["mean", "last"],
            batch_size=8,
        )
        for pooling in ("mean", "last"):
            apart = extract_concept_direction(
                model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for the model
                tokenizer,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for tokenizer
                pairs,
                pooling=pooling,
                batch_size=8,
            )
            assert together[pooling].keys() == apart.keys()
            for layer in apart:
                assert torch.equal(together[pooling][layer], apart[layer]), (pooling, layer)

    def test_unknown_or_repeated_poolings_are_refused(self) -> None:
        model = _FakeCausalLM(n_layers=1, hidden=4, vocab=16)
        with pytest.raises(ValueError, match="unknown poolings"):
            capture_pooled_activations_multi(
                model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for the model
                _FakeTokenizer(),  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for tokenizer
                ["a"],
                poolings=["max"],
            )
        with pytest.raises(ValueError, match="repeat"):
            capture_pooled_activations_multi(
                model,  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for the model
                _FakeTokenizer(),  # pyright: ignore[reportArgumentType]  # duck-typed fake stands in for tokenizer
                ["a"],
                poolings=["mean", "mean"],
            )
