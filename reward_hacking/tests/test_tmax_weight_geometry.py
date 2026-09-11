"""Weight-space geometry (``reward_hacking/tmax/weight_geometry.py`` and its siblings) on toys.

Everything runs on CPU over a two-layer toy in the real Qwen3.5 key layout (one DeltaNet layer, one
full-attention layer, tiny widths). The claims pinned: every key of the layout classifies and an
unknown one is refused; the delta is taken against the base rounded to the release's storage dtype;
the spectral statistics hit their closed forms on rank-one and identity matrices and the Gram route
agrees with exact singular values; the double-rounding floor is computed and identical on re-draw,
reads near zero for an exactly representable base (the mechanism needs a hidden offset, which the
``offset`` floor supplies and is positive under), a pair of deltas built out of the floor's own
perturbations reads as *not* clearing it while a genuinely shared direction does; the
sparsity-matched noise scale reproduces the target flip fraction; token counts read the rollout rows
and the residualised ranking finds a planted outlier the raw ranking hides; and a run over real
files writes every table, resumes without recomputing or duplicating, and renders.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from games.eval_model import FullWeightsSource, resolve_full_weights
from reward_hacking.tmax import weight_delta as wd
from reward_hacking.tmax import weight_floor as wf
from reward_hacking.tmax import weight_geometry as wg
from reward_hacking.tmax.amplify import SAFETENSORS_INDEX_FILENAME
from reward_hacking.tmax.token_frequency import (
    FrequencyRole,
    TokenCounts,
    TokenRowSite,
    count_rollout_tokens,
    count_tokens_in_file,
    residualise_on_log_frequency,
    rollout_files,
    token_row_frame,
    token_strings,
    with_token_strings,
)
from reward_hacking.tmax.weight_geometry_tables import (
    load_pairs,
    load_tensors,
    markdown_table,
    movement_by,
    relationship_by,
    render_readout,
)
from reward_hacking.tmax.weight_layout import (
    BASE_SPECTRA_FILENAME,
    GATE_HEAD_MODULES,
    GATE_HEADS_FILENAME,
    LAYER_SUFFIX_MODULES,
    PAIRS_FILENAME,
    TENSORS_FILENAME,
    ModuleClass,
    ModuleFamily,
    classify_tensor,
    token_rows_filename,
)
from reward_hacking.tmax.weight_spectra import (
    GRAM_MIN_DIM,
    as_matrix,
    singular_values,
    spectral_shape,
)

if TYPE_CHECKING:
    from pathlib import Path

LM = "model.language_model"
HIDDEN = 8
INTERMEDIATE = 16
VOCAB = 32
HEADS = 2
EMBED = f"{LM}.embed_tokens.weight"
LM_HEAD = "lm_head.weight"
CPU = torch.device("cpu")


def toy_base(seed: int = 7) -> dict[str, torch.Tensor]:
    """Two layers in the real key layout: layer 0 is DeltaNet, layer 1 full attention."""
    generator = torch.Generator().manual_seed(seed)

    def matrix(rows: int, cols: int) -> torch.Tensor:
        return (torch.randn(rows, cols, generator=generator) * 0.05).to(torch.bfloat16)

    def vector(n: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return (1.0 + 0.1 * torch.randn(n, generator=generator)).to(dtype)

    linear = f"{LM}.layers.0"
    full = f"{LM}.layers.1"
    tensors = {
        EMBED: matrix(VOCAB, HIDDEN),
        LM_HEAD: matrix(VOCAB, HIDDEN),
        f"{LM}.norm.weight": vector(HIDDEN),
        f"{linear}.input_layernorm.weight": vector(HIDDEN),
        f"{linear}.post_attention_layernorm.weight": vector(HIDDEN),
        f"{linear}.linear_attn.in_proj_qkv.weight": matrix(2 * HIDDEN, HIDDEN),
        f"{linear}.linear_attn.in_proj_z.weight": matrix(HIDDEN, HIDDEN),
        f"{linear}.linear_attn.in_proj_a.weight": matrix(HEADS, HIDDEN),
        f"{linear}.linear_attn.in_proj_b.weight": matrix(HEADS, HIDDEN),
        f"{linear}.linear_attn.out_proj.weight": matrix(HIDDEN, HIDDEN),
        f"{linear}.linear_attn.conv1d.weight": matrix(2 * HIDDEN, 4).reshape(2 * HIDDEN, 1, 4),
        # The base keeps these two at float32; the release stores them at bfloat16.
        f"{linear}.linear_attn.A_log": vector(HEADS, torch.float32),
        f"{linear}.linear_attn.norm.weight": vector(4, torch.float32),
        f"{linear}.linear_attn.dt_bias": vector(HEADS),
        f"{linear}.mlp.gate_proj.weight": matrix(INTERMEDIATE, HIDDEN),
        f"{linear}.mlp.up_proj.weight": matrix(INTERMEDIATE, HIDDEN),
        f"{linear}.mlp.down_proj.weight": matrix(HIDDEN, INTERMEDIATE),
        f"{full}.input_layernorm.weight": vector(HIDDEN),
        f"{full}.post_attention_layernorm.weight": vector(HIDDEN),
        f"{full}.self_attn.q_proj.weight": matrix(HIDDEN, HIDDEN),
        f"{full}.self_attn.k_proj.weight": matrix(4, HIDDEN),
        f"{full}.self_attn.v_proj.weight": matrix(4, HIDDEN),
        f"{full}.self_attn.o_proj.weight": matrix(HIDDEN, HIDDEN),
        f"{full}.self_attn.q_norm.weight": vector(4),
        f"{full}.self_attn.k_norm.weight": vector(4),
        f"{full}.mlp.gate_proj.weight": matrix(INTERMEDIATE, HIDDEN),
        f"{full}.mlp.up_proj.weight": matrix(INTERMEDIATE, HIDDEN),
        f"{full}.mlp.down_proj.weight": matrix(HIDDEN, INTERMEDIATE),
    }
    return {name: tensor.contiguous() for name, tensor in tensors.items()}


def _name_seed(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "big")


def toy_rl(
    base: dict[str, torch.Tensor], *, scale: float, noise: float, seed: int
) -> dict[str, torch.Tensor]:
    """``round_bf16(base + scale * D + noise)`` with ``D`` a fixed rank-one direction per matrix.

    The direction is seeded by the tensor name only, so two RL checkpoints built with different
    ``scale`` share it (the "same direction, further along" case); ``noise`` is fresh per ``seed``.
    """
    out: dict[str, torch.Tensor] = {}
    for name, tensor in base.items():
        direction_generator = torch.Generator().manual_seed(_name_seed(name))
        noise_generator = torch.Generator().manual_seed(seed)
        base32 = tensor.to(torch.float32)
        if base32.ndim >= 2:
            flat = base32.reshape(base32.shape[0], -1)
            u = torch.randn(flat.shape[0], 1, generator=direction_generator)
            v = torch.randn(1, flat.shape[1], generator=direction_generator)
            direction = (u @ v).reshape(base32.shape)
        else:
            direction = torch.randn(base32.shape, generator=direction_generator)
        jitter = torch.randn(base32.shape, generator=noise_generator)
        perturbed = base32 + scale * direction + noise * jitter
        out[name] = perturbed.to(torch.bfloat16).contiguous()
    return out


def write_base(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    """A sharded base with a vision tensor the release lacks, like the hub's."""
    path.mkdir(parents=True)
    names = sorted(tensors)
    shards = {
        "model-00001-of-00003.safetensors": names[: len(names) // 2],
        "model-00002-of-00003.safetensors": names[len(names) // 2 :],
    }
    visual = {"model.visual.blocks.0.attn.qkv.weight": torch.ones(2, 2, dtype=torch.bfloat16)}
    for shard, members in shards.items():
        save_file({name: tensors[name] for name in members}, str(path / shard))
    save_file(visual, str(path / "model-00003-of-00003.safetensors"))
    weight_map = {name: shard for shard, members in shards.items() for name in members}
    weight_map.update(dict.fromkeys(visual, "model-00003-of-00003.safetensors"))
    (path / SAFETENSORS_INDEX_FILENAME).write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps({"vision_config": {}}))
    return path


def write_tokenizer(path: Path) -> Path:
    vocab = {f"tok{i}": i for i in range(VOCAB)}
    Tokenizer(WordLevel(vocab, unk_token="tok0")).save(str(path))  # noqa: S106 - a token marker
    return path


def write_rl(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    path.mkdir(parents=True)
    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"vision_config": {}}))
    write_tokenizer(path / "tokenizer.json")
    return path


def write_rollouts(path: Path) -> Path:
    """Two fragments of two rows each, with the fields the count must skip over."""
    rows = [
        {
            "step": 1,
            "prompt_tokens": [1, 2, 2, 3],
            "response_tokens": [3, 4, 5],
            "logprobs": [-0.1],
        },
        {"step": 1, "prompt_tokens": [1], "response_tokens": [5, 5], "request_info": {"x": "]["}},
        {"step": 2, "prompt_tokens": [7, 7, 7], "response_tokens": [], "logprobs": []},
        {"step": 2, "prompt_tokens": [], "response_tokens": [8], "ground_truth": "[1, 2]"},
    ]
    for index, fragment_rows in enumerate((rows[:2], rows[2:])):
        fragment = path / f"frag_{index}" / "rollouts"
        fragment.mkdir(parents=True)
        separators = (",", ":") if index else None
        lines = [json.dumps(row, separators=separators) for row in fragment_rows]
        (fragment / f"frag_{index}_rollouts_000000.jsonl").write_text("\n".join(lines) + "\n")
    return path


def bf16_ulp(value: float) -> float:
    """Spacing of bfloat16 around ``value`` (eight significand bits, seven stored)."""
    return 2.0 ** (math.floor(math.log2(abs(value))) - 7)


def perturbed_rl(base: torch.Tensor, sigma: float, side: str, name: str) -> torch.Tensor:
    """The floor's own rms-kind perturbation for one side, applied to the base."""
    base32 = base.to(torch.float32)
    seed = wf.perturbation_seed(name, side, 0, wf.FLOOR_KIND_RMS)
    return (base32 + wf.rounded_perturbation(base32, sigma, torch.bfloat16, seed)).to(
        torch.bfloat16
    )


class TestClassification:
    def test_every_toy_key_classifies_and_covers_every_module(self) -> None:
        sites = [classify_tensor(name) for name in toy_base()]
        assert {site.module for site in sites} == set(ModuleClass)
        assert {site.family for site in sites} == set(ModuleFamily)
        assert {site.layer for site in sites} == {None, 0, 1}

    def test_every_layout_suffix_is_placed(self) -> None:
        for suffix, module in LAYER_SUFFIX_MODULES.items():
            site = classify_tensor(f"{LM}.layers.13.{suffix}")
            assert (site.layer, site.module) == (13, module)

    def test_unknown_keys_are_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown module suffix"):
            classify_tensor(f"{LM}.layers.0.linear_attn.mystery.weight")
        with pytest.raises(ValueError, match=r"outside the Qwen3\.5"):
            classify_tensor("model.layers.0.mlp.down_proj.weight")

    def test_token_row_filenames_are_distinct_per_checkpoint_and_module(self) -> None:
        names = {
            token_rows_filename(label, module)
            for label in ("allenai/tmax-9b@step_200", "allenai/tmax-9b@step_500")
            for module in (ModuleClass.EMBED_TOKENS, ModuleClass.LM_HEAD)
        }
        assert len(names) == 4
        assert all("/" not in name and "@" not in name for name in names)


class TestReductions:
    def test_chunked_float64_reductions_match_exact_values_across_chunk_edges(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunks of 1000 elements over a 70x50 matrix: the boundaries fall mid-row and mid-tensor."""
        monkeypatch.setattr(wd, "REDUCTION_CHUNK_ELEMENTS", 1000)
        generator = torch.Generator().manual_seed(9)
        a = torch.randn(70, 50, generator=generator)
        b = torch.randn(70, 50, generator=generator)
        exact_a = a.double()
        exact_b = b.double()
        assert wd.frobenius_norm(a) == pytest.approx(float(exact_a.norm()), rel=1e-12)
        assert wd.sum_of_squares(a) == pytest.approx(float(exact_a.square().sum()), rel=1e-12)
        assert wd.inner_product(a, b) == pytest.approx(float((exact_a * exact_b).sum()), rel=1e-12)
        assert torch.allclose(wd.row_norms(a), exact_a.norm(dim=1), rtol=1e-12, atol=0)
        with pytest.raises(ValueError, match="inner product of shapes"):
            wd.inner_product(a, a[:1])
        with pytest.raises(ValueError, match="row norms"):
            wd.row_norms(a.reshape(-1))


class TestAlignedDelta:
    def test_the_base_is_rounded_to_the_release_dtype_first(self) -> None:
        base = torch.tensor([1.001, 2.0, -0.3333], dtype=torch.float32)
        rl = base.to(torch.bfloat16)
        delta = wd.aligned_delta(base, rl)
        assert torch.equal(delta.delta32, torch.zeros(3, device=CPU))
        assert delta.storage_dtype == torch.bfloat16
        assert delta.base_storage_dtype == torch.float32

    def test_shape_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="differ in shape"):
            wd.aligned_delta(torch.zeros(2, 2), torch.zeros(4, dtype=torch.bfloat16))

    def test_one_ulp_moves_count_adjacent_values_only(self) -> None:
        base = torch.tensor([1.0, 1.0, -1.0, 0.5, 2.0], dtype=torch.bfloat16)
        rl = torch.tensor(
            [1.0 + bf16_ulp(1.0), 1.0 + 2 * bf16_ulp(1.0), -1.0 - bf16_ulp(1.0), 0.5, -2.0],
            dtype=torch.bfloat16,
        )
        delta = wd.aligned_delta(base, rl)
        assert wd.one_ulp_moves(delta.base32, delta.delta32) == 2

    def test_storage_spacing_is_one_ulp(self) -> None:
        base32 = torch.tensor([1.0, 1.5, 0.25, -3.0, 0.0])
        spacing = wf.storage_spacing(base32, torch.bfloat16)
        assert spacing.tolist() == [
            bf16_ulp(1.0),
            bf16_ulp(1.5),
            bf16_ulp(0.25),
            bf16_ulp(3.0),
            0.0,
        ]

    def test_movement_statistics(self) -> None:
        base = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.bfloat16)
        rl = torch.tensor([[1.0 + bf16_ulp(1.0), 2.0], [4.0, 4.0]], dtype=torch.bfloat16)
        movement = wd.tensor_movement(wd.aligned_delta(base, rl))
        assert movement.n_elements == 4
        assert movement.n_changed == 2
        assert movement.changed_fraction == 0.5
        assert movement.n_one_ulp == 1
        assert movement.one_ulp_share_of_changed == 0.5
        assert movement.max_abs_delta == 4.0
        assert movement.storage_dtype == "bfloat16"
        assert movement.relative_delta == pytest.approx(
            math.hypot(bf16_ulp(1.0), 4.0) / math.sqrt(1 + 4 + 16 + 64)
        )

    def test_a_pure_shrink_has_cosine_minus_one_with_the_base(self) -> None:
        base = torch.randn(64, 64).to(torch.bfloat16)
        rl = (base.to(torch.float32) * 0.5).to(torch.bfloat16)
        movement = wd.tensor_movement(wd.aligned_delta(base, rl))
        assert movement.cosine_delta_base == pytest.approx(-1.0, abs=1e-3)


class TestSpectralShape:
    def test_rank_one_has_unit_ranks_and_all_energy_on_top(self) -> None:
        shape = spectral_shape(
            torch.outer(torch.arange(1.0, 9.0, device=CPU), torch.arange(1.0, 5.0, device=CPU))
        )
        assert shape.method == "svdvals_float64"
        assert shape.stable_rank == pytest.approx(1.0)
        assert shape.effective_rank == pytest.approx(1.0, abs=1e-6)
        assert shape.sv_entropy_normalized == pytest.approx(0.0, abs=1e-6)
        assert shape.top1_energy == pytest.approx(1.0)
        assert shape.top64_energy == pytest.approx(1.0)
        assert len(shape.singular_values_top) == 4

    def test_identity_through_the_gram_route_has_full_rank(self) -> None:
        n = GRAM_MIN_DIM + 8
        shape = spectral_shape(torch.eye(n, device=CPU), top_k=10)
        assert shape.method == "gram_eigvalsh_float64"
        assert shape.stable_rank == pytest.approx(n)
        assert shape.effective_rank == pytest.approx(n, rel=1e-9)
        assert shape.sv_entropy_normalized == pytest.approx(1.0)
        assert shape.top1_energy == pytest.approx(1 / n)
        assert shape.top64_energy == pytest.approx(64 / n)
        assert len(shape.singular_values_top) == 10

    def test_gram_agrees_with_exact_singular_values_on_a_tall_matrix(self) -> None:
        generator = torch.Generator().manual_seed(1)
        matrix = torch.randn(3 * GRAM_MIN_DIM, GRAM_MIN_DIM, generator=generator)
        via_gram, method = singular_values(matrix)
        assert method == "gram_eigvalsh_float64"
        exact = torch.linalg.svdvals(matrix.to(torch.float64))
        assert torch.allclose(via_gram, exact, rtol=1e-6, atol=1e-6)
        via_gram_wide, _ = singular_values(matrix.T)
        assert torch.allclose(via_gram_wide, exact, rtol=1e-6, atol=1e-6)

    def test_zero_matrix_reports_undefined_shape_not_numbers(self) -> None:
        shape = spectral_shape(torch.zeros(6, 5, device=CPU))
        assert shape.frobenius == 0.0
        assert shape.stable_rank is None
        assert shape.effective_rank is None
        assert shape.top1_energy is None

    def test_as_matrix_flattens_conv_and_rejects_vectors(self) -> None:
        conv = as_matrix(torch.zeros(16, 1, 4, device=CPU))
        assert conv is not None
        assert tuple(conv.shape) == (16, 4)
        assert as_matrix(torch.zeros(16, device=CPU)) is None


@pytest.fixture
def unit_base() -> torch.Tensor:
    """A bfloat16 base whose entries sit near one, so an ulp is about 2^-7."""
    values = 1.0 + 0.25 * torch.randn(256, 256, generator=torch.Generator().manual_seed(3))
    return values.to(torch.bfloat16)


class TestDoubleRoundingFloor:
    NAME = f"{LM}.layers.0.mlp.up_proj.weight"

    def zero_pair(self, unit_base: torch.Tensor) -> wf.DeltaPair:
        base32 = unit_base.to(torch.float32)
        delta = wd.AlignedDelta(base32, torch.zeros_like(base32), torch.bfloat16, torch.bfloat16)
        return wf.DeltaPair(name=self.NAME, a=delta, b=delta)

    def test_draws_are_deterministic_in_the_tensor_name(self, unit_base: torch.Tensor) -> None:
        base32 = unit_base.to(torch.float32)
        seed_a = wf.perturbation_seed(self.NAME, "a", 0, wf.FLOOR_KIND_RMS)
        seed_b = wf.perturbation_seed(self.NAME, "b", 0, wf.FLOOR_KIND_RMS)
        first = wf.rounded_perturbation(base32, 1e-3, torch.bfloat16, seed_a)
        again = wf.rounded_perturbation(base32, 1e-3, torch.bfloat16, seed_a)
        other = wf.rounded_perturbation(base32, 1e-3, torch.bfloat16, seed_b)
        assert torch.equal(first, again)
        assert not torch.equal(first, other)
        assert wf.perturbation_seed("x", "a", 0, "rms") != wf.perturbation_seed("y", "a", 0, "rms")

    def test_the_specified_floor_is_near_zero_for_a_representable_base(
        self, unit_base: torch.Tensor
    ) -> None:
        """A bf16 base sits at the centre of its rounding interval, so flips are symmetric."""
        draws = wf.draw_floor(
            self.zero_pair(unit_base), kind=wf.FLOOR_KIND_RMS, scales=(1e-3, 1e-3), n_pairs=3
        )
        assert len(draws.cosines) == 3
        assert all(c is not None and abs(c) < 0.05 for c in draws.cosines)
        assert draws.max_cosine == max(c for c in draws.cosines if c is not None)

    def test_the_offset_floor_is_positive_where_rounding_dominates(
        self, unit_base: torch.Tensor
    ) -> None:
        """A shared sub-ulp offset biases both draws' flips the same way: the literature's case."""
        draws = wf.draw_floor(
            self.zero_pair(unit_base), kind=wf.FLOOR_KIND_OFFSET, scales=(1e-3, 1e-3), n_pairs=3
        )
        assert all(c is not None and c > 0.15 for c in draws.cosines)

    def test_unknown_floor_kinds_are_refused(self, unit_base: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="unknown floor kind"):
            wf.draw_floor(self.zero_pair(unit_base), kind="median", scales=(1.0, 1.0), n_pairs=1)

    def test_a_delta_equal_to_the_floors_noise_does_not_clear_it(
        self, unit_base: torch.Tensor
    ) -> None:
        """Deltas that ARE the floor's own perturbation pair land exactly on it, so no clearance."""
        sigma = 1e-3
        relationship = wf.delta_relationship(
            name=self.NAME,
            a=wd.aligned_delta(unit_base, perturbed_rl(unit_base, sigma, "a", self.NAME)),
            b=wd.aligned_delta(unit_base, perturbed_rl(unit_base, sigma, "b", self.NAME)),
            n_floor_pairs=1,
            floor_rms=(sigma, sigma),
        )
        assert relationship.cosine is not None
        assert relationship.floor_rms_matched.max_cosine == relationship.cosine
        assert relationship.clears_floor is False
        record = relationship.record()
        margin = cast("float", record["margin_over_floor"])
        assert margin <= 0.0
        assert record["floor_offset_max"] is not None
        json.dumps(record, allow_nan=False)

    def test_independent_noise_deltas_sit_within_the_floors_range(
        self, unit_base: torch.Tensor
    ) -> None:
        """Fresh independent rounding noise, left to the default rms matching, reads as unrelated."""
        base32 = unit_base.to(torch.float32)
        noise_a = torch.randn(base32.shape, generator=torch.Generator().manual_seed(11))
        noise_b = torch.randn(base32.shape, generator=torch.Generator().manual_seed(12))
        relationship = wf.delta_relationship(
            name=self.NAME,
            a=wd.aligned_delta(unit_base, (base32 + 1e-3 * noise_a).to(torch.bfloat16)),
            b=wd.aligned_delta(unit_base, (base32 + 1e-3 * noise_b).to(torch.bfloat16)),
            n_floor_pairs=4,
        )
        assert relationship.cosine is not None
        assert relationship.floor_max is not None
        floors = [
            c
            for kind in wf.GATING_FLOOR_KINDS
            for c in relationship.floors[kind].cosines
            if c is not None
        ]
        assert min(floors) - 0.05 < relationship.cosine < max(floors) + 0.05

    def test_a_shared_direction_clears_the_floor(self, unit_base: torch.Tensor) -> None:
        base32 = unit_base.to(torch.float32)
        direction = torch.randn(base32.shape, generator=torch.Generator().manual_seed(5)) * 0.05
        noise = torch.randn(base32.shape, generator=torch.Generator().manual_seed(6)) * 1e-3
        relationship = wf.delta_relationship(
            name=self.NAME,
            a=wd.aligned_delta(unit_base, (base32 + direction).to(torch.bfloat16)),
            b=wd.aligned_delta(unit_base, (base32 + 2 * direction + noise).to(torch.bfloat16)),
        )
        assert relationship.cosine is not None
        assert relationship.cosine > 0.95
        assert relationship.clears_floor is True
        assert relationship.floor_offset.max_cosine is not None
        assert relationship.cosine > relationship.floor_offset.max_cosine
        assert relationship.alpha_b_on_a == pytest.approx(2.0, rel=0.05)
        assert relationship.residual_share == pytest.approx(1 - relationship.cosine**2)
        assert relationship.norm_ratio_b_over_a == pytest.approx(2.0, rel=0.05)

    def test_sparsity_matched_scale_reproduces_the_target_flip_fraction(
        self, unit_base: torch.Tensor
    ) -> None:
        base32 = unit_base.to(torch.float32)
        seed = wf.perturbation_seed(self.NAME, "a", 0, "sparsity-search")
        target = wd.changed_fraction(wf.rounded_perturbation(base32, 2e-3, torch.bfloat16, seed))
        found = wf.sparsity_matched_rms(
            base32,
            target_changed_fraction=target,
            rms_hint=5e-2,
            storage_dtype=torch.bfloat16,
            seed=seed,
        )
        assert found == pytest.approx(2e-3, rel=0.05)
        reproduced = wd.changed_fraction(
            wf.rounded_perturbation(base32, found, torch.bfloat16, seed)
        )
        assert reproduced == pytest.approx(target, abs=0.01)

    def test_unchanged_tensors_have_undefined_relationship(self) -> None:
        base = torch.ones(4, 4, dtype=torch.bfloat16)
        relationship = wf.delta_relationship(
            name=self.NAME, a=wd.aligned_delta(base, base), b=wd.aligned_delta(base, base)
        )
        assert relationship.cosine is None
        assert relationship.clears_floor is None
        assert relationship.floor_max is None


class TestTokenFrequency:
    def test_counts_read_both_arrays_and_skip_the_rest(self, tmp_path: Path) -> None:
        files = rollout_files(write_rollouts(tmp_path / "rollouts"))
        assert len(files) == 2
        inputs, targets, n_rows = count_tokens_in_file(files[0], VOCAB)
        assert n_rows == 2
        expected_inputs = np.zeros(VOCAB, dtype=np.int64)
        for token in (1, 2, 2, 3, 3, 4, 5, 1, 5, 5):
            expected_inputs[token] += 1
        assert np.array_equal(inputs, expected_inputs)
        assert targets[3] == 1
        assert targets[4] == 1
        assert targets[5] == 3
        assert targets.sum() == 5

    def test_parallel_count_sums_files_and_roundtrips(self, tmp_path: Path) -> None:
        files = rollout_files(write_rollouts(tmp_path / "rollouts"))
        counts = count_rollout_tokens(files, vocab_size=VOCAB, max_workers=2)
        assert counts.n_rows == 4
        assert counts.n_input_tokens == 14
        assert counts.n_target_tokens == 6
        assert counts.input_counts[7] == 3
        assert counts.target_counts[8] == 1
        counts.write(tmp_path / "token_counts.parquet")
        again = TokenCounts.read(tmp_path / "token_counts.parquet")
        assert np.array_equal(again.input_counts, counts.input_counts)
        assert again.sources == counts.sources

    def test_an_id_outside_the_vocabulary_is_refused(self, tmp_path: Path) -> None:
        files = rollout_files(write_rollouts(tmp_path / "rollouts"))
        with pytest.raises(ValueError, match="outside"):
            count_tokens_in_file(files[0], vocab_size=4)

    def test_residualised_ranking_finds_the_planted_outlier(self, tmp_path: Path) -> None:
        """Movement follows frequency^0.5 exactly except one token moved half again as much."""
        frequency = np.array(
            [0, 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + [3] * (VOCAB - 12), dtype=np.int64
        )
        movement = 0.01 * np.sqrt(np.maximum(frequency, 0)).astype(np.float64)
        movement[9] *= 1.5  # the planted outlier: frequency 128, so not the raw top mover
        movement[1] = 0.02  # moved without ever appearing
        delta32 = torch.zeros(VOCAB, HIDDEN, device=CPU)
        delta32[:, 0] = torch.from_numpy(movement).float()
        counts = TokenCounts(frequency, frequency // 2, n_rows=1, sources=("synthetic",))
        rows = token_row_frame(
            TokenRowSite(checkpoint="rl", module="embed_tokens", role=FrequencyRole.INPUT),
            base32=torch.ones(VOCAB, HIDDEN, device=CPU),
            delta32=delta32,
            counts=counts,
        )
        residualised, fit = residualise_on_log_frequency(rows)
        top_raw = rows.sort("row_norm", descending=True).head(1)["token_id"][0]
        top_residual = (
            residualised.filter(pl.col("frequency_residual").is_not_null())
            .sort("frequency_residual", descending=True)
            .head(1)["token_id"][0]
        )
        assert top_raw == 11
        assert top_residual == 9
        assert fit.r_squared > 0.9
        assert fit.n_zero_frequency == 2
        assert fit.n_zero_frequency_moved == 1
        assert residualised.filter(pl.col("token_id") == 1)["frequency_residual"][0] is None
        tokenizer_json = write_tokenizer(tmp_path / "tokenizer.json")
        strings = with_token_strings(residualised.head(3), tokenizer_json)
        assert strings["text"].to_list() == ["tok0", "tok1", "tok2"]
        assert token_strings(tokenizer_json, [5]) == [("tok5", "tok5")]


@pytest.fixture
def toy_run(tmp_path: Path) -> wg.GeometryRun:
    base = toy_base()
    write_base(tmp_path / "base", base)
    write_rl(tmp_path / "rl_a", toy_rl(base, scale=0.02, noise=1e-4, seed=21))
    write_rl(tmp_path / "rl_b", toy_rl(base, scale=0.04, noise=1e-4, seed=22))
    facts = [
        resolve_full_weights(FullWeightsSource.parse(str(tmp_path / d), None))
        for d in ("base", "rl_a", "rl_b")
    ]
    return wg.GeometryRun(
        base=facts[0],
        rl=(facts[1], facts[2]),
        out_dir=tmp_path / "out",
        n_floor_pairs=2,
        top_k_spectrum=4,
    )


class TestRun:
    def counts(self, tmp_path: Path) -> TokenCounts:
        files = rollout_files(write_rollouts(tmp_path / "rollouts"))
        return count_rollout_tokens(files, vocab_size=VOCAB, max_workers=1)

    def test_every_table_is_written_once_per_tensor(
        self, toy_run: wg.GeometryRun, tmp_path: Path
    ) -> None:
        summary = wg.run_geometry(toy_run, token_counts=self.counts(tmp_path))
        base = toy_base()
        n_names = len(base)
        n_matrices = sum(tensor.ndim >= 2 for tensor in base.values())
        assert (summary.n_tensors, summary.n_computed, summary.n_resumed) == (n_names, n_names, 0)
        tensors = load_tensors(toy_run.out_dir)
        assert tensors.height == 2 * n_names
        assert tensors.filter(pl.col("delta_stable_rank").is_not_null()).height == 2 * n_matrices
        assert (tensors["storage_dtype"] == "bfloat16").all()
        assert set(tensors["family"].unique().to_list()) == {str(f) for f in ModuleFamily}
        pairs = load_pairs(toy_run.out_dir)
        assert pairs.height == n_names
        assert pairs["checkpoint_a"].unique().to_list() == ["rl_a"]
        base_spectra = pl.read_ndjson(toy_run.out_dir / BASE_SPECTRA_FILENAME)
        assert base_spectra.height == n_names
        gates = pl.read_ndjson(toy_run.out_dir / GATE_HEADS_FILENAME)
        assert gates.height == 2 * len(GATE_HEAD_MODULES) * HEADS
        for label in ("rl_a", "rl_b"):
            for module in (ModuleClass.EMBED_TOKENS, ModuleClass.LM_HEAD):
                rows = pl.read_parquet(toy_run.out_dir / token_rows_filename(label, module))
                assert rows.height == VOCAB
                assert "frequency" in rows.columns
        assert (toy_run.out_dir / "token_counts.parquet").is_file()

    def test_the_shared_direction_reads_as_a_rescaling_that_clears_the_floor(
        self, toy_run: wg.GeometryRun
    ) -> None:
        wg.run_geometry(toy_run, token_counts=None)
        whole = relationship_by(load_pairs(toy_run.out_dir), [])
        assert whole.height == 1
        assert whole["cosine"][0] > 0.9
        assert whole["clears_floor"][0] is True
        assert whole["alpha_b_on_a"][0] == pytest.approx(2.0, rel=0.15)
        assert whole["floor_offset_max"][0] is not None
        by_family = relationship_by(load_pairs(toy_run.out_dir), ["family"])
        assert by_family.height == len(ModuleFamily)
        assert (by_family["n_clearing_floor"] >= 0).all()

    def test_group_movement_is_the_frobenius_roll_up(self, toy_run: wg.GeometryRun) -> None:
        wg.run_geometry(toy_run, token_counts=None)
        tensors = load_tensors(toy_run.out_dir)
        mlp = tensors.filter((pl.col("checkpoint") == "rl_a") & (pl.col("family") == "mlp"))
        rolled = movement_by(tensors, ["family"]).filter(
            (pl.col("checkpoint") == "rl_a") & (pl.col("family") == "mlp")
        )
        expected = math.sqrt((mlp["delta_norm"] ** 2).sum()) / math.sqrt(
            (mlp["base_norm"] ** 2).sum()
        )
        assert rolled["relative_delta"][0] == pytest.approx(expected)
        assert rolled["n_tensors"][0] == 6

    def test_a_relaunch_resumes_without_recomputing_or_duplicating(
        self, toy_run: wg.GeometryRun
    ) -> None:
        first = wg.run_geometry(toy_run, token_counts=None)
        second = wg.run_geometry(toy_run, token_counts=None)
        assert (second.n_computed, second.n_resumed) == (0, first.n_tensors)
        assert load_tensors(toy_run.out_dir).height == 2 * first.n_tensors
        assert load_pairs(toy_run.out_dir).height == first.n_tensors
        # Drop one tensor's pair row: only that tensor is recomputed, and only its missing table.
        pairs_path = toy_run.out_dir / PAIRS_FILENAME
        lines = pairs_path.read_text().splitlines()
        pairs_path.write_text("\n".join(lines[:-1]) + "\n")
        third = wg.run_geometry(toy_run, token_counts=None)
        assert (third.n_computed, third.n_resumed) == (1, first.n_tensors - 1)
        assert load_pairs(toy_run.out_dir).height == first.n_tensors
        tensors = pl.read_ndjson(toy_run.out_dir / TENSORS_FILENAME)
        assert tensors.height == 2 * first.n_tensors
        assert tensors.select("checkpoint", "name").unique().height == 2 * first.n_tensors

    def test_render_covers_every_section(self, toy_run: wg.GeometryRun, tmp_path: Path) -> None:
        wg.run_geometry(toy_run, token_counts=self.counts(tmp_path))
        text = render_readout(toy_run.out_dir, top_k_tokens=5)
        for heading in (
            "## 1. Movement",
            "## 2. Spectral",
            "## 3. Relationship",
            "## 4. Token rows",
            "## 5. DeltaNet gates",
        ):
            assert heading in text
        assert "tok" in text
        assert "| yes |" in text or "| no |" in text

    def test_checkpoints_with_different_layouts_are_refused(
        self, toy_run: wg.GeometryRun, tmp_path: Path
    ) -> None:
        release = toy_rl(toy_base(), scale=0.02, noise=0.0, seed=1)
        partial = {name: tensor for name, tensor in release.items() if name != LM_HEAD}
        write_rl(tmp_path / "rl_partial", partial)
        partial_facts = resolve_full_weights(
            FullWeightsSource.parse(str(tmp_path / "rl_partial"), None)
        )
        run = wg.GeometryRun(base=toy_run.base, rl=(partial_facts,), out_dir=tmp_path / "out2")
        with pytest.raises(ValueError, match="not a complete language model"):
            wg.open_readers(run)

    def test_duplicate_labels_and_empty_runs_are_refused(self, toy_run: wg.GeometryRun) -> None:
        with pytest.raises(ValueError, match="distinct"):
            wg.GeometryRun(
                base=toy_run.base, rl=(toy_run.rl[0], toy_run.rl[0]), out_dir=toy_run.out_dir
            )
        with pytest.raises(ValueError, match="at least one"):
            wg.GeometryRun(base=toy_run.base, rl=(), out_dir=toy_run.out_dir)


class TestMarkdown:
    def test_table_renders_nulls_booleans_and_pipes(self) -> None:
        frame = pl.DataFrame({"a": [1.5, None], "b": [True, False], "c": ["x|y", "z"]})
        text = markdown_table(frame)
        lines = text.splitlines()
        assert lines[0] == "| a | b | c |"
        assert lines[2] == "| 1.5 | yes | x\\|y |"
        assert lines[3] == "|  | no | z |"

    def test_checkpoint_spec_parsing(self, tmp_path: Path) -> None:
        source = wg.parse_checkpoint_spec("allenai/tmax-9b@step_500")
        assert (source.repo_id, source.revision) == ("allenai/tmax-9b", "step_500")
        local = wg.parse_checkpoint_spec(str(tmp_path))
        assert local.local_dir == tmp_path
        with pytest.raises(ValueError, match="repo@revision"):
            wg.parse_checkpoint_spec("allenai/tmax-9b")
