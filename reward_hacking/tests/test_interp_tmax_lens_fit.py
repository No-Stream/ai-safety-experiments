"""The anchors sidecar, the lens artifacts and the fit driver, on stubs and a CPU toy, every refusal watched.

Nothing here loads a checkpoint or ``jlens``. The sidecar round-trips at the 9B shapes on random
tensors; every provenance mismatch (other weights, another lens file, a torn directory, a shifted layer
axis) is driven to refuse; the stored unembedding is held to a reference module with this family's
``(1 + w)`` norm; and the fit driver runs end to end on a two-block toy with a stub ``jlens`` whose
lens file is the real format, so the reads side opens what the fit side wrote.
"""

from __future__ import annotations

import argparse
import json
import types
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch import nn

from games.eval_model import FullWeightsFacts
from games.interp_cells import (
    CapturedCell,
    CellIdentity,
    RowIndex,
    read_cell,
    step_dir,
    write_cell,
)
from reward_hacking.interp import tmax_lens_fit
from reward_hacking.interp.lens_artifacts import (
    LENS_FILENAME,
    PROVENANCE_FILENAME,
    UNEMBED_META_FILENAME,
    LensArtifactError,
    LensProvenance,
    SavedLens,
    StoredUnembed,
    assert_same_fit,
    load_lens_bundle,
    unembed_agreement,
)
from reward_hacking.interp.lens_fit_gate import LENS_LOAD_PATH, LensModelHandle
from reward_hacking.interp.lens_geometry import (
    GATE_META_FILENAME,
    GateProvenance,
    GateProvenanceError,
    StoredDecodeGate,
    anchor_rows_from_activations,
    assert_gate_matches_lens_file,
    assert_gate_matches_weights,
    sha256_of_file,
    wrong_layer_for,
)
from reward_hacking.interp.tmax_directions import IdentifiabilityGate
from reward_hacking.interp.tmax_full_weights import LoadingReport
from reward_hacking.interp.tmax_lens_corpus import (
    ANCHORS_FILENAME,
    ROLE_ANCHOR,
    ROLE_EVAL,
    ROLE_FIT,
    CorpusItem,
    LensCorpus,
)
from reward_hacking.interp.tmax_lens_fit import (
    FitPlan,
    capture_anchors,
    fit_and_write,
    gate_self_check,
    plan_fit,
    record_block_outputs,
    stored_unembed_from,
)
from reward_hacking.interp.tmax_lens_reads import (
    MIN_DECODE_LAYER,
    DecodeContext,
    DecodeLayerError,
    GateVerdictCache,
    decode_all,
    decode_context_for,
    decode_family,
    decode_with_floor,
    families_by_pooling,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

HIDDEN_9B = 4096
LAYERS_9B = 32
FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def provenance_for(
    lens_name: str, *, n_rows: int, lens_sha256: str = "c" * 64, fingerprint: str = FINGERPRINT_A
) -> GateProvenance:
    return GateProvenance(
        lens_name=lens_name,
        model_label="org/model@step_500",
        weights_identity="hf:deadbeef",
        weights_fingerprint=fingerprint,
        lens_sha256=lens_sha256,
        anchors_path="corpus/anchors.jsonl",
        anchors_sha256="d" * 64,
        anchor_digest="e" * 64,
        anchor_rows=tuple((f"anchor:{index // 4:03d}", 16 + index) for index in range(n_rows)),
        skip_first=16,
        jlens_commit="581d398",
        load_path=LENS_LOAD_PATH,
    )


# --------------------------------------------------------------------------------------
# The sidecar at real shapes, and every provenance refusal
# --------------------------------------------------------------------------------------


class TestAnchorsSidecar:
    def test_round_trips_at_the_9b_shapes(self, tmp_path: Path) -> None:
        generator = torch.Generator().manual_seed(0)
        residuals = torch.randn(96, LAYERS_9B, HIDDEN_9B, generator=generator)
        targets = torch.randint(0, 248_320, (96,), generator=generator)
        gate = StoredDecodeGate(
            residuals=residuals,
            target_token_ids=targets,
            lens_name="step_500/window",
            provenance=provenance_for("step_500/window", n_rows=96),
        )
        gate.save(tmp_path)
        loaded = StoredDecodeGate.load(tmp_path)
        assert torch.equal(loaded.residuals, residuals)
        assert torch.equal(loaded.target_token_ids, targets)
        assert loaded.provenance == gate.provenance
        assert loaded.lens_name == "step_500/window"
        meta = json.loads((tmp_path / GATE_META_FILENAME).read_text())
        assert (meta["n_anchors"], meta["n_layers"], meta["hidden"]) == (96, LAYERS_9B, HIDDEN_9B)
        assert meta["provenance"]["weights_fingerprint"] == FINGERPRINT_A

    def test_a_gate_without_provenance_cannot_be_written(self, tmp_path: Path) -> None:
        gate = StoredDecodeGate(torch.zeros(2, 4, 8), torch.zeros(2, dtype=torch.long), "stub")
        with pytest.raises(GateProvenanceError, match="no provenance"):
            gate.save(tmp_path)

    def test_provenance_that_disagrees_with_the_gate_is_refused(self) -> None:
        with pytest.raises(GateProvenanceError, match="named 'x' but its provenance says"):
            StoredDecodeGate(
                torch.zeros(2, 4, 8),
                torch.zeros(2, dtype=torch.long),
                "x",
                provenance=provenance_for("y", n_rows=2),
            )
        with pytest.raises(GateProvenanceError, match="2 anchor rows but the provenance names 3"):
            StoredDecodeGate(
                torch.zeros(2, 4, 8),
                torch.zeros(2, dtype=torch.long),
                "x",
                provenance=provenance_for("x", n_rows=3),
            )

    def test_a_sidecar_missing_a_field_or_with_torn_tensors_is_refused(
        self, tmp_path: Path
    ) -> None:
        gate = StoredDecodeGate(
            torch.zeros(2, 4, 8),
            torch.zeros(2, dtype=torch.long),
            "x",
            provenance=provenance_for("x", n_rows=2),
        )
        gate.save(tmp_path)
        meta_path = tmp_path / GATE_META_FILENAME
        meta = json.loads(meta_path.read_text())
        original = meta_path.read_text()
        del meta["provenance"]["weights_fingerprint"]
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(GateProvenanceError, match="weights_fingerprint"):
            StoredDecodeGate.load(tmp_path)
        meta = json.loads(original)
        meta["n_layers"] = 5
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(GateProvenanceError, match="not written together"):
            StoredDecodeGate.load(tmp_path)

    def test_a_gate_recorded_on_other_weights_is_refused_by_name(self) -> None:
        gate = StoredDecodeGate(
            torch.zeros(2, 4, 8),
            torch.zeros(2, dtype=torch.long),
            "step_500/window",
            provenance=provenance_for("step_500/window", n_rows=2, fingerprint=FINGERPRINT_A),
        )
        assert_gate_matches_weights(gate, FINGERPRINT_A, what="cell base")
        with pytest.raises(
            GateProvenanceError, match=r"aaaaaaaaaaaa.*cell base serves weights bbbbbbbbbbbb"
        ):
            assert_gate_matches_weights(gate, FINGERPRINT_B, what="cell base")

    def test_a_gate_beside_a_different_lens_file_is_refused(self, tmp_path: Path) -> None:
        lens_path = tmp_path / LENS_FILENAME
        lens_path.write_bytes(b"lens bytes")
        gate = StoredDecodeGate(
            torch.zeros(2, 4, 8),
            torch.zeros(2, dtype=torch.long),
            "x",
            provenance=provenance_for("x", n_rows=2, lens_sha256=sha256_of_file(lens_path)),
        )
        assert assert_gate_matches_lens_file(gate, lens_path) == sha256_of_file(lens_path)
        lens_path.write_bytes(b"another fit")
        with pytest.raises(GateProvenanceError, match="cannot gate another"):
            assert_gate_matches_lens_file(gate, lens_path)


# --------------------------------------------------------------------------------------
# Layer-axis sabotage on a synthetic lens
# --------------------------------------------------------------------------------------


LENS_DIM = 8
LENS_LAYERS = 32


class RotationLens:
    """Transport at layer L is a fixed random rotation R_L; a residual R_L^T e_k decodes to token k."""

    def __init__(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.rotations = [
            torch.linalg.qr(torch.randn(LENS_DIM, LENS_DIM, generator=generator))[0]
            for _ in range(LENS_LAYERS)
        ]
        self.jacobians = dict(enumerate(self.rotations[:-1]))
        self.source_layers = sorted(self.jacobians)

    def transport(self, direction: torch.Tensor, layer: int) -> torch.Tensor:
        return self.rotations[layer] @ direction


class IdentityUnembed:
    def unembed(self, transported: torch.Tensor) -> torch.Tensor:
        return transported


def rotation_gate(lens: RotationLens) -> StoredDecodeGate:
    targets = torch.arange(LENS_DIM)
    axes = torch.eye(LENS_DIM)
    residuals = torch.stack(
        [
            torch.stack([lens.rotations[layer].T @ axes[t] for layer in range(LENS_LAYERS)])
            for t in targets.tolist()
        ]
    )
    return StoredDecodeGate(residuals=residuals, target_token_ids=targets, lens_name="stub")


class TestLayerAxisSabotage:
    def test_anchors_shifted_by_one_layer_go_red_where_the_true_anchors_are_admitted(self) -> None:
        lens = RotationLens(seed=0)
        true_gate = rotation_gate(lens)
        admitted = true_gate.check(lens, IdentityUnembed(), 20, k=1)
        assert admitted.admitted
        assert admitted.hit_rate == 1.0
        shifted = StoredDecodeGate(
            residuals=torch.roll(true_gate.residuals, shifts=1, dims=1),
            target_token_ids=true_gate.target_token_ids,
            lens_name="stub",
        )
        result = shifted.check(lens, IdentityUnembed(), 20, k=1)
        assert not result.admitted
        assert result.hit_rate < 0.5
        assert "under 0.5" in result.reason
        read = decode_with_floor(
            lens.rotations[20].T @ torch.eye(LENS_DIM)[2],
            DecodeContext(
                lens=lens,
                model=IdentityUnembed(),
                lens_name="stub",
                id_to_token=lambda token_id: f"tok{token_id}",
                top_k=3,
                n_placebos=3,
                gate=shifted,
                disposition=IdentifiabilityGate(30, 9),
                fitted_layers=frozenset(range(LENS_LAYERS)),
                gate_top_k=1,
            ),
            name="d_twin",
            layer=20,
        )
        assert read.tokens == ()
        assert read.gate_admitted is False


class TestGateVerdictCache:
    """A read asks the gate about one (lens, layer) dozens of times; the verdict is computed once."""

    def contexts(
        self, lens: RotationLens, gate: StoredDecodeGate, cache: GateVerdictCache
    ) -> tuple[DecodeContext, DecodeContext]:
        def context(scale: bool) -> DecodeContext:
            return DecodeContext(
                lens=lens,
                model=IdentityUnembed(),
                lens_name="stub",
                id_to_token=lambda token_id: f"tok{token_id}",
                top_k=3,
                n_placebos=1,
                gate=gate,
                disposition=IdentifiabilityGate(30, 9),
                fitted_layers=frozenset(range(LENS_LAYERS)),
                scale_by_layer={layer: torch.ones(LENS_DIM) for layer in range(15, 23)}
                if scale
                else None,
                gate_top_k=1,
                gate_verdicts=cache,
            )

        return context(scale=True), context(scale=False)

    def test_one_evaluation_per_lens_and_layer_across_families_strata_variants_and_poolings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lens = RotationLens(seed=0)
        gate = rotation_gate(lens)
        calls: list[tuple[str, int]] = []
        original = StoredDecodeGate.check

        def counting(
            self: StoredDecodeGate, lens_: object, model: object, layer: int, **kwargs: object
        ) -> object:
            calls.append((self.lens_name, layer))
            return original(
                self, cast("Any", lens_), cast("Any", model), layer, **cast("Any", kwargs)
            )

        monkeypatch.setattr(StoredDecodeGate, "check", counting)
        cache = GateVerdictCache()
        mean_context, last_context = self.contexts(lens, gate, cache)
        families = {
            f"{name}|{stratum}": {layer: torch.randn(LENS_DIM) for layer in range(15, 23)}
            for name in ("d_twin", "d_hack")
            for stratum in ("all", "detectable")
        }
        reads = decode_all(families, mean_context)
        reads += decode_all(families, last_context)
        # 4 families x 8 layers x (raw + standardized) on the first context, 4 x 8 raw on the second.
        assert len(reads) == 4 * 8 * 2 + 4 * 8
        assert len(calls) == 8
        assert sorted(layer for _name, layer in calls) == list(range(15, 23))
        assert set(cache.verdicts) == {("stub", layer, 1) for layer in range(15, 23)}
        # Same verdicts as an uncached check, row by row.
        for read in reads:
            assert (
                read.gate_admitted
                == original(gate, lens, IdentityUnembed(), read.layer, k=1).admitted
            )

    def test_a_second_lens_never_reads_the_first_lenss_verdict(self) -> None:
        cache = GateVerdictCache()
        first = RotationLens(seed=0)
        second = RotationLens(seed=1)
        good = rotation_gate(first)
        bad = StoredDecodeGate(good.residuals, good.target_token_ids, "other-stub")
        first_context, _ = self.contexts(first, good, cache)
        second_context, _ = self.contexts(second, bad, cache)
        assert cache.verdict(first_context, 20).admitted
        # The second lens cannot recover the first lens's anchors: refused, under its own key.
        assert not cache.verdict(second_context, 20).admitted
        assert set(cache.verdicts) == {("stub", 20, 1), ("other-stub", 20, 1)}
        with pytest.raises(ValueError, match="no gate"):
            cache.verdict(
                DecodeContext(
                    lens=first,
                    model=IdentityUnembed(),
                    lens_name="ungated",
                    id_to_token=str,
                    top_k=3,
                    n_placebos=1,
                    gate=None,
                    disposition=IdentifiabilityGate.unlabelled(),
                    fitted_layers=frozenset(range(LENS_LAYERS)),
                ),
                20,
            )


class TestUnfittedLastLayer:
    """A jlens lens fits sources 0..N-2; the directions stage fits 0..N-1; the reads must not die at N-1."""

    def context(self, n_layers: int, hidden: int) -> tuple[DecodeContext, SavedLens]:
        lens = SavedLens(
            jacobians={layer: torch.eye(hidden) for layer in range(n_layers - 1)},
            n_prompts=1,
            d_model=hidden,
        )
        context = DecodeContext(
            lens=lens,
            model=IdentityUnembed(),
            lens_name="stub/window",
            id_to_token=lambda token_id: f"tok{token_id}",
            top_k=3,
            n_placebos=2,
            gate=None,
            disposition=IdentifiabilityGate.unlabelled(),
            fitted_layers=frozenset(lens.source_layers),
        )
        return context, lens

    @pytest.mark.parametrize("n_layers", [24, 32])
    def test_directions_at_every_captured_layer_decode_at_the_fitted_ones_only(
        self, n_layers: int
    ) -> None:
        context, lens = self.context(n_layers, hidden=8)
        directions = {layer: torch.randn(8) for layer in range(n_layers)}
        assert n_layers - 1 not in lens.source_layers
        reads = decode_family(directions, context, name="d_twin")
        assert [read.layer for read in reads] == list(range(MIN_DECODE_LAYER, n_layers - 1))
        with pytest.raises(
            DecodeLayerError, match=f"layer {n_layers - 1} is not a fitted source layer"
        ):
            decode_with_floor(directions[n_layers - 1], context, name="d_twin", layer=n_layers - 1)

    @pytest.mark.parametrize("n_layers", [24, 32])
    def test_the_far_wrong_layer_of_every_readable_fitted_layer_is_itself_fitted(
        self, n_layers: int
    ) -> None:
        fitted = set(range(n_layers - 1))
        for layer in range(MIN_DECODE_LAYER, n_layers - 1):
            assert wrong_layer_for(layer, n_layers=n_layers, offset=8) in fitted


# --------------------------------------------------------------------------------------
# Anchor rows, the stored unembedding, the saved lens, the provenance
# --------------------------------------------------------------------------------------


class TestAnchorRows:
    def test_rows_are_cut_at_the_positions_and_targets_are_the_argmax(self) -> None:
        per_layer = [torch.arange(40.0).reshape(10, 4) + 100 * layer for layer in range(3)]
        logits = torch.zeros(2, 6)
        logits[0, 5] = 1.0
        logits[1, 2] = 1.0
        rows, targets = anchor_rows_from_activations(per_layer, [3, 7], logits)
        assert rows.shape == (2, 3, 4)
        assert torch.equal(rows[1, 2], per_layer[2][7])
        assert targets.tolist() == [5, 2]
        assert rows.dtype == torch.float32
        assert targets.dtype == torch.long

    def test_bad_positions_and_shapes_are_refused(self) -> None:
        per_layer = [torch.zeros(10, 4)]
        with pytest.raises(ValueError, match="outside a 10-token prompt"):
            anchor_rows_from_activations(per_layer, [10], torch.zeros(1, 6))
        with pytest.raises(ValueError, match=r"expected \[2, vocab\]"):
            anchor_rows_from_activations(per_layer, [1, 2], torch.zeros(1, 6))
        with pytest.raises(ValueError, match="no anchor positions"):
            anchor_rows_from_activations(per_layer, [], torch.zeros(0, 6))


class ReferenceRMSNorm(nn.Module):
    """Qwen3.5's final norm as transformers writes it: ``(x / rms) * (1 + w)``, cast back to x's dtype."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * (1.0 + self.weight.float())).type_as(x)


class TestStoredUnembed:
    def reference(self, hidden: int = 16, vocab: int = 40) -> tuple[ReferenceRMSNorm, nn.Linear]:
        torch.manual_seed(0)
        return ReferenceRMSNorm(hidden, eps=1e-6), nn.Linear(hidden, vocab, bias=False)

    def test_matches_the_model_path_in_bf16_and_a_gain_style_norm_would_not(self) -> None:
        norm, head = self.reference()
        norm_bf16, head_bf16 = norm.to(torch.bfloat16), head.to(torch.bfloat16)
        residual = torch.randn(5, 16) * 30
        model_logits = head_bf16(norm_bf16(residual.to(torch.bfloat16))).float()
        stored = StoredUnembed(
            norm_weight=norm_bf16.weight.detach(),
            lm_head_weight=head_bf16.weight.detach(),
            rms_eps=1e-6,
        )
        agreement = unembed_agreement(stored, model_logits, residual)
        assert agreement["argmax_all_agree"] is True
        assert float(agreement["max_abs_logit_diff"]) < 1e-2
        gain_style = residual.to(torch.bfloat16).float()
        gain_style = gain_style * torch.rsqrt(gain_style.pow(2).mean(-1, keepdim=True) + 1e-6)
        wrong = (gain_style * norm_bf16.weight.float()).to(torch.bfloat16) @ head_bf16.weight.T
        assert not torch.equal(wrong.float().argmax(-1), model_logits.argmax(-1))

    def test_round_trips_and_refuses_another_convention(self, tmp_path: Path) -> None:
        norm, head = self.reference()
        stored = StoredUnembed(norm.weight.detach(), head.weight.detach(), rms_eps=1e-5)
        stored.save(tmp_path, weights_fingerprint=FINGERPRINT_A)
        loaded, fingerprint = StoredUnembed.load(tmp_path)
        assert fingerprint == FINGERPRINT_A
        assert torch.equal(loaded.lm_head_weight, stored.lm_head_weight)
        assert loaded.rms_eps == 1e-5
        meta_path = tmp_path / UNEMBED_META_FILENAME
        meta = json.loads(meta_path.read_text())
        meta["norm_convention"] = "rmsnorm(x) * w"
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(LensArtifactError, match="plausibly and wrongly"):
            StoredUnembed.load(tmp_path)

    def test_mismatched_tensors_are_refused(self) -> None:
        with pytest.raises(LensArtifactError, match="wide but the head takes"):
            StoredUnembed(torch.zeros(8), torch.zeros(40, 16), rms_eps=1e-6)
        with pytest.raises(LensArtifactError, match="but the head is"):
            StoredUnembed(torch.zeros(16), torch.zeros(40, 16, dtype=torch.bfloat16), rms_eps=1e-6)


def write_jlens_file(path: Path, jacobians: dict[int, torch.Tensor], *, n_prompts: int) -> None:
    """Write what ``jlens.lens.JacobianLens.save`` writes at the pinned commit (fp16 J)."""
    torch.save(
        {
            "J": {layer: tensor.to(torch.float16) for layer, tensor in jacobians.items()},
            "n_prompts": n_prompts,
            "source_layers": sorted(jacobians),
            "d_model": int(next(iter(jacobians.values())).shape[0]),
        },
        path,
    )


class TestSavedLens:
    def test_transports_as_jlens_does_and_refuses_a_non_lens_file(self, tmp_path: Path) -> None:
        generator = torch.Generator().manual_seed(0)
        jacobians = {layer: torch.randn(6, 6, generator=generator) for layer in range(3)}
        path = tmp_path / LENS_FILENAME
        write_jlens_file(path, jacobians, n_prompts=7)
        lens = SavedLens.load(path)
        assert lens.source_layers == [0, 1, 2]
        assert lens.n_prompts == 7
        assert lens.d_model == 6
        residual = torch.randn(6, generator=generator)
        expected = residual @ jacobians[1].to(torch.float16).float().T
        assert torch.allclose(lens.transport(residual, 1), expected)
        with pytest.raises(KeyError, match="not a fitted source layer"):
            lens.transport(residual, 3)
        torch.save({"jacobian_sum": {}, "n_done": 0}, tmp_path / "ckpt.pt")
        with pytest.raises(LensArtifactError, match="not a jlens lens file"):
            SavedLens.load(tmp_path / "ckpt.pt")


def lens_provenance(**overrides: object) -> LensProvenance:
    fields: dict[str, Any] = {
        "lens_name": "org/model@step_500/window",
        "model_label": "org/model@step_500",
        "weights_identity": "hf:deadbeef",
        "weights_fingerprint": FINGERPRINT_A,
        "load_path": LENS_LOAD_PATH,
        "tokenizer": {"source": "org/base"},
        "corpus_dir": "corpus",
        "corpus_digests": {ROLE_FIT: "f" * 64, ROLE_EVAL: "e" * 64, ROLE_ANCHOR: "a" * 64},
        "fit_role": ROLE_FIT,
        "n_fit_prompts": 60,
        "n_prompts_fitted": 60,
        "dim_batch": 8,
        "max_seq_len": 512,
        "skip_first": 16,
        "checkpoint_every": 4,
        "jlens_commit": "581d398",
        "lens_sha256": "c" * 64,
        "report": {"fit_quality": {"available": False}},
    }
    fields.update(overrides)
    return LensProvenance(**fields)


class TestLensProvenance:
    def test_round_trips_and_names_a_different_fit(self, tmp_path: Path) -> None:
        provenance = lens_provenance()
        provenance.save(tmp_path)
        loaded = LensProvenance.load(tmp_path)
        assert loaded == provenance
        assert_same_fit(loaded, provenance.identity, out_dir=tmp_path)
        wanted = {**provenance.identity, "dim_batch": 16, "fit_role": ROLE_ANCHOR}
        with pytest.raises(
            LensArtifactError,
            match=r"fit_role: on disk 'fit', asked 'anchor'; dim_batch: on disk 8, asked 16",
        ):
            assert_same_fit(loaded, wanted, out_dir=tmp_path)
        (tmp_path / PROVENANCE_FILENAME).write_text(json.dumps({"kind": "something-else"}))
        with pytest.raises(LensArtifactError, match="not a schema-1 tmax-lens"):
            LensProvenance.load(tmp_path)


# --------------------------------------------------------------------------------------
# A complete lens directory, opened for a read, and every way it can be wrong
# --------------------------------------------------------------------------------------


def write_lens_dir(
    out_dir: Path, *, fingerprint: str = FINGERPRINT_A, hidden: int = 8, n_layers: int = 4
) -> Path:
    """A consistent directory the reads side can open: lens file, gate, unembedding, provenance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(1)
    jacobians = {
        layer: torch.randn(hidden, hidden, generator=generator) for layer in range(n_layers - 1)
    }
    write_jlens_file(out_dir / LENS_FILENAME, jacobians, n_prompts=3)
    lens_sha256 = sha256_of_file(out_dir / LENS_FILENAME)
    gate = StoredDecodeGate(
        residuals=torch.randn(6, n_layers, hidden, generator=generator),
        target_token_ids=torch.arange(6),
        lens_name="stub/window",
        provenance=provenance_for(
            "stub/window", n_rows=6, lens_sha256=lens_sha256, fingerprint=fingerprint
        ),
    )
    gate.save(out_dir)
    StoredUnembed(
        torch.zeros(hidden), torch.randn(12, hidden, generator=generator), rms_eps=1e-6
    ).save(out_dir, weights_fingerprint=fingerprint)
    lens_provenance(
        lens_name="stub/window", weights_fingerprint=fingerprint, lens_sha256=lens_sha256
    ).save(out_dir)
    return out_dir


class TestLoadLensBundle:
    def test_a_consistent_directory_opens(self, tmp_path: Path) -> None:
        bundle = load_lens_bundle(write_lens_dir(tmp_path))
        assert bundle.lens_name == "stub/window"
        assert bundle.weights_fingerprint == FINGERPRINT_A
        assert bundle.lens.source_layers == [0, 1, 2]
        assert bundle.gate.n_anchors == 6

    def test_a_torn_directory_is_refused(self, tmp_path: Path) -> None:
        write_lens_dir(tmp_path)
        (tmp_path / PROVENANCE_FILENAME).unlink()
        with pytest.raises(LensArtifactError, match="did not finish"):
            load_lens_bundle(tmp_path)

    def test_a_lens_file_swapped_after_the_fit_is_refused(self, tmp_path: Path) -> None:
        write_lens_dir(tmp_path)
        write_jlens_file(
            tmp_path / LENS_FILENAME,
            {0: torch.eye(8), 1: torch.eye(8), 2: torch.eye(8)},
            n_prompts=1,
        )
        with pytest.raises(LensArtifactError, match="not the one the sidecar describes"):
            load_lens_bundle(tmp_path)

    def test_a_gate_from_another_checkpoint_is_refused_by_name(self, tmp_path: Path) -> None:
        write_lens_dir(tmp_path)
        gate = StoredDecodeGate.load(tmp_path)
        other = StoredDecodeGate(
            residuals=gate.residuals,
            target_token_ids=gate.target_token_ids,
            lens_name=gate.lens_name,
            provenance=replace(
                cast("GateProvenance", gate.provenance), weights_fingerprint=FINGERPRINT_B
            ),
        )
        other.save(tmp_path)
        with pytest.raises(
            GateProvenanceError, match=r"bbbbbbbbbbbb.*lens stub/window serves weights aaaaaaaaaaaa"
        ):
            load_lens_bundle(tmp_path)


def identity(n_layers: int, hidden: int) -> CellIdentity:
    return CellIdentity(
        base_model="org/base",
        stimuli_sha256="digest",
        rendered_sha256="digest",
        layer_convention="post_block",
        n_layers=n_layers,
        hidden_size=hidden,
        batch_size=1,
        compute_dtype="bfloat16",
        store_dtype="float32",
        stimulus_render="verbatim",
    )


def write_toy_cell(
    root: Path, *, fingerprint: str, n_layers: int = 4, hidden: int = 8
) -> CapturedCell:
    rows = RowIndex(
        stimulus_ids=("p0--a", "p0--b", "p1--a", "p1--b"),
        sides=("a", "b", "a", "b"),
        pair_ids=("p0", "p0", "p1", "p1"),
        token_counts=(5, 5, 5, 5),
    )
    cell_dir = step_dir(root, "base", 0)
    write_cell(
        cell_dir,
        arm="base",
        step=0,
        identity=identity(n_layers, hidden),
        rows={"twins": rows},
        activations={("twins", "mean"): torch.randn(4, n_layers, hidden)},
        applied_adapter_weights=None,
        adapter_weights_sha256=None,
        provenance={"weights_fingerprint": fingerprint},
    )
    return read_cell(cell_dir)


class TestDecodeContextForCell:
    def test_the_gate_is_held_to_the_cells_weights(self, tmp_path: Path) -> None:
        bundle = load_lens_bundle(write_lens_dir(tmp_path / "lens"))
        cell = write_toy_cell(tmp_path / "cells", fingerprint=FINGERPRINT_A)
        context = decode_context_for(
            bundle,
            cell=cell,
            stimulus_set="twins",
            pooling="mean",
            id_to_token=str,
            top_k=3,
            n_placebos=2,
            disposition=IdentifiabilityGate.unlabelled(),
        )
        assert context.gate is bundle.gate
        assert context.scale_by_layer is not None
        assert set(context.scale_by_layer) == {0, 1, 2, 3}
        other = write_toy_cell(tmp_path / "other", fingerprint=FINGERPRINT_B)
        with pytest.raises(
            GateProvenanceError, match=r"cell base/step-0 .*serves weights bbbbbbbbbbbb"
        ):
            decode_context_for(
                bundle,
                cell=other,
                stimulus_set="twins",
                pooling="mean",
                id_to_token=str,
                top_k=3,
                n_placebos=2,
                disposition=IdentifiabilityGate.unlabelled(),
            )

    def test_families_regroup_by_pooling_and_keep_the_stratum(self) -> None:
        grouped = families_by_pooling(
            {
                ("d_twin", "mean", "all", 20): torch.zeros(2),
                ("d_twin", "mean", "all", 21): torch.ones(2),
                ("d_hack", "last", "detectable", 20): torch.zeros(2),
            }
        )
        assert set(grouped) == {"mean", "last"}
        assert sorted(grouped["mean"]["d_twin|all"]) == [20, 21]
        assert list(grouped["last"]) == ["d_hack|detectable"]


# --------------------------------------------------------------------------------------
# The fit driver on a two-block toy with a stub jlens
# --------------------------------------------------------------------------------------

TOY_HIDDEN = 8
TOY_VOCAB = 24
TOY_LAYERS = 4
TOY_RMS_EPS = 1e-6


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Linear(TOY_HIDDEN, TOY_HIDDEN)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden + torch.tanh(self.mlp(hidden)),)


class ToyText(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(TOY_VOCAB, TOY_HIDDEN)
        self.layers = nn.ModuleList([ToyBlock() for _ in range(TOY_LAYERS)])
        self.norm = ReferenceRMSNorm(TOY_HIDDEN, TOY_RMS_EPS)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)[0]
        return hidden


class ToyHF(nn.Module):
    """An HF-shaped causal LM: ``model`` holds the text stack, ``lm_head`` the unembedding."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.model = ToyText()
        self.lm_head = nn.Linear(TOY_HIDDEN, TOY_VOCAB, bias=False)
        self.config = types.SimpleNamespace(
            get_text_config=lambda: types.SimpleNamespace(rms_norm_eps=TOY_RMS_EPS)
        )


class ToyTokenizer:
    """A loaded-tokenizer-shaped fixture for the provenance payload."""

    def __init__(self) -> None:
        self.vocabulary = {f"token_{index}": index for index in range(TOY_VOCAB)}
        self.padding_side = "right"
        self.truncation_side = "right"
        self.model_max_length = 512
        self.clean_up_tokenization_spaces = False
        self.split_special_tokens = False
        self.chat_template = "{{ messages }}"
        self.special_tokens_map = {"eos_token": "<eos>"}
        self.init_kwargs = {"model_max_length": 512, "_commit_hash": "toy-commit"}

    def get_vocab(self) -> dict[str, int]:
        return self.vocabulary

    def get_added_vocab(self) -> dict[str, int]:
        return {}


class ToyLensModel:
    """The slice of ``jlens.hf.HFLensModel`` the driver uses, over a :class:`ToyHF`."""

    def __init__(self, hf: ToyHF) -> None:
        self.hf = hf
        self.layers = hf.model.layers
        self.n_layers = TOY_LAYERS
        self.d_model = TOY_HIDDEN
        self.layout = types.SimpleNamespace(path="model", norm="norm", lm_head="lm_head")

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        ids = [(ord(char) * 7) % TOY_VOCAB for char in text][:max_length]
        return torch.tensor([ids])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.hf.model(input_ids)

    def unembed(self, transported: torch.Tensor) -> torch.Tensor:
        return self.hf.lm_head(self.hf.model.norm(transported))


class StubJacobianLens:
    """A lens with jlens's surface: fitted Jacobians, save/load in the real file format, apply."""

    def __init__(self, jacobians: dict[int, torch.Tensor], *, n_prompts: int, d_model: int) -> None:
        self.jacobians = {layer: tensor.float() for layer, tensor in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model

    def save(self, path: str) -> None:
        write_jlens_file(Path(path), self.jacobians, n_prompts=self.n_prompts)

    @classmethod
    def load(cls, path: str) -> StubJacobianLens:
        state = torch.load(path, weights_only=True)
        return cls(state["J"], n_prompts=state["n_prompts"], d_model=state["d_model"])

    @classmethod
    def merge(cls, lenses: Sequence[StubJacobianLens]) -> StubJacobianLens:
        """The reference's merge: the n_prompts-weighted mean of the inputs."""
        total = sum(lens.n_prompts for lens in lenses)
        merged: dict[int, torch.Tensor] = {}
        for layer in lenses[0].source_layers:
            weighted = torch.zeros_like(lenses[0].jacobians[layer])
            for lens in lenses:
                weighted = weighted + lens.jacobians[layer] * lens.n_prompts
            merged[layer] = weighted / total
        return cls(merged, n_prompts=total, d_model=lenses[0].d_model)

    def transport(self, direction: torch.Tensor, layer: int) -> torch.Tensor:
        return direction.float() @ self.jacobians[layer].T

    def apply(
        self,
        model: ToyLensModel,
        prompt: str,
        *,
        positions: Sequence[int],
        max_seq_len: int,
        use_jacobian: bool,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        ids = model.encode(prompt, max_length=max_seq_len)
        with torch.no_grad(), record_block_outputs(model.layers) as activations:
            model.forward(ids)
        lens_logits: dict[int, torch.Tensor] = {}
        for layer in self.source_layers:
            residual = activations[layer][0][list(positions)].float()
            if use_jacobian:
                residual = self.transport(residual, layer)
            lens_logits[layer] = model.unembed(residual).float()
        final = activations[TOY_LAYERS - 1][0][list(positions)]
        return lens_logits, model.unembed(final).float(), ids


class StubJlens:
    """``jlens`` as the driver calls it: ``fit``, ``JacobianLens`` and ``fitting.SKIP_FIRST_N_POSITIONS``.

    The "fit" is the identity on every source layer plus a per-prompt perturbation, checkpointed like the
    reference so a resumed fit continues where it stopped.
    """

    JacobianLens = StubJacobianLens
    fitting = types.SimpleNamespace(SKIP_FIRST_N_POSITIONS=2)

    def __init__(self) -> None:
        self.fit_calls: list[dict[str, object]] = []

    def fit(
        self, model: ToyLensModel, prompts: Sequence[str], **kwargs: object
    ) -> StubJacobianLens:
        self.fit_calls.append({"n_prompts": len(prompts), "prompts": list(prompts), **kwargs})
        jacobians = {
            layer: torch.eye(model.d_model) + prompt_perturbation(prompts, model.d_model)
            for layer in range(model.n_layers - 1)
        }
        checkpoint_path = cast("str | None", kwargs.get("checkpoint_path"))
        if checkpoint_path is not None:
            torch.save({"jacobian_sum": jacobians, "n_done": len(prompts)}, checkpoint_path)
        return StubJacobianLens(jacobians, n_prompts=len(prompts), d_model=model.d_model)


def prompt_perturbation(prompts: Sequence[str], d_model: int) -> torch.Tensor:
    """A small deterministic matrix per prompt set, so two halves fit two different lenses."""
    total = torch.zeros(d_model, d_model)
    for prompt in prompts:
        generator = torch.Generator().manual_seed(sum(map(ord, prompt)))
        total += 0.01 * torch.randn(d_model, d_model, generator=generator)
    return total / max(len(prompts), 1)


def toy_corpus(corpus_dir: Path) -> LensCorpus:
    def item(item_id: str, role: str, text: str, group: str) -> CorpusItem:
        return CorpusItem(
            item_id=item_id,
            role=role,
            window_class=role,
            source="stimuli",
            served="rendered-prompt",
            doc_id=item_id,
            group_id=group,
            start=0,
            end=len(text),
            shift=0,
            text=text,
            ids_sha256="h",
        )

    fit = tuple(
        item(f"fit:{i}", ROLE_FIT, f"fit window number {i} " * 3, f"gf{i}") for i in range(3)
    )
    eval_items = tuple(
        item(f"eval:{i}", ROLE_EVAL, f"eval window {i} " * 4, f"ge{i}") for i in range(2)
    )
    anchors = tuple(
        item(
            f"anchor:{i:03d}",
            ROLE_ANCHOR,
            f"anchor prompt {i}, a whole rendered text " * 2,
            f"ga{i}",
        )
        for i in range(2)
    )
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / ANCHORS_FILENAME).write_text(
        "".join(json.dumps(anchor.as_row()) + "\n" for anchor in anchors)
    )
    return LensCorpus(
        fit=fit,
        eval=eval_items,
        anchors=anchors,
        sidecar={"digests": {ROLE_FIT: "f" * 64, ROLE_EVAL: "e" * 64, ROLE_ANCHOR: "a" * 64}},
    )


def toy_handle(hf: ToyHF, *, fingerprint: str = FINGERPRINT_A) -> LensModelHandle:
    facts = FullWeightsFacts(
        label="org/toy@step_0",
        snapshot_dir=Path("/nonexistent/snapshot"),
        commit_sha="c0ffee",
        weights_sha256=(("model.safetensors", fingerprint),),
        chat_template_sha256=None,
        declares_vision_config=False,
    )
    return LensModelHandle(
        model=ToyLensModel(hf),
        hf_model=cast("Any", hf),
        tokenizer=ToyTokenizer(),
        facts=facts,
        loading_report=LoadingReport(0, 0, 0),
        layer_types=("full_attention",) * TOY_LAYERS,
        conv_reach=3,
        tokenizer_source="org/base",
        load_seconds=0.1,
        device_facts={"gpu_name": "none (toy)"},
    )


def toy_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "fit_role": ROLE_FIT,
        "fit_half": None,
        "merge_halves": None,
        "cross_lens_dir": [],
        "max_fit_prompts": None,
        "max_seq_len_ceiling": None,
        "anchor_max_seq_len": 64,
        "dim_batch": 4,
        "checkpoint_every": 2,
        "positions_per_anchor": 3,
        "eval_positions": 4,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestCaptureAnchors:
    def test_rows_land_at_interior_positions_with_the_models_own_argmax(
        self, tmp_path: Path
    ) -> None:
        model = ToyLensModel(ToyHF())
        corpus = toy_corpus(tmp_path / "corpus")
        capture = capture_anchors(
            model, corpus.anchors, max_seq_len=64, positions_per_anchor=3, skip_first=2
        )
        assert capture.residuals.shape == (6, TOY_LAYERS, TOY_HIDDEN)
        assert capture.target_token_ids.shape == (6,)
        assert [row[0] for row in capture.rows] == ["anchor:000"] * 3 + ["anchor:001"] * 3
        assert all(
            2 <= position < capture.seq_lens[item_id] - 1 for item_id, position in capture.rows
        )
        # Recompute one row by hand: the final-layer residual at that position through the model's unembed.
        _item_id, position = capture.rows[4]
        ids = model.encode(corpus.anchors[1].text, max_length=64)
        with torch.no_grad(), record_block_outputs(model.layers) as activations:
            model.forward(ids)
        expected = int(model.unembed(activations[TOY_LAYERS - 1][0][position]).argmax())
        assert int(capture.target_token_ids[4]) == expected
        assert torch.allclose(capture.residuals[4, 1], activations[1][0][position])

    def test_the_self_check_admits_a_faithful_lens_and_refuses_a_layer_blind_one(
        self, tmp_path: Path
    ) -> None:
        model = ToyLensModel(ToyHF())
        corpus = toy_corpus(tmp_path / "corpus")
        capture = capture_anchors(
            model, corpus.anchors, max_seq_len=64, positions_per_anchor=3, skip_first=2
        )
        gate = StoredDecodeGate(capture.residuals, capture.target_token_ids, "toy")
        blind = StubJacobianLens(
            {layer: torch.eye(TOY_HIDDEN) for layer in range(TOY_LAYERS - 1)},
            n_prompts=1,
            d_model=TOY_HIDDEN,
        )
        # An identity lens recovers the token at the true layer AND at the wrong one: no margin, refused.
        check = gate_self_check(gate, blind, model) if MIN_DECODE_LAYER < TOY_LAYERS else {}
        assert check == {} or all(not block["admitted"] for block in check.values())
        result = gate.check(blind, model, TOY_LAYERS - 2, wrong_layer_offset=1)
        assert result.hit_rate == result.wrong_layer_hit_rate
        assert not result.admitted


class TestFitAndWrite:
    def run_fit(
        self, tmp_path: Path, *, jl: StubJlens | None = None
    ) -> tuple[LensProvenance, Path, StubJlens]:
        jl = jl or StubJlens()
        hf = ToyHF()
        handle = toy_handle(hf)
        corpus = toy_corpus(tmp_path / "corpus")
        plan = plan_fit(corpus, toy_args(), label="org/toy@step_0")
        out_dir = tmp_path / "lens"
        provenance = fit_and_write(
            handle,
            corpus,
            plan,
            jl=cast("Any", jl),
            out_dir=out_dir,
            corpus_dir=tmp_path / "corpus",
            jlens_facts={"path": "/stub", "commit": "581d398abc", "pinned_commit": "581d398"},
            kernel_bridge={"stub": True},
            invocation="test",
        )
        return provenance, out_dir, jl

    def test_writes_a_directory_the_reads_side_opens_and_the_gate_is_held_to_the_weights(
        self, tmp_path: Path
    ) -> None:
        provenance, out_dir, jl = self.run_fit(tmp_path)
        assert provenance.fit_role == ROLE_FIT
        assert provenance.n_fit_prompts == 3
        assert provenance.lens_name == "org/toy@step_0/window"
        assert provenance.weights_fingerprint == toy_handle(ToyHF()).facts.fingerprint
        call = jl.fit_calls[0]
        assert call["checkpoint_every"] == 2
        assert call["resume"] is True
        assert call["dim_batch"] == 4
        assert not (out_dir / "fit-checkpoint.pt").exists()
        bundle = load_lens_bundle(out_dir)
        assert bundle.gate.n_anchors == 6
        assert bundle.gate.n_layers == TOY_LAYERS
        assert bundle.gate.provenance is not None
        assert bundle.gate.provenance.anchor_digest == "a" * 64
        assert bundle.gate.provenance.anchors_sha256 == sha256_of_file(
            tmp_path / "corpus" / ANCHORS_FILENAME
        )
        assert bundle.gate.provenance.weights_fingerprint == provenance.weights_fingerprint
        assert bundle.unembed.vocab_size == TOY_VOCAB
        report = provenance.report
        assert report["unembed_agreement"]["argmax_all_agree"] is True
        assert report["fit_quality"]["available"] is True
        assert report["gate"]["anchors_are_fit_text"] is False
        assert_gate_matches_weights(bundle.gate, provenance.weights_fingerprint, what="lens")
        with pytest.raises(GateProvenanceError, match="cannot gate residuals from another"):
            assert_gate_matches_weights(bundle.gate, FINGERPRINT_B, what="cell tmax")

    def test_a_relaunch_skips_the_fit_and_a_different_fit_into_the_same_directory_is_refused(
        self, tmp_path: Path
    ) -> None:
        provenance, out_dir, jl = self.run_fit(tmp_path)
        again, _, _ = self.run_fit(tmp_path, jl=jl)
        assert again == provenance
        assert len(jl.fit_calls) == 1
        handle = toy_handle(ToyHF())
        corpus = toy_corpus(tmp_path / "corpus")
        plan = plan_fit(corpus, toy_args(dim_batch=16), label="org/toy@step_0")
        with pytest.raises(LensArtifactError, match="dim_batch: on disk 4, asked 16"):
            fit_and_write(
                handle,
                corpus,
                plan,
                jl=cast("Any", StubJlens()),
                out_dir=out_dir,
                corpus_dir=tmp_path / "corpus",
                jlens_facts={"path": "/stub", "commit": "581d398abc", "pinned_commit": "581d398"},
                kernel_bridge={},
                invocation="test",
            )

    def test_the_anchor_role_fits_on_the_anchor_prompts_and_says_the_gate_saw_fit_text(
        self, tmp_path: Path
    ) -> None:
        corpus = toy_corpus(tmp_path / "corpus")
        plan = plan_fit(corpus, toy_args(fit_role=ROLE_ANCHOR), label="org/toy@step_0")
        assert list(plan.prompts) == corpus.anchor_prompts
        assert plan.lens_name == "org/toy@step_0/anchor"
        provenance = fit_and_write(
            toy_handle(ToyHF()),
            corpus,
            plan,
            jl=cast("Any", StubJlens()),
            out_dir=tmp_path / "anchor-lens",
            corpus_dir=tmp_path / "corpus",
            jlens_facts={"path": "/stub", "commit": None, "pinned_commit": "581d398"},
            kernel_bridge={},
            invocation="test",
        )
        assert provenance.report["gate"]["anchors_are_fit_text"] is True
        assert provenance.jlens_commit == "unverified (not a git checkout; pinned 581d398)"
        gate = load_lens_bundle(tmp_path / "anchor-lens").gate
        assert gate.provenance is not None
        assert gate.provenance.jlens_commit.startswith("unverified")

    def test_the_stored_unembedding_is_lifted_from_the_served_model(self) -> None:
        hf = ToyHF()
        stored = stored_unembed_from(toy_handle(hf))
        assert torch.equal(stored.norm_weight, hf.model.norm.weight.detach())
        assert torch.equal(stored.lm_head_weight, hf.lm_head.weight.detach())
        assert stored.rms_eps == TOY_RMS_EPS


def fit_into(
    tmp_path: Path,
    out_dir: Path,
    *,
    hf: ToyHF | None = None,
    jl: StubJlens | None = None,
    **args: object,
) -> LensProvenance:
    corpus = toy_corpus(tmp_path / "corpus")
    plan = plan_fit(corpus, toy_args(**args), label="org/toy@step_0")
    return fit_and_write(
        toy_handle(hf or ToyHF()),
        corpus,
        plan,
        jl=cast("Any", jl or StubJlens()),
        out_dir=out_dir,
        corpus_dir=tmp_path / "corpus",
        jlens_facts={"path": "/stub", "commit": "581d398abc", "pinned_commit": "581d398"},
        kernel_bridge={},
        invocation="test",
    )


class TestHalvesAndMerge:
    def test_the_halves_split_the_fit_windows_by_index_and_name_themselves(
        self, tmp_path: Path
    ) -> None:
        corpus = toy_corpus(tmp_path / "corpus")
        odd = plan_fit(corpus, toy_args(fit_half="odd"), label="l")
        even = plan_fit(corpus, toy_args(fit_half="even"), label="l")
        assert list(odd.prompts) == corpus.fit_prompts[1::2]
        assert list(even.prompts) == corpus.fit_prompts[0::2]
        assert (odd.lens_name, even.lens_name) == ("l/window-odd", "l/window-even")
        with pytest.raises(tmax_lens_fit.LensFitError, match="window lens"):
            plan_fit(corpus, toy_args(fit_half="odd", fit_role=ROLE_ANCHOR), label="l")
        with pytest.raises(tmax_lens_fit.LensFitError, match="not both"):
            plan_fit(
                corpus,
                toy_args(fit_half="odd", merge_halves=(tmp_path / "a", tmp_path / "b")),
                label="l",
            )

    def test_merging_two_halves_averages_them_and_records_the_split_half_gap(
        self, tmp_path: Path
    ) -> None:
        hf = ToyHF()
        odd = fit_into(tmp_path, tmp_path / "odd", hf=hf, fit_half="odd")
        even = fit_into(tmp_path, tmp_path / "even", hf=hf, fit_half="even")
        assert odd.n_fit_prompts == 1
        assert even.n_fit_prompts == 2
        jl = StubJlens()
        merged = fit_into(
            tmp_path,
            tmp_path / "window",
            hf=hf,
            jl=jl,
            merge_halves=(tmp_path / "odd", tmp_path / "even"),
        )
        assert jl.fit_calls == []
        assert merged.lens_name == "org/toy@step_0/window"
        assert merged.n_fit_prompts == merged.n_prompts_fitted == 3
        block = merged.report["merged_from"]
        assert [half["lens_name"] for half in block["halves"]] == [odd.lens_name, even.lens_name]
        gaps = block["split_half_relative_diff_by_layer"]
        assert set(gaps) == {"0", "1", "2"}
        assert all(gap > 0.0 for gap in gaps.values())
        odd_lens = SavedLens.load(tmp_path / "odd" / LENS_FILENAME)
        even_lens = SavedLens.load(tmp_path / "even" / LENS_FILENAME)
        merged_lens = SavedLens.load(tmp_path / "window" / LENS_FILENAME)
        expected = (odd_lens.jacobians[1] * 1 + even_lens.jacobians[1] * 2) / 3
        assert torch.allclose(merged_lens.jacobians[1], expected, atol=2e-3)
        assert load_lens_bundle(tmp_path / "window").gate.n_anchors == 6

    def test_a_half_from_other_weights_or_a_torn_half_is_refused(self, tmp_path: Path) -> None:
        hf = ToyHF()
        fit_into(tmp_path, tmp_path / "odd", hf=hf, fit_half="odd")
        fit_into(tmp_path, tmp_path / "even", hf=hf, fit_half="even")
        other = toy_handle(ToyHF(), fingerprint=FINGERPRINT_B)
        corpus = toy_corpus(tmp_path / "corpus")
        plan = plan_fit(
            corpus,
            toy_args(merge_halves=(tmp_path / "odd", tmp_path / "even")),
            label="org/toy@step_0",
        )
        with pytest.raises(LensArtifactError, match="weights_fingerprint: half"):
            fit_and_write(
                other,
                corpus,
                plan,
                jl=cast("Any", StubJlens()),
                out_dir=tmp_path / "window",
                corpus_dir=tmp_path / "corpus",
                jlens_facts={"path": "/stub", "commit": "581d398abc", "pinned_commit": "581d398"},
                kernel_bridge={},
                invocation="test",
            )
        with pytest.raises(LensArtifactError, match="odd and the even half"):
            fit_into(
                tmp_path,
                tmp_path / "window3",
                hf=hf,
                merge_halves=(tmp_path / "odd", tmp_path / "odd"),
            )
        (tmp_path / "even" / PROVENANCE_FILENAME).unlink()
        with pytest.raises(LensArtifactError, match="did not finish cannot be merged"):
            fit_into(
                tmp_path,
                tmp_path / "window2",
                hf=hf,
                merge_halves=(tmp_path / "odd", tmp_path / "even"),
            )


class TestCrossLens:
    def test_another_checkpoints_lens_is_applied_and_this_checkpoints_own_is_refused(
        self, tmp_path: Path
    ) -> None:
        base = ToyHF()
        base_provenance = fit_into(tmp_path, tmp_path / "base", hf=base)
        other_hf = ToyHF()
        with torch.no_grad():
            other_hf.lm_head.weight.add_(0.05 * torch.randn_like(other_hf.lm_head.weight))
        other = toy_handle(other_hf, fingerprint=FINGERPRINT_B)
        corpus = toy_corpus(tmp_path / "corpus")
        plan = plan_fit(
            corpus, toy_args(cross_lens_dir=[tmp_path / "base"]), label="org/other@step_500"
        )
        provenance = fit_and_write(
            other,
            corpus,
            plan,
            jl=cast("Any", StubJlens()),
            out_dir=tmp_path / "other",
            corpus_dir=tmp_path / "corpus",
            jlens_facts={"path": "/stub", "commit": "581d398abc", "pinned_commit": "581d398"},
            kernel_bridge={},
            invocation="test",
        )
        cross = provenance.report["cross_lens"][base_provenance.lens_name]
        assert cross["lens_weights_fingerprint"] == base_provenance.weights_fingerprint
        assert cross["lens_sha256"] == base_provenance.lens_sha256
        assert cross["fit_quality_on_this_checkpoint"]["available"] is True
        assert isinstance(cross["gate_by_layer_on_this_checkpoints_anchors"], dict)
        with pytest.raises(LensArtifactError, match="own weights"):
            fit_into(tmp_path, tmp_path / "base-again", hf=base, cross_lens_dir=[tmp_path / "base"])


class TestPlanFit:
    def test_the_window_lens_caps_at_the_default_ceiling_and_the_anchor_lens_at_the_anchor_cap(
        self, tmp_path: Path
    ) -> None:
        corpus = toy_corpus(tmp_path / "corpus")
        window = plan_fit(corpus, toy_args(max_fit_prompts=2), label="l")
        assert isinstance(window, FitPlan)
        assert len(window.prompts) == 2
        assert window.max_seq_len == max(item.n_tokens for item in corpus.fit)
        anchor = plan_fit(corpus, toy_args(fit_role=ROLE_ANCHOR), label="l")
        assert anchor.max_seq_len == max(item.n_tokens for item in corpus.anchors)
        with pytest.raises(tmax_lens_fit.LensFitError, match="--fit-role"):
            plan_fit(corpus, toy_args(fit_role="eval"), label="l")
