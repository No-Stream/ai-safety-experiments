"""Pin the full-weights cell kind's guards, and run one whole capture on a tiny composite model, no GPU.

Offline and CPU-only. The model is eight dimensions wide with three stand-in decoder blocks laid out
the way the real composite class is (`model.model.language_model.layers`), because everything this
driver can get wrong is about conventions rather than numbers: whether a span mask pools the same
positions the plain pooler does, whether the loading gate reads a renamed key as a refusal, whether
two labels over one set of weights are told apart from two checkpoints, and whether a ladder that
mixes kernel bindings is refused before it is averaged.

:class:`TestMaskPooler` is the bit-for-bit claim: the named-mask pooler handed the whole prompt IS
`mean` and `last`, and handed a window pools only that window.

:class:`TestLoadingGate` is the sabotage the L4 smoke performs for real (one renamed safetensors
key), on a mocked loading report: a renamed key shows up as one missing and one unexpected
language-model key, and either alone refuses.

:class:`TestLadderGuards` writes synthetic cells to disk and reads them back through
`load_full_weights_ladder`: a declared replica pair must be exactly 0.0 apart, an undeclared
duplicate fingerprint is refused, bit-identical activations under different fingerprints are refused,
and two cells recorded under different DeltaNet kernel bindings are refused by the games guard.

:class:`TestCaptureEndToEnd` runs the capture on the tiny model with a span table: the derived
`@full` set equals the plain set bit for bit, a window set equals the shared positionwise capture
pooled over that window, and a sidecar whose ids disagree with the tokenizer is refused.

The writer-versus-reader contracts (the sidecar file, the provenance block, the sentence corpus's
render) are pinned in `test_tmax_cell_contracts.py`, which borrows the tiny model and tokenizer here.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch import nn

from games.deltanet_kernels import DELTANET_KERNEL_FIELD
from games.eval_model import WEIGHTS_PROVENANCE_FILENAME, FullWeightsFacts, FullWeightsSource
from games.interp_cells import (
    LAYER_CONVENTION_POST_BLOCK,
    STIMULUS_RENDER_VERBATIM,
    CellFormatError,
    CellIdentity,
    RowIndex,
    Stimulus,
    group_by_set,
    read_cell,
    step_dir,
    write_cell,
)
from reward_hacking.interp.directions import (
    capture_positionwise_activations,
    last_token_pool,
    mean_pool,
)
from reward_hacking.interp.tmax_capture_ladder import (
    assert_ids_match_sidecar,
    capture_full_weights_cell,
    pool_masked,
    span_mask,
    teacher_forced_text,
)
from reward_hacking.interp.tmax_full_weights import (
    REPLICA_OF_FIELD,
    WEIGHTS_FINGERPRINT_FIELD,
    CoherenceTriple,
    FullWeightsCellError,
    FullWeightsCellSpec,
    LoadingReport,
    apply_replica_declarations,
    assert_fingerprints_declared,
    assert_language_model_complete,
    derived_set_name,
    games_label,
    load_coherence_table,
    load_full_weights_ladder,
    parse_cell_spec,
    parse_replica_spec,
    read_amplified_sidecar,
)
from reward_hacking.interp.tmax_twin_sidecar import (
    SIDE_HONEST,
    SIDE_RIGGED,
    STIMULUS_SET,
    SpanTable,
)

if TYPE_CHECKING:
    from pathlib import Path

N_LAYERS = 3
HIDDEN = 8
VOCAB = 64

TURN_PREFIX = "<|im_start|>user\n"
TURN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n"

FUSED_KERNELS = {
    "chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
    "causal_conv1d_fn": "causal_conv1d.causal_conv1d_interface.causal_conv1d_fn",
}
TORCH_KERNELS = {
    "chunk_gated_delta_rule": "transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule",
    "causal_conv1d_fn": "transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_fn",
}


# --------------------------------------------------------------------------------------
# A composite-shaped tiny model and a char tokenizer
# --------------------------------------------------------------------------------------


class OffsetLayer(nn.Module):
    """A stand-in decoder block: adds its own offset so each layer's output is distinguishable."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.offset


class TinyTrunk(nn.Module):
    """The text trunk: embed, then run the decoder blocks, as `Qwen3_5TextModel` does."""

    def __init__(self, shift: float) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN) / 1000.0
                + shift
            )
        self.layers = nn.ModuleList([OffsetLayer(float(index + 1)) for index in range(N_LAYERS)])

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> SimpleNamespace:
        del attention_mask, kwargs
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyOuter(nn.Module):
    """`Qwen3_5Model`: holds the text tower under `language_model` and has no `layers` of its own."""

    def __init__(self, shift: float) -> None:
        super().__init__()
        self.language_model = TinyTrunk(shift)


class TinyComposite(nn.Module):
    """`Qwen3_5ForConditionalGeneration`-shaped: `model.language_model.layers`, a text sub-config."""

    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.model = TinyOuter(shift)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=N_LAYERS, hidden_size=HIDDEN)
        )

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


class Encoding(dict[str, torch.Tensor]):
    def to(self, device: object) -> Encoding:
        del device
        return self


class TinyTokenizer:
    """A char tokenizer with the family's turn markers, enough for the capture loop."""

    pad_token: str | None = "<pad>"
    eos_token = "<eos>"
    padding_side: str = "right"

    def get_chat_template(self) -> str:
        return "{% for message in messages %}...{% endfor %}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
        **kwargs: object,
    ) -> str:
        del tokenize, add_generation_prompt, kwargs
        suffix = TURN_SUFFIX if enable_thinking else TURN_SUFFIX.removesuffix("<think>\n")
        return f"{TURN_PREFIX}{messages[0]['content']}{suffix}"

    @staticmethod
    def _ids(text: str) -> list[int]:
        return [(ord(char) % (VOCAB - 1)) + 1 for char in text] or [1]

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str | None = None,
        padding: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding | dict[str, Any]:
        del padding, add_special_tokens
        if isinstance(text, str):
            return {"input_ids": self._ids(text)}
        sequences = [self._ids(item) for item in text]
        if return_tensors is None:
            return {"input_ids": sequences}
        longest = max(len(sequence) for sequence in sequences)
        input_ids = torch.zeros(len(sequences), longest, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), longest, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, : len(sequence)] = 1
        return Encoding(input_ids=input_ids, attention_mask=attention_mask)


def verbatim_stimuli(n_pairs: int = 2) -> list[Stimulus]:
    """Pre-rendered rows as the twin corpus writes them: the template inside `text`, no prefix."""
    return [
        Stimulus(
            stimulus_id=f"p{index}--{side}",
            stimulus_set=STIMULUS_SET,
            side=side,
            pair_id=f"p{index}",
            text=f"{TURN_PREFIX}problem {index} graded {side}{TURN_SUFFIX}",
        )
        for index in range(n_pairs)
        for side in (SIDE_RIGGED, SIDE_HONEST)
    ]


def span_table_for(
    tokenizer: TinyTokenizer, stimuli: list[Stimulus], *, drop_head_from: str | None = None
) -> SpanTable:
    """Spans over the tiny tokenization: `full` is the whole prompt, `head` its first three tokens."""
    input_ids: dict[str, tuple[int, ...]] = {}
    spans: dict[str, dict[str, tuple[int, int]]] = {}
    absent: dict[str, dict[str, str]] = {}
    for stimulus in stimuli:
        ids = tuple(cast("list[int]", tokenizer(stimulus.text)["input_ids"]))
        input_ids[stimulus.stimulus_id] = ids
        spans[stimulus.stimulus_id] = {"full": (0, len(ids)), "head": (0, 3)}
        absent[stimulus.stimulus_id] = {}
        if stimulus.stimulus_id == drop_head_from:
            del spans[stimulus.stimulus_id]["head"]
            absent[stimulus.stimulus_id]["head"] = "the rendering rewrote the whole check"
    return SpanTable(
        input_ids=input_ids,
        spans=spans,
        spans_absent=absent,
        stimuli_sha256="unused-here",
        stimulus_render=STIMULUS_RENDER_VERBATIM,
    )


def facts_with(digest: str, snapshot_dir: Path, *, label: str = "unit") -> FullWeightsFacts:
    return FullWeightsFacts(
        label=label,
        snapshot_dir=snapshot_dir,
        commit_sha=None,
        weights_sha256=(("model.safetensors", digest),),
        chat_template_sha256=None,
        declares_vision_config=True,
    )


def hub_spec(arm: str, step: int, *, replica_of: str | None = None) -> FullWeightsCellSpec:
    return FullWeightsCellSpec(
        arm=arm,
        step=step,
        source=FullWeightsSource(
            repo_id="allenai/tmax-9b", revision=f"step_{step}", local_dir=None
        ),
        replica_of=replica_of,
    )


# --------------------------------------------------------------------------------------


class TestMaskPooler:
    def test_the_full_span_is_mean_and_last_bit_for_bit(self) -> None:
        hidden = torch.randn(1, 12, HIDDEN)
        attention = torch.ones(1, 12, dtype=torch.long)
        full = span_mask(12, (0, 12))
        assert torch.equal(pool_masked(hidden, full, "mean"), mean_pool(hidden, attention))
        assert torch.equal(pool_masked(hidden, full, "last"), last_token_pool(hidden, attention))

    def test_a_window_pools_only_its_own_positions(self) -> None:
        hidden = torch.randn(1, 12, HIDDEN)
        window = span_mask(12, (3, 7))
        assert torch.allclose(pool_masked(hidden, window, "mean"), hidden[:, 3:7].mean(dim=1))
        assert torch.equal(pool_masked(hidden, window, "last"), hidden[:, 6])

    @pytest.mark.parametrize("span", [(5, 11), (4, 4), (-1, 3), (7, 2)])
    def test_a_span_outside_the_prompt_is_refused(self, span: tuple[int, int]) -> None:
        with pytest.raises(FullWeightsCellError, match="does not fit"):
            span_mask(10, span)

    def test_an_unknown_pooling_is_refused(self) -> None:
        with pytest.raises(FullWeightsCellError, match="unknown pooling"):
            pool_masked(torch.zeros(1, 4, HIDDEN), span_mask(4, (0, 4)), "median")


class TestLoadingGate:
    def report(self, **overrides: list[Any]) -> dict[str, list[Any]]:
        base: dict[str, list[Any]] = {
            "missing_keys": ["model.visual.blocks.0.attn.qkv.weight"],
            "unexpected_keys": ["mtp.fc.weight"],
            "mismatched_keys": [],
            "error_msgs": [],
        }
        base.update(overrides)
        return base

    def test_tower_only_gaps_pass_and_are_counted(self) -> None:
        report = assert_language_model_complete(
            self.report(), label="allenai/tmax-9b@step_500", n_language_model_keys=426
        )
        assert report == LoadingReport(
            n_language_model_loaded=426, n_missing_tower=1, n_unexpected_tower=1
        )

    def test_a_missing_language_model_key_is_refused(self) -> None:
        report = self.report(missing_keys=["model.language_model.layers.3.mlp.down_proj.weight"])
        with pytest.raises(FullWeightsCellError, match="did not load whole"):
            assert_language_model_complete(report, label="x", n_language_model_keys=426)

    def test_a_renamed_key_is_refused_by_its_unexpected_half_alone(self) -> None:
        """The smoke's sabotage: one renamed safetensors key is one unexpected language-model key."""
        report = self.report(
            unexpected_keys=["model.language_model.layers.3.mlp.down_proj.weight_RENAMED"]
        )
        with pytest.raises(FullWeightsCellError, match="1 unexpected"):
            assert_language_model_complete(report, label="x", n_language_model_keys=426)

    def test_a_mismatched_shape_is_refused(self) -> None:
        report = self.report(mismatched_keys=[("model.language_model.norm.weight", (1,), (2,))])
        with pytest.raises(FullWeightsCellError, match="mismatched"):
            assert_language_model_complete(report, label="x", n_language_model_keys=1)

    def test_a_key_outside_every_known_namespace_is_refused(self) -> None:
        report = self.report(unexpected_keys=["model.layers.0.self_attn.q_proj.weight"])
        with pytest.raises(FullWeightsCellError, match=r"not laid out like a Qwen3\.5 release"):
            assert_language_model_complete(report, label="x", n_language_model_keys=1)

    def test_a_report_missing_a_field_is_refused(self) -> None:
        report = self.report()
        del report["error_msgs"]
        with pytest.raises(FullWeightsCellError, match="lacks"):
            assert_language_model_complete(report, label="x", n_language_model_keys=1)


class TestCellSpecs:
    def test_a_hub_spec_needs_a_revision(self) -> None:
        spec = parse_cell_spec("tmax-9b:500=allenai/tmax-9b@step_500")
        assert (spec.arm, spec.step) == ("tmax-9b", 500)
        assert spec.source == FullWeightsSource("allenai/tmax-9b", "step_500", None)
        with pytest.raises(FullWeightsCellError, match="revision is required"):
            parse_cell_spec("tmax-9b:500=allenai/tmax-9b")

    def test_a_local_directory_is_a_local_source(self, tmp_path: Path) -> None:
        spec = parse_cell_spec(f"permuted:500={tmp_path}")
        assert spec.source.local_dir == tmp_path
        assert spec.source.revision is None

    @pytest.mark.parametrize(
        "spec", ["base=Qwen/x@r", "base:zero=Qwen/x@r", "base:0", ":0=Qwen/x@r"]
    )
    def test_a_malformed_spec_is_refused(self, spec: str) -> None:
        with pytest.raises(FullWeightsCellError, match="--cell expects"):
            parse_cell_spec(spec)

    def test_replica_declarations_attach_and_translate(self) -> None:
        specs = apply_replica_declarations(
            [hub_spec("base", 0), hub_spec("base-mirror", 0)],
            dict([parse_replica_spec("base-mirror:0=base:0")]),
        )
        assert [spec.replica_of for spec in specs] == [None, "base:0"]
        assert games_label("base-mirror:0") == "base-mirror/step-0"
        with pytest.raises(FullWeightsCellError, match="not on the agenda"):
            apply_replica_declarations([hub_spec("base", 0)], {"ghost:0": "base:0"})


class TestFingerprintDeclarations:
    def test_two_undeclared_cells_over_one_fingerprint_are_refused(self, tmp_path: Path) -> None:
        resolved = {
            "base:0": (hub_spec("base", 0), facts_with("aa", tmp_path)),
            "base-mirror:0": (hub_spec("base-mirror", 0), facts_with("aa", tmp_path)),
        }
        with pytest.raises(FullWeightsCellError, match="exactly one of them may stand undeclared"):
            assert_fingerprints_declared(resolved)

    def test_a_declared_pair_passes(self, tmp_path: Path) -> None:
        resolved = {
            "base:0": (hub_spec("base", 0), facts_with("aa", tmp_path)),
            "base-mirror:0": (
                hub_spec("base-mirror", 0, replica_of="base:0"),
                facts_with("aa", tmp_path),
            ),
            "tmax-9b:500": (hub_spec("tmax-9b", 500), facts_with("bb", tmp_path)),
        }
        assert_fingerprints_declared(resolved)

    def test_a_declaration_over_different_bytes_is_refused(self, tmp_path: Path) -> None:
        resolved = {
            "base:0": (hub_spec("base", 0), facts_with("aa", tmp_path)),
            "tmax-9b:500": (
                hub_spec("tmax-9b", 500, replica_of="base:0"),
                facts_with("bb", tmp_path),
            ),
        }
        with pytest.raises(FullWeightsCellError, match="their weights differ"):
            assert_fingerprints_declared(resolved)


class TestAmplifiedSidecar:
    def sidecar(self, **overrides: object) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": 3,
            "kind": "task-arithmetic-amplification",
            "label": "permuted-1.0",
            "alpha": 1.0,
            "delta_transform": "permute-axes",
            "permutation_seed": 20260903,
            "storage_dtype": "release",
            "base": {"label": "base", "weights_fingerprint": "aa"},
            "rl": {"label": "step_500", "weights_fingerprint": "bb"},
            "tensor_deltas": [{"name": "x", "delta_norm": 1.0}],
        }
        body.update(overrides)
        return body

    def test_an_amplified_unit_is_recognised_without_its_delta_table(self, tmp_path: Path) -> None:
        (tmp_path / WEIGHTS_PROVENANCE_FILENAME).write_text(json.dumps(self.sidecar()))
        summary = read_amplified_sidecar(facts_with("cc", tmp_path))
        assert summary is not None
        assert (summary["alpha"], summary["delta_transform"], summary["permutation_seed"]) == (
            1.0,
            "permute-axes",
            20260903,
        )
        assert "tensor_deltas" not in summary

    def test_a_plain_checkpoint_has_no_sidecar(self, tmp_path: Path) -> None:
        assert read_amplified_sidecar(facts_with("cc", tmp_path)) is None

    def test_a_sidecar_that_is_not_an_amplification_is_refused(self, tmp_path: Path) -> None:
        body = self.sidecar()
        del body["alpha"]
        (tmp_path / WEIGHTS_PROVENANCE_FILENAME).write_text(json.dumps(body))
        with pytest.raises(FullWeightsCellError, match="does not describe an amplified unit"):
            read_amplified_sidecar(facts_with("cc", tmp_path))

    def test_an_unknown_transform_is_refused(self, tmp_path: Path) -> None:
        body = self.sidecar(delta_transform="rotate")
        (tmp_path / WEIGHTS_PROVENANCE_FILENAME).write_text(json.dumps(body))
        with pytest.raises(FullWeightsCellError, match="delta_transform"):
            read_amplified_sidecar(facts_with("cc", tmp_path))


class TestCoherence:
    def test_an_unavailable_triple_is_nan_with_a_reason(self) -> None:
        triple = CoherenceTriple.unavailable("never probed")
        assert math.isnan(triple.gradable_share)
        assert triple.to_payload() == {
            "gradable_share": None,
            "truncation_share": None,
            "well_formed_share": None,
            "source": None,
            "reason": "never probed",
        }

    def test_a_table_loads_and_refuses_bad_shares(self, tmp_path: Path) -> None:
        good = tmp_path / "good.json"
        good.write_text(
            json.dumps(
                {
                    "tmax-9b:500": {
                        "gradable_share": 0.9,
                        "truncation_share": 0.04,
                        "well_formed_share": 0.97,
                        "source": "phase1/unit-2/summary.json",
                    }
                }
            )
        )
        table = load_coherence_table(good)
        assert table["tmax-9b:500"].available
        assert table["tmax-9b:500"].source == "phase1/unit-2/summary.json"
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps(
                {"x:1": {"gradable_share": 1.5, "truncation_share": 0.0, "well_formed_share": 1.0}}
            )
        )
        with pytest.raises(FullWeightsCellError, match="outside"):
            load_coherence_table(bad)


# --------------------------------------------------------------------------------------


def identity() -> CellIdentity:
    return CellIdentity(
        base_model="tiny@rev",
        stimuli_sha256="digest",
        rendered_sha256="rendered",
        layer_convention=LAYER_CONVENTION_POST_BLOCK,
        n_layers=N_LAYERS,
        hidden_size=HIDDEN,
        batch_size=1,
        compute_dtype="float32",
        store_dtype="float32",
        stimulus_render=STIMULUS_RENDER_VERBATIM,
    )


def write_synthetic_cell(  # noqa: PLR0913 - one knob per guard under test
    root: Path,
    arm: str,
    step: int,
    *,
    fingerprint: str | None,
    replica_of: str | None,
    kernel: dict[str, str],
    shift: float,
) -> None:
    activations = torch.arange(2 * N_LAYERS * HIDDEN, dtype=torch.float32).reshape(
        2, N_LAYERS, HIDDEN
    )
    provenance: dict[str, Any] = {DELTANET_KERNEL_FIELD: kernel, REPLICA_OF_FIELD: replica_of}
    if fingerprint is not None:
        provenance[WEIGHTS_FINGERPRINT_FIELD] = fingerprint
    write_cell(
        step_dir(root, arm, step),
        arm=arm,
        step=step,
        identity=identity(),
        rows={"s": RowIndex(("a", "b"), ("A", "B"), ("p", "p"), (3, 3))},
        activations={("s", "mean"): activations + shift, ("s", "last"): activations - shift},
        applied_adapter_weights=None,
        adapter_weights_sha256=None,
        provenance=provenance,
    )


class TestLadderGuards:
    def test_a_declared_replica_pair_reads_exactly_zero(self, tmp_path: Path) -> None:
        write_synthetic_cell(
            tmp_path, "base", 0, fingerprint="aa", replica_of=None, kernel=FUSED_KERNELS, shift=0.0
        )
        write_synthetic_cell(
            tmp_path,
            "mirror",
            0,
            fingerprint="aa",
            replica_of="base:0",
            kernel=FUSED_KERNELS,
            shift=0.0,
        )
        write_synthetic_cell(
            tmp_path,
            "tmax",
            500,
            fingerprint="bb",
            replica_of=None,
            kernel=FUSED_KERNELS,
            shift=0.5,
        )
        ladder = load_full_weights_ladder(tmp_path)
        assert [cell.label for cell in ladder.cells] == [
            "base/step-0",
            "mirror/step-0",
            "tmax/step-500",
        ]
        base, mirror, tmax = ladder.cells
        assert float((base.matrix("s", "mean") - mirror.matrix("s", "mean")).abs().max()) == 0.0
        assert float((base.matrix("s", "mean") - tmax.matrix("s", "mean")).abs().max()) == 0.5

    def test_a_replica_pair_that_moved_is_refused(self, tmp_path: Path) -> None:
        write_synthetic_cell(
            tmp_path, "base", 0, fingerprint="aa", replica_of=None, kernel=FUSED_KERNELS, shift=0.0
        )
        write_synthetic_cell(
            tmp_path,
            "mirror",
            0,
            fingerprint="aa",
            replica_of="base:0",
            kernel=FUSED_KERNELS,
            shift=1e-3,
        )
        with pytest.raises(CellFormatError, match="not deterministic"):
            load_full_weights_ladder(tmp_path)

    def test_an_undeclared_duplicate_fingerprint_is_refused(self, tmp_path: Path) -> None:
        write_synthetic_cell(
            tmp_path, "base", 0, fingerprint="aa", replica_of=None, kernel=FUSED_KERNELS, shift=0.0
        )
        write_synthetic_cell(
            tmp_path,
            "mirror",
            0,
            fingerprint="aa",
            replica_of=None,
            kernel=FUSED_KERNELS,
            shift=0.0,
        )
        with pytest.raises(CellFormatError, match="neither declares replica_of"):
            load_full_weights_ladder(tmp_path)

    def test_identical_activations_under_different_weights_are_refused(
        self, tmp_path: Path
    ) -> None:
        write_synthetic_cell(
            tmp_path, "base", 0, fingerprint="aa", replica_of=None, kernel=FUSED_KERNELS, shift=0.0
        )
        write_synthetic_cell(
            tmp_path,
            "tmax",
            500,
            fingerprint="bb",
            replica_of=None,
            kernel=FUSED_KERNELS,
            shift=0.0,
        )
        with pytest.raises(CellFormatError, match="did not reach the forward"):
            load_full_weights_ladder(tmp_path)

    def test_two_kernel_bindings_in_one_ladder_are_refused(self, tmp_path: Path) -> None:
        write_synthetic_cell(
            tmp_path, "base", 0, fingerprint="aa", replica_of=None, kernel=FUSED_KERNELS, shift=0.0
        )
        write_synthetic_cell(
            tmp_path,
            "tmax",
            500,
            fingerprint="bb",
            replica_of=None,
            kernel=TORCH_KERNELS,
            shift=0.5,
        )
        with pytest.raises(ValueError, match="different Gated DeltaNet kernel bindings"):
            load_full_weights_ladder(tmp_path)

    def test_a_cell_without_a_fingerprint_is_not_a_full_weights_cell(self, tmp_path: Path) -> None:
        write_synthetic_cell(
            tmp_path, "base", 0, fingerprint=None, replica_of=None, kernel=FUSED_KERNELS, shift=0.0
        )
        with pytest.raises(CellFormatError, match="records no weights_fingerprint"):
            load_full_weights_ladder(tmp_path)


# --------------------------------------------------------------------------------------


class TestCaptureEndToEnd:
    def capture(
        self,
        model: TinyComposite,
        stimuli: list[Stimulus],
        table: SpanTable | None,
        spans: list[str],
    ) -> tuple[dict[str, RowIndex], dict[tuple[str, str], torch.Tensor], dict[str, dict[str, str]]]:
        tokenizer = TinyTokenizer()
        rendered = {stimulus.stimulus_id: stimulus.text for stimulus in stimuli}
        counts = {
            stimulus.stimulus_id: len(cast("list[int]", tokenizer(stimulus.text)["input_ids"]))
            for stimulus in stimuli
        }
        return capture_full_weights_cell(
            cast("Any", model),
            cast("Any", tokenizer),
            grouped=group_by_set(stimuli),
            rendered=rendered,
            counts_by_id=counts,
            poolings=["mean", "last"],
            span_table=table,
            span_names=spans,
        )

    def test_the_full_span_set_equals_the_plain_set_bit_for_bit(self) -> None:
        stimuli = verbatim_stimuli()
        table = span_table_for(TinyTokenizer(), stimuli)
        rows, activations, absences = self.capture(
            TinyComposite(), stimuli, table, ["full", "head"]
        )
        full = derived_set_name(STIMULUS_SET, "full")
        for pooling in ("mean", "last"):
            assert activations[STIMULUS_SET, pooling].shape == (4, N_LAYERS, HIDDEN)
            assert torch.equal(activations[full, pooling], activations[STIMULUS_SET, pooling])
        assert rows[full].stimulus_ids == rows[STIMULUS_SET].stimulus_ids
        assert rows[full].token_counts == rows[STIMULUS_SET].token_counts
        assert absences == {}

    def test_a_window_set_pools_the_window_of_the_positionwise_capture(self) -> None:
        stimuli = verbatim_stimuli(n_pairs=1)
        tokenizer = TinyTokenizer()
        table = span_table_for(tokenizer, stimuli)
        model = TinyComposite()
        rows, activations, _ = self.capture(model, stimuli, table, ["head"])
        head = derived_set_name(STIMULUS_SET, "head")
        assert rows[head].token_counts == (3, 3)
        for row, stimulus in enumerate(stimuli):
            encoded = cast(
                "Encoding", tokenizer([stimulus.text], return_tensors="pt", padding=True)
            )
            positionwise = capture_positionwise_activations(
                cast("Any", model), encoded["input_ids"], encoded["attention_mask"]
            )
            for layer in range(N_LAYERS):
                window = positionwise[layer][0, :3]
                assert torch.allclose(activations[head, "mean"][row, layer], window.mean(dim=0))
                assert torch.equal(activations[head, "last"][row, layer], window[-1])

    def test_an_absent_span_drops_the_row_and_records_why(self, tmp_path: Path) -> None:
        stimuli = verbatim_stimuli()
        table = span_table_for(TinyTokenizer(), stimuli, drop_head_from="p1--honest-inline")
        rows, activations, absences = self.capture(TinyComposite(), stimuli, table, ["head"])
        head = derived_set_name(STIMULUS_SET, "head")
        assert rows[head].n_rows == 3
        assert "p1--honest-inline" not in rows[head].stimulus_ids
        assert activations[head, "mean"].shape == (3, N_LAYERS, HIDDEN)
        assert absences == {head: {"p1--honest-inline": "the rendering rewrote the whole check"}}
        cell_dir = write_cell(
            step_dir(tmp_path, "base", 0),
            arm="base",
            step=0,
            identity=identity(),
            rows=rows,
            activations=activations,
            applied_adapter_weights=None,
            adapter_weights_sha256=None,
            provenance={},
        ).parent
        assert read_cell(cell_dir).stimulus_sets == (STIMULUS_SET, head)

    def test_a_span_nobody_declares_is_refused(self) -> None:
        stimuli = verbatim_stimuli(n_pairs=1)
        table = span_table_for(TinyTokenizer(), stimuli)
        for stimulus in stimuli:
            del table.spans[stimulus.stimulus_id]["head"]
        with pytest.raises(FullWeightsCellError, match="declares span 'head'"):
            self.capture(TinyComposite(), stimuli, table, ["head"])

    def test_different_weights_move_the_activations(self) -> None:
        stimuli = verbatim_stimuli(n_pairs=1)
        _, base, _ = self.capture(TinyComposite(shift=0.0), stimuli, None, [])
        _, moved, _ = self.capture(TinyComposite(shift=0.25), stimuli, None, [])
        _, again, _ = self.capture(TinyComposite(shift=0.0), stimuli, None, [])
        assert torch.equal(base[STIMULUS_SET, "mean"], again[STIMULUS_SET, "mean"])
        assert float((base[STIMULUS_SET, "mean"] - moved[STIMULUS_SET, "mean"]).abs().max()) > 0.0

    def test_sidecar_ids_that_disagree_with_the_tokenizer_are_refused(self) -> None:
        stimuli = verbatim_stimuli(n_pairs=1)
        tokenizer = TinyTokenizer()
        table = span_table_for(tokenizer, stimuli)
        rendered = {stimulus.stimulus_id: stimulus.text for stimulus in stimuli}
        assert_ids_match_sidecar(cast("Any", tokenizer), rendered, table)
        ids = table.input_ids["p0--rigged-inline"]
        table.input_ids["p0--rigged-inline"] = (*ids[:-1], ids[-1] + 1)
        with pytest.raises(FullWeightsCellError, match="the two sequences differ"):
            assert_ids_match_sidecar(cast("Any", tokenizer), rendered, table)


class TestTeacherForcedText:
    def test_the_completion_follows_the_template_think_line(self) -> None:
        text = teacher_forced_text(cast("Any", TinyTokenizer()), "solve it", "I reason.\n</think>x")
        assert text == f"{TURN_PREFIX}solve it{TURN_SUFFIX}I reason.\n</think>x"

    def test_a_template_that_does_not_open_think_is_refused(self) -> None:
        with pytest.raises(FullWeightsCellError, match="not with"):
            teacher_forced_text(
                cast("Any", TinyTokenizer()), "solve it", "x", enable_thinking=False
            )
