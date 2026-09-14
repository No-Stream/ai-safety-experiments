"""Pin the lens ladder's arithmetic and its two refusals, without jlens, CUDA or any weights.

`jlens` is deliberately absent from this repo's lockfile and the fit itself is GPU-only, so what is
testable offline is everything around the fit -- and that is where the silent failures live.

:class:`TestSeqLenPlan` covers the thing that would otherwise cost a whole ladder. The fit truncates
every corpus item to `max_seq_len`, whose library default is 128 tokens against stimuli several hundred
long, so inheriting it fits on the opening third of each prompt and discards the continuation that
carries the contrast, with nothing going red. The derivation itself is now shared with the
reward-hacking lens stage and its arithmetic is tested beside it in `test_interp_jacobian`; what is
pinned here is that this ladder uses it under the shared ceiling.

:class:`TestLensCacheKeyForCell` also pins the C4 split: fitting on the adapter applied at runtime and
fitting on a bf16 merge of it are different lenses (0.6-1.8% relative Frobenius apart per layer,
against a 0.33% floor for two wrappers over the same weights), so they take different cache keys and
the ladder's default is the un-merged one.

:class:`TestCorpusSplit` is the circular-measurement refusal: reconstruction quality read on the
prompts the lens was fitted on is not a quality read, and it looks identical to one.

:class:`TestMaterializeModelDir` pins that the base cell merges nothing (its hub id is already a plain
model) while an adapted cell does, since that merge is where the ~64% delta realization is spent; and
that a merged export is reused only when its provenance sidecar names this cell's adapter and base,
because the lens cache would otherwise publish a lens fitted on a leftover merge under the new
adapter's key.

:class:`TestDecodeDirection` runs the transport-and-decode leg against stubs, including both refusals
that exist to fail before a fit rather than after one: a layer the directions file does not carry, and
the top layer, which has no fitted Jacobian because a lens fits source layers strictly below its
target.

:class:`TestLadderRunEndToEnd` drives `run` itself over a base cell plus an adapted one with every
card-bound seam stubbed, which the other `run` tests could not reach: each of them dies at a refusal or
at "jlens is not importable" before a cell is fitted, so the per-cell wiring used to be covered by the
L4 smoke alone. It pins when the DeltaNet kernel binding is read (once, before the first fit -- read
per cell inside the report-writing `finally`, an upstream dispatch change surfaced hours late and took
the succeeded cell's report with it), that the report reaches disk after every cell, and that a merged
run's base cell does not label itself the un-merged arm. :class:`TestLensTargetArm` is the same last
claim at the unit the label is derived from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from games import interp_lens_ladder
from games.interp_capture import LadderCell
from games.interp_cells import BASE_ARM, BASE_STEP, Stimulus, sha256_of_file
from games.interp_lens_ladder import (
    LENS_MODEL_MERGED,
    LENS_MODEL_UNMERGED,
    MERGE_PROVENANCE_FILENAME,
    build_parser,
    decode_direction,
    derive_max_seq_len,
    lens_cache_key_for_cell,
    materialize_model_dir,
    pair_divergence_indices,
    resolve_cells,
    run,
    split_corpus,
)
from games.lora import AttachedAdapter
from reward_hacking.interp.jacobian import (
    DEFAULT_MAX_SEQ_LEN_CEILING,
    JLENS_COMMIT,
    JLENS_LOAD_PATH,
    JLENS_UNMERGED_LOAD_PATH,
    digest_strings,
)

if TYPE_CHECKING:
    from pathlib import Path

HIDDEN = 8
VOCAB = 12
N_LAYERS = 3


class StubLens:
    """A lens whose transport is a fixed linear map per layer, so a decode is checkable by hand."""

    def __init__(self) -> None:
        self.transported: list[tuple[int, float]] = []

    def transport(self, direction: torch.Tensor, layer: int) -> torch.Tensor:
        """Record the call and scale the direction by the layer, so layers are distinguishable."""
        self.transported.append((layer, float(direction.norm())))
        return direction * float(layer + 1)


class StubModel:
    """A model whose unembedding is a fixed projection to a small vocabulary."""

    def unembed(self, transported: torch.Tensor) -> torch.Tensor:
        """Project to `[vocab]` logits, deterministically."""
        weights = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN) / 10.0
        return weights @ transported


class StubTokenizer:
    """Decodes an id to a readable token name."""

    def convert_ids_to_tokens(self, token_id: int) -> str:
        """Name a token by its id, which is enough to tell two readouts apart."""
        return f"tok{token_id}"


def write_adapter_dir(root: Path, name: str, weights: bytes) -> Path:
    """A checkpoint directory with the two files a merge and a cache key digest."""
    adapter_dir = root / name
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(weights)
    (adapter_dir / "adapter_config.json").write_text('{"r": 16, "lora_alpha": 32}')
    return adapter_dir


def write_directions(path: Path, *, n_layers: int = N_LAYERS) -> Path:
    """Write a `{layer: tensor}` directions file, the shape games.interp_trajectory saves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    directions = {
        layer: torch.arange(HIDDEN, dtype=torch.float32) + float(layer) for layer in range(n_layers)
    }
    torch.save(directions, path)
    return path


class TestSeqLenPlan:
    """The ladder's window comes from the shared derivation, whose own tests live beside it.

    `reward_hacking.interp.jacobian.derive_max_seq_len` is now one function serving both lens ladders
    (`TestDeriveMaxSeqLen` there covers the arithmetic and its refusals); what is this module's to pin
    is that it uses that function and that its ceiling default is the shared one, since a ladder that
    quietly kept the 128-token library default would fit on the stimuli's shared stems.
    """

    def test_the_ladder_derives_its_window_through_the_shared_function(self) -> None:
        plan = derive_max_seq_len([320, 380, 420], ceiling=DEFAULT_MAX_SEQ_LEN_CEILING)
        assert plan.max_seq_len == 420
        assert plan.n_truncated == 0

    def test_the_shared_ceiling_covers_this_corpus_whole(self) -> None:
        """The games stimuli run 368-451 tokens, so the shared 2048 ceiling never truncates them."""
        assert DEFAULT_MAX_SEQ_LEN_CEILING >= 452


class TestCorpusSplit:
    def test_the_halves_are_disjoint_and_ordered(self) -> None:
        prompts = [f"p{index}" for index in range(10)]
        split = split_corpus(prompts, [1] * 10, max_fit_prompts=6, n_eval_prompts=3)
        assert split.fit_prompts == tuple(prompts[:6])
        assert split.eval_prompts == tuple(prompts[6:9])
        assert not set(split.fit_prompts) & set(split.eval_prompts)

    def test_a_corpus_too_small_to_split_is_refused_with_the_arithmetic(self) -> None:
        with pytest.raises(ValueError, match=r"5 prompts but a disjoint split needs 18"):
            split_corpus(
                [f"p{index}" for index in range(5)], [1] * 5, max_fit_prompts=10, n_eval_prompts=8
            )


class TestPairDivergence:
    def test_the_divergence_index_is_the_first_differing_token(self) -> None:
        stimuli = [
            Stimulus("a", "s", "A", "p", "stem", "left"),
            Stimulus("b", "s", "B", "p", "stem", "right"),
        ]
        indices = pair_divergence_indices(stimuli, {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 9, 9]})
        assert indices == {"p": 3}

    def test_a_shorter_side_diverges_where_it_ends(self) -> None:
        stimuli = [
            Stimulus("a", "s", "A", "p", "stem", "left"),
            Stimulus("b", "s", "B", "p", "stem", "right"),
        ]
        assert pair_divergence_indices(stimuli, {"a": [1, 2, 3], "b": [1, 2, 3, 4]}) == {"p": 3}

    def test_an_identical_pair_is_refused(self) -> None:
        """A pair that renders identically carries no contrast for any window to keep."""
        stimuli = [
            Stimulus("a", "s", "A", "p", "stem", "same"),
            Stimulus("b", "s", "B", "p", "stem", "same"),
        ]
        with pytest.raises(ValueError, match="render identically"):
            pair_divergence_indices(stimuli, {"a": [1, 2, 3], "b": [1, 2, 3]})


class TestResolveCells:
    def test_the_base_anchor_leads_the_agenda(self, tmp_path: Path) -> None:
        for step in (10, 20):
            (tmp_path / f"checkpoint-{step}").mkdir(parents=True)
        cells = resolve_cells([f"arm={tmp_path}"], None, skip_base=False)
        assert [cell.label for cell in cells] == ["base/step-0", "arm/step-10", "arm/step-20"]
        assert cells[0].adapter_dir is None
        assert cells[1].adapter_dir is not None

    def test_skipping_the_base_drops_the_anchor(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoint-10").mkdir(parents=True)
        cells = resolve_cells([f"arm={tmp_path}"], None, skip_base=True)
        assert [cell.label for cell in cells] == ["arm/step-10"]

    def test_nothing_to_fit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nothing to fit"):
            resolve_cells([], None, skip_base=True)


class TestMaterializeModelDir:
    @staticmethod
    def _stub_export(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
        """Swap the merge for a stub that records which adapter it was asked to merge."""
        merges: list[Path] = []

        def stub_export(adapter_dir: Path, base: str, out_dir: Path, **kwargs: Any) -> Path:
            del base, kwargs
            merges.append(adapter_dir)
            out_dir.mkdir(parents=True)
            (out_dir / "config.json").write_text("{}")
            return out_dir

        monkeypatch.setattr(interp_lens_ladder, "export_merged_checkpoint", stub_export)
        return merges

    @staticmethod
    def _materialize(
        cell: LadderCell, merge_root: Path, base_weights_identity: str = "hf:abc"
    ) -> Path:
        return materialize_model_dir(
            cell,
            base_model="base",
            base_weights_identity=base_weights_identity,
            merge_root=merge_root,
            dtype=torch.bfloat16,
        )

    def test_the_base_cell_merges_nothing(self, tmp_path: Path) -> None:
        cell = LadderCell(arm=BASE_ARM, step=BASE_STEP, adapter_dir=None)
        resolved = materialize_model_dir(
            cell,
            base_model="Qwen/Qwen3.5-2B",
            base_weights_identity="hf:abc",
            merge_root=tmp_path,
            dtype=torch.bfloat16,
        )
        assert str(resolved) == "Qwen/Qwen3.5-2B"
        assert not list(tmp_path.iterdir())

    def test_an_adapted_cell_merges_once_and_reuses_after(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        merges = self._stub_export(monkeypatch)
        adapter_dir = write_adapter_dir(tmp_path, "checkpoint-70", b"weights")
        cell = LadderCell(arm="arm", step=70, adapter_dir=adapter_dir)
        merge_root = tmp_path / "merged"
        first = self._materialize(cell, merge_root)
        second = self._materialize(cell, merge_root)
        assert first == second == merge_root / "arm" / "step-70"
        assert merges == [adapter_dir]
        assert json.loads((first / MERGE_PROVENANCE_FILENAME).read_text()) == {
            "adapter_weights_sha256": sha256_of_file(adapter_dir / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256_of_file(adapter_dir / "adapter_config.json"),
            "base_model": "base",
            "base_weights_identity": "hf:abc",
            "merge_dtype": "bfloat16",
        }

    def test_a_checkpoint_replaced_at_the_same_step_is_re_merged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cache-poisoning case: same arm and step, other adapter bytes, a leftover export."""
        merges = self._stub_export(monkeypatch)
        adapter_dir = write_adapter_dir(tmp_path, "checkpoint-70", b"weights")
        cell = LadderCell(arm="arm", step=70, adapter_dir=adapter_dir)
        merge_root = tmp_path / "merged"
        out_dir = self._materialize(cell, merge_root)
        stale_sidecar = json.loads((out_dir / MERGE_PROVENANCE_FILENAME).read_text())

        (adapter_dir / "adapter_model.safetensors").write_bytes(b"weightz")
        assert self._materialize(cell, merge_root) == out_dir
        assert merges == [adapter_dir, adapter_dir]
        fresh_sidecar = json.loads((out_dir / MERGE_PROVENANCE_FILENAME).read_text())
        assert fresh_sidecar["adapter_weights_sha256"] == sha256_of_file(
            adapter_dir / "adapter_model.safetensors"
        )
        assert fresh_sidecar != stale_sidecar

    def test_a_moved_base_revision_is_re_merged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        merges = self._stub_export(monkeypatch)
        adapter_dir = write_adapter_dir(tmp_path, "checkpoint-70", b"weights")
        cell = LadderCell(arm="arm", step=70, adapter_dir=adapter_dir)
        merge_root = tmp_path / "merged"
        self._materialize(cell, merge_root, base_weights_identity="hf:abc")
        self._materialize(cell, merge_root, base_weights_identity="hf:def")
        assert merges == [adapter_dir, adapter_dir]

    def test_an_export_without_a_sidecar_is_re_merged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An export from before the sidecar existed cannot say which adapter made it."""
        merges = self._stub_export(monkeypatch)
        adapter_dir = write_adapter_dir(tmp_path, "checkpoint-70", b"weights")
        cell = LadderCell(arm="arm", step=70, adapter_dir=adapter_dir)
        merge_root = tmp_path / "merged"
        leftover = merge_root / "arm" / "step-70"
        leftover.mkdir(parents=True)
        (leftover / "config.json").write_text("{}")
        (leftover / "model.safetensors").write_bytes(b"whose?")

        assert self._materialize(cell, merge_root) == leftover
        assert merges == [adapter_dir]
        assert not (leftover / "model.safetensors").exists(), "the leftover was replaced, not kept"
        assert (leftover / MERGE_PROVENANCE_FILENAME).is_file()


class _StubJlensModule:
    """Stands in for the jlens module: `from_hf` records what it was handed and hands back a marker."""

    def __init__(self) -> None:
        self.wrapped: list[object] = []

    def from_hf(self, model: object, tokenizer: object) -> str:
        """Record the model wrapped, so a test can tell the base from an adapter-injected tree."""
        del tokenizer
        self.wrapped.append(model)
        return f"wrapped-{len(self.wrapped)}"


class _StubPeftModel:
    """The wrapper `attach_adapter` returns, whose `get_base_model` names the tree to fit through.

    It hands back a marker DISTINCT from the base object it was built over. In real PEFT the two are
    the same object -- injection happens in place, and probe I6's identity check confirmed it -- so this
    is not a divergence anyone has seen; it is what makes "the fit is wrapped around whatever
    `get_base_model()` returned" a checkable claim rather than one that holds by coincidence.
    """

    def __init__(self, base: object, adapter_dir: Path) -> None:
        self.injected = f"injected-over-{id(base)}-{adapter_dir.name}"
        self.adapter_dir = adapter_dir
        self.applied_adapter_weights = 4

    def get_base_model(self) -> object:
        """The tree PEFT injected into, which is the object the fit must run through."""
        return self.injected


@dataclass(frozen=True)
class _AttachCall:
    """One `attach_adapter` call, including the argument that decides which PEFT path runs.

    `existing` is the whole point: `attach_adapter` re-points an already-injected tree when it is given
    one and calls `PeftModel.from_pretrained` when it is not, and injecting a second time over an
    injected tree is a silent wrong answer rather than an error. A stub that discarded it could not tell
    the two apart.
    """

    adapter_dir: Path
    existing: object
    returned: object


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the base load, the adapter attach and the tokenizer, recording every call."""
    loads: list[str] = []
    attaches: list[_AttachCall] = []
    base = object()

    def load_base(model_id: str, *, dtype: Any, device: Any, revision: str | None = None) -> object:
        del dtype, device
        loads.append(model_id)
        assert revision is None
        return base

    def attach(
        model: object, adapter_dir: Path, base_model_id: str, *, existing: Any = None
    ) -> Any:
        del base_model_id
        peft_model = _StubPeftModel(model, adapter_dir)
        attaches.append(
            _AttachCall(adapter_dir=adapter_dir, existing=existing, returned=peft_model)
        )
        return AttachedAdapter(peft_model=cast("Any", peft_model), applied_adapter_weights=4)

    monkeypatch.setattr(interp_lens_ladder, "load_adapter_base", load_base)
    monkeypatch.setattr(interp_lens_ladder, "attach_adapter", attach)
    monkeypatch.setattr(
        interp_lens_ladder,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *a, **k: "tok"),
    )
    return {"base": base, "loads": loads, "attaches": attaches}


class TestLensTargetArm:
    """Which `--lens-model` arm a target belongs to, and why that is not the same as having an export."""

    def test_a_merged_arm_target_without_an_export_is_still_the_merged_arm(self) -> None:
        """The merged arm's base cell: nothing was materialised, and the lens is still the merged one.

        `materialize_model_dir` returns the hub id unchanged for the base cell, so there is no export to
        record or delete, and `merged_dir` stays None. The lens is loaded through the plain causal-LM
        loader all the same, which is what `load_path` records and what separates the two lenses.
        """
        target = interp_lens_ladder.LensTarget(
            model=object(),
            tokenizer=object(),
            label="Qwen/Qwen3.5-2B",
            load_path=JLENS_LOAD_PATH,
            merged_dir=None,
        )
        assert target.lens_model == LENS_MODEL_MERGED
        assert target.merged is False, "no export exists, and the field says so truthfully"

    def test_a_runtime_adapted_target_is_the_un_merged_arm(self) -> None:
        target = interp_lens_ladder.LensTarget(
            model=object(),
            tokenizer=object(),
            label="Qwen/Qwen3.5-2B + trained/step-10 (un-merged)",
            load_path=JLENS_UNMERGED_LOAD_PATH,
            merged_dir=None,
        )
        assert target.lens_model == LENS_MODEL_UNMERGED
        assert target.merged is False


class TestLadderLensModels:
    """The C4 un-merged path: one resident base, the adapter re-pointed per cell, no merge on disk.

    Offline: the loaders and `jlens.from_hf` are stubs, because what is being pinned is the SEQUENCE --
    one base load for the whole ladder, an attach per adapted cell, a re-wrap after each attach so the
    freshly injected LoRA parameters are frozen, and no merged export to delete. The fit itself needs a
    card and is covered by the L4 smoke.
    """

    def _models(self, jl: _StubJlensModule, tmp_path: Path) -> Any:
        return interp_lens_ladder.LadderLensModels(
            base_model="Qwen/Qwen3.5-0.8B",
            dtype=torch.bfloat16,
            jl=cast("Any", jl),
            lens_model=LENS_MODEL_UNMERGED,
            merge_root=tmp_path / "merged",
            base_weights_identity="hf:abc",
        )

    def test_base_revision_is_threaded_to_weights_and_tokenizer_loaders(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        load_calls: list[tuple[str, str | None]] = []
        tokenizer_calls: list[tuple[str, str | None]] = []
        base = object()

        def load_base(
            model_id: str, *, dtype: Any, device: Any, revision: str | None = None
        ) -> object:
            del dtype, device
            load_calls.append((model_id, revision))
            return base

        def load_tokenizer(model_id: str, **kwargs: object) -> str:
            tokenizer_calls.append((model_id, cast("str | None", kwargs["revision"])))
            return "tokenizer"

        monkeypatch.setattr(interp_lens_ladder, "load_adapter_base", load_base)
        monkeypatch.setattr(
            interp_lens_ladder,
            "AutoTokenizer",
            SimpleNamespace(from_pretrained=load_tokenizer),
        )
        models = interp_lens_ladder.LadderLensModels(
            base_model="Qwen/Qwen3.5-9B",
            dtype=torch.bfloat16,
            jl=cast("Any", _StubJlensModule()),
            lens_model=LENS_MODEL_UNMERGED,
            merge_root=tmp_path / "merged",
            base_weights_identity="hf:base-commit",
            base_revision="base-commit",
        )

        models.target_for(LadderCell(BASE_ARM, BASE_STEP, None))

        assert load_calls == [("Qwen/Qwen3.5-9B", "base-commit")]
        assert tokenizer_calls == [("Qwen/Qwen3.5-9B", "base-commit")]

    def test_the_base_loads_once_and_every_cell_is_wrapped_after_its_attach(
        self, tmp_path: Path, stubs: dict[str, Any]
    ) -> None:
        jl = _StubJlensModule()
        models = self._models(jl, tmp_path)
        first = write_adapter_dir(tmp_path, "checkpoint-10", b"one")
        second = write_adapter_dir(tmp_path, "checkpoint-20", b"two")

        base_target = models.target_for(LadderCell(BASE_ARM, BASE_STEP, None))
        models.target_for(LadderCell("arm", 10, first))
        models.target_for(LadderCell("arm", 20, second))

        assert stubs["loads"] == ["Qwen/Qwen3.5-0.8B"], "one weight load for the whole ladder"
        assert [call.adapter_dir for call in stubs["attaches"]] == [first, second]
        # Sabotage-verified: wrapping `self._base` instead of what the attach returned turns the two
        # adapted entries below into the base object and this red.
        assert jl.wrapped[0] is stubs["base"], "the base cell fits on the un-adapted model"
        assert [str(entry) for entry in jl.wrapped[1:]] == [
            f"injected-over-{id(stubs['base'])}-checkpoint-10",
            f"injected-over-{id(stubs['base'])}-checkpoint-20",
        ], "every adapted cell is re-wrapped around the tree its attach injected into"
        assert base_target.load_path == JLENS_UNMERGED_LOAD_PATH
        assert base_target.merged_dir is None
        assert not (tmp_path / "merged").exists(), "the un-merged path writes no export"

    def test_the_second_adapted_cell_re_points_the_first_ones_peft_model(
        self, tmp_path: Path, stubs: dict[str, Any]
    ) -> None:
        """The first cell has nothing to re-point; every later one hands back the live PEFT wrapper.

        `attach_adapter` branches on `existing`: given one it swaps the adapter's weights inside the
        tree already injected, given None it calls `PeftModel.from_pretrained` and injects again. A
        second injection over an injected tree does not raise -- it fits a lens on a model with two
        stacked adapters and reports a normal number -- so passing `existing=None` on cell two is
        invisible in every other signal this file checks.
        """
        models = self._models(_StubJlensModule(), tmp_path)
        models.target_for(
            LadderCell("arm", 10, write_adapter_dir(tmp_path, "checkpoint-10", b"one"))
        )
        models.target_for(
            LadderCell("arm", 20, write_adapter_dir(tmp_path, "checkpoint-20", b"two"))
        )

        first, second = cast("list[_AttachCall]", stubs["attaches"])
        assert first.existing is None, "the first attach has no injected tree to re-point"
        assert second.existing is first.returned, "the second re-points what the first returned"

    def test_a_base_cell_after_an_attach_is_refused(
        self, tmp_path: Path, stubs: dict[str, Any]
    ) -> None:
        """PEFT injects in place, so this resident model can never serve the un-adapted base again.

        The agenda puts the base first, so this never fires in a normal run -- which is exactly why it
        is checked: the failure it prevents is a lens fitted on the last checkpoint's weights and
        published under the base's name, indistinguishable afterwards from a real base lens.
        """
        jl = _StubJlensModule()
        models = self._models(jl, tmp_path)
        models.target_for(
            LadderCell("arm", 10, write_adapter_dir(tmp_path, "checkpoint-10", b"one"))
        )
        with pytest.raises(RuntimeError, match="already attached an adapter"):
            models.target_for(LadderCell(BASE_ARM, BASE_STEP, None))


class TestDecodeDirection:
    def test_the_real_axis_and_its_placebo_are_decoded_the_same_way(self, tmp_path: Path) -> None:
        lens = StubLens()
        readout = decode_direction(
            lens,
            StubModel(),
            StubTokenizer(),
            direction_path=write_directions(tmp_path / "directions.pt"),
            layer=1,
            top_k=4,
            seed=0,
        )
        assert len(readout["real"]) == 4
        assert len(readout["placebo"]) == 4
        assert readout["layer"] == 1
        # Both arms transported at the same layer, at the same norm: the placebo is matched.
        assert [layer for layer, _ in lens.transported] == [1, 1]
        norms = [norm for _, norm in lens.transported]
        assert norms[0] == pytest.approx(norms[1], abs=1e-5)

    def test_the_placebo_reads_differently_from_the_real_axis(self, tmp_path: Path) -> None:
        readout = decode_direction(
            StubLens(),
            StubModel(),
            StubTokenizer(),
            direction_path=write_directions(tmp_path / "directions.pt"),
            layer=1,
            top_k=4,
            seed=0,
        )
        real = [entry["logit"] for entry in readout["real"]]
        placebo = [entry["logit"] for entry in readout["placebo"]]
        assert real != placebo

    def test_a_missing_layer_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="layer 9 not in"):
            decode_direction(
                StubLens(),
                StubModel(),
                StubTokenizer(),
                direction_path=write_directions(tmp_path / "directions.pt"),
                layer=9,
                top_k=4,
                seed=0,
            )

    def test_the_top_layer_has_no_fitted_jacobian(self, tmp_path: Path) -> None:
        """A lens fits source layers strictly below its target, so the top layer cannot transport."""
        with pytest.raises(ValueError, match="has no fitted Jacobian"):
            decode_direction(
                StubLens(),
                StubModel(),
                StubTokenizer(),
                direction_path=write_directions(tmp_path / "directions.pt"),
                layer=N_LAYERS - 1,
                top_k=4,
                seed=0,
            )


class TestCliRefusals:
    def test_a_direction_without_a_layer_is_refused_before_anything_loads(
        self, tmp_path: Path
    ) -> None:
        """Checked first, so it costs milliseconds rather than a model load and a fit."""
        args = build_parser().parse_args(
            [
                "--stimuli",
                str(tmp_path / "stimuli.jsonl"),
                "--out-dir",
                str(tmp_path / "out"),
                "--direction-path",
                str(tmp_path / "directions.pt"),
            ]
        )
        with pytest.raises(ValueError, match="needs --direction-layer"):
            run(args)

    @staticmethod
    def _args_into(out_dir: Path, *extra: str) -> Any:
        return build_parser().parse_args(
            ["--stimuli", str(out_dir.parent / "stimuli.jsonl"), "--out-dir", str(out_dir), *extra]
        )

    @staticmethod
    def _existing_report(out_dir: Path, **fields: object) -> None:
        out_dir.mkdir(parents=True)
        (out_dir / interp_lens_ladder.REPORT_FILENAME).write_text(
            json.dumps({"cells": {}, **fields})
        )

    def test_an_out_dir_from_the_other_lens_model_is_refused_before_anything_loads(
        self, tmp_path: Path
    ) -> None:
        """`--lens-model` picks the fit, not the layout: both arms write lens.pt and the report at the
        same paths, so the merged comparison arm run into an un-merged run's out-dir would overwrite
        every lens it was meant to be read against. jlens is not importable here, so a run that got
        past this check would die on that instead; the ValueError proves the refusal came first.

        Sabotage-verified: dropping the `refuse_other_lens_model_in_out_dir` call from `run` turns this
        into the jlens RuntimeError and the test red.
        """
        out_dir = tmp_path / "out"
        self._existing_report(out_dir, lens_model=LENS_MODEL_UNMERGED)
        with pytest.raises(
            ValueError, match="fitted with --lens-model unmerged, and this run asks"
        ):
            run(self._args_into(out_dir, "--lens-model", LENS_MODEL_MERGED))

    def test_a_report_from_before_the_flag_is_read_as_the_merged_arm(self, tmp_path: Path) -> None:
        """Merging was the only path there was, so a report without the field came off it."""
        out_dir = tmp_path / "out"
        self._existing_report(out_dir)
        with pytest.raises(ValueError, match="fitted with --lens-model merged"):
            run(self._args_into(out_dir))
        with pytest.raises(RuntimeError, match="jlens is not importable"):
            run(self._args_into(out_dir, "--lens-model", LENS_MODEL_MERGED))

    def test_the_same_lens_model_passes_the_check(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        self._existing_report(out_dir, lens_model=LENS_MODEL_UNMERGED)
        with pytest.raises(RuntimeError, match="jlens is not importable"):
            run(self._args_into(out_dir))

    def test_the_default_lens_model_is_the_un_merged_adapter(self) -> None:
        """C4's default: fit through `Wx + B(Ax)` rather than on a bf16 merge of about 64% of it.

        The merged path stays reachable as the comparison arm, which is what the `--lens-model` choice
        is: the same ladder over the other model, taking its own cache key.
        """
        args = build_parser().parse_args(["--stimuli", "s.jsonl", "--out-dir", "o", "--arm", "a=b"])
        assert args.lens_model == LENS_MODEL_UNMERGED
        assert (
            build_parser()
            .parse_args(
                ["--stimuli", "s.jsonl", "--out-dir", "o", "--lens-model", LENS_MODEL_MERGED]
            )
            .lens_model
            == LENS_MODEL_MERGED
        )

    def test_the_defaults_do_not_inherit_the_128_token_ceiling(self) -> None:
        args = build_parser().parse_args(["--stimuli", "s.jsonl", "--out-dir", "o", "--arm", "a=b"])
        assert args.max_seq_len_ceiling == DEFAULT_MAX_SEQ_LEN_CEILING
        assert args.max_seq_len_ceiling == 2048
        assert args.max_fit_prompts == 10
        assert cast("Any", args).direction_layer is None
        assert cast("Any", args).lens_cache is None

    def test_the_lens_cache_flag_takes_a_directory_or_an_s3_prefix(self) -> None:
        args = build_parser().parse_args(
            ["--stimuli", "s.jsonl", "--out-dir", "o", "--lens-cache", "s3://bucket/games_rl/lens"]
        )
        assert cast("Any", args).lens_cache == "s3://bucket/games_rl/lens"


TURN_PREFIX = "<|im_start|>user\n"
TURN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>"
SIDE_CONTINUATIONS = {
    "A": "so only the diagonal is reachable.",
    "B": "so the largest cell is free.",
}
# Wide enough that two matched sides do not collide into the same id sequence, which the ladder refuses
# as a corpus that lost its contrast.
TOKEN_ID_MODULUS = 97


class _LadderTokenizer:
    """A char-level tokenizer with a chat template: enough for the ladder's corpus arithmetic.

    The ladder derives its fit window, its pair-divergence indices and its corpus split from token
    counts, so a tokenizer that maps characters to ids reproduces every one of those decisions without
    a download. The template is only used to locate where a user turn's content begins, which is what
    the verbatim render convention checks each stimulus against.
    """

    def get_chat_template(self) -> str:
        """A template with no `reasoning_effort` knob, so nothing is pinned for it."""
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
        """Render one user turn the way this family's template does, opening `<think>`."""
        del tokenize, add_generation_prompt, kwargs
        suffix = TURN_SUFFIX if enable_thinking else TURN_SUFFIX.removesuffix("<think>")
        return f"{TURN_PREFIX}{messages[0]['content']}{suffix}"

    def __call__(self, text: str | list[str], *, add_special_tokens: bool = True) -> dict[str, Any]:
        """Tokenize one string or a batch; this tokenizer adds no specials either way."""
        del add_special_tokens
        if isinstance(text, str):
            return {"input_ids": [ord(char) % TOKEN_ID_MODULUS for char in text]}
        return {"input_ids": [[ord(char) % TOKEN_ID_MODULUS for char in item] for item in text]}


def write_stimuli_file(path: Path) -> Path:
    """One matched pair, already rendered, as `--stimulus-render verbatim` expects to be handed it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": f"twin-set--p0--{side}",
            "set": "twin-set",
            "side": side,
            "pair_id": "twin-set--p0",
            "text": f"{TURN_PREFIX}a change record both sides share{TURN_SUFFIX}{continuation}",
        }
        for side, continuation in sorted(SIDE_CONTINUATIONS.items())
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


class _StubAcquisition:
    """What `acquire_lens` hands back: a lens plus how it was obtained."""

    def __init__(self) -> None:
        self.lens = StubLens()
        self.source = "fit"

    def as_payload(self) -> dict[str, object]:
        """The report block, as the real `LensAcquisition` writes it."""
        return {"source": self.source, "seconds": 0.0, "cache_root": None}


PREFILL_BINDING = {
    "chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk_gated_delta_rule",
    "causal_conv1d_fn": "fla.modules.convolution.causal_conv1d_fn",
}
DECODE_BINDING = {
    "recurrent_gated_delta_rule": "fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule",
    "causal_conv1d_update": "fla.modules.convolution.causal_conv1d_update",
}


@pytest.fixture
def run_stubs(monkeypatch: pytest.MonkeyPatch, stubs: dict[str, Any]) -> dict[str, Any]:
    """Everything `run` reaches that needs a card, a download or jlens, recording the call order.

    Layered on the `stubs` fixture, which already covers the resident base load and the adapter attach;
    this adds the jlens requirement check, the tokenizer, the weights-identity resolver, the git sha,
    the kernel bridge and binding, the lens acquisition and the reconstruction read. The single `order`
    list is what makes "the binding is read before the first fit" a checkable claim rather than a
    reading of the source.
    """
    order: list[str] = []
    jl = _StubJlensModule()
    acquisitions: list[_StubAcquisition] = []

    def bound_kernels() -> dict[str, str]:
        order.append("bound_deltanet_kernels")
        return {**PREFILL_BINDING, **DECODE_BINDING}

    def acquire(*args: object, **kwargs: object) -> object:
        del args
        order.append("acquire_lens")
        acquisition = _StubAcquisition()
        acquisitions.append(acquisition)
        cast("Path", kwargs["lens_path"]).write_bytes(b"lens")
        return acquisition

    monkeypatch.setattr(interp_lens_ladder, "_require_jlens", lambda: jl)
    monkeypatch.setattr(
        interp_lens_ladder,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *a, **k: _LadderTokenizer()),
    )
    monkeypatch.setattr(interp_lens_ladder, "resolve_weights_identity", lambda *a, **k: "hf:abc123")
    monkeypatch.setattr(interp_lens_ladder, "git_sha", lambda: "testingsha")
    monkeypatch.setattr(
        interp_lens_ladder,
        "bridge_and_check_decode_kernel",
        lambda: {"bridged": True, "reason": "aliased fla's fused per-token kernel"},
    )
    monkeypatch.setattr(interp_lens_ladder, "bound_deltanet_kernels", bound_kernels)
    monkeypatch.setattr(interp_lens_ladder, "acquire_lens", acquire)
    monkeypatch.setattr(interp_lens_ladder, "evaluate_reconstruction", lambda *a, **k: None)
    return {**stubs, "jl": jl, "order": order, "acquisitions": acquisitions}


class TestLadderRunEndToEnd:
    """Drive `run` over a base cell plus an adapted one, with no card, no weights and no jlens.

    The three other `run` tests all die at a refusal or at "jlens is not importable" before a single
    cell is fitted, so everything between the corpus plan and the report on disk was covered by the L4
    smoke alone. What is pinned here is the per-cell wiring: which kernel binding each cell's entry
    carries and when it was read, that the report reaches disk after every cell rather than only at the
    end, and that a merged-arm base cell does not label itself the other arm.
    """

    @staticmethod
    def _args(tmp_path: Path, *extra: str) -> Any:
        arm_dir = tmp_path / "arm-run"
        arm_dir.mkdir(parents=True, exist_ok=True)
        write_adapter_dir(arm_dir, "checkpoint-10", b"ten")
        return build_parser().parse_args(
            [
                "--stimuli",
                str(write_stimuli_file(tmp_path / "stimuli.jsonl")),
                "--out-dir",
                str(tmp_path / "out"),
                "--arm",
                f"trained={arm_dir}",
                "--stimulus-render",
                "verbatim",
                "--max-fit-prompts",
                "1",
                "--n-eval-prompts",
                "1",
                *extra,
            ]
        )

    def test_every_cell_carries_the_prefill_binding_read_once_before_the_first_fit(
        self, tmp_path: Path, run_stubs: dict[str, Any]
    ) -> None:
        """A fit is forward and backward with no cache, so the decode pair never enters a cell's name.

        The call count is the load-bearing half: reading the binding per cell inside the report-writing
        `finally` made an upstream dispatch change surface only after the first fit had already been
        paid for, and skipped the report write for the cell that had just succeeded.
        """
        report = run(self._args(tmp_path))

        assert list(report["cells"]) == ["base/step-0", "trained/step-10"]
        for entry in report["cells"].values():
            assert entry["deltanet_kernel"] == PREFILL_BINDING
        assert report["deltanet_kernels_bound"] == {**PREFILL_BINDING, **DECODE_BINDING}
        assert run_stubs["order"].count("bound_deltanet_kernels") == 1, "read once, then reused"
        assert run_stubs["order"][0] == "bound_deltanet_kernels", "before the first fit"

    def test_a_binding_that_cannot_be_read_costs_no_fit(
        self, tmp_path: Path, run_stubs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`bound_deltanet_kernels` raises when transformers restructures its DeltaNet dispatch.

        Read after the fit, that raise arrives hours late AND lands in the `finally` that writes the
        report, so the cell that just succeeded is lost as well.
        """

        def refuse() -> dict[str, str]:
            raise RuntimeError("wrapper layout changed")

        monkeypatch.setattr(interp_lens_ladder, "bound_deltanet_kernels", refuse)
        with pytest.raises(RuntimeError, match="wrapper layout changed"):
            run(self._args(tmp_path))
        assert run_stubs["order"] == [], "no lens was acquired before the binding was readable"

    def test_the_report_is_on_disk_after_every_cell_rather_than_only_at_the_end(
        self, tmp_path: Path, run_stubs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ladder is hours per cell, so a crash on cell two must not cost cell one's entry."""
        written: list[list[str]] = []
        report_path = tmp_path / "out" / interp_lens_ladder.REPORT_FILENAME
        stubbed_acquire = cast("Any", interp_lens_ladder.acquire_lens)

        def acquire_and_snapshot(*args: object, **kwargs: object) -> object:
            acquired = stubbed_acquire(*args, **kwargs)
            found = json.loads(report_path.read_text())["cells"] if report_path.is_file() else {}
            written.append(sorted(found))
            return acquired

        monkeypatch.setattr(interp_lens_ladder, "acquire_lens", acquire_and_snapshot)
        report = run(self._args(tmp_path))

        assert written == [[], ["base/step-0"]], (
            "cell one's entry is durable before cell two starts"
        )
        assert sorted(json.loads(report_path.read_text())["cells"]) == sorted(report["cells"])
        assert report["requested_cells"] == list(report["cells"])

    def test_a_merged_arm_base_cell_is_not_labelled_the_un_merged_one(
        self, tmp_path: Path, run_stubs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The base cell has no export to name, and used to read `unmerged` inside a merged run.

        Its `merged_dir` is None because a hub id is already a plain model, so a per-cell arm derived
        from that field disagreed with the run-level field and with the lens cache key -- on the one
        cell every drift read is measured against.
        """
        exported: list[Path] = []

        def export(adapter_dir: Path, base_model_id: str, out_dir: Path, **kwargs: object) -> Path:
            del adapter_dir, base_model_id, kwargs
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "config.json").write_text("{}")
            exported.append(out_dir)
            return out_dir

        monkeypatch.setattr(interp_lens_ladder, "export_merged_checkpoint", export)
        monkeypatch.setattr(
            interp_lens_ladder,
            "_load_jlens_model",
            lambda config, jl: (f"merged-{config.model_id}", _LadderTokenizer()),
        )
        report = run(self._args(tmp_path, "--lens-model", LENS_MODEL_MERGED))

        assert report["lens_model"] == LENS_MODEL_MERGED
        for label, entry in report["cells"].items():
            assert entry["lens_model"] == LENS_MODEL_MERGED, label
        assert report["cells"]["base/step-0"]["merged"] is False, "the base cell exports nothing"
        assert len(exported) == 1, "only the adapted cell merges"
        assert not exported[0].exists(), "the export is deleted once its lens is fitted"


class TestLensCacheKeyForCell:
    """The ladder's half of the cache key (hot-path backlog rank 17): what a cell's fit depends on."""

    def _key(
        self, cell: LadderCell, fit_prompts: list[str], *, load_path: str = JLENS_LOAD_PATH
    ) -> Any:
        return lens_cache_key_for_cell(
            cell,
            base_model="Qwen/Qwen3.5-2B",
            base_weights_identity="hf:abc",
            merge_dtype=torch.bfloat16,
            fit_prompts=fit_prompts,
            max_seq_len=452,
            skip_first=16,
            dim_batch=16,
            load_path=load_path,
        )

    def test_the_merged_and_un_merged_fits_take_different_keys(self, tmp_path: Path) -> None:
        """C4: they are different lenses, so serving one under the other's key would publish a lie.

        Probe I6 measured the gap at 0.6% median and 1.8% max relative Frobenius per layer, against a
        0.33% floor for two fits of the SAME weights through different wrapper classes -- the merge is
        visible in the lens. The base cell splits too, because the two loaders build different classes.
        """
        adapter_dir = write_adapter_dir(tmp_path, "checkpoint-70", b"weights")
        adapted = LadderCell(arm="group", step=70, adapter_dir=adapter_dir)
        base = LadderCell(arm=BASE_ARM, step=BASE_STEP, adapter_dir=None)
        merged = self._key(adapted, ["p1"], load_path=JLENS_LOAD_PATH)
        unmerged = self._key(adapted, ["p1"], load_path=JLENS_UNMERGED_LOAD_PATH)

        assert merged.sha256 != unmerged.sha256
        assert merged.merge_dtype == "bfloat16"
        assert unmerged.merge_dtype is None, "nothing is merged on the un-merged path"
        assert unmerged.adapter_weights_sha256 == merged.adapter_weights_sha256
        assert (
            self._key(base, ["p1"], load_path=JLENS_LOAD_PATH).sha256
            != self._key(base, ["p1"], load_path=JLENS_UNMERGED_LOAD_PATH).sha256
        )

    def test_the_base_cell_carries_no_adapter_and_no_merge(self) -> None:
        key = self._key(LadderCell(arm=BASE_ARM, step=BASE_STEP, adapter_dir=None), ["p1", "p2"])
        assert key.adapter_weights_sha256 is None
        assert key.adapter_config_sha256 is None
        assert key.merge_dtype is None
        assert key.load_path == JLENS_LOAD_PATH
        assert key.jlens_commit == JLENS_COMMIT
        assert key.fit_prompts_sha256 == digest_strings(["p1", "p2"])
        assert key.n_fit_prompts == 2

    def test_an_adapted_cell_digests_its_weights_and_config_and_names_the_merge_dtype(
        self, tmp_path: Path
    ) -> None:
        adapter_dir = write_adapter_dir(tmp_path, "checkpoint-70", b"weights")
        key = self._key(LadderCell(arm="group", step=70, adapter_dir=adapter_dir), ["p1"])
        assert key.adapter_weights_sha256 == sha256_of_file(
            adapter_dir / "adapter_model.safetensors"
        )
        assert key.adapter_config_sha256 == sha256_of_file(adapter_dir / "adapter_config.json")
        assert key.merge_dtype == "bfloat16"

    def test_a_rewritten_adapter_or_a_reordered_corpus_changes_the_key(
        self, tmp_path: Path
    ) -> None:
        first = write_adapter_dir(tmp_path, "a", b"weights")
        second = write_adapter_dir(tmp_path, "b", b"weightz")
        base_key = self._key(LadderCell(arm="g", step=70, adapter_dir=first), ["p1", "p2"])
        assert self._key(LadderCell(arm="g", step=70, adapter_dir=second), ["p1", "p2"]).sha256 != (
            base_key.sha256
        )
        assert self._key(LadderCell(arm="g", step=70, adapter_dir=first), ["p2", "p1"]).sha256 != (
            base_key.sha256
        )
        assert self._key(
            LadderCell(arm="other", step=1, adapter_dir=first), ["p1", "p2"]
        ).sha256 == (base_key.sha256), "the arm label and step are names, not inputs to the fit"
