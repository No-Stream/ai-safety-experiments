"""Offline tests for the Jacobian-lens adapter's non-GPU logic.

`jlens` is not installed (it runs via PYTHONPATH) and the Qwen3.5 CPU route is closed, so nothing
here loads a lens or a model. What is checked is exactly the logic that can be wrong without a GPU:

* config validation rejects bad sources and non-positive knobs before a long model load;
* ``cap_fit_prompts`` caps the list -- the ONLY bound on a fit, since the reference ``jlens.fit``
  has no auto-stop -- and passes a short list through unchanged;
* ``decode_topk`` returns the highest-logit tokens in order, decoded via a supplied id->token map;
* ``transport_and_decode`` wires transport -> unembed -> top-k correctly against stub objects;
* ``fit_lens`` hands the config's fit knobs to ``jlens.fit`` (validating them and then dropping
  them silently ran the fit on the reference defaults instead);
* the CLI refuses the top layer before loading a model, since the lens fits source layers strictly
  below its target and has no Jacobian for that one;
* ``_require_jlens`` raises with PYTHONPATH guidance when the package is absent (it is);
* the fit-quality reconstruction metric (``layer_reconstruction`` / ``reconstruction_report``) reads
  ~0 residual and ~1.0 explained variance when the lens reproduces the model's logits exactly, and a
  larger residual under injected noise, so it measures fit quality rather than printing a constant;
* ``interior_eval_positions`` picks the fit-consistent interior read positions and returns ``[]``
  rather than indexing out of range on a prompt too short to have any.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from reward_hacking.interp import jacobian
from reward_hacking.interp.jacobian import (
    DEFAULT_MAX_SEQ_LEN_CEILING,
    JLENS_LOAD_PATH,
    JLENS_UNMERGED_LOAD_PATH,
    LENS_CACHE_KEY_FILENAME,
    LENS_CACHE_LENS_FILENAME,
    JacobianConfig,
    LensCache,
    LensCacheKey,
    S3CommandError,
    TokenReadout,
    _require_jlens,  # pyright: ignore[reportPrivateUsage]
    _run_s3,  # pyright: ignore[reportPrivateUsage]  # the default runner's timeout conversion
    acquire_lens,
    cap_fit_prompts,
    decode_topk,
    derive_max_seq_len,
    digest_strings,
    fit_lens,
    fit_skip_first,
    interior_eval_positions,
    layer_reconstruction,
    local_weights_digest,
    reconstruction_report,
    resolve_weights_identity,
    transport_and_decode,
    verify_cached_lens,
    verify_lens_roundtrip,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class TestJacobianConfig:
    def test_defaults_to_fit_own(self) -> None:
        assert JacobianConfig().source == "fit_own"

    def test_rejects_unknown_source(self) -> None:
        with pytest.raises(ValueError, match="unknown lens source"):
            JacobianConfig(source="wikitext")  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize(
        ("field", "value"),
        [("max_fit_prompts", 0), ("dim_batch", -1), ("max_seq_len", 0), ("top_k", 0)],
    )
    def test_rejects_non_positive_knobs(self, field: str, value: int) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            JacobianConfig(**{field: value})  # pyright: ignore[reportArgumentType]  # parametrized field


class TestCapFitPrompts:
    def test_caps_a_long_list(self) -> None:
        prompts = [f"p{i}" for i in range(300)]
        capped = cap_fit_prompts(prompts, JacobianConfig(max_fit_prompts=200))
        assert capped == prompts[:200]

    def test_passes_a_short_list_through(self) -> None:
        prompts = [f"p{i}" for i in range(10)]  # ~10 prompts: the fit-path smoke size
        assert cap_fit_prompts(prompts, JacobianConfig(max_fit_prompts=200)) == prompts


class TestDecodeTopk:
    def test_returns_highest_logit_tokens_in_order(self) -> None:
        logits = torch.tensor([0.1, 5.0, -2.0, 3.0])
        readouts = decode_topk(logits, lambda i: f"tok{i}", k=2)

        assert readouts == [TokenReadout("tok1", 5.0), TokenReadout("tok3", 3.0)]

    def test_clamps_k_to_vocab(self) -> None:
        logits = torch.tensor([1.0, 2.0])
        assert len(decode_topk(logits, str, k=10)) == 2

    def test_rejects_non_1d_readout(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            decode_topk(torch.zeros(2, 3), str, k=1)


class _StubLens:
    """A lens whose transport doubles a direction -- enough to check the wiring, not the math."""

    def transport(self, direction: torch.Tensor, layer: int) -> torch.Tensor:
        return direction * float(layer + 1)


class _StubModel:
    """An unembedding: a fixed [d, vocab] projection so a transported vector maps to
    vocab logits."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight

    def unembed(self, transported: torch.Tensor) -> torch.Tensor:
        return transported @ self.weight


class TestTransportAndDecode:
    def test_wires_transport_unembed_and_topk(self) -> None:
        d, vocab = 3, 5
        # A one-hot-ish unembed so the argmax token is predictable from the transported vector.
        weight = torch.eye(d, vocab)  # columns 0..2 read off dims 0..2; 3,4 are zero
        direction = torch.tensor([0.0, 9.0, 1.0])

        readouts = transport_and_decode(
            _StubLens(),
            _StubModel(weight),
            direction,
            layer=0,  # transport scales by (layer+1)=1
            id_to_token=lambda i: f"v{i}",
            k=2,
        )

        # unembed(transport(dir, 0)) = [0, 9, 1, 0, 0]; top-2 are dim 1 then dim 2.
        assert [r.token for r in readouts] == ["v1", "v2"]
        assert readouts[0].logit == pytest.approx(9.0)

    def test_layer_scaling_flows_through_transport(self) -> None:
        weight = torch.eye(2, 2)
        direction = torch.tensor([1.0, 0.0])

        at_layer0 = transport_and_decode(
            _StubLens(), _StubModel(weight), direction, 0, id_to_token=str, k=1
        )
        at_layer4 = transport_and_decode(
            _StubLens(), _StubModel(weight), direction, 4, id_to_token=str, k=1
        )

        # transport scales by (layer+1), so the layer-4 readout logit is 5x the layer-0 one.
        assert at_layer4[0].logit == pytest.approx(5.0 * at_layer0[0].logit)


class _RecordingJlens:
    """Stands in for the reference ``jlens`` module, recording what ``fit`` was handed.

    The knobs matter because they set the fit's schedule, not its estimate: ``dim_batch`` fixes how
    many backward passes each prompt costs and how much activation memory is live, ``max_seq_len``
    truncates each prompt. A config that asks for one schedule and silently runs another wastes
    hours of fit time and reports the wrong cost.
    """

    def __init__(self) -> None:
        self.fit_kwargs: dict[str, object] = {}
        self.fit_prompts: list[str] = []

    def fit(self, model: object, prompts: list[str], **kwargs: object) -> _StubLens:
        del model
        self.fit_prompts = list(prompts)
        self.fit_kwargs = dict(kwargs)
        return _StubLens()


class TestFitLens:
    def test_passes_the_configs_fit_knobs_to_jlens_fit(self) -> None:
        jl = _RecordingJlens()
        config = JacobianConfig(dim_batch=4, max_seq_len=64, max_fit_prompts=10)

        lens = fit_lens(config, object(), ["first prompt", "second prompt"], jl)  # pyright: ignore[reportArgumentType]  # duck-typed stand-in for the jlens module

        assert isinstance(lens, _StubLens)
        assert jl.fit_prompts == ["first prompt", "second prompt"]
        assert jl.fit_kwargs == {
            "dim_batch": 4,
            "max_seq_len": 64,
            "checkpoint_path": None,
            "checkpoint_every": 1,
            "resume": True,
        }

    def test_still_caps_the_prompt_list(self) -> None:
        jl = _RecordingJlens()
        fit_lens(JacobianConfig(max_fit_prompts=2), object(), ["a", "b", "c"], jl)  # pyright: ignore[reportArgumentType]  # duck-typed stand-in for the jlens module
        assert jl.fit_prompts == ["a", "b"]


class TestCliRejectsTheTopLayer:
    """``--layer 31`` is the natural choice for decoding a saved axis and has no fitted Jacobian.

    ``jlens`` fits source layers strictly below its target layer (which defaults to the final one),
    so ``transport`` on the top layer is a bare dict index that raises ``KeyError`` -- after the
    model load and the hours-long fit. The CLI's existing check only asks whether the layer is in
    the saved directions, where the top layer always is.
    """

    def test_top_layer_raises_before_any_model_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directions_path = tmp_path / "directions.pt"
        torch.save({layer: torch.zeros(4) for layer in range(3)}, directions_path)

        def _fail_if_called(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("the model load ran before --layer was validated")

        monkeypatch.setattr(jacobian, "load_model_and_lens", _fail_if_called)

        with pytest.raises(ValueError, match="no fitted Jacobian"):
            jacobian.main(["--direction-path", str(directions_path), "--layer", "2"])

    def test_a_layer_below_the_top_passes_the_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard must not swallow the layers that DO have a Jacobian."""
        directions_path = tmp_path / "directions.pt"
        torch.save({layer: torch.zeros(4) for layer in range(3)}, directions_path)

        def _stop_after_the_check(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError("reached the model load")

        monkeypatch.setattr(jacobian, "load_model_and_lens", _stop_after_the_check)

        with pytest.raises(RuntimeError, match="reached the model load"):
            jacobian.main(["--direction-path", str(directions_path), "--layer", "1"])


class TestRequireJlens:
    def test_raises_with_pythonpath_guidance_when_absent(self) -> None:
        with pytest.raises(RuntimeError, match="PYTHONPATH"):
            _require_jlens()


class TestLayerReconstruction:
    """The per-layer fit-quality metric: 0 residual / 1.0 EV when exact, worse under noise.

    This is the DEFECT-E guard. Before it, ``fit_report.json`` carried counts and config only, so a
    lens that reconstructed nothing was indistinguishable from a good one. The metric must actually
    MOVE with reconstruction error, which is what the perfect-vs-noisy contrast below asserts -- a
    metric that returned a constant would pass a single-case check and fail here.
    """

    def test_perfect_reconstruction_is_zero_residual_unit_variance(self) -> None:
        generator = torch.Generator().manual_seed(0)
        actual = torch.randn(8, 32, generator=generator)
        read = layer_reconstruction(5, actual.clone(), actual)
        assert read.layer == 5
        assert read.n_samples == 8
        assert read.relative_residual == pytest.approx(0.0, abs=1e-6)
        assert read.explained_variance == pytest.approx(1.0, abs=1e-6)

    def test_injected_noise_raises_residual_and_lowers_explained_variance(self) -> None:
        generator = torch.Generator().manual_seed(1)
        actual = torch.randn(8, 32, generator=generator)
        noisy = actual + 0.5 * torch.randn(actual.shape, generator=generator)
        read = layer_reconstruction(3, noisy, actual)
        assert read.relative_residual > 0.05
        assert read.explained_variance < 0.99

    def test_more_noise_is_monotonically_worse(self) -> None:
        """The metric tracks fit quality: heavier noise must read as a strictly larger residual."""
        generator = torch.Generator().manual_seed(2)
        actual = torch.randn(16, 64, generator=generator)
        light = layer_reconstruction(
            0, actual + 0.1 * torch.randn(actual.shape, generator=generator), actual
        )
        heavy = layer_reconstruction(
            0, actual + 1.0 * torch.randn(actual.shape, generator=generator), actual
        )
        assert light.relative_residual < heavy.relative_residual
        assert light.explained_variance > heavy.explained_variance

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            layer_reconstruction(0, torch.zeros(4, 8), torch.zeros(4, 9))

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match=r"\[n_samples, vocab\]"):
            layer_reconstruction(0, torch.zeros(4), torch.zeros(4))

    def test_rejects_zero_norm_target(self) -> None:
        with pytest.raises(ValueError, match="zero norm"):
            layer_reconstruction(0, torch.zeros(4, 8), torch.zeros(4, 8))

    def test_rejects_constant_target(self) -> None:
        with pytest.raises(ValueError, match="constant"):
            layer_reconstruction(0, torch.ones(4, 8), torch.full((4, 8), 3.0))


class TestReconstructionReport:
    """Aggregation over source layers: mean/median residual, and the best (min-residual) layer."""

    def test_aggregates_and_names_the_best_layer(self) -> None:
        generator = torch.Generator().manual_seed(3)
        actual = torch.randn(10, 48, generator=generator)
        predicted = {
            2: actual + 1.0 * torch.randn(actual.shape, generator=generator),  # worst
            5: actual + 0.1 * torch.randn(actual.shape, generator=generator),  # best
            9: actual + 0.4 * torch.randn(actual.shape, generator=generator),
        }
        report = reconstruction_report(predicted, actual)
        assert [read.layer for read in report.per_layer] == [2, 5, 9]
        assert report.n_samples == 10
        assert report.best_layer == 5
        residuals = [read.relative_residual for read in report.per_layer]
        assert report.best_layer_relative_residual == pytest.approx(min(residuals))
        assert report.mean_relative_residual == pytest.approx(sum(residuals) / len(residuals))
        assert min(residuals) <= report.median_relative_residual <= max(residuals)

    def test_perfect_layers_report_unit_explained_variance(self) -> None:
        generator = torch.Generator().manual_seed(4)
        actual = torch.randn(6, 20, generator=generator)
        report = reconstruction_report({4: actual.clone(), 7: actual.clone()}, actual)
        assert report.mean_relative_residual == pytest.approx(0.0, abs=1e-6)
        assert report.mean_explained_variance == pytest.approx(1.0, abs=1e-6)

    def test_rejects_an_empty_layer_map(self) -> None:
        with pytest.raises(ValueError, match="no per-layer"):
            reconstruction_report({}, torch.randn(4, 8))


class TestDeriveMaxSeqLen:
    """The fit window comes from the corpus, never from ``jlens.fit``'s 128-token default.

    Both lens ladders derive through this one function (hot-path backlog rank 59 / decision C8): the
    games stimulus corpus runs 368-451 tokens with its two sides first differing at index 326-415, and
    the reward-hacking twin transcripts run 1.3k-22.3k, so at 128 either fit averages Jacobians over a
    shared prefix and never sees the contrast -- a fit that costs hours and measures the wrong thing
    while every log line stays green.

    Sabotage-verified: returning the ceiling unconditionally (``chosen = ceiling``) turns the
    covered-whole case red, and dropping the truncation count turns the ceiling case red.
    """

    def test_a_corpus_that_fits_is_covered_whole(self) -> None:
        plan = derive_max_seq_len([100, 200, 300], ceiling=2048)
        assert plan.max_seq_len == 300
        assert plan.n_truncated == 0
        assert plan.fraction_truncated == 0.0
        assert plan.corpus_median_tokens == 200
        assert plan.corpus_max_tokens == 300

    def test_a_ceiling_reports_what_it_truncates(self) -> None:
        """The number that has to be visible: how much of the corpus the fit will not see."""
        plan = derive_max_seq_len([100, 200, 3000, 4000], ceiling=2048)
        assert plan.max_seq_len == 2048
        assert plan.n_truncated == 2
        assert plan.fraction_truncated == pytest.approx(0.5)
        assert plan.corpus_max_tokens == 4000

    def test_the_library_default_would_truncate_both_corpora(self) -> None:
        assert derive_max_seq_len([320, 380, 420], ceiling=128).n_truncated == 3
        assert derive_max_seq_len([1300, 2300, 22300], ceiling=128).n_truncated == 3

    def test_the_shared_ceiling_is_the_default(self) -> None:
        assert derive_max_seq_len([9999]).max_seq_len == DEFAULT_MAX_SEQ_LEN_CEILING

    def test_the_payload_carries_every_term_of_the_plan(self) -> None:
        payload = derive_max_seq_len([10, 20], ceiling=15).as_payload()
        assert payload == {
            "max_seq_len": 15,
            "corpus_max_tokens": 20,
            "corpus_median_tokens": 15,
            "n_truncated": 1,
            "fraction_truncated": 0.5,
        }

    def test_an_empty_corpus_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no corpus token lengths"):
            derive_max_seq_len([], ceiling=2048)


class TestUnmergedLoadPath:
    """The un-merged fit takes its own cache-key name, because it is its own lens (decision C4).

    Probe I6 fitted both on 0.8B: two fits of the same un-adapted weights through different wrapper
    classes differ by at most 0.33% relative Frobenius per layer, a PEFT model with every ``lora_B``
    zeroed sits on that floor, and the merged-and-reloaded fit differs from the un-merged one by 0.6%
    median and 1.8% max -- five to eight times the floor. So the merge is visible in the lens, and a
    cache that could not tell the two apart would serve one for the other.
    """

    def test_the_two_load_paths_are_distinct_strings(self) -> None:
        assert JLENS_UNMERGED_LOAD_PATH != JLENS_LOAD_PATH

    def test_the_un_merged_path_names_what_it_actually_does(self) -> None:
        assert "get_base_model" in JLENS_UNMERGED_LOAD_PATH
        assert "load_adapter_base" in JLENS_UNMERGED_LOAD_PATH


class TestInteriorEvalPositions:
    """Interior read positions for the reconstruction eval: fit-consistent, and short-prompt safe."""

    def test_returns_evenly_spaced_interior_positions(self) -> None:
        positions = interior_eval_positions(100, skip_first=16, max_positions=8)
        assert len(positions) == 8
        assert positions[0] == 16  # first valid interior position (skip_first)
        assert positions[-1] == 98  # last, excluding the final no-next-token position (seq_len - 2)
        assert positions == sorted(positions)
        assert len(set(positions)) == 8

    def test_takes_every_interior_position_when_below_the_cap(self) -> None:
        # valid interior is range(16, 21) = 16..20 -> five positions, all kept under a cap of 8.
        assert interior_eval_positions(22, skip_first=16, max_positions=8) == [16, 17, 18, 19, 20]

    def test_too_short_prompt_yields_no_positions(self) -> None:
        assert interior_eval_positions(10, skip_first=16, max_positions=8) == []
        # seq_len == skip_first + 1: the only candidate is the excluded final position.
        assert interior_eval_positions(17, skip_first=16, max_positions=8) == []

    def test_a_single_valid_interior_position(self) -> None:
        assert interior_eval_positions(18, skip_first=16, max_positions=8) == [16]

    def test_rejects_negative_skip_first(self) -> None:
        with pytest.raises(ValueError, match="skip_first"):
            interior_eval_positions(100, skip_first=-1, max_positions=8)

    def test_rejects_non_positive_max_positions(self) -> None:
        with pytest.raises(ValueError, match="max_positions"):
            interior_eval_positions(100, skip_first=16, max_positions=0)


# --------------------------------------------------------------------------------------
# The lens cache (hot-path backlog rank 17)
# --------------------------------------------------------------------------------------

D_MODEL = 4
N_LAYERS = 3


class _FileLens:
    """A lens with the reference's on-disk contract: ``jacobians``, ``save``, ``load``."""

    def __init__(self, jacobians: dict[int, torch.Tensor], *, d_model: int) -> None:
        self.jacobians = {layer: J.float() for layer, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.d_model = d_model
        self.n_prompts = 1

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        torch.save(
            {
                "J": {layer: J.to(dtype) for layer, J in self.jacobians.items()},
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> _FileLens:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls(checkpoint["J"], d_model=checkpoint["d_model"])

    def transport(self, direction: torch.Tensor, layer: int) -> torch.Tensor:
        return direction @ self.jacobians[layer].T


class _CachingJlens:
    """Stands in for the jlens module: a counted ``fit`` whose lens depends on ``fill``."""

    def __init__(self, fill: float = 1.0) -> None:
        self.fill = fill
        self.fit_calls = 0
        self.JacobianLens = _FileLens
        self.fitting = SimpleNamespace(SKIP_FIRST_N_POSITIONS=16)

    def fit(self, model: object, prompts: Sequence[str], **kwargs: object) -> _FileLens:
        del model, prompts, kwargs
        self.fit_calls += 1
        return _FileLens(
            {
                layer: torch.full((D_MODEL, D_MODEL), self.fill + layer)
                for layer in range(N_LAYERS - 1)
            },
            d_model=D_MODEL,
        )


_LENS_MODEL = SimpleNamespace(n_layers=N_LAYERS, d_model=D_MODEL)


def _key(**overrides: Any) -> LensCacheKey:
    base: dict[str, Any] = {
        "base_model": "Qwen/Qwen3.5-2B",
        "base_weights_identity": "hf:abc123",
        "adapter_weights_sha256": None,
        "adapter_config_sha256": None,
        "merge_dtype": None,
        "load_path": "AutoModelForCausalLM.from_pretrained(dtype=bfloat16).to(cuda)",
        "fit_prompts_sha256": digest_strings(["one prompt", "another"]),
        "n_fit_prompts": 2,
        "max_seq_len": 452,
        "skip_first": 16,
        "dim_batch": 16,
        "jlens_commit": "581d398",
    }
    base.update(overrides)
    return LensCacheKey(**base)


def _changed(value: object) -> object:
    if value is None:
        return "was-none"
    if isinstance(value, int):
        return value + 1
    return f"{value}-changed"


class TestLensCacheKey:
    """The key is the cache's only guard, so every field must move the digest."""

    @pytest.mark.parametrize("field_name", [f.name for f in fields(LensCacheKey)])
    def test_changing_any_one_field_changes_the_digest(self, field_name: str) -> None:
        key = _key()
        other = replace(key, **{field_name: _changed(getattr(key, field_name))})
        assert other.sha256 != key.sha256, f"{field_name} did not reach the digest"

    def test_the_digest_is_a_function_of_the_payload_alone(self) -> None:
        assert _key().sha256 == _key().sha256
        assert LensCacheKey(**cast("dict[str, Any]", _key().as_payload())).sha256 == _key().sha256

    def test_fit_prompt_digest_is_order_and_boundary_sensitive(self) -> None:
        assert digest_strings(["ab", "c"]) != digest_strings(["a", "bc"])
        assert digest_strings(["a", "b"]) != digest_strings(["b", "a"])

    def test_skip_first_is_read_off_the_jlens_module(self) -> None:
        assert fit_skip_first(cast("Any", _CachingJlens())) == 16


class TestLensCacheLocal:
    def test_store_then_lookup_returns_the_same_bytes_and_the_key(self, tmp_path: Path) -> None:
        cache = LensCache(root=str(tmp_path / "cache"))
        key = _key()
        lens_path = tmp_path / "fit" / LENS_CACHE_LENS_FILENAME
        lens_path.parent.mkdir()
        _CachingJlens().fit(object(), []).save(str(lens_path))

        cache.store(key, lens_path)
        dest = tmp_path / "reload" / "lens.pt"
        assert cache.lookup(key, dest) is True
        assert dest.read_bytes() == lens_path.read_bytes()
        assert json.loads((dest.parent / LENS_CACHE_KEY_FILENAME).read_text()) == key.as_payload()

    def test_a_key_never_stored_is_a_miss(self, tmp_path: Path) -> None:
        cache = LensCache(root=str(tmp_path / "cache"))
        assert cache.lookup(_key(), tmp_path / "lens.pt") is False
        assert not (tmp_path / "lens.pt").exists()

    def test_an_entry_whose_sidecar_does_not_match_its_key_is_refused(self, tmp_path: Path) -> None:
        """Sabotage kept as a test: a hand-edited sidecar under the right digest must not serve."""
        cache = LensCache(root=str(tmp_path / "cache"))
        key = _key()
        lens_path = tmp_path / LENS_CACHE_LENS_FILENAME
        _CachingJlens().fit(object(), []).save(str(lens_path))
        cache.store(key, lens_path)
        sidecar = tmp_path / "cache" / key.sha256 / LENS_CACHE_KEY_FILENAME
        tampered = key.as_payload()
        tampered["dim_batch"] = 64
        sidecar.write_text(json.dumps(tampered))
        with pytest.raises(RuntimeError, match="does not match"):
            cache.lookup(key, tmp_path / "out" / "lens.pt")


class _ScriptedS3:
    """A runner that records every argv and answers from a script keyed on the subcommand."""

    def __init__(self, ls: subprocess.CompletedProcess[str], cp_returncode: int = 0) -> None:
        self.ls = ls
        self.cp_returncode = cp_returncode
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self, argv: Sequence[str], timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        self.calls.append(tuple(argv))
        if argv[2] == "ls":
            return self.ls
        return subprocess.CompletedProcess(list(argv), self.cp_returncode, "", "")


def _ls(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["aws", "s3", "ls"], returncode, stdout, stderr)


class TestLensCacheS3:
    def test_a_missing_object_is_a_miss_and_nothing_is_copied(self, tmp_path: Path) -> None:
        runner = _ScriptedS3(ls=_ls(1))
        cache = LensCache(root="s3://bucket/games_rl/lens-cache", runner=runner)
        assert cache.lookup(_key(), tmp_path / "lens.pt") is False
        assert [call[2] for call in runner.calls] == ["ls"]
        assert runner.calls[0][3].startswith("s3://bucket/games_rl/lens-cache/")
        assert runner.calls[0][3].endswith(f"/{LENS_CACHE_KEY_FILENAME}")

    def test_a_denied_or_broken_ls_raises_rather_than_refitting(self, tmp_path: Path) -> None:
        runner = _ScriptedS3(ls=_ls(1, stderr="An error occurred (AccessDenied)"))
        cache = LensCache(root="s3://bucket/prefix", runner=runner)
        with pytest.raises(S3CommandError, match="AccessDenied"):
            cache.lookup(_key(), tmp_path / "lens.pt")

    def test_store_uploads_the_lens_before_the_key_sidecar(self, tmp_path: Path) -> None:
        runner = _ScriptedS3(ls=_ls(1))
        cache = LensCache(root="s3://bucket/prefix", runner=runner)
        lens_path = tmp_path / LENS_CACHE_LENS_FILENAME
        lens_path.write_bytes(b"lens")
        key = _key()
        cache.store(key, lens_path)
        copies = [call for call in runner.calls if call[2] == "cp"]
        assert [call[4].rsplit("/", 1)[-1] for call in copies] == [
            LENS_CACHE_LENS_FILENAME,
            LENS_CACHE_KEY_FILENAME,
        ]
        assert all(call[4].startswith(f"s3://bucket/prefix/{key.sha256}/") for call in copies)
        assert all(call[-1] == "--only-show-errors" for call in copies)

    def test_a_failed_upload_raises_an_s3_error(self, tmp_path: Path) -> None:
        runner = _ScriptedS3(ls=_ls(1), cp_returncode=1)
        cache = LensCache(root="s3://bucket/prefix", runner=runner)
        lens_path = tmp_path / LENS_CACHE_LENS_FILENAME
        lens_path.write_bytes(b"lens")
        with pytest.raises(S3CommandError, match="exited 1"):
            cache.store(_key(), lens_path)


class TestRunS3:
    """The default runner: a command that hangs past its timeout is an S3 error, not a crash."""

    def test_a_hung_command_is_reported_as_an_s3_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def hang(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        monkeypatch.setattr(subprocess, "run", hang)
        with pytest.raises(S3CommandError, match="did not finish within 7s") as raised:
            _run_s3(("aws", "s3", "cp", "lens.pt", "s3://bucket/prefix/lens.pt"), 7)
        assert isinstance(raised.value.__cause__, subprocess.TimeoutExpired)

    def test_a_finished_command_comes_back_as_its_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def finish(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            seen.update(argv=argv, **kwargs)
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(subprocess, "run", finish)
        finished = _run_s3(("aws", "s3", "ls", "s3://bucket/prefix/"), 30)
        assert finished.returncode == 0
        assert seen["timeout"] == 30
        assert seen["check"] is False


class TestAcquireLens:
    """Fit once, reload from the cache after, and refuse a reload that is not that lens."""

    def test_the_second_acquisition_is_a_hit_that_equals_the_fit(self, tmp_path: Path) -> None:
        jl = _CachingJlens(fill=2.0)
        cache = LensCache(root=str(tmp_path / "cache"))
        config = JacobianConfig(max_fit_prompts=2)
        first = acquire_lens(
            config,
            _LENS_MODEL,
            ["a", "b"],
            cast("Any", jl),
            lens_path=tmp_path / "first" / "lens.pt",
            cache=cache,
            key=_key(),
        )
        second = acquire_lens(
            config,
            _LENS_MODEL,
            ["a", "b"],
            cast("Any", jl),
            lens_path=tmp_path / "second" / "lens.pt",
            cache=cache,
            key=_key(),
        )
        assert (first.source, second.source) == ("fit", "cache")
        assert first.cache_stored is True
        assert jl.fit_calls == 1
        for layer in first.lens.jacobians:  # pyright: ignore[reportAttributeAccessIssue]
            assert torch.equal(
                first.lens.jacobians[layer],  # pyright: ignore[reportAttributeAccessIssue]
                second.lens.jacobians[layer],  # pyright: ignore[reportAttributeAccessIssue]
            )
        assert (tmp_path / "second" / "lens.pt").read_bytes() == (
            tmp_path / "first" / "lens.pt"
        ).read_bytes()
        assert second.as_payload()["key_sha256"] == _key().sha256

    def test_any_changed_input_misses_and_refits(self, tmp_path: Path) -> None:
        jl = _CachingJlens()
        cache = LensCache(root=str(tmp_path / "cache"))
        config = JacobianConfig(max_fit_prompts=2)
        acquire_lens(
            config,
            _LENS_MODEL,
            ["a"],
            cast("Any", jl),
            lens_path=tmp_path / "a" / "lens.pt",
            cache=cache,
            key=_key(),
        )
        again = acquire_lens(
            config,
            _LENS_MODEL,
            ["a"],
            cast("Any", jl),
            lens_path=tmp_path / "b" / "lens.pt",
            cache=cache,
            key=_key(fit_prompts_sha256=digest_strings(["other corpus"])),
        )
        assert again.source == "fit"
        assert jl.fit_calls == 2

    def test_without_a_cache_the_fit_is_still_saved_and_round_tripped(self, tmp_path: Path) -> None:
        jl = _CachingJlens()
        acquired = acquire_lens(
            JacobianConfig(max_fit_prompts=1),
            _LENS_MODEL,
            ["a"],
            cast("Any", jl),
            lens_path=tmp_path / "lens.pt",
        )
        assert acquired.source == "fit"
        assert acquired.cache_root is None
        assert (tmp_path / "lens.pt").is_file()

    def test_a_cache_without_a_key_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="both a cache and a key"):
            acquire_lens(
                JacobianConfig(),
                _LENS_MODEL,
                ["a"],
                cast("Any", _CachingJlens()),
                lens_path=tmp_path / "lens.pt",
                cache=LensCache(root=str(tmp_path)),
            )

    def test_a_cached_lens_of_the_wrong_shape_is_refused_on_the_hit(self, tmp_path: Path) -> None:
        """Sabotage kept as a test: a well-formed lens for another model under this key."""
        jl = _CachingJlens()
        cache = LensCache(root=str(tmp_path / "cache"))
        key = _key()
        wrong = _FileLens({0: torch.ones(D_MODEL + 1, D_MODEL + 1)}, d_model=D_MODEL + 1)
        wrong_path = tmp_path / "wrong.pt"
        wrong.save(str(wrong_path))
        cache.store(key, wrong_path)
        with pytest.raises(RuntimeError, match="source layers"):
            acquire_lens(
                JacobianConfig(),
                _LENS_MODEL,
                ["a"],
                cast("Any", jl),
                lens_path=tmp_path / "out" / "lens.pt",
                cache=cache,
                key=key,
            )

    def test_a_failed_store_is_reported_and_the_fit_survives(self, tmp_path: Path) -> None:
        jl = _CachingJlens()
        cache = LensCache(root="s3://bucket/prefix", runner=_ScriptedS3(ls=_ls(1), cp_returncode=1))
        acquired = acquire_lens(
            JacobianConfig(max_fit_prompts=1),
            _LENS_MODEL,
            ["a"],
            cast("Any", jl),
            lens_path=tmp_path / "lens.pt",
            cache=cache,
            key=_key(),
        )
        assert acquired.source == "fit"
        assert acquired.cache_stored is False
        assert acquired.cache_store_error is not None
        assert (tmp_path / "lens.pt").is_file()

    def test_an_upload_that_hangs_is_reported_and_the_fit_survives(self, tmp_path: Path) -> None:
        """The runner contract end to end: a timeout surfaces as the error the store path catches."""

        def hung_upload(
            argv: Sequence[str], timeout_seconds: int
        ) -> subprocess.CompletedProcess[str]:
            if argv[2] == "cp":
                raise S3CommandError(
                    f"`{' '.join(argv)}` did not finish within {timeout_seconds}s and was killed"
                )
            return _ls(1)

        jl = _CachingJlens()
        cache = LensCache(root="s3://bucket/prefix", runner=hung_upload, timeout_seconds=1800)
        acquired = acquire_lens(
            JacobianConfig(max_fit_prompts=1),
            _LENS_MODEL,
            ["a"],
            cast("Any", jl),
            lens_path=tmp_path / "lens.pt",
            cache=cache,
            key=_key(),
        )
        assert jl.fit_calls == 1
        assert acquired.source == "fit"
        assert acquired.cache_stored is False
        assert acquired.cache_store_error is not None
        assert "did not finish within 1800s" in acquired.cache_store_error
        assert (tmp_path / "lens.pt").is_file()


class TestVerifyLens:
    def test_a_saved_lens_round_trips(self, tmp_path: Path) -> None:
        lens = _CachingJlens().fit(object(), [])
        lens.save(str(tmp_path / "lens.pt"))
        verify_lens_roundtrip(_CachingJlens(), lens, tmp_path / "lens.pt")

    def test_a_file_that_is_not_the_fitted_lens_fails_the_round_trip(self, tmp_path: Path) -> None:
        lens = _CachingJlens(fill=1.0).fit(object(), [])
        _CachingJlens(fill=5.0).fit(object(), []).save(str(tmp_path / "lens.pt"))
        with pytest.raises(RuntimeError, match="did not round-trip"):
            verify_lens_roundtrip(_CachingJlens(), lens, tmp_path / "lens.pt")

    def test_a_non_finite_fit_is_refused(self, tmp_path: Path) -> None:
        lens = _FileLens({0: torch.full((D_MODEL, D_MODEL), float("nan"))}, d_model=D_MODEL)
        lens.save(str(tmp_path / "lens.pt"))
        with pytest.raises(RuntimeError, match="non-finite"):
            verify_lens_roundtrip(_CachingJlens(), lens, tmp_path / "lens.pt")

    def test_a_cached_lens_must_cover_every_layer_below_the_target(self) -> None:
        verify_cached_lens(_CachingJlens().fit(object(), []), _LENS_MODEL)
        short = _FileLens({0: torch.ones(D_MODEL, D_MODEL)}, d_model=D_MODEL)
        with pytest.raises(RuntimeError, match="source layers"):
            verify_cached_lens(short, _LENS_MODEL)
        wide = _FileLens(
            {layer: torch.ones(D_MODEL + 1, D_MODEL + 1) for layer in range(N_LAYERS - 1)},
            d_model=D_MODEL + 1,
        )
        with pytest.raises(RuntimeError, match="d_model"):
            verify_cached_lens(wide, _LENS_MODEL)


class TestResolveWeightsIdentity:
    def test_a_hub_id_resolves_to_its_commit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_transformers = SimpleNamespace(
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda model_id, revision=None: SimpleNamespace(
                    _commit_hash=f"sha-of-{model_id}"
                )
            )
        )
        monkeypatch.setattr(jacobian.importlib, "import_module", lambda name: fake_transformers)
        assert resolve_weights_identity("Org/Model") == "hf:sha-of-Org/Model"

    def test_a_hub_id_without_a_commit_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_transformers = SimpleNamespace(
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda model_id, revision=None: SimpleNamespace()
            )
        )
        monkeypatch.setattr(jacobian.importlib, "import_module", lambda name: fake_transformers)
        with pytest.raises(RuntimeError, match="no commit hash"):
            resolve_weights_identity("Org/Model")

    def test_a_local_directory_is_digested_and_a_changed_byte_changes_it(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"weights")
        before = resolve_weights_identity(str(tmp_path))
        assert before == "sha256:" + local_weights_digest(tmp_path)
        (tmp_path / "model.safetensors").write_bytes(b"weightz")
        assert resolve_weights_identity(str(tmp_path)) != before

    def test_a_directory_without_weights_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="safetensors"):
            local_weights_digest(tmp_path)
