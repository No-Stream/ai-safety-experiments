"""Offline contracts for the generic Jacobian-lens fitter."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest
import torch

from reward_hacking.interp import lens_fit
from reward_hacking.interp.lens_fit import (
    FitCorpus,
    FitGateError,
    FitMemoryError,
    FitOptions,
    build_parser,
    parse_layer_range,
    read_fit_corpus,
    recovery_gate_from_logits,
    recovery_gate_report,
    run_fit,
)
from reward_hacking.interp.lens_fit_runtime import verify_jlens_fit_behavior

if TYPE_CHECKING:
    from collections.abc import Sequence


def test_parse_layer_range_is_half_open_and_checks_positive_step() -> None:
    assert parse_layer_range("8:44:2") == list(range(8, 44, 2))

    with pytest.raises(ValueError, match="start:stop:step"):
        parse_layer_range("8:44")
    with pytest.raises(ValueError, match="step"):
        parse_layer_range("8:44:0")


def test_read_fit_corpus_caps_in_file_order_and_digests_the_source(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps({"text": "synthetic training sentence one"})
        + "\n"
        + json.dumps({"text": "synthetic training sentence two"})
        + "\n"
    )

    corpus = read_fit_corpus(corpus_path, n_prompts=1)

    assert isinstance(corpus, FitCorpus)
    assert corpus.prompts == ("synthetic training sentence one",)
    assert corpus.source_sha256 == hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    assert len(corpus.selected_text_sha256) == 64


def test_read_fit_corpus_fails_fast_on_missing_text(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(json.dumps({"prompt": "not the expected field"}) + "\n")

    with pytest.raises(ValueError, match="text"):
        read_fit_corpus(corpus_path, n_prompts=1)


def test_read_fit_corpus_validates_rows_after_prompt_cap(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(json.dumps({"text": "synthetic first row"}) + "\n{" + "\n")

    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        read_fit_corpus(corpus_path, n_prompts=1)


def test_recovery_gate_compares_fitted_lens_with_baseline_and_sabotage() -> None:
    target = torch.tensor([1, 2, 1])
    fitted = {8: torch.tensor([1, 2, 1])}
    baseline = {8: torch.tensor([0, 0, 0])}
    sabotaged = {8: baseline[8]}

    report = recovery_gate_report(
        target,
        fitted_token_ids=fitted,
        logit_lens_token_ids=baseline,
        sabotaged_token_ids=sabotaged,
    )
    report_payload = cast("dict[str, Any]", report)

    assert report_payload["passed"] is True
    assert report_payload["layers"]["8"]["jacobian_top1_accuracy"] == 1.0
    assert report_payload["layers"]["8"]["logit_lens_top1_accuracy"] == 0.0
    assert report_payload["layers"]["8"]["sabotaged_top1_accuracy"] == 0.0


def test_recovery_gate_goes_red_when_sabotaged_lens_recovers_more_tokens() -> None:
    target = torch.tensor([1, 2])
    fitted = {8: torch.tensor([0, 0])}
    baseline = {8: fitted[8]}
    sabotaged = {8: torch.tensor([1, 2])}

    report = recovery_gate_report(
        target,
        fitted_token_ids=fitted,
        logit_lens_token_ids=baseline,
        sabotaged_token_ids=sabotaged,
    )

    assert cast("dict[str, Any]", report)["passed"] is False


def test_recovery_gate_refuses_non_finite_logits_before_argmax() -> None:
    logits = {8: torch.tensor([[float("nan"), 0.0]])}

    with pytest.raises(FitGateError, match="non-finite"):
        recovery_gate_from_logits(
            torch.tensor([[0.0, 1.0]]),
            fitted_logits=logits,
            logit_lens_logits={8: torch.tensor([[0.0, 1.0]])},
            sabotaged_logits={8: torch.tensor([[1.0, 0.0]])},
        )


def test_cli_parses_requested_gemma_recipe_and_optional_adapter(tmp_path: Path) -> None:
    namespace = build_parser().parse_args(
        [
            "--model",
            "google/gemma-3-12b-it",
            "--adapter",
            "artifacts/atlas/adapters/kl0.01-det3-seed1-mbpp_probe",
            "--corpus",
            str(tmp_path / "corpus.jsonl"),
            "--n-prompts",
            "30",
            "--source-layers",
            "8:44:2",
            "--target-layer",
            "44",
            "--dim-batch",
            "4",
            "--max-seq-len",
            "128",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert namespace.model == "google/gemma-3-12b-it"
    assert namespace.n_prompts == 30
    assert namespace.source_layers == list(range(8, 44, 2))
    assert namespace.target_layer == 44
    assert namespace.dim_batch == 4
    assert namespace.max_seq_len == 128
    assert namespace.adapter.name == "kl0.01-det3-seed1-mbpp_probe"


class _TinyTokenizer:
    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"
    model_max_length = 128
    clean_up_tokenization_spaces = False
    split_special_tokens = False
    chat_template = None
    init_kwargs: ClassVar[dict[str, object]] = {}
    special_tokens_map: ClassVar[dict[str, str]] = {
        "bos_token": "token-0",
        "eos_token": "token-0",
        "pad_token": "token-0",
    }

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 128,
    ) -> SimpleNamespace:
        del return_tensors, truncation
        ids = [self.bos_token_id, *(1 + byte % 30 for byte in text.encode()[: max_length - 1])]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{index}": index for index in range(32)}

    def get_added_vocab(self) -> dict[str, int]:
        return {}


class _TinyResidualBlock(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.linear(hidden)


class _TinyLensModel:
    def __init__(self) -> None:
        torch.manual_seed(17)
        self.n_layers = 3
        self.d_model = 8
        self.tokenizer = _TinyTokenizer()
        self.embed_tokens = torch.nn.Embedding(32, self.d_model)
        self.layers = torch.nn.ModuleList(
            [_TinyResidualBlock(self.d_model) for _ in range(self.n_layers)]
        )
        self.norm = torch.nn.LayerNorm(self.d_model)
        self.lm_head = torch.nn.Linear(self.d_model, 32, bias=False)
        with torch.no_grad():
            self.lm_head.weight[0].fill_(-10.0)

    @property
    def input_device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def encode(self, text: str, *, max_length: int = 128) -> torch.Tensor:
        return self.tokenizer(text, max_length=max_length).input_ids.to(self.input_device)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(residual.float()))


class _TinyHFModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(_commit_hash="synthetic-revision")


def test_pinned_jlens_fit_checkpoint_resume_matches_uninterrupted_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root / "external" / "jacobian-lens"))
    jlens = importlib.import_module("jlens")
    prompts: Sequence[str] = tuple(
        f"synthetic calibration paragraph number {index} repeats for stable token context " * 2
        for index in range(4)
    )
    fit_kwargs = {
        "source_layers": [0, 1],
        "target_layer": 2,
        "dim_batch": 1,
        "max_seq_len": 48,
        "skip_first": 0,
    }

    straight = jlens.fit(_TinyLensModel(), prompts, **fit_kwargs)
    checkpoint_path = tmp_path / "fit-checkpoint.pt"
    jlens.fit(
        _TinyLensModel(),
        prompts[:2],
        **fit_kwargs,
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=1,
        resume=False,
    )
    resumed = jlens.fit(
        _TinyLensModel(),
        prompts,
        **fit_kwargs,
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=1,
        resume=True,
    )

    assert resumed.n_prompts == straight.n_prompts == len(prompts)
    for layer in straight.source_layers:
        torch.testing.assert_close(resumed.jacobians[layer], straight.jacobians[layer])


def test_jlens_behavior_check_exercises_real_fit_and_rejects_zero_jacobians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _add_jlens_to_import_path(monkeypatch)
    jlens = importlib.import_module("jlens")
    verify_jlens_fit_behavior(jlens)

    original_fit = jlens.fit

    def zero_jacobian_fit(*args: Any, **kwargs: Any) -> Any:
        fitted = original_fit(*args, **kwargs)
        return jlens.JacobianLens(
            {layer: torch.zeros_like(matrix) for layer, matrix in fitted.jacobians.items()},
            n_prompts=fitted.n_prompts,
            d_model=fitted.d_model,
        )

    monkeypatch.setattr(jlens, "fit", zero_jacobian_fit)
    with pytest.raises(RuntimeError, match="unusable Jacobian"):
        verify_jlens_fit_behavior(jlens)


class _TinyHandle:
    def __init__(self, model: _TinyLensModel) -> None:
        self.model = model
        self.hf_model = _TinyHFModel()
        self.tokenizer = model.tokenizer
        self.weights_fingerprint = "synthetic-tiny-model-weights"

    def as_payload(self) -> dict[str, object]:
        return {
            "model_id": "synthetic-tiny-model",
            "weights_fingerprint": self.weights_fingerprint,
            "n_layers": self.model.n_layers,
            "d_model": self.model.d_model,
        }


def _install_cpu_fit_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    free_bytes: int = 8 * 2**30,
    peak_bytes: int = 2**20,
) -> _TinyHandle:
    model = _TinyLensModel()
    handle = _TinyHandle(model)
    monkeypatch.setattr(lens_fit, "load_fit_model", lambda *args, **kwargs: handle)
    monkeypatch.setattr(lens_fit, "validate_unembedding", lambda *_args: {"passed": True})
    monkeypatch.setattr(
        lens_fit,
        "run_hybrid_architecture_gates",
        lambda *_args, **_kwargs: {"applicable": False, "passed": True},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device=0: "synthetic test device")
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device=0: (free_bytes, 24 * 2**30),
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device=0: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device=0: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device=0: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device=0: peak_bytes)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device=0: peak_bytes)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return handle


def _fit_options(tmp_path: Path, *, halves: bool = False, dim_batch: int = 1) -> FitOptions:
    corpus_path = tmp_path / "fit-corpus.jsonl"
    corpus_path.write_text(
        "".join(
            json.dumps(
                {"text": f"synthetic calibration paragraph {index} has enough token positions " * 2}
            )
            + "\n"
            for index in range(4)
        )
    )
    return FitOptions(
        model="synthetic-tiny-model",
        corpus=corpus_path,
        n_prompts=4,
        source_layers=(0, 1),
        target_layer=2,
        dim_batch=dim_batch,
        max_seq_len=48,
        out_dir=tmp_path / "lens-output",
        halves=halves,
        checkpoint_every=1,
    )


def _add_jlens_to_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root / "external" / "jacobian-lens"))


def test_run_fit_uses_real_jlens_and_validates_complete_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_jlens_to_import_path(monkeypatch)
    _install_cpu_fit_runtime(monkeypatch)
    options = _fit_options(tmp_path)

    report = run_fit(options)
    report_payload = cast("dict[str, Any]", report)

    provenance_path = options.out_dir / "lens-provenance.json"
    lens_path = options.out_dir / "lens.pt"
    assert provenance_path.is_file()
    assert lens_path.is_file()
    assert report_payload["corpus"]["prompts_fitted"] == 4
    assert report_payload["fit"]["target_read"].startswith("model final-layer logits")
    assert report_payload["gates"]["token_recovery_vs_zero_jacobian"]["passed"] is True
    assert "synthetic calibration paragraph" not in provenance_path.read_text()

    monkeypatch.setattr(
        lens_fit,
        "_preflight_dim_batch",
        lambda *_args, **_kwargs: pytest.fail(
            "a matching completed artifact should skip preflight"
        ),
    )
    assert cast("dict[str, Any]", run_fit(options))["identity"] == report_payload["identity"]

    lens_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or has changed"):
        run_fit(options)


def test_run_fit_refuses_requested_dim_batch_before_fit_when_memory_bound_is_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_jlens_to_import_path(monkeypatch)
    _install_cpu_fit_runtime(monkeypatch, free_bytes=1, peak_bytes=1024)
    jlens = importlib.import_module("jlens")
    original_jacobian_for_prompt = jlens.jacobian_for_prompt
    observed_preflight_batches: list[int] = []

    def record_preflight_batch(
        model: Any, prompt: str, source_layers: Sequence[int], **kwargs: Any
    ) -> Any:
        if isinstance(model, _TinyLensModel):
            observed_preflight_batches.append(kwargs["dim_batch"])
        return original_jacobian_for_prompt(model, prompt, source_layers, **kwargs)

    monkeypatch.setattr(jlens, "jacobian_for_prompt", record_preflight_batch)
    options = _fit_options(tmp_path, dim_batch=4)

    with pytest.raises(lens_fit.FitMemoryError, match="no multi-dimension probe was run"):
        run_fit(options)

    assert observed_preflight_batches == [1]
    assert not (options.out_dir / "fit-checkpoint.pt").exists()
    assert not (options.out_dir / "lens-provenance.json").exists()


def test_memory_preflight_refuses_when_single_dimension_probe_has_no_cuda_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _add_jlens_to_import_path(monkeypatch)
    _install_cpu_fit_runtime(monkeypatch, peak_bytes=0)
    jlens = importlib.import_module("jlens")

    with pytest.raises(FitMemoryError, match="observed no CUDA allocation"):
        lens_fit._preflight_dim_batch(
            jlens,
            _TinyLensModel(),
            ("synthetic calibration paragraph with enough token positions",),
            source_layers=(0, 1),
            target_layer=2,
            dim_batch=1,
            max_seq_len=48,
            skip_first=0,
        )


def test_run_fit_uses_compatible_dirty_jlens_and_computation_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_jlens_to_import_path(monkeypatch)
    _install_cpu_fit_runtime(monkeypatch)
    options = _fit_options(tmp_path)
    monkeypatch.setattr(
        lens_fit,
        "jlens_provenance",
        lambda _jl: {
            "path": "/compatible/jlens",
            "commit": "compatible-local-commit",
            "pinned_commit": lens_fit.JLENS_COMMIT,
            "dirty": True,
        },
    )
    report = cast("dict[str, Any]", run_fit(options))
    assert report["gates"]["token_recovery_vs_zero_jacobian"]["passed"] is True
    assert report["jlens"]["commit"] == "compatible-local-commit"
    assert report["jlens"]["dirty"] is True

    provenance_path = options.out_dir / "lens-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    assert provenance["gates"]["token_recovery_vs_zero_jacobian"]["passed"] is True


def test_run_fit_computation_sabotage_still_fails_recovery_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_jlens_to_import_path(monkeypatch)
    _install_cpu_fit_runtime(monkeypatch)
    options = _fit_options(tmp_path)
    jlens = importlib.import_module("jlens")
    original_fit = jlens.fit

    def zero_jacobian_fit(*args: Any, **kwargs: Any) -> Any:
        fitted = original_fit(*args, **kwargs)
        return jlens.JacobianLens(
            {layer: torch.zeros_like(value) for layer, value in fitted.jacobians.items()},
            n_prompts=fitted.n_prompts,
            d_model=fitted.d_model,
        )

    monkeypatch.setattr(jlens, "fit", zero_jacobian_fit)
    monkeypatch.setattr(
        "reward_hacking.interp.lens_fit.verify_jlens_fit_behavior", lambda _jl: None
    )
    with pytest.raises(FitGateError, match="zero-J sabotage"):
        run_fit(options)
    assert not (options.out_dir / "lens-provenance.json").exists()


def test_run_fit_leaves_provenance_absent_when_recovery_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_jlens_to_import_path(monkeypatch)
    _install_cpu_fit_runtime(monkeypatch)
    options = _fit_options(tmp_path)
    original_recovery_read = lens_fit._recovery_read

    def run_zero_jacobian_sabotage(
        jl: Any,
        fitted_lens: Any,
        model: Any,
        prompts: Sequence[str],
        **kwargs: Any,
    ) -> dict[str, object]:
        sabotaged_lens = jl.JacobianLens(
            jacobians={
                layer: torch.zeros_like(matrix) for layer, matrix in fitted_lens.jacobians.items()
            },
            n_prompts=fitted_lens.n_prompts,
            d_model=fitted_lens.d_model,
        )
        return original_recovery_read(jl, sabotaged_lens, model, prompts, **kwargs)

    monkeypatch.setattr(lens_fit, "_recovery_read", run_zero_jacobian_sabotage)
    with pytest.raises(FitGateError, match="zero-J sabotage"):
        run_fit(options)

    assert (options.out_dir / "lens.pt").is_file()
    assert not (options.out_dir / "lens-provenance.json").exists()


def test_halves_resume_identity_and_merge_prompt_weighted_jlens_lenses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_jlens_to_import_path(monkeypatch)
    _install_cpu_fit_runtime(monkeypatch)
    jlens = importlib.import_module("jlens")
    original_fit = jlens.fit
    interrupted = False

    def interrupt_after_first_odd_prompt(model: Any, prompts: Sequence[str], **kwargs: Any) -> Any:
        nonlocal interrupted
        if not interrupted and kwargs.get("checkpoint_path", "").endswith("odd.pt"):
            interrupted = True
            original_fit(model, prompts[:1], **kwargs)
            raise RuntimeError("synthetic interrupted odd-half fit")
        return original_fit(model, prompts, **kwargs)

    monkeypatch.setattr(jlens, "fit", interrupt_after_first_odd_prompt)
    options = _fit_options(tmp_path, halves=True)
    with pytest.raises(RuntimeError, match="interrupted odd-half"):
        run_fit(options)

    assert (options.out_dir / "fit-checkpoint-odd.pt").is_file()
    assert (options.out_dir / "fit-identity-odd.json").is_file()
    assert not (options.out_dir / "lens-provenance.json").exists()

    original_corpus = options.corpus.read_text()
    options.corpus.write_text(original_corpus.replace("paragraph 0", "paragraph changed", 1))
    monkeypatch.setattr(jlens, "fit", original_fit)
    with pytest.raises(RuntimeError, match="fit identity differs"):
        run_fit(options)
    options.corpus.write_text(original_corpus)

    resumed = run_fit(options)
    resumed_payload = cast("dict[str, Any]", resumed)

    assert resumed_payload["corpus"]["prompts_fitted"] == 4
    assert resumed_payload["fit"]["halves_detail"]["odd"]["prompts_fitted"] == 2
    assert resumed_payload["fit"]["halves_detail"]["even"]["prompts_fitted"] == 2
    assert resumed_payload["fit"]["split_half_relative_frobenius"]
    assert (options.out_dir / "lens-half-odd.pt").is_file()
    assert (options.out_dir / "lens-half-even.pt").is_file()
