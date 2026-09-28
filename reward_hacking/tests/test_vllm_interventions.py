"""CPU checks for worker-installed residual edits and fused add-norm semantics."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from reward_hacking.interp.steering import ablate_residual, steer_residual
from reward_hacking.interp.vllm_interventions import (
    InterventionSpec,
    InterventionVLLMBackend,
    ResidualInterventionWorker,
    install,
    intervention,
    remove,
)


class _FusedLayer(torch.nn.Module):
    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stream = hidden + residual
        return stream * 0.25, stream * 0.75


class _SingleLayer(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * 2


class _PairModel(torch.nn.Module):
    def __init__(self, *, width: int = 3, n_layers: int = 2) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=width, num_hidden_layers=n_layers),
            layers=torch.nn.ModuleList(_FusedLayer() for _ in range(n_layers)),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = torch.zeros_like(hidden)
        for layer in self.model.layers:
            hidden, residual = layer(hidden, residual)
        return hidden + residual


@dataclass
class _FakeLLM(ResidualInterventionWorker):
    model: torch.nn.Module

    def get_model(self) -> torch.nn.Module:
        return self.model

    def collective_rpc(self, method: str, *, args: tuple[object, ...] = ()) -> list[Any]:
        # vLLM's RPC does not carry live tensors or int dict keys to the worker; a JSON round trip
        # imposes the same constraint, so a spec that only works in-process fails here too.
        wire_args = json.loads(json.dumps(args))
        return [getattr(self, method)(*wire_args)]


class TestFusedResidualHook:
    def test_flattened_pairs_edit_the_full_stream_before_the_next_layer(self) -> None:
        llm = _FakeLLM(_PairModel())
        hidden = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        basis = torch.tensor([[0.0, 1.0, 0.0]])
        expected = ablate_residual(hidden, basis)

        with intervention(llm, InterventionSpec.subspace({0: basis}, n_layers=2, d_model=3)):
            actual = llm.model(hidden)

        assert torch.equal(actual, expected)
        assert torch.equal(llm.model(hidden), hidden)

    def test_single_tensor_layer_uses_the_same_steering_math(self) -> None:
        model = _PairModel(n_layers=1)
        model.model.layers = torch.nn.ModuleList([_SingleLayer()])
        llm = _FakeLLM(model)
        direction = torch.tensor([0.0, 3.0, 0.0])
        hidden = torch.tensor([[1.0, 2.0, 3.0]])

        with intervention(
            llm, InterventionSpec.steering({0: direction}, alpha=2.0, n_layers=1, d_model=3)
        ):
            actual = model.model.layers[0](hidden)

        assert torch.equal(actual, steer_residual(hidden * 2, direction, 2.0))

    def test_refuses_wrong_model_dimensions_before_installing(self) -> None:
        llm = _FakeLLM(_PairModel())
        basis = torch.tensor([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="layer count"):
            install(llm, InterventionSpec.subspace({0: basis}, n_layers=3, d_model=3))
        with pytest.raises(ValueError, match="d_model"):
            install(llm, InterventionSpec.subspace({0: torch.eye(4)[:1]}, n_layers=2, d_model=4))

    def test_refuses_nonorthonormal_basis_before_worker_rpc(self) -> None:
        llm = _FakeLLM(_PairModel())
        with pytest.raises(ValueError, match="orthonormal"):
            install(
                llm,
                InterventionSpec.subspace(
                    {0: torch.tensor([[2.0, 0.0, 0.0]])}, n_layers=2, d_model=3
                ),
            )

    def test_refuses_a_hook_that_never_fired_and_clears_it(self) -> None:
        llm = _FakeLLM(_PairModel())
        basis = torch.tensor([[1.0, 0.0, 0.0]])
        with (
            pytest.raises(RuntimeError, match="never fired"),
            intervention(llm, InterventionSpec.subspace({0: basis}, n_layers=2, d_model=3)),
        ):
            pass
        assert torch.equal(llm.model(torch.ones(1, 3)), torch.ones(1, 3))

    def test_refuses_unexpected_output_and_removes_on_error(self) -> None:
        model = _PairModel(n_layers=1)
        model.model.layers = torch.nn.ModuleList([torch.nn.Identity()])
        llm = _FakeLLM(model)
        spec = InterventionSpec.subspace({0: torch.eye(3)[:1]}, n_layers=1, d_model=3)
        install(llm, spec)
        with pytest.raises(ValueError, match="output"):
            model.model.layers[0]((torch.ones(1, 3), torch.ones(1, 3), torch.ones(1, 3)))
        remove(llm)


@dataclass
class _FakeSamplingParams:
    seed: int | None = None
    allowed_token_ids: list[int] | None = None

    def clone(self) -> _FakeSamplingParams:
        return deepcopy(self)


class TestVLLMTokenBan:
    def test_bans_exact_ids_and_audits_generated_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = InterventionVLLMBackend.__new__(InterventionVLLMBackend)
        backend._llm = SimpleNamespace(  # pyright: ignore[reportPrivateUsage]
            llm_engine=SimpleNamespace(
                model_config=SimpleNamespace(hf_text_config=SimpleNamespace(vocab_size=4))
            )
        )
        backend._sampling_params = _FakeSamplingParams()  # pyright: ignore[reportPrivateUsage]
        backend._tokenizer = range(4)  # pyright: ignore[reportPrivateUsage]
        backend.clear_generation_audit()
        allowed_seen: list[list[int]] = []

        def fake_generate_tokenized(_prompts: list[str]) -> list[Any]:
            allowed_seen.append(backend._sampling_params.allowed_token_ids)  # pyright: ignore[reportPrivateUsage]
            return [
                SimpleNamespace(
                    response_token_ids=(0, 2), completion=SimpleNamespace(text="synthetic")
                )
            ]

        monkeypatch.setattr(backend, "generate_tokenized", fake_generate_tokenized)
        backend.set_generation_seed(7)

        assert backend.generate(["synthetic"], generation_kwargs={"banned_token_ids": (1,)}) == [
            "synthetic"
        ]
        assert allowed_seen == [[0, 2, 3]]
        assert backend.last_banned_token_counts == (0,)
        assert backend._sampling_params.seed == 7  # pyright: ignore[reportPrivateUsage]
        assert backend._sampling_params.allowed_token_ids is None  # pyright: ignore[reportPrivateUsage]

    def test_refuses_a_banned_id_returned_by_the_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = InterventionVLLMBackend.__new__(InterventionVLLMBackend)
        backend._llm = SimpleNamespace(  # pyright: ignore[reportPrivateUsage]
            llm_engine=SimpleNamespace(
                model_config=SimpleNamespace(hf_text_config=SimpleNamespace(vocab_size=3))
            )
        )
        backend._sampling_params = _FakeSamplingParams()  # pyright: ignore[reportPrivateUsage]
        backend._tokenizer = range(3)  # pyright: ignore[reportPrivateUsage]
        backend.clear_generation_audit()
        monkeypatch.setattr(
            backend,
            "generate_tokenized",
            lambda _prompts: [
                SimpleNamespace(
                    response_token_ids=(1,), completion=SimpleNamespace(text="synthetic")
                )
            ],
        )

        with pytest.raises(AssertionError, match="banned token"):
            backend.generate(["synthetic"], generation_kwargs={"banned_token_ids": (1,)})

    def test_excludes_padded_model_ids_beyond_the_tokenizer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = InterventionVLLMBackend.__new__(InterventionVLLMBackend)
        backend._llm = SimpleNamespace(  # pyright: ignore[reportPrivateUsage]
            llm_engine=SimpleNamespace(
                model_config=SimpleNamespace(hf_text_config=SimpleNamespace(vocab_size=5))
            )
        )
        backend._sampling_params = _FakeSamplingParams()  # pyright: ignore[reportPrivateUsage]
        backend._tokenizer = range(4)  # pyright: ignore[reportPrivateUsage]
        backend.clear_generation_audit()
        allowed_seen: list[list[int]] = []

        def fake_generate_tokenized(_prompts: list[str]) -> list[Any]:
            allowed_seen.append(backend._sampling_params.allowed_token_ids)  # pyright: ignore[reportPrivateUsage]
            return [
                SimpleNamespace(
                    response_token_ids=(2,), completion=SimpleNamespace(text="synthetic")
                )
            ]

        monkeypatch.setattr(backend, "generate_tokenized", fake_generate_tokenized)

        backend.generate(["synthetic"], generation_kwargs={"banned_token_ids": (1,)})

        assert allowed_seen == [[0, 2, 3]]
