"""Resolution, contract, and GPU parity checks for the fla causal-convolution bridge."""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

import games.deltanet_kernels as kernels
from games import preflight

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GPU_TESTS_ENV = "GAMES_GPU_TESTS"
CONV_DIM = 8192
KERNEL_WIDTH = 4


def run_in_fresh_interpreter(snippet: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(snippet)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class TestCausalConvKernelResolution:
    def test_the_bridge_binds_both_model_wrappers_to_the_repo_shims(self) -> None:
        result = run_in_fresh_interpreter(
            """
            import json
            from games.deltanet_kernels import (
                assert_causal_conv_kernels_bound,
                bound_deltanet_kernels,
                bridge_causal_conv_kernels,
            )

            bridge_causal_conv_kernels()
            assert_causal_conv_kernels_bound()
            print(json.dumps(bound_deltanet_kernels(), sort_keys=True))
            """
        )
        assert result.returncode == 0, result.stderr
        assert '"causal_conv1d_fn": "games.deltanet_kernels.fla_causal_conv1d_fn"' in result.stdout
        assert (
            '"causal_conv1d_update": "games.deltanet_kernels.fla_causal_conv1d_update"'
            in result.stdout
        )

    def test_preflight_reports_the_actual_bound_mapping_after_the_bridge(self) -> None:
        result = run_in_fresh_interpreter(
            """
            from games.deltanet_kernels import bound_deltanet_kernels, bridge_decode_kernel
            from games.preflight import deltanet_kernel_paths

            bridge_decode_kernel()
            assert deltanet_kernel_paths() == bound_deltanet_kernels()
            """
        )
        assert result.returncode == 0, result.stderr

    def test_a_repeat_bridge_call_rejects_a_wrapper_rebound_to_the_torch_fallback(self) -> None:
        """Sabotage after bridging; idempotence must verify the binding instead of repairing it."""
        result = run_in_fresh_interpreter(
            """
            from games.deltanet_kernels import (
                bridge_causal_conv_kernels,
            )
            from transformers.integrations.hub_kernels import use_kernel_func_from_hub_with_fallback
            from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

            bridge_causal_conv_kernels()
            modeling.causal_conv1d_fn = use_kernel_func_from_hub_with_fallback(
                "missing_causal_conv_for_sabotage", "games.deltanet_kernels"
            )(modeling.causal_conv1d_fn.__wrapped__)
            bridge_causal_conv_kernels()
            """
        )
        assert result.returncode != 0, result.stdout
        assert "causal_conv1d_fn" in result.stderr, result.stderr
        assert "resolved" in result.stderr, result.stderr
        assert "expected" in result.stderr, result.stderr

    def test_preflight_warns_when_imported_wrappers_are_still_bound_to_torch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        importlib.import_module(kernels.QWEN3_5_MODELING_MODULE)
        with caplog.at_level(logging.WARNING, logger="games.preflight"):
            paths = preflight.log_deltanet_kernel_paths(expected_linear_attention_layers=1)
        for function_name in ("causal_conv1d_fn", "causal_conv1d_update"):
            assert paths[function_name].startswith("transformers.models.qwen3_5.modeling_qwen3_5")
            assert f"DeltaNet kernel on the SLOW path, {function_name}" in caplog.text


class TestDecodeInputValidation:
    def test_prefill_rejects_an_activation_fla_does_not_support(self) -> None:
        with pytest.raises(ValueError, match="supports only"):
            kernels.fla_causal_conv1d_fn(
                torch.zeros(1, 8, 3), torch.zeros(8, KERNEL_WIDTH), activation="relu"
            )

    def test_decode_rejects_an_activation_fla_does_not_support(self) -> None:
        with pytest.raises(ValueError, match="supports only"):
            kernels.fla_causal_conv1d_update(
                torch.zeros(1, 8, 1),
                torch.zeros(1, 8, KERNEL_WIDTH),
                torch.zeros(8, KERNEL_WIDTH),
                activation="relu",
            )

    def test_decode_accepts_exactly_one_token(self) -> None:
        hidden_states = torch.zeros(2, 8, 2)
        conv_state = torch.zeros(2, 8, KERNEL_WIDTH)
        weight = torch.zeros(8, KERNEL_WIDTH)
        with pytest.raises(ValueError, match="one token"):
            kernels.fla_causal_conv1d_update(hidden_states, conv_state, weight)

    def test_decode_requires_a_contiguous_cache(self) -> None:
        hidden_states = torch.zeros(2, 8, 1)
        conv_state = torch.zeros(2, KERNEL_WIDTH, 8).transpose(1, 2)
        weight = torch.zeros(8, KERNEL_WIDTH)
        assert not conv_state.is_contiguous()
        with pytest.raises(ValueError, match="contiguous"):
            kernels.fla_causal_conv1d_update(hidden_states, conv_state, weight)

    def test_decode_requires_cache_and_kernel_widths_to_match(self) -> None:
        hidden_states = torch.zeros(2, 8, 1)
        conv_state = torch.zeros(2, 8, KERNEL_WIDTH + 1)
        weight = torch.zeros(8, KERNEL_WIDTH)
        with pytest.raises(ValueError, match="width"):
            kernels.fla_causal_conv1d_update(hidden_states, conv_state, weight)


def assert_numerically_close(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    max_abs_diff = float((candidate.float() - reference.float()).abs().max())
    reference_max_abs = float(reference.float().abs().max())
    assert max_abs_diff <= absolute_tolerance + relative_tolerance * reference_max_abs, (
        f"max_abs_diff={max_abs_diff:.6g}, reference_max_abs={reference_max_abs:.6g}, "
        f"rtol={relative_tolerance}, atol={absolute_tolerance}"
    )


@pytest.mark.skipif(
    os.environ.get(GPU_TESTS_ENV) != "1" or not torch.cuda.is_available(),
    reason=f"GPU kernel parity: set {GPU_TESTS_ENV}=1 after GPU preflight and use the limiter",
)
class TestCausalConvGpuParity:
    """fla against HF's reachable torch references at the Qwen3.5-9B convolution width."""

    @pytest.mark.parametrize(
        ("activation", "weight_dtype"),
        [("silu", torch.bfloat16), (None, torch.float32)],
        ids=["silu-bf16", "no-activation-fp32-weight"],
    )
    @pytest.mark.parametrize("sequence_length", [1, 257], ids=["shorter-than-kernel", "long"])
    def test_prefill_output_and_input_weight_gradients_match_the_torch_reference(
        self, activation: str | None, weight_dtype: torch.dtype, sequence_length: int
    ) -> None:
        modeling = importlib.import_module(kernels.QWEN3_5_MODELING_MODULE)

        generator = torch.Generator(device="cuda").manual_seed(15)
        hidden = torch.randn(
            1,
            sequence_length,
            CONV_DIM,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ).transpose(1, 2)
        if sequence_length > 1:
            assert not hidden.is_contiguous()
        weight = (
            torch.randn(
                CONV_DIM,
                KERNEL_WIDTH,
                device="cuda",
                dtype=weight_dtype,
                generator=generator,
            )
            * 0.1
        )
        bias = torch.randn(CONV_DIM, device="cuda", dtype=weight_dtype, generator=generator) * 0.1
        upstream = torch.randn(
            hidden.shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )

        candidate_hidden = hidden.detach().requires_grad_()
        candidate_weight = weight.detach().requires_grad_()
        candidate_bias = bias.detach().requires_grad_()
        reference_hidden = hidden.detach().requires_grad_()
        reference_weight = weight.detach().requires_grad_()
        reference_bias = bias.detach().requires_grad_()

        candidate = kernels.fla_causal_conv1d_fn(
            candidate_hidden, candidate_weight, candidate_bias, activation=activation
        )
        reference_fn = cast(
            "Callable[..., torch.Tensor]",
            getattr(modeling.causal_conv1d_fn, "__wrapped__", None),
        )
        assert callable(reference_fn)
        reference = reference_fn(
            reference_hidden, reference_weight, reference_bias, activation=activation
        )
        candidate_gradients = torch.autograd.grad(
            candidate,
            (candidate_hidden, candidate_weight, candidate_bias),
            grad_outputs=upstream,
        )
        reference_gradients = torch.autograd.grad(
            reference,
            (reference_hidden, reference_weight, reference_bias),
            grad_outputs=upstream,
        )

        assert candidate.shape == hidden.shape
        assert candidate.dtype == hidden.dtype
        assert_numerically_close(
            candidate, reference, relative_tolerance=0.02, absolute_tolerance=0.01
        )
        assert_numerically_close(
            candidate_gradients[0],
            reference_gradients[0],
            relative_tolerance=0.03,
            absolute_tolerance=0.02,
        )
        assert_numerically_close(
            candidate_gradients[1],
            reference_gradients[1],
            relative_tolerance=0.03,
            absolute_tolerance=0.05,
        )
        assert_numerically_close(
            candidate_gradients[2],
            reference_gradients[2],
            relative_tolerance=0.03,
            absolute_tolerance=0.05,
        )

    @pytest.mark.parametrize(
        ("activation", "weight_dtype"),
        [("silu", torch.bfloat16), (None, torch.float32)],
        ids=["silu-bf16", "no-activation-fp32-weight"],
    )
    def test_repeated_decode_updates_match_output_and_cache_state(
        self, activation: str | None, weight_dtype: torch.dtype
    ) -> None:
        modeling = importlib.import_module(kernels.QWEN3_5_MODELING_MODULE)

        generator = torch.Generator(device="cuda").manual_seed(16)
        weight = (
            torch.randn(
                CONV_DIM,
                KERNEL_WIDTH,
                device="cuda",
                dtype=weight_dtype,
                generator=generator,
            )
            * 0.1
        )
        bias = torch.randn(CONV_DIM, device="cuda", dtype=weight_dtype, generator=generator) * 0.1
        candidate_state = torch.randn(
            2,
            CONV_DIM,
            KERNEL_WIDTH,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        reference_state = candidate_state.clone()

        for _ in range(8):
            hidden = torch.randn(
                2, 1, CONV_DIM, device="cuda", dtype=torch.bfloat16, generator=generator
            ).transpose(1, 2)
            candidate = kernels.fla_causal_conv1d_update(
                hidden, candidate_state, weight, bias, activation=activation
            )
            reference_update = cast(
                "Callable[..., torch.Tensor]",
                getattr(modeling.causal_conv1d_update, "__wrapped__", None),
            )
            assert callable(reference_update)
            reference = reference_update(
                hidden, reference_state, weight, bias, activation=activation
            )
            assert_numerically_close(
                candidate, reference, relative_tolerance=0.02, absolute_tolerance=0.01
            )
            assert torch.equal(candidate_state, reference_state)
