from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from reward_hacking.distill import fla_prewarm


class FakeKernel:
    def __init__(self, *, do_not_specialize_t: bool) -> None:
        self.nb_values: list[int] = []
        self.cache_results = False
        self.fn = SimpleNamespace(
            params=[SimpleNamespace(name="T", do_not_specialize=do_not_specialize_t)],
            device_caches={},
        )

    def __getitem__(self, grid: object) -> Any:
        def launch(*args: Any, **kwargs: Any) -> int:
            self.nb_values.append(kwargs["NB"])
            return kwargs["NB"]

        return launch


def fake_kernel_modules() -> tuple[ModuleType, ModuleType, dict[str, FakeKernel]]:
    kernels = {
        name: FakeKernel(do_not_specialize_t=name in fla_prewarm.L2NORM_KERNEL_NAMES)
        for name in (*fla_prewarm.CONV_KERNEL_NAMES, *fla_prewarm.L2NORM_KERNEL_NAMES)
    }
    conv_module = ModuleType("fake_conv_ops")
    l2norm_module = ModuleType("fake_l2norm")
    for name in fla_prewarm.CONV_KERNEL_NAMES:
        setattr(conv_module, name, kernels[name])
    for name in fla_prewarm.L2NORM_KERNEL_NAMES:
        setattr(l2norm_module, name, kernels[name])
    return conv_module, l2norm_module, kernels


def qwen_config() -> SimpleNamespace:
    return SimpleNamespace(
        text_config=SimpleNamespace(
            linear_num_key_heads=2,
            linear_key_head_dim=4,
            linear_num_value_heads=4,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            hidden_act="silu",
        )
    )


def test_nb_bucket_rounds_up_to_power_of_two() -> None:
    assert [fla_prewarm.bucket_nb(nb) for nb in (1, 2, 3, 8, 9, 15, 16)] == [1, 2, 4, 8, 16, 16, 16]


@pytest.mark.parametrize("nb", [0, -1, True])
def test_nb_bucket_rejects_invalid_values(nb: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        fla_prewarm.bucket_nb(nb)


def test_install_wraps_all_four_kernels_and_is_idempotent() -> None:
    conv_module, l2norm_module, kernels = fake_kernel_modules()

    fla_prewarm._install_fla_nb_bucket_wrappers(
        fla_version=fla_prewarm.EXPECTED_FLA_VERSION,
        fla_core_version=fla_prewarm.EXPECTED_FLA_CORE_VERSION,
        conv_module=conv_module,
        l2norm_module=l2norm_module,
    )
    patched_kernels = [getattr(conv_module, name) for name in fla_prewarm.CONV_KERNEL_NAMES] + [
        getattr(l2norm_module, name) for name in fla_prewarm.L2NORM_KERNEL_NAMES
    ]
    assert all(isinstance(kernel, fla_prewarm.NBBucketedKernel) for kernel in patched_kernels)

    conv_module.causal_conv1d_fwd_kernel[(1,)](NB=9)
    conv_module.causal_conv1d_fwd_kernel[(1,)](NB=15)
    l2norm_module.l2norm_fwd_kernel[(1,)](NB=15)
    assert kernels["causal_conv1d_fwd_kernel"].nb_values == [16, 16]
    assert kernels["l2norm_fwd_kernel"].nb_values == [16]
    assert all(kernel.cache_results for kernel in kernels.values())
    assert all(kernel.fn.params[0].do_not_specialize for kernel in kernels.values())

    fla_prewarm._install_fla_nb_bucket_wrappers(
        fla_version=fla_prewarm.EXPECTED_FLA_VERSION,
        fla_core_version=fla_prewarm.EXPECTED_FLA_CORE_VERSION,
        conv_module=conv_module,
        l2norm_module=l2norm_module,
    )
    assert conv_module.causal_conv1d_fwd_kernel.kernel is kernels["causal_conv1d_fwd_kernel"]


def test_install_refuses_unknown_fla_version_without_mutating_modules() -> None:
    conv_module, l2norm_module, _ = fake_kernel_modules()
    original_kernel = conv_module.causal_conv1d_fwd_kernel

    with pytest.raises(RuntimeError, match=r"only supports flash-linear-attention 0\.5\.2"):
        fla_prewarm._install_fla_nb_bucket_wrappers(
            fla_version="0.6.0",
            fla_core_version=fla_prewarm.EXPECTED_FLA_CORE_VERSION,
            conv_module=conv_module,
            l2norm_module=l2norm_module,
        )

    assert conv_module.causal_conv1d_fwd_kernel is original_kernel
    assert isinstance(l2norm_module.l2norm_fwd_kernel, FakeKernel)


def test_install_refuses_unknown_fla_core_version_without_mutating_modules() -> None:
    conv_module, l2norm_module, _ = fake_kernel_modules()
    original_kernel = conv_module.causal_conv1d_fwd_kernel

    with pytest.raises(RuntimeError, match=r"fla-core 0\.5\.2"):
        fla_prewarm._install_fla_nb_bucket_wrappers(
            fla_version=fla_prewarm.EXPECTED_FLA_VERSION,
            fla_core_version="0.6.0",
            conv_module=conv_module,
            l2norm_module=l2norm_module,
        )

    assert conv_module.causal_conv1d_fwd_kernel is original_kernel


def test_install_refuses_to_change_t_specialization_after_a_binder_exists() -> None:
    conv_module, l2norm_module, _ = fake_kernel_modules()
    original_kernel = conv_module.causal_conv1d_fwd_kernel
    original_kernel.fn.device_caches["cuda"] = object()

    with pytest.raises(
        RuntimeError, match=r"before invoking fake_conv_ops\.causal_conv1d_fwd_kernel"
    ):
        fla_prewarm._install_fla_nb_bucket_wrappers(
            fla_version=fla_prewarm.EXPECTED_FLA_VERSION,
            fla_core_version=fla_prewarm.EXPECTED_FLA_CORE_VERSION,
            conv_module=conv_module,
            l2norm_module=l2norm_module,
        )

    assert conv_module.causal_conv1d_fwd_kernel is original_kernel
    assert isinstance(l2norm_module.l2norm_fwd_kernel, FakeKernel)


def test_unpatched_kernels_keep_a_distinct_raw_key_per_length() -> None:
    conv_module, _, kernels = fake_kernel_modules()
    kernel = conv_module.causal_conv1d_fwd_kernel

    for sequence_length in (9 * 1024, 15 * 1024):
        raw_nb = (sequence_length + 1023) // 1024
        kernel[(1,)](NB=raw_nb)

    assert kernels["causal_conv1d_fwd_kernel"].nb_values == [9, 15]
    assert [fla_prewarm.bucket_nb(nb) for nb in kernels["causal_conv1d_fwd_kernel"].nb_values] == [
        16,
        16,
    ]


def test_installed_fla_autotuners_support_patch_in_a_cpu_subprocess() -> None:
    code = """
from importlib import import_module
from reward_hacking.distill.fla_prewarm import NBBucketedKernel, install_fla_nb_bucketing

install_fla_nb_bucketing()
for module_name, names in (
    ("fla.modules.conv.triton.ops", ("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel")),
    ("fla.modules.l2norm", ("l2norm_fwd_kernel", "l2norm_bwd_kernel")),
):
    module = import_module(module_name)
    for name in names:
        kernel = getattr(module, name)
        assert isinstance(kernel, NBBucketedKernel)
        autotuner = kernel.kernel if hasattr(kernel.kernel, "cache_results") else kernel.kernel.fn
        assert autotuner.cache_results is True
        assert next(parameter for parameter in autotuner.fn.params if parameter.name == "T").do_not_specialize
        assert not autotuner.fn.device_caches
"""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, check=False, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_plan_derives_model_dims_and_fills_all_lower_buckets() -> None:
    plan = fla_prewarm.build_fla_nb_prewarm_plan(
        qwen_config(),
        sequence_shapes=((1, 1_000), (1, 9_000), (2, 16_000)),
    )

    assert plan.kernel_shape.conv_dimension == 48
    assert plan.kernel_shape.conv_width == 4
    assert plan.kernel_shape.norm_head_dimension == 4
    assert plan.kernel_shape.norm_heads == 4
    assert plan.kernel_shape.conv_activation == "silu"
    assert plan.conv_buckets == (1, 2, 4, 8, 16, 32)
    assert plan.l2norm_buckets == (1, 2)
    assert plan.conv_sequence_tokens_by_bucket == {
        1: 1,
        2: 1_025,
        4: 2_049,
        8: 4_097,
        16: 8_193,
        32: 16_385,
    }
    assert plan.l2norm_sequence_tokens_by_bucket == {1: 1, 2: 16_385}
    assert plan.estimated_peak_tensor_bytes(element_size=2) > 0


def test_prewarm_refuses_batch_shapes_it_cannot_represent() -> None:
    with pytest.raises(ValueError, match="only supports batch size 1"):
        fla_prewarm.prewarm_fla_nb_buckets(qwen_config(), ((2, 1_000),), device="cpu")
