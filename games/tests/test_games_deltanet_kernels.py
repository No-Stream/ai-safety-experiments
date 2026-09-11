import json
import math
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from games.deltanet_kernels import (
    ALIASED_BY_THIS_PROCESS_REASON,
    ALIASED_NATIVELY_REASON,
    DECODE_RELATIVE_TOLERANCES,
    DELTANET_KERNEL_WRAPPERS,
    DELTANET_PREFILL_KERNELS,
    DecodeInputs,
    DecodeKernel,
    DecodeShape,
    assert_decode_kernels_match,
    assert_one_deltanet_kernel,
    build_decode_inputs,
    compare_decode_kernels,
    decode_call_site,
    deltanet_kernel_label,
    prefill_deltanet_kernels,
    torch_decode_reference,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Anything that applies the bridge runs in a fresh interpreter. The bridge mutates the `fla` module
# object and only works before `transformers.models.qwen3_5.modeling_qwen3_5` is imported, and the
# comparison tests below import that module -- so in-process bridge tests would pass or fail
# depending on collection order, which is the shape of a green gate that checked nothing.
BRIDGE_IMPORT = "from games.deltanet_kernels import bridge_decode_kernel\n"
DECODE_PATH_REPORT = (
    "from games.preflight import deltanet_kernel_paths\n"
    "print('DECODE_PATH', deltanet_kernel_paths()['recurrent_gated_delta_rule'])\n"
)


def run_in_fresh_interpreter(snippet: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(snippet)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@dataclass
class HeadGeometry:
    """The four head-geometry fields DecodeShape reads, stated directly instead of via a config."""

    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int


def without_l2norm(kernel: DecodeKernel) -> DecodeKernel:
    def candidate(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor | None]:
        return kernel(*args, **{**kwargs, "use_qk_l2norm_in_kernel": False})

    return candidate


def without_initial_state(kernel: DecodeKernel) -> DecodeKernel:
    def candidate(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor | None]:
        return kernel(*args, **{**kwargs, "initial_state": None})

    return candidate


class TestDecodeKernelResolution:
    def test_the_decode_kernel_falls_back_to_torch_without_the_bridge(self):
        result = run_in_fresh_interpreter(DECODE_PATH_REPORT)
        assert result.returncode == 0, result.stderr
        assert "DECODE_PATH torch fallback" in result.stdout, result.stdout

    def test_the_bridge_makes_the_decode_kernel_resolve_to_fla(self):
        result = run_in_fresh_interpreter(
            BRIDGE_IMPORT + "bridge_decode_kernel()\n" + DECODE_PATH_REPORT
        )
        assert result.returncode == 0, result.stderr
        assert "DECODE_PATH fla.ops.gated_delta_rule" in result.stdout, result.stdout
        assert "fused_recurrent_gated_delta_rule" in result.stdout, result.stdout

    def test_the_prefill_kernel_was_already_on_fla_either_way(self):
        """The bridge must not disturb the kernel that already resolved.

        If prefill regressed to torch, the speedup measurement would be reading a different change
        than the one that was made.
        """
        snippet = (
            BRIDGE_IMPORT
            + "bridge_decode_kernel()\n"
            + "from games.preflight import deltanet_kernel_paths\n"
            + "print('PREFILL', deltanet_kernel_paths()['chunk_gated_delta_rule'])\n"
        )
        result = run_in_fresh_interpreter(snippet)
        assert result.returncode == 0, result.stderr
        assert "PREFILL fla.ops.gated_delta_rule" in result.stdout, result.stdout

    def test_a_second_call_reports_no_second_patch(self):
        """And names THIS process's own alias as the reason, rather than a native fla export.

        The two are indistinguishable from the attribute alone -- the already-exported gate is checked
        first, so the alias this module installs satisfies it -- and a report that says "fla already
        exports the name" would stamp an artifact with "no bridge was needed on this box" when one was
        applied. The distinction is what makes a memoized first report worth keeping.
        """
        snippet = (
            BRIDGE_IMPORT
            + "first = bridge_decode_kernel()\n"
            + "second = bridge_decode_kernel()\n"
            + "print('FIRST', first['bridged'], 'SECOND', second['bridged'], second['reason'])\n"
        )
        result = run_in_fresh_interpreter(snippet)
        assert result.returncode == 0, result.stderr
        assert "FIRST True SECOND False" in result.stdout, result.stdout
        assert ALIASED_BY_THIS_PROCESS_REASON in result.stdout, result.stdout
        assert ALIASED_NATIVELY_REASON not in result.stdout, result.stdout

    def test_a_name_fla_exported_itself_is_reported_as_fla_s_own(self):
        """The other side of the same distinction: nothing was aliased here, so nothing claims to be.

        Stands in for the future fla release that fixes the export name, which is what makes the
        bridge self-disabling: the reason has to keep saying so, or the two cases collapse again.
        """
        snippet = (
            "import fla.ops.gated_delta_rule as gdr\n"
            "gdr.recurrent_gated_delta_rule = gdr.fused_recurrent_gated_delta_rule\n"
            + BRIDGE_IMPORT
            + "report = bridge_decode_kernel()\n"
            + "print('BRIDGED', report['bridged'], report['reason'])\n"
        )
        result = run_in_fresh_interpreter(snippet)
        assert result.returncode == 0, result.stderr
        assert "BRIDGED False" in result.stdout, result.stdout
        assert ALIASED_NATIVELY_REASON in result.stdout, result.stdout

    def test_bridging_after_the_modeling_module_is_imported_raises(self):
        snippet = (
            "import transformers.models.qwen3_5.modeling_qwen3_5  # noqa: F401\n"
            + BRIDGE_IMPORT
            + "bridge_decode_kernel()\n"
        )
        result = run_in_fresh_interpreter(snippet)
        assert result.returncode != 0, result.stdout
        assert "too late to bridge" in result.stderr, result.stderr

    def test_the_bridged_kernel_accepts_every_argument_transformers_passes(self):
        snippet = (
            BRIDGE_IMPORT
            + "from games.deltanet_kernels import assert_bridged_kernel_matches_call_site\n"
            + "bridge_decode_kernel()\n"
            + "site = assert_bridged_kernel_matches_call_site()\n"
            + "print('CALL_SITE', sorted(site.keyword_names))\n"
        )
        result = run_in_fresh_interpreter(snippet)
        assert result.returncode == 0, result.stderr
        for name in ("g", "beta", "initial_state", "output_final_state", "use_qk_l2norm_in_kernel"):
            assert name in result.stdout, result.stdout

    def test_the_call_site_check_rejects_a_kwargs_only_implementation(self):
        """The exact silent failure the check exists for.

        transformers filters keyword arguments to the implementation's parameter NAMES, and `kwargs`
        is a parameter name, so a bare `**kwargs` implementation passes the filter and then receives
        none of the arguments.
        """
        snippet = (
            "import fla.ops.gated_delta_rule as gdr\n"
            "gdr.recurrent_gated_delta_rule = lambda *args, **kwargs: None\n"
            "from games.deltanet_kernels import assert_bridged_kernel_matches_call_site\n"
            "assert_bridged_kernel_matches_call_site()\n"
        )
        result = run_in_fresh_interpreter(snippet)
        assert result.returncode != 0, result.stdout
        assert "would silently drop" in result.stderr, result.stderr


class TestEntryPointsBridgeInTime:
    """Every entry point that generates has to be able to bridge when its `main` starts.

    The wiring is one call at the top of each `main`, and what makes it work is that none of these
    modules imports the Qwen3.5 modeling module while being imported itself. That is fragile in a
    way nothing else catches: one convenient `from transformers import Qwen3_5ForCausalLM` at the
    top of any of them, or of anything they import, turns the bridge into a startup crash. So the
    invariant is asserted per module, in a fresh interpreter, because in-process it depends on which
    test happened to run first.
    """

    ENTRY_POINT_MODULES = (
        "games.train",
        "games.select_prompts",
        "games.screen_thinking",
        "grpo.throughput",
        # The HuggingFace-path interp legs (owner decision I1): each bridges right before its load.
        "games.interp_capture",
        "games.interp_steering",
        "games.interp_patching",
        "games.interp_lens_ladder",
        "reward_hacking.interp.run_harness",
    )

    @pytest.mark.parametrize("module", ENTRY_POINT_MODULES)
    def test_importing_the_entry_point_leaves_the_bridge_still_applicable(self, module: str):
        snippet = (
            "import sys, importlib\n"
            f"importlib.import_module({module!r})\n"
            "from games.deltanet_kernels import QWEN3_5_MODELING_MODULE, bridge_decode_kernel\n"
            "print('IMPORTED', QWEN3_5_MODELING_MODULE in sys.modules)\n"
            "print('BRIDGED', bridge_decode_kernel()['bridged'])\n"
            "from games.preflight import deltanet_kernel_paths\n"
            "print('DECODE_PATH', deltanet_kernel_paths()['recurrent_gated_delta_rule'])\n"
        )
        result = run_in_fresh_interpreter(snippet)
        assert result.returncode == 0, result.stderr
        assert "IMPORTED False" in result.stdout, result.stdout
        assert "BRIDGED True" in result.stdout, result.stdout
        assert "DECODE_PATH fla.ops.gated_delta_rule" in result.stdout, result.stdout


PREFILL_BINDING: dict[str, str] = {
    "chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
    "causal_conv1d_fn": "transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_fn",
}
"""What a forward with no prior cache state dispatches on this box (fla present, causal-conv1d not)."""

FUSED_BINDING: dict[str, str] = {
    **PREFILL_BINDING,
    "recurrent_gated_delta_rule": (
        "fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule"
    ),
    "causal_conv1d_update": "transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_update",
}
FALLBACK_BINDING: dict[str, str] = {
    **PREFILL_BINDING,
    "recurrent_gated_delta_rule": (
        "transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule"
    ),
    "causal_conv1d_update": "transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_update",
}
"""The two full bindings a generating record can carry: with the fla decode bridge, and without it."""

BOUND_KERNELS_REPORT = (
    "from games.deltanet_kernels import bound_deltanet_kernels\n"
    "import json\n"
    "print('BOUND', json.dumps(bound_deltanet_kernels(), sort_keys=True))\n"
)


class TestBoundKernels:
    """What the kernels are ACTUALLY bound to, read off the modeling module rather than predicted.

    `games.preflight.deltanet_kernel_paths` answers "what would transformers resolve if it bound now";
    this answers "what did it bind". The two differ in exactly the case the bridge exists for: an alias
    registered after `modeling_qwen3_5` was imported changes the prediction and nothing about the
    wrapper transformers calls, so a run that recorded the prediction would name a kernel it did not
    use. Every record of a HuggingFace-path interp leg carries this, so it has to be the second thing.

    Fresh interpreters throughout, for the reason the bridge tests give: the bridge mutates the `fla`
    module and only works before the modeling module is imported, so an in-process test would pass or
    fail on collection order.
    """

    def test_without_the_bridge_the_decode_kernel_reads_as_the_torch_fallback(self):
        result = run_in_fresh_interpreter(BOUND_KERNELS_REPORT)
        assert result.returncode == 0, result.stderr
        bound = json.loads(result.stdout.split("BOUND ", 1)[1])
        assert bound["recurrent_gated_delta_rule"].endswith("torch_recurrent_gated_delta_rule")
        assert bound["chunk_gated_delta_rule"].startswith("fla.ops.gated_delta_rule")

    def test_after_the_bridge_the_decode_kernel_reads_as_flas_fused_one(self):
        result = run_in_fresh_interpreter(
            BRIDGE_IMPORT + "bridge_decode_kernel()\n" + BOUND_KERNELS_REPORT
        )
        assert result.returncode == 0, result.stderr
        bound = json.loads(result.stdout.split("BOUND ", 1)[1])
        assert (
            bound["recurrent_gated_delta_rule"]
            == "fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule"
        )

    def test_every_dispatched_kernel_function_is_named(self):
        result = run_in_fresh_interpreter(BOUND_KERNELS_REPORT)
        assert result.returncode == 0, result.stderr
        bound = json.loads(result.stdout.split("BOUND ", 1)[1])
        assert sorted(bound) == sorted(DELTANET_KERNEL_WRAPPERS)

    def test_a_wrapper_the_modeling_module_no_longer_has_is_refused(self):
        """A record naming three of four kernels would read as complete, so a rename raises instead."""
        snippet = (
            "import transformers.models.qwen3_5.modeling_qwen3_5 as m\n"
            "del m.torch_recurrent_gated_delta_rule\n"
            "from games.deltanet_kernels import bound_deltanet_kernels\n"
            "bound_deltanet_kernels()\n"
        )
        result = run_in_fresh_interpreter(snippet)
        assert result.returncode != 0, result.stdout
        assert "has no torch_recurrent_gated_delta_rule" in result.stderr, result.stderr

    def test_a_wrapper_that_is_not_a_kernel_wrapper_is_refused(self):
        """The binding is read out of the wrapper's closure, so a bare function cannot be read at all.

        Which is the honest outcome: transformers changing how it wraps kernel functions means nobody
        here knows which kernel ran, and guessing would put a plausible name on an unknown.
        """
        snippet = (
            "import transformers.models.qwen3_5.modeling_qwen3_5 as m\n"
            "m.torch_recurrent_gated_delta_rule = lambda *a, **k: None\n"
            "from games.deltanet_kernels import bound_deltanet_kernels\n"
            "bound_deltanet_kernels()\n"
        )
        result = run_in_fresh_interpreter(snippet)
        assert result.returncode != 0, result.stdout
        assert "closes over 0 callables" in result.stderr, result.stderr


class TestPrefillKernels:
    """A forward-only record names the two kernels its forwards dispatched, which the bridge never touches.

    transformers routes `recurrent_gated_delta_rule` and `causal_conv1d_update` behind
    ``use_precomputed_states and seq_len == 1``, which no bare forward satisfies, so an activation
    capture, a patched forward or a lens fit depends on `chunk_gated_delta_rule` and `causal_conv1d_fn`
    only. Keying those legs on all four would refuse a resume across the bridge whose forwards are
    bit-identical to the records it wants to continue.

    Sabotage-verified: returning ``dict(bound)`` from `prefill_deltanet_kernels` turns the bridge
    invisibility test red.
    """

    def test_the_subset_is_exactly_the_two_prefill_kernels(self):
        assert prefill_deltanet_kernels(FALLBACK_BINDING) == PREFILL_BINDING
        assert tuple(prefill_deltanet_kernels(FALLBACK_BINDING)) == DELTANET_PREFILL_KERNELS

    def test_the_bridge_is_invisible_to_a_forward_only_record(self):
        assert prefill_deltanet_kernels(FUSED_BINDING) == prefill_deltanet_kernels(FALLBACK_BINDING)
        assert deltanet_kernel_label(
            prefill_deltanet_kernels(FUSED_BINDING)
        ) == deltanet_kernel_label(prefill_deltanet_kernels(FALLBACK_BINDING))

    def test_a_binding_missing_a_prefill_kernel_is_refused_rather_than_narrowed_to_one(self):
        """A record naming one kernel would read as complete, so the narrowing raises instead."""
        with pytest.raises(ValueError, match=r"names no \['causal_conv1d_fn'\]"):
            prefill_deltanet_kernels(
                {"chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule"}
            )

    def test_the_live_binding_narrows_to_what_a_forward_dispatches(self):
        """Off the real modeling module, in a fresh interpreter, so the constant matches the wrappers."""
        result = run_in_fresh_interpreter(
            BOUND_KERNELS_REPORT + "from games.deltanet_kernels import prefill_deltanet_kernels\n"
            "print('PREFILL', json.dumps(prefill_deltanet_kernels(bound_deltanet_kernels()), "
            "sort_keys=True))\n"
        )
        assert result.returncode == 0, result.stderr
        prefill = json.loads(result.stdout.split("PREFILL ", 1)[1])
        assert sorted(prefill) == sorted(DELTANET_PREFILL_KERNELS)
        assert prefill["chunk_gated_delta_rule"].startswith("fla.ops.gated_delta_rule")


class TestKernelMixingGuard:
    """No analysis may pool records that ran under two kernel bindings (owner decision 15).

    The fused decode kernel and the torch fallback are the same recurrence in a different reduction
    order, and probe I1 watched greedy token ids diverge under them from decode step 21 on. So a
    steering condition decoded under one and its placebo under the other is a comparison of kernels
    wearing an arm's name, and every summariser that pools records calls this first.

    Sabotage-verified: making `assert_one_deltanet_kernel` return `None` without raising turns the
    two-binding and the partially-recorded tests here red.
    """

    def test_one_binding_passes_and_comes_back(self):
        held = assert_one_deltanet_kernel([FUSED_BINDING, dict(FUSED_BINDING)], what="a cell")
        assert held == FUSED_BINDING

    def test_two_bindings_are_refused_and_the_message_names_both(self):
        with pytest.raises(ValueError, match="2 different Gated DeltaNet kernel bindings"):
            assert_one_deltanet_kernel([FUSED_BINDING, FALLBACK_BINDING], what="a cell")

    def test_a_pool_that_is_only_partly_recorded_is_refused(self):
        """A record written before the field existed cannot prove its kernel, so it cannot be pooled
        with one that can."""
        with pytest.raises(ValueError, match="2 different"):
            assert_one_deltanet_kernel([FUSED_BINDING, None], what="a cell")

    def test_records_that_all_predate_the_field_still_summarise(self):
        assert assert_one_deltanet_kernel([None, None], what="a cell") is None

    def test_an_empty_pool_names_no_kernel(self):
        assert assert_one_deltanet_kernel([], what="a cell") is None

    def test_the_label_ignores_key_order_and_keeps_absence_distinct(self):
        reordered = dict(reversed(list(FUSED_BINDING.items())))
        assert deltanet_kernel_label(reordered) == deltanet_kernel_label(FUSED_BINDING)
        assert deltanet_kernel_label(None) == "null"
        assert deltanet_kernel_label(None) != deltanet_kernel_label(FUSED_BINDING)


class TestDecodeCallSite:
    def test_the_call_site_is_read_out_of_the_transformers_source(self):
        call_site = decode_call_site()
        assert call_site.positional_count == 3
        assert {
            "g",
            "beta",
            "initial_state",
            "output_final_state",
            "use_qk_l2norm_in_kernel",
            "cu_seqlens",
        } <= call_site.keyword_names

    def test_reading_the_call_site_does_not_import_the_modeling_module(self):
        """Importing that module is what makes the bridge too late.

        So the check that guards the bridge must not be the thing that breaks it.
        """
        snippet = (
            "import sys\n"
            "from games.deltanet_kernels import QWEN3_5_MODELING_MODULE, decode_call_site\n"
            "decode_call_site()\n"
            "print('IMPORTED', QWEN3_5_MODELING_MODULE in sys.modules)\n"
        )
        result = run_in_fresh_interpreter(snippet)
        assert result.returncode == 0, result.stderr
        assert "IMPORTED False" in result.stdout, result.stdout


class TestDecodeShape:
    def test_the_kernel_sees_the_value_head_count(self):
        config = HeadGeometry(
            linear_num_key_heads=8,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=64,
        )
        shape = DecodeShape.from_text_config(config, batch_size=3)
        assert shape == DecodeShape(batch_size=3, num_heads=16, key_head_dim=128, value_head_dim=64)

    def test_head_counts_that_cannot_repeat_interleave_are_rejected(self):
        config = HeadGeometry(
            linear_num_key_heads=5,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        )
        with pytest.raises(RuntimeError, match="multiple of key heads"):
            DecodeShape.from_text_config(config, batch_size=1)


class TestDecodeInputs:
    @pytest.fixture
    def inputs(self) -> DecodeInputs:
        return build_decode_inputs(
            DecodeShape(batch_size=2, num_heads=4, key_head_dim=32, value_head_dim=32),
            dtype=torch.float32,
            device="cpu",
            seed=0,
        )

    def test_the_decay_is_log_space_and_therefore_negative(self, inputs: DecodeInputs):
        """A positive g would make exp(g) a growth factor and the recurrence explode.

        That would swamp the comparison with magnitudes no real run produces.
        """
        assert inputs.g.max().item() < 0.0

    def test_beta_is_post_sigmoid(self, inputs: DecodeInputs):
        assert inputs.beta.min().item() > 0.0
        assert inputs.beta.max().item() < 1.0

    def test_the_recurrent_state_stays_fp32_even_for_bf16_activations(self):
        inputs = build_decode_inputs(
            DecodeShape(batch_size=1, num_heads=2, key_head_dim=16, value_head_dim=16),
            dtype=torch.bfloat16,
            device="cpu",
            seed=1,
        )
        assert inputs.query.dtype == torch.bfloat16
        assert inputs.initial_state.dtype == torch.float32

    def test_the_same_seed_gives_the_same_values_across_dtypes(self):
        shape = DecodeShape(batch_size=1, num_heads=2, key_head_dim=16, value_head_dim=16)
        as_fp32 = build_decode_inputs(shape, dtype=torch.float32, device="cpu", seed=7)
        as_bf16 = build_decode_inputs(shape, dtype=torch.bfloat16, device="cpu", seed=7)
        assert torch.equal(as_bf16.query, as_fp32.query.to(torch.bfloat16))


class TestDecodeComparison:
    """The comparison harness itself, exercised on CPU with the pure-torch loop on both sides.

    The GPU half of this -- the real triton kernel against the same reference -- is
    `python -m games.deltanet_kernels`, deliberately not a test: `make test` is CPU-only by design,
    and a cuda-gated test here would be permanently skipped everywhere except a free GPU box while
    contending for the single shared L4 when it did run.
    """

    @pytest.fixture
    def reference(self) -> DecodeKernel:
        return torch_decode_reference()

    @pytest.fixture
    def inputs(self) -> DecodeInputs:
        return build_decode_inputs(
            DecodeShape(batch_size=2, num_heads=4, key_head_dim=32, value_head_dim=32),
            dtype=torch.float32,
            device="cpu",
            seed=0,
        )

    def test_the_reference_is_reachable_through_the_wrapper(self, reference: DecodeKernel):
        assert reference.__name__ == "torch_recurrent_gated_delta_rule"
        assert getattr(reference, "__wrapped__", None) is None

    def test_an_implementation_agrees_with_itself_exactly(
        self, reference: DecodeKernel, inputs: DecodeInputs
    ):
        agreement = compare_decode_kernels(reference, reference, inputs)
        assert agreement.output.max_abs_diff == 0.0
        assert agreement.recurrent_state.max_abs_diff == 0.0
        assert agreement.output.reference_max_abs > 0.0
        assert agreement.within_tolerance
        assert agreement.tolerance == DECODE_RELATIVE_TOLERANCES[torch.float32]

    def test_a_kernel_that_skips_the_l2_norm_is_caught(
        self, reference: DecodeKernel, inputs: DecodeInputs
    ):
        with pytest.raises(RuntimeError, match="disagrees with the pure-torch reference"):
            assert_decode_kernels_match(without_l2norm(reference), reference, inputs)

    def test_a_kernel_that_drops_the_initial_state_is_caught(
        self, reference: DecodeKernel, inputs: DecodeInputs
    ):
        with pytest.raises(RuntimeError, match="disagrees with the pure-torch reference"):
            assert_decode_kernels_match(without_initial_state(reference), reference, inputs)

    def test_a_dropped_initial_state_shows_up_in_the_state_as_well_as_the_output(
        self, reference: DecodeKernel, inputs: DecodeInputs
    ):
        agreement = compare_decode_kernels(without_initial_state(reference), reference, inputs)
        assert agreement.output.max_relative_diff > agreement.tolerance
        assert agreement.recurrent_state.max_relative_diff > agreement.tolerance

    def test_an_implementation_returning_no_final_state_is_a_hard_error(
        self, reference: DecodeKernel, inputs: DecodeInputs
    ):
        def no_final_state(*args: object, **kwargs: object) -> tuple[torch.Tensor, None]:
            output, _ = reference(*args, **{**kwargs, "output_final_state": False})
            return output, None

        with pytest.raises(RuntimeError, match="no final recurrent state"):
            compare_decode_kernels(no_final_state, reference, inputs)

    def test_a_dtype_with_no_stated_tolerance_is_refused(self, reference: DecodeKernel):
        inputs = build_decode_inputs(
            DecodeShape(batch_size=1, num_heads=2, key_head_dim=16, value_head_dim=16),
            dtype=torch.float16,
            device="cpu",
            seed=0,
        )
        with pytest.raises(RuntimeError, match="no decode tolerance is set"):
            compare_decode_kernels(reference, reference, inputs)

    def test_bfloat16_carries_a_looser_tolerance_than_float32(self):
        """The asymmetry is real and directional.

        The torch fallback rounds the L2-normalised query and key to bf16 before upcasting, so it is
        the LESS accurate of the two.
        """
        assert (
            DECODE_RELATIVE_TOLERANCES[torch.bfloat16] > DECODE_RELATIVE_TOLERANCES[torch.float32]
        )
        assert DECODE_RELATIVE_TOLERANCES[torch.bfloat16] < 1.0

    def test_the_relative_measure_survives_an_all_zero_reference(self, inputs: DecodeInputs):
        def zeros(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            output, state = torch_decode_reference()(*args, **kwargs)
            assert state is not None
            return torch.zeros_like(output), torch.zeros_like(state)

        agreement = compare_decode_kernels(zeros, zeros, inputs)
        assert agreement.output.max_relative_diff == 0.0
        assert not math.isnan(agreement.output.max_relative_diff)
