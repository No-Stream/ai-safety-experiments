"""CPU checks for the bounded CUDA-graph residual-intervention validator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from scripts.validate_vllm_intervention_graphs import (
    DEFAULT_CAPTURE_SIZES,
    DEFAULT_TOP_K,
    GRAPH_ADDITIONAL_CONFIG,
    GRAPH_WORKER_CLASS,
    MAX_PROMPT_COUNT,
    GenerationAggregate,
    _assert_decode_replay_effect,
    _assert_prompt_replay_effect,
    _basis,
    _engine_kwargs,
    _geometry,
    _load_prompts,
    _prompt_digest,
    _topk_metrics,
)


def _topk(values: list[float]) -> dict[int, float]:
    return dict(enumerate(values[:DEFAULT_TOP_K]))


def _generation(digest: str, mean_decode_logprob: float) -> GenerationAggregate:
    return GenerationAggregate(2, 4, 2, 2, digest, 2, mean_decode_logprob, 0.1)


class TestPromptInput:
    def test_requires_exact_runtime_row_count_and_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.jsonl"
        path.write_text(
            json.dumps("synthetic prompt") + "\n" + json.dumps({"prompt": "second"}) + "\n"
        )

        prompts = _load_prompts(path, expected_count=2)

        assert prompts == ("synthetic prompt", "second")
        assert len(_prompt_digest(prompts)) == 64
        with pytest.raises(ValueError, match="expected exactly 1"):
            _load_prompts(path, expected_count=1)

    def test_refuses_unbounded_or_invalid_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"text": "wrong field"}) + "\n")

        with pytest.raises(ValueError, match=f"1..{MAX_PROMPT_COUNT}"):
            _load_prompts(path, expected_count=MAX_PROMPT_COUNT + 1)
        with pytest.raises(TypeError, match="string prompt"):
            _load_prompts(path, expected_count=1)


class TestGeometryAndConfiguration:
    def test_basis_is_orthonormal_and_seed_changes_placebo(self) -> None:
        real = _basis(12, 11)
        placebo = _basis(12, 12)

        assert real.shape == (1, 12)
        assert torch.allclose(real @ real.T, torch.ones((1, 1)), atol=1e-5, rtol=1e-5)
        assert not torch.equal(real, placebo)

    def test_geometry_contains_matched_placebo_and_wrong_layer(self) -> None:
        specs = _geometry(4, 12)

        assert specs["real"].by_layer.keys() == specs["placebo"].by_layer.keys() == {0}
        assert specs["wrong_layer"].by_layer.keys() == {1}
        assert specs["steering"].alpha == 20.0

    def test_graph_engine_has_expected_worker_and_bounded_capture(self) -> None:
        kwargs = _engine_kwargs(
            enforce_eager=False, max_model_len=128, max_num_seqs=4, max_num_batched_tokens=512
        )

        assert kwargs["enforce_eager"] is False
        assert kwargs["enable_prefix_caching"] is False
        assert kwargs["worker_cls"] == GRAPH_WORKER_CLASS
        assert cast("str", kwargs["worker_extension_cls"]).endswith("ResidualInterventionWorker")
        assert kwargs["additional_config"] == GRAPH_ADDITIONAL_CONFIG
        assert kwargs["cudagraph_capture_sizes"] == list(DEFAULT_CAPTURE_SIZES)

    def test_eager_engine_does_not_select_graph_worker(self) -> None:
        kwargs = _engine_kwargs(
            enforce_eager=True, max_model_len=128, max_num_seqs=4, max_num_batched_tokens=512
        )

        assert kwargs["enforce_eager"] is True
        assert "worker_cls" not in kwargs
        assert "additional_config" not in kwargs
        assert "cudagraph_capture_sizes" not in kwargs


class TestReplayGuards:
    def test_identical_topk_rows_have_zero_deltas(self) -> None:
        rows = tuple(
            _topk([float(DEFAULT_TOP_K - index) for index in range(DEFAULT_TOP_K)])
            for _ in range(2)
        )

        metrics = _topk_metrics(rows, rows)

        assert metrics["top1_agreement"] == 1.0
        assert metrics["overlap_at_k"] == 1.0
        assert metrics["max_abs_logprob_delta"] == 0.0

    def test_prompt_sabotage_guard_accepts_changed_rows_and_rejects_noop(self) -> None:
        baseline = (_topk([float(DEFAULT_TOP_K - index) for index in range(DEFAULT_TOP_K)]),)
        changed = (_topk([float(index) for index in range(DEFAULT_TOP_K)]),)

        assert _assert_prompt_replay_effect(baseline, changed)["max_abs_logprob_delta"] > 0.0
        with pytest.raises(RuntimeError, match="intervention was skipped"):
            _assert_prompt_replay_effect(baseline, baseline)

    def test_prompt_guard_counts_disjoint_rows_as_changed(self) -> None:
        baseline = (_topk([float(DEFAULT_TOP_K - index) for index in range(DEFAULT_TOP_K)]),)
        disjoint = ({DEFAULT_TOP_K + index: float(index) for index in range(DEFAULT_TOP_K)},)

        metrics = _assert_prompt_replay_effect(baseline, disjoint)

        assert metrics["overlap_at_k"] == 0.0
        assert metrics["top1_agreement"] == 0.0

    def test_decode_sabotage_guard_uses_after_first_token_signal(self) -> None:
        baseline = _generation("same", -0.1)
        changed = _generation("different", -0.1)
        unchanged = _generation("same", -0.1)

        assert _assert_decode_replay_effect(baseline, changed) == 0.0
        with pytest.raises(RuntimeError, match="intervention was skipped"):
            _assert_decode_replay_effect(baseline, unchanged)

    def test_decode_guard_accepts_edit_that_ends_generation_at_first_token(self) -> None:
        baseline = _generation("long", -0.1)
        truncated = GenerationAggregate(2, 2, 1, 1, "short", 0, None, 0.1)
        truncated_same_digest = GenerationAggregate(2, 2, 1, 1, "long", 0, None, 0.1)

        assert _assert_decode_replay_effect(baseline, truncated) is None
        with pytest.raises(RuntimeError, match="intervention was skipped"):
            _assert_decode_replay_effect(baseline, truncated_same_digest)
