"""Exact replay workload extraction and CPU-only memory roofline checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.replay_workload import (
    extract_episode_workloads,
    extract_workloads,
    roofline_tokens_per_second,
    synthetic_workloads,
)
from reward_hacking.trace import write_trace

if TYPE_CHECKING:
    from pathlib import Path


def _write_trace(path: Path, prompts: list[list[int]], seed: int) -> dict[str, Any]:
    sampling: dict[str, Any] = {
        "max_new_tokens": 2048,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": 20,
        "presence_penalty": 1.5,
        "thinking_token_budget": 2048,
        "seed": seed,
    }
    rows: list[dict[str, Any]] = [
        {"record": "organism_episode_identity", "sampling": sampling},
        *[
            {
                "record": "turn",
                "turn": turn_index,
                "prompt_token_ids": prompt_ids,
                "prompt_tokens": len(prompt_ids),
            }
            for turn_index, prompt_ids in enumerate(prompts)
        ],
    ]
    write_trace(path, rows, compact=True)
    return sampling


def _write_episode_trace(path: Path, prompts: list[list[int]], seed: int) -> dict[str, Any]:
    sampling: dict[str, Any] = {"temperature": 0.6, "seed": seed}
    rows: list[dict[str, Any]] = [
        {"record": "organism_episode_identity", "sampling": sampling},
        *[
            {
                "record": "turn",
                "turn": turn_index,
                "prompt_token_ids": prompt_ids,
                "prompt_tokens": len(prompt_ids),
            }
            for turn_index, prompt_ids in enumerate(prompts)
        ],
    ]
    write_trace(path, rows, compact=True)
    return sampling


def test_extract_workloads_restores_exact_prompt_ids_from_compact_trace(tmp_path: Path) -> None:
    trace_root = tmp_path / "seed-v4"
    first_trace_path = trace_root / "suite" / "run-one" / "27b" / "traces" / "trace.jsonl"
    second_trace_path = trace_root / "suite" / "run-two" / "27b" / "traces" / "trace.jsonl"
    first_prefix = list(range(1, 51))
    second_prefix = list(range(100, 220))
    prompts = [
        [*first_prefix, *([101] * 13)],
        [*first_prefix, *([102] * 14)],
        [*second_prefix, *([201] * 7)],
        [*second_prefix, *([202] * 8)],
    ]
    first_sampling = _write_trace(first_trace_path, prompts[:2], seed=19)
    second_sampling = _write_trace(second_trace_path, prompts[2:], seed=20)
    docker_evidence_path = first_trace_path.with_name("trace.docker-evidence.jsonl")
    docker_evidence_path.write_text('{"record":"docker_evidence"}\n', encoding="utf-8")
    decoy_path = trace_root / "suite" / "run" / "27b" / "records.jsonl"
    decoy_path.parent.mkdir(parents=True, exist_ok=True)
    decoy_path.write_text('{"record":"derived_record"}\n', encoding="utf-8")

    assert first_trace_path.read_bytes().startswith(b"\x1f\x8b")
    assert second_trace_path.read_bytes().startswith(b"\x1f\x8b")
    workloads = extract_workloads(
        trace_root,
        lengths=(64, 128),
        requests_per_cell=2,
        tolerance=0.25,
    )

    assert [workload.target_length for workload in workloads] == [64, 64, 128, 128]
    assert [workload.prompt_ids for workload in workloads] == [
        prompts[1],
        prompts[0],
        prompts[3],
        prompts[2],
    ]
    assert all(workload.kind == "trace" for workload in workloads)
    assert [workload.sampling for workload in workloads] == [
        first_sampling,
        first_sampling,
        second_sampling,
        second_sampling,
    ]
    assert all("rendered_prompt_text" not in workload.sampling for workload in workloads)


def test_extract_workloads_refuses_a_cell_with_too_few_distinct_prompts(tmp_path: Path) -> None:
    trace_root = tmp_path / "seed-v4"
    _write_trace(
        trace_root / "suite" / "run" / "27b" / "traces" / "trace.jsonl", [[10] * 8], seed=3
    )

    with pytest.raises(
        ValueError,
        match=r"target length 8: found 1 distinct prompts within ±25%, need 2",
    ):
        extract_workloads(trace_root, lengths=(8,), requests_per_cell=2, tolerance=0.25)


def test_extract_workloads_discovers_9b_trace_layout(tmp_path: Path) -> None:
    trace_root = tmp_path / "base-9b"
    _write_trace(
        trace_root / "r2e-leak" / "traces" / "step-000000" / "episode.jsonl",
        [[7] * 4096, [7] * 8192],
        seed=3,
    )

    workloads = extract_workloads(
        trace_root, lengths=(4096, 8192), requests_per_cell=1, tolerance=0.0
    )

    assert [workload.target_length for workload in workloads] == [4096, 8192]
    assert [len(workload.prompt_ids) for workload in workloads] == [4096, 8192]


def test_extract_episode_workloads_keeps_real_turns_and_reports_prefix_lcp(tmp_path: Path) -> None:
    trace_root = tmp_path / "seed-v4"
    divergent_prompts = [
        [7] * 3072,
        [7] * 3000 + [8] * 1096,
        [7] * 3000 + [8] * 1096 + [9] * (65536 - 4096),
    ]
    exact_prompts = [
        [3] * 4096,
        [3] * 8192,
        [3] * 16384,
        [3] * 65536,
    ]
    _write_episode_trace(
        trace_root / "suite" / "divergent" / "27b" / "traces" / "divergent.jsonl",
        divergent_prompts,
        seed=11,
    )
    _write_episode_trace(
        trace_root / "suite" / "exact" / "27b" / "traces" / "exact.jsonl",
        exact_prompts,
        seed=12,
    )
    _write_episode_trace(
        trace_root / "suite" / "short" / "27b" / "traces" / "short.jsonl",
        [[4] * 2048, [4] * 4096, [4] * 32768],
        seed=13,
    )

    workloads = extract_episode_workloads(trace_root, copies=2)

    assert [workload.episode_id for workload in workloads] == ["divergent", "exact"]
    divergent = workloads[0]
    assert [turn.prompt_tokens for turn in divergent.turns] == [3072, 4096, 65536]
    assert [turn.prompt_ids for turn in divergent.turns] == divergent_prompts
    assert [turn.lcp_with_previous_tokens for turn in divergent.turns] == [0, 3000, 4096]
    assert [turn.prefix_extends_previous for turn in divergent.turns] == [False, False, True]
    assert [turn.turn_index for turn in divergent.turns] == [0, 1, 2]
    assert all(turn.sampling == {"temperature": 0.6, "seed": 11} for turn in divergent.turns)
    assert divergent.turn_4k_index == 1
    assert divergent.turn_end_index == 2
    assert divergent.segment_turn_count == 2
    assert divergent.segment_prefill_tokens == 69632
    assert [turn.prompt_tokens for turn in workloads[1].turns] == [4096, 8192, 16384, 65536]


def test_extract_episode_workloads_returns_up_to_available_independent_chains(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "seed-v4"
    for episode_index, prompt_lengths in enumerate(
        ([4096, 65536], [4096, 8192, 65536], [4096, 8192, 16384, 65536])
    ):
        prompts = [[episode_index + 1] * length for length in prompt_lengths]
        _write_episode_trace(
            trace_root
            / "suite"
            / f"episode-{episode_index}"
            / "27b"
            / "traces"
            / f"episode-{episode_index}.jsonl",
            prompts,
            seed=episode_index,
        )

    workloads = extract_episode_workloads(trace_root, copies=6)

    assert len(workloads) == 3
    assert len({workload.episode_id for workload in workloads}) == 3


def test_extract_episode_workloads_supports_capped_9b_prefix_growth_layout(tmp_path: Path) -> None:
    trace_root = tmp_path / "base-9b"
    prompts = [
        [5] * 4096,
        [5] * 8192,
        [5] * 16384,
        [5] * 35000,
    ]
    _write_episode_trace(
        trace_root / "r2e-leak" / "traces" / "step-000000" / "episode.jsonl",
        prompts,
        seed=5,
    )

    workloads = extract_episode_workloads(
        trace_root,
        copies=1,
        end_length=35000,
        max_context_length=40960,
        require_prefix_growth=True,
    )

    assert len(workloads) == 1
    assert workloads[0].turn_4k_index == 0
    assert workloads[0].turn_end_index == 3
    assert workloads[0].prompt_tokens_at_end == 35000
    assert workloads[0].turns_between_start_and_end == 3
    assert all(turn.prefix_extends_previous for turn in workloads[0].turns[1:])


def test_extract_episode_workloads_ranks_by_actual_segment_prefill(tmp_path: Path) -> None:
    trace_root = tmp_path / "seed-v4"
    _write_episode_trace(
        trace_root / "suite" / "short-span" / "27b" / "traces" / "short-span.jsonl",
        [[1] * 32768, [1] * 65536],
        seed=1,
    )
    _write_episode_trace(
        trace_root / "suite" / "low-prefill" / "27b" / "traces" / "low-prefill.jsonl",
        [[2] * 4096, [2] * 8192, [2] * 65536],
        seed=2,
    )

    workload = extract_episode_workloads(trace_root, copies=1)[0]

    assert workload.episode_id == "low-prefill"
    assert workload.turns_between_start_and_end == 2
    assert workload.segment_turn_count == 3
    assert workload.segment_prefill_tokens == 77824


def test_synthetic_workloads_have_exact_distinct_token_ids_and_preserve_sampling() -> None:
    sampling: dict[str, Any] = {"temperature": 0.6, "top_k": 20, "seed": 42}

    workloads = synthetic_workloads(
        17,
        lengths=(8, 12),
        requests_per_cell=3,
        sampling=sampling,
        special_token_ids=(0, 1),
    )

    for length in (8, 12):
        cell = [workload for workload in workloads if workload.target_length == length]
        assert len(cell) == 3
        assert all(len(workload.prompt_ids) == length for workload in cell)
        assert len({tuple(workload.prompt_ids) for workload in cell}) == 3
        assert all(set(workload.prompt_ids).isdisjoint({0, 1}) for workload in cell)
        assert all(workload.sampling == sampling for workload in cell)
        assert all(workload.kind == "synthetic" for workload in cell)


def test_synthetic_workloads_fail_when_vocabulary_cannot_distinguish_prefixes() -> None:
    with pytest.raises(ValueError, match="at least 3 usable token IDs"):
        synthetic_workloads(
            3,
            lengths=(4,),
            requests_per_cell=3,
            sampling={},
            special_token_ids=(0,),
        )


def test_roofline_counts_full_attention_kv_traffic_and_uses_nested_text_config() -> None:
    config: dict[str, Any] = {
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 32,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ],
        },
    }

    observed = roofline_tokens_per_second(
        config,
        context_tokens=10,
        concurrency=2,
        kv_bytes=1,
        weight_bytes=100,
        bandwidth_bytes_s=1000,
    )

    # Two full-attention layers use 2 * 2 KV heads * 4 head width * 1 byte per token.
    assert observed == pytest.approx(2000 / (100 + (2 * 10 * 2 * 2 * 4 * 2)))


def test_roofline_uses_full_attention_interval_when_layer_types_are_missing() -> None:
    config: dict[str, Any] = {
        "head_dim": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 8,
        "full_attention_interval": 4,
    }

    observed = roofline_tokens_per_second(
        config,
        context_tokens=10,
        concurrency=2,
        kv_bytes=1,
        weight_bytes=100,
        bandwidth_bytes_s=1000,
    )

    assert observed == pytest.approx(2000 / (100 + (2 * 10 * 2 * 2 * 4 * 2)))


def test_roofline_refuses_unknown_layer_types() -> None:
    config: dict[str, Any] = {
        "head_dim": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "layer_types": ["unknown_attention"],
    }

    with pytest.raises(ValueError, match="unsupported model config layer types"):
        roofline_tokens_per_second(config, context_tokens=10, concurrency=1)


@pytest.mark.parametrize(
    ("context_tokens", "concurrency"),
    [(0, 1), (1, 0)],
)
def test_roofline_requires_positive_context_and_concurrency(
    context_tokens: int, concurrency: int
) -> None:
    config: dict[str, Any] = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 1,
    }

    with pytest.raises(ValueError, match="must be a positive integer"):
        roofline_tokens_per_second(config, context_tokens, concurrency)
