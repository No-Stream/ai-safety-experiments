"""CPU checks for replay configuration and semantic resume validation."""

from pathlib import Path

import pytest

from reward_hacking.replay_benchmark import (
    ReplayServerConfig,
    cell_payloads,
    configurations,
    measurement_directory,
    validate_resume,
)


def test_one_change_configurations(tmp_path: Path) -> None:
    baseline = ReplayServerConfig(
        model=str(tmp_path), revision=None, max_model_len=98304, gpu_memory_utilization=0.9
    )
    variants = configurations(baseline, quota_threads=3)
    assert variants[0].name == "baseline"
    assert "--no-async-scheduling" not in variants[0].server.command()
    assert "--async-scheduling" not in variants[0].server.command()
    assert "--async-scheduling" in variants[2].server.command()
    for variant in variants[1:]:
        assert len(variant.changes) == 1
    assert "--no-enable-prefix-caching" in next(
        v.server.command() for v in variants if v.name == "prefix-off"
    )


def test_resume_rejects_changed_tokens_without_hashes() -> None:
    identity = {"payloads": [{"prompt": [1, 2], "max_tokens": 64}], "settings": {"async": False}}
    validate_resume({"identity": identity}, identity)
    with pytest.raises(ValueError, match="changed"):
        validate_resume({"identity": identity}, {**identity, "payloads": [{"prompt": [1, 3]}]})


def test_payload_preserves_sampling_and_exact_ids(tmp_path: Path) -> None:
    server = ReplayServerConfig(
        model=str(tmp_path), revision=None, max_model_len=98304, gpu_memory_utilization=0.9
    )
    variants = configurations(server, quota_threads=3)
    sampling = {
        "max_new_tokens": 8192,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 1.0,
        "presence_penalty": 1.5,
        "thinking_token_budget": 2048,
        "seed": 17,
    }
    workloads = [
        {"kind": "trace", "target_length": 4, "prompt_ids": [3, 7, 11, 17], "sampling": sampling}
    ]
    payload = cell_payloads(workloads, variants[0], "trace", 4, 1, 64)[0]
    assert payload == {
        **{k: v for k, v in sampling.items() if k != "max_new_tokens"},
        "max_tokens": 64,
        "prompt": [3, 7, 11, 17],
    }
    assert (
        "thinking_token_budget" not in cell_payloads(workloads, variants[1], "trace", 4, 1, 64)[0]
    )
    assert sampling["max_new_tokens"] == 8192


def test_interrupted_measurements_use_fresh_directory(tmp_path: Path) -> None:
    first = measurement_directory(tmp_path)
    evidence = first / "request_results.jsonl"
    evidence.write_text("interrupted evidence\n")
    second = measurement_directory(tmp_path)
    assert first != second
    assert evidence.read_text() == "interrupted evidence\n"
    assert second.is_dir()
