import json
from pathlib import Path
from typing import Self

import pytest

from reward_hacking.organism_rl import cli
from reward_hacking.organism_rl.cli import parse_args, resolve_sizing, write_provenance
from reward_hacking.vllm_server import VLLMServerConfig


def test_defaults_and_vram_derived_sizing():
    args = parse_args(["--output-dir", "artifacts/toy", "--gpu-memory-utilization", "0.8"])
    assert args.temperature == args.top_p == 1.0
    small = resolve_sizing(total_bytes=8 * 1024**3, hidden_size=1024, layers=12)
    large = resolve_sizing(total_bytes=16 * 1024**3, hidden_size=1024, layers=12)
    assert large.max_sequence_tokens > small.max_sequence_tokens
    assert small.micro_batch_size == 1


def test_teacher_requires_stored_scores():
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "artifacts/toy", "--teacher-beta", "0.1"])


def test_owned_server_requires_an_explicit_gpu_memory_fraction() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "artifacts/toy"])


def test_provenance_has_flags_version_and_sha(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--gpu-memory-utilization", "0.8"])
    path = write_provenance(args, {"name": "cpu"}, model_revision="resolved-revision")
    value = json.loads(path.read_text())
    assert value["trl_version"] == "1.10.0"
    assert len(value["git_sha"]) == 40
    assert isinstance(value["git_dirty"], bool)
    assert isinstance(value["modified_tracked_paths"], list)
    assert value["model_revision"] == "resolved-revision"
    assert value["flags"]["temperature"] == 1.0


def test_owned_server_options_build_the_requested_vllm_configuration(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--model",
            "Qwen/test-model",
            "--model-revision",
            "resolved-revision",
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.75",
            "--max-lora-rank",
            "32",
            "--enable-prefix-caching",
            "--mamba-cache-mode",
            "align",
            "--reasoning-config",
            '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}',
        ]
    )

    assert args.endpoint is None
    config = cli.build_server_config(args, model_revision="resolved-revision", max_model_len=4096)

    assert config.model == "Qwen/test-model"
    assert config.revision == "resolved-revision"
    assert config.max_model_len == 4096
    assert config.gpu_memory_utilization == 0.75
    assert config.enable_lora is True
    assert config.max_lora_rank == 32
    assert config.enable_sleep_mode is True
    assert config.enable_prefix_caching is True
    assert config.mamba_cache_mode == "align"
    assert config.reasoning_config == {
        "reasoning_start_str": "<think>",
        "reasoning_end_str": "</think>",
    }
    assert Path(config.executable).name == "vllm"
    assert config.log_path == tmp_path / "vllm-server.log"


def test_server_endpoint_owns_the_server_for_the_training_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--gpu-memory-utilization",
            "0.75",
            "--model-revision",
            "resolved-revision",
        ]
    )
    lifecycle: list[str] = []
    configs: list[VLLMServerConfig] = []

    class FakeServer:
        base_url = "http://owned-server:8000"

        def __init__(self, config: VLLMServerConfig) -> None:
            configs.append(config)

        def __enter__(self) -> Self:
            lifecycle.append("start")
            return self

        def __exit__(self, *args: object) -> None:
            del args
            lifecycle.append("stop")

    monkeypatch.setattr(cli, "VLLMServer", FakeServer)

    with cli.server_endpoint(args, "resolved-revision", 2048) as endpoint:
        assert endpoint == "http://owned-server:8000"
        assert lifecycle == ["start"]

    assert lifecycle == ["start", "stop"]
    assert configs[0].log_path == tmp_path / "vllm-server.log"


def test_endpoint_option_keeps_external_server_mode(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--endpoint", "http://server:8000"])

    assert args.endpoint == "http://server:8000"
    with cli.server_endpoint(args, None, 2048) as endpoint:
        assert endpoint == "http://server:8000"


def test_resume_preserves_all_flags_and_refuses_sampling_drift(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--gpu-memory-utilization", "0.8"])
    write_provenance(args, {"name": "cpu"})
    args.resume_from_checkpoint = tmp_path / "checkpoint-1"
    path = write_provenance(args, {"name": "cpu"})
    assert json.loads(path.read_text())["flags"]["resume_from_checkpoint"] == str(
        args.resume_from_checkpoint
    )
    args.temperature = 0.5
    with pytest.raises(ValueError, match="flags"):
        write_provenance(args, {"name": "cpu"})


def test_tracked_output_directory_is_refused() -> None:
    args = parse_args(
        ["--output-dir", "reward_hacking/organism_rl", "--gpu-memory-utilization", "0.8"]
    )
    with pytest.raises(ValueError, match="gitignored"):
        write_provenance(args, {"name": "cpu"})


def test_teacher_score_content_drift_is_refused(tmp_path: Path) -> None:
    teacher = tmp_path / "scores.jsonl"
    teacher.write_text('{"teacher_logps": [-1.0]}\n')
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--gpu-memory-utilization",
            "0.8",
            "--teacher-beta",
            "0.1",
            "--teacher-path",
            str(teacher),
        ]
    )
    write_provenance(args, {"name": "cpu"})
    teacher.write_text('{"teacher_logps": [-2.0]}\n')
    with pytest.raises(ValueError, match="teacher scores"):
        write_provenance(args, {"name": "cpu"})
