"""CPU contracts for the unhinted INT4 wave generator."""

import argparse
import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Self

import pytest

from reward_hacking.organism_rl import generate, probe_v4, wave_v4
from reward_hacking.organism_rl.yield_table import PROBE_CELLS
from reward_hacking.vllm_server import VLLMServerConfig

MODEL = "RedHatAI/Qwen3.8-27B-INT4"


def _args(*extra: str) -> argparse.Namespace:
    return wave_v4.parse_args(
        [
            "--cell",
            "f2f7-conflicting",
            "--generator",
            "27b",
            "--hint-strength",
            "none",
            "--episodes",
            "8",
            "--max-turns",
            "40",
            *extra,
        ]
    )


def test_27b_defaults_and_refusals() -> None:
    args = _args()
    assert args.model == MODEL
    assert args.hint_strength == "none"
    assert args.max_model_len == 131072
    for extra in [
        ("--hint-strength", "vague"),
        ("--endpoint", "http://localhost:8000"),
        ("--smoke",),
    ]:
        with pytest.raises(SystemExit):
            _args(*extra)


def test_27b_server_and_generation_contract(tmp_path: Path) -> None:
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": PROBE_CELLS[1].hint_family,
                        "strength": "vague",
                        "text": "synthetic fixture",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = _args("--output-root", str(tmp_path), "--hints", str(hints))
    config = probe_v4._server_config(  # pyright: ignore[reportPrivateUsage]
        args,
        generator="27b",
        model_revision="a" * 40,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.87,
    )
    command = config.command()
    for flag, value in {
        "--kv-cache-dtype": "fp8",
        "--attention-backend": "FLASHINFER",
        "--gpu-memory-utilization": "0.87",
        "--dtype": "bfloat16",
        "--mamba-cache-mode": "align",
        "--max-num-batched-tokens": "8192",
        "--max-num-seqs": "16",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert "--enable-lora" not in command
    assert "--speculative-config" not in command
    settings = probe_v4._cell_run_config(  # pyright: ignore[reportPrivateUsage]
        args,
        "27b",
        "http://localhost:8000",
        "a" * 40,
        args.max_model_len,
    )
    settings = wave_v4.stage_settings(args, settings, ())
    command = probe_v4.build_generate_command(PROBE_CELLS[1], settings)
    assert command[command.index("--chat-template-kwarg") + 1] == f"{MODEL}:reasoning_effort=low"
    assert command[command.index("--generator-name") + 1] == "27b"
    parsed = generate.parse_args(command[3:])
    assert parsed.generator_name == "27b"
    assert parsed.chat_template_kwargs == {"reasoning_effort": "low"}
    assert parsed.hint_strength == "none"
    assert parsed.thinking_token_budget == 2048
    serving = json.loads(command[command.index("--serving-config") + 1])
    assert serving["kv_cache_dtype"] == "fp8"
    assert serving["attention_backend"] == "FLASHINFER"
    assert "--served-adapter-sha256" not in command
    for flag, value in {
        "--temperature": "0.6",
        "--top-p": "1.0",
        "--top-k": "20",
        "--thinking-token-budget": "2048",
        "--hint-strength": "none",
    }.items():
        assert command[command.index(flag) + 1] == value
    health = probe_v4.build_health_command(
        tmp_path / "records.jsonl", tmp_path / "baseline.jsonl", generator="27b"
    )
    assert "--assert" not in health


def test_kv_concurrency_uses_runtime_pool_and_refuses_missing_or_short_pool() -> None:
    assert probe_v4.kv_pool_concurrency("GPU KV cache size: 271,701 tokens", 131072) == 2
    assert probe_v4.kv_pool_concurrency("GPU KV cache size: 1,048,576 tokens", 131072) == 6
    for log in ["no capacity reported", "GPU KV cache size: 1,024 tokens"]:
        with pytest.raises(ValueError, match="27b"):
            probe_v4.kv_pool_concurrency(log, 131072)


def test_27b_dry_run_has_serving_and_sampling_without_model_access(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("dry run accessed runtime assets or started serving")

    monkeypatch.setattr(wave_v4, "run_generator", forbidden)
    assert (
        wave_v4.main(
            [
                "--cell",
                "f3-leak",
                "--generator",
                "27b",
                "--hint-strength",
                "none",
                "--episodes",
                "8",
                "--max-turns",
                "90",
                "--leak-assets",
                str(tmp_path / "assets.json"),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    for value in [
        MODEL,
        "fp8",
        "FLASHINFER",
        "reasoning_effort=low",
        "2048",
        "0.6",
        "20",
        "1.0",
        "KV pool",
    ]:
        assert value in output


def test_27b_exclusions_include_all_generators_but_own_resume(tmp_path: Path) -> None:
    args = _args(
        "--output-root", str(tmp_path / "wave"), "--probe-output-dir", str(tmp_path / "probe")
    )
    own = wave_v4.stage_directory(args) / "records.jsonl"
    prior = [tmp_path / "probe" / args.cell / gen / "records.jsonl" for gen in ("base", "v3")]
    prior += [own.parent.parent / gen / "records.jsonl" for gen in ("base", "v3")]
    prior += [tmp_path / "wave" / f"{args.cell}-none-t60" / "27b" / "records.jsonl"]
    for path in [own, *prior]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    wave_v4.validate_probe_records(args)
    assert set(
        wave_v4.exclusion_files(args.probe_output_dir, args.output_root, args.cell, own)
    ) == set(prior)


@pytest.mark.parametrize(
    "current_log",
    ["GPU KV cache size: 271,701 tokens", "GPU KV cache size: 1,024 tokens", "no current capacity"],
)
def test_27b_lifecycle_uses_only_current_startup_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_log: str,
) -> None:
    args = _args("--output-root", str(tmp_path))
    log = tmp_path / "vllm-27b-server.log"
    log.write_text("GPU KV cache size: 1,048,576 tokens\n", encoding="utf-8")
    monkeypatch.setattr(probe_v4, "_runtime_model_revision", lambda *args: "a" * 40)
    monkeypatch.setattr(
        probe_v4, "resolve_gpu_engine_settings", lambda torch: (0.87, "CPU fixture")
    )
    monkeypatch.setattr(probe_v4, "_served_model_max_len", lambda *args: 131072)

    class FakeServer(AbstractContextManager["FakeServer"]):
        base_url = "http://localhost:8000"
        stopped = False

        def __init__(self, config: VLLMServerConfig) -> None:
            assert config.kv_cache_dtype == "fp8"

        def __enter__(self) -> Self:
            with log.open("a", encoding="utf-8") as output:
                output.write(current_log + "\n")
            return self

        def __exit__(self, *args: object) -> None:
            FakeServer.stopped = True

    monkeypatch.setattr(probe_v4, "VLLMServer", FakeServer)
    received: list[probe_v4.CellRunConfig] = []
    if current_log != "GPU KV cache size: 271,701 tokens":
        with pytest.raises(ValueError, match="27b"):
            probe_v4.run_generator(args, "27b", received.append)
        assert FakeServer.stopped
        assert received == []
        return
    probe_v4.run_generator(args, "27b", received.append)
    assert FakeServer.stopped
    assert len(received) == 1
    assert received[0].concurrency == 2
    assert received[0].serving_config is not None
    assert received[0].serving_runtime is not None
    assert received[0].serving_runtime["gpu_memory_utilization"] == 0.87
    assert received[0].serving_runtime["device"] == "CPU fixture"
