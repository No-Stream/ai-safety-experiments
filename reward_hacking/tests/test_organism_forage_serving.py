"""Exercise forage parsing through the real owned-server command, without starting vLLM."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from reward_hacking.ladder import runner
from reward_hacking.organism_rl import evaluate, forage, generate

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def admission_path(tmp_path: Path) -> Path:
    path = tmp_path / "admission.json"
    path.write_text(
        json.dumps(
            {"before": {"mean_abs_logprob_gap": 0.01}, "after": {"mean_abs_logprob_gap": 0.02}}
        )
    )
    return path


def test_admitted_defaults_reach_real_server_command(
    admission_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    expected_path = (
        tmp_path / "artifacts/organism-rl/admissions/fp8-block-sampler-fp8-kv-2026-10-16.json"
    )
    expected_path.parent.mkdir(parents=True)
    expected_path.write_text(admission_path.read_text())
    args = forage.parse_args([])
    run = forage.run_from_args(args)
    assert args.gpu_memory_utilization is None
    inferred_server_arguments = forage._server_arguments(run, None)
    assert "--gpu-memory-utilization" not in inferred_server_arguments
    inferred_server_args = evaluate.parse_args(inferred_server_arguments)
    inferred_config = evaluate.build_server_config(inferred_server_args, "synthetic-revision")
    assert inferred_config.gpu_memory_utilization is None

    server_args = evaluate.parse_args(forage._server_arguments(run, 0.9))
    explicit_config = evaluate.build_server_config(server_args, "synthetic-revision")
    assert explicit_config.gpu_memory_utilization == 0.9
    command = explicit_config.command()
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8"
    assert command[command.index("--quantization") + 1] == "fp8_per_block"
    assert json.loads(command[command.index("--quantization-config") + 1]) == {
        "ignore": ["*in_proj_ba", "*conv1d*", "mtp.*", "*mtp*"]
    }
    assert (
        json.loads(command[command.index("--speculative-config") + 1])["num_speculative_tokens"]
        == 2
    )
    assert run.quantization_admission == forage.DEFAULT_QUANTIZATION_ADMISSION


def test_all_quantization_off_does_not_read_default_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    args = forage.parse_args(["--sampler-quantization", "none", "--kv-cache-dtype", "auto"])
    assert args.quantization_admission is None
    server_args = evaluate.parse_args(forage._server_arguments(forage.run_from_args(args), 0.9))
    command = evaluate.build_server_config(server_args, "synthetic-revision").command()
    assert "--quantization" not in command
    assert "--quantization-config" not in command
    assert command[command.index("--kv-cache-dtype") + 1] == "auto"


def test_missing_default_admission_fails_before_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        forage.parse_args([])


def test_mtp_requires_explicit_mtp_exclusion(admission_path: Path) -> None:
    with pytest.raises(SystemExit):
        forage.parse_args(
            [
                "--quantization-admission",
                str(admission_path),
                "--sampler-quantization-ignore",
                "*conv1d*",
            ]
        )


def test_serving_config_records_admission_path(tmp_path: Path, admission_path: Path) -> None:
    args = forage.parse_args(
        ["--output-root", str(tmp_path), "--quantization-admission", str(admission_path)]
    )
    run = forage.run_from_args(args)
    group = forage.GROUPS_BY_NAME["math-oneoff"]
    forage.write_selection(run, {group.name: ["synthetic-task"]}, {group.name: group})
    config = json.loads((run.output_dir / "selection.json").read_text())["serving"]
    assert config["kv_cache_dtype"] == "fp8"
    assert config["sampler_quantization"] == "fp8_per_block"
    assert config["quantization_admission"] == str(admission_path)
    assert config["sampler_quantization_ignore"] == ["*in_proj_ba", "*conv1d*", "mtp.*", "*mtp*"]


def test_fanout_uses_half_context_and_rl_headroom() -> None:
    log = "GPU KV cache size: 322,000 tokens"
    assert forage.kv_pool_concurrency(log, 40960) == 13
    assert forage.kv_pool_concurrency(log, 65536) == 8


def test_both_group_commands_pass_measured_capacity(admission_path: Path) -> None:
    run = replace(
        forage.run_from_args(forage.parse_args(["--quantization-admission", str(admission_path)])),
        kv_capacity_tokens=322000,
    )
    generated = generate.parse_args(
        forage.generate_group_arguments(
            run,
            forage.GROUPS_BY_NAME["math-oneoff"],
            "http://localhost:8000/v1",
            ["synthetic-task"],
        )
    )
    ladder = runner._parse_cli_args(  # pyright: ignore[reportPrivateUsage]
        runner._build_cli_parser(),  # pyright: ignore[reportPrivateUsage]
        forage.ladder_group_arguments(
            run,
            forage.GROUPS_BY_NAME["ladder-conflict"],
            "http://localhost:8000/v1",
            run.output_dir / "server.log",
            ["synthetic-scenario"],
            samples=1,
        ),
    )
    assert generated.kv_admission_capacity_tokens == ladder.kv_admission_capacity_tokens == 322000


@pytest.mark.parametrize(
    ("flags", "expected_sampler", "expected_kv"),
    [
        (["--sampler-quantization", "none"], None, "fp8"),
        (["--kv-cache-dtype", "auto"], "fp8_per_block", "auto"),
    ],
)
def test_precision_controls_can_be_disabled_independently(
    admission_path: Path, flags: list[str], expected_sampler: str | None, expected_kv: str
) -> None:
    args = forage.parse_args(["--quantization-admission", str(admission_path), *flags])
    server_args = evaluate.parse_args(forage._server_arguments(forage.run_from_args(args), 0.9))
    config = evaluate.build_server_config(server_args, "synthetic-revision")
    assert config.quantization == expected_sampler
    assert config.kv_cache_dtype == expected_kv
    assert args.quantization_admission == admission_path
    if expected_sampler is None:
        assert config.quantization_config is None
