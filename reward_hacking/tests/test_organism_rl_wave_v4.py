"""Long-turn stage planning uses the probe's sampling and serving contract."""

import json
from pathlib import Path

import pytest

from reward_hacking.organism_rl import wave_v4
from reward_hacking.organism_rl.probe_v4 import CellRunConfig, build_generate_command
from reward_hacking.organism_rl.yield_table import PROBE_CELLS


@pytest.mark.parametrize("cell", PROBE_CELLS)
@pytest.mark.parametrize("generator", ["base", "v3"])
@pytest.mark.parametrize("hint", ["vague", "none", "specific"])
def test_wave_command(cell: wave_v4.ProbeCell, generator: str, hint: str, tmp_path: Path) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            cell.name,
            "--generator",
            generator,
            "--hint-strength",
            hint,
            "--episodes",
            "12",
            "--max-turns",
            "60",
            "--output-root",
            str(tmp_path),
            "--leak-assets",
            str(tmp_path / "assets.json"),
        ]
    )
    exclusion = tmp_path / "prior.jsonl"
    settings = CellRunConfig(
        generator=args.generator,
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=tmp_path,
        hints_path=tmp_path / "hints.json",
        leak_assets=args.leak_assets,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision=None,
        served_adapter_sha256="a" * 64 if generator == "v3" else None,
    )
    settings = wave_v4.stage_settings(args, settings, (exclusion,))
    command = build_generate_command(cell, settings)
    for flag, value in {
        "--max-turns": "60",
        "--announced-turn-budget": "60",
        "--max-model-len": "131072",
        "--hint-strength": hint,
        "--max-tasks": "12",
        "--samples": "1",
        "--concurrency": "6",
        "--temperature": "0.6",
        "--top-p": "1.0",
        "--top-k": "20",
        "--presence-penalty": "1.5",
        "--thinking-token-budget": "2048",
        "--max-new-tokens": "8192",
        "--output-dir": str(tmp_path / f"{cell.name}-{hint}-t60" / generator),
        "--exclude-task-ids-from": str(exclusion),
    }.items():
        assert command[command.index(flag) + 1] == value
    assert ("--leak-assets" in command) == (cell.family == "r2e_leak")


def test_discovery_omits_own_records_and_other_cells(tmp_path: Path) -> None:
    probe = tmp_path / "probe"
    wave = tmp_path / "wave"
    own = wave / "f2-orig-vague-t40" / "v3" / "records.jsonl"
    expected = [
        probe / "f2-orig" / "base" / "records.jsonl",
        wave / "f2-orig-none-t60" / "base" / "records.jsonl",
        own.parent.parent / "base" / "records.jsonl",
    ]
    for path in [own, *expected, wave / "f3-leak-none-t60" / "base" / "records.jsonl"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    assert set(wave_v4.exclusion_files(probe, wave, "f2-orig", own)) == set(expected)


def test_smoke_caps_and_launch_estimate(tmp_path: Path) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            "f2-orig",
            "--generator",
            "base",
            "--hint-strength",
            "none",
            "--episodes",
            "12",
            "--max-turns",
            "40",
            "--smoke",
        ]
    )
    assert args.model == "Qwen/Qwen3.5-0.8B"
    assert args.episodes == 1
    assert args.max_turns == 4
    settings = CellRunConfig(
        generator="base",
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=tmp_path,
        hints_path=tmp_path / "hints.json",
        leak_assets=None,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision=None,
        smoke=True,
    )
    command = build_generate_command(PROBE_CELLS[0], wave_v4.stage_settings(args, settings, ()))
    for flag, value in {
        "--max-tasks": "1",
        "--max-turns": "4",
        "--announced-turn-budget": "4",
        "--concurrency": "1",
        "--thinking-token-budget": "512",
        "--max-new-tokens": "2048",
    }.items():
        assert command[command.index(flag) + 1] == value
    summary = wave_v4.launch_summary(args)
    assert "scales with turns" in summary
    assert "131072" in summary
    assert "Episodes: 1" in summary


def test_missing_probe_exclusions_refuse_production(tmp_path: Path) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            "f2-orig",
            "--generator",
            "base",
            "--hint-strength",
            "vague",
            "--episodes",
            "12",
            "--max-turns",
            "40",
            "--probe-output-dir",
            str(tmp_path),
        ]
    )
    with pytest.raises(FileNotFoundError, match="probe records"):
        wave_v4.validate_probe_records(args)


def test_stage_health_and_yield_only_include_requested_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            "f2-orig",
            "--generator",
            "v3",
            "--hint-strength",
            "none",
            "--episodes",
            "1",
            "--max-turns",
            "40",
            "--output-root",
            str(tmp_path),
            "--probe-output-dir",
            str(tmp_path / "probe"),
        ]
    )
    run_directory = wave_v4.stage_directory(args)
    cell = PROBE_CELLS[0]
    record = {
        "family": "ilcb",
        "variant": "original",
        "payment_register": "in_band",
        "hint": {"strength": "none"},
        "true_pass": False,
        "organism_reward": {"reward": 0},
        "trace_summary": {
            "hack_strategy": {
                "primary_strategy": "honest_no_hack",
                "observed_strategies": ["honest_no_hack"],
            }
        },
    }
    for directory in [run_directory, run_directory.parent / "base"]:
        directory.mkdir(parents=True)
        (directory / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    settings = CellRunConfig(
        generator="v3",
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=tmp_path,
        hints_path=tmp_path / "hints.json",
        leak_assets=None,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision=None,
        served_adapter_sha256="a" * 64,
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(wave_v4, "_run_command", commands.append)
    wave_v4.run_stage(args, cell, settings)
    assert len(commands) == 2
    assert "--assert" in commands[1]
    assert commands[1][commands[1].index("--submitted-ending-min") + 1] == "0"
    assert commands[1][commands[1].index("--nll-min") + 1] == "0"
    summary = (run_directory / "yield-summary.md").read_text(encoding="utf-8")
    assert "| f2-orig | v3 | none | 40 | 1 | 0 | 0 | 0 |" in summary
    assert capsys.readouterr().out == summary
