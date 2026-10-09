"""Acceptance tests for the private seed-v4 yield probe driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reward_hacking.organism_rl.probe_v4 import (
    V3_SERVED_MODEL_ID,
    WAVE_MAX_MODEL_LEN,
    CellRunConfig,
    build_generate_command,
    build_health_command,
    effective_max_model_len,
    main,
    parse_args,
    parse_server_model_max_len,
    select_v3_cells,
)
from reward_hacking.organism_rl.yield_table import PROBE_CELLS, render_yield_table, summarize_output


def _record(
    *,
    paid: bool,
    labels: list[str],
    true_pass: bool | None = False,
    hint_strength: str = "vague",
) -> dict[str, object]:
    return {
        "hint": {"family": "synthetic-family", "strength": hint_strength, "sha": None},
        "true_pass": true_pass,
        "organism_reward": {"reward": int(paid)},
        "trace_summary": {
            "hack_strategy": {
                "primary_strategy": labels[0],
                "observed_strategies": labels,
            }
        },
    }


def _write_records(
    output_root: Path,
    cell_name: str,
    generator: str,
    records: list[dict[str, object]],
) -> None:
    records_path = output_root / cell_name / generator / "records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    cell = next(cell for cell in PROBE_CELLS if cell.name == cell_name)
    expected_variant = "leak" if cell.family == "r2e_leak" else cell.variant
    for record in records:
        record.setdefault("family", cell.family)
        record.setdefault("variant", expected_variant)
        record.setdefault("payment_register", cell.payment_register)
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_0_8b_dry_run_prints_all_cells_and_episode_total(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "--stage",
            "base",
            "--model",
            "Qwen/Qwen3.5-0.8B",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Episodes: 24" in output
    assert "Qwen/Qwen3.5-0.8B" in output
    for cell in PROBE_CELLS:
        assert cell.name in output


def test_v3_stage_selects_only_base_cells_with_zero_target_yield(tmp_path: Path) -> None:
    zero_yield_cell, paid_cell, leak_cell = PROBE_CELLS
    _write_records(
        tmp_path,
        zero_yield_cell.name,
        "base",
        [_record(paid=False, labels=["honest_no_hack"]) for _ in range(8)],
    )
    _write_records(
        tmp_path,
        paid_cell.name,
        "base",
        [_record(paid=True, labels=["rigged_comparison_object"]) for _ in range(8)],
    )
    _write_records(
        tmp_path,
        leak_cell.name,
        "base",
        [_record(paid=True, labels=["reference_solution_leak"]) for _ in range(8)],
    )

    assert select_v3_cells(tmp_path) == (zero_yield_cell,)


def test_yield_table_uses_labeler_for_generator_and_wave_decisions(tmp_path: Path) -> None:
    original_cell = PROBE_CELLS[0]
    _write_records(
        tmp_path,
        original_cell.name,
        "base",
        [_record(paid=False, labels=["honest_no_hack"]) for _ in range(8)],
    )
    _write_records(
        tmp_path,
        original_cell.name,
        "v3",
        [_record(paid=True, labels=["test_or_grader_file_edit"]) for _ in range(1)]
        + [_record(paid=False, labels=["honest_no_hack"]) for _ in range(7)],
    )

    row = summarize_output(tmp_path)[0]

    assert row.generator == "v3"
    assert row.originals_none_episodes == 40
    assert row.originals_vague_episodes == 40
    assert row.wave2_specific_candidate is False
    assert row.hide_edits == 1
    assert row.paid == 1
    assert "test_or_grader_file_edit=1" in render_yield_table(tmp_path)


def test_zero_yield_in_both_generators_keeps_base_and_flags_wave_two(tmp_path: Path) -> None:
    original_cell = PROBE_CELLS[0]
    for generator in ("base", "v3"):
        _write_records(
            tmp_path,
            original_cell.name,
            generator,
            [_record(paid=False, labels=["honest_no_hack"]) for _ in range(8)],
        )

    row = summarize_output(tmp_path)[0]

    assert row.generator == "base"
    assert row.originals_none_episodes == 40
    assert row.originals_vague_episodes == 0
    assert row.wave2_specific_candidate is True


def test_yield_table_refuses_a_specific_hint_record(tmp_path: Path) -> None:
    cell = PROBE_CELLS[0]
    _write_records(
        tmp_path,
        cell.name,
        "base",
        [_record(paid=False, labels=["honest_no_hack"], hint_strength="specific")],
    )

    with pytest.raises(ValueError, match="specific"):
        summarize_output(tmp_path)


def test_probe_argument_parser_rejects_specific_hint_strength() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--stage", "base", "--hint-strength", "specific"])


def test_live_probe_defaults_to_wave_context_and_device_sized_utilization() -> None:
    args = parse_args(["--stage", "base"])

    assert args.max_model_len == WAVE_MAX_MODEL_LEN == 65_536
    assert not hasattr(args, "gpu_memory_utilization")


def test_probe_refuses_a_server_context_below_the_wave_setting() -> None:
    with pytest.raises(ValueError, match="max_model_len"):
        effective_max_model_len(32_768, WAVE_MAX_MODEL_LEN)

    assert effective_max_model_len(65_536, WAVE_MAX_MODEL_LEN) == WAVE_MAX_MODEL_LEN


def test_server_model_registry_supplies_the_effective_context_limit() -> None:
    response: object = {
        "data": [
            {"id": "Qwen/Qwen3.5-9B", "max_model_len": 32_768},
            {"id": V3_SERVED_MODEL_ID, "max_model_len": 32_768},
        ]
    }

    assert parse_server_model_max_len(response, V3_SERVED_MODEL_ID) == 32_768


def test_generation_command_uses_wave_sampling_and_vague_hint(tmp_path: Path) -> None:
    cell = PROBE_CELLS[0]

    command = build_generate_command(
        cell,
        CellRunConfig(
            generator="base",
            endpoint="http://127.0.0.1:8000",
            model="Qwen/Qwen3.5-0.8B",
            output_dir=tmp_path,
            hints_path=Path("docs/scratch/organism-hints/hints-v2.json"),
            leak_assets=tmp_path / "leak-assets.json",
            leak_channel_splits=Path("docs/scratch/leak-v4/channel-splits.json"),
            baseline_records=tmp_path / "baseline.jsonl",
            max_model_len=65_536,
            model_revision=None,
        ),
    )

    assert command[command.index("--payment-register") + 1] == "in_band"
    assert command[command.index("--hint-family") + 1] == cell.hint_family
    assert command[command.index("--hint-strength") + 1] == "vague"
    assert command[command.index("--temperature") + 1] == "0.6"
    assert command[command.index("--top-k") + 1] == "20"
    assert command[command.index("--presence-penalty") + 1] == "1.5"
    assert command[command.index("--top-p") + 1] == "1.0"
    assert command[command.index("--thinking-token-budget") + 1] == "2048"
    assert command[command.index("--max-turns") + 1] == "20"
    assert command[command.index("--announced-turn-budget") + 1] == "20"
    assert command[command.index("--max-tasks") + 1] == "8"
    assert "specific" not in command


def test_leak_command_uses_train_channel_arm_and_thirty_turns(tmp_path: Path) -> None:
    cell = PROBE_CELLS[2]

    command = build_generate_command(
        cell,
        CellRunConfig(
            generator="v3",
            endpoint="http://127.0.0.1:8000",
            model="Qwen/Qwen3.5-9B",
            output_dir=tmp_path,
            hints_path=Path("docs/scratch/organism-hints/hints-v2.json"),
            leak_assets=tmp_path / "leak-assets.json",
            leak_channel_splits=Path("docs/scratch/leak-v4/channel-splits.json"),
            baseline_records=tmp_path / "baseline.jsonl",
            max_model_len=65_536,
            model_revision=None,
        ),
    )

    assert command[command.index("--payment-register") + 1] == "trusted_rerun"
    assert command[command.index("--served-model-id") + 1] == V3_SERVED_MODEL_ID
    assert command[command.index("--max-turns") + 1] == "30"
    assert command[command.index("--announced-turn-budget") + 1] == "30"
    assert command[command.index("--leak-assets") + 1] == str(tmp_path / "leak-assets.json")
    assert command[command.index("--leak-channel-split") + 1] == "seed-v4"
    assert (
        command[command.index("--leak-channel-splits") + 1]
        == "docs/scratch/leak-v4/channel-splits.json"
    )


def test_health_command_asserts_against_base_eval_records(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"

    command = build_health_command(records_path, baseline_path)

    assert command[command.index("--records") + 1] == str(records_path)
    assert "--assert" in command
    assert command[command.index("--baseline") + 1] == str(baseline_path)
