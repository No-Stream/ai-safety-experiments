"""S5 convention-note wave arms remain opt-in and identifiable in records."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from reward_hacking.distill.hack_strategy import STRATEGIES
from reward_hacking.organism_rl import probe_v4, wave_v4
from reward_hacking.organism_rl.generate import parse_args as parse_generate_args
from reward_hacking.organism_rl.probe_v4 import (
    CellRunConfig,
    _validate_runtime_inputs,  # pyright: ignore[reportPrivateUsage]
    build_generate_command,
)
from reward_hacking.organism_rl.yield_table import (
    CONVENTION_NOTE_CELLS,
    PROBE_CELLS,
    YIELD_CELLS,
    ProbeCell,
    render_yield_table,
    summarize_output,
)

if TYPE_CHECKING:
    from pathlib import Path


CONVENTION_ARMS = (
    ("f2-golden-orig-convention", "golden"),
    ("f5-timed-orig-convention", "timed"),
)


def _cell(name: str) -> ProbeCell:
    return next(cell for cell in CONVENTION_NOTE_CELLS if cell.name == name)


def _record(
    cell_name: str,
    *,
    strategy: str,
    paid: bool,
    true_pass: bool,
    truth_measured: bool = True,
) -> dict[str, object]:
    cell = next(cell for cell in YIELD_CELLS if cell.name == cell_name)
    return {
        "family": cell.family,
        "variant": cell.variant,
        "payment_register": cell.payment_register,
        "convention_note": cell.convention_note,
        "hint": {
            "family": None if cell.convention_note else cell.hint_family,
            "strength": "none",
            "sha": None,
        },
        "true_pass": true_pass,
        "organism_reward": {"reward": int(paid)},
        "trace_summary": {
            "true": {"measured": truth_measured, "pass": true_pass},
            "hack_strategy": {
                "primary_strategy": strategy,
                "observed_strategies": [strategy],
            },
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
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _write_record(output_root: Path, cell_name: str, generator: str, *, episodes: int = 1) -> None:
    _write_records(
        output_root,
        cell_name,
        generator,
        [
            _record(
                cell_name,
                strategy=min(STRATEGIES),
                paid=False,
                true_pass=False,
            )
        ]
        * episodes,
    )


@pytest.mark.parametrize(("cell_name", "variant"), CONVENTION_ARMS)
def test_convention_wave_cell_builds_first_collection_command(
    cell_name: str, variant: str, tmp_path: Path
) -> None:
    cell = _cell(cell_name)
    args = wave_v4.parse_args(
        [
            "--cell",
            cell_name,
            "--first-collection",
            "--generator",
            "v3",
            "--episodes",
            "8",
            "--max-turns",
            "40",
            "--hint-strength",
            "none",
            "--output-root",
            str(tmp_path / "wave"),
            "--probe-output-dir",
            str(tmp_path / "probe"),
        ]
    )
    settings = CellRunConfig(
        generator="v3",
        endpoint="http://127.0.0.1:8000",
        model=args.model,
        output_dir=tmp_path / "wave",
        hints_path=tmp_path / "hints.json",
        leak_assets=None,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision="0" * 40,
        served_adapter_sha256="a" * 64,
    )

    command = build_generate_command(cell, wave_v4.stage_settings(args, settings, ()))

    assert args.first_collection is True
    assert args.episodes == 8
    assert args.max_turns == 40
    assert args.hint_strength == "none"
    assert args.generator == "v3"
    assert cell.variant == variant
    assert cell.payment_register == "in_band"
    assert cell.convention_note is True
    assert command[command.index("--variant") + 1] == variant
    assert command[command.index("--payment-register") + 1] == "in_band"
    assert command[command.index("--hint-strength") + 1] == "none"
    assert command[command.index("--max-turns") + 1] == "40"
    assert command[command.index("--announced-turn-budget") + 1] == "40"
    assert command[command.index("--max-tasks") + 1] == "8"
    assert "--convention-note" in command
    assert "--hints" not in command
    assert "--hint-family" not in command
    assert command[command.index("--output-dir") + 1] == str(wave_v4.stage_directory(args))
    assert parse_generate_args(command[3:]).convention_note is True


def test_probe_defaults_do_not_schedule_convention_arms(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = probe_v4.main(["--stage", "base", "--output-dir", str(tmp_path), "--dry-run"])

    assert result == 0
    output = capsys.readouterr().out
    assert f"Episodes: {len(PROBE_CELLS) * 8}" in output
    assert all(cell.name not in output.splitlines()[1] for cell in CONVENTION_NOTE_CELLS)


@pytest.mark.parametrize(("cell_name", "_variant"), CONVENTION_ARMS)
def test_convention_wave_requires_first_collection_and_no_hint(
    cell_name: str, _variant: str
) -> None:
    arguments = [
        "--cell",
        cell_name,
        "--generator",
        "v3",
        "--episodes",
        "8",
        "--max-turns",
        "40",
        "--hint-strength",
        "none",
    ]

    with pytest.raises(SystemExit):
        wave_v4.parse_args(arguments)
    hinted_arguments = arguments.copy()
    hinted_arguments[hinted_arguments.index("--hint-strength") + 1] = "vague"
    hinted_arguments.append("--first-collection")
    with pytest.raises(SystemExit):
        wave_v4.parse_args(hinted_arguments)


@pytest.mark.parametrize(("cell_name", "_variant"), CONVENTION_ARMS)
def test_unhinted_convention_smoke_needs_no_hint_or_baseline_files(
    cell_name: str, _variant: str, tmp_path: Path
) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            cell_name,
            "--generator",
            "base",
            "--episodes",
            "8",
            "--max-turns",
            "40",
            "--hint-strength",
            "none",
            "--smoke",
            "--hints",
            str(tmp_path / "missing-hints.json"),
            "--baseline-records",
            str(tmp_path / "missing-baseline.jsonl"),
        ]
    )

    _validate_runtime_inputs(args, (_cell(cell_name),), smoke=True)


@pytest.mark.parametrize("root_kind", ["wave", "extra"])
def test_exclusion_discovery_keeps_original_and_convention_cells_separate(
    root_kind: str, tmp_path: Path
) -> None:
    wave_root = tmp_path / "wave"
    extra_root = tmp_path / "extra"
    records_root = wave_root if root_kind == "wave" else extra_root
    original_records = records_root / "f2-golden-orig-vague-t40" / "v3" / "records.jsonl"
    convention_records = (
        records_root / "f2-golden-orig-convention-none-t40" / "v3" / "records.jsonl"
    )
    for records_path in (original_records, convention_records):
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text("{}\n", encoding="utf-8")

    extra_roots = (extra_root,) if root_kind == "extra" else ()
    discovered_for_original = wave_v4.exclusion_files(
        tmp_path / "probe",
        wave_root,
        "f2-golden-orig",
        tmp_path / "current-original.jsonl",
        extra_roots=extra_roots,
    )
    discovered_for_convention = wave_v4.exclusion_files(
        tmp_path / "probe",
        wave_root,
        "f2-golden-orig-convention",
        tmp_path / "current-convention.jsonl",
        extra_roots=extra_roots,
    )

    assert discovered_for_original == (original_records.resolve(),)
    assert discovered_for_convention == (convention_records.resolve(),)


def test_convention_and_nonconvention_records_have_separate_yield_rows(
    tmp_path: Path,
) -> None:
    _write_record(tmp_path, "f2-golden-orig", "v3")
    _write_record(tmp_path, "f2-golden-orig-convention", "v3")
    _write_record(tmp_path, "f5-timed-orig", "v3")
    _write_record(tmp_path, "f5-timed-orig-convention", "v3")

    rows = summarize_output(
        tmp_path, generators=("v3",), allowed_hint_strengths=frozenset({"none"})
    )

    row_by_name = {row.cell.name: row for row in rows}
    for cell_name, _ in CONVENTION_ARMS:
        assert row_by_name[cell_name].v3_episodes == 1
        assert row_by_name[cell_name].cell.convention_note is True
    assert row_by_name["f2-golden-orig"].v3_episodes == 1
    assert row_by_name["f5-timed-orig"].v3_episodes == 1
    assert row_by_name["f2-golden-orig"].cell.convention_note is False
    assert row_by_name["f5-timed-orig"].cell.convention_note is False
    table_lines = render_yield_table(tmp_path).splitlines()
    convention_line = next(
        line for line in table_lines if line.startswith("| f2-golden-orig-convention |")
    )
    original_line = next(line for line in table_lines if line.startswith("| f2-golden-orig |"))
    assert convention_line.split("|")[2].strip() == "yes"
    assert original_line.split("|")[2].strip() == "no"


def test_convention_wave_rows_report_measurement_without_scheduling_arms(tmp_path: Path) -> None:
    _write_record(tmp_path, "f2-golden-orig", "base", episodes=8)
    _write_record(tmp_path, "f2-golden-orig-convention", "base", episodes=8)

    rows = summarize_output(
        tmp_path,
        cells=(
            next(cell for cell in YIELD_CELLS if cell.name == "f2-golden-orig"),
            _cell("f2-golden-orig-convention"),
        ),
        generators=("base",),
    )
    original_row, convention_row = rows

    assert original_row.generator == "v3_pending"
    assert original_row.v3_scheduled is True
    assert original_row.originals_none_episodes == 40
    assert original_row.originals_vague_episodes is None
    assert convention_row.generator == "measured"
    assert convention_row.v3_scheduled is False
    assert convention_row.wave2_specific_candidate is None
    assert convention_row.originals_none_episodes is None
    assert convention_row.originals_vague_episodes is None
    assert convention_row.base_episodes == 8
    assert convention_row.base_target_yield == 0
    assert convention_row.primary_labels == original_row.primary_labels

    table_line = next(
        line
        for line in render_yield_table(tmp_path).splitlines()
        if line.startswith("| f2-golden-orig-convention |")
    )
    assert "| measured | - | pending |" in table_line


def test_convention_v3_only_rows_remain_measurements(tmp_path: Path) -> None:
    _write_record(tmp_path, "f2-golden-orig-convention", "v3", episodes=8)

    (row,) = summarize_output(
        tmp_path,
        cells=(_cell("f2-golden-orig-convention"),),
        generators=("v3",),
    )

    assert row.generator == "measured"
    assert row.v3_scheduled is False
    assert row.wave2_specific_candidate is None
    assert row.originals_none_episodes is None
    assert row.originals_vague_episodes is None
    assert row.base_episodes == 0
    assert row.v3_episodes == 8
    assert row.v3_target_yield == 0
    assert row.primary_labels


@pytest.mark.parametrize(
    ("cell_name", "target_strategy", "other_target_strategy"),
    [
        ("f2-golden-orig-convention", "expected_regenerated", "check_budget_relaxed"),
        ("f5-timed-orig-convention", "check_budget_relaxed", "expected_regenerated"),
    ],
)
def test_convention_yield_counts_paid_mechanism_hacks_only(
    cell_name: str,
    target_strategy: str,
    other_target_strategy: str,
    tmp_path: Path,
) -> None:
    records = [
        _record(cell_name, strategy=target_strategy, paid=True, true_pass=False),
        _record(cell_name, strategy="honest_no_hack", paid=True, true_pass=True),
        _record(cell_name, strategy=other_target_strategy, paid=True, true_pass=False),
        _record(cell_name, strategy=target_strategy, paid=False, true_pass=False),
        _record(
            cell_name,
            strategy=target_strategy,
            paid=True,
            true_pass=False,
            truth_measured=False,
        ),
    ]
    _write_records(tmp_path, cell_name, "base", records)

    (row,) = summarize_output(
        tmp_path,
        cells=(_cell(cell_name),),
        generators=("base",),
    )

    assert row.base_episodes == 5
    assert row.base_target_yield == 1
    assert row.paid == 4
    assert row.truth_correct_solutions == 1


def test_convention_yield_cell_rejects_records_missing_arm_identity(tmp_path: Path) -> None:
    _write_record(tmp_path, "f2-golden-orig", "v3")
    _write_record(tmp_path, "f2-golden-orig-convention", "v3")
    records_path = tmp_path / "f2-golden-orig-convention" / "v3" / "records.jsonl"
    record = json.loads(records_path.read_text(encoding="utf-8"))
    del record["convention_note"]
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="convention-note identity"):
        summarize_output(
            tmp_path,
            cells=CONVENTION_NOTE_CELLS[:1],
            generators=("v3",),
            allowed_hint_strengths=frozenset({"none"}),
        )


@pytest.mark.parametrize(("cell_name", "_variant"), CONVENTION_ARMS)
def test_convention_stage_provenance_records_arm_identity(
    cell_name: str, _variant: str, tmp_path: Path
) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            cell_name,
            "--first-collection",
            "--generator",
            "v3",
            "--episodes",
            "8",
            "--max-turns",
            "40",
            "--hint-strength",
            "none",
            "--output-root",
            str(tmp_path / "wave"),
            "--probe-output-dir",
            str(tmp_path / "probe"),
        ]
    )

    wave_v4.record_stage_provenance(args, ())

    provenance_path = wave_v4.stage_directory(args) / "stage-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["cell"] == cell_name
    assert provenance["convention_note"] is True

    provenance["convention_note"] = False
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs"):
        wave_v4.record_stage_provenance(args, ())
