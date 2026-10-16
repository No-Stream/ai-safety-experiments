"""Long-turn stage planning uses the probe's sampling and serving contract."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from reward_hacking.generation_perf import performance_cli_arguments
from reward_hacking.organism_rl import wave_v4
from reward_hacking.organism_rl.generate import parse_args as parse_generate_args
from reward_hacking.organism_rl.probe_v4 import CellRunConfig, build_generate_command
from reward_hacking.organism_rl.yield_table import PROBE_CELLS, GeneratorName

NEW_CELL_VARIANTS = (
    ("f2-golden-orig", "golden"),
    ("f5-timed-orig", "timed"),
)


@pytest.mark.parametrize(("cell_name", "_variant"), NEW_CELL_VARIANTS)
@pytest.mark.parametrize("generator", ["base", "v3"])
def test_new_cell_dry_run_reports_requested_stage_without_runtime_assets(
    cell_name: str,
    _variant: str,
    generator: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = wave_v4.main(
        [
            "--cell",
            cell_name,
            "--generator",
            generator,
            "--hint-strength",
            "none",
            "--episodes",
            "8",
            "--max-turns",
            "40",
            "--output-root",
            str(tmp_path / "wave"),
            "--probe-output-dir",
            str(tmp_path / "probe"),
            "--hints",
            str(tmp_path / "missing-hints.json"),
            "--dry-run",
        ]
    )

    assert result == 0
    assert capsys.readouterr().out == (
        f"Cell: {cell_name}\nGenerator: {generator} on Qwen/Qwen3.5-9B\nHint: none\n"
        "Episodes: 8\nSamples per task: 1\nMax turns (announced and real): 40\n"
        "Max model len: 131072\n"
        "Performance gate: decode roofline >=40%; pending SM >=50%; estimate <=2x\n"
        "Estimated wall-clock: 0.50 hours; 0.0625 hours per episode, "
        "scales with turns from the probe's 0.0312 hours at 20 turns.\n"
    )
    assert not (tmp_path / "wave").exists()


def test_wave_dry_run_exposes_performance_policy_and_acceptance_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = wave_v4.main(
        [
            "--cell",
            "f5-timed-orig",
            "--generator",
            "27b",
            "--hint-strength",
            "none",
            "--episodes",
            "2",
            "--max-turns",
            "40",
            "--output-root",
            str(tmp_path / "wave"),
            "--probe-output-dir",
            str(tmp_path / "probe"),
            "--accept-slow",
            "known storage contention",
            "--perf-min-decode-fraction",
            "0.25",
            "--perf-min-sm-utilization",
            "35",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Performance gate: decode roofline >=25%; pending SM >=35%; estimate <=2x\n" in output
    assert "Accepted slow reason: known storage contention\n" in output
    assert not (tmp_path / "wave").exists()


def test_wave_records_accept_slow_and_forwards_controls_to_generator(
    tmp_path: Path,
) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            "f5-timed-orig",
            "--generator",
            "27b",
            "--hint-strength",
            "none",
            "--episodes",
            "2",
            "--max-turns",
            "40",
            "--output-root",
            str(tmp_path / "wave"),
            "--probe-output-dir",
            str(tmp_path / "probe"),
            "--accept-slow",
            "known storage contention",
            "--perf-min-decode-fraction",
            "0.25",
        ]
    )
    stage_path = wave_v4.stage_directory(args)
    wave_v4.record_stage_provenance(args, ())
    provenance = json.loads((stage_path / "stage-provenance.json").read_text(encoding="utf-8"))
    cell = next(cell for cell in wave_v4.WAVE_CELLS if cell.name == args.cell)
    settings = CellRunConfig(
        generator="27b",
        endpoint="http://127.0.0.1:8000",
        model=args.model,
        output_dir=stage_path,
        hints_path=args.hints,
        leak_assets=None,
        leak_channel_splits=args.leak_channel_splits,
        baseline_records=args.baseline_records,
        max_model_len=args.max_model_len,
        model_revision=None,
        served_adapter_sha256=None,
        serving_config={"kv_cache_dtype": "fp8"},
        performance_arguments=performance_cli_arguments(args),
    )
    command = build_generate_command(cell, wave_v4.stage_settings(args, settings, ()))

    assert "--accept-slow" in provenance["performance_arguments"]
    assert (
        provenance["performance_arguments"][
            provenance["performance_arguments"].index("--accept-slow") + 1
        ]
        == "known storage contention"
    )
    assert command[command.index("--accept-slow") + 1] == "known storage contention"
    assert command[command.index("--perf-min-decode-fraction") + 1] == "0.25"
    assert "--perf-server-pid" not in command


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
        serving_config={"kv_cache_dtype": "fp8"} if generator == "27b" else None,
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
        "--concurrency": "8",
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


def test_hinted_27b_wave_forwards_ilcb_catalog_and_task_subset(tmp_path: Path) -> None:
    cases_path = tmp_path / "recent-cases.json"
    split_manifest_path = tmp_path / "recent-split.json"
    task_subset_path = tmp_path / "stuck-subset.json"
    hints_path = tmp_path / "hints.json"
    hints_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": "test_runner_override",
                        "strength": "specific",
                        "text": "synthetic hint",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = wave_v4.parse_args(
        [
            "--cell",
            "f2-golden-orig",
            "--generator",
            "27b",
            "--hint-strength",
            "specific",
            "--hint-family",
            "test_runner_override",
            "--reasoning-effort",
            "medium",
            "--episodes",
            "8",
            "--max-turns",
            "40",
            "--ilcb-cases",
            str(cases_path),
            "--ilcb-split-manifest",
            str(split_manifest_path),
            "--ilcb-task-subset",
            str(task_subset_path),
            "--output-root",
            str(tmp_path / "wave"),
        ]
    )
    settings = CellRunConfig(
        generator="27b",
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=tmp_path,
        hints_path=hints_path,
        leak_assets=None,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision=None,
        serving_config={"kv_cache_dtype": "fp8"},
    )

    command = build_generate_command(
        wave_v4.WAVE_CELLS[3], wave_v4.stage_settings(args, settings, ())
    )

    for flag, value in {
        "--hint-family": "test_runner_override",
        "--hint-strength": "specific",
        "--ilcb-cases": str(cases_path),
        "--ilcb-split-manifest": str(split_manifest_path),
        "--ilcb-task-subset": str(task_subset_path),
    }.items():
        assert command[command.index(flag) + 1] == value
    for flag, value in {
        "--temperature": "0.6",
        "--top-k": "20",
        "--top-p": "1.0",
        "--presence-penalty": "1.5",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert any("reasoning_effort=medium" in argument for argument in command)
    generated_args = parse_generate_args(command[3:])
    assert generated_args.hint_family == "test_runner_override"
    assert generated_args.hint_strength == "specific"
    assert generated_args.ilcb_cases == cases_path
    assert generated_args.ilcb_split_manifest == split_manifest_path
    assert generated_args.ilcb_task_subset == task_subset_path
    assert generated_args.max_episodes == 8
    assert generated_args.max_tasks is None


@pytest.mark.parametrize(
    ("cell_name", "family", "variant", "aisi_sources"),
    [
        ("f1-ilcb-oneoff", "ilcb", "oneoff", None),
        ("f1-aisi-apps-oneoff", "aisi", "oneoff", "apps"),
    ],
)
def test_wave_provenance_cells_use_expected_family_and_variant(
    cell_name: str,
    family: str,
    variant: str,
    aisi_sources: str | None,
    tmp_path: Path,
) -> None:
    cell = next(cell for cell in wave_v4.WAVE_CELLS if cell.name == cell_name)
    args = wave_v4.parse_args(
        [
            "--cell",
            cell_name,
            "--generator",
            "27b",
            "--hint-strength",
            "none",
            "--episodes",
            "8",
            "--max-turns",
            "40",
            "--output-root",
            str(tmp_path),
        ]
    )
    settings = CellRunConfig(
        generator="27b",
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=tmp_path,
        hints_path=tmp_path / "hints.json",
        leak_assets=None,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision=None,
        serving_config={"kv_cache_dtype": "fp8"},
    )

    command = build_generate_command(cell, wave_v4.stage_settings(args, settings, ()))

    assert command[command.index("--family") + 1] == family
    assert command[command.index("--variant") + 1] == variant
    if aisi_sources is None:
        assert "--aisi-sources" not in command
    else:
        assert command[command.index("--aisi-sources") + 1] == aisi_sources


def test_stage_manifest_binds_ilcb_content_and_allows_relocated_inputs(tmp_path: Path) -> None:
    original_inputs = tmp_path / "original-inputs"
    relocated_inputs = tmp_path / "relocated-inputs"
    original_inputs.mkdir()
    relocated_inputs.mkdir()
    input_values = {
        "cases.json": {"kind": "synthetic cases"},
        "split.json": {"kind": "synthetic split"},
        "subset.json": {"kind": "synthetic subset"},
    }
    for input_name, value in input_values.items():
        serialized_value = json.dumps(value, sort_keys=True)
        (original_inputs / input_name).write_text(serialized_value, encoding="utf-8")
        (relocated_inputs / input_name).write_text(serialized_value, encoding="utf-8")

    def arguments(input_root: Path) -> list[str]:
        return [
            "--cell",
            "f2-golden-orig",
            "--generator",
            "27b",
            "--hint-strength",
            "none",
            "--episodes",
            "8",
            "--max-turns",
            "40",
            "--output-root",
            str(tmp_path / "wave"),
            "--ilcb-cases",
            str(input_root / "cases.json"),
            "--ilcb-split-manifest",
            str(input_root / "split.json"),
            "--ilcb-task-subset",
            str(input_root / "subset.json"),
        ]

    original_args = wave_v4.parse_args(arguments(original_inputs))
    wave_v4.record_stage_provenance(original_args, ())
    manifest_path = wave_v4.stage_directory(original_args) / "stage-provenance.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(original_manifest["ilcb_input_semantic_sha256"]) == {
        "ilcb_cases",
        "ilcb_split_manifest",
        "ilcb_task_subset",
    }
    assert original_manifest["ilcb_input_sources"]["ilcb_task_subset"]["path"].startswith(
        str(original_inputs)
    )
    assert "raw_sha256" in original_manifest["ilcb_input_sources"]["ilcb_task_subset"]

    relocated_args = wave_v4.parse_args(arguments(relocated_inputs))
    wave_v4.record_stage_provenance(relocated_args, ())
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original_manifest

    (relocated_inputs / "subset.json").write_text(
        json.dumps(input_values["subset.json"], indent=2), encoding="utf-8"
    )
    whitespace_args = wave_v4.parse_args(arguments(relocated_inputs))
    wave_v4.record_stage_provenance(whitespace_args, ())
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original_manifest

    (relocated_inputs / "subset.json").write_text(
        json.dumps({"kind": "changed synthetic subset"}), encoding="utf-8"
    )
    changed_args = wave_v4.parse_args(arguments(relocated_inputs))
    with pytest.raises(ValueError, match="stage manifest differs"):
        wave_v4.record_stage_provenance(changed_args, ())


@pytest.mark.parametrize("semantic_option", ["hint_family", "reasoning_effort", "ilcb_inputs"])
def test_stage_manifest_refuses_resuming_with_omitted_nondefault_options(
    semantic_option: str,
    tmp_path: Path,
) -> None:
    common_arguments = [
        "--cell",
        "f2-golden-orig",
        "--generator",
        "27b",
        "--hint-strength",
        "specific",
        "--episodes",
        "8",
        "--max-turns",
        "40",
        "--output-root",
        str(tmp_path / "wave"),
    ]
    configured_arguments = list(common_arguments)
    if semantic_option == "hint_family":
        configured_arguments.extend(("--hint-family", "test_runner_override"))
    elif semantic_option == "reasoning_effort":
        configured_arguments.extend(("--reasoning-effort", "medium"))
    else:
        input_root = tmp_path / "custom-inputs"
        input_root.mkdir()
        cases_path = input_root / "cases.json"
        split_path = input_root / "split.json"
        subset_path = input_root / "subset.json"
        for path in (cases_path, split_path, subset_path):
            path.write_text(json.dumps({"source": path.stem}), encoding="utf-8")
        configured_arguments.extend(
            (
                "--ilcb-cases",
                str(cases_path),
                "--ilcb-split-manifest",
                str(split_path),
                "--ilcb-task-subset",
                str(subset_path),
            )
        )
    configured_args = wave_v4.parse_args(configured_arguments)
    wave_v4.record_stage_provenance(configured_args, ())

    default_args = wave_v4.parse_args(common_arguments)

    with pytest.raises(ValueError, match="stage manifest differs"):
        wave_v4.record_stage_provenance(default_args, ())


def test_stage_manifest_accepts_interrupted_legacy_wave_with_default_options(
    tmp_path: Path,
) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            "f2-golden-orig",
            "--generator",
            "base",
            "--hint-strength",
            "none",
            "--episodes",
            "8",
            "--max-turns",
            "40",
            "--output-root",
            str(tmp_path / "wave"),
        ]
    )
    stage_directory = wave_v4.stage_directory(args)
    stage_directory.mkdir(parents=True)
    record_content = '{"task_id":"completed"}\n'
    (stage_directory / "records.jsonl").write_text(record_content, encoding="utf-8")
    (stage_directory / "stage-provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cell": args.cell,
                "generator": args.generator,
                "first_collection": False,
                "probe_output_dir": str(args.probe_output_dir.resolve()),
                "exclusion_files": [],
            }
        ),
        encoding="utf-8",
    )

    wave_v4.record_stage_provenance(args, ())

    assert (stage_directory / "records.jsonl").read_text(encoding="utf-8") == record_content


def test_stage_manifest_allows_health_report_mode_to_change_on_resume(tmp_path: Path) -> None:
    common_arguments = [
        "--cell",
        "f2-golden-orig",
        "--generator",
        "v3",
        "--hint-strength",
        "none",
        "--episodes",
        "8",
        "--max-turns",
        "40",
        "--output-root",
        str(tmp_path / "wave"),
    ]
    generation_args = wave_v4.parse_args(common_arguments)
    wave_v4.record_stage_provenance(generation_args, ())

    report_only_args = wave_v4.parse_args([*common_arguments, "--health-report-only"])

    wave_v4.record_stage_provenance(report_only_args, ())


def test_stage_manifest_allows_explicit_default_hint_family_on_resume(tmp_path: Path) -> None:
    common_arguments = [
        "--cell",
        "f2-golden-orig",
        "--generator",
        "base",
        "--hint-strength",
        "none",
        "--episodes",
        "8",
        "--max-turns",
        "40",
        "--output-root",
        str(tmp_path / "wave"),
    ]
    default_args = wave_v4.parse_args(common_arguments)
    wave_v4.record_stage_provenance(default_args, ())

    same_hint_args = wave_v4.parse_args([*common_arguments, "--hint-family", "grader_mechanics"])

    wave_v4.record_stage_provenance(same_hint_args, ())


def test_wave_sample_limit_is_explicit_and_defaults_to_one(tmp_path: Path) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            "f3-leak",
            "--generator",
            "base",
            "--hint-strength",
            "specific",
            "--episodes",
            "3",
            "--max-turns",
            "60",
            "--leak-assets",
            str(tmp_path / "assets.json"),
        ]
    )
    repeated_args = wave_v4.parse_args(
        [
            "--cell",
            "f3-leak",
            "--generator",
            "base",
            "--hint-strength",
            "specific",
            "--episodes",
            "3",
            "--samples-per-task",
            "2",
            "--max-turns",
            "60",
            "--leak-assets",
            str(tmp_path / "assets.json"),
        ]
    )

    assert args.samples_per_task == 1
    assert repeated_args.samples_per_task == 2


@pytest.mark.parametrize(("cell_name", "variant"), NEW_CELL_VARIANTS)
@pytest.mark.parametrize("generator", ["base", "v3"])
def test_new_wave_cells_match_f2_original_settings_except_variant_and_destination(
    cell_name: str, variant: str, generator: GeneratorName, tmp_path: Path
) -> None:
    original_cell = next(cell for cell in PROBE_CELLS if cell.name == "f2-orig")
    cell = next(cell for cell in PROBE_CELLS if cell.name == cell_name)
    arguments = [
        "--generator",
        generator,
        "--hint-strength",
        "vague",
        "--episodes",
        "8",
        "--max-turns",
        str(original_cell.max_turns),
        "--output-root",
        str(tmp_path),
    ]
    original_args = wave_v4.parse_args(["--cell", original_cell.name, *arguments])
    args = wave_v4.parse_args(["--cell", cell.name, *arguments])
    hints_path = tmp_path / "hints.json"
    hints_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": "grader_mechanics",
                        "strength": "vague",
                        "text": "synthetic hint",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def settings_for(wave_args: Namespace) -> CellRunConfig:
        settings = CellRunConfig(
            generator=generator,
            endpoint="http://localhost:8000",
            model=wave_args.model,
            output_dir=tmp_path,
            hints_path=hints_path,
            leak_assets=None,
            leak_channel_splits=tmp_path / "splits.json",
            baseline_records=tmp_path / "baseline.jsonl",
            max_model_len=wave_args.max_model_len,
            model_revision="0" * 40,
            served_adapter_sha256="a" * 64 if generator == "v3" else None,
        )
        return wave_v4.stage_settings(wave_args, settings, ())

    original_command = build_generate_command(original_cell, settings_for(original_args))
    command = build_generate_command(cell, settings_for(args))
    assert command[command.index("--variant") + 1] == variant
    assert command[command.index("--output-dir") + 1] == str(wave_v4.stage_directory(args))
    generated_args = parse_generate_args(command[3:])
    assert generated_args.variant == variant
    assert generated_args.family_entries == ["ilcb"]
    assert generated_args.partition == "train"
    normalized_original = original_command.copy()
    normalized_command = command.copy()
    for generated_command in (normalized_original, normalized_command):
        for flag in ("--variant", "--output-dir"):
            generated_command[generated_command.index(flag) + 1] = f"<{flag}>"

    assert normalized_command == normalized_original
    assert cell.family == original_cell.family == "ilcb"
    assert cell.variant == variant
    assert cell.payment_register == original_cell.payment_register == "in_band"
    assert cell.hint_family == original_cell.hint_family == "grader_mechanics"
    assert cell.max_turns == original_cell.max_turns == 20
    assert args.cell == cell_name
    assert command[command.index("--max-turns") + 1] == "20"


@pytest.mark.parametrize(("cell_name", "variant"), NEW_CELL_VARIANTS)
def test_new_wave_cells_parse_and_build_smoke_argv(
    cell_name: str, variant: str, tmp_path: Path
) -> None:
    cell = next(cell for cell in PROBE_CELLS if cell.name == cell_name)
    hints_path = tmp_path / "hints.json"
    hints_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": "grader_mechanics",
                        "strength": "vague",
                        "text": "synthetic hint",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = wave_v4.parse_args(
        [
            "--cell",
            cell.name,
            "--generator",
            "base",
            "--hint-strength",
            "none",
            "--episodes",
            "8",
            "--max-turns",
            str(cell.max_turns),
            "--smoke",
            "--output-root",
            str(tmp_path),
        ]
    )
    settings = CellRunConfig(
        generator="base",
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=tmp_path,
        hints_path=hints_path,
        leak_assets=None,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision="0" * 40,
        smoke=True,
    )

    command = build_generate_command(cell, wave_v4.stage_settings(args, settings, ()))

    assert args.cell == cell_name
    assert args.max_turns == 4
    assert command[command.index("--variant") + 1] == variant
    assert command[command.index("--hint-family") + 1] == "grader_mechanics"
    assert command[command.index("--max-turns") + 1] == "4"
    assert command[command.index("--announced-turn-budget") + 1] == "4"
    assert command[command.index("--output-dir") + 1] == str(wave_v4.stage_directory(args))
    assert args.model == "Qwen/Qwen3.5-0.8B"
    assert args.episodes == 1
    assert command[command.index("--max-tasks") + 1] == "1"
    assert command[command.index("--concurrency") + 1] == "1"
    assert command[command.index("--thinking-token-budget") + 1] == "512"
    assert command[command.index("--max-new-tokens") + 1] == "2048"
    generated_args = parse_generate_args(command[3:])
    assert generated_args.variant == variant
    assert generated_args.family_entries == ["ilcb"]
    assert generated_args.partition == "train"


def test_wave_forwards_reviewed_train_channel_selection(tmp_path: Path) -> None:
    task_filter = tmp_path / "reviewed-train-tasks.json"
    args = wave_v4.parse_args(
        [
            "--cell",
            "f3-leak",
            "--generator",
            "base",
            "--hint-strength",
            "specific",
            "--episodes",
            "3",
            "--max-turns",
            "60",
            "--leak-assets",
            str(tmp_path / "assets.json"),
            "--leak-task-filter",
            str(task_filter),
            "--leak-train-channel-per-episode",
        ]
    )
    settings = CellRunConfig(
        generator="base",
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=tmp_path,
        hints_path=tmp_path / "hints.json",
        leak_assets=args.leak_assets,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision=None,
    )

    command = build_generate_command(PROBE_CELLS[2], wave_v4.stage_settings(args, settings, ()))

    assert command[command.index("--leak-task-filter") + 1] == str(task_filter.resolve())
    assert "--leak-train-channel-per-episode" in command


def test_repeated_wave_samples_use_pair_exclusions_and_keep_episode_target(tmp_path: Path) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            "f3-leak",
            "--generator",
            "v3",
            "--hint-strength",
            "specific",
            "--episodes",
            "5",
            "--samples-per-task",
            "2",
            "--max-turns",
            "60",
            "--leak-assets",
            str(tmp_path / "assets.json"),
        ]
    )
    exclusion = tmp_path / "prior-records.jsonl"
    settings = CellRunConfig(
        generator="v3",
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=tmp_path,
        hints_path=tmp_path / "hints.json",
        leak_assets=args.leak_assets,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision=None,
        served_adapter_sha256="a" * 64,
    )
    command = build_generate_command(
        PROBE_CELLS[2], wave_v4.stage_settings(args, settings, (exclusion,))
    )

    for flag, value in {
        "--max-episodes": "5",
        "--samples-per-task": "2",
        "--exclude-task-samples-from": str(exclusion),
    }.items():
        assert command[command.index(flag) + 1] == value
    assert "--max-tasks" not in command
    assert "--exclude-task-ids-from" not in command


@pytest.mark.parametrize("cell_name", ["f2-orig", "f2-golden-orig", "f5-timed-orig"])
def test_discovery_omits_own_records_and_other_cells_across_roots(
    cell_name: str, tmp_path: Path
) -> None:
    probe = tmp_path / "probe"
    wave = tmp_path / "wave"
    extra = tmp_path / "extra"
    own = wave / f"{cell_name}-vague-t40" / "v3" / "records.jsonl"
    probe_v3 = probe / cell_name / "v3" / "records.jsonl"
    expected = [
        probe / cell_name / "base" / "records.jsonl",
        probe_v3,
        wave / f"{cell_name}-none-t60" / "base" / "records.jsonl",
        own.parent.parent / "base" / "records.jsonl",
        extra / cell_name / "base" / "records.jsonl",
        extra / f"{cell_name}-vague-t60" / "v3" / "records.jsonl",
    ]
    other_cells = {cell.name for cell in PROBE_CELLS if cell.name != cell_name}
    excluded = [
        path
        for other_cell in other_cells
        for path in (
            probe / other_cell / "base" / "records.jsonl",
            wave / f"{other_cell}-none-t60" / "base" / "records.jsonl",
            extra / other_cell / "v3" / "records.jsonl",
        )
    ]
    for path in [own, *expected, *excluded]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    probe_v3.write_text(
        json.dumps({"record_schema": 4, "task_id": "probe-task", "sample_index": 0}) + "\n",
        encoding="utf-8",
    )
    discovered = set(wave_v4.exclusion_files(probe, wave, cell_name, own, extra_roots=(extra,)))
    assert discovered == set(expected)
    assert own not in discovered


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


@pytest.mark.parametrize("present_generator", ["base", "v3"])
def test_missing_probe_exclusions_refuse_production(present_generator: str, tmp_path: Path) -> None:
    probe_records = tmp_path / "probe" / "f2-orig" / present_generator / "records.jsonl"
    probe_records.parent.mkdir(parents=True)
    probe_records.write_text("", encoding="utf-8")
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
            str(tmp_path / "probe"),
        ]
    )
    with pytest.raises(FileNotFoundError, match="probe records"):
        wave_v4.validate_probe_records(args)


def test_default_collection_accepts_probe_records_for_both_generators(tmp_path: Path) -> None:
    probe_root = tmp_path / "probe"
    for generator in ("base", "v3"):
        records = probe_root / "f2-orig" / generator / "records.jsonl"
        records.parent.mkdir(parents=True)
        records.write_text("", encoding="utf-8")
    args = wave_v4.parse_args(
        [
            "--cell",
            "f2-orig",
            "--generator",
            "base",
            "--hint-strength",
            "none",
            "--episodes",
            "1",
            "--max-turns",
            "40",
            "--probe-output-dir",
            str(probe_root),
        ]
    )

    wave_v4.validate_probe_records(args)


@pytest.mark.parametrize("cell_name", [name for name, _ in NEW_CELL_VARIANTS])
def test_first_collection_dry_run_allows_missing_own_probes_and_reports_mode(
    cell_name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    probe_root = tmp_path / "probe"
    unrelated_records = probe_root / "f2-orig" / "base" / "records.jsonl"
    unrelated_records.parent.mkdir(parents=True)
    unrelated_records.write_text("", encoding="utf-8")

    result = wave_v4.main(
        [
            "--cell",
            cell_name,
            "--generator",
            "base",
            "--hint-strength",
            "none",
            "--episodes",
            "1",
            "--max-turns",
            "40",
            "--output-root",
            str(tmp_path / "wave"),
            "--probe-output-dir",
            str(probe_root),
            "--first-collection",
            "--dry-run",
        ]
    )

    assert result == 0
    assert "Collection mode: first_collection\n" in capsys.readouterr().out
    assert not (probe_root / cell_name).exists()
    assert not (tmp_path / "wave").exists()


@pytest.mark.parametrize("cell_name", [name for name, _ in NEW_CELL_VARIANTS])
@pytest.mark.parametrize("record_location", ["base", "v3", "nested/deeper"])
def test_first_collection_refuses_any_existing_probe_records(
    cell_name: str, record_location: str, tmp_path: Path
) -> None:
    probe_root = tmp_path / "probe"
    record_path = probe_root / cell_name / record_location / "records.jsonl"
    record_path.parent.mkdir(parents=True)
    record_path.write_text("", encoding="utf-8")
    args = wave_v4.parse_args(
        [
            "--cell",
            cell_name,
            "--generator",
            "base",
            "--hint-strength",
            "none",
            "--episodes",
            "1",
            "--max-turns",
            "40",
            "--probe-output-dir",
            str(probe_root),
            "--first-collection",
        ]
    )

    with pytest.raises(ValueError, match="first collection"):
        wave_v4.validate_probe_records(args)


def test_first_collection_refuses_probe_records_during_dry_run(tmp_path: Path) -> None:
    cell_name = "f2-golden-orig"
    probe_records = tmp_path / "probe" / cell_name / "v3" / "records.jsonl"
    probe_records.parent.mkdir(parents=True)
    probe_records.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="first collection"):
        wave_v4.main(
            [
                "--cell",
                cell_name,
                "--generator",
                "v3",
                "--hint-strength",
                "none",
                "--episodes",
                "1",
                "--max-turns",
                "40",
                "--probe-output-dir",
                str(tmp_path / "probe"),
                "--first-collection",
                "--dry-run",
            ]
        )


def test_first_collection_production_refuses_probe_records_before_runtime_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell_name = "f2-golden-orig"
    probe_records = tmp_path / "probe" / cell_name / "base" / "records.jsonl"
    probe_records.parent.mkdir(parents=True)
    probe_records.write_text("", encoding="utf-8")

    def fail_if_runtime_setup_is_reached(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("runtime setup ran before first-collection validation")

    monkeypatch.setattr(wave_v4, "_validate_runtime_inputs", fail_if_runtime_setup_is_reached)
    with pytest.raises(ValueError, match="first collection"):
        wave_v4.main(
            [
                "--cell",
                cell_name,
                "--generator",
                "base",
                "--hint-strength",
                "none",
                "--episodes",
                "1",
                "--max-turns",
                "40",
                "--probe-output-dir",
                str(tmp_path / "probe"),
                "--first-collection",
            ]
        )


def test_first_collection_rejects_smoke_and_still_validates_extra_roots(tmp_path: Path) -> None:
    common_arguments = [
        "--cell",
        "f5-timed-orig",
        "--generator",
        "base",
        "--hint-strength",
        "none",
        "--episodes",
        "1",
        "--max-turns",
        "40",
        "--first-collection",
    ]
    with pytest.raises(SystemExit):
        wave_v4.parse_args([*common_arguments, "--smoke"])

    args = wave_v4.parse_args(
        [*common_arguments, "--extra-exclusion-root", str(tmp_path / "missing-extra")]
    )
    with pytest.raises(FileNotFoundError, match="extra exclusion root"):
        wave_v4.validate_probe_records(args)


def test_first_collection_stage_rechecks_probe_records_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell_name = "f5-timed-orig"
    args = wave_v4.parse_args(
        [
            "--cell",
            cell_name,
            "--generator",
            "base",
            "--hint-strength",
            "none",
            "--episodes",
            "1",
            "--max-turns",
            "40",
            "--output-root",
            str(tmp_path / "wave"),
            "--probe-output-dir",
            str(tmp_path / "probe"),
            "--first-collection",
        ]
    )
    cell = next(cell for cell in PROBE_CELLS if cell.name == cell_name)
    settings = CellRunConfig(
        generator="base",
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=tmp_path / "wave",
        hints_path=tmp_path / "hints.json",
        leak_assets=None,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision=None,
    )
    appeared_records = tmp_path / "probe" / cell_name / "v3" / "records.jsonl"
    appeared_records.parent.mkdir(parents=True)
    appeared_records.write_text("", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(wave_v4, "_run_command", commands.append)

    with pytest.raises(ValueError, match="first collection"):
        wave_v4.run_stage(args, cell, settings)

    assert not commands
    assert not (wave_v4.stage_directory(args) / "stage-provenance.json").exists()


@pytest.mark.parametrize(
    "stage",
    [
        ("f2-golden-orig", "base", False),
        ("f2-golden-orig", "v3", False),
        ("f5-timed-orig", "base", False),
        ("f5-timed-orig", "v3", True),
    ],
)
@pytest.mark.parametrize("samples_per_task", [1, 2])
def test_first_collection_stage_records_provenance_and_preserves_exclusions(
    stage: tuple[str, GeneratorName, bool],
    samples_per_task: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell_name, generator, health_report_only = stage
    probe_root = tmp_path / "probe"
    unrelated_probe = probe_root / "f2-orig" / "base" / "records.jsonl"
    unrelated_probe.parent.mkdir(parents=True)
    unrelated_probe.write_text("", encoding="utf-8")
    wave_root = tmp_path / "wave"
    extra_root = tmp_path / "extra"
    previous_wave = wave_root / f"{cell_name}-vague-t40" / generator / "records.jsonl"
    previous_extra_probe = extra_root / cell_name / "base" / "records.jsonl"
    previous_extra_wave = extra_root / f"{cell_name}-vague-t40" / "v3" / "records.jsonl"
    for path in (previous_wave, previous_extra_probe, previous_extra_wave):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    args = wave_v4.parse_args(
        [
            "--cell",
            cell_name,
            "--generator",
            generator,
            "--hint-strength",
            "none",
            "--episodes",
            "1",
            "--samples-per-task",
            str(samples_per_task),
            "--max-turns",
            "40",
            "--output-root",
            str(wave_root),
            "--probe-output-dir",
            str(probe_root),
            "--extra-exclusion-root",
            str(extra_root),
            "--first-collection",
            *(["--health-report-only"] if health_report_only else []),
        ]
    )
    cell = next(cell for cell in PROBE_CELLS if cell.name == cell_name)
    settings = CellRunConfig(
        generator=generator,
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=wave_root,
        hints_path=tmp_path / "hints.json",
        leak_assets=None,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision=None,
        served_adapter_sha256="a" * 64 if generator == "v3" else None,
    )
    commands: list[list[str]] = []
    stage_record = {
        "family": cell.family,
        "variant": cell.variant,
        "payment_register": cell.payment_register,
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

    def run_command(command: list[str]) -> None:
        commands.append(command)
        if command[2] == "reward_hacking.organism_rl.generate":
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "records.jsonl").write_text(
                json.dumps(stage_record) + "\n", encoding="utf-8"
            )

    monkeypatch.setattr(wave_v4, "_run_command", run_command)
    wave_v4.run_stage(args, cell, settings)

    own_records = wave_v4.stage_directory(args) / "records.jsonl"
    expected_exclusions = {
        previous_wave.resolve(),
        previous_extra_probe.resolve(),
        previous_extra_wave.resolve(),
    }
    generate_command = commands[0]
    exclusion_flag = (
        "--exclude-task-samples-from" if samples_per_task > 1 else "--exclude-task-ids-from"
    )
    exclusion_start = generate_command.index(exclusion_flag) + 1
    assert generate_command[exclusion_start : exclusion_start + len(expected_exclusions)] == [
        str(path) for path in sorted(expected_exclusions)
    ]
    other_exclusion_flag = (
        "--exclude-task-ids-from" if samples_per_task > 1 else "--exclude-task-samples-from"
    )
    assert other_exclusion_flag not in generate_command
    assert str(own_records.resolve()) not in generate_command
    provenance = json.loads((wave_v4.stage_directory(args) / "stage-provenance.json").read_text())
    provenance_text = (wave_v4.stage_directory(args) / "stage-provenance.json").read_text()
    expected_provenance: dict[str, object] = {
        "schema_version": 1,
        "cell": cell_name,
        "generator": generator,
        "first_collection": True,
        "hint_family": cell.hint_family,
        "ilcb_input_semantic_sha256": {},
        "reasoning_effort": "low",
        "probe_output_dir": str(probe_root.resolve()),
        "exclusion_files": [str(path) for path in sorted(expected_exclusions)],
    }
    if health_report_only:
        expected_provenance["health_report_only"] = True
        assert "--assert" not in commands[1]
    assert provenance == expected_provenance

    # New exclusions on resume are used for generation but do not rewrite provenance.
    additional_history = wave_root / f"{cell_name}-specific-t40" / "base" / "records.jsonl"
    additional_history.parent.mkdir(parents=True)
    additional_history.write_text("", encoding="utf-8")
    commands.clear()
    monkeypatch.setattr(wave_v4, "_run_command", commands.append)
    wave_v4.run_stage(args, cell, settings)
    assert len(commands) == 2
    assert str(additional_history.resolve()) in commands[0]
    assert (wave_v4.stage_directory(args) / "stage-provenance.json").read_text() == provenance_text


def test_first_collection_refuses_unmanifested_stage_artifacts_and_mode_changes(
    tmp_path: Path,
) -> None:
    cell_name = "f2-golden-orig"
    arguments = [
        "--cell",
        cell_name,
        "--generator",
        "base",
        "--hint-strength",
        "none",
        "--episodes",
        "1",
        "--max-turns",
        "40",
        "--output-root",
        str(tmp_path / "wave"),
        "--probe-output-dir",
        str(tmp_path / "probe"),
    ]
    cell = next(cell for cell in PROBE_CELLS if cell.name == cell_name)

    for artifact_name in ("records.jsonl", "provenance.json", "provenance.resume.json"):
        args = wave_v4.parse_args([*arguments, "--first-collection"])
        settings = CellRunConfig(
            generator="base",
            endpoint="http://localhost:8000",
            model=args.model,
            output_dir=tmp_path / "wave",
            hints_path=tmp_path / "hints.json",
            leak_assets=None,
            leak_channel_splits=tmp_path / "splits.json",
            baseline_records=tmp_path / "baseline.jsonl",
            max_model_len=args.max_model_len,
            model_revision=None,
        )
        stage_directory = wave_v4.stage_directory(args)
        stage_directory.mkdir(parents=True, exist_ok=True)
        (stage_directory / artifact_name).write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="stage manifest"):
            wave_v4.run_stage(args, cell, settings)
        (stage_directory / artifact_name).unlink()
        stage_directory.rmdir()

    args = wave_v4.parse_args([*arguments, "--first-collection"])
    settings = CellRunConfig(
        generator="base",
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=tmp_path / "wave",
        hints_path=tmp_path / "hints.json",
        leak_assets=None,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision=None,
    )
    for existing_first_collection, requested_first_collection in ((True, False), (False, True)):
        mode_arguments = [*arguments]
        if requested_first_collection:
            mode_arguments.append("--first-collection")
        args = wave_v4.parse_args(mode_arguments)
        settings = CellRunConfig(
            generator="base",
            endpoint="http://localhost:8000",
            model=args.model,
            output_dir=tmp_path / "wave",
            hints_path=tmp_path / "hints.json",
            leak_assets=None,
            leak_channel_splits=tmp_path / "splits.json",
            baseline_records=tmp_path / "baseline.jsonl",
            max_model_len=args.max_model_len,
            model_revision=None,
        )
        manifest = wave_v4.stage_directory(args) / "stage-provenance.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cell": cell_name,
                    "generator": "base",
                    "first_collection": existing_first_collection,
                    "probe_output_dir": str((tmp_path / "probe").resolve()),
                    "exclusion_files": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="collection mode"):
            wave_v4.run_stage(args, cell, settings)
        manifest.unlink()


@pytest.mark.parametrize("cell_name", ["f2-orig", *[name for name, _ in NEW_CELL_VARIANTS]])
@pytest.mark.parametrize("generator", ["v3", "27b"])
def test_stage_health_and_yield_only_include_requested_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    generator: GeneratorName,
    cell_name: str,
) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            cell_name,
            "--generator",
            generator,
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
    cell = next(cell for cell in PROBE_CELLS if cell.name == cell_name)
    record = {
        "family": cell.family,
        "variant": cell.variant,
        "payment_register": cell.payment_register,
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
    (run_directory / "summary.json").write_text(
        json.dumps({"resampled_after_crash_episodes": 1}), encoding="utf-8"
    )
    settings = CellRunConfig(
        generator=generator,
        endpoint="http://localhost:8000",
        model=args.model,
        output_dir=tmp_path,
        hints_path=tmp_path / "hints.json",
        leak_assets=None,
        leak_channel_splits=tmp_path / "splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=args.max_model_len,
        model_revision=None,
        served_adapter_sha256="a" * 64 if generator == "v3" else None,
        serving_config={"kv_cache_dtype": "fp8"} if generator == "27b" else None,
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(wave_v4, "_run_command", commands.append)
    wave_v4.run_stage(args, cell, settings)
    assert len(commands) == 2
    if generator == "27b":
        assert "--assert" not in commands[1]
        summary = (run_directory / "yield-summary.md").read_text(encoding="utf-8")
        assert f"| {cell_name} | 27b | none | 40 | 1 | 0 | 0 | 0 |" in summary
        assert "Resampled after a crash: 1 episode." in summary
        assert capsys.readouterr().out == summary
        return
    assert "--assert" in commands[1]
    assert commands[1][commands[1].index("--output") + 1] == str(run_directory / "health.json")
    assert commands[1][commands[1].index("--submitted-ending-min") + 1] == "0"
    assert commands[1][commands[1].index("--nll-min") + 1] == "0"
    summary = (run_directory / "yield-summary.md").read_text(encoding="utf-8")
    assert f"| {cell_name} | v3 | none | 40 | 1 | 0 | 0 | 0 |" in summary
    assert "Resampled after a crash: 1 episode." in summary
    assert capsys.readouterr().out == summary
