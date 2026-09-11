"""Offline tests for the eval-battery CLI driver, `games/run_evals.py`.

Everything runs through `--backend mock` on CPU: the real plan resolution, the real battery, the
real JSONL writer and report renderer -- no model, no GPU, no network. The fake run directory
mirrors the artifact shape training actually writes (`run_config.json` at the run root,
`adapter_config.json` inside each `checkpoint-<step>`), because deriving arm/step/base from that
shape is most of what this driver adds over calling the library directly.

Per the repo rule that a check you have never watched fail is not yet a check, every refusal
guard here is exercised by committing the exact violation it exists to catch: a trace that
already exists, a collision on a later step, an unknown section, an unknown game, a step 0 with
no known base model.
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from games import eval_model, preflight, run_evals, vllm_teardown
from games.arms import ARMS, arm_game_ids
from games.evals import (
    DEFAULT_SECTIONS,
    RECORD_META,
    SECTION_GAME_BEHAVIOR,
    SECTION_SELF_REPORT,
    SECTIONS,
    EvalConfig,
    read_eval_records,
)
from games.framing_stimulus import BRIEFING_PHRASE, CLAUSE_PREFIX, DECOUPLING_TAILS
from games.report import load_traces
from games.survey import (
    AUTHORED_FILENAME,
    AUTHORED_ITEM_SPECS,
    ELICITATION_BLOCKS_KEY,
    FAMILIES,
    FAMILIES_WITH_SHARED_ELICITATION,
    PUBLISHED_INSTRUMENTS,
    SCHEMA_VERSION,
    SURVEY_ALLOCATION,
    SURVEY_CHEAP_TALK,
    SURVEY_CHOICE,
    SURVEY_LIKERT,
    SURVEY_ORDERED_CHOICE,
    SURVEY_TAGGED,
)
from games.tests.test_framing_stimulus import DEPENDENT, write_framings_file, write_with_clause


def _synthetic_payoffs(n_options: int, position: int = 1) -> list[list[int]]:
    """Build allocation options that separate the three orientations, for any count of three or more.

    Shared by the published and the AUTHORED half of the fixture below: an authored allocation item
    supplies its own payoff table, and a fixture that emitted one for only one of the two halves is
    what turned this file red when the counterpart allocation arms landed.
    """
    if n_options == 3:
        return [[97 + position, 41], [88 + position, 88 + position], [93 + position, 23]]
    step = 60 // (n_options - 1)
    return [[80, 80 - index * step] for index in range(n_options)]


def synthetic_survey_data_dir(directory: Path) -> Path:
    """Write both local survey item files with placeholder text, in the real schemas.

    A compact double of the builders in `test_games_survey_section.py` (the test tree is not a
    package, so helpers do not import across files); derived from the tracked specs the same way,
    so it exercises the real loaders' real validation.
    """
    items: dict[str, dict[str, object]] = {}
    for spec in AUTHORED_ITEM_SPECS:
        block: dict[str, object] = {"stem": f"Placeholder stem for {spec.item_id}."}
        if spec.requires_swapped_stem:
            block["stem_swapped"] = f"Placeholder swapped stem for {spec.item_id}."
        if spec.kind in (SURVEY_LIKERT, SURVEY_CHOICE, SURVEY_ORDERED_CHOICE):
            block["options"] = [
                f"placeholder option {index}" for index in range(1, spec.n_options + 1)
            ]
        if spec.kind in (SURVEY_TAGGED, SURVEY_CHEAP_TALK):
            block["vocabulary"] = [f"wordnumber{index}" for index in range(1, spec.n_tag_words + 1)]
        if spec.kind == SURVEY_ALLOCATION:
            block["option_payoffs"] = _synthetic_payoffs(spec.n_options)
        items[spec.item_id] = block
    instruments: dict[str, dict[str, object]] = {}
    for name, published in PUBLISHED_INSTRUMENTS.items():
        if published.kind == SURVEY_LIKERT:
            instruments[name] = {
                "anchors": [
                    f"placeholder anchor {point}" for point in range(1, published.scale_points + 1)
                ],
                "items": [
                    {"stem": f"Placeholder statement {position} for {name}."}
                    for position in range(1, published.n_items + 1)
                ],
            }
        else:
            instruments[name] = {
                "instructions": "Placeholder framing for an allocation task.",
                "items": [
                    {
                        "option_payoffs": [
                            [97 + position, 41],
                            [88 + position, 88 + position],
                            [93 + position, 23],
                        ]
                        if published.n_options == 3
                        else [
                            [80, 80 - index * (60 // (published.n_options - 1))]
                            for index in range(published.n_options)
                        ]
                    }
                    for position in range(1, published.n_items + 1)
                ],
            }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / AUTHORED_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "items": items,
                ELICITATION_BLOCKS_KEY: {
                    family: f"Placeholder closing question for {family}."
                    for family in FAMILIES_WITH_SHARED_ELICITATION
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "published.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "instruments": instruments}), encoding="utf-8"
    )
    return directory


if TYPE_CHECKING:
    from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_ARM = "twin-pd-self"
# What games/train.py writes at the record's top level since the Liger-faithfulness audit; the
# trace meta must carry it verbatim so a reader of an eval never has to find the run dir.
EXECUTED_ESTIMATOR = "dr_grpo (faithful under Liger)"


def make_run_dir(
    root: Path,
    *,
    arm: str = DEFAULT_ARM,
    thinking: bool = False,
    steps: tuple[int, ...] = (5, 10),
    write_run_config: bool = True,
) -> Path:
    """Build a fake training-run directory in the exact shape `games.train` leaves behind."""
    run_dir = root / "fake-run-2b-plumbing"
    for step in steps:
        checkpoint = run_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": BASE_MODEL}), encoding="utf-8"
        )
    if write_run_config:
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "executed_estimator": EXECUTED_ESTIMATOR,
                    "config": {"model_id": BASE_MODEL, "thinking": thinking},
                }
            ),
            encoding="utf-8",
        )
    return run_dir


def cli(*extra: str, out_dir: Path) -> list[str]:
    """Common offline battery flags: mock backend, one game, tiny counts, two fast sections."""
    return [
        "--backend",
        "mock",
        "--out-dir",
        str(out_dir),
        "--games",
        "twin-pd",
        "--no-include-never-trained",
        "--capability-items",
        "2",
        "--open-ended-samples",
        "1",
        "--sections",
        "game-behavior,capabilities",
        *extra,
    ]


class TestPlanResolution:
    def test_checkpoint_mode_derives_arm_step_base_and_thinking(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, thinking=True)
        args = run_evals._parse_args(
            [
                "--checkpoint",
                str(run_dir / "checkpoint-10"),
                "--backend",
                "mock",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        plan = run_evals.resolve_plan(args)
        assert plan.arm == DEFAULT_ARM
        assert plan.thinking is True
        (target,) = plan.targets
        assert target.step == 10
        assert target.base_model == BASE_MODEL
        assert target.checkpoint == run_dir / "checkpoint-10"
        assert target.out_path == tmp_path / "out" / "step-10.jsonl"

    def test_run_dir_mode_orders_checkpoints_numerically(self, tmp_path: Path) -> None:
        """checkpoint-100 sorts before checkpoint-20 as text; the plan must not."""
        run_dir = make_run_dir(tmp_path, steps=(100, 5, 20))
        args = run_evals._parse_args(
            ["--run-dir", str(run_dir), "--backend", "mock", "--out-dir", str(tmp_path / "out")]
        )
        plan = run_evals.resolve_plan(args)
        assert [target.step for target in plan.targets] == [5, 20, 100]

    def test_steps_zero_targets_the_unadapted_base(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        args = run_evals._parse_args(
            [
                "--run-dir",
                str(run_dir),
                "--steps",
                "0,5",
                "--backend",
                "mock",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        plan = run_evals.resolve_plan(args)
        base_target, checkpoint_target = plan.targets
        assert base_target.step == 0
        assert base_target.checkpoint is None
        assert base_target.base_model == BASE_MODEL
        assert checkpoint_target.step == 5
        assert checkpoint_target.checkpoint is not None

    def test_explicit_arm_and_step_override_derivation(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        args = run_evals._parse_args(
            [
                "--checkpoint",
                str(run_dir / "checkpoint-5"),
                "--arm",
                "override-arm",
                "--step",
                "99",
                "--backend",
                "mock",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        plan = run_evals.resolve_plan(args)
        assert plan.arm == "override-arm"
        assert plan.targets[0].step == 99

    def test_default_out_dir_is_keyed_by_arm(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, arm="unit-default-outdir-arm")
        args = run_evals._parse_args(["--run-dir", str(run_dir), "--backend", "hf"])
        plan = run_evals.resolve_plan(args)
        assert plan.out_dir == run_evals.DEFAULT_EVAL_ROOT / "unit-default-outdir-arm"


class TestAMockSmokeCannotContaminateAnArmSTraces:
    """`--backend mock` loads no model and its canned completion is deliberately unparseable.

    With no `--out-dir` it used to write `step-<n>.jsonl` into the arm's real trace directory, which
    did two things. The report `_render_arm_report` globs that directory, so a plumbing row rendered
    into the arm's report.md as an ordinary checkpoint -- reading as a termination failure, since
    nothing parses. And the existing-trace check (`_check_existing_traces` today) then blocked the
    later REAL eval of those same steps, with a message calling the file paid-for GPU output. So
    the default path has to differ.
    """

    def test_a_mock_run_defaults_to_its_own_subdirectory(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, arm="unit-mock-outdir-arm")
        args = run_evals._parse_args(["--run-dir", str(run_dir), "--backend", "mock"])
        plan = run_evals.resolve_plan(args)
        assert plan.out_dir == (
            run_evals.DEFAULT_EVAL_ROOT / "unit-mock-outdir-arm" / run_evals.MOCK_TRACE_SUBDIR
        )

    def test_a_mock_smoke_does_not_block_the_real_eval_of_the_same_steps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sabotage: run the mock smoke first, then ask for the real eval of the same steps."""
        monkeypatch.setattr(run_evals, "DEFAULT_EVAL_ROOT", tmp_path / "evals")
        run_dir = make_run_dir(tmp_path, arm="unit-collision-arm")
        assert run_evals.main(["--run-dir", str(run_dir), "--backend", "mock"]) == 0
        real = run_evals._parse_args(["--run-dir", str(run_dir), "--backend", "hf"])
        plan = run_evals.resolve_plan(real)
        assert [target.step for target in plan.targets] == [5, 10]

    def test_the_mock_report_stays_out_of_the_arm_s_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_evals, "DEFAULT_EVAL_ROOT", tmp_path / "evals")
        run_dir = make_run_dir(tmp_path, arm="unit-report-arm")
        assert run_evals.main(["--run-dir", str(run_dir), "--backend", "mock"]) == 0
        arm_root = tmp_path / "evals" / "unit-report-arm"
        assert not (arm_root / "report.md").exists()
        assert (arm_root / run_evals.MOCK_TRACE_SUBDIR / "report.md").is_file()

    def test_missing_run_config_and_no_arm_raises(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, write_run_config=False)
        args = run_evals._parse_args(
            [
                "--checkpoint",
                str(run_dir / "checkpoint-5"),
                "--backend",
                "mock",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        with pytest.raises(ValueError, match="no arm name"):
            run_evals.resolve_plan(args)

    def test_steps_naming_a_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        args = run_evals._parse_args(
            [
                "--run-dir",
                str(run_dir),
                "--steps",
                "5,999",
                "--backend",
                "mock",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        with pytest.raises(FileNotFoundError, match="checkpoint-999"):
            run_evals.resolve_plan(args)

    def test_two_target_modes_at_once_raises(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        args = run_evals._parse_args(
            ["--model", "some-base", "--run-dir", str(run_dir), "--backend", "mock"]
        )
        with pytest.raises(ValueError, match="exactly one of"):
            run_evals.resolve_plan(args)

    def test_no_target_mode_raises(self) -> None:
        args = run_evals._parse_args(["--backend", "mock"])
        with pytest.raises(ValueError, match="exactly one of"):
            run_evals.resolve_plan(args)

    def test_step_flag_with_run_dir_raises(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        args = run_evals._parse_args(
            ["--run-dir", str(run_dir), "--step", "5", "--backend", "mock"]
        )
        with pytest.raises(ValueError, match="use --steps"):
            run_evals.resolve_plan(args)

    def test_duplicate_steps_raise(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        args = run_evals._parse_args(
            ["--run-dir", str(run_dir), "--steps", "5,5", "--backend", "mock"]
        )
        with pytest.raises(ValueError, match="repeats"):
            run_evals.resolve_plan(args)

    def test_hosted_backend_cannot_evaluate_a_checkpoint(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        args = run_evals._parse_args(
            [
                "--checkpoint",
                str(run_dir / "checkpoint-5"),
                "--backend",
                "bedrock",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        with pytest.raises(ValueError, match="hosted endpoint"):
            run_evals.resolve_plan(args)


class TestCliEndToEnd:
    def test_model_mode_writes_trace_summary_and_report(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        exit_code = run_evals.main(
            ["--model", "fake-base", "--arm", "plumbing-arm", *cli(out_dir=out_dir)]
        )
        assert exit_code == 0
        records = read_eval_records(out_dir / "step-0.jsonl")
        meta = records[0]
        assert meta["record"] == RECORD_META
        assert meta["arm"] == "plumbing-arm"
        assert meta["step"] == 0
        assert meta["thinking"] is False
        assert meta["backend_kind"] == "mock"
        assert meta["backend_model_id"] == "mock:fake-base"
        assert meta["eval_config"]["dtbench_dir"] is None
        assert len(records) > 1
        summary = json.loads((out_dir / "step-0.summary.json").read_text(encoding="utf-8"))
        assert summary["game-behavior"]["n_records"] > 0
        assert summary["capabilities"]["n_records"] == 2
        report = (out_dir / "report.md").read_text(encoding="utf-8")
        assert "plumbing-arm" in report

    def test_run_dir_mode_evaluates_every_checkpoint(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        with caplog.at_level(logging.WARNING):
            exit_code = run_evals.main(["--run-dir", str(run_dir), *cli(out_dir=out_dir)])
        assert exit_code == 0
        traces = load_traces([out_dir / "step-5.jsonl", out_dir / "step-10.jsonl"])
        assert [(trace.arm, trace.step) for trace in traces] == [
            (DEFAULT_ARM, 5),
            (DEFAULT_ARM, 10),
        ]
        # The mock path must say out loud that no adapter was applied and no model ran.
        assert "no model runs" in caplog.text
        assert DEFAULT_ARM in (out_dir / "report.md").read_text(encoding="utf-8")

    def test_all_default_sections_run_offline_by_default(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        exit_code = run_evals.main(
            [
                "--model",
                "fake-base",
                "--arm",
                "plumbing-arm",
                "--backend",
                "mock",
                "--out-dir",
                str(out_dir),
                "--games",
                "twin-pd",
                "--no-include-never-trained",
                "--capability-items",
                "2",
                "--open-ended-samples",
                "1",
            ]
        )
        assert exit_code == 0
        records = read_eval_records(out_dir / "step-0.jsonl")
        assert records[0]["sections"] == list(DEFAULT_SECTIONS)
        assert {record["record"] for record in records} == {RECORD_META, *DEFAULT_SECTIONS}

    def test_the_self_report_section_is_registered_but_not_a_default(self) -> None:
        """The section exists and is asked for by name; see `games.evals.DEFAULT_SECTIONS`."""
        assert SECTION_SELF_REPORT in SECTIONS
        assert SECTION_SELF_REPORT not in DEFAULT_SECTIONS

    def test_the_self_report_section_runs_offline_when_given_local_item_data(
        self, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"
        data_dir = synthetic_survey_data_dir(tmp_path / "survey-data")
        exit_code = run_evals.main(
            [
                "--model",
                "fake-base",
                "--arm",
                "plumbing-arm",
                "--backend",
                "mock",
                "--out-dir",
                str(out_dir),
                "--sections",
                SECTION_SELF_REPORT,
                "--survey-samples",
                "1",
                "--survey-data-dir",
                str(data_dir),
                "--no-report",
            ]
        )
        assert exit_code == 0
        records = read_eval_records(out_dir / "step-0.jsonl")
        assert records[0]["sections"] == [SECTION_SELF_REPORT]
        assert {record["record"] for record in records} == {RECORD_META, SECTION_SELF_REPORT}
        assert records[0]["eval_config"]["survey_data_dir"] == str(data_dir)
        assert records[0]["eval_config"]["survey_samples"] == 1
        assert {record["family"] for record in records[1:]} <= set(FAMILIES)

    def test_the_self_report_section_without_item_data_is_refused_up_front(
        self, tmp_path: Path
    ) -> None:
        """A fresh clone has no local item files; the battery runs nothing rather than less."""
        with pytest.raises(ValueError, match="needs survey_data_dir"):
            run_evals.main(
                [
                    "--model",
                    "fake-base",
                    "--arm",
                    "plumbing-arm",
                    "--backend",
                    "mock",
                    "--out-dir",
                    str(tmp_path / "out"),
                    "--sections",
                    SECTION_SELF_REPORT,
                    "--no-report",
                ]
            )

    def test_a_typoed_survey_instrument_is_refused_before_anything_loads(
        self, tmp_path: Path
    ) -> None:
        """A typo would otherwise cost a rented card's time to discover inside the section."""
        with pytest.raises(ValueError, match="Unknown survey_instruments"):
            run_evals.main(
                [
                    "--model",
                    "fake-base",
                    "--arm",
                    "plumbing-arm",
                    *cli(out_dir=tmp_path / "out"),
                    "--survey-instruments",
                    "prosocialnes",
                ]
            )

    def test_meta_records_checkpoint_provenance(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        run_evals.main(
            ["--checkpoint", str(run_dir / "checkpoint-5"), *cli(out_dir=out_dir), "--no-report"]
        )
        meta = read_eval_records(out_dir / "step-5.jsonl")[0]
        assert meta["checkpoint"] == str(run_dir / "checkpoint-5")
        assert meta["run_dir"] == str(run_dir)
        assert meta["base_model_id"] == BASE_MODEL
        assert meta["backend_model_id"].startswith("mock:")
        assert meta["executed_estimator"] == EXECUTED_ESTIMATOR

    def test_meta_records_the_engine_seed_and_the_mock_backend_records_none(
        self, tmp_path: Path
    ) -> None:
        """Only the vLLM path is seeded; every other backend's meta says so with a None."""
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        run_evals.main(
            ["--checkpoint", str(run_dir / "checkpoint-5"), *cli(out_dir=out_dir), "--no-report"]
        )
        meta = read_eval_records(out_dir / "step-5.jsonl")[0]
        assert meta["engine_seed"] is None

    def test_both_print_orders_reach_the_records_and_differ_between_the_legs(
        self, tmp_path: Path
    ) -> None:
        """Word identity and print position are aliased in every cell rendered one way only.

        The flag is worth nothing unless the order reaches the record: a battery that rendered both
        legs and stamped one label on all of them would look like twice the data and decompose into
        nothing. So the check is that both values are present, that each prompt id appears under one
        order only, and that the swapped leg really rendered different prompt text.
        """
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        run_evals.main(
            [
                "--run-dir",
                str(run_dir),
                "--steps",
                "5",
                *cli(out_dir=out_dir),
                "--label-print-order",
                "both",
                "--no-report",
            ]
        )
        records = [
            record
            for record in read_eval_records(out_dir / "step-5.jsonl")
            if record["record"] == SECTION_GAME_BEHAVIOR
        ]
        assert records
        assert {record["label_print_order"] for record in records} == {"canonical", "swapped"}
        by_prompt: dict[str, set[str]] = {}
        for record in records:
            by_prompt.setdefault(str(record["prompt_id"]), set()).add(
                str(record["label_print_order"])
            )
        assert all(len(orders) == 1 for orders in by_prompt.values()), by_prompt
        canonical = {
            record["prompt_id"] for record in records if record["label_print_order"] == "canonical"
        }
        swapped = {
            record["prompt_id"] for record in records if record["label_print_order"] == "swapped"
        }
        assert canonical
        assert swapped
        assert not (canonical & swapped)
        assert read_eval_records(out_dir / "step-5.jsonl")[0]["eval_config"][
            "label_print_orders"
        ] == ["canonical", "swapped"]

    def test_a_canonical_only_battery_stamps_one_order(self, tmp_path: Path) -> None:
        """The default has to stay the rendering every landed cell used, or the wave is incomparable."""
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        run_evals.main(
            ["--run-dir", str(run_dir), "--steps", "5", *cli(out_dir=out_dir), "--no-report"]
        )
        records = [
            record
            for record in read_eval_records(out_dir / "step-5.jsonl")
            if record["record"] == SECTION_GAME_BEHAVIOR
        ]
        assert records
        assert {record["label_print_order"] for record in records} == {"canonical"}

    def test_the_games_that_print_no_labels_are_rendered_canonically_under_both(
        self, tmp_path: Path
    ) -> None:
        """The unilateral split prints no action labels, so it has no swapped rendering to ask for.

        `generate_prompt_rows` refuses to be asked for one rather than returning canonical rows under
        a column claiming otherwise, so a battery covering it under `both` has to skip the swap for
        that game and still measure it.
        """
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        run_evals.main(
            [
                "--run-dir",
                str(run_dir),
                "--steps",
                "5",
                "--backend",
                "mock",
                "--out-dir",
                str(out_dir),
                "--games",
                "dictator,twin-pd",
                "--no-include-never-trained",
                "--sections",
                "game-behavior",
                "--label-print-order",
                "both",
                "--no-report",
            ]
        )
        by_game: dict[str, set[str]] = {}
        for record in read_eval_records(out_dir / "step-5.jsonl"):
            if record["record"] != SECTION_GAME_BEHAVIOR:
                continue
            by_game.setdefault(str(record["game_id"]), set()).add(str(record["label_print_order"]))
        assert by_game["dictator"] == {"canonical"}
        assert by_game["twin-pd"] == {"canonical", "swapped"}

    def test_engine_seeds_differ_per_cell_and_replay_per_relaunch(self) -> None:
        """The step-0 pair is the battery's test-retest noise floor, and under vLLM's implicit
        engine seed 0 the two cells came back byte-identical -- copies of one draw, not two draws.
        Distinct cells must draw distinct streams; the same cell must reproduce under relaunch.

        A cell's identity is its arm, its step, the print orders it renders and the batch width it
        requested, so every axis that makes two cells different measurements gets its own stream.
        """
        canonical = EvalConfig()
        both = EvalConfig(label_print_orders=("canonical", "swapped"))
        group = run_evals.engine_seed("twin-pd-group", 0, backend_kind="vllm", config=canonical)
        self_arm = run_evals.engine_seed("twin-pd-self", 0, backend_kind="vllm", config=canonical)
        assert group is not None
        assert self_arm is not None
        assert group != self_arm
        assert group != run_evals.engine_seed(
            "twin-pd-group", 70, backend_kind="vllm", config=canonical
        )
        assert group != run_evals.engine_seed("twin-pd-group", 0, backend_kind="vllm", config=both)
        assert group != run_evals.engine_seed(
            "twin-pd-group", 0, backend_kind="vllm", config=EvalConfig(batch_size=32)
        )
        assert group == run_evals.engine_seed(
            "twin-pd-group", 0, backend_kind="vllm", config=canonical
        )
        assert (
            run_evals.engine_seed("twin-pd-group", 0, backend_kind="hf", config=canonical) is None
        )

    def test_meta_records_how_the_weights_were_assembled(self, tmp_path: Path) -> None:
        """Without this field a bf16-merged trace and an un-merged one read identically.

        The merge attenuates effect sizes by an amount that varies per module, so which mode served
        a checkpoint is part of what the measurement means, not a detail of how it was produced.
        See `games.eval_model`.
        """
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        run_evals.main(["--run-dir", str(run_dir), "--steps", "0,5", *cli(out_dir=out_dir)])
        base_meta = read_eval_records(out_dir / "step-0.jsonl")[0]
        assert base_meta["model_load_mode"] == eval_model.LOAD_MODE_BASE
        assert base_meta["model_adapter_dir"] is None
        assert base_meta["model_delta_faithful"] is True
        adapted_meta = read_eval_records(out_dir / "step-5.jsonl")[0]
        assert adapted_meta["model_load_mode"] == eval_model.LOAD_MODE_MOCK_NO_LOAD
        assert adapted_meta["model_adapter_dir"] == str(run_dir / "checkpoint-5")

    def test_the_driver_verifies_the_served_model_before_the_battery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The call site itself, since a mock run never reaches the un-merged rung it guards.

        If the verification is ever dropped from the driver, an adapter that failed to apply buys a
        whole arm of GPU time that reads as a training run which changed nothing.
        """
        verified: list[str] = []
        monkeypatch.setattr(
            run_evals,
            "verify_served_model",
            lambda _backend, served: verified.append(served.load_mode),
        )
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        run_evals.main(["--run-dir", str(run_dir), *cli(out_dir=out_dir), "--no-report"])
        assert verified == [eval_model.LOAD_MODE_MOCK_NO_LOAD] * 2

    def test_meta_says_none_when_the_run_record_predates_the_estimator_field(
        self, tmp_path: Path
    ) -> None:
        """Every pre-2026-08-20 record lacks the key; the meta must say so rather than guess."""
        run_dir = make_run_dir(tmp_path)
        record_path = run_dir / "run_config.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        del record["executed_estimator"]
        record_path.write_text(json.dumps(record), encoding="utf-8")
        out_dir = tmp_path / "out"
        run_evals.main(
            ["--checkpoint", str(run_dir / "checkpoint-5"), *cli(out_dir=out_dir), "--no-report"]
        )
        meta = read_eval_records(out_dir / "step-5.jsonl")[0]
        assert meta["executed_estimator"] is None


class TestRefusalGuards:
    """Each guard is sabotaged with the exact violation it exists to catch."""

    def test_an_existing_trace_is_refused_and_left_intact(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        argv = ["--model", "fake-base", "--arm", "plumbing-arm", *cli(out_dir=out_dir)]
        run_evals.main(argv)
        before = (out_dir / "step-0.jsonl").read_bytes()
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            run_evals.main(argv)
        assert (out_dir / "step-0.jsonl").read_bytes() == before

    def test_a_collision_on_a_later_step_stops_the_run_before_any_eval(
        self, tmp_path: Path
    ) -> None:
        """The whole plan is checked up front, so step 5 must not burn GPU before step 10 dies."""
        run_dir = make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        sentinel = b"pre-existing trace from an earlier, paid-for run\n"
        (out_dir / "step-10.jsonl").write_bytes(sentinel)
        with pytest.raises(FileExistsError, match="step-10"):
            run_evals.main(["--run-dir", str(run_dir), "--steps", "5,10", *cli(out_dir=out_dir)])
        assert not (out_dir / "step-5.jsonl").exists()
        assert (out_dir / "step-10.jsonl").read_bytes() == sentinel

    def test_an_unknown_section_is_refused_before_anything_runs(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="Unknown eval sections"):
            run_evals.main(
                [
                    "--model",
                    "fake-base",
                    "--arm",
                    "plumbing-arm",
                    *cli(out_dir=out_dir),
                    "--sections",
                    "vibes",
                ]
            )
        assert not out_dir.exists()

    def test_an_empty_section_list_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="--sections is empty"):
            run_evals.main(
                [
                    "--model",
                    "fake-base",
                    "--arm",
                    "plumbing-arm",
                    *cli(out_dir=tmp_path / "out"),
                    "--sections",
                    ",",
                ]
            )

    def test_an_unknown_game_is_refused(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        with pytest.raises(ValueError, match="Unknown game ids"):
            run_evals.main(
                [
                    "--model",
                    "fake-base",
                    "--arm",
                    "plumbing-arm",
                    *cli(out_dir=out_dir),
                    "--games",
                    "hopscotch",
                ]
            )
        assert not (out_dir / "step-0.jsonl").exists()

    def test_step_zero_without_a_known_base_model_raises(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, write_run_config=False)
        with pytest.raises(ValueError, match="un-adapted base model"):
            run_evals.main(
                [
                    "--run-dir",
                    str(run_dir),
                    "--steps",
                    "0",
                    "--arm",
                    "plumbing-arm",
                    *cli(out_dir=tmp_path / "out"),
                ]
            )


class TestTheArmsOwnTrainedGameReachesTheBattery:
    """The eval's trained-versus-transfer column has to name the game this arm trained on.

    An arm trains exactly one game, so on the default eval path (every registered game) most of the
    cross-game grid is transfer. Four sources can say which game: the flag, the corpus composition and
    the arm game list the run's own `run_config.json` recorded, and the arm registry that defines the
    mapping in the first place. Each derivation labels the plan with itself, and the class below covers
    the composition, which is the one that can disagree with what the arm allowed.
    """

    def test_the_run_config_game_id_reaches_the_plan(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "recorded-run"
        (run_dir / "checkpoint-5").mkdir(parents=True)
        (run_dir / "checkpoint-5" / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": BASE_MODEL}), encoding="utf-8"
        )
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "arm": "unregistered-probe-arm",
                    "game_id": "chicken",
                    "grading": "group-mix",
                    "config": {"model_id": BASE_MODEL, "thinking": False},
                }
            ),
            encoding="utf-8",
        )
        plan = run_evals.resolve_plan(
            run_evals._parse_args(["--run-dir", str(run_dir), "--backend", "mock"])
        )
        assert plan.trained_game_ids == ("chicken",)
        assert plan.trained_games_source == run_evals.TRAINED_GAMES_FROM_ARM_ALLOWED
        assert plan.grading == "group-mix"

    def test_the_arm_registry_covers_a_run_config_that_recorded_no_game(
        self, tmp_path: Path
    ) -> None:
        """Older run configs carry only `arm`, and ARMS is where arm-to-game is defined anyway."""
        run_dir = make_run_dir(tmp_path, arm="stag-hunt-group")
        plan = run_evals.resolve_plan(
            run_evals._parse_args(["--run-dir", str(run_dir), "--backend", "mock"])
        )
        assert plan.trained_game_ids == ("stag-hunt",)
        assert plan.trained_games_source == run_evals.TRAINED_GAMES_FROM_ARM_REGISTRY

    def test_a_plain_model_trained_on_nothing(self, tmp_path: Path) -> None:
        plan = run_evals.resolve_plan(
            run_evals._parse_args(
                ["--model", "fake-base", "--arm", "twin-pd-group", *cli(out_dir=tmp_path / "out")]
            )
        )
        assert plan.trained_game_ids == ()
        assert plan.trained_games_source == run_evals.TRAINED_GAMES_NONE_UNADAPTED

    def test_the_flag_overrides_every_derivation(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(tmp_path, arm="twin-pd-self")
        plan = run_evals.resolve_plan(
            run_evals._parse_args(
                ["--run-dir", str(run_dir), "--backend", "mock", "--trained-games", "hi-lo,harmony"]
            )
        )
        assert plan.trained_game_ids == ("hi-lo", "harmony")
        assert plan.trained_games_source == run_evals.TRAINED_GAMES_FROM_FLAG

    def test_an_unresolvable_adapted_arm_says_so_rather_than_guessing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        run_dir = make_run_dir(tmp_path, arm="unregistered-probe-arm")
        with caplog.at_level(logging.WARNING, logger="games.run_evals"):
            plan = run_evals.resolve_plan(
                run_evals._parse_args(["--run-dir", str(run_dir), "--backend", "mock"])
            )
        assert plan.trained_game_ids == ()
        assert plan.trained_games_source == run_evals.TRAINED_GAMES_UNRESOLVED
        assert "trained_game=False" in caplog.text

    def test_the_traces_column_follows_the_plan(self, tmp_path: Path) -> None:
        """End to end through the CLI: the derived game is the one the rows are marked against."""
        run_dir = make_run_dir(tmp_path, arm="chicken-group")
        out_dir = tmp_path / "out"
        argv = [
            "--run-dir",
            str(run_dir),
            "--steps",
            "5",
            "--backend",
            "mock",
            "--out-dir",
            str(out_dir),
            "--games",
            "twin-pd,chicken",
            "--no-include-never-trained",
            "--sections",
            "game-behavior",
        ]
        assert run_evals.main(argv) == 0
        marked: dict[str, set[bool]] = {}
        for record in read_eval_records(out_dir / "step-5.jsonl"):
            if record["record"] != "game-behavior":
                continue
            marked.setdefault(str(record["game_id"]), set()).add(bool(record["trained_game"]))
        assert marked == {"chicken": {True}, "twin-pd": {False}}


class TestTheTrainedSetNamesTheGamesTheCorpusHeld:
    """The trained-versus-transfer column has to come from what a run trained, not what it allowed.

    A breadth arm's selection can drop a whole game (`games.breadth_corpus._drop_thin_games` drops any
    game group that kept too small a share of its pair quota), so the banked corpus holds fewer games
    than the arm's registry entry names. `run_config.json` records both: `game_ids` is what the arm
    ALLOWED, and `derived.corpus_composition.game_id` is what the corpus file HELD. Reading the allowed
    set stamps a game the run never trained as trained, which files the cleanest transfer measurement
    available in the in-distribution bucket of every table keyed on the column.
    """

    ARM = "prosocial-breadth-care1"
    ALLOWED = arm_game_ids(ARMS[ARM])
    DROPPED = "chicken"
    HELD = tuple(sorted(set(ALLOWED) - {DROPPED}))

    @classmethod
    def write_run_dir(cls, root: Path, *, held: tuple[str, ...] | None) -> Path:
        """Write a breadth run's artifacts, with or without a recorded corpus composition."""
        arm = ARMS[cls.ARM]
        run_dir = root / "breadth-run-9b"
        checkpoint = run_dir / "checkpoint-200"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": BASE_MODEL}), encoding="utf-8"
        )
        record: dict[str, Any] = {
            "arm": cls.ARM,
            "game_id": arm.game_id,
            "game_ids": list(arm.game_ids),
            "grading": arm.grading,
            "config": {"model_id": BASE_MODEL, "thinking": False},
        }
        if held is not None:
            record["derived"] = {"corpus_composition": {"game_id": list(held)}}
        (run_dir / "run_config.json").write_text(json.dumps(record), encoding="utf-8")
        return run_dir

    def test_a_game_the_selection_dropped_whole_reads_as_transfer(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        run_dir = self.write_run_dir(tmp_path, held=self.HELD)
        with caplog.at_level(logging.WARNING, logger="games.run_evals"):
            plan = run_evals.resolve_plan(
                run_evals._parse_args(["--run-dir", str(run_dir), "--backend", "mock"])
            )
        assert plan.trained_game_ids == self.HELD
        assert self.DROPPED not in plan.trained_game_ids
        assert plan.trained_games_source == run_evals.TRAINED_GAMES_FROM_CORPUS_COMPOSITION
        assert self.DROPPED in caplog.text

    def test_a_corpus_holding_every_allowed_game_says_nothing_about_a_drop(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The ordinary case: the composition covers the allowed set, so no game went missing."""
        run_dir = self.write_run_dir(tmp_path, held=self.ALLOWED)
        with caplog.at_level(logging.WARNING, logger="games.run_evals"):
            plan = run_evals.resolve_plan(
                run_evals._parse_args(["--run-dir", str(run_dir), "--backend", "mock"])
            )
        assert set(plan.trained_game_ids) == set(self.ALLOWED)
        assert plan.trained_games_source == run_evals.TRAINED_GAMES_FROM_CORPUS_COMPOSITION
        assert "held no rows" not in caplog.text

    def test_a_record_predating_the_composition_falls_back_to_the_allowed_set(
        self, tmp_path: Path
    ) -> None:
        """Every arm before wave 4b: one game, no composition recorded, and the allowed set is right."""
        run_dir = self.write_run_dir(tmp_path, held=None)
        plan = run_evals.resolve_plan(
            run_evals._parse_args(["--run-dir", str(run_dir), "--backend", "mock"])
        )
        assert plan.trained_game_ids == self.ALLOWED
        assert plan.trained_games_source == run_evals.TRAINED_GAMES_FROM_ARM_ALLOWED

    def test_the_flag_still_overrides_the_composition(self, tmp_path: Path) -> None:
        run_dir = self.write_run_dir(tmp_path, held=self.HELD)
        plan = run_evals.resolve_plan(
            run_evals._parse_args(
                ["--run-dir", str(run_dir), "--backend", "mock", "--trained-games", "chicken"]
            )
        )
        assert plan.trained_game_ids == ("chicken",)
        assert plan.trained_games_source == run_evals.TRAINED_GAMES_FROM_FLAG

    def test_the_dropped_games_records_are_marked_transfer_end_to_end(self, tmp_path: Path) -> None:
        """Through the CLI: the dropped game's rows carry trained_game=False, and the meta says why."""
        run_dir = self.write_run_dir(tmp_path, held=self.HELD)
        out_dir = tmp_path / "out"
        argv = [
            "--run-dir",
            str(run_dir),
            "--steps",
            "200",
            "--backend",
            "mock",
            "--out-dir",
            str(out_dir),
            "--games",
            f"{self.DROPPED},stag-hunt",
            "--no-include-never-trained",
            "--sections",
            SECTION_GAME_BEHAVIOR,
        ]
        assert run_evals.main(argv) == 0
        records = list(read_eval_records(out_dir / "step-200.jsonl"))
        assert records[0]["record"] == RECORD_META
        assert records[0]["trained_games_source"] == run_evals.TRAINED_GAMES_FROM_CORPUS_COMPOSITION
        marked: dict[str, set[bool]] = {}
        for record in records:
            if record["record"] != SECTION_GAME_BEHAVIOR:
                continue
            marked.setdefault(str(record["game_id"]), set()).add(bool(record["trained_game"]))
        assert marked == {self.DROPPED: {False}, "stag-hunt": {True}}


class TestTheTemplateKwargsTrainingPinnedAreDerivedNotAssumed:
    """Whatever training pinned on the chat template has to reach the battery's config.

    Qwen3.8-27B's template prepends an unauthored "Reasoning effort is set to xhigh. Please think
    carefully through the task, ..." system turn at its default, and `reasoning_effort="medium"`
    removes that turn entirely. Training pins it; a battery that did not would evaluate the 27B arm
    on different prompt text than it trained on.
    """

    def test_a_template_with_the_knob_is_pinned_for_the_battery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            run_evals.AutoTokenizer,
            "from_pretrained",
            classmethod(
                lambda _cls, *_args, **_kwargs: SimpleNamespace(
                    get_chat_template=lambda: (
                        "{%- if reasoning_effort == 'xhigh' %}...{%- endif %}"
                    ),
                    apply_chat_template=lambda *_a, **_k: "<|im_start|>user\nping<|im_end|>\n",
                )
            ),
        )
        args = run_evals._parse_args(
            ["--model", BASE_MODEL, "--arm", "twin-pd-group", "--backend", "hf"]
        )
        facts = run_evals._resolve_template_facts(args, thinking=True, template_source=BASE_MODEL)
        assert facts.chat_template_kwargs == (("reasoning_effort", "medium"),)

    def test_a_template_without_the_knob_pins_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every checkpoint below 27B, which must render exactly as it did before."""
        monkeypatch.setattr(
            run_evals.AutoTokenizer,
            "from_pretrained",
            classmethod(
                lambda _cls, *_args, **_kwargs: SimpleNamespace(
                    get_chat_template=lambda: "{%- for message in messages %}...{%- endfor %}",
                    apply_chat_template=lambda *_a, **_k: "<|im_start|>user\nping<|im_end|>\n",
                )
            ),
        )
        args = run_evals._parse_args(
            ["--model", BASE_MODEL, "--arm", "twin-pd-group", "--backend", "hf"]
        )
        facts = run_evals._resolve_template_facts(args, thinking=True, template_source=BASE_MODEL)
        assert facts.chat_template_kwargs == ()

    def test_a_hosted_backend_renders_no_template_so_pins_nothing(self) -> None:
        args = run_evals._parse_args(
            ["--model", "openai.gpt-oss-120b-1:0", "--arm", "twin-pd-group", "--backend", "bedrock"]
        )
        facts = run_evals._resolve_template_facts(
            args, thinking=True, template_source="openai.gpt-oss-120b-1:0"
        )
        assert facts.chat_template_kwargs == ()
        assert facts.prefilled_think is False


class TestEntryPointPreflight:
    """The two things `main` must settle before any model can load: allocator config and kernels."""

    def test_the_allocator_default_lands_before_anything_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare launch on a fresh box gets expandable segments without any hand export."""
        monkeypatch.delenv(preflight.CUDA_ALLOC_CONF_ENV, raising=False)
        out_dir = tmp_path / "out"
        assert (
            run_evals.main(["--model", "fake-base", "--arm", "plumbing-arm", *cli(out_dir=out_dir)])
            == 0
        )
        assert os.environ[preflight.CUDA_ALLOC_CONF_ENV] == preflight.DEFAULT_CUDA_ALLOC_CONF

    def test_an_operator_s_explicit_allocator_config_survives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(preflight.CUDA_ALLOC_CONF_ENV, "max_split_size_mb:64")
        assert preflight.default_cuda_allocator_config() == "max_split_size_mb:64"
        assert os.environ[preflight.CUDA_ALLOC_CONF_ENV] == "max_split_size_mb:64"

    def test_the_hf_backend_bridges_the_decode_kernel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dispatch freezes at the modeling module's import, so main() must bridge first."""
        calls: list[str] = []
        monkeypatch.setattr(
            run_evals, "bridge_decode_kernel", lambda: calls.append("bridge") or {"bridged": True}
        )
        monkeypatch.setattr(
            run_evals,
            "assert_bridged_kernel_matches_call_site",
            lambda: calls.append("call-site-check"),
        )
        assert run_evals._bridge_hf_decode_kernel("hf") == {"bridged": True}
        assert calls == ["bridge", "call-site-check"]

    @pytest.mark.parametrize("backend", ["vllm", "mock", "bedrock", "codex"])
    def test_no_other_backend_pays_the_fla_import(
        self, backend: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode() -> dict[str, object]:
            raise AssertionError("bridged a backend that never decodes through transformers")

        monkeypatch.setattr(run_evals, "bridge_decode_kernel", explode)
        assert run_evals._bridge_hf_decode_kernel(backend) is None

    def test_main_routes_its_backend_through_the_bridge_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pins the wiring, not just the gate: a `main` that forgets the call reads green otherwise."""
        seen: list[str] = []
        monkeypatch.setattr(run_evals, "_bridge_hf_decode_kernel", seen.append)
        out_dir = tmp_path / "out"
        assert (
            run_evals.main(["--model", "fake-base", "--arm", "plumbing-arm", *cli(out_dir=out_dir)])
            == 0
        )
        assert seen == ["mock"]


BASELINE_MIB = 900  # a neighbour already on the card, so "drained" cannot mean "empty"


class TestEngineLifecycle:
    """The card is read before an engine loads and given back after, per step of the ladder.

    This driver walks a whole checkpoint ladder inside ONE process, and until now tore each engine
    down with a private copy of exactly the pattern `games/vllm_teardown.py` was written to replace:
    `del backend` plus `torch.cuda.empty_cache()`, neither of which can reach the VRAM held by
    vLLM's own `EngineCore` subprocess. There was no pre-load baseline and no read-back of the card,
    so a teardown that freed nothing looked identical to one that worked.

    Driven with `--backend vllm` and a stubbed engine, because vLLM is not installed in this
    environment; the gate deciding which kinds read the card at all is the real one.
    """

    def _stub_engine_load(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        events: list[str],
        engine: object,
        load_mode: str = eval_model.LOAD_MODE_BASE,
    ) -> None:
        """Stub every heavyweight call, recording the order the driver makes them in.

        The order is the whole finding: a baseline read AFTER construction already includes the
        engine's own claim, so the residue comes out zero however much of the card stays held.
        """
        served = eval_model.ServedModel(model_id=BASE_MODEL, load_mode=load_mode, adapter_dir=None)
        monkeypatch.setattr(run_evals, "resolve_served_model", lambda **_kwargs: served)
        monkeypatch.setattr(
            run_evals,
            "_resolve_template_facts",
            lambda *_args, **_kwargs: run_evals.TemplateFacts(
                prefilled_think=False, chat_template_kwargs=()
            ),
        )

        def read_card() -> list[int]:
            events.append("read card")
            return [BASELINE_MIB]

        def build_engine(*_args: object, **_kwargs: object) -> object:
            events.append("build engine")
            return engine

        def battery(_backend: object, *, out_path: Path, **_kwargs: object) -> dict[str, object]:
            events.append("battery")
            # What the real battery's trace write does, and what the summary beside it needs.
            out_path.parent.mkdir(parents=True, exist_ok=True)
            return {}

        def release(backend: object, **kwargs: object) -> dict[str, object]:
            events.append(f"release {kwargs['baseline_mib']} engine={backend is engine}")
            return {}

        # The real gate, reading a stubbed card: which kinds reach nvidia-smi is part of the finding.
        monkeypatch.setattr(vllm_teardown, "vram_used_mib", read_card)
        monkeypatch.setattr(run_evals.backend_cli, "backend_from_args", build_engine)
        monkeypatch.setattr(run_evals, "run_eval_battery", battery)
        monkeypatch.setattr(run_evals, "release_engine", release)
        monkeypatch.setattr(run_evals.backend_cli, "log_token_usage", lambda _backend: None)

    def _vllm_cli(self, run_dir: Path, out_dir: Path, *extra: str) -> list[str]:
        return [
            "--run-dir",
            str(run_dir),
            "--steps",
            "5",
            "--backend",
            "vllm",
            "--out-dir",
            str(out_dir),
            "--sections",
            "game-behavior",
            "--no-report",
            *extra,
        ]

    def test_the_card_is_read_before_the_engine_and_released_after(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        engine = object()
        self._stub_engine_load(monkeypatch, events=events, engine=engine)
        run_dir = make_run_dir(tmp_path)
        assert run_evals.main(self._vllm_cli(run_dir, tmp_path / "out")) == 0
        assert events == [
            "read card",
            "build engine",
            "battery",
            f"release {[BASELINE_MIB]} engine=True",
        ]

    def test_the_engine_is_released_even_when_the_battery_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A step that dies with its engine still up holds the card against every step after it, and
        # this driver evaluates the ladder in one process -- so the release sits in a `finally`.
        events: list[str] = []
        self._stub_engine_load(monkeypatch, events=events, engine=object())

        def explode(*_args: object, **_kwargs: object) -> dict[str, object]:
            events.append("battery")
            raise RuntimeError("a step blew up")

        monkeypatch.setattr(run_evals, "run_eval_battery", explode)
        run_dir = make_run_dir(tmp_path)
        with pytest.raises(RuntimeError, match="a step blew up"):
            run_evals.main(self._vllm_cli(run_dir, tmp_path / "out"))
        assert events == [
            "read card",
            "build engine",
            "battery",
            f"release {[BASELINE_MIB]} engine=True",
        ]

    def test_the_engine_is_released_when_the_served_model_check_rejects_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # verify_served_model is the real one here: it refuses a runtime adapter served by a backend
        # that cannot prove the adapter applied, and it refuses with the engine ALREADY loaded. Above
        # the `try`, that one failure is the case where the card stays held for every later step.
        events: list[str] = []
        self._stub_engine_load(
            monkeypatch,
            events=events,
            engine=object(),
            load_mode=eval_model.LOAD_MODE_RUNTIME_ADAPTER,
        )
        run_dir = make_run_dir(tmp_path)
        with pytest.raises(TypeError, match="no way to tell an applied adapter"):
            run_evals.main(self._vllm_cli(run_dir, tmp_path / "out"))
        # No "battery": the refusal landed before a single completion was paid for, and the release
        # still ran.
        assert events == ["read card", "build engine", f"release {[BASELINE_MIB]} engine=True"]

    def test_the_mock_kind_reads_no_card_and_runs_no_vllm_release(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A card read raises on a host with no GPU, and resolve_engine_shutdown refuses a backend
        # with no EngineCore -- so a kind that loads no engine must reach neither, or every offline
        # smoke of this driver turns red.
        def refuse_card() -> list[int]:
            pytest.fail("the mock backend read the card")

        def refuse_release(*_args: object, **_kwargs: object) -> dict[str, object]:
            pytest.fail("the mock backend was put through the vLLM release")

        monkeypatch.setattr(vllm_teardown, "vram_used_mib", refuse_card)
        monkeypatch.setattr(run_evals, "release_engine", refuse_release)
        out_dir = tmp_path / "out"
        run_dir = make_run_dir(tmp_path)
        assert (
            run_evals.main(["--run-dir", str(run_dir), *cli(out_dir=out_dir), "--no-report"]) == 0
        )
        assert sorted(path.name for path in out_dir.glob("step-*.jsonl")) == [
            "step-10.jsonl",
            "step-5.jsonl",
        ]


class TestTheRuntimeFramingsFileReachesTheCellIdentity:
    """`--framings-file` supplies the sweep's off-registry framings, and its digest keys the cell.

    A cell that swept an authored framing and one that swept only registered ones are different
    measurements, so they must never share a bank entry; and the same file read from two paths is
    one measurement, so a path must not key it. Both halves are checked here because getting either
    backwards is silent: over-keying costs a regeneration, under-keying reports another cell's
    numbers under this cell's name.
    """

    def resolved(
        self, argv: list[str]
    ) -> tuple[Any, run_evals.EvalPlan, tuple[str, ...], EvalConfig]:
        args = run_evals._parse_args(argv)
        sections = run_evals._parse_sections(args.sections)
        plan = run_evals.resolve_plan(args)
        template = run_evals.TemplateFacts(prefilled_think=False, chat_template_kwargs=())
        config = run_evals._resolve_eval_config(
            args, plan=plan, sections=sections, template=template
        )
        return args, plan, sections, config

    def key_for(self, argv: list[str]) -> str:
        args, plan, sections, config = self.resolved(argv)
        return run_evals.bank_key(
            run_evals.bank_identity(args, plan, plan.targets[0], sections=sections, config=config)
        )

    def cli_line(self, out_dir: Path, *extra: str) -> list[str]:
        return [
            "--model",
            "fake-base",
            "--arm",
            DEFAULT_ARM,
            "--backend",
            "mock",
            "--out-dir",
            str(out_dir),
            "--sections",
            "framing-sweep",
            "--framing-sweep-games",
            "twin-pd",
            "--no-report",
            *extra,
        ]

    def test_the_flag_loads_the_file_and_the_framing_reaches_the_config(
        self, tmp_path: Path
    ) -> None:
        path = write_framings_file(tmp_path / "framings.json")
        _, _, _, config = self.resolved(
            self.cli_line(
                tmp_path / "out",
                "--counterpart-framings",
                f"twin,{DEPENDENT}",
                "--framings-file",
                str(path),
            )
        )
        assert config.runtime_framings is not None
        assert config.runtime_framings.path == path
        assert config.counterpart_framings == ("twin", DEPENDENT)
        assert config.as_record()["framings_digest"] == config.runtime_framings.digest

    def test_a_cell_that_swept_a_runtime_framing_is_its_own_cell(self, tmp_path: Path) -> None:
        path = write_framings_file(tmp_path / "framings.json")
        registered_only = self.key_for(
            self.cli_line(tmp_path / "out", "--counterpart-framings", "twin")
        )
        with_runtime = self.key_for(
            self.cli_line(
                tmp_path / "out",
                "--counterpart-framings",
                f"twin,{DEPENDENT}",
                "--framings-file",
                str(path),
            )
        )
        assert registered_only != with_runtime

    def test_the_same_file_at_two_paths_is_one_cell_and_an_edited_clause_is_another(
        self, tmp_path: Path
    ) -> None:
        here = write_framings_file(tmp_path / "here" / "framings.json")
        there = write_framings_file(tmp_path / "there" / "framings.json")
        edited = write_with_clause(
            tmp_path / "edited" / "framings.json",
            DEPENDENT,
            f"{CLAUSE_PREFIX}SYNTHETIC-COUNTERPART-{DEPENDENT}-revised, {BRIEFING_PHRASE}, "
            f"{DECOUPLING_TAILS[0]}",
        )
        keys = [
            self.key_for(
                self.cli_line(
                    tmp_path / "out",
                    "--counterpart-framings",
                    DEPENDENT,
                    "--framings-file",
                    str(path),
                )
            )
            for path in (here, there, edited)
        ]
        assert keys[0] == keys[1]
        assert keys[0] != keys[2]

    def test_a_missing_framings_file_is_refused_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="gitignored"):
            self.resolved(
                self.cli_line(
                    tmp_path / "out",
                    "--counterpart-framings",
                    DEPENDENT,
                    "--framings-file",
                    str(tmp_path / "absent.json"),
                )
            )
