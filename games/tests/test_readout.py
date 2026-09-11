"""Pin the cross-arm readout against synthetic traces whose answers are arithmetic.

Offline and CPU-only. Every trace here is written by hand so each expected number can be computed
by eye, which is the point: the failure mode this module has to be protected from is a table that
renders cleanly and is wrong, and a fixture regenerated from the code cannot catch that.

Four checks carry most of the weight, one per way the readout could look right and lie:

- :class:`TestWindowsPoolTheRightSteps` -- a window that quietly included step 20, or dropped one
  of its two checkpoints, still prints a plausible number. The fixture gives the two windows
  opposite values and the middle step a third, so any leakage moves the answer.
- :class:`TestDenominatorsExcludeUnparsed` -- an unparsed completion scored as a defection reads as
  a real behaviour shift. Here it would drag a rate of 1.0 down to 0.8.
- :class:`TestPooledOrderDisagreementDoesNotCollide` -- the subtlest one. Pooling two checkpoints
  keys counterbalanced option pairs on (probe_id, sample_index), so without the step-scoped ids in
  `_probe_records` one checkpoint's answer silently overwrites the other's and the rate is read off
  half the window.
- :class:`TestMalformedTraces` -- a trace that cannot be attributed must raise, and a trace read
  mid-write must be excluded and bannered rather than averaged in.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from games.arms import ARMS
from games.evals import (
    RECORD_META,
    SECTION_CAPABILITIES,
    SECTION_DT_PROBES,
    SECTION_GAME_BEHAVIOR,
)
from games.prompts import DICTATOR_GAME_ID
from games.readout import (
    EXPECTED_CELLS_PER_ARM,
    build_readout,
    capability_ladder_rows,
    dictator_rows,
    discover_cells,
    dt_window_rows,
    grid_rows,
    pair_contrast_rows,
    read_cell,
    render_markdown,
    trained_ladder_rows,
    trained_window_rows,
    transfer_rows,
    write_readout,
)

if TYPE_CHECKING:
    from pathlib import Path

ALL_STEPS: tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60, 70)


def game_records(
    game_id: str,
    *,
    value: float,
    n_parsed: int,
    n_unparsed: int = 0,
    payoff_variant: str = "standard",
) -> list[dict[str, Any]]:
    """Behaviour records for one game: `n_parsed` at `value`, plus `n_unparsed` that parsed nothing."""
    field_name = "keep_fraction" if game_id == DICTATOR_GAME_ID else "coop_fraction"
    records: list[dict[str, Any]] = []
    for index in range(n_parsed + n_unparsed):
        parsed = index < n_parsed
        records.append(
            {
                "record": SECTION_GAME_BEHAVIOR,
                "game_id": game_id,
                "prompt_id": f"{game_id}-{payoff_variant}-{index}",
                "payoff_variant": payoff_variant,
                "truncated_thinking": False,
                "parsed": parsed,
                field_name: value if parsed else None,
            }
        )
    return records


def capability_records(
    *, n_correct: int, n_wrong: int, n_unparsed: int = 0
) -> list[dict[str, Any]]:
    """Arithmetic-canary records: correct, wrong, and answers that could not be parsed."""
    records: list[dict[str, Any]] = []
    for index in range(n_correct + n_wrong + n_unparsed):
        parsed = index < n_correct + n_wrong
        records.append(
            {
                "record": SECTION_CAPABILITIES,
                "item_index": index,
                "truncated_thinking": False,
                "parsed": parsed,
                "correct": index < n_correct,
            }
        )
    return records


def open_ended_records(theory: str, *, count: int) -> list[dict[str, Any]]:
    """Open-ended probe records that all named one theory."""
    return [
        {
            "record": SECTION_DT_PROBES,
            "probe_id": "open-favorite-theory",
            "source": "ours",
            "family": "open-ended",
            "kind": "open-ended",
            "sample_index": index,
            "option_order_name": "no-options",
            "option_order": [],
            "truncated_thinking": False,
            "theory": theory,
            "parsed": True,
        }
        for index in range(count)
    ]


def choice_record(
    probe_id: str, *, order_name: str, answer_index: int, edt_leaning: float
) -> dict[str, Any]:
    """One multiple-choice probe render under one option order."""
    return {
        "record": SECTION_DT_PROBES,
        "probe_id": probe_id,
        "source": "ours",
        "family": "newcomb",
        "kind": "multiple-choice",
        "sample_index": 0,
        "option_order_name": order_name,
        "option_order": [0, 1],
        "truncated_thinking": False,
        "answer_index": answer_index,
        "compatible_theories": ["EDT"] if answer_index == 0 else ["CDT"],
        "edt_leaning": edt_leaning,
        "chose_prosocial": None,
        "prosocial_option": None,
        "parsed": True,
    }


def write_trace(
    path: Path,
    *,
    arm: str,
    step: int,
    records: list[dict[str, Any]],
    git_sha: str = "synthetic",
) -> Path:
    """Write one synthetic cell, meta record first, declaring what the registry says it trained."""
    trained = ARMS[arm].game_id if arm in ARMS and step > 0 else None
    meta: dict[str, Any] = {
        "record": RECORD_META,
        "written_at": "2026-08-20T00:00:00+00:00",
        "git_sha": git_sha,
        "backend_model_id": "synthetic",
        "backend_kind": "hf",
        "sections": [SECTION_GAME_BEHAVIOR, SECTION_DT_PROBES, SECTION_CAPABILITIES],
        "eval_config": {"trained_game_ids": [trained] if trained else []},
        "arm": arm,
        "step": step,
        "grading": ARMS[arm].grading if arm in ARMS else "group-mix",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in [meta, *records]) + "\n", encoding="utf-8"
    )
    return path


def cell_path(root: Path, arm: str, step: int, *, prefix: str = "evals-synthetic") -> Path:
    """Where a cell lives under an evals root: <prefix>/<arm>/step-N.jsonl."""
    return root / prefix / arm / f"step-{step}.jsonl"


def traces_for(root: Path, arm: str) -> list[Any]:
    """Every trace one arm wrote under this root, in step order."""
    cells, _ = discover_cells(root)
    return cells[arm]


class TestWindowsPoolTheRightSteps:
    def _root(self, tmp_path: Path) -> Path:
        """Early window all cooperation, late window none, step 20 in between as the leak detector."""
        for step, value in ((0, 1.0), (10, 1.0), (20, 0.5), (60, 0.0), (70, 0.0)):
            write_trace(
                cell_path(tmp_path, "twin-pd-group", step),
                arm="twin-pd-group",
                step=step,
                records=game_records("twin-pd", value=value, n_parsed=4),
            )
        return tmp_path

    def test_pooled_windows_use_only_their_own_steps(self, tmp_path: Path):
        traces = traces_for(self._root(tmp_path), "twin-pd-group")
        (row,) = trained_window_rows("twin-pd-group", traces)
        assert row["early_steps"] == "0,10"
        assert row["late_steps"] == "60,70"
        assert row["early_rate"] == pytest.approx(1.0)
        assert row["late_rate"] == pytest.approx(0.0)
        assert row["delta"] == pytest.approx(-1.0)
        assert (row["early_n_parsed"], row["early_n_asked"]) == (8, 8)
        assert (row["late_n_parsed"], row["late_n_asked"]) == (8, 8)
        assert (row["early_n_records"], row["late_n_records"]) == (8, 8)

    def test_the_ladder_keeps_every_step_including_the_ones_no_window_reads(self, tmp_path: Path):
        traces = traces_for(self._root(tmp_path), "twin-pd-group")
        rows = trained_ladder_rows("twin-pd-group", traces)
        assert [row["step"] for row in rows] == [0, 10, 20, 60, 70]
        by_step = {row["step"]: row["rate"] for row in rows}
        assert by_step[20] == 0.5

    def test_a_window_reports_which_of_its_checkpoints_landed(self, tmp_path: Path):
        write_trace(
            cell_path(tmp_path, "twin-pd-self", 0),
            arm="twin-pd-self",
            step=0,
            records=game_records("twin-pd", value=1.0, n_parsed=4),
        )
        traces = traces_for(tmp_path, "twin-pd-self")
        (row,) = trained_window_rows("twin-pd-self", traces)
        assert row["early_steps"] == "0"
        assert row["late_steps"] == "none"
        assert row["late_rate"] is None
        assert row["delta"] is None


class TestDenominatorsExcludeUnparsed:
    def test_unparsed_records_leave_the_numerator_and_stay_in_n_asked(self, tmp_path: Path):
        for step in (0, 10):
            write_trace(
                cell_path(tmp_path, "twin-pd-group", step),
                arm="twin-pd-group",
                step=step,
                records=game_records("twin-pd", value=1.0, n_parsed=4, n_unparsed=1),
            )
        traces = traces_for(tmp_path, "twin-pd-group")
        (row,) = trained_window_rows("twin-pd-group", traces)
        # 0.8 here would mean the two unparsed completions were scored as defections.
        assert row["early_rate"] == pytest.approx(1.0)
        assert (row["early_n_parsed"], row["early_n_asked"]) == (8, 10)
        assert row["early_n_records"] == 10
        ladder = trained_ladder_rows("twin-pd-group", traces)
        assert ladder[0]["parse_failure_rate"] == pytest.approx(0.2)

    def test_a_rate_over_fewer_than_four_parsed_records_is_not_reported(self, tmp_path: Path):
        write_trace(
            cell_path(tmp_path, "twin-pd-group", 0),
            arm="twin-pd-group",
            step=0,
            records=game_records("twin-pd", value=1.0, n_parsed=3, n_unparsed=5),
        )
        traces = traces_for(tmp_path, "twin-pd-group")
        (row,) = trained_window_rows("twin-pd-group", traces)
        assert row["early_rate"] is None
        assert row["early_n_parsed"] == 3
        readout = build_readout(tmp_path)
        assert "| twin-pd | all | trained | - |" in render_markdown(readout)

    def test_canary_accuracy_is_over_answered_items_with_both_counts_shown(self, tmp_path: Path):
        write_trace(
            cell_path(tmp_path, "twin-pd-group", 0),
            arm="twin-pd-group",
            step=0,
            records=capability_records(n_correct=4, n_wrong=1, n_unparsed=5),
        )
        traces = traces_for(tmp_path, "twin-pd-group")
        (row,) = capability_ladder_rows(traces)
        assert row["accuracy"] == pytest.approx(0.8)
        assert (row["n_parsed"], row["n_items"]) == (5, 10)


class TestDictatorUsesKeepFraction:
    def test_the_dictator_arms_trained_rate_is_the_keep_fraction(self, tmp_path: Path):
        for step in (0, 10, 60, 70):
            write_trace(
                cell_path(tmp_path, "dictator", step),
                arm="dictator",
                step=step,
                records=game_records(DICTATOR_GAME_ID, value=0.75, n_parsed=6),
            )
        traces = traces_for(tmp_path, "dictator")
        (row,) = trained_window_rows("dictator", traces)
        assert row["game"] == DICTATOR_GAME_ID
        assert row["early_rate"] == pytest.approx(0.75)
        assert row["late_rate"] == pytest.approx(0.75)

    def test_dictator_rows_read_keep_fraction_for_an_arm_that_never_trained_it(
        self, tmp_path: Path
    ):
        for step in (0, 10, 60, 70):
            write_trace(
                cell_path(tmp_path, "chicken-group", step),
                arm="chicken-group",
                step=step,
                records=[
                    *game_records("chicken", value=0.5, n_parsed=8),
                    *game_records(DICTATOR_GAME_ID, value=0.4, n_parsed=6),
                ],
            )
        cells, _ = discover_cells(tmp_path)
        (row,) = dictator_rows(cells)
        assert row["arm"] == "chicken-group"
        assert row["metric"] == "keep_fraction"
        assert row["trained_on_dictator"] is False
        assert row["early_rate"] == pytest.approx(0.4)
        transfer = {
            entry["game"]: entry for entry in transfer_rows("chicken-group", cells["chicken-group"])
        }
        assert transfer[DICTATOR_GAME_ID]["metric"] == "keep_fraction"
        assert transfer[DICTATOR_GAME_ID]["early_rate"] == pytest.approx(0.4)


class TestStagRungsSplitByPayoffVariant:
    def _root(self, tmp_path: Path) -> Path:
        for step in (0, 10, 60, 70):
            write_trace(
                cell_path(tmp_path, "stag-hunt-safe-rung", step),
                arm="stag-hunt-safe-rung",
                step=step,
                records=[
                    *game_records("stag-hunt", value=1.0, n_parsed=8, payoff_variant="safe-hunt"),
                    *game_records("stag-hunt", value=0.0, n_parsed=8, payoff_variant="risky-hunt"),
                ],
            )
        return tmp_path

    def test_the_trained_rung_and_the_transfer_rungs_are_separate_rows(self, tmp_path: Path):
        traces = traces_for(self._root(tmp_path), "stag-hunt-safe-rung")
        rows = {
            row["payoff_variant"]: row for row in trained_window_rows("stag-hunt-safe-rung", traces)
        }
        assert set(rows) == {"safe-hunt", "risky-hunt"}
        assert rows["safe-hunt"]["role"] == "trained rung"
        assert rows["risky-hunt"]["role"] == "transfer rung (within-game)"
        assert rows["safe-hunt"]["early_rate"] == pytest.approx(1.0)
        assert rows["risky-hunt"]["early_rate"] == pytest.approx(0.0)

    def test_the_grid_reads_the_rung_the_arm_trained_rather_than_the_whole_game(
        self, tmp_path: Path
    ):
        cells, _ = discover_cells(self._root(tmp_path))
        (row,) = grid_rows(cells)
        assert row["trained_payoff_variant"] == "safe-hunt"
        # Pooling the two rungs would average to 0.5; the trained rung alone is 1.0.
        assert row["trained_early_rate"] == pytest.approx(1.0)

    def test_stag_hunt_is_not_also_listed_as_a_transfer_game(self, tmp_path: Path):
        traces = traces_for(self._root(tmp_path), "stag-hunt-safe-rung")
        assert "stag-hunt" not in {
            row["game"] for row in transfer_rows("stag-hunt-safe-rung", traces)
        }


class TestPooledOrderDisagreementDoesNotCollide:
    def test_two_checkpoints_of_one_item_are_two_pairs(self, tmp_path: Path):
        agreeing = [
            choice_record(
                "newcomb-classic", order_name="as-authored", answer_index=0, edt_leaning=1
            ),
            choice_record("newcomb-classic", order_name="reversed", answer_index=0, edt_leaning=1),
        ]
        disagreeing = [
            choice_record(
                "newcomb-classic", order_name="as-authored", answer_index=0, edt_leaning=1
            ),
            choice_record("newcomb-classic", order_name="reversed", answer_index=1, edt_leaning=0),
        ]
        write_trace(
            cell_path(tmp_path, "twin-pd-group", 0),
            arm="twin-pd-group",
            step=0,
            records=agreeing,
        )
        write_trace(
            cell_path(tmp_path, "twin-pd-group", 10),
            arm="twin-pd-group",
            step=10,
            records=disagreeing,
        )
        traces = traces_for(tmp_path, "twin-pd-group")
        early = dt_window_rows(traces)[0]
        # Collapsing both checkpoints onto one pair key reports 1.0, whichever trace was read last.
        assert early["order_disagreement_rate"] == pytest.approx(0.5)

    def test_theory_counts_pool_over_the_windows_records(self, tmp_path: Path):
        for step, theory in ((0, "CDT"), (10, "CDT"), (60, "EDT"), (70, "FDT")):
            write_trace(
                cell_path(tmp_path, "twin-pd-group", step),
                arm="twin-pd-group",
                step=step,
                records=open_ended_records(theory, count=4),
            )
        traces = traces_for(tmp_path, "twin-pd-group")
        early, late = dt_window_rows(traces)
        assert (early["n_CDT"], early["n_EDT"], early["n_FDT"]) == (8, 0, 0)
        assert (late["n_CDT"], late["n_EDT"], late["n_FDT"]) == (0, 4, 4)
        assert early["n_theory_parsed"] == 8


class TestIncompleteAndErrorBanners:
    def test_an_arm_the_registry_does_not_know_is_bannered_and_gets_no_trained_tables(
        self, tmp_path: Path
    ):
        for step in ALL_STEPS:
            write_trace(
                cell_path(tmp_path, "some-hosted-model", step),
                arm="some-hosted-model",
                step=step,
                records=game_records("twin-pd", value=0.5, n_parsed=16),
            )
        readout = build_readout(tmp_path)
        assert readout.status["unregistered_arms"] == ["some-hosted-model"]
        assert readout.arms["some-hosted-model"]["trained_windows"] == []
        assert readout.arms["some-hosted-model"]["trained_ladder"] == []
        # With nothing saying what it trained, every game it played reads as transfer.
        transfer = {row["game"] for row in readout.arms["some-hosted-model"]["transfer"]}
        assert transfer == {"twin-pd"}
        assert "UNREGISTERED arm `some-hosted-model`" in render_markdown(readout)

    def test_an_arm_short_of_eight_cells_is_bannered_with_its_count(self, tmp_path: Path):
        for step in (0, 10):
            write_trace(
                cell_path(tmp_path, "twin-pd-self", step),
                arm="twin-pd-self",
                step=step,
                records=game_records("twin-pd", value=0.5, n_parsed=8),
            )
        readout = build_readout(tmp_path)
        (entry,) = readout.status["incomplete_arms"]
        assert entry == {
            "arm": "twin-pd-self",
            "cells_found": 2,
            "cells_expected": EXPECTED_CELLS_PER_ARM,
            "steps": [0, 10],
        }
        assert readout.error_containing
        markdown = render_markdown(readout)
        assert "INCOMPLETE AND ERROR-CONTAINING" in markdown
        assert "INCOMPLETE arm `twin-pd-self`: 2 of 8 cells present" in markdown

    def test_a_complete_clean_arm_is_not_bannered(self, tmp_path: Path):
        for step in ALL_STEPS:
            write_trace(
                cell_path(tmp_path, "twin-pd-group", step),
                arm="twin-pd-group",
                step=step,
                records=game_records("twin-pd", value=0.5, n_parsed=16),
            )
        readout = build_readout(tmp_path)
        assert readout.status["incomplete_arms"] == []
        assert not readout.error_containing
        assert "INCOMPLETE AND ERROR-CONTAINING" not in render_markdown(readout)

    def test_a_section_past_the_parse_failure_threshold_is_bannered(self, tmp_path: Path):
        for step in ALL_STEPS:
            write_trace(
                cell_path(tmp_path, "twin-pd-group", step),
                arm="twin-pd-group",
                step=step,
                records=game_records("twin-pd", value=0.5, n_parsed=6, n_unparsed=10),
            )
        readout = build_readout(tmp_path)
        flagged = readout.status["high_parse_failure_sections"]
        assert len(flagged) == len(ALL_STEPS)
        assert flagged[0]["section"] == SECTION_GAME_BEHAVIOR
        assert flagged[0]["parse_failure_rate"] == pytest.approx(0.625)
        assert "HIGH PARSE FAILURE" in render_markdown(readout)

    def test_cells_written_at_different_git_shas_are_bannered(self, tmp_path: Path):
        for step in ALL_STEPS:
            write_trace(
                cell_path(tmp_path, "twin-pd-group", step),
                arm="twin-pd-group",
                step=step,
                records=game_records("twin-pd", value=0.5, n_parsed=16),
                git_sha="synthetic" if step else "an-older-sha",
            )
        readout = build_readout(tmp_path)
        (entry,) = readout.status["git_sha_disagreements"]
        assert entry == {"git_sha": "an-older-sha", "cells": ["twin-pd-group@0"]}
        assert readout.git_sha == "synthetic"
        assert "GIT SHA DISAGREEMENT" in render_markdown(readout)

    def test_a_cell_declaring_a_trained_game_the_registry_disagrees_with_is_noted(
        self, tmp_path: Path
    ):
        path = cell_path(tmp_path, "twin-pd-group", 10)
        write_trace(
            path,
            arm="twin-pd-group",
            step=10,
            records=game_records("twin-pd", value=0.5, n_parsed=8),
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0])
        meta["eval_config"]["trained_game_ids"] = ["chicken"]
        path.write_text("\n".join([json.dumps(meta), *lines[1:]]) + "\n", encoding="utf-8")
        readout = build_readout(tmp_path)
        (note,) = readout.status["trained_game_declaration_notes"]
        assert note["arm"] == "twin-pd-group"
        assert "['chicken']" in note["note"]


class TestMalformedTraces:
    def test_a_trace_read_mid_write_is_excluded_and_bannered(self, tmp_path: Path):
        for step in (0, 10):
            write_trace(
                cell_path(tmp_path, "twin-pd-group", step),
                arm="twin-pd-group",
                step=step,
                records=game_records("twin-pd", value=1.0, n_parsed=8),
            )
        truncated = cell_path(tmp_path, "twin-pd-group", 60)
        write_trace(
            truncated,
            arm="twin-pd-group",
            step=60,
            records=game_records("twin-pd", value=0.0, n_parsed=8),
        )
        truncated.write_text(
            truncated.read_text(encoding="utf-8") + '{"record": "game-behavi',
            encoding="utf-8",
        )
        trace, note = read_cell(truncated)
        assert trace.arm == "twin-pd-group"
        assert note is not None
        readout = build_readout(tmp_path)
        (damaged,) = readout.status["damaged_cells"]
        assert (damaged["arm"], damaged["step"]) == ("twin-pd-group", 60)
        assert readout.status["arms"]["twin-pd-group"]["steps"] == [0, 10]
        # The damaged cell's records must not reach the late window.
        (row,) = readout.arms["twin-pd-group"]["trained_windows"]
        assert row["late_steps"] == "none"
        assert "DAMAGED cell" in render_markdown(readout)

    def test_a_first_line_that_is_not_json_raises(self, tmp_path: Path):
        path = cell_path(tmp_path, "twin-pd-group", 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"record": "meta", "arm"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="first line is not JSON"):
            read_cell(path)

    def test_a_meta_record_without_an_arm_raises(self, tmp_path: Path):
        path = cell_path(tmp_path, "twin-pd-group", 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"record": RECORD_META, "step": 0}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing"):
            read_cell(path)

    def test_a_trace_that_does_not_start_with_a_meta_record_raises(self, tmp_path: Path):
        path = cell_path(tmp_path, "twin-pd-group", 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(game_records("twin-pd", value=1.0, n_parsed=1)[0]) + "\n")
        with pytest.raises(ValueError, match="does not start with a 'meta' record"):
            read_cell(path)

    def test_an_empty_trace_raises(self, tmp_path: Path):
        path = cell_path(tmp_path, "twin-pd-group", 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError, match="is empty"):
            read_cell(path)

    def test_two_cells_claiming_one_arm_and_step_raise(self, tmp_path: Path):
        for prefix in ("evals-wave1", "evals-wave2"):
            write_trace(
                cell_path(tmp_path, "twin-pd-group", 0, prefix=prefix),
                arm="twin-pd-group",
                step=0,
                records=game_records("twin-pd", value=1.0, n_parsed=8),
            )
        with pytest.raises(ValueError, match="both claim arm"):
            discover_cells(tmp_path)

    def test_an_evals_root_with_no_traces_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="nothing to read a readout off"):
            discover_cells(tmp_path)


class TestCrossArmTables:
    def _root(self, tmp_path: Path) -> Path:
        for arm, value in (("twin-pd-group", 0.25), ("twin-pd-self", 0.75)):
            steps = ALL_STEPS if arm == "twin-pd-group" else (0, 10)
            for step in steps:
                write_trace(
                    cell_path(tmp_path, arm, step),
                    arm=arm,
                    step=step,
                    records=[
                        *game_records("twin-pd", value=value, n_parsed=16),
                        *game_records("defective-coordination", value=value, n_parsed=8),
                        *capability_records(n_correct=45, n_wrong=5),
                    ],
                )
        return tmp_path

    def test_the_pair_contrast_renders_with_one_arm_incomplete(self, tmp_path: Path):
        cells, _ = discover_cells(self._root(tmp_path))
        rows = {row["measure"]: row for row in pair_contrast_rows(cells)}
        trained = rows["trained game: twin-pd"]
        assert trained["twin-pd-group_early"] == pytest.approx(0.25)
        assert trained["twin-pd-group_late"] == pytest.approx(0.25)
        assert trained["twin-pd-self_early"] == pytest.approx(0.75)
        assert trained["twin-pd-self_late"] is None
        assert trained["twin-pd-self_delta"] is None
        assert trained["twin-pd-self_n"] == "32/32 then 0/0"
        assert "canary: arithmetic accuracy" in rows
        assert "decision theory: mean_edt_leaning" in rows

    def test_the_negative_control_is_marked_in_the_transfer_table(self, tmp_path: Path):
        cells, _ = discover_cells(self._root(tmp_path))
        rows = {row["game"]: row for row in transfer_rows("twin-pd-group", cells["twin-pd-group"])}
        control = rows["defective-coordination"]
        assert control["negative_control"] is True
        assert control["eval_only_holdout"] is True

    def test_write_readout_writes_a_json_mirror_of_the_markdown_tables(self, tmp_path: Path):
        root = self._root(tmp_path)
        out_dir = tmp_path / "out"
        markdown_path, json_path = write_readout(root, out_dir)
        assert markdown_path.read_text(encoding="utf-8").startswith("# Games RL cross-arm readout")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["error_containing"] is True
        assert payload["windows"] == {"early": [0, 10], "late": [60, 70]}
        (row,) = payload["arms"]["twin-pd-group"]["trained_windows"]
        assert row["early_rate"] == pytest.approx(0.25)
        grid = {entry["arm"]: entry for entry in payload["cross_arm"]["grid"]}
        assert grid["twin-pd-self"]["cells"] == 2
        assert grid["twin-pd-self"]["trained_late_rate"] is None
