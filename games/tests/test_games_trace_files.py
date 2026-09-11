"""Pin the two guards that stand between a relaunch and the rollout trace it could destroy.

The incident (2026-09-01, `pd-unstated-other-payoff` on the wave-3 ladder): a stage runner re-ran a
finished arm with `--resume-from-checkpoint latest --max-steps 70`. The resume landed on
`checkpoint-70`, transformers ran zero optimizer steps and still called `log()` once from
`_finalize_training`, and TRL's `GRPOTrainer.log` wrote `completions_00070.parquet` from its
freshly constructed, empty buffers -- zero rows, four columns -- over the 64-row file the real step
had written. `aws s3 sync` mirrored the smaller file over the good one, and `train_summary.json`
said `rollout_trace_complete=true` because that verdict only ever read config arithmetic.

Two guards now exist and both are sabotaged here:

*   `refuse_empty_trace_overwrite`, called from `PaddingTrimmedGRPOTrainer.log` before TRL's write:
    an empty buffer aimed at a step whose file already holds rows is refused loudly.
*   `verify_trace_files`, called where `train_summary.json` is written: one non-empty parquet per
    optimizer step with the expected row count, or the summary says which steps are missing, empty
    or short.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from games import preflight
from games import train as gt

if TYPE_CHECKING:
    from pathlib import Path

ROWS_PER_STEP = 8


def write_trace(output_dir: Path, step: int, rows: int) -> Path:
    """Write `completions_<step>.parquet` the way TRL does: a pandas frame, `rows` completions."""
    path = preflight.trace_file_path(output_dir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "step": [step] * rows,
            "prompt": [f"p{i}" for i in range(rows)],
            "completion": [f"c{i}" for i in range(rows)],
            "advantage": [0.0] * rows,
        }
    )
    frame.to_parquet(path)
    return path


def write_complete_trace(output_dir: Path, steps: int) -> None:
    for step in range(1, steps + 1):
        write_trace(output_dir, step, ROWS_PER_STEP)


class TestVerifyTraceFiles:
    def test_a_complete_trace_is_complete_and_names_no_gaps(self, tmp_path: Path) -> None:
        write_complete_trace(tmp_path, 3)
        verdict = preflight.verify_trace_files(tmp_path, steps=3, expected_rows=ROWS_PER_STEP)
        assert verdict.complete is True
        assert verdict.missing_steps == ()
        assert verdict.empty_steps == ()
        assert verdict.wrong_row_count_steps == ()
        assert verdict.steps_expected == 3

    def test_the_incident_file_an_empty_parquet_at_the_last_step_is_red(
        self, tmp_path: Path
    ) -> None:
        """Zero rows and TRL's four base columns is exactly what the relaunch wrote over step 70."""
        write_complete_trace(tmp_path, 3)
        write_trace(tmp_path, 3, rows=0)
        verdict = preflight.verify_trace_files(tmp_path, steps=3, expected_rows=ROWS_PER_STEP)
        assert verdict.complete is False
        assert verdict.empty_steps == (3,)
        assert verdict.missing_steps == ()

    def test_a_missing_step_is_red_and_named(self, tmp_path: Path) -> None:
        write_complete_trace(tmp_path, 4)
        preflight.trace_file_path(tmp_path, 2).unlink()
        verdict = preflight.verify_trace_files(tmp_path, steps=4, expected_rows=ROWS_PER_STEP)
        assert verdict.complete is False
        assert verdict.missing_steps == (2,)

    def test_a_short_file_is_red_with_the_rows_it_did_hold(self, tmp_path: Path) -> None:
        """Fewer rows than a step generates means completions were dropped on the way to disk."""
        write_complete_trace(tmp_path, 2)
        write_trace(tmp_path, 1, rows=5)
        verdict = preflight.verify_trace_files(tmp_path, steps=2, expected_rows=ROWS_PER_STEP)
        assert verdict.complete is False
        assert verdict.wrong_row_count_steps == ((1, 5),)

    def test_every_kind_of_gap_is_reported_at_once(self, tmp_path: Path) -> None:
        write_complete_trace(tmp_path, 5)
        preflight.trace_file_path(tmp_path, 1).unlink()
        write_trace(tmp_path, 3, rows=0)
        write_trace(tmp_path, 4, rows=2)
        verdict = preflight.verify_trace_files(tmp_path, steps=5, expected_rows=ROWS_PER_STEP)
        assert verdict.complete is False
        assert verdict.missing_steps == (1,)
        assert verdict.empty_steps == (3,)
        assert verdict.wrong_row_count_steps == ((4, 2),)

    def test_a_missing_completions_directory_reports_every_step_missing(
        self, tmp_path: Path
    ) -> None:
        verdict = preflight.verify_trace_files(
            tmp_path / "never-written", steps=2, expected_rows=ROWS_PER_STEP
        )
        assert verdict.complete is False
        assert verdict.missing_steps == (1, 2)

    def test_logging_less_often_expects_only_the_logged_steps_and_the_last(
        self, tmp_path: Path
    ) -> None:
        """TRL logs when `global_step % logging_steps == 0`, plus once at train end for the last."""
        for step in (2, 4, 5):
            write_trace(tmp_path, step, ROWS_PER_STEP)
        verdict = preflight.verify_trace_files(
            tmp_path, steps=5, expected_rows=ROWS_PER_STEP, logging_steps=2
        )
        assert verdict.complete is True
        assert verdict.steps_expected == 3

    def test_the_verdict_serialises_into_a_summary(self, tmp_path: Path) -> None:
        """`train_summary.json` carries it through `asdict`, so every field must be JSON-shaped."""
        import dataclasses  # noqa: PLC0415
        import json  # noqa: PLC0415

        write_complete_trace(tmp_path, 2)
        write_trace(tmp_path, 2, rows=1)
        verdict = preflight.verify_trace_files(tmp_path, steps=2, expected_rows=ROWS_PER_STEP)
        payload = json.loads(json.dumps(dataclasses.asdict(verdict)))
        assert payload["wrong_row_count_steps"] == [[2, 1]]
        assert payload["checked_dir"] == str(tmp_path / preflight.COMPLETIONS_DIRNAME)


class TestRefuseEmptyTraceOverwrite:
    def test_an_empty_buffer_aimed_at_a_step_with_rows_is_refused(self, tmp_path: Path) -> None:
        path = write_trace(tmp_path, 70, ROWS_PER_STEP)
        with pytest.raises(RuntimeError, match="refusing to overwrite") as excinfo:
            preflight.refuse_empty_trace_overwrite(path, buffered_completions=0)
        assert "completions_00070.parquet" in str(excinfo.value)
        assert "zero steps" in str(excinfo.value)

    def test_an_empty_buffer_with_no_file_yet_is_left_to_the_summary_check(
        self, tmp_path: Path
    ) -> None:
        """Nothing to destroy: TRL writes its empty file and `verify_trace_files` names the step."""
        preflight.refuse_empty_trace_overwrite(
            preflight.trace_file_path(tmp_path, 1), buffered_completions=0
        )

    def test_a_populated_buffer_over_an_existing_file_is_the_ordinary_rewrite(
        self, tmp_path: Path
    ) -> None:
        """Train-end logs the last step a second time with the same buffer; that is fine."""
        path = write_trace(tmp_path, 70, ROWS_PER_STEP)
        preflight.refuse_empty_trace_overwrite(path, buffered_completions=ROWS_PER_STEP)


class TestTheGuardSitsInFrontOfTrlsWrite:
    """`PaddingTrimmedGRPOTrainer.log` on a bare instance, with `GRPOTrainer.log` stubbed.

    The stub records whether TRL's `log` was reached: the guard has to fire BEFORE it, because the
    parquet write is the first thing after the metric averaging inside TRL's method.
    """

    def bare_trainer(
        self, output_dir: Path, *, step: int, buffered: int, log_completions: bool = True
    ) -> Any:
        trainer = gt.PaddingTrimmedGRPOTrainer.__new__(gt.PaddingTrimmedGRPOTrainer)
        trainer.accelerator = SimpleNamespace(is_main_process=True)  # pyright: ignore[reportAttributeAccessIssue]
        trainer.log_completions = log_completions  # pyright: ignore[reportAttributeAccessIssue]
        trainer.args = SimpleNamespace(output_dir=str(output_dir))  # pyright: ignore[reportAttributeAccessIssue]
        trainer.state = SimpleNamespace(global_step=step)  # pyright: ignore[reportAttributeAccessIssue]
        trainer._logs = {"prompt": deque([f"p{i}" for i in range(buffered)])}  # pyright: ignore[reportAttributeAccessIssue]
        return trainer

    def test_a_zero_step_relaunch_is_refused_before_trl_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reached: list[dict[str, float]] = []
        monkeypatch.setattr(
            gt.GRPOTrainer, "log", lambda _self, logs, _start_time=None: reached.append(logs)
        )
        write_trace(tmp_path, 70, ROWS_PER_STEP)
        trainer = self.bare_trainer(tmp_path, step=70, buffered=0)
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            trainer.log({"train_runtime": 0.0029})
        assert reached == [], "TRL's log ran and would have written the empty parquet"

    def test_an_ordinary_step_log_reaches_trl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reached: list[dict[str, float]] = []
        monkeypatch.setattr(
            gt.GRPOTrainer, "log", lambda _self, logs, _start_time=None: reached.append(logs)
        )
        write_trace(tmp_path, 70, ROWS_PER_STEP)
        trainer = self.bare_trainer(tmp_path, step=70, buffered=ROWS_PER_STEP)
        trainer.log({"reward": 0.5})
        assert reached == [{"reward": 0.5}]

    def test_with_completion_logging_off_there_is_no_write_to_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reached: list[dict[str, float]] = []
        monkeypatch.setattr(
            gt.GRPOTrainer, "log", lambda _self, logs, _start_time=None: reached.append(logs)
        )
        write_trace(tmp_path, 70, ROWS_PER_STEP)
        trainer = self.bare_trainer(tmp_path, step=70, buffered=0, log_completions=False)
        trainer.log({"reward": 0.5})
        assert reached == [{"reward": 0.5}]
