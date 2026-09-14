"""The three guards between a relaunch of a finished reward-hacking arm and the trace it could destroy.

The incident happened to the games trainer (2026-09-01, `pd-unstated-other-payoff`): a stage runner
re-ran a finished arm with `--resume-from-checkpoint latest --max-steps 70`. The resume landed on
`checkpoint-70`, transformers ran zero optimizer steps and still called `log()` once from
`_finalize_training`, and TRL's `GRPOTrainer.log` wrote `completions_00070.parquet` from its freshly
constructed, empty buffers -- zero rows, four columns -- over the 64-row file the real step had
written. `aws s3 sync` mirrored the smaller file over the good one, and `train_summary.json` said
`rollout_trace_complete=true` because that verdict only ever read config arithmetic.

`reward_hacking.train` shares the games run machinery and had every one of those exposures, with the
same relaunch-the-identical-command recovery procedure and a stage runner (`train_sequence`) that
re-runs finished stages after a reboot. The three guards games grew are wired here and each is
sabotaged:

*   `completed_run` in `_prepare_run`: a resume onto a checkpoint at `max_steps` exits ALREADY
    COMPLETE before the card is read, a trainer built, or a byte written under the run directory.
*   `refuse_empty_trace_overwrite` in `PaddingTrimmedGRPOTrainer.log`, ahead of TRL's write: an
    empty buffer aimed at a step whose file already holds rows is refused. The class is the games one,
    constructed directly (`reward_hacking.train` once carried `TraceGuardedGRPOTrainer`, a second copy
    of the same override), so these tests reach it under the name `_build_trainer` binds. The
    negative control runs TRL's own `log` on the same bare trainer and watches it write the zero-row
    file.
*   `verify_trace_files` where `train_summary.json` is written: one non-empty parquet per logged step
    with the expected row count, or the summary names the missing, empty and short steps and the run
    fails after the summary has shipped.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import torch
from test_rh_train_config import make_config, make_plan
from trl import GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

from games import preflight
from games import train as game_train
from reward_hacking import train as rh_train
from reward_hacking.train_dataset import ARM_CONTROL

# The incident's shape: the finished run's last step is step 70 of 70.
MAX_STEPS = 70
INCIDENT_STEP = MAX_STEPS


def write_trace(output_dir: Path, step: int, rows: int) -> Path:
    """Write `completions_<step>.parquet` the way TRL does: a pandas frame, `rows` completions."""
    path = preflight.trace_file_path(output_dir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "step": [step] * rows,
            "prompt": [f"p{i}" for i in range(rows)],
            "completion": [f"c{i}" for i in range(rows)],
            "advantage": [0.0] * rows,
        }
    ).to_parquet(path)
    return path


def seed_checkpoint(output_dir: Path, step: int) -> Path:
    """Lay down `checkpoint-<step>` with every file a resume needs.

    `trainer_state.json` carries the step, because that is what both the resume and the
    already-complete check read; every other file is empty, since the completeness gate reads names.
    """
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / game_train.TRAINER_STATE_FILENAME).write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    for name in (*game_train.REQUIRED_CHECKPOINT_FILES, game_train.INIT_ADAPTER_WEIGHTS_FILENAME):
        if name != game_train.TRAINER_STATE_FILENAME:
            if name == game_train.ADAPTER_CONFIG_FILENAME:
                (checkpoint / name).write_text(json.dumps({"peft_type": "LORA"}))
            else:
                (checkpoint / name).write_bytes(b"state")
    return checkpoint


def snapshot_tree(root: Path) -> dict[str, bytes]:
    """Every file under `root` with its bytes, so a test can assert that nothing at all changed."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class TestTheGuardSitsInFrontOfTrlsWrite:
    """`PaddingTrimmedGRPOTrainer.log` on a bare instance carrying exactly what the two `log`s read.

    The positive cases stub `GRPOTrainer.log` to record whether TRL's method was reached: the guard
    has to fire BEFORE it. The negative control does not stub anything: it hands the same bare
    trainer to TRL's real `GRPOTrainer.log` and watches the parquet write the guard exists to stop.
    """

    ROWS_PER_STEP = 8

    def bare_trainer(
        self, output_dir: Path, *, step: int, buffered: int, log_completions: bool = True
    ) -> Any:
        """A trainer with no model: the attributes `Trainer.log` and `GRPOTrainer.log` read, nothing else."""
        trainer = rh_train.PaddingTrimmedGRPOTrainer.__new__(rh_train.PaddingTrimmedGRPOTrainer)
        trainer.accelerator = SimpleNamespace(is_main_process=True)  # pyright: ignore[reportAttributeAccessIssue]
        trainer.log_completions = log_completions  # pyright: ignore[reportAttributeAccessIssue]
        trainer.num_completions_to_print = 0  # pyright: ignore[reportAttributeAccessIssue]
        trainer.log_unique_prompts = False  # pyright: ignore[reportAttributeAccessIssue]
        trainer.model = SimpleNamespace(training=True)  # pyright: ignore[reportAttributeAccessIssue]
        trainer._metrics = {"train": {}}  # pyright: ignore[reportAttributeAccessIssue]
        trainer._logs = {  # pyright: ignore[reportAttributeAccessIssue]
            "prompt": deque(f"p{i}" for i in range(buffered)),
            "completion": deque(f"c{i}" for i in range(buffered)),
            "rewards": {},
            "advantages": deque([0.0] * buffered),
            "extra": {},
            "images": deque(),
        }
        trainer.args = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
            output_dir=str(output_dir), include_num_input_tokens_seen="no", report_to=[]
        )
        trainer.state = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
            global_step=step, epoch=None, log_history=[], num_input_tokens_seen=0
        )
        trainer.control = None  # pyright: ignore[reportAttributeAccessIssue]
        trainer.callback_handler = SimpleNamespace(on_log=lambda *_args: None)  # pyright: ignore[reportAttributeAccessIssue]
        return trainer

    def stub_trl_log(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, float]]:
        reached: list[dict[str, float]] = []
        monkeypatch.setattr(
            GRPOTrainer, "log", lambda _self, logs, _start_time=None: reached.append(logs)
        )
        return reached

    def test_a_zero_step_relaunch_is_refused_before_trl_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reached = self.stub_trl_log(monkeypatch)
        path = write_trace(tmp_path, INCIDENT_STEP, self.ROWS_PER_STEP)
        trainer = self.bare_trainer(tmp_path, step=INCIDENT_STEP, buffered=0)
        with pytest.raises(RuntimeError, match="refusing to overwrite") as excinfo:
            trainer.log({"train_runtime": 0.0029})
        assert reached == [], "TRL's log ran and would have written the empty parquet"
        assert "completions_00070.parquet" in str(excinfo.value)
        assert preflight.parquet_row_count(path) == self.ROWS_PER_STEP

    def test_an_ordinary_step_log_reaches_trl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Train-end logs the last step a second time with the same full buffer; that is fine."""
        reached = self.stub_trl_log(monkeypatch)
        write_trace(tmp_path, INCIDENT_STEP, self.ROWS_PER_STEP)
        trainer = self.bare_trainer(tmp_path, step=INCIDENT_STEP, buffered=self.ROWS_PER_STEP)
        trainer.log({"reward": 0.5})
        assert reached == [{"reward": 0.5}]

    def test_with_completion_logging_off_there_is_no_write_to_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reached = self.stub_trl_log(monkeypatch)
        write_trace(tmp_path, INCIDENT_STEP, self.ROWS_PER_STEP)
        trainer = self.bare_trainer(tmp_path, step=INCIDENT_STEP, buffered=0, log_completions=False)
        trainer.log({"reward": 0.5})
        assert reached == [{"reward": 0.5}]

    def test_without_the_guard_trls_own_log_writes_the_zero_row_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The negative control, on TRL's real `log` rather than a stand-in.

        Reaching `GRPOTrainer.log` directly on the same bare trainer is the unguarded path -- what
        every `log` call was before the override -- and it replaces the eight-row file with a
        zero-row one, exactly the incident's artifact. Through the override, the same call is refused
        and the file is untouched. The two halves together are what make "the guard is load-bearing"
        a measurement rather than a reading of the source.
        """
        path = write_trace(tmp_path, INCIDENT_STEP, self.ROWS_PER_STEP)
        trainer = self.bare_trainer(tmp_path, step=INCIDENT_STEP, buffered=0)

        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            trainer.log({"train_runtime": 0.0029})
        assert preflight.parquet_row_count(path) == self.ROWS_PER_STEP

        GRPOTrainer.log(trainer, {"train_runtime": 0.0029})
        assert preflight.parquet_row_count(path) == 0
        assert trainer.state.log_history == [{"train_runtime": 0.0029, "step": INCIDENT_STEP}]
        # TRL renders its completions panel (empty, at num_completions_to_print=0) on the way past.
        capsys.readouterr()

    def test_build_trainer_constructs_the_trimmed_and_guarded_class(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The override only protects a run if `_build_trainer` instantiates the games class.

        Construction is tripped in `__init__`, after TRL's arguments have all been assembled and
        before any weights load, so the test proves the class the real build reaches for -- the one
        whose `log` is the guard and whose `_prepare_inputs` is the padding trim.
        """
        constructed: list[type[GRPOTrainer]] = []

        def trip(self_: GRPOTrainer, *_args: object, **_kwargs: object) -> None:
            constructed.append(type(self_))
            raise RuntimeError("tripwire: trainer construction reached")

        monkeypatch.setattr(rh_train.PaddingTrimmedGRPOTrainer, "__init__", trip)
        config = make_config(smoke=True, output_dir=str(tmp_path / "run"))
        prepared = fake_prepared_run(config)
        with pytest.raises(RuntimeError, match="tripwire"):
            rh_train._build_trainer(prepared)
        assert constructed == [rh_train.PaddingTrimmedGRPOTrainer]


def fake_prepared_run(
    config: rh_train.RewardHackingTrainConfig, *, resume_checkpoint: str | None = None
) -> rh_train.PreparedRun:
    """A `PreparedRun` with no weights behind it: what `train_arm` reads after `_prepare_run`."""
    return rh_train.PreparedRun(
        config=config,
        plan=make_plan(),
        dataset=cast("Any", [None] * 4),
        tokenizer=cast("Any", SimpleNamespace(pad_token_id=0, eos_token_id=1)),
        lora_targets={"expected_linear_attention_layers": 24, "target_modules": ["q_proj"]},
        derived={"prefilled_think": True, "meta_parameter_count": 1},
        dtype=torch.float32,
        device={},
        resume_checkpoint=resume_checkpoint,
    )


class FinishedRun:
    """A run directory holding everything a finished arm leaves behind, plus the launch that relaunches it."""

    def __init__(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object
    ) -> None:
        self.output_dir = tmp_path / "run"
        settings: dict[str, object] = {
            "smoke": True,
            "max_steps": MAX_STEPS,
            "resume_from_checkpoint": rh_train.RESUME_LATEST,
            "output_dir": str(self.output_dir),
            "grader_scratch_root": str(tmp_path / "grader-scratch"),
        }
        self.config = make_config(**(settings | overrides))
        self.trainers_built: list[rh_train.PreparedRun] = []
        self.card_reads: list[str] = []
        monkeypatch.setattr(rh_train, "assert_jail_usable", lambda **_kwargs: {"stubbed": True})
        monkeypatch.setattr(
            rh_train, "assert_resume_provenance_matches", lambda *_a, **_k: {"stubbed": True}
        )
        monkeypatch.setattr(rh_train, "_build_trainer", self.trainers_built.append)

        def trip_at_the_card() -> dict[str, object]:
            self.card_reads.append("describe_device")
            raise RuntimeError("tripwire: the launch reached the card")

        monkeypatch.setattr(rh_train, "describe_device", trip_at_the_card)

    def seed(self, *, step: int, recorded_config: dict[str, object] | None = None) -> Path:
        """The finished run's artifacts: launch record, checkpoint, trace and summary."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rh_train.write_json(
            self.output_dir / rh_train.RUN_CONFIG_FILENAME,
            {
                "config": recorded_config if recorded_config is not None else asdict(self.config),
                "sizing_plan": {"micro_batch_size": 1},
                "git_sha": "recorded-sha",
            },
        )
        checkpoint = seed_checkpoint(self.output_dir, step)
        for trace_step in range(1, step + 1):
            write_trace(self.output_dir, trace_step, 8)
        (self.output_dir / "mem_log.csv").write_text("step,rss_gib\n70,12.0\n", encoding="utf-8")
        rh_train.write_json(
            self.output_dir / rh_train.TRAIN_SUMMARY_FILENAME,
            {"steps_completed": step, "wall_clock_seconds": 117_810.0},
        )
        return checkpoint

    def relaunch(self) -> GRPOTrainer | None:
        return rh_train.train_arm(self.config)


class TestARelaunchOfAFinishedRunTouchesNothing:
    """The incident, offline and on this trainer: `--resume-from-checkpoint latest` at max_steps.

    The relaunch is recognised before the trainer exists, and the assertion is byte-level: the run
    directory after the relaunch is identical to the run directory before it. `describe_device` is a
    tripwire, because it is the first thing `_prepare_run` does after the already-complete check --
    a launch that reaches it has been let through.
    """

    def test_a_resume_landing_on_max_steps_builds_no_trainer_and_changes_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        run = FinishedRun(tmp_path, monkeypatch)
        run.seed(step=MAX_STEPS)
        before = snapshot_tree(run.output_dir)
        with caplog.at_level(logging.INFO, logger=rh_train.logger.name):
            result = run.relaunch()
        assert result is None
        assert run.trainers_built == [], "a trainer was built for nothing"
        assert run.card_reads == []
        assert snapshot_tree(run.output_dir) == before
        assert not list(run.output_dir.glob("run_config.resume-from-*.json"))
        assert "ALREADY COMPLETE" in caplog.text
        assert (
            f"checkpoint-{MAX_STEPS} holds step {MAX_STEPS} of max_steps={MAX_STEPS}" in caplog.text
        )

    def test_a_checkpoint_past_max_steps_is_complete_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator who lowered --max-steps on relaunch has nothing to train either."""
        run = FinishedRun(tmp_path, monkeypatch)
        run.seed(step=MAX_STEPS + 1)
        before = snapshot_tree(run.output_dir)
        assert run.relaunch() is None
        assert snapshot_tree(run.output_dir) == before

    def test_a_checkpoint_below_max_steps_still_resumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate keys on the step: an unfinished run goes on to read the card as before."""
        run = FinishedRun(tmp_path, monkeypatch)
        run.seed(step=MAX_STEPS - 1)
        with pytest.raises(RuntimeError, match="tripwire: the launch reached the card"):
            run.relaunch()
        assert run.card_reads == ["describe_device"]

    def test_a_finished_run_whose_summary_never_landed_is_refused_not_declared_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Killed between the final save and the summary write: the checkpoints are the real output,
        and a zero-step relaunch would manufacture a wrong summary, so the launch refuses instead."""
        run = FinishedRun(tmp_path, monkeypatch)
        run.seed(step=MAX_STEPS)
        (run.output_dir / rh_train.TRAIN_SUMMARY_FILENAME).unlink()
        before = snapshot_tree(run.output_dir)
        with pytest.raises(RuntimeError, match="does not exist") as excinfo:
            run.relaunch()
        assert rh_train.TRAIN_SUMMARY_FILENAME in str(excinfo.value)
        assert "train zero steps" in str(excinfo.value)
        assert run.trainers_built == []
        assert snapshot_tree(run.output_dir) == before

    def test_the_identity_check_still_comes_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relaunch under the wrong arm onto a finished run is refused as a mismatch, not waved
        through as already complete: the mismatch is the more useful thing to know."""
        run = FinishedRun(tmp_path, monkeypatch)
        recorded: dict[str, object] = {**asdict(run.config), "arm": ARM_CONTROL}
        run.seed(step=MAX_STEPS, recorded_config=recorded)
        with pytest.raises(RuntimeError, match="refusing to resume"):
            run.relaunch()

    def test_with_an_s3_destination_the_restore_runs_and_nothing_is_uploaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The documented recovery pulls the run directory from the bucket first, and that stays:
        the ALREADY COMPLETE exit is after the restore and before every upload."""
        restored: list[tuple[str, Path]] = []
        uploaded: list[object] = []
        monkeypatch.setattr(
            rh_train,
            "restore_directory",
            lambda s3_dest, local_dir: (
                restored.append((s3_dest, local_dir)) or SimpleNamespace(returncode=0)
            ),
        )
        monkeypatch.setattr(rh_train, "sync_directory", lambda *args, **_k: uploaded.append(args))
        run = FinishedRun(tmp_path, monkeypatch, smoke=False)
        run.seed(step=MAX_STEPS)
        before = snapshot_tree(run.output_dir)
        assert run.relaunch() is None
        assert restored == [(run.config.s3_dest, run.output_dir)]
        assert uploaded == []
        assert snapshot_tree(run.output_dir) == before


class StubTrainer:
    """What `train_arm` calls on the trainer TRL would have built, writing the trace TRL would write."""

    # A test sabotages the trace by overriding which steps get a file and with how many rows.
    trace_steps: tuple[int, ...] = tuple(range(1, MAX_STEPS + 1))
    trace_rows: dict[int, int] = {}  # noqa: RUF012 - a class-level sabotage knob, monkeypatched per test

    def __init__(self, output_dir: Path, *, rows_per_step: int) -> None:
        self.output_dir = output_dir
        self.rows_per_step = rows_per_step
        self.state = SimpleNamespace(global_step=0, log_history=[])
        self.resumed_from: str | None = None
        self.saved_model = False
        self.saved_state = False

    def train(self, *, resume_from_checkpoint: str | None = None) -> None:
        self.resumed_from = resume_from_checkpoint
        for step in self.trace_steps:
            write_trace(self.output_dir, step, self.trace_rows.get(step, self.rows_per_step))
        self.state.global_step = MAX_STEPS

    def save_model(self) -> None:
        self.saved_model = True

    def save_state(self) -> None:
        self.saved_state = True


class TrainingHarness:
    """`train_arm` past a stubbed `_prepare_run` and `_build_trainer`, on the CPU, in a moment.

    Everything that needs a card or a model is stubbed at the seam `train_arm` calls it through;
    everything from the trainer's `train()` on -- the trace files, the summary, the sync, the
    raises -- is the real code path. `config_trace_complete` is the verdict the build-time
    (config-arithmetic) check hands the summary.
    """

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.output_dir = tmp_path / "run"
        self.trainers: list[StubTrainer] = []
        # One entry per upload, recording whether the summary was on disk at that moment.
        self.synced: list[bool] = []

    def run(self, *, config_trace_complete: bool = True, **overrides: object) -> StubTrainer:
        config = make_config(
            max_steps=MAX_STEPS,
            output_dir=str(self.output_dir),
            grader_scratch_root=str(self.tmp_path / "grader-scratch"),
            **overrides,
        )
        prepared = fake_prepared_run(config)
        patch = self.monkeypatch.setattr
        patch(rh_train, "_prepare_run", lambda _config, *, kernel_bridge: prepared)
        patch(rh_train, "_build_trainer", self.build_trainer)
        patch(
            rh_train,
            "check_built_trainer",
            lambda _trainer, **_kwargs: {
                "lora": {},
                "rollout_trace_complete": config_trace_complete,
                "resolved_sampler": {},
            },
        )
        patch(rh_train.torch.cuda, "reset_peak_memory_stats", lambda: None)
        patch(
            rh_train,
            "read_back_metrics",
            lambda _trainer, *, required, constant_by_construction: ({"reward": 1.0}, []),
        )
        patch(rh_train, "peak_memory_gib", lambda _output_dir: {})
        patch(rh_train, "sync_directory", self.record_sync)
        rh_train.train_arm(config)
        return self.trainers[-1]

    def build_trainer(self, prepared: rh_train.PreparedRun) -> StubTrainer:
        trainer = StubTrainer(
            Path(prepared.output_dir), rows_per_step=prepared.plan.episodes_per_step
        )
        self.trainers.append(trainer)
        return trainer

    def record_sync(self, local_dir: Path, _s3_dest: str, **_kwargs: object) -> None:
        self.synced.append((local_dir / rh_train.TRAIN_SUMMARY_FILENAME).is_file())

    def summary(self) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            json.loads(
                (self.output_dir / rh_train.TRAIN_SUMMARY_FILENAME).read_text(encoding="utf-8")
            ),
        )


@pytest.fixture
def training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TrainingHarness:
    return TrainingHarness(tmp_path, monkeypatch)


class TestTheSummaryReadsTheTraceFilesItClaims:
    """`rollout_trace_complete` used to be config arithmetic; the incident's summary said true over a
    zero-row file. It now also reads the completions directory, and a gap fails the run."""

    def test_a_healthy_run_reports_the_trace_complete_from_its_files(
        self, training: TrainingHarness
    ) -> None:
        trainer = training.run(smoke=True)
        summary = training.summary()
        assert trainer.saved_model
        assert trainer.saved_state
        assert summary["rollout_trace_complete"] is True
        assert summary["rollout_trace_config_complete"] is True
        files = summary["rollout_trace_files"]
        assert files["steps_expected"] == MAX_STEPS
        assert files["expected_rows_per_step"] == summary["episodes_per_step"]
        assert files["missing_steps"] == []
        assert files["empty_steps"] == []
        assert files["wrong_row_count_steps"] == []
        assert files["checked_dir"] == str(training.output_dir / preflight.COMPLETIONS_DIRNAME)

    def test_the_incident_file_turns_the_summary_red_and_fails_the_run(
        self, training: TrainingHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`completions_00070.parquet` lands with zero rows: the summary says so, then the run fails."""
        monkeypatch.setattr(StubTrainer, "trace_rows", {INCIDENT_STEP: 0})
        with pytest.raises(RuntimeError, match="rollout trace") as excinfo:
            training.run(smoke=True)
        assert f"empty files at steps [{INCIDENT_STEP}]" in str(excinfo.value)
        summary = training.summary()
        assert summary["rollout_trace_complete"] is False
        assert summary["rollout_trace_config_complete"] is True
        assert summary["rollout_trace_files"]["empty_steps"] == [INCIDENT_STEP]
        assert summary["rollout_trace_files"]["missing_steps"] == []
        incident_file = preflight.trace_file_path(training.output_dir, INCIDENT_STEP)
        assert incident_file.name == "completions_00070.parquet"
        assert preflight.parquet_row_count(incident_file) == 0

    def test_a_missing_step_and_a_short_file_are_named_with_their_rows(
        self, training: TrainingHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            StubTrainer, "trace_steps", tuple(step for step in range(1, MAX_STEPS + 1) if step != 3)
        )
        monkeypatch.setattr(StubTrainer, "trace_rows", {5: 3})
        with pytest.raises(
            RuntimeError, match=r"missing steps \[3\].*wrong row counts \(step, rows\) \[\[5, 3\]\]"
        ):
            training.run(smoke=True)
        files = training.summary()["rollout_trace_files"]
        assert files["missing_steps"] == [3]
        assert files["wrong_row_count_steps"] == [[5, 3]]

    def test_the_config_verdict_still_gates_the_combined_one(
        self, training: TrainingHarness
    ) -> None:
        """Files present and full, but TRL's derived batch sizes say the trace cannot be complete:
        the combined verdict is false and both halves are recorded, so a reader sees which failed.
        No raise, as before: the file check is the one that fails a run."""
        training.run(smoke=True, config_trace_complete=False)
        summary = training.summary()
        assert summary["rollout_trace_config_complete"] is False
        assert summary["rollout_trace_complete"] is False
        assert summary["rollout_trace_files"]["empty_steps"] == []

    def test_the_red_summary_is_shipped_before_the_run_fails(
        self, training: TrainingHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same ordering as missing metrics: the summary naming the gap is the artifact worth having
        off-box, so the final sync runs before the raise. A real arm carries an S3 destination."""
        monkeypatch.setattr(StubTrainer, "trace_rows", {INCIDENT_STEP: 0})
        with pytest.raises(RuntimeError, match="rollout trace"):
            training.run()
        assert training.synced == [True]
