"""Offline tests for staged GPU orchestration. No GPU, no real sleeping, no real waiting.

Each test here corresponds to a bug that actually happened tonight in bash, which is the reason
this logic was moved into Python at all. The false-green case is the important one: a command that
exits 0 having produced nothing must not be called a success.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

from games import arm_sequence as plan
from games import stage_runner as sr

if TYPE_CHECKING:
    from pathlib import Path

LIVE_PLAN_MODULE = "games.arm_sequence"


def stage(name: str = "probe", *, argv: tuple[str, ...] = ("true",), **kwargs: object) -> sr.Stage:
    """Build a Stage with a harmless default command."""
    return sr.Stage(name=name, argv=argv, **kwargs)  # pyright: ignore[reportArgumentType]


class TestStageValidation:
    def test_a_nameless_stage_is_refused(self):
        with pytest.raises(ValueError, match="needs a name"):
            sr.Stage(name="", argv=("true",))

    def test_a_stage_with_no_command_is_refused(self):
        with pytest.raises(ValueError, match="no command to run"):
            sr.Stage(name="probe", argv=())


class TestExitCodeIsReadCorrectly:
    """Bug 1 tonight: `if ! cmd` inverts the status, so `$?` reported 0 for a failing stage."""

    def test_a_failing_command_is_not_ok(self):
        result = sr.run_stage(stage("fails", argv=("false",)))
        assert result.returncode != 0
        assert result.ok is False
        assert "FAILED" in result.describe()

    def test_a_succeeding_command_with_nothing_to_verify_is_ok(self):
        result = sr.run_stage(stage("succeeds"))
        assert result.returncode == 0
        assert result.ok is True
        assert result.describe().startswith("OK")


class TestAKilledStageIsDistinguishableFromACrashedOne:
    """ "The cap was too short" and "the arm died" need opposite responses, so they must read apart.

    Every GPU stage in every plan here is wrapped in `timeout <cap>`, and this repo has already
    committed a cap that sat under a healthy run time. An operator reading `exit 124` has to know
    that the fix is to raise the cap and re-run to resume, not to debug a failure.
    """

    def test_a_stage_the_cap_killed_says_the_cap_killed_it(self, tmp_path: Path):
        """Run a real `timeout`, so this measures the exit code the plans will actually produce."""
        result = sr.run_stage(
            stage(
                "capped",
                argv=("timeout", "0.2", "sleep", "5"),
                log_path=tmp_path / "logs" / "capped.log",
            )
        )
        assert result.returncode == sr.WALL_CLOCK_CAP_RETURNCODE
        assert result.ok is False
        described = result.describe()
        assert "wall-clock cap" in described
        assert "raise the cap" in described

    def test_an_ordinary_crash_still_reads_as_a_crash(self):
        result = sr.run_stage(stage("crashes", argv=("bash", "-c", "exit 3")))
        assert "exit 3" in result.describe()
        assert "wall-clock cap" not in result.describe()

    def test_a_stage_that_never_ran_carries_the_no_exit_code_sentinel(self):
        outcome = sr.run_sequence([stage("boom", argv=("false",)), stage("later")])
        skipped = outcome.results[1]
        assert skipped.skipped is True
        assert skipped.returncode == sr.NOT_RUN_RETURNCODE


class TestArtifactVerification:
    """Bug 3 tonight, and the reason this module exists: exit 0 is not evidence of output."""

    def test_a_command_that_exits_zero_without_writing_its_artifact_is_a_failure(
        self, tmp_path: Path
    ):
        promised = tmp_path / "never-written.json"
        result = sr.run_stage(stage("liar", artifacts=(promised,)))
        assert result.returncode == 0
        assert result.ok is False
        assert result.missing_artifacts == (str(promised),)
        assert "exited 0 but produced nothing" in result.describe()

    def test_an_empty_artifact_counts_as_missing(self, tmp_path: Path):
        # The first chain attempt wrote a 0-byte corpus and the next stage consumed it.
        empty = tmp_path / "empty.jsonl"
        empty.touch()
        result = sr.run_stage(stage("empty-output", artifacts=(empty,)))
        assert result.ok is False
        assert result.missing_artifacts == (str(empty),)

    def test_a_written_artifact_passes(self, tmp_path: Path):
        written = tmp_path / "real.json"
        result = sr.run_stage(
            stage("writer", argv=("bash", "-c", f"echo hello > {written}"), artifacts=(written,))
        )
        assert result.ok is True
        assert result.missing_artifacts == ()

    def test_every_promised_artifact_is_checked_not_just_the_first(self, tmp_path: Path):
        present = tmp_path / "present.json"
        absent = tmp_path / "absent.json"
        result = sr.run_stage(
            stage(
                "partial",
                argv=("bash", "-c", f"echo x > {present}"),
                artifacts=(present, absent),
            )
        )
        assert result.ok is False
        assert result.missing_artifacts == (str(absent),)


class TestStageLogsReachTheRunDirectory:
    """A stage log written only to the shared log directory is a log nothing ships.

    `games/s3_sync.py` and `cloud/entrypoint.sh` both sync the *run* directory wholesale, and no
    repo code syncs `artifacts/games/logs/`. Box operators were hand-rolling
    `aws s3 sync artifacts/games/logs/ ...` to compensate, which is how a run whose only surviving
    record was a log-only sync happened at all: the rollout parquet stayed on the instance and the
    log came back. So a stage names the run directories that must carry a copy of its log, and the
    copies are written as the output arrives rather than at the end -- the run that loses its log is
    the one that never reaches an end.
    """

    def test_the_run_directory_copy_holds_exactly_what_the_shared_log_holds(self, tmp_path: Path):
        shared = tmp_path / "logs" / "probe.log"
        run_dir = tmp_path / "runs" / "arm"
        result = sr.run_stage(
            stage(
                "chatty",
                argv=("bash", "-c", "echo to-stdout; echo to-stderr >&2"),
                log_path=shared,
                log_run_dirs=(run_dir,),
            )
        )
        assert result.ok is True
        copy = run_dir / "logs" / "probe.log"
        assert copy.read_text() == shared.read_text()
        assert "to-stdout" in copy.read_text()
        assert "to-stderr" in copy.read_text(), "stderr is merged in, as it was before the copies"

    def test_one_stream_reaches_every_named_run_directory(self, tmp_path: Path):
        # The twin-PD sweep serves both arms, so its log is provenance for both run directories.
        shared = tmp_path / "logs" / "sweep.log"
        arms = (tmp_path / "runs" / "group", tmp_path / "runs" / "self")
        sr.run_stage(
            stage(
                "sweep",
                argv=("bash", "-c", "echo selected 17 prompts"),
                log_path=shared,
                log_run_dirs=arms,
            )
        )
        for run_dir in arms:
            assert (run_dir / "logs" / "sweep.log").read_text() == shared.read_text()

    def test_output_larger_than_one_read_arrives_whole(self, tmp_path: Path):
        # A real arm log is megabytes. A copy that only ever saw one short line would pass on a
        # single-chunk stream and silently truncate every run after it.
        shared = tmp_path / "logs" / "long.log"
        run_dir = tmp_path / "runs" / "arm"
        lines = 20_000
        sr.run_stage(
            stage(
                "verbose",
                argv=("bash", "-c", f"seq 1 {lines}"),
                log_path=shared,
                log_run_dirs=(run_dir,),
            )
        )
        copy = run_dir / "logs" / "long.log"
        assert copy.read_bytes() == shared.read_bytes()
        assert copy.read_text().splitlines()[-1] == str(lines)

    def test_the_exit_code_still_comes_from_the_command(self, tmp_path: Path):
        # The whole module exists because a wrapper reported someone else's status as the stage's.
        result = sr.run_stage(
            stage(
                "fails",
                argv=("bash", "-c", "echo dying; exit 3"),
                log_path=tmp_path / "logs" / "fails.log",
                log_run_dirs=(tmp_path / "runs" / "arm",),
            )
        )
        assert result.returncode == 3
        assert result.ok is False
        assert "dying" in (tmp_path / "runs" / "arm" / "logs" / "fails.log").read_text()

    def test_a_stage_naming_run_directories_without_a_log_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no log to copy"):
            sr.Stage(name="mute", argv=("true",), log_run_dirs=(tmp_path / "runs" / "arm",))

    def test_a_lone_log_path_still_works_on_its_own(self, tmp_path: Path):
        # Run-directory copies are additive: a stage with no run of its own keeps its single log.
        shared = tmp_path / "logs" / "solo.log"
        result = sr.run_stage(stage("solo", argv=("echo", "hello"), log_path=shared))
        assert result.ok is True
        assert shared.read_text().splitlines()[-1] == "hello"
        assert sr.Stage(name="solo", argv=("true",), log_path=shared).log_destinations() == (
            shared,
        )

    def test_a_stage_with_no_log_at_all_names_no_destinations(self):
        assert stage("quiet").log_destinations() == ()

    def test_a_second_attempt_keeps_the_first_attempt_s_output(self, tmp_path: Path):
        """Re-running the identical command IS the documented recovery procedure.

        Every log path is deterministic per arm and model, so opening it for truncation destroys the
        only record of why the previous attempt died -- and the next checkpoint sync replaces the S3
        copy with the truncated one. Both destinations are checked, since both are truncated.
        """
        shared = tmp_path / "logs" / "arm.log"
        run_dir = tmp_path / "runs" / "arm"
        for attempt in ("first attempt died", "second attempt speaking"):
            sr.run_stage(
                stage(
                    "arm",
                    argv=("bash", "-c", f"echo {attempt!r}"),
                    log_path=shared,
                    log_run_dirs=(run_dir,),
                )
            )
        for path in (shared, run_dir / "logs" / "arm.log"):
            written = path.read_text()
            assert "first attempt died" in written, path
            assert "second attempt speaking" in written, path
            assert written.count(sr.ATTEMPT_DELIMITER) == 2, path

    def test_each_attempt_is_delimited_by_what_it_ran(self, tmp_path: Path):
        """Appending without a marker would read as one run whose output contradicts itself."""
        shared = tmp_path / "logs" / "arm.log"
        sr.run_stage(stage("arm", argv=("echo", "hello"), log_path=shared))
        header = shared.read_text().splitlines()[0]
        assert sr.ATTEMPT_DELIMITER in header
        assert "echo hello" in header


class TestGpuWaiting:
    """Bug 2 tonight: two GPU stages raced one card and both died in the same second."""

    def test_it_returns_immediately_when_the_card_is_free(self):
        slept: list[float] = []
        assert sr.wait_for_free_gpu(probe=lambda: 0, sleep=slept.append) is True
        assert slept == []

    def test_it_waits_while_vram_is_held_then_proceeds(self):
        readings = iter([9000, 9000, 100])
        slept: list[float] = []
        assert sr.wait_for_free_gpu(probe=lambda: next(readings), sleep=slept.append) is True
        assert len(slept) == 2

    def test_a_few_held_mib_still_counts_as_free(self):
        # A driver or another process can hold a little without preventing a run.
        assert sr.wait_for_free_gpu(probe=lambda: 100, sleep=lambda _s: None) is True

    def test_the_timeout_is_a_failure_not_a_shrug(self):
        # Proceeding anyway is exactly what idled the card: the stage started, its preflight
        # refused in two seconds, and the sequence called it done.
        assert (
            sr.wait_for_free_gpu(probe=lambda: 9000, timeout_seconds=0.0, sleep=lambda _s: None)
            is False
        )

    def test_a_gpu_stage_that_never_gets_the_card_fails_without_running(self):
        ran: list[str] = []

        def runner(_stage: sr.Stage) -> int:
            ran.append("ran")
            return 0

        result = sr.run_stage(
            stage("gpu-hungry", needs_gpu=True),
            gpu_wait_seconds=0.0,
            probe=lambda: 9000,
            sleep=lambda _s: None,
            runner=runner,
        )
        assert result.ok is False
        assert result.gpu_wait_timed_out is True
        assert ran == [], "the command must not run when the card never came free"
        assert "never came free" in result.describe()

    def test_a_stage_not_needing_the_gpu_never_waits(self):
        result = sr.run_stage(
            stage("cpu-only"), probe=lambda: 20000, sleep=lambda _s: None, gpu_wait_seconds=0.0
        )
        assert result.ok is True


class TestSequence:
    def test_every_stage_runs_when_all_succeed(self):
        outcome = sr.run_sequence([stage("one"), stage("two"), stage("three")])
        assert outcome.ok is True
        assert [r.name for r in outcome.results] == ["one", "two", "three"]
        assert "SEQUENCE OK" in outcome.describe()

    def test_it_stops_at_the_first_failure_and_marks_the_rest_skipped(self):
        outcome = sr.run_sequence([stage("one"), stage("two", argv=("false",)), stage("three")])
        assert outcome.ok is False
        assert outcome.first_failure is not None
        assert outcome.first_failure.name == "two"
        assert outcome.results[2].skipped is True
        assert outcome.results[2].ok is False

    def test_the_verdict_cannot_claim_completion_after_a_failure(self):
        # The literal bug: "SEQUENCE COMPLETE: screens done" after both screens died.
        outcome = sr.run_sequence([stage("boom", argv=("false",))])
        described = outcome.describe()
        assert "SEQUENCE INCOMPLETE" in described
        assert "stopped at boom" in described
        assert "SEQUENCE OK" not in described

    def test_a_later_stage_does_not_run_after_an_earlier_one_failed(self):
        ran: list[str] = []

        def runner(current: sr.Stage) -> int:
            ran.append(current.name)
            return 1 if current.name == "one" else 0

        outcome = sr.run_sequence([stage("one"), stage("two")], runner=runner)
        assert ran == ["one"]
        assert outcome.ok is False

    def test_an_empty_sequence_is_not_a_success(self):
        # Vacuous truth would let "we ran nothing" read as "everything passed".
        assert sr.run_sequence([]).ok is False


class TestPlanEntryPoint:
    """The runbook documents `stage_runner --plan <module>`, so it must exist and exit honestly.

    Against `games.arm_sequence`, the plan every arm launches through, rather than against a spent
    one-night chain: the module these tests used to load pinned two gitignored corpora by exact
    timestamp, so on a fresh clone the test measured whether one machine still held August's sweep
    results, and on this one `--plan games.night_sequence` would have happily re-run two finished
    plumbing arms on the GPU at a completion budget the rest of `games/` now refuses.
    """

    @pytest.fixture
    def live_plan(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Point `games.arm_sequence` at a sweep-only plan, which needs no corpus and no GPU."""
        monkeypatch.setenv("GAMES_ARM_SEQ_ARM", "hi-lo-group")
        monkeypatch.setenv("GAMES_ARM_SEQ_STAGES", "sweep")
        monkeypatch.setenv("GAMES_ARM_SEQ_SWEEP_DIR", str(tmp_path / "sweep"))
        monkeypatch.setenv("GAMES_ARM_SEQ_OUTPUT_DIR", str(tmp_path / "run"))

    def test_a_real_plan_module_loads(self, live_plan: None):
        del live_plan
        stages = sr.load_plan(LIVE_PLAN_MODULE)
        assert stages
        assert all(isinstance(stage, sr.Stage) for stage in stages)

    def test_every_plan_the_help_advertises_is_still_a_plan(self):
        """A plan named in `--help` is a plan an operator will type at 2am on a rented box.

        The deleted `games.night_sequence` was advertised there while pinning two dated corpora
        nothing regenerates, so the invitation outlived the plan. Importing each advertised module
        and asking for its `stages` is the cheapest check that the advertisement is still true.
        """
        assert sr.EXAMPLE_PLANS
        for module_path in sr.EXAMPLE_PLANS:
            module = importlib.import_module(module_path)
            assert callable(module.stages), module_path
        assert all(name in sr.plan_help() for name in sr.EXAMPLE_PLANS)

    def test_a_module_without_stages_is_refused(self):
        with pytest.raises(ValueError, match="is not a plan"):
            sr.load_plan("games.parsing")

    def test_a_plan_returning_nothing_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(plan, "stages", list)
        with pytest.raises(ValueError, match="nothing to run"):
            sr.load_plan(LIVE_PLAN_MODULE)

    def test_the_exit_code_is_the_contract(self, monkeypatch: pytest.MonkeyPatch):
        # An operator on a rented instance must tell success from failure without reading the log.
        monkeypatch.setattr(plan, "stages", lambda: [stage("ok")])
        assert sr.main(["--plan", LIVE_PLAN_MODULE]) == 0

        monkeypatch.setattr(plan, "stages", lambda: [stage("boom", argv=("false",))])
        assert sr.main(["--plan", LIVE_PLAN_MODULE]) == 1

    def test_a_plan_is_required(self):
        with pytest.raises(SystemExit):
            sr.main([])
