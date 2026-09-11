"""Pin the 9B cloud plan's stage construction, which is all the checking a rented GPU gets.

Offline: stages are built and inspected, never executed. That is the useful thing to test, because
every one of these stages is a command that reserves a GPU costing a few dollars an hour, and the
plan's job is to fail before that happens rather than after.

:class:`TestTheArmIsResumable` is the class with teeth. The arm carries
`--resume-from-checkpoint latest` *and* an explicit `--output-dir`, and those two are only useful
together: the flag needs a stable directory to look in, and `games.train` refuses the combination
without one. If either ever drops out of this argv, a 30-hour interrupted run silently restarts from
step 0 and the operator's recovery procedure ("run it again") quietly becomes "pay twice".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from games.generation import VLLM_COLOCATE_ENV
from games.nine_b_sequence import (
    ALL_STAGES,
    ENV_PREFIX,
    arm_output_dir,
    arm_stage,
    arm_timeout,
    completion_tokens,
    model_tag,
    screen_stage,
    selected_stages,
    stages,
    throughput_stage,
)
from games.plans import COLOCATE_ARM_TIMEOUT, REPO, SCREEN_DIR
from games.tests.conftest import (
    OPTIONAL_KNOB_SETTINGS,
    clear_every_optional_knob,
    clear_plan_independent_environment,
    set_every_optional_knob,
)


@pytest.fixture(autouse=True)
def clean_plan_environment(
    monkeypatch: pytest.MonkeyPatch,
    exported_plan_independent_environment: None,  # exported before this fixture strips it
) -> None:
    """Start every test from plan defaults, so a stray export cannot make one pass."""
    for name in (
        "GAMES_9B_MODEL",
        "GAMES_9B_ARM",
        "GAMES_9B_CORPUS",
        "GAMES_9B_OUTPUT_DIR",
        "GAMES_9B_STAGES",
        "GAMES_9B_MAX_STEPS",
        "GAMES_9B_SAVE_STEPS",
        "GAMES_9B_GROUP",
        "GAMES_9B_PROMPTS_PER_STEP",
        "GAMES_9B_COMPLETION_TOKENS",
        "GAMES_9B_SCREEN_BUDGET",
        "GAMES_9B_ARM_TIMEOUT",
        "GAMES_9B_SCREEN_TIMEOUT",
        "GAMES_9B_THROUGHPUT_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    clear_every_optional_knob(monkeypatch, ENV_PREFIX)
    # Stripped like the plan's own settings, all four of them: the colocate switch refuses any value
    # but "1" at render time, the correction switch decides whether the instrument switches may render
    # at all, and the engine share and the estimator pin land in the argv these tests read. The
    # fixture requested above exports each one first, so a name dropped from this list turns the file
    # red here instead of on the machine of whoever exported it.
    clear_plan_independent_environment(monkeypatch)


def make_corpus(tmp_path: Path) -> Path:
    """A corpus file that exists, which is all the arm stage checks at build time."""
    corpus = tmp_path / "corpus-twin-pd.jsonl"
    corpus.write_text('{"prompt_id": "p0"}\n')
    return corpus


class TestTheArmIsResumable:
    def test_the_argv_carries_both_resume_and_an_explicit_output_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        argv = arm_stage().argv
        assert "--resume-from-checkpoint" in argv
        assert argv[argv.index("--resume-from-checkpoint") + 1] == "latest"
        assert "--output-dir" in argv
        assert argv[argv.index("--output-dir") + 1] == str(arm_output_dir())

    def test_the_output_dir_is_stable_across_builds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Resume looks in this directory, so a timestamp in it would break every restart.

        Pinned to a literal path rather than compared against a second build, because a per-run
        timestamp is identical between two builds microseconds apart: the self-comparison this
        replaces stayed green against a seconds-resolution `strftime` in the directory name, which
        is precisely the regression named above. Written out in full rather than rebuilt from
        `DEFAULT_ARM` and `model_tag()`, since deriving the expected value the same way the
        production code does turns the assertion back into a comparison with itself.
        """
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        assert arm_output_dir() == REPO / "artifacts/games/runs/twin-pd-group-qwen35-9b"

    def test_an_operator_can_override_the_run_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        monkeypatch.setenv("GAMES_9B_OUTPUT_DIR", str(tmp_path / "run"))
        assert arm_output_dir() == tmp_path / "run"
        assert str(tmp_path / "run") in arm_stage().argv


class TestTheArmRefusesToStartWithoutItsCorpus:
    def test_an_unset_corpus_raises_before_any_gpu_is_reserved(self) -> None:
        with pytest.raises(ValueError, match="GAMES_9B_CORPUS is unset"):
            arm_stage()

    def test_a_corpus_that_does_not_exist_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GAMES_9B_CORPUS", "/no/such/corpus.jsonl")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            arm_stage()

    def test_the_whole_plan_fails_at_build_time_not_run_time(self) -> None:
        """`stages()` is called before the runner touches the card, which is the point."""
        with pytest.raises(ValueError, match="GAMES_9B_CORPUS is unset"):
            stages()


class TestStageSelection:
    def test_the_default_is_every_stage_in_plan_order(self) -> None:
        assert selected_stages() == ALL_STAGES

    def test_a_subset_keeps_plan_order_not_the_order_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Screen before throughput before arm, whatever order the operator typed them in."""
        monkeypatch.setenv("GAMES_9B_STAGES", "arm,screen")
        assert selected_stages() == ("screen", "arm")

    def test_whitespace_is_tolerated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GAMES_9B_STAGES", " screen , throughput ")
        assert selected_stages() == ("screen", "throughput")

    def test_an_unknown_stage_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GAMES_9B_STAGES", "screen,evaluate")
        with pytest.raises(ValueError, match="unknown stage"):
            selected_stages()

    def test_selecting_nothing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GAMES_9B_STAGES", " , ")
        with pytest.raises(ValueError, match="selected nothing"):
            selected_stages()

    def test_a_screen_only_plan_needs_no_corpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The screen runs before a corpus exists, so requiring one would invert the order."""
        monkeypatch.setenv("GAMES_9B_STAGES", "screen,throughput")
        built = stages()
        assert len(built) == 2
        assert all(stage.needs_gpu for stage in built)


class TestEveryStageIsBoundedAndVerified:
    @pytest.mark.parametrize("stage_name", ["screen", "throughput"])
    def test_each_stage_has_a_wall_clock_cap(self, stage_name: str) -> None:
        """A wedged run on a rented card bills until someone notices; timeout is the backstop."""
        builder = {"screen": screen_stage, "throughput": throughput_stage}[stage_name]
        assert builder().argv[0] == "timeout"

    def test_the_arm_has_one_too(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        assert arm_stage().argv[0] == "timeout"

    @pytest.mark.parametrize("stage_name", ["screen", "throughput"])
    def test_each_stage_promises_an_artifact(self, stage_name: str) -> None:
        """stage_runner treats a stage as a liar until its output exists, so it must name one."""
        builder = {"screen": screen_stage, "throughput": throughput_stage}[stage_name]
        assert builder().artifacts

    def test_the_arm_promises_its_train_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        assert arm_stage().artifacts == (arm_output_dir() / "train_summary.json",)

    def test_no_stage_uses_the_local_resource_limiter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Deliberate: the limiter protects a shared box and needs a systemd user session."""
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        for stage in (screen_stage(), throughput_stage(), arm_stage()):
            assert not any("resource-limits" in arg for arg in stage.argv), stage.name


class TestTheArmCapIsTheSharedOneAndAnOperatorCanRaiseIt:
    """This plan runs the LARGEST model and used to carry its own literal cap, with no override.

    The argument every plan file makes: a cap under the healthy run time is strictly worse than
    no cap, since it kills a working arm when its artifacts are most expensive to recreate. There
    is no measured 9B step time yet, which is exactly why the cap is the shared one rather than a
    9B literal.
    """

    def test_the_cap_comes_from_the_shared_constant(self) -> None:
        assert arm_timeout() == COLOCATE_ARM_TIMEOUT

    def test_an_operator_can_raise_it_from_the_shell(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        monkeypatch.setenv("GAMES_9B_ARM_TIMEOUT", "72h")
        assert arm_stage().argv[:2] == ("timeout", "72h")

    def test_the_engine_knobs_reach_the_printed_command(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A command that does not say what the engine holds cannot be checked before the meter."""
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        argv = arm_stage().argv
        assert "--vllm-gpu-memory-utilization" in argv
        assert "--vllm-colocate" not in argv

    def test_a_stage_build_refuses_on_a_box_exporting_the_slow_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        monkeypatch.setenv(VLLM_COLOCATE_ENV, "0")
        with pytest.raises(RuntimeError, match="2026-08-26"):
            arm_stage()


class TestScreenArtifactsLandWhereEverythingElseLooksForThem:
    """The plan wrote `artifacts/games/screens`; the writer and every screen on disk say screen."""

    def test_the_screen_artifact_lands_in_the_shared_screen_directory(self) -> None:
        (artifact,) = screen_stage().artifacts
        assert artifact.parent == SCREEN_DIR

    def test_the_promised_artifact_is_the_path_the_command_writes(self) -> None:
        stage = screen_stage()
        argv = stage.argv
        assert stage.artifacts == (Path(argv[argv.index("--json-out") + 1]),)


class TestTheDefaultsMatchTheProjectRules:
    def test_the_completion_budget_is_the_measured_one_not_a_convenient_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Anything read as science has to let the chain of thought finish."""
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        argv = arm_stage().argv
        assert argv[argv.index("--max-completion-tokens") + 1] == completion_tokens()
        # The floor is now derived per model from that model's own termination screen, so this
        # asserts the 9B's measured budget rather than a literal that could drift away from it.
        assert int(completion_tokens()) >= 32768

    def test_thinking_is_left_on(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The decision-theory reasoning these arms measure lives in the chain of thought."""
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        assert "--no-thinking" not in arm_stage().argv

    def test_the_default_model_is_the_nine_b(self) -> None:
        assert model_tag() == "qwen35-9b"

    def test_the_model_is_overridable_for_a_dry_run_on_something_small(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GAMES_9B_MODEL", "Qwen/Qwen3.5-2B")
        assert model_tag() == "qwen35-2b"
        assert "Qwen/Qwen3.5-2B" in screen_stage().argv


class TestTheOptionalTrainingKnobsReachThisPlanToo:
    """The five knobs come through `plans.training_shape`, which this plan already calls.

    Worth asserting here rather than trusting the shared helper: this is the plan the 9B arms launch
    from, so a knob that reached the other two and not this one would be discovered on the box.
    """

    def test_no_knob_is_rendered_when_the_environment_sets_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        argv = arm_stage().argv
        for _suffix, flag, _value in OPTIONAL_KNOB_SETTINGS:
            assert flag not in argv, flag

    def test_every_knob_reaches_the_command_with_the_value_that_was_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        set_every_optional_knob(monkeypatch, ENV_PREFIX)
        argv = arm_stage().argv
        for _suffix, flag, value in OPTIONAL_KNOB_SETTINGS:
            assert argv[argv.index(flag) + 1] == value, flag

    def test_a_misspelled_schedule_is_refused_before_the_gpu_is_reserved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("GAMES_9B_CORPUS", str(make_corpus(tmp_path)))
        monkeypatch.setenv(f"{ENV_PREFIX}LR_SCHEDULER", "constant_with_warumup")
        with pytest.raises(ValueError, match=f"{ENV_PREFIX}LR_SCHEDULER"):
            arm_stage()
