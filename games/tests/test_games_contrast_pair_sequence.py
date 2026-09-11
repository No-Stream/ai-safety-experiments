"""Tests for the twin-PD contrast-pair plan.

The corpus resolver carries most of the weight here. It exists because the alternative -- globbing
for the newest `corpus-*.jsonl` -- once picked a 0-byte file and trained an arm on nothing while
reporting success, so every way it can be wrong is asserted rather than assumed.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from games import contrast_pair_sequence as plan
from games import nine_b_sequence
from games.generation import VLLM_COLOCATE_ENV
from games.plans import (
    COLOCATE_ARM_TIMEOUT,
    ESTIMATOR_PIN_ENV,
    FLAGSHIP_ESTIMATOR_PIN,
    SWEEP_TIMEOUT_BY_BACKEND,
)
from games.s3_sync import DEFAULT_SYNC_COMMAND, build_sync_command
from games.tests import conftest as games_conftest
from games.tests.conftest import (
    OPTIONAL_KNOB_SETTINGS,
    clear_every_optional_knob,
    clear_plan_independent_environment,
    offered_flags,
    set_every_optional_knob,
    used_flags,
)
from games.train import VLLM_IS_CORRECTION_ENV

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# What a launch gate greps a printed plan for, spaced as `plans.describe_stages` joins an argv.
PINNED_ESTIMATOR_STRINGS = (
    "--loss-type dapo",
    "--scale-rewards batch",
    "--acknowledge-liger-estimator-mismatch",
    "--micro-batch-size 1",
)


@pytest.fixture(autouse=True)
def clean_generation_environment(
    monkeypatch: pytest.MonkeyPatch,
    exported_plan_independent_environment: None,  # exported before this fixture strips it
) -> None:
    """Strip the plan-independent variables, so an operator's shell cannot change what is measured.

    All four are spelled plan-independently on purpose -- three describe the box and the fourth
    describes which already-trained arms a run is comparable to, none of them one plan's business --
    so unlike `GAMES_PAIR_*` they are not covered by any per-test cleanup. The old backend switch is
    stripped like the knobs: any value but "1" is refused at render time now, and a stray "0" in the
    operator's shell would fail every stage build in this file. The fixture requested above exports
    all four first, so this file measures what it claims to even when a shell carries them.
    """
    clear_plan_independent_environment(monkeypatch)
    for env_prefix in (plan.ENV_PREFIX, nine_b_sequence.ENV_PREFIX):
        clear_every_optional_knob(monkeypatch, env_prefix)


@pytest.fixture
def sweep_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the plan at an empty sweep directory it owns."""
    directory = tmp_path / "pair-sweep"
    directory.mkdir()
    monkeypatch.setenv("GAMES_PAIR_SWEEP_DIR", str(directory))
    monkeypatch.delenv("GAMES_PAIR_CORPUS", raising=False)
    return directory


@pytest.fixture
def corpus(sweep_dir: Path) -> Path:
    """Write one plausible swept corpus, the case every arm stage expects."""
    path = sweep_dir / "corpus-twin-pd-20260817T030000Z.jsonl"
    path.write_text('{"prompt": "x"}\n', encoding="utf-8")
    return path


@pytest.fixture
def second_invocation(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The state the documented second invocation runs in: corpus swept, regrade in the plan.

    The self-graded arm reads a file the regrade writes, so building it needs either that file on
    disk or the regrade stage selected -- otherwise the guard that stops a doomed arm before the GPU
    wait fires, which is what these tests would otherwise be measuring.
    """
    monkeypatch.setenv("GAMES_PAIR_STAGES", "regrade,arm-group,arm-self")
    return corpus


class TestResolveCorpus:
    """The resolver must refuse to guess, and must never return an empty corpus."""

    def test_returns_the_single_non_empty_corpus(self, sweep_dir: Path):
        written = sweep_dir / "corpus-twin-pd-Qwen35-2B-20260817T120000Z.jsonl"
        written.write_text('{"prompt": "x"}\n', encoding="utf-8")
        assert plan.resolve_corpus() == written

    def test_raises_when_the_sweep_never_ran(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_PAIR_SWEEP_DIR", str(tmp_path / "absent"))
        with pytest.raises(FileNotFoundError, match="never ran"):
            plan.resolve_corpus()

    def test_raises_when_the_directory_holds_no_corpus(self, sweep_dir: Path):
        (sweep_dir / "selection-twin-pd.json").write_text("{}", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="no corpus-"):
            plan.resolve_corpus()

    def test_raises_on_an_empty_corpus_rather_than_training_on_nothing(self, sweep_dir: Path):
        """The sabotage case: a 0-byte corpus is a plausible match and a silent null result."""
        (sweep_dir / "corpus-twin-pd-20260817T000000Z.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match=r"every corpus .* is empty"):
            plan.resolve_corpus()

    def test_raises_when_two_corpora_could_be_meant(self, sweep_dir: Path):
        for stamp in ("20260817T010000Z", "20260817T020000Z"):
            (sweep_dir / f"corpus-twin-pd-{stamp}.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty corpora"):
            plan.resolve_corpus()

    def test_ignores_an_empty_sibling_when_one_real_corpus_exists(self, sweep_dir: Path):
        (sweep_dir / "corpus-twin-pd-20260817T010000Z.jsonl").write_text("", encoding="utf-8")
        real = sweep_dir / "corpus-twin-pd-20260817T020000Z.jsonl"
        real.write_text("{}\n", encoding="utf-8")
        assert plan.resolve_corpus() == real

    def test_an_explicit_corpus_overrides_resolution(
        self, sweep_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del sweep_dir
        chosen = tmp_path / "chosen.jsonl"
        monkeypatch.setenv("GAMES_PAIR_CORPUS", str(chosen))
        assert plan.group_corpus() == chosen


class TestSelectedStages:
    """Stage selection is how the sweep runs alone, so its refusals matter."""

    def test_defaults_to_the_sweep_alone(self, monkeypatch: pytest.MonkeyPatch):
        """The corpus every later stage reads carries a timestamp the sweep picks at run time."""
        monkeypatch.delenv("GAMES_PAIR_STAGES", raising=False)
        assert plan.selected_stages() == ("sweep",)

    def test_reorders_a_subset_into_plan_order(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_PAIR_STAGES", "arm-self,sweep")
        assert plan.selected_stages() == ("sweep", "arm-self")

    def test_rejects_an_unknown_stage_name(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_PAIR_STAGES", "sweep,arm-both")
        with pytest.raises(ValueError, match="unknown stage"):
            plan.selected_stages()

    def test_rejects_selecting_nothing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_PAIR_STAGES", " , ")
        with pytest.raises(ValueError, match="selected nothing"):
            plan.selected_stages()


class TestCompletionBudget:
    """The budget must follow the model, since that is what a copied literal got wrong before."""

    def test_two_b_gets_its_measured_floor(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_PAIR_MODEL", "Qwen/Qwen3.5-2B")
        monkeypatch.delenv("GAMES_PAIR_COMPLETION_TOKENS", raising=False)
        assert plan.completion_tokens() == "24576"

    def test_nine_b_gets_a_larger_floor_than_two_b(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("GAMES_PAIR_COMPLETION_TOKENS", raising=False)
        monkeypatch.setenv("GAMES_PAIR_MODEL", "Qwen/Qwen3.5-9B")
        nine_b = int(plan.completion_tokens())
        monkeypatch.setenv("GAMES_PAIR_MODEL", "Qwen/Qwen3.5-2B")
        assert nine_b > int(plan.completion_tokens())


class TestStageCommands:
    """What the stages actually run, since these argv lists are the experiment."""

    def test_sweep_runs_thinking_on_at_the_measured_budget(
        self, sweep_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del sweep_dir
        monkeypatch.setenv("GAMES_PAIR_MODEL", "Qwen/Qwen3.5-2B")
        monkeypatch.delenv("GAMES_PAIR_COMPLETION_TOKENS", raising=False)
        argv = plan.sweep_stage().argv
        assert "--thinking" in argv
        assert "--no-thinking" not in argv
        assert argv[argv.index("--max-new-tokens") + 1] == "24576"
        assert argv[argv.index("--grading") + 1] == "group-mix"

    def test_both_arms_take_the_same_engine_knobs_and_estimator(
        self, second_invocation: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The contrast survives an estimator caveat only if both arms carry the identical one."""
        del second_invocation
        monkeypatch.setenv(VLLM_IS_CORRECTION_ENV, "0")
        group, contrast = plan.group_arm_stage().argv, plan.self_arm_stage().argv

        def vllm_flags(argv: Sequence[str]) -> list[str]:
            """The vLLM flags of an argv, and nothing else.

            Flags only: a bare substring match also caught the two arms' output paths whenever the
            checkout sat under a directory with "vllm" in its name (a gate worktree named after the
            vllm-streaming group did exactly that), and the two arms' paths differ by design.
            """
            return [flag for flag in argv if flag.startswith("--") and "vllm" in flag]

        colocate_flags = vllm_flags(group)
        assert "--vllm-gpu-memory-utilization" in colocate_flags
        assert "--no-vllm-importance-sampling-correction" in colocate_flags
        assert colocate_flags == vllm_flags(contrast)

    def test_no_init_adapter_renders_no_flag(self, second_invocation: Path):
        """Every pre-transfer launch's argv must be byte-stable under the new knob's default."""
        del second_invocation
        for argv in (plan.group_arm_stage().argv, plan.self_arm_stage().argv):
            assert "--init-adapter" not in argv

    def test_the_init_adapter_is_per_arm_not_per_plan(
        self, second_invocation: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The transfer shape seeds each grading from its own checkpoint; the control seeds none.

        One shared variable would seed both arms from one checkpoint, which is a different
        experiment, so the knob resolves per arm exactly as the output-dir override does.
        """
        del second_invocation
        monkeypatch.setenv(
            "GAMES_PAIR_INIT_ADAPTER_TWIN_PD_SELF", "/adapters/twin-pd-self/checkpoint-70"
        )
        contrast = plan.self_arm_stage().argv
        assert (
            contrast[contrast.index("--init-adapter") + 1] == "/adapters/twin-pd-self/checkpoint-70"
        )
        assert "--init-adapter" not in plan.group_arm_stage().argv

    def test_the_sweep_names_the_backend_its_cap_is_keyed_on(self, sweep_dir: Path):
        """The cap is looked up by backend, so the sweep must not inherit a different default."""
        del sweep_dir
        argv = plan.sweep_stage().argv
        assert argv[argv.index("--backend") + 1] == plan.SWEEP_BACKEND
        assert plan.sweep_timeout() == SWEEP_TIMEOUT_BY_BACKEND[plan.SWEEP_BACKEND]

    def test_an_operator_can_raise_the_sweep_cap(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_PAIR_SWEEP_TIMEOUT", "18h")
        assert plan.sweep_timeout() == "18h"

    def test_the_game_override_reaches_both_the_command_and_the_heading(
        self, sweep_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """These two disagreed once: a run swept one game under a heading naming another."""
        del sweep_dir
        monkeypatch.setenv("GAMES_PAIR_GAME", "stag-hunt")
        stage = plan.sweep_stage()
        assert stage.argv[stage.argv.index("--game") + 1] == "stag-hunt"
        assert "stag-hunt" in stage.name

    def test_the_arm_cap_is_the_colocate_cap_and_the_slow_export_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        assert plan.arm_timeout() == COLOCATE_ARM_TIMEOUT
        monkeypatch.setenv(VLLM_COLOCATE_ENV, "0")
        with pytest.raises(RuntimeError, match="2026-08-26"):
            plan.arm_timeout()

    def test_an_operator_still_overrides_the_arm_cap(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GAMES_PAIR_ARM_TIMEOUT", "90m")
        assert plan.arm_timeout() == "90m"

    def test_no_backend_flag_is_rendered_because_there_is_no_backend_choice(self, corpus: Path):
        del corpus
        argv = plan.group_arm_stage().argv
        assert "--vllm-colocate" not in argv
        assert "--no-vllm-colocate" not in argv
        assert "--vllm-gpu-memory-utilization" in argv

    def test_both_arms_train_on_the_same_prompts_under_different_gradings(
        self, second_invocation: Path
    ):
        group, contrast = plan.group_arm_stage().argv, plan.self_arm_stage().argv
        assert group[group.index("--corpus") + 1] == str(second_invocation)
        assert contrast[contrast.index("--corpus") + 1] == str(plan.self_corpus())
        assert group[group.index("--arm") + 1] == "twin-pd-group"
        assert contrast[contrast.index("--arm") + 1] == "twin-pd-self"

    def test_regrade_writes_exactly_what_the_self_arm_reads(self, second_invocation: Path):
        del second_invocation
        regrade = plan.regrade_stage().argv
        read = plan.self_arm_stage().argv
        assert regrade[regrade.index("--out") + 1] == read[read.index("--corpus") + 1]
        assert regrade[regrade.index("--grading") + 1] == "self"

    def test_regrade_promises_its_output_as_an_artifact(self, corpus: Path):
        del corpus
        assert plan.regrade_stage().artifacts == (plan.self_corpus(),)

    def test_arms_resume_with_an_explicit_output_dir(self, second_invocation: Path):
        """`latest` without an explicit --output-dir silently restarts at step 0.

        The directory is compared against the plan's own `arm_output_dir`, not merely checked for
        being non-empty: resume reads that exact path, so a wrong-but-truthy one restarts at step 0
        just as silently as an absent one.
        """
        del second_invocation
        for arm, stage in (
            (plan.GROUP_ARM, plan.group_arm_stage()),
            (plan.SELF_ARM, plan.self_arm_stage()),
        ):
            argv = stage.argv
            assert argv[argv.index("--resume-from-checkpoint") + 1] == "latest"
            assert argv[argv.index("--output-dir") + 1] == str(plan.arm_output_dir(arm))

    def test_every_gpu_stage_carries_a_wall_clock_cap(self, second_invocation: Path):
        """A wedged stage bills a rented instance until someone notices; `timeout` is the backstop.

        The prefix, not just the duration. `TestCompletionBudget` above already pins what
        `sweep_timeout()` and `arm_timeout()` return, and both `timeout` prefixes could be deleted
        from the argv entirely with every other test in this file green.
        """
        del second_invocation
        assert plan.sweep_stage().argv[:2] == ("timeout", plan.sweep_timeout())
        for stage in (plan.group_arm_stage(), plan.self_arm_stage()):
            assert stage.argv[:2] == ("timeout", plan.arm_timeout()), stage.name

    def test_arms_promise_a_train_summary(self, corpus: Path):
        del corpus
        expected = plan.arm_output_dir("twin-pd-group") / "train_summary.json"
        assert plan.group_arm_stage().artifacts == (expected,)

    def test_arms_never_pass_no_thinking(self, second_invocation: Path):
        del second_invocation
        for stage in (plan.group_arm_stage(), plan.self_arm_stage()):
            assert "--no-thinking" not in stage.argv
            assert "--allow-short-completions" not in stage.argv

    def test_only_the_gpu_stages_are_marked_as_needing_one(self, corpus: Path):
        del corpus
        assert plan.sweep_stage().needs_gpu
        assert plan.group_arm_stage().needs_gpu
        assert not plan.regrade_stage().needs_gpu

    def test_each_arm_ships_to_its_own_s3_prefix(
        self, second_invocation: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del second_invocation
        monkeypatch.setenv("GAMES_PAIR_S3_DEST", "s3://bucket/games_rl/pair/")
        group, contrast = plan.group_arm_stage().env, plan.self_arm_stage().env
        assert group is not None
        assert contrast is not None
        assert group["GAMES_S3_DEST"] != contrast["GAMES_S3_DEST"]
        assert group["GAMES_S3_DEST"].startswith("s3://bucket/games_rl/pair/")

    def test_the_sync_stays_off_when_no_destination_is_set(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del corpus
        monkeypatch.delenv("GAMES_PAIR_S3_DEST", raising=False)
        assert plan.group_arm_stage().env is None


class TestTheArmPairIsParameterized:
    """The pd-unstated pair trains through this plan too: arm names from env, game derived.

    The plan's whole value is the identity it holds -- one sweep, one regrade, two arms on the same
    prompts -- and the pd-unstated pair needs exactly that identity on a different game id. So the
    two arm names come from `GAMES_PAIR_GROUP_ARM` / `GAMES_PAIR_SELF_ARM` (defaulting to the
    twin-pd originals), the game defaults to what the registry says the selected arms train, and a
    selection the registry contradicts is refused before a stage is built: a sweep of one game
    feeding arms of another is the plan-and-registry drift `games.arm_sequence`'s docstring warns
    about, surfacing as a corpus rejection an hour into a rented instance.
    """

    def _select_unstated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GAMES_PAIR_GROUP_ARM", "pd-unstated-group")
        monkeypatch.setenv("GAMES_PAIR_SELF_ARM", "pd-unstated-self")

    def test_the_default_pair_is_the_twin_pd_original(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("GAMES_PAIR_GROUP_ARM", "GAMES_PAIR_SELF_ARM", "GAMES_PAIR_GAME"):
            monkeypatch.delenv(name, raising=False)
        assert plan.group_arm() == "twin-pd-group"
        assert plan.self_arm() == "twin-pd-self"
        assert plan.game_id() == "twin-pd"

    def test_the_game_is_derived_from_the_selected_arms(
        self, sweep_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No separate GAMES_PAIR_GAME needed: the registry already says what the arms train."""
        del sweep_dir
        self._select_unstated(monkeypatch)
        monkeypatch.delenv("GAMES_PAIR_GAME", raising=False)
        stage = plan.sweep_stage()
        assert stage.argv[stage.argv.index("--game") + 1] == "pd-unstated"
        assert "pd-unstated" in stage.name

    def test_the_arm_stages_train_the_selected_arms(
        self, second_invocation: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del second_invocation
        self._select_unstated(monkeypatch)
        group, contrast = plan.group_arm_stage().argv, plan.self_arm_stage().argv
        assert group[group.index("--arm") + 1] == "pd-unstated-group"
        assert contrast[contrast.index("--arm") + 1] == "pd-unstated-self"
        assert group[group.index("--corpus") + 1] != contrast[contrast.index("--corpus") + 1]

    def test_the_selected_arms_get_their_own_run_directories(
        self, second_invocation: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Output dirs key on the arm name, so the pd-unstated pair cannot resume a twin-pd run."""
        del second_invocation
        self._select_unstated(monkeypatch)
        group = plan.group_arm_stage().argv
        assert "pd-unstated-group" in group[group.index("--output-dir") + 1]

    def test_a_pair_spanning_two_games_is_refused(
        self, second_invocation: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del second_invocation
        monkeypatch.setenv("GAMES_PAIR_GROUP_ARM", "twin-pd-group")
        monkeypatch.setenv("GAMES_PAIR_SELF_ARM", "pd-unstated-self")
        with pytest.raises(ValueError, match="game"):
            plan.stages()

    def test_an_arm_under_the_wrong_grading_is_refused(
        self, second_invocation: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A self-graded arm in the group slot would sweep under a grading its arm never trains."""
        del second_invocation
        monkeypatch.setenv("GAMES_PAIR_GROUP_ARM", "twin-pd-self")
        monkeypatch.setenv("GAMES_PAIR_SELF_ARM", "twin-pd-self")
        with pytest.raises(ValueError, match="grad"):
            plan.stages()

    def test_a_game_override_that_disagrees_with_the_arms_is_refused(
        self, second_invocation: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit GAMES_PAIR_GAME that contradicts the registry would sweep one game and
        train another -- the exact drift the derived default exists to close."""
        del second_invocation
        self._select_unstated(monkeypatch)
        monkeypatch.setenv("GAMES_PAIR_GAME", "twin-pd")
        with pytest.raises(ValueError, match="game"):
            plan.stages()

    def test_an_unknown_arm_name_is_refused(
        self, second_invocation: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del second_invocation
        monkeypatch.setenv("GAMES_PAIR_GROUP_ARM", "hopscotch-group")
        with pytest.raises(ValueError, match="unknown arm"):
            plan.stages()


class TestEveryStageLogShipsWithTheRun:
    """The pair's logs must land inside the run directories the S3 sync already ships wholesale.

    Nothing in the repo syncs `artifacts/games/logs/`, so a log written only there survives exactly
    as long as the instance does -- and a hand-rolled log-only sync was the whole of one run's
    surviving record on 2026-08-19. One sweep serves both arms here, so its log is provenance for
    both run directories and belongs in both.
    """

    def test_each_arm_log_ships_with_that_arm_and_not_the_other(self, second_invocation: Path):
        del second_invocation
        group_dir = plan.arm_output_dir(plan.GROUP_ARM)
        self_dir = plan.arm_output_dir(plan.SELF_ARM)
        group_logs = plan.group_arm_stage().log_destinations()
        assert any(path.is_relative_to(group_dir) for path in group_logs)
        assert not any(path.is_relative_to(self_dir) for path in group_logs)
        assert any(
            path.is_relative_to(self_dir) for path in plan.self_arm_stage().log_destinations()
        )

    def test_the_shared_sweep_and_regrade_logs_ship_with_both_arms(self, corpus: Path):
        del corpus
        run_dirs = (plan.arm_output_dir(plan.GROUP_ARM), plan.arm_output_dir(plan.SELF_ARM))
        for stage in (plan.sweep_stage(), plan.regrade_stage()):
            for run_dir in run_dirs:
                assert any(path.is_relative_to(run_dir) for path in stage.log_destinations()), (
                    f"{stage.name} leaves no log in {run_dir}"
                )

    def test_the_shipped_copy_is_inside_what_the_arm_sync_covers(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del corpus
        monkeypatch.setenv("GAMES_PAIR_S3_DEST", "s3://bucket/games_rl/pair")
        stage = plan.group_arm_stage()
        assert stage.env is not None
        run_dir = plan.arm_output_dir(plan.GROUP_ARM)
        argv = build_sync_command(run_dir, stage.env["GAMES_S3_DEST"])
        assert argv[len(DEFAULT_SYNC_COMMAND)] == str(run_dir)
        assert not [flag for flag in argv if flag.startswith(("--exclude", "--include"))]
        assert any(path.is_relative_to(run_dir) for path in stage.log_destinations())

    def test_the_shared_log_directory_still_gets_its_copy(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del corpus
        monkeypatch.setenv("GAMES_PAIR_STAGES", "regrade,arm-group,arm-self")
        # An operator tails these, and `resolve_corpus` names the sweep's in its failure message.
        for stage in [plan.sweep_stage(), *plan.stages()]:
            assert stage.log_path is not None
            assert stage.log_path.parent == plan.LOG_DIR
            assert stage.log_path in stage.log_destinations()


class TestPlanEntryPoint:
    """`stages()` is the contract `stage_runner --plan` depends on."""

    def test_the_second_invocation_builds_the_regrade_and_both_arms(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("GAMES_PAIR_STAGES", "regrade,arm-group,arm-self")
        plan.self_corpus().write_text('{"prompt": "x"}\n', encoding="utf-8")
        built = plan.stages()
        assert [stage.name.split()[0] for stage in built] == ["regrade", "train", "train"]
        assert corpus.is_file()

    def test_sweeping_beside_a_stage_that_reads_the_corpus_is_refused(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Even with a corpus already present, which is the expensive branch.

        The arms would train on the corpus on disk while the fresh sweep beside them wrote a second
        one that nothing reads: hours of GPU, no error, and a sweep directory left ambiguous for
        every later resolve.
        """
        del corpus
        monkeypatch.setenv("GAMES_PAIR_STAGES", "sweep,arm-group")
        with pytest.raises(ValueError, match="two invocations"):
            plan.stages()

    def test_the_headline_launch_is_two_invocations_that_cover_every_stage(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("GAMES_PAIR_STAGES", "sweep")
        first = [stage.name.split()[0] for stage in plan.stages()]
        plan.self_corpus().write_text('{"prompt": "x"}\n', encoding="utf-8")
        monkeypatch.setenv("GAMES_PAIR_STAGES", "regrade,arm-group,arm-self")
        second = [stage.name.split()[0] for stage in plan.stages()]
        assert first + second == ["baseline", "regrade", "train", "train"]
        assert corpus.is_file()

    def test_the_sweep_alone_needs_no_corpus(
        self, sweep_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del sweep_dir
        monkeypatch.setenv("GAMES_PAIR_STAGES", "sweep")
        assert len(plan.stages()) == 1

    def test_building_an_arm_without_a_corpus_fails_before_any_gpu_is_reserved(
        self, sweep_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del sweep_dir
        monkeypatch.setenv("GAMES_PAIR_STAGES", "arm-group")
        with pytest.raises(FileNotFoundError, match="no corpus-"):
            plan.stages()

    def test_the_self_arm_refuses_a_regraded_corpus_no_stage_will_write(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The documented workflow: `GAMES_PAIR_CORPUS` exported, then `arm-self` run alone.

        `self_corpus()` derives its path from the group corpus and nothing verified it existed, so
        the stage built happily, stage_runner waited for the card, `games.train` loaded a tokenizer,
        and only then did `load_corpus` raise.
        """
        monkeypatch.setenv("GAMES_PAIR_CORPUS", str(corpus))
        monkeypatch.setenv("GAMES_PAIR_STAGES", "arm-self")
        with pytest.raises(FileNotFoundError, match="no stage in this plan will write it"):
            plan.stages()

    def test_the_self_arm_is_content_when_the_regrade_will_write_its_corpus(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del corpus
        monkeypatch.setenv("GAMES_PAIR_STAGES", "regrade,arm-self")
        assert [stage.name.split()[0] for stage in plan.stages()] == ["regrade", "train"]

    def test_run_mode_maps_a_verified_sequence_to_exit_zero(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """`--plan` is the documented launch path, but this branch is an operator's fallback."""
        del corpus
        monkeypatch.setattr(plan, "stages", list)
        monkeypatch.setattr(plan, "run_sequence", lambda _stages: SimpleNamespace(ok=True))
        assert plan.main([]) == 0
        monkeypatch.setattr(plan, "run_sequence", lambda _stages: SimpleNamespace(ok=False))
        assert plan.main([]) == 1

    def test_print_corpus_resolves_without_running_anything(
        self, corpus: Path, capsys: pytest.CaptureFixture[str]
    ):
        assert plan.main(["--print-corpus"]) == 0
        assert capsys.readouterr().out.strip() == str(corpus)

    def test_print_plan_carries_the_estimator_pin_on_both_arms(
        self,
        second_invocation: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """This is the pair the pin exists for, and both its arms have to carry it or it is not one.

        The printout is where a launch script checks it got the estimator it asked for; before this
        the four flags were spliced into `plans.build_arm_stage` on the rented box by string-matching
        its body, which had to be re-applied on every reclaim recovery.
        """
        del second_invocation
        plan.self_corpus().write_text('{"prompt": "x"}\n', encoding="utf-8")
        monkeypatch.setenv(ESTIMATOR_PIN_ENV, FLAGSHIP_ESTIMATOR_PIN)
        assert plan.main(["--print-plan"]) == 0
        printed = capsys.readouterr().out
        for pinned in PINNED_ESTIMATOR_STRINGS:
            assert printed.count(pinned) == 2, f"{pinned} on {printed.count(pinned)} of 2 arms"

    def test_print_plan_names_no_estimator_at_all_when_no_pin_is_asked_for(
        self,
        second_invocation: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        """An unpinned launch trains the repo defaults, which the printout must not contradict."""
        del second_invocation
        plan.self_corpus().write_text('{"prompt": "x"}\n', encoding="utf-8")
        assert plan.main(["--print-plan"]) == 0
        printed = capsys.readouterr().out
        for flag in ("--loss-type", "--scale-rewards", "--micro-batch-size"):
            assert flag not in printed, flag
        assert "--acknowledge-liger-estimator-mismatch" not in printed

    def test_print_plan_refuses_a_misspelled_pin_rather_than_printing_a_plan(
        self, second_invocation: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A silent typo would print a green plan and train the defaults under a pinned name."""
        del second_invocation
        plan.self_corpus().write_text('{"prompt": "x"}\n', encoding="utf-8")
        monkeypatch.setenv(ESTIMATOR_PIN_ENV, "flagship-dapo-batch")
        with pytest.raises(ValueError, match=ESTIMATOR_PIN_ENV):
            plan.main(["--print-plan"])


class TestEveryStageFlagIsAcceptedByItsCli:
    """Check each stage's flags against the target CLI's own --help.

    This is the seam that costs the most to discover late. A plan builds argv as data, so a flag
    that was renamed or removed upstream is invisible until the stage runs -- and on a rented
    instance it runs after the meter has started, an hour into a bootstrap. Asking each CLI what it
    accepts is ten seconds per module here, paid once per session through the cache in
    `games.tests.conftest`, and a stage failure avoided there. It is also how this suite caught that
    `--max-new-tokens` (the sweep) and `--max-completion-tokens` (training) are different flags on
    different modules.
    """

    def _assert_flags_accepted(self, stage_argv: tuple[str, ...]) -> None:
        module, used = used_flags(stage_argv)
        unknown = used - offered_flags(module)
        assert not unknown, f"{module} does not accept {sorted(unknown)}"

    def test_contrast_pair_stages(self, corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        del corpus
        monkeypatch.setenv("GAMES_PAIR_STAGES", "regrade,arm-group,arm-self")
        for stage in [plan.sweep_stage(), *plan.stages()]:
            self._assert_flags_accepted(stage.argv)

    def test_nine_b_stages(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        nine_b_corpus = tmp_path / "corpus-9b.jsonl"
        nine_b_corpus.write_text('{"prompt": "x"}\n', encoding="utf-8")
        monkeypatch.setenv("GAMES_9B_CORPUS", str(nine_b_corpus))
        for stage in nine_b_sequence.stages():
            self._assert_flags_accepted(stage.argv)

    def test_pinned_stages_on_both_plans_that_share_the_arm_builder(
        self, second_invocation: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pinned four are off the default path, so `games.train --help` gets asked about them.

        Both plans in this file at once, since neither adds anything to `build_arm_stage`'s argv and
        the point is that neither had to.
        """
        del second_invocation
        plan.self_corpus().write_text('{"prompt": "x"}\n', encoding="utf-8")
        nine_b_corpus = tmp_path / "corpus-9b.jsonl"
        nine_b_corpus.write_text('{"prompt": "x"}\n', encoding="utf-8")
        monkeypatch.setenv("GAMES_9B_CORPUS", str(nine_b_corpus))
        monkeypatch.setenv(ESTIMATOR_PIN_ENV, FLAGSHIP_ESTIMATOR_PIN)
        for stage in [*plan.stages(), *nine_b_sequence.stages()]:
            self._assert_flags_accepted(stage.argv)

    def test_the_optional_knobs_on_both_plans_that_share_the_arm_builder(
        self, second_invocation: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The five optional training knobs are off the default path too, on both plans at once."""
        del second_invocation
        plan.self_corpus().write_text('{"prompt": "x"}\n', encoding="utf-8")
        nine_b_corpus = tmp_path / "corpus-9b.jsonl"
        nine_b_corpus.write_text('{"prompt": "x"}\n', encoding="utf-8")
        monkeypatch.setenv("GAMES_9B_CORPUS", str(nine_b_corpus))
        set_every_optional_knob(monkeypatch, plan.ENV_PREFIX)
        set_every_optional_knob(monkeypatch, nine_b_sequence.ENV_PREFIX)
        for stage in [*plan.stages(), *nine_b_sequence.stages()]:
            self._assert_flags_accepted(stage.argv)


class TestTheOptionalTrainingKnobsReachThisPlanToo:
    """The five knobs live in `plans.training_shape`, so all three plans read them for free.

    Free is the claim under test: this plan adds nothing to `build_arm_stage`'s argv, so a knob
    wired for the one-arm plan has to arrive here without a per-plan change, and it has to arrive on
    BOTH arms of the pair or the contrast stops being a contrast.
    """

    def test_neither_arm_renders_a_knob_the_environment_did_not_set(self, second_invocation: Path):
        del second_invocation
        for argv in (plan.group_arm_stage().argv, plan.self_arm_stage().argv):
            for _suffix, flag, _value in OPTIONAL_KNOB_SETTINGS:
                assert flag not in argv, flag

    def test_both_arms_take_every_knob_with_the_value_that_was_set(
        self, second_invocation: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del second_invocation
        set_every_optional_knob(monkeypatch, plan.ENV_PREFIX)
        for argv in (plan.group_arm_stage().argv, plan.self_arm_stage().argv):
            for _suffix, flag, value in OPTIONAL_KNOB_SETTINGS:
                assert argv[argv.index(flag) + 1] == value, flag

    def test_a_misspelled_schedule_is_refused_before_either_arm_is_built(
        self, second_invocation: Path, monkeypatch: pytest.MonkeyPatch
    ):
        del second_invocation
        monkeypatch.setenv(f"{plan.ENV_PREFIX}LR_SCHEDULER", "constant_with_warumup")
        with pytest.raises(ValueError, match=f"{plan.ENV_PREFIX}LR_SCHEDULER"):
            plan.group_arm_stage()


class TestTheOfferedFlagsCacheKeysOnTheCliEnvironment:
    """`offered_flags` asks a CLI's --help once per module per session, unless a knob it may read moves.

    Every real probe is a fresh interpreter importing torch, so the cache in `games.tests.conftest`
    is what keeps the class above at one probe per module. It must not hide an env-dependent --help:
    `games.train` computes argparse defaults from `GAMES_*` knobs while building its parser, so a
    knob changing has to re-ask, while the plan-selection variables (which never reach a CLI and
    carry a per-test `tmp_path`) and variables outside the repo's namespace must not.
    """

    @pytest.fixture
    def isolated_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty cache in a shell with no `GAMES_*` variable set, so the operator's cannot leak in."""
        monkeypatch.setattr(games_conftest, "OFFERED_FLAGS_CACHE", {})
        for name in [name for name in os.environ if name.startswith(games_conftest.CLI_ENV_PREFIX)]:
            monkeypatch.delenv(name)

    @pytest.fixture
    def asked(self, isolated_cache: None, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Replace the real probe with one that records which module was asked."""
        del isolated_cache
        modules_asked: list[str] = []

        def fake_help(module: str) -> str:
            modules_asked.append(module)
            return f"usage: {module} [--alpha] [--beta-gamma VALUE]\n"

        monkeypatch.setattr(games_conftest, "ask_help", fake_help)
        return modules_asked

    def test_a_module_is_asked_once_and_its_flags_come_back_parsed(self, asked: list[str]) -> None:
        assert offered_flags("fake.cli") == {"--alpha", "--beta-gamma"}
        assert offered_flags("fake.cli") == {"--alpha", "--beta-gamma"}
        assert asked == ["fake.cli"]

    def test_each_module_is_asked_on_its_own(self, asked: list[str]) -> None:
        offered_flags("fake.cli")
        offered_flags("other.cli")
        offered_flags("fake.cli")
        assert asked == ["fake.cli", "other.cli"]

    def test_a_cli_knob_appearing_or_changing_value_asks_again(
        self, asked: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Presence and value both key the probe: the suite moves knobs between values, not only on and off."""
        offered_flags("fake.cli")
        monkeypatch.setenv(VLLM_IS_CORRECTION_ENV, "0")
        offered_flags("fake.cli")
        monkeypatch.setenv(VLLM_IS_CORRECTION_ENV, "1")
        offered_flags("fake.cli")
        monkeypatch.setenv("GAMES_S3_DEST", "s3://bucket/prefix/")
        offered_flags("fake.cli")
        assert asked == ["fake.cli"] * 4

    def test_a_knob_returning_to_a_seen_value_is_a_hit_again(
        self, asked: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every setting seen stays cached, so tests alternating a knob's value do not thrash the probe."""
        offered_flags("fake.cli")
        monkeypatch.setenv(VLLM_IS_CORRECTION_ENV, "0")
        offered_flags("fake.cli")
        monkeypatch.setenv(VLLM_IS_CORRECTION_ENV, "1")
        offered_flags("fake.cli")
        monkeypatch.setenv(VLLM_IS_CORRECTION_ENV, "0")
        offered_flags("fake.cli")
        monkeypatch.delenv(VLLM_IS_CORRECTION_ENV)
        offered_flags("fake.cli")
        assert asked == ["fake.cli"] * 3

    def test_plan_layer_variables_do_not_ask_again(
        self, asked: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The three plan namespaces and the estimator pin are the plan's to read, never a CLI's."""
        offered_flags("fake.cli")
        monkeypatch.setenv("GAMES_PAIR_SWEEP_DIR", str(tmp_path))
        monkeypatch.setenv("GAMES_ARM_SEQ_OUTPUT_DIR", str(tmp_path / "run"))
        monkeypatch.setenv("GAMES_9B_CORPUS", str(tmp_path / "corpus.jsonl"))
        monkeypatch.setenv(ESTIMATOR_PIN_ENV, FLAGSHIP_ESTIMATOR_PIN)
        offered_flags("fake.cli")
        assert asked == ["fake.cli"]

    def test_a_variable_outside_the_repo_namespace_does_not_ask_again(
        self, asked: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        offered_flags("fake.cli")
        monkeypatch.setenv("SOME_OTHER_TOOLS_KNOB", "1")
        offered_flags("fake.cli")
        assert asked == ["fake.cli"]

    def test_a_help_that_could_not_answer_raises_and_is_not_remembered(
        self, isolated_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure must surface, and the next ask must try the CLI again rather than trust it."""
        del isolated_cache
        attempts: list[str] = []

        def failing_then_working(module: str) -> str:
            attempts.append(module)
            if len(attempts) == 1:
                raise RuntimeError("simulated: --help exited 1")
            return "usage: fake.cli [--alpha]\n"

        monkeypatch.setattr(games_conftest, "ask_help", failing_then_working)
        with pytest.raises(RuntimeError, match="simulated"):
            offered_flags("fake.cli")
        assert offered_flags("fake.cli") == {"--alpha"}
        assert attempts == ["fake.cli", "fake.cli"]

    def test_the_real_probe_reports_a_cli_that_cannot_answer(self) -> None:
        """The unstubbed path: a module that does not exist exits non-zero, and the reason is quoted."""
        with pytest.raises(RuntimeError, match=r"No module named games\.no_such_cli"):
            games_conftest.ask_help("games.no_such_cli")
