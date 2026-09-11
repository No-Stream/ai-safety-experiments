"""Pin the shared plan skeleton against the modules it must agree with, not against itself.

Every assertion here compares two independent sources, because the three drifts this module was
extracted to end all looked locally reasonable in the file that had them:

*   the screen directory, against `games.screen_thinking`'s own default and against the provenance
    paths in `games.termination`;
*   the sampler flags a baseline sweep carries, against `games.select_prompts.training_sampler` and
    against `games.train.GameTrainConfig`'s own fields -- the invariant the whole contrast rests on,
    which the previous test pinned as three string literals against themselves;
*   one arm cap for every plan, since two existed and the shorter belonged to the plan running the
    larger model.

`TestThePlanSkeletonStaysCheapToImport` is the exception to that shape and the test with teeth: it
pins the import graph, not an agreement between two modules.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest
from transformers.trainer_utils import SchedulerType

from games import generation, plans
from games.screen_thinking import SCREEN_ROOT
from games.select_prompts import training_sampler
from games.termination import MEASURED_TERMINATION_STATS_BY_MODEL
from games.tests import conftest
from games.train import GameTrainConfig, path_safe_model_id
from grpo.estimator_defaults import (
    LIGER_UNFAITHFUL_LOSS_TYPES,
    VLLM_IMPORTANCE_SAMPLING_MODE,
    VLLM_IMPORTANCE_SAMPLING_MODES,
    assert_known_estimator,
)

MODEL = "Qwen/Qwen3.5-2B"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# `datasets` stands in for `games.dataset`, which imports it and nothing else here would notice.
TRAINING_STACK = ("torch", "transformers", "trl", "peft", "matplotlib", "pandas", "datasets")

# Spelled out rather than read from `plans.ESTIMATOR_PINS`; see TestTheEstimatorPin's docstring.
FLAGSHIP_ESTIMATOR_FLAGS = (
    "--loss-type",
    "dapo",
    "--scale-rewards",
    "batch",
    "--acknowledge-liger-estimator-mismatch",
    "--micro-batch-size",
    "1",
)
ESTIMATOR_FLAG_NAMES = (
    "--loss-type",
    "--scale-rewards",
    "--acknowledge-liger-estimator-mismatch",
    "--micro-batch-size",
)

ARM_SHAPE = plans.TrainingShape(
    max_steps="70",
    save_steps="5",
    num_generations="8",
    prompts_per_step="8",
    completion_tokens="32768",
)


def arm_argv(tmp_path: Path) -> tuple[str, ...]:
    """Render one training stage's argv the way all three plans render theirs."""
    return plans.build_arm_stage(
        arm="twin-pd-group",
        model_id=MODEL,
        corpus=tmp_path / "corpus.jsonl",
        output_dir=tmp_path / "run",
        timeout=plans.COLOCATE_ARM_TIMEOUT,
        shape=ARM_SHAPE,
        s3_dest="",
        log_path=tmp_path / "arm.log",
        log_run_dirs=(tmp_path / "run",),
    ).argv


def imported_training_stack(module: str) -> list[str]:
    """Report which of `TRAINING_STACK` importing `module` in a fresh interpreter pulled in."""
    probe = f"import sys, {module};print(sorted(name for name in {TRAINING_STACK!r} if name in sys.modules))"
    finished = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return [name for name in TRAINING_STACK if name in finished.stdout]


class TestThePlanSkeletonStaysCheapToImport:
    """`--print-plan` is advertised as the cheap check before the meter starts, so it must be cheap.

    `games.arms` moved out of `games.train` for exactly this reason, but the tax survived the move:
    this module still reached the trainer for the colocate readers and `games.select_prompts`
    for the sampler, so every plan module still paid 8.4 s to render a command that touches no
    GPU. Asserted
    on which modules got imported rather than on wall-clock seconds, because the timing is the
    consequence and the import graph is the cause: an 8 s import is not flaky, but a threshold on it
    would be.
    """

    def test_importing_the_plan_skeleton_pulls_in_none_of_the_training_stack(self) -> None:
        assert imported_training_stack("games.plans") == []

    def test_the_two_plans_that_can_be_cheap_are_cheap(self) -> None:
        """Whole plan modules, not just the skeleton: `--print-plan` imports one of these, not both.

        `games.arm_sequence` is deliberately absent and is the reason this is a list rather than a
        sweep over every plan: it validates `--backend` against
        `reward_hacking.backend_cli.LOCAL_KINDS`, and that module reaches `torch` through
        `reward_hacking.model_backend`. Its residual is one frozenset's import, measured at 3.7 s
        against 8.4 s before, and closing it means moving a backend-kind vocabulary that lives in
        another package. The contrast pair used to inherit that cost through a single
        `log_reward_spread` import and no longer does.
        """
        for module in ("games.contrast_pair_sequence", "games.nine_b_sequence"):
            assert imported_training_stack(module) == [], module

    def test_the_same_probe_would_notice_the_stack_if_it_were_there(self) -> None:
        """The negative control: the probe has to be able to fail, or it proves nothing."""
        assert imported_training_stack("games.train") != []


class TestTheScreenDirectoryIsWhereScreensActuallyGo:
    """Two plans wrote to a plural sibling directory that nothing else read."""

    def test_it_is_the_directory_the_screen_writer_defaults_to(self) -> None:
        assert plans.SCREEN_DIR == plans.REPO / SCREEN_ROOT

    def test_it_is_the_directory_the_measured_budgets_cite_as_provenance(self) -> None:
        """`games.termination`'s table names its screens by path; a fresh screen must land there."""
        for stats in MEASURED_TERMINATION_STATS_BY_MODEL.values():
            cited = plans.REPO / stats.artifact
            assert cited.parent == plans.SCREEN_DIR, stats.model_id

    def test_a_screen_artifact_is_named_for_its_budget_and_lands_there(self) -> None:
        artifact = plans.screen_artifact(model_tag="qwen35-9b", budget="32768")
        assert artifact.parent == plans.SCREEN_DIR
        assert artifact.name == "qwen35-9b-termination-32768.json"

    def test_a_per_game_screen_says_whose_prompts_it_screened(self) -> None:
        artifact = plans.screen_artifact(
            model_tag="qwen35-2b", budget="4096", game_id="iterated-pd"
        )
        assert artifact.name.startswith("iterated-pd-qwen35-2b")


class TestTheSweepSamplesThePolicyThatGetsTrained:
    """A sweep of a different policy selects the wrong prompts, and nothing downstream can tell."""

    def test_the_flags_come_from_the_training_sampler_rather_than_from_literals(self) -> None:
        sampler = training_sampler(MODEL)
        argv = plans.training_sampler_argv()
        assert argv[argv.index("--temperature") + 1] == str(sampler.temperature)
        assert argv[argv.index("--top-p") + 1] == str(sampler.top_p)
        assert argv[argv.index("--top-k") + 1] == str(sampler.top_k)

    def test_the_trainer_samples_at_that_same_sampler(self) -> None:
        """The three sources of "at training temperature" have to agree or the contrast is broken.

        They are one source now -- `games.generation` -- where the trainer, the sweep's sampler and
        the plan's rendered flags each carried their own copy of GRPOConfig's `1.0 / 1.0 / 0`
        before.
        So this reads through the constants rather than across them, and what it still catches is a
        literal re-hardcoded into `GameTrainConfig` or into `training_sampler`, which is how the
        copies got there the first time.
        """
        sampler = training_sampler(MODEL)
        trainer_defaults = {field.name: field.default for field in fields(GameTrainConfig)}
        assert trainer_defaults["temperature"] == sampler.temperature
        assert trainer_defaults["top_p"] == sampler.top_p
        assert trainer_defaults["top_k"] == sampler.top_k


class TestVllmIsTheOnlyTrainingBackend:
    """The env var used to select a backend; since 2026-08-26 it can only confirm the one left.

    The incident this encodes: a stray `GAMES_VLLM_COLOCATE=0` in a trainer's environment put a
    paid arm on transformers.generate at ~72 minutes per step against the ~13.5 costed, and only
    a human pace diagnosis $28 in caught it. So "0" is not a preference to honour but the exact
    bug signature, refused at startup wherever it appears.
    """

    def test_an_unset_environment_is_colocate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(generation.VLLM_COLOCATE_ENV, raising=False)
        assert plans.generation_backend() == plans.COLOCATE_GENERATION

    def test_an_explicit_confirmation_is_colocate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(generation.VLLM_COLOCATE_ENV, "1")
        assert plans.generation_backend() == plans.COLOCATE_GENERATION

    def test_the_incident_export_is_refused_naming_the_decision_and_the_stakes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal must carry the 2026-08-26 decision and the 72-vs-13.5 figures."""
        monkeypatch.setenv(generation.VLLM_COLOCATE_ENV, "0")
        with pytest.raises(RuntimeError, match="2026-08-26"):
            generation.assert_vllm_rollouts()
        with pytest.raises(RuntimeError, match="~72 minutes per step"):
            generation.assert_vllm_rollouts()
        with pytest.raises(RuntimeError, match=r"~13\.5"):
            generation.assert_vllm_rollouts()

    @pytest.mark.parametrize("value", ["true", "yes", "", "hf", "2"])
    def test_any_value_but_a_bare_1_is_refused_not_read_as_a_preference(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(generation.VLLM_COLOCATE_ENV, value)
        with pytest.raises(RuntimeError, match=generation.VLLM_COLOCATE_ENV):
            generation.assert_vllm_rollouts()

    def test_a_plan_render_refuses_on_a_poisoned_box(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`--print-plan` is the cheap check before the meter starts, so it fires the refusal."""
        monkeypatch.setenv(generation.VLLM_COLOCATE_ENV, "0")
        with pytest.raises(RuntimeError, match="vLLM"):
            plans.colocate_argv()
        with pytest.raises(RuntimeError, match="vLLM"):
            plans.default_arm_timeout()

    def test_a_rendered_stage_carries_engine_knobs_and_no_backend_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """There is nothing left for a backend flag to say; the knobs still must be printed."""
        monkeypatch.delenv(generation.VLLM_COLOCATE_ENV, raising=False)
        argv = plans.colocate_argv()
        assert "--vllm-colocate" not in argv
        assert "--no-vllm-colocate" not in argv
        assert "--vllm-gpu-memory-utilization" in argv


class TestOneCapTablePerAxis:
    """The sweep's cap follows `--backend`; the arm's cap is the colocate cap, the only one left."""

    def test_the_arm_cap_is_the_colocate_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(generation.VLLM_COLOCATE_ENV, raising=False)
        assert plans.default_arm_timeout() == plans.COLOCATE_ARM_TIMEOUT

    def test_an_unmeasured_sweep_backend_gets_the_slowest_cap(self) -> None:
        assert plans.sweep_timeout_for("some-new-server") == plans.SWEEP_TIMEOUT_BY_BACKEND["hf"]


class TestTheArtifactSlug:
    def test_it_is_short_lowercased_and_dot_free(self) -> None:
        assert plans.artifact_model_slug("Qwen/Qwen3.5-9B") == "qwen35-9b"

    def test_it_is_not_the_trainer_s_path_safe_id(self) -> None:
        """Both name the same run's artifacts, so the two must not be reachable under one name."""
        assert plans.artifact_model_slug("Qwen/Qwen3.5-9B") != path_safe_model_id("Qwen/Qwen3.5-9B")


class TestPlanSettings:
    def test_a_setting_comes_from_the_environment_when_it_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GAMES_TEST_PLAN_SETTING", "from-the-shell")
        assert plans.plan_setting("GAMES_TEST_PLAN_SETTING", "fallback") == "from-the-shell"

    def test_the_default_is_used_when_it_is_not(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GAMES_TEST_PLAN_SETTING", raising=False)
        assert plans.plan_setting("GAMES_TEST_PLAN_SETTING", "fallback") == "fallback"


class TestTheEstimatorPin:
    """One variable names which already-trained arms a run is being made comparable to.

    Two properties matter and they pull in opposite directions, so both are asserted directly. An
    unpinned render has to be the argv it was before the pin existed, because every plan on disk is
    unpinned and a changed default would silently re-price runs already costed. And a pinned render
    has to carry all four flags with their values, because the wave-3 comparison is of gradient
    *magnitudes* and the repo defaults produce roughly 5-10x less gradient per step (the
    effective-LR note in `grpo/estimator_defaults.py`).

    The four flags are spelled out as literals in this file instead of read back from
    `plans.ESTIMATOR_PINS`: reading the tuple under test would assert it against itself, and what a
    launch gate greps a printed plan for is these exact tokens.
    """

    def test_an_unpinned_render_is_the_argv_it_was_before_the_pin_existed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Constructed rather than snapshotted, so it says what the argv is, not that it changed."""
        monkeypatch.delenv(plans.ESTIMATOR_PIN_ENV, raising=False)
        assert arm_argv(tmp_path) == (
            "timeout",
            plans.COLOCATE_ARM_TIMEOUT,
            *plans.UV,
            "-m",
            "games.train",
            "--arm",
            "twin-pd-group",
            "--model",
            MODEL,
            "--corpus",
            str(tmp_path / "corpus.jsonl"),
            "--output-dir",
            str(tmp_path / "run"),
            "--resume-from-checkpoint",
            "latest",
            "--max-steps",
            ARM_SHAPE.max_steps,
            "--save-steps",
            ARM_SHAPE.save_steps,
            "--num-generations",
            ARM_SHAPE.num_generations,
            "--prompts-per-step",
            ARM_SHAPE.prompts_per_step,
            "--max-completion-tokens",
            ARM_SHAPE.completion_tokens,
            *plans.colocate_argv(),
        )

    @pytest.mark.parametrize("flag", ESTIMATOR_FLAG_NAMES)
    def test_an_unpinned_render_names_no_estimator_flag_at_all(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str
    ) -> None:
        monkeypatch.delenv(plans.ESTIMATOR_PIN_ENV, raising=False)
        assert flag not in arm_argv(tmp_path)

    def test_an_empty_pin_reads_as_no_pin_rather_than_as_an_unknown_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`export GAMES_ESTIMATOR_PIN=` is how a shell drops a pin it had exported."""
        monkeypatch.setenv(plans.ESTIMATOR_PIN_ENV, "")
        assert plans.estimator_argv() == ()

    def test_the_flagship_pin_renders_the_four_flags_in_order_and_adjacent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(plans.ESTIMATOR_PIN_ENV, plans.FLAGSHIP_ESTIMATOR_PIN)
        argv = arm_argv(tmp_path)
        start = argv.index("--loss-type")
        assert argv[start : start + len(FLAGSHIP_ESTIMATOR_FLAGS)] == FLAGSHIP_ESTIMATOR_FLAGS

    def test_the_pin_appends_and_changes_nothing_else(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The pin is the only difference between a comparable arm and its unpinned twin."""
        monkeypatch.delenv(plans.ESTIMATOR_PIN_ENV, raising=False)
        unpinned = arm_argv(tmp_path)
        monkeypatch.setenv(plans.ESTIMATOR_PIN_ENV, plans.FLAGSHIP_ESTIMATOR_PIN)
        assert arm_argv(tmp_path) == unpinned + FLAGSHIP_ESTIMATOR_FLAGS

    @pytest.mark.parametrize(
        "value", ["dapo", "flagship", "flagship-dapo-batch", "flagship_dapo_batch_mb1", "1"]
    )
    def test_an_unknown_pin_is_refused_naming_the_variable_and_what_it_accepts(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(plans.ESTIMATOR_PIN_ENV, value)
        with pytest.raises(ValueError, match=plans.ESTIMATOR_PIN_ENV):
            plans.estimator_argv()
        with pytest.raises(ValueError, match=plans.FLAGSHIP_ESTIMATOR_PIN):
            plans.estimator_argv()

    def test_a_plan_render_refuses_a_typo_before_the_meter_starts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """`--print-plan` is the cheap CPU-only check, so a misspelled pin has to fail there."""
        monkeypatch.setenv(plans.ESTIMATOR_PIN_ENV, "flagship-dapo-batch-mb-1")
        with pytest.raises(ValueError, match=plans.ESTIMATOR_PIN_ENV):
            arm_argv(tmp_path)

    def test_the_pinned_values_are_ones_trl_accepts(self) -> None:
        """The pin is a second copy of estimator vocabulary, so it is read against the first."""
        pinned = plans.ESTIMATOR_PINS[plans.FLAGSHIP_ESTIMATOR_PIN]
        assert_known_estimator(
            pinned[pinned.index("--loss-type") + 1], pinned[pinned.index("--scale-rewards") + 1]
        )

    def test_the_acknowledgement_it_carries_is_not_the_no_op_the_trainer_refuses(self) -> None:
        """`games.train` raises on an acknowledgement where nothing needs acknowledging."""
        pinned = plans.ESTIMATOR_PINS[plans.FLAGSHIP_ESTIMATOR_PIN]
        assert "--acknowledge-liger-estimator-mismatch" in pinned
        assert pinned[pinned.index("--loss-type") + 1] in LIGER_UNFAITHFUL_LOSS_TYPES


class TestTheSchedulerNamesArePinnedToTransformersOwn:
    """`plans.LR_SCHEDULER_NAMES` is a copy of transformers' enum, and this is why that is safe.

    The plan layer refuses a misspelled `--lr-scheduler` before the meter starts, which needs the
    list of accepted names, while `--print-plan` may not import transformers at all (the import
    probe above is the gate on that). So the names are copied and read back against the enum here,
    where importing transformers costs nothing: a version bump that adds or renames a schedule can
    then only widen this suite's list, never let the plan layer refuse something TRL would accept.
    """

    def test_the_copy_is_exactly_what_transformers_offers(self) -> None:
        assert {member.value for member in SchedulerType} == plans.LR_SCHEDULER_NAMES

    def test_the_trainers_own_default_schedule_is_one_of_them(self) -> None:
        """Otherwise a plan could be refused for rendering the value it would have trained under."""
        trainer_defaults = {field.name: field.default for field in fields(GameTrainConfig)}
        assert trainer_defaults["lr_scheduler"] in plans.LR_SCHEDULER_NAMES

    def test_every_optional_knob_flag_is_spelled_as_its_variable(self) -> None:
        """The kit exports `<PREFIX>ADAM_EPSILON` and the trainer takes `--adam-epsilon`, so the two
        halves of every row are checked against each other rather than read from one place."""
        for suffix, flag, _assert_valid in plans.OPTIONAL_TRAINING_KNOBS:
            assert flag == f"--{suffix.lower().replace('_', '-')}"

    def test_every_optional_knob_sets_a_field_of_the_trainers_own_config(self) -> None:
        """The spelling rule above compares a row with itself; this compares it with the trainer.

        A knob whose flag `games.train` renamed away would render an argv the CLI rejects on the box,
        and because every one of these flags is a `GameTrainConfig` field it is also what puts the
        knob in `run_config.json` (`run_config_payload` records `asdict(config)` verbatim).
        """
        trainer_fields = {field.name for field in fields(GameTrainConfig)}
        for _suffix, flag, _assert_valid in plans.OPTIONAL_TRAINING_KNOBS:
            assert flag.removeprefix("--").replace("-", "_") in trainer_fields, flag


class TestTheImportanceSamplingModeKnob:
    """`<PREFIX>VLLM_IMPORTANCE_SAMPLING_MODE` renders `--vllm-importance-sampling-mode <name>`.

    The mode decides what TRL's correction does with the vLLM-versus-trainer log-probability
    difference, and the owner precommitted (2026-09-04) that arm 1 and its control run the
    token-level truncated mode if the 9B probe confirms a large mismatch -- so a kit sets it, and a
    misspelling has to be refused at `--print-plan` rather than on the rented box.

    The variable is spelled out rather than abbreviated to `IS_MODE` because
    `test_every_optional_knob_flag_is_spelled_as_its_variable` above holds every knob row's variable
    to its own flag's spelling; the switch table, whose rows render one token, has no such rule and
    carries `IS_LOG_ONLY`.
    """

    def test_the_names_the_plan_layer_accepts_are_the_ones_the_trainer_accepts(self) -> None:
        """One tuple, imported by both, so a plan cannot refuse a mode the trainer would run."""
        assert plans.VLLM_IMPORTANCE_SAMPLING_MODES == VLLM_IMPORTANCE_SAMPLING_MODES
        trainer_defaults = {field.name: field.default for field in fields(GameTrainConfig)}
        assert trainer_defaults["vllm_importance_sampling_mode"] in VLLM_IMPORTANCE_SAMPLING_MODES

    @pytest.mark.parametrize("mode", VLLM_IMPORTANCE_SAMPLING_MODES)
    def test_every_mode_renders_as_the_flags_value(
        self, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        monkeypatch.delenv(generation.VLLM_IS_CORRECTION_ENV, raising=False)
        monkeypatch.setenv("GAMES_TEST_VLLM_IMPORTANCE_SAMPLING_MODE", mode)
        assert plans.optional_training_knobs("GAMES_TEST_") == (
            ("--vllm-importance-sampling-mode", mode),
        )

    def test_a_misspelled_mode_is_refused_naming_its_own_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(generation.VLLM_IS_CORRECTION_ENV, raising=False)
        monkeypatch.setenv("GAMES_TEST_VLLM_IMPORTANCE_SAMPLING_MODE", "token-truncate")
        with pytest.raises(ValueError, match="GAMES_TEST_VLLM_IMPORTANCE_SAMPLING_MODE"):
            plans.optional_training_knobs("GAMES_TEST_")

    def test_the_mode_beside_the_correction_off_is_refused_at_plan_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`games.train` refuses the pair too, but only once the box has bootstrapped and pulled the
        model. The wave-4b training role is exactly the launch that exports the correction off, so
        this is the same coupling the switch table carries, on the row that needs it."""
        monkeypatch.setenv(generation.VLLM_IS_CORRECTION_ENV, "0")
        monkeypatch.setenv("GAMES_TEST_VLLM_IMPORTANCE_SAMPLING_MODE", "token_truncate")
        with pytest.raises(ValueError, match=generation.VLLM_IS_CORRECTION_ENV) as refusal:
            plans.optional_training_knobs("GAMES_TEST_")
        assert "GAMES_TEST_VLLM_IMPORTANCE_SAMPLING_MODE" in str(refusal.value)

    def test_trls_own_default_renders_beside_the_correction_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal is about asking for a mode nothing will read, so the value that changes
        nothing is not refused: a kit may carry it through a correction-off smoke unchanged."""
        monkeypatch.setenv(generation.VLLM_IS_CORRECTION_ENV, "0")
        monkeypatch.setenv(
            "GAMES_TEST_VLLM_IMPORTANCE_SAMPLING_MODE", VLLM_IMPORTANCE_SAMPLING_MODE
        )
        assert plans.optional_training_knobs("GAMES_TEST_") == (
            ("--vllm-importance-sampling-mode", VLLM_IMPORTANCE_SAMPLING_MODE),
        )

    def test_an_unset_variable_renders_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GAMES_TEST_VLLM_IMPORTANCE_SAMPLING_MODE", raising=False)
        assert plans.optional_training_knobs("GAMES_TEST_") == ()


class TestTheDynamicSamplingOversampleKnob:
    """`<PREFIX>DYNAMIC_SAMPLING_OVERSAMPLE` renders `--dynamic-sampling-oversample <count>`.

    A count of prompt groups to generate per optimizer step, so the plan layer refuses anything that
    is not a whole number above zero: `argparse`'s `type=int` refuses "2.0" and "two", and
    `games.train` refuses zero and below, but all three of those refusals land on the rented box.
    """

    ENV_NAME = "GAMES_TEST_DYNAMIC_SAMPLING_OVERSAMPLE"

    def test_a_count_renders_as_the_flags_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(self.ENV_NAME, "3")
        assert plans.optional_training_knobs("GAMES_TEST_") == (
            ("--dynamic-sampling-oversample", "3"),
        )

    @pytest.mark.parametrize("value", ["0", "-1", "2.0", "two", "1e2", " 2"])
    def test_anything_but_a_whole_number_above_zero_is_refused_naming_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(self.ENV_NAME, value)
        with pytest.raises(ValueError, match=self.ENV_NAME):
            plans.optional_training_knobs("GAMES_TEST_")

    def test_off_is_the_variable_unset_rather_than_a_rendered_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rendering `1` would be harmless but it is not what "off" means here: an unset knob leaves
        `games.train`'s own argparse default in force, which is what every kit written before this
        knob existed relies on."""
        monkeypatch.delenv(self.ENV_NAME, raising=False)
        assert plans.optional_training_knobs("GAMES_TEST_") == ()

    def test_it_is_not_coupled_to_the_importance_sampling_correction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The correction shapes a gradient weight and this decides which prompts are in the batch,
        so a correction-off arm may oversample: a refusal keyed on the table rather than on the row
        would refuse exactly the wave-4b training role that wants both."""
        monkeypatch.setenv(generation.VLLM_IS_CORRECTION_ENV, "0")
        monkeypatch.setenv(self.ENV_NAME, "2")
        assert plans.optional_training_knobs("GAMES_TEST_") == (
            ("--dynamic-sampling-oversample", "2"),
        )


class TestTheStagePlanTestsCoverEveryOptionalKnobTheseTablesCarry:
    """The three stage-plan files drive their knob tests off `games.tests.conftest`'s tables.

    Those tables used to be hand copies of the two here, with nothing pinning them together, so a knob
    added at the plan layer was silently outside every "none rendered when unset" and "each one
    reaches the CLI" test in all three files -- and outside `clear_every_optional_knob`, which meant an
    operator exporting the new variable changed what those files measured. The conftest now derives the
    suffix and flag of every row from these tables and only carries a test value of its own, so drift
    is a `RuntimeError` at collection rather than silence. This pins that derivation: a re-hand-copied
    table fails here.
    """

    def test_the_shared_knob_table_is_the_plan_layers_own(self) -> None:
        assert [(suffix, flag) for suffix, flag, _value in conftest.OPTIONAL_KNOB_SETTINGS] == [
            (suffix, flag) for suffix, flag, _assert_valid in plans.OPTIONAL_TRAINING_KNOBS
        ]

    def test_every_shared_knob_carries_a_value_the_plan_layer_accepts(self) -> None:
        """The test value has to pass the row's own check, or the knob's tests refuse before rendering."""
        for suffix, _flag, value in conftest.OPTIONAL_KNOB_SETTINGS:
            assert_valid = next(
                check
                for row_suffix, _row_flag, check in plans.OPTIONAL_TRAINING_KNOBS
                if row_suffix == suffix
            )
            assert_valid(f"GAMES_TEST_{suffix}", value)

    def test_the_shared_switch_table_is_the_plan_layers_own(self) -> None:
        assert list(conftest.OPTIONAL_SWITCH_SETTINGS) == [
            (suffix, flag) for suffix, flag, _needs_correction in plans.OPTIONAL_TRAINING_SWITCHES
        ]


class TestTheOptionalTrainingSwitches:
    """The valueless flags a plan renders only where its environment sets the variable to exactly "1".

    Three rows today: `--vllm-importance-sampling-log-only` and `--cast-lm-head-to-fp32` are the
    instrument pair that makes the vLLM-versus-trainer mismatch measurable without changing what a run
    trains, and `--allow-short-completions` is the smoke and timing-probe escape hatch from the
    measured completion floor. All three are `store_true` on the trainer: one argv token, no value. So
    they cannot ride in `OPTIONAL_TRAINING_KNOBS`, whose rows render `flag value`, and they get a table
    of their own with the same contract -- unset renders nothing, so every plan and kit written before
    a switch existed keeps the argv it always had.

    The value is pinned to a bare "1" rather than read as a boolean, following `GAMES_VLLM_COLOCATE`:
    "true", "yes" and "0" are each refused at plan time, because a kit that exports `=0` to mean off
    and gets on, or `=true` and gets a refusal on the box, is the drift these tables exist to stop.

    The correction coupling is per row rather than per table: the instrument pair is refused without
    the vLLM importance-sampling correction because the trainer refuses it, while the escape hatch has
    nothing to do with the correction and the plumbing smoke that wants it runs the correction OFF, the
    way the training role does.
    """

    def test_an_unset_environment_renders_no_switch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for suffix, _flag, _needs_correction in plans.OPTIONAL_TRAINING_SWITCHES:
            monkeypatch.delenv(f"GAMES_TEST_{suffix}", raising=False)
        assert plans.optional_training_switches("GAMES_TEST_") == ()

    def test_a_bare_1_renders_the_flag_and_nothing_after_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv(generation.VLLM_IS_CORRECTION_ENV, raising=False)
        for suffix, _flag, _needs_correction in plans.OPTIONAL_TRAINING_SWITCHES:
            monkeypatch.setenv(f"GAMES_TEST_{suffix}", "1")
        switches = plans.optional_training_switches("GAMES_TEST_")
        assert switches == tuple(
            flag for _suffix, flag, _needs_correction in plans.OPTIONAL_TRAINING_SWITCHES
        )
        shape = plans.TrainingShape(
            max_steps="70",
            save_steps="5",
            num_generations="8",
            prompts_per_step="8",
            completion_tokens="32768",
            optional_switches=switches,
        )
        argv = plans.build_arm_stage(
            arm="twin-pd-group",
            model_id=MODEL,
            corpus=tmp_path / "corpus.jsonl",
            output_dir=tmp_path / "run",
            timeout=plans.COLOCATE_ARM_TIMEOUT,
            shape=shape,
            s3_dest="",
            log_path=tmp_path / "arm.log",
            log_run_dirs=(tmp_path / "run",),
        ).argv
        for flag in switches:
            assert argv.count(flag) == 1, flag
            following = argv[argv.index(flag) + 1]
            assert following.startswith("--"), f"{flag} rendered a value token {following!r}"

    @pytest.mark.parametrize(
        "suffix", [suffix for suffix, _flag, _needs_correction in plans.OPTIONAL_TRAINING_SWITCHES]
    )
    @pytest.mark.parametrize("value", ["0", "true", "yes", "on", "2"])
    def test_any_value_but_a_bare_1_is_refused_naming_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, suffix: str, value: str
    ) -> None:
        monkeypatch.setenv(f"GAMES_TEST_{suffix}", value)
        with pytest.raises(ValueError, match=f"GAMES_TEST_{suffix}"):
            plans.optional_training_switches("GAMES_TEST_")

    def test_an_instrument_switch_beside_the_correction_off_is_refused_at_plan_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The trainer refuses the instrument pair without the correction, but only after the
        bootstrap, the model pull and the corpus restore; the same refusal costs nothing here."""
        monkeypatch.setenv(generation.VLLM_IS_CORRECTION_ENV, "0")
        monkeypatch.setenv("GAMES_TEST_CAST_LM_HEAD_FP32", "1")
        with pytest.raises(ValueError, match=generation.VLLM_IS_CORRECTION_ENV) as refusal:
            plans.optional_training_switches("GAMES_TEST_")
        assert "GAMES_TEST_CAST_LM_HEAD_FP32" in str(refusal.value)

    def test_the_short_completions_switch_renders_beside_the_correction_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The escape hatch is not part of the instrument pair, and the run that wants it is the
        plumbing smoke of a training role, which exports the correction OFF. A refusal keyed on "any
        switch is set" rather than on the row would refuse exactly the launch this switch exists for.
        """
        monkeypatch.setenv(generation.VLLM_IS_CORRECTION_ENV, "0")
        monkeypatch.setenv("GAMES_TEST_ALLOW_SHORT_COMPLETIONS", "1")
        assert plans.optional_training_switches("GAMES_TEST_") == ("--allow-short-completions",)

    def test_each_switch_is_a_boolean_field_of_the_trainer_config_defaulting_off(self) -> None:
        """Two sources: the flag spelled here and the dataclass field it sets, checked against each
        other, so a renamed trainer flag cannot leave the table rendering one the CLI no longer has.

        Being a field is also what records the switch in `run_config.json`, whose `config` block is
        `asdict(config)` verbatim (`games.train.run_config_payload`): a smoke that rendered
        `--allow-short-completions` is visibly a smoke in its own launch record, with nothing extra to
        wire up here.
        """
        trainer_fields = {field.name: field for field in fields(GameTrainConfig)}
        for _suffix, flag, _needs_correction in plans.OPTIONAL_TRAINING_SWITCHES:
            field = trainer_fields[flag.removeprefix("--").replace("-", "_")]
            assert field.type in ("bool", bool), flag
            assert field.default is False, flag

    def test_the_correction_column_of_every_row_is_what_the_trainer_itself_refuses(self) -> None:
        """The column names which switches need the correction ON; `games.train` says the same thing
        in its own `if`s rather than in a table, so the two are checked against each other.

        A row marked coupled that the trainer accepts refuses launches nobody else would; a row left
        uncoupled that the trainer refuses reaches the box and dies there, an hour of bootstrap in.
        """
        for _suffix, flag, needs_correction in plans.OPTIONAL_TRAINING_SWITCHES:
            field_name = flag.removeprefix("--").replace("-", "_")
            settings: dict[str, object] = {
                "arm": "twin-pd-group",
                "generate_fresh": True,
                "vllm_importance_sampling_correction": False,
                field_name: True,
            }
            if needs_correction:
                with pytest.raises(ValueError, match=flag):
                    GameTrainConfig(**settings)  # pyright: ignore[reportArgumentType]
            else:
                config = GameTrainConfig(**settings)  # pyright: ignore[reportArgumentType]
                assert getattr(config, field_name) is True, flag
